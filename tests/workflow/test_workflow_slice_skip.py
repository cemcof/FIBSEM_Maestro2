# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Skipping the rest of a slice at the user's request: what runs, what is carried over, and when a pause comes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.logging.text.file import close_all_log_files
from fibsem_maestro.milling.milling import Milling
from fibsem_maestro.notifications.null_notifier import NullNotifier
from fibsem_maestro.ui.views.log_panel.parsing import LEVELS, level_value
from fibsem_maestro.ui.views.log_panel.reader import LogReader
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.action.outcome import CarriedOver, StepOutcome
    from fibsem_maestro.workflow.error import ActionError
    from fibsem_maestro.workflow.error_choice import ErrorChoice


class UserAtTheGui:
    """
    `WorkflowCallbacks` of a user who asks for a skip (and a pause) once a given action finishes.

    Args:
        workflow: The workflow it is the callbacks of.
        after: The action after which the requests are made.
        in_slice: The slice in which they are made.
        pause: Also request a pause.
    """

    def __init__(
        self, workflow: Workflow, after: str, in_slice: int, pause: bool = False
    ) -> None:
        self._workflow = workflow
        self._after = after
        self._in_slice = in_slice
        self._pause = pause
        self.skip_requested = False
        self.pause_requested = False
        self.skipped: list[list[str]] = []
        self.paused_at: list[tuple[int, list[str]]] = []
        """The workflow's slice and the actions pending there, at every pause."""

    def notify_action_finished(self, finished_action_index: int) -> None:
        action = self._workflow.actions[finished_action_index]
        # the action has advanced to the next slice
        if action.name == self._after and action.ctx.slice == self._in_slice + 1:
            self.skip_requested = True
            self.pause_requested = self._pause

    def notify_slice_finished(self) -> None:
        pass

    def notify_is_paused(self) -> None:
        self.paused_at.append(
            (
                self._workflow.ctx.slice,
                [a.name for a in self._workflow.pending_actions()],
            )
        )

    def is_pause_requested(self) -> bool:
        return self.pause_requested

    def wait_for_resume(self) -> None:
        self.pause_requested = False

    def request_error_choice(self, error: ActionError) -> ErrorChoice:
        raise AssertionError(f"no action should fail: {error}")

    def is_slice_skip_requested(self) -> bool:
        return self.skip_requested

    def notify_slice_skipped(self, skipped: Sequence[Action]) -> None:
        self.skip_requested = False
        self.skipped.append([action.name for action in skipped])


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> Callable[[Action], list[int]]:
    """Record the slices for which an action executes its step (not when skipped)."""

    def record(action: Action) -> list[int]:
        slices: list[int] = []
        run_step = action._run_step

        def recorded() -> StepOutcome:
            slices.append(action.ctx.slice)
            return run_step()

        monkeypatch.setattr(action, "_run_step", recorded)
        return slices

    return record


@pytest.fixture
def three(add_action: Callable[..., Action]) -> tuple[Action, Action, Action]:
    """Three prepared actions, "a", "b" and "c"."""
    a, b, c = (add_action("adjust_props", name) for name in ("a", "b", "c"))
    for action in (a, b, c):
        action.prepare()
    return a, b, c


def test_slice_skip_carries_the_rest_over_and_runs_the_next_slice_fully(
    workflow: Workflow,
    three: tuple[Action, Action, Action],
    executed: Callable[[Action], list[int]],
) -> None:
    a, b, c = three
    runs = {action.name: executed(action) for action in three}
    user = UserAtTheGui(workflow, after="a", in_slice=1)
    workflow.set_callbacks(user)

    workflow.run(last_slice=2)

    assert runs == {"a": [1, 2], "b": [2], "c": [2]}
    assert user.skipped == [["b", "c"]]
    assert workflow.ctx.slice == 3
    # carried over unchanged into slice 2
    for action in (b, c):
        store = action.ctx.props_store
        assert action.read_properties(store.at(2)) == action.read_properties(
            store.at(1)
        )
    _ = a


def test_slice_skip_requested_during_the_last_action_leaves_the_next_slice_alone(
    workflow: Workflow,
    three: tuple[Action, Action, Action],
    executed: Callable[[Action], list[int]],
) -> None:
    # a request that found nothing left must not skip the next slice
    runs = {action.name: executed(action) for action in three}
    user = UserAtTheGui(workflow, after="c", in_slice=1)
    workflow.set_callbacks(user)

    workflow.run(last_slice=2)

    assert runs == {"a": [1, 2], "b": [1, 2], "c": [1, 2]}
    assert user.skipped == [[]]


def test_slice_skip_never_empties_a_slice(
    workflow: Workflow,
    three: tuple[Action, Action, Action],
    executed: Callable[[Action], list[int]],
) -> None:
    # requested before the run started: the first action still runs
    runs = {action.name: executed(action) for action in three}
    user = UserAtTheGui(workflow, after="-", in_slice=1)
    user.skip_requested = True
    workflow.set_callbacks(user)

    workflow.run(last_slice=1)

    assert runs == {"a": [1], "b": [], "c": []}
    assert user.skipped == [["b", "c"]]


@pytest.mark.parametrize("while_executing", [False, True])
def test_slice_skip_goes_before_a_pause_which_waits_for_the_next_slice(
    workflow: Workflow,
    three: tuple[Action, Action, Action],
    executed: Callable[[Action], list[int]],
    monkeypatch: pytest.MonkeyPatch,
    while_executing: bool,
) -> None:
    a, _, _ = three
    runs = {action.name: executed(action) for action in three}
    if while_executing:
        # requested while 'a' executes, before the workflow checks for a pause
        user = UserAtTheGui(workflow, after="-", in_slice=1)
        run_step = a._run_step

        def requesting() -> StepOutcome:
            if a.ctx.slice == 1:
                user.skip_requested = user.pause_requested = True
            return run_step()

        monkeypatch.setattr(a, "_run_step", requesting)
    else:
        user = UserAtTheGui(workflow, after="a", in_slice=1, pause=True)
    workflow.set_callbacks(user)

    workflow.run(last_slice=2)

    assert user.skipped == [["b", "c"]]
    # paused once, before anything of slice 2 ran
    assert user.paused_at == [(2, ["a", "b", "c"])]
    assert runs == {"a": [1, 2], "b": [2], "c": [2]}


def test_slice_skip_keeps_the_milling_where_it_was(
    workflow: Workflow,
    add_action: Callable[..., Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a skipped milling mills nothing; the next slice mills the strip it would have
    first = add_action("adjust_props", "first")
    first.prepare()
    milling = add_action(
        "milling",
        "milling",
        milling_area=[
            RelativeArea(origin=RelativePoint(0.1, 0.1), width=0.5, height=0.2)
        ],
        pattern_type="rectangle",
        slice_distance=100.0,
    )
    milling.prepare()
    assert isinstance(milling, Milling)
    milled: list[object] = []
    monkeypatch.setattr(
        milling.microscope.ion_beam,
        "rectangle_milling",
        lambda area, *_: milled.append(area),
    )
    workflow.set_callbacks(UserAtTheGui(workflow, after="first", in_slice=1))

    workflow.run(last_slice=2)

    assert milling.state.slices_milled == 1
    assert len(milled) == 1


def test_slice_skip_resumes_at_the_next_slice_after_a_restart(
    workflow: Workflow, three: tuple[Action, Action, Action], tmp_path: Path
) -> None:
    workflow.set_callbacks(UserAtTheGui(workflow, after="a", in_slice=1))
    workflow.run(last_slice=1)

    reloaded = Workflow.import_from_dir_with_state(tmp_path, NullNotifier())

    assert reloaded.ctx.slice == 2
    assert [action.ctx.slice for action in reloaded.actions] == [2, 2, 2]
    assert not reloaded.slice_started()
    _ = three


def test_slice_skip_uses_the_actions_own_skip(
    workflow: Workflow,
    three: tuple[Action, Action, Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e.g. a drift correction's skip also tells its drift calculation
    _, b, _ = three
    reasons: list[str] = []
    skip = b.skip

    def recorded(reason: str) -> CarriedOver:
        reasons.append(reason)
        return skip(reason)

    monkeypatch.setattr(b, "skip", recorded)
    workflow.set_callbacks(UserAtTheGui(workflow, after="a", in_slice=1))

    workflow.run(last_slice=1)

    assert len(reasons) == 1
    assert "user" in reasons[0].lower()


def test_slice_skip_logs_the_skipped_actions_and_their_warnings(
    workflow: Workflow,
    three: tuple[Action, Action, Action],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, c = three
    monkeypatch.setattr(c, "skip_warning", lambda _skipped: "mind the gap")
    workflow.set_callbacks(UserAtTheGui(workflow, after="a", in_slice=1))

    workflow.run(last_slice=1)

    close_all_log_files()
    lines = LogReader(tmp_path).read(1, ["workflow"], level_value(LEVELS[0]))
    warnings = [line.message for line in lines if line.level == "WARNING"]
    assert any(all(part in m for part in ("1", "'b'", "'c'")) for m in warnings)
    assert any("'c'" in m and "mind the gap" in m for m in warnings)

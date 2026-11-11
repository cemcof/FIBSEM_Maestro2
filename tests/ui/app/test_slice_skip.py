# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.app.slice_skip import plan_slice_skip

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.workflow.workflow import Workflow

SLICE = 4


@pytest.fixture
def three(add_action: Callable[..., Action], workflow: Workflow) -> list[Action]:
    """Three actions in slice `SLICE`, none of which has run yet."""
    actions = [add_action("adjust_props", name) for name in ("a", "b", "c")]
    workflow.ctx.set_slice(SLICE)
    for action in actions:
        action.ctx.set_slice(SLICE)
    return actions


def ran(*actions: Action) -> None:
    """Make the actions done for the current slice."""
    for action in actions:
        action.ctx.set_slice(SLICE + 1)


@pytest.mark.parametrize("state", [AppState.PAUSED, AppState.RELOADED])
def test_slice_skip_plan_stopped_skips_everything_left(
    workflow: Workflow, three: list[Action], state: AppState
) -> None:
    a, b, c = three
    ran(a)

    plan = plan_slice_skip(workflow, state, awaiting_choice=False)

    assert plan is not None
    assert (plan.slice, plan.skipped, plan.finishing) == (SLICE, (b, c), None)


@pytest.mark.parametrize("state", [AppState.PAUSED, AppState.RELOADED])
def test_slice_skip_plan_stopped_needs_a_started_and_unfinished_slice(
    workflow: Workflow, three: list[Action], state: AppState
) -> None:
    # nothing ran: skipping would only empty the slice
    assert plan_slice_skip(workflow, state, awaiting_choice=False) is None

    ran(*three)
    assert plan_slice_skip(workflow, state, awaiting_choice=False) is None


def test_slice_skip_plan_running_lets_the_executing_action_finish(
    workflow: Workflow, three: list[Action]
) -> None:
    a, b, c = three

    plan = plan_slice_skip(workflow, AppState.RUNNING, awaiting_choice=False)
    assert plan is not None
    assert (plan.finishing, plan.skipped) == (a, (b, c))

    ran(a, b)
    # only the executing action is left
    assert plan_slice_skip(workflow, AppState.RUNNING, awaiting_choice=False) is None


def test_slice_skip_plan_interrupted_skips_the_failed_action_too(
    workflow: Workflow, three: list[Action]
) -> None:
    a, b, c = three
    ran(a)

    plan = plan_slice_skip(workflow, AppState.INTERRUPTED, awaiting_choice=True)
    assert plan is not None
    assert (plan.finishing, plan.skipped) == (None, (b, c))

    # the run ended with an error: nothing waits
    assert (
        plan_slice_skip(workflow, AppState.INTERRUPTED, awaiting_choice=False) is None
    )


@pytest.mark.parametrize(
    "state",
    [AppState.EDITING, AppState.FINISHED, AppState.STOPPING, AppState.SKIPPING],
)
def test_slice_skip_plan_is_none_in_other_states(
    workflow: Workflow, three: list[Action], state: AppState
) -> None:
    ran(three[0])
    assert plan_slice_skip(workflow, state, awaiting_choice=False) is None


def test_slice_skip_plan_carries_the_warnings_of_the_skipped_actions(
    workflow: Workflow, three: list[Action], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, c = three
    ran(a)
    seen: list[tuple[Action, ...]] = []

    def warn(skipped: tuple[Action, ...]) -> str:
        seen.append(tuple(skipped))
        return "mind the gap"

    monkeypatch.setattr(c, "skip_warning", warn)

    plan = plan_slice_skip(workflow, AppState.PAUSED, awaiting_choice=False)

    assert plan is not None
    (warning,) = plan.warnings
    assert "'c'" in warning and "mind the gap" in warning
    # judged against everything skipped with it
    assert seen == [(b, c)]

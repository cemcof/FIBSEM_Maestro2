# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gui_testkit.app import stop_run, wait_for_state
from gui_testkit.situations import busy_tasks, store_unappliable_props

from fibsem_maestro.ui.app.errors import NotAllowedError
from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTask
from fibsem_maestro.ui.app.run_controller import RunController
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.workflow.error import ActionError
from fibsem_maestro.workflow.error_choice import ErrorChoice
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable

    from gui_testkit.app import Services
    from gui_testkit.situations import HeldStep
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.session import WorkflowSession

TIMEOUT_MS = 30_000


@pytest.fixture
def actions(add_action: Callable[..., Action]) -> tuple[Action, Action]:
    """Two prepared actions."""
    return (
        add_action("adjust_props", "first", prepare=True),
        add_action("adjust_props", "second", prepare=True),
    )


@pytest.fixture
def last_slice(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Set the last slice a run acquires."""

    def set_last(value: int) -> None:
        monkeypatch.setattr(Workflow, "MAX_SLICE", value)

    return set_last


def states(run: RunController) -> list[AppState]:
    """The states `run` enters from now on."""
    entered: list[AppState] = []
    run.state_changed.connect(entered.append)
    return entered


def interrupt(qtbot: QtBot, run: RunController, action: Action) -> ActionError:
    """Start a run in which `action` fails; return the error once INTERRUPTED."""
    store_unappliable_props(action)
    with qtbot.waitSignal(run.action_failed, timeout=TIMEOUT_MS) as blocker:
        run.start()
    assert run.state is AppState.INTERRUPTED
    (error,) = blocker.args
    return error


def released(session: WorkflowSession) -> bool:
    """Whether the last run's worker was joined and released (it is no longer the callbacks)."""
    return session.workflow.callbacks is None


def test_run_initial_state_follows_workflow_slice(
    services: Services, session: WorkflowSession
) -> None:
    assert services.run.state is AppState.EDITING
    session.workflow.ctx.set_slice(3)
    assert RunController(session, services.tasks).state is AppState.RELOADED


def test_run_start_runs_and_locks_tasks(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    run, tasks = services.run, services.tasks
    finished: list[Action] = []
    run.action_finished.connect(finished.append)

    run.start()

    assert run.state is AppState.RUNNING
    assert tasks.locked
    qtbot.waitUntil(lambda: len(finished) >= 2, timeout=TIMEOUT_MS)
    assert finished[:2] == list(actions)


def test_run_pause_stops_after_current_action_then_resumes(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    run, tasks = services.run, services.tasks
    entered = states(run)
    run.start()

    run.pause()
    assert run.state is AppState.STOPPING
    assert tasks.locked
    wait_for_state(qtbot, run, AppState.PAUSED)
    assert not tasks.locked

    with qtbot.waitSignal(run.action_finished, timeout=TIMEOUT_MS):
        run.resume()
        assert run.state is AppState.RUNNING
        assert tasks.locked
    assert entered[:4] == [
        AppState.RUNNING,
        AppState.STOPPING,
        AppState.PAUSED,
        AppState.RUNNING,
    ]
    _ = actions


def test_run_finished_is_stable_and_joins_its_worker(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
    last_slice: Callable[[int], None],
) -> None:
    run = services.run
    last_slice(2)
    slices: list[int] = []
    run.slice_finished.connect(slices.append)

    run.start()
    wait_for_state(qtbot, run, AppState.FINISHED)

    qtbot.waitUntil(lambda: released(session), timeout=TIMEOUT_MS)
    with qtbot.assertNotEmitted(run.state_changed, wait=200):
        pass
    assert run.state is AppState.FINISHED
    assert not services.tasks.locked
    assert slices == [2, 3]
    assert [action.ctx.slice for action in actions] == [3, 3]


def test_run_start_from_finished_continues(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
    last_slice: Callable[[int], None],
) -> None:
    run = services.run
    last_slice(2)
    run.start()
    wait_for_state(qtbot, run, AppState.FINISHED)

    last_slice(4)
    run.start()
    assert run.state is AppState.RUNNING
    wait_for_state(qtbot, run, AppState.FINISHED)

    assert session.slice == 5
    assert [action.ctx.slice for action in actions] == [5, 5]


def test_run_action_error_interrupts_and_waits(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    first, second = actions
    error = interrupt(qtbot, services.run, first)

    assert isinstance(error, ActionError)
    assert error.action is first
    assert not services.tasks.locked
    with qtbot.assertNotEmitted(services.run.state_changed, wait=200):
        pass
    assert second.ctx.slice == 1


def test_run_retry_reexecutes_failed_action(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    last_slice: Callable[[int], None],
) -> None:
    first, second = actions
    last_slice(1)
    finished: list[Action] = []
    services.run.action_finished.connect(finished.append)
    interrupt(qtbot, services.run, first)
    first.prepare()  # the user fixes the cause while interrupted

    services.run.choose(ErrorChoice.RETRY)

    assert services.run.state is AppState.RUNNING
    wait_for_state(qtbot, services.run, AppState.FINISHED)
    assert finished == [first, second]


def test_run_skip_continues_with_next_action(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    last_slice: Callable[[int], None],
) -> None:
    first, second = actions
    last_slice(1)
    finished: list[Action] = []
    services.run.action_finished.connect(finished.append)
    interrupt(qtbot, services.run, first)

    services.run.choose(ErrorChoice.SKIP)

    assert services.run.state is AppState.RUNNING
    wait_for_state(qtbot, services.run, AppState.FINISHED)
    assert finished == [first, second]


def test_run_skip_and_pause_pauses_after_skipped_action(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    first, second = actions
    interrupt(qtbot, services.run, first)

    services.run.skip_and_pause()

    assert services.run.state is AppState.STOPPING
    wait_for_state(qtbot, services.run, AppState.PAUSED)
    assert (first.ctx.slice, second.ctx.slice) == (2, 1)


def test_run_choose_terminate_is_refused(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    interrupt(qtbot, services.run, actions[0])
    with pytest.raises(NotAllowedError):
        services.run.choose(ErrorChoice.TERMINATE)
    assert services.run.state is AppState.INTERRUPTED


def test_run_workflow_error_ends_the_run(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    add_action: Callable[..., Action],
) -> None:
    # an unprepared action cannot be initialised for the first slice
    add_action("adjust_props", "unprepared")
    run = services.run

    with qtbot.waitSignal(run.workflow_failed, timeout=TIMEOUT_MS) as blocker:
        run.start()

    assert isinstance(blocker.args[0], Exception)
    assert run.state is AppState.INTERRUPTED
    qtbot.waitUntil(lambda: released(session), timeout=TIMEOUT_MS)
    # nothing waits for a choice
    for call in (lambda: run.choose(ErrorChoice.RETRY), run.skip_and_pause):
        with pytest.raises(NotAllowedError):
            call()


@pytest.mark.parametrize(
    "call",
    [
        RunController.pause,
        RunController.resume,
        RunController.skip_and_pause,
        RunController.skip_slice,
    ],
    ids=["pause", "resume", "skip_and_pause", "skip_slice"],
)
def test_run_refuses_calls_while_editing(
    services: Services, call: Callable[[RunController], None]
) -> None:
    with pytest.raises(NotAllowedError):
        call(services.run)
    assert services.run.state is AppState.EDITING


def test_run_refuses_abort_and_editing_while_running(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    run = services.run
    run.start()
    for call in (run.abort, run.set_editing, run.start, run.resume):
        with pytest.raises(NotAllowedError):
            call()
    assert run.state is AppState.RUNNING
    _ = qtbot, actions


def test_run_refuses_start_while_task_busy(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
) -> None:
    with busy_tasks(qtbot, services.tasks), pytest.raises(NotAllowedError):
        services.run.start()
    assert services.run.state is AppState.EDITING
    assert session.workflow.callbacks is None  # no worker was created
    _ = actions


def test_run_refuses_resume_while_task_busy(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    run = services.run
    run.start()
    run.pause()
    wait_for_state(qtbot, run, AppState.PAUSED)

    with busy_tasks(qtbot, services.tasks), pytest.raises(NotAllowedError):
        run.resume()
    assert run.state is AppState.PAUSED
    _ = actions


@pytest.mark.parametrize("answer", ["retry", "skip_and_pause"])
def test_run_refuses_error_choice_while_task_busy(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action], answer: str
) -> None:
    run = services.run
    interrupt(qtbot, run, actions[0])
    call = (
        (lambda: run.choose(ErrorChoice.RETRY))
        if answer == "retry"
        else run.skip_and_pause
    )

    with busy_tasks(qtbot, services.tasks), pytest.raises(NotAllowedError):
        call()
    assert run.state is AppState.INTERRUPTED

    run.skip_and_pause()  # the choice is still awaited
    wait_for_state(qtbot, run, AppState.PAUSED)


def test_run_tasks_refused_while_running(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    services.run.start()
    ran: list[bool] = []
    handle = services.tasks.submit(
        MicroscopeTask(run=lambda: ran.append(True), description="x")
    )
    with qtbot.waitSignal(handle.failed, timeout=TIMEOUT_MS) as blocker:
        pass
    assert isinstance(blocker.args[0], NotAllowedError)
    assert ran == []
    _ = actions


def test_run_abort_while_paused_joins_worker(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
) -> None:
    run = services.run
    run.start()
    run.pause()
    wait_for_state(qtbot, run, AppState.PAUSED)

    run.abort()

    assert released(session)
    assert run.state is AppState.PAUSED
    with pytest.raises(NotAllowedError):
        run.resume()
    run.abort()  # idempotent
    run.set_editing()
    assert run.state is AppState.EDITING
    _ = actions


def test_run_abort_while_interrupted_joins_worker(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
) -> None:
    run = services.run
    interrupt(qtbot, run, actions[0])

    run.abort()

    assert released(session)
    assert run.state is AppState.INTERRUPTED
    with pytest.raises(NotAllowedError):
        run.choose(ErrorChoice.SKIP)
    run.set_editing()
    assert run.state is AppState.EDITING


def test_run_set_editing_refused_while_worker_exists(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    run = services.run
    run.start()
    run.pause()
    wait_for_state(qtbot, run, AppState.PAUSED)
    with pytest.raises(NotAllowedError):
        run.set_editing()
    _ = actions


def test_run_ignores_late_signal_of_aborted_run(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    last_slice: Callable[[int], None],
) -> None:
    run = services.run
    last_slice(3)
    run.start()
    run.pause()
    wait_for_state(qtbot, run, AppState.PAUSED)

    # the aborted worker's last signals are still queued when the next run starts
    run.abort()
    run.set_editing()
    run.start()

    wait_for_state(qtbot, run, AppState.FINISHED)
    assert [action.ctx.slice for action in actions] == [4, 4]


def held_in_first(
    qtbot: QtBot, run: RunController, hold: Callable[..., HeldStep], action: Action
) -> HeldStep:
    """Start a run and return once `action`, the first, executes its held step."""
    held = hold(action)
    run.start()
    qtbot.waitUntil(held.entered.is_set, timeout=TIMEOUT_MS)
    return held


def test_run_skip_slice_while_running_waits_for_the_current_action(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    hold: Callable[..., HeldStep],
) -> None:
    run = services.run
    first, second = actions
    entered = states(run)
    held = held_in_first(qtbot, run, hold, first)

    run.skip_slice()

    assert run.state is AppState.SKIPPING
    assert run.state.is_running and services.tasks.locked
    with qtbot.waitSignal(run.slice_skipped, timeout=TIMEOUT_MS) as blocker:
        held.release.set()
    assert blocker.args == [[second]]
    assert run.state is AppState.RUNNING
    assert entered[:3] == [AppState.RUNNING, AppState.SKIPPING, AppState.RUNNING]


def test_run_skip_slice_while_paused_skips_and_resumes(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    hold: Callable[..., HeldStep],
) -> None:
    run = services.run
    first, second = actions
    held = held_in_first(qtbot, run, hold, first)
    run.pause()
    held.release.set()
    wait_for_state(qtbot, run, AppState.PAUSED)
    assert (first.ctx.slice, second.ctx.slice) == (2, 1)

    with qtbot.waitSignal(run.slice_skipped, timeout=TIMEOUT_MS) as blocker:
        run.skip_slice()
        assert run.state is AppState.SKIPPING
    assert blocker.args == [[second]]
    assert run.state is AppState.RUNNING


def test_run_pause_while_skipping_pauses_before_the_next_slice(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
    hold: Callable[..., HeldStep],
) -> None:
    run = services.run
    held = held_in_first(qtbot, run, hold, actions[0])
    run.skip_slice()

    run.pause()
    assert run.state is AppState.STOPPING
    held.release.set()

    wait_for_state(qtbot, run, AppState.PAUSED)
    # the skip went first; nothing of the next slice ran
    assert session.slice == 2
    assert [action.ctx.slice for action in actions] == [2, 2]


def test_run_skip_slice_after_an_error_skips_the_failed_action_with_the_rest(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    last_slice: Callable[[int], None],
) -> None:
    run = services.run
    first, second = actions
    last_slice(1)
    interrupt(qtbot, run, first)

    with qtbot.waitSignal(run.slice_skipped, timeout=TIMEOUT_MS) as blocker:
        run.skip_slice()
        assert run.state is AppState.SKIPPING
    # the failed action was skipped as with Skip, then the rest
    assert blocker.args == [[second]]
    wait_for_state(qtbot, run, AppState.FINISHED)
    assert [action.ctx.slice for action in actions] == [2, 2]


def test_run_skip_requested_before_an_error_is_still_pending_after_the_choice(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    hold: Callable[..., HeldStep],
) -> None:
    run = services.run
    first, second = actions
    held = hold(first, fail=True)
    run.start()
    qtbot.waitUntil(held.entered.is_set, timeout=TIMEOUT_MS)
    run.skip_slice()
    with qtbot.waitSignal(run.action_failed, timeout=TIMEOUT_MS):
        held.release.set()

    with qtbot.waitSignal(run.slice_skipped, timeout=TIMEOUT_MS) as blocker:
        run.choose(ErrorChoice.RETRY)
        assert run.state is AppState.SKIPPING
    assert blocker.args == [[second]]
    assert run.state is AppState.RUNNING


def test_run_skip_slice_after_a_reload_starts_the_run(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    actions: tuple[Action, Action],
    hold: Callable[..., HeldStep],
    last_slice: Callable[[int], None],
) -> None:
    first, second = actions
    # stopped mid-slice, then the application restarted
    held = held_in_first(qtbot, services.run, hold, first)
    services.run.pause()
    held.release.set()
    wait_for_state(qtbot, services.run, AppState.PAUSED)
    services.run.abort()
    reloaded = RunController(session, services.tasks)
    assert reloaded.state is AppState.RELOADED
    last_slice(1)

    try:
        with qtbot.waitSignal(reloaded.slice_skipped, timeout=TIMEOUT_MS) as blocker:
            reloaded.skip_slice()
            assert reloaded.state is AppState.SKIPPING
        assert blocker.args == [[second]]
        wait_for_state(qtbot, reloaded, AppState.FINISHED)
        assert [action.ctx.slice for action in actions] == [2, 2]
    finally:
        stop_run(qtbot, reloaded)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""`ActionStatus` and `indicators_for`: the check mark, arrow and slice of each list item."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gui_testkit.app import wait_for_state
from gui_testkit.situations import store_unappliable_props

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.ui.app.action_status import (
    ActionIndicator,
    ActionStatus,
    indicators_for,
)
from fibsem_maestro.ui.app.run_controller import RunController
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable

    from gui_testkit.app import Services
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


def arrows(services: Services, actions: tuple[Action, ...]) -> list[bool]:
    return [services.status.indicator(action).next_to_run for action in actions]


def test_indicators_check_mark_needs_props(add_action: Callable[..., Action]) -> None:
    raw = add_action("adjust_props", "raw")
    prepared = add_action("adjust_props", "prepared", prepare=True)
    indicators = indicators_for([raw, prepared], None, AppState.EDITING)
    assert indicators[raw] == ActionIndicator(
        prepared=False, next_to_run=False, slice=0
    )
    assert indicators[prepared].prepared


def test_indicators_check_mark_needs_no_preparation_issues(
    add_action: Callable[..., Action],
) -> None:
    # any registered action that needs more than its props to run will do
    for key in ACTION_REGISTRY:
        action = add_action(key, f"{key} action")
        if action.preparation_issues():
            break
    else:
        pytest.skip("no registered action reports preparation issues")
    action.write_properties(action.collect_properties())

    assert action.has_stored_properties()
    assert not indicators_for([action], None, AppState.EDITING)[action].prepared


def test_indicators_arrow_only_outside_editing(
    actions: tuple[Action, Action],
) -> None:
    first, second = actions
    for state in AppState:
        indicators = indicators_for(actions, second, state)
        assert indicators[second].next_to_run is (state is not AppState.EDITING), state
        assert not indicators[first].next_to_run


def test_status_arrow_after_a_reload_marks_the_next_action_to_run(
    services: Services, session: WorkflowSession, actions: tuple[Action, Action]
) -> None:
    first, second = actions
    # stopped after the first action of slice 3, then the application restarted
    session.workflow.ctx.set_slice(3)
    first.ctx.set_slice(4)
    second.ctx.set_slice(3)
    run = RunController(session, services.tasks)
    assert run.state is AppState.RELOADED

    status = ActionStatus(session, run, services.tasks)

    assert not status.indicator(first).next_to_run
    assert status.indicator(second).next_to_run


def test_status_arrow_follows_interruptions_and_pauses(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    first, second = actions
    run = services.run
    store_unappliable_props(first)
    assert arrows(services, actions) == [False, False]

    run.start()
    wait_for_state(qtbot, run, AppState.INTERRUPTED)
    assert arrows(services, actions) == [True, False]

    run.skip_and_pause()
    wait_for_state(qtbot, run, AppState.PAUSED)
    # D5: the action after the pause point, from the slice counters
    assert arrows(services, actions) == [False, True]
    assert services.status.indicator(first).slice == 2

    store_unappliable_props(second)
    run.resume()
    wait_for_state(qtbot, run, AppState.INTERRUPTED)
    assert arrows(services, actions) == [False, True]

    run.skip_and_pause()
    wait_for_state(qtbot, run, AppState.PAUSED)
    # paused after the last action of a slice: the first action runs next
    assert arrows(services, actions) == [True, False]


def test_status_follows_finished_run(
    qtbot: QtBot,
    services: Services,
    actions: tuple[Action, Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Workflow, "MAX_SLICE", 2)
    changes: list[None] = []
    services.status.changed.connect(lambda: changes.append(None))

    services.run.start()
    wait_for_state(qtbot, services.run, AppState.FINISHED)

    assert changes
    assert services.status.indicator(actions[0]) == ActionIndicator(
        prepared=True, next_to_run=True, slice=3
    )


def test_status_hides_arrow_after_reset(
    qtbot: QtBot, services: Services, actions: tuple[Action, Action]
) -> None:
    store_unappliable_props(actions[0])
    services.run.start()
    wait_for_state(qtbot, services.run, AppState.INTERRUPTED)

    services.commands.reset_workflow()

    assert arrows(services, actions) == [False, False]
    assert not services.status.indicator(actions[0]).prepared


def test_status_emits_only_on_change(
    qtbot: QtBot, services: Services, session: WorkflowSession
) -> None:
    with qtbot.waitSignal(services.status.changed):
        action = session.add_action("adjust_props", "action")
    with qtbot.assertNotEmitted(services.status.changed):
        session.commit_settings(action)
    action.prepare()
    with qtbot.waitSignal(services.status.changed):
        session.props_changed(action)
    assert services.status.indicator(action).prepared

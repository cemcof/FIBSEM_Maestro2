# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""`EditPolicy` and `permissions_for`: every enable/disable rule."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pytest
from gui_testkit.situations import busy_tasks

from fibsem_maestro.ui.app.edit_policy import Permissions, permissions_for
from fibsem_maestro.ui.app.run_state import AppState

if TYPE_CHECKING:
    from collections.abc import Callable

    from gui_testkit.app import Services
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action

S = AppState


def spec_table(state: AppState, danger: bool, busy: bool) -> Permissions:
    """The permission table of the specification, written out independently."""
    not_running = state not in (S.RUNNING, S.STOPPING, S.SKIPPING)
    return Permissions(
        structure_editable=not_running and state is not S.INTERRUPTED and not busy,
        settings_editable=not_running or danger,
        rules_editable=not_running,
        action_tools_enabled=not_running and not busy,
        danger_tools_enabled=(not_running and not busy) or danger,
        preview_enabled=not_running and not busy,
        import_enabled=state is S.EDITING and not busy,
        new_enabled=not_running and not busy,
        reset_enabled=not_running and not busy,
        microscope_settings_editable=not_running or danger,
        error_choice_enabled=state is S.INTERRUPTED and not busy,
        slice_skip_enabled=state is S.RUNNING
        or (state in (S.PAUSED, S.RELOADED, S.INTERRUPTED) and not busy),
        task_busy=busy,
    )


@pytest.mark.parametrize(
    ("state", "danger", "busy"),
    list(itertools.product(AppState, (False, True), (False, True))),
)
def test_permissions_for_matches_spec_table(
    state: AppState, danger: bool, busy: bool
) -> None:
    assert permissions_for(state, danger, busy) == spec_table(state, danger, busy)


def test_permissions_for_keeps_rules_and_tools_in_interrupted() -> None:
    """Decision Q10/N4: the failed action stays, but it can be fixed and prepared."""
    permissions = permissions_for(S.INTERRUPTED, danger_mode=False, task_busy=False)
    assert not permissions.structure_editable
    assert permissions.settings_editable
    assert permissions.rules_editable
    assert permissions.action_tools_enabled
    assert permissions.reset_enabled
    assert permissions.new_enabled


def test_policy_starts_from_current_state(services: Services) -> None:
    assert services.policy.permissions == spec_table(S.EDITING, False, False)
    assert services.policy.preview_allowed()


def test_policy_follows_task_busy(qtbot: QtBot, services: Services) -> None:
    policy = services.policy
    received: list[Permissions] = []
    policy.changed.connect(received.append)

    with busy_tasks(qtbot, services.tasks):
        assert policy.permissions == spec_table(S.EDITING, False, True)
        assert not policy.preview_allowed()

    assert received == [
        spec_table(S.EDITING, False, True),
        spec_table(S.EDITING, False, False),
    ]


def test_policy_danger_mode_emits_permissions_only_on_change(
    qtbot: QtBot, services: Services
) -> None:
    policy = services.policy
    # outside a run, danger mode unlocks nothing new
    with (
        qtbot.waitSignal(policy.danger_mode_changed) as blocker,
        qtbot.assertNotEmitted(policy.changed),
    ):
        services.commands.set_danger_mode(True)
    assert blocker.args == [True]
    assert policy.danger_mode

    with qtbot.assertNotEmitted(policy.danger_mode_changed):
        policy.set_danger_mode(True)


def test_policy_follows_run_state(
    qtbot: QtBot, services: Services, add_action: Callable[..., Action]
) -> None:
    add_action("adjust_props", "action", prepare=True)
    policy = services.policy
    policy.set_danger_mode(True)

    with qtbot.waitSignal(policy.changed) as blocker:
        services.run.start()

    assert blocker.args == [spec_table(S.RUNNING, True, False)]
    assert policy.permissions.settings_editable
    assert not policy.permissions.rules_editable

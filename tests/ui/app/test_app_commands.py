# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
`AppCommands`: every user intent, its permission check and its orchestration.

Commands that start microscope work refuse through a handle emitting `failed`;
every other command raises `NotAllowedError`.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest
from gui_testkit.app import build_services, shut_down, wait_for_state, wait_for_task
from gui_testkit.situations import busy_tasks, store_unappliable_props
from gui_testkit.windows_lock import windows_like_delete

from fibsem_maestro.logging.text.file import close_all_log_files
from fibsem_maestro.milling.milling import MillingState
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.ui.app import commands as commands_module
from fibsem_maestro.ui.app.errors import (
    DirectoryNotClearedError,
    NotAllowedError,
    PropertiesChangedError,
    WorkflowResetError,
)
from fibsem_maestro.ui.app.files import WorkflowFiles
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.app.session import WorkflowSession
from fibsem_maestro.ui.ports import PreviewRequest
from fibsem_maestro.workflow.error_choice import ErrorChoice

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from gui_testkit.app import Services
    from gui_testkit.situations import HeldStep
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.user_config import UserConfig
    from fibsem_maestro.ui.ports import TaskHandle
    from fibsem_maestro.workflow.workflow import Workflow

KEY = "adjust_props"
TIMEOUT_MS = 30_000
PREVIEW = PreviewRequest(beam=None, offset_x_nm=0.0, previous_offset_nm=None, crop=None)


@pytest.fixture
def prepared(add_action: Callable[..., Action]) -> tuple[Action, Action]:
    first = add_action(KEY, "first", prepare=True)
    second = add_action(KEY, "second", prepare=True)
    return first, second


@pytest.fixture
def windows_services(
    qtbot: QtBot, workflow: Workflow, workflow_dir: Path, user_config: UserConfig
) -> Iterator[Services]:
    """Services whose deletions fail like on Windows while a file is open."""
    files = WorkflowFiles(windows_like_delete)
    services = build_services(
        WorkflowSession(workflow, workflow_dir, files), user_config, files
    )
    yield services
    shut_down(qtbot, services.run, services.tasks)


def interrupt(qtbot: QtBot, services: Services, action: Action) -> None:
    store_unappliable_props(action)
    services.commands.run_pressed()
    wait_for_state(qtbot, services.run, AppState.INTERRUPTED)


def pause(qtbot: QtBot, services: Services) -> None:
    services.commands.run_pressed()
    services.commands.run_pressed()
    wait_for_state(qtbot, services.run, AppState.PAUSED)


def refusal(qtbot: QtBot, handle: TaskHandle) -> object:
    """The error a refused task handle reports."""
    with qtbot.waitSignal(handle.failed, timeout=TIMEOUT_MS) as blocker:
        pass
    return blocker.args[0]


def run_log(ctx: ActionContext) -> str:
    """The text of the log of a context's current slice, flushed first."""
    close_all_log_files()
    return (ctx.current_view.path() / "run.log").read_text(encoding="utf-8")


def add_prepared(services: Services, name: str) -> Action:
    action = services.session.add_action(KEY, name)
    action.prepare()
    services.session.props_changed(action)
    return action


STRUCTURE_COMMANDS: dict[str, Callable[[AppCommands, Action], object]] = {
    "add": lambda c, _: c.add_action(KEY, "new"),
    "duplicate": lambda c, a: c.duplicate_action(a),
    "remove": lambda c, a: c.remove_action(a),
    "rename": lambda c, a: c.rename_action(a, "renamed"),
    "move": lambda c, _: c.move_actions(1, 1, 0),
}


def test_commands_structure_refused_while_task_busy(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    with busy_tasks(qtbot, services.tasks):
        for name, command in STRUCTURE_COMMANDS.items():
            with pytest.raises(NotAllowedError):
                command(services.commands, prepared[0])
                pytest.fail(f"{name} was not refused")
    assert services.session.actions == prepared
    assert prepared[0].name == "first"


def test_commands_structure_refused_while_interrupted(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    interrupt(qtbot, services, prepared[0])
    for command in STRUCTURE_COMMANDS.values():
        with pytest.raises(NotAllowedError):
            command(services.commands, prepared[0])
    assert services.session.actions == prepared


def test_commands_rules_refused_while_running(
    services: Services, session: WorkflowSession, prepared: tuple[Action, Action]
) -> None:
    parent, _ = prepared
    rule = services.commands.add_rule(parent)
    services.commands.run_pressed()

    for command in (
        lambda: services.commands.add_rule(parent),
        lambda: services.commands.set_rule_dependents(rule, ["second"]),
        lambda: services.commands.set_rule_properties(
            rule, PropertyNames(microscope=["x"])
        ),
        lambda: services.commands.remove_rule(rule),
    ):
        with pytest.raises(NotAllowedError):
            command()
    assert session.rules_of(parent) == [rule]
    assert rule.dependent_names == []


def test_commands_rules_allowed_while_interrupted(
    qtbot: QtBot,
    services: Services,
    session: WorkflowSession,
    prepared: tuple[Action, Action],
) -> None:
    interrupt(qtbot, services, prepared[0])
    rule = services.commands.add_rule(prepared[0])
    services.commands.set_rule_dependents(rule, ["second"])
    assert session.rules_of(prepared[0])[0].dependent_names == ["second"]


TASK_COMMANDS: dict[str, Callable[[AppCommands, Action, list[str]], TaskHandle]] = {
    "prepare": lambda c, a, _: c.prepare_action(a),
    "apply": lambda c, a, _: c.apply_properties(a),
    "test": lambda c, a, _: c.test_action(a),
    "custom": lambda c, a, ran: c.run_microscope_task(
        a, "Custom", lambda: ran.append("x")
    ),
    "grab": lambda c, _, __: c.grab(PREVIEW),
}


@pytest.mark.parametrize("command", list(TASK_COMMANDS))
def test_commands_tasks_refused_by_failing_handle_while_busy(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action], command: str
) -> None:
    ran: list[str] = []
    with busy_tasks(qtbot, services.tasks):
        handle = TASK_COMMANDS[command](services.commands, prepared[0], ran)
        assert isinstance(refusal(qtbot, handle), NotAllowedError)
    assert ran == []


def test_commands_tasks_refused_by_failing_handle_while_running(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    services.commands.run_pressed()
    for command in TASK_COMMANDS.values():
        handle = command(services.commands, prepared[0], [])
        assert isinstance(refusal(qtbot, handle), NotAllowedError)
        assert not services.tasks.busy


def test_commands_whole_workflow_refused_while_task_busy(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action], tmp_path: Path
) -> None:
    commands = services.commands
    with busy_tasks(qtbot, services.tasks):
        for command in (
            lambda: commands.new_workflow(tmp_path / "new"),
            lambda: commands.import_workflow(tmp_path / "source"),
            commands.reset_workflow,
            lambda: save_unchanged(commands, prepared[0]),
            lambda: commands.reset_action_state(prepared[0]),
            commands.run_pressed,
        ):
            with pytest.raises(NotAllowedError):
                command()
    assert services.run.state is AppState.EDITING


def test_commands_import_refused_unless_editing(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action], tmp_path: Path
) -> None:
    pause(qtbot, services)
    with pytest.raises(NotAllowedError):
        services.commands.import_workflow(tmp_path)
    assert services.session.actions == prepared


def test_commands_error_choices_refused_outside_interrupted(services: Services) -> None:
    with pytest.raises(NotAllowedError):
        services.commands.choose_error_option(ErrorChoice.SKIP)
    with pytest.raises(NotAllowedError):
        services.commands.skip_and_pause()


def test_commands_run_pressed_refused_unless_ready(
    services: Services, add_action: Callable[..., Action]
) -> None:
    add_action(KEY, "unprepared")
    with pytest.raises(NotAllowedError):
        services.commands.run_pressed()
    assert services.run.state is AppState.EDITING
    assert services.session.workflow.callbacks is None


def test_commands_run_pressed_starts_pauses_and_resumes(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    commands, run = services.commands, services.run
    commands.run_pressed()
    assert run.state is AppState.RUNNING
    commands.run_pressed()
    assert run.state is AppState.STOPPING
    commands.run_pressed()  # nothing to do while stopping
    wait_for_state(qtbot, run, AppState.PAUSED)

    services.validity.set_has_errors(prepared[0], True)
    with pytest.raises(NotAllowedError):
        commands.run_pressed()
    services.validity.set_has_errors(prepared[0], False)
    commands.run_pressed()
    assert run.state is AppState.RUNNING


def test_commands_skip_slice_refused_with_nothing_to_skip(
    services: Services, prepared: tuple[Action, Action]
) -> None:
    with pytest.raises(NotAllowedError):
        services.commands.skip_slice()
    assert services.run.state is AppState.EDITING
    assert [action.ctx.slice for action in prepared] == [0, 0]


def test_commands_skip_slice_from_paused_needs_a_ready_workflow(
    qtbot: QtBot,
    services: Services,
    prepared: tuple[Action, Action],
    hold: Callable[..., HeldStep],
) -> None:
    commands, run = services.commands, services.run
    held = hold(prepared[0])
    commands.run_pressed()
    qtbot.waitUntil(held.entered.is_set, timeout=TIMEOUT_MS)
    commands.run_pressed()
    held.release.set()
    wait_for_state(qtbot, run, AppState.PAUSED)

    # skipping resumes the run, as the run button would
    services.validity.set_has_errors(prepared[0], True)
    with pytest.raises(NotAllowedError):
        commands.skip_slice()
    assert run.state is AppState.PAUSED

    services.validity.set_has_errors(prepared[0], False)
    commands.skip_slice()
    assert run.state is AppState.SKIPPING
    wait_for_state(qtbot, run, AppState.PAUSED)


def test_commands_error_choice_resumes_interrupted_run(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    interrupt(qtbot, services, prepared[0])
    services.commands.skip_and_pause()
    wait_for_state(qtbot, services.run, AppState.PAUSED)
    assert prepared[0].ctx.slice == 2


def test_commands_new_workflow_from_interrupted_run(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action], tmp_path: Path
) -> None:
    interrupt(qtbot, services, prepared[0])
    old = services.session.workflow
    target = tmp_path / "new"
    (target / "leftover").mkdir(parents=True)

    services.commands.new_workflow(target)

    assert old.callbacks is None  # the waiting worker was released
    assert services.run.state is AppState.EDITING
    assert services.session.workflow_dir == target
    assert services.session.actions == ()
    assert not (target / "leftover").exists()


def test_commands_new_workflow_clear_failure_changes_nothing(
    qtbot: QtBot, windows_services: Services, tmp_path: Path
) -> None:
    services = windows_services
    add_prepared(services, "action")
    pause(qtbot, services)
    old = services.session.workflow
    target = tmp_path / "new"
    (target / "held").mkdir(parents=True)

    with (
        (target / "held" / "file.txt").open("w"),
        pytest.raises(DirectoryNotClearedError),
    ):
        services.commands.new_workflow(target)

    assert services.session.workflow is old
    assert services.run.state is AppState.PAUSED
    assert old.callbacks is not None  # the paused run was not abandoned


def test_commands_reset_from_interrupted_run(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    interrupt(qtbot, services, prepared[0])
    workflow = services.session.workflow

    services.commands.reset_workflow()

    assert workflow.callbacks is None
    assert services.run.state is AppState.EDITING
    assert [a.ctx.slice for a in prepared] == [0, 0]
    assert services.session.slice == 0


def test_commands_failed_reset_still_ends_in_editing(
    qtbot: QtBot, windows_services: Services
) -> None:
    services = windows_services
    action = add_prepared(services, "action")
    pause(qtbot, services)

    with (
        (action.ctx.current_view.path() / "held.txt").open("w"),
        pytest.raises(WorkflowResetError),
    ):
        services.commands.reset_workflow()

    assert services.run.state is AppState.EDITING
    assert services.session.workflow.callbacks is None


def test_commands_import_replaces_actions(
    services: Services,
    prepared: tuple[Action, Action],
    workflow_dir: Path,
    tmp_path: Path,
) -> None:
    # importing a copy of this very workflow from elsewhere
    close_all_log_files()
    shutil.copytree(workflow_dir, tmp_path / "copy")
    services.commands.remove_action(prepared[1])

    services.commands.import_workflow(tmp_path / "copy")

    assert [a.name for a in services.session.actions] == ["first", "second"]
    assert all(a not in prepared for a in services.session.actions)


def test_commands_terminate_logs_aborts_and_closes_logs(
    qtbot: QtBot,
    services: Services,
    prepared: tuple[Action, Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupt(qtbot, services, prepared[0])
    workflow = services.session.workflow
    closed: list[bool] = []
    monkeypatch.setattr(
        commands_module, "close_all_log_files", lambda: closed.append(True)
    )

    services.commands.terminate()

    assert workflow.callbacks is None
    assert closed == [True]
    assert "WARNING: Terminating workflow." in run_log(workflow.ctx)


def test_commands_prepare_exit_closes_logs(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(
        commands_module, "close_all_log_files", lambda: closed.append(True)
    )
    services.commands.prepare_exit()
    assert closed == [True]


def test_commands_prepare_announces_props_and_logs(
    qtbot: QtBot, services: Services, add_action: Callable[..., Action]
) -> None:
    action = add_action(KEY, "action")
    order: list[str] = []
    services.session.action_props_changed.connect(lambda _: order.append("props"))

    handle = services.commands.prepare_action(action)
    handle.succeeded.connect(lambda _: order.append("succeeded"))
    wait_for_task(qtbot, services.tasks, handle)

    assert order == ["props", "succeeded"]
    assert action.has_stored_properties()
    log = run_log(action.ctx)
    assert "Preparing 'action'..." in log
    assert "Preparing 'action' completed." in log


def test_commands_settings_committed_persists_and_announces(
    services: Services, prepared: tuple[Action, Action]
) -> None:
    action = prepared[0]
    changed: list[Action] = []
    services.session.action_settings_changed.connect(changed.append)
    action.settings.properties_to_collect = PropertyNames(microscope=["stage_position"])

    services.commands.settings_committed(action)

    stored = action.ctx.settings_store.read("settings.yaml", type(action.settings))
    assert stored.properties_to_collect.microscope == ["stage_position"]
    assert changed == [action]


def test_commands_settings_committed_ignores_removed_action(
    services: Services, add_action: Callable[..., Action]
) -> None:
    old = add_action(KEY, "action")
    services.commands.remove_action(old)
    new = add_action(KEY, "action")  # the same directory
    changed: list[Action] = []
    services.session.action_settings_changed.connect(changed.append)
    old.settings.properties_to_collect = PropertyNames(microscope=["stage_position"])

    services.commands.settings_committed(old)

    stored = new.ctx.settings_store.read("settings.yaml", type(new.settings))
    assert stored.properties_to_collect.microscope == []
    assert changed == []


def save_unchanged(commands: AppCommands, action: Action) -> None:
    """Save the action's stored props as opened just now, unchanged."""
    props = action.read_properties()
    commands.save_properties(
        action, props, opened_slice=action.ctx.slice, opened=props.model_copy()
    )


EDITED = GlobalProperties(electron_beam=BeamProperties(working_distance=4.0e6))
"""Edited props, unlike anything the tests store otherwise."""


def test_commands_save_properties_writes_props(
    services: Services, add_action: Callable[..., Action]
) -> None:
    source = add_action(KEY, "source", prepare=True)
    target = add_action(KEY, "target")
    services.commands.save_properties(
        target, source.read_properties(), opened_slice=target.ctx.slice, opened=None
    )
    assert target.has_stored_properties()


def test_commands_save_properties_refuses_props_changed_since_opened(
    services: Services, prepared: tuple[Action, Action]
) -> None:
    action = prepared[0]
    opened = action.read_properties()
    # e.g. the running workflow propagated to the action meanwhile
    meanwhile = GlobalProperties(electron_beam=BeamProperties(working_distance=1.0e6))
    action.write_properties(meanwhile)

    with pytest.raises(PropertiesChangedError):
        services.commands.save_properties(
            action, EDITED, opened_slice=action.ctx.slice, opened=opened
        )

    assert action.read_properties() == meanwhile


def test_commands_save_properties_refuses_props_of_an_earlier_slice(
    services: Services, prepared: tuple[Action, Action]
) -> None:
    action = prepared[0]
    opened_slice, opened = action.ctx.slice, action.read_properties()
    # e.g. the running workflow executed the action meanwhile
    action.ctx.props_store.copy_to("props.yaml", action.ctx.props_store.next)
    action.ctx.advance()

    with pytest.raises(PropertiesChangedError):
        services.commands.save_properties(
            action, EDITED, opened_slice=opened_slice, opened=opened
        )

    assert action.read_properties() == opened


def test_commands_save_properties_allowed_in_danger_mode_while_busy(
    qtbot: QtBot, services: Services, prepared: tuple[Action, Action]
) -> None:
    action = prepared[0]
    services.commands.set_danger_mode(True)

    with busy_tasks(qtbot, services.tasks):
        services.commands.save_properties(
            action,
            EDITED,
            opened_slice=action.ctx.slice,
            opened=action.read_properties(),
        )

    assert action.read_properties() == EDITED


def test_commands_apply_microscope_settings_saves_profile_on_request(
    services: Services, user_config: UserConfig
) -> None:
    services.commands.apply_microscope_settings(save_profile=False)
    assert user_config.load_microscope_profile() is None

    services.commands.apply_microscope_settings(save_profile=True)
    assert user_config.load_microscope_profile() == services.session.microscope.settings


def test_commands_save_properties_lets_every_action_discard_invalidated_state(
    services: Services,
    prepared: tuple[Action, Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = prepared
    replaced: list[Action] = []
    # e.g. a step-mode autofunction aborts its run when its imaging's props change
    monkeypatch.setattr(second, "_on_properties_replaced", replaced.append)

    save_unchanged(services.commands, first)

    assert replaced == [first]


def test_commands_reset_action_state_resets_and_announces(
    qtbot: QtBot, services: Services, add_action: Callable[..., Action]
) -> None:
    milling = add_action("milling", "milling")
    milling.set_state(MillingState(slices_milled=2))

    with qtbot.waitSignal(services.session.action_state_changed) as changed:
        services.commands.reset_action_state(milling)

    assert changed.args == [milling]
    assert milling.state == MillingState()


def test_commands_reset_action_state_ignores_a_removed_action(
    qtbot: QtBot, services: Services, add_action: Callable[..., Action]
) -> None:
    milling = add_action("milling", "milling")
    milling.set_state(MillingState(slices_milled=2))
    services.session.remove_action(milling)

    with qtbot.assertNotEmitted(services.session.action_state_changed):
        services.commands.reset_action_state(milling)

    assert milling.state == MillingState(slices_milled=2)

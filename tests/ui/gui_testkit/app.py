# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The application layer and the composed main window, built and shut down for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fibsem_maestro.ui.app.action_status import ActionStatus
from fibsem_maestro.ui.app.audit import SettingsAudit
from fibsem_maestro.ui.app.catalogs import MicroscopeCatalogs
from fibsem_maestro.ui.app.commands import AppCommands
from fibsem_maestro.ui.app.edit_policy import EditPolicy
from fibsem_maestro.ui.app.form_validity import FormValidity
from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
from fibsem_maestro.ui.app.previews import PreviewService
from fibsem_maestro.ui.app.readiness import Readiness
from fibsem_maestro.ui.app.run_controller import RunController
from fibsem_maestro.ui.app.run_state import AppState
from gui_testkit import TIMEOUT_MS

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.files import WorkflowFiles
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.app.user_config import UserConfig
    from fibsem_maestro.ui.ports import TaskHandle


@dataclass(frozen=True)
class Services:
    """
    The whole application layer, wired as the composition root wires it, without views.

    Attributes:
        session: The workflow session.
        tasks: The microscope task runner.
        run: The run controller.
        policy: The permissions.
        validity: Which settings forms show errors.
        catalogs: Pattern types and manufacturer property names.
        audit: The settings audit.
        readiness: Run readiness.
        status: Per-action indicators.
        commands: Every user intent.
    """

    session: WorkflowSession
    tasks: MicroscopeTasks
    run: RunController
    policy: EditPolicy
    validity: FormValidity
    catalogs: MicroscopeCatalogs
    audit: SettingsAudit
    readiness: Readiness
    status: ActionStatus
    commands: AppCommands


def build_services(
    session: WorkflowSession, user_config: UserConfig, files: WorkflowFiles
) -> Services:
    """
    Construct the application layer around a session.

    Args:
        session: The workflow session.
        user_config: The remembered profiles.
        files: Filesystem operations (the session's).

    Returns:
        The services; call `shut_down` when done.
    """
    tasks = MicroscopeTasks()
    run = RunController(session, tasks)
    policy = EditPolicy(run, tasks)
    validity = FormValidity()
    catalogs = MicroscopeCatalogs.query(session.microscope)
    audit = SettingsAudit(session, catalogs)
    readiness = Readiness(session, run, tasks, validity, audit)
    status = ActionStatus(session, run, tasks)
    previews = PreviewService(session, tasks)
    commands = AppCommands(
        session, run, tasks, previews, policy, readiness, user_config, files
    )
    return Services(
        session,
        tasks,
        run,
        policy,
        validity,
        catalogs,
        audit,
        readiness,
        status,
        commands,
    )


def stop_run(qtbot: QtBot, run: RunController) -> None:
    """
    Bring a run to a stop and join its worker, whatever state it is in.

    Args:
        qtbot: The Qt test helper.
        run: The run controller.
    """
    if run.state is AppState.RUNNING:
        run.pause()
    if run.state.is_running:
        qtbot.waitUntil(lambda: not run.state.is_running, timeout=TIMEOUT_MS)
    run.abort()


def shut_down(qtbot: QtBot, run: RunController, tasks: MicroscopeTasks) -> None:
    """
    Leave no thread running: stop the run, wait for a pending task, stop the task thread.

    Args:
        qtbot: The Qt test helper.
        run: The run controller.
        tasks: The task runner.
    """
    stop_run(qtbot, run)
    if tasks.busy:
        qtbot.waitUntil(lambda: not tasks.busy, timeout=TIMEOUT_MS)
    tasks.shutdown(timeout_ms=TIMEOUT_MS)


def wait_for_state(qtbot: QtBot, run: RunController, state: AppState) -> None:
    """
    Wait until the run controller reaches `state`.

    Args:
        qtbot: The Qt test helper.
        run: The run controller.
        state: The awaited state.
    """
    qtbot.waitUntil(lambda: run.state is state, timeout=TIMEOUT_MS)


def wait_for_task(qtbot: QtBot, tasks: MicroscopeTasks, handle: TaskHandle) -> None:
    """
    Wait until a submitted task has finished and the runner is idle again.

    Args:
        qtbot: The Qt test helper.
        tasks: The task runner.
        handle: The task's handle (kept alive by the caller until here).
    """
    _ = handle
    qtbot.waitUntil(lambda: not tasks.busy, timeout=TIMEOUT_MS)


def prepare(
    qtbot: QtBot, commands: AppCommands, tasks: MicroscopeTasks, action: Action
) -> None:
    """
    Prepare an action through the commands, as the Prepare tool does, and wait for it.

    Args:
        qtbot: The Qt test helper.
        commands: The commands.
        tasks: The task runner.
        action: The action to prepare.
    """
    wait_for_task(qtbot, tasks, commands.prepare_action(action))

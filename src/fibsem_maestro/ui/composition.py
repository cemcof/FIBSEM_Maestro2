# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
The composition root: constructs every service and view of the main window and wires them.

This is the only place where the application layer's services are created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fibsem_maestro.ui.app.action_status import ActionStatus
from fibsem_maestro.ui.app.audit import SettingsAudit
from fibsem_maestro.ui.app.commands import AppCommands
from fibsem_maestro.ui.app.edit_policy import EditPolicy
from fibsem_maestro.ui.app.form_validity import FormValidity
from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
from fibsem_maestro.ui.app.previews import PreviewService
from fibsem_maestro.ui.app.readiness import Readiness
from fibsem_maestro.ui.app.run_controller import RunController
from fibsem_maestro.ui.app.session import WorkflowSession
from fibsem_maestro.ui.forms.builder import FormBuilder
from fibsem_maestro.ui.forms.context import FormContext
from fibsem_maestro.ui.forms.registry import default_registry
from fibsem_maestro.ui.views.action_panel.host import (
    ActionPanelFactory,
    ActionPanelHost,
)
from fibsem_maestro.ui.views.main_window import MainWindow

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.catalogs import MicroscopeCatalogs
    from fibsem_maestro.ui.app.files import WorkflowFiles
    from fibsem_maestro.ui.app.session import SessionSeed
    from fibsem_maestro.ui.app.user_config import UserConfig
    from fibsem_maestro.ui.ports import PreviewPermission, PreviewPort


class FormContexts:
    """
    Builds the `FormContext` of each kind of form of the main window.

    Args:
        session: The workflow session (loggers, the action catalog).
        catalogs: What the connected microscope offers.
        previews: Acquires area-selector images (`AppCommands`).
        permission: Whether area selectors may acquire now (`EditPolicy`).
    """

    def __init__(
        self,
        session: WorkflowSession,
        catalogs: MicroscopeCatalogs,
        previews: PreviewPort,
        permission: PreviewPermission,
    ) -> None:
        self._session = session
        self._catalogs = catalogs
        self._previews = previews
        self._permission = permission

    def for_action_settings(self, action: Action | None = None) -> FormContext:
        """
        An action's settings form: workflow logger, actions, catalogs, previews.

        Args:
            action: The action whose settings are edited, for overlays drawn
                from its internal state and the marks on the fields resetting
                it; None if the form edits no action's settings.
        """
        return FormContext(
            log=self._session.text_logger(),
            actions=self._session,
            manufacturer_property_names=self._catalogs.manufacturer_property_names,
            pattern_types=self._catalogs.pattern_types,
            previews=self._previews,
            preview_permission=self._permission,
            state=(lambda: action.state) if action is not None else None,
            state_fields=action.state_fields() if action is not None else frozenset(),
            beam_properties=(
                action.stored_beam_properties if action is not None else None
            ),
        )

    def for_properties(self, action: Action) -> FormContext:
        """
        The properties dialog of an action: its logger and the catalogs.

        Args:
            action: The action whose props are edited.
        """
        return FormContext(
            log=action.ctx.text_logger,
            manufacturer_property_names=self._catalogs.manufacturer_property_names,
            pattern_types=self._catalogs.pattern_types,
        )

    def for_rules(self) -> FormContext:
        """A propagation rule's properties form: workflow logger and catalogs."""
        return self._logged_with_catalogs()

    def for_microscope_settings(self) -> FormContext:
        """The microscope settings dialog: workflow logger and catalogs."""
        return self._logged_with_catalogs()

    def _logged_with_catalogs(self) -> FormContext:
        return FormContext(
            log=self._session.text_logger(),
            manufacturer_property_names=self._catalogs.manufacturer_property_names,
            pattern_types=self._catalogs.pattern_types,
        )


@dataclass(frozen=True)
class Composition:
    """
    Every object of a composed main window, for the entry point and for tests.

    Attributes:
        window: The main window.
        session: The workflow session.
        tasks: The microscope task runner (its thread runs until shut down).
        run: The run controller (a run's thread exists while it runs).
        policy: The permissions.
        validity: Which settings forms show errors.
        readiness: Run readiness.
        status: Per-action indicators.
        commands: Every user intent.
        form_builder: Builds the forms.
        form_contexts: The environments of the forms.
    """

    window: MainWindow
    session: WorkflowSession
    tasks: MicroscopeTasks
    run: RunController
    policy: EditPolicy
    validity: FormValidity
    readiness: Readiness
    status: ActionStatus
    commands: AppCommands
    form_builder: FormBuilder
    form_contexts: FormContexts


def compose(
    seed: SessionSeed, user_config: UserConfig, files: WorkflowFiles
) -> Composition:
    """
    Construct and wire every service and view of the main window.

    Args:
        seed: The workflow, its directory and the microscope catalogs.
        user_config: The remembered profiles (shared with the start-up dialog).
        files: Filesystem operations (shared with the start-up dialog).

    Returns:
        Everything that was constructed.
    """
    session = WorkflowSession(seed.workflow, seed.workflow_dir, files)
    tasks = MicroscopeTasks()
    run = RunController(session, tasks)
    policy = EditPolicy(run, tasks)
    validity = FormValidity()
    catalogs = seed.catalogs
    audit = SettingsAudit(session, catalogs)
    readiness = Readiness(session, run, tasks, validity, audit)
    status = ActionStatus(session, run, tasks)
    previews = PreviewService(session, tasks)
    commands = AppCommands(
        session, run, tasks, previews, policy, readiness, user_config, files
    )
    form_builder = FormBuilder(default_registry())
    # AppCommands is the PreviewPort and EditPolicy the PreviewPermission
    form_contexts = FormContexts(
        session, catalogs, previews=commands, permission=policy
    )
    panel_factory = ActionPanelFactory(
        session, run, commands, policy, status, validity, form_builder, form_contexts
    )
    host = ActionPanelHost(panel_factory, session, validity)
    window = MainWindow(
        session,
        run,
        policy,
        readiness,
        status,
        commands,
        host,
        form_builder,
        form_contexts,
        files,
    )
    return Composition(
        window=window,
        session=session,
        tasks=tasks,
        run=run,
        policy=policy,
        validity=validity,
        readiness=readiness,
        status=status,
        commands=commands,
        form_builder=form_builder,
        form_contexts=form_contexts,
    )


def build_main_window(
    seed: SessionSeed, user_config: UserConfig, files: WorkflowFiles
) -> MainWindow:
    """
    Build the main window with all its services.

    The window keeps every service alive (through the objects it holds).

    Args:
        seed: The workflow, its directory and the microscope catalogs.
        user_config: The remembered profiles (shared with the start-up dialog).
        files: Filesystem operations (shared with the start-up dialog).

    Returns:
        The main window, not yet shown.
    """
    return compose(seed, user_config, files).window

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The panel showing one action: header with tools, settings form, sections, propagation rules."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox, QScrollArea, QVBoxLayout, QWidget

from fibsem_maestro.ui.actions_ui.spec import (
    SectionContext,
    ToolContext,
    ui_spec_for,
)
from fibsem_maestro.ui.views.action_panel.header import (
    ActionHeader,
    ActionHeaderViewData,
    action_header_view,
)
from fibsem_maestro.ui.views.action_panel.propagations import PropagationsEditor
from fibsem_maestro.ui.widgets.form_style import FORM_STYLESHEET

if TYPE_CHECKING:
    from collections.abc import Callable

    from PyQt6.QtCore import pyqtBoundSignal

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.actions_ui.spec import ToolSpec
    from fibsem_maestro.ui.app.action_status import ActionStatus
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.edit_policy import EditPolicy, Permissions
    from fibsem_maestro.ui.app.form_validity import FormValidity
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.composition import FormContexts
    from fibsem_maestro.ui.forms.builder import Form, FormBuilder


class ActionPanel(QWidget):
    """
    A scrollable panel editing a single action.

    The settings form edits the live settings and reports every commit and its
    validity; the header's tools, the propagation rules and any extra sections
    act through `AppCommands`. Read-only states and enabled tools follow the
    `EditPolicy`.

    Args:
        action: The action shown.
        session: The workflow session, for queries.
        run: The run controller (for extra sections).
        commands: Every user intent.
        policy: The permissions.
        status: The per-action indicators.
        validity: Receives whether the settings form shows an error.
        form_builder: Builds the forms.
        form_contexts: The environments of the forms.
        parent: Parent widget.
    """

    def __init__(
        self,
        action: Action,
        session: WorkflowSession,
        run: RunController,
        commands: AppCommands,
        policy: EditPolicy,
        status: ActionStatus,
        validity: FormValidity,
        form_builder: FormBuilder,
        form_contexts: FormContexts,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._action = action
        self._session = session
        self._run = run
        self._commands = commands
        self._policy = policy
        self._status = status
        self._validity = validity
        self._form_builder = form_builder
        self._form_contexts = form_contexts

        ui_spec = ui_spec_for(action)
        self._tools: dict[str, ToolSpec] = {tool.key: tool for tool in ui_spec.tools}

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(FORM_STYLESHEET)
        outer_layout.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        self._header = ActionHeader(ui_spec.tools, self._header_data(), parent=self)
        self._header.tool_clicked.connect(self._on_tool_clicked)
        self._header.reset_clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._header.frame)

        # settings; overlays may be drawn from the action's internal state
        settings_context = form_contexts.for_action_settings(action)
        self._form: Form = form_builder.build_live(
            action.settings, settings_context, on_commit=self._on_settings_committed
        )
        layout.addWidget(self._form.widget)

        section_context = SectionContext(
            action=action,
            commands=commands,
            session=session,
            run=run,
            status=status,
            policy=policy,
            form_builder=form_builder,
            form_context=settings_context,
        )
        for section in ui_spec.sections:
            if section.position == "after_settings":
                layout.addWidget(section.build(section_context))

        # propagations
        self._propagations: PropagationsEditor | None = None
        if action.PROPAGATION_RULES_ALLOWED:
            self._propagations = PropagationsEditor(
                action=action,
                session=session,
                commands=commands,
                form_builder=form_builder,
                form_context=form_contexts.for_rules(),
            )
            layout.addWidget(self._propagations)

        for section in ui_spec.sections:
            if section.position == "after_propagations":
                layout.addWidget(section.build(section_context))

        layout.addStretch()

        # a fresh form may already show an error
        validity.set_has_errors(action, self._form.has_errors())
        self._on_permissions_changed(policy.permissions)

        # the header also follows the slice (status, reset) and the run state,
        # since the props file of the current slice decides Edit and Apply, and
        # the state decides Reset state and the overlays drawn from it
        for signal, slot in self._service_connections():
            signal.connect(slot)

    @property
    def action(self) -> Action:
        """The action shown."""
        return self._action

    @property
    def form(self) -> Form:
        """The settings form."""
        return self._form

    @property
    def header(self) -> ActionHeader:
        """The header with the tool buttons."""
        return self._header

    @property
    def propagations(self) -> PropagationsEditor | None:
        """The propagation rules editor; None if the action takes no part in propagation."""
        return self._propagations

    def dispose(self) -> None:
        """
        Detach the panel from the services and make its settings form inert.

        Called when the panel is discarded: between this and its deferred
        deletion, no service change reaches the panel and nothing it shows
        writes to the action or reports validity. Idempotent.
        """
        self._form.dispose()
        for signal, slot in self._service_connections():
            # already disconnected (a second dispose)
            with contextlib.suppress(TypeError):
                signal.disconnect(slot)

    def _service_connections(
        self,
    ) -> tuple[tuple[pyqtBoundSignal, Callable[..., None]], ...]:
        """The (signal, bound method) pairs connecting this panel to long-lived objects."""
        return (
            (self._form.validity_changed, self._on_validity_changed),
            (self._policy.changed, self._on_permissions_changed),
            (self._session.action_renamed, self._on_action_event),
            (self._session.action_settings_changed, self._on_action_event),
            # dynamic defaults may be computed from any action's settings, and
            # from stored properties, also of another action (a refocus uses
            # its imaging's), which change with Prepare, Edit and every run
            (self._session.action_settings_changed, self._refresh_defaults),
            (self._session.action_props_changed, self._refresh_defaults),
            (self._run.action_finished, self._refresh_defaults),
            (self._session.structure_changed, self._form.refresh_defaults),
            (self._session.action_props_changed, self._on_action_event),
            (self._session.action_state_changed, self._on_action_event),
            (self._run.action_finished, self._on_action_event),
            (self._session.workflow_reset, self._on_workflow_reset),
            (self._run.state_changed, self._render_header),
            (self._status.changed, self._render_header),
        )

    def _header_data(self) -> ActionHeaderViewData:
        return action_header_view(
            self._action,
            self._policy.permissions,
            self._action.has_stored_properties(),
            tuple(self._tools.values()),
        )

    def _render_header(self, *_: object) -> None:
        self._header.render_data(self._header_data())

    def _on_action_event(self, action: object, *_: object) -> None:
        if action is self._action:
            self._render_header()
            # its state may have changed: it ran, was reset, or its settings or
            # props changed what it keeps
            self._form.refresh_overlays()

    def _on_workflow_reset(self) -> None:
        self._render_header()
        self._form.refresh_overlays()
        self._form.refresh_defaults()

    def _refresh_defaults(self, *_: object) -> None:
        self._form.refresh_defaults()

    def _on_reset_clicked(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset state",
            f"Reset the internal state of '{self._action.name}'?\n\n"
            "This discards the current progress of the action. The settings and the stored "
            "properties are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._commands.reset_action_state(self._action)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not store the reset state: {e}")

    def _on_permissions_changed(self, permissions: Permissions) -> None:
        # danger mode unlocks the settings and the tools usable in danger mode;
        # propagations and the other tools follow the state
        self._form.set_read_only(not permissions.settings_editable)
        if self._propagations is not None:
            self._propagations.set_read_only(not permissions.rules_editable)
        self._render_header()

    def _on_settings_committed(self, path: str) -> None:
        _ = path
        self._commands.settings_committed(self._action)

    def _on_validity_changed(self, has_errors: bool) -> None:
        self._validity.set_has_errors(self._action, has_errors)

    def _on_tool_clicked(self, key: str) -> None:
        tool = self._tools[key]
        tool.invoke(
            ToolContext(
                action=self._action,
                commands=self._commands,
                session=self._session,
                form_builder=self._form_builder,
                form_context=self._form_contexts.for_properties(self._action),
                parent=self,
                refresh=self._render_header,
                policy=self._policy,
            )
        )

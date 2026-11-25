# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The stack showing the selected action's panel, keeping a panel per action shown so far."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QStackedWidget, QWidget

from fibsem_maestro.ui.views.action_panel.building_page import BuildingPage
from fibsem_maestro.ui.views.action_panel.panel import ActionPanel
from fibsem_maestro.ui.widgets.busy import while_blocked

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.action_status import ActionStatus
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.edit_policy import EditPolicy
    from fibsem_maestro.ui.app.form_validity import FormValidity
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.composition import FormContexts
    from fibsem_maestro.ui.forms.builder import FormBuilder


class ActionPanelFactory:
    """
    Creates the panel of an action, with everything a panel needs.

    Args:
        session: The workflow session.
        run: The run controller.
        commands: Every user intent.
        policy: The permissions.
        status: The per-action indicators.
        validity: Which settings forms show errors.
        form_builder: Builds the forms.
        form_contexts: The environments of the forms.
    """

    def __init__(
        self,
        session: WorkflowSession,
        run: RunController,
        commands: AppCommands,
        policy: EditPolicy,
        status: ActionStatus,
        validity: FormValidity,
        form_builder: FormBuilder,
        form_contexts: FormContexts,
    ) -> None:
        self._session = session
        self._run = run
        self._commands = commands
        self._policy = policy
        self._status = status
        self._validity = validity
        self._form_builder = form_builder
        self._form_contexts = form_contexts

    def create(self, action: Action) -> ActionPanel:
        """
        Build the panel of an action.

        Args:
            action: The action.

        Returns:
            The new panel.
        """
        return ActionPanel(
            action=action,
            session=self._session,
            run=self._run,
            commands=self._commands,
            policy=self._policy,
            status=self._status,
            validity=self._validity,
            form_builder=self._form_builder,
            form_contexts=self._form_contexts,
        )


class ActionPanelHost(QStackedWidget):
    """
    Shows the panel of the selected action, or an empty widget.

    Panels are created the first time their action is shown and kept, so that
    collapsed groups, expanded area viewers and loaded images survive switching
    between actions. Once creating a panel takes `BUSY_DELAY_MS`, a page names
    its action and the cursor shows that the application is busy. A panel is
    discarded when its action is removed, and all panels when the workflow is
    replaced.

    Args:
        factory: Creates panels.
        session: The workflow session (removals, replacement).
        validity: Forgets the validity of discarded panels' forms.
        parent: Parent widget.
    """

    def __init__(
        self,
        factory: ActionPanelFactory,
        session: WorkflowSession,
        validity: FormValidity,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._factory = factory
        self._validity = validity
        self._panels: dict[Action, ActionPanel] = {}

        self._empty_panel = QWidget()
        self.addWidget(self._empty_panel)
        self.setCurrentWidget(self._empty_panel)

        self._building_page = BuildingPage()
        self.addWidget(self._building_page)

        session.action_removed.connect(self._on_action_removed)
        session.workflow_replaced.connect(self._on_workflow_replaced)

    def show_action(self, action: Action | None) -> None:
        """
        Show the panel of an action, creating it on first use.

        Args:
            action: The action, or None for the empty panel.
        """
        if action is None:
            self.setCurrentWidget(self._empty_panel)
            return

        panel = self._panels.get(action)
        if panel is None:
            panel = self._create_panel(action)
            self._panels[action] = panel
            self.addWidget(panel)
        self.setCurrentWidget(panel)

    @property
    def building_page(self) -> BuildingPage:
        """The page shown while a panel is created."""
        return self._building_page

    def _create_panel(self, action: Action) -> ActionPanel:
        """
        Create the panel of an action, showing the building page if it takes long.

        Args:
            action: The action.

        Returns:
            The new panel, not yet added to the stack.
        """
        previous = self.currentWidget()
        self._building_page.set_action_name(action.name)
        try:
            with while_blocked(self._show_building_frame, _restore_cursor):
                return self._factory.create(action)
        except Exception:
            self.setCurrentWidget(previous)
            raise

    def _show_building_frame(self, first: bool) -> None:
        """
        Show the building page, the first time with the wait cursor, else with its arc moved on.

        Args:
            first: Whether this is the first frame.
        """
        if first:
            self.setCurrentWidget(self._building_page)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            self._building_page.advance()
        self._building_page.repaint()

    def panel(self, action: Action) -> ActionPanel | None:
        """
        The cached panel of an action.

        Args:
            action: The action.

        Returns:
            The panel, or None if it was never shown or was discarded.
        """
        return self._panels.get(action)

    def _on_action_removed(self, action: Action) -> None:
        self._discard(action)

    def _on_workflow_replaced(self) -> None:
        for action in list(self._panels):
            self._discard(action)

    def _discard(self, action: Action) -> None:
        panel = self._panels.pop(action, None)
        if panel is None:
            return

        if self.currentWidget() is panel:
            self.setCurrentWidget(self._empty_panel)

        self.removeWidget(panel)
        panel.dispose()
        self._validity.forget(action)
        panel.deleteLater()


def _restore_cursor() -> None:
    """Undo the wait cursor of the building page."""
    QApplication.restoreOverrideCursor()

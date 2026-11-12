# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The "Propagations" group of an action panel: one editor per rule of the action."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

from fibsem_maestro.ui.forms.editors.collapsible_box import CollapsibleGroupBox
from fibsem_maestro.ui.views.action_panel.rule_editor import RuleEditor

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.forms.builder import FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext
    from fibsem_maestro.workflow.propagations import PropagationRule


class PropagationsEditor(CollapsibleGroupBox):
    """
    Editable list of the propagation rules of a single parent action.

    A section of the panel: a title with a divider line, then one card per rule.

    Shows one `RuleEditor` per rule of the action, matched to the rules by
    identity: when the rules change, editors of new rules are added, editors of
    deleted rules removed, and the others kept (with their collapse state).

    Args:
        action: The parent action.
        session: The workflow session (rules, actions), for queries only.
        commands: Every user intent.
        form_builder: Builds the rules' properties forms.
        form_context: The environment of the properties forms.
        parent: Parent widget.
    """

    def __init__(
        self,
        action: Action,
        session: WorkflowSession,
        commands: AppCommands,
        form_builder: FormBuilder,
        form_context: FormContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Propagations", parent, section=True)
        self._action = action
        self._session = session
        self._commands = commands
        self._form_builder = form_builder
        self._form_context = form_context
        self._rule_editors: list[RuleEditor] = []
        self._read_only = False

        # body container so the toggle can hide rules + button together
        body = QWidget()
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._set_body(body)

        self._add_btn = QPushButton("+ Add rule")
        self._add_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._add_btn.clicked.connect(self._on_add_rule)
        self._layout.addWidget(self._add_btn)

        self._reconcile()
        session.rules_changed.connect(self._reconcile)

    @property
    def rule_editors(self) -> tuple[RuleEditor, ...]:
        """The editors shown, in rule order."""
        return tuple(self._rule_editors)

    @property
    def add_button(self) -> QPushButton:
        """The "+ Add rule" button."""
        return self._add_btn

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing the rules.

        Args:
            read_only: If True, no rule can be added, edited or removed.
        """
        self._read_only = read_only
        for editor in self._rule_editors:
            editor.set_read_only(read_only)

        self._add_btn.setDisabled(read_only)

    def _reconcile(self) -> None:
        """Match the editors to the action's rules, by identity."""
        rules = self._session.rules_of(self._action)

        for editor in list(self._rule_editors):
            if not any(editor.rule is rule for rule in rules):
                self._rule_editors.remove(editor)
                self._layout.removeWidget(editor)
                editor.deleteLater()

        editors: list[RuleEditor] = []
        for rule in rules:
            editor = next((e for e in self._rule_editors if e.rule is rule), None)
            if editor is None:
                editor = self._new_editor(rule)
            editors.append(editor)

        # (re)insert in rule order, before the Add button
        for index, editor in enumerate(editors):
            if self._layout.indexOf(editor) != index:
                self._layout.insertWidget(index, editor)
        self._rule_editors = editors

    def _new_editor(self, rule: PropagationRule) -> RuleEditor:
        editor = RuleEditor(
            rule=rule,
            parent_action=self._action,
            session=self._session,
            commands=self._commands,
            form_builder=self._form_builder,
            form_context=self._form_context,
        )
        editor.set_read_only(self._read_only)
        editor.remove_requested.connect(self._on_remove_requested)
        return editor

    def _on_add_rule(self) -> None:
        self._commands.add_rule(self._action)

    def _on_remove_requested(self) -> None:
        sender = self.sender()
        for editor in self._rule_editors:
            if editor is sender:
                self._commands.remove_rule(editor.rule)
                return

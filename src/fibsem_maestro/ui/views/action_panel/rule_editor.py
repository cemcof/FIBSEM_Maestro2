# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The editor of one propagation rule: its dependents and the properties it propagates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.ui.app.errors import NotAllowedError
from fibsem_maestro.ui.forms.editors.checklist import make_checklist
from fibsem_maestro.ui.forms.editors.collapsible_box import (
    TITLE_TEXT_OFFSET,
    CollapsibleGroupBox,
)
from fibsem_maestro.ui.forms.editors.field_label import FieldLabel
from fibsem_maestro.ui.forms.editors.group import GroupWrapper
from fibsem_maestro.ui.forms.editors.object import LABEL_ROW_HEIGHT
from fibsem_maestro.ui.forms.issues import FieldValidationError

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.forms.builder import FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext
    from fibsem_maestro.workflow.propagations import PropagationRule


class RuleEditor(CollapsibleGroupBox):
    """
    Displays and allows editing of a single propagation rule.

    A card titled by the rule: its parent and dependents. Its title line ends
    with the "×" removing the rule; its body lists the dependents as a
    checklist and the propagated properties as a group.

    Every edit goes through `AppCommands`; the rule object is only read.

    Args:
        rule: The rule (one of the session's rules, edited in place by it).
        parent_action: The action whose panel holds the editor (the rule's parent).
        session: The workflow session, for queries only.
        commands: Every user intent.
        form_builder: Builds the properties form.
        form_context: The environment of the properties form.
        parent: Parent widget.
    """

    remove_requested = pyqtSignal()
    """The user clicked "×"."""

    def __init__(
        self,
        rule: PropagationRule,
        parent_action: Action,
        session: WorkflowSession,
        commands: AppCommands,
        form_builder: FormBuilder,
        form_context: FormContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("", parent)
        self._rule = rule
        self._parent_action = parent_action
        self._session = session
        self._commands = commands

        self.setProperty("dataclass_form", True)
        # the dependents may be many: the title is cut rather than widening the panel
        self.title_label.setMinimumWidth(1)

        self._remove_btn = QPushButton("×")
        self._remove_btn.setFixedSize(22, 22)
        self._remove_btn.clicked.connect(self.remove_requested)
        self.add_header_trailing(self._remove_btn)

        body = QWidget()
        body.setProperty("dataclass_form", True)
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self._set_body(body)

        self._dep_list = QListWidget()
        self._dep_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        make_checklist(self._dep_list)
        self._dep_list.itemSelectionChanged.connect(self._on_dependents_selected)
        self._populate_dep_list()

        dep_label = FieldLabel("dependents", self._dep_list)
        dep_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        # in line with the title of the properties card below, as in the forms
        dep_label.setContentsMargins(TITLE_TEXT_OFFSET, 0, 0, 0)
        dep_label.setMinimumHeight(LABEL_ROW_HEIGHT)

        grid.addWidget(dep_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        # at the left of its cell: the checklist's limited width must not
        # leave the spare width to the label column
        grid.addWidget(self._dep_list, 0, 1, Qt.AlignmentFlag.AlignLeft)

        self._props_form = form_builder.build_detached(PropertyNames, form_context)
        if rule.props_to_propagate is not None:
            self._props_form.set_value(rule.props_to_propagate.model_dump())

        # connected after the initial value, so showing it commits nothing
        self._props_form.changed.connect(self._on_properties_changed)
        self._props_widget = GroupWrapper(self._props_form.widget)
        self._props_widget.set_title_label(FieldLabel("properties", self._props_widget))
        grid.addWidget(self._props_widget, 1, 0, 1, 2)

        session.structure_changed.connect(self._populate_dep_list)
        session.action_renamed.connect(self._populate_dep_list)

    @property
    def rule(self) -> PropagationRule:
        """The rule this editor shows."""
        return self._rule

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing the rule.

        Args:
            read_only: If True, nothing can be edited or removed.
        """
        self._dep_list.setDisabled(read_only)
        self._props_widget.setDisabled(read_only)
        self._remove_btn.setDisabled(read_only)

    def _on_dependents_selected(self) -> None:
        names = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._dep_list.selectedItems()
        ]
        try:
            self._commands.set_rule_dependents(self._rule, names)
        except NotAllowedError:
            # rules became read-only meanwhile; show the rule as it is
            self._populate_dep_list()
            return
        self._show_title()

    def _on_properties_changed(self) -> None:
        try:
            names = self._props_form.value()
        except FieldValidationError:
            return
        try:
            self._commands.set_rule_properties(self._rule, names)
        except NotAllowedError:
            # rules became read-only meanwhile; the next change re-sends them all
            return

    def _show_title(self) -> None:
        """Title the card with the rule: its parent, and the dependents selected."""
        dependents = ", ".join(self._rule.dependent_names) or "(no dependents)"
        self.title_label.setText(f"{self._parent_action.name}  →  {dependents}")
        self.title_label.setToolTip(dependents)

    def _populate_dep_list(self, *_: object) -> None:
        """List the other actions that take part in propagation, selecting the rule's dependents."""
        self._show_title()
        self._dep_list.blockSignals(True)
        try:
            self._dep_list.clear()
            for action in self._session.actions:
                if (
                    action is self._parent_action
                    or not action.PROPAGATION_RULES_ALLOWED
                ):
                    continue

                item = QListWidgetItem(action.name)
                item.setData(Qt.ItemDataRole.UserRole, action.name)
                self._dep_list.addItem(item)
                if action.name in self._rule.dependent_names:
                    item.setSelected(True)
        finally:
            self._dep_list.blockSignals(False)

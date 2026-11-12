# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""One item of the action list: indicators, name (renamable inline), badge and subtitle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem_maestro.ui.schema.labels import class_name_to_label
from fibsem_maestro.ui.views.action_list.indicators import ArrowWidget, CheckWidget

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PyQt6.QtGui import QMouseEvent

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.action_status import ActionIndicator


@dataclass(frozen=True)
class ActionItemViewData:
    """
    What an action list item shows.

    Attributes:
        name: The action's name.
        subtitle: "<type label>  ·  <beam or —>".
        badge_text: "→<number of dependents>", or None to hide the badge.
        badge_tooltip: "Propagates to: <dependents>".
        name_tooltip: "slice number: <the action's slice>".
        prepared: Whether the check mark is drawn.
        next_to_run: Whether the arrow is drawn.
    """

    name: str
    subtitle: str
    badge_text: str | None
    badge_tooltip: str
    name_tooltip: str
    prepared: bool
    next_to_run: bool


def action_item_view(
    action: Action, indicator: ActionIndicator, dependents: Sequence[str]
) -> ActionItemViewData:
    """
    Decide what an action list item shows.

    Args:
        action: The action.
        indicator: Its check mark, arrow and slice.
        dependents: Unique names of the actions it propagates to.

    Returns:
        The view data.
    """
    beam = action.beam_type if action.beam_type is not None else "—"
    return ActionItemViewData(
        name=action.name,
        subtitle=f"{class_name_to_label(type(action).__name__)}  ·  {beam}",
        badge_text=f"→{len(dependents)}" if dependents else None,
        badge_tooltip=f"Propagates to: {', '.join(dependents)}" if dependents else "",
        name_tooltip=f"slice number: {indicator.slice}",
        prepared=indicator.prepared,
        next_to_run=indicator.next_to_run,
    )


class DoubleClickLabel(QLabel):
    """A label emitting `double_clicked` on a double click."""

    double_clicked = pyqtSignal()

    def mouseDoubleClickEvent(self, a0: QMouseEvent | None) -> None:
        """Emit `double_clicked`."""
        _ = a0
        self.double_clicked.emit()


class ActionListItem(QWidget):
    """
    Widget displayed inside each list item.

    Shows the indicators, the action name, the propagation badge and a subtitle
    with the type and beam. The name is editable by double-clicking, while
    renaming is allowed. The item is updated only through `render_data`.

    Args:
        data: What to show initially.
        parent: Parent widget.
    """

    rename_requested = pyqtSignal(str)
    """The inline editor finished with a changed name (stripped of whitespace)."""

    def __init__(self, data: ActionItemViewData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renamable = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # state indicators: arrow on the next action to run, check when prepared
        self._arrow = ArrowWidget()
        layout.addWidget(self._arrow)

        self._check = CheckWidget()
        layout.addWidget(self._check)

        # vertical group: name row + subtitle
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        # name row: name label + propagation badge
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(6)

        # name label (double-click to edit)
        self._name_label = DoubleClickLabel(data.name)
        # transparent: qdarkstyle paints labels in the window colour, which shows as a
        # box on the selected row
        self._name_label.setStyleSheet("font-weight: bold; background: transparent;")
        self._name_label.double_clicked.connect(self._start_name_edit)
        name_row.addWidget(self._name_label)

        # inline name editor (hidden by default)
        self._name_edit = QLineEdit(data.name)
        self._name_edit.hide()
        self._name_edit.returnPressed.connect(self._finish_name_edit)
        self._name_edit.editingFinished.connect(self._finish_name_edit)
        name_row.addWidget(self._name_edit)

        # propagation badge
        self._badge = QLabel()
        self._badge.setStyleSheet(
            "color: #5a9fd4; font-size: 11px; background: transparent;"
        )
        name_row.addWidget(self._badge)

        name_row.addStretch()
        text_layout.addLayout(name_row)

        # type + beam subtitle
        self._subtitle_label = QLabel(data.subtitle)
        self._subtitle_label.setStyleSheet(
            "color: #888888; font-size: 10px; background: transparent;"
        )
        text_layout.addWidget(self._subtitle_label)

        layout.addLayout(text_layout)

        self.render_data(data)

    @property
    def name_label(self) -> QLabel:
        """The label showing the name; double-click it to rename."""
        return self._name_label

    @property
    def name_edit(self) -> QLineEdit:
        """The inline name editor (hidden unless renaming)."""
        return self._name_edit

    @property
    def badge(self) -> QLabel:
        """The propagation badge."""
        return self._badge

    @property
    def subtitle_label(self) -> QLabel:
        """The label showing type and beam."""
        return self._subtitle_label

    @property
    def check_mark(self) -> CheckWidget:
        """The prepared check mark."""
        return self._check

    @property
    def arrow(self) -> ArrowWidget:
        """The next-to-run arrow."""
        return self._arrow

    def render_data(self, data: ActionItemViewData) -> None:
        """
        Show new view data.

        Args:
            data: What to show.
        """
        self._name_label.setText(data.name)
        self._name_label.setToolTip(data.name_tooltip)
        self._subtitle_label.setText(data.subtitle)

        if data.badge_text is not None:
            self._badge.setText(data.badge_text)
            self._badge.setToolTip(data.badge_tooltip)
            self._badge.show()
        else:
            self._badge.hide()

        self._arrow.set_on(data.next_to_run)
        self._check.set_on(data.prepared)

    def set_renamable(self, renamable: bool) -> None:
        """
        Allow or forbid starting the inline name editor by double-clicking.

        Args:
            renamable: True to allow renaming.
        """
        self._renamable = renamable

    def reset_name_edit(self) -> None:
        """Put the shown name back into the inline editor, e.g. after a refused rename."""
        self._name_edit.setText(self._name_label.text())

    def _start_name_edit(self) -> None:
        if not self._renamable:
            return
        self._name_label.hide()
        self._name_edit.setText(self._name_label.text())
        self._name_edit.show()
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_name_edit(self) -> None:
        new_name = self._name_edit.text().strip()
        old_name = self._name_label.text()

        # a refused rename resets the editor before its message box takes the
        # focus, so the focus loss that follows finds the name unchanged
        if new_name != old_name:
            self.rename_requested.emit(new_name)

        self._name_edit.hide()
        self._name_label.show()

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The page shown in place of an action's panel while the panel is built."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from fibsem_maestro.ui.forms.style import MUTED_TEXT_COLOR
from fibsem_maestro.ui.widgets.busy import BUSY_STEP_DEG, paint_busy_arc

if TYPE_CHECKING:
    from PyQt6.QtGui import QPaintEvent

_MARK_SIZE = 22
"""Edge of the busy mark (px)."""

_TRACK_COLOR = "#3a4a5a"
"""Colour of the circle the busy mark's arc lies on."""


class _BusyMark(QWidget):
    """A circle with the busy arc on it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(_MARK_SIZE, _MARK_SIZE)
        # start of the arc, in degrees clockwise from 3 o'clock
        self._angle = 270.0

    @property
    def angle(self) -> float:
        """Where the arc starts, in degrees clockwise from 3 o'clock."""
        return self._angle

    def advance(self) -> None:
        """Move the arc on by one frame."""
        self._angle = (self._angle + BUSY_STEP_DEG) % 360.0

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Paint the circle and the arc."""
        _ = a0
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
            painter.setPen(QPen(QColor(_TRACK_COLOR), 2))
            painter.drawEllipse(rect)
            paint_busy_arc(painter, rect, self._angle)
        finally:
            painter.end()


class BuildingPage(QWidget):
    """
    Says which action's panel is being built, centred in the panel area.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addStretch(2)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch()
        self._mark = _BusyMark()
        row.addWidget(self._mark)
        self._label = QLabel()
        font = self._label.font()
        font.setPointSize(11)
        self._label.setFont(font)
        self._label.setStyleSheet(
            f"color: {MUTED_TEXT_COLOR}; background: transparent;"
        )
        row.addWidget(self._label)
        row.addStretch()
        layout.addLayout(row)

        layout.addStretch(3)

    @property
    def arc_angle(self) -> float:
        """Where the busy mark's arc starts, in degrees clockwise from 3 o'clock."""
        return self._mark.angle

    def advance(self) -> None:
        """Move the busy mark's arc on by one frame."""
        self._mark.advance()

    @property
    def label(self) -> QLabel:
        """The label naming the action."""
        return self._label

    def set_action_name(self, name: str) -> None:
        """
        Name the action whose panel is being built.

        Args:
            name: The action's name.
        """
        self._label.setText(f"Building the settings of '{name}'…")

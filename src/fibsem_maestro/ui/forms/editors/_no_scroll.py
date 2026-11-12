# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Spin boxes and combo boxes that ignore the mouse wheel; spin boxes with their text inset."""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
)

if TYPE_CHECKING:
    from PyQt6.QtGui import QWheelEvent

SPIN_BOX_TEXT_INSET = 5
"""
Space added before a spin box's number (px).

Without it the number starts about 2 px from the frame, while the text of
line edits and combo boxes starts about 7 px from theirs. A style sheet's
padding does not reach the spin box's inner line edit, so it is set here.
"""


def _inset_text(spin_box: QAbstractSpinBox) -> None:
    """
    Move a spin box's text away from its left edge, as in the other inputs.

    Args:
        spin_box: The spin box.
    """
    if (line_edit := spin_box.lineEdit()) is not None:
        line_edit.setTextMargins(SPIN_BOX_TEXT_INSET, 0, 0, 0)


class NoScrollSpinBox(QSpinBox):
    """
    A QSpinBox that ignores mouse wheel events, with its number inset from the edge.

    Args:
        parent: The parent widget, if any.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _inset_text(self)

    def wheelEvent(self, e: QWheelEvent | None) -> None:
        """Ignore the wheel event so it propagates to the parent widget."""
        if e is not None:
            e.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """
    A QDoubleSpinBox that ignores mouse wheel events, with its number inset from the edge.

    Args:
        parent: The parent widget, if any.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _inset_text(self)

    def wheelEvent(self, e: QWheelEvent | None) -> None:
        """Ignore the wheel event so it propagates to the parent widget."""
        if e is not None:
            e.ignore()

    def textFromValue(self, v: float) -> str:
        """
        Format a value in the widget's locale, without trailing zeros beyond the first decimal.

        Args:
            v: The value to display.

        Returns:
            The value with at least one and at most `decimals()` decimals
            (none if `decimals()` is 0).
        """
        locale = self.locale()
        text = locale.toString(v, "f", self.decimals())
        if not self.isGroupSeparatorShown():
            text = text.replace(locale.groupSeparator(), "")

        point = locale.decimalPoint()
        if point in text:
            text = text.rstrip("0")
            if text.endswith(point):
                text += "0"

        # a negative value that rounds to zero would otherwise keep its sign
        value, ok = locale.toDouble(text)
        if ok and value == 0.0:
            text = text.lstrip("-")
        return text


class NoScrollComboBox(QComboBox):
    """A QComboBox that ignores mouse wheel events."""

    def wheelEvent(self, e: QWheelEvent | None) -> None:
        """Ignore the wheel event so it propagates to the parent widget."""
        if e is not None:
            e.ignore()

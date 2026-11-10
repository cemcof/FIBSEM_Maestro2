# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Resetting fields with a dynamic default to follow it again.

Right-clicking such a field (or its label) offers "Reset to default (…)",
which makes the field follow its computed default again.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QAction, QContextMenuEvent
from PyQt6.QtWidgets import QMenu, QWidget

if TYPE_CHECKING:
    from fibsem_maestro.ui.forms.editors.dynamic_default import DynamicDefaultWidget
    from fibsem_maestro.ui.forms.scope import BuildContext
    from fibsem_maestro.ui.schema.field_info import FieldInfo

RESET_TEXT = "Reset to default"
"""Text of the menu entry resetting a field, followed by the default in parentheses."""

_MAX_SHOWN = 40
"""Longest default shown in the menu entry, in characters."""


def shown_default(value: Any, suffix: str | None = None) -> str:
    """
    A default as shown in the menu entry resetting to it.

    Args:
        value: The default.
        suffix: The field's unit, if any.

    Returns:
        A short text, e.g. "18.4687 nm", "up" or "none".
    """
    match value:
        case None:
            return "none"
        case bool():
            return "checked" if value else "unchecked"
        case Enum():
            text = str(value.value)
        case float():
            text = f"{value:.6g}"
        case str():
            text = f'"{value}"' if value else "empty"
        case list() | tuple():
            text = ", ".join(_item_text(v) for v in value) if value else "empty"
        case _:
            text = str(value)

    if len(text) > _MAX_SHOWN:
        text = text[: _MAX_SHOWN - 1] + "…"
    return (
        f"{text} {suffix}" if suffix and not isinstance(value, list | tuple) else text
    )


class DefaultReset(QObject):
    """
    The right-click menu of a field with a dynamic default.

    Right-clicking the editor (or any part of it) or the label opens a menu
    with "Reset to default (…)", enabled while the field holds a value set by
    the user and the form is editable. Resetting makes the field follow its
    default again, which commits None like an edit.

    Args:
        fi: The field.
        editor: The field's editor.
        label: The field's label.
        context: The form's build; its `read_only` disables resetting.
    """

    def __init__(
        self,
        fi: FieldInfo,
        editor: DynamicDefaultWidget[Any],
        label: QWidget | None,
        context: BuildContext,
    ) -> None:
        super().__init__(editor)
        self._fi = fi
        self._editor = editor
        self._context = context

        for target in (editor, *editor.findChildren(QWidget), label):
            if target is not None:
                target.installEventFilter(self)

    def reset(self) -> None:
        """Follow the default again."""
        self._editor.follow_default()

    def menu_for(self, widget: QWidget) -> QMenu:
        """
        Build the right-click menu of the field, as opened on `widget`.

        Args:
            widget: The widget clicked: the editor, one of its parts, or the label.

        Returns:
            The menu, holding the reset entry.
        """
        menu = QMenu(widget)
        action = QAction(f"{RESET_TEXT} ({self._default_text()})", menu)
        action.setEnabled(not self._editor.following and not self._context.read_only)
        action.triggered.connect(self.reset)
        menu.addAction(action)
        return menu

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """
        Open the field's menu on a right-click on the editor or the label.

        Args:
            a0: The watched widget.
            a1: The event.

        Returns:
            True if the event opened the menu (and is consumed).
        """
        if (
            isinstance(a1, QContextMenuEvent)
            and a1.type() == QEvent.Type.ContextMenu
            and isinstance(a0, QWidget)
            and not self._context.disposed
        ):
            menu = self.menu_for(a0)
            menu.exec(a1.globalPos())
            menu.deleteLater()
            return True
        return super().eventFilter(a0, a1)

    def _default_text(self) -> str:
        """The default as shown in the menu entry."""
        value = self._editor.default()
        if value is None:
            return "cannot be computed"
        return shown_default(value, self._fi.unit.suffix if self._fi.unit else None)


def _item_text(value: Any) -> str:
    """An element of a shown list: names unquoted, everything else as shown alone."""
    return value if isinstance(value, str) else shown_default(value)

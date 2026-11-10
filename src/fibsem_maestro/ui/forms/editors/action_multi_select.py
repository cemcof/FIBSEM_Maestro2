# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Editor selecting any number of the workflow's actions by name."""

import contextlib
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem

from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.issues import FieldValidationError
from fibsem_maestro.ui.schema.choices import (
    accepts_action_type,
    unavailable_action_message,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from PyQt6.QtWidgets import QWidget

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.ports import ActionCatalog

_UNAVAILABLE_SUFFIX = " (not available)"
"""Appended to the row of a selected name no offered action carries."""

type _Entry = Action | str
"""What a row stands for: an offered action, or a selected name no offered action carries."""


class ActionMultiSelectWidget(QListWidget, FieldEditor[list[str]]):
    """
    A list for selecting any number of actions by name from a filtered list.

    The list follows the catalog: it is rebuilt when actions are added,
    removed or reordered, and a row is relabelled when its action is renamed.

    A selected name that no offered action carries (e.g. of a removed action)
    stays listed and selected, marked as not available, and the editor reads
    as invalid until the user deselects it. So the form never drops a link
    silently, and shows it as the settings audit reports it.

    The value lists the offered actions in workflow order, then the names
    that are not available.

    Args:
        actions: The workflow's actions.
        type_filter: Action types to include; actions of other types are omitted.
        type_exclude: Registry keys of action types omitted even though they
            pass `type_filter`.
        default: Names of the actions to select initially, if any.
        parent: The parent widget, if any.
    """

    def __init__(
        self,
        actions: ActionCatalog,
        type_filter: Sequence[type[Action]],
        type_exclude: Sequence[str] = (),
        default: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        FieldEditor.__init__(self)
        self._catalog = actions
        self._type_filter = tuple(type_filter)
        self._type_exclude = tuple(type_exclude)
        # one per row, in row order
        self._entries: list[_Entry] = []

        self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.setStyleSheet("""
            QListWidget::item:selected {
                background: #36678f;
                color: #ffffff;
            }
            QListWidget::item:selected:hover {
                background: #36678f;
            }
        """)

        self._populate(actions.actions, default or [])
        self.itemSelectionChanged.connect(self._emit)

        # bound methods: Qt drops these connections when this widget is destroyed
        actions.structure_changed.connect(self._on_structure_changed)
        actions.action_renamed.connect(self._on_action_renamed)

    def detach(self) -> None:
        """
        Stop following the catalog. Idempotent; safe after the widget was destroyed.

        A discarded form's widget lives on until its deferred deletion, and
        must not repopulate (and so emit) on catalog changes meanwhile.
        """
        for signal, slot in (
            (self._catalog.structure_changed, self._on_structure_changed),
            (self._catalog.action_renamed, self._on_action_renamed),
        ):
            # not connected: detached before, or Qt dropped it on destruction
            with contextlib.suppress(TypeError):
                signal.disconnect(slot)

    def _populate(self, actions: Sequence[Action], names: Iterable[str]) -> None:
        """
        List the actions passing the type filter, then the names none of them carries.

        Selects the rows of `names`. Emits nothing.

        Args:
            actions: The actions to offer from.
            names: The names to select.
        """
        wanted = dict.fromkeys(names)
        offered = [
            a
            for a in actions
            if accepts_action_type(type(a), self._type_filter, self._type_exclude)
        ]
        offered_names = {a.name for a in offered}

        with QSignalBlocker(self):
            self.clear()
            self._entries = []
            for action in offered:
                self._add_row(action, action.name in wanted)
            for name in wanted:
                if name not in offered_names:
                    self._add_row(name, True)

    def _add_row(self, entry: _Entry, selected: bool) -> None:
        if isinstance(entry, str):
            item = QListWidgetItem(entry + _UNAVAILABLE_SUFFIX)
            item.setToolTip(unavailable_action_message(entry))
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
        else:
            item = QListWidgetItem(entry.name)

        self.addItem(item)
        self._entries.append(entry)
        item.setSelected(selected)

    def _selection(self) -> dict[str, _Entry]:
        """The selected rows' entries, by name, in row order."""
        selection: dict[str, _Entry] = {}
        for row, entry in enumerate(self._entries):
            item = self.item(row)
            if item is not None and item.isSelected():
                selection[_name_of(entry)] = entry
        return selection

    def _reselect(self, names: Iterable[str]) -> None:
        """
        Repopulate from the catalog with `names` selected; notify if the selection changed.

        It changed if a name was added or dropped, or a selected name now
        stands for another entry: an action that was removed, re-added or
        replaced by another of the same name (e.g. after an import), so the
        link must be re-validated and written.

        Args:
            names: The names to select.
        """
        before = self._selection()
        self._populate(self._catalog.actions, names)
        after = self._selection()

        if before.keys() != after.keys() or any(
            not _same_entry(before[name], after[name]) for name in before
        ):
            self._emit()

    def _on_structure_changed(self) -> None:
        """Repopulate after actions were added, removed, reordered or renamed."""
        self._reselect(self._selection())

    def _on_action_renamed(self, action: Action, old_name: str) -> None:
        """
        Show the new name of a renamed action, if it is listed.

        Args:
            action: The renamed action.
            old_name: Its previous name.
        """
        _ = old_name
        for row, entry in enumerate(self._entries):
            if entry is action and (item := self.item(row)) is not None:
                item.setText(action.name)

    def get_value(self) -> list[str]:
        """
        Return the names of the selected actions.

        Returns:
            The names, in row order.

        Raises:
            FieldValidationError: If a selected name is not available; one
                problem per such name.
        """
        selection = self._selection()
        if missing := [name for name, e in selection.items() if isinstance(e, str)]:
            raise FieldValidationError(
                [(None, unavailable_action_message(name)) for name in missing]
            )
        return list(selection)

    def set_value(self, value: list[str]) -> None:
        """
        Select the actions with the given names, and only those.

        Names that no offered action carries are listed as not available.

        Args:
            value: The names of the actions to select.
        """
        self._reselect(value)

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable user interaction with the widget.

        Args:
            read_only: If True, disable the widget to prevent changes.
        """
        self.setEnabled(not read_only)


def _name_of(entry: _Entry) -> str:
    return entry if isinstance(entry, str) else entry.name


def _same_entry(a: _Entry, b: _Entry) -> bool:
    # actions by identity, names not available by value
    return a is b or (isinstance(a, str) and a == b)

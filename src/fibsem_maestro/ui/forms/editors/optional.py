# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""A checkbox gating an editor: unchecked means None."""

from typing import TYPE_CHECKING, Any, TypeVar, cast

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPoint, QTimer
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.editors.collapsible_box import CollapsibleGroupBox

if TYPE_CHECKING:
    from fibsem_maestro.ui.forms.issues import FieldIssue

T = TypeVar("T")


class OptionalWidget(QWidget, FieldEditor[T | None]):
    """
    Checkbox that gates an inner widget: unchecked -> value is None.

    The inner widget is hidden while unchecked. A group (a card) is not
    hidden: its checkbox sits on its title line, before the title, and only
    its body hides, so the unchecked group still shows its name.

    Args:
        inner: The widget edited when the checkbox is checked.
        inline: If True, lay the checkbox and inner widget out horizontally instead of vertically.
        enabled_by_default: If True, start checked with the inner widget shown.
        parent: The parent widget, if any.
    """

    def __init__(
        self,
        inner: FieldEditor[T],
        inline: bool = False,
        enabled_by_default: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        FieldEditor.__init__(self)

        self._inner = cast("QWidget", inner)
        self._group = inner if isinstance(inner, CollapsibleGroupBox) else None

        layout = QHBoxLayout(self) if inline else QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(self)
        self._slot: QWidget | None = None
        if self._group is not None:
            # kept a direct child (and so found as this field's checkbox),
            # placed over the slot the card reserves before its title
            layout.addWidget(self._inner)
            self._slot = self._group.title_slot()
            # the slot moves within the card, and the card within this widget
            self._slot.installEventFilter(self)
            self._group.installEventFilter(self)
            self._checkbox.raise_()
        else:
            layout.addWidget(self._checkbox)
            layout.addWidget(self._inner)
            layout.addStretch()

        self._checkbox.stateChanged.connect(self._on_toggled)
        self._checkbox.setChecked(enabled_by_default)
        self._show_inner(enabled_by_default)

        # issues reported on the gated editor are shown on this field's row
        inner.on_issues_changed(self._emit_issues_changed)

    @property
    def group(self) -> CollapsibleGroupBox | None:
        """The gated card, if the inner widget is a group; its checkbox is on its title line."""
        return self._group

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """
        Keep the checkbox over the card's slot when the slot or the card moves.

        The checkbox is placed once the layout has settled: during a relayout
        the slot's position relative to this widget is not final yet.

        Args:
            a0: The watched object.
            a1: The event.

        Returns:
            False: the event is not consumed.
        """
        if (
            a0 is not None
            and (a0 is self._slot or a0 is self._group)
            and a1 is not None
            and a1.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show)
        ):
            QTimer.singleShot(0, self._place_checkbox)
        return super().eventFilter(a0, a1)

    def _place_checkbox(self) -> None:
        """Move the checkbox over the card's slot."""
        # also when called late, after this widget was discarded
        if self._slot is None or sip.isdeleted(self) or sip.isdeleted(self._slot):
            return
        hint = self._checkbox.sizeHint()
        self._checkbox.resize(hint)
        corner = self._slot.mapTo(self, QPoint(0, 0))
        self._checkbox.move(
            corner.x() + (self._slot.width() - hint.width()) // 2,
            corner.y() + (self._slot.height() - hint.height()) // 2,
        )
        self._checkbox.raise_()

    def _show_inner(self, shown: bool) -> None:
        """Show or hide the gated editor; of a group, only what is below its title."""
        if self._group is not None:
            self._group.set_active(shown)
        else:
            self._inner.setVisible(shown)

    def _on_toggled(self, _: int = 0) -> None:
        """Show or hide the inner widget to match the checkbox, then notify."""
        self._show_inner(self._checkbox.isChecked())
        self._emit()

    def get_value(self) -> T | None:
        """
        Return the inner value, or None when unchecked.

        Returns:
            The inner widget's value if the checkbox is checked, else None.
        """

        if not self._checkbox.isChecked():
            return None
        return cast("FieldEditor[T]", self._inner).get_value()

    def set_value(self, value: T | None) -> None:
        """
        Set the value, checking the box only for a non-None value.

        A None value unchecks the box and hides the inner widget; otherwise
        the box is checked, the inner widget is shown, and the value is
        applied to it.

        Args:
            value: The value to apply, or None to disable the field.
        """

        if value is None:
            self._checkbox.setChecked(False)
            self._show_inner(False)
        else:
            self._checkbox.setChecked(True)
            self._show_inner(True)
            cast("FieldEditor[T]", self._inner).set_value(value)

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable the checkbox and the inner widget.

        Args:
            read_only: If True, prevent toggling and edits.
        """
        self._checkbox.setEnabled(not read_only)
        cast("FieldEditor[T]", self._inner).set_read_only(read_only)

    def issues(self) -> list[FieldIssue]:
        """
        Return the issues of this field, including those reported on the gated editor.

        Returns:
            This wrapper's own issues followed by the inner widget's.
        """
        return [*super().issues(), *self.inner.issues()]

    def highlight_target(self) -> QWidget:
        """Highlight the gated inner editor, not the checkbox row."""
        return self._inner

    @property
    def inner(self) -> FieldEditor[T]:
        """The wrapped widget."""
        return cast("FieldEditor[T]", self._inner)

    def value_editor(self) -> FieldEditor[Any]:
        """Return the value editor of the gated editor."""
        return self.inner.value_editor()

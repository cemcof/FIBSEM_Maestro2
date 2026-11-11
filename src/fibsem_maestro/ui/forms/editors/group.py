# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""A collapsible group box around a nested object or union editor."""

from typing import TYPE_CHECKING, Any, TypeVar, cast

from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.editors.collapsible_box import CollapsibleGroupBox

if TYPE_CHECKING:
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QResizeEvent

T = TypeVar("T")


class GroupWrapper(CollapsibleGroupBox, FieldEditor[T]):
    """
    Plain (non-checkable) QGroupBox wrapping an ObjectWidget for required nested objects.

    The box never grows taller than its content needs (vertical policy
    `Maximum`), so a form given spare height does not stretch its frames.
    Its size hint is the height the content needs at the box's current
    width, word-wrapped labels included; otherwise the `Maximum` policy
    would cap the box below that height and squash the content.

    Args:
        inner: The nested object or discriminated-union widget to wrap.
        parent: The parent widget, if any.
    """

    def __init__(self, inner: FieldEditor[T], parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        FieldEditor.__init__(self)
        self._inner = inner

        layout = QVBoxLayout(self)
        layout.addWidget(cast("QWidget", inner))
        layout.addStretch()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _collapse_body(self) -> QWidget:
        return cast("QWidget", self._inner)

    def sizeHint(self) -> QSize:
        """
        Return the preferred size, tall enough for the content at the current width.

        Returns:
            The size hint of the box, its height the content's height for
            the box's current width; before the box is shown, for its
            preferred width.
        """
        hint = super().sizeHint()
        if self.hasHeightForWidth():
            width = self.width() if self.isVisible() else hint.width()
            hint.setHeight(max(hint.height(), self.heightForWidth(width)))
        return hint

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """
        Let the enclosing layout ask for the size hint again when the width changed.

        Args:
            a0: The resize event.
        """
        super().resizeEvent(a0)
        if a0 is not None and a0.size().width() != a0.oldSize().width():
            self.updateGeometry()

    def get_value(self) -> T:
        """
        Return the inner widget's value.

        Returns:
            The value held by the wrapped widget.
        """

        return self._inner.get_value()

    def set_value(self, value: T) -> None:
        """
        Set the inner widget's value.

        Args:
            value: The value to pass to the wrapped widget.
        """

        self._inner.set_value(value)

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing of the inner widget.

        Args:
            read_only: If True, make the wrapped widget read-only.
        """
        self._inner.set_read_only(read_only)

    @property
    def inner(self) -> FieldEditor[T]:
        """The wrapped widget."""
        return self._inner

    def value_editor(self) -> FieldEditor[Any]:
        """Return the value editor of the wrapped editor."""
        return self._inner.value_editor()

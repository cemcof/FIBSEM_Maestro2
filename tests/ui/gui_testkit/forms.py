# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Helpers for driving built settings forms, and stand-ins for the ports forms use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from fibsem_maestro.logging.text.memory import MemoryTextLogger
from fibsem_maestro.ui.forms.editors.object import ObjectWidget
from fibsem_maestro.ui.forms.editors.optional import OptionalWidget
from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
from fibsem_maestro.ui.ports import TaskHandle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.forms.editor import FieldEditor
    from fibsem_maestro.ui.ports import (
        ActionCatalog,
        PreviewPermission,
        PreviewRequest,
    )


TWidget = TypeVar("TWidget", bound=QWidget)


def editor_at(root: QWidget, path: str) -> FieldEditor[Any]:
    """
    The value editor at a dotted path, descending into selected union variants.

    Args:
        root: The form's root widget.
        path: Dotted field path from the root.

    Returns:
        The value editor (optional and group wrappers removed).
    """
    editor = cast("FieldEditor[Any]", root)
    for part in path.split("."):
        found = editor.value_editor().child_editors(part, all_variants=False)
        assert found, f"no field {part!r} in {path!r}"
        editor = found[0]
    return editor


def field_at(root: QWidget, path: str) -> FieldEditor[Any]:
    """
    The outermost editor of a field: the one placed in its row, wrappers included.

    Args:
        root: The form's root widget.
        path: Dotted field path from the root.

    Returns:
        The field's editor, e.g. its optional checkbox wrapper.
    """
    parent_path, _, name = path.rpartition(".")
    parent: Any = editor_at(root, parent_path) if parent_path else root
    if isinstance(parent, DiscriminatedUnionWidget):
        parent = parent.selected_variant_widget()
    assert isinstance(parent, ObjectWidget), f"{parent_path!r} is not an object"
    editor = parent.field_widget(name)
    assert editor is not None, f"no field {path!r}"
    return editor


def optional_checkbox(wrapper: FieldEditor[Any]) -> QCheckBox:
    """The checkbox of an optional field's wrapper (see `field_at`)."""
    assert isinstance(wrapper, OptionalWidget), "the field is not optional"
    return wrapper.checkbox


def messages(editor: FieldEditor[Any]) -> list[str]:
    """The messages of the issues an editor reports."""
    return [issue.message for issue in editor.issues()]


def children[TWidget: QWidget](
    widget: object, kind: type[TWidget], direct_only: bool = False
) -> list[TWidget]:
    """
    The widgets of a type inside a widget, in widget-tree order.

    Args:
        widget: Where to look: a widget, or a field editor (which is one).
        kind: The type of the widgets to find.
        direct_only: Only direct children.

    Returns:
        The widgets found.
    """
    assert isinstance(widget, QWidget), f"{widget!r} is not a widget"
    options = (
        Qt.FindChildOption.FindDirectChildrenOnly
        if direct_only
        else Qt.FindChildOption.FindChildrenRecursively
    )
    return [
        c for c in widget.findChildren(kind, options=options) if isinstance(c, kind)
    ]


def labels(widget: object) -> list[str]:
    """The texts of every label inside a widget (or field editor)."""
    return [label.text() for label in children(widget, QLabel)]


def button(widget: object, text: str) -> QPushButton:
    """
    The one push button with a given text inside a widget.

    Args:
        widget: Where to look: a widget, or a field editor.
        text: The button's text.

    Returns:
        The button.
    """
    found = [b for b in children(widget, QPushButton) if b.text() == text]
    assert len(found) == 1, f"{len(found)} buttons {text!r}"
    return found[0]


_NAVIGATION = frozenset({"▼ Expand", "▲ Collapse"})
"""Area-selector buttons that stay enabled in a read-only form: they change nothing."""


def editable_controls(root: object) -> list[QWidget]:
    """
    Every control inside `root` a user could change something with.

    Spin boxes and text fields count while they are enabled and not read-only;
    buttons, check boxes, combo boxes and lists while they are enabled.

    Args:
        root: The widget to search.

    Returns:
        The controls, in widget-tree order.
    """
    editable: list[QWidget] = []
    for widget in children(root, QWidget):
        match widget:
            case QAbstractSpinBox() | QLineEdit() | QPlainTextEdit():
                if widget.isEnabled() and not widget.isReadOnly():
                    editable.append(widget)
            case QPushButton() if widget.text() in _NAVIGATION:
                pass
            case QAbstractButton() | QComboBox() | QListWidget():
                if widget.isEnabled():
                    editable.append(widget)
    return editable


def combo_texts(combo: QComboBox) -> list[str]:
    """The entries of a combo box, in order."""
    return [combo.itemText(i) for i in range(combo.count())]


def recording_logger() -> MemoryTextLogger:
    """A text logger keeping its records in memory, e.g. for `FormContext.log`."""
    return MemoryTextLogger(lambda: 0, "test")


def warnings_of(log: MemoryTextLogger) -> list[str]:
    """The messages of the warnings a recording logger received, in order."""
    return [
        record.message
        for records in log.records.values()
        for record in records
        if record.level == "warning"
    ]


class FakeCatalog(QObject):
    """
    An `ActionCatalog` over a list of actions the test controls.

    Args:
        actions: The initial actions.
    """

    structure_changed = pyqtSignal()
    action_renamed = pyqtSignal(object, str)

    def __init__(self, actions: Sequence[Action] = ()) -> None:
        super().__init__()
        self._actions = list(actions)

    @property
    def actions(self) -> Sequence[Action]:
        """The actions, in order."""
        return tuple(self._actions)

    def port(self) -> ActionCatalog:
        """This catalog, typed as the port it implements."""
        # pyqtSignal class attributes do not match the protocol's properties for
        # the type checker; at run time they are bound signals
        return cast("ActionCatalog", self)

    def replace(self, actions: Sequence[Action]) -> None:
        """Replace the actions and announce it."""
        self._actions = list(actions)
        self.structure_changed.emit()

    def rename(self, action: Action, new_name: str) -> None:
        """Rename one of the actions and announce it."""
        old = action.name
        action.rename(new_name)
        self.action_renamed.emit(action, old)
        self.structure_changed.emit()


class FakePreviews:
    """A `PreviewPort` recording requests; the test resolves each handle itself."""

    def __init__(self) -> None:
        self.requests: list[PreviewRequest] = []
        self.handles: list[TaskHandle] = []

    def grab(self, request: PreviewRequest) -> TaskHandle:
        """Record the request and return a handle that has not finished."""
        self.requests.append(request)
        handle = TaskHandle()
        self.handles.append(handle)
        return handle


class FakePermission(QObject):
    """A `PreviewPermission` the test switches."""

    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.allowed = True

    def preview_allowed(self) -> bool:
        """Whether previews are allowed now."""
        return self.allowed

    def port(self) -> PreviewPermission:
        """This permission, typed as the port it implements."""
        return cast("PreviewPermission", self)

    def set_allowed(self, allowed: bool) -> None:
        """Allow or forbid previews and announce it."""
        self.allowed = allowed
        self.changed.emit()

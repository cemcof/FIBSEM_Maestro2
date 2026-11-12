# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Editor of an object's fields, laid out as a labelled grid."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from html import escape
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel, ValidationError
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWIDGETSIZE_MAX, QGridLayout, QLabel, QWidget

from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.editors.collapsible_box import (
    TITLE_TEXT_OFFSET,
    CollapsibleGroupBox,
)
from fibsem_maestro.ui.forms.editors.field_label import FieldLabel
from fibsem_maestro.ui.forms.editors.optional import OptionalWidget
from fibsem_maestro.ui.forms.issues import (
    FieldIssue,
    FieldValidationError,
    Severity,
)
from fibsem_maestro.ui.forms.style import CARD_INDENT

if TYPE_CHECKING:
    from PyQt6.QtGui import QShowEvent

T = TypeVar("T")

_ISSUE_COLORS = {Severity.ERROR: "#e05252", Severity.WARNING: "#e0a030"}
"""Text colour of a field's label and message, by the most severe issue."""

RESETS_STATE_MARK = "R"
"""Shown after the label of a field whose change resets the action's internal state."""

RESETS_STATE_NOTE = "Changing this resets the action's internal state."
"""Added to the tooltip of a field whose change resets the action's internal state."""

_RESETS_STATE_COLOR = "#9FCBF2"
"""Light blue on qdarkstyle's darkest accent: the mark informs, it is not a control."""

_RESETS_STATE_BACKGROUND = "#26486B"
"""qdarkstyle's first accent, behind the mark."""

RESETS_STATE_MARK_HTML = (
    f'<span style="color: {_RESETS_STATE_COLOR}; '
    f"background-color: {_RESETS_STATE_BACKGROUND}; "
    f'font-size: 7pt; font-weight: bold;">&nbsp;{RESETS_STATE_MARK}&nbsp;</span>'
)
"""The mark as appended to the label's rich text, as a small pill."""

LABEL_ROW_HEIGHT = 26
"""Height of a field's label (px): that of a one-line input, so the label sits level with its first line."""

_COLUMN_SPACING = 12
"""Space between the labels and the editors (px)."""


@dataclass(frozen=True)
class _FieldRow:
    """
    The widgets showing one field's issues.

    Attributes:
        label: The field's label, coloured while the field has issues.
        message: The line below the editor describing the issues; hidden while
            there are none.
        base_style: The label's style without issues (a group's title is bold).
        in_column: Whether the label is in the label column; a group's label
            is the title of its card instead.
    """

    label: FieldLabel
    message: QLabel
    base_style: str
    in_column: bool


def _model_to_dict(value: Any) -> dict[str, Any] | None:
    """
    Convert a pydantic model or dataclass instance to a shallow dict.

    Args:
        value: The instance.

    Returns:
        Field name to value, or None if `value` is neither.
    """
    # pydantic model
    if isinstance(value, BaseModel):
        return {name: getattr(value, name) for name in type(value).model_fields}
    # plain dataclass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    return None


def _issue_html(issue: FieldIssue) -> str:
    """
    Render one issue as a line of rich text in its severity's colour.

    Args:
        issue: The issue to render.

    Returns:
        HTML-escaped message text, with line breaks kept.
    """
    text = escape(issue.message).replace("\n", "<br>")
    color = _ISSUE_COLORS[issue.severity]
    return f'<span style="color: {color}; font-size: 11px;">{text}</span>'


class ObjectWidget(QWidget, FieldEditor[T]):
    """
    A form widget that edits an object's fields in a labelled grid.

    Every field occupies two grid rows: the label and editor, and below the
    editor a message line that is shown only while the field reports issues
    (see `FieldEditor.set_issue`). A field with issues also has its label
    coloured by the most severe one.

    A group (a nested object or union, optional or not) spans both columns:
    its label is the title of its card, on the card's first line.

    Labels stay on one line, level with the first line of their editor, and
    start where the titles of the cards beside them do (`TITLE_TEXT_OFFSET`),
    so an object's labels and titles form one column of text. The editors of
    a form start on one column, however deeply nested: the root object
    editor aligns the label columns of all object editors in it when it is
    shown (`align_label_columns`).

    Args:
        cls: The dataclass or model type to construct from the field values.
        parent: The parent widget, if any.
    """

    def __init__(self, cls: type[T], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        FieldEditor.__init__(self)

        # required to highlight selected box
        self.setProperty("dataclass_form", True)

        self._cls = cls
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(_COLUMN_SPACING)
        self._layout.setVerticalSpacing(4)
        self._layout.setColumnStretch(1, 1)
        self._fields: dict[str, FieldEditor[Any]] = {}
        self._rows: dict[str, _FieldRow] = {}
        self._row_count = 0

    def add_field(
        self,
        name: str,
        label: str,
        widget: FieldEditor[Any],
        description: str = "",
        resets_state: bool = False,
    ) -> None:
        """
        Add a labelled field row to the form.

        Args:
            name: The field key used when reading and writing values.
            label: The text shown in the field's label.
            widget: The editor widget for the field's value.
            description: Optional tooltip text for the label.
            resets_state: The field's change resets the action's internal
                state: its label is marked and its tooltip says so.
        """

        label_widget = FieldLabel(label, widget.highlight_target())
        label_widget.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        label_widget.setMinimumHeight(LABEL_ROW_HEIGHT)
        if resets_state:
            label_widget.setTextFormat(Qt.TextFormat.RichText)
            label_widget.setText(f"{escape(label)} {RESETS_STATE_MARK_HTML}")
            description = "\n\n".join(filter(None, (description, RESETS_STATE_NOTE)))
        if description:
            label_widget.setToolTip(description)

        message = QLabel()
        message.setWordWrap(True)
        message.setTextFormat(Qt.TextFormat.RichText)
        message.hide()

        group = _group_of(widget)
        if group is not None:
            # the label is the card's title; the card spans both columns
            group.set_title_label(label_widget)
            self._layout.addWidget(cast("QWidget", widget), self._row_count, 0, 1, 2)
            self._layout.addWidget(message, self._row_count + 1, 0, 1, 2)
        else:
            # in line with the titles of the cards beside it
            label_widget.setContentsMargins(TITLE_TEXT_OFFSET, 0, 0, 0)
            self._layout.addWidget(
                label_widget, self._row_count, 0, Qt.AlignmentFlag.AlignTop
            )
            editor = cast("QWidget", widget)
            # an editor of limited width sits at the left of its cell; placed
            # unaligned, it would cap the editor column, and the label column
            # would take the spare width and push the editors off the common column
            limited = editor.maximumWidth() < QWIDGETSIZE_MAX
            self._layout.addWidget(
                editor,
                self._row_count,
                1,
                Qt.AlignmentFlag.AlignLeft if limited else Qt.AlignmentFlag(0),
            )
            # a row holding only a hidden widget takes no space
            self._layout.addWidget(message, self._row_count + 1, 1)
        self._row_count += 2

        self._fields[name] = widget
        self._rows[name] = _FieldRow(
            label_widget,
            message,
            base_style=label_widget.styleSheet(),
            in_column=group is None,
        )
        widget.on_issues_changed(partial(self._show_issues, name))
        # issues the editor reported while it was built
        self._show_issues(name)

    def _show_issues(self, name: str) -> None:
        """
        Redraw one field's label colour and message line from its current issues.

        Args:
            name: The field key given to `add_field`.
        """
        issues = self._fields[name].issues()
        row = self._rows[name]

        if not issues:
            row.label.setStyleSheet(row.base_style)
            row.message.clear()
            row.message.hide()
            return

        # errors first, each line in its own severity's colour
        ordered = sorted(issues, key=lambda issue: issue.severity is not Severity.ERROR)
        row.message.setText("<br>".join(_issue_html(issue) for issue in ordered))
        row.label.setStyleSheet(
            f"{row.base_style} color: {_ISSUE_COLORS[ordered[0].severity]};"
        )
        row.message.show()

    def natural_label_width(self) -> int:
        """
        The width the labels of this object's field rows need on one line.

        Returns:
            The widest label's width (px); 0 without labelled rows (e.g. only groups).
        """
        return max(
            (
                row.label.sizeHint().width()
                for row in self._rows.values()
                if row.in_column
            ),
            default=0,
        )

    def set_label_column(self, width: int) -> None:
        """
        Make the label column at least this wide, so the editors start on a common column.

        Args:
            width: The minimum width of the label column (px).
        """
        self._layout.setColumnMinimumWidth(0, width)

    def showEvent(self, a0: QShowEvent | None) -> None:
        """
        Align the label columns of the whole form when its root is shown.

        Args:
            a0: The show event.
        """
        super().showEvent(a0)
        if not _has_object_ancestor(self):
            align_label_columns(self)

    def has_errors(self) -> bool:
        """
        Check whether any field of this object reports an error.

        Fields of nested objects are not included; they belong to their own
        `ObjectWidget`.

        Returns:
            True if a field's issues include one of severity `ERROR`.
        """
        return any(
            issue.severity is Severity.ERROR
            for widget in self._fields.values()
            for issue in widget.issues()
        )

    def child_widgets(self) -> list[FieldEditor[Any]]:
        """
        Return the field editor widgets in insertion order.

        Returns:
            A list of the field widgets.
        """

        return list(self._fields.values())

    def values_dict(self) -> dict[str, Any]:
        """
        Return the current field values keyed by field name.

        Every field is read, so the problems of all invalid fields are reported
        together.

        Returns:
            A mapping from field name to that field's current value.

        Raises:
            FieldValidationError: If any field's value cannot be built. Problems
                concerning a field as a whole are attributed to that field's
                widget.
        """
        values: dict[str, Any] = {}
        problems: list[tuple[FieldEditor[Any], str]] = []
        first_error: FieldValidationError | None = None

        for name, w in self._fields.items():
            try:
                values[name] = w.get_value()
            except FieldValidationError as e:
                problems.extend(e.attributed_to(w))
                first_error = first_error or e

        if first_error is not None:
            raise FieldValidationError(problems) from first_error
        return values

    def get_value(self) -> T:
        """
        Construct an instance of the underlying class from the current field values.

        Returns:
            A new instance of the underlying class built from the field values.

        Raises:
            FieldValidationError: If a field's value cannot be built, or the
                instance fails validation. Problems on a field are attributed
                to that field's widget; model-level problems are left to the
                field containing this object.
        """
        values = self.values_dict()
        try:
            return self._cls(**values)
        except ValidationError as e:
            raise FieldValidationError.from_validation_error(
                e, self.field_widget
            ) from e

    def set_value(self, value: T) -> None:
        """
        Populate the fields from an object or dict.

        Accepts a dict or a model/dataclass instance; the latter is
        converted to a dict first. Fields absent from the data are left
        unchanged.

        Args:
            value: The object or dict to read field values from.
        """

        data: dict | None = value if isinstance(value, dict) else _model_to_dict(value)
        if not isinstance(data, dict):
            return

        for name, w in self._fields.items():
            if name in data:
                w.set_value(data[name])

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing of all field widgets.

        Args:
            read_only: If True, make every field read-only.
        """
        for w in self._fields.values():
            w.set_read_only(read_only)

    def field_label(self, name: str) -> FieldLabel | None:
        """
        Return the label of one field.

        Args:
            name: The field key given to `add_field`.

        Returns:
            The field's label, or None if this form has no such field.
        """
        row = self._rows.get(name)
        return None if row is None else row.label

    def field_widget(self, name: str) -> FieldEditor[Any] | None:
        """
        Return the editor widget for one field.

        Args:
            name: The field key given to `add_field`.

        Returns:
            The field's widget, or None if this form has no such field.
        """
        return self._fields.get(name)

    def child_editors(self, name: str, all_variants: bool) -> list[FieldEditor[Any]]:
        """
        Return the value editor of the field `name`.

        Args:
            name: The field name.
            all_variants: Unused; an object has no variants.

        Returns:
            The field's value editor, or an empty list if this object has no such field.
        """
        _ = all_variants
        child = self._fields.get(name)
        return [] if child is None else [child.value_editor()]


def _group_of(widget: FieldEditor[Any]) -> CollapsibleGroupBox | None:
    """
    The card of a group field's editor, which shows the field's label as its title.

    Args:
        widget: The editor placed in the field's row.

    Returns:
        The editor itself if it is a card, the gated card of an optional
        group, or None for any other editor.
    """
    if isinstance(widget, CollapsibleGroupBox):
        return widget
    if isinstance(widget, OptionalWidget):
        return widget.group
    return None


def _has_object_ancestor(widget: QWidget) -> bool:
    """Whether the widget is inside another object editor (it is not the root of a form)."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, ObjectWidget):
            return True
        parent = parent.parentWidget()
    return False


def _indent(widget: QWidget, root: QWidget) -> int:
    """
    How much further right than `root` the content of `widget` starts (px).

    Every card between them indents by its inset and border; the other
    containers of a form (union, optional and object editors) have no margins.

    Args:
        widget: A widget inside `root`.
        root: The form's root object editor.

    Returns:
        The indent.
    """
    indent = 0
    parent = widget.parentWidget()
    while parent is not None and parent is not root:
        if isinstance(parent, CollapsibleGroupBox):
            indent += CARD_INDENT
        parent = parent.parentWidget()
    return indent


def align_label_columns(root: ObjectWidget[Any]) -> None:
    """
    Make every editor of a form start on one column, and every card's variants with them.

    The column is where the widest label ends, its object editor's indent
    included. Each object editor's label column is widened to reach it, and
    each card's title block too, so that the variants on its title line
    start where the editors do.

    Args:
        root: The form's root object editor.
    """
    objects = [root, *root.findChildren(ObjectWidget)]
    indents = {id(obj): _indent(obj, root) for obj in objects}
    column = max(
        (
            obj.natural_label_width() + indents[id(obj)]
            for obj in objects
            if obj.natural_label_width() > 0
        ),
        default=0,
    )
    if column == 0:
        return
    for obj in objects:
        obj.set_label_column(column - indents[id(obj)])
    for card in root.findChildren(CollapsibleGroupBox):
        card.set_title_column(max(0, column - _indent(card, root) - CARD_INDENT))

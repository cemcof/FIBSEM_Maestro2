# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Editor of a field whose default is computed from other settings."""

from typing import TYPE_CHECKING, Any, TypeVar, cast

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.issues import FieldIssue, FieldValidationError, Severity

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

DEFAULT_UNAVAILABLE_ISSUE = "dynamic_default"
"""Key of the issue a field reports while it follows a default that cannot be computed."""

_INVALID = object()
"""Stands for an invalid inner value in change detection."""

AUTO_MARK = "auto"
"""Shown beside the value while the field follows its default."""

_AUTO_COLOR = "#8fa6b8"
"""The quiet grey of the other form marks."""


class DynamicDefaultWidget(QWidget, FieldEditor[T | None]):
    """
    Shows the computed default of a field until the user sets a value.

    While the field follows its default, its value is None and the inner
    editor shows the default as computed now; `refresh` recomputes it, e.g.
    after the settings it is computed from changed. A value set in the inner
    editor stops the following; `follow_default` resumes it.

    While the field follows its default, a grey "auto" mark is shown beside
    the value, its tooltip saying what the default is computed from. While it
    follows a default that cannot be computed, it reports a warning saying
    what to do.

    Args:
        inner: The editor of the value; any field type.
        compute: Returns the default now, or None if it cannot be computed.
        unavailable: Returns the warning shown while the default cannot be
            computed, saying why and what to do.
        source: What the default is computed from, completing "Computed from …".
        parent: The parent widget, if any.
    """

    def __init__(
        self,
        inner: FieldEditor[T],
        compute: Callable[[], T | None],
        unavailable: Callable[[], str],
        source: str = "other settings",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        FieldEditor.__init__(self)
        self._inner = inner
        self._compute = compute
        self._unavailable = unavailable
        self._following = True
        # set while the inner editor shows a value this widget chose, not the user
        self._updating = False

        inner_widget = cast("QWidget", inner)
        # only as wide as its controls, so the mark sits right after the value
        inner_widget.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self._auto = QLabel(AUTO_MARK)
        self._auto.setStyleSheet(f"color: {_AUTO_COLOR}; font-size: 8pt;")
        self._auto.setToolTip(
            f"Computed from {source}; follows it until a value is set. "
            "Right-click the field to follow it again."
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(inner_widget)
        layout.addWidget(self._auto)
        layout.addStretch()

        inner.on_change(self._on_inner_changed)
        # issues reported on the inner editor are shown on this field's row
        inner.on_issues_changed(self._emit_issues_changed)
        self.refresh()

    @property
    def auto_mark(self) -> QLabel:
        """The mark shown beside the value while the field follows its default."""
        return self._auto

    @property
    def following(self) -> bool:
        """True while the field follows its default (its value is None)."""
        return self._following

    def default(self) -> T | None:
        """
        The default as computed now.

        Returns:
            The default, or None if it cannot be computed.
        """
        return self._compute()

    def refresh(self) -> None:
        """
        Show the default as computed now, if the field follows it.

        Does not notify: the value (None) is unchanged. Listeners of the inner
        editor (e.g. overlays reading the shown value) are notified.
        """
        self._auto.setVisible(self._following)
        if not self._following:
            self.set_issue(DEFAULT_UNAVAILABLE_ISSUE, None)
            return

        value = self._compute()
        if value is None:
            self.set_issue(
                DEFAULT_UNAVAILABLE_ISSUE,
                FieldIssue(Severity.WARNING, self._unavailable()),
            )
            return

        self.set_issue(DEFAULT_UNAVAILABLE_ISSUE, None)
        self._show(value)

    def follow_default(self) -> None:
        """Follow the default again, showing it, and notify if the value changed."""
        self.set_value(None)

    def get_value(self) -> T | None:
        """
        Return the value set by the user, or None while following the default.

        Returns:
            The inner editor's value, or None.

        Raises:
            FieldValidationError: If the inner editor holds no valid value.
        """
        if self._following:
            return None
        return self._inner.get_value()

    def set_value(self, value: T | None) -> None:
        """
        Follow the default for None, otherwise show `value` as set by the user.

        Notifies only if the value changed.

        Args:
            value: The value, or None to follow the default.
        """
        before = self._current()
        self._following = value is None
        if value is None:
            self.refresh()
        else:
            self.refresh()
            self._show(value)
        if self._current() != before:
            self._emit()

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing of the value.

        Args:
            read_only: If True, prevent edits.
        """
        self._inner.set_read_only(read_only)

    def issues(self) -> list[FieldIssue]:
        """
        Return the issues of this field, including those reported on the inner editor.

        Returns:
            This editor's own issues followed by the inner editor's.
        """
        return [*super().issues(), *self._inner.issues()]

    def highlight_target(self) -> QWidget:
        """Highlight the inner editor's control."""
        return self._inner.highlight_target()

    def value_editor(self) -> FieldEditor[Any]:
        """Return the inner value editor, which shows the value in use (set or default)."""
        return self._inner.value_editor()

    @property
    def inner(self) -> FieldEditor[T]:
        """The editor of the value."""
        return self._inner

    def _show(self, value: T) -> None:
        """Show `value` in the inner editor without taking it for a user edit."""
        self._updating = True
        try:
            self._inner.set_value(value)
        finally:
            self._updating = False

    def _on_inner_changed(self) -> None:
        """A user edit of the inner editor sets the value: stop following the default."""
        if self._updating:
            return
        self._following = False
        self.refresh()
        self._emit()

    def _current(self) -> object:
        """The value for change detection; `_INVALID` for an invalid inner value."""
        if self._following:
            return None
        try:
            return self._inner.get_value()
        except FieldValidationError:
            return _INVALID

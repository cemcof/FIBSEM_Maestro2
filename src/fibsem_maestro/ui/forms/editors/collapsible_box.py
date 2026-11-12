# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""A card holding a group of fields, with a title line whose chevron collapses its body."""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem_maestro.ui.forms.style import (
    CARD_INDENT,
    CARD_INSET,
    CARD_OBJECT_NAME,
    CARD_STYLESHEET,
    MUTED_TEXT_COLOR,
    VARIANT_SPACING,
)

if TYPE_CHECKING:
    from PyQt6.QtGui import QMouseEvent, QPaintEvent, QShowEvent

_CHEVRON_SIZE = 14
"""Edge of the collapse chevron (px)."""

_CHEVRON_BOX = _CHEVRON_SIZE + 4
"""Edge of the chevron's clickable box (px)."""

_TITLE_SPACING = 4
"""Space between the chevron, the checkbox slot and the title (px)."""

TITLE_TEXT_OFFSET = CARD_INDENT + _CHEVRON_BOX + _TITLE_SPACING
"""
How far right of a card's left edge its title starts (px), without a checkbox before it.

Field labels next to the card start there too, so that labels and titles
of one object form one column of text.
"""

SECTION_OBJECT_NAME = "formSection"
"""Object name of a section (a box without a card, e.g. Propagations)."""

_HEADER_SPACING = 12
"""Space between the title block, the header widgets and the trailing widgets (px)."""

TITLE_STYLE = "font-weight: bold;"
"""Style of a card's title."""

_SECTION_TITLE_STYLE = "font-size: 14px; font-weight: bold;"
"""Style of a section's title."""


class ClickableLabel(QLabel):
    """
    A QLabel that exposes a clicked signal.

    Emits ``clicked`` on each mouse press.
    """

    clicked = pyqtSignal()

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        """
        Emit `clicked` and forward the event to the base class.

        Args:
            ev: The mouse press event.
        """

        self.clicked.emit()
        super().mousePressEvent(ev)


class Chevron(ClickableLabel):
    """The collapse toggle of a card: a chevron pointing down (expanded) or right (collapsed)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._collapsed = False
        self.setFixedSize(_CHEVRON_BOX, _CHEVRON_BOX)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background: transparent; border: none;")

    @property
    def collapsed(self) -> bool:
        """Whether the chevron points right (the card is collapsed)."""
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """
        Point right while collapsed, down while expanded.

        Args:
            collapsed: Whether the card is collapsed.
        """
        self._collapsed = collapsed
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """
        Draw the chevron: two strokes meeting at its tip.

        Args:
            a0: The paint event.
        """
        _ = a0
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(MUTED_TEXT_COLOR), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            cx, cy = self.width() / 2, self.height() / 2
            # half the chevron's width, and the depth of its tip (px)
            half, depth = 4.5, 2.5
            if self._collapsed:
                points = [(-depth, -half), (depth, 0.0), (-depth, half)]
            else:
                points = [(-half, -depth), (0.0, depth), (half, -depth)]
            painter.drawPolyline([QPointF(cx + x, cy + y) for x, y in points])
        finally:
            painter.end()


class CollapsibleGroupBox(QFrame):
    """
    A card with a title line above its body; the chevron in the title line collapses the body.

    The title line holds, from the left: the chevron, a slot for a checkbox
    (`title_slot`, e.g. of an optional group), the title, the widgets added
    with `add_header_widget` (e.g. the variants of a union), and the widgets
    added with `add_header_trailing` at the right end. The chevron, slot and
    title form the title block, whose width `set_title_column` can extend so
    that what follows starts on a given column.

    Nested cards alternate two background shades (`form_style.CARD_COLORS`).
    As a section (`section=True`, e.g. Propagations) the box has no card: its
    title is larger and followed by a divider line.

    Subclasses put their content in with `_set_body` and return the widget to
    hide from `_collapse_body`.

    Args:
        title: The title; empty for none (e.g. a field's label is set later
            with `set_title_label`).
        parent: The parent widget, if any.
        section: Draw a section instead of a card.
    """

    def __init__(
        self, title: str = "", parent: QWidget | None = None, section: bool = False
    ) -> None:
        super().__init__(parent)
        self._collapsed = False
        self._active = True
        self._empty = False
        self._section = section

        if section:
            self.setObjectName(SECTION_OBJECT_NAME)
            self.setStyleSheet(f"QFrame#{SECTION_OBJECT_NAME} {{ border: none; }}")
        else:
            self.setObjectName(CARD_OBJECT_NAME)
            self.setProperty("shade", "a")
            self.setStyleSheet(CARD_STYLESHEET)

        self._outer = QVBoxLayout(self)
        self._outer.setSpacing(6)

        self._header = QWidget()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(_HEADER_SPACING)
        self._header_layout = header_layout

        self._title_block = QWidget()
        title_layout = QHBoxLayout(self._title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(_TITLE_SPACING)
        self._title_layout = title_layout

        self._toggle = Chevron(self._title_block)
        self._toggle.clicked.connect(self._on_toggle_clicked)
        title_layout.addWidget(self._toggle)

        self._slot = QWidget()
        self._slot.setFixedSize(QSize(18, 18))
        self._slot.hide()
        title_layout.addWidget(self._slot)

        self._title_label: QLabel = QLabel(title)
        self._title_label.setStyleSheet(
            _SECTION_TITLE_STYLE if section else TITLE_STYLE
        )
        title_layout.addWidget(self._title_label)
        title_layout.addStretch()
        header_layout.addWidget(self._title_block)

        # between the variants (added before it) and the trailing widgets
        self._trailing_at = header_layout.count()
        self._header_widget_count = 0
        if section:
            divider = QFrame()
            divider.setFrameShape(QFrame.Shape.HLine)
            divider.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            header_layout.addWidget(divider, 1)
        else:
            header_layout.addStretch(1)

        self._outer.addWidget(self._header)
        self._body: QWidget | None = None
        self._update_margins()

    def _set_body(self, body: QWidget) -> None:
        """
        Place the content below the title line.

        Args:
            body: The content; hidden while collapsed.
        """
        self._body = body
        self._outer.addWidget(body)
        self._update_margins()

    def _collapse_body(self) -> QWidget:
        """
        Return the widget hidden when collapsed.

        Returns:
            The body widget the toggle should show and hide.

        Raises:
            NotImplementedError: If the subclass placed no body.
        """
        if self._body is None:
            raise NotImplementedError
        return self._body

    @property
    def collapsed(self) -> bool:
        """Whether the body is hidden by the chevron."""
        return self._collapsed

    @property
    def toggle(self) -> Chevron:
        """The chevron collapsing and expanding the body."""
        return self._toggle

    @property
    def title_label(self) -> QLabel:
        """The label showing the title."""
        return self._title_label

    def set_title_label(self, label: QLabel) -> None:
        """
        Show another label as the title, e.g. a field's label.

        Args:
            label: The label; styled as a title and placed in the title block.
        """
        index = self._title_layout.indexOf(self._title_label)
        self._title_layout.removeWidget(self._title_label)
        self._title_label.deleteLater()
        self._title_label = label
        label.setStyleSheet(TITLE_STYLE)
        self._title_layout.insertWidget(index, label)

    def add_header_widget(self, widget: QWidget) -> None:
        """
        Add a widget to the title line after the title, e.g. a variant's radio button.

        Header widgets are `VARIANT_SPACING` apart, further than the title
        line's other parts, so that each radio button's text is clearly apart
        from the next button.

        Args:
            widget: The widget; shown only while the card is active.
        """
        if self._header_widget_count > 0:
            self._header_layout.insertSpacing(
                self._trailing_at, VARIANT_SPACING - _HEADER_SPACING
            )
            self._trailing_at += 1
        self._header_layout.insertWidget(self._trailing_at, widget)
        self._trailing_at += 1
        self._header_widget_count += 1

    def add_header_trailing(self, widget: QWidget) -> None:
        """
        Add a widget at the right end of the title line, e.g. a remove button.

        Args:
            widget: The widget.
        """
        self._header_layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)

    def title_slot(self) -> QWidget:
        """
        Reserve the place before the title for a widget kept elsewhere, e.g. an optional checkbox.

        Returns:
            The placeholder, sized for a checkbox; the caller keeps its own
            widget over it.
        """
        self._slot.show()
        return self._slot

    def set_title_column(self, width: int) -> None:
        """
        Make the title block at least this wide, so the variants after it start on a column.

        Args:
            width: The minimum width of the chevron, slot and title (px).
        """
        self._title_block.setMinimumWidth(width)

    def set_active(self, active: bool) -> None:
        """
        Show or hide everything but the title line's chevron, slot and title.

        An inactive card (e.g. an optional group set to None) shows only its
        title line's start; its collapsed state is kept for when it is active
        again. The chevron stays, so the title does not shift.

        Args:
            active: Whether the body and the title line's other widgets show.
        """
        self._active = active
        for index in range(self._header_layout.count()):
            item = self._header_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None and widget is not self._title_block:
                widget.setVisible(active)
        self._show_body()

    def set_body_empty(self, empty: bool) -> None:
        """
        Hide the body while it has nothing to show, e.g. a union's variant without fields.

        The card then fits its title line, without the space below it.

        Args:
            empty: Whether the body shows nothing.
        """
        self._empty = empty
        self._show_body()

    def _show_body(self) -> None:
        """Show the body unless the card is inactive, collapsed or empty; fit the margins."""
        self._collapse_body().setVisible(
            self._active and not self._collapsed and not self._empty
        )
        self._update_margins()

    def _update_margins(self) -> None:
        """Margins around the title line, and below the body when it is shown."""
        if self._section:
            self._outer.setContentsMargins(0, 4, 0, 0)
            return
        body_shown = self._body is not None and not self._body.isHidden()
        self._outer.setContentsMargins(
            CARD_INSET, 4, CARD_INSET, CARD_INSET if body_shown else 4
        )

    def _on_toggle_clicked(self) -> None:
        """Toggle collapsed state, updating the chevron and body visibility."""
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """
        Collapse or expand the body.

        Args:
            collapsed: True to hide the body, False to show it.
        """
        self._collapsed = collapsed
        self._toggle.set_collapsed(collapsed)
        self._show_body()

    def showEvent(self, a0: QShowEvent | None) -> None:
        """
        Take the shade that alternates with the enclosing card's.

        Args:
            a0: The show event.
        """
        super().showEvent(a0)
        self._update_shade()

    def changeEvent(self, a0: QEvent | None) -> None:
        """
        Take the shade that alternates with the new enclosing card's, after a reparenting.

        Args:
            a0: The change event.
        """
        super().changeEvent(a0)
        if a0 is not None and a0.type() == QEvent.Type.ParentChange:
            self._update_shade()

    def _update_shade(self) -> None:
        """Set the `shade` property from the number of enclosing cards, and restyle if it changed."""
        if self._section:
            return
        depth = 0
        parent = self.parentWidget()
        while parent is not None:
            if parent.objectName() == CARD_OBJECT_NAME:
                depth += 1
            parent = parent.parentWidget()
        shade = "a" if depth % 2 == 0 else "b"
        if self.property("shade") != shade:
            self.setProperty("shade", shade)
            if (style := self.style()) is not None:
                style.unpolish(self)
                style.polish(self)
            self.update()

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Multi-selection lists drawn as checklists: a checkbox per row, ticked while the row is selected."""

from PyQt6 import sip
from PyQt6.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
)
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QHelpEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
)

from fibsem_maestro.ui.forms.style import BORDER_COLOR, WINDOW_COLOR

CHECKLIST_MAX_ROWS = 7
"""Rows a checklist shows before it scrolls."""

CHECKLIST_MIN_WIDTH = 160
"""Width of a checklist of short names (px)."""

CHECKLIST_MAX_WIDTH = 720
"""
Width a checklist does not grow beyond (px).

Up to it, a checklist is as wide as its longest name; a longer name is cut
with an ellipsis and shown whole in its tooltip.
"""

_ROW_HEIGHT = 22
"""Height of a row (px)."""

_BOX = 14
"""Edge of a row's checkbox (px)."""

_TEXT_LEFT = _BOX + 14
"""Where a row's text starts, from the row's left edge (px)."""

_TEXT_RIGHT = 6
"""Space after a row's text (px)."""

_CHECKED = "#1A72BB"
"""A ticked checkbox: qdarkstyle's third accent."""

_TEXT = "#DFE1E2"
"""A selected row's text: qdarkstyle's first text colour."""

_UNSELECTED_TEXT = "#C0C4C8"
"""An unselected row's text: qdarkstyle's second text colour."""

_DISABLED_TEXT = "#788D9C"
"""Text and boxes while the list is disabled: qdarkstyle's disabled colour."""

_HOVER = "#293544"
"""Behind the row under the mouse: qdarkstyle's second background."""

_BOX_BORDER = "#60798B"
"""An empty checkbox's border: qdarkstyle's sixth background, visible on the window colour."""


class ChecklistDelegate(QStyledItemDelegate):
    """
    Draws each row of a multi-selection list as a checkbox and its text.

    The checkbox is ticked while the row is selected; a selected row's text
    is bold. No selection highlight is drawn. A foreground colour set on an
    item (e.g. for an entry that is not available) is kept.

    Args:
        parent: The list, which owns the delegate.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """
        Return the size of a row: its text after the checkbox, at a fixed height.

        The width is that of the text in bold, as when the row is selected,
        so ticking a row never cuts its text.

        Args:
            option: The style options of the row.
            index: The row's index.

        Returns:
            The row's size.
        """
        text_width = _bold_metrics(option.font).horizontalAdvance(_text(index))
        return QSize(_TEXT_LEFT + text_width + _TEXT_RIGHT, _ROW_HEIGHT)

    def helpEvent(
        self,
        event: QHelpEvent | None,
        view: QAbstractItemView | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        """
        Show a cut row's whole text as its tooltip, unless the row has a tooltip of its own.

        Args:
            event: The help event.
            view: The list.
            option: The style options of the row.
            index: The row's index.

        Returns:
            Whether the event was handled.
        """
        if (
            event is not None
            and event.type() == QEvent.Type.ToolTip
            and index.data(Qt.ItemDataRole.ToolTipRole) is None
            and self.sizeHint(option, index).width() > _visible_width(option, view)
        ):
            QToolTip.showText(event.globalPos(), _text(index), view)
            return True
        return super().helpEvent(event, view, option, index)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        """
        Draw a row: hover background, checkbox, text.

        Args:
            painter: The painter.
            option: The style options of the row (its rectangle and state).
            index: The row's index.
        """
        if painter is None:
            return
        state = option.state
        rect = option.rect
        selected = bool(state & QStyle.StateFlag.State_Selected)
        enabled = bool(state & QStyle.StateFlag.State_Enabled)

        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if enabled and state & QStyle.StateFlag.State_MouseOver:
                painter.fillRect(rect, QColor(_HOVER))

            box = QRectF(
                rect.left() + 6, rect.center().y() - _BOX / 2 + 0.5, _BOX, _BOX
            )
            if selected:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(_CHECKED if enabled else _DISABLED_TEXT))
                painter.drawRoundedRect(box, 3, 3)
                tick = QPen(QColor("#FFFFFF"), 1.8)
                tick.setCapStyle(Qt.PenCapStyle.RoundCap)
                tick.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(tick)
                x, y = box.left(), box.top()
                painter.drawPolyline(
                    [
                        QPointF(x + 3.5, y + 7.5),
                        QPointF(x + 6, y + 10),
                        QPointF(x + 10.5, y + 4.5),
                    ]
                )
            else:
                painter.setPen(
                    QPen(QColor(_BOX_BORDER if enabled else BORDER_COLOR), 1)
                )
                painter.setBrush(QColor(WINDOW_COLOR))
                painter.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)

            foreground = index.data(Qt.ItemDataRole.ForegroundRole)
            if not enabled:
                color = QColor(_DISABLED_TEXT)
            elif isinstance(foreground, QBrush):
                color = foreground.color()
            elif isinstance(foreground, QColor):
                color = foreground
            else:
                color = QColor(_TEXT if selected else _UNSELECTED_TEXT)
            font = QFont(option.font)
            font.setBold(selected)
            painter.setFont(font)
            painter.setPen(color)
            # a row longer than the list is cut at the list's edge
            visible = _visible_width(option, option.widget)
            text_rect = rect.adjusted(
                _TEXT_LEFT, 0, visible - rect.width() - _TEXT_RIGHT, 0
            )
            text = QFontMetrics(font).elidedText(
                _text(index), Qt.TextElideMode.ElideRight, text_rect.width()
            )
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                text,
            )
        finally:
            painter.restore()


def _visible_width(option: QStyleOptionViewItem, view: object) -> int:
    """
    The width of the part of a row inside the list (px).

    A row is as wide as its text, which may be wider than the list.

    Args:
        option: The style options of the row (its rectangle).
        view: The list, if known.

    Returns:
        The row's width, cut at the right edge of the list's viewport.
    """
    rect = option.rect
    viewport = view.viewport() if isinstance(view, QAbstractItemView) else None
    if viewport is None:
        return rect.width()
    return max(0, min(rect.right() + 1, viewport.width()) - rect.left())


def _text(index: QModelIndex) -> str:
    """A row's text."""
    return str(index.data(Qt.ItemDataRole.DisplayRole) or "")


def _bold_metrics(font: QFont) -> QFontMetrics:
    """The metrics of a font in bold."""
    bold = QFont(font)
    bold.setBold(True)
    return QFontMetrics(bold)


def _row_width(widget: QListWidget, row: int) -> int:
    """The width a row of the list needs (px)."""
    item = widget.item(row)
    if item is None:
        return 0
    return widget.sizeHintForIndex(widget.indexFromItem(item)).width()


class _SizeFitter(QObject):
    """
    Keeps a checklist as tall as its rows, up to `CHECKLIST_MAX_ROWS`, and as wide as its longest row.

    The width stays between `CHECKLIST_MIN_WIDTH` and `CHECKLIST_MAX_WIDTH`.
    """

    def __init__(self, widget: QListWidget) -> None:
        super().__init__(widget)
        self._widget = widget
        model = widget.model()
        if model is not None:
            model.rowsInserted.connect(self.fit)
            model.rowsRemoved.connect(self.fit)
            model.modelReset.connect(self.fit)
            model.layoutChanged.connect(self.fit)
            # e.g. a renamed action
            model.dataChanged.connect(self.fit)
        widget.installEventFilter(self)
        self.fit()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """
        Refit when the style (and so the frame) or the font changes, and when shown.

        Args:
            a0: The watched object.
            a1: The event.

        Returns:
            False: the event is not consumed.
        """
        if sip.isdeleted(self._widget):
            return False
        if (
            a0 is self._widget
            and a1 is not None
            and a1.type()
            in (
                QEvent.Type.StyleChange,
                QEvent.Type.Polish,
                QEvent.Type.FontChange,
                QEvent.Type.Show,
            )
        ):
            self.fit()
        return super().eventFilter(a0, a1)

    def fit(self, *_: object) -> None:
        """Set the list's height to that of its rows and its width to that of its longest row."""
        # the model announces its rows' removal while the list is destroyed
        if sip.isdeleted(self._widget):
            return
        widget = self._widget
        count = widget.count()
        frame = 2 * widget.frameWidth()
        widget.setFixedHeight(
            max(1, min(count, CHECKLIST_MAX_ROWS)) * _ROW_HEIGHT + frame + 2
        )

        widest = max((_row_width(widget, row) for row in range(count)), default=0)
        scrollbar = widget.verticalScrollBar()
        bar = (
            scrollbar.sizeHint().width()
            if scrollbar is not None and count > CHECKLIST_MAX_ROWS
            else 0
        )
        width = widest + frame + bar + 2
        widget.setFixedWidth(max(CHECKLIST_MIN_WIDTH, min(width, CHECKLIST_MAX_WIDTH)))


def make_checklist(widget: QListWidget) -> None:
    """
    Draw a multi-selection list as a checklist, as tall and as wide as its rows, within limits.

    Selection works as before (a click toggles a row); only the drawing and
    the size change. There is no horizontal scroll bar: a row too long for
    `CHECKLIST_MAX_WIDTH` is cut with an ellipsis and shows its whole text
    as a tooltip.

    Args:
        widget: The list; keeps its selection mode and items.
    """
    widget.setItemDelegate(ChecklistDelegate(widget))
    widget.setMouseTracking(True)
    widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # rows differ in width, which sets the list's
    widget.setUniformItemSizes(False)
    _SizeFitter(widget)

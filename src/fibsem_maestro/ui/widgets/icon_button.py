# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""A flat, square, icon-only tool button."""

from PyQt6.QtCore import QEvent, QRectF, QSize, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPaintEvent, QPalette
from PyQt6.QtWidgets import QToolButton, QWidget

from fibsem_maestro.ui.widgets.busy import (
    BUSY_COLOR,
    BUSY_DELAY_MS,
    BUSY_FRAME_MS,
    BUSY_STEP_DEG,
    paint_busy_arc,
)
from fibsem_maestro.ui.widgets.icons import render_icon

_BUTTON_SIZE = 30  # px
_ICON_SIZE = 18  # px

_STYLESHEET = """
    QToolButton#iconButton {
        border: 1px solid transparent;
        border-radius: 4px;
        background: transparent;
        padding: 0px;
    }
    QToolButton#iconButton:hover {
        border-color: #346792;
        background: rgba(52, 103, 146, 40);
    }
    QToolButton#iconButton:pressed {
        background: rgba(52, 103, 146, 90);
    }
    QToolButton#iconButton:disabled {
        border-color: transparent;
        background: transparent;
    }
"""


class IconButton(QToolButton):
    """
    Flat, square, icon-only tool button with a consistent look on every platform.

    The icon is rendered from a built-in SVG body in the palette's button text
    colour, with an explicit disabled variant, and is re-rendered when the
    palette or style changes. Once busy (`set_busy`) for `BUSY_DELAY_MS`,
    the icon is in the busy colour, also when disabled, and an arc circles it.

    Args:
        icon_body: SVG elements on a 24x24 canvas, e.g. `ICON_EDIT`.
        tooltip: Help text shown on hover (rich text allowed).
        accessible_name: Name announced by screen readers.
        parent: Parent widget.
    """

    def __init__(
        self,
        icon_body: str,
        tooltip: str,
        accessible_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_body = icon_body
        self._busy = False
        self._busy_shown = False
        # start of the busy arc, in degrees clockwise from 3 o'clock
        self._busy_angle = 0.0
        self._busy_delay = QTimer(self)
        self._busy_delay.setSingleShot(True)
        self._busy_delay.setInterval(BUSY_DELAY_MS)
        self._busy_delay.timeout.connect(self._show_busy)
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(BUSY_FRAME_MS)
        self._busy_timer.timeout.connect(self._advance_busy_arc)

        self.setObjectName("iconButton")
        self.setStyleSheet(_STYLESHEET)
        self.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self.setToolTip(tooltip)
        self.setAccessibleName(accessible_name)

        self._refresh_icon()

    @property
    def busy(self) -> bool:
        """Whether work started by the button is running."""
        return self._busy

    @property
    def busy_shown(self) -> bool:
        """Whether the button shows that work it started is running."""
        return self._busy_shown

    @property
    def busy_arc_angle(self) -> float:
        """Where the busy arc starts, in degrees clockwise from 3 o'clock."""
        return self._busy_angle

    def set_busy(self, busy: bool) -> None:
        """
        Tell whether work started by this button is running.

        Args:
            busy: True while the work runs.
        """
        if busy == self._busy:
            return
        self._busy = busy
        if busy:
            self._busy_delay.start()
        else:
            self._busy_delay.stop()
            self._set_busy_shown(False)

    def show_blocked_frame(self, first: bool) -> None:
        """
        Show the busy look now, while work started by the button blocks the event loop.

        For `while_blocked`.

        Args:
            first: Whether this is the first frame; later ones move the arc on.
        """
        if first:
            self._set_busy_shown(True)
        else:
            self._advance_busy_arc()
        self.repaint()

    def end_blocked_frames(self) -> None:
        """End the busy look of `show_blocked_frame`, unless the button is still busy."""
        if not self._busy:
            self._set_busy_shown(False)

    def _show_busy(self) -> None:
        self._set_busy_shown(True)

    def _set_busy_shown(self, shown: bool) -> None:
        if shown == self._busy_shown:
            return
        self._busy_shown = shown
        self._busy_angle = 0.0
        if shown:
            self._busy_timer.start()
        else:
            self._busy_timer.stop()
        self._refresh_icon()
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Paint the button, and the busy arc around its icon while it is shown."""
        super().paintEvent(a0)
        if not self._busy_shown:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            paint_busy_arc(
                painter,
                QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5),
                self._busy_angle,
            )
        finally:
            painter.end()

    def _advance_busy_arc(self) -> None:
        self._busy_angle = (self._busy_angle + BUSY_STEP_DEG) % 360.0
        self.update()

    def changeEvent(self, a0: QEvent | None) -> None:
        """Re-render the icon when the palette or style changes."""
        super().changeEvent(a0)
        if a0 is not None and a0.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            self._refresh_icon()

    def _refresh_icon(self) -> None:
        """Render normal and disabled variants for common pixel ratios."""
        palette = self.palette()
        normal = palette.color(
            QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText
        )
        disabled = palette.color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText
        )
        if self._busy_shown:
            # the tools are disabled while a task runs, the busy one included
            normal = disabled = QColor(BUSY_COLOR)

        icon = QIcon()
        for ratio in sorted({1.0, 2.0, self.devicePixelRatioF()}):
            icon.addPixmap(
                render_icon(self._icon_body, normal, _ICON_SIZE, ratio),
                QIcon.Mode.Normal,
            )
            icon.addPixmap(
                render_icon(self._icon_body, disabled, _ICON_SIZE, ratio),
                QIcon.Mode.Disabled,
            )
        self.setIcon(icon)

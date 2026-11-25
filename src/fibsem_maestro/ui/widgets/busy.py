# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The arc marking work in progress, and showing it while work blocks the event loop."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

BUSY_COLOR = "#5a9fd4"
"""Colour of the arc, and of the icon of a busy button."""

BUSY_DELAY_MS = 250
"""How long work must run before it is shown as busy."""

BUSY_FRAME_MS = 50
"""Time between two positions of a turning arc."""

BUSY_STEP_DEG = 22.5
"""How far a turning arc moves per frame; a turn takes 16 frames."""

_ARC_SPAN_DEG = 110.0
"""Length of the arc, in degrees."""

_ARC_WIDTH = 2.0
"""Width of the arc's stroke (px)."""


def paint_busy_arc(painter: QPainter, rect: QRectF, angle: float) -> None:
    """
    Draw the arc inside a rectangle.

    Args:
        painter: An active painter.
        rect: The rectangle the arc's ellipse fills.
        angle: Where the arc starts, in degrees clockwise from 3 o'clock.
    """
    pen = QPen(QColor(BUSY_COLOR), _ARC_WIDTH)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # Qt's angles are in 1/16 degree, counter-clockwise
    painter.drawArc(rect, round(-angle * 16), round(_ARC_SPAN_DEG * 16))


class _BlockedProgress(QObject):
    """
    Calls `on_frame` while work blocking the event loop runs long, see `while_blocked`.

    Args:
        on_frame: Called with True once the work has run for `BUSY_DELAY_MS`,
            then with False every `BUSY_FRAME_MS`.
        on_end: Called when the frames stop, if any was shown.
    """

    def __init__(
        self, on_frame: Callable[[bool], None], on_end: Callable[[], None]
    ) -> None:
        super().__init__()
        self._on_frame = on_frame
        self._on_end = on_end
        self._started = time.monotonic()
        self._last_frame: float | None = None
        self._in_frame = False
        self._stopped = False
        self._modal = QApplication.activeModalWidget()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """
        Show a frame if it is time to.

        Args:
            a0: The object receiving the event.
            a1: The event; never filtered out.

        Returns:
            False.
        """
        _ = a0, a1
        # blocking work runs no timers; the events it sends, e.g. to widgets
        # it creates, serve as the clock instead. A frame sends events too.
        if self._stopped or self._in_frame:
            return False

        # a dialog opened by the work waits for the user: the work is not busy
        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not self._modal:
            self.stop()
            return False

        now = time.monotonic()
        if self._last_frame is None:
            if now - self._started < BUSY_DELAY_MS / 1000:
                return False
        elif now - self._last_frame < BUSY_FRAME_MS / 1000:
            return False

        self._in_frame = True
        try:
            self._on_frame(self._last_frame is None)
            self._last_frame = now
        finally:
            self._in_frame = False
        return False

    def stop(self) -> None:
        """Show no more frames, and end those shown."""
        if self._stopped:
            return
        self._stopped = True
        if self._last_frame is not None:
            self._on_end()


@contextmanager
def while_blocked(
    on_frame: Callable[[bool], None], on_end: Callable[[], None]
) -> Generator[None]:
    """
    Show progress while the work in the block runs long without running the event loop.

    The frames stop at the end of the block, or when the work opens a modal
    dialog. `on_frame` must only paint (e.g. `QWidget.repaint`) and change
    what is shown: it runs in the middle of the work.

    Args:
        on_frame: Called with True once the work has run for `BUSY_DELAY_MS`,
            then with False every `BUSY_FRAME_MS`.
        on_end: Called when the frames stop, if any was shown.

    Yields:
        None.
    """
    app = QApplication.instance()
    progress = _BlockedProgress(on_frame, on_end)
    if app is not None:
        app.installEventFilter(progress)
    try:
        yield
    finally:
        if app is not None:
            app.removeEventFilter(progress)
        progress.stop()

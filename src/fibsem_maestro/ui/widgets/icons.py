# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Every icon of the GUI, as SVG bodies drawn on a 24x24 canvas.

A body is a sequence of SVG elements. It is placed in a template that strokes
everything in one colour; a body may also fill shapes with `currentColor`,
which is the same colour.
"""

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtSvg import QSvgRenderer

ICON_PREPARE = (
    '<path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
    '<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/>'
)
"""Arrow into a tray: store the microscope state."""

ICON_APPLY = (
    '<path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
    '<path d="M12 15V3"/><path d="M7 8l5-5 5 5"/>'
)
"""Arrow out of a tray: apply the stored properties."""

ICON_EDIT = '<path d="M15 4l5 5L9 20H4v-5z"/><path d="M13 6l5 5"/>'
"""Pencil: edit the stored properties."""

ICON_TEST = (
    '<path d="M9 3h6"/>'
    '<path d="M10 3v6L4.5 18.5a1 1 0 0 0 .9 1.5h13.2a1 1 0 0 0 .9-1.5L14 9V3"/>'
    '<path d="M7 15h10"/>'
)
"""Lab flask: test the action outside the acquisition."""

ICON_RESET_STATE = (
    '<path d="M4 12a8 8 0 1 0 2.34-5.66"/><path d="M4 4v5h5"/>'
    '<circle cx="12" cy="12" r="1.5" fill="currentColor"/>'
)
"""Counter-clockwise arrow around a dot: discard the action's internal state."""

ICON_SKULL = (
    '<path d="M12 3C7.6 3 4 6.3 4 10.5c0 2.6 1.2 4.5 3 5.7V19a1 1 0 0 0 1 1h8'
    'a1 1 0 0 0 1-1v-2.8c1.8-1.2 3-3.1 3-5.7C20 6.3 16.4 3 12 3z"/>'
    '<circle cx="9" cy="11" r="1.5"/><circle cx="15" cy="11" r="1.5"/>'
    '<path d="M12 14l-1 2h2z"/>'
    '<path d="M10 18v2"/><path d="M14 18v2"/>'
)
"""Skull: danger mode, settings editable at any time."""

ICON_PLAY = '<path d="M7 4.5v15l12.5-7.5z" fill="currentColor"/>'
"""Filled triangle: start or resume the workflow."""

ICON_PAUSE = (
    '<rect x="6" y="4.5" width="4" height="15" rx="1" fill="currentColor"/>'
    '<rect x="14" y="4.5" width="4" height="15" rx="1" fill="currentColor"/>'
)
"""Two filled bars: pause the workflow."""

ICON_ERROR = (
    '<circle cx="12" cy="12" r="10" fill="#f44336" stroke="none"/>'
    '<path d="M8.5 8.5l7 7M15.5 8.5l-7 7" stroke="#ffffff"/>'
)
"""Red disc with a white cross: an error. Drawn in fixed colours."""

ICON_RETRY = '<path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v5h-5"/>'
"""Circular arrow: run the failed action again."""

ICON_SKIP = '<path d="M5 5.5v13l9.5-6.5z" fill="currentColor"/><path d="M18.5 5.5v13"/>'
"""Triangle against a bar: skip to the next action."""

ICON_CANCEL = '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>'
"""Circle with a cross: abandon the run."""

ICON_NEW_WORKFLOW = (
    '<path d="M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7z"/>'
    '<path d="M14 3v4h4"/><path d="M12 11v6M9 14h6"/>'
)
"""Sheet with a plus: create a new workflow."""

ICON_IMPORT_WORKFLOW = (
    '<path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4'
    'a1 1 0 0 1-1-1z"/>'
    '<path d="M12 10.5v5.5"/><path d="M9.5 13.5L12 16l2.5-2.5"/>'
)
"""Arrow into a folder: import the actions of another workflow."""

ICON_START_OVER = (
    '<path d="M4.5 5.5v13"/>'
    '<path d="M12.5 6.5v11L6.5 12z" fill="currentColor"/>'
    '<path d="M19.5 6.5v11L13.5 12z" fill="currentColor"/>'
)
"""Double triangle back against a bar: return to slice 0, deleting the acquired data."""

ICON_SKIP_SLICE = '<path d="M4 12h15"/><path d="M13 6l6 6-6 6"/>'
"""Single arrow forward: skip the rest of the slice, a small step forward."""

ICON_MICROSCOPE = (
    '<path d="M7.5 3.5l3 1.5-3.5 7-3-1.5z"/><path d="M5.5 11.5l-1 2"/>'
    '<path d="M12 8.5a5.5 5.5 0 0 1 2 10"/><path d="M8 15.5h6"/>'
    '<path d="M5 20.5h14"/><path d="M14 18.5v2"/>'
)
"""Microscope: the microscope settings."""

_SVG_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'color="{color}" stroke="{color}" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round">'
    "{body}</svg>"
)


def render_icon(
    body: str, color: QColor, size: int, device_pixel_ratio: float
) -> QPixmap:
    """
    Render an icon body as a square pixmap in the given colour.

    Args:
        body: SVG elements drawn on a 24x24 canvas.
        color: Stroke colour, also used for `currentColor` fills.
        size: Logical edge length in pixels.
        device_pixel_ratio: Ratio of physical to logical pixels.

    Returns:
        A transparent pixmap of `size` logical pixels.
    """
    svg = _SVG_TEMPLATE.format(color=color.name(), body=body).encode()
    renderer = QSvgRenderer(QByteArray(svg))

    side = round(size * device_pixel_ratio)
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()

    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap


def svg_icon(
    body: str,
    size: int,
    color: QColor | str | None = None,
    *,
    palette: QPalette | None = None,
    device_pixel_ratio: float | None = None,
) -> QIcon:
    """
    Build an icon with Normal and Disabled variants from an SVG body.

    Each variant is rendered at device-pixel ratios 1, 2 and the screen's.
    Needs a `QGuiApplication`.

    Args:
        body: SVG elements drawn on a 24x24 canvas.
        size: Logical edge length in pixels.
        color: Colour of the Normal variant; None for the palette's `ButtonText`.
        palette: Palette to take the colours from; None for the application's.
            Pass a widget's palette to follow that widget's style.
        device_pixel_ratio: The screen's ratio of physical to logical pixels;
            None for the primary screen's.

    Returns:
        The icon.
    """
    palette = palette if palette is not None else QGuiApplication.palette()
    normal = (
        QColor(color)
        if color is not None
        else palette.color(QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText)
    )
    disabled = palette.color(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText
    )

    if device_pixel_ratio is None:
        screen = QGuiApplication.primaryScreen()
        device_pixel_ratio = screen.devicePixelRatio() if screen is not None else 1.0

    icon = QIcon()
    for ratio in sorted({1.0, 2.0, device_pixel_ratio}):
        icon.addPixmap(render_icon(body, normal, size, ratio), QIcon.Mode.Normal)
        icon.addPixmap(render_icon(body, disabled, size, ratio), QIcon.Mode.Disabled)
    return icon

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Decorations drawn over the areas of an area selector, and the registry of overlay kinds.

An overlay kind (`AreaOverlay`) turns runtime values read from the form
(`OverlayValues`, keyed by `AreaOverlay.data_fields`) and the image's pixel
size into a decoration. New kinds are added with `OVERLAYS.add`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from PyQt6.QtCore import QMarginsF, QRectF, Qt
from PyQt6.QtGui import QBrush, QPainterPath, QTransform
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem

from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.registry import Registry
from fibsem_maestro.criterion.criterion import (
    MIN_TILE_PX,
    tile_origins,
    tile_size_in_pixels,
    tile_step_in_pixels,
)
from fibsem_maestro.settings.form_utils import AreaOverlay
from fibsem_maestro.ui.forms.editors.area_select._constants import (
    ARROW_COLOR,
    ARROW_PEN,
    LINE_PEN,
    MARGIN_PEN,
    MAX_LINES,
    MAX_TILES,
    NEXT_SLICE_FILL,
    NEXT_SLICE_PEN,
    TILE_PEN,
)

_FIT_TOLERANCE_PX = 1e-6
"""Slack allowed when checking that the next slice fits the area (image pixels)."""

if TYPE_CHECKING:
    from fibsem_maestro.ui.forms.editors.area_select._rectangle import (
        ResizableRect,
    )

OverlayValues = Mapping[str, Any]
"""
Runtime values feeding one overlay, keyed by its `AreaOverlay.data_fields`.

The keys in use: `margin_nm` (nm), `direction` (`Direction`), `tile_size_nm`
(nm), `tile_relative_overlap` (relative, 0-1), `slice_distance_nm` (nm) and
`slices_milled` (count, from the action's state). A missing key or a None
value means the value is not available.
"""


class AreaDecoration(ABC):
    """
    A non-interactive visual embellishment drawn on a `ResizableRect`.

    A decoration that draws beyond its area claims that space through
    `footprint_margins`; the area is then kept far enough inside the frame for
    the claimed space to fit as well.
    """

    @property
    def footprint_margins(self) -> QMarginsF:
        """
        Space this decoration claims beyond its area, which must stay inside the frame.

        Returns:
            Per-side extension in scene units (image pixels). Zero by default,
            for decorations drawn inside the area.
        """
        return QMarginsF()

    @property
    def problem(self) -> str | None:
        """
        Why this decoration cannot be drawn for the current geometry, if it cannot.

        Returns:
            A short message for the user, or None if the decoration is drawn.
            None by default.
        """
        return None

    @abstractmethod
    def attach(self, rect: ResizableRect) -> None:
        """Create the items as children of `rect` and lay them out."""

    @abstractmethod
    def update(self, rect: ResizableRect) -> None:
        """Re-lay-out the items after the rectangle's geometry changed."""

    @abstractmethod
    def detach(self) -> None:
        """Remove every item this decoration created from the scene."""


class MarginDecoration(AreaDecoration):
    """
    A soft halo extending the area outward by a fixed margin.

    The margin is supplied in scene units (image pixels), so it expands the
    rectangle uniformly on all sides. It is drawn behind the parent's fill in a
    fainter colour so the crisp area border stays legible on top. The halo
    claims its width as footprint, so the area is kept inside the frame together
    with its margin.

    Args:
        margin_px: Margin width in scene units (image pixels).
    """

    def __init__(self, margin_px: float) -> None:
        self._margin_px = margin_px
        self._item: QGraphicsRectItem | None = None

    @property
    def footprint_margins(self) -> QMarginsF:
        """
        The margin on every side, in scene units (image pixels).

        Returns:
            Uniform margins of `margin_px`; a negative margin claims nothing.
        """
        m = max(0.0, self._margin_px)
        return QMarginsF(m, m, m, m)

    def attach(self, rect: ResizableRect) -> None:
        """Create the halo as a child item of `rect`, behind the rectangle, and lay it out."""
        item = QGraphicsRectItem(rect)
        item.setPen(MARGIN_PEN)
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # draws behind the parent
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)
        self._item = item
        self.update(rect)

    def update(self, rect: ResizableRect) -> None:
        """Resize the halo to the rectangle grown by the margin on every side. Does nothing before `attach`."""
        if self._item is not None:
            m = self._margin_px
            self._item.setRect(rect.rect().adjusted(-m, -m, m, m))

    def detach(self) -> None:
        """Remove the halo from the scene. Idempotent."""
        if self._item is not None and (scene := self._item.scene()) is not None:
            scene.removeItem(self._item)
        self._item = None


class DirectionDecoration(AreaDecoration):
    """
    A coloured arrow through the centre of the area.

    Directions are in image space, where `Direction.UP` points toward the top of
    the image (decreasing y).

    Args:
        direction: The direction the arrow points.
    """

    _ANGLES: ClassVar[Mapping[Direction, float]] = {
        Direction.RIGHT: 0.0,
        Direction.DOWN: 90.0,
        Direction.LEFT: 180.0,
        Direction.UP: 270.0,
    }

    def __init__(self, direction: Direction) -> None:
        self._direction = direction
        self._item: QGraphicsPathItem | None = None

    def attach(self, rect: ResizableRect) -> None:
        """Create the arrow as a child item of `rect`, above the fill and below the handles, and lay it out."""
        item = QGraphicsPathItem(rect)
        item.setPen(ARROW_PEN)
        item.setBrush(QBrush(ARROW_COLOR))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # above the fill, below the handles
        item.setZValue(0.5)
        self._item = item
        self.update(rect)

    def update(self, rect: ResizableRect) -> None:
        """Scale and turn the arrow to the rectangle's size, centre and the direction. Does nothing before `attach`."""
        if self._item is None:
            return

        r = rect.rect()
        path = self._arrow_path(0.25 * min(r.width(), r.height()))
        transform = QTransform()
        transform.translate(r.center().x(), r.center().y())
        transform.rotate(self._ANGLES[self._direction])

        self._item.setPath(transform.map(path))

    def detach(self) -> None:
        """Remove the arrow from the scene. Idempotent."""
        if self._item is not None and (scene := self._item.scene()) is not None:
            scene.removeItem(self._item)
        self._item = None

    @staticmethod
    def _arrow_path(length: float) -> QPainterPath:
        """
        Build a filled arrow of the given length pointing along +x, centred at the origin.

        Args:
            length: Total arrow length in scene units.

        Returns:
            A closed path describing the arrow outline.
        """
        half = length / 2.0
        shaft = max(2.0, length * 0.12)
        head_len = length * 0.4
        head_half = shaft * 2.8
        tip, neck = half, half - head_len

        path = QPainterPath()
        path.moveTo(-half, -shaft)
        path.lineTo(neck, -shaft)
        path.lineTo(neck, -head_half)
        path.lineTo(tip, 0.0)
        path.lineTo(neck, head_half)
        path.lineTo(neck, shaft)
        path.lineTo(-half, shaft)
        path.closeSubpath()
        return path


class TileDecoration(AreaDecoration):
    """
    A grid of tiles over the area.

    Args:
        tile_px: Tile side length in pixels, already quantised.
        step_px: Distance between tile origins in pixels.
    """

    def __init__(
        self,
        tile_px: int,
        step_px: int,
    ) -> None:
        self._tile_px = tile_px
        self._step_px = step_px
        self._problem: str | None = None
        self._item: QGraphicsPathItem | None = None

    def attach(self, rect: ResizableRect) -> None:
        """Create the tile grid as a child item of `rect`, above the fill and below the handles, and lay it out."""
        item = QGraphicsPathItem(rect)
        item.setPen(TILE_PEN)
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # above the fill, below the handles
        item.setZValue(0.5)
        self._item = item
        self.update(rect)

    @property
    def problem(self) -> str | None:
        """
        Why the grid is hidden, if it is.

        Returns:
            A message if the area yields more than `MAX_TILES` tiles, else None.
        """
        return self._problem

    def update(self, rect: ResizableRect) -> None:
        """Re-draw the tile grid over the rectangle's area. Does nothing before `attach`."""
        if self._item is not None:
            self._item.setPath(self._grid_path(rect.rect()))

    def detach(self) -> None:
        """Remove the tile grid from the scene. Idempotent."""
        if self._item is not None and (scene := self._item.scene()) is not None:
            scene.removeItem(self._item)
        self._item = None

    def _grid_path(self, r: QRectF) -> QPainterPath:
        """
        Build the tile grid for an area rectangle.

        Args:
            r: The area rectangle, in the parent item's local coordinates.

        Returns:
            A path of tile outlines, or an empty path if a tile does not fit or
            the grid would exceed `MAX_TILES`.
        """
        xs = tile_origins(int(r.width()), self._tile_px, self._step_px)
        ys = tile_origins(int(r.height()), self._tile_px, self._step_px)

        path = QPainterPath()
        if (count := len(xs) * len(ys)) > MAX_TILES:
            self._problem = (
                f"The area yields {count} tiles, more than the {MAX_TILES} that "
                "can be displayed; the tiling overlay is hidden."
            )
            return path

        self._problem = None

        for y in ys:
            for x in xs:
                path.addRect(r.x() + x, r.y() + y, self._tile_px, self._tile_px)
        return path


class AreaSliceDecoration(AreaDecoration):
    """
    Dashed lines splitting the area into strips one shift apart, with the next slice highlighted.

    For `Direction.UP`/`Direction.DOWN` the lines run parallel to the x-axis
    and are spaced `shift_px` apart in y; for `Direction.LEFT`/`Direction.RIGHT`
    they run parallel to the y-axis and are spaced `shift_px` apart in x.
    Directions are in image space, where `Direction.UP` points toward the top
    of the image (decreasing y).

    Lines are anchored at the trailing edge (the edge the area moves away from).
    The strip `slices_milled` strips from the trailing edge, the slice milled
    next, is filled; nothing is filled if it does not fit the area (the area
    is used up) or the progress is unknown.

    Args:
        slice_dist_px: Slice distance, in scene units (image pixels).
        direction: The direction the area moves in.
        slices_milled: Slices milled so far, i.e. the index of the next slice
            counted from the trailing edge; None if unknown.
    """

    def __init__(
        self,
        slice_dist_px: float,
        direction: Direction,
        slices_milled: int | None = None,
    ) -> None:
        self._slice_dist_px = slice_dist_px
        self._direction = direction
        self._slices_milled = slices_milled
        self._item: QGraphicsPathItem | None = None
        self._next_item: QGraphicsRectItem | None = None

    def attach(self, rect: ResizableRect) -> None:
        """Create the slice lines and the next-slice highlight as child items of `rect`, and lay them out."""
        item = QGraphicsPathItem(rect)
        item.setPen(LINE_PEN)
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._item = item

        next_item = QGraphicsRectItem(rect)
        next_item.setPen(NEXT_SLICE_PEN)
        next_item.setBrush(NEXT_SLICE_FILL)
        next_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._next_item = next_item
        self.update(rect)

    def update(self, rect: ResizableRect) -> None:
        """Re-draw the slice lines and the highlight over the rectangle's area. Does nothing before `attach`."""
        if self._item is not None:
            self._item.setPath(
                self._split_path(rect.rect(), self._slice_dist_px, self._direction)
            )
        if self._next_item is not None:
            strip = self.next_slice(rect.rect())
            self._next_item.setVisible(strip is not None)
            self._next_item.setRect(strip if strip is not None else QRectF())

    def detach(self) -> None:
        """Remove the slice lines and the highlight from the scene. Idempotent."""
        for item in (self._item, self._next_item):
            if item is not None and (scene := item.scene()) is not None:
                scene.removeItem(item)
        self._item = None
        self._next_item = None

    def next_slice(self, area: QRectF) -> QRectF | None:
        """
        The strip of `area` milled next.

        Args:
            area: The area, in scene units (image pixels).

        Returns:
            The strip `slices_milled` slice distances from the trailing edge,
            one slice distance deep, in scene units; None if the progress is
            unknown, the slice distance is not positive, or the strip does
            not fit the area.
        """
        index = self._slices_milled
        step = self._slice_dist_px
        if index is None or step <= 0:
            return None

        area = area.normalized()
        horizontal = self._direction in (Direction.UP, Direction.DOWN)
        extent = area.height() if horizontal else area.width()
        near = index * step
        if near + step > extent + _FIT_TOLERANCE_PX:
            return None

        match self._direction:
            case Direction.DOWN:
                return QRectF(area.left(), area.top() + near, area.width(), step)
            case Direction.UP:
                return QRectF(
                    area.left(), area.bottom() - near - step, area.width(), step
                )
            case Direction.RIGHT:
                return QRectF(area.left() + near, area.top(), step, area.height())
            case Direction.LEFT:
                return QRectF(
                    area.right() - near - step, area.top(), step, area.height()
                )

    def _split_path(
        self, area: QRectF, shift_px: float, direction: Direction
    ) -> QPainterPath:
        """
        Build a path of lines splitting `area` into strips `shift_px` apart.

        Args:
            area: The area to split, in scene units (image pixels).
            shift_px: Line spacing, in scene units (image pixels). If not positive, the path is empty.
            direction: The direction the area moves in, in image space (`UP` = decreasing y).

        Returns:
            A path containing one subpath per line; empty if `shift_px` is not
            positive, no lines fit, or the line count would exceed `_MAX_LINES`.
        """
        area = area.normalized()
        match direction:
            case Direction.DOWN:
                start, sign, horizontal = area.top(), 1.0, True
            case Direction.UP:
                start, sign, horizontal = area.bottom(), -1.0, True
            case Direction.RIGHT:
                start, sign, horizontal = area.left(), 1.0, False
            case Direction.LEFT:
                start, sign, horizontal = area.right(), -1.0, False

        extent = area.height() if horizontal else area.width()
        path = QPainterPath()
        for offset in self._split_offsets(extent, shift_px):
            c = start + sign * offset
            if horizontal:
                path.moveTo(area.left(), c)
                path.lineTo(area.right(), c)
            else:
                path.moveTo(c, area.top())
                path.lineTo(c, area.bottom())
        return path

    def _split_offsets(
        self, extent: float, step: float, max_lines: int = MAX_LINES
    ) -> list[float]:
        """
        Offsets of split lines measured from the trailing edge of an area.

        Args:
            extent: Size of the area along the shift axis, in scene units (image pixels).
            step: Spacing between lines, in scene units (image pixels). If not positive, no lines are returned.
            max_lines: If more lines than this would be needed, no lines are returned.

        Returns:
            Offsets from the trailing edge, in scene units (image pixels), in increasing order.
        """
        if step <= 0:
            return []
        n_lines = max(0, math.ceil(extent / step) - 1)
        if n_lines > max_lines:
            return []
        return [k * step for k in range(1, n_lines + 1)]


def _no_problem(values: OverlayValues, pixel_size_nm: float | None) -> str | None:
    """The `OverlayKind.problem` of kinds whose configuration is always usable."""
    _ = values, pixel_size_nm
    return None


@dataclass(frozen=True)
class OverlayKind:
    """
    How one kind of overlay is drawn.

    Attributes:
        build: Builds the decoration from the overlay's values and the image
            pixel size in nm (None if no image is loaded); returns None when
            nothing should be drawn (values or scale missing, or a
            configuration `problem` reports).
        problem: Describes a configuration the action itself would reject, so
            the overlay is not drawn; None if it is usable or cannot be judged.
    """

    build: Callable[[OverlayValues, float | None], AreaDecoration | None]
    problem: Callable[[OverlayValues, float | None], str | None] = _no_problem


OVERLAYS: Registry[OverlayKind] = Registry("area overlay")
"""Overlay kinds keyed by `AreaOverlay.value`."""


def _margin(
    values: OverlayValues, pixel_size_nm: float | None
) -> AreaDecoration | None:
    """Build the `SHOW_MARGIN` decoration."""
    margin_nm = values.get("margin_nm")
    if margin_nm is None or not pixel_size_nm:
        return None
    return MarginDecoration(margin_px=margin_nm / pixel_size_nm)


def _direction(
    values: OverlayValues, pixel_size_nm: float | None
) -> AreaDecoration | None:
    """Build the `SHOW_DIRECTION` decoration."""
    _ = pixel_size_nm
    direction = values.get("direction")
    if direction is None:
        return None
    return DirectionDecoration(direction=direction)


def _tiles(values: OverlayValues, pixel_size_nm: float | None) -> AreaDecoration | None:
    """Build the `SHOW_TILES` decoration."""
    tiling = _tiling(values, pixel_size_nm)
    if not isinstance(tiling, tuple):
        return None
    tile_px, step_px = tiling
    return TileDecoration(tile_px=tile_px, step_px=step_px)


def _tiles_problem(values: OverlayValues, pixel_size_nm: float | None) -> str | None:
    """Describe a `SHOW_TILES` configuration the criterion would reject."""
    tiling = _tiling(values, pixel_size_nm)
    return tiling if isinstance(tiling, str) else None


def _area_slices(
    values: OverlayValues, pixel_size_nm: float | None
) -> AreaDecoration | None:
    """Build the `SHOW_AREA_SLICES` decoration."""
    slice_distance_nm = values.get("slice_distance_nm")
    direction = values.get("direction")
    if slice_distance_nm is None or direction is None or not pixel_size_nm:
        return None
    return AreaSliceDecoration(
        slice_dist_px=slice_distance_nm / pixel_size_nm,
        direction=direction,
        slices_milled=values.get("slices_milled"),
    )


OVERLAYS.add(AreaOverlay.SHOW_MARGIN.value, OverlayKind(build=_margin))
OVERLAYS.add(AreaOverlay.SHOW_DIRECTION.value, OverlayKind(build=_direction))
OVERLAYS.add(
    AreaOverlay.SHOW_TILES.value, OverlayKind(build=_tiles, problem=_tiles_problem)
)
OVERLAYS.add(AreaOverlay.SHOW_AREA_SLICES.value, OverlayKind(build=_area_slices))


def build_decorations(
    overlays: Sequence[tuple[AreaOverlay, OverlayValues]],
    pixel_size_nm: float | None,
) -> list[AreaDecoration]:
    """
    Build every decoration whose values and image scale are available.

    Args:
        overlays: Overlay kinds paired with their own runtime values.
        pixel_size_nm: Image pixel size in nanometers, or None if no image loaded.

    Returns:
        The buildable decorations, in declaration order. Overlays with missing
        values or missing scale are skipped.
    """
    return [
        decoration
        for kind, values in overlays
        if (decoration := OVERLAYS.get(kind.value).build(values, pixel_size_nm))
        is not None
    ]


def overlay_problems(
    overlays: Sequence[tuple[AreaOverlay, OverlayValues]],
    pixel_size_nm: float | None,
) -> list[str]:
    """
    Describe overlay configurations that the action itself would reject.

    Such an overlay is not drawn.

    Args:
        overlays: Overlay kinds paired with their own runtime values.
        pixel_size_nm: Image pixel size in nanometers, or None if no image loaded.

    Returns:
        One short message per unusable overlay, in declaration order.
    """
    return [
        problem
        for kind, values in overlays
        if (problem := OVERLAYS.get(kind.value).problem(values, pixel_size_nm))
        is not None
    ]


def _tiling(
    values: OverlayValues, pixel_size_nm: float | None
) -> tuple[int, int] | str | None:
    """
    Quantise a `SHOW_TILES` overlay's tile size and step to image pixels.

    Args:
        values: Runtime overlay values; reads the tile size (nm) and relative overlap.
        pixel_size_nm: Image pixel size in nanometers, or None if no image loaded.

    Returns:
        `(tile_px, step_px)` in image pixels; a message if the criterion would
        reject this tiling; or None if the values or the image scale are missing.
    """
    tile_size_nm = values.get("tile_size_nm")
    tile_relative_overlap = values.get("tile_relative_overlap")
    if tile_size_nm is None or tile_relative_overlap is None or not pixel_size_nm:
        return None

    tile_px = tile_size_in_pixels(tile_size_nm, pixel_size_nm)
    if tile_px < MIN_TILE_PX:
        return (
            f"A {tile_size_nm:g} nm tile is smaller than {MIN_TILE_PX} pixels "
            "at this pixel size; the criterion would fail."
        )

    step_px = tile_step_in_pixels(tile_px, tile_relative_overlap)
    if step_px <= 0:
        return (
            f"An overlap of {tile_relative_overlap:g} leaves no step between "
            "tiles; the criterion would fail."
        )

    return tile_px, step_px

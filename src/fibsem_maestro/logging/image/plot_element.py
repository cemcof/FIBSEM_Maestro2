# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

LineStyle = Literal["solid", "dashed", "dotted"]


@dataclass(frozen=True)
class PlotElement:
    """Base class for elements drawn on a plot."""

    color: str
    linewidth: float


@dataclass(frozen=True)
class Curve(PlotElement):
    """
    A single plot curve defined by x/y data.

    Attributes:
        x: Sequence of x-values. If None, the curve is plotted as y[i] vs. i.
        y: Sequence of y-values.
        marker: Matplotlib marker drawn at each point, or None for a plain line.
        label: Legend entry, or None for no entry.
    """

    x: Sequence[float] | None
    y: Sequence[float]
    marker: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class VerticalLine(PlotElement):
    """
    A vertical line across the panel at a fixed x position.

    Attributes:
        x: The x-coordinate at which to draw the line.
        linestyle: Stroke pattern.
        label: Legend entry, or None for no entry.
    """

    x: float
    linestyle: LineStyle = "solid"
    label: str | None = None


@dataclass(frozen=True)
class HorizontalLine(PlotElement):
    """
    A horizontal line across the panel at a fixed y position.

    Attributes:
        y: The y-coordinate at which to draw the line.
        linestyle: Stroke pattern.
        label: Legend entry, or None for no entry.
    """

    y: float
    linestyle: LineStyle = "solid"
    label: str | None = None


@dataclass(frozen=True)
class Bars(PlotElement):
    """
    Vertical bars rising from zero (or falling below it for negative heights).

    `color` fills the bars; `linewidth` is the outline width (0 for none).

    Attributes:
        x: Centre of each bar.
        heights: Height of each bar; same length as `x`.
        width: Width of every bar, in x units.
        label: Legend entry, or None for no entry.
    """

    x: Sequence[float]
    heights: Sequence[float]
    width: float = 0.8
    label: str | None = None


@dataclass(frozen=True)
class ColorScale:
    """
    Mapping from values to colors, drawn with a color bar.

    Attributes:
        kind: "sequential" for a magnitude (one hue, light to dark), or
            "diverging" for a signed quantity (two hues around a neutral 0).
        label: Text of the color bar.
        reverse: Map the values in the opposite direction (a sequential
            scale then runs dark to light).
    """

    kind: Literal["sequential", "diverging"]
    label: str
    reverse: bool = False


@dataclass(frozen=True)
class Points(PlotElement):
    """
    Markers at x/y positions.

    Filled with `color`, unless `values` and `scale` are given: then each marker
    is filled by its value and a color bar is drawn. `linewidth` is the marker
    outline width, drawn in `edgecolor`.

    Attributes:
        x: x-coordinate of each marker.
        y: y-coordinate of each marker; same length as `x`.
        values: Value coloring each marker, or None to use `color`.
        scale: How `values` map to colors; required with `values`.
        marker: Matplotlib marker style.
        size: Marker area, in points squared.
        edgecolor: Outline color, or None for the fill color.
        filled: False to draw outlines only (in `color`).
        label: Legend entry, or None for no entry.
    """

    x: Sequence[float]
    y: Sequence[float]
    values: Sequence[float] | None = None
    scale: ColorScale | None = None
    marker: str = "o"
    size: float = 40.0
    edgecolor: str | None = None
    filled: bool = True
    label: str | None = None

    def __post_init__(self) -> None:
        if (self.values is None) != (self.scale is None):
            raise ValueError("Points need both `values` and `scale`, or neither.")


@dataclass(frozen=True)
class Circle(PlotElement):
    """
    An unfilled circle.

    Attributes:
        center: `(x, y)` of the centre.
        radius: Radius, in data units (a circle only with equal axis scaling).
        linestyle: Stroke pattern.
        label: Legend entry, or None for no entry.
    """

    center: tuple[float, float]
    radius: float
    linestyle: LineStyle = "dashed"
    label: str | None = None


@dataclass(frozen=True)
class Arrow(PlotElement):
    """
    A straight arrow from `start` to `end`.

    Attributes:
        start: `(x, y)` of the tail.
        end: `(x, y)` of the head.
    """

    start: tuple[float, float]
    end: tuple[float, float]

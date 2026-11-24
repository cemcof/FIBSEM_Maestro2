# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle as CirclePatch

from fibsem_maestro.logging.image.plot_element import (
    Arrow,
    Bars,
    Circle,
    ColorScale,
    Curve,
    HorizontalLine,
    Points,
    VerticalLine,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.path import Path as MarkerPath

    from fibsem_maestro.logging.image.plot import Panel, Plot

# one hue, light to dark (magnitude)
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "maestro_sequential", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)
# two hues around a neutral grey (sign): negative red, positive blue
_DIVERGING = LinearSegmentedColormap.from_list(
    "maestro_diverging", ["#b8312f", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"]
)

# size of a single-panel plot (Matplotlib's default) and of one cell of a layout;
# a single column keeps the full default width
_SINGLE_SIZE = (6.4, 4.8)
_CELL_SIZE = (5.0, 3.4)

# markers drawn as strokes only; Matplotlib colors them by the face color
_STROKE_MARKERS = frozenset({"+", "x", "|", "_", "1", "2", "3", "4"})

# more x tick labels than this are written vertically
_MAX_HORIZONTAL_TICKS = 12

# largest marker area in a legend, in points squared
_LEGEND_MARKER_SIZE = 60.0


def render_plot(plot: Plot) -> Figure:
    """
    Draw a plot into a new figure.

    The figure is not registered with pyplot, so it does not need closing.

    Args:
        plot: The plot to draw.

    Returns:
        The figure, not yet laid out.

    Raises:
        TypeError: If an element's type has no renderer.
        ValueError: If Matplotlib rejects the data (e.g. x and y of different lengths).
    """
    rows = plot.rows
    n_rows, n_cols = len(rows), len(rows[0])
    if n_rows == n_cols == 1:
        size = _SINGLE_SIZE
    else:
        width = _SINGLE_SIZE[0] if n_cols == 1 else _CELL_SIZE[0] * n_cols
        size = (width, _CELL_SIZE[1] * n_rows)

    fig = Figure(figsize=size)
    axes = fig.subplot_mosaic([list(row) for row in rows])

    for key, panel in plot.panels.items():
        _draw_panel(fig, axes[key], panel)
    for key, panel in plot.panels.items():
        if panel.sharex is not None:
            axes[key].sharex(axes[panel.sharex])

    if plot.title:
        if len(plot.panels) == 1:
            next(iter(axes.values())).set_title(plot.title)
        else:
            fig.suptitle(plot.title)

    return fig


def _draw_panel(fig: Figure, ax: Axes, panel: Panel) -> None:
    """Draw one panel's elements and decorations into `ax`."""
    for element in panel.elements:
        match element:
            case Curve():
                x = range(len(element.y)) if element.x is None else element.x
                ax.plot(
                    x,
                    element.y,
                    color=element.color,
                    linewidth=element.linewidth,
                    marker=element.marker,
                    label=element.label,
                )
            case VerticalLine():
                ax.axvline(
                    x=element.x,
                    color=element.color,
                    linewidth=element.linewidth,
                    linestyle=element.linestyle,
                    label=element.label,
                )
            case HorizontalLine():
                ax.axhline(
                    y=element.y,
                    color=element.color,
                    linewidth=element.linewidth,
                    linestyle=element.linestyle,
                    label=element.label,
                )
            case Bars():
                ax.bar(
                    element.x,
                    element.heights,
                    width=element.width,
                    color=element.color,
                    linewidth=element.linewidth,
                    edgecolor=element.color,
                    label=element.label,
                )
            case Points():
                _draw_points(fig, ax, element)
            case Circle():
                ax.add_patch(
                    CirclePatch(
                        element.center,
                        element.radius,
                        fill=False,
                        edgecolor=element.color,
                        linewidth=element.linewidth,
                        linestyle=element.linestyle,
                        label=element.label,
                    )
                )
            case Arrow():
                ax.annotate(
                    "",
                    xy=element.end,
                    xytext=element.start,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": element.color,
                        "linewidth": element.linewidth,
                        "shrinkA": 0,
                        "shrinkB": 4,
                    },
                )
            case _:
                raise TypeError(f"Unsupported plot element: {type(element).__name__}.")

    if panel.title:
        ax.set_title(panel.title)
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    if panel.xticks is not None:
        rotation = 90 if len(panel.xticks) > _MAX_HORIZONTAL_TICKS else 0
        ax.set_xticks(
            [t.position for t in panel.xticks],
            [t.label for t in panel.xticks],
            rotation=rotation,
            fontsize="small",
        )
    if panel.invert_y:
        ax.invert_yaxis()
    if panel.equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if panel.legend and ax.get_legend_handles_labels()[0]:
        legend = ax.legend(fontsize="small")
        # large highlight markers (e.g. the ring around the best point) would
        # otherwise be drawn at full size in the legend
        for handle in legend.legend_handles:
            if isinstance(handle, PathCollection):
                handle.set_sizes(
                    [min(float(handle.get_sizes()[0]), _LEGEND_MARKER_SIZE)]
                )


def _draw_points(fig: Figure, ax: Axes, points: Points) -> None:
    """Draw markers, colored by value with a color bar if the points carry values."""
    if points.marker in _STROKE_MARKERS:
        ax.scatter(
            points.x,
            points.y,
            s=points.size,
            marker=_stroke_path(points.marker),
            facecolors="none",
            edgecolors=points.color,
            linewidths=points.linewidth,
            label=points.label,
            zorder=3,
            snap=False,
        )
        return

    if points.values is None or points.scale is None:
        ax.scatter(
            points.x,
            points.y,
            s=points.size,
            marker=points.marker,
            facecolors=points.color if points.filled else "none",
            edgecolors=points.edgecolor or points.color,
            linewidths=points.linewidth,
            label=points.label,
            zorder=3,
        )
        return

    cmap, norm = _color_mapping(points.scale, points.values)
    artist = ax.scatter(
        points.x,
        points.y,
        c=list(points.values),
        cmap=cmap,
        norm=norm,
        s=points.size,
        marker=points.marker,
        edgecolors=points.edgecolor or points.color,
        linewidths=points.linewidth,
        label=points.label,
        zorder=3,
    )
    fig.colorbar(artist, ax=ax, label=points.scale.label)


def _stroke_path(marker: str) -> MarkerPath:
    """
    Return the outline of a stroke marker, such as `+`, as a path.

    Args:
        marker: One of the stroke markers.

    Returns:
        The marker's path at Matplotlib's marker scale.
    """
    style = MarkerStyle(marker)
    return style.get_path().transformed(style.get_transform())


def _color_mapping(
    scale: ColorScale, values: Sequence[float]
) -> tuple[LinearSegmentedColormap, Normalize]:
    """Return the colormap and normalization of a color scale for the given values."""
    array = np.asarray(values, dtype=float)
    if scale.kind == "sequential":
        low, high = float(array.min()), float(array.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        cmap = _SEQUENTIAL.reversed() if scale.reverse else _SEQUENTIAL
        return cmap, Normalize(low, high)

    # symmetric around zero, so equal colors mean equal magnitudes on both sides
    extent = float(np.abs(array).max()) or 1.0
    cmap = _DIVERGING.reversed() if scale.reverse else _DIVERGING
    return cmap, TwoSlopeNorm(vcenter=0.0, vmin=-extent, vmax=extent)

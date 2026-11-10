# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from fibsem_maestro.autofunction.evaluation import (
    BasicEvaluation,
    InterleavedEvaluation,
)
from fibsem_maestro.autofunction.sweep_space import CircularSweepSpace, LinearSweepSpace
from fibsem_maestro.logging.image.plot import Panel, Plot, Tick
from fibsem_maestro.logging.image.plot_element import (
    Bars,
    Circle,
    ColorScale,
    Curve,
    HorizontalLine,
    PlotElement,
    Points,
    VerticalLine,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fibsem_maestro.autofunction.evaluation import (
        CandidateComparison,
        SweepEvaluation,
    )
    from fibsem_maestro.autofunction.sweep_space import SweepSpace
    from fibsem_maestro.autofunction.sweep_step import SweepPoint

_CYCLE_COLORS = ["#1f3a93", "#3d8fe0", "#7b3fbf", "#c2318f", "#e8473a", "#8c1d18"]
_BASE_COLOR = "orange"
_BEST_COLOR = "green"

_CANDIDATE_COLOR = "#2a78d6"
_NOT_COUNTED_COLOR = "#b5b4ae"
_GUIDE_COLOR = "#7a7974"

_POINT_SIZE = 70.0
"""Area of a swept point in the plane chart, in points squared."""

_BASE_RING_SIZE = 120.0
"""Area of the ring around a measured base."""


def af_chart(
    evaluation: SweepEvaluation, space: SweepSpace, base: SweepPoint | None
) -> Plot:
    """
    Build the chart of a finished sweep.

    Args:
        evaluation: What the sweeping strategy concluded, and from what.
        space: Geometry of the sweep.
        base: The value the sweep started from, if known.

    Returns:
        The chart.

    Raises:
        TypeError: If no chart exists for the evaluation and space.
    """
    match evaluation, space:
        case BasicEvaluation(), LinearSweepSpace():
            return _line_chart(evaluation, space, base)
        case BasicEvaluation(), CircularSweepSpace():
            return _circle_chart(evaluation, space, base)
        case InterleavedEvaluation(), _:
            return _interleaved_chart(evaluation, space)
        case _:
            raise TypeError(
                f"No chart for {type(evaluation).__name__} in {type(space).__name__}."
            )


def _line_chart(
    evaluation: BasicEvaluation, space: LinearSweepSpace, base: SweepPoint | None
) -> Plot:
    """Sharpness over the swept value, one curve per cycle; initial and best values marked."""
    elements: list[PlotElement] = []
    for rep, sharpness in evaluation.cycle_sharpness.items():
        points = sorted((value[0], s) for value, s in sharpness.items())
        elements.append(
            Curve(
                x=[v for v, _ in points],
                y=[s for _, s in points],
                color=_CYCLE_COLORS[rep % len(_CYCLE_COLORS)],
                linewidth=1.0,
                marker="o",
                label=f"cycle {rep + 1}",
            )
        )

    if base is not None:
        elements.append(
            VerticalLine(x=base[0], color=_BASE_COLOR, linewidth=1.0, label="base")
        )
    elements.append(
        VerticalLine(
            x=evaluation.best[0], color=_BEST_COLOR, linewidth=1.0, label="best"
        )
    )

    return Plot(
        panels={
            "main": Panel(
                elements=elements,
                xlabel=space.attributes[0],
                ylabel="Sharpness",
                legend=True,
            )
        },
        title="Autofunction criterion",
    )


def _circle_chart(
    evaluation: BasicEvaluation, space: CircularSweepSpace, base: SweepPoint | None
) -> Plot:
    """The swept plane colored by mean sharpness, and sharpness around the circle."""
    center = base if base is not None else _centroid(list(evaluation.mean_sharpness))

    plane = _plane_panel(
        space,
        center,
        evaluation.best,
        evaluation.mean_sharpness,
        ColorScale("sequential", "Mean sharpness"),
    )

    profile: list[PlotElement] = []
    for rep, sharpness in evaluation.cycle_sharpness.items():
        points = sorted(
            (space.angle_of(v, center), s)
            for v, s in sharpness.items()
            # exclude the center
            if not space.is_center(v, center)
        )

        # close the loop: the first point again at 360°
        if points and points[0][0] == 0.0:
            points.append((360.0, points[0][1]))

        profile.append(
            Curve(
                x=[a for a, _ in points],
                y=[s for _, s in points],
                color=_CYCLE_COLORS[rep % len(_CYCLE_COLORS)],
                linewidth=1.0,
                marker="o",
                label=f"cycle {rep + 1}",
            )
        )

    kept = space.is_center(evaluation.best, center)
    center_sharpness = [
        s for v, s in evaluation.mean_sharpness.items() if space.is_center(v, center)
    ]
    if center_sharpness:
        profile.append(
            HorizontalLine(
                y=center_sharpness[0],
                color=_BEST_COLOR if kept else _BASE_COLOR,
                linewidth=1.0,
                linestyle="dashed",
                label="center (best)" if kept else "center",
            )
        )
    if not kept:
        profile.append(
            VerticalLine(
                x=space.angle_of(evaluation.best, center),
                color=_BEST_COLOR,
                linewidth=1.0,
                label="best",
            )
        )

    angles = sorted(
        {
            space.angle_of(v, center)
            for v in evaluation.mean_sharpness
            if not space.is_center(v, center)
        }
    )

    return Plot(
        panels={
            "plane": plane,
            "profile": Panel(
                elements=profile,
                title="Sharpness around the circle",
                xlabel=f"Angle clockwise from +{space.attributes[1]} [°]",
                ylabel="Sharpness",
                xticks=[Tick(a, f"{a:.0f}°") for a in [*angles, 360.0]],
                legend=True,
            ),
        },
        layout=[["plane", "profile"]],
        title="Focus criterion (2D)",
    )


def _interleaved_chart(evaluation: InterleavedEvaluation, space: SweepSpace) -> Plot:
    """Sharpness in measurement order and the gain of every candidate over its base."""
    base, best = evaluation.base, evaluation.best
    moved = best != base
    ticks = [
        Tick(c.step, space.offset_label(c.value, base)) for c in evaluation.comparisons
    ]
    separators = _cycle_separators(evaluation)

    # measured sharpness: the base as a line (its drift is visible), candidates as dots
    measured: list[PlotElement] = [*separators]
    measured.append(
        Curve(
            x=[m.step for m in evaluation.base_measurements],
            y=[m.sharpness for m in evaluation.base_measurements],
            color=_BASE_COLOR,
            linewidth=1.0,
            marker="o",
            label="base",
        )
    )
    others = [c for c in evaluation.comparisons if not (moved and c.value == best)]
    chosen = [c for c in evaluation.comparisons if moved and c.value == best]
    for group, color, label in [
        (others, _CANDIDATE_COLOR, "candidate"),
        (chosen, _BEST_COLOR, "best"),
    ]:
        if group:
            measured.append(
                Points(
                    x=[c.step for c in group],
                    y=[c.sharpness for c in group],
                    color=color,
                    linewidth=0.0,
                    label=label,
                )
            )

    # gain over the neighbouring base, against the threshold
    gain: list[PlotElement] = [*separators]
    counted = [c for c in others if c.counted]
    not_counted = [c for c in others if not c.counted]
    for group, color, label in [
        (counted, _CANDIDATE_COLOR, "counted"),
        (not_counted, _NOT_COUNTED_COLOR, "not above min_diff"),
        (chosen, _BEST_COLOR, "best"),
    ]:
        if group:
            gain.append(
                Bars(
                    x=[c.step for c in group],
                    heights=[c.gain for c in group],
                    color=color,
                    linewidth=0.0,
                    label=label,
                )
            )
    gain.append(HorizontalLine(y=0.0, color=_GUIDE_COLOR, linewidth=0.8))
    gain.append(
        HorizontalLine(
            y=evaluation.min_diff,
            color=_GUIDE_COLOR,
            linewidth=1.0,
            linestyle="dashed",
            label="min_diff",
        )
    )

    tick_meaning = (
        "angle clockwise from +y"
        if isinstance(space, CircularSweepSpace)
        else "offset from base"
    )
    panels = {
        "sharpness": Panel(
            elements=measured,
            title="Measured sharpness",
            ylabel="Sharpness",
            xticks=ticks,
            legend=True,
            sharex="gain",
        ),
        "gain": Panel(
            elements=gain,
            title="Gain over the neighbouring base",
            xlabel=f"Candidate ({tick_meaning})",
            ylabel="Sharpness gain",
            xticks=ticks,
            legend=True,
        ),
    }
    layout = [["sharpness"], ["gain"]]

    if isinstance(space, CircularSweepSpace):
        panels["plane"] = _plane_panel(
            space,
            base,
            best,
            _mean_gain(evaluation.comparisons),
            ColorScale("diverging", "Mean gain over base"),
        )
        layout = [["plane", "sharpness"], ["plane", "gain"]]

    return Plot(panels=panels, layout=layout, title="Focus criterion (interleaved)")


def _plane_panel(
    space: CircularSweepSpace,
    center: SweepPoint,
    best: SweepPoint,
    values: Mapping[SweepPoint, float],
    scale: ColorScale,
) -> Panel:
    """
    The swept plane: the circle, its points colored by value, the base and the best.

    Args:
        space: Geometry of the sweep.
        center: The base value, at the center of the circle.
        best: The selected value.
        values: Value coloring each swept point.
        scale: How the values map to colors.
    """
    elements: list[PlotElement] = [
        Circle(
            center=(center[0], center[1]),
            radius=space.radius,
            color=_GUIDE_COLOR,
            linewidth=0.8,
        ),
    ]
    if values:
        elements.append(
            Points(
                x=[v[0] for v in values],
                y=[v[1] for v in values],
                values=list(values.values()),
                scale=scale,
                color=_GUIDE_COLOR,
                linewidth=0.5,
                size=_POINT_SIZE,
            )
        )

    if any(space.is_center(v, center) for v in values):
        # the centre was measured: a ring around its colored point keeps the color
        # visible, where a `+` would cover it
        base_marker = Points(
            x=[center[0]],
            y=[center[1]],
            color=_BASE_COLOR,
            linewidth=2.0,
            size=_BASE_RING_SIZE,
            filled=False,
            label="base",
        )
    else:
        # center was not measured, use + to indicate it
        base_marker = Points(
            x=[center[0]],
            y=[center[1]],
            color=_BASE_COLOR,
            linewidth=2.0,
            marker="+",
            size=150.0,
            label="base",
        )
    elements.append(base_marker)

    elements.append(
        Points(
            x=[best[0]],
            y=[best[1]],
            color=_BEST_COLOR,
            linewidth=2.0,
            size=260.0,
            filled=False,
            label="best",
        )
    )

    return Panel(
        elements=elements,
        title="Swept plane",
        xlabel=space.attributes[0],
        ylabel=space.attributes[1],
        equal_aspect=True,
        legend=True,
    )


def _cycle_separators(evaluation: InterleavedEvaluation) -> list[PlotElement]:
    """Dotted lines between consecutive sweep cycles, in step coordinates."""
    first_step: dict[int, int] = {}
    for step, rep in [
        *((m.step, m.repetition) for m in evaluation.base_measurements),
        *((c.step, c.repetition) for c in evaluation.comparisons),
    ]:
        first_step[rep] = min(step, first_step.get(rep, step))

    return [
        VerticalLine(
            x=step - 0.5, color=_GUIDE_COLOR, linewidth=0.8, linestyle="dotted"
        )
        for rep, step in sorted(first_step.items())
        if rep != min(first_step)
    ]


def _mean_gain(comparisons: Sequence[CandidateComparison]) -> dict[SweepPoint, float]:
    """Mean gain of every candidate value over all its comparisons, counted or not."""
    gains: dict[SweepPoint, list[float]] = defaultdict(list)
    for c in comparisons:
        gains[c.value].append(c.gain)
    return {v: float(np.mean(g)) for v, g in gains.items()}


def _centroid(values: Sequence[SweepPoint]) -> SweepPoint:
    """Mean of the given points; the center of a fully measured circle."""
    return tuple(float(c) for c in np.mean(np.asarray(values), axis=0))

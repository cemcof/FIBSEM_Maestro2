# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from fibsem_maestro.action_context.memory import MemoryActionContext
from fibsem_maestro.autofunction.af_chart import af_chart
from fibsem_maestro.autofunction.autofunction import Autofunction
from fibsem_maestro.autofunction.evaluation import InterleavedEvaluation
from fibsem_maestro.autofunction.result import AutofunctionResult
from fibsem_maestro.autofunction.sweep_space import CircularSweepSpace, LinearSweepSpace
from fibsem_maestro.autofunction.sweep_step import SweepPoint, SweepStep
from fibsem_maestro.autofunction.sweeping import Sweeping
from fibsem_maestro.autofunction.sweeping_strategy import (
    BasicSweepingStrategy,
    InterleavedSweepingStrategy,
    SweepingStrategy,
)
from fibsem_maestro.logging.image.plot import Panel, Plot
from fibsem_maestro.logging.image.plot_element import Bars, Curve, HorizontalLine
from fibsem_maestro.logging.image.plot_renderer import render_plot
from fibsem_maestro.logging.text.memory import MemoryTextLogger
from fibsem_maestro.settings.autofunction_settings import (
    AutofunctionSettings,
    BasicMode,
)
from fibsem_maestro.settings.criterion_settings import CriterionSettings
from fibsem_maestro.settings.sweeping_settings import (
    BasicStrategy as BasicStrategySettings,
)
from fibsem_maestro.settings.sweeping_settings import (
    CircularSweeping as CircularSweepingSettings,
)
from fibsem_maestro.settings.sweeping_settings import (
    InterleavedStrategy as InterleavedStrategySettings,
)
from fibsem_maestro.workflow.actions import Actions

if TYPE_CHECKING:
    from fibsem_maestro.microscope.microscope import Microscope

CIRCLE = CircularSweepSpace("stigmator_x", "stigmator_y", radius=2.0, steps=4)
BASE_2D: SweepPoint = (1.0, 1.0)


def _approx_points(points: list[SweepPoint]) -> list[Any]:
    return [pytest.approx(p, abs=1e-12) for p in points]


def _results(
    strategy: SweepingStrategy,
    space: CircularSweepSpace | LinearSweepSpace,
    base: SweepPoint,
    sharpness: Any,
    cycles: int = 1,
) -> list[AutofunctionResult]:
    """Results of a sweep scored by `sharpness(value)`."""
    values = [
        (rep, v) for rep in range(cycles) for v in strategy.generate(space, base, rep)
    ]
    return [
        AutofunctionResult(sharpness(v), SweepStep(rep, v, i))
        for i, (rep, v) in enumerate(values)
    ]


def _sharpest_at(target: SweepPoint) -> Any:
    return lambda v: -math.dist(v, target)


def test_circle_starts_at_plus_radius_on_y_and_spaces_points_evenly() -> None:
    points = CIRCLE.candidates(BASE_2D, reverse=False)

    assert points == _approx_points(
        [(1.0, 3.0), (3.0, 1.0), BASE_2D, (1.0, -1.0), (-1.0, 1.0)]
    )


def test_circle_reversed_keeps_the_first_point_and_turns_the_other_way() -> None:
    points = CIRCLE.candidates(BASE_2D, reverse=True)

    assert points == _approx_points(
        [(1.0, 3.0), (-1.0, 1.0), BASE_2D, (1.0, -1.0), (3.0, 1.0)]
    )


def test_linear_space_spans_the_range_around_the_base() -> None:
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 3)

    assert space.candidates((10.0,), reverse=False) == [(8.0,), (10.0,), (12.0,)]
    assert space.candidates((10.0,), reverse=True) == [(12.0,), (10.0,), (8.0,)]


def test_interleaved_measures_the_base_before_every_circle_point() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings())

    values = strategy.generate(CIRCLE, BASE_2D, 0)

    # the center is not a candidate here: the base is measured anyway
    circle = [p for p in CIRCLE.candidates(BASE_2D, reverse=False) if p != BASE_2D]
    assert values[0::2] == [BASE_2D] * 4
    assert values[1::2] == _approx_points(circle)


def test_basic_2d_picks_the_sharpest_circle_point() -> None:
    strategy = BasicSweepingStrategy(BasicStrategySettings())
    results = _results(strategy, CIRCLE, BASE_2D, _sharpest_at((5.0, 1.0)))

    assert strategy.evaluate(results).best == pytest.approx((3.0, 1.0))


def test_interleaved_2d_keeps_the_base_when_no_point_is_clearly_sharper() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.1))
    results = _results(strategy, CIRCLE, BASE_2D, _sharpest_at(BASE_2D))

    assert strategy.evaluate(results).best == BASE_2D


def test_interleaved_2d_moves_to_a_clearly_sharper_point() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.1))
    results = _results(strategy, CIRCLE, BASE_2D, _sharpest_at((1.0, -4.0)))

    assert strategy.evaluate(results).best == pytest.approx((1.0, -1.0))


def test_interleaved_keeps_a_candidate_close_to_a_large_base() -> None:
    """A relative tolerance on the absolute value would drop 20 nm around 4 mm."""
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings())
    space = LinearSweepSpace("working_distance", (-20.0, 20.0), 3)

    values = strategy.generate(space, (4.0e6,), 0)

    assert values == [(4.0e6,), (4.0e6 - 20.0,), (4.0e6,), (4.0e6 + 20.0,)]


def test_circle_edge_direction_points_from_base_to_best() -> None:
    direction = CIRCLE.edge_direction((3.0, 1.0), BASE_2D, [])

    assert direction == pytest.approx((1.0, 0.0))


def test_circle_has_no_edge_when_the_base_is_kept() -> None:
    assert CIRCLE.edge_direction(BASE_2D, BASE_2D, []) is None


@pytest.mark.parametrize(
    ("best", "direction"), [((12.0,), (1.0,)), ((8.0,), (-1.0,)), ((11.0,), None)]
)
def test_line_edge_direction_is_the_side_of_the_range(
    best: SweepPoint, direction: tuple[float, ...] | None
) -> None:
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 5)
    swept = space.candidates((10.0,), reverse=False)

    assert space.edge_direction(best, (10.0,), swept) == direction


class _FakeBeam:
    def __init__(self) -> None:
        self.stigmator_x = 0.0
        self.stigmator_y = 0.0

    def limits(self, name: str) -> tuple[float, float]:
        _ = name
        return (-1.0, 1.0)


def _sweeping(settings: CircularSweepingSettings, beam: _FakeBeam) -> Sweeping:
    return Sweeping(beam, settings, MemoryTextLogger(lambda: 0))  # ty: ignore[invalid-argument-type]


def test_sweeping_sets_both_attributes_of_a_2d_point() -> None:
    beam = _FakeBeam()
    sweeping = _sweeping(CircularSweepingSettings(radius=0.1, steps=4), beam)

    sweeping.set_attribute_value((0.2, -0.3))

    assert (beam.stigmator_x, beam.stigmator_y) == (0.2, -0.3)
    assert sweeping.get_attribute_value() == (0.2, -0.3)


def test_sweeping_skips_points_with_a_component_out_of_limits() -> None:
    beam = _FakeBeam()
    beam.stigmator_y = 0.95
    sweeping = _sweeping(CircularSweepingSettings(radius=0.1, steps=4), beam)

    values = [s.value for s in sweeping.sweep()]

    # (0, 1.05) exceeds the y limit; the other three points and the center remain
    assert values == _approx_points(
        [(0.1, 0.95), (0.0, 0.95), (0.0, 0.85), (-0.1, 0.95)]
    )


def test_2d_sweep_rejects_the_same_attribute_on_both_axes() -> None:
    with pytest.raises(ValidationError):
        CircularSweepingSettings(
            target_attribute_x="stigmator_x", target_attribute_y="stigmator_x"
        )


@pytest.mark.parametrize("field", [{"radius": 0.0}, {"steps": 2}])
def test_2d_sweep_rejects_a_degenerate_circle(field: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        CircularSweepingSettings(**field)  # ty: ignore[invalid-argument-type]


def test_interleaved_chart_shows_every_gain_and_the_threshold() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.3))
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 5)
    evaluation = strategy.evaluate(
        _results(strategy, space, (0.0,), _sharpest_at((1.5,)), cycles=2)
    )
    assert isinstance(evaluation, InterleavedEvaluation)

    gain = af_chart(evaluation, space, (0.0,)).panels["gain"]

    bars = [e for e in gain.elements if isinstance(e, Bars)]
    plotted = sorted(h for b in bars for h in b.heights)
    assert plotted == pytest.approx(sorted(c.gain for c in evaluation.comparisons))
    assert any(isinstance(e, HorizontalLine) and e.y == 0.3 for e in gain.elements)


def test_basic_1d_chart_draws_one_curve_per_cycle() -> None:
    strategy = BasicSweepingStrategy(BasicStrategySettings())
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 5)
    evaluation = strategy.evaluate(
        _results(strategy, space, (0.0,), _sharpest_at((1.0,)), cycles=3)
    )

    plot = af_chart(evaluation, space, (0.0,))

    (panel,) = plot.panels.values()
    assert len([e for e in panel.elements if isinstance(e, Curve)]) == 3


@pytest.mark.parametrize(
    "strategy",
    [
        BasicSweepingStrategy(BasicStrategySettings()),
        InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.0)),
    ],
)
def test_2d_charts_show_the_swept_plane(strategy: SweepingStrategy) -> None:
    evaluation = strategy.evaluate(
        _results(strategy, CIRCLE, BASE_2D, _sharpest_at((4.0, 1.0)), cycles=2)
    )

    plot = af_chart(evaluation, CIRCLE, BASE_2D)

    assert plot.panels["plane"].equal_aspect
    render_plot(plot)  # every element has a renderer


def test_plot_rejects_a_layout_that_does_not_place_every_panel() -> None:
    with pytest.raises(ValueError):
        Plot(panels={"a": Panel(elements=[]), "b": Panel(elements=[])}, layout=[["a"]])


def test_2d_autofunction_moves_the_stigmator_onto_the_circle(
    simulated_microscope: Microscope,
) -> None:
    beam = simulated_microscope.electron_beam
    beam.stigmator_x, beam.stigmator_y = 0.1, -0.1
    settings = AutofunctionSettings(
        mode=BasicMode(
            sweeping=CircularSweepingSettings(radius=0.01, steps=6),
            criterion=CriterionSettings(),
        )
    )
    ctx = MemoryActionContext("af")
    autofunction = Autofunction("af", simulated_microscope, settings, ctx, Actions())

    autofunction.test()

    moved = math.dist((beam.stigmator_x, beam.stigmator_y), (0.1, -0.1))
    assert moved == pytest.approx(0.01)
    (saved,) = [p for plots in ctx.image_logger.plots.values() for p in plots]  # ty: ignore[unresolved-attribute]
    assert saved.filename == "af_curve.png"
    assert "plane" in saved.plot.panels


@pytest.mark.parametrize("steps", [3, 4, 5, 8])
def test_circle_measures_its_center_once_halfway(steps: int) -> None:
    space = CircularSweepSpace("stigmator_x", "stigmator_y", radius=2.0, steps=steps)

    for reverse in (False, True):
        points = space.candidates(BASE_2D, reverse=reverse)
        centers = [i for i, p in enumerate(points) if space.is_center(p, BASE_2D)]
        assert centers == [steps // 2]
        assert len(points) == steps + 1


def test_circle_labels_its_center() -> None:
    assert CIRCLE.offset_label(BASE_2D, BASE_2D) == "center"
    assert CIRCLE.offset_label((3.0, 1.0), BASE_2D) == "90°"


def test_basic_2d_keeps_the_base_when_the_center_is_sharpest() -> None:
    """Without the center among the candidates, the base would always move away."""
    strategy = BasicSweepingStrategy(BasicStrategySettings())
    results = _results(strategy, CIRCLE, BASE_2D, _sharpest_at(BASE_2D), cycles=2)

    evaluation = strategy.evaluate(results)

    assert evaluation.best == BASE_2D
    assert CIRCLE.edge_direction(evaluation.best, BASE_2D, []) is None


@pytest.mark.parametrize(
    ("sharpest", "label"), [((4.0, 1.0), "center"), (BASE_2D, "center (best)")]
)
def test_basic_2d_chart_draws_the_center_as_a_line(
    sharpest: SweepPoint, label: str
) -> None:
    strategy = BasicSweepingStrategy(BasicStrategySettings())
    evaluation = strategy.evaluate(
        _results(strategy, CIRCLE, BASE_2D, _sharpest_at(sharpest))
    )

    profile = af_chart(evaluation, CIRCLE, BASE_2D).panels["profile"]

    (curve,) = [e for e in profile.elements if isinstance(e, Curve)]
    # four circle points, the first again at 360°; the center has no angle
    assert curve.x == pytest.approx([0.0, 90.0, 180.0, 270.0, 360.0])
    (line,) = [e for e in profile.elements if isinstance(e, HorizontalLine)]
    assert line.y == pytest.approx(evaluation.mean_sharpness[BASE_2D])
    assert line.label == label

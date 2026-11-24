# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import pytest

from fibsem_maestro.autofunction import InterleavedSweepingStrategy
from fibsem_maestro.autofunction.result import AutofunctionResult
from fibsem_maestro.autofunction.sweep_space import LinearSweepSpace
from fibsem_maestro.autofunction.sweep_step import SweepPoint, SweepStep
from fibsem_maestro.autofunction.sweeping_strategy import BasicSweepingStrategy
from fibsem_maestro.criterion.sharpness import Better, Sharpness
from fibsem_maestro.settings.sweeping_settings import (
    BasicStrategy as BasicStrategySettings,
)
from fibsem_maestro.settings.sweeping_settings import (
    InterleavedStrategy as InterleavedStrategySettings,
)


def _interleaved_results(
    values: list[SweepPoint], sharpness: dict[float, float]
) -> list[AutofunctionResult]:
    return [
        AutofunctionResult(
            Sharpness(sharpness[v[0]], Better.HIGHER), SweepStep(0, v, i)
        )
        for i, v in enumerate(values)
    ]


def test_interleaved_min_diff_is_an_absolute_sharpness_difference() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.5))
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 3)
    values = strategy.generate(space, (0.0,), 0)
    # +2 beats the base by 0.4 (< min_diff) although that is 4 % of the base
    sharpness = {0.0: 10.0, -2.0: 2.0, 2.0: 10.4}
    assert strategy.evaluate(_interleaved_results(values, sharpness)).best == (0.0,)
    # +2 beats the base by 0.6 (> min_diff) although that is only 0.06 % of the base
    sharpness = {0.0: 1000.0, -2.0: 2.0, 2.0: 1000.6}
    assert strategy.evaluate(_interleaved_results(values, sharpness)).best == (2.0,)


def _results(
    values: list[SweepPoint], nm: dict[float, float]
) -> list[AutofunctionResult]:
    """Results of a lower-is-sharper metric, e.g. the FRC resolution in nm."""
    return [
        AutofunctionResult(Sharpness(nm[v[0]], Better.LOWER, "nm"), SweepStep(0, v, i))
        for i, v in enumerate(values)
    ]


def test_basic_picks_the_least_value_of_a_lower_is_sharper_metric() -> None:
    strategy = BasicSweepingStrategy(BasicStrategySettings())
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 3)
    values = strategy.generate(space, (0.0,), 0)

    evaluation = strategy.evaluate(_results(values, {-2.0: 6.0, 0.0: 5.0, 2.0: 4.0}))

    assert evaluation.best == (2.0,)


def test_interleaved_gain_of_a_lower_is_sharper_metric_is_the_decrease_in_nm() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.5))
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 3)
    values = strategy.generate(space, (0.0,), 0)

    evaluation = strategy.evaluate(_results(values, {0.0: 5.0, -2.0: 6.0, 2.0: 4.4}))

    gains = {c.value: c.gain for c in evaluation.comparisons}
    assert gains[(2.0,)] == pytest.approx(0.6)
    assert gains[(-2.0,)] == pytest.approx(-1.0)
    assert evaluation.best == (2.0,)


def test_interleaved_keeps_the_base_unless_the_decrease_exceeds_min_diff() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.5))
    space = LinearSweepSpace("working_distance", (-2.0, 2.0), 3)
    values = strategy.generate(space, (0.0,), 0)

    evaluation = strategy.evaluate(_results(values, {0.0: 5.0, -2.0: 6.0, 2.0: 4.6}))

    assert evaluation.best == (0.0,)

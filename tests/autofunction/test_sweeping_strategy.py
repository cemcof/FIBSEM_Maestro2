# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from fibsem_maestro.autofunction import InterleavedSweepingStrategy
from fibsem_maestro.autofunction.result import AutofunctionResult
from fibsem_maestro.autofunction.sweep_space import LinearSweepSpace
from fibsem_maestro.autofunction.sweep_step import SweepPoint, SweepStep
from fibsem_maestro.settings.sweeping_settings import (
    InterleavedStrategy as InterleavedStrategySettings,
)


def _interleaved_results(
    values: list[SweepPoint], sharpness: dict[float, float]
) -> list[AutofunctionResult]:
    return [
        AutofunctionResult(sharpness[v[0]], SweepStep(0, v, i))
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

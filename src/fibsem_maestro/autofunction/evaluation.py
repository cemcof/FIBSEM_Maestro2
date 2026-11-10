# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fibsem_maestro.autofunction.sweep_step import SweepPoint


@dataclass(frozen=True)
class BasicEvaluation:
    """
    Outcome of the basic strategy: the value with the highest mean sharpness.

    Attributes:
        best: The selected value.
        mean_sharpness: Mean sharpness of each swept value over all cycles.
        cycle_sharpness: Mean sharpness of each swept value, per cycle
            (repetition index -> value -> sharpness).
    """

    best: SweepPoint
    mean_sharpness: Mapping[SweepPoint, float]
    cycle_sharpness: Mapping[int, Mapping[SweepPoint, float]]


@dataclass(frozen=True)
class BaseMeasurement:
    """
    Sharpness of one step measured at the base value.

    Attributes:
        step: Global index of the sweep step.
        repetition: Cycle the step belongs to.
        sharpness: Mean sharpness of the step's results.
    """

    step: int
    repetition: int
    sharpness: float


@dataclass(frozen=True)
class CandidateComparison:
    """
    One candidate step compared with the base measured around it.

    Attributes:
        step: Global index of the sweep step.
        repetition: Cycle the step belongs to.
        value: The candidate value.
        sharpness: Mean sharpness of the step's results.
        baseline: Mean sharpness of the nearest base steps before and after.
        gain: `sharpness - baseline`.
        counted: True if the gain exceeded `min_diff`, so it entered the selection.
    """

    step: int
    repetition: int
    value: SweepPoint
    sharpness: float
    baseline: float
    gain: float
    counted: bool


@dataclass(frozen=True)
class InterleavedEvaluation:
    """
    Outcome of the interleaved strategy.

    Attributes:
        best: The selected value: the candidate with the highest mean counted
            gain, or the base if no gain was counted.
        base: The base value.
        min_diff: Gain a candidate had to exceed to count (criterion units).
        base_measurements: Every base step, in measurement order.
        comparisons: Every candidate step that had a base step next to it,
            in measurement order.
    """

    best: SweepPoint
    base: SweepPoint
    min_diff: float
    base_measurements: tuple[BaseMeasurement, ...]
    comparisons: tuple[CandidateComparison, ...]


SweepEvaluation = BasicEvaluation | InterleavedEvaluation
"""Outcome of any sweeping strategy."""

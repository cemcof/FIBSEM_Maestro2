# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from fibsem_maestro.autofunction.error import AutofunctionError
from fibsem_maestro.autofunction.evaluation import (
    BaseMeasurement,
    BasicEvaluation,
    CandidateComparison,
    InterleavedEvaluation,
)
from fibsem_maestro.core.registry import Registry

if TYPE_CHECKING:
    from fibsem_maestro.autofunction.evaluation import SweepEvaluation
    from fibsem_maestro.autofunction.result import AutofunctionResult
    from fibsem_maestro.autofunction.sweep_space import SweepSpace
    from fibsem_maestro.autofunction.sweep_step import SweepPoint
    from fibsem_maestro.settings.sweeping_settings import (
        BasicStrategy as BasicStrategySettings,
    )
    from fibsem_maestro.settings.sweeping_settings import (
        InterleavedStrategy as InterleavedStrategySettings,
    )
    from fibsem_maestro.settings.sweeping_settings import (
        SweepingStrategySettings,
    )

# registry of sweeping strategies
SWEEPING_STRATEGIES = Registry[type["SweepingStrategy"]]("sweeping strategy")


class SweepingStrategy(ABC):
    """
    Abstract base class for autofunction sweeping strategies.

    A strategy decides the order in which the candidates of a sweep space are
    measured (and whether the base is measured too), and how the best value is
    chosen from the results. It does not know the geometry of the space, so every
    strategy works for 1D and 2D sweeps alike.
    """

    @abstractmethod
    def __init__(self, settings: SweepingStrategySettings):
        """
        Initialize the sweeping strategy.

        Args:
            settings: Strategy-specific configuration parameters.
        """

    @abstractmethod
    def generate(
        self, space: SweepSpace, base: SweepPoint, repetition: int
    ) -> list[SweepPoint]:
        """
        Generate the values to measure in one sweep cycle.

        Args:
            space: Geometry placing the candidates around the base.
            base: Current values of the swept attributes (absolute).
            repetition: Index of the sweep cycle.

        Returns:
            The values to measure, in measurement order (absolute).
        """

    @abstractmethod
    def evaluate(self, results: list[AutofunctionResult]) -> SweepEvaluation:
        """
        Choose the best value from the results of a sweep.

        Args:
            results: Results collected during the sweep; several results may
                share a step (line mode), and failed steps may be missing.

        Returns:
            The chosen value together with the numbers it was chosen from.

        Raises:
            AutofunctionError: If no results are provided.
        """


@SWEEPING_STRATEGIES.register("basic")
class BasicSweepingStrategy(SweepingStrategy):
    """Measures the candidates only and picks the one with the highest mean sharpness."""

    def __init__(self, settings: BasicStrategySettings):
        self._settings = settings

    def generate(
        self, space: SweepSpace, base: SweepPoint, repetition: int
    ) -> list[SweepPoint]:
        """
        Return the candidates, reversing the traversal on every odd cycle.

        Alternating the direction keeps a slow drift during the sweep from
        favouring the candidates measured first.
        """
        return space.candidates(base, reverse=repetition % 2 == 1)

    def evaluate(self, results: list[AutofunctionResult]) -> BasicEvaluation:
        """
        Select the value with the highest mean sharpness over all its results.

        The base can only be selected if the space includes it among the
        candidates: the centre of a circle always, a linear sweep if it has an
        odd number of steps over a symmetric range.
        """
        if not results:
            raise AutofunctionError("No autofunction results provided")

        by_value: dict[SweepPoint, list[float]] = defaultdict(list)
        by_cycle: dict[int, dict[SweepPoint, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in sorted(results, key=lambda r: r.sweep.index):
            by_value[r.sweep.value].append(r.sharpness)
            by_cycle[r.sweep.repetition][r.sweep.value].append(r.sharpness)

        mean_sharpness = {v: float(np.mean(s)) for v, s in by_value.items()}

        return BasicEvaluation(
            best=max(mean_sharpness, key=lambda v: mean_sharpness[v]),
            mean_sharpness=mean_sharpness,
            cycle_sharpness={
                rep: {v: float(np.mean(s)) for v, s in values.items()}
                for rep, values in by_cycle.items()
            },
        )


@SWEEPING_STRATEGIES.register("interleaved")
class InterleavedSweepingStrategy(SweepingStrategy):
    """Measures the base between every two candidates; switches only if clearly sharper."""

    def __init__(self, settings: InterleavedStrategySettings):
        self._settings = settings

    def generate(
        self, space: SweepSpace, base: SweepPoint, repetition: int
    ) -> list[SweepPoint]:
        """
        Return the candidates with the base measured before each of them.

            [base, c0, base, c1, base, c2, ...]

        The repetition index is ignored. A candidate coinciding with the base
        (the center of a circle, the middle of a linear sweep) is dropped: the
        base is measured anyway, and it is the default choice.
        """
        _ = repetition
        candidates = space.candidates(base, reverse=False)
        distances = [float(np.linalg.norm(np.subtract(c, base))) for c in candidates]
        # tolerance from the sweep's own scale: a relative tolerance on the
        # absolute value would drop real candidates near a large base (WD in nm)
        tolerance = 1e-9 * max(distances, default=0.0)

        values: list[SweepPoint] = []
        for candidate, distance in zip(candidates, distances):
            if distance <= tolerance:
                continue
            values += [base, candidate]
        return values

    def evaluate(self, results: list[AutofunctionResult]) -> InterleavedEvaluation:
        """
        Select the candidate that is clearly sharper than the base around it.

        Each candidate step is compared with the mean sharpness of the nearest
        base steps before and after it, so that slow drift during the sweep
        cancels out. Results sharing a step (line mode) are averaged first. A
        candidate counts only if its sharpness exceeds that baseline by more
        than `min_diff` (absolute difference, in the units of the selected
        criterion). Missing results (failed jobs) are tolerated.

        The base value is the value measured most often; ties go to the
        earliest step, which is a base step by construction.

        Returns:
            The candidate with the highest mean counted gain, or the base value
            if no candidate is clearly sharper, with every comparison made.
        """
        if not results:
            raise AutofunctionError("No autofunction results provided")

        # mean sharpness per sweep step
        by_step: dict[int, list[AutofunctionResult]] = defaultdict(list)
        for r in results:
            by_step[r.sweep.index].append(r)
        steps = sorted(by_step)
        value_of = {i: by_step[i][0].sweep.value for i in steps}
        repetition_of = {i: by_step[i][0].sweep.repetition for i in steps}
        sharpness_of = {
            i: float(np.mean([r.sharpness for r in by_step[i]])) for i in steps
        }

        # the base is measured between every pair of candidates
        counts: dict[SweepPoint, int] = defaultdict(int)
        for i in steps:
            counts[value_of[i]] += 1
        base = max(counts, key=lambda v: (counts[v], -_first_step(value_of, v)))
        base_steps = [i for i in steps if value_of[i] == base]

        comparisons: list[CandidateComparison] = []
        for i in steps:
            if value_of[i] == base:
                continue

            before = [b for b in base_steps if b < i]
            after = [b for b in base_steps if b > i]
            neighbours = ([before[-1]] if before else []) + (
                [after[0]] if after else []
            )
            if not neighbours:
                continue

            baseline = float(np.mean([sharpness_of[b] for b in neighbours]))
            gain = sharpness_of[i] - baseline
            comparisons.append(
                CandidateComparison(
                    step=i,
                    repetition=repetition_of[i],
                    value=value_of[i],
                    sharpness=sharpness_of[i],
                    baseline=baseline,
                    gain=gain,
                    counted=gain > self._settings.min_diff,
                )
            )

        return InterleavedEvaluation(
            best=_best_counted(comparisons, default=base),
            base=base,
            min_diff=self._settings.min_diff,
            base_measurements=tuple(
                BaseMeasurement(b, repetition_of[b], sharpness_of[b])
                for b in base_steps
            ),
            comparisons=tuple(comparisons),
        )


def _best_counted(
    comparisons: list[CandidateComparison], default: SweepPoint
) -> SweepPoint:
    """
    Return the value with the highest mean gain over its counted comparisons.

    Args:
        comparisons: Every comparison made during the sweep.
        default: The value returned if no comparison counted.
    """
    # a value is measured once per cycle, so it may have several counted gains
    gains: dict[SweepPoint, list[float]] = defaultdict(list)
    for c in comparisons:
        if c.counted:
            gains[c.value].append(c.gain)

    if not gains:
        return default

    return max(gains, key=lambda v: float(np.mean(gains[v])))


def _first_step(value_of: dict[int, SweepPoint], value: SweepPoint) -> int:
    """Return the first sweep step measured at `value`."""
    return min(i for i, v in value_of.items() if v == value)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from fibsem_maestro.settings.dynamic_default import required_value
from fibsem_maestro.settings.sweeping_settings import (
    CircularSweeping as CircularSweepingSettings,
)
from fibsem_maestro.settings.sweeping_settings import (
    LinearSweeping as LinearSweepingSettings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.autofunction.sweep_step import SweepDirection, SweepPoint
    from fibsem_maestro.settings.dynamic_default import DefaultContext
    from fibsem_maestro.settings.sweeping_settings import SweepingSettings


class SweepSpace(ABC):
    """
    Geometry of a sweep: which attributes it changes and where its candidates lie.

    The space only places candidates around a base value. The order in which they
    are measured and how the best one is chosen belong to the sweeping strategy.
    """

    @property
    @abstractmethod
    def attributes(self) -> tuple[str, ...]:
        """Names of the swept beam attributes; a sweep point has one value per name."""

    @abstractmethod
    def candidates(self, base: SweepPoint, reverse: bool) -> list[SweepPoint]:
        """
        Return the candidate values around `base`, in measurement order.

        The base itself may be one of them (the middle of an odd linear sweep, the
        center of a circle). A strategy that measures the base separately drops it.

        Args:
            base: Current values of the swept attributes (absolute).
            reverse: Traverse the candidates in the opposite direction.

        Returns:
            The candidates, as absolute values.
        """

    @abstractmethod
    def edge_direction(
        self, best: SweepPoint, base: SweepPoint, swept: Sequence[SweepPoint]
    ) -> SweepDirection | None:
        """
        Return the direction in which the optimum probably lies beyond the swept region.

        Args:
            best: The value chosen by the strategy.
            base: The value the sweep started from.
            swept: Every value measured during the sweep.

        Returns:
            A unit vector pointing from the base past the edge the best value
            lies on, or None if the best value is the base or lies inside the
            swept region.
        """

    @abstractmethod
    def offset_label(self, value: SweepPoint, base: SweepPoint) -> str:
        """
        Return a short label locating `value` relative to `base`, for chart ticks.

        Args:
            value: A swept value (absolute).
            base: The value the sweep started from (absolute).
        """


class LinearSweepSpace(SweepSpace):
    """
    Candidates evenly spaced over a range around the base value of one attribute.

    Args:
        attribute: Name of the swept attribute.
        range: `(min_offset, max_offset)` relative to the base, in units of the attribute.
        steps: Number of candidates.
    """

    def __init__(self, attribute: str, range: tuple[float, float], steps: int) -> None:
        self._attribute = attribute
        self._range = range
        self._steps = steps

    @property
    def attributes(self) -> tuple[str, ...]:
        return (self._attribute,)

    def candidates(self, base: SweepPoint, reverse: bool) -> list[SweepPoint]:
        (b,) = base
        low, high = b + self._range[0], b + self._range[1]
        if reverse:
            low, high = high, low

        return [(float(v),) for v in np.linspace(low, high, self._steps)]

    def edge_direction(
        self, best: SweepPoint, base: SweepPoint, swept: Sequence[SweepPoint]
    ) -> SweepDirection | None:
        if not swept:
            return None

        values = [v[0] for v in swept]

        if abs(best[0] - base[0]) <= 1e-9 * (max(values) - min(values)):
            return None
        if best[0] == max(values):
            return (1.0,)
        if best[0] == min(values):
            return (-1.0,)
        return None

    def offset_label(self, value: SweepPoint, base: SweepPoint) -> str:
        return f"{value[0] - base[0]:+.4g}"


class CircularSweepSpace(SweepSpace):
    """
    Candidates evenly spaced on a circle around the base value of two attributes,
    and the base itself at the center.

    The first candidate lies at `(0, +radius)` relative to the base; the others
    follow clockwise as seen with x to the right and y up, i.e. the k-th of `n`
    circle points lies at angle `360° * k / n` measured clockwise from +y. Reversed
    traversal starts at the same point and runs counter-clockwise.

    The center is a candidate so that the base can win: without it, a sweep around
    an already optimal base would always move by `radius`. It is measured halfway
    through the circle, so that a slow drift during the sweep affects it as much
    as the circle points on average.

    Args:
        attribute_x: Name of the attribute on the x-axis.
        attribute_y: Name of the attribute on the y-axis.
        radius: Radius of the circle, in units shared by both attributes.
        steps: Number of points on the circle; the center comes on top of them.
    """

    def __init__(
        self, attribute_x: str, attribute_y: str, radius: float, steps: int
    ) -> None:
        self._attributes = (attribute_x, attribute_y)
        self._radius = radius
        self._steps = steps

    @property
    def attributes(self) -> tuple[str, ...]:
        return self._attributes

    @property
    def radius(self) -> float:
        """Radius of the circle, in units of the attributes."""
        return self._radius

    def candidates(self, base: SweepPoint, reverse: bool) -> list[SweepPoint]:
        bx, by = base
        order = list(range(self._steps))
        if reverse:
            # keep the first point, traverse the rest the other way round
            order = order[:1] + order[:0:-1]

        points: list[SweepPoint] = []
        for k in order:
            angle = 2 * math.pi * k / self._steps
            points.append(
                (
                    bx + self._radius * math.sin(angle),
                    by + self._radius * math.cos(angle),
                )
            )

        # base value - center of the circle
        # in the middle of the list because of drift
        points.insert(len(points) // 2, (bx, by))

        return points

    def is_center(self, value: SweepPoint, base: SweepPoint) -> bool:
        """
        Return True if `value` is the center of the circle around `base`.

        Args:
            value: A swept value (absolute).
            base: Center of the circle (absolute).
        """
        return math.dist(value, base) <= 1e-9 * self._radius

    def edge_direction(
        self, best: SweepPoint, base: SweepPoint, swept: Sequence[SweepPoint]
    ) -> SweepDirection | None:
        # every candidate but the center lies on the boundary of the swept disc
        _ = swept
        if self.is_center(best, base):
            return None

        dx, dy = best[0] - base[0], best[1] - base[1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length)

    def angle_of(self, value: SweepPoint, base: SweepPoint) -> float:
        """
        Return the angle of `value` around `base`, in degrees clockwise from +y, in [0, 360).

        The center has no angle: the result is meaningless there, so callers
        should exclude it with `is_center` first.

        Args:
            value: A swept value (absolute).
            base: Center of the circle (absolute).
        """
        return math.degrees(math.atan2(value[0] - base[0], value[1] - base[1])) % 360.0

    def offset_label(self, value: SweepPoint, base: SweepPoint) -> str:
        if self.is_center(value, base):
            return "center"
        return f"{self.angle_of(value, base):.0f}°"


def sweep_attributes(settings: SweepingSettings) -> tuple[str, ...]:
    """
    The names of the attributes a sweep changes, without building its space.

    Args:
        settings: One- or two-dimensional sweeping settings.

    Returns:
        One name per swept attribute, in the order of a sweep point's values.
    """
    match settings:
        case LinearSweepingSettings():
            return (settings.target_attribute,)
        case CircularSweepingSettings():
            return (settings.target_attribute_x, settings.target_attribute_y)


def make_sweep_space(
    settings: SweepingSettings, defaults: DefaultContext
) -> SweepSpace:
    """
    Build the sweep space described by the sweeping settings.

    Args:
        settings: One- or two-dimensional sweeping settings.
        defaults: What a range following its default is computed from.

    Returns:
        The matching sweep space.

    Raises:
        DefaultUnavailableError: If the range follows a default that cannot
            be computed.
    """
    match settings:
        case LinearSweepingSettings():
            return LinearSweepSpace(
                settings.target_attribute,
                required_value(settings, "range", defaults),
                settings.steps,
            )
        case CircularSweepingSettings():
            return CircularSweepSpace(
                settings.target_attribute_x,
                settings.target_attribute_y,
                settings.radius,
                settings.steps,
            )

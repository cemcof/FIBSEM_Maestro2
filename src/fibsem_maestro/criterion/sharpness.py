# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""A sharpness value together with the direction in which it improves."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


class Better(Enum):
    """Whether higher or lower values of a metric mean a sharper image."""

    HIGHER = "higher"
    LOWER = "lower"

    def score(self, value: float) -> float:
        """
        Map a value to a score that is higher for a sharper image.

        Args:
            value: A value of a metric with this direction.

        Returns:
            The value itself, or its negative for a lower-is-better metric.
        """
        return value if self is Better.HIGHER else -value


@dataclass(frozen=True)
class Sharpness:
    """
    The output of a criterion: a value in the metric's own unit and its direction.

    Compare sharpness values only through `score`; `value` is for showing.

    Attributes:
        value: The metric's output, e.g. a resolution in nm.
        better: Whether higher or lower values mean sharper.
        unit: Unit of `value`, or None for a unitless metric.
    """

    value: float
    better: Better
    unit: str | None = None

    @property
    def score(self) -> float:
        """The value as a score that is higher for a sharper image."""
        return self.better.score(self.value)

    def with_value(self, value: float) -> Sharpness:
        """
        Return a sharpness of the same metric with another value.

        Args:
            value: The new value, in the same unit.

        Returns:
            The new sharpness.
        """
        return Sharpness(value, self.better, self.unit)

    def is_less_sharp_than(self, limit: float) -> bool:
        """
        Whether the image is less sharp than a limit given in the metric's unit.

        Args:
            limit: The limit, in `unit`.

        Returns:
            True if the value is on the less sharp side of the limit.
        """
        return self.score < self.better.score(limit)

    def __str__(self) -> str:
        """The value with its unit, e.g. `4.21 nm`."""
        return f"{self.value:.4g} {self.unit}" if self.unit else f"{self.value:.4g}"


def mean_of(values: Iterable[Sharpness]) -> Sharpness:
    """
    Return the mean of sharpness values of one metric.

    Args:
        values: At least one sharpness, all of the same direction and unit.

    Returns:
        A sharpness holding the mean value.

    Raises:
        ValueError: If there are no values, or they come from different metrics.
    """
    items = list(values)
    if not items:
        raise ValueError("The mean of no sharpness values is undefined.")
    first = items[0]
    if any((s.better, s.unit) != (first.better, first.unit) for s in items):
        raise ValueError("Sharpness values of different metrics cannot be averaged.")
    return first.with_value(float(np.mean([s.value for s in items])))

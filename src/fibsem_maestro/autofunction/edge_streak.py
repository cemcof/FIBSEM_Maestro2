# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Consecutive sweeps whose best value lay at the edge of the swept region."""

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fibsem_maestro.autofunction.sweep_step import SweepDirection

# in 1D the directions are +-1, so this only tells whether the edge is the same
_MAX_DIRECTION_CHANGE_DEG = 45.0


def same_direction(a: SweepDirection, b: SweepDirection) -> bool:
    """
    Tell whether two edge directions point the same way.

    Args:
        a: A unit vector in the swept space.
        b: Another unit vector in the swept space.

    Returns:
        True if they are at most 45 degrees apart.
    """
    cosine = float(np.dot(a, b))
    return cosine >= math.cos(math.radians(_MAX_DIRECTION_CHANGE_DEG)) - 1e-9


class EdgeStreak:
    """
    Counts consecutive sweeps, across runs, whose best value lay at the edge in the same direction.
    """

    def __init__(self) -> None:
        self._count = 0
        self._direction: SweepDirection | None = None

    @property
    def count(self) -> int:
        """Number of consecutive sweeps in the streak."""
        return self._count

    @property
    def direction(self) -> SweepDirection | None:
        """Direction of the edge the last sweep of the streak reached; None if there is no streak."""
        return self._direction

    def record(self, direction: SweepDirection | None) -> None:
        """
        Record a finished sweep.

        Args:
            direction: Direction of the edge its best value lay at; None if
                the best value lay inside the swept region.
        """
        if direction is None:
            self._count = 0
        elif self._direction is not None and same_direction(direction, self._direction):
            self._count += 1
        else:
            self._count = 1
        self._direction = direction

    def restore(self, count: int, direction: SweepDirection | None) -> None:
        """
        Restore a saved streak.

        Args:
            count: Number of consecutive sweeps in the streak.
            direction: Direction of the edge its last sweep reached.
        """
        self._count = count
        self._direction = direction

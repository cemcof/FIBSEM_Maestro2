# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from dataclasses import dataclass

SweepPoint = tuple[float, ...]
"""
Values of the swept attributes, one per attribute, in the order of
`SweepSpace.attributes` (one value for a 1D sweep, `(x, y)` for a 2D sweep).
Absolute values, in the units of the attributes.
"""

SweepDirection = tuple[float, ...]
"""Unit vector in the swept space, with one component per swept attribute."""


@dataclass(frozen=True)
class SweepStep:
    """
    A single step in a parameter sweep.

    Attributes:
        repetition: Zero-based index of the sweep cycle this step belongs to.
        value: The values of the swept attributes to apply at this step.
        index: Global sequential index of this step across all cycles.
        line_index: Row index of this step in the acquired image, used by
            line autofunction to map sharpness results back to their position
            in the image for logging. `None` for modes that do not operate
            on individual image rows.
    """

    repetition: int
    value: SweepPoint
    index: int
    line_index: int | None = None

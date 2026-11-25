# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""How a beam shift or a stage move ended."""

from enum import Enum


class PositioningOutcome(Enum):
    """How a beam shift or a stage move ended."""

    BEAM_SHIFTED = "beam_shifted"
    """The beam shift was applied within tolerance."""

    STAGE_MOVED = "stage_moved"
    """The stage was moved and reports being within tolerance of the target."""

    STAGE_OFF_TARGET = "stage_off_target"
    """The stage was moved but reports being further than the tolerance from the target."""

    @property
    def stage_moved(self) -> bool:
        """True if the stage was moved, whether it reports being on target or not."""
        return self is not PositioningOutcome.BEAM_SHIFTED

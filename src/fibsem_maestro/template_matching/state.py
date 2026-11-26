# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""What a set of drift-correction templates was acquired for."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import FieldUnit

FIELD_WIDTH_TOLERANCE = 0.01
"""Relative change in field width the templates still fit (1 %)."""


class TemplateMatchingState(BaseSettings):
    """
    The areas and frame geometry the stored templates were acquired with.

    Attributes:
        type: Tells this state from those of other drift calculation modes.
        areas: The areas, one template each, in template order.
        horizontal_field_width: Field width of the frames, in nm.
        resolution: Resolution of the frames.
    """

    type: Literal["template_matching"] = "template_matching"
    areas: list[RelativeArea] = Field(default_factory=list)
    horizontal_field_width: Annotated[float, FieldUnit(suffix="nm")] = 0.0
    resolution: Resolution = Field(default_factory=lambda: Resolution(0, 0))

    def differences(self, current: TemplateMatchingState) -> list[str]:
        """
        Name what differs from the current areas and frame geometry.

        Args:
            current: The current areas and geometry.

        Returns:
            E.g. `["areas", "field width"]`; empty if the templates still fit.
        """
        found = []
        if self.areas != current.areas:
            found.append("areas")
        # small changes leave the templates usable
        if not math.isclose(
            self.horizontal_field_width,
            current.horizontal_field_width,
            rel_tol=FIELD_WIDTH_TOLERANCE,
        ):
            found.append("field width")
        if self.resolution != current.resolution:
            found.append("resolution")
        return found

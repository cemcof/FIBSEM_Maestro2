# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import Annotated

from pydantic import Field, field_validator

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.frequency import Every, Frequency
from fibsem_maestro.core.pattern_type import PatternType
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import (
    AreaOverlay,
    BeamSource,
    FieldUnit,
    FormHint,
    OverlaySpec,
    WidgetType,
)
from fibsem_maestro.settings.property_names import PropertyNames


class MillingSettings(BaseSettings):
    milling_area: Annotated[
        list[RelativeArea],
        FormHint(
            widget=WidgetType.AREA_SELECT,
            min_areas=1,
            max_areas=1,
            overlays=(
                OverlaySpec.of(
                    AreaOverlay.SHOW_DIRECTION, direction="milling_direction"
                ),
                # the next slice is highlighted, from the milling's progress
                OverlaySpec.of(
                    AreaOverlay.SHOW_AREA_SLICES,
                    state={"slices_milled": "slices_milled"},
                    slice_distance_nm="slice_distance",
                    direction="milling_direction",
                ),
            ),
            beam_sources=(BeamSource("beam_type"),),
        ),
    ] = Field(
        default_factory=list,
        description="Area in which milling will be performed.",
    )
    beam_type: BeamType = Field(
        default=BeamType.ION,
        description="Beam used for milling.",
    )
    pattern_type: Annotated[
        PatternType, FormHint(widget=WidgetType.PATTERN_TYPE_SELECTOR)
    ] = Field(
        default=PatternType(""),
        description="Type of pattern to use for milling.",
    )
    milling_depth: Annotated[float, FieldUnit(suffix="nm"), Field(gt=0)] = Field(
        default=1.0, description="Depth of the milling."
    )
    slice_distance: Annotated[float, FieldUnit(suffix="nm"), Field(gt=0)] = Field(
        default=1.0,
        description="Thickness of each slice, i.e., distance to shift the pattern by after each milling step.",
    )
    milling_direction: Annotated[
        Direction,
        FormHint(
            widget=WidgetType.DROPDOWN, choices=lambda: [Direction.DOWN, Direction.UP]
        ),
    ] = Field(
        default=Direction.DOWN, description="Direction in which the slicing progresses."
    )
    do_not_mill: bool = Field(
        default=False,
        description="Prepare everything for milling but skip the actual milling process instead of performing it. Debug option.",
    )
    execution_frequency: Frequency = Field(
        default_factory=Every,
        description="How often the action runs",
    )
    properties_to_collect: PropertyNames = Field(
        default_factory=PropertyNames,
        description="Properties of the microscope and the beam relevant for milling.",
    )

    @field_validator("milling_direction")
    @classmethod
    def _validate_milling_direction(cls, v: Direction) -> Direction:
        if v not in (Direction.UP, Direction.DOWN):
            raise ValueError(f"milling_direction must be UP or DOWN, got {v!r}")
        return v

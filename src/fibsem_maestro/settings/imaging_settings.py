# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import Annotated, Literal

from pydantic import Field

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.frequency import Every, Frequency
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.criterion_settings import CriterionSettings
from fibsem_maestro.settings.form_utils import (
    BeamSource,
    FieldUnit,
    FormHint,
    WidgetType,
)
from fibsem_maestro.settings.property_names import PropertyNames


class StandardResolution(BaseSettings):
    type: Literal["standard"] = Field(
        default="standard",
        description="Images at one of the microscope's standard resolutions.",
    )


class ExtendedResolution(BaseSettings):
    type: Literal["extended"] = Field(
        default="extended",
        description="Images the scanning area at any pixel size; takes effect on Prepare.",
    )
    pixel_size: Annotated[float, Field(gt=0), FieldUnit(suffix="nm")] = Field(
        default=1.0, description="Requested size of each pixel."
    )


ResolutionMode = Annotated[
    StandardResolution | ExtendedResolution, Field(discriminator="type")
]


class ImagingSettings(BaseSettings):
    scanning_area: Annotated[
        list[RelativeArea],
        FormHint(
            widget=WidgetType.AREA_SELECT,
            max_areas=1,
            beam_sources=(BeamSource("beam_type"),),
        ),
    ] = Field(
        default_factory=list,
        description="Area that should be imaged.",
    )
    resolution_mode: ResolutionMode = Field(
        default=StandardResolution(),
        description="Use standard or extended resolution?",
    )
    beam_type: BeamType = Field(
        default=BeamType.ELECTRON,
        description="Beam used for imaging.",
    )
    criterion: CriterionSettings | None = Field(
        default=None,
        description="Settings for the criterion to use to calculate image sharpness.",
    )
    execution_frequency: Frequency = Field(
        default_factory=Every,
        description="How often the action runs",
    )
    properties_to_collect: PropertyNames = Field(
        default_factory=PropertyNames,
        description="Properties of the microscope and the beam relevant for imaging.",
    )

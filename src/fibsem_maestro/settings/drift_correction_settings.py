# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import Annotated, Literal

from pydantic import Field

from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.frequency import Every, Frequency
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.settings.template_matching_settings import TemplateMatchingSettings


class TemplateMatchingDriftCorrection(TemplateMatchingSettings):
    type: Literal["template_matching"] = Field(
        default="template_matching",
        description="Calculate the drift using template matching.",
    )


DriftCorrectionMode = Annotated[
    TemplateMatchingDriftCorrection, Field(discriminator="type")
]


class DriftCorrectionSettings(BaseSettings):
    drift_calculation_mode: DriftCorrectionMode = Field(
        default_factory=TemplateMatchingDriftCorrection,
        description="Method to use to calculate the drift.",
    )
    beam_type: BeamType = Field(
        default=BeamType.ELECTRON,
        description="Beam used for drift correction imaging.",
    )
    grab_frame_after: bool = Field(
        default=False,
        description="Grab a frame once the drift is corrected.",
    )
    stop_at_failure: bool = Field(
        default=True,
        description="If checked, a failed drift correction stops the execution. If not checked, a warning is printed.",
    )
    execution_frequency: Frequency = Field(
        default_factory=Every,
        description="How often the action runs",
    )
    properties_to_collect: PropertyNames = Field(
        default_factory=PropertyNames,
        description="Properties of the microscope and the beam relevant for drift correction.",
    )

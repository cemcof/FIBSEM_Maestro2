# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.frequency import Every, Frequency
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.criterion_settings import CriterionSettings
from fibsem_maestro.settings.form_utils import (
    AcquisitionOffset,
    BeamSource,
    FieldUnit,
    FormHint,
    NestedUnion,
    WidgetType,
)
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.settings.sweeping_settings import (
    LinearSweeping,
    SweepingSettings,
)

FocusOffset = Annotated[float, FieldUnit(suffix="nm")]
"""Beam shift along x, in nm, to focus away from the sample area during the sweep."""

ScanningArea = Annotated[
    list[RelativeArea],
    FormHint(
        widget=WidgetType.AREA_SELECT,
        max_areas=1,
        # a refocus has no beam of its own: it images with the beam of its
        # linked imaging
        beam_sources=(
            BeamSource("beam_type"),
            BeamSource("beam_type", via="linked_imaging"),
        ),
        offset=AcquisitionOffset(source="mode.delta_x"),
    ),
]
"""The area an autofocus mode scans during its sweep; empty for the whole frame."""


class BasicMode(BaseSettings):
    type: Literal["basic"] = Field(
        default="basic", description="Grab a frame for each value of the sweep."
    )
    scanning_area: ScanningArea = Field(
        default_factory=list[RelativeArea],
        description="Area scanned during the sweep. The whole frame if none is set.",
    )
    delta_x: FocusOffset = Field(
        default=0.0,
        description="Offset for out of sample focusing on the x-axis.",
    )
    sweeping: SweepingSettings = Field(
        default_factory=LinearSweeping,
        description="What is swept and how the best value is chosen.",
    )
    max_sweeps: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Repeat the sweep while the best is at the range edge, up to this many times.",
    )
    criterion: CriterionSettings = Field(
        default_factory=CriterionSettings,
        description="Setting for criterion to be used.",
    )


class LineMode(BaseSettings):
    type: Literal["line"] = Field(
        default="line", description="Perform the sweep while acquiring a single frame."
    )
    scanning_area: ScanningArea = Field(
        default_factory=list[RelativeArea],
        description="Area scanned during the sweep. The whole frame if none is set.",
    )
    delta_x: FocusOffset = Field(
        default=0.0,
        description="Offset for out of sample focusing on the x-axis.",
    )
    sweeping: SweepingSettings = Field(
        default_factory=LinearSweeping,
        description="What is swept and how the best value is chosen.",
    )
    max_sweeps: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Repeat the sweep while the best is at the range edge, up to this many times.",
    )
    criterion: CriterionSettings = Field(
        default_factory=CriterionSettings,
        description="Setting for criterion to be used.",
    )
    lines_per_sweep: Annotated[int, Field(gt=0)] = Field(
        default=5, description="Number of lines scanned per one sweep value."
    )
    pre_imaging_delay: Annotated[float, Field(ge=0.0), FieldUnit(suffix="s")] = Field(
        default=0.0,
        description="Delay before acquisition of the first section in scanning sweep.",
    )
    line_time_correction_factor: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0, description="Correction factor applied to estimated line time."
    )
    forbidden_stripe_indices: list[int] = Field(
        default_factory=list,
        description="Indices of stripes which should be excluded from the analysis.",
    )
    stripe_separator_threshold: Annotated[int, Field(gt=0)] = Field(
        default=10,
        description="Maximal average intensity of separator rows.",
    )
    minimal_stripe_width: Annotated[int, Field(gt=0)] = Field(
        default=5,
        description="Minimum distance (in rows) between two separator rows required to consider the region a valid stripe.",
    )


class CustomCriterion(BaseSettings):
    source: Literal["custom"] = Field(
        default="custom",
        description="Scores each frame with a criterion of its own.",
    )
    criterion: CriterionSettings = Field(
        default_factory=CriterionSettings,
        description="Setting for criterion to be used.",
    )


class ImagingCriterion(BaseSettings):
    source: Literal["imaging"] = Field(
        default="imaging",
        description="Reuses the sharpness the linked imaging calculates for each frame.",
    )


StepCriterion = Annotated[
    CustomCriterion | ImagingCriterion, Field(discriminator="source")
]


class StepMode(BaseSettings):
    type: Literal["step"] = Field(
        default="step",
        description="Apply the sweep values to the imaging and analyze the frames it acquires.",
    )
    sweeping: SweepingSettings = Field(
        default_factory=LinearSweeping,
        description="What is swept and how the best value is chosen.",
    )
    max_sweeps: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Repeat the sweep while the best is at the range edge, up to this many times.",
    )
    criterion: StepCriterion = Field(
        default_factory=CustomCriterion,
        description="Where the sharpness of each frame comes from.",
    )

    @model_validator(mode="before")
    @classmethod
    def _read_bare_criterion(cls, data: Any) -> Any:
        """
        Read step-mode settings saved before the criterion source existed.

        A criterion stored without a `source` is a custom criterion.

        Args:
            data: The raw input being validated.

        Returns:
            The input, with a bare criterion wrapped as a custom one.
        """
        if (
            isinstance(data, dict)
            and isinstance(criterion := data.get("criterion"), dict)
            and "source" not in criterion
        ):
            return {**data, "criterion": {"source": "custom", "criterion": criterion}}
        return data


class AutoscriptFunctionBase(BaseSettings):
    """
    Base for auto-functions delegated to Autoscript's `auto_functions` API.

    Attributes:
        type: Outer discriminator. Identical for every Autoscript variant.
    """

    type: Literal["autoscript"] = Field(
        default="autoscript", description="Use the built-in autoscript algorithm."
    )
    scanning_area: ScanningArea = Field(
        default_factory=list[RelativeArea],
        description="Area scanned during the sweep. The whole frame if none is set.",
    )
    delta_x: FocusOffset = Field(
        default=0.0,
        description="Offset for out of sample focusing on the x-axis.",
    )


class AutoscriptAutoFocusMethod(Enum):
    STANDARD = "standard"
    VOLUMESCOPE = "volumescope"


class AutoscriptAutoFocus(AutoscriptFunctionBase):
    target_attribute: Literal["working_distance"] = Field(
        default="working_distance", description="Autofocus (working distance)."
    )
    method: AutoscriptAutoFocusMethod = Field(
        default=AutoscriptAutoFocusMethod.STANDARD,
        description="Autofocus algorithm to use.",
    )
    working_distance_step: Annotated[
        float | None, Field(ge=0), FieldUnit(suffix="nm")
    ] = Field(
        default=None,
        description="Working distance step to use. Volumescope only; defaults if unset.",
    )
    number_of_frames: Annotated[int | None, Field(ge=0)] = Field(
        default=None,
        description="Number of frames to integrate. Volumescope only; defaults if unset.",
    )
    maximum_iterations: Annotated[int | None, Field(ge=0)] = Field(
        default=None,
        description="Max iterations for finding optimal focus. Volumescope only; defaults if unset.",
    )


class AutoscriptAutoStigmatorMethod(Enum):
    STANDARD = "standard"
    ONGETAL = "OngEtAl"


class AutoscriptAutoStigmator(AutoscriptFunctionBase):
    target_attribute: Literal["stigmator"] = Field(
        default="stigmator", description="Autostigmator."
    )
    method: AutoscriptAutoStigmatorMethod = Field(
        default=AutoscriptAutoStigmatorMethod.STANDARD,
        description="Autostigmator algorithm to use.",
    )
    stigmation_step: Annotated[float | None, Field(ge=0)] = Field(
        default=None,
        description="Stigmation step to use. OngEtAl only; defaults if unset.",
    )
    number_of_frames: Annotated[int | None, Field(ge=0)] = Field(
        default=None,
        description="Number of frames to integrate. OngEtAl only; defaults if unset.",
    )
    maximum_iterations: Annotated[int | None, Field(ge=0)] = Field(
        default=None,
        description="Max iterations for finding optimal stigmator settings. OngEtAl only; defaults if unset.",
    )


class AutoscriptAutoLensAlignmentModulationType(Enum):
    AUTOMATIC = "automatic"
    HIGH_VOLTAGE = "high voltage"
    WORKING_DISTANCE = "working distance"


class AutoscriptAutoLensAlignment(AutoscriptFunctionBase):
    target_attribute: Literal["lens_alignment"] = Field(
        default="lens_alignment", description="Auto lens alignment."
    )
    modulation_type: AutoscriptAutoLensAlignmentModulationType = Field(
        default=AutoscriptAutoLensAlignmentModulationType.AUTOMATIC,
        description="Modulation type for lens alignment adjustment",
    )
    number_of_frames: Annotated[int | None, Field(ge=0)] = Field(
        default=None,
        description="Number of frames to integrate. Defaults if unset.",
    )


class AutoscriptAutoSourceTilt(AutoscriptFunctionBase):
    target_attribute: Literal["source_tilt"] = Field(
        default="source_tilt", description="Auto source tilt."
    )


AutoscriptMode = Annotated[
    AutoscriptAutoFocus
    | AutoscriptAutoStigmator
    | AutoscriptAutoLensAlignment
    | AutoscriptAutoSourceTilt,
    Field(discriminator="target_attribute"),
    NestedUnion(label="autoscript mode"),
]

AutofocusMode = Annotated[
    BasicMode | LineMode | StepMode | AutoscriptMode,
    Field(discriminator="type"),
]


class AutofocusSettings(BaseSettings):
    linked_imaging: Annotated[
        str, FormHint(widget=WidgetType.ACTION_SELECTOR, action_type_filter=[Imaging])
    ] = Field(
        default="", description="Name of the imaging action linked to this autofocus."
    )
    mode: AutofocusMode = Field(
        default_factory=BasicMode,
        description="Autofocus mode to use.",
    )
    beam_type: BeamType = Field(
        default=BeamType.ELECTRON,
        description="Beam on which the autofocus should be performed.",
    )
    max_workers: Annotated[int, Field(gt=0)] = Field(
        default=1,
        description="Maximal number of threads used for the calculations in this action.",
    )
    grab_frame_after: bool = Field(
        default=False,
        description="Grab a frame once the best value is applied.",
    )
    sharpness_limit: Annotated[float, Field(gt=0)] | None = Field(
        default=None,
        description="Autofunction runs if image sharpness is below this limit. If not checked, this condition is not applied.",
    )
    execution_frequency: Frequency = Field(
        default_factory=Every,
        description="How often the action runs",
    )
    properties_to_collect: PropertyNames = Field(
        default_factory=PropertyNames,
        description="Selection of microscope and beam properties relevant for this action.",
    )

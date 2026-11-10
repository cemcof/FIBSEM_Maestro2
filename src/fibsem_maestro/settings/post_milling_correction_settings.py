# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Settings of the post-milling correction, whose shifts default to the linked milling's geometry."""

import math
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, model_validator

from fibsem_maestro.action.action import Action
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.frequency import Every, Frequency
from fibsem_maestro.milling.milling import Milling
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.dynamic_default import DynamicDefault, SettingsLookup
from fibsem_maestro.settings.form_utils import FieldUnit, FormHint, WidgetType
from fibsem_maestro.settings.milling_settings import MillingSettings
from fibsem_maestro.settings.property_names import PropertyNames

if TYPE_CHECKING:
    from fibsem_maestro.core.beam_type import BeamType

ION_ELECTRON_ANGLE_DEG = 52.0
"""Angle between the ion and electron beams, in degrees."""

NO_DEFAULT_CORRECTION = "Link a milling to compute the default, or set a value."
"""Shown on a correction that follows its default while no milling is linked."""

_CORRECTION_SOURCE = "the slice distance and direction of the linked milling"
"""What the default corrections are computed from."""


def geometric_correction(milling: MillingSettings) -> tuple[float, float]:
    """
    The shifts that keep an electron-beam image on the face one milled slice deeper.

    The face recedes by one slice distance along the ion beam, which is at
    `ION_ELECTRON_ANGLE_DEG` to the electron beam: seen by the electron beam,
    the face moves vertically in the image and away along the beam.

    Args:
        milling: Settings of the milling removing the slices.

    Returns:
        The vertical image shift and the working-distance change, both in nm;
        negated when milling up, which moves the face the other way.
    """
    angle = math.radians(ION_ELECTRON_ANGLE_DEG)
    y_correction = math.cos(angle) * milling.slice_distance
    wd_correction = math.sin(angle) * milling.slice_distance
    if milling.milling_direction is Direction.UP:
        return -y_correction, -wd_correction
    return y_correction, wd_correction


def _linked_milling(
    settings: PostMillingCorrectionSettings, lookup: SettingsLookup
) -> MillingSettings | None:
    """The settings of the linked milling, or None if no milling is linked."""
    linked = lookup(settings.linked_milling) if settings.linked_milling else None
    return linked if isinstance(linked, MillingSettings) else None


def _default_y_correction(
    settings: PostMillingCorrectionSettings, lookup: SettingsLookup
) -> float | None:
    """The vertical image shift of the linked milling's geometry, in nm."""
    milling = _linked_milling(settings, lookup)
    return None if milling is None else geometric_correction(milling)[0]


def _default_wd_correction(
    settings: PostMillingCorrectionSettings, lookup: SettingsLookup
) -> float | None:
    """The working-distance change of the linked milling's geometry, in nm."""
    milling = _linked_milling(settings, lookup)
    return None if milling is None else geometric_correction(milling)[1]


class PostMillingCorrectionSettings(BaseSettings):
    """
    Settings of the post-milling correction.

    The correction stores no microscope properties of its own and works on no
    beam of its own: it adds its shifts to the upcoming properties of the
    corrected actions, each on its own beam.

    Both shifts default to the geometry of the linked milling (see
    `geometric_correction`), which assumes the corrected actions image with
    the electron beam. The defaults follow the milling's settings until a
    value is set.
    """

    linked_milling: Annotated[
        str, FormHint(widget=WidgetType.ACTION_SELECTOR, action_type_filter=[Milling])
    ] = Field(
        default="",
        description="Milling whose slice distance and direction the default corrections are computed from.",
    )
    corrected_actions: Annotated[
        list[str],
        FormHint(
            widget=WidgetType.MULTI_ACTION_SELECTOR,
            action_type_filter=[Action],
            # correcting a correction would only shift a value it does not store
            action_type_exclude=("post_milling_correction",),
        ),
    ] = Field(
        default_factory=list,
        description="Actions whose image position and working distance follow the receding cross-section face.",
    )
    y_correction: Annotated[
        float | None,
        FieldUnit(suffix="nm"),
        DynamicDefault(
            _default_y_correction, NO_DEFAULT_CORRECTION, _CORRECTION_SOURCE
        ),
    ] = Field(
        default=None,
        description="Vertical image shift added to the corrected actions on every run.",
    )
    wd_correction: Annotated[
        float | None,
        FieldUnit(suffix="nm"),
        DynamicDefault(
            _default_wd_correction, NO_DEFAULT_CORRECTION, _CORRECTION_SOURCE
        ),
    ] = Field(
        default=None,
        description="Working-distance change added to the corrected actions on every run.",
    )
    execution_frequency: Frequency = Field(
        default_factory=Every,
        description="How often the action runs",
    )

    @model_validator(mode="before")
    @classmethod
    def _from_correction_mode(cls, data: Any) -> Any:
        # settings saved with a correction mode: a manual one set both shifts,
        # the geometric one followed the milling, as the defaults now do
        if not isinstance(data, dict) or "correction_mode" not in data:
            return data
        data = dict(data)
        mode = data.pop("correction_mode")
        if isinstance(mode, dict) and mode.get("type") == "manual":
            data.setdefault("y_correction", mode.get("y_correction", 0.0))
            data.setdefault("wd_correction", mode.get("wd_correction", 0.0))
        return data

    @property
    def beam_type(self) -> BeamType | None:
        """
        None: every corrected action is corrected on its own beam.

        So the action never selects a beam on the microscope.
        """
        return None

    @property
    def properties_to_collect(self) -> PropertyNames:
        """
        Nothing: the correction changes the corrected actions' properties, not the microscope.

        Stored properties of its own would only be applied before every run,
        imposing stale values on the microscope.
        """
        return PropertyNames()

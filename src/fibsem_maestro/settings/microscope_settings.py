# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import Annotated, Any

from pydantic import AfterValidator, Field, model_validator

from fibsem_maestro.microscope.registry import MICROSCOPE_CONTROLS
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import FieldUnit, FormHint, WidgetType

MicroscopeControlName = Annotated[
    str,
    AfterValidator(MICROSCOPE_CONTROLS.validate),
    FormHint(widget=WidgetType.DROPDOWN, choices=lambda: list(MICROSCOPE_CONTROLS)),
]


class MicroscopeSettings(BaseSettings):
    control: MicroscopeControlName = Field(description="Type of microscope control.")
    ip_address: str = Field(description="Microscope server address.")
    port: str = Field(
        default="",
        description="Microscope port to connect to. Leave empty to connect to the default port.",
    )
    beam_shift_tolerance: Annotated[float, FieldUnit(suffix="nm")] = Field(
        default=50, description="Maximal allowed beam shift error."
    )
    stage_tolerance: Annotated[float, FieldUnit(suffix="nm")] = Field(
        default=100, description="Maximal allowed stage error."
    )
    beam_shift_to_stage_move_electron: tuple[float, float] = Field(
        default=(-1.0, -1.0),
        description="Per-axis factor for converting beam shift in the electron beam to stage move.",
    )
    beam_shift_to_stage_move_ion: tuple[float, float] = Field(
        default=(-1.0, -1.0),
        description="Per-axis factor for converting beam shift in the ion beam to stage move.",
    )

    # TODO: remove later
    @model_validator(mode="before")
    @classmethod
    def _without_stage_trials(cls, data: Any) -> Any:
        # settings saved while stage moves were retried; they no longer are
        if isinstance(data, dict) and "stage_trials" in data:
            return {k: v for k, v in data.items() if k != "stage_trials"}
        return data

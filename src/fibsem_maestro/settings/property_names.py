# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import Annotated

from pydantic import Field

from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.microscope_properties import MicroscopeProperties
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import FormHint, WidgetType


class PropertyNames(BaseSettings):
    microscope: Annotated[
        list[str],
        FormHint(
            widget=WidgetType.MULTI_PROPERTY_SELECTOR,
            choices=lambda: list(MicroscopeProperties.model_fields.keys()),
        ),
    ] = Field(default_factory=list)
    electron_beam: Annotated[
        list[str],
        FormHint(
            widget=WidgetType.MULTI_SELECT,
            choices=lambda: list(BeamProperties.model_fields.keys()),
        ),
    ] = Field(default_factory=list)
    ion_beam: Annotated[
        list[str],
        FormHint(
            widget=WidgetType.MULTI_SELECT,
            choices=lambda: list(BeamProperties.model_fields.keys()),
        ),
    ] = Field(default_factory=list)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from fibsem_maestro.core.frequency import Every
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.settings.autofunction_settings import BasicMode, LineMode
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import FormHint, WidgetType
from fibsem_maestro.settings.property_names import PropertyNames

if TYPE_CHECKING:
    from fibsem_maestro.core.beam_type import BeamType

RefocusMode = Annotated[BasicMode | LineMode, Field(discriminator="type")]
"""The modes a refocus can run in: single-shot modes, completed within one slice."""


class RefocusSettings(BaseSettings):
    """
    Settings of a refocus, which keeps the trial frames of step-mode autofunctions in focus.

    While a step-mode autofunction linked to the same imaging has a run in
    progress, the refocus sweeps its own attribute under the conditions
    of the imaging's next frame, and hands the best value to the imaging.
    """

    linked_imaging: Annotated[
        str, FormHint(widget=WidgetType.ACTION_SELECTOR, action_type_filter=[Imaging])
    ] = Field(
        default="",
        description="Imaging whose step-mode trial frames are refocused.",
    )
    mode: RefocusMode = Field(
        default_factory=BasicMode,
        description="How the refocus sweeps.",
    )
    max_workers: Annotated[int, Field(gt=0)] = Field(
        default=1,
        description="Maximal number of threads used for the calculations in this action.",
    )

    @property
    def beam_type(self) -> BeamType | None:
        """
        None: the refocus uses the beam of its linked imaging (`Refocus.beam_type`).
        """
        return None

    @property
    def execution_frequency(self) -> Every:
        """
        Every slice; the refocus decides itself on which slices it runs.
        """
        return Every()

    @property
    def properties_to_collect(self) -> PropertyNames:
        """
        Nothing: the refocus works under the linked imaging's properties.
        """
        return PropertyNames()

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import TYPE_CHECKING

from pydantic import Field

from fibsem_maestro.action.action import Action
from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.core.area import NMArea, RelativeArea
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.point import NMPoint
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.milling.error import MillingError
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.milling_settings import MillingSettings

if TYPE_CHECKING:
    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.workflow.actions import Actions

_TOLERANCE_NM = 0.1
"""Slack allowed when checking that a slice fits the milling area (nm)."""


class PixelScale(BaseSettings):
    """
    Resolution and pixel size of a beam, for converting relative areas to nanometers.

    Attributes:
        resolution: Scan resolution, in pixels.
        pixel_size_nm: Size of one pixel, in nm.
    """

    resolution: Resolution
    pixel_size_nm: float

    def to_nanometers(self, area: RelativeArea) -> NMArea:
        """
        Convert a relative area to nanometers, rounding to whole pixels as the milling does.

        Args:
            area: The area, relative to the scanned image (0-1).

        Returns:
            The area in nm, measured from the image's top-left corner.
        """
        return area.to_nanometers(self.resolution, self.pixel_size_nm)


class MillingState(ActionState):
    """
    The progress of a milling through its milling area.

    The next slice lies `slices_milled` slice distances from the trailing edge
    of the milling area (the edge milling starts at), so it follows the area
    when the area is moved or resized.
    """

    slices_milled: int = Field(
        default=0,
        ge=0,
        description="Slices milled since the state was last reset.",
    )
    pixel_scale: PixelScale | None = Field(
        default=None,
        description="Resolution and pixel size of the beam at the last milling. "
        "Used to convert the milling area to nanometers when checking whether the "
        "next slice still fits, without asking the microscope. None until the "
        "first slice is milled.",
    )


_AREA_USED_UP = (
    "the milling area is used up; enlarge it or reset the milling's state to continue"
)


@ACTION_REGISTRY.register("milling")
class Milling(Action[MillingSettings, MillingState]):
    """
    Performs focused ion beam milling one slice at a time.

    Slices are milled one slice distance apart. The state counts the slices
    milled since it was last reset; the area may be moved or resized meanwhile,
    and the next slice follows it. The acquisition ends once the next slice no longer
    fits the area; enlarging the area, or resetting the state (which restarts at the start edge),
    lets it continue.
    """

    _STATE_FIELDS = frozenset({"slice_distance", "milling_direction", "beam_type"})

    def __init__(
        self,
        name: str,
        microscope: Microscope,
        settings: MillingSettings,
        ctx: ActionContext,
        actions: Actions,
    ):
        super().__init__(name, microscope, settings, ctx, actions)
        self._slices_milled = 0
        self._pixel_scale: PixelScale | None = None

    @classmethod
    def settings_cls(cls) -> type[MillingSettings]:
        return MillingSettings

    @classmethod
    def state_cls(cls) -> type[MillingState]:
        return MillingState

    @property
    def state(self) -> MillingState:
        return MillingState(
            slices_milled=self._slices_milled, pixel_scale=self._pixel_scale
        )

    def set_state(self, state: MillingState) -> None:
        self._slices_milled = state.slices_milled
        self._pixel_scale = state.pixel_scale

    def completion_reason(self) -> str | None:
        """
        Report the end of the acquisition once the next slice no longer fits the milling area.

        Returns:
            The reason if the next slice does not fit, otherwise None (also
            before the first milling, or without exactly one milling area).
        """
        areas = self._settings.milling_area
        if len(areas) != 1 or self._pixel_scale is None:
            return None
        if self._fits(self._slices_milled, self._pixel_scale.to_nanometers(areas[0])):
            return None
        return _AREA_USED_UP

    def test(self) -> None:
        raise MillingError(f"Testing is not implemented for {self.name}")

    def _run_step(self) -> StepOutcome:
        areas = self._settings.milling_area
        if len(areas) != 1:
            raise MillingError(f"Expected exactly one milling area, got {len(areas)}.")

        beam = self._microscope.beam
        self._pixel_scale = PixelScale(
            resolution=beam.resolution, pixel_size_nm=beam.pixel_size
        )
        milling_area_nm = self._pixel_scale.to_nanometers(areas[0])

        # never mill outside the designated area
        if not self._fits(self._slices_milled, milling_area_nm):
            raise MillingError(
                "The milling area is used up; enlarge it or reset the milling's "
                "state to continue."
            )

        milling_slice = self._slice_at(self._slices_milled, milling_area_nm)
        if not milling_slice.has_size():
            raise MillingError(f"Area to be milled has no size: {milling_slice}")

        self._ctx.text_logger.debug(
            f"Area to be milled in this slice in nanometers: {milling_slice}."
        )
        self._ctx.text_logger.info("Starting the milling procedure.")
        beam.rectangle_milling(
            milling_slice,
            self._settings.milling_depth,
            self._settings.milling_direction,
            self._settings.pattern_type,
            self._settings.do_not_mill,
        )
        self._ctx.text_logger.info("Milling procedure completed.")
        self._slices_milled += 1

        if self._fits(self._slices_milled, milling_area_nm):
            self._ctx.text_logger.debug(
                "Area to be milled in the next slice in nanometers: "
                f"{self._slice_at(self._slices_milled, milling_area_nm)}."
            )
        else:
            # the next slice would leave the area: this slice was the last one; the
            # workflow finishes before the next slice (see `completion_reason`)
            self._ctx.text_logger.info("The milling area is used up.")

        return Produced(self.collect_properties())

    def _fits(self, index: int, area_nm: NMArea) -> bool:
        """
        Whether the slice with the given index fits the milling area.

        Args:
            index: Index of the slice, counted from the trailing edge (0 = first).
            area_nm: The milling area, in nm.

        Returns:
            True if the slice ends within the area, up to `_TOLERANCE_NM`.
        """
        depth_nm = (index + 1) * self._settings.slice_distance
        return depth_nm <= area_nm.height + _TOLERANCE_NM

    def _slice_at(self, index: int, area_nm: NMArea) -> NMArea:
        """
        The slice with the given index, counted from the starting edge of the milling area.

        Args:
            index: Index of the slice (0 = the slice at the starting edge).
            area_nm: The milling area, in nm.

        Returns:
            The slice, in nm: as wide as the area, one slice distance high.

        Raises:
            MillingError: If the milling direction is not up or down.
        """
        slice_distance = self._settings.slice_distance
        match self._settings.milling_direction:
            case Direction.DOWN:
                y = area_nm.origin.y + index * slice_distance
            case Direction.UP:
                y = area_nm.origin.y + area_nm.height - (index + 1) * slice_distance
            case direction:
                raise MillingError(f"Invalid milling direction {direction}")

        return NMArea(
            origin=NMPoint(x=area_nm.origin.x, y=y),
            width=area_nm.width,
            height=slice_distance,
        )

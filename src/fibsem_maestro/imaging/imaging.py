# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import contextvars
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fibsem_maestro.action.action import Action
from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.format import ImageFormat
from fibsem_maestro.criterion.criterion import Criterion
from fibsem_maestro.imaging.error import ImagingError
from fibsem_maestro.logging.logging import with_logging_context
from fibsem_maestro.microscope.error import MicroscopeError
from fibsem_maestro.settings.dynamic_default import (
    DefaultUnavailableError,
    with_defaults,
)
from fibsem_maestro.settings.imaging_settings import (
    ExtendedResolution,
    ImagingSettings,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.core.image import Image
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.settings.criterion_settings import CriterionSettings
    from fibsem_maestro.slice.slice_view import SliceView
    from fibsem_maestro.workflow.actions import Actions


class ImagingState(ActionState):
    image_sharpness: float | None = None
    # slice the last frame (and its sharpness) was acquired on; None if none was
    frame_slice: int | None = None


@ACTION_REGISTRY.register("imaging")
class Imaging(Action[ImagingSettings, ImagingState]):
    """
    Orchestrates a single image acquisition cycle on the electron microscope.
    """

    # beam properties the extended-resolution geometry alters and restores;
    # pixel size and vertical field width follow from horizontal field width
    # and resolution, so restoring those two restores them as well
    _EXTENDED_RESOLUTION_STATE: tuple[str, ...] = (
        "beam_shift",
        "horizontal_field_width",
        "resolution",
    )

    def __init__(
        self,
        name: str,
        microscope: Microscope,
        settings: ImagingSettings,
        ctx: ActionContext,
        actions: Actions,
    ):
        super().__init__(name, microscope, settings, ctx, actions)

        # sharpness of the acquired image
        self._image_sharpness: float | None = None
        self._sharpness_thread: threading.Thread | None = None

        # save the last acquired image
        self._last_acquired_image: Image | None = None
        # slice the last frame was acquired on; a skipped slice acquires none
        self._frame_slice: int | None = None

    @classmethod
    def settings_cls(cls) -> type[ImagingSettings]:
        return ImagingSettings

    @classmethod
    def state_cls(cls) -> type[ImagingState]:
        return ImagingState

    @property
    def state(self) -> ImagingState:
        return ImagingState(
            image_sharpness=self._image_sharpness,
            frame_slice=self._frame_slice,
        )

    @property
    def scanning_area(self) -> RelativeArea | None:
        """
        The configured area in standard resolution; None in extended resolution.

        Extended resolution scans the full frame and images the area by
        shifting and shrinking the field of view instead (see `_extended_geometry`).
        """
        if isinstance(self._settings.resolution_mode, ExtendedResolution):
            return None

        return self._first_scanning_area()

    def set_state(self, state: ImagingState) -> None:
        self._image_sharpness = state.image_sharpness
        self._frame_slice = state.frame_slice

        # skip if the acquisition hasn't started yet
        if self._ctx.slice == 0:
            return

        # restore the last frame, which a step-mode autofunction may still have to score
        self._last_acquired_image = self._read_last_frame()

        if (
            state.image_sharpness is None
            and self.settings.criterion is not None
            and self._last_acquired_image is not None
        ):
            self._ctx.text_logger.warning(
                f"Restoring state of '{self.name}': image sharpness not computed, "
                "recovering it from the last frame."
            )
            if (criterion := self._resolved_criterion()) is not None:
                self._calculate_sharpness(
                    self._last_acquired_image, self._ctx.current_view, criterion
                )

    def _read_last_frame(self) -> Image | None:
        """
        Read the frame this imaging acquired last, for restoring after an interrupt.

        The current slice has a frame if the interrupt came after the
        acquisition but before the slice counter advanced; otherwise the last
        frame is the previous slice's.

        Returns:
            The frame, or None if neither slice has one (e.g. the imaging did
            not run on the previous slice).
        """
        stores = (self._ctx.frame_store, self._ctx.frame_store.at(self._ctx.slice - 1))
        for store in stores:
            try:
                return store.read(self._microscope.provenance)
            except FileNotFoundError:
                continue

        self._ctx.text_logger.warning(
            f"Restoring state of '{self.name}': no frame found for slice "
            f"{self._ctx.slice} or {self._ctx.slice - 1}."
        )
        return None

    @property
    def last_acquired_image(self) -> Image | None:
        """Get the last image acquired by this imaging."""
        return self._last_acquired_image

    @property
    def frame_slice(self) -> int | None:
        """
        The slice the last frame, and so its sharpness, was acquired on.

        A skipped slice acquires no frame and leaves this unchanged, so a
        reader of the last frame can tell whether it is new.

        Returns:
            The slice, or None if no frame was acquired since the state was
            last reset (or the state predates this record).
        """
        return self._frame_slice

    def _run_step(self) -> StepOutcome:
        self._ctx.frame_store.raise_if_exists(
            ImagingError,
            f"Frame for slice {self._ctx.slice} for action '{self.name}' already exists.",
        )

        with self._extended_geometry():
            self._last_acquired_image = self._microscope.beam.grab_frame(
                self._ctx.frame_store
            )
        self._frame_slice = self._ctx.slice

        # collected after the extended geometry is undone, so that the props
        # hold the base field of view and the base beam position
        props = self.collect_properties()
        self._start_sharpness_calculation(self._last_acquired_image)
        return Produced(props)

    def _start_sharpness_calculation(self, image: Image) -> None:
        """
        Start the background sharpness evaluation for an acquired image.

        Does nothing beyond logging if no criterion is configured. Call
        `wait_for_sharpness` to block until the result is available.

        Args:
            image: The image to evaluate.
        """
        # a calculation still running belongs to the previous frame; let it
        # finish first, or it would overwrite the result for this frame
        self.wait_for_sharpness()

        self._image_sharpness = None
        if self._settings.criterion is None:
            self._ctx.text_logger.debug(
                f"Criterion is not configured for {self.name}. "
                "Image sharpness will not be calculated."
            )
            return

        criterion = self._resolved_criterion()
        if criterion is None:
            return

        # run in a copy of the logging context, addressed to this slice
        ctx_snapshot = contextvars.copy_context()
        self._sharpness_thread = threading.Thread(
            target=ctx_snapshot.run,
            args=(self._calculate_sharpness, image, self._ctx.current_view, criterion),
        )
        self._sharpness_thread.start()

    @with_logging_context
    def test(self) -> None:
        """
        Acquire a single test frame and save it alongside the slice frames.

        The frame is taken as during acquisition: within the scanning area, or
        in extended resolution with its geometry. Persisted properties are
        applied first if they exist.
        """
        if (bt := self.beam_type) is not None:
            self._microscope.set_beam(bt)

        with self._scanning_area_applied():
            if self._ctx.props_store.exists("props.yaml"):
                self._ctx.text_logger.info("Loading saved microscope properties.")
                self.read_and_set_properties()
            else:
                self._ctx.text_logger.info("No saved microscope properties found.")

            with self._extended_geometry():
                output_path = self._grab_test_frame()

        self._ctx.text_logger.info(
            f"Completed test for {self.name}. Acquired image saved as {output_path}."
        )

    def _first_scanning_area(self) -> RelativeArea | None:
        """
        Return the configured scanning area, if any.

        Returns:
            The first scanning area from the settings,
            or `None` when no scanning area is configured.
        """
        return next(iter(self._settings.scanning_area), None)

    def _grab_test_frame(self) -> Path:
        """
        Grab a frame and save it with the `.test.tif` suffix.

        The frame is stored manually rather than through the frame store so
        that it does not block acquisition of further images for the current
        slice.

        Returns:
            The path the acquired image was written to.
        """
        image = self._microscope.beam.grab_frame()
        output_path = Path(str(self._ctx.frame_store.path())).with_suffix(".test.tif")
        image.save(output_path, ImageFormat.TIF)
        return output_path

    def wait_for_sharpness(self) -> float | None:
        """
        Block until the background sharpness calculation finishes.

        Returns:
            The calculated sharpness value, or `None` if no criterion is
            configured or the calculation failed.
        """
        if self._sharpness_thread is not None:
            self._sharpness_thread.join()
        return self._image_sharpness

    def wait_for_background_threads(self) -> None:
        self.wait_for_sharpness()

    def _resolved_criterion(self) -> CriterionSettings | None:
        """
        The criterion settings with their dynamic defaults computed from the stored properties.

        Computed on the acquiring thread, before the calculation starts, from
        the properties stored for the current slice.

        Returns:
            The settings, or None (with a warning logged) if no criterion is
            configured or a default cannot be computed.
        """
        if self._settings.criterion is None:
            return None
        try:
            return with_defaults(self._settings.criterion, self.default_context())
        except DefaultUnavailableError as e:
            self._ctx.text_logger.warning(
                f"Image sharpness will not be calculated: {e}"
            )
            return None

    def _calculate_sharpness(
        self, image: Image, view: SliceView, settings: CriterionSettings
    ) -> None:
        """
        Evaluate image sharpness on a background thread.

        Computes the sharpness score via the configured criterion and stores
        the result in `_image_sharpness`. Any exception raised during
        calculation is caught and logged as a warning so that a failure cannot
        crash the background thread.

        Args:
            image: The image to evaluate.
            view: The slice view to use for logging.
            settings: The criterion's settings, with their defaults computed
                (`_resolved_criterion`).
        """
        # get logger for the slice corresponding to the provided view
        text_logger = self._ctx.text_logger.at(view.slice_index)
        image_logger = self._ctx.image_logger.at(view.slice_index)

        criterion = Criterion(
            settings,
            text_logger.derive("criterion"),
            image_logger,
        )

        try:
            self._image_sharpness = criterion.calculate_sharpness(image)
            text_logger.debug(f"Image sharpness: {self._image_sharpness}.")
        except Exception as e:
            text_logger.warning(f"Could not calculate image sharpness: {e}")

    @contextmanager
    def _extended_geometry(self) -> Generator[None]:
        """
        Image the scanning area at the extended-resolution pixel size for the enclosed block.

        Does nothing in standard resolution. The geometry is derived anew from
        the settings and the base geometry on the microscope, and undone on
        exit: the emulated resolution is cleared, then resolution, field width
        and beam shift are restored to their base values.

        Yields:
            None. The base geometry is restored even if the block raises.

        Raises:
            MicroscopeError: If the offset to the area's centre exceeds the
                beam-shift range on top of the current beam shift.
        """
        mode = self._settings.resolution_mode
        if not isinstance(mode, ExtendedResolution):
            yield
            return

        beam = self._microscope.beam
        with ExitStack() as stack:
            for name in self._EXTENDED_RESOLUTION_STATE:
                # save the current values of the beam properties so that they are restored on exit
                stack.enter_context(
                    self._microscope.set_temporary_beam_property(
                        name, getattr(beam, name), self._settings.beam_type
                    )
                )

            # on exit, first clear the extended resolution
            stack.callback(beam.clear_extended_resolution)

            self._set_extended_geometry(
                self._first_scanning_area() or RelativeArea.full(), mode.pixel_size
            )
            yield

    def _set_extended_geometry(
        self, scanning_area: RelativeArea, new_pixel_size: float
    ) -> None:
        """
        Shift and shrink the field of view onto the scanning area at a new pixel size.

        The offset to the area's centre is added to the current beam shift, so
        the area follows the drift corrections the base geometry carries. It
        is a plain beam shift: a fallback stage move would displace the sample
        on every frame.

        Args:
            scanning_area: The area to image, relative to the base frame.
            new_pixel_size: The target pixel size in nanometers.
        """
        beam = self._microscope.beam

        if not scanning_area.is_full_frame():
            img_res = beam.resolution
            pixel_size = beam.pixel_size
            area_nm = scanning_area.to_nanometers(img_res, pixel_size)
            # the microscope squeezes frames taller than their field width
            # instead of scanning further vertically, so such an area cannot be
            # imaged with its own width as the field width
            if area_nm.height > area_nm.width:
                raise ImagingError(
                    "Extended resolution cannot image a scanning area taller than "
                    "it is wide. Make the area at least as wide as it is tall."
                )

            image_to_beam_shift = beam.image_to_beam_shift

            shift = BeamShift(
                image_to_beam_shift[0]
                * (
                    area_nm.origin.x
                    - (img_res.width // 2 * pixel_size)
                    + area_nm.width / 2.0
                ),
                image_to_beam_shift[1]
                * (
                    area_nm.origin.y
                    - (img_res.height // 2 * pixel_size)
                    + area_nm.height / 2.0
                ),
            )

            try:
                # we cannot use stage position fallback because stage position
                # does not get reverted and it would clash with drift correction
                self._microscope.add_beam_shift_without_stage_fallback(shift, beam)
            except MicroscopeError as e:
                raise ImagingError(
                    f"Extended resolution needs a beam shift of {shift} to reach the "
                    "scanning area, which exceeds the beam-shift range. Move the area "
                    "closer to the image centre, increase the magnification, or "
                    "centre the area with the stage."
                ) from e

            # HFW must be set before VFW: the VFW setter derives the new
            # resolution from the pixel size that setting HFW has just changed
            beam.horizontal_field_width = area_nm.width
            beam.vertical_field_width = area_nm.height

        # set resolution based on the new pixel size, even for the full frame
        beam.pixel_size = new_pixel_size

        self._ctx.text_logger.info(
            f"Extended resolution geometry: beam shift {beam.beam_shift}, "
            f"field width {beam.horizontal_field_width} nm, resolution {beam.resolution}."
        )

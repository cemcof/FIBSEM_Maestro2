# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import math
from typing import TYPE_CHECKING

import numpy as np

from fibsem_maestro.core.area import NMArea, RelativeArea
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.detector_mode import DetectorMode
from fibsem_maestro.core.format import ImageFormat
from fibsem_maestro.core.image import Image
from fibsem_maestro.core.lens_alignment import LensAlignment
from fibsem_maestro.core.provenance import Provenance
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.core.source_tilt import SourceTilt
from fibsem_maestro.core.stigmator import Stigmator
from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
from fibsem_maestro.microscope.error import MicroscopeError
from fibsem_maestro.microscope.simulated.sample import SimulatedSample

if TYPE_CHECKING:
    from fibsem_maestro.core.direction import Direction
    from fibsem_maestro.core.pattern_type import PatternType
    from fibsem_maestro.core.stage_position import StagePosition
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.store.frame.frame_store import FrameStore


class SimulatedBeamControl(BeamControl):
    """Simulated beam controller."""

    def __init__(
        self,
        name: str,
        sample: SimulatedSample,
        stage_position: StagePosition,
        txt_log: TextLogger,
        rng: np.random.Generator,
    ):
        self._name = name
        self._txt_log = txt_log
        self._rng = rng

        self._sample = sample

        # current state
        self._stage_position = stage_position
        self._working_distance_nm = 5_000_000.0
        self._stigmator = Stigmator(x=0.0, y=0.0)
        self._lens_alignment = LensAlignment(x=0.0, y=0.0)  # nm
        self._beam_shift = BeamShift(x=0.0, y=0.0)  # nm
        self._beam_current: float = 0.0  # nm

        self._detector_mode = DetectorMode.DEFAULT

        self._detector_contrast = 0.5
        self._detector_brightness = 0.5
        self._source_tilt = SourceTilt(x=0.0, y=0.0)  # degrees
        self._scan_rotation = 0.0  # degrees

        self._blanked = False
        self._acquiring = False

        self._line_integration = 1
        self._dwell_time = 100.0  # in ns
        self._bit_depth = 8
        self._resolution = Resolution(1024, 768)
        self._horizontal_field_width = 2000.0
        self._scanning_area = RelativeArea.full()

        self._image_to_beam_shift = (-1, 1)

        self._current_image: Image | None = None

        self._limits: dict[str, tuple[float, float]] = {
            "working_distance": (
                500_000.0,
                70_000_000.0,
            ),
            "stigmator_x": (-0.99, 0.88),
            "stigmator_y": (-0.99, 0.77),
            "lens_alignment_x": (-720_052.0, 697_917.0),
            "lens_alignment_y": (-691_406.0, 689_453.0),
            "beam_shift_x": (-200.0, 200.0),
            "beam_shift_y": (-200.0, 200.0),
            "detector_contrast": (0.0, 1.0),
            "detector_brightness": (0.0, 1.0),
            "source_tilt_x": (-5.0, 5.0),
            "source_tilt_y": (-5.0, 5.0),
            "horizontal_field_width": (1_000.0, 5_000_000_000.0),
            "vertical_field_width": (1_000.0, 5_000_000_000.0),
        }

    @classmethod
    def provenance(cls) -> Provenance:
        return Provenance.MAESTRO

    def limits(self, var: str) -> tuple[float, float]:
        return self._limits.get(var, (-float("inf"), float("inf")))

    @property
    def working_distance(self) -> float:
        value = self._working_distance_nm
        self._txt_log.debug(f"Getting working distance: {value}.")
        return value

    @working_distance.setter
    def working_distance(self, value: float):
        self._txt_log.debug(f"Setting working distance: {value}.")
        self._working_distance_nm = value

    @property
    def stigmator(self) -> Stigmator:
        value = self._stigmator
        self._txt_log.debug(f"Getting stigmator: {value}.")
        return value

    @stigmator.setter
    def stigmator(self, value: Stigmator) -> None:
        self._txt_log.debug(f"Setting stigmator: {value}.")
        self._stigmator = value

    @property
    def lens_alignment(self) -> LensAlignment:
        value = self._lens_alignment
        self._txt_log.debug(f"Getting lens alignment: {value}.")
        return value

    @lens_alignment.setter
    def lens_alignment(self, value: LensAlignment) -> None:
        self._txt_log.debug(f"Setting lens alignment: {value}.")
        self._lens_alignment = value

    @property
    def beam_shift(self) -> BeamShift:
        value = self._beam_shift
        self._txt_log.debug(f"Getting beam shift: {value}.")
        return value

    @beam_shift.setter
    def beam_shift(self, value: BeamShift) -> None:
        self._txt_log.debug(f"Setting beam shift: {value}.")
        self._beam_shift = value

        limit_x = self.limits("beam_shift_x")
        limit_y = self.limits("beam_shift_y")
        if (
            self._beam_shift.x < limit_x[0]
            or self._beam_shift.x > limit_x[1]
            or self._beam_shift.y < limit_y[0]
            or self._beam_shift.y > limit_y[1]
        ):
            raise MicroscopeError(
                f"Beam shift out of range: {self._beam_shift} (limits: x {limit_x}, y {limit_y})"
            )

    @property
    def detector_mode(self) -> DetectorMode:
        value = self._detector_mode
        self._txt_log.debug(f"Getting detector mode: {value}.")
        return value

    @detector_mode.setter
    def detector_mode(self, value: DetectorMode) -> None:
        self._txt_log.debug(f"Setting detector mode: {value}.")
        self._detector_mode = value

    @property
    def detector_contrast(self) -> float:
        value = self._detector_contrast
        self._txt_log.debug(f"Getting detector contrast: {value}.")
        return value

    @detector_contrast.setter
    def detector_contrast(self, value: float) -> None:
        self._txt_log.debug(f"Setting detector contrast: {value}.")
        self._detector_contrast = value

    @property
    def detector_brightness(self) -> float:
        value = self._detector_brightness
        self._txt_log.debug(f"Getting detector brightness: {value}.")
        return value

    @detector_brightness.setter
    def detector_brightness(self, value: float) -> None:
        self._txt_log.debug(f"Setting detector brightness: {value}.")
        self._detector_brightness = value

    @property
    def source_tilt(self) -> SourceTilt:
        value = self._source_tilt
        self._txt_log.debug(f"Getting source tilt: {value}.")
        return value

    @source_tilt.setter
    def source_tilt(self, value: SourceTilt) -> None:
        self._txt_log.debug(f"Setting source tilt: {value}.")
        self._source_tilt = value

    @property
    def beam_current(self) -> float:
        return self._beam_current

    @beam_current.setter
    def beam_current(self, value: float) -> None:
        self._beam_current = value

    @property
    def scan_rotation(self) -> float:
        value = self._scan_rotation
        self._txt_log.debug(f"Getting scan rotation: {value}.")
        return value

    @scan_rotation.setter
    def scan_rotation(self, value: float) -> None:
        self._txt_log.debug(f"Setting scan rotation: {value}.")
        self._scan_rotation = value

    def blank(self) -> None:
        self._txt_log.debug("Blanking.")
        self._blanked = True

    def unblank(self) -> None:
        self._txt_log.debug("Unblanking.")
        self._blanked = False

    def start_acquisition(self) -> None:
        self._txt_log.debug("Acquisition started.")
        self._acquiring = True

    def stop_acquisition(self) -> None:
        self._txt_log.debug("Acquisition stopped.")
        self._acquiring = False

    def grab_frame(self, frame_store: FrameStore | None = None) -> Image:
        self._txt_log.debug("Grabbing frame.")
        width, height = self.resolution.to_tuple()
        pos = self._stage_position

        # stretch the beam shift
        theta = np.radians(pos.tilt)
        cos_theta = np.cos(theta)
        stretch = 1.0 / cos_theta if cos_theta > 1e-4 else 1.0

        cx = pos.x - self.beam_shift.x  # stage position and beam shift in x are flipped
        cy = -pos.y + self.beam_shift.y * stretch

        X, Y = SimulatedSample.world_grid(
            cx,
            cy,
            width,
            height,
            self.horizontal_field_width,
            self.vertical_field_width,
        )

        # rotate around the center
        if pos.rotation != 0.0:
            X, Y = SimulatedSample.rotate_grid(X, Y, 0, 0, pos.rotation)

        # handle tilt
        tilt_rad = math.radians(pos.tilt)
        if tilt_rad != 0.0:
            sin_t = np.sin(tilt_rad)
            cos_t = np.cos(tilt_rad)

            dY = Y - cy

            Y_s = cy + dY / cos_t

            for _ in range(6):
                Z_s = self._sample.surface_z(X, Y_s)
                Y_s = cy + (dY - Z_s * sin_t) / cos_t

            Z_surface = self._sample.surface_z(X, Y_s)
            Z_lab = -(Y_s - cy) * sin_t + Z_surface * cos_t

            Y = Y_s
        else:
            Z_surface = self._sample.surface_z(X, Y)
            Z_lab = Z_surface

        # sample texture + shading at the resolved surface hit position
        image = self._sample.sample(X, Y)
        image = image * self._sample.surface_shading(X, Y)
        image = np.clip(image, 0.0, 1.0)

        # defocus
        defocus_map = pos.z - self.working_distance - Z_lab + self._sample.drift[2]
        image = SimulatedSample.apply_focus_and_astigmatism(
            image=image,
            defocus_map=defocus_map,
            pixel_size=self.pixel_size,
            stigmator_x=self.stigmator.x,
            stigmator_y=self.stigmator.y,
        )

        # detector noise, brightness/contrast
        noise_std = 1e-4 / np.sqrt(max(self.dwell_time * self.line_integration, 1e-12))
        image += self._rng.normal(0.0, noise_std, image.shape)
        image = SimulatedSample.apply_brightness_contrast(
            image, self.detector_brightness, self.detector_contrast
        )

        # convert image based on bit depth
        max_adc = (1 << self.bit_depth) - 1
        image = Image(
            np.round(image * max_adc).astype(
                np.uint8 if self.bit_depth == 8 else np.uint16
            ),
            self.pixel_size,
        )

        if not self.scanning_area.is_full_frame():
            image = image.crop(self.scanning_area)

        self._current_image = image

        path = frame_store.path() if frame_store is not None else None
        if path is not None:
            image.save(path, ImageFormat.TIF)
        elif frame_store is not None:
            frame_store.save_to_memory(image)

        return image

    def get_image(self, crop_to_scanning_area: bool = False) -> Image:
        self._txt_log.debug("Getting an image.")
        image = (
            self._current_image
            if self._current_image is not None
            else self.grab_frame()
        )

        if crop_to_scanning_area and not self.scanning_area.is_full_frame():
            return image.crop(self.scanning_area)

        return image

    def rectangle_milling(
        self,
        milling_area: NMArea,
        milling_depth: float,
        direction: Direction,
        pattern_type: PatternType,
        do_not_mill: bool,
    ) -> None:
        self._txt_log.info(
            f"Fake milling in area {milling_area} with depth of {milling_depth} nm."
        )
        self._txt_log.info(
            f"Fake milling has direction {direction} and uses a pattern {pattern_type}."
        )
        if do_not_mill:
            self._txt_log.warning(
                "Actual milling would not be performed because do_not_mill is True."
            )

    @property
    def line_integration(self) -> int:
        value = self._line_integration
        self._txt_log.debug(f"Getting line integration: {value}.")
        return value

    @line_integration.setter
    def line_integration(self, value: int) -> None:
        self._txt_log.debug(f"Setting line integration: {value}.")
        self._line_integration = value

    @property
    def dwell_time(self) -> float:
        value = self._dwell_time
        self._txt_log.debug(f"Getting dwell time: {value}.")
        return value

    @dwell_time.setter
    def dwell_time(self, value: float) -> None:
        self._txt_log.debug(f"Setting dwell time: {value}.")
        self._dwell_time = value

    @property
    def bit_depth(self) -> int:
        value = self._bit_depth
        self._txt_log.debug(f"Getting bit depth: {value}.")
        return value

    @bit_depth.setter
    def bit_depth(self, value: int) -> None:
        self._txt_log.debug(f"Setting bit depth: {value}.")
        self._bit_depth = value

    @property
    def resolution(self) -> Resolution:
        value = self._resolution
        self._txt_log.debug(f"Getting resolution: {value}.")
        return value

    @resolution.setter
    def resolution(self, value: Resolution) -> None:
        self._txt_log.debug(f"Setting resolution: {value}.")
        self._resolution = value

    def clear_extended_resolution(self) -> None:
        pass

    @property
    def horizontal_field_width(self) -> float:
        value = self._horizontal_field_width
        self._txt_log.debug(f"Getting horizontal field width: {value}.")
        return value

    @horizontal_field_width.setter
    def horizontal_field_width(self, value: float) -> None:
        self._txt_log.debug(f"Setting horizontal field width: {value}.")
        self._horizontal_field_width = value

    @property
    def vertical_field_width(self) -> float:
        value = (
            self.horizontal_field_width * self.resolution.height / self.resolution.width
        )
        self._txt_log.debug(f"Getting vertical field width: {value}.")
        return value

    @vertical_field_width.setter
    def vertical_field_width(self, value: float) -> None:
        pixel_size = self.pixel_size
        self.resolution = Resolution(self.resolution.width, int(value / pixel_size))
        self._txt_log.info(
            f"Resolution set to: {self.resolution!s} (via setting vertical field width)."
        )

    @property
    def pixel_size(self) -> float:
        value = self.horizontal_field_width / self.resolution.width
        self._txt_log.debug(f"Getting pixel size: {value}.")
        return value

    @pixel_size.setter
    def pixel_size(self, value: float) -> None:
        self.resolution = Resolution(
            int(self.horizontal_field_width / value),
            int(self.vertical_field_width / value),
        )
        self._txt_log.info(
            f"Extended resolution set to: {self.resolution!s} (via setting pixel size)."
        )

    @property
    def scanning_area(self) -> RelativeArea:
        value = self._scanning_area
        self._txt_log.debug(f"Getting scanning area: {value}.")
        return value

    @scanning_area.setter
    def scanning_area(self, value: RelativeArea) -> None:
        self._txt_log.debug(f"Setting scanning area: {value}.")
        self._scanning_area = value

    @property
    def image_to_beam_shift(self) -> tuple[float, float]:
        return self._image_to_beam_shift

    @property
    def acquired_image_extension(self) -> str:
        return "png"

    def _apply_numeric(
        self,
        var: str,
        requested: float,
        *,
        rel_noise: float = 0.0,
        abs_noise: float = 0.0,
    ) -> float:
        lo, hi = self.limits(var)

        v = requested
        if rel_noise:
            v += requested * self._rng.normal(0.0, rel_noise)
        if abs_noise:
            v += self._rng.normal(0.0, abs_noise)

        return min(max(v, lo), hi)

    @property
    def txt_log(self) -> TextLogger:
        return self._txt_log


class SimulatedElectronBeamControl(SimulatedBeamControl):
    @classmethod
    def beam_type(cls) -> BeamType:
        return BeamType.ELECTRON


class SimulatedIonBeamControl(SimulatedBeamControl):
    @classmethod
    def beam_type(cls) -> BeamType:
        return BeamType.ION

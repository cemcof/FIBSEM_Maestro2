# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import math
import time
from abc import abstractmethod
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, TypeVar

from autoscript_sdb_microscope_client.enumerations import (
    ImageFileFormat,
    ImagingDevice,
    PatternScanDirection,
    ScanningMode,
)
from autoscript_sdb_microscope_client.sdb_microscope.beams._electron_beam import (
    ElectronBeam as ElectronBeamAs,
)
from autoscript_sdb_microscope_client.sdb_microscope.beams._ion_beam import (
    IonBeam as IonBeamAs,
)
from autoscript_sdb_microscope_client.structures import AdornedImage, GrabFrameSettings

from fibsem_maestro.core.area import NMArea, RelativeArea
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.detector_mode import DetectorMode
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.image import Image
from fibsem_maestro.core.lens_alignment import LensAlignment
from fibsem_maestro.core.provenance import Provenance
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.core.source_tilt import SourceTilt
from fibsem_maestro.core.stigmator import Stigmator
from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
from fibsem_maestro.microscope.error import MicroscopeError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autoscript_sdb_microscope_client.sdb_microscope_client import (
        SdbMicroscopeClient,
    )

    from fibsem_maestro.core.pattern_type import PatternType
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.store.frame.frame_store import FrameStore

BeamT = TypeVar(
    "BeamT",
    ElectronBeamAs,
    IonBeamAs,
)


class AutoscriptBeamControl[BeamT: (ElectronBeamAs, IonBeamAs)](BeamControl):
    _LIMITED_VARIABLES: ClassVar[Mapping[str, tuple[str, int | None, float]]] = (
        MappingProxyType(
            {
                # variable: (Autoscript attribute, axis of a 2D value or None, scale to Maestro units)
                "working_distance": ("working_distance", None, 1e9),  # m -> nm
                "horizontal_field_width": (
                    "horizontal_field_width",
                    None,
                    1e9,
                ),  # m -> nm
                "beam_current": ("beam_current", None, 1e9),  # A -> nA
                "stigmator_x": ("stigmator", 0, 1.0),
                "stigmator_y": ("stigmator", 1, 1.0),
                "lens_alignment_x": ("lens_alignment", 0, 1e9),  # m -> nm
                "lens_alignment_y": ("lens_alignment", 1, 1e9),  # m -> nm
                "source_tilt_x": ("source_tilt", 0, math.degrees(1.0)),  # rad -> deg
                "source_tilt_y": ("source_tilt", 1, math.degrees(1.0)),  # rad -> deg
                "beam_shift_x": ("beam_shift", 0, 1e9),  # m -> nm
                "beam_shift_y": ("beam_shift", 1, 1e9),  # m -> nm
            }
        )
    )

    def __init__(
        self,
        autoscript_microscope: SdbMicroscopeClient,
        txt_log: TextLogger,
    ):
        self._microscope = autoscript_microscope
        self._txt_log = txt_log

        self._extended_resolution: Resolution | None = (
            None  # extended resolution is set only if the required resolution is not standard
        )

        # fallback for line integration
        # only used if the linked Autoscript version does not support setting line_integration via scanning
        self._line_integration: int = 1

        # fallback for scanning area
        # only used if the linked Autoscript version does not support getting reduced_area via scanning.mode
        self._scanning_area: RelativeArea = RelativeArea.full()

    @classmethod
    def provenance(cls) -> Provenance:
        return Provenance.AUTOSCRIPT

    @property
    @abstractmethod
    def _beam(self) -> BeamT:
        pass

    @property
    @abstractmethod
    def _modality(self) -> str:
        pass

    @cached_property
    def _standard_resolutions(self) -> frozenset[str]:
        """Resolutions the instrument scans natively, as "WIDTHxHEIGHT"."""
        return frozenset(self._beam.scanning.resolution.available_values)

    @abstractmethod
    def select_modality(self):
        """
        This method is used to switch the microscope's modality between the Electron Beam (eb) mode and the Ion Beam
        (ib) mode. The selection of the modality will have an impact on starting and stopping the acquisition,
        grab and get image, and on the selected detector.

        The Electron Beam mode is always in Quad 1, while the Ion Beam mode is always in Quad 2.
        """

    @property
    def stigmator(self) -> Stigmator:
        value = Stigmator.from_point_autoscript(self._beam.stigmator.value)
        self._txt_log.debug(f"Getting stigmator ({self._modality}): {value}.")
        return value

    @stigmator.setter
    def stigmator(self, value: Stigmator) -> None:
        self._txt_log.debug(f"Setting stigmator ({self._modality}): {value}.")
        self._beam.stigmator.value = value.to_point_autoscript()

    @property
    def beam_shift(self) -> BeamShift:
        value = BeamShift.from_point_autoscript(self._beam.beam_shift.value)
        self._txt_log.debug(f"Getting beam shift ({self._modality}): {value}.")
        return value

    @beam_shift.setter
    def beam_shift(self, value: BeamShift) -> None:
        self._txt_log.debug(f"Setting beam shift ({self._modality}): {value}.")
        self._beam.beam_shift.value = value.to_point_autoscript()

    @property
    def detector_mode(self) -> DetectorMode:
        self.select_modality()

        raw = self._microscope.detector.mode.value
        try:
            value = DetectorMode(raw)
        except ValueError as e:
            raise MicroscopeError(
                f"Unknown detector mode '{raw}' reported by the microscope ({self._modality})."
            ) from e

        self._txt_log.debug(f"Getting detector mode ({self._modality}): {value}.")
        return value

    @detector_mode.setter
    def detector_mode(self, value: DetectorMode) -> None:
        self.select_modality()

        # the available modes depend on the active detector
        available = self._microscope.detector.mode.available_values
        if value.value not in available:
            raise MicroscopeError(
                f"Detector mode '{value.value}' is not available on the active "
                f"detector ({self._modality}); available: {', '.join(available)}."
            )

        self._txt_log.debug(f"Setting detector mode ({self._modality}) to: {value}.")
        self._microscope.detector.mode.value = value.value

    @property
    def detector_contrast(self) -> float:
        self.select_modality()
        value = self._microscope.detector.contrast.value
        self._txt_log.debug(f"Getting detector contrast ({self._modality}): {value}.")
        return value

    @detector_contrast.setter
    def detector_contrast(self, value: float) -> None:
        self.select_modality()
        self._txt_log.debug(
            f"Setting detector contrast ({self._modality}) to: {value}."
        )
        self._microscope.detector.contrast.value = value

    @property
    def detector_brightness(self) -> float:
        self.select_modality()
        value = self._microscope.detector.brightness.value
        self._txt_log.debug(f"Getting detector brightness ({self._modality}): {value}.")
        return value

    @detector_brightness.setter
    def detector_brightness(self, value: float) -> None:
        self.select_modality()
        self._txt_log.debug(
            f"Setting detector brightness ({self._modality}) to: {value}."
        )
        self._microscope.detector.brightness.value = value

    def blank(self) -> None:
        self.select_modality()
        self._txt_log.debug(f"Blanking beam ({self._modality}).")
        self._beam.blank()

    def unblank(self) -> None:
        self.select_modality()
        self._txt_log.debug(f"Unblanking beam ({self._modality}).")
        self._beam.unblank()

    def start_acquisition(self):
        self.select_modality()
        self._txt_log.debug(f"Starting acquisition ({self._modality}).")
        self._microscope.imaging.start_acquisition()

    def stop_acquisition(self):
        self.select_modality()
        self._txt_log.debug(f"Stopping acquisition ({self._modality}).")
        self._microscope.imaging.stop_acquisition()

    def get_image(self, crop_to_scanning_area: bool = False) -> Image:
        self.select_modality()
        self._txt_log.debug(f"Getting an image ({self._modality}).")

        adorned_image = self._microscope.imaging.get_image()
        raw = adorned_image.data
        self._txt_log.debug(
            f"AdornedImage: {adorned_image.width}x{adorned_image.height} "
            f"bit_depth={adorned_image.bit_depth} encoding={adorned_image.encoding} | "
            f"raw data: shape={raw.shape} dtype={raw.dtype} "
            f"min={raw.min()} max={raw.max()}"
        )
        image = Image.from_autoscript(adorned_image)

        if crop_to_scanning_area and not self.scanning_area.is_full_frame():
            return image.crop(self.scanning_area)

        return image

    def grab_frame(self, frame_store: FrameStore | None = None) -> Image:
        self.select_modality()
        self._txt_log.debug(f"Grabbing frame ({self._modality}).")

        imaging_settings = GrabFrameSettings(
            line_integration=self.line_integration,
            bit_depth=self.bit_depth,
            resolution=str(self.resolution),
            dwell_time=self.dwell_time * 1e-9,
        )

        if not self.scanning_area.is_full_frame():
            imaging_settings.reduced_area = self.scanning_area.to_autoscript()

        self._txt_log.info(
            "Acquiring image "
            f"(bit_depth={self.bit_depth}, "
            f"resolution={self.resolution}, "
            f"pixel_size={self.pixel_size} nm, "
            f"line_integration={self.line_integration}, "
            f"scanning_area={self.scanning_area}, "
            f"dwell_time={self.dwell_time} ns, "
            f"working_distance={self.working_distance} nm)"
        )

        path = frame_store.path() if frame_store is not None else None

        # if a physical frame store is provided, then always frab frame to disk since we want to store it anyway
        # the previous pathway using `imaging.grab_frame` and storing it using the `AdornedImage.save` method
        # failed to store image metadata in the TIFF file for whatever reason
        if path is not None:
            self._microscope.imaging.grab_frame_to_disk(
                str(path), ImageFileFormat.TIFF, imaging_settings
            )
            image = Image.from_autoscript(AdornedImage.load(str(path)))
        else:
            grabbed = self._microscope.imaging.grab_frame(imaging_settings)
            image = Image.from_autoscript(grabbed)
            if frame_store is not None:
                frame_store.save_to_memory(image)

        self._txt_log.info("Image grabbed.")
        return image

    def rectangle_milling(
        self,
        milling_area: NMArea,
        milling_depth: float,
        direction: Direction,
        pattern_type: PatternType,
        do_not_mill: bool,
    ) -> None:
        milling_area_m = milling_area.to_meters()
        milling_depth_m = milling_depth * 1e-9

        # get the center of the milling area in image coordinates
        image_center_x = milling_area_m.origin.x + milling_area_m.width / 2.0
        image_center_y = milling_area_m.origin.y + milling_area_m.height / 2.0

        # convert to pattern coordinates
        center_x = image_center_x - self.horizontal_field_width * 1e-9 / 2.0
        center_y = -image_center_y + self.vertical_field_width * 1e-9 / 2.0

        self.txt_log.debug(
            f"Milling area in patterning coordinates: center = [{center_x}, {center_y}], width = {milling_area_m.width}, height = {milling_area_m.height}"
        )

        if "ccs" in str(pattern_type).lower():
            self._txt_log.debug("Using cleaning cross section pattern.")
            pattern_fn = self._microscope.patterning.create_cleaning_cross_section
        elif "rcs" in str(pattern_type).lower():
            self._txt_log.debug("Using regular cross section pattern.")
            pattern_fn = self._microscope.patterning.create_regular_cross_section
        else:
            self._txt_log.debug(
                f"Pattern type {pattern_type!s} not recognized, using rectangle pattern."
            )
            pattern_fn = self._microscope.patterning.create_rectangle

        self.select_modality()
        self._microscope.patterning.clear_patterns()
        self._microscope.patterning.set_default_beam_type(int(self.beam_type()))
        self._microscope.patterning.set_default_application_file(str(pattern_type))

        time.sleep(1)

        pattern = pattern_fn(
            center_x=center_x,
            center_y=center_y,
            width=milling_area_m.width,
            height=milling_area_m.height,
            depth=milling_depth_m,
        )

        match direction:
            case Direction.UP:
                pattern.scan_direction = PatternScanDirection.BOTTOM_TO_TOP
            case Direction.DOWN:
                pattern.scan_direction = PatternScanDirection.TOP_TO_BOTTOM
            case Direction.LEFT | Direction.RIGHT:
                raise MicroscopeError(f"Invalid milling direction: {direction}")

        self._txt_log.info(
            f"Milling in progress [x = {pattern.center_x}, "
            f"y = {pattern.center_y}, width = {pattern.width}, "
            f"height = {pattern.height}, direction = {pattern.scan_direction}]."
        )

        if do_not_mill:
            self._txt_log.warning(
                "Skipping milling - `do_not_mill` option is True. Sleeping for 20 seconds..."
            )
            time.sleep(20)
        else:
            self._microscope.patterning.run()

        self._microscope.patterning.clear_patterns()

    @property
    def line_integration(self) -> int:
        # while Autoscript 4.13 does support getting and setting line integration via `scanning.line_integration`,
        # older Autoscript versions do not - we need a workaround for those versions
        try:
            li = self._beam.scanning.line_integration
        except Exception:
            self._txt_log.warning_once(
                "get-line-integration-old-autoscript",
                "Linked autoscript version does not support getting line_integration via scanning. "
                "Cannot get line_integration from Thermofisher's Microscope Control - "
                "Line integration must be specified externally in FIBSEM Maestro!",
            )
            li = self._line_integration

        self._txt_log.debug(f"Getting line integration ({self._modality}): {li}.")
        return li

    @line_integration.setter
    def line_integration(self, value: int) -> None:
        # while Autoscript 4.13 does support getting and setting line integration via `scanning.line_integration`,
        # older Autoscript versions do not - we need a workaround for those versions
        self._txt_log.debug(f"Setting line integration to ({self._modality}): {value}.")
        try:
            self._beam.scanning.line_integration = value
        except Exception:
            self._txt_log.warning_once(
                "set-line-integration-old-autoscript",
                "Linked autoscript version does not support setting line_integration via scanning. Using a fallback.",
            )

        self._line_integration = value

    @property
    def dwell_time(self) -> float:
        value = self._beam.scanning.dwell_time.value * 1e9
        self._txt_log.debug(f"Getting dwell time ({self._modality}): {value}.")
        return value

    @dwell_time.setter
    def dwell_time(self, value: float) -> None:
        self._txt_log.debug(f"Setting dwell time to ({self._modality}): {value}.")
        self._beam.scanning.dwell_time.value = value * 1e-9

    @property
    def bit_depth(self) -> int:
        value = self._beam.scanning.bit_depth
        self._txt_log.debug(f"Getting bit depth ({self._modality}): {value}.")
        return value

    @bit_depth.setter
    def bit_depth(self, value: int) -> None:
        if value not in (8, 16):
            raise MicroscopeError(f"Bit depth must be 8 or 16, not {value}.")
        self._txt_log.debug(f"Setting bit depth to ({self._modality}): {value}.")
        self._beam.scanning.bit_depth = value

    @property
    def resolution(self) -> Resolution:
        if self._extended_resolution is None:
            x, y = (
                self._beam.scanning.resolution.width,
                self._beam.scanning.resolution.height,
            )
            self._txt_log.debug(
                f"Getting standard resolution ({self._modality}): {x}, {y}."
            )
            return Resolution(x, y)

        self._txt_log.debug(
            f"Getting extended resolution ({self._modality}): {self._extended_resolution}."
        )
        return self._extended_resolution

    @resolution.setter
    def resolution(self, value: Resolution) -> None:
        resolution = str(value)
        if resolution in self._standard_resolutions:
            self._txt_log.debug(
                f"Setting standard resolution to ({self._modality}): {resolution}."
            )
            self._beam.scanning.resolution.value = resolution
            self._extended_resolution = None
            return
        self._txt_log.debug(
            f"Setting extended resolution to ({self._modality}): {resolution}."
        )
        # extended resolution cannot be set directly to the beam
        self._extended_resolution = value

    def clear_extended_resolution(self) -> None:
        self._extended_resolution = None

    @property
    def extended_resolution(self) -> Resolution | None:
        return self._extended_resolution

    @property
    def horizontal_field_width(self) -> float:
        value = self._beam.horizontal_field_width.value * 1e9  # m -> nm
        self._txt_log.debug(
            f"Getting horizontal field width ({self._modality}): {value}."
        )
        return value

    @horizontal_field_width.setter
    def horizontal_field_width(self, value: float) -> None:
        self._txt_log.debug(
            f"Setting horizontal field width to ({self._modality}): {value}."
        )
        self._beam.horizontal_field_width.value = value * 1e-9  # nm -> m

    @property
    def vertical_field_width(self) -> float:
        value = (
            self.horizontal_field_width * self.resolution.height / self.resolution.width
        )
        self._txt_log.debug(
            f"Getting vertical field width ({self._modality}): {value}."
        )

        return value

    @vertical_field_width.setter
    def vertical_field_width(self, value: float) -> None:
        # change the resolution of the image so that the requested vertical field width is achieved
        # without changing the pixel size
        pixel_size = self.pixel_size
        self.resolution = Resolution(self.resolution.width, round(value / pixel_size))
        self._txt_log.info(
            f"Extended resolution set to: {self.resolution!s} (via setting vertical field width)."
        )

    @property
    def pixel_size(self) -> float:
        value = self.horizontal_field_width / self.resolution.width
        self._txt_log.debug(f"Getting pixel size ({self._modality}): {value}.")
        return value

    @pixel_size.setter
    def pixel_size(self, value: float) -> None:
        # change the resolution of the image so that the pixel size matches the provided value
        # but the field width is not changed
        self.resolution = Resolution(
            round(self.horizontal_field_width / value),
            round(self.vertical_field_width / value),
        )
        self._txt_log.info(
            f"Extended resolution set to: {self.resolution!s} (via setting pixel size)."
        )

    @property
    def scan_rotation(self) -> float:
        value = math.degrees(self._beam.scanning.rotation.value)
        self._txt_log.debug(f"Getting scanning rotation ({self._modality}): {value}°.")
        return value

    @scan_rotation.setter
    def scan_rotation(self, value: float) -> None:
        self._txt_log.debug(f"Setting scanning rotation ({self._modality}): {value}°.")
        self._beam.scanning.rotation.value = math.radians(value)

    @property
    def scanning_area(self) -> RelativeArea:
        # while Autoscript 4.13 does support getting scanning_area via `scanning.mode.reduced_area`,
        # older Autoscript versions do not - we need a workaround for those versions
        try:
            if self._beam.scanning.mode.value == ScanningMode.REDUCED_AREA:
                area = RelativeArea.from_autoscript(
                    self._beam.scanning.mode.reduced_area.value
                )
            else:
                area = RelativeArea.full()
        except Exception:
            self._txt_log.warning_once(
                "get-scanning-area-old-autoscript",
                "Linked autoscript version does not support getting reduced_area via scanning.mode. "
                "Cannot get scanning area from Thermofisher's Microscope Control - "
                "Scanning area must be specified externally in FIBSEM Maestro!",
            )

            area = self._scanning_area

        self._txt_log.debug(f"Getting scanning area ({self._modality}): {area}")
        return area

    @scanning_area.setter
    def scanning_area(self, value: RelativeArea) -> None:
        if value.is_full_frame():
            self._txt_log.debug(f"Disabling scanning area ({self._modality}).")
            try:
                self._beam.scanning.mode.reduced_area.value = value.to_autoscript()
            except Exception:
                self._txt_log.warning_once(
                    "set-scanning-area-old-autoscript",
                    "Linked autoscript version does not support setting reduced_area via scanning.mode. Using a fallback.",
                )
            # used for acquisition started by start_acquisition()
            self._beam.scanning.mode.set_full_frame()
        else:
            self._txt_log.debug(f"Setting scanning area ({self._modality}): {value}.")
            # used for acquisition started by start_acquisition()
            self._beam.scanning.mode.set_reduced_area(
                left=value.origin.x,
                top=value.origin.y,
                width=value.width,
                height=value.height,
            )

        # fallback for older Autoscript versions
        self._scanning_area = value

    @property
    def beam_current(self) -> float:
        current = self._beam.beam_current.value * 1e9
        self._txt_log.debug(f"Getting beam current ({self._modality}): {current} nA.")
        return current

    @beam_current.setter
    def beam_current(self, value: float) -> None:
        self._txt_log.debug(f"Setting beam current ({self._modality}): {value} nA.")
        self._beam.beam_current.value = value * 1e-9

    @property
    def txt_log(self) -> TextLogger:
        return self._txt_log

    def limits(self, var: str) -> tuple[float, float]:
        """
        Return the range the instrument accepts for a variable, in Maestro units.

        Read from Autoscript, so it is right for the connected instrument and beam.
        Variables Autoscript reports no range for are unbounded, as in the other
        beam controls.

        Args:
            var: Name of the variable, e.g. `working_distance` or `stigmator_x`.

        Returns:
            A `(min, max)` tuple.
        """
        if (entry := self._LIMITED_VARIABLES.get(var)) is None:
            return (-math.inf, math.inf)

        attribute, axis, scale = entry
        if (setting := getattr(self._beam, attribute, None)) is None:
            return (-math.inf, math.inf)

        limits = setting.limits
        if axis is not None:
            limits = limits.limits_x if axis == 0 else limits.limits_y

        low, high = limits.min * scale, limits.max * scale
        return (min(low, high), max(low, high))


class AutoscriptElectronBeamControl(AutoscriptBeamControl[ElectronBeamAs]):
    @classmethod
    def beam_type(cls) -> BeamType:
        return BeamType.ELECTRON

    @property
    def _beam(self) -> ElectronBeamAs:
        return self._microscope.beams.electron_beam

    @property
    def _modality(self) -> str:
        return "eb"

    def select_modality(self) -> None:
        self._microscope.imaging.set_active_view(1)
        self._microscope.imaging.set_active_device(ImagingDevice.ELECTRON_BEAM)

    @property
    def working_distance(self):
        wd = self._beam.working_distance.value * 1e9  # m -> nm
        self._txt_log.debug(f"Getting working distance ({self._modality}): {wd}.")
        return wd

    @working_distance.setter
    def working_distance(self, value: float):
        self._txt_log.debug(f"Setting working distance ({self._modality}): {value}.")
        self._beam.working_distance.set_value_no_degauss(value * 1e-9)  # nm -> m

    @property
    def lens_alignment(self) -> LensAlignment:
        value = LensAlignment.from_point_autoscript(self._beam.lens_alignment.value)
        self._txt_log.debug(f"Getting lens alignment ({self._modality}): {value}.")
        return value

    @lens_alignment.setter
    def lens_alignment(self, value: LensAlignment):
        self._txt_log.debug(f"Setting lens alignment ({self._modality}): {value}.")
        self._beam.lens_alignment.value = value.to_point_autoscript()

    @property
    def source_tilt(self) -> SourceTilt:
        self.select_modality()
        value = SourceTilt.from_point_autoscript(self._beam.source_tilt.value)
        self._txt_log.debug(f"Getting source tilt ({self._modality}): {value}.")
        return value

    @source_tilt.setter
    def source_tilt(self, value: SourceTilt) -> None:
        self.select_modality()
        self._txt_log.debug(f"Setting source tilt ({self._modality}): {value}.")
        self._beam.source_tilt.value = value.to_point_autoscript()

    @property
    def image_to_beam_shift(self) -> tuple[int, int]:
        return (-1, 1)


class AutoscriptIonBeamControl(AutoscriptBeamControl[IonBeamAs]):
    @classmethod
    def beam_type(cls) -> BeamType:
        return BeamType.ION

    @property
    def _beam(self) -> IonBeamAs:
        return self._microscope.beams.ion_beam

    @property
    def _modality(self) -> str:
        return "ib"

    def select_modality(self) -> None:
        self._microscope.imaging.set_active_view(2)
        self._microscope.imaging.set_active_device(ImagingDevice.ION_BEAM)

    @property
    def working_distance(self):
        wd = self._beam.working_distance.value * 1e9  # m -> nm
        self._txt_log.debug(f"Getting working distance ({self._modality}): {wd}.")
        return wd

    @working_distance.setter
    def working_distance(self, value: float):
        self._txt_log.debug(f"Setting working distance ({self._modality}): {value}.")
        self._beam.working_distance.value = value * 1e-9  # nm -> m

    @property
    def lens_alignment(self) -> LensAlignment:
        self._txt_log.warning("Lens alignment is not defined for an ion beam.")
        return LensAlignment(x=0.0, y=0.0)

    @lens_alignment.setter
    def lens_alignment(self, value: LensAlignment):
        _ = value
        self._txt_log.warning("Lens alignment is not defined for an ion beam.")

    @property
    def source_tilt(self) -> SourceTilt:
        self._txt_log.warning("Source tilt is not defined for an ion beam.")
        return SourceTilt(x=0.0, y=0.0)

    @source_tilt.setter
    def source_tilt(self, value: SourceTilt) -> None:
        _ = value
        self._txt_log.warning("Source tilt is not defined for an ion beam.")

    @property
    def image_to_beam_shift(self) -> tuple[int, int]:
        return (-1, 1)

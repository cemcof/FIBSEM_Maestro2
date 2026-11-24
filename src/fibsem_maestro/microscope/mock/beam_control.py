# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


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

if TYPE_CHECKING:
    from fibsem_maestro.core.direction import Direction
    from fibsem_maestro.core.pattern_type import PatternType
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.store.frame.frame_store import FrameStore


class MockBeamControl(BeamControl):
    """
    Minimal mock implementation of BeamControl for testing purposes.
    """

    def __init__(self, txt_log: TextLogger):
        self._txt_log = txt_log

        self._working_distance: float = 0.0
        self._stigmator = Stigmator(0.0, 0.0)
        self._lens_alignment = LensAlignment(0.0, 0.0)
        self._beam_shift = BeamShift(0.0, 0.0)
        self._beam_current: float = 0.0

        self._detector_mode = DetectorMode.DEFAULT

        self._detector_contrast: float = 0.0
        self._detector_brightness: float = 0.0
        self._source_tilt = SourceTilt(0.0, 0.0)
        self._scan_rotation: float = 0.0

        self._line_integration: int = 1
        self._dwell_time: float = 100.0
        self._bit_depth: int = 8
        self._resolution = Resolution(1, 1)

        self._horizontal_field_width: float = 0.0
        self._vertical_field_width: float = 0.0
        self._pixel_size: float = 0.0

        self._scanning_area = RelativeArea.full()

        self._image_to_beam_shift: tuple[float, float] = (1.0, 1.0)

        self._current_image: Image | None = None

        self._blanked = False
        self._acquiring = False

    @classmethod
    def provenance(cls) -> Provenance:
        return Provenance.MAESTRO

    @property
    def working_distance(self) -> float:
        return self._working_distance

    @working_distance.setter
    def working_distance(self, value: float) -> None:
        self._working_distance = value

    @property
    def stigmator(self) -> Stigmator:
        return self._stigmator

    @stigmator.setter
    def stigmator(self, value: Stigmator) -> None:
        self._stigmator = value

    @property
    def lens_alignment(self) -> LensAlignment:
        return self._lens_alignment

    @lens_alignment.setter
    def lens_alignment(self, value: LensAlignment) -> None:
        self._lens_alignment = value

    @property
    def beam_shift(self) -> BeamShift:
        return self._beam_shift

    @beam_shift.setter
    def beam_shift(self, value: BeamShift) -> None:
        self._beam_shift = value

    @property
    def detector_mode(self) -> DetectorMode:
        return self._detector_mode

    @detector_mode.setter
    def detector_mode(self, value: DetectorMode) -> None:
        self._detector_mode = value

    @property
    def detector_contrast(self) -> float:
        return self._detector_contrast

    @detector_contrast.setter
    def detector_contrast(self, value: float) -> None:
        self._detector_contrast = value

    @property
    def detector_brightness(self) -> float:
        return self._detector_brightness

    @detector_brightness.setter
    def detector_brightness(self, value: float) -> None:
        self._detector_brightness = value

    @property
    def source_tilt(self) -> SourceTilt:
        return self._source_tilt

    @source_tilt.setter
    def source_tilt(self, value: SourceTilt) -> None:
        self._source_tilt = value

    @property
    def beam_current(self) -> float:
        return self._beam_current

    @beam_current.setter
    def beam_current(self, value: float) -> None:
        self._beam_current = value

    @property
    def scan_rotation(self) -> float:
        return self._scan_rotation

    @scan_rotation.setter
    def scan_rotation(self, value: float) -> None:
        self._scan_rotation = value

    @property
    def line_integration(self) -> int:
        return self._line_integration

    @line_integration.setter
    def line_integration(self, value: int) -> None:
        self._line_integration = value

    @property
    def dwell_time(self) -> float:
        return self._dwell_time

    @dwell_time.setter
    def dwell_time(self, value: float) -> None:
        self._dwell_time = value

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    @bit_depth.setter
    def bit_depth(self, value: int) -> None:
        self._bit_depth = value

    @property
    def resolution(self) -> Resolution:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Resolution) -> None:
        self._resolution = value

    def clear_extended_resolution(self) -> None:
        pass

    @property
    def horizontal_field_width(self) -> float:
        return self._horizontal_field_width

    @horizontal_field_width.setter
    def horizontal_field_width(self, value: float) -> None:
        self._horizontal_field_width = value

    @property
    def vertical_field_width(self) -> float:
        return self._vertical_field_width

    @vertical_field_width.setter
    def vertical_field_width(self, value: float) -> None:
        self._vertical_field_width = value

    @property
    def pixel_size(self) -> float:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: float) -> None:
        self._pixel_size = value

    @property
    def scanning_area(self) -> RelativeArea:
        return self._scanning_area

    @scanning_area.setter
    def scanning_area(self, value: RelativeArea) -> None:
        self._scanning_area = value

    def blank(self) -> None:
        self._blanked = True

    def unblank(self) -> None:
        self._blanked = False

    def start_acquisition(self) -> None:
        self._acquiring = True

    def stop_acquisition(self) -> None:
        self._acquiring = False

    def grab_frame(self, frame_store: FrameStore | None = None) -> Image:
        data = np.zeros((8, 8), dtype=np.uint16)
        image = Image(data, pixel_size=self.pixel_size or 1.0)

        self._current_image = image

        path = frame_store.path() if frame_store is not None else None
        if path is not None:
            image.save(path, ImageFormat.PNG)
        elif frame_store is not None:
            frame_store.save_to_memory(image)

        return image

    def get_image(self, crop_to_scanning_area: bool = False) -> Image:
        _ = crop_to_scanning_area
        if self._current_image is None:
            raise MicroscopeError("No mock image set.")
        return self._current_image

    def rectangle_milling(
        self,
        milling_area: NMArea,
        milling_depth: float,
        direction: Direction,
        pattern_type: PatternType,
        do_not_mill: bool,
    ) -> None:
        raise NotImplementedError()

    def set_mock_image(self, image: Image) -> None:
        """Set the image returned by grab_frame and get_image."""
        self._current_image = image

    @property
    def image_to_beam_shift(self) -> tuple[float, float]:
        return self._image_to_beam_shift

    def limits(self, var: str) -> tuple[float, float]:
        mock_limits: dict[str, tuple[float, float]] = {
            "working_distance": (1e3, 1e9),
            "stigmator_x": (-1.0, 1.0),
            "stigmator_y": (-1.0, 1.0),
            "lens_alignment_x": (-1e6, 1e6),
            "lens_alignment_y": (-1e6, 1e6),
            "beam_shift_x": (-1e7, 1e7),
            "beam_shift_y": (-1e7, 1e7),
            "detector_contrast": (0.0, 1.0),
            "detector_brightness": (0.0, 1.0),
            "source_tilt_x": (-10.0, 10.0),
            "source_tilt_y": (-10.0, 10.0),
            "horizontal_field_width": (1.0, 1e9),
            "vertical_field_width": (1.0, 1e9),
            "dwell_time": (1e-9, 1.0),
        }

        return mock_limits.get(var, (-float("inf"), float("inf")))

    @property
    def txt_log(self) -> TextLogger:
        return self._txt_log


class MockElectronBeamControl(MockBeamControl):
    @classmethod
    def beam_type(cls) -> BeamType:
        return BeamType.ELECTRON


class MockIonBeamControl(MockBeamControl):
    @classmethod
    def beam_type(cls) -> BeamType:
        return BeamType.ION

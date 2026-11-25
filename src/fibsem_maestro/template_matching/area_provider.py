# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fibsem_maestro.core.area import PixelArea
from fibsem_maestro.core.point import PixelPoint
from fibsem_maestro.core.registry import Registry
from fibsem_maestro.settings.template_matching_settings import (
    FullFrameMode,
    ReducedAreaMode,
    TemplateMatchingSettings,
)

if TYPE_CHECKING:
    from fibsem_maestro.core.image import Image8Bit
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.microscope.microscope import Microscope

AREA_PROVIDERS = Registry[type["AreaProvider"]]("area provider")

_SCAN_PARAMETERS = ("resolution", "dwell_time", "line_integration", "bit_depth")
"""
Beam properties a reduced-area scan must share with the full frame.

The resolution is included because the scanned area is computed on the
full-frame pixel grid.
"""


class AreaProvider(ABC):
    """
    Provides image regions for template matching.

    Abstracts over how image regions are obtained - either by cropping
    a full-frame acquisition or by commanding the microscope to scan
    only the regions of interest.

    Args:
        microscope: Interface to the electron microscope.
        settings: Template matching configuration.
        txt_log: Logger for diagnostic and status messages.
    """

    def __init__(
        self,
        microscope: Microscope,
        settings: TemplateMatchingSettings,
        txt_log: TextLogger,
    ) -> None:
        self._microscope = microscope
        self._settings = settings
        self._txt_log = txt_log

    @property
    @abstractmethod
    def last_full_frame(self) -> Image8Bit | None:
        """
        The full-frame image from the most recent acquisition, if available.

        Returns:
            The full-frame 8-bit image if the provider acquires full frames
            (e.g. `FullFrameAreaProvider`), or `None` if the provider
            only scans subregions of the field of view.
        """

    @abstractmethod
    def acquire_search_regions(self) -> list[Image8Bit]:
        """
        Acquire search regions for drift calculation.

        Each returned image corresponds to one configured template area,
        expanded by the correction margin so the template can be located
        even under drift.

        Returns:
            One search region per configured area.
        """

    @abstractmethod
    def acquire_full_frame(self) -> Image8Bit:
        """
        Acquire a full frame, after the full-frame dummy scans of the mode.

        Returns:
            The full frame as an 8-bit image.
        """

    @abstractmethod
    def acquire_template_regions(self) -> list[Image8Bit]:
        """
        Acquire exact template regions for template creation.

        Each returned image corresponds to one configured template area
        cropped to its exact bounds (no margin), suitable for saving as
        a reference template.

        Returns:
            One template region per configured area.
        """


@AREA_PROVIDERS.register("full_frame")
class FullFrameAreaProvider(AreaProvider):
    """
    Acquires a full frame and crops each configured area from it.
    """

    @property
    def last_full_frame(self) -> Image8Bit | None:
        return self._last_full_frame

    def acquire_search_regions(self) -> list[Image8Bit]:
        """
        Acquire a full frame and crop each area with the correction margin.

        Returns:
            One cropped search region per configured area.
        """
        frame = self.acquire_full_frame()
        return [
            frame.crop_with_padding(area, self._settings.correction_margin)
            for area in self._settings.areas
        ]

    def acquire_full_frame(self) -> Image8Bit:
        # grab dummy frames
        assert isinstance(self._settings.frame_grabbing_mode, FullFrameMode)
        for i in range(self._settings.frame_grabbing_mode.dummy_scans):
            self._txt_log.info(
                f"Performing dummy scan {i + 1}/{self._settings.frame_grabbing_mode.dummy_scans}."
            )
            self._microscope.beam.grab_frame()

        self._last_full_frame = self._microscope.beam.grab_frame().to_8bit()
        return self._last_full_frame

    def acquire_template_regions(self) -> list[Image8Bit]:
        """
        Acquire a full frame and crop each area to its exact bounds.

        Returns:
            One cropped template region per configured area.
        """
        self._last_full_frame = self._microscope.beam.grab_frame().to_8bit()
        return [self._last_full_frame.crop(area) for area in self._settings.areas]


@AREA_PROVIDERS.register("reduced_area")
class ReducedAreaProvider(AreaProvider):
    """
    Commands the microscope to scan only the configured areas.

    Instead of acquiring a full frame and cropping, each configured area
    is scanned individually as a separate reduced-area acquisition. This
    avoids scanning regions of the field of view that are not needed for
    template matching.
    """

    @property
    def last_full_frame(self) -> None:
        return None

    def acquire_search_regions(self) -> list[Image8Bit]:
        assert isinstance(self._settings.frame_grabbing_mode, ReducedAreaMode)
        mode = self._settings.frame_grabbing_mode
        beam = self._microscope.beam

        self._full_frame_dummy_scans()

        # switching to a reduced area swaps in its own scan parameters
        # TODO: this should be done differently, it breaks architecture
        scan_parameters = beam.collect_properties(list(_SCAN_PARAMETERS))

        resolution = beam.resolution
        frame = PixelArea(
            origin=PixelPoint(0, 0), width=resolution.width, height=resolution.height
        )
        # the same margin in pixels as the full-frame provider uses
        margin_px = int(self._settings.correction_margin / beam.pixel_size)

        regions = []
        for area in self._settings.areas:
            # include the correction margin in the area to scan, but only scan
            # what lies inside the frame
            wanted = area.to_pixels(resolution).expanded(margin_px)
            scanned = wanted.intersection(frame)
            with self._microscope.set_temporary_beam_property(
                "scanning_area", scanned.to_relative(resolution)
            ):
                beam.set_properties(scan_parameters)

                # grab dummy frames
                for i in range(mode.reduced_area_dummy_scans):
                    self._txt_log.info(
                        f"Performing reduced area dummy scan {i + 1}/{mode.reduced_area_dummy_scans}."
                    )
                    beam.grab_frame()

                region = beam.grab_frame().to_8bit()

            # the matcher expects the template in the centre of the search region,
            # so the part outside the frame is filled with replicated edge pixels,
            # as in the full-frame provider
            regions.append(region.padded_to(scanned, wanted))
        return regions

    def acquire_full_frame(self) -> Image8Bit:
        self._full_frame_dummy_scans()
        return self._microscope.beam.grab_frame().to_8bit()

    def _full_frame_dummy_scans(self) -> None:
        """Grab the full-frame dummy scans of the mode."""
        assert isinstance(self._settings.frame_grabbing_mode, ReducedAreaMode)
        scans = self._settings.frame_grabbing_mode.full_frame_dummy_scans
        for i in range(scans):
            self._txt_log.info(f"Performing full frame dummy scan {i + 1}/{scans}.")
            self._microscope.beam.grab_frame()

    def acquire_template_regions(self) -> list[Image8Bit]:
        beam = self._microscope.beam
        # switching to a reduced area swaps in its own scan parameters
        # TODO: this should be done differently, it breaks architecture
        parameters = beam.collect_properties(list(_SCAN_PARAMETERS))

        regions = []
        for area in self._settings.areas:
            with self._microscope.set_temporary_beam_property("scanning_area", area):
                beam.set_properties(parameters)
                frame = self._microscope.beam.grab_frame()
            regions.append(frame.to_8bit())
        return regions

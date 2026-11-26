# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import re
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.optimize import curve_fit

from fibsem_maestro.core.drift import Drift
from fibsem_maestro.core.image import Image8Bit
from fibsem_maestro.core.point import PixelPoint
from fibsem_maestro.logging.image.overlay import RectangleOverlay
from fibsem_maestro.settings.template_matching_settings import (
    StandardMode,
    SubpixelMode,
    TemplateMatchingSettings,
)
from fibsem_maestro.template_matching.error import TemplateMatchingError
from fibsem_maestro.template_matching.result import (
    ShiftPrecision,
    TemplateMatchResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from fibsem_maestro.logging.image.image_logger import ImageLogger
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.store.image.image_store import ImageStore
    from fibsem_maestro.template_matching.area_provider import AreaProvider

_TEMPLATE_PREFIX = "template_"
_TEMPLATE_SUFFIX = ".tif"
_TEMPLATE_NAME_RE = re.compile(
    rf"{re.escape(_TEMPLATE_PREFIX)}\d+{re.escape(_TEMPLATE_SUFFIX)}"
)


class TemplateMatching:
    """
    Detects and measures image drift using normalized cross-correlation.

    Compares a newly acquired image against stored reference templates to
    determine how much the field of view has shifted since the templates were
    created. The measured drift can then be used by a drift correction action
    to compensate for sample movement between slices.

    Each template corresponds to a configured region of interest in the image.
    Multiple templates can be used simultaneously to improve robustness -
    the final drift estimate is the mean across all templates that meet the
    confidence threshold.

    Args:
        area_provider: Provider for the image regions for template matching.
        settings: Template matching configuration.
        image_store: Store for persisting and retrieving template images.
        txt_log: Logger for diagnostic and status messages.
        img_log: Logger for heatmap and overlay images.
    """

    def __init__(
        self,
        area_provider: AreaProvider,
        settings: TemplateMatchingSettings,
        image_store: ImageStore[Image8Bit],
        txt_log: TextLogger,
        img_log: ImageLogger,
    ):
        self._area_provider = area_provider
        self._settings = settings
        self._image_store = image_store
        self._txt_log = txt_log
        self._img_log = img_log

    def create_templates(self, store: ImageStore[Image8Bit] | None = None) -> None:
        """
        Acquire reference images and save an averaged template for each area.

        All existing templates in the target store are deleted first, including
        those of areas no longer configured.

        Then `settings.template_scans` sets of template regions are acquired, the
        crops are validated for consistency, and the per-area average is saved
        as the reference template.

        Args:
            store: Store to write templates to. If `None`, the current slice's store is used.

        Raises:
            TemplateMatchingError: If no template areas are configured, if the
                cross-correlation confidence between any two scans is too low,
                or if the drift between any two scans exceeds the configured
                maximum.
        """
        if not self._settings.areas:
            raise TemplateMatchingError(
                "Cannot create templates: no template matching areas are configured."
            )

        if store is None:
            store = self._image_store
        self._delete_templates(store)

        # acquire template regions for each scan
        all_scans: list[list[Image8Bit]] = []
        for i in range(self._settings.template_scans):
            self._txt_log.info(
                f"Acquiring template image {i + 1}/{self._settings.template_scans}."
            )
            all_scans.append(self._area_provider.acquire_template_regions())

        # transpose from per-scan to per-area, validate, and save
        for area_index in range(len(self._settings.areas)):
            template_scans = [scan[area_index] for scan in all_scans]
            self._validate_template_scan_consistency(template_scans, area_index)
            self._save_averaged_template(template_scans, area_index, store)

    def templates_exist(self, store: ImageStore[Image8Bit] | None = None) -> bool:
        """
        Check whether a template is stored for every configured area.

        Args:
            store: Store to check. If `None`, the current slice's store is used.

        Returns:
            `True` if areas are configured and each has a stored template.
        """
        if store is None:
            store = self._image_store

        return bool(self._settings.areas) and all(
            store.exists(self._construct_template_name(i))
            for i in range(len(self._settings.areas))
        )

    def copy_templates(
        self, src: ImageStore[Image8Bit], dest: ImageStore[Image8Bit]
    ) -> None:
        """
        Copy the template of every configured area unchanged between stores.

        Args:
            src: Store to read the templates from.
            dest: Store to write the templates to.
        """
        for i in range(len(self._settings.areas)):
            self._copy_template(i, src, dest)

    def update_templates(self, slice_number: int, confidence: float | None) -> None:
        """
        Refresh templates for the next slice, subject to the update policy.

        If the update conditions are met (slice number matches the update
        frequency or confidence is below the update threshold), acquires
        fresh frames and creates new templates for the next slice. Otherwise,
        copies the existing templates forward unchanged.

        Args:
            slice_number: The current slice index.
            confidence: Average template matching confidence from
                the most recent drift calculation. `None` if no confidence is available.
        """
        if self._should_update_templates(slice_number, confidence):
            # acquire new templates for the next slice
            self.create_templates(self._image_store.next)

        else:
            # copy templates from the current slice
            self.copy_templates(self._image_store, self._image_store.next)

    def calculate_drift(self, whole_frame: bool = False) -> Drift:
        """
        Calculate the image drift relative to the stored templates.

        Args:
            whole_frame: Search the templates in the whole frame, at pixel
                precision and without `maximal_drift`, instead of in the
                search regions around their areas.

        Returns:
            A `Drift` instance containing the x and y shift in nanometers
            and the average template matching confidence. If all template
            matches fall below the confidence threshold, `x` and `y` are
            `None`.
        """
        if whole_frame:
            self._txt_log.info(
                "Searching the whole frame for the templates, at pixel precision "
                "and without the maximal drift."
            )
            frame = self._area_provider.acquire_full_frame()
            matches = self._calculate_whole_frame_matches(frame)
            pixel_size = frame.pixel_size
            max_drift = None
        else:
            self._txt_log.info("Obtaining search regions for template matching.")
            search_regions = self._area_provider.acquire_search_regions()

            # perform template matching for each template
            match self._settings.matching_mode:
                case StandardMode():
                    self._txt_log.debug(
                        "Calculating template match in pixel resolution."
                    )
                    matches = self._calculate_template_matches(
                        search_regions, self._calculate_match
                    )
                case SubpixelMode():
                    self._txt_log.debug(
                        "Calculating template match in sub-pixel resolution."
                    )
                    matches = self._calculate_template_matches(
                        search_regions, self._calculate_subpixel_match
                    )

            frame = self._area_provider.last_full_frame
            pixel_size = search_regions[0].pixel_size
            max_drift = self._settings.maximal_drift

        # log the results of template matching
        self._log_heatmaps(matches)

        # log shifts in the image only if full frame has been acquired
        if frame is not None:
            self._log_image_shifts(frame, matches)

        # get image shift from template matching results
        shift = self._matches_to_image_shift(matches, pixel_size, max_drift)

        # average confidence of the individual template matches
        confidence = self._get_average_confidence(matches)

        if isinstance(shift, str):
            return Drift(x=None, y=None, confidence=confidence, reason=shift)

        return Drift(x=shift[0], y=shift[1], confidence=confidence)

    def _delete_templates(self, store: ImageStore[Image8Bit]) -> None:
        """
        Delete every template in a store, including those of areas no longer configured.

        Templates always form a contiguous run `template_0`, `template_1`, ...:
        creation saves the areas in order, copying copies them in order, and a
        failed creation leaves at most a prefix. They are therefore deleted in
        index order until the first missing one.

        Args:
            store: Store to delete the templates from.
        """
        index = 0
        while store.exists(name := self._construct_template_name(index)):
            self._txt_log.debug(f"Deleting template '{name}' from slice {store.slice}.")
            store.delete(name)
            index += 1

    def _calculate_match(
        self, template: Image8Bit, image: Image8Bit, expected: PixelPoint | None = None
    ) -> TemplateMatchResult[int]:
        """
        Compute the normalized cross-correlation between `template` and `image`.

        Optionally applies a Gaussian blur to both images before matching in order
        to suppress high-frequency noise that could otherwise dominate the
        correlation. The peak of the resulting heatmap is located and its offset
        from the heatmap centre is returned as the image-space displacement.

        Args:
            template: Reference patch to locate within `image`.
            image: Image region to search; must be larger than `template` by at
                least the expected drift on each side.
            expected: Top-left corner of the template in `image` without
                drift, in pixels; None if the template lies in its centre.

        Returns:
            A `TemplateMatchResult` containing the pixel displacement
            `(dx, dy)` from the expected location to the best-match location,
            together with the peak normalised cross-correlation score and the
            full heatmap array.
        """
        image = image.blured(self._settings.blur)
        template = template.blured(self._settings.blur)

        heatmap = cv2.matchTemplate(
            image,
            template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)

        best_x, best_y = max_loc

        # the heatmap is indexed by the template's top-left corner; without
        # drift a centred template lies at the centre of the heatmap
        if expected is None:
            expected = PixelPoint(x=heatmap.shape[1] // 2, y=heatmap.shape[0] // 2)

        dx = best_x - expected.x  # horizontal shift
        dy = best_y - expected.y  # vertical shift

        return TemplateMatchResult(
            dx=dx,
            dy=dy,
            confidence=float(max_val),
            heatmap=heatmap,
        )

    def _calculate_subpixel_match(
        self, template: Image8Bit, image: Image8Bit
    ) -> TemplateMatchResult[float]:
        """
        Compute the normalized cross-correlation between `template` and `image` with sub-pixel accuracy.

        Extends `_calculate_match` with two additional steps to achieve sub-pixel
        precision. First, both images are upsampled by the factor specified in
        the settings, increasing the effective pixel grid resolution.
        Second, independent 1D Gaussians are fitted to the correlation peak along
        each axis, interpolating the true maximum between discrete samples. The
        returned displacement is expressed in original (non-upsampled) pixel units.

        Args:
            template: Reference patch to locate within `image`.
            image: Image region to search; must be larger than `template` by at
                least the expected drift on each side.

        Returns:
            A `TemplateMatchResult` containing the sub-pixel displacement
            `(dx, dy)` from the image centre to the best-match location,
            together with the peak normalised cross-correlation score and the
            full heatmap array. If the Gaussian fit fails along either axis,
            that axis falls back to integer-pixel accuracy.
        """
        assert isinstance(self._settings.matching_mode, SubpixelMode)
        upsampling_factor = self._settings.matching_mode.upsampling_factor

        # blur the images, if requested
        image = image.blured(self._settings.blur)
        template = template.blured(self._settings.blur)

        # upsample the images
        image_upsampled = image.upsampled(upsampling_factor)
        template_upsampled = template.upsampled(upsampling_factor)

        # calculate match
        heatmap = cv2.matchTemplate(
            image_upsampled, template_upsampled, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        best_x_int, best_y_int = max_loc

        # restrict the gaussian fit to a symmetric neighbourhood of the integer peak
        neighborhood = int(np.ceil(10 * upsampling_factor))

        # extract a 1D profile along the x-axis at the best y position and fit a gaussian
        # to find the sub-pixel x-coordinate of the peak
        x_start = max(0, best_x_int - neighborhood)
        x_end = min(heatmap.shape[1], best_x_int + neighborhood + 1)
        x_indices = np.arange(x_start, x_end)
        x_profile = heatmap[best_y_int, x_start:x_end].squeeze()
        best_x = self._fit_subpixel_peak(x_profile, x_indices, best_x_int, max_val, "x")

        # extract a 1D profile along the y-axis at the best x position and fit a gaussian
        # to find the sub-pixel y-coordinate of the peak
        y_start = max(0, best_y_int - neighborhood)
        y_end = min(heatmap.shape[0], best_y_int + neighborhood + 1)
        y_indices = np.arange(y_start, y_end)
        y_profile = heatmap[y_start:y_end, best_x_int].squeeze()
        best_y = self._fit_subpixel_peak(y_profile, y_indices, best_y_int, max_val, "y")

        # without drift the peak lies at the centre of the heatmap, (size - 1) / 2;
        # express the peak as an offset from it, in pixels of the original image
        best_x = (best_x - (heatmap.shape[1] - 1) / 2.0) / upsampling_factor
        best_y = (best_y - (heatmap.shape[0] - 1) / 2.0) / upsampling_factor

        return TemplateMatchResult(
            dx=best_x,
            dy=best_y,
            confidence=float(max_val),
            heatmap=heatmap,
        )

    def _validate_template_scan_consistency(
        self, scans: list[Image8Bit], area_index: int
    ) -> None:
        """
        Validate that all scans of a template area are mutually consistent.

        Compares each unique pair of scans. Raises if any pair drifted apart by
        more than `settings.maximal_drift` (nm) or has a confidence below
        `settings.min_confidence`.

        Args:
            scans: Cropped 8-bit images of the area from each acquired frame.
            area_index: Zero-based index of the template area.

        Raises:
            TemplateMatchingError: If any pair of scans has insufficient
                confidence or excessive drift.
        """
        if len(scans) <= 1:
            return

        self._txt_log.debug("Validating consistency of template scans.")
        for i in range(len(scans)):
            for j in range(i + 1, len(scans)):
                drift_nm = _shift_between_px(scans[i], scans[j]) * scans[i].pixel_size
                self._txt_log.debug(f"Scans {i} x {j}: drift={drift_nm:.1f} nm")

                if (
                    max_drift := self._settings.maximal_drift
                ) is not None and drift_nm > max_drift:
                    raise TemplateMatchingError(
                        f"Cannot create template for area {area_index}: "
                        f"drift between scan {i} and scan {j} is "
                        f"{drift_nm:.1f} nm, exceeding the maximum of "
                        f"{max_drift:.1f} nm."
                    )

                # same-sized images give the correlation at zero offset only
                result = self._calculate_match(scans[i], scans[j])
                self._txt_log.debug(
                    f"Scans {i} x {j}: confidence={result.confidence:.3f}"
                )

                if (
                    min_conf := self._settings.min_confidence
                ) is not None and result.confidence < min_conf:
                    raise TemplateMatchingError(
                        f"Cannot create template for area {area_index}: "
                        f"confidence between scan {i} and scan {j} is "
                        f"{result.confidence:.3f}, below the required "
                        f"minimum of {self._settings.min_confidence:.3f}."
                    )

    def _save_averaged_template(
        self,
        scans: list[Image8Bit],
        area_index: int,
        store: ImageStore[Image8Bit] | None = None,
    ) -> None:
        """
        Average multiple scans and save the result as the reference template.

        Args:
            scans: Cropped 8-bit images to average.
            area_index: Zero-based index of the template area.
            store: Destination store. If `None`, the current slice's store
                is used.
        """
        averaged = Image8Bit(
            np.mean(scans, axis=0).astype(np.uint8),
            pixel_size=scans[0].pixel_size,
        )
        self._save_template(averaged, index=area_index, image_store=store)

    def _calculate_whole_frame_matches(
        self, frame: Image8Bit
    ) -> list[TemplateMatchResult[int]]:
        """
        Locate each template in a whole frame, at pixel precision.

        Args:
            frame: The full frame to search.

        Returns:
            One `TemplateMatchResult` per configured template area, in the
            same order as `settings.areas`.
        """
        return [
            self._calculate_match(
                self._load_template(i),
                frame,
                expected=area.to_pixels(frame.resolution).origin,
            )
            for i, area in enumerate(self._settings.areas)
        ]

    def _calculate_template_matches(
        self,
        search_regions: list[Image8Bit],
        match_fn: Callable[[Image8Bit, Image8Bit], TemplateMatchResult[ShiftPrecision]],
    ) -> list[TemplateMatchResult[ShiftPrecision]]:
        """
        Load each template and compute its match within the corresponding search region.

        Args:
            search_regions: Pre-acquired search regions, one per configured template area.
            match_fn: Method used to compute each match. Determines whether
                results carry integer or sub-pixel float precision, e.g.
                `self._calculate_match` or `self._calculate_subpixel_match`.

        Returns:
            One `TemplateMatchResult` per configured template area, in the
            same order as `settings.areas`. The numeric type of `dx` and `dy`
            matches the return type of `match_fn`.
        """
        matches = []
        for i, region in enumerate(search_regions):
            template = self._load_template(i)
            matches.append(match_fn(template, region))
        return matches

    def _matches_to_image_shift(
        self,
        matches: list[TemplateMatchResult[ShiftPrecision]],
        pixel_size: float,
        max_drift: float | None,
    ) -> tuple[float, float] | str:
        """
        Convert template match results to a mean image shift in nanometers.

        Converts each match's pixel displacement to nanometers and averages
        the results across all valid matches. Matches below the confidence
        threshold or above `max_drift` are logged as warnings and excluded.

        Args:
            matches: Template match results, one per configured area.
            pixel_size: Physical size of one pixel in nanometers.
            max_drift: Largest accepted drift of a match in nanometers; None
                for no limit.

        Returns:
            A `(dx_nm, dy_nm)` tuple representing the mean image shift, or
            why there is none if fewer than `min_valid_areas` matches are valid.
        """
        shifts_x = []
        shifts_y = []
        for i, match in enumerate(matches):
            # convert shift to nm
            dx_nm = match.dx * pixel_size
            dy_nm = match.dy * pixel_size

            self._txt_log.info(
                f"Image shift for template {i}: {dx_nm},{dy_nm}. Confidence: {match.confidence}."
            )

            # ignore matches for which the confidence is too low
            if (
                min_conf := self._settings.min_confidence
            ) is not None and match.confidence < min_conf:
                self._txt_log.warning(
                    f"Template match confidence ({match.confidence}) is too low (limit: {self._settings.min_confidence}). "
                    f"Ignoring match for template {i}."
                )
                continue

            # ignore matches for which the drift is too large
            drift = np.sqrt(dx_nm**2 + dy_nm**2)
            if max_drift is not None and drift > max_drift:
                self._txt_log.warning(
                    f"Drift ({drift}) is too large (limit: {max_drift}). "
                    f"Ignoring match for template {i}."
                )
                continue

            shifts_x.append(dx_nm)
            shifts_y.append(dy_nm)

        required = self._settings.min_valid_areas
        if len(shifts_x) < required:
            reason = (
                f"{len(shifts_x)} of {len(matches)} areas gave a valid match; "
                f"at least {required} required"
            )
            self._txt_log.warning(f"Template matching failed: {reason}.")
            return reason

        # calculate average image shift
        return float(np.mean(shifts_x)), float(np.mean(shifts_y))

    def _get_average_confidence(
        self, matches: list[TemplateMatchResult[ShiftPrecision]]
    ) -> float:
        """
        Compute the mean confidence across all template match results.

        Args:
            matches: Template match results to average.

        Returns:
            Mean normalised cross-correlation confidence across all matches.
        """
        return float(np.mean([match.confidence for match in matches]))

    def _fit_subpixel_peak(
        self,
        profile: NDArray[Any],
        indices: NDArray[Any],
        peak_int: int,
        max_val: float,
        axis_name: str,
    ) -> float:
        """
        Fit a 1-D Gaussian to an NCC profile and return the sub-pixel peak.

        Extracts the peak location from a Gaussian fit to a 1-D slice of the
        NCC correlation map. Falls back to the integer peak when the fit fails.

        Args:
            profile: 1-D NCC values extracted along the axis of interest.
            indices: Array of integer coordinate values corresponding to
                `profile`, in upsampled pixel units.
            peak_int: Integer coordinate of the NCC maximum along this axis,
                used as the initial guess and as the fallback value.
            max_val: Global NCC maximum, used as the amplitude initial guess.
            axis_name: Human-readable axis label (e.g. "x", "y").

        Returns:
            Sub-pixel peak coordinate in upsampled pixel units, or
            `float(peak_int)` if the fit failed.
        """
        try:
            popt, _ = curve_fit(
                gauss,
                indices,
                profile,
                p0=[max_val, peak_int, len(profile) / 4.0, profile.min()],
            )
            return float(popt[1])
        except Exception:
            self._txt_log.warning(
                f"Sub-pixel Gaussian fit along {axis_name} axis failed - falling back to integer accuracy."
            )
            return float(peak_int)

    def _should_update_templates(
        self, slice_number: int, confidence: float | None
    ) -> bool:
        """
        Decide whether templates should be refreshed for the next slice.

        Templates are updated if the slice number matches the configured update
        frequency or if the average confidence falls below the update confidence
        limit.

        If no confidence is provided, the update policy is based on the slice number alone.

        Args:
            slice_number: The current slice index.
            confidence: Average template matching confidence from the most
                recent drift calculation.

        Returns:
            `True` if templates should be refreshed, `False` otherwise.
        """
        if (
            (freq := self._settings.update_frequency) is not None
            # do not update on the first slice
            and slice_number != 1
            # slices are numbered from 1
            and (slice_number - 1) % freq == 0
        ):
            self._txt_log.info(
                f"Updating templates: slice {slice_number} matches update frequency ({freq})."
            )
            return True

        if (
            (conf_lim := self._settings.update_confidence_limit) is not None
            and confidence is not None
            and confidence < conf_lim
        ):
            self._txt_log.info(
                f"Updating templates: template matching confidence ({confidence:.4f}) is below the limit ({conf_lim:.4f})."
            )
            return True

        self._txt_log.debug("Not updating templates.")
        return False

    def _save_template(
        self,
        template: Image8Bit,
        index: int,
        image_store: ImageStore[Image8Bit] | None = None,
    ) -> None:
        """
        Save a template to the image store.

        Args:
            template: 8-bit image to save.
            index: Zero-based template index.
            image_store: Destination store. Defaults to the current-slice store if `None`.
        """
        # default: current slice
        store = image_store or self._image_store

        self._txt_log.debug(f"Saving template {index}.")
        store.write(self._construct_template_name(index), template)

    def _load_template(
        self, index: int, image_store: ImageStore[Image8Bit] | None = None
    ) -> Image8Bit:
        """
        Read a previously saved template from the image store.

        Args:
            index: Zero-based template index matching the one used when saving.
            image_store: Source store. Defaults to the current-slice store if `None`.

        Returns:
            The stored 8-bit template image.
        """
        # default: current slice
        store = image_store or self._image_store

        self._txt_log.debug(f"Loading template {index}.")
        return store.read(self._construct_template_name(index))

    def _copy_template(
        self, index: int, src: ImageStore[Image8Bit], dest: ImageStore[Image8Bit]
    ) -> None:
        """
        Copy a template image from one store to another without modification.

        Args:
            index: Zero-based template index.
            src: Store to read the template from.
            dest: Store to write the template to.
        """
        self._txt_log.debug(
            f"Copying template {index} from slice {src.slice} to slice {dest.slice}."
        )
        image = src.read(self._construct_template_name(index))
        dest.write(self._construct_template_name(index), image)

    def _construct_template_name(self, index: int) -> str:
        """
        Build the filename used to persist a template image.

        Args:
            index: Zero-based template index.

        Returns:
            A filename of the form `<name>_template_<index>.tif`,
            with spaces in the name replaced by underscores.
        """
        return f"template_{index}.tif"

    def _log_heatmaps(self, matches: list[TemplateMatchResult[ShiftPrecision]]) -> None:
        """
        Save a heatmap image for each template match result to the image log.

        Args:
            matches: Template match results whose heatmaps should be logged.
        """
        for i, match in enumerate(matches):
            self._log_heatmap(match.heatmap, i)

    def _log_image_shifts(
        self, image: Image8Bit, matches: list[TemplateMatchResult[ShiftPrecision]]
    ) -> None:
        """
        Overlay the original and shifted template areas on the drift-correction frame and log it.

        Args:
            image: Drift-correction frame to annotate.
            matches: Template match results providing per-area displacements
                and confidence scores.
        """

        overlays = []
        for i, (area, match) in enumerate(zip(self._settings.areas, matches)):
            area_px = area.to_pixels(image.resolution)

            overlays.append(
                RectangleOverlay(
                    x=area_px.origin.x,
                    y=area_px.origin.y,
                    width=area_px.width,
                    height=area_px.height,
                    color="red",
                )
            )

            if (
                min_conf := self._settings.min_confidence
            ) is not None and match.confidence < min_conf:
                self._txt_log.debug(
                    f"Confidence is too low for template {i}. Shifted template area will not be displayed in the log."
                )
                continue

            shifted_area_px = area_px.shifted(
                PixelPoint(x=int(match.dx), y=int(match.dy))
            )

            overlays.append(
                RectangleOverlay(
                    x=shifted_area_px.origin.x,
                    y=shifted_area_px.origin.y,
                    width=shifted_area_px.width,
                    height=shifted_area_px.height,
                    color="blue",
                )
            )

        try:
            self._img_log.save_image(
                "template_matching_log.png",
                image,
                overlays,
                title="Template matching drift correction",
            )
        except Exception as e:
            self._txt_log.warning(
                f"Could not log a template matching drift correction result image: {e}"
            )

    def _log_heatmap(self, heatmap: NDArray[Any], index: int) -> None:
        """
        Save a single normalised cross-correlation heatmap to the image log.

        Args:
            heatmap: 2D array of normalised cross-correlation scores.
            index: Zero-based template index, used to construct the filename.
        """
        try:
            self._img_log.save_image(
                f"template_matching_heatmap_{index}.png",
                heatmap,
                overlays=None,
                title=f"Template matching heatmap for template {index}",
            )
        except Exception as e:
            self._txt_log.warning(
                f"Could not log a template matching drift correction heatmap: {e}"
            )


def gauss(x: NDArray[np.floating[Any]], *p: float) -> NDArray[np.floating[Any]]:
    """
    Evaluate a 1-D Gaussian with a vertical offset.

    Args:
        x: Input values at which the Gaussian is evaluated.
        *p: Gaussian parameters in order:

            - `a` - Amplitude (peak height above the baseline).
            - `b` - Mean (centre position along the x axis).
            - `c` - Standard deviation (controls the width).
            - `d` - Baseline offset (vertical shift).

    Returns:
        Array of the same shape as `x` containing the evaluated Gaussian values.

    Raises:
        ValueError: If fewer than four values are supplied in `*p`.
    """
    if len(p) < 4:
        raise ValueError(f"Expected 4 Gaussian parameters (a, b, c, d), got {len(p)}.")
    a, b, c, d = p
    return a * np.exp(-np.power((x - b), 2.0) / (2.0 * c**2)) + d


def _shift_between_px(a: Image8Bit, b: Image8Bit) -> float:
    """
    Measure how far the content of `b` is shifted against `a`, in pixels.

    Uses phase correlation, which, unlike template matching, works on two
    images of the same size.

    Args:
        a: Reference image.
        b: Image of the same size.

    Returns:
        The length of the shift in pixels.
    """
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(
        np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), window
    )

    return float(np.hypot(dx, dy))

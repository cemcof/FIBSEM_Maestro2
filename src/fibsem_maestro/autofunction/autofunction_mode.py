# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from __future__ import annotations

import time
from abc import ABC, abstractmethod
from itertools import groupby
from typing import TYPE_CHECKING, Any

import numpy as np

from fibsem_maestro.autofunction.error import AutofunctionError
from fibsem_maestro.autofunction.result import AutofunctionResult
from fibsem_maestro.autofunction.sweep_step import SweepStep
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.image_tools import get_stripes
from fibsem_maestro.core.registry import Registry
from fibsem_maestro.settings.autofunction_settings import (
    AutoscriptAutoFocus,
    AutoscriptAutoFocusMethod,
    AutoscriptAutoLensAlignment,
    AutoscriptAutoLensAlignmentModulationType,
    AutoscriptAutoSourceTilt,
    AutoscriptAutoStigmator,
    AutoscriptAutoStigmatorMethod,
    AutoscriptFunctionBase,
)
from fibsem_maestro.settings.autofunction_settings import BasicMode as BasicModeSettings
from fibsem_maestro.settings.autofunction_settings import LineMode as LineModeSettings

if TYPE_CHECKING:
    from collections.abc import Generator

    from autoscript_sdb_microscope_client.sdb_microscope_client import (
        SdbMicroscopeClient,
    )

    from fibsem_maestro.autofunction.autofunction_context import AutofunctionContext
    from fibsem_maestro.autofunction.jobs_manager import JobsManager
    from fibsem_maestro.core.image import Image
    from fibsem_maestro.imaging.imaging import Imaging

# registry of autofunction modes
AUTOFUNCTION_MODES = Registry[type["AutofunctionMode"]]("autofunction mode")


class AutofunctionMode(ABC):
    """
    Abstract base class for autofunction mode implementations.

    Each subclass encodes a specific strategy for sweeping a beam parameter
    and submitting sharpness evaluation jobs. Concrete modes are registered
    with `AUTOFUNCTION_MODES` and retrieved by name at runtime.
    """

    @abstractmethod
    def execute(
        self,
        ctx: AutofunctionContext,
        jobs: JobsManager,
        imaging: Imaging | None,
        resume_from: int = 0,
    ) -> Generator[SweepStep]:
        """
        Drive the autofunction sweep and submit sharpness evaluation jobs.

        Single-shot modes run the whole sweep without yielding. Multi-slice
        modes yield once per slice: each yielded step is a trial whose value
        the linked imaging must use for its next acquisition, and execution
        resumes on the following slice.

        Args:
            ctx: Shared execution environment providing access to the
                microscope, sweeping controller, criterion, and logger.
            jobs: Job manager to which sharpness evaluation callables are
                submitted for asynchronous execution.
            imaging: Instance of the Imaging class used to acquire images.
            resume_from: Global index of the sweep to resume from (default 0).
                Only used by the StepMode.

        Yields:
            The trial step for the next acquisition (multi-slice modes only).
        """


@AUTOFUNCTION_MODES.register("basic")
class BasicMode(AutofunctionMode):
    """
    Autofunction mode that acquires one image per swept attribute value.

    For each value in the sweep range, the target beam attribute is set,
    a full frame is acquired, and its sharpness is evaluated. Once all
    frames have been collected and their sharpness scores computed, the
    best attribute value can be determined from the results.

    The stage is temporarily displaced to a nearby focusing area for the
    duration of the sweep and restored to its original position afterward.
    """

    def execute(
        self,
        ctx: AutofunctionContext,
        jobs: JobsManager,
        imaging: Imaging | None,
        resume_from: int = 0,
    ) -> Generator[SweepStep]:
        _ = imaging, resume_from

        if (sweeping := ctx.sweeping) is None:
            raise AutofunctionError(
                "Sweeping for basic mode autofunction is not defined."
            )

        mode = ctx.settings.mode
        assert isinstance(mode, BasicModeSettings)

        with ctx.microscope.add_temporary_beam_shift(BeamShift(x=mode.delta_x, y=0.0)):
            for sweep in sweeping.sweep():
                ctx.ctx.text_logger.info(
                    f"Autofunction step {sweep.index + 1} "
                    f"(repetition {sweep.repetition + 1}): {sweeping.describe(sweep.value)}"
                )
                sweeping.set_attribute_value(sweep.value)
                image = ctx.microscope.beam.grab_frame()
                jobs.submit(ctx.make_sharpness_job(image, sweep))

        yield from ()


@AUTOFUNCTION_MODES.register("line")
class LineMode(AutofunctionMode):
    def execute(
        self,
        ctx: AutofunctionContext,
        jobs: JobsManager,
        imaging: Imaging | None,
        resume_from: int = 0,
    ) -> Generator[SweepStep]:
        _ = imaging, resume_from

        if (sweeping := ctx.sweeping) is None:
            raise AutofunctionError(
                "Sweeping for line mode autofunction is not defined."
            )

        mode = ctx.settings.mode
        assert isinstance(mode, LineModeSettings)

        with ctx.microscope.add_temporary_beam_shift(BeamShift(x=mode.delta_x, y=0.0)):
            line_time = self._estimate_line_time(ctx)  # in ns

            # generate sweep steps once so both acquisition and processing see the same steps
            sweep_steps = list(sweeping.sweep())

            self._variable_sweeping_during_scan(ctx, line_time, sweep_steps)

            # outside a reduced scanning area the frame holds stale content
            line_focus_image = ctx.microscope.beam.get_image(crop_to_scanning_area=True)
            ctx.line_focus_image = line_focus_image
            self._process_image(ctx, jobs, line_focus_image, sweep_steps)

        yield from ()

    def _estimate_line_time(self, ctx: AutofunctionContext) -> float:
        """
        Estimate the time required to scan a single line.

        Args:
            ctx: Shared execution environment providing access to the microscope
                and beam parameters.

        Returns:
            Estimated line scan time in nanoseconds with line time correction factor applied.
        """
        mode = ctx.settings.mode
        assert isinstance(mode, LineModeSettings)
        dwell_time = ctx.microscope.beam.dwell_time
        line_integration = ctx.microscope.beam.line_integration
        resolution = ctx.microscope.beam.resolution
        scanning_area = ctx.microscope.beam.scanning_area
        correction_factor = mode.line_time_correction_factor

        estimated_line_time = (
            # time spent per pixel
            dwell_time
            # length of the full row in pixels
            * resolution.width
            # portion of the line actually scanned
            * scanning_area.width
            # number of times each row is scanned
            * line_integration
            # user-defined factor accounting for microscope delays
            * correction_factor
        )

        ctx.ctx.text_logger.debug(f"Estimated line time: {estimated_line_time} ns")

        return estimated_line_time

    def _variable_sweeping_during_scan(
        self,
        ctx: AutofunctionContext,
        line_time: float,
        sweep_steps: list[SweepStep],
    ) -> None:
        """
        Sweep the target beam attribute while an image is being acquired line by line.

        Starts a continuous acquisition and iterates over sweep cycles (repetitions).
        Before each cycle, the beam is blanked for `lines_per_sweep` lines to create
        a dark separator band. The beam is then unblanked and the target attribute is
        set to each sweep step value in turn, holding for `lines_per_sweep` lines at
        each value.

        The resulting image has the following structure:

        .. code-block:: text

            [dark separator]
            [lines at step 0]   <- first repetition
            [lines at step 1]
            ...
            [lines at step N]
            [dark separator]
            [lines at step 0]   <- second repetition
            ...

        Each group of lines between two dark separators forms one stripe, corresponding
        to one sweep cycle. Within a stripe, each contiguous block of `lines_per_sweep`
        rows corresponds to one sweep step value, and is later matched to its sweep step
        by index in `_process_image`.

        At the start of the first cycle, an optional `pre_imaging_delay` is applied
        while the beam is blanked, to allow the system to stabilise before scanning begins.

        Args:
            ctx: Shared execution environment providing access to the microscope,
                sweeping controller, and logger.
            line_time: Estimated time to scan a single line in nanoseconds, used to
                compute the hold duration per sweep step.
            sweep_steps: Pre-generated list of sweep steps, shared with
                `_process_image` to ensure consistency.
        """
        mode = ctx.settings.mode
        assert isinstance(mode, LineModeSettings)
        assert ctx.sweeping is not None

        pre_delay = mode.pre_imaging_delay  # in s
        hold = mode.lines_per_sweep * line_time * 1e-9  # in s
        ctx.ctx.text_logger.debug(f"Hold time per sweep value: {hold} s")

        ctx.microscope.beam.start_acquisition()
        try:
            # group consecutive sweep steps by repetition index so that each
            # cycle produces one stripe in the acquired image
            for repetition, steps in groupby(sweep_steps, key=lambda x: x.repetition):
                ctx.ctx.text_logger.info(f"Line sweep cycle {repetition + 1}.")

                # blank to create a dark separator band between the stripes
                with ctx.microscope.beam.total_blanked():
                    if repetition == 0 and pre_delay > 0:
                        time.sleep(pre_delay)
                    time.sleep(hold)

                # acquire part of the stripe with each sweep value
                for sweep in steps:
                    ctx.sweeping.set_attribute_value(sweep.value)
                    time.sleep(hold)

            # create a final dark separator band
            with ctx.microscope.beam.total_blanked():
                time.sleep(hold)

        finally:
            ctx.microscope.beam.stop_acquisition()

    def _process_image(
        self,
        ctx: AutofunctionContext,
        jobs: JobsManager,
        image: Image,
        sweep_steps: list[SweepStep],
    ) -> None:
        """
        Extract per-line sharpness jobs from the acquired image.

        Identifies horizontal stripes in the image using dark separator rows
        produced by blanking during acquisition. Each stripe corresponds to one
        sweep cycle (repetition). Within each stripe, the rows are split into
        equal sub-groups - one per sweep step in that repetition. Each line
        within a sub-group is evaluated independently, producing multiple
        sharpness measurements per sweep step value.

        Stripes listed in `forbidden_stripe_indices` are skipped entirely.

        Args:
            ctx: Shared execution environment providing access to settings
                and the logger.
            jobs: Job manager to which sharpness evaluation callables are
                submitted for asynchronous execution.
            image: The full image acquired during the sweep, containing
                horizontal stripes separated by dark bands.
            sweep_steps: Pre-generated list of sweep steps, shared with
                `_variable_sweeping_during_scan` to ensure consistency.
        """
        mode = ctx.settings.mode
        assert isinstance(mode, LineModeSettings)

        forbidden_stripes = mode.forbidden_stripe_indices
        separator_threshold = mode.stripe_separator_threshold
        min_stripe_width = mode.minimal_stripe_width

        # convert the image to 8-bit for stripe detection
        img_8bit = image.to_8bit()

        # identify stripes from the image;
        # each stripe corresponds to one sweeping repetition
        stripes = [
            stripe
            for stripe in get_stripes(img_8bit, separator_threshold, min_stripe_width)
            if len(stripe) > 0
        ]

        cycles = len({step.repetition for step in sweep_steps})
        if len(stripes) != cycles:
            raise AutofunctionError(
                f"Expected {cycles} stripe(s) in the line-focus frame, found "
                f"{len(stripes)}. Check the separator threshold, the minimal "
                "stripe width and whether the sweep fits into the scanned rows."
            )

        # group sweep steps by repetition to match against stripes
        sweep_groups = groupby(sweep_steps, key=lambda step: step.repetition)

        for stripe, (rep, steps_iter) in zip(stripes, sweep_groups):
            if rep in forbidden_stripes:
                continue

            steps = list(steps_iter)

            # split lines of one stripe into equal sub-groups
            # each group then corresponds to one value of the sweep variable
            sub_bins = np.array_split(stripe, len(steps))

            # submit a sharpness job for each line within each sub-bin
            # associating all lines in the sub-bin with their corresponding sweep step
            for step, sub_bin in zip(steps, sub_bins):
                for line_index in sub_bin:
                    image_line = image[line_index, :]

                    # create a new SweepStep containing row index
                    step_with_line_index = SweepStep(
                        repetition=step.repetition,
                        value=step.value,
                        index=step.index,
                        line_index=int(line_index),
                    )
                    jobs.submit(
                        ctx.make_sharpness_job(image_line, step_with_line_index)
                    )


@AUTOFUNCTION_MODES.register("step")
class StepMode(AutofunctionMode):
    """
    Autofunction mode that spreads a parameter sweep across consecutive slices.

    Instead of acquiring its own images, this mode uses the main acquisition
    loop: each slice's production image serves as one trial in the sweep.
    This avoids extra exposure and throughput overhead, at the cost of
    spreading a single autofunction run over many slices.

    On each tick (one per slice) the mode advances the sweep by one step,
    setting the attribute value that the upcoming acquisition will use.
    The image acquired at that value is scored on the following tick, once
    it becomes available from the main loop. After the final step, one
    trailing tick is needed to score the last acquired image before the
    best value can be selected.
    """

    def execute(
        self,
        ctx: AutofunctionContext,
        jobs: JobsManager,
        imaging: Imaging | None,
        resume_from: int = 0,
    ) -> Generator[SweepStep]:
        if not imaging:
            raise AutofunctionError(
                "Linking imaging to autofunction is required for step mode autofunction."
            )

        if (sweeping := ctx.sweeping) is None:
            raise AutofunctionError(
                "Sweeping for step mode autofunction is not defined."
            )
        previous_step: SweepStep | None = None

        for sweep in sweeping.sweep():
            # skip steps before the resume point
            if sweep.index < resume_from:
                ctx.ctx.text_logger.debug(
                    f"Skipping step {sweep.index + 1} (repetition {sweep.repetition + 1}) before resume point."
                )
                previous_step = sweep
                continue

            ctx.ctx.text_logger.info(
                f"Autofunction step {sweep.index + 1} "
                f"(repetition {sweep.repetition + 1}): {sweeping.describe(sweep.value)}"
            )

            # score the frame from the previous slice, taken at the previous step's value
            # this is skipped on the first tick
            if previous_step is not None:
                self._score_last_frame(ctx, jobs, imaging, previous_step)

            previous_step = sweep

            # the upcoming acquisition must use this step's value; Autofunction
            # hands it to the linked imaging as an explicit patch
            yield sweep

        # trailing tick: score the frame acquired at the final step's value
        assert previous_step is not None
        self._score_last_frame(ctx, jobs, imaging, previous_step)

    @staticmethod
    def _score_last_frame(
        ctx: AutofunctionContext, jobs: JobsManager, imaging: Imaging, step: SweepStep
    ) -> None:
        """
        Submit the sharpness of the imaging's last frame, acquired at `step`'s value.

        With the imaging criterion, the sharpness the imaging calculated for
        that frame is used as is; otherwise the frame is scored with the
        autofunction's own criterion.

        Args:
            ctx: Shared execution environment.
            jobs: Job manager collecting the sweep's results.
            imaging: The linked imaging that acquired the frame.
            step: The sweep step the frame was acquired at.
        """
        if ctx.uses_imaging_criterion:
            if (sharpness := imaging.wait_for_sharpness()) is None:
                raise AutofunctionError(
                    f"Sharpness of the last frame of '{imaging.name}' is not "
                    f"available for sweep step {step.index + 1}."
                )

            ctx.ctx.text_logger.info(
                f"Sharpness from {imaging.name} for sweep {step}: {sharpness}"
            )
            jobs.submit(lambda: AutofunctionResult(sharpness, step))
            return

        if (image := imaging.last_acquired_image) is None:
            raise AutofunctionError(
                f"Last acquired image of '{imaging.name}' is not available for "
                f"sweep step {step.index + 1}."
            )

        jobs.submit(ctx.make_sharpness_job(image, step))


def _given(**kwargs: Any) -> dict[str, Any]:
    """
    Drop the arguments that are None, so Autoscript applies its own defaults.

    Autoscript's settings structures default every parameter to None but
    annotate them as non-optional, so an explicit None fails type checking.
    """
    return {name: value for name, value in kwargs.items() if value is not None}


@AUTOFUNCTION_MODES.register("autoscript")
class AutoscriptMode(AutofunctionMode):
    """
    Autofunction mode that delegates to the Autoscript manufacturer routines.

    Executes the appropriate built-in autofunction (autofocus, autostigmator,
    lens alignment, or source tilt correction) based on the configured sweep
    attribute. The stage is temporarily displaced to a nearby focusing area
    for the duration of the operation.

    Raises:
        AutofunctionError: If the microscope is not Autoscript-controlled, or
            if the sweep attribute is not supported.
    """

    def execute(
        self,
        ctx: AutofunctionContext,
        jobs: JobsManager,
        imaging: Imaging | None,
        resume_from: int = 0,
    ) -> Generator[SweepStep]:
        _ = jobs, imaging, resume_from

        try:
            from fibsem_maestro.microscope.autoscript_control.microscope_control import (
                AutoscriptMicroscopeControl,
            )
        except ImportError as e:
            raise AutofunctionError(
                "Autoscript auto-functions need Autoscript, which is not available."
            ) from e

        if not isinstance(ctx.microscope.control, AutoscriptMicroscopeControl):
            raise AutofunctionError(
                "Microscope must be an Autoscript-controlled microscope."
            )

        autoscript_microscope: SdbMicroscopeClient = (
            ctx.microscope.control.autoscript_microscope
        )

        mode = ctx.settings.mode
        assert isinstance(mode, AutoscriptFunctionBase)

        with ctx.microscope.add_temporary_beam_shift(BeamShift(x=mode.delta_x, y=0.0)):
            match ctx.settings.mode:
                case AutoscriptAutoFocus() as mode:
                    self._run_autofunction(ctx, mode, autoscript_microscope)
                case AutoscriptAutoStigmator() as mode:
                    self._run_autostigmator(ctx, mode, autoscript_microscope)
                case AutoscriptAutoLensAlignment() as mode:
                    self._run_auto_lens_alignment(ctx, mode, autoscript_microscope)
                case AutoscriptAutoSourceTilt() as mode:
                    self._run_auto_source_tilt(ctx, mode, autoscript_microscope)
                case _:
                    raise AutofunctionError(
                        f"Unsupported mode '{ctx.settings.mode}' for AutoscriptMode."
                    )

        yield from ()

    def _run_autofunction(
        self,
        ctx: AutofunctionContext,
        mode: AutoscriptAutoFocus,
        autoscript_microscope: SdbMicroscopeClient,
    ) -> None:
        """
        Run the manufacturer autofunction routine.

        Args:
            ctx: Shared execution environment.
            mode: The Autoscript auto focus mode settings.
            autoscript_microscope: The Autoscript microscope client instance.
        """
        from autoscript_sdb_microscope_client.structures import RunAutoFocusSettings

        beam = ctx.microscope.beam
        if (scanning_area := beam.scanning_area).is_full_frame():
            reduced_area = None
        else:
            reduced_area = scanning_area.to_autoscript()

        match mode.method:
            case AutoscriptAutoFocusMethod.STANDARD:
                settings = RunAutoFocusSettings(**_given(reduced_area=reduced_area))

                autoscript_microscope.auto_functions.run_auto_focus(settings)

            case AutoscriptAutoFocusMethod.VOLUMESCOPE:
                wd_step_as = (
                    mode.working_distance_step * 1e-9
                    if mode.working_distance_step is not None
                    else None
                )

                settings = RunAutoFocusSettings(
                    method="Volumescope",
                    dwell_time=beam.dwell_time * 1e-9,
                    horizontal_field_width=beam.horizontal_field_width * 1e-9,
                    line_integration=beam.line_integration,
                    resolution=str(beam.resolution),
                    **_given(
                        reduced_area=reduced_area,
                        maximum_iterations=mode.maximum_iterations,
                        working_distance_step=wd_step_as,
                        number_of_frames=mode.number_of_frames,
                    ),
                )

                autoscript_microscope.auto_functions.run_auto_focus(settings)

    def _run_autostigmator(
        self,
        ctx: AutofunctionContext,
        mode: AutoscriptAutoStigmator,
        autoscript_microscope: SdbMicroscopeClient,
    ) -> None:
        """
        Run the manufacturer autostigmator routine.

        Args:
            ctx: Shared execution environment.
            mode: The Autoscript auto stigmator mode settings.
            autoscript_microscope: The Autoscript microscope client instance.
        """
        from autoscript_sdb_microscope_client.structures import RunAutoStigmatorSettings

        beam = ctx.microscope.beam
        if (scanning_area := beam.scanning_area).is_full_frame():
            reduced_area = None
        else:
            reduced_area = scanning_area.to_autoscript()

        match mode.method:
            case AutoscriptAutoStigmatorMethod.STANDARD:
                settings = RunAutoStigmatorSettings(**_given(reduced_area=reduced_area))
            case AutoscriptAutoStigmatorMethod.ONGETAL:
                settings = RunAutoStigmatorSettings(
                    method="OngEtAl",
                    dwell_time=beam.dwell_time * 1e-9,
                    resolution=str(beam.resolution),
                    horizontal_field_width=beam.horizontal_field_width * 1e-9,
                    line_integration=beam.line_integration,
                    **_given(
                        reduced_area=reduced_area,
                        maximum_iterations=mode.maximum_iterations,
                        stigmation_step=mode.stigmation_step,
                        number_of_frames=mode.number_of_frames,
                    ),
                )

        autoscript_microscope.auto_functions.run_auto_stigmator(settings)

    def _run_auto_lens_alignment(
        self,
        ctx: AutofunctionContext,
        mode: AutoscriptAutoLensAlignment,
        autoscript_microscope: SdbMicroscopeClient,
    ) -> None:
        """
        Run the manufacturer lens alignment routine.

        Args:
            ctx: Shared execution environment.
            mode: The Autoscript auto lens alignment mode settings.
            autoscript_microscope: The Autoscript microscope client instance.
        """
        from autoscript_sdb_microscope_client.structures import (
            RunAutoLensAlignmentSettings,
        )

        beam = ctx.microscope.beam
        if (scanning_area := beam.scanning_area).is_full_frame():
            reduced_area = None
        else:
            reduced_area = scanning_area.to_autoscript()

        match mode.modulation_type:
            case AutoscriptAutoLensAlignmentModulationType.AUTOMATIC:
                modulation_type_as = "Automatic"
            case AutoscriptAutoLensAlignmentModulationType.HIGH_VOLTAGE:
                modulation_type_as = "HighVoltage"
            case AutoscriptAutoLensAlignmentModulationType.WORKING_DISTANCE:
                modulation_type_as = "WorkingDistance"

        settings = RunAutoLensAlignmentSettings(
            modulation_type=modulation_type_as,
            dwell_time=beam.dwell_time * 1e-9,
            resolution=str(beam.resolution),
            line_integration=beam.line_integration,
            **_given(
                reduced_area=reduced_area,
                number_of_frames=mode.number_of_frames,
            ),
        )

        autoscript_microscope.auto_functions.run_auto_lens_alignment(settings)

    def _run_auto_source_tilt(
        self,
        ctx: AutofunctionContext,
        mode: AutoscriptAutoSourceTilt,
        autoscript_microscope: SdbMicroscopeClient,
    ) -> None:
        """
        Run the manufacturer source tilt correction routine.

        Temporarily switches the detector to TLD in secondary electron mode,
        which is required by the Volumescope source tilt method. The original
        detector settings are restored afterward, even if the routine raises.

        Args:
            ctx: Shared execution environment.
            mode: The Autoscript auto source tilt mode settings.
            autoscript_microscope: The Autoscript microscope client instance.
        """
        from autoscript_sdb_microscope_client.structures import (
            RunAutoSourceTiltSettings,
        )

        _ = mode

        beam = ctx.microscope.beam
        settings = RunAutoSourceTiltSettings(
            method="Volumescope",
            contrast=beam.detector_contrast,
            brightness=beam.detector_brightness,
            dwell_time=beam.dwell_time * 1e-9,
            resolution=str(beam.resolution),
        )

        autoscript_microscope.auto_functions.run_auto_source_tilt(settings)

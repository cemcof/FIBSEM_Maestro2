# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Runs autofunction sweeps for the actions that sweep beam attributes.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fibsem_maestro.autofunction import AUTOFUNCTION_MODES, LineMode
from fibsem_maestro.autofunction.af_chart import af_chart
from fibsem_maestro.autofunction.autofunction_context import AutofunctionContext
from fibsem_maestro.autofunction.jobs_manager import JobsManager
from fibsem_maestro.autofunction.sweep_space import sweep_attributes
from fibsem_maestro.autofunction.sweeping import Sweeping
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.point import PixelPoint
from fibsem_maestro.logging.image.overlay import PolylineOverlay
from fibsem_maestro.settings.autofunction_settings import AutoscriptFunctionBase
from fibsem_maestro.settings.autofunction_settings import BasicMode as BasicModeSettings
from fibsem_maestro.settings.autofunction_settings import LineMode as LineModeSettings
from fibsem_maestro.settings.autofunction_settings import StepMode as StepModeSettings

if TYPE_CHECKING:
    from collections.abc import Generator

    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.autofunction.autofunction_context import SweepSettingsLike
    from fibsem_maestro.autofunction.result import AutofunctionResult
    from fibsem_maestro.autofunction.sweep_step import (
        SweepDirection,
        SweepPoint,
        SweepStep,
    )
    from fibsem_maestro.imaging.imaging import Imaging
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.settings.autofunction_settings import AutofunctionMode
    from fibsem_maestro.settings.dynamic_default import DefaultContext

# another sweep follows only if the best value moved within this angle of the
# previous move's direction (2D); in 1D the directions are +-1, so it only
# matters whether the edge is the same
_MAX_DIRECTION_CHANGE_DEG = 45.0


@dataclass(frozen=True)
class SweepOutcome:
    """
    What a finished sweep concluded.

    Attributes:
        best: The best value, now applied to the microscope; None in the
            Autoscript modes, which do not sweep.
        sweep_again: True if the run continues with another sweep around it.
    """

    best: SweepPoint | None
    sweep_again: bool


class SweepRunner:
    """
    Runs the sweeps of autofunction runs and decides how a run continues.

    A run is one or more sweeps: after each, the best value is applied, and
    another sweep around it follows while the best keeps reaching the edge of
    the swept region in a consistent direction, up to `max_sweeps`.

    The runner keeps the run's bookkeeping (base value of the current sweep,
    sweeps done, last edge direction), so the owning action can store and
    restore it, but it never decides when a run starts.

    Args:
        microscope: The microscope instance.
        settings: The settings of the owning action; read on every `rebuild`.
        ctx: The owning action's context, for logging.
        beam_type: The beam whose attributes are swept.
        defaults: What the settings following their dynamic defaults
            are computed from: the owning action's `default_context()`.
    """

    def __init__(
        self,
        microscope: Microscope,
        settings: SweepSettingsLike,
        ctx: ActionContext,
        beam_type: BeamType,
        defaults: DefaultContext,
    ) -> None:
        self._microscope = microscope
        self._settings = settings
        self._ctx = ctx
        self._defaults = defaults

        # the pool is sized from max_workers; reset_jobs resizes it between sweeps
        self._executor_workers = settings.max_workers
        self._executor = ThreadPoolExecutor(self._executor_workers)
        self._jobs = JobsManager(executor=self._executor)

        # base value of the sweep in progress
        self._sweep_base_value: SweepPoint | None = None
        # sweeps completed in the current run
        self._sweeps_done = 0
        # direction of the edge the best value last reached
        self._last_direction: SweepDirection | None = None

        self.rebuild(beam_type)

    def rebuild(self, beam_type: BeamType) -> None:
        """
        Rebuild mode, sweeping, and autofunction context from the current settings.

        Args:
            beam_type: The beam whose attributes are swept.
        """
        self._mode = AUTOFUNCTION_MODES.get(self._settings.mode.type)()

        if isinstance(self._settings.mode, AutoscriptFunctionBase):
            # sweeping is not used in the Autoscript mode
            self._sweeping = None
        else:
            self._sweeping = Sweeping(
                self._microscope.electron_beam
                if beam_type is BeamType.ELECTRON
                else self._microscope.ion_beam,
                self._settings.mode.sweeping,
                self._ctx.text_logger.derive("sweeping"),
                self._defaults,
            )

        self._autofunction_ctx = AutofunctionContext(
            self._microscope,
            self._sweeping,
            self._settings,
            self._ctx,
            self._defaults,
        )

    @property
    def attributes(self) -> tuple[str, ...]:
        """Names of the swept beam attributes; empty in the Autoscript modes."""
        return swept_attributes(self._settings.mode)

    @property
    def sweep_base_value(self) -> SweepPoint | None:
        """Base value of the sweep in progress, or None between sweeps."""
        return self._sweep_base_value

    @property
    def sweeps_done(self) -> int:
        """Sweeps completed in the current run."""
        return self._sweeps_done

    @property
    def last_direction(self) -> SweepDirection | None:
        """Direction of the edge the best value of the run's last sweep reached."""
        return self._last_direction

    @property
    def uses_imaging_criterion(self) -> bool:
        """True if step mode reuses the sharpness its linked imaging calculates."""
        return self._autofunction_ctx.uses_imaging_criterion

    def collect_completed(self) -> list[AutofunctionResult]:
        """
        Return the results of the jobs of the current sweep that have finished.

        Returns:
            The finished results, without waiting for the others.
        """
        return self._jobs.collect_completed()

    def wait(self) -> None:
        """Block until every submitted sharpness job has finished."""
        self._jobs.wait()

    def restore(
        self,
        sweep_base_value: SweepPoint | None,
        sweeps_done: int,
        last_direction: SweepDirection | None,
        results: list[AutofunctionResult],
    ) -> None:
        """
        Restore the bookkeeping of an interrupted run.

        Args:
            sweep_base_value: Base value of the interrupted sweep, if any.
            sweeps_done: Sweeps the run had completed.
            last_direction: Direction of the edge the last sweep reached, if any.
            results: Results the interrupted sweep had collected; re-submitted,
                so that the sweep's evaluation sees them all.
        """
        self._sweep_base_value = sweep_base_value
        self._sweeps_done = sweeps_done
        self._last_direction = last_direction
        for result in results:
            self._jobs.submit(lambda r=result: r)

    def begin_sweep(self) -> None:
        """
        Prepare a new sweep around the values currently on the microscope.

        Discards the previous sweep's jobs and resizes the pool if
        `max_workers` changed.
        """
        self.reset_jobs()
        self._sweep_base_value = (
            self._sweeping.get_attribute_value() if self._sweeping is not None else None
        )

    def execute(
        self, imaging: Imaging | None, resume_from: int = 0
    ) -> Generator[SweepStep]:
        """
        Start the configured mode on the current sweep.

        Args:
            imaging: The linked imaging; required by step mode only.
            resume_from: Global step index to resume an interrupted step-mode sweep from.

        Returns:
            The mode's generator: single-shot modes finish without yielding,
            step mode yields one trial step per slice.
        """
        return self._mode.execute(
            self._autofunction_ctx, self._jobs, imaging, resume_from=resume_from
        )

    def finish_sweep(self, line_focus_name: str) -> SweepOutcome:
        """
        Evaluate the finished sweep, apply its best value, and decide whether to sweep again.

        Waits for the sweep's sharpness jobs. If another sweep follows, its
        direction is remembered for the next decision. If the evaluation
        fails, the run ends.

        Args:
            line_focus_name: Prefix of the line focus image's file name (line mode).

        Returns:
            The best value and whether another sweep follows.
        """
        try:
            results = self._jobs.wait_and_collect()
            best = self._apply_best_and_log(results, line_focus_name)
            direction = self._edge_direction(results, best)
        except Exception:
            self.end_run()
            raise
        finally:
            self._sweep_base_value = None

        self._sweeps_done += 1
        if not self._should_sweep_again(direction):
            return SweepOutcome(best, sweep_again=False)

        self._last_direction = direction
        self._ctx.text_logger.info(
            "Best value lies at the edge of the swept region. Starting sweep "
            f"{self._sweeps_done + 1} of at most {self._max_sweeps()}."
        )
        return SweepOutcome(best, sweep_again=True)

    def run_to_completion(
        self, line_focus_name: str, imaging: Imaging | None = None
    ) -> SweepPoint | None:
        """
        Run a whole run of a single-shot mode (basic, line, Autoscript) and apply its result.

        Args:
            line_focus_name: Prefix of the line focus image's file name (line mode).
            imaging: The linked imaging, if the mode needs one.

        Returns:
            The best value of the run's last sweep, now applied to the
            microscope; None in the Autoscript modes.
        """
        try:
            while True:
                self.begin_sweep()
                for _ in self.execute(imaging):
                    self._jobs.wait()

                outcome = self.finish_sweep(line_focus_name)
                if not outcome.sweep_again:
                    return outcome.best
        finally:
            self.end_run()

    def abort_sweep(self) -> None:
        """Forget the sweep in progress and end the run."""
        self._sweep_base_value = None
        self.end_run()

    def end_run(self) -> None:
        """Reset the sweep count and edge, so that the next run starts afresh."""
        self._sweeps_done = 0
        self._last_direction = None

    def reset_jobs(self) -> None:
        """
        Discard the previous sweep's jobs and resize the pool if `max_workers` changed.

        Called only between sweeps, so a sweep in progress keeps the pool its jobs run on.
        """
        self._jobs.wait_and_clear()
        if self._executor_workers == self._settings.max_workers:
            return

        # every job was waited for, so the old pool is idle
        self._executor.shutdown(wait=False)
        self._executor_workers = self._settings.max_workers
        self._executor = ThreadPoolExecutor(self._executor_workers)
        self._jobs = JobsManager(executor=self._executor)

    def _edge_direction(
        self, results: list[AutofunctionResult], best: SweepPoint | None
    ) -> SweepDirection | None:
        """
        Return the direction of the edge of the swept region the best value lies at.

        Args:
            results: The finished sweep's results.
            best: The best value applied, or None if nothing was swept (Autoscript).

        Returns:
            A unit vector pointing past that edge, or None if the best value
            lies inside the swept region or equals the base value: nothing
            moved, so sweeping again around it would repeat the same sweep.
        """
        if best is None or self._sweeping is None or self._sweep_base_value is None:
            return None

        return self._sweeping.space.edge_direction(
            best, self._sweep_base_value, [r.sweep.value for r in results]
        )

    def _should_sweep_again(self, direction: SweepDirection | None) -> bool:
        """
        Decide whether the run continues with another sweep.

        Another sweep follows while the best value keeps reaching the edge of
        the swept region in a consistent direction, i.e. while the optimum
        probably lies beyond it, and `max_sweeps` has not been reached. In 1D
        that is the same edge again; in 2D a direction within
        `_MAX_DIRECTION_CHANGE_DEG` of the last one.

        Args:
            direction: Direction of the edge the finished sweep's best value
                lies at, if any.

        Returns:
            True if another sweep should start.
        """
        if direction is None or self._sweeps_done >= self._max_sweeps():
            return False

        if self._last_direction is None:
            return True

        cosine = float(np.dot(direction, self._last_direction))
        return cosine >= math.cos(math.radians(_MAX_DIRECTION_CHANGE_DEG)) - 1e-9

    def _max_sweeps(self) -> int:
        """
        Return the most sweeps one run may perform.

        Returns:
            The mode's `max_sweeps`; 1 for the Autoscript modes, which do not sweep.
        """
        match self._settings.mode:
            case BasicModeSettings() | LineModeSettings() | StepModeSettings() as mode:
                return mode.max_sweeps
            case _:
                return 1

    def _apply_best_and_log(
        self, results: list[AutofunctionResult], line_focus_name: str
    ) -> SweepPoint | None:
        """
        Apply the best sweep value to the microscope and log the autofunction chart.

        Does nothing if no sweeping is configured (Autoscript mode).

        Args:
            results: Results collected during the sweep.
            line_focus_name: Prefix of the line focus image's file name (line mode).

        Returns:
            The best value applied, or None in Autoscript mode.
        """
        if self._sweeping is None:
            return None

        evaluation = self._sweeping.evaluate(results)
        best = evaluation.best
        self._ctx.text_logger.info(
            f"Best sweep value: {self._sweeping.describe(best)}."
        )
        self._sweeping.set_attribute_value(best)

        self._ctx.image_logger.save_plot(
            "af_curve.png",
            af_chart(evaluation, self._sweeping.space, self._sweep_base_value),
        )
        if isinstance(self._mode, LineMode):
            self._log_line_focus_image(results, line_focus_name)

        return best

    def _log_line_focus_image(
        self, results: list[AutofunctionResult], line_focus_name: str
    ) -> None:
        """
        Log the line focus image with per-line sharpness overlaid as a polyline.

        Args:
            results: Autofunction results collected during the sweep, one per line.
            line_focus_name: Prefix of the image's file name.
        """
        if not results:
            return

        # the frame the sweep was evaluated on; the line indices refer to it
        if (image := self._autofunction_ctx.line_focus_image) is None:
            self._ctx.text_logger.warning(
                "Line focus image is not available for logging."
            )
            return

        # the sharpness evaluation jobs can be completed in any order
        sorted_results = sorted(results, key=_get_line_index)
        sharpness_values = [r.sharpness for r in sorted_results]
        line_indices = [
            r.sweep.line_index for r in sorted_results if r.sweep.line_index is not None
        ]

        if max(sharpness_values) == 0:
            self._ctx.text_logger.warning(
                "Max sharpness value is 0. Unable to normalize the sharpness, skipping logging."
            )
            return

        # scale for normalizing sharpness to fit the image
        scale = image.shape[1] / max(sharpness_values)

        self._ctx.image_logger.save_image(
            filename=f"{line_focus_name}_line_focus.png",
            img=image,
            overlays=[
                PolylineOverlay(
                    points=[
                        PixelPoint(x=int(v * scale), y=line_idx)
                        for v, line_idx in zip(sharpness_values, line_indices)
                    ],
                    color="red",
                )
            ],
            title="Line focus plot",
        )


def swept_attributes(mode: AutofunctionMode) -> tuple[str, ...]:
    """
    Return the names of the beam attributes a mode sweeps.

    Args:
        mode: Settings of an autofunction mode.

    Returns:
        One name per swept attribute; empty for the Autoscript modes, which do not sweep.
    """
    if isinstance(mode, AutoscriptFunctionBase):
        return ()
    return sweep_attributes(mode.sweeping)


def _get_line_index(result: AutofunctionResult) -> int:
    """Extract global index of the line in the image that this result corresponds to."""
    assert result.sweep.line_index is not None

    return result.sweep.line_index

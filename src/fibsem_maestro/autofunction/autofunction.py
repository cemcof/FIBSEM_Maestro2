# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fibsem_maestro.action.action import Action
from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.autofunction.error import AutofunctionError
from fibsem_maestro.autofunction.result import AutofunctionResult
from fibsem_maestro.autofunction.sweep_runner import SweepRunner
from fibsem_maestro.autofunction.sweep_step import (
    SweepDirection,
    SweepPoint,
)
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.logging.logging import with_logging_context
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.autofunction_settings import (
    AutofunctionSettings,
    AutoscriptFunctionBase,
)
from fibsem_maestro.settings.autofunction_settings import BasicMode as BasicModeSettings
from fibsem_maestro.settings.autofunction_settings import LineMode as LineModeSettings
from fibsem_maestro.settings.autofunction_settings import StepMode as StepModeSettings

if TYPE_CHECKING:
    from collections.abc import Generator

    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.autofunction.sweep_step import (
        SweepStep,
    )
    from fibsem_maestro.core.area import RelativeArea
    from fibsem_maestro.core.beam_type import BeamType
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.settings.reactive import ChangePath
    from fibsem_maestro.workflow.actions import Actions


class AutofunctionState(ActionState):
    sweep_base_value: SweepPoint | None = None
    sweep_in_progress: bool = False
    current_step_index: int = 0
    collected_results: list[AutofunctionResult] = field(default_factory=list)
    sweeps_done: int = 0
    last_direction: SweepDirection | None = None
    # slice on which a step-mode run was requested that is still waiting for
    # the linked imaging to be free; None if no request is waiting
    queued_since: int | None = None


def _with_point_value(result: Any) -> Any:
    """Wrap the scalar sweep value of a stored result into a one-element sweep point."""
    if (
        isinstance(result, dict)
        and isinstance(sweep := result.get("sweep"), dict)
        and isinstance(value := sweep.get("value"), int | float)
    ):
        return {**result, "sweep": {**sweep, "value": (float(value),)}}

    return result


@dataclass(frozen=True)
class _TrialTarget:
    """
    Where a step-mode sweep puts its trial values, kept for restoring the imaging.

    Captured when the sweep starts, since the settings may have changed by
    the time the sweep is discarded.

    Attributes:
        imaging: The linked imaging whose upcoming properties get the trial values.
        beam: The beam whose properties are patched.
        attributes: Names of the swept attributes, in sweep-point order.
    """

    imaging: Imaging
    beam: BeamType
    attributes: tuple[str, ...]


@ACTION_REGISTRY.register("autofunction")
class Autofunction(Action[AutofunctionSettings, AutofunctionState]):
    """
    Optimises beam attributes by sweeping them and scoring the frames' sharpness.

    An autofunction is the generic form of autofocus (working distance),
    autostigmation (stigmator) and their relatives: whatever the mode sweeps
    (`swept_attributes`), or the Autoscript routine it delegates to.

    Manages the full autofunction lifecycle: deciding when to execute based on
    slice number and image sharpness, setting up the appropriate mode, advancing
    the execution generator, collecting sharpness results, and writing the best sweep value
    back to the microscope and property store.

    For single-shot modes (basic, line, Autoscript) the sweep completes within
    one execution. For step mode, execution is resumed across successive
    slices, one sweep step per slice, until the sweep is exhausted.
    """

    _STATE_FIELDS = frozenset({"mode", "beam_type", "linked_imaging"})

    def __init__(
        self,
        name: str,
        microscope: Microscope,
        settings: AutofunctionSettings,
        ctx: ActionContext,
        actions: Actions,
    ):
        super().__init__(name, microscope, settings, ctx, actions)

        # mode, sweeping, sharpness jobs and the bookkeeping of a run
        self._runner = SweepRunner(
            microscope, settings, ctx, settings.beam_type, self.default_context()
        )

        # the step-mode sweep in progress, advanced once per slice
        self._active_gen: Generator[SweepStep] | None = None
        self._current_step_index = 0

        # where the step-mode sweep in progress puts its trial values; None
        # while no step-mode sweep is in progress
        self._trial_target: _TrialTarget | None = None

        # slice of the step-mode run request waiting for the linked imaging
        self._queued_since: int | None = None

        # rebuild when the settings the runner is built from change
        self._settings.on_change_at(self._rebuild_if_affected)

    def _rebuild_if_affected(self, path: ChangePath) -> None:
        """
        Rebuild mode, sweeping and context if a change touched their inputs.

        Args:
            path: Location of the change relative to the settings; an empty path
                means the settings changed as a whole.
        """
        if not path or path[0] in self._STATE_FIELDS:
            self._runner.rebuild(self._settings.beam_type)

    @classmethod
    def settings_cls(cls) -> type[AutofunctionSettings]:
        return AutofunctionSettings

    @classmethod
    def state_cls(cls) -> type[AutofunctionState]:
        return AutofunctionState

    @property
    def state(self) -> AutofunctionState:
        return AutofunctionState(
            sweep_base_value=self._runner.sweep_base_value,
            sweep_in_progress=self._active_gen is not None,
            current_step_index=self._current_step_index,
            collected_results=self._runner.collect_completed()
            if self._active_gen is not None
            else [],
            sweeps_done=self._runner.sweeps_done,
            last_direction=self._runner.last_direction,
            queued_since=self._queued_since,
        )

    @property
    def run_in_progress(self) -> bool:
        """
        True while an autofunction run is in progress.

        A run lasts from its first sweep to the end of its last sweep, including
        the sweeps that follow because the best value reached the edge. Between
        slices only a step-mode run can be in progress.
        """
        return self._active_gen is not None

    @property
    def is_step_mode(self) -> bool:
        """True if the autofunction is configured for step mode."""
        return isinstance(self._settings.mode, StepModeSettings)

    @property
    def queued_since(self) -> int | None:
        """Slice the waiting step-mode run was requested on; None if no run waits."""
        return self._queued_since

    @property
    def swept_attributes(self) -> tuple[str, ...]:
        """Names of the beam attributes the autofunction sweeps; empty for Autoscript."""
        return self._runner.attributes

    @property
    def scanning_area(self) -> RelativeArea | None:
        """
        The scanning area of the basic, line or Autoscript mode, if set.

        Step mode has none: its frames come from the linked imaging.
        """
        match self._settings.mode:
            case (
                BasicModeSettings() | LineModeSettings() | AutoscriptFunctionBase()
            ) as mode if mode.scanning_area:
                return mode.scanning_area[0]
            case _:
                return None

    def set_state(self, state: AutofunctionState) -> None:
        # close a running sweep explicitly
        if self._active_gen is not None:
            self._abort_sweep()

        self._current_step_index = state.current_step_index
        self._queued_since = state.queued_since
        self._runner.restore(
            state.sweep_base_value,
            state.sweeps_done,
            state.last_direction,
            # the collected results of a sweep in progress, so that its evaluation sees them all
            state.collected_results if state.sweep_in_progress else [],
        )

        if not state.sweep_in_progress:
            self._active_gen = None
            return

        self._ctx.text_logger.info(
            f"Restoring state of '{self.name}': sweep in progress, resuming from "
            f"global step index {state.current_step_index}."
        )
        imaging = self._resolve_imaging()
        self._active_gen = self._runner.execute(
            imaging, resume_from=state.current_step_index
        )
        self._trial_target = self._trial_target_for(imaging)

    def should_execute(self) -> bool:
        """
        Decide whether autofunction runs on the current slice.

        A run in progress always continues. Otherwise autofunction is triggered if
        the slice matches the execution frequency, or if the sharpness of the
        linked imaging's latest frame is below `settings.sharpness_limit`.

        A triggered step-mode autofunction may have to wait: per linked imaging,
        only one step-mode run may be in progress, since each one varies that
        imaging's frames. The request then waits in first-come order and the
        run starts as soon as the imaging is free; triggers that arrive while
        it waits are ignored.

        Returns:
            `True` if autofunction should run for this slice.
        """
        if self._active_gen is not None:
            return True

        if not self.is_step_mode:
            # a request left from before a switch away from step mode runs now
            triggered = self._queued_since is not None or self._is_triggered()
            self._queued_since = None
            return triggered

        if self._queued_since is None:
            if not self._is_triggered():
                return False
            self._queued_since = self._ctx.slice

        if (blocker := _blocking_step_autofunction(self, self._actions)) is not None:
            self._log_waiting(blocker)
            return False

        if self._queued_since < self._ctx.slice:
            self._ctx.text_logger.info(
                f"'{self.name}' starts the step-mode run requested on slice "
                f"{self._queued_since}; autofunction queue of '{self._settings.linked_imaging}' is free."
            )
        self._queued_since = None
        return True

    def _is_triggered(self) -> bool:
        """
        Check the triggers of a new autofunction run.

        Blocks until the sharpness of the linked imaging's latest frame is
        available, if a sharpness limit is set.

        Returns:
            True if the slice matches the execution frequency, or the latest
            frame's sharpness is below the sharpness limit.
        """
        if self._settings.execution_frequency.matches(self._ctx.slice):
            self._ctx.text_logger.info(
                f"'{self.name}' triggered: slice {self._ctx.slice} matches "
                f"execution frequency ({self._settings.execution_frequency})."
            )
            return True

        if limit := self._settings.sharpness_limit:
            # only wait for sharpness calculation from imaging if we actually need it
            sharpness = self._resolve_imaging().wait_for_sharpness()
            self._ctx.text_logger.debug(f"Last image sharpness: {sharpness}.")

            if sharpness is not None and sharpness < limit:
                self._ctx.text_logger.info(
                    f"'{self.name}' triggered: image sharpness ({sharpness:.4f}) "
                    f"is below the limit ({limit:.4f})."
                )
                return True

        return False

    def _log_waiting(self, blocker: Autofunction) -> None:
        """
        Log that the waiting step-mode run cannot start yet.

        Logged at info level on the slice the request was made, at debug level
        on the slices it keeps waiting.

        Args:
            blocker: The autofunction the run waits for.
        """
        reason = (
            "has a step-mode run in progress"
            if blocker.run_in_progress
            else f"has been waiting since slice {blocker.queued_since}"
        )
        message = (
            f"'{self.name}' waits: '{blocker.name}' {reason} on "
            f"'{self._settings.linked_imaging}'."
        )
        if self._queued_since == self._ctx.slice:
            self._ctx.text_logger.info(message)
        else:
            self._ctx.text_logger.debug(message)

    @with_logging_context
    def test(self) -> None:
        """
        Run the autofunction pipeline once and apply the best sweep value.

        Intended for manual testing and diagnostics outside the normal
        acquisition loop.

        Raises:
            AutofunctionError: If the configured mode is a step mode, which spreads
                execution across multiple slices and cannot be run in a single call.
        """
        if self.is_step_mode:
            raise AutofunctionError("Test is not supported for step mode")

        with self._scanning_area_applied():
            if self._ctx.props_store.exists("props.yaml"):
                self._ctx.text_logger.info("Loading saved microscope properties.")
                self.read_and_set_properties()
            else:
                self._ctx.text_logger.info("No saved microscope properties found.")

            self._runner.run_to_completion(self.name_with_underscores)

            # the updated value may not be displayed in the microscope GUI until we start scanning
            self._microscope.beam.grab_frame()

    def wait_for_background_threads(self) -> None:
        self._runner.wait()

    def _run_step(self) -> StepOutcome:
        if self._active_gen is None:
            self._start_sweep()
        trial = self._advance()

        props = self.collect_properties()
        if trial is None:
            return Produced(props)

        # mid-sweep: the linked imaging must acquire its next frame at the trial value
        patch = GlobalProperties()
        for name, value in zip(self._runner.attributes, trial.value, strict=True):
            patch.set_property(name, value, self.beam_type)
        return Produced(props, patches={self._settings.linked_imaging: patch})

    def _start_sweep(self) -> None:
        """
        Start a new sweep from the currently applied properties.

        Raises:
            AutofunctionError: If step mode is configured with an unsuitable linked imaging.
        """
        imaging = self._resolve_imaging()
        if self.is_step_mode:
            self._check_step_mode_imaging(imaging)

        # also discards the jobs and results of the previous sweep
        self._runner.begin_sweep()
        self._current_step_index = 0
        self._active_gen = self._runner.execute(imaging)
        self._trial_target = self._trial_target_for(imaging)

    def _check_step_mode_imaging(self, imaging: Imaging) -> None:
        """
        Verify that the linked imaging provides one fresh trial frame per slice.

        Args:
            imaging: The linked imaging action.

        Raises:
            AutofunctionError: If the imaging does not run every slice, or uses a
                different beam than the autofunction.
        """
        if not imaging.settings.execution_frequency.runs_every_slice:
            raise AutofunctionError(
                f"Step mode requires '{imaging.name}' to run every slice "
                f"(execution frequency 1), got {imaging.settings.execution_frequency}."
            )

        if imaging.beam_type is not self.beam_type:
            raise AutofunctionError(
                f"Step mode requires '{imaging.name}' to use the same beam as "
                f"'{self.name}' ({self.beam_type}), got {imaging.beam_type}."
            )

        if self._runner.uses_imaging_criterion and imaging.settings.criterion is None:
            raise AutofunctionError(
                f"Step mode uses the criterion of '{imaging.name}', but it has none configured."
            )

    def _resolve_imaging(self) -> Imaging:
        imaging = self._actions.named(self._settings.linked_imaging)
        if not isinstance(imaging, Imaging):
            raise AutofunctionError(
                f"Linked action is not an Imaging action: {imaging.name}"
            )
        return imaging

    def _advance(self) -> SweepStep | None:
        """
        Advance the active sweep by one step, and start the next sweep if the run continues.

        When a sweep finishes, its best value is applied. If that value lies at
        the edge of the swept range, another sweep centered on it follows:
        basic and line mode complete it right away, step mode returns its first trial value.

        Returns:
            The trial step the next acquisition must use, or `None` if the
            autofunction run finished and its best value was applied.
        """
        assert self._active_gen is not None
        self._current_step_index += 1

        try:
            trial = next(self._active_gen)
            # wait so that the stored state contains every result; otherwise
            # restoring after an interrupt has to deal with half-finished jobs
            self._runner.wait()
        except StopIteration:
            pass
        except Exception:
            self._abort_sweep()
            raise
        else:
            return trial

        try:
            outcome = self._runner.finish_sweep(self.name_with_underscores)
        finally:
            # clear even if evaluation fails, so the next slice starts afresh
            self._active_gen = None
            self._trial_target = None
            self._current_step_index = 0

        if outcome.sweep_again:
            try:
                self._start_sweep()
            except Exception:
                self._runner.end_run()
                raise
            # basic and line mode complete the new sweep here; step mode
            # returns its first trial value
            return self._advance()

        self._runner.end_run()
        if self._settings.grab_frame_after:
            self._save_result_frame("after_autofunction")

        return None

    def reset_state(self) -> None:
        """
        Discard the state; a discarded step-mode run gives the imaging back its value.

        The value the sweep started from replaces the trial value in the
        linked imaging's upcoming properties. Covers every reset that discards
        a valid run: a state-defining settings change and the Reset state
        button, but not a reset of the whole workflow (`reset`).
        """
        self._restore_imaging()
        super().reset_state()

    def reset(self) -> None:
        """Reset to slice 0 and the default state, leaving the imaging's properties alone."""
        # the workflow restarts: the imaging's props of slice 0 are not the run's
        self._trial_target = None
        super().reset()

    def properties_would_discard_state(self, of: Action) -> bool:
        """
        Tell whether new properties of an action would abort the step-mode run in progress.

        Args:
            of: The action whose properties would be replaced.

        Returns:
            True while a step-mode run is in progress and `of` is this
            autofunction or the imaging the run puts its trial values into.
        """
        if self._active_gen is None:
            return False
        target = self._trial_target
        return of is self or (target is not None and of is target.imaging)

    def _on_properties_replaced(self, of: Action) -> None:
        """
        Abort a step-mode run when its own properties or its imaging's are replaced.

        The run started from the old properties, and its results were scored
        on frames taken with them. If the autofunction's own properties were
        replaced, the imaging gets back the value the sweep started from in
        place of the trial value; new properties of the imaging are kept. A
        waiting run request is kept: the run will start from the new
        properties.

        Args:
            of: The action whose properties were replaced.
        """
        if not self.properties_would_discard_state(of):
            return

        self._ctx.text_logger.info(
            f"Aborting the step-mode run of '{self.name}': new properties of "
            f"'{of.name}' were stored."
        )
        if of is self:
            self._restore_imaging()
        self._abort_sweep()
        self._ctx.state_store.write("state.yaml", self.state)

    def _trial_target_for(self, imaging: Imaging) -> _TrialTarget | None:
        """Where a sweep on `imaging` puts its trial values; None unless in step mode."""
        if not self.is_step_mode:
            return None
        beam = self.beam_type
        assert beam is not None
        return _TrialTarget(imaging, beam, self._runner.attributes)

    def _restore_imaging(self) -> None:
        """
        Put the value the step-mode sweep started from into the imaging's upcoming properties.

        Replaces the trial value the sweep patched in. Does nothing if no
        step-mode sweep is in progress, or the imaging stores no properties.
        """
        target = self._trial_target
        base = self._runner.sweep_base_value
        if target is None or base is None:
            return
        if not target.imaging.has_stored_properties():
            return

        props = target.imaging.read_properties()
        for name, value in zip(target.attributes, base, strict=True):
            props.set_property(name, value, target.beam)
        target.imaging.write_properties(props)
        self._ctx.text_logger.info(
            f"Restored {dict(zip(target.attributes, base, strict=True))} for the "
            f"next frame of '{target.imaging.name}'."
        )

    def _abort_sweep(self) -> None:
        """Close and discard the active sweep."""
        assert self._active_gen is not None
        self._active_gen.close()
        self._active_gen = None
        self._trial_target = None
        self._current_step_index = 0
        self._runner.abort_sweep()


def _blocking_step_autofunction(
    candidate: Autofunction, actions: Actions
) -> Autofunction | None:
    """
    Return the step-mode autofunction that `candidate` has to wait for, if any.

    Step-mode autofunctions linked to the same imaging run one at a time. A run in
    progress blocks the others until it ends, including the sweeps that follow
    it. Waiting requests start in the order of the slice they were made on;
    ties go to the autofunction earlier in the workflow.

    Args:
        candidate: A step-mode autofunction with a waiting request.
        actions: Every action of the workflow, in workflow order.

    Returns:
        The autofunction with a run in progress on the same imaging, else the
        first one waiting ahead of `candidate`; None if `candidate` may start.
    """
    assert candidate.queued_since is not None

    imaging = candidate.settings.linked_imaging
    position = {id(action): i for i, action in enumerate(actions)}
    rivals = [
        action
        for action in actions
        if isinstance(action, Autofunction)
        and action is not candidate
        and action.settings.linked_imaging == imaging
    ]

    if (running := next((a for a in rivals if a.run_in_progress), None)) is not None:
        return running

    own_place = (candidate.queued_since, position[id(candidate)])
    waiting = [
        ((rival.queued_since, position[id(rival)]), rival)
        for rival in rivals
        if rival.is_step_mode and rival.queued_since is not None
    ]
    ahead = [(place, rival) for place, rival in waiting if place < own_place]
    return min(ahead, key=lambda item: item[0])[1] if ahead else None

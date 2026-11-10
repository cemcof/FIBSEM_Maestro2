# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF
#
"""
The refocus action, which keeps the trial frames of step-mode autofunctions in focus.

During a step-mode autofunction run, each frame of the linked imaging is acquired
at a different trial value of the swept attribute, such as the stigmator. Before
each of these frames, the refocus runs its own short sweep, usually over the
working distance, under the same conditions as the upcoming frame. It then
passes the best value to the imaging. Each trial is therefore scored at its own
best focus.
"""

from typing import TYPE_CHECKING

from fibsem_maestro.action.action import Action
from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.autofunction.autofunction import Autofunction
from fibsem_maestro.autofunction.error import AutofunctionError
from fibsem_maestro.autofunction.sweep_runner import SweepRunner, swept_attributes
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.logging.logging import with_logging_context
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.refocus_settings import RefocusSettings

if TYPE_CHECKING:
    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.autofunction.sweep_step import SweepPoint
    from fibsem_maestro.core.area import RelativeArea
    from fibsem_maestro.core.beam_type import BeamType
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.workflow.actions import Actions


class RefocusState(ActionState):
    # a step-mode run was in progress when the refocus last checked, so the
    # first frame after the run still has to be refocused
    final_refocus_pending: bool = False


@ACTION_REGISTRY.register("refocus")
class Refocus(Action[RefocusSettings, RefocusState]):
    """
    Refocuses the linked imaging's next frame while it is a step-mode trial frame.

    Runs on every slice on which a step-mode autofunction linked to the same
    imaging has a run in progress, and once more after the run ends, so that
    the first frame at the run's result is in focus too. It never refocuses an
    attribute the running autofunction sweeps itself.

    The refocus applies the imaging's upcoming properties (including the
    trial value), runs its own basic or line sweep to completion, and patches
    the best value into the imaging's properties. It stores no properties of
    its own: everything that sets the focus must come from the imaging.
    Its only acquisition settings are the mode's scanning area and `delta_x`.

    It works on the beam of the linked imaging. In the workflow it must run
    after the step-mode autofunction and before the imaging (wrapping around the
    end of the list), so that it sees the trial value the imaging will use.
    """

    _STATE_FIELDS = frozenset({"mode", "linked_imaging"})

    # it stores no properties: nothing to propagate, and anything propagated
    # to it would impose values on the refocus frames
    PROPAGATION_RULES_ALLOWED = False

    def __init__(
        self,
        name: str,
        microscope: Microscope,
        settings: RefocusSettings,
        ctx: ActionContext,
        actions: Actions,
    ):
        super().__init__(name, microscope, settings, ctx, actions)

        # mode, sweeping and sharpness jobs; built for the imaging's beam at
        # the first refocus, since the imaging may come later in the workflow
        self._runner: SweepRunner | None = None

        self._final_refocus_pending = False

    @classmethod
    def settings_cls(cls) -> type[RefocusSettings]:
        return RefocusSettings

    @classmethod
    def state_cls(cls) -> type[RefocusState]:
        return RefocusState

    @property
    def state(self) -> RefocusState:
        return RefocusState(final_refocus_pending=self._final_refocus_pending)

    def set_state(self, state: RefocusState) -> None:
        self._final_refocus_pending = state.final_refocus_pending

    @property
    def beam_type(self) -> BeamType | None:
        """The beam of the linked imaging; None while no imaging is linked."""
        imaging = self._find_imaging()
        return imaging.beam_type if imaging is not None else None

    @property
    def scanning_area(self) -> RelativeArea | None:
        """The scanning area of the basic or line mode, if set."""
        if self._settings.mode.scanning_area:
            return self._settings.mode.scanning_area[0]
        return None

    def preparation_issues(self) -> list[str]:
        """
        Report a missing or unsuitable imaging, and a position that sees stale trial values.

        Returns:
            Human-readable problems; empty if the refocus is ready.
        """
        name = self._settings.linked_imaging
        if not name:
            return ["No imaging is linked."]

        linked = next((a for a in self._actions if a.name == name), None)
        if linked is None:
            return [f"The linked imaging '{name}' does not exist."]
        if not isinstance(linked, Imaging):
            return [f"The linked action '{name}' is not an imaging."]

        issues: list[str] = []
        position = {id(action): i for i, action in enumerate(self._actions)}
        if id(self) not in position:
            return issues

        for autofunction in self._step_autofunctions():
            if not _between(
                position[id(autofunction)],
                position[id(self)],
                position[id(linked)],
                len(self._actions),
            ):
                issues.append(
                    f"Must run after '{autofunction.name}' and before '{name}'; "
                    "otherwise it refocuses the previous trial value."
                )
        return issues

    def should_execute(self) -> bool:
        """
        Decide whether the linked imaging's next frame needs refocusing.

        While a step-mode autofunction linked to the same imaging has a run in
        progress, every next frame is a trial frame, so the refocus runs on
        every slice. After the run ends, it runs once more, for the first frame
        at the run's result.

        Returns:
            `True` if the refocus should run for this slice.
        """
        if (running := self._running_step_autofunction()) is not None:
            self._final_refocus_pending = True
            self._ctx.text_logger.info(
                f"'{self.name}' triggered: '{running.name}' has a step-mode run "
                f"in progress on '{self._settings.linked_imaging}'."
            )
            return True

        if self._final_refocus_pending:
            self._final_refocus_pending = False
            self._ctx.text_logger.info(
                f"'{self.name}' triggered: the step-mode run on "
                f"'{self._settings.linked_imaging}' has ended; refocusing its "
                "next frame at the run's result."
            )
            return True

        return False

    def _run_step(self) -> StepOutcome:
        imaging = self._resolve_imaging()
        best = self._refocus(imaging)

        # basic and line mode always sweep
        assert best is not None
        patch = GlobalProperties()
        attributes = swept_attributes(self._settings.mode)
        for name, value in zip(attributes, best, strict=True):
            patch.set_property(name, value, imaging.beam_type)

        # it stores no properties: the result goes to the imaging only
        return Produced(self.read_properties(), patches={imaging.name: patch})

    @with_logging_context
    def test(self) -> None:
        """
        Refocus once under the linked imaging's stored properties and apply the result.

        Intended for manual testing and diagnostics outside the normal
        acquisition loop.
        """
        imaging = self._resolve_imaging()
        if not imaging.has_stored_properties():
            raise AutofunctionError(
                f"'{imaging.name}' has no stored properties to refocus under; "
                "prepare it first."
            )

        with self._scanning_area_applied():
            self._refocus(imaging)

            # the updated value may not be displayed in the microscope GUI until we start scanning
            self._microscope.beam.grab_frame()

    def wait_for_background_threads(self) -> None:
        if self._runner is not None:
            self._runner.wait()

    def _refocus(self, imaging: Imaging) -> SweepPoint | None:
        """
        Apply the imaging's upcoming properties and run a sweep to completion.

        Args:
            imaging: The linked imaging.

        Returns:
            The best value, now on the microscope; None if nothing was swept.

        Raises:
            AutofunctionError: If the imaging has no beam.
        """
        if (beam_type := imaging.beam_type) is None:
            raise AutofunctionError(f"'{imaging.name}' does not use a beam.")

        runner = self._runner_for(beam_type)
        self._microscope.set_properties(imaging.read_properties(), beam=beam_type)
        return runner.run_to_completion(self.name_with_underscores)

    def _runner_for(self, beam_type: BeamType) -> SweepRunner:
        """
        Return the runner, built from the current settings for `beam_type`.

        Rebuilt before every refocus, so it always follows the settings and
        the imaging's beam.

        Args:
            beam_type: The beam of the linked imaging.

        Returns:
            The runner.
        """
        if self._runner is None:
            self._runner = SweepRunner(
                self._microscope, self._settings, self._ctx, beam_type
            )
        else:
            self._runner.rebuild(beam_type)
        return self._runner

    def _running_step_autofunction(self) -> Autofunction | None:
        """
        Return the step-mode autofunction whose trial frames this refocus must refocus.

        Returns:
            A step-mode autofunction linked to the same imaging with a run in
            progress that does not sweep any attribute the refocus sweeps;
            None if there is none.
        """
        own = set(swept_attributes(self._settings.mode))
        return next(
            (
                autofunction
                for autofunction in self._step_autofunctions()
                if autofunction.run_in_progress
                and not own.intersection(autofunction.swept_attributes)
            ),
            None,
        )

    def _step_autofunctions(self) -> list[Autofunction]:
        """Return the step-mode autofunctions linked to the same imaging."""
        return [
            action
            for action in self._actions
            if isinstance(action, Autofunction)
            and action.is_step_mode
            and action.settings.linked_imaging == self._settings.linked_imaging
        ]

    def _find_imaging(self) -> Imaging | None:
        """Return the linked imaging, or None if it does not exist or is not an imaging."""
        imaging = next(
            (a for a in self._actions if a.name == self._settings.linked_imaging),
            None,
        )
        return imaging if isinstance(imaging, Imaging) else None

    def _resolve_imaging(self) -> Imaging:
        """
        Return the linked imaging.

        Raises:
            AutofunctionError: If the linked action is not an imaging.
        """
        imaging = self._actions.named(self._settings.linked_imaging)
        if not isinstance(imaging, Imaging):
            raise AutofunctionError(
                f"Linked action is not an Imaging action: {imaging.name}"
            )
        return imaging


def _between(start: int, middle: int, end: int, length: int) -> bool:
    """
    Tell whether `middle` comes after `start` and before `end`, going forward cyclically.

    Args:
        start: Position the walk starts after.
        middle: Position to locate.
        end: Position the walk stops before.
        length: Number of positions; the walk wraps around after the last.

    Returns:
        True if walking forward from `start` reaches `middle` before `end`.
    """
    return 0 < (middle - start) % length < (end - start) % length

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import TYPE_CHECKING

from fibsem_maestro.action.action import Action
from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.image import Image8Bit
from fibsem_maestro.drift_correction import DRIFT_CALCULATION_MODES
from fibsem_maestro.drift_correction.error import DriftCorrectionError
from fibsem_maestro.settings.drift_correction_settings import DriftCorrectionSettings

if TYPE_CHECKING:
    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.core.drift import Drift
    from fibsem_maestro.drift_correction.drift_calculation_mode import (
        DriftCalculationMode,
    )
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.workflow.actions import Actions


class DriftCorrectionState(ActionState):
    pass


@ACTION_REGISTRY.register("drift_correction")
class DriftCorrection(Action[DriftCorrectionSettings, DriftCorrectionState]):
    """
    Corrects sample drift between slices by applying a compensating beam shift.
    """

    def __init__(
        self,
        name: str,
        microscope: Microscope,
        settings: DriftCorrectionSettings,
        ctx: ActionContext,
        actions: Actions,
    ):
        super().__init__(name, microscope, settings, ctx, actions)
        # set up the drift calculation method
        self._rebuild()
        # rebuild whenever the settings change
        self._settings.on_change(lambda _: self._rebuild())

    def _rebuild(self) -> None:
        self._drift_calc_name = self._settings.drift_calculation_mode.type
        self._drift_calc: DriftCalculationMode = DRIFT_CALCULATION_MODES.get(
            self._drift_calc_name
        )(
            self._microscope,
            self._settings.drift_calculation_mode,
            self._ctx.image_store(Image8Bit),
            self._ctx.text_logger.derive(self._drift_calc_name),
            self._ctx.image_logger,
        )

    @classmethod
    def settings_cls(cls) -> type[DriftCorrectionSettings]:
        return DriftCorrectionSettings

    @classmethod
    def state_cls(cls) -> type[DriftCorrectionState]:
        return DriftCorrectionState

    @property
    def state(self) -> DriftCorrectionState:
        return DriftCorrectionState()

    def set_state(self, state: DriftCorrectionState) -> None:
        _ = state

    def carry_over_to_next(self) -> None:
        super().carry_over_to_next()
        self._drift_calc.if_skipped(self._ctx.slice)

    def _prepare(self) -> None:
        self._ctx.text_logger.info(
            f"Setting up drift calculation for '{self.name}' for slice {self._ctx.slice}."
        )
        self._drift_calc.setup()

    def preparation_issues(self) -> list[str]:
        issues = self._drift_calc.settings_issues()
        if not self._drift_calc.is_set_up():
            issues.append("drift calculation is not set up; press the Prepare button")
        return issues

    def initialize_first_slice(self) -> None:
        # copy before the base advances the context, while the store still
        # addresses slice 0; if nothing was set up there is nothing to copy,
        # and the workflow's preparation check reports it clearly
        if self._drift_calc.is_set_up():
            store = self._ctx.image_store(Image8Bit)
            self._drift_calc.copy_setup(store, store.next)

        super().initialize_first_slice()

    def _run_step(self) -> StepOutcome:
        if not self._drift_calc.is_set_up():
            # the workflow checks this before each slice; this guards direct calls
            raise DriftCorrectionError(
                f"'{self.name}' has not been prepared: drift calculation is not set up."
            )

        if self._microscope.beam.scan_rotation != 0:
            raise DriftCorrectionError(
                "Scan rotation must be zero for drift correction."
            )

        self._drift_calc.before_calculate_drift(self._ctx.slice)

        beam_shift = self._calculate_correcting_beam_shift()
        if self._microscope.add_beam_shift_with_verification(beam_shift).stage_moved:
            # the stage was moved instead; remove its positioning error
            self._ctx.text_logger.info(
                "The stage was moved instead of the beam; fine-tuning searches the whole frame for its positioning error."
            )
            # the beam shift was reset by the stage move, so this one is in range
            self._microscope.add_beam_shift_with_verification(
                self._calculate_correcting_beam_shift(after_stage_move=True)
            )

        if self._settings.grab_frame_after:
            self._save_result_frame("after_drift_correction")

        self._drift_calc.after_calculate_drift(self._ctx.slice)
        return Produced(self.collect_properties())

    def test(self) -> None:
        raise DriftCorrectionError(f"Testing is not implemented for {self.name}")

    def _calculate_correcting_beam_shift(
        self, after_stage_move: bool = False
    ) -> BeamShift:
        """
        Calculate the beam shift required to compensate for the measured drift.

        Args:
            after_stage_move: True right after a stage move.

        Returns:
            The beam shift to apply in order to compensate for the detected
            drift, or a zero shift if drift calculation failed and the
            acquisition is configured to continue.

        Raises:
            DriftCorrectionError: If drift calculation fails and
                `settings.stop_at_failure` is `True`.
        """
        drift = self._drift_calc.calculate_drift(after_stage_move)

        if not drift.is_valid():
            message = "Drift correction failed: could not calculate drift"
            if drift.reason:
                message += f" ({drift.reason})"
            if self._settings.stop_at_failure:
                raise DriftCorrectionError(f"{message}.")

            self._ctx.text_logger.warning(f"{message}. Not performing correction.")

        # convert drift to beam shift
        return self._drift_to_beam_shift(drift)

    def _drift_to_beam_shift(self, drift: Drift) -> BeamShift:
        """
        Convert a drift measurement in nanometers to a compensating beam shift.

        Args:
            drift: The measured drift in nanometers.

        Returns:
            The beam shift to apply in order to compensate for the drift.
        """
        return BeamShift(
            x=(drift.x or 0.0) * self._microscope.beam.image_to_beam_shift[0],
            y=(drift.y or 0.0) * self._microscope.beam.image_to_beam_shift[1],
        )

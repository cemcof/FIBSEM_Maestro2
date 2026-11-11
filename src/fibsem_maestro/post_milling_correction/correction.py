# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import TYPE_CHECKING

from fibsem_maestro.action.action import Action
from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.post_milling_correction.error import PostMillingCorrectionError
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.dynamic_default import (
    DefaultUnavailableError,
    effective_value,
    required_value,
)
from fibsem_maestro.settings.milling_settings import MillingSettings
from fibsem_maestro.settings.post_milling_correction_settings import (
    PostMillingCorrectionSettings,
)

if TYPE_CHECKING:
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl

_WORKING_DISTANCE = "working_distance"
_BEAM_SHIFT = "beam_shift"
_Y_CORRECTION = "y_correction"
_WD_CORRECTION = "wd_correction"


class PostMillingCorrectionState(ActionState):
    pass


@ACTION_REGISTRY.register("post_milling_correction")
class PostMillingCorrection(
    Action[
        PostMillingCorrectionSettings,
        PostMillingCorrectionState,
    ]
):
    """
    Keeps the corrected actions on the cross-section face as it recedes with milling.

    Every run adds the shifts of one milling step to the upcoming properties of
    each corrected action, on the action's beam: the vertical image shift (as
    beam shift) and the working-distance change. Any action working on a beam
    can be corrected, as long as it collects the corrected properties. Shifts
    not set follow the geometry of the linked milling, which assumes the
    corrected actions image with the electron beam.

    The corrections accumulate, since every action carries its properties on to the next slice.

    The correction never changes the microscope (it does not even select a
    beam) and stores no properties of its own. It takes no part in propagation
    rules: its changes are relative, and reach exactly the corrected actions.
    """

    # it stores no properties, and its corrections go to the corrected actions only
    PROPAGATION_RULES_ALLOWED = False

    @classmethod
    def settings_cls(cls) -> type[PostMillingCorrectionSettings]:
        """
        Class of the class used for the action's settings.
        """
        return PostMillingCorrectionSettings

    @classmethod
    def state_cls(cls) -> type[PostMillingCorrectionState]:
        return PostMillingCorrectionState

    @property
    def state(self) -> PostMillingCorrectionState:
        # post milling correction has no persistent internal state
        return PostMillingCorrectionState()

    def set_state(
        self,
        state: PostMillingCorrectionState,
    ) -> None:
        _ = state

    def test(self) -> None:
        raise PostMillingCorrectionError(f"Testing is not implemented for {self.name}")

    def preparation_issues(self) -> list[str]:
        """
        Report a missing default correction, and corrected actions that cannot take the correction.

        Uses the settings only, never the stored properties: a corrected action
        that collects the corrected properties also stores them once prepared.

        Returns:
            Human-readable problems; empty if the correction is ready.
        """
        issues: list[str] = []
        if self._follows_default():
            issues.extend(self._milling_issues())

        names = self._settings.corrected_actions
        if not names:
            issues.append("No actions to correct are selected.")

        by_name = {action.name: action for action in self._actions}
        for name in dict.fromkeys(names):
            target = by_name.get(name)
            if target is None:
                issues.append(f"The corrected action '{name}' does not exist.")
            else:
                issues.extend(self._target_issues(target))
        return issues

    def _follows_default(self) -> bool:
        """Whether a correction follows its default, computed from the linked milling."""
        return (
            self._settings.y_correction is None or self._settings.wd_correction is None
        )

    def _milling_issues(self) -> list[str]:
        """
        The reasons the default corrections cannot be computed.

        Returns:
            Human-readable problems with the linked milling; empty if it is usable.
        """
        name = self._settings.linked_milling
        if not name:
            return ["No milling is linked to compute the default correction from."]
        linked = self.default_context().settings_of(name)
        if linked is None:
            return [f"The linked milling '{name}' does not exist."]
        if not isinstance(linked, MillingSettings):
            return [f"The linked action '{name}' is not a milling."]
        return []

    def _target_issues(self, target: Action) -> list[str]:
        """
        The reasons a corrected action cannot take the correction.

        Args:
            target: The corrected action.

        Returns:
            Human-readable problems; empty if it can.
        """
        beam = target.beam_type
        if beam is None:
            return [_no_beam_message(target)]

        collected = _collected(target, beam)
        return [
            f"'{target.name}' does not collect '{name}', so it cannot be corrected."
            for name in self._corrected_property_names()
            if name not in collected
        ]

    def _corrected_property_names(self) -> list[str]:
        """
        The properties every run may change: those of the non-zero corrections.

        A correction whose default cannot be computed counts as non-zero.
        """
        y_correction, wd_correction = (
            effective_value(self._settings, name, self.default_context())
            for name in (_Y_CORRECTION, _WD_CORRECTION)
        )
        names: list[str] = []
        if y_correction is None or y_correction:
            names.append(_BEAM_SHIFT)
        if wd_correction is None or wd_correction:
            names.append(_WORKING_DISTANCE)
        return names

    def _run_step(self) -> StepOutcome:
        y_correction, wd_correction = self._correction()
        self._ctx.text_logger.info(
            f"Correcting {', '.join(repr(n) for n in self._settings.corrected_actions)}"
            f" by y_correction={y_correction} nm, wd_correction={wd_correction} nm."
        )

        patches: dict[str, GlobalProperties] = {}
        if y_correction or wd_correction:
            for target in self._resolve_targets():
                patches[target.name] = self._corrected_properties(
                    target, y_correction, wd_correction
                )

        # it stores no properties; empty ones also replace any stored by an older version
        return Produced(GlobalProperties(), patches=patches)

    def _correction(self) -> tuple[float, float]:
        """
        The shifts of one run: the values set, or else their defaults.

        Returns:
            The vertical image shift and the working-distance change, both in nm.

        Raises:
            PostMillingCorrectionError: If a correction follows its default and
                the default cannot be computed.
        """
        context = self.default_context()
        try:
            # a default used is logged
            return (
                required_value(self._settings, _Y_CORRECTION, context),
                required_value(self._settings, _WD_CORRECTION, context),
            )
        except DefaultUnavailableError as e:
            problems = " ".join(self._milling_issues())
            raise PostMillingCorrectionError(
                f"The default correction cannot be computed: {problems}"
            ) from e

    def _resolve_targets(self) -> list[Action]:
        """
        The corrected actions, each once, in the order they are listed.

        Raises:
            PostMillingCorrectionError: If a corrected action does not exist.
        """
        targets: list[Action] = []
        for name in dict.fromkeys(self._settings.corrected_actions):
            try:
                targets.append(self._actions.named(name))
            except KeyError as e:
                raise PostMillingCorrectionError(
                    f"The corrected action '{name}' does not exist."
                ) from e
        return targets

    def _corrected_properties(
        self, target: Action, y_correction: float, wd_correction: float
    ) -> GlobalProperties:
        """
        The target's upcoming beam shift and working distance, with the correction added.

        The target's props file for its current slice holds its upcoming
        properties: those of this slice if it has not run yet, of the next one
        if it has. Patches are written to the same file.

        Args:
            target: The corrected action.
            y_correction: Vertical shift in image coordinates, in nm.
            wd_correction: Working-distance change, in nm.

        Returns:
            Only the corrected properties, those of the non-zero corrections,
            on the target's beam.

        Raises:
            PostMillingCorrectionError: If the target works on no beam, or stores
                no value of a property to correct.
        """
        beam = target.beam_type
        if beam is None:
            raise PostMillingCorrectionError(_no_beam_message(target))

        props = target.read_properties()
        stored = getattr(props, props.get_properties_attr_name(beam))
        if stored is None:
            stored = BeamProperties()

        patch = GlobalProperties()
        if y_correction:
            if stored.beam_shift is None:
                raise _not_stored(target, _BEAM_SHIFT, beam)
            to_beam_shift = self._beam_control_of(beam).image_to_beam_shift
            delta = BeamShift(0.0, to_beam_shift[1] * y_correction)
            patch.set_property(_BEAM_SHIFT, stored.beam_shift + delta, beam)

        if wd_correction:
            if stored.working_distance is None:
                raise _not_stored(target, _WORKING_DISTANCE, beam)
            patch.set_property(
                _WORKING_DISTANCE, stored.working_distance + wd_correction, beam
            )

        return patch

    def _beam_control_of(self, beam: BeamType) -> BeamControl:
        """
        The control of a corrected action's beam, for its image-to-beam-shift factors.

        Args:
            beam: The beam.

        Returns:
            Its control.
        """
        match beam:
            case BeamType.ELECTRON:
                return self._microscope.electron_beam
            case BeamType.ION:
                return self._microscope.ion_beam


def _no_beam_message(target: Action) -> str:
    """The problem of a corrected action that works on no beam (e.g. an adjust-props)."""
    return f"'{target.name}' works on no beam, so it cannot be corrected."


def _collected(target: Action, beam: BeamType) -> list[str]:
    """The names of the properties an action collects on a beam."""
    names = target.props_to_collect
    return getattr(names, GlobalProperties().get_properties_attr_name(beam))


def _not_stored(
    target: Action, name: str, beam: BeamType
) -> PostMillingCorrectionError:
    """The error for a corrected action that stores no value of a corrected property."""
    return PostMillingCorrectionError(
        f"'{target.name}' stores no '{name}' of the {beam.value} beam to correct; "
        "prepare it again."
    )

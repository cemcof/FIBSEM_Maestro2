# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from scipy.spatial import distance  # pyright: ignore[reportMissingTypeStubs]

from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.stage_position import StagePosition
from fibsem_maestro.microscope.error import MicroscopeError
from fibsem_maestro.microscope.positioning_outcome import PositioningOutcome
from fibsem_maestro.microscope.registry import MICROSCOPE_CONTROLS
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.properties.microscope_properties import MicroscopeProperties

if TYPE_CHECKING:
    from collections.abc import Generator

    from fibsem_maestro.core.provenance import Provenance
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
    from fibsem_maestro.microscope.abstract_control.microscope_control import (
        MicroscopeControl,
    )
    from fibsem_maestro.properties.beam_properties import BeamProperties
    from fibsem_maestro.settings.microscope_settings import MicroscopeSettings
    from fibsem_maestro.settings.property_names import PropertyNames


class Microscope:
    """
    Central abstraction layer for controlling an electron microscope.

    Args:
        settings: Microscope configuration including the control type, IP address,
            stage tolerances, and holder pretilt.
        txt_log: Logger for diagnostic and status messages.
    """

    def __init__(
        self,
        settings: MicroscopeSettings,
        txt_log: TextLogger,
    ):
        self._txt_log = txt_log

        self._settings = settings

        if self._settings.port:
            try:
                port_int: int | None = int(self._settings.port)
            except ValueError:
                raise MicroscopeError("Invalid port: must be an integer or empty.")
        else:
            port_int: int | None = None

        self._control = MICROSCOPE_CONTROLS.get(settings.control)(
            self._settings.ip_address, port_int, self._txt_log
        )
        self.beam = self._control.electron_beam

    @property
    def txt_log(self) -> TextLogger:
        """Logger for diagnostic and status messages."""
        return self._txt_log

    @property
    def electron_beam(self) -> BeamControl:
        """The electron beam control interface."""
        return self._control.electron_beam

    @property
    def ion_beam(self) -> BeamControl:
        """The ion beam control interface."""
        return self._control.ion_beam

    @property
    def control(self) -> MicroscopeControl:
        """The microscope control interface."""
        return self._control

    @property
    def settings(self) -> MicroscopeSettings:
        """The microscope settings."""
        return self._settings

    @property
    def provenance(self) -> Provenance:
        """The provenance of the microscope and it's control units."""
        return self._control.provenance()

    @settings.setter
    def settings(self, new_settings: MicroscopeSettings) -> None:
        """Sets the microscope settings. Does NOT reconnect the microscope even if connection fields changed."""
        self._settings = new_settings

    def set_beam(self, type: BeamType) -> None:
        """Set the active beam to the electron or ion beam.

        Updates `self.beam` to point to the selected beam control interface.
        Subsequent operations that use `self.beam` will act on the selected beam.

        Args:
            type: The beam type to activate, either `BeamType.ELECTRON` or `BeamType.ION`.
        """
        match type:
            case BeamType.ELECTRON:
                self.beam = self._control.electron_beam
            case BeamType.ION:
                self.beam = self._control.ion_beam

    def set_stage_position_with_verification(
        self, new_stage_position: StagePosition
    ) -> PositioningOutcome:
        """
        Move the stage to an absolute position and check the position it reports.

        Only the XY position is checked, against `stage_tolerance`; a stage
        off target is logged as a warning.

        Args:
            new_stage_position: The target stage position in nanometers and degrees.

        Returns:
            `STAGE_MOVED`, or `STAGE_OFF_TARGET` if the stage reports a position
            off target.
        """
        actual_position = self._control.try_set_stage_position(new_stage_position)
        return self._check_stage_position(actual_position, new_stage_position)

    def move_stage_position_with_verification(
        self, delta: StagePosition
    ) -> PositioningOutcome:
        """
        Move the stage by a relative offset and check the position it reports.

        Axes that `delta` leaves unchanged are never commanded. Only the XY
        position is checked, against `stage_tolerance`; a stage off target is
        logged as a warning.

        Args:
            delta: The relative stage movement in nanometers and degrees.

        Returns:
            `STAGE_MOVED`, or `STAGE_OFF_TARGET` if the stage reports a position
            off target.
        """
        target = self._control.stage_position + delta
        actual_position = self._control.try_move_stage_position(delta)
        return self._check_stage_position(actual_position, target)

    def _check_stage_position(
        self, actual: StagePosition, target: StagePosition
    ) -> PositioningOutcome:
        """
        Compare the XY position the stage reports with the target.

        Args:
            actual: The position the stage reports, in nanometers.
            target: The position it was sent to, in nanometers.

        Returns:
            `STAGE_MOVED` if they are within `stage_tolerance`, else `STAGE_OFF_TARGET`.
        """
        dist = distance.euclidean(actual.to_xy(), target.to_xy())
        if dist <= self._settings.stage_tolerance:
            return PositioningOutcome.STAGE_MOVED

        self._txt_log.warning(
            f"Stage off target: target={target}, actual={actual}, "
            f"distance={dist:.3f} nm > tolerance={self._settings.stage_tolerance} nm."
        )
        return PositioningOutcome.STAGE_OFF_TARGET

    def set_beam_shift_with_verification(
        self, new_beam_shift: BeamShift, beam: BeamControl | None = None
    ) -> PositioningOutcome:
        """
        Apply a beam shift and verify that it is within tolerance.

        Attempts to set the beam shift on the given beam.

        If the Euclidean distance between the requested and actual shift exceeds
        `beam_shift_tolerance` or if setting the beam shift fails internally,
        falls back to an equivalent stage move and resets the beam shift to zero.

        Args:
            new_beam_shift: The desired beam shift in nanometers.
            beam: The beam to shift. Defaults to the currently active beam.

        Returns:
            `BEAM_SHIFTED` if the beam shift was applied within tolerance,
            otherwise the outcome of the fallback stage move.
        """
        beam = beam or self.beam
        beam_type = beam.beam_type()

        # try setting beam shift
        try:
            beam.beam_shift = new_beam_shift
            actual_beam_shift = beam.beam_shift

            dist = distance.euclidean(
                actual_beam_shift.to_tuple(), new_beam_shift.to_tuple()
            )

            if dist > self._settings.beam_shift_tolerance:
                raise MicroscopeError("Beam shift out of range.")

            return PositioningOutcome.BEAM_SHIFTED
        except Exception as e:
            self._txt_log.warning(f"Beam shift error: {e}. Adjusting stage position.")

            beam_shift_to_stage_move = (
                self._settings.beam_shift_to_stage_move_electron
                if beam_type == BeamType.ELECTRON
                else self._settings.beam_shift_to_stage_move_ion
            )

            stage_move = [
                new_beam_shift.x * beam_shift_to_stage_move[0],
                new_beam_shift.y * beam_shift_to_stage_move[1],
            ]
            # move stage
            outcome = self.move_stage_position_with_verification(
                StagePosition(x=stage_move[0], y=stage_move[1])
            )
            # set beam shift to zero
            beam.beam_shift = BeamShift(0.0, 0.0)

            return outcome

    def add_beam_shift_with_verification(
        self, delta: BeamShift, beam: BeamControl | None = None
    ) -> PositioningOutcome:
        """
        Add a beam shift delta to the current shift and verify the result.

        Adds `delta` to the current beam shift and delegates to `set_beam_shift_with_verification`.

        Args:
            delta: The beam shift increment in nanometers.
            beam: The beam to shift. Defaults to the currently active beam.

        Returns:
            `BEAM_SHIFTED` if the resulting beam shift was applied within
            tolerance, otherwise the outcome of the fallback stage move.
        """
        beam = beam or self.beam
        return self.set_beam_shift_with_verification(beam.beam_shift + delta, beam)

    def set_beam_shift_without_stage_fallback(
        self, new_beam_shift: BeamShift, beam: BeamControl | None = None
    ) -> None:
        """
        Apply a beam shift and verify it, never moving the stage instead.

        For shifts that must not be replaced by a stage move, such as offsets
        applied to every frame, where a fallback stage move would displace the
        sample each time.

        Args:
            new_beam_shift: The desired beam shift in nanometers.
            beam: The beam to shift. Defaults to the currently active beam.

        Raises:
            MicroscopeError: If the shift cannot be set or ends up further than
                `beam_shift_tolerance` from the target. The beam may then be
                partly shifted; callers restore it.
        """
        beam = beam or self.beam
        try:
            beam.beam_shift = new_beam_shift
            actual_beam_shift = beam.beam_shift
        except Exception as e:
            raise MicroscopeError(
                f"Could not set beam shift to {new_beam_shift}: {e}"
            ) from e

        dist = distance.euclidean(
            actual_beam_shift.to_tuple(), new_beam_shift.to_tuple()
        )
        if dist > self._settings.beam_shift_tolerance:
            raise MicroscopeError(
                f"Beam shift {new_beam_shift} is out of range; the beam reached "
                f"{actual_beam_shift}."
            )

    def add_beam_shift_without_stage_fallback(
        self, delta: BeamShift, beam: BeamControl | None = None
    ) -> None:
        """
        Add a beam shift delta to the current shift, never moving the stage instead.

        Args:
            delta: The beam shift increment in nanometers.
            beam: The beam to shift. Defaults to the currently active beam.

        Raises:
            MicroscopeError: If the resulting shift is out of range (see
                `set_beam_shift_without_stage_fallback`).
        """
        beam = beam or self.beam
        self.set_beam_shift_without_stage_fallback(beam.beam_shift + delta, beam)

    def set_properties(
        self, properties: GlobalProperties, beam: BeamType | None
    ) -> None:
        """
        Apply a set of properties to the microscope and its beams.

        Sets stage position, manufacturer properties, and beam properties for
        the selected beam type. Beam shift is handled at this level rather than
        delegating to the beam control, since it may require a stage move as a
        fallback. After handling beam shift, the remaining beam properties are
        delegated to the beam control.

        Args:
            properties: The properties to apply to the microscope.
            beam: If specified, only properties for the selected beam and the
                microscope itself are applied. If `None`, properties for all
                beams are applied.

        Raises:
            MicroscopeError: If a manufacturer property cannot be set.
        """
        if (microscope := properties.microscope) is not None:
            if (stage_position := microscope.stage_position) is not None:
                self.set_stage_position_with_verification(stage_position)

            # set manufacturer properties of the microscope
            for field_name in filter(
                lambda x: x in self._control.manufacturer_prop_names,
                microscope.model_dump(exclude_none=True).keys(),
            ):
                try:
                    value = getattr(microscope, field_name)
                    self._control.set_manufacturer_prop(field_name, value)
                    continue
                except Exception as e:
                    raise MicroscopeError(
                        f"Could not set manufacturer property '{field_name}': {e}"
                    ) from e

        # set properties of the electron beam
        if properties.electron_beam is not None and (
            beam is None or beam is BeamType.ELECTRON
        ):
            self._set_beam_properties(
                properties.electron_beam, self._control.electron_beam
            )

        # set properties of the ion beam
        if properties.ion_beam is not None and (beam is None or beam is BeamType.ION):
            self._set_beam_properties(properties.ion_beam, self._control.ion_beam)

    def _set_beam_properties(self, props: BeamProperties, beam: BeamControl) -> None:
        """
        Apply beam properties, handling the beam shift here.

        The beam shift is set by the microscope rather than the beam control,
        since it may fall back to a stage move. `props` is left unchanged.

        Args:
            props: The properties to apply.
            beam: The beam to apply them to.
        """
        if props.beam_shift is not None:
            self.set_beam_shift_with_verification(props.beam_shift, beam)

        remaining = props.model_copy(deep=True)
        remaining.beam_shift = None
        beam.set_properties(remaining)

    def collect_properties(
        self, properties_to_collect: PropertyNames
    ) -> GlobalProperties:
        """
        Read the current microscope state into a `GlobalProperties` instance.

        Collects only the properties listed in `properties_to_collect`.

        Standard `MicroscopeProperties` fields are read from the control
        directly; manufacturer-specific properties are retrieved via
        `manufacturer_prop`. Unknown property names are logged as warnings
        and excluded from the result.

        Args:
            properties_to_collect: The names of the properties to collect from
                the microscope and each beam.

        Returns:
            A `GlobalProperties` instance containing the collected values.
        """
        # get field names to write out
        field_names = list(
            filter(
                lambda x: x in properties_to_collect.microscope,
                MicroscopeProperties.model_fields.keys(),
            )
        )

        # collect the values of the properties
        values = {}
        for field_name in field_names:
            values[field_name] = getattr(self._control, field_name)

        # collect internal properties
        for field_name in filter(
            lambda x: x in properties_to_collect.microscope,
            self._control.manufacturer_prop_names,
        ):
            values[field_name] = self._control.manufacturer_prop(field_name)

        # get unknown properties
        unknown = [
            prop for prop in properties_to_collect.microscope if prop not in values
        ]
        if len(unknown) > 0:
            self._txt_log.warning(
                f"The following selected microscope properties are not known: {' '.join(unknown)}"
            )

        electron_beam_properties = self._control.electron_beam.collect_properties(
            properties_to_collect.electron_beam
        )
        ion_beam_properties = self._control.ion_beam.collect_properties(
            properties_to_collect.ion_beam
        )

        return GlobalProperties(
            microscope=MicroscopeProperties(**values),
            electron_beam=electron_beam_properties,
            ion_beam=ion_beam_properties,
        )

    @contextmanager
    def set_temporary_properties(self, props: GlobalProperties) -> Generator[None]:
        """
        Temporarily set the microscope properties to the given values, and restore them when done.
        """
        backup = self.collect_properties(props.get_property_names())
        self.set_properties(props, None)
        try:
            yield
        finally:
            self.set_properties(backup, None)

    @contextmanager
    def set_temporary_beam_property(
        self, property: str, value: Any, beam: BeamType | None = None
    ) -> Generator[None]:
        """
        Temporarily set a single property of the given beam to the given value, and restore it when done.

        Args:
            property: The name of the property to set.
            value: The value to set the property to.
            beam: The beam to set the property on, or `None` to use the current beam.
        """
        match beam:
            case BeamType.ELECTRON:
                beam_control = self.control.electron_beam
            case BeamType.ION:
                beam_control = self.control.ion_beam
            case _:
                beam_control = self.beam

        backup = getattr(beam_control, property)
        setattr(beam_control, property, value)

        try:
            yield
        finally:
            try:
                setattr(beam_control, property, backup)
            except Exception as e:
                self._txt_log.warning(f"Could not restore property {property}: {e}")

    @contextmanager
    def add_temporary_beam_shift(
        self, offset: BeamShift, beam: BeamControl | None = None
    ) -> Generator[None]:
        """
        Temporarily shift the beam by the specified offset.

        Out-of-range shifts fall back to a stage move, which resets the beam
        shift to zero; undoing only the offset would then leave the stage
        displaced. The original stage position and beam shift are therefore
        restored as they were.

        Args:
            offset: The offset to shift the beam by, in nanometers.
            beam: The beam to shift. Defaults to the currently active beam.

        Yields:
            None: Control is yielded to the caller with the beam shifted.
        """
        beam = beam or self.beam
        base_shift = beam.beam_shift
        base_stage = self._control.stage_position
        self._txt_log.debug(
            f"Temporarily shifting beam by x = {offset.x}, y = {offset.y}."
        )

        # assume a stage move until the shift reports otherwise, so that a
        # failure halfway through the fallback still restores the stage
        stage_may_have_moved = True
        try:
            stage_may_have_moved = self.add_beam_shift_with_verification(
                offset, beam
            ).stage_moved

            if stage_may_have_moved:
                self._txt_log.info(
                    "Temporary beam shift is out of range; the stage was moved "
                    "instead and will be moved back afterwards."
                )
            yield

        finally:
            if stage_may_have_moved:
                try:
                    self._txt_log.debug(f"Restoring stage position to {base_stage}.")
                    current = self._control.stage_position
                    self.move_stage_position_with_verification(
                        StagePosition(
                            x=base_stage.x - current.x, y=base_stage.y - current.y
                        )
                    )
                except Exception as e:
                    self._txt_log.warning(f"Could not restore stage position: {e}")

            try:
                self._txt_log.debug(f"Restoring beam shift to {base_shift}.")
                if self.set_beam_shift_with_verification(base_shift, beam).stage_moved:
                    self._txt_log.warning(
                        f"Could not restore beam shift {base_shift}; the stage "
                        "was moved instead."
                    )
            except Exception as e:
                self._txt_log.warning(f"Could not restore beam shift: {e}")

    @contextmanager
    def add_temporary_stage_offset(self, offset: StagePosition) -> Generator[None]:
        """
        Temporarily move the stage by the specified offset.

        Yields:
            None: Control is yielded to the caller with the stage displaced.
        """
        # move the stage away
        self.move_stage_position_with_verification(offset)
        self._txt_log.info(
            f"Temporarily moving stage by x = {offset.x}, y = {offset.y}, z = {offset.z}, rotation = {offset.rotation}, tilt = {offset.tilt}."
        )

        try:
            yield
        finally:
            try:
                restore = offset.inverted()
                self._txt_log.debug(
                    "Restoring stage position. "
                    f"Changing by x = {restore.x}, y = {restore.y}, z = {restore.z}, rotation = {restore.rotation}, tilt = {restore.tilt}."
                )
                self.move_stage_position_with_verification(restore)
            except Exception as e:
                self._txt_log.warning(f"Could not restore stage position: {e}")

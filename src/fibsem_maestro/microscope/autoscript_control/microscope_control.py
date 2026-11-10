# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from functools import cached_property
from typing import TYPE_CHECKING, Any

from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from autoscript_sdb_microscope_client.sdb_microscope_client import SdbMicroscopeClient

from fibsem_maestro.core.pattern_type import PatternType
from fibsem_maestro.core.provenance import Provenance
from fibsem_maestro.core.stage_position import StagePosition
from fibsem_maestro.microscope.abstract_control.microscope_control import (
    MicroscopeControl,
)
from fibsem_maestro.microscope.autoscript_control.beam_control import (
    AutoscriptElectronBeamControl,
    AutoscriptIonBeamControl,
)
from fibsem_maestro.microscope.autoscript_control.manufacturer_props import (
    AutoscriptManufacturerPropertiesRegistry,
)

if TYPE_CHECKING:
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl


class AutoscriptMicroscopeControl(MicroscopeControl):
    def __init__(self, ip_address: str, port: int | None, txt_log: TextLogger):
        self._txt_log = txt_log
        self._microscope = SdbMicroscopeClient()

        if port:
            self._microscope.connect(ip_address, port)
            self._txt_log.info(f"Connecting to {ip_address}:{port}.")
        else:
            self._microscope.connect(ip_address)
            self._txt_log.info(f"Connecting to {ip_address}.")

        # Maestro reads and moves the stage in specimen coordinates with the Z
        # axis unlinked from the working distance
        self._unlink_stage()
        self._txt_log.info(
            "Stage set to specimen coordinates with Z unlinked from the working distance."
        )

        self._manufacturer_properties = AutoscriptManufacturerPropertiesRegistry(
            self._microscope
        )

        self._electron_beam: BeamControl = AutoscriptElectronBeamControl(
            self._microscope,
            self._txt_log.derive("electron_beam"),
        )
        self._ion_beam: BeamControl = AutoscriptIonBeamControl(
            self._microscope,
            self._txt_log.derive("ion_beam"),
        )

    def _unlink_stage(self):
        """Unlink the microscope stage and set the default coordinate system to SPECIMEN."""
        stage = self._microscope.specimen.stage
        stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
        stage.unlink()

    @classmethod
    def provenance(cls) -> Provenance:
        return Provenance.AUTOSCRIPT

    @property
    def autoscript_microscope(self) -> SdbMicroscopeClient:
        """The actual Autoscript microscope instance."""
        return self._microscope

    @property
    def stage_position(self) -> StagePosition:
        # we unlink stage every time before reading stage position
        # in case the user links it back
        self._unlink_stage()

        p = StagePosition.from_stage_position_autoscript(
            self._microscope.specimen.stage.current_position
        )

        self._txt_log.debug(f"Getting stage position {p}.")
        return p

    @property
    def electron_beam(self) -> BeamControl:
        return self._electron_beam

    @electron_beam.setter
    def electron_beam(self, beam: BeamControl):
        self._electron_beam = beam

    @property
    def ion_beam(self) -> BeamControl:
        return self._ion_beam

    @ion_beam.setter
    def ion_beam(self, beam: BeamControl):
        self._ion_beam = beam

    def manufacturer_prop(self, name: str) -> Any:
        property = self._manufacturer_properties.get(name)
        value = property.get()
        self._txt_log.debug(f"Getting manufacturer property '{name}': {value}.")
        return value

    def set_manufacturer_prop(self, name: str, value: Any):
        self._txt_log.debug(f"Setting manufacturer property '{name}': {value}.")
        property = self._manufacturer_properties.get(name)
        property.set(value)

    @property
    def manufacturer_prop_names(self) -> list[str]:
        return self._manufacturer_properties.allowed()

    def try_set_stage_position(self, pos: StagePosition) -> StagePosition:
        self._unlink_stage()

        self._txt_log.debug(f"Current stage position: {self.stage_position}.")
        pos_autoscript = pos.to_stage_position_autoscript()

        self._microscope.specimen.stage.absolute_move(pos_autoscript)
        self._txt_log.debug(f"Moving stage to {pos}.")

        return self.stage_position

    def try_move_stage_position(self, delta: StagePosition) -> StagePosition:
        self._unlink_stage()

        self._txt_log.debug(f"Current stage position: {self.stage_position}.")

        delta_autoscript = delta.to_stage_delta_autoscript()
        self._microscope.specimen.stage.relative_move(delta_autoscript)

        self._txt_log.debug(f"Moving stage by {delta}.")
        return self.stage_position

    @property
    def txt_log(self) -> TextLogger:
        return self._txt_log

    @cached_property
    def available_patterns(self) -> list[PatternType]:
        return [
            PatternType(x)
            for x in self._microscope.patterning.list_all_application_files()
        ]

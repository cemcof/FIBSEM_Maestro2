# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import pytest

pytest.importorskip("autoscript_sdb_microscope_client")

from autoscript_sdb_microscope_client.sdb_microscope_client import SdbMicroscopeClient

from fibsem_maestro.action.defaults import ActionDefaults
from fibsem_maestro.autofunction.sweeping import Sweeping
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.detector_mode import DetectorMode
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.logging.text.memory import MemoryTextLogger
from fibsem_maestro.microscope.autoscript_control.beam_control import (
    AutoscriptElectronBeamControl,
    AutoscriptIonBeamControl,
)
from fibsem_maestro.microscope.error import MicroscopeError
from fibsem_maestro.microscope.microscope import Microscope
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.microscope_settings import MicroscopeSettings
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.settings.sweeping_settings import (
    LinearSweeping as LinearSweepingSettings,
)

pytest.importorskip("autoscript_sdb_microscope_client")

from autoscript_fakes import FakeSdbClient

from fibsem_maestro.microscope.autoscript_control import (
    microscope_control as autoscript_microscope_control,
)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSdbClient:
    """The client the Autoscript control will connect through."""
    client = FakeSdbClient()
    monkeypatch.setattr(
        autoscript_microscope_control, "SdbMicroscopeClient", lambda: client
    )
    return client


_NO_DEFAULTS = ActionDefaults(None, (), lambda: None)
"""The ranges are set: no defaults to compute."""


@pytest.fixture
def autoscript_microscope(fake_client: FakeSdbClient) -> Microscope:
    _ = fake_client  # connects through the fake
    return Microscope(
        MicroscopeSettings(control="autoscript", ip_address="localhost"),
        MemoryTextLogger(lambda: 0),
    )


def test_ion_beam_working_distance_can_be_swept(
    autoscript_microscope: Microscope,
) -> None:
    sweeping = Sweeping(
        autoscript_microscope.ion_beam,
        LinearSweepingSettings(
            target_attribute="working_distance", range=(-1000.0, 1000.0), steps=3
        ),
        MemoryTextLogger(lambda: 0),
        _NO_DEFAULTS,
    )
    assert len(list(sweeping.sweep())) == 3


def test_sweep_values_outside_instrument_limits_are_skipped(
    autoscript_microscope: Microscope,
) -> None:
    # the fake ion beam allows 1-60 mm; 16.5 mm +- 50 mm leaves one valid value
    sweeping = Sweeping(
        autoscript_microscope.ion_beam,
        LinearSweepingSettings(
            target_attribute="working_distance", range=(-50e6, 50e6), steps=3
        ),
        MemoryTextLogger(lambda: 0),
        _NO_DEFAULTS,
    )
    assert [s.value[0] for s in sweeping.sweep()] == pytest.approx([16.5e6])


def test_autoscript_reapplying_pixel_size_keeps_standard_resolution(
    autoscript_microscope: Microscope, fake_client: FakeSdbClient
) -> None:
    beam = autoscript_microscope.electron_beam
    beam.resolution = Resolution(1024, 884)
    fake_client.beams.electron_beam.horizontal_field_width.value = 1192.7193611970083e-9

    beam.pixel_size = beam.pixel_size

    assert beam.resolution == Resolution(1024, 884)
    assert fake_client.beams.electron_beam.scanning.resolution.value == "1024x884"


@pytest.mark.parametrize(
    "beam_cls", [AutoscriptElectronBeamControl, AutoscriptIonBeamControl]
)
def test_beam_manufacturer_properties_belong_to_that_beam(beam_cls: type) -> None:
    beam = beam_cls(SdbMicroscopeClient(), MemoryTextLogger(lambda: 0))
    names = beam.manufacturer_prop_names

    assert names, "the beam's own properties must still be discovered"
    assert not [n for n in names if "application_client" in n]
    assert "beam_current.value" in names
    assert "scanning.dwell_time.value" in names


def test_microscope_manufacturer_properties_have_one_path_each() -> None:
    from fibsem_maestro.microscope.autoscript_control.manufacturer_props import (
        AutoscriptManufacturerPropertiesRegistry,
    )

    names = AutoscriptManufacturerPropertiesRegistry(SdbMicroscopeClient()).allowed()
    assert not [n for n in names if "application_client" in n]
    assert "beams.ion_beam.beam_current.value" in names


def test_detector_mode_is_applied_before_contrast_and_brightness(
    autoscript_microscope: Microscope, fake_client: FakeSdbClient
) -> None:
    """Switching the mode resets contrast and brightness, so it must go first."""
    props = GlobalProperties(
        electron_beam=BeamProperties(
            detector_contrast=0.8,
            detector_brightness=0.3,
            detector_mode=DetectorMode.BACKSCATTER_ELECTRONS,
        )
    )

    autoscript_microscope.set_properties(props, BeamType.ELECTRON)

    detector = fake_client.detector
    assert detector.mode.value == "BackscatterElectrons"
    assert (detector.contrast.value, detector.brightness.value) == (0.8, 0.3)


def test_unavailable_detector_mode_is_rejected(
    autoscript_microscope: Microscope, fake_client: FakeSdbClient
) -> None:
    with pytest.raises(MicroscopeError, match="not available"):
        autoscript_microscope.electron_beam.detector_mode = DetectorMode.SEGMENT_A

    assert fake_client.detector.mode.value == "SecondaryElectrons"


def test_detector_mode_is_collected_as_a_beam_property(
    autoscript_microscope: Microscope,
) -> None:
    props = autoscript_microscope.collect_properties(
        PropertyNames(electron_beam=["detector_mode"])
    )

    assert props.electron_beam is not None
    assert props.electron_beam.detector_mode is DetectorMode.SECONDARY_ELECTRONS

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import pytest

from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.stage_position import StagePosition
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.properties.microscope_properties import MicroscopeProperties
from fibsem_maestro.settings.property_names import PropertyNames


def test_accumulate_property_creates_electron_beam_from_none():
    props = GlobalProperties.model_validate({})
    assert props.electron_beam is None

    props.accumulate_property("dwell_time", 1.5e-6, beam_type=BeamType.ELECTRON)

    assert props.electron_beam is not None
    assert props.electron_beam.dwell_time == pytest.approx(1.5e-6)


def test_accumulate_property_creates_ion_beam_from_none():
    props = GlobalProperties.model_validate({})
    assert props.ion_beam is None

    props.accumulate_property("dwell_time", 2.0e-6, beam_type=BeamType.ION)

    assert props.ion_beam is not None
    assert props.ion_beam.dwell_time == pytest.approx(2.0e-6)


def test_accumulate_property_creates_microscope_from_none():
    props = GlobalProperties.model_validate({})
    assert props.microscope is None

    props.accumulate_property("stage_position", StagePosition(x=100.0), beam_type=None)

    assert props.microscope is not None
    assert props.microscope.stage_position.x == pytest.approx(100.0)


def test_accumulate_property_adds_to_existing_numeric():
    props = GlobalProperties.model_validate({})
    props.accumulate_property("dwell_time", 1.0e-6, beam_type=BeamType.ELECTRON)
    props.accumulate_property("dwell_time", 2.0e-6, beam_type=BeamType.ELECTRON)
    props.accumulate_property("dwell_time", 1.5e-6, beam_type=BeamType.ELECTRON)

    assert props.electron_beam is not None
    assert props.electron_beam.dwell_time == pytest.approx(4.5e-6)


def test_accumulate_property_sets_when_attribute_none():
    props = GlobalProperties.model_validate({})
    props.electron_beam = BeamProperties.model_validate({})
    assert props.electron_beam.dwell_time is None

    props.accumulate_property("dwell_time", 1.0e-6, beam_type=BeamType.ELECTRON)
    assert props.electron_beam.dwell_time == pytest.approx(1.0e-6)

    props.accumulate_property("dwell_time", 0.5e-6, beam_type=BeamType.ELECTRON)
    assert props.electron_beam.dwell_time == pytest.approx(1.5e-6)


def test_accumulate_property_sets_stage_position_when_none():
    props = GlobalProperties.model_validate({})
    props.microscope = MicroscopeProperties.model_validate({})
    assert props.microscope.stage_position is None

    props.accumulate_property("stage_position", StagePosition(x=1000.0), beam_type=None)
    assert props.microscope.stage_position is not None
    assert props.microscope.stage_position.x == pytest.approx(1000.0)

    props.accumulate_property("stage_position", StagePosition(x=500.0), beam_type=None)
    assert props.microscope.stage_position is not None
    assert props.microscope.stage_position.x == pytest.approx(1500.0)


def test_accumulate_property_line_integration_multiple_calls():
    props = GlobalProperties.model_validate({})

    for _ in range(5):
        props.accumulate_property("line_integration", 4, beam_type=BeamType.ELECTRON)

    assert props.electron_beam is not None
    assert props.electron_beam.line_integration == 20


def test_accumulate_property_bit_depth_accumulates():
    props = GlobalProperties.model_validate({})
    props.accumulate_property("bit_depth", 8, beam_type=BeamType.ELECTRON)
    props.accumulate_property("bit_depth", 8, beam_type=BeamType.ELECTRON)

    assert props.electron_beam is not None
    assert props.electron_beam.bit_depth == 16


def test_accumulate_property():
    props = GlobalProperties.model_validate({})
    props.accumulate_property(
        "beam_shift", BeamShift(x=100.0, y=-20.0), beam_type=BeamType.ELECTRON
    )
    props.accumulate_property(
        "beam_shift", BeamShift(x=80.0, y=25.0), beam_type=BeamType.ELECTRON
    )

    assert props.electron_beam is not None
    assert props.electron_beam.beam_shift is not None
    assert props.electron_beam.beam_shift.x == pytest.approx(180.0)
    assert props.electron_beam.beam_shift.y == pytest.approx(5.0)


def test_accumulate_property_mixed_none_and_values():
    """Properties can be mixed: some None, some with values."""
    props = GlobalProperties.model_validate({})
    props.electron_beam = BeamProperties.model_validate(
        {"dwell_time": 1.0e-6, "bit_depth": None}
    )

    props.accumulate_property("dwell_time", 0.5e-6, beam_type=BeamType.ELECTRON)
    assert props.electron_beam.dwell_time == pytest.approx(1.5e-6)

    props.accumulate_property("bit_depth", 8, beam_type=BeamType.ELECTRON)
    assert props.electron_beam.bit_depth == 8


def test_set_property_creates_electron_beam_from_none():
    props = GlobalProperties.model_validate({})
    assert props.electron_beam is None

    props.set_property("dwell_time", 2.5e-6, beam_type=BeamType.ELECTRON)

    assert props.electron_beam is not None
    assert props.electron_beam.dwell_time == pytest.approx(2.5e-6)


def test_set_property_creates_ion_beam_from_none():
    props = GlobalProperties.model_validate({})
    assert props.ion_beam is None

    props.set_property("line_integration", 8, beam_type=BeamType.ION)

    assert props.ion_beam is not None
    assert props.ion_beam.line_integration == 8


def test_set_property_creates_microscope_from_none():
    props = GlobalProperties.model_validate({})
    assert props.microscope is None

    props.set_property("working_distance", 4800.0, beam_type=None)

    assert props.microscope is not None
    assert props.microscope.working_distance == pytest.approx(4800.0)


def test_set_property_replaces_existing_value():
    props = GlobalProperties.model_validate({})
    props.accumulate_property("dwell_time", 5.0e-6, beam_type=BeamType.ELECTRON)

    props.set_property("dwell_time", 3.0e-6, beam_type=BeamType.ELECTRON)

    assert props.electron_beam is not None
    assert props.electron_beam.dwell_time == pytest.approx(3.0e-6)


def test_set_property_multiple_updates():
    props = GlobalProperties.model_validate({})

    props.set_property("line_integration", 2, beam_type=BeamType.ELECTRON)
    assert props.electron_beam is not None
    assert props.electron_beam.line_integration == 2

    props.set_property("line_integration", 4, beam_type=BeamType.ELECTRON)
    assert props.electron_beam.line_integration == 4

    props.set_property("line_integration", 8, beam_type=BeamType.ELECTRON)
    assert props.electron_beam.line_integration == 8


def test_global_properties_get_property_names_returns_empty_when_all_none():
    props = GlobalProperties()

    result = props.get_property_names()

    assert result.microscope == []
    assert result.electron_beam == []
    assert result.ion_beam == []


def test_global_properties_get_property_names_returns_correct_microscope_names():
    props = GlobalProperties(
        microscope=MicroscopeProperties(stage_position=StagePosition(x=0.0, y=0.0))
    )

    result = props.get_property_names()

    assert "stage_position" in result.microscope
    assert result.electron_beam == []
    assert result.ion_beam == []


def test_global_properties_get_property_names_returns_correct_beam_names():
    props = GlobalProperties(
        electron_beam=BeamProperties(working_distance=5_000_000.0),
        ion_beam=BeamProperties(pixel_size=2.0),
    )

    result = props.get_property_names()

    assert "working_distance" in result.electron_beam
    assert "pixel_size" in result.ion_beam
    assert result.microscope == []


def test_global_properties_get_property_names_returns_property_names_instance():
    props = GlobalProperties()

    result = props.get_property_names()

    assert isinstance(result, PropertyNames)


def _electron_names(*names: str) -> PropertyNames:
    return PropertyNames(microscope=[], electron_beam=list(names), ion_beam=[])


def test_select_missing_property_raises() -> None:
    props = GlobalProperties()
    props.set_property("working_distance", 4.0e6, BeamType.ELECTRON)  # nm

    with pytest.raises(KeyError):
        props.select(_electron_names("working_distance", "dwell_time"))


def test_select_extra_property_is_selected() -> None:
    props = GlobalProperties()
    props.set_property("vendor_specific_gain", 1.5, BeamType.ELECTRON)

    selected = props.select(_electron_names("vendor_specific_gain"))

    assert selected.electron_beam is not None
    assert selected.electron_beam.vendor_specific_gain == 1.5  # ty: ignore[missing-attribute]


def _names(
    *,
    microscope: tuple[str, ...] = (),
    electron: tuple[str, ...] = (),
    ion: tuple[str, ...] = (),
) -> PropertyNames:
    return PropertyNames(
        microscope=list(microscope),
        electron_beam=list(electron),
        ion_beam=list(ion),
    )


@pytest.fixture
def props() -> GlobalProperties:
    """Electron and ion beam properties with a few declared fields set."""
    p = GlobalProperties()
    p.set_property("working_distance", 4.0e6, BeamType.ELECTRON)  # nm
    p.set_property("dwell_time", 100.0, BeamType.ELECTRON)  # ns
    p.set_property("working_distance", 4.5e6, BeamType.ION)  # nm
    return p


def test_select_keeps_only_requested_properties(props: GlobalProperties) -> None:
    selected = props.select(_names(electron=("working_distance",)))

    assert selected.electron_beam is not None
    assert selected.electron_beam.working_distance == 4.0e6
    assert selected.electron_beam.dwell_time is None


def test_select_beams_are_independent(props: GlobalProperties) -> None:
    selected = props.select(_names(electron=("dwell_time",), ion=("working_distance",)))

    assert selected.electron_beam is not None
    assert selected.ion_beam is not None
    assert selected.electron_beam.working_distance is None
    assert selected.electron_beam.dwell_time == 100.0
    assert selected.ion_beam.working_distance == 4.5e6


def test_select_empty_names_returns_empty_properties(props: GlobalProperties) -> None:
    selected = props.select(_names())

    assert selected.microscope is None
    assert selected.electron_beam is None
    assert selected.ion_beam is None


def test_select_from_unset_beam_raises() -> None:
    p = GlobalProperties()
    p.set_property("working_distance", 4.0e6, BeamType.ELECTRON)

    with pytest.raises(KeyError, match="ion_beam"):
        p.select(_names(ion=("working_distance",)))


def test_select_unknown_property_raises(props: GlobalProperties) -> None:
    with pytest.raises(KeyError, match="no_such_property"):
        props.select(_names(electron=("no_such_property",)))


def test_select_does_not_modify_source(props: GlobalProperties) -> None:
    props.select(_names(electron=("working_distance",)))

    assert props.electron_beam is not None
    assert props.electron_beam.working_distance == 4.0e6
    assert props.electron_beam.dwell_time == 100.0
    assert props.ion_beam is not None
    assert props.ion_beam.working_distance == 4.5e6


def test_select_result_shares_no_mutable_state_with_source() -> None:
    p = GlobalProperties()
    p.set_property("vendor_offsets", [1.0, 2.0], BeamType.ELECTRON)

    selected = p.select(_names(electron=("vendor_offsets",)))
    assert selected.electron_beam is not None
    selected.electron_beam.vendor_offsets.append(3.0)  # ty: ignore[missing-attribute]

    assert p.electron_beam is not None
    assert p.electron_beam.vendor_offsets == [1.0, 2.0]  # ty: ignore[missing-attribute]


def test_set_property_extra_on_existing_inner_object() -> None:
    p = GlobalProperties()
    p.set_property("working_distance", 4.0e6, BeamType.ELECTRON)  # nm
    p.set_property("vendor_specific_gain", 1.5, BeamType.ELECTRON)

    assert p.electron_beam is not None
    assert p.electron_beam.vendor_specific_gain == 1.5  # ty: ignore[missing-attribute]
    assert p.electron_beam.working_distance == 4.0e6


def test_accumulate_property_unset_extra_is_set() -> None:
    p = GlobalProperties()
    p.set_property("working_distance", 4.0e6, BeamType.ELECTRON)
    p.accumulate_property("vendor_offset", 2.0, BeamType.ELECTRON)

    assert p.electron_beam is not None
    assert p.electron_beam.vendor_offset == 2.0  # ty: ignore[missing-attribute]


def test_accumulate_property_existing_extra_is_added() -> None:
    p = GlobalProperties()
    p.set_property("vendor_offset", 2.0, BeamType.ELECTRON)
    p.accumulate_property("vendor_offset", 0.5, BeamType.ELECTRON)

    assert p.electron_beam is not None
    assert p.electron_beam.vendor_offset == 2.5  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("name", ["_private", "model_config", "get_property_names"])
def test_set_property_reserved_name_raises(name: str) -> None:
    p = GlobalProperties()

    with pytest.raises(ValueError, match=name):
        p.set_property(name, 1.0, BeamType.ELECTRON)

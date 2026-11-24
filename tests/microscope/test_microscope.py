# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.microscope.error import MicroscopeError
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.properties.microscope_properties import MicroscopeProperties

if TYPE_CHECKING:
    from fibsem_maestro.microscope.microscope import Microscope


def test_set_properties_leaves_its_argument_unchanged(
    mock_microscope: Microscope,
) -> None:
    props = GlobalProperties(
        electron_beam=BeamProperties(
            beam_shift=BeamShift(10.0, 5.0), working_distance=4e6
        )
    )
    before = props.model_dump()

    mock_microscope.set_properties(props, None)

    assert props.model_dump() == before


def test_set_properties_still_applies_beam_shift(mock_microscope: Microscope) -> None:
    props = GlobalProperties(
        electron_beam=BeamProperties(beam_shift=BeamShift(10.0, 5.0))
    )
    mock_microscope.set_properties(props, None)
    assert mock_microscope.electron_beam.beam_shift == BeamShift(10.0, 5.0)


def test_set_properties_applies_manufacturer_properties_of_the_microscope(
    mock_microscope: Microscope,
) -> None:
    (name, *_) = mock_microscope.control.manufacturer_prop_names
    props = GlobalProperties(microscope=MicroscopeProperties(**{name: 2.5}))

    mock_microscope.set_properties(props, None)

    assert mock_microscope.control.manufacturer_prop(name) == 2.5


def test_set_properties_rejects_an_unknown_beam_property(
    mock_microscope: Microscope,
) -> None:
    # a manufacturer property of a beam is set through the microscope's path to it
    props = GlobalProperties(
        electron_beam=BeamProperties(**{"scanning.dwell_time.value": 1e-7})
    )

    with pytest.raises(MicroscopeError):
        mock_microscope.set_properties(props, None)

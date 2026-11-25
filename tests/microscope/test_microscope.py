# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import dataclasses
from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.stage_position import StagePosition
from fibsem_maestro.microscope.error import MicroscopeError
from fibsem_maestro.microscope.positioning_outcome import PositioningOutcome
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.properties.microscope_properties import MicroscopeProperties

if TYPE_CHECKING:
    from fibsem_maestro.logging.text.memory import MemoryTextLogger
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
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


@dataclasses.dataclass
class FakeStage:
    """
    A stage that reports where it was sent, plus an error.

    Attributes:
        error_x: Error of the reported position along x, in nm.
        position: The position it reports, in nm.
        moves: Number of moves commanded, absolute or relative.
    """

    error_x: float = 0.0
    position: StagePosition = dataclasses.field(default_factory=StagePosition)
    moves: int = 0

    def go_to(self, target: StagePosition) -> StagePosition:
        self.moves += 1
        self.position = target + StagePosition(x=self.error_x)
        return self.position


@pytest.fixture
def stage(mock_microscope: Microscope, monkeypatch: pytest.MonkeyPatch) -> FakeStage:
    fake = FakeStage()
    control = mock_microscope.control
    monkeypatch.setattr(
        type(control), "stage_position", property(lambda _: fake.position)
    )
    monkeypatch.setattr(control, "try_set_stage_position", fake.go_to)
    monkeypatch.setattr(
        control, "try_move_stage_position", lambda d: fake.go_to(fake.position + d)
    )
    return fake


@pytest.fixture
def beam_shift_out_of_range(
    mock_microscope: Microscope, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make every non-zero beam shift fail, as one out of range does."""
    beam_cls = type(mock_microscope.electron_beam)
    original = beam_cls.beam_shift

    def setter(beam: BeamControl, value: BeamShift) -> None:
        if value != BeamShift(0.0, 0.0):
            raise MicroscopeError("out of range")
        assert original.fset is not None
        original.fset(beam, value)

    monkeypatch.setattr(beam_cls, "beam_shift", property(original.fget, setter))


def _warned(text_log: MemoryTextLogger) -> bool:
    return any(r.level == "warning" for rs in text_log.records.values() for r in rs)


@pytest.mark.parametrize(
    ("error_x", "outcome"),
    [
        (50.0, PositioningOutcome.STAGE_MOVED),
        (500.0, PositioningOutcome.STAGE_OFF_TARGET),
    ],
)
@pytest.mark.parametrize("relative", [False, True])
def test_stage_move_is_made_once_and_reports_whether_it_is_on_target(
    mock_microscope: Microscope,
    stage: FakeStage,
    text_log: MemoryTextLogger,
    relative: bool,
    error_x: float,
    outcome: PositioningOutcome,
) -> None:
    stage.error_x = error_x
    stage.position = StagePosition(x=5000.0, y=-3000.0)
    target = StagePosition(x=1000.0, y=2000.0)

    if relative:
        result = mock_microscope.move_stage_position_with_verification(target)
    else:
        result = mock_microscope.set_stage_position_with_verification(target)

    assert result is outcome
    assert stage.moves == 1
    assert _warned(text_log) is (outcome is PositioningOutcome.STAGE_OFF_TARGET)


def test_beam_shift_in_range_is_applied_by_the_beam(
    mock_microscope: Microscope, stage: FakeStage
) -> None:
    outcome = mock_microscope.add_beam_shift_with_verification(BeamShift(10.0, 5.0))

    assert outcome is PositioningOutcome.BEAM_SHIFTED
    assert stage.moves == 0


@pytest.mark.usefixtures("beam_shift_out_of_range")
@pytest.mark.parametrize(
    ("error_x", "outcome"),
    [
        (0.0, PositioningOutcome.STAGE_MOVED),
        (500.0, PositioningOutcome.STAGE_OFF_TARGET),
    ],
)
def test_beam_shift_out_of_range_reports_the_stage_move(
    mock_microscope: Microscope,
    stage: FakeStage,
    error_x: float,
    outcome: PositioningOutcome,
) -> None:
    stage.error_x = error_x

    result = mock_microscope.add_beam_shift_with_verification(BeamShift(1e5, 0.0))

    assert result is outcome
    assert stage.moves == 1
    assert mock_microscope.electron_beam.beam_shift == BeamShift(0.0, 0.0)


@pytest.mark.usefixtures("beam_shift_out_of_range")
def test_temporary_beam_shift_out_of_range_moves_the_stage_back(
    mock_microscope: Microscope, stage: FakeStage
) -> None:
    start = StagePosition(x=100.0, y=200.0)
    stage.position = start

    with mock_microscope.add_temporary_beam_shift(BeamShift(1e5, 0.0)):
        assert stage.position != start

    assert stage.position.to_xy() == pytest.approx(start.to_xy())

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.core.area import NMArea, RelativeArea
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.milling.error import MillingError
from fibsem_maestro.milling.milling import Milling, MillingState
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.action.action import Action

# The simulated microscope images 1024 x 768 px at 1.953125 nm/px, so the frame
# is 1500 nm high: an area 0.2 high holds exactly three 100 nm slices.
SLICE_NM = 100.0


def _area(height: float, y: float = 0.1) -> RelativeArea:
    return RelativeArea(origin=RelativePoint(0.1, y), width=0.5, height=height)


@pytest.fixture
def milling(add_action: Callable[..., Action]) -> Milling:
    action = add_action(
        "milling",
        "milling",
        milling_area=[_area(0.2)],
        pattern_type="rectangle",
        slice_distance=SLICE_NM,
    )
    action.prepare()
    assert isinstance(action, Milling)
    return action


def test_workflow_run_stops_after_last_slice(
    workflow: Workflow, add_action: Callable[..., Action]
) -> None:
    add_action("adjust_props", "adjust").prepare()
    workflow.run(last_slice=3)
    # slices 1, 2 and 3 were acquired; the counter points at the next one
    assert workflow.ctx.slice == 4


def test_workflow_run_never_passes_max_slice(
    workflow: Workflow,
    add_action: Callable[..., Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Workflow, "MAX_SLICE", 2)
    add_action("adjust_props", "adjust").prepare()
    workflow.run(last_slice=50)
    assert workflow.ctx.slice == 3


def test_workflow_run_returns_on_completion_reason(
    workflow: Workflow,
    add_action: Callable[..., Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = add_action("adjust_props", "adjust")
    action.prepare()
    monkeypatch.setattr(
        type(action),
        "completion_reason",
        lambda self: "done" if self.ctx.slice >= 3 else None,
    )
    workflow.run()
    assert workflow.ctx.slice == 3
    assert workflow.completion_reasons() == ["'adjust': done"]


USED_UP = "the milling area is used up"


@pytest.fixture
def milled(monkeypatch: pytest.MonkeyPatch, milling: Milling) -> list[NMArea]:
    """The slices the milling mills, in order (nm)."""
    slices: list[NMArea] = []

    def record(area: NMArea, *_: object) -> None:
        slices.append(area)

    monkeypatch.setattr(milling.microscope.ion_beam, "rectangle_milling", record)
    return slices


def _area_nm(milling: Milling, area: RelativeArea) -> NMArea:
    beam = milling.microscope.ion_beam
    return area.to_nanometers(beam.resolution, beam.pixel_size)


def test_milling_finishes_at_end_of_area_and_images_last_slice(
    workflow: Workflow, milling: Milling, add_action: Callable[..., Action]
) -> None:
    imaging = add_action("adjust_props", "after milling")
    imaging.prepare()
    workflow.run()
    # three slices fit; the action after milling also ran in the last one
    assert workflow.ctx.slice == 4
    assert imaging.ctx.slice == 4
    reason = milling.completion_reason()
    assert reason is not None and reason.startswith(USED_UP)


@pytest.mark.parametrize("direction", [Direction.DOWN, Direction.UP])
def test_milling_mills_slices_from_the_trailing_edge(
    workflow: Workflow, milling: Milling, milled: list[NMArea], direction: Direction
) -> None:
    milling.settings.milling_direction = direction
    area = _area_nm(milling, _area(0.2))

    workflow.run()

    tops = [s.origin.y for s in milled]
    if direction is Direction.DOWN:
        expected = [area.origin.y + k * SLICE_NM for k in range(3)]
    else:
        bottom = area.origin.y + area.height
        expected = [bottom - (k + 1) * SLICE_NM for k in range(3)]
    assert tops == pytest.approx(expected)
    assert all(s.height == SLICE_NM and s.width == area.width for s in milled)


def test_milling_never_mills_outside_area(workflow: Workflow, milling: Milling) -> None:
    workflow.run()
    # what "Retry action" would do after the end: it must refuse, not mill
    with pytest.raises(MillingError, match="used up"):
        milling._run_step()


def test_milling_rejects_stored_progress_beyond_area(milling: Milling) -> None:
    milling.set_state(MillingState(slices_milled=5))
    with pytest.raises(MillingError, match="used up"):
        milling._run_step()


def test_milling_next_slice_follows_a_moved_area(
    workflow: Workflow, milling: Milling, milled: list[NMArea]
) -> None:
    workflow.run(last_slice=1)
    moved = _area(0.2, y=0.4)
    milling.settings.milling_area = [moved]

    workflow.run(last_slice=2)

    # the second slice of the moved area, not of the old one
    assert milling.state.slices_milled == 2
    assert milled[-1].origin.y == pytest.approx(
        _area_nm(milling, moved).origin.y + SLICE_NM
    )


def test_milling_moved_area_of_the_same_size_keeps_the_end(
    workflow: Workflow, milling: Milling
) -> None:
    workflow.run()
    milling.settings.milling_area = [_area(0.2, y=0.4)]

    assert milling.completion_reason() is not None
    assert workflow.completion_reasons() != []


def test_milling_enlarged_area_lifts_the_end(
    workflow: Workflow, milling: Milling, milled: list[NMArea]
) -> None:
    workflow.run()
    enlarged = _area(0.4)
    milling.settings.milling_area = [enlarged]
    assert milling.completion_reason() is None

    workflow.run()

    # milling went on from where it stopped, to the end of the enlarged area
    fitting = int(_area_nm(milling, enlarged).height // SLICE_NM)
    assert fitting > 3
    assert len(milled) == fitting
    assert milled[3].origin.y == pytest.approx(
        _area_nm(milling, enlarged).origin.y + 3 * SLICE_NM
    )
    assert milling.completion_reason() is not None


def test_milling_completion_reason_does_not_use_the_microscope(
    workflow: Workflow, milling: Milling, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow.run()

    def unavailable(_: object) -> float:
        raise AssertionError("completion_reason must not use the microscope")

    monkeypatch.setattr(
        type(milling.microscope.ion_beam), "pixel_size", property(unavailable)
    )
    assert milling.completion_reason() is not None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("slice_distance", 50.0),
        ("milling_direction", Direction.UP),
    ],
)
def test_milling_slice_geometry_change_restarts_at_the_trailing_edge(
    workflow: Workflow, milling: Milling, name: str, value: object
) -> None:
    workflow.run()
    setattr(milling.settings, name, value)
    assert milling.state == MillingState()
    assert milling.completion_reason() is None


def test_milling_area_change_keeps_the_progress(
    workflow: Workflow, milling: Milling
) -> None:
    workflow.run(last_slice=2)
    milling.settings.milling_area = [_area(0.3, y=0.2)]
    assert milling.state.slices_milled == 2


def test_milling_reset_restarts_at_the_trailing_edge(
    workflow: Workflow, milling: Milling
) -> None:
    workflow.run()
    milling.reset_state()
    assert milling.completion_reason() is None
    assert milling.state == MillingState()


def test_milling_completion_reason_without_area_is_none(
    workflow: Workflow, milling: Milling, add_action: Callable[..., Action]
) -> None:
    workflow.run()
    action = add_action("milling", "new milling")
    assert isinstance(action, Milling)
    action.set_state(milling.state)
    assert action.completion_reason() is None


def test_milling_progress_survives_resume(workflow: Workflow, milling: Milling) -> None:
    workflow.run()
    # the state written for the action's current slice is what a resume reads
    stored = milling.ctx.state_store.read("state.yaml", MillingState)
    assert stored == milling.state
    assert stored.slices_milled == 3

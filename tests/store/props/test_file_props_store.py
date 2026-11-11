# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.core.stage_position import StagePosition
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.properties.microscope_properties import MicroscopeProperties
from fibsem_maestro.slice.slice_view import SliceView
from fibsem_maestro.store.props.file import FilePropsStore

if TYPE_CHECKING:
    from pathlib import Path


class SliceCursor:
    """Mutable stand-in for the run loop's slice pointer; `SliceView` is immutable."""

    def __init__(self, action_dir: Path, slice_index: int = 0) -> None:
        self._action_dir = action_dir
        self._view = SliceView(action_dir, slice_index)

    def __call__(self) -> SliceView:
        return self._view

    def advance_to(self, slice_index: int) -> None:
        self._view = SliceView(self._action_dir, slice_index)

    def dir(self, slice_index: int) -> Path:
        return SliceView(self._action_dir, slice_index).expected_path

    @property
    def action_dir(self) -> Path:
        return self._action_dir


@pytest.fixture
def cursor(tmp_path: Path) -> SliceCursor:
    return SliceCursor(tmp_path)


@pytest.fixture
def store(cursor: SliceCursor) -> FilePropsStore:
    return FilePropsStore(cursor)


@pytest.fixture
def props() -> GlobalProperties:
    return GlobalProperties(
        microscope=MicroscopeProperties(
            stage_position=StagePosition(
                x=1.0e6, y=-2.0e6, z=5.0e5, rotation=90.0, tilt=38.0
            )
        ),
        electron_beam=BeamProperties(
            working_distance=4.2e6,
            dwell_time=100.0,
            beam_shift=BeamShift(x=-3.0, y=4.0),
            resolution=Resolution(width=1536, height=1024),
        ),
    )


def test_write_creates_the_file(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    assert (cursor.dir(0) / "props.yaml").is_file()


def test_write_creates_the_slice_directory(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    assert not cursor.dir(0).exists()

    store.write("props.yaml", props)

    assert cursor.dir(0).is_dir()


def test_round_trip_preserves_the_whole_tree(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    assert store.read("props.yaml") == props


def test_round_trip_preserves_nested_stage_position(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    props = store.read("props.yaml")
    assert props.microscope is not None
    stage = props.microscope.stage_position
    assert stage is not None
    assert stage.x == pytest.approx(1.0e6)
    assert stage.y == pytest.approx(-2.0e6)
    assert stage.tilt == pytest.approx(38.0)


def test_round_trip_preserves_nested_beam_shift(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    props = store.read("props.yaml")
    assert props.electron_beam is not None
    shift = props.electron_beam.beam_shift
    assert shift is not None
    assert shift.x == pytest.approx(-3.0)
    assert shift.y == pytest.approx(4.0)


def test_round_trip_preserves_integer_resolution(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    props = store.read("props.yaml")
    assert props.electron_beam is not None
    resolution = props.electron_beam.resolution
    assert resolution is not None
    assert (resolution.width, resolution.height) == (1536, 1024)


def test_round_trip_preserves_float_precision(store: FilePropsStore) -> None:
    props = GlobalProperties(electron_beam=BeamProperties(working_distance=4123456.789))

    store.write("props.yaml", props)

    props = store.read("props.yaml")
    assert props.electron_beam is not None
    wd = props.electron_beam.working_distance

    assert wd == pytest.approx(4123456.789)


def test_round_trip_keeps_absent_beams_none(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    assert store.read("props.yaml").ion_beam is None


def test_round_trip_keeps_unset_properties_none(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    props = store.read("props.yaml")
    assert props.electron_beam is not None
    assert props.electron_beam.stigmator_x is None


def test_round_trip_keeps_the_two_beams_distinct(store: FilePropsStore) -> None:
    props = GlobalProperties(
        electron_beam=BeamProperties(beam_current=1.5),
        ion_beam=BeamProperties(beam_current=30.0),
    )

    store.write("props.yaml", props)

    loaded = store.read("props.yaml")
    assert loaded.electron_beam is not None
    assert loaded.ion_beam is not None
    assert loaded.electron_beam.beam_current == pytest.approx(1.5)
    assert loaded.ion_beam.beam_current == pytest.approx(30.0)


def test_round_trip_preserves_dynamically_added_properties(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    """Manufacturer properties are added at runtime under `extra="allow"`."""
    assert props.electron_beam is not None
    props.electron_beam.custom_property = 12.5

    store.write("props.yaml", props)

    props = store.read("props.yaml")
    assert props.electron_beam is not None

    assert props.electron_beam.custom_property == pytest.approx(12.5)  # ty: ignore[missing-attribute]


def test_round_trip_of_empty_properties(store: FilePropsStore) -> None:
    store.write("props.yaml", GlobalProperties())

    assert store.read("props.yaml") == GlobalProperties()


def test_write_overwrites_an_existing_file(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)
    store.write(
        "props.yaml",
        GlobalProperties(electron_beam=BeamProperties(working_distance=9.0)),
    )

    props = store.read("props.yaml")
    assert props.electron_beam is not None
    assert props.electron_beam.working_distance == pytest.approx(9.0)


def test_write_does_not_merge_with_the_previous_contents(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)
    store.write("props.yaml", GlobalProperties())

    assert store.read("props.yaml").microscope is None


def test_write_follows_the_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)
    cursor.advance_to(1)
    store.write(
        "props.yaml",
        GlobalProperties(electron_beam=BeamProperties(working_distance=9.0)),
    )

    props1 = store.at(0).read("props.yaml")
    props2 = store.at(1).read("props.yaml")
    assert props1.electron_beam is not None
    assert props2.electron_beam is not None
    assert props1.electron_beam.working_distance == pytest.approx(4.2e6)
    assert props2.electron_beam.working_distance == pytest.approx(9.0)


def test_distinct_filenames_stay_apart(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("before.yaml", props)
    store.write("after.yaml", GlobalProperties())

    assert store.read("before.yaml").microscope is not None
    assert store.read("after.yaml").microscope is None


def test_read_raises_when_the_file_is_missing(store: FilePropsStore) -> None:
    with pytest.raises(FileNotFoundError, match="No props file found"):
        store.read("props.yaml")


def test_read_error_names_the_expected_path(store: FilePropsStore) -> None:
    with pytest.raises(FileNotFoundError, match="props.yaml"):
        store.read("props.yaml")


def test_read_is_per_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)
    cursor.advance_to(1)

    with pytest.raises(FileNotFoundError):
        store.read("props.yaml")


def test_read_does_not_create_the_slice_directory(
    store: FilePropsStore, cursor: SliceCursor
) -> None:
    with pytest.raises(FileNotFoundError):
        store.read("props.yaml")

    assert not cursor.dir(0).exists()


def test_read_does_not_consume_the_file(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    store.read("props.yaml")

    assert store.exists("props.yaml") is True


def test_exists_is_false_before_writing(store: FilePropsStore) -> None:
    assert store.exists("props.yaml") is False


def test_exists_is_true_after_writing(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    assert store.exists("props.yaml") is True


def test_exists_does_not_create_the_slice_directory(
    store: FilePropsStore, cursor: SliceCursor
) -> None:
    store.exists("props.yaml")

    assert not cursor.dir(0).exists()


def test_exists_is_per_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)
    cursor.advance_to(1)

    assert store.exists("props.yaml") is False


def test_copy_to_places_the_file_in_the_target_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    store.copy_to("props.yaml", store.at(3))

    assert (cursor.dir(3) / "props.yaml").is_file()


def test_copy_to_preserves_the_contents(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    store.copy_to("props.yaml", store.at(3))

    assert store.at(3).read("props.yaml") == props


def test_copy_to_leaves_the_source_in_place(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    store.copy_to("props.yaml", store.at(3))

    assert store.exists("props.yaml") is True


def test_copy_to_creates_the_target_slice_directory(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)
    assert not cursor.dir(3).exists()

    store.copy_to("props.yaml", store.at(3))

    assert cursor.dir(3).is_dir()


def test_copy_to_overwrites_an_existing_target(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    target = store.at(3)
    target.write("props.yaml", GlobalProperties())
    store.write("props.yaml", props)

    store.copy_to("props.yaml", target)

    assert target.read("props.yaml") == props


def test_copy_to_raises_when_the_source_is_missing(store: FilePropsStore) -> None:
    with pytest.raises(FileNotFoundError, match="No props file found"):
        store.copy_to("props.yaml", store.at(3))


def test_copy_to_the_same_slice_is_a_no_op(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    store.copy_to("props.yaml", store)

    assert store.read("props.yaml") == props


def test_copy_to_an_equivalent_view_is_a_no_op(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.write("props.yaml", props)

    store.copy_to("props.yaml", store.at(0))

    assert store.read("props.yaml") == props


def test_copy_to_a_store_in_another_action(
    store: FilePropsStore, tmp_path: Path, props: GlobalProperties
) -> None:
    other = FilePropsStore(SliceCursor(tmp_path / "other"))
    store.write("props.yaml", props)

    store.copy_to("props.yaml", other)

    assert other.read("props.yaml") == props


def test_slice_reports_the_current_index(
    store: FilePropsStore, cursor: SliceCursor
) -> None:
    assert store.slice == 0

    cursor.advance_to(7)

    assert store.slice == 7


def test_at_addresses_the_requested_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.at(5).write("props.yaml", props)

    assert (cursor.dir(5) / "props.yaml").is_file()


def test_at_reports_the_requested_slice(store: FilePropsStore) -> None:
    assert store.at(5).slice == 5


def test_at_does_not_move_the_original_store(
    store: FilePropsStore, props: GlobalProperties
) -> None:
    store.at(5).write("props.yaml", props)

    assert store.slice == 0
    assert store.exists("props.yaml") is False


def test_at_is_pinned_when_the_run_advances(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    pinned = store.at(3)

    cursor.advance_to(9)
    pinned.write("props.yaml", props)

    assert (cursor.dir(3) / "props.yaml").is_file()


def test_at_rejects_a_negative_slice(store: FilePropsStore) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        store.at(-1)


def test_next_addresses_the_following_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.next.write("props.yaml", props)

    assert (cursor.dir(1) / "props.yaml").is_file()


def test_next_is_relative_to_the_current_slice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    cursor.advance_to(4)

    store.next.write("props.yaml", props)

    assert (cursor.dir(5) / "props.yaml").is_file()


def test_next_reports_the_following_slice(
    store: FilePropsStore, cursor: SliceCursor
) -> None:
    cursor.advance_to(2)

    assert store.next.slice == 3


def test_next_is_pinned_at_access_time(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    view = store.next

    cursor.advance_to(9)
    view.write("props.yaml", props)

    assert (cursor.dir(1) / "props.yaml").is_file()


def test_next_of_next_advances_twice(
    store: FilePropsStore, cursor: SliceCursor, props: GlobalProperties
) -> None:
    store.next.next.write("props.yaml", props)

    assert (cursor.dir(2) / "props.yaml").is_file()

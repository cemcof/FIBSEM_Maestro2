# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from pathlib import Path
from typing import cast

import pytest
from pydantic import ConfigDict, Field, ValidationError

from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.slice.slice_view import SliceView
from fibsem_maestro.store.settings.file import FileSettingsStore


class SampleSettings(BaseSettings):
    """Stand-in for an action's settings model."""

    dwell_time_ns: int = 100
    beam_type: BeamType = BeamType.ELECTRON
    sharpness_limit: float | None = None
    output_dir: Path = Path("/data/scan")
    labels: list[str] = Field(default_factory=list)


class StrictOtherSettings(BaseSettings):
    """A different settings model, to check `read` honours the requested class."""

    model_config = ConfigDict(extra="forbid")

    passes: int = 1


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
def store(cursor: SliceCursor) -> FileSettingsStore:
    return FileSettingsStore(cursor)


@pytest.fixture
def settings() -> SampleSettings:
    return SampleSettings(
        dwell_time_ns=300,
        beam_type=BeamType.ION,
        sharpness_limit=0.8,
        output_dir=Path("/data/run7"),
        labels=["autofunction", "slice"],
    )


def test_write_creates_the_file(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert (cursor.dir(0) / "settings.yaml").is_file()


def test_write_creates_the_slice_directory(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    assert not cursor.dir(0).exists()

    store.write("settings.yaml", settings)

    assert cursor.dir(0).is_dir()


def test_write_emits_plain_yaml_scalars(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    """No python-specific tags: the file stays hand-editable and safe-loadable."""
    store.write("settings.yaml", settings)

    contents = (cursor.dir(0) / "settings.yaml").read_text()
    assert "!!python" not in contents
    assert "beam_type: ion" in contents


def test_write_overwrites_an_existing_file(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    store.write("settings.yaml", SampleSettings(dwell_time_ns=999))

    assert (
        cast(
            "SampleSettings", store.read("settings.yaml", SampleSettings)
        ).dwell_time_ns
        == 999
    )


def test_write_does_not_merge_with_the_previous_contents(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    store.write("settings.yaml", SampleSettings())

    loaded = cast("SampleSettings", store.read("settings.yaml", SampleSettings))
    assert loaded.sharpness_limit is None
    assert loaded.labels == []


def test_write_follows_the_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    cursor.advance_to(1)
    store.write("settings.yaml", SampleSettings(dwell_time_ns=999))

    assert (
        cast(
            "SampleSettings", store.at(0).read("settings.yaml", SampleSettings)
        ).dwell_time_ns
        == 300
    )
    assert (
        cast(
            "SampleSettings", store.at(1).read("settings.yaml", SampleSettings)
        ).dwell_time_ns
        == 999
    )


def test_distinct_filenames_stay_apart(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("before.yaml", settings)
    store.write("after.yaml", SampleSettings(dwell_time_ns=999))

    assert (
        cast("SampleSettings", store.read("before.yaml", SampleSettings)).dwell_time_ns
        == 300
    )
    assert (
        cast("SampleSettings", store.read("after.yaml", SampleSettings)).dwell_time_ns
        == 999
    )


def test_round_trip_preserves_every_field(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert store.read("settings.yaml", SampleSettings) == settings


def test_round_trip_preserves_an_enum_field(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert (
        cast("SampleSettings", store.read("settings.yaml", SampleSettings)).beam_type
        is BeamType.ION
    )


def test_round_trip_preserves_a_path_field(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert cast(
        "SampleSettings", store.read("settings.yaml", SampleSettings)
    ).output_dir == Path("/data/run7")


def test_round_trip_preserves_a_list_field(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert cast(
        "SampleSettings", store.read("settings.yaml", SampleSettings)
    ).labels == ["autofunction", "slice"]


def test_round_trip_preserves_an_unset_optional_field(
    store: FileSettingsStore,
) -> None:
    store.write("settings.yaml", SampleSettings(sharpness_limit=None))

    assert (
        cast(
            "SampleSettings", store.read("settings.yaml", SampleSettings)
        ).sharpness_limit
        is None
    )


def test_round_trip_preserves_float_precision(store: FileSettingsStore) -> None:
    store.write("settings.yaml", SampleSettings(sharpness_limit=0.123456789))

    loaded = cast("SampleSettings", store.read("settings.yaml", SampleSettings))

    assert loaded.sharpness_limit == pytest.approx(0.123456789)


def test_round_trip_preserves_integer_type(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert isinstance(
        cast(
            "SampleSettings", store.read("settings.yaml", SampleSettings)
        ).dwell_time_ns,
        int,
    )


def test_round_trip_of_defaults(store: FileSettingsStore) -> None:
    store.write("settings.yaml", SampleSettings())

    assert store.read("settings.yaml", SampleSettings) == SampleSettings()


def test_read_uses_the_requested_class(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert type(store.read("settings.yaml", SampleSettings)) is SampleSettings


def test_read_with_a_mismatched_class_raises(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    """A wrong `cls` is a loud error, not a silent partial load."""
    store.write("settings.yaml", settings)

    with pytest.raises(ValidationError):
        store.read("settings.yaml", StrictOtherSettings)


def test_read_raises_when_the_file_is_missing(store: FileSettingsStore) -> None:
    with pytest.raises(FileNotFoundError, match="No settings file found"):
        store.read("settings.yaml", SampleSettings)


def test_read_error_names_the_expected_path(store: FileSettingsStore) -> None:
    with pytest.raises(FileNotFoundError, match="settings.yaml"):
        store.read("settings.yaml", SampleSettings)


def test_read_is_per_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    cursor.advance_to(1)

    with pytest.raises(FileNotFoundError):
        store.read("settings.yaml", SampleSettings)


def test_read_does_not_create_the_slice_directory(
    store: FileSettingsStore, cursor: SliceCursor
) -> None:
    with pytest.raises(FileNotFoundError):
        store.read("settings.yaml", SampleSettings)

    assert not cursor.dir(0).exists()


def test_read_does_not_consume_the_file(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    store.read("settings.yaml", SampleSettings)

    assert store.exists("settings.yaml") is True


def test_read_can_be_repeated(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert store.read("settings.yaml", SampleSettings) == store.read(
        "settings.yaml", SampleSettings
    )


def test_read_returns_independent_instances(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    first = cast("SampleSettings", store.read("settings.yaml", SampleSettings))
    second = cast("SampleSettings", store.read("settings.yaml", SampleSettings))
    first.dwell_time_ns = 1

    assert second.dwell_time_ns == 300


def test_exists_is_false_before_writing(store: FileSettingsStore) -> None:
    assert store.exists("settings.yaml") is False


def test_exists_is_true_after_writing(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert store.exists("settings.yaml") is True


def test_exists_does_not_create_the_slice_directory(
    store: FileSettingsStore, cursor: SliceCursor
) -> None:
    store.exists("settings.yaml")

    assert not cursor.dir(0).exists()


def test_exists_is_per_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    cursor.advance_to(1)

    assert store.exists("settings.yaml") is False


def test_exists_is_per_filename(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert store.exists("other.yaml") is False


def test_copy_to_places_the_file_in_the_target_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", store.at(3))

    assert (cursor.dir(3) / "settings.yaml").is_file()


def test_copy_to_preserves_the_contents(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", store.at(3))

    assert store.at(3).read("settings.yaml", SampleSettings) == settings


def test_copy_to_leaves_the_source_in_place(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", store.at(3))

    assert store.exists("settings.yaml") is True


def test_copy_to_creates_the_target_slice_directory(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    assert not cursor.dir(3).exists()

    store.copy_to("settings.yaml", store.at(3))

    assert cursor.dir(3).is_dir()


def test_copy_to_overwrites_an_existing_target(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    target = store.at(3)
    target.write("settings.yaml", SampleSettings(dwell_time_ns=999))
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", target)

    assert target.read("settings.yaml", SampleSettings) == settings


def test_copy_to_raises_when_the_source_is_missing(store: FileSettingsStore) -> None:
    with pytest.raises(FileNotFoundError, match="No settings file found"):
        store.copy_to("settings.yaml", store.at(3))


def test_copy_to_does_not_create_the_target_when_the_source_is_missing(
    store: FileSettingsStore, cursor: SliceCursor
) -> None:
    with pytest.raises(FileNotFoundError):
        store.copy_to("settings.yaml", store.at(3))

    assert not cursor.dir(3).exists()


def test_copy_to_the_same_slice_is_a_no_op(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", store)

    assert store.read("settings.yaml", SampleSettings) == settings


def test_copy_to_an_equivalent_view_is_a_no_op(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", store.at(0))

    assert store.read("settings.yaml", SampleSettings) == settings


def test_copy_to_a_store_in_another_action(
    store: FileSettingsStore, tmp_path: Path, settings: SampleSettings
) -> None:
    other = FileSettingsStore(SliceCursor(tmp_path / "other"))
    store.write("settings.yaml", settings)

    store.copy_to("settings.yaml", other)

    assert other.read("settings.yaml", SampleSettings) == settings


def test_copy_to_is_independent_of_the_source_afterwards(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)
    target = store.at(3)
    store.copy_to("settings.yaml", target)

    store.write("settings.yaml", SampleSettings(dwell_time_ns=999))

    assert (
        cast(
            "SampleSettings", target.read("settings.yaml", SampleSettings)
        ).dwell_time_ns
        == 300
    )


def test_slice_reports_the_current_index(
    store: FileSettingsStore, cursor: SliceCursor
) -> None:
    assert store.slice == 0

    cursor.advance_to(7)

    assert store.slice == 7


def test_at_addresses_the_requested_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.at(5).write("settings.yaml", settings)

    assert (cursor.dir(5) / "settings.yaml").is_file()


def test_at_reports_the_requested_slice(store: FileSettingsStore) -> None:
    assert store.at(5).slice == 5


def test_at_does_not_move_the_original_store(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.at(5).write("settings.yaml", settings)

    assert store.slice == 0
    assert store.exists("settings.yaml") is False


def test_at_is_pinned_when_the_run_advances(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    pinned = store.at(3)

    cursor.advance_to(9)
    pinned.write("settings.yaml", settings)

    assert (cursor.dir(3) / "settings.yaml").is_file()


def test_at_reads_back_what_it_wrote(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.at(5).write("settings.yaml", settings)

    assert store.at(5).read("settings.yaml", SampleSettings) == settings


def test_at_rejects_a_negative_slice(store: FileSettingsStore) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        store.at(-1)


def test_at_accepts_the_current_slice(
    store: FileSettingsStore, settings: SampleSettings
) -> None:
    store.write("settings.yaml", settings)

    assert store.at(0).exists("settings.yaml") is True


def test_next_addresses_the_following_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.next.write("settings.yaml", settings)

    assert (cursor.dir(1) / "settings.yaml").is_file()


def test_next_is_relative_to_the_current_slice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    cursor.advance_to(4)

    store.next.write("settings.yaml", settings)

    assert (cursor.dir(5) / "settings.yaml").is_file()


def test_next_reports_the_following_slice(
    store: FileSettingsStore, cursor: SliceCursor
) -> None:
    cursor.advance_to(2)

    assert store.next.slice == 3


def test_next_is_pinned_at_access_time(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    view = store.next

    cursor.advance_to(9)
    view.write("settings.yaml", settings)

    assert (cursor.dir(1) / "settings.yaml").is_file()


def test_next_of_next_advances_twice(
    store: FileSettingsStore, cursor: SliceCursor, settings: SampleSettings
) -> None:
    store.next.next.write("settings.yaml", settings)

    assert (cursor.dir(2) / "settings.yaml").is_file()

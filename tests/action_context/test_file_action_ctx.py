# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fibsem_maestro.action_context.file import FileActionContext
from fibsem_maestro.core.image import Image, Image8Bit
from fibsem_maestro.logging.image.file import FileImageLogger
from fibsem_maestro.logging.text.file import FileTextLogger, close_all_log_files
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.store.frame.file import FileFrameStore
from fibsem_maestro.store.image.file import FileImageStore
from fibsem_maestro.store.props.file import FilePropsStore
from fibsem_maestro.store.settings.file import FileSettingsStore
from fibsem_maestro.store.state.file import FileStateStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def close_handlers():
    """Log handlers outlive the context; release them so tmp dirs can be removed."""
    yield
    close_all_log_files()


@pytest.fixture
def action_dir(tmp_path: Path) -> Path:
    return tmp_path / "milling"


@pytest.fixture
def context(action_dir: Path) -> FileActionContext:
    return FileActionContext(action_dir, "milling", log_level=logging.DEBUG)


def slice_dir(action_dir: Path, index: int) -> Path:
    return action_dir / f"slice_{index:04d}"


def test_starts_at_slice_zero_by_default(context: FileActionContext) -> None:
    assert context.slice == 0


def test_starts_at_an_explicit_slice(action_dir: Path) -> None:
    assert FileActionContext(action_dir, "milling", slice=7).slice == 7


def test_current_view_matches_the_initial_slice(action_dir: Path) -> None:
    context = FileActionContext(action_dir, "milling", slice=7)

    assert context.current_view.slice_index == 7


def test_construction_builds_a_view_for_the_starting_slice(
    context: FileActionContext, action_dir: Path
) -> None:
    assert context.current_view.action_dir == action_dir
    assert context.current_view.slice_index == 0


def test_construction_rejects_a_negative_slice(action_dir: Path) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        FileActionContext(action_dir, "milling", slice=-1)


def test_construction_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    target = tmp_path / "milling"

    FileActionContext(target, "milling")

    assert not target.exists()


def test_advance_increments_the_slice(context: FileActionContext) -> None:
    context.advance()

    assert context.slice == 1


def test_advance_returns_the_new_view(context: FileActionContext) -> None:
    assert context.advance().slice_index == 1


def test_advance_updates_the_current_view(context: FileActionContext) -> None:
    context.advance()

    assert context.current_view.slice_index == 1


def test_advance_returns_the_same_object_as_current_view(
    context: FileActionContext,
) -> None:
    assert context.advance() is context.current_view


def test_repeated_advance_counts_up(context: FileActionContext) -> None:
    assert [context.advance().slice_index for _ in range(3)] == [1, 2, 3]


def test_advance_from_an_explicit_start(action_dir: Path) -> None:
    context = FileActionContext(action_dir, "milling", slice=5)

    assert context.advance().slice_index == 6


def test_advance_builds_a_new_view_each_time(context: FileActionContext) -> None:
    first = context.current_view
    second = context.advance()
    third = context.advance()

    assert first is not second
    assert second is not third
    assert [v.slice_index for v in (first, second, third)] == [0, 1, 2]


def test_current_view_is_stable_between_advances(context: FileActionContext) -> None:
    assert context.current_view is context.current_view


def test_set_slice_moves_to_the_given_index(context: FileActionContext) -> None:
    context.set_slice(9)

    assert context.slice == 9


def test_set_slice_returns_the_new_view(context: FileActionContext) -> None:
    assert context.set_slice(9).slice_index == 9


def test_set_slice_updates_the_current_view(context: FileActionContext) -> None:
    context.set_slice(9)

    assert context.current_view.slice_index == 9


def test_set_slice_can_move_backwards(context: FileActionContext) -> None:
    context.set_slice(9)
    context.set_slice(2)

    assert context.slice == 2


def test_set_slice_to_the_current_index_is_allowed(
    context: FileActionContext,
) -> None:
    context.set_slice(0)

    assert context.slice == 0


def test_advance_continues_from_the_set_slice(context: FileActionContext) -> None:
    context.set_slice(9)

    assert context.advance().slice_index == 10


def test_set_slice_rejects_a_negative_index(context: FileActionContext) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        context.set_slice(-1)


def test_rejection_names_the_offending_index(context: FileActionContext) -> None:
    with pytest.raises(ValueError, match="got -5"):
        context.set_slice(-5)


def test_a_rejected_set_slice_leaves_the_context_unchanged(
    context: FileActionContext,
) -> None:
    context.advance()

    with pytest.raises(ValueError):
        context.set_slice(-1)

    assert context.slice == 1
    assert context.current_view.slice_index == 1


def test_reset_returns_to_slice_zero(context: FileActionContext) -> None:
    context.advance()
    context.advance()

    context.reset()

    assert context.slice == 0


def test_reset_returns_the_new_view(context: FileActionContext) -> None:
    context.advance()

    assert context.reset().slice_index == 0


def test_reset_updates_the_current_view(context: FileActionContext) -> None:
    context.advance()

    context.reset()

    assert context.current_view.slice_index == 0


def test_advance_after_reset_starts_again_from_one(
    context: FileActionContext,
) -> None:
    context.advance()
    context.advance()
    context.reset()

    assert context.advance().slice_index == 1


def test_reset_from_an_explicit_start_goes_to_zero(action_dir: Path) -> None:
    context = FileActionContext(action_dir, "milling", slice=7)

    context.reset()

    assert context.slice == 0


def test_reset_is_idempotent(context: FileActionContext) -> None:
    context.advance()

    context.reset()
    context.reset()

    assert context.slice == 0


def test_contexts_do_not_share_a_counter(tmp_path: Path) -> None:
    first = FileActionContext(tmp_path / "milling", "milling")
    second = FileActionContext(tmp_path / "imaging", "imaging")

    first.advance()
    first.advance()

    assert first.slice == 2
    assert second.slice == 0


def test_views_are_built_for_this_contexts_directory(tmp_path: Path) -> None:
    context = FileActionContext(tmp_path / "milling", "milling")

    assert context.current_view.action_dir == tmp_path / "milling"


def test_path_to_dir_is_the_action_directory(
    context: FileActionContext, action_dir: Path
) -> None:
    assert context.path_to_dir == action_dir


def test_construction_does_not_create_the_action_directory(action_dir: Path) -> None:
    FileActionContext(action_dir, "milling")

    assert not action_dir.exists()


def test_text_logger_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.text_logger, FileTextLogger)


def test_image_logger_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.image_logger, FileImageLogger)


def test_props_store_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.props_store, FilePropsStore)


def test_state_store_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.state_store, FileStateStore)


def test_settings_store_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.settings_store, FileSettingsStore)


def test_frame_store_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.frame_store, FileFrameStore)


def test_stores_are_the_same_object_on_every_access(
    context: FileActionContext,
) -> None:
    assert context.props_store is context.props_store
    assert context.frame_store is context.frame_store
    assert context.text_logger is context.text_logger


def test_text_logger_writes_into_the_current_slice(
    context: FileActionContext, action_dir: Path
) -> None:
    context.text_logger.info("started")
    close_all_log_files()

    assert (slice_dir(action_dir, 0) / "run.log").is_file()


def test_text_logger_follows_an_advance(
    context: FileActionContext, action_dir: Path
) -> None:
    context.text_logger.info("slice zero")
    context.advance()
    context.text_logger.info("slice one")
    close_all_log_files()

    assert "slice zero" in (slice_dir(action_dir, 0) / "run.log").read_text()
    assert "slice one" in (slice_dir(action_dir, 1) / "run.log").read_text()


def test_text_logger_uses_the_configured_name(action_dir: Path) -> None:
    context = FileActionContext(action_dir, "autofunction", log_level=logging.DEBUG)

    context.text_logger.info("x")
    close_all_log_files()

    assert "[autofunction]" in (slice_dir(action_dir, 0) / "run.log").read_text()


def test_text_logger_uses_the_configured_filename(action_dir: Path) -> None:
    context = FileActionContext(
        action_dir, "milling", log_filename="action.log", log_level=logging.DEBUG
    )

    context.text_logger.info("x")
    close_all_log_files()

    assert (slice_dir(action_dir, 0) / "action.log").is_file()


def test_text_logger_honours_the_configured_level(action_dir: Path) -> None:
    context = FileActionContext(action_dir, "milling", log_level=logging.WARNING)

    context.text_logger.debug("suppressed")

    assert not (slice_dir(action_dir, 0) / "run.log").exists()


def test_image_logger_writes_into_the_current_slice(
    context: FileActionContext, action_dir: Path
) -> None:
    context.image_logger.save_image("drift.png", np.ones((8, 8)))

    assert (slice_dir(action_dir, 0) / "drift.png").is_file()


def test_image_logger_follows_an_advance(
    context: FileActionContext, action_dir: Path
) -> None:
    context.advance()

    context.image_logger.save_image("drift.png", np.ones((8, 8)))

    assert (slice_dir(action_dir, 1) / "drift.png").is_file()


def test_props_store_writes_into_the_current_slice(
    context: FileActionContext, action_dir: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())

    assert (slice_dir(action_dir, 0) / "props.yaml").is_file()


def test_props_store_follows_an_advance(
    context: FileActionContext, action_dir: Path
) -> None:
    props = GlobalProperties(electron_beam=BeamProperties(working_distance=4.2e6))
    context.props_store.write("props.yaml", props)

    context.advance()

    assert context.props_store.exists("props.yaml") is False
    context.props_store.write("props.yaml", props)
    assert (slice_dir(action_dir, 1) / "props.yaml").is_file()


def test_frame_store_writes_into_the_flat_frames_directory(
    context: FileActionContext, action_dir: Path
) -> None:
    assert context.frame_store.path() == action_dir / "frames" / "slice_0000.tif"


def test_frame_store_follows_an_advance(
    context: FileActionContext, action_dir: Path
) -> None:
    context.advance()

    assert context.frame_store.path() == action_dir / "frames" / "slice_0001.tif"


def test_frame_store_uses_the_configured_directory_name(action_dir: Path) -> None:
    context = FileActionContext(action_dir, "milling", frames_directory_name="raw")

    path = context.frame_store.path()
    assert path is not None
    assert path.parent == action_dir / "raw"


def test_all_resources_move_together(
    context: FileActionContext, action_dir: Path
) -> None:
    """One advance must move every store and logger, not just the one used next."""
    context.advance()

    context.text_logger.info("x")
    context.image_logger.save_image("drift.png", np.ones((8, 8)))
    context.props_store.write("props.yaml", GlobalProperties())
    close_all_log_files()

    written = sorted(p.name for p in slice_dir(action_dir, 1).iterdir())
    assert written == ["drift.png", "props.yaml", "run.log"]
    assert not slice_dir(action_dir, 0).exists()


def test_resources_follow_set_slice(
    context: FileActionContext, action_dir: Path
) -> None:
    context.set_slice(4)

    context.props_store.write("props.yaml", GlobalProperties())

    assert (slice_dir(action_dir, 4) / "props.yaml").is_file()


def test_resources_follow_reset(context: FileActionContext, action_dir: Path) -> None:
    context.advance()
    context.reset()

    context.props_store.write("props.yaml", GlobalProperties())

    assert (slice_dir(action_dir, 0) / "props.yaml").is_file()


def test_image_store_is_file_backed(context: FileActionContext) -> None:
    assert isinstance(context.image_store(Image), FileImageStore)


def test_image_store_round_trips_an_image(context: FileActionContext) -> None:
    store = context.image_store(Image)
    image = Image(np.arange(64, dtype=np.uint16).reshape(8, 8), pixel_size=2.5)

    store.write("drift", image)

    assert np.array_equal(store.read("drift"), image)


def test_image_store_writes_into_the_current_slice(
    context: FileActionContext, action_dir: Path
) -> None:
    store = context.image_store(Image)

    store.write("drift", Image(np.ones((4, 4), dtype=np.uint16), pixel_size=1.0))

    assert (slice_dir(action_dir, 0) / "drift.tif").is_file()


def test_image_store_reads_back_the_requested_class(
    context: FileActionContext,
) -> None:
    store = context.image_store(Image8Bit)

    store.write("drift", Image8Bit(np.ones((4, 4), dtype=np.uint8), pixel_size=1.0))

    assert type(store.read("drift")) is Image8Bit


def test_image_store_returns_a_new_instance_per_call(
    context: FileActionContext,
) -> None:
    """Unlike the other resources, this is a factory, not a cached property."""
    assert context.image_store(Image) is not context.image_store(Image)


def test_image_stores_of_different_classes_are_independent(
    context: FileActionContext,
) -> None:
    integer_store = context.image_store(Image)
    eight_bit_store = context.image_store(Image8Bit)
    integer_store.write(
        "drift", Image(np.ones((4, 4), dtype=np.uint16), pixel_size=1.0)
    )

    assert type(integer_store.read("drift")) is Image
    assert type(eight_bit_store.read("drift")) is Image8Bit


def test_an_image_store_follows_an_advance(
    context: FileActionContext, action_dir: Path
) -> None:
    """The store holds the context's view provider, not a snapshot of the slice."""
    store = context.image_store(Image)

    context.advance()
    store.write("drift", Image(np.ones((4, 4), dtype=np.uint16), pixel_size=1.0))

    assert (slice_dir(action_dir, 1) / "drift.tif").is_file()


def test_change_action_dir_moves_the_directory(
    context: FileActionContext, action_dir: Path, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"

    context.change_action_dir(target)

    assert not action_dir.exists()
    assert (target / "slice_0000" / "props.yaml").is_file()


def test_change_action_dir_updates_path_to_dir(
    context: FileActionContext, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"

    context.change_action_dir(target)

    assert context.path_to_dir == target


def test_change_action_dir_updates_the_current_view(
    context: FileActionContext, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"

    context.change_action_dir(target)

    assert context.current_view.action_dir == target


def test_change_action_dir_keeps_the_slice_index(
    context: FileActionContext, tmp_path: Path
) -> None:
    context.advance()
    context.advance()
    context.props_store.write("props.yaml", GlobalProperties())

    context.change_action_dir(tmp_path / "moved")

    assert context.slice == 2
    assert context.current_view.slice_index == 2


def test_stores_write_to_the_new_directory_afterwards(
    context: FileActionContext, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"

    context.change_action_dir(target)
    context.props_store.write("later.yaml", GlobalProperties())

    assert (target / "slice_0000" / "later.yaml").is_file()


def test_frame_store_follows_the_new_directory(
    context: FileActionContext, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"

    context.change_action_dir(target)

    assert context.frame_store.path() == target / "frames" / "slice_0000.tif"


def test_change_action_dir_rejects_an_existing_target(
    context: FileActionContext, tmp_path: Path
) -> None:
    """Renaming onto an existing directory would merge or clobber it."""
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"
    target.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        context.change_action_dir(target)


def test_a_rejected_change_leaves_the_context_unchanged(
    context: FileActionContext, action_dir: Path, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())
    target = tmp_path / "moved"
    target.mkdir()

    with pytest.raises(FileExistsError):
        context.change_action_dir(target)

    assert context.path_to_dir == action_dir
    assert (action_dir / "slice_0000" / "props.yaml").is_file()


def test_change_action_dir_works_when_nothing_has_been_written(
    context: FileActionContext, tmp_path: Path
) -> None:
    target = tmp_path / "moved"

    context.change_action_dir(target)

    assert context.path_to_dir == target


def test_change_action_dir_before_any_write_creates_nothing(
    context: FileActionContext, action_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "moved"

    context.change_action_dir(target)

    assert not action_dir.exists()
    assert not target.exists()


def test_writes_after_an_empty_change_land_in_the_new_directory(
    context: FileActionContext, tmp_path: Path
) -> None:
    target = tmp_path / "moved"

    context.change_action_dir(target)
    context.props_store.write("props.yaml", GlobalProperties())

    assert (target / "slice_0000" / "props.yaml").is_file()


def test_change_action_dir_can_be_repeated(
    context: FileActionContext, tmp_path: Path
) -> None:
    context.props_store.write("props.yaml", GlobalProperties())

    context.change_action_dir(tmp_path / "first")
    context.change_action_dir(tmp_path / "second")

    assert context.path_to_dir == tmp_path / "second"
    assert (tmp_path / "second" / "slice_0000" / "props.yaml").is_file()

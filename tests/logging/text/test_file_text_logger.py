# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import logging
import re
import threading
from pathlib import Path

import pytest

from fibsem_maestro.logging.text.file import FileTextLogger, close_all_log_files
from fibsem_maestro.slice.slice_view import SliceView

LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[(?P<name>[^\]]*)\] "
    r"(?P<level>\w+): (?P<message>.*)$"
)


class SliceCursor:
    """
    Mutable stand-in for the run loop's slice pointer.

    `SliceView` is immutable, so advancing a slice means handing the logger a
    new view rather than mutating the old one. Instances are callable and so
    satisfy `Callable[[], SliceView]` directly.

    Args:
        action_dir: The action's root directory.
        slice_index: The slice the cursor starts on.
    """

    def __init__(self, action_dir: Path, slice_index: int = 0) -> None:
        self._action_dir = action_dir
        self._view = SliceView(action_dir, slice_index)

    def __call__(self) -> SliceView:
        return self._view

    def advance_to(self, slice_index: int) -> None:
        """Point the cursor at another slice, as the run loop does per slice."""
        self._view = SliceView(self._action_dir, slice_index)

    def log_path(self, slice_index: int, filename: str = "run.log") -> Path:
        """Resolve a slice's log file through `SliceView`, not by name convention."""
        return SliceView(self._action_dir, slice_index).path() / filename

    @property
    def action_dir(self) -> Path:
        return self._action_dir


@pytest.fixture(autouse=True)
def close_handlers():
    """Roots outlive the tests that make them; release handles between tests."""
    yield
    close_all_log_files()


@pytest.fixture
def cursor(tmp_path: Path) -> SliceCursor:
    return SliceCursor(tmp_path)


@pytest.fixture
def logger(cursor: SliceCursor) -> FileTextLogger:
    return FileTextLogger(cursor, "run", level=logging.DEBUG)


def read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def parsed(path: Path) -> list[re.Match]:
    lines = list(read_lines(path))
    matches = [m for line in lines if (m := LINE.match(line))]
    if len(matches) != len(lines):
        raise ValueError(f"unparsable lines in {path}")
    return matches


def test_info_writes_a_record_to_the_current_slice_file(logger, cursor):
    logger.info("acquisition started")
    logger.close()

    records = parsed(cursor.log_path(0))
    assert len(records) == 1
    assert records[0]["message"] == "acquisition started"


def test_record_contains_logger_name_and_level(logger, cursor):
    logger.warning("drift above threshold")
    logger.close()

    record = parsed(cursor.log_path(0))[0]
    assert record["name"] == "run"
    assert record["level"] == "WARNING"


@pytest.mark.parametrize(
    ("method", "level"),
    [("debug", "DEBUG"), ("info", "INFO"), ("warning", "WARNING"), ("error", "ERROR")],
)
def test_each_method_writes_its_own_level(logger, cursor, method, level):
    getattr(logger, method)("message")
    logger.close()

    assert parsed(cursor.log_path(0))[0]["level"] == level


def test_successive_calls_append_rather_than_truncate(logger, cursor):
    logger.info("first")
    logger.info("second")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["first", "second"]


def test_file_created_in_the_slice_directory(logger, cursor):
    logger.info("x")
    logger.close()

    assert cursor.log_path(0).exists()


def test_custom_filename_is_used(cursor):
    logger = FileTextLogger(cursor, "run", filename="autofunction.log")

    logger.info("x")
    logger.close()

    assert cursor.log_path(0, "autofunction.log").exists()
    assert not cursor.log_path(0).exists()


def test_debug_is_discarded_below_the_configured_level(cursor):
    logger = FileTextLogger(cursor, "run", level=logging.INFO)

    logger.debug("should not appear")
    logger.info("should appear")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["should appear"]


def test_below_level_call_does_not_create_the_file(cursor):
    logger = FileTextLogger(cursor, "run", level=logging.INFO)

    logger.debug("suppressed")

    assert not cursor.log_path(0).exists()


def test_level_threshold_is_inclusive(cursor):
    logger = FileTextLogger(cursor, "run", level=logging.WARNING)

    logger.warning("kept")
    logger.info("dropped")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["kept"]


def test_error_survives_a_high_threshold(cursor):
    logger = FileTextLogger(cursor, "run", level=logging.WARNING)

    logger.error("failure")
    logger.close()

    assert parsed(cursor.log_path(0))[0]["level"] == "ERROR"


def test_derived_logger_inherits_the_level(cursor):
    logger = FileTextLogger(cursor, "run", level=logging.INFO)

    logger.derive("milling").debug("suppressed")

    assert not cursor.log_path(0).exists()


def test_exception_writes_the_traceback(logger, cursor):
    try:
        raise ValueError("beam shift limit reached")
    except ValueError:
        logger.exception("could not correct drift")
    logger.close()

    contents = cursor.log_path(0).read_text()
    assert "could not correct drift" in contents
    assert "ValueError: beam shift limit reached" in contents
    assert "Traceback (most recent call last)" in contents


def test_exception_logs_at_error_level(logger, cursor):
    try:
        raise ValueError("x")
    except ValueError:
        logger.exception("failed")
    logger.close()

    first = LINE.match(read_lines(cursor.log_path(0))[0])
    assert first is not None
    assert first["level"] == "ERROR"


def test_exception_outside_an_except_block_logs_only_the_message(logger, cursor):
    logger.exception("nothing active")
    logger.close()

    assert len(parsed(cursor.log_path(0))) == 1
    assert "Traceback" not in cursor.log_path(0).read_text()


def test_derive_appends_to_the_parent_name(logger, cursor):
    logger.derive("milling").info("x")
    logger.close()

    assert parsed(cursor.log_path(0))[0]["name"] == "run.milling"


def test_derive_from_empty_name_uses_the_child_name_alone(cursor):
    FileTextLogger(cursor, "").derive("milling").info("x")
    close_all_log_files()

    assert parsed(cursor.log_path(0))[0]["name"] == "milling"


def test_derive_nests(logger, cursor):
    logger.derive("milling").derive("pattern").info("x")
    logger.close()

    assert parsed(cursor.log_path(0))[0]["name"] == "run.milling.pattern"


def test_derive_writes_to_the_same_file(logger, cursor):
    logger.info("parent")
    logger.derive("child").info("child")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["parent", "child"]


def test_derive_does_not_change_the_parent_name(logger, cursor):
    logger.derive("milling")
    logger.info("x")
    logger.close()

    assert parsed(cursor.log_path(0))[0]["name"] == "run"


def test_derive_returns_the_same_concrete_type(logger):
    assert type(logger.derive("milling")) is type(logger)


def test_derive_does_not_register_a_python_logger(logger):
    """Records go straight to the handler; no entries in the global registry."""
    logger.derive("milling").info("x")

    assert "run.milling" not in logging.Logger.manager.loggerDict


def test_rotates_when_the_view_moves_to_the_next_slice(logger, cursor):
    logger.info("slice zero")
    cursor.advance_to(1)
    logger.info("slice one")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["slice zero"]
    assert [r["message"] for r in parsed(cursor.log_path(1))] == ["slice one"]


def test_returning_to_an_earlier_slice_appends(logger, cursor):
    """Rotation must reopen in append mode, not truncate the earlier file."""
    logger.info("first")
    cursor.advance_to(1)
    logger.info("other slice")
    cursor.advance_to(0)
    logger.info("second")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["first", "second"]


def test_slice_reports_the_current_index(logger, cursor):
    assert logger.slice == 0

    cursor.advance_to(7)

    assert logger.slice == 7


def test_at_writes_to_the_requested_slice(logger, cursor):
    logger.at(5).info("out of band")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(5))] == ["out of band"]


def test_at_does_not_move_the_original_logger(logger, cursor):
    logger.at(5).info("elsewhere")
    logger.info("here")
    logger.close()

    assert logger.slice == 0
    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["here"]


def test_at_is_pinned_to_its_slice_when_the_view_moves(logger, cursor):
    pinned = logger.at(3)

    cursor.advance_to(9)
    pinned.info("still three")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(3))] == ["still three"]


def test_at_preserves_the_logger_name(logger, cursor):
    logger.derive("milling").at(2).info("x")
    logger.close()

    assert parsed(cursor.log_path(2))[0]["name"] == "run.milling"


def test_at_reports_the_requested_slice(logger):
    assert logger.at(4).slice == 4


def test_next_writes_to_the_following_slice(logger, cursor):
    logger.next.info("correction applied")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(1))] == ["correction applied"]


def test_next_is_relative_to_the_current_slice(logger, cursor):
    cursor.advance_to(4)

    logger.next.info("x")
    logger.close()

    assert cursor.log_path(5).exists()


def test_next_reports_the_following_slice(logger, cursor):
    cursor.advance_to(2)

    assert logger.next.slice == 3


def test_next_does_not_move_the_original_logger(logger):
    logger.next.info("ahead")

    assert logger.slice == 0


def test_next_preserves_the_logger_name(logger, cursor):
    logger.derive("drift").next.info("x")
    logger.close()

    assert parsed(cursor.log_path(1))[0]["name"] == "run.drift"


def test_next_of_next_advances_twice(logger, cursor):
    logger.next.next.info("x")
    logger.close()

    assert cursor.log_path(2).exists()


def open_log_handles(directory: Path) -> int:
    """Count open file descriptors pointing at log files under `directory`."""
    count = 0
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = str(entry.readlink())
        except OSError:
            continue
        if target.startswith(str(directory)) and target.endswith(".log"):
            count += 1
    return count


needs_procfs = pytest.mark.skipif(
    not Path("/proc/self/fd").exists(), reason="requires procfs"
)


@needs_procfs
def test_derived_loggers_share_one_file_handle(logger, cursor):
    logger.info("x")
    for name in ("milling", "imaging", "drift"):
        logger.derive(name).info("x")

    assert open_log_handles(cursor.action_dir) == 1


@needs_procfs
def test_at_does_not_open_a_second_file_handle(logger, cursor):
    logger.info("x")
    fifth = logger.at(5)
    sixth = logger.at(6)
    fifth.info("x")
    sixth.info("x")

    assert open_log_handles(cursor.action_dir) == 1


@needs_procfs
def test_next_does_not_open_a_second_file_handle(logger, cursor):
    logger.info("x")
    upcoming = logger.next
    upcoming.info("x")

    assert open_log_handles(cursor.action_dir) == 1


@needs_procfs
def test_close_releases_the_file_handle(logger, cursor):
    logger.info("x")

    logger.close()

    assert open_log_handles(cursor.action_dir) == 0


def test_logging_after_close_reopens_and_appends(logger, cursor):
    logger.info("before")
    logger.close()
    logger.info("after")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["before", "after"]


def test_close_is_idempotent(logger, cursor):
    logger.info("x")

    logger.close()
    logger.close()

    assert len(parsed(cursor.log_path(0))) == 1


def test_close_without_any_logging_is_safe(logger, cursor):
    logger.close()

    assert not cursor.log_path(0).exists()


def test_close_on_a_derived_logger_closes_the_shared_handle(logger, cursor):
    logger.info("x")

    logger.derive("milling").close()
    logger.info("y")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["x", "y"]


def test_close_allows_the_slice_directory_to_be_removed(logger, cursor):
    """Windows refuses to remove a directory holding an open file."""
    logger.info("x")

    logger.close()

    path = cursor.log_path(0)
    path.unlink()
    path.parent.rmdir()
    assert not path.parent.exists()


@needs_procfs
def test_close_all_log_files_closes_every_logger(cursor, tmp_path):
    other = SliceCursor(tmp_path / "other")
    first = FileTextLogger(cursor, "first")
    second = FileTextLogger(other, "second")
    first.info("x")
    second.info("x")
    assert open_log_handles(tmp_path) == 2

    close_all_log_files()

    assert open_log_handles(tmp_path) == 0


def test_close_all_log_files_preserves_written_content(cursor, logger):
    logger.info("x")

    close_all_log_files()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["x"]


def test_logging_after_close_all_log_files_reopens(logger, cursor):
    logger.info("before")

    close_all_log_files()
    logger.info("after")
    logger.close()

    assert [r["message"] for r in parsed(cursor.log_path(0))] == ["before", "after"]


def test_close_all_log_files_with_no_loggers_is_safe():
    close_all_log_files()


def test_close_all_log_files_is_idempotent(logger, cursor):
    logger.info("x")

    close_all_log_files()
    close_all_log_files()

    assert len(parsed(cursor.log_path(0))) == 1


def test_concurrent_writes_produce_complete_records(logger, cursor):
    """Actions log from worker threads; `Handler.handle` takes the I/O lock."""
    payload = "x" * 200

    def work(index: int) -> None:
        child = logger.derive(f"worker{index}")
        for counter in range(50):
            child.info(f"{index}-{counter}-{payload}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logger.close()

    records = parsed(cursor.log_path(0))
    assert len(records) == 8 * 50
    assert all(record["message"].endswith(payload) for record in records)


def test_warning_once_writes_the_first_occurrence_as_a_warning(logger, cursor):
    logger.warning_once("old-autoscript", "fallback used")
    logger.close()

    records = parsed(cursor.log_path(0))
    assert [(r["level"], r["message"]) for r in records] == [
        ("WARNING", "fallback used")
    ]


def test_warning_once_writes_repeats_as_debug(logger, cursor):
    logger.warning_once("old-autoscript", "fallback used")
    logger.warning_once("old-autoscript", "fallback used")
    logger.warning_once("old-autoscript", "fallback used")
    logger.close()

    levels = [r["level"] for r in parsed(cursor.log_path(0))]
    assert levels == ["WARNING", "DEBUG", "DEBUG"]


def test_warning_once_repeats_are_dropped_above_debug_level(cursor):
    """With the usual INFO threshold, a repeated condition adds nothing to the log."""
    logger = FileTextLogger(cursor, "run", level=logging.INFO)

    logger.warning_once("old-autoscript", "fallback used")
    logger.warning_once("old-autoscript", "fallback used")
    logger.close()

    assert [r["level"] for r in parsed(cursor.log_path(0))] == ["WARNING"]


def test_warning_once_deduplicates_by_key_not_by_message(logger, cursor):
    logger.warning_once("get", "fallback used")
    logger.warning_once("set", "fallback used")
    logger.warning_once("get", "a different text for the same condition")
    logger.close()

    levels = [r["level"] for r in parsed(cursor.log_path(0))]
    assert levels == ["WARNING", "WARNING", "DEBUG"]


def test_plain_warning_is_never_deduplicated(logger, cursor):
    logger.warning_once("old-autoscript", "fallback used")
    logger.warning("fallback used")
    logger.warning("fallback used")
    logger.close()

    assert [r["level"] for r in parsed(cursor.log_path(0))] == ["WARNING"] * 3


def test_warning_once_is_remembered_across_slices(logger, cursor):
    """A persistent condition warns once per run, not once per slice."""
    logger.warning_once("old-autoscript", "fallback used")
    cursor.advance_to(1)
    logger.warning_once("old-autoscript", "fallback used")
    logger.close()

    assert [r["level"] for r in parsed(cursor.log_path(0))] == ["WARNING"]
    assert [r["level"] for r in parsed(cursor.log_path(1))] == ["DEBUG"]


def test_warning_once_is_remembered_across_close(logger, cursor):
    """Closing releases the file, not what the logger has already reported."""
    logger.warning_once("old-autoscript", "fallback used")
    close_all_log_files()
    logger.warning_once("old-autoscript", "fallback used")
    logger.close()

    assert [r["level"] for r in parsed(cursor.log_path(0))] == ["WARNING", "DEBUG"]


def test_derived_logger_warns_once_on_its_own(logger, cursor):
    """Each component derives its own logger; one component's warning must not hide another's."""
    logger.warning_once("old-autoscript", "fallback used")
    logger.derive("ion").warning_once("old-autoscript", "fallback used")
    logger.close()

    records = [(r["name"], r["level"]) for r in parsed(cursor.log_path(0))]
    assert records == [("run", "WARNING"), ("run.ion", "WARNING")]

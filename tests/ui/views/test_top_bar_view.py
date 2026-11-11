# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The top bar: what `top_bar_view` decides, and the widget's buttons, dialogs and danger toggle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gui_testkit import TIMEOUT_MS
from gui_testkit.app import wait_for_state
from gui_testkit.driving import (
    add_prepared,
    answering,
    button,
    open_app,
    task_running,
)
from gui_testkit.workflows import simulated_workflow
from PyQt6.QtWidgets import QMessageBox

from fibsem_maestro.ui.app.edit_policy import permissions_for
from fibsem_maestro.ui.app.files import WorkflowFiles
from fibsem_maestro.ui.app.readiness import ReadinessReport
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.app.session import WorkflowSession
from fibsem_maestro.ui.views.dialogs.microscope_settings import (
    MicroscopeSettingsDialog,
)
from fibsem_maestro.ui.views.dialogs.slice_skip import SLICE_SKIP_TITLE
from fibsem_maestro.ui.views.top_bar import (
    SKIP_NOTHING_REASON,
    SKIP_STATE_REASON,
    SKIP_TOOLTIP,
    TASK_RUNNING_REASON,
    WAITING_FOR_CHOICE_TOOLTIP,
    top_bar_view,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from gui_testkit.dialogs import DirectoryChoices, MessageBoxes
    from gui_testkit.situations import HeldStep
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.ui.app.user_config import UserConfig
    from fibsem_maestro.ui.composition import Composition
    from fibsem_maestro.workflow.workflow import Workflow

READY = ReadinessReport()
NOT_READY = ReadinessReport(("'a' is not prepared for slice 0.", "'b': broken link"))
IDLE = permissions_for(AppState.EDITING, danger_mode=False, task_busy=False)


def test_top_bar_view_running_offers_an_enabled_pause() -> None:
    data = top_bar_view(
        AppState.RUNNING,
        3,
        NOT_READY,
        permissions_for(AppState.RUNNING, False, False),
        0,
    )
    assert (data.run_icon, data.run_enabled, data.run_tooltip) == ("pause", True, "")
    assert (data.slice_text, data.state_text) == ("3", "running")


def test_top_bar_view_stopping_leaves_the_run_button_alone() -> None:
    data = top_bar_view(
        AppState.STOPPING, 3, READY, permissions_for(AppState.STOPPING, False, False), 1
    )
    assert (data.run_icon, data.run_enabled, data.run_tooltip) == (None, None, None)


def test_top_bar_view_stopping_blinks() -> None:
    permissions = permissions_for(AppState.STOPPING, False, False)
    even = top_bar_view(AppState.STOPPING, 0, READY, permissions, 0)
    odd = top_bar_view(AppState.STOPPING, 0, READY, permissions, 1)
    assert even.color != odd.color


def test_top_bar_view_skipping_blinks_between_running_and_its_own_colour() -> None:
    permissions = permissions_for(AppState.SKIPPING, False, False)
    even = top_bar_view(AppState.SKIPPING, 0, READY, permissions, 0)
    odd = top_bar_view(AppState.SKIPPING, 0, READY, permissions, 1)
    running = top_bar_view(AppState.RUNNING, 0, READY, permissions, 1)
    stopping = top_bar_view(AppState.STOPPING, 0, READY, permissions, 1)

    assert even.state_text == odd.state_text == "skipping"
    assert even.color == running.color
    # its own colour, not that of a pause
    assert odd.color not in (running.color, stopping.color)


def test_top_bar_view_skipping_offers_no_pause_since_it_pauses_anyway() -> None:
    data = top_bar_view(
        AppState.SKIPPING, 3, READY, permissions_for(AppState.SKIPPING, False, False), 0
    )
    assert (data.run_icon, data.run_enabled) == ("pause", False)
    assert data.run_tooltip


@pytest.mark.parametrize(
    ("state", "busy", "readiness", "planned", "reason"),
    [
        (AppState.RUNNING, False, NOT_READY, True, None),
        (AppState.PAUSED, False, READY, True, None),
        (AppState.RELOADED, False, READY, True, None),
        (AppState.INTERRUPTED, False, NOT_READY, True, None),
        (AppState.PAUSED, False, NOT_READY, True, NOT_READY.reasons[0]),
        (AppState.RELOADED, False, NOT_READY, True, NOT_READY.reasons[0]),
        (AppState.PAUSED, False, READY, False, SKIP_NOTHING_REASON),
        (AppState.RELOADED, True, READY, True, TASK_RUNNING_REASON),
        (AppState.EDITING, False, READY, True, SKIP_STATE_REASON),
        (AppState.FINISHED, False, READY, True, SKIP_STATE_REASON),
        (AppState.STOPPING, False, READY, True, SKIP_STATE_REASON),
        (AppState.SKIPPING, False, READY, True, SKIP_STATE_REASON),
    ],
)
def test_top_bar_view_skip_button_says_why_it_is_disabled(
    state: AppState,
    busy: bool,
    readiness: ReadinessReport,
    planned: bool,
    reason: str | None,
) -> None:
    data = top_bar_view(
        state, 1, readiness, permissions_for(state, False, busy), 0, planned
    )

    assert data.skip_enabled is (reason is None)
    if reason is None:
        assert data.skip_tooltip == SKIP_TOOLTIP
    else:
        assert reason in data.skip_tooltip


def test_top_bar_view_interrupted_waits_for_the_error_dialog() -> None:
    data = top_bar_view(
        AppState.INTERRUPTED,
        1,
        READY,
        permissions_for(AppState.INTERRUPTED, False, False),
        0,
    )
    assert (data.run_icon, data.run_enabled) == ("play", False)
    assert data.run_tooltip == WAITING_FOR_CHOICE_TOOLTIP


@pytest.mark.parametrize(
    ("state", "heading"),
    [
        (AppState.EDITING, "Cannot start:"),
        (AppState.RELOADED, "Cannot start:"),
        (AppState.FINISHED, "Cannot start:"),
        (AppState.PAUSED, "Cannot resume:"),
    ],
)
def test_top_bar_view_not_ready_lists_the_reasons(
    state: AppState, heading: str
) -> None:
    data = top_bar_view(state, 1, NOT_READY, permissions_for(state, False, False), 0)

    assert (data.run_icon, data.run_enabled) == ("play", False)
    assert data.run_tooltip is not None
    lines = data.run_tooltip.splitlines()
    assert lines[0] == heading
    assert all(any(reason in line for line in lines) for reason in NOT_READY.reasons)


@pytest.mark.parametrize(
    "state", [AppState.EDITING, AppState.RELOADED, AppState.FINISHED, AppState.PAUSED]
)
def test_top_bar_view_ready_enables_run_without_a_tooltip(state: AppState) -> None:
    data = top_bar_view(state, 1, READY, permissions_for(state, False, False), 0)
    assert (data.run_icon, data.run_enabled, data.run_tooltip) == ("play", True, "")


def test_top_bar_view_running_task_blocks_run_and_comes_first() -> None:
    busy = permissions_for(AppState.EDITING, False, True)

    ready = top_bar_view(AppState.EDITING, 0, READY, busy, 0)
    not_ready = top_bar_view(AppState.EDITING, 0, NOT_READY, busy, 0)

    assert ready.run_enabled is False
    assert ready.run_tooltip is not None and TASK_RUNNING_REASON in ready.run_tooltip
    assert not_ready.run_tooltip is not None
    reasons = not_ready.run_tooltip.splitlines()[1:]
    assert TASK_RUNNING_REASON in reasons[0]


@pytest.mark.parametrize(
    ("state", "task_busy"),
    [
        (AppState.EDITING, False),
        (AppState.RELOADED, False),
        (AppState.RUNNING, False),
        (AppState.INTERRUPTED, False),
        (AppState.EDITING, True),
    ],
)
def test_top_bar_view_workflow_buttons_follow_the_permissions(
    state: AppState, task_busy: bool
) -> None:
    permissions = permissions_for(state, False, task_busy)
    data = top_bar_view(state, 0, READY, permissions, 0)
    assert (data.new_enabled, data.import_enabled, data.reset_enabled) == (
        permissions.new_enabled,
        permissions.import_enabled,
        permissions.reset_enabled,
    )


def test_top_bar_rules_hold_from_start_up(app: Composition) -> None:
    top_bar = app.window.top_bar
    assert top_bar.import_button.isEnabled()
    assert top_bar.new_button.isEnabled()
    assert not top_bar.run_button.isEnabled()
    assert top_bar.run_button.toolTip().startswith("Cannot start:")
    assert top_bar.directory_label.text() == str(app.session.workflow_dir)


def test_top_bar_resumed_workflow_disables_import_from_start_up(
    qtbot: QtBot,
    request: pytest.FixtureRequest,
    workflow: Workflow,
    workflow_dir: Path,
    user_config: UserConfig,
    files: WorkflowFiles,
) -> None:
    # a workflow past slice 0 is resumed (RELOADED); D13: the rule holds at once
    workflow.ctx.advance()
    resumed = open_app(qtbot, request, workflow, workflow_dir, user_config, files)

    assert resumed.run.state is AppState.RELOADED
    assert not resumed.window.top_bar.import_button.isEnabled()


def test_top_bar_run_button_starts_pauses_and_resumes(
    qtbot: QtBot, app: Composition
) -> None:
    add_prepared(qtbot, app, "first")
    top_bar = app.window.top_bar
    assert top_bar.run_button.isEnabled()
    assert top_bar.run_button.toolTip() == ""

    top_bar.run_button.click()
    assert app.run.state is AppState.RUNNING
    assert top_bar.view_data().run_icon == "pause"
    assert not top_bar.new_button.isEnabled()
    qtbot.waitUntil(
        lambda: top_bar.slice_label.text() not in ("", "0"), timeout=TIMEOUT_MS
    )

    top_bar.run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)
    assert top_bar.slice_label.text() == str(app.session.slice)
    assert top_bar.view_data().run_icon == "play"
    assert top_bar.run_button.isEnabled()

    top_bar.run_button.click()
    assert app.run.state is AppState.RUNNING


def test_top_bar_run_button_waits_for_a_microscope_task(
    qtbot: QtBot, app: Composition
) -> None:
    action = add_prepared(qtbot, app, "first")
    run_button = app.window.top_bar.run_button

    with task_running(qtbot, app, action):
        assert not run_button.isEnabled()
        assert TASK_RUNNING_REASON in run_button.toolTip()
        assert not app.window.top_bar.new_button.isEnabled()
        assert not app.window.top_bar.import_button.isEnabled()
        assert not app.window.top_bar.reset_button.isEnabled()

    assert run_button.isEnabled()


def test_top_bar_new_workflow_into_non_empty_directory_asks_first(
    tmp_path: Path,
    app: Composition,
    directory_choice: DirectoryChoices,
    boxes: MessageBoxes,
) -> None:
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / "keep.txt").write_text("x")
    directory_choice.path = str(new_dir)
    boxes.answer = QMessageBox.StandardButton.Cancel
    old_dir = app.session.workflow_dir

    app.window.top_bar.new_button.click()

    assert boxes.titles() == ["Directory not empty"]
    assert app.session.workflow_dir == old_dir
    assert (new_dir / "keep.txt").exists()

    boxes.answer = QMessageBox.StandardButton.Yes
    app.window.top_bar.new_button.click()

    assert directory_choice.titles == ["Choose workflow save directory"] * 2
    assert app.session.workflow_dir == new_dir
    assert not (new_dir / "keep.txt").exists()
    assert app.window.top_bar.directory_label.text() == str(new_dir)


def test_top_bar_new_workflow_reports_a_directory_it_could_not_clear(
    qtbot: QtBot,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    workflow: Workflow,
    workflow_dir: Path,
    user_config: UserConfig,
    directory_choice: DirectoryChoices,
    boxes: MessageBoxes,
) -> None:
    def refuse(_: Path) -> None:
        raise PermissionError("in use")

    app = open_app(
        qtbot, request, workflow, workflow_dir, user_config, WorkflowFiles(refuse)
    )
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / "locked.txt").write_text("x")
    directory_choice.path = str(new_dir)

    app.window.top_bar.new_button.click()

    assert boxes.titles() == ["Directory not empty", "Error"]
    assert boxes.texts()[1].startswith("Could not fully clear the workflow directory.")
    assert "locked.txt" in boxes.texts()[1]
    assert app.session.workflow_dir == workflow_dir


def test_top_bar_new_workflow_reports_a_failed_first_write(
    qtbot: QtBot,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    workflow: Workflow,
    workflow_dir: Path,
    user_config: UserConfig,
    directory_choice: DirectoryChoices,
    boxes: MessageBoxes,
) -> None:
    # deleting "succeeds" but leaves a file where the new workflow's folder goes
    app = open_app(
        qtbot,
        request,
        workflow,
        workflow_dir,
        user_config,
        WorkflowFiles(lambda _: None),
    )
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / "workflow").write_text("x")
    directory_choice.path = str(new_dir)

    app.window.top_bar.new_button.click()

    assert boxes.titles()[-1] == "Error"
    assert boxes.texts()[-1].startswith("Could not create the new workflow: ")
    assert app.session.workflow is workflow
    assert app.run.state is AppState.EDITING


def test_top_bar_import_replaces_the_actions_or_shows_why_not(
    tmp_path: Path,
    app: Composition,
    directory_choice: DirectoryChoices,
    boxes: MessageBoxes,
) -> None:
    app.commands.add_action("imaging", "old")
    not_a_workflow = tmp_path / "not_a_workflow"
    not_a_workflow.mkdir()
    directory_choice.path = str(not_a_workflow)

    app.window.top_bar.import_button.click()

    assert boxes.titles() == ["Error"]
    assert boxes.texts()[0].startswith("Could not import workflow: ")
    assert [a.name for a in app.session.actions] == ["old"]

    source = tmp_path / "source"
    source.mkdir()
    WorkflowSession(simulated_workflow(source), source, WorkflowFiles()).add_action(
        "imaging", "imported"
    )
    directory_choice.path = str(source)

    app.window.top_bar.import_button.click()

    assert directory_choice.titles == ["Select workflow directory"] * 2
    assert [a.name for a in app.session.actions] == ["imported"]


def test_top_bar_reset_asks_then_returns_to_slice_zero(
    qtbot: QtBot, app: Composition, boxes: MessageBoxes
) -> None:
    add_prepared(qtbot, app, "first")
    app.window.top_bar.run_button.click()
    qtbot.waitUntil(lambda: app.session.slice >= 1, timeout=TIMEOUT_MS)
    app.window.top_bar.run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)

    boxes.answer = QMessageBox.StandardButton.Cancel
    app.window.top_bar.reset_button.click()
    assert app.session.slice >= 1
    assert app.run.state is AppState.PAUSED

    boxes.answer = QMessageBox.StandardButton.Yes
    app.window.top_bar.reset_button.click()

    assert boxes.titles() == ["Data will be lost!", "Data will be lost!"]
    assert app.session.slice == 0
    assert app.run.state is AppState.EDITING
    assert app.window.top_bar.slice_label.text() == "0"


def test_top_bar_reset_failure_shows_an_error_and_ends_editing(
    qtbot: QtBot,
    request: pytest.FixtureRequest,
    workflow: Workflow,
    workflow_dir: Path,
    user_config: UserConfig,
    boxes: MessageBoxes,
) -> None:
    def refuse(_: Path) -> None:
        raise PermissionError("in use")

    app = open_app(
        qtbot, request, workflow, workflow_dir, user_config, WorkflowFiles(refuse)
    )
    app.commands.add_action("imaging", "imaging")

    app.window.top_bar.reset_button.click()

    assert boxes.titles() == ["Data will be lost!", "Error"]
    # the cause is reported
    assert "in use" in boxes.texts()[1]
    assert app.run.state is AppState.EDITING


def test_top_bar_danger_toggle_and_banner_follow_danger_mode(app: Composition) -> None:
    toggle = app.window.top_bar.danger_button
    banner = app.window.danger_banner
    assert not toggle.isChecked()
    assert not banner.isVisible()

    toggle.click()
    assert app.window.top_bar.danger_button.isChecked()
    assert banner.isVisible()

    # the toggle follows the mode, whoever changes it
    app.commands.set_danger_mode(False)
    assert not toggle.isChecked()
    assert not banner.isVisible()


def test_top_bar_microscope_settings_opens_an_editable_dialog_while_editing(
    app: Composition,
) -> None:
    def apply(dialog: MicroscopeSettingsDialog) -> None:
        assert not dialog.read_only
        button(dialog, "Apply").click()

    with answering(MicroscopeSettingsDialog, apply) as shown:
        app.window.top_bar.microscope_button.click()

    assert len(shown) == 1
    assert shown[0].result() == MicroscopeSettingsDialog.DialogCode.Accepted


def test_top_bar_skip_asks_then_skips_the_rest_of_the_slice(
    qtbot: QtBot,
    app: Composition,
    boxes: MessageBoxes,
    hold: Callable[..., HeldStep],
) -> None:
    first = add_prepared(qtbot, app, "first")
    add_prepared(qtbot, app, "second")
    top_bar = app.window.top_bar
    assert not top_bar.skip_button.isEnabled()
    held = hold(first)
    top_bar.run_button.click()
    qtbot.waitUntil(held.entered.is_set, timeout=TIMEOUT_MS)
    assert top_bar.skip_button.isEnabled()

    boxes.answer = QMessageBox.StandardButton.Cancel
    top_bar.skip_button.click()
    assert app.run.state is AppState.RUNNING

    boxes.answer = QMessageBox.StandardButton.Yes
    top_bar.skip_button.click()

    assert boxes.titles() == [SLICE_SKIP_TITLE] * 2
    assert all("'second'" in text for text in boxes.texts())
    assert app.run.state is AppState.SKIPPING
    assert top_bar.view_data().state_text == "skipping"
    assert not top_bar.skip_button.isEnabled()
    held.release.set()
    wait_for_state(qtbot, app.run, AppState.PAUSED)
    assert app.session.slice == 2

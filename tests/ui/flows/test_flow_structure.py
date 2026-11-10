# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Rename, remove, duplicate, reorder, import, new workflow and reset, as a user does them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gui_testkit import TIMEOUT_MS
from gui_testkit.app import wait_for_state
from gui_testkit.driving import (
    RUNNABLE_ACTION,
    add_prepared,
    add_through_dialog,
    button,
    choose_from_context_menu,
    fail_while,
    first_spin_box,
    item_of,
    panel_of,
    prepare_with_tool,
    rename_inline,
)
from gui_testkit.workflows import simulated_workflow

from fibsem_maestro.serializer.yaml_serializer import YamlSerializer
from fibsem_maestro.ui.app.files import WorkflowFiles
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.app.session import WorkflowSession

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from gui_testkit.dialogs import DirectoryChoices, MessageBoxes
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.composition import Composition


def names(app: Composition) -> list[str]:
    return [a.name for a in app.session.actions]


def workflow_state(app: Composition) -> dict:
    """The workflow's `state.yaml` for its current slice."""
    return YamlSerializer.load(
        app.session.workflow_dir
        / "workflow"
        / f"slice_{app.session.slice:04d}"
        / "state.yaml"
    )


def settings_file(app: Composition, action: Action) -> Path:
    return (
        app.session.workflow_dir
        / action.name_with_underscores
        / f"slice_{action.ctx.slice:04d}"
        / "settings.yaml"
    )


def test_flow_rename_keeps_rules_links_and_files(
    qtbot: QtBot, app: Composition
) -> None:
    imaging = add_through_dialog(app, "imaging", "imaging")
    # autofunction is used for its link to an imaging action (an action-selector field)
    autofunction = add_through_dialog(app, "autofunction", "autofunction")
    panel_of(app, autofunction).form.set_value({"linked_imaging": "imaging"})
    rule = app.commands.add_rule(autofunction)
    app.commands.set_rule_dependents(rule, ["imaging"])
    app.window.action_list.select(imaging)

    rename_inline(qtbot, item_of(app, imaging), "img")

    assert names(app) == ["img", "autofunction"]
    assert panel_of(app, imaging).header.name_label.text() == "img"
    assert rule.dependent_names == ["img"]
    assert autofunction.settings.linked_imaging == "img"
    assert "img" in settings_file(app, autofunction).read_text()
    assert workflow_state(app)["action_names"] == ["img", "autofunction"]
    assert not (app.session.workflow_dir / "imaging").exists()
    assert settings_file(app, imaging).is_file()
    assert app.window.log_panel.source_combo.findText("img") >= 0
    # the link does not break readiness
    assert not any("img" in r and "link" in r for r in app.readiness.report.reasons)


def test_flow_remove_drops_the_rules_the_directory_and_the_panel(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = add_through_dialog(app, RUNNABLE_ACTION, "parent")
    child = add_through_dialog(app, RUNNABLE_ACTION, "child")
    own_rule = app.commands.add_rule(child)
    pointing = app.commands.add_rule(parent)
    app.commands.set_rule_dependents(pointing, ["child"])
    panel = panel_of(app, child)

    choose_from_context_menu(monkeypatch, app.window.action_list, 1, "Remove")

    assert names(app) == ["parent"]
    assert app.session.workflow.propagations.rules == [pointing]
    assert own_rule not in app.session.workflow.propagations.rules
    assert pointing.dependent_names == []
    assert not (app.session.workflow_dir / "child").exists()
    assert app.window.panel_host.panel(child) is None
    assert app.window.panel_host.currentWidget() is not panel
    assert app.window.action_list.current_action is parent


def test_flow_duplicate_copies_settings_and_outgoing_rules(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = add_through_dialog(app, RUNNABLE_ACTION, "original")
    add_through_dialog(app, RUNNABLE_ACTION, "target")
    app.window.action_list.select(original)
    first_spin_box(panel_of(app, original).form.widget).stepBy(1)
    rule = app.commands.add_rule(original)
    app.commands.set_rule_dependents(rule, ["target"])

    choose_from_context_menu(monkeypatch, app.window.action_list, 0, "Duplicate")

    copy = app.session.action_named("original 2")
    assert copy is not None
    assert names(app) == ["original", "target", "original 2"]
    assert copy.settings == original.settings
    assert copy.settings is not original.settings
    assert [r.dependent_names for r in app.session.rules_of(copy)] == [["target"]]
    assert settings_file(app, copy).read_text() == (
        settings_file(app, original).read_text()
    )
    assert app.window.action_list.current_action is original


def test_flow_reorder_by_the_list_moves_actions_and_clamps_slices(
    qtbot: QtBot, app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = add_prepared(qtbot, app, "first")
    second = add_prepared(qtbot, app, "second")
    # pause right after `first` in some slice: `second` is one slice behind
    failing = [True]
    fail_while(monkeypatch, first, lambda: failing[0])
    app.window.top_bar.run_button.click()
    qtbot.waitUntil(lambda: app.window.error_dialog is not None, timeout=TIMEOUT_MS)
    dialog = app.window.error_dialog
    assert dialog is not None
    failing[0] = False
    button(dialog, "Skip action and pause").click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)
    assert first.ctx.slice == second.ctx.slice + 1

    # what an internal drag of the second row to the top does to the list's model
    model = app.window.action_list.list_widget.model()
    assert model is not None
    root = model.index(0, 0).parent()
    model.moveRow(root, 1, root, 0)

    assert names(app) == ["second", "first"]
    assert [item_of(app, a).name_label.text() for a in app.session.actions] == [
        "second",
        "first",
    ]
    # I5: an action may not be behind the one below it
    assert second.ctx.slice == first.ctx.slice
    assert settings_file(app, second).is_file()
    assert workflow_state(app)["action_names"] == ["second", "first"]

    # the moved action now addresses a slice it was never prepared for
    run_button = app.window.top_bar.run_button
    assert not run_button.isEnabled()
    assert f"'second' is not prepared for slice {second.ctx.slice}." in (
        run_button.toolTip()
    )
    prepare_with_tool(qtbot, app, second)

    # the workflow continues consistently with the new order
    resumed_at = app.session.slice
    run_button.click()
    qtbot.waitUntil(lambda: app.session.slice > resumed_at + 1, timeout=TIMEOUT_MS)
    run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)


def test_flow_import_replaces_actions_rules_and_panels(
    tmp_path: Path,
    app: Composition,
    directory_choice: DirectoryChoices,
) -> None:
    old = add_through_dialog(app, RUNNABLE_ACTION, "shared")
    first_spin_box(panel_of(app, old).form.widget).stepBy(1)
    add_through_dialog(app, RUNNABLE_ACTION, "old only")

    source = tmp_path / "source"
    source.mkdir()
    source_session = WorkflowSession(
        simulated_workflow(source), source, WorkflowFiles()
    )
    source_session.add_action(RUNNABLE_ACTION, "shared")
    imported_child = source_session.add_action(RUNNABLE_ACTION, "imported")
    source_rule = source_session.add_rule(imported_child)
    source_session.set_rule_dependents(source_rule, ["shared"])
    source_settings = (source / "shared" / "slice_0000" / "settings.yaml").read_text()
    directory_choice.path = str(source)

    app.window.top_bar.import_button.click()

    assert names(app) == ["shared", "imported"]
    imported = app.session.action_named("shared")
    assert imported is not None and imported is not old
    # the open panel of the replaced action is gone and wrote nothing over the import
    assert app.window.panel_host.panel(old) is None
    assert settings_file(app, imported).read_text() == source_settings
    assert not (app.session.workflow_dir / "old_only").exists()
    child = app.session.action_named("imported")
    assert child is not None
    assert [r.dependent_names for r in app.session.rules_of(child)] == [["shared"]]
    assert item_of(app, child).badge.text() == "→1"


def test_flow_new_workflow_after_a_run_starts_empty_in_the_new_directory(
    qtbot: QtBot,
    tmp_path: Path,
    app: Composition,
    directory_choice: DirectoryChoices,
    boxes: MessageBoxes,
) -> None:
    old = add_prepared(qtbot, app, "old")
    app.window.action_list.select(old)
    notifier = app.session.workflow.notifier
    app.window.top_bar.run_button.click()
    app.window.top_bar.run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    directory_choice.path = str(new_dir)

    app.window.top_bar.new_button.click()

    assert boxes.calls == []
    assert app.run.state is AppState.EDITING
    assert app.session.workflow_dir == new_dir
    assert app.session.actions == ()
    assert app.session.slice == 0
    assert app.session.workflow.notifier is notifier
    assert (new_dir / "workflow" / "slice_0000" / "state.yaml").is_file()
    assert (new_dir / "workflow" / "slice_0000" / "microscope_settings.yaml").is_file()
    assert app.window.action_list.list_widget.count() == 0
    assert app.window.panel_host.panel(old) is None
    assert app.window.top_bar.directory_label.text() == str(new_dir)
    combo = app.window.log_panel.source_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["all", "workflow"]

    # the new workflow works like any other
    action = add_through_dialog(app, RUNNABLE_ACTION, "fresh")
    assert (new_dir / "fresh" / "slice_0000" / "settings.yaml").is_file()
    assert app.window.action_list.current_action is action


def test_flow_reset_after_a_run_returns_everything_to_slice_zero(
    qtbot: QtBot, app: Composition, boxes: MessageBoxes
) -> None:
    first = add_prepared(qtbot, app, "first")
    second = add_prepared(qtbot, app, "second")
    app.window.top_bar.run_button.click()
    qtbot.waitUntil(lambda: app.session.slice >= 2, timeout=TIMEOUT_MS)
    app.window.top_bar.run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)

    app.window.top_bar.reset_button.click()

    assert boxes.titles() == ["Data will be lost!"]
    assert app.run.state is AppState.EDITING
    assert (app.session.slice, first.ctx.slice, second.ctx.slice) == (0, 0, 0)
    slices = sorted(p.name for p in (app.session.workflow_dir / "first").iterdir())
    assert slices == ["slice_0000"]
    for action in (first, second):
        assert not item_of(app, action).check_mark.is_on
        assert not item_of(app, action).arrow.is_on
    run_button = app.window.top_bar.run_button
    assert not run_button.isEnabled()
    assert "'first' is not prepared for slice 0." in run_button.toolTip()
    combo = app.window.log_panel.slice_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["current", "0"]

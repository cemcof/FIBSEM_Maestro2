# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The action panel: header tools, settings form, read-only states, the panel cache and `ACTION_UI`."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest
from gui_testkit import TIMEOUT_MS
from gui_testkit.app import wait_for_state
from gui_testkit.driving import (
    RUNNABLE_ACTION,
    add_prepared,
    answering,
    click_tool,
    editable,
    first_spin_box,
    logged_messages,
    panel_of,
    prepare_with_tool,
    press,
    task_running,
)
from gui_testkit.workflows import simulated_workflow
from PyQt6.QtWidgets import QDialogButtonBox, QLabel, QMessageBox, QWidget

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.registry import Registry
from fibsem_maestro.milling.milling import MillingState
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.form_utils import AreaOverlay
from fibsem_maestro.ui.actions_ui import spec as actions_ui_spec
from fibsem_maestro.ui.actions_ui.default_tools import DEFAULT_TOOLS, TEST_TOOL
from fibsem_maestro.ui.actions_ui.spec import (
    ActionUiSpec,
    SectionContext,
    SectionSpec,
    ToolContext,
    ToolSpec,
)
from fibsem_maestro.ui.app.edit_policy import permissions_for
from fibsem_maestro.ui.app.files import WorkflowFiles
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.app.session import WorkflowSession
from fibsem_maestro.ui.forms.editors.area_select.widget import AreaSelectWidget
from fibsem_maestro.ui.views.action_panel.header import action_header_view
from fibsem_maestro.ui.widgets.icons import ICON_TEST
from fibsem_maestro.ui.widgets.properties_dialog import PropertiesDialog

if TYPE_CHECKING:
    from pathlib import Path

    from gui_testkit.dialogs import DirectoryChoices, MessageBoxes
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.composition import Composition

ANY_TYPE = min(ACTION_REGISTRY)
"""Any action type: the panel treats every type alike."""


def shown(app: Composition, action: Action) -> None:
    app.window.action_list.select(action)


def tools_enabled(app: Composition, action: Action) -> dict[str, bool]:
    header = panel_of(app, action).header
    return {tool.key: header.button(tool.key).isEnabled() for tool in DEFAULT_TOOLS}


def test_action_panel_prepare_tool_stores_props_and_enables_edit_and_apply(
    qtbot: QtBot, app: Composition
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    shown(app, action)
    assert tools_enabled(app, action) == {
        "prepare": True,
        "edit": False,
        "apply": False,
        "test": True,
    }

    click_tool(app, action, "prepare")
    assert app.tasks.busy
    qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)

    assert action.has_stored_properties()
    assert all(tools_enabled(app, action).values())
    messages = logged_messages(app.session.workflow_dir, "first", 0)
    assert "Preparing 'first'..." in messages
    assert "Preparing 'first' completed." in messages


def test_action_panel_tools_of_every_panel_wait_for_a_task(
    qtbot: QtBot, app: Composition
) -> None:
    first = app.commands.add_action(ANY_TYPE, "first")
    second = app.commands.add_action(ANY_TYPE, "second")
    shown(app, first)
    shown(app, second)

    with task_running(qtbot, app, first):
        assert not any(tools_enabled(app, first).values())
        assert not any(tools_enabled(app, second).values())

    assert tools_enabled(app, second)["prepare"]


def test_action_panel_apply_and_test_tools_run_as_tasks(
    qtbot: QtBot, app: Composition
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    prepare_with_tool(qtbot, app, action)

    click_tool(app, action, "apply")
    qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)
    click_tool(app, action, "test")
    qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)

    messages = logged_messages(app.session.workflow_dir, "first", 0)
    assert "Applying properties of 'first' completed." in messages
    # whether testing succeeds depends on the action; it ran as a task either way
    assert "Testing 'first'..." in messages


def test_action_panel_edit_tool_saves_the_reviewed_props_unless_cancelled(
    qtbot: QtBot, app: Composition
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    prepare_with_tool(qtbot, app, action)

    def cancel(dialog: PropertiesDialog) -> None:
        press(dialog, QDialogButtonBox.StandardButton.Cancel)

    def save(dialog: PropertiesDialog) -> None:
        assert dialog.windowTitle() == "Properties of 'first'"
        press(dialog, QDialogButtonBox.StandardButton.Save)

    with (
        qtbot.assertNotEmitted(app.session.action_props_changed),
        answering(PropertiesDialog, cancel) as cancelled,
    ):
        click_tool(app, action, "edit")
    with (
        qtbot.waitSignal(app.session.action_props_changed) as saved,
        answering(PropertiesDialog, save) as accepted,
    ):
        click_tool(app, action, "edit")

    assert cancelled and accepted
    assert saved.args == [action]


def test_action_panel_edit_tool_saves_edited_values(
    qtbot: QtBot, app: Composition, boxes: MessageBoxes
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    prepare_with_tool(qtbot, app, action)
    before = GlobalProperties(electron_beam=BeamProperties(working_distance=4.0e6))
    action.write_properties(before)

    def edit_and_save(dialog: PropertiesDialog) -> None:
        first_spin_box(dialog).stepBy(1)
        press(dialog, QDialogButtonBox.StandardButton.Save)

    with (
        qtbot.waitSignal(app.session.action_props_changed),
        answering(PropertiesDialog, edit_and_save),
    ):
        click_tool(app, action, "edit")

    assert boxes.calls == []
    assert action.read_properties() != before


def test_action_panel_edit_tool_refreshes_when_the_props_vanished(
    qtbot: QtBot, app: Composition
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    prepare_with_tool(qtbot, app, action)
    for props in (app.session.workflow_dir / "first").glob("slice_*/props*.yaml"):
        props.unlink()
    assert not action.has_stored_properties()

    with answering(PropertiesDialog, lambda _: None) as dialogs:
        click_tool(app, action, "edit")

    assert dialogs == []
    assert not panel_of(app, action).header.button("edit").isEnabled()
    assert not panel_of(app, action).header.button("apply").isEnabled()


def test_action_panel_settings_edit_commits_and_persists(app: Composition) -> None:
    action = app.commands.add_action(RUNNABLE_ACTION, "first")
    shown(app, action)
    stored = app.session.workflow_dir / "first" / "slice_0000" / "settings.yaml"
    before = stored.read_text()
    committed: list[object] = []
    app.session.action_settings_changed.connect(committed.append)

    first_spin_box(panel_of(app, action).form.widget).stepBy(1)

    assert committed == [action]
    assert stored.read_text() != before


def test_action_panel_settings_are_read_only_while_running_unless_danger_mode(
    qtbot: QtBot, app: Composition
) -> None:
    action = add_prepared(qtbot, app, "first")
    shown(app, action)
    panel = panel_of(app, action)
    field = first_spin_box(panel.form.widget)
    rules = panel.propagations
    assert rules is not None
    assert editable(field)

    app.window.top_bar.run_button.click()
    assert not editable(field)
    assert not rules.add_button.isEnabled()
    assert not any(tools_enabled(app, action).values())

    app.commands.set_danger_mode(True)
    assert editable(field)
    # danger mode unlocks the settings and Edit only
    assert not rules.add_button.isEnabled()
    assert tools_enabled(app, action) == {
        "prepare": False,
        "edit": True,
        "apply": False,
        "test": False,
    }

    app.commands.set_danger_mode(False)
    assert not editable(field)
    app.window.top_bar.run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)
    assert editable(field)


def test_action_panel_reports_its_form_validity_when_built(app: Composition) -> None:
    for type_key in sorted(ACTION_REGISTRY):
        action = app.commands.add_action(type_key, type_key)
        shown(app, action)

        has_errors = panel_of(app, action).form.has_errors()
        assert (action in app.validity.owners_with_errors()) == has_errors, type_key


def test_action_panel_cache_keeps_a_panel_and_its_state_across_selection(
    app: Composition,
) -> None:
    first = app.commands.add_action(ANY_TYPE, "first")
    second = app.commands.add_action(ANY_TYPE, "second")
    shown(app, first)
    panel = panel_of(app, first)
    rules = panel.propagations
    assert rules is not None
    rules.set_collapsed(True)

    shown(app, second)
    shown(app, first)

    assert panel_of(app, first) is panel
    assert app.window.panel_host.currentWidget() is panel
    rules.add_button.isVisible()


def test_action_panel_is_discarded_with_its_action(app: Composition) -> None:
    first = app.commands.add_action(ANY_TYPE, "first")
    shown(app, first)
    panel = panel_of(app, first)

    app.commands.remove_action(first)

    assert app.window.panel_host.panel(first) is None
    assert app.window.panel_host.currentWidget() is not panel
    assert first not in app.validity.owners_with_errors()


def test_action_panels_are_discarded_on_import(
    tmp_path: Path, app: Composition, directory_choice: DirectoryChoices
) -> None:
    old = app.commands.add_action(ANY_TYPE, "first")
    shown(app, old)
    source = tmp_path / "source"
    source.mkdir()
    WorkflowSession(simulated_workflow(source), source, WorkflowFiles()).add_action(
        ANY_TYPE, "first"
    )
    directory_choice.path = str(source)

    app.window.top_bar.import_button.click()

    imported = app.session.action_named("first")
    assert imported is not None and imported is not old
    assert app.window.panel_host.panel(old) is None
    shown(app, imported)
    assert panel_of(app, imported).action is imported


class _Calls:
    def __init__(self) -> None:
        self.tools: list[ToolContext] = []
        self.sections: list[tuple[str, SectionContext]] = []


@pytest.fixture
def custom_ui(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """
    `ACTION_UI` with a spec for `ANY_TYPE`: the defaults plus a viewer tool, and two sections.

    The registry is replaced for the test only, so registering leaks nowhere.
    """
    calls = _Calls()

    def view(ctx: ToolContext) -> None:
        calls.tools.append(ctx)

    def section(name: str) -> SectionSpec:
        def build(ctx: SectionContext) -> QWidget:
            calls.sections.append((name, ctx))
            return QLabel(name)

        return SectionSpec(
            key=name,
            position="after_settings" if name == "upper" else "after_propagations",
            build=build,
        )

    registry = Registry[ActionUiSpec]("action UI")
    registry.add(
        ANY_TYPE,
        ActionUiSpec(
            tools=(
                *DEFAULT_TOOLS,
                ToolSpec(
                    key="viewer",
                    icon=ICON_TEST,
                    tooltip="View",
                    accessible_name="View",
                    requires_props=False,
                    invoke=view,
                    uses_microscope=False,
                    enabled_when=lambda action: action.name != "hidden",
                ),
            ),
            sections=(section("upper"), section("lower")),
        ),
    )
    monkeypatch.setattr(actions_ui_spec, "ACTION_UI", registry)
    return calls


def test_action_panel_shows_the_registered_tool_and_sections(
    app: Composition, custom_ui: _Calls
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    shown(app, action)
    panel = panel_of(app, action)

    assert [name for name, _ in custom_ui.sections] == ["upper", "lower"]
    assert all(ctx.action is action for _, ctx in custom_ui.sections)
    container = panel.form.widget.parentWidget()
    assert container is not None
    layout = container.layout()
    assert layout is not None
    order = [
        item.widget()
        for i in range(layout.count())
        if (item := layout.itemAt(i)) is not None
    ]
    labels = {w.text(): w for w in order if isinstance(w, QLabel)}
    assert (
        order.index(panel.form.widget)
        < order.index(labels["upper"])
        < order.index(panel.propagations)
        < order.index(labels["lower"])
    )

    panel.header.button("viewer").click()
    assert [ctx.action for ctx in custom_ui.tools] == [action]
    assert custom_ui.tools[0].parent is panel


@pytest.mark.parametrize("usable", [False, True])
def test_action_header_enables_a_tool_in_danger_mode_only_if_it_says_so(
    app: Composition, usable: bool
) -> None:
    action = app.commands.add_action(ANY_TYPE, "first")
    tool = dataclasses.replace(TEST_TOOL, usable_in_danger_mode=usable)

    def enabled(state: AppState, danger: bool, busy: bool = False) -> bool:
        permissions = permissions_for(state, danger, busy)
        view = action_header_view(action, permissions, False, [tool])
        return view.tools_enabled[tool.key]

    assert enabled(AppState.RUNNING, danger=True) is usable
    assert enabled(AppState.EDITING, danger=True, busy=True) is usable
    assert not enabled(AppState.RUNNING, danger=False)
    assert enabled(AppState.EDITING, danger=False)


def test_action_panel_registered_tool_follows_its_own_conditions(
    qtbot: QtBot, app: Composition, custom_ui: _Calls
) -> None:
    _ = custom_ui
    action = app.commands.add_action(ANY_TYPE, "first")
    hidden = app.commands.add_action(ANY_TYPE, "hidden")
    other_type = next(key for key in sorted(ACTION_REGISTRY) if key != ANY_TYPE)
    other = app.commands.add_action(other_type, "other")
    for each in (action, hidden, other):
        shown(app, each)

    # it does not use the microscope, so a task does not disable it
    with task_running(qtbot, app, action):
        assert panel_of(app, action).header.button("viewer").isEnabled()
        assert not panel_of(app, action).header.button("prepare").isEnabled()

    assert not panel_of(app, hidden).header.button("viewer").isEnabled()
    # a type without a registered spec gets the default tools only
    other_header = panel_of(app, other).header
    for tool in DEFAULT_TOOLS:
        other_header.button(tool.key)
    with pytest.raises(KeyError):
        other_header.button("viewer")


def _milling_with_progress(app: Composition, slices_milled: int = 2) -> Action:
    """A milling action with one area, `slices_milled` slices into it."""
    milling = app.commands.add_action("milling", "milling")
    milling.settings.milling_area = [RelativeArea.full()]
    milling.set_state(MillingState(slices_milled=slices_milled))
    return milling


def test_action_panel_reset_state_button_resets_after_confirmation(
    qtbot: QtBot, app: Composition, boxes: MessageBoxes
) -> None:
    milling = _milling_with_progress(app)
    shown(app, milling)
    reset = panel_of(app, milling).header.reset_button
    assert reset.isEnabled()

    boxes.answer = QMessageBox.StandardButton.Cancel
    with qtbot.assertNotEmitted(app.session.action_state_changed):
        reset.click()
    assert milling.state == MillingState(slices_milled=2)

    boxes.answer = QMessageBox.StandardButton.Yes
    with qtbot.waitSignal(app.session.action_state_changed) as changed:
        reset.click()

    assert changed.args == [milling]
    assert boxes.titles() == ["Reset state", "Reset state"]
    assert milling.state == MillingState()
    assert milling.ctx.state_store.read("state.yaml", MillingState) == MillingState()
    # nothing left to reset
    assert not reset.isEnabled()


def test_action_header_reset_state_needs_state_and_an_idle_microscope(
    app: Composition,
) -> None:
    milling = _milling_with_progress(app)

    def enabled(state: AppState, danger: bool = False, busy: bool = False) -> bool:
        permissions = permissions_for(state, danger, busy)
        return action_header_view(
            milling, permissions, True, DEFAULT_TOOLS
        ).reset_enabled

    assert enabled(AppState.EDITING)
    assert enabled(AppState.PAUSED)
    # danger mode does not unlock it
    assert not enabled(AppState.RUNNING, danger=True)
    assert not enabled(AppState.EDITING, busy=True)

    milling.set_state(MillingState())
    assert not enabled(AppState.EDITING)


def test_action_panel_slice_overlay_follows_the_milling_progress(
    app: Composition,
    boxes: MessageBoxes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = boxes  # the reset is confirmed
    pushed: list[int | None] = []
    draw = AreaSelectWidget.set_overlays

    def record(area: AreaSelectWidget, overlays: Any) -> None:
        values = dict(overlays).get(AreaOverlay.SHOW_AREA_SLICES, {})
        pushed.append(values.get("slices_milled"))
        draw(area, overlays)

    monkeypatch.setattr(AreaSelectWidget, "set_overlays", record)
    milling = _milling_with_progress(app)
    shown(app, milling)
    assert pushed[-1] == 2

    # the milling ran
    milling.set_state(MillingState(slices_milled=3))
    app.run.action_finished.emit(milling)
    assert pushed[-1] == 3

    panel_of(app, milling).header.reset_button.click()
    assert pushed[-1] == 0

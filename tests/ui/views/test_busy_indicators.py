# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The building page shown while a panel is built, and the busy arc of a running tool."""

from __future__ import annotations

import dataclasses
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from gui_testkit import TIMEOUT_MS
from gui_testkit.driving import add_prepared, click_tool, panel_of
from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QDialog

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.core.registry import Registry
from fibsem_maestro.ui.actions_ui import spec as actions_ui_spec
from fibsem_maestro.ui.actions_ui.default_tools import DEFAULT_TOOLS
from fibsem_maestro.ui.actions_ui.spec import ActionUiSpec, ToolContext, ToolSpec
from fibsem_maestro.ui.views.action_panel.building_page import BuildingPage
from fibsem_maestro.ui.views.action_panel.host import ActionPanelFactory
from fibsem_maestro.ui.widgets import busy as busy_module
from fibsem_maestro.ui.widgets.busy import BUSY_COLOR, BUSY_DELAY_MS
from fibsem_maestro.ui.widgets.icon_button import IconButton
from fibsem_maestro.ui.widgets.icons import ICON_TEST
from fibsem_maestro.ui.widgets.properties_dialog import PropertiesDialog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.composition import Composition
    from fibsem_maestro.ui.views.action_panel.panel import ActionPanel

ANY_TYPE = min(ACTION_REGISTRY)
"""Any action type: the panel treats every type alike."""


def _spinning_tools(app: Composition, action: Action) -> list[str]:
    header = panel_of(app, action).header
    return [tool.key for tool in DEFAULT_TOOLS if header.button(tool.key).busy_shown]


def _send_events_for(seconds: float) -> None:
    """Keep sending events, as building widgets does, without running the event loop."""
    receiver = QObject()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.sendEvent(receiver, QEvent(QEvent.Type.User))
        time.sleep(0.005)


def _slow_create(seconds: float, fail: bool = False) -> Callable[..., ActionPanel]:
    """A panel factory's `create` that takes `seconds`, sending events meanwhile, as building does."""
    create = ActionPanelFactory.create

    def slow(self: ActionPanelFactory, action: Action) -> ActionPanel:
        _send_events_for(seconds)
        if fail:
            raise RuntimeError("cannot build")
        return create(self, action)

    return slow


@dataclasses.dataclass
class _Build:
    """What a panel build showed while it ran, sampled when it finished."""

    page_shown: bool = False
    cursor_busy: bool = False
    page_text: str = ""


def _busy_coloured(button: IconButton) -> tuple[bool, bool]:
    """
    Whether pixels close to the busy colour are drawn on the icon, and around it.

    Returns:
        (on the icon, in the margin around the icon, where the arc runs)
    """
    image = button.grab().toImage()
    busy = QColor(BUSY_COLOR)
    # the 18 px icon is centred in the 30 px button
    icon = range(7, 23)
    found = [False, False]
    for x in range(image.width()):
        for y in range(image.height()):
            pixel = image.pixelColor(x, y)
            distance = (
                abs(pixel.red() - busy.red())
                + abs(pixel.green() - busy.green())
                + abs(pixel.blue() - busy.blue())
            )
            if distance < 30:
                found[0 if x in icon and y in icon else 1] = True
    return found[0], found[1]


@pytest.fixture
def held_test(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Action, threading.Event]]:
    """An action whose Test runs until the event is set."""
    release = threading.Event()
    action = app.commands.add_action(ANY_TYPE, "held")
    monkeypatch.setattr(action, "test", lambda: release.wait(TIMEOUT_MS / 1000))
    yield action, release
    release.set()


def test_running_tool_spins_once_its_task_is_slow_until_it_ends(
    qtbot: QtBot, app: Composition, held_test: tuple[Action, threading.Event]
) -> None:
    action, release = held_test
    app.window.action_list.select(action)

    click_tool(app, action, "test")

    assert _spinning_tools(app, action) == []
    qtbot.waitUntil(lambda: _spinning_tools(app, action) == ["test"], timeout=1000)
    release.set()
    qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)
    assert _spinning_tools(app, action) == []


def test_tool_of_a_quick_task_never_spins(
    qtbot: QtBot, app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    action = app.commands.add_action(ANY_TYPE, "quick")
    monkeypatch.setattr(action, "test", lambda: None)
    app.window.action_list.select(action)

    click_tool(app, action, "test")
    qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)
    # past the delay after which a running task would show
    qtbot.wait(2 * BUSY_DELAY_MS)

    assert _spinning_tools(app, action) == []


def test_tool_starting_no_task_does_not_spin(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry[ActionUiSpec]("action UI")
    registry.add(
        ANY_TYPE,
        ActionUiSpec(
            tools=(
                ToolSpec(
                    key="nothing",
                    icon=ICON_TEST,
                    tooltip="Nothing",
                    accessible_name="Nothing",
                    requires_props=False,
                    invoke=lambda _: None,
                    uses_microscope=False,
                ),
            ),
        ),
    )
    monkeypatch.setattr(actions_ui_spec, "ACTION_UI", registry)
    action = app.commands.add_action(ANY_TYPE, "first")
    app.window.action_list.select(action)

    click_tool(app, action, "nothing")

    assert not panel_of(app, action).header.button("nothing").busy


@dataclasses.dataclass
class _Seen:
    """Whether a tool showed itself busy, sampled at points of its work."""

    while_working: bool | None = None
    in_dialog: bool | None = None
    after_dialog: bool | None = None
    angles: set[float] = dataclasses.field(default_factory=set)


def _blocking_tool(
    app: Composition,
    monkeypatch: pytest.MonkeyPatch,
    work: Callable[[IconButton, ToolContext], None],
) -> tuple[Action, IconButton]:
    """An action whose only tool does `work` on the GUI thread; `work` gets the tool's button."""
    buttons: list[IconButton] = []

    def invoke(ctx: ToolContext) -> None:
        work(buttons[0], ctx)

    registry = Registry[ActionUiSpec]("action UI")
    registry.add(
        ANY_TYPE,
        ActionUiSpec(
            tools=(
                ToolSpec(
                    key="work",
                    icon=ICON_TEST,
                    tooltip="Work",
                    accessible_name="Work",
                    requires_props=False,
                    invoke=invoke,
                    uses_microscope=False,
                ),
            ),
        ),
    )
    monkeypatch.setattr(actions_ui_spec, "ACTION_UI", registry)
    action = app.commands.add_action(ANY_TYPE, "first")
    app.window.action_list.select(action)
    buttons.append(panel_of(app, action).header.button("work"))
    return action, buttons[0]


@pytest.mark.parametrize(
    ("seconds", "spins"), [(0.0, False), (3 * BUSY_DELAY_MS / 1000, True)]
)
def test_tool_working_on_the_gui_thread_spins_while_slow(
    app: Composition, monkeypatch: pytest.MonkeyPatch, seconds: float, spins: bool
) -> None:
    seen = _Seen()

    def work(button: IconButton, _: ToolContext) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            _send_events_for(0.05)
            if button.busy_shown:
                seen.angles.add(button.busy_arc_angle)
        seen.while_working = button.busy_shown

    action, button = _blocking_tool(app, monkeypatch, work)

    click_tool(app, action, "work")

    assert seen.while_working is spins
    # the arc turns
    assert (len(seen.angles) > 1) is spins
    assert not button.busy_shown


def test_tool_stops_spinning_when_it_opens_a_dialog(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _Seen()

    def work(button: IconButton, _: ToolContext) -> None:
        _send_events_for(3 * BUSY_DELAY_MS / 1000)
        seen.while_working = button.busy_shown
        dialog = QDialog()

        def look() -> None:
            seen.in_dialog = button.busy_shown
            dialog.reject()

        QTimer.singleShot(BUSY_DELAY_MS, look)
        dialog.exec()
        # e.g. saving what the dialog returned
        _send_events_for(3 * BUSY_DELAY_MS / 1000)
        seen.after_dialog = button.busy_shown

    action, button = _blocking_tool(app, monkeypatch, work)

    click_tool(app, action, "work")

    assert (seen.while_working, seen.in_dialog, seen.after_dialog) == (
        True,
        False,
        False,
    )
    assert not button.busy_shown


def test_tool_spinning_while_slow_keeps_spinning_for_the_task_it_starts(
    qtbot: QtBot, app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def work(_: IconButton, ctx: ToolContext) -> None:
        _send_events_for(3 * BUSY_DELAY_MS / 1000)
        ctx.commands.run_microscope_task(
            ctx.action, "Waiting", lambda: release.wait(TIMEOUT_MS / 1000)
        )

    action, button = _blocking_tool(app, monkeypatch, work)

    try:
        click_tool(app, action, "work")
        spinning = button.busy_shown
    finally:
        release.set()
        qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)

    assert spinning
    assert not button.busy_shown


def test_edit_tool_spins_while_its_dialog_is_slow_to_open(
    qtbot: QtBot, app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    action = add_prepared(qtbot, app, "first")
    seen = _Seen()

    def slow_review(*_: object, **__: object) -> None:
        _send_events_for(3 * BUSY_DELAY_MS / 1000)
        seen.while_working = panel_of(app, action).header.button("edit").busy_shown

    monkeypatch.setattr(PropertiesDialog, "review", slow_review)

    click_tool(app, action, "edit")

    assert seen.while_working
    assert not panel_of(app, action).header.button("edit").busy_shown


def test_busy_button_has_a_busy_icon_and_an_arc_around_it(qtbot: QtBot) -> None:
    button = IconButton(ICON_TEST, "Test", "Test")
    qtbot.addWidget(button)
    # the tools are disabled while a task runs
    button.setEnabled(False)

    button.set_busy(True)
    qtbot.waitUntil(lambda: button.busy_shown, timeout=1000)
    busy = _busy_coloured(button)
    button.set_busy(False)

    assert busy == (True, True)
    assert _busy_coloured(button) == (False, False)


def _watch_build(
    app: Composition,
    monkeypatch: pytest.MonkeyPatch,
    create: Callable[..., ActionPanel],
) -> _Build:
    """Make panels be built by `create`, recording what the build showed when it ended."""
    host = app.window.panel_host
    seen = _Build()

    def watched(self: ActionPanelFactory, action: Action) -> ActionPanel:
        try:
            return create(self, action)
        finally:
            seen.page_shown = host.currentWidget() is host.building_page
            seen.cursor_busy = QApplication.overrideCursor() is not None
            seen.page_text = host.building_page.label.text()

    monkeypatch.setattr(ActionPanelFactory, "create", watched)
    return seen


def test_quick_panel_build_shows_no_building_page(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    # however long the real build takes here, it ends within the delay
    monkeypatch.setattr(busy_module, "time", SimpleNamespace(monotonic=lambda: 0.0))
    seen = _watch_build(app, monkeypatch, ActionPanelFactory.create)
    action = app.commands.add_action(ANY_TYPE, "first")

    app.window.action_list.select(action)

    assert not seen.page_shown
    assert not seen.cursor_busy
    assert app.window.panel_host.currentWidget() is panel_of(app, action)


def test_slow_panel_build_shows_the_building_page_turning(
    app: Composition, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = app.window.panel_host
    frames: list[float] = []
    advance = BuildingPage.advance

    def counted(self: BuildingPage) -> None:
        advance(self)
        frames.append(self.arc_angle)

    monkeypatch.setattr(BuildingPage, "advance", counted)
    seen = _watch_build(app, monkeypatch, _slow_create(3 * BUSY_DELAY_MS / 1000))
    action = app.commands.add_action(ANY_TYPE, "first")

    app.window.action_list.select(action)

    assert seen.page_shown
    assert seen.cursor_busy
    assert "first" in seen.page_text
    assert len(set(frames)) > 1
    assert host.currentWidget() is panel_of(app, action)
    assert QApplication.overrideCursor() is None
    # the page no longer turns once the panel is built
    built = len(frames)
    _send_events_for(BUSY_DELAY_MS / 1000)
    assert len(frames) == built


@pytest.mark.parametrize("seconds", [0.0, 3 * BUSY_DELAY_MS / 1000])
def test_failed_panel_build_leaves_the_previous_panel_shown(
    app: Composition, monkeypatch: pytest.MonkeyPatch, seconds: float
) -> None:
    host = app.window.panel_host
    first = app.commands.add_action(ANY_TYPE, "first")
    app.window.action_list.select(first)
    second = app.commands.add_action(ANY_TYPE, "second")
    monkeypatch.setattr(ActionPanelFactory, "create", _slow_create(seconds, fail=True))

    with pytest.raises(RuntimeError):
        host.show_action(second)

    assert host.currentWidget() is panel_of(app, first)
    assert QApplication.overrideCursor() is None

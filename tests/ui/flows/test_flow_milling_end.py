# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
The run ends at the end of the milling area, and continues once the area is enlarged or the state reset.

This flow belongs to `Milling`, the reference action that ends an acquisition
(`completion_reason`), so it uses that action and its settings by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gui_testkit import TIMEOUT_MS
from gui_testkit.app import wait_for_state
from gui_testkit.driving import (
    RUNNABLE_ACTION,
    add_through_dialog,
    button,
    children,
    logged_messages,
    panel_of,
    prepare_with_tool,
)

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.forms.editors.area_select.widget import AreaSelectWidget

if TYPE_CHECKING:
    from gui_testkit.dialogs import MessageBoxes
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.composition import Composition

USED_UP = "the milling area is used up"


def area(y: float, height: float = 0.2) -> RelativeArea:
    # the simulated frame is 1500 nm high: 0.2 of it holds a few 100 nm slices
    return RelativeArea(origin=RelativePoint(0.1, y), width=0.5, height=height)


def place_area(
    qtbot: QtBot, app: Composition, milling: Action, y: float, height: float = 0.2
) -> None:
    """
    Put one area into the milling area selector and Load an image under it.

    The selector commits its areas when an image is loaded or an area is drawn
    with the mouse; showing a value alone commits nothing.
    """
    panel = panel_of(app, milling)
    panel.form.set_value({"milling_area": [area(y, height)]})
    (selector,) = children(panel, AreaSelectWidget)
    button(selector, "Load").click()
    qtbot.waitUntil(lambda: not app.tasks.busy, timeout=TIMEOUT_MS)


def set_up_milling(qtbot: QtBot, app: Composition) -> Action:
    """Add a milling action with one small area and prepare it."""
    milling = add_through_dialog(app, "milling", "milling")
    panel_of(app, milling).form.set_value(
        {
            "pattern_type": app.form_contexts.for_action_settings().pattern_types[-1],
            "slice_distance": 100.0,
        }
    )
    place_area(qtbot, app, milling, 0.1)
    (placed,) = milling.settings.milling_area
    assert placed.origin.y == pytest.approx(0.1)
    prepare_with_tool(qtbot, app, milling)
    return milling


def run_to_the_end(qtbot: QtBot, app: Composition) -> None:
    """Run the workflow until the milling area is used up."""
    run_button = app.window.top_bar.run_button
    assert run_button.isEnabled(), run_button.toolTip()
    run_button.click()
    wait_for_state(qtbot, app.run, AppState.FINISHED)
    assert not run_button.isEnabled()
    assert USED_UP in run_button.toolTip()


def test_flow_milling_end_finishes_after_the_last_slice_and_continues_on_an_enlarged_area(
    qtbot: QtBot, app: Composition
) -> None:
    milling = set_up_milling(qtbot, app)
    after = add_through_dialog(app, RUNNABLE_ACTION, "after")
    prepare_with_tool(qtbot, app, after)
    run_button = app.window.top_bar.run_button

    run_to_the_end(qtbot, app)

    finished_at = app.session.slice
    last = finished_at - 1
    assert last >= 1
    # the slice that used the area up was completed, imaging-like actions included
    assert after.ctx.slice == finished_at
    assert f"Completed 'after' for slice {last}." in logged_messages(
        app.session.workflow_dir, "after", last
    )
    assert app.window.error_dialog is None

    # moving the area keeps its progress, so it stays used up
    app.window.action_list.select(milling)
    place_area(qtbot, app, milling, 0.5)
    assert not run_button.isEnabled()

    # enlarging it lifts the stop: milling goes on where it stopped
    place_area(qtbot, app, milling, 0.5, height=0.4)
    assert run_button.isEnabled()
    assert USED_UP not in run_button.toolTip()

    run_to_the_end(qtbot, app)
    assert app.session.slice > finished_at


def test_flow_milling_end_is_lifted_by_resetting_the_state(
    qtbot: QtBot, app: Composition, boxes: MessageBoxes
) -> None:
    milling = set_up_milling(qtbot, app)
    run_to_the_end(qtbot, app)
    reset = panel_of(app, milling).header.reset_button
    assert reset.isEnabled()

    reset.click()

    assert boxes.titles() == ["Reset state"]
    assert milling.state == milling.state_cls()()
    assert not reset.isEnabled()
    assert app.window.top_bar.run_button.isEnabled()


def test_flow_milling_end_is_lifted_only_by_a_setting_defining_the_slices(
    qtbot: QtBot, app: Composition
) -> None:
    milling = set_up_milling(qtbot, app)
    run_to_the_end(qtbot, app)
    run_button = app.window.top_bar.run_button

    panel_of(app, milling).form.set_value({"milling_depth": 5.0})
    assert not run_button.isEnabled()

    # a new slice distance restarts milling at the trailing edge
    panel_of(app, milling).form.set_value({"slice_distance": 50.0})
    assert run_button.isEnabled()

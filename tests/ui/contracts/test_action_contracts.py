# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Properties every registered action must have in the GUI.

Each test runs for every key of `ACTION_REGISTRY`, on the action's default
settings, so a new action is covered without any change here. Nothing below
may name a field of a real settings class.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from gui_testkit.forms import children, editable_controls, warnings_of

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.ui.actions_ui.default_tools import DEFAULT_TOOLS
from fibsem_maestro.ui.actions_ui.spec import ACTION_UI, ui_spec_for
from fibsem_maestro.ui.forms.editors.object import ObjectWidget
from fibsem_maestro.ui.forms.issues import Severity
from fibsem_maestro.ui.schema.schema import get_field_infos
from fibsem_maestro.ui.widgets.icon_button import IconButton

if TYPE_CHECKING:
    from collections.abc import Callable

    from gui_testkit.app import Services
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.logging.text.memory import MemoryTextLogger
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.composition import Composition, FormContexts
    from fibsem_maestro.ui.forms.builder import Form, FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext

ACTION_KEYS = sorted(ACTION_REGISTRY)
"""Every registered action type."""

per_action = pytest.mark.parametrize("key", ACTION_KEYS)

NESTED_NOTE = "Contains an invalid value."
"""What a field shows when the error is in one of its nested fields."""


@pytest.fixture
def settings_form(
    qtbot: QtBot,
    services: Services,
    form_builder: FormBuilder,
    form_contexts: FormContexts,
    recorded: Callable[[FormContext], FormContext],
) -> Callable[[str], tuple[Action, Form]]:
    """Add an action of a type and build its settings form as its panel does."""

    def build(key: str) -> tuple[Action, Form]:
        action = services.session.add_action(key, f"{key} action")
        form = form_builder.build_live(
            action.settings, recorded(form_contexts.for_action_settings())
        )
        qtbot.addWidget(form.widget)
        return action, form

    return build


def _shown_without_error(form: Form, settings: Any) -> dict[str, Any]:
    """The values the form shows for the top-level fields that show no error."""
    shown: dict[str, Any] = {}
    for fi in get_field_infos(type(settings)):
        editor = form.widget.field_widget(fi.name)
        assert editor is not None
        if all(issue.severity is not Severity.ERROR for issue in editor.issues()):
            shown[fi.name] = editor.get_value()
    return shown


def _form_errors(form: Form) -> list[str]:
    """
    The error texts a form shows, without the notes pointing at nested errors.

    Every field of every object editor is read once (an optional wrapper
    reports its gated editor's issues).
    """
    objects = [form.widget, *children(form.widget, ObjectWidget)]
    return [
        issue.message
        for obj in objects
        for editor in obj.child_widgets()
        for issue in editor.issues()
        if issue.severity is Severity.ERROR and issue.message != NESTED_NOTE
    ]


def test_actions_are_registered() -> None:
    assert ACTION_KEYS


@per_action
def test_settings_form_builds_without_warnings(
    settings_form: Callable[[str], tuple[Action, Form]],
    log: MemoryTextLogger,
    key: str,
) -> None:
    settings_form(key)

    assert warnings_of(log) == []


@per_action
def test_settings_form_shows_the_live_settings(
    settings_form: Callable[[str], tuple[Action, Form]], key: str
) -> None:
    action, form = settings_form(key)
    settings = action.settings

    shown = _shown_without_error(form, settings)

    assert shown == {name: getattr(settings, name) for name in shown}


@per_action
def test_settings_form_round_trip_preserves_the_settings(
    settings_form: Callable[[str], tuple[Action, Form]], key: str
) -> None:
    action, form = settings_form(key)
    before = copy.deepcopy(action.settings)

    form.set_value(_shown_without_error(form, action.settings))
    form.set_value(copy.deepcopy(action.settings))

    assert action.settings == before


@per_action
def test_read_only_settings_form_leaves_nothing_editable(
    settings_form: Callable[[str], tuple[Action, Form]], key: str
) -> None:
    _, form = settings_form(key)
    assert editable_controls(form.widget)

    form.set_read_only(True)

    assert editable_controls(form.widget) == []


@per_action
def test_default_settings_audit_reports_what_the_fresh_form_shows(
    settings_form: Callable[[str], tuple[Action, Form]],
    services: Services,
    key: str,
) -> None:
    action, form = settings_form(key)

    issues = services.audit.issues_of(action)

    assert sorted(issue.message for issue in issues) == sorted(_form_errors(form))
    assert bool(issues) == form.has_errors()


@per_action
def test_action_lifecycle_through_the_session_keeps_its_files(
    session: WorkflowSession, key: str
) -> None:
    action = session.add_action(key, "original")
    assert action.ctx.settings_store.read("settings.yaml", type(action.settings)) == (
        action.settings
    )

    copied = session.duplicate_action(action)
    assert type(copied) is type(action)
    assert copied.settings == action.settings
    assert copied.settings is not action.settings
    assert copied.ctx.settings_store.exists("settings.yaml")

    old_dir = action.ctx.path_to_dir
    session.rename_action(action, "renamed action")
    assert action.name == "renamed action"
    assert old_dir is not None and not old_dir.exists()
    assert action.ctx.settings_store.read("settings.yaml", type(action.settings)) == (
        action.settings
    )

    renamed_dir = action.ctx.path_to_dir
    assert session.remove_action(action) is None
    assert action not in session.actions
    assert renamed_dir is not None and not renamed_dir.exists()
    assert [a.name for a in session.actions] == [copied.name]


@per_action
def test_action_panel_builds_in_the_main_window(app: Composition, key: str) -> None:
    action = app.session.add_action(key, f"{key} action")

    app.window.action_list.select(action)

    panel = app.window.panel_host.panel(action)
    assert panel is not None
    assert app.window.panel_host.currentWidget() is panel
    assert panel.action is action


@per_action
def test_action_panel_shows_its_tools(app: Composition, key: str) -> None:
    action = app.session.add_action(key, f"{key} action")
    app.window.action_list.select(action)
    panel = app.window.panel_host.panel(action)
    assert panel is not None
    tools = [tool.key for tool in ui_spec_for(action).tools]
    if key not in ACTION_UI:
        assert tools == [tool.key for tool in DEFAULT_TOOLS]

    buttons = children(panel.header.frame, IconButton)

    # the tools, then Reset state apart at the right end
    tool_buttons = [panel.header.button(tool) for tool in tools]
    assert [*tool_buttons, panel.header.reset_button] == buttons

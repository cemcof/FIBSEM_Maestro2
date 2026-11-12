# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The propagation rules editor of an action panel, against the session's rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gui_testkit.app import wait_for_state
from gui_testkit.driving import (
    add_prepared,
    button,
    children,
    item_of,
    panel_of,
    rename_inline,
)
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QListWidget

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.forms.editors.field_label import FieldLabel
from fibsem_maestro.ui.forms.editors.multi_select import MultiSelectWidget

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.composition import Composition
    from fibsem_maestro.ui.views.action_panel.propagations import PropagationsEditor
    from fibsem_maestro.ui.views.action_panel.rule_editor import RuleEditor

ANY_TYPE = min(ACTION_REGISTRY)
"""Any action type: rules work alike for every type."""


def rules_editor(app: Composition, action: Action) -> PropagationsEditor:
    app.window.action_list.select(action)
    editor = panel_of(app, action).propagations
    assert editor is not None
    return editor


def dependents_list(editor: RuleEditor) -> QListWidget:
    """The rule's "dependents" list (the property selectors are `MultiSelectWidget`s)."""
    (found,) = [w for w in children(editor, QListWidget) if type(w) is QListWidget]
    return found


def listed(editor: RuleEditor) -> dict[str, bool]:
    """The dependents offered, and whether each is selected."""
    widget = dependents_list(editor)
    items = [widget.item(row) for row in range(widget.count())]
    return {item.text(): item.isSelected() for item in items if item is not None}


def select_dependent(editor: RuleEditor, name: str) -> None:
    (item,) = dependents_list(editor).findItems(name, Qt.MatchFlag.MatchExactly)
    item.setSelected(True)


def collapse(editor: RuleEditor) -> None:
    """Click the chevron on the rule card's title line."""
    editor.toggle.clicked.emit()


def test_propagations_add_rule_shows_an_empty_rule_that_stays(app: Composition) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    app.commands.add_action(ANY_TYPE, "child")
    editor = rules_editor(app, parent)

    editor.add_button.click()

    (rule,) = app.session.rules_of(parent)
    (rule_editor,) = editor.rule_editors
    assert rule_editor.rule is rule
    assert rule.dependent_names == []
    assert listed(rule_editor) == {"child": False}

    # V3: an empty rule survives later structure changes
    app.commands.add_action(ANY_TYPE, "another")
    assert app.session.rules_of(parent) == [rule]
    assert editor.rule_editors == (rule_editor,)
    assert listed(rule_editor) == {"child": False, "another": False}


def test_propagations_dependents_selection_reaches_the_session(
    app: Composition,
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    app.commands.add_action(ANY_TYPE, "child")
    editor = rules_editor(app, parent)
    editor.add_button.click()

    select_dependent(editor.rule_editors[0], "child")

    (rule,) = app.session.rules_of(parent)
    assert rule.dependent_names == ["child"]
    assert item_of(app, parent).badge.text() == "→1"


def test_propagations_rule_card_is_titled_by_its_dependents(app: Composition) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    app.commands.add_action(ANY_TYPE, "child")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    (rule_editor,) = editor.rule_editors
    assert "child" not in rule_editor.title_label.text()

    select_dependent(rule_editor, "child")

    title = rule_editor.title_label.text()
    assert "parent" in title
    assert "child" in title


def test_propagations_dependents_label_starts_in_line_with_the_properties_title(
    app: Composition, qtbot: QtBot
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    (rule_editor,) = editor.rule_editors
    app.window.show()
    qtbot.waitExposed(app.window)
    qtbot.wait(20)
    assert rule_editor.isVisible()
    dependents = next(
        label
        for label in children(rule_editor, FieldLabel)
        if label.text() == "dependents"
    )
    properties = next(
        label
        for label in children(rule_editor, FieldLabel)
        if label.text() == "properties"
    )

    def text_left(label: FieldLabel) -> int:
        return label.mapTo(rule_editor, QPoint(label.contentsRect().left(), 0)).x()

    # right-aligned, it sat far right of the title below it
    assert text_left(dependents) == text_left(properties)


def test_propagations_properties_selection_reaches_the_session(
    app: Composition,
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    (rule_editor,) = editor.rule_editors
    selector = children(rule_editor, MultiSelectWidget)[0]
    chosen = selector.item(0)
    assert chosen is not None

    chosen.setSelected(True)

    (rule,) = app.session.rules_of(parent)
    assert rule.props_to_propagate is not None
    selected = rule.props_to_propagate.model_dump().values()
    assert [names for names in selected if names] == [[chosen.text()]]


def test_propagations_remove_button_deletes_the_rule(app: Composition) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    editor.add_button.click()
    first, second = editor.rule_editors

    button(first, "×").click()

    assert app.session.rules_of(parent) == [second.rule]
    assert editor.rule_editors == (second,)


def test_propagations_reconcile_by_identity_keeps_editors_and_their_collapse(
    app: Composition,
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    (first,) = editor.rule_editors
    collapse(first)
    assert first.collapsed

    editor.add_button.click()

    assert editor.rule_editors[0] is first
    assert first.collapsed
    assert not editor.rule_editors[1].collapsed


def test_propagations_dependents_follow_a_rename(
    qtbot: QtBot, app: Composition
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    child = app.commands.add_action(ANY_TYPE, "child")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    select_dependent(editor.rule_editors[0], "child")

    rename_inline(qtbot, item_of(app, child), "renamed")

    (rule,) = app.session.rules_of(parent)
    assert rule.dependent_names == ["renamed"]
    assert listed(editor.rule_editors[0]) == {"renamed": True}


def test_propagations_rules_survive_a_rename_of_their_parent(
    qtbot: QtBot, app: Composition
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    app.commands.add_action(ANY_TYPE, "child")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    select_dependent(editor.rule_editors[0], "child")
    rule_editor = editor.rule_editors[0]

    rename_inline(qtbot, item_of(app, parent), "renamed")

    assert [r.dependent_names for r in app.session.rules_of(parent)] == [["child"]]
    assert editor.rule_editors == (rule_editor,)


def test_propagations_removed_dependent_leaves_an_empty_rule(
    app: Composition,
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    child = app.commands.add_action(ANY_TYPE, "child")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    select_dependent(editor.rule_editors[0], "child")

    app.commands.remove_action(child)

    (rule,) = app.session.rules_of(parent)
    assert rule.dependent_names == []
    assert listed(editor.rule_editors[0]) == {}


def test_propagations_are_read_only_while_running(
    qtbot: QtBot, app: Composition
) -> None:
    parent = add_prepared(qtbot, app, "parent")
    editor = rules_editor(app, parent)
    editor.add_button.click()
    (rule_editor,) = editor.rule_editors

    app.window.top_bar.run_button.click()
    assert not editor.add_button.isEnabled()
    assert not dependents_list(rule_editor).isEnabled()
    assert not button(rule_editor, "×").isEnabled()

    app.window.top_bar.run_button.click()
    wait_for_state(qtbot, app.run, AppState.PAUSED)
    assert editor.add_button.isEnabled()
    assert dependents_list(rule_editor).isEnabled()


def test_propagations_leave_out_an_action_that_takes_no_part(
    app: Composition,
) -> None:
    parent = app.commands.add_action(ANY_TYPE, "parent")
    refocus = app.commands.add_action("refocus", "refocus")
    editor = rules_editor(app, parent)

    editor.add_button.click()

    (rule_editor,) = editor.rule_editors
    assert "refocus" not in listed(rule_editor)
    app.window.action_list.select(refocus)
    assert panel_of(app, refocus).propagations is None
    with pytest.raises(ValueError, match="takes no part in propagation"):
        app.session.add_rule(refocus)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Building live and detached forms, where their edits are committed, and disposing them."""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest
from gui_testkit.forms import (
    FakePreviews,
    button,
    children,
    editor_at,
    field_at,
    messages,
    optional_checkbox,
)
from gui_testkit.sample_settings import (
    KitchenSink,
    Leaf,
    Optionals,
    PreviewSettings,
    SquareShape,
    filled_kitchen_sink,
)
from PyQt6 import sip

from fibsem_maestro.core.image import Image
from fibsem_maestro.ui.forms.context import FormContext
from fibsem_maestro.ui.forms.editors.field_label import FieldLabel
from fibsem_maestro.ui.forms.editors.object import (
    RESETS_STATE_MARK_HTML,
    RESETS_STATE_NOTE,
    ObjectWidget,
)
from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
from fibsem_maestro.ui.schema.schema import get_field_infos

if TYPE_CHECKING:
    from collections.abc import Callable

    from PyQt6.QtWidgets import QWidget
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.ui.forms.builder import Form, FormBuilder


def _row_labels(form_widget: QWidget) -> list[tuple[str, str]]:
    """`(text, tooltip)` of the field labels of the form's root object, top to bottom."""
    found = children(form_widget, FieldLabel, direct_only=True)
    return [(label.text(), label.toolTip()) for label in found]


def test_live_form_shows_the_live_values(live_form: Callable[..., Form]) -> None:
    settings = filled_kitchen_sink()
    expected = copy.deepcopy(settings)

    form = live_form(settings)

    assert form.widget.get_value() == expected
    assert not form.has_errors()


def test_live_form_build_writes_nothing(live_form: Callable[..., Form]) -> None:
    settings = filled_kitchen_sink()
    expected = copy.deepcopy(settings)
    commits: list[str] = []

    live_form(settings, on_commit=commits.append)

    assert commits == []
    assert settings == expected


def test_detached_form_shows_the_declared_defaults(
    qtbot: QtBot, form_builder: FormBuilder, sink_context: FormContext
) -> None:
    form = form_builder.build_detached(KitchenSink, sink_context)
    qtbot.addWidget(form.widget)

    assert form.value() == KitchenSink()


def test_detached_form_round_trips_a_value(
    qtbot: QtBot, form_builder: FormBuilder, sink_context: FormContext
) -> None:
    form = form_builder.build_detached(KitchenSink, sink_context)
    qtbot.addWidget(form.widget)

    form.set_value(filled_kitchen_sink())

    assert form.value() == filled_kitchen_sink()


def test_detached_form_emits_changed_on_nested_and_list_item_edits(
    qtbot: QtBot, form_builder: FormBuilder, sink_context: FormContext
) -> None:
    form = form_builder.build_detached(KitchenSink, sink_context)
    qtbot.addWidget(form.widget)
    form.set_value(filled_kitchen_sink())

    with qtbot.waitSignal(form.changed, timeout=1000):
        editor_at(form.widget, "leaf.count").set_value(9)
    leaves = editor_at(form.widget, "leaves")
    with qtbot.waitSignal(form.changed, timeout=1000):
        leaves.set_value([Leaf(count=8)])

    assert form.value().leaf.count == 9
    assert form.value().leaves == [Leaf(count=8)]


def test_form_rows_follow_declaration_order_with_labels_and_tooltips(
    live_form: Callable[..., Form],
) -> None:
    form = live_form(KitchenSink())

    assert _row_labels(form.widget) == [
        (fi.name.replace("_", " "), fi.description)
        for fi in get_field_infos(KitchenSink)
    ]


def test_form_marks_the_top_level_fields_resetting_the_state(
    live_form: Callable[..., Form], sink_context: FormContext
) -> None:
    ctx = dataclasses.replace(sink_context, state_fields=frozenset({"count", "leaf"}))

    form = live_form(KitchenSink(), ctx=ctx)

    rows = {
        fi.name: row
        for fi, row in zip(get_field_infos(KitchenSink), _row_labels(form.widget))
    }
    marked = {
        name for name, (text, _) in rows.items() if RESETS_STATE_MARK_HTML in text
    }
    assert marked == {"count", "leaf"}
    text, tooltip = rows["count"]
    assert text.startswith("count ")
    assert (
        tooltip
        == f"{KitchenSink.model_fields['count'].description}\n\n{RESETS_STATE_NOTE}"
    )
    # a nested field of the same name defines nothing
    leaf = editor_at(form.widget, "leaf").value_editor()
    assert isinstance(leaf, ObjectWidget)
    assert all(RESETS_STATE_MARK_HTML not in text for text, _ in _row_labels(leaf))


def test_form_field_subset_keeps_declaration_order(
    live_form: Callable[..., Form],
) -> None:
    form = live_form(KitchenSink(), fields=["title", "flag"])

    assert [text for text, _ in _row_labels(form.widget)] == ["flag", "title"]
    assert form.widget.field_widget("count") is None


def test_subset_form_commits_only_its_fields(live_form: Callable[..., Form]) -> None:
    settings = KitchenSink()
    form = live_form(settings, fields=["count"])

    form.set_value(filled_kitchen_sink().model_dump())

    assert settings.count == filled_kitchen_sink().count
    assert settings.title == KitchenSink().title


def test_live_form_commits_a_top_level_edit_with_its_path(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink()
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)

    editor_at(form.widget, "count").set_value(42)

    assert settings.count == 42
    assert commits == ["count"]


def test_live_form_writes_into_a_nested_live_object_in_place(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink()
    leaf = settings.leaf
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)

    editor_at(form.widget, "leaf.count").set_value(6)

    assert settings.leaf is leaf
    assert leaf.count == 6
    assert commits == ["leaf.count"]


def test_union_page_edit_commits_the_whole_union(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink()
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)
    union = editor_at(form.widget, "shape")
    assert isinstance(union, DiscriminatedUnionWidget)

    union.select("square")
    editor_at(form.widget, "shape.side").set_value(5.0)

    assert settings.shape == SquareShape(side=5.0)
    assert commits == ["shape", "shape"]


def test_list_item_edit_commits_the_whole_list(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink(leaves=[Leaf(), Leaf()])
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)
    items = children(editor_at(form.widget, "leaves"), ObjectWidget)
    assert len(items) == 2

    editor_at(items[1], "count").set_value(4)

    assert settings.leaves == [Leaf(), Leaf(count=4)]
    assert commits == ["leaves"]


def test_unset_optional_group_commits_the_whole_group(
    live_form: Callable[..., Form],
) -> None:
    settings = Optionals()
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)

    optional_checkbox(field_at(form.widget, "extra")).click()
    first = settings.extra
    editor_at(form.widget, "extra.label").set_value("typed")

    assert first == Leaf()
    assert settings.extra == Leaf(label="typed")
    assert commits == ["extra", "extra"]


def test_form_set_value_commits_each_changed_field(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink()
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)

    form.set_value({"count": 9, "title": "set"})

    assert (settings.count, settings.title) == (9, "set")
    assert sorted(commits) == ["count", "title"]


def test_nested_edits_reach_the_group_after_untick_and_retick_d16(
    live_form: Callable[..., Form],
) -> None:
    settings = Optionals(extra=Leaf())
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)
    checkbox = optional_checkbox(field_at(form.widget, "extra"))

    checkbox.click()
    assert settings.extra is None
    # nothing to write to while the group is unticked
    editor_at(form.widget, "extra.label").set_value("while unticked")
    assert settings.extra is None

    checkbox.click()
    assert settings.extra == Leaf(label="while unticked")
    editor_at(form.widget, "extra.count").set_value(5)

    assert settings.extra == Leaf(count=5, label="while unticked")
    assert commits[-1] == "extra.count"


def test_disposed_form_neither_writes_nor_reports_validity(
    qtbot: QtBot, live_form: Callable[..., Form]
) -> None:
    settings = KitchenSink()
    commits: list[str] = []
    form = live_form(settings, on_commit=commits.append)

    form.dispose()

    with qtbot.assertNotEmitted(form.validity_changed):
        editor_at(form.widget, "count").set_value(50)
        editor_at(form.widget, "leaf.count").set_value(50)
        editor_at(form.widget, "area").set_value([])
    assert settings == KitchenSink()
    assert commits == []
    form.dispose()


def _image() -> Image:
    return Image(np.zeros((30, 40), dtype=np.uint16), pixel_size=2.0)


def _messages_after_offset_change(
    settings: PreviewSettings, form: Form, offset_nm: float
) -> list[str]:
    """Move the acquisition offset in the settings, not the form; read the area's messages."""
    settings.offset_x = offset_nm
    return messages(editor_at(form.widget, "area"))


def test_dispose_releases_the_settings_hooks(live_form: Callable[..., Form]) -> None:
    previews = FakePreviews()
    settings = PreviewSettings()
    form = live_form(settings, ctx=FormContext(previews=previews))
    area = editor_at(form.widget, "area")
    button(area, "Load").click()
    previews.handles[-1].succeeded.emit(_image())
    assert _messages_after_offset_change(settings, form, 5.0)
    assert not _messages_after_offset_change(settings, form, 0.0)

    form.dispose()

    assert not _messages_after_offset_change(settings, form, 7.0)


@pytest.mark.usefixtures("qtbot")
def test_destroyed_form_releases_the_settings_hooks(form_builder: FormBuilder) -> None:
    previews = FakePreviews()
    settings = PreviewSettings()
    form = form_builder.build_live(settings, FormContext(previews=previews))
    area = editor_at(form.widget, "area")
    button(area, "Load").click()
    previews.handles[-1].succeeded.emit(_image())

    sip.delete(form.widget)

    # a hook left behind would redraw the stale warning on deleted widgets
    settings.offset_x = 3.0

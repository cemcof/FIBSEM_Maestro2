# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Every field editor kind: its value, optional wrapping, unions, lists, ranges and read-only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from gui_testkit.forms import (
    button,
    children,
    combo_texts,
    editable_controls,
    editor_at,
    field_at,
    optional_checkbox,
    warnings_of,
)
from gui_testkit.sample_settings import (
    CircleShape,
    FastEngine,
    KitchenSink,
    Leaf,
    ManualMode,
    Optionals,
    PointShape,
    PreciseEngine,
    SquareShape,
    filled_kitchen_sink,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QRadioButton,
    QWidget,
)

from fibsem_maestro.settings.form_utils import WidgetType
from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
from fibsem_maestro.ui.schema.field_type import FieldType
from fibsem_maestro.ui.schema.schema import get_field_infos

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytestqt.qtbot import QtBot

    from fibsem_maestro.logging.text.memory import MemoryTextLogger
    from fibsem_maestro.ui.forms.builder import DetachedForm, Form, FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext

SINK_FIELDS = [fi.name for fi in get_field_infos(KitchenSink)]


@pytest.fixture
def detached_sink(
    qtbot: QtBot, form_builder: FormBuilder, sink_context: FormContext
) -> DetachedForm:
    """A detached `KitchenSink` form."""
    form = form_builder.build_detached(KitchenSink, sink_context)
    qtbot.addWidget(form.widget)
    return form


def _union(form: Form, path: str) -> DiscriminatedUnionWidget:
    union = editor_at(form.widget, path)
    assert isinstance(union, DiscriminatedUnionWidget)
    return union


def test_kitchen_sink_covers_every_editor_kind() -> None:
    infos = get_field_infos(KitchenSink)
    hints = {fi.hint.widget for fi in infos if fi.hint is not None}
    kinds = {type(fi.type) for fi in infos if fi.hint is None}

    assert hints == set(WidgetType)
    assert kinds == set(FieldType.__subclasses__())


@pytest.mark.parametrize("name", SINK_FIELDS)
def test_editor_round_trips_its_value(detached_sink: DetachedForm, name: str) -> None:
    value = getattr(filled_kitchen_sink(), name)
    assert value != getattr(KitchenSink(), name), "the value must not be the default"
    editor = field_at(detached_sink.widget, name)

    editor.set_value(value)

    assert editor.get_value() == value


def test_optional_number_is_none_until_ticked(live_form: Callable[..., Form]) -> None:
    settings = Optionals()
    form = live_form(settings)
    wrapper = field_at(form.widget, "limit")
    inner = editor_at(form.widget, "limit")
    assert isinstance(inner, QWidget)
    assert wrapper.get_value() is None
    assert not inner.isVisibleTo(form.widget)

    optional_checkbox(wrapper).click()
    inner.set_value(1.5)

    assert inner.isVisibleTo(form.widget)
    assert settings.limit == 1.5

    optional_checkbox(wrapper).click()

    assert settings.limit is None


def test_optional_choice_offers_none_entry_without_checkbox(
    live_form: Callable[..., Form],
) -> None:
    settings = Optionals()
    form = live_form(settings)
    combo = field_at(form.widget, "color")
    assert isinstance(combo, QComboBox)

    assert combo_texts(combo) == ["(none)", "red", "green", "blue"]
    assert children(combo, QCheckBox) == []

    combo.setCurrentIndex(2)
    assert settings.color is not None and settings.color.value == "green"
    combo.setCurrentIndex(0)
    assert settings.color is None


def test_optional_dropdown_has_checkbox_and_none_entry(
    live_form: Callable[..., Form],
) -> None:
    settings = Optionals()
    form = live_form(settings)
    combo = editor_at(form.widget, "choice")
    assert isinstance(combo, QComboBox)

    optional_checkbox(field_at(form.widget, "choice")).click()
    combo.setCurrentIndex(combo_texts(combo).index("y"))

    assert combo_texts(combo) == ["(none)", "x", "y"]
    assert settings.choice == "y"


@pytest.mark.parametrize(
    ("name", "ticked"),
    [("extra", Leaf()), ("shape", CircleShape()), ("numbers", [])],
)
def test_optional_container_is_none_until_ticked(
    live_form: Callable[..., Form], name: str, ticked: Any
) -> None:
    settings = Optionals()
    form = live_form(settings)
    wrapper = field_at(form.widget, name)
    assert wrapper.get_value() is None

    optional_checkbox(wrapper).click()

    assert getattr(settings, name) == ticked


def test_union_keeps_values_typed_in_other_variants(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink()
    form = live_form(settings)
    union = _union(form, "shape")
    union.select("square")
    editor_at(form.widget, "shape.side").set_value(7.0)

    union.select("circle")
    assert settings.shape == CircleShape()
    union.select("square")

    assert settings.shape == SquareShape(side=7.0)


def test_union_variant_without_fields_shows_no_page(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink()
    form = live_form(settings)
    union = _union(form, "shape")

    union.select("point")

    assert union.selected_variant_widget() is None
    assert settings.shape == PointShape()


def test_union_offers_a_radio_button_per_variant_with_its_description(
    live_form: Callable[..., Form],
) -> None:
    form = live_form(KitchenSink())
    union = _union(form, "shape")
    # shown on the title line of the union's card
    radios = union.variant_buttons()

    assert [(r.text(), r.toolTip()) for r in radios] == [
        ("circle shape", "A round shape."),
        ("square shape", "A square shape."),
        ("point shape", "A single point."),
    ]


def test_nested_union_follows_its_driving_field(
    live_form: Callable[..., Form], log: MemoryTextLogger
) -> None:
    settings = KitchenSink()
    form = live_form(settings)
    mode = _union(form, "mode")
    mode.select("engine")
    inner = mode.selected_variant_widget()
    assert isinstance(inner, DiscriminatedUnionWidget)
    assert settings.mode == FastEngine()

    editor_at(form.widget, "engine").set_value("precise")

    assert settings.mode == PreciseEngine()
    # the inner variant is chosen by the driving field only
    assert not any(
        radio.isVisibleTo(form.widget)
        for radio in children(inner, QRadioButton, direct_only=True)
    )
    assert warnings_of(log) == []


def test_nested_union_falls_back_to_its_first_variant_on_unknown_value(
    live_form: Callable[..., Form], log: MemoryTextLogger
) -> None:
    settings = KitchenSink(engine="precise", mode=PreciseEngine())
    form = live_form(settings)

    editor_at(form.widget, "engine").set_value("turbo")

    assert settings.mode == FastEngine()
    assert warnings_of(log) == [
        "'engine' is 'turbo', which matches no 'engine mode' variant; using 'fast'."
    ]


def test_nested_union_build_corrects_a_variant_disagreeing_with_its_driver(
    live_form: Callable[..., Form],
) -> None:
    settings = KitchenSink(engine="precise", mode=FastEngine())

    form = live_form(settings)

    assert settings.mode == PreciseEngine()
    assert not form.has_errors()


def test_leaf_arm_ignores_the_driving_field(live_form: Callable[..., Form]) -> None:
    settings = KitchenSink()
    form = live_form(settings)

    editor_at(form.widget, "engine").set_value("precise")

    assert settings.mode == ManualMode()


def test_list_adds_and_removes_items(live_form: Callable[..., Form]) -> None:
    settings = KitchenSink(numbers=[1, 2])
    form = live_form(settings)
    numbers = editor_at(form.widget, "numbers")

    button(numbers, "+").click()
    assert settings.numbers == [1, 2, 0]

    removes = [b for b in children(numbers, QPushButton) if b.text() == "×"]
    assert len(removes) == 3
    removes[0].click()

    assert settings.numbers == [2, 0]


def test_list_of_objects_adds_default_items(live_form: Callable[..., Form]) -> None:
    settings = KitchenSink()
    form = live_form(settings)

    button(editor_at(form.widget, "leaves"), "+").click()

    assert settings.leaves == [Leaf()]


def test_range_pair_keeps_low_not_above_high(live_form: Callable[..., Form]) -> None:
    settings = KitchenSink(window=(1.0, 2.0))
    form = live_form(settings)
    low, high = children(editor_at(form.widget, "window"), QDoubleSpinBox)

    low.setValue(5.0)
    assert settings.window == (5.0, 5.0)

    high.setValue(3.0)
    assert settings.window == (5.0, 5.0)

    high.setValue(9.0)
    low.setValue(4.0)
    assert settings.window == (4.0, 9.0)


def test_read_only_reaches_every_control_and_is_undone(
    live_form: Callable[..., Form],
) -> None:
    form = live_form(filled_kitchen_sink())
    editable = editable_controls(form.widget)
    assert editable

    form.set_read_only(True)
    assert editable_controls(form.widget) == []

    form.set_read_only(False)
    assert editable_controls(form.widget) == editable


def test_read_only_reaches_unticked_optional_groups(
    live_form: Callable[..., Form],
) -> None:
    form = live_form(Optionals())

    form.set_read_only(True)

    assert editable_controls(form.widget) == []

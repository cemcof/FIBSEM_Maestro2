# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Groups are cards titled by their field, labels start with the titles, inputs breathe, multi-selects are checklists."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
from gui_testkit.forms import editor_at, field_at, optional_checkbox
from gui_testkit.sample_settings import Color, KitchenSink, Optionals
from pydantic import Field
from PyQt6.QtCore import QEvent, QPoint, QRect
from PyQt6.QtGui import QHelpEvent
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QGridLayout,
    QLabel,
    QScrollArea,
    QStyleOptionViewItem,
    QToolTip,
    QWidget,
)

from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.ui.forms.editors._no_scroll import SPIN_BOX_TEXT_INSET
from fibsem_maestro.ui.forms.editors.checklist import (
    CHECKLIST_MAX_ROWS,
    CHECKLIST_MAX_WIDTH,
)
from fibsem_maestro.ui.forms.editors.collapsible_box import CollapsibleGroupBox
from fibsem_maestro.ui.forms.editors.field_label import FieldLabel
from fibsem_maestro.ui.forms.editors.multi_select import MultiSelectWidget
from fibsem_maestro.ui.forms.editors.object import ObjectWidget
from fibsem_maestro.ui.forms.editors.optional import OptionalWidget
from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
from fibsem_maestro.ui.forms.style import CARD_INDENT, VARIANT_SPACING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pytestqt.qtbot import QtBot

    from fibsem_maestro.ui.forms.builder import Form, FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext


class Choices(BaseSettings):
    """Only editors of a fixed width: they used to leave the editor column unstretchable."""

    color: Color = Field(default=Color.RED)
    a_rather_longer_label_for_another_colour: Color = Field(default=Color.RED)


class Capped(BaseSettings):
    """A short label at the root, a long one in a nested group."""

    hue: Color = Field(default=Color.RED)
    choices: Choices = Field(default_factory=Choices)


@pytest.fixture
def shown_form(
    qtbot: QtBot, form_builder: FormBuilder, sink_context: FormContext
) -> Callable[[BaseSettings], Form]:
    """Build a live form and show it in a resizable scroll area, as the action panel does."""
    # pytest-qt only keeps a weak reference: the scroll areas live as long as the test
    kept: list[QScrollArea] = []

    def show(settings: BaseSettings) -> Form:
        form = form_builder.build_live(settings, sink_context)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form.widget)
        qtbot.addWidget(scroll)
        kept.append(scroll)
        scroll.resize(1400, 900)
        scroll.show()
        qtbot.waitExposed(scroll)
        qtbot.wait(20)
        return form

    return show


@pytest.fixture
def show_widget(qtbot: QtBot) -> Callable[[QWidget], None]:
    """Show a top-level widget and keep it alive until the test's events are processed."""
    # pytest-qt only keeps a weak reference, and paints pending after the
    # test's body would reach a widget already destroyed with its last name
    kept: list[QWidget] = []

    def show(widget: QWidget) -> None:
        qtbot.addWidget(widget)
        kept.append(widget)
        widget.show()
        qtbot.waitExposed(widget)

    return show


def _rect_in(widget: QWidget, ancestor: QWidget) -> QRect:
    """The widget's rectangle in the coordinates of an ancestor."""
    return QRect(widget.mapTo(ancestor, QPoint(0, 0)), widget.size())


def _card(editor: object) -> CollapsibleGroupBox:
    """The card of a group field's editor (the gated card of an optional one)."""
    if isinstance(editor, OptionalWidget):
        assert editor.group is not None
        return editor.group
    assert isinstance(editor, CollapsibleGroupBox)
    return editor


def _column_labels(root: QWidget) -> list[FieldLabel]:
    """The visible labels of the label columns: those not titling a card."""
    return [
        label
        for label in root.findChildren(FieldLabel)
        if isinstance(label.parentWidget(), ObjectWidget) and label.isVisibleTo(root)
    ]


def _text_left(label: QLabel, root: QWidget) -> int:
    """Where the text of a label starts, in the root's coordinates."""
    return _rect_in(label, root).left() + label.contentsRect().left()


def _editor_lefts(form: Form) -> list[int]:
    """Where the editors beside the column labels start, in the form's coordinates."""
    root = form.widget
    lefts = []
    for label in _column_labels(root):
        owner = label.parentWidget()
        assert owner is not None
        grid = owner.layout()
        assert isinstance(grid, QGridLayout)
        row, _, _, _ = grid.getItemPosition(grid.indexOf(label))
        item = grid.itemAtPosition(row, 1)
        editor = item.widget() if item is not None else None
        assert editor is not None
        lefts.append(_rect_in(editor, root).left())
    return lefts


def test_group_label_is_the_title_of_its_card(live_form: Callable[..., Form]) -> None:
    form = live_form(KitchenSink())
    card = _card(field_at(form.widget, "leaf"))
    label = form.widget.field_label("leaf")

    assert label is card.title_label
    assert card.isAncestorOf(label)


def test_union_variants_sit_on_the_title_line_of_their_card(
    shown_form: Callable[[BaseSettings], Form],
) -> None:
    form = shown_form(KitchenSink())
    card = _card(field_at(form.widget, "shape"))
    union = editor_at(form.widget, "shape")
    assert isinstance(union, DiscriminatedUnionWidget)
    title = _rect_in(card.title_label, card)
    buttons = union.variant_buttons()

    # one per variant: circle, square, point
    assert len(buttons) == 3
    for button in buttons:
        assert card.isAncestorOf(button)
        assert not union.isAncestorOf(button)
        # level with the title
        assert abs(_rect_in(button, card).center().y() - title.center().y()) <= 3


def test_union_variants_are_spaced_apart(
    shown_form: Callable[[BaseSettings], Form],
) -> None:
    form = shown_form(KitchenSink())
    union = editor_at(form.widget, "shape")
    assert isinstance(union, DiscriminatedUnionWidget)
    buttons = sorted(union.variant_buttons(), key=lambda button: button.x())

    gaps = [
        right.geometry().left() - left.geometry().right() - 1
        for left, right in pairwise(buttons)
    ]

    assert gaps
    assert all(gap >= VARIANT_SPACING for gap in gaps)


def test_numbers_are_inset_from_the_edge_of_their_spin_box(
    live_form: Callable[..., Form],
) -> None:
    # the number used to start about 2 px from the frame
    form = live_form(KitchenSink())
    spin_boxes = form.widget.findChildren(QAbstractSpinBox)

    assert spin_boxes
    for spin_box in spin_boxes:
        line_edit = spin_box.lineEdit()
        assert line_edit is not None
        assert line_edit.textMargins().left() >= SPIN_BOX_TEXT_INSET


def test_union_card_shrinks_to_its_title_line_for_a_variant_without_fields(
    shown_form: Callable[[BaseSettings], Form], qtbot: QtBot
) -> None:
    form = shown_form(KitchenSink())
    card = _card(field_at(form.widget, "shape"))
    union = editor_at(form.widget, "shape")
    assert isinstance(union, DiscriminatedUnionWidget)
    with_fields = card.height()

    union.select("point")
    qtbot.wait(20)

    assert not union.isVisibleTo(card)
    assert card.height() < with_fields
    assert card.height() <= card.title_label.height() + 20

    union.select("circle")
    assert union.isVisibleTo(card)


def test_optional_group_checkbox_stays_on_the_title_line(
    shown_form: Callable[[BaseSettings], Form], qtbot: QtBot
) -> None:
    form = shown_form(Optionals())
    wrapper = field_at(form.widget, "extra")
    assert isinstance(wrapper, OptionalWidget)
    checkbox = optional_checkbox(wrapper)
    card = _card(wrapper)

    # it once landed outside the shrunk wrapper, where it could not be clicked
    for _ in range(3):
        checkbox.click()
        qtbot.wait(20)
        box = _rect_in(checkbox, wrapper)
        title = _rect_in(card.title_label, wrapper)
        assert wrapper.rect().contains(box)
        assert abs(box.center().y() - title.center().y()) <= 3
        assert box.right() < title.left()


def test_unticked_optional_group_keeps_its_title_and_hides_its_fields(
    shown_form: Callable[[BaseSettings], Form],
) -> None:
    form = shown_form(Optionals())
    wrapper = field_at(form.widget, "extra")
    card = _card(wrapper)
    inner = editor_at(form.widget, "extra")
    assert isinstance(inner, QWidget)

    assert wrapper.get_value() is None
    assert card.title_label.isVisibleTo(form.widget)
    assert not inner.isVisibleTo(form.widget)


@pytest.mark.parametrize("settings", [KitchenSink(), Capped()], ids=["sink", "capped"])
def test_editors_of_every_depth_start_on_one_column(
    shown_form: Callable[[BaseSettings], Form], settings: BaseSettings
) -> None:
    form = shown_form(settings)

    lefts = _editor_lefts(form)

    assert len(lefts) > 1
    assert max(lefts) - min(lefts) <= 1


def test_field_labels_start_in_line_with_the_titles_of_the_cards_beside_them(
    shown_form: Callable[[BaseSettings], Form],
) -> None:
    # right-aligned, short labels looked indented next to the titles
    form = shown_form(Capped())
    root = form.widget
    hue = form.widget.field_label("hue")
    choices = _card(field_at(root, "choices"))
    assert hue is not None
    inner = [label for label in _column_labels(root) if choices.isAncestorOf(label)]

    assert _text_left(hue, root) == _text_left(choices.title_label, root)
    # a nested label is further right than its card's title, by one card
    assert inner
    nested_title = _text_left(choices.title_label, root) + CARD_INDENT
    assert all(_text_left(label, root) == nested_title for label in inner)


def test_multi_select_is_as_tall_as_its_rows_up_to_a_limit(qtbot: QtBot) -> None:
    few = MultiSelectWidget(["a", "b", "c"], default=["b"])
    many = MultiSelectWidget([str(i) for i in range(3 * CHECKLIST_MAX_ROWS)])
    for widget in (few, many):
        qtbot.addWidget(widget)

    row = few.sizeHintForRow(0)
    assert few.height() < many.height()
    assert many.height() < (CHECKLIST_MAX_ROWS + 2) * row
    # the selection is the value, as before
    assert few.get_value() == ["b"]
    item = few.item(2)
    assert item is not None
    item.setSelected(True)
    assert few.get_value() == ["b", "c"]


@pytest.fixture
def thick_list_frames(qapp: QApplication) -> Iterator[None]:
    """An application style sheet giving lists a frame, as qdarkstyle does."""
    previous = qapp.styleSheet()
    qapp.setStyleSheet("QListWidget { border: 4px solid gray; }")
    yield
    qapp.setStyleSheet(previous)


@pytest.mark.usefixtures("thick_list_frames")
def test_multi_select_shows_all_its_rows_once_shown(
    show_widget: Callable[[QWidget], None],
) -> None:
    # fitted before the style sheet gave it its frame, it showed a scroll bar
    # instead of its last row
    widget = MultiSelectWidget(["a", "b"])
    show_widget(widget)

    scrollbar = widget.verticalScrollBar()
    assert scrollbar is not None
    assert not scrollbar.isVisible()
    viewport = widget.viewport()
    assert viewport is not None
    assert viewport.height() >= 2 * widget.sizeHintForRow(0)


def test_multi_select_widens_to_a_long_name_without_a_horizontal_scroll_bar(
    show_widget: Callable[[QWidget], None],
) -> None:
    short = MultiSelectWidget(["stage_position", "beam_shift"])
    long = MultiSelectWidget(
        ["stage_position", "microscope.a_rather_long_custom_parameter"]
    )
    for widget in (short, long):
        show_widget(widget)

    assert long.width() > short.width()
    viewport = long.viewport()
    assert viewport is not None
    item = long.item(1)
    assert item is not None
    assert long.visualItemRect(item).right() <= viewport.width()
    scrollbar = long.horizontalScrollBar()
    # it took the room of a row
    assert scrollbar is None or not scrollbar.isVisible()


def test_multi_select_name_beyond_the_width_limit_shows_whole_in_its_tooltip(
    show_widget: Callable[[QWidget], None],
) -> None:
    name = "microscope." + "very_long_parameter_" * 12
    widget = MultiSelectWidget([name])
    show_widget(widget)
    item = widget.item(0)
    delegate = widget.itemDelegate()
    viewport = widget.viewport()
    assert item is not None and delegate is not None and viewport is not None
    rect = widget.visualItemRect(item)
    option = QStyleOptionViewItem()
    option.initFrom(widget)
    option.rect = rect
    event = QHelpEvent(
        QEvent.Type.ToolTip, rect.center(), viewport.mapToGlobal(rect.center())
    )

    assert widget.width() <= CHECKLIST_MAX_WIDTH
    assert delegate.helpEvent(event, widget, option, widget.indexFromItem(item))
    assert QToolTip.text() == name

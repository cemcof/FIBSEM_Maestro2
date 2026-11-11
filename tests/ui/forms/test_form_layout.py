# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Every field of a form gets the height it needs, in nested groups too."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import Field
from PyQt6.QtWidgets import QAbstractScrollArea, QScrollArea, QVBoxLayout, QWidget

from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.ui.forms.editors.group import GroupWrapper

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.ui.forms.builder import FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext


class Inner(BaseSettings):
    """Labels long enough to wrap onto several lines in a narrow form."""

    a_label_long_enough_to_wrap_onto_several_lines: bool = Field(default=False)
    another_label_long_enough_to_wrap_onto_several_lines: bool = Field(default=False)
    # a spin box shrinks: the height missing elsewhere used to be taken from it
    value: float = Field(default=1.0)


class Middle(BaseSettings):
    """A group in a group."""

    inner: Inner = Field(default_factory=Inner)
    yet_another_label_long_enough_to_wrap_onto_several_lines: bool = Field(
        default=False
    )


class Outer(BaseSettings):
    """The form's root."""

    middle: Middle = Field(default_factory=Middle)
    after: float = Field(default=0.0)


def _needed_height(widget: QWidget) -> int:
    """The height a widget needs at its current width."""
    if widget.hasHeightForWidth():
        return widget.heightForWidth(widget.width())
    return widget.sizeHint().height()


def _squeezed(root: QWidget) -> list[str]:
    """The visible widgets under `root` given less height than they need."""
    found: list[str] = []
    for widget in (root, *root.findChildren(QWidget)):
        if not widget.isVisibleTo(root):
            continue
        # parts of Qt's own composite widgets (e.g. a spin box's line edit)
        if widget.objectName().startswith("qt_") or isinstance(
            widget.parentWidget(), QAbstractScrollArea
        ):
            continue
        if widget.height() + 1 < _needed_height(widget):
            found.append(
                f"{type(widget).__name__}: {widget.height()} < {_needed_height(widget)}"
            )
    return found


@pytest.fixture
def shown_form(
    qtbot: QtBot, form_builder: FormBuilder, form_context: FormContext
) -> tuple[QScrollArea, QWidget]:
    """`Outer`'s form in a resizable scroll area, as the action panel shows forms."""
    form = form_builder.build_live(Outer(), form_context)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(form.widget)
    layout.addStretch()
    scroll.setWidget(container)
    qtbot.addWidget(scroll)
    return scroll, form.widget


@pytest.mark.parametrize("widths", [(320,), (900, 320), (320, 900)])
def test_nested_groups_are_as_tall_as_their_wrapped_labels(
    qtbot: QtBot, shown_form: tuple[QScrollArea, QWidget], widths: tuple[int, ...]
) -> None:
    scroll, form = shown_form
    scroll.show()
    qtbot.waitExposed(scroll)

    # also after the width changed, so that the labels wrap differently
    for width in widths:
        scroll.resize(width, 600)
        qtbot.wait(20)

    assert _squeezed(form) == []


def test_groups_do_not_stretch_into_spare_height(
    qtbot: QtBot, form_builder: FormBuilder, form_context: FormContext
) -> None:
    # no stretch below the form: the scroll area gives it all its height
    form = form_builder.build_live(Outer(), form_context)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(form.widget)
    qtbot.addWidget(scroll)
    scroll.resize(900, 2000)
    scroll.show()
    qtbot.waitExposed(scroll)

    groups = form.widget.findChildren(GroupWrapper)
    assert groups
    for group in groups:
        assert group.height() <= _needed_height(group) + 1

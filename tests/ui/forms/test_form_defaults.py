# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Dynamic defaults follow their sources, are marked "auto" and can be followed again; read-only values are grey."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import pytest
from gui_testkit.forms import children
from pydantic import Field
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QDoubleSpinBox, QLineEdit, QScrollArea, QWidget

from fibsem_maestro.core.direction import Direction
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.dynamic_default import DynamicDefault, effective_value
from fibsem_maestro.settings.form_utils import FieldUnit
from fibsem_maestro.ui.forms.defaults import RESET_TEXT, DefaultReset, shown_default
from fibsem_maestro.ui.forms.editors.dynamic_default import DynamicDefaultWidget
from fibsem_maestro.ui.forms.issues import Severity
from fibsem_maestro.ui.widgets.form_style import FORM_STYLESHEET, READ_ONLY_COLOR

if TYPE_CHECKING:
    from collections.abc import Callable

    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QMenu
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.settings.dynamic_default import SettingsLookup
    from fibsem_maestro.ui.forms.builder import Form, FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext

UNAVAILABLE = "Set a positive base."
SOURCE = "the base"


def _doubled_base(settings: Defaults, lookup: SettingsLookup) -> float | None:
    _ = lookup
    return settings.base * 2 if settings.base > 0 else None


class Defaults(BaseSettings):
    """A field with a dynamic default, next to ordinary ones."""

    base: Annotated[float, FieldUnit(suffix="nm")] = Field(
        default=2.0, description="A number."
    )
    doubled: Annotated[
        float | None,
        FieldUnit(suffix="nm"),
        DynamicDefault(_doubled_base, UNAVAILABLE, SOURCE),
    ] = Field(default=None, description="Twice the base, unless set.")
    name: str = Field(default="x", description="A name.")


def dynamic(form: Form) -> DynamicDefaultWidget[Any]:
    """The editor of the dynamic-default field."""
    editor = form.widget.field_widget("doubled")
    assert isinstance(editor, DynamicDefaultWidget)
    return editor


def reset_of(form: Form) -> DefaultReset:
    """The right-click reset of the dynamic-default field."""
    found = form.widget.findChildren(DefaultReset)
    assert len(found) == 1, f"the form has {len(found)} resets"
    assert found[0].parent() is dynamic(form)
    return found[0]


def spin_of(form: Form, name: str) -> QDoubleSpinBox:
    """The spin box of a number field."""
    editor = form.widget.field_widget(name)
    assert isinstance(editor, QWidget)
    return children(editor, QDoubleSpinBox)[0]


def reset_entry(menu: QMenu) -> QAction:
    """The menu's only entry, the reset."""
    entries = menu.actions()
    assert len(entries) == 1
    assert entries[0].text().startswith(RESET_TEXT)
    return entries[0]


def auto_shown(form: Form) -> bool:
    """Whether the "auto" mark is shown (to the form, which may be hidden)."""
    return dynamic(form).auto_mark.isVisibleTo(form.widget)


@pytest.fixture
def commits() -> list[str]:
    return []


@pytest.fixture
def form_of(
    live_form: Callable[..., Form], commits: list[str]
) -> Callable[[Defaults], Form]:
    return lambda settings: live_form(settings, on_commit=commits.append)


def test_dynamic_default_is_shown_marked_auto_but_not_stored(
    form_of: Callable[[Defaults], Form], commits: list[str]
) -> None:
    settings = Defaults(base=3.0)
    form = form_of(settings)

    assert spin_of(form, "doubled").value() == 6.0
    assert settings.doubled is None
    assert effective_value(settings, "doubled", lambda _: None) == 6.0
    assert auto_shown(form)
    assert SOURCE in dynamic(form).auto_mark.toolTip()
    assert commits == []


def test_dynamic_default_follows_its_source_on_refresh(
    form_of: Callable[[Defaults], Form], commits: list[str]
) -> None:
    settings = Defaults()
    form = form_of(settings)

    spin_of(form, "base").setValue(5.0)
    form.refresh_defaults()

    assert spin_of(form, "doubled").value() == 10.0
    assert settings.doubled is None
    # only the edit of the source was written
    assert commits == ["base"]


def test_dynamic_default_edit_stores_the_value_and_stops_following(
    form_of: Callable[[Defaults], Form],
) -> None:
    settings = Defaults()
    form = form_of(settings)

    spin_of(form, "doubled").setValue(7.0)
    settings.base = 10.0
    form.refresh_defaults()

    assert settings.doubled == 7.0
    assert spin_of(form, "doubled").value() == 7.0
    assert not auto_shown(form)


def test_stored_value_is_not_marked_auto(form_of: Callable[[Defaults], Form]) -> None:
    form = form_of(Defaults(doubled=7.0))

    assert not auto_shown(form)


def test_reset_follows_the_default_again(
    form_of: Callable[[Defaults], Form], commits: list[str]
) -> None:
    settings = Defaults(doubled=7.0)
    form = form_of(settings)
    entry = reset_entry(reset_of(form).menu_for(form.widget))

    assert entry.text() == f"{RESET_TEXT} (4 nm)"
    entry.trigger()

    assert settings.doubled is None
    assert commits == ["doubled"]
    assert spin_of(form, "doubled").value() == 4.0
    assert auto_shown(form)


def test_reset_is_enabled_only_for_a_set_value_in_an_editable_form(
    form_of: Callable[[Defaults], Form],
) -> None:
    form = form_of(Defaults())
    label = form.widget.field_label("doubled")
    assert label is not None
    reset = reset_of(form)

    # following already
    assert not reset_entry(reset.menu_for(label)).isEnabled()

    spin_of(form, "doubled").setValue(7.0)
    assert reset_entry(reset.menu_for(label)).isEnabled()

    form.set_read_only(True)
    assert not reset_entry(reset.menu_for(label)).isEnabled()


def test_dynamic_default_that_cannot_be_computed_warns(
    form_of: Callable[[Defaults], Form],
) -> None:
    form = form_of(Defaults(base=-1.0))
    editor = dynamic(form)

    assert [(i.severity, i.message) for i in editor.issues()] == [
        (Severity.WARNING, UNAVAILABLE)
    ]
    assert not form.has_errors()
    # reported while the editor was built, before its row listened
    label = form.widget.field_label("doubled")
    assert label is not None and "color" in label.styleSheet()

    # a value set makes the default unnecessary
    spin_of(form, "doubled").setValue(1.0)
    assert editor.issues() == []


@pytest.mark.parametrize(
    ("value", "suffix", "text"),
    [
        (18.468712345, "nm", "18.4687 nm"),
        (Direction.UP, None, "up"),
        (None, "nm", "none"),
        (True, None, "checked"),
        ("", None, "empty"),
        (["a", "b"], None, "a, b"),
    ],
)
def test_shown_default_is_short_and_readable(
    value: object, suffix: str | None, text: str
) -> None:
    assert shown_default(value, suffix) == text


_TEXT_TOLERANCE = 8
"""Slack per colour channel (0-255) for rounding in the text rendering."""


def _text_pixels(spin: QDoubleSpinBox) -> list[QColor]:
    """
    The colours of the pixels the spin box's text covers, on its white background.

    Antialiased text blends its colour with the background, so no text pixel
    is darker, in any channel, than the colour the text is drawn in; the
    checks rely on that instead of on exact colours, which depend on the
    fonts and their rasterisation.
    """
    image = children(spin, QLineEdit)[0].grab().toImage()
    white = QColor("white")
    return [
        color
        for x in range(image.width())
        for y in range(image.height())
        if (color := image.pixelColor(x, y)) != white
    ]


def _darker_than(color: QColor, reference: QColor) -> bool:
    """Whether some channel of `color` is clearly darker than in `reference`."""
    return any(
        channel < ref - _TEXT_TOLERANCE
        for channel, ref in zip(color.getRgb()[:3], reference.getRgb()[:3])
    )


def test_values_grey_while_read_only(
    qtbot: QtBot, form_builder: FormBuilder, sink_context: FormContext
) -> None:
    form = form_builder.build_live(Defaults(base=88.0), sink_context)
    scroll = QScrollArea()
    scroll.setStyleSheet(FORM_STYLESHEET)
    scroll.setWidget(form.widget)
    qtbot.addWidget(scroll)
    spin = spin_of(form, "base")
    # a known background, and digits large enough to cover many pixels
    spin.setStyleSheet("background: white;")
    spin.setFont(QFont(spin.font().family(), 30))
    spin.setMinimumWidth(300)
    scroll.resize(600, 600)
    scroll.show()
    qtbot.waitExposed(scroll)
    grey = QColor(READ_ONLY_COLOR)

    editable = _text_pixels(spin)
    if not editable:
        pytest.skip("this system draws no text (no fonts available to Qt)")
    # the default text colour is darker than the grey
    assert any(_darker_than(color, grey) for color in editable)

    # read-only spin boxes used to keep their normal text colour
    form.set_read_only(True)
    qtbot.wait(10)
    read_only = _text_pixels(spin)
    assert read_only
    assert not any(_darker_than(color, grey) for color in read_only)

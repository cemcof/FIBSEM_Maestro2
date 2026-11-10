# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Cross-field wiring: area-selector overlays fed by other fields, and unresolved references."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from gui_testkit.forms import editor_at, field_at, warnings_of
from gui_testkit.sample_settings import (
    BrokenReferences,
    KitchenSink,
    ProgressOnly,
    SlicedArea,
    SquareShape,
)

from fibsem_maestro.core.direction import Direction
from fibsem_maestro.settings.form_utils import AreaOverlay
from fibsem_maestro.ui.forms.editors.area_select.widget import AreaSelectWidget
from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
from fibsem_maestro.ui.forms.issues import Severity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from fibsem_maestro.logging.text.memory import MemoryTextLogger
    from fibsem_maestro.ui.forms.builder import Form
    from fibsem_maestro.ui.forms.context import FormContext

Overlays = dict[AreaOverlay, "Mapping[str, Any]"]


class OverlaySpy:
    """Records what forms push to their area selectors, which then draw as usual."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pushed: list[tuple[object, Overlays]] = []
        draw = AreaSelectWidget.set_overlays
        spy = self

        def record(
            area: AreaSelectWidget,
            overlays: Sequence[tuple[AreaOverlay, Mapping[str, Any]]],
        ) -> None:
            spy.pushed.append((area, dict(overlays)))
            draw(area, overlays)

        monkeypatch.setattr(AreaSelectWidget, "set_overlays", record)

    def last(self, form: Form, path: str = "area") -> Overlays:
        """The overlay values pushed last to the area selector at `path`, by kind."""
        area = editor_at(form.widget, path)
        pushed = [overlays for target, overlays in self.pushed if target is area]
        assert pushed, "nothing was pushed"
        return pushed[-1]


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> OverlaySpy:
    return OverlaySpy(monkeypatch)


@pytest.fixture
def sink(spy: OverlaySpy, live_form: Callable[..., Form]) -> tuple[KitchenSink, Form]:
    _ = spy  # installed before the build
    settings = KitchenSink()
    return settings, live_form(settings)


def _shape(form: Form) -> DiscriminatedUnionWidget:
    union = editor_at(form.widget, "shape")
    assert isinstance(union, DiscriminatedUnionWidget)
    return union


def test_overlay_receives_its_source_value(
    sink: tuple[KitchenSink, Form], spy: OverlaySpy
) -> None:
    settings, form = sink

    editor_at(form.widget, "direction").set_value(Direction.UP)

    assert settings.direction is Direction.UP
    assert spy.last(form)[AreaOverlay.SHOW_DIRECTION] == {"direction": Direction.UP}


def test_overlay_source_in_a_union_is_followed_in_every_variant(
    sink: tuple[KitchenSink, Form], spy: OverlaySpy
) -> None:
    _, form = sink

    editor_at(form.widget, "shape.margin").set_value(3.0)
    assert spy.last(form)[AreaOverlay.SHOW_MARGIN] == {"margin_nm": 3.0}

    # selected after the build: its margin is a different editor
    _shape(form).select("square")
    editor_at(form.widget, "shape.margin").set_value(1.5)
    assert spy.last(form)[AreaOverlay.SHOW_MARGIN] == {"margin_nm": 1.5}

    _shape(form).select("point")
    assert spy.last(form)[AreaOverlay.SHOW_MARGIN] == {}


def test_overlay_values_are_withheld_while_a_requirement_fails(
    sink: tuple[KitchenSink, Form], spy: OverlaySpy
) -> None:
    _, form = sink
    editor_at(form.widget, "overlap").set_value(0.5)
    assert spy.last(form)[AreaOverlay.SHOW_TILES] == {}

    _shape(form).select("square")

    assert spy.last(form)[AreaOverlay.SHOW_TILES] == {
        "tile_size_nm": SquareShape().side,
        "tile_relative_overlap": 0.5,
    }


def test_overlay_values_are_pushed_at_build(
    spy: OverlaySpy, live_form: Callable[..., Form]
) -> None:
    form = live_form(KitchenSink(direction=Direction.UP, shape=SquareShape()))

    assert spy.last(form) == {
        AreaOverlay.SHOW_DIRECTION: {"direction": Direction.UP},
        AreaOverlay.SHOW_MARGIN: {"margin_nm": 0.0},
        AreaOverlay.SHOW_TILES: {
            "tile_size_nm": SquareShape().side,
            "tile_relative_overlap": 0.0,
        },
    }


def test_source_showing_a_value_it_cannot_hold_does_not_break_the_build(
    live_form: Callable[..., Form],
) -> None:
    # the dropdown offers only DOWN and UP
    form = live_form(KitchenSink(direction=Direction.LEFT))

    assert [i.severity for i in field_at(form.widget, "direction").issues()] == [
        Severity.ERROR
    ]


def test_invalid_requirement_source_does_not_break_the_build(
    live_form: Callable[..., Form],
) -> None:
    # a stored square that its own validator rejects: the union cannot form a value
    values: dict[str, Any] = dict(KitchenSink())
    values["shape"] = SquareShape.model_construct(kind="square", side=1.0, margin=5.0)
    settings = KitchenSink.model_construct(**values)

    form = live_form(settings)

    assert form.has_errors()


def test_unresolved_references_are_warned_about_and_skipped(
    live_form: Callable[..., Form], log: MemoryTextLogger
) -> None:
    settings = BrokenReferences()
    form = live_form(settings)
    mode = editor_at(form.widget, "mode")
    assert isinstance(mode, DiscriminatedUnionWidget)

    mode.select("engine")

    assert warnings_of(log) == [
        (
            "Variant 'engine mode' follows 'absent', which was not found in this form; "
            "'fast' is used."
        ),
        (
            "Overlay source 'missing' for field 'area' was not found in this form; "
            "show_margin will not be drawn."
        ),
    ]
    assert settings.mode.kind == "engine"
    assert not form.has_errors()


@pytest.fixture
def progress() -> SimpleNamespace:
    """A stand-in action state: `progress.done` slices milled."""
    return SimpleNamespace(progress=SimpleNamespace(done=2))


@pytest.fixture
def sliced(
    spy: OverlaySpy,
    live_form: Callable[..., Form],
    sink_context: FormContext,
    progress: SimpleNamespace,
) -> Form:
    _ = spy  # installed before the build
    ctx = dataclasses.replace(sink_context, state=lambda: progress)
    return live_form(SlicedArea(), ctx=ctx)


def test_overlay_receives_its_value_from_the_action_state(
    sliced: Form, spy: OverlaySpy
) -> None:
    assert spy.last(sliced)[AreaOverlay.SHOW_AREA_SLICES] == {
        "slice_distance_nm": 10.0,
        "direction": Direction.DOWN,
        "slices_milled": 2,
    }


def test_refreshing_overlays_reads_the_action_state_again(
    sliced: Form, spy: OverlaySpy, progress: SimpleNamespace
) -> None:
    progress.progress.done = 0

    sliced.refresh_overlays()

    assert spy.last(sliced)[AreaOverlay.SHOW_AREA_SLICES]["slices_milled"] == 0


def test_disposed_form_does_not_refresh_its_overlays(
    sliced: Form, spy: OverlaySpy
) -> None:
    pushes = len(spy.pushed)
    sliced.dispose()

    sliced.refresh_overlays()

    assert len(spy.pushed) == pushes


def test_overlay_state_values_are_missing_outside_an_action(
    spy: OverlaySpy, live_form: Callable[..., Form]
) -> None:
    form = live_form(SlicedArea())

    assert "slices_milled" not in spy.last(form)[AreaOverlay.SHOW_AREA_SLICES]


def test_overlay_drawn_only_from_the_action_state_is_pushed(
    spy: OverlaySpy,
    live_form: Callable[..., Form],
    sink_context: FormContext,
    progress: SimpleNamespace,
) -> None:
    ctx = dataclasses.replace(sink_context, state=lambda: progress)

    form = live_form(ProgressOnly(), ctx=ctx)

    assert spy.last(form)[AreaOverlay.SHOW_AREA_SLICES] == {"slices_milled": 2}

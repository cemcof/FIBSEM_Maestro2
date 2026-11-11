# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The area selector: previews through the preview port, staleness, and its area limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from gui_testkit.forms import (
    FakeCatalog,
    FakePermission,
    FakePreviews,
    button,
    editor_at,
    labels,
    messages,
    warnings_of,
)
from gui_testkit.sample_settings import QUARTER, PreviewSettings
from PyQt6 import sip
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QGraphicsView, QWidget

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.image import Image
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.settings.imaging_settings import ImagingSettings
from fibsem_maestro.settings.refocus_settings import RefocusSettings
from fibsem_maestro.ui.forms.context import FormContext
from fibsem_maestro.ui.ports import PreviewRequest

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.logging.text.memory import MemoryTextLogger
    from fibsem_maestro.ui.forms.builder import Form, FormBuilder
    from fibsem_maestro.ui.forms.editor import FieldEditor

NOT_CONNECTED = "Acquisition failed: FIBSEM Maestro is not connected to a microscope."

CORNER = RelativeArea(origin=RelativePoint(x=0.0, y=0.0), width=0.2, height=0.2)
"""A small area in the top-left corner, leaving the rest of the frame free."""


@dataclass
class LinkedImagingStandIn:
    """What a beam source reached through a link reads of an imaging: its name and settings."""

    name: str
    settings: ImagingSettings


@dataclass
class LinkedActionStandIn:
    """What a crop source reached through a link reads of an action: its name and settings."""

    name: str
    settings: PreviewSettings


@pytest.fixture
def previews() -> FakePreviews:
    return FakePreviews()


@pytest.fixture
def permission(qtbot: QtBot) -> FakePermission:
    _ = qtbot  # a QApplication exists before the QObject
    return FakePermission()


@pytest.fixture
def links() -> FakeCatalog:
    return FakeCatalog()


@pytest.fixture
def preview_context(
    previews: FakePreviews, permission: FakePermission, links: FakeCatalog
) -> FormContext:
    return FormContext(
        actions=links.port(),
        previews=previews,
        preview_permission=permission.port(),
    )


@pytest.fixture
def area_of(
    live_form: Callable[..., Form], preview_context: FormContext
) -> Callable[..., tuple[Form, FieldEditor[Any]]]:
    """Build a live form of the settings and return it with its `area` selector."""

    def build(
        settings: PreviewSettings, ctx: FormContext | None = None
    ) -> tuple[Form, FieldEditor[Any]]:
        form = live_form(settings, ctx=ctx if ctx is not None else preview_context)
        return form, editor_at(form.widget, "area")

    return build


def _widget(editor: FieldEditor[Any]) -> QWidget:
    assert isinstance(editor, QWidget)
    return editor


def _load(area: FieldEditor[Any]) -> None:
    button(area, "Load").click()


def _image(width: int = 40, height: int = 30) -> Image:
    return Image(np.zeros((height, width), dtype=np.uint16), pixel_size=2.0)


def test_load_without_preview_port_reports_not_connected(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
) -> None:
    _, area = area_of(PreviewSettings(), FormContext())

    _load(area)

    assert NOT_CONNECTED in labels(area)
    assert button(area, "Load").isEnabled()


@pytest.mark.parametrize("expanded_before", [False, True])
def test_load_expands_the_selector(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
    expanded_before: bool,
) -> None:
    _, area = area_of(PreviewSettings())
    if expanded_before:
        button(area, "▼ Expand").click()

    _load(area)

    # never collapses: the image is loaded to draw on it
    assert button(area, "▲ Collapse").isVisibleTo(_widget(area))


def test_load_requests_the_beam_offset_and_crop_of_the_live_settings(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]], previews: FakePreviews
) -> None:
    settings = PreviewSettings(scan_area=[QUARTER])
    _, area = area_of(settings)
    # read when Load is pressed, not when the form was built
    settings.beam = BeamType.ELECTRON
    settings.offset_x = 150.0

    _load(area)

    assert previews.requests == [
        PreviewRequest(
            beam=BeamType.ELECTRON,
            offset_x_nm=150.0,
            previous_offset_nm=None,
            crop=QUARTER,
        )
    ]
    assert "Loading image..." in labels(area)
    assert not button(area, "Load").isEnabled()


def test_load_without_crop_area_requests_the_full_frame(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]], previews: FakePreviews
) -> None:
    _, area = area_of(PreviewSettings())

    _load(area)

    assert previews.requests[-1].crop is None


def test_load_crops_to_the_linked_action_without_offset(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
    previews: FakePreviews,
    links: FakeCatalog,
) -> None:
    linked = LinkedActionStandIn("source", PreviewSettings(scan_area=[CORNER]))
    links.replace([cast("Action", linked)])
    settings = PreviewSettings(linked="source", offset_x=150.0, scan_area=[QUARTER])
    _, area = area_of(settings)

    _load(area)

    assert previews.requests[-1].crop == CORNER
    assert previews.requests[-1].offset_x_nm == 0.0


@pytest.mark.parametrize(
    ("own", "linked_beam", "expected"),
    [
        # its own beam decides first
        (BeamType.ION, BeamType.ELECTRON, BeamType.ION),
        # without one, the linked action's
        (None, BeamType.ELECTRON, BeamType.ELECTRON),
        # with neither, the active beam
        (None, None, None),
    ],
)
def test_load_requests_its_own_beam_else_the_linked_actions(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
    previews: FakePreviews,
    links: FakeCatalog,
    own: BeamType | None,
    linked_beam: BeamType | None,
    expected: BeamType | None,
) -> None:
    linked = LinkedActionStandIn("source", PreviewSettings(beam=linked_beam))
    links.replace([cast("Action", linked)])
    _, area = area_of(PreviewSettings(beam=own, linked="source"))

    _load(area)

    assert previews.requests[-1].beam == expected


def test_load_requests_the_beam_of_a_refocus_s_linked_imaging(
    live_form: Callable[..., Form],
    preview_context: FormContext,
    previews: FakePreviews,
    links: FakeCatalog,
) -> None:
    imaging = LinkedImagingStandIn("imaging", ImagingSettings(beam_type=BeamType.ION))
    links.replace([cast("Action", imaging)])
    settings = RefocusSettings(linked_imaging="imaging")
    assert settings.beam_type is None
    form = live_form(settings, ctx=preview_context)

    for path in ("mode.scanning_area", "mode.criterion.area"):
        _load(editor_at(form.widget, path))

    assert [request.beam for request in previews.requests] == [
        BeamType.ION,
        BeamType.ION,
    ]


def test_delivered_image_is_shown(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]], previews: FakePreviews
) -> None:
    settings = PreviewSettings()
    _, area = area_of(settings)
    _load(area)

    previews.handles[-1].succeeded.emit(_image())

    assert "Image: 40×30 px" in labels(area)
    assert button(area, "Load").isEnabled()
    assert area.get_value() == [QUARTER]
    assert settings.area == [QUARTER]


def test_failed_preview_is_reported(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]], previews: FakePreviews
) -> None:
    _, area = area_of(PreviewSettings())
    _load(area)

    previews.handles[-1].failed.emit(RuntimeError("beam is off"))

    assert "Acquisition failed: beam is off" in labels(area)
    assert button(area, "Load").isEnabled()


def test_preview_delivered_after_the_widget_is_destroyed_is_discarded(
    form_builder: FormBuilder, preview_context: FormContext, previews: FakePreviews
) -> None:
    form = form_builder.build_live(PreviewSettings(), preview_context)
    _load(editor_at(form.widget, "area"))

    sip.delete(form.widget)

    # Qt dropped the connections to the destroyed widget: nothing is called
    previews.handles[-1].succeeded.emit(_image())
    previews.handles[-1].failed.emit(RuntimeError("late"))


def test_load_is_enabled_only_when_permitted_editable_and_idle(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
    permission: FakePermission,
    previews: FakePreviews,
) -> None:
    form, area = area_of(PreviewSettings())
    load = button(area, "Load")
    assert load.isEnabled()

    permission.set_allowed(False)
    assert not load.isEnabled()
    permission.set_allowed(True)
    assert load.isEnabled()

    form.set_read_only(True)
    assert not load.isEnabled()
    form.set_read_only(False)
    assert load.isEnabled()

    load.click()
    permission.set_allowed(True)
    assert not load.isEnabled()
    previews.handles[-1].succeeded.emit(_image())
    assert load.isEnabled()


def test_offset_changed_after_load_marks_the_image_stale(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]], previews: FakePreviews
) -> None:
    settings = PreviewSettings()
    form, area = area_of(settings)
    _load(area)
    previews.handles[-1].succeeded.emit(_image())
    assert area.issues() == []

    editor_at(form.widget, "offset_x").set_value(5.0)

    assert messages(area) == [
        "Image loaded at an offset of 0 nm, the offset is now 5 nm. Load it again."
    ]
    assert not form.has_errors()

    _load(area)
    assert previews.requests[-1].previous_offset_nm == 0.0
    assert previews.requests[-1].offset_x_nm == 5.0
    previews.handles[-1].succeeded.emit(_image())
    assert area.issues() == []


def test_detached_area_selector_warns_that_its_context_cannot_be_resolved(
    qtbot: QtBot,
    form_builder: FormBuilder,
    log: MemoryTextLogger,
) -> None:
    form = form_builder.build_detached(PreviewSettings, FormContext(log=log))
    qtbot.addWidget(form.widget)

    assert warnings_of(log) == [
        (
            "Field 'area' declares beam sources but is not bound to live "
            "settings; the active beam is used."
        ),
        (
            "Field 'area' declares crop sources but is not bound to live settings; "
            "the whole frame is shown."
        ),
        (
            "Field 'area' declares an acquisition offset but is not bound to live "
            "settings; images are grabbed at the current beam position."
        ),
    ]


def _expanded_viewer(
    qtbot: QtBot, area: FieldEditor[Any], previews: FakePreviews
) -> QGraphicsView:
    """Load an image into the area selector, which expands it, and return its viewer."""
    widget = _widget(area)
    window = widget.window()
    assert window is not None
    window.show()
    _load(area)
    previews.handles[-1].succeeded.emit(_image(400, 300))
    viewer = widget.findChild(QGraphicsView)
    assert isinstance(viewer, QGraphicsView)
    qtbot.waitUntil(viewer.isVisible)
    return viewer


def _drag(
    qtbot: QtBot,
    viewer: QGraphicsView,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Drag with the left button between two points given relative to the frame."""
    port = viewer.viewport()
    assert port is not None
    rect = viewer.sceneRect()

    def at(point: tuple[float, float]) -> QPoint:
        x, y = point
        return viewer.mapFromScene(rect.width() * x, rect.height() * y)

    qtbot.mousePress(port, Qt.MouseButton.LeftButton, pos=at(start))
    qtbot.mouseMove(port, at(end))
    qtbot.mouseRelease(port, Qt.MouseButton.LeftButton, pos=at(end))


def test_too_few_areas_is_an_error(
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
) -> None:
    settings = PreviewSettings()
    form, area = area_of(settings)

    area.set_value([])

    assert messages(area) == ["At least 1 area(s) required."]
    assert form.has_errors()
    assert settings.area == [QUARTER]


def test_drawing_adds_areas_up_to_the_maximum(
    qtbot: QtBot,
    area_of: Callable[..., tuple[Form, FieldEditor[Any]]],
    previews: FakePreviews,
) -> None:
    settings = PreviewSettings(area=[CORNER])
    _, area = area_of(settings)
    viewer = _expanded_viewer(qtbot, area, previews)

    _drag(qtbot, viewer, (0.5, 0.5), (0.7, 0.7))
    assert len(settings.area) == 2

    _drag(qtbot, viewer, (0.3, 0.75), (0.45, 0.95))

    assert len(settings.area) == 2
    assert "Maximum of 2 area(s) reached." in labels(area)

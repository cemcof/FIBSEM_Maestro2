# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The slice overlay of an area selector: where the next slice is highlighted."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QGraphicsRectItem

from fibsem_maestro.core.direction import Direction
from fibsem_maestro.settings.form_utils import AreaOverlay
from fibsem_maestro.ui.forms.editors.area_select._rectangle import ResizableRect
from fibsem_maestro.ui.forms.editors.area_select.overlays import (
    AreaSliceDecoration,
    build_decorations,
)

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

AREA = QRectF(10.0, 20.0, 40.0, 30.0)
"""An area 40 px wide and 30 px high, holding three 10 px slices along y."""


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        # from the trailing edge: the top when moving down, the bottom when moving up
        (Direction.DOWN, QRectF(10.0, 40.0, 40.0, 10.0)),
        (Direction.UP, QRectF(10.0, 20.0, 40.0, 10.0)),
        (Direction.RIGHT, QRectF(30.0, 20.0, 10.0, 30.0)),
        (Direction.LEFT, QRectF(20.0, 20.0, 10.0, 30.0)),
    ],
)
def test_next_slice_lies_slices_milled_strips_from_the_trailing_edge(
    direction: Direction, expected: QRectF
) -> None:
    decoration = AreaSliceDecoration(10.0, direction, slices_milled=2)

    assert decoration.next_slice(AREA) == expected


@pytest.mark.parametrize("slices_milled", [None, 3, 10])
def test_next_slice_is_not_highlighted_if_unknown_or_beyond_the_area(
    slices_milled: int | None,
) -> None:
    decoration = AreaSliceDecoration(10.0, Direction.DOWN, slices_milled)

    assert decoration.next_slice(AREA) is None


def test_slice_overlay_is_built_with_the_progress_of_its_values() -> None:
    (decoration,) = build_decorations(
        [
            (
                AreaOverlay.SHOW_AREA_SLICES,
                {
                    "slice_distance_nm": 20.0,
                    "direction": Direction.DOWN,
                    "slices_milled": 1,
                },
            )
        ],
        pixel_size_nm=2.0,
    )

    assert isinstance(decoration, AreaSliceDecoration)
    assert decoration.next_slice(AREA) == QRectF(10.0, 30.0, 40.0, 10.0)


def _highlights(rect: ResizableRect) -> list[QGraphicsRectItem]:
    """The visible filled strips drawn on an area."""
    return [
        item
        for item in rect.childItems()
        if isinstance(item, QGraphicsRectItem) and item.isVisible()
    ]


def test_attached_slice_overlay_fills_the_next_slice_and_follows_the_area(
    qtbot: QtBot,
) -> None:
    _ = qtbot  # a QApplication exists
    rect = ResizableRect(AREA)
    decoration = AreaSliceDecoration(10.0, Direction.DOWN, slices_milled=2)

    decoration.attach(rect)
    (strip,) = _highlights(rect)
    assert strip.rect() == QRectF(10.0, 40.0, 40.0, 10.0)

    # one slice too short now: nothing is filled
    rect.setRect(QRectF(10.0, 20.0, 40.0, 25.0))
    decoration.update(rect)
    assert _highlights(rect) == []

    decoration.detach()

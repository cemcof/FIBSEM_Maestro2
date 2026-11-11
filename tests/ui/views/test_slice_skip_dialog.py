# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PyQt6.QtWidgets import QMessageBox, QWidget

from fibsem_maestro.ui.app.slice_skip import SliceSkipPlan
from fibsem_maestro.ui.views.dialogs.slice_skip import (
    confirm_slice_skip,
    slice_skip_question,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gui_testkit.dialogs import MessageBoxes
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action

WARNING = "'b': mind the gap"


@pytest.fixture
def actions(add_action: Callable[..., Action]) -> tuple[Action, Action, Action]:
    a, b, c = (add_action("adjust_props", name) for name in ("a", "b", "c"))
    return a, b, c


def plan_of(
    actions: tuple[Action, Action, Action], *warnings: str, running: bool = False
) -> SliceSkipPlan:
    a, b, c = actions
    return SliceSkipPlan(
        slice=7, skipped=(b, c), finishing=a if running else None, warnings=warnings
    )


def test_slice_skip_question_tells_what_is_skipped_and_what_follows(
    actions: tuple[Action, Action, Action],
) -> None:
    text = slice_skip_question(plan_of(actions, WARNING, running=True))

    for part in ("7", "8", "'a'", "'b'", "'c'", WARNING):
        assert part in text


def test_slice_skip_question_without_warnings_or_running_action(
    actions: tuple[Action, Action, Action],
) -> None:
    text = slice_skip_question(plan_of(actions))

    assert "'a'" not in text
    assert "warning" not in text.lower()


@pytest.mark.parametrize(
    ("warnings", "kind"), [((), "question"), ((WARNING,), "warning")]
)
@pytest.mark.parametrize(
    ("answer", "confirmed"),
    [
        (QMessageBox.StandardButton.Yes, True),
        (QMessageBox.StandardButton.Cancel, False),
    ],
)
def test_confirm_slice_skip_warns_only_when_it_must(
    qtbot: QtBot,
    boxes: MessageBoxes,
    actions: tuple[Action, Action, Action],
    warnings: tuple[str, ...],
    kind: str,
    answer: QMessageBox.StandardButton,
    confirmed: bool,
) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    boxes.answer = answer

    assert confirm_slice_skip(parent, plan_of(actions, *warnings)) is confirmed
    assert [call[0] for call in boxes.calls] == [kind]

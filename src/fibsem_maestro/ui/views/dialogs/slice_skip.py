# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The confirmation of skipping the rest of the current slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

    from fibsem_maestro.ui.app.slice_skip import SliceSkipPlan

SLICE_SKIP_TITLE = "Skip the rest of the slice"
"""Title of the confirmation."""


def slice_skip_question(plan: SliceSkipPlan) -> str:
    """
    The text asking to confirm a skip of the rest of the slice.

    Args:
        plan: What the skip would do.

    Returns:
        Which actions will not run, what happens next, and the warnings, if any.
    """
    current, following = plan.slice, plan.slice + 1
    names = ", ".join(f"'{action.name}'" for action in plan.skipped)
    lines = [f"Skip the rest of slice {current}?", ""]
    if plan.finishing is not None:
        lines.append(f"'{plan.finishing.name}' finishes first.")
    lines += [
        f"These actions will not run for slice {current}: {names}.",
        (
            f"They keep their properties and state for slice {following}, and the "
            f"workflow pauses at the start of slice {following}."
        ),
    ]
    if plan.warnings:
        lines += ["", "Warning:", *(f"• {warning}" for warning in plan.warnings)]
    return "\n".join(lines)


def confirm_slice_skip(parent: QWidget, plan: SliceSkipPlan) -> bool:
    """
    Ask the user to confirm a skip of the rest of the slice.

    A warning box if skipping some action deserves a warning, a question otherwise.

    Args:
        parent: The dialog's parent.
        plan: What the skip would do.

    Returns:
        True if the user confirmed.
    """
    ask = QMessageBox.warning if plan.warnings else QMessageBox.question
    answer = ask(
        parent,
        SLICE_SKIP_TITLE,
        slice_skip_question(plan),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes

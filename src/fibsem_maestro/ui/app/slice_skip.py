# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""What skipping the rest of the current slice would do, and when it is possible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fibsem_maestro.ui.app.run_state import AppState

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.workflow.workflow import Workflow


@dataclass(frozen=True)
class SliceSkipPlan:
    """
    What skipping the rest of the current slice would do, for the user to confirm.

    The skipped actions carry their properties and state over to the next
    slice unchanged (`Action.skip`); the run then continues with the next slice.

    Attributes:
        slice: The workflow's current slice, the one whose rest is skipped.
        skipped: The actions that would not run for it, in workflow order.
        finishing: The action executing now, which finishes first; None if
            no action executes.
        warnings: Why skipping some of the actions deserves a warning
            (`Workflow.skip_warnings`).
    """

    slice: int
    skipped: tuple[Action, ...]
    finishing: Action | None
    warnings: tuple[str, ...]


def plan_slice_skip(
    workflow: Workflow, state: AppState, awaiting_choice: bool
) -> SliceSkipPlan | None:
    """
    Plan a skip of the rest of the workflow's current slice, if one makes sense now.

    Possible while running (the executing action finishes first), while paused
    or reloaded after some action of the slice has run, and while a failed
    action awaits a choice (it is skipped too). Never with nothing left to
    skip, and never at the start of a slice, where it would only leave the
    slice empty.

    Args:
        workflow: The workflow.
        state: The run state.
        awaiting_choice: Whether a failed action waits for Retry or Skip;
            INTERRUPTED alone may also mean the run ended with an error.

    Returns:
        The plan, or None if the rest of the slice cannot be skipped now.
    """
    pending = workflow.pending_actions()
    finishing: Action | None = None
    match state:
        case AppState.RUNNING:
            if not pending:
                return None
            finishing, *rest = pending
            skipped = tuple(rest)
        case AppState.PAUSED | AppState.RELOADED if workflow.slice_started():
            skipped = tuple(pending)
        case AppState.INTERRUPTED if awaiting_choice:
            skipped = tuple(pending)
        case _:
            return None

    if not skipped:
        return None
    return SliceSkipPlan(
        slice=workflow.ctx.slice,
        skipped=skipped,
        finishing=finishing,
        warnings=tuple(workflow.skip_warnings(skipped)),
    )

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Per-action indicators of the action list: check mark, arrow and slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from fibsem_maestro.ui.app.run_state import AppState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession


@dataclass(frozen=True)
class ActionIndicator:
    """
    What the action list shows next to one action.

    Attributes:
        prepared: Green check mark: props stored for the action's current slice
            and no preparation issues.
        next_to_run: Blue arrow: the next action the workflow will execute.
        slice: The action's own slice counter (tooltip "slice number: N").
    """

    prepared: bool
    next_to_run: bool
    slice: int


def indicators_for(
    actions: Sequence[Action], current_or_next: Action | None, state: AppState
) -> dict[Action, ActionIndicator]:
    """
    Compute the indicators of every action.

    Args:
        actions: The workflow's actions.
        next_action: The currently running action, the next action to run, or None.
        state: The run state; the arrow is shown only outside EDITING and RELOADED.

    Returns:
        The indicator of each action.
    """
    show_arrow = state not in (AppState.EDITING, AppState.RELOADED)
    return {
        action: ActionIndicator(
            prepared=action.has_stored_properties() and not action.preparation_issues(),
            next_to_run=show_arrow and action is current_or_next,
            slice=action.ctx.slice,
        )
        for action in actions
    }


class ActionStatus(QObject):
    """
    The current indicators of every action, recomputed whenever they may have changed.

    Args:
        session: The workflow session.
        run: The run controller.
        tasks: The microscope task runner (a failed Prepare can leave props on
            disk without any session signal).
        parent: Optional Qt parent.
    """

    changed = pyqtSignal()
    """Some indicator changed."""

    def __init__(
        self,
        session: WorkflowSession,
        run: RunController,
        tasks: MicroscopeTasks,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._run = run
        self._indicators = self._compute()

        for signal in (
            session.structure_changed,
            session.action_settings_changed,
            session.action_props_changed,
            session.action_state_changed,
            session.rules_changed,
            session.workflow_reset,
            session.workflow_replaced,
            run.state_changed,
            run.slice_finished,
            run.action_finished,
            tasks.busy_changed,
        ):
            signal.connect(self._recompute)

    def indicator(self, action: Action) -> ActionIndicator:
        """
        The indicator of an action.

        Args:
            action: An action of the workflow.

        Returns:
            Its current indicator (computed on the spot for an action not seen yet).
        """
        if (indicator := self._indicators.get(action)) is not None:
            return indicator
        return indicators_for([action], self._next_action(), self._run.state)[action]

    def _next_action(self) -> Action | None:
        actions = self._session.actions
        # None between the last action of a slice and its end: the first runs next
        return self._session.workflow.next_pending_action() or (
            actions[0] if actions else None
        )

    def _compute(self) -> dict[Action, ActionIndicator]:
        return indicators_for(
            self._session.actions, self._next_action(), self._run.state
        )

    def _recompute(self, *_: object) -> None:
        indicators = self._compute()
        if indicators != self._indicators:
            self._indicators = indicators
            self.changed.emit()

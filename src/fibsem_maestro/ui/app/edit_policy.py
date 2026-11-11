# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Every enable/disable rule of the GUI, in one place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from fibsem_maestro.ui.app.run_state import AppState

if TYPE_CHECKING:
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.app.run_controller import RunController


@dataclass(frozen=True)
class Permissions:
    """
    What the user may do right now.

    Attributes:
        structure_editable: Add, remove, duplicate, reorder and rename actions (not rules).
        settings_editable: Edit action settings forms.
        rules_editable: Edit propagation rules.
        action_tools_enabled: Use Prepare / Edit / Apply / Test (Edit and Apply
            also need stored props).
        danger_tools_enabled: Use the tools usable in danger mode
            (`ToolSpec.usable_in_danger_mode`, e.g. Edit): whenever
            `action_tools_enabled`, and in any state in danger mode.
        preview_enabled: Load an area-selector image (also needs an editable form).
        import_enabled: Import a workflow.
        new_enabled: Create a new workflow.
        reset_enabled: Reset the workflow.
        microscope_settings_editable: Edit the microscope settings dialog
            (evaluated when it opens).
        error_choice_enabled: Use the Retry / Skip / Skip and pause buttons.
        slice_skip_enabled: Skip the rest of the slice, as far as the state
            allows; the workflow must also have something to skip
            (`RunController.planned_slice_skip`).
        task_busy: A microscope task is running (an input, exposed for views).
    """

    structure_editable: bool
    settings_editable: bool
    rules_editable: bool
    action_tools_enabled: bool
    danger_tools_enabled: bool
    preview_enabled: bool
    import_enabled: bool
    new_enabled: bool
    reset_enabled: bool
    microscope_settings_editable: bool
    error_choice_enabled: bool
    slice_skip_enabled: bool
    task_busy: bool


def permissions_for(state: AppState, danger_mode: bool, task_busy: bool) -> Permissions:
    """
    Compute the permissions for a situation.

    Args:
        state: The run state.
        danger_mode: Whether settings, and the tools usable in danger mode,
            may be used in any state.
        task_busy: Whether a microscope task is running.

    Returns:
        The permissions.
    """
    not_running = not state.is_running
    idle = not_running and not task_busy
    settings_editable = not_running or danger_mode
    return Permissions(
        structure_editable=idle and state is not AppState.INTERRUPTED,
        settings_editable=settings_editable,
        rules_editable=not_running,
        action_tools_enabled=idle,
        danger_tools_enabled=idle or danger_mode,
        preview_enabled=idle,
        import_enabled=state is AppState.EDITING and not task_busy,
        new_enabled=idle,
        reset_enabled=idle,
        microscope_settings_editable=settings_editable,
        error_choice_enabled=state is AppState.INTERRUPTED and not task_busy,
        # while running, the skip waits for the current action like a pause
        slice_skip_enabled=state is AppState.RUNNING
        or (
            state in (AppState.PAUSED, AppState.RELOADED, AppState.INTERRUPTED)
            and not task_busy
        ),
        task_busy=task_busy,
    )


class EditPolicy(QObject):
    """
    The current permissions and danger mode, observable.

    Also the `PreviewPermission` port of the forms (`changed`, `preview_allowed()`).

    Args:
        run: The run controller (state).
        tasks: The microscope task runner (busy).
        parent: Optional Qt parent.
    """

    changed = pyqtSignal(object)
    """The new `Permissions`, emitted only when they differ."""

    danger_mode_changed = pyqtSignal(bool)
    """The new danger mode."""

    def __init__(
        self,
        run: RunController,
        tasks: MicroscopeTasks,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._run = run
        self._tasks = tasks
        self._danger_mode = False
        self._permissions = permissions_for(run.state, False, tasks.busy)

        run.state_changed.connect(self._recompute)
        tasks.busy_changed.connect(self._recompute)

    @property
    def permissions(self) -> Permissions:
        """The current permissions."""
        return self._permissions

    @property
    def danger_mode(self) -> bool:
        """
        Whether settings can be edited in any state, including while the workflow runs.

        Only settings and the tools usable in danger mode (e.g. editing stored
        properties) are unlocked; microscope access from the GUI and the
        structure of the workflow still follow the state.
        """
        return self._danger_mode

    def set_danger_mode(self, enabled: bool) -> None:
        """
        Turn danger mode on or off.

        Args:
            enabled: True to allow editing settings, and using the tools
                usable in danger mode, in any state.
        """
        if enabled == self._danger_mode:
            return
        self._danger_mode = enabled
        self.danger_mode_changed.emit(enabled)
        self._recompute()

    def preview_allowed(self) -> bool:
        """Whether an area selector may acquire an image now (`permissions.preview_enabled`)."""
        return self._permissions.preview_enabled

    def _recompute(self, *_: object) -> None:
        permissions = permissions_for(
            self._run.state, self._danger_mode, self._tasks.busy
        )
        if permissions != self._permissions:
            self._permissions = permissions
            self.changed.emit(permissions)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Every user intent: the only way views change anything."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fibsem_maestro.logging.text.file import close_all_log_files
from fibsem_maestro.ui.app.errors import (
    DirectoryNotClearedError,
    NotAllowedError,
    PropertiesChangedError,
)
from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTask
from fibsem_maestro.ui.app.run_state import AppState

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.properties.global_properties import GlobalProperties
    from fibsem_maestro.settings.property_names import PropertyNames
    from fibsem_maestro.ui.app.edit_policy import EditPolicy, Permissions
    from fibsem_maestro.ui.app.files import WorkflowFiles
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.app.previews import PreviewService
    from fibsem_maestro.ui.app.readiness import Readiness
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.app.user_config import UserConfig
    from fibsem_maestro.ui.ports import PreviewRequest, TaskHandle
    from fibsem_maestro.workflow.error_choice import ErrorChoice
    from fibsem_maestro.workflow.propagations import PropagationRule


class AppCommands:
    """
    The complete list of things a user can do.

    Each command checks `EditPolicy` (views disable the controls anyway).
    Commands that start microscope work return a `TaskHandle` and refuse by
    having it emit `failed(NotAllowedError)`; all other commands raise
    `NotAllowedError`. Also the forms' `PreviewPort` (`grab`).

    Args:
        session: The workflow session.
        run: The run controller.
        tasks: The microscope task runner.
        previews: The preview service.
        policy: The edit policy.
        readiness: The run readiness (only its report is read, by `run_pressed`).
        user_config: The remembered profiles.
        files: Filesystem operations.
    """

    def __init__(
        self,
        session: WorkflowSession,
        run: RunController,
        tasks: MicroscopeTasks,
        previews: PreviewService,
        policy: EditPolicy,
        readiness: Readiness,
        user_config: UserConfig,
        files: WorkflowFiles,
    ) -> None:
        self._session = session
        self._run = run
        self._tasks = tasks
        self._previews = previews
        self._policy = policy
        self._readiness = readiness
        self._user_config = user_config
        self._files = files

    def new_workflow(self, directory: Path) -> None:
        """
        Clear `directory` and start an empty workflow in it.

        Args:
            directory: The new workflow directory (the user confirmed clearing it).

        Raises:
            NotAllowedError: If `new_enabled` is false.
            DirectoryNotClearedError: If some children could not be deleted;
                nothing else is changed then.
        """
        self._check(
            self._permissions().new_enabled, "Cannot create a new workflow now."
        )
        if failures := self._files.clear(directory):
            raise DirectoryNotClearedError(failures)
        self._run.abort()
        try:
            self._session.new_workflow(directory)
        finally:
            self._run.set_editing()

    def import_workflow(self, source: Path) -> None:
        """
        Replace the actions and rules by those of another workflow directory.

        Args:
            source: Directory of the workflow to import.

        Raises:
            NotAllowedError: If `import_enabled` is false.
            WorkflowImportError: If the import failed.
        """
        self._check(self._permissions().import_enabled, "Cannot import a workflow now.")
        self._session.import_workflow(source)

    def reset_workflow(self) -> None:
        """
        Abandon any run and reset the workflow to slice 0.

        The state is EDITING afterwards, even if the reset failed.

        Raises:
            NotAllowedError: If `reset_enabled` is false.
            WorkflowResetError: If the reset failed.
        """
        self._check(self._permissions().reset_enabled, "Cannot reset the workflow now.")
        self._run.abort()
        try:
            self._session.reset()
        finally:
            self._run.set_editing()

    def run_pressed(self) -> None:
        """
        The run button: start, resume or pause, depending on the state.

        EDITING, RELOADED and FINISHED start a run; PAUSED resumes it; RUNNING
        and SKIPPING request a pause; any other state does nothing.

        Raises:
            NotAllowedError: If starting or resuming while the workflow is not
                ready, or while a microscope task runs.
        """
        match self._run.state:
            case AppState.EDITING | AppState.RELOADED | AppState.FINISHED:
                self._require_ready()
                self._run.start()
            case AppState.PAUSED:
                self._require_ready()
                self._run.resume()
            case AppState.RUNNING | AppState.SKIPPING:
                self._run.pause()
            case _:
                pass

    def set_danger_mode(self, enabled: bool) -> None:
        """
        Turn danger mode on or off.

        Args:
            enabled: True to allow editing settings in any state.
        """
        self._policy.set_danger_mode(enabled)

    def choose_error_option(self, choice: ErrorChoice) -> None:
        """
        Answer a failed action with Retry or Skip.

        Args:
            choice: `ErrorChoice.RETRY` or `ErrorChoice.SKIP`.

        Raises:
            NotAllowedError: If `error_choice_enabled` is false or no choice is awaited.
        """
        self._check(
            self._permissions().error_choice_enabled, "Cannot answer the error now."
        )
        self._run.choose(choice)

    def skip_and_pause(self) -> None:
        """
        Skip the failed action and pause right after it.

        Raises:
            NotAllowedError: If `error_choice_enabled` is false or no choice is awaited.
        """
        self._check(
            self._permissions().error_choice_enabled, "Cannot answer the error now."
        )
        self._run.skip_and_pause()

    def skip_slice(self) -> None:
        """
        Skip the rest of the current slice and continue with the next one.

        See `RunController.skip_slice`.

        Raises:
            NotAllowedError: If `slice_skip_enabled` is false, nothing can be
                skipped now, or the run would resume or start while the
                workflow is not ready.
        """
        self._check(
            self._permissions().slice_skip_enabled,
            "Cannot skip the rest of the slice now.",
        )
        plan = self._run.planned_slice_skip()
        if plan is None:
            raise NotAllowedError("Nothing of the slice is left to skip.")
        if self._run.state in (AppState.PAUSED, AppState.RELOADED):
            # the run resumes or starts, as with the run button
            self._require_ready()
        self._session.text_logger().info(
            f"Skip of the rest of slice {plan.slice} requested: "
            + ", ".join(f"'{action.name}'" for action in plan.skipped)
            + "."
        )
        self._run.skip_slice()

    def terminate(self) -> None:
        """Abandon an interrupted run before the application closes."""
        self._session.text_logger().warning("Terminating workflow.")
        self._run.abort()
        close_all_log_files()

    def prepare_exit(self) -> None:
        """Close (flush) all log files before the process exits."""
        close_all_log_files()

    def add_action(self, type_key: str, name: str) -> Action:
        """
        Append a new action.

        Args:
            type_key: Registry key of the action type.
            name: Name of the new action.

        Returns:
            The new action.

        Raises:
            NotAllowedError: If `structure_editable` is false.
            InvalidActionNameError: If the name is invalid.
        """
        self._require_structure_editable()
        return self._session.add_action(type_key, name)

    def duplicate_action(self, action: Action) -> Action:
        """
        Append a copy of an action.

        Args:
            action: The action to duplicate.

        Returns:
            The copy.

        Raises:
            NotAllowedError: If `structure_editable` is false.
        """
        self._require_structure_editable()
        return self._session.duplicate_action(action)

    def remove_action(self, action: Action) -> OSError | None:
        """
        Remove an action and its directory.

        Args:
            action: The action to remove.

        Returns:
            The error if its directory could not be deleted, else None.

        Raises:
            NotAllowedError: If `structure_editable` is false.
        """
        self._require_structure_editable()
        return self._session.remove_action(action)

    def rename_action(self, action: Action, name: str) -> None:
        """
        Rename an action.

        Args:
            action: The action to rename.
            name: The new name.

        Raises:
            NotAllowedError: If `structure_editable` is false.
            InvalidActionNameError: If the name is invalid.
            ActionRenameError: If the directory could not be moved.
        """
        self._require_structure_editable()
        self._session.rename_action(action, name)

    def move_actions(self, start: int, count: int, destination: int) -> None:
        """
        Move a contiguous block of actions.

        Args:
            start: Index of the first moved action before the move.
            count: Number of moved actions.
            destination: Index of the first moved action after the move.

        Raises:
            NotAllowedError: If `structure_editable` is false.
        """
        self._require_structure_editable()
        self._session.move_actions(start, count, destination)

    def settings_committed(self, action: Action) -> None:
        """
        A settings form wrote a field (the read-only form is the gate).

        An action that is no longer in the workflow is ignored: its directory
        may now belong to a different action of the same name (import), whose
        settings file must not be overwritten.

        Args:
            action: The action whose settings were written.
        """
        if not any(a is action for a in self._session.actions):
            return
        self._session.commit_settings(action)

    def reset_action_state(self, action: Action) -> None:
        """
        Discard an action's internal state.

        An action that is no longer in the workflow is ignored: its directory
        may now belong to a different action of the same name (import).

        Args:
            action: The action.

        Raises:
            NotAllowedError: If `action_tools_enabled` is false.
        """
        self._check(
            self._permissions().action_tools_enabled,
            "Cannot reset the state of an action now.",
        )
        if not any(a is action for a in self._session.actions):
            return
        self._session.reset_action_state(action)

    def save_properties(
        self,
        action: Action,
        props: GlobalProperties,
        *,
        opened_slice: int,
        opened: GlobalProperties | None,
    ) -> None:
        """
        Store edited props for the action's current slice, unless they changed meanwhile.

        In danger mode the workflow may run while the props are edited; it
        may execute the action (moving it to the next slice) or propagate to
        it. Saving edits of props that are no longer the stored ones would
        silently undo that, so they are refused.

        Args:
            action: The action.
            props: The edited properties.
            opened_slice: The action's slice when the props were opened.
            opened: The props stored for that slice when they were opened
                (unedited), or None if none were stored.

        Raises:
            NotAllowedError: If `danger_tools_enabled` is false (Edit is
                usable in danger mode).
            PropertiesChangedError: If the action's slice or its stored props
                differ from what was opened. Nothing is written.
        """
        self._check(
            self._permissions().danger_tools_enabled, "Cannot save properties now."
        )

        stored = action.read_properties() if action.has_stored_properties() else None
        if action.ctx.slice != opened_slice or stored != opened:
            raise PropertiesChangedError(
                f"The properties of '{action.name}' changed after they were opened "
                "(e.g. the workflow ran it); the edits were not saved. Open them "
                "again to edit the current ones."
            )
        self._session.write_properties(action, props)

    def apply_microscope_settings(self, save_profile: bool) -> None:
        """
        Persist the microscope settings edited in the dialog.

        Args:
            save_profile: Also remember them as the last microscope profile.
        """
        self._session.commit_microscope_settings()
        if save_profile:
            self._user_config.save_microscope_profile(self._session.microscope.settings)

    def prepare_action(self, action: Action) -> TaskHandle:
        """
        Prepare an action on the microscope task thread.

        Args:
            action: The action.

        Returns:
            The task's handle.
        """
        session = self._session
        return self._submit_action_task(
            action,
            f"Preparing '{action.name}'",
            action.prepare,
            then=lambda _: session.props_changed(action),
        )

    def apply_properties(self, action: Action) -> TaskHandle:
        """
        Set the action's stored props on the microscope, on the task thread.

        Args:
            action: The action.

        Returns:
            The task's handle.
        """
        return self._submit_action_task(
            action,
            f"Applying properties of '{action.name}'",
            action.read_and_set_properties,
        )

    def test_action(self, action: Action) -> TaskHandle:
        """
        Test an action on the microscope task thread.

        Args:
            action: The action.

        Returns:
            The task's handle.
        """
        return self._submit_action_task(action, f"Testing '{action.name}'", action.test)

    def run_microscope_task(
        self, action: Action, description: str, work: Callable[[], Any]
    ) -> TaskHandle:
        """
        Run custom microscope work for an action's tool.

        Args:
            action: The action; lines are logged on its logger.
            description: E.g. "Previewing 'mask'"; logged as for Prepare.
            work: The work; executed on the task thread.

        Returns:
            The task's handle; connect `succeeded` / `failed` to bound methods.
        """
        return self._submit_action_task(action, description, work)

    def grab(self, request: PreviewRequest) -> TaskHandle:
        """
        Acquire an area-selector image (the forms' `PreviewPort`).

        Args:
            request: What to image.

        Returns:
            The task's handle; a refusal is a handle that emits `failed`.
        """
        if not self._policy.permissions.preview_enabled:
            return self._tasks.refuse(
                NotAllowedError("Cannot acquire an image while the microscope is busy.")
            )
        return self._previews.grab(request)

    def add_rule(self, parent: Action) -> PropagationRule:
        """
        Add an empty propagation rule.

        Args:
            parent: The parent action.

        Returns:
            The new rule.

        Raises:
            NotAllowedError: If `rules_editable` is false.
        """
        self._require_rules_editable()
        return self._session.add_rule(parent)

    def set_rule_dependents(self, rule: PropagationRule, names: Sequence[str]) -> None:
        """
        Replace a rule's dependents.

        Args:
            rule: The rule.
            names: Names of the dependent actions.

        Raises:
            NotAllowedError: If `rules_editable` is false.
        """
        self._require_rules_editable()
        self._session.set_rule_dependents(rule, names)

    def set_rule_properties(self, rule: PropagationRule, names: PropertyNames) -> None:
        """
        Replace the properties a rule propagates.

        Args:
            rule: The rule.
            names: The properties.

        Raises:
            NotAllowedError: If `rules_editable` is false.
        """
        self._require_rules_editable()
        self._session.set_rule_properties(rule, names)

    def remove_rule(self, rule: PropagationRule) -> None:
        """
        Delete a rule.

        Args:
            rule: The rule.

        Raises:
            NotAllowedError: If `rules_editable` is false.
        """
        self._require_rules_editable()
        self._session.remove_rule(rule)

    def _submit_action_task(
        self,
        action: Action,
        description: str,
        work: Callable[[], Any],
        then: Callable[[Any], None] | None = None,
    ) -> TaskHandle:
        if not self._policy.permissions.action_tools_enabled:
            return self._tasks.refuse(
                NotAllowedError(f"{description} is not possible now.")
            )

        return self._tasks.submit(
            MicroscopeTask(
                run=work,
                description=description,
                log=action.ctx.text_logger,
                then=then,
            )
        )

    def _permissions(self) -> Permissions:
        return self._policy.permissions

    @staticmethod
    def _check(allowed: bool, message: str) -> None:
        if not allowed:
            raise NotAllowedError(message)

    def _require_structure_editable(self) -> None:
        self._check(
            self._permissions().structure_editable,
            "Cannot change the workflow's actions now.",
        )

    def _require_rules_editable(self) -> None:
        self._check(
            self._permissions().rules_editable, "Cannot change propagation rules now."
        )

    def _require_ready(self) -> None:
        if not self._readiness.report.ready:
            raise NotAllowedError("The workflow is not ready to run.")

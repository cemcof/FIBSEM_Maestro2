# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import re
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, ClassVar, Protocol, Self

from fibsem_maestro.action.outcome import Produced, StepOutcome
from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.action_context.file import FileActionContext
from fibsem_maestro.logging.logging import logging_context
from fibsem_maestro.logging.text.contextual import ContextualTextLogger
from fibsem_maestro.microscope.microscope import Microscope
from fibsem_maestro.notifications.null_notifier import NullNotifier
from fibsem_maestro.settings.microscope_settings import MicroscopeSettings
from fibsem_maestro.workflow.actions import Actions
from fibsem_maestro.workflow.error import ActionError, WorkflowError
from fibsem_maestro.workflow.error_choice import ErrorChoice
from fibsem_maestro.workflow.propagations import PropagationRule, Propagations

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.notifications.notification_service import NotificationService


class WorkflowState(ActionState):
    action_names: list[str]
    action_types: list[str]
    slice_index: int
    propagations: list[PropagationRule]


class WorkflowCallbacks(Protocol):
    def notify_action_finished(self, finished_action_index: int) -> None: ...
    def notify_slice_finished(self) -> None: ...
    def notify_is_paused(self) -> None: ...
    def is_pause_requested(self) -> bool: ...
    def wait_for_resume(self) -> None: ...
    def request_error_choice(self, error: ActionError) -> ErrorChoice: ...

    def is_slice_skip_requested(self) -> bool:
        """Whether the user asked to skip the rest of the current slice."""
        ...

    def notify_slice_skipped(self, skipped: Sequence[Action]) -> None:
        """
        Announce that a requested skip of the rest of the slice is done.

        Args:
            skipped: The actions skipped; empty if none was left to skip.
        """
        ...


class Workflow:
    MAX_SLICE: ClassVar[int] = 9999
    """The last slice a workflow ever acquires."""

    def __init__(
        self,
        microscope: Microscope,
        actions: Actions,
        propagations: Propagations,
        context: ActionContext,
        notifier: NotificationService | None = None,
    ):
        """
        Orchestrates sequential execution of actions across multiple slices.

        Runs the workflow's action sequence once per slice, executing
        synchronization rules after each action to propagate property
        updates to dependent actions.

        Args:
            microscope: THe microscope instance.
            actions: The sequence of actions to execute each slice.
            propagations: The synchronization rules for property propagation.
            context: The action context for managing slice state.
            notifier: The service for sending notifications; notifications
                are discarded if None.
        """

        self.microscope = microscope
        self.actions = actions
        self.propagations = propagations
        self.ctx = context
        self.notifier: NotificationService = (
            notifier if notifier is not None else NullNotifier()
        )

        self.callbacks: WorkflowCallbacks | None = None

        self._discard_forbidden_rules()

    def set_callbacks(self, callbacks: WorkflowCallbacks) -> None:
        self.callbacks = callbacks

    @classmethod
    def import_from_dir(
        cls,
        dir: Path,
        microscope: Microscope,
        current_workflow_dir: Path,
        notifier: NotificationService,
    ) -> Self:
        # collect all available workflow slices, sorted descending
        workflow_dir = dir / "workflow"
        if not workflow_dir.is_dir():
            raise WorkflowError("No workflow directory found")

        workflow_slice_dirs = sorted(
            [
                (int(m.group(1)), p)
                for p in workflow_dir.glob("slice_*")
                if p.is_dir() and (m := re.search(r"(\d+)", p.name))
            ],
            reverse=True,
        )

        if not workflow_slice_dirs:
            raise WorkflowError("No workflow slice found in directory")

        # retry workflow slices until one loads successfully
        workflow_state: WorkflowState | None = None
        for _, workflow_path in workflow_slice_dirs:
            try:
                workflow_state = WorkflowState.from_file(workflow_path / "state.yaml")
                break
            except Exception:
                continue

        if workflow_state is None:
            raise WorkflowError(
                "Failed to load workflow state from any available slice"
            )

        workflow_ctx = FileActionContext(current_workflow_dir / "workflow", "workflow")

        # load actions - each retries its own slices independently
        actions = Actions()
        for name, type in zip(workflow_state.action_names, workflow_state.action_types):
            name_with_underscores = name.replace(" ", "_")

            # collect all available action slices, sorted descending
            action_subdir = dir / name_with_underscores
            if not action_subdir.is_dir():
                raise WorkflowError(f"No directory found for action {name}")

            action_slice_dirs = sorted(
                [
                    (int(m.group(1)), p)
                    for p in action_subdir.glob("slice_*")
                    if p.is_dir() and (m := re.search(r"(\d+)", p.name))
                ],
                reverse=True,
            )

            if not action_slice_dirs:
                raise WorkflowError(f"No slice found for action {name}")

            last_error: Exception | None = None
            action: Action | None = None
            for _, action_slice_path in action_slice_dirs:
                try:
                    ActionCls = ACTION_REGISTRY.get(type)
                    action_settings = ActionCls.settings_cls().from_file(
                        action_slice_path / "settings.yaml"
                    )
                    action_ctx = FileActionContext(
                        current_workflow_dir / name_with_underscores,
                        name,
                    )
                    action = ActionCls(
                        name, microscope, action_settings, action_ctx, actions
                    )
                    break
                except Exception as e:
                    last_error = e
                    continue

            if action is None:
                raise WorkflowError(
                    f"Failed to load action '{name}' from any available slice: {last_error}"
                ) from last_error

            actions.append(action)

        propagations = Propagations.from_rules(workflow_state.propagations)
        return cls(microscope, actions, propagations, workflow_ctx, notifier)

    @classmethod
    def import_from_dir_with_state(
        cls, dir: Path, notifier: NotificationService
    ) -> Self:
        # find the latest slice of the workflow
        if not (result := find_subdir_with_largest_int(dir / "workflow", "slice_*")):
            raise WorkflowError("No workflow slice found in directory")
        workflow_path, workflow_slice_number = result

        # load the workflow state and construct the workflow context
        workflow_state = WorkflowState.from_file(workflow_path / "state.yaml")
        workflow_ctx = FileActionContext(
            dir / "workflow",
            "workflow",
            slice=workflow_slice_number,
        )

        # load the microscope settings and construct the microscope
        microscope_settings = MicroscopeSettings.from_file(
            workflow_path / "microscope_settings.yaml"
        )
        microscope = Microscope(
            microscope_settings,
            ContextualTextLogger(fallback=workflow_ctx.text_logger).derive(
                "microscope"
            ),
        )

        # load the settings of actions and construct them
        actions = Actions()
        for name, type in zip(workflow_state.action_names, workflow_state.action_types):
            # find the latest slice of the action
            name_with_underscores = name.replace(" ", "_")
            if not (
                result := find_subdir_with_largest_int(
                    dir / name_with_underscores, "slice_*"
                )
            ):
                raise WorkflowError(f"No slice found for action {name}")
            action_slice_path, action_slice_number = result

            # load the action settings
            ActionCls = ACTION_REGISTRY.get(type)
            action_settings = ActionCls.settings_cls().from_file(
                action_slice_path / "settings.yaml"
            )
            action_ctx = FileActionContext(
                dir / name_with_underscores, name, slice=action_slice_number
            )

            # construct the action
            action = ActionCls(name, microscope, action_settings, action_ctx, actions)

            actions.append(action)

        # load the action states
        for action in actions:
            action.set_state(
                action.ctx.state_store.read("state.yaml", action.state_cls())
            )

        # build the propagations
        propagations = Propagations.from_rules(workflow_state.propagations)

        return cls(microscope, actions, propagations, workflow_ctx, notifier)

    def run(self, last_slice: int = MAX_SLICE) -> None:
        """
        Runs the acquisition workflow, continuing from the current slice.

        The run ends normally after `last_slice` has been acquired, or before
        starting a slice for which any action reports a completion reason
        (see `completion_reasons`).

        Args:
            last_slice: The last slice to acquire, at most `MAX_SLICE`.
        """
        with logging_context(self.ctx.text_logger):
            if self.ctx.slice == 0:
                # if this is slice 0, initialize all actions
                self.ctx.advance()
                self.ctx.state_store.write("state.yaml", self.state)
                self.ctx.settings_store.write(
                    "microscope_settings.yaml", self.microscope.settings
                )

                for action in self.actions:
                    try:
                        action.initialize_first_slice()
                    except Exception as e:
                        # log all exceptions raised during the initialization of the action
                        # and re-raise them to interrupt the workflow
                        action.ctx.text_logger.error(
                            f"Failed to initialize action '{action.name}': {e}"
                        )
                        raise WorkflowError(e) from e

            # a pause requested while a slice was skipped waits for the next slice
            pause_first = False
            while self.ctx.slice <= min(last_slice, self.MAX_SLICE):
                # completion is checked only between slices, so a slice that has
                # started (e.g. the one that used up the milling area) always ends
                if reasons := self.completion_reasons():
                    self.ctx.text_logger.info(
                        "Workflow finished: " + "; ".join(reasons)
                    )
                    return
                pause_first = self._run_slice(pause_first)

    def _run_slice(self, pause_first: bool = False) -> bool:
        """
        Executes all actions for a single slice and performs synchronization.

        A skip of the rest of the slice requested by the user is honoured
        before the next action starts, once some action of the slice has run
        (`WorkflowCallbacks.is_slice_skip_requested`). It takes precedence over
        a pause requested at the same time, which then takes effect at the
        start of the next slice.

        Args:
            pause_first: Pause, if requested, before any action of the slice
                runs; for a pause requested while the previous slice was skipped.

        Returns:
            True if a skip of the rest of the slice was requested and done.
        """

        self.ctx.text_logger.info(f"Starting slice {self.ctx.slice}.")
        # sleep for 1 ms to avoid overlapping slice log messages
        sleep(0.001)

        if pause_first:
            self._pause_if_requested()

        # check for issues with action preparation
        issues = self.preparation_issues()
        if issues:
            raise WorkflowError("Actions are not ready to run:\n" + "\n".join(issues))

        skipped = False
        while (action := self.next_pending_action()) is not None:
            # requested before any action of the slice ran (e.g. while the run
            # started), the first action still runs: a skip never empties a slice
            if self._slice_skip_requested() and self.slice_started():
                self._skip_rest_of_slice()
                skipped = True
                break

            # execute the action
            outcome = self._execute_with_recovery(action)
            self._finish_step(action, outcome)

            # a skip requested meanwhile goes first; a pause then waits for the next slice
            if not self._slice_skip_requested():
                self._pause_if_requested()
            self._notify_action_finished(action)

        if self._slice_skip_requested() and self.callbacks:
            # requested while the last action of the slice executed
            self.ctx.text_logger.info(
                f"No action of slice {self.ctx.slice} is left to skip."
            )
            self.callbacks.notify_slice_skipped([])
            skipped = True

        # at the end of each slice, increment the workflow slice counter
        self.ctx.advance()
        # store the state of the workflow and the microscope settings
        self.ctx.state_store.write("state.yaml", self.state)
        self.ctx.settings_store.write(
            "microscope_settings.yaml", self.microscope.settings
        )
        # verify consistency of the slice counter with actions
        for action in self.actions:
            if self.ctx.slice != action.ctx.slice:
                raise WorkflowError(
                    f"Desynchronization: workflow slice counter and '{action.name}' slice counter are out of sync"
                )

        if self.callbacks:
            self.callbacks.notify_slice_finished()
        return skipped

    def _finish_step(self, action: Action, outcome: StepOutcome) -> None:
        """
        Store what an action's step left behind and advance the action to the next slice.

        Args:
            action: The action that has just executed or been skipped.
            outcome: Its outcome; only `Produced` properties are propagated.

        Raises:
            WorkflowError: If the action's slice counter did not advance past
                the workflow's.
        """
        # store the state and the current settings of the action
        action.ctx.state_store.next.write("state.yaml", action.state)
        action.ctx.settings_store.next.write("settings.yaml", action.settings)

        # propagate properties to other actions, if the action was executed
        if isinstance(outcome, Produced):
            self.propagations.propagate(
                action, outcome.props, self.actions, self.ctx.text_logger
            )
            # explicit patches go last, so they win over rule-based propagation
            self.propagations.apply_patches(
                outcome.patches, self.actions, self.ctx.text_logger
            )

        # advance the slice counter for the action
        self.ctx.text_logger.debug(
            f"Advancing slice counter for action '{action.name}'."
        )
        action.ctx.advance()

        # the loop terminates only when no action is pending, so a counter that
        # fails to advance would spin forever
        if action.ctx.slice <= self.ctx.slice:
            raise WorkflowError(
                f"Action '{action.name}' did not advance its slice counter past "
                f"{self.ctx.slice}; refusing to loop."
            )

    def _slice_skip_requested(self) -> bool:
        """Whether the user asked to skip the rest of the current slice."""
        return self.callbacks is not None and self.callbacks.is_slice_skip_requested()

    def _skip_rest_of_slice(self) -> None:
        """
        Skip every action not yet run for the current slice, carrying its properties over.

        The skipped actions and the warnings about skipping them
        (`skip_warnings`) are logged.
        """
        skipped = self.pending_actions()
        self.ctx.text_logger.warning(
            f"Skipping the rest of slice {self.ctx.slice} at the user's request: "
            + ", ".join(f"'{action.name}'" for action in skipped)
            + "."
        )
        for warning in self.skip_warnings(skipped):
            self.ctx.text_logger.warning(warning)

        for action in skipped:
            outcome = action.skip("skipped with the rest of the slice by the user")
            self._finish_step(action, outcome)
            self._notify_action_finished(action)

        if self.callbacks:
            self.callbacks.notify_slice_skipped(skipped)

    def _pause_if_requested(self) -> None:
        """Pause, if the user asked to, once the actions' background threads are done."""
        if self.callbacks and self.callbacks.is_pause_requested():
            # all actions should wait for their background threads to finish
            for pending in self.actions:
                pending.wait_for_background_threads()
            # then we notify that the workflow is actually paused
            self.callbacks.notify_is_paused()
            # and wait for the resume signal
            self.callbacks.wait_for_resume()

    def _notify_action_finished(self, action: Action) -> None:
        """Announce an action that has just finished, unless it was removed from the workflow."""
        if self.callbacks:
            # find the index of the action in the workflow list
            index = next((i for i, a in enumerate(self.actions) if a is action), None)
            # the action may have been already removed from the workflow
            # in which case we do not notify
            if index is not None:
                self.callbacks.notify_action_finished(index)

    def preparation_issues(self) -> list[str]:
        """
        Describe what must still be prepared before the workflow can run a slice.

        Returns:
            One `"'<action name>': <issue>"` line per issue; empty if ready.
        """
        return [
            f"'{action.name}': {issue}"
            for action in self.actions
            for issue in action.preparation_issues()
        ]

    def _discard_forbidden_rules(self) -> None:
        """
        Remove the actions that take no part in propagation from every rule.
        """
        excluded = sorted(
            action.name
            for action in self.actions
            if not action.PROPAGATION_RULES_ALLOWED
        )

        if self.propagations.discard_rules_of(excluded):
            self.ctx.text_logger.warning(
                "Removed propagation rules from or to "
                + ", ".join(f"'{name}'" for name in excluded)
                + ": these actions take no part in propagation."
            )

    def completion_reasons(self) -> list[str]:
        """
        Describe why the acquisition is complete, according to the actions.

        Returns:
            One `"'<action name>': <reason>"` line per action reporting a reason;
            empty if the acquisition may continue.
        """
        return [
            f"'{action.name}': {reason}"
            for action in self.actions
            if (reason := action.completion_reason()) is not None
        ]

    def pending_actions(self) -> list[Action]:
        """
        The actions that have not yet run for the current slice.

        Returns:
            The actions whose slice counter has not advanced past the
            workflow's, in workflow order.
        """
        return [action for action in self.actions if action.ctx.slice <= self.ctx.slice]

    def slice_started(self) -> bool:
        """
        Whether some action has already run (or been skipped) for the current slice.

        Returns:
            True if an action's slice counter is past the workflow's.
        """
        return any(action.ctx.slice > self.ctx.slice for action in self.actions)

    def skip_warnings(self, skipped: Sequence[Action]) -> list[str]:
        """
        Why skipping some actions for the current slice deserves a warning.

        Args:
            skipped: The actions to skip.

        Returns:
            One `"'<action name>': <warning>"` line per skipped action
            warning about it (`Action.skip_warning`); empty if none does.
        """
        return [
            f"'{action.name}': {warning}"
            for action in skipped
            if (warning := action.skip_warning(skipped)) is not None
        ]

    def next_pending_action(self) -> Action | None:
        """
        Finds the first action that has not yet run for the current slice.

        The action list may be mutated (actions added or removed) while the
        workflow is paused, so the next action is resolved by rescanning the
        list rather than by a cached index or an active iterator.

        Returns:
            The first action whose slice counter has not yet advanced past the
            workflow slice counter, or None if every action currently in the
            workflow has been executed for this slice.
        """
        for action in self.actions:
            if action.ctx.slice <= self.ctx.slice:
                return action
        return None

    def _execute_with_recovery(self, action: Action) -> StepOutcome:
        """
        Execute an action, offering the user recovery options if it fails.

        Args:
            action: The action to execute.

        Returns:
            The action's outcome, or a carry-over if the user chose to skip it.

        Raises:
            ActionError: If the user chose to terminate, or there is no GUI
                to ask.
        """
        while True:
            try:
                return action.execute()
            except Exception as e:
                action.ctx.text_logger.error(f"Action '{action.name}' failed: {e}")

                try:
                    self.notifier.notify(
                        "FIBSEM Maestro Error", f"Action '{action.name}' failed: {e}"
                    )
                except Exception as notify_error:
                    action.ctx.text_logger.error(f"Failed to notify: {notify_error}")

                error = ActionError(action, str(e))
                if self.callbacks is None:
                    raise error from e

                match self.callbacks.request_error_choice(error):
                    case ErrorChoice.RETRY:
                        action.ctx.text_logger.warning(
                            f"Retrying action '{action.name}'."
                        )
                        continue
                    case ErrorChoice.SKIP:
                        return action.skip("failed and skipped by the user")
                    case ErrorChoice.TERMINATE:
                        raise error from e

    @property
    def state(self) -> WorkflowState:
        return WorkflowState(
            action_names=[action.name for action in self.actions],
            action_types=[
                ACTION_REGISTRY.key_of(type(action)) for action in self.actions
            ],
            slice_index=self.ctx.slice,
            propagations=self.propagations.rules,
        )


def find_subdir_with_largest_int(
    directory: str | Path, pattern: str
) -> tuple[Path, int] | None:
    """
    Find the subdirectory matching a glob pattern that contains the largest
    integer value in its name.

    Args:
        directory: The parent directory to search in.
        pattern:   Glob pattern to filter subdirectories (e.g. "slice_*").

    Returns:
        A (path, largest_int) tuple for the matching subdirectory with the
        largest integer in its name, or None if no matches are found.

    Raises:
        NotADirectoryError: If `directory` does not exist or is not a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {directory}")

    best_path: Path | None = None
    best_value: int | None = None

    for subdir in directory.glob(pattern):
        if not subdir.is_dir():
            continue

        integers = [int(m) for m in re.findall(r"\d+", subdir.name)]
        if not integers:
            continue

        candidate = max(integers)
        if best_value is None or candidate > best_value:
            best_value = candidate
            best_path = subdir

    if best_path is None or best_value is None:
        return None

    return best_path, best_value

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The run state machine and the worker thread executing the workflow."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PyQt6.QtCore import (
    Q_ARG,
    QMetaObject,
    QObject,
    Qt,
    QThread,
    pyqtSignal,
    pyqtSlot,
)

from fibsem_maestro.ui.app.errors import NotAllowedError
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.app.slice_skip import SliceSkipPlan, plan_slice_skip
from fibsem_maestro.workflow.error_choice import ErrorChoice
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.workflow.error import ActionError


class AbortedError(Exception):
    """Indicates that the workflow was aborted before completion."""


class _RunWorker(QObject):
    """
    Executes one run of the workflow on its own thread (implements `WorkflowCallbacks`).

    The GUI thread controls it through `pause`, `resume`, `request_slice_skip`,
    `submit_error_choice` and `abort`, which only set flags and `threading.Event`s. It never touches
    widgets or services; it only emits.

    Args:
        workflow: The workflow to run.
    """

    paused = pyqtSignal()
    action_finished = pyqtSignal(object)
    """The `Action` that has just finished."""
    slice_finished = pyqtSignal(int)
    """The workflow's slice counter after the slice."""
    action_error = pyqtSignal(object)
    """An `ActionError`; the worker then waits for `submit_error_choice`."""
    slice_skipped = pyqtSignal(object)
    """The list of `Action`s skipped at the user's request (empty if none was left)."""
    finished = pyqtSignal()
    """`Workflow.run` returned normally."""
    failed = pyqtSignal(object)
    """The exception that ended the run."""
    done = pyqtSignal()
    """The run is over, whatever the outcome; always the last signal."""

    def __init__(self, workflow: Workflow) -> None:
        super().__init__()
        self._workflow = workflow
        self._pause_requested = False
        self._slice_skip_requested = False
        self._resume_event = threading.Event()
        # start in resumed state
        self._resume_event.set()

        self._error_event = threading.Event()
        self._error_choice: ErrorChoice | None = None

        self._aborted = False

    def notify_action_finished(self, finished_action_index: int) -> None:
        """Announce the action that has just finished."""
        self.action_finished.emit(self._workflow.actions[finished_action_index])

    def notify_slice_finished(self) -> None:
        """Announce the workflow's slice counter after a slice."""
        self.slice_finished.emit(self._workflow.ctx.slice)

    def notify_is_paused(self) -> None:
        """Announce that the requested pause has taken effect."""
        self.paused.emit()

    def is_pause_requested(self) -> bool:
        """Whether the workflow should pause after the current action."""
        return self._pause_requested

    def is_slice_skip_requested(self) -> bool:
        """Whether the workflow should skip the rest of the slice before its next action."""
        return self._slice_skip_requested

    def notify_slice_skipped(self, skipped: Sequence[Action]) -> None:
        """
        Announce that the requested skip is done; a new skip may be requested.

        Args:
            skipped: The actions skipped; empty if none was left to skip.
        """
        self._slice_skip_requested = False
        self.slice_skipped.emit(list(skipped))

    def wait_for_resume(self) -> None:
        """
        Block until resumed or aborted.

        Raises:
            AbortedError: If the run was aborted while paused.
        """
        while not self._resume_event.wait(timeout=0.1):
            pass

        if self._aborted:
            raise AbortedError()

    def request_error_choice(self, error: ActionError) -> ErrorChoice:
        """
        Ask the GUI how to handle a failed action.

        Called on the worker thread. Blocks until the GUI thread answers via
        submit_error_choice, or the workflow is aborted.

        Args:
            error: The failure.

        Returns:
            The user's choice.

        Raises:
            AbortedError: If the run was aborted while waiting.
        """
        self._error_choice = None
        self._error_event.clear()
        self.action_error.emit(error)

        while not self._error_event.wait(timeout=0.1):
            pass

        if self._aborted:
            raise AbortedError()

        assert self._error_choice is not None
        return self._error_choice

    def submit_error_choice(self, choice: ErrorChoice) -> None:
        """Called on the GUI thread to unblock request_error_choice."""
        self._error_choice = choice
        self._error_event.set()

    def pause(self) -> None:
        """Request a pause after the current action."""
        self._pause_requested = True
        self._resume_event.clear()

    def request_slice_skip(self) -> None:
        """Request skipping the rest of the slice before the next action starts."""
        self._slice_skip_requested = True

    def resume(self) -> None:
        """Release a pause (or cancel a pending one)."""
        self._pause_requested = False
        self._resume_event.set()

    def abort(self) -> None:
        """Make a paused or waiting run end with `AbortedError`."""
        self._aborted = True
        # unblock wait_for_resume and request_error_choice
        self._resume_event.set()
        self._error_event.set()

    @pyqtSlot(int)
    def run(self, last_slice: int) -> None:
        """
        Run the workflow up to `last_slice`, with this worker as its callbacks.

        Invoked on the worker's thread. Emits `finished` if the workflow
        returned normally, `failed` if it raised (an abort ends it silently),
        and `done` last in every case.

        Args:
            last_slice: The last slice to acquire.
        """
        try:
            self._workflow.set_callbacks(self)
            self._workflow.run(last_slice)
            self.finished.emit()
        except AbortedError:
            # clean exit
            pass
        except Exception as e:
            self.failed.emit(e)
        finally:
            self.done.emit()


class _ActiveRun:
    """
    The thread, worker and workflow of the current run.

    Args:
        thread: The run's thread.
        worker: The run's worker, living on `thread`.
        workflow: The workflow being run; `worker` is its callbacks.
    """

    def __init__(self, thread: QThread, worker: _RunWorker, workflow: Workflow) -> None:
        self.thread = thread
        # the worker has no Qt parent (it was moved to the thread), so this
        # reference keeps its Python wrapper, and the C++ object, alive until
        # the thread's `finished` deletes it
        self.worker = worker
        self.workflow = workflow


class RunController(QObject):
    """
    Owns the run state (`AppState`) and the worker thread of the current run.

    One fresh thread and worker exist per run; none exists while editing. The
    microscope task runner is locked exactly while the workflow executes
    (`AppState.is_running`). Calls not allowed in the current state raise
    `NotAllowedError`.

    Args:
        session: The workflow session; its workflow is read when a run starts.
        tasks: The microscope task runner, locked while the workflow executes.
        parent: Optional Qt parent.
    """

    state_changed = pyqtSignal(object)
    """The new `AppState`."""
    action_finished = pyqtSignal(object)
    """The `Action` that just finished on the worker."""
    slice_finished = pyqtSignal(int)
    """The workflow's slice counter after the slice."""
    action_failed = pyqtSignal(object)
    """An `ActionError`; the worker waits for `choose()` or `skip_and_pause()`."""
    workflow_failed = pyqtSignal(object)
    """The exception that ended the run."""
    slice_skipped = pyqtSignal(object)
    """The list of `Action`s skipped at the user's request."""

    def __init__(
        self,
        session: WorkflowSession,
        tasks: MicroscopeTasks,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._tasks = tasks
        self._state = AppState.EDITING if session.slice == 0 else AppState.RELOADED
        self._active: _ActiveRun | None = None
        # True while the worker waits in request_error_choice
        self._awaiting_choice = False
        # True from a skip request until the worker has skipped
        self._skip_pending = False

    @property
    def state(self) -> AppState:
        """The current run state."""
        return self._state

    def start(self) -> None:
        """
        Start a run from EDITING, RELOADED or FINISHED on a fresh worker thread.

        The run continues from wherever the workflow is and stops after
        `Workflow.MAX_SLICE` at the latest.

        Raises:
            NotAllowedError: In any other state, or while a microscope task runs.
        """
        self._start(skip_slice=False)

    def _start(self, skip_slice: bool) -> None:
        """
        Start a run (see `start`).

        Args:
            skip_slice: Skip the rest of the current slice first (SKIPPING
                instead of RUNNING).

        Raises:
            NotAllowedError: As `start`.
        """
        if self._state not in (AppState.EDITING, AppState.RELOADED, AppState.FINISHED):
            raise NotAllowedError(f"Cannot start the workflow while {self._state}.")

        self._refuse_if_task_busy()

        if self._active is not None and self._state is AppState.FINISHED:
            # `finished` is handled before `done`; join the run that just ended
            self._join_and_release()

        assert self._active is None

        workflow = self._session.workflow
        thread = QThread()
        thread.setObjectName("workflow-run")
        worker = _RunWorker(workflow)
        worker.moveToThread(thread)

        worker.paused.connect(self._on_paused)
        worker.action_finished.connect(self._on_action_finished)
        worker.slice_finished.connect(self._on_slice_finished)
        worker.action_error.connect(self._on_action_error)
        worker.slice_skipped.connect(self._on_slice_skipped)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.done.connect(self._on_done)
        # deleted on its own thread while that thread's event loop still runs; a
        # worker deleteLater-ed after wait() would never be deleted
        thread.finished.connect(worker.deleteLater)

        self._active = _ActiveRun(thread, worker, workflow)
        self._awaiting_choice = False
        if skip_slice:
            self._request_slice_skip(worker)
        thread.start()
        self._set_state(AppState.SKIPPING if skip_slice else AppState.RUNNING)
        QMetaObject.invokeMethod(
            worker,
            "run",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, Workflow.MAX_SLICE),
        )

    def pause(self) -> None:
        """
        Request a pause; it takes effect after the current action (RUNNING -> STOPPING).

        While SKIPPING, the skip goes first and the run pauses at the start of
        the next slice.

        Raises:
            NotAllowedError: If neither RUNNING nor SKIPPING.
        """
        if (
            self._state not in (AppState.RUNNING, AppState.SKIPPING)
            or self._active is None
        ):
            raise NotAllowedError(f"Cannot pause the workflow while {self._state}.")
        self._active.worker.pause()
        self._set_state(AppState.STOPPING)

    def resume(self) -> None:
        """
        Resume a paused run (PAUSED -> RUNNING).

        Raises:
            NotAllowedError: If not PAUSED, or while a microscope task runs.
        """
        if self._state is not AppState.PAUSED or self._active is None:
            raise NotAllowedError(f"Cannot resume the workflow while {self._state}.")
        self._refuse_if_task_busy()
        self._active.worker.resume()
        self._set_state(AppState.RUNNING)

    def choose(self, choice: ErrorChoice) -> None:
        """
        Answer a failed action with Retry or Skip (INTERRUPTED -> RUNNING).

        A pause still pending from before the error then takes effect as usual.

        Args:
            choice: `ErrorChoice.RETRY` or `ErrorChoice.SKIP`.

        Raises:
            NotAllowedError: If no action error awaits a choice, while a
                microscope task runs, or for `ErrorChoice.TERMINATE` (an
                interrupted run is abandoned with `abort()`).
        """
        if choice is ErrorChoice.TERMINATE:
            raise NotAllowedError("Terminate is not a choice for a failed action.")

        active = self._require_awaiting_choice()
        self._refuse_if_task_busy()
        self._awaiting_choice = False
        active.worker.submit_error_choice(choice)
        # a skip requested before the error is still to happen
        self._set_state(AppState.SKIPPING if self._skip_pending else AppState.RUNNING)

    def skip_and_pause(self) -> None:
        """
        Skip the failed action, then pause the workflow right after it (INTERRUPTED -> STOPPING).

        The pause is requested before the worker is released from waiting for
        the error choice, so it cannot run past its next pause check first.

        Raises:
            NotAllowedError: If no action error awaits a choice, or while a
                microscope task runs.
        """
        active = self._require_awaiting_choice()
        self._refuse_if_task_busy()
        self._awaiting_choice = False
        active.worker.pause()
        active.worker.submit_error_choice(ErrorChoice.SKIP)
        self._set_state(AppState.STOPPING)

    def planned_slice_skip(self) -> SliceSkipPlan | None:
        """
        What `skip_slice` would do now.

        Returns:
            The plan, or None if the rest of the slice cannot be skipped now
            (see `plan_slice_skip`).
        """
        return plan_slice_skip(
            self._session.workflow, self._state, self._awaiting_choice
        )

    def skip_slice(self) -> None:
        """
        Skip the rest of the current slice; the run then continues with the next slice.

        RUNNING: after the current action (-> SKIPPING). PAUSED: right away,
        resuming the run. RELOADED: right away, starting a run. INTERRUPTED: the
        failed action is skipped with the rest. The state is SKIPPING until the
        skip is done, then RUNNING (or PAUSED, if a pause was requested meanwhile).

        Raises:
            NotAllowedError: If `planned_slice_skip` is None, or (except while
                RUNNING) while a microscope task runs.
        """
        if self.planned_slice_skip() is None:
            raise NotAllowedError(
                f"Cannot skip the rest of the slice while {self._state}."
            )

        match self._state:
            case AppState.RELOADED:
                self._start(skip_slice=True)
                return
            case AppState.RUNNING:
                active = self._require_active()
                self._request_slice_skip(active.worker)
            case AppState.PAUSED:
                self._refuse_if_task_busy()
                active = self._require_active()
                self._request_slice_skip(active.worker)
                active.worker.resume()
            case AppState.INTERRUPTED:
                active = self._require_awaiting_choice()
                self._refuse_if_task_busy()
                self._awaiting_choice = False
                # requested first: the worker checks it right after the skip
                self._request_slice_skip(active.worker)
                active.worker.submit_error_choice(ErrorChoice.SKIP)
            case _:
                raise NotAllowedError(
                    f"Cannot skip the rest of the slice while {self._state}."
                )
        self._set_state(AppState.SKIPPING)

    def abort(self) -> None:
        """
        Unblock and join a worker that is paused or waiting for an error choice.

        Idempotent; does nothing without a worker. The state is unchanged.

        Raises:
            NotAllowedError: While RUNNING or STOPPING: an action cannot be
                stopped half-way.
        """
        if self._state.is_running:
            raise NotAllowedError("Cannot abort the workflow while an action executes.")
        if self._active is None:
            return
        self._active.worker.abort()
        self._join_and_release()

    def set_editing(self) -> None:
        """
        Return to EDITING after a reset or a new workflow.

        Raises:
            NotAllowedError: While RUNNING or STOPPING, or while a worker still
                exists (call `abort()` first).
        """
        if self._state.is_running or self._active is not None:
            raise NotAllowedError(f"Cannot return to editing while {self._state}.")
        self._awaiting_choice = False
        self._skip_pending = False
        self._set_state(AppState.EDITING)

    def _on_paused(self) -> None:
        if not self._from_active_worker():
            return
        # also from RUNNING: a pause still pending after an error choice
        if self._state.is_running:
            self._set_state(AppState.PAUSED)
        else:
            self._ignore("paused")

    def _on_action_finished(self, action: object) -> None:
        if self._from_active_worker():
            self.action_finished.emit(action)

    def _on_slice_finished(self, slice_index: int) -> None:
        if self._from_active_worker():
            self.slice_finished.emit(slice_index)

    def _on_action_error(self, error: ActionError) -> None:
        if not self._from_active_worker():
            return
        if self._state.is_running:
            self._awaiting_choice = True
            self._set_state(AppState.INTERRUPTED)
            self.action_failed.emit(error)
        else:
            self._ignore("action_error")

    def _on_slice_skipped(self, skipped: list[Action]) -> None:
        if not self._from_active_worker():
            return
        self._skip_pending = False
        if self._state is AppState.SKIPPING:
            self._set_state(AppState.RUNNING)
        if skipped:
            self.slice_skipped.emit(skipped)

    def _on_finished(self) -> None:
        if not self._from_active_worker():
            return
        if self._state.is_running:
            self._set_state(AppState.FINISHED)
        else:
            self._ignore("finished")

    def _on_failed(self, error: Exception) -> None:
        if not self._from_active_worker():
            return
        if self._state.is_running:
            self._set_state(AppState.INTERRUPTED)
            self.workflow_failed.emit(error)
        else:
            self._ignore("failed")

    def _on_done(self) -> None:
        if self._from_active_worker():
            self._join_and_release()

    def _join_and_release(self) -> None:
        """
        Stop the run's thread, wait for it and let Qt delete it.

        Afterwards the workflow no longer holds the worker as its callbacks.
        """
        assert self._active is not None
        active = self._active
        thread = active.thread
        thread.quit()
        # Blocking the GUI thread here cannot hang: this is only called when
        # the worker's `run` has returned or is returning (after `done`, after
        # `finished`) or right after `abort()` released its only blocking
        # waits (paused or awaiting a choice; never while an action executes).
        # The worker never waits for the GUI thread (its signals are queued),
        # so its event loop reaches `quit` promptly.
        thread.wait()
        thread.deleteLater()
        if active.workflow.callbacks is active.worker:
            # the worker is deleted with its thread; nothing may call into it
            active.workflow.callbacks = None
        self._active = None
        self._awaiting_choice = False
        self._skip_pending = False

    def _from_active_worker(self) -> bool:
        # a late signal of a previous run must not disturb the current one
        return self._active is not None and self.sender() is self._active.worker

    def _require_active(self) -> _ActiveRun:
        if self._active is None:
            raise NotAllowedError(f"No run to change while {self._state}.")
        return self._active

    def _request_slice_skip(self, worker: _RunWorker) -> None:
        self._skip_pending = True
        worker.request_slice_skip()

    def _require_awaiting_choice(self) -> _ActiveRun:
        if (
            self._state is not AppState.INTERRUPTED
            or not self._awaiting_choice
            or self._active is None
        ):
            raise NotAllowedError("No failed action is waiting for a choice.")
        return self._active

    def _refuse_if_task_busy(self) -> None:
        if self._tasks.busy:
            raise NotAllowedError("A microscope task is running.")

    def _set_state(self, state: AppState) -> None:
        self._state = state
        self._tasks.set_locked(state.is_running)
        self.state_changed.emit(state)

    def _ignore(self, event: str) -> None:
        self._session.text_logger().debug(
            f"Ignoring the run event '{event}' in state '{self._state}'."
        )

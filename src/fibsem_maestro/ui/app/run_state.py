# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The run state of the application."""

from enum import Enum


class AppState(Enum):
    """
    Where the workflow run stands; the value is what the top bar displays.

    STOPPING means a pause was requested and the current action is still executing.
    SKIPPING means a skip of the rest of the slice was requested; the run
    executes until the skip is done (e.g. after the current action), then
    pauses at the start of the next slice.
    """

    EDITING = "editing"
    RUNNING = "running"
    STOPPING = "stopping"
    SKIPPING = "skipping"
    PAUSED = "paused"
    RELOADED = "reloaded"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"

    def __str__(self) -> str:
        return self.value

    @property
    def is_running(self) -> bool:
        """True while the workflow thread executes actions, including while a pause or a skip is pending."""
        return self in (AppState.RUNNING, AppState.STOPPING, AppState.SKIPPING)

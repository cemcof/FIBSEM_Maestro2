# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The bar at the top of the main window: workflow buttons, run control and state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from fibsem_maestro.ui.app.errors import (
    DirectoryNotClearedError,
    NotAllowedError,
    WorkflowImportError,
    WorkflowResetError,
)
from fibsem_maestro.ui.app.readiness import readiness_tooltip
from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.ui.views.common import vertical_separator
from fibsem_maestro.ui.views.connection import workflow_directory
from fibsem_maestro.ui.views.dialogs.microscope_settings import (
    MicroscopeSettingsDialog,
)
from fibsem_maestro.ui.views.dialogs.slice_skip import confirm_slice_skip
from fibsem_maestro.ui.widgets.icon_button import IconButton
from fibsem_maestro.ui.widgets.icons import (
    ICON_IMPORT_WORKFLOW,
    ICON_MICROSCOPE,
    ICON_NEW_WORKFLOW,
    ICON_PAUSE,
    ICON_PLAY,
    ICON_SKIP_SLICE,
    ICON_SKULL,
    ICON_START_OVER,
    svg_icon,
)

if TYPE_CHECKING:
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.edit_policy import EditPolicy, Permissions
    from fibsem_maestro.ui.app.files import WorkflowFiles
    from fibsem_maestro.ui.app.readiness import Readiness, ReadinessReport
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.composition import FormContexts
    from fibsem_maestro.ui.forms.builder import FormBuilder

TASK_RUNNING_REASON = "A microscope task is running."
"""Run-tooltip line shown while a microscope task blocks starting or resuming."""

WAITING_FOR_CHOICE_TOOLTIP = "Waiting for a choice in the error dialog."
"""Run tooltip in INTERRUPTED."""

SKIPPING_RUN_TOOLTIP = "The workflow pauses once the rest of the slice is skipped."
"""Run tooltip in SKIPPING."""

NEW_NAME = "New workflow"
"""User-facing name of the "New workflow" button (`AppCommands.new_workflow`)."""

IMPORT_NAME = "Import workflow"
"""User-facing name of the "Import workflow" button (`AppCommands.import_workflow`)."""

START_OVER_NAME = "Start over"
"""User-facing name of `AppCommands.reset_workflow`, called "reset" in the code."""

SKIP_NAME = "Skip rest of slice"
"""User-facing name of `AppCommands.skip_slice`."""

MICROSCOPE_NAME = "Microscope settings"
"""User-facing name of the microscope settings button."""

NEW_TOOLTIP = f"{NEW_NAME}: start an empty workflow in a new directory."

IMPORT_TOOLTIP = (
    f"{IMPORT_NAME}: replace the actions and propagation rules by those of "
    "another workflow directory."
)

START_OVER_TOOLTIP = (
    f"{START_OVER_NAME}: delete the data of every slice and return to slice 0."
)

SKIP_TOOLTIP = (
    f"{SKIP_NAME}: skip the actions of the current slice that have not run yet."
)
"""Tooltip of the skip button while it is enabled."""

SKIP_STATE_REASON = (
    "Possible while the workflow runs, is paused or reloaded, or waits for a "
    "choice after an error."
)
"""Skip-tooltip line in the states that do not allow skipping."""

SKIP_NOTHING_REASON = (
    "Nothing to skip: no action of the current slice has run yet, or none is left."
)
"""Skip-tooltip line when the workflow has nothing to skip."""

_STATE_COLORS: dict[AppState, str] = {
    AppState.EDITING: "#f0c040",
    AppState.RUNNING: "#4caf50",
    AppState.SKIPPING: "#ab47bc",
    AppState.PAUSED: "#9e9e9e",
    AppState.RELOADED: "#9e9e9e",
    AppState.INTERRUPTED: "#f44336",
    AppState.FINISHED: "#3734eb",
}

_BLINKING: dict[AppState, AppState] = {
    AppState.STOPPING: AppState.PAUSED,
    AppState.SKIPPING: AppState.SKIPPING,
}
"""The states that blink, alternating the running colour with that of the given state."""

_RUN_ICON_SIZE = 20  # px


@dataclass(frozen=True)
class TopBarViewData:
    """
    What the top bar shows.

    Attributes:
        slice_text: The workflow's slice counter.
        state_text: The run state's text.
        color: Colour of both the slice and the state label.
        run_icon: Icon of the run button; None to leave it as it is (STOPPING).
        run_enabled: Whether the run button is enabled; None to leave it (STOPPING).
        run_tooltip: Tooltip of the run button ("" for none); None to leave it (STOPPING).
        new_enabled: Whether "New workflow" is enabled.
        import_enabled: Whether "Import workflow" is enabled.
        reset_enabled: Whether "Start over" is enabled.
        skip_enabled: Whether "Skip rest of slice" is enabled.
        skip_tooltip: Tooltip of "Skip rest of slice".
    """

    slice_text: str
    state_text: str
    color: str
    run_icon: Literal["play", "pause"] | None
    run_enabled: bool | None
    run_tooltip: str | None
    new_enabled: bool
    import_enabled: bool
    reset_enabled: bool
    skip_enabled: bool = False
    skip_tooltip: str = ""


def top_bar_view(
    state: AppState,
    slice_index: int,
    readiness: ReadinessReport,
    permissions: Permissions,
    frame: int,
    skip_planned: bool = False,
) -> TopBarViewData:
    """
    Decide what the top bar shows.

    Args:
        state: The run state.
        slice_index: The workflow's slice counter.
        readiness: Why the workflow cannot run, if it cannot.
        permissions: The current permissions (incl. whether a task runs).
        frame: The animation tick counter; STOPPING and SKIPPING blink on odd frames.
        skip_planned: Whether the workflow has something to skip
            (`RunController.planned_slice_skip`).

    Returns:
        The view data.
    """
    if (other := _BLINKING.get(state)) is not None:
        color = _STATE_COLORS[AppState.RUNNING if frame % 2 == 0 else other]
    else:
        color = _STATE_COLORS[state]

    run_icon: Literal["play", "pause"] | None
    run_enabled: bool | None
    run_tooltip: str | None
    match state:
        case AppState.STOPPING:
            # the pause is on its way; leave the button as it is
            run_icon, run_enabled, run_tooltip = None, None, None
        case AppState.RUNNING:
            run_icon, run_enabled, run_tooltip = "pause", True, ""
        case AppState.SKIPPING:
            # the run pauses by itself after the skip
            run_icon, run_enabled = "pause", False
            run_tooltip = SKIPPING_RUN_TOOLTIP
        case AppState.INTERRUPTED:
            run_icon, run_enabled = "play", False
            run_tooltip = WAITING_FOR_CHOICE_TOOLTIP
        case _:
            run_icon = "play"
            run_enabled = readiness.ready and not permissions.task_busy
            if run_enabled:
                run_tooltip = ""
            else:
                reasons = (
                    (TASK_RUNNING_REASON,) if permissions.task_busy else ()
                ) + readiness.reasons
                heading = (
                    "Cannot resume:" if state is AppState.PAUSED else "Cannot start:"
                )
                run_tooltip = readiness_tooltip(reasons, heading)

    skip_enabled, skip_tooltip = _skip_button(
        state, readiness, permissions, skip_planned
    )
    return TopBarViewData(
        slice_text=str(slice_index),
        state_text=str(state),
        color=color,
        run_icon=run_icon,
        run_enabled=run_enabled,
        run_tooltip=run_tooltip,
        new_enabled=permissions.new_enabled,
        import_enabled=permissions.import_enabled,
        reset_enabled=permissions.reset_enabled,
        skip_enabled=skip_enabled,
        skip_tooltip=skip_tooltip,
    )


def _skip_button(
    state: AppState,
    readiness: ReadinessReport,
    permissions: Permissions,
    skip_planned: bool,
) -> tuple[bool, str]:
    """
    Whether "Skip rest of slice" is enabled, and its tooltip.

    Args:
        state: The run state.
        readiness: Why the workflow cannot run; a paused or reloaded workflow
            resumes or starts running when skipping.
        permissions: The current permissions.
        skip_planned: Whether the workflow has something to skip.

    Returns:
        Whether it is enabled; the tooltip, listing the reasons if it is not.
    """
    reasons: tuple[str, ...]
    if not permissions.slice_skip_enabled:
        reasons = (
            (TASK_RUNNING_REASON,)
            if permissions.task_busy and not state.is_running
            else (SKIP_STATE_REASON,)
        )
    elif not skip_planned:
        reasons = (SKIP_NOTHING_REASON,)
    elif state in (AppState.PAUSED, AppState.RELOADED):
        reasons = readiness.reasons
    else:
        reasons = ()

    if reasons:
        # the button shows no text: the tooltip names it
        return False, f"{SKIP_NAME}\n" + readiness_tooltip(reasons, "Cannot skip:")
    return True, SKIP_TOOLTIP


class _RunButton(QPushButton):
    """
    The run button: a push button with an SVG play or pause icon.

    The icon is rendered in the button's text colour and re-rendered when the
    palette or style changes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon_body = ICON_PLAY
        self.setIconSize(QSize(_RUN_ICON_SIZE, _RUN_ICON_SIZE))
        self.setFixedSize(QSize(_RUN_ICON_SIZE + 8, _RUN_ICON_SIZE + 8))
        self._refresh_icon()

    @property
    def icon_body(self) -> str:
        """The SVG body of the icon shown."""
        return self._icon_body

    def set_icon_body(self, body: str) -> None:
        """
        Show another icon.

        Args:
            body: SVG elements on a 24x24 canvas.
        """
        if body != self._icon_body:
            self._icon_body = body
            self._refresh_icon()

    def changeEvent(self, e: QEvent | None) -> None:
        """Re-render the icon when the palette or style changes."""
        super().changeEvent(e)
        if e is not None and e.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            self._refresh_icon()

    def _refresh_icon(self) -> None:
        self.setIcon(
            svg_icon(
                self._icon_body,
                _RUN_ICON_SIZE,
                palette=self.palette(),
                device_pixel_ratio=self.devicePixelRatioF(),
            )
        )


class TopBar(QWidget):
    """
    Horizontal bar at the top of the main window.

    Args:
        session: The workflow session (slice, directory, microscope settings).
        run: The run controller (state).
        policy: The permissions and danger mode.
        readiness: Why the workflow cannot run, if it cannot.
        commands: Every user intent.
        files: Filesystem queries (whether a chosen directory is empty).
        form_builder: Builds the microscope settings form.
        form_contexts: The environments of forms; the microscope settings
            form's is built each time the dialog opens, since it names the
            current workflow's logger.
        parent: Parent widget.
    """

    def __init__(
        self,
        session: WorkflowSession,
        run: RunController,
        policy: EditPolicy,
        readiness: Readiness,
        commands: AppCommands,
        files: WorkflowFiles,
        form_builder: FormBuilder,
        form_contexts: FormContexts,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._run = run
        self._policy = policy
        self._readiness = readiness
        self._commands = commands
        self._files = files
        self._form_builder = form_builder
        self._form_contexts = form_contexts
        self._animation_frame = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._new_btn = IconButton(ICON_NEW_WORKFLOW, NEW_TOOLTIP, NEW_NAME)
        self._new_btn.clicked.connect(self._on_new_workflow)
        layout.addWidget(self._new_btn)

        self._import_btn = IconButton(ICON_IMPORT_WORKFLOW, IMPORT_TOOLTIP, IMPORT_NAME)
        self._import_btn.clicked.connect(self._on_import_workflow)
        layout.addWidget(self._import_btn)

        self._reset_btn = IconButton(
            ICON_START_OVER, START_OVER_TOOLTIP, START_OVER_NAME
        )
        self._reset_btn.clicked.connect(self._on_reset_workflow)
        layout.addWidget(self._reset_btn)

        # the tooltip follows the state (`TopBarViewData.skip_tooltip`)
        self._skip_btn = IconButton(ICON_SKIP_SLICE, SKIP_TOOLTIP, SKIP_NAME)
        self._skip_btn.clicked.connect(self._on_skip_slice)
        layout.addWidget(self._skip_btn)

        layout.addWidget(vertical_separator())

        self._microscope_btn = IconButton(
            ICON_MICROSCOPE, MICROSCOPE_NAME, MICROSCOPE_NAME
        )
        self._microscope_btn.clicked.connect(self._on_microscope_settings)
        layout.addWidget(self._microscope_btn)

        layout.addStretch()

        self._run_btn = _RunButton()
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        status_layout = QHBoxLayout()
        status_layout.setSpacing(1)
        status_layout.setContentsMargins(0, 0, 0, 0)

        self._slice_label = QLabel()
        slice_font = self._slice_label.font()
        slice_font.setPointSize(20)
        slice_font.setBold(True)
        self._slice_label.setFont(slice_font)
        self._slice_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter
        )
        self._slice_label.setFixedWidth(100)
        status_layout.addWidget(self._slice_label)

        self._state_label = QLabel()
        state_font = self._state_label.font()
        state_font.setPointSize(15)
        state_font.setBold(True)
        self._state_label.setFont(state_font)
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._state_label.setFixedWidth(120)
        status_layout.addWidget(self._state_label)

        layout.addLayout(status_layout)
        layout.addStretch()

        self._dir_label = QLabel(str(session.workflow_dir))
        self._dir_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )
        self._dir_label.setStyleSheet("color: #888888;")
        dir_font = self._dir_label.font()
        dir_font.setPointSize(9)
        self._dir_label.setFont(dir_font)
        layout.addWidget(self._dir_label)

        layout.addSpacing(12)
        self._danger_mode_btn = IconButton(
            ICON_SKULL,
            "Allow changing settings at any time. A change takes effect the next "
            "time an action reads it. This can break stuff!",
            "Danger mode",
        )
        self._danger_mode_btn.setCheckable(True)
        self._danger_mode_btn.setChecked(policy.danger_mode)
        # red while on; appended, so it wins over the hover and pressed looks
        self._danger_mode_btn.setStyleSheet(
            self._danger_mode_btn.styleSheet()
            + "QToolButton#iconButton:checked { background: #c62828; border-color: #c62828; }"
        )
        self._danger_mode_btn.toggled.connect(self._on_danger_toggled)
        layout.addWidget(self._danger_mode_btn)

        # blinks the STOPPING and SKIPPING states
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(500)
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._animation_timer.start()

        run.state_changed.connect(self._refresh)
        run.slice_finished.connect(self._refresh)
        # what is left to skip changes with every action, and with the structure
        run.action_finished.connect(self._refresh)
        session.structure_changed.connect(self._refresh)
        session.workflow_reset.connect(self._refresh)
        readiness.changed.connect(self._refresh)
        policy.changed.connect(self._refresh)
        policy.danger_mode_changed.connect(self._on_danger_mode_changed)
        session.workflow_replaced.connect(self._on_workflow_replaced)

        # applied at construction too, so the rules hold from start-up (D13)
        self._refresh()

    @property
    def run_button(self) -> QPushButton:
        """The run / pause button."""
        return self._run_btn

    @property
    def skip_button(self) -> IconButton:
        """The "Skip rest of slice" button."""
        return self._skip_btn

    @property
    def new_button(self) -> IconButton:
        """The "New workflow" button."""
        return self._new_btn

    @property
    def import_button(self) -> IconButton:
        """The "Import workflow" button."""
        return self._import_btn

    @property
    def reset_button(self) -> IconButton:
        """The "Start over" button (`AppCommands.reset_workflow`)."""
        return self._reset_btn

    @property
    def microscope_button(self) -> IconButton:
        """The "Microscope settings" button."""
        return self._microscope_btn

    @property
    def danger_button(self) -> IconButton:
        """The danger-mode toggle."""
        return self._danger_mode_btn

    @property
    def slice_label(self) -> QLabel:
        """The label showing the workflow's slice counter."""
        return self._slice_label

    @property
    def directory_label(self) -> QLabel:
        """The label showing the workflow directory."""
        return self._dir_label

    def view_data(self) -> TopBarViewData:
        """The view data for the current state of the services."""
        return top_bar_view(
            self._run.state,
            self._session.slice,
            self._readiness.report,
            self._policy.permissions,
            self._animation_frame,
            skip_planned=self._run.planned_slice_skip() is not None,
        )

    def _refresh(self, *_: object) -> None:
        """Apply the current view data."""
        data = self.view_data()

        self._slice_label.setText(data.slice_text)
        self._slice_label.setStyleSheet(f"color: {data.color};")
        self._state_label.setText(data.state_text)
        self._state_label.setStyleSheet(f"color: {data.color};")

        if data.run_icon is not None:
            self._run_btn.set_icon_body(
                ICON_PAUSE if data.run_icon == "pause" else ICON_PLAY
            )
        if data.run_enabled is not None:
            self._run_btn.setEnabled(data.run_enabled)
        if data.run_tooltip is not None:
            self._run_btn.setToolTip(data.run_tooltip)

        self._new_btn.setEnabled(data.new_enabled)
        self._import_btn.setEnabled(data.import_enabled)
        self._reset_btn.setEnabled(data.reset_enabled)
        self._skip_btn.setEnabled(data.skip_enabled)
        self._skip_btn.setToolTip(data.skip_tooltip)

    def _on_animation_tick(self) -> None:
        """Advance the animation frame and re-render."""
        self._animation_frame += 1
        self._refresh()

    def _on_danger_mode_changed(self, enabled: bool) -> None:
        self._danger_mode_btn.setChecked(enabled)

    def _on_workflow_replaced(self) -> None:
        self._dir_label.setText(str(self._session.workflow_dir))

    def _on_new_workflow(self) -> None:
        directory = workflow_directory.choose_new(self, self._files)
        if directory is None:
            return

        try:
            self._commands.new_workflow(directory)
        except DirectoryNotClearedError as e:
            workflow_directory.report_not_cleared(self, e)
        except OSError as e:
            self._session.text_logger().error(
                f"Could not create the new workflow in {directory}: {e}"
            )
            QMessageBox.critical(
                self, "Error", f"Could not create the new workflow: {e}"
            )

    def _on_import_workflow(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select workflow directory")
        if not path:
            return

        try:
            self._commands.import_workflow(Path(path))
        except WorkflowImportError as e:
            QMessageBox.critical(self, "Error", f"Could not import workflow: {e}")

    def _on_reset_workflow(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Data will be lost!",
            "Starting over PERMANENTLY deletes the data of every slice (images, "
            "logs and stored properties) and returns the workflow to slice 0. "
            "The actions and their settings are kept.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            self._commands.reset_workflow()
        except WorkflowResetError as e:
            QMessageBox.critical(self, "Error", f"Could not start over: {e}")

    def _on_microscope_settings(self) -> None:
        """Open the microscope settings dialog."""
        dialog = MicroscopeSettingsDialog(
            session=self._session,
            commands=self._commands,
            permissions=self._policy.permissions,
            form_builder=self._form_builder,
            form_context=self._form_contexts.for_microscope_settings(),
            parent=self,
        )
        dialog.exec()
        dialog.deleteLater()

    def _on_run(self) -> None:
        try:
            self._commands.run_pressed()
        except NotAllowedError:
            # readiness or a task changed since the button was enabled; the
            # button follows on the next refresh
            self._refresh()

    def _on_skip_slice(self) -> None:
        plan = self._run.planned_slice_skip()
        if plan is None or not confirm_slice_skip(self, plan):
            self._refresh()
            return
        try:
            self._commands.skip_slice()
        except NotAllowedError as e:
            # the run moved on while the user decided (e.g. the slice ended)
            QMessageBox.information(self, "Not skipped", str(e))
        self._refresh()

    def _on_danger_toggled(self, enabled: bool) -> None:
        self._commands.set_danger_mode(enabled)

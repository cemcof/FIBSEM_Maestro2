# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Fixtures shared by every GUI test (see README.md).

Being a `conftest.py` in a directory without `__init__.py`, this file puts
`tests/ui` on `sys.path`, so tests import their helpers as `gui_testkit`.
"""

from __future__ import annotations

import os

# GUI tests never need a display; set before anything creates a QApplication
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import TYPE_CHECKING

import pytest
from gui_testkit.app import Services, build_services, shut_down
from gui_testkit.dialogs import DirectoryChoices, ExitCalls, MessageBoxes
from gui_testkit.situations import HeldStep, hold_first_step
from gui_testkit.workflows import seed_for, simulated_workflow
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from fibsem_maestro.app_paths import HOME_VARIABLE, AppPaths
from fibsem_maestro.logging.text.file import close_all_log_files
from fibsem_maestro.ui.app.files import WorkflowFiles
from fibsem_maestro.ui.app.session import WorkflowSession
from fibsem_maestro.ui.app.user_config import UserConfig
from fibsem_maestro.ui.composition import Composition, compose
from fibsem_maestro.ui.forms.builder import FormBuilder
from fibsem_maestro.ui.forms.context import FormContext
from fibsem_maestro.ui.forms.registry import default_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.workflow.workflow import Workflow


@pytest.fixture(autouse=True)
def maestro_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppPaths:
    """The FIBSEM Maestro folder (profiles, logs) of this test, inside `tmp_path`."""
    home = tmp_path / "maestro_home"
    monkeypatch.setenv(HOME_VARIABLE, str(home))
    return AppPaths(home)


@pytest.fixture(autouse=True)
def exits(monkeypatch: pytest.MonkeyPatch) -> ExitCalls:
    """`os._exit` only records its code: no test may end the process."""
    calls = ExitCalls()
    monkeypatch.setattr(os, "_exit", calls.codes.append)
    return calls


@pytest.fixture(autouse=True)
def boxes(monkeypatch: pytest.MonkeyPatch) -> MessageBoxes:
    """Message boxes never block: they are recorded and answered with `boxes.answer`."""
    stub = MessageBoxes()
    for kind in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, stub.recorder(kind))
    return stub


@pytest.fixture(autouse=True)
def directory_choice(monkeypatch: pytest.MonkeyPatch) -> DirectoryChoices:
    """Directory dialogs never block: they return `directory_choice.path` ("" = cancel)."""
    stub = DirectoryChoices()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", stub.choose)
    return stub


@pytest.fixture(autouse=True)
def _close_log_files() -> Iterator[None]:
    # log files are process-wide; handles into a finished test's tmp_path must not leak
    yield
    close_all_log_files()


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    """The workflow directory of this test."""
    directory = tmp_path / "workflow_dir"
    directory.mkdir()
    return directory


@pytest.fixture
def workflow(workflow_dir: Path) -> Workflow:
    """An empty workflow on the simulated microscope."""
    return simulated_workflow(workflow_dir)


@pytest.fixture
def files() -> WorkflowFiles:
    """Filesystem operations."""
    return WorkflowFiles()


@pytest.fixture
def user_config(maestro_home: AppPaths) -> UserConfig:
    """The remembered profiles, in this test's FIBSEM Maestro folder."""
    return UserConfig(maestro_home)


@pytest.fixture
def session(
    qtbot: QtBot, workflow: Workflow, workflow_dir: Path, files: WorkflowFiles
) -> WorkflowSession:
    """A session around the empty workflow."""
    _ = qtbot  # a QApplication must exist before QObjects are created
    return WorkflowSession(workflow, workflow_dir, files)


@pytest.fixture
def add_action(session: WorkflowSession) -> Callable[..., Action]:
    """
    Add an action through the session, optionally preparing it synchronously.

    Use it to set up state; to test Prepare itself, go through the commands.
    """

    def add(type_key: str, name: str, *, prepare: bool = False) -> Action:
        action = session.add_action(type_key, name)
        if prepare:
            action.prepare()
            session.props_changed(action)
        return action

    return add


@pytest.fixture
def services(
    qtbot: QtBot,
    session: WorkflowSession,
    user_config: UserConfig,
    files: WorkflowFiles,
) -> Iterator[Services]:
    """The whole application layer without views; no thread survives the test."""
    built = build_services(session, user_config, files)
    yield built
    shut_down(qtbot, built.run, built.tasks)


@pytest.fixture
def app(
    qtbot: QtBot,
    workflow: Workflow,
    workflow_dir: Path,
    user_config: UserConfig,
    files: WorkflowFiles,
) -> Iterator[Composition]:
    """The composed and shown main window around the empty workflow."""
    composition = compose(seed_for(workflow, workflow_dir), user_config, files)
    qtbot.addWidget(composition.window)
    composition.window.show()
    yield composition
    shut_down(qtbot, composition.run, composition.tasks)


@pytest.fixture
def form_builder() -> FormBuilder:
    """A form builder with the default editor registry."""
    return FormBuilder(default_registry())


@pytest.fixture
def form_context() -> FormContext:
    """An empty form context (no workflow, no microscope)."""
    return FormContext()


@pytest.fixture
def hold(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., HeldStep]]:
    """
    Hold the first step of an action until released (`hold_first_step`).

    Every held step is released when the test ends, so that no worker waits.
    """
    held: list[HeldStep] = []

    def install(action: Action, fail: bool = False) -> HeldStep:
        step = hold_first_step(monkeypatch, action, fail)
        held.append(step)
        return step

    yield install
    for step in held:
        step.release.set()

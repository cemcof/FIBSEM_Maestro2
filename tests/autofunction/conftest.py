# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action_context.file import FileActionContext
from fibsem_maestro.logging.text.contextual import ContextualTextLogger
from fibsem_maestro.logging.text.memory import MemoryTextLogger
from fibsem_maestro.microscope.microscope import Microscope
from fibsem_maestro.settings.microscope_settings import MicroscopeSettings
from fibsem_maestro.workflow.actions import Actions
from fibsem_maestro.workflow.propagations import Propagations
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fibsem_maestro.action.action import Action


@pytest.fixture
def simulated_microscope() -> Microscope:
    return Microscope(
        MicroscopeSettings(control="simulated", ip_address="localhost"),
        MemoryTextLogger(lambda: 0),
    )


@pytest.fixture
def workflow(tmp_path: Path) -> Workflow:
    """An empty workflow on the simulated microscope, writing into `tmp_path`."""
    ctx = FileActionContext(tmp_path / "workflow", "workflow")
    microscope = Microscope(
        MicroscopeSettings(control="simulated", ip_address="localhost"),
        ContextualTextLogger(fallback=ctx.text_logger).derive("microscope"),
    )
    return Workflow(microscope, Actions(), Propagations(), ctx)


@pytest.fixture
def add_action(workflow: Workflow, tmp_path: Path) -> Callable[..., Action]:
    """Add an action, prepared to run, at the end of the workflow or at `index`."""

    def add(type_key: str, name: str, index: int | None = None, **settings: object):
        action_cls = ACTION_REGISTRY.get(type_key)
        action = action_cls(
            name=name,
            microscope=workflow.microscope,
            settings=action_cls.settings_cls()(**settings),
            ctx=FileActionContext(tmp_path / name, name, slice=workflow.ctx.slice),
            actions=workflow.actions,
        )
        action.write_properties(action.collect_properties())
        action.ctx.state_store.write("state.yaml", action.state)
        action.ctx.settings_store.write("settings.yaml", action.settings)
        workflow.actions.insert(
            len(workflow.actions) if index is None else index, action
        )
        return action

    return add

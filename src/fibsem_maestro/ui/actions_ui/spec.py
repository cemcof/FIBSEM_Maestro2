# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Per-action user interface: which tools an action's header shows and which extra
sections its panel holds.

Every action gets the default tools (Prepare, Edit, Apply, Test) and no extra
section unless an `ActionUiSpec` is registered for its registry key:

```python
ACTION_UI.add("segmentation", ActionUiSpec(tools=(*DEFAULT_TOOLS, preview_tool)))
```

Modules under `ui/actions_ui/` are imported with the package, so registering
at module level is enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.core.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Callable

    from PyQt6.QtWidgets import QWidget

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.settings.reactive import ChangePath, ReactiveNode
    from fibsem_maestro.ui.app.action_status import ActionStatus
    from fibsem_maestro.ui.app.commands import AppCommands
    from fibsem_maestro.ui.app.edit_policy import EditPolicy
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.forms.builder import FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext


@dataclass(frozen=True)
class ToolContext:
    """
    What a tool may use when it is invoked.

    Attributes:
        action: The action whose header holds the tool.
        commands: Every user intent; the only way to change anything.
        session: The workflow session, for queries only.
        form_builder: Builds forms, e.g. for a dialog the tool opens.
        form_context: The environment of forms about `action` (its logger and
            the microscope catalogs).
        parent: Parent widget for dialogs.
        refresh: Re-renders the header; for a tool that finds a precondition
            gone that changed without any signal (e.g. a deleted props file).
        policy: The permissions and danger mode, e.g. to skip a confirmation
            in danger mode.
    """

    action: Action
    commands: AppCommands
    session: WorkflowSession
    form_builder: FormBuilder
    form_context: FormContext
    parent: QWidget
    refresh: Callable[[], None]
    policy: EditPolicy


@dataclass(frozen=True)
class ToolSpec:
    """
    One button of an action's header.

    A tool is enabled iff the run state and microscope tasks allow microscope
    work (if `uses_microscope`) or danger mode is on (if
    `usable_in_danger_mode`), the action has props stored for its current
    slice (if `requires_props`), and `enabled_when(action)` holds (if given).

    Attributes:
        key: Unique among the action's tools, e.g. "prepare".
        icon: SVG body for the `IconButton` (24x24 canvas).
        tooltip: Rich text shown on hover.
        accessible_name: Name announced by screen readers.
        requires_props: Disabled until props are stored for the current slice.
        invoke: Called on the GUI thread when the button is clicked.
        uses_microscope: If False, not disabled by runs or tasks (e.g. a viewer).
        enabled_when: Extra, tool-specific condition; None for none.
        usable_in_danger_mode: If True, also enabled in danger mode while the
            workflow runs or a microscope task runs. The commands the tool calls
            must allow it too: a microscope task is refused whenever the
            workflow runs, whatever this says.
    """

    key: str
    icon: str
    tooltip: str
    accessible_name: str
    requires_props: bool
    invoke: Callable[[ToolContext], None]
    uses_microscope: bool = True
    enabled_when: Callable[[Action], bool] | None = None
    usable_in_danger_mode: bool = False


@dataclass(frozen=True)
class SectionContext:
    """
    What an extra panel section may use when it is built.

    Sections connect `run`, `status` and `policy` signals only to bound methods
    of their own widgets, and watch settings with `subscribe`.

    Attributes:
        action: The action shown in the panel.
        commands: Every user intent; the only way to change anything.
        session: The workflow session, for queries only.
        run: The run controller, e.g. to refresh on `slice_finished`.
        status: The per-action indicators.
        policy: The permissions.
        form_builder: Builds forms.
        form_context: The environment of the action's settings forms.
    """

    action: Action
    commands: AppCommands
    session: WorkflowSession
    run: RunController
    status: ActionStatus
    policy: EditPolicy
    form_builder: FormBuilder
    form_context: FormContext


@dataclass(frozen=True)
class SectionSpec:
    """
    An extra section of an action's panel.

    Attributes:
        key: Unique among the action's sections.
        position: Where the section goes: right after the settings form, or
            after the propagation rules.
        build: Builds the section's widget; the panel owns it afterwards.
    """

    key: str
    position: Literal["after_settings", "after_propagations"]
    build: Callable[[SectionContext], QWidget]


def _default_tools() -> tuple[ToolSpec, ...]:
    # imported here: default_tools imports this module for ToolSpec
    from fibsem_maestro.ui.actions_ui.default_tools import DEFAULT_TOOLS

    return DEFAULT_TOOLS


@dataclass(frozen=True)
class ActionUiSpec:
    """
    The user interface of one action type beyond its settings form.

    Attributes:
        tools: The header's tool buttons, in order; Prepare, Edit, Apply and
            Test by default.
        sections: Extra panel sections.
    """

    tools: tuple[ToolSpec, ...] = field(default_factory=_default_tools)
    sections: tuple[SectionSpec, ...] = ()


ACTION_UI = Registry[ActionUiSpec]("action UI")
"""Per-action UI specs, keyed like `ACTION_REGISTRY`."""


def ui_spec_for(action: Action) -> ActionUiSpec:
    """
    The UI spec of an action.

    Args:
        action: The action.

    Returns:
        The spec registered for the action's registry key, or `ActionUiSpec()`
        (the default tools, no sections) if there is none.
    """
    try:
        key = ACTION_REGISTRY.key_of(type(action))
    except RegistryError:
        return ActionUiSpec()
    return ACTION_UI.get(key) if key in ACTION_UI else ActionUiSpec()


def subscribe(
    owner: QWidget, node: ReactiveNode, hook: Callable[[ChangePath], None]
) -> None:
    """
    Register a settings hook that is removed when `owner` is destroyed.

    Args:
        owner: The widget the hook belongs to.
        node: The reactive settings node to watch (e.g. `action.settings`).
        hook: Called with the path of every change below `node`.
    """
    unsubscribe = node.on_change_at(hook)
    # `destroyed` is emitted by the dying owner itself; the receiver must not be
    # a bound method of the owner (it is gone by then), so a plain callable
    owner.destroyed.connect(lambda *_: unsubscribe())

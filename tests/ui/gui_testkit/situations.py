# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Put the application layer into situations that are hard to reach otherwise."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.ui.app.links import ActionLink, iter_links
from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTask
from gui_testkit import TIMEOUT_MS

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.ports import TaskHandle


@contextmanager
def busy_tasks(qtbot: QtBot, tasks: MicroscopeTasks) -> Generator[TaskHandle]:
    """
    Keep the microscope task runner busy with a blocking task for the duration of the block.

    Args:
        qtbot: The Qt test helper.
        tasks: The task runner.

    Yields:
        The blocking task's handle; the task ends (successfully) when the block exits,
        and the runner is idle again afterwards.
    """
    release = threading.Event()
    handle = tasks.submit(
        MicroscopeTask(
            run=lambda: release.wait(TIMEOUT_MS / 1000), description="Blocking"
        )
    )
    assert tasks.busy, "the blocking task was refused"
    try:
        yield handle
    finally:
        release.set()
        qtbot.waitUntil(lambda: not tasks.busy, timeout=TIMEOUT_MS)


def store_unappliable_props(action: Action) -> None:
    """
    Store props for the action's current slice that the microscope refuses to apply.

    Executing the action then fails (an action error, as a real microscope fault
    would) until it is prepared again. Skipping it still works, because skipping
    only copies the props to the next slice.

    Args:
        action: The action; its props for its current slice are replaced.
    """
    action.write_properties(
        GlobalProperties(electron_beam=BeamProperties(no_such_setting=1.0))
    )


def link_pairs() -> list[tuple[str, str, str]]:
    """
    Every way one registered action can link to another, found through the schema.

    Returns:
        `(linking type key, link field name, linked type key)` for every
        action-selector field (single or multiple) and every registered type
        it offers. Set and read such a link with `link_to` and `linked_names`.
    """
    pairs: list[tuple[str, str, str]] = []
    for key in ACTION_REGISTRY:
        settings = ACTION_REGISTRY.get(key).settings_cls()()
        for link in iter_links(settings):
            pairs.extend(
                (key, link.field_name, target)
                for target in ACTION_REGISTRY
                if link.accepts(ACTION_REGISTRY.get(target))
            )
    return pairs


def link_field(settings: object, field: str) -> ActionLink:
    """
    The link a settings field is.

    Args:
        settings: The settings object.
        field: The name of a link field.

    Returns:
        The link.
    """
    return next(link for link in iter_links(settings) if link.field_name == field)


def link_to(settings: object, field: str, name: str) -> None:
    """
    Link one action, whether the field holds one name or a list of them.

    Args:
        settings: The settings object.
        field: The name of a link field.
        name: The name to link.
    """
    many = link_field(settings, field).many
    setattr(settings, field, [name] if many else name)


def linked_names(settings: object, field: str) -> list[str]:
    """
    The names a link field holds.

    Args:
        settings: The settings object.
        field: The name of a link field.

    Returns:
        The linked names, in order.
    """
    return link_field(settings, field).names(settings)

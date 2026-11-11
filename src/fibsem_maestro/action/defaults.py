# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.properties.beam_properties import BeamProperties


class ActionDefaults:
    """
    The `settings.dynamic_default.DefaultContext` of an action.

    Args:
        settings: The action's settings.
        actions: The workflow's actions, for the settings of linked actions.
        beam_properties: Returns the action's stored properties on its beam
            (`Action.stored_beam_properties`).
    """

    def __init__(
        self,
        settings: object,
        actions: Iterable[Action],
        beam_properties: Callable[[], BeamProperties | None],
    ) -> None:
        self._settings = settings
        self._actions = actions
        self._beam_properties = beam_properties

    @property
    def settings(self) -> object:
        """The action's settings."""
        return self._settings

    def settings_of(self, name: str) -> object | None:
        """
        The settings of another action of the workflow.

        Args:
            name: The action's name.

        Returns:
            Its settings, or None if the workflow has no such action.
        """
        action = next((a for a in self._actions if a.name == name), None)
        return None if action is None else action.settings

    def beam_properties(self) -> BeamProperties | None:
        """The action's stored properties on its beam, or None if none are stored."""
        return self._beam_properties()

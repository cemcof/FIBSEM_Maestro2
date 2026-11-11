# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Everything a form may need beyond the object it edits."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fibsem_maestro.ui.schema.choices import ChoiceContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.core.pattern_type import PatternType
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.properties.beam_properties import BeamProperties
    from fibsem_maestro.settings.dynamic_default import DefaultContext, UsedDefault
    from fibsem_maestro.ui.ports import ActionCatalog, PreviewPermission, PreviewPort


@dataclass(frozen=True)
class FormContext:
    """
    The environment a form is built in. Every part is optional.

    Attributes:
        log: Receives build-time warnings; None to drop them.
        actions: The workflow's actions, for action selectors and linked crop sources.
        manufacturer_property_names: Extra names offered by the property selectors.
        pattern_types: Choices of the pattern-type selector.
        previews: Acquires area-selector images; None when no microscope is connected.
        preview_permission: Whether area selectors may acquire images now.
        state: Returns the current internal state of the action whose settings
            the form edits, for overlays drawn from it
            (`OverlaySpec.state_sources`); None outside an action's panel.
        state_fields: Top-level fields of the root object whose change resets
            the internal state of that action (`Action.state_fields`); their
            labels are marked.
        beam_properties: Returns the stored properties on the beam of the
            action whose settings the form edits (`Action.stored_beam_properties`),
            for dynamic defaults computed from them; None outside an action's
            panel.
    """

    log: TextLogger | None = None
    actions: ActionCatalog | None = None
    manufacturer_property_names: tuple[str, ...] = ()
    pattern_types: tuple[PatternType, ...] = ()
    previews: PreviewPort | None = None
    preview_permission: PreviewPermission | None = None
    state: Callable[[], object] | None = None
    state_fields: frozenset[str] = frozenset()
    beam_properties: Callable[[], BeamProperties | None] | None = None

    def choice_context(self) -> ChoiceContext:
        """The parts `choices_for` needs."""
        return ChoiceContext(
            actions=tuple(self.actions.actions) if self.actions is not None else None,
            manufacturer_property_names=self.manufacturer_property_names,
            pattern_types=self.pattern_types,
        )

    def settings_of(self, name: str) -> object | None:
        """
        The settings of a workflow action, for computing dynamic defaults.

        A `settings.dynamic_default.SettingsLookup`.

        Args:
            name: The action's name.

        Returns:
            Its settings, or None outside a workflow or if there is no such action.
        """
        if self.actions is None:
            return None
        action = next((a for a in self.actions.actions if a.name == name), None)
        return None if action is None else action.settings

    def defaults_for(self, settings: object | None) -> DefaultContext:
        """
        What the dynamic defaults of the form's fields are computed from.

        Args:
            settings: The settings the form edits (its live root), or None.

        Returns:
            A `DefaultContext` over `settings`, the workflow's actions and the
            stored properties of the action whose settings are edited.
        """
        return _FormDefaults(self, settings)

    def warn(self, message: str) -> None:
        """
        Log a build-time warning, if a logger was supplied.

        Args:
            message: The warning text.
        """
        if self.log is not None:
            self.log.warning(message)


class _FormDefaults:
    """
    The `DefaultContext` of a form.

    Args:
        form: The form's environment.
        settings: The settings the form edits, or None.
    """

    def __init__(self, form: FormContext, settings: object | None) -> None:
        self._form = form
        self._settings = settings

    @property
    def settings(self) -> object | None:
        """The settings the form edits."""
        return self._settings

    def settings_of(self, name: str) -> object | None:
        """
        The settings of a workflow action.

        Args:
            name: The action's name.

        Returns:
            Its settings, or None outside a workflow or if there is no such action.
        """
        return self._form.settings_of(name)

    def beam_properties(self) -> BeamProperties | None:
        """The stored beam properties of the edited action, or None outside its panel."""
        source = self._form.beam_properties
        return None if source is None else source()

    def default_used(self, used: UsedDefault) -> None:
        """
        Ignore it: the forms only show defaults, the actions log what they use.

        Args:
            used: The field and the value it takes.
        """
        _ = used

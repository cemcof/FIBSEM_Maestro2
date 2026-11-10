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
    """

    log: TextLogger | None = None
    actions: ActionCatalog | None = None
    manufacturer_property_names: tuple[str, ...] = ()
    pattern_types: tuple[PatternType, ...] = ()
    previews: PreviewPort | None = None
    preview_permission: PreviewPermission | None = None
    state: Callable[[], object] | None = None
    state_fields: frozenset[str] = frozenset()

    def choice_context(self) -> ChoiceContext:
        """The parts `choices_for` needs."""
        return ChoiceContext(
            actions=tuple(self.actions.actions) if self.actions is not None else None,
            manufacturer_property_names=self.manufacturer_property_names,
            pattern_types=self.pattern_types,
        )

    def warn(self, message: str) -> None:
        """
        Log a build-time warning, if a logger was supplied.

        Args:
            message: The warning text.
        """
        if self.log is not None:
            self.log.warning(message)

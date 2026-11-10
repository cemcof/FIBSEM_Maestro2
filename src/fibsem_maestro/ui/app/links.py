# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Links between actions and action names.

A link is a top-level settings field edited with an action selector
(`WidgetType.ACTION_SELECTOR`); it holds the name of another action.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fibsem_maestro.settings.form_utils import WidgetType
from fibsem_maestro.ui.schema.choices import accepts_action_type
from fibsem_maestro.ui.schema.schema import get_field_infos

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fibsem_maestro.action.action import Action

_LINK_WIDGETS = frozenset(
    {WidgetType.ACTION_SELECTOR, WidgetType.MULTI_ACTION_SELECTOR}
)
"""The widgets of link fields."""


@dataclass(frozen=True)
class ActionLink:
    """
    One link field of a settings object.

    Attributes:
        field_name: Name of the settings field holding the linked action's name
            (or names).
        allowed_types: Action classes the link may name.
        excluded_types: Registry keys of action types the link may not name,
            even though they are among `allowed_types`.
        optional: True if the field may hold None (no link).
        many: True if the field holds a list of names.
    """

    field_name: str
    allowed_types: tuple[type[Action], ...]
    optional: bool
    many: bool = False
    excluded_types: tuple[str, ...] = ()

    def accepts(self, action_type: type[Action]) -> bool:
        """
        Whether the link may name actions of a type.

        Args:
            action_type: The action's class.

        Returns:
            True if the link's selector offers actions of `action_type`.
        """
        return accepts_action_type(action_type, self.allowed_types, self.excluded_types)

    def names(self, settings: object) -> list[str]:
        """
        The names the link holds on a settings object.

        Args:
            settings: The settings object holding the field.

        Returns:
            The linked names, in order; empty if there is no link.
        """
        value = getattr(settings, self.field_name)
        if value is None:
            return []
        return list(value) if self.many else [value]


def iter_links(settings: object) -> Iterator[ActionLink]:
    """
    Yield the link fields of a settings object, in declaration order.

    Only top-level fields are links; nested objects are not searched.

    Args:
        settings: A settings object (pydantic model or dataclass).

    Yields:
        One `ActionLink` per action-selector field.
    """
    for fi in get_field_infos(type(settings)):
        if fi.hint is None or fi.hint.widget not in _LINK_WIDGETS:
            continue
        yield ActionLink(
            field_name=fi.name,
            allowed_types=tuple(fi.hint.action_type_filter),
            optional=fi.optional,
            many=fi.hint.widget is WidgetType.MULTI_ACTION_SELECTOR,
            excluded_types=fi.hint.action_type_exclude,
        )


def retarget_links(settings: object, old_name: str, new_name: str) -> list[str]:
    """
    Point every link holding `old_name` at `new_name`.

    The fields are assigned on `settings` itself, so its change hooks fire. A
    list of names is assigned as a new list, with the name replaced wherever
    it occurs.

    Args:
        settings: A settings object (pydantic model or dataclass).
        old_name: The renamed action's previous name.
        new_name: Its new name.

    Returns:
        The names of the fields that were changed, in declaration order.
    """
    changed: list[str] = []
    for link in iter_links(settings):
        names = link.names(settings)
        if old_name not in names:
            continue
        renamed = [new_name if name == old_name else name for name in names]
        setattr(settings, link.field_name, renamed if link.many else new_name)
        changed.append(link.field_name)
    return changed


def validate_action_name(name: str, existing_names: set[str]) -> str | None:
    """
    Validate a candidate action name against existing names.

    Names are compared ignoring case, with spaces and underscores treated as
    equal, because such names map to the same action directory on a
    case-insensitive filesystem (Windows). The reserved name "workflow" (the
    workflow's own directory) is compared the same way.

    Args:
        name: The candidate name to validate.
        existing_names: Names already in use (excluding the name being edited, if any).

    Returns:
        An error message if invalid, or None if valid.
    """
    if not name:
        return "Name cannot be empty."
    key = _directory_key(name)
    if any(key == _directory_key(existing) for existing in existing_names):
        return f"Name '{name}' is already in use."
    if key == "workflow":
        return "Name 'workflow' is reserved."
    return None


def _directory_key(name: str) -> str:
    """The form of an action name under which two names share a directory."""
    return name.lower().replace(" ", "_")

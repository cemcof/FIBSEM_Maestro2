# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
The values a choice editor offers for a field.

Shared by the field editors and the settings audit, so that "which values can
this editor show" has a single definition.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.settings.form_utils import WidgetType
from fibsem_maestro.ui.schema.field_type import EnumType, LiteralType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.core.pattern_type import PatternType
    from fibsem_maestro.ui.schema.field_info import FieldInfo

CHOICE_WIDGETS: frozenset[WidgetType] = frozenset(
    {
        WidgetType.DROPDOWN,
        WidgetType.PROPERTY_SELECTOR,
        WidgetType.PATTERN_TYPE_SELECTOR,
        WidgetType.ACTION_SELECTOR,
        WidgetType.MULTI_ACTION_SELECTOR,
        WidgetType.MULTI_SELECT,
        WidgetType.MULTI_PROPERTY_SELECTOR,
    }
)
"""Hinted widgets whose values come from `choices_for`."""


@dataclass(frozen=True)
class ChoiceContext:
    """
    What choice lists depend on beyond the field itself.

    Attributes:
        actions: The workflow's actions, or None outside a workflow.
        manufacturer_property_names: Extra property names offered by the microscope.
        pattern_types: Milling pattern types offered by the microscope.
    """

    actions: Sequence[Action] | None = None
    manufacturer_property_names: tuple[str, ...] = ()
    pattern_types: tuple[PatternType, ...] = field(default_factory=tuple)


def unavailable_action_message(name: str) -> str:
    """
    The error shown for a link to an action that is not offered.

    Args:
        name: The linked name.

    Returns:
        The message, e.g. "'imaging' is not an available action.".
    """
    return f"'{name}' is not an available action."


def accepts_action_type(
    action_type: type[Action],
    allowed: Sequence[type[Action]],
    excluded: Sequence[str] = (),
) -> bool:
    """
    Whether an action selector offers actions of a type.

    Args:
        action_type: The action's class.
        allowed: The classes offered (`FormHint.action_type_filter`).
        excluded: Registry keys of types not offered, subclasses included
            (`FormHint.action_type_exclude`). Keys not registered exclude nothing.

    Returns:
        True if `action_type` is one of `allowed` and none of `excluded`.
    """
    if not issubclass(action_type, tuple(allowed)):
        return False
    return not any(
        key in ACTION_REGISTRY and issubclass(action_type, ACTION_REGISTRY.get(key))
        for key in excluded
    )


def has_choices(fi: FieldInfo) -> bool:
    """
    Whether `choices_for` applies to the field.

    Args:
        fi: The field.

    Returns:
        True for the hinted choice widgets and for Enum and Literal fields.
    """
    if fi.hint is not None:
        return fi.hint.widget in CHOICE_WIDGETS
    return isinstance(fi.type, EnumType | LiteralType)


def choices_for(fi: FieldInfo, ctx: ChoiceContext) -> list[Any]:
    """
    The values an editor of `fi` offers, without "(none)".

    Args:
        fi: The field.
        ctx: The workflow and microscope context.

    Returns:
        The choices, in the order the editor lists them.

    Raises:
        ValueError: If the field is not a choice field (see `has_choices`).
    """
    hint = fi.hint
    if hint is not None:
        listed = hint.choices() if hint.choices else []
        match hint.widget:
            case WidgetType.DROPDOWN | WidgetType.MULTI_SELECT:
                return list(listed)
            case WidgetType.PROPERTY_SELECTOR | WidgetType.MULTI_PROPERTY_SELECTOR:
                return list(listed) + list(ctx.manufacturer_property_names)
            case WidgetType.PATTERN_TYPE_SELECTOR:
                return list(ctx.pattern_types)
            case WidgetType.ACTION_SELECTOR | WidgetType.MULTI_ACTION_SELECTOR:
                return [
                    action.name
                    for action in ctx.actions or ()
                    if accepts_action_type(
                        type(action),
                        hint.action_type_filter,
                        hint.action_type_exclude,
                    )
                ]
            case _:
                raise ValueError(f"field {fi.name!r} is not a choice field")

    match fi.type:
        case EnumType(enum_type=enum_cls):
            return list(enum_cls)
        case LiteralType(choices=choices):
            return list(choices)
        case _:
            raise ValueError(f"field {fi.name!r} is not a choice field")

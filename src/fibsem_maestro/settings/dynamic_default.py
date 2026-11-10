# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Defaults computed from other settings, e.g. from the settings of a linked action.

A field with a dynamic default is declared `Annotated[T | None, DynamicDefault(...)]`
with the default None. None means "follow the default": the effective value is
computed whenever it is needed, so it changes with the settings it is computed
from. A value set by the user is stored and no longer follows. The action and
the forms both read the effective value through `effective_value`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

SettingsLookup = Callable[[str], object | None]
"""Returns the settings of the workflow's action with the given name, or None."""


@dataclass(frozen=True)
class DynamicDefault:
    """
    Marks a field whose default is computed from other settings.

    Attributes:
        compute: Called with the object holding the field and a lookup of the
            other actions' settings by name; returns the default, or None if it
            cannot be computed (e.g. no milling is linked).
        unavailable: Why there is no default when `compute` returns None, and
            what to do about it, e.g. "Link a milling, or set a value.".
        source: What the default is computed from, completing "Computed
            from …", e.g. "the linked milling's slice distance and direction".
            Shown in the forms beside a value following the default.
    """

    compute: Callable[[Any, SettingsLookup], Any]
    unavailable: str = "No default can be computed; set a value."
    source: str = "other settings"


def dynamic_default_of(cls: type, name: str) -> DynamicDefault | None:
    """
    The dynamic default declared on a field, if any.

    Args:
        cls: The settings class (a pydantic model).
        name: The field name.

    Returns:
        The declaration, or None if the field has none (or `cls` is no model).
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        return None
    field = cls.model_fields.get(name)
    if field is None:
        return None
    return next((m for m in field.metadata if isinstance(m, DynamicDefault)), None)


def effective_value(owner: object, name: str, lookup: SettingsLookup) -> Any:
    """
    The value of a field as it is used: the stored value, or else its dynamic default.

    Args:
        owner: The object holding the field.
        name: The field name.
        lookup: Returns the settings of another action by name.

    Returns:
        The stored value if it is not None or the field has no dynamic default;
        otherwise the computed default, which may be None if it cannot be computed.
    """
    value = getattr(owner, name)
    if value is not None:
        return value
    declared = dynamic_default_of(type(owner), name)
    return declared.compute(owner, lookup) if declared is not None else None

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from fibsem_maestro.settings.form_utils import FieldUnit

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.properties.beam_properties import BeamProperties


class DefaultContext(Protocol):
    """What a dynamic default may be computed from, beyond the object holding the field."""

    @property
    def settings(self) -> object | None:
        """The settings of the action the field belongs to, or None outside an action."""
        ...

    def settings_of(self, name: str) -> object | None:
        """
        The settings of another action of the workflow.

        Args:
            name: The action's name.

        Returns:
            Its settings, or None if there is no such action.
        """
        ...

    def beam_properties(self) -> BeamProperties | None:
        """
        The action's stored properties on its beam, for the current slice.

        Returns:
            The properties the action applies before it runs, or None if none
            are stored (the action was not prepared) or it works on no beam.
        """
        ...

    def default_used(self, used: UsedDefault) -> None:
        """
        Take note of a default computed to be used, e.g. log it.

        Not called for defaults only shown or checked (`effective_value`,
        `unavailable_defaults`).

        Args:
            used: The field and the value it takes.
        """
        ...


@dataclass(frozen=True)
class UsedDefault:
    """
    A default computed to be used.

    Attributes:
        path: The dotted path of the field within the settings the default
            was computed for, e.g. "sharpness_metric.detail" for a criterion's.
        value: The default.
        source: What it was computed from (`DynamicDefault.source`).
        unit: The field's unit, e.g. "nm", or None.
    """

    path: str
    value: Any
    source: str
    unit: str | None = None

    def __str__(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"'{self.path}' follows its default: {_shown(self.value)}{unit}, "
            f"computed from {self.source}."
        )


class DefaultUnavailableError(ValueError):
    """Raised when a field follows a default that cannot be computed."""


@dataclass(frozen=True)
class DynamicDefault:
    """
    Marks a field whose default is computed from other settings or from properties.

    Attributes:
        compute: Called with the object holding the field and the context;
            returns the default, or None if it cannot be computed (e.g. no
            milling is linked, no properties are stored).
        unavailable: Why there is no default when `compute` returns None, and
            what to do about it, e.g. "Link a milling, or set a value.". A
            callable gets the same arguments as `compute`, for a reason that
            depends on what is missing.
        source: What the default is computed from, completing "Computed
            from …", e.g. "the linked milling's slice distance and direction".
            Shown in the forms beside a value following the default.
    """

    compute: Callable[[Any, DefaultContext], Any]
    unavailable: str | Callable[[Any, DefaultContext], str] = (
        "No default can be computed; set a value."
    )
    source: str = "other settings"

    def reason(self, owner: object, context: DefaultContext) -> str:
        """
        Why the default cannot be computed now.

        Args:
            owner: The object holding the field.
            context: What the default is computed from.

        Returns:
            The `unavailable` text.
        """
        if isinstance(self.unavailable, str):
            return self.unavailable
        return self.unavailable(owner, context)


def properties_missing(quantity: str, value_word: str = "a value") -> str:
    """
    The reason for a default computed from properties that are not stored yet.

    Args:
        quantity: What the default is computed from, e.g. "pixel size".
        value_word: What the user may set instead, e.g. "a range".

    Returns:
        E.g. "Computed from the pixel size once the properties are collected
        (Prepare); or set a value.".
    """
    return (
        f"Computed from the {quantity} once the action is prepared; "
        f"or set {value_word}."
    )


def properties_not_collected(
    quantity: str, names: str, value_word: str = "a value"
) -> str:
    """
    The reason for a default computed from properties the action does not collect.

    Args:
        quantity: What the default is computed from, e.g. "field width".
        names: The properties providing it, e.g. "'horizontal_field_width'".
        value_word: What the user may set instead, e.g. "a range".

    Returns:
        E.g. "The stored properties lack the field width
        ('horizontal_field_width'); collect it and prepare again, or set a
        range.".
    """
    return (
        f"The stored properties lack the {quantity} ({names}); collect it and "
        f"prepare the action again, or set {value_word}."
    )


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


def effective_value(owner: object, name: str, context: DefaultContext) -> Any:
    """
    The value of a field as it is used: the stored value, or else its dynamic default.

    Args:
        owner: The object holding the field.
        name: The field name.
        context: What the default is computed from.

    Returns:
        The stored value if it is not None or the field has no dynamic default;
        otherwise the computed default, which may be None if it cannot be computed.
    """
    value = getattr(owner, name)
    if value is not None:
        return value
    declared = dynamic_default_of(type(owner), name)
    return declared.compute(owner, context) if declared is not None else None


def required_value(owner: object, name: str, context: DefaultContext) -> Any:
    """
    The effective value of a field with a dynamic default, which must be available.

    A computed default is reported to the context as used.

    Args:
        owner: The object holding the field.
        name: The field name.
        context: What the default is computed from.

    Returns:
        The stored value, or else the computed default.

    Raises:
        DefaultUnavailableError: If the field follows a default that cannot be
            computed; the message says why.
    """
    value = getattr(owner, name)
    if value is not None:
        return value
    declared = dynamic_default_of(type(owner), name)
    if declared is None:
        raise DefaultUnavailableError(f"'{name}': not set")
    value = declared.compute(owner, context)
    if value is None:
        raise DefaultUnavailableError(f"'{name}': {declared.reason(owner, context)}")
    context.default_used(_used(owner, name, name, value, declared))
    return value


def unavailable_defaults(
    settings: BaseModel, context: DefaultContext
) -> list[tuple[str, str]]:
    """
    The fields following a default that cannot be computed now.

    Walks the nested settings objects (the selected variants of unions, the
    objects of optional groups), not list elements.

    Args:
        settings: The settings to check.
        context: What the defaults are computed from.

    Returns:
        `(dotted path, reason)` per such field, in declaration order.
    """
    found: list[tuple[str, str]] = []
    # only checked, not used: nothing is reported
    _walk(settings, context, "", found, [])
    return found


def with_defaults[T: BaseModel](settings: T, context: DefaultContext) -> T:
    """
    A copy of the settings with every field following its default set to the default now.

    The copy is meant to be read, e.g. by a criterion run once; it is not
    validated again, and the settings themselves are left unchanged.

    Args:
        settings: The settings, at any depth (e.g. a criterion's settings).
        context: What the defaults are computed from.

    Returns:
        The copy, or `settings` itself if no field follows a default.

    Raises:
        DefaultUnavailableError: If a default cannot be computed; the message
            names every such field and says why.
    """
    missing: list[tuple[str, str]] = []
    used: list[UsedDefault] = []
    resolved = _walk(settings, context, "", missing, used)
    if missing:
        raise DefaultUnavailableError(
            "; ".join(f"'{path}': {reason}" for path, reason in missing)
        )
    for default in used:
        context.default_used(default)
    return resolved


def _walk[T: BaseModel](
    obj: T,
    context: DefaultContext,
    prefix: str,
    missing: list[tuple[str, str]],
    used: list[UsedDefault],
) -> T:
    """
    Fill the defaults of `obj` and of the objects nested in it.

    Args:
        obj: The object.
        context: What the defaults are computed from.
        prefix: The dotted path of `obj`, ending in "." unless empty.
        missing: Receives `(path, reason)` of each default that cannot be computed.
        used: Receives each default computed.

    Returns:
        A copy with the defaults filled in, or `obj` if nothing changed.
    """
    updates: dict[str, Any] = {}
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        declared = dynamic_default_of(type(obj), name)
        if value is None and declared is not None:
            default = declared.compute(obj, context)
            if default is None:
                missing.append((prefix + name, declared.reason(obj, context)))
            else:
                updates[name] = default
                used.append(_used(obj, name, prefix + name, default, declared))
        elif isinstance(value, BaseModel):
            resolved = _walk(value, context, f"{prefix}{name}.", missing, used)
            if resolved is not value:
                updates[name] = resolved
    if not updates:
        return obj
    # constructed, not copied: the copy gets no change hooks or parent of the original
    values = {name: getattr(obj, name) for name in type(obj).model_fields}
    return type(obj).model_construct(
        _fields_set=obj.model_fields_set, **(values | updates)
    )


def _used(
    owner: object, name: str, path: str, value: Any, declared: DynamicDefault
) -> UsedDefault:
    """The report of a default computed for the field `name` of `owner`."""
    field = type(owner).model_fields.get(name) if isinstance(owner, BaseModel) else None
    units = (
        [m.suffix for m in field.metadata if isinstance(m, FieldUnit)] if field else []
    )
    return UsedDefault(path, value, declared.source, units[0] if units else None)


def _shown(value: Any) -> str:
    """A value as written in a log message: floats short, dataclasses by their fields."""
    match value:
        case bool():
            return str(value)
        case float():
            return f"{value:.6g}"
        case Enum():
            return str(value.value)
        case tuple() | list():
            return "(" + ", ".join(_shown(v) for v in value) + ")"
        case BaseModel():
            names = type(value).model_fields
            return ", ".join(f"{n}={_shown(getattr(value, n))}" for n in names)
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return ", ".join(
                f"{f.name}={_shown(getattr(value, f.name))}"
                for f in dataclasses.fields(value)
            )
        case _:
            return str(value)

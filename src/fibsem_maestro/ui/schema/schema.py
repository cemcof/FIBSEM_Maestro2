# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Classifying the fields of a settings class into `FieldInfo`s; the entry point of the schema package."""

import types
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import NoneType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from fibsem_maestro.settings.dynamic_default import DynamicDefault
from fibsem_maestro.settings.form_utils import NestedUnion
from fibsem_maestro.ui.schema._annotated import split_annotated
from fibsem_maestro.ui.schema._extraction import (
    get_raw_fields,
    unwrap_optional,
)
from fibsem_maestro.ui.schema._predicates import (
    is_float_tuple,
    is_model,
)
from fibsem_maestro.ui.schema._resolve import (
    pydantic_discriminator,
    resolve_default,
    resolve_description,
)
from fibsem_maestro.ui.schema._union_helpers import (
    get_discriminator_key,
    make_union,
)
from fibsem_maestro.ui.schema.constraints import extract_bounds
from fibsem_maestro.ui.schema.field_info import FieldInfo
from fibsem_maestro.ui.schema.field_type import (
    BoolType,
    DataclassType,
    DiscriminatedUnionType,
    EnumType,
    FieldType,
    FloatTupleType,
    FloatType,
    IntType,
    ListType,
    LiteralType,
    StrType,
    UnionVariant,
    UnknownType,
)
from fibsem_maestro.ui.schema.labels import field_name_to_label

if TYPE_CHECKING:
    from collections.abc import Sequence


def classify_type(t: Any, discriminator: str | None = None) -> FieldType:
    """
    Classify an already-unwrapped, non-`Optional` hint into a descriptor.

    The caller must have stripped `Annotated[]` and `Optional[]` first.
    Numeric bounds and the Pydantic single-variant discriminator are resolved
    by `get_field_infos`, not here, since both require field metadata.

    Args:
        t: A bare type hint.
        discriminator: The discriminator key declared by the field's
                    `Field(discriminator=...)`, if any. Takes precedence over
                    inference when `t` is a union.

    Returns:
        The matching `FieldType` descriptor; `UnknownType` as a fallback.
    """
    # bool before int, since bool is a subclass of int
    if t is bool:
        return BoolType()
    if t is int:
        return IntType()
    if t is float:
        return FloatType()
    if is_float_tuple(t):
        return FloatTupleType(length=len(get_args(t)))

    # filesystem paths are edited as strings
    if t in (str, Path, Path | str):
        return StrType()

    if get_origin(t) is Literal:
        return LiteralType(choices=get_args(t))

    if isinstance(t, type) and issubclass(t, Enum):
        return EnumType(enum_type=t)

    if is_model(t):
        return DataclassType(model=t)

    origin = get_origin(t)

    # union (X | Y | ...): a tagged union only if every arm is a model and they
    # share a Literal discriminator field
    if origin is Union or origin is types.UnionType:
        args = tuple(a for a in get_args(t) if a is not NoneType)
        if args:
            union = _classify_union(args, discriminator)
            if union is not None:
                return union

    if origin is list:
        args = get_args(t)
        # argless list has no editable element type -> text-area fallback
        if args:
            return ListType(item=classify_type(args[0]))
        return UnknownType(hint=t)

    return UnknownType(hint=t)


def get_field_infos(cls: type) -> list[FieldInfo]:
    """
    Return a `FieldInfo` for every field of a dataclass or Pydantic model.

    Args:
        cls: A dataclass class or Pydantic ``BaseModel`` subclass.

    Returns:
        One ``FieldInfo`` per field, in declaration order.
    """
    result: list[FieldInfo] = []

    for rf in get_raw_fields(cls):
        outer = split_annotated(rf.type_hint)

        # unwrap optional, then strip a possibly-annotated inner type
        optional, unwrapped = unwrap_optional(outer.bare)
        inner = split_annotated(unwrapped)

        # outer annotations take precedence; inner ones fill any gaps
        form_hint = outer.hint or inner.hint
        field_unit = outer.unit or inner.unit
        all_extras = outer.extras + inner.extras + rf.metadata

        result.append(
            FieldInfo(
                name=rf.name,
                label=field_name_to_label(rf.name),
                description=resolve_description(rf, all_extras),
                optional=optional,
                default=resolve_default(rf),
                hint=form_hint,
                unit=field_unit,
                bounds=extract_bounds(all_extras),
                type=_classify_field(inner.bare, all_extras),
                dynamic_default=next(
                    (e for e in all_extras if isinstance(e, DynamicDefault)), None
                ),
            )
        )

    return result


def _classify_field(inner_hint: Any, extras: tuple) -> FieldType:
    """
    Classify the field type, applying the Pydantic single-variant promotion.

    A model field carrying `Field(discriminator=...)` is a one-armed
    discriminated union even though only one variant is visible on the hint.

    The key from `Field(discriminator=...)` takes precedence over inference
    for unions, because a variant set can share more than one single-value
    `Literal` field and inference cannot then tell which is the tag.
    """
    discriminator = pydantic_discriminator(extras)
    field_type = classify_type(inner_hint, discriminator)

    if isinstance(field_type, DataclassType) and discriminator is not None:
        model = (field_type.model,)
        key = get_discriminator_key([field_type.model]) or discriminator
        return make_union(model, key)

    return field_type


def _classify_union(
    args: tuple[Any, ...], discriminator: str | None
) -> DiscriminatedUnionType | None:
    """
    Build a tagged-union descriptor, folding in nested unions.

    Each arg is either a model (a leaf arm) or an `Annotated` alias wrapping a
    union and carrying a `NestedUnion` marker (a nested arm). Anything else
    makes the union unclassifiable, and the field falls back to a text area.

    Args:
        args: The union's arms, with `NoneType` already removed.
        discriminator: The declared outer discriminator key, or None to infer.

    Returns:
        The descriptor, or None if the arms do not form a tagged union.
    """
    leaves: list[type] = []
    nested: list[tuple[NestedUnion, DiscriminatedUnionType, tuple[type, ...]]] = []

    for arg in args:
        if is_model(arg):
            leaves.append(arg)
            continue

        parts = _nested_union(arg)
        if parts is None:
            return None
        nested.append(parts)

    if not nested:
        # unchanged path: plain union of models
        key = discriminator or get_discriminator_key(leaves)
        return make_union(tuple(leaves), key) if key else None

    inner_classes = [cls for _, _, classes in nested for cls in classes]
    key = discriminator or get_discriminator_key(leaves + inner_classes)
    if key is None:
        return None

    variants: list[UnionVariant] = []
    for cls in leaves:
        value = _literal_value(cls, key)

        if value is None:
            return None

        variants.append(UnionVariant(discriminator_value=value, variant_type=cls))

    for marker, inner, classes in nested:
        values = {_literal_value(cls, key) for cls in classes}

        if len(values) != 1 or None in values:
            raise ValueError(
                f"nested union {marker.label!r} must assign one shared "
                f"{key!r} value to every variant, got {sorted(map(str, values))}"
            )
        base = _common_base(classes)

        if base is None:
            raise ValueError(f"nested union {marker.label!r} needs a common base class")

        variants.append(
            UnionVariant(
                discriminator_value=values.pop(),  # ty:ignore[invalid-argument-type]
                variant_type=base,
                label=marker.label,
                nested=inner,
            )
        )

    return DiscriminatedUnionType(discriminator_key=key, variants=tuple(variants))


def _nested_union(
    arg: Any,
) -> tuple[NestedUnion, DiscriminatedUnionType, tuple[type, ...]] | None:
    """
    Interpret a union arm as a nested union, if it is one.

    Returns:
        `(marker, inner descriptor, inner variant classes)`, or None if `arg`
        is not an `Annotated` union carrying a `NestedUnion` marker.
    """
    split = split_annotated(arg)
    marker = next((e for e in split.extras if isinstance(e, NestedUnion)), None)
    if marker is None:
        return None

    if get_origin(split.bare) not in (Union, types.UnionType):
        return None
    classes = tuple(a for a in get_args(split.bare) if a is not NoneType)
    if not classes or not all(is_model(c) for c in classes):
        return None

    inner_key = pydantic_discriminator(split.extras) or get_discriminator_key(
        list(classes)
    )
    if inner_key is None:
        return None

    inner = make_union(classes, inner_key)
    return marker, replace(inner, follows=marker.follows), classes


def _literal_value(cls: type, key: str) -> str | None:
    """Return the single `Literal` value `cls` assigns to field `key`."""
    for rf in get_raw_fields(cls):
        if rf.name != key:
            continue
        bare = split_annotated(rf.type_hint).bare
        if get_origin(bare) is Literal and len(values := get_args(bare)) == 1:
            return values[0]
    return None


def _common_base(classes: Sequence[type]) -> type | None:
    """Return the nearest model base shared by all `classes`, excluding them."""
    first, *rest = classes
    for base in first.__mro__:
        if base in classes or not is_model(base):
            continue
        if all(issubclass(c, base) for c in rest):
            return base
    return None

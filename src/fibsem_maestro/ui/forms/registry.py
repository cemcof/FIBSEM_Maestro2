# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Which editor edits which field.

A field with a `FormHint` gets the editor registered for its hint's widget
type; any other field gets the editor registered for the kind of its
`FieldType`. A factory returns the final editor: it applies the optional
wrapping and connects the request's `on_change` itself, using `optional_wrap`
and `connect_leaf`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from fibsem_maestro.core.registry import RegistryError
from fibsem_maestro.settings.form_utils import WidgetType
from fibsem_maestro.ui.forms.editor import FieldEditor, OnChange, noop
from fibsem_maestro.ui.forms.editors.action_multi_select import (
    ActionMultiSelectWidget,
)
from fibsem_maestro.ui.forms.editors.action_select import ActionSelectWidget
from fibsem_maestro.ui.forms.editors.bool import BoolWidget
from fibsem_maestro.ui.forms.editors.detail_band import DetailBandWidget
from fibsem_maestro.ui.forms.editors.dynamic_default import DynamicDefaultWidget
from fibsem_maestro.ui.forms.editors.enum import EnumWidget
from fibsem_maestro.ui.forms.editors.float import FloatWidget
from fibsem_maestro.ui.forms.editors.float_tuple import FloatTupleWidget
from fibsem_maestro.ui.forms.editors.group import GroupWrapper
from fibsem_maestro.ui.forms.editors.int import IntWidget
from fibsem_maestro.ui.forms.editors.list import ListWidget
from fibsem_maestro.ui.forms.editors.multi_select import MultiSelectWidget
from fibsem_maestro.ui.forms.editors.optional import OptionalWidget
from fibsem_maestro.ui.forms.editors.range_pair import RangePairWidget
from fibsem_maestro.ui.forms.editors.string import StringWidget
from fibsem_maestro.ui.forms.editors.text_area import TextAreaWidget
from fibsem_maestro.ui.forms.scope import resolve_path
from fibsem_maestro.ui.schema.choices import choices_for
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
    UnknownType,
    is_scalar,
)

if TYPE_CHECKING:
    from fibsem_maestro.settings.dynamic_default import DynamicDefault
    from fibsem_maestro.ui.forms.context import FormContext
    from fibsem_maestro.ui.forms.editors.object import ObjectWidget
    from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
    from fibsem_maestro.ui.forms.scope import BuildScope, ObjectPath
    from fibsem_maestro.ui.schema.constraints import NumericBounds
    from fibsem_maestro.ui.schema.field_info import FieldInfo


class SubBuilder(Protocol):
    """The recursion the form builder offers to editor factories."""

    def build_object(
        self,
        cls: type,
        scope: BuildScope,
        field_infos: list[FieldInfo] | None = None,
        on_change: OnChange = noop,
        live_path: ObjectPath | None = None,
    ) -> ObjectWidget[Any]:
        """
        Build the editor of a nested object.

        Args:
            cls: The object's dataclass or model class.
            scope: The scope of the field holding the object.
            field_infos: The fields to show, or None for all of `cls`'s fields.
            on_change: Threaded to every field of a value-only object; ignored
                for a live one, whose fields get their own committers.
            live_path: Path of the live object to bind to, or None to build
                value-only.

        Returns:
            The object editor.
        """
        ...

    def build_union(
        self,
        union_type: DiscriminatedUnionType,
        on_change: OnChange,
        scope: BuildScope,
        exclude: frozenset[str] = frozenset(),
    ) -> DiscriminatedUnionWidget:
        """
        Build the editor of a discriminated union, nested arms included.

        Args:
            union_type: The union's descriptor.
            on_change: Threaded to every field of every variant page.
            scope: The scope of the union field.
            exclude: Discriminator names of enclosing unions, hidden from every page.

        Returns:
            The union editor, not connected to `on_change` itself.
        """
        ...

    def item_factory(
        self, item: FieldType, on_change: OnChange, scope: BuildScope
    ) -> Callable[[], FieldEditor[Any]]:
        """
        Return a factory building the editor of one list element.

        Args:
            item: The element type.
            on_change: Connected to every element editor.
            scope: The (unaddressed) scope of the elements.

        Returns:
            The factory.
        """
        ...


@dataclass(frozen=True)
class EditorRequest:
    """
    Everything an editor factory needs to build the editor of one field.

    Attributes:
        field: The field.
        value: The live value if there is one, else the field's declared
            default, else None. The value the editor shows initially.
        live_value: The live value; None in a value-only subtree (and for a
            live field holding None). Decides whether an optional checkbox
            starts checked and whether a nested object is built live.
        on_change: The field's committer in a live object, otherwise the
            callback inherited from the enclosing field.
        scope: The scope of the field.
        ctx: The environment the form is built in.
        build: Recursion into nested objects, unions and list elements.
    """

    field: FieldInfo
    value: Any
    live_value: Any
    on_change: OnChange
    scope: BuildScope
    ctx: FormContext
    build: SubBuilder

    def live_path(self) -> ObjectPath | None:
        """
        Return the path of the live object this field holds, if it holds one.

        Returns:
            The path from the form root, or None if the field has no live
            value or is not a field of a live object.
        """
        if self.live_value is None or self.scope.owner is None:
            return None
        return (*self.scope.owner, self.field.name)


EditorFactory = Callable[[EditorRequest], FieldEditor[Any]]
"""Builds the final, connected editor of a field."""


class EditorRegistry:
    """Maps hint widget types and field-type kinds to editor factories."""

    def __init__(self) -> None:
        self._hints: dict[WidgetType, EditorFactory] = {}
        self._types: dict[type[FieldType], EditorFactory] = {}

    def register_hint(self, widget: WidgetType, factory: EditorFactory) -> None:
        """
        Register the factory for fields hinted with `widget`.

        Args:
            widget: The hint's widget type.
            factory: The factory.

        Raises:
            RegistryError: If a factory is already registered for `widget`.
        """
        if widget in self._hints:
            raise RegistryError(f"An editor for hint {widget} is already registered.")
        self._hints[widget] = factory

    def register_type(self, kind: type[FieldType], factory: EditorFactory) -> None:
        """
        Register the factory for unhinted fields whose type descriptor is a `kind`.

        Args:
            kind: The `FieldType` subclass.
            factory: The factory.

        Raises:
            RegistryError: If a factory is already registered for `kind`.
        """
        if kind in self._types:
            raise RegistryError(f"An editor for {kind.__name__} is already registered.")
        self._types[kind] = factory

    def create(self, request: EditorRequest) -> FieldEditor[Any]:
        """
        Build the editor of a field.

        A hinted field uses its hint's factory, otherwise its field type's
        factory, otherwise the YAML text-area fallback. A field with a dynamic
        default gets that editor, built as a required field, inside a
        `DynamicDefaultWidget`.

        Args:
            request: The field and its build environment.

        Returns:
            The final editor, optional wrapping applied and `on_change` connected.
        """
        fi = request.field
        if fi.dynamic_default is not None:
            return self._create_dynamic(request, fi.dynamic_default)
        if fi.hint is not None:
            return self._hints.get(fi.hint.widget, _hinted_text_area)(request)
        return self._types.get(type(fi.type), _text_area)(request)

    def _create_dynamic(
        self, request: EditorRequest, declared: DynamicDefault
    ) -> FieldEditor[Any]:
        """
        Build the editor of a field with a dynamic default.

        The default is computed from the live object holding the field, so it
        cannot be computed in a value-only subtree (or a detached form).

        Args:
            request: The field and its build environment.
            declared: The field's dynamic default.

        Returns:
            The editor, connected and registered for `Form.refresh_defaults`.
        """
        fi = request.field
        # None means "follow the default", never "no value": no optional checkbox
        inner = self.create(
            replace(
                request,
                field=replace(fi, optional=False, dynamic_default=None),
                value=None,
                on_change=noop,
            )
        )
        scope, ctx = request.scope, request.ctx

        def compute() -> Any:
            owner = (
                resolve_path(scope.root, scope.owner)
                if scope.owner is not None
                else None
            )
            return None if owner is None else declared.compute(owner, ctx.settings_of)

        editor = DynamicDefaultWidget(
            inner, compute, declared.unavailable, declared.source
        )
        editor.set_value(request.value)
        editor.on_change(request.on_change)
        scope.context.dynamic_defaults.append(editor)
        return editor


def optional_wrap(
    editor: FieldEditor[Any], request: EditorRequest, inline: bool
) -> FieldEditor[Any]:
    """
    Wrap an editor in an optional checkbox if the field is optional, and connect it.

    The checkbox starts checked iff the field has a live value. `on_change`
    is connected to the editor and, if wrapped, to the checkbox.

    Args:
        editor: The value editor.
        request: The field's request.
        inline: Lay the checkbox out beside the editor rather than above it.

    Returns:
        The wrapper, or `editor` itself for a required field.
    """
    result = editor
    if request.field.optional:
        result = OptionalWidget(
            editor, inline=inline, enabled_by_default=request.live_value is not None
        )

    editor.on_change(request.on_change)
    if result is not editor:
        result.on_change(request.on_change)
    return result


def connect_leaf(editor: FieldEditor[Any], request: EditorRequest) -> FieldEditor[Any]:
    """
    Connect `on_change` to an editor that is never wrapped.

    Args:
        editor: The editor.
        request: The field's request.

    Returns:
        `editor`.
    """
    editor.on_change(request.on_change)
    return editor


def int_bounds(bounds: NumericBounds | None) -> tuple[int | None, int | None]:
    """
    Convert numeric bounds to inclusive integer limits for a spin box.

    Args:
        bounds: The field's bounds, or None.

    Returns:
        The minimum and maximum, each None if unbounded.
    """
    if bounds is None:
        return None, None

    minimum = (
        None
        if bounds.minimum is None
        else int(bounds.minimum.value) + (1 if bounds.minimum.exclusive else 0)
    )

    maximum = (
        None
        if bounds.maximum is None
        else int(bounds.maximum.value) - (1 if bounds.maximum.exclusive else 0)
    )

    return minimum, maximum


def float_bounds(bounds: NumericBounds | None) -> tuple[float | None, float | None]:
    """
    Return raw float limits; exclusivity is not represented for floats.

    Args:
        bounds: The field's bounds, or None.

    Returns:
        The minimum and maximum, each None if unbounded.
    """
    if bounds is None:
        return None, None
    minimum = None if bounds.minimum is None else bounds.minimum.value
    maximum = None if bounds.maximum is None else bounds.maximum.value
    return minimum, maximum


def fallback_type(field_type: FieldType) -> Any:
    """
    Return the type a YAML text-area fallback loads values into.

    Args:
        field_type: The field's type descriptor.

    Returns:
        The original hint of an unknown type, the model of a nested object,
        otherwise `Any`.
    """
    if isinstance(field_type, UnknownType):
        return field_type.hint
    if isinstance(field_type, DataclassType):
        return field_type.model
    return Any


def _suffix(fi: FieldInfo) -> str | None:
    """The unit suffix shown after the field's editor, if any."""
    return fi.unit.suffix if fi.unit else None


def _finish_leaf(editor: FieldEditor[Any], request: EditorRequest) -> FieldEditor[Any]:
    """Wrap a leaf editor (inline for scalar fields) and connect it."""
    return optional_wrap(editor, request, inline=is_scalar(request.field.type))


def _choice_dropdown(request: EditorRequest) -> FieldEditor[Any]:
    """DROPDOWN, PROPERTY_SELECTOR and PATTERN_TYPE_SELECTOR."""
    fi = request.field
    editor = EnumWidget(
        choices_for(fi, request.ctx.choice_context()),
        default=request.value,
        optional=fi.optional,
    )
    return optional_wrap(editor, request, inline=True)


def _multi_select(request: EditorRequest) -> FieldEditor[Any]:
    """MULTI_SELECT and MULTI_PROPERTY_SELECTOR."""
    editor = MultiSelectWidget(
        choices_for(request.field, request.ctx.choice_context()),
        default=request.value,
    )
    return optional_wrap(editor, request, inline=True)


def _range_pair(request: EditorRequest) -> FieldEditor[Any]:
    """RANGE_PAIR."""
    fi = request.field
    minimum, maximum = float_bounds(fi.bounds)
    editor = RangePairWidget(
        default=request.value, minimum=minimum, maximum=maximum, suffix=_suffix(fi)
    )
    return optional_wrap(editor, request, inline=True)


def _detail_band(request: EditorRequest) -> FieldEditor[Any]:
    """DETAIL_BAND."""
    fi = request.field
    _, maximum = float_bounds(fi.bounds)
    editor = DetailBandWidget(
        default=request.value,
        minimum=0.0,  # manual override
        maximum=maximum,
        suffix=_suffix(fi),
    )
    return optional_wrap(editor, request, inline=True)


def _action_select(request: EditorRequest) -> FieldEditor[Any]:
    """
    ACTION_SELECTOR.

    Raises:
        ValueError: If the form context has no action catalog.
    """
    fi = request.field
    if request.ctx.actions is None:
        raise ValueError(
            f"field {fi.name!r} needs an action catalog, but the form context has none"
        )
    hint = fi.hint
    editor = ActionSelectWidget(
        actions=request.ctx.actions,
        type_filter=hint.action_type_filter if hint is not None else [],
        type_exclude=hint.action_type_exclude if hint is not None else (),
        default=request.value,
        optional=fi.optional,
    )
    request.scope.context.disposers.append(editor.detach)
    return optional_wrap(editor, request, inline=True)


def _action_multi_select(request: EditorRequest) -> FieldEditor[Any]:
    """
    MULTI_ACTION_SELECTOR.

    Raises:
        ValueError: If the form context has no action catalog.
    """
    fi = request.field
    if request.ctx.actions is None:
        raise ValueError(
            f"field {fi.name!r} needs an action catalog, but the form context has none"
        )
    hint = fi.hint
    editor = ActionMultiSelectWidget(
        actions=request.ctx.actions,
        type_filter=hint.action_type_filter if hint is not None else [],
        type_exclude=hint.action_type_exclude if hint is not None else (),
        default=request.value,
    )
    request.scope.context.disposers.append(editor.detach)
    return optional_wrap(editor, request, inline=True)


def _hinted_text_area(request: EditorRequest) -> FieldEditor[Any]:
    """STRING and any hint without a registered editor."""
    editor = TextAreaWidget(fallback_type(request.field.type), default=request.value)
    return optional_wrap(editor, request, inline=True)


def _area_select(request: EditorRequest) -> FieldEditor[Any]:
    """AREA_SELECT."""
    # imported here: the area-selector factory itself uses this module's helpers
    from fibsem_maestro.ui.forms.editors.area_select.factory import (
        create_area_select,
    )

    return create_area_select(request)


def _bool(request: EditorRequest) -> FieldEditor[Any]:
    """BoolType."""
    default = request.value
    editor = BoolWidget(default=bool(default) if default is not None else False)
    return _finish_leaf(editor, request)


def _int(request: EditorRequest) -> FieldEditor[Any]:
    """IntType."""
    fi, default = request.field, request.value
    minimum, maximum = int_bounds(fi.bounds)
    editor = IntWidget(
        default=int(default) if default is not None else 0,
        minimum=minimum,
        maximum=maximum,
        suffix=_suffix(fi),
    )
    return _finish_leaf(editor, request)


def _float(request: EditorRequest) -> FieldEditor[Any]:
    """FloatType."""
    fi, default = request.field, request.value
    minimum, maximum = float_bounds(fi.bounds)
    editor = FloatWidget(
        default=float(default) if default is not None else 0.0,
        minimum=minimum,
        maximum=maximum,
        suffix=_suffix(fi),
    )
    return _finish_leaf(editor, request)


def _str(request: EditorRequest) -> FieldEditor[Any]:
    """StrType."""
    default = request.value
    editor = StringWidget(
        default=str(default) if default is not None else "",
        suffix=_suffix(request.field),
    )
    return _finish_leaf(editor, request)


def _choice(request: EditorRequest) -> FieldEditor[Any]:
    """EnumType and LiteralType: optionality is a "(none)" entry, never a checkbox."""
    fi = request.field
    editor = EnumWidget(
        choices_for(fi, request.ctx.choice_context()),
        default=request.value,
        optional=fi.optional,
    )
    return connect_leaf(editor, request)


def _dataclass(request: EditorRequest) -> FieldEditor[Any]:
    """DataclassType: live if the field holds a live object, otherwise value-only."""
    fi = request.field
    if not isinstance(fi.type, DataclassType):
        raise TypeError(f"field {fi.name!r} is not a nested object")

    # value-only: the inherited on_change is threaded to the children; live:
    # every child field gets its own committer on the nested object
    inner = request.build.build_object(
        fi.type.model,
        request.scope,
        on_change=request.on_change,
        live_path=request.live_path(),
    )
    if fi.optional:
        # toggling None <-> instance is a change to the enclosing field
        group = OptionalWidget(
            GroupWrapper(inner),
            inline=False,
            enabled_by_default=request.live_value is not None,
        )
        group.on_change(request.on_change)
        return group
    return GroupWrapper(inner)


def _union(request: EditorRequest) -> FieldEditor[Any]:
    """DiscriminatedUnionType: always value-only; variant pages keep their values."""
    fi = request.field
    if not isinstance(fi.type, DiscriminatedUnionType):
        raise TypeError(f"field {fi.name!r} is not a discriminated union")

    union = request.build.build_union(fi.type, request.on_change, request.scope)
    # switching variants is a change
    union.on_change(request.on_change)
    if request.live_value is not None:
        union.set_value(request.live_value)

    group = GroupWrapper(union)
    if fi.optional:
        result = OptionalWidget(
            group, inline=False, enabled_by_default=request.live_value is not None
        )
        result.on_change(request.on_change)
        return result
    return group


def _list(request: EditorRequest) -> FieldEditor[Any]:
    """ListType: elements are value-only and unaddressed."""
    fi, default = request.field, request.value
    if not isinstance(fi.type, ListType):
        raise TypeError(f"field {fi.name!r} is not a list")

    editor = ListWidget(
        item_factory=request.build.item_factory(
            fi.type.item, request.on_change, request.scope.unaddressed()
        ),
        default=list(default) if default is not None else [],
    )
    return _finish_leaf(editor, request)


def _float_tuple(request: EditorRequest) -> FieldEditor[Any]:
    """FloatTupleType."""
    fi, default = request.field, request.value
    if not isinstance(fi.type, FloatTupleType):
        raise TypeError(f"field {fi.name!r} is not a tuple of floats")

    editor = FloatTupleWidget(
        length=fi.type.length,
        default=tuple(default) if default is not None else None,
    )
    return _finish_leaf(editor, request)


def _text_area(request: EditorRequest) -> FieldEditor[Any]:
    """UnknownType, and any field type without a registered editor."""
    editor = TextAreaWidget(fallback_type(request.field.type), default=request.value)
    return _finish_leaf(editor, request)


def default_registry() -> EditorRegistry:
    """
    Return a registry holding the editors of every hint and field type.

    Returns:
        A new registry; callers may register more editors on it.
    """
    registry = EditorRegistry()

    registry.register_hint(WidgetType.DROPDOWN, _choice_dropdown)
    registry.register_hint(WidgetType.PROPERTY_SELECTOR, _choice_dropdown)
    registry.register_hint(WidgetType.PATTERN_TYPE_SELECTOR, _choice_dropdown)
    registry.register_hint(WidgetType.MULTI_SELECT, _multi_select)
    registry.register_hint(WidgetType.MULTI_PROPERTY_SELECTOR, _multi_select)
    registry.register_hint(WidgetType.AREA_SELECT, _area_select)
    registry.register_hint(WidgetType.RANGE_PAIR, _range_pair)
    registry.register_hint(WidgetType.DETAIL_BAND, _detail_band)
    registry.register_hint(WidgetType.ACTION_SELECTOR, _action_select)
    registry.register_hint(WidgetType.MULTI_ACTION_SELECTOR, _action_multi_select)
    registry.register_hint(WidgetType.STRING, _hinted_text_area)

    registry.register_type(BoolType, _bool)
    registry.register_type(IntType, _int)
    registry.register_type(FloatType, _float)
    registry.register_type(StrType, _str)
    registry.register_type(EnumType, _choice)
    registry.register_type(LiteralType, _choice)
    registry.register_type(DataclassType, _dataclass)
    registry.register_type(DiscriminatedUnionType, _union)
    registry.register_type(ListType, _list)
    registry.register_type(FloatTupleType, _float_tuple)
    registry.register_type(UnknownType, _text_area)

    return registry

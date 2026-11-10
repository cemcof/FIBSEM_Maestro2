# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Builds settings forms from settings classes.

A *live* form edits an object in place: every field of a live object commits
on its own (`FieldCommitter`). A *detached* form edits no object and is read
with `DetachedForm.value()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal

from fibsem_maestro.ui.forms.commit import FieldCommitter
from fibsem_maestro.ui.forms.editor import FieldEditor, OnChange, noop
from fibsem_maestro.ui.forms.editors.bool import BoolWidget
from fibsem_maestro.ui.forms.editors.enum import EnumWidget
from fibsem_maestro.ui.forms.editors.float import FloatWidget
from fibsem_maestro.ui.forms.editors.group import GroupWrapper
from fibsem_maestro.ui.forms.editors.int import IntWidget
from fibsem_maestro.ui.forms.editors.object import ObjectWidget
from fibsem_maestro.ui.forms.editors.string import StringWidget
from fibsem_maestro.ui.forms.editors.text_area import TextAreaWidget
from fibsem_maestro.ui.forms.editors.union import DiscriminatedUnionWidget
from fibsem_maestro.ui.forms.registry import (
    EditorRegistry,
    EditorRequest,
    fallback_type,
)
from fibsem_maestro.ui.forms.scope import (
    BuildContext,
    BuildScope,
    ObjectPath,
    noop_commit,
    resolve_path,
)
from fibsem_maestro.ui.forms.wiring import (
    FollowerBinding,
    collect_overlays,
    wire_followers,
    wire_overlays,
)
from fibsem_maestro.ui.schema.field_type import (
    BoolType,
    DataclassType,
    DiscriminatedUnionType,
    EnumType,
    FieldType,
    FloatType,
    IntType,
    StrType,
)
from fibsem_maestro.ui.schema.schema import get_field_infos

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fibsem_maestro.ui.forms.context import FormContext
    from fibsem_maestro.ui.schema.field_info import FieldInfo


class Form(QObject):
    """
    A form editing a live object in place.

    Emits `validity_changed` whenever the form starts or stops showing an
    error, whatever its source (a value the committer rejects, an area
    selector's minimum-area error, any other editor-level error).

    Args:
        widget: The form's root object editor.
        context: The build the form came from.
        parent: The parent QObject, if any.
    """

    validity_changed = pyqtSignal(bool)
    """Emitted with True when some field starts showing an error, False when none does."""

    def __init__(
        self,
        widget: ObjectWidget[Any],
        context: BuildContext,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._widget = widget
        self._context = context
        self._has_errors = self.has_errors()
        context.on_issues_changed(self._on_issues_changed)

    @property
    def widget(self) -> ObjectWidget[Any]:
        """The form's root object editor (a `QWidget`), to be placed in a layout."""
        return self._widget

    def has_errors(self) -> bool:
        """
        Check whether any field of the form shows an error.

        An erroneous value is not written to the settings, so the form and the
        settings disagree until it is fixed.

        Returns:
            True if a field of the root object, or of any nested object editor
            (unselected union pages and hidden groups included), reports an
            issue of severity ERROR.
        """
        objects = [self._widget, *self._widget.findChildren(ObjectWidget)]
        return any(obj.has_errors() for obj in objects if isinstance(obj, ObjectWidget))

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing of every field.

        Args:
            read_only: If True, no field can be edited.
        """
        self._widget.set_read_only(read_only)

    def set_value(self, value: Any) -> None:
        """
        Show the fields of `value` in the form.

        Whether a shown value is also committed to the live object depends on
        the field's editor: most editors report a value set this way like an
        edit, so it is committed; the area selector does not (it commits only
        a correction it had to make to fit areas to the frame). Use this to
        refresh what the form shows; to change the live object, assign to it.

        Args:
            value: An instance of the form's class, or a dict of field values.
                Fields absent from it are left unchanged.
        """
        self._widget.set_value(value)

    def refresh_overlays(self) -> None:
        """
        Redraw the area-selector overlays from their current sources.

        For overlays drawn from the action's internal state, which changes
        without any editor noticing (e.g. when the action ran or its state
        was reset). Does nothing once the form is disposed.
        """
        if self._context.disposed:
            return
        for binding in self._context.overlays:
            if binding.wired:
                binding.push()

    def dispose(self) -> None:
        """
        Make the form inert and remove everything it registered outside itself. Idempotent.

        Afterwards no edit or catalog change reaches the live object, and
        `validity_changed` is no longer emitted, even while the widgets wait
        for deletion.
        """
        self._context.dispose()

    def _on_issues_changed(self) -> None:
        """Emit `validity_changed` if the form started or stopped showing an error."""
        if self._context.disposed:
            return
        has_errors = self.has_errors()
        if has_errors != self._has_errors:
            self._has_errors = has_errors
            self.validity_changed.emit(has_errors)


class DetachedForm(QObject):
    """
    A form bound to no object; its value is read with `value()`.

    Args:
        widget: The form's root object editor.
        context: The build the form came from.
        parent: The parent QObject, if any.
    """

    changed = pyqtSignal()
    """Emitted whenever any editor of the form changed."""

    def __init__(
        self,
        widget: ObjectWidget[Any],
        context: BuildContext,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._widget = widget
        context.on_changed(self._on_changed)

    @property
    def widget(self) -> ObjectWidget[Any]:
        """The form's root object editor (a `QWidget`), to be placed in a layout."""
        return self._widget

    def value(self) -> Any:
        """
        Build an instance of the form's class from the values shown.

        Returns:
            The instance; fields not in the form take their defaults.

        Raises:
            FieldValidationError: If the values shown do not form a valid instance.
        """
        return self._widget.get_value()

    def set_value(self, value: Any) -> None:
        """
        Show the fields of `value`.

        Args:
            value: An instance of the form's class, or a dict of field values.
                Fields absent from it are left unchanged.
        """
        self._widget.set_value(value)

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable editing of every field.

        Args:
            read_only: If True, no field can be edited.
        """
        self._widget.set_read_only(read_only)

    def _on_changed(self) -> None:
        """Forward an editor change."""
        self.changed.emit()


class FormBuilder:
    """
    Builds forms from settings classes.

    Stateless: all per-build state lives in a `BuildContext` created per call,
    so one builder serves every form of the application.

    Args:
        registry: Decides which editor edits which field.
    """

    def __init__(self, registry: EditorRegistry) -> None:
        self._recursion = _Recursion(registry)

    def build_live(
        self,
        obj: Any,
        ctx: FormContext,
        fields: Sequence[str] | None = None,
        on_commit: Callable[[str], None] = noop_commit,
    ) -> Form:
        """
        Build a form editing `obj` in place.

        Every field of `obj`, and of every nested object holding a value, is
        written back on its own as soon as it is edited and valid.

        Args:
            obj: The live object: a pydantic model, or a pydantic or plain dataclass.
            ctx: The environment the form is built in.
            fields: Names of the top-level fields to show, or None for all;
                declaration order is kept.
            on_commit: Called with the dotted path (from `obj`) of every field
                written, after the write.

        Returns:
            The form.
        """
        context = BuildContext(form=ctx, on_commit=on_commit)
        widget = self._recursion.build_root(type(obj), obj, fields, context, noop)
        return Form(widget, context)

    def build_detached(
        self, cls: type, ctx: FormContext, fields: Sequence[str] | None = None
    ) -> DetachedForm:
        """
        Build a form for `cls` bound to no object.

        Fields show their declared defaults; nothing is written anywhere.

        Args:
            cls: The dataclass or model class.
            ctx: The environment the form is built in.
            fields: Names of the fields to show, or None for all; declaration
                order is kept.

        Returns:
            The form.
        """
        context = BuildContext(form=ctx)
        widget = self._recursion.build_root(
            cls, None, fields, context, context.notify_changed
        )
        return DetachedForm(widget, context)


class _Recursion:
    """
    The build algorithm: objects, unions and list elements, recursively.

    Implements `registry.SubBuilder` for the editor factories.

    Args:
        registry: Decides which editor edits which field.
    """

    def __init__(self, registry: EditorRegistry) -> None:
        self._registry = registry

    def build_root(
        self,
        cls: type,
        obj: Any | None,
        fields: Sequence[str] | None,
        context: BuildContext,
        on_change: OnChange,
    ) -> ObjectWidget[Any]:
        """
        Build the root object, then wire the cross-field bindings and check every committer.

        Args:
            cls: The class at the form root.
            obj: The live root object, or None for a detached form.
            fields: Names of the fields to show, or None for all.
            context: The build's state.
            on_change: Threaded to every field of a detached form.

        Returns:
            The root object editor.
        """
        infos = get_field_infos(cls)
        if fields is not None:
            infos = [fi for fi in infos if fi.name in fields]

        scope = BuildScope(context=context, root=obj)
        widget = self.build_object(
            cls,
            scope,
            field_infos=infos,
            on_change=on_change,
            live_path=() if obj is not None else None,
        )
        wire_followers(context)
        wire_overlays(context)
        # validate the values as shown, after followers have picked their variants
        for committer in context.committers:
            committer.check()
        return widget

    def build_object(
        self,
        cls: type,
        scope: BuildScope,
        field_infos: list[FieldInfo] | None = None,
        on_change: OnChange = noop,
        live_path: ObjectPath | None = None,
    ) -> ObjectWidget[Any]:
        """
        Build an object editor for `cls`.

        Live (`live_path` given): each field gets its own committer, assigning
        into the object at `live_path`. Value-only: `on_change` is threaded to
        every field, so edits reach the enclosing field's committer.

        Every addressable field's editor is registered in the build context,
        so bindings declared elsewhere in the form can reference it by path.

        Args:
            cls: The dataclass or model class.
            scope: The scope of the field holding the object (the root scope
                for the root object).
            field_infos: The fields to show, or None for all of `cls`'s fields.
            on_change: The callback threaded down in value-only mode.
            live_path: Path from the root to the live object, or None.

        Returns:
            The object editor.
        """
        context = scope.context
        if live_path is not None:
            scope = scope.enter_live(live_path)
            owner = resolve_path(scope.root, live_path)
        else:
            scope = scope.enter_value_only()
            owner = None

        obj = ObjectWidget(cls=cls)
        infos = field_infos if field_infos is not None else get_field_infos(cls)

        editors: dict[str, FieldEditor[Any]] = {}
        for fi in infos:
            field_scope = scope.child(fi.name)

            if owner is not None and live_path is not None:
                live_value = getattr(owner, fi.name, None)
                committer = FieldCommitter(scope.root, live_path, fi, context)
                editor = self._create(fi, live_value, committer, field_scope)
                committer.bind(editor)
                context.committers.append(committer)
            else:
                editor = self._create(fi, None, on_change, field_scope)

            obj.add_field(
                fi.name,
                fi.label,
                editor,
                fi.description,
                # only the action's own top-level fields define its state
                resets_state=scope.path == () and fi.name in context.form.state_fields,
            )
            editor.on_issues_changed(context.notify_issues_changed)
            editors[fi.name] = editor
            if (path := field_scope.dotted) is not None:
                context.editors[path] = editor

        collect_overlays(infos, editors, scope)
        return obj

    def build_union(
        self,
        union_type: DiscriminatedUnionType,
        on_change: OnChange,
        scope: BuildScope,
        exclude: frozenset[str] = frozenset(),
    ) -> DiscriminatedUnionWidget:
        """
        Build a union editor, recursing into nested union arms.

        A nested arm is built as its own union editor and installed as the
        outer arm's page. If it declares a driving field, a follower binding
        is queued so the selection can be wired once the form exists.

        Args:
            union_type: The union descriptor to build.
            on_change: Threaded into every variant's fields.
            scope: Scope of the union field.
            exclude: Discriminator names of enclosing unions, hidden from every
                variant's page.

        Returns:
            The union editor, with nested arms already built and installed.
        """
        nested: dict[str, FieldEditor[Any]] = {}
        labels: dict[str, str] = {}
        descriptions: dict[str, str] = {}

        for variant in union_type.variants:
            if variant.label is not None:
                labels[variant.discriminator_value] = variant.label
            if description := _variant_description(
                variant.variant_type, union_type.discriminator_key
            ):
                descriptions[variant.discriminator_value] = description

            if variant.nested is None:
                continue

            inner = self.build_union(
                variant.nested,
                on_change,
                scope,
                exclude | {union_type.discriminator_key},
            )

            # switching the inner variant is a change to the same field
            inner.on_change(on_change)
            nested[variant.discriminator_value] = inner

            if variant.nested.follows is None:
                continue

            binding = FollowerBinding(
                union=inner,
                source_path=variant.nested.follows,
                fallback=variant.nested.variants[0].discriminator_value,
                field_name=variant.label or variant.discriminator_value,
                scope=scope,
            )
            # a rebuilt or discarded subtree must stop receiving pushes
            inner.destroyed.connect(binding.deactivate)
            scope.context.followers.append(binding)

        return DiscriminatedUnionWidget(
            variants=[
                (v.discriminator_value, v.variant_type) for v in union_type.variants
            ],
            discriminator_key=union_type.discriminator_key,
            build_object=self._variant_builder(on_change, scope),
            nested=nested,
            labels=labels,
            descriptions=descriptions,
            exclude=exclude,
            show_selector=union_type.follows is None,
        )

    def item_factory(
        self, item: FieldType, on_change: OnChange, scope: BuildScope
    ) -> Callable[[], FieldEditor[Any]]:
        """
        Return a factory building the editor of a single list element.

        Every element's `on_change` is the list field's own, so editing an
        element rebuilds and reassigns the whole list. `scope` is unaddressed:
        elements come and go, so their fields are not registered and can only
        be referenced as siblings.

        Args:
            item: The element type.
            on_change: Connected to every element editor.
            scope: The unaddressed scope of the elements.

        Returns:
            The factory.
        """
        match item:
            case BoolType():
                return lambda: _connected(BoolWidget(), on_change)
            case IntType():
                return lambda: _connected(IntWidget(), on_change)
            case FloatType():
                return lambda: _connected(FloatWidget(), on_change)
            case StrType():
                return lambda: _connected(StringWidget(), on_change)
            case EnumType(enum_type=enum_cls):
                return lambda: _connected(EnumWidget(list(enum_cls)), on_change)
            case DataclassType(model=model):
                return lambda: GroupWrapper(
                    self.build_object(model, scope, on_change=on_change)
                )
            case _:
                return lambda: _connected(
                    TextAreaWidget(fallback_type(item)), on_change
                )

    def _create(
        self, fi: FieldInfo, live_value: Any, on_change: OnChange, scope: BuildScope
    ) -> FieldEditor[Any]:
        """
        Build the editor of one field through the registry.

        Args:
            fi: The field.
            live_value: The field's live value; None in a value-only subtree.
            on_change: The field's committer, or the inherited callback.
            scope: The field's scope.

        Returns:
            The final, connected editor.
        """
        if live_value is not None:
            value = live_value
        elif fi.default is not None:
            value = fi.default.value
        else:
            value = None

        return self._registry.create(
            EditorRequest(
                field=fi,
                value=value,
                live_value=live_value,
                on_change=on_change,
                scope=scope,
                ctx=scope.context.form,
                build=self,
            )
        )

    def _variant_builder(
        self, on_change: OnChange, scope: BuildScope
    ) -> Callable[[type, list[FieldInfo] | None], ObjectWidget[Any]]:
        """
        Return the union editor's page factory, scoped to the union field.

        Args:
            on_change: Threaded into the variant's fields.
            scope: Scope of the union field itself.

        Returns:
            A callable matching `DiscriminatedUnionWidget`'s `build_object`.
        """

        def build(cls: type, field_infos: list[FieldInfo] | None) -> ObjectWidget[Any]:
            return self.build_object(
                cls, scope, field_infos=field_infos, on_change=on_change
            )

        return build


def _connected(editor: FieldEditor[Any], on_change: OnChange) -> FieldEditor[Any]:
    """Connect `on_change` to a list-element editor and return it."""
    editor.on_change(on_change)
    return editor


def _variant_description(variant_type: type, discriminator_key: str) -> str:
    """
    Return the hint for a union variant: the description of its discriminator field.

    The discriminator identifies the variant and is never shown as a field of
    its own, so its `Field(description=...)` describes the variant. For a
    nested arm, `variant_type` is the inner variants' common base class, which
    declares the outer discriminator.

    Args:
        variant_type: The variant's class.
        discriminator_key: The name of the union's discriminator field.

    Returns:
        The description, or `""` if the discriminator has none.
    """
    for fi in get_field_infos(variant_type):
        if fi.name == discriminator_key:
            return fi.description
    return ""

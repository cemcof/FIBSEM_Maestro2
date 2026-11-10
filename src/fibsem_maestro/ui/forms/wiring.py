# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Cross-field wiring: area-selector overlays and nested-union followers.

Both read values from other fields of the same form. They are collected while
the form is built and wired once the whole tree exists, because a source may
live in a branch built later. Field paths are resolved against the declaring
object's siblings first, then against the form's editors by dotted path, from
the innermost enclosing scope out to the root. The walk goes through
`FieldEditor.value_editor()` / `child_editors()` only, so any container editor
implementing those takes part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fibsem_maestro.ui.forms.issues import FieldValidationError
from fibsem_maestro.ui.forms.scope import resolve_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from PyQt6.QtCore import pyqtBoundSignal

    from fibsem_maestro.settings.form_utils import AreaOverlay, OverlaySpec
    from fibsem_maestro.ui.forms.editor import FieldEditor
    from fibsem_maestro.ui.forms.scope import BuildContext, BuildScope
    from fibsem_maestro.ui.schema.field_info import FieldInfo


def _shown_value(editor: FieldEditor[Any]) -> Any:
    """
    The value an editor currently shows, or None if it cannot form one.

    Args:
        editor: The editor to read.

    Returns:
        The editor's value; None while it holds an invalid value.
    """
    try:
        return editor.get_value()
    except FieldValidationError:
        # e.g. a multi-tile criterion whose reduction function is not offered by
        # its dropdown: the field shows its own error, and a decoration or variant
        # depending on it must not break building the whole form
        return None


@runtime_checkable
class OverlayTarget(Protocol):
    """An editor drawing area overlays (the area selector)."""

    destroyed: pyqtBoundSignal
    """The widget's `QObject.destroyed` signal."""

    def set_overlays(
        self, overlays: Sequence[tuple[AreaOverlay, Mapping[str, Any]]]
    ) -> None:
        """
        Replace the overlays drawn.

        Args:
            overlays: Overlay kinds paired with their values, keyed by
                `AreaOverlay.data_fields`.
        """
        ...


class FollowerTarget(Protocol):
    """A union editor whose variant is selected from outside (a nested union)."""

    def select(self, discriminator_value: str) -> bool:
        """
        Select a variant by tag.

        Args:
            discriminator_value: The tag of the variant to show.

        Returns:
            True if a matching variant exists and is now selected.
        """
        ...


@dataclass
class OverlayBinding:
    """
    An area selector's overlays, awaiting or holding a subscription to their sources.

    Attributes:
        area: The editor the decorations are drawn on.
        area_path: Dotted path of the declaring field, or None if unaddressable.
        field_name: The declaring field's name, for log messages.
        specs: The overlays declared on that field.
        scope: Scope of the declaring object; source paths resolve from here.
        siblings: The declaring object's editors by field name. Held live so
            sibling lookup works in subtrees that have no address.
        wired: True once source subscriptions have been made.
        active: False once the area editor is gone; pushes become no-ops.
    """

    area: OverlayTarget
    area_path: str | None
    field_name: str
    specs: tuple[OverlaySpec, ...]
    scope: BuildScope
    siblings: dict[str, FieldEditor[Any]]
    wired: bool = False
    active: bool = True

    def deactivate(self, *_: object) -> None:
        """Stop this binding from touching its area editor."""
        self.active = False

    @property
    def reads_state(self) -> bool:
        """True if an overlay is drawn from the action's internal state."""
        return any(spec.state_sources for spec in self.specs)

    def push(self) -> None:
        """Copy the current source values, and the action's state, into the area editor's overlays."""
        if not self.active:
            return

        state = self._state()
        overlays: list[tuple[AreaOverlay, Mapping[str, Any]]] = []
        for spec in self.specs:
            values: dict[str, Any] = {}
            if self._requirements_met(spec):
                for data_field, path in spec.sources:
                    source = self._resolve_source(path)
                    if source is not None:
                        values[data_field] = _shown_value(source)
                if state is not None:
                    for data_field, path in spec.state_sources:
                        values[data_field] = resolve_path(state, tuple(path.split(".")))
            overlays.append((spec.kind, values))

        self.area.set_overlays(overlays)

    def _state(self) -> object | None:
        """The action's current internal state, or None if the form has none."""
        provider = self.scope.context.form.state
        return provider() if provider is not None and self.reads_state else None

    def sources(self) -> list[FieldEditor[Any]]:
        """
        Resolve the binding's source paths in every union variant, warning about the missing ones.

        Returns:
            The distinct source editors to subscribe to, in declaration order.
        """
        found: dict[int, FieldEditor[Any]] = {}
        for spec in self.specs:
            paths = [path for _, path in spec.sources]
            paths += [path for path, _ in spec.requires]
            for path in paths:
                editors = editors_at(self.siblings, self.scope, path, all_variants=True)
                if not editors:
                    self.scope.context.form.warn(
                        f"Overlay source {path!r} for field "
                        f"{self.field_name!r} was not found in this form; "
                        f"{spec.kind.value} will not be drawn."
                    )
                    continue

                for editor in editors:
                    found.setdefault(id(editor), editor)

        return list(found.values())

    def _resolve_source(self, path: str) -> FieldEditor[Any] | None:
        """
        Find the editor an overlay source or requirement path refers to.

        Args:
            path: Dotted path relative to the declaring object.

        Returns:
            The editor holding the current value, or None if the path matches
            nothing in the form as it now stands. A path leading into a union
            variant that is not selected resolves to None.
        """
        editors = editors_at(self.siblings, self.scope, path, all_variants=False)
        return editors[0] if editors else None

    def _requirements_met(self, spec: OverlaySpec) -> bool:
        """
        Check a spec's requirements against the values shown in the form.

        Args:
            spec: The overlay whose requirements to check.

        Returns:
            True if every required path holds a value of its required type.
        """
        for path, kind in spec.requires:
            editor = self._resolve_source(path)
            value = _shown_value(editor) if editor is not None else None
            if not isinstance(value, kind):
                return False
        return True


@dataclass
class FollowerBinding:
    """
    A nested union whose variant is chosen by a field elsewhere in the form.

    Attributes:
        union: The inner union editor, built with its selector hidden.
        source_path: Dotted path to the driving field, relative to `scope`,
            resolved outward toward the form root.
        fallback: Discriminator value selected when the source holds a value
            matching no variant, or when the source cannot be resolved.
        field_name: Name of the union arm, used in warnings.
        scope: The scope of the union field.
        wired: True once the source subscription has been made.
    """

    union: FollowerTarget
    source_path: str
    fallback: str
    field_name: str
    scope: BuildScope
    wired: bool = False
    _active: bool = field(default=True, init=False, repr=False)

    @property
    def active(self) -> bool:
        """False once the union editor has been destroyed."""
        return self._active

    def deactivate(self, *_: object) -> None:
        """Stop pushing to a union editor that is gone."""
        self._active = False

    def push(self, source: FieldEditor[Any]) -> None:
        """
        Select the nested variant matching the driving field's current value.

        Args:
            source: The editor holding the driving value.
        """
        if not self._active:
            return

        value = _shown_value(source)
        tag = value.value if isinstance(value, Enum) else value

        if isinstance(tag, str) and self.union.select(tag):
            return

        self.scope.context.form.warn(
            f"{self.source_path!r} is {value!r}, which matches no "
            f"{self.field_name!r} variant; using {self.fallback!r}."
        )
        self.union.select(self.fallback)


def collect_overlays(
    infos: Sequence[FieldInfo],
    editors: dict[str, FieldEditor[Any]],
    scope: BuildScope,
) -> None:
    """
    Queue the overlays of an object's area selectors for wiring after the build.

    Args:
        infos: Field infos of the object just built.
        editors: Its editors, keyed by field name.
        scope: The scope of that object.
    """
    for fi in infos:
        if fi.hint is None or not fi.hint.overlays:
            continue

        editor = editors.get(fi.name)
        area = editor.value_editor() if editor is not None else None
        if not isinstance(area, OverlayTarget):
            scope.context.form.warn(
                f"Field {fi.name!r} declares area overlays but is not an "
                "area selector; they will be ignored."
            )
            continue

        binding = OverlayBinding(
            area=area,
            area_path=scope.child(fi.name).dotted,
            field_name=fi.name,
            specs=tuple(fi.hint.overlays),
            scope=scope,
            siblings=editors,
        )

        # a removed list item destroys its area editor without any other
        # notification, so stop pushing when the widget dies
        area.destroyed.connect(binding.deactivate)
        scope.context.overlays.append(binding)


def wire_followers(context: BuildContext) -> None:
    """
    Subscribe every nested union to the field driving its selection, then push.

    Args:
        context: The build whose followers should be wired.
    """
    for binding in list(context.followers):
        if binding.wired or not binding.active:
            continue
        binding.wired = True

        sources = editors_at({}, binding.scope, binding.source_path, all_variants=False)
        if not sources:
            context.form.warn(
                f"Variant {binding.field_name!r} follows "
                f"{binding.source_path!r}, which was not found in this "
                f"form; {binding.fallback!r} is used."
            )
            binding.union.select(binding.fallback)
            continue

        source = sources[0]
        push = partial(binding.push, source)
        source.on_change(push)
        push()


def wire_overlays(context: BuildContext) -> None:
    """
    Subscribe every overlay binding to its sources, then push.

    One callback per area selector drives all of that field's overlays, so
    each push carries complete values per overlay and no partial state is
    observable when several sources feed one decoration.

    Args:
        context: The build whose overlay bindings should be wired.
    """
    for binding in list(context.overlays):
        if binding.wired or not binding.active:
            continue
        binding.wired = True

        sources = binding.sources()
        if not sources and not binding.reads_state:
            continue

        for source in sources:
            source.on_change(binding.push)
        binding.push()


def editors_at(
    siblings: Mapping[str, FieldEditor[Any]],
    scope: BuildScope,
    path: str,
    all_variants: bool,
) -> list[FieldEditor[Any]]:
    """
    Walk a dotted path down through the form's editors.

    Args:
        siblings: Editors of the object the path is relative to, by field
            name. Pass an empty mapping to resolve from the form's editors
            alone, which is what a reference made from no particular object needs.
        scope: The scope the reference is made from.
        path: Dotted path relative to that scope.
        all_variants: If True, descend into every variant of a union rather
            than only the selected one. Use this when subscribing, so an edit
            in a variant selected later still reaches the subscriber.

    Returns:
        The matching value editors. At most one unless `all_variants` is set.
    """
    parts = path.split(".")

    first = siblings.get(parts[0])
    if first is None:
        return _registry_lookup(scope, path)

    found = [first.value_editor()]
    for part in parts[1:]:
        found = [
            child
            for editor in found
            for child in editor.child_editors(part, all_variants)
        ]
    return found


def _registry_lookup(scope: BuildScope, path: str) -> list[FieldEditor[Any]]:
    """
    Resolve a path that starts outside the declaring object.

    Args:
        scope: The scope the reference is made from.
        path: Dotted path relative to some enclosing scope.

    Returns:
        The single matching value editor, or an empty list.
    """
    editors = scope.context.editors
    for candidate in scope.candidates(path):
        if (editor := editors.get(candidate)) is not None:
            return [editor.value_editor()]
    return []

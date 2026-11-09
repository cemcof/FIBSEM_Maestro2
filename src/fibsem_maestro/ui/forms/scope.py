# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Per-build state of a form, and the position of the object being built."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.ui.forms.commit import FieldCommitter
    from fibsem_maestro.ui.forms.context import FormContext
    from fibsem_maestro.ui.forms.editor import FieldEditor, OnChange
    from fibsem_maestro.ui.forms.wiring import FollowerBinding, OverlayBinding

T = TypeVar("T")

ObjectPath = tuple[str, ...]
"""Attribute names leading from a form's root object to a nested object."""


def noop_commit(path: str) -> None:
    """A do-nothing `on_commit` (`BuildContext.on_commit`, `FormBuilder.build_live`)."""
    _ = path


def resolve_path(root: Any, path: ObjectPath) -> Any | None:
    """
    Follow attribute names from `root`.

    Args:
        root: The object to start from.
        path: The attribute names, outermost first; empty for `root` itself.

    Returns:
        The object found, or None if an attribute is missing or None on the way.
    """
    current = root
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


@dataclass
class BuildContext:
    """
    State shared by every scope within one form build.

    Attributes:
        form: The environment the form is built in.
        on_commit: Called by every committer with the dotted path of the field
            it wrote, after the write.
        editors: Built field editors by dotted path from the form root. Only
            addressable fields appear here (list elements are excluded).
        overlays: Overlay bindings collected during the build, wired once the
            whole tree exists.
        followers: Nested-union bindings collected during the build, wired once
            the whole tree exists.
        committers: Every committer, checked once the whole form exists so a
            stored value the form cannot show is marked at once.
        disposers: Callables releasing what the form registered outside itself
            (settings hooks, catalog connections); run by `dispose`.
        disposed: True once `dispose` ran. A disposed form is inert: its
            committers neither validate nor write, and it reports no change
            of validity, even while its widgets still exist.
    """

    form: FormContext
    on_commit: Callable[[str], None] = noop_commit
    editors: dict[str, FieldEditor[Any]] = field(default_factory=dict)
    overlays: list[OverlayBinding] = field(default_factory=list)
    followers: list[FollowerBinding] = field(default_factory=list)
    committers: list[FieldCommitter] = field(default_factory=list)
    disposers: list[Callable[[], None]] = field(default_factory=list)
    disposed: bool = False
    _issue_listeners: list[OnChange] = field(default_factory=list, repr=False)
    _change_listeners: list[OnChange] = field(default_factory=list, repr=False)

    def notify_issues_changed(self) -> None:
        """Tell the listeners that the issues of some field of the form changed."""
        for listener in list(self._issue_listeners):
            listener()

    def on_issues_changed(self, listener: OnChange) -> None:
        """
        Register a callback fired whenever the issues of a field of the form change.

        Covers fields of every object editor of the form, including those built
        after the initial build (list items).

        Args:
            listener: The callback.
        """
        self._issue_listeners.append(listener)

    def notify_changed(self) -> None:
        """Tell the listeners that an editor of a detached form changed."""
        for listener in list(self._change_listeners):
            listener()

    def on_changed(self, listener: OnChange) -> None:
        """
        Register a callback fired whenever an editor of a detached form changes.

        Args:
            listener: The callback.
        """
        self._change_listeners.append(listener)

    def dispose(self) -> None:
        """Make the form inert, then run and forget every disposer. Idempotent."""
        self.disposed = True
        disposers, self.disposers = self.disposers, []
        for disposer in disposers:
            disposer()


@dataclass(frozen=True)
class BuildScope:
    """
    The position of the object currently being built within the settings tree.

    Live objects are referred to by their path from the root, never held
    directly, so an object replaced in the settings (e.g. by toggling an
    optional group) is found again on the next lookup.

    Attributes:
        context: Per-build shared state.
        root: The live root object, or None for a detached form.
        live: Paths of the live objects enclosing this scope, outermost first.
        owner: Path of the live object whose fields are built in this scope,
            or None in a value-only subtree.
        path: Field names from the root, or None if this subtree is
            unaddressable (list elements).
    """

    context: BuildContext
    root: Any | None = None
    live: tuple[ObjectPath, ...] = ()
    owner: ObjectPath | None = None
    path: tuple[str, ...] | None = ()

    @property
    def dotted(self) -> str | None:
        """The dotted path of this scope, or None if it has no address."""
        return None if self.path is None else ".".join(self.path)

    @property
    def is_live(self) -> bool:
        """True if some live object encloses this scope."""
        return bool(self.live)

    def enter_live(self, path: ObjectPath) -> BuildScope:
        """
        Return the scope of the live object at `path`, whose fields are built next.

        Args:
            path: The object's path from the root.

        Returns:
            This scope with `path` appended as the innermost live object.
        """
        return replace(self, live=(*self.live, path), owner=path)

    def enter_value_only(self) -> BuildScope:
        """Return this scope for an object built without a live instance."""
        return replace(self, owner=None)

    def child(self, name: str) -> BuildScope:
        """
        Return the scope of field `name` within this object.

        Args:
            name: The field name.

        Returns:
            The scope, addressed by this scope's path plus `name`, or
            unaddressed if this scope is.
        """
        if self.path is None:
            return self
        return replace(self, path=(*self.path, name))

    def unaddressed(self) -> BuildScope:
        """Return this scope with its address dropped (for list elements)."""
        return replace(self, path=None)

    def rebased(self, root: Any) -> BuildScope:
        """
        Return a scope resolving values against another live root only.

        Args:
            root: The object to resolve against, e.g. a linked action's settings.

        Returns:
            A scope whose only live object is `root`.
        """
        return replace(self, root=root, live=((),), owner=())

    def candidates(self, path: str) -> tuple[str, ...]:
        """
        Absolute candidate paths for a reference made from this scope.

        Args:
            path: Dotted path relative to this scope, e.g. `"milling_depth"` or
                `"milling.milling_depth"`.

        Returns:
            Candidate absolute paths, innermost scope first, ending at the root.
            An unaddressed scope can only offer the root-relative path.
        """
        if self.path is None:
            return (path,)
        prefixes = (self.path[:i] for i in range(len(self.path), -1, -1))
        return tuple(".".join((*prefix, path)) for prefix in prefixes)

    def value(self, path: str, kind: type[T]) -> T | None:
        """
        Resolve a dotted path against the live objects, innermost first.

        Each live object is looked up from the root now, so the value reflects
        the settings as they are at the time of the call.

        Args:
            path: Dotted attribute path relative to a live object.
            kind: Required type of the resolved value.

        Returns:
            The first value of type `kind` found, or None if no enclosing live
            object defines one.
        """
        parts = tuple(path.split("."))
        for object_path in reversed(self.live):
            obj = resolve_path(self.root, object_path)
            if obj is None:
                continue
            current = resolve_path(obj, parts)
            if isinstance(current, kind):
                return current
        return None

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import builtins
import inspect
import itertools
from typing import TYPE_CHECKING

from fibsem_maestro.microscope.abstract_control.manufacturer_props import (
    ManufacturerPropertiesRegistry,
    ManufacturerProperty,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_SHARED_MEMBERS = frozenset({"application_client"})
"""
Members that point at shared infrastructure rather than at a part of the owner.

Every Autoscript proxy keeps a reference to the client it was created by.
Following it would make the whole microscope reachable from any beam, under
paths that depend on traversal order.
"""


class AutoscriptManufacturerPropertiesRegistry(ManufacturerPropertiesRegistry):
    """
    Implementation of the ManufacturerPropertiesRegistry for autoscript.
    """

    def _build(self, root: object) -> None:
        reg: dict[str, ManufacturerProperty] = {}
        for prop in find_manufacturer_properties(root):
            reg[str(prop)] = prop
        self._registry = reg


def _clean_member_name(raw_name: str, owner_cls: type) -> str:
    """
    Normalize an instance attribute name to a public member name.

    This removes Python name-mangling and strips all leading underscores so that
    member names can be used consistently in object paths.

    Examples:
        "_x" -> "x"
        "__x" -> "x"
        "_Class__x" -> "x"

    Args:
        raw_name (str): Raw attribute name as stored on the instance.
        owner_cls (type): Class of the owning instance, used to resolve
            name-mangled attributes.

    Returns:
        str: Cleaned member name without leading underscores.
    """
    # unmangle "_Class__x" -> "__x" by stripping the "_Class" part if present
    for base in inspect.getmro(owner_cls):
        prefix = f"_{base.__name__}__"
        if raw_name.startswith(prefix):
            raw_name = "__" + raw_name[len(prefix) :]
            break

    # remove all leading underscores
    return raw_name.lstrip("_")


def _iter_instance_members(obj: object) -> Iterable[tuple[str, object]]:
    """
    Iterate over member variables stored directly on an instance.

    This yields cleaned member names and their values without invoking property
    getters. Both `__dict__` attributes and `__slots__` (including name-mangled
    slots) are supported.

    Args:
        obj (object): Instance whose member variables should be inspected.

    Yields:
        tuple[str, object]: Pairs of `(member_name, value)` where `member_name`
        has no leading underscores.
    """
    owner_cls = type(obj)

    # __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k, v in d.items():
            yield _clean_member_name(str(k), owner_cls), v

    # __slots__
    for base in inspect.getmro(owner_cls):
        slots = getattr(base, "__slots__", ())
        if not slots:
            continue
        if isinstance(slots, str):
            slots = (slots,)

        for slot in slots:
            if slot in ("__dict__", "__weakref__"):
                continue

            # access name may be mangled; declared name is `slot`
            access_name = slot
            if slot.startswith("__") and not slot.endswith("__"):
                access_name = f"_{base.__name__}{slot}"

            try:
                value = getattr(obj, access_name)
            except AttributeError:
                continue
            except Exception:
                continue

            yield _clean_member_name(slot, owner_cls), value


def _is_traversable_instance(x: object) -> bool:
    """
    Determine whether a value should be traversed as part of the object graph.

    This filters out primitives, containers, modules, functions, methods, and
    classes, leaving only user-defined object instances.

    Args:
        x (object): Value to test.

    Returns:
        bool: `True` if the value should be traversed, `False` otherwise.
    """
    if x is None:
        return False
    if isinstance(x, (str, bytes, bytearray, int, float, bool, complex)):
        return False
    if isinstance(x, (list, tuple, set, frozenset, dict)):
        return False

    return not (
        inspect.ismodule(x)
        or inspect.isfunction(x)
        or inspect.ismethod(x)
        or inspect.isclass(x)
    )


def _properties_with_setters(cls: type) -> list[str]:
    """
    Return names of properties on a class that define a setter.

    The method walks the class MRO and collects property names whose descriptors
    include a non-`None` setter function.

    Args:
        cls (type): Class to inspect.

    Returns:
        list[str]: Names of properties that have setters.
    """
    names: list[str] = []
    seen: set[str] = set()
    for base in inspect.getmro(cls):
        for name, attr in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(attr, builtins.property) and attr.fset is not None:
                names.append(name)
    return names


def _read_only_properties(obj: object) -> Iterable[tuple[str, object]]:
    """
    Iterate over the public properties without a setter of an object, with their values.

    Properties whose getter raises are skipped.

    Args:
        obj (object): Instance whose properties should be read.

    Yields:
        tuple[str, object]: Pairs of `(property_name, value)`.
    """
    seen: set[str] = set()
    for base in inspect.getmro(type(obj)):
        for name, attr in vars(base).items():
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)
            if not isinstance(attr, builtins.property) or attr.fset is not None:
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            yield name, value


def find_manufacturer_properties(root: object) -> list[ManufacturerProperty]:
    """
    Discover writable properties reachable from a root object.

    The object graph is traversed through member variables, and on the root
    also through its public read-only properties (e.g. the subsystems of a
    microscope client, such as `detector`). Paths are constructed using
    member and property names without leading underscores and contain no
    class or type names.

    Args:
        root (object): Root object from which traversal begins.

    Returns:
        list[ManufacturerProperty]: All discovered writable properties with their
        associated owners and member-based paths.
    """
    out: list[ManufacturerProperty] = []
    visited: set[int] = set()
    stack: list[tuple[object, str]] = [(root, "")]  # (instance, path_of_member_names)

    while stack:
        obj, obj_path = stack.pop()
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)

        for prop_name in _properties_with_setters(type(obj)):
            out.append(
                ManufacturerProperty(owner=obj, owner_path=obj_path, name=prop_name)
            )

        members = _iter_instance_members(obj)
        if obj is root:
            # some subsystems of the client (e.g. the detector) are no members
            # of it, so they would be missed: `application_client` is not followed
            members = itertools.chain(members, _read_only_properties(obj))

        for member_name, child in members:
            if member_name in _SHARED_MEMBERS:
                continue
            if _is_traversable_instance(child):
                child_path = f"{obj_path}.{member_name}" if obj_path else member_name
                stack.append((child, child_path))

    return out

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

T = TypeVar("T")
TObj = TypeVar("TObj")


class RegistryError(Exception):
    """Raised when a registry operation fails."""


class Registry[T]:
    """
    Maps string names to objects (classes, functions, or other values).

    Args:
        name: Human-readable name for the kind of object stored,
            used in error messages (e.g. "autofunction mode", "criterion").
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    def add(self, key: str, obj: T) -> None:
        """
        Register an object under the given name.

        Args:
            key: The name under which to register the object.
            obj: The object to register.

        Raises:
            RegistryError: If the name is already registered.
        """
        if key in self._entries:
            raise RegistryError(
                f"{self._name.capitalize()} '{key}' is already registered."
            )
        self._entries[key] = obj

    def register(self, key: str) -> Callable[[TObj], TObj]:
        """
        Decorator that registers an object under the given name.

        Args:
            key: The name under which to register the object.

        Returns:
            A decorator that registers the object and returns it
            unchanged.

        Raises:
            RegistryError: If the name is already registered.
        """

        def decorator(obj: TObj) -> TObj:
            self.add(key, cast("T", obj))
            return obj

        return decorator

    def get(self, key: str) -> T:
        """
        Return the registered object for the given name.

        Args:
            key: The registered name to look up.

        Returns:
            The registered object.

        Raises:
            RegistryError: If the name is not registered.
        """
        if key not in self._entries:
            raise RegistryError(
                f"Unknown {self._name} '{key}'. "
                f"Available: {', '.join(sorted(self._entries))}."
            )
        return self._entries[key]

    def key_of(self, obj: T) -> str:
        """
        Return the name under which an object is registered.

        Args:
            obj: The registered object to look up.

        Returns:
            The name under which the object is registered.

        Raises:
            RegistryError: If the object is not registered.
        """
        for key, registered in self._entries.items():
            if registered is obj:
                return key
        raise RegistryError(
            f"Object {obj!r} is not registered in the {self._name} registry."
        )

    def validate(self, key: str) -> str:
        """
        Validate that a name is registered.

        Raises `ValueError` on unknown keys, making it directly usable
        as a Pydantic `AfterValidator`:

        ```python
        from pydantic import AfterValidator
        from typing import Annotated

        ReductionName = Annotated[str, AfterValidator(self.validate)]
        ```

        Args:
            key: The name to validate.

        Returns:
            The name, unchanged.

        Raises:
            ValueError: If the name is not registered.
        """
        if key not in self._entries:
            raise ValueError(
                f"Unknown {self._name} '{key}'. "
                f"Available: {', '.join(sorted(self._entries))}."
            )
        return key

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._entries))
        return f"Registry[{self._name}]({keys})"

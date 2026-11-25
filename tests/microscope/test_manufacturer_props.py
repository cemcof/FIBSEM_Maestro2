# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Discovery of the writable manufacturer properties reachable from a client."""

from fibsem_maestro.microscope.autoscript_control.manufacturer_props import (
    find_manufacturer_properties,
)


class _Value:
    def __init__(self) -> None:
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value


class _Detector:
    def __init__(self, client: object) -> None:
        self._application_client = client
        self._brightness = _Value()

    @property
    def mode(self) -> _Value:
        """A read-only property of a subsystem: not followed."""
        return _Value()


class _Beams:
    def __init__(self) -> None:
        self._beam_current = _Value()


class _Client:
    """A client holding one subsystem as a member and creating another on access."""

    def __init__(self) -> None:
        self._beams = _Beams()

    @property
    def detector(self) -> _Detector:
        return _Detector(self)

    @property
    def broken(self) -> object:
        raise RuntimeError("not connected")


def _names(root: object) -> list[str]:
    return sorted(str(p) for p in find_manufacturer_properties(root))


def test_manufacturer_properties_include_subsystems_created_on_access() -> None:
    # the detector of the Autoscript client was missing from the list
    assert "detector.brightness.value" in _names(_Client())


def test_manufacturer_properties_keep_the_subsystems_held_as_members() -> None:
    assert "beams.beam_current.value" in _names(_Client())


def test_manufacturer_properties_follow_read_only_properties_of_the_root_only() -> None:
    names = _names(_Client())

    assert not [n for n in names if "mode" in n]
    assert not [n for n in names if "application_client" in n]

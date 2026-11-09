# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fibsem_maestro.action.action import Action


class WidgetType(Enum):
    DROPDOWN = "dropdown"
    MULTI_SELECT = "multi_select"
    PROPERTY_SELECTOR = "property_selector"
    MULTI_PROPERTY_SELECTOR = "multi_property_selector"
    AREA_SELECT = "area_select"
    RANGE_PAIR = "range_pair"
    DETAIL_BAND = "detail_band"
    STRING = "string"
    ACTION_SELECTOR = "action_selector"
    MULTI_ACTION_SELECTOR = "multi_action_selector"
    PATTERN_TYPE_SELECTOR = "pattern_type_selector"


class AreaOverlay(Enum):
    """Optional, display-only decoration drawn over every acquisition area."""

    SHOW_MARGIN = "show_margin"
    SHOW_DIRECTION = "show_direction"
    SHOW_TILES = "show_tiles"
    SHOW_AREA_SLICES = "show_area_slices"

    @property
    def data_fields(self) -> frozenset[str]:
        """Names of the `OverlayData` attributes this overlay reads from."""
        return {
            AreaOverlay.SHOW_MARGIN: frozenset({"margin_nm"}),
            AreaOverlay.SHOW_DIRECTION: frozenset({"direction"}),
            AreaOverlay.SHOW_TILES: frozenset(
                {"tile_size_nm", "tile_relative_overlap"}
            ),
            AreaOverlay.SHOW_AREA_SLICES: frozenset({"slice_distance_nm", "direction"}),
        }[self]


@dataclass(frozen=True)
class OverlaySpec:
    """
    One decoration to draw over an area selector, and where its values come from.

    Attributes:
        kind: Which decoration to draw.
        sources: `(OverlayData attribute, dotted field path)` pairs. Paths are
            resolved against the settings object declaring the hint, then
            outward toward the form root; the innermost match wins. An
            unresolved or None-valued source leaves that attribute None, which
            suppresses this decoration without affecting the others on the
            same field.
        requires: `(dotted path, required type)` pairs that must all hold in the
            live settings for this decoration to be drawn.

    """

    kind: AreaOverlay
    sources: tuple[tuple[str, str], ...] = ()
    requires: tuple[tuple[str, type], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.sources]
        if unknown := set(names) - self.kind.data_fields:
            raise ValueError(
                f"{self.kind.name} does not read {sorted(unknown)}; "
                f"it reads {sorted(self.kind.data_fields)}"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"{self.kind.name} has duplicate overlay sources")

    @classmethod
    def of(
        cls,
        kind: AreaOverlay,
        requires: Mapping[str, type] | None = None,
        **paths: str,
    ) -> OverlaySpec:
        """
        Build a spec from keyword sources.

        Args:
            kind: Which decoration to draw.
            requires: Dotted path to required type, e.g.
                `{"tiling_mode": MultiTileMode}`.
            **paths: `OverlayData` attribute name to dotted field path.

        Returns:
            The spec.
        """
        return cls(kind, tuple(paths.items()), tuple((requires or {}).items()))


@dataclass(frozen=True)
class NestedUnion:
    """
    Marks a discriminated union used as one arm of an enclosing union.

    Attributes:
        label: Text of the enclosing union's radio button for this arm.
        follows: Name of a field outside the union whose value selects the
            inner variant. Dotted paths are resolved against the object
            declaring the union field, then outward toward the form root.
            If None, the inner union shows its own radio row.
    """

    label: str
    follows: str | None = None


@dataclass(frozen=True)
class AcquisitionOffset:
    """
    Offset applied to the beam while an area selector grabs its image.

    Attributes:
        source: Dotted path to the field holding the x offset, in nm.
            Resolved against the settings object declaring the hint, then
            outward toward the form root.
    """

    source: str


@dataclass(frozen=True)
class BeamSource:
    """
    Where an area selector finds the beam its preview is acquired with.

    Attributes:
        path: Dotted path to a `BeamType`. Resolved against the settings
            declaring the hint, then outward toward the form root.
        via: Name of a field holding the name of another action. If set, `path`
            is resolved in that action's settings instead, e.g. for a refocus,
            which images with the beam of its linked imaging.
    """

    path: str
    via: str | None = None


@dataclass(frozen=True)
class CropSource:
    """
    Where an area selector finds the scanning area its preview is cropped to.

    Areas drawn on a cropped preview are relative to the scanned area, which is
    what the frames they are applied to show.

    Attributes:
        path: Dotted path to a `list[RelativeArea]`; its first area is the crop
            and an empty list means the whole frame. Resolved against the
            settings declaring the hint, then outward toward the form root.
        via: Name of a field holding the name of another action. If set, `path`
            is resolved in that action's settings instead: the frames come from
            that action, so the preview is taken without the acquisition offset.
    """

    path: str
    via: str | None = None


@dataclass
class FormHint:
    widget: WidgetType
    choices: Callable[[], list[str]] | None = None
    file_filter: str | None = None
    max_areas: int | None = None
    min_areas: int | None = None
    action_type_filter: list[type[Action]] = field(default_factory=list)
    overlays: tuple[OverlaySpec, ...] = ()
    # tried in order; the first that resolves to a beam decides
    beam_sources: tuple[BeamSource, ...] = ()
    offset: AcquisitionOffset | None = None
    # tried in order; the first that resolves to a list decides, even an empty one
    crop_sources: tuple[CropSource, ...] = ()


@dataclass
class FieldUnit:
    suffix: str

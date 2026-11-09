# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
Synthetic settings classes for the tests of the form machinery.

They are built like the real settings (`BaseSettings`, `Annotated[...]` with
`Field`, `FormHint` and `FieldUnit`) but never change with them, so the form
tests do not depend on what the real actions happen to contain. None of them
is registered as an action.

`KitchenSink` holds one field of every editor kind the default editor registry
builds; when a new editor kind is added, add a field using it here.

No `from __future__ import annotations` here: the `choices` lambdas inside
string annotations would be compiled without this module's names.
"""

from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fibsem_maestro.action.action import Action
from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.detail_band import DetailBand
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.core.pattern_type import PatternType
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import (
    AcquisitionOffset,
    AreaOverlay,
    BeamSource,
    CropSource,
    FieldUnit,
    FormHint,
    NestedUnion,
    OverlaySpec,
    WidgetType,
)

PATTERN_TYPES: tuple[PatternType, ...] = (PatternType("rect"), PatternType("circle"))
"""The pattern types a form context offers to `KitchenSink.pattern`."""

VENDOR_PROPERTIES: tuple[str, ...] = ("vendor_gain",)
"""The manufacturer property names a form context offers to the property selectors."""

LINKED_NAME = "linked action"
"""Name of the action `filled_kitchen_sink` links to; the form's catalog must hold it."""

QUARTER = RelativeArea(origin=RelativePoint(x=0.25, y=0.25), width=0.5, height=0.5)
"""A centred area covering a quarter of the frame."""


class Color(Enum):
    """A plain enumeration."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Leaf(BaseSettings):
    """A small nested object: a nested group, an optional group, list items."""

    count: Annotated[int, Field(ge=0)] = Field(default=1, description="How many.")
    label: str = Field(default="leaf", description="A short name.")


class CircleShape(BaseSettings):
    """Union variant with a constrained field."""

    kind: Literal["circle"] = Field(default="circle", description="A round shape.")
    radius: Annotated[float, Field(gt=0), FieldUnit(suffix="nm")] = Field(
        default=1.0, description="Its radius."
    )
    margin: Annotated[float, Field(ge=0), FieldUnit(suffix="nm")] = Field(
        default=0.0, description="Space around it."
    )


class SquareShape(BaseSettings):
    """Union variant with a model-level validator."""

    kind: Literal["square"] = Field(default="square", description="A square shape.")
    side: Annotated[float, Field(gt=0), FieldUnit(suffix="nm")] = Field(
        default=2.0, description="Its side."
    )
    margin: Annotated[float, Field(ge=0), FieldUnit(suffix="nm")] = Field(
        default=0.0, description="Space around it."
    )

    @model_validator(mode="after")
    def _margin_inside(self) -> Self:
        if self.margin >= self.side:
            raise ValueError("The margin must be smaller than the side.")
        return self


class PointShape(BaseSettings):
    """Union variant without any field besides its tag (an empty page)."""

    kind: Literal["point"] = Field(default="point", description="A single point.")


Shape = Annotated[CircleShape | SquareShape | PointShape, Field(discriminator="kind")]
"""A plain discriminated union."""


class ManualMode(BaseSettings):
    """Leaf arm of `Mode`."""

    kind: Literal["manual"] = Field(default="manual", description="Done by hand.")
    steps: Annotated[int, Field(gt=0)] = Field(default=2, description="How many steps.")


class EngineBase(BaseSettings):
    """Common base of the arms of the nested union `EngineMode`."""

    kind: Literal["engine"] = Field(default="engine", description="Done by an engine.")


class FastEngine(EngineBase):
    """Inner variant of `EngineMode`."""

    engine: Literal["fast"] = "fast"
    speed: Annotated[int, Field(gt=0)] = Field(default=10, description="How fast.")


class PreciseEngine(EngineBase):
    """Inner variant of `EngineMode`."""

    engine: Literal["precise"] = "precise"
    precision: Annotated[float, Field(gt=0)] = Field(
        default=0.5, description="How precise."
    )


EngineMode = Annotated[
    FastEngine | PreciseEngine,
    Field(discriminator="engine"),
    NestedUnion(label="engine mode", follows="engine"),
]
"""A nested union arm whose variant follows the field `engine`."""

Mode = Annotated[ManualMode | EngineMode, Field(discriminator="kind")]
"""A union with a leaf arm and a nested arm."""


def _engines() -> list[str]:
    # "turbo" matches no `EngineMode` variant
    return ["fast", "precise", "turbo"]


class KitchenSink(BaseSettings):
    """
    One field of every editor kind the default editor registry builds.

    `KitchenSink()` is valid in a form context offering `PATTERN_TYPES` and an
    action catalog (see `sink_context`): no field shows an error.
    """

    flag: bool = Field(default=False, description="Switch it on.")
    count: Annotated[int, Field(ge=0, le=100)] = Field(
        default=3, description="How many."
    )
    width: Annotated[float, Field(gt=0), FieldUnit(suffix="nm")] = Field(
        default=5.0, description="How wide."
    )
    title: str = Field(default="sample", description="A short name.")
    color: Color = Field(default=Color.RED, description="Pick a colour.")
    size: Literal["small", "large"] = Field(default="small", description="Pick a size.")
    direction: Annotated[
        Direction,
        FormHint(
            widget=WidgetType.DROPDOWN, choices=lambda: [Direction.DOWN, Direction.UP]
        ),
    ] = Field(default=Direction.DOWN, description="Which way.")
    engine: Annotated[str, FormHint(widget=WidgetType.DROPDOWN, choices=_engines)] = (
        Field(default="fast", description="Which engine.")
    )
    property_name: Annotated[
        str,
        FormHint(
            widget=WidgetType.PROPERTY_SELECTOR, choices=lambda: ["focus", "contrast"]
        ),
    ] = Field(default="focus", description="One property.")
    tags: Annotated[
        list[str],
        FormHint(widget=WidgetType.MULTI_SELECT, choices=lambda: ["a", "b", "c"]),
    ] = Field(default_factory=list, description="Some tags.")
    property_names: Annotated[
        list[str],
        FormHint(
            widget=WidgetType.MULTI_PROPERTY_SELECTOR,
            choices=lambda: ["focus", "contrast"],
        ),
    ] = Field(default_factory=list, description="Some properties.")
    pattern: Annotated[
        PatternType, FormHint(widget=WidgetType.PATTERN_TYPE_SELECTOR)
    ] = Field(default=PATTERN_TYPES[0], description="A pattern.")
    linked: Annotated[
        str | None,
        FormHint(widget=WidgetType.ACTION_SELECTOR, action_type_filter=[Action]),
    ] = Field(default=None, description="Another action.")
    linked_many: Annotated[
        list[str],
        FormHint(widget=WidgetType.MULTI_ACTION_SELECTOR, action_type_filter=[Action]),
    ] = Field(default_factory=list, description="Other actions.")
    area: Annotated[
        list[RelativeArea],
        FormHint(
            widget=WidgetType.AREA_SELECT,
            min_areas=1,
            max_areas=2,
            overlays=(
                OverlaySpec.of(AreaOverlay.SHOW_DIRECTION, direction="direction"),
                OverlaySpec.of(AreaOverlay.SHOW_MARGIN, margin_nm="shape.margin"),
                OverlaySpec.of(
                    AreaOverlay.SHOW_TILES,
                    requires={"shape": SquareShape},
                    tile_size_nm="shape.side",
                    tile_relative_overlap="overlap",
                ),
            ),
        ),
    ] = Field(default_factory=lambda: [RelativeArea.full()], description="Where.")
    overlap: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.0, description="Tile overlap."
    )
    window: Annotated[
        tuple[float, float], FormHint(widget=WidgetType.RANGE_PAIR), FieldUnit("nm")
    ] = Field(default=(1.0, 2.0), description="A range.")
    band: Annotated[
        DetailBand, FormHint(widget=WidgetType.DETAIL_BAND), FieldUnit("nm")
    ] = Field(default=DetailBand(low=1.0, high=4.0), description="A detail band.")
    notes: Annotated[str, FormHint(widget=WidgetType.STRING)] = Field(
        default="first line", description="Free text."
    )
    leaf: Leaf = Field(default_factory=Leaf, description="A nested object.")
    extra: Leaf | None = Field(default=None, description="An optional object.")
    shape: Shape = Field(default_factory=CircleShape, description="A shape.")
    mode: Mode = Field(default_factory=ManualMode, description="How it is done.")
    numbers: list[int] = Field(default_factory=lambda: [1, 2], description="Numbers.")
    leaves: list[Leaf] = Field(default_factory=list, description="Objects.")
    offset: tuple[float, float] = Field(default=(0.0, 0.0), description="A pair.")
    mapping: dict[str, int] = Field(
        default_factory=lambda: {"a": 1}, description="No dedicated editor."
    )
    limit: Annotated[float, Field(gt=0)] | None = Field(
        default=None, description="An optional number."
    )


def filled_kitchen_sink() -> KitchenSink:
    """
    A valid `KitchenSink` in which every field differs from its default.

    Its `linked` and `linked_many` fields name `LINKED_NAME`, so the form's
    action catalog must hold an action of that name.

    Returns:
        The instance.
    """
    return KitchenSink(
        flag=True,
        count=7,
        width=2.5,
        title="other",
        color=Color.BLUE,
        size="large",
        direction=Direction.UP,
        engine="precise",
        property_name=VENDOR_PROPERTIES[0],
        tags=["a", "c"],
        property_names=["contrast", VENDOR_PROPERTIES[0]],
        pattern=PATTERN_TYPES[1],
        linked=LINKED_NAME,
        linked_many=[LINKED_NAME],
        area=[QUARTER],
        overlap=0.25,
        window=(3.0, 8.0),
        band=DetailBand(low=2.0, high=6.0),
        notes="second line",
        leaf=Leaf(count=4, label="inner"),
        extra=Leaf(count=2, label="present"),
        shape=SquareShape(side=9.0, margin=1.0),
        mode=PreciseEngine(precision=0.25),
        numbers=[5, 6, 7],
        leaves=[Leaf(count=3, label="first"), Leaf(count=5, label="second")],
        offset=(1.5, -2.5),
        mapping={"b": 2, "c": 3},
        limit=4.5,
    )


class Optionals(BaseSettings):
    """Every way a field can be optional, all None by default."""

    limit: Annotated[float, Field(gt=0)] | None = Field(
        default=None, description="Checkbox beside a number."
    )
    color: Color | None = Field(default=None, description="A '(none)' entry.")
    choice: Annotated[
        str | None, FormHint(widget=WidgetType.DROPDOWN, choices=lambda: ["x", "y"])
    ] = Field(default=None, description="Checkbox and a '(none)' entry.")
    extra: Leaf | None = Field(default=None, description="An optional group.")
    shape: Shape | None = Field(default=None, description="An optional union.")
    numbers: list[int] | None = Field(default=None, description="An optional list.")


class RequiredLink(BaseSettings):
    """A link to another action that must be set."""

    target: Annotated[
        str, FormHint(widget=WidgetType.ACTION_SELECTOR, action_type_filter=[Action])
    ] = Field(default="", description="The linked action.")


class PreviewSettings(BaseSettings):
    """An area selector using every preview context provider (beam, offset, crop)."""

    beam: BeamType | None = Field(
        default=BeamType.ION, description="Beam to image with; None for the linked one."
    )
    offset_x: Annotated[float, FieldUnit(suffix="nm")] = Field(
        default=0.0, description="Shift while imaging."
    )
    linked: str = Field(default="", description="Action whose frames are used.")
    scan_area: Annotated[
        list[RelativeArea], FormHint(widget=WidgetType.AREA_SELECT, max_areas=1)
    ] = Field(default_factory=list, description="Area that is scanned.")
    area: Annotated[
        list[RelativeArea],
        FormHint(
            widget=WidgetType.AREA_SELECT,
            min_areas=1,
            max_areas=2,
            # its own beam decides first, then that of a linked action
            beam_sources=(BeamSource("beam"), BeamSource("beam", via="linked")),
            offset=AcquisitionOffset(source="offset_x"),
            # a linked action decides first; its frames have no offset
            crop_sources=(
                CropSource("scan_area", via="linked"),
                CropSource("scan_area"),
            ),
        ),
    ] = Field(default_factory=lambda: [QUARTER], description="Area that is used.")


class FramedLayout(BaseSettings):
    """Union variant holding an area selector that needs an area."""

    kind: Literal["framed"] = Field(default="framed", description="With a frame.")
    frame: Annotated[
        list[RelativeArea], FormHint(widget=WidgetType.AREA_SELECT, min_areas=1)
    ] = Field(default_factory=lambda: [RelativeArea.full()], description="The frame.")


class PlainLayout(BaseSettings):
    """Union variant without fields."""

    kind: Literal["plain"] = Field(default="plain", description="Without a frame.")


class PagedArea(BaseSettings):
    """An editor-level error (area minimum) inside a union page."""

    layout: Annotated[FramedLayout | PlainLayout, Field(discriminator="kind")] = Field(
        default_factory=FramedLayout, description="The layout."
    )


class BrokenReferences(BaseSettings):
    """Cross-field references that cannot be resolved: the form warns and goes on."""

    area: Annotated[
        list[RelativeArea],
        FormHint(
            widget=WidgetType.AREA_SELECT,
            overlays=(OverlaySpec.of(AreaOverlay.SHOW_MARGIN, margin_nm="missing"),),
        ),
    ] = Field(default_factory=list, description="Where.")
    mode: Annotated[
        ManualMode
        | Annotated[
            FastEngine | PreciseEngine,
            Field(discriminator="engine"),
            NestedUnion(label="engine mode", follows="absent"),
        ],
        Field(discriminator="kind"),
    ] = Field(default_factory=ManualMode, description="How it is done.")

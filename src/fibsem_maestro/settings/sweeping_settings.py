# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Settings of the sweeps; a working-distance range defaults to the stored field width."""

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, ConfigDict, Field, field_validator, model_validator

from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.dynamic_default import (
    DefaultContext,
    DynamicDefault,
    properties_missing,
    properties_not_collected,
)
from fibsem_maestro.settings.form_utils import FormHint, WidgetType


class BasicStrategy(BaseSettings):
    type: Literal["basic"] = Field(
        default="basic",
        description="Steps evenly through the range, reversing direction on every cycle.",
    )


class InterleavedStrategy(BaseSettings):
    type: Literal["interleaved"] = Field(
        default="interleaved",
        description="Alternates base and sweep points; switches only if clearly sharper.",
    )
    min_diff: float = Field(
        default=0.005,
        ge=0.0,
        description="Minimal sharpness gain over the base required to switch to a candidate.",
    )


SweepingStrategySettings = Annotated[
    BasicStrategy | InterleavedStrategy, Field(discriminator="type")
]


NON_SCALAR_ATTRIBUTES = frozenset(
    {"stigmator", "lens_alignment", "beam_shift", "source_tilt", "resolution"}
)
"""Beam properties that are not single numbers and so cannot be swept."""


def _check_scalar(attribute: str) -> str:
    """
    Reject a beam property that is not a single number.

    Args:
        attribute: Name of the beam property to sweep.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If the property is not a single number.
    """
    if attribute in NON_SCALAR_ATTRIBUTES:
        raise ValueError(
            f"'{attribute}' is not a single number and cannot be swept; "
            f"use its x or y component, e.g. '{attribute}_x'."
        )
    return attribute


TargetAttribute = Annotated[
    str,
    AfterValidator(_check_scalar),
    FormHint(
        widget=WidgetType.PROPERTY_SELECTOR,
        # manufacturer properties are added dynamically
        choices=lambda: [
            name
            for name in BeamProperties.model_fields
            if name not in NON_SCALAR_ATTRIBUTES
        ],
    ),
]
"""Name of a scalar beam property that a sweep changes."""


RANGE_FIELD_DIVISOR = 20
"""The default working-distance range is ±(horizontal field width / this)."""

_RANGED_ATTRIBUTE = "working_distance"
"""The attribute whose range has a default."""


def _default_range(
    sweeping: LinearSweeping, context: DefaultContext
) -> tuple[float, float] | None:
    """±(stored field width / `RANGE_FIELD_DIVISOR`), for the working distance (nm)."""
    if sweeping.target_attribute != _RANGED_ATTRIBUTE:
        return None
    props = context.beam_properties()
    if props is None or props.horizontal_field_width is None:
        return None
    half = props.horizontal_field_width / RANGE_FIELD_DIVISOR
    return (-half, half)


def _no_default_range(sweeping: LinearSweeping, context: DefaultContext) -> str:
    """Why the range cannot be computed: another attribute, no properties, or no field width."""
    if sweeping.target_attribute != _RANGED_ATTRIBUTE:
        return (
            f"There is no default range for sweeping '{sweeping.target_attribute}'; "
            "set a range."
        )
    if context.beam_properties() is None:
        return properties_missing("field width", "a range")
    return properties_not_collected(
        "field width", "'horizontal_field_width'", "a range"
    )


class LinearSweeping(BaseSettings):
    """A sweep of one attribute over a range around its current value."""

    model_config = ConfigDict(use_enum_values=True)

    dimension: Literal["1d"] = Field(
        default="1d",
        description="Sweep one attribute over a range around its current value.",
    )
    target_attribute: TargetAttribute = Field(
        default="working_distance",
        description="Attribute to optimize.",
    )
    strategy: SweepingStrategySettings = Field(
        default_factory=BasicStrategy,
        description="Sweeping strategy to use.",
    )
    range: Annotated[
        tuple[float, float] | None,
        FormHint(widget=WidgetType.RANGE_PAIR),
        DynamicDefault(
            _default_range,
            _no_default_range,
            "the horizontal field width (HFW) of the stored properties: "
            f"±HFW/{RANGE_FIELD_DIVISOR}, for the working distance",
        ),
    ] = Field(
        default=None,
        description="Range of the sweep relative to the current value, in units of the attribute.",
    )
    steps: Annotated[int, Field(gt=0)] = Field(
        default=3,
        description="Number of steps in the sweeping range.",
    )
    cycles: Annotated[int, Field(gt=0)] = Field(
        default=1,
        description="Number of sweeping repeats.",
    )

    @field_validator("range")
    @classmethod
    def _check_range(cls, r: tuple[float, float] | None) -> tuple[float, float] | None:
        if r is not None and r[0] > r[1]:
            raise ValueError(
                f"First value of 'range' ('{r[0]}') cannot be larger than the second ('{r[1]}')"
            )
        return r


class CircularSweeping(BaseSettings):
    """
    A sweep of two attributes on a circle around their current values.

    The first point lies at `(0, +radius)` relative to the current values, the
    others follow evenly spaced around the circle. Both attributes are offset by
    the same radius, so they must share units (e.g. `stigmator_x`/`stigmator_y`).
    """

    model_config = ConfigDict(use_enum_values=True)

    dimension: Literal["2d"] = Field(
        default="2d",
        description="Sweep two attributes on a circle around their current values.",
    )
    target_attribute_x: TargetAttribute = Field(
        default="stigmator_x",
        description="Attribute on the x-axis of the swept plane.",
    )
    target_attribute_y: TargetAttribute = Field(
        default="stigmator_y",
        description="Attribute on the y-axis of the swept plane.",
    )
    strategy: SweepingStrategySettings = Field(
        default_factory=BasicStrategy,
        description="Sweeping strategy to use.",
    )
    radius: Annotated[float, Field(gt=0.0)] = Field(
        default=0.1,
        description="Radius of the circle, in units shared by both attributes.",
    )
    steps: Annotated[int, Field(ge=3)] = Field(
        default=8,
        description="Number of points on the circle.",
    )
    cycles: Annotated[int, Field(gt=0)] = Field(
        default=1,
        description="Number of sweeping repeats.",
    )

    @model_validator(mode="after")
    def _check_distinct_attributes(self) -> Self:
        if self.target_attribute_x == self.target_attribute_y:
            raise ValueError(
                f"Both axes of a 2D sweep target '{self.target_attribute_x}'; "
                "choose two different attributes."
            )
        return self


SweepingSettings = Annotated[
    LinearSweeping | CircularSweeping,
    Field(discriminator="dimension"),
]
"""Settings of a sweep, one- or two-dimensional."""

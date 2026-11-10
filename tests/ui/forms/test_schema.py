# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The Qt-free description of settings classes: field infos, choices and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any

import pytest
from gui_testkit.sample_settings import (
    PATTERN_TYPES,
    VENDOR_PROPERTIES,
    CircleShape,
    Color,
    KitchenSink,
    Leaf,
    Optionals,
    PointShape,
    SquareShape,
)
from pydantic import Field, ValidationError

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import WidgetType
from fibsem_maestro.ui.schema.choices import ChoiceContext, choices_for, has_choices
from fibsem_maestro.ui.schema.constraints import Bound, NumericBounds
from fibsem_maestro.ui.schema.field_type import (
    DataclassType,
    DiscriminatedUnionType,
    FloatType,
    IntType,
    ListType,
    LiteralType,
)
from fibsem_maestro.ui.schema.schema import get_field_infos
from fibsem_maestro.ui.schema.validation import (
    REQUIRED_MESSAGE,
    detail_message,
    short_message,
    validate_assignment,
    validation_messages,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.schema.field_info import FieldInfo


def _info(cls: type, name: str) -> FieldInfo:
    (found,) = [fi for fi in get_field_infos(cls) if fi.name == name]
    return found


def test_field_infos_follow_declaration_order() -> None:
    assert [fi.name for fi in get_field_infos(KitchenSink)] == list(
        KitchenSink.model_fields
    )


def test_field_info_carries_label_description_unit_and_hint() -> None:
    width = _info(KitchenSink, "width")
    window = _info(KitchenSink, "window")

    assert (width.label, width.description) == ("width", "How wide.")
    assert width.unit is not None and width.unit.suffix == "nm"
    assert width.hint is None
    assert _info(KitchenSink, "property_name").label == "property name"
    assert window.hint is not None and window.hint.widget is WidgetType.RANGE_PAIR


def test_field_info_resolves_plain_and_factory_defaults() -> None:
    plain = _info(KitchenSink, "count").default
    factory = _info(KitchenSink, "numbers").default
    none_default = _info(Optionals, "limit").default

    assert plain is not None and plain.value == 3
    assert factory is not None and factory.value == [1, 2]
    assert none_default is not None and none_default.value is None


@pytest.mark.parametrize(
    ("cls", "name", "bounds"),
    [
        (
            KitchenSink,
            "count",
            NumericBounds(Bound(0.0, exclusive=False), Bound(100.0, exclusive=False)),
        ),
        (KitchenSink, "width", NumericBounds(Bound(0.0, exclusive=True), None)),
        (KitchenSink, "title", None),
        (Optionals, "limit", NumericBounds(Bound(0.0, exclusive=True), None)),
    ],
)
def test_field_info_extracts_numeric_bounds(
    cls: type, name: str, bounds: NumericBounds | None
) -> None:
    assert _info(cls, name).bounds == bounds


def test_optional_fields_are_unwrapped() -> None:
    limit = _info(Optionals, "limit")
    extra = _info(Optionals, "extra")

    assert limit.optional and isinstance(limit.type, FloatType)
    assert extra.optional and extra.type == DataclassType(model=Leaf)
    assert not _info(KitchenSink, "leaf").optional


def test_field_types_are_classified() -> None:
    assert isinstance(_info(KitchenSink, "count").type, IntType)
    assert _info(KitchenSink, "size").type == LiteralType(choices=("small", "large"))
    assert _info(KitchenSink, "numbers").type == ListType(item=IntType())
    assert _info(KitchenSink, "leaves").type == ListType(item=DataclassType(model=Leaf))


def test_union_with_nested_arm_is_described_with_its_follower() -> None:
    mode = _info(KitchenSink, "mode").type
    assert isinstance(mode, DiscriminatedUnionType)

    manual, engine = mode.variants

    assert mode.discriminator_key == "kind"
    assert (manual.discriminator_value, manual.nested) == ("manual", None)
    assert (engine.discriminator_value, engine.label) == ("engine", "engine mode")
    assert engine.nested is not None
    assert engine.nested.follows == "engine"
    assert [v.discriminator_value for v in engine.nested.variants] == [
        "fast",
        "precise",
    ]


@pytest.fixture
def choice_context(linked_action: Action) -> ChoiceContext:
    return ChoiceContext(
        actions=(linked_action,),
        manufacturer_property_names=VENDOR_PROPERTIES,
        pattern_types=PATTERN_TYPES,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("color", list(Color)),
        ("size", ["small", "large"]),
        ("engine", ["fast", "precise", "turbo"]),
        ("property_name", ["focus", "contrast", *VENDOR_PROPERTIES]),
        ("tags", ["a", "b", "c"]),
        ("property_names", ["focus", "contrast", *VENDOR_PROPERTIES]),
        ("pattern", list(PATTERN_TYPES)),
    ],
)
def test_choices_for_lists_what_the_editor_offers(
    choice_context: ChoiceContext, name: str, expected: list[Any]
) -> None:
    fi = _info(KitchenSink, name)

    assert has_choices(fi)
    assert choices_for(fi, choice_context) == expected


def test_choices_for_action_selector_lists_actions_of_allowed_types(
    choice_context: ChoiceContext, linked_action: Action
) -> None:
    fi = _info(KitchenSink, "linked")

    assert choices_for(fi, choice_context) == [linked_action.name]
    assert choices_for(fi, ChoiceContext()) == []


def test_choices_for_action_selector_skips_other_types(
    add_action: Callable[..., Action],
) -> None:
    fi = _info(KitchenSink, "linked")
    assert fi.hint is not None
    kinds = list(ACTION_REGISTRY)
    actions = (add_action(kinds[0], "a"), add_action(kinds[-1], "b"))
    only = type(actions[0])
    restricted = replace(fi, hint=replace(fi.hint, action_type_filter=[only]))

    names = choices_for(restricted, ChoiceContext(actions=actions))

    assert names == [a.name for a in actions if isinstance(a, only)]


def test_choices_for_action_selector_skips_excluded_types_and_their_subclasses(
    add_action: Callable[..., Action],
) -> None:
    fi = _info(KitchenSink, "linked")
    assert fi.hint is not None
    milling, imaging = add_action("milling", "a"), add_action("imaging", "b")
    excluding = replace(fi, hint=replace(fi.hint, action_type_exclude=("milling",)))
    subclass = type("SubMilling", (type(milling),), {})
    milling.__class__ = subclass

    names = choices_for(excluding, ChoiceContext(actions=(milling, imaging)))

    assert names == [imaging.name]


@pytest.mark.parametrize("name", ["count", "leaf", "area", "notes", "window"])
def test_choices_for_rejects_fields_without_choices(
    choice_context: ChoiceContext, name: str
) -> None:
    fi = _info(KitchenSink, name)

    assert not has_choices(fi)
    with pytest.raises(ValueError, match=name):
        choices_for(fi, choice_context)


def test_validate_assignment_leaves_the_owner_and_its_hooks_alone() -> None:
    owner = KitchenSink()
    changes: list[object] = []
    owner.on_change(changes.append)

    value = validate_assignment(owner, "numbers", (4, 5))

    assert value == [4, 5]
    assert owner == KitchenSink()
    assert changes == []


def test_validate_assignment_runs_field_constraints() -> None:
    with pytest.raises(ValidationError) as raised:
        validate_assignment(KitchenSink(), "width", 0.0)

    assert validation_messages(raised.value) == ["Input should be greater than 0"]


def test_validate_assignment_on_a_plain_dataclass_returns_the_value() -> None:
    @dataclass
    class Plain:
        count: int = 0

    assert validate_assignment(Plain(), "count", "not checked") == "not checked"


def test_none_in_a_required_field_reads_a_value_is_required() -> None:
    with pytest.raises(ValidationError) as raised:
        validate_assignment(KitchenSink(), "engine", None)

    (detail,) = raised.value.errors()
    assert detail_message(detail) == REQUIRED_MESSAGE == "A value is required."


def test_validator_messages_lose_their_error_type_prefix() -> None:
    with pytest.raises(ValidationError) as raised:
        SquareShape(side=1.0, margin=2.0)

    assert validation_messages(raised.value) == [
        "The margin must be smaller than the side."
    ]
    assert short_message("Assertion failed, low is high") == "low is high"
    assert short_message("Field required") == "Field required"


type _Count = Annotated[int, Field(gt=0)]
type _Counts = list[int]
type _AnyShape = Annotated[
    CircleShape | SquareShape | PointShape, Field(discriminator="kind")
]


class _Aliased(BaseSettings):
    count: _Count = 1
    maybe: _Count | None = None
    counts: _Counts = Field(default_factory=list)
    shape: _AnyShape = Field(default_factory=CircleShape)


def test_type_statement_aliases_are_classified_like_their_values() -> None:
    count = _info(_Aliased, "count")
    maybe = _info(_Aliased, "maybe")

    assert isinstance(count.type, IntType) and count.bounds is not None
    assert maybe.optional and isinstance(maybe.type, IntType)
    assert _info(_Aliased, "counts").type == ListType(item=IntType())
    assert isinstance(_info(_Aliased, "shape").type, DiscriminatedUnionType)

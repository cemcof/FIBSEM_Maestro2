# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
`SettingsAudit`: the errors fresh settings forms would show, for every action.

Breakable fields of the registered actions are discovered through `ui/schema`, so
these tests follow the settings as they change. Behaviour no registered action has
today (an optional link, list items) uses a small synthetic action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest
from gui_testkit.situations import link_field, link_pairs, link_to
from pydantic import BaseModel, Field

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action_context.file import FileActionContext
from fibsem_maestro.adjust.adjust_props import AdjustProps
from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.settings.adjust_props_settings import AdjustPropsSettings
from fibsem_maestro.settings.form_utils import FormHint, WidgetType
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.ui.app.audit import RULE_WHERE, AuditIssue, SettingsAudit
from fibsem_maestro.ui.app.catalogs import MicroscopeCatalogs
from fibsem_maestro.ui.schema.choices import (
    ChoiceContext,
    choices_for,
    unavailable_action_message,
)
from fibsem_maestro.ui.schema.field_type import StrType
from fibsem_maestro.ui.schema.labels import field_name_to_label
from fibsem_maestro.ui.schema.schema import get_field_infos

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.schema.field_info import FieldInfo

KEY = "adjust_props"
REQUIRED = "A value is required."


def registered_fields(
    accept: Callable[[FieldInfo, FormHint], bool],
) -> list[tuple[str, str]]:
    """`(action type key, field name)` of every top-level hinted settings field accepted."""
    return [
        (key, fi.name)
        for key in ACTION_REGISTRY
        for fi in get_field_infos(ACTION_REGISTRY.get(key).settings_cls())
        if fi.hint is not None and accept(fi, fi.hint)
    ]


def field_info(action: Action, name: str) -> FieldInfo:
    return next(fi for fi in get_field_infos(type(action.settings)) if fi.name == name)


def problems(audit: SettingsAudit, action: Action) -> set[tuple[str, str]]:
    return {(issue.where, issue.message) for issue in audit.issues_of(action)}


@pytest.fixture
def audit(session: WorkflowSession) -> SettingsAudit:
    return SettingsAudit(session, MicroscopeCatalogs.query(session.microscope))


@pytest.mark.parametrize(
    ("key", "name"),
    registered_fields(
        lambda fi, hint: (
            not fi.optional
            and isinstance(fi.type, StrType)
            and hint.widget
            in (
                WidgetType.DROPDOWN,
                WidgetType.ACTION_SELECTOR,
            )
        )
    ),
)
def test_audit_reports_required_choice_not_offered(
    session: WorkflowSession,
    audit: SettingsAudit,
    add_action: Callable[..., Action],
    key: str,
    name: str,
) -> None:
    # one action of every type, so that every link has something to choose from
    for other in ACTION_REGISTRY:
        add_action(other, f"{other} target")
    action = add_action(key, "action")
    fi = field_info(action, name)
    catalogs = MicroscopeCatalogs.query(session.microscope)
    offered = choices_for(
        fi,
        ChoiceContext(
            session.actions,
            catalogs.manufacturer_property_names,
            catalogs.pattern_types,
        ),
    )
    issue = (fi.label, REQUIRED)

    setattr(action.settings, name, offered[0])
    assert issue not in problems(audit, action)

    setattr(action.settings, name, "not among the choices")
    assert issue in problems(audit, action)


@pytest.mark.parametrize(
    ("key", "name"),
    registered_fields(lambda _, hint: hint.widget is WidgetType.PATTERN_TYPE_SELECTOR),
)
def test_audit_reports_pattern_type_the_microscope_lacks(
    session: WorkflowSession,
    audit: SettingsAudit,
    add_action: Callable[..., Action],
    key: str,
    name: str,
) -> None:
    action = add_action(key, "action")
    catalogs = MicroscopeCatalogs.query(session.microscope)
    setattr(action.settings, name, catalogs.pattern_types[0])
    issue = (field_name_to_label(name), REQUIRED)
    assert issue not in problems(audit, action)

    # V8: a microscope whose pattern types could not be read offers none
    assert issue in problems(SettingsAudit(session, MicroscopeCatalogs()), action)


@pytest.mark.parametrize(
    ("key", "name"), registered_fields(lambda _, hint: bool(hint.min_areas))
)
def test_audit_reports_too_few_areas(
    audit: SettingsAudit, add_action: Callable[..., Action], key: str, name: str
) -> None:
    action = add_action(key, "action")
    fi = field_info(action, name)
    minimum = fi.hint.min_areas if fi.hint is not None else None
    assert minimum
    issue = (fi.label, f"At least {minimum} area(s) required.")

    setattr(action.settings, name, [RelativeArea.full()] * minimum)
    assert issue not in problems(audit, action)

    setattr(action.settings, name, [RelativeArea.full()] * (minimum - 1))
    assert issue in problems(audit, action)


@pytest.mark.parametrize(("linker_key", "field", "target_key"), link_pairs())
def test_audit_reports_link_to_removed_action(
    session: WorkflowSession,
    audit: SettingsAudit,
    add_action: Callable[..., Action],
    linker_key: str,
    field: str,
    target_key: str,
) -> None:
    target = add_action(target_key, "target")
    linker = add_action(linker_key, "linker")
    link_to(linker.settings, field, "target")
    many = link_field(linker.settings, field).many
    message = unavailable_action_message("target") if many else REQUIRED
    issue = (field_name_to_label(field), message)
    assert issue not in problems(audit, linker)

    session.remove_action(target)

    assert issue in problems(audit, linker)


def test_audit_reports_nothing_for_plain_action(
    audit: SettingsAudit, add_action: Callable[..., Action]
) -> None:
    add_action(KEY, "action")
    assert audit.issues() == []


class Letter(BaseModel):
    letter: Annotated[
        str, FormHint(widget=WidgetType.DROPDOWN, choices=lambda: ["a", "b"])
    ]


class LinkingSettings(AdjustPropsSettings):
    optional_link: Annotated[
        str | None,
        FormHint(widget=WidgetType.ACTION_SELECTOR, action_type_filter=[AdjustProps]),
    ] = Field(default=None, description="An optional link.")
    many_links: Annotated[
        list[str],
        FormHint(
            widget=WidgetType.MULTI_ACTION_SELECTOR, action_type_filter=[AdjustProps]
        ),
    ] = Field(default_factory=list, description="Several links.")
    letters: list[Letter] = Field(
        default_factory=list, description="Choices in a list."
    )


class LinkingAction(AdjustProps):
    """An adjust-props action with links and a list of models (never registered)."""

    @classmethod
    def settings_cls(cls) -> type[LinkingSettings]:
        return LinkingSettings


def append_linking(session: WorkflowSession) -> Action:
    """
    Append a synthetic linking action to the workflow.

    Append it last: the session cannot persist a workflow holding an unregistered type.
    """
    action = LinkingAction(
        "linking",
        session.microscope,
        LinkingSettings(),
        FileActionContext(session.workflow_dir / "linking", "linking"),
        session.workflow.actions,
    )
    session.workflow.actions.append(action)
    return action


@pytest.fixture
def linking(session: WorkflowSession) -> Action:
    return append_linking(session)


def test_audit_reports_optional_link_to_unknown_action(
    audit: SettingsAudit, linking: Action
) -> None:
    assert problems(audit, linking) == set()

    linking.settings.optional_link = (
        "linking"  # itself: an available adjust-props action
    )
    assert problems(audit, linking) == set()

    linking.settings.optional_link = "gone"
    assert problems(audit, linking) == {
        ("optional link", "'gone' is not an available action.")
    }


def test_audit_reports_each_unknown_name_of_a_multiple_link_once(
    audit: SettingsAudit, linking: Action
) -> None:
    linking.settings.many_links = ["linking"]
    assert problems(audit, linking) == set()

    linking.settings.many_links = ["gone", "linking", "lost", "gone"]
    assert problems(audit, linking) == {
        (
            "many links",
            "'gone' is not an available action.\n'lost' is not an available action.",
        )
    }


def test_audit_walks_list_items(audit: SettingsAudit, linking: Action) -> None:
    linking.settings.letters = [Letter(letter="a"), Letter(letter="c")]
    assert problems(audit, linking) == {("letters → item 2 → letter", REQUIRED)}


def test_audit_issue_text_names_action(linking: Action) -> None:
    issue = AuditIssue(linking, "letters → item 2 → letter", REQUIRED)
    assert issue.text() == "'linking': letters → item 2 → letter: A value is required."


@pytest.fixture
def family(add_action: Callable[..., Action]) -> tuple[Action, Action]:
    return add_action(KEY, "parent"), add_action(KEY, "child")


def make_rule(
    session: WorkflowSession,
    parent: Action,
    dependents: list[str],
    names: PropertyNames,
) -> None:
    rule = session.add_rule(parent)
    session.set_rule_dependents(rule, dependents)
    session.set_rule_properties(rule, names)


def test_audit_reports_rule_property_not_collected(
    session: WorkflowSession, audit: SettingsAudit, family: tuple[Action, Action]
) -> None:
    parent, _ = family
    make_rule(
        session, parent, ["child"], PropertyNames(electron_beam=["working_distance"])
    )

    assert audit.issues() == [
        AuditIssue(
            parent,
            RULE_WHERE,
            "'working_distance' is not among the properties 'parent' collects.",
        )
    ]
    parent.settings.properties_to_collect.electron_beam = ["working_distance"]
    assert audit.issues() == []


def test_audit_reports_unknown_manufacturer_property(
    session: WorkflowSession, audit: SettingsAudit, family: tuple[Action, Action]
) -> None:
    parent, _ = family
    known = MicroscopeCatalogs.query(session.microscope).manufacturer_property_names[0]
    names = PropertyNames(microscope=[known, "vendor.unknown"])
    parent.settings.properties_to_collect = names
    make_rule(session, parent, ["child"], names)

    assert [issue.text() for issue in audit.issues()] == [
        "'parent': propagation rule: 'vendor.unknown' is not available on this microscope."
    ]


def test_audit_skips_incomplete_rules(
    session: WorkflowSession, audit: SettingsAudit, family: tuple[Action, Action]
) -> None:
    parent, _ = family
    make_rule(session, parent, [], PropertyNames(microscope=["stage_position"]))
    make_rule(session, parent, ["child"], PropertyNames())
    assert audit.issues() == []


def test_audit_lists_settings_before_rules(
    session: WorkflowSession, audit: SettingsAudit, family: tuple[Action, Action]
) -> None:
    parent, _ = family
    make_rule(session, parent, ["child"], PropertyNames(microscope=["stage_position"]))
    linking = append_linking(session)
    linking.settings.optional_link = "gone"

    assert [(issue.action, issue.where) for issue in audit.issues()] == [
        (linking, "optional link"),
        (parent, RULE_WHERE),
    ]
    # issues_of covers settings only
    assert audit.issues_of(parent) == []

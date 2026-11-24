# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
What a freshly built settings form would reject, for every action, and rules that would fail.

A live settings object is always valid for its model, so a fresh form shows an
error only where an editor cannot represent the live value: a choice field
whose value is not among the editor's choices (the editor then reads None),
and an area selector holding fewer areas than its minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from fibsem_maestro.settings.form_utils import WidgetType
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.ui.schema.choices import (
    ChoiceContext,
    choices_for,
    unavailable_action_message,
)
from fibsem_maestro.ui.schema.field_type import (
    DataclassType,
    DiscriminatedUnionType,
    ListType,
)
from fibsem_maestro.ui.schema.labels import field_name_to_label
from fibsem_maestro.ui.schema.schema import get_field_infos
from fibsem_maestro.ui.schema.validation import (
    validate_assignment,
    validation_messages,
)

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.catalogs import MicroscopeCatalogs
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.schema.field_info import FieldInfo
    from fibsem_maestro.workflow.propagations import PropagationRule

RULE_WHERE = "propagation rule"
"""`AuditIssue.where` of every rule issue."""

_SINGLE_CHOICE_WIDGETS: frozenset[WidgetType] = frozenset(
    {
        WidgetType.DROPDOWN,
        WidgetType.PATTERN_TYPE_SELECTOR,
        WidgetType.ACTION_SELECTOR,
    }
)
"""Hinted editors that read None when the value is not among their choices."""

_PATH_SEPARATOR = " → "


@dataclass(frozen=True)
class AuditIssue:
    """
    One problem the audit found.

    Attributes:
        action: The action whose settings (or whose rule, as parent) have the problem.
        where: The field path, e.g. "pattern type" or "criterion -> mask", or
            "propagation rule".
        message: Exactly the text the form shows, e.g. "A value is required.".
    """

    action: Action
    where: str
    message: str

    def text(self) -> str:
        """The issue as one line: `'<action>': <where>: <message>`."""
        return f"'{self.action.name}': {self.where}: {self.message}"


class SettingsAudit:
    """
    Finds, without building any widget, the errors fresh settings forms would show.

    Also reports propagation rules that would fail at run time. The walk follows
    the form's structure (nested objects, the selected variant of each union,
    present optional values, list items) and uses the same `choices_for` as the
    editors.

    Args:
        session: The workflow session (actions and rules).
        catalogs: The microscope's choice lists.
    """

    def __init__(self, session: WorkflowSession, catalogs: MicroscopeCatalogs) -> None:
        self._session = session
        self._catalogs = catalogs

    def issues(self) -> list[AuditIssue]:
        """
        Every issue: settings issues in action order, then rule issues in rule order.

        Returns:
            The issues; empty if there are none.
        """
        ctx = self._choice_context()
        issues: list[AuditIssue] = []
        for action in self._session.actions:
            issues.extend(self._settings_issues(action, ctx))
        for rule in self._session.workflow.propagations.rules:
            issues.extend(self._rule_issues(rule, ctx))
        return issues

    def issues_of(self, action: Action) -> list[AuditIssue]:
        """
        The settings issues of one action (not its rules).

        Args:
            action: The action.

        Returns:
            The issues of its settings, in field order.
        """
        return self._settings_issues(action, self._choice_context())

    def _choice_context(self) -> ChoiceContext:
        return ChoiceContext(
            actions=self._session.actions,
            manufacturer_property_names=self._catalogs.manufacturer_property_names,
            pattern_types=self._catalogs.pattern_types,
        )

    def _settings_issues(self, action: Action, ctx: ChoiceContext) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        for path, message in _walk(action.settings, (), ctx):
            issues.append(AuditIssue(action, _PATH_SEPARATOR.join(path), message))
        return issues

    def _rule_issues(
        self, rule: PropagationRule, ctx: ChoiceContext
    ) -> list[AuditIssue]:
        # the workflow skips incomplete rules, so they cannot fail
        if not rule.is_complete():
            return []
        parent = self._session.action_named(rule.parent_name)
        if parent is None:
            return []

        issues: list[AuditIssue] = []
        collected = parent.props_to_collect
        for fi in get_field_infos(PropertyNames):
            available = choices_for(fi, ctx)
            for prop in getattr(rule.props_to_propagate, fi.name):
                if prop not in getattr(collected, fi.name):
                    issues.append(
                        AuditIssue(
                            parent,
                            RULE_WHERE,
                            f"'{prop}' is not among the properties "
                            f"'{parent.name}' collects.",
                        )
                    )

                # an unknown manufacturer property is never collected
                if prop not in available:
                    issues.append(
                        AuditIssue(
                            parent,
                            RULE_WHERE,
                            f"'{prop}' is not available on this microscope.",
                        )
                    )
        return issues


def _walk(
    obj: object, path: tuple[str, ...], ctx: ChoiceContext
) -> list[tuple[tuple[str, ...], str]]:
    """
    Find the problems a fresh form would show for an object and everything nested in it.

    Args:
        obj: A settings object (pydantic model or dataclass).
        path: Labels of the fields leading to `obj`.
        ctx: The choice context.

    Returns:
        `(field path, message)` pairs, in field order.
    """
    problems: list[tuple[tuple[str, ...], str]] = []
    for fi in get_field_infos(type(obj)):
        value = getattr(obj, fi.name, None)
        field_path = (*path, field_name_to_label(fi.name))

        # a hinted field is a leaf editor, whatever its type
        if fi.hint is not None:
            if (message := _hinted_problem(obj, fi, value, ctx)) is not None:
                problems.append((field_path, message))
            continue

        # an absent optional value shows an unticked box, which is never an error
        if value is None:
            continue

        match fi.type:
            case DataclassType() | DiscriminatedUnionType():
                # a union value is an instance of its selected variant
                problems.extend(_walk(value, field_path, ctx))
            case ListType(item=DataclassType() | DiscriminatedUnionType()):
                for index, item in enumerate(value):
                    problems.extend(
                        _walk(item, (*field_path, f"item {index + 1}"), ctx)
                    )
            case _:
                pass
    return problems


def _hinted_problem(
    owner: object, fi: FieldInfo, value: Any, ctx: ChoiceContext
) -> str | None:
    """
    The error a fresh form shows on a hinted field, if any.

    Args:
        owner: The object holding the field.
        fi: The field.
        value: Its live value.
        ctx: The choice context.

    Returns:
        The message, or None.
    """
    hint = fi.hint
    assert hint is not None

    if hint.widget is WidgetType.AREA_SELECT:
        minimum = hint.min_areas
        if minimum and value is not None and len(value) < minimum:
            return f"At least {minimum} area(s) required."
        return None

    # the editor keeps the names it does not offer, marked, until they are deselected
    if hint.widget is WidgetType.MULTI_ACTION_SELECTOR:
        available = choices_for(fi, ctx)
        missing = [name for name in dict.fromkeys(value or ()) if name not in available]
        return "\n".join(unavailable_action_message(name) for name in missing) or None

    if hint.widget not in _SINGLE_CHOICE_WIDGETS or value in choices_for(fi, ctx):
        return None

    # the editor shows nothing selected and reads None, which the committer validates
    if (message := _none_rejected(owner, fi.name)) is not None:
        return message

    # an optional link shows "(none)" without an error, but a link to a removed
    # action still blocks the run
    if hint.widget is WidgetType.ACTION_SELECTOR and value is not None:
        return unavailable_action_message(value)

    return None


def _none_rejected(owner: object, name: str) -> str | None:
    """
    The form's message for assigning None to a field, as the committer validates it.

    Args:
        owner: The object holding the field.
        name: The field name.

    Returns:
        The messages (joined by newlines, as an editor shows them), or None if
        None is accepted or the owner does not validate (plain dataclass).
    """
    try:
        validate_assignment(owner, name, None)
    except ValidationError as e:
        return "\n".join(dict.fromkeys(validation_messages(e)))

    return None

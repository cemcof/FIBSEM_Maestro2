# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Whether the workflow may run (start, resume, continue), and why not."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from fibsem_maestro.ui.app.run_state import AppState
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.app.audit import AuditIssue, SettingsAudit
    from fibsem_maestro.ui.app.form_validity import FormValidity
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.app.run_controller import RunController
    from fibsem_maestro.ui.app.session import WorkflowSession


@dataclass(frozen=True)
class ReadinessReport:
    """
    Every reason the workflow cannot run.

    Attributes:
        reasons: Human-readable reasons, in a fixed order; empty iff ready.
    """

    reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """True if nothing prevents running."""
        return not self.reasons


def assess_readiness(
    actions: Sequence[Action],
    workflow_slice: int,
    preparation_issues: Sequence[str],
    completion_reasons: Sequence[str],
    audit: Sequence[AuditIssue],
    forms_with_errors: Collection[Action],
    paused: bool,
) -> ReadinessReport:
    """
    Collect every reason the workflow cannot run, in this order.

    1. "The workflow has no actions."
    2. "All 9999 slices have been acquired." once the workflow slice passed
       `Workflow.MAX_SLICE`.
    3. "'<name>' is not prepared for slice <n>." for every action without
       stored props (n is the action's own slice counter).
    4. Every preparation issue, as the workflow formats it ("'<name>': <issue>").
    5. Every completion reason ("'<name>': <reason>"), unless `paused`: a paused
       run is inside a slice, and the workflow checks completion only at the
       next slice boundary, so Resume must stay possible.
    6. Every audit issue (`AuditIssue.text()`), in the audit's order.
    7. "'<name>': the settings form shows an invalid value." for every action
       in `forms_with_errors` without an audit issue.

    Args:
        actions: The workflow's actions, in order.
        workflow_slice: The workflow's slice counter.
        preparation_issues: `Workflow.preparation_issues()`.
        completion_reasons: `Workflow.completion_reasons()`.
        audit: `SettingsAudit.issues()`.
        forms_with_errors: `FormValidity.owners_with_errors()`.
        paused: Whether the run state is PAUSED.

    Returns:
        The report.
    """
    reasons: list[str] = []
    if not actions:
        reasons.append("The workflow has no actions.")
    if workflow_slice > Workflow.MAX_SLICE:
        reasons.append(f"All {Workflow.MAX_SLICE} slices have been acquired.")
    reasons.extend(
        f"'{action.name}' is not prepared for slice {action.ctx.slice}."
        for action in actions
        if not action.has_stored_properties()
    )
    reasons.extend(preparation_issues)
    if not paused:
        reasons.extend(completion_reasons)
    reasons.extend(issue.text() for issue in audit)

    audited = {id(issue.action) for issue in audit}
    reasons.extend(
        f"'{action.name}': the settings form shows an invalid value."
        for action in actions
        if action in forms_with_errors and id(action) not in audited
    )
    return ReadinessReport(tuple(reasons))


def readiness_tooltip(reasons: Sequence[str], heading: str, limit: int = 12) -> str:
    """
    Format reasons for a tooltip.

    Args:
        reasons: The reasons, in order.
        heading: First line, e.g. "Cannot start:".
        limit: Most reasons listed; the rest are summarised in one more line.

    Returns:
        "" if there are no reasons; otherwise `heading`, then one "• <reason>"
        line per reason up to `limit`, then "• … and <k> more" if any were left out.
    """
    if not reasons:
        return ""
    lines = [heading, *(f"• {reason}" for reason in reasons[:limit])]
    if len(reasons) > limit:
        lines.append(f"• … and {len(reasons) - limit} more")
    return "\n".join(lines)


class Readiness(QObject):
    """
    The current `ReadinessReport`, recomputed whenever an input may have changed.

    Args:
        session: The workflow session.
        run: The run controller.
        tasks: The microscope task runner (a failed Prepare can leave props on
            disk without any session signal).
        validity: Which open forms show errors.
        audit: The settings audit.
        parent: Optional Qt parent.
    """

    changed = pyqtSignal(object)
    """The new `ReadinessReport`, emitted only when it differs."""

    def __init__(
        self,
        session: WorkflowSession,
        run: RunController,
        tasks: MicroscopeTasks,
        validity: FormValidity,
        audit: SettingsAudit,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._run = run
        self._validity = validity
        self._audit = audit
        self._report = self._assess()

        for signal in (
            session.structure_changed,
            session.action_settings_changed,
            session.action_props_changed,
            session.action_state_changed,
            session.rules_changed,
            session.workflow_reset,
            session.workflow_replaced,
            validity.changed,
            run.state_changed,
            run.slice_finished,
            run.action_finished,
            tasks.busy_changed,
        ):
            signal.connect(self._recompute)

    @property
    def report(self) -> ReadinessReport:
        """The current report."""
        return self._report

    def _assess(self) -> ReadinessReport:
        workflow = self._session.workflow
        return assess_readiness(
            actions=self._session.actions,
            workflow_slice=workflow.ctx.slice,
            preparation_issues=workflow.preparation_issues(),
            completion_reasons=workflow.completion_reasons(),
            audit=self._audit.issues(),
            forms_with_errors=self._validity.owners_with_errors(),
            paused=self._run.state is AppState.PAUSED,
        )

    def _recompute(self, *_: object) -> None:
        report = self._assess()
        if report != self._report:
            self._report = report
            self.changed.emit(report)

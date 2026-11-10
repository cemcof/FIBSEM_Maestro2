# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The workflow being edited: its actions, rules, directories and their persistence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action_context.file import FileActionContext
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.ui.app.errors import (
    ActionRenameError,
    InvalidActionNameError,
    WorkflowImportError,
    WorkflowResetError,
)
from fibsem_maestro.ui.app.links import retarget_links, validate_action_name
from fibsem_maestro.workflow.actions import Actions
from fibsem_maestro.workflow.propagations import PropagationRule, Propagations
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.properties.global_properties import GlobalProperties
    from fibsem_maestro.settings.base_settings import BaseSettings
    from fibsem_maestro.ui.app.catalogs import MicroscopeCatalogs
    from fibsem_maestro.ui.app.files import WorkflowFiles


@dataclass(frozen=True)
class SessionSeed:
    """
    Everything the main window is built from, produced by the start-up dialog.

    Attributes:
        workflow: The workflow, freshly created or loaded for resumption.
        workflow_dir: Directory holding the workflow's `workflow/` and action directories.
        catalogs: What the connected microscope offers, read once after connecting.
    """

    workflow: Workflow
    workflow_dir: Path
    catalogs: MicroscopeCatalogs


class WorkflowSession(QObject):
    """
    The workflow being edited: its actions, rules, directories and their persistence.

    This is the only code in `ui` that mutates `Workflow.actions`,
    `Workflow.propagations`, an action's name or directory, or writes the
    workflow's `state.yaml`, `microscope_settings.yaml` or an action's
    `settings.yaml`. Every command keeps these invariants and persists before
    it emits:

    - I1: action names are non-empty, unique ignoring case with spaces equal to
      underscores, and never "workflow".
    - I2: an action's directory is `workflow_dir / name.replace(" ", "_")`.
    - I3: rules follow renames; removing an action deletes the rules it is the
      parent of and drops it from every rule's dependents; a rule left without
      dependents stays.
    - I4: renaming retargets every link (action-selector field) holding the old name.
    - I5: after a move, the moved actions' slices are clamped between their neighbours.
    - I6: new actions start at the workflow's slice and are persisted at once.
    - I7: structure and rule changes write the workflow's `state.yaml`.
    - I8: persist first, then emit fine-grained signals, then `structure_changed`
      or `rules_changed`.
    - I9: the workflow's `Actions` list is mutated in place, never replaced by a copy.
    - I10: rules are edited in place, never replaced.
    - I11: other services never cache `workflow`; it is replaced by `new_workflow`.

    Args:
        workflow: The workflow, freshly created or loaded for resumption.
        workflow_dir: Directory holding the workflow's `workflow/` and action directories.
        files: Filesystem operations.
        parent: Optional Qt parent.
    """

    structure_changed = pyqtSignal()
    """Actions were added, removed, reordered or renamed."""

    action_removed = pyqtSignal(object)
    """An `Action` left the workflow (also every action replaced by an import)."""

    action_renamed = pyqtSignal(object, str)
    """`(action, old_name)` after an action was renamed."""

    action_settings_changed = pyqtSignal(object)
    """An `Action` whose settings were committed by a form or had a link retargeted."""

    action_props_changed = pyqtSignal(object)
    """An `Action` whose props for its current slice were written."""

    action_state_changed = pyqtSignal(object)
    """An `Action` whose internal state was reset by the user."""

    rules_changed = pyqtSignal()
    """Propagation rules were added, removed or edited."""

    workflow_replaced = pyqtSignal()
    """"New workflow": a different workflow and directory."""

    workflow_reset = pyqtSignal()
    """"Reset workflow": the same actions, back to slice 0."""

    def __init__(
        self,
        workflow: Workflow,
        workflow_dir: Path,
        files: WorkflowFiles,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._workflow = workflow
        self._workflow_dir = workflow_dir
        self._files = files

        # store microscope settings and initial workflow state
        self._write_microscope_settings()
        self._write_workflow_state()

    @property
    def workflow(self) -> Workflow:
        """The workflow being edited; replaced by `new_workflow`, so never cache it."""
        return self._workflow

    @property
    def workflow_dir(self) -> Path:
        """Directory holding the workflow's `workflow/` and action directories."""
        return self._workflow_dir

    @property
    def microscope(self) -> Microscope:
        """The connected microscope."""
        return self._workflow.microscope

    @property
    def actions(self) -> tuple[Action, ...]:
        """The actions, in workflow order."""
        return tuple(self._workflow.actions)

    @property
    def slice(self) -> int:
        """The workflow's slice counter."""
        return self._workflow.ctx.slice

    def action_named(self, name: str) -> Action | None:
        """
        Find an action by its exact name.

        Args:
            name: The action name.

        Returns:
            The action, or None if no action has that name.
        """
        return next((a for a in self._workflow.actions if a.name == name), None)

    def rules_of(self, parent: Action) -> list[PropagationRule]:
        """
        The propagation rules whose parent is `parent`, in rule order.

        Args:
            parent: The parent action.

        Returns:
            The rule objects themselves (edited in place by the session).
        """
        return [
            rule
            for rule in self._workflow.propagations.rules
            if rule.parent_name == parent.name
        ]

    def dependents_of(self, parent: Action) -> list[str]:
        """
        Unique dependent names over all rules of `parent`, in first-seen order.

        Args:
            parent: The parent action.

        Returns:
            The names of the actions `parent` propagates to.
        """
        return list(
            dict.fromkeys(
                name for rule in self.rules_of(parent) for name in rule.dependent_names
            )
        )

    def text_logger(self) -> TextLogger:
        """The workflow context's text logger."""
        return self._workflow.ctx.text_logger

    def add_action(self, type_key: str, name: str) -> Action:
        """
        Append a new action with default settings at the workflow's slice.

        Args:
            type_key: Registry key of the action type.
            name: Name of the new action.

        Returns:
            The new action.

        Raises:
            InvalidActionNameError: If the name is empty, taken or reserved.
        """
        if (error := validate_action_name(name, self._names())) is not None:
            raise InvalidActionNameError(error)

        action = self._build_action(type_key, name, self._workflow.ctx.slice)
        self._workflow.actions.append(action)
        self._write_workflow_state()
        self.structure_changed.emit()
        return action

    def duplicate_action(self, action: Action) -> Action:
        """
        Append a copy of an action, with deep-copied settings and its outgoing rules.

        The copy is named after the original with " 2", " 3", ... appended.

        Args:
            action: The action to duplicate.

        Returns:
            The new action.
        """
        new_name = self._generate_unique_name(action.name)
        type_key = ACTION_REGISTRY.key_of(type(action))
        new_action = self._build_action(
            type_key,
            new_name,
            self._workflow.ctx.slice,
            copy.deepcopy(action.settings),
        )
        self._workflow.actions.append(new_action)
        self._workflow.propagations.copy_rules_from(action.name, new_name)
        self._write_workflow_state()
        self.rules_changed.emit()
        self.structure_changed.emit()
        return new_action

    def remove_action(self, action: Action) -> OSError | None:
        """
        Remove an action, its rules and its directory.

        Rules whose parent is the action are deleted; the action is dropped
        from every other rule's dependents (a rule left without dependents
        stays). Links to the action in other settings are kept and become broken.

        Args:
            action: The action to remove.

        Returns:
            The error if the action's directory could not be deleted, else None.
            The action is removed from the workflow either way.
        """
        self._workflow.actions.remove(action)

        rules = self._workflow.propagations.rules
        rules[:] = [rule for rule in rules if rule.parent_name != action.name]
        for rule in rules:
            if action.name in rule.dependent_names:
                rule.dependent_names[:] = [
                    name for name in rule.dependent_names if name != action.name
                ]

        error: OSError | None = None
        try:
            self._files.delete_tree(self._workflow_dir / action.name_with_underscores)
        except OSError as e:
            error = e

        self._write_workflow_state()
        self.action_removed.emit(action)
        self.rules_changed.emit()
        self.structure_changed.emit()
        return error

    def rename_action(self, action: Action, new_name: str) -> None:
        """
        Rename an action, its directory and loggers, its rules and every link to it.

        An unchanged name does nothing.

        Args:
            action: The action to rename.
            new_name: The new name.

        Raises:
            InvalidActionNameError: If the name is empty, taken or reserved.
            ActionRenameError: If the directory could not be moved (chained to
                the OSError; its message is the OSError's). Nothing is changed.
            OSError: If writing a file failed after the action was renamed. The
                rename stands, and every signal is still emitted.
        """
        old_name = action.name
        if new_name == old_name:
            return

        existing = {a.name for a in self._workflow.actions if a is not action}
        if (error := validate_action_name(new_name, existing)) is not None:
            raise InvalidActionNameError(error)

        # release the log files before the action directory is moved
        self._files.close_logs()
        try:
            # renames the action, its directory and its loggers
            action.rename(new_name)
        except OSError as e:
            raise ActionRenameError(str(e)) from e

        retargeted: list[Action] = []
        try:
            self._workflow.propagations.rename_action(old_name, new_name)
            for other in self._workflow.actions:
                if retarget_links(other.settings, old_name, new_name):
                    retargeted.append(other)
                    self._write_action_settings(other)
            self._write_workflow_state()
        finally:
            # the action is renamed whatever failed later; announce what changed
            for other in retargeted:
                self.action_settings_changed.emit(other)
            self.action_renamed.emit(action, old_name)
            self.rules_changed.emit()
            self.structure_changed.emit()

    def move_actions(self, start: int, count: int, destination: int) -> None:
        """
        Move a contiguous block of actions.

        The moved actions' slices are then clamped between their new
        neighbours; an action whose slice changed gets its `state.yaml` and
        `settings.yaml` written for the new slice.

        Args:
            start: Index of the first moved action before the move.
            count: Number of contiguous actions moved.
            destination: Index of the first moved action after the move.

        Raises:
            IndexError: If the block or the destination is out of range.
        """
        actions = self._workflow.actions
        if (
            count < 1
            or start < 0
            or start + count > len(actions)
            or not 0 <= destination <= len(actions) - count
        ):
            raise IndexError(
                f"cannot move {count} action(s) from {start} to {destination} "
                f"in a list of {len(actions)}"
            )

        block = list(actions[start : start + count])
        rest = [a for a in actions if not any(a is moved for moved in block)]
        reordered = rest[:destination] + block + rest[destination:]

        # the Actions object is shared by every action (I9), so reorder it in place
        actions.clear()
        actions.extend(reordered)

        for action in self._reconcile_slices(destination, count):
            # the action now addresses another slice, which must hold its files
            action.ctx.state_store.write("state.yaml", action.state)
            self._write_action_settings(action)
        self._write_workflow_state()
        self.structure_changed.emit()

    def add_rule(self, parent: Action) -> PropagationRule:
        """
        Add an empty propagation rule (no dependents, no properties) to `parent`.

        Args:
            parent: The parent action.

        Returns:
            The new rule.

        Raises:
            ValueError: If the action takes no part in propagation.
        """
        if not parent.PROPAGATION_RULES_ALLOWED:
            raise ValueError(f"'{parent.name}' takes no part in propagation")

        rule = PropagationRule(
            parent_name=parent.name,
            dependent_names=[],
            props_to_propagate=PropertyNames(),
        )
        self._workflow.propagations.rules.append(rule)
        self._write_workflow_state()
        self.rules_changed.emit()
        return rule

    def set_rule_dependents(self, rule: PropagationRule, names: Sequence[str]) -> None:
        """
        Replace the dependents of a rule, in place.

        Args:
            rule: A rule of this workflow.
            names: Names of the dependent actions.

        Raises:
            ValueError: If the rule is not one of this workflow's rules.

        Raises:
            ValueError: If the rule is not one of this workflow's rules, or a
                dependent takes no part in propagation.
        """
        self._require_rule(rule)
        if excluded := [
            action.name
            for action in self._workflow.actions
            if action.name in names and not action.PROPAGATION_RULES_ALLOWED
        ]:
            raise ValueError(
                f"{', '.join(repr(n) for n in excluded)} take no part in propagation"
            )

        rule.dependent_names[:] = list(names)
        self._write_workflow_state()
        self.rules_changed.emit()

    def set_rule_properties(self, rule: PropagationRule, names: PropertyNames) -> None:
        """
        Replace the properties a rule propagates.

        Args:
            rule: A rule of this workflow.
            names: The properties to propagate.

        Raises:
            ValueError: If the rule is not one of this workflow's rules.
        """
        self._require_rule(rule)
        rule.props_to_propagate = names
        self._write_workflow_state()
        self.rules_changed.emit()

    def remove_rule(self, rule: PropagationRule) -> None:
        """
        Delete a rule.

        Args:
            rule: A rule of this workflow; matched by identity, since equal rules may exist.

        Raises:
            ValueError: If the rule is not one of this workflow's rules.
        """
        self._require_rule(rule)
        rules = self._workflow.propagations.rules
        rules[:] = [r for r in rules if r is not rule]
        self._write_workflow_state()
        self.rules_changed.emit()

    def commit_settings(self, action: Action) -> None:
        """
        A form wrote a field of the action's settings (at any depth): persist and announce it.

        Args:
            action: The action whose settings changed.
        """
        self._write_action_settings(action)
        self.action_settings_changed.emit(action)

    def write_properties(self, action: Action, props: GlobalProperties) -> None:
        """
        Store props for the action's current slice (the properties dialog).

        Every action may then discard internal state the new props invalidate
        (`Action.replace_properties`), e.g. a step-mode autofunction run. The
        action's `settings.yaml` is also written for that slice.

        Args:
            action: The action.
            props: The properties to store.

        Raises:
            Exception: Whatever the props or settings store raises; nothing is
                emitted then.
        """
        action.replace_properties(props)
        self._write_action_settings(action)
        self.action_props_changed.emit(action)

    def actions_losing_state_to(self, action: Action) -> list[Action]:
        """
        The actions whose internal state new properties of `action` would discard.

        Args:
            action: The action whose properties would be stored (Prepare or
                edited properties).

        Returns:
            The actions, in workflow order; `action` itself included if it is
            one of them.
        """
        return [a for a in self.actions if a.properties_would_discard_state(action)]

    def reset_action_state(self, action: Action) -> None:
        """
        Discard the action's internal state, and announce it.

        The default state is stored for the action's current slice
        (`Action.reset_state`); settings, props and the slice are kept.

        Args:
            action: The action.

        Raises:
            Exception: Whatever the state store raises; nothing is emitted then.
        """
        action.reset_state()
        self.action_state_changed.emit(action)

    def props_changed(self, action: Action) -> None:
        """
        The backend wrote the action's props (Prepare succeeded): persist and announce it.

        The action's `settings.yaml` is written for its current slice first.

        Args:
            action: The action.
        """
        self._write_action_settings(action)
        self.action_props_changed.emit(action)

    def commit_microscope_settings(self) -> None:
        """Persist the microscope settings (`microscope_settings.yaml`)."""
        self._write_microscope_settings()

    def new_workflow(self, directory: Path) -> None:
        """
        Replace the workflow by an empty one in `directory`, keeping the microscope and notifier.

        The directory must already be cleared by the caller.

        Args:
            directory: The new workflow directory.

        Raises:
            OSError: If the new workflow's files could not be written; the
                session is unchanged then.
        """
        workflow = Workflow(
            microscope=self._workflow.microscope,
            actions=Actions(),
            propagations=Propagations(),
            context=FileActionContext(
                action_dir=directory / "workflow", name="workflow"
            ),
            notifier=self._workflow.notifier,
        )
        _write_workflow_state(workflow)
        _write_microscope_settings(workflow)

        self._workflow = workflow
        self._workflow_dir = directory
        self.workflow_replaced.emit()
        self.structure_changed.emit()
        self.rules_changed.emit()

    def import_workflow(self, source: Path) -> None:
        """
        Replace the actions and rules by those of another workflow directory.

        The source is loaded first, so a failure there changes nothing. Then the
        current action directories are deleted and the imported actions written
        to this workflow's directory.

        Args:
            source: Directory of the workflow to import.

        Raises:
            WorkflowImportError: If loading, deleting or writing failed (chained
                to the cause; its message is the cause's).
        """
        try:
            imported = Workflow.import_from_dir(
                source,
                self._workflow.microscope,
                self._workflow_dir,
                self._workflow.notifier,
            )
        except Exception as e:
            raise WorkflowImportError(str(e)) from e

        removed = list(self._workflow.actions)
        try:
            # delete all action directories, keeping the workflow directory
            self._files.clear_except(self._workflow_dir, "workflow")

            # store action states and settings
            for action in imported.actions:
                action.ctx.state_store.write("state.yaml", action.state)
                action.ctx.settings_store.write("settings.yaml", action.settings)
        except Exception as e:
            raise WorkflowImportError(str(e)) from e

        # the imported actions were built with `imported.actions`, so that very
        # object becomes the workflow's list (I9)
        self._workflow.actions = imported.actions
        self._workflow.propagations = imported.propagations
        self._write_workflow_state()

        for action in removed:
            self.action_removed.emit(action)
        self.structure_changed.emit()
        self.rules_changed.emit()

    def reset(self) -> None:
        """
        Return every action and the workflow to slice 0, deleting all slice data.

        Raises:
            WorkflowResetError: If deleting or writing failed (chained to the
                cause; its message is the cause's).
        """
        try:
            self._files.close_logs()

            for action in self._workflow.actions:
                action.reset()
                # delete the action's directory content
                if (action_dir := action.ctx.path_to_dir) is not None:
                    self._clear_or_raise(action_dir)

                # then write the action's settings and state
                action.ctx.state_store.write("state.yaml", action.state)
                self._write_action_settings(action)

            self._workflow.ctx.reset()
            if (workflow_dir := self._workflow.ctx.path_to_dir) is not None:
                self._clear_or_raise(workflow_dir)

                # then write the workflow's state and microscope settings
                self._write_workflow_state()
                self._write_microscope_settings()
        except Exception as e:
            raise WorkflowResetError(str(e)) from e

        self.workflow_reset.emit()

    def _names(self) -> set[str]:
        return {action.name for action in self._workflow.actions}

    def _build_action(
        self,
        type_key: str,
        name: str,
        slice_number: int,
        init_settings: BaseSettings | None = None,
    ) -> Action:
        """Construct a new Action from the registry with default settings."""
        action_cls = ACTION_REGISTRY.get(type_key)
        settings_cls = action_cls.settings_cls()
        settings = init_settings or settings_cls()

        action = action_cls(
            name=name,
            microscope=self._workflow.microscope,
            settings=settings,
            ctx=FileActionContext(
                action_dir=self._workflow_dir / name.replace(" ", "_"),
                name=name.replace(" ", "_"),
                slice=slice_number,
            ),
            actions=self._workflow.actions,
        )

        # persisted at once (I6); an action's state changes only while it runs,
        # so this state.yaml stays current until then
        action.ctx.state_store.write("state.yaml", action.state)
        action.ctx.settings_store.write("settings.yaml", action.settings)

        return action

    def _generate_unique_name(self, base_name: str) -> str:
        """
        A valid new action name (I1): `base_name`, else `base_name` + " 2", " 3", ...

        Args:
            base_name: The name to start from.

        Returns:
            The first candidate `validate_action_name` accepts.
        """
        existing = self._names()
        candidate = base_name
        i = 2
        while validate_action_name(candidate, existing) is not None:
            candidate = f"{base_name} {i}"
            i += 1
        return candidate

    def _reconcile_slices(self, first_row: int, count: int) -> list[Action]:
        """
        Clamp the slices of the moved actions to satisfy their new neighbours.

        Args:
            first_row: Index of the first moved action in its new position.
            count: Number of contiguous actions that were moved.

        Returns:
            The actions whose slice was changed.
        """
        actions = self._workflow.actions
        last_row = first_row + count - 1

        above = actions[first_row - 1].ctx.slice if first_row > 0 else None
        below = actions[last_row + 1].ctx.slice if last_row + 1 < len(actions) else None

        changed: list[Action] = []
        for action in actions[first_row : last_row + 1]:
            new_slice = clamp_slice(action.ctx.slice, above, below)
            if new_slice != action.ctx.slice:
                action.ctx.set_slice(new_slice)
                changed.append(action)
        return changed

    def _require_rule(self, rule: PropagationRule) -> None:
        # rules compare by value, so membership is checked by identity
        if not any(r is rule for r in self._workflow.propagations.rules):
            raise ValueError(
                f"the rule of '{rule.parent_name}' is not a rule of this workflow"
            )

    def _clear_or_raise(self, directory: Path) -> None:
        if failures := self._files.clear(directory):
            _, error = failures[0]
            raise error

    def _write_workflow_state(self) -> None:
        _write_workflow_state(self._workflow)

    def _write_microscope_settings(self) -> None:
        _write_microscope_settings(self._workflow)

    @staticmethod
    def _write_action_settings(action: Action) -> None:
        action.ctx.settings_store.write("settings.yaml", action.settings)


def _write_workflow_state(workflow: Workflow) -> None:
    """Write the workflow's `state.yaml` for its current slice."""
    workflow.ctx.state_store.write("state.yaml", workflow.state)


def _write_microscope_settings(workflow: Workflow) -> None:
    """Write the workflow's `microscope_settings.yaml` for its current slice."""
    workflow.ctx.settings_store.write(
        "microscope_settings.yaml", workflow.microscope.settings
    )


def clamp_slice(current: int, above: int | None, below: int | None) -> int:
    """
    Clamp a slice index to the range permitted by its neighbours.

    Slice indices are non-increasing down the action list: an action must
    have a slice no greater than the action above it and no less than the
    action below it.

    Args:
        current: The moved action's slice index.
        above: The slice index of the action directly above, or None if the
            moved action is now first.
        below: The slice index of the action directly below, or None if the
            moved action is now last.

    Returns:
        The clamped slice index, unchanged if it already satisfies both bounds.
    """
    clamped = current
    if below is not None:
        clamped = max(clamped, below)
    if above is not None:
        clamped = min(clamped, above)
    return clamped

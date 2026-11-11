# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, TypeVar, final

from pydantic import BaseModel

from fibsem_maestro.action.defaults import ActionDefaults
from fibsem_maestro.action.outcome import CarriedOver, Produced, StepOutcome
from fibsem_maestro.action.settings_protocol import ActionSettingsLike
from fibsem_maestro.action.state import ActionState
from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.format import ImageFormat
from fibsem_maestro.logging.logging import with_logging_context
from fibsem_maestro.settings.dynamic_default import unavailable_defaults

if TYPE_CHECKING:
    from collections.abc import Generator

    from fibsem_maestro.action_context.action_context import ActionContext
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.properties.beam_properties import BeamProperties
    from fibsem_maestro.properties.global_properties import GlobalProperties
    from fibsem_maestro.settings.dynamic_default import DefaultContext
    from fibsem_maestro.settings.property_names import PropertyNames
    from fibsem_maestro.settings.reactive import ChangePath
    from fibsem_maestro.store.props.props_store import PropsStore
    from fibsem_maestro.workflow.actions import Actions


class LinkedActions(ABC):
    """Base class for action links."""


TSettings = TypeVar("TSettings", bound=ActionSettingsLike)
TState = TypeVar("TState", bound=ActionState)


class Action[TSettings: ActionSettingsLike, TState: ActionState](ABC):
    """
    A step of the acquisition workflow, executed at most once per slice.

    The base class owns the per-slice lifecycle in `execute`: gating, applying
    the stored properties, committing the outcome to the next slice, and
    logging. Subclasses implement `_run_step` and may refine `should_execute`.

    Terminology:
        carry over: copy this action's own properties to its next slice.
        propagate: send this action's produced properties to other actions.
            The workflow does this, and only for `Produced` outcomes.

    Args:
        name: Human-readable identifier, unique within the workflow.
        microscope: The microscope instance.
        settings: The action's settings.
        ctx: Slice navigation, stores and loggers for this action.
        actions: All actions in the workflow, for resolving linked actions.
    """

    _STATE_FIELDS: ClassVar[frozenset[str]] = frozenset()
    """Top-level settings fields whose change invalidates the internal state of the action."""

    PROPAGATION_RULES_ALLOWED: ClassVar[bool] = True
    """
    False for actions that may be neither the parent nor a dependent of a propagation rule.

    Such an action's properties must not spread to other actions, nor be
    overwritten from them. Explicit patches (`Produced.patches`) are not rules
    and are unaffected.
    """

    def __init__(
        self,
        name: str,
        microscope: Microscope,
        settings: TSettings,
        ctx: ActionContext,
        actions: Actions,
    ) -> None:
        self._name = name
        self._microscope = microscope
        self._settings = settings
        self._ctx = ctx
        self._actions = actions

        if self._STATE_FIELDS:
            self._settings.on_change_at(self._on_settings_changed_at)

    @classmethod
    def settings_cls(cls) -> type[TSettings]:
        """
        Return the class used for the action's settings.

        Raises:
            NotImplementedError: If the subclass does not override this.
        """
        raise NotImplementedError(f"settings_type not implemented for {cls.__name__}")

    @classmethod
    def state_cls(cls) -> type[TState]:
        """
        Return the class used for the action's state.

        Raises:
            NotImplementedError: If the subclass does not override this.
        """
        raise NotImplementedError(f"state_cls not implemented for {cls.__name__}")

    @classmethod
    def state_fields(cls) -> frozenset[str]:
        """
        Names of the top-level settings fields whose change resets the internal state.

        Returns:
            The field names; empty if no settings change resets the state.
        """
        return cls._STATE_FIELDS

    @final
    @with_logging_context
    def prepare(self) -> None:
        """
        Prepare the action to execute its current slice from the microscope as it is now.

        Captures the action's properties from the microscope and stores them
        for the current slice, then runs the action-specific `_prepare`
        hook, then stores the resulting state for the current slice. The
        properties are captured first, so they reflect the user's setup and
        not whatever the hook leaves on the microscope.

        Finally, every action of the workflow may discard internal state the
        new properties invalidate (`_on_properties_replaced`).
        """
        if (bt := self.beam_type) is not None:
            self._microscope.set_beam(bt)

        # capture the scan parameters as set for full-frame scanning; the
        # microscope GUI's own reduced area is ignored, as during execution
        self._clear_scanning_area()
        self.write_properties(self.collect_properties())
        self._prepare()
        self._ctx.state_store.write("state.yaml", self.state)
        self._announce_properties_replaced()

    def _prepare(self) -> None:
        """
        Action-specific preparation, run after the properties are captured.

        Does nothing by default.
        """

    def preparation_issues(self) -> list[str]:
        """
        Describe what must still be prepared before the action can execute.

        The GUI evaluates this after every edit, so implementations must be
        cheap: no microscope access and no image I/O (checking that a file
        exists is fine).

        Returns:
            Human-readable problems; empty if the action is ready. The default
            reports the fields following a default that cannot be computed
            (`_default_issues`).
        """
        return self._default_issues()

    def _default_issues(self) -> list[str]:
        """
        Report the settings fields following a dynamic default that cannot be computed.

        Returns:
            One "'<dotted path>': <reason>" line per such field.
        """
        if not isinstance(self._settings, BaseModel):
            return []
        return [
            f"'{path}': {reason}"
            for path, reason in unavailable_defaults(
                self._settings, self.default_context()
            )
        ]

    def default_context(self) -> DefaultContext:
        """
        What the dynamic defaults of the action's settings are computed from now.

        Returns:
            The action's settings, the workflow's actions and the action's
            stored properties (`stored_beam_properties`).
        """
        return ActionDefaults(
            self._settings, self._actions, self.stored_beam_properties
        )

    def stored_beam_properties(self) -> BeamProperties | None:
        """
        The properties stored for the current slice, on the action's beam.

        These are the properties the action applies before it runs, so the
        defaults computed from them match what it works with. Read from the
        store without logging, since the forms ask after every change.

        Returns:
            The beam's stored properties, or None if the action works on no
            beam, or none are stored or they cannot be read.
        """
        beam = self.beam_type
        if beam is None or not self.has_stored_properties():
            return None
        try:
            props = self._ctx.props_store.read("props.yaml")
        # a validation error of the props file is a ValueError
        except OSError, ValueError:
            return None
        return getattr(props, props.get_properties_attr_name(beam))

    def completion_reason(self) -> str | None:
        """
        Tell whether this action considers the acquisition complete.

        The workflow checks this before every slice and finishes, without an
        error, when any action reports a reason. The GUI evaluates this after
        every edit, so implementations must be cheap: no microscope access and
        no image I/O.

        Returns:
            A human-readable reason, e.g. "the milling area is used up; change it
            to continue", or None if the acquisition may continue. The default
            is None.
        """
        return None

    @final
    @with_logging_context
    def execute(self) -> StepOutcome:
        """
        Run one step of the action for the current slice.

        If `should_execute` declines, the current properties are carried over
        to the next slice and the microscope is not touched. Otherwise the
        stored properties are applied, `_run_step` runs, and its outcome is
        committed: produced properties are written to the next slice, and a
        carry-over copies the current ones unchanged.

        Subclasses customise `should_execute` and `_run_step`, not this method.

        Returns:
            The step outcome. The workflow propagates only `Produced` outcomes.
        """
        if not self.should_execute():
            return self.skip("execution conditions not met")

        self._ctx.text_logger.info(
            f"Started '{self.name}' for slice {self._ctx.slice}."
        )
        with self._scanning_area_applied():
            self.read_and_set_properties()
            outcome = self._run_step()

        match outcome:
            case Produced(props=props):
                self.write_properties(props, self._ctx.props_store.next)
            case CarriedOver():
                self.carry_over_to_next()

        self._ctx.text_logger.info(
            f"Completed '{self.name}' for slice {self._ctx.slice}."
        )
        return outcome

    def should_execute(self) -> bool:
        """
        Decide whether the action runs on the current slice.

        The default follows `settings.execution_frequency`. Overrides may add
        conditions but must not change microscope state.

        Returns:
            `True` if the action should run for this slice.
        """
        return self._settings.execution_frequency.matches(self._ctx.slice)

    def skip(self, reason: str) -> CarriedOver:
        """
        Skip the current slice, carrying the current properties over unchanged.

        Args:
            reason: Why the slice is skipped, for logging.

        Returns:
            The carry-over outcome.
        """
        self._ctx.text_logger.info(
            f"Skipping '{self.name}' for slice {self._ctx.slice}: {reason}."
        )
        self.carry_over_to_next()
        return CarriedOver(reason)

    def carry_over_to_next(self) -> None:
        """
        Copy this action's current properties to its next slice unchanged.

        Override to carry over additional files or notify collaborators; call `super()`.
        """
        self.write_properties(self.read_properties(), self._ctx.props_store.next)

    def rename(self, name: str) -> None:
        """
        Rename the action, together with its directory and its loggers.

        Args:
            name: The new action name.

        Raises:
            OSError: If the action directory cannot be moved, e.g.
                `FileExistsError` if a directory for the new name already exists.
        """
        self._ctx.rename(name.replace(" ", "_"))
        self._name = name

    @abstractmethod
    def _run_step(self) -> StepOutcome:
        """
        Perform the action's work for the current slice.

        Called by `execute` after the stored properties have been applied.

        Returns:
            `Produced` with the properties for the next slice, or `CarriedOver`
            if the step finished without new properties (e.g. mid-sweep).
        """

    @abstractmethod
    def set_state(self, state: TState) -> None:
        """
        Restore the action's internal state.

        Args:
            state: The state to restore.
        """

    @abstractmethod
    def test(self) -> None:
        """Run the action once outside the acquisition loop, for diagnostics."""

    def wait_for_background_threads(self) -> None:
        """
        Wait for all background threads spawned by this action to complete.

        The default does nothing; override if the action spawns threads.
        """

    def reset(self) -> None:
        """Reset the action to slice 0 and its default state."""
        self.ctx.reset()
        self.reset_state()

    def reset_state(self) -> None:
        """
        Discard the internal state, keeping the slice position.
        """
        self.set_state(self.state_cls()())
        self._ctx.state_store.write("state.yaml", self.state)

    @with_logging_context
    def initialize_first_slice(self) -> None:
        """
        Initialize the action for the first slice.

        Copies props, settings and state from slice 0 to slice 1, then
        advances the action context. Assumes these files exist for slice 0.
        """
        self.ctx.text_logger.debug(f"Initializing action {self.name}.")
        self.ctx.props_store.copy_to("props.yaml", self.ctx.props_store.next)
        self.ctx.settings_store.copy_to("settings.yaml", self.ctx.settings_store.next)
        self.ctx.state_store.copy_to("state.yaml", self.ctx.state_store.next)
        self.ctx.advance()

    @property
    def name(self) -> str:
        """Human-readable identifier of the action."""
        return self._name

    @property
    def name_with_underscores(self) -> str:
        """Name of the action with spaces replaced by underscores."""
        return self._name.replace(" ", "_")

    @property
    def ctx(self) -> ActionContext:
        """Slice navigation, stores and loggers for this action."""
        return self._ctx

    @property
    def microscope(self) -> Microscope:
        """The microscope instance."""
        return self._microscope

    @property
    def settings(self) -> TSettings:
        """The action's settings."""
        return self._settings

    @property
    def beam_type(self) -> BeamType | None:
        """Beam the action works with, or `None` for both or neither."""
        return self._settings.beam_type

    @property
    def props_to_collect(self) -> PropertyNames:
        """Names of the microscope properties this action collects."""
        return self._settings.properties_to_collect

    @property
    @abstractmethod
    def state(self) -> TState:
        """Internal state of the action."""

    @property
    def scanning_area(self) -> RelativeArea | None:
        """
        Area this action scans on its beam, relative to the full frame.

        Applied for the whole execution. The default is None: the action scans
        the full frame, whatever reduced area the microscope's own GUI has set.
        Actions with a configurable area override this.
        """
        return None

    def _beam_control(self) -> BeamControl | None:
        """
        Return the control of the action's beam.

        Returns:
            The electron or ion beam, or None for actions using both or neither.
        """
        match self.beam_type:
            case BeamType.ELECTRON:
                return self._microscope.electron_beam
            case BeamType.ION:
                return self._microscope.ion_beam
            case _:
                return None

    @with_logging_context
    def read_properties(self, store: PropsStore | None = None) -> GlobalProperties:
        """
        Read this action's stored microscope properties.

        Args:
            store: Store to read from; defaults to the current slice.

        Returns:
            The stored properties.
        """
        store = store or self._ctx.props_store
        self._ctx.text_logger.debug(f"Reading microscope properties for {self.name}.")
        return store.read("props.yaml")

    @with_logging_context
    def read_and_set_properties(self, store: PropsStore | None = None) -> None:
        """
        Read this action's stored properties and apply them to the microscope.

        Selects the action's beam first, if it has one.

        Args:
            store: Store to read from; defaults to the current slice.
        """
        props = self.read_properties(store)
        if self.beam_type is not None:
            self._microscope.set_beam(self.beam_type)
        self._ctx.text_logger.debug(f"Setting microscope properties for {self.name}.")
        self._microscope.set_properties(props, beam=self.beam_type)

    @with_logging_context
    def collect_properties(self) -> GlobalProperties:
        """
        Collect this action's properties from the microscope.

        Returns:
            The collected properties.
        """
        self._ctx.text_logger.debug(
            f"Collecting microscope properties for {self.name}."
        )
        return self._microscope.collect_properties(self.props_to_collect)

    @with_logging_context
    def write_properties(
        self, props: GlobalProperties, store: PropsStore | None = None
    ) -> None:
        """
        Write microscope properties to this action's store.

        Args:
            props: The properties to write.
            store: Store to write to; defaults to the current slice.
        """
        store = store or self._ctx.props_store
        self._ctx.text_logger.debug(f"Writing microscope properties for {self.name}.")
        store.write("props.yaml", props)

    def replace_properties(self, props: GlobalProperties) -> None:
        """
        Store properties for the current slice from outside the acquisition, e.g. edited ones.

        Unlike `write_properties`, which the workflow uses for the properties
        it produces, propagates and carries over, every action of the
        workflow may then discard internal state the new properties
        invalidate (`_on_properties_replaced`).

        Args:
            props: The properties to store.
        """
        self.write_properties(props)
        self._announce_properties_replaced()

    def _announce_properties_replaced(self) -> None:
        """Let every action of the workflow, and this one, react to its new properties."""
        others = [action for action in self._actions if action is not self]
        for action in (self, *others):
            action._on_properties_replaced(self)

    def properties_would_discard_state(self, of: Action) -> bool:
        """
        Tell whether new properties of an action would discard this action's internal state.

        Answers what `_on_properties_replaced` would do, without doing it, so
        that the user can be asked first. False by default.

        Args:
            of: The action whose properties would be replaced (Prepare or edited
                properties); may be this action.

        Returns:
            True if storing new properties of `of` would discard state that is
            not the default one.
        """
        _ = of
        return False

    def _on_properties_replaced(self, of: Action) -> None:
        """
        Discard internal state that new properties of an action invalidate.

        Called on every action of the workflow, including `of` itself, after
        properties were stored for `of` from outside the acquisition: by
        Prepare, or as edited properties (`replace_properties`). Never for
        the properties the workflow writes itself. Does nothing by default:
        e.g. the position of a milling does not depend on any properties.

        Args:
            of: The action whose properties were replaced.
        """
        _ = of

    def has_stored_properties(self) -> bool:
        """
        Check whether properties are stored for the current slice.

        Returns:
            `True` if the props file for the current slice exists.
        """
        return self._ctx.props_store.exists("props.yaml")

    def _on_settings_changed_at(self, path: ChangePath) -> None:
        """
        Reset the internal state if a change affected a state-defining field.

        Nothing happens while the state is still the default one, e.g. before
        the workflow has run, since there is nothing to discard.

        Args:
            path: Location of the change relative to the settings. An empty
                path means the settings changed as a whole, which may include
                a state-defining field, so the state is reset then as well.
        """
        if path and path[0] not in self._STATE_FIELDS:
            return
        if self.state == self.state_cls()():
            return

        what = f"'{path[0]}'" if path else "the settings"
        self._ctx.text_logger.info(
            f"Resetting internal state of '{self.name}': {what} changed."
        )
        self.reset_state()

    def _save_result_frame(self, label: str) -> None:
        """
        Grab a frame in the current microscope state and save it with this action's slice.

        For checking what an action achieved, e.g. the focus after an
        autofunction. The frame is written next to this action's frames for the
        current slice, named by `label`, and does not go through the frame
        store, so it never blocks or replaces a slice frame. A failure is
        logged rather than raised, since the frame is only diagnostic.

        Args:
            label: Name part identifying the frame, e.g. `"after_autofunction"`.
        """
        if self._ctx.path_to_dir is None:
            self._ctx.text_logger.warning(
                f"Frame after '{self.name}' not saved: the action has no directory."
            )
            return

        try:
            image = self._microscope.beam.grab_frame()
            output_path = self._ctx.current_view.path() / f"{label}.tif"
            image.save(output_path, ImageFormat.TIF)
        except Exception as e:
            self._ctx.text_logger.warning(
                f"Could not save the frame after '{self.name}': {e}"
            )
            return

        self._ctx.text_logger.info(
            f"Saved the frame after '{self.name}' as {output_path}."
        )

    @contextmanager
    def _scanning_area_applied(self) -> Generator[None]:
        """
        Scan the action's scanning area for the enclosed block, then the full frame again.

        Must be entered before the stored properties are applied: Autoscript
        keeps separate scan parameters for reduced-area scanning, so the
        properties have to land in the mode that will actually scan. Actions
        without a beam of their own leave the scanning area alone.

        Yields:
            None. The full frame is restored even if the block raises.
        """
        beam = self._beam_control()
        if beam is None:
            yield
            return

        beam.scanning_area = self.scanning_area or RelativeArea.full()
        try:
            yield
        finally:
            try:
                self._clear_scanning_area()
            except Exception as e:
                self._ctx.text_logger.warning(
                    f"Could not clear the scanning area of '{self.name}': {e}"
                )

    def _clear_scanning_area(self) -> None:
        """Make the action's beam scan the full frame, discarding any reduced area."""
        if (beam := self._beam_control()) is not None:
            beam.scanning_area = RelativeArea.full()

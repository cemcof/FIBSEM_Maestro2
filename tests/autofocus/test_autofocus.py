# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import dataclasses
import threading
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any

import pytest

from fibsem_maestro.action.registry import ACTION_REGISTRY
from fibsem_maestro.action_context.file import FileActionContext
from fibsem_maestro.autofocus import (
    BasicSweepingStrategy,
    InterleavedSweepingStrategy,
)
from fibsem_maestro.autofocus.autofocus import Autofocus, AutofocusState
from fibsem_maestro.autofocus.jobs_manager import JobsManager
from fibsem_maestro.autofocus.result import AutofocusResult
from fibsem_maestro.autofocus.sweep_space import LinearSweepSpace
from fibsem_maestro.autofocus.sweep_step import SweepPoint, SweepStep
from fibsem_maestro.core.frequency import Every
from fibsem_maestro.logging.text.contextual import ContextualTextLogger
from fibsem_maestro.microscope.microscope import Microscope
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.autofocus_settings import BasicMode as BasicModeSettings
from fibsem_maestro.settings.autofocus_settings import StepMode as StepModeSettings
from fibsem_maestro.settings.microscope_settings import MicroscopeSettings
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.settings.sweeping_settings import (
    InterleavedStrategy as InterleavedStrategySettings,
)
from fibsem_maestro.settings.sweeping_settings import (
    LinearSweeping,
)
from fibsem_maestro.workflow.actions import Actions
from fibsem_maestro.workflow.propagations import Propagations
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.autofocus.evaluation import BasicEvaluation

SPACE = LinearSweepSpace("working_distance", (-2.0, 2.0), 3)
BASE: SweepPoint = (0.0,)


def _interleaved_results(
    values: list[SweepPoint], sharpness: dict[float, float]
) -> list[AutofocusResult]:
    return [
        AutofocusResult(sharpness[v[0]], SweepStep(0, v, i))
        for i, v in enumerate(values)
    ]


def test_interleaved_picks_candidate_sharper_than_base() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.0))
    values = strategy.generate(SPACE, BASE, 0)
    # -2 is much worse than the base, +2 is the only value sharper than the base
    sharpness = {0.0: 10.0, -2.0: 2.0, 2.0: 11.0}
    assert strategy.evaluate(_interleaved_results(values, sharpness)).best == (2.0,)


def test_interleaved_keeps_base_when_nothing_is_sharper() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.0))
    values = strategy.generate(SPACE, BASE, 0)
    sharpness = {0.0: 10.0, -2.0: 2.0, 2.0: 9.0}
    assert strategy.evaluate(_interleaved_results(values, sharpness)).best == (0.0,)


def test_interleaved_tolerates_a_missing_result() -> None:
    strategy = InterleavedSweepingStrategy(InterleavedStrategySettings(min_diff=0.0))
    values = strategy.generate(SPACE, BASE, 0)
    sharpness = {0.0: 10.0, -2.0: 2.0, 2.0: 11.0}
    results = _interleaved_results(values, sharpness)
    # the job scoring the second base frame failed and was dropped
    del results[2]
    assert strategy.evaluate(results).best == (2.0,)


class _LateCallbackFuture:
    """A future whose done-callbacks run some time after its result is ready."""

    def __init__(self, delay_s: float) -> None:
        self._done = threading.Event()
        self._delay_s = delay_s
        self._result: Any = None
        self._callbacks: list[Callable[[Any], None]] = []

    def run(self, fn: Callable[[], Any]) -> None:
        self._result = fn()
        self._done.set()  # waiters in result() wake up here ...
        threading.Timer(self._delay_s, self._fire).start()  # ... callbacks later

    def _fire(self) -> None:
        for callback in self._callbacks:
            callback(self)

    def add_done_callback(self, fn: Callable[[Any], None]) -> None:
        self._callbacks.append(fn)

    def result(self, timeout: float | None = None) -> Any:
        self._done.wait(timeout)
        return self._result

    def done(self) -> bool:
        return self._done.is_set()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._done.wait(timeout)
        return None


class LateCallbackExecutor(Executor):
    """Models a thread switch between a future completing and its callbacks."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        future = _LateCallbackFuture(delay_s=0.2)
        threading.Thread(target=future.run, args=(lambda: fn(*args, **kwargs),)).start()
        return future


def test_jobs_manager_returns_every_finished_result() -> None:
    jobs = JobsManager(executor=LateCallbackExecutor())
    for i in range(3):
        jobs.submit(lambda i=i: AutofocusResult(float(i), SweepStep(0, (float(i),), i)))

    results = jobs.wait_and_collect()

    assert sorted(r.sharpness for r in results) == [0.0, 1.0, 2.0]


def test_jobs_manager_drops_failed_jobs() -> None:
    jobs = JobsManager()

    def fail() -> AutofocusResult:
        raise RuntimeError("criterion failed")

    jobs.submit(fail)
    jobs.submit(lambda: AutofocusResult(1.0, SweepStep(0, (1.0,), 1)))
    assert [r.sharpness for r in jobs.wait_and_collect()] == [1.0]


# steps of every sweep; a single sweep keeps its run in progress for this many slices
STEPS = 3

# triggers on slice 1 only, within the length of these tests
ONCE = Every(n=1000)


@pytest.fixture
def workflow(tmp_path: Path) -> Workflow:
    ctx = FileActionContext(tmp_path / "workflow", "workflow")
    microscope = Microscope(
        MicroscopeSettings(control="simulated", ip_address="localhost"),
        ContextualTextLogger(fallback=ctx.text_logger).derive("microscope"),
    )
    return Workflow(microscope, Actions(), Propagations(), ctx)


@pytest.fixture
def add_action(workflow: Workflow, tmp_path: Path) -> Callable[..., Action]:
    """Add an action, prepared to run, at the end of the workflow or at `index`."""

    def add(type_key: str, name: str, index: int | None = None, **settings: object):
        action_cls = ACTION_REGISTRY.get(type_key)
        action = action_cls(
            name=name,
            microscope=workflow.microscope,
            settings=action_cls.settings_cls()(**settings),
            ctx=FileActionContext(tmp_path / name, name, slice=workflow.ctx.slice),
            actions=workflow.actions,
        )

        action.write_properties(action.collect_properties())
        action.ctx.state_store.write("state.yaml", action.state)
        action.ctx.settings_store.write("settings.yaml", action.settings)
        workflow.actions.insert(
            len(workflow.actions) if index is None else index, action
        )
        return action

    return add


@pytest.fixture
def add_step_af(add_action: Callable[..., Action]) -> Callable[..., Autofocus]:
    """Add a step-mode working-distance autofocus linked to `imaging`."""

    def add(
        name: str,
        imaging: str = "imaging",
        frequency: Every = ONCE,
        max_sweeps: int = 1,
        index: int | None = None,
    ) -> Autofocus:
        mode = StepModeSettings(
            sweeping=LinearSweeping(
                target_attribute="working_distance",
                range=(-1000.0, 1000.0),
                steps=STEPS,
            ),
            max_sweeps=max_sweeps,
        )
        action = add_action(
            "autofocus",
            name,
            index=index,
            linked_imaging=imaging,
            mode=mode,
            execution_frequency=frequency,
        )
        assert isinstance(action, Autofocus)
        return action

    return add


def _run_slice(workflow: Workflow) -> None:
    """Run the next slice of the workflow."""
    # before the first slice the counter is still 0
    workflow.run(last_slice=max(workflow.ctx.slice, 1))


def _run(workflow: Workflow, slices: int, *afs: Autofocus) -> list[list[bool]]:
    """Run `slices` more slices; per slice, whether each autofocus has a run in progress."""
    history = []
    for _ in range(slices):
        _run_slice(workflow)
        history.append([af.run_in_progress for af in afs])
    return history


def _running_slices(history: list[list[bool]], which: int) -> list[int]:
    """Indices of the slices after which autofocus `which` had a run in progress."""
    return [i for i, row in enumerate(history) if row[which]]


def test_step_autofocuses_on_one_imaging_run_one_at_a_time(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> None:
    add_action("imaging", "imaging")
    first = add_step_af("first")
    second = add_step_af("second")

    history = _run(workflow, 2 * STEPS + 2, first, second)

    assert not any(all(row) for row in history)
    first_slices = _running_slices(history, 0)
    second_slices = _running_slices(history, 1)
    assert len(first_slices) == len(second_slices) == STEPS
    # the second starts on the slice the first finishes (it comes later in the list)
    assert second_slices[0] == first_slices[-1] + 1


def test_waiting_request_keeps_its_slice_and_ignores_new_triggers(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> None:
    add_action("imaging", "imaging")
    first = add_step_af("first")
    # triggered on every slice while it waits
    second = add_step_af("second", frequency=Every(n=1))

    for _ in range(STEPS):
        _run_slice(workflow)
        assert first.run_in_progress
        assert second.queued_since == 1
        assert not second.run_in_progress


def test_step_autofocus_waits_for_every_sweep_of_the_run(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BasicSweepingStrategy.evaluate

    def best_at_upper_edge(
        self: BasicSweepingStrategy, results: list[AutofocusResult]
    ) -> BasicEvaluation:
        evaluation = original(self, results)
        return dataclasses.replace(evaluation, best=max(evaluation.mean_sharpness))

    # the best value always lies at the same edge, so every run takes max_sweeps sweeps
    monkeypatch.setattr(BasicSweepingStrategy, "evaluate", best_at_upper_edge)

    add_action("imaging", "imaging")
    first = add_step_af("first", max_sweeps=2)
    second = add_step_af("second")

    history = _run(workflow, 3 * STEPS + 2, first, second)

    assert not any(all(row) for row in history)
    first_slices = _running_slices(history, 0)
    second_slices = _running_slices(history, 1)
    assert len(first_slices) == 2 * STEPS
    assert second_slices[0] == first_slices[-1] + 1


def test_step_autofocuses_on_different_imagings_run_together(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> None:
    add_action("imaging", "imaging")
    add_action("imaging", "other imaging")
    first = add_step_af("first")
    second = add_step_af("second", imaging="other imaging")

    history = _run(workflow, 1, first, second)

    assert history == [[True, True]]


def test_waiting_requests_start_in_the_order_they_were_made(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> None:
    add_action("imaging", "imaging")
    running = add_step_af("running")
    early = add_step_af("early")  # requested on slice 1
    workflow.run(last_slice=2)
    # added later but placed before `early` in the workflow; requested on slice 3
    late = add_step_af("late", frequency=Every(n=1), index=2)

    history = _run(workflow, 3 * STEPS, running, late, early)

    assert not any(sum(row) > 1 for row in history)
    late_slices = _running_slices(history, 1)
    early_slices = _running_slices(history, 2)
    assert early_slices[-1] < late_slices[0]


def test_removing_the_running_autofocus_frees_the_imaging(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> None:
    add_action("imaging", "imaging")
    first = add_step_af("first")
    second = add_step_af("second")
    workflow.run(last_slice=1)
    assert second.queued_since == 1

    # removed while the workflow is paused
    workflow.actions.remove(first)
    workflow.run(last_slice=2)

    assert second.run_in_progress
    assert second.queued_since is None


def test_basic_autofocus_is_not_blocked_by_a_step_run(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> None:
    add_action("imaging", "imaging")
    step = add_step_af("step")
    workflow.run(last_slice=1)
    assert step.run_in_progress

    basic = add_action(
        "autofocus",
        "basic",
        linked_imaging="imaging",
        mode=BasicModeSettings(),
        execution_frequency=Every(n=1),
    )

    assert basic.should_execute()


def test_waiting_request_survives_saving_the_state(tmp_path: Path) -> None:
    file = tmp_path / "state.yaml"
    AutofocusState(queued_since=7).to_file(file)

    assert AutofocusState.from_file(file).queued_since == 7


def test_state_saved_without_a_waiting_request_still_loads(tmp_path: Path) -> None:
    file = tmp_path / "state.yaml"
    file.write_text("sweeps_done: 0\n", encoding="utf-8")

    assert AutofocusState.from_file(file).queued_since is None


def _next_frame_wd(imaging: Action) -> float:
    """Working distance of the imaging's next frame (nm)."""
    props = imaging.read_properties().electron_beam
    assert props is not None and props.working_distance is not None
    return props.working_distance


@pytest.fixture
def running(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_step_af: Callable[..., Autofocus],
) -> tuple[Action, Autofocus]:
    """An imaging and a step-mode autofocus whose run is in progress, after slice 1."""
    imaging = add_action(
        "imaging",
        "imaging",
        properties_to_collect=PropertyNames(electron_beam=["working_distance"]),
    )
    autofocus = add_step_af("af")
    _run_slice(workflow)
    assert autofocus.run_in_progress
    return imaging, autofocus


def _stored_state(autofocus: Autofocus) -> AutofocusState:
    return autofocus.ctx.state_store.read("state.yaml", AutofocusState)


def test_new_properties_of_a_step_autofocus_abort_its_run_and_restore_the_imaging(
    running: tuple[Action, Autofocus],
) -> None:
    imaging, autofocus = running
    base = autofocus.state.sweep_base_value
    assert base is not None
    assert _next_frame_wd(imaging) != pytest.approx(base[0])

    autofocus.replace_properties(autofocus.read_properties())

    assert not autofocus.run_in_progress
    assert not _stored_state(autofocus).sweep_in_progress
    # the next frame is no longer taken at the trial value
    assert _next_frame_wd(imaging) == pytest.approx(base[0])


def test_preparing_a_step_autofocus_aborts_its_run(
    running: tuple[Action, Autofocus],
) -> None:
    _, autofocus = running

    autofocus.prepare()

    assert not autofocus.run_in_progress
    assert not _stored_state(autofocus).sweep_in_progress


def test_new_properties_of_the_linked_imaging_abort_the_run_and_are_kept(
    running: tuple[Action, Autofocus],
) -> None:
    imaging, autofocus = running
    edited = GlobalProperties(electron_beam=BeamProperties(working_distance=4.2e6))

    imaging.replace_properties(edited)

    assert not autofocus.run_in_progress
    assert imaging.read_properties() == edited


def test_preparing_the_linked_imaging_aborts_the_run(
    running: tuple[Action, Autofocus],
) -> None:
    imaging, autofocus = running

    imaging.prepare()

    assert not autofocus.run_in_progress


def test_new_properties_of_other_actions_keep_the_run(
    running: tuple[Action, Autofocus],
    add_action: Callable[..., Action],
) -> None:
    _, autofocus = running
    other = add_action("imaging", "other")

    other.replace_properties(other.read_properties())
    other.prepare()

    assert autofocus.run_in_progress


def test_aborted_run_keeps_a_waiting_request(
    workflow: Workflow,
    running: tuple[Action, Autofocus],
    add_step_af: Callable[..., Autofocus],
) -> None:
    _, autofocus = running
    waiting = add_step_af("waiting", frequency=Every(n=1))
    _run_slice(workflow)
    assert waiting.queued_since == 2

    autofocus.replace_properties(autofocus.read_properties())

    assert not autofocus.run_in_progress
    assert waiting.queued_since == 2


def test_settings_change_discarding_the_run_restores_the_imaging(
    running: tuple[Action, Autofocus],
) -> None:
    imaging, autofocus = running
    base = autofocus.state.sweep_base_value
    assert base is not None

    autofocus.settings.linked_imaging = "elsewhere"

    assert not autofocus.run_in_progress
    # restored on the imaging the run patched, not the newly linked one
    assert _next_frame_wd(imaging) == pytest.approx(base[0])


def test_resetting_the_state_of_a_step_autofocus_restores_the_imaging(
    running: tuple[Action, Autofocus],
) -> None:
    imaging, autofocus = running
    base = autofocus.state.sweep_base_value
    assert base is not None

    # e.g. the Reset state button
    autofocus.reset_state()

    assert not autofocus.run_in_progress
    assert _next_frame_wd(imaging) == pytest.approx(base[0])


def test_workflow_reset_leaves_the_imaging_props_alone(
    running: tuple[Action, Autofocus],
) -> None:
    imaging, autofocus = running
    before = imaging.read_properties()

    autofocus.reset()

    assert not autofocus.run_in_progress
    assert imaging.read_properties() == before

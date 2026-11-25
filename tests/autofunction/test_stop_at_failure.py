# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Failed autofunction runs, and runs whose best value keeps lying at the edge."""

import dataclasses
from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.autofunction.autofunction import Autofunction, AutofunctionState
from fibsem_maestro.autofunction.autofunction_mode import BasicMode
from fibsem_maestro.autofunction.edge_streak import EdgeStreak
from fibsem_maestro.autofunction.error import (
    AutofunctionError,
    AutofunctionNotConvergingError,
)
from fibsem_maestro.autofunction.sweeping_strategy import BasicSweepingStrategy
from fibsem_maestro.core.detail_band import DetailBand
from fibsem_maestro.core.frequency import Every
from fibsem_maestro.criterion.criterion import Criterion
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.settings.autofunction_settings import BasicMode as BasicModeSettings
from fibsem_maestro.settings.autofunction_settings import (
    CustomCriterion,
    ImagingCriterion,
)
from fibsem_maestro.settings.autofunction_settings import StepMode as StepModeSettings
from fibsem_maestro.settings.criterion_settings import BandpassMetric, CriterionSettings
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.settings.sweeping_settings import LinearSweeping
from fibsem_maestro.workflow.error import ActionError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.autofunction.evaluation import BasicEvaluation
    from fibsem_maestro.autofunction.result import AutofunctionResult
    from fibsem_maestro.autofunction.sweep_step import SweepPoint, SweepStep
    from fibsem_maestro.workflow.workflow import Workflow

# half the swept working-distance range (nm); a sweep at the edge moves by this much
STEP = 500.0

# steps of every sweep: base - STEP, base, base + STEP
STEPS = 3

CRITERION = CriterionSettings(
    sharpness_metric=BandpassMetric(detail=DetailBand(low=1.0, high=1.0))
)
"""A criterion with its detail band set: the stored properties hold no pixel size."""

WD = PropertyNames(electron_beam=["working_distance"])


@dataclasses.dataclass
class Sweeps:
    """
    Where the best value of every basic-strategy sweep lies.

    Attributes:
        inside: Indices of the sweeps whose best value is their middle value,
            inside the swept region; every other sweep picks its largest value.
        bests: The best value of every sweep so far.
    """

    inside: set[int] = dataclasses.field(default_factory=set)
    bests: list[SweepPoint] = dataclasses.field(default_factory=list)


@pytest.fixture
def sweeps(monkeypatch: pytest.MonkeyPatch) -> Sweeps:
    script = Sweeps()
    original = BasicSweepingStrategy.evaluate

    def scripted(
        self: BasicSweepingStrategy, results: list[AutofunctionResult]
    ) -> BasicEvaluation:
        evaluation = original(self, results)
        values = sorted(evaluation.mean_sharpness)
        inside = len(script.bests) in script.inside
        best = values[len(values) // 2] if inside else values[-1]
        script.bests.append(best)
        return dataclasses.replace(evaluation, best=best)

    monkeypatch.setattr(BasicSweepingStrategy, "evaluate", scripted)
    return script


@pytest.fixture
def failing_sharpness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every sharpness calculation fail."""

    def fail(*_: object) -> None:
        raise RuntimeError("no sharpness")

    monkeypatch.setattr(Criterion, "calculate_sharpness", fail)


@pytest.fixture
def add_basic_af(add_action: Callable[..., Action]) -> Callable[..., Autofunction]:
    """Add an imaging, then a basic working-distance autofunction running every slice."""

    def add(
        max_sweeps: int = 1, collect: PropertyNames = WD, **settings: object
    ) -> Autofunction:
        add_action("imaging", "imaging", properties_to_collect=WD)
        action = add_action(
            "autofunction",
            "af",
            linked_imaging="imaging",
            mode=BasicModeSettings(
                sweeping=LinearSweeping(
                    target_attribute="working_distance",
                    range=(-STEP, STEP),
                    steps=STEPS,
                ),
                criterion=CRITERION,
                max_sweeps=max_sweeps,
            ),
            execution_frequency=Every(n=1),
            properties_to_collect=collect,
            **settings,
        )
        assert isinstance(action, Autofunction)
        return action

    return add


@pytest.fixture
def add_step_af(add_action: Callable[..., Action]) -> Callable[..., Autofunction]:
    """Add a step-mode working-distance autofunction, then its imaging."""

    def add(imaging_criterion: bool = False, **settings: object) -> Autofunction:
        action = add_action(
            "autofunction",
            "af",
            linked_imaging="imaging",
            mode=StepModeSettings(
                sweeping=LinearSweeping(
                    target_attribute="working_distance",
                    range=(-STEP, STEP),
                    steps=STEPS,
                ),
                criterion=ImagingCriterion()
                if imaging_criterion
                else CustomCriterion(criterion=CRITERION),
            ),
            execution_frequency=Every(n=1000),
            **settings,
        )
        add_action(
            "imaging",
            "imaging",
            properties_to_collect=WD,
            criterion=CRITERION if imaging_criterion else None,
        )
        assert isinstance(action, Autofunction)
        return action

    return add


def _run_slice(workflow: Workflow) -> None:
    # before the first slice the counter is still 0
    workflow.run(last_slice=max(workflow.ctx.slice, 1))


def _failure(workflow: Workflow) -> BaseException | None:
    """Run the next slice; return why its action failed, or None."""
    with pytest.raises(ActionError) as info:
        _run_slice(workflow)
    return info.value.__cause__


def _wd(workflow: Workflow) -> float:
    """Working distance on the microscope (nm)."""
    return workflow.microscope.electron_beam.working_distance


def _stored_wd(action: Action) -> float:
    """Working distance in the action's properties for the current slice (nm)."""
    props = action.read_properties().electron_beam
    assert props is not None and props.working_distance is not None
    return props.working_distance


@pytest.mark.parametrize(
    ("second", "count"),
    [((1.0, 0.0), 2), ((0.8, 0.6), 2), ((0.0, 1.0), 1), ((-1.0, 0.0), 1)],
)
def test_edge_streak_counts_sweeps_at_the_edge_in_the_same_direction(
    second: tuple[float, float], count: int
) -> None:
    streak = EdgeStreak()
    streak.record((1.0, 0.0))

    streak.record(second)

    assert streak.count == count


def test_edge_streak_ends_with_a_sweep_ending_inside() -> None:
    streak = EdgeStreak()
    streak.record((1.0,))

    streak.record(None)

    assert streak.count == 0


@pytest.mark.usefixtures("sweeps")
def test_edge_streak_fails_the_run_that_completes_it_across_runs(
    workflow: Workflow, add_basic_af: Callable[..., Autofunction]
) -> None:
    add_basic_af(max_edge_sweeps=3)

    _run_slice(workflow)
    _run_slice(workflow)

    assert isinstance(_failure(workflow), AutofunctionNotConvergingError)


def test_edge_streak_fails_a_run_that_never_converges_once_it_ends(
    workflow: Workflow, add_basic_af: Callable[..., Autofunction], sweeps: Sweeps
) -> None:
    add_basic_af(max_sweeps=8, max_edge_sweeps=3)

    assert isinstance(_failure(workflow), AutofunctionNotConvergingError)
    # the run was not stopped while it still had sweeps left
    assert len(sweeps.bests) == 8


def test_edge_streak_restarts_after_a_sweep_ending_inside(
    workflow: Workflow, add_basic_af: Callable[..., Autofunction], sweeps: Sweeps
) -> None:
    sweeps.inside = {2}
    add_basic_af(max_edge_sweeps=3)

    for _ in range(5):
        _run_slice(workflow)

    assert isinstance(_failure(workflow), AutofunctionNotConvergingError)


def test_edge_streak_is_not_checked_without_a_limit(
    workflow: Workflow, add_basic_af: Callable[..., Autofunction], sweeps: Sweeps
) -> None:
    af = add_basic_af(max_edge_sweeps=None)

    for _ in range(5):
        _run_slice(workflow)

    assert _stored_wd(af) == pytest.approx(sweeps.bests[-1][0])


def test_edge_streak_failure_keeps_the_best_value_on_the_microscope(
    workflow: Workflow, add_basic_af: Callable[..., Autofunction], sweeps: Sweeps
) -> None:
    add_basic_af(max_edge_sweeps=2)
    _run_slice(workflow)

    _failure(workflow)

    assert _wd(workflow) == pytest.approx(sweeps.bests[-1][0])


def test_edge_streak_without_stop_at_failure_keeps_the_result(
    workflow: Workflow, add_basic_af: Callable[..., Autofunction], sweeps: Sweeps
) -> None:
    af = add_basic_af(max_edge_sweeps=2, stop_at_failure=False)

    for _ in range(3):
        _run_slice(workflow)

    assert _stored_wd(af) == pytest.approx(sweeps.bests[-1][0])


@pytest.mark.usefixtures("sweeps")
def test_edge_streak_survives_saving_the_state(
    workflow: Workflow,
    add_basic_af: Callable[..., Autofunction],
    tmp_path: Path,
) -> None:
    af = add_basic_af(max_edge_sweeps=3)
    _run_slice(workflow)
    _run_slice(workflow)
    file = tmp_path / "state.yaml"
    af.state.to_file(file)

    af.set_state(AutofunctionState())
    _run_slice(workflow)  # a streak of 1 again
    af.set_state(AutofunctionState.from_file(file))

    assert isinstance(_failure(workflow), AutofunctionNotConvergingError)


@pytest.mark.usefixtures("failing_sharpness")
@pytest.mark.parametrize("collect", [WD, PropertyNames()])
@pytest.mark.parametrize("stop_at_failure", [True, False])
def test_failed_run_restores_the_working_distance(
    workflow: Workflow,
    add_basic_af: Callable[..., Autofunction],
    stop_at_failure: bool,
    collect: PropertyNames,
) -> None:
    af = add_basic_af(collect=collect, stop_at_failure=stop_at_failure)
    before = _wd(workflow)

    if stop_at_failure:
        assert isinstance(_failure(workflow), AutofunctionError)
    else:
        _run_slice(workflow)

    assert _wd(workflow) == pytest.approx(before)
    assert not af.run_in_progress


@pytest.mark.usefixtures("failing_sharpness")
def test_failed_run_without_stop_at_failure_propagates_nothing(
    workflow: Workflow,
    add_basic_af: Callable[..., Autofunction],
) -> None:
    af = add_basic_af(stop_at_failure=False)
    imaging = workflow.actions.named("imaging")
    workflow.propagations.register_rule("af", ["imaging"], WD)
    imaging_wd = _stored_wd(imaging)
    props = af.read_properties()
    assert props.electron_beam is not None
    props.electron_beam.working_distance = imaging_wd + STEP
    af.write_properties(props)

    _run_slice(workflow)

    assert _stored_wd(af) == pytest.approx(imaging_wd + STEP)
    assert _stored_wd(imaging) == pytest.approx(imaging_wd)


@pytest.mark.parametrize("imaging_criterion", [False, True])
def test_step_mode_skips_a_trial_it_cannot_score(
    workflow: Workflow,
    add_step_af: Callable[..., Autofunction],
    monkeypatch: pytest.MonkeyPatch,
    imaging_criterion: bool,
) -> None:
    af = add_step_af(imaging_criterion=imaging_criterion)
    evaluated: list[int] = []
    evaluate = BasicSweepingStrategy.evaluate

    def counting(
        self: BasicSweepingStrategy, results: list[AutofunctionResult]
    ) -> BasicEvaluation:
        evaluated.append(len(results))
        return evaluate(self, results)

    monkeypatch.setattr(BasicSweepingStrategy, "evaluate", counting)
    # nothing to score the frame of slice 2 with
    if imaging_criterion:
        wait = Imaging.wait_for_sharpness
        monkeypatch.setattr(
            Imaging,
            "wait_for_sharpness",
            lambda self: None if self.frame_slice == 2 else wait(self),
        )
    else:
        frame = Imaging.last_acquired_image.fget
        assert frame is not None
        monkeypatch.setattr(
            Imaging,
            "last_acquired_image",
            property(lambda self: None if self.frame_slice == 2 else frame(self)),
        )

    for _ in range(STEPS + 1):
        _run_slice(workflow)

    assert evaluated == [STEPS - 1]
    assert not af.run_in_progress


@pytest.mark.parametrize("stop_at_failure", [True, False])
def test_failed_run_restores_the_properties_it_stores(
    workflow: Workflow,
    add_basic_af: Callable[..., Autofunction],
    monkeypatch: pytest.MonkeyPatch,
    stop_at_failure: bool,
) -> None:
    af = add_basic_af(
        collect=PropertyNames(electron_beam=["working_distance", "dwell_time"]),
        stop_at_failure=stop_at_failure,
    )
    beam = workflow.microscope.electron_beam
    before = beam.dwell_time

    # e.g. an Autoscript routine that fails after changing the beam
    def changing_and_failing(*_: object, **__: object) -> Generator[SweepStep]:
        beam.dwell_time = 2 * before
        raise AutofunctionError("routine failed")
        yield

    monkeypatch.setattr(BasicMode, "execute", changing_and_failing)

    if stop_at_failure:
        assert isinstance(_failure(workflow), AutofunctionError)
    else:
        _run_slice(workflow)

    assert beam.dwell_time == pytest.approx(before)
    assert not af.run_in_progress


@pytest.mark.parametrize("stop_at_failure", [True, False])
def test_step_run_without_any_score_restores_the_imaging(
    workflow: Workflow,
    add_step_af: Callable[..., Autofunction],
    monkeypatch: pytest.MonkeyPatch,
    stop_at_failure: bool,
) -> None:
    af = add_step_af(stop_at_failure=stop_at_failure)
    imaging = workflow.actions.named("imaging")
    base = _stored_wd(imaging)
    monkeypatch.setattr(Imaging, "last_acquired_image", property(lambda _: None))

    for _ in range(STEPS):
        _run_slice(workflow)
    # the trailing slice, which scores the last trial and ends the sweep
    if stop_at_failure:
        assert isinstance(_failure(workflow), AutofunctionError)
    else:
        _run_slice(workflow)

    assert not af.run_in_progress
    assert _stored_wd(imaging) == pytest.approx(base)


@pytest.mark.usefixtures("failing_sharpness")
@pytest.mark.parametrize("stop_at_failure", [True, False])
def test_failed_test_run_restores_the_working_distance(
    workflow: Workflow,
    add_basic_af: Callable[..., Autofunction],
    stop_at_failure: bool,
) -> None:
    af = add_basic_af(stop_at_failure=stop_at_failure)
    before = _wd(workflow)

    if stop_at_failure:
        with pytest.raises(AutofunctionError):
            af.test()
    else:
        af.test()

    assert _wd(workflow) == pytest.approx(before)

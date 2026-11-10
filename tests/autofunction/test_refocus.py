# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


import dataclasses
from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.autofunction.autofunction import Autofunction
from fibsem_maestro.autofunction.refocus import Refocus, RefocusState
from fibsem_maestro.autofunction.sweeping_strategy import BasicSweepingStrategy
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.frequency import Every
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.settings.autofunction_settings import BasicMode as BasicModeSettings
from fibsem_maestro.settings.autofunction_settings import StepMode as StepModeSettings
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.settings.sweeping_settings import CircularSweeping, LinearSweeping
from fibsem_maestro.workflow.propagations import PropagationRule, Propagations
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.autofunction.evaluation import BasicEvaluation
    from fibsem_maestro.autofunction.result import AutofunctionResult

# the refocus sweeps the working distance by up to this much (nm) and picks the top
REFOCUS_STEP = 100.0

# radius of the stigmator circle
RADIUS = 0.01

# triggers on slice 1 only, within the length of these tests
ONCE = Every(n=1000)


@pytest.fixture(autouse=True)
def best_at_largest_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every basic-strategy sweep pick its largest candidate."""
    original = BasicSweepingStrategy.evaluate

    def largest(
        self: BasicSweepingStrategy, results: list[AutofunctionResult]
    ) -> BasicEvaluation:
        evaluation = original(self, results)
        return dataclasses.replace(evaluation, best=max(evaluation.mean_sharpness))

    monkeypatch.setattr(BasicSweepingStrategy, "evaluate", largest)


@pytest.fixture
def add_imaging(add_action: Callable[..., Action]) -> Callable[..., Imaging]:
    """Add the imaging `imaging`, which records the working distance of its frames."""

    def add(index: int | None = None) -> Imaging:
        action = add_action(
            "imaging",
            "imaging",
            index=index,
            properties_to_collect=PropertyNames(electron_beam=["working_distance"]),
        )
        assert isinstance(action, Imaging)
        return action

    return add


@pytest.fixture
def add_stigmator_af(add_action: Callable[..., Action]) -> Callable[..., Autofunction]:
    """Add a 2D stigmator step-mode autofunction linked to `imaging`."""

    def add(name: str = "stigmator", index: int | None = None) -> Autofunction:
        action = add_action(
            "autofunction",
            name,
            index=index,
            linked_imaging="imaging",
            mode=StepModeSettings(
                sweeping=CircularSweeping(radius=RADIUS, steps=4),
            ),
            execution_frequency=ONCE,
            # leaves its stale working distance on the microscope before the refocus
            properties_to_collect=PropertyNames(electron_beam=["working_distance"]),
        )
        assert isinstance(action, Autofunction)
        return action

    return add


@pytest.fixture
def add_refocus(add_action: Callable[..., Action]) -> Callable[..., Refocus]:
    """Add a working-distance refocus linked to `imaging`."""

    def add(target: str = "working_distance", index: int | None = None) -> Refocus:
        action = add_action(
            "refocus",
            "refocus",
            index=index,
            linked_imaging="imaging",
            mode=BasicModeSettings(
                sweeping=LinearSweeping(
                    target_attribute=target, range=(0.0, REFOCUS_STEP), steps=2
                ),
            ),
        )
        assert isinstance(action, Refocus)
        return action

    return add


def _next_frame_wd(imaging: Imaging) -> float:
    """Working distance of the imaging's next frame (nm)."""
    props = imaging.read_properties().electron_beam
    assert props is not None and props.working_distance is not None
    return props.working_distance


def _run(
    workflow: Workflow, slices: int, imaging: Imaging, autofunction: Autofunction
) -> tuple[list[bool], list[bool]]:
    """
    Run `slices` more slices.

    Returns:
        Per slice: whether the autofunction had a run in progress afterwards, and
        whether the refocus changed the working distance of the next frame.
    """
    running, refocused = [], []
    for _ in range(slices):
        before = _next_frame_wd(imaging)
        # before the first slice the counter is still 0
        workflow.run(last_slice=max(workflow.ctx.slice, 1))
        running.append(autofunction.run_in_progress)
        refocused.append(
            _next_frame_wd(imaging) == pytest.approx(before + REFOCUS_STEP)
        )
    return running, refocused


def test_refocus_runs_during_a_step_run_and_once_after_it(
    workflow: Workflow,
    add_imaging: Callable[..., Imaging],
    add_stigmator_af: Callable[..., Autofunction],
    add_refocus: Callable[..., Refocus],
) -> None:
    imaging = add_imaging()
    stigmator = add_stigmator_af()
    add_refocus()

    running, refocused = _run(workflow, 9, imaging, stigmator)

    last_running = max(i for i, r in enumerate(running) if r)
    # every trial frame, plus the first frame at the run's result
    assert refocused == [i <= last_running + 1 for i in range(len(running))]


def test_refocus_keeps_the_trial_value_and_its_own_properties(
    workflow: Workflow,
    add_imaging: Callable[..., Imaging],
    add_stigmator_af: Callable[..., Autofunction],
    add_refocus: Callable[..., Refocus],
) -> None:
    imaging = add_imaging()
    add_stigmator_af()
    refocus = add_refocus()
    stigmator = workflow.microscope.electron_beam.stigmator
    own_props = refocus.read_properties()

    workflow.run(last_slice=1)

    next_frame = imaging.read_properties().electron_beam
    assert next_frame is not None
    # the first trial lies at (0, +radius) from the base
    assert next_frame.stigmator_x == pytest.approx(stigmator.x)
    assert next_frame.stigmator_y == pytest.approx(stigmator.y + RADIUS)
    assert refocus.read_properties() == own_props


def test_refocus_never_refocuses_an_attribute_the_run_sweeps(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_refocus: Callable[..., Refocus],
) -> None:
    add_action("imaging", "imaging")
    add_action(
        "autofunction",
        "focus",
        linked_imaging="imaging",
        mode=StepModeSettings(
            sweeping=LinearSweeping(
                target_attribute="working_distance", range=(-1000.0, 1000.0), steps=3
            ),
        ),
        execution_frequency=ONCE,
    )
    refocus = add_refocus()

    for _ in range(6):
        workflow.run(last_slice=max(workflow.ctx.slice, 1))
        # a qualifying run would leave a final refocus pending
        assert refocus.state == RefocusState()


@pytest.mark.parametrize(
    ("order", "valid"),
    [
        (["imaging", "stigmator", "refocus"], True),
        (["refocus", "imaging", "stigmator"], True),
        (["stigmator", "refocus", "imaging"], True),
        (["stigmator", "imaging", "refocus"], False),
        (["imaging", "refocus", "stigmator"], False),
        (["refocus", "stigmator", "imaging"], False),
    ],
)
def test_refocus_must_run_between_the_step_autofunction_and_the_imaging(
    add_imaging: Callable[..., Imaging],
    add_stigmator_af: Callable[..., Autofunction],
    add_refocus: Callable[..., Refocus],
    order: list[str],
    valid: bool,
) -> None:
    adders = {
        "imaging": add_imaging,
        "stigmator": add_stigmator_af,
        "refocus": add_refocus,
    }
    actions = {name: adders[name]() for name in order}
    refocus = actions["refocus"]

    issues = refocus.preparation_issues()

    assert (issues == []) == valid
    if not valid:
        assert "'stigmator'" in issues[0]


def test_refocus_uses_the_beam_of_its_imaging(
    add_action: Callable[..., Action],
    add_refocus: Callable[..., Refocus],
) -> None:
    add_action("imaging", "imaging", beam_type=BeamType.ION)
    refocus = add_refocus()

    assert refocus.beam_type is BeamType.ION


def test_refocus_reports_a_missing_imaging(
    add_refocus: Callable[..., Refocus],
) -> None:
    refocus = add_refocus()

    assert refocus.preparation_issues() == [
        "The linked imaging 'imaging' does not exist."
    ]


def test_pending_final_refocus_survives_saving_the_state(tmp_path: Path) -> None:
    file = tmp_path / "state.yaml"
    RefocusState(final_refocus_pending=True).to_file(file)

    assert RefocusState.from_file(file).final_refocus_pending


def test_workflow_drops_propagation_rules_from_or_to_a_refocus(
    workflow: Workflow,
    add_action: Callable[..., Action],
    add_refocus: Callable[..., Refocus],
) -> None:
    add_action("imaging", "imaging")
    add_action("imaging", "other")
    add_refocus()
    wd = PropertyNames(electron_beam=["working_distance"])
    rules = [
        PropagationRule("refocus", ["imaging"], wd),
        PropagationRule("imaging", ["refocus", "other"], wd),
    ]

    # e.g. loaded from a workflow saved before the refocus excluded itself
    loaded = Workflow(
        workflow.microscope,
        workflow.actions,
        Propagations.from_rules(rules),
        workflow.ctx,
    )

    assert loaded.propagations.rules == [PropagationRule("imaging", ["other"], wd)]

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.action.outcome import Produced
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.core.direction import Direction
from fibsem_maestro.logging.text.file import close_all_log_files
from fibsem_maestro.post_milling_correction.correction import PostMillingCorrection
from fibsem_maestro.post_milling_correction.error import PostMillingCorrectionError
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.post_milling_correction_settings import (
    PostMillingCorrectionSettings,
)
from fibsem_maestro.settings.property_names import PropertyNames
from fibsem_maestro.ui.views.log_panel.parsing import LEVELS, level_value
from fibsem_maestro.ui.views.log_panel.reader import LogReader
from fibsem_maestro.workflow.propagations import PropagationRule, Propagations
from fibsem_maestro.workflow.workflow import Workflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.properties.beam_properties import BeamProperties

# per run (nm)
Y_CORRECTION = 10.0
WD_CORRECTION = 20.0

IMAGE_TO_BEAM_SHIFT = (3.0, -2.0)
ION_IMAGE_TO_BEAM_SHIFT = (5.0, 7.0)

CORRECTED = ["working_distance", "beam_shift"]


def mentions(text: str, *parts: str) -> bool:
    """
    Whether a message contains every part, ignoring case.

    Messages are checked for the information they carry, not their wording.
    """
    lowered = text.lower()
    return all(part.lower() in lowered for part in parts)


@pytest.fixture(autouse=True)
def image_to_beam_shift(monkeypatch: pytest.MonkeyPatch, workflow: Workflow) -> None:
    for beam, factors in (
        (workflow.microscope.electron_beam, IMAGE_TO_BEAM_SHIFT),
        (workflow.microscope.ion_beam, ION_IMAGE_TO_BEAM_SHIFT),
    ):
        monkeypatch.setattr(
            type(beam), "image_to_beam_shift", property(lambda _, f=factors: f)
        )


@pytest.fixture
def add_prepared(add_action: Callable[..., Action]) -> Callable[..., Action]:
    """Add an action at the end of the workflow, with its properties collected."""

    def add(type_key: str, name: str, **settings: object) -> Action:
        action = add_action(type_key, name, **settings)
        action.write_properties(action.collect_properties())
        return action

    return add


@pytest.fixture
def add_imaging(add_prepared: Callable[..., Action]) -> Callable[..., Action]:
    """Add an imaging collecting the corrected properties of its beam."""

    def add(name: str, beam_type: BeamType = BeamType.ELECTRON) -> Action:
        names = (
            PropertyNames(electron_beam=CORRECTED)
            if beam_type is BeamType.ELECTRON
            else PropertyNames(ion_beam=CORRECTED)
        )
        return add_prepared(
            "imaging", name, beam_type=beam_type, properties_to_collect=names
        )

    return add


@pytest.fixture
def add_correction(add_prepared: Callable[..., Action]) -> Callable[..., Action]:
    """Add a post-milling correction, by default with both shifts set (None follows the default)."""

    def add(
        corrected: list[str],
        y: float | None = Y_CORRECTION,
        wd: float | None = WD_CORRECTION,
        **settings: object,
    ) -> Action:
        return add_prepared(
            "post_milling_correction",
            "correction",
            corrected_actions=corrected,
            y_correction=y,
            wd_correction=wd,
            **settings,
        )

    return add


def geometric(slice_distance: float, direction: Direction) -> tuple[float, float]:
    """The shifts of one slice, seen by the electron beam at 52 degrees to the ion beam (nm)."""
    sign = -1.0 if direction is Direction.UP else 1.0
    angle = math.radians(52.0)
    return (
        sign * slice_distance * math.cos(angle),
        sign * slice_distance * math.sin(angle),
    )


def upcoming(action: Action) -> BeamProperties:
    """The electron-beam properties the action applies on its next run."""
    props = action.read_properties().electron_beam
    assert props is not None
    return props


def shifted(
    props: BeamProperties, runs: int, y: float = Y_CORRECTION, wd: float = WD_CORRECTION
) -> tuple[float, BeamShift]:
    """The working distance and beam shift after `runs` corrections."""
    assert props.working_distance is not None and props.beam_shift is not None
    delta = BeamShift(0.0, IMAGE_TO_BEAM_SHIFT[1] * y * runs)
    return props.working_distance + wd * runs, props.beam_shift + delta


def corrected_values(props: BeamProperties) -> tuple[float, BeamShift]:
    assert props.working_distance is not None and props.beam_shift is not None
    return props.working_distance, props.beam_shift


def test_corrections_accumulate_on_actions_before_and_after_it(
    workflow: Workflow,
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    before = add_imaging("before")
    add_correction(["before", "after"])
    after = add_imaging("after")
    initial = {action.name: upcoming(action) for action in (before, after)}

    workflow.run(last_slice=3)

    for action in (before, after):
        expected = shifted(initial[action.name], runs=3)
        assert corrected_values(upcoming(action)) == pytest.approx(expected)


def test_correction_never_changes_the_microscope(
    workflow: Workflow,
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_imaging("imaging")
    correction = add_correction(["imaging"])
    microscope = workflow.microscope
    microscope.set_beam(BeamType.ION)
    beam = microscope.electron_beam
    before = (beam.working_distance, beam.beam_shift)

    outcome = correction.execute()

    # not even the beam is selected
    assert microscope.beam is microscope.ion_beam

    assert isinstance(outcome, Produced)
    assert (beam.working_distance, beam.beam_shift) == before


def test_correction_patches_only_the_corrected_properties(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    imaging = add_imaging("imaging")
    correction = add_correction(["imaging"])
    expected_wd, expected_shift = shifted(upcoming(imaging), runs=1)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    assert set(outcome.patches) == {"imaging"}
    patch = outcome.patches["imaging"]
    assert set(patch.get_property_names().electron_beam) == set(CORRECTED)
    assert patch.microscope is None and patch.ion_beam is None
    assert patch.electron_beam is not None
    assert patch.electron_beam.working_distance == pytest.approx(expected_wd)
    assert patch.electron_beam.beam_shift == expected_shift
    # it stores no properties of its own
    assert outcome.props == GlobalProperties()


def test_each_target_is_corrected_on_its_own_beam(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    electron = add_imaging("electron")
    ion = add_imaging("ion", beam_type=BeamType.ION)
    correction = add_correction(["electron", "ion"])
    ion_props = ion.read_properties().ion_beam
    assert ion_props is not None and ion_props.working_distance is not None
    assert ion_props.beam_shift is not None
    expected_ion = (
        ion_props.working_distance + WD_CORRECTION,
        ion_props.beam_shift
        + BeamShift(0.0, ION_IMAGE_TO_BEAM_SHIFT[1] * Y_CORRECTION),
    )
    expected_electron = shifted(upcoming(electron), runs=1)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    ion_patch = outcome.patches["ion"]
    assert ion_patch.electron_beam is None and ion_patch.ion_beam is not None
    assert corrected_values(ion_patch.ion_beam) == pytest.approx(expected_ion)
    electron_patch = outcome.patches["electron"]
    assert electron_patch.ion_beam is None and electron_patch.electron_beam is not None
    assert corrected_values(electron_patch.electron_beam) == pytest.approx(
        expected_electron
    )


def test_action_of_any_type_is_corrected(
    add_prepared: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    milling = add_prepared(
        "milling",
        "milling",
        beam_type=BeamType.ELECTRON,
        properties_to_collect=PropertyNames(electron_beam=CORRECTED),
    )
    correction = add_correction(["milling"])
    expected = shifted(upcoming(milling), runs=1)

    outcome = correction.execute()

    assert correction.preparation_issues() == []
    assert isinstance(outcome, Produced)
    patch = outcome.patches["milling"].electron_beam
    assert patch is not None
    assert corrected_values(patch) == pytest.approx(expected)


def test_action_without_a_beam_cannot_be_corrected(
    add_prepared: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    # it even collects the corrected properties, but has no beam to correct them on
    add_prepared(
        "adjust_props",
        "adjust",
        properties_to_collect=PropertyNames(electron_beam=CORRECTED),
    )
    correction = add_correction(["adjust"])

    assert correction.preparation_issues() == [
        "'adjust' works on no beam, so it cannot be corrected."
    ]
    with pytest.raises(PostMillingCorrectionError, match="works on no beam"):
        correction.execute()


def test_zero_correction_patches_nothing(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_imaging("imaging")
    correction = add_correction(["imaging"], y=0.0, wd=0.0)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    assert outcome.patches == {}


def test_manual_correction_patches_only_its_non_zero_part(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_imaging("imaging")
    correction = add_correction(["imaging"], y=0.0)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    assert outcome.patches["imaging"].get_property_names() == PropertyNames(
        electron_beam=["working_distance"]
    )


@pytest.mark.parametrize("direction", [Direction.DOWN, Direction.UP])
def test_default_correction_follows_the_milling_geometry(
    add_prepared: Callable[..., Action],
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
    direction: Direction,
) -> None:
    add_prepared("milling", "milling", slice_distance=10.0, milling_direction=direction)
    imaging = add_imaging("imaging")
    correction = add_correction(["imaging"], y=None, wd=None, linked_milling="milling")
    y, wd = geometric(10.0, direction)
    expected = shifted(upcoming(imaging), runs=1, y=y, wd=wd)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    patch = outcome.patches["imaging"].electron_beam
    assert patch is not None
    assert corrected_values(patch) == pytest.approx(expected)


def test_default_correction_follows_later_changes_of_the_milling(
    add_prepared: Callable[..., Action],
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    milling = add_prepared("milling", "milling", slice_distance=10.0)
    imaging = add_imaging("imaging")
    correction = add_correction(["imaging"], y=None, wd=None, linked_milling="milling")

    # nothing of the old slice distance is kept by the correction
    milling.settings.slice_distance = 30.0
    y, wd = geometric(30.0, Direction.DOWN)
    expected = shifted(upcoming(imaging), runs=1, y=y, wd=wd)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    patch = outcome.patches["imaging"].electron_beam
    assert patch is not None
    assert corrected_values(patch) == pytest.approx(expected)


def test_value_set_replaces_only_its_own_default(
    add_prepared: Callable[..., Action],
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_prepared("milling", "milling", slice_distance=10.0)
    imaging = add_imaging("imaging")
    correction = add_correction(
        ["imaging"], y=Y_CORRECTION, wd=None, linked_milling="milling"
    )
    _, wd = geometric(10.0, Direction.DOWN)
    expected = shifted(upcoming(imaging), runs=1, y=Y_CORRECTION, wd=wd)

    outcome = correction.execute()

    assert isinstance(outcome, Produced)
    patch = outcome.patches["imaging"].electron_beam
    assert patch is not None
    assert corrected_values(patch) == pytest.approx(expected)


def test_default_correction_without_a_milling_fails(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_imaging("imaging")
    correction = add_correction(["imaging"], wd=None)

    with pytest.raises(PostMillingCorrectionError, match="(?i)milling"):
        correction.execute()


def test_set_corrections_need_no_milling(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_imaging("imaging")
    correction = add_correction(["imaging"])

    assert correction.preparation_issues() == []
    assert isinstance(correction.execute(), Produced)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            {"type": "manual", "y_correction": 3.0, "wd_correction": 4.0},
            (3.0, 4.0),
        ),
        ({"type": "dynamic_focus"}, (None, None)),
    ],
)
def test_settings_with_a_correction_mode_keep_their_corrections(
    mode: dict[str, object], expected: tuple[float | None, float | None]
) -> None:
    settings = PostMillingCorrectionSettings.model_validate(
        {"linked_milling": "milling", "correction_mode": mode}
    )

    assert (settings.y_correction, settings.wd_correction) == expected


def test_correction_fails_if_a_target_stores_no_value_to_correct(
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    imaging = add_imaging("imaging")
    # e.g. collected before the properties were added
    imaging.write_properties(GlobalProperties())
    correction = add_correction(["imaging"])

    with pytest.raises(PostMillingCorrectionError, match="stores no"):
        correction.execute()


@pytest.fixture
def ready(
    add_prepared: Callable[..., Action],
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> Action:
    """A correction of `imaging` that is ready to run."""
    add_imaging("imaging")
    add_prepared("milling", "milling")
    return add_correction(["imaging"], linked_milling="milling")


def test_ready_correction_reports_no_issues(ready: Action) -> None:
    assert ready.preparation_issues() == []


@pytest.mark.parametrize(
    ("change", "parts"),
    [
        ({"corrected_actions": []}, ("correct",)),
        ({"corrected_actions": ["imaging", "ghost"]}, ("ghost",)),
        ({"y_correction": None, "linked_milling": ""}, ("milling",)),
        ({"wd_correction": None, "linked_milling": "ghost"}, ("milling", "ghost")),
        ({"y_correction": None, "linked_milling": "imaging"}, ("imaging", "milling")),
        # a milling is only needed for a default
        ({"linked_milling": "ghost"}, None),
    ],
)
def test_correction_reports_what_it_cannot_correct(
    ready: Action, change: dict[str, object], parts: tuple[str, ...] | None
) -> None:
    for name, value in change.items():
        setattr(ready.settings, name, value)

    issues = ready.preparation_issues()

    if parts is None:
        assert issues == []
    else:
        [issue] = issues
        assert mentions(issue, *parts)


@pytest.mark.parametrize(
    ("y", "wd", "missing"),
    [
        (0.0, WD_CORRECTION, ["working_distance"]),
        (Y_CORRECTION, 0.0, ["beam_shift"]),
        (0.0, 0.0, []),
        # the default, from the linked milling, is not zero
        (None, 0.0, ["beam_shift"]),
    ],
)
def test_correction_requires_targets_to_collect_what_it_corrects(
    add_prepared: Callable[..., Action],
    add_correction: Callable[..., Action],
    y: float | None,
    wd: float | None,
    missing: list[str],
) -> None:
    add_prepared("milling", "milling")
    add_prepared("imaging", "imaging", properties_to_collect=PropertyNames())
    correction = add_correction(["imaging"], y=y, wd=wd, linked_milling="milling")

    issues = correction.preparation_issues()

    # one issue per property not collected, naming it
    assert len(issues) == len(missing)
    for name in missing:
        assert any(mentions(issue, "imaging", name) for issue in issues)


def test_workflow_drops_propagation_rules_from_or_to_a_correction(
    workflow: Workflow,
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_imaging("imaging")
    add_imaging("other")
    add_correction(["imaging"])
    wd = PropertyNames(electron_beam=["working_distance"])
    rules = [
        PropagationRule("correction", ["imaging"], wd),
        PropagationRule("imaging", ["correction", "other"], wd),
    ]

    # e.g. loaded from a workflow saved while the correction propagated
    loaded = Workflow(
        workflow.microscope,
        workflow.actions,
        Propagations.from_rules(rules),
        workflow.ctx,
    )

    assert isinstance(loaded.actions[2], PostMillingCorrection)
    assert loaded.propagations.rules == [PropagationRule("imaging", ["other"], wd)]


def test_default_that_cannot_be_computed_counts_as_a_correction(
    add_prepared: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_prepared("imaging", "imaging", properties_to_collect=PropertyNames())
    # without a milling, the default may be anything once one is linked
    correction = add_correction(["imaging"], y=None, wd=0.0)

    issues = correction.preparation_issues()

    assert len(issues) == 2
    assert any(mentions(issue, "milling") for issue in issues)
    assert any(mentions(issue, "imaging", "beam_shift") for issue in issues)


def test_default_correction_used_is_logged(
    tmp_path: Path,
    add_prepared: Callable[..., Action],
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> None:
    add_prepared("milling", "milling", slice_distance=10.0)
    add_imaging("imaging")
    correction = add_correction(
        ["imaging"], y=Y_CORRECTION, wd=None, linked_milling="milling"
    )
    _, wd = geometric(10.0, Direction.DOWN)

    correction.execute()

    close_all_log_files()
    lines = LogReader(tmp_path).read(0, ["correction"], level_value(LEVELS[0]))
    debug = [line.message for line in lines if line.level == "DEBUG"]
    # the field and its value, to three significant digits whatever the format
    assert any(mentions(m, "wd_correction", f"{wd:.3g}", "nm") for m in debug)
    # set by the user: not a default
    assert not any(mentions(m, "y_correction") for m in debug)


@pytest.fixture
def linked(
    add_prepared: Callable[..., Action],
    add_imaging: Callable[..., Action],
    add_correction: Callable[..., Action],
) -> tuple[Action, Action]:
    """A milling, the imaging it moves away from, and the correction linked to the milling."""
    milling = add_prepared("milling", "milling", slice_distance=10.0)
    add_imaging("imaging")
    correction = add_correction(["imaging"], linked_milling="milling")
    return milling, correction


def test_skipped_correction_warns_while_its_milling_mills(
    workflow: Workflow, linked: tuple[Action, Action]
) -> None:
    _, correction = linked

    warning = correction.skip_warning([correction])

    assert warning is not None
    assert mentions(warning, "milling", "imaging")
    # the workflow names the action warning, for the dialog and the log
    (line,) = workflow.skip_warnings([correction])
    assert mentions(line, "correction", warning)


def test_correction_skipped_with_its_milling_does_not_warn(
    linked: tuple[Action, Action],
) -> None:
    milling, correction = linked

    assert correction.skip_warning([milling, correction]) is None


def test_correction_without_its_milling_does_not_warn(
    add_imaging: Callable[..., Action], add_correction: Callable[..., Action]
) -> None:
    add_imaging("imaging")
    correction = add_correction(["imaging"], linked_milling="gone")

    assert correction.skip_warning([correction]) is None

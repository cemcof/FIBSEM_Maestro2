# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy import ndimage

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.drift import Drift
from fibsem_maestro.core.image import Image8Bit
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.drift_correction.drift_calculation_mode import TemplateMatchingDrift
from fibsem_maestro.drift_correction.drift_correction import DriftCorrectionState
from fibsem_maestro.logging.image.memory import MemoryImageLogger
from fibsem_maestro.logging.text.memory import MemoryTextLogger
from fibsem_maestro.microscope.positioning_outcome import PositioningOutcome
from fibsem_maestro.settings.drift_correction_settings import (
    TemplateMatchingDriftCorrection,
)
from fibsem_maestro.settings.template_matching_settings import (
    ReducedAreaMode,
    StandardMode,
    SubpixelMode,
    TemplateMatchingSettings,
)
from fibsem_maestro.store.image.memory import MemoryImageStore
from fibsem_maestro.template_matching.area_provider import AreaProvider
from fibsem_maestro.template_matching.error import TemplateMatchingError
from fibsem_maestro.template_matching.state import FIELD_WIDTH_TOLERANCE
from fibsem_maestro.template_matching.template_matching import TemplateMatching
from fibsem_maestro.workflow.error import ActionError

if TYPE_CHECKING:
    from fibsem_maestro.action.action import Action
    from fibsem_maestro.workflow.workflow import Workflow

PIXEL_SIZE_NM = 10.0
AREA = RelativeArea(origin=RelativePoint(0.4, 0.4), width=0.2, height=0.2)


def _texture() -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(0)
    base = ndimage.gaussian_filter(rng.random((600, 600)), 3)
    return (base - base.min()) / (base.max() - base.min()) * 255


TEXTURE = _texture()


def _scene(dx: float = 0.0, dy: float = 0.0) -> Image8Bit:
    """The texture with its content moved by (dx, dy) px; +x right, +y down."""
    shifted = ndimage.shift(TEXTURE, (dy, dx), order=3, mode="nearest")
    return Image8Bit(np.clip(shifted, 0, 255).astype(np.uint8), PIXEL_SIZE_NM)


FIELD_WIDTH_NM = 6000.0
RESOLUTION = Resolution(600, 600)


class ScriptedAreaProvider(AreaProvider):
    """
    Serves pre-made frames in order, cropped as the full-frame provider does.

    Attributes:
        geometry: The field width (nm) and resolution `frame_geometry` reports.
    """

    def __init__(
        self,
        settings: TemplateMatchingSettings,
        txt_log: MemoryTextLogger,
        frames: list[Image8Bit],
    ) -> None:
        super().__init__(None, settings, txt_log)  # ty: ignore[invalid-argument-type]
        self._frames = list(frames)
        self.geometry = (FIELD_WIDTH_NM, RESOLUTION)

    def frame_geometry(self) -> tuple[float, Resolution]:
        return self.geometry

    @property
    def last_full_frame(self) -> None:
        return None

    def acquire_template_regions(self) -> list[Image8Bit]:
        frame = self._frames.pop(0)
        return [frame.crop(area) for area in self._settings.areas]

    def acquire_search_regions(self) -> list[Image8Bit]:
        frame = self._frames.pop(0)
        return [
            frame.crop_with_padding(area, self._settings.correction_margin)
            for area in self._settings.areas
        ]

    def acquire_full_frame(self) -> Image8Bit:
        return self._frames.pop(0)


def _matcher(
    settings: TemplateMatchingSettings,
    frames: list[Image8Bit],
    log: MemoryTextLogger | None = None,
    *,
    provider: ScriptedAreaProvider | None = None,
    image_store: MemoryImageStore[Image8Bit] | None = None,
) -> TemplateMatching:
    log = log or MemoryTextLogger(lambda: 0)
    return TemplateMatching(
        provider or ScriptedAreaProvider(settings, log, frames),
        settings,
        image_store or MemoryImageStore(lambda: 0, Image8Bit),
        log,
        MemoryImageLogger(lambda: 0),
    )


def _measured_shift_px(
    mode: StandardMode | SubpixelMode, dx: float, dy: float
) -> tuple[float, float]:
    settings = TemplateMatchingSettings(
        areas=[AREA], correction_margin=200.0, matching_mode=mode
    )
    matcher = _matcher(settings, [_scene(), _scene(dx, dy)])
    matcher.create_templates()
    drift = matcher.calculate_drift()
    assert drift.x is not None and drift.y is not None
    return drift.x / PIXEL_SIZE_NM, drift.y / PIXEL_SIZE_NM


@pytest.mark.parametrize("factor", [1.0, 2.0, 4.0])
def test_subpixel_match_without_drift_reports_no_drift(factor: float) -> None:
    dx, dy = _measured_shift_px(SubpixelMode(upsampling_factor=factor), 0.0, 0.0)
    assert dx == pytest.approx(0.0, abs=0.1)
    assert dy == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("factor", [1.0, 4.0])
def test_subpixel_match_recovers_fractional_shift(factor: float) -> None:
    dx, dy = _measured_shift_px(SubpixelMode(upsampling_factor=factor), 2.3, -1.6)
    assert dx == pytest.approx(2.3, abs=0.15)
    assert dy == pytest.approx(-1.6, abs=0.15)


def test_standard_match_recovers_integer_shift() -> None:
    # guards the sign convention that the sub-pixel fix must keep
    assert _measured_shift_px(StandardMode(), 3.0, -2.0) == (3.0, -2.0)


@pytest.mark.parametrize(
    "area",
    [
        RelativeArea(origin=RelativePoint(0.0, 0.0), width=0.2, height=0.2),
        RelativeArea(origin=RelativePoint(0.85, 0.85), width=0.15, height=0.15),
    ],
    ids=["top-left corner", "bottom-right corner"],
)
def test_reduced_area_drift_correction_near_frame_edge_measures_no_drift(
    workflow: Workflow, add_action: Callable[..., Action], area: RelativeArea
) -> None:
    add_action(
        "drift_correction",
        "dc",
        drift_calculation_mode=TemplateMatchingDriftCorrection(
            areas=[area],
            correction_margin=60.0,
            frame_grabbing_mode=ReducedAreaMode(),
        ),
    )
    for action in workflow.actions:
        action.prepare()

    workflow.run(last_slice=2)

    # the simulated sample does not drift, so no correction may be applied
    beam = workflow.microscope.electron_beam
    one_pixel = beam.pixel_size
    assert abs(beam.beam_shift.x) < one_pixel
    assert abs(beam.beam_shift.y) < one_pixel


def test_template_scans_that_drifted_apart_are_rejected() -> None:
    settings = TemplateMatchingSettings(
        areas=[AREA], template_scans=2, maximal_drift=1.0
    )
    matcher = _matcher(settings, [_scene(), _scene(8.0, 0.0)])
    with pytest.raises(TemplateMatchingError, match="drift"):
        matcher.create_templates()


def test_identical_template_scans_are_accepted() -> None:
    settings = TemplateMatchingSettings(
        areas=[AREA], template_scans=2, maximal_drift=1.0
    )
    matcher = _matcher(settings, [_scene(), _scene()])
    state = matcher.create_templates()
    assert matcher.templates_exist(state)


def test_template_scan_drift_limit_is_in_nanometers() -> None:
    # 3 px at 10 nm/px is 30 nm: above a 20 nm limit, below a 50 nm one
    too_strict = TemplateMatchingSettings(
        areas=[AREA], template_scans=2, maximal_drift=20.0
    )
    with pytest.raises(TemplateMatchingError, match="drift"):
        _matcher(too_strict, [_scene(), _scene(3.0, 0.0)]).create_templates()

    lenient = TemplateMatchingSettings(
        areas=[AREA], template_scans=2, maximal_drift=50.0
    )
    _matcher(lenient, [_scene(), _scene(3.0, 0.0)]).create_templates()


# away from the centre of the frame, which the whole-frame search must not assume
OFF_CENTRE_AREA = RelativeArea(origin=RelativePoint(0.1, 0.6), width=0.2, height=0.2)


def _whole_frame_drift_px(
    settings: TemplateMatchingSettings, dx: float, dy: float
) -> tuple[float | None, float | None]:
    matcher = _matcher(settings, [_scene(), _scene(dx, dy)])
    matcher.create_templates()
    drift = matcher.calculate_drift(whole_frame=True)
    if drift.x is None or drift.y is None:
        return None, None
    return drift.x / PIXEL_SIZE_NM, drift.y / PIXEL_SIZE_NM


def test_whole_frame_search_finds_a_shift_beyond_the_margin_and_maximal_drift() -> None:
    # 40 px is 400 nm: far beyond a 20 nm margin and a 50 nm maximal drift
    settings = TemplateMatchingSettings(
        areas=[OFF_CENTRE_AREA], correction_margin=20.0, maximal_drift=50.0
    )

    assert _whole_frame_drift_px(settings, 40.0, -25.0) == (40.0, -25.0)


def test_whole_frame_search_is_at_pixel_precision() -> None:
    settings = TemplateMatchingSettings(
        areas=[OFF_CENTRE_AREA], matching_mode=SubpixelMode(upsampling_factor=4.0)
    )

    dx, dy = _whole_frame_drift_px(settings, 2.3, -1.6)

    assert (dx, dy) == (2.0, -2.0)


def test_whole_frame_search_is_logged() -> None:
    log = MemoryTextLogger(lambda: 0)
    settings = TemplateMatchingSettings(areas=[AREA])
    matcher = _matcher(settings, [_scene(), _scene()], log)
    matcher.create_templates()

    matcher.calculate_drift(whole_frame=True)

    infos = [r.message.lower() for r in log.records[0] if r.level == "info"]
    assert any(
        "whole frame" in m and "maximal drift" in m and "pixel" in m for m in infos
    )


@pytest.mark.parametrize(
    ("outcome", "searches"),
    [
        (PositioningOutcome.BEAM_SHIFTED, [False]),
        (PositioningOutcome.STAGE_MOVED, [False, True]),
        (PositioningOutcome.STAGE_OFF_TARGET, [False, True]),
    ],
)
def test_drift_correction_searches_the_whole_frame_after_a_stage_move(
    workflow: Workflow,
    add_action: Callable[..., Action],
    monkeypatch: pytest.MonkeyPatch,
    outcome: PositioningOutcome,
    searches: list[bool],
) -> None:
    add_action(
        "drift_correction",
        "dc",
        drift_calculation_mode=TemplateMatchingDriftCorrection(
            areas=[AREA], correction_margin=60.0
        ),
    )
    for action in workflow.actions:
        action.prepare()

    searched: list[bool] = []
    calculate_drift = TemplateMatching.calculate_drift

    def recording(self: TemplateMatching, whole_frame: bool = False) -> Drift:
        searched.append(whole_frame)
        return calculate_drift(self, whole_frame)

    monkeypatch.setattr(TemplateMatching, "calculate_drift", recording)
    microscope = workflow.microscope
    add_beam_shift = microscope.add_beam_shift_with_verification
    # the first correction ends as `outcome`, later ones as usual
    outcomes = [outcome]
    monkeypatch.setattr(
        microscope,
        "add_beam_shift_with_verification",
        lambda delta, beam=None: (
            outcomes.pop() if outcomes else add_beam_shift(delta, beam)
        ),
    )

    workflow.run(last_slice=1)

    assert searched == searches


# three areas apart from each other; the last one is spoiled in `_spoiled_scene`
THREE_AREAS = [
    RelativeArea(origin=RelativePoint(0.1, 0.1), width=0.2, height=0.2),
    RelativeArea(origin=RelativePoint(0.6, 0.1), width=0.2, height=0.2),
    RelativeArea(origin=RelativePoint(0.35, 0.6), width=0.2, height=0.2),
]


def _spoiled_scene() -> Image8Bit:
    """The scene with the last of `THREE_AREAS` replaced by unrelated noise."""
    scene = _scene()
    area = THREE_AREAS[-1].to_pixels(scene.resolution)
    rng = np.random.default_rng(1)
    rows = slice(area.origin.y, area.origin.y + area.height)
    cols = slice(area.origin.x, area.origin.x + area.width)
    scene[rows, cols] = rng.integers(0, 256, scene[rows, cols].shape)
    return scene


@pytest.mark.parametrize(("required", "valid"), [(1, True), (2, True), (3, False)])
def test_drift_needs_min_valid_areas_valid_matches(required: int, valid: bool) -> None:
    # the spoiled area matches with a low confidence, so 2 of 3 matches are valid
    settings = TemplateMatchingSettings(
        areas=THREE_AREAS, min_confidence=0.8, min_valid_areas=required
    )
    matcher = _matcher(settings, [_scene(), _spoiled_scene()])
    matcher.create_templates()

    drift = matcher.calculate_drift()

    assert drift.is_valid() is valid
    if not valid:
        assert drift.reason is not None
        assert "2" in drift.reason and "3" in drift.reason


def test_matches_beyond_the_maximal_drift_are_not_valid_areas() -> None:
    # 8 px is 80 nm, beyond a 50 nm maximal drift, in every area
    settings = TemplateMatchingSettings(
        areas=THREE_AREAS, correction_margin=200.0, maximal_drift=50.0
    )
    matcher = _matcher(settings, [_scene(), _scene(8.0, 0.0)])
    matcher.create_templates()

    assert not matcher.calculate_drift().is_valid()


def _drift_correction(
    add_action: Callable[..., Action], areas: list[RelativeArea], required: int
) -> Action:
    return add_action(
        "drift_correction",
        "dc",
        drift_calculation_mode=TemplateMatchingDriftCorrection(
            areas=areas, min_valid_areas=required
        ),
    )


def test_requiring_more_valid_areas_than_defined_is_a_preparation_issue(
    add_action: Callable[..., Action],
) -> None:
    enough = _drift_correction(add_action, THREE_AREAS[:2], 2)
    too_many = _drift_correction(add_action, THREE_AREAS[:2], 3)

    issues = set(too_many.preparation_issues()) - set(enough.preparation_issues())

    assert len(issues) == 1
    (issue,) = issues
    assert "3" in issue and "2" in issue


@pytest.mark.parametrize("stop_at_failure", [True, False])
def test_failed_drift_correction_tells_why(
    workflow: Workflow,
    add_action: Callable[..., Action],
    monkeypatch: pytest.MonkeyPatch,
    stop_at_failure: bool,
) -> None:
    action = _drift_correction(add_action, [AREA], 1)
    action.settings.stop_at_failure = stop_at_failure
    for each in workflow.actions:
        each.prepare()
    reason = "1 of 3 areas gave a valid match; at least 2 required"
    monkeypatch.setattr(
        TemplateMatching,
        "calculate_drift",
        lambda *_, **__: Drift(None, None, reason=reason),
    )
    logged: list[str] = []
    monkeypatch.setattr(action.ctx.text_logger, "warning", logged.append)

    if stop_at_failure:
        with pytest.raises(ActionError) as error:
            workflow.run(last_slice=1)
        assert reason in str(error.value)
    else:
        workflow.run(last_slice=1)
        assert any(reason in message for message in logged)


TWO_AREAS = THREE_AREAS[:2]
MOVED_AREA = RelativeArea(origin=RelativePoint(0.15, 0.1), width=0.2, height=0.2)

Change = Callable[[TemplateMatchingSettings, "ScriptedAreaProvider"], None]


def _move_first_area(settings: TemplateMatchingSettings, *_: object) -> None:
    settings.areas = [MOVED_AREA, *TWO_AREAS[1:]]


def _add_area(settings: TemplateMatchingSettings, *_: object) -> None:
    settings.areas = list(THREE_AREAS)


def _remove_area(settings: TemplateMatchingSettings, *_: object) -> None:
    settings.areas = TWO_AREAS[:1]


def _widen_field(_: object, provider: ScriptedAreaProvider) -> None:
    provider.geometry = (2 * FIELD_WIDTH_NM, RESOLUTION)


def _halve_resolution(_: object, provider: ScriptedAreaProvider) -> None:
    provider.geometry = (FIELD_WIDTH_NM, Resolution(300, 300))


CHANGES = pytest.mark.parametrize(
    "change",
    [_move_first_area, _add_area, _remove_area, _widen_field, _halve_resolution],
    ids=["area moved", "area added", "area removed", "field width", "resolution"],
)


def _two_area_matcher(
    image_store: MemoryImageStore[Image8Bit] | None = None,
) -> tuple[TemplateMatching, TemplateMatchingSettings, ScriptedAreaProvider]:
    """A matcher for `TWO_AREAS`, with its settings and area provider."""
    settings = TemplateMatchingSettings(areas=list(TWO_AREAS))
    log = MemoryTextLogger(lambda: 0)
    provider = ScriptedAreaProvider(settings, log, [_scene()] * 8)
    matcher = _matcher(settings, [], log, provider=provider, image_store=image_store)
    return matcher, settings, provider


def test_created_templates_fit_until_something_changes() -> None:
    matcher, _, provider = _two_area_matcher()
    state = matcher.create_templates()
    assert matcher.outdated_reason(state) is None

    # a change within the tolerance
    provider.geometry = (FIELD_WIDTH_NM * (1 + FIELD_WIDTH_TOLERANCE / 2), RESOLUTION)
    assert matcher.outdated_reason(state) is None


@CHANGES
def test_templates_no_longer_fit_after_a_change(change: Change) -> None:
    matcher, settings, provider = _two_area_matcher()
    state = matcher.create_templates()

    change(settings, provider)

    assert matcher.outdated_reason(state) is not None


def test_templates_without_a_state_do_not_fit() -> None:
    matcher, _, _ = _two_area_matcher()
    matcher.create_templates()

    assert not matcher.templates_exist(None)
    assert matcher.outdated_reason(None) is not None


def test_deleted_templates_do_not_fit() -> None:
    matcher, _, _ = _two_area_matcher()
    state = matcher.create_templates()

    matcher.delete_templates()

    assert not matcher.templates_exist(state)
    assert matcher.outdated_reason(state) is not None


@CHANGES
def test_copying_copies_the_stored_templates_whatever_the_areas(change: Change) -> None:
    image_store = MemoryImageStore(lambda: 0, Image8Bit)
    matcher, settings, provider = _two_area_matcher(image_store)
    state = matcher.create_templates()
    change(settings, provider)

    matcher.copy_templates(image_store, image_store.next)

    assert matcher.templates_exist(state, image_store.next)
    # only the stored ones: nothing beyond them for `_delete_templates` to find
    assert not image_store.next.exists(f"template_{len(TWO_AREAS)}.tif")


def test_copying_replaces_the_templates_already_there() -> None:
    image_store = MemoryImageStore(lambda: 0, Image8Bit)
    matcher, settings, _ = _two_area_matcher(image_store)
    settings.areas = list(THREE_AREAS)
    matcher.create_templates(image_store.next)
    settings.areas = list(TWO_AREAS)
    matcher.create_templates()

    matcher.copy_templates(image_store, image_store.next)

    assert not image_store.next.exists(f"template_{len(TWO_AREAS)}.tif")


def test_template_update_tells_what_new_templates_are_for() -> None:
    matcher, settings, provider = _two_area_matcher()
    matcher.create_templates()
    _move_first_area(settings, provider)
    settings.update_confidence_limit = 0.5

    # a confidence below the limit forces new templates
    acquired = matcher.update_templates(1, confidence=0.1)
    copied = matcher.update_templates(1, confidence=None)

    assert acquired is not None
    assert acquired.areas == settings.areas
    assert copied is None


def _template_drift_correction(add_action: Callable[..., Action]) -> Action:
    return add_action(
        "drift_correction",
        "dc",
        drift_calculation_mode=TemplateMatchingDriftCorrection(
            areas=list(TWO_AREAS), correction_margin=60.0
        ),
    )


def _mode(action: Action) -> TemplateMatchingDriftCorrection:
    return action.settings.drift_calculation_mode


def _keep(action: Action) -> None:
    _ = action


def _reset(action: Action) -> None:
    action.reset_state()


def _forget_the_state(action: Action) -> None:
    # e.g. a workflow whose templates were acquired before the state recorded them
    action.set_state(DriftCorrectionState())


def _move_area_of(action: Action) -> None:
    _mode(action).areas = [MOVED_AREA, *TWO_AREAS[1:]]


def _add_area_to(action: Action) -> None:
    _mode(action).areas = list(THREE_AREAS)


def _remove_area_of(action: Action) -> None:
    _mode(action).areas = TWO_AREAS[:1]


def _change_the_margin(action: Action) -> None:
    _mode(action).correction_margin = 70.0


def _replace_the_mode_settings(action: Action) -> None:
    action.settings.drift_calculation_mode = TemplateMatchingDriftCorrection(
        areas=list(TWO_AREAS), correction_margin=60.0
    )


def _change_the_field_width(action: Action) -> None:
    # the drift correction keeps the field width it finds on the microscope
    action.microscope.electron_beam.horizontal_field_width *= 2


@dataclass
class _Run:
    """What the drift correction did per slice: the slices it ran, measured, ..."""

    executed: list[int] = field(default_factory=list)
    measured: list[int] = field(default_factory=list)
    corrected: list[int] = field(default_factory=list)
    warned: list[int] = field(default_factory=list)


@pytest.fixture
def run_log(
    workflow: Workflow, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Action], _Run]:
    """Records, by slice, what a drift correction does in `workflow.run`."""

    def watch(action: Action) -> _Run:
        run = _Run()
        after = TemplateMatchingDrift.after_calculate_drift
        calculate = TemplateMatching.calculate_drift
        microscope = workflow.microscope
        add_beam_shift = microscope.add_beam_shift_with_verification
        log = action.ctx.text_logger
        warning = log.warning

        def after_recording(self: TemplateMatchingDrift, slice_number: int) -> None:
            run.executed.append(action.ctx.slice)
            after(self, slice_number)

        def calculate_recording(
            self: TemplateMatching, whole_frame: bool = False
        ) -> Drift:
            run.measured.append(action.ctx.slice)
            return calculate(self, whole_frame)

        def shift_recording(delta: Any, beam: Any = None) -> PositioningOutcome:
            run.corrected.append(action.ctx.slice)
            return add_beam_shift(delta, beam)

        def warning_recording(message: str) -> None:
            run.warned.append(action.ctx.slice)
            warning(message)

        monkeypatch.setattr(
            TemplateMatchingDrift, "after_calculate_drift", after_recording
        )
        monkeypatch.setattr(TemplateMatching, "calculate_drift", calculate_recording)
        monkeypatch.setattr(
            microscope, "add_beam_shift_with_verification", shift_recording
        )
        monkeypatch.setattr(log, "warning", warning_recording)
        return run

    return watch


@pytest.mark.parametrize(
    ("change", "refit"),
    [
        (_keep, False),
        (_change_the_margin, False),
        (_replace_the_mode_settings, False),
        (_reset, True),
        (_forget_the_state, True),
        (_move_area_of, True),
        (_add_area_to, True),
        (_remove_area_of, True),
        (_change_the_field_width, True),
    ],
    ids=[
        "unchanged",
        "margin changed",
        "mode settings replaced",
        "reset state",
        "no state",
        "area moved",
        "area added",
        "area removed",
        "field width",
    ],
)
def test_drift_correction_acquires_templates_instead_of_measuring_when_they_do_not_fit(
    workflow: Workflow,
    add_action: Callable[..., Action],
    run_log: Callable[[Action], _Run],
    change: Callable[[Action], None],
    refit: bool,
) -> None:
    action = _template_drift_correction(add_action)
    for each in workflow.actions:
        each.prepare()
    change(action)
    run = run_log(action)

    workflow.run(last_slice=2)

    assert len(run.executed) == 2
    first, second = run.executed
    if refit:
        # the first execution only acquires the templates, the second uses them
        assert run.measured == [second]
        assert first not in run.corrected
        assert first in run.warned
    else:
        assert run.measured == run.executed
        assert not run.warned


def test_drift_correction_resumes_from_its_stored_state(
    workflow: Workflow,
    add_action: Callable[..., Action],
    run_log: Callable[[Action], _Run],
) -> None:
    action = _template_drift_correction(add_action)
    for each in workflow.actions:
        each.prepare()
    workflow.run(last_slice=1)
    stored = action.ctx.state_store.read("state.yaml", DriftCorrectionState)

    # as if the workflow were loaded again
    action.set_state(DriftCorrectionState())
    action.set_state(stored)
    run = run_log(action)
    workflow.run(last_slice=2)

    assert run.measured == run.executed
    assert not run.warned


def test_prepare_stores_the_state_of_the_templates(
    workflow: Workflow, add_action: Callable[..., Action]
) -> None:
    action = _template_drift_correction(add_action)

    action.prepare()

    stored = action.ctx.state_store.read("state.yaml", DriftCorrectionState)
    assert stored != DriftCorrectionState()
    assert stored == action.state
    _ = workflow


def test_reset_state_discards_the_state_and_the_templates(
    workflow: Workflow, add_action: Callable[..., Action]
) -> None:
    action = _template_drift_correction(add_action)
    for each in workflow.actions:
        each.prepare()
    store = action.ctx.image_store(Image8Bit)
    assert store.exists("template_0.tif")

    action.reset_state()

    assert action.state == DriftCorrectionState()
    stored = action.ctx.state_store.read("state.yaml", DriftCorrectionState)
    assert stored == DriftCorrectionState()
    assert not store.exists("template_0.tif")
    assert action.preparation_issues() == []


def test_reset_state_that_cannot_delete_the_templates_still_resets(
    workflow: Workflow,
    add_action: Callable[..., Action],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _template_drift_correction(add_action)
    for each in workflow.actions:
        each.prepare()

    def locked(*_: object) -> None:
        raise PermissionError("the template is open elsewhere")

    monkeypatch.setattr(TemplateMatching, "delete_templates", locked)
    warned: list[str] = []
    monkeypatch.setattr(action.ctx.text_logger, "warning", warned.append)

    action.reset_state()

    assert action.state == DriftCorrectionState()
    stored = action.ctx.state_store.read("state.yaml", DriftCorrectionState)
    assert stored == DriftCorrectionState()
    assert warned


def test_templates_acquired_by_an_update_become_the_state(
    workflow: Workflow, add_action: Callable[..., Action]
) -> None:
    action = _template_drift_correction(add_action)
    for each in workflow.actions:
        each.prepare()
    _move_area_of(action)
    _mode(action).update_frequency = 1

    # a skipped slice updates the templates of the next one
    action.carry_over_to_next()

    state = action.state.drift_calculation
    assert state is not None
    assert state.areas == _mode(action).areas

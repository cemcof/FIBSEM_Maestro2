# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy import ndimage

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.drift import Drift
from fibsem_maestro.core.image import Image8Bit
from fibsem_maestro.core.point import RelativePoint
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
from fibsem_maestro.template_matching.template_matching import TemplateMatching
from fibsem_maestro.workflow.error import ActionError

if TYPE_CHECKING:
    from collections.abc import Callable

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


class ScriptedAreaProvider(AreaProvider):
    """Serves pre-made frames in order, cropped as the full-frame provider does."""

    def __init__(
        self,
        settings: TemplateMatchingSettings,
        txt_log: MemoryTextLogger,
        frames: list[Image8Bit],
    ) -> None:
        super().__init__(None, settings, txt_log)  # ty: ignore[invalid-argument-type]
        self._frames = list(frames)

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
) -> TemplateMatching:
    log = log or MemoryTextLogger(lambda: 0)
    return TemplateMatching(
        ScriptedAreaProvider(settings, log, frames),
        settings,
        MemoryImageStore(lambda: 0, Image8Bit),
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
    matcher.create_templates()
    assert matcher.templates_exist()


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

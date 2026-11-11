# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Defaults computed from stored properties: the detail band, the working-distance range."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fibsem_maestro.action.defaults import ActionDefaults
from fibsem_maestro.autofunction.sweep_space import make_sweep_space
from fibsem_maestro.autofunction.sweeping import Sweeping
from fibsem_maestro.core.detail_band import DetailBand
from fibsem_maestro.core.resolution import Resolution
from fibsem_maestro.imaging.imaging import Imaging
from fibsem_maestro.logging.text.memory import MemoryTextLogger
from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.settings.autofunction_settings import BasicMode
from fibsem_maestro.settings.criterion_settings import BandpassMetric, CriterionSettings
from fibsem_maestro.settings.dynamic_default import (
    DefaultUnavailableError,
    effective_value,
    unavailable_defaults,
    with_defaults,
)
from fibsem_maestro.settings.sweeping_settings import LinearSweeping

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.workflow.workflow import Workflow

HFW_NM = 10_000.0
PIXEL_NM = 5.0


def stored(props: BeamProperties | None) -> ActionDefaults:
    """A context whose action stores `props` on its beam."""
    return ActionDefaults(None, (), lambda: props)


def test_detail_band_is_60_to_100_times_the_stored_pixel_size() -> None:
    metric = BandpassMetric()

    band = effective_value(
        metric, "detail", stored(BeamProperties(pixel_size=PIXEL_NM))
    )

    assert band == DetailBand(low=60 * PIXEL_NM, high=100 * PIXEL_NM)


def test_detail_band_pixel_size_falls_back_to_field_width_over_resolution() -> None:
    props = BeamProperties(
        horizontal_field_width=HFW_NM, resolution=Resolution(width=2000, height=1000)
    )

    band = effective_value(BandpassMetric(), "detail", stored(props))

    # 10 µm over 2000 px: 5 nm pixels
    assert band == DetailBand(low=300.0, high=500.0)


def test_detail_band_set_by_the_user_ignores_the_pixel_size() -> None:
    metric = BandpassMetric(detail=DetailBand(low=1.0, high=2.0))

    band = effective_value(metric, "detail", stored(BeamProperties(pixel_size=1.0)))

    assert band == DetailBand(low=1.0, high=2.0)


def test_working_distance_range_is_a_twentieth_of_the_stored_field_width() -> None:
    sweeping = LinearSweeping(target_attribute="working_distance")

    sweep_range = effective_value(
        sweeping, "range", stored(BeamProperties(horizontal_field_width=HFW_NM))
    )

    assert sweep_range == pytest.approx((-500.0, 500.0))


def test_range_of_another_attribute_has_no_default() -> None:
    # a range derived from the field width would be meaningless for it
    sweeping = LinearSweeping(target_attribute="dwell_time")
    context = stored(BeamProperties(horizontal_field_width=HFW_NM))

    assert effective_value(sweeping, "range", context) is None
    assert unavailable_defaults(sweeping, context) == [
        (
            "range",
            "There is no default range for sweeping 'dwell_time'; set a range.",
        )
    ]


@pytest.mark.parametrize(
    ("props", "reason"),
    [
        (
            None,
            (
                "Computed from the field width once the action is prepared; or set a range."
            ),
        ),
        (
            BeamProperties(working_distance=4e6),
            (
                "The stored properties lack the field width ('horizontal_field_width'); "
                "collect it and prepare the action again, or set a range."
            ),
        ),
    ],
)
def test_range_says_which_properties_it_waits_for(
    props: BeamProperties | None, reason: str
) -> None:
    sweeping = LinearSweeping(target_attribute="working_distance")

    assert unavailable_defaults(sweeping, stored(props)) == [("range", reason)]


def test_with_defaults_fills_nested_defaults_and_leaves_the_settings_alone() -> None:
    mode = BasicMode(sweeping=LinearSweeping(target_attribute="working_distance"))
    props = BeamProperties(horizontal_field_width=HFW_NM, pixel_size=PIXEL_NM)

    resolved = with_defaults(mode, stored(props))

    assert isinstance(resolved.sweeping, LinearSweeping)
    assert resolved.sweeping.range == pytest.approx((-500.0, 500.0))
    assert isinstance(resolved.criterion.sharpness_metric, BandpassMetric)
    assert resolved.criterion.sharpness_metric.detail == DetailBand(300.0, 500.0)
    # still following
    assert isinstance(mode.sweeping, LinearSweeping)
    assert mode.sweeping.range is None
    assert isinstance(mode.criterion.sharpness_metric, BandpassMetric)
    assert mode.criterion.sharpness_metric.detail is None


def test_with_defaults_names_every_default_it_cannot_compute() -> None:
    mode = BasicMode(sweeping=LinearSweeping(target_attribute="working_distance"))

    with pytest.raises(DefaultUnavailableError) as raised:
        with_defaults(mode, stored(None))

    assert "'sweeping.range'" in str(raised.value)
    assert "'criterion.sharpness_metric.detail'" in str(raised.value)


def test_sweep_space_takes_the_range_of_the_properties_stored_now() -> None:
    sweeping = LinearSweeping(target_attribute="working_distance", steps=3)
    props = BeamProperties(horizontal_field_width=HFW_NM)
    context = ActionDefaults(None, (), lambda: props)

    def offsets() -> list[float]:
        space = make_sweep_space(sweeping, context)
        return [value for (value,) in space.candidates((0.0,), reverse=False)]

    first = offsets()
    props = BeamProperties(horizontal_field_width=2 * HFW_NM)
    second = offsets()

    assert first == pytest.approx([-500.0, 0.0, 500.0])
    assert second == pytest.approx([-1000.0, 0.0, 1000.0])


@pytest.fixture
def autofunction(add_action: Callable[..., Action]) -> Action:
    """A basic working-distance autofunction whose range and detail band follow their defaults."""
    add_action("imaging", "imaging")
    return add_action(
        "autofunction",
        "focus",
        linked_imaging="imaging",
        mode=BasicMode(sweeping=LinearSweeping(target_attribute="working_distance")),
    )


def test_unprepared_action_reports_the_defaults_waiting_for_properties(
    autofunction: Action,
) -> None:
    issues = autofunction.preparation_issues()

    assert issues == [
        (
            "'mode.sweeping.range': Computed from the field width once the action is prepared; or set a range."
        ),
        (
            "'mode.criterion.sharpness_metric.detail': Computed from the pixel size "
            "once the action is prepared; or set a value."
        ),
    ]


def test_action_computes_its_defaults_from_its_stored_properties(
    autofunction: Action,
) -> None:
    autofunction.write_properties(
        GlobalProperties(
            electron_beam=BeamProperties(
                horizontal_field_width=HFW_NM, pixel_size=PIXEL_NM
            )
        )
    )
    settings = autofunction.settings
    assert isinstance(settings.mode, BasicMode)

    resolved = with_defaults(settings.mode, autofunction.default_context())

    assert autofunction.preparation_issues() == []
    assert isinstance(resolved.sweeping, LinearSweeping)
    assert resolved.sweeping.range == pytest.approx((-500.0, 500.0))


def test_action_ignores_the_properties_of_another_beam(autofunction: Action) -> None:
    # the autofunction works on the electron beam
    autofunction.write_properties(
        GlobalProperties(ion_beam=BeamProperties(horizontal_field_width=HFW_NM))
    )

    assert autofunction.stored_beam_properties() is None


def test_refocus_computes_its_defaults_from_the_imaging_properties(
    add_action: Callable[..., Action],
) -> None:
    imaging = add_action("imaging", "imaging")
    refocus = add_action(
        "refocus",
        "refocus",
        linked_imaging="imaging",
        mode=BasicMode(
            sweeping=LinearSweeping(target_attribute="working_distance"),
            criterion=CriterionSettings(
                sharpness_metric=BandpassMetric(detail=DetailBand(1.0, 1.0))
            ),
        ),
    )
    assert refocus.preparation_issues() != []

    # the refocus stores no properties of its own
    imaging.write_properties(
        GlobalProperties(electron_beam=BeamProperties(horizontal_field_width=HFW_NM))
    )

    assert refocus.preparation_issues() == []
    assert refocus.stored_beam_properties() == BeamProperties(
        horizontal_field_width=HFW_NM
    )


@pytest.mark.parametrize(
    ("props", "calculated"),
    [
        (BeamProperties(pixel_size=PIXEL_NM), True),
        # the band cannot be computed: no sharpness, rather than a crashed thread
        (BeamProperties(working_distance=4e6), False),
    ],
)
def test_imaging_scores_its_frames_with_the_band_of_its_stored_pixel_size(
    add_action: Callable[..., Action], props: BeamProperties, calculated: bool
) -> None:
    imaging = add_action("imaging", "imaging", criterion=CriterionSettings())
    assert isinstance(imaging, Imaging)
    imaging.write_properties(GlobalProperties(electron_beam=props))

    imaging.execute()

    assert (imaging.wait_for_sharpness() is not None) is calculated


class _FakeBeam:
    """A beam with a working distance and wide limits."""

    def __init__(self) -> None:
        self.working_distance = 4e6

    def limits(self, name: str) -> tuple[float, float]:
        _ = name
        return (0.0, 1e9)


def test_every_sweep_takes_the_range_of_the_properties_stored_when_it_starts() -> None:
    props = BeamProperties(horizontal_field_width=HFW_NM)
    sweeping = Sweeping(
        _FakeBeam(),  # ty: ignore[invalid-argument-type]
        LinearSweeping(target_attribute="working_distance", steps=3),
        MemoryTextLogger(lambda: 0),
        ActionDefaults(None, (), lambda: props),
    )

    first = [step.value[0] - 4e6 for step in sweeping.sweep()]
    # e.g. propagated to the next slice
    props = BeamProperties(horizontal_field_width=2 * HFW_NM)
    second = [step.value[0] - 4e6 for step in sweeping.sweep()]

    assert first == pytest.approx([-500.0, 0.0, 500.0])
    assert second == pytest.approx([-1000.0, 0.0, 1000.0])


def test_autofunction_sweeps_and_scores_with_the_defaults_of_its_properties(
    workflow: Workflow, add_action: Callable[..., Action]
) -> None:
    add_action("imaging", "imaging")
    autofunction = add_action(
        "autofunction",
        "focus",
        linked_imaging="imaging",
        mode=BasicMode(
            sweeping=LinearSweeping(target_attribute="working_distance", steps=3)
        ),
    )
    autofunction.write_properties(
        GlobalProperties(
            electron_beam=BeamProperties(
                horizontal_field_width=HFW_NM,
                pixel_size=PIXEL_NM,
                working_distance=4e6,
            )
        )
    )

    # the range and the detail band follow their defaults
    autofunction.test()

    best = workflow.microscope.electron_beam.working_distance
    assert min(abs(best - wd) for wd in (4e6 - 500.0, 4e6, 4e6 + 500.0)) < 1e-3

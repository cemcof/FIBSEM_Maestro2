# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Sharpness values, their direction, and the direction of every metric."""

from typing import get_args

import pytest

from fibsem_maestro.criterion.functions import CRITERION_METRICS
from fibsem_maestro.criterion.sharpness import Better, Sharpness, mean_of
from fibsem_maestro.settings.criterion_settings import SharpnessMetric


def _metric_types() -> list[str]:
    """The `type` of every metric the settings offer."""
    (union, *_) = get_args(SharpnessMetric)
    return [variant.model_fields["type"].default for variant in get_args(union)]


@pytest.mark.parametrize("key", _metric_types())
def test_every_metric_of_the_settings_is_registered(key: str) -> None:
    assert key in CRITERION_METRICS


def test_frc_is_a_resolution_in_nm_lower_is_sharper() -> None:
    metric = CRITERION_METRICS.get("frc")

    assert (metric.better, metric.unit) == (Better.LOWER, "nm")


@pytest.mark.parametrize("key", ["bandpass", "bandpass_var", "fft"])
def test_band_metrics_are_higher_is_sharper(key: str) -> None:
    assert CRITERION_METRICS.get(key).better is Better.HIGHER


@pytest.mark.parametrize(
    ("better", "sharper", "blurrier"),
    [(Better.HIGHER, 2.0, 1.0), (Better.LOWER, 3.0, 6.0)],
)
def test_sharpness_score_is_higher_for_the_sharper_value(
    better: Better, sharper: float, blurrier: float
) -> None:
    assert Sharpness(sharper, better).score > Sharpness(blurrier, better).score


@pytest.mark.parametrize(
    ("better", "value", "less_sharp"),
    [
        (Better.HIGHER, 0.5, True),
        (Better.HIGHER, 1.5, False),
        (Better.LOWER, 5.0, True),
        (Better.LOWER, 3.0, False),
    ],
)
def test_sharpness_is_less_sharp_than_a_limit_on_the_metric_side(
    better: Better, value: float, less_sharp: bool
) -> None:
    limit = 1.0 if better is Better.HIGHER else 4.0

    assert Sharpness(value, better).is_less_sharp_than(limit) is less_sharp


def test_mean_of_keeps_the_metric() -> None:
    mean = mean_of(
        [Sharpness(2.0, Better.LOWER, "nm"), Sharpness(4.0, Better.LOWER, "nm")]
    )

    assert mean == Sharpness(3.0, Better.LOWER, "nm")


def test_mean_of_different_metrics_raises() -> None:
    with pytest.raises(ValueError):
        mean_of([Sharpness(2.0, Better.LOWER, "nm"), Sharpness(4.0, Better.HIGHER)])


def test_sharpness_text_has_its_unit() -> None:
    assert "nm" in str(Sharpness(4.2, Better.LOWER, "nm"))

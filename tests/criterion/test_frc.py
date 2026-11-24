# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The single-image FRC resolution, against a two-image FRC of the same object."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from fibsem_maestro.criterion.frc import (
    FRC_THRESHOLD,
    _first_crossing,
    _ring_correlation,
    frc_resolution,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

SIDE = 512


def _object(blur_px: float, seed: int = 0) -> NDArray[Any]:
    """A random texture blurred by a Gaussian of `blur_px` pixels."""
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.standard_normal((SIDE, SIDE)), blur_px)
    return texture / texture.std()


def _noisy(image: NDArray[Any], noise: float, seed: int) -> NDArray[Any]:
    return image + noise * np.random.default_rng(seed).standard_normal(image.shape)


def _two_image_resolution(a: NDArray[Any], b: NDArray[Any], pixel_size: float) -> float:
    """The FRC resolution of two independent recordings of one object, in nm."""
    window = np.outer(np.hanning(SIDE), np.hanning(SIDE))
    frc = _ring_correlation((a - a.mean()) * window, (b - b.mean()) * window)
    return SIDE * pixel_size / _first_crossing(frc, FRC_THRESHOLD)


@pytest.mark.parametrize("blur_px", [3.0, 5.0])
@pytest.mark.parametrize("noise", [0.05, 0.2])
def test_frc_resolution_agrees_with_a_two_image_frc(
    blur_px: float, noise: float
) -> None:
    obj = _object(blur_px)
    first, second = _noisy(obj, noise, 1), _noisy(obj, noise, 2)

    single = frc_resolution(first, 1.0)
    reference = _two_image_resolution(first, second, 1.0)

    # converting with sqrt(2) * pixel size gave about 0.77 of the reference;
    # the diagonal offset of the half-images leaves a small overestimate
    assert 0.95 <= single / reference <= 1.2


def test_frc_resolution_is_in_nm() -> None:
    image = _noisy(_object(4.0), 0.1, 1)

    assert frc_resolution(image, 2.0) == pytest.approx(2.0 * frc_resolution(image, 1.0))


def test_frc_resolution_of_a_very_sharp_image_is_about_four_pixels() -> None:
    # detail finer than the half-images sample reads as their limit
    image = _noisy(_object(1.0), 0.01, 1)

    assert 4.0 <= frc_resolution(image, 1.0) <= 4.8


def test_frc_resolution_is_lower_for_a_sharper_image() -> None:
    sharp = _noisy(_object(2.5), 0.1, 1)
    blurred = _noisy(_object(5.0), 0.1, 1)

    assert frc_resolution(sharp, 1.0) < frc_resolution(blurred, 1.0)

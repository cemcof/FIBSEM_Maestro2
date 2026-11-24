# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

FRC_THRESHOLD = 1.0 / 7.0
"""FRC value below which a ring counts as noise (the 1/7 criterion)."""

MIN_FRC_SIDE_PX = 16
"""Smallest image side, in pixels; 8 px half-images still give 4 rings."""


def frc_resolution(img: NDArray[Any], pixel_size: float) -> float:
    """
    Return the resolution at which an image's single-image FRC drops below 1/7.

    The image is split into two half-images of diagonal pixel neighbours.
    The largest centred square of even side is used, so that both half-images
    are square and equal. Each half-image has its mean removed and a 2-D Hann
    window applied to suppress edge effects.

    Args:
        img: 2-D image.
        pixel_size: Pixel size of `img`, in nm.

    Returns:
        The resolution, in nm; lower means sharper. If the FRC never drops
        below the threshold, the finest resolution the half-images sample,
        about 4 pixels, is returned.

    Raises:
        ValueError: If the image is not 2-D, is smaller than `MIN_FRC_SIDE_PX`
            on a side, has no structure (a constant image), or its FRC drops
            below the threshold at the lowest frequency.
    """
    if img.ndim != 2:
        raise ValueError(f"FRC needs a 2-D image, got {img.ndim} dimensions.")

    side = min(img.shape) // 2 * 2
    if side < MIN_FRC_SIDE_PX:
        raise ValueError(
            f"FRC needs at least {MIN_FRC_SIDE_PX}x{MIN_FRC_SIDE_PX} pixels, "
            f"got {img.shape[1]}x{img.shape[0]}."
        )

    top = (img.shape[0] - side) // 2
    left = (img.shape[1] - side) // 2
    square = np.asarray(img[top : top + side, left : left + side], dtype=np.float64)

    half_side = side // 2
    window = np.outer(np.hanning(half_side), np.hanning(half_side))
    half_a = (square[0::2, 0::2] - square[0::2, 0::2].mean()) * window
    half_b = (square[1::2, 1::2] - square[1::2, 1::2].mean()) * window

    frc = _ring_correlation(half_a, half_b)
    if not np.all(np.isfinite(frc)):
        raise ValueError("FRC is undefined for an image without structure.")

    crossing = _first_crossing(frc, FRC_THRESHOLD)
    if crossing <= 0.0:
        raise ValueError("FRC shows no signal shared by the half-images.")

    # ring r of an n-pixel half-image lies at r / n cycles per half-image pixel;
    # a half-image takes every other row and column, so its pixels are 2 *
    # pixel_size apart (the two half-images are offset diagonally, but that
    # offset is not their sampling distance)
    cycles_per_half_px = crossing / half_side
    return float(2.0 * pixel_size / cycles_per_half_px)


def _ring_correlation(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the FRC of two equal square images for rings up to the inscribed circle.

    Args:
        a: First image, n x n.
        b: Second image, n x n.

    Returns:
        One FRC value per ring of integer radius 0 .. n/2 - 1, measured in
        frequency samples from the zero frequency.
    """
    n = a.shape[0]
    fa = np.fft.fftshift(np.fft.fft2(a))
    fb = np.fft.fftshift(np.fft.fft2(b))

    y, x = np.indices(a.shape)
    radius = np.rint(np.hypot(y - n // 2, x - n // 2)).astype(np.intp)
    inside = radius < n // 2
    rings = radius[inside]

    cross = np.bincount(rings, np.real(fa * np.conj(fb))[inside], minlength=n // 2)
    power_a = np.bincount(rings, (np.abs(fa) ** 2)[inside], minlength=n // 2)
    power_b = np.bincount(rings, (np.abs(fb) ** 2)[inside], minlength=n // 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cross / np.sqrt(power_a * power_b)


def _first_crossing(frc: NDArray[np.float64], threshold: float) -> float:
    """
    Return the fractional ring index at which the FRC first drops below a threshold.

    Args:
        frc: FRC per ring, starting at the zero frequency.
        threshold: The threshold.

    Returns:
        The crossing as a fractional ring index, or the last ring's index if
        the FRC never drops below the threshold.
    """
    below = np.nonzero(frc[1:] < threshold)[0]
    if below.size == 0:
        return float(len(frc) - 1)

    i = int(below[0]) + 1
    above, under = frc[i - 1], frc[i]
    return (i - 1) + (above - threshold) / (above - under)

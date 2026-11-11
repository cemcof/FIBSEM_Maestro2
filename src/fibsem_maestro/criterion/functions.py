# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

from fibsem_maestro.core.registry import Registry
from fibsem_maestro.criterion.error import CriterionError
from fibsem_maestro.criterion.frc import frc_cutoff_frequency
from fibsem_maestro.settings.criterion_settings import (
    SharpnessMetric,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fibsem_maestro.core.detail_band import DetailBand
    from fibsem_maestro.core.image import Image
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.settings.criterion_settings import (
        DetailBandMetric,
        FrcMetric,
    )


def _detail(metric: DetailBandMetric) -> DetailBand:
    """
    The metric's detail band, which must have been computed if it follows its default.

    Args:
        metric: The metric's settings, with their defaults computed
            (`settings.dynamic_default.with_defaults`).

    Returns:
        The detail band, in nm.

    Raises:
        CriterionError: If the band still follows its default.
    """
    if metric.detail is None:
        raise CriterionError(
            "The detail band follows its default, which was not computed; "
            "compute the defaults of the criterion's settings first."
        )
    return metric.detail


CriterionFunction = Callable[["Image", SharpnessMetric, "TextLogger"], np.floating]
"""A metric: image, the metric's own settings variant, and a logger to a sharpness."""

# registry of criterion functions
CRITERION_FUNCTIONS = Registry[CriterionFunction]("criterion function")


@CRITERION_FUNCTIONS.register("bandpass")
def bandpass_criterion(
    img: Image, metric: DetailBandMetric, _logger: TextLogger
) -> np.floating:
    """
    Compute the mean absolute response of a band-pass filtered image.

    Args:
        img (Image): Input image.
        metric (DetailBandMetric): Metric settings with the detail band.
        _logger (TextLogger): Logger to use for textual logging.

    Returns:
        np.floating: Mean of the absolute band-passed image.
    """
    img_low = gauss_filter(img, img.pixel_size, _detail(metric).low)
    img_high = gauss_filter(img, img.pixel_size, _detail(metric).high)

    return np.mean(abs(img_high - img_low))


@CRITERION_FUNCTIONS.register("bandpass_var")
def bandpass_var_criterion(
    img: Image, metric: DetailBandMetric, _logger: TextLogger
) -> np.floating:
    """
    Compute the variance of the band-pass filtered image.

    Args:
        img (Image): Input image.
        metric (DetailBandMetric): Metric settings with the detail band.
        _logger (TextLogger): Logger to use for textual logging.

    Returns:
        np.floating: Variance of the band-passed image.
    """
    img_low = gauss_filter(img, img.pixel_size, _detail(metric).low)
    img_high = gauss_filter(img, img.pixel_size, _detail(metric).high)

    return np.var(img_high - img_low)


@CRITERION_FUNCTIONS.register("fft_1d")
def fft_1d_criterion(
    img: Image, metric: DetailBandMetric, _logger: TextLogger
) -> np.floating:
    """
    Compute the 1D FFT amplitude sum within a frequency band.

    Args:
        img (Image): 1D image or signal vector.
        metric (DetailBandMetric): Metric settings with the detail band.
        _logger (TextLogger): Logger to use for textual logging.

    Returns:
        np.floating: Sum of FFT amplitudes in the allowed frequency band.
    """
    # remove 0 frequency
    img0 = img - np.mean(img)
    # fft
    fft_line = np.fft.fft(img0)

    # get freq axis
    freq = np.fft.fftfreq(len(img), img.pixel_size)
    # remove negative frequencies
    fft_line = fft_line[freq > 0]
    # remove negative frequencies from freq axis
    freq = freq[freq > 0]

    # filter frequencies
    low_frequency, high_frequency = _detail(metric).to_frequency_range()
    band_i = np.where((freq < high_frequency) & (freq > low_frequency))[0]

    # sum of amplitudes of all filtered frequencies
    return np.sum(np.float64(np.abs(fft_line[band_i])))


@CRITERION_FUNCTIONS.register("fft_2d")
def fft_2d_criterion(
    img: Image, metric: DetailBandMetric, _logger: TextLogger
) -> np.floating:
    """
    Compute the 2D FFT amplitude sum within a radial frequency band.

    Args:
        img (Image): 2D image.
        metric (DetailBandMetric): Metric settings with the detail band.
        _logger (TextLogger): Logger to use for textual logging.

    Returns:
        np.floating: Sum of FFT amplitudes in the radial frequency band.
    """
    # remove 0 frequency
    img0 = img - np.mean(img)
    # fft
    fft_img = np.fft.fft2(img0)

    # get x freq axis
    freq1 = np.fft.fftfreq(fft_img.shape[0], img.pixel_size)
    # get y freq axis
    freq2 = np.fft.fftfreq(fft_img.shape[1], img.pixel_size)

    freq1 = np.repeat(freq1[:, np.newaxis], freq2.shape[0], axis=1)
    freq2 = np.repeat(freq2[:, np.newaxis], freq1.shape[0], axis=1).T
    # make freq matrix
    freq = np.sqrt(freq1**2 + freq2**2)

    # lowest and highest detail frequency
    low_frequency, high_frequency = _detail(metric).to_frequency_range()

    # make freq filter and filter frequency
    band_mask = (freq >= low_frequency) & (freq <= high_frequency)
    return np.sum(np.abs(fft_img) * band_mask)


@CRITERION_FUNCTIONS.register("fft")
def fft_criterion(
    img: Image, metric: DetailBandMetric, logger: TextLogger
) -> np.floating:
    """
    FFT-based criterion for both 1D and 2D images.

    This function is a unified wrapper that dispatches to either
    `fft_1d_criterion` or `fft_2d_criterion` depending on the dimensionality
    of the input image.

    Args:
        img (Image): Input image (1D or 2D).
        metric (DetailBandMetric): Metric settings with the detail band.
        logger (TextLogger): Logger to use for textual logging.

    Returns:
        np.floating: FFT-based criterion value.

    Raises:
        NotImplementedError: If the input dimensionality is not 1 or 2.
    """
    if np.ndim(img) == 1:
        return fft_1d_criterion(img, metric, logger)
    if np.ndim(img) == 2:
        return fft_2d_criterion(img, metric, logger)

    raise NotImplementedError(
        "Only 1D and 2D images are currently supported for focus criterion."
    )


@CRITERION_FUNCTIONS.register("frc")
def frc_criterion(img: Image, metric: FrcMetric, logger: TextLogger) -> np.floating:
    """
    Compute the single-image Fourier ring correlation cut-off frequency.

    Unlike the band-pass and FFT criteria it needs no detail band: the
    cut-off is where the signal shared by two half-images of diagonal pixel
    neighbours drops to noise level.

    Args:
        img (Image): 2-D image.
        metric (FrcMetric): Metric settings (no parameters yet).
        _logger (TextLogger): Logger to use for textual logging.

    Returns:
        np.floating: Cut-off frequency in 1/nm; higher values indicate a
            sharper image.

    Raises:
        ValueError: If the image is not 2-D, is smaller than 16x16 pixels,
            or has no structure.
    """
    _ = metric
    logger.debug("Starting frc criterion calculation.")
    return np.float64(frc_cutoff_frequency(img, img.pixel_size))


def gauss_filter(x: Image, px_size: float, detail: float) -> NDArray[np.floating]:
    """
    Apply a Gaussian filter to an image.

    Args:
        x (Image): The input image array.
        px_size (float): Pixel size in nanometers.
        detail (float): Detail parameter controlling filter width.

    Returns:
        NDArray[np.float32]: The Gaussian-filtered image.
    """
    px = detail / px_size
    sigma = 1 / (2 * np.pi * (1 / px))
    return gaussian_filter(x.astype(np.float32), sigma, mode="nearest", truncate=6)

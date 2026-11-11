# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import cv2
import numpy as np
from scipy import ndimage
from tifffile import TiffFile, TiffFileError

from fibsem_maestro.core.errors import AutoscriptNotAvailableError
from fibsem_maestro.core.format import ImageFormat
from fibsem_maestro.core.provenance import Provenance
from fibsem_maestro.core.resolution import Resolution

if TYPE_CHECKING:
    from pathlib import Path

    from autoscript_sdb_microscope_client.structures import (
        AdornedImage as AdornedImageAs,
    )
    from cv2.typing import MatLike
    from numpy.typing import NDArray

    from fibsem_maestro.core.area import PixelArea, RelativeArea

TDType = TypeVar("TDType", bound=np.generic)


class ImageError(Exception):
    """
    Exception raised for errors related to images.
    """


class _ImageBase[TDType: np.generic](np.ndarray[Any, np.dtype[TDType]]):
    """
    Base class for microscope images backed by a NumPy array.

    Extends `np.ndarray` with a `pixel_size` attribute (in nanometers)
    that is automatically preserved across slicing, copying, and other array
    operations.

    Attributes:
        pixel_size: Physical size of a single pixel in nanometers.
    """

    # in nanometers
    pixel_size: float

    def __new__(cls, image: NDArray[TDType], pixel_size: float) -> Self:
        """
        Create a new image instance from an array and a pixel size.

        Args:
            image: Source array whose data is used to back this image.
            pixel_size: Physical size of a single pixel in nanometers.

        Returns:
            A new instance of this class wrapping the given array.
        """

        obj = np.asarray(image).view(cls)
        obj.pixel_size = pixel_size
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """
        Preserve `pixel_size` when the array is sliced or copied.

        Called by NumPy whenever a new array of this type is created as a
        view of another. Copies `pixel_size` from the source object.
        """

        if obj is None:
            return
        # preserve pixel_size when array is sliced or copied
        self.pixel_size = getattr(obj, "pixel_size", 1)

    def __getitem__(self, key: Any) -> Self:
        """
        Return a slice of this image as an instance of the same subclass.

        Args:
            key: Index or slice used to select elements.

        Returns:
            A view of the selected elements as an instance of this class.
        """
        result = super().__getitem__(key)
        return result.view(type(self))

    @classmethod
    def from_autoscript(cls, as_image: AdornedImageAs) -> Self:
        """
        Construct a native Image from Autoscript's AdornedImage.

        Args:
            as_image (AdornedImageAs): The adorned image from the Autoscript library.

        Returns:
            Self: An instance of this class constructed from the Autoscript image.

        Raises:
            ImageError: If metadata or pixel size is not available on the image.
            AutoscriptNotAvailableError: If the Autoscript library is not installed.
        """
        try:
            from autoscript_sdb_microscope_client.structures import (
                AdornedImage,  # noqa: F401
            )
        except ImportError as e:
            raise AutoscriptNotAvailableError() from e

        if (metadata := as_image.metadata) is None:
            raise ImageError(
                "Could not convert autoscript image. Metadata not available."
            )

        if (pixel_size := metadata.binary_result.pixel_size) is None:
            raise ImageError(
                "Could not convert autoscript image. Pixel size not available."
            )

        return cls(np.asarray(as_image.data), pixel_size.x * 1e9)

    @classmethod
    def from_file(cls, path: Path, origin: Provenance = Provenance.MAESTRO) -> Self:
        """
        Load an image from a TIFF file.

        Args:
            path: Path to the TIFF file.
            origin: Which software produced the file. `MAESTRO` reads ImageJ
                metadata; `AUTOSCRIPT` delegates to `AdornedImage.load` and
                requires Autoscript to be installed.

        Returns:
            A new instance with pixel size in nanometres.

        Raises:
            ImageError: If the file cannot be read or carries no pixel size.
        """
        match origin:
            case Provenance.MAESTRO:
                try:
                    with TiffFile(path) as tiff_file:
                        return cls.from_tiff(tiff_file)
                except (OSError, TiffFileError) as exc:
                    raise ImageError(f"Could not load TIFF: {path}") from exc
            case Provenance.AUTOSCRIPT:
                try:
                    from autoscript_sdb_microscope_client.structures import (
                        AdornedImage,
                    )
                except ImportError as e:
                    raise AutoscriptNotAvailableError() from e

                try:
                    adorned = AdornedImage.load(str(path))
                except OSError as exc:
                    raise ImageError(f"Could not load Autoscript TIFF: {path}") from exc
                return cls.from_autoscript(adorned)

    @classmethod
    def from_tiff(cls, tiff_file: TiffFile) -> Self:
        """
        Construct an image from an open TiffFile.

        Only use to read TIFF files written by FIBSEM Maestro.

        Reads the array data and pixel size from the file's ImageJ metadata.
        The caller is responsible for closing the TiffFile after this call.

        Args:
            tiff_file: An open TiffFile instance to read from.

        Returns:
            A new instance of this class containing the image data and pixel
            size read from the file.

        Raises:
            ImageError: If the file has no ImageJ metadata or does not contain
                a `pixel_size` entry.
        """
        if (metadata := tiff_file.imagej_metadata) is None or (
            pixel_size := metadata.get("pixel_size")
        ) is None:
            raise ImageError("Missing metadata for tiff file.")

        return cls(tiff_file.asarray(), pixel_size)

    def crop(self, relative_area: RelativeArea) -> Self:
        """
        Crops the image to the specified relative area.

        Converts the given relative area to pixel coordinates and returns
        the corresponding subregion of the image. The cropped image retains the
        same pixel size as the original image.

        Args:
            relative_area (RelativeArea): The relative area specifying the region to crop.
                This area is defined in relative coordinates (0 to 1).

        Returns:
            Self: The cropped image as an instance of the same class as the original image.
        """
        pixel_area = relative_area.to_pixels(self.resolution)
        return self[
            pixel_area.origin.y : pixel_area.origin.y + pixel_area.height,
            pixel_area.origin.x : pixel_area.origin.x + pixel_area.width,
        ].copy()

    def crop_with_padding(self, relative_area: RelativeArea, padding_nm: float) -> Self:
        """
        Crop the image to a relative area extended by a padding border.

        Crops a region that is `padding_nm` nanometers larger than the requested
        area on all four sides, using real neighbouring pixels from the original
        image.

        When the padded region extends beyond the original image boundary, edge
        pixel values are replicated to fill the missing area.

        Args:
            relative_area: The region of interest expressed in relative coordinates
                (0-1). The actual crop will extend `padding_nm` beyond this area
                on each side.
            padding_nm: The amount of padding to add around the crop area, in nanometers.

        Returns:
            A cropped image extended by ``padding_px`` pixels on each side,
            with the same pixel size as the original.
        """
        pixel_size = self.pixel_size
        pixel_area = relative_area.to_pixels(self.resolution)
        padding_px = int(padding_nm / pixel_size)

        # pad the image to accommodate the border region
        image_padded = type(self)(
            np.pad(self, padding_px, mode="edge"), self.pixel_size
        )

        return image_padded[
            pixel_area.origin.y : pixel_area.origin.y
            + pixel_area.height
            + 2 * padding_px,
            pixel_area.origin.x : pixel_area.origin.x
            + pixel_area.width
            + 2 * padding_px,
        ].copy()

    def blured(self, sigma: int | None) -> Self:
        """
        Return a copy of the image with Gaussian blur applied to it.

        Args:
            sigma: Standard deviation for the Gaussian kernel. Blurring is
                skipped when `sigma` is `None` or `zero`.

        Returns:
            Blurred copy of the image with the same pixel size as the input.
        """
        if sigma is None or sigma == 0:
            return self.copy()
        return type(self)(ndimage.gaussian_filter(self, sigma=sigma), self.pixel_size)

    def upsampled(self, factor: float) -> Self:
        """
        Return a copy of the image upsampled by a given factor using bilinear interpolation.

        Args:
            factor: Scale factor applied to both axes.

        Returns:
            Upsampled copy of the image with adjusted pixel size.
        """

        upsampled = cv2.resize(
            cast("MatLike", self),
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_LINEAR,
        )

        return type(self)(cast("NDArray[Any]", upsampled), self.pixel_size / factor)

    def save(self, file_name: Path, format: ImageFormat) -> None:
        """
        Save the image to disk in the specified format.

        Args:
            file_name: Destination file path.
            format: Output image format.
        """
        match format:
            case ImageFormat.PNG:
                self._save_png(file_name)
            case ImageFormat.TIF:
                self._save_tif(file_name)

    def _save_png(self, file_name: Path) -> None:
        """
        Save the image as a PNG file using matplotlib.

        The image is rendered in grayscale with pixel intensity range inferred
        from the detected bit depth. Axes and whitespace are removed so the
        output contains only the image content.

        Args:
            file_name: Destination file path.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(self.shape[1] / 100, self.shape[0] / 100), dpi=100
        )

        _, range = self._estimate_bit_depth_and_range()

        ax.imshow(
            self, cmap="gray", interpolation="nearest", vmin=range[0], vmax=range[1]
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(file_name, format="png", dpi=100)
        plt.close(fig)

    def _save_tif(self, file_name: Path) -> None:
        """
        Save the image as a TIF file with pixel size stored in ImageJ metadata.

        Args:
            file_name: Destination file path.
        """
        import tifffile

        tifffile.imwrite(
            file_name, self, imagej=True, metadata={"pixel_size": self.pixel_size}
        )

    def _estimate_bit_depth_and_range(self) -> tuple[int, tuple[int, int]]:
        """
        Estimate the detector bit depth from image values and return its valid range.

        The bit depth is inferred from the maximum value present in the image,
        assuming unsigned integer detector behaviour.

        Returns:
            A tuple containing:
                - The estimated bit depth as an integer.
                - A `(min_value, max_value)` tuple giving the valid intensity
                  range for that bit depth, where `min_value` is always 0.
        """
        max_value = float(np.nanmax(self))  # ty: ignore

        bit_depth = int(np.ceil(np.log2(max_value + 1)))
        max_allowed = (1 << bit_depth) - 1

        return bit_depth, (0, max_allowed)

    @property
    def resolution(self) -> Resolution:
        """
        Spatial resolution of the image in pixels.

        Returns:
            A `Resolution` instance with `width` and `height` matching
            the image's column and row count respectively.
        """
        shape = self.shape
        return Resolution(shape[1], shape[0])

    def padded_to(self, covered: PixelArea, target: PixelArea) -> Self:
        """
        Extend the image so that it covers a larger area, replicating edge pixels.

        Both areas are in pixels of the same, larger frame. Only the origin of
        `covered` is used: the image's own shape is trusted over the size of
        `covered`, so a one-pixel rounding difference (e.g. in a reduced area
        scanned by the microscope) is absorbed at the right and bottom edges.

        Args:
            covered: The area of the frame this image shows.
            target: The area of the frame the result must cover. Its origin
                must not lie right of or below the origin of `covered`.

        Returns:
            A copy with shape `(target.height, target.width)` and the same pixel size.

        Raises:
            ValueError: If `target` starts right of or below `covered`.
        """
        top = covered.origin.y - target.origin.y
        left = covered.origin.x - target.origin.x
        if top < 0 or left < 0:
            raise ValueError(
                f"The target area {target} must start at or before the covered "
                f"area {covered}."
            )

        bottom = max(target.height - top - self.shape[0], 0)
        right = max(target.width - left - self.shape[1], 0)
        padded = np.pad(self, ((top, bottom), (left, right)), mode="edge")

        return type(self)(
            np.ascontiguousarray(padded[: target.height, : target.width]),
            self.pixel_size,
        )


class Image(_ImageBase[np.integer[Any]]):
    """
    Integer-valued microscope image.

    The standard image type for raw detector output. Pixel values are
    integers of any width. Use `to_8bit` to obtain a display-ready copy.
    """

    def __new__(cls, image: NDArray[np.integer[Any]], pixel_size: float) -> Self:
        return super().__new__(cls, image, pixel_size)

    def to_8bit(self) -> Image8Bit:
        """
        Return an 8-bit copy of this image.

        If the maximum pixel value exceeds 255, the entire image is linearly
        scaled so that the maximum maps to 255. Values already within the
        8-bit range are preserved exactly.

        Returns:
            A new `Image8Bit` with dtype `uint8` and the same pixel size.
        """
        pixel_size = self.pixel_size
        output = self.copy()
        if np.max(output) > 255:
            output = output / np.max(output) * 255
        output = np.ascontiguousarray(np.uint8(output))
        return Image8Bit(output, pixel_size)


class Image8Bit(_ImageBase[np.uint8]):
    """
    8-bit unsigned integer microscope image.
    """

    def __new__(cls, image: NDArray[np.uint8], pixel_size: float) -> Self:
        return super().__new__(cls, image, pixel_size)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.detail_band import DetailBand
from fibsem_maestro.criterion.reductors import REDUCTORS
from fibsem_maestro.settings.base_settings import BaseSettings
from fibsem_maestro.settings.form_utils import (
    AcquisitionOffset,
    AreaOverlay,
    BeamSource,
    CropSource,
    FieldUnit,
    FormHint,
    OverlaySpec,
    WidgetType,
)

ReductionName = Annotated[
    str,
    AfterValidator(REDUCTORS.validate),
    FormHint(widget=WidgetType.DROPDOWN, choices=lambda: list(REDUCTORS)),
]


class BasicMode(BaseSettings):
    type: Literal["basic"] = Field(
        default="basic", description="Basic sharpness calculation."
    )


class MaskMode(BaseSettings):
    type: Literal["mask"] = Field(
        default="mask", description="Sharpness calculation using a mask."
    )
    mask_name: str = Field(default="mask", description="Name of the mask to use.")
    region_reduction_fn: ReductionName = Field(
        default="mean",
        description="Numpy method for calculating final criterion from all masked regions.",
    )


CriterionCalculationMode = Annotated[BasicMode | MaskMode, Field(discriminator="type")]


class SingleTileMode(BaseSettings):
    type: Literal["single"] = Field(
        default="single", description="Calculate sharpness for the whole image."
    )


class MultiTileMode(BaseSettings):
    type: Literal["multi"] = Field(
        default="multi", description="Calculate sharpness for individual tiles."
    )
    tile_reduction_fn: ReductionName = Field(
        default="mean",
        description="Numpy method for calculating final criterion from all tiles.",
    )
    tile_size: Annotated[float, Field(ge=0), FieldUnit(suffix="nm")] = Field(
        default=1.0, description="Tile size for criterion calculation."
    )
    relative_overlap: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.0, description="Relative overlap between the tiles."
    )


CriterionTilingMode = Annotated[
    SingleTileMode | MultiTileMode, Field(discriminator="type")
]


class DetailBandMetric(BaseSettings):
    detail: Annotated[
        DetailBand,
        FormHint(widget=WidgetType.DETAIL_BAND),
        FieldUnit(suffix="nm"),
    ] = Field(
        default=DetailBand(low=1.0, high=1.0),
        description="Bandpass parameters: low and high details to filter out.",
    )


class BandpassMetric(DetailBandMetric):
    type: Literal["bandpass"] = Field(
        default="bandpass",
        description="Mean amplitude of the image band-passed to the detail range.",
    )


class BandpassVarMetric(DetailBandMetric):
    type: Literal["bandpass_var"] = Field(
        default="bandpass_var",
        description="Variance of the image band-passed to the detail range.",
    )


class FftMetric(DetailBandMetric):
    type: Literal["fft"] = Field(
        default="fft",
        description="FFT amplitude in the detail range; works on images and single lines.",
    )


class FrcMetric(BaseSettings):
    type: Literal["frc"] = Field(
        default="frc",
        description="Resolution limit from single-image Fourier ring correlation; no detail range needed.",
    )


SharpnessMetric = Annotated[
    BandpassMetric | BandpassVarMetric | FftMetric | FrcMetric,
    Field(discriminator="type"),
]


class CriterionSettings(BaseSettings):
    sharpness_metric: SharpnessMetric = Field(
        default_factory=BandpassMetric,
        description="Metric used to calculate sharpness.",
    )
    area: Annotated[
        list[RelativeArea],
        FormHint(
            widget=WidgetType.AREA_SELECT,
            min_areas=1,
            max_areas=1,
            # a refocus has no beam of its own: it images with the beam of
            # its linked imaging
            beam_sources=(
                BeamSource("beam_type"),
                BeamSource("beam_type", via="linked_imaging"),
            ),
            offset=AcquisitionOffset(source="mode.delta_x"),
            # imaging and the basic and line autofocus modes scan their own
            # area; step mode scores the frames of its linked imaging
            crop_sources=(
                CropSource("scanning_area"),
                CropSource("mode.scanning_area"),
                CropSource("scanning_area", via="linked_imaging"),
            ),
            overlays=(
                OverlaySpec.of(
                    AreaOverlay.SHOW_TILES,
                    requires={"tiling_mode": MultiTileMode},
                    tile_size_nm="tiling_mode.tile_size",
                    tile_relative_overlap="tiling_mode.relative_overlap",
                ),
            ),
        ),
    ] = Field(
        default_factory=lambda: [RelativeArea.full()],
        description="Area of the image to be used for image criterion calculation.",
    )
    log_sharpness_map: bool = Field(
        default=False,
        description="Should log sharpness map(s).",
    )
    log_best_tile: bool = Field(
        default=False,
        description="Should log the tile with the best sharpness.",
    )
    calculation_mode: CriterionCalculationMode = Field(
        default=BasicMode(),
        description="Mode of calculation.",
    )
    tiling_mode: CriterionTilingMode = Field(
        default=SingleTileMode(),
        description="Mode of tiling.",
    )

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fibsem_maestro.core.registry import Registry
from fibsem_maestro.template_matching.area_provider import (
    AREA_PROVIDERS,
)
from fibsem_maestro.template_matching.template_matching import TemplateMatching

if TYPE_CHECKING:
    from fibsem_maestro.core.drift import Drift
    from fibsem_maestro.core.image import Image8Bit
    from fibsem_maestro.logging.image.image_logger import ImageLogger
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.settings.drift_correction_settings import DriftCorrectionMode
    from fibsem_maestro.settings.template_matching_settings import (
        TemplateMatchingSettings,
    )
    from fibsem_maestro.store.image.image_store import ImageStore

DRIFT_CALCULATION_MODES = Registry[type["DriftCalculationMode"]](
    "drift calculation mode"
)


class DriftCalculationMode(ABC):
    """
    Abstract interface for drift calculation implementations.

    Defines the lifecycle of a drift calculation method, consisting of a
    one-time setup step and three per-slice hooks: an optional pre-processing
    step, the core drift calculation, and an optional post-processing step.
    The pre and post hooks do nothing by default and can be overridden by
    concrete subclasses to implement slice-level state management such as
    template updates or confidence tracking.

    Concrete subclasses are registered with `DriftCalculationRegistry` and
    retrieved by name at runtime.

    Args:
        microscope: Interface to the electron microscope.
        settings: Drift correction mode configuration.
        image_store: Store for persisting and retrieving images.
        txt_log: Logger for diagnostic and status messages.
        img_log: Logger for annotated images.
    """

    @abstractmethod
    def __init__(
        self,
        microscope: Microscope,
        settings: DriftCorrectionMode,
        image_store: ImageStore[Image8Bit],
        txt_log: TextLogger,
        img_log: ImageLogger,
    ):
        pass

    @abstractmethod
    def calculate_drift(self, after_stage_move: bool = False) -> Drift:
        """
        Calculate the drift of the current image relative to a reference.

        Args:
            after_stage_move: True right after a stage move, when the image may
                be shifted much further than by drift.

        Returns:
            A `Drift` instance containing the measured x and y shift in
            nanometers and the associated confidence score.
        """

    def setup(self, store: ImageStore[Image8Bit] | None = None) -> None:
        """
        Perform one-time initialization when the action is prepared.

        Called by `DriftCorrection` when the user prepares the action. Does
        nothing by default; override to perform initialization such as
        acquiring reference templates.

        Args:
            store: Optional image store to write to. Defaults to the current slice.
        """

    def is_set_up(self) -> bool:
        """
        Check whether everything `setup` stores is present for the current slice.

        Returns:
            `True` by default, for modes that need no setup.
        """
        return True

    def before_calculate_drift(self, slice_number: int) -> None:
        """
        Perform per-slice preparation before drift calculation.

        Called once per slice before `calculate_drift`.
        Does nothing by default - override to perform slice-level pre-processing.

        Args:
            slice_number: The current slice index.
        """

    def after_calculate_drift(self, slice_number: int) -> None:
        """
        Perform per-slice cleanup after drift calculation.

        Called once per slice after `calculate_drift`.
        Does nothing by default - override to perform slice-level
        post-processing such as updating reference templates.

        Args:
            slice_number: The current slice index.
        """

    def if_skipped(self, slice_number: int) -> None:
        """
        Perform actions when drift calculation is skipped for a slice.

        Called once per slice if drift calculation was skipped.
        Does nothing by default - override to perform slice-level
        post-processing such as updating reference templates.

        Args:
            slice_number: The current slice index.
        """

    def copy_setup(
        self, source: ImageStore[Image8Bit], target: ImageStore[Image8Bit]
    ) -> None:
        """
        Copy the data stored by `setup` from one slice's store to another's.

        Called when the setup slice (slice 0) is carried into the first
        acquired slice. Does nothing by default; override if `setup` stores
        anything.

        Args:
            source: Store holding the data written by `setup`.
            target: Store to copy it to.
        """


@DRIFT_CALCULATION_MODES.register("template_matching")
class TemplateMatchingDrift(DriftCalculationMode):
    """
    Drift calculation mode based on normalized cross-correlation template matching.

    Delegates drift measurement to a `TemplateMatching` instance.

    Args:
        microscope: Interface to the electron microscope.
        settings: Template matching configuration.
        image_store: Store for persisting and retrieving template images.
        txt_log: Logger for diagnostic and status messages.
        img_log: Logger for heatmap and overlay images.
    """

    def __init__(
        self,
        microscope: Microscope,
        settings: TemplateMatchingSettings,
        image_store: ImageStore[Image8Bit],
        txt_log: TextLogger,
        img_log: ImageLogger,
    ):
        self._template_matching = TemplateMatching(
            AREA_PROVIDERS.get(settings.frame_grabbing_mode.type)(
                microscope, settings, txt_log
            ),
            settings,
            image_store,
            txt_log,
            img_log,
        )

        self._last_confidence: float | None = None

    def setup(self, store: ImageStore[Image8Bit] | None = None) -> None:
        self._template_matching.create_templates(store)

    def is_set_up(self) -> bool:
        return self._template_matching.templates_exist()

    def calculate_drift(self, after_stage_move: bool = False) -> Drift:
        drift = self._template_matching.calculate_drift(whole_frame=after_stage_move)
        self._last_confidence = drift.confidence

        return drift

    def after_calculate_drift(self, slice_number: int) -> None:
        self._template_matching.update_templates(slice_number, self._last_confidence)

        self._last_confidence = None

    def if_skipped(self, slice_number: int) -> None:
        self.after_calculate_drift(slice_number)

    def copy_setup(
        self, source: ImageStore[Image8Bit], target: ImageStore[Image8Bit]
    ) -> None:
        self._template_matching.copy_templates(source, target)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

import math
from dataclasses import dataclass
from typing import Annotated

from fibsem_maestro.settings.form_utils import FieldUnit


@dataclass(frozen=True)
class Drift:
    """
    Represents a measured image drift along the x and y dimensions.

    Attributes:
        x: Drift along the x-axis in nanometers. None if undetermined.
        y: Drift along the y-axis in nanometers. None if undetermined.
        confidence: Confidence of the drift estimate in the range [0, 1].
            None if the estimator does not provide a confidence value.
        reason: Why the drift is undetermined; None if it is determined or
            the estimator gives no reason.
    """

    x: Annotated[float | None, FieldUnit(suffix="nm")]
    y: Annotated[float | None, FieldUnit(suffix="nm")]
    confidence: float | None = None
    reason: str | None = None

    def is_valid(self) -> bool:
        """
        Check if the drift is fully defined and finite in all dimensions.

        Returns:
            True if both axes hold a finite value, False if either is None,
            NaN, or infinite.
        """
        return (
            self.x is not None
            and self.y is not None
            and math.isfinite(self.x)
            and math.isfinite(self.y)
        )

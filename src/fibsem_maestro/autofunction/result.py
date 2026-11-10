# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from dataclasses import dataclass

# Pydantic evaluates this dataclass's annotations (it is a field of AutofunctionState)
from fibsem_maestro.autofunction.sweep_step import SweepStep  # noqa: TC001


@dataclass(frozen=True)
class AutofunctionResult:
    sharpness: float
    sweep: SweepStep

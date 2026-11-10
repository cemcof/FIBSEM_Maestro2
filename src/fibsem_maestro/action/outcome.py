# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fibsem_maestro.properties.global_properties import GlobalProperties


@dataclass(frozen=True)
class Produced:
    """
    The action ran and produced new properties for the next slice.

    Attributes:
        props: The properties written to the action's next-slice store. These
            are also the only source for propagation to dependent actions.
        patches: Explicit property patches for named actions, applied to their
            props files after rule-based propagation, so they take precedence.
            Used where one action must set another's input directly, e.g.
            step-mode autofunction handing trial values to its linked imaging.
    """

    props: GlobalProperties
    patches: Mapping[str, GlobalProperties] = field(default_factory=dict)


@dataclass(frozen=True)
class CarriedOver:
    """
    The action produced nothing new for the next slice.

    Its current properties were copied to the next slice unchanged, and
    nothing is propagated to dependents.

    Attributes:
        reason: Human-readable reason, for logging.
    """

    reason: str


type StepOutcome = Produced | CarriedOver

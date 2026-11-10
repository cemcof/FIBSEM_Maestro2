# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF


from typing import TYPE_CHECKING

from fibsem_maestro.autofunction import SWEEPING_STRATEGIES
from fibsem_maestro.autofunction.sweep_space import SweepSpace, make_sweep_space
from fibsem_maestro.autofunction.sweep_step import SweepPoint, SweepStep

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fibsem_maestro.autofunction.evaluation import SweepEvaluation
    from fibsem_maestro.autofunction.result import AutofunctionResult
    from fibsem_maestro.logging.text.text_logger import TextLogger
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
    from fibsem_maestro.settings.sweeping_settings import SweepingSettings


class Sweeping:
    """
    Controller for parameter sweeping on a microscope beam.

    Sweeps one (1D) or two (2D) scalar beam attributes. The sweep space places
    the candidates around the current values, the sweeping strategy orders them
    for each cycle and chooses the best one. Only values within the beam's
    hardware limits are yielded; out-of-range values are logged and skipped.

    Args:
        beam: The beam control whose attributes will be swept.
        settings: Sweeping configuration: the swept attributes and their
            geometry, the number of cycles, and the strategy.
        txt_log: Logger for sweep-related status and warning messages.
    """

    def __init__(
        self,
        beam: BeamControl,
        settings: SweepingSettings,
        txt_log: TextLogger,
    ):
        self._beam = beam
        self._txt_log = txt_log
        self._cycles = settings.cycles
        self._space = make_sweep_space(settings)
        self._sweeping_strategy = SWEEPING_STRATEGIES.get(settings.strategy.type)(
            settings.strategy
        )

    @property
    def attributes(self) -> tuple[str, ...]:
        """Names of the swept beam attributes, in the order of a sweep point's values."""
        return self._space.attributes

    @property
    def space(self) -> SweepSpace:
        """Geometry of the sweep."""
        return self._space

    def sweep(self) -> Iterator[SweepStep]:
        """
        Iterate over all sweep steps across all configured cycles.

        Reads the current attribute values once before the sweep begins and
        uses them as the base for all generated values. Each yielded step
        carries its repetition index, value, and global step index.

        Yields:
            SweepStep: The next valid sweep step in sequence.
        """
        base = self.get_attribute_value()

        steps = (
            (rep, s)
            for rep in range(self._cycles)
            for s in self._sweep_inner(rep, base)
        )

        for index, (repetition, value) in enumerate(steps):
            self._txt_log.info(f"Sweep cycle {repetition + 1}/{self._cycles}.")
            yield SweepStep(repetition, value, index)

    def get_attribute_value(self) -> SweepPoint:
        """
        Return the current values of the swept attributes.

        Returns:
            One value per swept attribute, read from the beam.
        """
        return tuple(float(getattr(self._beam, name)) for name in self.attributes)

    def set_attribute_value(self, value: SweepPoint) -> None:
        """
        Set the swept attributes on the beam.

        Args:
            value: One value per swept attribute.
        """
        for name, component in zip(self.attributes, value, strict=True):
            setattr(self._beam, name, component)

    def describe(self, value: SweepPoint) -> str:
        """
        Return a readable `name=value` listing of a sweep point, for log messages.

        Args:
            value: One value per swept attribute.
        """
        return ", ".join(
            f"{name}={component:g}"
            for name, component in zip(self.attributes, value, strict=True)
        )

    def evaluate(self, results: list[AutofunctionResult]) -> SweepEvaluation:
        """
        Choose the best value from the results of a sweep.

        Delegates to the configured sweeping strategy.

        Args:
            results: Autofunction results collected during the sweep, one per submitted job.

        Returns:
            The strategy's evaluation; its `best` is the value to apply.
        """
        return self._sweeping_strategy.evaluate(results)

    def _sweep_inner(self, repetition: int, base: SweepPoint) -> Iterator[SweepPoint]:
        """
        Generate valid sweep values for a single repetition cycle.

        Produces values via the sweeping strategy and filters out any with a
        component outside the beam's hardware limits. Out-of-range values are
        logged as warnings and skipped.

        Args:
            repetition: Index of the current sweep cycle, passed to the strategy.
            base: The attribute values at the start of the sweep.

        Yields:
            Valid sweep values that lie within the beam's hardware limits.
        """
        limits = {name: self._beam.limits(name) for name in self.attributes}

        for value in self._sweeping_strategy.generate(self._space, base, repetition):
            out_of_range = [
                f"'{name}' = {component} is out of range [{limits[name][0]} - {limits[name][1]}]"
                for name, component in zip(self.attributes, value, strict=True)
                if not limits[name][0] <= component <= limits[name][1]
            ]

            if out_of_range:
                self._txt_log.warning(
                    f"Skipping sweep value: {'; '.join(out_of_range)}."
                )
                continue
            yield value

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fibsem_maestro.logging.image.plot_element import PlotElement


@dataclass(frozen=True)
class Tick:
    """
    A labelled tick on an axis.

    Attributes:
        position: Position on the axis, in data units.
        label: Text shown at the tick.
    """

    position: float
    label: str


@dataclass(frozen=True, kw_only=True)
class Panel:
    """
    One set of axes and the elements drawn in it.

    Attributes:
        elements: Elements to draw, in drawing order (later on top).
        title: Text above the panel.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        xticks: Labelled ticks replacing the automatic x ticks, or None.
        equal_aspect: Scale both axes equally (needed to show a circle as one).
        legend: Draw a legend of the labelled elements.
        sharex: Key of another panel of the same plot whose x-axis this one shares, or None.
        invert_y: Draw the y-axis with its values decreasing upwards.
    """

    elements: Sequence[PlotElement]
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xticks: Sequence[Tick] | None = None
    equal_aspect: bool = False
    legend: bool = False
    sharex: str | None = None
    invert_y: bool = False


@dataclass(frozen=True, kw_only=True)
class Plot:
    """
    A figure made of one or more panels.

    Attributes:
        panels: The panels, by key.
        layout: Rows of panel keys, in the style of a Matplotlib mosaic; a key
            repeated in neighbouring cells makes its panel span them. None
            stacks the panels in one column, in the order of `panels`.
        title: Text above the whole figure.
    """

    panels: Mapping[str, Panel]
    layout: Sequence[Sequence[str]] | None = None
    title: str | None = None
    _rows: tuple[tuple[str, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = (
            tuple((key,) for key in self.panels)
            if self.layout is None
            else tuple(tuple(row) for row in self.layout)
        )

        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError(
                "A plot layout needs at least one row, all of equal length."
            )

        placed = {key for row in rows for key in row}
        if placed != set(self.panels):
            raise ValueError(
                f"The layout places panels {sorted(placed)}, but the plot has "
                f"{sorted(self.panels)}."
            )

        for key, panel in self.panels.items():
            if panel.sharex is not None and panel.sharex not in self.panels:
                raise ValueError(
                    f"Panel '{key}' shares the x-axis of unknown panel '{panel.sharex}'."
                )

        object.__setattr__(self, "_rows", rows)

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        """The resolved layout: rows of panel keys."""
        return self._rows

    @classmethod
    def single(
        cls,
        elements: Sequence[PlotElement],
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> Plot:
        """
        Build a plot with one panel.

        Args:
            elements: Elements to draw.
            title: Title of the plot.
            xlabel: Label of the x-axis.
            ylabel: Label of the y-axis.

        Returns:
            A one-panel plot.
        """
        return cls(
            panels={"main": Panel(elements=elements, xlabel=xlabel, ylabel=ylabel)},
            title=title,
        )

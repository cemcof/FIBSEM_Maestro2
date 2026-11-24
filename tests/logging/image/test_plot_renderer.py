# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Rendering of a reversed colour scale and an inverted y-axis."""

from typing import Literal

import numpy as np
import pytest

from fibsem_maestro.logging.image.plot import Panel, Plot
from fibsem_maestro.logging.image.plot_element import ColorScale, Points
from fibsem_maestro.logging.image.plot_renderer import render_plot


def _colors(scale: ColorScale) -> np.ndarray:
    """The colours of a low and a high value drawn with the scale."""
    points = Points(
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        values=[1.0, 2.0],
        scale=scale,
        color="black",
        linewidth=0.5,
    )
    figure = render_plot(Plot(panels={"main": Panel(elements=[points])}))
    (collection,) = figure.axes[0].collections
    return np.asarray(collection.to_rgba(np.asarray([1.0, 2.0])))


@pytest.mark.parametrize("kind", ["sequential", "diverging"])
def test_reversed_color_scale_swaps_the_colors_of_low_and_high_values(
    kind: Literal["sequential", "diverging"],
) -> None:
    plain = _colors(ColorScale(kind, "value"))
    reversed_ = _colors(ColorScale(kind, "value", reverse=True))

    assert not np.allclose(plain, reversed_)


def test_inverted_panel_has_its_y_axis_decreasing_upwards() -> None:
    plot = Plot(panels={"main": Panel(elements=[], invert_y=True)})

    assert render_plot(plot).axes[0].yaxis_inverted()

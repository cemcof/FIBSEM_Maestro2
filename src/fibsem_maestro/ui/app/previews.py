# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Acquisition of the images area selectors draw on."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.beam_shift import BeamShift
from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTask

if TYPE_CHECKING:
    from fibsem_maestro.core.image import Image
    from fibsem_maestro.microscope.abstract_control.beam_control import BeamControl
    from fibsem_maestro.microscope.microscope import Microscope
    from fibsem_maestro.ui.app.microscope_tasks import MicroscopeTasks
    from fibsem_maestro.ui.app.session import WorkflowSession
    from fibsem_maestro.ui.ports import PreviewRequest, TaskHandle


class PreviewService:
    """
    Acquires area-selector images on the microscope task thread.

    Forms reach it only through `AppCommands.grab`, which checks the permission first.

    Args:
        session: The workflow session (for the microscope).
        tasks: The microscope task runner.
    """

    def __init__(self, session: WorkflowSession, tasks: MicroscopeTasks) -> None:
        self._session = session
        self._tasks = tasks

    def grab(self, request: PreviewRequest) -> TaskHandle:
        """
        Submit the acquisition.

        Args:
            request: What to image.

        Returns:
            The task's handle; `succeeded` carries the (cropped) `Image`. Nothing
            is logged.
        """
        return self._tasks.submit(
            MicroscopeTask(
                run=partial(acquire_preview, self._session.microscope, request),
                description="Acquiring a preview",
                log=None,
            )
        )


def acquire_preview(microscope: Microscope, request: PreviewRequest) -> Image:
    """
    Acquire the image an area selector shows. Runs on the task thread.

    Args:
        microscope: The microscope.
        request: What to image.

    Returns:
        The image, cropped to `request.crop` if given.
    """
    beam = _resolve_beam(microscope, request.beam)
    offset_x = request.offset_x_nm

    # Load avoids an extra frame where it can: with no offset and the
    # full frame scanned, the microscope's last frame is assumed to be
    # the current view. A fresh full-frame image is grabbed when an
    # offset applies, when this widget's previous Load used one (the
    # last frame is then the shifted one), or when the microscope's own
    # GUI has a reduced area set, which previews ignore as actions do.
    #
    # The assumption can still be wrong: anything else that acquired
    # with a shifted beam since (another area selector with an offset,
    # a basic or line autofunction sweep or test) leaves its frame as the
    # last one, and a Load here then shows that frame instead of the
    # current view. Only grabbing a fresh frame on every Load avoids
    # this, at the cost of an extra frame (and ion dose) each time.
    if (
        offset_x != 0.0
        or request.previous_offset_nm not in (None, 0.0)
        or not beam.scanning_area.is_full_frame()
    ):
        with (
            microscope.set_temporary_beam_property(
                "scanning_area", RelativeArea.full(), beam.beam_type()
            ),
            microscope.add_temporary_beam_shift(BeamShift(x=offset_x, y=0), beam),
        ):
            image = beam.grab_frame()
    else:
        image = beam.get_image()

    # areas are drawn on what the frames they are applied to show
    if (crop := request.crop) is not None:
        image = image.crop(crop)

    return image


def _resolve_beam(microscope: Microscope, beam_type: BeamType | None) -> BeamControl:
    """
    Return the beam to image with.

    Args:
        microscope: The microscope.
        beam_type: The requested beam, or None.

    Returns:
        The electron or ion beam if one is requested, otherwise the currently
        active beam.
    """
    match beam_type:
        case BeamType.ELECTRON:
            return microscope.electron_beam
        case BeamType.ION:
            return microscope.ion_beam
        case None:
            return microscope.beam

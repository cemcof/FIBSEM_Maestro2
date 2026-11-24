# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The dialog showing stored properties in a form."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fibsem_maestro.properties.beam_properties import BeamProperties
from fibsem_maestro.properties.global_properties import GlobalProperties
from fibsem_maestro.ui.forms.editors.object import ObjectWidget
from fibsem_maestro.ui.widgets.properties_dialog import PropertiesDialog

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from fibsem_maestro.ui.forms.builder import FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext


def test_properties_dialog_keeps_its_form_packed_in_a_tall_window(
    qtbot: QtBot, form_builder: FormBuilder, form_context: FormContext
) -> None:
    properties = GlobalProperties(electron_beam=BeamProperties(dwell_time=100.0))
    dialog = PropertiesDialog(properties, form_builder, form_context)
    qtbot.addWidget(dialog)
    dialog.resize(800, 2000)
    dialog.show()
    qtbot.waitExposed(dialog)

    objects = dialog.findChildren(ObjectWidget)
    form = next(
        widget
        for widget in objects
        if not any(
            other is not widget and other.isAncestorOf(widget) for other in objects
        )
    )

    # the spare height went between the rows, which floated apart
    assert form.height() == form.sizeHint().height()

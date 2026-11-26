# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The user interface of the drift correction."""

from fibsem_maestro.ui.actions_ui.default_tools import DEFAULT_TOOLS
from fibsem_maestro.ui.actions_ui.spec import ACTION_UI, ActionUiSpec

ACTION_UI.add(
    "drift_correction",
    ActionUiSpec(
        tools=DEFAULT_TOOLS,
        reset_note=(
            "The templates are deleted too - new ones are acquired at the next execution."
        ),
    ),
)

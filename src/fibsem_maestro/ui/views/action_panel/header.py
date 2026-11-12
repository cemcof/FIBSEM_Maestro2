# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The header of an action panel: name, type, beam and the tool buttons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from fibsem_maestro.ui.forms.style import (
    BORDER_COLOR,
    CARD_COLORS,
    CARD_RADIUS,
    MUTED_TEXT_COLOR,
)
from fibsem_maestro.ui.schema.labels import class_name_to_label
from fibsem_maestro.ui.widgets.icon_button import IconButton
from fibsem_maestro.ui.widgets.icons import ICON_RESET_STATE

RESET_STATE_TOOLTIP = (
    "<b>Reset state</b><br>"
    "Discard the action's internal state. The settings, the "
    "stored properties and the slice are kept."
)

_HEADER_OBJECT_NAME = "actionHeader"
_STRIP_OBJECT_NAME = "toolStrip"

_HEADER_STYLESHEET = f"""
    QFrame#{_HEADER_OBJECT_NAME} {{
        border: none;
        border-bottom: 1px solid {BORDER_COLOR};
    }}
    QFrame#{_STRIP_OBJECT_NAME} {{
        background-color: {CARD_COLORS["a"][0]};
        border: 1px solid {BORDER_COLOR};
        border-radius: {CARD_RADIUS}px;
    }}
"""
"""A divider below the header instead of a frame around it; the tools in a rounded strip."""

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fibsem_maestro.action.action import Action
    from fibsem_maestro.ui.actions_ui.spec import ToolSpec
    from fibsem_maestro.ui.app.edit_policy import Permissions


@dataclass(frozen=True)
class ActionHeaderViewData:
    """
    What an action panel's header shows.

    Attributes:
        name: The action's name (17 px bold).
        type_text: The type's label, e.g. "autofunction".
        beam_text: The beam, e.g. "electron", or "—" for none.
        tools_enabled: Whether each tool button is enabled, by `ToolSpec.key`.
        reset_enabled: Whether the Reset state button is enabled.
    """

    name: str
    type_text: str
    beam_text: str
    tools_enabled: dict[str, bool]
    reset_enabled: bool


def action_header_view(
    action: Action, permissions: Permissions, has_props: bool, tools: Sequence[ToolSpec]
) -> ActionHeaderViewData:
    """
    Decide what an action panel's header shows.

    A tool is enabled iff microscope work is allowed (if it uses the
    microscope) or danger mode is on (if it is usable in danger mode), props
    are stored for the action's current slice (if it needs them) and its own
    condition holds (if it has one).

    Reset state is enabled iff the tools may be used (`action_tools_enabled`;
    danger mode does not unlock it) and the action's state is not the default
    one, i.e. there is something to discard.

    Args:
        action: The action.
        permissions: The current permissions.
        has_props: Whether props are stored for the action's current slice.
        tools: The header's tools.

    Returns:
        The view data.
    """
    beam = str(action.beam_type) if action.beam_type is not None else "—"
    return ActionHeaderViewData(
        name=action.name,
        type_text=class_name_to_label(type(action).__name__),
        beam_text=beam,
        tools_enabled={
            tool.key: (
                (
                    permissions.action_tools_enabled
                    or not tool.uses_microscope
                    or (permissions.danger_tools_enabled and tool.usable_in_danger_mode)
                )
                and (has_props or not tool.requires_props)
                and (tool.enabled_when is None or tool.enabled_when(action))
            )
            for tool in tools
        },
        reset_enabled=(
            permissions.action_tools_enabled and action.state != action.state_cls()()
        ),
    )


class ActionHeader(QObject):
    """
    Header: the name above "type · beam", the tool buttons beside them in one strip.

    The Reset state button sits apart at the right end: it discards what the
    action has done, unlike the tools. A divider line runs below the header.

    The header is a plain `QFrame` (`frame`), styled by its object name.

    Args:
        tools: The tools, in button order.
        data: What to show initially.
        parent: Parent object (owns the header, not the frame).
    """

    tool_clicked = pyqtSignal(str)
    """The `ToolSpec.key` of a clicked tool button."""

    reset_clicked = pyqtSignal()
    """The Reset state button was clicked."""

    def __init__(
        self,
        tools: Sequence[ToolSpec],
        data: ActionHeaderViewData,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._frame = QFrame()
        self._frame.setObjectName(_HEADER_OBJECT_NAME)
        self._frame.setStyleSheet(_HEADER_STYLESHEET)
        header_layout = QHBoxLayout(self._frame)
        header_layout.setContentsMargins(4, 2, 4, 10)
        header_layout.setSpacing(0)

        labels_layout = QVBoxLayout()
        labels_layout.setSpacing(2)

        self._name_label = QLabel(data.name)
        self._name_label.setStyleSheet("font-size: 17px; font-weight: bold;")
        labels_layout.addWidget(self._name_label)

        # "type · beam", as below the name in the action list
        self._subtitle_label = QLabel(_subtitle(data))
        self._subtitle_label.setStyleSheet(
            f"font-size: 12px; color: {MUTED_TEXT_COLOR};"
        )
        labels_layout.addWidget(self._subtitle_label)

        self._buttons: dict[str, IconButton] = {}
        strip = QFrame()
        strip.setObjectName(_STRIP_OBJECT_NAME)
        buttons_layout = QHBoxLayout(strip)
        buttons_layout.setContentsMargins(4, 3, 4, 3)
        buttons_layout.setSpacing(4)
        for tool in tools:
            button = IconButton(tool.icon, tool.tooltip, tool.accessible_name)
            button.clicked.connect(self._on_button_clicked)
            self._buttons[tool.key] = button
            buttons_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._reset_button = IconButton(
            ICON_RESET_STATE, RESET_STATE_TOOLTIP, "Reset internal state"
        )
        self._reset_button.clicked.connect(self.reset_clicked)

        header_layout.addLayout(labels_layout)
        header_layout.addSpacing(32)
        header_layout.addWidget(strip, 0, Qt.AlignmentFlag.AlignVCenter)
        strip.setVisible(bool(tools))
        header_layout.addStretch(1)
        header_layout.addWidget(self._reset_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self.render_data(data)

    @property
    def frame(self) -> QFrame:
        """The header's widget, to be placed in a layout."""
        return self._frame

    @property
    def name_label(self) -> QLabel:
        """The label showing the action's name."""
        return self._name_label

    @property
    def subtitle_label(self) -> QLabel:
        """The label showing the action's type and beam."""
        return self._subtitle_label

    @property
    def reset_button(self) -> IconButton:
        """The Reset state button."""
        return self._reset_button

    def render_data(self, data: ActionHeaderViewData) -> None:
        """
        Show new view data.

        Args:
            data: What to show.
        """
        self._name_label.setText(data.name)
        self._subtitle_label.setText(_subtitle(data))
        for key, button in self._buttons.items():
            button.setEnabled(data.tools_enabled.get(key, False))
        self._reset_button.setEnabled(data.reset_enabled)

    def button(self, key: str) -> IconButton:
        """
        The button of a tool.

        Args:
            key: The tool's `ToolSpec.key`.

        Returns:
            The button.

        Raises:
            KeyError: If the header has no such tool.
        """
        return self._buttons[key]

    def _on_button_clicked(self) -> None:
        sender = self.sender()
        for key, button in self._buttons.items():
            if button is sender:
                self.tool_clicked.emit(key)
                return


def _subtitle(data: ActionHeaderViewData) -> str:
    """The line below the name: "type · beam"."""
    return f"{data.type_text}  ·  {data.beam_text}"

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""
The look of the cards grouping a form's fields.

The colours are those of qdarkstyle's dark palette
(`qdarkstyle.dark.palette.DarkPalette`), which the application applies to
every widget (see `main`); the names say which palette entry each one is.
"""

WINDOW_COLOR = "#19232D"
"""qdarkstyle's first background: windows, and the inside of inputs."""

BORDER_COLOR = "#455364"
"""qdarkstyle's fourth background: the borders of inputs and frames."""

MUTED_TEXT_COLOR = "#9DA9B5"
"""qdarkstyle's fourth text colour: secondary text, e.g. an action's type and beam."""

CARD_COLORS: dict[str, tuple[str, str]] = {
    "a": ("#212D3A", "#37414F"),
    "b": ("#293544", "#455364"),
}
"""
Background and border of a group card, by shade.

Nested cards alternate the shades, each a step lighter than the window
(between qdarkstyle's first, second and fourth backgrounds), so the depth
of a group stays readable.
"""

CARD_RADIUS = 7
"""Corner radius of a group card (px)."""

CARD_INSET = 10
"""Space between a card's border and its content, left and right (px)."""

CARD_INDENT = CARD_INSET + 1
"""How much further right the content of a card starts than the card itself (px): inset and border."""

VARIANT_SPACING = 24
"""Space between the radio buttons choosing a union's variant (px)."""

CARD_OBJECT_NAME = "formCard"
"""Object name of a group card (`CollapsibleGroupBox`), selecting its style."""


CARD_STYLESHEET = "\n".join(
    [
        *(
            f"""
    QFrame#{CARD_OBJECT_NAME}[shade="{shade}"] {{
        background-color: {background};
        border: 1px solid {border};
        border-radius: {CARD_RADIUS}px;
    }}"""
            for shade, (background, border) in CARD_COLORS.items()
        ),
        f"""
    QFrame#{CARD_OBJECT_NAME} QLabel,
    QFrame#{CARD_OBJECT_NAME} QCheckBox,
    QFrame#{CARD_OBJECT_NAME} QRadioButton,
    QFrame#{CARD_OBJECT_NAME} .QWidget {{
        background-color: transparent;
    }}""",
    ]
)
"""
Style of a group card, set on each card.

The card's background follows its `shade` property. Labels, checkboxes and
radio buttons inside are transparent, as are plain `QWidget` containers (the
class selector `.QWidget` leaves out the inputs deriving from it): qdarkstyle
paints them in the window colour, which would show as dark boxes on the card. Inputs keep their own
(window-coloured) background.
"""

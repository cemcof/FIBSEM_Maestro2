# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The style sheet of a scroll area holding settings forms."""

HIGHLIGHT_COLOR = "#346792"
"""Frame of the field under the mouse (qdarkstyle's second accent)."""

READ_ONLY_COLOR = "#788d9c"
"""Text of every value while the fields cannot be edited (qdarkstyle's disabled text)."""

_READ_ONLY_STATES = {
    "QAbstractSpinBox": (":read-only", ":disabled"),
    "QComboBox": (":disabled",),
    "QLineEdit": (":read-only", ":disabled"),
    "QPlainTextEdit": (":read-only", ":disabled"),
    "QListWidget": (":disabled",),
}
"""The controls showing a value as text, and their states in which it cannot be edited."""

_READ_ONLY_TEXT = ",\n    ".join(
    f"{control}{state}"
    for control, states in _READ_ONLY_STATES.items()
    for state in states
)

FORM_STYLESHEET = f"""
    QWidget[dataclass_form="true"] QWidget[highlighted="true"] {{
        border: 1px solid {HIGHLIGHT_COLOR};
        border-radius: 3px;
    }}
    {_READ_ONLY_TEXT} {{
        color: {READ_ONLY_COLOR};
    }}
"""
"""
Style of every form inside.

Frames the field under the mouse (its `highlighted` property) and greys every
value that cannot be edited; read-only spin boxes would otherwise keep their
normal text.
"""

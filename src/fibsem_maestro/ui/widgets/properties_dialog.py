# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""A dialog showing stored properties in an editable form."""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem_maestro.ui.widgets.form_style import FORM_STYLESHEET

if TYPE_CHECKING:
    from fibsem_maestro.properties.global_properties import GlobalProperties
    from fibsem_maestro.ui.forms.builder import FormBuilder
    from fibsem_maestro.ui.forms.context import FormContext


class PropertiesDialog(QDialog):
    """
    Shows a `GlobalProperties` instance in an editable form.

    Args:
        properties: The live properties instance to bind the form to. It is
            mutated in place as the user edits.
        form_builder: Builds the form.
        form_context: The environment of the form (a logger for build-time
            warnings, the microscope catalogs).
        title: Window title.
        hint: Short explanation shown above the form.
        parent: Parent widget.
    """

    def __init__(
        self,
        properties: GlobalProperties,
        form_builder: FormBuilder,
        form_context: FormContext,
        title: str = "Properties",
        hint: str = "Edit the values if needed, then save.",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._properties = properties

        self.setWindowTitle(title)
        self.setModal(True)
        if (screen := self.screen()) is not None:
            self.setGeometry(screen.availableGeometry())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        hint_label = QLabel(hint)
        hint_label.setStyleSheet("font-size: 11px; color: #888888;")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(FORM_STYLESHEET)
        layout.addWidget(scroll)

        self._form = form_builder.build_live(properties, form_context)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self._form.widget)
        # the form keeps its height; the rest of a tall dialog stays empty below it
        container_layout.addStretch()
        scroll.setWidget(container)

        standard = (
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons = QDialogButtonBox(standard)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    @property
    def properties(self) -> GlobalProperties:
        """The bound properties instance, including any edits made in the form."""
        return self._properties

    def done(self, a0: int) -> None:
        """Release the form's hooks on the properties when the dialog closes."""
        self._form.dispose()
        super().done(a0)

    @classmethod
    def review(
        cls,
        properties: GlobalProperties,
        form_builder: FormBuilder,
        form_context: FormContext,
        title: str = "Properties",
        hint: str = "Edit the values if needed, then save.",
        parent: QWidget | None = None,
    ) -> GlobalProperties | None:
        """
        Show the dialog modally and return the reviewed properties.

        Args:
            properties: The live properties instance to review. Mutated in place.
            form_builder: Builds the form.
            form_context: The environment of the form.
            title: Window title.
            hint: Short explanation shown above the form.
            parent: Parent widget.

        Returns:
            The edited instance if the user accepted, else None. `properties` is
            mutated either way; a None return means the caller should discard it.
        """
        dialog = cls(
            properties=properties,
            form_builder=form_builder,
            form_context=form_context,
            title=title,
            hint=hint,
            parent=parent,
        )

        try:
            if dialog.exec():
                return dialog.properties
            return None
        finally:
            # a parented dialog would otherwise live as long as its parent
            dialog.deleteLater()

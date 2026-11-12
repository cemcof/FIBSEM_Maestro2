# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Editor of a discriminated union: one radio button and one page per variant."""

from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.editors.object import ObjectWidget
from fibsem_maestro.ui.forms.issues import FieldValidationError
from fibsem_maestro.ui.forms.style import VARIANT_SPACING
from fibsem_maestro.ui.schema.labels import class_name_to_label
from fibsem_maestro.ui.schema.schema import get_field_infos

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.ui.schema.field_info import FieldInfo


class DiscriminatedUnionWidget(QWidget, FieldEditor[Any]):
    """
    Radio buttons + stacked variant pages for discriminated unions.

    One radio button per variant. Selecting a variant shows its extra fields
    (fields beyond the discriminators) below the radio row. Variants with no
    extra fields show nothing.

    A variant may be a nested union rather than a plain model: pass its
    pre-built widget in `nested`, keyed by the outer discriminator value. That
    page then owns construction of the value, since the inner leaf classes
    already carry the outer tag.

    Args:
        variants: The (discriminator value, variant class) pairs to offer.
        discriminator_key: The field name whose value identifies the variant.
        build_object: Factory building an `ObjectWidget` for a variant class
            from a list of field infos.
        nested: Pre-built widgets to use as the page for the given
            discriminator values, instead of building an `ObjectWidget` from
            the variant class. Used for nested unions.
        labels: Radio-button text overrides, keyed by discriminator value.
            Falls back to the variant class name.
        descriptions: Radio-button tooltips, keyed by discriminator value.
            Variants without one get no tooltip.
        exclude: Field names to omit from every variant's form, in addition to
            `discriminator_key`. Used to hide the discriminators of enclosing
            unions, which a nested variant class also carries.
        show_selector: If False, the radio buttons are hidden and the variant
            is chosen only via `select()` or `set_value()`.
        parent: The parent widget, if any.
    """

    def __init__(
        self,
        variants: list[tuple[str, type]],
        discriminator_key: str,
        build_object: Callable[[type, list[FieldInfo] | None], ObjectWidget[Any]],
        nested: dict[str, FieldEditor[Any]] | None = None,
        labels: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        exclude: frozenset[str] = frozenset(),
        show_selector: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        FieldEditor.__init__(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._discriminator_key = discriminator_key
        self._discriminator_values: list[str] = []
        self._variant_classes: list[type] = []
        self._empty_indices: set[int] = set()
        self._nested_indices: set[int] = set()
        self._variant_widgets: list[QWidget] = []
        self._exclude = exclude | {discriminator_key}

        nested = nested or {}
        labels = labels or {}
        descriptions = descriptions or {}

        self._button_group = QButtonGroup(self)
        self._show_selector = show_selector
        radio_row = QHBoxLayout()
        radio_row.setSpacing(VARIANT_SPACING)
        self._radio_row = radio_row

        for i, (disc_value, variant_cls) in enumerate(variants):
            self._discriminator_values.append(disc_value)
            self._variant_classes.append(variant_cls)

            btn = QRadioButton(
                labels.get(disc_value) or class_name_to_label(variant_cls.__name__)
            )
            # kept in the layout and the group even when hidden: a button held
            # only by the QButtonGroup is collected and leaves a dangling id
            btn.setVisible(show_selector)
            if description := descriptions.get(disc_value):
                btn.setToolTip(description)
            self._button_group.addButton(btn, i)
            radio_row.addWidget(btn)

            inner = nested.get(disc_value)
            if inner is not None:
                variant_widget: QWidget = cast("QWidget", inner)
                self._nested_indices.add(i)
            else:
                extra_field_infos = [
                    fi
                    for fi in get_field_infos(variant_cls)
                    if fi.name not in self._exclude
                ]

                if extra_field_infos:
                    variant_widget = build_object(variant_cls, extra_field_infos)
                else:
                    variant_widget = QWidget()
                    self._empty_indices.add(i)

            self._variant_widgets.append(variant_widget)
            layout.addWidget(variant_widget)
            variant_widget.hide()

        # radio buttons on top, packed to the left, variant widgets below
        radio_row.addStretch()
        layout.insertLayout(0, radio_row)

        first_btn = self._button_group.button(0)
        if first_btn:
            first_btn.setChecked(True)
            self._on_selection_changed(0)

        self._button_group.idClicked.connect(self._on_selection_changed)

    def variant_buttons(self) -> list[QRadioButton]:
        """
        Return the radio buttons, wherever they are shown.

        Returns:
            One button per variant, in variant order (hidden without a selector).
        """
        return cast("list[QRadioButton]", self._button_group.buttons())

    def take_selector(self) -> list[QRadioButton]:
        """
        Hand the radio buttons over to be shown elsewhere, e.g. on a card's title line.

        The buttons stay in the button group, so selecting works as before.
        Without a selector (`show_selector=False`) nothing is handed over, and
        the hidden buttons stay in this widget.

        Returns:
            The radio buttons, in variant order; the caller places them in its
            own layout. Empty if the selector is not shown.
        """
        if not self._show_selector:
            return []
        buttons = self.variant_buttons()
        for button in buttons:
            self._radio_row.removeWidget(button)
        # the empty row would still take its spacing
        if (layout := self.layout()) is not None:
            layout.removeItem(self._radio_row)
        return buttons

    def _on_selection_changed(self, index: int) -> None:
        """
        Show the selected variant's fields, hide the rest, then notify.

        Args:
            index: The index of the newly selected variant.
        """
        for i, widget in enumerate(self._variant_widgets):
            widget.setVisible(i == index and index not in self._empty_indices)
        self._emit()

    def get_value(self) -> Any:
        """
        Construct the selected variant instance from its fields.

        Returns:
            An instance of the currently selected variant class, or - for a
            nested arm - of whichever inner variant that arm has selected.

        Raises:
            FieldValidationError: If the variant cannot be constructed from the
                form's values. Problems on fields of the variant's page are
                attributed to those fields' widgets.
        """
        index = self._button_group.checkedId()
        widget = self._variant_widgets[index]

        # a nested union builds its own leaf, already carrying our tag
        if index in self._nested_indices and isinstance(widget, FieldEditor):
            return widget.get_value()

        variant_cls = self._variant_classes[index]
        data = {self._discriminator_key: self._discriminator_values[index]}

        page = widget if isinstance(widget, ObjectWidget) else None
        if index not in self._empty_indices and page is not None:
            data.update(page.values_dict())

        try:
            return variant_cls(**data)
        except ValidationError as e:
            raise FieldValidationError.from_validation_error(
                e, page.field_widget if page is not None else lambda _: None
            ) from e

    def set_value(self, value: Any) -> None:
        """
        Select and populate the variant matching a value's discriminator.

        A nested arm receives the same value and reads its own discriminator
        from it, so a leaf of a nested union routes to the right page at both
        levels.

        Args:
            value: The variant instance to display.
        """
        disc_value = getattr(value, self._discriminator_key, None)
        if disc_value not in self._discriminator_values:
            return

        index = self._discriminator_values.index(disc_value)
        if index not in self._empty_indices:
            widget = self._variant_widgets[index]
            if isinstance(widget, FieldEditor):
                widget.set_value(value)

        btn = self._button_group.button(index)
        if btn:
            btn.setChecked(True)
        self._on_selection_changed(index)

    def select(self, discriminator_value: str) -> bool:
        """
        Select a variant by tag, without user interaction.

        Args:
            discriminator_value: The tag of the variant to show.

        Returns:
            True if a matching variant exists and is now selected; False if the
            value matches no variant, in which case the selection is unchanged.
        """
        if discriminator_value not in self._discriminator_values:
            return False

        index = self._discriminator_values.index(discriminator_value)
        # already there: returning early also breaks write-back re-entrancy
        if self._button_group.checkedId() == index:
            return True

        btn = self._button_group.button(index)
        if btn:
            btn.setChecked(True)
        self._on_selection_changed(index)
        return True

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable the radio buttons and all variant widgets.

        Args:
            read_only: If True, prevent variant selection and edits.
        """
        for btn in self._button_group.buttons():
            btn.setEnabled(not read_only)

        for widget in self._variant_widgets:
            if isinstance(widget, FieldEditor):
                widget.set_read_only(read_only)

    def selected_variant_widget(self) -> FieldEditor[Any] | None:
        """
        Return the selected variant's page.

        Returns:
            The `ObjectWidget` for the selected variant, the nested
            `DiscriminatedUnionWidget` for a nested arm, or None if that
            variant has no fields beyond the discriminators.
        """
        index = self._button_group.checkedId()
        if index < 0 or index in self._empty_indices:
            return None
        widget = self._variant_widgets[index]
        return widget if isinstance(widget, FieldEditor) else None

    def variant_widgets(self) -> list[FieldEditor[Any]]:
        """
        Return every variant's page, selected or not.

        Returns:
            The `ObjectWidget`s and nested union widgets of the variants that
            have pages.
        """
        return [w for w in self._variant_widgets if isinstance(w, FieldEditor)]

    def child_editors(self, name: str, all_variants: bool) -> list[FieldEditor[Any]]:
        """
        Return the editors of the field `name` in the variant pages.

        Args:
            name: The field name.
            all_variants: If True, look in every variant's page; otherwise only
                in the selected variant's page.

        Returns:
            The matching value editors; empty if no page has such a field.
        """
        if all_variants:
            pages = self.variant_widgets()
        else:
            selected = self.selected_variant_widget()
            pages = [] if selected is None else [selected]
        return [
            child
            for page in pages
            for child in page.value_editor().child_editors(name, all_variants)
        ]

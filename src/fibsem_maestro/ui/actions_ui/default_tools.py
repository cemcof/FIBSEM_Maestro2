# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The tools every action's header shows by default: Prepare, Edit, Apply and Test."""

from PyQt6.QtWidgets import QMessageBox

from fibsem_maestro.ui.actions_ui.spec import ToolContext, ToolSpec
from fibsem_maestro.ui.app.errors import PropertiesChangedError
from fibsem_maestro.ui.widgets.icons import (
    ICON_APPLY,
    ICON_EDIT,
    ICON_PREPARE,
    ICON_TEST,
)
from fibsem_maestro.ui.widgets.properties_dialog import PropertiesDialog


def confirm_state_loss(ctx: ToolContext) -> bool:
    """
    Ask before new properties of the action discard the internal state of actions.

    Nobody is asked if no action would lose state (e.g. before the first
    slice), or in danger mode.

    Args:
        ctx: The tool's context.

    Returns:
        True if the properties may be stored.
    """
    losing = ctx.session.actions_losing_state_to(ctx.action)
    if not losing or ctx.policy.danger_mode:
        return True

    names = ", ".join(f"'{action.name}'" for action in losing)
    answer = QMessageBox.question(
        ctx.parent,
        "Reset state",
        f"New properties of '{ctx.action.name}' reset the internal state of "
        f"{names}.\n\nContinue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes


def prepare(ctx: ToolContext) -> None:
    """
    Prepare the action on the microscope task thread, after confirming any state loss.

    Args:
        ctx: The tool's context.
    """
    if confirm_state_loss(ctx):
        ctx.commands.prepare_action(ctx.action)


def edit_properties(ctx: ToolContext) -> None:
    """
    Open the props stored for the action's current slice for editing, and save them.

    In danger mode this works while the workflow runs. If the stored props
    change before the user saves (the workflow executed the action, or
    propagated to it), the edits are not saved and the user is told so.
    Saving asks first if the new props reset the internal state of an action
    (`confirm_state_loss`). Other failures to read or save are logged on the
    action's logger.

    Args:
        ctx: The tool's context.
    """
    action = ctx.action
    # the file may have disappeared since the buttons were last updated
    if not action.has_stored_properties():
        ctx.refresh()
        return

    opened_slice = action.ctx.slice
    try:
        props = action.read_properties()
    except Exception as e:
        action.ctx.text_logger.error(
            f"Reading properties for '{action.name}' failed: {e}"
        )
        return
    # the dialog edits `props` in place
    opened = props.model_copy(deep=True)

    edited = PropertiesDialog.review(
        properties=props,
        form_builder=ctx.form_builder,
        form_context=ctx.form_context,
        title=f"Properties of '{action.name}'",
        hint=(
            f"Properties stored for slice {opened_slice}. "
            "Edit them if needed, then save. If they change before you save, "
            "the edits are not saved."
        ),
        parent=ctx.parent,
    )
    if edited is None or not confirm_state_loss(ctx):
        return

    try:
        ctx.commands.save_properties(
            action, edited, opened_slice=opened_slice, opened=opened
        )
    except PropertiesChangedError as e:
        QMessageBox.warning(ctx.parent, "Properties not saved", str(e))
        ctx.refresh()
    except Exception as e:
        action.ctx.text_logger.error(
            f"Saving properties for '{action.name}' failed: {e}"
        )


def apply_properties(ctx: ToolContext) -> None:
    """
    Set the props stored for the action's current slice on the microscope.

    Args:
        ctx: The tool's context.
    """
    # the file may have disappeared since the buttons were last updated
    if not ctx.action.has_stored_properties():
        ctx.refresh()
        return

    ctx.commands.apply_properties(ctx.action)


def test(ctx: ToolContext) -> None:
    """
    Run the action once outside the acquisition, on the microscope task thread.

    Args:
        ctx: The tool's context.
    """
    ctx.commands.test_action(ctx.action)


PREPARE_TOOL = ToolSpec(
    key="prepare",
    icon=ICON_PREPARE,
    tooltip=(
        "<b>Prepare</b><br>"
        "Capture the action's properties from the microscope and store them "
        "for the current slice, replacing any stored ones. Some actions do "
        "more, e.g. drift correction also acquires its reference templates.<br>"
        "<i>Microscope -> stored properties</i>"
    ),
    accessible_name="Prepare action",
    requires_props=False,
    invoke=prepare,
)

EDIT_TOOL = ToolSpec(
    key="edit",
    icon=ICON_EDIT,
    tooltip=(
        "<b>Edit</b><br>"
        "View and edit the properties stored for the current slice.<br>"
        "Unavailable until properties are stored. Also available while the "
        "workflow runs, in danger mode."
    ),
    accessible_name="Edit properties",
    requires_props=True,
    invoke=edit_properties,
    usable_in_danger_mode=True,
)

APPLY_TOOL = ToolSpec(
    key="apply",
    icon=ICON_APPLY,
    tooltip=(
        "<b>Apply</b><br>"
        "Set the properties stored for the current slice on the microscope.<br>"
        "<i>Stored properties -> microscope</i><br>"
        "Unavailable until properties are stored."
    ),
    accessible_name="Apply properties",
    requires_props=True,
    invoke=apply_properties,
)

TEST_TOOL = ToolSpec(
    key="test",
    icon=ICON_TEST,
    tooltip=(
        "<b>Test</b><br>"
        "Run the action once outside the acquisition, using the stored "
        "properties if there are any."
    ),
    accessible_name="Test action",
    requires_props=False,
    invoke=test,
)

DEFAULT_TOOLS: tuple[ToolSpec, ...] = (PREPARE_TOOL, EDIT_TOOL, APPLY_TOOL, TEST_TOOL)
"""Prepare, Edit, Apply and Test, in header order."""

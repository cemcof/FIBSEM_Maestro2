# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""Builds the area selector of an AREA_SELECT field, with its context providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from fibsem_maestro.core.beam_type import BeamType
from fibsem_maestro.settings.reactive import ReactiveNode
from fibsem_maestro.ui.forms.editors.area_select.stale_watch import StaleImageWatch
from fibsem_maestro.ui.forms.editors.area_select.widget import AreaSelectWidget
from fibsem_maestro.ui.forms.registry import optional_wrap

if TYPE_CHECKING:
    from collections.abc import Callable

    from fibsem_maestro.core.area import RelativeArea
    from fibsem_maestro.settings.form_utils import BeamSource, CropSource
    from fibsem_maestro.ui.forms.editor import FieldEditor
    from fibsem_maestro.ui.forms.registry import EditorRequest
    from fibsem_maestro.ui.forms.scope import BuildScope
    from fibsem_maestro.ui.ports import ActionCatalog


class _Crop(NamedTuple):
    """
    A resolved preview crop.

    Attributes:
        area: The scanning area to crop to, or None for the whole frame.
        linked: True if the area came from a linked action's settings, whose
            frames are taken at that action's own position.
    """

    area: RelativeArea | None
    linked: bool


def create_area_select(request: EditorRequest) -> FieldEditor[Any]:
    """
    Build the editor of an AREA_SELECT field.

    The beam, acquisition offset and crop of the preview are read from the
    live settings whenever Load is pressed. A form without live settings
    cannot resolve them; each one declared is then dropped with a warning.

    Args:
        request: The field's request; the field must carry a `FormHint`.

    Returns:
        The area selector, wrapped in an inline optional checkbox if the
        field is optional, with `on_change` connected.

    Raises:
        ValueError: If the field has no hint.
    """
    fi = request.field
    hint = fi.hint
    if hint is None:
        raise ValueError(f"no hint for field {fi.name!r}")
    scope = request.scope
    ctx = request.ctx

    beam_provider: Callable[[], BeamType | None] | None = None
    if hint.beam_sources:
        if not scope.is_live:
            ctx.warn(
                f"Field {fi.name!r} declares beam sources but is not bound to "
                "live settings; the active beam is used."
            )
        else:
            beam_sources = hint.beam_sources

            def beam_provider() -> BeamType | None:
                """The beam to acquire the preview with, or None for the active one."""
                return _resolve_beam(scope, beam_sources, ctx.actions)

    crop_sources: tuple[CropSource, ...] = ()
    crop_provider: Callable[[], RelativeArea | None] | None = None
    if hint.crop_sources:
        if not scope.is_live:
            ctx.warn(
                f"Field {fi.name!r} declares crop sources but is not "
                "bound to live settings; the whole frame is shown."
            )
        else:
            crop_sources = hint.crop_sources

            def crop_provider() -> RelativeArea | None:
                """The scanning area to crop the preview to, or None."""
                return _resolve_crop(scope, crop_sources, ctx.actions).area

    offset_provider: Callable[[], float | None] | None = None
    if (offset := hint.offset) is not None:
        if not scope.is_live:
            ctx.warn(
                f"Field {fi.name!r} declares an acquisition offset "
                "but is not bound to live settings; images are "
                "grabbed at the current beam position."
            )
        else:
            offset_path = offset.source

            def offset_provider() -> float | None:
                """X offset applied when grabbing, in nm; None if unresolved."""
                # frames acquired by a linked action are taken at its position
                if (
                    crop_sources
                    and _resolve_crop(scope, crop_sources, ctx.actions).linked
                ):
                    return 0.0
                return scope.value(offset_path, float)

    area = AreaSelectWidget(
        previews=ctx.previews,
        preview_permission=ctx.preview_permission,
        min_areas=hint.min_areas if hint.min_areas else None,
        max_areas=hint.max_areas if hint.max_areas else None,
        default=request.value,
        beam_provider=beam_provider,
        offset_provider=offset_provider,
        crop_provider=crop_provider,
    )

    # an offset changed after Load makes the displayed image stale
    if offset_provider is not None and isinstance(scope.root, ReactiveNode):
        watch = StaleImageWatch.attach(area, scope.root)
        scope.context.disposers.append(watch.release)

    return optional_wrap(area, request, inline=True)


def _resolve_beam(
    scope: BuildScope,
    sources: tuple[BeamSource, ...],
    actions: ActionCatalog | None,
) -> BeamType | None:
    """
    Find the beam an area selector's preview is acquired with.

    Sources are tried in order against the live settings; the first that
    resolves to a beam decides.

    Args:
        scope: The scope of the area selector's field.
        sources: The field's beam sources.
        actions: The workflow's actions, for sources reached through a link.

    Returns:
        The beam, or None if no source resolves (the active beam is used).
    """
    for source in sources:
        if source.via is None:
            beam = scope.value(source.path, BeamType)
        else:
            beam = _linked_value(scope, actions, source.via, source.path, BeamType)

        if beam is not None:
            return beam
    return None


def _resolve_crop(
    scope: BuildScope,
    sources: tuple[CropSource, ...],
    actions: ActionCatalog | None,
) -> _Crop:
    """
    Find the scanning area an area selector's preview is cropped to.

    Sources are tried in order against the live settings; the first that
    resolves to a list decides, even an empty one (the whole frame).

    Args:
        scope: The scope of the area selector's field.
        sources: The field's crop sources.
        actions: The workflow's actions, for sources reached through a link.

    Returns:
        The crop, and whether it came from a linked action's settings.
    """
    for source in sources:
        if source.via is None:
            areas = scope.value(source.path, list)
        else:
            areas = _linked_value(scope, actions, source.via, source.path, list)

        if areas is not None:
            return _Crop(areas[0] if areas else None, source.via is not None)
    return _Crop(None, False)


def _linked_value[T](
    scope: BuildScope,
    actions: ActionCatalog | None,
    via: str,
    path: str,
    kind: type[T],
) -> T | None:
    """
    Resolve a path in the settings of the action a link field names.

    Args:
        scope: The scope the link field is resolved from.
        actions: The workflow's actions; None outside a workflow.
        via: Path to the field holding the linked action's name.
        path: Path to resolve in that action's settings.
        kind: Required type of the resolved value.

    Returns:
        The value of type `kind` found there, or None if the link or the path
        does not resolve.
    """
    if actions is None or (name := scope.value(via, str)) is None:
        return None

    action = next((a for a in actions.actions if a.name == name), None)
    if action is None:
        return None

    return scope.rebased(action.settings).value(path, kind)

# Released under GPL-3.0 License.
# Copyright (c) 2024-2026 CEMCOF

"""The area selector: an editor of acquisition areas drawn over a preview image."""

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem_maestro.core.area import RelativeArea
from fibsem_maestro.core.image import Image
from fibsem_maestro.core.point import RelativePoint
from fibsem_maestro.ui.forms.editor import FieldEditor
from fibsem_maestro.ui.forms.editors.area_select._mipmap import (
    MipmapPixmapItem,
)
from fibsem_maestro.ui.forms.editors.area_select._rectangle import (
    ResizableRect,
)
from fibsem_maestro.ui.forms.editors.area_select._viewer import AreaViewer
from fibsem_maestro.ui.forms.editors.area_select.overlays import (
    AreaDecoration,
    OverlayValues,
    build_decorations,
    overlay_problems,
)
from fibsem_maestro.ui.forms.issues import FieldIssue, Severity
from fibsem_maestro.ui.ports import PreviewPermission, PreviewPort, PreviewRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fibsem_maestro.core.beam_type import BeamType
    from fibsem_maestro.settings.form_utils import AreaOverlay

_OVERLAY_ISSUE = "overlay"
"""Source key of the warning this widget reports when an overlay cannot be drawn."""

_STALE_ISSUE = "stale_image"
"""Source key of the warning that the image was loaded at a different offset."""

_MIN_AREAS_ISSUE = "min_areas"
"""Source key of the error that the field holds fewer areas than required."""

_NOT_CONNECTED = "FIBSEM Maestro is not connected to a microscope."
"""Why Load fails in a form built without a preview port."""


class AreaSelectWidget(QWidget, FieldEditor[list[RelativeArea]]):
    """
    Accordion-style selector for rectangular acquisition areas.

    Collapsed, it shows a thumbnail with a region overlay.
    Expanded, it shows an interactive viewer for drawing, moving, resizing,
    and deleting areas over the last acquired image.

    Args:
        previews: Acquires the image on Load; None if no microscope is
            connected, in which case Load reports that.
        preview_permission: Whether Load may acquire now; None to always allow.
        min_areas: Fewest areas the field must hold; fewer is an error.
        max_areas: Most areas that can be drawn.
        default: The initial areas, relative to the full frame.
        beam_provider: The beam to image with; None (or returning None) for
            the microscope's active beam.
        offset_provider: Beam shift along x applied while grabbing, in nm;
            None (or returning None) for no shift.
        crop_provider: Scanning area the preview is cropped to, relative to
            the full frame; None (or returning None) for the full frame.
        parent: The parent widget, if any.
    """

    _THUMBNAIL_HEIGHT = 100
    _EXPANDED_HEIGHT = 600
    _MINIMUM_WIDTH = 800

    def __init__(
        self,
        previews: PreviewPort | None,
        preview_permission: PreviewPermission | None = None,
        min_areas: int | None = None,
        max_areas: int | None = None,
        default: list[RelativeArea] | None = None,
        beam_provider: Callable[[], BeamType | None] | None = None,
        offset_provider: Callable[[], float | None] | None = None,
        crop_provider: Callable[[], RelativeArea | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        FieldEditor.__init__(self)

        self._previews = previews
        self._preview_permission = preview_permission
        self._min_areas = min_areas
        self._max_areas = max_areas
        self._image_size: tuple[int, int] | None = None
        self._last_pixmap: QPixmap | None = None
        self._pending_regions: list[RelativeArea] = default or []
        self._expanded = False
        self._read_only = False

        # True from Load until the preview port answers
        self._loading = False
        # acquisition offset of the requested image, in nm
        self._requested_offset = 0.0
        # acquisition offset of the displayed image, in nm; None before any Load
        self._loaded_offset: float | None = None

        self._overlays: list[tuple[AreaOverlay, OverlayValues]] = []
        self._pixel_size: float | None = None
        self._beam_provider = beam_provider
        self._offset_provider = offset_provider
        self._crop_provider = crop_provider

        self.setMinimumWidth(self._MINIMUM_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # header row (always visible)
        header = QWidget()
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.addStretch()

        self._load_btn = QPushButton("Load")
        self._load_btn.setFixedWidth(70)
        self._load_btn.clicked.connect(self._load_image)
        header_layout.addWidget(self._load_btn)

        self._toggle_btn = QPushButton("▼ Expand")
        self._toggle_btn.setFixedWidth(80)
        self._toggle_btn.clicked.connect(self._toggle)
        header_layout.addWidget(self._toggle_btn)
        outer.addWidget(header)

        # thumbnail (used when collapsed)
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setFixedHeight(self._THUMBNAIL_HEIGHT)
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setStyleSheet(
            "background: #1a1a1a; border: 0.5px solid #444; border-radius: 4px;"
        )
        self._thumbnail_label.setText("No image")
        self._thumbnail_label.mousePressEvent = lambda _: self._toggle()  # type: ignore
        self._thumbnail_label.setCursor(Qt.CursorShape.PointingHandCursor)
        outer.addWidget(self._thumbnail_label)

        # expanded viewer (initially hidden)
        self._viewer_container = QWidget()
        self._viewer_container.hide()
        viewer_layout = QVBoxLayout(self._viewer_container)
        viewer_layout.setContentsMargins(0, 4, 0, 0)
        viewer_layout.setSpacing(4)

        status_box = QHBoxLayout()
        status_box.addStretch()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 11px;")
        status_box.addWidget(self._status_label)
        viewer_layout.addLayout(status_box)

        self._scene = QGraphicsScene()
        self._scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        # scene.changed drives ONLY the thumbnail
        # committing changes is done via the viewer's edit-finished callback,
        # so a drag does not spam writes
        self._scene.changed.connect(self._on_scene_changed)
        self._viewer = AreaViewer(
            self._scene,
            self._status_label.setText,
            self._min_areas,
            self._max_areas,
            on_edit_finished=self._handle_edit_finished,
            decoration_factory=self._build_decorations,
        )
        self._viewer.setFixedHeight(self._EXPANDED_HEIGHT)
        self._viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        viewer_layout.addWidget(self._viewer)
        outer.addWidget(self._viewer_container)

        self._check_min_areas()

        if preview_permission is not None:
            # a bound method: Qt drops the connection when this widget is destroyed
            preview_permission.changed.connect(self._on_preview_permission_changed)
        self._update_load_button()

    def set_overlays(
        self, overlays: Sequence[tuple[AreaOverlay, OverlayValues]]
    ) -> None:
        """
        Replace the decorations drawn on every area and redraw them live.

        Areas that no longer fit the frame together with the new overlays'
        footprint (e.g. after the margin grew) are shrunk to fit, and that
        correction is emitted as a change.

        Args:
            overlays: Overlay kinds paired with their runtime values (e.g.
                margin in nm, arrow direction). An overlay whose values are
                missing is silently not drawn, so a partially configured form
                still shows the overlays that are ready.
        """
        self._overlays = list(overlays)
        if self._refresh_decorations():
            self._emit()

    def get_value(self) -> list[RelativeArea]:
        """
        Return the current areas as image-relative fractions.

        Before an image is loaded, returns the pending regions unchanged.
        Coordinates are clamped to the unit square.

        Returns:
            The areas currently defined, in scene draw order.
        """
        if self._image_size is None:
            return self._pending_regions
        w, h = self._image_size
        result: list[RelativeArea] = []

        # reversed so reloaded workflows keep the original ordering
        for item in reversed(list(self._scene.items())):
            if isinstance(item, ResizableRect):
                sr = item.scene_rect()
                result.append(
                    RelativeArea(
                        origin=RelativePoint(
                            x=max(0.0, min(1.0, sr.x() / w)),
                            y=max(0.0, min(1.0, sr.y() / h)),
                        ),
                        width=min(1.0, sr.width() / w),
                        height=min(1.0, sr.height() / h),
                    )
                )
        return result

    def set_value(self, value: list[RelativeArea]) -> None:
        """
        Replace the current areas.

        If no image is loaded yet, the areas are stored as pending and realized
        on the next `convert_image`. No change is emitted, unless an area had to
        be shrunk to fit the frame and the footprint of its overlays: that
        correction is emitted, so the settings match what is shown.

        Args:
            value: The areas to display, or empty list to clear.
        """
        self._clear_rects()
        regions = value or []

        if self._image_size is None:
            self._pending_regions = regions
        else:
            for area in regions:
                self._add_relative_area(area)
            if self._refresh_decorations():
                self._emit()

        self._check_min_areas()

    def set_read_only(self, read_only: bool) -> None:
        """
        Freeze the areas while leaving navigation and collapse available.

        In read-only mode the user can still expand/collapse, zoom, and pan, but
        cannot load a new image, draw, delete, move, or resize areas. Resize
        handles are hidden, since editing is disabled.

        Args:
            read_only: True to freeze area editing and image loading.
        """
        self._read_only = read_only

        self._update_load_button()
        # toggle stays enabled: collapse/expand is navigation, not editing
        self._viewer.set_read_only(read_only)
        for item in self._scene.items():
            if isinstance(item, ResizableRect):
                item.set_read_only(read_only)

    def _toggle(self) -> None:
        """Toggle between the collapsed thumbnail and the expanded viewer."""
        self._expanded = not self._expanded
        self._thumbnail_label.setVisible(not self._expanded)
        self._viewer_container.setVisible(self._expanded)
        self._toggle_btn.setText("▲ Collapse" if self._expanded else "▼ Expand")

        if not self._expanded:
            self._update_thumbnail()

    def _load_image(self) -> None:
        """Request a preview image; it is shown when the preview port delivers it."""
        if self._previews is None:
            self._status_label.setText(f"Acquisition failed: {_NOT_CONNECTED}")
            return

        self._load_btn.setEnabled(False)
        self._status_label.setText("Loading image...")
        try:
            # the providers read the live settings, so the request is built here,
            # on the GUI thread, and not where the image is acquired
            request = PreviewRequest(
                beam=self._beam_provider() if self._beam_provider is not None else None,
                offset_x_nm=self._resolve_offset(),
                previous_offset_nm=self._loaded_offset,
                crop=self._resolve_crop(),
            )
        except Exception as e:
            self._status_label.setText(f"Acquisition failed: {e}")
            self._update_load_button()
            return

        self._loading = True
        self._requested_offset = request.offset_x_nm
        handle = self._previews.grab(request)
        # bound methods: if this widget is destroyed first, Qt drops the
        # connections and the late result is discarded
        handle.succeeded.connect(self._on_preview_image)
        handle.failed.connect(self._on_preview_error)

    def _on_preview_image(self, image: object) -> None:
        """
        Show a delivered preview image.

        Args:
            image: The acquired image, already cropped to the requested area.
        """
        self._loading = False
        try:
            if not isinstance(image, Image):
                raise TypeError(f"expected an Image, got {type(image).__name__}")
            self.convert_image(image)
            self._loaded_offset = self._requested_offset
            self.check_stale()
        except Exception as e:
            self._status_label.setText(f"Acquisition failed: {e}")
        finally:
            self._update_load_button()

    def _on_preview_error(self, error: object) -> None:
        """
        Report a failed or refused acquisition.

        Args:
            error: The exception raised by the acquisition, or the refusal.
        """
        self._loading = False
        self._status_label.setText(f"Acquisition failed: {error}")
        self._update_load_button()

    def _resolve_offset(self) -> float:
        """
        Return the x offset to apply while grabbing, in nanometres.

        Returns:
            The declared offset, or 0.0 when none is declared or the declared
            one does not resolve: the field it names exists only where an
            offset applies (e.g. not in imaging or step-mode autofunction).
        """
        if self._offset_provider is None:
            return 0.0

        offset_nm = self._offset_provider()
        return 0.0 if offset_nm is None else offset_nm

    def _resolve_crop(self) -> RelativeArea | None:
        """
        Return the scanning area to crop a loaded image to.

        Returns:
            The area, relative to the grabbed frame, or None to show the whole frame.
        """
        return self._crop_provider() if self._crop_provider is not None else None

    def check_stale(self) -> None:
        """
        Warn on this field if the acquisition offset changed since the image was loaded.

        The areas stay relative to the image as loaded, so the warning asks
        for a new Load rather than changing anything. Without a loaded image
        nothing is shown or cleared. Safe to call at any time; the owner
        calls it whenever the offset may have changed.
        """
        if self._loaded_offset is None:
            return

        current = self._resolve_offset()
        self.set_issue(
            _STALE_ISSUE,
            None
            if current == self._loaded_offset
            else FieldIssue(
                Severity.WARNING,
                f"Image loaded at an offset of {self._loaded_offset:g} nm, the "
                f"offset is now {current:g} nm. Load it again.",
            ),
        )

    def convert_image(self, image: Image) -> None:
        """
        Display an image, keeping the current areas in place relative to the frame.

        Areas are scene items in pixels of the displayed image, so they are read
        back as relative areas before the image is replaced and re-created in
        pixels of the new one. A reload at a different resolution or aspect
        ratio therefore keeps every area at the same fraction of the frame.

        Args:
            image: The image to display beneath the area overlay.
        """

        arr = np.ascontiguousarray(image.to_8bit())
        h, w = arr.shape[:2]

        # read the areas in the old image's pixels before the size changes;
        # before any image, this returns the pending regions
        regions = self.get_value()

        self._image_size = (w, h)
        self._pixel_size = image.pixel_size

        if arr.ndim == 2:
            q_image = QImage(
                arr.tobytes(), w, h, arr.strides[0], QImage.Format.Format_Grayscale8
            ).copy()
        else:
            q_image = QImage(
                arr.tobytes(), w, h, arr.strides[0], QImage.Format.Format_RGB888
            ).copy()

        self._last_pixmap = QPixmap.fromImage(q_image)

        # replace the whole scene: the background and the area rectangles,
        # which are re-created below in pixels of the new image
        for item in list(self._scene.items()):
            if item.parentItem() is None:
                self._scene.removeItem(item)

        # a mipmapped item: Qt's bilinear filter only reads a 2x2 neighbourhood
        # however far the image is minified, so a 6144x4096 frame in an 885 px
        # viewport carries pixel noise through at full amplitude instead of
        # averaging it away
        pixmap_item = MipmapPixmapItem(self._last_pixmap)
        pixmap_item.setZValue(-1)
        self._scene.addItem(pixmap_item)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._viewer.reset_zoom()

        for area in regions:
            self._add_relative_area(area)
        self._pending_regions = []

        self._refresh_decorations()
        self._viewer.set_image_loaded()
        self._status_label.setText(f"Image: {w}×{h} px")

        # write back: areas may have been shrunk to fit the new frame and the
        # footprint of their overlays at the new pixel size
        self._emit()

    def _on_scene_changed(self, regions: list[QRectF]) -> None:
        """Refresh the thumbnail on any scene change (whatever the changed regions)."""
        _ = regions
        if not self._expanded:
            self._update_thumbnail()

    def _update_thumbnail(self) -> None:
        """Render the current scene into the collapsed thumbnail."""
        if self._last_pixmap is None:
            return

        scene_rect = self._scene.sceneRect()
        aspect = (
            scene_rect.width() / scene_rect.height() if scene_rect.height() > 0 else 1.0
        )
        thumb_w = int(self._THUMBNAIL_HEIGHT * aspect)
        thumb_h = self._THUMBNAIL_HEIGHT

        # render at a higher resolution, then downscale with a smoothing filter
        supersample = 2
        hi = QPixmap(thumb_w * supersample, thumb_h * supersample)
        hi.fill(Qt.GlobalColor.black)

        rects = [it for it in self._scene.items() if isinstance(it, ResizableRect)]
        for r in rects:
            r.set_handles_visible(False)
        try:
            painter = QPainter(hi)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self._scene.render(
                painter, QRectF(0, 0, hi.width(), hi.height()), scene_rect
            )
            painter.end()
        finally:
            for r in rects:
                r.restore_handles()

        thumbnail = hi.scaled(
            thumb_w,
            thumb_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail_label.setPixmap(thumbnail)

    def _add_relative_area(self, area: RelativeArea) -> None:
        """
        Add one relative area to the scene as an interactive rectangle.

        Args:
            area: The area to add, in image-relative fractions.
        """
        if self._image_size is None:
            return

        w, h = self._image_size
        bounds = QRectF(0, 0, w, h)
        rect = ResizableRect(
            bounds.intersected(
                QRectF(
                    area.origin.x * w,
                    area.origin.y * h,
                    area.width * w,
                    area.height * h,
                )
            ),
            on_edit_finished=self._handle_edit_finished,
        )

        rect.set_read_only(self._read_only)

        self._scene.addItem(rect)
        self._update_thumbnail()

    def _clear_rects(self) -> None:
        """Remove all area rectangles from the scene."""
        for item in list(self._scene.items()):
            if isinstance(item, ResizableRect):
                self._scene.removeItem(item)
        self._update_thumbnail()

    def _handle_edit_finished(self) -> None:
        """Decorate any newly created areas, then emit the change."""
        self._refresh_decorations()
        self._check_min_areas()
        self._emit()

    def _check_min_areas(self) -> None:
        """Mark this field as an error while it holds fewer areas than `min_areas`."""
        if self._min_areas is None:
            return

        missing = len(self.get_value()) < self._min_areas
        self.set_issue(
            _MIN_AREAS_ISSUE,
            FieldIssue(Severity.ERROR, f"At least {self._min_areas} area(s) required.")
            if missing
            else None,
        )

    def _build_decorations(self) -> list[AreaDecoration]:
        """
        Build one area's decorations from the current overlays and image scale.

        Returns:
            Fresh decoration instances; each area needs its own.
        """
        return build_decorations(self._overlays, self._pixel_size)

    def _refresh_decorations(self) -> bool:
        """
        Rebuild every rectangle's decorations, then fit each into its new bounds.

        Also reports, as a warning on this field, every overlay that cannot be
        drawn: a configuration the criterion would reject, or a decoration that
        does not fit its area.

        Returns:
            True if any area was resized or moved to fit the frame less the
            footprint of its decorations. The caller decides whether to emit.
        """
        fitted = False
        problems = overlay_problems(self._overlays, self._pixel_size)
        for item in self._scene.items():
            if isinstance(item, ResizableRect):
                item.apply_decorations(self._build_decorations())
                fitted |= item.fit_to_bounds()
                problems.extend(item.decoration_problems())

        unique = list(dict.fromkeys(problems))
        self.set_issue(
            _OVERLAY_ISSUE,
            FieldIssue(Severity.WARNING, "\n".join(unique)) if unique else None,
        )

        self._update_thumbnail()
        return fitted

    def _on_preview_permission_changed(self, *_: object) -> None:
        """Re-evaluate Load when previews may have become allowed or forbidden."""
        self._update_load_button()

    def _update_load_button(self) -> None:
        """Enable Load only when areas may be edited, previews are allowed and none is loading."""
        allowed = (
            self._preview_permission is None
            or self._preview_permission.preview_allowed()
        )
        self._load_btn.setEnabled(not self._read_only and allowed and not self._loading)

"""
Manual segmentation widget for napari.

Provides interactive tools for manually segmenting cells using:
- Paint brush (freehand drawing)
- Rectangle selection
- Polygon selection
- Eraser

Users can draw regions, finalize (relabel) the mask, and save it to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QSpinBox, QGroupBox, QFileDialog,
    QMessageBox,
)
from qtpy.QtCore import Qt
import napari
from napari.utils.notifications import show_info


class ManualSegmentationWidget(QWidget):
    """
    A Qt widget for manual cell segmentation in napari.

    Workflow:
    1. Select an image layer from the dropdown
    2. Click "New Label Layer" to create an empty mask
    3. Use Paint / Rectangle / Polygon tools to mark cells
    4. Click "Convert Shapes → Labels" to rasterise any shapes drawn
    5. Click "Finalize Mask" to relabel connected regions sequentially
    6. Click "Save Mask" to export the mask as a .tif file
    """

    # Default pixels to pan per arrow-key press
    _DEFAULT_PAN_STEP = 50

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Internal state
        self._labels_layer: Optional[napari.layers.Labels] = None
        self._shapes_layer: Optional[napari.layers.Shapes] = None
        self._current_label_id: int = 1
        self._prev_mode: Optional[str] = None  # for Space toggle

        self._build_ui()
        self._connect_signals()
        self._register_keybindings()

        # Refresh image list when layers change
        self.viewer.layers.events.inserted.connect(self._refresh_image_list)
        self.viewer.layers.events.removed.connect(self._refresh_image_list)
        self._refresh_image_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ---- Image selection ----
        grp_image = QGroupBox("Image Selection")
        grp_image_layout = QVBoxLayout()

        self.combo_image = QComboBox()
        self.combo_image.setToolTip("Select the image layer to segment")
        grp_image_layout.addWidget(QLabel("Image Layer:"))
        grp_image_layout.addWidget(self.combo_image)

        self.btn_new_label = QPushButton("✦ New Label Layer")
        self.btn_new_label.setToolTip(
            "Create an empty labels layer matching the selected image"
        )
        grp_image_layout.addWidget(self.btn_new_label)

        grp_image.setLayout(grp_image_layout)
        layout.addWidget(grp_image)

        # ---- Drawing tools ----
        grp_tools = QGroupBox("Drawing Tools")
        grp_tools_layout = QVBoxLayout()

        # Tool buttons row 1: Paint / Erase / Fill / Pan
        row1 = QHBoxLayout()
        self.btn_paint = QPushButton("🖌 Paint")
        self.btn_paint.setToolTip("Freehand brush painting on labels layer")
        self.btn_erase = QPushButton("🧹 Erase")
        self.btn_erase.setToolTip("Erase labels")
        self.btn_fill = QPushButton("🪣 Fill")
        self.btn_fill.setToolTip("Flood-fill a closed region")
        self.btn_pan = QPushButton("🖐 Pan")
        self.btn_pan.setToolTip("Switch to pan mode to drag the canvas (Space to toggle)")
        row1.addWidget(self.btn_paint)
        row1.addWidget(self.btn_erase)
        row1.addWidget(self.btn_fill)
        row1.addWidget(self.btn_pan)
        grp_tools_layout.addLayout(row1)

        # Tool buttons row 2: Rectangle / Polygon / Ellipse
        row2 = QHBoxLayout()
        self.btn_rect = QPushButton("▭ Rectangle")
        self.btn_rect.setToolTip("Draw a rectangle shape to define a cell region")
        self.btn_polygon = QPushButton("⬠ Polygon")
        self.btn_polygon.setToolTip("Draw a polygon shape to define a cell region")
        self.btn_ellipse = QPushButton("⬭ Ellipse")
        self.btn_ellipse.setToolTip("Draw an ellipse shape to define a cell region")
        row2.addWidget(self.btn_rect)
        row2.addWidget(self.btn_polygon)
        row2.addWidget(self.btn_ellipse)
        grp_tools_layout.addLayout(row2)

        # Brush size
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush Size:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(5)
        self.spin_brush.setToolTip("Paint brush diameter in pixels")
        brush_row.addWidget(self.spin_brush)
        grp_tools_layout.addLayout(brush_row)

        # Pan step size
        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan Step (px):"))
        self.spin_pan_step = QSpinBox()
        self.spin_pan_step.setRange(10, 500)
        self.spin_pan_step.setValue(self._DEFAULT_PAN_STEP)
        self.spin_pan_step.setSingleStep(10)
        self.spin_pan_step.setToolTip("Pixels to move per arrow-key press")
        pan_row.addWidget(self.spin_pan_step)
        grp_tools_layout.addLayout(pan_row)

        # Label ID selector
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Current Label ID:"))
        self.spin_label_id = QSpinBox()
        self.spin_label_id.setRange(1, 10000)
        self.spin_label_id.setValue(1)
        self.spin_label_id.setToolTip(
            "Each unique ID represents one cell. "
            "Increment this to start painting a new cell."
        )
        label_row.addWidget(self.spin_label_id)

        self.btn_next_label = QPushButton("+ Next Cell")
        self.btn_next_label.setToolTip("Increment label ID to start a new cell")
        label_row.addWidget(self.btn_next_label)
        grp_tools_layout.addLayout(label_row)

        grp_tools.setLayout(grp_tools_layout)
        layout.addWidget(grp_tools)

        # ---- Actions ----
        grp_actions = QGroupBox("Actions")
        grp_actions_layout = QVBoxLayout()

        self.btn_shapes_to_labels = QPushButton("⊞ Convert Shapes → Labels")
        self.btn_shapes_to_labels.setToolTip(
            "Rasterise all shapes (rectangles / polygons / ellipses) "
            "into the labels layer, then clear the shapes layer."
        )
        grp_actions_layout.addWidget(self.btn_shapes_to_labels)

        self.btn_finalize = QPushButton("✓ Finalize Mask")
        self.btn_finalize.setToolTip(
            "Relabel connected components sequentially (1, 2, 3, …) "
            "and display the cell count."
        )
        grp_actions_layout.addWidget(self.btn_finalize)

        self.btn_save = QPushButton("💾 Save Mask")
        self.btn_save.setToolTip("Save the current labels layer as a .tif file")
        grp_actions_layout.addWidget(self.btn_save)

        grp_actions.setLayout(grp_actions_layout)
        layout.addWidget(grp_actions)

        # ---- Shortcuts hint ----
        grp_shortcuts = QGroupBox("Keyboard Shortcuts")
        shortcuts_layout = QVBoxLayout()
        shortcuts_text = QLabel(
            "<b>Arrow keys / WASD</b>: Pan view<br>"
            "<b>Space</b>: Toggle Pan ↔ previous tool<br>"
            "<b>+ / -</b>: Zoom in / out"
        )
        shortcuts_text.setWordWrap(True)
        shortcuts_layout.addWidget(shortcuts_text)
        grp_shortcuts.setLayout(shortcuts_layout)
        layout.addWidget(grp_shortcuts)

        # ---- Status ----
        self.lbl_status = QLabel("Ready. Select an image and create a label layer.")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()
        self.setLayout(layout)

        # Let the widget receive key events
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.btn_new_label.clicked.connect(self._on_new_label_layer)
        self.btn_paint.clicked.connect(self._on_paint)
        self.btn_erase.clicked.connect(self._on_erase)
        self.btn_fill.clicked.connect(self._on_fill)
        self.btn_pan.clicked.connect(self._on_pan)
        self.btn_rect.clicked.connect(self._on_rect)
        self.btn_polygon.clicked.connect(self._on_polygon)
        self.btn_ellipse.clicked.connect(self._on_ellipse)
        self.btn_next_label.clicked.connect(self._on_next_label)
        self.btn_shapes_to_labels.clicked.connect(self._on_shapes_to_labels)
        self.btn_finalize.clicked.connect(self._on_finalize)
        self.btn_save.clicked.connect(self._on_save)
        self.spin_brush.valueChanged.connect(self._on_brush_size_changed)
        self.spin_label_id.valueChanged.connect(self._on_label_id_changed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refresh_image_list(self, *_args):
        """Populate the image combo box with current Image layers."""
        current_text = self.combo_image.currentText()
        self.combo_image.blockSignals(True)
        self.combo_image.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.combo_image.addItem(layer.name)
        # Restore previous selection if still present
        idx = self.combo_image.findText(current_text)
        if idx >= 0:
            self.combo_image.setCurrentIndex(idx)
        self.combo_image.blockSignals(False)

    def _get_selected_image(self) -> Optional[napari.layers.Image]:
        """Return the currently selected Image layer, or None."""
        name = self.combo_image.currentText()
        if not name:
            return None
        try:
            layer = self.viewer.layers[name]
            if isinstance(layer, napari.layers.Image):
                return layer
        except KeyError:
            pass
        return None

    def _ensure_labels_layer(self) -> bool:
        """Return True if a labels layer is ready, False otherwise."""
        if self._labels_layer is not None and self._labels_layer in self.viewer.layers:
            return True
        self._set_status("⚠ No label layer. Click 'New Label Layer' first.")
        return False

    def _ensure_shapes_layer(self):
        """
        Create or retrieve the shapes helper layer used for
        rectangle / polygon / ellipse drawing.
        """
        name = "_manual_seg_shapes"
        if self._shapes_layer is not None and self._shapes_layer in self.viewer.layers:
            return self._shapes_layer
        # Check if it already exists
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == name:
                self._shapes_layer = layer
                return self._shapes_layer
        # Create new shapes layer
        self._shapes_layer = self.viewer.add_shapes(
            name=name,
            edge_color="yellow",
            edge_width=2,
            face_color="transparent",
            opacity=0.6,
        )
        return self._shapes_layer

    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)
        show_info(msg)

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------
    def _register_keybindings(self):
        """Register keyboard shortcuts on the napari viewer."""

        @self.viewer.bind_key("Up", overwrite=True)
        def _pan_up(viewer):
            self._pan_camera(dy=-self.spin_pan_step.value())

        @self.viewer.bind_key("Down", overwrite=True)
        def _pan_down(viewer):
            self._pan_camera(dy=self.spin_pan_step.value())

        @self.viewer.bind_key("Left", overwrite=True)
        def _pan_left(viewer):
            self._pan_camera(dx=-self.spin_pan_step.value())

        @self.viewer.bind_key("Right", overwrite=True)
        def _pan_right(viewer):
            self._pan_camera(dx=self.spin_pan_step.value())

        @self.viewer.bind_key("w", overwrite=True)
        def _pan_w(viewer):
            self._pan_camera(dy=-self.spin_pan_step.value())

        @self.viewer.bind_key("s", overwrite=True)
        def _pan_s(viewer):
            self._pan_camera(dy=self.spin_pan_step.value())

        @self.viewer.bind_key("a", overwrite=True)
        def _pan_a(viewer):
            self._pan_camera(dx=-self.spin_pan_step.value())

        @self.viewer.bind_key("d", overwrite=True)
        def _pan_d(viewer):
            self._pan_camera(dx=self.spin_pan_step.value())

        @self.viewer.bind_key("=", overwrite=True)
        def _zoom_in(viewer):
            self._zoom_camera(factor=1.2)

        @self.viewer.bind_key("-", overwrite=True)
        def _zoom_out(viewer):
            self._zoom_camera(factor=1 / 1.2)

        @self.viewer.bind_key("Space", overwrite=True)
        def _toggle_pan(viewer):
            self._toggle_pan_mode()

    def _pan_camera(self, dx: float = 0, dy: float = 0):
        """Shift the camera center by (dx, dy) pixels."""
        cam = self.viewer.camera
        cy, cx = cam.center[-2], cam.center[-1]
        # camera.center is in (z, y, x) or (y, x) format
        if len(cam.center) == 3:
            cam.center = (cam.center[0], cy + dy, cx + dx)
        else:
            cam.center = (cy + dy, cx + dx)

    def _zoom_camera(self, factor: float = 1.2):
        """Zoom the camera by the given factor (>1 = zoom in)."""
        self.viewer.camera.zoom *= factor

    def _toggle_pan_mode(self):
        """Toggle between pan mode and the previous drawing mode."""
        active = self.viewer.layers.selection.active
        if active is None:
            return

        if isinstance(active, napari.layers.Labels) and active.mode == "pan_zoom":
            # Return to previous mode
            if self._prev_mode and self._prev_mode != "pan_zoom":
                active.mode = self._prev_mode
                self._set_status(f"Returned to {self._prev_mode} mode.")
            else:
                active.mode = "paint"
                self._set_status("Returned to paint mode.")
        elif isinstance(active, napari.layers.Labels):
            # Save current mode and switch to pan
            self._prev_mode = active.mode
            active.mode = "pan_zoom"
            self._set_status("🖐 Pan mode – drag to move. Press Space again to resume.")
        else:
            # For non-labels layers, just set pan_zoom
            active.mode = "pan_zoom"
            self._set_status("🖐 Pan mode.")

    # ------------------------------------------------------------------
    # Slot: New Label Layer
    # ------------------------------------------------------------------
    def _on_new_label_layer(self):
        img_layer = self._get_selected_image()
        if img_layer is None:
            self._set_status("⚠ Please select an image layer first.")
            return

        img = np.asarray(img_layer.data)
        # For 2D images, shape is (H, W) or (H, W, C)
        if img.ndim == 2:
            mask_shape = img.shape
        elif img.ndim == 3:
            # Multichannel 2D: use only spatial dims
            mask_shape = img.shape[:2]
        else:
            self._set_status(f"⚠ Unsupported image dimensions: {img.ndim}D")
            return

        mask_name = f"{img_layer.name}_manual_mask"

        # Remove existing layer with the same name to start fresh
        for layer in list(self.viewer.layers):
            if layer.name == mask_name:
                self.viewer.layers.remove(layer)

        empty_mask = np.zeros(mask_shape, dtype=np.int32)
        self._labels_layer = self.viewer.add_labels(
            empty_mask, name=mask_name, opacity=0.5
        )

        # Reset label ID
        self._current_label_id = 1
        self.spin_label_id.setValue(1)
        self._labels_layer.selected_label = 1

        # Activate paint mode with current brush size
        self._labels_layer.mode = "paint"
        self._labels_layer.brush_size = self.spin_brush.value()
        self.viewer.layers.selection.active = self._labels_layer

        self._set_status(
            f"✓ Created label layer '{mask_name}' ({mask_shape[1]}×{mask_shape[0]}). "
            f"Start painting cells!"
        )

    # ------------------------------------------------------------------
    # Slot: Drawing tools
    # ------------------------------------------------------------------
    def _on_paint(self):
        if not self._ensure_labels_layer():
            return
        self.viewer.layers.selection.active = self._labels_layer
        self._labels_layer.mode = "paint"
        self._labels_layer.brush_size = self.spin_brush.value()
        self._labels_layer.selected_label = self.spin_label_id.value()
        self._prev_mode = "paint"
        self._set_status("🖌 Paint mode – draw freehand on the labels layer. (Arrow keys to pan)")

    def _on_erase(self):
        if not self._ensure_labels_layer():
            return
        self.viewer.layers.selection.active = self._labels_layer
        self._labels_layer.mode = "erase"
        self._labels_layer.brush_size = self.spin_brush.value()
        self._prev_mode = "erase"
        self._set_status("🧹 Erase mode – erase labels by painting over them.")

    def _on_fill(self):
        if not self._ensure_labels_layer():
            return
        self.viewer.layers.selection.active = self._labels_layer
        self._labels_layer.mode = "fill"
        self._labels_layer.selected_label = self.spin_label_id.value()
        self._prev_mode = "fill"
        self._set_status("🪣 Fill mode – click a closed region to flood-fill it.")

    def _on_pan(self):
        if not self._ensure_labels_layer():
            return
        # Save current mode before switching
        if self._labels_layer.mode != "pan_zoom":
            self._prev_mode = self._labels_layer.mode
        self.viewer.layers.selection.active = self._labels_layer
        self._labels_layer.mode = "pan_zoom"
        self._set_status("🖐 Pan mode – drag to move the view. Press Space to return.")

    def _on_rect(self):
        if not self._ensure_labels_layer():
            return
        shapes = self._ensure_shapes_layer()
        self.viewer.layers.selection.active = shapes
        shapes.mode = "add_rectangle"
        self._set_status(
            "▭ Rectangle mode – draw rectangles on the shapes layer. "
            "Click 'Convert Shapes → Labels' when done."
        )

    def _on_polygon(self):
        if not self._ensure_labels_layer():
            return
        shapes = self._ensure_shapes_layer()
        self.viewer.layers.selection.active = shapes
        shapes.mode = "add_polygon"
        self._set_status(
            "⬠ Polygon mode – draw polygons. "
            "Double-click to close. Convert when done."
        )

    def _on_ellipse(self):
        if not self._ensure_labels_layer():
            return
        shapes = self._ensure_shapes_layer()
        self.viewer.layers.selection.active = shapes
        shapes.mode = "add_ellipse"
        self._set_status(
            "⬭ Ellipse mode – draw ellipses. "
            "Convert when done."
        )

    # ------------------------------------------------------------------
    # Slot: Brush size / Label ID
    # ------------------------------------------------------------------
    def _on_brush_size_changed(self, value: int):
        if self._labels_layer is not None and self._labels_layer in self.viewer.layers:
            self._labels_layer.brush_size = value

    def _on_label_id_changed(self, value: int):
        self._current_label_id = value
        if self._labels_layer is not None and self._labels_layer in self.viewer.layers:
            self._labels_layer.selected_label = value

    def _on_next_label(self):
        new_id = self.spin_label_id.value() + 1
        self.spin_label_id.setValue(new_id)
        self._set_status(f"Label ID set to {new_id}. Paint a new cell.")

    # ------------------------------------------------------------------
    # Slot: Convert shapes → labels
    # ------------------------------------------------------------------
    def _on_shapes_to_labels(self):
        if not self._ensure_labels_layer():
            return

        if (
            self._shapes_layer is None
            or self._shapes_layer not in self.viewer.layers
            or len(self._shapes_layer.data) == 0
        ):
            self._set_status("⚠ No shapes to convert. Draw some shapes first.")
            return

        from skimage.draw import polygon as draw_polygon

        mask = np.asarray(self._labels_layer.data)
        label_id = self.spin_label_id.value()
        n_shapes = len(self._shapes_layer.data)

        for shape_data in self._shapes_layer.data:
            # shape_data is an (N, D) array of vertices; D = 2 for 2D
            vertices = np.asarray(shape_data)
            if vertices.ndim != 2 or vertices.shape[1] < 2:
                continue

            # Use the last two columns as (row, col) for 2D
            rr, cc = draw_polygon(
                vertices[:, -2], vertices[:, -1], shape=mask.shape
            )
            mask[rr, cc] = label_id
            label_id += 1

        # Update the labels layer data
        self._labels_layer.data = mask

        # Update the label ID spinner
        self.spin_label_id.setValue(label_id)

        # Clear shapes layer
        self._shapes_layer.data = []

        self._set_status(
            f"✓ Converted {n_shapes} shape(s) to labels. "
            f"Next label ID: {label_id}."
        )

    # ------------------------------------------------------------------
    # Slot: Finalize Mask
    # ------------------------------------------------------------------
    def _on_finalize(self):
        if not self._ensure_labels_layer():
            return

        from skimage.measure import label as sk_label

        mask = np.asarray(self._labels_layer.data).copy()

        # Relabel connected components sequentially
        relabeled = sk_label(mask > 0, connectivity=2).astype(np.int32)

        self._labels_layer.data = relabeled
        n_cells = relabeled.max()

        # Reset label ID to next available
        self.spin_label_id.setValue(n_cells + 1)

        self._set_status(
            f"✓ Mask finalized: {n_cells} cell(s) labeled sequentially (1–{n_cells})."
        )

    # ------------------------------------------------------------------
    # Slot: Save Mask
    # ------------------------------------------------------------------
    def _on_save(self):
        if not self._ensure_labels_layer():
            return

        mask = np.asarray(self._labels_layer.data)
        if mask.max() == 0:
            self._set_status("⚠ The mask is empty. Nothing to save.")
            return

        # Default filename based on layer name
        default_name = self._labels_layer.name + ".tif"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation Mask",
            default_name,
            "TIFF Files (*.tif *.tiff);;All Files (*)",
        )
        if not file_path:
            return  # User cancelled

        from tifffile import imwrite

        save_path = Path(file_path)
        imwrite(str(save_path), mask.astype(np.uint16))
        n_cells = len(np.unique(mask)) - 1  # exclude background
        self._set_status(
            f"✓ Saved mask to {save_path.name} ({n_cells} cells)."
        )

"""Check installed-package discovery and every declared widget without model downloads."""

from importlib import metadata

import napari
import numpy as np
import pytest
from npe2 import PluginManifest
from tifffile import imread


def test_installed_manifest():
    manifest = PluginManifest.from_distribution("napari-cell-registration")
    manifest.validate_imports()
    assert manifest.display_name == "TopoAlign"
    assert len(manifest.contributions.widgets) == 5
    import napari_cell_registration

    assert napari_cell_registration.__version__ == metadata.version("napari-cell-registration")


@pytest.mark.parametrize("widget_name", [
    "Cell Segmentation (Cellpose)",
    "Save Mask Layers",
    "Manual Segmentation",
    "Registration Workflow",
    "WSI Segmentation",
])
def test_widget_opens(qtbot, widget_name):
    viewer = napari.Viewer(show=False)
    try:
        dock, widget = viewer.window.add_plugin_dock_widget("napari-cell-registration", widget_name)
        assert widget is not None
        assert dock.widget() is not None
    finally:
        viewer.close()


def test_save_mask_layer_from_widget(qtbot, tmp_path):
    from napari.components import ViewerModel
    from napari_cell_registration._widget import save_mask_layers_factory

    # Export only needs the layer model, not a GPU/OpenGL canvas.
    viewer = ViewerModel()
    widget = save_mask_layers_factory()
    qtbot.addWidget(widget.native)
    try:
        mask = np.zeros((24, 24), dtype=np.uint16)
        mask[2:6, 2:6] = 1
        mask[12:18, 12:18] = 2
        viewer.add_labels(mask, name="nuclei")
        widget.viewer.bind(viewer)
        widget.reset_choices()
        assert "nuclei" in widget.mask_layer.choices
        widget.mask_layer.value = "nuclei"
        widget.output_dir.value = tmp_path
        widget()
        files = list(tmp_path.glob("*.tif"))
        assert len(files) == 1
        np.testing.assert_array_equal(imread(files[0]), mask)
    finally:
        widget.close()

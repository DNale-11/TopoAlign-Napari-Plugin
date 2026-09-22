"""
napari-cell-registration

DAPI-based cell segmentation, matching, and registration plugin for napari.
"""

__version__ = "0.1.0"

try:
    from ._qt_init import configure_qt

    configure_qt()
except Exception:
    pass

try:
    from ._widget import save_mask_layers_widget, segment_cells_widget, registration_workflow_widget
    from ._wsi_widget import wsi_segmentation_widget
    from ._manual_seg_widget import ManualSegmentationWidget
except Exception:
    # Allow importing core utilities in headless environments without napari.
    segment_cells_widget = None
    save_mask_layers_widget = None
    registration_workflow_widget = None
    wsi_segmentation_widget = None
    ManualSegmentationWidget = None

__all__ = [
    "segment_cells_widget",
    "save_mask_layers_widget",
    "registration_workflow_widget",
    "wsi_segmentation_widget",
    "ManualSegmentationWidget",
]

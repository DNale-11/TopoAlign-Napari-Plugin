# Changelog

## 0.1.0

- First standalone TopoAlign napari plugin repository.
- Includes Cellpose segmentation, manual masks, mask export, normal/FISH/WSI
  registration, and the optional CellViT++ runner.
- Retains the `napari-cell-registration` distribution and plugin identifiers;
  displays `TopoAlign` in the napari Plugins menu.
- Adds installation guides, BSD-3-Clause license, wheel/source builds,
  synthetic-data smoke tests, and GitHub CI/release workflows.
- Fixes the Save Mask Layers dropdown reading the magicgui parameter widget
  instead of its bound napari viewer.
- Uses Python 3.10–3.11, napari 0.6.x, and NumPy 1.x. OpenCV is capped below
  4.12 to avoid requiring NumPy 2.

Source: `DNale-11/cell_registration`, base commit `786aeb0`, plus the local
plugin changes present at extraction on 2026-09-22 (including FISH registration).
This is a working-tree snapshot, not an export of that commit alone. Registration
and segmentation algorithm implementations were copied without changes; the
mask-export UI binding received the fix described above.

# TopoAlign usage guide

DAPI-based cell segmentation, matching, and registration plugin for napari.

## Features

- **Cell Segmentation**: Automatic nuclear segmentation using Cellpose-SAM
- **Feature Extraction**: Extract morphological features from segmented cells
- **Cell Matching**: Match cells between two imaging rounds using feature similarity
- **Rigid Registration**: Estimate and apply rigid transforms between rounds
- **Interactive Visualization**: View and validate results directly in napari

## Installation

```bash
python -m pip install ".[gui]"
```

For development:
```bash
python -m pip install -e ".[gui,dev]"
```

## Usage

1. Open napari:
   ```bash
   napari
   ```

2. Load your DAPI images as layers

3. From the Plugins → TopoAlign menu, select:
   - **Cell Segmentation** - for quick segmentation only
   - **Registration Workflow** - for complete registration pipeline
   - **WSI Segmentation** - for OpenSlide + CellViT++ whole-slide input

### Large Image Segmentation

The **Cell Segmentation** widget has a `mode` option:

- `auto - chunk only if large`: use chunked segmentation only when the image is larger than `large_image_threshold_mp`.
- `full image - ignore chunk settings`: keep the previous full-image Cellpose path.
- `chunked - use label stitching`: always segment with overlapping chunks.

Use `chunk_size` to control each inference tile and `chunk_overlap` to give Cellpose context at tile borders. `stitch_labels` merges labels that touch in overlap regions; keep it enabled unless you need to inspect raw tile boundaries. Registration is unchanged.

### WSI Segmentation and Registration

The **WSI Segmentation** widget reads fixed/moving whole-slide images with
OpenSlide, then segments each WSI with the selected backend: CellViT++ or
Cellpose-SAM. Inspect the image and mask layers in napari before registration.

The unified **Registration Workflow** registers inspected WSI mask layers. HE is
the fixed/reference space, mIF-DAPI is the moving channel used to estimate the
transform, and multichannel mIF intensity data is not loaded during transform
estimation.

Required inputs:

- fixed and moving WSI files selected from the file picker
- fixed/moving segmentation model choice, default fixed `CellViT++` and moving `Cellpose-SAM`
- CellViT++ model choice, default `SAM`
- `use_gpu` enabled when CellViT++ is selected
- `use_cell_shapes` enabled to render CellViT++ contours as instance masks
- mIF moving WSI channels are detected from QPTIFF metadata; `DAPI` is selected by default.
- selected moving mIF channel is used by Cellpose-SAM; CellViT++ reads the original WSI path.
- `read_level` controls napari preview only; `cellpose_read_level` controls the DAPI image used for Cellpose-SAM and defaults to `2` for whole-slide runs.
- Large Cellpose WSI inputs are streamed tile-by-tile instead of loading the full DAPI image into memory.
- `max_cellpose_tiles` stops accidental very long full-resolution runs; set it to `0` only when you intentionally want a full run.
- WSI mask/centroid coordinates require metadata: source WSI path, mask path, coordinate space, origin, downsample, image size, MPP fields, channel, and segmentation method.
- Final transforms are saved as `mIF level-0 pixel -> HE level-0 pixel`; local deformation grids are defined in HE level-0 coordinates.
- `add_moving_channel_stack` is for visual inspection only. The 8-channel mIF stack is warped tile-wise only after the DAPI-derived transform has been exported.

## Requirements

- Python 3.10 or 3.11
- napari >= 0.6.6, < 0.7
- cellpose >= 4.0
- See [pyproject.toml](../pyproject.toml) for the complete dependency list and
  the [installation guide](../README.md) for optional WSI dependencies.

## License

BSD-3-Clause

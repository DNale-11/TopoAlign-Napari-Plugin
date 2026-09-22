# TopoAlign napari plugin

[中文说明](README.zh-CN.md) · [Usage guide](docs/usage.md) · [Releases](https://github.com/DNale-11/TopoAlign-Napari-Plugin/releases)

TopoAlign provides cell segmentation, morphology and topology based matching,
and microscopy image registration inside napari. This is the standalone plugin
extracted from [TopoAlign](https://github.com/DNale-11/cell_registration).

The Python distribution remains **`napari-cell-registration`**, the import
package remains **`napari_cell_registration`**, and the napari menu name is
**TopoAlign**.

## Features

- Cellpose-SAM nuclear segmentation, including overlapping tiles for large images.
- Manual segmentation and export of existing label layers as TIFF masks.
- Cell matching using morphology, spatial relationships, and local topology.
- Normal, FISH, and WSI registration modes, with transform and result export.
- Optional OpenSlide WSI input and CellViT++ segmentation.

## Installation

Use **Python 3.10 or 3.11** in a dedicated environment. This release targets
napari 0.6.x and NumPy 1.x to match the existing scientific dependency stack.

```bash
conda create -n topoalign-napari python=3.11 -y
conda activate topoalign-napari
git clone https://github.com/DNale-11/TopoAlign-Napari-Plugin.git
cd TopoAlign-Napari-Plugin
python -m pip install ".[gui]"
napari
```

The `gui` extra installs the PyQt5 backend. If napari and a Qt backend are already
installed, use `python -m pip install .`. Private repositories require GitHub
access when cloning or downloading release assets.

Alternatively, download the wheel from a GitHub release, then run:

```bash
python -m pip install "napari_cell_registration-0.1.0-py3-none-any.whl[gui]"
```

For an NVIDIA GPU, install the PyTorch build matching your system using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/) before
installing this plugin. Verify it with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

On Windows this release requires PyTorch 2.7.x and torchvision 0.22.x: newer
Windows wheels failed to initialize `c10.dll` when loaded with Qt in our CI.
Use the [PyTorch 2.7.1 installation commands](https://pytorch.org/get-started/previous-versions/#v271)
to select a CPU or compatible CUDA build. CI uses `torch==2.7.1` and
`torchvision==0.22.1` CPU wheels.

Cellpose downloads model weights on first use. Existing masks can be used for
registration without running segmentation. This repository does not include
microscopy datasets or model weights.

## Quick start

1. Open napari and load the fixed/reference image and moving image.
2. Open **Plugins → TopoAlign → Cell Segmentation (Cellpose)** and generate masks,
   or load existing integer instance masks as Labels layers.
3. Open **Registration Workflow**, select fixed and moving masks and images,
   and choose `auto`, `normal`, `fish - contour + topology`, or `wsi` as appropriate.
4. Inspect matched cells and registered image layers. Enable `save_results` and
   select `output_dir` to export results.

Each instance mask uses `0` for background and a positive integer per cell.
Use **Manual Segmentation** to create/edit masks and **Save Mask Layers** to
export them. See the [usage guide](docs/usage.md) for tiling and WSI coordinates.

## Optional WSI support

```bash
python -m pip install ".[gui,wsi]"
```

WSI reading additionally needs the native OpenSlide library; follow the
[OpenSlide Python installation guide](https://openslide.org/api/python/).
The `wsi` extra only installs the Python binding. CellViT++ and its model weights
are separate dependencies and are **not** installed by this extra. To use that
backend, install a compatible CellViT++/pathopatch environment with a GPU; the
plugin invokes `cellvit.detect_cells` through its bundled compatibility runner.
Its patches depend on upstream internals, so check backend compatibility before
processing large slides. The ordinary image/mask workflow does not require it.

## Development and checks

```bash
python -m pip install ".[gui,dev]"
python -m pytest -q
python -m npe2 validate src/napari_cell_registration/napari.yaml --imports
python -m build
python -m twine check dist/*
```

GUI smoke tests run with an offscreen Qt platform and synthetic data; they do
not download models. CI checks Python 3.10 on Windows and Python 3.11 on Linux.
GPU segmentation and external WSI inference are not covered by those checks.

Tagged releases (`v*`) build and attach a wheel and source distribution to a
GitHub release after CI succeeds. This workflow does not publish to PyPI or the
napari plugin index. See [CHANGELOG.md](CHANGELOG.md) for release provenance.

## License

[BSD-3-Clause](LICENSE). External dependencies and model weights retain their
own licenses.

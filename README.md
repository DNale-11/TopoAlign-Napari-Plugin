# TopoAlign napari plugin

[中文](README.zh-CN.md) · [Usage guide](docs/usage.md) · [Releases](https://github.com/DNale-11/TopoAlign-Napari-Plugin/releases)

Cellpose-SAM segmentation, manual mask editing, and topology-guided cell matching and registration for microscopy images, including FISH and WSI workflows.

## Install

Requires Python 3.10–3.11; dependencies use napari 0.6.x and NumPy 1.x.

```bash
conda create -n topoalign-napari python=3.11 -y
conda activate topoalign-napari
git clone https://github.com/DNale-11/TopoAlign-Napari-Plugin.git
cd TopoAlign-Napari-Plugin
python -m pip install ".[gui]"
napari
```

For GPU use, install a compatible [PyTorch build](https://pytorch.org/get-started/previous-versions/#v271) first. Windows requires PyTorch 2.7.x / torchvision 0.22.x. Cellpose downloads model weights on first use.

## Use

1. Load fixed and moving images in napari.
2. Open **Plugins → TopoAlign** to segment cells, or load existing instance masks as Labels layers.
3. Open **Registration Workflow**, select images and masks, then run registration.
4. Enable `save_results` and set `output_dir` to export results.

WSI support requires `python -m pip install ".[gui,wsi]"` and native OpenSlide. CellViT++ requires a separately configured GPU environment and model weights. See the [usage guide](docs/usage.md).

[BSD-3-Clause](LICENSE) · Python package: `napari-cell-registration`

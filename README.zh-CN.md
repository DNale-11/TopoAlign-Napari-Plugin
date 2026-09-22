# TopoAlign napari 插件

[English](README.md) · [详细使用说明](docs/usage.md) · [下载安装包](https://github.com/DNale-11/TopoAlign-Napari-Plugin/releases)

这是从 [TopoAlign 主项目](https://github.com/DNale-11/cell_registration) 整理出来的独立 napari 插件，支持 Cellpose-SAM 细胞核分割、手动分割、掩膜保存、形态与拓扑匹配，以及普通图像、FISH 和 WSI 配准。

安装包名保留为 `napari-cell-registration`，Python 导入名为 `napari_cell_registration`，napari 菜单名称为 **TopoAlign**。

## 安装

本版本使用 **Python 3.10 / 3.11、napari 0.6.x 和 NumPy 1.x**。建议新建环境：

```bash
conda create -n topoalign-napari python=3.11 -y
conda activate topoalign-napari
git clone https://github.com/DNale-11/TopoAlign-Napari-Plugin.git
cd TopoAlign-Napari-Plugin
python -m pip install ".[gui]"
napari
```

`gui` 会安装 PyQt5；已有 napari 和 Qt 环境时可执行 `python -m pip install .`。私有仓库需要对应 GitHub 访问权限。也可以下载 Release 中的 wheel，在下载目录执行：

```bash
python -m pip install "napari_cell_registration-0.1.0-py3-none-any.whl[gui]"
```

如需 NVIDIA GPU，请先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装适合本机的 PyTorch，再安装插件。执行 `python -c "import torch; print(torch.cuda.is_available())"` 检查 GPU。Cellpose 首次运行会下载模型；已有掩膜可直接配准。

## 基本操作

1. 在 napari 中载入固定图像和移动图像。
2. 从 **Plugins → TopoAlign → Cell Segmentation (Cellpose)** 生成掩膜，或载入已有 Labels 图层。
3. 打开 **Registration Workflow**，选择对应图像与掩膜，并选择 `auto`、`normal`、`fish - contour + topology` 或 `wsi` 模式。
4. 检查匹配点和配准图层，勾选 `save_results` 并设置 `output_dir` 保存结果。

实例掩膜中 `0` 代表背景，每个细胞使用独立的正整数标签。可使用 **Manual Segmentation** 制作或编辑掩膜，使用 **Save Mask Layers** 导出 TIFF。大图分块设置与 WSI 坐标要求见[详细使用说明](docs/usage.md)。

## WSI 可选功能

执行 `python -m pip install ".[gui,wsi]"` 安装 OpenSlide Python 绑定，并按 [OpenSlide 文档](https://openslide.org/api/python/) 安装原生库。

CellViT++、pathopatch、GPU 环境和模型权重需要另行准备，`wsi` 扩展不会自动安装它们。插件内置兼容处理依赖这些项目的内部接口，应先验证版本兼容性。普通图像和已有掩膜配准不需要 CellViT++。

## 开发与发布

```bash
python -m pip install ".[gui,dev]"
python -m pytest -q
python -m npe2 validate src/napari_cell_registration/napari.yaml --imports
python -m build
python -m twine check dist/*
```

自动检查使用合成数据验证插件发现、窗口创建和核心配准，不下载模型；未覆盖 GPU 模型推理或真实 WSI 全流程。推送 `v*` 标签后，工作流在检查通过后向 GitHub Release 上传 wheel 与源码包，不会自动发布到 PyPI 或 napari 插件索引。

许可证：[BSD-3-Clause](LICENSE)。本仓库不包含实验图像、模型权重或分析输出。

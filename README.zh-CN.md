# TopoAlign napari 插件

[English](README.md) · [使用说明](docs/usage.md) · [下载安装包](https://github.com/DNale-11/TopoAlign-Napari-Plugin/releases)

支持 Cellpose-SAM 细胞分割、手动掩膜编辑、拓扑引导的细胞匹配与图像配准，适用于普通显微图像、FISH 和 WSI。

## 安装

使用 Python 3.10–3.11，依赖 napari 0.6.x 和 NumPy 1.x。

```bash
conda create -n topoalign-napari python=3.11 -y
conda activate topoalign-napari
git clone https://github.com/DNale-11/TopoAlign-Napari-Plugin.git
cd TopoAlign-Napari-Plugin
python -m pip install ".[gui]"
napari
```

使用 GPU 时，请先安装匹配的 [PyTorch](https://pytorch.org/get-started/previous-versions/#v271)。Windows 限定 PyTorch 2.7.x / torchvision 0.22.x。Cellpose 首次运行会下载模型。

## 使用

1. 在 napari 中载入固定图像和移动图像。
2. 打开 **Plugins → TopoAlign** 分割细胞，或将已有实例掩膜载入为 Labels 图层。
3. 打开 **Registration Workflow**，选择图像和掩膜后运行配准。
4. 勾选 `save_results`，设置 `output_dir` 保存结果。

WSI 功能需执行 `python -m pip install ".[gui,wsi]"` 并安装 OpenSlide 原生库；CellViT++ 需另行配置 GPU 环境和模型权重。详见[使用说明](docs/usage.md)。

[BSD-3-Clause](LICENSE) · 安装包名：`napari-cell-registration`

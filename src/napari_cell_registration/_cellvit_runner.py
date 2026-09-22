"""CellViT++ CLI runner with local compatibility fixes."""

from __future__ import annotations

import inspect
import textwrap


def _patch_pathopatch_pydantic2() -> None:
    try:
        import pydantic
        from pathopatch.patch_extraction import dataset
    except ImportError:
        return

    major = int(pydantic.__version__.split(".", maxsplit=1)[0])
    if major < 2:
        return

    config_cls = dataset.LivePatchWSIConfig
    if getattr(config_cls, "_napari_cell_registration_patched", False):
        return

    original_init = config_cls.__init__
    defaults = {
        "target_mag": None,
        "level": None,
        "annotation_path": None,
        "label_map_file": None,
        "label_map": None,
        "normalization_vector_json": None,
        "tissue_annotation": None,
        "tissue_annotation_intersection_ratio": None,
        "otsu_annotation": None,
    }

    def __init__(self, **data):
        for key, value in defaults.items():
            data.setdefault(key, value)
        original_init(self, **data)

    config_cls.__init__ = __init__
    config_cls._napari_cell_registration_patched = True


def _patch_pathopatch_full_wsi_sampling() -> None:
    from pathopatch.patch_extraction import dataset

    dataset_cls = dataset.LivePatchWSIDataset
    if getattr(dataset_cls, "_napari_cell_registration_patched", False):
        return

    source = textwrap.dedent(inspect.getsource(dataset_cls._prepare_slide))
    source, tissue_replacements = source.replace(
        "tissue_region: List[Polygon] = []",
        "tissue_region = None",
        1,
    ), source.count("tissue_region: List[Polygon] = []")
    marker = "    # get the interesting coordinates: no background, filtered by annotation etc.\n"
    source, mask_replacements = source.replace(
        marker,
        "    mask_images = {}\n" + marker,
        1,
    ), source.count(marker)
    if tissue_replacements < 1 or mask_replacements < 1:
        raise RuntimeError("Failed to patch pathopatch WSI sampling for mIF input.")
    namespace = {}
    exec(source, dataset.__dict__, namespace)
    dataset_cls._prepare_slide = namespace["_prepare_slide"]
    dataset_cls._napari_cell_registration_patched = True


def _patch_cellvit_process_wsi_defaults() -> None:
    from cellvit.inference.inference import CellViTInference

    if getattr(CellViTInference, "_napari_cell_registration_patched", False):
        return

    original_process_wsi = CellViTInference.process_wsi

    def process_wsi(self, *args, **kwargs):
        kwargs.setdefault("apply_prefilter", False)
        kwargs.setdefault("min_intersection_ratio", 0.0)
        return original_process_wsi(self, *args, **kwargs)

    CellViTInference.process_wsi = process_wsi
    CellViTInference._napari_cell_registration_patched = True


def main() -> None:
    _patch_pathopatch_pydantic2()
    _patch_pathopatch_full_wsi_sampling()
    _patch_cellvit_process_wsi_defaults()
    from cellvit.detect_cells import main as cellvit_main

    cellvit_main()


if __name__ == "__main__":
    main()

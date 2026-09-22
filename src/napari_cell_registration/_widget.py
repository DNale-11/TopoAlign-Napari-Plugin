"""
napari widgets for cell registration workflows.
"""

from typing import Annotated
from pathlib import Path
from enum import Enum
import re

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.types import ImageData
from napari.layers import Labels, Layer
from napari.utils.notifications import show_info

from .core import (
    CellFeaturesConfig,
    CellposeConfig,
    CellposeSegmenter,
    MatchingConfig,
    MIN_MATCHES_FOR_REFINEMENT,
    apply_rigid_to_points,
    assign_patches,
    cast_warped_like_original,
    compute_cell_features,
    compute_match_residuals,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    greedy_match_cells,
    rigid_transform_to_affine,
    two_stage_match_cells,
)
from .core.matching import apply_transform_to_features, _filter_candidate_matches_by_hard_constraints
from .core.point_registration import (
    fit_tps_from_matches,
    warp_image_with_transform,
    warp_image_with_tps,
    compute_valid_overlap_mask,
)
from ._qt_init import apply_default_font
from .wsi_registration import (
    KnnLocalAffineWarp,
    WSICentroidMetadata,
    estimate_wsi_translation_from_matches,
    estimate_local_translation_grid,
    filter_wsi_landmark_residuals,
    filter_parallel_displacements,
    find_wsi_landmark_matches,
    knn_local_affine_leave_one_out_residuals,
    run_wsi_centroid_registration,
    run_wsi_feature_registration,
    select_main_displacement_cluster,
    shift_array_xy,
    warp_array_with_knn_local_affine,
    warp_array_with_translation_grid,
)
from .core.fish_registration import register_large_fish_masks


class CellposeModel(Enum):
    """Cellpose model options."""

    CPSAM = "cpsam"
    CYTO = "cyto"
    NUCLEI = "nuclei"
    CYTO2 = "cyto2"
    CYTO3 = "cyto3"


class SegmentationMode(Enum):
    """Segmentation execution modes."""

    AUTO = "auto - chunk only if large"
    FULL_IMAGE = "full image - ignore chunk settings"
    CHUNKED = "chunked - use label stitching"


class RegistrationMode(Enum):
    """Registration workflow modes."""

    AUTO = "auto"
    NORMAL = "normal"
    FISH = "fish - contour + topology"
    WSI = "wsi"


class WSIRefineModel(Enum):
    """WSI registration refinement models."""

    KNN_LOCAL_AFFINE = "knn local affine"
    LOCAL_TRANSLATION_GRID = "4x4 local translation grid"


class ImageChannelAxis(Enum):
    """Channel-axis layout for 2D multichannel images."""

    AUTO = "auto"
    FIRST = "first (C,Y,X)"
    LAST = "last (Y,X,C)"
    NONE = "none (Y,X)"


class RegisteredDisplayMode(Enum):
    """How registered multichannel images are displayed in napari."""

    COLORED_STACK = "colored channel stack"
    COMPOSITE = "composite RGB preview"
    BOTH = "composite + colored stack"


def _channel_axis_key(channel_axis: ImageChannelAxis | str) -> str:
    value = channel_axis.value if isinstance(channel_axis, ImageChannelAxis) else str(channel_axis)
    return value.split()[0].strip().lower()


def _resolve_channel_index(channel_count: int, channel: int) -> int:
    idx = int(channel)
    if idx < 0 or idx >= int(channel_count):
        raise ValueError(f"DAPI channel {channel} is out of range for {channel_count} channels.")
    return idx


def _infer_channel_count(
    image: ImageData | np.ndarray,
    channel_axis: ImageChannelAxis | str,
) -> int | None:
    data = np.asarray(image)
    axis = _channel_axis_key(channel_axis)

    if data.ndim == 2 or axis == "none":
        return 1
    if data.ndim != 3:
        return None
    if axis == "first":
        return int(data.shape[0])
    if axis == "last":
        return int(data.shape[-1])
    if data.shape[0] <= 10 and data.shape[1] > 10 and data.shape[2] > 10:
        return int(data.shape[0])
    if data.shape[-1] <= 10:
        return int(data.shape[-1])
    return None


def _dapi_channel_choices(widget) -> list[tuple[str, int]]:
    parent = getattr(widget, "parent", None)
    if parent is None:
        return [("0", 0)]

    images_value = getattr(parent.images, "value", None)
    if images_value is None:
        images = []
    elif isinstance(images_value, (list, tuple)):
        images = list(images_value)
    else:
        images = [images_value]

    axis_value = getattr(parent.channel_axis, "value", ImageChannelAxis.AUTO)
    counts = [
        count
        for image in images
        if (count := _infer_channel_count(image, axis_value)) is not None
    ]
    channel_count = min(counts) if counts else 1
    return [(str(idx), idx) for idx in range(max(1, int(channel_count)))]


def _segment_cells_widget_init(widget) -> None:
    def reset_dapi_channel_choices(*_) -> None:
        current = widget.dapi_channel.value
        widget.dapi_channel.reset_choices()
        choices = tuple(widget.dapi_channel.choices)
        if not choices:
            return
        widget.dapi_channel.value = current if current in choices else choices[0]

    widget.images.changed.connect(reset_dapi_channel_choices)
    widget.channel_axis.changed.connect(reset_dapi_channel_choices)
    reset_dapi_channel_choices()


def _extract_dapi_channel(
    image: ImageData | np.ndarray,
    channel_axis: ImageChannelAxis | str,
    dapi_channel: int,
) -> np.ndarray:
    """Return the 2D DAPI image used for segmentation and landmark registration."""
    data = np.asarray(image)
    axis = _channel_axis_key(channel_axis)

    if axis == "none":
        if data.ndim != 2:
            raise ValueError(f"Channel axis 'none' expects a 2D image, got shape {data.shape}.")
        return data

    if axis == "first":
        if data.ndim != 3:
            raise ValueError(f"Channel axis 'first' expects C,Y,X data, got shape {data.shape}.")
        ch = _resolve_channel_index(data.shape[0], dapi_channel)
        return data[ch, :, :]

    if axis == "last":
        if data.ndim != 3:
            raise ValueError(f"Channel axis 'last' expects Y,X,C data, got shape {data.shape}.")
        ch = _resolve_channel_index(data.shape[-1], dapi_channel)
        return data[..., ch]

    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[0] <= 10 and data.shape[1] > 10 and data.shape[2] > 10:
        ch = _resolve_channel_index(data.shape[0], dapi_channel)
        return data[ch, :, :]
    if data.ndim == 3 and data.shape[-1] <= 10:
        ch = _resolve_channel_index(data.shape[-1], dapi_channel)
        return data[..., ch]
    raise ValueError(
        f"Cannot infer a 2D DAPI channel from image shape {data.shape}; "
        "set channel_axis to first, last, or none."
    )


def _napari_channel_axis(channel_axis: ImageChannelAxis | str, image: np.ndarray) -> int | None:
    axis = _channel_axis_key(channel_axis)
    if image.ndim != 3:
        return None
    if axis == "first":
        return 0
    if axis == "last":
        return -1
    if axis == "auto" and image.shape[0] <= 10 and image.shape[1] > 10 and image.shape[2] > 10:
        return 0
    if axis == "auto" and image.shape[-1] <= 10:
        return -1
    return None


def _add_registered_image_layer(
    viewer: napari.Viewer,
    image: np.ndarray,
    name: str,
    channel_axis: ImageChannelAxis | str,
    display_mode: RegisteredDisplayMode | str,
    opacity: float = 0.5,
) -> None:
    mode_value = display_mode.value if isinstance(display_mode, RegisteredDisplayMode) else str(display_mode)
    show_composite = mode_value.startswith("composite")
    show_colored_stack = mode_value.startswith("colored") or "colored stack" in mode_value
    display_image, display_scale = _downsample_registered_image_for_display(image, channel_axis)
    display_kwargs = {"scale": display_scale} if display_scale is not None else {}
    rgb_display_scale = _spatial_scale_for_rgb(display_scale)
    rgb_display_kwargs = {"scale": rgb_display_scale} if rgb_display_scale is not None else {}

    composite = _make_rgb_composite(display_image, channel_axis) if show_composite else None
    if composite is not None:
        viewer.add_image(
            composite,
            name=f"{name} Composite",
            opacity=opacity,
            blending="additive",
            rgb=True,
            **rgb_display_kwargs,
        )

    if show_colored_stack and _add_registered_channels_as_layers(
        viewer,
        display_image,
        name,
        channel_axis,
        opacity,
        rgb_display_kwargs,
    ):
        return

    if composite is not None:
        return

    image_kwargs = {"name": name, "opacity": opacity, "blending": "additive"}
    if image.ndim == 2:
        image_kwargs["colormap"] = "green"
    image_kwargs.update(display_kwargs)
    viewer.add_image(display_image, **image_kwargs)


def _downsample_registered_image_for_display(
    image: np.ndarray,
    channel_axis: ImageChannelAxis | str,
    max_axis: int = 30000,
) -> tuple[np.ndarray, tuple[float, ...] | None]:
    data = np.asarray(image)
    spatial_shape = _spatial_shape_for_display(data, channel_axis)
    if spatial_shape is None:
        return data, None
    factor = int(np.ceil(max(spatial_shape) / float(max_axis)))
    if factor <= 1:
        return data, None

    show_info(
        "  Registered image is too large for stable OpenGL display; "
        f"showing a {factor}x downsampled preview layer."
    )
    if data.ndim == 2:
        return data[::factor, ::factor], (float(factor), float(factor))

    napari_axis = _napari_channel_axis(channel_axis, data)
    if data.ndim == 3 and napari_axis == 0:
        return data[:, ::factor, ::factor], (1.0, float(factor), float(factor))
    if data.ndim == 3 and napari_axis == -1:
        return data[::factor, ::factor, :], (float(factor), float(factor), 1.0)
    return data, None


def _spatial_shape_for_display(
    image: np.ndarray,
    channel_axis: ImageChannelAxis | str,
) -> tuple[int, int] | None:
    data = np.asarray(image)
    if data.ndim == 2:
        return int(data.shape[0]), int(data.shape[1])
    napari_axis = _napari_channel_axis(channel_axis, data)
    if data.ndim == 3 and napari_axis == 0:
        return int(data.shape[1]), int(data.shape[2])
    if data.ndim == 3 and napari_axis == -1:
        return int(data.shape[0]), int(data.shape[1])
    return None


def _spatial_scale_for_rgb(scale: tuple[float, ...] | None) -> tuple[float, float] | None:
    if scale is None:
        return None
    if len(scale) == 2:
        return float(scale[0]), float(scale[1])
    if len(scale) == 3 and scale[0] == 1.0:
        return float(scale[1]), float(scale[2])
    if len(scale) == 3 and scale[2] == 1.0:
        return float(scale[0]), float(scale[1])
    return None


def _add_registered_channels_as_layers(
    viewer: napari.Viewer,
    image: np.ndarray,
    name: str,
    channel_axis: ImageChannelAxis | str,
    opacity: float,
    image_kwargs: dict,
) -> bool:
    channels = _channels_first_view(image, channel_axis)
    if channels is None:
        return False

    palette = _display_palette()
    for idx in range(channels.shape[0]):
        color = palette[idx % len(palette)]
        norm = _normalize_channel_for_composite(channels[idx])
        rgb = np.clip(norm[..., None] * color[None, None, :], 0.0, 1.0)
        viewer.add_image(
            rgb,
            name=f"{name} C{idx}",
            opacity=opacity,
            blending="additive",
            rgb=True,
            **image_kwargs,
        )
    return True


def _make_rgb_channel_stack(image: np.ndarray, channel_axis: ImageChannelAxis | str) -> np.ndarray | None:
    data = np.asarray(image)
    if data.ndim != 3:
        return None
    channels = _channels_first_view(data, channel_axis)
    if channels is None:
        return None

    palette = _display_palette()
    rgb_stack = np.zeros((*channels.shape, 3), dtype=np.float32)
    for idx in range(channels.shape[0]):
        color = palette[idx % len(palette)]
        norm = _normalize_channel_for_composite(channels[idx])
        rgb_stack[idx] = norm[..., None] * color[None, None, :]
    return np.clip(rgb_stack, 0.0, 1.0)


def _make_rgb_composite(image: np.ndarray, channel_axis: ImageChannelAxis | str) -> np.ndarray | None:
    data = np.asarray(image)
    if data.ndim != 3:
        return None
    channels = _channels_first_view(data, channel_axis)
    if channels is None:
        return None

    palette = _display_palette()
    rgb = np.zeros((*channels.shape[1:], 3), dtype=np.float32)
    for idx in range(channels.shape[0]):
        color = palette[idx % len(palette)]
        norm = _normalize_channel_for_composite(channels[idx])
        rgb += norm[..., None] * color[None, None, :]
    return np.clip(rgb, 0.0, 1.0)


def _channels_first_view(image: np.ndarray, channel_axis: ImageChannelAxis | str) -> np.ndarray | None:
    data = np.asarray(image)
    napari_axis = _napari_channel_axis(channel_axis, data)
    if data.ndim == 3 and napari_axis == 0:
        return data
    if data.ndim == 3 and napari_axis == -1:
        return np.moveaxis(data, -1, 0)
    return None


def _display_palette() -> np.ndarray:
    return np.array(
        [
            [0.10, 0.25, 1.00],  # DAPI: blue
            [0.00, 1.00, 0.25],  # signal 1: green
            [1.00, 0.05, 0.05],  # signal 2: red
            [1.00, 0.00, 1.00],  # signal 3: magenta
            [1.00, 0.85, 0.00],  # signal 4: yellow
            [0.00, 0.90, 1.00],  # extra: cyan
        ],
        dtype=np.float32,
    )


def _normalize_channel_for_composite(channel: np.ndarray) -> np.ndarray:
    arr = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    p_low, p_high = np.percentile(arr[finite], (1, 99.8))
    if p_high <= p_low:
        max_value = float(np.max(arr[finite]))
        if max_value <= 0:
            return np.zeros(arr.shape, dtype=np.float32)
        return np.clip(arr / max_value, 0.0, 1.0)
    return np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)


def _tiff_axes_for_image(image: np.ndarray, channel_axis: ImageChannelAxis | str) -> str | None:
    if image.ndim == 2:
        return "YX"
    napari_axis = _napari_channel_axis(channel_axis, image)
    if image.ndim == 3 and napari_axis == 0:
        return "CYX"
    if image.ndim == 3 and napari_axis == -1:
        return "YXC"
    return None


def segment_cells_widget(
    viewer: napari.Viewer,
    images: list[ImageData],
    model: CellposeModel = CellposeModel.CPSAM,
    gpu: bool = False,
    channel_axis: ImageChannelAxis = ImageChannelAxis.AUTO,
    dapi_channel: int = 0,
    diameter: float = 15,
    flow_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = -2.0,
    cellprob_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = 1.0,
    min_size: int = 5,
    mode: SegmentationMode = SegmentationMode.AUTO,
    large_image_threshold_mp: Annotated[float, {"min": 1.0, "max": 1000.0, "step": 1.0}] = 64.0,
    chunk_size: Annotated[int, {"min": 256, "max": 8192, "step": 256}] = 2048,
    chunk_overlap: Annotated[int, {"min": 0, "max": 1024, "step": 32}] = 128,
    stitch_labels: bool = True,
    save_masks: bool = False,
    output_dir: str = "./segmentation_output",
):
    """
    Segment cells using Cellpose.

    Supports batch processing of multiple images. Large 2D images can be
    segmented chunk-by-chunk to reduce peak inference memory.
    """
    from tifffile import imwrite

    apply_default_font()

    if not images:
        show_info("Please select at least one image layer.")
        return

    show_info("=== Starting Cell Segmentation ===")
    show_info(f"Model: {model.value} | Total images: {len(images)}")
    show_info(f"DAPI source: channel_axis={_channel_axis_key(channel_axis)} channel={int(dapi_channel)}")

    config = CellposeConfig(
        gpu=gpu,
        pretrained_model=model.value,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    segmenter = CellposeSegmenter(config)

    if save_masks:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Masks will be saved to: {output_path.absolute()}")

    for idx, image in enumerate(images):
        progress = f"[{idx + 1}/{len(images)}]"
        layer_name = _find_layer_name(viewer, image, idx)
        dapi_image = _extract_dapi_channel(image, channel_axis, int(dapi_channel))

        use_chunked, megapixels = _should_use_chunked(
            segmenter,
            dapi_image,
            mode,
            large_image_threshold_mp,
        )
        show_info(f"{progress} Processing {layer_name} ({megapixels:.1f} MP)...")
        if use_chunked:
            stitch_text = "on" if stitch_labels else "off"
            show_info(
                f"{progress} Chunked segmentation: chunk={chunk_size}px "
                f"overlap={chunk_overlap}px stitch={stitch_text}"
            )
            mask, flows, styles = segmenter.segment_array_chunked(
                dapi_image,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                stitch_labels=stitch_labels,
            )
        else:
            show_info(f"{progress} Full-image segmentation; chunk settings are ignored.")
            mask, flows, styles = segmenter.segment_array(dapi_image)
        n_cells = int(mask.max())

        # Truncate long layer names for cleaner UI
        MAX_NAME_LEN = 30
        if len(layer_name) > MAX_NAME_LEN:
            short_name = layer_name[:MAX_NAME_LEN - 3] + "..."
            # Rename the source image layer too
            try:
                viewer.layers[layer_name].name = short_name
            except (KeyError, ValueError):
                pass
            layer_name = short_name

        mask_name = f"{layer_name}_mask"
        _add_mask_layer(viewer, mask, name=mask_name, opacity=0.5)
        show_info(f"{progress} {mask_name}: Found {n_cells} cells")

        if save_masks:
            mask_filename = output_path / f"{layer_name}_mask.tif"
            imwrite(str(mask_filename), _mask_for_saving(mask))
            show_info(f"{progress} Saved: {mask_filename.name}")

    show_info(f"=== Segmentation Complete! Processed {len(images)} image(s) ===")


segment_cells_factory = magic_factory(
    segment_cells_widget,
    call_button="Segment cells",
    widget_init=_segment_cells_widget_init,
    images={"label": "Image layers"},
    channel_axis={"label": "Channel layout"},
    dapi_channel={
        "widget_type": "ComboBox",
        "choices": _dapi_channel_choices,
        "label": "DAPI channel",
    },
)


def _find_layer_name(viewer: napari.Viewer, image: ImageData, idx: int) -> str:
    """Find the selected image layer name without materializing large arrays."""
    for layer in viewer.layers:
        if hasattr(layer, "data") and layer.data is image:
            return layer.name
    return f"Image_{idx + 1}"


def _should_use_chunked(
    segmenter: CellposeSegmenter,
    image: ImageData,
    mode: SegmentationMode,
    large_image_threshold_mp: float,
) -> tuple[bool, float]:
    height, width = segmenter.spatial_shape(image)
    megapixels = (height * width) / 1_000_000.0
    mode_value = mode.value if isinstance(mode, SegmentationMode) else str(mode)
    if mode_value.startswith("chunked"):
        return True, megapixels
    if mode_value.startswith("full image"):
        return False, megapixels
    return megapixels >= float(large_image_threshold_mp), megapixels


def _mask_for_saving(mask: np.ndarray) -> np.ndarray:
    """Use uint32 when uint16 would truncate labels."""
    if int(mask.max()) > np.iinfo(np.uint16).max:
        return mask.astype(np.uint32, copy=False)
    return mask.astype(np.uint16, copy=False)


def _safe_output_stem(name: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(name)).strip(" ._")
    return stem or "mask"


def save_mask_layers_widget(
    viewer: napari.Viewer,
    mask_layer: str = "__all__",
    output_dir: Path = Path("./segmentation_output"),
    relabel_binary_masks: bool = True,
) -> None:
    """Save existing 2D mask layers after segmentation has finished."""
    from tifffile import imwrite

    apply_default_font()
    masks = _selected_label_layers(viewer, mask_layer)
    if not masks:
        show_info("Select at least one mask layer to save.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    show_info(f"Saving {len(masks)} mask layer(s) to: {output_path.absolute()}")

    used_names: set[str] = set()
    for idx, layer in enumerate(masks, start=1):
        layer_name = getattr(layer, "name", f"mask_{idx}")
        try:
            mask = _mask_layer_to_array(layer, "mask")
        except ValueError as exc:
            show_info(f"Skipped {layer_name}: {exc}")
            continue
        mask_to_save = _instance_mask_for_saving(mask, relabel_binary_masks)

        stem = _safe_output_stem(layer_name)
        base_stem = stem
        suffix = 2
        while stem.lower() in used_names:
            stem = f"{base_stem}_{suffix}"
            suffix += 1
        used_names.add(stem.lower())

        out_file = output_path / f"{stem}.tif"
        imwrite(str(out_file), mask_to_save, metadata={"axes": "YX"})
        show_info(f"Saved mask: {out_file.name}")

    show_info("Mask saving complete.")


def _instance_mask_for_saving(mask: np.ndarray, relabel_binary_masks: bool) -> np.ndarray:
    arr = np.asarray(mask)
    if relabel_binary_masks and _looks_like_binary_mask(arr):
        from skimage.measure import label

        arr = label(arr > 0, connectivity=1)
    return _mask_for_saving(arr)


def _looks_like_binary_mask(mask: np.ndarray) -> bool:
    arr = np.asarray(mask)
    if arr.size == 0:
        return False
    values = np.unique(arr)
    nonzero = values[values != 0]
    return len(nonzero) == 1


def _label_layer_choices(widget) -> list[tuple[str, str]]:
    parent = getattr(widget, "parent", None)
    viewer_widget = getattr(parent, "viewer", None)
    viewer = getattr(viewer_widget, "value", None)
    choices = [("All label layers", "__all__")]
    if viewer is None:
        return choices
    choices.extend((layer.name, layer.name) for layer in viewer.layers if isinstance(layer, Labels))
    return choices


def _selected_label_layers(viewer: napari.Viewer, mask_layer: str) -> list[Labels]:
    if str(mask_layer) == "__all__":
        return [layer for layer in viewer.layers if isinstance(layer, Labels)]
    try:
        layer = viewer.layers[str(mask_layer)]
    except KeyError:
        return []
    return [layer] if isinstance(layer, Labels) else []


save_mask_layers_factory = magic_factory(
    save_mask_layers_widget,
    call_button="Save mask layers",
    mask_layer={
        "widget_type": "ComboBox",
        "choices": _label_layer_choices,
        "label": "Mask layer",
    },
    output_dir={"label": "Save to", "widget_type": "FileEdit", "mode": "d"},
    relabel_binary_masks={"label": "Relabel binary masks"},
)


def _mask_for_display(mask: np.ndarray) -> np.ndarray:
    """Use unsigned label textures to avoid unstable int64 labels rendering in napari/vispy."""
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.astype(np.uint16, copy=False)
    if int(np.nanmax(mask)) > np.iinfo(np.uint16).max:
        return mask.astype(np.uint32, copy=False)
    return mask.astype(np.uint16, copy=False)


def _add_mask_layer(
    viewer: napari.Viewer,
    mask: np.ndarray,
    name: str,
    opacity: float,
) -> Layer:
    """Display masks as Labels so label colors stay visible in napari."""
    display_mask = _mask_for_display(mask)
    return viewer.add_labels(display_mask, name=name, opacity=opacity)


def _should_use_wsi_registration(
    image_shape: tuple[int, ...],
    mode: RegistrationMode,
    wsi_threshold_mp: float,
    *,
    metadata_available: bool = False,
) -> tuple[bool, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    megapixels = (height * width) / 1_000_000.0
    mode_value = mode.value if isinstance(mode, RegistrationMode) else str(mode)
    mode_value = mode_value.strip().lower()
    if mode_value == RegistrationMode.WSI.value:
        return True, megapixels
    if mode_value == RegistrationMode.NORMAL.value:
        return False, megapixels
    if mode_value.startswith("fish"):
        return False, megapixels
    # Automatic WSI routing is reserved for layers produced by the WSI
    # workflow.  Large ordinary TIFF/FISH masks have no level-0 coordinate
    # metadata and must stay on the normal mask registration path.
    return megapixels >= float(wsi_threshold_mp) and bool(metadata_available), megapixels


def _run_wsi_registration(
    viewer: napari.Viewer,
    mask1: np.ndarray,
    mask2: np.ndarray,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    fixed_layer_metadata: dict | None,
    moving_layer_metadata: dict | None,
    top_k: int,
    image_megapixels: float,
    channel_axis: ImageChannelAxis | str,
    registered_display: RegisteredDisplayMode | str,
    wsi_refine_model: WSIRefineModel | str,
    wsi_knn_k: int,
    wsi_knn_power: float,
    wsi_residual_filter: bool,
    save_results: bool = False,
    output_path: Path | None = None,
) -> None:
    import time as _time

    _t0 = _time.perf_counter()
    model_value = wsi_refine_model.value if isinstance(wsi_refine_model, WSIRefineModel) else str(wsi_refine_model)
    use_knn = model_value.startswith("knn")
    show_info(f"[3/5] WSI centroid registration in level-0 coordinates ({image_megapixels:.1f} MP)...")
    try:
        fixed_meta = _wsi_metadata_from_layer(fixed_layer_metadata, mask1.shape, "HE", "unknown")
        moving_meta = _wsi_metadata_from_layer(moving_layer_metadata, mask2.shape, "DAPI", "unknown")
        result = run_wsi_centroid_registration(
            fixed_centroids=feats1,
            fixed_metadata=fixed_meta,
            moving_centroids=feats2,
            moving_metadata=moving_meta,
            output_dir=output_path if save_results else None,
            top_k=top_k,
            use_knn_local_affine=use_knn,
            knn_k=max(3, int(wsi_knn_k)),
            knn_power=float(wsi_knn_power),
            residual_filter=bool(wsi_residual_filter),
            write_preview=save_results,
        )
    except ValueError as exc:
        show_info(f"  WSI registration failed: {exc}")
        return
    except NotImplementedError as exc:
        show_info(f"  WSI export skipped: {exc}")
        return

    matches = result.matches.copy()
    residuals = result.residuals
    affine = result.global_affine_mif_to_he
    tx, ty = affine[0, 2], affine[1, 2]
    show_info(f"  WSI centroid matches selected: {len(matches)} spatial cell pairs")
    show_info(
        "  WSI affine mIF level-0 -> HE level-0: "
        f"scale_x={float(affine[0, 0]):.6g} scale_y={float(affine[1, 1]):.6g} "
        f"tx={float(tx):.1f}px ty={float(ty):.1f}px"
    )
    if residuals.size:
        show_info(
            f"  WSI centroid residual mean={float(residuals.mean()):.2f}px "
            f"max={float(residuals.max()):.2f}px in HE level-0 space"
        )
    _add_wsi_centroid_landmark_layers(viewer, result, fixed_meta, moving_meta)

    all_pts_r2_registered_xy = result.registered_moving_centroids[
        ["registered_centroid_x", "registered_centroid_y"]
    ].to_numpy(dtype=float)

    if result.local_affine_warp is not None:
        show_info("[4/5] Applying WSI KNN local affine to mIF-DAPI centroids...")
        method_text = "KNN Local Affine"
    else:
        show_info("[4/5] Applying WSI global affine to mIF-DAPI centroids...")
        method_text = "Global Affine"

    display_registered_yx = _level0_xy_to_layer_yx(all_pts_r2_registered_xy, fixed_meta)
    viewer.add_points(
        display_registered_yx,
        name=f"Registered Points Round 2 (WSI {method_text})",
        size=5,
        face_color="red",
        opacity=0.7,
    )
    show_info("  WSI mode estimated transform from centroids only; 8-channel mIF is not read during registration.")
    if save_results and output_path is not None:
        show_info("  Saved WSI centroid matches, transform JSON, deformation grid, and registered centroid tables.")
    show_info(f"[5/5] WSI registration complete in {_time.perf_counter() - _t0:.2f}s")


def _find_wsi_landmark_matches(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    matches = find_wsi_landmark_matches(feats1, feats2, top_k=top_k, max_angle_deg=2.0)
    if matches.empty:
        show_info("  WSI: no stable real-cell landmark set found.")
        return matches
    show_info(
        "  WSI dominant displacement: "
        f"dx={float(matches['dx'].median()):.1f}px dy={float(matches['dy'].median()):.1f}px, "
        f"mean line length={float(matches['match_line_length_px'].mean()):.1f}px, "
        f"mean neighbor diff={float(matches['neighbor_profile_diff'].mean()):.3f}, "
        f"kept={len(matches)} one-to-one landmarks"
    )
    if "orientation_diff_deg" in matches.columns:
        show_info(
            "  WSI landmark quality: "
            f"mean orientation diff={float(matches['orientation_diff_deg'].mean()):.2f} deg, "
            f"mean neighbor vector diff={float(matches['neighbor_vector_diff'].mean()):.3f}"
        )
    if {"wsi_grid_x", "wsi_grid_y"}.issubset(matches.columns):
        show_info("  WSI 4x4 landmark coverage:\n" + _format_wsi_grid_counts(matches, grid=4))
    return matches


def _empty_wsi_match_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "displacement_residual_px",
            "cell_id_1",
            "cell_id_2",
        ]
    )


def _format_wsi_grid_counts(matches: pd.DataFrame, grid: int) -> str:
    counts = np.zeros((grid, grid), dtype=int)
    for _, row in matches.iterrows():
        gx = int(row["wsi_grid_x"])
        gy = int(row["wsi_grid_y"])
        if 0 <= gx < grid and 0 <= gy < grid:
            counts[gy, gx] += 1
    return "\n".join("    " + " | ".join(f"{counts[y, x]:4d}" for x in range(grid)) for y in range(grid))


def _format_wsi_translation_grid(translation_grid: np.ndarray) -> str:
    rows = []
    for y in range(translation_grid.shape[0]):
        row = []
        for x in range(translation_grid.shape[1]):
            dx, dy = translation_grid[y, x]
            row.append(f"({dx:.0f},{dy:.0f})")
        rows.append("    " + " | ".join(f"{cell:>13s}" for cell in row))
    return "\n".join(rows)


def _robust_standardize_feature_tables(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    vals1 = feats1.loc[:, feature_columns].to_numpy(dtype=float)
    vals2 = feats2.loc[:, feature_columns].to_numpy(dtype=float)
    combined = np.vstack([vals1, vals2])
    med = np.nanmedian(combined, axis=0)
    mad = np.nanmedian(np.abs(combined - med[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    vals1 = np.nan_to_num((vals1 - med[None, :]) / scale[None, :])
    vals2 = np.nan_to_num((vals2 - med[None, :]) / scale[None, :])
    return vals1, vals2


def _select_main_displacement_cluster(displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return select_main_displacement_cluster(displacements)


def _filter_parallel_displacements(
    displacements: np.ndarray,
    reference_disp: np.ndarray,
    max_angle_deg: float,
) -> np.ndarray:
    return filter_parallel_displacements(displacements, reference_disp, max_angle_deg)


def _add_wsi_landmark_layers(
    viewer: napari.Viewer,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
) -> None:
    pts1 = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    pts2 = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_y", "centroid_x"]].to_numpy(dtype=float)
    viewer.add_points(pts1, name="WSI Landmark Round 1", size=9, face_color="yellow")
    viewer.add_points(pts2, name="WSI Landmark Round 2", size=9, face_color="orange")
    lines = [[pts1[idx], pts2[idx]] for idx in range(len(matches))]
    if lines:
        viewer.add_shapes(lines, shape_type="line", edge_width=1, edge_color="cyan", name="WSI Landmark Lines")


def _wsi_metadata_from_layer(
    layer_metadata: dict | None,
    mask_shape: tuple[int, int],
    channel: str,
    segmentation_method: str,
) -> WSICentroidMetadata:
    payload = dict(layer_metadata or {})
    if "wsi_centroid_metadata" in payload and isinstance(payload["wsi_centroid_metadata"], dict):
        payload = dict(payload["wsi_centroid_metadata"])
    if not payload:
        raise ValueError(
            "WSI mode requires mask layers with WSI centroid metadata. "
            "Run WSI Segmentation first or provide centroid CSV files with paired metadata JSON."
        )
    h, w = int(mask_shape[0]), int(mask_shape[1])
    origin_x = float(payload.get("origin_x", 0.0))
    origin_y = float(payload.get("origin_y", 0.0))
    downsample = float(payload.get("downsample", 1.0))
    payload.setdefault("source_wsi_path", payload.get("wsi_path") or payload.get("source_path"))
    payload.setdefault("source_mask_path", payload.get("mask_path"))
    payload.setdefault("coordinate_space", "mask_pixel")
    payload.setdefault("origin_x", origin_x)
    payload.setdefault("origin_y", origin_y)
    payload.setdefault("downsample", downsample)
    payload.setdefault("image_width", int(np.ceil(origin_x + w * downsample)))
    payload.setdefault("image_height", int(np.ceil(origin_y + h * downsample)))
    payload.setdefault("mpp_x", payload.get("mpp_x"))
    payload.setdefault("mpp_y", payload.get("mpp_y"))
    payload.setdefault("mpp_reliable", payload.get("mpp_reliable", False))
    payload.setdefault("channel", payload.get("channel", channel))
    payload.setdefault("segmentation_method", payload.get("segmentation_method", segmentation_method))
    return WSICentroidMetadata.from_mapping(payload, source_name="mask layer metadata")


def _level0_xy_to_layer_yx(points_xy: np.ndarray, metadata: WSICentroidMetadata) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    x = (pts[:, 0] - float(metadata.origin_x)) / float(metadata.downsample)
    y = (pts[:, 1] - float(metadata.origin_y)) / float(metadata.downsample)
    return np.column_stack([y, x])


def _add_wsi_centroid_landmark_layers(
    viewer: napari.Viewer,
    result,
    fixed_metadata: WSICentroidMetadata,
    moving_metadata: WSICentroidMetadata,
) -> None:
    matches = result.matches
    if matches.empty:
        return
    fixed_xy = matches[["fixed_x", "fixed_y"]].to_numpy(dtype=float)
    moving_xy = matches[["moving_x", "moving_y"]].to_numpy(dtype=float)
    fixed_yx = _level0_xy_to_layer_yx(fixed_xy, fixed_metadata)
    moving_yx = _level0_xy_to_layer_yx(moving_xy, moving_metadata)
    viewer.add_points(fixed_yx, name="WSI HE Landmark Centroids", size=9, face_color="yellow")
    viewer.add_points(moving_yx, name="WSI mIF-DAPI Landmark Centroids", size=9, face_color="orange")
    fixed_display_for_registered = _level0_xy_to_layer_yx(fixed_xy, fixed_metadata)
    moving_registered_xy = result.local_affine_warp.predict(moving_xy) if result.local_affine_warp is not None else _apply_affine_for_display(moving_xy, result.global_affine_mif_to_he)
    moving_registered_yx = _level0_xy_to_layer_yx(moving_registered_xy, fixed_metadata)
    lines = [[fixed_display_for_registered[idx], moving_registered_yx[idx]] for idx in range(len(matches))]
    if lines:
        viewer.add_shapes(lines, shape_type="line", edge_width=1, edge_color="cyan", name="WSI Registered Landmark Residuals")


def _apply_affine_for_display(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    hom = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
    out = hom @ np.asarray(matrix, dtype=float).T
    return out[:, :2] / np.maximum(out[:, 2:3], 1e-12)


def _shift_array_xy(arr: np.ndarray, translation_xy: np.ndarray, order: int) -> np.ndarray:
    return shift_array_xy(arr, translation_xy, order)


def _mask_layer_to_array(layer: Layer, parameter_name: str) -> np.ndarray:
    """Accept either Labels layers or image-loaded mask layers."""
    data = np.asarray(layer.data if hasattr(layer, "data") else layer)
    if data.ndim != 2:
        raise ValueError(f"{parameter_name} must be a 2D mask layer, got shape {data.shape}.")
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data)
    if np.nanmin(data) < 0:
        raise ValueError(f"{parameter_name} contains negative labels; expected non-negative mask labels.")
    return data.astype(np.int32, copy=False)


def registration_workflow_widget(
    viewer: napari.Viewer,
    image_round1: ImageData,
    image_round2: ImageData,
    mask_round1: Layer,
    mask_round2: Layer,
    image_channel_axis: ImageChannelAxis = ImageChannelAxis.AUTO,
    registered_display: RegisteredDisplayMode = RegisteredDisplayMode.COLORED_STACK,
    top_k: Annotated[int, {"min": 1, "max": 10000, "step": 100}] = 320,
    max_match_distance_px: int = 100,
    position_weight: float = 1.0,
    residual_prune_quantile: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}] = 0.0,
    use_ransac_transform: bool = True,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 2.0,
    registration_mode: RegistrationMode = RegistrationMode.AUTO,
    wsi_threshold_mp: Annotated[float, {"min": 1.0, "max": 5000.0, "step": 1.0}] = 80.0,
    wsi_refine_model: WSIRefineModel = WSIRefineModel.KNN_LOCAL_AFFINE,
    wsi_knn_k: Annotated[int, {"min": 3, "max": 64, "step": 1}] = 8,
    wsi_knn_power: Annotated[float, {"min": 0.0, "max": 8.0, "step": 0.5}] = 2.0,
    wsi_residual_filter: bool = True,
    min_area: int = 0,
    max_area: int = 0,
    use_gpu: bool = True,
    save_results: bool = False,
    output_dir: str = "./registration_output",
):
    """Run the current cell-registration workflow on pre-segmented masks."""
    from tifffile import imwrite

    apply_default_font()

    def _add_match_layers(current_matches: pd.DataFrame, fixed_feats: pd.DataFrame, moving_feats: pd.DataFrame) -> None:
        if current_matches.empty:
            return

        matched_pts1 = fixed_feats.loc[current_matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
        matched_pts2 = moving_feats.loc[current_matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(matched_pts1, name="Matched Points Round 1", size=8, face_color="yellow")
        viewer.add_points(matched_pts2, name="Matched Points Round 2", size=8, face_color="orange")

        lines = [[matched_pts1[idx], matched_pts2[idx]] for idx in range(len(current_matches))]
        if not lines:
            return

        viewer.add_shapes(
            lines,
            shape_type="line",
            edge_width=1,
            edge_color="cyan",
            name="Match Lines",
        )
        pixel_distances = np.linalg.norm(matched_pts1 - matched_pts2, axis=1)
        show_info(
            "  Match line distances: "
            f"min={pixel_distances.min():.1f}px, "
            f"max={pixel_distances.max():.1f}px, "
            f"mean={pixel_distances.mean():.1f}px"
        )

    show_info("=== Starting Cell Registration Workflow ===")
    show_info(f"  Image channel axis: {_channel_axis_key(image_channel_axis)}")
    display_value = registered_display.value if isinstance(registered_display, RegisteredDisplayMode) else str(registered_display)
    show_info(f"  Registered display: {display_value}")
    from .core import gpu_ops as _gpu_ops
    if use_gpu:
        _torch = _gpu_ops._get_torch_cuda()
        if _torch is not None:
            show_info(f"  GPU accelerated: {_torch.cuda.get_device_name(0)}")
        else:
            show_info("  GPU: not available (using CPU)")
    else:
        show_info("  GPU: disabled by user")
        _gpu_ops.tps_predict_gpu = lambda *a, **k: None
        _gpu_ops.pairwise_cdist_gpu = lambda *a, **k: None

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Results will be saved to: {output_path.absolute()}")

    mask1 = _mask_layer_to_array(mask_round1, "mask_round1")
    mask2 = _mask_layer_to_array(mask_round2, "mask_round2")
    fixed_layer_metadata = dict(getattr(mask_round1, "metadata", {}) or {})
    moving_layer_metadata = dict(getattr(mask_round2, "metadata", {}) or {})
    has_wsi_metadata = all(
        isinstance(metadata.get("wsi_centroid_metadata"), dict)
        for metadata in (fixed_layer_metadata, moving_layer_metadata)
    )
    use_wsi, registration_mp = _should_use_wsi_registration(
        mask1.shape,
        registration_mode,
        wsi_threshold_mp,
        metadata_available=has_wsi_metadata,
    )
    if use_wsi:
        show_info(f"  WSI mode selected from mask shape {mask1.shape} ({registration_mp:.1f} MP).")
    else:
        img1 = np.asarray(image_round1)
        img2 = np.asarray(image_round2)
        show_info(f"  Round 1 image shape: {img1.shape}, mask shape: {mask1.shape}")
        show_info(f"  Round 2 image shape: {img2.shape}, mask shape: {mask2.shape}")
        if img2.ndim == 3:
            axis_key = _channel_axis_key(image_channel_axis)
            if axis_key == "first" or (
                axis_key == "auto" and img2.shape[0] <= 10 and img2.shape[1] > 10 and img2.shape[2] > 10
            ):
                show_info(f"  Round 2 registered output will warp all {img2.shape[0]} channels (C,Y,X).")
            elif axis_key == "last" or (axis_key == "auto" and img2.shape[-1] <= 10):
                show_info(f"  Round 2 registered output will warp all {img2.shape[-1]} channels (Y,X,C).")
        elif img2.ndim == 2:
            show_info("  Round 2 image is 2D; only the selected image layer will be registered.")

    show_info("[1/5] Extracting features from Round 1...")
    feat_config = CellFeaturesConfig(
        min_area=min_area if min_area > 0 else None,
        max_area=max_area if max_area > 0 else None,
        topology_neighbor_k=5,
    )
    feats1 = compute_cell_features(mask1, feat_config)
    n_cells1 = len(feats1)
    show_info(f"  Round 1: {n_cells1} cells after filtering")

    show_info("[2/5] Extracting features from Round 2...")
    feats2 = compute_cell_features(mask2, feat_config)
    n_cells2 = len(feats2)
    show_info(f"  Round 2: {n_cells2} cells after filtering")

    if not feats1.empty:
        pts1 = feats1[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts1, name="All Points Round 1", size=4, face_color="cyan", opacity=0.5)

    if not feats2.empty:
        pts2 = feats2[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts2, name="All Points Round 2", size=4, face_color="magenta", opacity=0.5)

    if n_cells1 == 0 or n_cells2 == 0:
        show_info("No cells available after feature extraction.")
        return

    if use_wsi:
        _run_wsi_registration(
            viewer,
            mask1,
            mask2,
            feats1,
            feats2,
            fixed_layer_metadata=fixed_layer_metadata,
            moving_layer_metadata=moving_layer_metadata,
            top_k=max(1, int(top_k)),
            image_megapixels=registration_mp,
            channel_axis=image_channel_axis,
            registered_display=registered_display,
            wsi_refine_model=wsi_refine_model,
            wsi_knn_k=wsi_knn_k,
            wsi_knn_power=wsi_knn_power,
            wsi_residual_filter=wsi_residual_filter,
            save_results=save_results,
            output_path=output_path if save_results else None,
        )
        return
    mode_value = registration_mode.value if isinstance(registration_mode, RegistrationMode) else str(registration_mode)
    if mode_value.strip().lower().startswith("fish"):
        show_info(f"  FISH mode ({registration_mp:.1f} MP): mask contour coarse registration + local topology matching.")
        fish_result = register_large_fish_masks(
            mask1,
            mask2,
            feats1,
            feats2,
            top_k=max(3, int(top_k)),
        )
        transform = fish_result.transform
        matches = fish_result.matches
        affine_transform = rigid_transform_to_affine(transform)
        rotation_deg = float(np.degrees(np.arctan2(transform.rotation[1, 0], transform.rotation[0, 0])))
        show_info(
            f"  Contour NCC={fish_result.coarse_ncc:.3f}; topology landmarks={len(matches)}; "
            f"residual median={float(np.median(fish_result.residuals)):.2f}px, "
            f"P95={float(np.percentile(fish_result.residuals, 95)):.2f}px"
        )
        show_info(
            f"  Final B->A rigid: rotation={rotation_deg:.3f} deg, "
            f"translation=({float(transform.translation[0]):.1f}, {float(transform.translation[1]):.1f}) px"
        )
        _add_match_layers(matches, feats1, feats2)
        img2_warped = warp_image_with_transform(img2, affine_transform, mask1.shape[:2], order=1)
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        _add_registered_image_layer(
            viewer,
            img2_registered,
            name="Registered Image Round 2 (FISH contour+topology)",
            channel_axis=image_channel_axis,
            display_mode=registered_display,
            opacity=0.5,
        )
        mask2_warped = warp_image_with_transform(
            mask2.astype(float), affine_transform, mask1.shape[:2], order=0
        )
        mask2_registered = np.rint(mask2_warped).astype(np.int32)
        _add_mask_layer(
            viewer,
            mask2_registered,
            name="Registered Mask Round 2 (FISH, pure moving mask)",
            opacity=0.35,
        )
        all_pts_r2_xy = feats2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        all_pts_r2_registered_xy = affine_transform(all_pts_r2_xy)
        viewer.add_points(
            all_pts_r2_registered_xy[:, ::-1],
            name="Registered Points Round 2 (FISH)",
            size=5,
            face_color="red",
            opacity=0.7,
        )
        if save_results:
            image_metadata = (
                {"axes": axes}
                if (axes := _tiff_axes_for_image(img2_registered, image_channel_axis))
                else None
            )
            imwrite(str(output_path / "registered_image_fish.tif"), img2_registered, metadata=image_metadata)
            imwrite(
                str(output_path / "registered_mask_fish_pure_moving.tif"),
                mask2_registered,
                metadata={"axes": "YX"},
            )
            matches.to_csv(output_path / "matches_fish_topology.csv", index=False)
            with open(output_path / "transform_fish.txt", "w", encoding="utf-8") as handle:
                handle.write("Registration Method: FISH mask contour + local cell topology + rigid RANSAC\n")
                handle.write("Transform Direction: moving Round 2 -> fixed Round 1\n")
                handle.write(f"Contour NCC: {fish_result.coarse_ncc:.6f}\n")
                handle.write(f"Topology Landmarks: {len(matches)}\n")
                handle.write(f"Residual Median Px: {float(np.median(fish_result.residuals)):.6f}\n")
                handle.write(f"Residual P95 Px: {float(np.percentile(fish_result.residuals, 95)):.6f}\n")
                handle.write(f"Affine Matrix:\n{affine_transform.params}\n")
        show_info("=== FISH Registration Complete! ===")
        return
    show_info(f"  Normal registration mode ({registration_mp:.1f} MP); using existing workflow.")

    show_info("[3/5] Running morphology-guided matching...")
    max_dist = max(1, int(max_match_distance_px))
    MIN_CONSENSUS_FOR_TPS = 50  # retry with wider window if fewer

    for _window_scale in (1.0, 2.0):
        effective_max_dist = max_dist * _window_scale
        if _window_scale > 1.0:
            show_info(
                f"  Retrying with wider window: {effective_max_dist:.0f}px "
                f"(consensus had too few control points)"
            )
        match_result = two_stage_match_cells(
            feats1,
            feats2,
            mask1.shape,
            feature_weight=1.0,
            topology_weight=0.0,
            position_weight=position_weight,
            top_k=max(1, int(top_k)),
            distance_threshold=None,
            spatial_window_size=float(effective_max_dist),
            min_cells_for_two_stage=10,
            coarse_top_k=max(24, int(top_k)),
            coarse_distance_threshold=2.0,
            coarse_matching_mode="morphology_guided",
            coarse_allow_scale=False,
            coarse_prefer_affine=False,
            coarse_residual_threshold=max(5.0, float(ransac_residual_threshold) * 2.0),
            coarse_max_trials=min(max(int(ransac_max_trials), 200), 2000),
        )
        matches = match_result.matches.copy()

        if len(match_result.coarse_matches) >= 3:
            tx, ty = match_result.coarse_offset_xy
            if match_result.coarse_transform_accepted:
                show_info(
                    "  Coarse translation accepted: "
                    f"{match_result.coarse_inlier_count}/{len(match_result.coarse_matches)} inliers, "
                    f"median residual={match_result.coarse_median_inlier_residual:.2f}px, "
                    f"shift=({float(tx):.1f}, {float(ty):.1f}) px"
                )
            else:
                show_info(
                    "  Coarse translation rejected: "
                    f"{match_result.coarse_inlier_count}/{len(match_result.coarse_matches)} inliers, "
                    f"median residual={match_result.coarse_median_inlier_residual:.2f}px"
                )
        else:
            show_info("  Coarse translation skipped; insufficient confident candidates.")

        if not matches.empty and "distance" in matches.columns:
            show_info(
                "  Match distances: "
                f"min={matches['distance'].min():.2f}, "
                f"max={matches['distance'].max():.2f}, "
                f"mean={matches['distance'].mean():.2f}"
            )

        if matches.empty:
            show_info("No matches available after matching; registration aborted.")
            return

        show_info(f"  Selected matches: {len(matches)}")

        show_info("[4/5] Estimating registration transform...")
        if len(matches) < 3:
            show_info(f"  Only {len(matches)} matches found; need at least 3 to estimate a transform.")
            return

        if use_ransac_transform:
            transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
                feats1,
                feats2,
                matches,
                max_trials=int(ransac_max_trials),
                residual_threshold=float(ransac_residual_threshold),
                min_inliers=MIN_MATCHES_FOR_REFINEMENT,
            )
            matches = matches.copy()
            matches["ransac_inlier"] = inlier_mask
            inlier_count = int(inlier_mask.sum())
            show_info(f"  RANSAC support: {inlier_count}/{len(matches)} inliers")
            all_residuals = compute_match_residuals(feats1, feats2, matches, transform)
            if inlier_count >= MIN_MATCHES_FOR_REFINEMENT:
                inlier_residuals = all_residuals[inlier_mask]
                inlier_median = float(np.median(inlier_residuals))
                inlier_mad = float(np.median(np.abs(inlier_residuals - inlier_median)))
                robust_scale = max(1.4826 * inlier_mad, 0.5)
                model_threshold = max(
                    float(ransac_residual_threshold) * 2.0,
                    inlier_median + 3.0 * robust_scale,
                )
                keep_mask = all_residuals <= model_threshold
                keep_count = int(keep_mask.sum())
                if MIN_MATCHES_FOR_REFINEMENT <= keep_count < len(matches):
                    matches = matches.loc[keep_mask].reset_index(drop=True)
                    transform, _ = estimate_rigid_transform_from_matches_ransac(
                        feats1,
                        feats2,
                        matches,
                        max_trials=int(ransac_max_trials),
                        residual_threshold=float(ransac_residual_threshold),
                        min_inliers=MIN_MATCHES_FOR_REFINEMENT,
                    )
                    show_info(
                        "  RANSAC model consistency filter: "
                        f"kept {keep_count}/{len(keep_mask)} matches at <= {model_threshold:.2f}px"
                    )
            else:
                show_info("  Too few RANSAC inliers to filter matches safely; keeping all matches")
        else:
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        residuals = compute_match_residuals(feats1, feats2, matches, transform)
        matches = matches.copy()
        matches["residual_px"] = residuals

        if (
            0.0 < float(residual_prune_quantile) < 1.0
            and len(matches) >= MIN_MATCHES_FOR_REFINEMENT
        ):
            threshold = float(np.quantile(residuals, float(residual_prune_quantile)))
            keep_mask = residuals <= threshold
            kept = int(keep_mask.sum())
            if kept >= MIN_MATCHES_FOR_REFINEMENT and kept < len(matches):
                matches = matches.loc[keep_mask].reset_index(drop=True)
                transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
                matches["residual_px"] = compute_match_residuals(feats1, feats2, matches, transform)
                show_info(
                    f"  Residual pruning: kept {kept}/{len(residuals)} matches at <= {threshold:.2f}px"
                )

        # --- Guided rematch: re-match on transform-aligned data with patch coverage ---
        if len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
            show_info("  Guided rematch: re-matching under estimated transform...")
            initial_affine = rigid_transform_to_affine(transform)
            aligned_feats2 = apply_transform_to_features(feats2, initial_affine, mask1.shape)
            rematch_window = max(10.0, float(effective_max_dist) * 0.6)
            if _window_scale > 1.0:
                rematch_window = max(rematch_window, 150.0)
            rematch_config = MatchingConfig(
                feature_weight=1.0,
                topology_weight=0.0,
                position_weight=position_weight,
                top_k=max(1, int(top_k)),
                distance_threshold=None,
                spatial_window_size=rematch_window,
            )
            rematch_matches = greedy_match_cells(
                feats1, aligned_feats2, rematch_config, image_shape=mask1.shape,
                coverage_patch_grid=4,
            )
            if len(rematch_matches) >= MIN_MATCHES_FOR_REFINEMENT:
                transform = estimate_rigid_transform_from_matches(
                    feats1, feats2, rematch_matches,
                )
                rematch_residuals = compute_match_residuals(
                    feats1, feats2, rematch_matches, transform,
                )
                rematch_matches = rematch_matches.copy()
                rematch_matches["residual_px"] = rematch_residuals
                show_info(
                    f"  Guided rematch: {len(rematch_matches)} matches, "
                    f"mean residual={float(rematch_residuals.mean()):.2f}px"
                )
                matches = rematch_matches
            else:
                show_info("  Guided rematch: too few matches; keeping original.")

        affine_transform = rigid_transform_to_affine(transform)
        rotation_deg = float(np.degrees(np.arctan2(transform.rotation[1, 0], transform.rotation[0, 0])))
        show_info(
            "  Final rigid baseline: "
            f"rotation={rotation_deg:.2f} deg, "
            f"translation=({float(transform.translation[0]):.1f}, {float(transform.translation[1]):.1f}) px"
        )
        if len(matches) > 0 and "residual_px" in matches.columns:
            show_info(
                "  Rigid residuals: "
                f"min={matches['residual_px'].min():.2f}px, "
                f"max={matches['residual_px'].max():.2f}px, "
                f"mean={matches['residual_px'].mean():.2f}px"
            )

        # --- Orientation filter ---
        n_before_orient = len(matches)
        if n_before_orient >= MIN_MATCHES_FOR_REFINEMENT:
            matches = _filter_candidate_matches_by_hard_constraints(
                matches,
                max_area_ratio=None,
                max_aspect_ratio_ratio=None,
                max_orientation_diff_deg=5.0,
                min_orientation_eccentricity=0.15,
            )
            n_after_orient = len(matches)
            if n_after_orient < n_before_orient:
                show_info(
                    f"  Orientation filter (<=5°): kept {n_after_orient}/{n_before_orient} matches"
                )
                if n_after_orient >= MIN_MATCHES_FOR_REFINEMENT:
                    transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        # --- Local displacement consistency filter (per-patch) ---
        n_before_disp = len(matches)
        if n_before_disp >= MIN_MATCHES_FOR_REFINEMENT:
            pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][
                ["centroid_x", "centroid_y"]
            ].to_numpy(dtype=float)
            pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][
                ["centroid_x", "centroid_y"]
            ].to_numpy(dtype=float)
            disp_vectors = pts_f - pts_m

            h, w = mask1.shape[:2]
            grid = 4
            px = np.clip(np.floor(pts_f[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
            py = np.clip(np.floor(pts_f[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
            patch_ids = py * grid + px

            keep_mask = np.ones(n_before_disp, dtype=bool)
            angle_threshold_deg = 5.0
            for pid in range(grid * grid):
                in_patch = patch_ids == pid
                n_in = int(in_patch.sum())
                if n_in < 3:
                    continue
                patch_disp = disp_vectors[in_patch]
                patch_indices = np.where(in_patch)[0]
                patch_norms = np.linalg.norm(patch_disp, axis=1)

                cos_thresh = np.cos(np.radians(angle_threshold_deg))
                votes = np.zeros(n_in, dtype=int)
                for a in range(n_in):
                    if patch_norms[a] < 1.0:
                        continue
                    for b in range(n_in):
                        if patch_norms[b] < 1.0:
                            continue
                        cos_ab = np.dot(patch_disp[a], patch_disp[b]) / (
                            patch_norms[a] * patch_norms[b] + 1e-8
                        )
                        if cos_ab >= cos_thresh:
                            votes[a] += 1

                if votes.max() < 2:
                    continue
                consensus_idx = int(np.argmax(votes))
                consensus_disp = patch_disp[consensus_idx]
                consensus_norm = patch_norms[consensus_idx]

                consensus_members = []
                for k in range(n_in):
                    if patch_norms[k] < 1.0:
                        keep_mask[patch_indices[k]] = False
                        continue
                    cos_val = np.dot(patch_disp[k], consensus_disp) / (
                        patch_norms[k] * consensus_norm + 1e-8
                    )
                    if cos_val < cos_thresh:
                        keep_mask[patch_indices[k]] = False
                    else:
                        consensus_members.append(k)

                if len(consensus_members) >= 3:
                    member_lengths = np.array([patch_norms[k] for k in consensus_members])
                    mean_len = float(np.mean(member_lengths))
                    for k in consensus_members:
                        if abs(patch_norms[k] - mean_len) > 5.0:
                            keep_mask[patch_indices[k]] = False

            n_after_disp = int(keep_mask.sum())
            if n_after_disp >= MIN_MATCHES_FOR_REFINEMENT and n_after_disp < n_before_disp:
                matches = matches.loc[keep_mask].reset_index(drop=True)
                show_info(
                    f"  Displacement consensus filter: kept {n_after_disp}/{n_before_disp} matches"
                )
                transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        # Check if enough control points for TPS; if not, retry with wider window
        if len(matches) >= MIN_CONSENSUS_FOR_TPS:
            break  # enough points, no retry needed
        show_info(
            f"  Only {len(matches)} control points after filtering "
            f"(need {MIN_CONSENSUS_FOR_TPS})"
        )

    # --- Rigid fallback: relaxed matching + dominant direction ---
    if len(matches) < MIN_CONSENSUS_FOR_TPS:
        show_info("  Rigid fallback: relaxed matching at 100px, finding dominant direction...")
        fb_result = two_stage_match_cells(
            feats1, feats2, mask1.shape,
            feature_weight=1.0, topology_weight=0.0, position_weight=position_weight,
            top_k=max(1, int(top_k)), distance_threshold=None,
            spatial_window_size=float(max_dist),
            min_cells_for_two_stage=10, coarse_top_k=max(24, int(top_k)),
            coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
            coarse_allow_scale=False, coarse_prefer_affine=False,
            coarse_residual_threshold=max(5.0, float(ransac_residual_threshold) * 2.0),
            coarse_max_trials=min(max(int(ransac_max_trials), 200), 2000),
        )
        fb_matches = fb_result.matches.copy()

        # RANSAC
        fb_transform, _ = estimate_rigid_transform_from_matches_ransac(
            feats1, feats2, fb_matches,
            max_trials=int(ransac_max_trials), residual_threshold=float(ransac_residual_threshold),
            min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )

        # Guided rematch
        if len(fb_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            fb_affine = rigid_transform_to_affine(fb_transform)
            fb_aligned = apply_transform_to_features(feats2, fb_affine, mask1.shape)
            fb_cfg = MatchingConfig(
                feature_weight=1.0, topology_weight=0.0, position_weight=position_weight,
                top_k=max(1, int(top_k)), distance_threshold=None,
                spatial_window_size=max(10.0, float(max_dist) * 0.6),
            )
            fb_rematch = greedy_match_cells(feats1, fb_aligned, fb_cfg, image_shape=mask1.shape, coverage_patch_grid=4)
            if len(fb_rematch) >= MIN_MATCHES_FOR_REFINEMENT:
                fb_transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_rematch)
                fb_matches = fb_rematch

        # Orientation filter only (no consensus)
        fb_matches = _filter_candidate_matches_by_hard_constraints(
            fb_matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
            max_orientation_diff_deg=5.0, min_orientation_eccentricity=0.15,
        )
        if len(fb_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            fb_transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_matches)

        # Find dominant displacement group (angle≤5°, length≤5px)
        if len(fb_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            pts_f = feats1.iloc[fb_matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            pts_m = feats2.iloc[fb_matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            disps = pts_f - pts_m
            norms = np.linalg.norm(disps, axis=1)
            angles = np.arctan2(disps[:, 1], disps[:, 0])
            angle_tol = np.radians(5.0)
            length_tol = 5.0
            n_fb = len(fb_matches)
            best_group = []
            for i in range(n_fb):
                group = []
                for j in range(n_fb):
                    a_diff = abs(angles[i] - angles[j])
                    a_diff = min(a_diff, 2 * np.pi - a_diff)
                    if a_diff <= angle_tol and abs(norms[i] - norms[j]) <= length_tol:
                        group.append(j)
                if len(group) > len(best_group):
                    best_group = group
            show_info(f"  Dominant direction group: {len(best_group)}/{n_fb} matches")
            if len(best_group) >= MIN_MATCHES_FOR_REFINEMENT:
                fb_matches = fb_matches.iloc[best_group].reset_index(drop=True)
                transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_matches)
                matches = fb_matches

    affine_transform = rigid_transform_to_affine(transform)
    use_tps = len(matches) >= MIN_CONSENSUS_FOR_TPS

    if use_tps:
        # --- Fit TPS (Thin Plate Spline) for non-rigid registration ---
        show_info("  Fitting TPS non-rigid transform from matched landmarks...")
        pts_fixed_xy = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][
            ["centroid_x", "centroid_y"]
        ].to_numpy(dtype=float)
        pts_moving_xy = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][
            ["centroid_x", "centroid_y"]
        ].to_numpy(dtype=float)
        tps = fit_tps_from_matches(
            pts_fixed_xy,
            pts_moving_xy,
            output_shape=mask1.shape[:2],
            rigid_transform=affine_transform,
            regularization=1e-3,
            n_boundary_per_side=4,
            add_boundary_anchors_flag=True,
        )
        tps_predicted = tps.predict(pts_fixed_xy)
        tps_residuals = np.linalg.norm(
            tps_predicted - pts_moving_xy, axis=1
        )[: len(pts_fixed_xy)]
        show_info(
            f"  TPS fitted with {len(pts_fixed_xy)} control points + boundary anchors, "
            f"control-point residual: mean={float(tps_residuals.mean()):.3f}px"
        )

    show_info("[5/5] Applying TPS transformation and creating overlay...")
    _add_match_layers(matches, feats1, feats2)

    import time as _time

    if use_tps:
        # Warp image with TPS (non-rigid)
        show_info(f"  Warping image with TPS; moving image shape {img2.shape}")
        _t0 = _time.perf_counter()
        img2_warped = warp_image_with_tps(img2, tps, mask1.shape[:2], order=1)
        _warp_sec = _time.perf_counter() - _t0
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        show_info(f"  TPS image warp completed in {_warp_sec:.2f}s; registered shape {img2_registered.shape}")
        _add_registered_image_layer(
            viewer,
            img2_registered,
            name="Registered Image Round 2 (TPS)",
            channel_axis=image_channel_axis,
            display_mode=registered_display,
            opacity=0.5,
        )

        mask2_warped = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
        mask2_registered = np.rint(mask2_warped).astype(np.int32)
    else:
        # Rigid fallback warp
        show_info(f"  Rigid fallback warp: {len(matches)} matches, moving image shape {img2.shape}")
        _t0 = _time.perf_counter()
        img2_warped = warp_image_with_transform(img2, affine_transform, mask1.shape[:2], order=1)
        img2_registered = cast_warped_like_original(img2_warped, img2.dtype)
        show_info(f"  Rigid registered moving image shape: {img2_registered.shape}")
        _add_registered_image_layer(
            viewer,
            img2_registered,
            name="Registered Image Round 2 (Rigid)",
            channel_axis=image_channel_axis,
            display_mode=registered_display,
            opacity=0.5,
        )

        mask2_warped = warp_image_with_transform(mask2.astype(float), affine_transform, mask1.shape[:2], order=0)
        mask2_registered = np.rint(mask2_warped).astype(np.int32)
        _warp_sec = _time.perf_counter() - _t0
        show_info(f"  Rigid warp completed in {_warp_sec:.2f}s")

    # Keep multichannel registered intensities unchanged; 2D legacy view keeps the old fusion overlay.
    mask2_registered = np.where(mask2_registered == 0, mask1, mask2_registered)

    if img1.ndim == 2 and img1.shape == img2_registered.shape:
        img2_registered = np.maximum(img1, img2_registered)
    _add_mask_layer(viewer, mask2_registered, name="Registered Mask Round 2", opacity=0.35)

    # Warp ALL round2 centroids
    all_pts_r2_xy = feats2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    if use_tps:
        all_pts_r2_registered_xy = tps.predict(all_pts_r2_xy)
    else:
        ones_all = np.ones((len(all_pts_r2_xy), 1))
        all_pts_r2_registered_xy = (affine_transform.params @ np.hstack([all_pts_r2_xy, ones_all]).T).T[:, :2]
    viewer.add_points(
        all_pts_r2_registered_xy[:, ::-1],
        name="Registered Points Round 2",
        size=5,
        face_color="red",
        opacity=0.7,
    )

    # Draw patch grid lines (4x4)
    h, w = mask1.shape[:2]
    grid = 4
    grid_lines = []
    for i in range(1, grid):
        # Vertical lines: x = i * w / grid (in napari coords: col = x, row = y)
        x = i * w / grid
        grid_lines.append(np.array([[0, x], [h, x]]))
        # Horizontal lines: y = i * h / grid
        y = i * h / grid
        grid_lines.append(np.array([[y, 0], [y, w]]))
    viewer.add_shapes(
        grid_lines, shape_type="line", edge_color="yellow",
        edge_width=2, name="Patch Grid (4x4)", opacity=0.6,
    )

    show_info("  Visualization complete.")

    if save_results:
        show_info("Saving results...")
        img_filename = output_path / "registered_image.tif"
        image_metadata = {"axes": axes} if (axes := _tiff_axes_for_image(img2_registered, image_channel_axis)) else None
        imwrite(str(img_filename), img2_registered, metadata=image_metadata)
        show_info(f"  Saved: {img_filename.name}")

        mask_filename = output_path / "registered_mask.tif"
        imwrite(str(mask_filename), mask2_registered, metadata={"axes": "YX"})
        show_info(f"  Saved: {mask_filename.name}")

        feats1_csv = output_path / "features_round1.csv"
        feats1.to_csv(feats1_csv, index=False)
        show_info(f"  Saved: {feats1_csv.name}")

        feats2_csv = output_path / "features_round2.csv"
        feats2.to_csv(feats2_csv, index=False)
        show_info(f"  Saved: {feats2_csv.name}")

        matches_csv = output_path / "matches.csv"
        matches.to_csv(matches_csv, index=False)
        show_info(f"  Saved: {matches_csv.name} ({len(matches)} matches)")

        registered_feats2 = feats2.copy()
        registered_feats2["centroid_x"] = all_pts_r2_registered_xy[:, 0]
        registered_feats2["centroid_y"] = all_pts_r2_registered_xy[:, 1]
        registered_feats2["pos_x_norm"] = registered_feats2["centroid_x"] / float(max(mask1.shape[1], 1))
        registered_feats2["pos_y_norm"] = registered_feats2["centroid_y"] / float(max(mask1.shape[0], 1))
        registered_feats2 = assign_patches(registered_feats2, mask1.shape[1], mask1.shape[0])

        registered_feats_csv = output_path / "registered_features_round2.csv"
        registered_feats2.to_csv(registered_feats_csv, index=False)
        show_info(f"  Saved: {registered_feats_csv.name}")

        pts_r2_reg_df = registered_feats2.loc[:, ["cell_id", "centroid_y", "centroid_x"]]
        reg_pts_csv = output_path / "registered_centroids_round2.csv"
        pts_r2_reg_df.to_csv(reg_pts_csv, index=False)
        show_info(f"  Saved: {reg_pts_csv.name}")

        transform_file = output_path / "transform_info.txt"
        with open(transform_file, "w", encoding="utf-8") as handle:
            handle.write("Registration Method: TPS (Thin Plate Spline)\n")
            handle.write(f"Control Points: {len(pts_fixed_xy)}\n")
            handle.write(f"TPS Regularization: 1e-3\n\n")
            handle.write("Rigid Baseline (Affine Transform Matrix):\n")
            handle.write(f"{affine_transform.params}\n\n")
            handle.write("Rotation Matrix:\n")
            handle.write(f"{transform.rotation}\n\n")
            handle.write("Translation Vector:\n")
            handle.write(f"{transform.translation}\n")
        show_info(f"  Saved: {transform_file.name}")

    show_info("=== Registration Complete! ===")
    show_info("Check the registered image, mask, and point layers for results.")

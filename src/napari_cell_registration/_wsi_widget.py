"""OpenSlide WSI segmentation and registration widgets."""

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import napari
from magicgui import magicgui
from napari.utils.notifications import show_info
from skimage.draw import disk, polygon
from skimage.transform import resize

from ._qt_init import apply_default_font
from .core import CellposeConfig, CellposeSegmenter


class CellViTModel(Enum):
    """Installed CellViT++ model choices."""

    SAM = "SAM"
    HIPT = "HIPT"


class WSISegmentationModel(Enum):
    """WSI segmentation backend choices."""

    CELLVIT = "CellViT++"
    CELLPOSE = "Cellpose-SAM"


@dataclass(frozen=True)
class WSIChannel:
    index: int
    name: str
    color: tuple[int, int, int] | None = None


def wsi_segmentation_widget(
    viewer: napari.Viewer,
    fixed_wsi_path: Path = Path("."),
    moving_wsi_path: Path = Path("."),
    fixed_segmentation_model: WSISegmentationModel = WSISegmentationModel.CELLVIT,
    moving_segmentation_model: WSISegmentationModel = WSISegmentationModel.CELLPOSE,
    cellvit_model: CellViTModel = CellViTModel.SAM,
    redownload_cellvit_model: bool = False,
    output_dir: str = "./wsi_registration_output",
    use_cell_shapes: bool = True,
    binary_cell_segmentation: bool = True,
    use_gpu: bool = True,
    resolution: float = 0.25,
    batch_size: int = 8,
    read_level: int = 4,
    region_x: int = 0,
    region_y: int = 0,
    region_width: int = 0,
    region_height: int = 0,
    moving_segmentation_channel: str = "Auto DAPI",
    show_moving_segmentation_channel: bool = True,
    show_all_moving_channels: bool = False,
    add_moving_channel_stack: bool = True,
    cellpose_read_level: int = 2,
    export_cellpose_source_image: bool = False,
    cellpose_chunk_size: int = 4096,
    cellpose_chunk_overlap: int = 128,
    max_cellpose_tiles: int = 80,
    point_radius_px: int = 5,
) -> None:
    """Read WSI files with OpenSlide and add segmentation layers."""
    apply_default_font()

    fixed_path = Path(fixed_wsi_path)
    moving_path = Path(moving_wsi_path)
    out_root = Path(output_dir)
    uses_cellvit = (
        fixed_segmentation_model == WSISegmentationModel.CELLVIT
        or moving_segmentation_model == WSISegmentationModel.CELLVIT
    )
    if not fixed_path.is_file() or not moving_path.is_file():
        show_info("Select existing fixed and moving WSI files.")
        return
    if not use_gpu and uses_cellvit:
        show_info("CellViT++ inference currently requires CUDA; enable GPU to run segmentation.")
        return
    if uses_cellvit and float(resolution) not in (0.25, 0.5):
        show_info("CellViT++ resolution must be 0.25 or 0.5.")
        return

    show_info("=== Starting WSI Segmentation (OpenSlide) ===")
    out_root.mkdir(parents=True, exist_ok=True)
    fixed_img, fixed_meta = _read_wsi_region(
        fixed_path, read_level, region_x, region_y, region_width, region_height
    )
    moving_img, moving_meta = _read_wsi_region(
        moving_path, read_level, region_x, region_y, region_width, region_height
    )
    viewer.add_image(fixed_img, name=f"{fixed_path.stem} WSI", rgb=True)
    viewer.add_image(moving_img, name=f"{moving_path.stem} WSI", rgb=True)
    moving_channels = _read_wsi_channels(moving_path)
    moving_channel = _match_channel(moving_channels, moving_segmentation_channel) if moving_channels else None
    moving_channel_img = None
    if moving_channels:
        _show_channel_inventory(moving_path, moving_channels)
        moving_channel_img = _add_selected_channel_layers(
            viewer,
            moving_path,
            moving_channels,
            read_level,
            region_x,
            region_y,
            region_width,
            region_height,
            moving_segmentation_channel,
            show_moving_segmentation_channel,
            show_all_moving_channels,
        )
        if add_moving_channel_stack:
            _add_channel_stack_layer(
                viewer,
                moving_path,
                moving_channels,
                read_level,
                region_x,
                region_y,
                region_width,
                region_height,
            )
    show_info("WSI previews loaded. Running selected WSI segmentation models...")

    fixed_cellvit_dir = out_root / "fixed_cellvit"
    moving_cellvit_dir = out_root / "moving_cellvit"
    try:
        if fixed_segmentation_model == WSISegmentationModel.CELLVIT:
            _run_cellvit(
                fixed_path, "", cellvit_model.value, fixed_cellvit_dir,
                use_cell_shapes, binary_cell_segmentation, use_gpu, resolution, batch_size,
                redownload_cellvit_model,
            )
            fixed_mask = _cellvit_output_to_mask(
                fixed_path,
                fixed_cellvit_dir,
                fixed_img.shape[:2],
                fixed_meta,
                use_cell_shapes,
                point_radius_px,
            )
        else:
            fixed_mask = _segment_cellpose_source_to_target(
                fixed_path,
                None,
                cellpose_read_level,
                region_x,
                region_y,
                region_width,
                region_height,
                out_root,
                "fixed WSI",
                use_gpu,
                cellpose_chunk_size,
                cellpose_chunk_overlap,
                max_cellpose_tiles,
                fixed_meta,
                fixed_img.shape[:2],
                export_cellpose_source_image,
            )

        if moving_segmentation_model == WSISegmentationModel.CELLPOSE:
            moving_label = (
                f"moving WSI channel {moving_channel.index}: {moving_channel.name}"
                if moving_channel is not None
                else "moving WSI"
            )
            moving_mask = _segment_cellpose_source_to_target(
                moving_path,
                moving_channel,
                cellpose_read_level,
                region_x,
                region_y,
                region_width,
                region_height,
                out_root,
                moving_label,
                use_gpu,
                cellpose_chunk_size,
                cellpose_chunk_overlap,
                max_cellpose_tiles,
                moving_meta,
                moving_img.shape[:2],
                export_cellpose_source_image,
            )
        else:
            if moving_channel is not None:
                show_info(
                    "CellViT++ reads the original moving WSI. "
                    "Use Cellpose-SAM if you want segmentation from the selected DAPI channel."
                )
            _run_cellvit(
                moving_path, "", cellvit_model.value, moving_cellvit_dir,
                use_cell_shapes, binary_cell_segmentation, use_gpu, resolution, batch_size,
                False,
            )
            moving_mask = _cellvit_output_to_mask(
                moving_path,
                moving_cellvit_dir,
                moving_img.shape[:2],
                moving_meta,
                use_cell_shapes,
                point_radius_px,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        show_info(str(exc))
        return

    viewer.add_labels(
        fixed_mask,
        name=f"{fixed_path.stem} {fixed_segmentation_model.value} mask",
        opacity=0.45,
        metadata={
            "wsi_centroid_metadata": _make_wsi_centroid_layer_metadata(
                fixed_path,
                fixed_meta,
                fixed_mask.shape[:2],
                channel="HE",
                segmentation_method=fixed_segmentation_model.value,
            )
        },
    )
    viewer.add_labels(
        moving_mask,
        name=f"{moving_path.stem} {moving_segmentation_model.value} mask",
        opacity=0.45,
        metadata={
            "wsi_centroid_metadata": _make_wsi_centroid_layer_metadata(
                moving_path,
                moving_meta,
                moving_mask.shape[:2],
                channel=(moving_channel.name if moving_channel is not None else "DAPI"),
                segmentation_method=moving_segmentation_model.value,
            )
        },
    )
    show_info(
        "WSI masks loaded: "
        f"fixed={int(fixed_mask.max())} cells, moving={int(moving_mask.max())} cells"
    )
    show_info("Inspect segmentation layers, then run Registration Workflow; WSI mode will use centroid metadata.")


def _read_wsi_region(
    path: Path,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide-python is required for WSI reading.") from exc

    slide = openslide.OpenSlide(str(path))
    try:
        level_group = _select_openslide_level_group(slide, level)
        read_level = level_group[0]
        downsample = float(slide.level_downsamples[read_level])
        level_w, level_h = slide.level_dimensions[read_level]
        if width <= 0 or height <= 0:
            location = (0, 0)
            size = (int(level_w), int(level_h))
            origin_x = 0
            origin_y = 0
        else:
            origin_x = int(max(x, 0))
            origin_y = int(max(y, 0))
            location = (origin_x, origin_y)
            size = (
                max(1, int(np.ceil(width / downsample))),
                max(1, int(np.ceil(height / downsample))),
            )
        if len(level_group) > 1:
            rgb = _read_multichannel_wsi_composite(slide, level_group, location, size)
        else:
            rgb = np.asarray(slide.read_region(location, read_level, size).convert("RGB"))
            rgb = _auto_contrast_grayscale_rgb(rgb)
        meta = _slide_coordinate_metadata(slide, origin_x, origin_y, downsample)
    finally:
        slide.close()
    return rgb, meta


def _slide_coordinate_metadata(slide, origin_x: int, origin_y: int, downsample: float) -> dict[str, float | int | bool | None]:
    mpp_x = _optional_float(slide.properties.get("openslide.mpp-x"))
    mpp_y = _optional_float(slide.properties.get("openslide.mpp-y"))
    width, height = slide.level_dimensions[0]
    return {
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "downsample": float(downsample),
        "image_width": int(width),
        "image_height": int(height),
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "mpp_reliable": mpp_x is not None and mpp_y is not None,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _make_wsi_centroid_layer_metadata(
    wsi_path: Path,
    region_meta: dict[str, Any],
    mask_shape: tuple[int, int],
    channel: str,
    segmentation_method: str,
) -> dict[str, Any]:
    h, w = int(mask_shape[0]), int(mask_shape[1])
    downsample = float(region_meta.get("downsample", 1.0))
    origin_x = float(region_meta.get("origin_x", 0.0))
    origin_y = float(region_meta.get("origin_y", 0.0))
    return {
        "source_wsi_path": str(wsi_path),
        "source_mask_path": None,
        "coordinate_space": "mask_pixel",
        "origin_x": origin_x,
        "origin_y": origin_y,
        "downsample": downsample,
        "image_width": int(region_meta.get("image_width", np.ceil(origin_x + w * downsample))),
        "image_height": int(region_meta.get("image_height", np.ceil(origin_y + h * downsample))),
        "mpp_x": region_meta.get("mpp_x"),
        "mpp_y": region_meta.get("mpp_y"),
        "mpp_reliable": bool(region_meta.get("mpp_reliable", False)),
        "channel": str(channel),
        "segmentation_method": str(segmentation_method),
    }


def _read_wsi_channels(path: Path) -> list[WSIChannel]:
    try:
        import tifffile
    except ImportError:
        return []

    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.series or len(tif.series[0].shape) < 3:
                return []
            series = tif.series[0]
            axes = getattr(series, "axes", "")
            if "C" in axes:
                channel_axis = axes.index("C")
            elif series.shape[0] <= 64 and series.shape[-1] not in (3, 4):
                channel_axis = 0
            else:
                return []
            channel_count = int(series.shape[channel_axis])
            if channel_count <= 1:
                return []
            channels = []
            for index in range(channel_count):
                page = tif.pages[index]
                name, color = _parse_channel_description(page.description or "")
                channels.append(WSIChannel(index=index, name=name or f"Channel {index}", color=color))
            return channels
    except Exception:
        return []


def _parse_channel_description(description: str) -> tuple[str | None, tuple[int, int, int] | None]:
    if not description:
        return None, None
    try:
        root = ET.fromstring(description.encode("utf-16") if description.startswith("<?xml") else description)
    except ET.ParseError:
        return None, None
    name = root.findtext("Name")
    color_text = root.findtext("Color")
    color = None
    if color_text:
        try:
            parts = [int(float(part.strip())) for part in color_text.split(",")]
            if len(parts) == 3:
                color = tuple(int(np.clip(part, 0, 255)) for part in parts)
        except ValueError:
            color = None
    return name, color


def _show_channel_inventory(path: Path, channels: list[WSIChannel]) -> None:
    names = ", ".join(f"{channel.index}:{channel.name}" for channel in channels)
    show_info(f"Detected moving WSI channels for {path.name}: {names}")


def _add_selected_channel_layers(
    viewer: napari.Viewer,
    path: Path,
    channels: list[WSIChannel],
    read_level: int,
    x: int,
    y: int,
    width: int,
    height: int,
    selected_channel: str,
    show_selected: bool,
    show_all: bool,
) -> np.ndarray | None:
    if not show_selected and not show_all:
        return None
    selected = _match_channel(channels, selected_channel)
    channels_to_show = channels if show_all else ([selected] if selected is not None else [])
    if show_selected and selected is None and not show_all:
        show_info(f"Moving segmentation channel was not found: {selected_channel}")
        return None
    selected_img = None
    for channel in channels_to_show:
        channel_img, _ = _read_wsi_channel_region(path, channel.index, read_level, x, y, width, height)
        if selected is not None and channel.index == selected.index:
            selected_img = channel_img
        viewer.add_image(
            channel_img,
            name=f"{path.stem} channel {channel.index} - {channel.name}",
            colormap=_napari_colormap(channel),
            blending="additive",
        )
    if selected_img is None and selected is not None:
        selected_img, _ = _read_wsi_channel_region(path, selected.index, read_level, x, y, width, height)
    return selected_img


def _add_channel_stack_layer(
    viewer: napari.Viewer,
    path: Path,
    channels: list[WSIChannel],
    read_level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    stack = []
    names = []
    for channel in channels:
        channel_img, _ = _read_wsi_channel_region(path, channel.index, read_level, x, y, width, height)
        stack.append(channel_img)
        names.append(f"{channel.index}:{channel.name}")
    if not stack:
        return
    viewer.add_image(
        np.stack(stack, axis=0),
        name=f"{path.stem} all channels stack",
        channel_axis=0,
        blending="additive",
        metadata={"channel_names": names},
    )
    show_info(
        f"Added moving all-channel stack for inspection ({len(stack)} channels, C/Y/X). "
        "WSI registration estimates transform from mIF-DAPI only; all channels are warped tile-wise after transform export."
    )


def _image_to_cellpose_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.asarray(image)
    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return _stretch_channel(gray.astype(np.float32))
    raise ValueError(f"Unsupported image shape for Cellpose segmentation: {image.shape}")


def _cellpose_source_info(
    path: Path,
    channel: WSIChannel | None,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int], dict[str, float], str]:
    if channel is None:
        shape, meta = _wsi_region_shape(path, level, x, y, width, height)
        return shape, meta, "grayscale"
    shape, meta = _wsi_channel_region_shape(path, channel.index, level, x, y, width, height)
    return shape, meta, f"channel_{channel.index}_{channel.name}"


def _wsi_region_shape(
    path: Path,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int], dict[str, float]]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        level_group = _select_openslide_level_group(slide, level)
        read_level = level_group[0]
        downsample = float(slide.level_downsamples[read_level])
        level_w, level_h = slide.level_dimensions[read_level]
        if width <= 0 or height <= 0:
            shape = (int(level_h), int(level_w))
            origin_x = 0
            origin_y = 0
        else:
            shape = (
                max(1, int(np.ceil(height / downsample))),
                max(1, int(np.ceil(width / downsample))),
            )
            origin_x = int(max(x, 0))
            origin_y = int(max(y, 0))
        meta = _slide_coordinate_metadata(slide, origin_x, origin_y, downsample)
    finally:
        slide.close()
    return shape, meta


def _wsi_channel_region_shape(
    path: Path,
    channel_index: int,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int], dict[str, float]]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        level_group = _select_openslide_level_group(slide, level)
        channel_level = level_group[int(np.clip(channel_index, 0, len(level_group) - 1))]
        downsample = float(slide.level_downsamples[channel_level])
        level_w, level_h = slide.level_dimensions[channel_level]
        if width <= 0 or height <= 0:
            shape = (int(level_h), int(level_w))
            origin_x = 0
            origin_y = 0
        else:
            shape = (
                max(1, int(np.ceil(height / downsample))),
                max(1, int(np.ceil(width / downsample))),
            )
            origin_x = int(max(x, 0))
            origin_y = int(max(y, 0))
        meta = _slide_coordinate_metadata(slide, origin_x, origin_y, downsample)
    finally:
        slide.close()
    return shape, meta


def _read_cellpose_source(
    path: Path,
    channel: WSIChannel | None,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float], str]:
    if channel is None:
        rgb, meta = _read_wsi_region(path, level, x, y, width, height)
        return _image_to_cellpose_channel(rgb), meta, "grayscale"
    channel_img, meta = _read_wsi_channel_region(path, channel.index, level, x, y, width, height)
    return channel_img, meta, f"channel_{channel.index}_{channel.name}"


def _segment_cellpose_source_to_target(
    path: Path,
    channel: WSIChannel | None,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
    out_root: Path,
    label: str,
    use_gpu: bool,
    chunk_size: int,
    chunk_overlap: int,
    max_tiles: int,
    target_meta: dict[str, float],
    target_shape: tuple[int, int],
    export_source_image: bool,
) -> np.ndarray:
    source_shape, source_meta, source_label = _cellpose_source_info(path, channel, level, x, y, width, height)
    source_h, source_w = source_shape
    tile_count = _estimate_tile_count(source_h, source_w, chunk_size, chunk_overlap)
    show_info(
        f"Cellpose source for {label}: {source_label}, level={int(level)}, "
        f"shape={source_h}x{source_w}, downsample={source_meta['downsample']:g}, "
        f"{source_h * source_w / 1e6:.1f} MP, ~{tile_count} tiles."
    )
    if int(max_tiles) > 0 and tile_count > int(max_tiles):
        raise RuntimeError(
            f"Cellpose source needs ~{tile_count} tiles, above max_cellpose_tiles={int(max_tiles)}. "
            "Use a smaller ROI, increase cellpose_read_level, or set max_cellpose_tiles=0 to run anyway."
        )
    if source_h * source_w <= 120_000_000:
        source, source_meta, source_label = _read_cellpose_source(path, channel, level, x, y, width, height)
        _export_cellpose_source(out_root, path, source_label, level, source, export_source_image)
        mask_raw = _segment_image_with_cellpose(
            source,
            label,
            use_gpu,
            chunk_size,
            chunk_overlap,
        )
        return _project_mask_to_target(mask_raw, source_meta, target_meta, target_shape)
    if export_source_image:
        show_info(
            "Skipping full Cellpose source export because the selected source is too large; "
            "set a smaller ROI if you need to inspect the exact full-resolution DAPI image."
        )
    return _segment_cellpose_source_streamed(
        path,
        channel,
        int(level),
        int(x),
        int(y),
        int(width),
        int(height),
        source_shape,
        source_meta,
        label,
        use_gpu,
        int(chunk_size),
        int(chunk_overlap),
        target_shape,
    )


def _export_cellpose_source(
    out_root: Path,
    wsi_path: Path,
    source_label: str,
    level: int,
    image: np.ndarray,
    enabled: bool,
) -> None:
    if not enabled:
        return
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("tifffile is required to export the Cellpose source image.") from exc

    source_dir = out_root / "cellpose_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_filename(source_label)
    out_path = source_dir / f"{wsi_path.stem}_{safe_label}_level{int(level)}.tif"
    tifffile.imwrite(out_path, np.asarray(image), bigtiff=True, photometric="minisblack")
    show_info(f"Exported Cellpose source image: {out_path}")


def _segment_cellpose_source_streamed(
    path: Path,
    channel: WSIChannel | None,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
    source_shape: tuple[int, int],
    source_meta: dict[str, float],
    label: str,
    use_gpu: bool,
    chunk_size: int,
    chunk_overlap: int,
    target_shape: tuple[int, int],
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("cellpose_chunk_size must be > 0.")
    if chunk_overlap < 0:
        raise ValueError("cellpose_chunk_overlap must be >= 0.")
    if chunk_overlap * 2 >= chunk_size:
        raise ValueError("cellpose_chunk_overlap * 2 must be smaller than cellpose_chunk_size.")

    config = CellposeConfig(
        gpu=use_gpu,
        pretrained_model="cpsam",
        diameter=8,
        flow_threshold=-2,
        cellprob_threshold=-2,
        min_size=1,
    )
    segmenter = CellposeSegmenter(config)
    source_h, source_w = source_shape
    source_downsample = float(source_meta["downsample"])
    source_origin_x = float(source_meta["origin_x"])
    source_origin_y = float(source_meta["origin_y"])
    target_h, target_w = target_shape
    target_mask = np.zeros((target_h, target_w), dtype=np.int32)
    next_label = 1
    skipped_blank = 0
    no_mask = 0
    tiles = list(_iter_stream_tiles(source_h, source_w, chunk_size, chunk_overlap))
    start_time = time.perf_counter()
    for tile_index, tile in enumerate(tiles, start=1):
        read_x0, read_y0, read_x1, read_y1, core_x0, core_y0, core_x1, core_y1 = tile
        read_level_x = int(round(source_origin_x + read_x0 * source_downsample))
        read_level_y = int(round(source_origin_y + read_y0 * source_downsample))
        read_level_w = int(round((read_x1 - read_x0) * source_downsample))
        read_level_h = int(round((read_y1 - read_y0) * source_downsample))
        if channel is None:
            tile_img, _ = _read_wsi_region(
                path,
                level,
                read_level_x,
                read_level_y,
                read_level_w,
                read_level_h,
            )
            tile_img = _image_to_cellpose_channel(tile_img)
        else:
            tile_img, _ = _read_wsi_channel_region(
                path,
                channel.index,
                level,
                read_level_x,
                read_level_y,
                read_level_w,
                read_level_h,
            )
        if _is_blank_cellpose_tile(tile_img):
            skipped_blank += 1
            _show_cellpose_progress(tile_index, len(tiles), label, start_time, skipped_blank, no_mask)
            continue
        local_mask, _, _ = segmenter.segment_array(tile_img)
        local_mask = np.asarray(local_mask, dtype=np.int32)
        local_core = local_mask[
            core_y0 - read_y0: core_y1 - read_y0,
            core_x0 - read_x0: core_x1 - read_x0,
        ]
        if local_core.size == 0 or int(local_core.max()) == 0:
            no_mask += 1
            _show_cellpose_progress(tile_index, len(tiles), label, start_time, skipped_blank, no_mask)
            continue
        target_y0 = int(round(core_y0 * target_h / source_h))
        target_y1 = int(round(core_y1 * target_h / source_h))
        target_x0 = int(round(core_x0 * target_w / source_w))
        target_x1 = int(round(core_x1 * target_w / source_w))
        if target_y1 <= target_y0 or target_x1 <= target_x0:
            continue
        local_core = np.where(local_core > 0, local_core + next_label - 1, 0)
        next_label = int(local_core.max()) + 1
        target_core = resize(
            local_core,
            (target_y1 - target_y0, target_x1 - target_x0),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int32)
        target_mask[target_y0:target_y1, target_x0:target_x1] = target_core
        _show_cellpose_progress(tile_index, len(tiles), label, start_time, skipped_blank, no_mask)
    show_info(
        f"{label} streamed Cellpose-SAM mask loaded: "
        f"{int(target_mask.max())} cells, skipped_blank={skipped_blank}, no_mask={no_mask}"
    )
    return target_mask


def _show_cellpose_progress(
    tile_index: int,
    tile_count: int,
    label: str,
    start_time: float,
    skipped_blank: int,
    no_mask: int,
) -> None:
    if tile_index != 1 and tile_index != tile_count and tile_index % 5 != 0:
        return
    elapsed = max(0.001, time.perf_counter() - start_time)
    sec_per_tile = elapsed / max(1, tile_index)
    remaining = sec_per_tile * max(0, tile_count - tile_index)
    show_info(
        f"  Cellpose streamed tiles: {tile_index}/{tile_count} for {label}, "
        f"elapsed={elapsed / 60:.1f} min, eta={remaining / 60:.1f} min, "
        f"skipped_blank={skipped_blank}, no_mask={no_mask}"
    )


def _is_blank_cellpose_tile(tile: np.ndarray) -> bool:
    if tile.size == 0:
        return True
    tile = np.asarray(tile)
    nonzero_fraction = float(np.count_nonzero(tile)) / float(tile.size)
    if nonzero_fraction < 0.001:
        return True
    return float(np.percentile(tile, 99.0)) <= 5.0


def _iter_stream_tiles(
    height: int,
    width: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    step = max(1, int(chunk_size) - 2 * max(0, int(overlap)))
    tiles = []
    for core_y0 in range(0, int(height), step):
        core_y1 = min(int(height), core_y0 + step)
        read_y0 = max(0, core_y0 - int(overlap))
        read_y1 = min(int(height), core_y1 + int(overlap))
        for core_x0 in range(0, int(width), step):
            core_x1 = min(int(width), core_x0 + step)
            read_x0 = max(0, core_x0 - int(overlap))
            read_x1 = min(int(width), core_x1 + int(overlap))
            tiles.append((read_x0, read_y0, read_x1, read_y1, core_x0, core_y0, core_x1, core_y1))
    return tiles


def _project_mask_to_target(
    mask: np.ndarray,
    source_meta: dict[str, float],
    target_meta: dict[str, float],
    target_shape: tuple[int, int],
) -> np.ndarray:
    if mask.shape == target_shape:
        return np.asarray(mask, dtype=np.int32)
    source_downsample = float(source_meta["downsample"])
    target_downsample = float(target_meta["downsample"])
    if not np.isclose(source_downsample, target_downsample):
        show_info(
            "Projecting Cellpose mask from "
            f"downsample {source_downsample:g} to preview downsample {target_downsample:g}."
        )
    projected = resize(
        np.asarray(mask),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return np.asarray(projected, dtype=np.int32)


def _segment_image_with_cellpose(
    image: np.ndarray,
    label: str,
    use_gpu: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Cellpose WSI segmentation expects a 2D channel image, got {image.shape}.")
    height, width = image.shape[:2]
    tile_count = _estimate_tile_count(height, width, chunk_size, chunk_overlap)
    show_info(
        f"Segmenting {label} with Cellpose-SAM "
        f"({height}x{width}, {height * width / 1e6:.1f} MP, ~{tile_count} tiles)"
    )
    config = CellposeConfig(
        gpu=use_gpu,
        pretrained_model="cpsam",
        diameter=8,
        flow_threshold=-2,
        cellprob_threshold=-2,
        min_size=1,
    )
    segmenter = CellposeSegmenter(config)
    mask, _, _ = segmenter.segment_array_chunked(
        image,
        chunk_size=int(chunk_size),
        overlap=int(chunk_overlap),
        stitch_labels=True,
    )
    show_info(f"{label} Cellpose-SAM mask loaded: {int(mask.max())} cells")
    return np.asarray(mask, dtype=np.int32)


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")
    return safe or "image"


def _estimate_tile_count(height: int, width: int, chunk_size: int, overlap: int) -> int:
    step = max(1, int(chunk_size) - 2 * max(0, int(overlap)))
    rows = int(np.ceil(max(1, height) / step))
    cols = int(np.ceil(max(1, width) / step))
    return rows * cols


def _match_channel(channels: list[WSIChannel], requested: str) -> WSIChannel | None:
    requested_key = str(requested).strip().lower()
    if requested_key in {"", "auto", "auto dapi"}:
        requested_key = "dapi"
    if ":" in requested_key:
        index_text, _, name_text = requested_key.partition(":")
        try:
            index = int(index_text.strip())
            matched = next((channel for channel in channels if channel.index == index), None)
            if matched is not None:
                return matched
        except ValueError:
            requested_key = name_text.strip()
    for channel in channels:
        if channel.name.strip().lower() == requested_key:
            return channel
    for channel in channels:
        if requested_key in channel.name.strip().lower():
            return channel
    try:
        index = int(requested_key)
    except ValueError:
        return None
    return next((channel for channel in channels if channel.index == index), None)


def _moving_cellvit_dir_name(channel: WSIChannel | None) -> str:
    if channel is None:
        return "moving_cellvit"
    safe_name = "".join(char if char.isalnum() else "_" for char in channel.name).strip("_")
    return f"moving_cellvit_channel_{channel.index}_{safe_name or 'channel'}"


def _read_wsi_channel_region(
    path: Path,
    channel_index: int,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float]]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        level_group = _select_openslide_level_group(slide, level)
        channel_level = level_group[int(np.clip(channel_index, 0, len(level_group) - 1))]
        downsample = float(slide.level_downsamples[channel_level])
        level_w, level_h = slide.level_dimensions[channel_level]
        if width <= 0 or height <= 0:
            location = (0, 0)
            size = (int(level_w), int(level_h))
            origin_x = 0
            origin_y = 0
        else:
            origin_x = int(max(x, 0))
            origin_y = int(max(y, 0))
            location = (origin_x, origin_y)
            size = (
                max(1, int(np.ceil(width / downsample))),
                max(1, int(np.ceil(height / downsample))),
            )
        rgb = np.asarray(slide.read_region(location, channel_level, size).convert("RGB"))
        meta = _slide_coordinate_metadata(slide, origin_x, origin_y, downsample)
    finally:
        slide.close()
    return _stretch_channel(rgb[..., 0]), {
        **meta,
    }


def _napari_colormap(channel: WSIChannel) -> str:
    name = channel.name.lower()
    if "dapi" in name:
        return "blue"
    if channel.color is None:
        return "gray"
    r, g, b = channel.color
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    if b >= r and b >= g:
        return "blue"
    return "gray"


def _select_openslide_level_group(slide: Any, requested_level: int) -> list[int]:
    groups: list[list[int]] = []
    for idx, dimensions in enumerate(slide.level_dimensions):
        if groups and slide.level_dimensions[groups[-1][0]] == dimensions:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    group_index = int(np.clip(requested_level, 0, len(groups) - 1))
    return groups[group_index]


def _read_multichannel_wsi_composite(
    slide: Any,
    levels: list[int],
    location: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
    colors = np.asarray(
        [
            (0.1, 0.2, 1.0),
            (0.0, 1.0, 0.2),
            (1.0, 0.1, 0.1),
            (1.0, 0.0, 1.0),
            (1.0, 0.8, 0.0),
            (0.0, 0.9, 1.0),
            (1.0, 0.45, 0.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=np.float32,
    )
    composite = np.zeros((size[1], size[0], 3), dtype=np.uint16)
    for channel_index, level in enumerate(levels[: len(colors)]):
        channel_rgb = np.asarray(slide.read_region(location, level, size).convert("RGB"))
        channel = channel_rgb[..., 0]
        stretched = _stretch_channel(channel).astype(np.uint16)
        color = colors[channel_index]
        for axis in range(3):
            composite[..., axis] += (stretched * color[axis]).astype(np.uint16)
    return np.clip(composite, 0, 255).astype(np.uint8)


def _auto_contrast_grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        return rgb
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
        return rgb
    if float(np.percentile(rgb[..., 0], 99.0)) >= 20.0:
        return rgb
    stretched = _stretch_channel(rgb[..., 0])
    return np.repeat(stretched[..., None], 3, axis=2)


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    values = channel[channel > 0]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.8))
    if high <= low:
        high = float(values.max())
    if high <= low:
        return np.zeros_like(channel, dtype=np.uint8)
    stretched = (channel.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _run_cellvit(
    wsi_path: Path,
    cellvit_entry: str,
    model_path: str,
    outdir: Path,
    use_cell_shapes: bool,
    binary_cell_segmentation: bool,
    use_gpu: bool,
    resolution: float,
    batch_size: int,
    redownload_model: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    expected_output = "cells.json" if use_cell_shapes else "cell_detection.json"
    try:
        _find_cellvit_output(outdir / wsi_path.stem, outdir, expected_output)
        show_info(f"Using existing CellViT++ output for {wsi_path.name}.")
        return
    except FileNotFoundError:
        pass

    cmd, cwd, cli_style = _cellvit_command(cellvit_entry)
    cache_file = _cellvit_cache_file(model_path) if cli_style == "pypi" else None
    if redownload_model and cache_file is not None and cache_file.exists():
        cache_file.unlink()
    if cli_style == "pypi":
        cmd += [
            "--model", str(model_path).upper(),
            "--outdir", str(outdir),
            "--gpu", "0" if use_gpu else "-1",
            "--batch_size", str(int(batch_size)),
        ]
        if binary_cell_segmentation:
            cmd += ["--nuclei_taxonomy", "binary"]
        if use_cell_shapes:
            cmd.append("--geojson")
        cmd += [
            "process_wsi",
            "--wsi_path", str(wsi_path),
            "--wsi_mpp", str(float(resolution)),
            "--wsi_magnification", _magnification_from_resolution(resolution),
        ]
    else:
        cmd += [
            "--model", str(model_path),
            "--outdir", str(outdir),
            "--gpu", "0" if use_gpu else "-1",
            "--resolution", str(float(resolution)),
            "--batch_size", str(int(batch_size)),
        ]
        if binary_cell_segmentation:
            cmd.append("--binary")
        if use_cell_shapes:
            cmd.append("--geojson")
        cmd += [
            "process_wsi",
            "--wsi_path", str(wsi_path),
            "--wsi_mpp", str(float(resolution)),
            "--wsi_magnification", _magnification_from_resolution(resolution),
        ]

    show_info(f"Running CellViT++ on {wsi_path.name}...")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    stdout_log = outdir / "cellvit_stdout.log"
    stderr_log = outdir / "cellvit_stderr.log"
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        text = result.stderr or result.stdout
        hint = _cellvit_failure_hint(text, cache_file)
        raise RuntimeError(
            f"CellViT++ failed for {wsi_path.name}. See log: {stderr_log}.{hint}"
        )


def _cellvit_command(entry: str) -> tuple[list[str], str | None, str]:
    entry_path = Path(entry) if entry else None
    if entry_path and not entry_path.exists():
        raise FileNotFoundError(f"CellViT++ entry path does not exist: {entry_path}")
    if entry_path and entry_path.is_dir():
        script = entry_path / "cellvit" / "detect_cells.py"
        if not script.exists():
            raise FileNotFoundError(f"CellViT++ detect_cells.py was not found: {script}")
        return [sys.executable, str(script)], str(entry_path), "repo"
    if entry_path and entry_path.is_file():
        return [sys.executable, str(entry_path)], str(entry_path.parent), "repo"
    return [sys.executable, "-m", "napari_cell_registration._cellvit_runner"], None, "pypi"


def _cellvit_cache_file(model_path: str) -> Path | None:
    model_name = str(model_path).upper()
    if model_name == "SAM":
        filename = "CellViT-SAM-H-x40-AMP.pth"
    elif model_name == "HIPT":
        filename = "CellViT-256-x40-AMP.pth"
    else:
        return None
    cache_dir = Path(os.getenv("CELLVIT_CACHE", str(Path.home() / ".cache" / "cellvit")))
    return cache_dir / filename


def _magnification_from_resolution(resolution: float) -> str:
    return "40" if float(resolution) == 0.25 else "20"


def _cellvit_failure_hint(stderr_or_stdout: str, cache_file: Path | None) -> str:
    if (
        "PytorchStreamReader failed reading zip archive" not in stderr_or_stdout
        and "failed finding central directory" not in stderr_or_stdout
    ):
        return ""
    cache_hint = f" Cached model: {cache_file}." if cache_file is not None else ""
    return (
        " The CellViT++ cached model checkpoint looks incomplete or corrupt."
        f"{cache_hint} Enable 'redownload_cellvit_model' once and run again."
    )


def wsi_segmentation_factory():
    widget = magicgui(
        wsi_segmentation_widget,
        call_button="Run WSI segmentation",
        fixed_wsi_path={
            "widget_type": "FileEdit",
            "mode": "r",
            "label": "Fixed WSI",
        },
        moving_wsi_path={
            "widget_type": "FileEdit",
            "mode": "r",
            "label": "Moving WSI",
        },
        moving_segmentation_channel={
            "widget_type": "ComboBox",
            "choices": ["Auto DAPI"],
            "label": "Moving segmentation channel",
        },
    )

    def _update_moving_channel_choices(*_args):
        path = Path(widget.moving_wsi_path.value)
        channels = _read_wsi_channels(path) if path.is_file() else []
        choices = ["Auto DAPI"] + [f"{channel.index}: {channel.name}" for channel in channels]
        previous = str(widget.moving_segmentation_channel.value)
        widget.moving_segmentation_channel.choices = choices
        if previous in choices:
            widget.moving_segmentation_channel.value = previous
        else:
            dapi_choice = next((choice for choice in choices if "dapi" in choice.lower()), choices[0])
            widget.moving_segmentation_channel.value = dapi_choice

    widget.moving_wsi_path.changed.connect(_update_moving_channel_choices)
    _update_moving_channel_choices()
    return widget


def _cellvit_output_to_mask(
    wsi_path: Path,
    outdir: Path,
    shape: tuple[int, int],
    meta: dict[str, float],
    use_cell_shapes: bool,
    point_radius_px: int,
) -> np.ndarray:
    result_dir = outdir / wsi_path.stem
    if use_cell_shapes:
        payload = _load_cellvit_json(_find_cellvit_output(result_dir, outdir, "cells.json"))
        return _contours_to_mask(payload.get("cells", []), shape, meta)
    payload = _load_cellvit_json(_find_cellvit_output(result_dir, outdir, "cell_detection.json"))
    return _centroids_to_mask(payload.get("cells", []), shape, meta, point_radius_px)


def _find_cellvit_output(result_dir: Path, outdir: Path, filename: str) -> Path:
    candidates = [
        result_dir / filename,
        outdir / filename,
        outdir / f"{result_dir.name}_{filename}",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(outdir.glob(f"*/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Expected CellViT++ output was not found: {result_dir / filename}. "
        "If CellViT++ logs say 'No patches sampled' or 'No cells have been extracted', "
        "the WSI produced no usable CellViT patches."
    )


def _load_cellvit_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected CellViT++ output was not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _contours_to_mask(cells: list[dict[str, Any]], shape: tuple[int, int], meta: dict[str, float]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int32)
    origin = np.array([meta["origin_x"], meta["origin_y"]], dtype=float)
    downsample = float(meta["downsample"])
    for label, cell in enumerate(cells, start=1):
        contour = np.asarray(cell.get("contour", []), dtype=float)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        pts = (contour - origin[None, :]) / downsample
        rr, cc = polygon(pts[:, 1], pts[:, 0], shape=shape)
        mask[rr, cc] = label
    return mask


def _centroids_to_mask(
    cells: list[dict[str, Any]],
    shape: tuple[int, int],
    meta: dict[str, float],
    radius_px: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int32)
    origin = np.array([meta["origin_x"], meta["origin_y"]], dtype=float)
    downsample = float(meta["downsample"])
    radius = max(1, int(radius_px))
    for label, cell in enumerate(cells, start=1):
        centroid = np.asarray(cell.get("centroid", []), dtype=float)
        if centroid.shape != (2,):
            continue
        x, y = (centroid - origin) / downsample
        if x < 0 or y < 0 or x >= shape[1] or y >= shape[0]:
            continue
        rr, cc = disk((float(y), float(x)), radius=radius, shape=shape)
        mask[rr, cc] = label
    return mask

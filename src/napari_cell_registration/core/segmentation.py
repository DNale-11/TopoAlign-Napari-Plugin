"""Cellpose-SAM segmentation wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import CellposeConfig
from .io_utils import load_image, save_mask

try:
    from cellpose import models
except ImportError as exc:
    raise ImportError(
        "cellpose is required for segmentation. Install with `pip install cellpose` "
        "and ensure Cellpose-SAM weights are available."
    ) from exc


class CellposeSegmenter:
    """Segment images using Cellpose-SAM."""

    def __init__(self, config: CellposeConfig):
        self.config = config
        model_kwargs = {"gpu": config.gpu}
        # Prefer explicit built-in/custom checkpoint name; model_type is only for older cellpose.
        if config.pretrained_model:
            model_kwargs["pretrained_model"] = config.pretrained_model
        if config.model_type:
            model_kwargs["model_type"] = config.model_type
        self.model = models.CellposeModel(**model_kwargs)

    def _select_channel(self, img: np.ndarray) -> np.ndarray:
        """Select a single channel for segmentation (assumes DAPI is last if multiple)."""
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            return img[..., -1]
        raise ValueError(f"Unsupported image shape {img.shape}; expected 2D or 3D.")

    def _select_channel_zstack(self, img: np.ndarray) -> np.ndarray:
        """Select the DAPI channel for Z-stacks (Z, Y, X[, C])."""
        if img.ndim == 3:
            return img  # single-channel Z-stack
        if img.ndim == 4:
            return img[..., -1]  # assume DAPI is last channel
        raise ValueError(f"Unsupported Z-stack shape {img.shape}; expected (Z, Y, X) or (Z, Y, X, C).")

    def spatial_shape(self, img: np.ndarray) -> tuple[int, int]:
        """Return the 2D segmentation shape after channel selection."""
        channel_img = self._select_channel(img)
        if channel_img.ndim != 2:
            raise ValueError(f"Chunked segmentation expects a 2D image after channel selection, got {channel_img.shape}.")
        return int(channel_img.shape[0]), int(channel_img.shape[1])

    def segment_array(self, img: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
        """
        Segment a numpy array and return masks along with Cellpose outputs.

        Returns
        -------
        Tuple[np.ndarray, dict, np.ndarray]
            (masks, flows, styles)
        """
        channel_img = self._select_channel(img)
        # cellpose 4.x returns (masks, flows, styles); older versions returned 4 items.
        eval_kwargs = dict(
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
        )
        try:
            result = self.model.eval(channel_img, **eval_kwargs)
        except TypeError:
            result = self.model.eval(channel_img, **eval_kwargs, channels=[0, 0])
        if len(result) == 4:
            masks, flows, styles, _ = result
        else:
            masks, flows, styles = result
        return masks, flows, styles

    def segment_array_chunked(
        self,
        img: np.ndarray,
        chunk_size: int = 2048,
        overlap: int = 128,
        stitch_labels: bool = True,
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        """
        Segment a large 2D image in overlapping chunks and stitch labels.

        The full image is only sliced chunk-by-chunk for Cellpose inference. A full-size
        label image is still created because napari needs one labels layer to display.
        """
        channel_img = self._select_channel(img)
        if channel_img.ndim != 2:
            raise ValueError(f"Chunked segmentation expects a 2D image after channel selection, got {channel_img.shape}.")

        chunk_size = int(chunk_size)
        overlap = int(overlap)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        if overlap < 0:
            raise ValueError("overlap must be >= 0.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        height, width = channel_img.shape
        stitched = np.zeros((height, width), dtype=np.int64)
        unions = _UnionFind()
        next_label = 1

        for tile in _iter_tiles(height, width, chunk_size, overlap):
            crop = np.asarray(channel_img[tile.y0:tile.y1, tile.x0:tile.x1])
            local_mask, _, _ = self.segment_array(crop)
            local_mask = np.asarray(local_mask)
            if local_mask.shape != crop.shape:
                raise ValueError(
                    f"Cellpose returned mask shape {local_mask.shape}, expected crop shape {crop.shape}."
                )
            if local_mask.max() == 0:
                continue

            global_mask = local_mask.astype(np.int64, copy=True)
            foreground = global_mask > 0
            global_mask[foreground] += next_label - 1
            max_global = int(global_mask.max())
            next_label = max_global + 1

            target_crop = stitched[tile.y0:tile.y1, tile.x0:tile.x1]
            both = (target_crop > 0) & (global_mask > 0)
            if stitch_labels and np.any(both):
                pairs = np.column_stack((target_crop[both], global_mask[both]))
                for existing_label, new_label in np.unique(pairs, axis=0):
                    unions.union(int(existing_label), int(new_label))

            core_y0 = tile.core_y0 - tile.y0
            core_y1 = tile.core_y1 - tile.y0
            core_x0 = tile.core_x0 - tile.x0
            core_x1 = tile.core_x1 - tile.x0
            core_mask = global_mask[core_y0:core_y1, core_x0:core_x1]
            target_core = stitched[tile.core_y0:tile.core_y1, tile.core_x0:tile.core_x1]
            fill = (target_core == 0) & (core_mask > 0)
            target_core[fill] = core_mask[fill]

        masks = _relabel_from_unions(stitched, unions)
        return masks, {}, np.array([])

    def segment_zstack(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
        """
        Segment a 3D Z-stack image and return both 3D and projected 2D masks.

        Parameters
        ----------
        img : np.ndarray
            Array of shape (Z, Y, X) or (Z, Y, X, C).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict, np.ndarray]
            (mask_3d, mask_2d, flows, styles) where mask_2d is a max projection of mask_3d.
        """
        channel_img = self._select_channel_zstack(img)
        eval_kwargs = dict(
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
            channels=[0, 0],
            do_3D=True,
            # Explicit axes for 3D: z is axis 0, channel_axis None for ZYX or last for ZYXC
            z_axis=0,
        )
        if channel_img.ndim == 4:
            eval_kwargs["channel_axis"] = -1
        else:
            eval_kwargs["channel_axis"] = None
        if self.config.anisotropy is not None:
            eval_kwargs["anisotropy"] = self.config.anisotropy
        result = self.model.eval(channel_img, **eval_kwargs)
        if len(result) == 4:
            masks_3d, flows, styles, _ = result
        else:
            masks_3d, flows, styles = result
        masks_2d = project_labels_max(masks_3d)
        return masks_3d, masks_2d, flows, styles

    def segment_file(
        self, path: str | Path, save_mask_path: Optional[str | Path] = None
    ) -> np.ndarray:
        """
        Segment an image file and optionally save the resulting mask.

        Parameters
        ----------
        path : str or Path
            Image path.
        save_mask_path : str or Path, optional
            If provided, save the mask to this path.

        Returns
        -------
        np.ndarray
            Label mask.
        """
        img = load_image(path)
        masks, _, _ = self.segment_array(img)
        if save_mask_path is not None:
            save_mask(save_mask_path, masks)
        return masks


@dataclass(frozen=True)
class _Tile:
    y0: int
    y1: int
    x0: int
    x1: int
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int


class _UnionFind:
    """Small disjoint-set helper for merging labels seen in overlapping chunks."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def _iter_tiles(height: int, width: int, chunk_size: int, overlap: int) -> list[_Tile]:
    """Yield overlapping crop windows covering a 2D image."""
    tiles: list[_Tile] = []
    for core_y0 in range(0, height, chunk_size):
        core_y1 = min(core_y0 + chunk_size, height)
        y0 = max(0, core_y0 - overlap)
        y1 = min(height, core_y1 + overlap)
        for core_x0 in range(0, width, chunk_size):
            core_x1 = min(core_x0 + chunk_size, width)
            x0 = max(0, core_x0 - overlap)
            x1 = min(width, core_x1 + overlap)
            tiles.append(
                _Tile(
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    core_y0=core_y0,
                    core_y1=core_y1,
                    core_x0=core_x0,
                    core_x1=core_x1,
                )
            )
    return tiles


def _relabel_from_unions(mask: np.ndarray, unions: _UnionFind) -> np.ndarray:
    """Apply union-find label merges and compact labels to 1..N."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.zeros(mask.shape, dtype=np.int32)

    roots = np.array([unions.find(int(label)) for label in labels], dtype=np.int64)
    root_to_new: dict[int, int] = {}
    new_labels = np.zeros(len(labels), dtype=np.int32)
    for idx, root in enumerate(roots):
        root_int = int(root)
        if root_int not in root_to_new:
            root_to_new[root_int] = len(root_to_new) + 1
        new_labels[idx] = root_to_new[root_int]

    out = np.zeros(mask.shape, dtype=np.int32)
    foreground = mask > 0
    out[foreground] = new_labels[np.searchsorted(labels, mask[foreground])]
    return out


def project_labels_max(mask_3d: np.ndarray) -> np.ndarray:
    """
    Project a 3D label volume (Z, Y, X) into a 2D label image (Y, X) using max across Z.

    If multiple labels overlap along Z at the same (Y, X), the highest label id is kept.
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected a 3D mask to project, got shape {mask_3d.shape}.")
    return np.max(mask_3d, axis=0)

"""Visualization helpers for segmentation and matching."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage import draw
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from scipy.ndimage import binary_dilation


def _select_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[..., -1]
    raise ValueError(f"Unsupported image shape {img.shape}; expected 2D or 3D.")


def _normalize_for_overlay(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    if p99 - p1 == 0:
        return np.zeros_like(img, dtype=np.float32)
    norm = (img - p1) / (p99 - p1)
    return np.clip(norm, 0, 1)


def _pad_to_height(arr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == target_h:
        return arr
    pad_before = (target_h - h) // 2
    pad_after = target_h - h - pad_before
    pad_width = [(pad_before, pad_after), (0, 0)]
    if arr.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(arr, pad_width, mode="constant")


def _color_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.2, 1.0, size=(n, 3))
    return colors


def launch_napari_viewer(
    image: np.ndarray,
    mask: np.ndarray,
    features: Optional[pd.DataFrame] = None,
    title: str = "segmentation",
):
    """
    Launch a napari viewer to inspect segmentation results.
    """
    try:
        import napari  # type: ignore
    except ImportError as exc:
        raise ImportError("napari is required for interactive visualization.") from exc

    viewer = napari.Viewer(title=title)
    viewer.add_image(image, name=f"{title}-image", blending="additive")
    viewer.add_labels(mask, name=f"{title}-mask", opacity=0.5)

    if features is not None and not features.empty:
        pts = features[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts, name=f"{title}-centroids", size=6, face_color="cyan")

    return viewer


def export_match_table(match_df: pd.DataFrame, path: str | Path) -> None:
    """Save match table to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(path, index=False)


def save_match_overlay(
    image1: np.ndarray,
    image2: np.ndarray,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    match_df: pd.DataFrame,
    path: str | Path,
    marker_radius: int = 4,
) -> None:
    """
    Create a side-by-side overlay of matched centroids and save as an RGB image.

    Lines connect matched cells between image1 (left) and image2 (right).
    """
    img1 = _normalize_for_overlay(_select_channel(image1))
    img2 = _normalize_for_overlay(_select_channel(image2))

    h = max(img1.shape[0], img2.shape[0])
    img1 = _pad_to_height(img1, h)
    img2 = _pad_to_height(img2, h)

    rgb1 = np.stack([img1] * 3, axis=-1)
    rgb2 = np.stack([img2] * 3, axis=-1)

    combined = np.concatenate([rgb1, rgb2], axis=1)
    offset_x = rgb1.shape[1]

    colors = _color_palette(len(match_df))

    for color, (_, row) in zip(colors, match_df.iterrows()):
        y1, x1 = feats1.loc[row["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
        y2, x2 = feats2.loc[row["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
        x2_shift = x2 + offset_x

        rr1, cc1 = draw.disk((y1, x1), radius=marker_radius, shape=combined.shape[:2])
        rr2, cc2 = draw.disk((y2, x2_shift), radius=marker_radius, shape=combined.shape[:2])
        combined[rr1, cc1] = color
        combined[rr2, cc2] = color

        rrl, ccl = draw.line(int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2_shift)))
        combined[rrl, ccl] = color

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(Path(path), (np.clip(combined, 0, 1) * 255).astype(np.uint8))


def save_segmentation_plot(
    image: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
    title: str = "Segmentation",
) -> None:
    """
    Save a matplotlib figure showing grayscale image with mask boundaries.
    """
    img = _normalize_for_overlay(_select_channel(image))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.contour(mask, colors="lime", linewidths=0.6)
    ax.set_title(title)
    ax.axis("off")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_registration_overlay(
    image1: np.ndarray,
    mask1: np.ndarray,
    image2: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    path: str | Path,
    alpha: float = 0.4,
) -> None:
    """
    Warp mask1 (Image1 space) into Image2 space using rigid transform, then overlay on Image2.
    """
    warped_mask = warp_mask_to_image2(mask1, image2.shape[:2], rotation, translation)

    base = _normalize_for_overlay(_select_channel(image2))
    rgb = np.stack([base, base, base], axis=-1)

    mask_area = warped_mask > 0
    edges = np.logical_xor(mask_area, binary_dilation(mask_area, structure=np.ones((3, 3))))
    # Fill mask with red, edges with yellow for visibility
    rgb[mask_area] = (1 - alpha) * rgb[mask_area] + alpha * np.array([1, 0, 0])
    rgb[edges] = (1 - alpha) * rgb[edges] + alpha * np.array([1, 1, 0])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(Path(path), (np.clip(rgb, 0, 1) * 65535).astype(np.uint16))


def warp_mask_to_image2(
    mask1: np.ndarray,
    image2_shape: tuple[int, int],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Warp mask1 into image2 coordinates using the estimated rigid transform.
    """
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    R_inv = R.T  # orthonormal assumption
    t_inv = -R_inv @ t

    affine = AffineTransform(
        matrix=np.array(
            [
                [R_inv[0, 0], R_inv[0, 1], t_inv[0]],
                [R_inv[1, 0], R_inv[1, 1], t_inv[1]],
                [0, 0, 1],
            ],
            dtype=float,
        )
    )

    warped_mask = warp(
        mask1.astype(float),
        inverse_map=affine,
        order=0,
        preserve_range=True,
        output_shape=image2_shape,
    )
    return warped_mask


def save_match_plot(
    image1: np.ndarray,
    image2: np.ndarray,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    match_df: pd.DataFrame,
    path: str | Path,
    marker_size: int = 6,
) -> None:
    """
    Save a matplotlib figure with matched centroids and connecting lines.
    """
    img1 = _normalize_for_overlay(_select_channel(image1))
    img2 = _normalize_for_overlay(_select_channel(image2))
    h = max(img1.shape[0], img2.shape[0])
    img1 = _pad_to_height(img1, h)
    img2 = _pad_to_height(img2, h)

    rgb1 = np.stack([img1] * 3, axis=-1)
    rgb2 = np.stack([img2] * 3, axis=-1)
    combined = np.concatenate([rgb1, rgb2], axis=1)
    offset_x = rgb1.shape[1]

    colors = _color_palette(len(match_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined)

    for color, (_, row) in zip(colors, match_df.iterrows()):
        y1, x1 = feats1.loc[row["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
        y2, x2 = feats2.loc[row["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
        x2_shift = x2 + offset_x
        ax.plot([x1, x2_shift], [y1, y2], color=color, linewidth=0.8, alpha=0.8)
        ax.scatter([x1, x2_shift], [y1, y2], c=[color], s=marker_size * 5, edgecolors="k", linewidths=0.4)

    ax.axis("off")
    ax.set_title("Matched centroids")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

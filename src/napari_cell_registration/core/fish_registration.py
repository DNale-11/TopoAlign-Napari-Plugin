"""Coarse mask-shape and fine cell-topology registration for large FISH images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import shift as ndi_shift
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, EuclideanTransform

from .registration import RigidTransform
from ..wsi_registration import add_neighbor_profile_scores


@dataclass(frozen=True)
class FishRegistrationResult:
    matches: pd.DataFrame
    coarse_transform: AffineTransform
    transform: RigidTransform
    coarse_ncc: float
    residuals: np.ndarray


def estimate_mask_contour_transform(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    downsample: int = 12,
    max_rotation_deg: float = 2.0,
    rotation_step_deg: float = 0.1,
) -> tuple[AffineTransform, float]:
    """Estimate a moving-to-fixed Euclidean transform from mask density contours."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "FISH contour registration requires opencv-python-headless."
        ) from exc
    fixed = np.asarray(fixed_mask) > 0
    moving = np.asarray(moving_mask) > 0
    if fixed.ndim != 2 or moving.ndim != 2:
        raise ValueError("FISH contour registration requires two 2D masks.")

    full_h = max(fixed.shape[0], moving.shape[0])
    full_w = max(fixed.shape[1], moving.shape[1])
    ds = max(2, int(downsample))
    small_h = int(np.ceil(full_h / ds))
    small_w = int(np.ceil(full_w / ds))

    def density_image(mask: np.ndarray) -> np.ndarray:
        canvas = np.zeros((full_h, full_w), dtype=np.uint8)
        canvas[: mask.shape[0], : mask.shape[1]] = mask
        density = cv2.resize(
            canvas.astype(np.float32),
            (small_w, small_h),
            interpolation=cv2.INTER_AREA,
        )
        density = cv2.GaussianBlur(density, (0, 0), 8.0)
        density -= float(density.min())
        peak = float(density.max())
        if peak > 0:
            density /= peak
        window = np.outer(np.hanning(small_h), np.hanning(small_w)).astype(np.float32)
        return density * window

    fixed_density = density_image(fixed)
    moving_density = density_image(moving)
    center = (small_w / 2.0, small_h / 2.0)
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None

    angles = np.arange(
        -float(max_rotation_deg),
        float(max_rotation_deg) + 0.5 * float(rotation_step_deg),
        float(rotation_step_deg),
    )
    for angle_deg in angles:
        matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        rotated = cv2.warpAffine(
            moving_density,
            matrix,
            (small_w, small_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        shift_yx, _, _ = phase_cross_correlation(
            fixed_density,
            rotated,
            upsample_factor=10,
            normalization=None,
        )
        aligned = ndi_shift(rotated, shift_yx, order=1, mode="constant", prefilter=False)
        denom = np.sqrt(np.sum(fixed_density**2) * np.sum(aligned**2)) + 1e-9
        ncc = float(np.sum(fixed_density * aligned) / denom)
        if best is None or ncc > best[0]:
            best = (ncc, float(angle_deg), np.asarray(shift_yx, dtype=float), matrix)

    if best is None:
        raise ValueError("Mask contour registration could not estimate a transform.")
    ncc, _, shift_yx, matrix = best
    matrix = np.asarray(matrix, dtype=float).copy()
    matrix[0, 2] += float(shift_yx[1])
    matrix[1, 2] += float(shift_yx[0])
    small_transform = np.vstack([matrix, [0.0, 0.0, 1.0]])
    scale = np.diag([small_w / full_w, small_h / full_h, 1.0])
    full_transform = np.linalg.inv(scale) @ small_transform @ scale
    return AffineTransform(matrix=full_transform), ncc


def match_cells_by_local_topology(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    coarse_transform: AffineTransform,
    *,
    search_radius_px: float = 45.0,
    neighbor_k: int = 8,
    topology_threshold: float = 0.42,
    max_area_ratio: float = 2.5,
    top_k: int = 1000,
) -> pd.DataFrame:
    """Match cells locally after coarse alignment, then retain mutual topology minima."""
    fixed_xy = fixed_features[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    moving_xy = moving_features[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    aligned_moving_xy = coarse_transform(moving_xy)
    fixed_tree = cKDTree(fixed_xy)

    rows: list[tuple[int, int, float]] = []
    for moving_idx, aligned_xy in enumerate(aligned_moving_xy):
        for fixed_idx in fixed_tree.query_ball_point(aligned_xy, r=float(search_radius_px)):
            residual = float(np.linalg.norm(fixed_xy[int(fixed_idx)] - aligned_xy))
            rows.append((int(fixed_idx), int(moving_idx), residual))
    if not rows:
        return pd.DataFrame(columns=["idx1", "idx2", "distance"])

    candidates = pd.DataFrame(rows, columns=["idx1", "idx2", "coarse_residual_px"])
    aligned_moving_features = moving_features.copy()
    aligned_moving_features["centroid_x"] = aligned_moving_xy[:, 0]
    aligned_moving_features["centroid_y"] = aligned_moving_xy[:, 1]
    candidates = add_neighbor_profile_scores(
        candidates,
        fixed_features,
        aligned_moving_features,
        neighbor_k=int(neighbor_k),
    )

    fixed_area = fixed_features.iloc[candidates["idx1"].to_numpy(dtype=int)]["area"].to_numpy(dtype=float)
    moving_area = moving_features.iloc[candidates["idx2"].to_numpy(dtype=int)]["area"].to_numpy(dtype=float)
    candidates["area_ratio"] = np.maximum(fixed_area, moving_area) / np.maximum(
        np.minimum(fixed_area, moving_area), 1.0
    )
    candidates = candidates[
        (candidates["neighbor_profile_diff"] <= float(topology_threshold))
        & (candidates["area_ratio"] <= float(max_area_ratio))
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["idx1", "idx2", "distance"])

    candidates["distance"] = (
        candidates["neighbor_profile_diff"]
        + 0.20 * np.log(candidates["area_ratio"])
        + 0.002 * candidates["coarse_residual_px"]
    )
    best_for_fixed = candidates.loc[candidates.groupby("idx1")["distance"].idxmin()]
    best_for_moving = candidates.loc[candidates.groupby("idx2")["distance"].idxmin()]
    matches = best_for_fixed.merge(
        best_for_moving[["idx1", "idx2"]],
        on=["idx1", "idx2"],
        how="inner",
    )
    return matches.sort_values("distance").head(max(3, int(top_k))).reset_index(drop=True)


def estimate_topology_rigid_transform(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    residual_threshold_px: float = 3.0,
    max_trials: int = 5000,
) -> tuple[RigidTransform, pd.DataFrame, np.ndarray]:
    """Fit a moving-to-fixed Euclidean transform and keep its RANSAC consensus."""
    if len(matches) < 3:
        raise ValueError(f"Topology matching found {len(matches)} pairs; need at least 3.")
    fixed_xy = fixed_features.iloc[matches["idx1"].to_numpy(dtype=int)][
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    moving_xy = moving_features.iloc[matches["idx2"].to_numpy(dtype=int)][
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    from skimage.measure import ransac

    model, inliers = ransac(
        (moving_xy, fixed_xy),
        EuclideanTransform,
        min_samples=3,
        residual_threshold=float(residual_threshold_px),
        max_trials=int(max_trials),
        rng=0,
    )
    if model is None or inliers is None or int(np.sum(inliers)) < 3:
        raise ValueError("Topology landmarks did not produce a valid rigid consensus.")
    kept = matches.loc[inliers].reset_index(drop=True)
    residuals = np.linalg.norm(model(moving_xy[inliers]) - fixed_xy[inliers], axis=1)
    transform = RigidTransform(
        rotation=np.asarray(model.params[:2, :2], dtype=float),
        translation=np.asarray(model.params[:2, 2], dtype=float),
    )
    return transform, kept, residuals


def register_large_fish_masks(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    *,
    top_k: int = 1000,
) -> FishRegistrationResult:
    coarse, coarse_ncc = estimate_mask_contour_transform(fixed_mask, moving_mask)
    candidates = match_cells_by_local_topology(
        fixed_features,
        moving_features,
        coarse,
        top_k=top_k,
    )
    transform, matches, residuals = estimate_topology_rigid_transform(
        fixed_features,
        moving_features,
        candidates,
    )
    return FishRegistrationResult(matches, coarse, transform, coarse_ncc, residuals)

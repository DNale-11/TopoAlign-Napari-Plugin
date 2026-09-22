"""Cell matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from skimage.measure import ransac
from skimage.transform import AffineTransform

from .config import MatchingConfig
from .robust_alignment import TranslationTransform


@dataclass
class TwoStageMatchResult:
    matches: pd.DataFrame
    coarse_matches: pd.DataFrame
    aligned_df2: pd.DataFrame
    coarse_offset_xy: np.ndarray
    coarse_transform: AffineTransform
    coarse_transform_method: str
    coarse_inlier_count: int
    coarse_inlier_ratio: float
    coarse_median_inlier_residual: float
    coarse_transform_accepted: bool


def _standardize_features(
    df1: pd.DataFrame, df2: pd.DataFrame, feature_columns: Tuple[str, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    if len(feature_columns) == 0:
        return np.zeros((len(df1), 0), dtype=float), np.zeros((len(df2), 0), dtype=float)
    scaler = StandardScaler()
    combined = pd.concat(
        [df1.loc[:, feature_columns], df2.loc[:, feature_columns]], axis=0, ignore_index=True
    )
    scaled = scaler.fit_transform(combined)
    f1 = scaled[: len(df1)]
    f2 = scaled[len(df1) :]
    return f1, f2


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    if len(cols) == 0:
        return
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns {missing}. "
            "Compute features first via `compute_cell_features` (regionprops-based)."
        )


def _pairwise_feature_distance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: Tuple[str, ...],
) -> np.ndarray:
    if len(feature_columns) == 0:
        return np.zeros((len(df1), len(df2)), dtype=float)
    _ensure_columns(df1, feature_columns)
    _ensure_columns(df2, feature_columns)
    f1, f2 = _standardize_features(df1, df2, feature_columns)
    if f1.shape[1] == 0:
        return np.zeros((len(df1), len(df2)), dtype=float)
    # Try GPU-accelerated pairwise distance
    from .gpu_ops import pairwise_cdist_gpu
    result = pairwise_cdist_gpu(f1, f2)
    if result is not None:
        return result
    # CPU fallback
    return np.linalg.norm(f1[:, None, :] - f2[None, :, :], axis=2)


def _all_feature_columns(config: MatchingConfig) -> tuple[str, ...]:
    shape_cols = tuple(getattr(config, "feature_columns", ()))
    topology_cols = tuple(getattr(config, "topology_feature_columns", ()))
    return tuple(dict.fromkeys(shape_cols + topology_cols))


def _append_match_report_columns(
    match_df: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    report_cols: Sequence[str],
) -> pd.DataFrame:
    if match_df.empty:
        return match_df

    idx1 = match_df["idx1"].to_numpy(dtype=int)
    idx2 = match_df["idx2"].to_numpy(dtype=int)
    out = match_df.copy()
    out["cell_id_1"] = df1.iloc[idx1]["cell_id"].to_numpy()
    out["cell_id_2"] = df2.iloc[idx2]["cell_id"].to_numpy()

    for col in report_cols:
        out[f"{col}_1"] = df1.iloc[idx1][col].to_numpy()
        out[f"{col}_2"] = df2.iloc[idx2][col].to_numpy()

    return out


def _compute_robust_feature_scales(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    if len(feature_columns) == 0:
        return np.zeros(0, dtype=float)

    combined = pd.concat(
        [df1.loc[:, feature_columns], df2.loc[:, feature_columns]],
        axis=0,
        ignore_index=True,
    )
    scales: list[float] = []
    for col in feature_columns:
        values = combined[col].to_numpy(dtype=float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        scale = max(1.4826 * mad, 1e-3)
        scales.append(float(scale))
    return np.asarray(scales, dtype=float)


def _resolve_guided_spatial_window(
    spatial_window_size: float | None,
    guided_spatial_window_size: float | None,
    residual_threshold: float,
) -> float:
    if guided_spatial_window_size is not None and guided_spatial_window_size > 0:
        return float(guided_spatial_window_size)
    if spatial_window_size is not None and spatial_window_size > 0:
        return float(spatial_window_size)
    return max(20.0, float(residual_threshold) * 4.0)


def _resolve_candidate_spatial_window(
    image_shape: Sequence[int],
    spatial_window_size: float | None,
    coarse_spatial_window_size: float | None,
    guided_spatial_window_size: float,
) -> float:
    if coarse_spatial_window_size is not None and coarse_spatial_window_size > 0:
        return float(coarse_spatial_window_size)
    if spatial_window_size is not None and spatial_window_size > 0:
        return max(float(spatial_window_size) * 2.0, guided_spatial_window_size * 1.5)

    h, w = image_shape[:2]
    return max(guided_spatial_window_size * 2.5, 0.15 * float(max(h, w)))


def _orientation_difference_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    delta = np.abs(np.asarray(angle1, dtype=float) - np.asarray(angle2, dtype=float))
    delta = np.mod(delta, np.pi)
    delta = np.minimum(delta, np.pi - delta)
    return np.rad2deg(delta)


def compute_match_distance_matrix(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
) -> np.ndarray:
    shape_cols = tuple(getattr(config, "feature_columns", ()))
    topology_cols = tuple(getattr(config, "topology_feature_columns", ()))
    feature_weight = float(getattr(config, "feature_weight", 1.0) or 0.0)
    topology_weight = float(getattr(config, "topology_weight", 0.0) or 0.0)
    position_weight = float(getattr(config, "position_weight", 0.0) or 0.0)

    dist_matrix = np.zeros((len(df1), len(df2)), dtype=float)
    if feature_weight > 0 and len(shape_cols) > 0:
        dist_matrix += feature_weight * _pairwise_feature_distance(df1, df2, shape_cols)
    if topology_weight > 0 and len(topology_cols) > 0:
        dist_matrix += topology_weight * _pairwise_feature_distance(df1, df2, topology_cols)

    if position_weight > 0:
        _ensure_columns(df1, ("pos_x_norm", "pos_y_norm"))
        _ensure_columns(df2, ("pos_x_norm", "pos_y_norm"))
        coords1 = df1[["pos_x_norm", "pos_y_norm"]].to_numpy()
        coords2 = df2[["pos_x_norm", "pos_y_norm"]].to_numpy()
        pos_dist = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
        dist_matrix += position_weight * pos_dist

    if config.spatial_window_size is not None:
        _ensure_columns(df1, ("centroid_x", "centroid_y"))
        _ensure_columns(df2, ("centroid_x", "centroid_y"))
        coords1_pixels = df1[["centroid_x", "centroid_y"]].to_numpy()
        coords2_pixels = df2[["centroid_x", "centroid_y"]].to_numpy()
        spatial_dist_pixels = np.linalg.norm(
            coords1_pixels[:, None, :] - coords2_pixels[None, :, :], axis=2
        )
        dist_matrix[spatial_dist_pixels > config.spatial_window_size] = np.inf

    return dist_matrix


def _update_feature_positions(
    df: pd.DataFrame,
    coords_xy: np.ndarray,
    image_shape: Sequence[int],
) -> pd.DataFrame:
    h, w = image_shape[:2]
    out = df.copy()
    out["centroid_x"] = coords_xy[:, 0]
    out["centroid_y"] = coords_xy[:, 1]
    out["pos_x_norm"] = coords_xy[:, 0] / float(max(w, 1))
    out["pos_y_norm"] = coords_xy[:, 1] / float(max(h, 1))
    return out


def _add_patch_coordinates(
    df: pd.DataFrame,
    image_shape: Sequence[int],
    patch_rows: int,
    patch_cols: int,
) -> pd.DataFrame:
    out = df.copy()
    h, w = image_shape[:2]
    x = out["centroid_x"].to_numpy(dtype=float)
    y = out["centroid_y"].to_numpy(dtype=float)
    out["patch_x"] = np.clip(
        np.floor(x * float(max(patch_cols, 1)) / float(max(w, 1))).astype(int),
        0,
        max(patch_cols - 1, 0),
    )
    out["patch_y"] = np.clip(
        np.floor(y * float(max(patch_rows, 1)) / float(max(h, 1))).astype(int),
        0,
        max(patch_rows - 1, 0),
    )
    return out


def _resolve_patch_top_k(
    global_top_k: int,
    patch_rows: int,
    patch_cols: int,
    patch_top_k_per_patch: int | None,
) -> int:
    if patch_top_k_per_patch is not None and patch_top_k_per_patch > 0:
        return int(patch_top_k_per_patch)
    patches = max(int(patch_rows) * int(patch_cols), 1)
    return max(1, int(np.ceil(float(global_top_k) / float(patches))))


def _build_local_morphology_candidates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    *,
    image_shape: Sequence[int],
    top_k_per_cell: int,
    spatial_window_size: float,
    max_candidates: int | None,
) -> pd.DataFrame:
    shape_cols = tuple(getattr(config, "feature_columns", ()))
    report_cols = tuple(
        col
        for col in dict.fromkeys(_all_feature_columns(config) + ("orientation",))
        if col in df1.columns and col in df2.columns
    )
    _ensure_columns(df1, tuple(dict.fromkeys(shape_cols + ("centroid_x", "centroid_y"))))
    _ensure_columns(df2, tuple(dict.fromkeys(shape_cols + ("centroid_x", "centroid_y"))))
    if len(df1) == 0 or len(df2) == 0:
        return pd.DataFrame(
            columns=[
                "idx1",
                "idx2",
                "distance",
                "spatial_dist_px",
                "translation_dx",
                "translation_dy",
                "cell_id_1",
                "cell_id_2",
            ]
        )

    top_k_per_cell = max(1, int(top_k_per_cell))
    spatial_window_size = float(spatial_window_size)
    fixed_coords = df1[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    moving_coords = df2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    tree = cKDTree(moving_coords)

    fixed_shape = df1.loc[:, shape_cols].to_numpy(dtype=float)
    moving_shape = df2.loc[:, shape_cols].to_numpy(dtype=float)
    scales = _compute_robust_feature_scales(df1, df2, shape_cols)
    if scales.size == 0:
        scales = np.ones(1, dtype=float)

    rows: list[dict[str, float | int]] = []
    for idx1, fixed_xy in enumerate(fixed_coords):
        local_idxs = tree.query_ball_point(fixed_xy, r=spatial_window_size)
        if not local_idxs:
            continue

        local_idxs = np.asarray(local_idxs, dtype=int)
        diffs = fixed_shape[idx1] - moving_shape[local_idxs]
        residuals = diffs / scales
        if residuals.ndim == 1:
            residuals = residuals[:, None]
        shape_scores = np.sqrt(np.mean(np.square(residuals), axis=1))
        spatial_dist = np.linalg.norm(moving_coords[local_idxs] - fixed_xy, axis=1)
        order = np.lexsort((spatial_dist, shape_scores))

        kept = 0
        for order_idx in order:
            score = float(shape_scores[order_idx])
            if config.distance_threshold is not None and score > float(config.distance_threshold):
                break

            idx2 = int(local_idxs[order_idx])
            moving_xy = moving_coords[idx2]
            rows.append(
                {
                    "idx1": int(idx1),
                    "idx2": idx2,
                    "distance": score,
                    "spatial_dist_px": float(spatial_dist[order_idx]),
                    "translation_dx": float(fixed_xy[0] - moving_xy[0]),
                    "translation_dy": float(fixed_xy[1] - moving_xy[1]),
                }
            )
            kept += 1
            if kept >= top_k_per_cell:
                break

    if not rows:
        return pd.DataFrame(
            columns=[
                "idx1",
                "idx2",
                "distance",
                "spatial_dist_px",
                "translation_dx",
                "translation_dy",
                "cell_id_1",
                "cell_id_2",
            ]
        )

    match_df = pd.DataFrame(rows).sort_values(
        ["distance", "spatial_dist_px"], ascending=[True, True]
    )
    if max_candidates is not None and max_candidates > 0 and len(match_df) > max_candidates:
        match_df = match_df.head(int(max_candidates))

    return _append_match_report_columns(match_df.reset_index(drop=True), df1, df2, report_cols)


def _build_sparse_fine_candidates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    *,
    image_shape: Sequence[int],
) -> pd.DataFrame:
    """Build fine-stage candidates without allocating an all-pairs matrix.

    The legacy fine matcher formed an ``N_fixed x N_moving x n_features``
    broadcast array before applying its spatial cutoff.  Large FISH masks can
    therefore require hundreds of GiB.  Reuse the KD-tree local morphology
    candidate builder and add the positional term only on those local edges.
    """
    spatial_window = config.spatial_window_size
    if spatial_window is None or float(spatial_window) <= 0:
        raise ValueError("Sparse fine matching requires a positive spatial_window_size.")

    n1 = max(len(df1), 1)
    # Keep a bounded number of local edges per fixed cell.  The final global
    # selection below still enforces one-to-one matching and top_k.
    per_cell = min(32, max(8, int(np.ceil(float(config.top_k) / n1 * 8.0))))
    candidates = _build_local_morphology_candidates(
        df1,
        df2,
        config,
        image_shape=image_shape,
        top_k_per_cell=per_cell,
        spatial_window_size=float(spatial_window),
        max_candidates=None,
    )
    if candidates.empty:
        return candidates

    position_weight = float(getattr(config, "position_weight", 0.0) or 0.0)
    if position_weight:
        candidates["distance"] = candidates["distance"].to_numpy(dtype=float) + position_weight * (
            candidates["spatial_dist_px"].to_numpy(dtype=float) / max(float(spatial_window), 1e-6)
        )
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance"])
    if config.distance_threshold is not None:
        candidates = candidates[candidates["distance"] <= float(config.distance_threshold)]
    return candidates.sort_values(["distance", "spatial_dist_px", "idx1", "idx2"]).reset_index(drop=True)


def _filter_candidate_matches_by_hard_constraints(
    coarse_matches: pd.DataFrame,
    *,
    max_area_ratio: float | None,
    max_aspect_ratio_ratio: float | None,
    max_orientation_diff_deg: float | None,
    min_orientation_eccentricity: float,
) -> pd.DataFrame:
    if coarse_matches.empty:
        return coarse_matches

    out = coarse_matches.copy()
    keep_mask = np.ones(len(out), dtype=bool)
    eps = 1e-6

    if {"area_1", "area_2"}.issubset(out.columns):
        area_1 = np.maximum(out["area_1"].to_numpy(dtype=float), eps)
        area_2 = np.maximum(out["area_2"].to_numpy(dtype=float), eps)
        area_ratio = np.maximum(area_1, area_2) / np.minimum(area_1, area_2)
        out["area_ratio"] = area_ratio
        if max_area_ratio is not None and max_area_ratio > 0:
            keep_mask &= area_ratio <= float(max_area_ratio)

    if {"aspect_ratio_1", "aspect_ratio_2"}.issubset(out.columns):
        ar_1 = np.maximum(out["aspect_ratio_1"].to_numpy(dtype=float), eps)
        ar_2 = np.maximum(out["aspect_ratio_2"].to_numpy(dtype=float), eps)
        aspect_ratio_ratio = np.maximum(ar_1, ar_2) / np.minimum(ar_1, ar_2)
        out["aspect_ratio_ratio"] = aspect_ratio_ratio
        if max_aspect_ratio_ratio is not None and max_aspect_ratio_ratio > 0:
            keep_mask &= aspect_ratio_ratio <= float(max_aspect_ratio_ratio)

    if {"orientation_1", "orientation_2"}.issubset(out.columns):
        angle_diff = _orientation_difference_deg(
            out["orientation_1"].to_numpy(dtype=float),
            out["orientation_2"].to_numpy(dtype=float),
        )
        out["orientation_diff_deg"] = angle_diff
        if max_orientation_diff_deg is not None and max_orientation_diff_deg > 0:
            reliable_orientation = np.ones(len(out), dtype=bool)
            if {"eccentricity_1", "eccentricity_2"}.issubset(out.columns):
                reliable_orientation = (
                    np.minimum(
                        out["eccentricity_1"].to_numpy(dtype=float),
                        out["eccentricity_2"].to_numpy(dtype=float),
                    )
                    >= float(min_orientation_eccentricity)
                )
            keep_mask &= (~reliable_orientation) | (angle_diff <= float(max_orientation_diff_deg))

    return out.loc[keep_mask].reset_index(drop=True)


def apply_transform_to_features(
    df: pd.DataFrame,
    transform: AffineTransform,
    image_shape: Sequence[int],
) -> pd.DataFrame:
    _ensure_columns(df, ("centroid_x", "centroid_y"))
    coords_xy = df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    coords_xy = transform(coords_xy)
    return _update_feature_positions(df, coords_xy, image_shape)


def estimate_global_offset_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    coarse_matches: pd.DataFrame,
    max_pairs: int = 20,
    mad_factor: float = 3.5,
) -> np.ndarray:
    if coarse_matches.empty:
        return np.zeros(2, dtype=float)

    best = coarse_matches.nsmallest(min(max_pairs, len(coarse_matches)), "distance")
    pts1 = df1.iloc[best["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2 = df2.iloc[best["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    offsets = pts1 - pts2
    median = np.median(offsets, axis=0)

    mad = np.median(np.abs(offsets - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1.0)
    inliers = np.all(np.abs(offsets - median) <= mad_factor * scale, axis=1)
    if int(inliers.sum()) >= 3:
        median = np.median(offsets[inliers], axis=0)

    return median.astype(float, copy=False)


def _estimate_coarse_transform_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    coarse_matches: pd.DataFrame,
    *,
    allow_scale: bool,
    prefer_affine: bool,
    residual_threshold: float,
    similarity_residual_threshold: float | None,
    max_trials: int,
) -> tuple[AffineTransform, np.ndarray, str, int, float]:
    if len(coarse_matches) < 3:
        return AffineTransform(), np.zeros(2, dtype=float), "identity", 0, float("inf")

    pts_fixed = df1.iloc[coarse_matches["idx1"].to_numpy(dtype=int)][
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    pts_moving = df2.iloc[coarse_matches["idx2"].to_numpy(dtype=int)][
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)

    try:
        _ = (allow_scale, prefer_affine, similarity_residual_threshold)
        min_samples = min(max(len(coarse_matches) // 10, 2), 6)
        model, inliers = ransac(
            (pts_moving, pts_fixed),
            TranslationTransform,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
        if model is None or inliers is None:
            raise RuntimeError("RANSAC returned no translation model.")

        residuals = model.residuals(pts_moving, pts_fixed)
        transform = AffineTransform(translation=tuple(np.asarray(model.translation, dtype=float)))
        method = "translation_ransac"
        inlier_count = int(np.count_nonzero(inliers))
        median_inlier_residual = (
            float(np.median(residuals[inliers])) if inlier_count > 0 else float("inf")
        )
    except Exception:
        offset_xy = estimate_global_offset_from_matches(df1, df2, coarse_matches)
        transform = AffineTransform(translation=(float(offset_xy[0]), float(offset_xy[1])))
        method = "translation_median"
        inlier_count = 0
        median_inlier_residual = float("inf")

    offset_xy = np.asarray(transform.translation, dtype=float)
    return transform, offset_xy, method, inlier_count, float(median_inlier_residual)


def _should_accept_coarse_transform(
    n_coarse_matches: int,
    inlier_count: int,
    median_inlier_residual: float,
    *,
    min_inlier_count: int,
    min_inlier_ratio: float,
    max_median_inlier_residual: float | None,
) -> bool:
    if n_coarse_matches <= 0:
        return False
    if inlier_count < int(min_inlier_count):
        return False
    if (float(inlier_count) / float(n_coarse_matches)) < float(min_inlier_ratio):
        return False
    if max_median_inlier_residual is not None:
        if not np.isfinite(median_inlier_residual):
            return False
        if float(median_inlier_residual) > float(max_median_inlier_residual):
            return False
    return True


def two_stage_match_cells(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    image_shape: Sequence[int],
    *,
    feature_weight: float = 1.0,
    topology_weight: float = 0.0,
    position_weight: float,
    top_k: int,
    distance_threshold: float | None,
    spatial_window_size: float | None,
    min_cells_for_two_stage: int = 20,
    coarse_top_k: int = 50,
    coarse_distance_threshold: float | None = 2.0,
    coarse_matching_mode: str = "morphology_guided",
    coarse_patch_rows: int = 3,
    coarse_patch_cols: int = 3,
    coarse_patch_top_k_per_patch: int | None = None,
    coarse_candidates_per_cell: int = 5,
    coarse_spatial_window_size: float | None = None,
    coarse_max_area_ratio: float | None = 2.5,
    coarse_max_aspect_ratio_ratio: float | None = 2.0,
    coarse_max_orientation_diff_deg: float | None = 45.0,
    coarse_min_orientation_eccentricity: float = 0.35,
    coarse_allow_scale: bool = False,
    coarse_prefer_affine: bool = False,
    coarse_residual_threshold: float = 5.0,
    coarse_similarity_residual_threshold: float | None = None,
    coarse_max_trials: int = 200,
    coarse_min_inlier_count: int = 6,
    coarse_min_inlier_ratio: float = 0.35,
    coarse_max_median_inlier_residual: float | None = None,
    guided_spatial_window_size: float | None = None,
) -> TwoStageMatchResult:
    coarse_matches = pd.DataFrame(columns=["idx1", "idx2", "distance"])
    aligned_df2 = df2
    coarse_offset_xy = np.zeros(2, dtype=float)
    coarse_transform = AffineTransform()
    coarse_transform_method = "identity"
    coarse_inlier_count = 0
    coarse_inlier_ratio = 0.0
    coarse_median_inlier_residual = float("inf")
    coarse_transform_accepted = False
    guided_window_size = _resolve_guided_spatial_window(
        spatial_window_size,
        guided_spatial_window_size,
        coarse_residual_threshold,
    )

    if len(df1) >= min_cells_for_two_stage and len(df2) >= min_cells_for_two_stage:
        coarse_config = MatchingConfig(
            feature_weight=feature_weight,
            topology_weight=topology_weight,
            position_weight=0.0,
            top_k=min(coarse_top_k, len(df1), len(df2)),
            distance_threshold=coarse_distance_threshold,
            spatial_window_size=coarse_spatial_window_size,
        )
        coarse_strategy = str(coarse_matching_mode).strip().lower()
        if coarse_strategy in {"morphology_guided", "guided", "shape_knn", "local_shape"}:
            candidate_window_size = _resolve_candidate_spatial_window(
                image_shape,
                spatial_window_size,
                coarse_spatial_window_size,
                guided_window_size,
            )
            coarse_matches = _build_local_morphology_candidates(
                df1,
                df2,
                coarse_config,
                image_shape=image_shape,
                top_k_per_cell=coarse_candidates_per_cell,
                spatial_window_size=candidate_window_size,
                max_candidates=None,
            )
            coarse_matches = _filter_candidate_matches_by_hard_constraints(
                coarse_matches,
                max_area_ratio=coarse_max_area_ratio,
                max_aspect_ratio_ratio=coarse_max_aspect_ratio_ratio,
                max_orientation_diff_deg=coarse_max_orientation_diff_deg,
                min_orientation_eccentricity=coarse_min_orientation_eccentricity,
            )
            if len(coarse_matches) > coarse_config.top_k:
                coarse_matches = (
                    coarse_matches.sort_values(
                        ["distance", "spatial_dist_px"], ascending=[True, True]
                    )
                    .head(coarse_config.top_k)
                    .reset_index(drop=True)
                )
        elif coarse_strategy == "patch":
            patch_rows = max(1, int(coarse_patch_rows))
            patch_cols = max(1, int(coarse_patch_cols))
            fixed_patched = _add_patch_coordinates(df1, image_shape, patch_rows, patch_cols)
            moving_patched = _add_patch_coordinates(df2, image_shape, patch_rows, patch_cols)
            local_top_k = _resolve_patch_top_k(
                coarse_config.top_k,
                patch_rows,
                patch_cols,
                coarse_patch_top_k_per_patch,
            )
            coarse_matches = match_cells_per_patch(
                fixed_patched,
                moving_patched,
                coarse_config,
                top_k_per_patch=local_top_k,
            )
            if len(coarse_matches) > coarse_config.top_k:
                coarse_matches = (
                    coarse_matches.sort_values("distance", ascending=True)
                    .head(coarse_config.top_k)
                    .reset_index(drop=True)
                )
        else:
            coarse_matches = greedy_match_cells(df1, df2, coarse_config)

        if len(coarse_matches) >= 3:
            coarse_transform, coarse_offset_xy, coarse_transform_method, coarse_inlier_count, coarse_median_inlier_residual = (
                _estimate_coarse_transform_from_matches(
                    df1,
                    df2,
                    coarse_matches,
                    allow_scale=coarse_allow_scale,
                    prefer_affine=coarse_prefer_affine,
                    residual_threshold=coarse_residual_threshold,
                    similarity_residual_threshold=coarse_similarity_residual_threshold,
                    max_trials=coarse_max_trials,
                )
            )
            coarse_inlier_ratio = float(coarse_inlier_count) / float(max(len(coarse_matches), 1))
            coarse_transform_accepted = _should_accept_coarse_transform(
                len(coarse_matches),
                coarse_inlier_count,
                coarse_median_inlier_residual,
                min_inlier_count=coarse_min_inlier_count,
                min_inlier_ratio=coarse_min_inlier_ratio,
                max_median_inlier_residual=(
                    coarse_residual_threshold
                    if coarse_max_median_inlier_residual is None
                    else coarse_max_median_inlier_residual
                ),
            )
            if coarse_transform_accepted:
                aligned_df2 = apply_transform_to_features(df2, coarse_transform, image_shape)

    fine_spatial_window_size = (
        guided_window_size
        if coarse_transform_accepted or spatial_window_size is None
        else spatial_window_size
    )
    fine_config = MatchingConfig(
        feature_weight=feature_weight,
        topology_weight=topology_weight,
        position_weight=position_weight,
        top_k=top_k,
        distance_threshold=distance_threshold,
        spatial_window_size=fine_spatial_window_size,
    )
    matches = greedy_match_cells(df1, aligned_df2, fine_config, image_shape=image_shape)

    return TwoStageMatchResult(
        matches=matches,
        coarse_matches=coarse_matches,
        aligned_df2=aligned_df2,
        coarse_offset_xy=coarse_offset_xy,
        coarse_transform=coarse_transform,
        coarse_transform_method=coarse_transform_method,
        coarse_inlier_count=coarse_inlier_count,
        coarse_inlier_ratio=coarse_inlier_ratio,
        coarse_median_inlier_residual=coarse_median_inlier_residual,
        coarse_transform_accepted=coarse_transform_accepted,
    )


def _assign_patch_labels(
    df: pd.DataFrame,
    image_shape: Sequence[int],
    grid: int,
) -> np.ndarray:
    """Return integer patch labels (row-major) for each cell based on centroid."""
    h, w = image_shape[:2]
    x = df["centroid_x"].to_numpy(dtype=float)
    y = df["centroid_y"].to_numpy(dtype=float)
    px = np.clip(np.floor(x * float(grid) / float(max(w, 1))).astype(int), 0, grid - 1)
    py = np.clip(np.floor(y * float(grid) / float(max(h, 1))).astype(int), 0, grid - 1)
    return py * grid + px


def _ensure_patch_coverage(
    match_df: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    dist_matrix: np.ndarray,
    config: MatchingConfig,
    image_shape: Sequence[int],
    coverage_patch_grid: int = 2,
    min_per_patch: int = 1,
) -> pd.DataFrame:
    """Re-select matches to **guarantee** every patch has landmarks.

    Strategy (aggressive):
    1. **Phase 1 – Forced guarantee**: For each patch, force-select at least
       ``min_per_patch`` matches using the best morphological pair available,
       **ignoring** the distance_threshold.  Only requires a finite distance.
       This ensures no patch is left without landmarks.
    2. **Phase 2 – Quota fill**: Continue filling each patch up to its full
       quota (``top_k // n_patches``) with the normal distance_threshold.
    3. **Phase 3 – Global fill**: Fill remaining ``top_k`` slots from the
       best unused pairs globally (with distance_threshold).
    """
    n_patches = coverage_patch_grid * coverage_patch_grid
    top_k = int(config.top_k)
    per_patch_quota = max(min_per_patch, top_k // n_patches)

    patch_labels_1 = _assign_patch_labels(df1, image_shape, coverage_patch_grid)

    # Pre-sort all candidate pairs by distance (ascending)
    n1, n2 = dist_matrix.shape
    flat_idx = np.argsort(dist_matrix, axis=None)
    rows_flat = flat_idx // n2
    cols_flat = flat_idx % n2

    used_1: set[int] = set()
    used_2: set[int] = set()
    patch_counts: dict[int, int] = {p: 0 for p in range(n_patches)}
    selected: list[tuple[int, int, float]] = []

    # Phase 1: Force at least min_per_patch matches per patch (NO distance_threshold)
    for idx in range(len(flat_idx)):
        if all(c >= min_per_patch for c in patch_counts.values()):
            break  # every patch has its guaranteed minimum
        i = int(rows_flat[idx])
        j = int(cols_flat[idx])
        d = float(dist_matrix[i, j])
        if not np.isfinite(d):
            continue  # skip inf but don't break — other patches may still need pairs
        if i in used_1 or j in used_2:
            continue
        patch_id = int(patch_labels_1[i])
        if patch_counts[patch_id] >= min_per_patch:
            continue  # this patch already has its guaranteed minimum
        selected.append((i, j, d))
        used_1.add(i)
        used_2.add(j)
        patch_counts[patch_id] += 1

    # Phase 2: Fill each patch up to its full quota (with distance_threshold)
    if len(selected) < top_k:
        for idx in range(len(flat_idx)):
            if len(selected) >= top_k:
                break
            i = int(rows_flat[idx])
            j = int(cols_flat[idx])
            d = float(dist_matrix[i, j])
            if not np.isfinite(d):
                break
            if config.distance_threshold is not None and d > float(config.distance_threshold):
                break
            if i in used_1 or j in used_2:
                continue
            patch_id = int(patch_labels_1[i])
            if patch_counts[patch_id] >= per_patch_quota:
                continue
            selected.append((i, j, d))
            used_1.add(i)
            used_2.add(j)
            patch_counts[patch_id] += 1

    # Phase 3: Fill remaining global slots with best pairs (with distance_threshold)
    if len(selected) < top_k:
        for idx in range(len(flat_idx)):
            if len(selected) >= top_k:
                break
            i = int(rows_flat[idx])
            j = int(cols_flat[idx])
            d = float(dist_matrix[i, j])
            if not np.isfinite(d):
                break
            if config.distance_threshold is not None and d > float(config.distance_threshold):
                break
            if i in used_1 or j in used_2:
                continue
            selected.append((i, j, d))
            used_1.add(i)
            used_2.add(j)

    if not selected:
        return match_df

    # Log patch distribution
    final_patch_counts = {p: 0 for p in range(n_patches)}
    for i, _, _ in selected:
        final_patch_counts[int(patch_labels_1[i])] += 1
    grid = coverage_patch_grid
    dist_lines = []
    for py in range(grid):
        row_counts = [str(final_patch_counts[py * grid + px]) for px in range(grid)]
        dist_lines.append(" | ".join(row_counts))
    dist_str = "\n    ".join(dist_lines)
    print(
        f"  Patch coverage ({grid}x{grid}): {len(selected)} matches distributed as:\n"
        f"    {dist_str}"
    )

    return pd.DataFrame(selected, columns=["idx1", "idx2", "distance"])


def greedy_match_cells(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    image_shape: Sequence[int] | None = None,
    coverage_patch_grid: int = 4,
) -> pd.DataFrame:
    """
    Greedy 1-to-1 matching between two sets of cells using feature similarity.

    When *image_shape* is provided, matches are distributed across a
    ``coverage_patch_grid x coverage_patch_grid`` spatial grid so that each
    patch receives at least ``top_k // n_patches`` matches before filling the
    remaining quota globally.  This prevents all landmarks from clustering in
    a single image region.
    """
    report_cols = _all_feature_columns(config)
    _ensure_columns(df1, report_cols)
    _ensure_columns(df2, report_cols)

    # Large microscopy masks must use the spatially sparse path.  The dense
    # distance matrix is retained only for legacy callers that do not provide
    # an image shape/spatial window.
    if image_shape is not None and config.spatial_window_size is not None:
        sparse = _build_sparse_fine_candidates(df1, df2, config, image_shape=image_shape)
        if sparse.empty:
            return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2"])
        used_1: set[int] = set()
        used_2: set[int] = set()
        selected: list[tuple[int, int, float]] = []
        for row in sparse.itertuples(index=False):
            i = int(row.idx1)
            j = int(row.idx2)
            if i in used_1 or j in used_2:
                continue
            selected.append((i, j, float(row.distance)))
            used_1.add(i)
            used_2.add(j)
            if len(selected) >= int(config.top_k):
                break
        match_df = pd.DataFrame(selected, columns=["idx1", "idx2", "distance"])
        if match_df.empty:
            return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2"])
        return _append_match_report_columns(match_df, df1, df2, report_cols)

    dist_matrix = compute_match_distance_matrix(df1, df2, config)

    # --- patch-balanced matching when image_shape is available ---------------
    if image_shape is not None and coverage_patch_grid >= 2:
        match_df = _ensure_patch_coverage(
            pd.DataFrame(),  # placeholder, not used inside
            df1,
            df2,
            dist_matrix,
            config,
            image_shape,
            coverage_patch_grid=coverage_patch_grid,
        )
        if match_df.empty:
            return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2"])
        return _append_match_report_columns(match_df, df1, df2, report_cols)

    # --- original global greedy matching ------------------------------------
    pairs = []
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            pairs.append((i, j, dist_matrix[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    used_1 = set()
    used_2 = set()
    matches = []

    for i, j, d in pairs_sorted:
        if i in used_1 or j in used_2:
            continue
        if not np.isfinite(d):
            break
        if config.distance_threshold is not None and d > config.distance_threshold:
            break
        matches.append((i, j, d))
        used_1.add(i)
        used_2.add(j)
        if len(matches) >= config.top_k:
            break

    match_df = pd.DataFrame(matches, columns=["idx1", "idx2", "distance"])
    if match_df.empty:
        return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2"])
    return _append_match_report_columns(match_df, df1, df2, report_cols)


def match_cells_per_patch(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    top_k_per_patch: int = 6,
) -> pd.DataFrame:
    """
    Match cells within each patch separately and keep the best N pairs per patch.
    """
    if top_k_per_patch < 1:
        raise ValueError("top_k_per_patch must be >= 1.")

    report_cols = _all_feature_columns(config)
    _ensure_columns(df1, report_cols)
    _ensure_columns(df2, report_cols)
    _ensure_columns(df1, ("patch_x", "patch_y"))
    _ensure_columns(df2, ("patch_x", "patch_y"))

    rows: list[tuple[int, int, float, int, int]] = []
    patches = sorted(set(zip(df1["patch_x"], df1["patch_y"])) & set(zip(df2["patch_x"], df2["patch_y"])))

    for px, py in patches:
        idxs1 = df1.index[(df1["patch_x"] == px) & (df1["patch_y"] == py)].to_list()
        idxs2 = df2.index[(df2["patch_x"] == px) & (df2["patch_y"] == py)].to_list()
        if not idxs1 or not idxs2:
            continue

        local_df1 = df1.loc[idxs1].reset_index(drop=True)
        local_df2 = df2.loc[idxs2].reset_index(drop=True)
        dist_matrix = compute_match_distance_matrix(local_df1, local_df2, config)

        candidates = []
        for i_local, idx1 in enumerate(idxs1):
            for j_local, idx2 in enumerate(idxs2):
                candidates.append((idx1, idx2, dist_matrix[i_local, j_local]))

        candidates_sorted = sorted(candidates, key=lambda x: x[2])
        used1: set[int] = set()
        used2: set[int] = set()
        kept = 0
        for idx1, idx2, d in candidates_sorted:
            if idx1 in used1 or idx2 in used2:
                continue
            if not np.isfinite(d):
                break
            if config.distance_threshold is not None and d > config.distance_threshold:
                break
            rows.append((idx1, idx2, d, px, py))
            used1.add(idx1)
            used2.add(idx2)
            kept += 1
            if kept >= top_k_per_patch:
                break

    if not rows:
        return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2", "patch_x", "patch_y"])

    match_df = pd.DataFrame(rows, columns=["idx1", "idx2", "distance", "patch_x", "patch_y"])
    return _append_match_report_columns(match_df.reset_index(drop=True), df1, df2, report_cols)


def match_cells_per_cluster(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config: MatchingConfig,
    cluster_col: str = "cluster_id",
    top_k_per_cluster: int | None = None,
) -> pd.DataFrame:
    """
    Match cells within each spatial cluster separately and keep the best pairs per cluster.
    """
    if cluster_col not in df1.columns or cluster_col not in df2.columns:
        raise ValueError(f"Missing '{cluster_col}' column required for cluster-based matching.")

    report_cols = _all_feature_columns(config)
    _ensure_columns(df1, report_cols)
    _ensure_columns(df2, report_cols)

    rows: list[dict] = []
    clusters = sorted(set(df1[cluster_col]) & set(df2[cluster_col]))
    for cluster in clusters:
        idxs1 = df1.index[df1[cluster_col] == cluster].to_list()
        idxs2 = df2.index[df2[cluster_col] == cluster].to_list()
        if not idxs1 or not idxs2:
            continue

        local_df1 = df1.loc[idxs1].reset_index(drop=True)
        local_df2 = df2.loc[idxs2].reset_index(drop=True)
        local_matches = greedy_match_cells(local_df1, local_df2, config)
        if local_matches.empty:
            continue

        for _, match in local_matches.iterrows():
            global_idx1 = idxs1[int(match["idx1"])]
            global_idx2 = idxs2[int(match["idx2"])]
            rows.append(
                {
                    "idx1": global_idx1,
                    "idx2": global_idx2,
                    "distance": match["distance"],
                    cluster_col: cluster,
                    "cell_id_1": df1.loc[global_idx1, "cell_id"],
                    "cell_id_2": df2.loc[global_idx2, "cell_id"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["idx1", "idx2", "distance", "cell_id_1", "cell_id_2", cluster_col])

    match_df = pd.DataFrame(rows)
    if top_k_per_cluster is not None and top_k_per_cluster > 0:
        kept_groups: list[pd.DataFrame] = []
        for _, group in match_df.groupby(cluster_col):
            kept_groups.append(group.sort_values("distance", ascending=True).head(top_k_per_cluster))
        match_df = pd.concat(kept_groups, ignore_index=True)

    for col in report_cols:
        match_df[f"{col}_1"] = df1.loc[match_df["idx1"], col].to_numpy()
        match_df[f"{col}_2"] = df2.loc[match_df["idx2"], col].to_numpy()

    return match_df.reset_index(drop=True)

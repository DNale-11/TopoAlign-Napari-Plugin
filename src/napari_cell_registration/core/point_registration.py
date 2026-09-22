"""Point-set registration utilities for cell centroids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform, estimate_transform
from skimage.transform import warp

try:
    import napari  # type: ignore
except Exception:
    napari = None


@dataclass
class RegistrationResult:
    transform: AffineTransform
    pts_r1: np.ndarray
    pts_r2: np.ndarray
    pts_r2_reg: np.ndarray


@dataclass
class RobustTransformResult:
    transform: AffineTransform
    inliers: np.ndarray
    residuals: np.ndarray
    method: str

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inliers))

    @property
    def median_inlier_residual(self) -> float:
        if self.inlier_count == 0:
            return float("inf")
        return float(np.median(self.residuals[self.inliers]))

    @property
    def mean_inlier_residual(self) -> float:
        if self.inlier_count == 0:
            return float("inf")
        return float(np.mean(self.residuals[self.inliers]))

    def score(self) -> tuple[int, float, float]:
        return (
            self.inlier_count,
            -self.median_inlier_residual,
            -self.mean_inlier_residual,
        )


def load_data(
    r1_path: str | Path,
    r2_path: str | Path,
    matches_path: str | Path,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load round1, round2, and matches CSVs."""
    r1 = pd.read_csv(r1_path)
    r2 = pd.read_csv(r2_path)
    matches = pd.read_csv(matches_path)
    for col in (x_col, y_col):
        if col not in r1.columns or col not in r2.columns:
            raise ValueError(f"Missing required column '{col}' in input CSVs.")
    # Accept either (cell_id_r1, cell_id_r2) or (cell_id_1, cell_id_2)
    if {"cell_id_r1", "cell_id_r2"}.issubset(matches.columns):
        r1_col, r2_col = "cell_id_r1", "cell_id_r2"
    elif {"cell_id_1", "cell_id_2"}.issubset(matches.columns):
        r1_col, r2_col = "cell_id_1", "cell_id_2"
    else:
        raise ValueError(
            "top_matches.csv must have columns 'cell_id_r1'/'cell_id_r2' (preferred) "
            "or 'cell_id_1'/'cell_id_2' as exported by the main pipeline."
        )
    matches = matches.rename(columns={r1_col: "cell_id_r1", r2_col: "cell_id_r2"})
    return r1, r2, matches


def _select_landmark_matches(matches: pd.DataFrame, top_n: int | None, sort_by: str | None) -> pd.DataFrame:
    """
    Keep only the top-N matches (optionally sorted by a score column) for estimating the transform.
    """
    m = matches
    if sort_by is not None and sort_by in m.columns:
        m = m.sort_values(sort_by, ascending=True)
    if top_n is not None:
        if top_n < 3:
            raise ValueError("Need at least 3 landmark pairs; increase --top-n-landmarks.")
        m = m.head(top_n)
    return m.reset_index(drop=True)


def _gather_landmarks(
    r1: pd.DataFrame, r2: pd.DataFrame, matches: pd.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build landmark arrays aligned by matches."""
    pts_r1 = []
    pts_r2 = []
    for _, row in matches.iterrows():
        cid1 = row["cell_id_r1"]
        cid2 = row["cell_id_r2"]
        rec1 = r1.loc[r1["cell_id"] == cid1]
        rec2 = r2.loc[r2["cell_id"] == cid2]
        if rec1.empty or rec2.empty:
            continue
        pts_r1.append([rec1.iloc[0][x_col], rec1.iloc[0][y_col]])
        pts_r2.append([rec2.iloc[0][x_col], rec2.iloc[0][y_col]])
    if len(pts_r1) < 3:
        raise ValueError("Need at least 3 matched landmarks to estimate a transform.")
    return np.asarray(pts_r1, dtype=float), np.asarray(pts_r2, dtype=float)


def _as_affine_transform(transform) -> AffineTransform:
    return AffineTransform(matrix=np.asarray(transform.params, dtype=float))


def _point_residuals(transform: AffineTransform, pts_moving: np.ndarray, pts_fixed: np.ndarray) -> np.ndarray:
    pts_reg = transform(pts_moving)
    return np.linalg.norm(pts_reg - pts_fixed, axis=1)


def estimate_robust_transform(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    *,
    prefer_affine: bool = False,
    allow_scale: bool = False,
    residual_threshold: float = 3.0,
    similarity_residual_threshold: float | None = None,
    max_trials: int = 500,
    min_inliers: int = 3,
    fallback_to_translation: bool = True,
) -> RobustTransformResult:
    if len(pts_fixed) != len(pts_moving):
        raise ValueError("Point arrays must have the same length.")
    if len(pts_fixed) < 3:
        raise ValueError("Need at least 3 matched landmarks to estimate a transform.")

    similarity_threshold = (
        residual_threshold if similarity_residual_threshold is None else similarity_residual_threshold
    )

    candidates: list[tuple[str, str, type[AffineTransform], int, float]] = []
    if not allow_scale:
        candidates.append(("rigid", "euclidean", EuclideanTransform, 2, residual_threshold))
    else:
        if prefer_affine and len(pts_fixed) >= 4:
            candidates.append(("affine", "affine", AffineTransform, 4, residual_threshold))
        candidates.append(("similarity", "similarity", SimilarityTransform, 3, similarity_threshold))

    best: RobustTransformResult | None = None
    for display_name, estimate_method, model_cls, min_samples, threshold in candidates:
        try:
            model_robust, inliers = ransac(
                (pts_moving, pts_fixed),
                model_cls,
                min_samples=min_samples,
                residual_threshold=threshold,
                max_trials=max_trials,
            )
        except Exception:
            continue

        if model_robust is None or inliers is None or int(inliers.sum()) < max(min_inliers, min_samples):
            continue

        try:
            refit = estimate_transform(estimate_method, pts_moving[inliers], pts_fixed[inliers])
            transform = _as_affine_transform(refit)
        except Exception:
            transform = _as_affine_transform(model_robust)

        residuals = _point_residuals(transform, pts_moving, pts_fixed)
        candidate = RobustTransformResult(
            transform=transform,
            inliers=np.asarray(inliers, dtype=bool),
            residuals=residuals,
            method=display_name,
        )
        if best is None or candidate.score() > best.score():
            best = candidate

    if best is not None:
        return best

    if not fallback_to_translation:
        raise RuntimeError("RANSAC failed to find a valid transform.")

    translation = np.median(pts_fixed - pts_moving, axis=0)
    transform = AffineTransform(translation=(float(translation[0]), float(translation[1])))
    residuals = _point_residuals(transform, pts_moving, pts_fixed)
    inliers = residuals <= max(residual_threshold, similarity_threshold)
    return RobustTransformResult(
        transform=transform,
        inliers=inliers,
        residuals=residuals,
        method="translation",
    )


def compose_affine_transforms(base: AffineTransform, post_transform: AffineTransform) -> AffineTransform:
    return AffineTransform(matrix=np.asarray(post_transform.params) @ np.asarray(base.params))


def _refit_transform_from_mask(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    inlier_mask: np.ndarray,
    method: str,
) -> AffineTransform | None:
    keep = np.asarray(inlier_mask, dtype=bool)
    if int(np.count_nonzero(keep)) < 3:
        return None

    method_key = str(method).strip().lower()
    if method_key == "rigid":
        estimate_method = "euclidean"
    elif method_key == "similarity":
        estimate_method = "similarity"
    elif method_key == "affine":
        estimate_method = "affine"
    elif method_key == "translation":
        translation = np.median(pts_fixed[keep] - pts_moving[keep], axis=0)
        return AffineTransform(translation=(float(translation[0]), float(translation[1])))
    else:
        return None

    try:
        refit = estimate_transform(estimate_method, pts_moving[keep], pts_fixed[keep])
    except Exception:
        return None
    return AffineTransform(matrix=np.asarray(refit.params, dtype=float))


def clip_robust_transform_inliers_by_mad(
    robust: RobustTransformResult,
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    *,
    mad_factor: float,
    residual_threshold: float,
    similarity_residual_threshold: float | None = None,
    min_inliers: int = 12,
    max_drop_fraction: float = 0.2,
    min_median_gain_px: float = 0.1,
) -> tuple[RobustTransformResult, dict[str, float | int | bool]]:
    diagnostics: dict[str, float | int | bool] = {
        "used": False,
        "limit": float("nan"),
        "kept_inliers": int(robust.inlier_count),
        "median_before": float(robust.median_inlier_residual),
        "median_after": float(robust.median_inlier_residual),
    }
    min_inliers = max(3, int(min_inliers))
    mad_factor = float(mad_factor)
    if mad_factor <= 0 or robust.inlier_count < min_inliers:
        return robust, diagnostics

    inlier_residuals = np.asarray(robust.residuals[robust.inliers], dtype=float)
    if inlier_residuals.size < min_inliers:
        return robust, diagnostics

    median_residual = float(np.median(inlier_residuals))
    mad = float(np.median(np.abs(inlier_residuals - median_residual)))
    scale = max(1.4826 * mad, 1e-6)
    clip_limit = median_residual + mad_factor * scale
    diagnostics["limit"] = float(clip_limit)

    clipped_seed_mask = np.asarray(robust.inliers, dtype=bool) & (np.asarray(robust.residuals) <= clip_limit)
    kept_inliers = int(np.count_nonzero(clipped_seed_mask))
    diagnostics["kept_inliers"] = kept_inliers
    if kept_inliers < min_inliers:
        return robust, diagnostics

    max_drop = max(1, int(np.floor(float(robust.inlier_count) * float(max_drop_fraction))))
    if kept_inliers < int(robust.inlier_count) - max_drop:
        return robust, diagnostics

    clipped_transform = _refit_transform_from_mask(pts_fixed, pts_moving, clipped_seed_mask, robust.method)
    if clipped_transform is None:
        return robust, diagnostics

    clipped_residuals = _point_residuals(clipped_transform, pts_moving, pts_fixed)
    threshold = (
        float(residual_threshold)
        if str(robust.method).strip().lower() != "similarity"
        else float(residual_threshold if similarity_residual_threshold is None else similarity_residual_threshold)
    )
    clipped_inliers = clipped_residuals <= threshold
    clipped = RobustTransformResult(
        transform=clipped_transform,
        inliers=clipped_inliers,
        residuals=clipped_residuals,
        method=robust.method,
    )
    diagnostics["median_after"] = float(clipped.median_inlier_residual)

    if clipped.inlier_count < min_inliers:
        return robust, diagnostics

    inlier_drop = int(robust.inlier_count) - int(clipped.inlier_count)
    median_gain = float(robust.median_inlier_residual) - float(clipped.median_inlier_residual)
    if inlier_drop > max_drop or median_gain < float(min_median_gain_px):
        return robust, diagnostics

    diagnostics["used"] = True
    return clipped, diagnostics


def _to_registration_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3:
        if img.shape[0] <= 10 and img.shape[1] > 10 and img.shape[2] > 10:
            return img[-1].astype(np.float64)
        if img.shape[2] <= 10:
            return img[..., -1].astype(np.float64)
        return np.max(img, axis=0).astype(np.float64)
    return np.asarray(img, dtype=np.float64)


def warp_image_with_transform(
    moving_image: np.ndarray,
    transform: AffineTransform,
    output_shape: tuple[int, int],
    *,
    order: int = 1,
) -> np.ndarray:
    if moving_image.ndim == 2:
        return warp(
            moving_image.astype(float),
            inverse_map=transform.inverse,
            output_shape=output_shape,
            preserve_range=True,
            order=order,
        )

    if moving_image.ndim == 3 and moving_image.shape[0] <= 10 and moving_image.shape[1] > 10 and moving_image.shape[2] > 10:
        warped = np.zeros((moving_image.shape[0], *output_shape), dtype=float)
        for c in range(moving_image.shape[0]):
            warped[c] = warp(
                moving_image[c].astype(float),
                inverse_map=transform.inverse,
                output_shape=output_shape,
                preserve_range=True,
                order=order,
            )
        return warped

    if moving_image.ndim == 3 and moving_image.shape[2] <= 10:
        warped = np.zeros((*output_shape, moving_image.shape[2]), dtype=float)
        for c in range(moving_image.shape[2]):
            warped[..., c] = warp(
                moving_image[..., c].astype(float),
                inverse_map=transform.inverse,
                output_shape=output_shape,
                preserve_range=True,
                order=order,
            )
        return warped

    return warp(
        moving_image.astype(float),
        inverse_map=transform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        order=order,
    )


def compute_valid_overlap_mask(
    moving_shape: tuple[int, int],
    transform: AffineTransform,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp an all-ones image to mark pixels backed by source data."""
    source_mask = np.ones(tuple(int(v) for v in moving_shape), dtype=np.uint8)
    warped = warp(
        source_mask.astype(float),
        inverse_map=transform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        order=0,
    )
    return warped > 0.5


def estimate_initial_transform(
    pts_r1: np.ndarray,
    pts_r2: np.ndarray,
    method: Literal["similarity", "affine", "euclidean"] = "euclidean",
    use_ransac: bool = True,
) -> AffineTransform:
    """Estimate a global transform mapping r2 -> r1 from landmarks."""
    if method == "similarity":
        model_cls = SimilarityTransform
        estimate_method = "similarity"
        prefer_affine = False
        allow_scale = True
    elif method == "affine":
        model_cls = AffineTransform
        estimate_method = "affine"
        prefer_affine = True
        allow_scale = True
    elif method == "euclidean":
        model_cls = EuclideanTransform
        estimate_method = "euclidean"
        prefer_affine = False
        allow_scale = False
    else:
        raise ValueError("method must be 'similarity', 'affine', or 'euclidean'")

    if use_ransac:
        robust = estimate_robust_transform(
            pts_r1,
            pts_r2,
            prefer_affine=prefer_affine,
            allow_scale=allow_scale,
            residual_threshold=5.0,
            max_trials=100,
            min_inliers=4 if method == "affine" else 3 if method == "similarity" else 2,
            fallback_to_translation=False,
        )
        return robust.transform

    return estimate_transform(estimate_method, pts_r2, pts_r1)  # type: ignore


def apply_transform(df: pd.DataFrame, transform: AffineTransform, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """Apply transform to all centroids in df, adding x_reg/y_reg columns."""
    pts = df[[x_col, y_col]].to_numpy(dtype=float)
    pts_reg = transform(pts)
    df_out = df.copy()
    df_out["x_reg"] = pts_reg[:, 0]
    df_out["y_reg"] = pts_reg[:, 1]
    return df_out


def _similarity_from_params(params: np.ndarray) -> AffineTransform:
    """Build a similarity transform from params=[theta, tx, ty, log_scale]."""
    theta, tx, ty, log_s = params
    s = np.exp(log_s)
    c, sgn = np.cos(theta), np.sin(theta)
    matrix = np.array(
        [
            [s * c, -s * sgn, tx],
            [s * sgn, s * c, ty],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return AffineTransform(matrix=matrix)


def _rigid_from_params(params: np.ndarray) -> AffineTransform:
    """Build a rigid transform from params=[theta, tx, ty]."""
    theta, tx, ty = params
    c = np.cos(theta)
    s = np.sin(theta)
    matrix = np.array(
        [
            [c, -s, tx],
            [s, c, ty],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return AffineTransform(matrix=matrix)


def _neighbor_distance_vectors(pts: np.ndarray, k: int) -> list[np.ndarray]:
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=min(k + 1, len(pts)))  # include self at idx 0
    # drop self distance at 0
    return [d[1:] for d in dists]


def refine_transform_with_neighbors(
    pts_r1: np.ndarray,
    pts_r2: np.ndarray,
    initial: AffineTransform,
    k: int = 5,
    max_iter: int = 80,
    neighbor_weight: float = 1.0,
    landmark_weight: float = 10.0,
    max_theta_deg: float = 10.0,
    max_translation: float | None = 30.0,
    max_scale_change: float = 0.05,
    accept_worse: float = 1.05,
    allow_scale: bool = False,
    optimize_translation_only: bool = False,
) -> AffineTransform:
    """
    Small refinement: adjust similarity params to preserve neighbor distances
    around matched landmarks. Returns refined transform.

    Landmark alignment is given a strong weight so top matches stay aligned, and
    parameter updates are clamped to small rotations/scales to avoid flips.
    """
    if len(pts_r1) < 3:
        return initial

    def _landmark_losses(pts_r2_reg: np.ndarray) -> tuple[float, float]:
        diff = pts_r1 - pts_r2_reg
        per_point = np.sum(diff * diff, axis=1)
        return float(np.mean(per_point)), float(np.median(per_point))

    def _neighbor_loss(pts_r2_reg: np.ndarray) -> float:
        nb1 = _neighbor_distance_vectors(pts_r1, k=k)
        nb2 = _neighbor_distance_vectors(pts_r2_reg, k=k)
        diffs = []
        for d1, d2 in zip(nb1, nb2):
            m = min(len(d1), len(d2))
            if m == 0:
                continue
            diffs.append(((d1[:m] - d2[:m]) ** 2).mean())
        if not diffs:
            return np.inf
        return float(np.mean(diffs))

    def _loss_components(transform: AffineTransform) -> tuple[float, float]:
        pts_r2_reg = transform(pts_r2)
        lm_mean, _ = _landmark_losses(pts_r2_reg)
        return lm_mean, _neighbor_loss(pts_r2_reg)

    def _loss_from_params(params: np.ndarray) -> tuple[float, float]:
        T = _similarity_from_params(params)
        pts_r2_reg = T(pts_r2)
        lm_mean, _ = _landmark_losses(pts_r2_reg)
        return lm_mean, _neighbor_loss(pts_r2_reg)

    def _loss_from_rigid_params(params: np.ndarray) -> tuple[float, float]:
        T = _rigid_from_params(params)
        pts_r2_reg = T(pts_r2)
        lm_mean, _ = _landmark_losses(pts_r2_reg)
        return lm_mean, _neighbor_loss(pts_r2_reg)

    base_mean, base_median = _landmark_losses(initial(pts_r2))

    if optimize_translation_only:
        params0 = np.array([initial.params[0, 2], initial.params[1, 2]], dtype=float)
        if max_translation is None:
            bounds = ((-np.inf, np.inf), (-np.inf, np.inf))
        else:
            bounds = (
                (params0[0] - max_translation, params0[0] + max_translation),
                (params0[1] - max_translation, params0[1] + max_translation),
            )

        def transform_from_translation(params: np.ndarray) -> AffineTransform:
            matrix = np.asarray(initial.params, dtype=float).copy()
            matrix[0, 2] = params[0]
            matrix[1, 2] = params[1]
            return AffineTransform(matrix=matrix)

        def objective_translation(params: np.ndarray) -> float:
            landmark_loss, nb_loss = _loss_components(transform_from_translation(params))
            if np.isinf(nb_loss):
                nb_loss = 1e6
            return float(landmark_weight * landmark_loss + neighbor_weight * nb_loss)

        res = minimize(
            objective_translation,
            params0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )
        if not res.success:
            return initial

        candidate = transform_from_translation(res.x)
        cand_mean_full, cand_median_full = _landmark_losses(candidate(pts_r2))
        if cand_mean_full > base_mean * accept_worse or cand_median_full > base_median * accept_worse:
            return initial
        return candidate

    A = initial.params
    s_est = np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2)
    theta_est = np.arctan2(A[1, 0] / max(s_est, 1e-8), A[0, 0] / max(s_est, 1e-8))
    tx_est, ty_est = A[0, 2], A[1, 2]

    theta_margin = np.deg2rad(max_theta_deg)
    if max_translation is None:
        tx_bounds = (-np.inf, np.inf)
        ty_bounds = (-np.inf, np.inf)
    else:
        tx_bounds = (tx_est - max_translation, tx_est + max_translation)
        ty_bounds = (ty_est - max_translation, ty_est + max_translation)
    if allow_scale:
        x0 = np.array([theta_est, tx_est, ty_est, np.log(s_est + 1e-8)], dtype=float)
        s_lower = max(1e-4, s_est * (1.0 - max_scale_change))
        s_upper = s_est * (1.0 + max_scale_change)
        bounds = (
            (theta_est - theta_margin, theta_est + theta_margin),
            tx_bounds,
            ty_bounds,
            (np.log(s_lower), np.log(s_upper)),
        )

        def objective(params: np.ndarray) -> float:
            landmark_loss, nb_loss = _loss_from_params(params)
            if np.isinf(nb_loss):
                nb_loss = 1e6
            return float(landmark_weight * landmark_loss + neighbor_weight * nb_loss)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )
        if not res.success:
            return initial
        candidate = _similarity_from_params(res.x)
    else:
        x0 = np.array([theta_est, tx_est, ty_est], dtype=float)
        bounds = (
            (theta_est - theta_margin, theta_est + theta_margin),
            tx_bounds,
            ty_bounds,
        )

        def objective(params: np.ndarray) -> float:
            landmark_loss, nb_loss = _loss_from_rigid_params(params)
            if np.isinf(nb_loss):
                nb_loss = 1e6
            return float(landmark_weight * landmark_loss + neighbor_weight * nb_loss)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )
        if not res.success:
            return initial
        candidate = _rigid_from_params(res.x)

    cand_mean_full, cand_median_full = _landmark_losses(candidate(pts_r2))
    if cand_mean_full > base_mean * accept_worse or cand_median_full > base_median * accept_worse:
        return initial
    return candidate


def plot_registration(
    pts_r1: np.ndarray,
    pts_r2: np.ndarray,
    pts_r2_reg: np.ndarray,
    out_path: str | Path | None = None,
) -> None:
    """Simple scatter/connection plot of landmark alignment."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts_r1[:, 0], pts_r1[:, 1], c="dodgerblue", label="Round1 landmarks", s=20)
    ax.scatter(pts_r2_reg[:, 0], pts_r2_reg[:, 1], c="tomato", label="Round2 registered", s=20)
    for a, b in zip(pts_r1, pts_r2_reg):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="gray", alpha=0.6, linewidth=0.8)
    ax.legend()
    ax.set_aspect("equal")
    ax.invert_yaxis()  # common for image coordinates
    ax.set_title("Landmark alignment (r2 -> r1)")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def run_registration(
    r1_csv: str | Path,
    r2_csv: str | Path,
    matches_csv: str | Path,
    output_registered_csv: str | Path = "round2_cells_registered.csv",
    plot_path: str | Path | None = "registration_plot.png",
    napari_view: bool = False,
    method: Literal["euclidean", "similarity", "affine"] = "euclidean",
    use_ransac: bool = True,
    refine: bool = True,
    neighbor_k: int = 5,
    neighbor_weight: float = 1.0,
    landmark_weight: float = 10.0,
    top_n_landmarks: int | None = None,
    sort_by: str | None = "distance",
    max_rotation_deg: float = 10.0,
    max_scale_change: float = 0.05,
    max_translation: float | None = 30.0,
    accept_worse: float = 1.05,
) -> RegistrationResult:
    """Full pipeline: load, estimate transform, optional refine, apply, and save."""
    r1, r2, matches = load_data(r1_csv, r2_csv, matches_csv, x_col="x", y_col="y")
    matches_landmarks = _select_landmark_matches(matches, top_n=top_n_landmarks, sort_by=sort_by)
    pts_r1, pts_r2 = _gather_landmarks(r1, r2, matches_landmarks, x_col="x", y_col="y")

    T_init = estimate_initial_transform(pts_r1, pts_r2, method=method, use_ransac=use_ransac)
    T_final = T_init
    if refine:
        T_final = refine_transform_with_neighbors(
            pts_r1,
            pts_r2,
            initial=T_init,
            k=neighbor_k,
            neighbor_weight=neighbor_weight,
            landmark_weight=landmark_weight,
            max_theta_deg=max_rotation_deg,
            max_translation=max_translation,
            max_scale_change=max_scale_change,
            accept_worse=accept_worse,
            allow_scale=(method != "euclidean"),
        )

    pts_r2_reg = T_final(pts_r2)
    r2_reg = apply_transform(r2, T_final, x_col="x", y_col="y")
    Path(output_registered_csv).parent.mkdir(parents=True, exist_ok=True)
    r2_reg.to_csv(output_registered_csv, index=False)

    if plot_path is not None:
        plot_registration(pts_r1, pts_r2, pts_r2_reg, out_path=plot_path)

    if napari_view:
        if napari is None:
            print("napari not installed; skipping napari view.")
        else:
            all_r1 = r1[["x", "y"]].to_numpy(float)
            all_r2 = r2[["x", "y"]].to_numpy(float)
            all_r2_reg = T_final(all_r2)

            viewer = napari.Viewer(title="Registration (all cells)")
            # All cells
            viewer.add_points(all_r1, name="round1 all", face_color="dodgerblue", size=4, opacity=0.5)
            viewer.add_points(all_r2, name="round2 all (pre)", face_color="orange", size=4, opacity=0.3)
            viewer.add_points(all_r2_reg, name="round2 all (reg)", face_color="red", size=4, opacity=0.6)
            # Landmarks for reference
            viewer.add_points(pts_r1, name="landmarks r1", face_color="cyan", size=8)
            viewer.add_points(pts_r2, name="landmarks r2 (pre)", face_color="yellow", size=8, opacity=0.6)
            viewer.add_points(pts_r2_reg, name="landmarks r2 (reg)", face_color="magenta", size=9, opacity=0.9)
            viewer.window._qt_window.raise_()
            napari.run()

    return RegistrationResult(transform=T_final, pts_r1=pts_r1, pts_r2=pts_r2, pts_r2_reg=pts_r2_reg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register round2 nuclei coordinates into round1 space.")
    parser.add_argument("round1_csv", type=Path, help="CSV with round1 cells (columns: cell_id,x,y, ...)")
    parser.add_argument("round2_csv", type=Path, help="CSV with round2 cells (columns: cell_id,x,y, ...)")
    parser.add_argument("top_matches_csv", type=Path, help="CSV with columns cell_id_r1, cell_id_r2, (score optional).")
    parser.add_argument("--output-registered-csv", type=Path, default="round2_cells_registered.csv")
    parser.add_argument("--plot-path", type=Path, default="registration_plot.png", help="If set, save landmark alignment plot.")
    parser.add_argument(
        "--method",
        choices=["euclidean", "similarity", "affine"],
        default="euclidean",
        help="Transform model to fit from landmarks.",
    )
    parser.add_argument("--no-ransac", action="store_true", help="Disable RANSAC; fit directly to all landmarks.")
    parser.add_argument("--no-refine", action="store_true", help="Disable neighbor-based refinement.")
    parser.add_argument("--neighbor-k", type=int, default=5, help="k-NN used in refinement diagnostics.")
    parser.add_argument("--neighbor-weight", type=float, default=1.0, help="Weight of neighbor distance loss.")
    parser.add_argument(
        "--landmark-weight",
        type=float,
        default=10.0,
        help="Weight to keep top landmark pairs aligned during refinement.",
    )
    parser.add_argument(
        "--top-n-landmarks",
        type=int,
        default=None,
        help="Use only the best N matches (sorted by --sort-by) to estimate the transform.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="distance",
        help="Column in matches CSV to sort by before selecting top landmarks (default: distance ascending).",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=10.0,
        help="Maximum rotation change (degrees) allowed during refinement.",
    )
    parser.add_argument(
        "--max-scale-change",
        type=float,
        default=0.05,
        help="Maximum relative scale change allowed during refinement (0.05 = ±5%).",
    )
    parser.add_argument(
        "--max-translation",
        type=float,
        default=30.0,
        help="Maximum translation change (pixels) allowed during refinement.",
    )
    parser.add_argument(
        "--accept-worse",
        type=float,
        default=1.05,
        help="Reject refinement if landmark MSE worsens beyond this factor vs. the initial transform.",
    )
    parser.add_argument("--napari", action="store_true", help="Open napari viewer to inspect landmark alignment.")
    args = parser.parse_args()

    run_registration(
        r1_csv=args.round1_csv,
        r2_csv=args.round2_csv,
        matches_csv=args.top_matches_csv,
        output_registered_csv=args.output_registered_csv,
        plot_path=args.plot_path,
        method=args.method,
        use_ransac=not args.no_ransac,
        refine=not args.no_refine,
        neighbor_k=args.neighbor_k,
        neighbor_weight=args.neighbor_weight,
        landmark_weight=args.landmark_weight,
        top_n_landmarks=args.top_n_landmarks,
        sort_by=args.sort_by,
        max_rotation_deg=args.max_rotation_deg,
        max_scale_change=args.max_scale_change,
        max_translation=args.max_translation,
        accept_worse=args.accept_worse,
        napari_view=args.napari,
    )


# ---------------------------------------------------------------------------
# Thin Plate Spline (TPS) registration
# ---------------------------------------------------------------------------

class ThinPlateSpline:
    # 2D Thin Plate Spline estimated from matched control point pairs.
    #
    # The TPS minimises bending energy while interpolating exactly at each
    # control point (when regularization=0).  In regions far from any control
    # point the warp smoothly extrapolates from the nearest points.
    #
    # Mapping:  f(x) = a0 + a1*x + a2*y + Sum_i [w_i * U(||x - p_i||)]
    # Kernel:   U(r) = r^2 * log(r),  U(0) = 0
    #
    # Parameters
    # ----------
    # regularization : float
    #     Lambda added to the diagonal of K to stabilise the system.
    #     0 -> exact interpolation at each control point.

    def __init__(self, regularization: float = 1e-3) -> None:
        self.regularization = float(regularization)
        self._ctrl: np.ndarray | None = None
        self._wx: np.ndarray | None = None
        self._wy: np.ndarray | None = None

    @staticmethod
    def _kernel(r: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = r * r * np.log(np.maximum(r, 1e-10))
        return np.where(r < 1e-10, 0.0, v)

    def _build_K(self, pts: np.ndarray) -> np.ndarray:
        diff = pts[:, None, :] - pts[None, :, :]
        return self._kernel(np.linalg.norm(diff, axis=2))

    def _eval_K_row(self, query: np.ndarray) -> np.ndarray:
        diff = query[:, None, :] - self._ctrl[None, :, :]
        return self._kernel(np.linalg.norm(diff, axis=2))

    def fit(self, pts_source: np.ndarray, pts_target: np.ndarray) -> "ThinPlateSpline":
        pts_source = np.asarray(pts_source, dtype=float)
        pts_target = np.asarray(pts_target, dtype=float)
        if pts_source.ndim != 2 or pts_source.shape[1] != 2:
            raise ValueError("pts_source must be (N, 2)")
        if pts_target.shape != pts_source.shape:
            raise ValueError("pts_source and pts_target must have the same shape")
        N = len(pts_source)
        self._ctrl = pts_source.copy()
        K = self._build_K(pts_source)
        if self.regularization > 0:
            K += np.eye(N) * self.regularization
        P = np.hstack([np.ones((N, 1)), pts_source])
        A = np.vstack([
            np.hstack([K, P]),
            np.hstack([P.T, np.zeros((3, 3))]),
        ])
        rhs_x = np.concatenate([pts_target[:, 0], np.zeros(3)])
        rhs_y = np.concatenate([pts_target[:, 1], np.zeros(3)])
        self._wx = np.linalg.solve(A, rhs_x)
        self._wy = np.linalg.solve(A, rhs_y)
        return self

    def predict(self, pts_query: np.ndarray) -> np.ndarray:
        if self._ctrl is None:
            raise RuntimeError("ThinPlateSpline has not been fitted yet.")
        pts_query = np.asarray(pts_query, dtype=float)
        if pts_query.ndim == 1:
            pts_query = pts_query[None, :]
        # Try GPU-accelerated evaluation first
        from .gpu_ops import tps_predict_gpu
        result = tps_predict_gpu(self._ctrl, self._wx, self._wy, pts_query)
        if result is not None:
            return result
        # CPU fallback
        Kq = self._eval_K_row(pts_query)
        P = np.hstack([np.ones((len(pts_query), 1)), pts_query])
        basis = np.hstack([Kq, P])
        return np.stack([basis @ self._wx, basis @ self._wy], axis=1)


def _add_boundary_anchors(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    output_shape: tuple[int, int],
    rigid_transform: AffineTransform | None = None,
    n_side: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    # Add virtual anchor points on the image boundary to prevent wild TPS
    # extrapolation outside the convex hull of real landmarks.
    H, W = output_shape
    xs = np.linspace(0, W - 1, n_side)
    ys = np.linspace(0, H - 1, n_side)
    border: list[list[float]] = []
    for x in xs:
        border.append([x, 0.0])
        border.append([x, float(H - 1)])
    for y in ys[1:-1]:
        border.append([0.0, y])
        border.append([float(W - 1), y])
    border_fixed = np.array(border, dtype=float)
    border_moving = rigid_transform.inverse(border_fixed) if rigid_transform is not None else border_fixed.copy()
    return np.vstack([pts_fixed, border_fixed]), np.vstack([pts_moving, border_moving])


def fit_tps_from_matches(
    pts_fixed: np.ndarray,
    pts_moving: np.ndarray,
    output_shape: tuple[int, int],
    rigid_transform: AffineTransform | None = None,
    *,
    regularization: float = 1e-3,
    n_boundary_per_side: int = 4,
    add_boundary_anchors_flag: bool = True,
) -> ThinPlateSpline:
    # Build a ThinPlateSpline mapping fixed-image pixel coords to
    # moving-image pixel coords (the inverse mapping for image warping).
    #
    # Parameters
    # ----------
    # pts_fixed  : (N, 2) landmark centroids in fixed image (x, y).
    # pts_moving : (N, 2) corresponding centroids in moving image.
    # output_shape : (H, W) of the fixed / output image.
    # rigid_transform : global AffineTransform (moving->fixed); its .inverse
    #                   predicts moving coords for boundary anchors.
    # regularization  : TPS lambda; larger = smoother but less exact.
    # n_boundary_per_side : virtual anchors per image edge (>= 2).
    # add_boundary_anchors_flag : whether to add stabilising boundary anchors.
    src = np.asarray(pts_fixed, dtype=float)
    tgt = np.asarray(pts_moving, dtype=float)
    if add_boundary_anchors_flag and n_boundary_per_side >= 2:
        src, tgt = _add_boundary_anchors(
            src, tgt, output_shape,
            rigid_transform=rigid_transform,
            n_side=n_boundary_per_side,
        )
    tps = ThinPlateSpline(regularization=regularization)
    tps.fit(src, tgt)
    return tps


def warp_image_with_tps(
    moving_image: np.ndarray,
    tps: ThinPlateSpline,
    output_shape: tuple[int, int],
    *,
    order: int = 1,
    chunk_size: int = 65536,
) -> np.ndarray:
    # Warp moving_image into fixed-image space using a pre-fitted TPS.
    #
    # Evaluates the TPS at every output pixel to obtain the source coordinate
    # in the moving image, then samples with scipy.ndimage.map_coordinates.
    #
    # Parameters
    # ----------
    # moving_image : 2-D (H_m, W_m) or 3-D array.
    # tps          : fitted ThinPlateSpline (fixed coords -> moving coords).
    # output_shape : (H, W) of the desired output.
    # order        : interpolation order (1=bilinear, 0=nearest for masks).
    # chunk_size   : pixels per TPS evaluation batch (speed vs memory).
    from scipy.ndimage import map_coordinates

    H, W = int(output_shape[0]), int(output_shape[1])
    total = H * W

    gy, gx = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    queries_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)

    src_xy = np.empty_like(queries_xy)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        src_xy[start:end] = tps.predict(queries_xy[start:end])

    src_row = src_xy[:, 1].reshape(H, W)
    src_col = src_xy[:, 0].reshape(H, W)

    def _warp_ch(ch: np.ndarray) -> np.ndarray:
        return map_coordinates(ch.astype(float), [src_row, src_col],
                               order=order, mode="constant", cval=0.0)

    if moving_image.ndim == 2:
        return _warp_ch(moving_image)
    if moving_image.ndim == 3 and moving_image.shape[0] <= 10 and moving_image.shape[1] > 10:
        warped = np.zeros((moving_image.shape[0], H, W), dtype=float)
        for c in range(moving_image.shape[0]):
            warped[c] = _warp_ch(moving_image[c])
        return warped
    if moving_image.ndim == 3 and moving_image.shape[2] <= 10:
        warped = np.zeros((H, W, moving_image.shape[2]), dtype=float)
        for c in range(moving_image.shape[2]):
            warped[..., c] = _warp_ch(moving_image[..., c])
        return warped
    return _warp_ch(moving_image)

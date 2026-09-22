"""
Robust global alignment using RANSAC on feature-based candidate matches.

Uses a translation-only model (no rotation) since cell imaging rounds
are assumed to differ only by a small shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage.measure import ransac
from skimage.transform import EuclideanTransform


class TranslationTransform:
    """Translation-only 2D transform compatible with skimage RANSAC.

    This model estimates only (tx, ty) with zero rotation, which is more
    robust than ``EuclideanTransform`` when feature-based candidate matches
    are noisy and could otherwise lead RANSAC to fit spurious large rotations.
    """

    def __init__(self):
        self._translation = np.zeros(2, dtype=float)

    # -- properties compatible with EuclideanTransform --
    @property
    def rotation(self) -> float:
        """Always 0 – no rotation is estimated."""
        return 0.0

    @property
    def translation(self) -> np.ndarray:
        return self._translation.copy()

    @property
    def params(self) -> np.ndarray:
        """3×3 homogeneous matrix (translation only)."""
        m = np.eye(3, dtype=float)
        m[0, 2] = self._translation[0]
        m[1, 2] = self._translation[1]
        return m

    # -- skimage RANSAC interface --
    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """Estimate translation as the mean of (dst − src)."""
        self._translation = np.mean(dst - src, axis=0)
        return True

    def residuals(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Per-point Euclidean residual after applying the translation."""
        transformed = src + self._translation
        return np.sqrt(np.sum((transformed - dst) ** 2, axis=1))

    # -- callable interface --
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Apply translation to an (N, 2) coordinate array."""
        return np.asarray(coords, dtype=float) + self._translation


def get_feature_candidates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: tuple[str, ...],
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find candidate matches based purely on feature similarity.

    Returns source and destination coordinate arrays suitable for RANSAC.
    """
    for col in feature_columns:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Missing feature column {col}")

    scaler = StandardScaler()
    combined = pd.concat(
        [df1[list(feature_columns)], df2[list(feature_columns)]],
        axis=0,
        ignore_index=True,
    )
    scaled = scaler.fit_transform(combined)
    f1 = scaled[: len(df1)]
    f2 = scaled[len(df1) :]

    n_neighbors = min(int(top_k), len(df1))
    if n_neighbors <= 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(f1)
    _, indices = nn.kneighbors(f2)

    src_coords = df2[["centroid_x", "centroid_y"]].to_numpy()
    dst_coords = df1[["centroid_x", "centroid_y"]].to_numpy()

    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    for idx_df2 in range(len(df2)):
        src_point = src_coords[idx_df2]
        for idx_df1 in indices[idx_df2]:
            src_list.append(src_point)
            dst_list.append(dst_coords[idx_df1])

    return np.asarray(src_list), np.asarray(dst_list)


def perform_global_registration(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_columns: tuple[str, ...] = (
        "area",
        "perimeter",
        "roundness",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
    ),
    top_k_candidates: int = 5,
    ransac_min_samples: int = 3,
    ransac_residual_threshold: float = 2.0,
    ransac_max_trials: int = 2000,
) -> TranslationTransform | None:
    """
    Estimate a global **translation-only** transform aligning df2 to df1.

    Uses RANSAC with a ``TranslationTransform`` model so that noisy
    feature candidates cannot produce spurious large rotations.
    """
    if df1.empty or df2.empty:
        return None

    src, dst = get_feature_candidates(df1, df2, feature_columns, top_k=top_k_candidates)
    if len(src) < ransac_min_samples:
        return None

    try:
        np.random.seed(42)
        model, _ = ransac(
            (src, dst),
            TranslationTransform,
            min_samples=ransac_min_samples,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
        )
    except Exception:
        return None

    return model


def apply_transform_to_coordinates(
    df: pd.DataFrame,
    transform,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.DataFrame:
    """
    Apply a transform (TranslationTransform or EuclideanTransform) to
    feature coordinates and return a copy.
    """
    out = df.copy()
    coords = out[[x_col, y_col]].to_numpy(dtype=float)
    aligned = transform(coords)
    out[x_col] = aligned[:, 0]
    out[y_col] = aligned[:, 1]
    return out

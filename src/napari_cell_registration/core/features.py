"""Feature extraction for segmented cells."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from scipy.spatial import cKDTree

from .config import CellFeaturesConfig

# Core properties needed for matching/registration
BASE_PROPS: List[str] = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "centroid",
    "moments_hu",
]


def _roundness(area: np.ndarray, perimeter: np.ndarray) -> np.ndarray:
    """Compute roundness metric safely."""
    perimeter_safe = np.where(perimeter == 0, np.nan, perimeter)
    roundness = 4 * np.pi * area / (perimeter_safe**2)
    return np.nan_to_num(roundness)


def _safe_ratio(numer: np.ndarray, denom: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    denom_safe = np.where(np.abs(denom) <= 1e-6, np.nan, denom)
    return np.nan_to_num(numer / denom_safe, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _add_topology_features(df: pd.DataFrame, neighbor_k: int) -> pd.DataFrame:
    out = df.copy()
    k_target = max(int(neighbor_k), 0)
    if out.empty:
        for col in (
            "nn_dist_1",
            "nn_dist_2",
            "nn_dist_3",
            "local_density",
            "neighbor_area_ratio_mean",
            "neighbor_roundness_mean",
        ):
            out[col] = []
        return out

    if len(out) < 2 or k_target < 1:
        out["nn_dist_1"] = 1.0
        out["nn_dist_2"] = 1.0
        out["nn_dist_3"] = 1.0
        out["local_density"] = 1.0
        out["neighbor_area_ratio_mean"] = 1.0
        out["neighbor_roundness_mean"] = out["roundness"].to_numpy(dtype=float)
        return out

    coords = out[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    k_eff = min(k_target, len(out) - 1)
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k_eff + 1)
    neighbor_dists = np.asarray(dists[:, 1:], dtype=float)
    neighbor_idxs = np.asarray(idxs[:, 1:], dtype=int)

    global_scale = float(np.median(neighbor_dists[:, 0])) if neighbor_dists.size else 1.0
    global_scale = max(global_scale, 1.0)
    for ordinal in range(3):
        col = f"nn_dist_{ordinal + 1}"
        if ordinal < neighbor_dists.shape[1]:
            out[col] = neighbor_dists[:, ordinal] / global_scale
        else:
            out[col] = 1.0

    mean_neighbor_dist = np.mean(neighbor_dists, axis=1)
    out["local_density"] = _safe_ratio(
        np.full(len(out), global_scale, dtype=float),
        mean_neighbor_dist,
        fill_value=1.0,
    )

    areas = out["area"].to_numpy(dtype=float)
    roundness = out["roundness"].to_numpy(dtype=float)
    neighbor_area_mean = np.mean(areas[neighbor_idxs], axis=1)
    neighbor_roundness_mean = np.mean(roundness[neighbor_idxs], axis=1)
    out["neighbor_area_ratio_mean"] = _safe_ratio(neighbor_area_mean, areas, fill_value=1.0)
    out["neighbor_roundness_mean"] = np.nan_to_num(neighbor_roundness_mean, nan=0.0)
    return out


def compute_cell_features(mask: np.ndarray, config: CellFeaturesConfig) -> pd.DataFrame:
    """
    Compute morphological/geometry features for each labeled cell using regionprops.

    Required outputs (for registration/matching):
    - centroid (x, y)
    - area, perimeter
    - roundness, eccentricity, solidity
    - major/minor axis length
    - major axis direction (orientation and unit vector)
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")

    # Collect all properties in a single regionprops_table call (includes area/perimeter/eccentricity/solidity).
    prop_names = list(dict.fromkeys(BASE_PROPS + list(config.extra_properties)))
    props: Dict[str, np.ndarray] = regionprops_table(
        mask,
        properties=prop_names,
    )

    df = pd.DataFrame(props)
    df = df.rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
        }
    )

    df["roundness"] = _roundness(df["area"].to_numpy(), df["perimeter"].to_numpy())
    df["aspect_ratio"] = _safe_ratio(
        df["major_axis_length"].to_numpy(dtype=float),
        df["minor_axis_length"].to_numpy(dtype=float),
        fill_value=1.0,
    )
    df["elongation"] = np.clip(
        1.0
        - _safe_ratio(
            df["minor_axis_length"].to_numpy(dtype=float),
            df["major_axis_length"].to_numpy(dtype=float),
            fill_value=1.0,
        ),
        0.0,
        1.0,
    )
    df["equivalent_diameter"] = np.sqrt(4.0 * np.maximum(df["area"].to_numpy(dtype=float), 0.0) / np.pi)
    # Normalized positions in [0,1] relative to image size for spatial matching.
    h, w = mask.shape
    df["pos_x_norm"] = df["centroid_x"] / float(max(w, 1))
    df["pos_y_norm"] = df["centroid_y"] / float(max(h, 1))

    # Orientation (radians) is measured CCW from the horizontal axis to the major axis
    # Provide a unit vector for downstream cosine similarity.
    df["orientation"] = df["orientation"].astype(float)
    df["axis_vec_x"] = np.cos(df["orientation"])
    df["axis_vec_y"] = np.sin(df["orientation"])

    # Hu moments: log-transform for numerical stability (values span many orders of magnitude)
    for i in range(7):
        raw_col = f"moments_hu-{i}"
        if raw_col in df.columns:
            vals = df[raw_col].to_numpy(dtype=float)
            # -sign(h) * log10(|h| + 1e-30) — standard Hu moment normalization
            df[f"hu_{i}"] = -np.sign(vals) * np.log10(np.abs(vals) + 1e-30)
            df = df.drop(columns=[raw_col])

    if config.min_area is not None:
        df = df[df["area"] >= config.min_area]
    if config.max_area is not None:
        df = df[df["area"] <= config.max_area]

    df = df.reset_index(drop=True)
    return _add_topology_features(df, config.topology_neighbor_k)

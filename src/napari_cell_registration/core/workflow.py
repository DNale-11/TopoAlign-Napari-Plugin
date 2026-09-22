"""Helpers for the napari registration workflow."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.transform import AffineTransform

from .registration import RigidTransform, estimate_rigid_transform_from_matches


MAIN_MATCHING_FEATURE_COLUMNS: tuple[str, ...] = (
    "area",
    "perimeter",
    "roundness",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
)
MIN_MATCHES_FOR_REFINEMENT = 3
PATCH_GRID = 3
PATCH_W = 1.0 / PATCH_GRID
PATCH_H = 1.0 / PATCH_GRID


def assign_patches(
    df: pd.DataFrame,
    image_width: float,
    image_height: float,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.DataFrame:
    """Attach normalized coordinates and 3x3 patch indices to a feature table."""
    out = df.copy()
    out["x_norm"] = (out[x_col] / float(image_width)).clip(0.0, 1.0)
    out["y_norm"] = (out[y_col] / float(image_height)).clip(0.0, 1.0)
    out["patch_x"] = np.clip((out["x_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
    out["patch_y"] = np.clip((out["y_norm"] * PATCH_GRID).astype(int), 0, PATCH_GRID - 1)
    out["patch_id"] = list(zip(out["patch_y"], out["patch_x"]))
    return out


def assign_clusters_from_round1(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    n_clusters: int,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    cluster_col: str = "cluster_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit KMeans on round-1 coordinates and assign labels to both rounds."""
    feats1_out = feats1.copy()
    feats2_out = feats2.copy()
    n_eff = min(int(n_clusters), len(feats1_out))
    if n_eff <= 1:
        feats1_out[cluster_col] = 0
        feats2_out[cluster_col] = 0
        return feats1_out, feats2_out

    coords1 = feats1_out[[x_col, y_col]].to_numpy()
    kmeans = KMeans(n_clusters=n_eff, random_state=0, n_init="auto")
    kmeans.fit(coords1)
    feats1_out[cluster_col] = kmeans.labels_
    if len(feats2_out) == 0:
        feats2_out[cluster_col] = pd.Series(dtype=int)
    else:
        coords2 = feats2_out[[x_col, y_col]].to_numpy()
        feats2_out[cluster_col] = kmeans.predict(coords2)
    return feats1_out, feats2_out


def _patch_center(patch_x: int, patch_y: int) -> tuple[float, float]:
    x0 = patch_x / PATCH_GRID
    y0 = patch_y / PATCH_GRID
    return x0 + PATCH_W / 2.0, y0 + PATCH_H / 2.0


def _patch_diag_px(image_width: float, image_height: float) -> float:
    patch_w_px = image_width / PATCH_GRID
    patch_h_px = image_height / PATCH_GRID
    return math.hypot(patch_w_px, patch_h_px)


def apply_rigid_to_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to an array of (x, y) points."""
    return (rotation @ points.T).T + translation


def rigid_transform_to_affine(transform: RigidTransform) -> AffineTransform:
    """Convert a RigidTransform into a skimage AffineTransform."""
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = np.asarray(transform.rotation, dtype=float)
    matrix[:2, 2] = np.asarray(transform.translation, dtype=float)
    return AffineTransform(matrix=matrix)


def cast_warped_like_original(warped: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    """Cast warped output back to the input dtype with clipping for integers."""
    dtype = np.dtype(original_dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(warped), info.min, info.max).astype(dtype)
    return warped.astype(dtype, copy=False)


def compute_L_pos(
    cell_r1: pd.Series,
    cell_r2: pd.Series,
    image_width: float,
    image_height: float,
) -> float:
    """Position loss between two cells inside the same patch."""
    _ = (image_width, image_height)
    cx, cy = _patch_center(int(cell_r1["patch_x"]), int(cell_r1["patch_y"]))
    rx1 = (cell_r1["x_norm"] - cx) / (PATCH_W / 2.0)
    ry1 = (cell_r1["y_norm"] - cy) / (PATCH_H / 2.0)
    rx2 = (cell_r2["x_norm"] - cx) / (PATCH_W / 2.0)
    ry2 = (cell_r2["y_norm"] - cy) / (PATCH_H / 2.0)
    dx = rx1 - rx2
    dy = ry1 - ry2
    return math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)


def _knn_distances(
    df_patch: pd.DataFrame,
    anchor: pd.Series,
    k: int,
    x_col: str,
    y_col: str,
) -> np.ndarray:
    if len(df_patch) <= 1:
        return np.array([])
    coords = df_patch[[x_col, y_col]].to_numpy()
    query = np.array([[anchor[x_col], anchor[y_col]]], dtype=float)
    k_eff = min(k + 1, len(df_patch))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree")
    nn.fit(coords)
    dists, _ = nn.kneighbors(query, return_distance=True)
    return dists[0][1:]


def compute_L_nei(
    cell_r1: pd.Series,
    cell_r2: pd.Series,
    df_r1_patch: pd.DataFrame,
    df_r2_patch: pd.DataFrame,
    k: int,
    image_width: float,
    image_height: float,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> float:
    """Neighbor-structure loss using k-NN distance vectors inside the same patch."""
    k_eff = min(int(k), len(df_r1_patch) - 1, len(df_r2_patch) - 1)
    if k_eff <= 0:
        return float("inf")
    d1 = _knn_distances(df_r1_patch, cell_r1, k_eff, x_col=x_col, y_col=y_col)
    d2 = _knn_distances(df_r2_patch, cell_r2, k_eff, x_col=x_col, y_col=y_col)
    if len(d1) != len(d2) or len(d1) == 0:
        return float("inf")
    diag = _patch_diag_px(image_width, image_height)
    diff = (d1 / diag) - (d2 / diag)
    return float(np.sqrt(np.mean(diff**2)))


def filter_top_pairs(
    candidate_matches: pd.DataFrame,
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    k: int,
    image_width: float,
    image_height: float,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> pd.DataFrame:
    """Filter candidate matches by positional and neighborhood consistency."""
    if candidate_matches.empty:
        return pd.DataFrame(columns=[id_r1_col, id_r2_col, "L_pos", "L_nei", "patch_x", "patch_y"])

    r1_lookup = df_r1.set_index(cell_id_col)
    r2_lookup = df_r2.set_index(cell_id_col)
    patch_groups_r1 = {pid: group for pid, group in df_r1.groupby("patch_id")}
    patch_groups_r2 = {pid: group for pid, group in df_r2.groupby("patch_id")}

    rows: list[dict[str, float | int]] = []
    for _, cand in candidate_matches.iterrows():
        cid1 = cand[id_r1_col]
        cid2 = cand[id_r2_col]
        if cid1 not in r1_lookup.index or cid2 not in r2_lookup.index:
            continue
        cell1 = r1_lookup.loc[cid1]
        cell2 = r2_lookup.loc[cid2]
        if (cell1["patch_x"], cell1["patch_y"]) != (cell2["patch_x"], cell2["patch_y"]):
            continue

        patch_id = cell1["patch_id"]
        df_r1_patch = patch_groups_r1.get(patch_id)
        df_r2_patch = patch_groups_r2.get(patch_id)
        if df_r1_patch is None or df_r2_patch is None:
            continue

        l_pos = compute_L_pos(cell1, cell2, image_width, image_height)
        if l_pos > tau_pos:
            continue
        l_nei = compute_L_nei(
            cell1,
            cell2,
            df_r1_patch,
            df_r2_patch,
            k,
            image_width,
            image_height,
            x_col=x_col,
            y_col=y_col,
        )
        if l_nei > tau_nei:
            continue

        rows.append(
            {
                id_r1_col: cid1,
                id_r2_col: cid2,
                "L_pos": l_pos,
                "L_nei": l_nei,
                "patch_x": int(cell1["patch_x"]),
                "patch_y": int(cell1["patch_y"]),
            }
        )

    return pd.DataFrame(rows)


def map_neighbors_for_pair(
    pair: pd.Series,
    df_r1_patch: pd.DataFrame,
    df_r2_patch: pd.DataFrame,
    k_neighbor: int,
    tau_map: float,
    image_width: float,
    image_height: float,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> list[dict[str, float | int | bool]]:
    """Map neighbors around a trusted pair using relative displacement."""
    if df_r1_patch.empty or df_r2_patch.empty:
        return []

    anchor1 = df_r1_patch[df_r1_patch[cell_id_col] == pair[id_r1_col]]
    anchor2 = df_r2_patch[df_r2_patch[cell_id_col] == pair[id_r2_col]]
    if anchor1.empty or anchor2.empty:
        return []

    anchor1 = anchor1.iloc[0]
    anchor2 = anchor2.iloc[0]
    diag = _patch_diag_px(image_width, image_height)
    rows: list[dict[str, float | int | bool]] = []

    coords_r1 = df_r1_patch[[x_col, y_col]].to_numpy()
    nn1 = NearestNeighbors(
        n_neighbors=min(int(k_neighbor) + 1, len(df_r1_patch)),
        algorithm="kd_tree",
    ).fit(coords_r1)

    coords_r2 = df_r2_patch[[x_col, y_col]].to_numpy()
    nn2 = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords_r2)
    idxs1 = nn1.kneighbors([[anchor1[x_col], anchor1[y_col]]], return_distance=False)

    for idx in idxs1[0][1:]:
        neigh = df_r1_patch.iloc[idx]
        dx = neigh[x_col] - anchor1[x_col]
        dy = neigh[y_col] - anchor1[y_col]
        x_pred = anchor2[x_col] + dx
        y_pred = anchor2[y_col] + dy
        dist_px, idx2 = nn2.kneighbors(np.array([[x_pred, y_pred]], dtype=float), return_distance=True)
        dist_norm = float(dist_px[0][0] / diag)
        target = df_r2_patch.iloc[idx2[0][0]]
        rows.append(
            {
                "cell_id_r1": neigh[cell_id_col],
                "cell_id_r2": target[cell_id_col],
                "pair_top_r1": anchor1[cell_id_col],
                "pair_top_r2": anchor2[cell_id_col],
                "dist_norm": dist_norm,
                "within_threshold": dist_norm <= tau_map,
                "patch_x": int(pair["patch_x"]),
                "patch_y": int(pair["patch_y"]),
            }
        )

    return rows


def map_all_neighbors(
    top_pairs: pd.DataFrame,
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    k_neighbor: int,
    tau_map: float,
    image_width: float,
    image_height: float,
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """Run neighbor mapping for all trusted pairs."""
    all_rows: list[dict[str, float | int | bool]] = []
    for _, pair in top_pairs.iterrows():
        px = int(pair["patch_x"])
        py = int(pair["patch_y"])
        df_r1_patch = df_r1[(df_r1["patch_x"] == px) & (df_r1["patch_y"] == py)]
        df_r2_patch = df_r2[(df_r2["patch_x"] == px) & (df_r2["patch_y"] == py)]
        mapped = map_neighbors_for_pair(
            pair,
            df_r1_patch,
            df_r2_patch,
            k_neighbor,
            tau_map,
            image_width,
            image_height,
            cell_id_col=cell_id_col,
        )
        all_rows.extend(mapped)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "cell_id_r1",
                "cell_id_r2",
                "pair_top_r1",
                "pair_top_r2",
                "dist_norm",
                "within_threshold",
                "patch_x",
                "patch_y",
            ]
        )
    return pd.DataFrame(all_rows)


def run_topology_matching_df(
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    candidate_matches: pd.DataFrame,
    image_width: float,
    image_height: float,
    k_pos_nei: int,
    k_neighbor: int,
    tau_pos: float = 0.5,
    tau_nei: float = 0.3,
    tau_map: float = 0.3,
    id_r1_col: str = "cell_id_r1",
    id_r2_col: str = "cell_id_r2",
    cell_id_col: str = "cell_id",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In-memory topology filtering pipeline."""
    trusted_pairs = filter_top_pairs(
        candidate_matches,
        df_r1,
        df_r2,
        k=k_pos_nei,
        image_width=image_width,
        image_height=image_height,
        tau_pos=tau_pos,
        tau_nei=tau_nei,
        id_r1_col=id_r1_col,
        id_r2_col=id_r2_col,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
    )
    neighbor_matches = map_all_neighbors(
        trusted_pairs,
        df_r1,
        df_r2,
        k_neighbor=k_neighbor,
        tau_map=tau_map,
        image_width=image_width,
        image_height=image_height,
        cell_id_col=cell_id_col,
    )
    return trusted_pairs, neighbor_matches


def compute_match_residuals(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
    transform: RigidTransform,
) -> np.ndarray:
    """Compute Euclidean residuals in pixels for matched centroids."""
    if matches.empty:
        return np.array([], dtype=float)
    pts1 = feats1.iloc[matches["idx1"]][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2 = feats2.iloc[matches["idx2"]][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2_warped = apply_rigid_to_points(pts2, transform.rotation, transform.translation)
    return np.linalg.norm(pts1 - pts2_warped, axis=1)


def estimate_rigid_transform_from_matches_ransac(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
    max_trials: int = 1000,
    residual_threshold: float = 2.0,
    min_inliers: int = MIN_MATCHES_FOR_REFINEMENT,
) -> tuple[RigidTransform, np.ndarray]:
    """Estimate a robust rigid transform using RANSAC over match pairs."""
    n_matches = len(matches)
    if n_matches == 0:
        raise ValueError("No matches provided to estimate transform.")
    if n_matches < min_inliers:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
        return transform, np.ones(n_matches, dtype=bool)

    rng = np.random.default_rng(0)
    best_transform: RigidTransform | None = None
    best_inliers = np.zeros(n_matches, dtype=bool)
    best_inlier_count = 0
    best_residual_sum = np.inf
    sample_size = min(2, n_matches)

    for _ in range(max_trials):
        sample_indices = rng.choice(n_matches, size=sample_size, replace=False)
        sample_matches = matches.iloc[sample_indices].reset_index(drop=True)
        try:
            candidate_transform = estimate_rigid_transform_from_matches(feats1, feats2, sample_matches)
        except Exception:
            continue

        residuals = compute_match_residuals(feats1, feats2, matches, candidate_transform)
        inliers = residuals <= residual_threshold
        count = int(inliers.sum())
        if count > best_inlier_count:
            best_transform = candidate_transform
            best_inliers = inliers
            best_inlier_count = count
            best_residual_sum = float(residuals[inliers].sum()) if count > 0 else np.inf
        elif count == best_inlier_count and count > 0:
            residual_sum = float(residuals[inliers].sum())
            if residual_sum < best_residual_sum:
                best_transform = candidate_transform
                best_inliers = inliers
                best_residual_sum = residual_sum

    if best_transform is not None and best_inlier_count >= min_inliers:
        refined_matches = matches.loc[best_inliers].reset_index(drop=True)
        best_transform = estimate_rigid_transform_from_matches(feats1, feats2, refined_matches)
        return best_transform, best_inliers

    transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
    return transform, np.ones(n_matches, dtype=bool)

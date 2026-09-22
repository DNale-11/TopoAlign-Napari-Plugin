"""Validation utilities for cell matching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import _ensure_columns, compute_match_distance_matrix


def _compute_neighbor_profile_diff(
    matches: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    neighbor_k: int,
) -> np.ndarray:
    if matches.empty or neighbor_k <= 0 or len(matches) < 3:
        return np.full(len(matches), np.nan, dtype=float)

    coords1 = df1.loc[matches["idx1"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    coords2 = df2.loc[matches["idx2"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    if len(coords1) != len(coords2) or len(coords1) < 3:
        return np.full(len(matches), np.nan, dtype=float)

    k = min(int(neighbor_k), len(matches) - 1)
    if k < 2:
        return np.full(len(matches), np.nan, dtype=float)

    dist1 = np.linalg.norm(coords1[:, None, :] - coords1[None, :, :], axis=2)
    dist2 = np.linalg.norm(coords2[:, None, :] - coords2[None, :, :], axis=2)
    profile1 = np.sort(dist1, axis=1)[:, 1 : k + 1]
    profile2 = np.sort(dist2, axis=1)[:, 1 : k + 1]
    denom = np.maximum(np.maximum(profile1, profile2), 1.0)
    return np.median(np.abs(profile1 - profile2) / denom, axis=1)


def _compute_match_ambiguity_scores(
    matches: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    if matches.empty:
        return np.full(0, np.nan, dtype=float), np.full(0, np.nan, dtype=float)

    feat_cols = tuple(getattr(config, "feature_columns", ()))
    topology_cols = tuple(getattr(config, "topology_feature_columns", ()))
    _ensure_columns(df1, tuple(dict.fromkeys(feat_cols + topology_cols)))
    _ensure_columns(df2, tuple(dict.fromkeys(feat_cols + topology_cols)))
    dist_matrix = compute_match_distance_matrix(df1, df2, config)

    distance_threshold = getattr(config, "distance_threshold", None)
    if distance_threshold is not None:
        dist_matrix[dist_matrix > float(distance_threshold)] = np.inf

    ratios = np.full(len(matches), np.nan, dtype=float)
    gaps = np.full(len(matches), np.nan, dtype=float)
    for match_pos, (_, match_row) in enumerate(matches.iterrows()):
        idx1 = int(match_row["idx1"])
        idx2 = int(match_row["idx2"])
        best = float(match_row["distance"])

        row_alternatives = np.array(dist_matrix[idx1], copy=True)
        col_alternatives = np.array(dist_matrix[:, idx2], copy=True)
        row_alternatives[idx2] = np.inf
        col_alternatives[idx1] = np.inf

        second_row = float(np.min(row_alternatives)) if row_alternatives.size else float("inf")
        second_col = float(np.min(col_alternatives)) if col_alternatives.size else float("inf")
        second_best = min(second_row, second_col)

        if not np.isfinite(best) or not np.isfinite(second_best):
            ratios[match_pos] = 0.0
            gaps[match_pos] = np.inf
            continue

        denom = max(second_best, 1e-9)
        ratios[match_pos] = best / denom
        gaps[match_pos] = second_best - best

    return ratios, gaps


def validate_matches(
    matches: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Validate and filter matches based on quality criteria.
    
    Parameters
    ----------
    matches : pd.DataFrame
        Initial matches from greedy_match_cells
    df1, df2 : pd.DataFrame
        Feature dataframes for round 1 and 2
    config : MatchingConfig
        Matching configuration with validation parameters
        
    Returns
    -------
    pd.DataFrame
        Validated matches with confidence scores
    """
    if matches.empty:
        return matches
    
    validated = matches.copy()
    
    # Add confidence score (inverse of distance, normalized to [0, 1])
    max_dist = validated['distance'].max()
    if max_dist > 0:
        validated['confidence'] = 1.0 - (validated['distance'] / (max_dist + 1.0))
    else:
        validated['confidence'] = 1.0
    
    # Filter by minimum confidence
    if config.min_confidence > 0:
        validated = validated[validated['confidence'] >= config.min_confidence]
    
    # Feature similarity validation: check relative differences
    if config.max_feature_diff > 0 and config.max_feature_diff < 1.0:
        for col in config.feature_columns:
            col1 = f"{col}_1"
            col2 = f"{col}_2"
            if col1 in validated.columns and col2 in validated.columns:
                vals1 = validated[col1].values
                vals2 = validated[col2].values
                denom = np.maximum(np.maximum(np.abs(vals1), np.abs(vals2)), 1e-6)
                relative_diff = np.abs(vals1 - vals2) / denom
                validated = validated[relative_diff < config.max_feature_diff]

    neighbor_k = int(getattr(config, "validation_neighbor_k", 0) or 0)
    max_profile_diff = float(getattr(config, "validation_max_neighbor_profile_diff", 0.0) or 0.0)
    if neighbor_k > 0 and max_profile_diff > 0 and len(validated) >= 3:
        profile_diff = _compute_neighbor_profile_diff(validated, df1, df2, neighbor_k)
        validated = validated.copy()
        validated["neighbor_profile_diff"] = profile_diff
        finite_mask = np.isfinite(profile_diff)
        if np.any(finite_mask):
            keep_mask = ~finite_mask | (profile_diff <= max_profile_diff)
            validated = validated[keep_mask]

    ambiguity_ratio = float(getattr(config, "validation_ambiguity_ratio", 0.0) or 0.0)
    ambiguity_min_gap = float(getattr(config, "validation_ambiguity_min_gap", 0.0) or 0.0)
    if (ambiguity_ratio > 0 or ambiguity_min_gap > 0) and not validated.empty:
        ambiguity_scores, ambiguity_gaps = _compute_match_ambiguity_scores(validated, df1, df2, config)
        validated = validated.copy()
        validated["ambiguity_ratio"] = ambiguity_scores
        validated["ambiguity_gap"] = ambiguity_gaps
        keep_mask = np.ones(len(validated), dtype=bool)
        finite_ratio = np.isfinite(ambiguity_scores)
        finite_gap = np.isfinite(ambiguity_gaps)
        if ambiguity_ratio > 0:
            keep_mask &= ~finite_ratio | (ambiguity_scores <= ambiguity_ratio)
        if ambiguity_min_gap > 0:
            keep_mask &= ~finite_gap | (ambiguity_gaps >= ambiguity_min_gap)
        validated = validated[keep_mask]
    
    return validated.reset_index(drop=True)


def compute_match_quality_stats(matches: pd.DataFrame) -> dict:
    """
    Compute quality statistics for matches.
    
    Parameters
    ----------
    matches : pd.DataFrame
        Matches with distance and optional confidence columns
        
    Returns
    -------
    dict
        Statistics including min/max/mean distance, confidence, etc.
    """
    if matches.empty:
        return {
            'n_matches': 0,
            'distance_min': np.nan,
            'distance_max': np.nan,
            'distance_mean': np.nan,
            'distance_std': np.nan,
        }
    
    stats = {
        'n_matches': len(matches),
        'distance_min': matches['distance'].min(),
        'distance_max': matches['distance'].max(),
        'distance_mean': matches['distance'].mean(),
        'distance_std': matches['distance'].std(),
    }
    
    if 'confidence' in matches.columns:
        stats['confidence_min'] = matches['confidence'].min()
        stats['confidence_mean'] = matches['confidence'].mean()
    
    return stats

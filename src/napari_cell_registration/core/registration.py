"""Rigid/similarity transform estimation from matched cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class RigidTransform:
    """Rigid (or similarity) transform in 2D."""

    rotation: np.ndarray  # shape (2, 2)
    translation: np.ndarray  # shape (2,)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.rotation, self.translation


def estimate_rigid_transform_from_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    match_df: pd.DataFrame,
    use_scale: bool = False,  # kept for API compatibility; ignored for pure translation
) -> RigidTransform:
    """
    Estimate a translation-only transform using matched cell centroids.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Feature tables containing centroid_x and centroid_y.
    match_df : pd.DataFrame
        Output of greedy_match_cells with idx1 and idx2 columns.
    use_scale : bool, optional
        Ignored; translation-only model.

    Returns
    -------
    RigidTransform
        Estimated transform.
    """
    if match_df.empty:
        raise ValueError("No matches provided to estimate transform.")

    pts_a = []
    pts_b = []
    for _, row in match_df.iterrows():
        a = df1.loc[row["idx1"], ["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        b = df2.loc[row["idx2"], ["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        pts_a.append(a)
        pts_b.append(b)

    A = np.vstack(pts_a)
    B = np.vstack(pts_b)

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # Translation-only: identity rotation, shift by centroid difference.
    R = np.eye(2, dtype=float)
    t = centroid_A - centroid_B
    return RigidTransform(rotation=R, translation=t)


def refine_with_neighbors(*args, **kwargs):
    """Placeholder for future neighborhood-based refinement."""
    raise NotImplementedError("Neighborhood-based refinement is not implemented yet.")

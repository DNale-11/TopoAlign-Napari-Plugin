"""Use synthetic masks to exercise feature extraction and moving-to-fixed transforms."""

import numpy as np
import pandas as pd
import pytest

from napari_cell_registration.core.config import CellFeaturesConfig
from napari_cell_registration.core.features import compute_cell_features
from napari_cell_registration.core.registration import estimate_rigid_transform_from_matches


def test_translated_masks_recover_moving_to_fixed_shift():
    fixed_mask = np.zeros((96, 96), dtype=np.uint16)
    moving_mask = np.zeros_like(fixed_mask)
    for label, (y, x) in enumerate([(12, 18), (30, 48), (57, 24), (65, 62)], start=1):
        fixed_mask[y:y + 5, x:x + 6] = label
        moving_mask[y + 7:y + 12, x + 11:x + 17] = label
    fixed = compute_cell_features(fixed_mask, CellFeaturesConfig())
    moving = compute_cell_features(moving_mask, CellFeaturesConfig())
    assert len(fixed) == len(moving) == 4
    matches = pd.DataFrame({"idx1": fixed.index, "idx2": moving.index})
    transform = estimate_rigid_transform_from_matches(fixed, moving, matches)
    np.testing.assert_allclose(transform.rotation, np.eye(2))
    np.testing.assert_allclose(transform.translation, [-11, -7])
    recovered = moving[["centroid_x", "centroid_y"]].to_numpy() @ transform.rotation.T + transform.translation
    np.testing.assert_allclose(recovered, fixed[["centroid_x", "centroid_y"]].to_numpy())


def test_empty_matches_fail_explicitly():
    with pytest.raises(ValueError, match="No matches"):
        estimate_rigid_transform_from_matches(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

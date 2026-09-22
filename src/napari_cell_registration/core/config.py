"""Global configuration objects and defaults for cell registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class CellposeConfig:
    """Configuration for Cellpose-SAM segmentation."""

    gpu: bool = True
    # Use SAM checkpoint by default.
    pretrained_model: str = "cpsam"
    # model_type is ignored in cellpose 4.x; keep None.
    model_type: Optional[str] = None
    diameter: Optional[float] = 8
    flow_threshold: float = -2
    cellprob_threshold: float = -2
    min_size: int = 1
    # Optional anisotropy for 3D Z-stacks (None -> isotropic / default behavior).
    anisotropy: Optional[float] = None


@dataclass
class CellFeaturesConfig:
    """Configuration for cell feature extraction."""

    min_area: Optional[int] = None
    max_area: Optional[int] = None
    extra_properties: Tuple[str, ...] = field(default_factory=tuple)
    topology_neighbor_k: int = 3


@dataclass
class MatchingConfig:
    """Configuration for cell matching."""
    
    feature_columns: Tuple[str, ...] = (
        "area",
        "perimeter",
        "roundness",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "aspect_ratio",
        "elongation",
        "equivalent_diameter",
        "hu_0",
        "hu_1",
        "hu_2",
        "hu_3",
        "hu_4",
        "hu_5",
        "hu_6",
    )
    # Weight for morphology/feature similarity in the combined distance.
    # Lower values make matching rely more on spatial consistency.
    feature_weight: float = 1.0
    # Local topology features help distinguish cells with similar morphology but different neighborhoods.
    topology_feature_columns: Tuple[str, ...] = (
        "nn_dist_1",
        "nn_dist_2",
        "nn_dist_3",
        "local_density",
        "neighbor_area_ratio_mean",
        "neighbor_roundness_mean",
    )
    topology_weight: float = 0.0
    # Weight for spatial proximity (uses normalized positions)
    position_weight: float = 1.0
    # Maximum number of matches to return
    top_k: int = 10
    # Maximum distance threshold (standardized units) - matches beyond this are rejected
    distance_threshold: Optional[float] = None
    # Minimum confidence score (0-1) - matches below this are rejected
    min_confidence: float = 0.0
    # Maximum allowed relative difference in features (0-1, e.g. 0.5 = 50% difference)
    max_feature_diff: float = 0.5
    # Number of nearest matched neighbors used for local geometry profile validation (0 disables).
    validation_neighbor_k: int = 0
    # Maximum allowed median relative difference between local matched-neighbor distance profiles.
    validation_max_neighbor_profile_diff: float = 0.35
    # Maximum allowed ambiguity ratio against the next-best feasible alternative.
    # <= 0 disables this filter. Lower values require a more distinctive best match.
    validation_ambiguity_ratio: float = 0.0
    # Minimum gap required between the chosen distance and the next-best feasible alternative.
    # <= 0 disables this filter.
    validation_ambiguity_min_gap: float = 0.0
    # Spatial search window size in pixels (None = no spatial constraint)
    # Cells can only match other cells within this pixel distance
    spatial_window_size: Optional[float] = None


# Default instances that can be imported elsewhere
DEFAULT_CELLPOSE_CONFIG = CellposeConfig()
DEFAULT_FEATURE_CONFIG = CellFeaturesConfig()
DEFAULT_MATCHING_CONFIG = MatchingConfig()

"""WSI mask-based landmark search and translation registration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import shift
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from tifffile import imread, imwrite

from .core.config import CellFeaturesConfig, MatchingConfig
from .core.features import compute_cell_features
from .core.registration import RigidTransform
from .core.workflow import cast_warped_like_original, compute_match_residuals


@dataclass
class WSIRegistrationResult:
    matches: pd.DataFrame
    transform: RigidTransform
    residuals: np.ndarray
    registered_image: np.ndarray | None
    registered_mask: np.ndarray | None
    local_translation_grid: np.ndarray | None = None
    local_affine_warp: "KnnLocalAffineWarp | None" = None


WSI_CENTROID_METADATA_FIELDS = (
    "source_wsi_path",
    "source_mask_path",
    "coordinate_space",
    "origin_x",
    "origin_y",
    "downsample",
    "image_width",
    "image_height",
    "mpp_x",
    "mpp_y",
    "mpp_reliable",
    "channel",
    "segmentation_method",
)


@dataclass(frozen=True)
class WSICentroidMetadata:
    """Coordinate metadata required to interpret WSI centroid CSV files."""

    source_wsi_path: str | None
    source_mask_path: str | None
    coordinate_space: str
    origin_x: float
    origin_y: float
    downsample: float
    image_width: int
    image_height: int
    mpp_x: float | None
    mpp_y: float | None
    mpp_reliable: bool
    channel: str
    segmentation_method: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, source_name: str = "metadata") -> "WSICentroidMetadata":
        normalized = _normalize_centroid_metadata_mapping(payload)
        missing = [field for field in WSI_CENTROID_METADATA_FIELDS if field not in normalized]
        if missing:
            raise ValueError(f"{source_name} is missing WSI centroid metadata fields: {missing}")

        def _optional_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out if np.isfinite(out) else None

        mpp_x = _optional_float(normalized.get("mpp_x"))
        mpp_y = _optional_float(normalized.get("mpp_y"))
        downsample = float(normalized["downsample"])
        if not np.isfinite(downsample) or downsample <= 0:
            raise ValueError(f"{source_name} has invalid downsample={normalized['downsample']!r}.")
        image_width = int(normalized["image_width"])
        image_height = int(normalized["image_height"])
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"{source_name} has invalid image size {image_width}x{image_height}.")
        return cls(
            source_wsi_path=_none_if_empty(normalized.get("source_wsi_path")),
            source_mask_path=_none_if_empty(normalized.get("source_mask_path")),
            coordinate_space=str(normalized["coordinate_space"]).strip().lower(),
            origin_x=float(normalized["origin_x"]),
            origin_y=float(normalized["origin_y"]),
            downsample=downsample,
            image_width=image_width,
            image_height=image_height,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            mpp_reliable=_parse_bool(normalized["mpp_reliable"]) and mpp_x is not None and mpp_y is not None,
            channel=str(normalized["channel"]),
            segmentation_method=str(normalized["segmentation_method"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "WSICentroidMetadata":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"WSI centroid metadata must be a JSON object: {path}")
        return cls.from_mapping(payload, source_name=str(path))

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def image_shape_yx(self) -> tuple[int, int]:
        return int(self.image_height), int(self.image_width)


@dataclass
class WSICentroidRegistrationResult:
    """Result of mIF-DAPI level-0 centroid registration into HE level-0 space."""

    fixed_centroids: pd.DataFrame
    moving_centroids: pd.DataFrame
    registered_moving_centroids: pd.DataFrame
    matches: pd.DataFrame
    global_affine_mif_to_he: np.ndarray
    inverse_global_affine_he_to_mif: np.ndarray
    residuals: np.ndarray
    fixed_metadata: WSICentroidMetadata
    moving_metadata: WSICentroidMetadata
    local_deformation_grid: pd.DataFrame
    local_affine_warp: "KnnLocalAffineWarp | None" = None
    inverse_local_affine_warp: "KnnLocalAffineWarp | None" = None
    diagnostics: dict[str, Any] | None = None


def load_wsi_centroids(
    csv_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, WSICentroidMetadata]:
    """Load a centroid CSV and its required WSI coordinate metadata JSON."""
    csv_path = Path(csv_path)
    if metadata_path is None:
        metadata_path = csv_path.with_suffix(".json")
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise ValueError(
            f"WSI centroid CSV requires a paired metadata JSON. Missing: {metadata_path}"
        )
    return pd.read_csv(csv_path), WSICentroidMetadata.from_json(metadata_path)


def run_wsi_centroid_registration(
    fixed_centroids: pd.DataFrame | str | Path,
    fixed_metadata: WSICentroidMetadata | dict[str, Any] | str | Path | None,
    moving_centroids: pd.DataFrame | str | Path,
    moving_metadata: WSICentroidMetadata | dict[str, Any] | str | Path | None,
    output_dir: str | Path | None = None,
    top_k: int = 320,
    use_knn_local_affine: bool = True,
    knn_k: int = 8,
    knn_power: float = 2.0,
    residual_filter: bool = True,
    match_radius_px: float | None = None,
    deformation_grid: int = 16,
    fixed_tissue_mask: np.ndarray | str | Path | None = None,
    moving_tissue_mask: np.ndarray | str | Path | None = None,
    fixed_tissue_downsample: float | None = None,
    moving_tissue_downsample: float | None = None,
    refine_global_affine_with_centroids: bool | None = None,
    write_preview: bool = True,
    write_registered_wsi: bool = False,
    moving_wsi_path: str | Path | None = None,
    channel_region_reader: Any | None = None,
    channel_count: int | None = None,
    output_dtype: Any | None = None,
) -> WSICentroidRegistrationResult:
    """
    Register mIF-DAPI centroids to HE centroids in level-0 pixel coordinates.

    The saved forward transform is always mIF level-0 pixel -> HE level-0 pixel.
    """
    fixed_df, fixed_meta = _coerce_centroids_and_metadata(fixed_centroids, fixed_metadata, "fixed")
    moving_df, moving_meta = _coerce_centroids_and_metadata(moving_centroids, moving_metadata, "moving")

    fixed_l0 = centroid_table_to_level0_pixel(fixed_df, fixed_meta)
    moving_l0 = centroid_table_to_level0_pixel(moving_df, moving_meta)
    if len(fixed_l0) < 3 or len(moving_l0) < 3:
        raise ValueError(
            f"WSI centroid registration needs at least 3 centroids per side; "
            f"got fixed={len(fixed_l0)}, moving={len(moving_l0)}."
        )

    global_affine, coarse_method = estimate_initial_mif_to_he_affine(
        fixed_l0[["centroid_x", "centroid_y"]].to_numpy(dtype=float),
        moving_l0[["centroid_x", "centroid_y"]].to_numpy(dtype=float),
        fixed_meta,
        moving_meta,
    )
    if fixed_tissue_mask is not None and moving_tissue_mask is not None:
        global_affine, tissue_diag = estimate_tissue_similarity_affine(
            fixed_tissue_mask,
            moving_tissue_mask,
            fixed_tissue_downsample or fixed_meta.downsample,
            moving_tissue_downsample or moving_meta.downsample,
            initial_affine=global_affine,
        )
        coarse_method = "tissue_similarity_affine"
    else:
        tissue_diag = None
    if refine_global_affine_with_centroids is None:
        refine_global_affine_with_centroids = tissue_diag is None
    matches, refined_affine = refine_affine_from_centroid_structure(
        fixed_l0,
        moving_l0,
        global_affine,
        top_k=top_k,
        match_radius_px=match_radius_px,
        residual_filter=residual_filter,
    )
    if refine_global_affine_with_centroids and len(matches) >= 3:
        global_affine = refined_affine
    inverse_global = _invert_affine_matrix(global_affine)

    moving_xy = moving_l0[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    registered_xy = _apply_affine_matrix(moving_xy, global_affine)
    local_affine = None
    inverse_local_affine = None
    if use_knn_local_affine and len(matches) >= 3:
        moving_match_xy = matches[["moving_x", "moving_y"]].to_numpy(dtype=float)
        fixed_match_xy = matches[["fixed_x", "fixed_y"]].to_numpy(dtype=float)
        local_affine = KnnLocalAffineWarp(
            moving_match_xy,
            fixed_match_xy,
            k=knn_k,
            power=knn_power,
        )
        inverse_local_affine = KnnLocalAffineWarp(
            fixed_match_xy,
            moving_match_xy,
            k=knn_k,
            power=knn_power,
        )
        registered_xy = local_affine.predict(moving_xy)

    registered_moving = moving_l0.copy()
    registered_moving["registered_centroid_x"] = registered_xy[:, 0]
    registered_moving["registered_centroid_y"] = registered_xy[:, 1]
    registered_moving["centroid_x"] = registered_moving["registered_centroid_x"]
    registered_moving["centroid_y"] = registered_moving["registered_centroid_y"]

    local_grid = build_local_deformation_grid(
        fixed_meta.image_shape_yx,
        inverse_global,
        inverse_local_affine,
        grid=deformation_grid,
    )
    residuals = matches["residual_px"].to_numpy(dtype=float) if "residual_px" in matches.columns else np.array([], dtype=float)
    diagnostics: dict[str, Any] = {
        "direction": "mIF_level0_pixel_to_HE_level0_pixel",
        "coarse_method": coarse_method,
        "fixed_centroids": int(len(fixed_l0)),
        "moving_centroids": int(len(moving_l0)),
        "matches": int(len(matches)),
        "residual_mean_px": float(np.mean(residuals)) if residuals.size else None,
        "residual_max_px": float(np.max(residuals)) if residuals.size else None,
        "local_deformation_grid_coordinate_space": "HE_level0_pixel",
        "local_refinement": "knn_local_affine" if local_affine is not None else "global_affine_only",
    }
    if tissue_diag is not None:
        diagnostics["tissue_similarity"] = tissue_diag

    result = WSICentroidRegistrationResult(
        fixed_centroids=fixed_l0,
        moving_centroids=moving_l0,
        registered_moving_centroids=registered_moving,
        matches=matches,
        global_affine_mif_to_he=global_affine,
        inverse_global_affine_he_to_mif=inverse_global,
        residuals=residuals,
        fixed_metadata=fixed_meta,
        moving_metadata=moving_meta,
        local_deformation_grid=local_grid,
        local_affine_warp=local_affine,
        inverse_local_affine_warp=inverse_local_affine,
        diagnostics=diagnostics,
    )
    if output_dir is not None:
        save_wsi_centroid_registration_result(result, output_dir, write_preview=write_preview)
    if write_registered_wsi:
        write_registered_multichannel_wsi(
            result,
            moving_wsi_path=moving_wsi_path or moving_meta.source_wsi_path,
            output_dir=output_dir,
            channel_region_reader=channel_region_reader,
            channel_count=channel_count,
            output_dtype=output_dtype,
        )
    return result


def centroid_table_to_level0_pixel(
    centroids: pd.DataFrame,
    metadata: WSICentroidMetadata,
) -> pd.DataFrame:
    """Normalize centroid CSV coordinates into source WSI level-0 pixel coordinates."""
    df = normalize_wsi_feature_table(centroids, metadata.image_shape_yx)
    xy = df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    space = metadata.coordinate_space
    if space in {"level0_pixel", "level0", "wsi_level0_pixel"}:
        if not (np.isclose(metadata.origin_x, 0.0) and np.isclose(metadata.origin_y, 0.0) and np.isclose(metadata.downsample, 1.0)):
            xy[:, 0] = metadata.origin_x + xy[:, 0] * metadata.downsample
            xy[:, 1] = metadata.origin_y + xy[:, 1] * metadata.downsample
    elif space in {"roi_pixel", "mask_pixel", "downsampled_pixel", "preview_pixel"}:
        xy[:, 0] = metadata.origin_x + xy[:, 0] * metadata.downsample
        xy[:, 1] = metadata.origin_y + xy[:, 1] * metadata.downsample
    else:
        raise ValueError(
            f"Unsupported WSI centroid coordinate_space={metadata.coordinate_space!r}; "
            "expected level0_pixel, roi_pixel, mask_pixel, downsampled_pixel, or preview_pixel."
        )
    out = df.copy()
    out["source_centroid_x"] = df["centroid_x"].to_numpy(dtype=float)
    out["source_centroid_y"] = df["centroid_y"].to_numpy(dtype=float)
    out["centroid_x"] = xy[:, 0]
    out["centroid_y"] = xy[:, 1]
    out["pos_x_norm"] = out["centroid_x"] / float(max(metadata.image_width, 1))
    out["pos_y_norm"] = out["centroid_y"] / float(max(metadata.image_height, 1))
    return out.reset_index(drop=True)


def estimate_initial_mif_to_he_affine(
    fixed_xy: np.ndarray,
    moving_xy: np.ndarray,
    fixed_metadata: WSICentroidMetadata,
    moving_metadata: WSICentroidMetadata,
) -> tuple[np.ndarray, str]:
    """Estimate coarse mIF level-0 px -> HE level-0 px affine from MPP or density extent."""
    fixed_xy = np.asarray(fixed_xy, dtype=float)
    moving_xy = np.asarray(moving_xy, dtype=float)
    if _has_reliable_mpp(fixed_metadata) and _has_reliable_mpp(moving_metadata):
        sx = float(moving_metadata.mpp_x) / float(fixed_metadata.mpp_x)
        sy = float(moving_metadata.mpp_y) / float(fixed_metadata.mpp_y)
        method = "mpp_scale_plus_centroid_density"
    else:
        sx, sy = _robust_extent_scale(fixed_xy, moving_xy)
        method = "centroid_density_extent_affine"
    scaled_moving = moving_xy * np.array([sx, sy], dtype=float)[None, :]
    translation = _robust_center(fixed_xy) - _robust_center(scaled_moving)
    matrix = np.array(
        [
            [sx, 0.0, float(translation[0])],
            [0.0, sy, float(translation[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return matrix, method


def estimate_tissue_similarity_affine(
    fixed_tissue_mask: np.ndarray | str | Path,
    moving_tissue_mask: np.ndarray | str | Path,
    fixed_downsample: float,
    moving_downsample: float,
    initial_affine: np.ndarray,
    max_rotation_deg: float = 20.0,
    scale_bounds: tuple[float, float] = (0.9, 1.1),
    translation_window_px: float = 18000.0,
    optimize_stride: int = 8,
    maxiter: int = 60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize a tissue-level similarity transform in level-0 pixel coordinates."""
    from scipy.ndimage import affine_transform
    from scipy.optimize import differential_evolution

    fixed = _load_binary_mask(fixed_tissue_mask)
    moving = _load_binary_mask(moving_tissue_mask)
    if fixed.size == 0 or moving.size == 0 or not np.any(fixed) or not np.any(moving):
        raise ValueError("Tissue similarity affine requires non-empty fixed and moving tissue masks.")

    fds = float(fixed_downsample)
    mds = float(moving_downsample)
    if fds <= 0 or mds <= 0:
        raise ValueError("Tissue mask downsample values must be positive.")

    stride = max(1, int(optimize_stride))
    fixed_opt = fixed[::stride, ::stride]
    moving_opt = moving[::stride, ::stride]
    fds_opt = fds * stride
    mds_opt = mds * stride

    initial_affine = np.asarray(initial_affine, dtype=float)
    base_scale = float(np.sqrt(abs(np.linalg.det(initial_affine[:2, :2]))))
    if not np.isfinite(base_scale) or base_scale <= 0:
        base_scale = 1.0
    init_tx = float(initial_affine[0, 2])
    init_ty = float(initial_affine[1, 2])
    center_tx, center_ty = _tissue_center_translation(fixed, moving, fds, mds, base_scale)
    if not np.isfinite(init_tx) or not np.isfinite(init_ty):
        init_tx, init_ty = center_tx, center_ty

    def _matrix(theta_deg: float, scale_mult: float, tx: float, ty: float) -> np.ndarray:
        theta = np.deg2rad(float(theta_deg))
        scale = base_scale * float(scale_mult)
        c = np.cos(theta) * scale
        s = np.sin(theta) * scale
        return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=float)

    def _warp(mask: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int], out_ds: float, in_ds: float) -> np.ndarray:
        inv = _invert_affine_matrix(matrix)
        affine_yx = np.array(
            [
                [inv[1, 1] * out_ds / in_ds, inv[1, 0] * out_ds / in_ds],
                [inv[0, 1] * out_ds / in_ds, inv[0, 0] * out_ds / in_ds],
            ],
            dtype=float,
        )
        offset_yx = np.array([inv[1, 2] / in_ds, inv[0, 2] / in_ds], dtype=float)
        return affine_transform(
            mask.astype(np.float32),
            matrix=affine_yx,
            offset=offset_yx,
            output_shape=output_shape,
            order=0,
            mode="constant",
            cval=0.0,
        ) > 0.5

    def _dice_for(params: np.ndarray, use_full: bool = False) -> float:
        theta, scale_mult, tx, ty = params
        matrix = _matrix(theta, scale_mult, tx, ty)
        if use_full:
            warped = _warp(moving, matrix, fixed.shape, fds, mds)
            target = fixed
        else:
            warped = _warp(moving_opt, matrix, fixed_opt.shape, fds_opt, mds_opt)
            target = fixed_opt
        inter = float(np.logical_and(target, warped).sum())
        denom = float(target.sum() + warped.sum())
        return (2.0 * inter / denom) if denom > 0 else 0.0

    def _objective(params: np.ndarray) -> float:
        return -_dice_for(params, use_full=False)

    tx_center = init_tx if np.isfinite(init_tx) else center_tx
    ty_center = init_ty if np.isfinite(init_ty) else center_ty
    bounds = [
        (-float(max_rotation_deg), float(max_rotation_deg)),
        (float(scale_bounds[0]), float(scale_bounds[1])),
        (tx_center - float(translation_window_px), tx_center + float(translation_window_px)),
        (ty_center - float(translation_window_px), ty_center + float(translation_window_px)),
    ]
    initial_params = np.array([0.0, 1.0, tx_center, ty_center], dtype=float)
    initial_dice = _dice_for(initial_params, use_full=False)
    result = differential_evolution(
        _objective,
        bounds,
        seed=2,
        popsize=10,
        maxiter=int(maxiter),
        tol=5e-4,
        polish=True,
        workers=1,
        updating="immediate",
    )
    best_params = np.asarray(result.x, dtype=float)
    best_affine = _matrix(float(best_params[0]), float(best_params[1]), float(best_params[2]), float(best_params[3]))
    diagnostics = {
        "initial_dice_optimized_resolution": float(initial_dice),
        "optimized_dice_optimized_resolution": float(-result.fun),
        "optimized_dice_full_tissue": float(_dice_for(best_params, use_full=True)),
        "rotation_deg": float(best_params[0]),
        "scale_multiplier": float(best_params[1]),
        "base_scale": float(base_scale),
        "translation_x": float(best_params[2]),
        "translation_y": float(best_params[3]),
        "fixed_downsample": float(fds),
        "moving_downsample": float(mds),
        "optimize_stride": int(stride),
    }
    return best_affine, diagnostics


def _load_binary_mask(mask: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        arr = imread(str(mask))
    else:
        arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return np.asarray(arr > 0, dtype=bool)


def _tissue_center_translation(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_downsample: float,
    moving_downsample: float,
    scale: float,
) -> tuple[float, float]:
    def _center(mask: np.ndarray, downsample: float) -> np.ndarray:
        y, x = np.nonzero(mask)
        if len(x) == 0:
            return np.zeros(2, dtype=float)
        return np.array([float(np.mean(x)) * downsample, float(np.mean(y)) * downsample], dtype=float)

    fixed_center = _center(fixed_mask, fixed_downsample)
    moving_center = _center(moving_mask, moving_downsample)
    translation = fixed_center - float(scale) * moving_center
    return float(translation[0]), float(translation[1])


def refine_affine_from_centroid_structure(
    fixed_centroids: pd.DataFrame,
    moving_centroids: pd.DataFrame,
    initial_affine: np.ndarray,
    top_k: int = 320,
    match_radius_px: float | None = None,
    residual_filter: bool = True,
    iterations: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Refine affine using centroid spatial structure only, not morphology features."""
    fixed_xy = fixed_centroids[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    moving_xy = moving_centroids[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    affine = np.asarray(initial_affine, dtype=float).copy()
    matches = _empty_centroid_match_frame()
    for _ in range(max(1, int(iterations))):
        predicted = _apply_affine_matrix(moving_xy, affine)
        radius = _centroid_match_radius(fixed_xy, match_radius_px)
        matches = _mutual_nearest_centroid_matches(
            fixed_centroids,
            moving_centroids,
            fixed_xy,
            moving_xy,
            predicted,
            radius,
            top_k=max(int(top_k), 3),
        )
        if len(matches) < 3:
            break
        candidate = _fit_affine_matrix(
            matches[["moving_x", "moving_y"]].to_numpy(dtype=float),
            matches[["fixed_x", "fixed_y"]].to_numpy(dtype=float),
            fallback=affine,
        )
        predicted_match = _apply_affine_matrix(matches[["moving_x", "moving_y"]].to_numpy(dtype=float), candidate)
        residuals = np.linalg.norm(predicted_match - matches[["fixed_x", "fixed_y"]].to_numpy(dtype=float), axis=1)
        matches = matches.copy()
        matches["residual_px"] = residuals
        if residual_filter and len(matches) > 3:
            keep = _robust_residual_mask(residuals, min_keep=3)
            if int(keep.sum()) >= 3:
                matches = matches.loc[keep].reset_index(drop=True)
                candidate = _fit_affine_matrix(
                    matches[["moving_x", "moving_y"]].to_numpy(dtype=float),
                    matches[["fixed_x", "fixed_y"]].to_numpy(dtype=float),
                    fallback=candidate,
                )
                predicted_match = _apply_affine_matrix(matches[["moving_x", "moving_y"]].to_numpy(dtype=float), candidate)
                matches["residual_px"] = np.linalg.norm(
                    predicted_match - matches[["fixed_x", "fixed_y"]].to_numpy(dtype=float),
                    axis=1,
                )
        affine = candidate
    return matches, affine


def build_local_deformation_grid(
    fixed_shape: tuple[int, int],
    inverse_global_affine: np.ndarray,
    inverse_local_affine: "KnnLocalAffineWarp | None",
    grid: int = 16,
) -> pd.DataFrame:
    """Build a local deformation grid sampled in HE level-0 fixed coordinate space."""
    h, w = int(fixed_shape[0]), int(fixed_shape[1])
    grid = max(2, int(grid))
    xs = np.linspace(0.0, max(w - 1, 0), grid)
    ys = np.linspace(0.0, max(h - 1, 0), grid)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    he_xy = np.column_stack([gx.ravel(), gy.ravel()])
    global_mif = _apply_affine_matrix(he_xy, inverse_global_affine)
    if inverse_local_affine is None:
        local_mif = global_mif.copy()
    else:
        local_mif = inverse_local_affine.predict(he_xy)
    out = pd.DataFrame(
        {
            "he_level0_x": he_xy[:, 0],
            "he_level0_y": he_xy[:, 1],
            "mif_level0_x_global_inverse": global_mif[:, 0],
            "mif_level0_y_global_inverse": global_mif[:, 1],
            "mif_level0_x_local_inverse": local_mif[:, 0],
            "mif_level0_y_local_inverse": local_mif[:, 1],
        }
    )
    out["delta_mif_x"] = out["mif_level0_x_local_inverse"] - out["mif_level0_x_global_inverse"]
    out["delta_mif_y"] = out["mif_level0_y_local_inverse"] - out["mif_level0_y_global_inverse"]
    return out


def save_wsi_centroid_registration_result(
    result: WSICentroidRegistrationResult,
    output_dir: str | Path,
    write_preview: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.fixed_centroids.to_csv(output_dir / "fixed_centroids_level0.csv", index=False)
    result.moving_centroids.to_csv(output_dir / "moving_centroids_level0.csv", index=False)
    result.registered_moving_centroids.to_csv(output_dir / "registered_moving_centroids_he_level0.csv", index=False)
    result.matches.to_csv(output_dir / "wsi_centroid_matches.csv", index=False)
    result.local_deformation_grid.to_csv(output_dir / "local_deformation_grid_he_level0.csv", index=False)
    transform_payload = {
        "direction": "mIF_level0_pixel_to_HE_level0_pixel",
        "global_affine_mif_level0_to_he_level0": result.global_affine_mif_to_he.tolist(),
        "inverse_global_affine_he_level0_to_mif_level0": result.inverse_global_affine_he_to_mif.tolist(),
        "local_deformation_grid_coordinate_space": "HE_level0_pixel",
        "fixed_centroid_metadata": result.fixed_metadata.to_json_dict(),
        "moving_centroid_metadata": result.moving_metadata.to_json_dict(),
        "diagnostics": result.diagnostics or {},
    }
    with open(output_dir / "wsi_transform.json", "w", encoding="utf-8") as handle:
        json.dump(transform_payload, handle, indent=2)
    with open(output_dir / "fixed_centroids_metadata.normalized.json", "w", encoding="utf-8") as handle:
        json.dump(result.fixed_metadata.to_json_dict(), handle, indent=2)
    with open(output_dir / "moving_centroids_metadata.normalized.json", "w", encoding="utf-8") as handle:
        json.dump(result.moving_metadata.to_json_dict(), handle, indent=2)
    if write_preview:
        preview = make_centroid_registration_preview(
            result.fixed_centroids[["centroid_x", "centroid_y"]].to_numpy(dtype=float),
            result.registered_moving_centroids[["registered_centroid_x", "registered_centroid_y"]].to_numpy(dtype=float),
            result.fixed_metadata.image_shape_yx,
        )
        imwrite(str(output_dir / "centroid_registration_preview.tif"), preview, photometric="rgb")


def make_centroid_registration_preview(
    fixed_xy: np.ndarray,
    registered_moving_xy: np.ndarray,
    fixed_shape: tuple[int, int],
    max_size: int = 2048,
) -> np.ndarray:
    """Create an RGB density overlay preview: HE fixed red, registered mIF green."""
    h, w = int(fixed_shape[0]), int(fixed_shape[1])
    scale = min(float(max_size) / float(max(h, w, 1)), 1.0)
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    fixed_img = _centroid_density_image(fixed_xy * scale, (out_h, out_w))
    moving_img = _centroid_density_image(registered_moving_xy * scale, (out_h, out_w))
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[..., 0] = fixed_img
    rgb[..., 1] = moving_img
    return rgb


def write_registered_multichannel_wsi(
    result: WSICentroidRegistrationResult,
    moving_wsi_path: str | Path | None,
    output_dir: str | Path | None,
    tile_size: int = 1024,
    channel_batch_size: int = 1,
    channel_region_reader: Any | None = None,
    channel_count: int | None = None,
    output_dtype: Any | None = None,
    output_path: str | Path | None = None,
    progress_path: str | Path | None = None,
) -> None:
    """
    Tile-wise warp multichannel moving WSI into HE level-0 output space.

    ``channel_region_reader`` must read only the requested source region and
    return a 2D array: reader(path, channel_index, x, y, width, height).
    """
    if moving_wsi_path is None:
        raise ValueError("Cannot write registered 8-channel WSI without moving_wsi_path/source_wsi_path.")
    if output_dir is None and output_path is None:
        raise ValueError("Provide output_dir or output_path for registered WSI export.")
    if channel_region_reader is None:
        raise NotImplementedError(
            "Tile-wise multichannel WSI export requires a format-specific channel_region_reader. "
            "The registration transform has been saved; no full multichannel image was loaded."
        )
    if channel_count is None or int(channel_count) <= 0:
        raise ValueError("channel_count must be provided for tile-wise multichannel WSI export.")

    moving_wsi_path = Path(moving_wsi_path)
    if output_path is None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "registered_mif_to_he_level0.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = result.fixed_metadata.image_shape_yx
    c_count = int(channel_count)
    tile_size = max(64, int(tile_size))
    batch = max(1, int(channel_batch_size))
    if output_dtype is None:
        sample = np.asarray(channel_region_reader(moving_wsi_path, 0, 0, 0, 1, 1))
        output_dtype = sample.dtype if sample.size else np.uint16

    import tifffile
    from scipy.ndimage import map_coordinates

    out = tifffile.memmap(
        str(output_path),
        shape=(c_count, h, w),
        dtype=np.dtype(output_dtype),
        bigtiff=True,
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    moving_w = int(result.moving_metadata.image_width)
    moving_h = int(result.moving_metadata.image_height)
    total_tiles = int(np.ceil(h / tile_size)) * int(np.ceil(w / tile_size))
    done_tiles = 0
    progress_path = Path(progress_path) if progress_path is not None else None
    try:
        for y0 in range(0, h, tile_size):
            y1 = min(h, y0 + tile_size)
            for x0 in range(0, w, tile_size):
                x1 = min(w, x0 + tile_size)
                yy, xx = np.meshgrid(
                    np.arange(y0, y1, dtype=float),
                    np.arange(x0, x1, dtype=float),
                    indexing="ij",
                )
                he_xy = np.column_stack([xx.ravel(), yy.ravel()])
                src_xy = _predict_he_level0_to_mif_level0(result, he_xy).reshape(y1 - y0, x1 - x0, 2)
                valid = (
                    (src_xy[..., 0] >= 0)
                    & (src_xy[..., 0] < moving_w)
                    & (src_xy[..., 1] >= 0)
                    & (src_xy[..., 1] < moving_h)
                )
                if not np.any(valid):
                    out[:, y0:y1, x0:x1] = 0
                    continue
                sx0 = max(0, int(np.floor(np.nanmin(src_xy[..., 0][valid]))) - 2)
                sy0 = max(0, int(np.floor(np.nanmin(src_xy[..., 1][valid]))) - 2)
                sx1 = min(moving_w, int(np.ceil(np.nanmax(src_xy[..., 0][valid]))) + 3)
                sy1 = min(moving_h, int(np.ceil(np.nanmax(src_xy[..., 1][valid]))) + 3)
                read_w = max(1, sx1 - sx0)
                read_h = max(1, sy1 - sy0)
                coords_y = src_xy[..., 1] - float(sy0)
                coords_x = src_xy[..., 0] - float(sx0)
                for c0 in range(0, c_count, batch):
                    c1 = min(c_count, c0 + batch)
                    for channel in range(c0, c1):
                        source = np.asarray(channel_region_reader(moving_wsi_path, channel, sx0, sy0, read_w, read_h))
                        warped = map_coordinates(
                            source.astype(float, copy=False),
                            [coords_y, coords_x],
                            order=1,
                            mode="constant",
                            cval=0.0,
                            prefilter=False,
                        )
                        if np.issubdtype(np.dtype(output_dtype), np.integer):
                            info = np.iinfo(np.dtype(output_dtype))
                            warped = np.clip(np.rint(warped), info.min, info.max)
                        warped = np.where(valid, warped, 0)
                        out[channel, y0:y1, x0:x1] = warped.astype(output_dtype, copy=False)
                done_tiles += 1
                if progress_path is not None and (done_tiles == 1 or done_tiles == total_tiles or done_tiles % 10 == 0):
                    progress_path.write_text(
                        json.dumps(
                            {
                                "status": "running" if done_tiles < total_tiles else "done",
                                "done_tiles": int(done_tiles),
                                "total_tiles": int(total_tiles),
                                "tile_size": int(tile_size),
                                "channel_count": int(c_count),
                                "output_path": str(output_path),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if done_tiles % 10 == 0:
                    import gc

                    gc.collect()
            out.flush()
        out.flush()
    finally:
        mmap = getattr(out, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del out


def _predict_he_level0_to_mif_level0(
    result: WSICentroidRegistrationResult,
    he_xy: np.ndarray,
) -> np.ndarray:
    if result.inverse_local_affine_warp is not None:
        return result.inverse_local_affine_warp.predict(he_xy)
    return _apply_affine_matrix(he_xy, result.inverse_global_affine_he_to_mif)



def _normalize_centroid_metadata_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize current and older segmentation metadata into the required WSI schema."""
    out = dict(payload)
    if "source_wsi_path" not in out:
        out["source_wsi_path"] = out.get("input") or out.get("wsi_path") or out.get("source_path")
    if "source_mask_path" not in out:
        out["source_mask_path"] = out.get("mask_path") or out.get("source_mask")
    if "coordinate_space" not in out:
        level = out.get("level")
        if level is not None and int(level) == 0:
            out["coordinate_space"] = "level0_pixel"
        else:
            out["coordinate_space"] = "mask_pixel"
    if "origin_x" not in out:
        out["origin_x"] = out.get("x", out.get("region_x", 0))
    if "origin_y" not in out:
        out["origin_y"] = out.get("y", out.get("region_y", 0))
    if "downsample" not in out:
        level = out.get("level")
        out["downsample"] = 1.0 if level is None or int(level) == 0 else out.get("level_downsample", 1.0)
    if "image_width" not in out:
        out["image_width"] = out.get("width") or out.get("wsi_width") or out.get("image_w")
    if "image_height" not in out:
        out["image_height"] = out.get("height") or out.get("wsi_height") or out.get("image_h")
    if "mpp_reliable" not in out:
        out["mpp_reliable"] = out.get("mpp_x") is not None and out.get("mpp_y") is not None
    if "channel" not in out:
        cellpose = out.get("cellpose") if isinstance(out.get("cellpose"), dict) else {}
        out["channel"] = out.get("source_channel") or cellpose.get("source_channel") or out.get("channel_name") or out.get("channel", "")
    if "segmentation_method" not in out:
        if "cellpose" in out:
            out["segmentation_method"] = "Cellpose"
        elif "cellvit" in str(out).lower():
            out["segmentation_method"] = "CellViT"
        else:
            out["segmentation_method"] = out.get("method", "unknown")
    return out


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "reliable"}


def _coerce_centroids_and_metadata(
    centroids: pd.DataFrame | str | Path,
    metadata: WSICentroidMetadata | dict[str, Any] | str | Path | None,
    label: str,
) -> tuple[pd.DataFrame, WSICentroidMetadata]:
    if isinstance(centroids, (str, Path)):
        csv_path = Path(centroids)
        if metadata is None or isinstance(metadata, (str, Path)):
            return load_wsi_centroids(csv_path, metadata)
        df = pd.read_csv(csv_path)
    else:
        df = centroids.copy()
    if metadata is None:
        raise ValueError(f"{label} WSI centroid DataFrame requires explicit metadata JSON/dict.")
    if isinstance(metadata, WSICentroidMetadata):
        meta = metadata
    elif isinstance(metadata, (str, Path)):
        meta = WSICentroidMetadata.from_json(metadata)
    elif isinstance(metadata, dict):
        meta = WSICentroidMetadata.from_mapping(metadata, source_name=f"{label} metadata")
    else:
        raise TypeError(f"Unsupported {label} metadata type: {type(metadata)!r}")
    return df, meta


def _has_reliable_mpp(metadata: WSICentroidMetadata) -> bool:
    return (
        bool(metadata.mpp_reliable)
        and metadata.mpp_x is not None
        and metadata.mpp_y is not None
        and float(metadata.mpp_x) > 0
        and float(metadata.mpp_y) > 0
    )


def _robust_center(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    if len(points_xy) == 0:
        return np.zeros(2, dtype=float)
    return np.nanmedian(points_xy, axis=0)


def _robust_extent(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    if len(points_xy) < 2:
        return np.ones(2, dtype=float)
    lo = np.nanpercentile(points_xy, 1.0, axis=0)
    hi = np.nanpercentile(points_xy, 99.0, axis=0)
    return np.maximum(hi - lo, 1.0)


def _robust_extent_scale(fixed_xy: np.ndarray, moving_xy: np.ndarray) -> tuple[float, float]:
    fixed_extent = _robust_extent(fixed_xy)
    moving_extent = _robust_extent(moving_xy)
    scale = fixed_extent / np.maximum(moving_extent, 1.0)
    scale = np.clip(scale, 0.05, 20.0)
    return float(scale[0]), float(scale[1])


def _apply_affine_matrix(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    was_1d = points.ndim == 1
    if was_1d:
        points = points[None, :]
    ones = np.ones((len(points), 1), dtype=float)
    hom = np.hstack([points, ones])
    out = hom @ np.asarray(matrix, dtype=float).T
    out_xy = out[:, :2] / np.maximum(out[:, 2:3], 1e-12)
    return out_xy[0] if was_1d else out_xy


def _invert_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    try:
        inv = np.linalg.inv(np.asarray(matrix, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError("WSI affine transform is singular and cannot be inverted.") from exc
    if not np.all(np.isfinite(inv)):
        raise ValueError("WSI affine inverse contains non-finite values.")
    return inv


def _fit_affine_matrix(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    params = _fit_weighted_affine_params(
        np.asarray(source_xy, dtype=float),
        np.asarray(target_xy, dtype=float),
        fallback=_affine_matrix_to_params(fallback),
    )
    return _affine_params_to_matrix(params)


def _affine_matrix_to_params(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return np.array(
        [
            [matrix[0, 0], matrix[1, 0]],
            [matrix[0, 1], matrix[1, 1]],
            [matrix[0, 2], matrix[1, 2]],
        ],
        dtype=float,
    )


def _affine_params_to_matrix(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    return np.array(
        [
            [params[0, 0], params[1, 0], params[2, 0]],
            [params[0, 1], params[1, 1], params[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _centroid_match_radius(fixed_xy: np.ndarray, explicit_radius: float | None) -> float:
    if explicit_radius is not None and float(explicit_radius) > 0:
        return float(explicit_radius)
    if len(fixed_xy) < 2:
        return 256.0
    tree = cKDTree(fixed_xy)
    dists, _ = tree.query(fixed_xy, k=min(2, len(fixed_xy)))
    nn = np.asarray(dists, dtype=float)
    if nn.ndim == 2 and nn.shape[1] > 1:
        base = float(np.nanmedian(nn[:, 1]))
    else:
        base = 32.0
    return float(np.clip(max(64.0, base * 8.0), 64.0, 2048.0))


def _mutual_nearest_centroid_matches(
    fixed_centroids: pd.DataFrame,
    moving_centroids: pd.DataFrame,
    fixed_xy: np.ndarray,
    moving_xy: np.ndarray,
    predicted_moving_xy: np.ndarray,
    radius: float,
    top_k: int,
) -> pd.DataFrame:
    fixed_tree = cKDTree(fixed_xy)
    moving_tree = cKDTree(predicted_moving_xy)
    dist_m_to_f, idx_fixed = fixed_tree.query(predicted_moving_xy, k=1, distance_upper_bound=radius)
    _, idx_moving_back = moving_tree.query(fixed_xy, k=1, distance_upper_bound=radius)
    rows: list[dict[str, Any]] = []
    for idx_moving, (dist, idx_f) in enumerate(zip(dist_m_to_f, idx_fixed)):
        if not np.isfinite(dist) or int(idx_f) >= len(fixed_xy):
            continue
        if int(idx_moving_back[int(idx_f)]) != int(idx_moving):
            continue
        moving_raw = moving_xy[idx_moving]
        moving_pred = predicted_moving_xy[idx_moving]
        fixed = fixed_xy[int(idx_f)]
        rows.append(
            {
                "idx1": int(idx_f),
                "idx2": int(idx_moving),
                "distance": float(dist),
                "fixed_x": float(fixed[0]),
                "fixed_y": float(fixed[1]),
                "moving_x": float(moving_raw[0]),
                "moving_y": float(moving_raw[1]),
                "predicted_moving_x": float(moving_pred[0]),
                "predicted_moving_y": float(moving_pred[1]),
                "dx": float(fixed[0] - moving_pred[0]),
                "dy": float(fixed[1] - moving_pred[1]),
                "cell_id_1": float(fixed_centroids.iloc[int(idx_f)]["cell_id"]),
                "cell_id_2": float(moving_centroids.iloc[int(idx_moving)]["cell_id"]),
            }
        )
    if not rows:
        return _empty_centroid_match_frame()
    out = pd.DataFrame(rows).sort_values("distance", ascending=True).reset_index(drop=True)
    return out.head(max(3, int(top_k))).copy()


def _robust_residual_mask(residuals: np.ndarray, min_keep: int = 3) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) <= min_keep:
        return np.ones(len(residuals), dtype=bool)
    median = float(np.nanmedian(residuals))
    mad = float(np.nanmedian(np.abs(residuals - median)))
    threshold = max(median + 3.0 * max(1.4826 * mad, 1.0), 2.0)
    keep = residuals <= threshold
    if int(keep.sum()) < int(min_keep):
        order = np.argsort(residuals)
        keep = np.zeros(len(residuals), dtype=bool)
        keep[order[: int(min_keep)]] = True
    return keep


def _empty_centroid_match_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx1",
            "idx2",
            "distance",
            "fixed_x",
            "fixed_y",
            "moving_x",
            "moving_y",
            "predicted_moving_x",
            "predicted_moving_y",
            "dx",
            "dy",
            "residual_px",
            "cell_id_1",
            "cell_id_2",
        ]
    )


def _centroid_density_image(points_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    img = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=float)
    if pts.size == 0:
        return img.astype(np.uint8)
    x = np.rint(pts[:, 0]).astype(int)
    y = np.rint(pts[:, 1]).astype(int)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(valid):
        return img.astype(np.uint8)
    np.add.at(img, (y[valid], x[valid]), 1.0)
    try:
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, sigma=1.2)
    except Exception:
        pass
    vmax = float(np.percentile(img[img > 0], 99.0)) if np.any(img > 0) else 0.0
    if vmax <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    return np.clip(img * (255.0 / vmax), 0, 255).astype(np.uint8)


def find_wsi_landmark_matches(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    top_k: int = 320,
    max_angle_deg: float = 2.0,
    coverage_grid: int = 4,
    neighbor_k: int = 8,
    neighbor_profile_threshold: float = 0.35,
    max_cell_orientation_diff_deg: float = 10.0,
    min_orientation_eccentricity: float = 0.15,
) -> pd.DataFrame:
    """Find one-to-one real-cell landmark pairs for large-translation WSI masks."""
    feature_columns = tuple(
        col for col in MatchingConfig().feature_columns
        if col in feats1.columns and col in feats2.columns
    )
    if not feature_columns:
        return _empty_wsi_match_frame()

    f1, f2 = _robust_standardize_feature_tables(feats1, feats2, feature_columns)
    if len(f1) == 0 or len(f2) == 0:
        return _empty_wsi_match_frame()

    neighbors_per_cell = min(20, len(f2))
    tree = cKDTree(f2)
    distances, idxs2 = tree.query(f1, k=neighbors_per_cell)
    distances = np.asarray(distances, dtype=float)
    idxs2 = np.asarray(idxs2, dtype=int)
    if distances.ndim == 1:
        distances = distances[:, None]
        idxs2 = idxs2[:, None]

    idxs1 = np.repeat(np.arange(len(feats1), dtype=int), neighbors_per_cell)
    idxs2_flat = idxs2.reshape(-1)
    distances_flat = distances.reshape(-1)
    valid = np.isfinite(distances_flat) & (idxs2_flat >= 0)
    idxs1 = idxs1[valid]
    idxs2_flat = idxs2_flat[valid]
    distances_flat = distances_flat[valid]
    if len(idxs1) < 3:
        return _empty_wsi_match_frame()

    max_candidates = min(len(idxs1), max(200_000, int(top_k) * 500))
    if len(idxs1) > max_candidates:
        keep = np.argpartition(distances_flat, max_candidates - 1)[:max_candidates]
        idxs1 = idxs1[keep]
        idxs2_flat = idxs2_flat[keep]
        distances_flat = distances_flat[keep]

    pts1 = feats1.iloc[idxs1][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts2 = feats2.iloc[idxs2_flat][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    displacements = pts1 - pts2

    candidate_mask, median_disp = select_main_displacement_cluster(displacements)
    if int(candidate_mask.sum()) < 3:
        return _empty_wsi_match_frame()

    idxs1 = idxs1[candidate_mask]
    idxs2_flat = idxs2_flat[candidate_mask]
    distances_flat = distances_flat[candidate_mask]
    displacements = displacements[candidate_mask]

    parallel_mask = filter_parallel_displacements(displacements, median_disp, max_angle_deg=max_angle_deg)
    if int(parallel_mask.sum()) < 3:
        return _empty_wsi_match_frame()

    idxs1 = idxs1[parallel_mask]
    idxs2_flat = idxs2_flat[parallel_mask]
    distances_flat = distances_flat[parallel_mask]
    displacements = displacements[parallel_mask]
    length_mask, _ = filter_long_match_lines(displacements)
    if int(length_mask.sum()) < 3:
        return _empty_wsi_match_frame()

    idxs1 = idxs1[length_mask]
    idxs2_flat = idxs2_flat[length_mask]
    distances_flat = distances_flat[length_mask]
    displacements = displacements[length_mask]
    residuals = np.linalg.norm(displacements - median_disp[None, :], axis=1)

    order = np.lexsort((distances_flat, residuals))
    used1: set[int] = set()
    used2: set[int] = set()
    rows: list[tuple[int, int, float, float, float, float, float, float, float]] = []
    for pos in order:
        i = int(idxs1[pos])
        j = int(idxs2_flat[pos])
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        dx, dy = displacements[pos]
        line_length = float(np.linalg.norm(displacements[pos]))
        rows.append((
            i,
            j,
            float(distances_flat[pos]),
            float(dx),
            float(dy),
            line_length,
            float(residuals[pos]),
            float(feats1.iloc[i]["cell_id"]),
            float(feats2.iloc[j]["cell_id"]),
        ))

    if not rows:
        return _empty_wsi_match_frame()
    candidates = pd.DataFrame(
        rows,
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "match_line_length_px",
            "displacement_residual_px",
            "cell_id_1",
            "cell_id_2",
        ],
    )
    candidates = add_cell_orientation_scores(candidates, feats1, feats2)
    if "orientation_diff_deg" in candidates.columns:
        reliable_orientation = (
            np.minimum(
                candidates["eccentricity_1"].to_numpy(dtype=float),
                candidates["eccentricity_2"].to_numpy(dtype=float),
            )
            >= float(min_orientation_eccentricity)
        )
        orientation_ok = (~reliable_orientation) | (
            candidates["orientation_diff_deg"].to_numpy(dtype=float) <= float(max_cell_orientation_diff_deg)
        )
        candidates = candidates[orientation_ok].copy()
        if len(candidates) < 3:
            return _empty_wsi_match_frame()

    candidates = add_neighbor_profile_scores(
        candidates,
        feats1,
        feats2,
        neighbor_k=neighbor_k,
    )
    candidates = candidates[
        candidates["neighbor_profile_diff"] <= float(neighbor_profile_threshold)
    ].copy()
    if len(candidates) < 3:
        return _empty_wsi_match_frame()
    matches = select_landmarks_with_grid_coverage(candidates, feats1, top_k=top_k, grid=coverage_grid)
    if matches.empty:
        return _empty_wsi_match_frame()
    matches["idx1"] = matches["idx1"].astype(int)
    matches["idx2"] = matches["idx2"].astype(int)
    return matches


def run_wsi_mask_registration(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_image: np.ndarray | None = None,
    moving_image: np.ndarray | None = None,
    top_k: int = 320,
    max_angle_deg: float = 2.0,
    min_area: int = 0,
    max_area: int = 0,
    use_knn_local_affine: bool = True,
    knn_k: int = 8,
    knn_power: float = 2.0,
    residual_filter: bool = True,
    warp_registered_mask: bool = True,
) -> WSIRegistrationResult:
    feature_config = CellFeaturesConfig(
        min_area=min_area if min_area > 0 else None,
        max_area=max_area if max_area > 0 else None,
        topology_neighbor_k=5,
    )
    feats1 = compute_cell_features(np.asarray(fixed_mask), feature_config)
    feats2 = compute_cell_features(np.asarray(moving_mask), feature_config)
    result = run_wsi_feature_registration(
        fixed_features=feats1,
        moving_features=feats2,
        fixed_shape=fixed_mask.shape[:2],
        top_k=top_k,
        max_angle_deg=max_angle_deg,
        min_area=min_area,
        max_area=max_area,
        use_knn_local_affine=use_knn_local_affine,
        knn_k=knn_k,
        knn_power=knn_power,
        residual_filter=residual_filter,
    )

    registered_image = None
    registered_mask = None
    if moving_image is not None or warp_registered_mask:
        fixed_pts, moving_pts = _matched_xy(feats1, feats2, result.matches)
        if use_knn_local_affine and len(result.matches) >= 3:
            inverse_warp = KnnLocalAffineWarp(
                fixed_pts,
                moving_pts,
                k=knn_k,
                power=knn_power,
            )
            if moving_image is not None:
                warped = warp_array_with_knn_local_affine(
                    np.asarray(moving_image),
                    inverse_warp,
                    output_shape=fixed_mask.shape[:2],
                    order=1,
                )
                registered_image = cast_warped_like_original(warped, np.asarray(moving_image).dtype)
            if warp_registered_mask:
                registered_mask = np.rint(
                    warp_array_with_knn_local_affine(
                        np.asarray(moving_mask).astype(np.int32, copy=False),
                        inverse_warp,
                        output_shape=fixed_mask.shape[:2],
                        order=0,
                    )
                ).astype(np.int32)
        else:
            local_grid = result.local_translation_grid
            if local_grid is None:
                local_grid = estimate_local_translation_grid(
                    result.matches,
                    fixed_mask.shape[:2],
                    grid=4,
                    fallback_translation=result.transform.translation,
                )
            if moving_image is not None:
                warped = warp_array_with_translation_grid(
                    np.asarray(moving_image),
                    local_grid,
                    output_shape=fixed_mask.shape[:2],
                    order=1,
                )
                registered_image = cast_warped_like_original(warped, np.asarray(moving_image).dtype)
            if warp_registered_mask:
                registered_mask = np.rint(
                    warp_array_with_translation_grid(
                        np.asarray(moving_mask).astype(np.int32, copy=False),
                        local_grid,
                        output_shape=fixed_mask.shape[:2],
                        order=0,
                    )
                ).astype(np.int32)

    result.registered_image = registered_image
    result.registered_mask = registered_mask
    return result


def run_wsi_feature_registration(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    fixed_shape: tuple[int, int],
    top_k: int = 320,
    max_angle_deg: float = 2.0,
    min_area: int = 0,
    max_area: int = 0,
    use_knn_local_affine: bool = True,
    knn_k: int = 8,
    knn_power: float = 2.0,
    residual_filter: bool = True,
) -> WSIRegistrationResult:
    feats1 = normalize_wsi_feature_table(fixed_features, fixed_shape, min_area=min_area, max_area=max_area)
    feats2 = normalize_wsi_feature_table(moving_features, fixed_shape, min_area=min_area, max_area=max_area)
    matches = find_wsi_landmark_matches(feats1, feats2, top_k=top_k, max_angle_deg=max_angle_deg)
    if len(matches) < 3:
        raise ValueError(f"WSI landmark search found {len(matches)} pairs; need at least 3.")

    transform = estimate_wsi_translation_from_matches(matches)
    residuals = compute_match_residuals(feats1, feats2, matches, transform)
    matches = matches.copy()
    matches["residual_px"] = residuals
    if residual_filter:
        matches = filter_wsi_landmark_residuals(matches)
        transform = estimate_wsi_translation_from_matches(matches)
        residuals = compute_match_residuals(feats1, feats2, matches, transform)
        matches = matches.copy()
        matches["residual_px"] = residuals

    fixed_pts, moving_pts = _matched_xy(feats1, feats2, matches)
    local_grid = None
    local_affine = None
    if use_knn_local_affine and len(matches) >= 3:
        local_affine = KnnLocalAffineWarp(
            moving_pts,
            fixed_pts,
            k=knn_k,
            power=knn_power,
        )
    else:
        local_grid = estimate_local_translation_grid(
            matches,
            fixed_shape,
            grid=4,
            fallback_translation=transform.translation,
        )
    return WSIRegistrationResult(
        matches=matches,
        transform=transform,
        residuals=residuals,
        registered_image=None,
        registered_mask=None,
        local_translation_grid=local_grid,
        local_affine_warp=local_affine,
    )


def normalize_wsi_feature_table(
    features: pd.DataFrame,
    image_shape: tuple[int, int],
    min_area: int = 0,
    max_area: int = 0,
) -> pd.DataFrame:
    df = features.copy()
    rename = {
        "label": "cell_id",
        "id": "cell_id",
        "x": "centroid_x",
        "y": "centroid_y",
        "centroid-1": "centroid_x",
        "centroid-0": "centroid_y",
    }
    df = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
    required = {"centroid_x", "centroid_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature table must include centroid_x and centroid_y columns, got {list(df.columns)}.")

    if "cell_id" not in df.columns:
        df["cell_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
    if "area" not in df.columns:
        df["area"] = 1.0

    numeric_cols = ["cell_id", "centroid_x", "centroid_y", "area"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df["centroid_x"]) & np.isfinite(df["centroid_y"]) & np.isfinite(df["area"])].copy()
    df["area"] = np.maximum(df["area"].to_numpy(dtype=float), 1.0)

    if min_area > 0:
        df = df[df["area"] >= float(min_area)]
    if max_area > 0:
        df = df[df["area"] <= float(max_area)]

    equiv = np.sqrt(4.0 * df["area"].to_numpy(dtype=float) / np.pi)
    defaults = {
        "equivalent_diameter": equiv,
        "perimeter": np.pi * equiv,
        "roundness": np.ones(len(df), dtype=float),
        "eccentricity": np.zeros(len(df), dtype=float),
        "solidity": np.ones(len(df), dtype=float),
        "major_axis_length": equiv,
        "minor_axis_length": equiv,
        "aspect_ratio": np.ones(len(df), dtype=float),
        "elongation": np.zeros(len(df), dtype=float),
        "orientation": np.zeros(len(df), dtype=float),
        "axis_vec_x": np.ones(len(df), dtype=float),
        "axis_vec_y": np.zeros(len(df), dtype=float),
    }
    for idx in range(7):
        defaults[f"hu_{idx}"] = np.zeros(len(df), dtype=float)
    for col, values in defaults.items():
        if col not in df.columns:
            df[col] = values
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(pd.Series(values, index=df.index))

    h, w = int(image_shape[0]), int(image_shape[1])
    df["pos_x_norm"] = df["centroid_x"] / float(max(w, 1))
    df["pos_y_norm"] = df["centroid_y"] / float(max(h, 1))
    return df.reset_index(drop=True)


def _roundness(area: np.ndarray, perimeter: np.ndarray) -> np.ndarray:
    perimeter_safe = np.where(perimeter == 0, np.nan, perimeter)
    return np.nan_to_num(4.0 * np.pi * area / (perimeter_safe**2))


def _safe_ratio(numer: np.ndarray, denom: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    denom_safe = np.where(np.abs(denom) <= 1e-6, np.nan, denom)
    return np.nan_to_num(numer / denom_safe, nan=fill_value, posinf=fill_value, neginf=fill_value)


def load_cellvit_json_features(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cells = payload.get("cells", payload) if isinstance(payload, dict) else payload
    if not isinstance(cells, list):
        raise ValueError(f"CellViT JSON {path} does not contain a cells list.")

    rows = []
    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or "centroid" not in cell:
            continue
        centroid = cell["centroid"]
        if len(centroid) < 2:
            continue
        row = _cellvit_contour_region_features(cell.get("contour"))
        if not row:
            area = _contour_area(cell.get("contour"))
            if area <= 0.0:
                area = _bbox_area(cell.get("bbox"))
            row = {"area": max(float(area), 1.0)}
        row.update(
            {
                "cell_id": idx,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cellvit_contour_region_features(contour: object) -> dict[str, float]:
    if not contour:
        return {}
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return {}
    area = max(_contour_area(pts), 1.0)
    closed = np.vstack([pts, pts[0]])
    deltas = np.diff(closed, axis=0)
    perimeter = float(np.sum(np.linalg.norm(deltas, axis=1)))

    centered = pts - np.mean(pts, axis=0, keepdims=True)
    if len(pts) >= 3:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvec = eigvecs[:, order[0]]
        major = float(4.0 * np.sqrt(max(eigvals[0], 1e-6)))
        minor = float(4.0 * np.sqrt(max(eigvals[-1], 1e-6)))
        orientation = float(np.arctan2(eigvec[1], eigvec[0]))
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-6))))
    else:
        major = minor = float(np.sqrt(area))
        orientation = 0.0
        eccentricity = 0.0

    row = {
        "area": area,
        "perimeter": perimeter,
        "eccentricity": eccentricity,
        "solidity": 1.0,
        "major_axis_length": major,
        "minor_axis_length": minor,
        "orientation": orientation,
    }
    row["roundness"] = float(_roundness(np.array([area]), np.array([perimeter]))[0])
    row["aspect_ratio"] = float(_safe_ratio(np.array([major]), np.array([minor]))[0])
    row["elongation"] = float(np.clip(1.0 - _safe_ratio(np.array([minor]), np.array([major]))[0], 0.0, 1.0))
    row["equivalent_diameter"] = float(np.sqrt(4.0 * area / np.pi))
    row["axis_vec_x"] = float(np.cos(orientation))
    row["axis_vec_y"] = float(np.sin(orientation))
    for idx in range(7):
        row[f"hu_{idx}"] = 0.0
    return row


def _contour_area(contour: object) -> float:
    if contour is None:
        return 0.0
    pts = np.asarray(contour, dtype=float)
    if pts.size == 0:
        return 0.0
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _bbox_area(bbox: object) -> float:
    if not bbox:
        return 0.0
    arr = np.asarray(bbox, dtype=float)
    if arr.shape != (2, 2):
        return 0.0
    return float(abs(arr[1, 0] - arr[0, 0]) * abs(arr[1, 1] - arr[0, 1]))


def select_main_displacement_cluster(displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bin_size = 512.0
    bins = np.floor(displacements / bin_size).astype(np.int64)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    best_bin_idx = int(np.argmax(counts))
    seed_mask = inverse == best_bin_idx
    if int(seed_mask.sum()) < 3:
        return seed_mask, np.median(displacements[seed_mask], axis=0)

    seed_median = np.median(displacements[seed_mask], axis=0)
    seed_residuals = np.linalg.norm(displacements - seed_median[None, :], axis=1)
    broad_mask = seed_residuals <= bin_size * 1.5
    if int(broad_mask.sum()) < 3:
        return broad_mask, seed_median

    refined_median = np.median(displacements[broad_mask], axis=0)
    refined_residuals = np.linalg.norm(displacements - refined_median[None, :], axis=1)
    inlier_residuals = refined_residuals[broad_mask]
    med_residual = float(np.median(inlier_residuals))
    mad_residual = float(np.median(np.abs(inlier_residuals - med_residual)))
    threshold = max(50.0, med_residual + 3.0 * max(1.4826 * mad_residual, 1.0))
    return refined_residuals <= threshold, refined_median


def estimate_wsi_translation_from_matches(matches: pd.DataFrame) -> RigidTransform:
    """Estimate WSI translation robustly from matched real-cell displacement medians."""
    if matches.empty:
        raise ValueError("No WSI landmark matches provided.")
    if not {"dx", "dy"}.issubset(matches.columns):
        raise ValueError("WSI matches must contain dx and dy columns.")
    translation = np.array(
        [
            float(np.median(matches["dx"].to_numpy(dtype=float))),
            float(np.median(matches["dy"].to_numpy(dtype=float))),
        ],
        dtype=float,
    )
    return RigidTransform(rotation=np.eye(2, dtype=float), translation=translation)


def estimate_local_translation_grid(
    matches: pd.DataFrame,
    image_shape: tuple[int, int],
    grid: int = 4,
    fallback_translation: np.ndarray | None = None,
) -> np.ndarray:
    grid = max(1, int(grid))
    if fallback_translation is None:
        fallback_translation = estimate_wsi_translation_from_matches(matches).translation
    local = np.full((grid, grid, 2), np.nan, dtype=float)
    if {"wsi_grid_x", "wsi_grid_y"}.issubset(matches.columns):
        gx = matches["wsi_grid_x"].to_numpy(dtype=int)
        gy = matches["wsi_grid_y"].to_numpy(dtype=int)
    else:
        h, w = image_shape[:2]
        gx = np.clip(np.floor(matches["fixed_x"].to_numpy(dtype=float) * grid / max(w, 1)).astype(int), 0, grid - 1)
        gy = np.clip(np.floor(matches["fixed_y"].to_numpy(dtype=float) * grid / max(h, 1)).astype(int), 0, grid - 1)

    for y in range(grid):
        for x in range(grid):
            in_cell = (gx == x) & (gy == y)
            if np.any(in_cell):
                local[y, x, 0] = float(np.median(matches.loc[in_cell, "dx"].to_numpy(dtype=float)))
                local[y, x, 1] = float(np.median(matches.loc[in_cell, "dy"].to_numpy(dtype=float)))

    missing = ~np.isfinite(local[..., 0])
    if np.any(missing):
        known_yx = np.argwhere(~missing)
        if len(known_yx) == 0:
            local[:, :, 0] = float(fallback_translation[0])
            local[:, :, 1] = float(fallback_translation[1])
        else:
            for y, x in np.argwhere(missing):
                nearest_idx = int(np.argmin(np.sum((known_yx - np.array([y, x])) ** 2, axis=1)))
                nearest_y, nearest_x = known_yx[nearest_idx]
                local[y, x] = local[nearest_y, nearest_x]
    return local


def filter_wsi_landmark_residuals(
    matches: pd.DataFrame,
    min_matches: int = 3,
    sigma: float = 3.0,
    min_threshold_px: float = 2.0,
) -> pd.DataFrame:
    """Drop landmarks whose displacement is far from the robust global displacement."""
    if matches.empty or len(matches) <= min_matches:
        return matches.copy()
    if not {"dx", "dy"}.issubset(matches.columns):
        return matches.copy()

    disp = matches[["dx", "dy"]].to_numpy(dtype=float)
    center = np.median(disp, axis=0)
    residuals = np.linalg.norm(disp - center[None, :], axis=1)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = max(1.4826 * mad, 1.0)
    threshold = max(float(min_threshold_px), median + float(sigma) * robust_scale)
    keep = residuals <= threshold
    if int(keep.sum()) < int(min_matches):
        return matches.copy()
    out = matches.loc[keep].copy().reset_index(drop=True)
    out["knn_filter_residual_px"] = residuals[keep]
    out["knn_filter_threshold_px"] = threshold
    return out


def _matched_xy(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    matches: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    idx1 = matches["idx1"].to_numpy(dtype=int)
    idx2 = matches["idx2"].to_numpy(dtype=int)
    fixed_xy = feats1.iloc[idx1][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    moving_xy = feats2.iloc[idx2][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    return fixed_xy, moving_xy


def _translation_affine_params(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    translation = np.median(target_xy - source_xy, axis=0)
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [float(translation[0]), float(translation[1])],
        ],
        dtype=float,
    )


def _fit_weighted_affine_params(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: np.ndarray | None = None,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if source_xy.shape != target_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both be shaped (N, 2).")
    if len(source_xy) < 3:
        return fallback.copy() if fallback is not None else _translation_affine_params(source_xy, target_xy)

    design = np.column_stack([source_xy, np.ones(len(source_xy), dtype=float)])
    if np.linalg.matrix_rank(design) < 3:
        return fallback.copy() if fallback is not None else _translation_affine_params(source_xy, target_xy)
    rhs = target_xy
    if weights is not None:
        w = np.sqrt(np.maximum(np.asarray(weights, dtype=float), 1e-12))
        design = design * w[:, None]
        rhs = rhs * w[:, None]
    try:
        params, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return fallback.copy() if fallback is not None else _translation_affine_params(source_xy, target_xy)
    if rank < 3 or not np.all(np.isfinite(params)):
        return fallback.copy() if fallback is not None else _translation_affine_params(source_xy, target_xy)
    return params


def _apply_affine_params(points_xy: np.ndarray, params: np.ndarray) -> np.ndarray:
    design = np.column_stack([points_xy, np.ones(len(points_xy), dtype=float)])
    return design @ params


def knn_local_affine_leave_one_out_residuals(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """Estimate landmark prediction residuals without letting each point use itself."""
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if source_xy.shape != target_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both be shaped (N, 2).")
    n_points = len(source_xy)
    if n_points < 4:
        return np.full(n_points, np.nan, dtype=float)

    k_eff = max(3, min(int(k), n_points - 1))
    tree = cKDTree(source_xy)
    global_params = _fit_weighted_affine_params(
        source_xy,
        target_xy,
        fallback=_translation_affine_params(source_xy, target_xy),
    )
    residuals = np.empty(n_points, dtype=float)
    for idx in range(n_points):
        dists, nbrs = tree.query(source_xy[idx], k=min(k_eff + 1, n_points))
        dists = np.atleast_1d(np.asarray(dists, dtype=float))
        nbrs = np.atleast_1d(np.asarray(nbrs, dtype=int))
        keep = nbrs != idx
        nbrs = nbrs[keep][:k_eff]
        dists = dists[keep][:k_eff]
        if len(nbrs) < 3:
            pred = _apply_affine_params(source_xy[idx:idx + 1], global_params)[0]
        else:
            weights = 1.0 / np.power(np.maximum(dists, 1e-6), max(float(power), 0.0))
            params = _fit_weighted_affine_params(
                source_xy[nbrs],
                target_xy[nbrs],
                weights=weights,
                fallback=global_params,
            )
            pred = _apply_affine_params(source_xy[idx:idx + 1], params)[0]
        residuals[idx] = float(np.linalg.norm(pred - target_xy[idx]))
    return residuals


@dataclass
class KnnLocalAffineWarp:
    """KNN local affine mapping from source_xy to target_xy."""

    source_xy: np.ndarray
    target_xy: np.ndarray
    k: int = 8
    power: float = 2.0
    exact_eps: float = 1e-6

    def __post_init__(self) -> None:
        self.source_xy = np.asarray(self.source_xy, dtype=float)
        self.target_xy = np.asarray(self.target_xy, dtype=float)
        if self.source_xy.shape != self.target_xy.shape or self.source_xy.ndim != 2 or self.source_xy.shape[1] != 2:
            raise ValueError("source_xy and target_xy must both be shaped (N, 2).")
        if len(self.source_xy) < 1:
            raise ValueError("KnnLocalAffineWarp needs at least one landmark.")
        self.k = max(1, min(int(self.k), len(self.source_xy)))
        self.power = max(float(self.power), 0.0)
        self.tree = cKDTree(self.source_xy)
        self.global_params = _fit_weighted_affine_params(
            self.source_xy,
            self.target_xy,
            fallback=_translation_affine_params(self.source_xy, self.target_xy),
        )

    def predict(self, query_xy: np.ndarray, chunk_size: int = 50000) -> np.ndarray:
        query = np.asarray(query_xy, dtype=float)
        was_1d = query.ndim == 1
        if was_1d:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("query_xy must be shaped (N, 2).")

        out = np.empty_like(query, dtype=float)
        for start in range(0, len(query), int(chunk_size)):
            end = min(len(query), start + int(chunk_size))
            out[start:end] = self._predict_chunk(query[start:end])
        return out[0] if was_1d else out

    def _predict_chunk(self, query: np.ndarray) -> np.ndarray:
        dists, idxs = self.tree.query(query, k=self.k)
        dists = np.asarray(dists, dtype=float)
        idxs = np.asarray(idxs, dtype=int)
        if self.k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        out = np.empty_like(query, dtype=float)
        exact = dists[:, 0] <= float(self.exact_eps)
        if np.any(exact):
            out[exact] = self.target_xy[idxs[exact, 0]]

        for row in np.where(~exact)[0]:
            src = self.source_xy[idxs[row]]
            dst = self.target_xy[idxs[row]]
            if len(src) < 3:
                out[row] = _apply_affine_params(query[row:row + 1], self.global_params)[0]
                continue
            weights = 1.0 / np.power(np.maximum(dists[row], 1e-6), self.power)
            params = _fit_weighted_affine_params(src, dst, weights=weights, fallback=self.global_params)
            out[row] = _apply_affine_params(query[row:row + 1], params)[0]
        return out


def warp_array_with_knn_local_affine(
    arr: np.ndarray,
    inverse_warp: KnnLocalAffineWarp,
    output_shape: tuple[int, int],
    order: int,
    field_spacing: int = 128,
) -> np.ndarray:
    """Warp array by sampling an inverse KNN local-affine coordinate field."""
    from scipy.ndimage import map_coordinates

    h, w = int(output_shape[0]), int(output_shape[1])
    spacing = max(16, int(field_spacing))
    grid_y = np.unique(np.r_[np.arange(0, h, spacing, dtype=int), h - 1])
    grid_x = np.unique(np.r_[np.arange(0, w, spacing, dtype=int), w - 1])
    gy, gx = np.meshgrid(grid_y.astype(float), grid_x.astype(float), indexing="ij")
    fixed_xy = np.column_stack([gx.ravel(), gy.ravel()])
    source_xy = inverse_warp.predict(fixed_xy).reshape(len(grid_y), len(grid_x), 2)
    interp_x = RegularGridInterpolator((grid_y.astype(float), grid_x.astype(float)), source_xy[..., 0], bounds_error=False, fill_value=None)
    interp_y = RegularGridInterpolator((grid_y.astype(float), grid_x.astype(float)), source_xy[..., 1], bounds_error=False, fill_value=None)

    yy, xx = np.meshgrid(np.arange(h, dtype=float), np.arange(w, dtype=float), indexing="ij")
    points_yx = np.column_stack([yy.ravel(), xx.ravel()])
    src_x = interp_x(points_yx).reshape(h, w)
    src_y = interp_y(points_yx).reshape(h, w)

    def _warp_channel(channel: np.ndarray) -> np.ndarray:
        return map_coordinates(
            channel.astype(float),
            [src_y, src_x],
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=(order > 1),
        )

    arr = np.asarray(arr)
    if arr.ndim == 2:
        return _warp_channel(arr)
    if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        warped = np.zeros((arr.shape[0], h, w), dtype=float)
        for c in range(arr.shape[0]):
            warped[c] = _warp_channel(arr[c])
        return warped
    if arr.ndim == 3 and arr.shape[-1] <= 8:
        warped = np.zeros((h, w, arr.shape[-1]), dtype=float)
        for c in range(arr.shape[-1]):
            warped[..., c] = _warp_channel(arr[..., c])
        return warped
    raise ValueError(f"WSI KNN local affine warp supports 2D, CYX, or YXC arrays, got shape {arr.shape}.")


def warp_array_with_translation_grid(
    arr: np.ndarray,
    translation_grid: np.ndarray,
    output_shape: tuple[int, int],
    order: int,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    h, w = int(output_shape[0]), int(output_shape[1])
    grid_h, grid_w = translation_grid.shape[:2]
    y_centers = (np.arange(grid_h, dtype=float) + 0.5) * h / float(grid_h)
    x_centers = (np.arange(grid_w, dtype=float) + 0.5) * w / float(grid_w)
    interp_dx = RegularGridInterpolator((y_centers, x_centers), translation_grid[..., 0], bounds_error=False, fill_value=None)
    interp_dy = RegularGridInterpolator((y_centers, x_centers), translation_grid[..., 1], bounds_error=False, fill_value=None)

    yy, xx = np.meshgrid(np.arange(h, dtype=float), np.arange(w, dtype=float), indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])
    dx = interp_dx(points).reshape(h, w)
    dy = interp_dy(points).reshape(h, w)
    src_y = yy - dy
    src_x = xx - dx

    if arr.ndim == 2:
        return map_coordinates(arr.astype(float), [src_y, src_x], order=order, mode="constant", cval=0.0)
    if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        warped = np.zeros((arr.shape[0], h, w), dtype=float)
        for c in range(arr.shape[0]):
            warped[c] = map_coordinates(arr[c].astype(float), [src_y, src_x], order=order, mode="constant", cval=0.0)
        return warped
    if arr.ndim == 3 and arr.shape[-1] <= 8:
        warped = np.zeros((h, w, arr.shape[-1]), dtype=float)
        for c in range(arr.shape[-1]):
            warped[..., c] = map_coordinates(arr[..., c].astype(float), [src_y, src_x], order=order, mode="constant", cval=0.0)
        return warped
    raise ValueError(f"WSI local translation warp supports 2D, CYX, or YXC arrays, got shape {arr.shape}.")


def select_landmarks_with_grid_coverage(
    candidates: pd.DataFrame,
    feats1: pd.DataFrame,
    top_k: int,
    grid: int = 4,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    grid = max(1, int(grid))
    top_k = max(1, int(top_k))
    if grid < 2:
        return candidates.sort_values(["displacement_residual_px", "distance"], ascending=True).head(top_k).reset_index(drop=True)

    idx1 = candidates["idx1"].to_numpy(dtype=int)
    xs = feats1.iloc[idx1]["centroid_x"].to_numpy(dtype=float)
    ys = feats1.iloc[idx1]["centroid_y"].to_numpy(dtype=float)
    max_x = max(float(feats1["centroid_x"].max()), 1.0)
    max_y = max(float(feats1["centroid_y"].max()), 1.0)
    gx = np.clip(np.floor(xs * grid / max_x).astype(int), 0, grid - 1)
    gy = np.clip(np.floor(ys * grid / max_y).astype(int), 0, grid - 1)
    patch_ids = gy * grid + gx

    ranked = candidates.copy()
    ranked["_patch_id"] = patch_ids
    ranked = ranked.sort_values(["displacement_residual_px", "distance"], ascending=True)

    per_patch_quota = max(1, int(np.ceil(top_k / float(grid * grid))))
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    for patch_id in range(grid * grid):
        patch_rows = ranked[ranked["_patch_id"] == patch_id].head(per_patch_quota)
        for idx in patch_rows.index.to_list():
            selected_indices.append(int(idx))
            selected_set.add(int(idx))

    if len(selected_indices) < top_k:
        for idx in ranked.index.to_list():
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            selected_indices.append(idx_int)
            selected_set.add(idx_int)
            if len(selected_indices) >= top_k:
                break

    out = candidates.loc[selected_indices].copy().reset_index(drop=True)
    idx1_out = out["idx1"].to_numpy(dtype=int)
    xs_out = feats1.iloc[idx1_out]["centroid_x"].to_numpy(dtype=float)
    ys_out = feats1.iloc[idx1_out]["centroid_y"].to_numpy(dtype=float)
    out["wsi_grid_x"] = np.clip(np.floor(xs_out * grid / max_x).astype(int), 0, grid - 1)
    out["wsi_grid_y"] = np.clip(np.floor(ys_out * grid / max_y).astype(int), 0, grid - 1)
    return out


def add_neighbor_profile_scores(
    candidates: pd.DataFrame,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    neighbor_k: int = 8,
) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        out["neighbor_profile_diff"] = []
        return out

    coords1 = feats1[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    coords2 = feats2[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    profiles1 = _neighbor_distance_profiles(coords1, neighbor_k)
    profiles2 = _neighbor_distance_profiles(coords2, neighbor_k)
    vectors1 = _neighbor_vector_profiles(coords1, neighbor_k)
    vectors2 = _neighbor_vector_profiles(coords2, neighbor_k)
    idx1 = out["idx1"].to_numpy(dtype=int)
    idx2 = out["idx2"].to_numpy(dtype=int)
    p1 = profiles1[idx1]
    p2 = profiles2[idx2]
    denom = np.maximum(np.maximum(p1, p2), 1.0)
    distance_diffs = np.median(np.abs(p1 - p2) / denom, axis=1)

    v1 = _normalize_neighbor_vectors(vectors1[idx1])
    v2 = _normalize_neighbor_vectors(vectors2[idx2])
    pair_dists = np.linalg.norm(v1[:, :, None, :] - v2[:, None, :, :], axis=3)
    vector_diffs = 0.5 * (
        np.median(np.min(pair_dists, axis=2), axis=1)
        + np.median(np.min(pair_dists, axis=1), axis=1)
    )
    out["neighbor_distance_diff"] = np.nan_to_num(distance_diffs, nan=np.inf, posinf=np.inf, neginf=np.inf)
    out["neighbor_vector_diff"] = np.nan_to_num(vector_diffs, nan=np.inf, posinf=np.inf, neginf=np.inf)
    out["neighbor_profile_diff"] = np.maximum(out["neighbor_distance_diff"], out["neighbor_vector_diff"])
    return out


def add_cell_orientation_scores(
    candidates: pd.DataFrame,
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.copy()
    required = {"orientation", "eccentricity"}
    if not required.issubset(feats1.columns) or not required.issubset(feats2.columns):
        return out
    idx1 = out["idx1"].to_numpy(dtype=int)
    idx2 = out["idx2"].to_numpy(dtype=int)
    orient1 = feats1.iloc[idx1]["orientation"].to_numpy(dtype=float)
    orient2 = feats2.iloc[idx2]["orientation"].to_numpy(dtype=float)
    out["orientation_diff_deg"] = _orientation_difference_deg(orient1, orient2)
    out["eccentricity_1"] = feats1.iloc[idx1]["eccentricity"].to_numpy(dtype=float)
    out["eccentricity_2"] = feats2.iloc[idx2]["eccentricity"].to_numpy(dtype=float)
    return out


def _orientation_difference_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    diff = np.abs(angle1 - angle2)
    diff = np.mod(diff, np.pi)
    diff = np.minimum(diff, np.pi - diff)
    return np.degrees(diff)


def _neighbor_distance_profiles(coords: np.ndarray, neighbor_k: int) -> np.ndarray:
    n = len(coords)
    k = max(1, int(neighbor_k))
    if n <= 1:
        return np.zeros((n, k), dtype=float)
    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_eff + 1)
    profiles = np.asarray(dists[:, 1:], dtype=float)
    if k_eff < k:
        pad = np.repeat(profiles[:, -1:], k - k_eff, axis=1)
        profiles = np.hstack([profiles, pad])
    return profiles


def _neighbor_vector_profiles(coords: np.ndarray, neighbor_k: int) -> np.ndarray:
    n = len(coords)
    k = max(1, int(neighbor_k))
    if n <= 1:
        return np.zeros((n, k, 2), dtype=float)
    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k_eff + 1)
    neighbor_idxs = np.asarray(idxs[:, 1:], dtype=int)
    vectors = coords[neighbor_idxs] - coords[:, None, :]
    if k_eff < k:
        pad = np.repeat(vectors[:, -1:, :], k - k_eff, axis=1)
        vectors = np.concatenate([vectors, pad], axis=1)
    return vectors.astype(float, copy=False)


def _normalize_neighbor_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=2)
    scale = np.maximum(np.median(lengths, axis=1), 1.0)
    return vectors / scale[:, None, None]


def filter_long_match_lines(displacements: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    lengths = np.linalg.norm(displacements, axis=1)
    if len(lengths) == 0:
        return np.zeros(0, dtype=bool), {"mean": 0.0, "median": 0.0, "threshold": 0.0}
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    mad_len = float(np.median(np.abs(lengths - median_len)))
    robust_sigma = max(1.4826 * mad_len, 1.0)
    threshold = max(mean_len, median_len + 3.0 * robust_sigma)
    return lengths <= threshold, {
        "mean": mean_len,
        "median": median_len,
        "threshold": float(threshold),
    }


def filter_parallel_displacements(
    displacements: np.ndarray,
    reference_disp: np.ndarray,
    max_angle_deg: float,
) -> np.ndarray:
    norms = np.linalg.norm(displacements, axis=1)
    ref_norm = float(np.linalg.norm(reference_disp))
    if ref_norm < 1.0:
        return np.zeros(len(displacements), dtype=bool)
    valid = norms >= 1.0
    cos_values = np.full(len(displacements), -1.0, dtype=float)
    cos_values[valid] = (
        displacements[valid] @ reference_disp
    ) / (norms[valid] * ref_norm + 1e-8)
    cos_values = np.clip(cos_values, -1.0, 1.0)
    angle_diff = np.degrees(np.arccos(cos_values))
    return valid & (angle_diff <= float(max_angle_deg))


def shift_array_xy(arr: np.ndarray, translation_xy: np.ndarray, order: int) -> np.ndarray:
    tx, ty = float(translation_xy[0]), float(translation_xy[1])
    if arr.ndim == 2:
        shift_values = (ty, tx)
    elif arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        shift_values = (0.0, ty, tx)
    elif arr.ndim == 3 and arr.shape[-1] <= 8:
        shift_values = (ty, tx, 0.0)
    else:
        raise ValueError(f"WSI translation warp supports 2D, CYX, or YXC arrays, got shape {arr.shape}.")
    return shift(arr, shift=shift_values, order=order, mode="constant", cval=0.0, prefilter=(order > 1))


def cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="WSI registration from real mask cell landmarks.")
    parser.add_argument("--fixed-mask", type=Path)
    parser.add_argument("--moving-mask", type=Path)
    parser.add_argument("--fixed-centroids", type=Path, help="HE centroid CSV; requires paired metadata JSON.")
    parser.add_argument("--moving-centroids", type=Path, help="mIF-DAPI centroid CSV; requires paired metadata JSON.")
    parser.add_argument("--fixed-centroids-metadata", type=Path)
    parser.add_argument("--moving-centroids-metadata", type=Path)
    parser.add_argument("--fixed-tissue-mask", type=Path)
    parser.add_argument("--moving-tissue-mask", type=Path)
    parser.add_argument("--fixed-tissue-downsample", type=float)
    parser.add_argument("--moving-tissue-downsample", type=float)
    parser.add_argument("--refine-global-affine-with-centroids", action="store_true")
    parser.add_argument("--fixed-features", type=Path)
    parser.add_argument("--moving-features", type=Path)
    parser.add_argument("--fixed-cellvit-json", type=Path)
    parser.add_argument("--moving-cellvit-json", type=Path)
    parser.add_argument("--fixed-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--fixed-image", type=Path, default=None)
    parser.add_argument("--moving-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("wsi_registration_output"))
    parser.add_argument("--top-k", type=int, default=320)
    parser.add_argument("--max-angle-deg", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--max-area", type=int, default=0)
    parser.add_argument("--legacy-grid", action="store_true", help="Use the old 4x4 local translation grid instead of KNN local affine.")
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--knn-power", type=float, default=2.0)
    parser.add_argument("--no-residual-filter", action="store_true")
    parser.add_argument("--estimate-only", action="store_true", help="Only write landmarks, transform, and registered centroids; skip full-resolution mask/image warp.")
    parser.add_argument("--write-registered-wsi", action="store_true", help="After transform estimation, tile-wise warp moving WSI channels when a channel-region reader is available.")
    args = parser.parse_args()

    if args.fixed_centroids is not None or args.moving_centroids is not None:
        if args.fixed_centroids is None or args.moving_centroids is None:
            raise ValueError("Provide both --fixed-centroids and --moving-centroids.")
        result = run_wsi_centroid_registration(
            fixed_centroids=args.fixed_centroids,
            fixed_metadata=args.fixed_centroids_metadata,
            moving_centroids=args.moving_centroids,
            moving_metadata=args.moving_centroids_metadata,
            output_dir=args.output_dir,
            top_k=args.top_k,
            use_knn_local_affine=not args.legacy_grid,
            knn_k=args.knn_k,
            knn_power=args.knn_power,
            residual_filter=not args.no_residual_filter,
            fixed_tissue_mask=args.fixed_tissue_mask,
            moving_tissue_mask=args.moving_tissue_mask,
            fixed_tissue_downsample=args.fixed_tissue_downsample,
            moving_tissue_downsample=args.moving_tissue_downsample,
            refine_global_affine_with_centroids=args.refine_global_affine_with_centroids if (args.fixed_tissue_mask is not None and args.moving_tissue_mask is not None) else None,
            write_registered_wsi=args.write_registered_wsi,
            moving_wsi_path=args.moving_image,
        )
        print(f"WSI centroid matches: {len(result.matches)}")
        print("Global affine mIF->HE:")
        print(np.array2string(result.global_affine_mif_to_he, precision=6, suppress_small=True))
        if result.residuals.size:
            print(f"Residual mean/max: {float(result.residuals.mean()):.3f}/{float(result.residuals.max()):.3f}")
        print(f"Refinement: {result.diagnostics.get('local_refinement') if result.diagnostics else 'unknown'}")
        return

    fixed_shape = tuple(args.fixed_shape) if args.fixed_shape is not None else None
    fixed_features = _load_cli_features(args.fixed_features, args.fixed_cellvit_json)
    moving_features = _load_cli_features(args.moving_features, args.moving_cellvit_json)

    normalized_moving_features = None
    if fixed_features is not None or moving_features is not None:
        if fixed_features is None or moving_features is None:
            raise ValueError("Provide both fixed and moving feature sources.")
        if fixed_shape is None:
            if args.fixed_mask is None:
                raise ValueError("Feature-based registration requires --fixed-shape HEIGHT WIDTH or --fixed-mask for shape metadata.")
            fixed_shape = _tiff_shape(args.fixed_mask)
        result = run_wsi_feature_registration(
            fixed_features=fixed_features,
            moving_features=moving_features,
            fixed_shape=fixed_shape,
            top_k=args.top_k,
            max_angle_deg=args.max_angle_deg,
            min_area=args.min_area,
            max_area=args.max_area,
            use_knn_local_affine=not args.legacy_grid,
            knn_k=args.knn_k,
            knn_power=args.knn_power,
            residual_filter=not args.no_residual_filter,
        )
        normalized_moving_features = normalize_wsi_feature_table(
            moving_features,
            fixed_shape,
            min_area=args.min_area,
            max_area=args.max_area,
        )
    else:
        if args.fixed_mask is None or args.moving_mask is None:
            raise ValueError("Provide either fixed/moving masks or fixed/moving feature sources.")
        fixed_mask = imread(str(args.fixed_mask)).astype(np.int32)
        moving_mask = imread(str(args.moving_mask)).astype(np.int32)
        fixed_image = imread(str(args.fixed_image)) if args.fixed_image is not None else None
        moving_image = imread(str(args.moving_image)) if args.moving_image is not None else None

        result = run_wsi_mask_registration(
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            fixed_image=fixed_image,
            moving_image=moving_image,
            top_k=args.top_k,
            max_angle_deg=args.max_angle_deg,
            min_area=args.min_area,
            max_area=args.max_area,
            use_knn_local_affine=not args.legacy_grid,
            knn_k=args.knn_k,
            knn_power=args.knn_power,
            residual_filter=not args.no_residual_filter,
            warp_registered_mask=not args.estimate_only,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if result.registered_mask is not None:
        imwrite(str(args.output_dir / "registered_mask.tif"), result.registered_mask)
    if result.registered_image is not None:
        imwrite(str(args.output_dir / "registered_image.tif"), result.registered_image)
    result.matches.to_csv(args.output_dir / "wsi_landmarks.csv", index=False)
    if normalized_moving_features is not None:
        registered_features = normalized_moving_features.copy()
        moving_xy = registered_features[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        if result.local_affine_warp is not None:
            registered_xy = result.local_affine_warp.predict(moving_xy)
        else:
            registered_xy = moving_xy + result.transform.translation[None, :]
        registered_features["registered_centroid_x"] = registered_xy[:, 0]
        registered_features["registered_centroid_y"] = registered_xy[:, 1]
        registered_features.to_csv(args.output_dir / "registered_moving_features.csv", index=False)
    np.savetxt(args.output_dir / "translation_xy.txt", result.transform.translation[None, :], fmt="%.6f")
    print(f"WSI landmarks: {len(result.matches)}")
    print(f"Translation xy: {result.transform.translation.tolist()}")
    print(f"Residual mean/max: {float(result.residuals.mean()):.3f}/{float(result.residuals.max()):.3f}")
    print(f"Refinement: {'KNN local affine' if result.local_affine_warp is not None else '4x4 local translation grid'}")


def _load_cli_features(csv_path: Path | None, cellvit_json_path: Path | None) -> pd.DataFrame | None:
    if csv_path is not None and cellvit_json_path is not None:
        raise ValueError("Use only one of --features or --cellvit-json per side.")
    if csv_path is not None:
        return pd.read_csv(csv_path)
    if cellvit_json_path is not None:
        return load_cellvit_json_features(cellvit_json_path)
    return None


def _tiff_shape(path: Path) -> tuple[int, int]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
    if len(shape) < 2:
        raise ValueError(f"Expected a 2D TIFF shape, got {shape}.")
    return int(shape[-2]), int(shape[-1])


def _empty_wsi_match_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx1",
            "idx2",
            "distance",
            "dx",
            "dy",
            "match_line_length_px",
            "displacement_residual_px",
            "orientation_diff_deg",
            "eccentricity_1",
            "eccentricity_2",
            "neighbor_distance_diff",
            "neighbor_vector_diff",
            "neighbor_profile_diff",
            "cell_id_1",
            "cell_id_2",
        ]
    )


def _robust_standardize_feature_tables(
    feats1: pd.DataFrame,
    feats2: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    vals1 = feats1.loc[:, feature_columns].to_numpy(dtype=float)
    vals2 = feats2.loc[:, feature_columns].to_numpy(dtype=float)
    combined = np.vstack([vals1, vals2])
    med = np.nanmedian(combined, axis=0)
    mad = np.nanmedian(np.abs(combined - med[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    vals1 = np.nan_to_num((vals1 - med[None, :]) / scale[None, :])
    vals2 = np.nan_to_num((vals2 - med[None, :]) / scale[None, :])
    return vals1, vals2


if __name__ == "__main__":
    cli_main()

"""Image I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import imageio.v3 as iio
import numpy as np

from .config import SUPPORTED_EXTENSIONS


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array as float32. Channels are preserved if present.
    """
    img = iio.imread(path)
    return np.asarray(img, dtype=np.float32)


def find_images(directory: str | Path) -> List[Path]:
    """
    Recursively find supported image files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search.

    Returns
    -------
    List[Path]
        List of file paths.
    """
    root = Path(directory)
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    """
    Save a label mask to disk.

    Parameters
    ----------
    path : str or Path
        Destination path.
    mask : np.ndarray
        Label mask to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, mask.astype(np.int32))


def infer_image_mode(img: np.ndarray) -> Literal["2d", "3d_zstack"]:
    """
    Infer whether an image should be treated as 2D or 3D Z-stack based on shape.

    Rules
    -----
    - ndim == 2: 2D
    - ndim == 3: 2D if last dim looks like channels (<= 4), otherwise 3D Z-stack (Z, Y, X)
    - ndim == 4: 3D Z-stack with channels last (Z, Y, X, C)
    """
    if img.ndim == 2:
        return "2d"
    if img.ndim == 3:
        return "2d" if img.shape[-1] <= 4 else "3d_zstack"
    if img.ndim == 4:
        return "3d_zstack"
    raise ValueError(f"Unsupported image ndim {img.ndim} for mode inference.")


def project_intensity_max(img: np.ndarray) -> np.ndarray:
    """
    Create a 2D intensity projection for visualization/overlays.

    For 3D Z-stacks, takes a max projection over Z after selecting the last channel when present.
    For 2D inputs, returns the input or the last channel for HxWxC arrays.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[-1] <= 4:
            return img[..., -1]
        return np.max(img, axis=0)
    if img.ndim == 4:
        return np.max(img[..., -1], axis=0)
    raise ValueError(f"Unsupported image shape {img.shape} for intensity projection.")

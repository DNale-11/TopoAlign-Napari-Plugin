"""GPU-accelerated operations for cell registration (PyTorch CUDA).

Every public function returns ``None`` when CUDA is unavailable or an error
occurs, so callers can fall back to the existing NumPy/SciPy path with zero
risk.

TPS kernel uses float64 for accuracy (r²·log(r) accumulates over ~200 terms).
Pairwise cdist uses float32 (standardized features, small values, no accumulation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _get_torch_cuda():
    """Return the torch module if CUDA is ready, else None."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# TPS predict – GPU (float64 for accuracy)
# ---------------------------------------------------------------------------

def tps_predict_gpu(
    ctrl_pts: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    query_pts: np.ndarray,
    chunk_size: int = 500_000,
) -> Optional[np.ndarray]:
    """Evaluate a fitted TPS at *query_pts* on GPU.

    Parameters
    ----------
    ctrl_pts : (N, 2) control points used during fit.
    wx, wy   : (N+3,) weight vectors from TPS solve.
    query_pts: (M, 2) points to evaluate.
    chunk_size : max query points per GPU batch (memory control).

    Returns
    -------
    (M, 2) predicted coordinates, or ``None`` on failure / no CUDA.
    """
    torch = _get_torch_cuda()
    if torch is None:
        return None

    try:
        device = torch.device("cuda")
        dtype = torch.float64

        ctrl = torch.as_tensor(np.asarray(ctrl_pts, dtype=np.float64), device=device)
        w_x = torch.as_tensor(np.asarray(wx, dtype=np.float64), device=device)
        w_y = torch.as_tensor(np.asarray(wy, dtype=np.float64), device=device)

        query = np.asarray(query_pts, dtype=np.float64)
        if query.ndim == 1:
            query = query[None, :]
        M = query.shape[0]
        out = np.empty((M, 2), dtype=np.float64)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            q = torch.as_tensor(query[start:end], device=device)  # (B, 2)
            B = end - start

            # Pairwise distances: (B, N)
            diff = q[:, None, :] - ctrl[None, :, :]  # (B, N, 2)
            r = torch.linalg.norm(diff, dim=2)  # (B, N)

            # TPS kernel: r^2 * log(r), with 0 where r ~ 0
            r_sq = r * r
            log_r = torch.log(torch.clamp(r, min=1e-10))
            K = torch.where(r < 1e-10, torch.zeros_like(r_sq), r_sq * log_r)

            # Polynomial part: [1, x, y]
            ones = torch.ones(B, 1, dtype=dtype, device=device)
            P = torch.cat([ones, q], dim=1)  # (B, 3)

            # Full basis: [K | P]  @ weights
            basis = torch.cat([K, P], dim=1)  # (B, N+3)
            out_x = basis @ w_x
            out_y = basis @ w_y

            out[start:end, 0] = out_x.cpu().numpy()
            out[start:end, 1] = out_y.cpu().numpy()

        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pairwise feature distance – GPU (float32 sufficient)
# ---------------------------------------------------------------------------

def pairwise_cdist_gpu(
    f1: np.ndarray,
    f2: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute pairwise Euclidean distance matrix on GPU.

    Parameters
    ----------
    f1 : (N, D) standardized feature array.
    f2 : (M, D) standardized feature array.

    Returns
    -------
    (N, M) distance matrix, or ``None`` on failure / no CUDA.
    """
    torch = _get_torch_cuda()
    if torch is None:
        return None

    try:
        device = torch.device("cuda")
        t1 = torch.as_tensor(np.asarray(f1, dtype=np.float32), device=device)
        t2 = torch.as_tensor(np.asarray(f2, dtype=np.float32), device=device)
        dist = torch.cdist(t1, t2, p=2.0)
        return dist.cpu().numpy().astype(np.float64)
    except Exception:
        return None

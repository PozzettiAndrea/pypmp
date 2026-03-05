"""
High-level Python wrapper for pmp-library remeshing.
"""

import numpy as np
from numpy.typing import NDArray

from pypmp._pypmp import remesh_uniform as _remesh_uniform
from pypmp._pypmp import remesh_adaptive as _remesh_adaptive


def remesh_uniform(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    edge_length: float,
    iterations: int = 10,
    use_projection: bool = True,
    verbose: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Uniform isotropic remeshing via edge split/collapse/flip + tangential relaxation.

    Remeshes the input triangle mesh so that all edges are close to the
    target edge length. Uses incremental remeshing with back-projection
    to the input surface.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    edge_length : float
        Target edge length for output triangles.
    iterations : int, default 10
        Number of remeshing iterations. More iterations produce better quality.
    use_projection : bool, default True
        Project vertices back onto the input surface after each iteration.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3), float64
        Output vertex positions.
    faces_out : ndarray, shape (Q, 3), int32
        Output triangle face indices.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    return _remesh_uniform(v, f, edge_length, iterations, use_projection, verbose)


def remesh_adaptive(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    min_edge_length: float,
    max_edge_length: float,
    approx_error: float,
    iterations: int = 10,
    use_projection: bool = True,
    verbose: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Adaptive isotropic remeshing driven by local curvature.

    Uses smaller triangles in high-curvature regions and larger triangles
    in flat regions, within the specified edge length bounds.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    min_edge_length : float
        Minimum edge length (used in high-curvature areas).
    max_edge_length : float
        Maximum edge length (used in flat areas).
    approx_error : float
        Maximum geometric approximation error.
    iterations : int, default 10
        Number of remeshing iterations.
    use_projection : bool, default True
        Project vertices back onto the input surface after each iteration.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3), float64
        Output vertex positions.
    faces_out : ndarray, shape (Q, 3), int32
        Output triangle face indices.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    return _remesh_adaptive(
        v, f, min_edge_length, max_edge_length, approx_error,
        iterations, use_projection, verbose,
    )

"""Tests for pypmp: uniform and adaptive remeshing."""

import numpy as np
import pytest


def make_icosphere():
    """Create a simple icosphere for testing."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio

    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    return vertices, faces


def subdivide_icosphere(subdivisions=2):
    """Create a subdivided icosphere with more triangles."""
    vertices, faces = make_icosphere()

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []

        verts_list = vertices.tolist()

        def get_midpoint(i0, i1):
            key = (min(i0, i1), max(i0, i1))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = [
                (verts_list[i0][0] + verts_list[i1][0]) / 2,
                (verts_list[i0][1] + verts_list[i1][1]) / 2,
                (verts_list[i0][2] + verts_list[i1][2]) / 2,
            ]
            # Project onto unit sphere
            norm = (mid[0]**2 + mid[1]**2 + mid[2]**2) ** 0.5
            mid = [mid[0] / norm, mid[1] / norm, mid[2] / norm]
            idx = len(verts_list)
            verts_list.append(mid)
            edge_midpoints[key] = idx
            return idx

        for f in faces:
            v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            new_faces.extend([
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c],
            ])

        vertices = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int32)

    return vertices, faces


# ── Uniform Remeshing ─────────────────────────────────────────────


def test_remesh_uniform_basic():
    """Uniform remeshing should produce a valid triangle mesh."""
    import pypmp

    verts, faces = subdivide_icosphere(2)
    v_out, f_out = pypmp.remesh_uniform(verts, faces, edge_length=0.3)

    assert isinstance(v_out, np.ndarray)
    assert isinstance(f_out, np.ndarray)
    assert v_out.ndim == 2
    assert f_out.ndim == 2
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_remesh_uniform_preserves_scale():
    """Remeshing should roughly preserve the bounding box."""
    import pypmp

    verts, faces = subdivide_icosphere(2)
    v_out, _ = pypmp.remesh_uniform(verts, faces, edge_length=0.3)

    original_extent = np.max(np.abs(verts))
    remeshed_extent = np.max(np.abs(v_out))
    assert abs(remeshed_extent - original_extent) < 0.2


def test_remesh_uniform_edge_length_affects_density():
    """Smaller edge length should produce more vertices."""
    import pypmp

    verts, faces = subdivide_icosphere(2)

    _, f_small = pypmp.remesh_uniform(verts, faces, edge_length=0.2)
    _, f_large = pypmp.remesh_uniform(verts, faces, edge_length=0.5)

    assert len(f_small) > len(f_large)


def test_remesh_uniform_iterations():
    """Custom iteration count should work."""
    import pypmp

    verts, faces = subdivide_icosphere(2)
    v_out, f_out = pypmp.remesh_uniform(
        verts, faces, edge_length=0.3, iterations=3
    )

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0


def test_remesh_uniform_no_projection():
    """Remeshing without projection should work."""
    import pypmp

    verts, faces = subdivide_icosphere(2)
    v_out, f_out = pypmp.remesh_uniform(
        verts, faces, edge_length=0.3, use_projection=False
    )

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0


# ── Adaptive Remeshing ────────────────────────────────────────────


def test_remesh_adaptive_basic():
    """Adaptive remeshing should produce a valid triangle mesh."""
    import pypmp

    verts, faces = subdivide_icosphere(2)
    v_out, f_out = pypmp.remesh_adaptive(
        verts, faces,
        min_edge_length=0.1,
        max_edge_length=0.5,
        approx_error=0.01,
    )

    assert isinstance(v_out, np.ndarray)
    assert isinstance(f_out, np.ndarray)
    assert v_out.ndim == 2
    assert f_out.ndim == 2
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_remesh_adaptive_preserves_scale():
    """Adaptive remeshing should roughly preserve the bounding box."""
    import pypmp

    verts, faces = subdivide_icosphere(2)
    v_out, _ = pypmp.remesh_adaptive(
        verts, faces,
        min_edge_length=0.1,
        max_edge_length=0.5,
        approx_error=0.01,
    )

    original_extent = np.max(np.abs(verts))
    remeshed_extent = np.max(np.abs(v_out))
    assert abs(remeshed_extent - original_extent) < 0.2


# ── Input Validation ──────────────────────────────────────────────


def test_remesh_uniform_input_validation():
    """Invalid inputs should raise errors."""
    import pypmp

    # Wrong vertex shape
    bad_verts = np.zeros((10, 2), dtype=np.float64)
    good_faces = np.zeros((5, 3), dtype=np.int32)

    with pytest.raises((ValueError, RuntimeError)):
        pypmp.remesh_uniform(bad_verts, good_faces, edge_length=0.3)


def test_remesh_uniform_negative_edge_length():
    """Negative edge length should raise an error."""
    import pypmp

    verts, faces = make_icosphere()

    with pytest.raises(RuntimeError):
        pypmp.remesh_uniform(verts, faces, edge_length=-1.0)


def test_remesh_adaptive_invalid_range():
    """min_edge > max_edge should raise an error."""
    import pypmp

    verts, faces = subdivide_icosphere(2)

    with pytest.raises(RuntimeError):
        pypmp.remesh_adaptive(
            verts, faces,
            min_edge_length=1.0,
            max_edge_length=0.1,
            approx_error=0.01,
        )

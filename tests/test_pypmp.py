"""Tests for pypmp: uniform and adaptive remeshing."""

import numpy as np
import pytest


# ── Uniform Remeshing ─────────────────────────────────────────────


def test_remesh_uniform_basic(icosphere):
    """Uniform remeshing should produce a valid triangle mesh."""
    import pypmp

    verts, faces = icosphere
    v_out, f_out = pypmp.remesh_uniform(verts, faces, edge_length=0.3)

    assert isinstance(v_out, np.ndarray)
    assert isinstance(f_out, np.ndarray)
    assert v_out.ndim == 2
    assert f_out.ndim == 2
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_remesh_uniform_preserves_scale(icosphere):
    """Remeshing should roughly preserve the bounding box."""
    import pypmp

    verts, faces = icosphere
    v_out, _ = pypmp.remesh_uniform(verts, faces, edge_length=0.3)

    original_extent = np.max(np.abs(verts))
    remeshed_extent = np.max(np.abs(v_out))
    assert abs(remeshed_extent - original_extent) < 0.2


def test_remesh_uniform_edge_length_affects_density(icosphere):
    """Smaller edge length should produce more vertices."""
    import pypmp

    verts, faces = icosphere

    _, f_small = pypmp.remesh_uniform(verts, faces, edge_length=0.2)
    _, f_large = pypmp.remesh_uniform(verts, faces, edge_length=0.5)

    assert len(f_small) > len(f_large)


def test_remesh_uniform_iterations(icosphere):
    """Custom iteration count should work."""
    import pypmp

    verts, faces = icosphere
    v_out, f_out = pypmp.remesh_uniform(
        verts, faces, edge_length=0.3, iterations=3
    )

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0


def test_remesh_uniform_no_projection(icosphere):
    """Remeshing without projection should work."""
    import pypmp

    verts, faces = icosphere
    v_out, f_out = pypmp.remesh_uniform(
        verts, faces, edge_length=0.3, use_projection=False
    )

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0


# ── Adaptive Remeshing ────────────────────────────────────────────


def test_remesh_adaptive_basic(icosphere):
    """Adaptive remeshing should produce a valid triangle mesh."""
    import pypmp

    verts, faces = icosphere
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


def test_remesh_adaptive_preserves_scale(icosphere):
    """Adaptive remeshing should roughly preserve the bounding box."""
    import pypmp

    verts, faces = icosphere
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


def test_remesh_uniform_negative_edge_length(icosphere):
    """Negative edge length should raise an error."""
    import pypmp

    verts, faces = icosphere

    with pytest.raises(RuntimeError):
        pypmp.remesh_uniform(verts, faces, edge_length=-1.0)


def test_remesh_adaptive_invalid_range(icosphere):
    """min_edge > max_edge should raise an error."""
    import pypmp

    verts, faces = icosphere

    with pytest.raises(RuntimeError):
        pypmp.remesh_adaptive(
            verts, faces,
            min_edge_length=1.0,
            max_edge_length=0.1,
            approx_error=0.01,
        )

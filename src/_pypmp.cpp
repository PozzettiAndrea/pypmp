// Python bindings for pmp-library remeshing via nanobind.
//
// Exposes: remesh_uniform, remesh_adaptive.

#include <cstring>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "array_support.h"

#include <pmp/surface_mesh.h>
#include <pmp/algorithms/remeshing.h>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Mesh conversion helpers
// ---------------------------------------------------------------------------

// Convert numpy arrays (N,3) vertices + (M,3) faces into a pmp::SurfaceMesh.
static void numpy_to_pmp(
    const NDArray<const double, 2> verts,
    const NDArray<const int, 2> faces,
    pmp::SurfaceMesh &mesh
) {
    const size_t nv = verts.shape(0);
    const size_t nf = faces.shape(0);

    if (verts.shape(1) != 3) {
        throw std::runtime_error("Vertex array must have shape (N, 3)");
    }
    if (faces.shape(1) != 3) {
        throw std::runtime_error("Face array must have shape (M, 3) — input must be a triangle mesh");
    }

    mesh.clear();

    // Add vertices
    std::vector<pmp::Vertex> vertex_handles(nv);
    for (size_t i = 0; i < nv; ++i) {
        vertex_handles[i] = mesh.add_vertex(
            pmp::Point(verts(i, 0), verts(i, 1), verts(i, 2))
        );
    }

    // Add triangular faces
    for (size_t i = 0; i < nf; ++i) {
        int i0 = faces(i, 0);
        int i1 = faces(i, 1);
        int i2 = faces(i, 2);

        if (i0 < 0 || i0 >= (int)nv ||
            i1 < 0 || i1 >= (int)nv ||
            i2 < 0 || i2 >= (int)nv) {
            throw std::runtime_error(
                "Face index out of bounds at face " + std::to_string(i)
            );
        }

        mesh.add_triangle(vertex_handles[i0],
                           vertex_handles[i1],
                           vertex_handles[i2]);
    }
}

// Convert a pmp::SurfaceMesh back to numpy arrays: (vertices, faces).
static nb::tuple pmp_to_numpy(const pmp::SurfaceMesh &mesh) {
    const size_t nv = mesh.n_vertices();
    const size_t nf = mesh.n_faces();

    NDArray<double, 2> verts_arr = MakeNDArray<double, 2>(
        {static_cast<int>(nv), 3});
    double *verts = verts_arr.data();

    // Build a mapping from vertex handle idx to contiguous index,
    // in case there are deleted vertices (gaps in idx sequence).
    std::vector<int> idx_map(mesh.vertices_size(), -1);
    int vi = 0;
    for (auto v : mesh.vertices()) {
        const pmp::Point &p = mesh.position(v);
        verts[vi * 3 + 0] = p[0];
        verts[vi * 3 + 1] = p[1];
        verts[vi * 3 + 2] = p[2];
        idx_map[v.idx()] = vi;
        vi++;
    }

    NDArray<int, 2> faces_arr = MakeNDArray<int, 2>(
        {static_cast<int>(nf), 3});
    int *faces = faces_arr.data();

    int fi = 0;
    for (auto f : mesh.faces()) {
        int corner = 0;
        for (auto v : mesh.vertices(f)) {
            if (corner < 3) {
                faces[fi * 3 + corner] = idx_map[v.idx()];
            }
            corner++;
        }
        fi++;
    }

    return nb::make_tuple(verts_arr, faces_arr);
}

// ---------------------------------------------------------------------------
// Wrapped functions
// ---------------------------------------------------------------------------

static nb::tuple py_remesh_uniform(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double edge_length,
    unsigned int iterations,
    bool use_projection,
    int verbose
) {
    if (edge_length <= 0.0) {
        throw std::runtime_error("edge_length must be positive");
    }

    pmp::SurfaceMesh mesh;
    numpy_to_pmp(vertices, faces, mesh);

    if (!mesh.is_triangle_mesh()) {
        throw std::runtime_error("Input must be a triangle mesh");
    }

    pmp::uniform_remeshing(mesh, edge_length, iterations, use_projection,
                           verbose);
    mesh.garbage_collection();

    return pmp_to_numpy(mesh);
}

static nb::tuple py_remesh_adaptive(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double min_edge_length,
    double max_edge_length,
    double approx_error,
    unsigned int iterations,
    bool use_projection,
    int verbose
) {
    if (min_edge_length <= 0.0) {
        throw std::runtime_error("min_edge_length must be positive");
    }
    if (max_edge_length <= min_edge_length) {
        throw std::runtime_error(
            "max_edge_length must be greater than min_edge_length");
    }
    if (approx_error <= 0.0) {
        throw std::runtime_error("approx_error must be positive");
    }

    pmp::SurfaceMesh mesh;
    numpy_to_pmp(vertices, faces, mesh);

    if (!mesh.is_triangle_mesh()) {
        throw std::runtime_error("Input must be a triangle mesh");
    }

    pmp::adaptive_remeshing(mesh, min_edge_length, max_edge_length,
                            approx_error, iterations, use_projection,
                            verbose);
    mesh.garbage_collection();

    return pmp_to_numpy(mesh);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

NB_MODULE(_pypmp, m) {
    m.doc() = "Python bindings for pmp-library remeshing algorithms";

    m.def(
        "remesh_uniform",
        &py_remesh_uniform,
        R"doc(
        Uniform isotropic remeshing via edge split/collapse/flip + tangential relaxation.

        Remeshes the input triangle mesh so that all edges are close to the
        target edge length. Uses incremental remeshing with back-projection
        to the input surface.

        Parameters
        ----------
        vertices : ndarray, shape (N, 3), float64
            Input vertex positions.
        faces : ndarray, shape (M, 3), int32
            Input triangle face indices.
        edge_length : float
            Target edge length.
        iterations : int
            Number of remeshing iterations (default 10).
        use_projection : bool
            Project vertices back onto the input surface (default True).

        Returns
        -------
        vertices_out : ndarray, shape (P, 3), float64
        faces_out : ndarray, shape (Q, 3), int32
        )doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("edge_length"),
        nb::arg("iterations") = 10,
        nb::arg("use_projection") = true,
        nb::arg("verbose") = 0
    );

    m.def(
        "remesh_adaptive",
        &py_remesh_adaptive,
        R"doc(
        Adaptive isotropic remeshing driven by local curvature.

        Uses smaller triangles in high-curvature regions and larger triangles
        in flat regions, within the specified edge length bounds.

        Parameters
        ----------
        vertices : ndarray, shape (N, 3), float64
            Input vertex positions.
        faces : ndarray, shape (M, 3), int32
            Input triangle face indices.
        min_edge_length : float
            Minimum edge length (used in high-curvature areas).
        max_edge_length : float
            Maximum edge length (used in flat areas).
        approx_error : float
            Maximum geometric approximation error.
        iterations : int
            Number of remeshing iterations (default 10).
        use_projection : bool
            Project vertices back onto the input surface (default True).

        Returns
        -------
        vertices_out : ndarray, shape (P, 3), float64
        faces_out : ndarray, shape (Q, 3), int32
        )doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("min_edge_length"),
        nb::arg("max_edge_length"),
        nb::arg("approx_error"),
        nb::arg("iterations") = 10,
        nb::arg("use_projection") = true,
        nb::arg("verbose") = 0
    );
}

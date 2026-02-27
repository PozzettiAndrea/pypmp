# Binding Coverage

## Mapped

| Function | Description |
|----------|-------------|
| `remesh_uniform` | Isotropic remeshing with target edge length |
| `remesh_adaptive` | Curvature-driven adaptive remeshing |

## Not Mapped

| Capability | Notes |
|------------|-------|
| `decimate` | Edge collapse mesh simplification |
| `triangulate` | Convert n-gons to triangles |
| `catmull_clark_subdivision` | Catmull-Clark subdivision |
| `loop_subdivision` | Loop subdivision |
| `fill_hole` | Close mesh boundaries |
| `explicit_smoothing` | Explicit Laplacian smoothing |
| `implicit_smoothing` | Implicit Laplacian smoothing |
| `fair` | Surface fairing via k-harmonic equations |
| `minimize_area` / `minimize_curvature` | Surface optimization |
| `curvature` | Per-vertex curvature (min, max, mean, Gaussian) |
| `detect_features` | Sharp edge detection by dihedral angle |
| `connected_components` | Connected region analysis |
| `harmonic_parameterization` | Discrete harmonic UV mapping |
| `lscm_parameterization` | Least-squares conformal UV mapping |
| `geodesics` / `geodesics_heat` | Geodesic distance computation |
| Laplace / gradient / divergence matrices | Discrete differential operators |
| Primitive generators | Icosphere, torus, cylinder, cone, etc. |
| `read` / `write` | Mesh file I/O |
| `face_area` / `surface_area` / `volume` | Geometric property queries |
| `dual` | Geometric dual construction |

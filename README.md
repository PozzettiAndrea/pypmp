# pypmp

Python bindings for [pmp-library](https://www.pmp-library.org/) remeshing algorithms.

## Installation

```bash
pip install pypmp
```

## Usage

```python
import numpy as np
import pypmp

# Load your mesh as numpy arrays
vertices = ...  # (N, 3) float64
faces = ...     # (M, 3) int32

# Uniform remeshing — target edge length
v_out, f_out = pypmp.remesh_uniform(vertices, faces, edge_length=0.1)

# Adaptive remeshing — curvature-driven edge lengths
v_out, f_out = pypmp.remesh_adaptive(
    vertices, faces,
    min_edge_length=0.05,
    max_edge_length=0.2,
    approx_error=0.001,
)
```

## License

MIT (same as pmp-library)

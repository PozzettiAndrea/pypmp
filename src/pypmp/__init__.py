"""
pypmp — Python bindings for pmp-library remeshing.

Exposes:
- remesh_uniform: Uniform isotropic remeshing (target edge length)
- remesh_adaptive: Curvature-adaptive isotropic remeshing
"""

from pypmp.remesh import remesh_uniform, remesh_adaptive

__version__ = "0.1.0"
__all__ = [
    "remesh_uniform",
    "remesh_adaptive",
    "__version__",
]

"""
Generate visual demo of pypmp for GitHub Pages.

Shows uniform and adaptive isotropic remeshing — each with
the actual Python code used alongside before/after renders.
"""

import os
import shutil
import time
import textwrap
import numpy as np

import pyvista as pv

pv.OFF_SCREEN = True

import pypmp

OUT_DIR = os.path.join(os.path.dirname(__file__), "_site")

# Dark theme
BG_COLOR = "#1a1a2e"
MESH_COLOR_IN = "#4fc3f7"
MESH_COLOR_OUT = "#81c784"
EDGE_COLOR = "#222244"
TEXT_COLOR = "#e0e0e0"


def pv_mesh_from_numpy(verts, faces):
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def render_mesh(mesh, filename, title, color=MESH_COLOR_IN,
                window_size=(800, 600)):
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    if mesh.n_points > 0:
        pl.add_mesh(mesh, color=color, show_edges=True, edge_color=EDGE_COLOR,
                    line_width=0.5, lighting=True, smooth_shading=True)
    pl.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()


def get_mesh():
    """Get Stanford bunny, fallback to icosphere."""
    try:
        bunny = pv.examples.download_bunny()
        verts = np.array(bunny.points, dtype=np.float64)
        faces = np.array(bunny.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
        return verts, faces, "bunny.stl"
    except Exception:
        sphere = pv.Icosphere(nsub=4, radius=1.0)
        verts = np.array(sphere.points, dtype=np.float64)
        faces = np.array(sphere.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
        return verts, faces, "sphere.stl"


def run_demo(name, func, verts_in, faces_in, code, after_label="Output"):
    """Run pypmp, render before/after, return demo dict."""
    t0 = time.perf_counter()
    verts_out, faces_out = func(verts_in, faces_in)
    elapsed = time.perf_counter() - t0

    mesh_in = pv_mesh_from_numpy(verts_in, faces_in)
    mesh_out = pv_mesh_from_numpy(verts_out, faces_out)

    prefix = os.path.join(OUT_DIR, name)
    render_mesh(mesh_in, f"{prefix}_before.png",
                f"Input: {len(verts_in):,} verts, {len(faces_in):,} tris")
    render_mesh(mesh_out, f"{prefix}_after.png",
                f"{after_label}: {len(verts_out):,} verts, {len(faces_out):,} tris  ({elapsed:.2f}s)",
                color=MESH_COLOR_OUT)

    return {
        "name": name,
        "verts_in": len(verts_in),
        "faces_in": len(faces_in),
        "verts_out": len(verts_out),
        "faces_out": len(faces_out),
        "elapsed": elapsed,
        "code": code,
        "after_label": after_label,
    }


TEMPLATE_DIR = os.path.dirname(__file__)


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_demo(d):
    code_html = html_escape(d["code"])
    label = d.get("after_label", "Output")
    return f"""
    <section class="demo">
      <div class="demo-grid">
        <div class="demo-code">
          <pre><code>{code_html}</code></pre>
          <p class="timing">{d['elapsed']:.2f}s &mdash; {d['verts_in']:,} &rarr; {d['verts_out']:,} verts, {d['faces_in']:,} &rarr; {d['faces_out']:,} tris</p>
        </div>
        <div class="demo-images">
          <div class="comparison">
            <div class="panel">
              <img src="{d['name']}_before.png" alt="Before">
              <span class="label">Input</span>
            </div>
            <div class="panel">
              <img src="{d['name']}_after.png" alt="After">
              <span class="label">{label}</span>
            </div>
          </div>
        </div>
      </div>
    </section>"""


def generate_html(sections):
    sections_html = ""
    for section in sections:
        sections_html += f"""
    <h2 class="section-title">{section['title']}</h2>
    <p class="section-sub">{section['subtitle']}</p>"""
        for d in section["demos"]:
            sections_html += _render_demo(d)

    with open(os.path.join(TEMPLATE_DIR, "template.html")) as f:
        template = f.read()

    html = template.replace("{{sections}}", sections_html)

    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(html)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    verts, faces, mesh_name = get_mesh()
    sections = []

    # ── Uniform Remeshing ────────────────────────────────────────
    uniform_demos = []

    uniform_demos.append(run_demo("uniform_default",
        lambda v, f: pypmp.remesh_uniform(v, f, edge_length=0.02),
        verts, faces,
        textwrap.dedent(f"""\
            import pypmp
            import trimesh

            mesh = trimesh.load("{mesh_name}")
            v, f = pypmp.remesh_uniform(
                mesh.vertices, mesh.faces,
                edge_length=0.02,
            )"""),
        after_label="Uniform Remeshed"))

    uniform_demos.append(run_demo("uniform_fine",
        lambda v, f: pypmp.remesh_uniform(v, f, edge_length=0.01),
        verts, faces,
        textwrap.dedent(f"""\
            # Finer remesh (smaller edge length)
            v, f = pypmp.remesh_uniform(
                mesh.vertices, mesh.faces,
                edge_length=0.01,
            )"""),
        after_label="Fine Remeshed"))

    uniform_demos.append(run_demo("uniform_coarse",
        lambda v, f: pypmp.remesh_uniform(v, f, edge_length=0.05),
        verts, faces,
        textwrap.dedent(f"""\
            # Coarser remesh (larger edge length)
            v, f = pypmp.remesh_uniform(
                mesh.vertices, mesh.faces,
                edge_length=0.05,
            )"""),
        after_label="Coarse Remeshed"))

    sections.append({
        "title": "Uniform Remeshing",
        "subtitle": "Isotropic remeshing via edge split/collapse/flip with tangential relaxation",
        "demos": uniform_demos,
    })

    # ── Adaptive Remeshing ───────────────────────────────────────
    adaptive_demos = []

    adaptive_demos.append(run_demo("adaptive_default",
        lambda v, f: pypmp.remesh_adaptive(v, f,
            min_edge_length=0.005, max_edge_length=0.05, approx_error=0.001),
        verts, faces,
        textwrap.dedent(f"""\
            import pypmp

            v, f = pypmp.remesh_adaptive(
                mesh.vertices, mesh.faces,
                min_edge_length=0.005,
                max_edge_length=0.05,
                approx_error=0.001,
            )"""),
        after_label="Adaptive Remeshed"))

    adaptive_demos.append(run_demo("adaptive_tight",
        lambda v, f: pypmp.remesh_adaptive(v, f,
            min_edge_length=0.003, max_edge_length=0.03, approx_error=0.0005),
        verts, faces,
        textwrap.dedent(f"""\
            # Tighter error tolerance
            v, f = pypmp.remesh_adaptive(
                mesh.vertices, mesh.faces,
                min_edge_length=0.003,
                max_edge_length=0.03,
                approx_error=0.0005,
            )"""),
        after_label="Adaptive (tight)"))

    sections.append({
        "title": "Adaptive Remeshing",
        "subtitle": "Curvature-driven remeshing — finer in curved areas, coarser in flat regions",
        "demos": adaptive_demos,
    })

    # ── Projection vs No Projection ──────────────────────────────
    proj_demos = []

    proj_demos.append(run_demo("with_projection",
        lambda v, f: pypmp.remesh_uniform(v, f, edge_length=0.03, use_projection=True),
        verts, faces,
        textwrap.dedent(f"""\
            # With back-projection (default)
            v, f = pypmp.remesh_uniform(
                mesh.vertices, mesh.faces,
                edge_length=0.03,
                use_projection=True,
            )"""),
        after_label="With Projection"))

    proj_demos.append(run_demo("without_projection",
        lambda v, f: pypmp.remesh_uniform(v, f, edge_length=0.03, use_projection=False),
        verts, faces,
        textwrap.dedent(f"""\
            # Without back-projection (faster, less accurate)
            v, f = pypmp.remesh_uniform(
                mesh.vertices, mesh.faces,
                edge_length=0.03,
                use_projection=False,
            )"""),
        after_label="Without Projection"))

    sections.append({
        "title": "Surface Projection",
        "subtitle": "Back-projection keeps vertices on the original surface — disable for speed",
        "demos": proj_demos,
    })

    generate_html(sections)

    # Preview image for README
    try:
        from PIL import Image
        d = uniform_demos[0]
        before = Image.open(os.path.join(OUT_DIR, f"{d['name']}_before.png"))
        after = Image.open(os.path.join(OUT_DIR, f"{d['name']}_after.png"))
        w, h = before.size
        grid = Image.new("RGB", (w * 2, h), "#0d1117")
        grid.paste(before, (0, 0))
        grid.paste(after, (w, 0))
        grid.save(os.path.join(OUT_DIR, "preview.png"))
    except Exception as e:
        print(f"Skipping preview: {e}")

    print(f"Demo: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()

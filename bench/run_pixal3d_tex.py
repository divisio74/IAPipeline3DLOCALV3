#!/usr/bin/env python
"""run_pixal3d_tex.py — TEXTURE Pixal3D multi-vues d'un mesh existant (⚠ .venv-pixal3d).

Chaque pixel des vues COULEUR est retroprojete dans la grille 3D par la camera de sa vue (meme principe
que la geometrie Pixal3D) : le dos est peint depuis la vraie vue de dos, au lieu d'etre invente depuis la
seule face comme avec l'ancienne texture TRELLIS.2 (qui derivait en teinte et assombrissait, 2026-09-29).
Les vues couleur ont le MEME masque que les argiles qui ont servi a la geometrie -> meme normalisation,
meme cadrage, memes cameras. Reglages = workflow officiel « MeshTexturing with Pixal3D.json ».

Usage : run_pixal3d_tex.py <dossier vues couleur front/left/back/right.png> <mesh.glb> <sortie.glb>
"""
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pixal3d_mv as R  # noqa: E402  (charge aussi l'env ATTN_BACKEND=sdpa)


def main():
    vdir, mesh_p, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    pils = [R.normalise(Image.open(vdir / f"{v}.png").convert("RGBA")) for v, _ in R.VUES]
    images = R.to_comfy(pils)
    C = R.charge_noeuds()
    import t2node.trellis2.modules.sparse.config as sparseconfig
    _set = sparseconfig.set_attn_backend
    sparseconfig.set_attn_backend = lambda b: _set("sdpa")
    pipe, = R.appelle(C, "Trellis2LoadModel", ["TencentARC/Pixal3D", "sdpa", "cuda", True, False,
                                               "flex_gemm", "flash_attn", False, True])[:1]
    mv = R.appelle(C, "Trellis2Pixal3DMultiViewConfig",
                   [",".join(str(a) for _, a in R.VUES), ",".join("0" for _ in R.VUES), 0.1, "rad", 1, 0,
                    False, False, "auto"], images=images)[0]
    tm = R.appelle(C, "Trellis2LoadMesh", [str(mesh_p.resolve()), False])[0]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = R.appelle(C, "Trellis2MeshTexturingPixal3DMultiView",
                    [12345, "fixed", int(os.environ.get("P3D_PAS_TEX", 12)), 1, 0.5, 3, 1024, 2048, "OPAQUE", False, 0.6, 0.9, False, False, 60,
                     "auto", "euler", "telea", False, 0, 5, 1],
                    pipeline=pipe, trimesh=tm, pixal3d_mv_views=mv)
    tex_mesh = out[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tex_mesh.export(str(dst))
    print(f"[p3d-tex] {len(tex_mesh.faces)} faces, {time.time() - t0:.0f}s, "
          f"pic VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} Go -> {dst}", flush=True)


if __name__ == "__main__":
    main()

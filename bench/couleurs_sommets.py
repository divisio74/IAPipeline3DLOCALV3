#!/usr/bin/env python
"""couleurs_sommets.py (2026-10-11) — couleur de chaque sommet de geo_16k prise sur le modèle texturé tex_p3d (moyenne
des 8 points échantillonnés les plus proches), dans l'ordre des sommets du glb (pour un export animé sans UV).
Usage (.venv-trellis2) : python bench/couleurs_sommets.py <id>  ->  bench/out/<id>/geo_16k_couleurs.npy (N, 3) dans [0, 1]"""
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
O = f"C:/Users/ec/Desktop/documents/DEV_IA/pipeline IA/bench/out/{sys.argv[1]}"
geo = trimesh.load(f"{O}/geo_16k.glb", force="mesh", process=False)
tex = trimesh.load(f"{O}/tex_p3d.glb", force="mesh", process=False)
pts, fi = tex.sample(400000, return_index=True)
img = np.asarray(tex.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255
bary = trimesh.triangles.points_to_barycentric(tex.triangles[fi], pts)
puv = (tex.visual.uv[tex.faces[fi]] % 1.0 * bary[:, :, None]).sum(1)
h, w = img.shape[:2]
col = img[((1 - puv[:, 1]) * (h - 1)).astype(int).clip(0, h - 1), (puv[:, 0] * (w - 1)).astype(int).clip(0, w - 1)]
_, nn = cKDTree(pts).query(geo.vertices, k=8)
np.save(f"{O}/geo_16k_couleurs.npy", col[nn].mean(1).astype(np.float32))
print("[couleurs]", len(geo.vertices), "sommets")

#!/usr/bin/env python
"""t2_simplify.py — decimation d'un mesh TRELLIS.2 / Pixal3D vers une cible de
triangles, avec la recette du noeud ComfyUI-Trellis2 (CuMesh, GPU).

POURQUOI PAS bl_decimate_glb.py : TRELLIS.2 produit volontairement des surfaces
ouvertes et de la geometrie non-manifold (c'est ce qui garde les tentacules
fines). Mesure sur gen_slenderman : 23 392 aretes non-manifold et 27 121 bords
ouverts -> le Decimate de Blender refuse d'effondrer ces aretes et plafonne
(2,37 M -> 116 708 tris, puis 111 087 en 2e passe, cible 19 000 jamais
atteinte). CuMesh repare les aretes non-manifold ENTRE deux simplifications.

⚠ A lancer avec .venv-trellis2 (cumesh est une wheel cp311/Torch270).
Usage : .venv-trellis2/Scripts/python.exe bench/t2_simplify.py <in.glb> <out.glb> [cible=19000]
"""
import sys

import numpy as np
import torch
import trimesh
import cumesh as CuMesh


def main():
    src, dst = sys.argv[1], sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 19000
    m = trimesh.load(src, force="mesh", process=False)
    v = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32)).cuda()
    f = torch.from_numpy(np.asarray(m.faces, dtype=np.int32)).cuda()

    cm = CuMesh.CuMesh()
    cm.init(v, f)
    print(f"[t2s] entree : {cm.num_vertices} sommets, {cm.num_faces} faces", flush=True)

    def clean():
        cm.remove_duplicate_faces()
        cm.repair_non_manifold_edges()
        cm.remove_small_connected_components(1e-5)

    # recette du noeud (nodes.py, branche standard) : 3x la cible, nettoyage,
    # cible, nettoyage
    cm.simplify(target * 3, verbose=False)
    clean()
    cm.simplify(target, verbose=False)
    clean()

    v2, f2 = cm.read()
    out = trimesh.Trimesh(v2.cpu().numpy(), f2.cpu().numpy(), process=False)
    out.export(dst)
    print(f"[t2s] sortie : {len(out.vertices)} sommets, {len(out.faces)} faces -> {dst}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""run_pixal3d_mv.py — Pixal3D MULTI-VUES (TRELLIS.2 + cameras EXPLICITES) hors de ComfyUI.

⚠ .venv-pixal3d (Python 3.13, torch 2.10 cu130, natten) : Pixal3D passe par NAF, qui exige natten.
Pourquoi Pixal3D : TRELLIS.2 n'a aucune entree de pose camera (il DEVINE d'ou il regarde) -> son
multi-vues melange des devinettes contradictoires (test du 2026-09-29). Pixal3D retroprojette les
features de chaque vue dans la grille 3D a partir de sa camera (azimut 0/90/180/270).

Rejoue le workflow officiel du noeud (example_workflows/MeshOnly_MultiView_Pixal3D.json) en appelant
les classes de noeuds directement, avec LEURS valeurs de widgets.
Les vues doivent etre cadrees comme un tour de camera (meme distance) : on les NORMALISE ici (meme
hauteur de silhouette, centre de la boite au centre de l'image). Azimuts : 0 face, 90 « left » =
creature tournee vers le BORD GAUCHE de l'image, 180 dos, 270 « right » (convention Blender du noeud).

Usage : run_pixal3d_mv.py <dossier_vues avec front/left/back/right.png RGBA> <sortie.glb> [GI]
  GI = intervalle de guidage, ex. 0.3,1.0 (defaut : celui du workflow officiel, 0.6,1)
"""
import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
COMFY = ROOT / "third_party" / "ComfyUI"
NODE = COMFY / "custom_nodes" / "ComfyUI-Trellis2"
sys.path.insert(0, str(COMFY))
os.environ.setdefault("ATTN_BACKEND", "sdpa")     # lu a l'IMPORT de trellis2 (pas de flash_attn sous Windows)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
VUES = [("front", 0), ("left", 90), ("back", 180), ("right", 270)]
# P3D_VUES=front -> TEMOIN a une vue : memes poids MV, meme code (V=1 degenere en vue unique, cf. le noeud)
VUES = [v for v in VUES if v[0] in os.environ.get("P3D_VUES", "front,left,back,right").split(",")]
_CTRL = {"fixed", "randomize", "increment", "decrement"}


def normalise(img, taille=1024, haut=0.80):
    """Silhouette a hauteur commune, centre de sa boite au centre : ce qu'une camera en orbite
    (elevation 0) verrait d'un objet rigide — la hauteur ne change pas d'une vue a l'autre, et le
    centre de la boite 2D de chaque vue est la projection du centre de la boite 3D."""
    a = np.asarray(img)[..., 3] > 127
    ys, xs = np.where(a)
    c = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    s = haut * taille / c.height
    c = c.resize((max(1, round(c.width * s)), round(c.height * s)), Image.LANCZOS)
    out = Image.new("RGBA", (taille, taille), (0, 0, 0, 0))
    out.paste(c, ((taille - c.width) // 2, (taille - c.height) // 2), c)
    return out


def to_comfy(pils):
    """liste de PIL RGBA -> IMAGE RGBA [B,H,W,4] 0..1 (comme Trellis2LoadImageWithTransparency[2])."""
    return torch.from_numpy(np.stack([np.asarray(p).astype(np.float32) / 255.0 for p in pils]))


def charge_noeuds():
    spec = importlib.util.spec_from_file_location("t2node", NODE / "__init__.py",
                                                  submodule_search_locations=[str(NODE)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t2node"] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


def appelle(classes, nom, widgets, **liens):
    """Appelle un noeud : `widgets` = widgets_values du workflow (dans l'ordre d'INPUT_TYPES),
    `liens` = entrees connectees (par nom)."""
    cls = classes[nom]
    it = cls.INPUT_TYPES()
    params = list(it.get("required", {}).items()) + list(it.get("optional", {}).items())
    kw, w = {}, list(widgets)
    for pname, spec in params:
        t = spec[0]
        est_widget = isinstance(t, list) or t in ("INT", "FLOAT", "STRING", "BOOLEAN")
        if not est_widget or not w:
            continue
        val = w.pop(0)              # un widget BRANCHE garde sa place dans widgets_values
        if t == "INT" and w and isinstance(w[0], str) and w[0] in _CTRL:
            w.pop(0)
        if pname not in liens:
            kw[pname] = val
    kw.update(liens)
    fn = getattr(cls(), cls.FUNCTION)
    acc = inspect.signature(fn).parameters
    kw = {k: v for k, v in kw.items() if k in acc or any(p.kind == p.VAR_KEYWORD for p in acc.values())}
    t0 = time.time()
    out = fn(**kw)
    if isinstance(out, dict) and "result" in out:
        out = out["result"]
    print(f"[p3d] {nom} : {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    vdir, dst = Path(sys.argv[1]), Path(sys.argv[2])
    gi = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    pils = [normalise(Image.open(vdir / f"{v}.png").convert("RGBA")) for v, _ in VUES]
    dst.parent.mkdir(parents=True, exist_ok=True)
    for (v, _), p in zip(VUES, pils):
        p.save(dst.parent / f"p3d_vue_{v}.png")          # tracabilite : ce que le modele a recu
    images = to_comfy(pils)

    C = charge_noeuds()
    # le noeud n'offre que xformers / flash_attn pour l'attention CREUSE (absents sous Windows) :
    # on force sdpa (supporte par trellis2/modules/sparse, cf. run_trellis2.py)
    import t2node.trellis2.modules.sparse.config as sparseconfig
    _set = sparseconfig.set_attn_backend
    sparseconfig.set_attn_backend = lambda b: _set("sdpa")
    pipe, = appelle(C, "Trellis2LoadModel", ["TencentARC/Pixal3D", "sdpa", "cuda", True, False,
                                             "flex_gemm", "flash_attn", False, True])[:1]
    cam, = appelle(C, "Trellis2FovMoGeCameraConfig", [0.1, "rad", 0, 1])[:1]
    face = appelle(C, "Trellis2PreProcessImage", [5, False, 1024], image=images[:1])[0]
    mv, cam2, _ = appelle(C, "Trellis2Pixal3DMultiViewConfig",
                          [",".join(str(a) for _, a in VUES), ",".join("0" for _ in VUES), 20, "deg", 1, 0, False,
                           False, "auto"],
                          images=images, moge_camera_config=cam)
    cond, cond2, pipe = appelle(C, "Trellis2ImageCondGenerator", [16], pipeline=pipe, image=face)[:3]
    g0, g1 = gi if gi else (0.6, 1)
    # nombres d'étapes réglables pour les essais de vitesse (2026-10-11) ; défauts = réglages validés
    ps, pf = int(os.environ.get("P3D_PAS_SPARSE", 25)), int(os.environ.get("P3D_PAS_FORME", 12))
    smp = os.environ.get("P3D_SAMPLER", "heun")
    coords, ssres, pipe = appelle(
        C, "Trellis2SparseGenerator",
        [1000, "fixed", ps, 7.5, 0.7, 5, smp, 32, g0, g1, True, 1, False, 0.65, 2, "flood_fill", 0.92, True],
        pipeline=pipe, image_cond=cond, image=face, moge_camera_config=cam2, pixal3d_mv_views=mv)
    slat, res, pipe = appelle(
        C, "Trellis2ShapeGenerator", [512, pf, 7.5, 0.5, 3, smp, g0, g1, False, 0.65, 2, 0.92],
        pipeline=pipe, image_cond=cond, coords=coords, image=face, moge_camera_config=cam2,
        pixal3d_mv_views=mv)
    slat2, res2, pipe, _ = appelle(
        C, "Trellis2ShapeCascadeGenerator",
        [0, 1024, 32, 999999, pf, 7.5, 0.5, 3, smp, g0, g1, False, 0.65, 2, 0.92],
        pipeline=pipe, image_cond=cond2, shape_slat=slat, image=face, moge_camera_config=cam2,
        pixal3d_mv_views=mv, from_resolution=res, sparse_structure_resolution=ssres)
    # decodeur en tuiles COUPE : pans non cousus (mesure du 2026-09-28) -> False (workflow : True)
    mesh = appelle(C, "Trellis2DecodeLatents", [0, False], pipeline=pipe, shape_slat=slat2,
                   resolution=res2)[0]
    mesh = appelle(C, "Trellis2ReconstructMeshWithQuad", [1, 1024, True, True], mesh=mesh)[0]
    mesh = appelle(C, "Trellis2SimplifyMesh", [500000, "Cumesh"], mesh=mesh)[0]
    mesh = appelle(C, "Trellis2FillHolesWithMeshlib", [], mesh=mesh)[0]
    tm = appelle(C, "Trellis2MeshWithVoxelToTrimesh", ["None"], mesh=mesh)[0]
    import trimesh
    ext = tm.extents
    print(f"[p3d] etendues brutes x={ext[0]:.3f} y={ext[1]:.3f} z={ext[2]:.3f}", flush=True)
    # Sortie DEJA en Y-haut (verifie sur Siren Head et Slenderman, 2026-09-29) : aucun redressement.
    # (Un test « z > y » ferait basculer a tort un monstre plus long que haut, ex. Godzilla et sa queue.)
    tm.export(str(dst))
    print(f"[p3d] {len(tm.faces)} faces, pic VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} Go -> {dst}",
          flush=True)


if __name__ == "__main__":
    main()

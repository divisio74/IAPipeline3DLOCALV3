#!/usr/bin/env python
"""exp_vues_3d.py — les vues Qwen (edition du concept) ameliorent-elles la 3D de Siren Head ?

Vues : face = concept v2q (cand_31474), gauche_8 / droite_8 (profils stricts), dos_precis_9.
MEME traitement que la face pour chaque vue : detourage birefnet-dis + argile A2 (seule variable =
le nombre de vues). Conventions :
  - Hunyuan3D-2mv : « left » = creature tournee vers le BORD GAUCHE de l'image (verifie sur ses
    exemples, assets/example_mv_images) = notre « gauche ».
  - TRELLIS.2 run_multiview : melange spatial, gauche = -X ; correspondance image INCONNUE
    -> M1 (left = gauche) et M2 (left = droite), comme en 2026-09-29.

  prep : argiles des 3 vues -> inputs/v2q_siren_head/vq_M1|vq_M2/{left,right,back}.png
         et views/v2q_siren_head/{front,left,right,back}.png (Hunyuan)
  hy   : Hunyuan3D-2 mono (argile de face) et 2mv (4 argiles), forme SEULE, sans peinture
         -> out/v2q_siren_head/hy_mono.glb, hy_mv.glb
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
NOM = "v2q_siren_head"
INP = ROOT / "bench" / "inputs" / NOM
OUT = ROOT / "bench" / "out" / NOM
VQ = ROOT / "bench" / "inputs" / "q21_vues_siren_head"
SRC = {"gauche": VQ / "gauche_8.png", "droite": VQ / "droite_8.png", "dos": VQ / "dos_precis_9.png"}
DESC = ("two large grey metal air raid siren loudspeakers mounted on top of a tall thin pole instead of a head, "
        "Siren Head, giant emaciated creature, dry cracked brown skin, extremely long skinny arms and legs, "
        "long bony fingers")


def prep():
    from make_monster_v2 import argile_A2, masque_dis
    argiles = {}
    for v, p in SRC.items():
        rgb_img = Image.open(p).convert("RGB")
        m = masque_dis(rgb_img)
        _, a2 = argile_A2(np.asarray(rgb_img), DESC, m)
        argiles[v] = a2
        print(f"[vues] argile {v} ok (objet {m.mean() * 100:.1f} % de l'image)", flush=True)
    for d, (l, r) in {"vq_M1": ("gauche", "droite"), "vq_M2": ("droite", "gauche")}.items():
        (INP / d).mkdir(parents=True, exist_ok=True)
        argiles[l].save(INP / d / "left.png"); argiles[r].save(INP / d / "right.png")
        argiles["dos"].save(INP / d / "back.png")
    hv = ROOT / "bench" / "views" / NOM
    hv.mkdir(parents=True, exist_ok=True)
    Image.open(INP / "02_cut.png").save(hv / "front.png")
    argiles["gauche"].save(hv / "left.png"); argiles["droite"].save(hv / "right.png")
    argiles["dos"].save(hv / "back.png")
    print("[vues] prep TERMINE", flush=True)


def hy():
    import torch
    sys.path.insert(0, str(ROOT / "third_party" / "Hunyuan3D-2"))
    from hy3dgen.shapegen import DegenerateFaceRemover, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline
    hv = ROOT / "bench" / "views" / NOM
    for mode in ("mono", "mv"):
        if mode == "mono":
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
            image = Image.open(hv / "front.png")
        else:
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16")
            image = {v: Image.open(hv / f"{v}.png") for v in ("front", "left", "back", "right")}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mesh = pipe(image=image, octree_resolution=384, num_inference_steps=50,
                    generator=torch.manual_seed(2024), output_type="trimesh")[0]
        mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
        mesh.export(str(OUT / f"hy_{mode}.glb"))
        print(f"[vues] hunyuan {mode} : {len(mesh.faces)} faces, {time.time() - t0:.0f}s, "
              f"pic VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} Go", flush=True)
        del pipe
        torch.cuda.empty_cache()


def _config_slenderman():
    """Slenderman : concept de reference gen_slenderman (face = argile A2 deja faite, cmp_slenderman_A2)."""
    global NOM, INP, OUT, VQ, SRC, DESC
    NOM = "v2q_slenderman"
    INP = ROOT / "bench" / "inputs" / NOM
    OUT = ROOT / "bench" / "out" / NOM
    VQ = ROOT / "bench" / "inputs" / "q21_vues_slenderman"
    SRC = {"gauche": VQ / "gauche_{g}.png", "droite": VQ / "droite_{d}.png", "dos": VQ / "dos_precis_{b}.png"}
    DESC = "Slender Man, extremely tall thin faceless humanoid in a suit, several long tentacles from its back"
    INP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if not (INP / "02_cut.png").exists():
        Image.open(ROOT / "bench" / "inputs" / "cmp_slenderman_A2" / "02_cut.png").save(INP / "02_cut.png")


def prep_job(chemin, ref, g, d, b):
    """Chaine v3 (2026-09-29) depuis une fiche bench/jobs/<id>.json : argile A2 des 4 vues (face = concept
    retenu), concept colore detoure pour la texture. -> views/<id>/{front,left,right,back}.png,
    inputs/<id>/02_cut.png, out/<id>/couleur_rgba.png"""
    import json
    from make_monster_v2 import argile_A2, masque_dis
    j = json.load(open(chemin, encoding="utf-8"))
    ident, desc = j["id"], j["desc_courte"]
    inp, out, hv = ROOT / "bench" / "inputs", ROOT / "bench" / "out" / ident, ROOT / "bench" / "views" / ident
    vq = inp / f"q21_vues_{ident}"
    src = {"front": inp / f"q21_{ident}_nat" / ref, "left": vq / f"gauche_{g}.png",
           "right": vq / f"droite_{d}.png", "back": vq / f"dos_precis_{b}.png"}
    for p in (out, hv, inp / ident):
        p.mkdir(parents=True, exist_ok=True)
    for v, p in src.items():
        rgb_img = Image.open(p).convert("RGB")
        m = masque_dis(rgb_img)
        if v == "front":
            Image.fromarray(np.dstack([np.asarray(rgb_img), np.where(m, 255, 0)]).astype(np.uint8)).save(
                out / "couleur_rgba.png")
        _, a2 = argile_A2(np.asarray(rgb_img), desc, m)
        a2.save(hv / f"{v}.png")
        if v == "front":
            a2.save(inp / ident / "02_cut.png")
        print(f"[vues] {ident} argile {v} ok (objet {m.mean() * 100:.1f} % de l'image)", flush=True)


def couleurs_job(chemin, ref, g, d, b):
    """Vues COULEUR detourees avec le MEME masque birefnet-dis que les argiles (donc meme cadrage apres
    normalisation) -> views_couleur/<id>/{front,left,right,back}.png, pour la texture Pixal3D multi-vues."""
    import json
    from make_monster_v2 import masque_dis
    j = json.load(open(chemin, encoding="utf-8"))
    ident = j["id"]
    inp, hv = ROOT / "bench" / "inputs", ROOT / "bench" / "views_couleur" / ident
    vq = inp / f"q21_vues_{ident}"
    src = {"front": inp / f"q21_{ident}_nat" / ref, "left": vq / f"gauche_{g}.png",
           "right": vq / f"droite_{d}.png", "back": vq / f"dos_precis_{b}.png"}
    hv.mkdir(parents=True, exist_ok=True)
    for v, p in src.items():
        rgb_img = Image.open(p).convert("RGB")
        m = masque_dis(rgb_img)
        Image.fromarray(np.dstack([np.asarray(rgb_img), np.where(m, 255, 0)]).astype(np.uint8)).save(hv / f"{v}.png")
        print(f"[vues] {ident} couleur {v} ok", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "couleurs-job":
        couleurs_job(sys.argv[2], sys.argv[3], *sys.argv[4:7])
        sys.exit(0)
    # exp_vues_3d.py prep-job <fiche.json> <cand_x.png> <graine gauche> <graine droite> <graine dos>
    if sys.argv[1] == "prep-job":
        prep_job(sys.argv[2], sys.argv[3], *sys.argv[4:7])
        sys.exit(0)
    # exp_vues_3d.py prep|hy [slenderman gauche=7 droite=7 dos=7]
    if len(sys.argv) > 2 and sys.argv[2] == "slenderman":
        _config_slenderman()
        g = dict(x.split("=") for x in sys.argv[3:])
        SRC = {k: Path(str(v).format(g=g.get("gauche", 7), d=g.get("droite", 7), b=g.get("dos", 7)))
               for k, v in SRC.items()}
    {"prep": prep, "hy": hy}[sys.argv[1]]()

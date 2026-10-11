#!/usr/bin/env python
"""directeur2.py — DIRECTEUR D'ANIMATION, ETAPE 1 : passes « membres » puis « objets » (2026-09-30).

Vision du user : le VLM regarde en plusieurs passes avec un CONTEXTE (la fiche du monstre) : d'abord les membres,
ensuite les armes/objets ; puis un rapport par element ; et la position de chaque element doit etre CLAIRE en 3D
pour le modele qui anime.
Ici (etape 1) :
  passe 1 : liste des MEMBRES (types, sans gauche/droite) + cadres sur 4 vues a CAMERAS CONNUES ;
  passe 2 : liste des OBJETS tenus/portes (sachant les membres trouves) + cadres ;
  3D      : les cameras ortho etant connues, un point du modele appartient a un element s'il tombe dans ses cadres
            sur les vues ou l'element est visible (« sculpture par cadres ») -> chaque element devient une region
            3D du modele. Pas de PartField, pas de regle de position : le VLM reconnait, la geometrie projette.
Sorties : directeur2/elements.json (+ etiquette 3D de chaque sommet) et une planche de controle.
Usage (⚠ .venv-trellis2) : python bench/directeur2.py <m>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

B = Path(__file__).resolve().parent
VUES = ["face", "gauche", "dos", "droite"]
NOM_VUE = {"face": "front", "gauche": "left side (the creature faces the left edge of the image)", "dos": "back",
           "droite": "right side (the creature faces the right edge of the image)"}


class VLM:
    def __init__(self, mid="Qwen/Qwen3-VL-4B-Instruct"):
        self.proc = AutoProcessor.from_pretrained(mid, min_pixels=128 * 28 * 28, max_pixels=448 * 28 * 28)
        self.model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()

    def __call__(self, images, texte, n=60):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in images]
                 + [{"type": "text", "text": texte}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=n, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def liste(rep):
    rep = re.sub(r"[\n.;]", ",", rep.lower())
    items = [re.sub(r"^(left|right|a|an|the) ", "", x.strip(" -*'\"")) for x in rep.split(",")]
    return list(dict.fromkeys(x for x in items if x and x not in ("none", "nothing", "no")))[:10]


def projeter(P, vue, cam):
    """sommets glTF (Y haut) -> pixels (u, v) de la vue ortho (cf. bl_directeur.py)"""
    cx, cy, cz = cam["centre_gltf"]
    S, R = cam["ortho"], cam["res"]
    h = {"face": P[:, 0] - cx, "dos": cx - P[:, 0], "gauche": cz - P[:, 2], "droite": P[:, 2] - cz}[vue]
    return (h / S + 0.5) * R, (0.5 - (P[:, 1] - cy) / S) * R


def main():
    m = sys.argv[1]
    D = B / "out" / f"v3_{m}" / "directeur2"
    cam = json.load(open(D / "vue_cameras.json"))
    job = json.load(open(B / "jobs" / f"v3_{m}.json", encoding="utf-8"))
    im = {v: Image.open(D / f"vue_{v}.png").convert("RGB") for v in VUES}
    ctx = f"Context: this 3D game monster was made from this description: \"{job['concept'][:350]}\" "
    vlm = VLM()

    # PASSE 1 : membres
    membres = liste(vlm(list(im.values()), ctx + "The 4 images show it from the front, left side, back and right "
                        "side. List the parts of its BODY (limbs and body segments), as types, singular, without "
                        "left/right (for example: head, torso, arm, leg, tail, wing, tentacle, claw). Do not include "
                        "objects it holds or wears. Answer with a comma-separated list."))
    # PASSE 2 : objets, en sachant les membres
    objets = liste(vlm(list(im.values()), ctx + f"Its body parts are: {', '.join(membres)}. Now list the separate "
                       "OBJECTS it holds, carries or wears (weapons, tools, shields, accessories, clothes), or "
                       "'none'. Answer with a comma-separated list."))
    objets = [o for o in objets if o not in membres]
    print(f"[directeur2] passe 1 membres : {membres}\n[directeur2] passe 2 objets : {objets}", flush=True)

    elements = [{"nom": e, "genre": "membre"} for e in membres] + [{"nom": o, "genre": "objet"} for o in objets]
    for e in elements:
        e["cadres"] = {}
        for v in VUES:
            r = vlm([im[v]], f"This is the {NOM_VUE[v]} view of a 3D creature. Locate every {e['nom']} of it in "
                    "the image (there may be several, or none if hidden). Output a JSON list: "
                    "[{\"bbox_2d\": [x1, y1, x2, y2]}, ...], or [] if none is visible.", n=160)
            e["cadres"][v] = [[int(x) / 1000 for x in g] for g in
                              re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", r)]
        print(f"[directeur2] « {e['nom']} » cadres par vue : "
              f"{ {v: len(e['cadres'][v]) for v in VUES} }", flush=True)

    # 3D : sculpture par cadres sur le modele texture
    mesh = trimesh.load(B / "out" / f"v3_{m}" / "tex_p3d.glb", force="mesh", process=False)
    P = np.asarray(mesh.vertices)
    R = cam["res"]
    score = np.zeros((len(elements), len(P)))
    aire = np.zeros(len(elements))
    for j, e in enumerate(elements):
        n_vues = 0
        for v in VUES:
            cs = e["cadres"][v]
            if not cs:
                continue
            n_vues += 1
            u, w = projeter(P, v, cam)
            dedans = np.zeros(len(P), bool)
            for c in cs:
                dedans |= (u >= c[0] * R) & (u <= c[2] * R) & (w >= c[1] * R) & (w <= c[3] * R)
                aire[j] += (c[2] - c[0]) * (c[3] - c[1])
            score[j] += dedans
        if n_vues:
            score[j] /= n_vues
            aire[j] /= n_vues
    # un sommet va a l'element dont il est dans les cadres sur (presque) toutes les vues ou celui-ci est visible ;
    # a egalite, l'element le plus PETIT (le plus precis : la batte plutot que le bras qui la chevauche)
    etiq = np.full(len(P), -1)
    for i in range(len(P)):
        ok = np.nonzero(score[:, i] >= 0.75)[0]
        if len(ok):
            meilleur = score[ok, i].max()
            ok = [j for j in ok if score[j, i] >= meilleur - 1e-6]
            etiq[i] = min(ok, key=lambda j: aire[j])
    for j, e in enumerate(elements):
        e["sommets"] = int((etiq == j).sum())
        sel = P[etiq == j]
        e["boite_3d"] = [sel.min(0).round(3).tolist(), sel.max(0).round(3).tolist()] if len(sel) else None
        print(f"[directeur2] {e['genre']} « {e['nom']} » : {e['sommets']} sommets, boite 3D {e['boite_3d']}")
    print(f"[directeur2] sommets sans element : {(etiq < 0).sum()} / {len(P)}")
    np.save(D / "etiquettes_sommets.npy", etiq)
    json.dump({"membres": membres, "objets": objets, "elements": elements}, open(D / "elements.json", "w"),
              indent=1, ensure_ascii=False)

    # controle : cadres sur les 4 vues + modele colore par element (pour Blender)
    pal = np.array([(230, 60, 60), (60, 120, 230), (60, 190, 90), (240, 170, 30), (170, 80, 200), (40, 190, 200),
                    (230, 110, 180), (140, 140, 60), (120, 70, 40), (200, 200, 200)])
    f = ImageFont.truetype("arial.ttf", 13)
    tuiles = []
    for v in VUES:
        a = im[v].copy()
        d = ImageDraw.Draw(a)
        for j, e in enumerate(elements):
            for c in e["cadres"][v]:
                d.rectangle([c[0] * R, c[1] * R, c[2] * R, c[3] * R], outline=tuple(pal[j % 9]), width=3)
                d.text((c[0] * R + 3, c[1] * R + 2), e["nom"], fill=tuple(pal[j % 9]), font=f)
        tuiles.append(a)
    pl = Image.new("RGB", (R * 4, R), (0, 0, 0))
    for k, t in enumerate(tuiles):
        pl.paste(t, (k * R, 0))
    pl.save(D / "planche_cadres.png")
    col = np.where(etiq[:, None] >= 0, pal[np.maximum(etiq, 0) % 9], 90).astype(np.uint8)
    mm = trimesh.Trimesh(P, mesh.faces, process=False)
    mm.visual = trimesh.visual.ColorVisuals(mm, vertex_colors=np.hstack([col, np.full((len(P), 1), 255, np.uint8)]))
    mm.export(D / "elements_colores.glb")
    print(f"[directeur2] planche -> {D / 'planche_cadres.png'}")


if __name__ == "__main__":
    main()

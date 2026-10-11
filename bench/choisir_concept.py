#!/usr/bin/env python
"""choisir_concept.py — CHOIX AUTOMATIQUE DU CONCEPT parmi les candidats générés (2026-10-11).
Remplace le choix « à l'œil ». Le VLM (Qwen3-VL-4B, celui du directeur) note chaque candidat sur ce dont la suite de
la pipeline a besoin, sans rien de propre à un type de créature :
  - NOMBRES : chaque partie dont la description donne le nombre (« four arms », « three heads ») est comptée sur
    l'image ; un compte faux est éliminatoire en pratique (la 3D et l'animation héritent de l'erreur) ;
  - CADRAGE ET POSE : vue de face, corps entier visible, une seule créature, membres écartés du corps et entre eux
    (règle de pose « membres séparés »), fond uni sans sol ni ombre ;
  - FIDÉLITÉ : note de 0 à 10 de la ressemblance à la description.
Score = 10 x nombres justes + 3 x critères de cadrage/pose remplis + fidélité. Les candidats à 3 points du
meilleur sont départagés par l'ÉCARTEMENT de la silhouette (part de vide dans son enveloppe convexe).
Usage (⚠ .venv-trellis2) : python bench/choisir_concept.py <id> [dossier des candidats]
Sortie : <dossier>/choix.json et le nom du candidat retenu en dernière ligne (« [concept] retenu cand_XXXX.png »)"""
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allure_marche import CHIFFRES, NUM  # noqa: E402
from directeur2 import VLM  # noqa: E402

B = Path(__file__).resolve().parent
CRITERES = [
    ("face", "Is the creature seen from the front, facing the camera (not from the side, not from behind)?"),
    ("entier", "Is the whole body visible, from the top of the head to the feet, without being cut by the image border?"),
    ("unique", "Is there exactly one single creature in the image (no second character, no duplicate)?"),
    ("separes", "Are all its limbs (arms, legs, wings, tentacles) held clearly away from its body and from each other, "
                "with visible gaps between them, rather than touching, crossing or hidden behind the body?"),
    ("fond", "Is the background a plain uniform colour, with no floor, no ground, no pedestal, no scenery and no cast "
             "shadow?"),
]


def nombres(desc):
    """{partie: nombre} lus dans la description : nombre suivi (à 4 mots près) d'un nom au pluriel ou d'un adjectif
    en -ed (« four arms », « three-headed », « two large leathery wings ») ; générique, pas de liste de parties"""
    inv = {v: k for k, v in CHIFFRES.items()}
    r = {}
    for m in re.finditer(r"\b" + NUM + r"\b", desc, re.I):
        suite = re.split(r",|;|\bwith\b|\band\b|\bon\b", desc[m.end():], maxsplit=1, flags=re.I)[0]
        mots = [w.strip("-").lower() for w in re.findall(r"[\w-]+", suite)[:4]]
        noms = [w for w in mots if (w.endswith("s") and len(w) > 3) or w.endswith("ed")]
        if noms:
            w = m.group(1).lower()
            n = int(w) if w.isdigit() else inv[w]
            nom = noms[-1]
            if nom.endswith("ed"):                                  # « headed » -> « head », « legged » -> « leg »
                nom = nom[:-2]
                nom = nom[:-1] if len(nom) > 2 and nom[-1] == nom[-2] else nom
            nom = nom[:-1] if nom.endswith("s") else nom
            if n >= 2:
                r[nom] = n
    return r


def ecartement(m):
    """part de VIDE dans l'enveloppe convexe de la silhouette (masque BiRefNet, booléen)"""
    import numpy as np
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return 0.0
    return 1.0 - m.sum() / ConvexHull(np.c_[xs, ys]).volume


def oui(rep):
    return rep.strip().lower().startswith("yes")


def main():
    ident = sys.argv[1]
    D = Path(sys.argv[2]) if len(sys.argv) > 2 else B / "inputs" / f"q21_{ident}_nat"
    job = json.load(open(B / "jobs" / f"{ident}.json", encoding="utf-8"))
    desc = job["desc_courte"]
    att = nombres(desc)
    cands = sorted(D.glob("cand_*.png"))
    if not cands:
        sys.exit(f"aucun candidat dans {D}")
    # silhouettes BiRefNet d'abord (détourage de masque_gpu.py), puis libération avant de charger le VLM
    import torch
    from masque_gpu import Detoureur
    det = Detoureur()
    masques = {c.name: det(Image.open(c).convert("RGB").resize((512, 512 * Image.open(c).height // Image.open(c).width)))
               > 127 for c in cands}
    del det
    torch.cuda.empty_cache()
    vlm = VLM()
    res = {}
    for c in cands:
        im = Image.open(c).convert("RGB")
        r = {"nombres": {}, "criteres": {}}
        for nom, n in att.items():
            rep = vlm([im], f"How many {nom}s does the creature in this image have? Count carefully, including the "
                             "ones partly hidden. Answer with a single number.", n=8)
            m = re.search(r"\d+", rep)
            r["nombres"][nom] = [int(m.group(0)) if m else -1, n]
        for k, q in CRITERES:
            r["criteres"][k] = oui(vlm([im], q + " Answer YES or NO.", n=4))
        rep = vlm([im], f"This image should show: \"{desc}\". How faithful is it to this description, from 0 (not at "
                         "all) to 10 (perfect)? Answer with a single number.", n=6)
        m = re.search(r"\d+", rep)
        r["fidelite"] = min(10, int(m.group(0))) if m else 0
        r["score"] = (10 * sum(a == b for a, b in r["nombres"].values()) + 3 * sum(r["criteres"].values())
                      + r["fidelite"])
        res[c.name] = r
        print(f"[concept] {c.name} : score {r['score']} | nombres {r['nombres']} | "
              f"{[k for k, v in r['criteres'].items() if not v] or 'cadrage/pose ok'} | fidélité {r['fidelite']}",
              flush=True)
    # DÉPARTAGE des ex æquo (à 3 points du meilleur) : les notes du VLM se ressemblent toutes (9/10) et, en duel
    # (deux images côte à côte), Qwen3-VL-4B répond « Image 1 » quel que soit l'ordre (biais de position, essai du
    # 2026-10-11). On départage donc par une MESURE : l'écartement de la silhouette = part de vide dans son enveloppe
    # convexe (membres écartés du corps et entre eux = règle de pose « membres séparés », utile au rig et à l'animation)
    top = max(r["score"] for r in res.values())
    for k in res:
        res[k]["ecartement"] = round(ecartement(masques[k]), 3)
    meil = max(res, key=lambda k: (res[k]["score"] >= top - 3, res[k]["ecartement"], res[k]["score"]))
    print("[concept] écartement : " + ", ".join(f"{k} {res[k]['ecartement']}" for k in res
                                               if res[k]["score"] >= top - 3), flush=True)
    json.dump({"attendus": att, "candidats": res, "retenu": meil}, open(D / "choix.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"[concept] retenu {meil}", flush=True)


if __name__ == "__main__":
    main()

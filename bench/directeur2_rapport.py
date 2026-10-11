#!/usr/bin/env python
"""directeur2_rapport.py — DIRECTEUR, étape A (suite) : RAPPORT DU MODÈLE DE RÉFÉRENCE (2026-09-30).

Principe (user) : le modèle 3D issu de la vidéo de tour FAIT FOI ; le directeur ne corrige rien, il DÉCRIT ce qui
existe pour que le rig et l'animation s'y ADAPTENT. Le rapport fixe le vocabulaire commun à toutes les étapes
suivantes (étude des vidéos d'action, motion capture).
Partage des rôles :
  - GÉOMÉTRIE (plomberie, pas de reconnaissance) : exemplaires de chaque élément (composantes connexes de sa
    région 3D), côté (convention : +x = gauche de la créature, +z = avant), voisinages, membre en CONTACT avec
    chaque objet (le plus proche) et point de contact ;
  - VLM (avec la fiche du monstre en contexte) : rigide ou souple ; il donne aussi le NOMBRE d'exemplaires
    (cadres) quand la geometrie en fusionne (jambes qui se touchent) ;
  - articulation ou bloc d'un seul tenant : MESURE (rapport des epaisseurs), le VLM 4B se contredisant.
Entrées : directeur2/{elements.json, etiquettes_sommets.npy, vue_*.png} (directeur2.py), tex_p3d.glb, fiche.
Sorties : directeur2/rapport.json (+ rapport.txt lisible).
Usage (⚠ .venv-trellis2) : python bench/directeur2_rapport.py <m>
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directeur2 import VLM  # noqa: E402

B = Path(__file__).resolve().parent


def oui(rep):
    return rep.strip().lower().startswith("yes")


def main():
    m = sys.argv[1]
    D = B / "out" / f"v3_{m}" / "directeur2"
    job = json.load(open(B / "jobs" / f"v3_{m}.json", encoding="utf-8"))
    el = json.load(open(D / "elements.json", encoding="utf-8"))
    elements = el["elements"]
    etiq = np.load(D / "etiquettes_sommets.npy")
    mesh = trimesh.load(B / "out" / f"v3_{m}" / "tex_p3d.glb", force="mesh", process=False)
    P, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)

    # sommets dupliques aux coutures d'UV -> meme position = meme point (sinon une coupure d'UV coupe un membre)
    _, grp = np.unique(P.round(5), axis=0, return_inverse=True)
    grp = grp.ravel()
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = grp[e]
    n = grp.max() + 1
    lab_g = np.full(n, -1)
    lab_g[grp] = etiq
    first_idx = np.zeros(n, int)
    first_idx[grp[::-1]] = np.arange(len(grp))[::-1]            # un sommet representant par point
    # centre gauche/droite = celui de la plus grosse partie du CORPS (le tronc en general) : la boite de tout le
    # modele est decentree par un objet tenu d'un cote
    j_corps = max((j for j, x in enumerate(elements) if x["genre"] == "membre"), key=lambda j: (etiq == j).sum())
    centre_x = float(np.median(P[etiq == j_corps][:, 0]))

    # EXEMPLAIRES : composantes connexes de chaque region (aretes internes a la region)
    parts = []
    for j, x in enumerate(elements):
        ok = (lab_g[g[:, 0]] == j) & (lab_g[g[:, 1]] == j)
        A = coo_matrix((np.ones(ok.sum()), (g[ok, 0], g[ok, 1])), shape=(n, n))
        _, comp = connected_components(A, directed=False)
        pts_g = np.nonzero(lab_g == j)[0]
        if not len(pts_g):
            continue
        cs, cnt = np.unique(comp[pts_g], return_counts=True)
        garde = cs[cnt >= 0.08 * cnt.max()]                     # miettes ecartees
        groupes = [pts_g[comp[pts_g] == c] for c in garde]
        # le VLM a vu N exemplaires (max de cadres sur une vue) mais la geometrie en trouve moins (ex. 2 jambes
        # qui se touchent aux hanches) -> la plus grosse composante est scindee (k-moyennes sur les positions)
        nb = [len(c) for c in x["cadres"].values() if c]
        n_vlm = max(set(nb), key=lambda k: (nb.count(k), k)) if nb else 1    # compte le plus frequent (egalite
        # -> le plus grand) : robuste a un cadre errone sur une vue
        while len(groupes) < n_vlm:
            gros = max(range(len(groupes)), key=lambda i: len(groupes[i]))
            q = groupes.pop(gros)
            pos_q = P[first_idx[q]][:, [0, 2]]      # plan HORIZONTAL : des exemplaires se cotoient (gauche/droite,
            from scipy.cluster.vq import kmeans2  # avant/arriere), on ne coupe pas un membre en haut/bas
            _, lab = kmeans2(pos_q, 2, minit="++", seed=0)
            groupes += [q[lab == 0], q[lab == 1]]
        for q_g in groupes:
            sel = np.isin(grp, q_g)
            q = P[sel]
            cx = q[:, 0].mean()
            cote = ("gauche" if cx > centre_x else "droite") if abs(cx - centre_x) > 0.03 else "centre"
            parts.append({"element": x["nom"], "genre": x["genre"], "cote": cote, "sommets": int(sel.sum()),
                          "centre": q.mean(0).round(3).tolist(),
                          "boite": [q.min(0).round(3).tolist(), q.max(0).round(3).tolist()], "_sel": sel})
    for x in elements:
        k = [p for p in parts if p["element"] == x["nom"]]
        for p in k:
            p["id"] = x["nom"] if len(k) == 1 else f"{x['nom']} {p['cote']}"

    # VOISINAGES (aretes entre deux regions) et CONTACT objet -> membre le plus proche
    lp = np.full(len(P), -1)
    for i, p in enumerate(parts):
        lp[p["_sel"]] = i
    lpg = np.full(n, -1)
    lpg[grp] = lp
    a, b = lpg[g[:, 0]], lpg[g[:, 1]]
    fr = (a >= 0) & (b >= 0) & (a != b)
    voisins = {tuple(sorted(v)) for v in np.stack([a[fr], b[fr]], 1).tolist()}
    arbres = {i: cKDTree(P[p["_sel"]]) for i, p in enumerate(parts)}
    for i, p in enumerate(parts):
        if p["genre"] != "objet":
            continue
        q = P[p["_sel"]]
        best = min((arbres[k].query(q)[0].min(), k) for k, o in enumerate(parts) if o["genre"] == "membre")
        d_min, k = best
        dd, ii = arbres[k].query(q)
        p["porteur"] = parts[k]["id"]
        p["contact"] = q[dd.argmin()].round(3).tolist()
        p["distance_contact"] = round(float(d_min), 4)

    # VLM : rigide / souple, et articulation entre parties voisines (avec la fiche en contexte)
    vlm = VLM()
    ims = [Image.open(D / f"vue_{v}.png").convert("RGB") for v in ("face", "gauche", "dos", "droite")]
    ctx = f"Context: this 3D game monster was made from this description: \"{job['concept'][:350]}\" "
    types = {}
    for x in elements:
        r = vlm(ims, ctx + f"Look at its {x['nom']}. Is the {x['nom']} RIGID (it keeps its shape when the creature "
                "moves, like wood, bone, metal, a shell) or SOFT (it bends or flexes along its length, like a "
                "tentacle, a tail, cloth, a snake)? Answer with one word: rigid or soft.", n=5)
        types[x["nom"]] = "souple" if "soft" in r.lower() else "rigide"
    # ARTICULATION = question de FORME, mesuree (le VLM 4B se contredisait selon la formulation) : on compare
    # l'EPAISSEUR (plus petite etendue, percentiles 5-95 %) des deux parties voisines. Presque egales (la tete est
    # le haut de la meme buche) -> BLOC ; la plus fine nettement plus fine (bras, jambe, queue, cou) -> ARTICULATION.
    def epaisseur(q):
        c = q - q.mean(0)
        axes = np.linalg.svd(c[:: max(1, len(c) // 4000)], full_matrices=False)[2]
        pr = c @ axes.T
        return float(min(np.percentile(pr[:, k], 95) - np.percentile(pr[:, k], 5) for k in range(3)))
    articulations = []
    for i, k in sorted(voisins):
        pa, pb = parts[i], parts[k]
        if "objet" in (pa["genre"], pb["genre"]) or pa["element"] == pb["element"]:
            continue                                    # deux exemplaires qui se touchent : pas une liaison
        ea, eb = epaisseur(P[pa["_sel"]]), epaisseur(P[pb["_sel"]])
        rap = min(ea, eb) / max(ea, eb)
        articulations.append({"entre": [pa["id"], pb["id"]], "articule": bool(rap < 0.7),
                              "rapport_epaisseurs": round(rap, 2)})

    rapport = {"monstre": m, "concept": job["concept"], "convention": "+x = gauche de la creature, +z = avant",
               "elements": [], "articulations": articulations}
    for p in parts:
        d = {k: v for k, v in p.items() if not k.startswith("_")}
        d["matiere"] = types[p["element"]]
        rapport["elements"].append(d)
    json.dump(rapport, open(D / "rapport.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    np.save(D / "parties_sommets.npy", lp)                       # partie (index dans rapport.elements) par sommet

    L = [f"RAPPORT DU MODÈLE — {m}"]
    for p in rapport["elements"]:
        s = f"- {p['id']} ({p['genre']}, {p['matiere']}) : {p['sommets']} sommets, centre {p['centre']}"
        if "porteur" in p:
            s += f" ; TENU PAR « {p['porteur']} » (contact {p['contact']}, écart {p['distance_contact']})"
        L.append(s)
    L.append("Liaisons entre parties :")
    L += [f"- {x['entre'][0]} ↔ {x['entre'][1]} : {'ARTICULÉE' if x['articule'] else 'BLOC UNIQUE (pas d’articulation)'}"
          f" (épaisseurs {x['rapport_epaisseurs']})" for x in articulations]
    (D / "rapport.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()

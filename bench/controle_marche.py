#!/usr/bin/env python
"""controle_marche.py — CONTRÔLES AUTOMATIQUES d'une vidéo de MARCHE de profil, avant R-DMesh (2026-10-01).

Sur les images 0..64 (celles que R-DMesh lit) d'une vidéo filmée par une caméra FIXE, créature qui marche vers le bord
GAUCHE. Silhouette = masque birefnet-dis (enregistré dans <video>/masque/ pour l'étape « fond noir »). Mesures :
  - AVANCÉE : déplacement du centre de la silhouette vers la gauche, en longueurs de corps (largeur à l'image 0),
              recul quasi nul, et pas d'arrêt prolongé (chaque quart de la vidéo avance) ;
  - ZOOM    : hauteur de la silhouette (lissée) quasi constante ;
  - CADRE   : la silhouette ne touche jamais les bords ;
  - PÉRIODE : autocorrélation de la bande BASSE de la silhouette (pattes), recalée sur le centre -> motif répété.
              ⚠ en silhouette de profil, patte proche et patte éloignée se confondent : la période vue vaut souvent
              un DEMI-cycle de pas. On rapporte donc les « répétitions du motif », seuil à caler à l'œil.
Seuils PROVISOIRES (à caler sur les premières vidéos, verdict du user). Rien de propre à un type de créature.
Usage (.venv) : python bench/controle_marche.py <dossier_video> [--n 65]
Sortie : <dossier_video>/controle.json (+ controle.png : courbes)"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

SEUILS = {"avancee_min": 0.7, "recul_max": 0.10, "quart_min": 0.25, "zoom_max": 1.15, "bord_px": 2,
          "periode_corr_min": 0.3, "repetitions_min": 3.0}


def masques(V, n):
    D = V / "masque"
    D.mkdir(exist_ok=True)
    ses = None
    out = []
    for i in range(n):
        p = D / f"frame_{i:04d}.png"
        if not p.exists():
            if ses is None:
                from rembg import new_session
                ses = new_session("birefnet-dis")
            from rembg import remove
            im = Image.open(V / f"frame_{i:04d}.png").convert("RGB")
            Image.fromarray(((np.asarray(remove(im, session=ses, only_mask=True)) > 127) * 255).astype(np.uint8)).save(p)
        out.append(np.asarray(Image.open(p)) > 127)
    return np.stack(out)


def autocorr_bande(M, cx, bas, h0, L0):
    """séquence de la bande basse (40 % de la hauteur) recalée sur le centre, ramenée en 32x64, centrée dans le temps"""
    T, H, W = M.shape
    seq = []
    for t in range(T):
        y1 = int(bas[t]) + 1
        y0 = max(0, int(y1 - 0.4 * h0))
        x0 = int(cx[t] - 0.75 * L0)
        x1 = int(cx[t] + 0.75 * L0)
        pad = np.zeros((y1 - y0, x1 - x0), bool)
        a, b = max(0, x0), min(W, x1)
        pad[:, a - x0:b - x0] = M[t, y0:y1, a:b]
        seq.append(np.asarray(Image.fromarray(pad.astype(np.uint8) * 255).resize((64, 32), Image.BILINEAR), float))
    S = np.stack(seq).reshape(T, -1)
    S = S - S.mean(0)
    c = [1.0]
    for k in range(1, T // 2 + 1):
        a, b = S[:-k], S[k:]
        c.append(float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)))
    return np.array(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=65)
    ap.add_argument("--sur-place", action="store_true",
                    help="marche SUR PLACE (tapis roulant) : la créature doit RESTER au même endroit (avancée < 0,3 longueur)")
    a = ap.parse_args()
    V = Path(a.video)
    M = masques(V, a.n)
    T, H, W = M.shape
    vide = [t for t in range(T) if not M[t].any()]
    if vide:
        r = {"ok": False, "raison": f"silhouette absente sur {len(vide)} images"}
        json.dump(r, open(V / "controle.json", "w"), indent=1)
        print(json.dumps(r))
        return
    xs = [np.nonzero(M[t].any(0))[0] for t in range(T)]
    ys = [np.nonzero(M[t].any(1))[0] for t in range(T)]
    gauche = np.array([x.min() for x in xs]); droite = np.array([x.max() for x in xs])
    haut = np.array([y.min() for y in ys]); bas = np.array([y.max() for y in ys])
    cx = np.array([np.nonzero(M[t])[1].mean() for t in range(T)])
    h = median_filter((bas - haut + 1).astype(float), 7, mode="nearest")
    L0, h0 = float(droite[0] - gauche[0] + 1), float(h[0])

    cxs = median_filter(cx, 5, mode="nearest")
    avancee = float((cxs[0] - cxs[-1]) / L0)                       # > 0 = vers la gauche
    dx = -np.diff(cxs)                                              # > 0 = avance
    recul = float((dx < -0.5).mean())
    q = np.array_split(np.arange(T), 4)
    quarts = [float((cxs[i[0]] - cxs[i[-1]]) / max(cxs[0] - cxs[-1], 1e-6)) for i in q]
    # échelle = racine de la SURFACE de la silhouette (et non sa hauteur : une tête qui se lève ou une queue qui balaie
    # change la boîte englobante sans zoom ; un vrai zoom ×1,2 change la surface ×1,44). Faux échecs vus le 2026-10-07.
    ech = median_filter(np.sqrt(M.reshape(T, -1).sum(1).astype(float)), 7, mode="nearest")
    zoom_s = float(ech.max() / ech.min())
    zoom_h = float(h.max() / h.min())
    # un vrai zoom gonfle les DEUX ; tête levée -> surtout la hauteur ; pattes écartées -> surtout la surface
    zoom = min(zoom_s, zoom_h)
    bord = int(((gauche <= SEUILS["bord_px"]) | (droite >= W - 1 - SEUILS["bord_px"])
                | (haut <= SEUILS["bord_px"]) | (bas >= H - 1 - SEUILS["bord_px"])).sum())
    ac = autocorr_bande(M, cx, bas, h0, L0)
    # première crête locale après le premier passage sous zéro (sinon le lissage temporel fait une fausse crête à k=1)
    k0 = next((k for k in range(1, len(ac)) if ac[k] < 0), None)
    crete = None
    if k0 is not None:
        for k in range(k0 + 1, len(ac) - 1):
            if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]:
                crete = k
                break
    periode = crete if crete is not None and ac[crete] >= SEUILS["periode_corr_min"] else None
    reps = float(T / periode) if periode else 0.0

    tests = {
        "avancee": abs(avancee) <= 0.3 if a.sur_place else avancee >= SEUILS["avancee_min"],
        "sans_recul": True if a.sur_place else recul <= SEUILS["recul_max"],
        # chaque quart fait >= 25 % de sa part (1/4) ; sans objet sur place
        "reguliere": True if a.sur_place else min(quarts) >= SEUILS["quart_min"] * 0.25,
        "sans_zoom": zoom <= SEUILS["zoom_max"],
        "dans_cadre": bord == 0,
        "periodique": reps >= SEUILS["repetitions_min"],
    }
    # la mesure de périodicité rate des pas bien visibles (loup, 2026-10-02) : rapportée, mais hors décision
    r = {"ok": all(v for k, v in tests.items() if k != "periodique"), "tests": tests,
         "mesures": {"avancee_longueurs": round(avancee, 2), "recul_frac": round(recul, 3),
                     "avancee_par_quart": [round(x, 2) for x in quarts], "zoom": round(zoom, 3), "zoom_surface": round(zoom_s, 3), "zoom_hauteur": round(zoom_h, 3),
                     "images_au_bord": bord, "periode_images": periode,
                     "corr_crete": round(float(ac[crete]), 3) if crete else None, "repetitions": round(reps, 2),
                     "corps_px": [round(L0), round(h0)]},
         "seuils": SEUILS}
    json.dump(r, open(V / "controle.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 3, figsize=(13, 3.2))
        ax[0].plot((cx[0] - cx) / L0); ax[0].set_title(f"avancée (longueurs) : {avancee:.2f}")
        ax[0].set_xlabel("image")
        ax[1].plot(h / h0); ax[1].set_title(f"hauteur relative (zoom) : max/min {zoom:.2f}")
        ax[1].set_xlabel("image")
        ax[2].plot(ac); ax[2].axhline(0, c="k", lw=.5)
        ax[2].set_title(f"autocorr. pattes : période {periode} img, {reps:.1f} répét.")
        ax[2].set_xlabel("décalage (images)")
        f.suptitle(("OK" if r["ok"] else "ÉCHEC ") + " " + ", ".join(k for k, v in tests.items() if not v))
        f.tight_layout(); f.savefig(V / "controle.png", dpi=80); plt.close(f)
    except ImportError:
        pass
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

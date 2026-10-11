#!/usr/bin/env python
"""boucle_marche.py — BOUCLE DE MARCHE pour le jeu (2026-10-11) : à partir des trajectoires de sommets (R-DMesh ou
Motion 3-to-4), retire l'avancée et découpe un cycle de pas qui se répète sans saut. Générique (aucune hypothèse sur
le nombre de pattes) :
  1. AVANCÉE : la dérive HORIZONTALE du centre du corps est ajustée par une droite et retirée (le balancement et le
     rebond restent) ; sa pente = vitesse de marche, rendue au jeu (« root motion ») ;
  2. PÉRIODE : distance moyenne entre les poses (centrées) à t et t + L, pour chaque décalage L ; on prend le creux
     le plus bas au-delà de 0,75 s (un cycle complet : chaque membre revient à sa position) ;
  3. DÉCOUPE : le début i qui ressemble le plus à i + L ; l'écart résiduel est réparti linéairement sur le cycle,
     donc la dernière image enchaîne exactement sur la première.
Usage (numpy seul, n'importe quel venv) : python bench/boucle_marche.py <trajs.npz> <sortie.npz> [img/s = 32]
Sortie : <sortie>.npz (obj0 = cycle, L images) + <sortie>.json (période, début, écart, vitesse en unités/s)"""
import json
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 32.0
X = np.load(src)["obj0"].astype(np.float64)                    # (T, N, 3), Z vers le haut (Blender)
T = len(X)
taille = float(np.ptp(X[0], 0).max())

# 1. avancée horizontale retirée (droite ajustée sur le centre du corps)
c = X.mean(1)
t = np.arange(T, dtype=np.float64)
pente = np.polyfit(t, c[:, :2], 1)                             # (2, 2) : pente et ordonnée pour x, y
derive = np.outer(t - t[0], pente[0])                          # dérive depuis l'image 0
Y = X.copy()
Y[:, :, :2] -= derive[:, None, :]
vitesse = float(np.linalg.norm(pente[0]) * FPS)

# 2. période : pose centrée (le centre horizontal de chaque image retiré pour comparer les formes)
P = Y - np.concatenate([Y.mean(1)[:, None, :2], np.zeros((T, 1, 1))], 2)
sous = P[:, ::max(1, P.shape[1] // 2000)]                      # 2 000 sommets suffisent pour comparer
D = np.sqrt(((sous[:, None] - sous[None]) ** 2).sum(-1)).mean(-1) / taille      # (T, T)
# cycle d'au moins 0,75 s (une marche de jeu fait 0,8 à 2 s) ; creux le plus bas, ou le premier à 10 % près de lui
# ⚠ R-DMesh ne lit que 65 images (2 s à 32 img/s) : souvent un seul cycle, la ressemblance fin/début reste imparfaite
# et l'écart est réparti sur le cycle (voir « ecart_avant_correction » dans le .json)
lmin, lmax = int(round(0.75 * FPS)), T - 4
prof = np.array([np.diag(D, L).mean() for L in range(lmin, lmax + 1)])
k = next(k for k in range(len(prof)) if prof[k] <= 1.10 * prof.min()
         and (k == len(prof) - 1 or prof[k] <= prof[k + 1]))
L = lmin + k

# 3. découpe : début le plus ressemblant à sa fin, écart réparti sur le cycle
i = int(np.argmin([D[j, j + L] for j in range(T - L)]))
cyc = Y[i:i + L].copy()
ecart = Y[i + L] - Y[i]                                        # ce que la dernière image devrait rattraper
cyc -= ecart[None] * (np.arange(L)[:, None, None] / L)
np.savez(dst, obj0=cyc.astype(np.float32))
info = {"images_source": T, "periode": L, "debut": i, "duree_s": round(L / FPS, 3),
        "ecart_avant_correction": round(float(D[i, i + L]), 4),
        "vitesse_unites_par_s": round(vitesse, 4), "direction_xy": (pente[0] / (np.linalg.norm(pente[0]) + 1e-9)).round(3).tolist()}
json.dump(info, open(dst.rsplit(".", 1)[0] + ".json", "w"), indent=1)
print(f"[boucle] cycle de {L} images ({L / FPS:.2f} s) à partir de l'image {i}, écart {D[i, i + L]:.4f} "
      f"(en tailles), vitesse {vitesse:.3f} u/s -> {dst}", flush=True)

#!/usr/bin/env python
"""choisir_vues.py — choisit dans la VIDÉO DE TOUR les images face / gauche / dos / droite (2026-09-30).

Le tour de caméra de Wan n'est pas régulier (Tung Tung : profil à 39, dos à 51, autre profil à 65 sur 81) : on
ne peut pas prendre des images à intervalles fixes. Le VLM regarde chaque image, AVEC l'image 0 (face) en
référence, et dit si on la voit de face, de profil ou de dos ; pour chacune on retient le MILIEU de la plus longue
suite d'images qui la montrent. Le côté des profils (le VLM confond gauche/droite) vient de la structure du tour
(face -> profil 1 -> dos -> profil 2) et du flux optique mesuré entre la face et le profil 1. Convention des vues (Pixal3D) : « gauche » = créature tournée vers le BORD
GAUCHE de l'image.
Sortie : inputs/q21_vues_<id>/{gauche_7, droite_7, dos_precis_7}.png (carré 1024, fond complété) + choix.json.
Usage (⚠ .venv-trellis2) : python bench/choisir_vues.py <dossier_video_tour> <id ex. v3_godzillav> [--essai]
  VUES_PAS=1 : classer toutes les images (2 par défaut) ; --essai : affiche le choix sans rien écrire
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directeur2 import VLM  # noqa: E402

B = Path(__file__).resolve().parent
Q = ("The first image shows the creature from the FRONT (it faces the camera). In the second image the camera has "
     "turned around it. In the second image, which way does the creature face? Answer with one word: 'camera' "
     "(we see its front), 'left' (it faces the left edge of the image, we see its profile), 'right' (it faces the "
     "right edge of the image, we see its profile), or 'away' (we see its back).")


def carre(im, taille=1024):
    c = max(im.size)
    f = Image.new("RGB", (c, c), im.getpixel((3, im.height // 2)))
    f.paste(im, ((c - im.width) // 2, (c - im.height) // 2))
    return f.resize((taille, taille), Image.LANCZOS)


def main():
    src, ident = Path(sys.argv[1]), sys.argv[2]
    fr = sorted(src.glob("frame_*.png"))
    ims = [Image.open(f).convert("RGB") for f in fr]
    vlm = VLM()
    # UNE IMAGE SUR PAS (2 par défaut, 2026-10-11) : le choix ne dépend que des LIMITES des suites face / profil / dos
    # (interpolation des profils), une image de plus ou de moins aux limites ne change presque rien ; 2x moins
    # d'appels au VLM. Les images sautées prennent l'étiquette de l'image classée précédente.
    pas = int(os.environ.get("VUES_PAS", "2"))
    vus = sorted(set(range(0, len(ims), pas)) | {len(ims) - 1})
    et = {}
    def classe(i):
        r = vlm([ims[0], ims[i]], Q, n=4).lower()
        et[i] = next((k for k in ("camera", "left", "right", "away") if k in r), "?")
    for i in vus:
        classe(i)
    # AFFINAGE : une image sautée entre deux images classées DIFFÉREMMENT est classée aussi (les limites des suites
    # redeviennent exactes ; Slender sans affinage : profil à 35 au lieu de 32)
    for a, b in zip(vus, vus[1:]):
        if et[a] != et[b]:
            for i in range(a + 1, b):
                classe(i)
    lab = [et[max(j for j in et if j <= i)] for i in range(len(ims))]
    print("[vues] " + " ".join(f"{i}:{l[0]}" for i, l in enumerate(lab)), flush=True)
    # le VLM confond gauche et droite (il dit « left » pour les deux profils) : on ne lui demande que profil / dos /
    # face, et la STRUCTURE du tour fait le reste — face -> profil 1 -> dos -> profil 2 -> face : les deux profils
    # sont opposés ; le côté du profil 1 se MESURE au flux optique (la surface avant glisse vers le côté où la
    # créature se tourne).
    lab = ["profil" if l in ("left", "right") else l for l in lab]

    def runs(cle, lo=0, hi=None):
        hi = len(lab) if hi is None else hi
        out, a = [], None
        for i in range(lo, hi + 1):
            l = lab[i] if i < hi else "?"
            if l == cle and a is None:
                a = i
            elif l != cle and a is not None:
                out.append((i - a, a))
                a = None
        return out
    rd = runs("away")
    if not rd:
        sys.exit("ECHEC : aucune image de dos dans la video de tour")
    nd, ad = max(rd)
    r1, r2 = runs("profil", 0, ad), runs("profil", ad + nd)
    if not r1 or not r2:
        sys.exit("ECHEC : il faut un profil avant ET apres le dos")
    (n1, a1), (n2, a2) = max(r1), max(r2)
    import cv2
    import numpy as np
    dx = []
    for i in range(0, a1 + n1 // 2):
        g0 = cv2.cvtColor(np.asarray(ims[i]), cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(np.asarray(ims[i + 1]), cv2.COLOR_RGB2GRAY)
        fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fond = np.median(g0[:, :20])
        m = np.abs(g0.astype(int) - fond) > 25                          # pixels de la creature
        if m.sum() > 100:
            dx.append(float(np.median(fl[..., 0][m])))
    sens = float(np.sum(dx))
    p1, p2 = ("gauche", "droite") if sens < 0 else ("droite", "gauche")
    print(f"[vues] flux horizontal cumule face -> profil 1 : {sens:+.1f} px -> profil 1 = {p1}", flush=True)
    # (le TOURNOI de profils par duels VLM, gardé « pour contrôle » jusqu'au 2026-10-11, est supprimé : ~100 appels au
    # VLM dont le résultat était jeté ; ses profils tombaient souvent sur un trois-quarts arrière)
    # DOS = centre de TOUTES les images vues de dos autour de la plus longue suite (créature presque symétrique,
    # araignée : le VLM ne voit le dos que par fragments 40-41, 46-47 -> la « plus longue suite » tombait au bord)
    dos_im = [i for i, l in enumerate(lab) if l == "away" and abs(i - (ad + nd // 2)) <= 12]
    dos = int(round(float(np.median(dos_im)))) if dos_im else ad + nd // 2
    # profils INCOHÉRENTS (pas nettement de part et d'autre du dos) -> mi-chemin face/dos et dos/retour de face
    fin_face = next((i for i, l in enumerate(lab) if l != "camera"), 1)
    retour = next((i for i in range(len(lab) - 1, dos, -1) if lab[i] != "camera"), len(lab) - 1) + 1
    # PROFILS = INTERPOLATION entre face, dos et retour de face (méthode par défaut depuis 2026-10-01) : le tournoi du
    # VLM choisit souvent un trois-quarts arrière (loup : 37/53 au lieu de ~25/~60 ; araignée : 37/48) ; l'interpolation
    # donnait 25/63 (loup), 24/60 (araignée), 34/65 (Tung Tung, vrai 39/65).
    t1, t2 = (fin_face + dos) // 2, (dos + min(retour, len(lab) - 1)) // 2
    choix = {"face": 0, "dos": dos, p1: t1, p2: t2}
    print(f"[vues] choix : {choix}", flush=True)
    if "--essai" in sys.argv:                 # vérification : n'écrit rien
        return
    dst = B / "inputs" / f"q21_vues_{ident}"
    dst.mkdir(parents=True, exist_ok=True)
    for v, nom in (("gauche", "gauche_7"), ("droite", "droite_7"), ("dos", "dos_precis_7")):
        carre(ims[choix[v]]).save(dst / f"{nom}.png")
    json.dump({"video": str(src), "choix": choix, "etiquettes": lab}, open(dst / "choix.json", "w"), indent=1)


if __name__ == "__main__":
    main()

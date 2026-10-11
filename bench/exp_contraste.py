#!/usr/bin/env python
"""CONTRASTE FAVORABLE A LA 3D — recherche du meilleur « rendu argile » d'entree.

MECANISME MESURE (2026-09-28) : TRELLIS.2 premultiplie l'objet sur FOND NOIR.
Slenderman colore : 67,6 % de l'objet quasi noir (lum < 40) -> invisible pour le
modele. Argile : 2,6 %. Et les zones BRULEES (lum > 235 : yeux lumineux de la
gargouille, os blanc sature) n'ont plus d'ombrage -> fausse geometrie.
CIBLE : tout l'objet dans [40, 235], mediane ~130, avec de l'ecart pour le relief.

  A  argile SDXL + ControlNet, prompt anti-lueur, PUIS gris uniforme recale
     [55, 215] + ecretage des halos           (python principal)
  B  argile SYNTHETIQUE : normales MoGe du concept colore + eclairage 3 points
     choisi + albedo uniforme -> aucun halo possible  (.venv-trellis2)
Meme masque que cmp_<m>_couleur pour tous (seule variable = l'image d'entree).

Usage : python bench/exp_contraste.py A | B | mesure <dossiers...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "bench" / "inputs"
MONSTRES = ["slenderman", "araignee", "os", "spectre", "gargouille"]
CIBLE = (55, 215)


def masque(k, source="ancien"):
    """ancien = masques de cmp_<k>_couleur (u2net + finalize_cut).
    dis   = birefnet-dis (bench/exp_masques.py) : le seul bon sur les 5 monstres
            (tentacules completes, pas de fond entre les pattes, baton du spectre).
            PAS de rattrapage de luminance ni de bouchage de poches : c'est le
            rattrapage de finalize_cut qui avait mis le « triangle lumineux »
            derriere les jambes de la gargouille (u2net seul decoupait bien)."""
    if source == "dis":
        a = np.asarray(Image.open(INP / "masques" / f"{k}_birefnet-dis.png").convert("L")) > 127
        lab, n = ndimage.label(a)
        if n:
            t = ndimage.sum(a, lab, range(1, n + 1))
            a = np.isin(lab, 1 + np.where(t >= 0.002 * a.sum())[0])   # retire les poussieres
        return a
    return np.asarray(Image.open(INP / f"cmp_{k}_couleur" / "02_cut.png").convert("RGBA"))[..., 3] > 0


def couleur(k):
    return np.asarray(Image.open(INP / f"cmp_{k}_couleur" / "02_cut.png").convert("RGB"))


def recale_gris(lum, m, doux=False):
    """Gris uniforme recale dans la CIBLE.
    doux=False (A) : ecretage net a p97 puis etirement. ⚠ DEFAUT MESURE : sur le
      Slenderman, 22 % de la tete bloques a la valeur max (relief 89 -> 50) ->
      tete CREUSE en 3D (remarque du user). Un ecretage fabrique une zone brulee.
    doux=True (A2) : genou tanh au-dessus de 85 % : les hautes lumieres sont
      comprimees mais gardent leur degrade (strictement croissant, pas de palier)."""
    if not doux:
        v = lum[m]
        hi = np.percentile(v, 97)
        lum = np.minimum(lum, hi)
        lo, hi = np.percentile(lum[m], 2), np.percentile(lum[m], 98)
        out = CIBLE[0] + (lum - lo) * (CIBLE[1] - CIBLE[0]) / max(hi - lo, 1e-6)
        return np.clip(out, CIBLE[0] - 15, CIBLE[1] + 15)
    lo, hi = np.percentile(lum[m], 1), np.percentile(lum[m], 99.5)
    x = (lum - lo) / max(hi - lo, 1e-6)
    k = 0.85
    x = np.where(x > k, k + (1 - k) * np.tanh((x - k) / (1 - k)), x)
    x = np.clip(x, 0, 1)
    return CIBLE[0] + x * (CIBLE[1] - CIBLE[0])


def ecrit(k, var, lum, m):
    g = np.where(m, lum, 0).astype(np.uint8)
    d = INP / f"cmp_{k}_{var}"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack([g, g, g, np.where(m, 255, 0)]).astype(np.uint8)).save(d / "02_cut.png")


def systeme_A():
    import torch
    sys.path.insert(0, str(ROOT / "bench"))
    from exp_argile import contours, MONSTRES as DESCS
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                         torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True, cache_dir=str(ROOT / "models"))
    pipe.enable_model_cpu_offload()
    neg = ("glowing eyes, glow, emissive, luminous, light rays, halo, bloom, lens flare, neon, "
           "fire, specular highlights, shiny, glossy, reflections, color, colorful, dark clay, "
           "black, shadow, floor, ground, pedestal, rock, text, watermark, multiple creatures")
    for k in MONSTRES:
        rgb = couleur(k)
        prompt = (f"grey clay sculpture, {DESCS[k][1]}, uniform matte light grey clay, no glowing "
                  "parts, no emissive light, soft three-point studio lighting, subtle rim light, "
                  "isolated on pure black background")
        img = pipe(prompt=prompt, negative_prompt=neg, image=Image.fromarray(contours(rgb)),
                   controlnet_conditioning_scale=0.9, num_inference_steps=30, guidance_scale=7.0,
                   width=1024, height=1024, generator=torch.Generator("cpu").manual_seed(7)).images[0]
        a = np.asarray(img).astype(float)
        brut = INP / f"cmp_{k}_A" / "argile_brut.png"
        brut.parent.mkdir(parents=True, exist_ok=True)
        img.save(brut)
        m = masque(k)
        ecrit(k, "A", recale_gris(a.mean(2), m), m)
        m2 = masque(k, "dis")
        ecrit(k, "A2", recale_gris(a.mean(2), m2, doux=True), m2)
        print(f"A/A2 {k} ok", flush=True)


def systeme_B():
    import types
    import torch
    node = ROOT / "third_party" / "ComfyUI" / "custom_nodes" / "ComfyUI-Trellis2"
    sys.path.insert(0, str(ROOT / "third_party" / "ComfyUI")); sys.path.insert(0, str(node))
    pkg = types.ModuleType("t2node"); pkg.__path__ = [str(node)]; sys.modules["t2node"] = pkg
    from t2node.moge.model.v2 import MoGeModel
    mp = ROOT / "third_party" / "ComfyUI" / "models" / "Ruicheng" / "moge-2-vitl" / "model.pt"
    model = MoGeModel.from_pretrained(str(mp)).cuda().eval()
    for k in MONSTRES:
        rgb = couleur(k)
        m = masque(k)
        t = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).cuda()
        with torch.no_grad():
            out = model.infer(t, apply_mask=False)
        if "normal" in out:
            n = out["normal"].float().cpu().numpy()
            src = "tete normales"
        else:                                         # repli : normales depuis la carte de points
            P = out["points"].float().cpu().numpy()
            dx = np.gradient(P, axis=1); dy = np.gradient(P, axis=0)
            n = np.cross(dx, dy)
            src = "derivees des points"
        n = n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-8)
        # orientation : la surface doit FAIRE FACE a la camera (convention auto-detectee)
        s = -1.0 if np.nanmean(n[m][:, 2]) > 0 else 1.0
        nx, ny, nz = n[..., 0], n[..., 1], s * n[..., 2]          # nz < 0 = face camera
        face = np.clip(-nz, 0, 1)
        def lam(L):
            L = np.asarray(L, float); L /= np.linalg.norm(L)
            return np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        key = lam([-0.5, -0.6, -1.0])        # haut-gauche, cote camera (x droite, y bas)
        fill = lam([0.7, 0.2, -1.0])         # contre-jour doux a droite
        rim = (1 - face) ** 2                # liseré sur les contours (silhouette lisible)
        sh = 0.30 + 0.55 * key + 0.20 * fill + 0.35 * rim
        ecrit(k, "B", recale_gris(sh * 255, m), m)
        print(f"B {k} ok ({src})", flush=True)


def mesure(dossiers):
    print(f"{'entree':<26}{'mediane':>8}{'% < 40':>9}{'% > 235':>9}{'ecart p10-p90':>15}")
    for d in dossiers:
        p = Path(d) / "02_cut.png"
        im = np.asarray(Image.open(p).convert("RGBA")).astype(float)
        m = im[..., 3] > 0
        L = im[..., :3].mean(2)[m]
        print(f"{Path(d).name:<26}{np.median(L):>8.0f}{100*(L<40).mean():>8.1f}%{100*(L>235).mean():>8.2f}%"
              f"{np.percentile(L,90)-np.percentile(L,10):>15.0f}")


if __name__ == "__main__":
    {"A": systeme_A, "B": systeme_B}.get(sys.argv[1], lambda: mesure(sys.argv[2:]))()

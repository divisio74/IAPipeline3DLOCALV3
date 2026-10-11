#!/usr/bin/env python
"""IDEE DU USER (2026-09-28) : separer FORME et TEXTURE. La 3D recoit une image
concue POUR LA GEOMETRIE : la meme creature en ARGILE GRISE sur FOND NOIR.
  - detourage trivial par construction (ecart au noir), plus de rembg ;
  - le relief se lit (une tentacule noire n'a presque pas d'ombrage visible).
La texture viendra ensuite du concept colore (etage separe).

MEME CREATURE garantie par un ControlNet Canny (SDXL) : il recopie les CONTOURS
exacts du concept colore ; SDXL ne fait que changer la matiere. (Kontext serait
l'alternative, mais son depot officiel est sous licence a accepter : HTTP 403.)

Usage : python bench/exp_argile.py
Sortie : bench/inputs/argile_<id>/02_cut.png + bench/out/_argile_planche.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "bench" / "inputs"

MONSTRES = {
    "slenderman": (INP / "gen_slenderman" / "02_cut.png",
                   "Slender Man, extremely tall thin faceless humanoid in a suit, several long tentacles from its back"),
}
for k in ("araignee", "os", "spectre", "gargouille"):
    # ancien banc d'essai (supprimé au nettoyage du 2026-10-01) : ignoré s'il n'existe plus, sinon l'IMPORT de ce
    # module (make_monster_v2 -> contours) plantait et cassait toute la chaîne 3D
    if not (INP / f"banc_{k}" / "source.txt").exists():
        continue
    src = (INP / f"banc_{k}" / "source.txt").read_text(encoding="utf-8").strip()
    desc = (INP / f"banc_{k}" / "views_prompt.txt").read_text(encoding="utf-8").split(", 3d model")[0]
    MONSTRES[k] = (Path(src), desc)

SEED = 7
CN_SCALE = 0.9


def contours(rgb):
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    e = cv2.Canny(g, 60, 160)
    return np.stack([e] * 3, -1)


def detourage_noir(rgb):
    """Fond noir -> objet = ecart au fond > 40 ; on garde le corps principal et
    ce qui le touche ; on ne bouche que les trous qui ne sont pas du fond."""
    f = rgb.astype(float)
    bg = np.median(np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]]), 0)
    ecart = np.abs(f - bg).max(2)
    m = ndimage.binary_opening(ecart > 40, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m, bg
    t = ndimage.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.where(t >= 0.002 * m.sum())[0])
    trous = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(trous)
    if hn:
        moy = np.asarray(ndimage.mean(ecart, hl, range(1, hn + 1)))
        m |= np.isin(hl, 1 + np.where(moy > 40)[0])
    return m, bg


def main():
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                         torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        cache_dir=str(ROOT / "models"))
    pipe.enable_model_cpu_offload()
    neg = ("color, colorful, paint, skin texture, fabric, background, floor, ground, "
           "shadow, pedestal, base, rock, text, watermark, blurry, multiple creatures")
    W = 256
    planche = Image.new("RGB", (4 * W, len(MONSTRES) * W), (20, 20, 20))
    for i, (k, (src, desc)) in enumerate(MONSTRES.items()):
        rgb = np.asarray(Image.open(src).convert("RGB").resize((1024, 1024)))
        edge = contours(rgb)
        prompt = (f"grey clay sculpture, {desc}, untextured matte light grey clay, "
                  "isolated on pure black background, soft even studio lighting")
        img = pipe(prompt=prompt, negative_prompt=neg, image=Image.fromarray(edge),
                   controlnet_conditioning_scale=CN_SCALE, num_inference_steps=30,
                   guidance_scale=7.0, width=1024, height=1024,
                   generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
        a = np.asarray(img)
        m, bg = detourage_noir(a)
        d = INP / f"argile_{k}"
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / "argile_brut.png")
        Image.fromarray(np.dstack([a, np.where(m, 255, 0)]).astype(np.uint8)).save(d / "02_cut.png")
        for c, im in enumerate([Image.fromarray(rgb), Image.fromarray(edge), img,
                                Image.fromarray((m * 255).astype(np.uint8))]):
            planche.paste(im.convert("RGB").resize((W, W)), (c * W, i * W))
        print(f"ARGILE {k:<11} fond {bg.astype(int)} | objet {m.sum():>7} px", flush=True)
    planche.save(ROOT / "bench" / "out" / "_argile_planche.png")
    print("planche : bench/out/_argile_planche.png (concept | contours | argile | detourage)")


if __name__ == "__main__":
    main()

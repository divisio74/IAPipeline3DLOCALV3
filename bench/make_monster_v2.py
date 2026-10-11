#!/usr/bin/env python
"""make_monster_v2.py — fonctions de l'étape ARGILE A2 (validée les 2026-09-28/29), utilisées par exp_vues_3d.py :
  - masque_dis(rgb) : détourage birefnet-dis (rembg) ;
  - argile_A2(rgb, desc, masque) : ControlNet Canny recopie les contours de la vue, SDXL rend la MÊME créature en
    argile grise (prompt anti-lueur), gris recalé [55, 215] par compression douce (le relief se lit ; plus rien de
    noir ni de brûlé).
L'ancienne chaîne autonome (concepts SDXL, géométrie et texture TRELLIS.2) a été retirée le 2026-10-11 : la
pipeline actuelle est bench/fabrique_monstre.sh (historique git pour l'ancienne).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BENCH))

_SESSION_DIS = None


def masque_dis(rgb_img):
    # session chargée UNE fois (avant : rechargée à chaque appel). Volontairement sur CPU : ici le détourage alterne
    # avec SDXL (argiles A2) ; sur CUDA, la session ONNX garde ~2-3 Go de VRAM -> SDXL déborde en mémoire partagée et
    # l'étape passe de ~2 min à plus de 30 (régression constatée le 2026-10-06 sur Siren Head)
    global _SESSION_DIS
    from rembg import new_session, remove
    if _SESSION_DIS is None:
        _SESSION_DIS = new_session("birefnet-dis", providers=["CPUExecutionProvider"])
    a = np.asarray(remove(rgb_img, session=_SESSION_DIS, only_mask=True)) > 127
    lab, n = ndimage.label(a)
    if n:
        t = ndimage.sum(a, lab, range(1, n + 1))
        a = np.isin(lab, 1 + np.where(t >= 0.002 * a.sum())[0])
    return a


def argile_A2(rgb, desc, m):
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    from exp_argile import contours
    from exp_contraste import recale_gris
    cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                         torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True, cache_dir=str(ROOT / "models"))
    pipe.enable_model_cpu_offload()
    neg = ("glowing eyes, glow, emissive, luminous, light rays, halo, bloom, lens flare, neon, "
           "fire, specular highlights, shiny, glossy, reflections, color, colorful, dark clay, "
           "black, shadow, floor, ground, pedestal, rock, text, watermark, multiple creatures")
    prompt = (f"grey clay sculpture, {desc}, uniform matte light grey clay, no glowing parts, "
              "no emissive light, soft three-point studio lighting, subtle rim light, "
              "isolated on pure black background")
    img = pipe(prompt=prompt, negative_prompt=neg, image=Image.fromarray(contours(rgb)),
               controlnet_conditioning_scale=0.9, num_inference_steps=30, guidance_scale=7.0,
               width=1024, height=1024, generator=torch.Generator("cpu").manual_seed(7)).images[0]
    del pipe; torch.cuda.empty_cache()
    lum = np.asarray(img).astype(float).mean(2)
    g = np.where(m, recale_gris(lum, m, doux=True), 0).astype(np.uint8)
    return img, Image.fromarray(np.dstack([g, g, g, np.where(m, 255, 0)]).astype(np.uint8))

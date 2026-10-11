#!/usr/bin/env python
"""masque_gpu.py — masques de détourage BiRefNet (poids DIS5K) sur GPU, en PyTorch demi-précision (2026-10-06).

Remplace rembg « birefnet-dis » (ONNX) pour les vidéos : mêmes poids (BiRefNet DIS5K), masques identiques à 99,5 %
(IoU), mais 0,15 s/image au lieu de ~10 s sur CPU (onnxruntime) ou 2,3 s avec onnxruntime-gpu (les convolutions
déformables de BiRefNet n'ont pas de noyau CUDA dans ONNX et retombent sur le processeur).
Écrit <dossier>/masque/frame_XXXX.png (0/255, seuil 127 — même format que controle_marche.py, qui saute alors le calcul).
Usage (⚠ .venv-trellis2) : python bench/masque_gpu.py <dossier contenant frame_XXXX.png> [--n 65]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForImageSegmentation

MODELE = "ZhengPeng7/BiRefNet-DIS5K"


class Detoureur:
    def __init__(self):
        self.m = AutoModelForImageSegmentation.from_pretrained(MODELE, trust_remote_code=True).to("cuda").eval().half()
        self.tf = T.Compose([T.Resize((1024, 1024)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def __call__(self, im):
        p = self.m(self.tf(im.convert("RGB")).unsqueeze(0).cuda().half())[-1].sigmoid()[0, 0].float().cpu().numpy()
        return np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dossier")
    ap.add_argument("--n", type=int, default=None, help="nombre d'images (défaut : toutes)")
    a = ap.parse_args()
    V = Path(a.dossier)
    D = V / "masque"
    D.mkdir(exist_ok=True)
    fs = sorted(V.glob("frame_*.png"))[:a.n]
    det = Detoureur()
    for f in fs:
        if not (D / f.name).exists():
            Image.fromarray(((det(Image.open(f)) > 127) * 255).astype(np.uint8)).save(D / f.name)
    print(f"[masque-gpu] {len(fs)} masques -> {D}", flush=True)


if __name__ == "__main__":
    main()

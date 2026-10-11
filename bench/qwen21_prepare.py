#!/usr/bin/env python
"""qwen21_prepare.py — telecharge Qwen-Image-2.1 dans ~17 Go de disque au lieu de ~33.

- transformeur : GGUF Q8_0 d'unsloth (7,6 Go, quasi sans perte), lu directement par diffusers ;
- encodeur de prompt (Qwen3-VL 8B, 17,5 Go en bf16) : chaque morceau est telecharge, converti en
  int8 par ligne (poids seuls, echelle par canal de sortie) puis SUPPRIME avant le suivant. On ne garde
  que le modele de langage : la tour vision et lm_head ne servent pas en texte -> image ;
- VAE, configs, processeur : tels quels.
A lancer avec : PYTHONPATH=.pylib-qwen21 HF_HUB_DISABLE_XET=1 .venv/Scripts/python.exe bench/qwen21_prepare.py
"""
import json
import os
import pathlib

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO, GGUF_REPO = "Qwen/Qwen-Image-2.1", "unsloth/Qwen-Image-2.1-GGUF"
DST = pathlib.Path(__file__).resolve().parent.parent / "models" / "qwen-image-2.1"
PETITS = ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
          "text_encoder/generation_config.json", "transformer/config.json", "vae/config.json",
          "vae/diffusion_pytorch_model.safetensors"] + [f"processor/{f}" for f in (
              "added_tokens.json", "chat_template.jinja", "merges.txt", "preprocessor_config.json",
              "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
              "video_preprocessor_config.json", "vocab.json")]


def quantifie(t):
    """int8 symetrique par ligne : w ~= q * s, s = max|ligne| / 127."""
    t = t.float()
    s = t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(t / s).clamp(-127, 127).to(torch.int8), s.to(torch.bfloat16)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in PETITS:
        hf_hub_download(REPO, f, local_dir=DST)
    hf_hub_download(GGUF_REPO, "qwen-image-2.1-Q8_0.gguf", local_dir=DST)
    print("[prep] configs, VAE, GGUF ok", flush=True)

    enc = DST / "text_encoder_int8"
    enc.mkdir(exist_ok=True)
    idx = json.load(open(hf_hub_download(REPO, "text_encoder/model.safetensors.index.json", local_dir=DST)))
    for shard in sorted(set(idx["weight_map"].values())):
        out = enc / shard
        if out.exists():
            continue
        src = hf_hub_download(REPO, f"text_encoder/{shard}", local_dir=DST)
        sd, nq, nk = {}, 0, 0
        with safe_open(src, "pt") as fh:
            for k in fh.keys():
                if "visual" in k or k.startswith("lm_head"):
                    continue
                t = fh.get_tensor(k)
                if t.ndim == 2 and (k.endswith("_proj.weight") or k.endswith("embed_tokens.weight")):
                    sd[k], sd[k + ".__echelle"] = quantifie(t)
                    nq += 1
                else:
                    sd[k] = t.contiguous()
                    nk += 1
        save_file(sd, str(out))
        os.remove(src)
        print(f"[prep] {shard} : {nq} matrices int8, {nk} tenseurs gardes -> {out.stat().st_size / 1e9:.2f} Go",
              flush=True)
    print("[prep] TERMINE", flush=True)


def vision():
    """Tour vision seule (edition d'image : l'encodeur doit VOIR l'image de reference). 2e passe, ajoutee
    le 2026-09-29 quand l'edition a ete testee : morceau 1 retelecharge, tour extraite, morceau supprime."""
    out = DST / "text_encoder_int8" / "visual.safetensors"
    if out.exists():
        return
    src = hf_hub_download(REPO, "text_encoder/model-00001-of-00004.safetensors", local_dir=DST)
    sd = {}
    with safe_open(src, "pt") as fh:
        for k in fh.keys():
            if not k.startswith("model.visual."):
                continue
            t = fh.get_tensor(k)
            if t.ndim == 2 and k.endswith(("qkv.weight", "proj.weight", "linear_fc1.weight", "linear_fc2.weight")):
                sd[k], sd[k + ".__echelle"] = quantifie(t)
            else:
                sd[k] = t.contiguous()
    save_file(sd, str(out))
    os.remove(src)
    print(f"[prep] tour vision : {len(sd)} tenseurs -> {out.stat().st_size / 1e9:.2f} Go", flush=True)


if __name__ == "__main__":
    import sys
    vision() if "--vision" in sys.argv else main()

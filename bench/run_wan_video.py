#!/usr/bin/env python
"""run_wan_video.py — VIDEO CONCEPT : Wan 2.2 TI2V-5B (image -> video) via un serveur ComfyUI local (2026-09-30).

Idee du user : la video concept devient l'element central de la pipeline.
  - video de TOUR (monstre immobile, camera qui tourne autour) -> vues COHERENTES pour la 3D (une batte ne peut
    pas changer de sens d'une image a l'autre, contrairement aux vues Qwen generees une par une) ;
  - videos d'ACTION (camera fixe) -> motion capture generique (Puppeteer animation : CoTracker3 + profondeur).
Workflow = gabarit officiel Wan 2.2 5B de ComfyUI (ModelSamplingSD3 shift 8, uni_pc/simple, cfg 5, 20 pas),
modele en GGUF (ComfyUI-GGUF) pour tenir sur 10 Go.

⚠ .venv-pixal3d (le venv qui fait deja tourner ComfyUI). Lance son propre serveur ComfyUI (port 8189, seul le
noeud perso GGUF), soumet le graphe, attend, recupere les images, arrete le serveur.
Usage : run_wan_video.py <image_depart.png> <dossier_sortie> "<prompt>" [--frames 81] [--l 704] [--h 1280]
        [--seed 1] [--pas 20] [--modele 5b|14b]
Sorties : <dossier>/frame_XXXX.png + video.mp4 (si imageio-ffmpeg) + params.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY = ROOT / "third_party" / "ComfyUI"
PORT = 8189
NEG = ("blurry, low quality, distorted, deformed, extra limbs, duplicated objects, "
       "text, watermark, cut off, morphing, changing shape")


def api(chemin, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{chemin}",
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def graphe(image, prompt, a):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-TI2V-5B-Q8_0.gguf"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "3": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "umt5-xxl-encoder-Q8_0.gguf", "type": "wan"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEG}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image}},
        "8": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["6", 0], "width": a.l, "height": a.h, "length": a.frames, "batch_size": 1,
                         "start_image": ["7", 0]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["8", 0],
                         "seed": a.seed, "steps": a.pas, "cfg": 5.0, "sampler_name": "uni_pc",
                         "scheduler": "simple", "denoise": 1.0}},
        "10": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["9", 0], "vae": ["6", 0], "tile_size": 512, "overlap": 64,
                          "temporal_size": 64, "temporal_overlap": 8}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "wan/frame"}},
    }


def graphe_14b(image, prompt, a):
    """Wan 2.2 I2V A14B : 2 experts (bruit fort puis faible) + LoRA Lightx2v 4 pas (cfg 1), gabarit officiel.
    VAE Wan 2.1 (le 14B n'utilise pas le VAE 2.2 du 5B)."""
    mi = a.pas // 2
    g = {
        "3": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "umt5-xxl-encoder-Q8_0.gguf", "type": "wan"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEG}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image}},
        "8": {"class_type": "WanImageToVideo",
              "inputs": {"positive": ["4", 0], "negative": ["5", 0], "vae": ["6", 0], "width": a.l, "height": a.h,
                         "length": a.frames, "batch_size": 1, "start_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["22", 0], "vae": ["6", 0], "tile_size": 512, "overlap": 64,
                          "temporal_size": 64, "temporal_overlap": 8}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "wan/frame"}},
    }
    # --sans-lightx2v : pas d'accelerateur (20 pas, cfg 3.5 : plus lent, mouvements plus amples) ;
    # --lora <prefixe> : LoRA supplementaire par expert (<prefixe>_high_noise / _low_noise.safetensors), ex. orbite
    for k, (bruit, n) in enumerate((("HighNoise", "high_noise"), ("LowNoise", "low_noise"))):
        b = 12 + 4 * k
        g[str(b)] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": f"Wan2.2-I2V-A14B-{bruit}-Q4_K_M.gguf"}}
        src = [str(b), 0]
        if not a.sans_lightx2v:
            g[str(b + 1)] = {"class_type": "LoraLoaderModelOnly",
                             "inputs": {"model": src, "strength_model": 1.0,
                                        "lora_name": f"wan2.2_i2v_lightx2v_4steps_lora_v1_{n}.safetensors"}}
            src = [str(b + 1), 0]
        if a.lora:
            g[str(100 + k)] = {"class_type": "LoraLoaderModelOnly",
                               "inputs": {"model": src, "strength_model": a.lora_force,
                                          "lora_name": f"{a.lora}_{n}.safetensors"}}
            src = [str(100 + k), 0]
        g[str(b + 2)] = {"class_type": "ModelSamplingSD3", "inputs": {"model": src, "shift": 5.0}}
    g["21"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["14", 0], "add_noise": "enable", "noise_seed": a.seed, "steps": a.pas, "cfg": a.cfg,
                          "sampler_name": "euler", "scheduler": "simple", "positive": ["8", 0], "negative": ["8", 1],
                          "latent_image": ["8", 2], "start_at_step": 0, "end_at_step": mi,
                          "return_with_leftover_noise": "enable"}}
    g["22"] = {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["18", 0], "add_noise": "disable", "noise_seed": a.seed, "steps": a.pas,
                          "cfg": a.cfg, "sampler_name": "euler", "scheduler": "simple", "positive": ["8", 0],
                          "negative": ["8", 1], "latent_image": ["21", 0], "start_at_step": mi,
                          "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"), ap.add_argument("sortie"), ap.add_argument("prompt")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--l", type=int, default=704)
    ap.add_argument("--h", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pas", type=int, default=None, help="defaut : 20 (5b), 4 (14b, Lightx2v)")
    ap.add_argument("--modele", choices=["5b", "14b"], default="5b")
    ap.add_argument("--sans-lightx2v", action="store_true", help="14b sans accelerateur : 20 pas, cfg 3.5")
    ap.add_argument("--lora", default=None, help="14b : LoRA en plus, prefixe des fichiers _high_noise/_low_noise")
    ap.add_argument("--lora-force", type=float, default=1.0)
    ap.add_argument("--brut", action="store_true",
                    help="image prise TELLE QUELLE (redimensionnee, sans recadrage) : video alignee sur une camera "
                         "connue (mocap Puppeteer : la 1re image est un rendu du modele)")
    a = ap.parse_args()
    a.pas = a.pas or (4 if a.modele == "14b" and not a.sans_lightx2v else 20)
    a.cfg = 3.5 if a.sans_lightx2v else 1.0
    dst = Path(a.sortie).resolve()        # le serveur tourne dans le dossier ComfyUI : chemins ABSOLUS
    dst.mkdir(parents=True, exist_ok=True)
    tmp_out = dst / "_comfy_out"
    shutil.rmtree(tmp_out, ignore_errors=True)
    (COMFY / "input").mkdir(exist_ok=True)
    nom_img = f"wan_depart_{dst.name}.png"
    # le noeud Wan RECADRE au centre vers l'aspect demande : on COMPLETE plutot le concept (fond = couleur du
    # coin) pour ne rien couper (la batte depasse sur le cote)
    from PIL import Image
    import numpy as np
    im = Image.open(a.image).convert("RGB")
    px = np.asarray(im).astype(int)
    if a.brut:
        assert abs(im.width / im.height - a.l / a.h) < 1e-3, "--brut : meme rapport largeur/hauteur exige"
        im.resize((a.l, a.h), Image.LANCZOS).save(COMFY / "input" / nom_img)
        shutil.copy(COMFY / "input" / nom_img, dst / "depart.png")
    else:
        ys, xs = np.nonzero(np.abs(px - px[2, 2]).sum(-1) > 40)          # silhouette sur fond uni
        m = int(0.08 * (ys.max() - ys.min()))                              # marge : le tour elargit la silhouette
        im = im.crop((max(0, xs.min() - m), max(0, ys.min() - m), min(im.width, xs.max() + m),
                      min(im.height, ys.max() + m)))
        W, H = im.size
        if W / H > a.l / a.h:
            W2, H2 = W, round(W * a.h / a.l)
        else:
            W2, H2 = round(H * a.l / a.h), H
        fond = Image.new("RGB", (W2, H2), im.getpixel((2, 2)))
        fond.paste(im, ((W2 - W) // 2, (H2 - H) // 2))
        fond.resize((a.l, a.h), Image.LANCZOS).save(COMFY / "input" / nom_img)
        shutil.copy(COMFY / "input" / nom_img, dst / "depart.png")

    log = open(dst / "comfy.log", "w", encoding="utf-8")
    srv = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
                            "--disable-auto-launch", "--disable-all-custom-nodes",
                            "--whitelist-custom-nodes", "ComfyUI-GGUF", "--output-directory", str(tmp_out)]
                           # WAN_COMFY_ARGS : options ComfyUI en plus (ex. « --disable-smart-memory » : décharge
                           # l'encodeur de texte après usage -> plus de VRAM pour le modèle vidéo 14B)
                           + os.environ.get("WAN_COMFY_ARGS", "").split(),
                           cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            try:
                api("/system_stats")
                break
            except Exception:
                if srv.poll() is not None:
                    sys.exit(f"ECHEC : le serveur ComfyUI s'est arrete (voir {dst / 'comfy.log'})")
                time.sleep(2)
        t0 = time.time()
        r = api("/prompt", {"prompt": (graphe_14b if a.modele == "14b" else graphe)(nom_img, a.prompt, a)})
        pid = r["prompt_id"]
        print(f"[wan] soumis {pid} ({a.modele}, {a.frames} images {a.l}x{a.h}, {a.pas} pas)", flush=True)
        while True:
            time.sleep(5)
            if srv.poll() is not None:
                sys.exit(f"ECHEC : serveur arrete pendant le calcul (voir {dst / 'comfy.log'})")
            h = api(f"/history/{pid}")
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    sys.exit(f"ECHEC : {json.dumps(st.get('messages', [])[-1:], ensure_ascii=False)[:1500]}")
                if st.get("completed"):
                    break
        print(f"[wan] termine en {time.time() - t0:.0f}s", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()

    frames = sorted((tmp_out / "wan").glob("frame_*.png"))
    for f in dst.glob("frame_*.png"):
        f.unlink()
    for i, f in enumerate(frames):
        shutil.move(str(f), dst / f"frame_{i:04d}.png")
    shutil.rmtree(tmp_out, ignore_errors=True)
    json.dump({"image": str(a.image), "prompt": a.prompt, "neg": NEG, **{k: getattr(a, k) for k in
               ("modele", "frames", "l", "h", "seed", "pas")}}, open(dst / "params.json", "w"), indent=1, ensure_ascii=False)
    try:
        import imageio.v2 as iio
        w = iio.get_writer(dst / "video.mp4", fps=16 if a.modele == "14b" else 24, quality=8)
        for i in range(len(frames)):
            w.append_data(iio.imread(dst / f"frame_{i:04d}.png"))
        w.close()
    except Exception as e:
        print(f"[wan] pas de mp4 ({e}) — les images sont la", flush=True)
    print(f"[wan] {len(frames)} images -> {dst}", flush=True)


if __name__ == "__main__":
    main()

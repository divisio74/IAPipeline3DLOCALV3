#!/usr/bin/env python
"""qwen21_concepts.py — concepts de monstres par Qwen-Image-2.1, pour comparer a SDXL.

Deux etapes dans deux processus (10 Go de VRAM ne tiennent pas les deux modeles a la fois) :
  --etape encode : encodeur Qwen3-VL 8B en int8 (qwen21_prepare.py) -> embeddings sur disque ;
  --etape gen    : transformeur GGUF Q8_0 + VAE, dechargement CPU entre les deux.
Lancer avec PYTHONPATH=.pylib-qwen21 (diffusers/transformers recents, isoles du .venv SDXL).
Sortie : bench/inputs/q21_<monstre>_<variante>/cand_<seed>.png (meme nommage que make_monster).
"""
import argparse
import json
import pathlib
import time

import torch
import torch.nn.functional as F
from torch import nn

B = pathlib.Path(__file__).resolve().parent
M = B.parent / "models" / "qwen-image-2.1"
EMB = B / "out" / "q21_embeddings.pt"
SEEDS = [31337, 31474, 31611, 31748]          # memes graines que les candidats SDXL

SUF = ", single creature, full body, front view, plain light grey background, soft studio lighting"
PROMPTS = {
    # « sdxl » : le texte EXACT donne a SDXL (v2p_*) -> seule variable = le modele d'image.
    ("siren_head", "sdxl"): "two large grey metal air raid siren loudspeakers mounted on top of a tall thin pole "
                            "instead of a head, Siren Head, giant emaciated creature, dry cracked brown skin, "
                            "extremely long skinny arms and legs, long bony fingers" + SUF,
    ("godzilla", "sdxl"): "Godzilla, giant upright reptilian kaiju, vivid dark green scaly skin, three rows of large "
                          "jagged bone-white dorsal plates along its spine, thick long tail, powerful legs, small "
                          "clawed arms, roaring open jaws" + SUF,
    # « nat » : phrases completes, possibles maintenant que le prompt est lu par un modele de langage.
    ("siren_head", "nat"): (
        "Full-body concept art of Siren Head, the towering horror creature by Trevor Henderson. It is an extremely "
        "tall, emaciated humanoid standing upright. It has no head and no face: its neck continues upward into a "
        "long thin rusty metal pole, and at the top of that pole two large rusty air raid siren loudspeakers are "
        "mounted side by side, their horn openings facing forward. Its body is skeletal, covered in dry, cracked, "
        "leathery brown skin stretched tightly over the bones. Its arms are extremely long and thin, hanging down "
        "past its knees, ending in long bony fingers. Its legs are very long and thin. One single creature, "
        "centered, seen from the front, the whole body visible from the top of the sirens to the feet. Plain "
        "uniform light grey studio background, no floor, no ground, no pedestal, no cast shadow, soft even "
        "studio lighting."),
    ("godzilla", "nat"): (
        "Full-body concept art of Godzilla, the classic giant kaiju. It stands upright on two massive, powerful "
        "legs, with a thick muscular body and a very long, heavy tail curving behind it. Its skin is rough, "
        "craggy and scaly, dark green. Three rows of large, jagged, bone-white dorsal plates run down its back "
        "from the neck to the tail. It has small arms with sharp claws, and its jaws are wide open in a roar, "
        "showing its teeth. One single creature, centered, seen from the front, the whole body visible from "
        "head to tail tip. Plain uniform light grey studio background, no floor, no ground, no city, no "
        "pedestal, no cast shadow, soft even studio lighting."),
}


class QLinear(nn.Module):
    """Poids int8 par ligne, dequantifies a la volee en bf16 (le calcul reste en bf16)."""

    def __init__(self, q, s, b=None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.register_buffer("b", b)

    def forward(self, x):
        b = None if self.b is None else self.b.to(x.dtype)
        return F.linear(x, self.q.to(x.dtype) * self.s.to(x.dtype), b)


class QEmbedding(nn.Module):
    def __init__(self, q, s):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("s", s)

    def forward(self, ids):
        return self.q[ids].to(torch.bfloat16) * self.s[ids]


def charge_encodeur():
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLTextRotaryEmbedding,
                                                                Qwen3VLVisionRotaryEmbedding)

    cfg = AutoConfig.from_pretrained(M / "text_encoder")
    with torch.device("meta"):
        mod = Qwen3VLForConditionalGeneration(cfg)
    vision = (M / "text_encoder_int8" / "visual.safetensors").exists()
    if not vision:
        mod.model.visual = None        # texte -> image seul : la tour vision ne sert pas
    mod.lm_head = nn.Identity()        # seuls les etats caches sont lus
    noms = {n for n, _ in mod.named_parameters()}
    sd = {}
    for f in sorted((M / "text_encoder_int8").glob("*.safetensors")):
        sd.update(load_file(f))
    simples = {}
    for k, t in sd.items():
        if k.endswith(".__echelle") or (k.endswith(".bias") and k[:-5] + ".weight.__echelle" in sd):
            continue
        if k + ".__echelle" in sd:
            chemin = k[: -len(".weight")]
            parent, _, feuille = chemin.rpartition(".")
            q, s = t.cuda(), sd[k + ".__echelle"].cuda()
            b = sd.get(chemin + ".bias")
            new = QEmbedding(q, s) if feuille == "embed_tokens" else QLinear(q, s, None if b is None else b.cuda())
            setattr(mod.get_submodule(parent), feuille, new)
        else:
            simples[k] = t.cuda()
    manquants = mod.load_state_dict(simples, strict=False, assign=True)
    reste = [n for n, p in mod.named_parameters() if p.is_meta]
    inconnus = [k for k in simples if k not in noms]
    print(f"[q21] encodeur : {len(sd)} tenseurs, params encore vides {reste[:5]}, cles inconnues "
          f"{inconnus[:5]}", flush=True)
    assert not reste and not inconnus, "noms de cles incompatibles"
    lm = mod.model.language_model
    lm.rotary_emb = Qwen3VLTextRotaryEmbedding(config=cfg.text_config, device="cuda")
    if vision:
        mod.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(cfg.vision_config).to("cuda")
    for n, b in mod.named_buffers():
        assert not b.is_meta, f"tampon non initialise : {n}"
    del manquants
    return mod.eval()


def etape_encode():
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    t0 = time.time()
    proc = Qwen3VLProcessor.from_pretrained(M / "processor")
    enc = charge_encodeur()
    print(f"[q21] encodeur charge en {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} Go",
          flush=True)
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=enc, processor=proc, transformer=None)
    out = {}
    with torch.no_grad():
        for (monstre, var), p in PROMPTS.items():
            e, m, _ = pipe._get_qwen_prompt_embeds([p], device=torch.device("cuda"))
            out[f"{monstre}|{var}"] = (e.cpu(), m.cpu())
            print(f"[q21] {monstre}/{var} : {e.shape[1]} jetons", flush=True)
    torch.save(out, EMB)
    print(f"[q21] embeddings -> {EMB}, pic VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} Go", flush=True)


def etape_gen(steps):
    pipe = _pipe_gen()
    emb = torch.load(EMB)
    _boucle_t2i(pipe, emb, steps)


def _pipe_gen():
    from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                           QwenImage21Pipeline, QwenImage21Transformer2DModel)
    from transformers import Qwen3VLProcessor

    tr = QwenImage21Transformer2DModel.from_single_file(
        str(M / "qwen-image-2.1-Q8_0.gguf"), config=str(M), subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16)
    # Bug diffusers (main, 2026-09-29) : les tenseurs BF16 NON quantifies du GGUF restent des octets bruts
    # (4096 bf16 -> 8192 uint8) ; les normes les lisent tels quels. On les decode une fois pour toutes.
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    import gguf
    n = 0
    for mod in tr.modules():
        for nom, p in list(mod.named_parameters(recurse=False)):
            if isinstance(p, GGUFParameter) and p.quant_type == gguf.GGMLQuantizationType.BF16:
                vrai = dequantize_gguf_tensor(p).to(torch.bfloat16)
                setattr(mod, nom, nn.Parameter(vrai.as_subclass(torch.Tensor).clone(), requires_grad=False))
                n += 1
    print(f"[q21] {n} tenseurs BF16 du GGUF decodes", flush=True)
    pipe = QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(M, subfolder="scheduler"),
        vae=AutoencoderKLQwenImage21.from_pretrained(M, subfolder="vae", torch_dtype=torch.bfloat16),
        text_encoder=None, processor=Qwen3VLProcessor.from_pretrained(M / "processor"), transformer=tr)
    pipe.enable_model_cpu_offload()
    return pipe


def _boucle_t2i(pipe, emb, steps):
    for cle, (e, m) in emb.items():
        monstre, var = cle.split("|")
        dst = B / "inputs" / f"q21_{monstre}_{var}"
        dst.mkdir(parents=True, exist_ok=True)
        json.dump({"prompt": PROMPTS[(monstre, var)], "modele": "Qwen-Image-2.1 GGUF Q8_0", "steps": steps},
                  open(dst / "prompt.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for seed in SEEDS:
            p = dst / f"cand_{seed}.png"
            if p.exists():
                continue
            t0 = time.time()
            img = pipe(prompt_embeds=e.cuda(), prompt_embeds_mask=m.cuda(), height=1024, width=1024,
                       num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            img.save(p)
            print(f"[q21] {monstre}/{var} seed {seed} : {time.time() - t0:.0f}s, mode {img.mode}, "
                  f"pic VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} Go", flush=True)


# ---------------------------------------------------------------- VUES par EDITION (2026-09-29)
# Qwen-Image-2.1 edite une image de reference : on lui demande la MEME creature sous un autre angle.
VUES_REF = B / "inputs" / "q21_siren_head_nat" / "cand_31474.png"   # concept retenu par le gate (v2q)
VUES_DST = B / "inputs" / "q21_vues_siren_head"
VUES_EMB = B / "out" / "q21_vues_embeddings.pt"
_GARDE = ("Keep the exact same creature: identical anatomy, proportions, body shape, the same two siren "
          "loudspeakers with the same shape and size, same skin colour and texture, same materials. Same plain "
          "uniform light grey studio background, same soft even studio lighting, same scale, the whole body "
          "visible from the top of the sirens to the feet, no floor, no cast shadow.")
VUES = {
    "gauche": "Rotate the camera 90 degrees around the creature: show this exact same creature in a strict side "
              "profile view, its body facing towards the LEFT edge of the image. " + _GARDE,
    "droite": "Rotate the camera 90 degrees around the creature: show this exact same creature in a strict side "
              "profile view, its body facing towards the RIGHT edge of the image. " + _GARDE,
    "dos": "Rotate the camera 180 degrees around the creature: show this exact same creature seen from directly "
           "behind, showing its back, the back of the sirens and the back of its legs. " + _GARDE,
    # 2026-09-29 : « dos » a donne 1 fois sur 2 une sirene encore vue par son ouverture (incoherent). Le prompt
    # decrit maintenant CE QU'ON DOIT VOIR de dos (l'encodeur est un modele de langage, il lit la phrase).
    "dos_precis": "Rotate the camera 180 degrees around the creature: show this exact same creature seen from "
                  "directly behind. We see its back and spine, the back of its arms, the back of its legs and its "
                  "heels. At the top of the pole, BOTH siren loudspeakers point away from the viewer, towards the "
                  "far side of the image: we only see the closed, rounded rear housing of each horn, with the "
                  "driver unit in its middle. Neither horn opening is visible; both sirens are seen from the back "
                  "and look symmetric. " + _GARDE,
}


FILTRE = None

# Slenderman (2026-09-29) : concept de REFERENCE des tests de septembre (SDXL, gen_slenderman), pose sur fond
# gris clair pour Qwen. Le prompt de vue decrit ce qu'on doit voir (lecon du « dos_precis »).
_GARDE_SL = ("Keep the exact same creature: the identical smooth white faceless head, the same black suit, white "
             "shirt and black tie, the same proportions, and the same black tentacles growing from its upper back, "
             "with the same number, length, thickness and spread. Same plain uniform light grey studio background, "
             "same soft even studio lighting, same scale, the whole body and all the tentacles visible, no floor, "
             "no cast shadow.")
VUES_SLENDER = {
    "gauche": "Rotate the camera 90 degrees around the creature: show this exact same creature in a strict side "
              "profile view, its body facing towards the LEFT edge of the image. The tentacles grow out of its "
              "upper back, so in profile they start behind its shoulders and spread out behind and around it. "
              + _GARDE_SL,
    "droite": "Rotate the camera 90 degrees around the creature: show this exact same creature in a strict side "
              "profile view, its body facing towards the RIGHT edge of the image. The tentacles grow out of its "
              "upper back, so in profile they start behind its shoulders and spread out behind and around it. "
              + _GARDE_SL,
    "dos_precis": "Rotate the camera 180 degrees around the creature: show this exact same creature seen from "
                  "directly behind. We see the smooth white back of its head (no face), the back of its black suit "
                  "jacket, the back of its trousers and the heels of its shoes, and the roots of all the black "
                  "tentacles growing out of its upper back and spreading outward to both sides. " + _GARDE_SL,
}


def _config_slenderman():
    global VUES_REF, VUES_DST, VUES_EMB, VUES
    from PIL import Image
    VUES_DST = B / "inputs" / "q21_vues_slenderman"
    VUES_DST.mkdir(parents=True, exist_ok=True)
    VUES_REF = VUES_DST / "ref.png"
    if not VUES_REF.exists():
        c = Image.open(B / "inputs" / "cmp_slenderman_couleur" / "02_cut.png").convert("RGBA")
        fond = Image.new("RGBA", c.size, (225, 225, 225, 255))
        fond.alpha_composite(c)
        fond.convert("RGB").save(VUES_REF)
    VUES_EMB = B / "out" / "q21_vues_slenderman_embeddings.pt"
    VUES = VUES_SLENDER


def _config_job(chemin, ref=None):
    """Fiche monstre bench/jobs/<id>.json (2026-09-29, chaine v3) : concept en phrases + vues decrites.
    Sorties : inputs/q21_<id>_nat/cand_<seed>.png (concepts) et inputs/q21_vues_<id>/ (vues)."""
    global PROMPTS, EMB, VUES_REF, VUES_DST, VUES_EMB, VUES
    j = json.load(open(chemin, encoding="utf-8"))
    PROMPTS = {(j["id"], "nat"): j["concept"]}
    EMB = B / "out" / f"q21_emb_{j['id']}.pt"
    VUES_DST = B / "inputs" / f"q21_vues_{j['id']}"
    VUES_EMB = B / "out" / f"q21_vues_emb_{j['id']}.pt"
    VUES = {v: f"{t} {j['garde']}" for v, t in j["vues"].items()}
    if ref:
        VUES_REF = B / "inputs" / f"q21_{j['id']}_nat" / ref


def _ref_redim(pipe):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from PIL import Image
    img = Image.open(VUES_REF).convert("RGBA")          # meme pretraitement que __call__
    w, h, _ = calculate_dimensions(1024 * 1024, img.width / img.height)
    return img, pipe.image_processor.resize(img, width=w, height=h)


def etape_encode_vues():
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    proc = Qwen3VLProcessor.from_pretrained(M / "processor")
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=charge_encodeur(), processor=proc,
                               transformer=None)
    _, ref = _ref_redim(pipe)
    out = {}
    with torch.no_grad():
        for v, p in VUES.items():
            if FILTRE and v not in FILTRE:
                continue
            e, m, ipm = pipe.encode_prompt(p, image=[ref], device=torch.device("cuda"))
            out[v] = tuple(None if t is None else t.cpu() for t in (e, m, ipm))
            print(f"[q21] vue {v} : {e.shape[1]} jetons dont {int(ipm.sum())} d'image", flush=True)
    torch.save(out, VUES_EMB)
    print(f"[q21] pic VRAM encodeur+vision {torch.cuda.max_memory_allocated() / 1e9:.1f} Go", flush=True)


def _vae_dehors(pipe, i, t, kw):
    """Le dechargement auto ne fait descendre QUE le modele precedent de la chaine (encodeur -> transformeur ->
    VAE) : le VAE, monte pour encoder l'image de reference, reste sur la carte pendant tout le debruitage
    (+1,35 Go -> debordement en RAM partagee, 23-40 s/pas). Son crochet le remontera pour le decodage."""
    if i == 0:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
    return kw


def etape_gen_vues(steps, graines, cache=True):
    pipe = _pipe_gen()
    img, _ = _ref_redim(pipe)
    emb = torch.load(VUES_EMB)
    VUES_DST.mkdir(parents=True, exist_ok=True)
    for v, (e, m, ipm) in emb.items():
        fixe = tuple(None if t is None else t.cuda() for t in (e, m, ipm))
        pipe.encode_prompt = lambda *a, _f=fixe, **k: _f      # embeddings calcules par l'autre processus
        for seed in graines:
            p = VUES_DST / f"{v}_{seed}.png"
            if p.exists():
                continue
            t0 = time.time()
            out = pipe(prompt_embeds=fixe[0], image=[img], height=1024, width=1024, num_inference_steps=steps,
                       generator=torch.Generator("cuda").manual_seed(seed), use_kv_cache=cache,
                       callback_on_step_end=_vae_dehors).images[0]
            out.save(p)
            print(f"[q21] vue {v} seed {seed} : {time.time() - t0:.0f}s, "
                  f"pic VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} Go", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etape", choices=["encode", "gen", "encode-vues", "gen-vues"], required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--graines", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--vues", nargs="+", default=None, help="n'encoder que ces vues (cles de VUES)")
    ap.add_argument("--monstre", default="siren_head", choices=["siren_head", "slenderman"])
    ap.add_argument("--job", default=None, help="fiche bench/jobs/<id>.json (remplace --monstre)")
    ap.add_argument("--ref", default=None, help="avec --job : concept retenu (cand_<seed>.png) pour les vues")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="graines des concepts")
    ap.add_argument("--sans-cache", action="store_true",
                    help="edition : le cache KV de l'image de reference (~2,7 Go) fait deborder 10 Go -> 23 s/pas")
    a = ap.parse_args()
    FILTRE = a.vues
    if a.monstre == "slenderman":
        _config_slenderman()
    if a.job:
        _config_job(a.job, a.ref)
    if a.seeds:
        SEEDS = a.seeds
    {"encode": etape_encode, "gen": lambda: etape_gen(a.steps), "encode-vues": etape_encode_vues,
     "gen-vues": lambda: etape_gen_vues(a.steps, a.graines, not a.sans_cache)}[a.etape]()

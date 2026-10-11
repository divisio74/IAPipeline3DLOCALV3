# IAPipeline3DLOCAL V2 — pipeline « monstres » 100 % locale

Pipeline qui transforme **une fiche texte** en **monstre 3D texturé et animé**, sans intervention
humaine, sur **une seule carte graphique grand public** (RTX 3080, 10 Go).

C'est la première « usine à contenu » d'un futur jeu où chaque monstre et chaque objet est généré pour
un joueur, puis jeté. La vision complète est dans [`VISION.md`](VISION.md). Ce README s'adresse aux
**développeurs** : comment c'est construit, comment l'installer, comment le faire tourner, et
**comment en dériver une pipeline plus simple** (par exemple des meubles, §8).

> État : 11 octobre 2026. La chaîne complète tourne **seule, de la fiche au `.glb` animé prêt pour un moteur de jeu**,
> en une commande (`bench/fabrique_monstre.sh`, environ 40 min par monstre). Seule action animée : la marche. Voir §7.

---

## Sommaire

1. [Ce que fait la pipeline](#1-ce-que-fait-la-pipeline)
2. [L'architecture et pourquoi](#2-larchitecture-et-pourquoi)
3. [Les étapes, script par script](#3-les-étapes-script-par-script)
4. [Conventions à connaître](#4-conventions-à-connaître)
5. [Installation](#5-installation)
6. [Faire tourner la pipeline](#6-faire-tourner-la-pipeline)
7. [État, résultats et limites connues](#7-état-résultats-et-limites-connues)
8. [Guide : dériver une pipeline « meubles »](#8-guide--dériver-une-pipeline--meubles-)
9. [Règles de travail et pièges](#9-règles-de-travail-et-pièges)

---

## 1. Ce que fait la pipeline

```
fiche JSON ─► concepts ─► choix auto ─► vidéo de tour ─► 4 vues ─► modèle 3D + texture ─► directeur ─► marche
           ─► boucle ─► .glb animé (texture et normal map cuites)
```

- **Entrée** : une fiche `bench/jobs/<id>.json`, qui décrit le monstre en phrases complètes.
- **Sorties** (dans `bench/out/<id>/`) :
  - `tex_p3d.glb` : modèle 3D texturé ;
  - `geo_16k.glb` : maillage léger (environ 7 500 sommets) pour l'animation ;
  - `directeur2/rapport.txt|json` : membres, objets et articulations du modèle ;
  - `rdmesh_marche_trajs.npz` + GIF : animation de marche ;
  - **`final/<id>_marche.glb`** : le monstre animé pour le jeu (marche en boucle, sans avancée, ~16 000 triangles,
    couleur + normal map 2048 px) et `final/<id>_marche.json` (durée du cycle, vitesse de marche à appliquer).

Rien n'est codé pour un type de créature précis. La même chaîne traite un humanoïde, une araignée, un
loup, une limace ou un personnage qui tient un objet.

## 2. L'architecture et pourquoi

Note de l'auteur :
l'archi actuelle ets basé sur 4 vues coté model 3D et la video entiere acvec suelement una ngle de camera pour l'animation la est le point faible de l'animation actuellement pour les models les plus complex généralement les créature non bipede avec des membres hors du commun

### 2.1 Les rôles

| Rôle | Qui | Ce qu'il fait |
|---|---|---|
| **Cerveau** | un LLM | écrit les fiches, les consignes (ex. l'allure de marche adaptée au corps) |
| **Yeux** | un VLM (Qwen3-VL-4B) | regarde les images et les vidéos : choix des vues, liste des membres et des objets |
| **Mains** | le code (scripts `bench/`) | rendus, mesures, géométrie, assemblage |
| **Mémoire** | la fiche + le rapport | suivent le monstre d'une étape à l'autre ; chaque étape y lit son contexte |

Le **modèle 3D fait foi** : le directeur ne corrige pas le modèle, il le **décrit**, et les étapes
suivantes s'adaptent à ce qui existe. On évite ainsi les boucles « générer → contrôler →
regénérer », trop coûteuses.

### 2.2 Les choix structurants, et les essais qui les justifient

1. **La vidéo est l'élément central.** Une IA vidéo (Wan 2.2 14B) filme le monstre en tournant
   autour. On en tire des **vues cohérentes** (face, profils, dos) pour la 3D. Générer chaque vue
   séparément (MV-Adapter, édition d'image) donnait des vues contradictoires, par exemple un objet
   tenu dans la main gauche sur une vue et dans la droite sur une autre.
2. **LoRA « orbite » obligatoire pour le tour.** Sans elle, Wan ne montre jamais le dos des corps
   allongés (loup à deux têtes, limace qui revient vers la face).
3. **3D multi-vues avec caméras connues (Pixal3D).** TRELLIS.2 n'a pas de notion de caméra, et son
   « multi-vues » mélange des devinettes contradictoires. Pixal3D rétroprojette les 4 vues dans la
   grille 3D. C'est nettement meilleur (verdict visuel).
4. **« Argile » avant la 3D (variante A2).** Chaque vue est redessinée en argile grise uniforme
   (ControlNet Canny SDXL, gris recalé dans [55, 215]) avant la reconstruction. Le relief devient
   lisible, même pour un monstre noir, et il n'y a plus de reflets brûlés. La couleur est réappliquée
   ensuite (texture Pixal3D à partir des vues couleur).
5. **Règle de pose dans le concept** : chaque membre est séparé du corps par un vide visible (A-pose,
   pattes en étoile). Sinon la 3D soude les bras au tronc, et l'animation crée des « membranes ».
6. **Animation pilotée par une vidéo d'action.** Le LLM écrit l'action adaptée au corps, Wan la
   filme en partant d'un rendu du modèle 3D, puis R-DMesh déforme le maillage pour suivre la vidéo,
   sans squelette.
7. **Une variable à la fois, verdict à l'œil.** Plusieurs indicateurs chiffrés se sont révélés
   trompeurs (une tentacule cassée en 5 morceaux comptait pour 5 tentacules, des facettes comptaient
   comme du détail…). On mesure, mais **on regarde toujours le rendu** avant de conclure.

## 3. Les étapes, script par script

Tous les scripts sont dans `bench/`. Les chaînes `.sh` enchaînent les scripts Python et Blender.

| # | Étape | Script(s) | Environnement |
|---|---|---|---|
| — | **Tout, de la fiche au `.glb` final, avec reprise après erreur** | **`fabrique_monstre.sh <id>`** | — |
| 0 | Téléchargement / préparation de Qwen-Image-2.1 (une fois) | `qwen21_prepare.py` | `.venv` + `.pylib-qwen21` |
| 1 | **Concepts** : 3 images de face depuis la fiche (3 graines) | `qwen21_concepts.py --job` (2 passes : `encode` puis `gen`) | `.venv` + `.pylib-qwen21` |
| 2 | **Choix du concept** : le VLM vérifie les nombres de parties de la description, le cadrage, la pose ; ex æquo départagés par l'écartement de la silhouette | `choisir_concept.py` (+ `masque_gpu.py`) | `.venv-trellis2` |
| 3 | **Vidéo de tour** (Wan 2.2 14B + Lightx2v + LoRA orbite) | `run_wan_video.py` (pilote un serveur ComfyUI local) | `.venv-pixal3d` |
| 4 | **Choix des vues** (VLM face/profil/dos sur 1 image sur 2 + frontières, flux optique, profils interpolés) | `choisir_vues.py` | `.venv-trellis2` |
| 5 | **Argiles A2** des 4 vues | `exp_vues_3d.py` → `make_monster_v2.py` (+ `exp_argile.py`, `exp_contraste.py`) | `.venv` |
| 6 | **3D 4 vues** (Pixal3D) puis simplification | `chaine_v3_3d.sh` → `run_pixal3d_mv.py`, `t2_simplify.py` | `.venv-pixal3d`, `.venv-trellis2` |
| 7 | **Texture** 4 vues (Pixal3D) | `run_pixal3d_tex.py` (vues couleur via `exp_vues_3d.py couleurs-job`) | `.venv-pixal3d` |
| — | Étapes 3 à 7 enchaînées | `chaine_complete.sh <id> <cand.png>` | — |
| 8 | **Directeur** : membres, objets, articulations | `directeur_seul.sh` → `bl_directeur.py`, `directeur2.py`, `directeur2_rapport.py` | `.venv-trellis2` + Blender |
| 9 | **Marche** (recette validée le 2026-10-06) : allure LLM avec le nombre exact de chaque partie → rendu de départ de profil → vidéo Wan → masques GPU → contrôles → fond noir, 32 img/s → R-DMesh (bf16) sur `geo_16k` pivoté → GIF | `chaine_marche.sh <id>` → `allure_marche.py`, `bl_depart_marche.py`, `run_wan_video.py`, `masque_gpu.py`, `controle_marche.py`, `t2_simplify.py`, `bl_rendu_trajs.py` | plusieurs |
| 10 | **Boucle de marche** : avancée retirée (vitesse rendue au jeu), cycle découpé au creux de ressemblance, écart réparti | `boucle_marche.py` | n'importe quel venv (numpy) |
| 11 | **Export moteur de jeu** : `.glb` à cibles de morphing (une par image), de face, posé au sol ; **cuisson** couleur + normal map depuis le modèle texturé complet | `bl_export_anim_glb.py` (Blender) ; `couleurs_sommets.py` pour une variante sans cuisson (couleurs de sommets) | Blender |

### Structure des dossiers (créés à l'exécution, non versionnés)

```
bench/
├── jobs/<id>.json                 # fiches (versionnées)
├── inputs/q21_<id>_nat/choix.json         # choix automatique du concept
├── inputs/q21_<id>_nat/cand_<graine>.png   # concepts générés
├── views/<id>/, views_couleur/<id>/        # vues argile / couleur
└── out/<id>/
    ├── video_tour/frame_XXXX.png
    ├── p3d_mv.glb, geo_100k.glb, geo_16k.glb, geo_16k_profil.glb (pivoté pour R-DMesh), tex_p3d.glb
    ├── directeur2/ (vues, cadres, rapport)
    ├── marche_allure.txt, depart_marche.png, marche_prompt.txt, marche_video_retenue.txt
    ├── video_marche_s1/ (images, masque/, noir/, noir32/, controle.json|png)
    ├── rdmesh_marche_trajs.npz, rdmesh_marche/, rdmesh_marche_boucle.npz|json
    └── final/ (<id>_marche.glb, <id>_marche.json, textures cuites, GIF)
```

## 4. Conventions à connaître

- **Repère des modèles (glTF)** : Y vers le haut, **avant de la créature = +Z**, **gauche de la
  créature = +X**. Blender importe en Z vers le haut : glTF (x, y, z) → Blender (x, −z, y).
- **TRELLIS.2 sort en Z vers le haut** : `run_trellis2.py` (non inclus) le redressait. Pixal3D sort
  déjà en Y vers le haut.
- **Azimut des caméras** (`bl_directeur.py`, `bl_depart_marche.py`) : 0 = face, 90 = profil gauche
  (créature tournée vers le bord gauche de l'image), 180 = dos, 270 = profil droit.
- **R-DMesh** :
  - maillage d'au plus 8 192 sommets (limite de ses données d'entraînement) : `geo_16k.glb` = 16 000 faces,
    environ 7 500 sommets (`t2_simplify.py`), meilleur à l'œil que l'ancien `geo_20k` ;
  - vidéo sur **fond noir**, 65 premières images, recadrée **au centre en carré 256×256** : on complète
    donc la vidéo paysage par des bandes noires plutôt que de la laisser recadrer ;
  - la vidéo Wan (16 img/s) est **interpolée à 32 img/s** (ffmpeg minterpolate) :
    R-DMesh a appris sur des animations à 24-30 img/s ;
  - réglages des auteurs : `--num_hops 5 --alpha_hops 0.7 --num_traj 4096 --guidance_scale 1.5`, calcul en
    bf16 (`RDMESH_BF16=1`, indiscernable du float32 à l'œil) ;
  - il **suppose une caméra de face** : on **pivote le maillage** du même angle que la caméra de la
    vidéo avant de le lui donner (`geo_16k_profil.glb`, fait par `chaine_marche.sh`) ;
  - une créature sombre est éclaircie au gamma, sinon elle disparaît sur le fond noir.
- **Fiche** (`bench/jobs/<id>.json`) : `concept` (prompt complet en phrases, avec la règle de pose et
  « strictly from the front »), `desc_courte` (réutilisée dans tous les prompts vidéo), `garde`,
  `vues`.

## 5. Installation

### 5.1 Matériel et logiciels testés

- Windows 11, RTX 3080 10 Go (pilote compatible CUDA 13), 32 Go de RAM, environ 120 Go de disque
  pour les modèles.
- **Git Bash** pour les scripts `.sh`. Attention : depuis PowerShell, `bash` lance **WSL**, pas Git
  Bash. Utilisez `"C:\Program Files\Git\bin\bash.exe"`.
- **Blender 5.1** dans `C:/Program Files/Blender Foundation/Blender 5.1/` (chemin écrit en tête des
  chaînes `.sh`, à adapter).
- **ffmpeg** dans le PATH (interpolation de la vidéo de marche).
- Python 3.10, 3.11 et 3.13 installés (lanceur `py`).

### 5.2 Disposition attendue

Les scripts se lancent depuis la racine du dépôt et attendent cette arborescence :

```
<racine>/
├── bench/                 # ce dépôt
├── patches/               # ce dépôt
├── requirements/          # ce dépôt
├── .venv, .venv-trellis2, .venv-pixal3d, .venv-unimate
├── .pylib-qwen21/         # diffusers/transformers récents, isolés
├── models/qwen-image-2.1/
└── third_party/
    ├── ComfyUI/ (+ custom_nodes/ComfyUI-Trellis2, custom_nodes/ComfyUI-GGUF)
    └── R-DMesh/
```

### 5.3 Environnements Python

Chaque environnement a son `requirements/<nom>.txt` (sortie de `pip freeze`). Les versions de torch
diffèrent volontairement : chaque modèle a ses roues précompilées.

| Environnement | Python | torch | Sert à |
|---|---|---|---|
| `.venv` | 3.10 | 2.5.1 cu121 | SDXL + ControlNet (argile), birefnet (rembg), outils ; avec `PYTHONPATH=.pylib-qwen21` : Qwen-Image-2.1 |
| `.venv-trellis2` | 3.11 | 2.7.0 cu128 | VLM Qwen3-VL-4B (choix du concept, choix des vues, directeur, allure), BiRefNet GPU (masques), CuMesh (simplification) |
| `.venv-pixal3d` | 3.13 | 2.10.0 cu130 | Pixal3D (3D + texture), client ComfyUI pour Wan |
| `.venv-unimate` | 3.10 | 2.5.1 cu124 | R-DMesh |

```bash
py -3.11 -m venv .venv-trellis2
./.venv-trellis2/Scripts/python.exe -m pip install -r requirements/venv-trellis2.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Remarques :

- **Roues Windows précompilées** de `ComfyUI-Trellis2` (cumesh, nvdiffrast, flex_gemm, o_voxel,
  natten…) : dossier `custom_nodes/ComfyUI-Trellis2/wheels/Windows/`. Prenez `Torch270/` (cp311) pour
  `.venv-trellis2` et `Torch2100/CUDA 13.1/` (cp313) pour `.venv-pixal3d`.
- `.venv-trellis2` : `torchaudio==2.7.0` doit être réinstallé avec `--no-deps --force-reinstall`, sinon
  `WinError 127` à l'import de transformers. Fixez `ATTN_BACKEND=sdpa` (pas de flash-attn sous Windows).
- `.pylib-qwen21` : `pip install --target .pylib-qwen21 "git+https://github.com/huggingface/diffusers"
  transformers==5.17.0 gguf==0.19.0`, utilisé uniquement via `PYTHONPATH`.

### 5.4 Dépôts tiers et correctifs

| Dépôt | Commit testé | Correctif |
|---|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | `7a0b5ee` | — |
| [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) (dans `custom_nodes/`) | `1459741` | — |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (dans `custom_nodes/`) | `6ea2651` | — |
| [R-DMesh](https://github.com/Tencent-Hunyuan/R-DMesh) | `467cb35` | `patches/R-DMesh.patch` |

```bash
git -C third_party/R-DMesh checkout 467cb35 && git -C third_party/R-DMesh apply --binary ../../patches/R-DMesh.patch
```

Ce que font les correctifs :

- **R-DMesh** :
  - attention SDPA avec masque, car flash-attn n'existe pas sous Windows ;
  - chargement sans T5 (`RDMESH_SANS_T5=1`, contexte nul fourni) ;
  - VAE Wan 2.2 de ComfyUI (`RDMESH_VAE`) ;
  - Wan 5B lu depuis le GGUF Q8, 11 premiers blocs seulement (`RDMESH_WAN_GGUF`) ;
  - `num_workers=0`, correction de `num_traj=-1` ;
  - sauvegarde des trajectoires `<video>_trajs.pt` ;
  - `RDMESH_SANS_RENDU=1` saute le rendu Cycles interne ;
  - `RDMESH_BF16=1` : modèle et calcul en bfloat16.

### 5.5 Modèles à télécharger

| Modèle | Où le mettre | Source |
|---|---|---|
| Qwen-Image-2.1 (GGUF Q8_0 + encodeur converti en int8) | `models/qwen-image-2.1/` | `qwen21_prepare.py` (télécharge `Qwen/Qwen-Image-2.1` + `unsloth/Qwen-Image-2.1-GGUF`) |
| Wan 2.2 I2V A14B HighNoise / LowNoise Q4_K_M (GGUF) | `third_party/ComfyUI/models/unet/` | dépôt GGUF de Wan 2.2 I2V A14B (ex. QuantStack) |
| Wan 2.2 TI2V 5B Q8_0 (GGUF, utilisé par R-DMesh) | `…/models/unet/` | idem (TI2V-5B) |
| LoRA Lightx2v 4 pas (high/low) | `…/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors` | dépôt Lightx2v / Comfy-Org repackaged |
| LoRA orbite (high/low), déclencheur « orbit 360 » | `…/models/loras/wan22_14b_i2v_orbit_{high,low}_noise.safetensors` | `ostris/wan22_i2v_14b_orbit_shot_lora` |
| Encodeur texte umt5-xxl Q8_0 (GGUF) | `…/models/text_encoders/` | dépôt GGUF umt5 (ex. city96) |
| VAE Wan 2.2 / 2.1 | `…/models/vae/` | Comfy-Org repackaged |
| Pixal3D multi-vues (`ckpts/*_mv`, forme + texture) | `…/models/TencentARC/Pixal3D/ckpts/` | `TencentARC/Pixal3D` |
| DINOv3 ViT-L/16 (lecture des vues par Pixal3D) | `…/models/facebook/dinov3-vitl16-pretrain-lvd1689m/` | `visualbruno/dinov3-vitl16-pretrain-lvd1689m` (téléchargé par les nœuds au premier lancement) |
| Encodeur de forme TRELLIS.2 (`ckpts/shape_enc_next_dc_f16c32_fp16.{json,safetensors}` seulement : Pixal3D en a besoin pour texturer un maillage existant) | `…/models/microsoft/TRELLIS.2-4B/ckpts/` | `microsoft/TRELLIS.2-4B` (téléchargé par les nœuds au premier lancement) |
| MoGe-2 ViT-L (estimation de la caméra des vues, chargé par Pixal3D) | `…/models/Ruicheng/moge-2-vitl/` | `Ruicheng/moge-2-vitl` (téléchargé par les nœuds au premier lancement) |
| SDXL base 1.0 + ControlNet Canny SDXL | cache HF (`models/`) | `stabilityai/stable-diffusion-xl-base-1.0`, `diffusers/controlnet-canny-sdxl-1.0` |
| Qwen3-VL-4B-Instruct | cache HF | `Qwen/Qwen3-VL-4B-Instruct` |
| BiRefNet DIS5K (détourage GPU) | cache HF | `ZhengPeng7/BiRefNet-DIS5K` (téléchargé par `masque_gpu.py`) |
| birefnet-dis (détourage CPU, argiles) | cache rembg | téléchargé automatiquement par `rembg` |
| R-DMesh (`ckpts/dvae`, `dvae_factor`, `rf_model`) | `third_party/R-DMesh/ckpts/` | README de R-DMesh |

Utilisez `HF_HUB_DISABLE_XET=1` pour les téléchargements HF, et vérifiez l'espace disque avant chaque
gros modèle.

## 6. Faire tourner la pipeline

Depuis Git Bash, à la racine. La fabrication est longue (~40 min) : lancez-la **détachée**. Depuis PowerShell, le
plus sûr est un petit script `lance.sh` qui contient `cd <racine> && bash bench/fabrique_monstre.sh "$1" > bench/out/_fabrique_$1.log 2>&1`,
puis `Start-Process "C:\Program Files\Gitinash.exe" -ArgumentList @("C:/chemin/lance.sh", "v3_quadrupede")`.

```bash
# Prérequis : la fiche bench/jobs/<id>.json (concept, desc_courte, garde, vues), écrite par le LLM « créateur ».
# Tout, de la fiche au monstre animé pour le jeu (reprise automatique : relancez la même commande après une erreur)
bash bench/fabrique_monstre.sh v3_quadrupede            # graines des concepts : 31337 31474 31611 par défaut
#    -> bench/out/v3_quadrupede/final/v3_quadrupede_marche.glb  (+ .json : durée du cycle, vitesse de marche)
#    -> bench/out/animations_gif/marche_v3_quadrupede.gif       (vidéo Wan + animation, pour vérifier à l'œil)
```

Les étapes peuvent aussi être lancées une à une (voir le tableau du §3 et l'en-tête de `fabrique_monstre.sh`).

Dans **Godot 4** : importer le `.glb` et jouer l'animation « marche » en boucle ; le déplacement du monstre est à
faire par le jeu, à la vitesse donnée dans le `.json` (l'avancée a été retirée de l'animation).

Temps mesurés (RTX 3080, aucune autre application 3D ouverte), environ **40 min par monstre** :

| Étape | Durée |
|---|---|
| Concepts (3 images) | ~4 min |
| Choix du concept | ~1 min |
| Vidéo de tour | ~4 min |
| Choix des vues | ~40 s |
| Argiles + forme Pixal3D + texture Pixal3D | ~16 min |
| Directeur | ~3 min |
| Marche (allure, vidéo Wan, contrôles, R-DMesh, GIF) | ~8 min (+4 min si la 1re graine est rejetée) |
| Boucle + export avec cuisson | ~1 min |

⚠ **Ne lancez pas de scène 3D (Godot, jeu…) pendant une fabrication** : Qwen-Image (8,6 Go) et Pixal3D débordent
alors en mémoire partagée et deviennent jusqu'à 7 fois plus lents (essai du 2026-10-11 : 1 h 31 au lieu de ~45 min).

## 7. État, résultats et limites connues

| Étape | État |
|---|---|
| Concept Qwen-Image-2.1 + choix automatique | ✅ (prompts en phrases complètes > mots-clés) ; le juge VLM rejette les comptes faux (Cerbère à 6 pattes) ; entre candidats proches, il prend la silhouette la plus écartée |
| Vidéo de tour + choix des vues | ✅ (le VLM se trompe sur les corps en étoile : vues interpolées par défaut) |
| 3D + texture Pixal3D | ✅ sur humanoïde, araignée, loup, limace, Godzilla, personnage avec objet ; Godzilla déborde de VRAM (18 Go partagés) |
| Directeur | ✅, mais il regroupe les pattes d'araignée par 2 |
| Marche (profil + `geo_16k` pivoté + 32 img/s + R-DMesh bf16) | ✅ recette validée le 2026-10-06 (Siren Head, loup, dragon, Cerbère, Centaure, Godzilla) ; limite : de profil, les parties cachées (2 des 4 bras de Goro) peuvent disparaître dans la vidéo Wan sans être détectées |
| Export moteur (glb à cibles de morphing + cuisson) | ✅ testé dans Godot 4.6 (environ 16 000 triangles, textures 2048 px) |
| Boucle de marche | ✅ cycle de 1,5 à 1,9 s ; R-DMesh ne lit que 2 s de vidéo (~1 cycle), d'où un écart de jonction de 2 à 6 % de la taille, réparti sur le cycle |
| Chaîne complète `fabrique_monstre.sh` | ✅ deux monstres neufs fabriqués sans intervention (pieuvre à épée, Pennywise) ; un plantage ponctuel de SDXL (argiles) a été réglé par simple relance |
| Contrôles de marche | ⚠ test de périodicité approximatif (rapporté, hors décision) |
| Autres actions (repos, attaque…), sons, code de comportement | à faire (voir `VISION.md`) |

Essais **écartés**, à ne pas refaire sans raison nouvelle :

- maillage non pivoté (R-DMesh voit de trois-quarts et attribue les pas des pattes arrière à une patte
  avant) ;
- caméra en orbite pendant l'action (R-DMesh suit jusqu'à 90° puis déforme le maillage) ;
- caméra fixe de trois-quarts (le sujet s'approche de la caméra : tête écrasée, pattes croisées) ;
- multi-vues TRELLIS.2 ;
- « config max » TRELLIS.2 (facettes) ;
- Hunyuan3D (licence, et moins bon en détail) ;
- **deux vues** (écran partagé profil + face dans une même vidéo Wan, puis fusion ou ajustement multi-vues) : les
  deux moitiés ne font pas exactement le même mouvement → membres qui changent de forme ;
- **refilmer la vidéo sous un autre angle** : SV4D 2.0 (profils flous, anatomie abîmée), ReCamMaster (ne sait pas
  tourner de 90°), AnyView (invente un décor, hors de son domaine) ;
- **marche filmée de face** + contrôle d'anatomie + Motion 3-to-4 : de face, l'amplitude des pas est presque
  invisible → pas timides ; le profil reste meilleur (verdict à l'œil du 2026-10-11) ;
- allure « ample » (gros mouvements de chaque partie) : Wan perd ou fusionne des membres ;
- rig + animation de squelette (Puppeteer, MoCapAnything, UniMate) : abandonnés au profit de l'animation par
  sommets de R-DMesh (scripts retirés le 2026-10-11, voir l'historique git) ;
- moins d'étapes de calcul (Pixal3D, concepts) pour gagner du temps : 2-3 min gagnées, détail visiblement perdu.

## 8. Guide : dériver une pipeline « meubles »

Un meuble est **le cas facile** de cette pipeline : il ne bouge pas tout seul et n'a pas
d'anatomie. Ses seules animations sont des pièces qui pivotent (porte, couvercle) ou qui glissent
(tiroir). En revanche, il a besoin d'un **code d'interaction** (s'ouvrir, stocker des objets,
s'allumer…). Voici quoi garder, quoi retirer et quoi ajouter.

### 8.1 Ce qu'on garde tel quel

| Étape | Pourquoi ça marche aussi pour un meuble |
|---|---|
| Fiche JSON + concept Qwen | adapter le prompt : « product shot of a single piece of furniture, three-quarter front view, doors and drawers closed, plain light grey background » |
| Vidéo de tour + LoRA orbite | un objet **immobile** est le cas idéal pour la LoRA orbite (elle a été entraînée sur des sujets immobiles) |
| Choix des vues, argile A2, Pixal3D 4 vues + texture | aucune hypothèse anatomique dedans |
| `t2_simplify.py` | les meubles supportent bien des budgets de polygones bas |

### 8.2 Ce qu'on retire

- Tout l'étage animation par vidéo : `chaine_marche.sh`, `boucle_marche.py`, R-DMesh, Wan pour
  l'action. **C'est la moitié du temps de calcul en moins.**
- (le rig de squelette a déjà été retiré de la pipeline)

### 8.3 Ce qu'on ajoute

1. **Un directeur « fonctionnel »** dérivé de `directeur2.py`. On garde la mécanique (le VLM pose des
   cadres sur 4 vues à caméras connues, et chaque région 3D en découle : « sculpture par cadres »).
   On change seulement les questions :
   - passe 1 : « liste les pièces **mobiles** (porte, tiroir, couvercle, abattant, roue) » ;
   - passe 2 : pour chaque pièce, le **type de mouvement** (rotation ou translation) et le côté de la
     charnière.

   Le rapport donne la région 3D de chaque pièce. L'**axe** se calcule en géométrie : arête de la
   boîte de la porte du côté de la charnière, direction de sortie du tiroir = normale de sa face
   avant (+Z).
2. **Séparer les pièces mobiles** du maillage. Le modèle sort d'un bloc, comme les bras soudés des
   monstres, donc deux options :
   - découper le maillage selon la région 3D du rapport, puis boucher les trous ;
   - ou faire générer le meuble **ouvert** (la même règle de pose que les membres séparés : « doors
     slightly open, drawer slightly pulled out ») pour que la 3D laisse un vide visible entre les
     pièces.
3. **Animations procédurales dans Blender** (nouveau script, par exemple `bl_anim_pieces.py`) : chaque
   pièce devient un os rigide (ou un objet enfant). Les clips `open` / `close` sont des rotations ou
   translations calculées depuis l'axe et l'amplitude du rapport (ex. porte 0 → 100°, tiroir 0 → 80 %
   de sa profondeur). Export glTF avec clips nommés. Pas d'IA ici : c'est plus fiable et instantané.
4. **Génération du code d'interaction** (nouveau script, par exemple `generer_code.py`). Un LLM reçoit
   dans son prompt système :
   - l'API de script du jeu ;
   - les règles (capacité de stockage maximale par taille, etc.) ;
   - 2-3 exemples ;
   - la fiche et le rapport : quelles pièces, quels clips existent.

   Il écrit un script (`on_interact` → `play("open")`, inventaire de 6 cases, etc.) et un
   `manifeste.json`.
5. **Harnais de validation** (même principe que pour les monstres, voir `VISION.md` §6) :
   - le script ne cite que des fonctions et des clips qui existent ;
   - il passe un test en simulation sans affichage (ouvrir, fermer, ranger, retirer) ;
   - sinon l'erreur est renvoyée au LLM (2-3 essais), puis on bascule sur un **comportement
     générique** (« meuble décoratif » ou « coffre simple »).

### 8.4 Squelette de chaîne proposé

```bash
# bench/chaine_meuble.sh <id> <cand.png>   (à créer)
bash bench/chaine_complete.sh $ID $REF        # inchangé : tour -> vues -> 3D -> texture
$PT bench/directeur_meuble.py ${ID#v3_}                         # pièces mobiles + type de mouvement (dérivé de directeur2.py)
$PY bench/pieces_axes.py $ID                                    # régions -> pièces séparées + axes/amplitudes
"$BL" -b --python bench/bl_anim_pieces.py -- $O/tex_p3d.glb $O/pieces.json $O/meuble_anime.glb
$PY bench/generer_code.py $ID                                   # LLM -> comportement.lua + manifeste.json
$PY bench/valider_code.py $ID                                   # harnais : lint + simulation ; secours sinon
```

Fiche d'exemple :

```json
{
  "id": "meuble_armoire01",
  "concept": "Product shot of a single tall oak wardrobe for a dark fantasy video game, two doors slightly open with a visible gap, one drawer at the bottom slightly pulled out, iron hinges and handles. Three-quarter front view, the whole object visible, centered. Plain uniform light grey studio background, no floor, no shadow, soft even lighting.",
  "desc_courte": "tall dark oak wardrobe with two doors and one bottom drawer, iron fittings",
  "fonction": "rangement : 12 emplacements, s'ouvre au contact du joueur"
}
```

Le champ `fonction` est nouveau : c'est la consigne donnée au LLM qui écrira le code.

## 9. Règles de travail et pièges

**Règles**

- **Une variable à la fois**, et **un seul calcul GPU à la fois** (10 Go de VRAM).
- **Rien de codé pour un type précis** (pas de « si araignée »). Si une règle dépend du corps, le VLM ou
  le LLM la déduit du rapport.
- **Regarder le rendu** avant de croire un chiffre. Chaque nouvel indicateur doit être validé sur des
  cas vus à l'œil.
- Les chaînes vérifient chaque étape (`echec "..."` + `exit 1`). Gardez ce garde-fou : tuer un
  sous-processus fait sinon passer le script à l'étape suivante, et deux calculs GPU tournent alors en
  parallèle.

**Pièges déjà rencontrés**

- **Blender en mode batch** ignore les chemins de sortie relatifs pour les rendus : passez des chemins
  absolus (`$(pwd -W)/...`).
- **VRAM partagée** : les overlays (Epic, Overwolf, Discord) et les onglets de navigateur font déborder
  la VRAM. Un calcul passe alors de 3 s à 30 s par pas, et le débordement reste collé au processus.
- **RAM** : le serveur ComfyUI monte à environ 24 Go pendant le chargement de Wan 14B. Évitez Wan 14B
  sans Lightx2v (deux experts + CFG, la RAM sature).
- **Ne tuez jamais des processus par motif large** (ex. tous ceux dont la ligne de commande contient
  l'id du monstre) : l'orchestrateur porte les ids en argument. Tuez par PID.
- Le nœud `ComfyUI-Trellis2` a un **tiret** dans son nom : ajoutez son dossier au `sys.path` au lieu de
  l'importer comme paquet.
- Pixal3D télécharge **NAF** à l'exécution via `torch.hub` (invisible à un grep). C'est lui qui exige
  `natten`, d'où le Python 3.13 et les roues CUDA 13.

---

Licence : à définir par le propriétaire du dépôt. Les modèles tiers gardent leurs licences
respectives (à vérifier avant toute distribution).

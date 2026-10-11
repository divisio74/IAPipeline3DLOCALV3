#!/usr/bin/env bash
# chaine_marche.sh — MARCHE, recette validée le 2026-10-06 : UNE action, la marche, filmée DE PROFIL par une caméra FIXE.
#   rapport du directeur (doit exister) -> allure écrite par le LLM (nombre exact de chaque partie) -> rendu de départ
#   de profil (cadre pour N longueurs) -> vidéo Wan 14B -> masques GPU -> CONTRÔLES (avancée, zoom, cadre, périodicité ;
#   2 graines max) -> fond noir + carré par BANDES NOIRES (R-DMesh recadre au centre en carré : sans bandes, la créature
#   sortirait du cadre) -> 32 img/s -> R-DMesh sur geo_16k pivoté -> rendus + GIF.
# Essais écartés (voir README) : autres angles de caméra, marche sur place, allure ample, recadrage serré, départ en
# argile, perspective — leurs options ont été retirées le 2026-10-11 (historique git).
# Usage : bash bench/chaine_marche.sh <id ex. v3_quadrupede> [longueurs=2]
cd "$(dirname "$0")/.."
ID=$1; NL=${2:-2}; AZ=90; ACT=marche
R="$(pwd -W)"; O="$R/bench/out/$ID"; L="bench/out/${ID}_$ACT"
PY=./.venv/Scripts/python.exe; PX=./.venv-pixal3d/Scripts/python.exe; PT=./.venv-trellis2/Scripts/python.exe
PU="$R/.venv-unimate/Scripts/python.exe"
BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"; RD="$R/third_party/R-DMesh"
MOD="$(cd third_party/ComfyUI/models && pwd -W)"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1
echec() { echo "ECHEC $1"; exit 1; }
[ -f "$O/directeur2/rapport.txt" ] || echec "rapport du directeur absent (bash bench/directeur_seul.sh $ID)"

echo "######## $ID : allure (LLM)"
[ -f "$O/marche_allure.txt" ] || $PT bench/allure_marche.py $ID > "${L}_allure.log" 2>&1 || echec "allure"
ALLURE=$(cat "$O/marche_allure.txt"); echo "$ALLURE"

echo "######## $ID : rendu de départ, caméra de profil ($NL longueurs)"
"$BL" -b --python bench/bl_depart_marche.py -- "$O/tex_p3d.glb" "$O/depart_$ACT.png" 832 480 $NL - $AZ ortho 2>&1 | grep -a "depart-marche"
[ -f "$O/depart_$ACT.png" ] || echec "rendu de depart"
DESC=$($PY -c "import json;print(json.load(open('bench/jobs/$ID.json',encoding='utf-8'))['desc_courte'])")
NLT=$($PY -c "print({1:'one',1.5:'one and a half',2:'two',3:'three'}.get(float('$NL'),'$NL'))")
PROMPT="A 3D animation of $DESC, seen exactly from the side. $ALLURE It walks forward in the direction it faces, from the right part of \
the frame toward the left edge, covering about $NLT of its body lengths, with at least two full walk cycles. The camera is \
completely static: fixed side shot, no zoom, no camera movement, no panning, no tracking, the whole body stays \
visible. Plain light grey background."
echo "$PROMPT" > "$O/${ACT}_prompt.txt"

MEIL=""; MSC=-1
for SEED in 1 2; do
  V="$O/video_${ACT}_s$SEED"
  echo "######## $ID : vidéo de marche, graine $SEED"
  if [ ! -f "$V/frame_0064.png" ]; then
    $PX bench/run_wan_video.py "$O/depart_$ACT.png" "$V" "$PROMPT" --modele 14b --frames 81 --l 832 --h 480 \
      --seed $SEED --brut > "${L}_video_s$SEED.log" 2>&1 || echec "video graine $SEED"
  fi
  # masques BiRefNet sur GPU (~10 s au lieu de ~11 min sur CPU) ; controle_marche.py saute ceux qui existent
  $PT bench/masque_gpu.py "$V" --n 65 > "${L}_masques_s$SEED.log" 2>&1 || echec "masques graine $SEED"
  $PY bench/controle_marche.py "$V" > "${L}_controle_s$SEED.log" 2>&1 || echec "controle graine $SEED"
  tail -1 "${L}_controle_s$SEED.log"
  SC=$($PY -c "import json;r=json.load(open(r'$V/controle.json'));t=r.get('tests',{});print(sum(t.values())*100+max(0,min(99,int(10*r.get('mesures',{}).get('avancee_longueurs',0)))))")
  [ "$SC" -gt "$MSC" ] && { MSC=$SC; MEIL=$V; }
  $PY -c "import json,sys;sys.exit(0 if json.load(open(r'$V/controle.json'))['ok'] else 1)" && break
done
echo "######## $ID : vidéo retenue $MEIL (score $MSC)"
echo "$MEIL" > "$O/${ACT}_video_retenue.txt"

echo "######## $ID : fond noir + carré + R-DMesh"
$PY - << PYEOF || echec "fond noir"
import numpy as np
from pathlib import Path
from PIL import Image
S = Path(r"$MEIL"); D = S / "noir"
D.mkdir(exist_ok=True)
for i in range(65):
    a = np.asarray(Image.open(S / f"frame_{i:04d}.png").convert("RGB")).astype(np.float32) / 255
    m = np.asarray(Image.open(S / "masque" / f"frame_{i:04d}.png")) > 127
    # créature sombre -> relevée au gamma (R-DMesh exige un fond NOIR)
    if i == 0:
        med = float(np.median(a[m].mean(1))) if m.any() else 0.5
        gam = float(np.log(0.45) / np.log(max(med, 1e-3))) if med < 0.35 else 1.0
        print(f"[fond-noir] luminance mediane {med:.2f} -> gamma {gam:.2f}", flush=True)
    a = np.clip(a, 1e-4, 1) ** gam
    a[~m] = 0
    h, w = m.shape
    c = np.zeros((w, w, 3), np.float32)              # carré : bandes noires en haut et en bas
    c[(w - h) // 2:(w - h) // 2 + h] = a
    Image.fromarray((c * 255).astype(np.uint8)).save(D / f"frame_{i:04d}.png")
PYEOF
# Rythme : R-DMesh a appris sur des animations à 24-30 img/s, Wan sort du 16 img/s (2x plus de mouvement par image) :
# vidéo noire interpolée à 32 img/s (ffmpeg minterpolate) ; R-DMesh lit alors 2 s au lieu de 4.
# Verdict user 2026-10-06 (essai 5) : légèrement meilleur sur les pattes arrière.
mkdir -p "$MEIL/noir32"
ffmpeg -loglevel error -y -framerate 16 -i "$MEIL/noir/frame_%04d.png" \
  -vf "minterpolate=fps=32:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -start_number 0 "$MEIL/noir32/frame_%04d.png" \
  || echec "interpolation"
FPS=32
$PX -c "
import imageio.v2 as iio, glob
w=iio.get_writer(r'$RD/test_data/videos/${ID}_$ACT.mp4',fps=$FPS,quality=9)
for f in sorted(glob.glob(r'$MEIL/noir32/*.png'))[:65]: w.append_data(iio.imread(f))
w.close()" || echec mp4
# R-DMesh suppose une caméra DE FACE (il « redresse » le maillage vers la 1re image, mais seulement à moitié : essai 1
# du loup = vu de trois-quarts, pattes arrière attribuées à la patte avant gauche). La vidéo montre le FLANC GAUCHE,
# tête vers le bord gauche : on PIVOTE le maillage pour que sa vue de face soit déjà ce profil.
# Rotation de -AZ autour de la verticale (Blender Z = glTF Y) : la caméra d'azimut AZ devient la caméra de face.
# glTF : x' = x cos t + z sin t ; z' = -x sin t + z cos t, t = -AZ (AZ = 90 : (x, y, z) -> (-z, y, x)).
# Maillage <= 8192 sommets (limite des données d'entraînement de R-DMesh) : 16 000 faces -> ~7 700 sommets.
# Verdict user 2026-10-06 : meilleur que le geo_20k (~9 700 sommets).
[ -f "$O/geo_16k.glb" ] || $PT bench/t2_simplify.py "$O/p3d_mv.glb" "$O/geo_16k.glb" 16000 2>&1 | grep -a "sortie"
[ -f "$O/geo_16k.glb" ] || echec "maillage 16k"
GEO="$O/geo_16k_profil.glb"
$PY -c "
import numpy as np, trimesh
t = np.radians(-$AZ)
sc = trimesh.load(r'$O/geo_16k.glb', process=False)
M = np.eye(4); M[:3, :3] = np.round([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]], 12)
sc.apply_transform(M)
sc.export(r'$GEO')
m = trimesh.load(r'$GEO', force='mesh', process=False); print('[pivot] az $AZ, sommets', len(m.vertices))" || echec "maillage pivote"
cp "$GEO" "$RD/test_data/meshes/$ID.glb"
(cd "$RD" && RDMESH_BF16=1 RDMESH_SANS_RENDU=1 RDMESH_SANS_T5=1 RDMESH_VAE="$MOD/vae/wan2.2_vae.safetensors" RDMESH_WAN_GGUF="$MOD/unet/Wan2.2-TI2V-5B-Q8_0.gguf" \
  "$PU" test_drive.py --mesh_list $ID.glb --video_list ${ID}_$ACT.mp4 --rf_exp rdmeshdit --rf_epoch f \
  --num_hops 5 --alpha_hops 0.7 --num_traj 4096 --guidance_scale 1.5 \
  > "$R/${L}_rdmesh.log" 2>&1) || echec "R-DMesh (voir ${L}_rdmesh.log)"
# ^ réglages recommandés par le README des auteurs (défauts du script : 4 / 0.5 / ~V/8 / 1.0)
T="$RD/output_videos/rdmeshdit/f/rdmesh/${ID}_${ID}_${ACT}_trajs.pt"
[ -f "$T" ] || echec "trajectoires absentes"
"$PU" -c "
import torch, numpy as np
d=torch.load(r'$T',weights_only=False)
np.savez(r'$O/rdmesh_${ACT}_trajs.npz', **{f'obj{i}': t.numpy() for i,t in enumerate(d['trajs'])})" || echec npz

echo "######## $ID : rendus (profil = angle de la caméra, trois-quarts) + GIF"
TRAJ_RES=768x384 "$BL" -b --python bench/bl_rendu_trajs.py -- "$GEO" "$O/rdmesh_${ACT}_trajs.npz" \
  "$O/rdmesh_$ACT" 0,-45 2 2>&1 | grep -a "rendu-trajs"
$PX - << PYEOF || echec gif
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
O, V = Path(r"$O"), Path(r"$MEIL")
f = ImageFont.truetype("arial.ttf", 14)
ims = []
for i in range(0, 64, 2):
    c = Image.new("RGB", (480, 3 * 240 + 3 * 20), (40, 40, 40))
    d = ImageDraw.Draw(c)
    for k, (p, t) in enumerate(((V / f"frame_{i * 16 // $FPS:04d}.png", "video Wan (camera fixe, profil)"),
                                (O / "rdmesh_$ACT" / "az0" / f"f{i:03d}.png", "R-DMesh - angle de la camera"),
                                (O / "rdmesh_$ACT" / "az-45" / f"f{i:03d}.png", "R-DMesh - 45 deg plus loin"))):
        im = Image.open(p).convert("RGB"); im.thumbnail((480, 240))
        c.paste(im, ((480 - im.width) // 2, k * 260 + 20 + (240 - im.height) // 2))
        d.text((8, k * 260 + 2), t, fill="white", font=f)
    ims.append(c)
dst = O.parent / "animations_gif" / "${ACT}_$ID.gif"
ims[0].save(dst, save_all=True, append_images=ims[1:], duration=125, loop=0)
print(f"[gif] {dst}")
PYEOF
echo "######## $ID MARCHE TERMINEE"

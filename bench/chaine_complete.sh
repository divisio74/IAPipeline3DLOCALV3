#!/usr/bin/env bash
# chaine_complete.sh — du CONCEPT au MODÈLE 3D TEXTURÉ (2026-09-30, réduit à la 3D le 2026-10-11) :
#   vidéo de tour (Wan 2.2 14B) -> choix des vues (VLM : face/profil/dos + flux optique) ->
#   3D (argiles A2 -> Pixal3D 4 vues -> texture Pixal3D) -> tex_p3d.glb.
#   La suite (directeur, marche, boucle, export) est enchaînée par fabrique_monstre.sh. L'ancienne suite (rig Puppeteer,
#   UniMate) a été retirée (historique git).
# Prérequis : bench/jobs/<id>.json (concept, desc_courte) et bench/inputs/q21_<id>_nat/<cand.png> (concept de face).
# Usage : bash bench/chaine_complete.sh <id ex. v3_godzillav> <cand.png>
cd "$(dirname "$0")/.."
ID=$1; REF=$2
R="$(pwd -W)"; O="$R/bench/out/$ID"; L="bench/out/${ID}"
PY=./.venv/Scripts/python.exe; PT=./.venv-trellis2/Scripts/python.exe; PX=./.venv-pixal3d/Scripts/python.exe
BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1
set -o pipefail
echec() { echo "ECHEC $1"; exit 1; }
mkdir -p "$O"

echo "######## $ID : vidéo de tour"
DESC=$($PY -c "import json;print(json.load(open('bench/jobs/$ID.json',encoding='utf-8'))['desc_courte'])")
if [ ! -f "$O/video_tour/frame_0080.png" ]; then
# LoRA « orbite » (ostris/wan22_i2v_14b_orbit_shot_lora, déclencheur « orbit 360 ») : sans elle Wan ne fait pas le
# tour complet des corps allongés (limace : aller-retour vers la face ; loup : 2e tête) — essais du 2026-10-01
$PX bench/run_wan_video.py "bench/inputs/q21_${ID}_nat/$REF" "$O/video_tour" "orbit 360 around a 3D character render of $DESC. The \
character stays completely still in exactly the same pose, like a statue, while the camera smoothly orbits a full 360 \
degrees around it at constant speed and constant distance: we see its front, then its side, then its back, then its \
other side, then its front again. Turntable video, plain light grey studio background, soft even lighting." \
  --modele 14b --frames 81 --l 480 --h 832 --lora wan22_14b_i2v_orbit > "${L}_tour.log" 2>&1 || echec "video de tour"
fi
echo "######## $ID : choix des vues"
TOUR="$O/video_tour"
if ! $PT bench/choisir_vues.py "$TOUR" $ID > "${L}_vues.log" 2>&1; then
  grep -a ECHEC "${L}_vues.log"
  # Wan ne fait parfois qu'un DEMI-TOUR (Godzilla : face -> profil -> dos en 5 s). On exploite la continuité : la
  # SUITE du tour est générée à partir de la dernière image, puis les deux vidéos sont recollées (81 + 80 images).
  if [ ! -f "$O/video_tour_suite/frame_0080.png" ]; then
    echo "   tour incomplet -> suite generee depuis la derniere image"
    $PX bench/run_wan_video.py "$O/video_tour/frame_0080.png" "$O/video_tour_suite" "orbit 360 around a 3D character render of $DESC. \
The character stays completely still in exactly the same pose, like a statue, while the camera continues to smoothly \
orbit around it in the same direction at constant speed and constant distance, until it has gone all the way round: \
we see it from the angle shown, then its other side, then its back, then its front again. Turntable video, plain \
light grey studio background, soft even lighting." --modele 14b --frames 81 --l 480 --h 832 --brut --lora wan22_14b_i2v_orbit \
      > "${L}_tour_suite.log" 2>&1 || echec "suite de la video de tour"
  fi
  TOUR="$O/video_tour_complet"; rm -rf "$TOUR"; mkdir -p "$TOUR"
  for i in $(seq 0 80); do cp "$O/video_tour/frame_$(printf %04d $i).png" "$TOUR/frame_$(printf %04d $i).png"; done
  for i in $(seq 1 80); do cp "$O/video_tour_suite/frame_$(printf %04d $i).png" "$TOUR/frame_$(printf %04d $((80 + i))).png"; done
  $PT bench/choisir_vues.py "$TOUR" $ID > "${L}_vues.log" 2>&1 || { grep -a ECHEC "${L}_vues.log"; echec "choix des vues (tour complete)"; }
fi
grep -a "choix :" "${L}_vues.log"

echo "######## $ID : 3D"
bash bench/chaine_v3_3d.sh $ID $REF 7 7 7 > "${L}_chaine3d.log" 2>&1
[ -f "$O/geo_100k.glb" ] || echec "chaine 3D (voir ${L}_chaine3d.log)"
$PY bench/exp_vues_3d.py couleurs-job bench/jobs/$ID.json $REF 7 7 7 > /dev/null 2>&1 || echec "vues couleur"
$PX bench/run_pixal3d_tex.py bench/views_couleur/$ID "$O/geo_100k.glb" "$O/tex_p3d.glb" > "${L}_p3dtex.log" 2>&1 || echec texture

echo "######## $ID 3D TERMINEE"

#!/usr/bin/env bash
# fabrique_monstre.sh — DE LA FICHE AU MONSTRE ANIMÉ PRÊT POUR LE JEU, en une commande (2026-10-11).
#   fiche bench/jobs/<id>.json (écrite par le LLM « créateur » : concept, desc_courte, garde, vues)
#   1. concepts Qwen-Image-2.1 (plusieurs graines)        -> bench/inputs/q21_<id>_nat/cand_*.png
#   2. choix automatique du concept (choisir_concept.py) -> choix.json
#   3. vidéo de tour -> vues -> 3D -> texture (chaine_complete.sh)
#   4. directeur (membres, objets)                         -> directeur2/rapport.txt
#   5. marche, recette validée (chaine_marche.sh)          -> rdmesh_marche_trajs.npz + GIF
#   6. boucle de marche sans avancée (boucle_marche.py)    -> rdmesh_marche_boucle.npz/.json (vitesse)
#   7. export jeu : glb animé, couleur + normal map cuites -> bench/out/<id>/final/<id>_marche.glb
# REPRISE : chaque étape est sautée si son résultat existe déjà (relancer la commande après une erreur suffit).
# Usage : bash bench/fabrique_monstre.sh <id ex. v3_tentacules> [graines des concepts = "31337 31474 31611"]
cd "$(dirname "$0")/.."
ID=$1; GRAINES=${2:-"31337 31474 31611"}
R="$(pwd -W)"; O="$R/bench/out/$ID"; L="bench/out/${ID}_fabrique"; C="bench/inputs/q21_${ID}_nat"
PY=./.venv/Scripts/python.exe; PT=./.venv-trellis2/Scripts/python.exe
BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1
echec() { echo "ECHEC $1 ($(date +%H:%M))"; exit 1; }
etape() { echo "######## $ID : $1 ($(date +%H:%M))"; }
[ -f "bench/jobs/$ID.json" ] || echec "fiche bench/jobs/$ID.json absente"
mkdir -p "$O"

etape "1. concepts (graines $GRAINES)"
N=$(ls $C/cand_*.png 2>/dev/null | wc -l)
if [ "$N" -lt 1 ]; then
  for E in encode gen; do       # 2 processus : l'encodeur et le générateur ne tiennent pas ensemble dans 10 Go
    PYTHONPATH=.pylib-qwen21 $PY bench/qwen21_concepts.py --etape $E --job bench/jobs/$ID.json --seeds $GRAINES \
      >> "${L}_concepts.log" 2>&1 || echec "concepts ($E, voir ${L}_concepts.log)"
  done
fi
ls $C/cand_*.png > /dev/null 2>&1 || echec "aucun concept généré"

etape "2. choix du concept"
[ -f "$C/choix.json" ] || $PT bench/choisir_concept.py $ID > "${L}_choix.log" 2>&1 || echec "choix du concept"
REF=$($PY -c "import json;print(json.load(open(r'$C/choix.json',encoding='utf-8'))['retenu'])")
grep -a "\[concept\]" "${L}_choix.log" 2>/dev/null | tail -2; echo "concept retenu : $REF"

etape "3. vidéo de tour, vues, 3D, texture"
if [ ! -f "$O/tex_p3d.glb" ]; then
  bash bench/chaine_complete.sh $ID $REF > "${L}_3d.log" 2>&1 || echec "3D (voir ${L}_3d.log)"
fi
[ -f "$O/tex_p3d.glb" ] || echec "modèle texturé absent"

etape "4. directeur"
[ -f "$O/directeur2/rapport.txt" ] || bash bench/directeur_seul.sh $ID > "${L}_directeur.log" 2>&1 \
  || echec "directeur (voir ${L}_directeur.log)"
[ -f "$O/directeur2/rapport.txt" ] || echec "rapport du directeur absent"

etape "5. marche (recette validée)"
if [ ! -f "$O/rdmesh_marche_trajs.npz" ]; then
  bash bench/chaine_marche.sh $ID > "${L}_marche.log" 2>&1 || echec "marche (voir ${L}_marche.log)"
fi
[ -f "$O/rdmesh_marche_trajs.npz" ] || echec "trajectoires de marche absentes"

etape "6. boucle de marche"
[ -f "$O/rdmesh_marche_boucle.npz" ] || $PY bench/boucle_marche.py "$O/rdmesh_marche_trajs.npz" \
  "$O/rdmesh_marche_boucle.npz" 32 || echec "boucle"

etape "7. export jeu (glb animé, textures cuites)"
mkdir -p "$O/final"
if [ ! -f "$O/final/${ID}_marche.glb" ]; then
  "$BL" -b --python bench/bl_export_anim_glb.py -- "$O/geo_16k_profil.glb" "$O/rdmesh_marche_boucle.npz" - \
    "$O/final/${ID}_marche.glb" 90 32 marche "$O/tex_p3d.glb" 2048 > "${L}_export.log" 2>&1
  grep -a "\[cuisson\] alignement\|\[export-anim\]" "${L}_export.log"
fi
[ -f "$O/final/${ID}_marche.glb" ] || echec "export (voir ${L}_export.log)"
cp "$O/rdmesh_marche_boucle.json" "$O/final/${ID}_marche.json"
cp "$O/../animations_gif/marche_$ID.gif" "$O/final/" 2>/dev/null
echo "######## $ID MONSTRE TERMINÉ ($(date +%H:%M)) -> $O/final/"
ls "$O/final"

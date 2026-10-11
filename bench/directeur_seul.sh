#!/usr/bin/env bash
# directeur_seul.sh — relance SEULE l'étape directeur (rendus 4 vues + passes VLM + rapport) d'un modèle déjà fait.
# Usage : bash bench/directeur_seul.sh <id ex. v3_quadrupede> [id ...]
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"; PT=./.venv-trellis2/Scripts/python.exe
for ID in "$@"; do
  O="$(pwd -W)/bench/out/$ID"; M=${ID#v3_}; L=bench/out/${ID}
  echo "######## $ID : directeur"
  mkdir -p "$O/directeur2"
  "$BL" -b --python bench/bl_directeur.py -- "$O/tex_p3d.glb" "$O/directeur2/vue" face,gauche,dos,droite 448 > /dev/null 2>&1
  $PT bench/directeur2.py $M > "${L}_directeur2.log" 2>&1 || { echo "ECHEC directeur $ID"; exit 1; }
  $PT bench/directeur2_rapport.py $M > "${L}_rapport.log" 2>&1 || { echo "ECHEC rapport $ID"; exit 1; }
  cat "$O/directeur2/rapport.txt"
done
echo "######## DIRECTEUR TERMINE"

#!/usr/bin/env bash
# chaine_v3_3d.sh — de la fiche au modele texture (chaine v3, validee par le user le 2026-09-29) :
#   argile A2 des 4 vues -> PIXAL3D 4 vues (cameras) -> simplification 100k (CuMesh).
#   (texture : Pixal3D 4 vues, faite par chaine_complete.sh)
# Usage : bash bench/chaine_v3_3d.sh <id> <cand_x.png> <graine gauche> <graine droite> <graine dos>
# (les concepts et les vues Qwen doivent exister : qwen21_concepts.py --job ...)
cd "$(dirname "$0")/.."
ID=$1; REF=$2; G=$3; D=$4; B=$5
PY=./.venv/Scripts/python.exe; PP=./.venv-pixal3d/Scripts/python.exe; PT=./.venv-trellis2/Scripts/python.exe
BL="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"; O="$(pwd)/bench/out/$ID"
echo "######## $ID : argiles"
$PY bench/exp_vues_3d.py prep-job bench/jobs/$ID.json $REF $G $D $B > bench/out/${ID}_prep.log 2>&1 || { echo "ECHEC prep"; tail -5 bench/out/${ID}_prep.log; exit 1; }
grep -a "\[vues\]" bench/out/${ID}_prep.log
echo "######## $ID : Pixal3D 4 vues"
PYTHONIOENCODING=utf-8 $PP bench/run_pixal3d_mv.py bench/views/$ID "$O/p3d_mv.glb" > bench/out/${ID}_p3d.log 2>&1 || { echo "ECHEC pixal3d"; tail -5 bench/out/${ID}_p3d.log; exit 1; }
grep -a "\[p3d\] [0-9]" bench/out/${ID}_p3d.log
echo "######## $ID : simplification"
$PT bench/t2_simplify.py "$O/p3d_mv.glb" "$O/geo_100k.glb" 100000 2>&1 | grep -a "\[t2s\]"
# (2026-10-11) supprimés : texture TRELLIS.2 depuis le concept (tex.glb, ~1,4 min) et ses rendus de contrôle — la
# chaîne utilise la texture Pixal3D 4 vues (tex_p3d.glb, run_pixal3d_tex.py dans chaine_complete.sh). Pour la
# retrouver : bench/texture_t2.py dans l'historique git du dépôt.
echo "######## $ID TERMINE"

"""bl_depart_marche.py — IMAGE DE DÉPART du gabarit WALK (2026-10-01) : rendu ortho du modèle sous un angle de caméra
FIXE (défaut : profil gauche, créature tournée vers le bord GAUCHE), créature placée dans la partie DROITE d'un cadre
paysage assez large pour qu'elle puisse avancer de N longueurs de corps sans sortir. Même rendu que bl_directeur.py
(texture sans éclairage, fond gris).
Usage : blender -b --python bench/bl_depart_marche.py -- <in.glb> <out.png> [larg=832] [haut=480] [longueurs=2] [centre|-]
        [azimut=90] [ortho|persp]
« centre » : même échelle, mais créature CENTRÉE (vidéo en orbite).
azimut (convention bl_directeur) : 0 = face, 90 = profil gauche, 45 = trois-quarts avant gauche.
« persp » : caméra 50 mm légèrement plongeante, comme les rendus d'entraînement de R-DMesh (défaut : orthographique).
Écrit aussi <out>.json : cadrage (longueur projetée et hauteur du corps en pixels) pour les contrôles."""
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = argv[:2]
RX = int(argv[2]) if len(argv) > 2 else 832
RY = int(argv[3]) if len(argv) > 3 else 480
NL = float(argv[4]) if len(argv) > 4 else 2.0
CENTRE = len(argv) > 5 and argv[5] == "centre"
AZ = math.radians(float(argv[6]) if len(argv) > 6 else 90)
PERSP = len(argv) > 7 and argv[7] == "persp"
MARGE = 0.05                                     # marge de chaque côté, en fraction de la largeur
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
sc = bpy.context.scene
objs = [o for o in sc.objects if o.type == "MESH"]
P = np.concatenate([np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices]) for o in objs])
mn, mx = P.min(0), P.max(0)
c = (mn + mx) / 2
# caméra en (sin a, -cos a) autour du centre, regardant le centre ; l'axe « droite » de l'image vaut (cos a, sin a, 0)
# (a = 90 : caméra sur +X, +Y Blender à droite, avant de la créature = -Y Blender vers le bord GAUCHE)
droite = np.array([math.cos(AZ), math.sin(AZ), 0.0])
s = (P - c) @ droite
L, H = float(s.max() - s.min()), float(mx[2] - mn[2])   # longueur PROJETÉE sur l'image, hauteur
# largeur du cadre : départ collé à droite, arrivée collée à gauche après NL longueurs ; et assez haut pour le corps
W = max((NL + 1) * L / (1 - 2 * MARGE), 1.5 * H * RX / RY)
Hc = W * RY / RX
a_couleurs = any(o.data.color_attributes for o in objs)
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO" if a_couleurs else "FLAT"
sc.display.shading.color_type = "VERTEX" if a_couleurs else "TEXTURE"
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x, sc.render.resolution_y = RX, RY
sc.world = bpy.data.worlds.new("W"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.9, 1)
cd = bpy.data.cameras.new("C"); cd.type = "ORTHO"; cd.ortho_scale = W
cam = bpy.data.objects.new("C", cd); sc.collection.objects.link(cam); sc.camera = cam
o = 0.0 if CENTRE else float(s.max()) + MARGE * W - W / 2   # corps collé au bord DROIT (marge), ou centré
cz = float(mn[2]) - 0.12 * Hc + Hc / 2                     # pieds à 12 % du bas
cible = Vector((c[0] + droite[0] * o, c[1] + droite[1] * o, cz))
dist = 5 * float(max(mx - mn))
if PERSP:
    # comme les données d'entraînement de R-DMesh (data_curation/render_dmesh_single_file.py) : objectif 50 mm
    # (champ ~39,6°), caméra légèrement au-dessus (z + 0,2 x taille à 1,5 x la distance de cadrage -> ~5,5°).
    # La largeur W du cadre est respectée à la distance de la créature.
    cd.type = "PERSP"; cd.lens = 50; cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36
    dist = (W / 2) / math.tan(cd.angle_x / 2)
    elev = math.atan(0.2 / (1.5 / math.tan(math.radians(39.6) / 2) / 2))
    cam.location = cible + Vector((math.sin(AZ) * dist, -math.cos(AZ) * dist, dist * math.tan(elev)))
else:
    cam.location = cible + Vector((math.sin(AZ) * dist, -math.cos(AZ) * dist, 0))
cam.rotation_euler = (cible - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
px = RX / W
json.dump({"res": [RX, RY], "longueurs": NL, "azimut": math.degrees(AZ), "corps_l_px": L * px, "corps_h_px": H * px},
          open(OUT[:-4] + ".json", "w"))
print(f"[depart-marche] ok az={math.degrees(AZ):.0f} L={L:.3f} H={H:.3f} cadre={W:.3f} -> corps {L * px:.0f}x{H * px:.0f} px",
      flush=True)

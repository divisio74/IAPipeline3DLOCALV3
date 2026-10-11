"""bl_directeur.py — rendus pour le DIRECTEUR D'ANIMATION (VLM) : vues orthographiques d'un GLB.
Couleurs de sommets si le GLB en a (parties surlignees), sinon texture SANS eclairage.
Usage : blender -b --python bench/bl_directeur.py -- <in.glb> <prefixe_sortie> <vues ex. face,gauche,dos,34> [res]
Vues : face (az 0), gauche (az 90, creature tournee vers le bord gauche), dos (180), droite (270), 34 (-40)."""
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, PRE, VUES = argv[:3]
RES = int(argv[3]) if len(argv) > 3 else 448
AZ = {"face": 0, "gauche": 90, "dos": 180, "droite": 270, "34": -40}
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
sc = bpy.context.scene
objs = [o for o in sc.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
c, s = (mn + mx) / 2, max(mx - mn)
a_couleurs = any(o.data.color_attributes for o in objs)
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO" if a_couleurs else "FLAT"
sc.display.shading.color_type = "VERTEX" if a_couleurs else "TEXTURE"
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x = sc.render.resolution_y = RES
sc.world = bpy.data.worlds.new("W"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.9, 1)
cd = bpy.data.cameras.new("C"); cd.type = "ORTHO"; cd.ortho_scale = s * 1.1
cam = bpy.data.objects.new("C", cd); sc.collection.objects.link(cam); sc.camera = cam
for v in VUES.split(","):
    a = math.radians(AZ[v])
    cam.location = (c.x + math.sin(a) * 5 * s, c.y - math.cos(a) * 5 * s, c.z)
    cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{PRE}_{v}.png"
    bpy.ops.render.render(write_still=True)
# CAMERAS CONNUES (pour passer d'un cadre 2D du VLM a des positions 3D) : centre en coordonnees glTF (Y haut),
# echelle ortho, resolution. Blender (X, Y, Z) = glTF (x, -z, y).
import json
json.dump({"centre_gltf": [c.x, c.z, -c.y], "ortho": s * 1.1, "res": RES, "vues": VUES.split(",")},
          open(f"{PRE}_cameras.json", "w"))
print("[directeur-rendu] ok", flush=True)

"""bl_rendu_trajs.py — rend un maillage animé par TRAJECTOIRES DE SOMMETS (sortie de R-DMesh / AnimateAnyMesh) (2026-09-30).

Le module bpy 4.0 de .venv-unimate n'écrit ni MP4 ni images ici : les trajectoires sont exportées par test_drive.py
(patch « _trajs.pt » -> converti en .npz) et rendues avec le Blender installé.
Les trajectoires sont en coordonnées MONDE Blender, par objet, dans l'ordre des sommets à l'import (transforms appliqués).
Usage : blender -b --python bench/bl_rendu_trajs.py -- <maillage .glb/.fbx> <trajs.npz> <dossier_sortie> [azimuts ex. 0,90] [pas]
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, TRJ, OUT = argv[:3]
AZ = [float(a) for a in (argv[3] if len(argv) > 3 else "0").split(",")]
PAS = int(argv[4]) if len(argv) > 4 else 1
bpy.ops.wm.read_factory_settings(use_empty=True)
if SRC.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=SRC)
else:
    bpy.ops.import_scene.gltf(filepath=SRC)
sc = bpy.context.scene
objs = [o for o in sc.objects if o.type == "MESH"]
for o in sc.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = objs[0]
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in [o for o in sc.objects if o.type == "ARMATURE"]:
    bpy.data.objects.remove(o)
for o in objs:
    for m in list(o.modifiers):
        o.modifiers.remove(m)
d = np.load(TRJ)
trajs = [d[f"obj{i}"] for i in range(len(objs))]
T = trajs[0].shape[0]
for o, t in zip(objs, trajs):
    assert t.shape[1] == len(o.data.vertices), (o.name, t.shape, len(o.data.vertices))

tous = np.concatenate([t.reshape(-1, 3) for t in trajs])
mn, mx = tous.min(0), tous.max(0)
c, s = Vector(((mn + mx) / 2).tolist()), float((mx - mn).max())
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "TEXTURE"
sc.view_settings.view_transform = "Standard"
# TRAJ_RES=LxH (ex. 768x384) : cadre paysage pour une marche qui traverse l'image ; le cadre garde la hauteur entière
RX, RY = (int(v) for v in os.environ.get("TRAJ_RES", "384x384").split("x"))
sc.render.resolution_x, sc.render.resolution_y = RX, RY
sc.world = bpy.data.worlds.new("W")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.25, 0.25, 1)
cd = bpy.data.cameras.new("C")
cd.type = "ORTHO"
cd.ortho_scale = max(s, float(mx[2] - mn[2]) * RX / RY) * 1.1
cam = bpy.data.objects.new("C", cd)
sc.collection.objects.link(cam)
sc.camera = cam
for az in AZ:
    dst = os.path.join(OUT, f"az{int(az)}")
    os.makedirs(dst, exist_ok=True)
    a = math.radians(az)
    cam.location = (c.x + math.sin(a) * 5 * s, c.y - math.cos(a) * 5 * s, c.z + 0.3 * s)
    cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
    for f in range(0, T, PAS):
        for o, t in zip(objs, trajs):
            o.data.vertices.foreach_set("co", t[f].astype(np.float32).ravel())
            o.data.update()
        sc.render.filepath = os.path.join(dst, f"f{f:03d}.png")
        bpy.ops.render.render(write_still=True)
print(f"[rendu-trajs] {T} images x {len(AZ)} vues -> {OUT}", flush=True)

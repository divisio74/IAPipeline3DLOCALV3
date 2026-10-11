"""bl_export_anim_glb.py (2026-10-11) — exporte une animation par TRAJECTOIRES DE SOMMETS (R-DMesh / Motion 3-to-4) en
.glb jouable dans un moteur (Godot) : une clé de forme par image, animée image par image (cibles de morphing glTF),
couleurs de sommets prises sur le modèle texturé, maillage remis de face (dépivot) et posé au sol à l'origine.
Usage : blender -b --python bench/bl_export_anim_glb.py -- <maillage pivoté .glb> <trajs.npz> <couleurs.npy | -> <sortie.glb>
        [azimut du pivot = 90] [img/s = 32] [nom de l'animation = marche] [modèle texturé complet à cuire] [taille texture]"""
import math
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, TRJ, COL, OUT = argv[:4]
AZ = float(argv[4]) if len(argv) > 4 else 90.0
FPS = int(argv[5]) if len(argv) > 5 else 32
NOM = argv[6] if len(argv) > 6 else "marche"
# 8e argument (optionnel) : modèle texturé complet (tex_p3d.glb) -> cuisson couleur + normal map au lieu des couleurs
# de sommets ; 9e : taille des textures (2048)
HAUT = argv[7] if len(argv) > 7 else None
TEX = int(argv[8]) if len(argv) > 8 else 2048

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
sc = bpy.context.scene
objs = [o for o in sc.objects if o.type == "MESH"]
assert len(objs) == 1, [o.name for o in objs]
ob = objs[0]
for o in sc.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = ob
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in [o for o in sc.objects if o.type != "MESH"]:
    bpy.data.objects.remove(o)
me = ob.data
tr = np.load(TRJ)["obj0"].astype(np.float32)          # (T, N, 3) monde Blender
T, N = tr.shape[:2]
assert N == len(me.vertices), (N, len(me.vertices))

mat = bpy.data.materials.new("monstre")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.8
if HAUT is None:
    # couleurs de sommets
    col = np.load(COL).astype(np.float32)
    assert len(col) == N
    att = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    att.data.foreach_set("color", np.c_[col, np.ones(N, np.float32)].ravel())
    me.color_attributes.active_color = att
    ca = nt.nodes.new("ShaderNodeVertexColor")
    ca.layer_name = "Col"
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
    me.materials.clear()
    me.materials.append(mat)
else:
    # CUISSON (2026-10-11) : UV automatiques sur le maillage léger (l'ordre des sommets ne change pas : les UV sont
    # par coin de face), puis couleur + normal map cuites depuis le modèle texturé complet, dans la pose de REPOS
    # (avant les clés de forme). Le modèle complet est dans le repère glTF d'origine : on lui applique le même pivot
    # (-AZ autour de la verticale) que le maillage donné à l'animateur.
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.003)
    # rangement serré des îlots (le rangement de smart_project laissait ~80 % de la texture vide)
    try:
        bpy.ops.uv.pack_islands(shape_method="CONCAVE", rotate=True, margin=0.002)
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")
    avant = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=HAUT)
    hauts = [o for o in set(bpy.data.objects) - avant if o.type == "MESH"]
    for o in set(bpy.data.objects) - avant:
        o.select_set(o in hauts)
    bpy.context.view_layer.objects.active = hauts[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)   # transformations d'import d'abord
    from mathutils import Matrix
    for o in hauts:
        o.data.transform(Matrix.Rotation(math.radians(-AZ), 4, "Z"))
        o.data.update()
        # la passe DIFFUSE/COULEUR vaut couleur de base x (1 - métallique) : matériau PBR métallique (loup de Pixal3D)
        # -> cuisson presque noire. On cuit l'albédo pur : métallique forcé à 0 sur le modèle complet.
        for ms in o.material_slots:
            if ms.material and ms.material.node_tree:
                for nd in ms.material.node_tree.nodes:
                    if nd.type == "BSDF_PRINCIPLED":
                        for lk in list(nd.inputs["Metallic"].links):
                            ms.material.node_tree.links.remove(lk)
                        nd.inputs["Metallic"].default_value = 0.0
    lo = np.array([v.co for v in me.vertices]); diag = float(np.linalg.norm(lo.max(0) - lo.min(0)))
    hv = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in hauts])
    from mathutils import kdtree
    kd = kdtree.KDTree(len(hv))
    for i, p in enumerate(hv):
        kd.insert(p, i)
    kd.balance()
    dd = np.array([kd.find(p)[2] for p in lo[::7]]) / diag
    print(f"[cuisson] alignement léger -> complet : distance médiane {np.median(dd):.4f}, 95 % {np.percentile(dd, 95):.4f}"
          " (en diagonales)", flush=True)
    imgs = {}
    for nom, nc in (("couleur", False), ("normale", True)):
        im = bpy.data.images.new(f"{NOM}_{nom}", TEX, TEX, alpha=False, float_buffer=False)
        if nc:
            im.colorspace_settings.name = "Non-Color"
        imgs[nom] = im
    tc = nt.nodes.new("ShaderNodeTexImage"); tc.image = imgs["couleur"]
    tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = imgs["normale"]
    nm = nt.nodes.new("ShaderNodeNormalMap")
    me.materials.clear()
    me.materials.append(mat)
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 4
    sc.cycles.device = "CPU"
    sc.render.bake.margin = 8
    for o in bpy.data.objects:
        o.select_set(o in hauts or o == ob)
    bpy.context.view_layer.objects.active = ob
    for nom, kind, filt in (("couleur", "DIFFUSE", {"COLOR"}), ("normale", "NORMAL", None)):
        nt.nodes.active = tc if nom == "couleur" else tn
        kw = dict(type=kind, use_selected_to_active=True, cage_extrusion=0.01 * diag, max_ray_distance=0.03 * diag,
                  margin=8)
        if filt:
            kw["pass_filter"] = filt
        bpy.ops.object.bake(**kw)
        imgs[nom].filepath_raw = OUT.rsplit(".", 1)[0] + f"_{nom}.png"
        imgs[nom].file_format = "PNG"
        imgs[nom].save()
        imgs[nom].pack()
        print(f"[cuisson] {nom} cuite ({TEX} px)", flush=True)
    for o in hauts:
        bpy.data.objects.remove(o)
    nt.links.new(tc.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    sc.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in [e.identifier for e in
                                                               bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else sc.render.engine

# repos = image 0 ; une clé de forme par image suivante
me.vertices.foreach_set("co", tr[0].ravel())
ob.shape_key_add(name="Basis", from_mix=False)
for t in range(1, T):
    k = ob.shape_key_add(name=f"f{t:03d}", from_mix=False)
    k.data.foreach_set("co", tr[t].ravel())
# remettre de face (le maillage donné à l'animateur était pivoté de -AZ) puis poser au sol, centré à l'origine
ob.rotation_euler = (0, 0, math.radians(AZ))
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
v = np.array([ob.matrix_world @ p.co for p in me.vertices])
ob.location = (-(v[:, 0].min() + v[:, 0].max()) / 2, -(v[:, 1].min() + v[:, 1].max()) / 2, -v[:, 2].min())
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

# animation : la clé t vaut 1 à l'image t, 0 aux images voisines (interpolation linéaire entre images)
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 0, T - 1
kb = me.shape_keys.key_blocks
for t in range(1, T):
    k = kb[f"f{t:03d}"]
    for fr, val in ((t - 1, 0.0), (t, 1.0), (t + 1, 0.0)):
        if 0 <= fr <= T - 1:
            k.value = val
            k.keyframe_insert("value", frame=fr)
act = me.shape_keys.animation_data.action
act.name = NOM
for fc in getattr(act, "fcurves", []):
    for p in fc.keyframe_points:
        p.interpolation = "LINEAR"

opts = dict(filepath=OUT, export_format="GLB", export_animations=True, export_morph=True, export_morph_normal=False,
            export_force_sampling=True, export_frame_range=True)
if HAUT is not None:
    opts.update(export_tangents=True, export_morph_tangent=False)
try:
    bpy.ops.export_scene.gltf(**opts, export_vertex_color="ACTIVE" if HAUT is None else "NONE")
except TypeError:
    bpy.ops.export_scene.gltf(**opts)
print(f"[export-anim] {T} images, {N} sommets -> {OUT}")

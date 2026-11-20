"""Blender side of `exploded3d.render`: build the scene written by `build_scene` and render it.

Run as  blender -b --factory-startup -P render_exploded.py -- <scene_dir> <out.png>
The scene is also saved as <scene_dir>/scene.blend for inspection.
"""
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

scene_dir, out_png = map(Path, sys.argv[sys.argv.index("--") + 1:][:2])
spec = json.loads((scene_dir / "scene.json").read_text())
cfg = json.loads((scene_dir / "render.json").read_text())
arrays = np.load(scene_dir / "meshes.npz")

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
scene = bpy.context.scene


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, a)


def principled(name, rgba, roughness=0.55, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        for attr, value in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, value)
                except TypeError:
                    pass
    return mat


def image_material(name, path):
    """A flat, unlit picture; pixels with alpha below 1 are see-through."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path))
    tex.interpolation = "Cubic"
    emit = nt.nodes.new("ShaderNodeEmission")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(clear.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    for attr, value in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, value)
            except TypeError:
                pass
    return mat


alphas = spec.get("alphas", {})                              # per-material opacity overrides
materials = {name: principled(name, hex_rgba(c), alpha=alphas.get(name, spec.get("contact_alpha", 1.0)))
             for name, c in spec["colors"].items()}
# contact points stay opaque
point_materials = {name: principled(name + "_point", hex_rgba(c)) for name, c in spec["colors"].items()}
for name, c in spec["colors"].items():                       # contact traces: a darker shade
    r, g, bl, _ = hex_rgba(c)
    materials["trace" + name[len("contact"):]] = principled("trace" + name, (0.55 * r, 0.55 * g, 0.55 * bl, 1.0),
                                                             roughness=0.5)
for name, (c, a) in spec.get("ghosts", {}).items():         # see-through, paler parts of contacts
    materials[name] = principled(name, hex_rgba(c), alpha=a)

# optional looks for the contact surfaces (solid and ghost)
look = set(spec.get("look_materials", ()))                  # build_layers names its surface materials
surface_mats = [m for n, m in materials.items() if n.startswith(("contact", "ghost")) or n in look]
if spec.get("matte"):                                         # no plastic sheen: the colours read cleanly
    for mat in surface_mats:
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = 0.8
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
if spec.get("sheen"):                                         # brighter where a sheet turns away from the
    for mat in surface_mats:                                  # view, so closely stacked sheets separate
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Sheen Weight"].default_value = spec["sheen"]
        bsdf.inputs["Sheen Roughness"].default_value = 0.3
if spec.get("backface_darken"):                               # the underside of a sheet a shade darker,
    k = 1.0 - spec["backface_darken"]                         # so overturned limbs read as such
    for mat in surface_mats:
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["A"].default_value = tuple(bsdf.inputs["Base Color"].default_value)
        mix.inputs["B"].default_value = (k, k, k, 1.0)
        nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Backfacing"], mix.inputs["Factor"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
if spec.get("texture"):                                       # a fine rock grain: noise in world space
    g = spec["texture"]                                       # shades the colour a little and bumps the
    grain = spec.get("texture_size", 25.0) * spec.get("scale", 1e-3)   # normal, so sheets read as material
    for mat in surface_mats:
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        coord = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 1.0 / grain
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.6
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        shade = nt.nodes.new("ShaderNodeMapRange")            # noise 0.3-0.7 -> 1 - g/2 .. 1 + g/4
        shade.inputs["From Min"].default_value, shade.inputs["From Max"].default_value = 0.3, 0.7
        shade.inputs["To Min"].default_value, shade.inputs["To Max"].default_value = 1.0 - 0.5 * g, 1.0 + 0.25 * g
        nt.links.new(noise.outputs["Fac"], shade.inputs["Value"])
        base = bsdf.inputs["Base Color"]
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        if base.is_linked:
            nt.links.new(base.links[0].from_socket, mix.inputs["A"])
        else:
            mix.inputs["A"].default_value = tuple(base.default_value)
        nt.links.new(shade.outputs["Result"], mix.inputs["B"])
        nt.links.new(mix.outputs["Result"], base)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.35 * g
        bump.inputs["Distance"].default_value = 0.2 * grain
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
if spec.get("glass"):                                         # a pane of glass marking a section
    gl = spec["glass"]
    mat = principled("glass", hex_rgba(gl.get("color", "#DCE6EE")), roughness=0.08, alpha=gl.get("alpha", 0.12))
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Specular IOR Level"].default_value = 1.0     # bright reflections: reads as glass
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    materials["glass"] = mat
    materials["glassframe"] = principled("glassframe", hex_rgba(gl.get("frame", "#5A6670")), roughness=0.4,
                                         alpha=gl.get("frame_alpha", 0.9))

materials["hole"] = principled("hole", hex_rgba("#1f1f1f"), roughness=0.4)
materials["axis"] = principled("axis", hex_rgba(spec.get("axis_color", "#3a3a3a")), roughness=0.35)
ext_color = spec.get("extremum_color", "#AA3377")
materials["extremum"] = principled("extremum", hex_rgba(ext_color), roughness=0.35)
materials["shell"] = principled("shell", hex_rgba(ext_color), roughness=0.3, alpha=0.7)

def vertex_color_material(name, strength=0.55):
    """Photo-like surface: the mesh's vertex colours, lit by the sun plus some self-illumination."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = "col"
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    for key in ("Emission Color", "Emission"):
        if key in bsdf.inputs:
            nt.links.new(col.outputs["Color"], bsdf.inputs[key])
            break
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength
    return mat


materials["photo"] = vertex_color_material("photo")

sphere_mesh = bpy.data.meshes.new("sphere")
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
bm.to_mesh(sphere_mesh)
bm.free()
for p in sphere_mesh.polygons:
    p.use_smooth = True


def link(ob):
    scene.collection.objects.link(ob)
    return ob


for o in spec["objects"]:
    kind = o["type"]
    if kind == "mesh":
        v, f = arrays[f"{o['name']}_v"], arrays[f"{o['name']}_f"]
        me = bpy.data.meshes.new(o["name"])
        me.from_pydata(v.tolist(), [], f.tolist())
        me.validate()
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(materials[o["material"]])
        ob = link(bpy.data.objects.new(o["name"], me))
        levels = int(spec.get("subdiv") or 0)
        if levels and o["material"].startswith(("contact", "ghost")):
            # Render only: the mesh written to the scene stays the cut triangulation.
            sub = ob.modifiers.new("subdiv", "SUBSURF")
            sub.levels = sub.render_levels = levels
            sub.boundary_smooth = "PRESERVE_CORNERS"
        if spec.get("slab") and o["material"].startswith(("contact", "ghost")):
            # a thin slab instead of a sheet; its walls (where the sheet ends) in the darker trace shade
            wall = materials.get("trace" + o["material"][len("contact"):]) if o["material"].startswith("contact") else None
            if wall is not None:
                me.materials.append(wall)
            mod = ob.modifiers.new("slab", "SOLIDIFY")
            mod.thickness = spec["slab"] * spec.get("scale", 1e-3)
            mod.offset = 0.0                                    # centred on the surface
            mod.use_even_offset = True
            mod.use_quality_normals = True
            mod.material_offset_rim = 1 if wall is not None else 0
    elif kind == "colormesh":                    # a mesh with one RGB colour per vertex
        v, f, c = arrays[f"{o['name']}_v"], arrays[f"{o['name']}_f"], arrays[f"{o['name']}_c"]
        me = bpy.data.meshes.new(o["name"])
        me.from_pydata(v.tolist(), [], f.tolist())
        me.validate()
        for p in me.polygons:
            p.use_smooth = True
        attr = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)      # sRGB -> linear
        attr.data.foreach_set("color", np.c_[lin, np.ones(len(lin))].ravel().astype(np.float32))
        me.materials.append(materials["photo"])
        link(bpy.data.objects.new(o["name"], me))
    elif kind == "points":                       # a point cloud, one RGB colour per point
        v = arrays[f"{o['name']}_v"]
        c = (arrays[f"{o['name']}_c8"] / 255.0 if f"{o['name']}_c8" in arrays
             else arrays[f"{o['name']}_c"])
        me = bpy.data.meshes.new(o["name"])
        me.vertices.add(len(v))                  # foreach_set: millions of points without Python lists
        me.vertices.foreach_set("co", np.asarray(v, dtype=np.float32).ravel())
        attr = me.attributes.new("col", "FLOAT_COLOR", "POINT")
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)      # sRGB -> linear
        attr.data.foreach_set("color", np.c_[lin, np.ones(len(lin))].ravel().astype(np.float32))
        ob = link(bpy.data.objects.new(o["name"], me))
        if "pointcolor" not in materials:
            mat = bpy.data.materials.new("pointcolor")
            mat.use_nodes = True
            nt = mat.node_tree
            bsdf = nt.nodes["Principled BSDF"]
            att = nt.nodes.new("ShaderNodeAttribute")
            att.attribute_type = "GEOMETRY"
            att.attribute_name = "col"
            if o.get("unlit"):                   # the photograph's own colours: no light added
                bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                bsdf.inputs["Specular IOR Level"].default_value = 0.0
            else:
                nt.links.new(att.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 0.9
            for key in ("Emission Color", "Emission"):
                if key in bsdf.inputs:
                    nt.links.new(att.outputs["Color"], bsdf.inputs[key])
                    break
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0 if o.get("unlit") else o.get("emission", 0.6)
            materials["pointcolor"] = mat
        ng = bpy.data.node_groups.new(o["name"] + "_gn", "GeometryNodeTree")
        ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        n_in, n_out = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
        to_pts = ng.nodes.new("GeometryNodeMeshToPoints")
        to_pts.inputs["Radius"].default_value = o["radius"]
        set_mat = ng.nodes.new("GeometryNodeSetMaterial")
        set_mat.inputs["Material"].default_value = materials["pointcolor"]
        ng.links.new(n_in.outputs["Geometry"], to_pts.inputs["Mesh"])
        ng.links.new(to_pts.outputs["Points"], set_mat.inputs["Geometry"])
        ng.links.new(set_mat.outputs["Geometry"], n_out.inputs["Geometry"])
        mod = ob.modifiers.new("points", "NODES")
        mod.node_group = ng
    elif kind == "quad":
        me = bpy.data.meshes.new(o["name"])
        me.from_pydata(o["corners"], [], [[0, 1, 2, 3]])
        uv = me.uv_layers.new()
        for loop, xy in zip(me.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            uv.data[loop.index].uv = xy
        me.materials.append(image_material(o["name"], scene_dir / o["image"]))
        link(bpy.data.objects.new(o["name"], me))
    elif kind == "tube":
        cu = bpy.data.curves.new(o["name"], "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = o["radius"]
        cu.bevel_resolution = 4
        sp = cu.splines.new("POLY")
        sp.points.add(len(o["points"]) - 1)
        for pt, xyz in zip(sp.points, o["points"]):
            pt.co = (*xyz, 1.0)
        cu.materials.append(materials[o["material"]])
        link(bpy.data.objects.new(o["name"], cu))
    elif kind == "cone":                         # arrow head: base centre to tip
        base, tip = Vector(o["base"]), Vector(o["tip"])
        axis = tip - base
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=o["radius"], radius2=0.0, depth=axis.length)
        me = bpy.data.meshes.new(o["name"])
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(materials[o["material"]])
        ob = link(bpy.data.objects.new(o["name"], me))
        ob.location = (base + tip) / 2
        ob.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
    elif kind == "spheres":
        me = sphere_mesh.copy()                  # one mesh per material, shared by its spheres
        me.materials.append(point_materials.get(o["material"], materials[o["material"]]))
        for i, xyz in enumerate(o["points"]):
            ob = link(bpy.data.objects.new(f"{o['name']}_{i}", me))
            ob.location = xyz
            ob.scale = (o["radius"],) * 3

# camera
c = cfg["camera"]
cam_data = bpy.data.cameras.new("camera")
if "lens" in c:                                  # perspective (exploded3d.perspective_camera)
    cam_data.type = "PERSP"
    cam_data.lens = c["lens"]
    cam_data.sensor_width = c.get("sensor", 36.0)
    cam_data.sensor_fit = "AUTO"
else:
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = c["ortho_scale"]
cam_data.clip_start, cam_data.clip_end = 0.001, 100.0
cam = link(bpy.data.objects.new("camera", cam_data))
cam.location = c["location"]
cam.rotation_euler = (Vector(c["target"]) - Vector(c["location"])).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

# light and world
sun = link(bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")))
sun.data.energy = spec.get("sun_energy", 2.5)
sun.data.angle = np.radians(15)
sun.rotation_euler = (np.radians(40), np.radians(15), np.radians(-35))
if spec.get("sun_direction"):                               # towards the light, model frame
    sun.rotation_euler = (-Vector(spec["sun_direction"]).normalized()).to_track_quat("-Z", "Y").to_euler()
if spec.get("rim_light"):                                     # a second sun from behind the scene, towards
    to_cam = Vector(c["location"]) - Vector(c["target"])     # the camera and a little down: it catches the
    to_cam.normalize()                                        # edges of the sheets and lifts them off the
    travel = (to_cam + Vector((0.0, 0.0, -0.6))).normalized() # background
    rim = link(bpy.data.objects.new("rim", bpy.data.lights.new("rim", "SUN")))
    rim.data.energy = spec["rim_light"]
    rim.data.angle = np.radians(10)
    rim.rotation_euler = travel.to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = spec.get("world_strength", 0.9)
scene.world = world

# render settings
engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
want = cfg["engine"].upper()
if want == "CYCLES":
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg["samples"]
    scene.cycles.use_denoising = True
else:
    scene.render.engine = next(e for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT") if e in engines)
    scene.eevee.taa_render_samples = cfg["samples"]
    if spec.get("occlusion"):                                 # screen-space occlusion and soft sun shadows:
        e = scene.eevee                                       # the gaps between nested sheets darken
        e.use_raytracing = True
        e.ray_tracing_method = "SCREEN"
        e.use_fast_gi = True
        e.fast_gi_method = "AMBIENT_OCCLUSION_ONLY"
        e.fast_gi_distance = spec["occlusion"] * spec.get("scale", 1e-3)
        e.use_shadows = True
        e.fast_gi_ray_count, e.fast_gi_step_count = 4, 8      # more occlusion rays: less grain
        e.shadow_ray_count, e.shadow_step_count = 2, 12
        sun.data.use_shadow = True
        sun.data.use_shadow_jitter = False                    # jitter is grainy at these sample counts;
        sun.data.shadow_filter_radius = 3.0                   # soften the shadow edge by filtering instead
scene.render.resolution_x, scene.render.resolution_y = cfg["resolution"]
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.render.filepath = str(out_png)
bpy.ops.wm.save_as_mainfile(filepath=str(scene_dir / "scene.blend"))
bpy.ops.render.render(write_still=True)
print("rendered", out_png)

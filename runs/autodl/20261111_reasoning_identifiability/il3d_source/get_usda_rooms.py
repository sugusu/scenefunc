from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import json
import os
import random
import argparse
from tqdm import tqdm
import numpy as np



def load_scene_json(scene_json_path: str):
    with open(scene_json_path, 'r') as f:
        scene_json = json.load(f)
    return scene_json

def create_scene(scene_json: dict, asset_root_path: str, save_path: str):
    stage = Usd.Stage.CreateInMemory()
    root_prim = stage.DefinePrim("/Root")

    for i, obj in enumerate(scene_json['objects']):
        object_name = obj['object_name']
        object_name = object_name.replace("-", "_")
        object_name = object_name.replace(" ", "_")
        object_name = object_name.replace("'", "")
        object_name = object_name.replace("(", "_")
        object_name = object_name.replace(")", "_")
        object_name = "obj_" + object_name

        prim_path = f"/Root/{object_name}"
        prim = stage.DefinePrim(prim_path)

        asset_path = os.path.join(asset_root_path, obj['object_path'])
        prim.GetReferences().AddReference(asset_path)

        translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3)
        translate_attr.Set(Gf.Vec3d(
            obj['position'][0] * 100,
            obj['position'][1] * 100,
            obj['position'][2] * 100
        ))

        scale_attr = prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
        scale_attr.Set(Gf.Vec3f(
            obj['scale'][0],
            obj['scale'][1],
            obj['scale'][2]
        ))
        
        det = len(obj['rotation'])
        if det == 3:
            rotate_attr = prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
            rotate_attr.Set(Gf.Vec3d(
                obj['rotation'][0],
                obj['rotation'][1],
                obj['rotation'][2]
            ))

            xform_op_order = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
            xform_op_order.Set(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
        elif det == 4:
            rotate_attr = prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)
            rotate_attr.Set(Gf.Quatd(
                obj['rotation'][3],
                obj['rotation'][0],
                obj['rotation'][1],
                obj['rotation'][2]
            ))

            xform_op_order = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
            xform_op_order.Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
        else:
            raise NotImplementedError

    stage.SetDefaultPrim(root_prim)
    stage.GetRootLayer().Export(save_path)

    return stage

def read_mesh_attr(mesh):
    v = np.array(mesh['xyz']).astype(np.float64) * 100
    faces = np.array(mesh['faces']).astype(np.int8)
    mat = None
    vt = None
    try:
        uv_array = np.reshape(mesh['uv'], [-1, 2]).astype(np.float64)
    except:
        uv_array = None

    if uv_array is not None:
        if mesh['material'] is not None:
            mat = np.reshape(mesh['material'], [3, 3]).astype(np.float64)
            if mat is not None:
                uv_array = (mat[:2,:2] @ uv_array.T).T
        
        vt = []
        for uv in uv_array:
            vt.append([float(uv[0]), float(uv[1])])
    
    return v, faces, vt, mat

def create_floor(scene_json: dict, data_dir: str, savepath: str):
    stage = Usd.Stage.CreateNew(savepath)
    root_prim = stage.DefinePrim("/Root")
    stage.SetDefaultPrim(root_prim)
    materials_scope = stage.DefinePrim("/Root/Materials", "Scope")
    
    obj = scene_json['meshes'][0]
    object_name = obj['object_name']
    object_name = object_name.replace("/", "_")
    object_name = object_name.replace(" ", "_")
    object_name = object_name.replace("'", "")

    v, faces, vt, mat = read_mesh_attr(obj)
    floor_path = f"/Root/{object_name}"
    floor_mesh = UsdGeom.Mesh.Define(stage, floor_path)

    points = [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in v]
    floor_mesh.CreatePointsAttr(points)
    face_vertex_counts = [len(face) for face in faces]
    floor_mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    face_vertex_indices = [int(index) for face in faces for index in face]
    floor_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)

    if vt is not None and len(vt) > 0:
        tex_coords = UsdGeom.PrimvarsAPI(floor_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        tex_coords.Set([Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in vt])
    
    material_name = f"{object_name}_material"
    material_path = f"{materials_scope.GetPath()}/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    
    shader = UsdShade.Shader.Define(stage, f"{material_path}/shader")
    shader.SetSourceAsset("UsdPreviewSurface", "usdShade")
    
    if obj.get('texture') and os.path.exists(os.path.join(data_dir, "texture", obj["jid"], "texture.png")):
        texture_path = os.path.abspath(os.path.join(data_dir, "texture", obj["jid"], "texture.png"))
        tex_reader = UsdShade.Shader.Define(stage, f"{material_path}/tex_reader")
        tex_reader.SetSourceAsset("UsdUVTexture", "usdShade")
        tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3).ConnectTo(
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3)
        )
      
        UsdShade.ConnectableAPI(floor_mesh).ConnectToSource(
            "st", tex_reader.ConnectableAPI(), "st"
        )
    else:
        color = Gf.Vec3f(220/255, 220/255, 220/255)
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(color)
    
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    
    UsdShade.MaterialBindingAPI(floor_mesh).Bind(material)

    stage.Save()
    return stage


def create_walls(scene_json: dict, savepath: str, include_ceiling = True, wall_height = 3.0):
    floor_data = scene_json['meshes'][0]
    xyz = floor_data['xyz']
    scale = 100.0
    
    points = [Gf.Vec3f(
        float(coord[0]) * scale,
        float(coord[1]) * scale,
        float(coord[2]) * scale
    ) for coord in xyz]
    
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_z = min(p[2] for p in points)
    max_z = max(p[2] for p in points)
    floor_y = points[0][1]
    
    floor_length = max_x - min_x
    floor_width = max_z - min_z
    floor_center_x = (max_x + min_x) / 2
    floor_center_z = (max_z + min_z) / 2
    
    wall_height = wall_height * scale
    ceiling_y = floor_y + wall_height
    
    if os.path.exists(savepath):
        stage = Usd.Stage.Open(savepath)
    else:
        stage = Usd.Stage.CreateNew(savepath)
    
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        root_prim = stage.DefinePrim("/Root")
        stage.SetDefaultPrim(root_prim)
    
    materials_scope = stage.GetPrimAtPath("/Root/Materials")
    if not materials_scope:
        materials_scope = stage.DefinePrim("/Root/Materials", "Scope")
    
    wall_material_name = "wall_material"
    wall_material_path = f"{materials_scope.GetPath()}/{wall_material_name}"
    wall_material = UsdShade.Material.Define(stage, wall_material_path)
    
    wall_shader = UsdShade.Shader.Define(stage, f"{wall_material_path}/shader")
    wall_shader.SetSourceAsset("UsdPreviewSurface", "usdShade")
    wall_color = Gf.Vec3f(180/255, 180/255, 180/255)
    wall_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(wall_color)
    wall_material.CreateSurfaceOutput().ConnectToSource(wall_shader.ConnectableAPI(), "surface")
    
    if include_ceiling:
        ceiling_material_name = "ceiling_material"
        ceiling_material_path = f"{materials_scope.GetPath()}/{ceiling_material_name}"
        ceiling_material = UsdShade.Material.Define(stage, ceiling_material_path)
        
        ceiling_shader = UsdShade.Shader.Define(stage, f"{ceiling_material_path}/shader")
        ceiling_shader.SetSourceAsset("UsdPreviewSurface", "usdShade")
        ceiling_color = Gf.Vec3f(230/255, 230/255, 230/255)
        ceiling_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(ceiling_color)
        ceiling_material.CreateSurfaceOutput().ConnectToSource(ceiling_shader.ConnectableAPI(), "surface")
    
    walls = [
        {
            "name": "front_wall",
            "position": (floor_center_x, floor_y + wall_height/2, max_z),
            "size": (floor_length, wall_height)
        },
        {
            "name": "back_wall",
            "position": (floor_center_x, floor_y + wall_height/2, min_z),
            "size": (floor_length, wall_height)
        },
        {
            "name": "left_wall",
            "position": (min_x, floor_y + wall_height/2, floor_center_z),
            "size": (floor_width, wall_height)
        },
        {
            "name": "right_wall",
            "position": (max_x, floor_y + wall_height/2, floor_center_z),
            "size": (floor_width, wall_height)
        }
    ]
    
    for wall in walls:
        wall_name = wall["name"].replace("/", "_").replace(" ", "_")
        wall_path = f"/Root/{wall_name}"
        wall_mesh = UsdGeom.Mesh.Define(stage, wall_path)
        
        x, y, z = wall["position"]
        size_x, size_y = wall["size"]
        
        if wall["name"] in ["front_wall", "back_wall"]:
            vertices = [
                (x - size_x/2, y - size_y/2, z),
                (x + size_x/2, y - size_y/2, z),
                (x + size_x/2, y + size_y/2, z),
                (x - size_x/2, y + size_y/2, z)
            ]
        else:
            vertices = [
                (x, y - size_y/2, z - size_x/2),
                (x, y - size_y/2, z + size_x/2),
                (x, y + size_y/2, z + size_x/2),
                (x, y + size_y/2, z - size_x/2)
            ]
        
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        wall_mesh.CreatePointsAttr(points)
        wall_mesh.CreateFaceVertexCountsAttr([4])
        wall_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdShade.MaterialBindingAPI(wall_mesh).Bind(wall_material)
    
    if include_ceiling:
        ceiling_name = "ceiling".replace("/", "_").replace(" ", "_")
        ceiling_path = f"/Root/{ceiling_name}"
        ceiling_mesh = UsdGeom.Mesh.Define(stage, ceiling_path)
        
        ceiling_vertices = [
            (min_x, ceiling_y, min_z),
            (max_x, ceiling_y, min_z),
            (max_x, ceiling_y, max_z),
            (min_x, ceiling_y, max_z)
        ]
        
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in ceiling_vertices]
        ceiling_mesh.CreatePointsAttr(points)
        ceiling_mesh.CreateFaceVertexCountsAttr([4])
        ceiling_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdShade.MaterialBindingAPI(ceiling_mesh).Bind(ceiling_material)
    
    stage.Save()
    return stage




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_path', type=str, default="./layout", help="layout files folder")
    parser.add_argument('--save_path', type=str, default="./room", help="Output folder for scenes")
    parser.add_argument('--data_path', type=str, default="./", help="dateset root")

    args = parser.parse_args()
    path = args.layout_path
    res = args.save_path
    # The data_dir must be an absolute path!!!
    data_dir = os.path.join(os.getcwd(), args.data_path)

    files = os.listdir(path)

    os.makedirs(res, exist_ok=True)
    z_max = {}
    for file in tqdm(files, desc="Create Scene Objects"):
        file = file.split(".")[0]
        z_max[file] = 0
        os.makedirs(os.path.join(res, file), exist_ok=True)
        input_path = os.path.join(path, file + ".json")
        scene_json = load_scene_json(input_path)
        mate_data = scene_json['objects']
        for data in mate_data:
            if data['position'][1] + data['bbox'][1] > z_max[file]:
                z_max[file] = data['position'][1] + data['bbox'][1]
        output_path = os.path.join(res, file, "object.usda")
        create_scene({"objects": mate_data}, data_dir, output_path)

    for file in tqdm(files, desc="Create Scene Floors"):
        file = file.split(".")[0]
        wall_height = z_max[file]
        if wall_height <= 2:
            wall_height = 3
        if os.path.exists(os.path.join(res, file)):
            input_path = os.path.join(path, file + ".json")
            scene_json = load_scene_json(input_path)
            mate_data = scene_json['meshes']
            floor_path = os.path.join(res, file, "floor.usda")
            wall_path = os.path.join(res, file, "wall.usda")
            create_floor({"meshes": mate_data}, data_dir, floor_path)
            # create_walls({"meshes": mate_data}, wall_path, wall_height=wall_height)


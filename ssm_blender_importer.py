bl_info = {
    "name": "Import Startopia .SSM",
    "author": "Consalvo",
    "version": (0, 3),
    "blender": (2, 80, 0),
    "location": "File > Import > Startopia (.ssm)",
    "description": "Import Startopia SSM mesh files",
    "category": "Import-Export",
}

import bpy
import struct
import os
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator
from bpy.props import StringProperty
from mathutils import Euler
from mathutils import Vector
from mathutils import Matrix
from math import radians

class ImportSSMOperator(Operator, ImportHelper):
    bl_idname = "import_mesh.ssm"
    bl_label = "Import SSM"
    bl_options = {'UNDO'}

    filename_ext = ".ssm"
    filter_glob: StringProperty(
        default="*.ssm",
        options={'HIDDEN'},
    )

    importBonesShapes: bpy.props.BoolProperty(
        name="Import Bone Shapes",
        description="Excludes meshes that are called after a bone.",
        default=False
    )

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Scale factor applied to imported model.",
        default=1/254.0,
        min=0.0001,
        max=1000.0
    )
    
    def draw(self, context):
        layout = self.layout
    
        # Create a box for transform-related settings
        box = layout.box()
        box.label(text="Settings")
        box.prop(self, "importBonesShapes")
        box.prop(self, "scale")

    def execute(self, context):
        filepath = self.filepath

        def log(msg):
            print(msg)

        def read_null_terminated_string(data, offset):
            """Read a null-terminated string from data starting at offset"""
            start = offset
            while offset < len(data) and data[offset] != 0:
                offset += 1
            return data[start:offset].decode('utf-8', errors='ignore'), offset + 1

        try:
            with open(filepath, 'rb') as f:
                cur = f.read()

            log(f"[DEBUG] File size: {len(cur)} bytes!")

            cur_offset = 0

            SSM_objects = []
            meshes = dict( {'Scene Root': None} )
            armatures = dict()
            bones = dict( {-1: None} )

            while cur_offset + 1 <= len(cur):
                objectType = struct.unpack_from('<B', cur, cur_offset)[0]
                cur_offset += 1

                if objectType == 0x08:
                    log(f"[OBJ] Found object type {objectType} at offset {cur_offset:08X} (SSM_Unknown)")
                elif objectType == 0x00 or objectType == 0x04 or objectType == 0x05:
                    log(f"[OBJ] Found object type {objectType} at offset {cur_offset:08X} (SSM_Object)")
                    positions = []
                    pos_vert_map = dict( {-1: [0]} )
                    materials = []
                    joint_count = 0
                    joints = []
                    parent_joints = []
                    frame_rate = 60
                    frame_count = 0
                    frames = []
                    weights = []
                    takes = []
                    log(f"[OBJ] Found buffer type 0x00 at offset {cur_offset:08X} (SSM_PointArray)")
                    position_count = struct.unpack_from('<I', cur, cur_offset)[0]
                    log(f"[OBJ] Position count {position_count} {cur_offset:08X}")
                    cur_offset += 4

                    for i in range(position_count):
                        x, y, z = struct.unpack_from('<fff', cur, cur_offset)
                        positions.append((x * -self.scale, y * self.scale, z * -self.scale))
                        pos_vert_map.update( {i: []} )
                        cur_offset += 12

                    bufferType = struct.unpack_from('<B', cur, cur_offset)[0]
                    if bufferType == 0x01:
                        cur_offset += 1
                        log(f"[OBJ] Found buffer type 0x01 at offset {cur_offset:08X} (SSM_Animation)")
                        joint_count = struct.unpack_from('<I', cur, cur_offset)[0]
                        log(f"[OBJ] Bone count {joint_count} {cur_offset:08X}")
                        cur_offset += 4

                        # Joints
                        for i in range(joint_count):
                            joint_name, cur_offset = read_null_terminated_string(cur, cur_offset)
                            m00, m10, m20, m01, m11, m21, m02, m12, m22, m03, m13, m23 = struct.unpack_from('<ffffffffffff', cur, cur_offset)
                            cur_offset += 48
                            x, y, z = struct.unpack_from('<fff', cur, cur_offset)
                            cur_offset += 12
                            hierarchy_next_idx, parent_idx, child_idx = struct.unpack_from('<iii', cur, cur_offset)
                            cur_offset += 12

                            log(f"[OBJ] Joint {i}_{joint_name}_{parent_idx}_{child_idx} at {cur_offset:08X}")
                            joints.append({
                                'name': joint_name,
                                'transform': [[m00, m01, -m02, -m03*self.scale], [m10, m11, -m12, -m13*self.scale], [-m20, -m21, m22, m23*self.scale], [0, 0, 0, 1]],
                                'bbox': (x * self.scale, y * self.scale, z * self.scale),
                                'hierarchy_next_idx': hierarchy_next_idx,
                                'parent_idx': parent_idx,
                                'child_indices': []
                            })

                        for i in range(joint_count):
                            current_joint = joints[ i ]
                            parent_idx = current_joint['parent_idx']
                            if parent_idx != -1:
                                parent_joint = joints[parent_idx]
                                parent_joint['child_indices'].append( i )
                            else:
                                parent_joints.append( i )
                           
                        log( f"[DEBUG] {joints}" )

                        frame_count = struct.unpack_from('<I', cur, cur_offset)[0]
                        log(f"[OBJ] Frame count {frame_count} at {cur_offset:08X}")
                        cur_offset += 4

                        frame_rate = struct.unpack_from('<f', cur, cur_offset)
                        log(f"[OBJ] Frame rate {frame_rate} at {cur_offset:08X}")
                        cur_offset += 4

                        # Frames
                        for frame_index in range(frame_count):
                            joint_transforms = []
                            for joint_index in range(joint_count):
                                m00, m10, m20, m01, m11, m21, m02, m12, m22, m03, m13, m23 = struct.unpack_from('<ffffffffffff', cur, cur_offset)
                                cur_offset += 48
                                x, y, z = struct.unpack_from('<fff', cur, cur_offset)
                                cur_offset += 12
                                joint_transforms.append({
                                    'transform': [[m00, m01, -m02, -m03*self.scale], [m10, m11, -m12, -m13*self.scale], [-m20, -m21, m22, m23*self.scale], [0, 0, 0, 1]],
                                    'bbox': (x * self.scale, y * self.scale, z * self.scale)
                                })
                            frames.append( joint_transforms )

                        # Bone Weights
                        while struct.unpack_from('<i', cur, cur_offset)[0] >= 0:
                            joint_weights = []
                            while struct.unpack_from('<i', cur, cur_offset)[0] >= 0:
                                joint_index = struct.unpack_from('<I', cur, cur_offset)
                                weight = struct.unpack_from('<f', cur, cur_offset + 4)
                                joint_weights.append({
                                    'joint_index': joint_index,
                                    'weight': weight
                                })
                                cur_offset += 8
                            cur_offset += 4 # Marker
                            weights.append(joint_weights)
                        cur_offset += 4 # Marker -14

                        # Takes
                        while struct.unpack_from('<c', cur, cur_offset)[0].isalnum():
                            # Read name  
                            take_name, cur_offset = read_null_terminated_string(cur, cur_offset)
                            start_frame = struct.unpack_from('<I', cur, cur_offset)[0]
                            cur_offset += 4
                            log(f"[OBJ] Take {take_name} : {start_frame} {cur_offset:08X}")
                            takes.append({
                                'name': take_name,
                                'start_frame': start_frame
                            })
    
                    bufferType = struct.unpack_from('<B', cur, cur_offset)[0]
                    if bufferType == 0x00:
                        cur_offset += 1
                        name = ""
                        parent_name = 'Scene Root'
 
                        log(f"[OBJ] Found buffer type 0x00 at offset {cur_offset:08X} (SSM_Submesh)")
                        sub_mesh_count = struct.unpack_from('<H', cur, cur_offset)[0]
                        cur_offset += 2
                        flag0 = struct.unpack_from('<B', cur, cur_offset)[0]
                        cur_offset += 1
                        flag1 = struct.unpack_from('<B', cur, cur_offset)[0]
                        cur_offset += 1
    
                        for submeshIndex in range(sub_mesh_count):
                            texture_name = ""
                            material_name = ""
                            hasMaterial = struct.unpack_from('<B', cur, cur_offset)[0]
                            if hasMaterial == 0:
                                cur_offset += 1
                            else:
                                if hasMaterial == 1:
                                    cur_offset += 1
    
                                    # Read material name  
                                    material_name, cur_offset = read_null_terminated_string(cur, cur_offset)
                                else:
                                    # Read texture filename
                                    texture_name, cur_offset = read_null_terminated_string(cur, cur_offset)
                                    # Read material name  
                                    material_name, cur_offset = read_null_terminated_string(cur, cur_offset)

                                if struct.unpack_from('c', cur, cur_offset)[0].isalnum():
                                    # Read reflection name  
                                    reflection_name, cur_offset = read_null_terminated_string(cur, cur_offset)
                                    # Skip flags (3 bytes based on pattern structure)
                                    cur_offset += 3
                                    
                                cur_offset += 1
                                flag4 = struct.unpack_from('<B', cur, cur_offset)[0]
                                cur_offset += 1
                                if flag4 == 1:
                                    cur_offset += 1
                                cur_offset += 4
    
                            vertexDataSize = struct.unpack_from('<I', cur, cur_offset)[0]
                            cur_offset += 4
    
                            # Read material buffers (position_index + UV + normal)
                            material_vertices = []
                            for i in range(vertexDataSize):
                                position_index = struct.unpack_from('<I', cur, cur_offset)[0]
                                uv_x, uv_y = struct.unpack_from('<ff', cur, cur_offset + 4)
                                normal_x, normal_y, normal_z = struct.unpack_from('<fff', cur, cur_offset + 12)
    
                                material_vertices.append({
                                    'position_index': position_index,
                                    'uv': (uv_x, 1.0-uv_y),
                                    'normal': (-normal_x, normal_y, -normal_z)
                                })
                                cur_offset += 24
    
                            # Read face count
                            face_count = struct.unpack_from('<I', cur, cur_offset)[0]
                            cur_offset += 4
                            log(f"[MATERIAL] Face count: {face_count}")
    
                            # Read faces
                            faces = []
                            for i in range(face_count):
                                # Read 3 vertex indices (24-bit each) + flags
                                face_indices = []
                                for j in range(3):
                                    # Read 24-bit index (3 bytes)
                                    idx_bytes = struct.unpack_from('<BBB', cur, cur_offset)
                                    vertex_idx = idx_bytes[0] + (idx_bytes[1] << 8) + (idx_bytes[2] << 16)
                                    flag = struct.unpack_from('<B', cur, cur_offset + 3)[0]
                                    face_indices.append(vertex_idx)
                                    cur_offset += 4
    
                                # Read face normal
                                face_normal = struct.unpack_from('<fff', cur, cur_offset)
                                cur_offset += 12
    
                                faces.append({
                                    'indices': face_indices,
                                    'normal': face_normal
                                })
    
                            log(f"[MATERIAL] Texture: {texture_name}, Material: {material_name}")
 
                            materials.append({
                                'texture_name': texture_name,
                                'material_name': material_name,
                                'vertices': material_vertices,
                                'faces': faces
                            })
    
                        # Read object name  
                        name, cur_offset = read_null_terminated_string(cur, cur_offset)
    
                        # Read parent name
                        if objectType == 0x00:
                            parent_name, cur_offset = read_null_terminated_string(cur, cur_offset)

                        model = {
                            'name': name,
                            'parent_name': parent_name,
                            'positions': positions,
                            'groups_count': sub_mesh_count,
                            'materials': materials,
                            'pos_vert_map': pos_vert_map
                        }

                        SSM_object = {
                            'name': name,
                            'model': model,
                            'joint_count': joint_count,
                            'joints': joints,
                            'weights': weights,
                            'parent_joints': parent_joints,
                            'frame_rate': frame_rate,
                            'frames': frames,
                            'takes': takes
                        }
                        SSM_objects.append( SSM_object )
                elif objectType == 0x01:
                    cur_offset += 1
                    log(f"[OBJ] Found object type 0x01 at offset {cur_offset - 2:08X} (SSM_Light)")

                    # Read object name  
                    name, cur_offset = read_null_terminated_string(cur, cur_offset)
    
                    # Read parent name  
                    parent_name, cur_offset = read_null_terminated_string(cur, cur_offset)
                        
                    x, y, z = struct.unpack_from('<fff', cur, cur_offset)
                    cur_offset += 12
                    x, y, z = struct.unpack_from('<fff', cur, cur_offset)
                    cur_offset += 12
                    x, y, z = struct.unpack_from('<fff', cur, cur_offset)
                    cur_offset += 12

                    number0 = struct.unpack_from('<I', cur, cur_offset)
                    cur_offset += 4
                else:
                    log(f"[OBJ] Unknown object type 0x{objectType:02X} at offset {cur_offset - 2:08X}, stopping.")
                    break
                
            
            main_object = None
            if self.importBonesShapes == False:
                for object in SSM_objects:
                    if object['joint_count'] > 0:
                        main_object = object
                        main_object['model']['parent_name'] = 'Scene Root'
                        break
            
            # Get the set of joint names for easy lookup (if main_object found)
            joints_names = set()
            if main_object and 'joints' in main_object:
                joints_names = {joint['name'] for joint in main_object['joints']}

            for object in SSM_objects:
                log(f"[Model] Name: {object['name']}, Parent: {object['model']['parent_name']}")

                if main_object and self.importBonesShapes == False:
                    # If any main_object['joints'] ... ['name'] equal object['name'] skip
                    if object['name'] in joints_names:
                        continue

                # Create Blender mesh
                if positions and materials:
                    self.create_blender_mesh( meshes, filepath, object['model'], object['joint_count'], object['joints'], object['weights'], log)
                    self.create_blender_armature( meshes, bones, armatures, object['model'], object['joint_count'], object['joints'], log)
                           
                    # Create animations if we have frame data
                    if object['frames'] and object['takes'] and object['joint_count'] > 0:
                        self.create_blender_animations( armatures, object['model'], object['joints'], object['parent_joints'], object['frames'], object['takes'], object['frame_rate'], log)
                    self.report({'INFO'}, f"Imported {len(object['model']['materials'])} submeshes with {len(object['model']['positions'])} positions")

        except Exception as e:
            self.report({'ERROR'}, f"Failed to import SSM: {e}")
            print(f"[ERROR] Exception: {e}")

        return {'FINISHED'}
    
    def create_blender_mesh(self, meshes, filepath, model, joint_count, joints, weights, log):
        """Create Blender mesh with materials and UV mapping"""
        
        mesh_name = model['name']
        parent_name = model['parent_name']
        positions = model['positions']
        materials = model['materials']
        pos_vert_map = model['pos_vert_map']
        
        faces = []
        face_material_map = {}

        vertices = []
        normals = []
        uvs = []

        # Create new mesh
        mesh = bpy.data.meshes.new(mesh_name + "_mesh")
            
        # Create object
        obj = bpy.data.objects.new(mesh_name, mesh)

        bpy.context.collection.objects.link(obj)
        
        obj.parent = meshes[ parent_name ]

        for mat_idx, material_data in enumerate(materials):
            face_material_map[mat_idx] = []

            # Get material vertices and create vertex mapping
            vertices_data = material_data['vertices']
            mat_vertex_offset = len( vertices )
            mat_vetex_count = 0
            
            for i, vertex_data in enumerate(vertices_data):
                position_index = vertex_data['position_index']
                if position_index < len(positions):
                    mat_vetex_count += 1
                    pos_vert_map[position_index].append( len(vertices) )
                    vertices.append(positions[position_index])
                    normals.append(vertex_data['normal'])
                    uvs.append(vertex_data['uv'])

            # Create faces
            for face_idx, face_data in enumerate(material_data['faces']):
                face_indices = []
                for idx in face_data['indices']:
                    if idx < mat_vetex_count:
                        face_indices.append(idx + mat_vertex_offset)
                
                if len(face_indices) == 3:
                    face_material_map[mat_idx].append(len(faces))
                    faces.append(face_indices)
            
            # Create material
            if material_data['material_name'] in bpy.data.materials:
                mat = bpy.data.materials[material_data['material_name']]
            else:
                mat = bpy.data.materials.new(name=material_data['material_name'])
            mat.use_nodes = True
            obj.data.materials.append(mat)
            
            # Try to load texture if it exists
            if len( material_data['texture_name'] ) > 0:
                texture_path = os.path.join(os.path.dirname(filepath), material_data['texture_name'])
                if os.path.exists(texture_path):
                    # Create texture node
                    bsdf = mat.node_tree.nodes["Principled BSDF"]
                    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
                    tex_image.image = bpy.data.images.load(texture_path)
                    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
                    mat.node_tree.links.new(bsdf.inputs['Alpha'], tex_image.outputs['Alpha'])
            
        log(f"[MESH] Created mesh '{mesh_name}' with {len(vertices)} vertices, {len(faces)} faces")

        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # Assign materials
        for mat_idx, material_data in enumerate(materials):
            log(f"[MAT] Faces for mat {mat_idx} : {face_material_map[mat_idx]}")
            for face_index in face_material_map[mat_idx]:
                obj.data.polygons[face_index].material_index = mat_idx

        # Set vertex normals
        if normals:
            mesh.normals_split_custom_set_from_vertices(normals)
            mesh.update()
            
        # Add UV coordinates
        if uvs and mesh.polygons:
            mesh.uv_layers.new(name="UVMap")
            uv_layer = mesh.uv_layers["UVMap"]
                
            for poly in mesh.polygons:
               for loop_idx in poly.loop_indices:
                    vertex_idx = mesh.loops[loop_idx].vertex_index
                    if vertex_idx < len(uvs):
                        uv_layer.data[loop_idx].uv = uvs[vertex_idx]

        log(f"[MESH] Created mesh uvs")
        
        # Create vertex groups for bones
        vertex_groups = dict( {int: bpy.types.VertexGroup} )
        if joint_count > 0:
            for joint_index, joint in enumerate(joints):
                vertex_groups.update( { joint_index: obj.vertex_groups.new(name=joint['name']) } )

        log(f"[MESH] Created bones vertex groups")

        # Apply bone weights
        if weights and joint_count > 0:
            for vertex_index, vertex_weights in enumerate(weights):
                if vertex_index < len(vertices):
                    for weight_data in vertex_weights:
                        joint_index = weight_data['joint_index'][0]
                        joint = joints[ joint_index ]
                        weight_value = weight_data['weight'][0]

                        vertex_group = vertex_groups[ joint_index ]
                        vertex_group.add(pos_vert_map[vertex_index], weight_value, 'ADD')
        
        log(f"[MESH] Assigned bone weights")
        meshes.update( { mesh_name: obj } )
    
    def create_blender_armature(self, meshes, bones, armatures, model, joint_count, joints, log):
        if joint_count == 0:
            return
        
        mesh_name = model['name']
        parent_name = model['parent_name']

        # Create armature
        armature = bpy.data.armatures.new(mesh_name + "_rig")
        # Create object
        obj = bpy.data.objects.new(mesh_name + "_armature", armature)
        obj.parent = meshes[ parent_name ]

        armature_modifier = meshes[ mesh_name ].modifiers.new(name="Armature", type='ARMATURE')
        armature_modifier.object = obj

        bpy.context.collection.objects.link( obj )
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode = 'EDIT')

        log(f"[ARMATURE] Created armature '{mesh_name}'")
        
        for joint_index, joint in enumerate( joints ):
            joint_name = joint['name']
            current_bone = armature.edit_bones.new( joint_name )
            log(f"[ARMATURE] Created bone '{joint_index}': '{joint_name}'")

            current_bone.use_local_location = False
            bones.update( {joint_index: current_bone} )

            if joint_name in meshes:
                joint_mesh = meshes[ joint_name ]
                armature_modifier = joint_mesh.modifiers.new(name="Armature", type='ARMATURE')
                armature_modifier.object = obj

                vertex_group = joint_mesh.vertex_groups.new(name=joint_name)
                vertex_group.add(list(range(len(joint_mesh.data.vertices))), 1, 'REPLACE')
        
        for joint_index, joint in enumerate( joints ):
            parent_idx = joint['parent_idx']
            parent_bone = bones[parent_idx]
            current_bone = bones[joint_index]
            child_indices = joint['child_indices']

            transform = Matrix( joint['transform'] )
            
            bbox = Vector(joint['bbox'])
            
            current_bone.length = bbox.magnitude
            current_bone.matrix = transform
            #current_bone.head = transform.to_translation()
            #current_bone.tail = ( (transform @ Matrix.Translation( transform.to_quaternion() @ Vector((0, 0, bbox.magnitude )) )) @ Vector((0, 0, 0, 1)) ).to_3d()

            current_bone.parent = parent_bone
            
        armatures.update( {mesh_name: obj } )
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def create_blender_animations(self, armatures, model, joints, parent_joints, frames, takes, frame_rate, log):
        """Create Blender animations from SSM frame data"""
        
        mesh_name = model['name']
        armature_obj = armatures[mesh_name]
        
        log(f"[ANIMATION] Creating animations for armature '{armature_obj.name}'")
        
        # Calculate frame ranges for each take
        sorted_takes = sorted(takes, key=lambda x: x['start_frame'])
        
        # Create action for each take
        for i, take in enumerate(sorted_takes):
            take_name = take['name']
            start_frame = take['start_frame']
            
            # Calculate end frame (start of next take or total frames)
            if i + 1 < len(sorted_takes):
                end_frame = sorted_takes[i + 1]['start_frame'] - 1
            else:
                end_frame = len(frames) - 1
            
            frame_count = end_frame - start_frame + 1
            
            # Create new action
            action = bpy.data.actions.new(name=f"{mesh_name}_{take_name}")
            armature_obj.animation_data_create()
            armature_obj.animation_data.action = action
            
            # Set Blender to pose mode
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='POSE')

            def recursive_set_anim_joint( self, model, joints, frames, frame_index, takes, frame_rate, log, joint_index ):
                joint = joints[joint_index]
                frame_data = frames[frame_index][joint_index]
                bone_transform = Matrix( joint['transform'] )
                parent_idx = joint['parent_idx']
                frame_transform = Matrix( frame_data['transform'] )
                parent_transform = Matrix.Identity(4)
                frame_parent_transform = Matrix.Identity(4)
                if parent_idx != -1:
                    parent_transform = Matrix( joints[parent_idx]['transform'] )
                    frame_parent_transform = Matrix( frames[frame_index][parent_idx]['transform'] )

                bone_name = joint['name']
                pose_bone = armature_obj.pose.bones.get(bone_name)
                
                if not pose_bone:
                    log(f"[ANIMATION] Failed to find bone {bone_name}")
                    return

                # Add keyframes for this take's frame
                if frame_index < len(frames):
                    frame_local = frame_parent_transform.inverted() @ frame_transform
                    delta_local = parent_transform.inverted() @ bone_transform
                    transform_delta = delta_local.inverted() @ frame_local
                    pose_bone.location = transform_delta.to_translation()
                    pose_bone.rotation_quaternion = transform_delta.to_quaternion()
                    
                    # Set Blender frame (relative to take start)
                    blender_frame = (frame_index - start_frame) * 5 + 1
                    pose_bone.keyframe_insert(data_path="location", frame=blender_frame)
                    pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
                    pose_bone.keyframe_insert(data_path="scale", frame=blender_frame)

                child_indices = joint['child_indices']
                for child_idx in child_indices:
                    recursive_set_anim_joint( self, model, joints, frames, frame_index, takes, frame_rate, log, child_idx )

            for frame_index in range(start_frame, min(end_frame + 1, len(frames))):
                for joint_index in parent_joints:
                    recursive_set_anim_joint( self, model, joints, frames, frame_index, takes, frame_rate, log, joint_index )
            
            # Set frame range for this animation
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = (frame_count - 1) * 5
            bpy.context.scene.render.fps = int(frame_rate[0])
            
            log(f"[ANIMATION] Created action '{action.name}' from frame {start_frame} to {end_frame} ({frame_count} frames)")
        
        bpy.ops.object.mode_set(mode='OBJECT')

def menu_func_import(self, context):
    self.layout.operator(ImportSSMOperator.bl_idname, text="Space Station Model (.ssm)")

def register():
    bpy.utils.register_class(ImportSSMOperator)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportSSMOperator)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()

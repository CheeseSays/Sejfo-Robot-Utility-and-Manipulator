# Interprets KRL a1 a2 a3 a4 a5 a6 commands and converts them to forward kinematics calculations
# using Denavit-Hartenberg parameters for a 6-DOF robotic arm.
# The output is the end-effector position and orientation in Cartesian coordinates, as well as joint angles.
# The animation will be applied to a 3D model hierarchy that looks like this:
# base_link
#  ├── link_1_joint
#  │    └── link_1
#  │         └── link_2_joint
#  │              └── link_2
#  │                   └── link_3_joint
#  │                        └── link_3
#  │                             └── link_4_joint
#  │                                  └── link_4
#  │                                       └── link_5_joint
#  │                                            └── link_5
#  │                                                 └── link_6_joint
#  │                                                      └── link_6

import bpy
import re
import math
import numpy as np
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, Operator
from bpy.path import abspath

# Regular expressions for parsing KRL joint angle commands
# Matches: A1 10.0 A2 20.0 A3 30.0 A4 0 A5 90 A6 0
JOINT_ANGLES_RE = re.compile(
    r"A(?P<joint>[1-6])\s+(?P<angle>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Matches named joint configurations: HOME={A1 0, A2 -90, A3 90, A4 0, A5 90, A6 0}
NAMED_JOINT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*\{(?P<block>[^}]*)\}",
    re.IGNORECASE,
)

# Matches PTP commands with joint angles
PTP_JOINT_RE = re.compile(
    r"^\s*PTP\s+(?P<name>[A-Za-z_]\w*)\b",
    re.IGNORECASE,
)

def parse_joint_angles(text: str) -> dict:
    """Parse joint angles from a KRL command string."""
    angles = {}
    for match in JOINT_ANGLES_RE.finditer(text):
        joint_num = int(match.group("joint"))
        angle = float(match.group("angle"))
        angles[f"A{joint_num}"] = angle
    
    # Ensure all 6 joints have values (default to 0 if not specified)
    for i in range(1, 7):
        angles.setdefault(f"A{i}", 0.0)
    
    return angles

def parse_fk_program(text: str) -> list:
    """Parse a KRL program for joint angle commands."""
    named_configs = {}
    commands = []
    
    lines = text.splitlines()
    for line in lines:
        line_wo_comments = line.split(';')[0].strip()
        if not line_wo_comments:
            continue
        
        # Named joint configuration
        named_match = NAMED_JOINT_RE.match(line_wo_comments)
        if named_match:
            name = named_match.group("name").upper()
            block = named_match.group("block")
            angles = parse_joint_angles(block)
            named_configs[name] = angles
            continue
        
        # PTP with reference to named configuration
        ptp_match = PTP_JOINT_RE.match(line_wo_comments)
        if ptp_match:
            name = ptp_match.group("name").upper()
            if name in named_configs:
                commands.append(named_configs[name])
            continue
        
        # Direct joint angle specification in PTP command
        if re.search(r"\bPTP\b", line_wo_comments, re.IGNORECASE):
            angles = parse_joint_angles(line_wo_comments)
            if any(v != 0.0 for v in angles.values()):
                commands.append(angles)
    
    return commands

def dh_transform(theta, d, a, alpha):
    """
    Create a Denavit-Hartenberg transformation matrix.
    
    Args:
        theta: Joint angle (rotation about z-axis)
        d: Link offset (translation along z-axis)
        a: Link length (translation along x-axis)
        alpha: Link twist (rotation about x-axis)
    
    Returns:
        4x4 transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def dh_transform_with_axis(theta, d, a, alpha, rotation_axis='Z'):
    """Create a DH transformation matrix with rotation around a specified axis.
    
    Args:
        theta: Joint angle
        d: Link offset (translation along z-axis)
        a: Link length (translation along x-axis)
        alpha: Link twist (rotation about x-axis)
        rotation_axis: 'X', 'Y', or 'Z' - axis of joint rotation
    
    Returns:
        4x4 transformation matrix
    """
    # Standard DH assumes Z-axis rotation
    if rotation_axis == 'Z':
        return dh_transform(theta, d, a, alpha)
    
    # For X or Y axis rotation, we need to modify the transformation
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    if rotation_axis == 'X':
        # Rotation around X-axis instead of Z
        # First do translation, then rotate around X, then apply twist
        T_trans = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])
        T_rot_x = np.array([
            [1,  0,   0,  0],
            [0,  ct, -st, 0],
            [0,  st,  ct, 0],
            [0,  0,   0,  1]
        ])
        T_twist = np.array([
            [1,  0,   0,  0],
            [0,  ca, -sa, 0],
            [0,  sa,  ca, 0],
            [0,  0,   0,  1]
        ])
        return T_trans @ T_rot_x @ T_twist
    
    elif rotation_axis == 'Y':
        # Rotation around Y-axis instead of Z
        T_trans = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])
        T_rot_y = np.array([
            [ct,  0, st, 0],
            [0,   1, 0,  0],
            [-st, 0, ct, 0],
            [0,   0, 0,  1]
        ])
        T_twist = np.array([
            [1,  0,   0,  0],
            [0,  ca, -sa, 0],
            [0,  sa,  ca, 0],
            [0,  0,   0,  1]
        ])
        return T_trans @ T_rot_y @ T_twist
    
    return dh_transform(theta, d, a, alpha)

def standard_6dof_dh_params():
    """
    Return standard DH parameters for a typical 6-DOF robot arm.
    These are example values and should be adjusted for specific robot models.
    
    Format: [(d, a, alpha), ...]
    theta is the variable joint angle
    """
    return [
        # Joint 1: (d1, a1, alpha1)
        (0.4, 0.025, np.pi/2),
        # Joint 2: (d2, a2, alpha2)
        (0.0, 0.315, 0.0),
        # Joint 3: (d3, a3, alpha3)
        (0.0, 0.035, np.pi/2),
        # Joint 4: (d4, a4, alpha4)
        (0.365, 0.0, -np.pi/2),
        # Joint 5: (d5, a5, alpha5)
        (0.0, 0.0, np.pi/2),
        # Joint 6: (d6, a6, alpha6)
        (0.095, 0.0, 0.0),
    ]

def forward_kinematics(joint_angles: list, dh_params: list = None, rotation_axes: list = None):
    """
    Calculate forward kinematics for a 6-DOF robot arm.
    
    Args:
        joint_angles: List of 6 joint angles in radians [a1, a2, a3, a4, a5, a6]
        dh_params: List of DH parameters [(d, a, alpha), ...]. If None, use standard params.
        rotation_axes: List of rotation axes ['X'/'Y'/'Z', ...]. If None, use 'Z' for all.
    
    Returns:
        List of 4x4 transformation matrices for each joint (cumulative transformations)
    """
    if dh_params is None:
        dh_params = standard_6dof_dh_params()
    
    if rotation_axes is None:
        rotation_axes = ['Z'] * 6
    
    transformations = []
    T = np.eye(4)  # Start with identity matrix
    
    for i, (theta, (d, a, alpha), axis) in enumerate(zip(joint_angles, dh_params, rotation_axes)):
        # Calculate transformation for this joint with specified rotation axis
        T_i = dh_transform_with_axis(theta, d, a, alpha, axis)
        T = T @ T_i
        transformations.append(T.copy())
    
    return transformations

def matrix_to_blender_transform(matrix):
    """
    Convert a 4x4 numpy transformation matrix to Blender location and rotation.
    
    Returns:
        (location, rotation_matrix) tuple
    """
    import mathutils
    
    # Extract location (translation part)
    location = mathutils.Vector((matrix[0, 3], matrix[1, 3], matrix[2, 3]))
    
    # Extract rotation (3x3 upper-left submatrix)
    rotation_matrix = mathutils.Matrix((
        (matrix[0, 0], matrix[0, 1], matrix[0, 2]),
        (matrix[1, 0], matrix[1, 1], matrix[1, 2]),
        (matrix[2, 0], matrix[2, 1], matrix[2, 2])
    ))
    
    return location, rotation_matrix

def calculate_dh_parameters_from_joints(base_link, joints, joint_axes):
    """Calculate DH parameters from the actual joint setup in the scene.
    
    Args:
        base_link: The base link object
        joints: List of 6 joint objects
        joint_axes: List of rotation axes for each joint
    
    Returns:
        List of (d, a, alpha) tuples for each joint
    """
    import mathutils
    
    dh_params = []
    
    # Start from base_link position
    prev_pos = base_link.matrix_world.translation if base_link else mathutils.Vector((0, 0, 0))
    prev_z = mathutils.Vector((0, 0, 1))  # Initial Z axis
    
    for i, (joint, axis) in enumerate(zip(joints, joint_axes)):
        if not joint:
            # Use default parameters if joint is not set
            dh_params.append((0.0, 0.0, 0.0))
            continue
        
        # Get joint position in world space
        joint_pos = joint.matrix_world.translation
        
        # Calculate link offset (d) - distance along previous Z axis
        offset_vec = joint_pos - prev_pos
        d = offset_vec.dot(prev_z)
        
        # Calculate link length (a) - distance in XY plane perpendicular to Z
        z_component = prev_z * d
        xy_component = offset_vec - z_component
        a = xy_component.length
        
        # Calculate link twist (alpha) - angle between Z axes
        # Get the joint's Z axis based on rotation axis setting
        joint_matrix = joint.matrix_world.to_3x3()
        if axis == 'X':
            current_z = joint_matrix @ mathutils.Vector((1, 0, 0))
        elif axis == 'Y':
            current_z = joint_matrix @ mathutils.Vector((0, 1, 0))
        else:  # Z
            current_z = joint_matrix @ mathutils.Vector((0, 0, 1))
        
        # Alpha is the angle between prev_z and current_z
        cos_alpha = prev_z.dot(current_z)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))  # Clamp to valid range
        alpha = math.acos(cos_alpha)
        
        dh_params.append((d, a, alpha))
        
        # Update for next iteration
        prev_pos = joint_pos
        prev_z = current_z
    
    return dh_params

def apply_fk_to_joint(joint_obj, transformation_matrix, base_link=None):
    """Apply a forward kinematics transformation matrix to a joint object.
    
    Args:
        joint_obj: The joint object to transform
        transformation_matrix: 4x4 numpy transformation matrix from FK
        base_link: Optional base link for relative positioning
    """
    import mathutils
    
    if not joint_obj:
        return
    
    # Convert numpy matrix to Blender matrix
    blender_matrix = mathutils.Matrix((
        (transformation_matrix[0, 0], transformation_matrix[0, 1], transformation_matrix[0, 2], transformation_matrix[0, 3]),
        (transformation_matrix[1, 0], transformation_matrix[1, 1], transformation_matrix[1, 2], transformation_matrix[1, 3]),
        (transformation_matrix[2, 0], transformation_matrix[2, 1], transformation_matrix[2, 2], transformation_matrix[2, 3]),
        (transformation_matrix[3, 0], transformation_matrix[3, 1], transformation_matrix[3, 2], transformation_matrix[3, 3])
    ))
    
    # Apply relative to base_link if provided
    if base_link:
        world_matrix = base_link.matrix_world @ blender_matrix
    else:
        world_matrix = blender_matrix
    
    # Apply to joint object
    joint_obj.matrix_world = world_matrix

def set_joint_rotation(joint_obj, angle_degrees, axis='Z'):
    """Set rotation on a joint object around the specified axis.
    
    Args:
        joint_obj: The joint object to rotate
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis ('X', 'Y', or 'Z')
    """
    if joint_obj and joint_obj.type == 'EMPTY':
        angle_rad = math.radians(angle_degrees)
        joint_obj.rotation_mode = 'XYZ'
        current_rotation = joint_obj.rotation_euler.copy()
        
        # Set rotation based on specified axis
        if axis == 'X':
            current_rotation.x = angle_rad
        elif axis == 'Y':
            current_rotation.y = angle_rad
        else:  # Default to Z
            current_rotation.z = angle_rad
        
        joint_obj.rotation_euler = current_rotation

def keyframe_joint(joint_obj, frame: int):
    """Insert keyframe for joint rotation."""
    if joint_obj:
        joint_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

class FKImporterSettings(PropertyGroup):
    filepath: StringProperty(
        name="File Path",
        description="Path to the KRL program file with joint angle commands (A1-A6)",
        default="",
        subtype='FILE_PATH'
    ) # type: ignore
    
    base_link: PointerProperty(
        name="Base Link",
        type=bpy.types.Object,
        description="Root object of the robot hierarchy (base_link)",
    ) # type: ignore
    
    joint_1: PointerProperty(
        name="Joint 1",
        type=bpy.types.Object,
        description="Link 1 joint object (rotates for A1)",
    ) # type: ignore
    
    joint_2: PointerProperty(
        name="Joint 2",
        type=bpy.types.Object,
        description="Link 2 joint object (rotates for A2)",
    ) # type: ignore
    
    joint_3: PointerProperty(
        name="Joint 3",
        type=bpy.types.Object,
        description="Link 3 joint object (rotates for A3)",
    ) # type: ignore
    
    joint_4: PointerProperty(
        name="Joint 4",
        type=bpy.types.Object,
        description="Link 4 joint object (rotates for A4)",
    ) # type: ignore
    
    joint_5: PointerProperty(
        name="Joint 5",
        type=bpy.types.Object,
        description="Link 5 joint object (rotates for A5)",
    ) # type: ignore
    
    joint_6: PointerProperty(
        name="Joint 6",
        type=bpy.types.Object,
        description="Link 6 joint object (rotates for A6)",
    ) # type: ignore
    
    joint_1_axis: EnumProperty(
        name="J1 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 1"
    ) # type: ignore
    
    joint_2_axis: EnumProperty(
        name="J2 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 2"
    ) # type: ignore
    
    joint_3_axis: EnumProperty(
        name="J3 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 3"
    ) # type: ignore
    
    joint_4_axis: EnumProperty(
        name="J4 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 4"
    ) # type: ignore
    
    joint_5_axis: EnumProperty(
        name="J5 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 5"
    ) # type: ignore
    
    joint_6_axis: EnumProperty(
        name="J6 Axis",
        items=[('X', 'X', 'Rotate around X axis'),
               ('Y', 'Y', 'Rotate around Y axis'),
               ('Z', 'Z', 'Rotate around Z axis')],
        default='Z',
        description="Rotation axis for Joint 6"
    ) # type: ignore
    
    start_frame: IntProperty(
        name="Start Frame",
        description="Frame to start inserting keyframes",
        default=1,
    ) # type: ignore
    
    frame_step: IntProperty(
        name="Frame Step",
        description="Number of frames between keyframes",
        default=25,
    ) # type: ignore
    
    create_action: BoolProperty(
        name="Create New Action",
        default=True,
        description="Create a new action for each joint",
    ) # type: ignore
    
    # DH Parameters (calculated from scene)
    dh_params_calculated: BoolProperty(
        name="DH Params Calculated",
        default=False,
        description="Whether DH parameters have been calculated from the scene",
    ) # type: ignore
    
    show_dh_params: BoolProperty(
        name="Show DH Parameters",
        default=False,
        description="Show the calculated DH parameters",
    ) # type: ignore
    
    tcp_object: PointerProperty(
        name="TCP Object (Optional)",
        type=bpy.types.Object,
        description="Object to animate with calculated TCP position from forward kinematics",
    ) # type: ignore
    
    animate_tcp: BoolProperty(
        name="Animate TCP",
        default=False,
        description="Calculate and animate TCP position using forward kinematics",
    ) # type: ignore
    
    # Store DH parameters as strings (Blender properties can't store tuples/lists directly)
    dh_param_d1: FloatProperty(name="d1", default=0.0) # type: ignore
    dh_param_a1: FloatProperty(name="a1", default=0.0) # type: ignore
    dh_param_alpha1: FloatProperty(name="alpha1", default=0.0) # type: ignore
    dh_param_d2: FloatProperty(name="d2", default=0.0) # type: ignore
    dh_param_a2: FloatProperty(name="a2", default=0.0) # type: ignore
    dh_param_alpha2: FloatProperty(name="alpha2", default=0.0) # type: ignore
    dh_param_d3: FloatProperty(name="d3", default=0.0) # type: ignore
    dh_param_a3: FloatProperty(name="a3", default=0.0) # type: ignore
    dh_param_alpha3: FloatProperty(name="alpha3", default=0.0) # type: ignore
    dh_param_d4: FloatProperty(name="d4", default=0.0) # type: ignore
    dh_param_a4: FloatProperty(name="a4", default=0.0) # type: ignore
    dh_param_alpha4: FloatProperty(name="alpha4", default=0.0) # type: ignore
    dh_param_d5: FloatProperty(name="d5", default=0.0) # type: ignore
    dh_param_a5: FloatProperty(name="a5", default=0.0) # type: ignore
    dh_param_alpha5: FloatProperty(name="alpha5", default=0.0) # type: ignore
    dh_param_d6: FloatProperty(name="d6", default=0.0) # type: ignore
    dh_param_a6: FloatProperty(name="a6", default=0.0) # type: ignore
    dh_param_alpha6: FloatProperty(name="alpha6", default=0.0) # type: ignore

class ANIM_OT_calculate_dh(Operator):
    bl_idname = "anim.calculate_dh"
    bl_label = "Calculate DH Parameters"
    bl_description = "Calculate Denavit-Hartenberg parameters from the current joint setup"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.fk_importer_settings
        
        # Collect joint objects and axes
        joints = [
            settings.joint_1,
            settings.joint_2,
            settings.joint_3,
            settings.joint_4,
            settings.joint_5,
            settings.joint_6,
        ]
        
        joint_axes = [
            settings.joint_1_axis,
            settings.joint_2_axis,
            settings.joint_3_axis,
            settings.joint_4_axis,
            settings.joint_5_axis,
            settings.joint_6_axis,
        ]
        
        # Check if all joints are set
        missing_joints = []
        for i, joint in enumerate(joints, 1):
            if not joint:
                missing_joints.append(f"Joint {i}")
        
        if missing_joints:
            self.report({'WARNING'}, f"Some joints not set: {', '.join(missing_joints)}. Using defaults for missing joints.")
        
        # Calculate DH parameters
        dh_params = calculate_dh_parameters_from_joints(settings.base_link, joints, joint_axes)
        
        # Store them in settings
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            setattr(settings, f"dh_param_d{i}", d)
            setattr(settings, f"dh_param_a{i}", a)
            setattr(settings, f"dh_param_alpha{i}", alpha)
        
        settings.dh_params_calculated = True
        
        # Store them in the scene for display (we'll store as a formatted string)
        params_str = "DH Parameters (d, a, alpha):\n"
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            params_str += f"Joint {i}: d={d:.4f}m, a={a:.4f}m, α={math.degrees(alpha):.2f}°\n"
        
        self.report({'INFO'}, f"DH parameters calculated from scene geometry")
        print(params_str)
        
        return {'FINISHED'}

class ANIM_OT_import_fk(Operator):
    bl_idname = "anim.import_fk"
    bl_label = "Import FK Program"
    bl_description = "Import a KRL program with joint angles and animate robot using forward kinematics"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        settings = context.scene.fk_importer_settings
        
        if not settings.filepath:
            self.report({'ERROR'}, "No file path specified")
            return {'CANCELLED'}
        
        # Collect joint objects and their rotation axes
        joints = [
            settings.joint_1,
            settings.joint_2,
            settings.joint_3,
            settings.joint_4,
            settings.joint_5,
            settings.joint_6,
        ]
        
        joint_axes = [
            settings.joint_1_axis,
            settings.joint_2_axis,
            settings.joint_3_axis,
            settings.joint_4_axis,
            settings.joint_5_axis,
            settings.joint_6_axis,
        ]
        
        # Verify all joints are set
        missing_joints = []
        for i, joint in enumerate(joints, 1):
            if not joint:
                missing_joints.append(f"Joint {i}")
        
        if missing_joints:
            self.report({'ERROR'}, f"Missing joint assignments: {', '.join(missing_joints)}")
            return {'CANCELLED'}
        
        # Read program file
        try:
            with open(abspath(settings.filepath), "r", encoding="utf-8", errors="ignore") as f:
                program_text = f.read()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read file: {e}")
            return {'CANCELLED'}
        
        # Parse joint angle commands
        commands = parse_fk_program(program_text)
        if not commands:
            self.report({'ERROR'}, "No valid joint angle commands found in the program")
            return {'CANCELLED'}
        
        # Calculate DH parameters from the scene
        dh_params = calculate_dh_parameters_from_joints(settings.base_link, joints, joint_axes)
        
        # Store them in settings
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            setattr(settings, f"dh_param_d{i}", d)
            setattr(settings, f"dh_param_a{i}", a)
            setattr(settings, f"dh_param_alpha{i}", alpha)
        
        settings.dh_params_calculated = True
        
        # Display calculated parameters
        print("\n=== Calculated DH Parameters ===")
        for i, (d, a, alpha) in enumerate(dh_params, 1):
            print(f"Joint {i}: d={d:.4f}m, a={a:.4f}m, α={math.degrees(alpha):.2f}°")
        print("================================\n")
        
        # Setup TCP animation if requested
        tcp_obj = None
        if settings.animate_tcp and settings.tcp_object:
            tcp_obj = settings.tcp_object
            if settings.create_action:
                tcp_obj.animation_data_create()
                action_name = bpy.path.basename(settings.filepath).replace('.src', '').replace('.krl', '').replace('.txt', '')
                tcp_obj.animation_data.action = bpy.data.actions.new(name=f"{action_name}_TCP")
        
        # Create actions if requested
        if settings.create_action:
            action_name = bpy.path.basename(settings.filepath).replace('.src', '').replace('.krl', '').replace('.txt', '')
            for i, joint in enumerate(joints, 1):
                if joint:
                    joint.animation_data_create()
                    joint.animation_data.action = bpy.data.actions.new(name=f"{action_name}_J{i}")
        
        # Animate joints
        frame = settings.start_frame
        
        for command in commands:
            # Extract joint angles
            angles = [
                command.get('A1', 0.0),
                command.get('A2', 0.0),
                command.get('A3', 0.0),
                command.get('A4', 0.0),
                command.get('A5', 0.0),
                command.get('A6', 0.0),
            ]
            
            # Convert angles to radians for FK calculation
            angles_rad = [math.radians(a) for a in angles]
            
            # Calculate forward kinematics transformations for all joints with rotation axes
            transformations = forward_kinematics(angles_rad, dh_params, joint_axes)
            
            # Apply FK transformations to each joint
            for i, (joint_obj, transform) in enumerate(zip(joints, transformations)):
                if joint_obj:
                    apply_fk_to_joint(joint_obj, transform, settings.base_link)
                    # Keyframe position and rotation
                    joint_obj.keyframe_insert(data_path="location", frame=frame)
                    joint_obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            
            # Calculate and animate TCP position using forward kinematics
            if tcp_obj and settings.animate_tcp and transformations:
                final_transform = transformations[-1]
                
                # Apply FK transformation to TCP
                apply_fk_to_joint(tcp_obj, final_transform, settings.base_link)
                
                # Keyframe TCP
                tcp_obj.keyframe_insert(data_path="location", frame=frame)
                tcp_obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            
            frame += settings.frame_step
        
        self.report({'INFO'}, f"Imported {len(commands)} joint configurations into robot")
        return {'FINISHED'}

class VIEW3D_PT_fk_importer(bpy.types.Panel):
    bl_label = "FK Importer"
    bl_idname = "VIEW3D_PT_fk_importer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation'
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.fk_importer_settings
        
        layout.prop(settings, "filepath")
        
        box = layout.box()
        box.label(text="Robot Joints:")
        box.prop(settings, "base_link")
        
        # Joint 1
        row = box.row()
        row.prop(settings, "joint_1")
        row.prop(settings, "joint_1_axis", text="")
        
        # Joint 2
        row = box.row()
        row.prop(settings, "joint_2")
        row.prop(settings, "joint_2_axis", text="")
        
        # Joint 3
        row = box.row()
        row.prop(settings, "joint_3")
        row.prop(settings, "joint_3_axis", text="")
        
        # Joint 4
        row = box.row()
        row.prop(settings, "joint_4")
        row.prop(settings, "joint_4_axis", text="")
        
        # Joint 5
        row = box.row()
        row.prop(settings, "joint_5")
        row.prop(settings, "joint_5_axis", text="")
        
        # Joint 6
        row = box.row()
        row.prop(settings, "joint_6")
        row.prop(settings, "joint_6_axis", text="")
        
        layout.separator()
        
        column = layout.column(align=True)
        column.prop(settings, "start_frame")
        column.prop(settings, "frame_step")
        
        layout.prop(settings, "create_action")
        
        layout.separator()
        
        # DH Parameters section
        box = layout.box()
        box.label(text="DH Parameters:", icon='SETTINGS')
        box.operator("anim.calculate_dh", text="Calculate DH from Scene", icon='CON_KINEMATIC')
        
        if settings.dh_params_calculated:
            box.label(text="✓ Parameters calculated", icon='CHECKMARK')
            box.label(text="(Check console for values)")
        
        layout.separator()
        
        # TCP Animation section
        box = layout.box()
        box.label(text="TCP Animation (Optional):", icon='EMPTY_AXIS')
        box.prop(settings, "animate_tcp")
        if settings.animate_tcp:
            box.prop(settings, "tcp_object")
            if not settings.tcp_object:
                box.label(text="⚠ Select TCP object", icon='ERROR')
        
        layout.separator()
        
        layout.operator("anim.import_fk", text="Import FK Program", icon='ARMATURE_DATA')

import bpy
from . KRL_translator import *
from bpy.types import PropertyGroup, Operator, Scene
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from bpy.utils import register_class, unregister_class

classes = (
    KRLImporterSettings,
    ANIM_OT_import_krl,
    VIEW3D_PT_krl_importer,
)

def register():
    for cls in classes:
        register_class(cls)
    Scene.krl_importer_settings = PointerProperty(type=KRLImporterSettings)

def unregister():
    for cls in reversed(classes):
        unregister_class(cls)
    del Scene.krl_importer_settings

if __name__ == "__main__":
    register()
import bpy
from . KRL_translator import *
from . KRL_Forward_Kinematics_translator import *
from bpy.types import PropertyGroup, Operator, Scene, AddonPreferences
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from bpy.utils import register_class, unregister_class

def update_krl_importer(self, context):
    """Update function to register/unregister KRL importer based on preference"""
    if self.enable_krl_importer:
        # Register KRL importer classes
        if not hasattr(bpy.types, 'VIEW3D_PT_krl_importer'):
            register_class(KRLImporterSettings)
            register_class(ANIM_OT_import_krl)
            register_class(VIEW3D_PT_krl_importer)
            Scene.krl_importer_settings = PointerProperty(type=KRLImporterSettings)
    else:
        # Unregister KRL importer classes
        if hasattr(bpy.types, 'VIEW3D_PT_krl_importer'):
            if hasattr(Scene, 'krl_importer_settings'):
                del Scene.krl_importer_settings
            unregister_class(VIEW3D_PT_krl_importer)
            unregister_class(ANIM_OT_import_krl)
            unregister_class(KRLImporterSettings)

def update_fk_importer(self, context):
    """Update function to register/unregister FK importer based on preference"""
    if self.enable_fk_importer:
        # Register FK importer classes
        if not hasattr(bpy.types, 'VIEW3D_PT_fk_importer'):
            register_class(FKImporterSettings)
            register_class(ANIM_OT_import_fk)
            register_class(VIEW3D_PT_fk_importer)
            Scene.fk_importer_settings = PointerProperty(type=FKImporterSettings)
    else:
        # Unregister FK importer classes
        if hasattr(bpy.types, 'VIEW3D_PT_fk_importer'):
            if hasattr(Scene, 'fk_importer_settings'):
                del Scene.fk_importer_settings
            unregister_class(VIEW3D_PT_fk_importer)
            unregister_class(ANIM_OT_import_fk)
            unregister_class(FKImporterSettings)

class SERUM_Preferences(AddonPreferences):
    bl_idname = __package__

    enable_krl_importer: BoolProperty(
        name="Enable KRL Importer",
        default=True,
        description="Enable the KRL Importer add-on",
        update=update_krl_importer,
    ) # type: ignore

    enable_fk_importer: BoolProperty(
        name="Enable FK Importer",
        default=True,
        description="Enable the Forward Kinematics Importer add-on",
        update=update_fk_importer,
    ) # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.label(text="SeRUM Add-on Preferences")
        layout.prop(self, "enable_krl_importer")
        layout.prop(self, "enable_fk_importer")

krl_classes = (
    KRLImporterSettings,
    ANIM_OT_import_krl,
    VIEW3D_PT_krl_importer,
)

fk_classes = (
    FKImporterSettings,
    ANIM_OT_import_fk,
    VIEW3D_PT_fk_importer,
)

def register():
    # Always register preferences first
    register_class(SERUM_Preferences)
    
    # Get preferences and check if KRL importer should be enabled
    preferences = bpy.context.preferences.addons[__package__].preferences
    
    if preferences.enable_krl_importer:
        for cls in krl_classes:
            register_class(cls)
        Scene.krl_importer_settings = PointerProperty(type=KRLImporterSettings)
    
    if preferences.enable_fk_importer:
        for cls in fk_classes:
            register_class(cls)
        Scene.fk_importer_settings = PointerProperty(type=FKImporterSettings)

def unregister():
    # Unregister FK importer classes if they are registered
    if hasattr(bpy.types, 'VIEW3D_PT_fk_importer'):
        if hasattr(Scene, 'fk_importer_settings'):
            del Scene.fk_importer_settings
        for cls in reversed(fk_classes):
            unregister_class(cls)
    
    # Unregister KRL importer classes if they are registered
    if hasattr(bpy.types, 'VIEW3D_PT_krl_importer'):
        if hasattr(Scene, 'krl_importer_settings'):
            del Scene.krl_importer_settings
        for cls in reversed(krl_classes):
            unregister_class(cls)
    
    # Always unregister preferences last
    unregister_class(SERUM_Preferences)

if __name__ == "__main__":
    register()
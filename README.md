# Sejfo Robot Utility and Manipulator - SeRUM
*This project is currently in early stages of development*

### **Requirements:** 
- Blender 4.2+

### **Installation**
 Drag the .zip file into your Blender window and click OK in the widget that pops up.
 After installation you will find the KRL importer in the Animation tab in your N-panel.

### **Usage**
In its current state it has support for KRL XYZABC linear and cirular statements, as well as WAIT SEC statements.

You'll need a robot model with an IK ready armature. 

When importing you will have to choose an *Empty* object to apply the animation to.
You have the option to set the origin of your robot base to set the animation in relation to your robot's position.

**Milimeters to Meters** and **Global Scale** can be left untouched unless you intend to use a different world scale. (In general when importing CAD-type data into Blender you need to scale it at a factor of 0.001 for the data to appear correctly.)

**Start Frame** is how far along into the timeline (counting from 0) the animation should start.

**Frame Step** is the amount of frames it will take for a pose to be fully executed.

**Create New Action** - while checked it will create a new animation action with the name of the entered program.

When imported the animation will be applied to the **Target Empty** - Make the IK Target Bone of your robot rig a child (using bone constraints) of the empty to animate the robot. 

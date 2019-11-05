import bpy
import csv


fp= open('C:/Users/m8avhru/Documents/MATLAB/Projects/Stereo-Odometry-SOFT-master/code/voxel_p10.csv')
rdr = csv.reader(fp, delimiter=',')

def color(p):
    if p >= 0.7 and p < 0.75:
        mat = bpy.data.materials.new(name = 'blue')
        # Set some properties of the material.
        mat.diffuse_color = (0., 0., 1.)
        cube = bpy.context.active_object
        mesh = cube.data
        mesh.materials.append(mat)

    elif p >= 0.75 and p < 0.8:
        mat = bpy.data.materials.new(name = 'violet')
        # Set some properties of the material.
        mat.diffuse_color = (float(0.5), 0., 1.)
        cube = bpy.context.active_object
        mesh = cube.data
        mesh.materials.append(mat)

    elif p >= 0.8 and p < 0.9:
        mat = bpy.data.materials.new(name = 'pink')
        # Set some properties of the material.
        mat.diffuse_color = (float(0.5), 0., 0.)
        cube = bpy.context.active_object
        mesh = cube.data
        mesh.materials.append(mat)

    else:
        mat = bpy.data.materials.new(name = 'red')
        # Set some properties of the material.
        mat.diffuse_color = (1., 0., 0.)
        cube = bpy.context.active_object
        mesh = cube.data
        mesh.materials.append(mat)

    return
  

# Create a material.
#mat = bpy.data.materials.new(name = 'red')
# Set some properties of the material.
#mat.diffuse_color = (1., 0., 0.)
#mat.diffuse_shader = 'LAMBERT' 
#mat.diffuse_intensity = 1.0 
#mat.specular_color = (1., 1., 1.)
#mat.specular_shader = 'COOKTORR'
#mat.specular_intensity = 0.5
#mat.alpha = 1
#mat.ambient = 1

line_count = 0
for row in rdr:
    if line_count >= 9000 and line_count<12000:
        x = float(row[1])
        y = float(row[3])
        z = -(float(row[2])-0.8)
        p = float(row[0])
        bpy.ops.mesh.primitive_cube_add(location=(x,y,z))
        bpy.context.scene.objects.active.scale = (0.1, 0.1, 0.1) 
        color(p)
        line_count = line_count+1
           
    else:
        line_count = line_count+1



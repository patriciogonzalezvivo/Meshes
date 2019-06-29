#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from math import sqrt, cos, sin
from Meshes.matrix import *
from Meshes.tools import boundingBox

try:
    basestring
except NameError:
    basestring = str

class Mesh(object):
    def __init__(self, name = ''):
        self.name = name
        self.vertices = []
        self.vertices_colors = []
        self.vertices_normals = []
        self.vertices_texcoords = []
        self.indices = []
        self.indices_normals = []
        self.indices_texcoords= []
        self.edge_indices = []
        self.edge_color = []
        self.materials = []

    def add( self, mesh ):
        offset = len(self.vertices)

        self.vertices.extend(mesh.vertices)
        self.vertices_colors.extend(mesh.vertices_colors)
        self.vertices_normals.extend(mesh.vertices_normals)
        self.vertices_texcoords.extend(mesh.vertices_texcoords)

        for i in range(len(mesh.indices)):
            self.indices.append( offset + mesh.indices[i] );

        for i in range(len(mesh.materials)):
            index = offset + mesh.materials[i][0]
            mat = mesh.materials[i][1]
            self.addMaterial( mat, index )

    # VERTICES

    def addVertex( self, v ):
        self.vertices.append( np.array(v.copy()) )

    def totalVertices( self ):
        return len(self.vertices)

    def vertexString( self, index ):
        return '%f %f %f' % (self.vertices[index][0], self.vertices[index][1], self.vertices[index][2])

    # TEXCOORDS

    def addTexCoord( self, vt ):
        self.vertices_texcoords.append( np.array(vt.copy()) )

    def addTexCoordIndex( self, index ):
        self.indices_texcoords.append( index );

    def addTexCoordTriangle( self, i1, i2, i3 ):
        self.addTexCoordIndex( i1 )
        self.addTexCoordIndex( i2 )
        self.addTexCoordIndex( i3 )

    def texCoordString( self, index ):
        return ' %f %f' % (self.vertices_texcoords[index][0], self.vertices_texcoords[index][1])

    # NORMALS

    def addNormal( self, vn ):
        self.vertices_normals.append( np.array(vn.copy()) )

    def addNormalIndex( self, index ):
        self.indices_normals.append( index )

    def addNormalTriangle( self, i1, i2, i3 ):
        self.addNormalIndex( i1 )
        self.addNormalIndex( i2 )
        self.addNormalIndex( i3 )

    def normalString( self, index):
        n = self.vertices_normals[index]
        return ' %f %f %f' % (n[0], n[1], n[2])

    # COLORS

    def addColor( self, vc ):
        if isinstance(vc, basestring) or isinstance(vc, str):
            vc = vc.lstrip('#')
            lv = len(vc)
            color = tuple(int(vc[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            self.vertices_colors.append( [color[0], color[1], color[2]] )
        else:
            self.vertices_colors.append( vc.copy() )

    def colorString( self, index, alpha = True ):
        if len(self.vertices_colors[index]) == 3:
            return ' %i %i %i' % (self.vertices_colors[index][0], self.vertices_colors[index][1], self.vertices_colors[index][2])
        elif len(self.vertices_colors[index]) == 4:
            if alpha:
                return ' %f %f %f %f' % (self.vertices_colors[index][0], self.vertices_colors[index][1], self.vertices_colors[index][2], self.vertices_colors[index][3])
            else:
                return ' %f %f %f' % (self.vertices_colors[index][0], self.vertices_colors[index][1], self.vertices_colors[index][2])

    # EDGES

    def addEdge( self, i1, i2, color = None ):
        self.edge_indices.append( i1 );
        self.edge_indices.append( i2 );
        if color:
            self.edge_color.append( color )

    def totalEdges( self ):
        return int(len(self.edge_indices)/2)

    def edgeString( self, number ):
        v1 = self.edge_indices[number*2]
        v2 = self.edge_indices[number*2+1]

        string = '%i %i' % (v1, v2)

        if len(self.edge_color) > 0:
            if len(self.edge_color[number]) == 3:
                string += ' %i %i %i' % (self.edge_color[number][0], self.edge_color[number][1], self.edge_color[number][2])
            elif len(self.edge_color[number]) == 4:
                string += ' %f %f %f %f' % (self.edge_color[number][0], self.edge_color[number][1], self.edge_color[number][2], self.edge_color[number][3])
        
        return string

    # TRIANGLES / FACES

    def addIndex( self, index ):
        self.indices.append( index )

    def totalIndices( self ):
        return len(self.indices)

    def addTriangle( self, i1, i2, i3 ):
        self.addIndex( i1 )
        self.addIndex( i2 )
        self.addIndex( i3 )

    def triangleString( self, index ):
        v1 = self.indices[index*3+0]
        v2 = self.indices[index*3+1]
        v3 = self.indices[index*3+2]
        return ' %i %i %i' % (v1, v2, v3)

    def totalFaces( self ):
        return int(len(self.indices)/3)

    def faceString( self, number ):
        v1 = vt1 = vn1 = self.indices[number*3] + 1
        v2 = vt2 = vn2 = self.indices[number*3+1] + 1
        v3 = vt3 = vn3 = self.indices[number*3+2] + 1

        if len(self.indices_texcoords) > 0:
            vt1 = self.indices_texcoords[number*3] + 1
            vt2 = self.indices_texcoords[number*3+1] + 1
            vt3 = self.indices_texcoords[number*3+2] + 1

        if len(self.indices_normals) > 0:
            vn1 = self.indices_normals[number*3] + 1
            vn2 = self.indices_normals[number*3+1] + 1
            vn3 = self.indices_normals[number*3+2] + 1

        if len(self.vertices_texcoords) > 0:
            if len(self.vertices_normals) > 0:
                return ' %i/%i/%i %i/%i/%i %i/%i/%i' % (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3)
            else:
                return ' %i/%i %i/%i %i/%i' % (v1, vt1, v2, vt2, v3, vt3)
        elif len(self.vertices_normals) > 0:
            return ' %i//%i %i//%i %i//%i' % (v1, vn1, v2, vn2, v3, vn3)
        else:
            return ' %i %i %i' % (v1, v2, v3)

    #  MATERIAL

    def addMaterial( self, mat, index = None ):
        if index == None:
            index = len(self.vertices)
        self.materials.append( [index, mat] )

    # OPERATIONS

    def clear( self ):
        self.vertices = []
        self.vertices_colors = []
        self.vertices_normals = []
        self.vertices_texcoords = []

        self.indices = []
        self.indices_normals = []
        self.indices_texcoords= []

        self.edge_indices = []
        self.edge_color = []

        self.offset = 0

    def invertNormals( self ):
        # tig: flip face(=triangle) winding order, so that we are consistent with all other ofPrimitives.
        # i wish there was a more elegant way to do this, but anything happening before 'split vertices'
        # makes things very, very complicated.
        for i in range(0, len(self.indices))[::3]:
            tmp = self.indices[i+1]
            self.indices[i+1] = self.indices[i+2]
            self.indices[i+2] = tmp

        for i in range(0, len(self.vertices_normals)):
            self.vertices_normals[i] = np.array(self.vertices_normals[i]) * -1.

    def flatNormals( self ):
        # get copy original mesh data
        numIndices = len(self.indices)
        indices = self.indices
        verts = self.vertices
        texCoords = self.vertices_texcoords
        colors = self.vertices_colors
        
        # remove all data to start from scratch
        self.clear();
        
        # add mesh data back, duplicating vertices and recalculating normals
        normal = []
        for i in range(0, numIndices):
            indexCurr = indices[i];
    
            if i % 3 == 0:
                indexNext1 = indices[i + 1]
                indexNext2 = indices[i + 2]

                e1 = verts[indexCurr] - verts[indexNext1]
                e2 = verts[indexNext2] - verts[indexNext1]
                t = np.cross(e1, e2) * -1.
                dist = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
                normal = t / dist
    
            self.addIndex(i);
            self.addNormal(normal);
    
            if indexCurr < len(texCoords):
                self.addTexCoord(texCoords[indexCurr])
    
            if indexCurr < len(verts):
                self.addVertex(verts[indexCurr])
    
            if indexCurr < len(colors):
                self.addColor(colors[indexCurr])


    def scale( self, scale ):
        mat = mat4_scale(scale)
        self.transform(mat)


    def translateX( self, d ):
        mat = mat4_translateX(d)
        self.transform(mat)


    def translateY( self, d ):
        mat = mat4_translateY(d)
        self.transform(mat)


    def translateZ( self, d ):
        mat = mat4_translateZ(d)
        self.transform(mat)
        

    def translate( self, dir ):
        mat = np.identity(4)

        if isinstance( dir, (np.ndarray, np.generic) ):
            if dir.shape[0] == 3:
                mat = mat4_translate( dir ) 
            elif len(dir.shape) == 2 and dir.shape[1] == 4:
                mat = dir
        elif isinstance( dir, (list, tuple) ):
            mat = mat4_translate( dir ) 

        self.transform( mat )


    def rotateX( self, deg ):
        mat = mat4_rotateX(deg)
        self.rotate_mat4(mat)


    def rotateY( self, deg ):
        mat = mat4_rotateY(deg)
        self.rotate_mat4(mat)


    def rotateZ( self, deg ):
        mat = mat4_rotateZ(deg)
        self.rotate_mat4(mat)


    def rotate_quat(self, quaternion):
        mat = mat4_from_quat( quaternion)
        self.rotate_mat4(mat)


    def rotate_axis( self, angle, direction, point=None):
        mat = mat4_rotate(angle, direction, point)
        self.rotate_mat4(mat)


    def rotate_axis_euler( self, ai, aj, ak, axes='sxyz'):
        mat = mat4_from_euler(ai, aj, ak, axes)
        self.rotate_mat4(mat)


    def rotate_normal( self, normal, up=[0.0, 0.0, 1.0]):
        self.rotate_from_A_to_B(vec3(up), vec3(normal))


    def rotate_from_A_to_B( self, A_vec, B_vec):
        mat = mat4_from_A_to_B(A_vec, B_vec)
        self.rotate_mat4(mat)


    def rotate_mat4( self, mat4 ):
        self.transform( mat4 )
        self.transform_normals( mat4 )

    
    def transform( self, mat ):
        for i in range(len(self.vertices)):
            self.vertices[i] = mat4_mult(mat, self.vertices[i])


    def transform_normals( self, mat ):
        for i in range(len(self.vertices_normals)):
            self.vertices_normals[i] = mat4_mult(mat, self.vertices_normals[i])


    def center( self ):
        bbox = boundingBox(self.vertices)
        dx = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        dz = bbox[5] - bbox[2]
        self.translateX(-bbox[3] + dx * 0.5 )
        self.translateY(-bbox[4] + dy * 0.5 )
        self.translateZ(-bbox[5] + dy * 0.5 )

    # EXPORT/IMPORT

    def toObj( self, file_name = None ):
        lines = '# OBJ by Patricio Gonzalez Vivo\n'

        # Materials Library
        if file_name != None and len(self.materials) > 0:
            mat_lines = ''
            mat_names = []
            for mat in self.materials:
                name = mat[1].name
                if not name in mat_names:
                    mat_names.append(name)
                    mat_lines += mat[1].toMtl()

            mat_filename = os.path.splitext(file_name)[0] + '.mtl'
            file = open( mat_filename, 'w' )
            file.write( mat_lines )
            file.close()
            lines += 'mtllib ' + os.path.basename(mat_filename) + '\n'

        # Name
        if len(self.name) > 0:
            lines += 'o ' + self.name + '\n'

        # Vertices (and optional color)
        color = len(self.vertices_colors) > 0
        for index in range( len(self.vertices) ):
            lines += 'v ' + self.vertexString( index ) 
            if color:
                lines += self.colorString( index, False )
            lines += '\n'

        # Texture Coords
        for index in range( len(self.vertices_texcoords) ):
            lines += 'vt' + self.texCoordString( index ) + '\n'

        # Normals    
        for index in range( len(self.vertices_normals) ):
            lines += 'vn' + self.normalString( index ) + '\n'

        # Faces
        material_counter = 0
        for index in range( self.totalFaces() ):
            if material_counter < len(self.materials):
                if self.materials[material_counter][0] <= self.indices[index*3] or self.materials[material_counter][0] <= self.indices[index*3+1] or self.materials[material_counter][0] <= self.indices[index*3+2]:
                    lines += 'usemtl ' + self.materials[material_counter][1].name + '\n'
                    material_counter += 1
                    lines += 's 1\n'
            lines += 'f' + self.faceString( index ) + '\n'

        if file_name:
            file = open(file_name, 'w')
            file.write( lines )
            file.close()
        else:
            return lines

    def fromObj( self, file_name ):
        for line in open(file_name, 'r'):
            # Skip comments
            if line.startswith('#'):
                continue

            # Skip empty lines
            if line == "":
                continue
            
            values = line.split()

            # Skip if there is not enough information
            if len(values) < 2:
                continue

            type = values[0]
            args = values[1:]

            if type == 'v':
                if len(args) == 3:
                    v = map(float, args)
                    self.addVertex(np.array(v))
            elif type == 'vt':
                if len(args) == 2:
                    vt = map(float, args)
                    self.addTexCoord(np.array(vt))
            elif type == 'vn':
                if len(args) == 3:
                    vn = map(float, args)
                    self.addNormal(np.array(vn))
            elif type == 'f':
                if len(args) == 3:
                    A = map(int, args[0].split('/'))
                    B = map(int, args[1].split('/'))
                    C = map(int, args[2].split('/'))

                    self.addTriangle(A[0]-1, B[0]-1, C[0]-1)

                    # if (A[0] != A[1] != A[2]) or (B[0] != B[1] != B[2]) or (C[0] != C[1] != C[2]):
                    self.addTexCoordTriangle(A[1]-1, B[1]-1, C[1]-1)
                    self.addNormalTriangle(A[2]-1, B[2]-1, C[2]-1)
                elif len(args) > 3:
                    values = []

                    for i in range(len(args)):
                        values.append( map(int, args[i].split('/')) )

                    # Add first triangle
                    self.addTriangle(values[0][0]-1, values[1][0]-1, values[2][0]-1)
                    # if (values[0][0] != values[0][1] != values[0][2]) or (values[1][0] != values[1][1] != values[1][2]) or (values[2][0] != values[2][1] != values[2][2]):
                    self.addTexCoordTriangle(values[0][1]-1, values[1][1]-1, values[2][1]-1)
                    self.addNormalTriangle(values[0][2]-1, values[1][2]-1, values[2][2]-1)

                    for i in range(3, len(values)):
                        self.addTriangle(values[i-3][0]-1, values[i-1][0]-1, values[i][0]-1)
                        # if (values[i-3][0] != values[i-3][1] != values[i-3][2]) or (values[i-1][0] != values[i-1][1] != values[i-1][2]) or (values[i][0] != values[i][1] != values[i][2]):
                        self.addTexCoordTriangle(values[i-3][1]-1, values[i-1][1]-1, values[i][1]-1)
                        self.addNormalTriangle(values[i-3][2]-1, values[i-1][2]-1, values[i][2]-1)

    def toPly( self, file_name = None ):
        lines = '''ply
format ascii 1.0
element vertex '''+str(len(self.vertices))+'''
property float x
property float y
property float z
'''
        if len(self.vertices_normals) > 0:
            lines += 'property float nx\n'
            lines += 'property float ny\n'
            lines += 'property float nx\n'

        if len(self.vertices_colors) > 0:
            if len(self.vertices_colors[0]) == 3:
                lines += 'property uchar red\n'
                lines += 'property uchar green\n'
                lines += 'property uchar blue\n'
            elif len(self.vertices_colors[0]) == 4:
                lines += 'property float r\n'
                lines += 'property float g\n'
                lines += 'property float b\n'
                lines += 'property float a\n'

        if len(self.vertices_texcoords) > 0:
            lines += 'property float texture_u\n'
            lines += 'property float texture_v\n'

        if len( self.indices ) > 2:
            lines += 'element face '+str( self.totalFaces() )+'\n'
            lines += 'property list uchar int vertex_indices\n'

        if len( self.edge_indices ) > 1:
            lines += 'element edge '+str( self.totalEdges() )+'\n'
            lines += 'property int32 vertex1\n'
            lines += 'property int32 vertex2\n'
            if len(self.edge_color) > 0:
                if len(self.edge_color[0]) == 3:
                    lines += 'property uchar red\n'
                    lines += 'property uchar green\n'
                    lines += 'property uchar blue\n'
                elif len(self.edge_color[0]) == 4:
                    lines += 'property float r\n'
                    lines += 'property float g\n'
                    lines += 'property float b\n'
                    lines += 'property float a\n'

        lines += 'end_header\n'
        for index in range( len(self.vertices) ):
            line = self.vertexString( index )
            if len(self.vertices_normals) > 0:
                line += self.normalString( index )
            if len(self.vertices_colors) > 0:
                line += self.colorString( index )
            if len(self.vertices_texcoords) > 0:
                line += self.texCoordString( index )
            
            lines += line+'\n'

        if len( self.indices ) > 2:
            for t in range( self.totalFaces() ):
                lines += '3' + self.triangleString(t) + '\n'

        if len( self.edge_indices ) > 1:
            for t in range( self.totalEdges() ):
                lines += self.edgeString(t) + '\n'

        if file_name:
            file = open(file_name, 'w')
            file.write( lines )
            file.close()
        else:
            return lines

    def fromPly(self, file_name):
        lineNum = -1

        class Enum(set):
            def __getattr__(self, name):
                if name in self:
                    return name
                raise AttributeError

        State = Enum(["Header", "VertexDef", "FaceDef", "Vertices", "Normals", "Faces"])
        state = State.Header

        orderVertices = -1
        orderIndices = -1

        expectedVertices = 0
        expectedFaces = 0

        vertexCoordsFound = 0
        colorCompsFound = 0
        texCoordsFound = 0
        normalsCoordsFound = 0

        currentVertex = 0
        currentFace = 0

        floatColor = False

        for line in open(file_name, 'r'):
            lineNum += 1
            # get rid of the new line
            line = line.rstrip()
            # print(str(lineNum) + " " + line)

            if lineNum == 0:
                if line != 'ply':
                    print("wrong format, expecting 'ply'")
                    return
            elif lineNum == 1:
                if line != "format ascii 1.0":
                    print("wrong format, expecting 'format ascii 1.0'")
                    return
            
            if 'comment' in line:
                continue

            # HEADER 
            if (state==State.Header or state==State.FaceDef) and line.startswith('element vertex'):
                state = State.VertexDef
                orderVertices = max(orderIndices, 0)+1
                expectedVertices = int(line[15:])
                # print(state)
                # print(line[15:])
                continue;

            if (state==State.Header or state==State.VertexDef) and line.startswith('element face'):
                state = State.FaceDef
                orderIndices = max(orderVertices, 0)+1
                expectedFaces = int(line[13:])
                # print(state)
                # print(line[13:])
                continue

            # Vertex Def
            if state==State.VertexDef:

                if line.startswith('property float x') or line.startswith('property float y') or line.startswith('property float z'):
                    vertexCoordsFound += 1
                    # print('vertexCoordsFound ' + str(vertexCoordsFound))
                    continue

                if line.startswith('property float nx') or line.startswith('property float ny') or line.startswith('property float nz'):
                    normalsCoordsFound += 1
                    # print('normalsCoordsFound ' + str(normalsCoordsFound))
                    continue

                if line.startswith('property float r') or line.startswith('property float g') or line.startswith('property float b') or line.startswith('property float a'):
                    colorCompsFound += 1
                    # print('colorCompsFound ' + str(colorCompsFound))
                    floatColor = True
                    continue
            
                if line.startswith('property uchar red') or line.startswith('property uchar green') or line.startswith('property uchar blue') or line.startswith('property uchar alpha'):
                    colorCompsFound += 1
                    # print('colorCompsFound ' + str(colorCompsFound))
                    floatColor = False
                    continue

                if line.startswith('property float u') or line.startswith('property float v'):
                    texCoordsFound += 1
                    # print('texCoordsFound ' + str(texCoordsFound))
                    continue

                if line.startswith('property float texture_u') or line.startswith('property float texture_v'):
                    texCoordsFound += 1
                    # print('texCoordsFound ' + str(texCoordsFound))
                    continue

            # if state==State.FaceDef and line.find('property list')!=0 and line!='end_header':
            #     print('wrong face definition')

            if line=='end_header':
                # Check that all basic elements seams ok and healthy
                if colorCompsFound > 0 and colorCompsFound < 3:
                    print('data has color coordiantes but not correct number of components. Found ' + str(colorCompsFound) + ' expecting 3 or 4')
                    return

                if normalsCoordsFound != 3:
                    print('data has normal coordiantes but not correct number of components. Found ' + str(normalsCoordsFound) + ' expecting 3')
                    return

                if expectedVertices == 0:
                    print('mesh loaded has no vertices')
                    return

                if orderVertices == -1:
                    orderVertices = 9999
                if orderIndices == -1:
                    orderIndices = 9999;

                if orderVertices < orderIndices:
                    state = State.Vertices
                else:
                    state = State.Faces

                continue
            
            if state == State.Vertices:
                values = line.split()

                # Extract vertex
                v = [0.0, 0.0, 0.0]
                v[0] = float(values.pop(0))
                v[1] = float(values.pop(0))
                if vertexCoordsFound > 2:
                    v[2] = float(values.pop(0))
                self.addVertex(np.array(v))

                # Extract normal
                if normalsCoordsFound > 0:
                    n = [0.0, 0.0, 0.0]
                    n[0] = float(values.pop(0))
                    n[1] = float(values.pop(0))
                    n[2] = float(values.pop(0))
                    self.addNormal(np.array(n))

                # Extract color
                if colorCompsFound > 0:
                    c = [1.0, 1.0, 1.0, 1.0]
                    div = 255.0
                    if floatColor:
                        div = 1.0

                    c[0] = float(values.pop(0))/div
                    c[1] = float(values.pop(0))/div
                    c[2] = float(values.pop(0))/div
                    if colorCompsFound > 3:
                        c[3] = float(values.pop(0))/div
                    self.addColor(np.array(c))

                # Extract UVs
                if texCoordsFound > 0:
                    uv = [0.0, 0.0]
                    uv[0] = float(values.pop(0))
                    uv[1] = float(values.pop(0))
                    self.addTexCoord(np.array(uv))

                if len(self.vertices) == expectedVertices:
                    if orderVertices < orderIndices:
                        state = State.Faces
                    else:
                        state = State.Vertices
                    continue

            if state == State.Faces:
                values = line.split()
                numV = int(values.pop(0))

                if numV != 3:
                    print("face not a triangle")

                for i in range(numV):
                    index = int(values.pop(0))
                    self.addIndex( index )
                    if normalsCoordsFound:
                        self.addNormalIndex(index)
                    if texCoordsFound:
                        self.addTexCoordIndex(index)

                if currentFace == expectedFaces:
                    print("finish w indices")
                    if orderVertices<orderIndices:
                        state = State.Vertices
                    else:
                        state = State.Faces
                    continue

                currentFace += 1

    def toBlenderMesh( self, blender_mesh ):
        edges = []
        for edge in range( self.totalEdges() ):
            v1 = self.edge_indices[edge*2]
            v2 = self.edge_indices[edge*2+1]
            edges.append( (v1, v2) ) 

        faces = []
        for face in range( self.totalFaces() ):
            v1 = self.indices[face*3]
            v2 = self.indices[face*3+1]
            v3 = self.indices[face*3+2]
            faces.append( (v1, v2, v3) )

        blender_mesh.from_pydata( self.vertices, edges, faces )

        # Texture coordinates per vertex *per polygon loop*.
        # Create UV coordinate layer and set values
        if len(self.vertices_texcoords) > 0:
            uv_layer = blender_mesh.uv_layers.new()
            for i, uv in enumerate(uv_layer.data):
                index = self.indices[i]
                uv.uv = self.vertices_texcoords[index]

        # Vertex color per vertex *per polygon loop*    
        # Create vertex color layer and set values
        if len(self.vertices_colors) > 0:
            vcol_lay = blender_mesh.vertex_colors.new()
            for i, col in enumerate(vcol_lay.data):
                index = self.indices[i]
                col.color[0] = self.vertices_colors[index][0]
                col.color[1] = self.vertices_colors[index][1]
                col.color[2] = self.vertices_colors[index][2]
                col.color[3] = 1.0                     # Alpha?
            
        # We're done setting up the mesh values, update mesh object and 
        # let Blender do some checks on it
        blender_mesh.update()
        blender_mesh.validate()

        return blender_mesh



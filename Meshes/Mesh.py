#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from math import sqrt, cos, sin
from Meshes.matrix import mv_mult, mat4_rotateX, mat4_rotateY, mat4_rotateZ, mat4_translateX, mat4_translateY, mat4_translateZ, mat4_scale
from Meshes.tools import boundingBox

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

    def addVertex( self, v ):
        self.vertices.append( np.array(v) )

    def vertexString( self, index ):
        return '%f %f %f' % (self.vertices[index][0], self.vertices[index][1], self.vertices[index][2])

    def addTexCoord( self, vt ):
        self.vertices_texcoords.append( np.array(vt) )

    def texCoordString( self, index ):
        return ' %f %f' % (self.vertices_texcoords[index][0], self.vertices_texcoords[index][1])

    def addNormal( self, vn ):
        self.vertices_normals.append( np.array(vn) )

    def normalString( self, index):
        n = self.vertices_normals[index]
        return ' %f %f %f' % (n[0], n[1], n[2])

    def addColor( self, vc ):
        self.vertices_colors.append( vc )

    def colorString( self, index, alpha = True ):
        if len(self.vertices_colors[index]) == 3:
            return ' %i %i %i' % (self.vertices_colors[index][0], self.vertices_colors[index][1], self.vertices_colors[index][2])
        elif len(self.vertices_colors[index]) == 4:
            if alpha:
                return ' %f %f %f %f' % (self.vertices_colors[index][0], self.vertices_colors[index][1], self.vertices_colors[index][2], self.vertices_colors[index][3])
            else:
                return ' %f %f %f' % (self.vertices_colors[index][0], self.vertices_colors[index][1], self.vertices_colors[index][2])
    
    def addIndex( self, index ):
        self.indices.append( index );

    def addTriangle( self, i1, i2, i3 ):
        self.addIndex( i1 )
        self.addIndex( i2 )
        self.addIndex( i3 )

    def triangleString( self, index ):
        v1 = self.indices[index*3+0]
        v2 = self.indices[index*3+1]
        v3 = self.indices[index*3+2]

        return ' %i %i %i' % (v1, v2, v3)

    def addNormalIndex( self, index ):
        self.indices_normals.append( index );

    def addNormalTriangle( self, i1, i2, i3 ):
        self.addNormalIndex( i1 )
        self.addNormalIndex( i2 )
        self.addNormalIndex( i3 )

    def addTexCoordIndex( self, index ):
        self.indices_texcoords.append( index );

    def addTexCoordTriangle( self, i1, i2, i3 ):
        self.addTexCoordIndex( i1 )
        self.addTexCoordIndex( i2 )
        self.addTexCoordIndex( i3 )

    def faceString( self, index ):
        v1 = vt1 = vn1 = self.indices[index*3] + 1
        v2 = vt2 = vn2 = self.indices[index*3+1] + 1
        v3 = vt3 = vn3 = self.indices[index*3+2] + 1

        if len(self.indices_texcoords) > 0:
            vt1 = self.indices_texcoords[index*3] + 1
            vt2 = self.indices_texcoords[index*3+1] + 1
            vt3 = self.indices_texcoords[index*3+2] + 1

        if len(self.indices_normals) > 0:
            vn1 = self.indices_normals[index*3] + 1
            vn2 = self.indices_normals[index*3+1] + 1
            vn3 = self.indices_normals[index*3+2] + 1

        if len(self.vertices_texcoords) > 0:
            if len(self.vertices_normals) > 0:
                return ' %i/%i/%i %i/%i/%i %i/%i/%i' % (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3)
            else:
                return ' %i/%i %i/%i %i/%i' % (v1, vt1, v2, vt2, v3, vt3)
        elif len(self.vertices_normals) > 0:
            return ' %i//%i %i//%i %i//%i' % (v1, vn1, v2, vn2, v3, vn3)
        else:
            return ' %i %i %i' % (v1, v2, v3)

    def totalFaces(self):
        return int(len(self.indices)/3)

    def addMaterial(self, mat, index = None):
        if index == None:
            index = len(self.vertices)
        self.materials.append( [index, mat] )

    def clear(self):
        self.vertices = []
        self.vertices_colors = []
        self.vertices_normals = []
        self.vertices_texcoords = []
        self.indices = []
        self.offset = 0

    def invertNormals(self):
        # tig: flip face(=triangle) winding order, so that we are consistent with all other ofPrimitives.
        # i wish there was a more elegant way to do this, but anything happening before "split vertices"
        # makes things very, very complicated.
        for i in range(0, len(self.indices))[::3]:
            tmp = self.indices[i+1]
            self.indices[i+1] = self.indices[i+2]
            self.indices[i+2] = tmp

        for i in range(0, len(self.vertices_normals)):
            self.vertices_normals[i] = np.array(self.vertices_normals[i]) * -1.

    def flatNormals(self):
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

    def rotateX(self, deg):
        mat = mat4_rotateX(deg)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

        for i in range(len(self.vertices_normals)):
            self.vertices_normals[i] = mv_mult(mat, self.vertices_normals[i])

    def rotateY(self, deg):
        mat = mat4_rotateY(deg)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

        for i in range(len(self.vertices_normals)):
            self.vertices_normals[i] = mv_mult(mat, self.vertices_normals[i])

    def rotateZ(self, deg):
        mat = mat4_rotateZ(deg)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

        for i in range(len(self.vertices_normals)):
            self.vertices_normals[i] = mv_mult(mat, self.vertices_normals[i])

    def translateX(self, d):
        mat = mat4_translateX(d)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

    def translateY(self, d):
        mat = mat4_translateY(d)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

    def translateZ(self, d):
        mat = mat4_translateZ(d)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

    def scale(self, sx, sy, sz):
        mat = mat4_scale(d)
        for i in range(len(self.vertices)):
            self.vertices[i] = mv_mult(mat, self.vertices[i])

    def center(self):
        bbox = boundingBox(self.vertices)
        dx = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        dz = bbox[5] - bbox[2]
        self.translateX(-bbox[3] + dx*.5)
        self.translateY(-bbox[4] + dy*.5)
        self.translateZ(-bbox[5] + dy*.5)

    def toObj(self, file_name = None):
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

    def fromObj(self, file_name):
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




    def toPly(self, file_name = None):
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
            lines += 'property float u\n'
            lines += 'property float v\n'

        lines += '''element face '''+str( self.totalFaces() )+'''
property list uchar int vertex_indices
end_header
'''
        for index in range( len(self.vertices) ):
            line = self.vertexString( index )
            if len(self.vertices_normals) > 0:
                line += self.normalString( index )
            if len(self.vertices_colors) > 0:
                line += self.colorString( index )
            if len(self.vertices_texcoords) > 0:
                line += self.texCoordString( index )
            
            lines += line+'\n'

        for t in range( self.totalFaces() ):
            lines += '3' + self.triangleString(t) + '\n'

        if file_name:
            file = open(file_name, 'w')
            file.write( lines )
            file.close()
        else:
            return lines

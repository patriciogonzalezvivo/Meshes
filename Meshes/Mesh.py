#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
from math import cos, sin, sqrt, atan2, fabs
import numpy as np

class Mesh(object):
    def __init__(self, name = ''):
        self.name = name
        self.vertices = []
        self.vertices_colors = []
        self.vertices_normals = []
        self.vertices_texcoords = []
        self.indices = []
        self.materials = []

    def addMesh( self, mesh ):
        offset = len(self.vertices)

        self.vertices.extend(mesh.vertices)
        self.vertices_colors.extend(mesh.vertices_colors)
        self.vertices_normals.extend(mesh.vertices_normals)
        self.vertices_texcoords.extend(mesh.vertices_texcoords)

        for index in range(len(mesh.indices)):
            self.indices.append( offset + mesh.indices[index] );

        for mat in mesh.materials:
            self.materials.append( [offset + mat[0], mat[1]] )

    def addVertex( self, v ):
        self.vertices.append( v )

    def vertexString( self, index ):
        return '%f %f %f' % (self.vertices[index][0], self.vertices[index][1], self.vertices[index][2])

    def addTexCoord( self, vt ):
        self.vertices_texcoords.append( vt )

    def texCoordString( self, index ):
        return ' %f %f' % (self.vertices_texcoords[index][0], self.vertices_texcoords[index][1])

    def addNormal( self, vn ):
        self.vertices_normals.append( vn )

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

    def faceString( self, index ):
        v1 = self.indices[index*3] + 1
        v2 = self.indices[index*3+1] + 1
        v3 = self.indices[index*3+2] + 1

        if len(self.vertices_texcoords) > 0:
            if len(self.vertices_normals) > 0:
                return ' %i/%i/%i %i/%i/%i %i/%i/%i' % (v1, v1, v1, v2, v2, v2, v3, v3, v3)
            else:
                return ' %i/%i %i/%i %i/%i' % (v1, v1, v2, v2, v3, v3)
        else:
            return ' %i %i %i' % (v1, v2, v3)

    def addMaterial(self, mat):
        self.materials.append([len(self.vertices), mat])

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

                e1 = np.array(verts[indexCurr]) - np.array(verts[indexNext1])
                e2 = np.array(verts[indexNext2]) - np.array(verts[indexNext1])
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

    def toObj(self, file_name = None):
        lines = '# OBJ by Patricio Gonzalez Vivo\n'

        # Materials Library
        if file_name != None and len(self.materials) > 0:
            mat_lines = ''
            for mat in self.materials:
                mat_lines += mat[1].toMtl()

            mat_filename = os.path.splitext(file_name)[0] + '.mtl'
            file = open( mat_filename, 'w' )
            file.write( mat_lines )
            file.close()
            lines += 'mtllib ' + mat_filename + '\n'

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
        for index in range( int( len(self.indices)/3 ) ):
            if material_counter < len(self.materials):
                if self.materials[material_counter][0] <= self.indices[index*3] or self.materials[material_counter][0] <= self.indices[index*3+1] or self.materials[material_counter][0] <= self.indices[index*3+2]:
                    print('self.materials[material_counter][0] <=',index)
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

    def toPly(self, file_name = None):
        lines = '''ply
format ascii 1.0
element vertex '''+str(len(self.vertices))+'''
property float x
property float y
property float z
'''

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

        if len(self.vertices_normals) > 0:
            lines += 'property float nx\n'
            lines += 'property float ny\n'
            lines += 'property float nx\n'

        lines += '''element face '''+str(int(len(self.indices)/3))+'''
property list uchar int vertex_indices
end_header
'''
        for index in range( len(self.vertices) ):
            line = self.vertexString( index )
            if len(self.vertices_colors) > 0:
                line += self.colorString( index )
            if len(self.vertices_texcoords) > 0:
                line += self.texCoordString( index )
            if len(self.vertices_normals) > 0:
                line += self.normalString( index )
            
            lines += line+'\n'

        for t in range( int( len(self.indices)/3 ) ):
            lines += '3' + self.triangleString(t) + '\n'

        if file_name:
            file = open(file_name, 'w')
            file.write( lines )
            file.close()
        else:
            return lines

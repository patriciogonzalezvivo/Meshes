#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import floor, ceil
import numpy as np
from triangle import triangulate

from Meshes.quaternion import quat_from_axis, quat_mult
from Meshes.tools import boundingBox, remap

def tessIsoRect( mesh, width, height, precision = 1.0, z = 0.0, color = None):
    offset = len(mesh.vertices)
    normal = [0., 0., 1.]

    w = ceil(width / precision)
    h = ceil(height / precision)
    for y in range(0, int(h)):
        for x in range(0, int(w)):
            offsetX = 0
            if x%2 == 1:
                offsetY = 0.5 
            else:
                offsetY = 0.0

            mesh.addVertex( [(x + offsetX) * precision, (y + offsetY) * precision, z] )
            if color:
                mesh.addColor( color )
            mesh.addNormal( normal )
            mesh.addTexCoord( [float(x+offsetX)/float(w-1), float(y+offsetY)/float(h-1)] )
    
    for y in range(0, int(h)-1):
        for x in range(0, int(w)-1):
            if x%2 == 0:
                mesh.addIndex(offset + x + y * w)               # a
                mesh.addIndex(offset + (x + 1) + y * w)         # b
                mesh.addIndex(offset + x + (y + 1) * w)         # d
                
                mesh.addIndex(offset + (x + 1) + y * w)         # b
                mesh.addIndex(offset + (x + 1) + (y + 1) * w)   # c
                mesh.addIndex(offset + x + (y + 1) * w)         # d
            else:
                mesh.addIndex(offset + (x + 1) + (y + 1) * w)   # c
                mesh.addIndex(offset + x + y * w)               # a
                mesh.addIndex(offset + (x + 1 ) + y * w)        # b
                
                mesh.addIndex(offset + (x + 1) + (y + 1) * w)   # c
                mesh.addIndex(offset + x + (y + 1) * w)         # d
                mesh.addIndex(offset + x + y * w)               # a
    return mesh

def tessRect( mesh, width, height, precision = 1.0, z = 0.0, color = None):
    offset = len(mesh.vertices)
    normal = [0., 0., 1.]

    w = ceil(width / precision)
    h = ceil(height / precision)
    for y in range(0, int(h)):
        for x in range(0, int(w)):
            mesh.addVertex( [float(x) * precision - float(width) * 0.5, 
                             float(y) * precision - float(height) * 0.5, 
                             z] )
            if color:
                mesh.addColor( color )
            mesh.addNormal( normal )
            mesh.addTexCoord( [float(x)/float(w-1), float(y)/float(h-1)] )
    
    for y in range(0, int(h)-1):
        for x in range(0, int(w)-1):
            if x%2 == 0:
                mesh.addIndex(offset + x + y * w)               # a
                mesh.addIndex(offset + (x + 1) + y * w)         # b
                mesh.addIndex(offset + x + (y + 1) * w)         # d
                
                mesh.addIndex(offset + (x + 1) + y * w)         # b
                mesh.addIndex(offset + (x + 1) + (y + 1) * w)   # c
                mesh.addIndex(offset + x + (y + 1) * w)         # d
            else:
                mesh.addIndex(offset + (x + 1) + (y + 1) * w)   # c
                mesh.addIndex(offset + x + y * w)               # a
                mesh.addIndex(offset + (x + 1 ) + y * w)        # b
                
                mesh.addIndex(offset + (x + 1) + (y + 1) * w)   # c
                mesh.addIndex(offset + x + (y + 1) * w)         # d
                mesh.addIndex(offset + x + y * w)               # a
    return mesh

def tessPolygon( mesh, positions, z, color = None, flipped = False,  texture_boundingBox = None ):
    offset = len(mesh.vertices)

    n = [0., 0., 1.]

    # Get rid of begining/ending duplicate
    if positions[0][0] == positions[-1][0] and positions[0][1] == positions[-1][1]:
        positions.pop()

    rotY = []
    if flipped: 
        n = [0., 0., -1.]
        rotY = quat_from_axis(np.radians(180), (0, 1, 0))

    if texture_boundingBox == None:
        texture_boundingBox = boundingBox(positions)

    points = []
    for point in positions:
        v = np.array([point[0], -point[1], z])
        if flipped:
            v = quat_mult(rotY, (v[0], v[1], v[2]) )

        points.append( [point[0], point[1]] )
        
        mesh.addTexCoord( [ remap(point[0], texture_boundingBox[0], texture_boundingBox[2], 0.0, 1.0), 
                            remap(point[1], texture_boundingBox[1], texture_boundingBox[3], 0.0, 1.0)] )

        mesh.addVertex( v )
        mesh.addNormal( n )
        if color:
            mesh.addColor( color )

    segments = []
    for i in range( len(positions) ):
        segments.append( [i, (i + 1) % len(positions) ] )

    cndt = triangulate(dict(vertices=points, segments=segments),'p')

    for face in cndt['triangles']:
        mesh.addTriangle(   offset + ( face[0] % len(points) ), 
                            offset + ( face[1] % len(points) ), 
                            offset + ( face[2] % len(points) )  )
    
    # offset += len(positions)

    return mesh
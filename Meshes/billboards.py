#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

# inspired from https://github.com/keijiro/Pcx/blob/master/Assets/Pcx/Shaders/Disk.cginc 
def circle(mesh, pos=[0,0,0], num_vertices = 8, color = None):
    offset = len(mesh.vertices)  
    normal = [0., 0., 1.]  

    # Top vertex
    mesh.addVertex( pos )
    if color:
        mesh.addColor( color )
    mesh.addNormal( normal )
    mesh.addTexCoord( [0, 1] )  

    slices = int(math.floor(num_vertices / 2))
    for i in range(1, slices):
        sin = math.sin(i * math.pi / slices)
        cos = math.cos(i * math.pi / slices)

        # Right-side vertex
        mesh.addVertex( pos )
        if color:
            mesh.addColor( color )
        mesh.addNormal( normal ) 
        mesh.addTexCoord( [sin, cos] )

        # Left-side vertex
        mesh.addVertex( pos )
        if color:
            mesh.addColor( color )
        mesh.addNormal( normal )
        mesh.addTexCoord( [-sin, cos] )

    # Bottom vertex
    mesh.addVertex( pos )
    if color:
        mesh.addColor( color )
    mesh.addNormal( normal )
    mesh.addTexCoord( [0, -1] )

    # Indices
    for i in range(2 * (slices - 1)):
        mesh.addIndex(offset + i)
        mesh.addIndex(offset + i + 1)
        mesh.addIndex(offset + i + 2)


def halfcircle(mesh, pos=[0,0,0], num_vertices = 8, color = None):
    offset = len(mesh.vertices)  
    normal = [0., 0., 1.]  

    # Top vertex
    mesh.addVertex( pos )
    if color:
        mesh.addColor( color )
    mesh.addNormal( normal )
    mesh.addTexCoord( [0, 1] )  

    slices = int(math.floor(num_vertices / 2))
    for i in range(1, slices):
        sin = math.sin(i * math.pi * 0.5 / slices)
        cos = math.cos(i * math.pi * 0.5 / slices)

        # Right-side vertex
        mesh.addVertex( pos )
        if color:
            mesh.addColor( color )
        mesh.addNormal( normal ) 
        mesh.addTexCoord( [sin, cos] )

        # Left-side vertex
        mesh.addVertex( pos )
        if color:
            mesh.addColor( color )
        mesh.addNormal( normal )
        mesh.addTexCoord( [-sin, cos] )

    # # Bottom vertex
    # mesh.addVertex( pos )
    # if color:
    #     mesh.addColor( color )
    # mesh.addNormal( normal )
    # mesh.addTexCoord( [0, -1] )

    # Indices
    for i in range(2 * (slices - 1) - 1):
        mesh.addIndex(offset + i)
        mesh.addIndex(offset + i + 1)
        mesh.addIndex(offset + i + 2)


def square(mesh, pos=[0,0,0], color = None):
    offset = len(mesh.vertices)
    normal = [0., 0., 1.]

    w = math.ceil(2.0)
    h = math.ceil(2.0)
    for y in range(0, int(h)):
        for x in range(0, int(w)):
            mesh.addVertex( pos )
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
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from Meshes.Mesh import Mesh

def box(mesh, width, height, depth, resX, resY, resZ, color = None):
    resX = resX + 1
    resY = resY + 1
    resZ = resZ + 1

    width = float(width)
    height = float(height)
    depth = float(depth)

    if resX < 2:
        resX = 0
    if resY < 2:
        resY = 0
    if resZ < 2:
        resZ = 0

    # halves
    halfW = float(width * 0.5)
    halfH = float(height * 0.5)
    halfD = float(depth * 0.5)

    vert = [0.0, 0.0, 0.0]
    texcoord = [0.0, 0.0]
    normal = [0.0, 0.0, 0.0]
    vertOffset = len(mesh.vertices)

    # TRIANGLES

    #  Front Face
    normal = [0.0, 0.0, 1.0]

    # add the vertexes
    for iy in range(resY):
        for ix in range(resX):

            # normalized tex coords
            texcoord[0] = float(float(ix) / (float(resX) - 1.0))
            texcoord[1] = 1.0 - float(float(iy) / (float(resY) - 1.0))

            vert[0] = texcoord[0] * width - halfW
            vert[1] = -(texcoord[1] - 1.0) * height - halfH
            vert[2] = halfD;

            mesh.addVertex( np.array( vert ) ) 
            mesh.addTexCoord( np.array( texcoord ) )
            mesh.addNormal( np.array( normal ) )
            if color:
                mesh.addColor( color )
    

    for y in range(resY - 1):
        for x in range(resX - 1):

            # first triangle
            mesh.addIndex((y) * resX + x + vertOffset)
            mesh.addIndex((y) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)

            # second triangle
            mesh.addIndex((y) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)

    vertOffset = len(mesh.vertices)

    # Right Side Face
    normal = [1.0, 0.0, 0.0]

    # add the vertexes
    for iy in range(resY):
        for ix in range(resZ):
            # normalized tex coords 
            texcoord[0] = (float(ix) / (float(resZ) - 1.0))
            texcoord[1] = 1.0 - (float(iy) / (float(resY) - 1.0))

            vert[0] = halfW
            vert[1] = -(texcoord[1] - 1.0) * height - halfH
            vert[2] = texcoord[0] * -depth + halfD

            mesh.addVertex( np.array( vert ) ) 
            mesh.addTexCoord( np.array( texcoord ) )
            mesh.addNormal( np.array( normal ) )
            if color:
                mesh.addColor( color )
        

    for y in range(resY - 1):
        for x in range(resZ - 1):
            # first triangle
            mesh.addIndex((y) * resZ + x + vertOffset)
            mesh.addIndex((y) * resZ + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resZ + x + vertOffset)

            # second triangle
            mesh.addIndex((y) * resZ + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resZ + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resZ + x + vertOffset)

    vertOffset = len(mesh.vertices)

    # Left Side Face
    normal = [-1.0, 0.0, 0.0]

    # add the vertexes
    for iy in range(resY):
        for ix in range(resZ):

            # normalized tex coords
            texcoord[0] = (float(ix)/(float(resZ) - 1.0))
            texcoord[1] = 1.0 - (float(iy) / (float(resY) - 1.0))

            vert[0] = -halfW
            vert[1] = -(texcoord[1] - 1.0) * height - halfH
            vert[2] = texcoord[0] * depth - halfD

            mesh.addVertex( np.array( vert ) ) 
            mesh.addTexCoord( np.array( texcoord ) )
            mesh.addNormal( np.array( normal ) )
            if color:
                mesh.addColor( color )
        

    for y in range(resY - 1):
        for x in range(resZ - 1):
            # first triangle
            mesh.addIndex((y) * resZ + x + vertOffset)
            mesh.addIndex((y) * resZ + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resZ + x + vertOffset)

            # second triangle
            mesh.addIndex((y) * resZ + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resZ + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resZ + x + vertOffset)

    vertOffset = len(mesh.vertices)


    #  Back Face
    normal = [0.0, 0.0, -1.0]
    #  add the vertexes
    for iy in range(resY):
        for ix in range(resX):

            # normalized tex coords
            texcoord[0] = (float(ix) / (float(resX) - 1.0))
            texcoord[1] = 1.0 - (float(iy) / (float(resY) - 1.0))

            vert[0] = texcoord[0] * -width + halfW
            vert[1] = -(texcoord[1] - 1.0) * height - halfH
            vert[2] = -halfD

            mesh.addVertex( np.array( vert ) ) 
            mesh.addTexCoord( np.array( texcoord ) )
            mesh.addNormal( np.array( normal ) )
            if color:
                mesh.addColor( color )
        

    for y in range(resY - 1):
        for x in range(resX - 1):
            # first triangle
            mesh.addIndex((y) * resX + x + vertOffset)
            mesh.addIndex((y) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)

            # second triangle
            mesh.addIndex((y) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)

    vertOffset = len(mesh.vertices)


    #  Top Face
    normal = [0.0, -1.0, 0.0]
    #  add the vertexes
    for iy in range(resZ):
        for ix in range(resX):

            # normalized tex coords
            texcoord[0] = (float(ix) / (float(resX) - 1.0))
            texcoord[1] = 1.0 - (float(iy) / (float(resZ) - 1.0))

            vert[0] = texcoord[0] * width - halfW
            vert[1] = -halfH
            vert[2] = texcoord[1] * depth - halfD

            mesh.addVertex( np.array( vert ) ) 
            mesh.addTexCoord( np.array( texcoord ) )
            mesh.addNormal( np.array( normal ) )
            if color:
                mesh.addColor( color )
        

    for y in range(resZ - 1):
        for x in range(resX - 1):
            # first triangle
            mesh.addIndex((y) * resX + x + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)
            mesh.addIndex((y) * resX + x + 1 + vertOffset)

            # second triangle
            mesh.addIndex((y + 1) * resX + x + 1 + vertOffset)
            mesh.addIndex((y) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)
        

    vertOffset = len(mesh.vertices)


    #  Bottom Face
    normal = [0.0, 1.0, 0.0]
    #  add the vertexes
    for iy in range(resZ):
        for ix in range(resX):

            # normalized tex coords
            texcoord[0] = (float(ix) / (float(resX) - 1.0))
            texcoord[1] = 1.0 - (float(iy) / (float(resZ) - 1.0))

            vert[0] = texcoord[0] * width - halfW
            vert[1] = halfH
            vert[2] = texcoord[1] * -depth + halfD

            mesh.addVertex( np.array( vert ) ) 
            mesh.addTexCoord( np.array( texcoord ) )
            mesh.addNormal( np.array( normal ) )
            if color:
                mesh.addColor( color )
        

    for y in range(resZ - 1):
        for x in range(resX - 1):
            # first triangle
            mesh.addIndex((y) * resX + x + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)
            mesh.addIndex((y) * resX + x + 1 + vertOffset)

            # second triangle
            mesh.addIndex((y + 1) * resX + x + 1 + vertOffset)
            mesh.addIndex((y) * resX + x + 1 + vertOffset)
            mesh.addIndex((y + 1) * resX + x + vertOffset)
        
    return mesh

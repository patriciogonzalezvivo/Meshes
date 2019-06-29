#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from math import cos, sin, sqrt, atan2
from triangle import triangulate

from Meshes.quaternion import quat_from_axis, quat_mult
from Meshes.vector import perpendicular, normalize
from Meshes.tessellate import tessPolygon
from Meshes.path import circle2D
from Meshes.Mesh import Mesh

TAU = np.pi * 2
QUAD_TEXTCOORDS = [[0,0],[0,1],[1,1],[1,0]]

def toSphere (coord, radius):
    lngQuat = quat_from_axis(np.radians(coord[0]), (1, 0, 0))
    latQuat = quat_from_axis(np.radians(coord[1]), (0, 1, 0))
    level = quat_from_axis(np.radians(90), (0, 0, 1))
    
    return quat_mult(level, quat_mult(lngQuat, quat_mult(latQuat, (0, 0, radius))))

    
def spherePoint( mesh, position, radius = 1, point_size = None, color = None):
    offset = len(mesh.vertices)

    v = toSphere( position, radius )
    v_tmp = toSphere( [position[0]+1, position[1]+1], radius)

    theta = np.pi * .5 # 90 deg
    for i in range(4):
        a = np.pi + i * theta
        extrude_normal = perpendicular(v, v_tmp, a)
        normal = normalize(v)

        mesh.addTexCoord( QUAD_TEXTCOORDS[i] );

        if point_size:
            mesh.addNormal( normal );
            mesh.addVertex( v + extrude_normal * point_size )
        else:
            mesh.addNormal( extrude_normal );
            mesh.addVertex( v )

        if color:
            mesh.addColor( color )

    mesh.addTriangle( offset, offset + 1, offset + 2 )
    mesh.addTriangle( offset + 2, offset + 3, offset )

    return mesh


def sphereDot( mesh, position, radius = 1, dot_size = None, color = None):
    dot_radius = 0.0
    prev_mats = len(mesh.materials)

    if dot_size:
        dot_radius = dot_size

    dot_mesh = tessPolygon(Mesh('Dot-'+str(prev_mats)), circle2D(0, 0, dot_radius, 6), 0, color)

    #  Put on place
    dot_mesh.translateZ(radius)
    dot_mesh.rotateY(position[1])
    dot_mesh.rotateX(position[0])
    dot_mesh.rotateZ(90)
    
    # for i in range(len(dot_mesh.vertices)):
        # if dot_size == None:

    mesh.add(dot_mesh)

    return mesh


def sphereSpline( mesh, points, radius = 1, width = None, color = None):
    offset = len(mesh.vertices)

    for i in range(1, len(points)):
        v_prev = toSphere( points[i-1], radius)
        v_this = toSphere( points[i], radius)

        theta = np.pi * .5 # 90 deg
        for i in range(4):
            a = theta * .5 + i * theta
            normal = perpendicular(v_prev, v_this, a)

            mesh.addTexCoord( QUAD_TEXTCOORDS[i] );

            if width:
                mesh.addNormal( normalize(v_prev) );
                if i < 2:
                    mesh.addVertex( v_prev + np.array(normal) * width )
                else:
                    mesh.addVertex( v_this + np.array(normal) * width )
            else:
                mesh.addNormal( normal );
                if i < 2:
                    mesh.addVertex( v_prev )
                else:
                    mesh.addVertex( v_this )

            if color:
                mesh.addColor( color )
        
        mesh.addTriangle( offset, offset + 1, offset + 2 )
        mesh.addTriangle( offset + 2, offset + 3, offset )
        offset += 4

    return mesh


def spherePolygon( mesh, points, radius=1, color=None ):
    offset = len(mesh.vertices)

    if points[0][0] == points[-1][0]:
        points.pop()

    sphere_pts = []
    for point in points:
        sphere_pts.append(point)
        v = toSphere(point, radius)
        normal = normalize(v)
        
        mesh.vertices_texcoords([.5+point[0]/360., .5+point[1]/180.]);
        # mesh.addNormal( normal )

        mesh.addVertex( v )

        if color:
            mesh.addColor( color )

    segments = []
    for i in range( len(points) ):
        segments.append([i, (i + 1) % len(points) ] )

    cndt = triangulate(dict(vertices=sphere_pts,segments=segments),'p')
    for face in cndt['triangles']:
        mesh.addTriangle(   offset + ( face[0] % len(sphere_pts) ), 
                            offset + ( face[1] % len(sphere_pts) ), 
                            offset + ( face[2] % len(sphere_pts) ) )
    offset += len(points)

    return mesh


def sphere(mesh, radius=1, resolution=12, color=None):
    offset = len(mesh.vertices)

    doubleRes = resolution * 2
    polarInc = np.pi / float(resolution)    # ringAngle
    azimInc = TAU / float(doubleRes)        # segAngle

    vert = [ 0.0, 0.0 , 0.0 ]
    tcoord = [ 0.0, 0.0 ]

    for i in range( resolution + 1 ):
        tr = sin( np.pi - float(i) * polarInc )
        ny = cos( np.pi - float(i) * polarInc )

        tcoord[1] = (float(i) / float(resolution))

        for j in range( doubleRes + 1):
            nx = tr * sin(float(j) * azimInc)
            nz = tr * cos(float(j) * azimInc)

            tcoord[0] = float(j) / float(doubleRes)

            vert = np.array( [ nx, ny, nz ] )
            mesh.addNormal( vert )
            vert *= radius
            mesh.addVertex( vert )
            if color:
                mesh.addColor( color )
            mesh.addTexCoord( np.array(tcoord) )

    nr = doubleRes + 1
    for iy in range( resolution ):
        for ix in range( doubleRes ):
            # first tri
            if iy > 0:
                mesh.addIndex(offset + (iy+0) * (nr) + (ix+0)) # 1
                mesh.addIndex(offset + (iy+0) * (nr) + (ix+1)) # 2
                mesh.addIndex(offset + (iy+1) * (nr) + (ix+0)) # 3
                

            #second tri
            if iy < resolution-1:
                mesh.addIndex(offset + (iy+0) * (nr) + (ix+1)) # 1
                mesh.addIndex(offset + (iy+1) * (nr) + (ix+1)) # 2
                mesh.addIndex(offset + (iy+1) * (nr) + (ix+0)) # 3
                

    return mesh


# Port from C++ https://bitbucket.org/transporter/ogre-procedural/src/ca6eb3363a53c2b53c055db5ce68c1d35daab0d5/library/src/ProceduralIcoSphereGenerator.cpp?at=default&fileviewer=file-view-default
def icosphere(mesh, radius=1, resolution=2, color=None):

    # Step 1 : Generate icosahedron
    sqrt5 = sqrt(5.0);
    phi = (1.0 + sqrt5) * 0.5;
    invnorm = 1.0/sqrt(phi*phi+1.0);

    mesh.addVertex(invnorm * np.array([-1,  phi, 0]))  #0
    mesh.addVertex(invnorm * np.array([ 1,  phi, 0]))  #1
    mesh.addVertex(invnorm * np.array([0,   1,  -phi]))#2
    mesh.addVertex(invnorm * np.array([0,   1,   phi]))#3
    mesh.addVertex(invnorm * np.array([-phi,0,  -1]))  #4
    mesh.addVertex(invnorm * np.array([-phi,0,   1]))  #5
    mesh.addVertex(invnorm * np.array([ phi,0,  -1]))  #6
    mesh.addVertex(invnorm * np.array([ phi,0,   1]))  #7
    mesh.addVertex(invnorm * np.array([0,   -1, -phi]))#8
    mesh.addVertex(invnorm * np.array([0,   -1,  phi]))#9
    mesh.addVertex(invnorm * np.array([-1,-phi, 0]))  #10
    mesh.addVertex(invnorm * np.array([ 1,-phi, 0]))  #11

    if color:
        for i in range(12):
            mesh.addColor( color )
         
    firstFaces = [
        0,1,2,
        0,3,1,
        0,4,5,
        1,7,6,
        1,6,2,
        1,3,7,
        0,2,4,
        0,5,3,
        2,6,8,
        2,8,4,
        3,5,9,
        3,9,7,
        11,6,7,
        10,5,4,
        10,4,8,
        10,9,5,
        11,8,6,
        11,7,9,
        10,8,11,
        10,11,9
    ]

    for i in range(0, 60)[::3]:
        mesh.addTriangle(firstFaces[i], firstFaces[i+1], firstFaces[i+2])

    size = len(mesh.indices)
    # Step 2: tessellate
    for iteration in range(0, resolution):
        size*=4;
        newFaces = []
        for i in range(0, int(size/12)):
            i1 = mesh.indices[i*3]
            i2 = mesh.indices[i*3+1]
            i3 = mesh.indices[i*3+2]

            i12 = len(mesh.vertices)
            i23 = i12 + 1
            i13 = i12 + 2

            v1 = mesh.vertices[i1]
            v2 = mesh.vertices[i2]
            v3 = mesh.vertices[i3]

            # make 1 vertice at the center of each edge and project it onto the mesh
            mesh.vertices.append(np.array(normalize( v1 + v2 )))
            mesh.vertices.append(np.array(normalize( v2 + v3 )))
            mesh.vertices.append(np.array(normalize( v1 + v3 )))

            if color:
                for i in range(3):
                    mesh.addColor( color )

            # now recreate indices
            newFaces.append(i1)
            newFaces.append(i12)
            newFaces.append(i13)
            newFaces.append(i2)
            newFaces.append(i23)
            newFaces.append(i12)
            newFaces.append(i3)
            newFaces.append(i13)
            newFaces.append(i23)
            newFaces.append(i12)
            newFaces.append(i23)
            newFaces.append(i13)

        mesh.indices = list(newFaces);
    
    #  Step 3 : generate texcoords
    texCoords = []
    for i in range(0, len(mesh.vertices)):
        vec = mesh.vertices[i]
        r0 = sqrt(vec[0] * vec[0] + vec[2] * vec[2])
        alpha = atan2(vec[2], vec[0])

        u = alpha/TAU + .5;
        v = atan2(vec[1], r0)/np.pi + .5;

        # reverse the u coord, so the default is texture mapped left to
        # right on the outside of a sphere 
        # reverse the v coord, so that texture origin is at top left
        texCoords.append( np.array( [1.0-u, v ] ))

    # Step 4 : fix texcoords
    # find vertices to split
    indexToSplit = []
    for i in range(0, int(len(mesh.indices)/3)):
        t0 = texCoords[ mesh.indices[i*3+0] ]
        t1 = texCoords[ mesh.indices[i*3+1] ]
        t2 = texCoords[ mesh.indices[i*3+2] ]

        if abs(t2[0]-t0[0]) > 0.5:
            if t0[0] < 0.5:
                indexToSplit.append( mesh.indices[i*3] )
            else:
                indexToSplit.append( mesh.indices[i*3+2] )
        
        if abs(t1[0]-t0[0]) > 0.5:
            if t0[0] < 0.5:
                indexToSplit.append( mesh.indices[i*3] )
            else:
                indexToSplit.append( mesh.indices[i*3+1] )
        
        if abs(t2[0]-t1[0]) > 0.5:
            if t1[0] < 0.5:
                indexToSplit.append( mesh.indices[i*3+1] )
            else:
                indexToSplit.append( mesh.indices[i*3+2] )

    # split vertices
    for i in range(0, int(len(indexToSplit)/3)):
        index = indexToSplit[i]
        # duplicate vertex
        v = mesh.vertices[index]
        t = texCoords[index] + np.array( [1., 0.] )

        mesh.vertices.append(v)
        texCoords.append(t)

        if color:
           mesh.addColor( color )
        
        newIndex = len(mesh.vertices)-1

        # reassign indices
        for j in range(len(mesh.indices)):
            if mesh.indices[j] == index:
                index1 = mesh.indices[ int((j+1)%3+(j/3)*3) ]
                index2 = mesh.indices[ int((j+2)%3+(j/3)*3) ]
                if (texCoords[index1][0] > 0.5) or (texCoords[index2][0] > 0.5):
                    mesh.indices[j] = newIndex;

    for vert in mesh.vertices:
        mesh.addNormal( normalize(vert) )

    for st in texCoords:
        mesh.addTexCoord( st )

    for i in range(len( mesh.vertices )):
        mesh.vertices[i] = mesh.vertices[i] * radius

    return mesh


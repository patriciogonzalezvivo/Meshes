#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from math import sqrt
from triangle import triangulate

from Meshes.vector import perpendicular, normalize, dot
from Meshes.quaternion import axisangle_to_q, qv_mult
from Meshes.tools import boundingBox, remap

def tessSpline( mesh, positions, z = 0.0, line_width = 0.0, color = None, flipped = False):
    offset = len( mesh.vertices )

    UP_NORMAL = [0., 0., 1.]

    if flipped: 
        UP_NORMAL = [0., 0., -1.]

    # Right normal to segment between previous and current m_points
    normi = [0.0, 0.0, 0.0]
    # Right normal to segment between current and next m_points
    normip1 = [0.0, 0.0, 0.0]         
    # Right "normal" at current point, scaled for miter joint 
    rightNorm = [0.0, 0.0, 0.0]    
    
    # Make sure all points have XYZ
    points = []
    for i in range(0, len(positions)):
        if len(positions[0]) == 1:
            points.append( [positions[i][0], 0.0, z] )
        elif len(positions[0]) == 2:
            points.append( [positions[i][0], positions[i][1], z] )
        else:
            points.append( [positions[i][0], positions[i][1], positions[i][2]] )

    # Previous point coordinates
    im1 = [0.0, 0.0, 0.0]
    # Current point coordinates
    i0 = points[0]
    # Next point coordinates
    ip1 = points[1]
    
    # Get Perpendicular
    normip1[0] = ip1[1] - i0[1]
    normip1[1] = i0[0] - ip1[0]
    normip1[2] = 0.0
    
    normip1 = normalize( normip1 )
    rightNorm = normip1 * 0.8

    mesh.addVertex( i0 + np.array(rightNorm) * line_width )
    mesh.addTexCoord( [1.0, 0.0] )

    mesh.addVertex( i0 - np.array(rightNorm) * line_width )
    mesh.addTexCoord( [0.0, 0.0] )

    if line_width == 0.0:
        mesh.addNormal( rightNorm )
        mesh.addNormal( -rightNorm )
    else:
        mesh.addNormal( UP_NORMAL )
        mesh.addNormal( UP_NORMAL )

    if color:
        mesh.addColor( color )
        mesh.addColor( color )
    
    for i in range(1, len(points) - 1):
        im1 = i0
        i0 = ip1 
        ip1 = points[ i + 1 ]
        
        normi = normip1
        normip1[0] = ip1[1] - i0[1]
        normip1[1] = i0[0] - ip1[0]
        normip1[2] = 0.0
        normip1 = normalize( normip1 )
        rightNorm = normi + normip1

        if line_width == 0.0:
            scale = sqrt(2.0 / (1.0 + dot(normi, normip1) )) * 0.5
            rightNorm *= scale
            mesh.addVertex( i0 )
            mesh.addNormal( rightNorm )
            
            mesh.addVertex( i0 )
            mesh.addNormal( -rightNorm )
        else:
            scale = sqrt(2.0 / (1.0 + dot(normi, normip1) )) * line_width * 0.5;
            rightNorm *= scale;
            
            mesh.addVertex( i0 + rightNorm )
            mesh.addNormal( UP_NORMAL )
            
            mesh.addVertex( i0 - rightNorm )
            mesh.addNormal( UP_NORMAL )

        y_pct = float(i) / float( len( points ) - 1 )
        mesh.addTexCoord( [1.0, y_pct] );
        mesh.addTexCoord( [0.0, y_pct] );

        if color:
            mesh.addColor( color )
            mesh.addColor( color )


    # Get Perpendicular
    normip1[0] = ip1[1] - i0[1]
    normip1[1] = i0[0] - ip1[0]
    normip1[2] = 0.0
    normip1 = normalize( normip1 )
    rightNorm = normip1 * 0.8

    if line_width == 0.0:
        mesh.addVertex( ip1 )
        mesh.addNormal( rightNorm )

        mesh.addVertex( ip1 )
        mesh.addNormal( -rightNorm )
    else:
        mesh.addVertex(ip1 + rightNorm * line_width);
        mesh.addNormal(UP_NORMAL);
        mesh.addVertex(ip1 - rightNorm * line_width);
        mesh.addNormal(UP_NORMAL);

    mesh.addTexCoord( [1.0, 1.0] );
    mesh.addTexCoord( [0.0, 1.0] );

    if color:
        mesh.addColor( color )
        mesh.addColor( color )

    for i in range(0, len( points ) - 1 ):
        mesh.addIndex(offset + 2 * i + 3)
        mesh.addIndex(offset + 2 * i + 2)
        mesh.addIndex(offset + 2 * i )
        
        mesh.addIndex(offset + 2 * i )
        mesh.addIndex(offset + 2 * i + 1)
        mesh.addIndex(offset + 2 * i + 3)

    return mesh


def tessPolygon( mesh, positions, z, color = None, flipped = False):
    offset = len(mesh.vertices)

    n = [0., 0., 1.]

    # Get rid of begining/ending duplicate
    if positions[0][0] == positions[-1][0] and positions[0][1] == positions[-1][1]:
        positions.pop()

    rotY = []
    if flipped: 
        n = [0., 0., -1.]
        rotY = axisangle_to_q((0, 1, 0), np.radians(180))

    min_x, min_y, max_x, max_y = boundingBox(positions)

    points = []
    for point in positions:
        v = np.array([point[0], point[1], z])
        if flipped:
            v = qv_mult(rotY, (v[0], v[1], v[2]))

        points.append( [point[0], point[1]] )
        mesh.addTexCoord([ remap(point[0], min_x, max_x, 0.0, 1.0), remap(point[1], min_y, max_y, 0.0, 1.0) ])
        mesh.addVertex( v )

        mesh.addNormal( n )
        if color:
            mesh.addColor( color )

    segments = []
    for i in range( len(positions) ):
        segments.append([i, (i + 1) % len(positions) ] )

    cndt = triangulate(dict(vertices=points, segments=segments),'p')
    for face in cndt['triangles']:
        mesh.addTriangle(   offset + ( face[0] % len(points) ), 
                            offset + ( face[1] % len(points) ), 
                            offset + ( face[2] % len(points) ) )
    offset += len(positions)

    return mesh
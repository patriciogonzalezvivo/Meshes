#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from math import sqrt

from .vector import perpendicular, normalize, dot

def extrudeLine( mesh, positions, z = 0.0, line_width = 0.0, color = None, flipped = False):
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

    for i in range(len( points ) - 1 ):
        mesh.addIndex(offset + 2 * i + 3)
        mesh.addIndex(offset + 2 * i + 2)
        mesh.addIndex(offset + 2 * i )
        
        mesh.addIndex(offset + 2 * i )
        mesh.addIndex(offset + 2 * i + 1)
        mesh.addIndex(offset + 2 * i + 3)

    return mesh

def extrudePoly( mesh, positions, depth, z=0.0, color = None):
    offset = len( mesh.vertices )

    index = 0
    for p in positions:
        mesh.addVertex( [p[0], p[1], 0] )
        mesh.addVertex( [p[0], p[1], depth] )
        index += 2

    if index > 3:
        for i in range(index - 1 ):
            mesh.addIndex(offset + 2 * i + 3)
            mesh.addIndex(offset + 2 * i + 2)
            mesh.addIndex(offset + 2 * i )
            
            mesh.addIndex(offset + 2 * i )
            mesh.addIndex(offset + 2 * i + 1)
            mesh.addIndex(offset + 2 * i + 3)

    return mesh


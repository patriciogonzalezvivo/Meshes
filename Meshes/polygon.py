#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import numpy as np

from math import cos, sin, sqrt, atan2, fabs
from triangle import triangulate, plot as tplot

from Meshes.vector import axisangle_to_q, qv_mult
from Meshes.tools import boundingBox, remap

def polygon( mesh, positions, z, color = None, flipped = False):
    offset = len(mesh.vertices)

    n = [0., 0., 1.]

    if positions[0][0] == positions[-1][0]:
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
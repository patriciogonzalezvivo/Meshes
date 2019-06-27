#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from Meshes import Mesh, extrudeLine

num_points = 10

if len(sys.argv) > 1:
    if int(sys.argv[1]):
        num_points = int(sys.argv[1])

points = []
for i in range(0, num_points):
    points.append([i / (num_points - 1), 0.0])

mesh = Mesh("Spline")
mesh = extrudeLine(mesh, points, 0.0, 0.1)

mesh.toPly('spline_' + str(num_points * 2) + 'v.ply')
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Meshes import Mesh, tessSpline

points = [ [0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]]
mesh = Mesh("Spline")
mesh = tessSpline(mesh, points, 0.0, 0.1);

mesh.toObj('spline.obj')
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Meshes import Mesh, box

import sys

size = 10
precision = 1

if len(sys.argv) > 1:
    if int(sys.argv[1]):
        size = int(sys.argv[1])

if len(sys.argv) > 2:
    if int(sys.argv[2]):
        precision = int(sys.argv[2])


mesh = Mesh("Cube")
mesh = box(mesh, size, size, size, precision, precision, precision);

mesh.toPly('cube_' + str(len(mesh.vertices))+ 'v.ply')

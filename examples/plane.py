#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Meshes import Mesh, tessIsoRect

import sys

size = 100
precision = 1

if len(sys.argv) > 1:
    if int(sys.argv[1]):
        size = int(sys.argv[1])

if len(sys.argv) > 2:
    if int(sys.argv[2]):
        precision = int(sys.argv[2])


mesh = Mesh("Plane")
mesh = tessIsoRect(mesh, size, size, precision);

mesh.toPly('plane_' + str(len(mesh.vertices))+ 'v.ply')

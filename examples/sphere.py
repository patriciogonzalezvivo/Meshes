#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Meshes import Mesh, sphere

import sys

size = 1
precision = 2

if len(sys.argv) > 1:
    if int(sys.argv[1]):
        size = int(sys.argv[1])

if len(sys.argv) > 2:
    if int(sys.argv[2]):
        precision = int(sys.argv[2])


mesh = Mesh("Sphere")
mesh = sphere(mesh, size, precision, color=[1.0,0.0,0.0]);

print(mesh.totalVertices())
print(len(mesh.vertices_texcoords))
# print(mesh.vertexBuffer())
print(mesh.colorBuffer())
# print(mesh.normalBuffer())
# print(mesh.texCoordBuffer())
# print(mesh.indexBuffer())
# print(mesh.loopStartsBuffer())
# print(mesh.loopLengthsBuffer())

mesh.toPly('sphere_' + str(len(mesh.vertices))+ 'v.ply')

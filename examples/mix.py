#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Meshes import Mesh


mesh_a = Mesh("A")
mesh_a.fromPly('A.ply')

mesh_b = Mesh("B")
mesh_b.fromPly('B.ply')

mesh = Mesh("C")
mesh.add( mesh_a )
mesh.add( mesh_b )
mesh.toPly("C.ply")


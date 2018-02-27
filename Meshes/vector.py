#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from math import cos, sin, sqrt

def length(v):
    return sqrt(sum(n * n for n in v))

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)

def perpendicular( v1, v2, angle ):
    P1 = np.array( v1 )
    P2 = np.array( v2 )
    sphereCenter = np.array( [0, 0, 0] )
    R = np.cross( P2 - P1, sphereCenter - P1)
    S = np.cross( R, sphereCenter - P1)
    R = normalize( R )
    S = normalize( S )
    n = []
    for axis in range(3):
        n.append( R[axis] * cos(angle) + S[axis] * sin(angle) )
    return normalize( n )
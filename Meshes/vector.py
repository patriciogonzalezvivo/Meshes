#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from math import cos, sin, sqrt

def length(v):
    return sqrt(sum(n * n for n in v))

def dot(v1, v2):
    n = 0
    lim = min( len(v1) , len(v2) )
    for i in range(lim):
        n += v1[i] * v2[i]
    return n

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

def vec3(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


# TODO
#       - MAKE vec2 vec3 mat4  https://github.com/bengolder/python-geometry/tree/master/geometry
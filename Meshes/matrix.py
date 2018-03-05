#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from math import cos, sin, sqrt

def mv_mult(m, v):
    v4 = v
    if len(v) == 3:
        v4 = np.append(v, 1)
    v4 = v4.dot(m)
    if len(v) == 3:
        return np.delete(v4, 3)
    else:
        return v4

def mat4_rotateX(deg):
    rad = np.radians(deg)
    return np.array([
        [1, 0, 0, 0],
        [0, cos(rad), sin(rad), 0],
        [0, -sin(rad), cos(rad), 0],
        [0, 0, 0, 1]
    ])

def mat4_rotateY(deg):
    rad = np.radians(deg)
    return np.array([
        [cos(rad), 0, -sin(rad), 0],
        [0, 1, 0, 0],
        [sin(rad), 0, cos(rad), 0],
        [0, 0, 0, 1]
    ])

def mat4_rotateZ(deg):
    rad = np.radians(deg)
    return np.array([
        [cos(rad), sin(rad), 0, 0],
        [-sin(rad), cos(rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def mat4_translateX(d):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [d, 0, 0, 1]
    ])

def mat4_translateY(d):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, d, 0, 1]
    ])

def mat4_translateZ(d):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, d, 1]
    ])

def mat4_scale(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from math import cos, sin, sqrt, atan2, fabs

def boundingBox(points):
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    for point in points:
        if point[0] < min_x:
            min_x = point[0]
        if point[1] < min_y:
            min_y = point[1]

        if point[0] > max_x:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
    return min_x, min_y, max_x, max_y

def remap(value, in_min, in_max, out_min, out_max):
    in_span = in_max - in_min
    out_span = out_max - out_min

    value = float(value - in_min)
    if value != 0:
        value /= float(in_span)
    return out_min + (value * out_span)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import numpy as np

from math import cos, sin, sqrt, atan2, fabs

def rect2D(width, height, centered = False):
    left    = 0
    right   = width
    bottom  = 0
    top     = height

    if centered:
        left    = -width/2
        right   = width/2
        bottom  = -height/2
        top     = height/2
    
    return [ [left,bottom], [left,top], [right,top], [right,bottom] ]

def rectRound2D(width, height, radius, centered = False, resolution = 36):
    left    = 0
    right   = width
    bottom  = 0
    top     = height
    corners = [radius, radius, radius, radius]

    if centered:
        left    = -width/2
        right   = width/2
        bottom  = -height/2
        top     = height/2

    points = []

    a = -np.pi

    x = left + corners[0]
    y = bottom + corners[0]
    while a < - np.pi / 2.:
        a += np.pi / resolution
        points.append([
            x + cos(a) * corners[0],
            y + sin(a) * corners[0]
            ])

    x = right - corners[1]
    y = bottom + corners[1]
    while a < 0:
        a += np.pi / resolution
        points.append([
            x + cos(a) * corners[1],
            y + sin(a) * corners[1]
            ])

    x = right - corners[2]
    y = top - corners[2]
    while a < np.pi / 2.:
        a += np.pi / resolution
        points.append([
            x + cos(a) * corners[2],
            y + sin(a) * corners[2]
            ])

    x = left + corners[3]
    y = top - corners[3]
    while a < np.pi:
        a += np.pi / resolution
        points.append([
            x + cos(a) * corners[3],
            y + sin(a) * corners[3]
            ])

    return points
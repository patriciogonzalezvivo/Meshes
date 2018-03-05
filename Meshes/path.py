#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from math import cos, sin

def rect2D(x, y, width, height, centered = False):
    left    = x
    right   = x + width
    bottom  = y
    top     = y + height

    if centered:
        left    = x-width/2
        right   = x+width/2
        bottom  = y-height/2
        top     = y+height/2
    
    return [ [left,bottom], [left,top], [right,top], [right,bottom] ]

def rectRound2D(x, y, width, height, radius, centered = False, resolution = 36):
    left    = x
    right   = x + width
    bottom  = y
    top     = y + height
    corners = [radius, radius, radius, radius]

    if centered:
        left    = x-width/2
        right   = x+width/2
        bottom  = y-height/2
        top     = y+height/2

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

def circle2D(x, y, radius, resolution = 36):
    a = -np.pi
    points = []
    while a < np.pi:
        a += np.pi / resolution
        points.append([
            x + cos(a) * radius,
            y + sin(a) * radius
            ])

    return points
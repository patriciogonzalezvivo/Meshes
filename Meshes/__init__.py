# outer __init__.py

# Mesh/Material Classes
from .Mesh import Mesh
from .Material import Material

# 3D Primitives
from .box import box
from .sphere import sphere, spherePoint, sphereDot, sphereSpline, spherePolygon, icosphere
from .billboards import circle, halfcircle, square

# 2D primitives
from .path import rect2D, rectRound2D, circle2D

# Functions
from .tessellate import tessRect, tessIsoRect, tessPolygon
from .extrude import extrudeLine, extrudePoly

from .vector import *
from .matrix import *
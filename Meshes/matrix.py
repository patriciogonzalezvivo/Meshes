#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from math import cos, sin, sqrt
from .vector import vec3

def mat4_mult(m, v):
    v4 = v
    if len(v) == 3:
        v4 = np.append(v, 1)
    v4 = v4.dot(m)
    if len(v) == 3:
        return np.delete(v4, 3)
    else:
        return v4

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

# https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_translate( direction ):
    """Return matrix to translate by direction vector.

    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M.transpose() 

# def mat4_scale(val):
#     s = [1.0, 1.0, 1.0]

#     if isinstance(val, tuple) or isinstance(val, list):
#         for i in range(min(len(val), 3)):
#             s[i] = val[i]
#     else:
#          for i in range(3):
#              s[i] = val
             
#     return np.array([
#         [s[0], 0, 0, 0],
#         [0, s[1], 0, 0],
#         [0, 0, s[2], 0],
#         [0, 0, 0, 1]
#     ])

# https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_scale(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (np.random.rand(4, 5) - 0.5) * 20
    >>> v[3] = 1
    >>> S = scale_matrix(-1.234)
    >>> np.allclose(np.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if direction is None:
        # uniform scaling
        M = np.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = vec3(direction[:3])
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M

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

#  https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_rotate(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = sin(angle)
    cosa = cos(angle)
    direction = vec3(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(  [[ 0.0,         -direction[2],  direction[1]],
                    [ direction[2], 0.0,          -direction[0]],
                    [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M.transpose() 


def mat4_from_A_to_B(A, B):
    v = np.cross(A, B)
    u = v/np.linalg.norm(v)
    c = np.dot(A, B)
    h = (1 - c)/(1 - c**2)

    vx, vy, vz = v
    return  np.array(   [[c + h*vx**2,   h*vx*vy - vz,   h*vx*vz + vy,   0.0],
                        [h*vx*vy+vz,    c+h*vy**2,      h*vy*vz-vx,     0.0],
                        [h*vx*vz - vy,  h*vy*vz + vx,   c+h*vz**2,      0.0],
                        [0.0,           0.0,            0.0,            1.0] ] ).transpose()


#  https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_from_euler(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = mat4_from_euler(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = mat4_from_euler(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = mat4_from_euler(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = mat4_from_euler(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M.transpose() 


# https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_from_quat(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = mat4_from_quat([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = mat4_from_quat([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = mat4_from_quat([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]]).transpose() 


# https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_inverse(matrix):
    """Return inverse of square transformation matrix.

    >>> M0 = random_rotation_matrix()
    >>> M1 = inverse_matrix(M0.T)
    >>> np.allclose(M1, np.linalg.inv(M0.T))
    True
    >>> for size in range(1, 7):
    ...     M0 = np.random.rand(size, size)
    ...     M1 = inverse_matrix(M0)
    ...     if not np.allclose(M1, np.linalg.inv(M0)): print(size)

    """
    return np.linalg.inv(matrix)


# https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
def mat4_projection(    point, normal, direction=None,
                        perspective=None, pseudo=False  ):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = mat4_projection([0, 0, 0], [1, 0, 0])
    >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
    True
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = mat4_projection(point, normal)
    >>> P1 = mat4_projection(point, normal, direction=direct)
    >>> P2 = mat4_projection(point, normal, perspective=persp)
    >>> P3 = mat4_projection(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, np.dot(P0, P3))
    True
    >>> P = mat4_projection([3, 0, 0], [1, 1, 0], [1, 0, 0])
    >>> v0 = (np.random.rand(4, 5) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(P, v0)
    >>> np.allclose(v1[1], v0[1])
    True
    >>> np.allclose(v1[0], 3-v1[1])
    True

    """
    M = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = vec3(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64,
                                  copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective-point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective+normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = np.array(direction[:3], dtype=np.float64, copy=False)
        scale = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M.transpose() 

def mat4_clip(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustum.

    The frustum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustum.

    If perspective is True the frustum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (divided by w coordinate).

    >>> frustum = np.random.rand(6)
    >>> frustum[1] += frustum[0]
    >>> frustum[3] += frustum[2]
    >>> frustum[5] += frustum[4]
    >>> M = mat4_clip(perspective=False, *frustum)
    >>> np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    array([-1., -1., -1.,  1.])
    >>> np.dot(M, [frustum[1], frustum[3], frustum[5], 1])
    array([ 1.,  1.,  1.,  1.])
    >>> M = mat4_clip(perspective=True, *frustum)
    >>> v = np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = np.dot(M, [frustum[1], frustum[3], frustum[4], 1])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError('invalid frustum')
    if perspective:
        if near <= _EPS:
            raise ValueError('invalid frustum: near <= 0')
        t = 2.0 * near
        M = [[t/(left-right), 0.0, (right+left)/(right-left), 0.0],
             [0.0, t/(bottom-top), (top+bottom)/(top-bottom), 0.0],
             [0.0, 0.0, (far+near)/(near-far), t*far/(far-near)],
             [0.0, 0.0, -1.0, 0.0]]
    else:
        M = [[2.0/(right-left), 0.0, 0.0, (right+left)/(left-right)],
             [0.0, 2.0/(top-bottom), 0.0, (top+bottom)/(bottom-top)],
             [0.0, 0.0, 2.0/(far-near), (far+near)/(near-far)],
             [0.0, 0.0, 0.0, 1.0]]
    return np.array(M).transpose() 


def mat4_orthogonal(lengths, angles):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in degrees.

    The de-orthogonalization matrix is the inverse.

    >>> O = mat4_orthogonal([10, 10, 10], [90, 90, 90])
    >>> np.allclose(O[:3, :3], np.identity(3, float) * 10)
    True
    >>> O = mat4_orthogonal([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> np.allclose(np.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = np.radians(angles)
    sina, sinb, _ = np.sin(angles)
    cosa, cosb, cosg = np.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return np.array([
        [ a*sinb*sqrt(1.0-co*co),  0.0,    0.0, 0.0],
        [-a*sinb*co,                    b*sina, 0.0, 0.0],
        [ a*cosb,                       b*cosa, c,   0.0],
        [ 0.0,                          0.0,    0.0, 1.0]]).transpose() 


def mat4_compose(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """Return transformation matrix from sequence of transformations.

    This is the inverse of the demat4_compose function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    >>> scale = numpy.random.random(3) - 0.5
    >>> shear = numpy.random.random(3) - 0.5
    >>> angles = (numpy.random.random(3) - 0.5) * (2*math.pi)
    >>> trans = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(4) - 0.5
    >>> M0 = mat4_compose(scale, shear, angles, trans, persp)
    >>> result = demat4_compose(M0)
    >>> M1 = mat4_compose(*result)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.identity(4)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]
        M = np.dot(M, P)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]
        M = np.dot(M, T)
    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = np.dot(M, R)
    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)
    M /= M[3, 3]
    return M.transpose() 
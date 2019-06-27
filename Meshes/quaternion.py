#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from math import cos, sin, acos, pi, sqrt
from Meshes.vector import vec3 # ,normalize


def quat_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quat_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = cos(ai)
    si = sin(ai)
    cj = cos(aj)
    sj = sin(aj)
    ck = cos(ak)
    sk = sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


def quat_from_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quat_from_axis(0.123, [1, 0, 0])
    >>> np.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vec3(q)
    if qlen > _EPS:
        q *= sin(angle/2.0) / qlen
    q[0] = cos(angle/2.0)
    return q


# def axisangle_to_quat(v, theta):
#     v = normalize(v)
#     x, y, z = v
#     theta /= 2
#     w = cos(theta)
#     x = x * sin(theta)
#     y = y * sin(theta)
#     z = z * sin(theta)
#     return w, x, y, z


def quat_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quat_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quat_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quat_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quat_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quat_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quat_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quat_from_matrix(R, isprecise=False),
    ...                    quat_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quat_from_matrix(R, isprecise=False),
    ...                    quat_from_matrix(R, isprecise=True))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q

# def quat_mult(q1, q2):
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#     return w, x, y, z

# def quat_mult(q1, v1):
#     q2 = (0.0,) + v1
#     return quat_mult(quat_mult(q1, q2), quat_conjugate(q1))[1:]

def quat_mult(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quat_mult([4, 1, -2, 3], [8, -5, 6, 7])
    >>> np.allclose(q, [28, -44, -14, 48])
    True

    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([    -x1*x0 - y1*y0 - z1*z0 + w1*w0,
                            x1*w0 + y1*z0 - z1*y0 + w1*x0,
                            -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                            x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)


# def quat_conjugate(q):
#     w, x, y, z = q
#     return (w, -x, -y, -z)

def quat_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q


def quat_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [1, 0, 0, 0])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def quat_real(quaternion):
    """Return real part of quaternion.

    >>> quat_real([3, 0, 1, 2])
    3.0

    """
    return float(quaternion[0])


def quat_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quat_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)


def quat_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.

    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quat_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = acos(np.dot(q0, q))
    >>> np.allclose(2, acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, acos(-np.dot(q0, q1)) / angle)
    True

    """
    q0 = vec3(quat0[:4])
    q1 = vec3(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = acos(d) + spin * pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / sin(angle)
    q0 *= sin((1.0 - fraction) * angle) * isin
    q1 *= sin(fraction * angle) * isin
    q0 += q1
    return q0

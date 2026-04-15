"""Library of optical surfaces"""

import numpy as np
import scipy
import itertools
from optics.raytracing import OpticalSurface
from optics.utils import (
    closest_point_on_polynomial_graph,
    normal_to_polynomial_graph,
    )

class Rectangle(OpticalSurface):
    def __init__(self, origin, extents, name = 'rectangle'):
        OpticalSurface.__init__(self, name)
        self.origin = np.array(origin)
        self.extents = np.array(extents)
        normal = np.cross(extents[0], extents[1])
        normal = normal / np.linalg.norm(normal)
        self.normalVec = np.array(normal)

    def drawing(self, projection_matrix):
        pts3d = np.array([self.origin+x*self.extents[0]/2+y*self.extents[1]/2
            for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
        pts2d = (projection_matrix @ pts3d.T).T
        return [pts2d]

    def distance(self,pt):
        z = np.dot(pt - self.origin, self.normalVec)
        xs = [np.dot(pt - self.origin, v) / np.linalg.norm(v)
            for v in self.extents]
        xs = [max(0, abs(x) - np.linalg.norm(v) / 2)
            for x, v in zip(xs, self.extents)]
        return np.linalg.norm(xs + [z])

    def normal(self,pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 0.5 * max([np.linalg.norm(v) for v in self.extents])
        return self.origin - radius, self.origin + radius

class Plane(OpticalSurface):
    def __init__(self,name='plane',origin=None,normal=None):
        if origin is None:
            origin=np.zeros(3)
        if normal is None:
            normal=np.array([1.0, 0.0, 0.0])
        OpticalSurface.__init__(self,name)
        self.origin = np.array(origin)
        self.normalVec = np.array(normal)

    def drawing(self, projection_matrix):
        tangent1 = np.cross(self.normalVec, [0., 0., 1.])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(self.normalVec, tangent1)
        extents = [100.0*tangent1, 100.0*tangent2]
        pts3d = np.array([self.origin+x*extents[0]/2+y*extents[1]/2
            for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
        pts2d = (projection_matrix @ pts3d.T).T
        return [pts2d]

    def distance(self, pt):
        pt = np.array(pt)
        return abs(np.dot(pt-self.origin, self.normalVec))

    def normal(self,pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin-radius, self.origin+radius

class Pinhole(OpticalSurface):
    def __init__(self, inner_radius, outer_radius,
            name='pinhole', origin = None, normal = None):
        if origin is None:
            origin=np.zeros(3)
        if normal is None:
            normal=np.array([1.0, 0.0, 0.0])
        OpticalSurface.__init__(self,name)
        self.origin = np.array(origin)
        self.normalVec = np.array(normal)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def drawing(self, projection_matrix):
        tangent1 = np.cross(self.normalVec, [0., 0., 1.])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(self.normalVec, tangent1)
        pts = np.array([tangent1*np.cos(u)+tangent2*np.sin(u)
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dA = (projection_matrix @ (pts*self.inner_radius+self.origin).T).T
        pts2dB = (projection_matrix @ (pts*self.outer_radius+self.origin).T).T
        return [pts2dA, pts2dB]

    def distance(self, pt):
        pt = np.array(pt)
        d = np.dot(self.normalVec, pt - self.origin)
        ptp = pt - self.normalVec * d
        r = np.linalg.norm(ptp - self.origin)
        if r > self.outer_radius:
            return np.sqrt(d ** 2 + (r - self.outer_radius) ** 2)
        elif r > self.inner_radius:
            return abs(d)
        else:
            return np.sqrt(d ** 2 + (r - self.inner_radius) ** 2)

    def normal(self, pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin - radius, self.origin + radius

class Slit(OpticalSurface):
    def __init__(self, width, length, outer_radius,
            name='slit', origin = None, normal = None, slit_direction = None):
        if origin is None:
            origin=np.zeros(3)
        if normal is None:
            normal=np.array([1.0, 0.0, 0.0])
        if slit_direction is None:
            slit_direction=np.array([0.0, 1.0, 0.0])
        OpticalSurface.__init__(self,name)
        self.origin = np.array(origin)
        self.normalVec = np.array(normal)
        self.dirVec = np.array(slit_direction)
        self.width = width
        self.length = length
        self.outer_radius = outer_radius

    def drawing(self, projection_matrix):
        ex = self.dirVec
        ey = np.cross(self.normalVec, self.dirVec)
        pts = np.array([ex*x*self.width/2+ey*y*self.length/2
            for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
        pts2dA = (projection_matrix @ (pts*self.outer_radius+self.origin).T).T
        pts = np.array([ex*np.cos(u)+ey*np.sin(u)
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dB = (projection_matrix @ (pts*self.outer_radius+self.origin).T).T
        return [pts2dA, pts2dB]

    def distance(self, pt):
        pt = np.array(pt)
        d = np.dot(self.normalVec, pt - self.origin)
        ptp = pt - self.normalVec * d
        x = np.dot(self.dirVec, ptp)
        y = np.dot(np.cross(self.normalVec, self.dirVec), ptp)
        r = np.sqrt(x**2 + y**2)
        if abs(x) < self.width/2 and abs(y) < self.length/2:
            return np.sqrt(d ** 2 + min(abs(abs(x)-self.width/2), abs(abs(y)-self.length/2))**2)
        elif r < self.outer_radius:
            return abs(d)
        else:
            return np.sqrt(d ** 2 + (r - self.outer_radius) ** 2)

    def normal(self, pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin - radius, self.origin + radius


class Sphere(OpticalSurface):
    def __init__(self, name='sphere', origin=None, radius=1.0):
        if origin is None:
            origin=np.zeros(3)
        OpticalSurface.__init__(self, name)
        self.origin = np.array(origin)
        self.radius = radius

    def drawing(self, projection_matrix):
        ez = np.cross(*projection_matrix)
        ez = ez / np.linalg.norm(ez)
        ex = projection_matrix[0] / np.linalg.norm(projection_matrix[0])
        ey = np.cross(ez, ex)
        pts = np.array([ex*np.cos(u)+ey*np.sin(u)
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2d = (projection_matrix @ (pts*self.radius+self.origin).T).T
        return [pts2d]

    def distance(self, pt):
        pt = np.array(pt)
        return abs(np.linalg.norm(pt-self.origin)-self.radius)

    def normal(self, pt):
        x = pt-self.origin
        return x/np.linalg.norm(x)

    def bbox(self):
        return self.origin-self.radius,self.origin+self.radius

class SphericalCap(OpticalSurface):
    def __init__(self, name="spherical cap", origin=None, invRadius=1.0, direction=None, r=0.5):
        if origin is None:
            origin=np.zeros(3)
        if direction is None:
            direction=np.array([1.0, 0.0, 0.0])
        OpticalSurface.__init__(self,name)
        self.origin = origin
        self.invRadius = invRadius
        self.r = r
        self.direction = direction

    def drawing(self, projection_matrix):
        ex = np.cross(self.direction, [0.0, 0.0, 1.0])
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(self.direction, ex)
        ez = -self.direction

        R = np.linspace(-self.r, self.r, 101)
        Z = R**2*self.invRadius/(1+np.sqrt(1-(self.invRadius*R)**2))

        pts = np.array([ex*r+ez*z for r, z in zip(R, Z)])
        pts2dA = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ey*r+ez*z for r, z in zip(R, Z)])
        pts2dB = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ex*R[-1]*np.cos(u)+ey*R[-1]*np.sin(u)+ez*Z[-1]
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dC = (projection_matrix @ (pts+self.origin).T).T
        return [pts2dA, pts2dB, pts2dC]

    @property
    def dz(self):
        return self.r**2*\
            self.invRadius/(1+np.sqrt(1-(self.invRadius*self.r)**2))

    def distance(self, pt):
        pt = np.array(pt)
        y = -np.dot(self.direction, pt-self.origin)
        x = np.linalg.norm(pt-self.origin+y*self.direction)
        A = (x**2+y**2)*self.invRadius-2*y
        A = A/(np.sqrt((x*self.invRadius)**2+(1-y*self.invRadius)**2)+1)
        y0 = self.r**2*self.invRadius/(
            1+np.sqrt(1-(self.r*self.invRadius)**2))
        if x/(A*self.invRadius+1)<self.r and y*self.invRadius<1.0:
            return abs(A)
        else:
            return np.sqrt((y0-y)**2+(x-self.r)**2)

    def normal(self, pt):
        pt = np.array(pt)
        y = -np.dot(self.direction, pt-self.origin)
        x = np.linalg.norm(pt-self.origin+y*self.direction)
        ex = (pt-self.origin+y*self.direction)/x
        s = x*self.invRadius
        c = np.sqrt(1-s**2)
        return c*self.direction+s*ex

    def bbox(self):
        y0 = self.r**2*self.invRadius/(
            1+np.sqrt(1-(self.r*self.invRadius)**2))
        pt = self.origin-self.direction*y0
        return pt-self.r, pt+self.r

class PolynomialCap(OpticalSurface):
    def __init__(
            self,
            coefs,
            name='polynomial cap',
            origin=None,
            direction=None,
            r=0.5):
        if origin is None:
            origin=np.zeros(3)
        if direction is None:
            direction=np.array([1.0, 0.0, 0.0])
        OpticalSurface.__init__(self, name)
        self.coefs = np.array(coefs)
        self.origin = origin
        self.r = r
        self.direction = direction

    def drawing(self, projection_matrix):
        ex = np.cross(self.direction, [0.0, 0.0, 1.0])
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(self.direction, ex)
        ez = -self.direction

        R = np.linspace(-self.r, self.r, 101)
        Z = np.polyval(self.coefs[::-1], R)

        pts = np.array([ex*r+ez*z for r, z in zip(R, Z)])
        pts2dA = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ey*r+ez*z for r, z in zip(R, Z)])
        pts2dB = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ex*R[-1]*np.cos(u)+ey*R[-1]*np.sin(u)+ez*Z[-1]
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dC = (projection_matrix @ (pts+self.origin).T).T
        return [pts2dA, pts2dB, pts2dC]

    @property
    def dz(self):
        return np.polyval(self.coefs[::-1], self.r)

    def distance(self, pt):
        pt = np.array(pt)
        y = -np.dot(self.direction, pt-self.origin)
        x = np.linalg.norm(pt-self.origin+y*self.direction)
        return closest_point_on_polynomial_graph(
            [x, y],
            self.coefs,
            [-self.r, self.r]
            )[2]

    def normal(self, pt):
        pt = np.array(pt)
        ey = -self.direction
        y = np.dot(ey, pt-self.origin)
        x = np.linalg.norm(pt-self.origin-y*ey)
        ex = (pt-self.origin-y*ey)/x
        n = normal_to_polynomial_graph(self.coefs, x)
        return n[0]*ex + n[1]*ey

    def bbox(self):
        pt = self.origin
        return pt-self.r, pt+self.r

class EvenAsphere(PolynomialCap):
    def __init__(
            self,
            name='even asphere',
            origin=None,
            invRadius=1.0,
            coefs=None,
            direction=None,
            r=0.5):
        if coefs is None:
            coefs = np.zeros(12)
        if isinstance(coefs, dict):
            coefs_arr = np.zeros(max(coefs.keys()))
            for key, value in coefs.items():
                coefs_arr[key-1] = value
            coefs = coefs_arr

        for n in range(1, len(coefs)+1):
            coefs[n-1] += (
                scipy.special.factorial2(2*n-1) * invRadius**(2*n-1) /
                ((2*n-1)*scipy.special.factorial2(2*n))
                )

        coefs_all = np.zeros(2*len(coefs)+2)
        coefs_all[2::2] = coefs

        PolynomialCap.__init__(
            self,
            coefs_all,
            name=name,
            origin=origin,
            direction=direction,
            r=r,
            )

class ConicalSlice(OpticalSurface):
    def __init__(self, name="conical slice", origin=None, r1=1.0, r2=1.0, h=1.0, direction=None):
        if origin is None:
            origin=np.zeros(3)
        if direction is None:
            direction=np.array([1.0, 0.0, 0.0])
        OpticalSurface.__init__(self, name)
        self.origin = np.array(origin)
        self.r1 = r1
        self.r2 = r2
        self.h = h
        self.direction = direction

    def drawing(self, projection_matrix):
        ex = np.cross(self.direction, [0.0, 0.0, 1.0])
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(self.direction, ex)
        ez = self.direction

        pts = np.array([ex*self.r1*np.cos(u)+ey*self.r1*np.sin(u)-ez*self.h/2
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dA = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ex*self.r2*np.cos(u)+ey*self.r2*np.sin(u)+ez*self.h/2
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dB = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ex*self.r1-ez*self.h/2, ex*self.r2+ez*self.h/2])
        pts2dC1 = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([-ex*self.r1-ez*self.h/2, -ex*self.r2+ez*self.h/2])
        pts2dC2 = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ey*self.r1-ez*self.h/2, ey*self.r2+ez*self.h/2])
        pts2dC3 = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([-ey*self.r1-ez*self.h/2, -ey*self.r2+ez*self.h/2])
        pts2dC4 = (projection_matrix @ (pts+self.origin).T).T
        return [pts2dA, pts2dB, pts2dC1, pts2dC2, pts2dC3, pts2dC4]

    def distance(self, pt):
        pt = np.array(pt)
        z = np.dot(self.direction, pt-self.origin)
        R = np.linalg.norm(pt-self.origin)
        r = np.sqrt(R**2-z**2)
        dx = np.array([z,r-(self.r1+self.r2)/2])
        ex = np.array([self.h, self.r2-self.r1])
        ex = ex/np.linalg.norm(ex)
        ey = np.array([[0, -1], [1, 0]]).dot(ex)
        X = np.dot(ex,dx)
        L = np.sqrt(self.h**2+(self.r1-self.r2)**2)
        Y = abs(np.dot(ey, dx))
        if abs(X)<L/2:
            return Y
        else:
            return np.sqrt((abs(X)-L/2)**2+Y**2)

    def normal(self, pt):
        pt = np.array(pt)
        ey=pt-self.origin-np.dot(self.direction,pt-self.origin)*self.direction
        ey = ey/np.linalg.norm(ey)
        n = self.direction*(self.r1-self.r2)+ey*self.h
        n = n/np.linalg.norm(n)
        return n

    def bbox(self):
        R = np.sqrt(max(self.r1,self.r2)**2+(self.h/2)**2)
        return self.origin-R,self.origin+R

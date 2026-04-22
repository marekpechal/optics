"""Library of optical surfaces"""

import numpy as np
import scipy
import itertools
from optics.raytracing import OpticalSurface, Ray
from optics.utils import (
    closest_point_on_polynomial_graph,
    normal_to_polynomial_graph,
    projector_perpendicular,
    quadratic_real_solutions,
    coefs_of_composition,
    )

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

    def ray_intersection(self, ray: Ray):
        a = np.dot(self.normalVec, ray.direction)
        b = np.dot(self.normalVec, self.origin - ray.origin)
        if a != 0:
            t = b / a
            if t >= 0: # intersection in front of the ray origin
                return t
            else: # intersection begind the ray origin
                return None
        else:
            if b == 0: # ray lies in plane -> infinitely many intersections
                return 0.0
            else: # ray parallel to plane but not in it -> no intersections
                return None

    def normal(self,pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin-radius, self.origin+radius

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

    def ray_intersection(self, ray: Ray):
        t = Plane.ray_intersection(self, ray)
        if t is not None:
            pt = ray.origin + t * ray.direction
            if all([abs(np.dot(u, pt - self.origin)) < np.dot(u, u)/2
                    for u in self.extents]):
                return t
        return None

    def normal(self,pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 0.5 * max([np.linalg.norm(v) for v in self.extents])
        return self.origin - radius, self.origin + radius

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

    def ray_intersection(self, ray: Ray):
        t = Plane.ray_intersection(self, ray)
        if t is not None:
            pt = ray.origin + t * ray.direction
            r = np.linalg.norm(pt - self.origin)
            if (self.inner_radius < r < self.outer_radius):
                return t
        return None

    def normal(self, pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin - radius, self.origin + radius

class Disk(OpticalSurface):
    def __init__(self, radius,
            name='disk', origin = None, normal = None):
        if origin is None:
            origin=np.zeros(3)
        if normal is None:
            normal=np.array([1.0, 0.0, 0.0])
        OpticalSurface.__init__(self, name)
        self.origin = np.array(origin)
        self.normalVec = np.array(normal)
        self.radius = radius

    def drawing(self, projection_matrix):
        tangent1 = np.cross(self.normalVec, [0., 0., 1.])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(self.normalVec, tangent1)
        pts = np.array([tangent1*np.cos(u)+tangent2*np.sin(u)
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2d = (projection_matrix @ (pts*self.radius+self.origin).T).T
        return [pts2d]

    def ray_intersection(self, ray: Ray):
        t = Plane.ray_intersection(self, ray)
        if t is not None:
            pt = ray.origin + t * ray.direction
            if np.linalg.norm(pt - self.origin) < self.radius:
                return t
        return None

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
        self.perpVec = np.cross(self.normalVec, self.dirVec)
        self.width = width
        self.length = length
        self.outer_radius = outer_radius

    def drawing(self, projection_matrix):
        ex = self.dirVec
        ey = self.perpVec
        pts = np.array([ex*x*self.width/2+ey*y*self.length/2
            for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
        pts2dA = (projection_matrix @ (pts+self.origin).T).T
        pts = np.array([ex*np.cos(u)+ey*np.sin(u)
            for u in np.linspace(0, 2*np.pi, 81)])
        pts2dB = (projection_matrix @ (pts*self.outer_radius+self.origin).T).T
        return [pts2dA, pts2dB]

    def ray_intersection(self, ray: Ray):
        t = Plane.ray_intersection(self, ray)
        if t is not None:
            pt = ray.origin + t * ray.direction
            if (np.linalg.norm(pt - self.origin) < self.outer_radius and not
                    all([abs(np.dot(u, pt - self.origin)) < a/2 for u, a in
                    [(self.dirVec, self.width), (self.perpVec, self.length)]])):
                return t
        return None

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

    def ray_intersection(self, ray: Ray, return_all: bool = False):
        dr = ray.origin - self.origin
        a = np.dot(ray.direction, ray.direction)
        b = 2*np.dot(dr, ray.direction)
        c = np.dot(dr, dr) - self.radius**2
        ts = quadratic_real_solutions(a, b, c)
        ts = [t for t in ts if t >= 0]
        if return_all:
            return ts
        else:
            if ts:
                return ts[0]
            else:
                return None

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

    def ray_intersection(self, ray: Ray):
        if self.invRadius != 0:
            r = 1/self.invRadius
            sphere = Sphere("", self.origin-r*self.direction, radius=r)
            ts = sphere.ray_intersection(ray, return_all=True)
        else:
            t = Plane("", self.origin, self.direction).ray_intersection(ray)
            ts = [t] if t is not None else []
        proj = projector_perpendicular(self.direction)
        ts_and_pts = [(t, ray.origin+ray.direction*t) for t in ts if t>0]
        ts = sorted([t for t, pt in ts_and_pts
            if abs(np.dot(pt-self.origin, self.direction)*self.invRadius)<1.0
            and np.linalg.norm(proj @ (pt-self.origin)) < self.r])
        return ts[0] if ts else None

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

class EvenPolynomialCap(OpticalSurface):
    def __init__(
            self,
            coefs,
            name='even polynomial cap',
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
        Z = np.polyval(self.coefs[::-1], R**2)

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

    def ray_intersection(self, ray: Ray):
        # proj @ (ray.origin - self.origin + ray.direction * t)
        proj = projector_perpendicular(self.direction)
        u = proj @ (ray.origin - self.origin)
        v = proj @ ray.direction
        a = np.dot(v, v)
        b = 2*np.dot(u, v)
        c = np.dot(u, u)
        # R^2 = a * t^2 + b * t + c
        A = np.dot(-self.direction, ray.direction)
        B = np.dot(-self.direction, ray.origin - self.origin)
        # z = A * t + B
        # A * t + B == P( a * t^2 + b * t + c )
        coefs = coefs_of_composition(self.coefs, [c, b, a])
        coefs[0] -= B
        coefs[1] -= A
        ts_and_pts = [(t.real, ray.origin+ray.direction*t.real)
            for t in np.roots(coefs[::-1]) if t.imag==0 and t.real>0]
        ts = sorted([t for t, pt in ts_and_pts
            if np.linalg.norm(proj @ (pt-self.origin)) < self.r])
        return ts[0] if ts else None

    def normal(self, pt):
        pt = np.array(pt)
        ey = -self.direction
        y = np.dot(ey, pt-self.origin)
        x = np.linalg.norm(pt-self.origin-y*ey)
        ex = (pt-self.origin-y*ey)/x
        coefs = []
        for c in self.coefs:
            coefs += [c, 0.0]
        n = normal_to_polynomial_graph(coefs, x)
        return n[0]*ex + n[1]*ey

    def bbox(self):
        pt = self.origin
        return pt-self.r, pt+self.r

class EvenAsphere(EvenPolynomialCap):
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
        coefs = np.concatenate(([0.0], coefs))

        EvenPolynomialCap.__init__(
            self,
            coefs,
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

    def ray_intersection(self, ray: Ray):
        k = (self.r2-self.r1)/self.h
        r0 = (self.r2+self.r1)/2
        a = np.dot(self.direction, ray.direction)
        b = np.dot(self.direction, ray.origin - self.origin)
        # z == a * t + b
        # r == r0 + k*z
        c = np.dot(ray.direction, ray.direction)
        d = 2*np.dot(ray.direction, ray.origin-self.origin)
        e = np.dot(ray.origin-self.origin, ray.origin-self.origin)
        # R^2 == c * t^2 + d * t + e == r^2 + z^2
        A = c - (k**2+1)*a**2
        B = d - 2*a*(b*(k**2+1)+r0*k)
        C = e - (r0**2+(k**2+1)*b**2+2*r0*k*b)
        ts = quadratic_real_solutions(A, B, C)

        ts_and_pts = [(t, ray.origin+ray.direction*t) for t in ts if t>0]
        ts = sorted([t for t, pt in ts_and_pts
            if abs(np.dot(self.direction, pt-self.origin))<self.h/2])
        return ts[0] if ts else None

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

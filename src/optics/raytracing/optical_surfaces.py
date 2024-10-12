"""Library of optical surfaces"""

import numpy as np
import itertools
from optics.raytracing import OpticalSurface

class Rectangle(OpticalSurface):
    def __init__(self, origin, extents, name = 'rectangle'):
        OpticalSurface.__init__(self, name)
        self.origin = np.array(origin)
        self.extents = np.array(extents)
        if len(self.origin) == 2:
            normal = self.extents[0][::-1] * np.array([1, -1])
        else:
            normal = np.cross(extents[0], extents[1])
        normal = normal / np.linalg.norm(normal)
        self.normalVec = np.array(normal)

    def pointList(self):
        res = []
        for i in range(len(self.extents)):
            for p in itertools.product([-1, 1], repeat = len(self.extents) - 1):
                v1 = np.array(p[:i] + (-1,) + p[i:])
                v2 = np.array(p[:i] + (+1,) + p[i:])
                p1 = self.origin + sum([x * u / 2 for x, u in zip(v1, self.extents)])
                p2 = self.origin + sum([x * u / 2 for x, u in zip(v2, self.extents)])
                res.append(np.array([p1, p2]))


        return res

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
        if origin is None: origin=np.zeros(2)
        if normal is None: normal=np.array([1.,0.])
        OpticalSurface.__init__(self,name)
        self.origin = np.array(origin)
        self.normalVec = np.array(normal)

    def pointList(self):
        if len(self.normalVec) == 2:
            tangent = np.array([[0.,1.],[-1.,0.]]).dot(self.normalVec)
            return np.array([
                self.origin+100.0*tangent,
                self.origin-100.0*tangent])
        elif len(self.normalVec) == 3:
            tangent1 = np.cross(self.normalVec, [0., 0., 1.])
            tangent1 = tangent1 / np.linalg.norm(tangent1)
            tangent2 = np.cross(self.normalVec, tangent1)
            return [
                np.array([
                    self.origin+100.0*tangent1,
                    self.origin-100.0*tangent1]),
                np.array([
                    self.origin+100.0*tangent2,
                    self.origin-100.0*tangent2])
                ]

    def distance(self,pt):
        pt = np.array(pt)
        return abs(np.dot(pt-self.origin,self.normalVec))

    def normal(self,pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin-radius,self.origin+radius

class Pinhole(OpticalSurface):
    def __init__(self, inner_radius, outer_radius,
            name='pinhole', origin = None, normal = None):
        if origin is None: origin=np.zeros(2)
        if normal is None: normal=np.array([1.,0.])
        OpticalSurface.__init__(self,name)
        self.origin = np.array(origin)
        self.normalVec = np.array(normal)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def pointList(self):
        if len(self.normalVec) == 2:
            tangent = np.array([[0.,1.],[-1.,0.]]).dot(self.normalVec)
            return [
                np.array([
                    self.origin + self.inner_radius * tangent,
                    self.origin + self.outer_radius * tangent]),
                np.array([
                    self.origin - self.inner_radius * tangent,
                    self.origin - self.outer_radius * tangent])
                ]
        elif len(self.normalVec) == 3:
            tangent1 = np.cross(self.normalVec, [0., 0., 1.])
            tangent1 = tangent1 / np.linalg.norm(tangent1)
            tangent2 = np.cross(self.normalVec, tangent1)
            return [
                np.array([
                    self.origin + self.inner_radius * tangent1,
                    self.origin + self.outer_radius * tangent1]),
                np.array([
                    self.origin - self.inner_radius * tangent1,
                    self.origin - self.outer_radius * tangent1]),
                np.array([
                    self.origin + self.inner_radius * tangent2,
                    self.origin + self.outer_radius * tangent2]),
                np.array([
                    self.origin - self.inner_radius * tangent2,
                    self.origin - self.outer_radius * tangent2])
                ]

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

    def normal(self,pt):
        return self.normalVec.view(np.ndarray)

    def bbox(self):
        radius = 100.0
        return self.origin - radius, self.origin + radius


class Sphere(OpticalSurface):
    def __init__(self,name='sphere',origin=None,radius=1.0):
        if origin is None: origin=np.zeros(2)
        OpticalSurface.__init__(self,name)
        self.origin = np.array(origin)
        self.radius = radius

    def pointList(self):
        return np.array([self.origin
            +self.radius*np.array([np.cos(u),np.sin(u)])
            for u in np.linspace(0,2*np.pi,1001)])

    def distance(self,pt):
        pt = np.array(pt)
        return abs(np.linalg.norm(pt-self.origin)-self.radius)

    def normal(self,pt):
        x = pt-self.origin
        return x/np.linalg.norm(x)

    def bbox(self):
        return self.origin-self.radius,self.origin+self.radius

class SphericalCap(OpticalSurface):
    def __init__(self,name='spherical cap',origin=None,invRadius=1.0,direction=None,r=0.5):
        if origin is None: origin=np.zeros(2)
        if direction is None: direction=np.array([1.0,0.0])
        OpticalSurface.__init__(self,name)
        self.origin = origin
        self.invRadius = invRadius
        self.r = r
        self.direction = direction

    def pointList(self):
        x = np.linspace(-self.r,self.r,101)
        y = x**2*self.invRadius/(1+np.sqrt(1-(self.invRadius*x)**2))
        pts = np.array([x,y])
        ey = -self.direction
        if len(self.origin) == 2:
            ex = np.array([[0,1],[-1,0]]).dot(self.direction)
            pts = np.array([ex,ey]).transpose().dot(pts)
            return self.origin+pts.transpose()
        else:
            ex1 = np.cross(self.direction, [0., 0., 1.])
            ex1 = ex1 / np.linalg.norm(ex1)
            ex2 = np.cross(self.direction, ex1)
            pts1 = np.array([ex1,ey]).transpose().dot(pts)
            pts2 = np.array([ex2,ey]).transpose().dot(pts)
            return [
                self.origin + pts1.transpose(),
                self.origin + pts2.transpose()
                ]


    def distance(self,pt):
        pt = np.array(pt)
        y = -np.dot(self.direction,pt-self.origin)
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
        y = -np.dot(self.direction,pt-self.origin)
        x = np.linalg.norm(pt-self.origin+y*self.direction)
        ex = (pt-self.origin+y*self.direction)/x
        s = x*self.invRadius
        c = np.sqrt(1-s**2)
        return c*self.direction+s*ex

    def bbox(self):
        y0 = self.r**2*self.invRadius/(
            1+np.sqrt(1-(self.r*self.invRadius)**2))
        pt = self.origin-self.direction*y0
        return pt-self.r,pt+self.r

class ConicalSlice(OpticalSurface):
    def __init__(self,name='conical slice',origin=None,r1=1.0,r2=1.0,h=1.0,direction=None):
        if origin is None: origin=np.zeros(2)
        if direction is None: direction=np.array([1.0,0.0])
        OpticalSurface.__init__(self,name)
        self.origin = origin #np.array(origin)
        self.r1 = r1
        self.r2 = r2
        self.h = h
        self.direction = direction

    def pointList(self):
        n = np.array([[0,1],[-1,0]]).dot(self.direction)
        return self.origin + np.array([
            -0.5*self.h*self.direction+self.r1*n,
            -0.5*self.h*self.direction-self.r1*n,
            +0.5*self.h*self.direction-self.r2*n,
            +0.5*self.h*self.direction+self.r2*n,
            -0.5*self.h*self.direction+self.r1*n
            ])

    def distance(self,pt):
        pt = np.array(pt)
        z = np.dot(self.direction,pt-self.origin)
        R = np.linalg.norm(pt-self.origin)
        r = np.sqrt(R**2-z**2)
        dx = np.array([z,r-(self.r1+self.r2)/2])
        ex = np.array([self.h,self.r2-self.r1])
        ex = ex/np.linalg.norm(ex)
        ey = np.array([[0,-1],[1,0]]).dot(ex)
        X = np.dot(ex,dx)
        L = np.sqrt(self.h**2+(self.r1-self.r2)**2)
        Y = abs(np.dot(ey,dx))
        if abs(X)<L/2:
            return Y
        else:
            return np.sqrt((abs(X)-L/2)**2+Y**2)


    def normal(self,pt):
        pt = np.array(pt)
        ey=pt-self.origin-np.dot(self.direction,pt-self.origin)*self.direction
        n = self.direction*(self.r1-self.r2)+ey*self.h
        n = n/np.linalg.norm(n)
        return n

    def bbox(self):
        R = np.sqrt(max(self.r1,self.r2)**2+(self.h/2)**2)
        return self.origin-R,self.origin+R

"""Tools for working with lenses."""

import numpy as np
from optics.raytracing import OpticalSurfaceCollection
from optics.raytracing.optical_surfaces import SphericalCap, ConicalSlice
import optics.glass_library as gllib

def lens_from_zmx(fname, origin = None, direction = None):
    surfaceName = None
    lensName = None
    surfaces = {}
    with open(fname, "r") as f:
        for line in f:
            line = line.strip("\n")
            line = line.replace("\x00", "")
            if len(line) == 0: continue

            if line.startswith("NAME "):
                lensName = line[5:]
            if line.startswith("SURF "):
                surfaceName = line[5:]
                surfaces[surfaceName] = {}

            if surfaceName is not None:
                if line.startswith("  CURV "):
                    surfaces[surfaceName]["curvature"] = float(line[7:])
                if line.startswith("  DIAM "):
                    surfaces[surfaceName]["clear_aperture_diameter"] = \
                        2*float(line[7:].split(" ")[0])
                if line.startswith("  MEMA "):
                    surfaces[surfaceName]["diameter"] = \
                        2*float(line[7:].split(" ")[0])
                if line.startswith("  GLAS "):
                    surfaces[surfaceName]["glass"] = line[7:]
                if line.startswith("  DISZ "):
                    surfaces[surfaceName]["disz"] = float(line[7:])

    lens = OpticalSurfaceCollection("zmx lens", [])
    if origin is None:
        origin = np.zeros(2)
    if direction is None:
        direction = np.array([1.0, 0.0])

    z = 0.0
    f2 = lambda _: 1.0
    for key in surfaces:

        if "clear_aperture_diameter" in surfaces[key]:
            f1 = f2
            if "glass" in surfaces[key]:
                s = surfaces[key]["glass"]
                if s.startswith("N-"):
                    s = s[2:]
                f2 = getattr(gllib, "n_"+s)
            else:
                f2 = lambda _: 1.0

            surf = SphericalCap(
                origin = origin + direction * z,
                invRadius = surfaces[key]["curvature"],
                r = 0.5 * surfaces[key]["diameter"],
                direction = -direction)
            lens.elements.append(surf)
            surf.makeRefractive(
                n = lambda lam,
                f1 = f1,
                f2 = f2:
                f2(lam) / f1(lam))

            z += surfaces[key]["disz"]


    return lens


class SymmetricLens(OpticalSurfaceCollection):
    def __init__(
            self,
            radius=1.0,
            origin=None,
            invR=0.25,
            centerThickness=0.4,
            direction=None):
        OpticalSurfaceCollection.__init__(self,"symmetric lens",[])
        if origin is None: origin = [0.,0.]
        if direction is None: direction = [1.,0.]
        self.origin = np.array(origin)
        self.radius = radius
        self.invR = invR
        self.centerThickness = centerThickness
        d = self.invR*self.radius**2/(1+np.sqrt(1-(self.radius*self.invR)**2))
        self.direction = np.array(direction)
        self.cap1 = SphericalCap(
            origin=self.origin-self.direction*self.centerThickness/2,
            invRadius=self.invR,
            r = self.radius,
            direction=-self.direction)
        self.cap2 = SphericalCap(
            origin=self.origin+self.direction*self.centerThickness/2,
            invRadius=self.invR,
            r = self.radius,
            direction=self.direction)
        self.edge = ConicalSlice(
            origin=self.origin,
            r1=self.radius,r2=self.radius,h=self.centerThickness-2*d,
            direction=self.direction
        )
        self.elements += [self.cap1, self.cap2, self.edge]

    def on_change(self, *args):
        d = self.invR*self.radius**2/(1+np.sqrt(1-(self.radius*self.invR)**2))
        self.cap1.origin = self.origin-self.direction*self.centerThickness/2
        self.cap1.invRadius = self.invR
        self.cap1.r=self.radius
        np.copyto(self.cap1.direction,-self.direction)
        self.cap2.origin=self.origin+self.direction*self.centerThickness/2
        self.cap2.invRadius=self.invR
        self.cap2.r=self.radius
        np.copyto(self.cap2.direction,self.direction)
        self.edge.origin=self.origin
        self.edge.r1=self.radius
        self.edge.r2=self.radius
        self.edge.h=self.centerThickness-2*d
        np.copyto(self.edge.direction,self.direction)

class SemiPlanarLens(OpticalSurfaceCollection):
    def __init__(
            self,
            radius=1.0,
            origin=None,
            invR=0.25,
            centerThickness=0.2,
            direction=None):
        OpticalSurfaceCollection.__init__(self,"semi-planar lens",[])
        if origin is None: origin = [0.,0.]
        if direction is None: direction = [1.,0.]
        self.origin = np.array(origin)
        self.radius = radius
        self.invR = invR
        self.centerThickness = centerThickness
        d = self.invR*self.radius**2/(1+np.sqrt(1-(self.radius*self.invR)**2))
        self.direction = np.array(direction)
        self.cap1 = SphericalCap(
            origin=self.origin-self.direction*(self.centerThickness+d)/2,
            invRadius=self.invR,
            r = self.radius,
            direction=-self.direction)
        self.cap2 = SphericalCap(
            origin=self.origin+self.direction*(self.centerThickness-d)/2,
            invRadius=0.0,
            r = self.radius,
            direction=self.direction)
        self.edge = ConicalSlice(
            origin=self.origin,
            r1=self.radius,r2=self.radius,h=self.centerThickness-d,
            direction=self.direction
        )
        self.elements += [self.cap1, self.cap2, self.edge]

    def on_change(self, *args):
        d = self.invR*self.radius**2/(1+np.sqrt(1-(self.radius*self.invR)**2))
        self.cap1.origin = self.origin-self.direction*(self.centerThickness+d)/2
        self.cap1.invRadius = self.invR
        self.cap1.r=self.radius
        np.copyto(self.cap1.direction,-self.direction)
        self.cap2.origin=self.origin+self.direction*(self.centerThickness-d)/2
        self.cap2.invRadius=0.0
        self.cap2.r=self.radius
        np.copyto(self.cap2.direction,self.direction)
        self.edge.origin=self.origin
        self.edge.r1=self.radius
        self.edge.r2=self.radius
        self.edge.h=self.centerThickness-d
        np.copyto(self.edge.direction,self.direction)

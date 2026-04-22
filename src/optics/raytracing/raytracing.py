import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import os
import types
import scipy.interpolate
import itertools, copy
from optics.cie import spectral_color_srgb

# decorators =======
def makeAdderMethods(cls):
    """Decorator automatically attaching adder methods to a class.

    For each classmethod `foo` already defined for the class, make
    a new method `addFoo` which when called as `obj.addFoo(...)`
    1. calls `cls.foo(...)`
    2. appends the returned object to `obj.elements`

    This is used for instance for LightSource which has classmethods
    such as `point_source`. Then the automatically added `addpoint_source`
    can be used to add elementary point sources to a composite LightSource.
    """
    def makeAdderMethod(name):
        nname = "add" + name[0].upper() + name[1:]
        def f(self, *args, **kwargs):
            self.elements.append(getattr(cls, name)(*args, **kwargs))
        f.__name__ = nname
        return f

    for name in dir(cls):
        if isinstance(getattr(cls, name), types.MethodType):
            addMethod = makeAdderMethod(name)
            setattr(cls, addMethod.__name__, addMethod)

    return cls

def onPrimitives(combiningFunc=None):
    def decorator(f):
        def wrapper(*args, **kwargs):
            self = args[0]
            lst = [f(*((obj,)+args[1:]), **kwargs)
                for obj in get_primitives(self)]
            if combiningFunc is not None:
                return combiningFunc(lst)
        return wrapper
    return decorator

def attachMethodsFromFile(fname):
    def makeAdderMethod(line):
        parts = line.split("|")
        if len(parts) == 2:
            name, expr = parts
        else:
            name, expr, post = parts
            post = post.strip()
        name = name.strip()
        expr = expr.strip()
        def method(self):
            obj = eval(expr)
            exec("obj."+post)
            self.elements.append(obj)
        method.__name__ = name.strip()
        return method

    def decorator(cls):
        if not hasattr(cls, "__exposedMethods__"):
            cls.__exposedMethods__ = []
        with open(fname, "r") as f:
            for line in f:
                method = makeAdderMethod(line)
                cls.__exposedMethods__.append(method.__name__)
                setattr(cls, method.__name__, method)
        return cls
    return decorator

def get_primitives(collection):
    if isinstance(collection, OpticalSurfaceCollection):
        for element in collection.elements:
            yield from get_primitives(element)
    else:
        yield collection

def _combine_bounding_boxes(lst):
    if lst:
        p1 = [min([pts[0][i] for pts in lst]) for i in range(len(lst[0][0]))]
        p2 = [max([pts[1][i] for pts in lst]) for i in range(len(lst[0][1]))]
        return np.array(p1), np.array(p2)

class Ray(object):
    def __init__(self, origin, direction, parameters, **kwargs):
        if "name" in kwargs:
            self.__name__ = kwargs["name"]
            del kwargs["name"]
        else:
            self.__name__ = "ray"
        if not isinstance(origin, np.ndarray):
            self.origin = np.array(origin)
        else:
            self.origin = origin
        direction = np.array(direction)
        self.direction = direction
        self.parameters = parameters

@makeAdderMethods
class LightSource(object):
    def __init__(self, name, elements):
        self.__name__ = name
        self.elements = elements
        self.result = []

        self.parameters = {"wavelength": 500.0e-9}

    def raytrace(self, opticalSystem):
        self.result = []
        for ray in self.get_rays():
            self.result.append(opticalSystem.raytrace(ray))

    def get_rays(self):
        for element in self.elements:
            if isinstance(element, LightSource):
                yield from element.get_rays()
            else:
                yield element

    def geometry(self):
        res = []
        cmap = matplotlib.cm.get_cmap("Spectral")
        for paths in self.result:
            for path in paths:
                pts = []
                for ray in path["rays"]:
                    if isinstance(ray, Ray):
                        pts.append(ray.origin)
                    else:
                        pts.append(ray)
                if path["status"] == "escape":
                    pts.append(pts[-1]+100.0*path["rays"][-1].direction)

                esccol = "green"
                if hasattr(ray, "parameters") and "wavelength" in ray.parameters:
                    lam = ray.parameters["wavelength"]
                    x = (lam-380.0e-9)/(740.0e-9-380.0e-9)
                    esccol = cmap(1.0-x)
                    esccol = tuple([int(255*x) for x in esccol])

                color = {
                    "escape": esccol,
                    "maxsteps": "red",
                    "maxrecursion": "orange",
                    "absorption": esccol #"blue"
                }[path["status"]]
                res.append({
                    "type": "line", "points": pts, "width": 2, "color": color
                    })
                if path["status"] == "escape":
                    pts = pts[:-1]
                for pt in pts:
                    res.append({
                        "type": "dot", "center": pt, "radius": 2, "color": color
                        })

        return res

    def add_ray(self):
        self.elements.append(Ray([0.0, 0.0], [1.0, 0.0], {}))

    @classmethod
    def point_source(cls):
        numRays = 16
        offset = 0.0
        origin = np.array([0.0, 0.0])
        angle = 2*np.pi
        obj = cls("point source", [])
        obj.offset = offset
        obj.angle = angle
        obj.numRays = numRays
        obj.origin = origin
        def make_rays():
            obj.elements = [
                Ray(obj.origin, [np.cos(u), np.sin(u)], {})
                for u in obj.offset+obj.angle*np.linspace(
                    -0.5, 0.5, obj.numRays+1)[:-1]]
            for ray in obj.elements:
                ray.parameters = obj.parameters
        obj.make_rays = make_rays
        obj.make_rays()

        return obj

    @classmethod
    def dir_source(cls):
        numRays = 16
        origin = np.array([-50.0, 0.0])
        beamWidth = 20.0
        direction = np.array([1.0, 0.0])

        R = np.array([[0., 1.], [-1., 0.]])
        obj = cls("directional source", [])
        for ray in obj.elements:
            ray.parameters = obj.parameters
        obj.direction = direction
        obj.beamWidth = beamWidth
        obj.numRays = numRays
        obj.origin = origin
        def make_rays():
            obj.elements = [
                Ray(obj.origin+R.dot(obj.direction)*u, obj.direction, {})
                for u in obj.beamWidth*np.linspace(-0.5, 0.5, obj.numRays)]
            for ray in obj.elements:
                ray.parameters = obj.parameters
        obj.make_rays = make_rays
        obj.make_rays()

        return obj

    def on_change(self, *args):
        if hasattr(self, "make_rays"):
            self.make_rays()

class SurfaceRayInteraction(object):
    def __init__(self,f,inttype=None):
        self.f = f
        self.inttype = inttype
        self.kwargs = {}

    def __call__(self, rays, point, normal, tol):
        if self.f is None:
                return []
        if isinstance(rays, Ray):
                rays = [rays]
        lst = []
        for ray in rays:
            res = self.f(ray, point, normal, tol, **self.kwargs)
            if isinstance(res, Ray):
                res = [res]
            lst += res
        return lst

    @classmethod
    def absorber(cls):
        return cls(None, inttype="absorber")

    @staticmethod
    def f_abs(ray, point, normal, tol):
        ndir = ray.direction.view(np.ndarray)

        shift = 10*tol/abs(np.dot(normal,ndir))
        pt = point + shift*ndir
        return Ray(pt, ndir, ray.parameters)

    @classmethod
    def transparent(cls):
        return cls(cls.f_abs, inttype="transparent")

    @staticmethod
    def f_mirror(ray, point, normal, tol):
        ndir = ray.direction.view(np.ndarray)
        ndir = ndir-2*normal*np.dot(normal, ndir)

        shift = 10*tol/abs(np.dot(normal, ndir))
        pt = point + shift*ndir
        return Ray(pt, ndir, ray.parameters)

    @classmethod
    def mirror(cls):
        return cls(cls.f_mirror, inttype="mirror")

    @staticmethod
    def f_refr(ray, point, normal, tol, n=1.5):
        if callable(n):
            if hasattr(ray,"parameters") and "wavelength" in ray.parameters:
                lam = ray.parameters["wavelength"]
            else:
                lam = 580.0e-9
            n = n(lam)

        ndir = ray.direction.view(np.ndarray)
        tcos = -np.dot(normal, ndir)
        en = normal*(-1.0 if tcos>=0.0 else 1.0)
        et = ndir-en*np.dot(en, ndir)
        etnorm = np.linalg.norm(et)
        if etnorm > 0.0:
            et = et/etnorm
        else:
            et = np.ones_like(en)
        tsin = np.sqrt(1.0-tcos**2)
        if tcos>0.0:
            tsinRefr = tsin/n
        else:
            tsinRefr = tsin*n

        if tsinRefr>=1.0:
            ndir = ndir-2*normal*np.dot(normal, ndir)
        else:
            tcosRefr = np.sqrt(1.0-tsinRefr**2)
            ndir = tcosRefr*en + tsinRefr*et

        shift = 10*tol/abs(np.dot(normal, ndir))
        pt = point + shift*ndir
        return Ray(pt, ndir, ray.parameters)

    @staticmethod
    def f_grating(ray, point, normal, tol, dk=None):
        if dk is None: dk = np.array([0., 2*np.pi*1e6])
        ndir = ray.direction.view(np.ndarray)
        dkNorm = np.linalg.norm(dk)
        dk = dk - normal*np.dot(normal,dk)
        dk = dk*dkNorm/np.linalg.norm(dk)

        if hasattr(ray,"parameters") and "wavelength" in ray.parameters:
            lam = ray.parameters["wavelength"]
        else:
            lam = 560.0e-9
        k0 = 2*np.pi/lam
        kn = np.dot(k0*ndir, normal)
        kt = k0*ndir-normal*kn

        lst = []
        for j in [0, -1, 1]:
            kt2 = kt+j*dk
            kt2sqr = np.linalg.norm(kt2)**2
            if kt2sqr>k0**2: continue
            kn2 = np.sqrt(k0**2-kt2sqr)
            k2 = kt2+kn2*normal

            ndir = k2/k0

            shift = 10*tol/abs(np.dot(normal, ndir))
            pt = point + shift*ndir
            lst.append(Ray(pt, ndir, ray.parameters))

        return lst


    @staticmethod
    def f_refl_grating(ray, point, normal, tol, dk=None):
        if dk is None:
            dk = np.array([0., 2*np.pi*1e6])
        ndir = ray.direction.view(np.ndarray)
        dkNorm = np.linalg.norm(dk)
        dk = dk - normal*np.dot(normal,dk)
        dk = dk*dkNorm/np.linalg.norm(dk)

        if hasattr(ray, "parameters") and "wavelength" in ray.parameters:
            lam = ray.parameters["wavelength"]
        else:
            lam = 560.0e-9
        k0 = 2*np.pi/lam
        kn = np.dot(k0*ndir, normal)
        kt = k0*ndir-normal*kn

        lst = []
        for j in [0, -1, 1]:
            kt2 = kt+j*dk
            kt2sqr = np.linalg.norm(kt2)**2
            if kt2sqr>k0**2: continue
            kn2 = np.sqrt(k0**2-kt2sqr)
            k2 = kt2-kn2*normal

            ndir = k2/k0

            shift = 10*tol/abs(np.dot(normal, ndir))
            pt = point + shift*ndir
            lst.append(Ray(pt, ndir, ray.parameters))

        return lst


    @classmethod
    def refraction(cls, idxRefr):
        obj = cls(cls.f_refr, inttype="refraction")
        obj.kwargs = {"n": idxRefr}
        return obj

    @classmethod
    def grating(cls, dk):
        obj = cls(cls.f_grating, inttype="grating")
        obj.kwargs = {"dk": dk}
        return obj

    @classmethod
    def refl_grating(cls, dk):
        obj = cls(cls.f_refl_grating, inttype="refl_grating")
        obj.kwargs = {"dk": dk}
        return obj

class OpticalSurface(object):
    def __init__(self, name):
        self.__name__ = name
        self.surface_ray_interaction = SurfaceRayInteraction.transparent()

    def as_collection(self):
        return OpticalSurfaceCollection(self.__name__,[self])

    def __add__(self,other):
        return OpticalSurfaceCollection(
            "union("+self.__name__+","+other.__name__+")",
            self.as_collection().elements+other.as_collection().elements)

    def make_reflective(self):
        self.surface_ray_interaction = \
            SurfaceRayInteraction.mirror()
        return self

    def make_refractive(self, n = 1.5):
        self.surface_ray_interaction = \
            SurfaceRayInteraction.refraction(n)
        return self

    def make_absorptive(self):
        self.surface_ray_interaction = \
            SurfaceRayInteraction.absorber()
        return self

    def make_grating(self, lineDensity = 1e6):
        self.surface_ray_interaction = \
            SurfaceRayInteraction.grating(np.array([0.,2*np.pi*lineDensity]))
        return self

    def make_refl_grating(self, lineDensity = 1e6):
        self.surface_ray_interaction = \
            SurfaceRayInteraction.refl_grating(np.array([0.,2*np.pi*lineDensity]))
        return self

    def geometry(self):
        pts = self.pointList()
        return [{"type":"line","points":pts,"width":2,"color":"black"}]

    def normal(self, pt: np.ndarray):
        raise NotImplementedError(
            "`normal` method needs to be overridden for "\
            "children classes of OpticalSurface")

    def bbox(self):
        raise NotImplementedError(
            "`bbox` method needs to be overridden for "\
            "children classes of OpticalSurface")

class OpticalSurfaceCollection(object):
    def __init__(self, name, elements):
        self.__name__ = name
        self.elements = elements
        self.attrLinks = []

    def as_collection(self):
        return self

    def __add__(self,other):
        return OpticalSurfaceCollection(
            "union("+self.__name__+","+other.__name__+")",
            self.elements+other.as_collection().elements)

    def __setattr__(self,name,val):
        if hasattr(self,"attrLinks") and name in self.attrLinks:
            for dstobj,dstname in self.attrLinks[name]:
                setattr(dstobj,dstname,val)
        object.__setattr__(self,name,val)

    def draw(self,ax,
            color=None,
            showBbox=False,
            autoSetRange=True,
            draw=True,
            projection_matrix=None,
            ):
        lst = []
        if draw:
            if projection_matrix is None:
                projection_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            for p in get_primitives(self):
                pts = p.drawing(projection_matrix)
                kw = {} if color is None else {"color":color}
                if isinstance(pts, list):
                    for ptss in pts:
                        lst.append(ax.plot(ptss[:,0],ptss[:,1],"-",**kw)[0])
                        if not "color" in kw:
                            kw["color"] = lst[-1].get_color()
                else:
                    lst.append(ax.plot(pts[:,0],pts[:,1],"-",**kw)[0])

        pt1,pt2 = self.bbox()

        if showBbox:
            X = [pt1[0],pt2[0],pt2[0],pt1[0],pt1[0]]
            Y = [pt1[1],pt1[1],pt2[1],pt2[1],pt1[1]]
            lst.append(ax.plot(X,Y,"k--")[0])

        if autoSetRange:
            w = pt2[0]-pt1[0]
            h = pt2[1]-pt1[1]
            ctr = (pt1+pt2)/2
            size = 1.5*max(w,h)
            ax.set_xlim((ctr[0]-0.5*size,ctr[0]+0.5*size))
            ax.set_ylim((ctr[1]-0.5*size,ctr[1]+0.5*size))

        return lst

    def raytrace(self, ray, maxrecursion=12, tol=1e-6):
        ray = copy.deepcopy(ray)
        if maxrecursion == 0:
            return [{"rays": [ray], "status": "maxrecursion"}]
        pt,p,normal,distance,_,status = self.find_ray_intersection(
            ray.origin,
            ray.direction.view(np.ndarray),
            tol=tol)
        if status == "escape":
            res = [{"rays": [ray], "status": "escape"}]
        elif status == "maxsteps":
            res = [{"rays": [ray], "status": "maxsteps"}]
        else:
            res = []
            raysAfter = p.surface_ray_interaction(ray, pt, normal, tol)
            if raysAfter:
                for ray2 in raysAfter:
                    subTrace = self.raytrace(ray2,
                        maxrecursion=maxrecursion-1, tol=tol)
                    for subres in subTrace:
                        res.append({
                            "rays": [ray]+subres["rays"],
                            "status": subres["status"]
                            })
            else:
                res.append({
                    "rays": [ray, pt],
                    "status": "absorption"
                    })

        return res

    def find_ray_intersection(self, origin, rdir,
            maxsteps = 1000,
            tol = 1e-6):
        rdir = np.array(rdir) / np.linalg.norm(rdir)
        elements_exact = [p for p in get_primitives(self)
            if hasattr(p, "ray_intersection")]
        maxdist = np.inf
        closest = None
        for p in elements_exact:
            t = p.ray_intersection(Ray(origin, rdir, {}))
            if t is not None and t < maxdist:
                maxdist, closest = t, p
        elements_nonexact = [p for p in get_primitives(self)
            if not hasattr(p, "ray_intersection")]

        if elements_nonexact:
            coll = OpticalSurfaceCollection("", elements_nonexact)
            pt, prim, normal, dist, maxsteps, status = \
                coll._find_ray_intersection_using_distance(origin, rdir,
                    maxsteps=maxsteps, tol=tol, maxdist=maxdist)
        else:
            if maxdist < np.inf:
                pt = origin + maxdist*rdir
                return pt, closest, closest.normal(pt), None, None, "tol"
            else:
                return origin, None, None, None, None, "escape"

        if status == "tol" and np.linalg.norm(pt-origin) < maxdist:
            return pt, prim, normal, dist, maxsteps, status
        elif maxdist == np.inf and status != "tol":
            return pt, prim, normal, dist, maxsteps, status
        elif maxdist < np.inf:
            pt = origin + maxdist*rdir/np.linalg.norm(rdir)
            return pt, closest, closest.normal(pt), None, None, "tol"
        else:
            return origin, None, None, None, None, "escape"

    # def _find_ray_intersection_using_distance(self, origin, rdir,
    #         maxsteps = 1000,
    #         maxdist = np.inf,
    #         tol = 1e-6):
    #     bbox = self.bbox()
    #     pt = origin.copy()
    #     rdir = np.array(rdir)
    #     rdir = rdir / np.linalg.norm(rdir)
    #
    #     if bbox is None:
    #         return pt, None, None, None, None, "escape"
    #
    #     a = np.ones(len(pt))
    #     b = bbox[1].copy()
    #     mask = rdir<0.0
    #     a[mask] = -1.0
    #     b[mask] = -bbox[0][mask]
    #
    #     tot_dist = 0.0
    #     for c in range(maxsteps):
    #         dist = self.distance(pt)
    #         if np.any(a*pt > b):
    #             return pt, None, None, dist, c, "escape"
    #         if dist < tol:
    #             p,normal = self.closest_primitive_and_normal(pt)
    #             return pt, p, normal, dist, c, "tol"
    #         pt += dist*rdir
    #         tot_dist += dist
    #         if tot_dist > maxdist:
    #             return pt, None, None, dist, c, "escape"
    #
    #     p, normal = self.closest_primitive_and_normal(pt)
    #     return pt, p, normal, dist, maxsteps, "maxsteps"
    #
    # def closest_primitive_and_normal(self,pt):
    #     lst = [(p.distance(pt), p, p.normal(pt)) for p in get_primitives(self)]
    #     return sorted(lst, key=lambda p:p[0])[0][1:]

    def geometry(self):
        res = []
        for p in get_primitives(self):
            res += p.geometry()
        return res

    @onPrimitives(min)
    def distance(self,pt):
        return self.distance(pt)

    @onPrimitives(_combine_bounding_boxes)
    def bbox(self):
        return self.bbox()

    def make_reflective(self):
        self.surface_ray_interaction = SurfaceRayInteraction.mirror()
        for obj in get_primitives(self):
            obj.surface_ray_interaction = self.surface_ray_interaction
        return self

    def make_refractive(self,n=1.5):
        self.surface_ray_interaction = SurfaceRayInteraction.refraction(n)
        for obj in get_primitives(self):
            obj.surface_ray_interaction = self.surface_ray_interaction
        return self

    def make_absorptive(self):
        self.surface_ray_interaction = SurfaceRayInteraction.absorber()
        for obj in get_primitives(self):
            obj.surface_ray_interaction = self.surface_ray_interaction
        return self

def draw_raytracing_result(result, ax, projection_matrix=None, coldct=None):
    if coldct is None:
        coldct = {
            "escape": "green",
            "maxsteps": "red",
            "maxrecursion": "orange",
            }
    if projection_matrix is None:
        projection_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for bunch in result:
        pts = [(ray.origin if isinstance(ray, Ray) else ray)
            for ray in bunch["rays"]]
        if bunch["status"] == "escape":
            findir = bunch["rays"][-1].direction.view(np.ndarray)
            pts.append(pts[-1]+1e6*findir)
        pts = np.array(pts)
        pts = (projection_matrix @ pts.T).T
        if bunch["status"] == "absorption" and not "absorption" in coldct:
            if not "wavelength" in bunch["rays"][0].parameters:
                color = "blue"
            else:
                color = np.concatenate((
                    spectral_color_srgb(
                        bunch["rays"][0].parameters["wavelength"]/1e-9,
                        amp=0.3),
                    [0.2]
                    ))
        else:
            color = coldct[bunch["status"]]
        ax.plot(pts[:,0], pts[:,1], ".-", color=color)

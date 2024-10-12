import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import os
import types
import scipy.interpolate
import itertools

# decorators =======
def makeAdderMethods(cls):
    """Decorator automatically attaching adder methods to a class.

    For each classmethod `foo` already defined for the class, make
    a new method `addFoo` which when called as `obj.addFoo(...)`
    1. calls `cls.foo(...)`
    2. appends the returned object to `obj.elements`

    This is used for instance for LightSource which has classmethods
    such as `pointSource`. Then the automatically added `addPointSource`
    can be used to add elementary point sources to a composite LightSource.
    """
    def makeAdderMethod(name):
        nname = 'add' + name[0].upper() + name[1:]
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
        def wrapper(*args,**kwargs):
            self = args[0]
            lst = [f(*((obj,)+args[1:]),**kwargs)
                for obj in getPrimitives(self)]
            if combiningFunc is not None:
                return combiningFunc(lst)
        return wrapper
    return decorator

def attachMethodsFromFile(fname):
    def makeAdderMethod(line):
        parts = line.split('|')
        if len(parts) == 2:
            name,expr = parts
        else:
            name,expr,post = parts
            post = post.strip()
        name = name.strip()
        expr = expr.strip()
        def method(self):
            obj = eval(expr)
            exec('obj.'+post)
            self.elements.append(obj)
        method.__name__ = name.strip()
        return method

    def decorator(cls):
        if not hasattr(cls,'__exposedMethods__'):
            cls.__exposedMethods__ = []
        with open(fname,'r') as f:
            for line in f:
                method = makeAdderMethod(line)
                cls.__exposedMethods__.append(method.__name__)
                setattr(cls,method.__name__,method)
        return cls
    return decorator

def getPrimitives(collection):
    if isinstance(collection,OpticalSurfaceCollection):
        for element in collection.elements:
            yield from getPrimitives(element)
    else:
        yield collection

def _combine_bounding_boxes(lst):
    if lst:
        p1 = [min([pts[0][i] for pts in lst]) for i in range(len(lst[0][0]))]
        p2 = [max([pts[1][i] for pts in lst]) for i in range(len(lst[0][1]))]
        return np.array(p1),np.array(p2)

class Ray(object):
    def __init__(self,origin,direction,parameters,**kwargs):
        if 'name' in kwargs:
            self.__name__ = kwargs['name']
            del kwargs['name']
        else:
            self.__name__ = 'ray'
        if not isinstance(origin,np.ndarray):
            self.origin = np.array(origin)
        else:
            self.origin = origin
        direction = np.array(direction)
        self.direction = direction
        self.parameters = parameters

@makeAdderMethods
class LightSource(object):
    def __init__(self,name,elements):
        self.__name__ = name
        self.elements = elements
        self.result = []

        self.parameters = {'wavelength': 560.0e-9}

    def rayTrace(self,opticalSystem):
        self.result = []
        for ray in self.getRays():
            self.result.append(opticalSystem.rayTrace(ray))

    def getRays(self):
        for element in self.elements:
            if isinstance(element, LightSource):
                yield from element.getRays()
            else:
                yield element

    def geometry(self):
        res = []
        cmap = matplotlib.cm.get_cmap('Spectral')
        for paths in self.result:
            for path in paths:
                pts = []
                for ray in path['rays']:
                    if isinstance(ray,Ray):
                        pts.append(ray.origin)
                    else:
                        pts.append(ray)
                if path['status'] == 'escape':
                    pts.append(pts[-1]+100.0*path['rays'][-1].direction)

                esccol = 'green'
                if hasattr(ray,'parameters') and 'wavelength' in ray.parameters:
                    lam = ray.parameters['wavelength']
                    x = (lam-380.0e-9)/(740.0e-9-380.0e-9)
                    esccol = cmap(1.0-x)
                    esccol = tuple([int(255*x) for x in esccol])

                color = {
                    'escape': esccol,
                    'maxsteps': 'red',
                    'maxrecursion': 'orange',
                    'absorption': esccol #'blue'
                }[path['status']]
                res.append({
                    'type':'line','points':pts,'width':2,'color':color
                    })
                if path['status'] == 'escape':
                    pts = pts[:-1]
                for pt in pts:
                    res.append({
                        'type':'dot','center':pt,'radius':2,'color':color
                        })

        return res

    def addRay(self):
        self.elements.append(Ray([0.0,0.0],[1.0,0.0],{}))

    @classmethod
    def pointSource(cls):
        numRays = 16
        offset = 0.0
        origin = np.array([0.0,0.0])
        angle = 2*np.pi
        obj = cls('point source',[
            Ray(origin,[np.cos(u),np.sin(u)],{})
            for u in offset+angle*np.linspace(-0.5,0.5,numRays+1)[:-1]])
        for ray in obj.elements: ray.parameters = obj.parameters
        obj.offset = offset
        obj.angle = angle
        obj.numRays = numRays
        obj.origin = origin

        return obj

    @classmethod
    def dirSource(cls):
        numRays = 16
        origin = np.array([-50.0,0.0])
        beamWidth = 20.0
        direction = np.array([1.0,0.0])

        R = np.array([[0.,1.],[-1.,0.]])
        obj = cls('directional source',[
            Ray(origin+R.dot(direction)*u,direction,{})
            for u in beamWidth*np.linspace(-0.5,0.5,numRays)])
        for ray in obj.elements: ray.parameters = obj.parameters
        obj.direction = direction
        obj.beamWidth = beamWidth
        obj.numRays = numRays
        obj.origin = origin

        return obj

class SurfaceRayInteraction(object):
    def __init__(self,f,inttype=None):
        self.f = f
        self.inttype = inttype
        self.kwargs = {}

    def __call__(self,rays,point,normal,tol):
        if self.f is None: return []
        if isinstance(rays,Ray): rays = [rays]
        lst = []
        for ray in rays:
            res = self.f(ray,point,normal,tol,**self.kwargs)
            if isinstance(res,Ray): res = [res]
            lst += res
        return lst

    @classmethod
    def absorber(cls):
        return cls(None,inttype='absorber')

    @staticmethod
    def f_abs(ray,point,normal,tol):
        ndir = ray.direction.view(np.ndarray)

        shift = 10*tol/abs(np.dot(normal,ndir))
        pt = point + shift*ndir
        return Ray(pt,ndir,ray.parameters)

    @classmethod
    def transparent(cls):
        return cls(cls.f_abs,inttype='transparent')

    @staticmethod
    def f_mirror(ray,point,normal,tol):
        ndir = ray.direction.view(np.ndarray)
        ndir = ndir-2*normal*np.dot(normal,ndir)

        shift = 10*tol/abs(np.dot(normal,ndir))
        pt = point + shift*ndir
        return Ray(pt,ndir,ray.parameters)

    @classmethod
    def mirror(cls):
        return cls(cls.f_mirror,inttype='mirror')

    @staticmethod
    def f_refr(ray,point,normal,tol,n=1.5):
        if callable(n):
            if hasattr(ray,'parameters') and 'wavelength' in ray.parameters:
                lam = ray.parameters['wavelength']
            else:
                lam = 580.0e-9
            n = n(lam)

        ndir = ray.direction.view(np.ndarray)
        tcos = -np.dot(normal,ndir)
        en = normal*(-1.0 if tcos>=0.0 else 1.0)
        et = ndir-en*np.dot(en,ndir)
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
            ndir = ndir-2*normal*np.dot(normal,ndir)
        else:
            tcosRefr = np.sqrt(1.0-tsinRefr**2)
            ndir = tcosRefr*en + tsinRefr*et

        shift = 10*tol/abs(np.dot(normal,ndir))
        pt = point + shift*ndir
        return Ray(pt,ndir,ray.parameters)

    @staticmethod
    def f_grating(ray,point,normal,tol,dk=None):
        if dk is None: dk = np.array([0.,2*np.pi*1e6])
        ndir = ray.direction.view(np.ndarray)
        dkNorm = np.linalg.norm(dk)
        dk = dk - normal*np.dot(normal,dk)
        dk = dk*dkNorm/np.linalg.norm(dk)

        if hasattr(ray,'parameters') and 'wavelength' in ray.parameters:
            lam = ray.parameters['wavelength']
        else:
            lam = 560.0e-9
        k0 = 2*np.pi/lam
        kn = np.dot(k0*ndir,normal)
        kt = k0*ndir-normal*kn

        lst = []
        for j in [0,-1,1]:
            kt2 = kt+j*dk
            kt2sqr = np.linalg.norm(kt2)**2
            if kt2sqr>k0**2: continue
            kn2 = np.sqrt(k0**2-kt2sqr)
            k2 = kt2+kn2*normal

            ndir = k2/k0

            shift = 10*tol/abs(np.dot(normal,ndir))
            pt = point + shift*ndir
            lst.append(Ray(pt,ndir,ray.parameters))

        return lst


    @staticmethod
    def f_refl_grating(ray,point,normal,tol,dk=None):
        if dk is None: dk = np.array([0.,2*np.pi*1e6])
        ndir = ray.direction.view(np.ndarray)
        dkNorm = np.linalg.norm(dk)
        dk = dk - normal*np.dot(normal,dk)
        dk = dk*dkNorm/np.linalg.norm(dk)

        if hasattr(ray,'parameters') and 'wavelength' in ray.parameters:
            lam = ray.parameters['wavelength']
        else:
            lam = 560.0e-9
        k0 = 2*np.pi/lam
        kn = np.dot(k0*ndir,normal)
        kt = k0*ndir-normal*kn

        lst = []
        for j in [0,-1,1]:
            kt2 = kt+j*dk
            kt2sqr = np.linalg.norm(kt2)**2
            if kt2sqr>k0**2: continue
            kn2 = np.sqrt(k0**2-kt2sqr)
            k2 = kt2-kn2*normal

            ndir = k2/k0

            shift = 10*tol/abs(np.dot(normal,ndir))
            pt = point + shift*ndir
            lst.append(Ray(pt,ndir,ray.parameters))

        return lst


    @classmethod
    def refraction(cls,idxRefr):
        obj = cls(cls.f_refr,inttype='refraction')
        obj.kwargs = {'n':idxRefr}
        return obj

    @classmethod
    def grating(cls,dk):
        obj = cls(cls.f_grating,inttype='grating')
        obj.kwargs = {'dk':dk}
        return obj

    @classmethod
    def refl_grating(cls,dk):
        obj = cls(cls.f_refl_grating,inttype='refl_grating')
        obj.kwargs = {'dk':dk}
        return obj

class OpticalSurface(object):
    def __init__(self,name):
        self.__name__ = name
        self.surfaceRayInteraction = SurfaceRayInteraction.transparent()

    def asCollection(self):
        return OpticalSurfaceCollection(self.__name__,[self])

    def __add__(self,other):
        return OpticalSurfaceCollection(
            'union('+self.__name__+','+other.__name__+')',
            self.asCollection().elements+other.asCollection().elements)

    def makeReflective(self):
        self.surfaceRayInteraction = SurfaceRayInteraction.mirror()

    def makeRefractive(self, n = 1.5):
        self.surfaceRayInteraction = SurfaceRayInteraction.refraction(n)

    def makeAbsorptive(self):
        self.surfaceRayInteraction = SurfaceRayInteraction.absorber()

    def makeGrating(self, lineDensity = 1e6):
        self.surfaceRayInteraction = SurfaceRayInteraction.grating(np.array([0.,2*np.pi*lineDensity]))

    def makeReflGrating(self, lineDensity = 1e6):
        self.surfaceRayInteraction = SurfaceRayInteraction.refl_grating(np.array([0.,2*np.pi*lineDensity]))

    def geometry(self):
        pts = self.pointList()
        return [{'type':'line','points':pts,'width':2,'color':'black'}]

    def distance(self,pt):
        raise NotImplementedError("""
            "distance" method needs to be overridden for children classes
            of OpticalSurface
        """)

    def normal(self,pt):
        raise NotImplementedError("""
            "normal" method needs to be overridden for children classes
            of OpticalSurface
        """)

    def bbox(self):
        raise NotImplementedError("""
            "bbox" method needs to be overridden for children classes
            of OpticalSurface
        """)


#@attachMethodsFromFile('lenses.txt')
class OpticalSurfaceCollection(object):
    def __init__(self,name,elements):
        self.__name__ = name
        self.elements = elements
        self.attrLinks = []

    def asCollection(self):
        return self

    def __add__(self,other):
        return OpticalSurfaceCollection(
            'union('+self.__name__+','+other.__name__+')',
            self.elements+other.asCollection().elements)

    def __setattr__(self,name,val):
        if hasattr(self,'attrLinks') and name in self.attrLinks:
            for dstobj,dstname in self.attrLinks[name]:
                setattr(dstobj,dstname,val)
        object.__setattr__(self,name,val)

    def draw(self,ax,
            color=None,
            showBbox=False,
            autoSetRange=True,
            draw=True):
        lst = []
        if draw:
            for p in getPrimitives(self):
                pts = p.pointList()
                kw = {} if color is None else {'color':color}
                if isinstance(pts, list):
                    for ptss in pts:
                        lst.append(ax.plot(ptss[:,0],ptss[:,1],'-',**kw)[0])
                else:
                    lst.append(ax.plot(pts[:,0],pts[:,1],'-',**kw)[0])

        pt1,pt2 = self.bbox()

        if showBbox:
            X = [pt1[0],pt2[0],pt2[0],pt1[0],pt1[0]]
            Y = [pt1[1],pt1[1],pt2[1],pt2[1],pt1[1]]
            lst.append(ax.plot(X,Y,'k--')[0])

        if autoSetRange:
            w = pt2[0]-pt1[0]
            h = pt2[1]-pt1[1]
            ctr = (pt1+pt2)/2
            size = 1.5*max(w,h)
            ax.set_xlim((ctr[0]-0.5*size,ctr[0]+0.5*size))
            ax.set_ylim((ctr[1]-0.5*size,ctr[1]+0.5*size))

        return lst

    def rayTrace(self,ray,maxrecursion=12,tol=1e-6,ax=None,coldct=None):
        if maxrecursion == 0:
            return [{'rays':[ray],'status':'maxrecursion'}]
        pt,p,normal,distance,_,status = self.findRayIntersection(
            ray.origin,
            ray.direction.view(np.ndarray),
            tol=tol)
        if status == 'escape':
            res = [{'rays':[ray],'status':'escape'}]
        elif status == 'maxsteps':
            res = [{'rays':[ray],'status':'maxsteps'}]
        else:
            res = []
            raysAfter = p.surfaceRayInteraction(ray,pt,normal,tol)
            if raysAfter:
                for ray2 in raysAfter:
                    subTrace = self.rayTrace(ray2,
                        maxrecursion=maxrecursion-1,tol=tol,ax=None)
                    for subres in subTrace:
                        res.append({
                            'rays':[ray]+subres['rays'],
                            'status':subres['status']
                            })
            else:
                res.append({
                    'rays':[ray,pt],
                    'status':'absorption'
                    })


        if ax is not None:
            if coldct is None:
                coldct = {
                    'escape': 'green',
                    'maxsteps': 'red',
                    'maxrecursion': 'orange',
                    'absorption': 'blue'
                }
            axc = []
            for bunch in res:
                pts = [(ray.origin if isinstance(ray,Ray) else ray)
                    for ray in bunch['rays']]
                if bunch['status'] == 'escape':
                    findir = bunch['rays'][-1].direction.view(np.ndarray)
                    pts.append(pts[-1]+1e6*findir)
                pts = np.array(pts)
                axc.append(
                    ax.plot(pts[:,0],pts[:,1],
                        '.-',color=coldct[bunch['status']])[0]
                    )

            return res,axc

        return res

    def findRayIntersection(self, origin, rdir,
            maxsteps = 1000,
            tol = 1e-6):
        bbox = self.bbox()
        pt = origin.copy()
        rdir = np.array(rdir)
        rdir = rdir / np.linalg.norm(rdir)

        if bbox is None:
            return pt, None, None, None, None, 'escape'

        a = np.ones(len(pt))
        b = bbox[1].copy()
        mask = rdir<0.0
        a[mask] = -1.0
        b[mask] = -bbox[0][mask]

        for c in range(maxsteps):
            dist = self.distance(pt)
            if np.any(a*pt > b):
                return pt, None, None, dist, c, 'escape'
            if dist < tol:
                p,normal = self.closestPrimitiveAndNormal(pt)
                return pt, p, normal, dist, c, 'tol'
            pt += dist*rdir

        p, normal = self.closestPrimitiveAndNormal(pt)
        return pt, p, normal, dist, maxsteps, 'maxsteps'

    def closestPrimitiveAndNormal(self,pt):
        lst = [(p.distance(pt), p, p.normal(pt)) for p in getPrimitives(self)]
        return sorted(lst, key=lambda p:p[0])[0][1:]

    def geometry(self):
        res = []
        for p in getPrimitives(self):
            res += p.geometry()
        return res

    @onPrimitives(min)
    def distance(self,pt):
        return self.distance(pt)

    @onPrimitives(_combine_bounding_boxes)
    def bbox(self):
        return self.bbox()

    def makeReflective(self):
        self.surfaceRayInteraction = SurfaceRayInteraction.mirror()
        for obj in getPrimitives(self):
            obj.surfaceRayInteraction = self.surfaceRayInteraction

    def makeRefractive(self,n=1.5):
        self.surfaceRayInteraction = SurfaceRayInteraction.refraction(n)
        for obj in getPrimitives(self):
            obj.surfaceRayInteraction = self.surfaceRayInteraction

    def makeAbsorptive(self):
        self.surfaceRayInteraction = SurfaceRayInteraction.absorber()
        for obj in getPrimitives(self):
            obj.surfaceRayInteraction = self.surfaceRayInteraction

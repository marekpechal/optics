from optics.raytracing.optical_surfaces import (
    SphericalCap,
    ConicalSlice,
    PolynomialCap,
    )
from optics.raytracing import Ray, OpticalSurfaceCollection
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    r1 = 0.3
    h1 = 0.2
    coefs1 = np.array([0.0, 0.0, h1/r1**2])

    R2 = 1.0
    r2 = 0.5
    h2 = R2 - np.sqrt(R2**2-r2**2)

    th = 0.0
    axis_dir = np.array([np.cos(th), 0.0, np.sin(th)])

    d = (2*h1/r1)*(r2-r1)

    x1 = -d/2-h1
    x2 = d/2+h2

    obj1 = PolynomialCap(
        coefs1,
        origin=np.array([x1*np.cos(th), 0.0, x1*np.sin(th)]),
        direction=-axis_dir,
        r=r1)
    obj1.makeAbsorptive()

    obj2 = SphericalCap(
        origin=np.array([x2*np.cos(th), 0.0, x2*np.sin(th)]),
        invRadius=1/R2,
        direction=axis_dir,
        r=r2)
    obj2.makeAbsorptive()

    obj3 = ConicalSlice(
        origin=np.array([0.0, 0.0, 0.0]),
        r1=r1,
        r2=r2,
        h=d,
        direction=axis_dir)
    obj3.makeAbsorptive()

    coll = OpticalSurfaceCollection("collection", [obj1, obj2, obj3])

    xrng = np.linspace(-0.55, 0.55, 101)
    yrng = np.linspace(-0.55, 0.55, 101)
    ray_dir = np.array([0.0, 0.0, 1.0])
    im = np.zeros((len(yrng), len(xrng)))

    for i, y in enumerate(yrng):
        for j, x in enumerate(xrng):
            ray = Ray(np.array([x, y, -10.0]), ray_dir, {})
            res = coll.rayTrace(ray)[0]
            if res["status"] == "absorption":
                endpt = res["rays"][-1]
                _, n = coll.closestPrimitiveAndNormal(endpt)
                im[i, j] = 0.2 + abs(np.dot(n, ray_dir))
    plt.imshow(im[::-1])
    plt.colorbar()
    plt.show()

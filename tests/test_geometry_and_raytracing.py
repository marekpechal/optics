from optics.raytracing.optical_surfaces import (
    SphericalCap,
    ConicalSlice,
    EvenPolynomialCap,
    )
from optics.raytracing import Ray, OpticalSurfaceCollection
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    t0 = time.time()

    r1 = 0.3
    h1 = 0.2
    coefs1 = np.array([0.0, h1/r1**2])

    R2 = 1.0
    r2 = 0.5
    h2 = R2 - np.sqrt(R2**2-r2**2)

    th = 0.0
    axis_dir = np.array([np.cos(th), 0.0, np.sin(th)])

    d = (2*h1/r1)*(r2-r1)

    x1 = -d/2-h1
    x2 = d/2+h2

    obj1 = EvenPolynomialCap(
        coefs1,
        origin=np.array([x1*np.cos(th), 0.0, x1*np.sin(th)]),
        direction=-axis_dir,
        r=r1)

    obj2 = SphericalCap(
        origin=np.array([x2*np.cos(th), 0.0, x2*np.sin(th)]),
        invRadius=1/R2,
        direction=axis_dir,
        r=r2)

    obj3 = ConicalSlice(
        origin=np.array([0.0, 0.0, 0.0]),
        r1=r1,
        r2=r2,
        h=d,
        direction=axis_dir)

    coll = OpticalSurfaceCollection("collection", [obj1, obj2, obj3])

    xrng = np.linspace(-0.55, 0.55, 201)
    yrng = np.linspace(-0.55, 0.55, 201)
    ray_dir = np.array([0.0, 0.0, 1.0])
    im = np.zeros((len(yrng), len(xrng)))

    for i, y in enumerate(yrng):
        for j, x in enumerate(xrng):
            res = coll.find_ray_intersection(np.array([x, y, -10.0]), ray_dir)
            if res[-1] == "tol":
                n = res[2]
                im[i, j] = 0.2 + abs(np.dot(n, ray_dir))

    print(f"done in {time.time()-t0:.2f} seconds")

    plt.imshow(im[::-1])
    plt.colorbar()
    plt.show()

from optics.raytracing.optical_surfaces import (
    SphericalCap,
    ConicalSlice,
    PolynomialCap,
    Plane,
    )
from optics.read_zmx import ZemaxData
from optics.raytracing.lenses import lens_from_zemax_data
from optics.raytracing import (
    Ray,
    OpticalSurfaceCollection,
    draw_raytracing_result,
    )
from optics.cie import spectral_color_srgb
import numpy as np
import matplotlib.pyplot as plt
import os

lambdas = np.arange(425.0, 675.0, 20.0)
colors = [np.concatenate((spectral_color_srgb(lam, amp = 0.3), [0.2]))
    for lam in lambdas]

fnames = [
    "US004333714_Example01P.zmx",
    "US003817603_Example01P.zmx"
    ]

if __name__ == "__main__":
    for idx, fname in enumerate(fnames):
        plt.subplot(1, len(fnames), idx+1)
        data = ZemaxData(os.path.join("zmx_examples", fname), encoding="utf-8")
        lens = lens_from_zemax_data(
            data,
            origin = np.array([0.0, 0.0, 0.0]),
            direction = np.array([1.0, 0.0, 0.0]),
            air_n = 1.000277,
            )
        plane = Plane(origin=np.array([80.04, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]))
        plane.makeAbsorptive()

        collection = OpticalSurfaceCollection("collection", [lens, plane])

        for lam_nm in [450.0, 587.5618, 650.0]:
            for y in np.linspace(-5.0, 5.0, 10):
                ray = Ray(np.array([-50.0, y, 0.0]), np.array([1.0, 0.0, 0.0]), {"wavelength": lam_nm * 1e-9})
                result = collection.rayTrace(ray, maxrecursion=20)
                draw_raytracing_result(result, plt.gca(),
                    coldct = {
                        "maxrecursion": "red",
                        "absorption": np.concatenate((spectral_color_srgb(lam_nm, amp=0.3), [0.2]))
                        })

        collection.draw(plt.gca())

        plt.xlim((-20.0, 80.0))
        plt.ylim((-50.0, 50.0))
        plt.gca().set_aspect("equal")
        plt.title(data.name)

    plt.show()

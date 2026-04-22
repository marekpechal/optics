from optics.raytracing.optical_surfaces import (
    SphericalCap,
    ConicalSlice,
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

if __name__ == "__main__":
    # Edmund Optics Stock #49-656
    for idx, (ignore_aspheres, label) in enumerate(
            [(False, "with asphere"), (True, "without asphere")]):

        plt.subplot(1, 2, idx+1)

        lens = lens_from_zemax_data(
            ZemaxData(os.path.join("zmx_examples", "zmax_49656.ZAR")),
            origin = np.array([0.0, 0.0, 0.0]),
            direction = np.array([1.0, 0.0, 0.0]),
            air_n = 1.000277,
            ignore_aspheres = ignore_aspheres,
            add_boundaries = False,
            )
        plane = Plane(
            origin=np.array([40.0, 0.0, 0.0]),
            normal=np.array([1.0, 0.0, 0.0])).make_absorptive()

        collection = OpticalSurfaceCollection("collection", [lens, plane])

        for lam_nm in [450.0, 587.5618, 650.0]:
            for y in np.linspace(-4.4, 4.4, 10):
                ray = Ray(np.array([-5.0, y, 0.0]), np.array([1.0, 0.0, 0.0]), {"wavelength": lam_nm * 1e-9})
                result = collection.raytrace(ray)
                draw_raytracing_result(result, plt.gca(), coldct = {
                    "absorption": np.concatenate((spectral_color_srgb(lam_nm, amp=0.3), [0.2]))
                    })

        collection.draw(plt.gca())

        plt.plot(lens.elements[-1].origin[0]+8.162, 0, "ro", label="specified focus")
        # remaining 10um deviation is likely from the missing cement layer

        plt.legend()

        plt.xlim((-10.0, 20.0))
        plt.ylim((-15.0, 15.0))
        plt.gca().set_aspect("equal")
        plt.title(label)

    plt.show()

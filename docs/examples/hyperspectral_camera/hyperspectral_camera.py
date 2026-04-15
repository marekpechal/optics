from optics.raytracing.optical_surfaces import (
    SphericalCap,
    ConicalSlice,
    PolynomialCap,
    Plane,
    Pinhole,
    Slit,
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


data = ZemaxData("US003817603_Example01P.zmx", encoding="utf-8")

ftb_focal_distance = 69.66

lens1 = lens_from_zemax_data(
    data,
    origin = np.array([0.0, 0.0, 0.0]),
    direction = np.array([1.0, 0.0, 0.0]),
    air_n = 1.000277,
    )
lens2 = lens_from_zemax_data(
    data,
    origin = np.array([2*ftb_focal_distance, 0.0, 0.0]),
    direction = np.array([-1.0, 0.0, 0.0]),
    air_n = 1.000277,
    )
slit = Slit(0.1, 12.0, 10.0, origin = np.array([ftb_focal_distance, 0.0, 0.0]), normal = np.array([1.0, 0.0, 0.0]))
slit.makeAbsorptive()

plane = Plane(origin=np.array([160, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]))
plane.makeAbsorptive()


if __name__ == "__main__":

    collection = OpticalSurfaceCollection("collection", [lens1, lens2, plane, slit])

    results = []
    for th_y, th_z in [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)]:
        result = []
        for lam_nm in [450.0, 587.5618, 650.0]:
            for y in np.linspace(-1.0, 1.0, 10) * 50.0/1.7/2 * 0.9:
                ray_dir = np.array([np.cos(th_y)*np.cos(th_z), np.sin(th_y), np.sin(th_z)])
                th = np.arccos(ray_dir[0])
                ray_x0 = -50.0
                dx = 20.0-ray_x0
                ray = Ray(np.array([ray_x0, y - ray_dir[1]/ray_dir[0]*dx, -ray_dir[2]/ray_dir[0]*dx]), ray_dir, {"wavelength": lam_nm * 1e-9})
                result += collection.rayTrace(ray, maxrecursion=36)

        results.append(result)

    for idx, result in enumerate(results):
        for jdx, projection_matrix in enumerate([None, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]):
            plt.subplot(len(results), 2, 2*idx+jdx+1)

            draw_raytracing_result(result, plt.gca(),
                projection_matrix=projection_matrix,
                coldct = {
                    "maxrecursion": "red",
                    "maxsteps": "red",
                    "escape": "red",
                    "absorption": np.concatenate((spectral_color_srgb(lam_nm, amp=0.3), [0.2]))
                    },
                )
            collection.draw(
                plt.gca(),
                projection_matrix=projection_matrix,
                )

            plt.xlim((-20.0, 160.0))
            plt.ylim((-50.0, 50.0))
            plt.gca().set_aspect("equal")

    plt.suptitle(data.name)
    plt.show()

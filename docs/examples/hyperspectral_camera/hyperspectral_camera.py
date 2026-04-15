from optics.raytracing.optical_surfaces import (
    SphericalCap,
    ConicalSlice,
    PolynomialCap,
    Disk,
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
import numpy as np
import matplotlib.pyplot as plt
import os

lambdas = np.arange(425.0, 675.0, 20.0)

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
slit = Slit(0.1, 12.0, 10.0,
    origin = np.array([ftb_focal_distance, 0.0, 0.0]),
    normal = np.array([1.0, 0.0, 0.0])
    ).make_absorptive()
absorber_disk = Disk(
    radius=20.0,
    origin=np.array([160, 0.0, 0.0]),
    normal=np.array([1.0, 0.0, 0.0])
    ).make_absorptive()

lens1_last = [e for e in lens1.elements if isinstance(e, SphericalCap)][-1]
lens2_last = [e for e in lens2.elements if isinstance(e, SphericalCap)][-1]
x1 = lens1_last.origin[0] + lens1_last.dz
x2 = lens2_last.origin[0] - lens2_last.dz
r1 = lens1_last.r
r2 = lens2_last.r
tube = ConicalSlice(
    origin = np.array([(x1+x2)/2, 0.0, 0.0]),
    r1=r1, r2=r2, h=x2-x1).make_absorptive()

if __name__ == "__main__":

    collection = OpticalSurfaceCollection("collection",
        [lens1, lens2, absorber_disk, slit, tube])

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
                result += collection.raytrace(ray, maxrecursion=36)

        results.append(result)

    plt.figure(figsize=(16, 8))
    for idx, result in enumerate(results):
        for jdx, projection_matrix in enumerate([
                None,
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([[0.8, 0.0, 0.6], [-0.36, 0.8, 0.48]])
                ]):
            plt.subplot(len(results), 3, 3*idx+jdx+1)

            collection.draw(
                plt.gca(),
                projection_matrix=projection_matrix,
                color=(0.0, 0.0, 0.0, 0.3)
                )
            all_x = [x for line in plt.gca().lines for x in line.get_xdata()]
            all_y = [y for line in plt.gca().lines for y in line.get_ydata()]
            ext_mat = np.array([[1.1, -0.1], [-0.1, 1.1]])
            plt.xlim(*(ext_mat @ [min(all_x), max(all_x)]))
            plt.ylim(*(ext_mat @ [min(all_y), max(all_y)]))

            draw_raytracing_result(result, plt.gca(),
                projection_matrix=projection_matrix)

            plt.gca().set_aspect("equal")

    plt.suptitle(data.name)
    plt.show()

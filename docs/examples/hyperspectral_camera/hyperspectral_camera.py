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
    SurfaceRayInteraction,
    draw_raytracing_result,
    )
import numpy as np
import matplotlib.pyplot as plt
import os, time, multiprocessing
from optics.utils import Process

lambdas = np.arange(425.0, 675.0, 20.0)

# SMC Pentax-M 50/1.7
data_objective = ZemaxData("US003817603_Example01P.zmx", encoding="utf-8")
# # Edmund Optics #47-874
# data_field_lens = ZemaxData("zmax_47874.ZAR")
# # Edmund Optics #47-343
# data_field_lens = ZemaxData("zmax_47343.ZAR")
# Edmund Optics #63-619
data_field_lens = ZemaxData("zmax_63619.ZAR")

ftb_focal_distance = 69.66

lens1 = lens_from_zemax_data(
    data_objective,
    origin = np.array([0.0, 0.0, 0.0]),
    direction = np.array([1.0, 0.0, 0.0]),
    air_n = 1.000277,
    )
lens2 = lens_from_zemax_data(
    data_objective,
    origin = np.array([2*ftb_focal_distance, 0.0, 0.0]),
    direction = np.array([-1.0, 0.0, 0.0]),
    air_n = 1.000277,
    )
field_lens = lens_from_zemax_data(
    data_field_lens,
    # origin = np.array([73.2, 0.0, 0.0]),
    origin = np.array([76.7, 0.0, 0.0]),
    direction = np.array([-1.0, 0.0, 0.0]),
    air_n = 1.000277,
    )
lens3 = lens_from_zemax_data(
    data_objective,
    origin = np.array([2*ftb_focal_distance+5.0, 0.0, 0.0]),
    direction = np.array([1.0, 0.0, 0.0]),
    air_n = 1.000277,
    )
grating = Disk(
    radius=20.0,
    origin=np.array([2*ftb_focal_distance+2.5, 0.0, 0.0]),
    normal=np.array([1.0, 0.0, 0.0])
    )
grating.surface_ray_interaction = \
    SurfaceRayInteraction.grating(np.array([0.0, 2*np.pi*300.0/1e-3, 0.0]))
slit = Slit(0.1, 16.0, 12.5,
    origin = np.array([ftb_focal_distance, 0.0, 0.0]),
    normal = np.array([1.0, 0.0, 0.0])
    ).make_absorptive()
absorber_disk = Disk(
    radius=20.0,
    origin=np.array([214.7, 0.0, 0.0]),
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

def worker(idx, collection, rays, wavelengths, queue):
    result = []
    for ray in rays:
        for wavelength in wavelengths:
            ray.parameters["wavelength"] = wavelength
            result.append(collection.raytrace(ray, maxrecursion=48))
    queue.put((idx, result))

if __name__ == "__main__":

    t0 = time.time()
    print(f"starting at {t0}")

    wavelengths = [450.0e-9, 500.0e-9, 550.0e-9, 600.0e-9, 650.0e-9]

    queue = multiprocessing.Queue()

    collection = OpticalSurfaceCollection("collection",
        [lens1, lens2, absorber_disk, slit, tube, field_lens, lens3, grating])

    processes = []

    for idx, (th_y, th_z) in enumerate([(0.0, 0.0), (0.002, 0.0), (0.0, 0.15), (0.0, -0.15)]):

        rays = []
        for r, phi in [(0.0, 0.0)] + [(r, u) for u in np.linspace(0, 2*np.pi, 5)[:-1] for r in [2.5, 5.0]]:
            y = r*np.cos(phi) + 1e-6
            z = r*np.sin(phi)
            ray_dir = np.array([np.cos(th_y)*np.cos(th_z), np.sin(th_y), np.sin(th_z)])
            th = np.arccos(ray_dir[0])
            ray_x0 = -50.0
            dx = 20.0-ray_x0
            ray = Ray(np.array([ray_x0, y - ray_dir[1]/ray_dir[0]*dx, z - ray_dir[2]/ray_dir[0]*dx]), ray_dir, {})
            rays.append(ray)

        proc = Process(target=worker, args=(idx, collection, rays[0::4], wavelengths, queue))
        processes.append(proc)
        proc = Process(target=worker, args=(idx, collection, rays[1::4], wavelengths, queue))
        processes.append(proc)
        proc = Process(target=worker, args=(idx, collection, rays[2::4], wavelengths, queue))
        processes.append(proc)
        proc = Process(target=worker, args=(idx, collection, rays[3::4], wavelengths, queue))
        processes.append(proc)

    for pidx, proc in enumerate(processes):
        print(f"starting process {pidx}")
        proc.start()

    results = {} #[[] for _ in range(len(processes))]
    for _ in range(len(processes)):
        idx, result = queue.get()
        print(f"got results belonging to batch {idx}")
        if not idx in results:
            results[idx] = []
        for lst in result:
            results[idx] += lst
    results = [results.get(i, []) for i in range(max(results.keys())+1)]

    for pidx, proc in enumerate(processes):
        print(f"joining process {pidx}")
        proc.join()


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

    print(f"done in {time.time()-t0:.1f} seconds")

    plt.show()

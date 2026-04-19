from optics.raytracing import (
    Ray,
    draw_raytracing_result,
    )
import numpy as np
import matplotlib.pyplot as plt
import os, time, multiprocessing
from optics.utils import Process

from hyperspectral_camera_design import generate_optical_system

def worker(idx, collection, rays, wavelengths, queue):
    result = []
    for ray in rays:
        for wavelength in wavelengths:
            ray.parameters["wavelength"] = wavelength
            result.append(collection.raytrace(ray, maxrecursion=48))
    queue.put((idx, result))

def raytrace(collection, rays, wavelengths):
    queue = multiprocessing.Queue()

    processes = []
    processes.append(Process(target=worker, args=(0, collection, rays[0::4], wavelengths[0::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[0::4], wavelengths[1::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[1::4], wavelengths[0::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[1::4], wavelengths[1::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[2::4], wavelengths[0::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[2::4], wavelengths[1::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[3::4], wavelengths[0::2], queue)))
    processes.append(Process(target=worker, args=(0, collection, rays[3::4], wavelengths[1::2], queue)))
    for pidx, proc in enumerate(processes):
        print(f"starting process {pidx}")
        proc.start()

    results = []
    for _ in range(len(processes)):
        _, result = queue.get()
        print(f"got results")
        for lst in result:
            results += lst

    for pidx, proc in enumerate(processes):
        print(f"joining process {pidx}")
        proc.join()

    return results


if __name__ == "__main__":

    t0 = time.time()
    print(f"starting at {t0}")

    wavelengths = [450.0e-9, 500.0e-9, 550.0e-9, 600.0e-9, 650.0e-9]
    collection = generate_optical_system()

    rays = []
    for idx, (th_y, th_z) in enumerate([(0.0, 0.0), (0.002, 0.0), (0.0, 0.15), (0.0, -0.15)]):

        ray_bundle = []
        for r, phi in [(0.0, 0.0)] + [(r, u) for u in np.linspace(0, 2*np.pi, 5)[:-1] for r in [2.5, 5.0]]:
            y = r*np.cos(phi) + 1e-6
            z = r*np.sin(phi)
            ray_dir = np.array([np.cos(th_y)*np.cos(th_z), np.sin(th_y), np.sin(th_z)])
            th = np.arccos(ray_dir[0])
            ray_x0 = -50.0
            dx = 20.0-ray_x0
            ray = Ray(np.array([ray_x0, y - ray_dir[1]/ray_dir[0]*dx, z - ray_dir[2]/ray_dir[0]*dx]), ray_dir, {})
            ray_bundle.append(ray)
        rays.append(ray_bundle)


    # results = [raytrace(collection, ray_bundle, wavelengths)
    #     for ray_bundle in rays]

    results = [raytrace(collection, rays[0], [wavelengths[0], wavelengths[-1]])]

    print(f"done in {time.time()-t0:.1f} seconds")
    print("plotting")

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

    plt.show()

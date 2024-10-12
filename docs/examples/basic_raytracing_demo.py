import matplotlib.pyplot as plt
import numpy as np
from optics.raytracing import (
    Ray,
    SurfaceRayInteraction,
    OpticalSurfaceCollection
    )
from optics.raytracing.optical_surfaces import Sphere

if __name__ == "__main__":
    c1 = Sphere('circle 1',[0.0,0.0],1.5)
    c1.surfaceRayInteraction = SurfaceRayInteraction.mirror()
    c2 = Sphere('circle 2',[1.0,1.0],2.0)
    c2.surfaceRayInteraction = SurfaceRayInteraction.refraction(1.5)
    c3 = Sphere('circle 2',[1.8,2.1],0.5)
    c3.surfaceRayInteraction = SurfaceRayInteraction.absorber()
    group1 = c1+c2
    group = OpticalSurfaceCollection('group',[group1,c3])

    plt.figure(figsize=(6,6))
    ax = plt.subplot(1,1,1)
    group.draw(ax)
    y = 1.0
    for angle in np.linspace(-0.5,0.5,21):
        ray = Ray([-3.0,y],[np.cos(angle),np.sin(angle)],{})
        group.rayTrace(ray,tol=1e-6,ax=ax)

    plt.axis('off')
    plt.show()

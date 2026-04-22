from optics.raytracing.optical_surfaces import (
    EvenAsphere,
    SphericalCap,
    )

from optics.raytracing import OpticalSurfaceCollection
import matplotlib.pyplot as plt

coll = OpticalSurfaceCollection("collection", [
    SphericalCap(invRadius=1.0, r=0.5),
    EvenAsphere(invRadius=1.0, r=0.5),
    SphericalCap(invRadius=-1.0, r=0.7),
    EvenAsphere(invRadius=-1.0, r=0.7),
    ])

coll.draw(plt.gca())
plt.show()

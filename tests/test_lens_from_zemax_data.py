from optics.read_zmx import ZemaxData
from optics.raytracing.lenses import lens_from_zemax_data
import os
import matplotlib.pyplot as plt
import numpy as np

lens1 = lens_from_zemax_data(ZemaxData(os.path.join("zmx_examples", "zmax_32494.zmx")))
lens2 = lens_from_zemax_data(ZemaxData(os.path.join("zmx_examples", "zmax_49656.ZAR")))

plt.subplot(2, 2, 1)
ax = plt.gca()
lens1.draw(ax)

plt.subplot(2, 2, 2)
ax = plt.gca()
lens2.draw(ax)

plt.subplot(2, 2, 3)
ax = plt.gca()
lens1.draw(ax, projection_matrix=np.array([[0.8, 0.0, 0.6], [-0.36, 0.8, 0.48]]))

plt.subplot(2, 2, 4)
ax = plt.gca()
lens2.draw(ax, projection_matrix=np.array([[0.8, 0.0, 0.6], [-0.36, 0.8, 0.48]]))

plt.show()

data = ZemaxData(
    os.path.join("zmx_examples", "US004333714_Example01P.zmx"),
    encoding="utf-8")
lens = lens_from_zemax_data(data)
plt.subplot(1, 2, 1)
lens.draw(plt.gca())
plt.subplot(1, 2, 2)
lens.draw(plt.gca(), projection_matrix=np.array([[0.8, 0.0, 0.6], [-0.36, 0.8, 0.48]]))
plt.show()

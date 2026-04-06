from optics.read_zmx import ZemaxData
from optics.raytracing.lenses import lens_from_zemax_data
import os
import matplotlib.pyplot as plt

lens1 = lens_from_zemax_data(ZemaxData(os.path.join("zmx_examples", "zmax_32494.zmx")))
lens2 = lens_from_zemax_data(ZemaxData(os.path.join("zmx_examples", "zmax_49656.ZAR")))

plt.subplot(2, 1, 1)
ax = plt.gca()
lens1.draw(ax)

plt.subplot(2, 1, 2)
ax = plt.gca()
lens2.draw(ax)

plt.show()

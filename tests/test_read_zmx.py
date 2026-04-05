import os
from optics.read_zmx import ZemaxData

zmx = ZemaxData(os.path.join("zmx_examples", "zmax_49656.ZAR"))
print(zmx)

zmx = ZemaxData(os.path.join("zmx_examples", "zmax_32494.zmx"))
print(zmx)

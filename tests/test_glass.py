from optics.glass import Glass
from optics.read_zmx import ZemaxData
import matplotlib.pyplot as plt
import numpy as np
import os

zmx = ZemaxData(os.path.join("zmx_examples", "zmax_49656.ZAR"))
for file_name, f in zmx.additional_files.items():
    if file_name == "SCHOTT.AGF":
        shott_db_file = f.unpacked_contents

lam_arr = np.linspace(480, 670, 21)*1e-9

for glass_name, nd, Vd in [
        ("BK7", 1.51680, 64.17),
        ("SF5", 1.6727, 32.25),
        ("SF11", 1.7847, 25.68),
        ("LASF45", 1.80107, 34.97),
        ]:

    glass = Glass.from_library(glass_name)
    plt.plot(lam_arr/1e-9, glass.n(lam_arr),
        label=glass_name+" (library)")

    glass = Glass.from_two_term_model(glass_name, nd, Vd)
    plt.plot(lam_arr/1e-9, glass.n(lam_arr), "--",
        label=glass_name+" (two-term)")

    glass = Glass.from_agf_file("N-"+glass_name, shott_db_file)
    plt.plot(lam_arr/1e-9, glass.n(lam_arr), "-.",
        label=glass_name+" (agf file)")

    glass = Glass.from_zemax_data("N-"+glass_name, zmx)
    plt.plot(lam_arr/1e-9, glass.n(lam_arr), ".",
        label=glass_name+" (zemax data)")

plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Refractive index")
plt.show()

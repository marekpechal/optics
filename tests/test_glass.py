from optics.glass import Glass
import matplotlib.pyplot as plt
import numpy as np

lam_arr = np.linspace(370, 770, 101)*1e-9

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

plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Refractive index")
plt.show()

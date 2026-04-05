import optics.glass_library as gllib
import numpy as np

class Glass:
    def __init__(self, name, refractive_index_evaluator):
        self.name = name
        self.refractive_index_evaluator = refractive_index_evaluator

    def n(self, wavelength: float | np.ndarray) -> float:
        """
        Return refractive index at a given wavelength.

        Parameters
        ----------
        wavelength : float | np.ndarray
            Wavelength or array of wavelengths in meters.

        Returns
        -------
        n : float | np.ndarray
            Refractive index value or array of values.
        """
        if isinstance(wavelength, np.ndarray):
            return np.array([self.n(x) for x in wavelength])
        return self.refractive_index_evaluator(wavelength)

    @classmethod
    def from_library(cls, glass_name: str):
        """
        Load glass from library.

        Parameters
        ----------
        glass_name : str
        """
        return cls(glass_name, getattr(gllib, "n_"+glass_name))

    @classmethod
    def from_two_term_model(cls, name: str, nd: float, Vd: float):
        """
        Get glass from two-term model.

        Parameters
        ----------
        name : str
        nd : float
            Refractive index at yellow d-line (587.56nm).
        Vd : float
            Abbe number.
        """
        lam_d = 587.56e-9
        lam_F = 486.13e-9
        lam_C = 656.27e-9

        B = ((nd-1)/Vd) / (1/lam_F**2 - 1/lam_C**2)
        A = nd - B/lam_d**2

        return cls(name, lambda lam: A + B/lam**2)

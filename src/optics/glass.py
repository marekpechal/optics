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

    @classmethod
    def from_sellmeier(cls, name: str, coefsK: np.ndarray, coefsL: np.ndarray):
        """
        Get glass from Sellmeier formula.

        `n^2 - 1 == sum_i K_i * lam^2 / (lam^2 - L_i)`

        Parameters
        ----------
        name : str
        coefsK : np.ndarray
        coefsL : np.ndarray
        """
        return cls(
            name,
            lambda lam, coefsK=coefsK, coefsL=coefsL:
                np.sqrt(sum(K*lam**2/(lam**2-L)
                for K, L in zip(coefsK, coefsL)) + 1))

    @classmethod
    def from_agf_file(
            cls,
            name: str,
            f: str | bytes,
            error_if_not_found: bool = True):
        """
        Load glass from AGF file.

        Parameters
        ----------
        name : str
        f : str | bytes
            File name of file contents (as bytes).
        error_if_not_found: bool, optional
            Whether to raise an error when glass is not found. Defaults to True.
        """
        if not isinstance(f, bytes):
            with open(f, "rb") as f_handle:
                f = f_handle.read()
        lines = f.decode("utf-8").split("\n")
        active = False
        data = None
        keys = []
        for line in lines:
            parts = line.split(None)
            if not parts:
                continue
            if parts[0] == "NM":
                keys.append(parts[1])
                active = (parts[1] == name)
                if active:
                    data = {
                        "formula_number": int(parts[2]),
                        "mil_id": parts[3],
                        "nd": float(parts[4]),
                        "Vd": float(parts[5]),
                        }
            elif active and parts[0] == "CD":
                data["coefs"] = np.array([float(s) for s in parts[1:]])
        if data is None:
            if error_if_not_found:
                raise KeyError(f"key {name} not found among {keys}")
            else:
                return None
        if data["formula_number"] == 2:
            return cls.from_sellmeier(name,
                data["coefs"][0:6][0::2],
                data["coefs"][0:6][1::2]*1e-12)
        else:
            raise NotImplementedError(
                f"formula number {data['formula_number']}")

    @classmethod
    def from_zemax_data(
            cls,
            name: str,
            zemax_data: "ZemaxData"):
        """
        Load glass from Zemax data.

        Parameters
        ----------
        name : str
        zemax_data : ZemaxData
            Zemax data object.
        """

        file_names = zemax_data.catalogues if zemax_data.catalogues else []
        for file_name in file_names:
            obj = cls.from_agf_file(
                name,
                zemax_data.additional_files[file_name+".AGF"].unpacked_contents,
                error_if_not_found = False)
            if obj is not None:
                return obj
        raise KeyError(f"material {name} not found")

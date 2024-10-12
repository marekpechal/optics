"""Library of glass properties"""

import numpy as np

def n_BK7(lam):
    """Refractive index as a function of wavelength for glass BK7.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
    A = 1.03961212 * lam ** 2 / (lam ** 2 - 0.00600069867)
    B = 0.231792344 * lam ** 2 / (lam ** 2 - 0.0200179144)
    C = 1.01046945 * lam ** 2 / (lam ** 2 - 103.560653)
    return np.sqrt(1 + A + B + C)

def n_K5(lam):
    """Refractive index as a function of wavelength for glass K5.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=SCHOTT-K&page=N-K5
    A = 1.08511833 * lam ** 2 / (lam ** 2 - 0.00661099503)
    B = 0.199562005 * lam ** 2 / (lam ** 2 - 0.024110866)
    C = 0.930511663 * lam ** 2 / (lam ** 2 - 111.982777)
    return np.sqrt(1 + A + B + C)

def n_CaF2(lam):
    """Refractive index as a function of wavelength for CaF2 (calcium fluoride).

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=main&book=CaF2&page=Malitson
    A = 0.5675888 * lam ** 2 / (lam ** 2 - 0.050263605 ** 2)
    B = 0.4710914 * lam ** 2 / (lam ** 2 - 0.1003909 ** 2)
    C = 3.8484723 * lam ** 2 / (lam ** 2 - 34.649040 ** 2)
    return np.sqrt(1 + A + B + C)

def n_SF5(lam):
    """Refractive index as a function of wavelength for glass SF5.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=SF5&page=SCHOTT
    A = 1.52481889 * lam ** 2 / (lam ** 2 - 0.011254756)
    B = 0.187085527 * lam ** 2 / (lam ** 2 - 0.0588995392)
    C = 1.42729015 * lam ** 2 / (lam ** 2 - 129.141675)
    return np.sqrt(1 + A + B + C)

def n_SF11(lam):
    """Refractive index as a function of wavelength for glass SF11.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=SF11&page=SCHOTT
    A = 1.73759695 * lam ** 2 / (lam ** 2 - 0.013188707)
    B = 0.313747346 * lam ** 2 / (lam ** 2 - 0.0623068142)
    C = 1.89878101 * lam ** 2 / (lam ** 2 - 155.23629)
    return np.sqrt(1 + A + B + C)

def n_BAK1(lam):
    """Refractive index as a function of wavelength for glass BAK1.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=SCHOTT-BaK&page=N-BAK1
    A = 1.12365662 * lam ** 2 / (lam ** 2 - 0.00644742752)
    B = 0.309276848 * lam ** 2 / (lam ** 2 - 0.0222284402)
    C = 0.881511957 * lam ** 2 / (lam ** 2 - 107.297751)
    return np.sqrt(1 + A + B + C)

def n_SF15(lam):
    """Refractive index as a function of wavelength for glass SF15.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=SCHOTT-SF&page=N-SF15
    A = 1.57055634 * lam ** 2 / (lam ** 2 - 0.0116507014)
    B = 0.218987094 * lam ** 2 / (lam ** 2 - 0.0597856897)
    C = 1.50824017 * lam ** 2 / (lam ** 2 - 132.709339)
    return np.sqrt(1 + A + B + C)

def n_LASF45(lam):
    """Refractive index as a function of wavelength for glass LASF45.

    Args:
        lam (float): Wavelength in meters.

    Returns:
        float: Refractive index.
    """
    lam = lam / 1e-6 # convert from m to um
    # formula from
    # https://refractiveindex.info/?shelf=glass&book=SCHOTT-LaSF&page=N-LASF45
    A = 1.87140198 * lam ** 2 / (lam ** 2 - 0.011217192)
    B = 0.267777879 * lam ** 2 / (lam ** 2 - 0.0505134972)
    C = 1.73030008 * lam ** 2 / (lam ** 2 - 147.106505)
    return np.sqrt(1 + A + B + C)

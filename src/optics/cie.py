"""Tools for work with the CIE color space."""

import numpy as np
import scipy
import os

def spectral_color_srgb(lam_nm, amp = 1.0):
    """RGB values for monochromatic light using the CIE 1964 observer data.

    Args:
        lam_nm (float): Wavelength in nanometers.
        amp (float, optional): Intensity of the light. Defaults to 1.0.

    Returns:
        array: 3d vector of RGB values between 0.0 and 1.0.
    """
    arr = []
    with open(os.path.join(os.path.dirname(__file__), 'cie64.txt'), 'r') as f:
        for line in f:
            arr.append([float(s) for s in line.split(',')])
    arr = np.array(arr)
    f = scipy.interpolate.interp1d(arr[:,0], arr[:,1:], axis = 0)
    xyzvec = amp * f(lam_nm)
    rgbvec = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
        ]).dot(xyzvec)
    def gamma(x):
        if x <= 0.0031308:
            return 12.92 * x
        else:
            return 1.055 * x ** (1 / 2.4) - 0.055
    gammav = np.vectorize(gamma)
    return np.clip(gammav(rgbvec), 0.0, 1.0)

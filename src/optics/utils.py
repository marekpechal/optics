import numpy as np
import dill
import multiprocessing
import logging

logger_multiprocessing = logging.getLogger("multiprocessing")

class Process(multiprocessing.Process):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)
        self._args = Process._pickle_args(self._args)

    @staticmethod
    def _pickle_args(args):
        return tuple((dill.dumps(obj)
                if not isinstance(obj, multiprocessing.queues.Queue) else obj)
            for obj in args)

    @staticmethod
    def _unpickle_args(args):
        return tuple((dill.loads(obj)
                if not isinstance(obj, multiprocessing.queues.Queue) else obj)
            for obj in args)

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)
            self._args = Process._unpickle_args(self._args)
            logger_multiprocessing.debug(f"starting process {self}")
            self._target(*self._args, **self._kwargs)

def coefs_of_product(coefs1, coefs2):
    result = np.zeros(len(coefs1)+len(coefs2)-1)
    for i in range(len(result)):
        for j in range(i+1):
            if j < len(coefs1) and i-j < len(coefs2):
                result[i] += coefs1[j]*coefs2[i-j]
    return result

def coefs_of_derivative(coefs):
    return coefs[1:]*np.arange(1, len(coefs))

def closest_point_on_polynomial_graph(
        pt: np.ndarray,
        coefs: np.ndarray,
        bounds: tuple[float],
        ) -> np.ndarray:
    """
    Find closest point on a bounded polynomial graph to a given point.

    Parameters
    ----------
    pt : np.ndarray
        Point whose distance to minimize.
    coefs : np.ndarray
        Coefficients of the polynomial (0th order first).
    bounds : tuple[float]
        Bounds (xmin, xmax).

    Returns
    -------
    np.ndarray
        Array `[xopt, yopt, dist]`, where `xopt` and `yopt` are the coordinates
        of the closest point and `dist` its distance.
    """

    p = coefs.copy()
    p[0] -= pt[1]
    p = coefs_of_product(p, coefs_of_derivative(p))
    p[0] -= pt[0]
    p[1] += 1
    roots = np.roots(p[::-1])
    roots = roots[roots.imag==0].real
    roots = roots[np.logical_and(bounds[0] < roots, roots < bounds[1])]
    candidates = np.concatenate((roots, bounds))
    xopt = sorted(candidates,
        key=lambda x: (np.polyval(coefs[::-1], x)-pt[1])**2+(x-pt[0])**2)[0]
    yopt = np.polyval(coefs[::-1], xopt)
    dist = np.sqrt((xopt-pt[0])**2 + (yopt-pt[1])**2)
    return np.array([xopt, yopt, dist])

def normal_to_polynomial_graph(
        coefs: np.ndarray,
        x: float,
        ) -> np.ndarray:
    e = np.array([np.polyval(coefs_of_derivative(coefs)[::-1], x), -1.0])
    return e / np.linalg.norm(e)

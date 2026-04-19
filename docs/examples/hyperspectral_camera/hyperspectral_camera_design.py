
from optics.read_zmx import ZemaxData
from optics.raytracing.lenses import lens_from_zemax_data
from optics.raytracing.optical_surfaces import (
    ConicalSlice,
    SphericalCap,
    Disk,
    Slit,
    )
from optics.raytracing import (
    OpticalSurfaceCollection,
    SurfaceRayInteraction,
    )

import numpy as np

def generate_optical_system():
    # SMC Pentax-M 50/1.7
    data_objective = ZemaxData("US003817603_Example01P.zmx", encoding="utf-8")
    # # Edmund Optics #47-874
    # data_field_lens = ZemaxData("zmax_47874.ZAR")
    # # Edmund Optics #47-343
    # data_field_lens = ZemaxData("zmax_47343.ZAR")
    # Edmund Optics #63-619
    data_field_lens = ZemaxData("zmax_63619.ZAR")

    ftb_focal_distance = 69.66

    lens1 = lens_from_zemax_data(
        data_objective,
        origin = np.array([0.0, 0.0, 0.0]),
        direction = np.array([1.0, 0.0, 0.0]),
        air_n = 1.000277,
        )
    lens2 = lens_from_zemax_data(
        data_objective,
        origin = np.array([2*ftb_focal_distance, 0.0, 0.0]),
        direction = np.array([-1.0, 0.0, 0.0]),
        air_n = 1.000277,
        )
    field_lens = lens_from_zemax_data(
        data_field_lens,
        # origin = np.array([73.2, 0.0, 0.0]),
        origin = np.array([76.7, 0.0, 0.0]),
        direction = np.array([-1.0, 0.0, 0.0]),
        air_n = 1.000277,
        )
    lens3 = lens_from_zemax_data(
        data_objective,
        origin = np.array([2*ftb_focal_distance+5.0, 0.0, 0.0]),
        direction = np.array([1.0, 0.0, 0.0]),
        air_n = 1.000277,
        )
    grating = Disk(
        radius=20.0,
        origin=np.array([2*ftb_focal_distance+2.5, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0])
        )
    grating.surface_ray_interaction = \
        SurfaceRayInteraction.grating(np.array([0.0, 2*np.pi*300.0/1e-3, 0.0]))
    slit = Slit(0.1, 16.0, 12.5,
        origin = np.array([ftb_focal_distance, 0.0, 0.0]),
        normal = np.array([1.0, 0.0, 0.0])
        ).make_absorptive()
    absorber_disk = Disk(
        radius=20.0,
        origin=np.array([214.7, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0])
        ).make_absorptive()

    lens1_last = [e for e in lens1.elements if isinstance(e, SphericalCap)][-1]
    lens2_last = [e for e in lens2.elements if isinstance(e, SphericalCap)][-1]
    x1 = lens1_last.origin[0] + lens1_last.dz
    x2 = lens2_last.origin[0] - lens2_last.dz
    r1 = lens1_last.r
    r2 = lens2_last.r
    tube = ConicalSlice(
        origin = np.array([(x1+x2)/2, 0.0, 0.0]),
        r1=r1, r2=r2, h=x2-x1).make_absorptive()

    return OpticalSurfaceCollection("collection",
        [lens1, lens2, absorber_disk, slit, tube, field_lens, lens3, grating])

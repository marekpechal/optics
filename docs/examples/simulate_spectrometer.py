import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import optics.raytracing as rt
import optics.cie as cie
from optics.raytracing.lenses import lens_from_zmx
from optics.raytracing.optical_surfaces import Rectangle, Pinhole
import optics.glass_library as gllib

def getRayBundleSpread(rays, position, normalVec):
    drArr = []
    for ray in rays:
        t = np.dot(normalVec, position) - np.dot(normalVec, ray.origin)
        t = t / np.dot(normalVec, ray.direction)
        dr = ray.origin + ray.direction * t - position
        drArr.append(dr)

    drMean = np.mean(drArr, axis = 0)
    drSqrMean = np.mean([np.linalg.norm(dr) ** 2 for dr in drArr])
    return np.sqrt(drSqrMean - np.linalg.norm(drMean) ** 2)


lens1 = lens_from_zmx('zmax_49770.zmx')
lens2 = lens_from_zmx('zmax_63697.zmx',
    origin = np.array([148.0, 0.0]),
    direction = np.array([-1.0, 0.0]))


# position and orientation at "prism output"
ptc = np.array([182.92883976, 4.66732297])
meanAngle = 0.7401762236504678
e = np.array([np.cos(meanAngle), np.sin(meanAngle)])

lens3 = lens_from_zmx('zmax_67151.zmx',
    origin = ptc + 7.0 * e,
    direction = e)

# edmundoptics #45-950; 151 USD
prismCenter = np.array([180.0, 5.5])
prismApexAngle = 90.0 * np.pi / 180
prismRotation = -3.1
prismSide = 15.0
prismR = 0.5 * prismSide / np.cos(prismApexAngle / 2)
ex = np.array([np.cos(prismRotation), np.sin(prismRotation)])
ey = np.array([-np.sin(prismRotation), np.cos(prismRotation)])
prismPt1 = prismCenter + prismR * ex
prismPt2 = prismCenter - prismR * ex * np.cos(prismApexAngle) + prismR * ey * np.sin(prismApexAngle)
prismPt3 = prismCenter - prismR * ex * np.cos(prismApexAngle) - prismR * ey * np.sin(prismApexAngle)
prism = (
    Rectangle((prismPt1 + prismPt2) / 2, [prismPt2 - prismPt1]) +
    Rectangle((prismPt2 + prismPt3) / 2, [prismPt3 - prismPt2]) +
    Rectangle((prismPt3 + prismPt1) / 2, [prismPt1 - prismPt3])
    )
prism.makeRefractive(gllib.n_SF11)


pinhole = Pinhole(0.05, 20.0, origin = [105.7, 0.0])
pinhole.makeAbsorptive()

ptc = np.array([190.28242201, 11.44786648])
meanAngle = 0.7392430303400024
e = np.array([np.cos(meanAngle), np.sin(meanAngle)])
mirrorAngle = 1.9
em = np.array([np.cos(mirrorAngle), np.sin(mirrorAngle)])
mirror1 = Rectangle(ptc + 20.0 * e, [em * 20.0])
mirror1.makeReflective()


ptc = np.array([203.71414796, 25.0384887])
meanAngle = 3.0607569696599977
e = np.array([np.cos(meanAngle), np.sin(meanAngle)])
mirrorAngle = 1.6
em = np.array([np.cos(mirrorAngle), np.sin(mirrorAngle)])
mirror2 = Rectangle(ptc + 180.0 * e, [em * 30.0])
mirror2.makeReflective()

scene = lens1 + pinhole + lens2 + prism + lens3 + mirror1 + mirror2

# RAYTRACING

plt.figure(figsize=(6,6))
ax = plt.subplot(1,1,1)
scene.draw(ax, 'k')
plt.axis('off')

lambdas = np.arange(420.0, 740.0, 20.0)
colors = [np.concatenate((cie.spectral_color_srgb(lam, amp = 0.3), [0.2]))
    for lam in lambdas]

phi = 0.000
finalRayBundles = []
for lam_nm, col in zip(lambdas, colors):

    finalRays = []
    for y in np.linspace(-10.0, 10.0, 11):
        ray = rt.Ray([-10.0, y + 1e-6], [np.cos(phi), np.sin(phi)], {'wavelength': lam_nm * 1e-9})
        res = scene.rayTrace(ray, tol = 1e-6, ax = ax,
            maxrecursion = 15,
            coldct = {
                'escape': col,
                'maxsteps': 'black',
                'maxrecursion': 'black',
                'absorption': (0, 0, 0, 0.1)
            })
        if hasattr(res[0][0]['rays'][-1], 'direction'):
            finalRays.append(res[0][0]['rays'][-1])
    finalRayBundles.append(finalRays)

# POSTPROCESSING

finalRaysFlat = [ray for finalRays in finalRayBundles for ray in finalRays]

dirs = [np.angle(np.dot([1,1j],ray.direction)) for ray in finalRaysFlat]
meanAngle = (max(dirs) + min(dirs)) / 2

print('angle spread =',max(dirs) - min(dirs))

u = np.array([np.cos(meanAngle), np.sin(meanAngle)])
w = np.array([-np.sin(meanAngle), np.cos(meanAngle)])
pte = sorted(finalRaysFlat, key = lambda ray: np.dot(u, ray.origin))[-1].origin

ints = [ray.origin + ray.direction * np.dot(u, pte - ray.origin) / np.dot(u, ray.direction) for ray in finalRaysFlat]
lst = [np.dot(w, pt) for pt in ints]
t1, t2 = min(lst), max(lst)
pt1 = ints[0] + w * (t1 - np.dot(w, ints[0]))
pt2 = ints[0] + w * (t2 - np.dot(w, ints[0]))
ptc = (pt1 + pt2) / 2

print('ptc =', ptc)
print('mean angle =', meanAngle)

ys = np.array([np.dot(w, pt - ptc) for pt in ints])

ydirs = []
i = 0
for bundle in finalRayBundles:
    lst = []
    for ray in bundle:
        lst.append([ys[i], dirs[i]])
        i += 1
    ydirs.append(lst)

for ray in finalRaysFlat:
    x, y = ray.origin
    plt.plot(x, y, 'bo')

plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'k--')
plt.plot(ptc[0], ptc[1], 'ko')

plt.xlim((-20.0, 220.0))
plt.ylim((-20.0, 80.0))
plt.grid()
plt.show()

drng = np.linspace(0, 400.0, 801)
varrngs = []
for bundle, color in zip(finalRayBundles, colors):
    varrng = [getRayBundleSpread(bundle, ptc + d * u, u) for d in drng]
    plt.semilogy(drng, varrng, color = color)
    varrngs.append(varrng)

varrngMax = np.max(varrngs, axis = 0)
print('min spread =', min(varrngMax))

plt.semilogy(drng, varrngMax, color = 'black')

plt.grid()
plt.xlabel('Distance from focusing lens [mm]')
plt.ylabel('Spot size [mm]')
plt.show()

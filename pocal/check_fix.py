"""
Quick check that the substrate incidence-angle fix is working.

Put this next to setup.py and materialLibrary.json, then run:

    python check_fix.py

All expected values are derived from your own materialLibrary.json, so this
works whatever glass the library happens to contain.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")          # no plot windows
import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists("materialLibrary.json"):
    sys.exit("materialLibrary.json not found -- run this from the folder that contains it.")

# The bundled library has no entry for Air, the most common incident medium of
# all. Add it once, on the same wavelength grid Glass uses.
with open("materialLibrary.json") as fh:
    library = json.load(fh)

if not any(entry["material"] == "Air" for entry in library):
    grid = [entry for entry in library if entry["material"] == "Glass"][0]["wavelength"]
    library.append({
        "material": "Air",
        "wavelength": list(grid),
        "real": [1.0] * len(grid),
        "complex": [0.0] * len(grid),
    })
    with open("materialLibrary.json", "w") as fh:
        json.dump(library, fh, indent=1)
    print(f"Added an Air entry to materialLibrary.json ({len(grid)} wavelengths, n=1, k=0).")

from pocal.thinfilms import pocal

with open("_check_glass_glass.txt", "w") as fh:
    fh.write("Glass\t0\nGlass\t0\n")
with open("_check_glass_air.txt", "w") as fh:
    fh.write("Glass\t0\nAir\t0\n")

# Cubic-spline interpolation of the library leaves k at ~1e-12 rather than
# exactly zero, so "zero transmittance" gets a tolerance well above that noise
# floor but far below anything physically meaningful.
ZERO = 1e-6


def transmittance(prescription, angle):
    p = pocal(prescription, angle, 450, 750, 50, 550, False, None)
    result = p.s_polarization("transmittance", savefile=False, savefig=False)
    plt.close("all")
    return np.ravel(result[1])


probe = pocal("_check_glass_glass.txt", 0, 450, 750, 50, 550, False, None)
n_glass, _ = probe.search_from_nr_k_generator(probe.nr_k_array, "Glass")
waves = probe.wave_spacing
theta_c = np.rad2deg(np.arcsin(1.0 / n_glass))

print()
print(f"Your library's Glass: n = {n_glass[0]:.5f} at {waves[0]}nm "
      f"to {n_glass[-1]:.5f} at {waves[-1]}nm")
print(f"So the critical angle runs {theta_c.min():.3f} deg to {theta_c.max():.3f} deg.")

ok = True

print()
print("TEST 1  Glass on Glass -- no real interface, so T must be 100% everywhere")
print("-" * 72)
for angle in (0, 20, 45, 70):
    T = transmittance("_check_glass_glass.txt", angle)
    good = np.allclose(T, 100.0, atol=ZERO)
    ok &= good
    print(f"  {angle:5.1f} deg   {waves[0]}nm={T[0]:9.5f}%   {waves[-1]}nm={T[-1]:9.5f}%   "
          f"{'PASS' if good else 'FAIL'}")

print()
print("TEST 2  Glass on Air -- the critical angle must shift with wavelength")
print("-" * 72)
angle = 0.5 * (theta_c.min() + theta_c.max())
T = transmittance("_check_glass_air.txt", angle)
past = angle > theta_c                       # these wavelengths are evanescent
good = np.all(T[past] < ZERO) and np.all(T[~past] > 1.0)
ok &= good
print(f"  {angle:5.3f} deg")
for i, w in enumerate(waves):
    want = "0" if past[i] else "> 0"
    print(f"      {w}nm  T={T[i]:9.5f}%   (theta_c={theta_c[i]:.3f}, want {want})")
print(f"      {'PASS' if good else 'FAIL'}")

print()
print("TEST 3  Glass on Air -- past every critical angle, T must vanish")
print("-" * 72)
for angle in (theta_c.max() + 1, theta_c.max() + 5, theta_c.max() + 20):
    T = transmittance("_check_glass_air.txt", angle)
    good = np.all(T < ZERO)
    ok &= good
    print(f"  {angle:5.1f} deg   max over all wavelengths = {T.max():.3e}%   "
          f"{'PASS' if good else 'FAIL'}")

os.remove("_check_glass_glass.txt")
os.remove("_check_glass_air.txt")

print()
print("=" * 72)
print("ALL CHECKS PASSED -- the fix is working." if ok else
      "SOMETHING FAILED -- the old thinfilms.py is probably still being imported.")
print("=" * 72)

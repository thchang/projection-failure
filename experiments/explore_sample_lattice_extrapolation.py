# Mesh from a (sampled) lattice, lots of extrapolation points, highlight which simplices are touched.

import math
import random
import numpy as np
from delaunay_py import delsparse
from tlux.random import well_spaced_sphere
from tlux.random import mesh as meshgrid
from tlux.plot import Plot


# Generate a suitably large meshgrid that has an equal number of points
#  along each axis and randomly select points from it.
n = 300
d = 3
num_points = 5000
scale = (2**(-10))
skew = None


# Generate the lattice points.
n_on_axis = max(2, math.ceil(n ** (1/d)))
print("n_on_axis: ", n_on_axis, flush=True)
if (skew is None):
    skew = scale * np.ones(d) # No skew.
elif (skew == "linear"):
    skew = np.linspace(2**(-25), 1.0, d) # Linear skew.
elif (skew == "exponential"):
    skew = 2 ** np.linspace(-23, 0, d) # Exponential skew.
else:
    raise(ValueError(f"skew={repr(skew)} is unrecognized, expected one of (None, 'linear', 'exponential')."))

# Pull n random points from a lattice in the given dimension.
random.seed(0)
x = meshgrid(np.linspace(0, 1, n_on_axis), dimension=d, sample=n)
x -= x.mean(axis=0)
x /= abs(x).max(axis=0)
x = x * skew

# Generate test points, shift and scale them so they are surrounding the box.
points = (0.1 + d**(1/2)) * well_spaced_sphere(num_points=num_points, dimension=d) * skew

# Details about "x".
fmt = "% 8.5e "
print("x: ", x.shape, flush=True)
print("x.min():         ", (fmt*d)%tuple(x.min(axis=0).round(3).tolist()), flush=True)
print("x.max():         ", (fmt*d)%tuple(x.max(axis=0).round(3).tolist()), flush=True)
print("x.max()-x.min(): ", (fmt*d)%tuple((x.max(axis=0)-x.min(axis=0)).round(3).tolist()), flush=True)
print()

# Details about "points".
print("points: ", points.shape, flush=True)
print("points.min():              ", (fmt*d)%tuple(points.min(axis=0).round(3).tolist()), flush=True)
print("points.max():              ", (fmt*d)%tuple(points.max(axis=0).round(3).tolist()), flush=True)
print("points.max()-points.min(): ", (fmt*d)%tuple((points.max(axis=0)-points.min(axis=0)).round(3).tolist()), flush=True)
print()

# Get the simplex.
simps, weights, errors = delsparse(x, points, eps=None, ibudget=10000, extrap=2.0**52)
i = simps - 1
w = weights
print("simps:   ")
print(i, flush=True)
print("weights: ")
print(w, flush=True)

# Dimension.
if (d in {2,3}):
    p = Plot()
    p.add("Data", *x.T, color=1, marker_size=2, marker_line_width=0.5)
    p.add("Points", *points.T, color=(0,0,0,0.2), marker_size=2, marker_line_width=0.5)
    all_indices = sorted(set(simps.flatten()))
    p.add("Simplex", *x[all_indices].T, color=0, marker_size=3)
    # p.show(aspect_mode='data')
    p.show(aspect_mode='cube')



# One simplex, lots of extrapolation points.

import tqdm
import numpy as np
from delaunay_py import delsparse
from builtins import print as true_print
from tlux.random import well_spaced_sphere
from tlux.math import regular_simplex, orthogonalize
from tlux.plot import Plot


seed = 3
dimension = 3
skew = 2
num_points = 4000
radius = 0.5

# Disable printing for higher dimension.
if (dimension > 10):
    def print(*args, force=False, **kwargs):
        if force: true_print(*args, **kwargs)
else:
    def print(*args, force=None, **kwargs):
        true_print(*args, **kwargs)

# Seed and store a locally named variable.
np.random.seed(seed)
d = dimension

# Make a regular simplex.
simplex = regular_simplex(d=d+1) * radius
print("simplex:", flush=True)
print(simplex)
print("", flush=True)

# Make a random rotation.
rotation, _ = orthogonalize(np.random.normal(size=(d,d)))
rotation = (rotation.T * np.linspace(skew, 1/skew, d)).T
print("rotation:", flush=True)
print(rotation)
print("", flush=True)

# Rotate the simplex
print("random-simplex:", flush=True)
x = simplex @ rotation
print(x)

# Use Delaunay to project points outside the hull onto the simplex.
points = well_spaced_sphere(num_points, d)
 
# Get the weights.
simps, weights, errors = delsparse(x, points, eps=None)

# Plot the simplex and the data.
if (d in {2,3}):
    p = Plot()
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            p.add("simplex", *np.vstack([x[i], x[j]]).T, mode="lines", color=(0,0,0,0.3), line_width=2, show_in_legend=False, group="simplex")
    p.add("simplex", *x.T, marker_size=8, marker_line_width=1, color=2, show_in_legend=True, group="simplex")

# The list of groups seen in legend already..
seen_groups = set()

for index in tqdm.tqdm(range(num_points)):
    i = simps[index]
    w = weights[index]
    nonzero_indices = tuple(sorted((_i for _i, _w in zip(i, w) if _w > 2**(-26))))
    color = abs(hash(nonzero_indices)) % 30
    # Generate the projected point (onto the convex hull of the simplex).
    projection = np.asarray([(x[i].T * w).sum(axis=1)])
    if (d in {2,3}):
        # Plot the point, the projection line, and the projected point.
        # p.add(f"{index} projection line", *np.vstack((points[index:index+1], projection)).T, mode="lines", dash="dash", color=(0,0,0,0.5), group=index)
        p.add(f"{color}  {nonzero_indices}", *points[index:index+1].T, marker_size=4, marker_line_width=1, color=color, group=str(nonzero_indices), show_in_legend=(nonzero_indices not in seen_groups))
        seen_groups.add(nonzero_indices)
        # p.add(f"{index} projection", *projection.T, marker_size=4, marker_line_width=1, color=1, group=index)


if (d in {2,3}):
    p.graph(show_grid=False, scene_settings=dict(aspectmode="data"), show_legend=True)


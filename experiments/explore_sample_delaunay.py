# Demonstration of Delaunay interpolation for analytic test function.

import numpy as np
from delaunay_py import delsparse
from tlux.random import well_spaced_ball
from tlux.plot import Plot


# Scale up a numpy value by multiplying by 1.0 + "p".
# Keyword arguments are passed to "np.mean" and "np.std".
def scale_up(vals, p=0.1, **kwargs):
    vals = vals.copy()
    shift = -np.mean(vals, **kwargs)
    vals += shift
    scale = np.std(vals, **kwargs)
    vals /= scale
    vals *= (1.0 + p)
    vals *= scale
    vals -= shift
    return vals

print("configure", flush=True)
n = 1000
d = 2
shift = -0.1
scale = 4 * np.pi

print("compute x and y", flush=True)
x = well_spaced_ball(num_points=n, dimension=d)
y = np.sin(np.linalg.norm(x + shift, axis=1) * scale)

print("make plot", flush=True)
p = Plot()

print("plot data", flush=True)
p.add("data", *x.T, y)

print("init model", flush=True)

# Define a function that evaluates the Delaunay simplicial mesh.
def fhat(q):
    simps, weights, errors = delsparse(pts=x, q=q)
    values = np.zeros(len(q), dtype="float64")
    for i in range(len(q)):
        values[i] = y[simps[i]] @ weights[i]
    return values

# Evaluate the model.
print("plot func", flush=True)
min_max = scale_up(np.percentile(x, [0,100], axis=0), axis=0)
p.add_func("fit", fhat, *min_max.T, plot_points=5000, vectorized=True)

print("show plot", flush=True)
p.show(z_range=[-1.5, 2.5])

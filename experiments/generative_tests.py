# Run a sequence of generative tests to measure error rates.

# Python builtins.
import os
import math
import time
import pickle
import random
import hashlib
import functools
import itertools
# External library.
import numpy as np
# 'tlux' imports
from tlux.math import regular_simplex, orthogonalize
from tlux.plot import Plot
from tlux.random import mesh as meshgrid
from tlux.random import well_spaced_ball
from tlux.random import well_spaced_box
from tlux.random import well_spaced_sphere
from tlux.system import Timer
# Local utilities.
from delaunay_py import delsparse


# Concepts for the generative testing scheme:
# 
# Different data geometries
# - lattice
# - [well spaced] sphere | ball | box
# 
# Different transformations
# - skew that is linear | exponential
# - rotation
# 
# Different variables
# - dimension
# - number of points
# - strength of skew
# 
# Different measurements
# - percentage of extrapolation points that fail (by error code)
# - directional bias of extrapolation points that fail (by error code)
# - stats about simplices that extrapolation points landed in (successfully)
# - stats about simplices in general (for interpolation points)


# Core modifiers.
def identity(gen):
    @functools.wraps(gen)
    def new_gen(*args, skew=1, scale=1, **kwargs):
        return scale * gen(*args, **kwargs)
    return new_gen

def linear_skew(gen, base=2):
    @functools.wraps(gen)
    def new_gen(*args, skew=1, scale=1, **kwargs):
        result = gen(*args, **kwargs)
        multiplier = np.linspace(1, skew, result.shape[1])
        return scale * multiplier * result
    return new_gen

def exponential_skew(gen, base=2):
    @functools.wraps(gen)
    def new_gen(*args, skew=1, scale=1, **kwargs):
        result = gen(*args, **kwargs)
        multiplier = base ** np.linspace(0, math.log(skew, base), result.shape[1])
        return scale * multiplier * result
    return new_gen

# Core generators.
def box(n, d):
    return well_spaced_box(n, d)

def ball(n, d):
    return well_spaced_ball(n, d)

def lattice(n, d):
    n_on_axis = max(2, math.ceil(n ** (1/d)))
    x = meshgrid(np.linspace(0, 1, n_on_axis), dimension=d, sample=n)
    x -= x.mean(axis=0)
    x /= abs(x).max(axis=0)
    return x


# Enumerate all modifiers and generators combined.
data_modifiers = [
    ("", identity),
    ("exponential_skew", exponential_skew),
    ("linear_skew", linear_skew),
]

data_generators = [
    ("lattice", lattice),
    ("box", box),
    ("ball", ball),
]
data_set_creators = []
for (modifier_name, modifier) in data_modifiers:
    for (generator_name, generator) in data_generators:
        data_set_creators.append((
            modifier_name + ("_" if len(modifier_name) > 0 else "") + generator_name,
            modifier(generator)
        ))

# All possible parameter values.
data_variable_values = list(map(dict, itertools.product(
    [("d", v) for v in (2, 8, 16)],
    [("n", v) for v in (1, 2**9, 2**14)],
    [("skew", v) for v in (1, 2**10, 2**20)],
    [("scale", v) for v in (2**(-20), 1, 2**63)],
)))

max_attempts_to_make_nondegenerate_data = 100
num_trials = 10
output_dir = f"generative_results"
os.makedirs(output_dir, exist_ok=True)


# When this is the 'main' program, execute the tests (but not when it is imported).
if __name__ == "__main__":
    # Iterate over both Delaunay versions.
    for trial in range(num_trials):
        # Run all experiments.
        for (name, creator) in data_set_creators:
            for kwargs in data_variable_values:
                kwargs_hash = abs(int.from_bytes(hashlib.sha256(pickle.dumps(sorted(kwargs.items()))).digest())) % (2**31)
                for version in (1, 2):
                    # Make sure we get consistent data generation.
                    seed = kwargs_hash + trial
                    np.random.seed(seed)
                    random.seed(seed)
                    # Skip skew for creators that don't have any skew option.
                    if ("skew" not in name) and (kwargs["skew"] != 1):
                        continue
                    # Ensure that "n" is at least "d+1".
                    kwargs["n"] = max(kwargs["n"], kwargs["d"]+1)
                    # Generate the path for saving the results.
                    data_path = os.path.join(output_dir, f"v{version}-{name}-{trial+1}_" + "_".join((k+f"={v}" for (k,v) in kwargs.items())) + ".pkl")
                    print(time.strftime("%H:%M:%S "), data_path)
                    if os.path.exists(data_path):
                        with open(data_path, "rb") as f:
                            data = pickle.load(f)
                            print(" ", data["prediction_time"], "seconds,",
                                  data["num_errors"], "errors with codes",
                                  [e for e in data["errors"].keys() if e != 1])
                            continue
                    # Run the experiment.
                    t = Timer()
                    for _ in range(max_attempts_to_make_nondegenerate_data):
                        x = creator(**kwargs)
                        _, lengths = orthogonalize(x)  # Treats columns as vectors.
                        # Avoid degenerate data by ensuring that the columns (dimensions) are not degenerate.
                        #  Use EPSILON(1.0_REAL64) as the cutoff value for being "nonzero".
                        if (min(lengths) > 2**(-52)):
                            break
                    else:
                        raise(RuntimeError("Failed to generate points that are not degenerate."))
                    # TODO: Generate "in distribution" extrapolation points come from same generator?
                    # TODO: Tighter center is actually between the two most separated points.
                    center = x.mean(axis=0)
                    radius = np.sqrt(((x-center)**2).sum(axis=1)).max()
                    data_gen_time = t.stop()
                    # Generate test points, shift and scale them so they are surrounding the box.
                    num_test_points = 1000
                    extrapolation_ratio = 1.00
                    ibudget = 1000
                    points = extrapolation_ratio * radius * well_spaced_sphere(
                        num_points=num_test_points, dimension=kwargs["d"]
                    ) + center
                    # Compute the simplices for all extrapolation points.
                    t = Timer()
                    simps, weights, errors = delsparse(x, points, eps=None, ibudget=ibudget, extrap=2.0**52, version=version)
                    prediction_time = t.stop()
                    num_extrapolations = len(errors[1])
                    num_errors = sum(map(len, errors.values())) - num_extrapolations
                    print(" ", t, "seconds,", num_errors, "errors,", min(lengths), "from degenerate")
                    # Save the data and results to a file.
                    with open(data_path, "wb") as f:
                        pickle.dump(dict(
                            x=x,
                            center=center,
                            radius=radius,
                            data_gen_time=data_gen_time,
                            num_test_points=num_test_points,
                            extrapolation_ratio=extrapolation_ratio,
                            points=points,
                            simps=simps,
                            weights=weights,
                            errors=errors,
                            prediction_time=prediction_time,
                            num_extrapolations=num_extrapolations,
                            num_errors=num_errors,
                        ), f)


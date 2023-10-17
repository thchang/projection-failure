import numpy as np
import ds_v1.DelaunaySparse.python.delsparse as delsparse_v1
import ds_v2.DelaunaySparse.python.delsparse as delsparse_v2

# Use the appropriate version of Delaunay to identify the containing
#  simplex and weights of a given set of query points.
def delsparse(pts, q, eps=2**(-26), ibudget=10000, extrap=100.0,
              version=1, parallel=True, print_errors=True, **kwargs):
    # Import the source delaunaysparse library.
    if (version == 1):
        delsparse = delsparse_v1
    elif (version == 2):
        delsparse = delsparse_v2
    else:
        raise(ValueError(f"Bad version provided {version}, expected 1 or 2."))
    # Determine whether to use the parallel or serial version of delaunaysparse.
    if parallel:
        delsparse = delsparse.delaunaysparsep
    else:
        delsparse = delsparse.delaunaysparses
    # Get the predictions from Delaunay Fortran code.
    d = pts.shape[1]
    n = pts.shape[0]
    pts = np.array(np.asarray(pts, dtype="float64").T, order="F")
    m = q.shape[0]
    q = np.array(np.asarray(q, dtype="float64").T, order="F")
    simps = np.ones(shape=(d+1, m), dtype="int32", order="F")
    weights = np.ones(shape=(d+1, m), dtype="float64", order="F")
    ierr = np.ones(shape=(m,), dtype="int32", order="F")
    interp_in = y = None
    delsparse(d, n, pts, m, q, simps, weights, ierr,
              interp_in=interp_in, interp_out=y,
              ibudget=ibudget, eps=eps, extrap=extrap,
              **kwargs)
    # Initialize a holder for errors that will be returned.
    errors = {}
    # Store and remove extrapolation indicators for ierr.
    errors[1] = np.arange(len(ierr))[ierr == 1]
    ierr = np.where(ierr == 1, 0, ierr)
    # Handle any errors that may have occurred.
    if (ierr.sum() > 0):
        if print_errors:
            unique_errors = sorted(np.unique(ierr))
            print(" [Delaunay errors:",end="")
            for e in unique_errors:
                if (e == 0): continue
                errors[e] = np.arange(len(ierr))[ierr == e]
                print(" %3i"%e,"at","{"+",".join(tuple(
                    str(i) for i in range(len(ierr))
                    if (ierr[i] == e)))+"}", end=";")
            print("] ")
        # Reset the errors to simplex of 0s (to be -1) and weights of 0s.
        bad_indices = (ierr > 0)
        simps[:,bad_indices] = 0
        weights[:,bad_indices] = 0.0
    # Return the weights and indices.
    return (simps-1).T, weights.T, errors


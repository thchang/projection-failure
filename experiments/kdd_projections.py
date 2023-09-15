import csv
import ctypes
import cvxpy as cp
import ds_v1.DelaunaySparse.python.delsparse as delsparse_v1
import ds_v2.DelaunaySparse.python.delsparse as delsparse_v2
import numpy as np
import random
import warnings

# Initialize data directory and metadata
d = 0; n = 0; m = 0
fname = "../data/KDDCUP99_DEL_format.dat"

# Load the data set for testing

with open(fname, "r") as fp:
    csv_reader = csv.reader(fp)
    for i, rowi in enumerate(csv_reader):
        if i == 0:
            print("\n\n")
            print(f"="*80)
            print(f"New problem: {rowi[0]}")
            print(f"="*80)
            print()
        elif i == 1:
            d = int(rowi[0])
            n = int(rowi[1])
            m = int(rowi[2])
            train = np.zeros((d, n))
            test = np.zeros((d, m))
        elif i - 2 < n:
            for j, colj in enumerate(rowi):
                train[j, i-2] = float(colj.strip())
        else:
            for j, colj in enumerate(rowi):
                test[j, i-2-n] = float(colj.strip())

# Generate various training sizes
for nk in [1000, 5000, 10000, 15000, 20000]:

    print()
    print("="*80)
    print(f"Training size: {nk}")
    print("="*80)
    print()

    # Subsample the data
    mk = 100
    pts = np.zeros((d, nk))
    q = np.zeros((d, mk))

    # Use 10 different random seeds
    for j in range(1,11):

        print()
        print(f"New seed: {j}")
        print("-"*80)

        # Reset number of test points to sample
        mk = 100

        # Randomly select nk train pts and mk test pts
        np.random.seed(j)
        itrain = np.random.choice(n, nk)
        itest = np.random.choice(m, mk)
        pts[:, :] = train[:,itrain]
        q[:, :] = test[:,itest]

        # Solve with DELAUNAYSPARSE v2 (BQPD method)

        # Copy pts, q into Fortran contiguous array
        p_in = np.zeros(shape=pts.shape, dtype=ctypes.c_double, order="F")
        for i, pi in enumerate(pts):
            p_in[i, :] = pi[:]
        q_in = np.zeros(shape=q.shape, dtype=ctypes.c_double, order="F")
        for i, qi in enumerate(q):
            q_in[i, :] = qi[:]
        # Allocate output arrays
        error_out = np.ones(shape=(mk,), dtype=np.int32, order="F")
        rnorm_out = np.ones(shape=(mk,), dtype=ctypes.c_double, order="F")
        # Call DelaunaySparse v2
        for i, qi in enumerate(q_in.T):
            _, rnorm_out[i], error_out[i], _ = delsparse_v2.project(d, nk, p_in, qi)
        # Count the number of failures and print
        error_count = 0
        extrap_pts = []
        for i, ierr in enumerate(error_out):
            if ierr == 0:
                if rnorm_out[i] > 1.0e-10:
                    extrap_pts.append(i)
            elif ierr not in range(70, 80):
                print(f"WARNING: an unexpected error occurred: {ierr}")
            else:
                error_count += 1
        # Update number of extrapolation points
        print(f"Extrapolation points: {len(extrap_pts)} / {mk}")
        mk = len(extrap_pts)
        print("Method: DelaunaySparse v2 (BQPD),\t" +
              f"% solved: {100 - (100 * error_count / mk)}")

        # Solve with DELAUNAYSPARSE v1 (SLATEC / DWNNLS method)

        # Copy pts, q into Fortran contiguous array
        p_in = np.zeros(shape=pts.shape, dtype=ctypes.c_double, order="F")
        for i, pi in enumerate(pts):
            p_in[i, :] = pi[:]
        q_in = np.zeros(shape=q[:,extrap_pts].shape, dtype=ctypes.c_double, order="F")
        for i, qi in enumerate(q[:,extrap_pts]):
            q_in[i, :] = qi[:]
        # Allocate output arrays
        simp_out = np.ones(shape=(d+1, mk), dtype=np.int32, order="F")
        weights_out = np.ones(shape=(d+1, mk), dtype=ctypes.c_double, order="F")
        error_out = np.ones(shape=(mk,), dtype=np.int32, order="F")
        # Call DelaunaySparse v1
        delsparse_v1.delaunaysparses(d, nk, p_in, mk, q_in, simp_out,
                                     weights_out, error_out, extrap=100.0)
        # Count the number of failures and print
        error_count = 0
        for ierr in error_out:
            if ierr <= 2:
                continue
            elif ierr != 71:
                print(f"WARNING: an unexpected error occurred: {ierr}")
            else:
                error_count += 1
        print("Method: DelaunaySparse v1 (DWNNLS),\t" +
              f"% solved: {100 - (100 * error_count / mk)}")

        # Solve with CVXPY with the OSQP solver

        # Loop over all extrapolation points and count the number of errors
        error_count = 0
        for qi in q[:,extrap_pts].T:
            # Allocate memory for problem
            A = pts.copy()
            b = qi.copy()
            x = cp.Variable(A.shape[1])
            # Define the QP
            prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)),
                              [cp.sum(x) == 1, x >= 0])
            # Try to solve the QP and count the number of errors
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    prob.solve(
                               solver="OSQP",
                               max_iter=10000,
                               eps_rel=1.8e-12,
                               # verbose=True
                              )
                ### Compute the actual projection.
                ### Uncomment below to calculate z_hat = <x, pts>.
                # z_hat = np.dot(x.value, pts.T).flatten()
            except:
                error_count += 1
        print(f"Method: CVXPY {cp.__version__} (OSQP solver),\t" +
              f"% solved: {100 - (100 * error_count / mk)}")

        # Solve with CVXPY with the ECOS solver

        # Loop over all extrapolation points and count the number of errors
        error_count = 0
        for qi in q[:,extrap_pts].T:
            # Allocate memory for problem
            A = pts.copy()
            b = qi.copy()
            x = cp.Variable(A.shape[1])
            # Define the QP
            prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)),
                              [cp.sum(x) == 1, x >= 0])
            # Try to solve the QP and count the number of errors
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    prob.solve(
                               solver="ECOS",
                               max_iters=10000,
                               reltol=1.8e-12,
                               # verbose=True,
                              )
                ### Compute the actual projection.
                ### Uncomment below to calculate z_hat = <x, pts>.
                # z_hat = np.dot(x.value, pts.T).flatten()
            except:
                error_count += 1
        print(f"Method: CVXPY {cp.__version__} (ECOS solver),\t" +
              f"% solved: {100 - (100 * error_count / mk)}")

        # Solve with CVXPY with the SCS solver

        # Loop over all extrapolation points and count the number of errors
        error_count = 0
        for qi in q[:,extrap_pts].T:
            # Allocate memory for problem
            A = pts.copy()
            b = qi.copy()
            x = cp.Variable(A.shape[1])
            # Define the QP
            prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)),
                              [cp.sum(x) == 1, x >= 0])
            # Try to solve the QP and count the number of errors
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    prob.solve(
                               solver="SCS",
                               max_iters=10000,
                               eps=1.8e-12,
                               # verbose=True,
                              )
                ### Compute the actual projection.
                ### Uncomment below to calculate z_hat = <x, pts>.
                # z_hat = np.dot(x.value, pts.T).flatten()
            except:
                error_count += 1
        print(f"Method: CVXPY {cp.__version__} (SCS solver),\t" +
              f"% solved: {100 - (100 * error_count / mk)}")

print("\nDone.\n\n")

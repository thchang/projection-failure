import csv
import cvxpy as cp
import ds_v1.DelaunaySparse.python.delsparse as delsparse_v1
import ds_v2.DelaunaySparse.python.delsparse as delsparse_v2
import numpy as np
import os
import warnings

# Initialize data directory and metadata
data_dir = "../data"
d = 0; n = 0; m = 0

# Loop over all datasets in the data_dir
for fname in os.scandir(data_dir):
    if fname.is_file():

        # Load the data set for testing

        with open(fname.path, "r") as fp:
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
                    pts = np.zeros((d, n))
                    q = np.zeros((d, m))
                elif i - 2 < n:
                    for j, colj in enumerate(rowi):
                        pts[j, i-2] = float(colj.strip())
                else:
                    for j, colj in enumerate(rowi):
                        q[j, i-2-n] = float(colj.strip())

            ### DEBUG STUFF below, uncomment to run ###
            # print(pts)
            # print(q)
            # print(d, n, m)
            # print(pts.shape)
            # print(q.shape)

        # Solve with DELAUNAYSPARSE v1 (SLATEC / DWNNLS method)

        # Copy pts, q into Fortran contiguous array
        p_in = np.zeros(shape=pts.shape, dtype=np.float64, order="F")
        for i, pi in enumerate(pts):
            p_in[i, :] = pi[:]
        q_in = np.zeros(shape=q.shape, dtype=np.float64, order="F")
        for i, qi in enumerate(q):
            q_in[i, :] = qi[:]
        # Allocate output arrays
        simp_out = np.ones(shape=(d+1, m), dtype=np.int32, order="F")
        weights_out = np.ones(shape=(d+1, m), dtype=np.float64, order="F")
        error_out = np.ones(shape=(m,), dtype=np.int32, order="F")
        # Call DelaunaySparse v1
        delsparse_v1.delaunaysparses(d, n, p_in, m, q_in, simp_out,
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
              f"% solved: {100 - (100 * error_count / m)}")

        # Solve with DELAUNAYSPARSE v2 (BQPD method)

        # Copy pts, q into Fortran contiguous array
        p_in = np.zeros(shape=pts.shape, dtype=np.float64, order="F")
        for i, pi in enumerate(pts):
            p_in[i, :] = pi[:]
        q_in = np.zeros(shape=q.shape, dtype=np.float64, order="F")
        for i, qi in enumerate(q):
            q_in[i, :] = qi[:]
        # Allocate output arrays
        simp_out = np.ones(shape=(d+1, m), dtype=np.int32, order="F")
        weights_out = np.ones(shape=(d+1, m), dtype=np.float64, order="F")
        error_out = np.ones(shape=(m,), dtype=np.int32, order="F")
        # Call DelaunaySparse v2
        delsparse_v2.delaunaysparses(d, n, p_in, m, q_in, simp_out,
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
        print("Method: DelaunaySparse v2 (BQPD),\t" +
              f"% solved: {100 - (100 * error_count / m)}")

        # Solve with CVXPY with the OSQP solver

        # Loop over all extrapolation points and count the number of errors
        error_count = 0
        for qi in q.T:
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
              f"% solved: {100 - (100 * error_count / m)}")

        # Solve with CVXPY with the ECOS solver

        # Loop over all extrapolation points and count the number of errors
        error_count = 0
        for qi in q.T:
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
              f"% solved: {100 - (100 * error_count / m)}")

        # Solve with CVXPY with the SCS solver

        # Loop over all extrapolation points and count the number of errors
        error_count = 0
        for qi in q.T:
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
              f"% solved: {100 - (100 * error_count / m)}")

print("\nDone.\n\n")

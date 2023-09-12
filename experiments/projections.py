import csv
import cvxpy as cp
import ds_v1.DelaunaySparse.python.delsparse as delsparse_v1
import ds_v2.DelaunaySparse.python.delsparse as delsparse_v2
import numpy as np
import os

data_dir = "../data"

d = 0; n = 0; m = 0
for fname in os.scandir(data_dir):
    if fname.is_file():
        with open(fname.path, "r") as fp:
            csv_reader = csv.reader(fp)
            for i, rowi in enumerate(csv_reader):
                if i == 0:
                    print(f"New problem: {rowi[0]}")
                elif i == 1:
                    d = int(rowi[0])
                    n = int(rowi[1])
                    m = int(rowi[2])
                    pts = np.zeros((n, d))
                    q = np.zeros((m, d))
                elif i - 2 < n:
                    for j, colj in enumerate(rowi):
                        pts[i-2, j] = float(colj.strip())
                else:
                    for j, colj in enumerate(rowi):
                        q[i-2-n, j] = float(colj.strip())
            ## DEBUG STUFF below
            print(pts)
            print(q)
            print(d, n, m)
            print(pts.shape)
            print(q.shape)

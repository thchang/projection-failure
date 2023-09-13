# Geometric Projection Failures for Remark on ACM TOMS Algorithm 1012

This repository contains demonstrations of failure cases for various
quadratic program (QP) solver codes on the projection problem from
ACM TOMS 1012.

## Problem Details

At certain points in ACM TOMS Algorithm 1012, one needs to compute the
projection of an extrapolation point $z$ onto the convex hull of the
training points $P$, denoted $CH(P)$.

This is done by solving
$$
\min_{x\in\mathbb{R}^n} || Wx - z ||^2
$$
such that
$$ \sum x = 1; x \geq 0$$
where $P$ consists of $n$ points in $\mathbb{R}^d$,
and $W = [p_1 | p_2 | ... | p_n]$ for
$p_i \in P$.

Note that this problem can be posed as an equality-constrained
nonnegative least-squares problem, which is a special case of a
convex QP.

Due to the unusual shape of the matrix $W$, this is a nonstandard
use-case for many QP solvers and an extremely difficult problem
when the number of training points ($n$) is large.
This results in most open-source solvers failing to achieve the
necessary accuracy.

We have proposed the usage of the BQPD solver of R. Fletcher to solve
this problem, using a dot-product kernel that exploits the structure
of the problem.
The purpose of this repository is to demonstrate this solution compared
to other open-source QP solvers and the SLATEC DWNNLS solver, which was
used in the original DelaunaySparse code.

## Setup

Before attempting to reproduce our results, take the following steps
to install and build dependencies:

 - First make sure that you have a copy of ``gfortran`` installed on
   your machine (using the shell command ``gfortran`` to compile);
 - Next make sure that you have ``python 3.8`` or newer;
 - ``pip install`` the ``REQUIREMENTS.txt`` file in the base directory;
 - Fetch the old and new versions of DelaunaySparse into the ``experiments``
   subdirectory using the command
   ``git pull --recurse-submodules``
 - To build and test DelaunaySparse's shared object libraries, use the
   command
   ``cd experiments/ds_v1/DelaunaySparse/python && python example.py``
   and
   ``cd experiments/ds_v2/DelaunaySparse/python && python example.py``

## Reproducing Results

After following the above steps:

 - The ``experiments`` subdirectory contains the script and dependencies that
   reproduce our results.
   To run it: ``cd experiments && python projection.py``
 - the ``data`` subdirectory contains a list of csv data files containing the
   test problems used. The first line is the name of the problem, the second
   line contains the integer values ``d,n,m`` in that order. The next ``n``
   lines contain the data points (defining the convex hull) as
   comma-separated row-vectors.
   The last ``m`` lines contain the points to project onto the convex hull
   as comma-separated row-vectors.

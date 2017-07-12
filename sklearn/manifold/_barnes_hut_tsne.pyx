# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# Implementation by Chris Moody & Nick Travers
# See http://homepage.tudelft.nl/19j49/t-SNE.html for reference
# implementations and papers describing the technique


from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport sqrt, log
import numpy as np
cimport numpy as np

from sklearn.neighbors import quad_tree
from sklearn.neighbors cimport quad_tree

cdef char* EMPTY_STRING = ""

cdef extern from "math.h":
    float fabsf(float x) nogil

# Smallest strictly positive value that can be represented by floating
# point numbers for different precision levels. This is useful to avoid
# taking the log of zero when computing the KL divergence.
cdef float FLOAT32_TINY = np.finfo(np.float32).tiny

# Useful to void division by zero or divergence to +inf.
cdef float FLOAT64_EPS = np.finfo(np.float64).eps

# This is effectively an ifdef statement in Cython
# It allows us to write printf debugging lines
# and remove them at compile time
cdef enum:
    DEBUGFLAG = 0

cdef extern from "time.h":
    # Declare only what is necessary from `tm` structure.
    ctypedef long clock_t
    clock_t clock() nogil
    double CLOCKS_PER_SEC


cdef float compute_gradient(float[:] val_P,
                            float[:, :] pos_reference,
                            np.int64_t[:] neighbors,
                            np.int64_t[:] indptr,
                            float[:, :] tot_force,
                            quad_tree._QuadTree qt,
                            float theta,
                            float dof,
                            long start,
                            long stop) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef:
        long i, coord
        int ax
        long n_samples = pos_reference.shape[0]
        int n_dimensions = qt.n_dimensions
        double[1] sum_Q
        clock_t t1, t2
        float sQ, error

    if qt.verbose > 11:
        printf("[t-SNE] Allocating %li elements in force arrays\n",
                n_samples * n_dimensions * 2)
    cdef float* neg_f = <float*> malloc(sizeof(float) * n_samples * n_dimensions)
    cdef float* pos_f = <float*> malloc(sizeof(float) * n_samples * n_dimensions)

    sum_Q[0] = 0.0
    t1 = clock()
    compute_gradient_negative(pos_reference, neg_f, qt, sum_Q,
                              dof, theta, start, stop)
    t2 = clock()
    if qt.verbose > 15:
        printf("[t-SNE] Computing negative gradient: %e ticks\n", ((float) (t2 - t1)))
    sQ = sum_Q[0]
    t1 = clock()
    error = compute_gradient_positive(val_P, pos_reference, neighbors, indptr,
                                      pos_f, n_dimensions, dof, sQ, start,
                                      qt.verbose)
    t2 = clock()
    if qt.verbose > 15:
        printf("[t-SNE] Computing positive gradient: %e ticks\n", ((float) (t2 - t1)))
    for i in range(start, n_samples):
        for ax in range(n_dimensions):
            coord = i * n_dimensions + ax
            tot_force[i, ax] = pos_f[coord] - (neg_f[coord] / sQ)

    free(neg_f)
    free(pos_f)
    return error


cdef float compute_gradient_positive(float[:] val_P,
                                     float[:, :] pos_reference,
                                     np.int64_t[:] neighbors,
                                     np.int64_t[:] indptr,
                                     float* pos_f,
                                     int n_dimensions,
                                     float dof,
                                     double sum_Q,
                                     np.int64_t start,
                                     int verbose) nogil:
    # Sum over the following expression for i not equal to j
    # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
    # This is equivalent to compute_edge_forces in the authors' code
    # It just goes over the nearest neighbors instead of all the data points
    # (unlike the non-nearest neighbors version of `compute_gradient_positive')
    cdef:
        int ax
        long i, j, k
        long n_samples = indptr.shape[0] - 1
        float dij, qij, pij
        float C = 0.0
        float exponent = (dof + 1.0) / -2.0
        float[3] buff
        clock_t t1, t2

    t1 = clock()
    for i in range(start, n_samples):
        # Init the gradient vector
        for ax in range(n_dimensions):
            pos_f[i * n_dimensions + ax] = 0.0
        # Compute the positive interaction for the nearest neighbors
        for k in range(indptr[i], indptr[i+1]):
            j = neighbors[k]
            dij = 0.0
            pij = val_P[k]
            for ax in range(n_dimensions):
                buff[ax] = pos_reference[i, ax] - pos_reference[j, ax]
                dij += buff[ax] * buff[ax]
            qij = (((1.0 + dij) / dof) ** exponent)
            dij = pij * qij
            qij /= sum_Q
            C += pij * log(max(pij, FLOAT32_TINY)
                           / max(qij, FLOAT32_TINY))
            for ax in range(n_dimensions):
                pos_f[i * n_dimensions + ax] += dij * buff[ax]
    t2 = clock()
    dt = ((float) (t2 - t1))
    if verbose > 10:
        printf("[t-SNE] Computed error=%1.4f in %1.1e ticks\n", C, dt)
    return C


cdef void compute_gradient_negative(float[:, :] pos_reference,
                                    float* neg_f,
                                    quad_tree._QuadTree qt,
                                    double* sum_Q,
                                    float dof,
                                    float theta,
                                    long start,
                                    long stop) nogil:
    if stop == -1:
        stop = pos_reference.shape[0]
    cdef:
        int ax
        int n_dimensions = qt.n_dimensions
        long i, j, idx
        long n = stop - start
        long dta = 0
        long dtb = 0
        long offset = n_dimensions + 2
        long* l
        float size, dist2s, mult
        double qijZ
        float[1] iQ
        float[3] force, neg_force, pos
        clock_t t1, t2, t3

    summary = <float*> malloc(sizeof(float) * n * offset)

    for i in range(start, stop):
        # Clear the arrays
        for ax in range(n_dimensions):
            force[ax] = 0.0
            neg_force[ax] = 0.0
            pos[ax] = pos_reference[i, ax]
        iQ[0] = 0.0
        # Find which nodes are summarizing and collect their centers of mass
        # deltas, and sizes, into vectorized arrays
        t1 = clock()
        idx = qt.summarize(pos, summary, theta*theta)
        t2 = clock()
        # Compute the t-SNE negative force
        # for the digits dataset, walking the tree
        # is about 10-15x more expensive than the
        # following for loop
        exponent = (dof + 1.0) / -2.0
        for j in range(idx // offset):

            dist2s = summary[j * offset + n_dimensions]
            size = summary[j * offset + n_dimensions + 1]
            qijZ = ((1.0 + dist2s) / dof) ** exponent  # 1/(1+dist)
            sum_Q[0] += size * qijZ   # size of the node * q
            mult = size * qijZ * qijZ
            for ax in range(n_dimensions):
                neg_force[ax] += mult * summary[j * offset + ax]
        t3 = clock()
        for ax in range(n_dimensions):
            neg_f[i * n_dimensions + ax] = neg_force[ax]
        dta += t2 - t1
        dtb += t3 - t2
    if qt.verbose > 20:
        printf("[t-SNE] Tree: %li clock ticks | ", dta)
        printf("Force computation: %li clock ticks\n", dtb)

    # Put sum_Q to machine EPSILON to avoid divisions by 0
    sum_Q[0] = max(sum_Q[0], FLOAT64_EPS)
    free(summary)


def gradient(float[:] val_P,
             float[:, :] pos_output,
             np.int64_t[:] neighbors,
             np.int64_t[:] indptr,
             float[:, :] forces,
             float theta,
             int n_dimensions,
             int verbose,
             float dof = 1.0,
             long skip_num_points=0):
    # This function is designed to be called from external Python
    # it passes the 'forces' array by reference and fills thats array
    # up in-place
    cdef float C
    n = pos_output.shape[0]
    assert val_P.itemsize == 4
    assert pos_output.itemsize == 4
    assert forces.itemsize == 4
    m = "Forces array and pos_output shapes are incompatible"
    assert n == forces.shape[0], m
    m = "Pij and pos_output shapes are incompatible"
    assert n == indptr.shape[0] - 1, m
    if verbose > 10:
        printf("[t-SNE] Initializing tree of n_dimensions %i\n", n_dimensions)
    cdef quad_tree._QuadTree qt = quad_tree._QuadTree(pos_output.shape[1],
                                                      verbose)
    if verbose > 10:
        printf("[t-SNE] Inserting %li points\n", pos_output.shape[0])
    qt.build_tree(pos_output)
    if verbose > 10:
        # XXX: format hack to workaround lack of `const char *` type
        # in the generated C code that triggers error with gcc 4.9
        # and -Werror=format-security
        printf("[t-SNE] Computing gradient\n%s", EMPTY_STRING)
    C = compute_gradient(val_P, pos_output, neighbors, indptr, forces,
                         qt, theta, dof, skip_num_points, -1)
    if verbose > 10:
        # XXX: format hack to workaround lack of `const char *` type
        # in the generated C code
        # and -Werror=format-security
        printf("[t-SNE] Checking tree consistency\n%s", EMPTY_STRING)
    m = "Tree consistency failed: unexpected number of points on the tree"
    assert qt.cells[0].cumulative_size == qt.n_points, m
    return C

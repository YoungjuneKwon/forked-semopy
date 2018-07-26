import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport isnan
from cython cimport view

cdef object function
cdef object gradient
cdef object hessian

# iv positions
DEF max_fcalls = 16
DEF max_iters  = 17
## Must be zeros:
#DEF outlev = 18
#DEF parprt = 19
#DEF prunit = 20
#DEF solprt = 21
#DEF statpr = 22
#DEF x0prt  = 23 
#
# v positions
DEF abstol = 30
DEF reltol = 31
DEF xtol   = 32
DEF xftol  = 33


cdef int call_f(int *n, double *x, int *nf, double *f, int *uip, double *urp,
                U_fp ufp):
    global function
    global count
    cdef view.array xt = <double[:n[0]]>x
    f[0] = <double>function(xt)
    if isnan(f[0]):
        nf[0] = 0
    return 0  

cdef int call_g(int *n, double *x, int *nf, double *g, int *uip, double *urp,
                U_fp ufp):
    global gradient
    cdef view.array xt = <double[:n[0]]>x
    cdef view.array gt = <double[:n[0]]>g
    cdef double[:] gr = gradient(xt)
    gt[:] = gr[:]
    if np.any(np.isnan(gt)):
        nf[0] = 0
    return 0  

cdef int call_h(int *n, double *x, int *nf, double *g, double *h,
                int *uip, double *urp, U_fp ufp):
    global gradient
    global hessian
    cdef int N = n[0]
    cdef int Nh = N * (N + 1) // 2
    cdef view.array xt = <double[:N]>x
    cdef view.array gt = <double[:N]>g
    cdef view.array ht = <double[:Nh]>h
    cdef double[:] gr = gradient(xt)
    cdef double[:] hr = hessian(xt)
    gt[:] = gr[:]
    ht[:] = hr[:]
    if np.any(np.isnan(gt)):
        nf[0] = 0
    return 0  


def minimize(f, double[:] x0, grad=None, hess=None, maxiter=1000,
             maxfcalls=2000, abs_tol=1e-18, rel_tol=1e-10, x_tol=1.5e-8,
             print_info=False):
    cdef double[:] x = x0[:]
    cdef int iv[60]
    cdef int liv = 60 
    cdef int n = len(x0)
    cdef int lv = 78 + n * (n + 12)
    cdef double *v = <double*>malloc(lv * sizeof(double))
    cdef double *d = <double*>malloc(n * sizeof(double))
    cdef int ui = 2
    global function
    global gradient
    global hessian
    function = f; gradient = grad; hessian = hess
    
    deflt(&ui, iv, &liv, &lv, v)
    iv[max_fcalls] = maxfcalls
    iv[max_iters] = maxiter
    v[abstol] = abs_tol
    v[reltol] = rel_tol
    v[xtol] = x_tol
    for i in range(n):
        d[i] = 1.0
    if grad is None:
        smsno(&n, d, &x[0], <S_fp>call_f, &iv[0], &liv, &lv, v,
              NULL, NULL, NULL)
    elif hess is None:
        sumsl(&n, d, &x[0], <S_fp>call_f, <S_fp>call_g, &iv[0], &liv, &lv, v,
              NULL, NULL, NULL)
    else:
        humsl(&n, d, &x[0], <S_fp>call_f, <S_fp>call_h, &iv[0], &liv, &lv, v,
              NULL, NULL, NULL)
        pass
    free(v)
    free(d)
    if print_info:
        print("Error code: {}\nnfcalls: {} ngcalls: {} niters: {}\nFunction value: {}".format(iv[0], iv[5], iv[29], iv[30], v[9]))
    return x
    
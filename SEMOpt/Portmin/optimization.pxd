ctypedef int integer;
ctypedef float real;
ctypedef double doublereal;
ctypedef long int logical;
ctypedef int (*S_fp)();
ctypedef int (*U_fp)();


cdef extern from "toms611.c":
    int deflt "deflt_" (integer *alg, integer *iv, integer *liv, integer *lv,
                        doublereal *v)
    int sumsl "sumsl_" (integer *n, doublereal *d, doublereal *x,
                        S_fp calcf, S_fp calcg, integer *iv, integer *liv,
                        integer *lv, doublereal *v, integer *uiparm,
                        doublereal *urparm, U_fp ufparm)
    int humsl "humsl_"(integer *n, doublereal *d__, doublereal *x, 
                       S_fp cf, S_fp calcgh, integer *iv, integer *liv,
                       integer *lv, doublereal *v, integer *uiparm,
                       doublereal *urparm, U_fp ufparm)
    int smsno "smsno_"(integer *n, doublereal *d__, doublereal *x,
                       S_fp calcf, integer *iv, integer *liv, integer *lv,
                       doublereal *v, integer *uiparm, doublereal *urparm,
                       U_fp ufparm)
// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_LAPACK_COMPLETE_H
#define ROL_LAPACK_COMPLETE_H

#include "ROL_config.h"

/** \class ROL::LAPACK
  \brief Complete LAPACK interface for ROL (adapted from Teuchos)
  
  This provides a complete LAPACK interface similar to Teuchos but scoped
  within the ROL namespace for standalone builds.
  */

/* A) Define PREFIX and ROL_fcd based on platform. */
#if defined(INTEL_CXML)
#  define PREFIX __stdcall
#  define ROL_fcd const char *, unsigned int
#elif defined(INTEL_MKL)
#  define PREFIX
#  define ROL_fcd const char *
#else
#  define PREFIX
#  define ROL_fcd const char *
#endif

// For non-const char arguments
#if defined(INTEL_CXML)
#  define PREFIX __stdcall
#  define ROL_nonconst_fcd char *, unsigned int
#elif defined(INTEL_MKL)
#  define PREFIX
#  define ROL_nonconst_fcd char *
#else
#  define PREFIX
#  define ROL_nonconst_fcd char *
#endif

/* B) Character handling macros (from Teuchos) */
#ifdef CHAR_MACRO
#undef CHAR_MACRO
#endif
#if defined (INTEL_CXML)
#define CHAR_MACRO(char_var) &char_var, one
#else
#define CHAR_MACRO(char_var) &char_var
#endif

#ifdef CHARPTR_MACRO
#undef CHARPTR_MACRO
#endif
#if defined (INTEL_CXML)
#define CHARPTR_MACRO(charptr_var) charptr_var, one
#else
#define CHARPTR_MACRO(charptr_var) charptr_var
#endif

/* C) LAPACK function name mangling - comprehensive set from Teuchos */
// Factorization routines
#define DGEQRF_F77  F77_BLAS_MANGLE(dgeqrf,DGEQRF)
#define DGEQR2_F77  F77_BLAS_MANGLE(dgeqr2,DGEQR2)
#define DGETRF_F77  F77_BLAS_MANGLE(dgetrf,DGETRF)
#define DGETRS_F77  F77_BLAS_MANGLE(dgetrs,DGETRS)
#define DGETRI_F77  F77_BLAS_MANGLE(dgetri,DGETRI)
#define DPOTRF_F77  F77_BLAS_MANGLE(dpotrf,DPOTRF)
#define DPOTRS_F77  F77_BLAS_MANGLE(dpotrs,DPOTRS)
#define DPOTRI_F77  F77_BLAS_MANGLE(dpotri,DPOTRI)
#define DGTTRF_F77  F77_BLAS_MANGLE(dgttrf,DGTTRF)
#define DGTTRS_F77  F77_BLAS_MANGLE(dgttrs,DGTTRS)
#define DPTTRF_F77  F77_BLAS_MANGLE(dpttrf,DPTTRF)
#define DPTTRS_F77  F77_BLAS_MANGLE(dpttrs,DPTTRS)

// Linear solve routines
#define DGESV_F77   F77_BLAS_MANGLE(dgesv,DGESV)
#define DGESVX_F77  F77_BLAS_MANGLE(dgesvx,DGESVX)
#define DPOSV_F77   F77_BLAS_MANGLE(dposv,DPOSV)
#define DPOSVX_F77  F77_BLAS_MANGLE(dposvx,DPOSVX)

// Least squares routines
#define DGELS_F77   F77_BLAS_MANGLE(dgels,DGELS)
#define DGELSS_F77  F77_BLAS_MANGLE(dgelss,DGELSS)
#define DGGLSE_F77  F77_BLAS_MANGLE(dgglse,DGGLSE)

// Eigenvalue routines  
#define DGEEV_F77   F77_BLAS_MANGLE(dgeev,DGEEV)
#define DGEEVX_F77  F77_BLAS_MANGLE(dgeevx,DGEEVX)
#define DGGEV_F77   F77_BLAS_MANGLE(dggev,DGGEV)
#define DGGES_F77   F77_BLAS_MANGLE(dgges,DGGES)
#define DGGEVX_F77  F77_BLAS_MANGLE(dggevx,DGGEVX)

// SVD routines
#define DGESVD_F77  F77_BLAS_MANGLE(dgesvd,DGESVD)

// Triangular routines
#define DTRTRS_F77  F77_BLAS_MANGLE(dtrtrs,DTRTRS)
#define DTRTRI_F77  F77_BLAS_MANGLE(dtrtri,DTRTRI)
#define DLATRS_F77  F77_BLAS_MANGLE(dlatrs,DLATRS)

// Orthogonal routines
#define DORGQR_F77  F77_BLAS_MANGLE(dorgqr,DORGQR)
#define DORMQR_F77  F77_BLAS_MANGLE(dormqr,DORMQR)
#define DORM2R_F77  F77_BLAS_MANGLE(dorm2r,DORM2R)
#define DORGHR_F77  F77_BLAS_MANGLE(dorghr,DORGHR)
#define DORMHR_F77  F77_BLAS_MANGLE(dormhr,DORMHR)

// Hessenberg routines
#define DGEHRD_F77  F77_BLAS_MANGLE(dgehrd,DGEHRD)
#define DHSEQR_F77  F77_BLAS_MANGLE(dhseqr,DHSEQR)

// Condition number estimation
#define DGECON_F77  F77_BLAS_MANGLE(dgecon,DGECON)
#define DPOCON_F77  F77_BLAS_MANGLE(dpocon,DPOCON)

// Equilibration routines
#define DGEEQU_F77  F77_BLAS_MANGLE(dgeequ,DGEEQU)
#define DPOEQU_F77  F77_BLAS_MANGLE(dpoequ,DPOEQU)

// Refinement routines  
#define DGERFS_F77  F77_BLAS_MANGLE(dgerfs,DGERFS)
#define DPORFS_F77  F77_BLAS_MANGLE(dporfs,DPORFS)

// Scaling routines
#define DLASCL_F77  F77_BLAS_MANGLE(dlascl,DLASCL)
#define DLASWP_F77  F77_BLAS_MANGLE(dlaswp,DLASWP)

// Machine parameters
#define DLAMCH_F77  F77_BLAS_MANGLE(dlamch,DLAMCH)

namespace ROL {

extern "C" {
    // Factorization routines
    void PREFIX DGEQRF_F77(const int* m, const int* n, double* a, const int* lda, double* tau, double* work, const int* lwork, int* info);
    void PREFIX DGEQR2_F77(const int* m, const int* n, double* a, const int* lda, double* tau, double* work, int* info);
    void PREFIX DGETRF_F77(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
    void PREFIX DGETRS_F77(ROL_fcd, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* x, const int* ldx, int* info);
    void PREFIX DGETRI_F77(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info);
    void PREFIX DPOTRF_F77(ROL_fcd, const int* n, double* a, const int* lda, int* info);
    void PREFIX DPOTRS_F77(ROL_fcd, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info);
    void PREFIX DPOTRI_F77(ROL_fcd, const int* n, double* a, const int* lda, int* info);
    void PREFIX DGTTRF_F77(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info);
    void PREFIX DGTTRS_F77(ROL_fcd, const int* n, const int* nrhs, const double* dl, const double* d, const double* du, const double* du2, const int* ipiv, double* x, const int* ldx, int* info);
    void PREFIX DPTTRF_F77(const int* n, double* d, double* e, int* info);
    void PREFIX DPTTRS_F77(const int* n, const int* nrhs, const double* d, const double* e, double* x, const int* ldx, int* info);

    // Linear solve routines
    void PREFIX DGESV_F77(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info);
    void PREFIX DPOSV_F77(ROL_fcd, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, int* info);

    // Least squares routines
    void PREFIX DGELS_F77(ROL_fcd, const int* m, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, double* work, const int* lwork, int* info);
    void PREFIX DGELSS_F77(const int* m, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, double* s, const double* rcond, int* rank, double* work, const int* lwork, int* info);

    // Eigenvalue routines
    void PREFIX DGEEV_F77(ROL_fcd, ROL_fcd, const int* n, double* a, const int* lda, double* wr, double* wi, double* vl, const int* ldvl, double* vr, const int* ldvr, double* work, const int* lwork, int* info);

    // SVD routines
    void PREFIX DGESVD_F77(ROL_fcd, ROL_fcd, const int* m, const int* n, double* a, const int* lda, double* s, double* u, const int* ldu, double* v, const int* ldv, double* work, const int* lwork, int* info);

    // Triangular routines
    void PREFIX DTRTRS_F77(ROL_fcd, ROL_fcd, ROL_fcd, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info);
    void PREFIX DLATRS_F77(ROL_fcd, ROL_fcd, ROL_fcd, ROL_fcd, const int* n, double* a, const int* lda, double* x, double* scale, double* cnorm, int* info);

    // Orthogonal routines
    void PREFIX DORGQR_F77(const int* m, const int* n, const int* k, double* a, const int* lda, const double* tau, double* work, const int* lwork, int* info);

    // Machine parameters
    double PREFIX DLAMCH_F77(ROL_fcd);
}

// LAPACK class template
template<typename OrdinalType, typename ScalarType>
class LAPACK {
public:
    // Eigenvalue routines  
    void GEEV(const char& JOBVL, const char& JOBVR, const OrdinalType& n, ScalarType* A, const OrdinalType& lda, ScalarType* WR, ScalarType* WI, ScalarType* VL, const OrdinalType& ldvl, ScalarType* VR, const OrdinalType& ldvr, ScalarType* WORK, const OrdinalType& lwork, ScalarType* RWORK, OrdinalType* info) const;
    void GEEV(const char& JOBVL, const char& JOBVR, const OrdinalType& n, ScalarType* A, const OrdinalType& lda, ScalarType* WR, ScalarType* WI, ScalarType* VL, const OrdinalType& ldvl, ScalarType* VR, const OrdinalType& ldvr, ScalarType* WORK, const OrdinalType& lwork, OrdinalType* info) const;

    // Least squares routines
    void GELS(const char& TRANS, const OrdinalType& m, const OrdinalType& n, const OrdinalType& nrhs, ScalarType* A, const OrdinalType& lda, ScalarType* B, const OrdinalType& ldb, ScalarType* WORK, const OrdinalType& lwork, OrdinalType* info) const;
    void GELSS(const OrdinalType& m, const OrdinalType& n, const OrdinalType& nrhs, ScalarType* A, const OrdinalType& lda, ScalarType* B, const OrdinalType& ldb, ScalarType* S, const ScalarType& rcond, OrdinalType* rank, ScalarType* WORK, const OrdinalType& lwork, OrdinalType* info) const;

    // SVD routines
    void GESVD(const char& JOBU, const char& JOBVT, const OrdinalType& m, const OrdinalType& n, ScalarType* A, const OrdinalType& lda, ScalarType* S, ScalarType* U, const OrdinalType& ldu, ScalarType* V, const OrdinalType& ldv, ScalarType* WORK, const OrdinalType& lwork, ScalarType* RWORK, OrdinalType* info) const;

    // Triangular routines
    void LATRS(const char& UPLO, const char& TRANS, const char& DIAG, const char& NORMIN, const OrdinalType& N, ScalarType* A, const OrdinalType& LDA, ScalarType* X, ScalarType* SCALE, ScalarType* CNORM, OrdinalType* INFO) const;
    void TRTRS(const char& UPLO, const char& TRANS, const char& DIAG, const OrdinalType& n, const OrdinalType& nrhs, const ScalarType* A, const OrdinalType& lda, ScalarType* B, const OrdinalType& ldb, OrdinalType* info) const;

    // Factorization routines
    void GTTRF(const OrdinalType& n, ScalarType* dl, ScalarType* d, ScalarType* du, ScalarType* du2, OrdinalType* IPIV, OrdinalType* info) const;
    void GTTRS(const char& TRANS, const OrdinalType& n, const OrdinalType& nrhs, const ScalarType* dl, const ScalarType* d, const ScalarType* du, const ScalarType* du2, const OrdinalType* IPIV, ScalarType* X, const OrdinalType& LDX, OrdinalType* info) const;
    void GETRF(const OrdinalType& m, const OrdinalType& n, ScalarType* A, const OrdinalType& lda, OrdinalType* IPIV, OrdinalType* info) const;
    void GETRI(const OrdinalType& n, ScalarType* A, const OrdinalType& lda, const OrdinalType* IPIV, ScalarType* WORK, const OrdinalType& lwork, OrdinalType* info) const;
    void GETRS(const char& TRANS, const OrdinalType& n, const OrdinalType& nrhs, const ScalarType* A, const OrdinalType& lda, const OrdinalType* IPIV, ScalarType* X, const OrdinalType& LDX, OrdinalType* info) const;
    void GEQRF(const OrdinalType& m, const OrdinalType& n, ScalarType* A, const OrdinalType& lda, ScalarType* TAU, ScalarType* WORK, const OrdinalType& lwork, OrdinalType* info) const;
    void ORGQR(const OrdinalType& m, const OrdinalType& n, const OrdinalType& k, ScalarType* A, const OrdinalType& lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType& lwork, OrdinalType* info) const;

    // Tridiagonal routines
    void PTTRF(const OrdinalType& n, ScalarType* d, ScalarType* e, OrdinalType* info) const;
    void PTTRS(const OrdinalType& n, const OrdinalType& nrhs, const ScalarType* d, const ScalarType* e, ScalarType* X, const OrdinalType& LDX, OrdinalType* info) const;
};

// Template specialization for int, double
template<>
inline void LAPACK<int, double>::GEEV(const char& JOBVL, const char& JOBVR, const int& n, double* A, const int& lda, double* WR, double* WI, double* VL, const int& ldvl, double* VR, const int& ldvr, double* WORK, const int& lwork, double* /* RWORK */, int* info) const {
    DGEEV_F77(CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), &n, A, &lda, WR, WI, VL, &ldvl, VR, &ldvr, WORK, &lwork, info);
}

// Overload for real matrices without RWORK parameter (to match Teuchos interface exactly)
template<>
inline void LAPACK<int, double>::GEEV(const char& JOBVL, const char& JOBVR, const int& n, double* A, const int& lda, double* WR, double* WI, double* VL, const int& ldvl, double* VR, const int& ldvr, double* WORK, const int& lwork, int* info) const {
    DGEEV_F77(CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), &n, A, &lda, WR, WI, VL, &ldvl, VR, &ldvr, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::GELS(const char& TRANS, const int& m, const int& n, const int& nrhs, double* A, const int& lda, double* B, const int& ldb, double* WORK, const int& lwork, int* info) const {
    DGELS_F77(CHAR_MACRO(TRANS), &m, &n, &nrhs, A, &lda, B, &ldb, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::GELSS(const int& m, const int& n, const int& nrhs, double* A, const int& lda, double* B, const int& ldb, double* S, const double& rcond, int* rank, double* WORK, const int& lwork, int* info) const {
    DGELSS_F77(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rcond, rank, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::GESVD(const char& JOBU, const char& JOBVT, const int& m, const int& n, double* A, const int& lda, double* S, double* U, const int& ldu, double* V, const int& ldv, double* WORK, const int& lwork, double* /* RWORK */, int* info) const {
    DGESVD_F77(CHAR_MACRO(JOBU), CHAR_MACRO(JOBVT), &m, &n, A, &lda, S, U, &ldu, V, &ldv, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::LATRS(const char& UPLO, const char& TRANS, const char& DIAG, const char& NORMIN, const int& N, double* A, const int& LDA, double* X, double* SCALE, double* CNORM, int* INFO) const {
    DLATRS_F77(CHAR_MACRO(UPLO), CHAR_MACRO(TRANS), CHAR_MACRO(DIAG), CHAR_MACRO(NORMIN), &N, A, &LDA, X, SCALE, CNORM, INFO);
}

template<>
inline void LAPACK<int, double>::TRTRS(const char& UPLO, const char& TRANS, const char& DIAG, const int& n, const int& nrhs, const double* A, const int& lda, double* B, const int& ldb, int* info) const {
    DTRTRS_F77(CHAR_MACRO(UPLO), CHAR_MACRO(TRANS), CHAR_MACRO(DIAG), &n, &nrhs, A, &lda, B, &ldb, info);
}

template<>
inline void LAPACK<int, double>::GTTRF(const int& n, double* dl, double* d, double* du, double* du2, int* IPIV, int* info) const {
    DGTTRF_F77(&n, dl, d, du, du2, IPIV, info);
}

template<>
inline void LAPACK<int, double>::GTTRS(const char& TRANS, const int& n, const int& nrhs, const double* dl, const double* d, const double* du, const double* du2, const int* IPIV, double* X, const int& LDX, int* info) const {
    DGTTRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, dl, d, du, du2, IPIV, X, &LDX, info);
}

template<>
inline void LAPACK<int, double>::GETRF(const int& m, const int& n, double* A, const int& lda, int* IPIV, int* info) const {
    DGETRF_F77(&m, &n, A, &lda, IPIV, info);
}

template<>
inline void LAPACK<int, double>::GETRI(const int& n, double* A, const int& lda, const int* IPIV, double* WORK, const int& lwork, int* info) const {
    DGETRI_F77(&n, A, &lda, IPIV, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::GETRS(const char& TRANS, const int& n, const int& nrhs, const double* A, const int& lda, const int* IPIV, double* X, const int& LDX, int* info) const {
    DGETRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, IPIV, X, &LDX, info);
}

template<>
inline void LAPACK<int, double>::GEQRF(const int& m, const int& n, double* A, const int& lda, double* TAU, double* WORK, const int& lwork, int* info) const {
    DGEQRF_F77(&m, &n, A, &lda, TAU, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::ORGQR(const int& m, const int& n, const int& k, double* A, const int& lda, const double* TAU, double* WORK, const int& lwork, int* info) const {
    DORGQR_F77(&m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
}

template<>
inline void LAPACK<int, double>::PTTRF(const int& n, double* d, double* e, int* info) const {
    DPTTRF_F77(&n, d, e, info);
}

template<>
inline void LAPACK<int, double>::PTTRS(const int& n, const int& nrhs, const double* d, const double* e, double* X, const int& LDX, int* info) const {
    DPTTRS_F77(&n, &nrhs, d, e, X, &LDX, info);
}

} // namespace ROL

#endif /* ROL_LAPACK_COMPLETE_H */
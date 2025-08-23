// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef ROL_LAPACK_H
#define ROL_LAPACK_H

/** \class ROL::LAPACK
  \brief Provides interface to LAPACK using LAPACKE (C interface)
  */

#include <lapacke/lapacke.h>
#include <vector>
#include <algorithm>

namespace ROL { 

  template<typename Index, typename Real>
  struct LAPACK { 
    /// \brief Computes for an \c n by \c n real nonsymmetric matrix \c A, the eigenvalues and, optionally, the left and/or right eigenvectors.
    ///
    /// Real and imaginary parts of the eigenvalues are returned in
    /// separate arrays, WR for real and WI for complex.  The RWORK
    /// array is only referenced if ScalarType is complex.
    void GEEV(const char& JOBVL, const char& JOBVR, const Index& n, Real* A, const Index& lda, Real* WR, Real* WI, Real* VL, const Index& ldvl, Real* VR, const Index& ldvr, Real* WORK, const Index& lwork, Real* RWORK, Index* info) const;

    //\brief Solves an over/underdetermined real \c m by \c n linear system \c A using QR or LQ factorization of A.
    void GELS(const char& TRANS, const Index& m, const Index& n, const Index& nrhs, Real* A, const Index& lda, Real* B, const Index& ldb, Real* WORK, const Index& lwork, Index* info) const;

    void GELSS(const Index& m, const Index& n, const Index& nrhs, Real* A, const Index& lda, Real* B, const Index& ldb, Real* S, const Real& rcond, Index* rank, Real* WORK, const Index& lwork, Index* info) const;

    /// \brief Robustly solve a possibly singular triangular linear system.
    ///
    /// \note This routine is slower than the BLAS' TRSM, but can
    ///   detect possible singularity of A.
    void LATRS (const char& UPLO, const char& TRANS, const char& DIAG, const char& NORMIN,
        const Index& N, Real* A, const Index& LDA, Real* X, Real* SCALE, Real* CNORM, Index* INFO) const;

    //! Computes an LU factorization of a \c n by \c n tridiagonal matrix \c A using partial pivoting with row interchanges.
    void GTTRF(const Index& n, Real* dl, Real* d, Real* du, Real* du2, Index* IPIV, Index* info) const;

    //! Solves a system of linear equations \c A*X=B or \c A'*X=B or \c A^H*X=B with a tridiagonal matrix \c A using the LU factorization computed by GTTRF.
    void GTTRS(const char& TRANS, const Index& n, const Index& nrhs, const Real* dl,
        const Real* d, const Real* du, const Real* du2, const Index* IPIV, Real* B,
        const Index& ldb, Index* info) const;

    //! Computes the eigenvalues and, optionally, eigenvectors of a symmetric tridiagonal \c n by \c n matrix \c A using implicit QL/QR.  The eigenvectors can only be computed if \c A was reduced to tridiagonal form by SYTRD.
    void STEQR(const char& COMPZ, const Index& n, Real* D, Real* E, Real* Z, const Index& ldz, Real* WORK, Index* info) const;

    //! Solves a triangular linear system of the form \c A*X=B or \c A**T*X=B, where \c A is a triangular matrix.
    void TRTRS(const char& UPLO, const char& TRANS, const char& DIAG, const Index& n, const Index& nrhs, const Real* A, const Index& lda, Real* B, const Index& ldb, Index* info) const;

    //! Computes an LU factorization of a general \c m by \c n matrix \c A using partial pivoting with row interchanges.
    void GETRF(const Index& m, const Index& n, Real* A, const Index& lda, Index* IPIV, Index* info) const;

    //! Computes the inverse of a matrix \c A using the LU factorization computed by GETRF.
    void GETRI(const Index& n, Real* A, const Index& lda, const Index* IPIV, Real* WORK, const Index& lwork, Index* info) const;

    //! Solves a system of linear equations \c A*X=B or \c A'*X=B with a general \c n by \c n matrix \c A using the LU factorization computed by GETRF.
    void GETRS(const char& TRANS, const Index& n, const Index& nrhs, const Real* A, const Index& lda, const Index* IPIV, Real* B, const Index& ldb, Index* info) const;

    //! Computes a QR factorization of a general \c m by \c n matrix \c A.
    void GEQRF (const Index& m, const Index& n, Real* A, const Index& lda, Real* TAU, Real* WORK, const Index& lwork, Index* info) const;

    /// \brief Compute explicit Q factor from QR factorization (GEQRF) (real case).
    ///
    /// Generate the \c m by \c n matrix Q with orthonormal columns
    /// corresponding to the first \c n columns of a product of \c k
    /// elementary reflectors of order \c m, as returned by \c GEQRF.
    ///
    /// \note This method is not defined when Real is complex.
    /// Call \c UNGQR in that case.  ("OR" stands for "orthogonal";
    /// "UN" stands for "unitary.")
    void ORGQR(const Index& m, const Index& n, const Index& k, Real* A, const Index& lda, const Real* TAU, Real* WORK, const Index& lwork, Index* info) const;

    //! Computes the \c L*D*L' factorization of a Hermitian/symmetric positive definite tridiagonal matrix \c A.
    void PTTRF(const Index& n, Real* d, Real* e, Index* info) const;

    //! Solves a tridiagonal system \c A*X=B using the \L*D*L' factorization of \c A computed by PTTRF.
    void PTTRS(const Index& n, const Index& nrhs, const Real* d, const Real* e, Real* B, const Index& ldb, Index* info) const;

    //! Computes the singular value decomposition (SVD) of a general \c m by \c n matrix \c A.
    void GESVD(const char& JOBU, const char& JOBVT, const Index& m, const Index& n, Real* A, const Index& lda, Real* S, Real* U, const Index& ldu, Real* VT, const Index& ldvt, Real* WORK, const Index& lwork, Index* info) const;
  };

  template<>
  struct LAPACK<int, double> { 

    void LATRS (const char& UPLO, const char& TRANS, const char& DIAG, const char& NORMIN,
                const int& N, double* A, const int& LDA, double* X, double* SCALE, double* CNORM,
                int* INFO) const {
      // LAPACKE doesn't have dlatrs, so we use dtrtrs as a fallback
      // This is less robust but should work for non-singular cases
      *SCALE = 1.0;  // Assume no scaling needed
      *INFO = LAPACKE_dtrtrs(LAPACK_COL_MAJOR, UPLO, TRANS, DIAG, N, 1, A, LDA, X, N);
      // Set CNORM to 1.0 as a simple approximation
      if (CNORM) *CNORM = 1.0;
    }

    void GTTRS(const char& TRANS, const int& n, const int& nrhs, const double* dl, const double* d,
               const double* du, const double* du2, const int* IPIV, double* B, const int& ldb, int* info) const {
      *info = LAPACKE_dgttrs(LAPACK_COL_MAJOR, TRANS, n, nrhs, dl, d, du, du2, IPIV, B, ldb);
    }

    void GTTRF(const int& n, double* dl, double* d, double* du, double* du2, int* IPIV, int* info) const {
      *info = LAPACKE_dgttrf(n, dl, d, du, du2, IPIV);
    }

    void STEQR(const char& COMPZ, const int& n, double* D, double* E, double* Z, const int& ldz, double* WORK, int* info) const {
      *info = LAPACKE_dsteqr(LAPACK_COL_MAJOR, COMPZ, n, D, E, Z, ldz);
    }

    void TRTRS(const char& UPLO, const char& TRANS, const char& DIAG, const int& n, const int& nrhs, const double* A, const int& lda, double* B, const int& ldb, int* info) const { 
      *info = LAPACKE_dtrtrs(LAPACK_COL_MAJOR, UPLO, TRANS, DIAG, n, nrhs, A, lda, B, ldb);
    }

    void GETRF(const int& m, const int& n, double* A, const int& lda, int* IPIV, int* info) const {
      *info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, A, lda, IPIV);
    }
 
    void GETRI(const int& n, double* A, const int& lda, const int* IPIV, double* WORK, const int& lwork, int* info) const {
      *info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, A, lda, IPIV);
    }

    void GETRS(const char& TRANS, const int& n, const int& nrhs, const double* A, const int& lda, const int* IPIV, double* B, const int& ldb, int* info) const {
      *info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, TRANS, n, nrhs, A, lda, IPIV, B, ldb);
    }

    void GEQRF( const int& m, const int& n, double* A, const int& lda, double* TAU, double* WORK, const int& lwork, int* info) const {
      *info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, TAU);
    }

    void ORGQR(const int& m, const int& n, const int& k, double* A, const int& lda, const double* TAU, double* WORK, const int& lwork, int* info) const {
      *info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, k, A, lda, TAU);
    }

    void PTTRS(const int& n, const int& nrhs, const double* d, const double* e, double* B, const int& ldb, int* info) const { 
      *info = LAPACKE_dpttrs(LAPACK_COL_MAJOR, n, nrhs, d, e, B, ldb);
    }

    void GEEV(const char& JOBVL, const char& JOBVR, const int& n, double* A, const int& lda, double* WR, double* WI, double* VL, const int& ldvl, double* VR, const int& ldvr, double* WORK, const int& lwork, int* info) const {
      *info = LAPACKE_dgeev(LAPACK_COL_MAJOR, JOBVL, JOBVR, n, A, lda, WR, WI, VL, ldvl, VR, ldvr);
    }

    void GELS(const char& TRANS, const int& m, const int& n, const int& nrhs, double* A, const int& lda, double* B, const int& ldb, double* WORK, const int& lwork, int* info) const { 
      *info = LAPACKE_dgels(LAPACK_COL_MAJOR, TRANS, m, n, nrhs, A, lda, B, ldb);
    }

    void GELSS(const int& m, const int& n, const int& nrhs, double* A, const int& lda, double* B, const int& ldb, double* S, const double& rcond, int* rank, double* WORK, const int& lwork, int* info) const { 
      *info = LAPACKE_dgelss(LAPACK_COL_MAJOR, m, n, nrhs, A, lda, B, ldb, S, rcond, rank);
    }

    void PTTRF(const int& n, double* d, double* e, int* info) const {
      *info = LAPACKE_dpttrf(n, d, e);
    }

    void GESVD(const char& JOBU, const char& JOBVT, const int& m, const int& n, double* A, const int& lda, double* S, double* U, const int& ldu, double* VT, const int& ldvt, double* WORK, const int& lwork, int* info) const {
      std::vector<double> superb(std::min(m,n)-1);
      *info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, JOBU, JOBVT, m, n, A, lda, S, U, ldu, VT, ldvt, superb.data());
    }
  };

}

#endif
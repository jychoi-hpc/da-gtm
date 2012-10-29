/*
 *  vec.c
 *  DAGTM
 *
 *  Created by Jong Youl Choi on 1/26/11.
 *  Copyright 2011 Indiana University. All rights reserved.
 *
 */

#include "vec.h"
#include "comm.h"
#include <math.h>
#include <gsl/gsl_errno.h>

int dagtm_vector_scale_f_vec(size_t Vsize, size_t Vstride, double V[Vsize],
                             const double f)
{
    if (Vstride == 1)
    {
        for (size_t i = 0; i < Vsize; i++)
        {
            V[i] = V[i] * f;
        }
    }
    else
    {
        for (size_t i = 0; i < Vsize * Vstride; i += Vstride)
        {
            V[i] = V[i] * f;
        }
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_scale_f_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                             double M[][Mtda], const double f)
{
    for (size_t i = 0; i < Msize1; i++)
    {
        for (size_t j = 0; j < Msize2; j++)
        {
            M[i][j] = M[i][j] * f;
        }
    }
    return GSL_SUCCESS;
}

/*
 * Assume vector V is continuous
 * Should Msize1 == Vsize
 */
int dagtm_matrix_scale_by_row_vec(size_t Msize1, size_t Msize2,
                                  size_t Mtda, double M[][Mtda],
                                  size_t Vsize, double V[Vsize])
{
    for (size_t i = 0; i < Msize1; i++)
    {
        double f = V[i];
        for (size_t j = 0; j < Msize2; j++)
        {
            M[i][j] = M[i][j] * f;
        }
    }
    return GSL_SUCCESS;
}

/*
 * Assume vector V is continuous
 * Should Msize2 == Vsize
 */
int dagtm_matrix_scale_by_col_vec(size_t Msize1, size_t Msize2,
                                  size_t Mtda, double M[][Mtda],
                                  size_t Vsize, double V[Vsize])
{
    for (size_t i = 0; i < Msize1; i++)
    {
#pragma ivdep
        for (size_t j = 0; j < Msize2; j++)
        {
            M[i][j] = M[i][j] * V[j];
        }
    }
    return GSL_SUCCESS;
}

/*
 * Assume A and B has same dimension
 * a[i,j] = a[i,j] * b[i,j]
 */
int dagtm_matrix_scale_by_mat_vec(size_t Asize1, size_t Asize2,
                                  size_t Atda, double A[][Atda],
                                  size_t Bsize1, size_t Bsize2,
                                  size_t Btda, double B[][Btda])
{
    for (size_t i = 0; i < Asize1; i++)
    {
#pragma ivdep
        for (size_t j = 0; j < Asize2; j++)
        {
            A[i][j] = A[i][j] * B[i][j];
        }
    }
    return GSL_SUCCESS;
}

/*
 * Assume vector V is continuous
 * Should Msize2 == Vsize
 */
int dagtm_matrix_div_by_col_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                                double M[][Mtda], size_t Vsize,
                                double V[Vsize])
{
    for (size_t i = 0; i < Msize1; i++)
    {
#pragma ivdep
        for (size_t j = 0; j < Msize2; j++)
        {
            M[i][j] = M[i][j] / V[j];
        }
    }
    return GSL_SUCCESS;
}

int dagtm_matrix_apply_exp_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                               double M[][Mtda])
{
    for (size_t i = 0; i < Msize1; i++)
    {
        for (size_t j = 0; j < Msize2; j++)
        {
            M[i][j] = exp(M[i][j]);
        }
    }
    return GSL_SUCCESS;
}

int dagtm_matrix_trans_vec(size_t Asize1, size_t Asize2, size_t Atda, double A[][Atda], 
			   size_t Bsize1, size_t Bsize2, size_t Btda, double B[][Btda])
{
  for (size_t j = 0; j < Asize2; j++)
    {
#pragma ivdep
      for (size_t i = 0; i < Asize1; i++)
	{
	  B[j][i] = A[i][j];
	}
    }
    return GSL_SUCCESS;
}

int dagtm_dist_vec(size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
                   size_t Dsize2, size_t Dtda, double D[][Dtda])
{
    for (size_t i = 0; i < Asize1; i++)
    {
        for (size_t j = 0; j < Bsize1; j++)
        {
            D[i][j] = 0;
#pragma vector aligned
#pragma ivdep
            for (size_t k = 0; k < Asize2; k++)
            {
                double d = A[i][k] - B[j][k];
                D[i][j] += d * d;
            }
        }
    }
    return GSL_SUCCESS;
}

int dagtm_gsl_blas_gemm_vec(const double c, size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
                   size_t Dsize2, size_t Dtda, double D[][Dtda])
{
    for (size_t i = 0; i < Asize1; i++)
    {
        for (size_t j = 0; j < Bsize1; j++)
        {
	  double d = 0;
            for (size_t k = 0; k < Asize2; k++)
            {
                d += c*A[i][k]*B[j][k];
            }
	    D[i][j] = d;
        }
    }
    return GSL_SUCCESS;
}

#define BSI 512
#define BSJ 512
#define BSK 512
#define MIN(a,b) (((a)<(b))?(a):(b))

int dagtm_gsl_blas_gemm_vec_cb(const double c, size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
                   size_t Dsize2, size_t Dtda, double D[][Dtda])
{
  memset(&D[0][0], 0, Dsize1*Dtda*sizeof(double));
      for (size_t ii = 0; ii < Asize1; ii += BSJ)
	{
  for (size_t jj = 0; jj < Bsize1; jj += BSI)
    {
	  for (size_t kk = 0; kk < Asize2; kk += BSK)
	    {
	      size_t kmax = MIN(kk + BSK, Asize2);
	      for (size_t i = ii; i < MIN(ii + BSI, Asize1); i++)
		{
		  for (size_t j = jj; j < MIN(jj + BSJ, Bsize1); j++)
		    {
		      double d1;
		      //for (size_t k = kk; k < MIN(kk + BSK, Asize2); k++)
#pragma vector aligned
#pragma ivdep
		      for (size_t k = kk; k < kmax; k++)
			{
			  d1 += c*A[i][k]*B[j][k];
			}
		      D[i][j] += d1;

		    }
		}
	    }
	}
    }
  
  return GSL_SUCCESS;
}

#include <emmintrin.h> 
int dagtm_gsl_blas_gemm_vec_ss(const double c, size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
                   size_t Dsize2, size_t Dtda, double D[][Dtda])
{
  memset(&D[0][0], 0, Dsize1*Dtda*sizeof(double));
  __m128d _c = _mm_set1_pd(c);
  for (size_t ii = 0; ii < Asize1; ii += BSJ) 
    {
      for (size_t jj = 0; jj < Bsize1; jj += BSI) 
	{
	  for (size_t kk = 0; kk < Asize2; kk += BSK)
	    {
	      size_t kmax = MIN(kk + BSK, Asize2);
	      for (size_t i = ii; i < MIN(ii + BSI, Asize1); i++)
		{
		  for (size_t j = jj; j < MIN(jj + BSJ, Bsize1); j += 2)
		    {
		      double d1;
		      //for (size_t k = kk; k < MIN(kk + BSK, Asize2); k++)
		      //#pragma vector aligned
		      //#pragma ivdep
		      for (size_t k = kk; k < kmax; k += 2)
			{
			  __m128d _a = _mm_load_pd(&(A[i][k]));
			  __m128d _b = _mm_load_pd(&(B[j][k]));
			  __m128d _d = _mm_load_pd(&(D[i][j]));
			  __m128d _t = _mm_mul_pd(_a, _b);
			  _t = _mm_mul_pd(_t, _c);
			  _d = _mm_add_pd(_d, _t);

			  _mm_store_pd(&(D[i][j]), _d);

			  //d1 += c*A[i][k]*B[j][k];
			}
		      //D[i][j] += d1;
		    }
		}
	    }
	}
    }
  
  return GSL_SUCCESS;
}

int dagtm_matrix_sum_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                         double M[][Mtda], double sum[1])
{
    for (size_t i = 0; i < Msize1; i++)
    {
#pragma vector aligned
#pragma ivdep
        for (size_t j = 0; j < Msize2; j++)
        {
            sum[0] += M[i][j];
        }
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_rowsum_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                            double M[][Mtda], size_t Vsize,
                            double V[Vsize])
{
    for (size_t i = 0; i < Msize1; i++)
    {
        V[i] = 0.0;
#pragma vector aligned
#pragma ivdep
        for (size_t j = 0; j < Msize2; j++)
        {
            V[i] += M[i][j];
        }
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_colsum_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                            double M[][Mtda], size_t Vsize,
                            double V[Vsize])
{
    for (size_t j = 0; j < Msize2; j++)
    {
        V[j] = 0.0;
    }

    for (size_t i = 0; i < Msize1; i++)
    {
#pragma vector aligned
#pragma ivdep
        for (size_t j = 0; j < Msize2; j++)
        {
            V[j] += M[i][j];
        }
    }

    return GSL_SUCCESS;
}

/*
 * Cannot make vectorized
 */
int dagtm_trans_copy_vec(size_t Asize1, size_t Asize2, size_t Atda,
                         double A[][Atda], size_t Bsize1, size_t Bsize2,
                         size_t Btda, double B[][Btda])
{
    for (size_t i = 0; i < Asize1; i++)
    {
#pragma vector aligned
#pragma ivdep
        for (size_t j = 0; j < Asize2; j++)
        {
            B[j][i] = A[i][j];
        }
    }
    return GSL_SUCCESS;
}

int dagtm_gsl_matrix_memcpy_vec(size_t Dsize1, size_t Dsize2, size_t Dtda, double D[][Dtda], 
                                size_t Ssize1, size_t Ssize2, size_t Stda, double S[][Stda])
{
    if (Dsize1 != Ssize1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (Dsize2 != Ssize2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (Dtda != Stda)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    memcpy((void *) D, (void *) S, (Dsize1 * Dtda * sizeof(double)));
    return GSL_SUCCESS;
}

int dagtm_gsl_matrix_memcpy_byrow_vec(size_t Dsize1, size_t Dsize2, size_t Dtda, double D[][Dtda], 
                                      size_t Ssize1, size_t Ssize2, size_t Stda, double S[][Stda])
{
    if (Dsize1 != Ssize1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (Dsize2 != Ssize2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    for (size_t i = 0; i < Dsize1; i++)
    {
        memcpy((void *) &(D[i][0]), (void *) &(S[i][0]), Dsize2 * sizeof(double));
    }

    return GSL_SUCCESS;
}

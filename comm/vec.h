/*
 *  vec.h
 *  DAGTM
 *
 *  Created by Jong Youl Choi on 1/26/11.
 *  Copyright 2011 Indiana University. All rights reserved.
 *
 */

#ifndef __DAGTM_VEC_H__
#define __DAGTM_VEC_H__
#include <string.h>

int dagtm_vector_scale_f_vec(size_t Vsize, size_t Vstride, double V[Vsize],
                             const double f);

int dagtm_matrix_scale_f_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                             double M[][Mtda], const double f);

/*
 * Assume vector V is continuous
 * Should Msize1 == Vsize
 */
int dagtm_matrix_scale_by_row_vec(size_t Msize1, size_t Msize2,
                                  size_t Mtda, double M[][Mtda],
                                  size_t Vsize, double V[Vsize]);

/*
 * Assume vector V is continuous
 * Should Msize2 == Vsize
 */
int dagtm_matrix_scale_by_col_vec(size_t Msize1, size_t Msize2,
                                  size_t Mtda, double M[][Mtda],
                                  size_t Vsize, double V[Vsize]);

/*
 * Assume A and B has same dimension
 * a[i,j] = a[i,j] * b[i,j]
 */
int dagtm_matrix_scale_by_mat_vec(size_t Asize1, size_t Asize2,
                                  size_t Atda, double A[][Atda],
                                  size_t Bsize1, size_t Bsize2,
                                  size_t Btda, double B[][Btda]);

/*
 * Assume vector V is continuous
 * Should Msize2 == Vsize
 */
int dagtm_matrix_div_by_col_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                                double M[][Mtda], size_t Vsize,
                                double V[Vsize]);

int dagtm_matrix_apply_exp_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                               double M[][Mtda]);

int dagtm_matrix_trans_vec(size_t Asize1, size_t Asize2, size_t Atda, double A[][Atda], 
			   size_t Bsize1, size_t Bsize2, size_t Btda, double B[][Btda]);

int dagtm_dist_vec(size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
                   size_t Dsize2, size_t Dtda, double D[][Dtda]);

int dagtm_gsl_blas_gemm_vec(const double c, size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
			    size_t Dsize2, size_t Dtda, double D[][Dtda]);

int dagtm_gsl_blas_gemm_vec_cb(const double c, size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
			       size_t Dsize2, size_t Dtda, double D[][Dtda]);

int dagtm_gsl_blas_gemm_vec_ss(const double c, size_t Asize1, size_t Asize2, size_t Atda,
                   double A[][Atda], size_t Bsize1, size_t Bsize2,
                   size_t Btda, double B[][Btda], size_t Dsize1,
			       size_t Dsize2, size_t Dtda, double D[][Dtda]);

int dagtm_matrix_sum_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                         double M[][Mtda], double sum[1]);

/*
 * Assume vector V is continuous
 */
int dagtm_matrix_rowsum_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                            double M[][Mtda], size_t Vsize,
                            double V[Vsize]);

/*
 * Assume vector V is continuous
 */
int dagtm_matrix_colsum_vec(size_t Msize1, size_t Msize2, size_t Mtda,
                            double M[][Mtda], size_t Vsize,
                            double V[Vsize]);

int dagtm_trans_copy_vec(size_t Asize1, size_t Asize2, size_t Atda,
                         double A[][Atda], size_t Bsize1, size_t Bsize2,
                         size_t Btda, double B[][Btda]);

int dagtm_gsl_matrix_memcpy_vec(size_t Dsize1, size_t Dsize2, size_t Dtda, double D[][Dtda], 
                                size_t Ssize1, size_t Ssize2, size_t Stda, double S[][Stda]);

int dagtm_gsl_matrix_memcpy_byrow_vec(size_t Dsize1, size_t Dsize2, size_t Dtda, double D[][Dtda], 
                                      size_t Ssize1, size_t Ssize2, size_t Stda, double S[][Stda]);

#endif                          //__DAGTM_VEC_H__

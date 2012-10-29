#include "comm.h"
#include "vec.h"
#include <stdlib.h>

int dagtm_myid;
int dagtm_loglevel;
FILE *dagtm_flog;
unsigned long dagtm_seed;

// See http://heather.cs.ucdavis.edu/~matloff/pardebug.html
//int DebugWait = 1;
//#define DEBUG_STOP_HERE \
//DEBUG(DAGTM_INFO_MSG, "Waiting for debugging ... "); \
//while (DebugWait) { sleep(1000); }

#ifdef COMM_IPP_ON
#include "ipp.h"

void dagtm_init()
{
    ippInit();
}
#endif

double rint(double x)
{
    return floor(x + 0.5);
}

void dagtm_setfilelog(int id)
{
    char buff[MAXLEN];
    sprintf(buff, "dagtm.P%03d.log", id);
    printf("buff = %s", buff);
    fflush(stdout);
    dagtm_flog = fopen(buff, "w");
}

void dagtm_setloglevel(int l)
{
    dagtm_loglevel = l;
}

void dagtm_setmyid(int id)
{
    dagtm_myid = id;
}

int dagtm_getloglevel()
{
    return dagtm_loglevel;
}

int dagtm_blas_dgemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB,
                     double alpha, const gsl_matrix * A,
                     const gsl_matrix * B, double beta, gsl_matrix * C)
{
    size_t i, j, d;

    if (A->size2 != B->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (A->size1 != C->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (B->size2 != C->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    for (i = 0; i < A->size1; i++)
    {
        for (j = 0; j < B->size2; j++)
        {
            double sum = 0;
            for (d = 0; d < A->size2; d++)
            {
                sum += gsl_matrix_get(A, i, d) * gsl_matrix_get(B, d, j);
            }
            gsl_matrix_set(C, i, j, sum);
        }
    }

    return GSL_SUCCESS;
}

double dagtm_matrix_sum(const gsl_matrix * M)
{
    double s = 0;
    dagtm_matrix_sum_vec(M->size1, M->size2, M->tda,
                         (double (*)[]) M->data, &s);

    /*
       size_t i, j;
       double s = 0;

       for (i = 0; i < M->size1; i++)
       {
       for (j = 0; j < M->size2; j++)
       {
       s += gsl_matrix_get(M, i, j);
       }
       }
     */

    return s;
}

void dagtm_matrix_summary(gsl_matrix * M)
{
    size_t i, j;

    if (M == NULL)
    {
        printf("len = NULL\n");
        return;
    }

    printf("dim = (%ud, %ud)\n", (unsigned int) M->size1,
           (unsigned int) M->size2);

    for (i = 0; i < MIN(M->size1, HEAD_ROW_MAX); i++)
    {
        printf("[%ud] ", (unsigned int) i);
        for (j = 0; j < M->size2; j++)
        {
            printf("%g ", gsl_matrix_get(M, i, j));
        }
        printf("\n");
    }
}

void dagtm_vector_summary(gsl_vector * v)
{
    size_t i;

    if (v == NULL)
    {
        printf("len = NULL\n");
        return;
    }

    printf("len = %d\n", (unsigned int) v->size);

    printf("[-] ");
    for (i = 0; i < MIN(v->size, HEAD_COL_MAX); i++)
    {
        printf("%g ", gsl_vector_get(v, i));
    }
    printf("\n");
}

int dagtm_matrix_apply(gsl_matrix * M, double (*func) (double))
{
    size_t i, j;
    double d;
    for (i = 0; i < M->size1; i++)
    {
        for (j = 0; j < M->size2; j++)
        {
            d = gsl_matrix_get(M, i, j);
            gsl_matrix_set(M, i, j, (*func) (d));
        }
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_apply2(gsl_matrix * M, double (*func) (double, double),
                        const double arg2)
{
    size_t i, j;
    double d;
    for (i = 0; i < M->size1; i++)
    {
        for (j = 0; j < M->size2; j++)
        {
            d = gsl_matrix_get(M, i, j);
            gsl_matrix_set(M, i, j, (*func) (d, arg2));
        }
    }

    return GSL_SUCCESS;
}

int dagtm_vector_apply(gsl_vector * v, double (*func) (double))
{
    size_t i;
    double d;
    for (i = 0; i < v->size; i++)
    {
        d = gsl_vector_get(v, i);
        gsl_vector_set(v, i, (*func) (d));
    }

    return GSL_SUCCESS;
}

double dagtm_vector_sum(const gsl_vector * v)
{
    size_t i;
    double s = 0;
    for (i = 0; i < v->size; i++)
    {
        s += gsl_vector_get(v, i);
    }

    return s;
}

int dagtm_matrix_centering(gsl_matrix * X, const gsl_vector * m)
{
    size_t j;

    if (X->size2 != m->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    for (j = 0; j < X->size1; j++)
    {
        gsl_vector_view v = gsl_matrix_row(X, j);
        gsl_vector_sub(&v.vector, m);
    }

    return GSL_SUCCESS;
}


int dagtm_matrix_colapply(const gsl_matrix * M,
                          double (*func) (const gsl_vector *),
                          gsl_vector * col)
{
    size_t j;

    if (M->size2 != col->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    for (j = 0; j < M->size2; j++)
    {
        gsl_vector_const_view v = gsl_matrix_const_column(M, j);
        double result = (*func) (&v.vector);
        gsl_vector_set(col, j, result);
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_scale_f(gsl_matrix * M, const double f)
{
    return dagtm_matrix_scale_f_vec(M->size1, M->size2, M->tda,
                                    (double (*)[]) M->data, f);
}

int dagtm_matrix_apply_exp(gsl_matrix * M)
{
    return dagtm_matrix_apply_exp_vec(M->size1, M->size2, M->tda,
                                      (double (*)[]) M->data);
}

int dagtm_matrix_scale_by_row(gsl_matrix * M, const gsl_vector * v)
{
    size_t i;

    if (M->size1 != v->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (v->stride != 1)
        DAGTM_ERROR("Vector is not continuous", GSL_EBADLEN);

    return dagtm_matrix_scale_by_row_vec(M->size1, M->size2, M->tda,
                                         (double (*)[]) M->data, v->size,
                                         (double (*)) v->data);

    /*
       for (i = 0; i < M->size1; i++)
       {
       gsl_vector_view row = gsl_matrix_row(M, i);
       gsl_vector_scale(&row.vector, gsl_vector_get(v, i));
       }

       return GSL_SUCCESS;
     */
}

int dagtm_matrix_scale_by_col(gsl_matrix * M, const gsl_vector * v)
{
    size_t i;

    if (M->size2 != v->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (v->stride != 1)
        DAGTM_ERROR("Vector is not continuous", GSL_EBADLEN);

    return dagtm_matrix_scale_by_col_vec(M->size1, M->size2, M->tda,
                                         (double (*)[]) M->data, v->size,
                                         (double (*)) v->data);
}

int dagtm_matrix_div_by_col(gsl_matrix * M, const gsl_vector * v)
{
    size_t i;

    if (M->size2 != v->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (v->stride != 1)
        DAGTM_ERROR("Vector is not continuous", GSL_EBADLEN);

    return dagtm_matrix_div_by_col_vec(M->size1, M->size2, M->tda,
                                       (double (*)[]) M->data, v->size,
                                       (double (*)) v->data);
}

double dagtm_stats_mean(const gsl_vector * v)
{
    return gsl_stats_mean(v->data, v->stride, v->size);
}

double dagtm_stats_sd(const gsl_vector * v)
{
    return gsl_stats_sd(v->data, v->stride, v->size);
}

double dagtm_stats_sd_m(const gsl_vector * v, const double mean)
{
    return gsl_stats_sd_m(v->data, v->stride, v->size, mean);
}

int dagtm_matrix_colmean(const gsl_matrix * M, gsl_vector * m)
{
    if (M->size2 != m->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    for (size_t i = 0; i < M->size2; i++)
    {
        gsl_vector_view v = gsl_matrix_column((gsl_matrix *)M, i);
        gsl_vector_set(m, i, dagtm_stats_mean(&v.vector));
    }

    return GSL_SUCCESS;
}

double dagtm_sq(double d)
{
    return d * d;
}

/*
 *	Compute D, where d(i,j) = ( a(i,) - b(j,) )^2
 *  Input:
 *  A : m-by-d
 *  B : n-by-d
 *  Output:
 *  C : m-by-n
 */
int dagtm_dist2(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * D)
{
    size_t i, j, k;
    size_t m = A->size1, n = B->size1, d1 = A->size2;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            double dist = 0;
            for (k = 0; k < d1; k++)
            {
                double d =
                    gsl_matrix_get(A, i, k) - gsl_matrix_get(B, j, k);
                dist += d * d;
            }
            gsl_matrix_set(D, i, j, dist);
        }
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_trans(const gsl_matrix * A, gsl_matrix * B)
{
    if (A->size1 != B->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (A->size2 != B->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

  return dagtm_matrix_trans_vec(A->size1, A->size2, A->tda, (double(*)[])A->data,
				B->size1, B->size2, B->tda, (double(*)[])B->data);
}

int dagtm_dist(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * D)
{
#ifdef COMM_VECTORIZED_DIST_ON
#warning Use vectorized dist
    return dagtm_dist_vec(A->size1, A->size2, A->tda,
                          (double (*)[*]) A->data, B->size1, B->size2,
                          B->tda, (double (*)[*]) B->data, D->size1,
                          D->size2, D->tda, (double (*)[*]) D->data);
#else
    size_t i;
    size_t m = A->size1, n = B->size1, d1 = A->size2, d2 = B->size2;
    double d;
    gsl_vector *varow, *vbrow, *voneArow, *voneBrow;

    if (d1 != d2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (m != D->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (n != D->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    varow = gsl_vector_alloc(A->size1);
    vbrow = gsl_vector_alloc(B->size1);
    voneArow = gsl_vector_alloc(A->size1);
    voneBrow = gsl_vector_alloc(B->size1);

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, -2, A, B, 0, D);

    gsl_vector_set_all(voneArow, 1);
    gsl_vector_set_all(voneBrow, 1);

    for (i = 0; i < A->size1; i++)
    {
        gsl_vector_const_view row = gsl_matrix_const_row(A, i);
        gsl_blas_ddot(&row.vector, &row.vector, &d);
        gsl_vector_set(varow, i, d);
    }

    for (i = 0; i < B->size1; i++)
    {
        gsl_vector_const_view row = gsl_matrix_const_row(B, i);
        gsl_blas_ddot(&row.vector, &row.vector, &d);
        gsl_vector_set(vbrow, i, d);
    }

    gsl_blas_dger(1, varow, voneBrow, D);
    gsl_blas_dger(1, voneArow, vbrow, D);

    gsl_vector_free(varow);
    gsl_vector_free(vbrow);
    gsl_vector_free(voneArow);
    gsl_vector_free(voneBrow);

    return GSL_SUCCESS;
#endif
}

int dagtm_dist_ws_test(const gsl_matrix * A, const gsl_matrix * B,
                  gsl_matrix * D, dagtm_dist_workspace * ws)
{
#ifdef COMM_VECTORIZED_DIST_ON
#warning Use vectorized dist
    return dagtm_dist_vec(A->size1, A->size2, A->tda,
                          (double (*)[*]) A->data, B->size1, B->size2,
                          B->tda, (double (*)[*]) B->data, D->size1,
                          D->size2, D->tda, (double (*)[*]) D->data);
#else
    size_t i;
    size_t m = A->size1, n = B->size1, d1 = A->size2, d2 = B->size2;
    double d;
    gsl_vector *varow, *vbrow, *voneArow, *voneBrow;

    if (d1 != d2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (m != D->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (n != D->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    gsl_matrix * B_T = gsl_matrix_alloc(B->size2, B->size1);
    dagtm_matrix_trans(B, B_T);

    // Begin of test
    LOG_TIC(LOG_DIST_GEMM_MLIB);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, -2, A, B, 0, D);
    LOG_TOC(LOG_DIST_GEMM_MLIB);
    //DUMPM(D);

    // Begin of test
    LOG_TIC(LOG_DIST_GEMM_MLIB2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -2, A, B_T, 0, D);
    LOG_TOC(LOG_DIST_GEMM_MLIB2);
    //DUMPM(D);

    LOG_TIC(LOG_DIST_GEMM_BASE);
    dagtm_gsl_blas_gemm_vec(-2, A->size1, A->size2, A->tda, (double (*)[])A->data,
			    B->size1, B->size2, B->tda, (double (*)[])B->data,
			    D->size1, D->size2, D->tda, (double (*)[])D->data);
    LOG_TOC(LOG_DIST_GEMM_BASE);
    //DUMPM(D);

    LOG_TIC(LOG_DIST_GEMM_CCBL);
    dagtm_gsl_blas_gemm_vec_cb(-2, A->size1, A->size2, A->tda, (double (*)[])A->data,
			    B->size1, B->size2, B->tda, (double (*)[])B->data,
			    D->size1, D->size2, D->tda, (double (*)[])D->data);
    LOG_TOC(LOG_DIST_GEMM_CCBL);
    //DUMPM(D);

    LOG_TIC(LOG_DIST_GEMM_STRS);
    dagtm_gsl_blas_gemm_vec_ss(-2, A->size1, A->size2, A->tda, (double (*)[])A->data,
			    B->size1, B->size2, B->tda, (double (*)[])B->data,
			    D->size1, D->size2, D->tda, (double (*)[])D->data);
    LOG_TOC(LOG_DIST_GEMM_STRS);
    // End of test

    LOG_TIC(LOG_BETA_DIST_GEMM);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, -2, A, B, 0, D);
    LOG_TOC(LOG_BETA_DIST_GEMM);

    //gsl_vector_set_all(voneArow, 1);
    //gsl_vector_set_all(voneBrow, 1);

    LOG_TIC(LOG_BETA_DIST_DOT1);
    for (i = 0; i < A->size1; i++)
    {
        gsl_vector_const_view row = gsl_matrix_const_row(A, i);
        gsl_blas_ddot(&row.vector, &row.vector, &d);
        gsl_vector_set(ws->v1, i, d);
    }
    LOG_TOC(LOG_BETA_DIST_DOT1);

    LOG_TIC(LOG_BETA_DIST_DOT2);
    for (i = 0; i < B->size1; i++)
    {
        gsl_vector_const_view row = gsl_matrix_const_row(B, i);
        gsl_blas_ddot(&row.vector, &row.vector, &d);
        gsl_vector_set(ws->v2, i, d);
    }
    LOG_TOC(LOG_BETA_DIST_DOT2);

    LOG_TIC(LOG_BETA_DIST_GER1);
    gsl_blas_dger(1, ws->v1, ws->vones2, D);
    LOG_TOC(LOG_BETA_DIST_GER1);

    LOG_TIC(LOG_BETA_DIST_GER2);
    gsl_blas_dger(1, ws->vones1, ws->v2, D);
    LOG_TOC(LOG_BETA_DIST_GER2);

    //gsl_vector_free(varow);
    //gsl_vector_free(vbrow);
    //gsl_vector_free(voneArow);
    //gsl_vector_free(voneBrow);

    return GSL_SUCCESS;
#endif
}

int dagtm_dist_ws(const gsl_matrix * A, const gsl_matrix * B,
                  gsl_matrix * D, dagtm_dist_workspace * ws)
{
#ifdef COMM_VECTORIZED_DIST_ON
#warning Use vectorized dist
    return dagtm_dist_vec(A->size1, A->size2, A->tda,
                          (double (*)[*]) A->data, B->size1, B->size2,
                          B->tda, (double (*)[*]) B->data, D->size1,
                          D->size2, D->tda, (double (*)[*]) D->data);
#else
    size_t i;
    size_t m = A->size1, n = B->size1, d1 = A->size2, d2 = B->size2;
    double d;
    gsl_vector *varow, *vbrow, *voneArow, *voneBrow;

    if (d1 != d2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (m != D->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (n != D->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    //varow = gsl_vector_alloc(A->size1);
    //vbrow = gsl_vector_alloc(B->size1);
    //voneArow = gsl_vector_alloc(A->size1);
    //voneBrow = gsl_vector_alloc(B->size1);

    LOG_TIC(LOG_BETA_DIST_GEMM);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, -2, A, B, 0, D);
    LOG_TOC(LOG_BETA_DIST_GEMM);

    //gsl_vector_set_all(voneArow, 1);
    //gsl_vector_set_all(voneBrow, 1);

    LOG_TIC(LOG_BETA_DIST_DOT1);
    for (i = 0; i < A->size1; i++)
    {
        gsl_vector_const_view row = gsl_matrix_const_row(A, i);
        gsl_blas_ddot(&row.vector, &row.vector, &d);
        gsl_vector_set(ws->v1, i, d);
    }
    LOG_TOC(LOG_BETA_DIST_DOT1);

    LOG_TIC(LOG_BETA_DIST_DOT2);
    for (i = 0; i < B->size1; i++)
    {
        gsl_vector_const_view row = gsl_matrix_const_row(B, i);
        gsl_blas_ddot(&row.vector, &row.vector, &d);
        gsl_vector_set(ws->v2, i, d);
    }
    LOG_TOC(LOG_BETA_DIST_DOT2);

    LOG_TIC(LOG_BETA_DIST_GER1);
    gsl_blas_dger(1, ws->v1, ws->vones2, D);
    LOG_TOC(LOG_BETA_DIST_GER1);

    LOG_TIC(LOG_BETA_DIST_GER2);
    gsl_blas_dger(1, ws->vones1, ws->v2, D);
    LOG_TOC(LOG_BETA_DIST_GER2);

    //gsl_vector_free(varow);
    //gsl_vector_free(vbrow);
    //gsl_vector_free(voneArow);
    //gsl_vector_free(voneBrow);

    return GSL_SUCCESS;
#endif
}

int dagtm_resp(gsl_matrix * D, const double beta)
{
    size_t j;

    gsl_vector *vSum, *vones;

    vSum = gsl_vector_alloc(D->size2);
    vones = gsl_vector_alloc(D->size1);

    gsl_matrix_scale(D, -beta / 2);
#ifdef COMM_IPP_ON
    IppStatus st = ippsExp_64f_I(D->data, D->size1 * D->size2);
#else
    dagtm_matrix_apply(D, &exp);
#endif
    dagtm_matrix_colsum(D, vSum);

    // Need to update to call dagtm_matrix_scale_by_col_vec 
    for (j = 0; j < vSum->size; j++)
    {
        gsl_vector_view col = gsl_matrix_column(D, j);
        double d = gsl_vector_get(vSum, j);

        if (d == 0)
        {
            d = (double) D->size1;
            gsl_vector_set_all(&col.vector, 1 / d);
        }
        else
        {
            gsl_vector_scale(&col.vector, 1 / d);
        }
    }

    gsl_vector_free(vSum);
    gsl_vector_free(vones);

    return GSL_SUCCESS;
}

int dagtm_matrix_rowsum(const gsl_matrix * M, gsl_vector * v)
{
    if (M->size1 != v->size)
        return 1;

    return dagtm_matrix_rowsum_vec(M->size1, M->size2, M->tda,
                                   (double (*)[]) M->data, v->size,
                                   (double (*)) v->data);
    /*
       size_t i, j;

       if (A->size1 != v->size)
       return 1;

       for (i = 0; i < A->size1; i++)
       {
       double sum = 0;
       for (j = 0; j < A->size2; j++)
       {
       sum += gsl_matrix_get(A, i, j);
       }
       gsl_vector_set(v, i, sum);
       }
     */

    /*
       gsl_vector *one;

       if (A->size1 != v->size) return 1;

       one = gsl_vector_alloc(A->size2);
       gsl_vector_set_all(one, 1);
       gsl_blas_dgemv(CblasNoTrans, 1, A, one, 0, v);
       gsl_vector_free(one);
     */
    //return GSL_SUCCESS;
}

int dagtm_matrix_colsum(const gsl_matrix * A, gsl_vector * v)
{
    size_t i, j;

    if (A->size2 != v->size)
        return 1;

    // 4.376
    return dagtm_matrix_colsum_vec(A->size1, A->size2, A->tda,
                                   (double (*)[]) A->data, v->size,
                                   (double (*)) v->data);
    /* 4.818 sec
       gsl_matrix *C = gsl_matrix_alloc(A->size2, A->size1);
       dagtm_trans_copy_vec(A->size1, A->size2, A->tda, (double (*)[])A->data, 
       C->size1, C->size2, C->tda, (double (*)[])C->data);

       dagtm_matrix_rowsum_vec(C->size1, C->size2, C->tda, (double (*)[])C->data, 
       v->size, (double (*))v->data);

       gsl_matrix_free(C);

       return GSL_SUCCESS;
     */

    /* 4.827 sec
       for (j = 0; j < A->size2; j++)
       {
       double sum = 0;
       for (i = 0; i < A->size1; i++)
       {
       sum += gsl_matrix_get(A, i, j);
       }
       gsl_vector_set(v, j, sum);
       }
     */
    /* 4.401 sec
       gsl_vector *one;

       if (A->size2 != v->size) return 1;

       one = gsl_vector_alloc(A->size1);
       gsl_vector_set_all(one, 1);
       gsl_blas_dgemv(CblasTrans, 1, A, one, 0, v);
       gsl_vector_free(one);
     */

    //return GSL_SUCCESS;
}

int dagtm_matrix_colsum2(const gsl_matrix * A, gsl_vector * v)
{
    size_t i, j;

    if (A->size2 != v->size)
        return 1;

    for (j = 0; j < A->size2; j++)
    {
        double sum = 0;
        for (i = 0; i < A->size1; i++)
        {
            sum += gsl_matrix_get(A, i, j);
        }
        gsl_vector_set(v, j, sum);
    }
    /*
       gsl_vector *one;

       if (A->size2 != v->size) return 1;

       one = gsl_vector_alloc(A->size1);
       gsl_vector_set_all(one, 1);
       gsl_blas_dgemv(CblasTrans, 1, A, one, 0, v);
       gsl_vector_free(one);
     */
    return GSL_SUCCESS;
}

int dagtm_matrix_set_diag(const gsl_vector * v, gsl_matrix * A)
{
    size_t i;
    for (i = 0; i < v->size; i++)
    {
        gsl_matrix_set(A, i, i, gsl_vector_get(v, i));
    }

    return GSL_SUCCESS;
}

int dagtm_matrix_fprintf(FILE * fp, const gsl_matrix * M, const char *fmt,
                         const char *del, const size_t nrow,
                         const size_t ncol)
{
    size_t i, j;
    char _fmt[MAXLEN];

    if (M == NULL)
        return GSL_EFAULT;

    strcat(strcpy(_fmt, fmt), del);
    for (i = 0; i < MIN(M->size1, nrow); i++)
    {
        for (j = 0; j < MIN(M->size2, ncol); j++)
        {
            fprintf(fp, _fmt, gsl_matrix_get(M, i, j));
        }
        fprintf(fp, "\n");
    }

    return GSL_SUCCESS;
}

int dagtm_vector_fprintf(FILE * fp, const gsl_vector * v, const char *fmt,
                         const char *del, const size_t ncol)
{
    size_t i;
    char _fmt[MAXLEN];

    if (v == NULL)
        return GSL_EFAULT;

    strcat(strcpy(_fmt, fmt), del);
    for (i = 0; i < MIN(v->size, ncol); i++)
    {
        fprintf(fp, _fmt, gsl_vector_get(v, i));
    }
    fprintf(fp, "\n");

    return GSL_SUCCESS;
}

int dagtm_vector_matrix_fprintf(FILE * fp, const gsl_vector * v,
                                const gsl_matrix * M, const char *fmt,
                                const char *del)
{
    size_t i, j;
    char _fmt[MAXLEN];

    if (M == NULL)
        return GSL_EFAULT;

    strcat(strcpy(_fmt, fmt), del);
    for (i = 0; i < M->size1; i++)
    {
        fprintf(fp, _fmt, gsl_vector_get(v, i));

        for (j = 0; j < M->size2; j++)
        {
            fprintf(fp, _fmt, gsl_matrix_get(M, i, j));
        }
        fprintf(fp, "\n");
    }

    return GSL_SUCCESS;
}

int dagtm_split(const int n, const int nsplit, int *counts, int *offsets)
{
    int i = 0;

    if (nsplit == 0)
        DAGTM_ERROR("Cannot split by zero", GSL_EZERODIV);

    // i = 0
    offsets[0] = 0;
    counts[0] = ceil(n / (double) nsplit);

    // i > 0
    for (i = 1; i < nsplit; i++)
    {
        offsets[i] = offsets[i - 1] + counts[i - 1];
        counts[i] = ceil((n - offsets[i]) / (double) (nsplit - i));
    }

    return GSL_SUCCESS;
}

gsl_matrix *dagtm_matrix_alloc(const size_t n1, const size_t n2,
                               const char *filepath)
{
    FILE *fp;
    gsl_matrix *M;
    //char line[MAXLEN];

    //_getcwd(line, MAXLEN);
    fp = fopen(filepath, "r");
    if (fp == NULL)
        DAGTM_ERROR_VAL("No file", GSL_EOF, NULL);

    if (fgetc(fp) == '#')
    {
        // skip 1 line
        while (fgetc(fp) != '\n');
    }
    else
    {
        // Rewind
        fseek(fp, 0L, SEEK_SET);
    }
    //fgets(line, MAXLEN, fp);

    M = gsl_matrix_alloc(n1, n2);
    gsl_matrix_fscanf(fp, M);
    fclose(fp);

    return M;
}

int dagtm_matrix_fileread(const char *filepath, gsl_matrix * M)
{
    FILE *fp;

    //_getcwd(line, MAXLEN);
    fp = fopen(filepath, "r");
    if (fp == NULL)
        DAGTM_ERROR("No file", GSL_EOF);

    if (fgetc(fp) == '#')
    {
        // skip 1 line
        while (fgetc(fp) != '\n');
    }
    else
    {
        // Rewind
        fseek(fp, 0L, SEEK_SET);
    }
    //fgets(line, MAXLEN, fp);

    gsl_matrix_fscanf(fp, M);
    fclose(fp);

    return GSL_SUCCESS;
}

gsl_vector *dagtm_vector_alloc(const size_t n, const char *filepath)
{
    FILE *fp;
    gsl_vector *v;

    v = gsl_vector_alloc(n);

    fp = fopen(filepath, "r");
    gsl_vector_fscanf(fp, v);
    fclose(fp);

    return v;
}

int dagtm_hilbert(const unsigned nDims, const unsigned nBits, gsl_matrix * X) 
{
    unsigned nPrints, r;
    nPrints = 1ULL << (nDims * nBits);
    if (nPrints != X->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    bitmask_t coord[64];
    
    //hilbert_i2c(*nDims, *nBits, 1, coord);
    for (r = 0; r < nPrints; ++r) {
        hilbert_i2c(nDims, nBits, r, coord);
        
        for (size_t j = 0; j < nDims; j++)
        {
            gsl_matrix_set(X, r, j, (double)2/((1ULL << nBits) - 1)*coord[j] - 1);
            //DUMP("X[%ld, %ld] = %f (%d)", r, j, gsl_matrix_get(X, r, j), coord[j]);
        }
        //DUMP("\n");
    }

    return GSL_SUCCESS;
}

/**
 *  Setup
 *  @param[in] noLatDim number of latent dimension (=L)
 *  @param[in] noLatVarSmpl number of latent variables (=K)
 *  @param[in] noBasFn number of base function (=M)
 *  @param[in] s
 *  @param[in] gridtype sweep, space-filling-curve
 *  @param[out] X
 *  @param[out] FI
 */
int dagtm_stp(const int noLatDim, const int noLatVarSmpl,
              const int noBasFn, const double s, const GRID gridtype,
              gsl_matrix * X, gsl_matrix * FI)
{
    int gridXdim = 0, gridFIdim = 0;
    double sigma, scale;

    gsl_matrix *MU = gsl_matrix_alloc(noBasFn, noLatDim);

    if (s <= 0)
        return -1;

    switch (gridtype)
    {
    case GRID_SWEEP:
        gridXdim = (int) rint(pow(noLatVarSmpl, 1 / (double) noLatDim));
        gridFIdim = (int) rint(pow(noBasFn, 1 / (double) noLatDim));

        if (pow(gridXdim, noLatDim) != noLatVarSmpl)
            DAGTM_ERROR("K is not acceptable", GSL_EBADLEN);

        if (pow(gridFIdim, noLatDim) != noBasFn)
            DAGTM_ERROR("M is not acceptable", GSL_EBADLEN);

        switch (noLatDim)
        {
        case 2:
            dagtm_rctg(gridXdim, gridXdim, 0, X);
            dagtm_rctg(gridFIdim, gridFIdim, 0, MU);
            break;
        case 3:
            dagtm_rctg(gridXdim, gridXdim, gridXdim, X);
            dagtm_rctg(gridFIdim, gridFIdim, gridFIdim, MU);
            break;
        default:
            DAGTM_ERROR("Latent dimension should be only either 2 or 3.",
                        GSL_EINVAL);
            break;
        }
        break;
    case GRID_SPACEFILLING:
    {
        gridXdim = (int) rint(pow(noLatVarSmpl, 1 / (double) noLatDim));
        gridFIdim = (int) rint(pow(noBasFn, 1 / (double) noLatDim));

        if (pow(gridXdim, noLatDim) != noLatVarSmpl)
            DAGTM_ERROR("K is not acceptable", GSL_EBADLEN);

        if (pow(gridFIdim, noLatDim) != noBasFn)
            DAGTM_ERROR("M is not acceptable", GSL_EBADLEN);

        unsigned gridXdim_nBits = (unsigned) rint(log2(gridXdim));
        unsigned gridFIdim_nBits = (unsigned) rint(log2(gridFIdim));

        if (pow(2, gridXdim_nBits) != gridXdim)
            DAGTM_ERROR("K is not acceptable", GSL_EBADLEN);

        if (pow(2, gridFIdim_nBits) != gridFIdim)
            DAGTM_ERROR("M is not acceptable", GSL_EBADLEN);

        dagtm_hilbert(noLatDim, gridXdim_nBits, X);
        dagtm_hilbert(noLatDim, gridFIdim_nBits, MU);
    }
        break;
    default:
        DAGTM_ERROR("Not a valid grid type", GSL_EINVAL);
        break;
    }

    /*
      if (gridtype == GRID_SWEEP)
      {
      gridXdim = (int) rint(pow(noLatVarSmpl, 1 / (double) noLatDim));
      gridFIdim = (int) rint(pow(noBasFn, 1 / (double) noLatDim));

      if (pow(gridXdim, noLatDim) != noLatVarSmpl)
      DAGTM_ERROR("K is not acceptable", GSL_EBADLEN);

      if (pow(gridFIdim, noLatDim) != noBasFn)
      DAGTM_ERROR("M is not acceptable", GSL_EBADLEN);

      switch (noLatDim)
      {
      case 2:
      dagtm_rctg(gridXdim, gridXdim, 0, X);
      dagtm_rctg(gridFIdim, gridFIdim, 0, MU);
      break;
      case 3:
      dagtm_rctg(gridXdim, gridXdim, gridXdim, X);
      dagtm_rctg(gridFIdim, gridFIdim, gridFIdim, MU);
      break;
      default:
      DAGTM_ERROR("Latent dimension should be only either 2 or 3.",
      GSL_EINVAL);
      break;
      }
      }
    */

    scale = (double) gridFIdim / (gridFIdim - 1);
    gsl_matrix_scale(MU, scale);
    sigma = s * (gsl_matrix_get(MU, 1, 0) - gsl_matrix_get(MU, 0, 0));
    dagtm_gbf(MU, X, sigma, FI);
    gsl_matrix_free(MU);

    return GSL_SUCCESS;

}

/**
 *  Random setup
 *  @param[in] vTmu mean
 *  @param[in] vTsd sd
 *  @param[in] noLatDim number of latent dimension (=L)
 *  @param[in] noLatVarSmpl number of latent variables (=K)
 *  @param[in] noBasFn number of base function (=M)
 *  @param[in] gridtype sweep, space-filling-curve
 *  @param[out] X
 *  @param[out] FI
 *  @param[out] W
 *  @param[out] beta
 */
int dagtm_stp3_rnd(const gsl_vector * vTmu, const gsl_vector * vTsd,
                   const int noLatDim, const int noLatVarSmpl,
                   const int noBasFn, const double s, const GRID gridtype,
                   gsl_matrix * X, gsl_matrix * FI, gsl_matrix * W,
                   double *beta)
{
    int ret = 0;

    // Setup X and FI
    ret = dagtm_stp(noLatDim, noLatVarSmpl, noBasFn, s, gridtype, X, FI);

    if (ret != GSL_SUCCESS)
        return -1;


    // Randomize Y¡
    gsl_matrix *Yrnd = gsl_matrix_alloc(noLatVarSmpl, vTmu->size);

    // Bug? Doesn't generate the same random number for different sizes of grids!!
    dagtm_yrnd2(vTmu, vTsd, Yrnd);

    dagtm_solve(FI, Yrnd, W);

    // dagtm_bi will allocate memory of size noLatVarSmpl^2. 
    // If noLatVarSmpl is 64k, it's about 32GB.
    if (Yrnd->size1 > 1000)
    {
        *beta = 0.1;
    }
    else
    {
        *beta = dagtm_bi(Yrnd);
    }


    gsl_matrix_free(Yrnd);

    return GSL_SUCCESS;
}

/*
 * Compute initial beta. If Y->size1 > 1k, it's very memory consuming.
 */
double dagtm_bi(const gsl_matrix * Y)
{
    size_t i;
    gsl_matrix *D = gsl_matrix_alloc(Y->size1, Y->size1);
    gsl_vector *v = gsl_vector_alloc(Y->size1);
    double m;

    dagtm_dist(Y, Y, D);

    for (i = 0; i < Y->size1; i++)
    {
        gsl_matrix_set(D, i, i, DBL_MAX);
    }

    dagtm_matrix_colapply(D, &gsl_vector_min, v);
    m = dagtm_stats_mean(v);

    if (m == 0)
        m = 1;

    gsl_matrix_free(D);
    gsl_vector_free(v);

    return (2 / m);
}

/*
int
gsl_linalg_SV_solve (const gsl_matrix * U,
                     const gsl_matrix * V,
                     const gsl_vector * S,
                     const gsl_vector * b, gsl_vector * x)
{
	if (U->size1 != b->size)
    {
		GSL_ERROR ("first dimension of matrix U must size of vector b",
				   GSL_EBADLEN);
    }
	else if (U->size2 != S->size)
    {
		GSL_ERROR ("length of vector S must match second dimension of matrix U",
				   GSL_EBADLEN);
    }
	else if (V->size1 != V->size2)
    {
		GSL_ERROR ("matrix V must be square", GSL_ENOTSQR);
    }
	else if (S->size != V->size1)
    {
		GSL_ERROR ("length of vector S must match size of matrix V",
				   GSL_EBADLEN);
    }
	else if (V->size2 != x->size)
    {
		GSL_ERROR ("size of matrix V must match size of vector x", GSL_EBADLEN);
    }
	else
    {
		const size_t N = U->size2;
		size_t i;
		
		gsl_vector *w = gsl_vector_calloc (N);
		
		gsl_blas_dgemv (CblasTrans, 1.0, U, b, 0.0, w);
		
		for (i = 0; i < N; i++)
        {
			double wi = gsl_vector_get (w, i);
			double alpha = gsl_vector_get (S, i);
			if (alpha != 0)
				alpha = 1.0 / alpha;
			gsl_vector_set (w, i, alpha * wi);
        }
		
		gsl_blas_dgemv (CblasNoTrans, 1.0, V, w, 0.0, x);
		
		gsl_vector_free (w);
		
		return GSL_SUCCESS;
    }
}
*/

/*
 *  Solve t(A) * A * X = t(A) * B for X
 */
int dagtm_solve(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * X)
{
    gsl_matrix *U, *V, *AB;
    gsl_vector *s, *w;

    size_t j;
    size_t m = A->size1, n = A->size2, k = B->size2;

    if (m != B->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (n != X->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (k != X->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    U = gsl_matrix_alloc(n, n);
    V = gsl_matrix_alloc(n, n);
    AB = gsl_matrix_alloc(n, k);
    s = gsl_vector_alloc(n);
    w = gsl_vector_alloc(n);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, A, A, 0, U);

    // SVD. t(A) * A = U S t(V) 
    gsl_linalg_SV_decomp(U, V, s, w);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, A, B, 0, AB);

    for (j = 0; j < k; j++)
    {
        gsl_vector_const_view bcol = gsl_matrix_const_column(AB, j);
        gsl_vector_view xcol = gsl_matrix_column(X, j);

        // A x = b, where A = U S t(V)
        gsl_linalg_SV_solve(U, V, s, &bcol.vector, &xcol.vector);

    }

    gsl_matrix_free(U);
    gsl_matrix_free(V);
    gsl_matrix_free(AB);
    gsl_vector_free(s);
    gsl_vector_free(w);

    return GSL_SUCCESS;
}

/*
 *  Solve A * X = B for X
 */
int dagtm_solve2(const gsl_matrix * A, const gsl_matrix * B,
                 gsl_matrix * X)
{
    gsl_matrix *U, *V;
    //, *AB;
    gsl_vector *s, *w;

    size_t j;
    size_t m = A->size1, n = A->size2, k = B->size2;

    if (m != n)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (m != B->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (n != X->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (k != X->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    U = gsl_matrix_alloc(n, n);
    V = gsl_matrix_alloc(n, n);
    //AB = gsl_matrix_alloc(n, k);
    s = gsl_vector_alloc(n);
    w = gsl_vector_alloc(n);

    //gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, A, A, 0, U);
    gsl_matrix_memcpy(U, A);
    gsl_linalg_SV_decomp(U, V, s, w);

    //gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, A, B, 0, AB);

    for (j = 0; j < k; j++)
    {
        gsl_vector_const_view bcol = gsl_matrix_const_column(B, j);
        gsl_vector_view xcol = gsl_matrix_column(X, j);

        gsl_linalg_SV_solve(U, V, s, &bcol.vector, &xcol.vector);
    }

    gsl_matrix_free(U);
    gsl_matrix_free(V);
    //gsl_matrix_free(AB);
    gsl_vector_free(s);
    gsl_vector_free(w);

    return GSL_SUCCESS;
}

int dagtm_yrnd(const gsl_matrix * T, gsl_matrix * Yrnd)
{
    size_t i, j;

    const gsl_rng_type *t;
    gsl_rng *r;
    gsl_vector *mu, *sd;

    if (T->size2 != Yrnd->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    //dagtm_matrix_colsum(T, mu);
    //gsl_vector_scale(mu, (double) 1/T->size1);

    mu = gsl_vector_alloc(T->size2);
    sd = gsl_vector_alloc(T->size2);

    dagtm_matrix_colapply(T, &(dagtm_stats_mean), mu);
    dagtm_matrix_colapply(T, &(dagtm_stats_sd), sd);

    gsl_rng_env_setup();
    t = gsl_rng_default;
    r = gsl_rng_alloc(t);
    {
        if (dagtm_seed == 0)
        {
            time_t seed = time(NULL);
            dagtm_seed = seed;
        }
        gsl_rng_set(r, dagtm_seed);
    }

    for (i = 0; i < Yrnd->size1; i++)
    {
        for (j = 0; j < mu->size; j++)
        {
            double d = gsl_ran_gaussian(r,
                                        gsl_vector_get(sd,
                                                       j)) +
                gsl_vector_get(mu,
                               j);
            gsl_matrix_set(Yrnd, i, j, d);
        }
    }

    gsl_rng_free(r);
    gsl_vector_free(mu);
    gsl_vector_free(sd);
    return GSL_SUCCESS;
}

int dagtm_yrnd2(const gsl_vector * mu, const gsl_vector * sd,
                gsl_matrix * Yrnd)
{
    size_t i, j;

    const gsl_rng_type *t;
    gsl_rng *r;

    if (mu->size != Yrnd->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (sd->size != Yrnd->size2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    gsl_rng_env_setup();
    t = gsl_rng_default;
    r = gsl_rng_alloc(t);
    {
        if (dagtm_seed == 0)
        {
            time_t seed = time(NULL);
            dagtm_seed = seed;
        }
        gsl_rng_set(r, dagtm_seed);
    }

    double roundoff = 1E-15;
    for (i = 0; i < Yrnd->size1; i++)
    {
        for (j = 0; j < mu->size; j++)
        {
            // gsl_ran_gaussian is not consistent
            //double rnd = gsl_ran_gaussian(r, gsl_vector_get(sd,j)) + DBL_EPSILON;
            //double rnd = ceil(gsl_rng_uniform(r)/roundoff)*roundoff*2*gsl_vector_get(sd,j) - gsl_vector_get(sd,j);

            // 6 sigma
            double dev = gsl_vector_get(sd, j) * 6;
            double rnd = ceil(gsl_rng_uniform(r) / roundoff) * roundoff;
            double m = gsl_vector_get(mu, j);
            gsl_matrix_set(Yrnd, i, j, 2 * dev * rnd - dev + m);
            //gsl_matrix_set(Yrnd, i, j, m);
        }
    }

    gsl_rng_free(r);

    return GSL_SUCCESS;
}

/*
 *	input matrix X should be a xDim*yDim*zDim-by-3 matrix
 */
int dagtm_rctg(const size_t xDim, const size_t yDim, const size_t zDim,
               gsl_matrix * X)
{
    size_t x, y, z;
    int idx = 0;
    int m = MAX(MAX(xDim, yDim), zDim) - 1;

    if (zDim == 0)
    {
        for (y = 0; y < yDim; y++)
        {
            for (x = 0; x < xDim; x++)
            {
                gsl_matrix_set(X, idx, 0, x);
                gsl_matrix_set(X, idx, 1, y);
                idx += 1;
            }
        }
    }
    else
    {
        for (z = 0; z < zDim; z++)
        {
            for (y = 0; y < yDim; y++)
            {
                for (x = 0; x < xDim; x++)
                {
                    gsl_matrix_set(X, idx, 0, x);
                    gsl_matrix_set(X, idx, 1, y);
                    gsl_matrix_set(X, idx, 2, z);
                    idx += 1;
                }
            }
        }
    }

    gsl_matrix_scale(X, 2 / (double) m);
    gsl_matrix_add_constant(X, -1);

    return GSL_SUCCESS;
}

int dagtm_gbf(const gsl_matrix * MU, const gsl_matrix * X,
              const double sigma, gsl_matrix * FI)
{
    const size_t K = X->size1;
    const size_t L = X->size2;
    const size_t M = MU->size1;
    const size_t L2 = MU->size2;

    if (L != L2)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    {
        size_t i, j;
        gsl_matrix *DIST = gsl_matrix_calloc(K, M);
        dagtm_dist(X, MU, DIST);
        gsl_matrix_scale(DIST, -0.5 / pow(sigma, 2));
        dagtm_matrix_apply(DIST, &(exp));

        for (i = 0; i < DIST->size1; i++)
        {
            for (j = 0; j < DIST->size2; j++)
            {
                gsl_matrix_set(FI, i, j, gsl_matrix_get(DIST, i, j));
            }
            gsl_matrix_set(FI, i, j, 1);
        }
        gsl_matrix_free(DIST);
    }

    return GSL_SUCCESS;
}

int dagtm_strntok(char *str)
{
    char delims[] = " ,\t";
    char *result = NULL;
    int cnt = 0;

    result = strtok(str, delims);
    while (result != NULL)
    {
        cnt++;
        result = strtok(NULL, delims);
    }

    return cnt;
}

char *dagtm_file_readline(FILE * fp, char **buf, unsigned int *len)
{
    char *newbuf;
    //newbuf = buf;
    char *result = NULL;
    result = fgets(*buf, *len + 1, fp);

    while (strlen(*buf) == *len)
    {
        // Need to read more
        //*len = *len * 2;
        newbuf = (char *) malloc((*len * 2 + 1) * sizeof(char));
        newbuf[0] = '\0';
        strcat(newbuf, *buf);
        result = fgets(newbuf + *len, *len + 1, fp);
        *len = *len * 2;
        free(*buf);
        *buf = newbuf;
    }

    return result;
}

int dagtm_matrix_dim(const char *filepath, unsigned int *nrow,
                     unsigned int *ncol)
{
    FILE *fp;
    char *line;
    unsigned int len = MAXLEN;

    int nlines = 0;
    int ncols = 0;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        DAGTM_ERROR("No file", GSL_EOF);

    line = (char *) malloc((MAXLEN + 1) * sizeof(char));

    while (dagtm_file_readline(fp, &line, &len) != NULL)
    {
        if (strncmp(line, "#", 1) != 0)
        {
            nlines++;
            ncols = dagtm_strntok(line);
            break;
        }
    }

    while (dagtm_file_readline(fp, &line, &len) != NULL)
    {
        if (strncmp(line, "#", 1) != 0)
        {
            int newncols;
            nlines++;
            newncols = dagtm_strntok(line);

            if (ncols != newncols)
                DAGTM_ERROR("Different number of columns", GSL_EBADLEN);
        }

    }

    *nrow = nlines;
    *ncol = ncols;

    free(line);
    return GSL_SUCCESS;
}

int dagtm_file_lines(const char *filepath)
{
    FILE *fp;
    char *buff;
    unsigned int len = MAXLEN;
    int lines = 0;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        DAGTM_ERROR("No file", GSL_EOF);

    buff = (char *) malloc((MAXLEN + 1) * sizeof(char));

    while (dagtm_file_readline(fp, &buff, &len) != NULL)
    {
        if (strncmp(buff, "#", 1) != 0)
        {
            lines++;
        }
    }

    free(buff);
    return lines;
}

int dagtm_collection_alloc(dagtm_collection * col, const size_t size)
{
    col->vector = gsl_vector_alloc(size);
    if (col->vector == NULL)
    {
        return GSL_ENOMEM;
    }
    col->next = 0;
    col->init = size;

    return GSL_SUCCESS;
}

void dagtm_collection_free(dagtm_collection * col)
{
    gsl_vector_free(col->vector);
}

int dagtm_collection_expand(dagtm_collection * col)
{
    size_t size = col->vector->size;
    gsl_vector *newvec = gsl_vector_alloc(size * 2);

    if (newvec == NULL)
    {
        newvec = gsl_vector_alloc(size + col->init);
    }

    if (newvec == NULL)
    {
        return GSL_ENOMEM;
    }

    gsl_vector_view sub = gsl_vector_subvector(newvec, 0, size);
    gsl_vector_memcpy(&sub.vector, col->vector);
    gsl_vector_free(col->vector);
    col->vector = newvec;

    return GSL_SUCCESS;
}

int dagtm_collection_push(dagtm_collection * col, double val)
{
    if (col->next >= col->vector->size)
    {
        dagtm_collection_expand(col);
    }

    gsl_vector_set(col->vector, col->next, val);
    (col->next)++;

    return GSL_SUCCESS;
}


int dagtm_queue_alloc(dagtm_queue * que, const size_t size,
                      const double init)
{
    que->vector = gsl_vector_alloc(size);
    if (que->vector == NULL)
    {
        return GSL_ENOMEM;
    }

    dagtm_queue_reset(que, init);

    return GSL_SUCCESS;
}

int dagtm_queue_free(dagtm_queue * que)
{
    gsl_vector_free(que->vector);

    return GSL_SUCCESS;
}

int dagtm_queue_reset(dagtm_queue * que, const double init)
{
    gsl_vector_set_all(que->vector, init);
    que->next = 0;

    return GSL_SUCCESS;
}

int dagtm_queue_push(dagtm_queue * que, double val)
{
    gsl_vector_set(que->vector, que->next, val);
    (que->next)++;
    que->next = que->next % que->vector->size;

    return GSL_SUCCESS;
}

double dagtm_queue_mean(const dagtm_queue * que)
{
    double sum = dagtm_vector_sum(que->vector);
    return sum / que->vector->size;
}

TYPE(gsl_block) * FUNCTION(dagtm_gsl_block, alloc) (const size_t n)
{
    TYPE(gsl_block) * b;

    /*
    if (n == 0)
    {
        GSL_ERROR_VAL("block length n must be positive integer",
                      GSL_EINVAL, 0);
    }
    */

#ifdef __INTEL_COMPILER
    b = (TYPE(gsl_block) *) _mm_malloc(sizeof(TYPE(gsl_block)), CLS);
#else
    // No reference. Bug?
    // warning: implicit declaration of function 'posix_memalign'
    //posix_memalign (&b, CLS, sizeof (TYPE (gsl_block)));
    b = (TYPE(gsl_block) *) malloc(sizeof(TYPE(gsl_block)));
#endif

    if (b == 0)
    {
        GSL_ERROR_VAL("failed to allocate space for block struct",
                      GSL_ENOMEM, 0);
    }

#ifdef __INTEL_COMPILER
    b->data =
        (ATOMIC *) _mm_malloc(MULTIPLICITY * n * sizeof(ATOMIC), CLS);
#else
    //posix_memalign (b->data, CLS, MULTIPLICITY * n * sizeof (ATOMIC));
    b->data = (ATOMIC *) malloc(MULTIPLICITY * n * sizeof(ATOMIC));
#endif

    if (b->data == 0)
    {
#ifdef __INTEL_COMPILER
        _mm_free(b);
#else
        free(b);                /* exception in constructor, avoid memory leak */
#endif

        GSL_ERROR_VAL("failed to allocate space for block data",
                      GSL_ENOMEM, 0);
    }

    b->size = n;

    return b;
}

TYPE(gsl_block) * FUNCTION(dagtm_gsl_block, calloc) (const size_t n)
{
    size_t i;

    TYPE(gsl_block) * b = FUNCTION(dagtm_gsl_block, alloc) (n);

    if (b == 0)
        return 0;

    /* initialize block to zero */
    /*
       for (i = 0; i < MULTIPLICITY * n; i++)
       {
       b->data[i] = 0;
       }
     */
    memset(b->data, 0, MULTIPLICITY * n * sizeof(ATOMIC));


    return b;
}

void FUNCTION(dagtm_gsl_block, free) (TYPE(gsl_block) * b) {
    RETURN_IF_NULL(b);

#ifdef __INTEL_COMPILER
    _mm_free(b->data);
    _mm_free(b);
#else
    free(b->data);
    free(b);                    /* exception in constructor, avoid memory leak */
#endif
}

TYPE(gsl_matrix) * 
FUNCTION(dagtm_gsl_matrix, alloc) (const size_t n1, const size_t n2)
{
    //DUMP("dagtm_gsl_matrix_alloc :(%ld * %ld):%ld", n1, n2, n1*n2);
    TYPE(gsl_block) * block;
    TYPE(gsl_matrix) * m;

    /*
      if (n1 == 0)
      {
      GSL_ERROR_VAL("matrix dimension n1 must be positive integer",
      GSL_EINVAL, 0);
      }
      else if (n2 == 0)
      {
      GSL_ERROR_VAL("matrix dimension n2 must be positive integer",
      GSL_EINVAL, 0);
      }
    */

#ifdef __INTEL_COMPILER
    m = (TYPE(gsl_matrix) *) _mm_malloc(sizeof(TYPE(gsl_matrix)), CLS);
#else
    //posix_memalign (m, CLS, sizeof (TYPE (gsl_matrix)));
    m = (TYPE(gsl_matrix) *) malloc(sizeof(TYPE(gsl_matrix)));
#endif

    if (m == 0)
    {
        GSL_ERROR_VAL("failed to allocate space for matrix struct",
                      GSL_ENOMEM, 0);
    }

    /* FIXME: n1*n2 could overflow for large dimensions */

    size_t sm = CLS / sizeof(ATOMIC);
    size_t tda;
    if (n2 % sm)
    {
        tda = n2 + sm - n2%sm;
    }
    else
    {
        tda = n2;
    }

    // To advoid cache thrashing(32K bytes which is the case of AMD Opteron)
    // FIXME : Need to be more general.
    if (tda % (32 * 1024 / sizeof(ATOMIC)) == 0)
    {
        tda = tda + sm;
        DEBUG(DAGTM_INFO_MSG, "Cache thrashing may occur. Pads are added ... ");
    }

    block = FUNCTION(dagtm_gsl_block, alloc) (n1 * tda);

    if (block == 0)
    {
        GSL_ERROR_VAL("failed to allocate space for block", GSL_ENOMEM, 0);
    }

    m->data = block->data;
    m->size1 = n1;
    m->size2 = n2;
    m->tda = tda;
    m->block = block;
    m->owner = 1;

    return m;
}

TYPE(gsl_matrix) *
FUNCTION(dagtm_gsl_matrix, calloc) (const size_t n1, const size_t n2)
{
    //DUMP("dagtm_gsl_matrix_calloc :(%ld * %ld):%ld", n1, n2, n1*n2);
    //DUMP("dagtm_gsl_matrix_alloc :-(%ld * %ld):-%ld", n1, n2, n1*n2);
    size_t i;

    TYPE(gsl_matrix) * m = FUNCTION(dagtm_gsl_matrix, alloc) (n1, n2);

    if (m == 0)
        return 0;

    /* initialize matrix to zero */
    /*
       for (i = 0; i < MULTIPLICITY * n1 * n2; i++)
       {
       m->data[i] = 0;
       }
     */
    memset(m->data, 0, MULTIPLICITY * n1 * m->tda * sizeof(ATOMIC));

    return m;
}

void FUNCTION(dagtm_gsl_matrix, free) (TYPE(gsl_matrix) * m) {
    //DUMP("dagtm_gsl_matrix_free :-(%ld * %ld):-%ld", m->size1, m->size2, (m->size1) * (m->size2));
    RETURN_IF_NULL(m);

    if (m->owner)
    {
        FUNCTION(dagtm_gsl_block, free) (m->block);
    }

#ifdef __INTEL_COMPILER
    _mm_free(m);
#else
    free(m);
#endif
}

TYPE(gsl_vector) * FUNCTION(dagtm_gsl_vector, alloc) (const size_t n)
{
    //DUMP("dagtm_gsl_vector_alloc :%ld:%ld", n, n);
    TYPE(gsl_block) * block;
    TYPE(gsl_vector) * v;

    /*
    if (n == 0)
    {
        GSL_ERROR_VAL("vector length n must be positive integer",
                      GSL_EINVAL, 0);
    }
    */

#ifdef __INTEL_COMPILER
    v = (TYPE(gsl_vector) *) _mm_malloc(sizeof(TYPE(gsl_vector)), CLS);
#else
    //posix_memalign (v, CLS, sizeof (TYPE (gsl_vector)));
    v = (TYPE(gsl_vector) *) malloc(sizeof(TYPE(gsl_vector)));
#endif

    if (v == 0)
    {
        GSL_ERROR_VAL("failed to allocate space for vector struct",
                      GSL_ENOMEM, 0);
    }

    block = FUNCTION(dagtm_gsl_block, alloc) (n);

    if (block == 0)
    {
#ifdef __INTEL_COMPILER
        _mm_free(v);
#else
        free(v);
#endif

        GSL_ERROR_VAL("failed to allocate space for block", GSL_ENOMEM, 0);
    }

    v->data = block->data;
    v->size = n;
    v->stride = 1;
    v->block = block;
    v->owner = 1;

    return v;
}

TYPE(gsl_vector) * FUNCTION(dagtm_gsl_vector, calloc) (const size_t n)
{
    //DUMP("dagtm_gsl_vector_calloc :%ld:%ld", n, n);
    //DUMP("dagtm_gsl_vector_alloc :-%ld:-%ld", n, n);
    size_t i;

    TYPE(gsl_vector) * v = FUNCTION(dagtm_gsl_vector, alloc) (n);

    if (v == 0)
        return 0;

    /* initialize vector to zero */
    /*
       for (i = 0; i < MULTIPLICITY * n; i++)
       {
       v->data[i] = 0;
       }
     */
    memset(v->data, 0, MULTIPLICITY * n * sizeof(ATOMIC));

    return v;
}

void FUNCTION(dagtm_gsl_vector, free) (TYPE(gsl_vector) * v) {
    //DUMP("dagtm_gsl_vector_free :-%ld:-%ld", v->size, v->size);
    RETURN_IF_NULL(v);

    if (v->owner)
    {
        FUNCTION(dagtm_gsl_block, free) (v->block);
    }

#ifdef __INTEL_COMPILER
    _mm_free(v);
#else
    free(v);
#endif
}

//void FUNCTION(dagtm_gsl_matrix, realloc) (TYPE(gsl_matrix) * m, const size_t n1, const size_t n2) 
/*
 * FIXME: doesn't work with Intel compiler (icc)
 */
void dagtm_gsl_matrix_realloc (gsl_matrix * m, const size_t n1, const size_t n2) 
{
    RETURN_IF_NULL(m);

    FUNCTION(dagtm_gsl_matrix, free) (m);
    m = FUNCTION(dagtm_gsl_matrix, calloc) (n1, n2);
    //return m;
}

void FUNCTION(dagtm_gsl_vector, realloc) (TYPE(gsl_vector) * v, const size_t n) 
{
    RETURN_IF_NULL(v);

    FUNCTION(dagtm_gsl_vector, free) (v);
    v = FUNCTION(dagtm_gsl_vector, calloc) (n);
}

int dagtm_gsl_vector_scale(gsl_vector * a, const double x)
{
    return dagtm_vector_scale_f_vec(a->size, a->stride,
                                    (double (*)) a->data, x);
}

int dagtm_gsl_matrix_mul_elements(gsl_matrix * a, const gsl_matrix * b)
{
    return dagtm_matrix_scale_by_mat_vec(a->size1, a->size2, a->tda,
                                         (double (*)[]) a->data, b->size1,
                                         b->size2, b->tda,
                                         (double (*)[]) b->data);
}

int dagtm_gsl_matrix_memcpy(gsl_matrix * dest, const gsl_matrix * src)
{
    if (src->tda == dest->tda)
    {
        return dagtm_gsl_matrix_memcpy_vec(dest->size1, dest->size2, dest->tda, (double (*)[]) dest->data,
                                           src->size1, src->size2, src->tda, (double (*)[]) src->data);
    }
    else
    {
        return dagtm_gsl_matrix_memcpy_byrow_vec(dest->size1, dest->size2, dest->tda, (double (*)[]) dest->data,
                                           src->size1, src->size2, src->tda, (double (*)[]) src->data);
    }
}

dagtm_ctemp_workspace *dagtm_ctemp_workspace_alloc(const size_t dim1,
                                                   const size_t dim2)
{
    dagtm_ctemp_workspace *ws =
        (dagtm_ctemp_workspace *) malloc(sizeof(dagtm_ctemp_workspace));
    ws->N = dim1;
    ws->D = dim2;

    ws->Xcentered = gsl_matrix_alloc(dim1, dim2);
    ws->resp = gsl_vector_alloc(dim1);
    ws->S1 = gsl_matrix_alloc(dim2, dim2);
    ws->S11 = gsl_matrix_alloc(dim2, dim2);
    ws->S12 = gsl_matrix_alloc(dim2, dim2);
    ws->H = gsl_matrix_alloc(2 * dim2, 2 * dim2);
    ws->w = gsl_eigen_symm_alloc(2 * dim2);
    ws->lambda = gsl_vector_alloc(2 * dim2);

    return ws;
}

void dagtm_ctemp_workspace_free(dagtm_ctemp_workspace * ws)
{
    gsl_matrix_free(ws->Xcentered);
    gsl_vector_free(ws->resp);
    gsl_matrix_free(ws->S1);
    gsl_matrix_free(ws->S11);
    gsl_matrix_free(ws->S12);
    gsl_matrix_free(ws->H);
    gsl_eigen_symm_free(ws->w);
    gsl_vector_free(ws->lambda);
    free(ws);
}

dagtm_dist_workspace *dagtm_dist_workspace_alloc(const size_t dim1,
                                                 const size_t dim2)
{
    dagtm_dist_workspace *ws =
        (dagtm_dist_workspace *) malloc(sizeof(dagtm_dist_workspace));

    ws->v1 = gsl_vector_alloc(dim1);
    ws->v2 = gsl_vector_alloc(dim2);
    ws->vones1 = gsl_vector_alloc(dim1);
    ws->vones2 = gsl_vector_alloc(dim2);

    gsl_vector_set_all(ws->vones1, 1);
    gsl_vector_set_all(ws->vones2, 1);

    return ws;
}

void dagtm_dist_workspace_free(dagtm_dist_workspace * ws)
{
    gsl_vector_free(ws->v1);
    gsl_vector_free(ws->v2);
    gsl_vector_free(ws->vones1);
    gsl_vector_free(ws->vones2);
    free(ws);
}

double dagtm_vector_max(gsl_vector * v, double limit)
{
    size_t i = 0;
    for (i = 0; i < v->size; i++)
    {
        if (gsl_vector_get(v, i) >= limit)
        {
            gsl_vector_set(v, i, 0.0);
        }
    }

    return gsl_vector_max(v);
}


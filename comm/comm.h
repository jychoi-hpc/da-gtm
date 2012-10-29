#ifndef __DAGTM_COMM_H__
#define __DAGTM_COMM_H__
//#include <intrin.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
//#include <direct.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics.h>

#include "hilbert.h"
#include "log.h"

extern int dagtm_myid;
extern int dagtm_loglevel;
extern FILE *dagtm_flog;
extern unsigned long dagtm_seed;

#define __DAGTM_COMM__
#define MAXLEN 160
#define HEAD_ROW_MAX 10
#define HEAD_COL_MAX 10
#define FMTSTR "%.020f"
#define TRUE 1
#define FALSE 0
#define FAIL -1

#define DAGTM_CRITIC_MSG   0
#define DAGTM_WARNING_MSG  1
#define DAGTM_INFO_MSG     2

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define ABS(a) (((a)>(0))?(a):(-a))

/**
 *  COMM Definitions
 *  COMM_IPP_ON (Default : OFF)
 *  COMM_VECTORIZED_DIST_ON (Default : OFF) -- Experimental
 *  COMM_ALIGN_OFF (Default : ON)
 *  CLS (Default : 64)
 *  SM (Default : 64 / sizeof(double))
 */

#ifdef COMM_IPP_ON
# undef COMM_IPP_OFF
#else
# define COMM_IPP_OFF
#endif

#ifdef COMM_VECTORIZED_DIST_ON
# undef COMM_VECTORIZED_DIST_OFF
#else
# define COMM_VECTORIZED_DIST_OFF
#endif

#ifdef COMM_ALIGN_OFF
# undef COMM_ALIGN_ON
#else
# define COMM_ALIGN_ON
#endif

#ifndef CLS
# define CLS 64
#endif
#define SM (CLS / sizeof (double))


#ifdef WIN32
//typedef unsigned __int64 tick_t;
#else
//typedef unsigned long long int tick_t;
#endif

typedef enum { GRID_SWEEP, GRID_SPACEFILLING } GRID;

#define DAGTM_SETLOGLEVEL(l) dagtm_loglevel = l
#define DAGTM_SETMYID(id) dagtm_myid = id
#define DAGTM_SETLOG(fp) dagtm_flog = fp
#define DAGTM_SETLOGFILE() { char fname[MAXLEN]; sprintf(fname, "dagtm.P%03d.log", dagtm_myid); dagtm_flog = fopen(fname, "w"); }
#define DAGTM_CLOSELOG() fclose(dagtm_flog)

#define DEBUGND(fmt, ...) fprintf (dagtm_flog, fmt"\n", ## __VA_ARGS__); fflush(dagtm_flog);
#define DEBUG(level, fmt, ...) if (level <= dagtm_loglevel) fprintf (dagtm_flog, ">>> [%03d] "fmt"\n", dagtm_myid, ## __VA_ARGS__); fflush(dagtm_flog);
#define DUMP(fmt, ...) fprintf(dagtm_flog, ">>> [%03d] "fmt"\n", dagtm_myid, ## __VA_ARGS__); fflush(dagtm_flog);
#define DUMPV(V) fprintf(dagtm_flog, ">>> [%03d] %s (len : %ld, sum : %.020f): \n", dagtm_myid, #V, V->size, dagtm_vector_sum(V)); dagtm_vector_fprintf(dagtm_flog, V, FMTSTR, " ", HEAD_COL_MAX); fflush(dagtm_flog);
#define DUMPM(M) fprintf(dagtm_flog, ">>> [%03d] %s (dim : (%ld,%ld), sum : %.020f): \n", dagtm_myid, #M, M->size1, M->size2, dagtm_matrix_sum(M)); dagtm_matrix_fprintf(dagtm_flog, M, FMTSTR, " ", HEAD_ROW_MAX, HEAD_COL_MAX); fflush(dagtm_flog);
#define DUMPVF(V) fprintf(dagtm_flog, ">>> [%03d] %s (len : %ld, sum : %.020f): \n", dagtm_myid, #V, V->size, dagtm_vector_sum(V)); dagtm_vector_fprintf(dagtm_flog, V, FMTSTR, " ", V->size); fflush(dagtm_flog);
#define DUMPMF(M) fprintf(dagtm_flog, ">>> [%03d] %s (dim : (%ld,%ld), sum : %.020f): \n", dagtm_myid, #M, M->size1, M->size2, dagtm_matrix_sum(M)); dagtm_matrix_fprintf(dagtm_flog, M, FMTSTR, " ", M->size1, M->size2); fflush(dagtm_flog);
#define DUMPA(cmd) \
  fprintf(dagtm_flog, ">>> [%03d] ", dagtm_myid); \
  cmd; \
  fflush(dagtm_flog);

#define DONLY(id, cmd) \
  if (dagtm_myid == id) { \
    cmd;\
  }

#ifdef WIN32
# define DAGTM_ERROR(reason, gsl_errno) \
       do { \
       gsl_error (reason, __FILE__"("__FUNCTION__")" , __LINE__, gsl_errno); \
       return gsl_errno ; \
       } while (0)

# define DAGTM_ERROR_VAL(reason, gsl_errno, value) \
       do { \
       gsl_error (reason, __FILE__"("__FUNCTION__")" , __LINE__, gsl_errno); \
       return value ; \
       } while (0)
#else
# define DAGTM_ERROR(reason, gsl_errno) \
       do { \
       gsl_error (reason, __FUNCTION__, __LINE__, gsl_errno); \
       return gsl_errno ; \
       } while (0)
# define DAGTM_ERROR_VAL(reason, gsl_errno, value) \
       do { \
       gsl_error (reason, __FUNCTION__, __LINE__, gsl_errno); \
       return value ; \
       } while (0)
#endif

typedef struct {
    size_t N;
    size_t D;
    gsl_matrix *Xcentered;
    gsl_vector *resp;
    gsl_matrix *S1;
    //gsl_matrix* S2;
    gsl_matrix *S11;
    gsl_matrix *S12;
    gsl_matrix *H;
    gsl_eigen_symm_workspace *w;
    gsl_vector *lambda;
} dagtm_ctemp_workspace;

typedef struct {
    gsl_vector *v1;
    gsl_vector *v2;
    gsl_vector *vones1;
    gsl_vector *vones2;
} dagtm_dist_workspace;

void dagtm_init();
void dagtm_setfilelog(int id);
void dagtm_setloglevel(int l);
void dagtm_setmyid(int id);
int dagtm_getloglevel();

double dagtm_bi(const gsl_matrix * Y);
double dagtm_sq(double d);
double dagtm_stats_mean(const gsl_vector * v);
double dagtm_stats_sd(const gsl_vector * v);
double dagtm_stats_sd_m(const gsl_vector * v, const double mean);
int dagtm_matrix_colmean(const gsl_matrix * M, gsl_vector * m);
double dagtm_vector_sum(const gsl_vector * v);
double rint(double x);
double dagtm_matrix_sum(const gsl_matrix * M);
gsl_matrix *dagtm_matrix_alloc(const size_t n1, const size_t n2,
                               const char *filepath);
gsl_vector *dagtm_vector_alloc(const size_t n, const char *filepath);
int dagtm_dist(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * D);
int dagtm_dist_ws(const gsl_matrix * A, const gsl_matrix * B,
                  gsl_matrix * C, dagtm_dist_workspace * ws);
int dagtm_dist_ws_test(const gsl_matrix * A, const gsl_matrix * B,
                  gsl_matrix * C, dagtm_dist_workspace * ws);
//int dagtm_dist2(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C);
int dagtm_file_lines(const char *fname);
int dagtm_gbf(const gsl_matrix * MU, const gsl_matrix * X,
              const double sigma, gsl_matrix * FI);
int dagtm_blas_dgemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB,
                     double alpha, const gsl_matrix * A,
                     const gsl_matrix * B, double beta, gsl_matrix * C);
int dagtm_matrix_apply(gsl_matrix * M, double (*func) (double));
int dagtm_matrix_apply2(gsl_matrix * M, double (*func) (double, double),
                        const double arg2);
int dagtm_matrix_apply_exp(gsl_matrix * M);

int dagtm_matrix_centering(gsl_matrix * X, const gsl_vector * m);
int dagtm_matrix_colapply(const gsl_matrix * M,
                          double (*func) (const gsl_vector *),
                          gsl_vector * col);
int dagtm_matrix_colsum(const gsl_matrix * A, gsl_vector * v);
int dagtm_matrix_dim(const char *filepath, unsigned int *nrow,
                     unsigned int *ncol);
int dagtm_matrix_fileread(const char *filepath, gsl_matrix * M);
int dagtm_matrix_fprintf(FILE * fp, const gsl_matrix * M, const char *fmt,
                         const char *del, const size_t nrow,
                         const size_t ncol);
int dagtm_vector_fprintf(FILE * fp, const gsl_vector * v, const char *fmt,
                         const char *del, const size_t ncol);
int dagtm_vector_matrix_fprintf(FILE * fp, const gsl_vector * v,
                                const gsl_matrix * M, const char *fmt,
                                const char *del);
int dagtm_matrix_rowsum(const gsl_matrix * A, gsl_vector * v);
int dagtm_matrix_scale_f(gsl_matrix * M, const double f);
int dagtm_matrix_scale_by_row(gsl_matrix * M, const gsl_vector * v);
int dagtm_matrix_scale_by_col(gsl_matrix * M, const gsl_vector * v);
int dagtm_matrix_div_by_col(gsl_matrix * M, const gsl_vector * v);
int dagtm_matrix_set_diag(const gsl_vector * v, gsl_matrix * A);
int dagtm_rctg(const size_t xDim, const size_t yDim, const size_t zDim,
               gsl_matrix * X);
int dagtm_resp(gsl_matrix * D, const double beta);
int dagtm_solve(const gsl_matrix * A, const gsl_matrix * B,
                gsl_matrix * X);
int dagtm_solve2(const gsl_matrix * A, const gsl_matrix * B,
                 gsl_matrix * X);
int dagtm_split(const int n, const int nsplit, int *counts, int *offsets);
int dagtm_stp(const int noLatDim, const int noLatVarSmpl,
              const int noBasFn, const double s, const GRID gridtype,
              gsl_matrix * X, gsl_matrix * FI);
int dagtm_stp3_rnd(const gsl_vector * vTmu, const gsl_vector * vTsd,
                   const int noLatDim, const int noLatVarSmpl,
                   const int noBasFn, const double s, const GRID gridtype,
                   gsl_matrix * X, gsl_matrix * FI, gsl_matrix * W,
                   double *beta);
int dagtm_vector_apply(gsl_vector * v, double (*func) (double));
int dagtm_yrnd(const gsl_matrix * T, gsl_matrix * Yrnd);
int dagtm_yrnd2(const gsl_vector * mu, const gsl_vector * sd,
                gsl_matrix * Yrnd);
void dagtm_matrix_summary(gsl_matrix * m);
void dagtm_vector_summary(gsl_vector * v);

typedef struct {
    gsl_vector *vector;
    size_t next;
    size_t init;
} dagtm_collection;

typedef struct {
    gsl_vector *vector;
    size_t next;
} dagtm_queue;

int dagtm_collection_alloc(dagtm_collection * col, const size_t size);
void dagtm_collection_free(dagtm_collection * col);
int dagtm_collection_expand(dagtm_collection * col);
int dagtm_collection_push(dagtm_collection * col, double val);

int dagtm_queue_alloc(dagtm_queue * que, const size_t size,
                      const double init);
int dagtm_queue_free(dagtm_queue * que);
int dagtm_queue_reset(dagtm_queue * que, const double init);
int dagtm_queue_push(dagtm_queue * que, double val);
double dagtm_queue_mean(const dagtm_queue * que);

double dagtm_vector_max(gsl_vector * v, double limit);

dagtm_ctemp_workspace *dagtm_ctemp_workspace_alloc(const size_t dim1,
                                                   const size_t dim2);

void dagtm_ctemp_workspace_free(dagtm_ctemp_workspace * ws);

dagtm_dist_workspace *dagtm_dist_workspace_alloc(const size_t dim1,
                                                 const size_t dim2);

void dagtm_dist_workspace_free(dagtm_dist_workspace * ws);


#ifdef COMM_ALIGN_ON
# define gsl_block_alloc dagtm_gsl_block_alloc
# define gsl_block_calloc dagtm_gsl_block_calloc
# define gsl_block_free dagtm_gsl_block_free

# define gsl_matrix_alloc dagtm_gsl_matrix_alloc
# define gsl_matrix_calloc dagtm_gsl_matrix_calloc
# define gsl_matrix_free dagtm_gsl_matrix_free

# define gsl_vector_alloc dagtm_gsl_vector_alloc
# define gsl_vector_calloc dagtm_gsl_vector_calloc
# define gsl_vector_free dagtm_gsl_vector_free

# define gsl_vector_scale dagtm_gsl_vector_scale
# define gsl_matrix_mul_elements dagtm_gsl_matrix_mul_elements
# define gsl_matrix_memcpy dagtm_gsl_matrix_memcpy
#else
# warning "Using non-optimized GSL ..."
#endif

// Copied from GSL
#define CONCAT2x(a,b) a ## _ ## b
#define CONCAT2(a,b) CONCAT2x(a,b)

#define TYPE(dir) dir
#define FUNCTION(dir,name) CONCAT2(dir,name)
#define MULTIPLICITY 1
#define ATOMIC double
#define RETURN_IF_NULL(x) if (!x) { return ; }

TYPE(gsl_block) * FUNCTION(dagtm_gsl_block, alloc) (const size_t n);

TYPE(gsl_block) * FUNCTION(dagtm_gsl_block, calloc) (const size_t n);

void FUNCTION(dagtm_gsl_block, free) (TYPE(gsl_block) * b);

TYPE(gsl_matrix) *
FUNCTION(dagtm_gsl_matrix, alloc) (const size_t n1, const size_t n2);

TYPE(gsl_matrix) *
FUNCTION(dagtm_gsl_matrix, calloc) (const size_t n1, const size_t n2);

void FUNCTION(dagtm_gsl_matrix, free) (TYPE(gsl_matrix) * m);

TYPE(gsl_vector) * FUNCTION(dagtm_gsl_vector, alloc) (const size_t n);

TYPE(gsl_vector) * FUNCTION(dagtm_gsl_vector, calloc) (const size_t n);

void FUNCTION(dagtm_gsl_vector, free) (TYPE(gsl_vector) * v);

void FUNCTION(dagtm_gsl_matrix, realloc) (TYPE(gsl_matrix) * m, const size_t n1, const size_t n2);

void FUNCTION(dagtm_gsl_vector, realloc) (TYPE(gsl_vector) * v, const size_t n) ;

int dagtm_gsl_vector_scale(gsl_vector * a, const double x);
int dagtm_gsl_matrix_mul_elements(gsl_matrix * a, const gsl_matrix * b);
int dagtm_gsl_matrix_memcpy(gsl_matrix * dest, const gsl_matrix * src);

/*
gsl_block* dagtm_gsl_block_alloc(const size_t n);
gsl_block* dagtm_gsl_block_calloc(const size_t n);
gsl_block* dagtm_gsl_block_free(gsl_block * b);

gsl_block* dagtm_gsl_matrix_alloc(const size_t n1, const size_t n2);
gsl_block* dagtm_gsl_matrix_calloc(const size_t n1, const size_t n2);
gsl_block* dagtm_gsl_matrix_free(gsl_matrix * m);

gsl_block* dagtm_gsl_vector_alloc(const size_t n);
gsl_block* dagtm_gsl_vector_calloc(const size_t n);
gsl_block* dagtm_gsl_vector_free(gsl_block * b);
*/


#endif                          //__DAGTM_COMM_H__

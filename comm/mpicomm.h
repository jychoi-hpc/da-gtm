#ifndef __DAGTM_MPICOMM_H__
#define __DAGTM_MPICOMM_H__
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>

#include "comm.h"
#include "log.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
* Pacheco's cartesian grid setup
*/
typedef struct {
    int np;                     /* Total number of processes    */
    MPI_Comm comm;              /* Communicator for entire grid */
    MPI_Comm row_comm;          /* Communicator for my row      */
    MPI_Comm col_comm;          /* Communicator for my col      */
    MPI_Group first_row_group;  /* Group for my row             */
    MPI_Group first_col_group;  /* Group for my col             */
    MPI_Comm first_row_comm;    /* Communicator for my row      */
    MPI_Comm first_col_comm;    /* Communicator for my col      */
    int nrow;                   /* Order of row grid            */
    int ncol;                   /* Order of col grid            */
    int my_row_coord;           /* My row coordinate number     */
    int my_col_coord;           /* My column coordinate number  */
    int my_rank_in_first_row;   /* My rank in the first row comm */
    int my_rank_in_first_col;   /* My rank in the first col comm */
    int my_rank;                /* My rank in the grid comm     */
} GRID_INFO_T;

//int dagtm_mpi_llh(const int N, const int K, const int D, const double beta, gsl_vector *v, double *llh, const MPI_Comm comm);
int setup_grid(GRID_INFO_T * grid, const int pdim, const int qdim);
int dagtm_mpi_qual(const int N, const int K, const int D,
                   const double beta, gsl_vector * v, const int isDA,
                   const double temp, double *qual, const MPI_Comm comm);
int dagtm_mpi_beta(const gsl_matrix * mYsub, const gsl_matrix * mTsub,
                   const gsl_matrix * mRsub, const int N, const int D,
                   gsl_matrix * mDsub, double *beta,
                   dagtm_dist_workspace * ws, const MPI_Comm comm);
int dagtm_mpi_resp(const unsigned int K, const double beta,
                   const gsl_matrix * D, const int isDA, const double temp,
                   gsl_matrix * R, gsl_vector * vgcolsum,
                   gsl_vector * vgcolsum2, const MPI_Comm comm);
int dagtm_mpi_resp_gg(const gsl_matrix * R, const int nsplit,
                      const int *subsetCount, gsl_vector * vgrowsum,
                      const MPI_Comm row_comm,
                      const MPI_Comm first_row_comm);
int dagtm_mpi_RT(const gsl_matrix * mRTsub, const int nsplit,
                 const int *subsetCount, const int multiplier,
                 void *recvbuf, const MPI_Comm row_comm,
                 const MPI_Comm first_col_comm);
int dagtm_mpi_scatterv(void *sendbuf, const int nsplit,
                       const int *subsetCount, const int multiplier,
                       void *recvbuf, const MPI_Comm comm);
int dagtm_mpi_scatterv_init(gsl_matrix * M, int nsplit, int *subsetCount,
                            int multiplier, void *recvbuf,
                            const MPI_Comm first_comm, const int grank,
                            const MPI_Comm second_comm);
int dagtm_mpi_ctemp(const gsl_matrix * mTsub, const unsigned int N,
                    double *ctemp, const MPI_Comm comm);

int dagtm_mpi_next_ctemp(const gsl_matrix * mTsub, 
                         const gsl_matrix * mYsub,
                         const gsl_matrix * mRsub,
                         const gsl_vector * vggsub,
                         const double temp,
                         const double beta,
                         const double eps,
                         dagtm_ctemp_workspace *ws, /* scratch */
                         gsl_vector * vctemp, /* scratch */
                         double * ctemp,
                         const MPI_Comm row_comm,
                         const MPI_Comm col_comm);

int dagtm_mpi_mean(const gsl_matrix * X, const unsigned int N,
                   gsl_vector * m, const MPI_Comm comm);
int dagtm_mpi_sd(const gsl_matrix * X, const unsigned int N,
                 const gsl_vector * m, gsl_vector * sd,
                 const MPI_Comm comm);

#endif                          // __DAGTM_MPICOMM_H__

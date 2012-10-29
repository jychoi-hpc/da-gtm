#include "mpicomm.h"
#include <assert.h>

int setup_grid(GRID_INFO_T * grid, const int nrow, const int ncol)
{
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];
    MPI_Group MPI_GROUP_WORLD;
    int *process_ranks;
    int i;

    /* Set up Global Grid Information */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->np));

    if (grid->np != nrow * ncol)
        return 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    /* We assume m-by-n grid */
    grid->nrow = nrow;
    grid->ncol = ncol;
    dimensions[0] = nrow;
    dimensions[1] = ncol;

    /* We want a circular shift in second dimension. */
    /* Don't care about first                        */
    wrap_around[0] = wrap_around[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1,
                    &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row_coord = coordinates[0];
    grid->my_col_coord = coordinates[1];

    /* Set up row communicators */
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords, &(grid->row_comm));

    /* Set up column communicators */
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords, &(grid->col_comm));

    /* Set up first row communicator */
    process_ranks = (int *) malloc(ncol * sizeof(int));
    for (i = 0; i < ncol; i++)
        process_ranks[i] = i;

    MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
    MPI_Group_incl(MPI_GROUP_WORLD, ncol, process_ranks,
                   &(grid->first_row_group));
    MPI_Comm_create(MPI_COMM_WORLD, grid->first_row_group,
                    &(grid->first_row_comm));

    if (grid->first_row_comm != MPI_COMM_NULL)
        MPI_Comm_rank(grid->first_row_comm, &(grid->my_rank_in_first_row));
    free(process_ranks);

    /* Set up first col communicator */
    process_ranks = (int *) malloc(nrow * sizeof(int));
    for (i = 0; i < nrow; i++)
        process_ranks[i] = i * ncol;

    //MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
    MPI_Group_incl(MPI_GROUP_WORLD, nrow, process_ranks,
                   &(grid->first_col_group));
    MPI_Comm_create(MPI_COMM_WORLD, grid->first_col_group,
                    &(grid->first_col_comm));
    if (grid->first_col_comm != MPI_COMM_NULL)
        MPI_Comm_rank(grid->first_col_comm, &(grid->my_rank_in_first_col));

    free(process_ranks);

    return GSL_SUCCESS;
}                               /* Setup_grid */

int dagtm_mpi_scatterv(void *sendbuf, const int nsplit,
                       const int *subsetCount, const int multiplier,
                       void *recvbuf, const MPI_Comm comm)
{
    int i;
    int sum = 0;
    int myrank;
    int *displs = (int *) malloc(nsplit * sizeof(int));
    int *scounts = (int *) malloc(nsplit * sizeof(int));

    if (comm == MPI_COMM_NULL)
        return 1;
    if (displs == NULL)
        DAGTM_ERROR("Memory allocation failed", GSL_ENOMEM);
    if (scounts == NULL)
        DAGTM_ERROR("Memory allocation failed", GSL_ENOMEM);

    MPI_Comm_rank(comm, &myrank);

    for (i = 0; i < nsplit; i++)
    {
        displs[i] = sum;
        scounts[i] = subsetCount[i] * multiplier;
        sum += scounts[i];
    }

    MPI_Scatterv(sendbuf, scounts, displs, MPI_DOUBLE, recvbuf,
                 subsetCount[myrank] * multiplier, MPI_DOUBLE, 0, comm);

    free(displs);
    free(scounts);

    return GSL_SUCCESS;
}

int dagtm_mpi_scatterv_init(gsl_matrix * M, int nsplit, int *subsetCount,
                            int multiplier, void *recvbuf,
                            const MPI_Comm first_comm, const int grank,
                            const MPI_Comm second_comm)
{
    if (first_comm != MPI_COMM_NULL)
    {
        int myrank_in_first_comm;
        MPI_Comm_rank(first_comm, &myrank_in_first_comm);

        if (myrank_in_first_comm == 0)
        {
            dagtm_mpi_scatterv(M->data, nsplit, subsetCount, multiplier,
                               recvbuf, first_comm);
        }
        else
        {
            dagtm_mpi_scatterv(NULL, nsplit, subsetCount, multiplier,
                               recvbuf, first_comm);
        }
    }

    if (second_comm != MPI_COMM_NULL)
    {
        MPI_Bcast(recvbuf, subsetCount[grank] * multiplier, MPI_DOUBLE, 0,
                  second_comm);
    }

    return GSL_SUCCESS;
}

int dagtm_mpi_RT(const gsl_matrix * mRTsub, const int nsplit,
                 const int *subsetCount, const int multiplier,
                 void *recvbuf, const MPI_Comm row_comm,
                 const MPI_Comm first_col_comm)
{
    gsl_matrix *mRTsubsum = gsl_matrix_alloc(mRTsub->size1, mRTsub->size2);

    MPI_Reduce(mRTsub->data, mRTsubsum->data,
               (int) (mRTsub->size1 * mRTsub->tda), MPI_DOUBLE, MPI_SUM,
               0, row_comm);

    if (first_col_comm != MPI_COMM_NULL)
    {

        int rank;
        MPI_Comm_rank(first_col_comm, &rank);

        if (rank == 0)
        {
            int i, sum = 0;
            int *displs = (int *) malloc(nsplit * sizeof(int));
            int *scounts = (int *) malloc(nsplit * sizeof(int));

            if (displs == NULL)
                DAGTM_ERROR("Memory allocation failed", GSL_ENOMEM);
            if (scounts == NULL)
                DAGTM_ERROR("Memory allocation failed", GSL_ENOMEM);

            for (i = 0; i < nsplit; i++)
            {
                displs[i] = sum;
                scounts[i] = subsetCount[i] * multiplier;
                sum += scounts[i];
            }

            MPI_Gatherv(mRTsubsum->data,
                        (int) (mRTsubsum->size1 * mRTsubsum->tda),
                        MPI_DOUBLE, recvbuf, scounts, displs, MPI_DOUBLE,
                        0, first_col_comm);

            free(displs);
            free(scounts);
        }
        else
        {
            MPI_Gatherv(mRTsubsum->data,
                        (int) (mRTsubsum->size1 * mRTsubsum->tda),
                        MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                        first_col_comm);
        }

    }

    gsl_matrix_free(mRTsubsum);

    return GSL_SUCCESS;
}

int dagtm_mpi_resp_gg(const gsl_matrix * R, const int nsplit,
                      const int *subsetCount, gsl_vector * vgrowsum,
                      const MPI_Comm row_comm,
                      const MPI_Comm first_col_comm)
{
    gsl_vector *vlsubrowsum = gsl_vector_alloc(R->size1);
    gsl_vector *vgsubrowsum = gsl_vector_alloc(R->size1);

    dagtm_matrix_rowsum(R, vlsubrowsum);
    MPI_Reduce(vlsubrowsum->data, vgsubrowsum->data,
               (int) vlsubrowsum->size, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    if (first_col_comm != MPI_COMM_NULL)
    {

        int rank;
        MPI_Comm_rank(first_col_comm, &rank);

        if (rank == 0)
        {
            int i, sum = 0;
            int *displs = (int *) malloc(nsplit * sizeof(int));
            int *scounts = (int *) malloc(nsplit * sizeof(int));

            if (displs == NULL)
                DAGTM_ERROR("Memory allocation failed", GSL_ENOMEM);
            if (scounts == NULL)
                DAGTM_ERROR("Memory allocation failed", GSL_ENOMEM);

            for (i = 0; i < nsplit; i++)
            {
                displs[i] = sum;
                scounts[i] = subsetCount[i];
                sum += scounts[i];
            }

            MPI_Gatherv(vgsubrowsum->data, (int) vgsubrowsum->size,
                        MPI_DOUBLE, vgrowsum->data, scounts, displs,
                        MPI_DOUBLE, 0, first_col_comm);

            free(displs);
            free(scounts);
        }
        else
        {
            MPI_Gatherv(vgsubrowsum->data, (int) vgsubrowsum->size,
                        MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                        first_col_comm);
        }

    }

    gsl_vector_free(vlsubrowsum);
    gsl_vector_free(vgsubrowsum);

    return GSL_SUCCESS;
}

/**
 *  Compute beta
 *  @param[in] mYsub
 *  @param[in] mTsub
 *  @param[in] mRsub
 *  @param[in] N
 *  @param[in] D
 *  @param[out] mDsub
 *  @param[out] beta
 *  @param[in] ws workspace 
 *  @param[in] comm
 */
int dagtm_mpi_beta(const gsl_matrix * mYsub, const gsl_matrix * mTsub,
                   const gsl_matrix * mRsub, const int N, const int D,
                   gsl_matrix * mDsub, double *beta,
                   dagtm_dist_workspace * ws, const MPI_Comm comm)
{
    double sum;

    LOG_TIC(LOG_BETA_DIST);
    dagtm_dist_ws(mYsub, mTsub, mDsub, ws);
    LOG_TOC(LOG_BETA_DIST);

    LOG_TIC(LOG_BETA_MULE);
    gsl_matrix_mul_elements((gsl_matrix *)mRsub, mDsub);
    LOG_TOC(LOG_BETA_MULE);

    LOG_TIC(LOG_BETA_MSUM);
    sum = dagtm_matrix_sum(mRsub);
    LOG_TOC(LOG_BETA_MSUM);

    MPI_TICTOC(LOG_BETA_COMM,
               MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
                             comm));

    *beta = (N * D) / sum;

    return GSL_SUCCESS;
}

/**
 *  Compute responsibility
 *  @param[in] K
 *  @param[in] beta
 *  @param[in] D
 *  @param[in] isDA
 *  @param[in] temp
 *  @param[out] R
 *  @param[out] vgcolsum
 *  @param[out] vgcolsum2
 *  @param[in] comm
 */
int dagtm_mpi_resp(const unsigned int K, const double beta,
                   const gsl_matrix * D, const int isDA, const double temp,
                   gsl_matrix * R, gsl_vector * vgcolsum,
                   gsl_vector * vgcolsum2, const MPI_Comm comm)
{
    size_t j;
    int result;

    LOG_TIC(LOG_RESP_MCPY);
    gsl_matrix_memcpy(R, D);
    LOG_TOC(LOG_RESP_MCPY);

    LOG_TIC(LOG_RESP_SCLE);
    gsl_matrix_scale(R, -beta / 2.0);
    LOG_TOC(LOG_RESP_SCLE);

    LOG_TIC(LOG_RESP_EXPR);
    //dagtm_matrix_apply(R, &exp);
    dagtm_matrix_apply_exp(R);
    LOG_TOC(LOG_RESP_EXPR);

    LOG_TIC(LOG_RESP_CSUM);
    dagtm_matrix_colsum(R, vgcolsum);
    LOG_TOC(LOG_RESP_CSUM);

    MPI_TICTOC(LOG_RESP_COMM, result =
               MPI_Allreduce(MPI_IN_PLACE, vgcolsum->data,
                             (int) vgcolsum->size, MPI_DOUBLE, MPI_SUM,
                             comm));

    gsl_vector *v = vgcolsum;

    if (isDA)
    {
        dagtm_matrix_apply2(R, &pow, 1.0 / temp);
        dagtm_matrix_colsum(R, vgcolsum2);
        result =
            MPI_Allreduce(MPI_IN_PLACE, vgcolsum2->data,
                          (int) vgcolsum2->size, MPI_DOUBLE, MPI_SUM,
                          comm);
        v = vgcolsum2;
    }

    /*
       // 2.580 sec
       // Multiflying is faster than div.
       // But, this is unsafe. It looks like 1.0 / xx can be easily blown up.
       assert(v->stride == 1);
       for (size_t j = 0; j < v->size; j++) 
       {
       if (v->data[j] == 0.0)
       {
       v->data[j] = 1.0 / (double) K;
       gsl_vector_view col = gsl_matrix_column(R, j);
       gsl_vector_set_all(&col.vector, 1.0);
       }
       else
       {
       v->data[j] = 1.0 / v->data[j];
       }
       }
       dagtm_matrix_scale_by_col(R, v);
     */

    // 2.666
    LOG_TIC(LOG_RESP_CDIV);
    for (size_t j = 0; j < v->size; j++)
    {
        if (v->data[j] == 0.0)
        {
            v->data[j] = (double) K;
            gsl_vector_view col = gsl_matrix_column(R, j);
            gsl_vector_set_all(&col.vector, 1.0);
        }
    }
    dagtm_matrix_div_by_col(R, v);
    LOG_TOC(LOG_RESP_CDIV);

    /*
       // 2.783
       // Normalize R to make colum sum be 1
       for (j = 0; j < v->size; j++)
       {
       gsl_vector_view col = gsl_matrix_column(R, j);
       double sum = gsl_vector_get(v, j);

       if (sum == 0)
       {
       gsl_vector_set_all(&col.vector, 1.0 / (double) K);
       }
       else
       {
       gsl_vector_scale(&col.vector, 1.0 / sum);
       }
       }
     */

    return GSL_SUCCESS;
}

int dagtm_mpi_llh(const int N, const int K, const int D, const double beta,
                  gsl_vector * v, double *llh, const MPI_Comm comm)
{

    double gsum = 0, lsum = 0;
    int rank;

    *llh = 0;

    if (comm != MPI_COMM_NULL)
    {
        MPI_Comm_rank(comm, &rank);

        dagtm_vector_apply(v, &log);
        lsum = dagtm_vector_sum(v);

        MPI_TICTOC(LOG_LGLH_COMM,
                   MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM,
                                 comm));
        //MPI_Reduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

        //if (rank == 0)
        //{
        *llh =
            gsum + N * ((D / 2.0) * log(beta / (2.0 * M_PI)) -
                        log((double) K));
        //}
    }

    return GSL_SUCCESS;
}

/**
 *  Compute qualities, log-likelihood or negative free energy
 *  log-likelihood will be maximized (GTM) and free energy be minimized (DA-GTM)
 *
 *  @param[in] N
 *  @param[in] K
 *  @param[in] D
 *  @param[in] beta
 *  @param[in] v
 *  @param[in] isDA
 *  @param[in] temp
 *  @param[out] qual
 *  @param[in] comm
 *  @return 
 */
int dagtm_mpi_qual(const int N, const int K, const int D,
                   const double beta, gsl_vector * v, const int isDA,
                   const double temp, double *qual, const MPI_Comm comm)
{

    double sum = 0;
    int rank;

    if (comm != MPI_COMM_NULL)
    {
        MPI_Comm_rank(comm, &rank);

        dagtm_vector_apply(v, &log);
        sum = dagtm_vector_sum(v);

        MPI_TICTOC(LOG_LGLH_COMM,
                   MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE,
                                 MPI_SUM, comm));
        //MPI_Reduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

        *qual =
            N * ((D / 2.0) * log(beta / (2.0 * M_PI)) - log((double) K));

        if (isDA)
        {
            *qual += temp * sum;
        }
        else
        {
            *qual += sum;
        }

        // Normalized Quality 
        *qual = (*qual) / (double) N;
    }

    return GSL_SUCCESS;
}

/**
 *  Compute 1st critical temperature
 *
 *  @param[in] mTsub
 *  @param[in] N
 *  @param[out] ctemp
 *  @param[in] comm
 *  @return 
 */
int dagtm_mpi_ctemp(const gsl_matrix * mTsub, const unsigned int N,
                    double *ctemp, const MPI_Comm comm)
{
    size_t D = mTsub->size2;
    gsl_vector *vlenD = gsl_vector_alloc(D);
    gsl_vector *vcenter = gsl_vector_alloc(D);
    gsl_vector *vones = gsl_vector_alloc(mTsub->size1);
    gsl_matrix *mCentered = gsl_matrix_calloc(mTsub->size1, D);
    gsl_matrix *mS = gsl_matrix_alloc(D, D);
    int result;

    gsl_vector_set_all(vones, 1.0);

    dagtm_matrix_colsum(mTsub, vcenter);

    result =
        MPI_Allreduce(MPI_IN_PLACE, vcenter->data, (int) vlenD->size,
                      MPI_DOUBLE, MPI_SUM, comm);
    gsl_vector_scale(vcenter, 1.0 / N);

    gsl_blas_dger(-1.0, vones, vcenter, mCentered);
    gsl_matrix_add(mCentered, mTsub);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mCentered, mCentered,
                   0.0, mS);
    result =
        MPI_Allreduce(MPI_IN_PLACE, mS->data, (int) mS->size1 * mS->tda,
                      MPI_DOUBLE, MPI_SUM, comm);

    gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(D);
    gsl_eigen_symm(mS, vlenD, w);
    double lambda_max = gsl_vector_max(vlenD) / N;
    dagtm_matrix_apply2(mCentered, &pow, 2.0);
    double lsum = dagtm_matrix_sum(mCentered);
    double gsum;
    result = MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);

    double beta = N * D / gsum;

    *ctemp = lambda_max * beta;

    gsl_vector_free(vlenD);
    gsl_vector_free(vcenter);
    gsl_vector_free(vones);
    gsl_matrix_free(mCentered);
    gsl_matrix_free(mS);
    gsl_eigen_symm_free(w);

    return GSL_SUCCESS;
}

double dagtm_mpi_next_ctemp_helper(const gsl_matrix * lT, const gsl_vector * lc,
                                   const gsl_vector * lr, const double Tcurrent,
                                   const double beta, const double gg, const double eps,
                                   dagtm_ctemp_workspace * ws, const MPI_Comm comm)
{
    unsigned int Nbar = lT->size1;
    unsigned int D = lT->size2;

    if (lc->size != D)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (lr->size != Nbar)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (gg == 0)
    {
        return HUGE_VAL;
    }

    gsl_matrix_memcpy(ws->Xcentered, lT);
    dagtm_matrix_centering(ws->Xcentered, lc);

    gsl_vector_memcpy(ws->resp, lr);
    dagtm_vector_apply(ws->resp, &sqrt);

    // 2*S1
    dagtm_matrix_scale_by_row(ws->Xcentered, ws->resp);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 2.0, ws->Xcentered,
                   ws->Xcentered, 0.0, ws->S1);
    MPI_Allreduce(MPI_IN_PLACE, ws->S1->data, (ws->S1->size1) * (ws->S1->tda), 
                  MPI_DOUBLE, MPI_SUM, comm);

    // S12 <- -S2
    dagtm_matrix_scale_by_row(ws->Xcentered, ws->resp);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0, ws->Xcentered,
                   ws->Xcentered, 0.0, ws->S12);
    MPI_Allreduce(MPI_IN_PLACE, ws->S12->data, (ws->S12->size1) * (ws->S12->tda), 
                  MPI_DOUBLE, MPI_SUM, comm);

    // S11 <- 2*S1 - S2
    gsl_matrix_memcpy(ws->S11, ws->S1);
    // S1 is already multiplied by 2
    gsl_matrix_add(ws->S11, ws->S12);

    //gsl_matrix* H = gsl_matrix_alloc(2*D, 2*D);
    gsl_matrix_view Hsub = gsl_matrix_submatrix(ws->H, 0, 0, D, D);
    gsl_matrix_memcpy(&Hsub.matrix, ws->S11);
    Hsub = gsl_matrix_submatrix(ws->H, D, D, D, D);
    gsl_matrix_memcpy(&Hsub.matrix, ws->S11);
    Hsub = gsl_matrix_submatrix(ws->H, 0, D, D, D);
    gsl_matrix_memcpy(&Hsub.matrix, ws->S12);
    Hsub = gsl_matrix_submatrix(ws->H, D, 0, D, D);
    gsl_matrix_memcpy(&Hsub.matrix, ws->S12);

#define GG_MIN 1E-250
    // When gg < 1E-295, gsl_eigen_symm got hung. Bug?
    if (gg < GG_MIN)
    {
        gsl_matrix_scale(ws->H, 1.0 / GG_MIN);
    }

    gsl_eigen_symm(ws->H, ws->lambda, ws->w);

    if (gg < 1E-250)
    {
        gsl_vector_scale(ws->lambda, GG_MIN);
    }

    double limit = (Tcurrent - eps) * gg * 2 / beta;
    double lambda_max = dagtm_vector_max(ws->lambda, limit);
    double ctemp = beta / 2 / gg * lambda_max;

    return ctemp;
}


/**
 *  Compute 1st critical temperature
 *
 *  @param[in] mTsub
 *  @param[in] N
 *  @param[out] ctemp
 *  @param[in] comm
 *  @return 
 */
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
                         const MPI_Comm col_comm)
{
    if (vctemp->size != mYsub->size1)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    for (size_t k = 0; k < vctemp->size; k++)
    {
        gsl_vector_view yk = gsl_matrix_row((gsl_matrix *)mYsub, k);
        gsl_vector_view lr = gsl_matrix_row((gsl_matrix *)mRsub, k);
        double gg_k = gsl_vector_get(vggsub, k);
        //DUMPV((&yk.vector));
        //DUMP("temp : %f", temp);
        double ctemp_k =
            dagtm_mpi_next_ctemp_helper(mTsub, &yk.vector, &lr.vector, temp,
                                        beta, gg_k, eps, ws, /*grid.*/row_comm);
        //DUMP("ctemp_k[%d] : %f", k, ctemp_k);
        gsl_vector_set(vctemp, k, ctemp_k);
    }

    double ctemp_max = gsl_vector_max(vctemp);
    MPI_Allreduce(MPI_IN_PLACE, &ctemp_max, 1, MPI_DOUBLE,
                  MPI_MAX, /*grid.*/col_comm);

    *ctemp = ctemp_max;

    return GSL_SUCCESS;
}


int dagtm_mpi_mean(const gsl_matrix * X, const unsigned int N,
                   gsl_vector * m, const MPI_Comm comm)
{
    int result;

    if (X->size2 != m->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    dagtm_matrix_colsum(X, m);

    result =
        MPI_Allreduce(MPI_IN_PLACE, m->data, (int) m->size, MPI_DOUBLE,
                      MPI_SUM, comm);
    gsl_vector_scale(m, 1.0 / N);

    return GSL_SUCCESS;
}

int dagtm_mpi_sd(const gsl_matrix * X, const unsigned int N,
                 const gsl_vector * m, gsl_vector * sd,
                 const MPI_Comm comm)
{
    int result;

    if (X->size2 != m->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    if (X->size2 != sd->size)
        DAGTM_ERROR("Dimension mismatch", GSL_EBADLEN);

    gsl_matrix *mCentered = gsl_matrix_alloc(X->size1, X->size2);
    gsl_matrix_memcpy(mCentered, X);

    dagtm_matrix_centering(mCentered, m);
    dagtm_matrix_apply2(mCentered, &pow, 2.0);
    dagtm_matrix_colsum(mCentered, sd);

    result =
        MPI_Allreduce(MPI_IN_PLACE, sd->data, (int) sd->size, MPI_DOUBLE,
                      MPI_SUM, comm);
    gsl_vector_scale(sd, 1.0 / (N - 1));
    dagtm_vector_apply(sd, &sqrt);

    return GSL_SUCCESS;
}

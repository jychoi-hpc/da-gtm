#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

//#include <ctype.h>
//#include <stdlib.h>
#include "getopt.h"
#include "comm.h"
#if defined MPILIB
#include "mpicomm.h"
#endif
#include "log.h"


#define BUFFSIZE 160
//#define NO_INTPMAIN

#define _DAGTM__REV_ "$Rev: 29 $"
#define _DAGTM__DATE_ "$Date: 2010-03-01 21:28:01 -0500 (Mon, 01 Mar 2010) $"

#define LOG_MAIN 0

#ifndef NO_INTPMAIN

char *progname;                 /* program name */

char *basename(char *filename)
{
    char *p = strrchr(filename, '/');
    return p ? p + 1 : (char *) filename;
}

static void usage()
{
    fprintf(stderr, "usage: %s [OPTIONS]\n", progname);
}

static void help()
{
    static char *help_msg[] = {
        "-b filename  : datafile for beta (default: beta)",
        "-f filename  : datafile for FI (default: FI)",
        "-h           : show help",
        "-l           : log to file",
        "-n number    : number of data N (eg, 100, 1k, 1M)",
        "-o filename  : outfile for interpolated maps",
        "-p string    : prefix",
        "-t filename  : datafile for T (default: T)",
        "-v number    : set verbose level",
        "-w filename  : datafile for W (default: W)",
        "-x filename  : datafile for X (default: X)",
        "-z PxQ       : define P-by-Q compute grid",
        0
    };
    char **p = help_msg;

    fprintf(stderr, "%s (%s, %s)\n", progname, _DAGTM__REV_,
            _DAGTM__DATE_);
    usage();
    while (*p)
        fprintf(stderr, "\t%s\n", *p++);
}

int main(int argc, char *argv[])
{
    //int aflag = 0;
    //int bflag = 0;
    //char *cvalue = NULL;
    FILE *fp;

    int L = 3;
    int D = 166;
    int K = 125;
    int N = 100;
    int M = 8;
    int result;

    double beta;
    gsl_matrix *mlT, *mT, *mX, *mFI, *mW, *mY, *mR, *mM;
    gsl_matrix_view vmlT;
    gsl_vector *vbeta, *vlenN, *vlenK, *vl;

    char path[BUFFSIZE];
    char *prefix = "";
    char *filename_mT = "T";
    char *filename_mX = "X";
    char *filename_mFI = "FI";
    char *filename_mW = "W";
    char *filename_vbeta = "beta";
    char *filename_output = NULL;

#if defined MPILIB
    int p;
    GRID_INFO_T grid;
    int MPI_P_DIM = 2;
    int MPI_Q_DIM = 3;
#endif
    char c;
    int blogtofile = FALSE;

    progname = basename(argv[0]);

    opterr = 0;
    while ((c = getopt(argc, argv, "b:f:hln:o:p:t:w:v:x:z:")) != -1)
    {
        switch (c)
        {
        case 'b':
            filename_vbeta = optarg;
            break;
        case 'f':
            filename_mFI = optarg;
            break;
        case 'h':
            help();
            return GSL_SUCCESS;
            break;
        case 'l':
            blogtofile = TRUE;
            break;
        case 'n':
            {
                float num, unit = 1.0;
                char ch;
                sscanf(optarg, "%f%c", &num, &ch);
                switch (ch)
                {
                case 'k':
                case 'K':
                    unit = 1e3;
                    break;
                case 'm':
                case 'M':
                    unit = 1e6;
                    break;
                case 'g':
                case 'G':
                    unit = 1e9;
                    break;
                }

                N = (int) (num * unit);
            }
            break;
        case 'o':
            filename_output = optarg;
            break;
        case 'p':
            prefix = optarg;
            break;
        case 't':
            filename_mT = optarg;
            break;
        case 'w':
            filename_mW = optarg;
            break;
        case 'v':
            DAGTM_SETLOGLEVEL(atoi(optarg));
            break;
        case 'x':
            filename_mX = optarg;
            break;
#if defined MPILIB
        case 'z':
            sscanf(optarg, "%dx%d", &MPI_P_DIM, &MPI_Q_DIM);
            break;
#endif
        case '?':
            if (optopt == 'c')
                fprintf(stderr, "Option -%c requires an argument.\n",
                        optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr,
                        "Unknown option character `\\x%x'.\n", optopt);
            return 1;
        default:
            abort();
        }
    }

#if defined MPILIB
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p > 1)
    {
        //N = dagtm_file_lines(strcat(strcpy(path, prefix), filename_mT));
        int *KsubCount;
        int *NsubCount;

        gsl_matrix *mTsub, *mXsub, *mYsub, *mRsub, *mDsub, *mMsub,
            *mMsubReduced;
        gsl_vector *vlenNsub, *vlenKsub, *vSumReduced;

        double *sendbuf = NULL;
        double *recvbuf = NULL;


        if (setup_grid(&grid, MPI_P_DIM, MPI_Q_DIM))
        {
            DEBUG(DAGTM_CRITIC_MSG, "MPI setup failure");
            return 1;
        }

        DAGTM_SETMYID(grid.my_rank);

        if (blogtofile)
        {
            DAGTM_SETLOGFILE();
        }
        else
        {
            DAGTM_SETLOG(stdout);
        }

        DEBUG(DAGTM_CRITIC_MSG, _DAGTM__REV_);
        DEBUG(DAGTM_CRITIC_MSG, _DAGTM__DATE_);
        DEBUG(DAGTM_CRITIC_MSG, "Starting ... ");
        DEBUG(DAGTM_CRITIC_MSG, "Rank = %d (%d,%d)", grid.my_rank,
              grid.my_rank_in_row, grid.my_rank_in_col);

        NsubCount = (int *) malloc(MPI_Q_DIM * sizeof(int));
        dagtm_split(N, MPI_Q_DIM, NsubCount);

        KsubCount = (int *) malloc(MPI_P_DIM * sizeof(int));
        dagtm_split(K, MPI_P_DIM, KsubCount);

        vlenN = gsl_vector_alloc(N);
        vlenK = gsl_vector_alloc(K);
        vlenNsub = gsl_vector_alloc(NsubCount[grid.my_rank_in_col]);
        vlenKsub = gsl_vector_alloc(KsubCount[grid.my_rank_in_row]);
        vSumReduced = gsl_vector_alloc(NsubCount[grid.my_rank_in_col]);

        DEBUG(DAGTM_INFO_MSG, "mTsub is allocating ... ");
        mTsub = gsl_matrix_calloc(NsubCount[grid.my_rank_in_col], D);
        DEBUG(DAGTM_INFO_MSG, "mXsub is allocating ... ");
        mXsub = gsl_matrix_calloc(KsubCount[grid.my_rank_in_row], L);
        DEBUG(DAGTM_INFO_MSG, "mYsub is allocating ... ");
        mYsub = gsl_matrix_calloc(KsubCount[grid.my_rank_in_row], D);
        DEBUG(DAGTM_INFO_MSG, "mDsub is allocating ... ");
        mDsub = gsl_matrix_calloc(mYsub->size1, mTsub->size1);
        DEBUG(DAGTM_INFO_MSG, "mRsub is allocating ... ");
        mRsub = gsl_matrix_calloc(mYsub->size1, mTsub->size1);
        DEBUG(DAGTM_INFO_MSG, "mMsub is allocating ... ");
        mMsub = gsl_matrix_alloc(mRsub->size2, mXsub->size2);

        vSumReduced = gsl_vector_alloc(mRsub->size2);

        if (grid.my_rank == 0)
        {
            DEBUG(DAGTM_INFO_MSG, "vbeta is allocating ... ");
            //vbeta = dagtm_vector_alloc(1, strcat(strcpy(path, prefix), filename_vbeta));
            vbeta = gsl_vector_alloc(1);

            fp = fopen(strcat(strcpy(path, prefix), filename_vbeta), "r");
            gsl_vector_fscanf(fp, vbeta);
            fclose(fp);

            beta = gsl_vector_get(vbeta, 0);
            //DUMP("%g", beta);
            DEBUG(DAGTM_INFO_MSG, "vbeta is allocating ... Done.");

            DEBUG(DAGTM_INFO_MSG, "mT is allocating ... ");
            mlT =
                dagtm_matrix_alloc(N, D + 1,
                                   strcat(strcpy(path, prefix),
                                          filename_mT));
            vmlT = gsl_matrix_submatrix(mlT, 0, 1, N, D);
            mT = gsl_matrix_alloc(N, D);
            vl = gsl_vector_alloc(N);
            {
                int i, j;
                gsl_vector_view vvl;
                //gsl_matrix_memcpy(mT, &vmlT.matrix);
                for (i = 0; i < vmlT.matrix.size1; i++)
                {
                    for (j = 0; j < vmlT.matrix.size2; j++)
                    {
                        gsl_matrix_set(mT, i, j,
                                       gsl_matrix_get(&vmlT.matrix, i, j));
                    }
                }

                vvl = gsl_matrix_column(mlT, 0);
                gsl_vector_memcpy(vl, &vvl.vector);
                gsl_matrix_free(mlT);
            }

            DEBUG(DAGTM_INFO_MSG, "mX is allocating ... ");
            mX = dagtm_matrix_alloc(K, L,
                                    strcat(strcpy(path, prefix),
                                           filename_mX));

            DEBUG(DAGTM_INFO_MSG, "mFI is allocating ... ");
            mFI =
                dagtm_matrix_alloc(K, M + 1,
                                   strcat(strcpy(path, prefix),
                                          filename_mFI));
            DEBUG(DAGTM_INFO_MSG, "mW is allocating ... ");
            mW = dagtm_matrix_alloc(M + 1, D,
                                    strcat(strcpy(path, prefix),
                                           filename_mW));
            DEBUG(DAGTM_INFO_MSG, "mY is allocating ... ");
            mY = gsl_matrix_alloc(K, D);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, mFI, mW, 0, mY);
        }
        else
        {
            mY = NULL;
        }

        if (grid.my_rank_in_row == 0)
        {
            DEBUG(DAGTM_INFO_MSG, "mMsubReduced is allocating ... ");
            mMsubReduced = gsl_matrix_calloc(mRsub->size2, mXsub->size2);
        }

        //--------------------
        // Real work starts from here
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Starting ... ");
        //loginit(grid.my_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        logtick(LOG_MAIN);


        //--------------------
        // Distribute beta
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Distribute beta ... ");
        MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //--------------------
        // Distribute mT
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Distribute mT ... ");
        dagtm_mpi_scatterv_init(mT, MPI_Q_DIM, NsubCount, D, mTsub->data,
                                grid.first_row_comm, grid.my_rank_in_col,
                                grid.col_comm);
        //dagtm_matrix_rowsum(mTsub, vlenNsub);
        //DUMPV(vlenNsub);

        //--------------------
        // Distribute mX
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Distribute mX ... ");
        dagtm_mpi_scatterv_init(mX, MPI_P_DIM, KsubCount, L, mXsub->data,
                                grid.first_col_comm, grid.my_rank_in_row,
                                grid.row_comm);
        //DUMPM(mXsub);
        //dagtm_matrix_rowsum(mXsub, vlenKsub);
        //DUMPV(vlenKsub);

        //--------------------
        // Distribute mY
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Distribute mY ... ");
        dagtm_mpi_scatterv_init(mY, MPI_P_DIM, KsubCount, D, mYsub->data,
                                grid.first_col_comm, grid.my_rank_in_row,
                                grid.row_comm);
        //dagtm_matrix_rowsum(mYsub, vlenKsub);
        //DUMPV(vlenKsub);

        //--------------------
        // Comptue DIST and RESP
        //--------------------
        MPI_Barrier(grid.comm);
        logtick(LOG_MAIN);

        DEBUG(DAGTM_INFO_MSG, "Compute DIST ... ");
        dagtm_dist(mYsub, mTsub, mDsub);
        //DUMPM(mRsub);

        DEBUG(DAGTM_INFO_MSG, "Compute RESP ... ");
        dagtm_mpi_resp(beta, mDsub, mRsub, vSumReduced, grid.col_comm);

        {
            // llh 
            double gsum = 0, llh = 0, lsum = 0;
            int j;

            DEBUG(DAGTM_INFO_MSG, "Compute LLH ... ");
            dagtm_vector_apply(vSumReduced, &log);
            for (j = 0; j < vSumReduced->size; j++)
            {
                lsum += gsl_vector_get(vSumReduced, j);
            }

            MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM,
                          grid.row_comm);
            llh =
                gsum + N * ((D / 2) * log(beta / (2 * M_PI)) -
                            log((double) K));
            //DUMP("%f", llh);
        }


        //--------------------
        // Comptue MAP
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
        result =
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                           mMsub);
        //DUMPM(mMsub);

        MPI_Barrier(grid.comm);
        logtock(LOG_MAIN);

        if (grid.my_rank_in_row == 0)
        {
            recvbuf = mMsubReduced->data;
        }

        DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
        MPI_Reduce(mMsub->data, recvbuf,
                   (int) mMsub->size1 * (int) mMsub->size2, MPI_DOUBLE,
                   MPI_SUM, 0, grid.col_comm);

        MPI_Barrier(MPI_COMM_WORLD);
        //logtock(LOG1);
        //logticks(stdout);

        //printf(">>> %I64d ticks\n", tock(s));

        //--------------------
        // Write to file
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Writing to file ... ");
        if ((filename_output != NULL) && (grid.my_rank_in_row == 0))
        {
            //DUMPM(mMsubReduced);
            sprintf(path, "%s.%03d", filename_output,
                    grid.my_rank_in_first_row);
            fp = fopen(path, "w");
            dagtm_vector_matrix_fprintf(fp, vl, mMsubReduced, FMTSTR, " ");
            fclose(fp);
        }

        DEBUG(DAGTM_INFO_MSG, "Verifying ... ");
        {
            // Verifying ...
            gsl_vector *vlenNsubReduced = gsl_vector_alloc(vlenNsub->size);
            dagtm_matrix_colsum(mRsub, vlenNsub);
            // Reduce from memebers
            MPI_Allreduce(vlenNsub->data, vlenNsubReduced->data,
                          (int) vlenNsub->size, MPI_DOUBLE, MPI_SUM,
                          grid.col_comm);
            //DUMPV(vlenNsubReduced);
            gsl_vector_free(vlenNsubReduced);
        }

        gsl_vector_free(vlenN);
        gsl_vector_free(vlenK);
        gsl_vector_free(vlenNsub);
        gsl_vector_free(vlenKsub);
        free(NsubCount);
        free(KsubCount);

        gsl_matrix_free(mTsub);
        gsl_matrix_free(mXsub);
        gsl_matrix_free(mYsub);
        gsl_matrix_free(mRsub);
        gsl_matrix_free(mMsub);

        if (grid.my_rank == 0)
        {
            gsl_vector_free(vbeta);
            gsl_matrix_free(mT);
            gsl_matrix_free(mX);
            gsl_matrix_free(mFI);
            gsl_matrix_free(mW);
            gsl_matrix_free(mY);
        }

        if (grid.my_rank_in_row == 0)
        {
            gsl_matrix_free(mMsubReduced);
        }

        //logfprintf(stdout);
        DEBUG(DAGTM_INFO_MSG, "MPI Finalizing ... ");
        MPI_Finalize();
    }
    else
    {
#endif
        /*
         *  Single process
         */
        DAGTM_SETMYID(0);
        if (blogtofile)
        {
            //dagtm_setfilelog(grid.my_rank);
            DAGTM_SETLOGFILE();
        }
        else
        {
            DAGTM_SETLOG(stdout);
        }

        DEBUG(DAGTM_CRITIC_MSG, _DAGTM__REV_);
        DEBUG(DAGTM_CRITIC_MSG, _DAGTM__DATE_);
        DEBUG(DAGTM_CRITIC_MSG, "Starting ... ");

        mlT =
            dagtm_matrix_alloc(N, D + 1,
                               strcat(strcpy(path, prefix), filename_mT));
        vmlT = gsl_matrix_submatrix(mlT, 0, 1, N, D);
        mT = gsl_matrix_alloc(N, D);
        vl = gsl_vector_alloc(N);
        {
            size_t i, j;
            gsl_vector_view vvl;
            //gsl_matrix_memcpy(mT, &vmlT.matrix);
            for (i = 0; i < vmlT.matrix.size1; i++)
            {
                for (j = 0; j < vmlT.matrix.size2; j++)
                {
                    gsl_matrix_set(mT, i, j,
                                   gsl_matrix_get(&vmlT.matrix, i, j));
                }
            }

            vvl = gsl_matrix_column(mlT, 0);
            gsl_vector_memcpy(vl, &vvl.vector);
            gsl_matrix_free(mlT);
        }
        mX = dagtm_matrix_alloc(K, L,
                                strcat(strcpy(path, prefix), filename_mX));
        mFI =
            dagtm_matrix_alloc(K, M + 1,
                               strcat(strcpy(path, prefix), filename_mFI));
        mW = dagtm_matrix_alloc(M + 1, D,
                                strcat(strcpy(path, prefix), filename_mW));
        vbeta =
            dagtm_vector_alloc(1,
                               strcat(strcpy(path, prefix),
                                      filename_vbeta));
        mY = gsl_matrix_alloc(K, D);
        mR = gsl_matrix_alloc(K, N);
        mM = gsl_matrix_alloc(N, L);

        vlenN = gsl_vector_alloc(N);
        vlenK = gsl_vector_alloc(K);
        beta = gsl_vector_get(vbeta, 0);

        //DUMPM(mT);
        //DUMPM(mX);
        //DUMPM(mFI);
        //DUMPM(mW);
        //DUMPV(vbeta);

        //--------------------
        // Real work starts from here
        //--------------------
        logtic(LOG_MAIN);

        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, mFI, mW, 0, mY);

        dagtm_dist(mY, mT, mR);

        dagtm_resp(mR, beta);

        result =
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mR, mX, 0, mM);

        logtoc(LOG_MAIN);

        if (filename_output != NULL)
        {
            fp = fopen(filename_output, "w");
            dagtm_vector_matrix_fprintf(fp, vl, mM, FMTSTR, " ");
            fclose(fp);
        }

        {
            // Verifying ...
            dagtm_matrix_colsum(mR, vlenN);
            //DUMPV(vlenN);
        }

        gsl_vector_free(vlenN);
        gsl_vector_free(vlenK);
        gsl_vector_free(vbeta);
        gsl_matrix_free(mT);
        gsl_matrix_free(mX);
        gsl_matrix_free(mFI);
        gsl_matrix_free(mW);
        gsl_matrix_free(mY);
        gsl_matrix_free(mM);
        gsl_matrix_free(mR);
#if defined MPILIB
    }
#endif

    printf("Done.\n");
    return 0;

}
#endif

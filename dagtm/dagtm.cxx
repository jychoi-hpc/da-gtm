#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h> 

#ifdef WIN32
#else
# include <sys/times.h>
#endif
#ifdef WIN32
# include <WinSock2.h>
#else
# include <unistd.h>
#endif

// HDF5 
#include "hdf5.h"

//#include <ctype.h>
//#include <stdlib.h>
extern "C" {
#ifdef WIN32
# include "getopt.h"
#endif
#include "comm.h"
#include "mpicomm.h"
#include "log.h"
#include "h5comm.h"
}
#define _DAGTM__REV_ "$Rev: 56 $"
#define _DAGTM__DATE_ "$Date: 2010-08-08 17:46:44 -0400 (Sun, 08 Aug 2010) $"
#define BUFFSIZE 160
//#define NO_MAIN
#define RANK 2
#define HID_INVALID -1
#define INIT_COLLECTION_SIZE 1000
#define INIT_QUEUE_SIZE 50
#ifndef HUGE_VAL
typedef union {
    unsigned char __c[8];
    double __d;
} __huge_val_t;
# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define __HUGE_VAL_bytes	{ 0, 0, 0, 0, 0, 0, 0xf0, 0x7f }
# endif
static __huge_val_t __huge_val = { __HUGE_VAL_bytes };

# define HUGE_VAL	(__huge_val.__d)
#endif

#ifndef HUGE_VALF
typedef union {
    unsigned char __c[4];
    float __f;
} __huge_valf_t;
# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define __HUGE_VALF_bytes	{ 0, 0, 0x80, 0x7f }
# endif
static __huge_valf_t __huge_valf = { __HUGE_VALF_bytes };

# define HUGE_VALF	(__huge_valf.__f)
#endif

#define H5FCLOSEANDSET(hid)                     \
    H5Fclose(hid);                              \
    hid=HID_INVALID

#define CHECK(cond)                                                     \
    if (!(cond))                                                        \
    {                                                                   \
        if (h5infileid != HID_INVALID) H5FCLOSEANDSET(h5infileid);      \
        if (h5outfileid != HID_INVALID) H5FCLOSEANDSET(h5outfileid);    \
        fprintf(stderr, "Fatal error : %s\n", #cond );                  \
        MPI_Finalize();                                                 \
        return -1;                                                      \
    }                                                                   \

int debugwait;
#define DEBUGWAIT                               \
    {                                           \
        while (debugwait == 0) sleep(5);        \
    }                                           \

#define MATRIX_REALLOC(m, size1, size2)         \
    gsl_matrix_free(m);                         \
    m = gsl_matrix_calloc(size1, size2)

#define VECTOR_REALLOC(v, size)                 \
    gsl_vector_free(v);                         \
    v = gsl_vector_alloc(size)

#ifndef WIN32
static clock_t st_time;
static clock_t en_time;
static struct tms st_cpu;
static struct tms en_cpu;
static long clk_tck;
#endif

/**
 *  DAGTM Definitions
 *  DAGTM_PVIZRPC_SERVER_ON : Default OFF
 */

#ifdef DAGTM_PVIZRPC_SERVER_ON
# undef DAGTM_PVIZRPC_SERVER_OFF
#else
# define DAGTM_PVIZRPC_SERVER_OFF
#endif

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
        "-a number    : alpha (default: 0.99)",
        "-b name      : dataset name for beta (default: beta)",
        "-B name      : dataset name for label (default: lb)",
        "-c schedule  : cooling schedule [auto|exp|linear] (default : exp)",
        "-e number    : precision",
        "-f dataset   : dataset name for FI (default: FI)",
        "-h           : show help",
        "-i filename  : input HDF5 filename",
        "-j number    : maximum number of iterations",
        "-K number    : number of latent data K (eg, 100, 1k, 1M)",
        "-l           : log to file",
        "-L number    : latent dimension",
        "-m           : EM",
        "-M number    : number of model",
        "-o filename  : output HDF5 filename",
        "-p           : log progress",
        "-P number    : checkpointing per num. loop",
        "-r filename  : restarting",
        "-s number    : seed",
        "-S number    : start temp",
        "-t dataset   : dataset name for T (default: T)",
        "-v number    : set verbose level",
        "-w prefix    : checkpointing file prefix",
        "-W dataset   : dataset name for W (default: W)",
        "-x dataset   : dataset name for X (default: X)",
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

void dagtm_gsl_errorhandler(const char *reason,
                            const char *file, int line, int gsl_errno)
{
    DUMP("(%s:%d) ERROR (%d): %s", file, line, gsl_errno, reason);
}

herr_t h5rw(const int type, const hid_t file_id, const char *filename,
            const hid_t dtype, const int rank, const hsize_t * dsetDims,
            const hsize_t * dsetOffsets, const hsize_t * viewDims,
            const hsize_t * viewOffsets, gsl_matrix * Msub)
{
    herr_t ret = 0;
    hsize_t count[2];
    hsize_t offset[2];
    gsl_matrix_view vMsub;
    double *data;

    if ((viewDims[0] != 0) && (viewOffsets[0] < Msub->size1))
    {
        vMsub =
            gsl_matrix_submatrix(Msub, viewOffsets[0], viewOffsets[1],
                                 viewDims[0], viewDims[1]);
        count[0] = vMsub.matrix.size1;
        count[1] = vMsub.matrix.size2;
        data = vMsub.matrix.data;
    }
    else
    {
        count[0] = viewDims[0];
        count[1] = viewDims[1];
        data = NULL;
    }

    offset[0] = dsetOffsets[0] + viewOffsets[0];
    offset[1] = dsetOffsets[1] + viewOffsets[1];

    hsize_t memdim[2] = { vMsub.matrix.size1, vMsub.matrix.tda };
    ret = dagtm_h5_rw(type, file_id, filename, dtype,
                      rank, dsetDims, count, offset, memdim, data);

    return ret;
}

herr_t h5rw_scalar(const int type, const hid_t file_id, const char *filename,
                   const hid_t dtype, void *data)
{
    herr_t ret = 0;
    hsize_t count[1] = {1};
    hsize_t offset[1] = {0};

    ret = dagtm_h5_rw(type, file_id, filename, dtype,
                      1, count, count, offset, count, data);

    return ret;
}

herr_t h5read(const hid_t file_id, const char *filename, const hid_t dtype,
              const int rank,
              const hsize_t * dsetDims, const hsize_t * dsetOffsets,
              const hsize_t * viewDims, const hsize_t * viewOffsets,
              gsl_matrix * Msub)
{
    return h5rw(TRUE, file_id, filename, dtype,
                rank, dsetDims, dsetOffsets, viewDims, viewOffsets, Msub);
}

herr_t h5save(const hid_t file_id, const char *filename, const hid_t dtype,
              const int rank,
              const hsize_t * dsetDims, const hsize_t * dsetOffsets,
              const hsize_t * viewDims, const hsize_t * viewOffsets,
              gsl_matrix * Msub)
{
    return h5rw(FALSE, file_id, filename, dtype,
                rank, dsetDims, dsetOffsets, viewDims, viewOffsets, Msub);
}

herr_t h5read_scalar(const hid_t file_id, const char *filename,
                     const hid_t dtype, void *data)
{
    return h5rw_scalar(TRUE, file_id, filename, dtype, data); 
}

herr_t h5save_scalar(const hid_t file_id, const char *filename,
                     const hid_t dtype, void *data)
{
    return h5rw_scalar(FALSE, file_id, filename, dtype, data); 
}

hid_t h5open(const char *filename, const MPI_Comm comm)
{
    /* 
     * Set up file access property list with parallel I/O access
     */
    hid_t property;
    herr_t ret;
    hid_t h5file;

    property = H5Pcreate(H5P_FILE_ACCESS);
    assert(property != FAIL);

#ifdef H5COMM_MPIIO_ON
    ret = H5Pset_fapl_mpio(property, comm, MPI_INFO_NULL);
    assert(ret != FAIL);
#endif

    /*
     * Create a new file collectively and release property list identifier.
     */
    h5file = H5Fopen(filename, H5F_ACC_RDONLY, property);

    H5Pclose(property);

    return h5file;
}

hid_t h5create(const char *filename, const MPI_Comm comm)
{
    /* 
     * Set up file access property list with parallel I/O access
     */
    hid_t property;
    herr_t ret;
    hid_t h5file;

    property = H5Pcreate(H5P_FILE_ACCESS);
    assert(property != FAIL);

#ifdef H5COMM_MPIIO_ON
    ret = H5Pset_fapl_mpio(property, comm, MPI_INFO_NULL);
    assert(ret != FAIL);
#endif

    /*
     * Create a new file collectively and release property list identifier.
     */
    h5file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, property);

    H5Pclose(property);

    return h5file;
}

hid_t h5append(const char *filename, const MPI_Comm comm)
{
    /* 
     * Set up file access property list with parallel I/O access
     */
    hid_t property;
    herr_t ret;
    hid_t h5file;

    property = H5Pcreate(H5P_FILE_ACCESS);
    assert(property != FAIL);

#ifdef H5COMM_MPIIO_ON
    ret = H5Pset_fapl_mpio(property, comm, MPI_INFO_NULL);
    assert(ret != FAIL);
#endif

    /*
     * Create a new file collectively and release property list identifier.
     */
    /*
      H5Fcreate(filename, H5F_ACC_EXCL, H5P_DEFAULT, property);

      if (h5file < 0)
      {
      DEBUG(DAGTM_INFO_MSG, "The file already exists. Try to open to append.");
      h5file = H5Fopen(filename, H5F_ACC_RDWR, property);
      }
    */

    h5file = H5Fopen(filename, H5F_ACC_RDWR, property);

    H5Pclose(property);

    return h5file;
}

herr_t h5save_byrow(const hid_t h5file, const char *dsetname_mT,
                    gsl_matrix * mMsub, const hsize_t dsetDim1,
                    const hsize_t dsetDim2, const hsize_t dsetRowOffset,
                    const hsize_t viewRowDim, const hsize_t viewRowOffset)
{
    hsize_t dsetDims[2];
    hsize_t dsetOffsets[2];
    hsize_t viewDims[2];
    hsize_t viewOffsets[2];
    herr_t ret;

    dsetDims[0] = dsetDim1;
    dsetDims[1] = dsetDim2;
    dsetOffsets[0] = dsetRowOffset;
    dsetOffsets[1] = 0;
    viewDims[0] = viewRowDim;
    viewDims[1] = dsetDim2;
    viewOffsets[0] = viewRowOffset;
    viewOffsets[1] = 0;
    ret = h5save(h5file, dsetname_mT, H5T_NATIVE_DOUBLE,
                 RANK, dsetDims, dsetOffsets, viewDims, viewOffsets,
                 mMsub);

    return ret;
}

herr_t h5save_byblock(const hid_t h5file, const char *dsetname_mT,
                      gsl_matrix * mMsub, const hsize_t dsetDim1,
                      const hsize_t dsetDim2, const hsize_t dsetRowOffset1,
                      const hsize_t dsetRowOffset2)
{
    hsize_t dsetDims[2];
    hsize_t dsetOffsets[2];
    hsize_t viewDims[2];
    hsize_t viewOffsets[2];
    herr_t ret;

    dsetDims[0] = dsetDim1;
    dsetDims[1] = dsetDim2;
    dsetOffsets[0] = dsetRowOffset1;
    dsetOffsets[1] = dsetRowOffset2;
    viewDims[0] = mMsub->size1;
    viewDims[1] = mMsub->size2;
    viewOffsets[0] = 0;
    viewOffsets[1] = 0;
    ret = h5save(h5file, dsetname_mT, H5T_NATIVE_DOUBLE,
                 RANK, dsetDims, dsetOffsets, viewDims, viewOffsets,
                 mMsub);

    return ret;
}

herr_t h5read_byrow(const hid_t h5file, const char *dsetname_mT,
                    gsl_matrix * mMsub, const hsize_t dsetDim1,
                    const hsize_t dsetDim2, const hsize_t dsetRowOffset,
                    const hsize_t viewRowDim, const hsize_t viewRowOffset)
{
    hsize_t dsetDims[2];
    hsize_t dsetOffsets[2];
    hsize_t viewDims[2];
    hsize_t viewOffsets[2];
    herr_t ret;

    dsetDims[0] = dsetDim1;
    dsetDims[1] = dsetDim2;
    dsetOffsets[0] = dsetRowOffset;
    dsetOffsets[1] = 0;
    viewDims[0] = viewRowDim;
    viewDims[1] = dsetDim2;
    viewOffsets[0] = viewRowOffset;
    viewOffsets[1] = 0;
    ret = h5read(h5file, dsetname_mT, H5T_NATIVE_DOUBLE,
                 RANK, dsetDims, dsetOffsets, viewDims, viewOffsets,
                 mMsub);

    return ret;
}

enum COOLING { AUTO, EXP, LINEAR };

#ifdef DAGTM_PVIZRPC_SERVER_ON
#include "pvizrpc.pb.h"

#include "RpcControllerImp.h"
#include "RpcServer.h"

class PvizRpcServiceImp:public cgl::pviz::rpc::PvizRpcService {
public:
    PvizRpcServiceImp(gsl_vector_int * id,
                      gsl_vector_int * lb,
                      gsl_matrix * Mpos,
                      gsl_matrix * Xpos):id_(id), lb_(lb), Mpos_(Mpos),
                                         Xpos_(Xpos) {
    };

    void getIds(::google::protobuf::RpcController * controller,
                const::cgl::pviz::rpc::VoidMessage *
                request,::cgl::pviz::rpc::ListInt *
                response,::google::protobuf::Closure * done) {
        DEBUG(DAGTM_INFO_MSG, "getIds ... ");
        response->clear_val();
        if (id_ != NULL)
        {
            for (size_t i = 0; i < id_->size; i++)
            {
                response->add_val(gsl_vector_int_get(id_, i));
            }
        }
        done->Run();
    };

    void getLabels(::google::protobuf::RpcController * controller,
                   const::cgl::pviz::rpc::VoidMessage *
                   request,::cgl::pviz::rpc::ListInt *
                   response,::google::protobuf::Closure * done) {
        DEBUG(DAGTM_INFO_MSG, "getLabels ... ");
        response->clear_val();
        if (lb_ != NULL)
        {
            for (size_t i = 0; i < lb_->size; i++)
            {
                response->add_val(gsl_vector_int_get(lb_, i));
            }
        }
        done->Run();
    };

    void getPositions(::google::protobuf::RpcController * controller,
                      const::cgl::pviz::rpc::VoidMessage *
                      request,::cgl::pviz::rpc::ListPosition *
                      response,::google::protobuf::Closure * done) {
        DEBUG(DAGTM_INFO_MSG, "getPositions ... ");
        response->clear_pos();
        if (Mpos_ != NULL)
        {
            for (size_t i = 0; i < Mpos_->size1; i++)
            {
                cgl::pviz::rpc::Position * pos = response->add_pos();
                gsl_vector_view v = gsl_matrix_row(Mpos_, i);
                for (size_t j = 0; j < Mpos_->size2; j++)
                {
                    pos->set_x(v.vector.data[0]);
                    pos->set_y(v.vector.data[1]);
                    pos->set_z(v.vector.data[2]);
                }
            }
        }
        done->Run();
    };

    void getLatentPositions(::google::protobuf::RpcController * controller,
                            const::cgl::pviz::rpc::VoidMessage *
                            request,::cgl::pviz::rpc::ListPosition *
                            response,::google::protobuf::Closure * done) {
        DEBUG(DAGTM_INFO_MSG, "getLatentPositions ... ");
        response->clear_pos();
        if (Xpos_ != NULL)
        {
            for (size_t i = 0; i < Xpos_->size1; i++)
            {
                cgl::pviz::rpc::Position * pos = response->add_pos();
                gsl_vector_view v = gsl_matrix_row(Xpos_, i);
                for (size_t j = 0; j < Xpos_->size2; j++)
                {
                    pos->set_x(v.vector.data[0]);
                    pos->set_y(v.vector.data[1]);
                    pos->set_z(v.vector.data[2]);
                }
            }
        }
        done->Run();
    };

    void Pause(::google::protobuf::RpcController * controller,
               const::cgl::pviz::rpc::VoidMessage *
               request,::cgl::pviz::rpc::VoidMessage *
               response,::google::protobuf::Closure * done) {
        DEBUG(DAGTM_INFO_MSG, "Pause ... ");
        done->Run();
    };

private:
    gsl_vector_int * id_;
    gsl_vector_int *lb_;
    gsl_matrix *Mpos_;
    gsl_matrix *Xpos_;
};

#endif

#ifdef DAGTM_NB_SERVER_ON
#include "ServiceClient.h"
#include "Profile.h"
#endif

#ifdef DAGTM_ACTIVEMQ_SERVER_ON
#include <decaf/lang/Thread.h>
#include <decaf/lang/Runnable.h>
#include <decaf/util/concurrent/CountDownLatch.h>
#include <decaf/lang/Long.h>
#include <decaf/util/Date.h>
#include <activemq/core/ActiveMQConnectionFactory.h>
#include <activemq/util/Config.h>
#include <activemq/library/ActiveMQCPP.h>
#include <cms/Connection.h>
#include <cms/Session.h>
#include <cms/TextMessage.h>
#include <cms/BytesMessage.h>
#include <cms/MapMessage.h>
#include <cms/ExceptionListener.h>
#include <cms/MessageListener.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <memory>

using namespace activemq;
using namespace activemq::core;
using namespace decaf;
using namespace decaf::lang;
using namespace decaf::util;
using namespace decaf::util::concurrent;
using namespace cms;
using namespace std;

#include "message.pb.h"
#endif

unsigned int L = 3;
unsigned int D = 166;
unsigned int K = 125;
unsigned int Kmax = 32768; // 32^3
//unsigned int Kmax = 262144; // 64^3
unsigned int N = 100;
unsigned int M = 8;
GRID gridType = GRID_SWEEP;
int result;

double beta;
double diff = DBL_MAX;
double eps = 1e-6;
double eps2 = 1e-3;
unsigned int nloop = 0;
unsigned int lloop = 0;
unsigned int maxloop = UINT_MAX;

// Cooling Schedule
double alpha = 0.99;
double delta = 0.1;
enum COOLING schedule = EXP;

gsl_matrix *mX = NULL, *mFI = NULL, *mW, *mA, *mB;
gsl_vector *vlenN, *vlenK;

char *filename_h5input = "INPUT.h5";
char *filename_h5output = NULL;
char *filename_h5checkpoint = NULL;
char *filename_h5ckprefix = "ckp.";
char *dsetname_mT = "T.dat";
char *dsetname_mX = "X.dat";
char *dsetname_mFI = "FI.dat";
char *dsetname_mW = "W.dat";
char *dsetname_mY = "Y.dat";
char *dsetname_mR = "R.dat";
char *dsetname_vbeta = "beta.dat";
char *dsetname_mM = "M.dat";
char *dsetname_vid = "id.dat";
char *dsetname_vlb = "lb.dat";
char *dsetname_vgg = "gg.dat";
char *dsetname_temp = "temp";
char *dsetname_iter = "iter";
char *dsetname_seed = "seed";

GRID_INFO_T grid;
int MPI_P_DIM = 0;
int MPI_Q_DIM = 0;
int MPI_NNODE = 0;          // P-by-Q (=R) grid

char c;
//int isRandomSetup = FALSE;
int isLogToFile = FALSE;
int checkpointingPerNloop = 100;
int isCheckpointing = TRUE;
int isForceCheckpointing = FALSE;
int doInterpolation = FALSE;

int Kbar;
int Nbar;

gsl_matrix *mTsub = NULL, *mXsub = NULL, *mFIsub = NULL;
gsl_matrix *mYsub = NULL, *mDsub = NULL, *mRsub = NULL;
gsl_matrix *mRTsub = NULL, *mMsub = NULL;   //, *mMsubReduced = NULL;
gsl_matrix *mFIRsub, *mFIRTsub;
gsl_vector_int *vidsub = NULL, *vlbsub = NULL;
gsl_vector *vlenNsub = NULL, *vlenKsub = NULL, *vgc = NULL, *vgc2 =
    NULL;

//gsl_matrix *mgRTsub;
gsl_matrix *mGFIsub;
gsl_vector *vggsub;

// Variables for DA
int isDA = TRUE;
int isHDA = FALSE;
double temp = 1.0, startTemp = 1.0, firstCTemp = 1.0;
double prevExpandedTemp;
double llh = 0, nfree = 0, qual = 0, qual_old = 0;
int doNeedFirstCTemp = TRUE;
int isLast = FALSE;
//char *strLLH = "L";
//char *strFREE = "-F";
//char *strQual = strLLH;
int doLogProgress = FALSE;
int doSaveR = FALSE;

// Variables for Adaptive Cooling Schedule (Auto) 
dagtm_ctemp_workspace *ws_ctemp = NULL;
dagtm_dist_workspace *ws_dist = NULL;

// Variables for assert
int ret;

// HDF5
hid_t h5infileid = HID_INVALID;
hid_t h5outfileid = HID_INVALID;
hid_t h5ckpfileid = HID_INVALID;

// Vec
dagtm_collection TempCollection;
dagtm_collection LLHCollection;
dagtm_collection FreeCollection;

// 
dagtm_queue QualQueue;

int *KsplitbyPcounts;
int *KsplitbyPoffsets;

int *NsplitbyQcounts;
int *NsplitbyQoffsets;

int *NbarSplitbyPcounts;
int *NbarSplitbyPoffsets;

int *KbarSplitbyQcounts;
int *KbarSplitbyQoffsets;

int *MsplitbyRcounts;
int *MsplitbyRoffsets;

/* handler prototype for SIGCHLD */
void sig_handler(int);

/*
 * Memory requirement
 * (2 K + 2 D + M + L + 4) N + 50 + 4 M D + 2 K M + 3 K D + 3 K + K L + 2 M^2
 *
 */
int main(int argc, char *argv[])
{
    progname = basename(argv[0]);
    
    opterr = 0;
    while ((c =
            getopt(argc, argv,
                   "a:b:B:c:e:E:f:g:hHi:Ij:k:K:lL:mM:no:pP:r:Rs:S:t:v:w:W:x:z:")) != -1)
    {
        //printf("char : %c\n", c);
        switch (c)
        {
        case 'a':
            alpha = atof(optarg);
            break;
        case 'b':
            dsetname_vbeta = optarg;
            break;
        case 'B':
            dsetname_vlb = optarg;
            break;
        case 'c':
            if (strcasecmp(optarg, "auto") == 0)
            {
                schedule = AUTO;
            }
            else if (strcasecmp(optarg, "exp") == 0)
            {
                schedule = EXP;
            }
            else if (strcasecmp(optarg, "linear") == 0)
            {
                schedule = LINEAR;
            }
            else
            {
                fprintf(stderr,
                        "Unknown schedule option : %s", optarg);
                return 1;
            }
            break;
        case 'e':
            eps = atof(optarg);
            break;
        case 'E':
            eps2 = atof(optarg);
            break;
        case 'f':
            dsetname_mFI = optarg;
            break;
        case 'g':
            if (strcasecmp(optarg, "sweep") == 0)
            {
                gridType = GRID_SWEEP;
            }
            else if (strcasecmp(optarg, "spacefilling") == 0)
            {
                gridType = GRID_SPACEFILLING;
            }
            else
            {
                fprintf(stderr,
                        "Unknown grid type : %s", optarg);
                return 1;
            }
            break;
        case 'h':
            help();
            return 0;
            break;
        case 'H':
            isHDA = !isHDA;
            break;
        case 'i':
            filename_h5input = optarg;
            break;
        case 'I':
            doInterpolation = TRUE;
            break;
        case 'j':
            maxloop = atoi(optarg);
            break;
        case 'k':
            Kmax = atoi(optarg);
            break;
        case 'K':
        {
            float num, unit = 1.0;
            char ch;
            sscanf(optarg, "%f%c", &num, &ch);
            switch (ch)
            {
            case 'k':
            case 'K':
                unit = 1024;
                break;
            case 'm':
            case 'M':
                unit = 1024 * 1024;
                break;
            case 'g':
            case 'G':
                unit = 1024 * 1024 * 1024;
                break;
            }

            K = (int) (num * unit);
        }
        break;
        case 'l':
            isLogToFile = !isLogToFile;
            break;
        case 'L':
            L = atoi(optarg);
            break;
        case 'm':
            isDA = !isDA;
            break;
        case 'M':
            M = atoi(optarg);
            break;
        case 'n':
            isCheckpointing = FALSE;
            break;
        case 'o':
            filename_h5output = optarg;
            break;
        case 'p':
            doLogProgress = !doLogProgress;
            break;
        case 'P':
            checkpointingPerNloop = atoi(optarg);
            break;
        case 'r':
            filename_h5checkpoint = optarg;
            break;
        case 'R':
            doSaveR = !doSaveR;
            break;
        case 's':
            dagtm_seed = atoi(optarg);
            break;
        case 'S':
            startTemp = atof(optarg);
            doNeedFirstCTemp = FALSE;
            break;
        case 't':
            dsetname_mT = optarg;
            break;
        case 'v':
            DAGTM_SETLOGLEVEL(atoi(optarg));
            break;
        case 'w':
            filename_h5ckprefix = optarg;
            break;
        case 'W':
            dsetname_mW = optarg;
            break;
        case 'x':
            dsetname_mX = optarg;
            break;
        case 'z':
            sscanf(optarg, "%dx%d", &MPI_P_DIM, &MPI_Q_DIM);
            break;
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

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_NNODE);
    MPI_Comm_rank(MPI_COMM_WORLD, &grid.my_rank);
    DAGTM_SETMYID(grid.my_rank);

#ifdef COMM_IPP_ON
    dagtm_init();
#endif

    if (isLogToFile)
    {
        //dagtm_setfilelog(grid.my_rank);
        DAGTM_SETLOGFILE();
    }
    else
    {
        DAGTM_SETLOG(stdout);
    }

    if (grid.my_rank == 0)
    {
        DEBUG(DAGTM_CRITIC_MSG, _DAGTM__REV_);
        DEBUG(DAGTM_CRITIC_MSG, _DAGTM__DATE_);
    }

    // We need to know N and D first
    h5infileid = h5open(filename_h5input, MPI_COMM_WORLD);
    CHECK(h5infileid != FAIL);

    DEBUG(DAGTM_INFO_MSG, "H5Fopen succeed");
    {
        hsize_t dims[RANK];
        herr_t ret;
        ret = dagtm_h5_dim(h5infileid, dsetname_mT, RANK, dims);
        CHECK(ret != FAIL);
        N = dims[0];
        D = dims[1];
    }

    // Try to make a square block
    // Small MPI_P_DIM is preferred
    if ((MPI_P_DIM == 0) && (MPI_Q_DIM == 0))
    {
        //MPI_P_DIM = (int)floor(sqrt(MPI_NNODE));
        double ratio = (double) N / K;
        MPI_P_DIM = (int) floor(sqrt(MPI_NNODE / ratio));
        if (MPI_P_DIM == 0)
        {
            MPI_P_DIM = 1;
        }

        MPI_Q_DIM = MPI_NNODE / MPI_P_DIM;

        while ((MPI_NNODE % MPI_P_DIM) != 0)
        {
            MPI_P_DIM--;
        }
        MPI_Q_DIM = MPI_NNODE / MPI_P_DIM;
    }

    if (MPI_P_DIM == 0)
    {
        MPI_P_DIM = MPI_NNODE / MPI_Q_DIM;
    }

    if (MPI_Q_DIM == 0)
    {
        MPI_Q_DIM = MPI_NNODE / MPI_P_DIM;
    }

    if ((MPI_P_DIM * MPI_Q_DIM) != MPI_NNODE)
    {
        DEBUG(DAGTM_CRITIC_MSG, "Grid setup failure");
        MPI_Finalize();
        return 1;
    }

    if (setup_grid(&grid, MPI_P_DIM, MPI_Q_DIM))
    {
        DEBUG(DAGTM_CRITIC_MSG, "MPI setup failure");
        MPI_Finalize();
        return 1;
    }

    /* execute child_handler() when receiving a signal of type SIGCHLD */
    signal(SIGCHLD, &sig_handler);
    
    //struct sigaction new_action;
    ///* Set up the structure to specify the new action. */
    //new_action.sa_handler = sig_handler;
    //sigemptyset (&new_action.sa_mask);
    //new_action.sa_flags = 0;
    //
    //sigaction(SIGUSR1, &new_action, NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    //DEBUG(DAGTM_CRITIC_MSG, "Starting ... ");

    /* Turn off H5 error handling permanently */
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    gsl_set_error_handler(dagtm_gsl_errorhandler);

    char path[BUFFSIZE];
    gethostname(path, BUFFSIZE);
    DEBUG(DAGTM_CRITIC_MSG, "Hostname = %s, Rank = %d (%d,%d)", path,
          grid.my_rank, grid.my_row_coord, grid.my_col_coord);

    MPI_Barrier(MPI_COMM_WORLD);

    KsplitbyPcounts = (int *) malloc(MPI_P_DIM * sizeof(int));
    KsplitbyPoffsets = (int *) malloc(MPI_P_DIM * sizeof(int));

    dagtm_split(K, MPI_P_DIM, KsplitbyPcounts, KsplitbyPoffsets);
    Kbar = KsplitbyPcounts[grid.my_row_coord];

    NsplitbyQcounts = (int *) malloc(MPI_Q_DIM * sizeof(int));
    NsplitbyQoffsets = (int *) malloc(MPI_Q_DIM * sizeof(int));

    dagtm_split(N, MPI_Q_DIM, NsplitbyQcounts, NsplitbyQoffsets);
    Nbar = NsplitbyQcounts[grid.my_col_coord];

    NbarSplitbyPcounts = (int *) malloc(MPI_P_DIM * sizeof(int));
    NbarSplitbyPoffsets = (int *) malloc(MPI_P_DIM * sizeof(int));

    dagtm_split(Nbar, MPI_P_DIM, NbarSplitbyPcounts, NbarSplitbyPoffsets);

    KbarSplitbyQcounts = (int *) malloc(MPI_Q_DIM * sizeof(int));
    KbarSplitbyQoffsets = (int *) malloc(MPI_Q_DIM * sizeof(int));

    dagtm_split(Kbar, MPI_Q_DIM, KbarSplitbyQcounts, KbarSplitbyQoffsets);

    MsplitbyRcounts = (int *) malloc(MPI_NNODE * sizeof(int));
    MsplitbyRoffsets = (int *) malloc(MPI_NNODE * sizeof(int));

    dagtm_split(M + 1, MPI_NNODE, MsplitbyRcounts, MsplitbyRoffsets);


    vlenN = gsl_vector_alloc(N);
    vlenK = gsl_vector_alloc(K);
    vlenNsub = gsl_vector_alloc(Nbar);
    vlenKsub = gsl_vector_alloc(Kbar);
    vgc = gsl_vector_alloc(Nbar);
    vgc2 = gsl_vector_alloc(Nbar);

    ws_ctemp = dagtm_ctemp_workspace_alloc(Nbar, D);
    ws_dist = dagtm_dist_workspace_alloc(Kbar, Nbar);

    //--------------------
    // Aloocating memory
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Allocating memory ... ");
    mTsub = gsl_matrix_calloc(Nbar, D);
    mXsub = gsl_matrix_calloc(Kbar, L);
    mYsub = gsl_matrix_calloc(Kbar, D);
    mFIsub = gsl_matrix_calloc(Kbar, M + 1);
    mDsub = gsl_matrix_calloc(mYsub->size1, mTsub->size1);
    mRsub = gsl_matrix_calloc(mYsub->size1, mTsub->size1);
    mRTsub = gsl_matrix_calloc(mYsub->size1, D);
    mFIRsub = gsl_matrix_calloc(M + 1, mTsub->size1);
    mFIRTsub = gsl_matrix_calloc(M + 1, D);
    mMsub = gsl_matrix_alloc(mRsub->size2, mXsub->size2);
    mW = gsl_matrix_alloc(M + 1, D);

    vidsub = gsl_vector_int_calloc(Nbar);
    vlbsub = gsl_vector_int_calloc(Nbar);

    mA = gsl_matrix_alloc(M + 1, M + 1);
    mB = gsl_matrix_alloc(M + 1, D);

    //mgRTsub = gsl_matrix_alloc(mRTsub->size1, mRTsub->size2);
    vggsub = gsl_vector_alloc(mRsub->size1);
    //vgggsub = gsl_vector_alloc(mRsub->size1);
    mGFIsub = gsl_matrix_alloc(mFIsub->size1, mFIsub->size2);

    //--------------------
    // Reading dataset
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Reading dataset ... ");

    // T
    h5read_byrow(h5infileid, dsetname_mT, mTsub,
                 N, D, NsplitbyQoffsets[grid.my_col_coord], Nbar, 0);

    // id
    hsize_t dim = N;
    hsize_t ldim = Nbar;
    hsize_t offset = NsplitbyQoffsets[grid.my_col_coord];

    ret = dagtm_h5_rw(TRUE, h5infileid, dsetname_vid, H5T_NATIVE_INT,
                      1, &dim, &ldim, &offset, &ldim, vidsub->data);
    if (ret == FAIL)
    {
        DEBUG(DAGTM_WARNING_MSG, "Error in reading id dataset.");
        gsl_vector_int_free(vidsub);
        vidsub = NULL;
    }

    // lb
    ret = dagtm_h5_rw(TRUE, h5infileid, dsetname_vlb, H5T_NATIVE_INT,
                      1, &dim, &ldim, &offset, &ldim, vlbsub->data);
    if (ret == FAIL)
    {
        DEBUG(DAGTM_WARNING_MSG, "Error in reading label dataset.");
        gsl_vector_int_free(vlbsub);
        vlbsub = NULL;
    }
    H5FCLOSEANDSET(h5infileid); // Close infile
	
    if (filename_h5checkpoint == NULL)
    {
        DEBUG(DAGTM_INFO_MSG, "mX is allocating ... ");
        mX = gsl_matrix_alloc(K, L);
        DEBUG(DAGTM_INFO_MSG, "mFI is allocating ... ");
        mFI = gsl_matrix_alloc(K, M + 1);

        gsl_vector *vTmu = gsl_vector_alloc(D);
        gsl_vector *vTsd = gsl_vector_alloc(D);

        dagtm_mpi_mean(mTsub, N, vTmu, grid.row_comm);
        dagtm_mpi_sd(mTsub, N, vTmu, vTsd, grid.row_comm);

        DEBUG(DAGTM_INFO_MSG, "Random setting up ... ");
        ret =
            dagtm_stp3_rnd(vTmu, vTsd, L, K, M, 2, gridType, mX, mFI, mW,
                           &beta);
        //DEBUG(DAGTM_CRITIC_MSG, "Init beta : %f", beta);
        CHECK(ret == GSL_SUCCESS);

        int offset = KsplitbyPoffsets[grid.my_row_coord];
        gsl_matrix_const_view vXsub =
            gsl_matrix_const_submatrix(mX, offset, 0, Kbar, L);
        gsl_matrix_memcpy(mXsub, &vXsub.matrix);

        gsl_matrix_const_view vFIsub =
            gsl_matrix_const_submatrix(mFI, offset, 0, Kbar, M + 1);
        gsl_matrix_memcpy(mFIsub, &vFIsub.matrix);

        gsl_vector_free(vTmu);
        gsl_vector_free(vTsd);
        gsl_matrix_free(mX);
        gsl_matrix_free(mFI);

        //--------------------
        // Compute mYsub
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Compute mYsub ... ");
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, mFIsub, mW, 0, mYsub);
        
        //--------------------
        // Comptue 1st Critical Temp
        //--------------------
        if (isDA)
        {
            if (doNeedFirstCTemp)
            {
                DEBUG(DAGTM_INFO_MSG, "Compute 1st Critical Temp ... ");
                dagtm_mpi_ctemp(mTsub, N, &firstCTemp, grid.row_comm);
                startTemp = firstCTemp / pow(alpha, 10);
            }
            temp = startTemp;
            prevExpandedTemp = temp;
        }

    }
    else
    {
        DEBUG(DAGTM_CRITIC_MSG, "Restarting ... ");

        h5ckpfileid = h5open(filename_h5checkpoint, MPI_COMM_WORLD);
        CHECK(h5ckpfileid != FAIL);

        // X
        DEBUG(DAGTM_INFO_MSG, "X is being read ... ");
        h5read_byrow(h5ckpfileid, dsetname_mX, mXsub,
                     K, L, KsplitbyPoffsets[grid.my_row_coord],
                     Kbar, 0);

        // FI
        DEBUG(DAGTM_INFO_MSG, "mFIsub is being saved ... ");
        h5read_byrow(h5ckpfileid, dsetname_mFI, mFIsub,
                     K, M + 1, KsplitbyPoffsets[grid.my_row_coord],
                     Kbar, 0);

        // Y
        DEBUG(DAGTM_INFO_MSG, "mYsub is being read ... ");
        h5read_byrow(h5ckpfileid, dsetname_mY, mYsub,
                     K, D, KsplitbyPoffsets[grid.my_row_coord],
                     Kbar, 0);

        // beta
        DEBUG(DAGTM_INFO_MSG, "beta is being read ... ");
        h5read_scalar(h5ckpfileid, dsetname_vbeta,
                      H5T_NATIVE_DOUBLE, &beta);

        // temp
        DEBUG(DAGTM_INFO_MSG, "Temp is being read ... ");
        h5read_scalar(h5ckpfileid, dsetname_temp, 
                      H5T_NATIVE_DOUBLE, &temp);
        startTemp = temp;

        // iter
        DEBUG(DAGTM_INFO_MSG, "iter is being read ... ");
        h5read_scalar(h5ckpfileid, dsetname_iter, 
                      H5T_NATIVE_INT, &nloop);

        // seed
        DEBUG(DAGTM_INFO_MSG, "seed is being read ... ");
        h5read_scalar(h5ckpfileid, dsetname_seed,
                      H5T_NATIVE_INT, &dagtm_seed);

        H5FCLOSEANDSET(h5ckpfileid); // Close infile
    }

    // Open outfile
    if (filename_h5output != NULL)
    {
        h5outfileid = h5create(filename_h5output, MPI_COMM_WORLD);
        H5FCLOSEANDSET(h5outfileid);
    }

    //--------------------
    // Real work starts from here
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Starting ... ");
    MPI_Barrier(MPI_COMM_WORLD);

    //--------------------
    // Comptue DIST and RESP
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute DIST ... ");
    dagtm_dist(mYsub, mTsub, mDsub);

    if (doLogProgress)
    {
        ret = dagtm_collection_alloc(&TempCollection, INIT_COLLECTION_SIZE);
        CHECK(ret == GSL_SUCCESS);
        ret = dagtm_collection_alloc(&LLHCollection, INIT_COLLECTION_SIZE);
        CHECK(ret == GSL_SUCCESS);
        ret = dagtm_collection_alloc(&FreeCollection, INIT_COLLECTION_SIZE);
        CHECK(ret == GSL_SUCCESS);
    }

    ret = dagtm_queue_alloc(&QualQueue, INIT_QUEUE_SIZE, DBL_MAX);
    CHECK(ret == GSL_SUCCESS);

    if (doInterpolation)
    {
        DEBUG(DAGTM_CRITIC_MSG, "Interpolating ... ");
     
        if (filename_h5checkpoint == NULL)
        {
            DEBUG(DAGTM_CRITIC_MSG, "No checkpointing file.. ");
        }
   
        //--------------------
        // Comptue Resp
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Compute RESP ... ");
        dagtm_mpi_resp(K, beta, mDsub, isDA, temp, mRsub,
                       vgc, vgc2, grid.col_comm);

        //--------------------
        // Comptue MAP
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
        result =
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                           mMsub);

        DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
        MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                      (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                      MPI_SUM, grid.col_comm);

        //--------------------
        // Write to file
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Writing to file ... ");

        h5outfileid = h5append(filename_h5output, grid.comm);
        {
            // Write M
            DEBUG(DAGTM_INFO_MSG, "mMsub is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mM, mMsub,
                         N, L, NsplitbyQoffsets[grid.my_col_coord],
                         NbarSplitbyPcounts[grid.my_row_coord],
                         NbarSplitbyPoffsets[grid.my_row_coord]);
        }
        
        H5FCLOSEANDSET(h5outfileid);
        
        MPI_Finalize();
        return 0;
    }

#ifdef DAGTM_PVIZRPC_SERVER_ON
    boost::asio::io_service io_service;
    cgl::protorpc::RpcServer * server;

    //--------------------
    // Comptue Resp
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute RESP ... ");
    dagtm_mpi_resp(K, beta, mDsub, isDA, temp, mRsub,
                   vgc, vgc2, grid.col_comm);

    //--------------------
    // Comptue MAP
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
    result =
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                       mMsub);

    DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
    MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                  (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                  MPI_SUM, grid.col_comm);

    if (grid.my_rank == 0)
    {
        google::protobuf::Service * service =
            new PvizRpcServiceImp(vidsub, vlbsub, mMsub, mXsub);
        google::protobuf::RpcController * controller =
            new cgl::protorpc::RpcControllerImp();

        server =
            new cgl::protorpc::RpcServer(io_service, "127.0.0.1", 12345,
                                         service, controller);
        server->setMultiSessionMode(false);

        server->Run();
        io_service.run();
        io_service.reset();

        DEBUG(DAGTM_INFO_MSG, "io_service done. ");
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef DAGTM_NB_SERVER_ON
    //--------------------
    // Comptue Resp
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute RESP ... ");
    dagtm_mpi_resp(K, beta, mDsub, isDA, temp, mRsub,
                   vgc, vgc2, grid.col_comm);
    
    //--------------------
    // Comptue MAP
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
    result =
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                       mMsub);
    
    DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
    MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                  (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                  MPI_SUM, grid.col_comm);
    
    string host = "156.56.104.176";
    int port = 55045;
    int templateId=55745;

    srand((unsigned) time(0));
    int entityId = (rand() % 10000) + 10000;
    string contentSynopsis = "/tmp/topic1";

    ServiceClient serviceClient;

    if (grid.my_rank == 0)
    {
        serviceClient.init(host,port,entityId,templateId);
        DEBUG(DAGTM_CRITIC_MSG, "Connection established to NB ... ");
        //cout<<"Publishing = "<<contentSynopsis<<"   "<<msg<<endl;
        char msg[100] = "Init ... ";

        serviceClient.publish(contentSynopsis,(char *)msg,strlen(msg));
        sleep(5);
    }
#endif

#ifdef DAGTM_ACTIVEMQ_SERVER_ON
    //--------------------
    // Comptue Resp
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute RESP ... ");
    dagtm_mpi_resp(K, beta, mDsub, isDA, temp, mRsub,
                   vgc, vgc2, grid.col_comm);
    
    //--------------------
    // Comptue MAP
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
    result =
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                       mMsub);
    
    DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
    MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                  (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                  MPI_SUM, grid.col_comm);
    
    Connection* connection;
    Session* session;
    
    Topic* sendTopic;
    Topic* listenTopic;
    
    MessageProducer* producer;
    MessageConsumer* consumer;
    //bool useTopic = false;
    //bool clientAck = false;
    unsigned int numMessages;
    
    std::string destURI = "topic2"; 
    std::string listenURI = "topic1"; 
    
    activemq::library::ActiveMQCPP::initializeLibrary();
    
    std::string brokerURI =
        "failover://(tcp://156.56.104.176:61616"
        "?wireFormat.maxInactivityDuration=0"
        //        "?wireFormat=openwire"
        //        "&connection.useAsyncSend=true"
        //        "&transport.commandTracingEnabled=true"
        //        "&transport.tcpTracingEnabled=true"
        //        "&wireFormat.tightEncodingEnabled=true"
        ")";
    
    // Create a ConnectionFactory
    auto_ptr<ActiveMQConnectionFactory> connectionFactory( new ActiveMQConnectionFactory( brokerURI ) );
    
    // Create a Connection
    try{
        connection = connectionFactory->createConnection();
        connection->start();
    } catch( CMSException& e ) {
        e.printStackTrace();
        throw e;
    }
    
    session = connection->createSession( Session::AUTO_ACKNOWLEDGE );
    
    sendTopic = session->createTopic( destURI );
    
    // Wait SYNC
    // Create a MessageConsumer from the Session to the Topic or Queue
    //consumer = session->createConsumer( destination );
    //consumer->setMessageListener( this );
    

    // Create a MessageProducer from the Session to the Topic or Queue
    producer = session->createProducer( sendTopic );
    producer->setDeliveryMode( DeliveryMode::NON_PERSISTENT );
    
    cgl::pviz::rpc::PvizMessage pvizmessage;
    pvizmessage.set_type(cgl::pviz::rpc::PvizMessage::DATA);
    pvizmessage.set_timestamp(1004);
    pvizmessage.set_stepid(0);

    for (unsigned int i = 0; i < vlbsub->size; i++)
    {
        pvizmessage.add_labels(gsl_vector_int_get(vlbsub, i));
    }
    
    std::string text;
    pvizmessage.SerializeToString(&text);
    //std::cout << pvizmessage.DebugString() << std::endl;
    std::cout << "serialized text size :" << text.length() << std::endl;
    
    //TextMessage* message = session->createTextMessage( text );
    //std::cout << "size :" << message->getText().length() << std::endl;
    //std::cout << "text :" << message->getText() << std::endl;

    BytesMessage* message = session->createBytesMessage();
    //message->writeBytes((unsigned char *) text.c_str(), 0, text.length());
    //std::cout << "getBodyLength :" << message->getBodyLength() << std::endl;
    
    message->writeInt(0); // LABEL DATA(0), POSITION DATA (1)
    message->writeDouble(1004); // Timestamp
    message->writeInt(0); // Stepid

    message->writeInt(vlbsub->size); // Vector length
    for (unsigned int i = 0; i < vlbsub->size; i++)
    {
        message->writeInt(gsl_vector_int_get(vlbsub, i));
    }
    
    producer->send( message );
    delete message;    
    
    sleep(5);
#endif

#ifndef WIN32
    clk_tck = sysconf(_SC_CLK_TCK);
    st_time = times(&st_cpu);
#endif

    bool bReadyToExpand = FALSE;
    bool bDoExpand = FALSE;
    int numOfExpanded = 0;

    do
    {
        if (temp < 1.0)
        {
            temp = 1.0;
            isLast = TRUE;
        }

        if (grid.my_rank == 0)
        {
            fprintf(dagtm_flog, "\nT= %.10g\n", temp);
        }

        ret = dagtm_queue_reset(&QualQueue, DBL_MAX);

        double qual_before_temp = llh;

        if (grid.my_rank == 0)
        {
            fprintf(dagtm_flog, ">>> %15s %15s %15s %15s %15s %15s\n", "GSEQ", "SEQ", "K", "TEMP", "NFREE", "LLH");
        }

        lloop = 0;
        while (TRUE)
        {
            MPI_Barrier(grid.comm);
            logtic(LOG_MAIN);

            DEBUG(DAGTM_INFO_MSG, "Compute RESP ... ");
            // Start logging LOG_RESP
            MPI_TIC(LOG_RESP);

            // LOG_RESP_COMM
            dagtm_mpi_resp(K, beta, mDsub, isDA, temp,
                           mRsub, vgc, vgc2, grid.col_comm);

            // End logging LOG_RESP
            MPI_TOC(LOG_RESP);

            DEBUG(DAGTM_INFO_MSG, "Compute Quality ... ");
            // Start logging LOG_LGLH
            MPI_TIC(LOG_LGLH);

            qual_old = qual;
            lloop++;
            nloop++;

            if (isDA)
            {
                dagtm_mpi_qual(N, K, D, beta, vgc, FALSE, temp,
                               &llh, grid.row_comm);

                dagtm_mpi_qual(N, K, D, beta, vgc2, TRUE, temp,
                               &nfree, grid.row_comm);

                qual = nfree;
                diff = qual - qual_old;

                if (grid.my_rank == 0)
                {
                    //fprintf(dagtm_flog, "-F= %.10g\tL= %.10g\tD= %.10g\n",
                    //        nfree, llh, diff);
                    fprintf(dagtm_flog, "::: %15d %15d %15d %15.7f %15.7f %15.7f %15.7f\n",
                            nloop, lloop, K, temp, nfree, llh, diff);
                    fflush(dagtm_flog);
                }

                if (doLogProgress)
                {
                    dagtm_collection_push(&TempCollection, temp);
                    dagtm_collection_push(&LLHCollection, llh);
                    dagtm_collection_push(&FreeCollection, nfree);
                }
            }
            else
            {
                // LOG_LGLH_COMM
                dagtm_mpi_qual(N, K, D, beta, vgc, FALSE, temp,
                               &llh, grid.row_comm);

                qual = llh;
                diff = qual - qual_old;

                if (grid.my_rank == 0)
                {
                    //fprintf(dagtm_flog, "L= %.15g\tD= %.15g\n", llh, diff);
                    fprintf(dagtm_flog, "::: %15d %15d %15d %15.7f %15.7f %15.7f %15.7f\n",
                            nloop, lloop, K, temp, -llh, llh, diff);
                    fflush(dagtm_flog);
                }

                if (doLogProgress)
                {
                    dagtm_collection_push(&TempCollection, temp);
                    dagtm_collection_push(&LLHCollection, llh);
                }
            }

            // End logging LOG_LGLH 
            MPI_TOC(LOG_LGLH);

            //diff = qual - qual_old;
            if ((ABS(diff) < eps))
            {
                DEBUG(DAGTM_INFO_MSG, "Threshold reached ... ");
                break;
            }

            if (maxloop == UINT_MAX)    // maxloop is not set
            {
                if (nloop > INIT_QUEUE_SIZE)    // try tp detect repeating (repeating windows size is INIT_QUEUE_SIZE) 
                {
                    double qualavg_old = dagtm_queue_mean(&QualQueue);
                    dagtm_queue_push(&QualQueue, qual);
                    double qualavg = dagtm_queue_mean(&QualQueue);
                    double diffavg = qualavg - qualavg_old;
                    if (ABS(diffavg) < eps)
                    {
                        DEBUG(DAGTM_INFO_MSG,
                              "Average of %d iterations has been converged ... ",
                              INIT_QUEUE_SIZE);
                        break;
                    }
                }
            }

            DEBUG(DAGTM_INFO_MSG, "Compute vgg ... ");
            // Start logging LOG_LGLH 
            MPI_TIC(LOG_VGGG);

            dagtm_matrix_rowsum(mRsub, vggsub);

            // LOG_VGGG_COMM
            MPI_TICTOC(LOG_VGGG_COMM,
                       MPI_Allreduce(MPI_IN_PLACE, vggsub->data,
                                     (int) vggsub->size, MPI_DOUBLE,
                                     MPI_SUM, grid.row_comm));
            // End logging LOG_VGGG 
            MPI_TOC(LOG_VGGG);

            DEBUG(DAGTM_INFO_MSG, "Compute mAsub ... ");
            // Start logging LOG_ASUB 
            MPI_TIC(LOG_ASUB);

            gsl_matrix_memcpy(mGFIsub, mFIsub);
            dagtm_matrix_scale_by_row(mGFIsub, vggsub);
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mFIsub, mGFIsub, 0,
                           mA);

            // LOG_ASUB_COMM
            MPI_TICTOC(LOG_ASUB_COMM,
                       MPI_Allreduce(MPI_IN_PLACE, mA->data,
                                     (int) (mA->size1 * mA->tda),
                                     MPI_DOUBLE, MPI_SUM, grid.col_comm));
            // End logging LOG_ASUB 
            MPI_TOC(LOG_ASUB);

            DEBUG(DAGTM_INFO_MSG, "Compute mBsub ... ");
            // Start logging LOG_BSUB 
            MPI_TIC(LOG_BSUB);

            LOG_TIC(LOG_BSUB_GEM1);
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1,
                           mFIsub, mRsub, 0, mFIRsub);
            LOG_TOC(LOG_BSUB_GEM1);

            LOG_TIC(LOG_BSUB_GEM2);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1,
                           mFIRsub, mTsub, 0, mB /*mFIRTsub */ );
            LOG_TOC(LOG_BSUB_GEM2);

            // LOG_BSUB_COMM
            MPI_TICTOC(LOG_BSUB_COMM,
                       MPI_Allreduce(MPI_IN_PLACE, mB->data,
                                     (int) (mFIRTsub->size1 *
                                            mFIRTsub->tda),
                                     MPI_DOUBLE, MPI_SUM, grid.comm));
            // End logging LOG_BSUB 
            MPI_TOC(LOG_BSUB);

            // Start logging LOG_SOLV
            MPI_TIC(LOG_SOLV);
            dagtm_solve2(mA, mB, mW);

            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1,
                           mFIsub, mW, 0, mYsub);

            // End logging LOG_SOLV
            MPI_TOC(LOG_SOLV);

            DEBUG(DAGTM_INFO_MSG, "Compute beta ... ");
            // Start logging LOG_BETA 
            MPI_TIC(LOG_BETA);

            dagtm_mpi_beta(mYsub, mTsub, mRsub, N, D, mDsub,
                           &beta, ws_dist, grid.comm);

            // End logging LOG_BETA 
            MPI_TOC(LOG_BETA);

            //DEBUG(DAGTM_INFO_MSG, "beta : %f", beta);

            MPI_Barrier(grid.comm);
            logtoc(LOG_MAIN);

            if (
                (isCheckpointing && ((nloop % checkpointingPerNloop) == 0)) ||
                isForceCheckpointing
                )
            {
                //--------------------
                // Write to file
                //--------------------
                DEBUG(DAGTM_INFO_MSG, "Writing to file ... ");

                char filename_h5check[80];
                sprintf(filename_h5check, "%s%d.h5", filename_h5ckprefix, nloop);
                DEBUG(DAGTM_INFO_MSG, "Writing to file ... %s", filename_h5check);

                h5ckpfileid = h5create(filename_h5check, MPI_COMM_WORLD);
                CHECK(h5ckpfileid != FAIL);

                // X
                DEBUG(DAGTM_INFO_MSG, "mXsub is being saved ... ");
                h5save_byrow(h5ckpfileid, dsetname_mX, mXsub,
                             K, L, KsplitbyPoffsets[grid.my_row_coord],
                             KbarSplitbyQcounts[grid.my_col_coord],
                             KbarSplitbyQoffsets[grid.my_col_coord]);

                // FI
                DEBUG(DAGTM_INFO_MSG, "mFIsub is being saved ... ");
                h5save_byrow(h5ckpfileid, dsetname_mFI, mFIsub,
                             K, M + 1, KsplitbyPoffsets[grid.my_row_coord],
                             KbarSplitbyQcounts[grid.my_col_coord],
                             KbarSplitbyQoffsets[grid.my_col_coord]);

                // Y
                DEBUG(DAGTM_INFO_MSG, "mYsub is being saved ... ");
                h5save_byrow(h5ckpfileid, dsetname_mY, mYsub,
                             K, D, KsplitbyPoffsets[grid.my_row_coord],
                             KbarSplitbyQcounts[grid.my_col_coord],
                             KbarSplitbyQoffsets[grid.my_col_coord]);

                //// W
                //DEBUG(DAGTM_INFO_MSG, "mW is being saved ... ");
                //h5save_byrow(h5ckpfileid, dsetname_mW, mW,
                //             M + 1, D, 0,
                //             MsplitbyRcounts[grid.my_rank],
                //             MsplitbyRoffsets[grid.my_rank]);

                // beta
                DEBUG(DAGTM_INFO_MSG, "beta is being saved ... ");
                h5save_scalar(h5ckpfileid, dsetname_vbeta,
                              H5T_NATIVE_DOUBLE, &beta);

                // temp
                DEBUG(DAGTM_INFO_MSG, "Temp is being saved ... ");
                h5save_scalar(h5ckpfileid, dsetname_temp, 
                              H5T_NATIVE_DOUBLE, &temp);

                // iter
                DEBUG(DAGTM_INFO_MSG, "iter is being saved ... ");
                h5save_scalar(h5ckpfileid, dsetname_iter, 
                              H5T_NATIVE_INT, &nloop);
                
                // seed
                DEBUG(DAGTM_INFO_MSG, "seed is being saved ... ");
                h5save_scalar(h5ckpfileid, dsetname_seed, 
                              H5T_NATIVE_INT, &dagtm_seed);
                
                H5FCLOSEANDSET(h5ckpfileid);

                if (isForceCheckpointing)
                {
                    DEBUG(DAGTM_CRITIC_MSG, "Force checkpointing done");
                    MPI_Finalize();
                    return 0;
                }
            }

            if (nloop >= maxloop)
            {
                isLast = TRUE;
                DEBUG(DAGTM_INFO_MSG, "Max loop reached ... ");
                break;
            }
            //else
            //{
            //    nloop++;
            //}


#ifdef DAGTM_NB_SERVER_ON
            //--------------------
            // Comptue MAP
            //--------------------
            DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
            result =
                gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                               mMsub);

            DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
            MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                          (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                          MPI_SUM, grid.col_comm);
        
            if (grid.my_rank == 0)
            {
                char msg[100];
                sprintf(msg, "Loop : %d", nloop);
                serviceClient.publish(contentSynopsis,(char *)msg,strlen(msg));
                sleep(5);
            }
#endif

#ifdef DAGTM_ACTIVEMQ_SERVER_ON
            //--------------------
            // Comptue MAP
            //--------------------
            DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
            result =
                gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                               mMsub);
            
            DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
            MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                          (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                          MPI_SUM, grid.col_comm);
            
            if (grid.my_rank == 0)
            {
                /*
                  char msg[100];
                  sprintf(msg, "Loop : %d", nloop);

                  //TextMessage* message = session->createTextMessage( std::string(msg) );
                  //producer->send( message );
                  //delete message;
                
                  string text;
                  text = std::string(msg);
                
                  BytesMessage* message = session->createBytesMessage();
                  //message->writeString(text);
                  message->writeBytes((unsigned char *) text.c_str(), 0, text.length());
                  std::cout << "length :" << text.length() << std::endl;
                
                  producer->send( message );
                  delete message;
                */
                
                cgl::pviz::rpc::PvizMessage pvizmessage;
                pvizmessage.set_type(cgl::pviz::rpc::PvizMessage::DATA);
                pvizmessage.set_timestamp(0);
                pvizmessage.set_stepid(nloop);
                
                //cgl::pviz::rpc::Position* position = pvizmessage.mutable_positions();
                
                for (unsigned int i = 0; i < mMsub->size1; i++)
                {
                    cgl::pviz::rpc::Position* pos = pvizmessage.add_positions();
                    pos->set_x(gsl_matrix_get(mMsub, i, 0));
                    pos->set_y(gsl_matrix_get(mMsub, i, 1));
                    pos->set_z(gsl_matrix_get(mMsub, i, 2));
                }
                
                std::string text;
                pvizmessage.SerializeToString(&text);
                //std::cout << pvizmessage.DebugString() << std::endl;
                std::cout << "serialized text size :" << text.length() << std::endl;
                
                //TextMessage* message = session->createTextMessage( text );
                //std::cout << "size :" << message->getText().length() << std::endl;
                //std::cout << "text :" << message->getText() << std::endl;
                
                BytesMessage* message = session->createBytesMessage();
                //message->writeBytes((unsigned char *) text.c_str(), 0, text.length());
                //std::cout << "getBodyLength :" << message->getBodyLength() << std::endl;
                
                message->writeInt(1); // LABEL DATA(0), POSITION DATA (1)
                message->writeDouble(1004); // Timestamp
                message->writeInt(nloop); // Stepid
                
                message->writeInt(mMsub->size1 * mMsub->size2); // Vector length
                message->writeInt(mMsub->size2); // Vector length
                for (unsigned int i = 0; i < mMsub->size1; i++)
                {
                    message->writeDouble(gsl_matrix_get(mMsub, i, 0));
                    message->writeDouble(gsl_matrix_get(mMsub, i, 1));
                    message->writeDouble(gsl_matrix_get(mMsub, i, 2));
                }
                
                producer->send( message );
                delete message;    

                sleep(5);
            }
#endif

        } // End of while

        //MPI_Barrier(grid.comm);
        double qual_diff_at_temp = llh - qual_before_temp;
        DONLY(0, DUMP("Quality improved : %f", qual_diff_at_temp));

        if (isDA && !isLast)
        {
            switch (schedule)
            {
            case AUTO:
            {
                double ctemp_max;
                dagtm_mpi_next_ctemp(mTsub, mYsub, mRsub, vggsub, temp, beta, eps, 
                                     ws_ctemp, vlenKsub, &ctemp_max, grid.row_comm, grid.col_comm);

                if (ctemp_max >= temp)
                {
                    DEBUG(DAGTM_INFO_MSG,
                          "No valid next ctemp (ctemp_max = %g)",
                          ctemp_max);
                    ctemp_max = temp * alpha;
                }

                temp = ctemp_max;

            }
            break;
            case EXP:
            {
                if ((isHDA) && (K < Kmax))
                {
                    double ctemp_max;
                    gsl_vector * vctemp = vlenKsub;

                    if (bReadyToExpand)
                    {
                        if (qual_diff_at_temp < eps2)
                        {
                            DONLY(0, DEBUG(DAGTM_CRITIC_MSG, "Nothing happened. Cancel expanding ... "));
                            bReadyToExpand = FALSE;
                        }
                        else
                        {
                            DONLY(0, DEBUG(DAGTM_CRITIC_MSG, "Need to expand ... "));
                            bDoExpand = TRUE;
                        }
                    }

                    if (!bReadyToExpand)
                    {
                        dagtm_mpi_next_ctemp(mTsub, mYsub, mRsub, vggsub, temp, beta, eps, 
                                             ws_ctemp, vctemp, &ctemp_max, grid.row_comm, grid.col_comm);
                        DONLY(0, DEBUG(DAGTM_CRITIC_MSG, "ctemp_max : %f", ctemp_max));

                        if (ctemp_max >= (temp * alpha))
                        {
                            DONLY(0, DEBUG(DAGTM_CRITIC_MSG, "Be ready to expand ... "));
                            bReadyToExpand = TRUE;
                            qual_diff_at_temp = 0.0;
                        }
                    }

                    if (bDoExpand)
                    {
                        DONLY(0, DEBUG(DAGTM_CRITIC_MSG, "Do expand ... "));
                        bDoExpand = FALSE;
                        bReadyToExpand = FALSE;
                        numOfExpanded++;

                        //DUMPVF(vctemp);
                        //prevExpandedTemp = temp;

                        // Save MAP first ... 
                        if (filename_h5output != NULL)
                        {
                            if (doLogProgress)
                            {
                                //--------------------
                                // Comptue MAP
                                //--------------------
                                DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
                                result =
                                    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                                                   mMsub);

                                DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
                                MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                                              (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                                              MPI_SUM, grid.col_comm);

                                //-------------------
                                // Write to file
                                //--------------------
                                h5outfileid = h5append(filename_h5output, grid.comm);

                                //--------------------
                                // Write to file
                                //--------------------
                                DEBUG(DAGTM_INFO_MSG, "Writing to file ... ");
                                char dsetname[80];
                                size_t idx = TempCollection.next - 1;

                                // Write M
                                DEBUG(DAGTM_INFO_MSG, "mMsub is being saved ... ");
                                sprintf(dsetname, "M.%ld.dat", idx);
                                h5save_byrow(h5outfileid, dsetname, mMsub,
                                             N, L, NsplitbyQoffsets[grid.my_col_coord],
                                             NbarSplitbyPcounts[grid.my_row_coord],
                                             NbarSplitbyPoffsets[grid.my_row_coord]);

                                // Write X
                                DEBUG(DAGTM_INFO_MSG, "mXsub is being saved ... ");
                                sprintf(dsetname, "X.%ld.dat", idx);
                                h5save_byrow(h5outfileid, dsetname, mXsub,
                                             K, L, KsplitbyPoffsets[grid.my_row_coord],
                                             KbarSplitbyQcounts[grid.my_col_coord],
                                             KbarSplitbyQoffsets[grid.my_col_coord]);

                                // Write Y
                                DEBUG(DAGTM_INFO_MSG, "mYsub is being saved ... ");
                                sprintf(dsetname, "Y.%ld.dat", idx);
                                h5save_byrow(h5outfileid, dsetname, mYsub,
                                             K, D, KsplitbyPoffsets[grid.my_row_coord],
                                             KbarSplitbyQcounts[grid.my_col_coord],
                                             KbarSplitbyQoffsets[grid.my_col_coord]);

                                // Write FI
                                DEBUG(DAGTM_INFO_MSG, "mFIsub is being saved ... ");
                                sprintf(dsetname, "FI.%ld.dat", idx);
                                h5save_byrow(h5outfileid, dsetname, mFIsub,
                                             K, M + 1, KsplitbyPoffsets[grid.my_row_coord],
                                             KbarSplitbyQcounts[grid.my_col_coord],
                                             KbarSplitbyQoffsets[grid.my_col_coord]);

                                // Write W
                                DEBUG(DAGTM_INFO_MSG, "mW is being saved ... ");
                                sprintf(dsetname, "W.%ld.dat", idx);
                                h5save_byrow(h5outfileid, dsetname, mW,
                                             M + 1, D, 0,
                                             MsplitbyRcounts[grid.my_rank],
                                             MsplitbyRoffsets[grid.my_rank]);

                                // Write ctemp
                                hsize_t dims[2];
                                hsize_t viewDims[2];
                                hsize_t offsets[2];

                                //dims[1] = 1;
                                dims[0] = K;
                                //offsets[1] = 0;
                                offsets[0] = KsplitbyPoffsets[grid.my_row_coord] + KbarSplitbyQoffsets[grid.my_col_coord];
                                //viewDims[1] = 1;
                                viewDims[0] = KbarSplitbyQcounts[grid.my_col_coord];

                                gsl_vector_view view;

                                // Write ctemp
                                DEBUG(DAGTM_INFO_MSG, "ctemp is being saved ... ");
                                sprintf(dsetname, "ctemp.%ld.dat", idx);

                                view =
                                    gsl_vector_subvector(vctemp,
                                                         KbarSplitbyQoffsets[grid.my_col_coord],
                                                         KbarSplitbyQcounts[grid.my_col_coord]);

                                ret =
                                    dagtm_h5_rw(FALSE, h5outfileid, dsetname,
                                                H5T_NATIVE_DOUBLE, 1, dims, viewDims,
                                                offsets, viewDims, view.vector.data);
                                // beta
                                DEBUG(DAGTM_INFO_MSG, "beta is being saved ... ");
                                sprintf(dsetname, "beta.%ld.dat", idx);

                                dims[0] = 1;
                                offsets[0] = 0;
                                if (grid.my_rank == 0)
                                {
                                    ret =
                                        dagtm_h5_rw(FALSE, h5outfileid, dsetname,
                                                    H5T_NATIVE_DOUBLE, 1, dims, dims, offsets,
                                                    dims, &beta);
                                }
                                else
                                {
                                    ret =
                                        dagtm_h5_rw(FALSE, h5outfileid, dsetname,
                                                    H5T_NATIVE_DOUBLE, 1, dims, offsets, dims,
                                                    offsets, &beta);
                                }

                                H5FCLOSEANDSET(h5outfileid);
                            }
                        }

                        K = pow((int)round(pow(K, 1.0/L)) * 2, L);

                        if (grid.my_rank == 0)
                        {
                            DEBUG(DAGTM_CRITIC_MSG, "new K : %d", K);
                        }

                        DEBUG(DAGTM_INFO_MSG, "mX is allocating ... ");
                        mX = gsl_matrix_alloc(K, L);

                        DEBUG(DAGTM_INFO_MSG, "mFI is allocating ... ");
                        mFI = gsl_matrix_alloc(K, M + 1);

                        DEBUG(DAGTM_INFO_MSG, "New setting up ... ");
                        ret = dagtm_stp(L, K, M, 2, gridType, mX, mFI);

                        if (ret != GSL_SUCCESS)
                            return -1;

                        // Update
                        dagtm_split(K, MPI_P_DIM, KsplitbyPcounts, KsplitbyPoffsets);
                        Kbar = KsplitbyPcounts[grid.my_row_coord];

                        dagtm_split(Kbar, MPI_Q_DIM, KbarSplitbyQcounts, KbarSplitbyQoffsets);

                        // Re-allocate ...
                        dagtm_dist_workspace_free(ws_dist);
                        ws_dist = dagtm_dist_workspace_alloc(Kbar, Nbar);

                        VECTOR_REALLOC(vlenKsub, Kbar);
                        MATRIX_REALLOC(mXsub, Kbar, L);
                        MATRIX_REALLOC(mYsub, Kbar, D);
                        MATRIX_REALLOC(mFIsub, Kbar, M + 1);

                        MATRIX_REALLOC(mDsub, mYsub->size1, mTsub->size1);
                        MATRIX_REALLOC(mRsub, mYsub->size1, mTsub->size1);
                        MATRIX_REALLOC(mRTsub, mYsub->size1, D);

                        VECTOR_REALLOC(vggsub, mRsub->size1);
                        MATRIX_REALLOC(mGFIsub, mFIsub->size1, mFIsub->size2);

                        int offset = KsplitbyPoffsets[grid.my_row_coord];
                        gsl_matrix_const_view vXsub =
                            gsl_matrix_const_submatrix(mX, offset, 0, Kbar, L);
                        gsl_matrix_memcpy(mXsub, &vXsub.matrix);

                        gsl_matrix_const_view vFIsub =
                            gsl_matrix_const_submatrix(mFI, offset, 0, Kbar, M + 1);
                        gsl_matrix_memcpy(mFIsub, &vFIsub.matrix);

                        gsl_matrix_free(mX);
                        gsl_matrix_free(mFI);

                        // Re-compute ...
                        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1,
                                       mFIsub, mW, 0, mYsub);

                        dagtm_dist_ws(mYsub, mTsub, mDsub, ws_dist);
                    }

                }

                temp *= alpha; /* next temperature */
            }
            break;
            case LINEAR:
                temp -= delta;
                break;
            default:
                break;
            }

        }
        else
        {
            break;
        }
        
#ifdef DAGTM_PVIZRPC_SERVER_ON
        //--------------------
        // Comptue MAP
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
        result =
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                           mMsub);

        DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
        MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                      (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                      MPI_SUM, grid.col_comm);

        if (grid.my_rank == 0)
        {
            server->resumeSessions();
            io_service.run();
            io_service.reset();
        }
#endif

        MPI_Barrier(grid.comm);
    }
    while (!isLast);            // End of do

#ifndef WIN32
    en_time = times(&en_cpu);
#endif

    //--------------------
    // Printing summary
    //--------------------
    if (grid.my_rank == 0)
    {
        fprintf(dagtm_flog, "Done.\n");
        fprintf(dagtm_flog,
                "==================== SUMMARY ====================\n");
        fprintf(dagtm_flog, "%20s = %s\n", "Data filename",
                filename_h5input);
        fprintf(dagtm_flog, "%20s = %dx%d\n", "Compute Grid", MPI_P_DIM,
                MPI_Q_DIM);
        fprintf(dagtm_flog, "%20s = %d\n", "N", N);
        fprintf(dagtm_flog, "%20s = %d\n", "D", D);
        fprintf(dagtm_flog, "%20s = %d\n", "K", K);
        fprintf(dagtm_flog, "%20s = %d\n", "L", L);
        fprintf(dagtm_flog, "%20s = %d\n", "M", M);
        fprintf(dagtm_flog, "%20s = %ld\n", "SEED", dagtm_seed);
        fprintf(dagtm_flog, "%20s = %.10g\n", "Max. Qual", qual);

        switch(gridType)
        {
        case GRID_SWEEP:
            fprintf(dagtm_flog, "%20s = %s\n", "Grid Type", "SWEEP");
            break;
        case GRID_SPACEFILLING:
            fprintf(dagtm_flog, "%20s = %s\n", "Grid Type", "SPACEFILLING");
            break;
        }

        if (isDA)
        {
            fprintf(dagtm_flog, "%20s = %.10g\n", "Start Temp", startTemp);
            fprintf(dagtm_flog, "%20s = %.10g\n", "1st Critical Temp",
                    firstCTemp);
            switch (schedule)
            {
            case AUTO:
                fprintf(dagtm_flog, "%20s = %s\n", "Cooling Schedule",
                        "AUTO");
                break;
            case EXP:
                fprintf(dagtm_flog, "%20s = %s\n", "Cooling Schedule",
                        "EXP");
                fprintf(dagtm_flog, "%20s = %g\n", "alpha", alpha);
                break;
            case LINEAR:
                fprintf(dagtm_flog, "%20s = %s\n", "Cooling Schedule",
                        "LINEAR");
                fprintf(dagtm_flog, "%20s = %g\n", "delta", delta);
                break;
            default:
                break;
            }
        }
        fprintf(dagtm_flog, "%20s = %d\n", "Num. of Iterations", nloop);
        fprintf(dagtm_flog,
                "-------------------------------------------------\n");
#ifndef WIN32
        fprintf(dagtm_flog, "%20s = %.03f\n", "Real Time(s)",
                (intmax_t) (en_time - st_time) / (double) clk_tck);
        fprintf(dagtm_flog, "%20s = %.03f\n", "User Time(s)",
                (intmax_t) (en_cpu.tms_utime -
                            st_cpu.tms_utime) / (double) clk_tck);
        fprintf(dagtm_flog, "%20s = %.03f\n", "System Time(s)",
                (intmax_t) (en_cpu.tms_stime -
                            st_cpu.tms_stime) / (double) clk_tck);
#endif
        fprintf(dagtm_flog,
                "=================================================\n");
        fflush(dagtm_flog);
    }
    MPI_Barrier(grid.comm);

    logfullWithPID(dagtm_myid, dagtm_flog);

#ifdef DAGTM_ACTIVEMQ_SERVER_ON
    //--------------------
    // Comptue MAP
    //--------------------
    DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
    result =
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                       mMsub);
    
    DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
    MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                  (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                  MPI_SUM, grid.col_comm);

    message = session->createBytesMessage();
    
    message->writeInt(-2); // LABEL DATA(0), POSITION DATA (1)
    message->writeDouble(1004); // Timestamp
    message->writeInt(nloop); // Stepid
    
    producer->send( message );
    delete message;    

    if (grid.my_rank == 0)
    {
        // Destroy resources.
        try{
            if( producer != NULL ) delete producer;
        }catch ( CMSException& e ) { e.printStackTrace(); }
        producer = NULL;
        
        // Close open resources.
        try{
            if( session != NULL ) session->close();
            if( connection != NULL ) connection->close();
        }catch ( CMSException& e ) { e.printStackTrace(); }
        
        try{
            if( session != NULL ) delete session;
        }catch ( CMSException& e ) { e.printStackTrace(); }
        session = NULL;
        
        try{
            if( connection != NULL ) delete connection;
        }catch ( CMSException& e ) { e.printStackTrace(); }
        connection = NULL;
        
        activemq::library::ActiveMQCPP::shutdownLibrary();        
    }
#endif

    //--------------------
    // Saving output
    //--------------------
    if (filename_h5output != NULL)
    {
        //--------------------
        // Comptue MAP
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Compute MAP ... ");
        result =
            gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, mRsub, mXsub, 0,
                           mMsub);

        DEBUG(DAGTM_INFO_MSG, "Reduce mMsub ... ");
        MPI_Allreduce(MPI_IN_PLACE, mMsub->data,
                      (int) (mMsub->size1 * mMsub->tda), MPI_DOUBLE,
                      MPI_SUM, grid.col_comm);

        //--------------------
        // Write to file
        //--------------------
        DEBUG(DAGTM_INFO_MSG, "Writing to file ... ");

        h5outfileid = h5append(filename_h5output, grid.comm);
        {
            herr_t ret;
            // Write T
            DEBUG(DAGTM_INFO_MSG, "mTsub is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mT, mTsub,
                         N, D, NsplitbyQoffsets[grid.my_col_coord],
                         NbarSplitbyPcounts[grid.my_row_coord],
                         NbarSplitbyPoffsets[grid.my_row_coord]);

            // Write M
            DEBUG(DAGTM_INFO_MSG, "mMsub is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mM, mMsub,
                         N, L, NsplitbyQoffsets[grid.my_col_coord],
                         NbarSplitbyPcounts[grid.my_row_coord],
                         NbarSplitbyPoffsets[grid.my_row_coord]);

            // Write X
            DEBUG(DAGTM_INFO_MSG, "mXsub is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mX, mXsub,
                         K, L, KsplitbyPoffsets[grid.my_row_coord],
                         KbarSplitbyQcounts[grid.my_col_coord],
                         KbarSplitbyQoffsets[grid.my_col_coord]);

            // Write FI
            DEBUG(DAGTM_INFO_MSG, "mFIsub is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mFI, mFIsub,
                         K, M + 1, KsplitbyPoffsets[grid.my_row_coord],
                         KbarSplitbyQcounts[grid.my_col_coord],
                         KbarSplitbyQoffsets[grid.my_col_coord]);

            // Write Y
            DEBUG(DAGTM_INFO_MSG, "mYsub is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mY, mYsub,
                         K, D, KsplitbyPoffsets[grid.my_row_coord],
                         KbarSplitbyQcounts[grid.my_col_coord],
                         KbarSplitbyQoffsets[grid.my_col_coord]);

            // Write W
            DEBUG(DAGTM_INFO_MSG, "mW is being saved ... ");
            h5save_byrow(h5outfileid, dsetname_mW, mW,
                         M + 1, D, 0,
                         MsplitbyRcounts[grid.my_rank],
                         MsplitbyRoffsets[grid.my_rank]);

            if (doSaveR)
            {
                // Write R
                DEBUG(DAGTM_INFO_MSG, "mRsub is being saved ... ");
                h5save_byblock(h5outfileid, dsetname_mR, mRsub,
                               K, N,
                               KsplitbyPoffsets[grid.my_row_coord],
                               NsplitbyQoffsets[grid.my_col_coord]);
            }

            hsize_t dims[2];
            hsize_t viewDims[2];
            hsize_t offsets[2];

            // beta
            DEBUG(DAGTM_INFO_MSG, "beta is being saved ... ");
            dims[0] = 1;
            offsets[0] = 0;
            if (grid.my_rank == 0)
            {
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, dsetname_vbeta,
                                H5T_NATIVE_DOUBLE, 1, dims, dims, offsets,
                                dims, &beta);
            }
            else
            {
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, dsetname_vbeta,
                                H5T_NATIVE_DOUBLE, 1, dims, offsets, dims,
                                offsets, &beta);
            }

            DEBUG(DAGTM_INFO_MSG, "id is being saved ... ");
            hsize_t dim = N;
            hsize_t ldim = NbarSplitbyPcounts[grid.my_row_coord];
            hsize_t offset =
                NsplitbyQoffsets[grid.my_col_coord] +
                NbarSplitbyPoffsets[grid.my_row_coord];

            // id
            if (vidsub != NULL)
            {
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, dsetname_vid,
                                H5T_NATIVE_INT, 1, &dim, &ldim, &offset,
                                &ldim,
                                &(vidsub->
                                  data[NbarSplitbyPoffsets
                                       [grid.my_row_coord]]));
            }

            // lb
            if (vlbsub != NULL)
            {
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, dsetname_vlb,
                                H5T_NATIVE_INT, 1, &dim, &ldim, &offset,
                                &ldim,
                                &(vlbsub->
                                  data[NbarSplitbyPoffsets
                                       [grid.my_row_coord]]));
            }

            if (doLogProgress)
            {
                int *TsplitbyRcounts = MsplitbyRcounts;
                int *TsplitbyRoffsets = MsplitbyRoffsets;

                dagtm_split(TempCollection.next, MPI_NNODE,
                            TsplitbyRcounts, TsplitbyRoffsets);

                if (TsplitbyRcounts[grid.my_rank] == 0)
                {
                    TsplitbyRcounts[grid.my_rank] = TsplitbyRcounts[0];
                    TsplitbyRoffsets[grid.my_rank] = TsplitbyRoffsets[0];
                }

                //dims[1] = 1;
                dims[0] = TempCollection.next;
                //offsets[1] = 0;
                offsets[0] = TsplitbyRoffsets[grid.my_rank];
                //viewDims[1] = 1;
                viewDims[0] = TsplitbyRcounts[grid.my_rank];

                gsl_vector_view view;

                // TempCollection
                DEBUG(DAGTM_INFO_MSG,
                      "TempCollection is being saved ... ");
                view =
                    gsl_vector_subvector(TempCollection.vector,
                                         TsplitbyRoffsets[grid.my_rank],
                                         TsplitbyRcounts[grid.my_rank]);
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, "Temp.dat",
                                H5T_NATIVE_DOUBLE, 1, dims, viewDims,
                                offsets, viewDims, view.vector.data);

                // LLHCollection
                DEBUG(DAGTM_INFO_MSG,
                      "TempCollection is being saved ... ");
                view =
                    gsl_vector_subvector(LLHCollection.vector,
                                         TsplitbyRoffsets[grid.my_rank],
                                         TsplitbyRcounts[grid.my_rank]);
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, "LLH.dat",
                                H5T_NATIVE_DOUBLE, 1, dims, viewDims,
                                offsets, viewDims, view.vector.data);

                // FreeCollection
                DEBUG(DAGTM_INFO_MSG,
                      "TempCollection is being saved ... ");
                view =
                    gsl_vector_subvector(FreeCollection.vector,
                                         TsplitbyRoffsets[grid.my_rank],
                                         TsplitbyRcounts[grid.my_rank]);
                ret =
                    dagtm_h5_rw(FALSE, h5outfileid, "NFree.dat",
                                H5T_NATIVE_DOUBLE, 1, dims, viewDims,
                                offsets, viewDims, view.vector.data);
            }
        }

        H5FCLOSEANDSET(h5outfileid);
    }

    /*
     * Close/release resources.
     */

    gsl_vector_free(vlenN);
    gsl_vector_free(vlenK);
    gsl_vector_free(vlenNsub);
    gsl_vector_free(vlenKsub);
    gsl_vector_free(vgc);

    free(KsplitbyPoffsets);
    free(KsplitbyPcounts);
    free(NsplitbyQoffsets);
    free(NsplitbyQcounts);
    free(NbarSplitbyPcounts);
    free(NbarSplitbyPoffsets);
    free(KbarSplitbyQcounts);
    free(KbarSplitbyQoffsets);
    free(MsplitbyRcounts);
    free(MsplitbyRoffsets);

    gsl_matrix_free(mTsub);
    gsl_matrix_free(mXsub);
    gsl_matrix_free(mYsub);
    gsl_matrix_free(mDsub);
    gsl_matrix_free(mRsub);
    gsl_matrix_free(mRTsub);
    gsl_matrix_free(mMsub);
    gsl_matrix_free(mFIsub);
    gsl_matrix_free(mFIRsub);
    gsl_matrix_free(mFIRTsub);
    gsl_matrix_free(mW);
    //gsl_matrix_free(mMsubReduced);

    //gsl_matrix_free(mgRTsub);
    gsl_matrix_free(mA);
    gsl_matrix_free(mB);
    gsl_matrix_free(mGFIsub);
    gsl_vector_free(vggsub);

    dagtm_ctemp_workspace_free(ws_ctemp);
    dagtm_dist_workspace_free(ws_dist);

    if (doLogProgress)
    {
        dagtm_collection_free(&TempCollection);
        dagtm_collection_free(&LLHCollection);
        dagtm_collection_free(&FreeCollection);
    }
    dagtm_queue_free(&QualQueue);

    //logfprintf(stdout);
    DEBUG(DAGTM_INFO_MSG, "MPI Finalizing ... ");

#ifndef WIN32
    DEBUG(DAGTM_CRITIC_MSG, "Done. (R: %.03f, U: %.03f, S: %.03f)",
          (intmax_t) (en_time - st_time) / (double) clk_tck,
          (intmax_t) (en_cpu.tms_utime -
                      st_cpu.tms_utime) / (double) clk_tck,
          (intmax_t) (en_cpu.tms_stime -
                      st_cpu.tms_stime) / (double) clk_tck);
#endif
    MPI_Finalize();

    if (isLogToFile)
    {
        DAGTM_CLOSELOG();
    }

    return 0;
}


/* handler definition for SIGCHLD */
void sig_handler(int sig_type)
{
    printf("Got signal : %d\n", sig_type);
    isForceCheckpointing = TRUE;
    return;
}

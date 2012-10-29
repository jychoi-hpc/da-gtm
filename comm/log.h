#ifndef __DAGTM_LOG_H__
#define __DAGTM_LOG_H__

#include <stdio.h>
//#include <intrin.h>

#define LOGMAX 100

/**
 *  LOG Definitions
 *  LOG_TICTOC_ON (Default : OFF)
 *  CPU_MHZ (Default : 2500)
 */

#ifdef LOG_TICTOC_ON
# undef LOG_TICTOC_OFF
#else
# define LOG_TICTOC_OFF
#endif

//typedef unsigned short logid_t;
//typedef unsigned short logtype_t;
#ifdef WIN32
typedef unsigned __int64 tick_t;
#else
typedef unsigned long long int tick_t;
#endif

typedef struct {
    tick_t tick;
    tick_t elap;
    unsigned int count;
} logentry;

#define LOG_MAIN 0

#define LOG_RESP 10
#define LOG_RESP_MCPY 11
#define LOG_RESP_SCLE 12
#define LOG_RESP_EXPR 13
#define LOG_RESP_CSUM 14
#define LOG_RESP_CDIV 15
#define LOG_RESP 10
#define LOG_LGLH 20
#define LOG_VGGG 30
#define LOG_ASUB 40
#define LOG_BSUB 50
#define LOG_BSUB_GEM1 51
#define LOG_BSUB_GEM2 52
#define LOG_SOLV 60
#define LOG_BETA 70
#define LOG_BETA_DIST 71
#define LOG_BETA_DIST_GEMM 72
#define LOG_BETA_DIST_DOT1 73
#define LOG_BETA_DIST_DOT2 74
#define LOG_BETA_DIST_GER1 75
#define LOG_BETA_DIST_GER2 76
#define LOG_BETA_MULE 77
#define LOG_BETA_MSUM 78

#define LOG_RESP_COMM 19
#define LOG_LGLH_COMM 29
#define LOG_VGGG_COMM 39
#define LOG_ASUB_COMM 49
#define LOG_BSUB_COMM 59
#define LOG_SOLV_COMM 69
#define LOG_BETA_COMM 79

// Testing...
#define LOG_DIST_GEMM_MLIB 91
#define LOG_DIST_GEMM_MLIB2 92
#define LOG_DIST_GEMM_BASE 93
#define LOG_DIST_GEMM_CCBL 94
#define LOG_DIST_GEMM_STRS 95


/*
#define LOG_COMM 9
#define LOG_RESP 20
#define LOG_RESP_MEMC 21
#define LOG_RESP_COMP 22
#define LOG_RESP_COLS 23
#define LOG_RESP_DIVV 24
#define LOG_RESP_COMM 29
#define LOG_LGLH 30
#define LOG_LGLH_COMM 39
#define LOG_ASUB 40
#define LOG_ASUB_COMM 49
#define LOG_BSUB 50
#define LOG_BSUB_GEM1 51
#define LOG_BSUB_GEM2 52
#define LOG_BSUB_COMM 59
#define LOG_SOLV 60
#define LOG_SOLV_GEM1 61
//#define LOG_SOLV_COMM 69
#define LOG_BETA 70
#define LOG_BETA_DIST 71
#define LOG_BETA_MULT 72
#define LOG_BETA_COMM 79
//#define LOG_GEM1 6
//#define LOG_GEM4 9
*/

#ifdef LOG_TICTOC_ON
# define MPI_TIC(id) \
	MPI_Barrier(MPI_COMM_WORLD);\
	logtic(id)
# define MPI_TOC(id) \
	MPI_Barrier(MPI_COMM_WORLD);\
	logtoc(id)
# define MPI_TICTOC(id, cmd) \
	MPI_Barrier(MPI_COMM_WORLD);\
	logtic(id);\
	cmd;\
	MPI_Barrier(MPI_COMM_WORLD);\
	logtoc(id)
# define LOG_TIC(id) \
	logtic(id)
# define LOG_TOC(id) \
	logtoc(id)
# define LOG_TICTOC(id, cmd) \
	logtic(id);\
	cmd;\
	logtoc(id)
#else
# define MPI_TIC(id)
# define MPI_TOC(id)
# define MPI_TICTOC(id, cmd) cmd
# define LOG_TIC(id)
# define LOG_TOC(id)
# define LOG_TICTOC(id, cmd) cmd
#endif

void logtick(int id);
void logtock(int id);
void logfprintf(FILE * fp, int id);

void logtic(int id);
void logtoc(int id);
void logfprintf(FILE * fp, int id);

void logfull(FILE * fp);
void logfullWithPID(int pid, FILE * fp);

/*
typedef struct {
	logid_t id; 
	logtype_t type; 
	tick_t tick;
} logentry; 

extern FILE *logfp;
extern logentry logdb[];
extern int logidx;
extern int logmyid;

void loginit(int myid);
int loggetmyid();
void loginfo();
void logtick(logid_t id);
void logtock(logid_t id);
void logfprintf(FILE* fp);
tick_t logelap(int e, int s);
*/

#endif                          // __DAGTM_LOG_H__

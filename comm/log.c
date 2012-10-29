#include "log.h"

#ifndef CPU_MHZ
#define CPU_MHZ 2500
//#warning message("Default CPU Frequency is used : 2.5GHz")
#warning Default CPU Frequency is used : 2.5 GHz
#endif

FILE *logfp;
logentry logdb[LOGMAX];
//int logidx;
//int logmyid;

#ifndef WIN32
/*
 * Copied from http://www.mcs.anl.gov/~kazutomo/rdtsc.html
 */
static __inline__ unsigned long long __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc":"=a"(lo), "=d"(hi));
    return ((unsigned long long) lo) | (((unsigned long long) hi) << 32);
}
#endif

void logtic(int id)
{
    logdb[id].tick = __rdtsc();
}

void logtoc(int id)
{
    tick_t tock = __rdtsc();
    logdb[id].elap += tock - logdb[id].tick;
    logdb[id].count += 1;
}

void logfprintf(FILE * fp, int id)
{
    fprintf(fp, "[ID=%05d] Total Ticks : %llu\tIter. : %d\tSecs : %.03f\n",
            id, logdb[id].elap, logdb[id].count,
            (double) logdb[id].elap / (double) CPU_MHZ / 1000000);
}

void logfull(FILE * fp)
{
    for (int id = 0; id < LOGMAX; id++)
    {
        if (logdb[id].count != 0)
        {
            fprintf(fp, "[ID=%05d] Total Ticks : %llu\tIter. : %d\tSecs : %.03f\n",
                    id, logdb[id].elap, logdb[id].count,
                    (double) logdb[id].elap / (double) CPU_MHZ / 1000000);
        }
    }
}

void logfullWithPID(int pid, FILE * fp)
{
    for (int id = 0; id < LOGMAX; id++)
    {
        if (logdb[id].count != 0)
        {
            fprintf(fp, ">>> [%03d] [ID=%05d] Total Ticks : %llu\tIter. : %d\tSecs : %.03f\n",
                    pid, id, logdb[id].elap, logdb[id].count,
                    (double) logdb[id].elap / (double) CPU_MHZ / 1000000);
        }
    }
}

/*
  void loginit(int myid)
  {
  logidx = 0;
  logmyid = myid;
  }

  int loggetmyid()
  {
  return logmyid;
  }

  void loginfo()
  {
  printf("[loginfo] logidx = %d\n", logidx);
  }

  void logtick(logid_t id)
  {
  logdb[logidx].id = id;
  logdb[logidx].type = 0;
  logdb[logidx].tick = __rdtsc();
  logidx ++;
  }

  void logtock(logid_t id)
  {
  logdb[logidx].id = id;
  logdb[logidx].type = 1;
  logdb[logidx].tick = __rdtsc();
  logidx ++;
  }

  void logfprintf(FILE* fp)
  {
  int i;
  int max = (logidx < LOGMAX)? logidx : LOGMAX;
  for (i = 0; i < max ; i++)
  {
  fprintf(fp, "[PID=%05d] [EVENT=%05d] [%d] %I64d\n", logmyid, logdb[i].id, logdb[i].type, logdb[i].tick);
  }
  }

  tick_t logelap(int e, int s)
  {
  return logdb[e].tick - logdb[s].tick;
  }
*/

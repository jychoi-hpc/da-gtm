/*
 *  h5comm.h
 *  comm
 *
 *  Created by jychoi on 7/6/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __DAGTM_H5COMM_H__
#define __DAGTM_H5COMM_H__
#include "hdf5.h"

/**
 *  H5COMM Definitions
 *  H5COMM_MPIIO_OFF : Default ON
 */

#ifdef H5COMM_MPIIO_OFF
# undef H5COMM_MPIIO_ON
#else
# define H5COMM_MPIIO_ON
#endif

herr_t dagtm_h5_rw(const int type, const hid_t file_id,
                   const char *dsetname, const hid_t dtype, const int rank,
                   const hsize_t * dimsf, const hsize_t * count,
                   hsize_t * offset, const hsize_t * memdim, void *data);
herr_t dagtm_h5_read(const hid_t file_id, const char *dsetname,
                     const hid_t dtype, const int rank,
                     const hsize_t * dimsf, const hsize_t * count,
                     const hsize_t * offset, const hsize_t * memdim,
                     void *data);
herr_t dagtm_h5_write(const hid_t file_id, const char *dsetname,
                      const hid_t dtype, const int rank,
                      const hsize_t * dimsf, const hsize_t * count,
                      const hsize_t * offset, const hsize_t * memdim,
                      void *data);
herr_t dagtm_h5_dim(const hid_t fileid, const char *datasetname,
                    const int rank, hsize_t * dims);

#endif                          //__DAGTM_H5COMM_H__

/*
 *  h5comm.c
 *  comm
 *
 *  Created by jychoi on 7/6/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "h5comm.h"
#include "comm.h"
#define TRUE 1
#define FALSE 0

/**
 *  HDF5 read/write
 *
 *  @param[in] type 1(read) or 0(write)
 *  @param[in] file_id hdf5 file id
 *  @param[in] dsetname
 *  @param[in] dtype
 *  @param[in] rank
 *  @param[in] dimsf full dataset dimension
 *  @param[in] count subset dimension
 *  @param[in] offset offsets of subset data
 *  @param[in] memdim memory dimension
 *  @param[in] data
 */
herr_t dagtm_h5_rw(const int type, const hid_t file_id,
                   const char *dsetname, const hid_t dtype, const int rank,
                   const hsize_t dimsf[rank], const hsize_t count[rank],
                   hsize_t offset[rank], const hsize_t memdim[rank],
                   void *data)
{
    hid_t dataset;              /* file and dataset identifiers */
    hid_t dataspace, memspace;  /* file and memory dataspace identifiers */
    hid_t property;             /* property list identifier */
    herr_t ret;
    int isValidCount = TRUE;
    int doProcess = TRUE;

    if (type)
    {
        if (H5Lexists(file_id, dsetname, H5P_DEFAULT) == FALSE)
        {
            DEBUG(DAGTM_WARNING_MSG, "No %s exists", dsetname);
            return FAIL;
        }

        dataset = H5Dopen(file_id, dsetname);
        dataspace = H5Dget_space(dataset);
    }
    else
    {
        dataspace = H5Screate_simple(rank, dimsf, NULL);
        dataset =
            H5Dcreate(file_id, dsetname, dtype, dataspace, H5P_DEFAULT);
        //H5Sclose(dataspace);
    }

    for (int i = 0; i < rank; i++)
    {
        if (count[i] < 1)
        {
            isValidCount = FALSE;
            break;
        }
    }

    /*
     * Hyperslab Example: 
     *
     * Select hyperslab for the dataset in the file, using 3x2 blocks, 
     * (4,3) stride and (2,4) count starting at the position (0,1). 
     *
     *                    0  1  2  0  3  4  0  5  6  0  7  8 
     *                    0  9 10  0 11 12  0 13 14  0 15 16
     *                    0 17 18  0 19 20  0 21 22  0 23 24
     *                    0  0  0  0  0  0  0  0  0  0  0  0
     *                    0 25 26  0 27 28  0 29 30  0 31 32
     *                    0 33 34  0 35 36  0 37 38  0 39 40
     *                    0 41 42  0 43 44  0 45 46  0 47 48
     *                    0  0  0  0  0  0  0  0  0  0  0  0
     */
    if (isValidCount)
    {
        memspace = H5Screate_simple(rank, memdim, NULL);
        hsize_t blockcount_[2] = { 1, 1 };
        hsize_t offset_[2] = { 0, 0 };

        H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_, NULL,
                            blockcount_, count);

        //dataspace = H5Dget_space(dataset);
        ret =
            H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL,
                                blockcount_, count);

    }
    else
    {
        H5Sselect_none(dataspace);
        memspace = H5Scopy(dataspace);
        H5Sselect_none(memspace);
    }

    /*
     * Create property list for collective dataset write.
     */
    property = H5Pcreate(H5P_DATASET_XFER);

#ifdef H5COMM_MPIIO_ON
    /*
     * Use Collective IO : Need to participate all processes
     */
    ret = H5Pset_dxpl_mpio(property, H5FD_MPIO_COLLECTIVE);
    doProcess = TRUE;
#else
    /*
     * Independent IO. Don't need to participate all processes
     * Result is very unreliable (Please don't use)
     */
#warning "MPI-IO is disabled. Please use MPI-IO"    
    doProcess = isValidCount;
#endif

    if (doProcess)
    {
        if (type)
        {
            ret =
                H5Dread(dataset, dtype, memspace, dataspace, property,
                        data);
        }
        else
        {
            ret =
                H5Dwrite(dataset, dtype, memspace, dataspace, property,
                         data);
        }
    }

    if (isValidCount)
    {
        H5Sclose(dataspace);
        H5Sclose(memspace);
    }

    H5Pclose(property);
    H5Dclose(dataset);

    return ret;
}

herr_t dagtm_h5_read(const hid_t file_id, const char *dsetname,
                     const hid_t dtype, const int rank,
                     const hsize_t * dimsf, const hsize_t * count,
                     const hsize_t * offset, const hsize_t * memdim,
                     void *data)
{
    return dagtm_h5_rw(TRUE, file_id, dsetname, dtype,
                       rank, dimsf, count, (hsize_t *) offset, memdim,
                       data);
}

herr_t dagtm_h5_write(const hid_t file_id, const char *dsetname,
                      const hid_t dtype, const int rank,
                      const hsize_t * dimsf, const hsize_t * count,
                      const hsize_t * offset, const hsize_t * memdim,
                      void *data)
{
    return dagtm_h5_rw(FALSE, file_id, dsetname, dtype,
                       rank, dimsf, count, (hsize_t *) offset, memdim,
                       data);
}

herr_t dagtm_h5_dim(const hid_t fileid, const char *datasetname,
                    const int rank, hsize_t * dims)
{
    hid_t dataset, dataspace;
    herr_t ret;
    int _rank;

    dataset = H5Dopen(fileid, datasetname);
    dataspace = H5Dget_space(dataset);
    _rank = H5Sget_simple_extent_ndims(dataspace);

    if (_rank != rank)
        return -1;

    ret = H5Sget_simple_extent_dims(dataspace, dims, NULL);

    /*
     * Close/release resources.
     */
    H5Sclose(dataspace);
    H5Dclose(dataset);

    return ret;
}

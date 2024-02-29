/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpioimpl.h"

#ifdef HAVE_WEAK_SYMBOLS

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_File_close = PMPI_File_close
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_File_close MPI_File_close
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_File_close as PMPI_File_close
/* end of weak pragmas */
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_File_close(MPI_File * fh) __attribute__ ((weak, alias("PMPI_File_close")));
#endif

/* Include mapping from MPI->PMPI */
#define MPIO_BUILD_PROFILING
#include "mpioprof.h"
#endif

/*@
    MPI_File_close - Closes a file

Input Parameters:
. fh - file handle (handle)

.N fortran
@*/
int MPI_File_close(MPI_File * fh)
{
    int error_code;
    ADIO_File adio_fh;
    static char myname[] = "MPI_FILE_CLOSE";
#ifdef MPI_hpux
    int fl_xmpi;

    HPMP_IO_WSTART(fl_xmpi, BLKMPIFILECLOSE, TRDTBLOCK, *adio_fh);
#endif /* MPI_hpux */

    ROMIO_THREAD_CS_ENTER();

    adio_fh = MPIO_File_resolve(*fh);

#ifdef WKL_DEBUG
MPI_Comm tmp_comm;
char *filename = strdup(adio_fh->filename);
MPI_Comm_dup((adio_fh)->comm, &tmp_comm);
#endif

    /* --BEGIN ERROR HANDLING-- */
    MPIO_CHECK_FILE_HANDLE(adio_fh, myname, error_code);
    /* --END ERROR HANDLING-- */

    if (ADIO_Feature(adio_fh, ADIO_SHARED_FP)) {
        ADIOI_Free((adio_fh)->shared_fp_fname);
        /* POSIX semantics say a deleted file remains available until all
         * processes close the file.  But since when was NFS posix-compliant?
         */
        /* this used to be gated by the lack of the UNLINK_AFTER_CLOSE feature,
         * but a race condition in GPFS necessated this.  See ticket #2214 */
        MPI_Barrier((adio_fh)->comm);
        if ((adio_fh)->shared_fp_fd != ADIO_FILE_NULL) {
            ADIO_File *fh_shared = &(adio_fh->shared_fp_fd);
            ADIO_Close((adio_fh)->shared_fp_fd, &error_code);
            MPIO_File_free(fh_shared);
            /* --BEGIN ERROR HANDLING-- */
            if (error_code != MPI_SUCCESS)
                goto fn_fail;
            /* --END ERROR HANDLING-- */
        }
    }

    /* Because ROMIO expects the MPI library to provide error handler management
     * routines but it doesn't ever participate in MPI_File_close, we have to
     * somehow inform the MPI library that we no longer hold a reference to any
     * user defined error handler.  We do this by setting the errhandler at this
     * point to MPI_ERRORS_RETURN. */
    error_code = MPI_File_set_errhandler(*fh, MPI_ERRORS_RETURN);
    if (error_code != MPI_SUCCESS)
        goto fn_fail;

    ADIO_Close(adio_fh, &error_code);
    MPIO_File_free(&adio_fh);
    /* --BEGIN ERROR HANDLING-- */
    if (error_code != MPI_SUCCESS)
        goto fn_fail;
    /* --END ERROR HANDLING-- */

#ifdef MPI_hpux
    HPMP_IO_WEND(fl_xmpi);
#endif /* MPI_hpux */

  fn_exit:
    ROMIO_THREAD_CS_EXIT();

#ifdef WKL_DEBUG
int myrank;
MPI_Comm_rank(tmp_comm, &myrank);
MPI_Offset mem_size, max_size, min_size;
wkl_inq_malloc_max_size(&mem_size);
MPI_Reduce(&mem_size, &max_size, 1, MPI_OFFSET, MPI_MAX, 0, tmp_comm);
MPI_Reduce(&mem_size, &min_size, 1, MPI_OFFSET, MPI_MIN, 0, tmp_comm);
if (myrank == 0) printf("%s: %s malloc high watermark = %lld (max=%.2f min=%.2f MB)\n",__func__,filename,max_size,(float)max_size/1048576.0,(float)min_size/1048576.0);
wkl_inq_malloc_size(&mem_size);
if (mem_size > 0) {
    printf("rank %d: %s file %s mem_size=%lld > 0\n",myrank,__func__,filename,mem_size);
    wkl_inq_malloc_list();
}
/*
extern int ADIOI_Flattened_type_keyval;
if (ADIOI_Flattened_type_keyval != MPI_KEYVAL_INVALID)
    MPI_Type_free_keyval(&ADIOI_Flattened_type_keyval);
*/

extern int flat_mem, flat_hits, flat_miss;
int max_flat, min_flat;
MPI_Reduce(&flat_mem, &max_flat, 1, MPI_INT, MPI_MAX, 0, tmp_comm);
MPI_Reduce(&flat_mem, &min_flat, 1, MPI_INT, MPI_MIN, 0, tmp_comm);
if (myrank == 0) {
    printf("%s: %s flat hits=%d miss=%d malloc=%d (max=%.2f min=%.2f MB)\n",__func__,filename,flat_hits,flat_miss,flat_mem,(float)max_flat/1048576.0,(float)min_flat/1048576.0);
    fflush(stdout);
}
free(filename);
MPI_Comm_free(&tmp_comm);
#endif

    return error_code;
  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    error_code = MPIO_Err_return_file(adio_fh, error_code);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

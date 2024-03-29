/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests collective read and write calls using a noncontiguous
 * user buffer datatype, which consists of two contiguous blocks separated by a
 * gap. This is to test how many memcpy is called in
 * ADIOI_LUSTRE_Fill_send_buffer().
 *
 * For large read/write requests, if the Lustre striping size is small, then
 * the number of memcpy can become large, hurting the performance.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* getopt() */

#include <mpi.h>

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        nerrs++; \
        goto err_out; \
    }

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-h | -q | -n num | file_name]\n"
    "       [-h] Print this help\n"
    "       [-q] quiet mode\n"
    "       [-n num] length of each contiguous write\n"
    "       [filename] output file name\n";
    fprintf(stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
#ifdef ROMIO_INSIDE_MPICH
    return 0;
#else
    extern int optind;
    extern char *optarg;
    char filename[256];
    int i, err, nerrs=0, rank, mode, verbose=1, count, blocklen[2];
    char *buf;
    double timing, max_timing;
    MPI_Aint displace[2], extent;
    MPI_Datatype dtype;
    MPI_File fh;
    MPI_Offset off;
    MPI_Status status;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    count = 4; /* default block size */

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hqn:")) != EOF)
        switch(i) {
            case 'q': verbose = 0;
                      break;
            case 'n': count = atoi(optarg);
                      break;
            case 'h':
            default:  if (rank==0) usage(argv[0]);
                      MPI_Finalize();
                      return 1;
        }
    if (argv[optind] == NULL)
        sprintf(filename, "%s.out", argv[0]);
    else
        snprintf(filename, 256, "%s", argv[optind]);

    if (verbose && rank == 0) {
        printf("Creating a buffer datatype consisting 2 blocks, separated by 4 bytes\n");
        printf("Each block is of size %d bytes\n", count);
    }

    /* create a datatype consists of two blocks, with a 4-byte gap in between */
    blocklen[0] = count;
    blocklen[1] = count;
    displace[0] = 0;
    displace[1] = count+4;
    err = MPI_Type_create_hindexed(2, blocklen, displace, MPI_BYTE, &dtype); ERR
    err = MPI_Type_commit(&dtype); ERR

    mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh); ERR

    /* each process writes to a non-overlapped file space */
    off = (count * 2 + 4) * rank;

    /* allocate I/O buffer */
    MPI_Type_extent(dtype, &extent);
    buf = (char*) calloc(extent, 1);

    /* write to the file */
    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();
    err = MPI_File_write_at_all(fh, off, buf, 1, dtype, &status); ERR
    err = MPI_File_read_at_all(fh, off, buf, 1, dtype, &status); ERR
    timing = MPI_Wtime() - timing;

    err = MPI_File_close(&fh); ERR
    err = MPI_Type_free(&dtype); ERR
    free(buf);

    MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Time of collective write and read = %.2f sec\n", max_timing);

err_out:
    MPI_Finalize();
    return (nerrs > 0);
#endif
}


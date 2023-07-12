/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "adio.h"

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

static int construct_aggr_list(ADIO_File fd, int root, int *error_code);

/* Generic version of a "collective open".  Assumes a "real" underlying
 * file system (meaning no wonky consistency semantics like NFS).
 *
 * optimization: by having just one process create a file, close it,
 * then have all N processes open it, we can possibly avoid contention
 * for write locks on a directory for some file systems.
 *
 * Happy side-effect: exclusive create (error if file already exists)
 * just falls out
 *
 * Note: this is not a "scalable open" (c.f. "The impact of file systems
 * on MPI-IO scalability").
 */

/* generate an MPI datatype describing the members of the ADIO_File struct that
 * we want to ensure all processes have.  In deferred open, aggregators will
 * open the file and possibly read layout and other information.
 * non-aggregators will skip the open, but still need to know how the file is
 * being treated and what optimizations to apply */

static MPI_Datatype make_stats_type(ADIO_File fd)
{
    enum file_stats {
        BLOCKSIZE = 0,
        STRIPE_SIZE,
        STRIPE_FACTOR,
        START_IODEVICE,
        STAT_ITEMS
    };

    int lens[STAT_ITEMS];
    MPI_Aint offsets[STAT_ITEMS];
    MPI_Datatype types[STAT_ITEMS];
    MPI_Datatype newtype;

    lens[BLOCKSIZE] = 1;
    MPI_Get_address(&fd->blksize, &offsets[BLOCKSIZE]);
    types[BLOCKSIZE] = MPI_LONG;

    lens[STRIPE_SIZE] = lens[STRIPE_FACTOR] = lens[START_IODEVICE] = 1;
    types[STRIPE_SIZE] = types[STRIPE_FACTOR] = types[START_IODEVICE] = MPI_INT;
    MPI_Get_address(&fd->hints->striping_unit, &offsets[STRIPE_SIZE]);
    MPI_Get_address(&fd->hints->striping_factor, &offsets[STRIPE_FACTOR]);
    MPI_Get_address(&fd->hints->start_iodevice, &offsets[START_IODEVICE]);


    MPI_Type_create_struct(STAT_ITEMS, lens, offsets, types, &newtype);
    MPI_Type_commit(&newtype);
    return newtype;

}

/*
 *   1. root creates/opens the file
 *      a. root sets/obtains striping info
 *      b. root closes file
 *   2. root determines cb_nodes and ranklist
 *      a. all processes send root its proc_name
 *      b. root broadcasts cb_nodes and ranklist
 *   3. When deferred_open is true:
 *      then only aggregators open the file
 *      else only aggregators open the file
 *   4. all processes sync and set file striping info
 *      a. root bcasts striping info
 *      b. all processes set hints
 */
void ADIOI_GEN_OpenColl(ADIO_File fd, int rank, int access_mode, int *error_code)
{
    char value[MPI_MAX_INFO_VAL + 1];
    int i, root, orig_amode_excl, orig_amode_wronly;
    MPI_Comm tmp_comm;
    MPI_Datatype stats_type;    /* deferred open: some processes might not
                                 * open the file, so we'll exchange some
                                 * information with those non-aggregators */

    orig_amode_excl = access_mode;

    root = (fd->hints->ranklist == NULL) ? 0 : fd->hints->ranklist[0];

    if ((access_mode & ADIO_CREATE) || (fd->file_system == ADIO_LUSTRE)) {
        /* root process creates the file first, followed by all processes open
         * the file.
         * For Lustre, we need to obtain file striping info (striping_factor,
         * striping_unit, and num_osts) in order to select the I/O aggregators
         * in fd->hints->ranklist, no matter its is open or create mode.
         */

        if (rank == root) {
            /* remove delete_on_close flag if set */
            if (access_mode & ADIO_DELETE_ON_CLOSE)
                fd->access_mode = access_mode ^ ADIO_DELETE_ON_CLOSE;
            else
                fd->access_mode = access_mode;

            tmp_comm = fd->comm;
            fd->comm = MPI_COMM_SELF;
            (*(fd->fns->ADIOI_xxx_Open)) (fd, error_code);
            fd->comm = tmp_comm;
            MPI_Bcast(error_code, 1, MPI_INT, root, fd->comm);
            /* if no error, close the file and reopen normally below */
            if (*error_code == MPI_SUCCESS)
                (*(fd->fns->ADIOI_xxx_Close)) (fd, error_code);

            fd->access_mode = access_mode;      /* back to original */
        } else
            MPI_Bcast(error_code, 1, MPI_INT, root, fd->comm);

        if (*error_code != MPI_SUCCESS) {
            return;
        } else {
            /* turn off CREAT (and EXCL if set) for real multi-processor open */
            if (access_mode & ADIO_CREATE)
                access_mode ^= ADIO_CREATE;
            if (access_mode & ADIO_EXCL)
                access_mode ^= ADIO_EXCL;
        }

        if (fd->file_system == ADIO_LUSTRE) {
            /* Use file striping count and number of unique OSTs to construct
             * the I/O aggregator rank list, fd->hints->ranklist[].
             */
            construct_aggr_list(fd, root, error_code);
            if (*error_code != MPI_SUCCESS)
                return;
        }
    }

    fd->blksize = 1024 * 1024 * 4;
    /* this large default value should be good for most file systems. any ROMIO
     * driver is free to stat the file and find an optimal value */

    /* add to fd->info the hint "aggr_list", list of aggregators' rank IDs */
    value[0] = '\0';
    for (i = 0; i < fd->hints->cb_nodes; i++) {
        char str[16];
        if (i == 0)
            MPL_snprintf(str, sizeof(value), "%d", fd->hints->ranklist[i]);
        else
            MPL_snprintf(str, sizeof(value), " %d", fd->hints->ranklist[i]);
        if (strlen(value) + strlen(str) >= MPI_MAX_INFO_VAL-5) {
            strcat(value, " ...");
            break;
        }
        strcat(value, str);
    }
    ADIOI_Info_set(fd->info, "aggr_list", value);

    /* if we are doing deferred open, non-aggregators should return now */
    if (fd->hints->deferred_open) {
        if (!(fd->is_agg)) {
            /* we might have turned off EXCL for the aggregators.
             * restore access_mode that non-aggregators get the right
             * value from get_amode */
            fd->access_mode = orig_amode_excl;

            /* In file-system specific open, a driver might collect some
             * information via stat().  Deferred open means not every process
             * participates in fs-specific open, but they all participate in
             * this open call.  Broadcast a bit of information in case
             * lower-level file system driver (e.g. 'bluegene') collected it
             * (not all do)*/
            stats_type = make_stats_type(fd);
            MPI_Bcast(MPI_BOTTOM, 1, stats_type, root, fd->comm);
            ADIOI_Assert(fd->blksize > 0);

            /* set file striping hints */
            MPL_snprintf(value, sizeof(value), "%d", fd->hints->striping_unit);
            ADIOI_Info_set(fd->info, "striping_unit", value);

            MPL_snprintf(value, sizeof(value), "%d", fd->hints->striping_factor);
            ADIOI_Info_set(fd->info, "striping_factor", value);

            MPL_snprintf(value, sizeof(value), "%d", fd->hints->start_iodevice);
            ADIOI_Info_set(fd->info, "start_iodevice", value);

            *error_code = MPI_SUCCESS;
            MPI_Type_free(&stats_type);
            return;
        }
    }

/* For writing with data sieving, a read-modify-write is needed. If
   the file is opened for write_only, the read will fail. Therefore,
   if write_only, open the file as read_write, but record it as write_only
   in fd, so that get_amode returns the right answer. */

    /* observation from David Knaak: file systems that do not support data
     * sieving do not need to change the mode */

    orig_amode_wronly = access_mode;
    if ((access_mode & ADIO_WRONLY) && ADIO_Feature(fd, ADIO_DATA_SIEVING_WRITES)) {
        access_mode = access_mode ^ ADIO_WRONLY;
        access_mode = access_mode | ADIO_RDWR;
    }
    fd->access_mode = access_mode;

    (*(fd->fns->ADIOI_xxx_Open)) (fd, error_code);

    /* if error, may be it was due to the change in amode above.
     * therefore, reopen with access mode provided by the user. */
    fd->access_mode = orig_amode_wronly;
    if (*error_code != MPI_SUCCESS)
        (*(fd->fns->ADIOI_xxx_Open)) (fd, error_code);

    /* if we turned off EXCL earlier, then we should turn it back on */
    if (fd->access_mode != orig_amode_excl)
        fd->access_mode = orig_amode_excl;

    /* broadcast information to all processes in communicator, not just
     * those who participated in open.
     */
    stats_type = make_stats_type(fd);
    MPI_Bcast(MPI_BOTTOM, 1, stats_type, root, fd->comm);
    MPI_Type_free(&stats_type);
    /* file domain code will get terribly confused in a hard-to-debug way
     * if gpfs blocksize not sensible */
    ADIOI_Assert(fd->blksize > 0);

    /* set file striping hints */
    MPL_snprintf(value, sizeof(value), "%d", fd->hints->striping_unit);
    ADIOI_Info_set(fd->info, "striping_unit", value);

    MPL_snprintf(value, sizeof(value), "%d", fd->hints->striping_factor);
    ADIOI_Info_set(fd->info, "striping_factor", value);

    MPL_snprintf(value, sizeof(value), "%d", fd->hints->start_iodevice);
    ADIOI_Info_set(fd->info, "start_iodevice", value);

    /* for deferred open: this process has opened the file (because if we are
     * not an aggregator and we are doing deferred open, we returned earlier)*/
    fd->is_open = 1;

    /* sync optimization: we can omit the fsync() call if we do no writes */
    fd->dirty_write = 0;
}

/*----< construct_aggr_list() >----------------------------------------------*/
/* Allocate and construct fd->hints->ranklist[].
 * Overwrite fd->hints->cb_nodes and set hint cb_nodes.
 * Set fd->is_agg, whether this rank is an I/O aggregator
 *     fd->hints->cb_nodes
 *     fd->hints->fs_hints.lustre.num_osts
 */
static int construct_aggr_list(ADIO_File fd, int root, int *error_code)
{
    int i, j, k, rank, nprocs, num_aggr, my_procname_len, num_nodes;
    int msg[2], striping_factor;
    int *all_procname_lens = NULL;
    int *nprocs_per_node, **ranks_per_node;
    char value[MPI_MAX_INFO_VAL + 1], my_procname[MPI_MAX_PROCESSOR_NAME];
    char **all_procnames = NULL;
    static char myname[] = "ADIO_OPENCOLL construct_aggr_list";

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &rank);

    /* Collect info about compute nodes in order to select I/O aggregators.
     * Note my_procname is null character terminated, but my_procname_len
     * does not include the null character.
     */
    MPI_Get_processor_name(my_procname, &my_procname_len);

    if (rank == root) {
        /* root collects all procnames */
        all_procnames = (char **) ADIOI_Malloc(nprocs * sizeof(char *));
        if (all_procnames == NULL) {
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                               __LINE__, MPI_ERR_OTHER,
                                               "**nomem2", 0);
            return 0;
        }

        all_procname_lens = (int *) ADIOI_Malloc(nprocs * sizeof(int));
        if (all_procname_lens == NULL) {
            ADIOI_Free(all_procnames);
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                                __LINE__, MPI_ERR_OTHER,
                                                "**nomem2", 0);
            return 0;
        }
    }
    /* gather process name lengths from all processes first */
    MPI_Gather(&my_procname_len, 1, MPI_INT, all_procname_lens, 1, MPI_INT,
               root, fd->comm);

    if (rank == root) {
        int *disp;
        size_t alloc_size = 0;

        for (i = 0; i < nprocs; i++)
            /* Must include the null terminate character */
            alloc_size += ++all_procname_lens[i];

        all_procnames[0] = (char *) ADIOI_Malloc(alloc_size);
        if (all_procnames[0] == NULL) {
            ADIOI_Free(all_procname_lens);
            ADIOI_Free(all_procnames);
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                               __LINE__, MPI_ERR_OTHER,
                                               "**nomem2", 0);
            return 0;
        }

        /* Construct displacement array for the MPI_Gatherv, as each process
         * may have a different length for its process name.
         */
        disp = (int *) ADIOI_Malloc(nprocs * sizeof(int));
        disp[0] = 0;
        for (i = 1; i < nprocs; i++) {
            all_procnames[i] = all_procnames[i - 1] + all_procname_lens[i - 1];
            disp[i] = disp[i - 1] + all_procname_lens[i - 1];
        }

        /* gather all process names, each includes the null terminate character */
        MPI_Gatherv(my_procname, my_procname_len + 1, MPI_CHAR,
                    all_procnames[0], all_procname_lens, disp, MPI_CHAR,
                    root, fd->comm);

        ADIOI_Free(disp);
        ADIOI_Free(all_procname_lens);
    } else
        /* send process name, including the null terminate character */
        MPI_Gatherv(my_procname, my_procname_len + 1, MPI_CHAR,
                    NULL, NULL, NULL, MPI_CHAR, root, fd->comm);

    if (rank == root) {
        /* all_procnames[] can tell us the number of nodes and number of
         * processes per node.
         */
        char **node_names;
        int last, *node_ids;

        /* number of MPI processes running on each node */
        nprocs_per_node = (int *) ADIOI_Malloc(nprocs * sizeof(int));

        /* ech MPI process's compute node ID */
        node_ids = (int *) ADIOI_Malloc(nprocs * sizeof(int));

        /* array of pointers pointing to unique host names (compute nodes) */
        node_names = (char **) ADIOI_Malloc(nprocs * sizeof(char *));

        /* calculate nprocs_per_node[] and node_ids[] */
        last = 0;
        num_nodes = 0; /* number of unique compute nodes */
        for (i = 0; i < nprocs; i++) {
            k = last;
            for (j = 0; j < num_nodes; j++) {
                /* check if [i] has already appeared in [] */
                if (!strcmp(all_procnames[i], node_names[k])) { /* found */
                    node_ids[i] = k;
                    nprocs_per_node[k]++;
                    break;
                }
                k = (k == num_nodes - 1) ? 0 : k + 1;
            }
            if (j < num_nodes)  /* found, next iteration, start with node n */
                last = k;
            else {      /* not found, j == num_nodes, add a new node */
                node_names[j] = ADIOI_Strdup(all_procnames[i]);
                nprocs_per_node[j] = 1;
                node_ids[i] = j;
                last = j;
                num_nodes++;
            }
        }
        /* num_nodes is now the number of compute nodes (unique node names) */

        for (i = 0; i < num_nodes; i++)
            ADIOI_Free(node_names[i]);
        ADIOI_Free(node_names);
        ADIOI_Free(all_procnames[0]);
        ADIOI_Free(all_procnames);

        /* construct rank IDs of MPI processes running on each node */
        ranks_per_node = (int **) ADIOI_Malloc(num_nodes * sizeof(int *));
        ranks_per_node[0] = (int *) ADIOI_Malloc(nprocs * sizeof(int));
        for (i = 1; i < num_nodes; i++)
            ranks_per_node[i] = ranks_per_node[i - 1] + nprocs_per_node[i - 1];
        for (i = 0; i < num_nodes; i++)
            nprocs_per_node[i] = 0;

        /* Populate ranks_per_node[], list of MPI ranks running on each node.
         * Populate nprocs_per_node[], number of MPI processes on each node.
         */
        for (i = 0; i < nprocs; i++) {
            k = node_ids[i];
            ranks_per_node[k][nprocs_per_node[k]] = i;
            nprocs_per_node[k]++;
        }
        ADIOI_Free(node_ids);

        /* Given the number of nodes, num_nodes, and processes per node,
         * nprocs_per_node, we can now set num_aggr, the number of I/O
         * aggregators. At this moment, root should have obtained the file
         * striping settings.
         */
        striping_factor = fd->hints->striping_factor;

        if (striping_factor > nprocs) {
            /* When number of MPI processes is less than striping_factor, set
             * num_aggr to the max number less than nprocs that divides
             * striping_factor. An naive way is:
             *     num_aggr = nprocs;
             *     while (striping_factor % num_aggr > 0)
             *         num_aggr--;
             * Below is equivalent, but faster.
             */
            int divisor = 2;
            num_aggr = 1;
            /* try to divide */
            while (striping_factor >= divisor * divisor) {
                if ((striping_factor % divisor) == 0) {
                    if (striping_factor / divisor <= nprocs) {
                        /* The value is found ! */
                        num_aggr = striping_factor / divisor;
                        break;
                    }
                    /* if divisor is less than nprocs, divisor is a solution,
                     * but it is not sure that it is the best one
                     */
                    else if (divisor <= nprocs)
                        num_aggr = divisor;
                }
                divisor++;
            }
        }
        else { /* striping_factor <= nprocs */
            /* Select striping_factor processes to be I/O aggregators */
            // if (fd->hints->cb_nodes == 0) /* hint cb_nodes is not set by user */
if (fd->hints->cb_nodes == 0 || fd->access_mode & ADIO_RDONLY) {
/* for now, do not mess up ranklist for read operations */
printf("ADIO_RDONLY file %s\n",fd->filename);
                num_aggr = fd->hints->striping_factor;
}
            else if (fd->hints->cb_nodes <= striping_factor)
                /* User has set hint cb_nodes and cb_nodes <= striping_factor.
                 * Ignore user's hint and set cb_nodes to striping_factor. */
                num_aggr = striping_factor;
            else {
                /* User has set hint cb_nodes and cb_nodes > striping_factor */
                if (nprocs < fd->hints->cb_nodes)
                    num_aggr = nprocs; /* BAD cb_nodes set by users */
                else
                    num_aggr = fd->hints->cb_nodes;

                /* Number of processes per node may not be enough to be picked
                 * as aggregators. If this case, reduce num_aggr (cb_nodes).
                 * Consider the following case: number of processes = 18,
                 * number of nodes = 7, striping_factor = 8, cb_nodes = 16.
                 * cb_nodes should be reduced to 8 and the ranks of aggregators
                 * should be 0, 3, 6, 9, 12, 14, 16, 1.
                 * If the number of processes changes to 25, then cb_nodes
                 * should be 16 and the ranks of aggregators should be 0, 4, 8,
                 * 12, 16, 19, 22, 1, 2, 6, 10, 14, 18, 21, 24, 3.
                 */
                int max_nprocs_node = 0;
                for (i = 0; i < num_nodes; i++)
                    max_nprocs_node = MAX(max_nprocs_node, nprocs_per_node[i]);
                int max_naggr_node = striping_factor / num_nodes;
                if (striping_factor % num_nodes) max_naggr_node++;
                /* max_naggr_node is the max number of processes per node to be
                 * picked as aggregator in each round.
                 */
                int rounds = num_aggr / striping_factor;
                if (num_aggr % striping_factor) rounds++;
                while (max_naggr_node * rounds > max_nprocs_node) rounds--;
                num_aggr = striping_factor * rounds;
            }
        }

        /* TODO: the above setting for num_aggr is for collective writes. Reads
         * should be the number of nodes.
         */

        /* Next step is to determine the MPI rank IDs of I/O aggregators into
         * ranklist[].
         */
        fd->hints->ranklist = (int *) ADIOI_Malloc(num_aggr * sizeof(int));
        if (fd->hints->ranklist == NULL) {
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                               __LINE__, MPI_ERR_OTHER,
                                               "**nomem2", 0);
            return 0;
        }

        if (striping_factor <= num_nodes) {
            /* When number of OSTs is less than number of compute nodes,
             * first select number of nodes equal to the number of OSTs by
             * spread the selection evenly across all compute nodes and then
             * pick processes from the selected nodes, also evenly spread
             * evenly among processes on each selected node to be aggregators.
             */
            int avg = num_aggr / striping_factor;
            int stride = num_nodes / striping_factor;
            if (num_aggr % striping_factor) avg++;
            for (i = 0; i < num_aggr; i++) {
                j = (i % striping_factor) * stride; /* to select from node j */
                k = (i / striping_factor) * (nprocs_per_node[j] / avg);
                assert(k < nprocs_per_node[j]);
                fd->hints->ranklist[i] = ranks_per_node[j][k];
            }
        }
        else { /* striping_factor > num_nodes */
            /* When number of OSTs is more than number of compute nodes, I/O
             * aggregators are selected from all nodes are selected. Within
             * each node, aggregators are spread evenly instead of the first
             * few ranks.
             */
            int *naggr_per_node, *idx_per_node, avg;
            idx_per_node = (int*) ADIOI_Calloc(num_nodes, sizeof(int));
            naggr_per_node = (int*) ADIOI_Malloc(num_nodes * sizeof(int));
            for (i = 0; i < striping_factor % num_nodes; i++)
                naggr_per_node[i] = striping_factor / num_nodes + 1;
            for (; i < num_nodes; i++)
                naggr_per_node[i] = striping_factor / num_nodes;
            avg = num_aggr / striping_factor;
            if (avg > 0)
                for (i = 0; i < num_nodes; i++)
                    naggr_per_node[i] *= avg;
            for (i = 0; i < num_nodes; i++)
                naggr_per_node[i] = MIN(naggr_per_node[i], nprocs_per_node[i]);
            /* naggr_per_node[] is the number of aggregators that can be
             * selected as I/O aggregators
             */

printf("%d: num_nodes=%d nprocs_per_node[0]=%d num_aggr=%d naggr_per_node[0]=%d\n",rank,num_nodes,nprocs_per_node[0],num_aggr,naggr_per_node[0]);
            for (i = 0; i < num_aggr; i++) {
                int stripe_i = i % striping_factor;
                j = stripe_i % num_nodes; /* to select from node j */
                k = nprocs_per_node[j] / naggr_per_node[j];
                k *= idx_per_node[j];
                idx_per_node[j]++;
                assert(k < nprocs_per_node[j]);
                fd->hints->ranklist[i] = ranks_per_node[j][k];
            }
            ADIOI_Free(naggr_per_node);
            ADIOI_Free(idx_per_node);
        }
printf("%d: num_nodes=%d nprocs_per_node[0]=%d num_aggr=%d ranklist=%d %d %d %d (%s)\n",rank,num_nodes,nprocs_per_node[0],num_aggr,fd->hints->ranklist[0],fd->hints->ranklist[1],fd->hints->ranklist[2],fd->hints->ranklist[3],fd->filename);
fflush(stdout);

        /* TODO: we can keep these two arrays in case for dynamic construction
         * of fd->hints->ranklist[], such as in group-cyclic file domain
         * assignment method, used in each collective write call.
         */
        ADIOI_Free(nprocs_per_node);
        ADIOI_Free(ranks_per_node[0]);
        ADIOI_Free(ranks_per_node);

        msg[0] = num_aggr;
        msg[1] = fd->hints->fs_hints.lustre.num_osts;
    }

    /* bcast cb_nodes and lustre.num_osts to all processes */
    MPI_Bcast(msg, 2, MPI_INT, root, fd->comm);
    num_aggr = msg[0];

    /* set file striping hints */
    fd->hints->cb_nodes = num_aggr;
    sprintf(value, "%d", fd->hints->cb_nodes);
    ADIOI_Info_set(fd->info, "cb_nodes", value);

    fd->hints->fs_hints.lustre.num_osts = msg[1];
    sprintf(value, "%d", fd->hints->fs_hints.lustre.num_osts);
    ADIOI_Info_set(fd->info, "lustre_num_osts", value);

    if (rank != root) {
        /* ranklist[] contains the MPI ranks of I/O aggregators */
        fd->hints->ranklist = (int *) ADIOI_Malloc(num_aggr * sizeof(int));
        if (fd->hints->ranklist == NULL) {
            *error_code = MPIO_Err_create_code(*error_code,
                                               MPIR_ERR_RECOVERABLE, myname,
                                               __LINE__, MPI_ERR_OTHER,
                                               "**nomem2", 0);
            return 0;
        }
    }

    MPI_Bcast(fd->hints->ranklist, fd->hints->cb_nodes, MPI_INT, root, fd->comm);

    /* check whether this process is selected as an I/O aggregator */
    fd->is_agg = 0;
    for (i = 0; i < num_aggr; i++) {
        if (rank == fd->hints->ranklist[i]) {
            fd->is_agg = 1;
            break;
        }
    }

    return 0;
}


/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "ad_lustre.h"
#include "adio_extern.h"


#ifdef HAVE_LUSTRE_LOCKAHEAD
/* in ad_lustre_lock.c */
void ADIOI_LUSTRE_lock_ahead_ioctl(ADIO_File fd,
                                   int cb_nodes, ADIO_Offset next_offset, int *error_code);

/* Handle lock ahead.  If this write is outside our locked region, lock it now */
#define ADIOI_LUSTRE_WR_LOCK_AHEAD(fd,cb_nodes,offset,error_code)           \
if (fd->hints->fs_hints.lustre.lock_ahead_write) {                          \
    if (offset > fd->hints->fs_hints.lustre.lock_ahead_end_extent) {        \
        ADIOI_LUSTRE_lock_ahead_ioctl(fd, cb_nodes, offset, error_code);    \
    }                                                                       \
    else if (offset < fd->hints->fs_hints.lustre.lock_ahead_start_extent) { \
        ADIOI_LUSTRE_lock_ahead_ioctl(fd, cb_nodes, offset, error_code);    \
    }                                                                       \
}
#else
#define ADIOI_LUSTRE_WR_LOCK_AHEAD(fd,cb_nodes,offset,error_code)
#endif

#ifdef MPI_STATUSES_IGNORE
#define AD_LUSTRE_WAITALL(nreqs, reqs) \
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
#else
#define AD_LUSTRE_WAITALL(nreqs, reqs) { \
    MPI_Status *statuses = (MPI_Status *) ADIOI_Malloc((nreqs) * sizeof(MPI_Status)); \
    MPI_Waitall(nreqs, reqs, statuses); \
    ADIOI_Free(statuses); \
}
#endif

#define MEMCPY_UNPACK(x, inbuf, start, count, outbuf) {          \
    int _k;                                                      \
    char *_ptr = (inbuf);                                        \
    MPI_Aint    *mem_ptrs = others_req[x].mem_ptrs + (start)[x]; \
    ADIO_Offset *mem_lens = others_req[x].lens     + (start)[x]; \
    for (_k=0; _k<(count)[x]; _k++) {                            \
        memcpy((outbuf) + mem_ptrs[_k], _ptr, mem_lens[_k]);     \
        _ptr += mem_lens[_k];                                    \
    }                                                            \
}

typedef struct {
    int          num; /* number of elements in the above off-len list */
    int         *len; /* list of write lengths by this rank in round m */
    ADIO_Offset *off; /* list of write offsets by this rank in round m */
} off_len_list;


/* prototypes of functions used for collective writes only. */
static void ADIOI_LUSTRE_Exch_and_write(ADIO_File fd,
                                        const void *buf,
                                        MPI_Datatype datatype,
                                        ADIOI_Access * others_req,
                                        ADIOI_Access * my_req,
                                        ADIO_Offset * offset_list,
                                        ADIO_Offset * len_list,
                                        ADIO_Offset min_st_loc,
                                        ADIO_Offset max_end_loc,
                                        int contig_access_count,
                                        int my_aggr_idx,
                                        ADIO_Offset **buf_idx,
                                        int *error_code);
static void ADIOI_LUSTRE_Fill_send_buffer(ADIO_File fd, const void *buf,
                                          int *n_buftypes,
                                          int *flat_buf_idx,
                                          int *flat_buf_sz,
                                          const ADIOI_Flatlist_node * flat_buf,
                                          char **send_buf,
                                          int *fileview_indx,
                                          ADIO_Offset * offset_list,
                                          ADIO_Offset * len_list,
                                          ADIO_Offset total_send_size_iter,
                                          const int *send_size,
                                          MPI_Request *send_reqs,
                                          int *n_send_reqs,
                                          int contig_access_count,
                                          int my_aggr_idx,
                                          int iter,
                                          ADIO_Offset buftype_extent);
static void ADIOI_LUSTRE_W_Exchange_data(ADIO_File fd, const void *buf,
                                         char *write_buf,
                                         char **recve_buf,
                                         char ***send_buf,
                                         int *n_buftypes,
                                         int *flat_buf_idx,
                                         int *flat_buf_sz,
                                         const ADIOI_Flatlist_node *flat_buf,
                                         int *fileview_indx,
                                         ADIO_Offset *offset_list,
                                         ADIO_Offset *len_list,
                                         const int *send_size,
                                         const int *recv_size,
                                         ADIO_Offset real_off,
                                         int real_size,
                                         const int *count,
                                         const int *start_pos,
                                         int buftype_is_contig,
                                         int contig_access_count,
                                         int my_aggr_idx,
                                         const ADIOI_Access *others_req,
                                         int iter,
                                         ADIO_Offset buftype_extent,
                                         const ADIO_Offset * buf_idx,
                                         off_len_list *srt_off_len,
                                         MPI_Request *reqs,
                                         int *n_reqs,
                                         int *error_code);
void ADIOI_Heap_merge(ADIOI_Access * others_req, int *count,
                      ADIO_Offset * srt_off, int *srt_len, int *start_pos,
                      int nprocs, int nprocs_recv, int total_elements);

static void ADIOI_LUSTRE_IterateOneSided(ADIO_File fd, const void *buf, int *striping_info,
                                         ADIO_Offset * offset_list, ADIO_Offset * len_list,
                                         int contig_access_count, int currentValidDataIndex,
                                         int count, int file_ptr_type, ADIO_Offset offset,
                                         ADIO_Offset start_offset, ADIO_Offset end_offset,
                                         ADIO_Offset firstFileOffset, ADIO_Offset lastFileOffset,
                                         MPI_Datatype datatype, int myrank, int *error_code);

/* ADIOI_LUSTRE_Calc_my_req() - calculate what portions of the access requests
 * of this process are located in the file domains of I/O aggregators.
 */
static
void ADIOI_LUSTRE_Calc_my_req(ADIO_File fd,
                              ADIO_Offset *offset_list,
                              ADIO_Offset *len_list,
                              int contig_access_count,
                              int *count_my_req_procs_ptr,
                              int **count_my_req_per_proc_ptr,
                              ADIOI_Access **my_req_ptr,
                              ADIO_Offset ***buf_idx_ptr)
{
    int *count_my_req_per_proc, count_my_req_procs;
    int i, l, proc, *aggr_ranks, cb_nodes;
    size_t nelems;
    ADIO_Offset avail_len, rem_len, curr_idx, off, **buf_idx, *ptr;
    ADIO_Offset *avail_lens;
    ADIOI_Access *my_req;

    cb_nodes = fd->hints->cb_nodes;

    *count_my_req_per_proc_ptr = (int *) ADIOI_Calloc(cb_nodes, sizeof(int));
    count_my_req_per_proc = *count_my_req_per_proc_ptr;
    /* count_my_req_per_proc[i] gives the number of contiguous requests of this
     * process that fall in aggregator i's file domain.
     */

    /* First pass just to calculate how much space to allocate for my_req.
     * contig_access_count has been calculated way back in
     * ADIOI_Calc_my_off_len()
     */
    aggr_ranks = (int *) ADIOI_Malloc(contig_access_count * sizeof(int));
    avail_lens = (ADIO_Offset *) ADIOI_Malloc(contig_access_count * sizeof(ADIO_Offset));

    /* nelems will be the number of offset-length pairs for my_req[] */
    nelems = 0;
    for (i = 0; i < contig_access_count; i++) {
        /* short circuit offset/len processing if len == 0
         * (zero-byte  read/write
         */
        /* short circuit offset/len processing if zero-byte read/write. */
        if (len_list[i] == 0)
            continue;

        off = offset_list[i];
        avail_len = len_list[i];
        /* ADIOI_LUSTRE_Calc_aggregator() modifies the value of avail_len to
         * return the amount that is only covered by the proc's file domain.
         * The remaining will still need to determine to whose file domain it
         * belongs. As ADIOI_LUSTRE_Calc_aggregator() can be expensive for
         * large value of contig_access_count, keep a copy of the returned
         * values proc and avail_len in aggr_ranks[] and avail_lens[] to be
         * used in the next for loop.
         */
        proc = ADIOI_LUSTRE_Calc_aggregator(fd, off, &avail_len);
        aggr_ranks[i] = proc;
        avail_lens[i] = avail_len;
        count_my_req_per_proc[proc]++;
        nelems++;

        /* rem_len is the amount of i's offset-length pair that is not covered
         * by proc's file domain.
         */
        rem_len = len_list[i] - avail_len;

        while (rem_len != 0) {
            off += avail_len;   /* point to first remaining byte */
            avail_len = rem_len;        /* save remaining size, pass to calc */
            proc = ADIOI_LUSTRE_Calc_aggregator(fd, off, &avail_len);
            count_my_req_per_proc[proc]++;
            nelems++;
            rem_len -= avail_len;       /* reduce remaining length by amount from fd */
        }
    }

    /* buf_idx is relevant only if buftype_is_contig.  buf_idx[i] gives the
     * starting index in user_buf where data will be sent to proc 'i'. This
     * allows sends to be done without extra buffer.
     */
    ptr = (ADIO_Offset *) ADIOI_Malloc((nelems * 3 + cb_nodes) * sizeof(ADIO_Offset));

    /* allocate space for buf_idx */
    buf_idx = (ADIO_Offset **) ADIOI_Malloc(cb_nodes * sizeof(ADIO_Offset *));
    buf_idx[0] = ptr;
    for (i = 1; i < cb_nodes; i++)
        buf_idx[i] = buf_idx[i - 1] + count_my_req_per_proc[i - 1] + 1;
    ptr += nelems + cb_nodes;     /* "+ cb_nodes" puts a terminal index at the end */

    /* allocate space for my_req and its members offsets and lens */
    *my_req_ptr = (ADIOI_Access *) ADIOI_Malloc(cb_nodes * sizeof(ADIOI_Access));
    my_req = *my_req_ptr;
    my_req[0].offsets = ptr;

    /* count_my_req_procs is the number of aggregators whose file domains have
     * this rank's write requests
     */
    count_my_req_procs = 0;
    for (i = 0; i < cb_nodes; i++) {
        if (count_my_req_per_proc[i]) {
            my_req[i].offsets = ptr;
            ptr += count_my_req_per_proc[i];
            my_req[i].lens = ptr;
            ptr += count_my_req_per_proc[i];
            count_my_req_procs++;
        }
        my_req[i].count = 0;    /* will be incremented where needed later */
    }

    /* now fill in my_req */
    curr_idx = 0;
    for (i = 0; i < contig_access_count; i++) {
        /* short circuit offset/len processing if zero-byte read/write. */
        if (len_list[i] == 0)
            continue;

        off = offset_list[i];
        proc = aggr_ranks[i];
        avail_len = avail_lens[i];

        l = my_req[proc].count;
        ADIOI_Assert(l < count_my_req_per_proc[proc]);
        buf_idx[proc][l] = curr_idx;
        curr_idx += avail_len;
        rem_len = len_list[i] - avail_len;

        /* Each my_req[i] contains the number of this process's noncontiguous
         * requests that fall into process i's file domain. my_req[i].offsets[]
         * and my_req[i].lens store the offsets and lengths of the requests.
         */
        my_req[proc].offsets[l] = off;
        my_req[proc].lens[l] = avail_len;
        my_req[proc].count++;

        while (rem_len != 0) {
            off += avail_len;
            avail_len = rem_len;
            proc = ADIOI_LUSTRE_Calc_aggregator(fd, off, &avail_len);
            l = my_req[proc].count;
            ADIOI_Assert(l < count_my_req_per_proc[proc]);
            buf_idx[proc][l] = curr_idx;
            curr_idx += avail_len;
            rem_len -= avail_len;

            my_req[proc].offsets[l] = off;
            my_req[proc].lens[l] = avail_len;
            my_req[proc].count++;
        }
    }
    ADIOI_Free(aggr_ranks);
    ADIOI_Free(avail_lens);

#ifdef AGG_DEBUG
    for (i = 0; i < cb_nodes; i++) {
        if (count_my_req_per_proc[i] > 0) {
            FPRINTF(stdout, "data needed from %d (count = %d):\n", i, my_req[i].count);
            for (l = 0; l < my_req[i].count; l++) {
                FPRINTF(stdout, "   off[%d] = %lld, len[%d] = %d\n",
                        l, (long long) my_req[i].offsets[l], l, (long long) my_req[i].lens[l]);
            }
        }
    }
#endif

    *count_my_req_procs_ptr = count_my_req_procs;
    *buf_idx_ptr = buf_idx;
}

static
void ADIOI_LUSTRE_Calc_others_req(ADIO_File fd,
                                  int count_my_req_aggr,
                                  int *count_my_req_per_aggr,
                                  ADIOI_Access *my_req,
                                  int nprocs,
                                  int myrank,
                                  int my_aggr_idx,
                                  int *count_others_req_procs_ptr,
                                  int **count_others_req_per_proc_ptr,
                                  ADIOI_Access **others_req_ptr)
{
    /* determine what requests of other processes lie in this process's file
     * domain.
     *
     * count_others_req_procs = number of processes whose requests lie in this
     * process's file domain (including this process itself)
     * count_others_req_per_proc[i] indicates how many separate contiguous
     * requests of proc. i lie in this process's file domain.
     */

    int i, j, *count_my_req_per_proc;
    int *count_others_req_per_proc, count_others_req_procs;
    MPI_Request *requests;
    ADIOI_Access *others_req;
    size_t memLen;
    ADIO_Offset *ptr;
    MPI_Aint *mem_ptrs;

/* first find out how much to send/recv and from/to whom */
#ifdef AGGREGATION_PROFILE
    MPE_Log_event(5026, 0, NULL);
#endif
    count_others_req_per_proc = (int *) ADIOI_Malloc(nprocs * sizeof(int));

    /* count_my_req_per_aggr[i]: the number of contiguous requests of this
     * process that fall in aggregator i's file domain.
     * count_others_req_per_proc[i] indicates how many noncontiguous requests
     * from process i that fall into this process's file domain.
     */
    count_my_req_per_proc = (int*) ADIOI_Calloc(nprocs, sizeof(int));
    for (i=0; i<fd->hints->cb_nodes; i++)
        count_my_req_per_proc[fd->hints->ranklist[i]] = count_my_req_per_aggr[i];

    MPI_Alltoall(count_my_req_per_proc, 1, MPI_INT,
                 count_others_req_per_proc, 1, MPI_INT, fd->comm);
    ADIOI_Free(count_my_req_per_proc);

    *others_req_ptr = (ADIOI_Access *) ADIOI_Malloc(nprocs * sizeof(ADIOI_Access));
    others_req = *others_req_ptr;

    memLen = 0;
    for (i = 0; i < nprocs; i++)
        memLen += count_others_req_per_proc[i];
    ptr = (ADIO_Offset *) ADIOI_Malloc(memLen * 2 * sizeof(ADIO_Offset));
    mem_ptrs = (MPI_Aint *) ADIOI_Malloc(memLen * sizeof(MPI_Aint));
    others_req[0].offsets = ptr;
    others_req[0].mem_ptrs = mem_ptrs;

    count_others_req_procs = 0;
    for (i = 0; i < nprocs; i++) {
        if (count_others_req_per_proc[i]) {
            others_req[i].count = count_others_req_per_proc[i];
            others_req[i].offsets = ptr;
            ptr += count_others_req_per_proc[i];
            others_req[i].lens = ptr;
            ptr += count_others_req_per_proc[i];
            others_req[i].mem_ptrs = mem_ptrs;
            mem_ptrs += count_others_req_per_proc[i];
            count_others_req_procs++;
        } else
            others_req[i].count = 0;
    }
    *count_others_req_per_proc_ptr = count_others_req_per_proc;

/* now send the calculated offsets and lengths to respective processes */

    requests = (MPI_Request *)
        ADIOI_Malloc(1 + (count_my_req_aggr + count_others_req_procs) * sizeof(MPI_Request));
/* +1 to avoid a 0-size malloc */

    j = 0;
    for (i = 0; i < nprocs; i++) {
        if (others_req[i].count == 0)
            continue;
        if (i == myrank)
            /* send to self uses memcpy(), here
             * others_req[i].count == my_req[my_aggr_idx].count
             */
            memcpy(others_req[i].offsets, my_req[my_aggr_idx].offsets,
                   2 * my_req[my_aggr_idx].count * sizeof(ADIO_Offset));
        else
            MPI_Irecv(others_req[i].offsets, 2 * others_req[i].count,
                      ADIO_OFFSET, i, 0, fd->comm, &requests[j++]);
    }

    for (i = 0; i < fd->hints->cb_nodes; i++) {
        if (my_req[i].count && i != my_aggr_idx) {
            int dest = fd->hints->ranklist[i];
            MPI_Isend(my_req[i].offsets, 2 * my_req[i].count,
                      ADIO_OFFSET, dest, 0, fd->comm, &requests[j++]);
        }
    }

    if (j) {
#ifdef MPI_STATUSES_IGNORE
        MPI_Waitall(j, requests, MPI_STATUSES_IGNORE);
#else
        MPI_Status *statuses = (MPI_Status *) ADIOI_Malloc(j * sizeof(MPI_Status));
        MPI_Waitall(j, requests, statuses);
        ADIOI_Free(statuses);
#endif
    }

    ADIOI_Free(requests);

    *count_others_req_procs_ptr = count_others_req_procs;
#ifdef AGGREGATION_PROFILE
    MPE_Log_event(5027, 0, NULL);
#endif
}

void ADIOI_LUSTRE_WriteStridedColl(ADIO_File fd, const void *buf, MPI_Aint count,
                                   MPI_Datatype datatype,
                                   int file_ptr_type, ADIO_Offset offset,
                                   ADIO_Status * status, int *error_code)
{
    /* Uses a generalized version of the extended two-phase method described in
     * "An Extended Two-Phase Method for Accessing Sections of Out-of-Core
     * Arrays", Rajeev Thakur and Alok Choudhary, Scientific Programming,
     * (5)4:301--317, Winter 1996.
     * http://www.mcs.anl.gov/home/thakur/ext2ph.ps
     */

    int i, nprocs, nonzero_nprocs, myrank, old_error, tmp_error, my_aggr_idx;
    int striping_info[3], do_collect = 0, contig_access_count = 0;
    ADIO_Offset orig_fp, start_offset, end_offset;
    ADIO_Offset min_st_loc = -1, max_end_loc = -1;
    ADIO_Offset *offset_list = NULL, *len_list = NULL;

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);

    orig_fp = fd->fp_ind;

    /* Check if collective write is actually necessary, if cb_write hint isn't
     * disabled by users.
     */
    if (fd->hints->cb_write != ADIOI_HINT_DISABLE) {
        int is_interleaved;
        ADIO_Offset st_end[2], *st_end_all = NULL;

        /* Calculate and construct the list of starting file offsets and
         * lengths of write requests of this process into offset_list and
         * len_list, respectively. No inter-process communication is needed.
         * The number of elements in both offset_list[] and len_list[] is
         * contig_access_count.
         *
         * From start_offset to end_offset is this process's aggregate access
         * file region. Note: end_offset points to the last byte-offset to be
         * accessed.  e.g., if start_offset=0 and 100 bytes to be read,
         * end_offset=99. No inter-process communication is needed. If this
         * process has no data to write, end_offset == (start_offset - 1)
         */
        ADIOI_Calc_my_off_len(fd, count, datatype, file_ptr_type, offset,
                              &offset_list, &len_list, &start_offset,
                              &end_offset, &contig_access_count);

        /* All processes gather starting and ending file offsets of requests
         * from all processes into st_end_all[]. Even indices of st_end_all[]
         * are start offsets, odd indices are end offsets. st_end_all[] is used
         * below to tell whether access across all process is interleaved.
         */
        st_end[0] = start_offset;
        st_end[1] = end_offset;
        st_end_all = (ADIO_Offset *) ADIOI_Malloc(nprocs * 2 * sizeof(ADIO_Offset));
        MPI_Allgather(st_end, 2, ADIO_OFFSET, st_end_all, 2, ADIO_OFFSET, fd->comm);

        /* Find the starting and ending file offsets of aggregate access region
         * and the number of processes that have non-zero length write
         * requests. Also, check whether accesses are interleaved across
         * processes. Below is a rudimentary check for interleaving, but should
         * suffice for the moment.
         */
        is_interleaved = 0;
        min_st_loc = st_end_all[0];
        max_end_loc = st_end_all[1];
        nonzero_nprocs = (st_end_all[0] - 1 == st_end_all[1]) ? 0 : 1;
        for (i = 2; i < nprocs * 2; i += 2) {
            if (st_end_all[i] - 1 == st_end_all[i + 1]) {
                /* process rank (i/2) has no data to write */
                continue;
            }
            if (st_end_all[i] < st_end_all[i - 1]) {
                /* start offset of process rank (i/2) is less than the end
                 * offset of process rank (i/2-1)
                 */
                is_interleaved = 1;
            }
            min_st_loc = MPL_MIN(st_end_all[i], min_st_loc);
            max_end_loc = MPL_MAX(st_end_all[i + 1], max_end_loc);
            nonzero_nprocs++;
        }
        ADIOI_Free(st_end_all);

        /* Two typical access patterns can benefit from collective write.
         *   1) access file regions among all processes are interleaved, and
         *   2) the individual request sizes are not too big, i.e. no bigger
         *      than hint coll_threshold.  Large individual requests may cause
         *      a high communication cost for redistributing requests to the
         *      I/O aggregators.
         */
        if (is_interleaved > 0) {
            do_collect = 1;
        } else {
            /* This ADIOI_LUSTRE_Docollect() calls MPI_Allreduce(), so all
             * processes must participate.
             */
            do_collect = ADIOI_LUSTRE_Docollect(fd, contig_access_count, len_list, nprocs);
        }
    }

    /* If collective I/O is not necessary, use independent I/O */
    if ((!do_collect && fd->hints->cb_write == ADIOI_HINT_AUTO) ||
        fd->hints->cb_write == ADIOI_HINT_DISABLE) {

        int buftype_is_contig, filetype_is_contig;

        if (offset_list != NULL)
            ADIOI_Free(offset_list);

        fd->fp_ind = orig_fp;

        ADIOI_Datatype_iscontig(datatype, &buftype_is_contig);
        ADIOI_Datatype_iscontig(fd->filetype, &filetype_is_contig);

        if (buftype_is_contig && filetype_is_contig) {
            ADIO_Offset off = 0;
            if (file_ptr_type == ADIO_EXPLICIT_OFFSET)
                off = fd->disp + offset * fd->etype_size;
            ADIO_WriteContig(fd, buf, count, datatype, file_ptr_type, off, status, error_code);
        } else {
            ADIO_WriteStrided(fd, buf, count, datatype, file_ptr_type, offset, status, error_code);
        }
        return;
    }

    /* Get Lustre hints information */
    striping_info[0] = fd->hints->striping_unit;
    striping_info[1] = fd->hints->striping_factor;
    striping_info[2] = fd->hints->cb_nodes;

    if ((fd->romio_write_aggmethod == 1) || (fd->romio_write_aggmethod == 2)) {
        /* If user has set hint ROMIO_WRITE_AGGMETHOD env variable to to use a
         * one-sided aggregation method then do that at this point instead of
         * using the traditional MPI point-to-point communication, i.e.
         * MPI_Issend and MPI_Irecv.
         */
        ADIOI_LUSTRE_IterateOneSided(fd, buf, striping_info, offset_list, len_list,
                                     contig_access_count, nonzero_nprocs, count,
                                     file_ptr_type, offset, start_offset, end_offset,
                                     min_st_loc, max_end_loc, datatype, myrank, error_code);
    } else {
        /* my_req[cb_nodes] is an array of access info, one for each I/O
         * aggregator whose file domain has this rank's request.
         */
        ADIOI_Access *my_req;
        int count_my_req_aggr, *count_my_req_per_aggr;

        /* others_req[nprocs] is an array of access info, one for each process
         * whose write requests fall into this process's file domain.
         */
        ADIOI_Access *others_req;
        int count_others_req_procs, *count_others_req_per_proc;
        ADIO_Offset **buf_idx = NULL;

        /* Calculate the portions of this process's write requests that fall
         * into the file domains of each I/O aggregator. No inter-process
         * communication is needed.
         * count_my_req_per_aggr[i]: the number of contiguous requests of this
         * process that fall in aggregator i's file domain.
         * count_my_req_aggr is the number of aggregators whose file domains
         * have this rank's write requests>
         */
        ADIOI_LUSTRE_Calc_my_req(fd, offset_list, len_list, contig_access_count,
                                 &count_my_req_aggr, &count_my_req_per_aggr,
                                 &my_req, &buf_idx);

        /* Find my MPI rank ID if this rank is an aggregator. Otherwise -1 */
        my_aggr_idx = -1;
        for (i = 0; i < fd->hints->cb_nodes; i++) {
            if (myrank == fd->hints->ranklist[i]) {
                my_aggr_idx = i;
                break;
            }
        }

        /* Calculate the portions of all other process's requests fall into
         * this process's file domain (note only I/O aggregators are assigned
         * file domains). Inter-process communication is required to construct
         * others_req[], including MPI_Alltoall, MPI_Issend, MPI_Irecv, and
         * MPI_Waitall.
         *
         * count_others_req_procs is the number of processes whose requests
         * (including this process itself) fall into this process's file
         * domain.
         * count_others_req_per_proc[i] indicates how many noncontiguous
         * requests from process i that fall into this process's file domain.
         */
        ADIOI_LUSTRE_Calc_others_req(fd, count_my_req_aggr,
                                     count_my_req_per_aggr, my_req, nprocs,
                                     myrank, my_aggr_idx,
                                     &count_others_req_procs,
                                     &count_others_req_per_proc, &others_req);

        /* Two-phase I/O: first communication phase to exchange write data from
         * all processes to the I/O aggregators, followed by the write phase
         * where only I/O aggregators write to the file. There is no collective
         * MPI communication in ADIOI_LUSTRE_Exch_and_write(), only MPI_Issend,
         * MPI_Irecv, and MPI_Waitall.
         */
        ADIOI_LUSTRE_Exch_and_write(fd, buf, datatype, others_req, my_req,
                                    offset_list, len_list, min_st_loc,
                                    max_end_loc, contig_access_count,
                                    my_aggr_idx, buf_idx, error_code);

        /* free all memory allocated */
        ADIOI_Free_others_req(nprocs, count_others_req_per_proc, others_req);

        ADIOI_Free(count_my_req_per_aggr);
        ADIOI_Free(buf_idx[0]);
        ADIOI_Free(buf_idx);
        ADIOI_Free(my_req);
    }
    ADIOI_Free(offset_list);

    /* If this collective write is followed by an independent write, it's
     * possible to have those subsequent writes on other processes race ahead
     * and sneak in before the read-modify-write completes.  We carry out a
     * collective communication at the end here so no one can start independent
     * I/O before collective I/O completes.
     *
     * need to do some gymnastics with the error codes so that if something
     * went wrong, all processes report error, but if a process has a more
     * specific error code, we can still have that process report the
     * additional information */
    old_error = *error_code;
    if (*error_code != MPI_SUCCESS)
        *error_code = MPI_ERR_IO;

    /* optimization: if only one process performing I/O, we can perform
     * a less-expensive Bcast. */
#ifdef ADIOI_MPE_LOGGING
    MPE_Log_event(ADIOI_MPE_postwrite_a, 0, NULL);
#endif
    if (fd->hints->cb_nodes == 1)
        MPI_Bcast(error_code, 1, MPI_INT, fd->hints->ranklist[0], fd->comm);
    else {
        tmp_error = *error_code;
        MPI_Allreduce(&tmp_error, error_code, 1, MPI_INT, MPI_MAX, fd->comm);
    }
#ifdef ADIOI_MPE_LOGGING
    MPE_Log_event(ADIOI_MPE_postwrite_b, 0, NULL);
#endif

    if ((old_error != MPI_SUCCESS) && (old_error != MPI_ERR_IO))
        *error_code = old_error;

#ifdef HAVE_STATUS_SET_BYTES
    if (status) {
        MPI_Count bufsize, size;
        /* Don't set status if it isn't needed */
        MPI_Type_size_x(datatype, &size);
        bufsize = size * count;
        MPIR_Status_set_bytes(status, datatype, bufsize);
    }
    /* This is a temporary way of filling in status. The right way is to
     * keep track of how much data was actually written during collective I/O.
     */
#endif

    fd->fp_sys_posn = -1;       /* set it to null. */
}

/* If successful, error_code is set to MPI_SUCCESS.  Otherwise an error code is
 * created and returned in error_code.
 */
static void ADIOI_LUSTRE_Exch_and_write(ADIO_File fd,
                                        const void *buf,
                                        MPI_Datatype datatype,
                                        ADIOI_Access *others_req,
                                        ADIOI_Access *my_req,
                                        ADIO_Offset *offset_list,
                                        ADIO_Offset *len_list,
                                        ADIO_Offset min_st_loc,
                                        ADIO_Offset max_end_loc,
                                        int contig_access_count,
                                        int my_aggr_idx,
                                        ADIO_Offset **buf_idx,
                                        int *error_code)
{
    /* Each process sends all its write requests to I/O aggregators based on
     * the file domain assignment to the aggregators. In this implementation,
     * a file is first divided into stripes and all its stripes are assigned to
     * I/O aggregators in a round-robin fashion. The collective write is
     * carried out in 'ntimes' rounds of two-phase I/O. Each round covers an
     * aggregate file region of size equal to the file stripe size times the
     * number of I/O aggregators. In other words, the 'collective buffer size'
     * used in each aggregator is always set equally to the file stripe size,
     * ignoring the MPI-IO hint 'cb_buffer_size'. There are other algorithms
     * allowing an aggregator to write more than a file stripe size in each
     * round, up to the cb_buffer_size hint. For those, refer to the paper:
     * Wei-keng Liao, and Alok Choudhary. "Dynamically Adapting File Domain
     * Partitioning Methods for Collective I/O Based on Underlying Parallel
     * File System Locking Protocols", in The Supercomputing Conference, 2008.
     */

    char **write_buf = NULL, **recv_buf = NULL, ***send_buf = NULL;
    int i, j, m, nprocs, myrank, ntimes, nbufs, buftype_is_contig;
    int *recv_curr_offlen_ptr, **recv_size=NULL, **recv_count=NULL;
    int **recv_start_pos=NULL;
    int *send_curr_offlen_ptr, *send_size;
    int batch_idx, *nreqs, batch_nreqs, cb_nodes, striping_unit;
    ADIO_Offset end_loc, req_off, iter_end_off, *off_list, step_size;
    ADIO_Offset *this_buf_idx, buftype_extent;
    ADIOI_Flatlist_node *flat_buf = NULL;
    off_len_list *srt_off_len = NULL;

    *error_code = MPI_SUCCESS;

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);

    /* The aggregate access region (across all processes) of this collective
     * write starts from min_st_loc and ends at max_end_loc. The collective
     * write is carried out in 'ntimes' rounds of two-phase I/O. Each round
     * covers an aggregate file region of size 'step_size' written only by
     * cb_nodes number of I/O aggregators. Note
     * non-aggregators must also participate all ntimes rounds to send their
     * requests to I/O aggregators.
     */

    cb_nodes = fd->hints->cb_nodes;
    striping_unit = fd->hints->striping_unit;

    /* step_size is the size of aggregate access region covered by each round
     * of two-phase I/O
     */
    step_size = (ADIO_Offset)cb_nodes * striping_unit;

    /* align min_st_loc downward to the nearest file stripe boundary */
    min_st_loc -= min_st_loc % (ADIO_Offset) striping_unit;

    /* ntimes is the number of rounds of two-phase I/O */
    ntimes = (max_end_loc - min_st_loc + 1) / step_size;
    if ((max_end_loc - min_st_loc + 1) % step_size)
        ntimes++;

    /* off_list[m] is the starting file offset of this process's write region
     * in iteration m (file domain of iteration m). This offset may not be
     * aligned with file stripe boundaries. end_loc is the ending file offset
     * of this process's file domain.
     */
    off_list = (ADIO_Offset *) ADIOI_Malloc(ntimes * sizeof(ADIO_Offset));
    end_loc = -1;
    for (m = 0; m < ntimes; m++)
        off_list[m] = max_end_loc;
    for (i = 0; i < nprocs; i++) {
        for (j = 0; j < others_req[i].count; j++) {
            req_off = others_req[i].offsets[j];
            m = (int) ((req_off - min_st_loc) / step_size);
            off_list[m] = MPL_MIN(off_list[m], req_off);
            end_loc = MPL_MAX(end_loc, (others_req[i].offsets[j] + others_req[i].lens[j] - 1));
        }
    }

    /* collective buffer size is divided by 2, one for write buffer and the
     * other for receive messages from other processes. nbufs is the number of
     * write buffers, each of size equal to Lustre stripe size. Write requests
     * are sent to aggregators for nbufs amount before writing to the file.
     */
    nbufs = fd->hints->cb_buffer_size / striping_unit / 2;
    if (fd->hints->cb_buffer_size % striping_unit) nbufs++;
    nbufs = (nbufs > ntimes) ? ntimes : nbufs;
    if (nbufs == 0) nbufs = 1; /* must at least 1 */

    /* end_loc >= 0 indicates this process has something to write. Only I/O
     * aggregators can have end_loc > 0. write_buf is the collective buffer and
     * only matter for I/O aggregators. recv_buf is the buffer used only in
     * aggregators to receive requests from other processes. Its size may be
     * larger then the file stripe size. In this case, it will be realloc-ed in
     * ADIOI_LUSTRE_W_Exchange_data(). The received data is later copied over
     * to write_buf, whose contents will be written to file.
     */
    if (end_loc >= 0) {
        write_buf = (char **) ADIOI_Malloc(nbufs * sizeof(char*));
        write_buf[0] = (char *) ADIOI_Malloc(nbufs * striping_unit);
        for (j = 1; j < nbufs; j++)
            write_buf[j] = write_buf[j-1] + striping_unit;

        recv_buf = (char **) ADIOI_Malloc(nbufs * sizeof(char*));
        for (j = 0; j < nbufs; j++)
            /* recv_buf[j] may be realloc in ADIOI_LUSTRE_W_Exchange_data() */
            recv_buf[j] = (char *) ADIOI_Malloc(striping_unit);
    }

    /* send_buf[] will be allocated in ADIOI_LUSTRE_W_Exchange_data(), when the
     * use buffer is not contiguous.
     */
    send_buf = (char ***) ADIOI_Malloc(nbufs * sizeof(char**));

    /* this_buf_idx contains indices to user buffer for sending this rank's
     * write data to remote aggregators. It is used only when user buffer is
     * contiguous.
     */
    this_buf_idx = (ADIO_Offset *) ADIOI_Malloc(cb_nodes * sizeof(ADIO_Offset));

    /* Allocate multiple buffers of type int altogether at once in a single
     * calloc call. Their use is explained below. calloc initializes to 0.
     */
    recv_curr_offlen_ptr = (int *) ADIOI_Calloc((nprocs + 2 * cb_nodes), sizeof(int));
    send_curr_offlen_ptr = recv_curr_offlen_ptr + nprocs;

    /* array of data sizes to be sent to each aggregator in a 2-phase round */
    send_size = send_curr_offlen_ptr + cb_nodes;

    ADIOI_Datatype_iscontig(datatype, &buftype_is_contig);
    if (!buftype_is_contig) {
        flat_buf = ADIOI_Flatten_and_find(datatype);
#if MPI_VERSION >= 4
        MPI_Count lb, extent;
        MPI_Type_get_extent_c(datatype, &lb, &extent);
#else
        MPI_Aint lb, extent;
        MPI_Type_get_extent(datatype, &lb, &extent);
#endif
        buftype_extent = (ADIO_Offset)extent;

        if (flat_buf->count == 1) /* actually contiguous */
            buftype_is_contig = 1;

if (flat_buf->count == 1) printf("%d: -- CHANGE noncontig to contig ntimes=%d flat_buf->count=%ld\n",myrank,ntimes,flat_buf->count);
/*
for (i=0; i<flat_buf->count; i++)
    printf("    flat_buf[%d] off=%9lld len=%9lld end=%9lld\n",i, flat_buf->indices[i], flat_buf->blocklens[i], flat_buf->indices[i]+flat_buf->blocklens[i]);
 */
    }
if (!buftype_is_contig) printf("%d: -- ntimes=%d buftype NOT contig\n",myrank,ntimes);

    /* I need to check if there are any outstanding nonblocking writes to
     * the file, which could potentially interfere with the writes taking
     * place in this collective write call. Since this is not likely to be
     * common, let me do the simplest thing possible here: Each process
     * completes all pending nonblocking operations before completing.
     */
    /*ADIOI_Complete_async(error_code);
     * if (*error_code != MPI_SUCCESS) return;
     * MPI_Barrier(fd->comm);
     */

    /* min_st_loc has been downward aligned to the nearest file stripe
     * boundary, iter_end_off is the ending file offset of aggregate write
     * region of iteration m, upward aligned to the file stripe boundary.
     */
    iter_end_off = min_st_loc + step_size;

    /* the number of off-len pairs to be received from each proc in a round. */
    recv_count = (int**) ADIOI_Malloc(3 * nbufs * sizeof(int*));
    recv_count[0] = (int*) ADIOI_Malloc(3 * nbufs * nprocs * sizeof(int));
    for (i = 1; i < nbufs; i++)
        recv_count[i] = recv_count[i-1] + nprocs;

    /* recv_size is array of data sizes to be received from each proc in a
     * round. */
    recv_size = recv_count + nbufs;
    recv_size[0] = recv_count[0] + nbufs * nprocs;
    for (i = 1; i < nbufs; i++)
        recv_size[i] = recv_size[i-1] + nprocs;

    /* recv_start_pos[j][i] stores the starting value of
     * recv_curr_offlen_ptr[i] for remote rank i in round j
     */
    recv_start_pos = recv_size + nbufs;
    recv_start_pos[0] = recv_size[0] + nbufs * nprocs;
    for (i = 1; i < nbufs; i++)
        recv_start_pos[i] = recv_start_pos[i-1] + nprocs;

    srt_off_len = (off_len_list*) ADIOI_Malloc(nbufs * sizeof(off_len_list));

    /* upper bound numbers of send and receive */
    int n_send_recv_ub = 0;
    for (i = 0; i < cb_nodes; i++) {
        /* my_req[] is an array of nprocs access structures, one for each
         * aggregator whose file domain has this process's request */
        if (my_req[i].count && i != my_aggr_idx)
            n_send_recv_ub++;
    }
    for (i = 0; i < nprocs; i++) {
        /* others_req[] is an array of nprocs access structures, one for each
         * other process whose requests fall into this process's file domain,
         * i.e. is written by this process. */
        if (others_req[i].count && i != myrank)
            n_send_recv_ub++;
    }
    n_send_recv_ub *= nbufs;

    MPI_Request *reqs;
    reqs = (MPI_Request *) ADIOI_Malloc(n_send_recv_ub * sizeof(MPI_Request));
    batch_idx = batch_nreqs = 0;
    nreqs = (int*) ADIOI_Malloc(nbufs * sizeof(int));

    int interleaving_waitall_n_write = 1;
    int ibuf = 0;
    int fileview_indx = 0;
int n_buftypes = 0;
int flat_buf_idx = 0;
int flat_buf_sz = (buftype_is_contig) ? 0 : flat_buf->blocklens[0];
    for (m = 0; m < ntimes; m++) {
        int real_size;
        ADIO_Offset real_off;

        /* Note that MPI standard requires that displacements in filetypes are
         * in a monotonically non-decreasing order and that, for writes, the
         * filetypes cannot specify overlapping regions in the file. This
         * simplifies implementation a bit compared to reads.
         *
         * real_off      = starting file offset of this process's write region
         *                 for this round (may not be aligned to stripe
         *                 boundary)
         * real_size     = size (in bytes) of this process's write region for
         *                 this round
         * iter_end_off  = ending file offset of aggregate write region of this
         *                 round, and upward aligned to the file stripe
         *                 boundary. Note the aggregate write region of this
         *                 round starts from (iter_end_off-step_size) to
         *                 iter_end_off, aligned with file stripe boundaries.
         * send_size[i]  = total size in bytes of this process's write data
         *                 fall into aggregator i's FD in this round.
         * recv_size[j][i] = total size in bytes of write data to be received
         *                 by this process (aggregator) in round j.
         * recv_count[j][i] = number of noncontiguous offset-length pairs from
         *                 process i fall into this aggregator's write region
         *                 in round j.
         */

        /* reset communication metadata to all 0s for this round */
        for (i = 0; i < nprocs; i++)
            recv_count[ibuf][i] = recv_size[ibuf][i] = recv_start_pos[ibuf][i] = 0;

        for (i = 0; i < cb_nodes; i++)
            send_size[i] = 0;

        real_off = off_list[m];
        real_size = (int) MPL_MIN(striping_unit - real_off % striping_unit, end_loc - real_off + 1);

        /* First calculate what should be communicated, by going through all
         * others_req and my_req to check which will be sent and received in
         * this round.
         */
        for (i = 0; i < cb_nodes; i++) {
            if (my_req[i].count) {
                this_buf_idx[i] = buf_idx[i][send_curr_offlen_ptr[i]];
                for (j = send_curr_offlen_ptr[i]; j < my_req[i].count; j++) {
                    if (my_req[i].offsets[j] < iter_end_off)
                        send_size[i] += my_req[i].lens[j];
                    else
                        break;
                }
                send_curr_offlen_ptr[i] = j;
            }
        }
        for (i = 0; i < nprocs; i++) {
            if (others_req[i].count) {
                recv_start_pos[ibuf][i] = recv_curr_offlen_ptr[i];
                for (j = recv_curr_offlen_ptr[i]; j < others_req[i].count; j++) {
                    if (others_req[i].offsets[j] < iter_end_off) {
                        recv_count[ibuf][i]++;
                        others_req[i].mem_ptrs[j] =
                            (MPI_Aint) (others_req[i].offsets[j] - real_off);
                        recv_size[ibuf][i] += others_req[i].lens[j];
                    } else {
                        break;
                    }
                }
                recv_curr_offlen_ptr[i] = j;
            }
        }
        iter_end_off += step_size;

        /* redistribute (exchange) this process's write requests to I/O
         * aggregators. In ADIOI_LUSTRE_W_Exchange_data(), communication are
         * Issend and Irecv, but no collective communication.
         */
        char *wbuf = (write_buf == NULL) ? NULL : write_buf[ibuf];
        char *rbuf = (recv_buf  == NULL) ? NULL :  recv_buf[ibuf];
        send_buf[ibuf] = NULL;
        ADIOI_LUSTRE_W_Exchange_data(fd,
                                     buf,
                                     wbuf,               /* OUT: updated in each round */
                                     &rbuf,              /* OUT: updated in each round */
                                     &send_buf[ibuf],    /* OUT: updated in each round */
                                     &n_buftypes,
                                     &flat_buf_idx,
                                     &flat_buf_sz,
                                     flat_buf,
                                     &fileview_indx,
                                     offset_list,
                                     len_list,
                                     send_size,            /* IN: changed each round */
                                     recv_size[ibuf],      /* IN: changed each round */
                                     real_off,             /* IN: changed each round */
                                     real_size,            /* IN: changed each round */
                                     recv_count[ibuf],     /* IN: changed each round */
                                     recv_start_pos[ibuf], /* IN: changed each round */
                                     buftype_is_contig,
                                     contig_access_count,
                                     my_aggr_idx,
                                     others_req,           /* IN: changed each round */
                                     m,
                                     buftype_extent,
                                     this_buf_idx,         /* IN: changed each round */
                                     &srt_off_len[ibuf],   /* OUT: list of write request off-len pairs */
                                     reqs + batch_nreqs,   /* OUT: nonblocking recv+send request IDs */
                                     &nreqs[ibuf],         /* OUT: number of recv+send request IDs */
                                     error_code);

        if (*error_code != MPI_SUCCESS)
            goto over;

        /* rbuf might be realloc-ed */
        if (recv_buf != NULL) recv_buf[ibuf] = rbuf;

        /* accumulate nreqs across nbufs rounds */
        batch_nreqs += nreqs[ibuf];

        /* Check if accumulated pending requests can cause high traffic
         * congestion. If yes, then write to file now.
         */
        int write_now;
        if (fd->is_agg)
            write_now = (batch_nreqs >= nprocs-2);  /* -2 to exclude send/recv self */
        else
            /* For I/O patterns that each non-aggregator has a request falling
             * into all aggregtor's file domain, set the trigger condition
             * (cb_nodes * 2) appears to be the best. FYI, setting the
             * condition to (batch_nreqs >= nprocs) and it is no better, as
             * non-aggregators may post too many issend.
             */
            write_now = (batch_nreqs >= cb_nodes * 2);

        /* commit writes for this batch of numBufs */
        if (m % nbufs == nbufs - 1 || m == ntimes - 1 || write_now) {
            MPI_Request *req_ptr;
            int numBufs = ibuf + 1;

            /* reset ibuf to the first element of nbufs */
            ibuf = 0;

            /* Note that atomic mode calls only blocking recv and all
             * nonblocking sends are waited at the end of each round. So,
             * nreqs should be 0 for all ranks.
             */
            if (fd->atomicity) assert(batch_nreqs == 0);

            /* interleaving_waitall_n_write or not does not seem to make a
             * noticeable difference.
             */
            if (interleaving_waitall_n_write) {
                /* non-aggregators wait for all Issend to complete */
                if (!fd->is_agg) {
                    if (batch_nreqs > 0)
                        AD_LUSTRE_WAITALL(batch_nreqs, reqs);

                    batch_nreqs = 0;
                    /* free send_buf that may be allocated in
                     * ADIOI_LUSTRE_W_Exchange_data() */
                    for (j=0; j<numBufs; j++) {
                        if (send_buf[j] != NULL) {
                            ADIOI_Free(send_buf[j][0]);
                            ADIOI_Free(send_buf[j]);
                            send_buf[j] = NULL;
                        }
                    }
                    continue; /* non-aggregators are done for this batch */
                }
                /* aggregators will interleave waitall with write later */
            }
            else {
                /* all processes wait for all isend/irecv to complete */
                if (batch_nreqs > 0) {
                    AD_LUSTRE_WAITALL(batch_nreqs, reqs);
                    batch_nreqs = 0;
                }

                /* free send_buf allocated in ADIOI_LUSTRE_W_Exchange_data() */
                for (j=0; j<numBufs; j++) {
                    if (send_buf[j] != NULL) {
                        ADIOI_Free(send_buf[j][0]);
                        ADIOI_Free(send_buf[j]);
                        send_buf[j] = NULL;
                    }
                }

                if (!fd->is_agg) /* non-aggregators are done for this batch */
                    continue; /* next run of loop m */

                /* Only aggregators will run codes below, to unpack from
                 * recv_buf[] to write_buf[] all numBufs at once */
                if (end_loc >= 0) {
                    for (j=0; j<numBufs; j++) {
                        char *buf_ptr = recv_buf[j];
                        for (i = 0; i < nprocs; i++) {
                            if (recv_count[j][i] > 1 && i != myrank) {
                                /* When recv_count[j][i] == 1, this case has
                                 * been taken care of earlier by receiving the
                                 * message directly into write_buf.
                                 */
                                MEMCPY_UNPACK(i, buf_ptr, recv_start_pos[j],
                                              recv_count[j], write_buf[j]);
                                buf_ptr += recv_size[j][i];
                            }
                        }
                    }
                }
            }

            req_ptr = reqs;
            batch_nreqs = 0;

            /* write to numBufs number of stripes */
            for (j=0; j<numBufs; j++) {
                int real_size;
                ADIO_Offset real_off;

                if (interleaving_waitall_n_write) {
                    /* wait for isend/irecv for round j to complete */
                    AD_LUSTRE_WAITALL(nreqs[j], req_ptr);
                    req_ptr += nreqs[j];

                    /* free send_buf that may be allocated in
                     * ADIOI_LUSTRE_W_Exchange_data() */
                    if (send_buf[j] != NULL) {
                        ADIOI_Free(send_buf[j][0]);
                        ADIOI_Free(send_buf[j]);
                        send_buf[j] = NULL;
                    }

                    if (end_loc >= 0) {
                        /* unpack from recv_buf[] to write_buf[] */
                        char *buf_ptr = recv_buf[j];
                        for (i = 0; i < nprocs; i++) {
                            if (recv_count[j][i] > 1 && i != myrank) {
                                /* When recv_count[j][i] == 1, this case has
                                 * been taken care of earlier by receiving the
                                 * message directly into write_buf.
                                 */
                                MEMCPY_UNPACK(i, buf_ptr, recv_start_pos[j],
                                              recv_count[j], write_buf[j]);
                                buf_ptr += recv_size[j][i];
                            }
                        }
                    }
                }

                /* if there is no data to write in round (batch_idx + j) */
                if (srt_off_len[j].num == 0)
                    continue;

                real_off = off_list[batch_idx + j];
                real_size = (int) MPL_MIN(striping_unit - real_off % striping_unit,
                                          end_loc - real_off + 1);

                /* lock ahead the file starting from real_off */
                ADIOI_LUSTRE_WR_LOCK_AHEAD(fd, cb_nodes, real_off, error_code);
                if (*error_code != MPI_SUCCESS)
                    goto over;

                /* When srt_off_len[j].num == 1, either there is no hole in the
                 * write buffer or the file domain has been read into write
                 * buffer and updated with the received write data. When
                 * srt_off_len[j].num > 1, holes have been found and the list
                 * of sorted offset-length pairs describing noncontiguous
                 * writes have been constructed. Call writes for each
                 * offset-length pair. Note the offset-length pairs
                 * (represented by srt_off_len[j].off, srt_off_len[j].len, and
                 * srt_off_len[j].num) have been coalesced in
                 * ADIOI_LUSTRE_W_Exchange_data().
                 */

                for (i = 0; i < srt_off_len[j].num; i++) {
                    MPI_Status status;

                    /* all write requests should fall into this stripe ranging
                     * [real_off, real_off+real_size). This assertion should
                     * never fail.
                     */
                    ADIOI_Assert(srt_off_len[j].off[i] < real_off + real_size &&
                                 srt_off_len[j].off[i] >= real_off);

                    ADIO_WriteContig(fd,
                                     write_buf[j] + (srt_off_len[j].off[i] - real_off),
                                     srt_off_len[j].len[i],
                                     MPI_BYTE,
                                     ADIO_EXPLICIT_OFFSET,
                                     srt_off_len[j].off[i],
                                     &status, error_code);
                    if (*error_code != MPI_SUCCESS)
                        goto over;
                }
                if (srt_off_len[j].num > 0) {
                    ADIOI_Free(srt_off_len[j].off);
                    ADIOI_Free(srt_off_len[j].len);
                    srt_off_len[j].num = 0;
                }
            }
            batch_idx += numBufs; /* only matters for aggregators */
        }
        else
            ibuf++;
    }

  over:
    ADIOI_Free(reqs);
    ADIOI_Free(nreqs);
    if (srt_off_len)
        ADIOI_Free(srt_off_len);
    if (write_buf != NULL) {
        ADIOI_Free(write_buf[0]);
        ADIOI_Free(write_buf);
    }
    if (recv_buf != NULL) {
        for (j = 0; j < nbufs; j++)
            ADIOI_Free(recv_buf[j]);
        ADIOI_Free(recv_buf);
    }
    if (recv_count != NULL) {
        ADIOI_Free(recv_count[0]);
        ADIOI_Free(recv_count);
    }
    ADIOI_Free(recv_curr_offlen_ptr);
    ADIOI_Free(off_list);
    ADIOI_Free(this_buf_idx);
    if (send_buf != NULL)
        ADIOI_Free(send_buf);
}

/* This subroutine is copied from ADIOI_Heap_merge(), but modified to coalesce
 * sorted offset-length pairs whenever possible.
 *
 * Heapify(a, i, heapsize); Algorithm from Cormen et al. pg. 143 modified for a
 * heap with smallest element at root. The recursion has been removed so that
 * there are no function calls. Function calls are too expensive.
 */
static
void heap_merge(const ADIOI_Access * others_req, const int *count, ADIO_Offset * srt_off,
                int *srt_len, const int *start_pos, int nprocs, int nprocs_recv,
                int *total_elements)
{
    typedef struct {
        ADIO_Offset *off_list;
        ADIO_Offset *len_list;
        int nelem;
    } heap_struct;

    heap_struct *a, tmp;
    int i, j, heapsize, l, r, k, smallest;

    a = (heap_struct *) ADIOI_Malloc((nprocs_recv + 1) * sizeof(heap_struct));

    j = 0;
    for (i = 0; i < nprocs; i++) {
        if (count[i]) {
            a[j].off_list = others_req[i].offsets + start_pos[i];
            a[j].len_list = others_req[i].lens + start_pos[i];
            a[j].nelem = count[i];
            j++;
        }
    }

#define SWAP(x, y, tmp) { tmp = x ; x = y ; y = tmp ; }

    heapsize = nprocs_recv;

    /* Build a heap out of the first element from each list, with the smallest
     * element of the heap at the root. The first for loop is to find and move
     * the smallest a[*].off_list[0] to a[0].
     */
    for (i = heapsize / 2 - 1; i >= 0; i--) {
        k = i;
        for (;;) {
            r = 2 * (k + 1);
            l = r - 1;
            if ((l < heapsize) && (*(a[l].off_list) < *(a[k].off_list)))
                smallest = l;
            else
                smallest = k;

            if ((r < heapsize) && (*(a[r].off_list) < *(a[smallest].off_list)))
                smallest = r;

            if (smallest != k) {
                SWAP(a[k], a[smallest], tmp);
                k = smallest;
            } else
                break;
        }
    }

    /* The heap keeps the smallest element in its first element, i.e.
     * a[0].off_list[0].
     */
    j = 0;
    for (i = 0; i < *total_elements; i++) {
        /* extract smallest element from heap, i.e. the root */
        if (j == 0 || srt_off[j - 1] + srt_len[j - 1] < *(a[0].off_list)) {
            srt_off[j] = *(a[0].off_list);
            srt_len[j] = *(a[0].len_list);
            j++;
        } else {
            /* this offset-length pair can be coalesced into the previous one */
            srt_len[j - 1] = *(a[0].off_list) + *(a[0].len_list) - srt_off[j - 1];
        }
        (a[0].nelem)--;

        if (a[0].nelem) {
            (a[0].off_list)++;
            (a[0].len_list)++;
        } else {
            a[0] = a[heapsize - 1];
            heapsize--;
        }

        /* Heapify(a, 0, heapsize); */
        k = 0;
        for (;;) {
            r = 2 * (k + 1);
            l = r - 1;
            if ((l < heapsize) && (*(a[l].off_list) < *(a[k].off_list)))
                smallest = l;
            else
                smallest = k;

            if ((r < heapsize) && (*(a[r].off_list) < *(a[smallest].off_list)))
                smallest = r;

            if (smallest != k) {
                SWAP(a[k], a[smallest], tmp);
                k = smallest;
            } else
                break;
        }
    }
    ADIOI_Free(a);
    *total_elements = j;
}

static void ADIOI_LUSTRE_W_Exchange_data(
            ADIO_File            fd,
      const void                *buf,          /* user buffer */
            char                *write_buf,    /* OUT: internal buffer used to write in round iter */
            char               **recv_buf,     /* OUT: internal buffer used to receive in round iter */
            char              ***send_buf,     /* OUT: internal buffer used to send in round iter */
            int                 *n_buftypes,   /* IN/OUT: round of user buffer types */
            int                 *flat_buf_idx, /* IN/OUT: index of currently processed segment */
            int                 *flat_buf_sz,  /* IN/OUT: size yet to be processed in the segment */
      const ADIOI_Flatlist_node *flat_buf,     /* offset-length info of this rank's write buffer */
            int                 *fileview_indx,
            ADIO_Offset         *offset_list,  /* this process's access offsets */
            ADIO_Offset         *len_list,     /* this process's access lengths */
      const int                 *send_size,    /* send_size[i] is amount of this rank sent to aggregator i in round iter */
      const int                 *recv_size,    /* recv_size[i] is amount of this rank recv from rank i in round iter */
            ADIO_Offset          real_off,     /* starting file offset of this process's write region in round iter */
            int                  real_size,    /* amount of this rank's write region in round iter */
      const int                 *recv_count,   /* recv_count[i] is No. offset-length pairs received from rank i in round iter */
      const int                 *start_pos,    /* start_pos[i] starting value of recv_curr_offlen_ptr[i] in round iter */
            int                  buftype_is_contig,
            int                  contig_access_count,
            int                  my_aggr_idx,
      const ADIOI_Access        *others_req,   /* others_req[i] is rank i's write requests fall into this rank's file domain */
            int                  iter,         /* round ID */
            ADIO_Offset          buftype_extent,
      const ADIO_Offset         *buf_idx,      /* indices to user buffer for sending this rank's write data to aggregator i */
            off_len_list        *srt_off_len,  /* OUT: list of writes by this rank in this round */
            MPI_Request         *reqs,         /* OUT: MPI nonblocking recv+send request IDs */
            int                 *nreqs,        /* OUT: number of nonblocking recv+send posted */
            int                 *error_code)   /* OUT: */
{
    char *buf_ptr, *contig_buf;
    int i, nprocs, myrank, nprocs_recv, nprocs_send, err;
    int sum_recv, hole, check_hole, cb_nodes, striping_unit;
    ADIO_Offset total_send_size_iter;
    MPI_Status status;
    static char myname[] = "ADIOI_LUSTRE_W_EXCHANGE_DATA";

    cb_nodes = fd->hints->cb_nodes;
    striping_unit = fd->hints->striping_unit;

    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);

    /* srt_off_len->num   OUT: number of elements in the above off-len list */
    /* srt_off_len->off[] OUT: list of write offsets by this rank in this round */
    /* srt_off_len->len[] OUT: list of write lengths by this rank in this round */

    /* calculate send receive metadata */
    srt_off_len->num = 0;
    srt_off_len->off = NULL;
    sum_recv = 0;
    nprocs_recv = 0;
    nprocs_send = 0;
    for (i = 0; i < nprocs; i++) {
        srt_off_len->num += recv_count[i];
        sum_recv += recv_size[i];
        if (recv_size[i])
            nprocs_recv++;
    }

    /* total send size of this round */
    total_send_size_iter = 0;
    for (i = 0; i < cb_nodes; i++) {
        total_send_size_iter += send_size[i];
        if (send_size[i])
            nprocs_send++;
    }

    /* determine whether checking holes is necessary */
    check_hole = 1;
    if (srt_off_len->num == 0) {
        /* this process has nothing to receive and hence no hole */
        check_hole = 0;
        hole = 0;
    } else if (fd->hints->ds_write == ADIOI_HINT_AUTO) {
        if (srt_off_len->num > fd->hints->ds_wr_lb) {
            /* Number of offset-length pairs is too large, making merge sort
             * expensive. Skip the sorting in hole checking and proceed with
             * read-modify-write.
             */
            check_hole = 0;
            hole = 1;
        }
        /* else: merge sort is less expensive, proceed to check_hole */
    }
    /* else: fd->hints->ds_write == ADIOI_HINT_ENABLE or ADIOI_HINT_DISABLE,
     * proceed to check_hole, as we must construct srt_off_len->off and srt_off_len->len.
     */

    if (check_hole) {
        /* merge the offset-length pairs of all others_req[] (already sorted
         * individually) into a single list of offset-length pairs (srt_off_len->off and
         * srt_off_len->len) in an increasing order of file offsets using a heap-merge
         * sorting algorithm.
         */
        srt_off_len->off = (ADIO_Offset *) ADIOI_Malloc(srt_off_len->num * sizeof(ADIO_Offset));
        srt_off_len->len = (int *) ADIOI_Malloc(srt_off_len->num * sizeof(int));

        heap_merge(others_req, recv_count, srt_off_len->off, srt_off_len->len, start_pos,
                   nprocs, nprocs_recv, &srt_off_len->num);

        /* srt_off_len->num has been updated in heap_merge() such that srt_off_len->off and
         * srt_off_len->len were coalesced
         */
        hole = (srt_off_len->num > 1);
    }

    /* data sieving */
    if (fd->hints->ds_write != ADIOI_HINT_DISABLE && hole) {
        ADIO_ReadContig(fd, write_buf, real_size, MPI_BYTE, ADIO_EXPLICIT_OFFSET,
                        real_off, &status, &err);
        if (err != MPI_SUCCESS) {
            *error_code = MPIO_Err_create_code(err,
                                               MPIR_ERR_RECOVERABLE,
                                               myname, __LINE__, MPI_ERR_IO, "**ioRMWrdwr", 0);
            return;
        }

        /* Once read, holes have been filled and thus the number of
         * offset-length pairs, srt_off_len->num, becomes one.
         */
        srt_off_len->num = 1;
        if (srt_off_len->off == NULL) {
            srt_off_len->off = (ADIO_Offset *) ADIOI_Malloc(sizeof(ADIO_Offset));
            srt_off_len->len = (int *) ADIOI_Malloc(sizeof(int));
        }
        srt_off_len->off[0] = real_off;
        srt_off_len->len[0] = real_size;
    }

    /* It is possible sum_recv (sum of message sizes to be received) is larger
     * than the size of collective buffer, write_buf, if writes from multiple
     * remote processes overlap. Receiving messages into overlapped regions of
     * the same write_buffer may cause a problem. To avoid it, we allocate a
     * temporary buffer big enough to receive all messages into disjointed
     * regions. Earlier in ADIOI_LUSTRE_Exch_and_write(), write_buf is already
     * allocated with twice amount of the file stripe size, wth the second half
     * to be used to receive messages. If sum_recv is smaller than file stripe
     * size, we can reuse that space. But if sum_recv is bigger (an overlap
     * case, which is rare), we allocate a separate buffer of size sum_recv.
     */
    sum_recv -= recv_size[myrank];
    if (sum_recv > striping_unit)
        *recv_buf = (char *) ADIOI_Realloc(*recv_buf, sum_recv);
    contig_buf = *recv_buf;

    /* nrecv/nsend is the number of Issend/Irecv to be posted */
    int nrecv = 0, nsend = 0;

    if (!fd->atomicity) {
        /* post receives and receive messages into contig_buf, a temporary
         * buffer */
        buf_ptr = contig_buf;
        for (i = 0; i < nprocs; i++) {
            if (recv_size[i] == 0)
                continue;
            if (i != myrank) {
                if (recv_count[i] > 1) {
                    MPI_Irecv(buf_ptr, recv_size[i], MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm,
                              &reqs[nrecv++]);
                    buf_ptr += recv_size[i];
                } else {
                    /* recv_count[i] is the number of noncontiguous
                     * offset-length pairs describing the write requests of
                     * process i that fall into this aggregator's file domain.
                     * When recv_count[i] is 1, there is only one such pair,
                     * meaning the receive message is to be stored
                     * contiguously. Such message can be received directly into
                     * write_buf.
                     */
                    MPI_Irecv(write_buf + others_req[i].mem_ptrs[start_pos[i]], recv_size[i],
                              MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm,
                              &reqs[nrecv++]);
                }
            } else if (buftype_is_contig) {
                /* send/recv to/from self uses memcpy()
                 * buftype_is_contig == 0 is handled at the send time below.
                 */
                char *fromBuf = (char *) buf + buf_idx[my_aggr_idx];
                MEMCPY_UNPACK(i, fromBuf, start_pos, recv_count, write_buf);
            }
        }
    }
    *nreqs = nrecv;
    reqs += nrecv;

    if (buftype_is_contig) {
        /* If buftype_is_contig, data can be directly sent from user buf
         * at location given by buf_idx.
         */
        for (i = 0; i < cb_nodes; i++) {
            int dest = fd->hints->ranklist[i];
            if (send_size[i] && i != my_aggr_idx) {
                ADIOI_Assert(buf_idx[i] != -1);
                MPI_Issend((char *) buf + buf_idx[i], send_size[i],
                           MPI_BYTE, dest, ADIOI_COLL_TAG(dest, iter),
                           fd->comm, &reqs[nsend++]);
            }
        }
    } else if (nprocs_send) {
        /* If buftype_is_contig is not true, allocate send_buf[] a
         * contiguous space, copy data to send_buf, including ones to be sent
         * to self, and then use send_buf to send.
         */
        size_t send_total_size = 0;
        for (i = 0; i < cb_nodes; i++)
            send_total_size += send_size[i];
        *send_buf = (char **) ADIOI_Malloc(cb_nodes * sizeof(char *));
        (*send_buf)[0] = (char *) ADIOI_Malloc(send_total_size);
        for (i = 1; i < cb_nodes; i++)
            (*send_buf)[i] = (*send_buf)[i - 1] + send_size[i - 1];

        ADIOI_LUSTRE_Fill_send_buffer(fd, buf, n_buftypes, flat_buf_idx,
                                      flat_buf_sz, flat_buf, (*send_buf),
                                      fileview_indx, offset_list, len_list,
                                      total_send_size_iter, send_size, reqs, &nsend,
                                      contig_access_count, my_aggr_idx, iter,
                                      buftype_extent);
        /* MPI_Issend calls have been posted in ADIOI_Fill_send_buffer. It is
         * possible nsend is 0. */
    }
    *nreqs += nsend;

    if (fd->atomicity) {
        /* In atomic mode, we must receive write data in the increasing order
         * of MPI process rank IDs,
         */
        for (i = 0; i < nprocs; i++) {
            if (recv_size[i] == 0)
                continue;
            if (i != myrank) {
                MPI_Recv(contig_buf, recv_size[i], MPI_BYTE, i, ADIOI_COLL_TAG(i, iter), fd->comm, &status);
                MEMCPY_UNPACK(i, contig_buf, start_pos, recv_count, write_buf);
            } else {
                /* send/recv to/from self uses memcpy() */
                char *ptr = (*send_buf)[my_aggr_idx];
                if (buftype_is_contig)
                    ptr = (char *) buf + buf_idx[my_aggr_idx];
                MEMCPY_UNPACK(i, ptr, start_pos, recv_count, write_buf);
            }
        }
        if (nsend > 0)
            AD_LUSTRE_WAITALL(nsend, reqs);
        *nreqs = 0;

    } else if (!buftype_is_contig) {
        if (my_aggr_idx >= 0 && send_size[my_aggr_idx] > 0)
            /* contents of user buf that must be sent to self has been copied
             * into send_buf[my_aggr_idx]. Now unpack it into write_buf.
             */
            MEMCPY_UNPACK(myrank, (*send_buf)[my_aggr_idx], start_pos, recv_count, write_buf);
    }

    /* free send_buf[] if allocated */
    if (*send_buf != NULL) {
        ADIOI_Free((*send_buf)[0]);
        ADIOI_Free(*send_buf);
        *send_buf = NULL;
    }
}


#define ADIOI_BUF_COPY \
{ \
    while (size) { \
        int size_in_buf = MPL_MIN(size, *flat_buf_sz); \
        copy_size_delayed += size_in_buf; \
        user_buf_idx += size_in_buf; \
        send_size_rem -= size_in_buf; \
        *flat_buf_sz -= size_in_buf; \
        if (*flat_buf_sz == 0) { /* move on to next flat_buf segment */ \
            if (send_size_rem) { /* after this copy send_buf[q] is still not full */ \
                isUserBuf = 0; \
                memcpy(send_buf_ptr, user_buf_ptr, copy_size_delayed); \
                send_buf_ptr += copy_size_delayed; \
                copy_size_delayed = 0; \
num_memcpy++; \
            } else if (isUserBuf == 0) { \
                /* send_buf[q] is full and not using user buf, \
                 * copy the remaining delayed data */ \
                memcpy(send_buf_ptr, user_buf_ptr, copy_size_delayed); \
            } \
            /* update flat_buf_idx, flat_buf_sz, n_buftypes, and user_buf_idx */ \
            if (*flat_buf_idx < (flat_buf->count - 1)) \
                (*flat_buf_idx)++; \
            else { \
                *flat_buf_idx = 0; \
                (*n_buftypes)++; \
            } \
            user_buf_idx = flat_buf->indices[*flat_buf_idx] + \
                (ADIO_Offset)(*n_buftypes)*buftype_extent; \
            *flat_buf_sz = flat_buf->blocklens[*flat_buf_idx]; \
            user_buf_ptr = (char*) buf + user_buf_idx; \
        } \
        else if (send_size_rem == 0 && isUserBuf == 0) { \
            /* *flat_buf_sz > 0, send_buf[q] is full, and not using user \
             * buf to send, copy the remaining delayed data */ \
            memcpy(send_buf_ptr, user_buf_ptr, copy_size_delayed); \
            user_buf_ptr += copy_size_delayed; \
        } \
        size -= size_in_buf; \
    } \
}

static void ADIOI_LUSTRE_Fill_send_buffer(ADIO_File fd,
                                          const void *buf,
                                          int *n_buftypes,
                                          int *flat_buf_idx,
                                          int *flat_buf_sz,
                                          const ADIOI_Flatlist_node *flat_buf,
                                          char **send_buf,
                                          int *fileview_indx,
                                          ADIO_Offset *offset_list,
                                          ADIO_Offset *len_list,
                                          ADIO_Offset total_send_size_iter,
                                          const int *send_size,
                                          MPI_Request *send_reqs,
                                          int *n_send_reqs, /* number of issend posted */
                                          int contig_access_count,
                                          int my_aggr_idx,
                                          int iter,
                                          ADIO_Offset buftype_extent)
{
    /* this function is only called if buftype is not contig */
    int ii, jj, q, size, send_size_rem=0;
    int first_q=-1, copy_size_delayed, isUserBuf;
    char *user_buf_ptr, *send_buf_ptr, *same_buf_ptr;
    ADIO_Offset off, len, rem_len, user_buf_idx;

    jj = 0;

    /* user_buf_idx is to the index offset to buf, indicating the starting
     * location to be copied..
     *
     * flat_buf stores the offset-length pairs of the flattened user buffer
     *     data type. Note this stores offset-length pairs of the data type,
     *     and write amount can be a multiple of the data type.
     * n_buftypes stores the current number of data types being processed.
     * flat_buf->count: the number of pairs
     * flat_buf->indices[i]: the ith pair's byte offset to buf. Note the
     *     flattened offsets of user buffer type may not be sorted in an
     *     increasing order, unlike fileview which is required by MPI to be
     *     sorted in a monotonically non-decreasing order.
     * flat_buf->blocklens[i]: length of the ith pair
     * flat_buf_idx: index to the offset-length pair currently being processed,
     *     incremented each round.
     * flat_buf_sz: amount of data in the pair that has not been copied over,
     *     changed each round.
     */
    user_buf_idx = flat_buf->indices[*flat_buf_idx]
                 + flat_buf->blocklens[*flat_buf_idx]
                 - (*flat_buf_sz);
                 /* in case data left to be copied from previous round */

int num_memcpy = 0;

    /* contig_access_count: the number of contiguous file segments this
     *     rank writes to. Each segment i is described by offset_list[i] and
     *     len_list[i].
     * fileview_indx: the index to the offset_list[], len_list[] that have been
     *     processed in the previous round.
     * For each contiguous off-len pair in this rank's file view, pack write
     * data into send buffers, send_buf[].
     */
    for (ii = *fileview_indx; ii < contig_access_count; ii++) {
        off = offset_list[ii];
        rem_len = len_list[ii];

        /* this off-len request may span to more than one I/O aggregator */
        while (rem_len != 0) {
            len = rem_len;
            q = ADIOI_LUSTRE_Calc_aggregator(fd, off, &len);
            /* NOTE: len will be modified by ADIOI_Calc_aggregator() to be no
             * more than a file stripe unit size that aggregator "q" is
             * responsible for. Note q is not the MPI rank ID, It is the array
             * index to fd->hints->ranklist[].
             *
             * Now len is the amount of data in ith off-len pair that should be
             * sent to aggregator q. Note q can also be self. In this case,
             * data is also packed into send_buf[q] or pointed to a segment of
             * buf when the data to be packed is contiguous. send_buf[q] will
             * later be copied to write buffer in MEMCPY_UNPACK, instead of
             * calling MPI_Issend to send.
             *
             * send_size[q]: data amount of this rank needs to send to
             * aggregator q in this round.
             *
             * len and send_size[q] are all always <= striping_unit
             */

            if (first_q != q) {
                ADIOI_Assert(send_size_rem == 0);
                first_q = q;
                isUserBuf = 1;
                send_size_rem = send_size[q];
                copy_size_delayed = 0;
                same_buf_ptr = (char*)buf + user_buf_idx; /* no increment */
                user_buf_ptr = same_buf_ptr; /* increment after each memcpy */
                send_buf_ptr = send_buf[q]; /* increment after each memcpy */
            }

            /* copy len amount of data from buf to send_buf[q] */
            size = (int)len;
            ADIOI_BUF_COPY

            if (send_size_rem == 0) { /* data to q is fully packed */
                first_q = -1;

                if (q != my_aggr_idx) { /* send only if not self rank */
                    /* get the aggregator's MPI rank ID */
                    int dest = fd->hints->ranklist[q];
                    if (isUserBuf)
                        MPI_Issend(same_buf_ptr, send_size[q], MPI_BYTE, dest,
                                   ADIOI_COLL_TAG(dest, iter), fd->comm, &send_reqs[jj++]);
                    else
                        MPI_Issend(send_buf[q], send_size[q], MPI_BYTE, dest,
                                   ADIOI_COLL_TAG(dest, iter), fd->comm, &send_reqs[jj++]);
                }
                else if (isUserBuf) { /* send buffer is also (part of) buf */
                    /* send to self and user buf is contiguous, then make a
                     * single memcpy from buf directly to send_buf[q]
                     */
                    memcpy(send_buf[q], same_buf_ptr, send_size[q]);
                }
            }
            /* len is the amount of data copied */
            off += len;
            rem_len -= len;
            offset_list[ii] += len;
            len_list[ii] -= len;
            total_send_size_iter -= len;
            if (total_send_size_iter == 0) goto done;
        }
    }
done:
    *n_send_reqs = jj;

    if (len_list[ii] == 0) ii++;
    *fileview_indx = ii;

printf("---- iter=%d contig_access_count=%d ii=%d n_buftypes=%lld fileview_indx=%d num_memcpy=%d\n",iter,contig_access_count,ii,*n_buftypes,*fileview_indx,num_memcpy);
}

/* This function calls ADIOI_OneSidedWriteAggregation iteratively to
 * essentially pack stripes of data into the collective buffer and then
 * flush the collective buffer to the file when fully packed, repeating this
 * process until all the data is written to the file.
 */
static void ADIOI_LUSTRE_IterateOneSided(ADIO_File fd, const void *buf, int *striping_info,
                                         ADIO_Offset * offset_list, ADIO_Offset * len_list,
                                         int contig_access_count, int currentValidDataIndex,
                                         int count, int file_ptr_type, ADIO_Offset offset,
                                         ADIO_Offset start_offset, ADIO_Offset end_offset,
                                         ADIO_Offset firstFileOffset, ADIO_Offset lastFileOffset,
                                         MPI_Datatype datatype, int myrank, int *error_code)
{
    int i;
    int stripesPerAgg = fd->hints->cb_buffer_size / striping_info[0];
    if (stripesPerAgg == 0) {
        /* The striping unit is larger than the collective buffer size
         * therefore we must abort since the buffer has already been
         * allocated during the open.
         */
        FPRINTF(stderr, "Error: The collective buffer size %d is less "
                "than the striping unit size %d - the ROMIO "
                "Lustre one-sided write aggregation algorithm "
                "cannot continue.\n", fd->hints->cb_buffer_size, striping_info[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Based on the co_ratio the number of aggregators we can use is the number of
     * stripes used in the file times this co_ratio - each stripe is written by
     * co_ratio aggregators this information is contained in the striping_info.
     */
    int numStripedAggs = striping_info[2];

    int orig_cb_nodes = fd->hints->cb_nodes;
    fd->hints->cb_nodes = numStripedAggs;

    /* Declare ADIOI_OneSidedStripeParms here - these parameters will be locally managed
     * for this invocation of ADIOI_LUSTRE_IterateOneSided.  This will allow for concurrent
     * one-sided collective writes via multi-threading as well as multiple communicators.
     */
    ADIOI_OneSidedStripeParms stripeParms;
    stripeParms.stripeSize = striping_info[0];
    stripeParms.stripedLastFileOffset = lastFileOffset;
    stripeParms.iWasUsedStripingAgg = 0;
    stripeParms.numStripesUsed = 0;
    stripeParms.amountOfStripedDataExpected = 0;
    stripeParms.bufTypeExtent = 0;
    stripeParms.lastDataTypeExtent = 0;
    stripeParms.lastFlatBufIndice = 0;
    stripeParms.lastIndiceOffset = 0;

    /* The general algorithm here is to divide the file up into segments, a segment
     * being defined as a contiguous region of the file which has up to one occurrence
     * of each stripe - the data for each stripe being written out by a particular
     * aggregator.  The segmentLen is the maximum size in bytes of each segment
     * (stripeSize*number of aggs).  Iteratively call ADIOI_OneSidedWriteAggregation
     * for each segment to aggregate the data to the collective buffers, but only do
     * the actual write (via flushCB stripe parm) once stripesPerAgg stripes
     * have been packed or the aggregation for all the data is complete, minimizing
     * synchronization.
     */
    stripeParms.segmentLen = ((ADIO_Offset) numStripedAggs) * ((ADIO_Offset) (striping_info[0]));

    /* These arrays define the file offsets for the stripes for a given segment - similar
     * to the concept of file domains in GPFS, essentially file domains for the segment.
     */
    ADIO_Offset *segment_stripe_start =
        (ADIO_Offset *) ADIOI_Malloc(numStripedAggs * sizeof(ADIO_Offset));
    ADIO_Offset *segment_stripe_end =
        (ADIO_Offset *) ADIOI_Malloc(numStripedAggs * sizeof(ADIO_Offset));

    /* Find the actual range of stripes in the file that have data in the offset
     * ranges being written -- skip holes at the front and back of the file.
     */
    int currentOffsetListIndex = 0;
    int fileSegmentIter = 0;
    int startingStripeWithData = 0;
    int foundStartingStripeWithData = 0;
    while (!foundStartingStripeWithData) {
        if (((startingStripeWithData + 1) * (ADIO_Offset) (striping_info[0])) > firstFileOffset)
            foundStartingStripeWithData = 1;
        else
            startingStripeWithData++;
    }

    ADIO_Offset currentSegementOffset =
        (ADIO_Offset) startingStripeWithData * (ADIO_Offset) (striping_info[0]);

    int numSegments =
        (int) ((lastFileOffset + (ADIO_Offset) 1 - currentSegementOffset) / stripeParms.segmentLen);
    if ((lastFileOffset + (ADIO_Offset) 1 - currentSegementOffset) % stripeParms.segmentLen > 0)
        numSegments++;

    /* To support read-modify-write use a while-loop to redo the aggregation if necessary
     * to fill in the holes.
     */
    int doAggregation = 1;
    int holeFound = 0;

    /* Remember romio_onesided_no_rmw setting if we have to re-do
     * the aggregation if holes are found.
     */
    int prev_romio_onesided_no_rmw = fd->romio_onesided_no_rmw;

    while (doAggregation) {

        int totalDataWrittenLastRound = 0;

        /* This variable tracks how many segment stripes we have packed into the agg
         * buffers so we know when to flush to the file system.
         */
        stripeParms.segmentIter = 0;

        /* stripeParms.stripesPerAgg is the number of stripes to aggregate before doing a flush.
         */
        stripeParms.stripesPerAgg = stripesPerAgg;
        if (stripeParms.stripesPerAgg > numSegments)
            stripeParms.stripesPerAgg = numSegments;

        for (fileSegmentIter = 0; fileSegmentIter < numSegments; fileSegmentIter++) {

            int dataWrittenThisRound = 0;

            /* Define the segment range in terms of file offsets.
             */
            ADIO_Offset segmentFirstFileOffset = currentSegementOffset;
            if ((currentSegementOffset + stripeParms.segmentLen - (ADIO_Offset) 1) > lastFileOffset)
                currentSegementOffset = lastFileOffset;
            else
                currentSegementOffset += (stripeParms.segmentLen - (ADIO_Offset) 1);
            ADIO_Offset segmentLastFileOffset = currentSegementOffset;
            currentSegementOffset++;

            ADIO_Offset segment_stripe_offset = segmentFirstFileOffset;
            for (i = 0; i < numStripedAggs; i++) {
                if (firstFileOffset > segment_stripe_offset)
                    segment_stripe_start[i] = firstFileOffset;
                else
                    segment_stripe_start[i] = segment_stripe_offset;
                if ((segment_stripe_offset + (ADIO_Offset) (striping_info[0])) > lastFileOffset)
                    segment_stripe_end[i] = lastFileOffset;
                else
                    segment_stripe_end[i] =
                        segment_stripe_offset + (ADIO_Offset) (striping_info[0]) - (ADIO_Offset) 1;
                segment_stripe_offset += (ADIO_Offset) (striping_info[0]);
            }

            /* In the interest of performance for non-contiguous data with large offset lists
             * essentially modify the given offset and length list appropriately for this segment
             * and then pass pointers to the sections of the lists being used for this segment
             * to ADIOI_OneSidedWriteAggregation.  Remember how we have modified the list for this
             * segment, and then restore it appropriately after processing for this segment has
             * concluded, so it is ready for the next segment.
             */
            int segmentContigAccessCount = 0;
            int startingOffsetListIndex = -1;
            int endingOffsetListIndex = -1;
            ADIO_Offset startingOffsetAdvancement = 0;
            ADIO_Offset startingLenTrim = 0;
            ADIO_Offset endingLenTrim = 0;

            while (((offset_list[currentOffsetListIndex] +
                     ((ADIO_Offset) (len_list[currentOffsetListIndex])) - (ADIO_Offset) 1) <
                    segmentFirstFileOffset) && (currentOffsetListIndex < (contig_access_count - 1)))
                currentOffsetListIndex++;
            startingOffsetListIndex = currentOffsetListIndex;
            endingOffsetListIndex = currentOffsetListIndex;
            int offsetInSegment = 0;
            ADIO_Offset offsetStart = offset_list[currentOffsetListIndex];
            ADIO_Offset offsetEnd =
                (offset_list[currentOffsetListIndex] +
                 ((ADIO_Offset) (len_list[currentOffsetListIndex])) - (ADIO_Offset) 1);

            if (len_list[currentOffsetListIndex] == 0)
                offsetInSegment = 0;
            else if ((offsetStart >= segmentFirstFileOffset) &&
                     (offsetStart <= segmentLastFileOffset)) {
                offsetInSegment = 1;
            } else if ((offsetEnd >= segmentFirstFileOffset) &&
                       (offsetEnd <= segmentLastFileOffset)) {
                offsetInSegment = 1;
            } else if ((offsetStart <= segmentFirstFileOffset) &&
                       (offsetEnd >= segmentLastFileOffset)) {
                offsetInSegment = 1;
            }

            if (!offsetInSegment) {
                segmentContigAccessCount = 0;
            } else {
                /* We are in the segment, advance currentOffsetListIndex until we are out of segment.
                 */
                segmentContigAccessCount = 1;

                while ((offset_list[currentOffsetListIndex] <= segmentLastFileOffset) &&
                       (currentOffsetListIndex < contig_access_count)) {
                    dataWrittenThisRound += (int) len_list[currentOffsetListIndex];
                    currentOffsetListIndex++;
                }

                if (currentOffsetListIndex > startingOffsetListIndex) {
                    /* If we did advance, if we are at the end need to check if we are still in segment.
                     */
                    if (currentOffsetListIndex == contig_access_count) {
                        currentOffsetListIndex--;
                    } else if (offset_list[currentOffsetListIndex] > segmentLastFileOffset) {
                        /* We advanced into the last one and it still in the segment.
                         */
                        currentOffsetListIndex--;
                    } else {
                        dataWrittenThisRound += (int) len_list[currentOffsetListIndex];
                    }
                    segmentContigAccessCount += (currentOffsetListIndex - startingOffsetListIndex);
                    endingOffsetListIndex = currentOffsetListIndex;
                }
            }

            if (segmentContigAccessCount > 0) {
                /* Trim edges here so all data in the offset list range fits exactly in the segment.
                 */
                if (offset_list[startingOffsetListIndex] < segmentFirstFileOffset) {
                    startingOffsetAdvancement =
                        segmentFirstFileOffset - offset_list[startingOffsetListIndex];
                    offset_list[startingOffsetListIndex] += startingOffsetAdvancement;
                    dataWrittenThisRound -= (int) startingOffsetAdvancement;
                    startingLenTrim = startingOffsetAdvancement;
                    len_list[startingOffsetListIndex] -= startingLenTrim;
                }

                if ((offset_list[endingOffsetListIndex] +
                     ((ADIO_Offset) (len_list[endingOffsetListIndex])) - (ADIO_Offset) 1) >
                    segmentLastFileOffset) {
                    endingLenTrim =
                        offset_list[endingOffsetListIndex] +
                        ((ADIO_Offset) (len_list[endingOffsetListIndex])) - (ADIO_Offset) 1 -
                        segmentLastFileOffset;
                    len_list[endingOffsetListIndex] -= endingLenTrim;
                    dataWrittenThisRound -= (int) endingLenTrim;
                }
            }

            int holeFoundThisRound = 0;

            /* Once we have packed the collective buffers do the actual write.
             */
            if ((stripeParms.segmentIter == (stripeParms.stripesPerAgg - 1)) ||
                (fileSegmentIter == (numSegments - 1))) {
                stripeParms.flushCB = 1;
            } else
                stripeParms.flushCB = 0;

            stripeParms.firstStripedWriteCall = 0;
            stripeParms.lastStripedWriteCall = 0;
            if (fileSegmentIter == 0) {
                stripeParms.firstStripedWriteCall = 1;
            } else if (fileSegmentIter == (numSegments - 1))
                stripeParms.lastStripedWriteCall = 1;

            /* The difference in calls to ADIOI_OneSidedWriteAggregation is based on the whether the buftype is
             * contiguous.  The algorithm tracks the position in the source buffer when called
             * multiple times --  in the case of contiguous data this is simple and can be externalized with
             * a buffer offset, in the case of non-contiguous data this is complex and the state must be tracked
             * internally, therefore no external buffer offset.  Care was taken to minimize
             * ADIOI_OneSidedWriteAggregation changes at the expense of some added complexity to the caller.
             */
            int bufTypeIsContig;
            ADIOI_Datatype_iscontig(datatype, &bufTypeIsContig);
            if (bufTypeIsContig) {
                ADIOI_OneSidedWriteAggregation(fd,
                                               (ADIO_Offset *) &
                                               (offset_list[startingOffsetListIndex]),
                                               (ADIO_Offset *) &
                                               (len_list[startingOffsetListIndex]),
                                               segmentContigAccessCount,
                                               buf + totalDataWrittenLastRound, datatype,
                                               error_code, segmentFirstFileOffset,
                                               segmentLastFileOffset, currentValidDataIndex,
                                               segment_stripe_start, segment_stripe_end,
                                               &holeFoundThisRound, &stripeParms);
                /* numNonZeroDataOffsets is not used in ADIOI_OneSidedWriteAggregation()? */
            } else {
                ADIOI_OneSidedWriteAggregation(fd,
                                               (ADIO_Offset *) &
                                               (offset_list[startingOffsetListIndex]),
                                               (ADIO_Offset *) &
                                               (len_list[startingOffsetListIndex]),
                                               segmentContigAccessCount, buf, datatype, error_code,
                                               segmentFirstFileOffset, segmentLastFileOffset,
                                               currentValidDataIndex, segment_stripe_start,
                                               segment_stripe_end, &holeFoundThisRound,
                                               &stripeParms);
            }

            if (stripeParms.flushCB) {
                stripeParms.segmentIter = 0;
                if (stripesPerAgg > (numSegments - fileSegmentIter - 1))
                    stripeParms.stripesPerAgg = numSegments - fileSegmentIter - 1;
                else
                    stripeParms.stripesPerAgg = stripesPerAgg;
            } else
                stripeParms.segmentIter++;

            if (holeFoundThisRound)
                holeFound = 1;

            /* If we know we won't be doing a pre-read in a subsequent call to
             * ADIOI_OneSidedWriteAggregation which will have a barrier to keep
             * feeder ranks from doing rma to the collective buffer before the
             * write completes that we told it do with the stripeParms.flushCB
             * flag then we need to do a barrier here.
             */
            if (!fd->romio_onesided_always_rmw && stripeParms.flushCB) {
                if (fileSegmentIter < (numSegments - 1)) {
                    MPI_Barrier(fd->comm);
                }
            }

            /* Restore the offset_list and len_list to values that are ready for the
             * next iteration.
             */
            if (segmentContigAccessCount > 0) {
                offset_list[endingOffsetListIndex] += len_list[endingOffsetListIndex];
                len_list[endingOffsetListIndex] = endingLenTrim;
            }
            totalDataWrittenLastRound += dataWrittenThisRound;
        }       // fileSegmentIter for-loop

        /* Check for holes in the data unless romio_onesided_no_rmw is set.
         * If a hole is found redo the entire aggregation and write.
         */
        if (!fd->romio_onesided_no_rmw) {
            int anyHolesFound = 0;
            MPI_Allreduce(&holeFound, &anyHolesFound, 1, MPI_INT, MPI_MAX, fd->comm);

            if (anyHolesFound) {
                ADIOI_Free(offset_list);
                ADIOI_Calc_my_off_len(fd, count, datatype, file_ptr_type, offset,
                                      &offset_list, &len_list, &start_offset,
                                      &end_offset, &contig_access_count);

                currentSegementOffset =
                    (ADIO_Offset) startingStripeWithData *(ADIO_Offset) (striping_info[0]);
                fd->romio_onesided_always_rmw = 1;
                fd->romio_onesided_no_rmw = 1;

                /* Holes are found in the data and the user has not set
                 * romio_onesided_no_rmw --- set romio_onesided_always_rmw to 1
                 * and redo the entire aggregation and write and if the user has
                 * romio_onesided_inform_rmw set then inform him of this condition
                 * and behavior.
                 */
                if (fd->romio_onesided_inform_rmw && (myrank == 0)) {
                    FPRINTF(stderr, "Information: Holes found during one-sided "
                            "write aggregation algorithm --- re-running one-sided "
                            "write aggregation with ROMIO_ONESIDED_ALWAYS_RMW set to 1.\n");
                }
            } else
                doAggregation = 0;
        } else
            doAggregation = 0;
    }   // while doAggregation
    fd->romio_onesided_no_rmw = prev_romio_onesided_no_rmw;

    ADIOI_Free(segment_stripe_start);
    ADIOI_Free(segment_stripe_end);

    fd->hints->cb_nodes = orig_cb_nodes;

}


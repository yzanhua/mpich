/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "ad_lustre.h"
#include "adio_extern.h"

#undef AGG_DEBUG

void ADIOI_LUSTRE_Get_striping_info(ADIO_File fd, int *striping_info, int mode)
{
    /* get striping information:
     *  striping_info[0]: stripe_size
     *  striping_info[1]: stripe_count
     *  striping_info[2]: avail_cb_nodes
     */
    int stripe_size, stripe_count, CO = 1;
    int avail_cb_nodes, divisor, nprocs_for_coll = fd->hints->cb_nodes;

    /* Get hints value */
    /* stripe size */
    stripe_size = fd->hints->striping_unit;
    /* stripe count */
    /* stripe_size and stripe_count have been validated in ADIOI_LUSTRE_Open() */
    stripe_count = fd->hints->striping_factor;

    /* Calculate the available number of I/O clients */
    if (!mode) {
        /* for collective read,
         * if "CO" clients access the same OST simultaneously,
         * the OST disk seek time would be much. So, to avoid this,
         * it might be better if 1 client only accesses 1 OST.
         * So, we set CO = 1 to meet the above requirement.
         */
        CO = 1;
        /*XXX: maybe there are other better way for collective read */
    } else {
        /* CO also has been validated in ADIOI_LUSTRE_Open(), >0 */
        CO = fd->hints->fs_hints.lustre.co_ratio;
    }
    /* Calculate how many IO clients we need */
    /* Algorithm courtesy Pascal Deveze (pascal.deveze@bull.net) */
    /* To avoid extent lock conflicts,
     * avail_cb_nodes should either
     *  - be a multiple of stripe_count,
     *  - or divide stripe_count exactly
     * so that each OST is accessed by a maximum of CO constant clients. */
    if (nprocs_for_coll >= stripe_count)
        /* avail_cb_nodes should be a multiple of stripe_count and the number
         * of procs per OST should be limited to the minimum between
         * nprocs_for_coll/stripe_count and CO
         *
         * e.g. if stripe_count=20, nprocs_for_coll=42 and CO=3 then
         * avail_cb_nodes should be equal to 40 */
        avail_cb_nodes = stripe_count * MPL_MIN(nprocs_for_coll / stripe_count, CO);
    else {
        /* nprocs_for_coll is less than stripe_count */
        /* avail_cb_nodes should divide stripe_count */
        /* e.g. if stripe_count=60 and nprocs_for_coll=8 then
         * avail_cb_nodes should be egal to 6 */
        /* This could be done with :
         * while (stripe_count % avail_cb_nodes != 0) avail_cb_nodes--;
         * but this can be optimized for large values of nprocs_for_coll and
         * stripe_count */
        divisor = 2;
        avail_cb_nodes = 1;
        /* try to divise */
        while (stripe_count >= divisor * divisor) {
            if ((stripe_count % divisor) == 0) {
                if (stripe_count / divisor <= nprocs_for_coll) {
                    /* The value is found ! */
                    avail_cb_nodes = stripe_count / divisor;
                    break;
                }
                /* if divisor is less than nprocs_for_coll, divisor is a
                 * solution, but it is not sure that it is the best one */
                else if (divisor <= nprocs_for_coll)
                    avail_cb_nodes = divisor;
            }
            divisor++;
        }
    }

    striping_info[0] = stripe_size;
    striping_info[1] = stripe_count;
    striping_info[2] = avail_cb_nodes;
}

int ADIOI_LUSTRE_Calc_aggregator(ADIO_File fd, ADIO_Offset off, ADIO_Offset * len)
{
    ADIO_Offset avail_bytes, stripe_id;

    stripe_id = off / fd->hints->striping_unit;

    avail_bytes = (stripe_id + 1) * fd->hints->striping_unit - off;
    if (avail_bytes < *len) {
        /* The request [off, off+len) has only [off, off+avail_bytes) part
         * falling into aggregator's file domain */
        *len = avail_bytes;
    }
    /* return the index to ranklist[] */
    return (stripe_id % fd->hints->cb_nodes);
}

/* ADIOI_LUSTRE_Calc_my_req() - calculate what portions of the access requests
 * of this process are located in the file domains of I/O aggregators.
 */
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

    int striping_unit = fd->hints->striping_unit;
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

int ADIOI_LUSTRE_Docollect(ADIO_File fd, int contig_access_count,
                           ADIO_Offset * len_list, int nprocs)
{
    /* If the processes are non-interleaved, we will check the req_size.
     *   if (avg_req_size > big_req_size) {
     *       docollect = 0;
     *   }
     */

    int i, docollect = 1, big_req_size = 0;
    ADIO_Offset req_size = 0, total_req_size;
    int avg_req_size, total_access_count;

    /* calculate total_req_size and total_access_count */
    for (i = 0; i < contig_access_count; i++)
        req_size += len_list[i];
    MPI_Allreduce(&req_size, &total_req_size, 1, MPI_LONG_LONG_INT, MPI_SUM, fd->comm);
    MPI_Allreduce(&contig_access_count, &total_access_count, 1, MPI_INT, MPI_SUM, fd->comm);
    /* avoid possible divide-by-zero) */
    if (total_access_count != 0) {
        /* estimate average req_size */
        avg_req_size = (int) (total_req_size / total_access_count);
    } else {
        avg_req_size = 0;
    }
    /* get hint of big_req_size */
    big_req_size = fd->hints->fs_hints.lustre.coll_threshold;
    /* Don't perform collective I/O if there are big requests */
    if ((big_req_size > 0) && (avg_req_size > big_req_size))
        docollect = 0;

    return docollect;
}

void ADIOI_LUSTRE_Free_my_req(int nprocs, int *count_my_req_per_proc,
                              ADIOI_Access * my_req, ADIO_Offset ** buf_idx)
{
    ADIOI_Free(count_my_req_per_proc);
    ADIOI_Free(buf_idx[0]);
    ADIOI_Free(buf_idx);
    ADIOI_Free(my_req);
}

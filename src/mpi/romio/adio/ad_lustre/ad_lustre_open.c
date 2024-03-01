/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "ad_lustre.h"
#ifndef MIMIC_LUSTRE
#include <lustre/lustreapi.h>

/* what is the basis for this define?
 * what happens if there are more than 1k UUIDs? */

#define MAX_LOV_UUID_COUNT      1000

int ADIOI_LUSTRE_clear_locks(ADIO_File fd);     /* in ad_lustre_lock.c */
int ADIOI_LUSTRE_request_only_lock_ioctl(ADIO_File fd); /* in ad_lustre_lock.c */

static int __u32_compare(const void *a, const void *b)
{
     if (*(__u32*)a > *(__u32*)b) return (1);
     if (*(__u32*)a < *(__u32*)b) return (-1);
     return (0);
}
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif
static void *alloc_lum()
{
    int v1, v3, join;

    v1 = sizeof(struct lov_user_md_v1) +
         LOV_MAX_STRIPE_COUNT * sizeof(struct lov_user_ost_data_v1);
    v3 = sizeof(struct lov_user_md_v3) +
         LOV_MAX_STRIPE_COUNT * sizeof(struct lov_user_ost_data_v1);

    return malloc(MAX(v1, v3));
}
static int num_uniq_osts(const char *path)
{
    struct lov_user_md *lum_file = NULL;
    int rc, i, num;
    __u32 *ost_idx;

    lum_file = alloc_lum();

    rc = llapi_file_get_stripe(path, lum_file);
    assert(rc == 0);

    ost_idx = (__u32*) malloc(lum_file->lmm_stripe_count * sizeof(__u32));
    for (i=0; i<lum_file->lmm_stripe_count; i++)
        ost_idx[i] = lum_file->lmm_objects[i].l_ost_idx;
    qsort(ost_idx, lum_file->lmm_stripe_count, sizeof(__u32), __u32_compare);
    num = 0;
    for (i=1; i<lum_file->lmm_stripe_count; i++) {
        if (ost_idx[i] > ost_idx[num]) ost_idx[++num] = ost_idx[i];
    }
    num++;
    free(ost_idx);
    free(lum_file);
    return num;
}
#endif

void ADIOI_LUSTRE_Open(ADIO_File fd, int *error_code)
{
    int perm, old_mask, amode, amode_direct, root;
    int lumlen, myrank, flag, set_layout = 0, err;
#ifndef MIMIC_LUSTRE
    struct lov_user_md *lum = NULL;
#endif
    char *value;
    ADIO_Offset str_factor = -1, str_unit = 0, start_iodev = -1;
    size_t value_sz = (MPI_MAX_INFO_VAL + 1) * sizeof(char);
    static char myname[] = "ADIOI_LUSTRE_OPEN";

    MPI_Comm_rank(fd->comm, &myrank);

    if (fd->perm == ADIO_PERM_NULL) {
        old_mask = umask(022);
        umask(old_mask);
        perm = old_mask ^ 0666;
    } else
        perm = fd->perm;

    amode = 0;
    if (fd->access_mode & ADIO_CREATE)
        amode = amode | O_CREAT;
    if (fd->access_mode & ADIO_RDONLY)
        amode = amode | O_RDONLY;
    if (fd->access_mode & ADIO_WRONLY)
        amode = amode | O_WRONLY;
    if (fd->access_mode & ADIO_RDWR)
        amode = amode | O_RDWR;
    if (fd->access_mode & ADIO_EXCL)
        amode = amode | O_EXCL;

    amode_direct = amode | O_DIRECT;

#ifndef MIMIC_LUSTRE
    /* odd length here because lov_user_md contains some fixed data and
     * then a list of 'lmm_objects' representing stripe */
    lumlen = sizeof(struct lov_user_md) + MAX_LOV_UUID_COUNT * sizeof(struct lov_user_ost_data);
    lum = (struct lov_user_md *) ADIOI_Calloc(1, lumlen);
#endif

    value = (char *) ADIOI_Malloc(value_sz);
    /* we already validated in LUSTRE_SetInfo that these are going to be the same */
    if (fd->info != MPI_INFO_NULL) {
        /* striping information */
        ADIOI_Info_get(fd->info, "striping_unit", MPI_MAX_INFO_VAL, value, &flag);
        if (flag)
            str_unit = atoll(value);

        ADIOI_Info_get(fd->info, "striping_factor", MPI_MAX_INFO_VAL, value, &flag);
        if (flag)
            str_factor = atoll(value);

        ADIOI_Info_get(fd->info, "start_iodevice", MPI_MAX_INFO_VAL, value, &flag);
        if (flag)
            start_iodev = atoll(value);
    }
    if ((str_factor > 0) || (str_unit > 0) || (start_iodev >= 0))
        set_layout = 1;

#ifndef MIMIC_LUSTRE
    /* if hints were set, we need to delay creation of any lustre objects.
     * However, if we open the file with O_LOV_DELAY_CREATE and don't call the
     * follow-up ioctl, subsequent writes will fail */
    if (myrank == 0 && set_layout)
        amode = amode | O_LOV_DELAY_CREATE;
#endif

    fd->fd_sys = open(fd->filename, amode, perm);
    if (fd->fd_sys == -1)
        goto fn_exit;

    root = (fd->hints->ranklist == NULL) ? 0 : fd->hints->ranklist[0];

#ifdef MIMIC_LUSTRE
#define xstr(s) str(s)
#define str(s) #s
#define STRIPE_SIZE 1024
#define STRIPE_COUNT 4

        fd->hints->striping_unit = STRIPE_SIZE;
        ADIOI_Info_set(fd->info, "striping_unit", xstr(STRIPE_SIZE));

        fd->hints->striping_factor = STRIPE_COUNT;
        ADIOI_Info_set(fd->info, "striping_factor", xstr(STRIPE_COUNT));

        fd->hints->start_iodevice = 0;
        ADIOI_Info_set(fd->info, "romio_lustre_start_iodevice", "0");

        fd->hints->fs_hints.lustre.num_osts = STRIPE_COUNT;
        ADIOI_Info_set(fd->info, "lustre_num_osts", xstr(STRIPE_COUNT));

#else

    /* we can only set these hints on new files */
    /* It was strange and buggy to open the file in the hint path.  Instead,
     * we'll apply the file tunings at open time */
    if ((amode & O_CREAT) && set_layout) {
        /* if user has specified striping info, first aggregator tries to set
         * it */
        if (myrank == root || fd->comm == MPI_COMM_SELF) {
            lum->lmm_magic = LOV_USER_MAGIC;
            lum->lmm_pattern = 0;
            /* crude check for overflow of lustre internal datatypes.
             * Silently cap to large value if user provides a value
             * larger than lustre supports */
            if (str_unit > UINT_MAX)
                lum->lmm_stripe_size = UINT_MAX;
            else
                lum->lmm_stripe_size = str_unit;

            if (str_factor > USHRT_MAX)
                lum->lmm_stripe_count = USHRT_MAX;
            else
                lum->lmm_stripe_count = str_factor;

            if (start_iodev > USHRT_MAX)
                lum->lmm_stripe_offset = USHRT_MAX;
            else
                lum->lmm_stripe_offset = start_iodev;
            err = ioctl(fd->fd_sys, LL_IOC_LOV_SETSTRIPE, lum);
            if (err == -1 && errno != EEXIST) {
                fprintf(stderr, "Failure to set stripe info %s \n", strerror(errno));
                /* not a fatal error, but user might care to know */
            }
        }       /* End of striping parameters validation */
    }

    if (myrank == root || fd->comm == MPI_COMM_SELF) {
        /* Pascal Deveze reports that, even though we pass a "GETSTRIPE" (read)
         * flag to the ioctl, if some of the values of this struct are
         * uninitialized, the call can give an error.  zero it out in case
         * there are other members that must be initialized and in case
         * lov_user_md struct changes in future.
         */
        memset(lum, 0, lumlen);
        lum->lmm_magic = LOV_USER_MAGIC;
        err = ioctl(fd->fd_sys, LL_IOC_LOV_GETSTRIPE, (void *) lum);
        if (!err) {
            /* striping hints will be set later after return to ADIO_Open() */
            fd->hints->striping_unit = lum->lmm_stripe_size;
            fd->hints->striping_factor = lum->lmm_stripe_count;
            fd->hints->start_iodevice = lum->lmm_stripe_offset;
            fd->hints->fs_hints.lustre.num_osts = num_uniq_osts(fd->filename);
        }
    }
#endif

#ifdef WKL_DEBUG
if ((fd->access_mode & ADIO_CREATE) && myrank == 0) printf("%2d: %s line %3d striping ---- unit=%d factor=%d\n",myrank,__func__,__LINE__,fd->hints->striping_unit,fd->hints->striping_factor);
#endif

    if (fd->access_mode & ADIO_APPEND)
        fd->fp_ind = fd->fp_sys_posn = lseek(fd->fd_sys, 0, SEEK_END);

    fd->fd_direct = -1;
    if (fd->direct_write || fd->direct_read) {
        fd->fd_direct = open(fd->filename, amode_direct, perm);
        if (fd->fd_direct != -1) {
            fd->d_mem = fd->d_miniosz = (1 << 12);
        } else {
            perror("cannot open file with O_Direct");
            fd->direct_write = fd->direct_read = 0;
        }
    }
#ifdef HAVE_LUSTRE_LOCKAHEAD
    if (fd->hints->fs_hints.lustre.lock_ahead_read || fd->hints->fs_hints.lustre.lock_ahead_write) {
        ADIOI_LUSTRE_clear_locks(fd);
        ADIOI_LUSTRE_request_only_lock_ioctl(fd);
    }
#endif


  fn_exit:
#ifndef MIMIC_LUSTRE
    ADIOI_Free(lum);
#endif
    ADIOI_Free(value);
    /* --BEGIN ERROR HANDLING-- */
    if (fd->fd_sys == -1 || ((fd->fd_direct == -1) && (fd->direct_write || fd->direct_read))) {
        *error_code = ADIOI_Err_create_code(myname, fd->filename, errno);
    }
    /* --END ERROR HANDLING-- */
    else
        *error_code = MPI_SUCCESS;

}

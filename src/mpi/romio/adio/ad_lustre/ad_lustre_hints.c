/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "ad_lustre.h"
#include "adio_extern.h"
#include "hint_fns.h"
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

void ADIOI_LUSTRE_SetInfo(ADIO_File fd, MPI_Info users_info, int *error_code)
{
    int flag;
    static char myname[] = "ADIOI_LUSTRE_SETINFO";
    char *p = NULL;

    if (fd->hints->initialized && users_info == MPI_INFO_NULL) {
        *error_code = MPI_SUCCESS;
        return;
    }

    if (!fd->hints->initialized) {
        /* Hints here are the ones that can only be set at MPI_File_open */
        MPI_Info_create(&(fd->info));

        ADIOI_Info_set(fd->info, "direct_read", "false");
        ADIOI_Info_set(fd->info, "direct_write", "false");
        fd->direct_read = fd->direct_write = 0;
        /* initialize lustre hints */
        ADIOI_Info_set(fd->info, "romio_lustre_co_ratio", "1");
        fd->hints->fs_hints.lustre.co_ratio = 1;
        ADIOI_Info_set(fd->info, "romio_lustre_coll_threshold", "0");
        fd->hints->fs_hints.lustre.coll_threshold = 0;
        fd->hints->fs_hints.lustre.phase_shuffle = 0;

#ifdef HAVE_LUSTRE_LOCKAHEAD
        /* Set lock ahead default hints */
        fd->hints->fs_hints.lustre.lock_ahead_read = 0;
        fd->hints->fs_hints.lustre.lock_ahead_write = 0;
        fd->hints->fs_hints.lustre.lock_ahead_num_extents = 500;
        fd->hints->fs_hints.lustre.lock_ahead_flags = 0;
#endif

        /* has user specified striping or server buffering parameters
         * and do they have the same value on all processes? */
        if (users_info != MPI_INFO_NULL) {
            char *value;
            value = (char *) ADIOI_Malloc((MPI_MAX_INFO_VAL + 1) * sizeof(char));

            /* striping information */
            ADIOI_Info_get(users_info, "striping_unit", MPI_MAX_INFO_VAL, value, &flag);
            if (flag)
                ADIOI_Info_set(fd->info, "striping_unit", value);

            ADIOI_Info_get(users_info, "striping_factor", MPI_MAX_INFO_VAL, value, &flag);
            if (flag)
                ADIOI_Info_set(fd->info, "striping_factor", value);

            ADIOI_Info_get(users_info, "start_iodevice", MPI_MAX_INFO_VAL, value, &flag);
            if (flag)
                ADIOI_Info_set(fd->info, "start_iodevice", value);


            /* direct read and write */
            ADIOI_Info_get(users_info, "direct_read", MPI_MAX_INFO_VAL, value, &flag);
            if (flag && (!strcmp(value, "true") || !strcmp(value, "TRUE"))) {
                ADIOI_Info_set(fd->info, "direct_read", "true");
                fd->direct_read = 1;
            }
            ADIOI_Info_get(users_info, "direct_write", MPI_MAX_INFO_VAL, value, &flag);
            if (flag && (!strcmp(value, "true") || !strcmp(value, "TRUE"))) {
                ADIOI_Info_set(fd->info, "direct_write", "true");
                fd->direct_write = 1;
            }

            ADIOI_Info_check_and_install_int (fd, users_info, "romio_lustre_phase_shuffle",
                                              &(fd->hints->fs_hints.lustre.phase_shuffle), myname,
                                              error_code);
            p = getenv ("ROMIO_LUSTRE_PHASE_SHUFFLE");
            if (p != NULL) { fd->hints->fs_hints.lustre.phase_shuffle = atoi (p); }

#ifdef HAVE_LUSTRE_LOCKAHEAD
            /* Get lock ahead hints */

            ADIOI_Info_check_and_install_int(fd, users_info,
                                             "romio_lustre_cb_lock_ahead_write",
                                             &(fd->hints->fs_hints.lustre.lock_ahead_write),
                                             myname, error_code);
            ADIOI_Info_check_and_install_int(fd, users_info,
                                             "romio_lustre_cb_lock_ahead_read",
                                             &(fd->hints->fs_hints.lustre.lock_ahead_read),
                                             myname, error_code);

            /* If, and only if, we're using lock ahead,
             * process/set the number of extents to pre-lock and the flags */
            if (fd->hints->fs_hints.lustre.lock_ahead_read ||
                fd->hints->fs_hints.lustre.lock_ahead_write) {
                /* Get user's number of extents */
                ADIOI_Info_check_and_install_int(fd, users_info,
                                                 "romio_lustre_cb_lock_ahead_num_extents",
                                                 &(fd->hints->fs_hints.
                                                   lustre.lock_ahead_num_extents), myname,
                                                 error_code);

                /* ADIOI_Info_check_and_install_int doesn't set the
                 * value in fd unless it was in user_info, but knowing
                 * the value - default or explicit - is useful.
                 * Set the final number of extents in the fd->info */
                MPL_snprintf(value, MPI_MAX_INFO_VAL + 1, "%d",
                             fd->hints->fs_hints.lustre.lock_ahead_num_extents);
                ADIOI_Info_set(fd->info, "romio_lustre_cb_lock_ahead_num_extents", value);

                /* Get user's flags */
                ADIOI_Info_check_and_install_int(fd, users_info,
                                                 "romio_lustre_cb_lock_ahead_flags",
                                                 &(fd->hints->fs_hints.lustre.lock_ahead_flags),
                                                 myname, error_code);
            }
#endif
            ADIOI_Free(value);
        }
    }

    /* hints below are allowed in both MPI_File_open and MPI_File_setview */
    if (users_info != MPI_INFO_NULL) {
        /* CO: IO Clients/OST,
         * to keep the load balancing between clients and OSTs */
        ADIOI_Info_check_and_install_int(fd, users_info, "romio_lustre_co_ratio",
                                         &(fd->hints->fs_hints.lustre.co_ratio), myname,
                                         error_code);

        /* coll_threshold:
         * if the req size is bigger than this, collective IO may not be performed.
         */
        ADIOI_Info_check_and_install_int(fd, users_info, "romio_lustre_coll_threshold",
                                         &(fd->hints->fs_hints.lustre.coll_threshold), myname,
                                         error_code);
    }
    /* set the values for collective I/O and data sieving parameters */
    ADIOI_GEN_SetInfo(fd, users_info, error_code);

    if (ADIOI_Direct_read)
        fd->direct_read = 1;
    if (ADIOI_Direct_write)
        fd->direct_write = 1;

    *error_code = MPI_SUCCESS;
}

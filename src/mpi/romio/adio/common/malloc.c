/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* These are routines for allocating and deallocating memory.
   They should be called as ADIOI_Malloc(size) and
   ADIOI_Free(ptr). In adio.h, they are macro-replaced to
   ADIOI_Malloc(size,__LINE__,__FILE__) and
   ADIOI_Free(ptr,__LINE__,__FILE__).

   Later on, add some tracing and error checking, similar to
   MPID_trmalloc. */

#include "adio.h"
#include <stdlib.h>
#include <stdio.h>
#include "mpipr.h"

/* for the style checker */
/* style: allow:malloc:1 sig:0 */
/* style: allow:free:1 sig:0 */
/* style: allow:calloc:1 sig:0 */
/* style: allow:realloc:1 sig:0 */

#define FPRINTF fprintf

void *ADIOI_Malloc_fn(size_t size, int lineno, const char *fname);
void *ADIOI_Calloc_fn(size_t nelem, size_t elsize, int lineno, const char *fname);
void *ADIOI_Realloc_fn(void *ptr, size_t size, int lineno, const char *fname);
void ADIOI_Free_fn(void *ptr, int lineno, const char *fname);

#ifdef WKL_DEBUG
#include <search.h> /* tfind(), tsearch() and tdelete() */
/* static variables for malloc tracing (initialized to 0s) */
static void   *wkl_mem_root;
static size_t  wkl_mem_alloc;
static size_t  wkl_max_mem_alloc;
typedef struct {
    void   *self;
    void   *buf;
    size_t  size;
    int     lineno;
    char   *filename;
} wkl_mem_entry;

static
int wkl_cmp(const void *a, const void *b) {
    wkl_mem_entry *fa = (wkl_mem_entry*)a;
    wkl_mem_entry *fb = (wkl_mem_entry*)b;

    if (fa->buf > fb->buf) return  1;
    if (fa->buf < fb->buf) return -1;
    return 0;
}

static
void walker(const void *node, const VISIT which, const int depth) {
    wkl_mem_entry *f;
    f = *(wkl_mem_entry **)node;
    if (which == preorder || which == leaf)
        fprintf(stdout, "Warning: malloc yet to be freed (buf=%p size=%zd filename=%s line=%d)\n", f->buf, f->size, f->filename, f->lineno);
}

/*----< wkl_add_mem_entry() >----------------------------------------------*/
/* add a new malloc entry to the table */
static
void wkl_add_mem_entry(void       *buf,
                          size_t      size,
                          const int   lineno,
                          const char *filename)
{
    /* use C tsearch utility */
    wkl_mem_entry *node = (wkl_mem_entry*) malloc(sizeof(wkl_mem_entry));
    node->self     = node;
    node->buf      = buf;
    node->size     = size;
    node->lineno   = lineno;
    node->filename = (char*)malloc(strlen(filename)+1);
    strcpy(node->filename, filename);
    node->filename[strlen(filename)] = '\0';

    /* search and add a new item */
    void *ret = tsearch(node, &wkl_mem_root, wkl_cmp);
    if (ret == NULL) {
        fprintf(stderr, "Error at line %d file %s: tsearch()\n",
                __LINE__,__FILE__);
    }
    else {
        wkl_mem_alloc += size;
        wkl_max_mem_alloc = (wkl_max_mem_alloc > wkl_mem_alloc) ? wkl_max_mem_alloc : wkl_mem_alloc;
    }
}

/*----< wkl_del_mem_entry() >---------------------------------------------*/
/* delete a malloc entry from the table */
static
void wkl_del_mem_entry(void *buf)
{
    /* use C tsearch utility */
    if (wkl_mem_root != NULL) {
        wkl_mem_entry node;
        node.buf  = buf;
        void *ret = tfind(&node, &wkl_mem_root, wkl_cmp);
        wkl_mem_entry **found = (wkl_mem_entry**) ret;
        if (ret == NULL) {
            fprintf(stderr, "Error at line %d file %s: tfind() buf=%p\n",
                    __LINE__,__FILE__,buf);
            goto fn_exit;
        }
        /* free space for filename */
        free((*found)->filename);

        /* subtract the space amount to be freed */
        wkl_mem_alloc -= (*found)->size;
        void *tmp = (*found)->self;
        ret = tdelete(&node, &wkl_mem_root, wkl_cmp);
        if (ret == NULL) {
            fprintf(stderr, "Error at line %d file %s: tdelete() buf=%p\n",
                    __LINE__,__FILE__,buf);
            goto fn_exit;
        }
        free(tmp);
    }
    else
        fprintf(stderr, "Error at line %d file %s: wkl_mem_root is NULL\n",
                __LINE__,__FILE__);
fn_exit:
    return;
}

/*----< ncmpi_inq_malloc_max_size() >----------------------------------------*/
/* This is an independent subroutine.
 * get the max watermark ever researched by malloc (aggregated amount) */
void wkl_malloc_reset(void) {
    wkl_max_mem_alloc = 0;
    wkl_mem_alloc = 0;
}

void wkl_inq_malloc_max_size(MPI_Offset *size)
{ *size = (MPI_Offset)wkl_max_mem_alloc; }

void wkl_inq_malloc_size(MPI_Offset *size)
{ *size = (MPI_Offset)wkl_mem_alloc; }

void wkl_inq_malloc_list(void)
{
    /* check if malloc tree is empty */
    if (wkl_mem_root != NULL)
        twalk(wkl_mem_root, walker);
}

#endif

void *ADIOI_Malloc_fn(size_t size, int lineno, const char *fname)
{
    void *new;

#ifdef ROMIO_XFS
    new = (void *) memalign(XFS_MEMALIGN, size);
#else
    new = (void *) MPL_malloc(size, MPL_MEM_IO);
#endif
    if (!new && size) {
        FPRINTF(stderr, "Out of memory in file %s, line %d\n", fname, lineno);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPL_VG_MEM_INIT(new, size);
    DBG_FPRINTF(stderr, "ADIOI_Malloc %s:<%d> %p (%#zX)\n", fname, lineno, new, size);
#ifdef WKL_DEBUG
    wkl_add_mem_entry(new, size, lineno, fname);
#endif

    return new;
}


void *ADIOI_Calloc_fn(size_t nelem, size_t elsize, int lineno, const char *fname)
{
    void *new;

    new = (void *) MPL_calloc(nelem, elsize, MPL_MEM_IO);
    if (!new && nelem) {
        FPRINTF(stderr, "Out of memory in file %s, line %d\n", fname, lineno);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    DBG_FPRINTF(stderr, "ADIOI_Calloc %s:<%d> %p\n", fname, lineno, new);
#ifdef WKL_DEBUG
    wkl_add_mem_entry(new, nelem * elsize, lineno, fname);
#endif

    return new;
}


void *ADIOI_Realloc_fn(void *ptr, size_t size, int lineno, const char *fname)
{
    void *new;

#ifdef WKL_DEBUG
    wkl_del_mem_entry(ptr);
#endif

    new = (void *) MPL_realloc(ptr, size, MPL_MEM_IO);
    if (!new && size) {
        FPRINTF(stderr, "realloc failed in file %s, line %d\n", fname, lineno);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    DBG_FPRINTF(stderr, "ADIOI_Realloc %s:<%d> %p\n", fname, lineno, new);
#ifdef WKL_DEBUG
    wkl_add_mem_entry(new, size, lineno, fname);
#endif


    return new;
}


void ADIOI_Free_fn(void *ptr, int lineno, const char *fname)
{
    DBG_FPRINTF(stderr, "ADIOI_Free %s:<%d> %p\n", fname, lineno, ptr);
    if (!ptr) {
        FPRINTF(stderr, "Attempt to free null pointer in file %s, line %d\n", fname, lineno);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef WKL_DEBUG
    wkl_del_mem_entry(ptr);
#endif

    MPL_free(ptr);
}

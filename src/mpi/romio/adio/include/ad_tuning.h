/* ---------------------------------------------------------------- */
/* (C)Copyright IBM Corp.  2007, 2008                               */
/* ---------------------------------------------------------------- */
/**
 * \file ad_tuning.h
 * \brief Defines common performance tuning env var options
 */

/*---------------------------------------------------------------------
 * ad_tuning.h
 *
 * declares common global variables and functions for performance tuning
 *---------------------------------------------------------------------*/

#ifndef AD_TUNING_H_INCLUDED
#define AD_TUNING_H_INCLUDED

#include "adio.h"


/*-----------------------------------------
 *  Global variables for the control of performance tuning.
 *-----------------------------------------*/

/* set internal variables for tuning environment variables */
void ad_get_env_vars(ADIO_File fd);

#endif /* AD_TUNING_H_INCLUDED */

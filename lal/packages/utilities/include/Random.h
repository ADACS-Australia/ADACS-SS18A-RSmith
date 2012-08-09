/*
*  Copyright (C) 2007 Alexander Dietz, Jolien Creighton
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with with program; see the file COPYING. If not, write to the
*  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*  MA  02111-1307  USA
*/

#ifndef _RANDOM_H
#define _RANDOM_H

#include <lal/LALDatatypes.h>
#include <lal/AVFactories.h>

#ifdef  __cplusplus
extern "C" {
#endif


/**
 * \addtogroup Random_h
 * \author Creighton, J. D. E. and Tibbits, M. M.
 *
 * \brief Generates random numbers.
 *
 * \heading{Synopsis}
 * \code
 * #include <lal/Random.h>
 * \endcode
 *
 * This header covers the routines for generating random numbers.
 *
*/
/*@{*/

/**\name Error Codes */
/*@{*/
#define RANDOMH_ENULL 1		/**< Null pointer */
#define RANDOMH_ENNUL 2		/**< Non-null pointer */
#define RANDOMH_ESIZE 4		/**< Invalid size */
#define RANDOMH_ELNTH 8		/**< Must have more than one data point */
#define RANDOMH_ESEGZ 16		/**< Invalid number of segments */
#define RANDOMH_ENUMZ 32		/**< Invalid number of points in segment */
#define RANDOMH_EALOC 64		/**< Memory Allocation Error */
#define RANDOMH_EINIT 128		/**< Params must be initialized with CreateParams first */
#define RANDOMH_EZERO 256		/**< Output Vector length must be greater than zero */
#define RANDOMH_ESEED 512		/**< Improper seed value */
/*@}*/

/*@}*/

/** \cond DONT_DOXYGEN */
#define RANDOMH_MSGENULL "Null pointer"
#define RANDOMH_MSGENNUL "Non-null pointer"
#define RANDOMH_MSGESIZE "Invalid size"
#define RANDOMH_MSGELNTH "Must have more than one data point"
#define RANDOMH_MSGESEGZ "Invalid number of segments"
#define RANDOMH_MSGENUMZ "Invalid number of points in segment"
#define RANDOMH_MSGEALOC "Memory Allocation Error"
#define RANDOMH_MSGEINIT "Params must be initialized with CreateParams first"
#define RANDOMH_MSGEZERO "Output Vector length must be greater than zero"
#define RANDOMH_MSGESEED "Improper seed value"
/** \endcond */

/** \ingroup Random_h
 * \brief This structure contains the parameters necessary for generating the current
 * sequence of random numbers (based on the initial seed).
 * \note The contents should not be manually adjusted.
 */
typedef struct
tagRandomParams
{
  INT4 i;
  INT4 y;
  INT4 v[32];
}
RandomParams;

typedef struct tagMTRandomParams MTRandomParams;


INT4 XLALBasicRandom( INT4 i );
RandomParams * XLALCreateRandomParams( INT4 seed );
void XLALResetRandomParams( RandomParams *params, INT4 seed );
void XLALDestroyRandomParams( RandomParams *params );
REAL4 XLALUniformDeviate( RandomParams *params );
int XLALNormalDeviates( REAL4Vector *deviates, RandomParams *params );
REAL4 XLALNormalDeviate( RandomParams *params );

void
LALCreateRandomParams (
    LALStatus        *status,
    RandomParams **params,
    INT4           seed
    );

void
LALDestroyRandomParams (
    LALStatus        *status,
    RandomParams **params
    );

void
LALUniformDeviate (
    LALStatus    *status,
    REAL4        *deviate,
    RandomParams *params
    );

void
LALNormalDeviates (
    LALStatus    *status,
    REAL4Vector  *deviates,
    RandomParams *params
    );

void LALMersenneRandom(
    LALStatus      *status,
    REAL8          *output,
    MTRandomParams *params
    );

void LALMersenneRandomVector(
    LALStatus      *status,
    REAL8Vector    *output,
    MTRandomParams *params
    );

void LALCreateMTRandomParams(
    LALStatus       *status,
    REAL8            seed,
    MTRandomParams **params
    );

void LALDestroyMTRandomParams(
    LALStatus       *status,
    MTRandomParams **params
    );

#ifdef  __cplusplus
}
#endif

#endif /* _RANDOM_H */

/*
 * Copyright (C) 2008 Karl Wette
 * Copyright (C) 2005 Reinhard Prix
 *
 *  [partially based on the MSG_LOG class in BOINC:
 *  Copyright (C) 2005 University of California]
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

#ifndef _LOGPRINTF_H  /* Double-include protection. */
#define _LOGPRINTF_H

/* C++ protection. */
#ifdef  __cplusplus
extern "C" {
#endif

/*---------- INCLUDES ----------*/
#include <stdarg.h>
#include <gsl/gsl_matrix.h>

#include <lal/LALDatatypes.h>

/**
 * \defgroup LogPrintf_h Header LogPrintf.h
 * \ingroup lal_support
 * \author Reinhard Prix
 * \date 2005
 * \brief General-purpose log-message handling, mostly modelled after the MSG_LOG class in BOINC.
 *
 */
/*@{*/

/*---------- DEFINES ----------*/
/*---------- TYPES ----------*/

/** Argument-type for LogPrintf(): determines log-level of this message */
typedef enum
  {
    LOG_NONE = 0,	/**< internal: don't use */
    LOG_CRITICAL,	/**< log-level for critical errors */
    LOG_NORMAL,		/**< 'normal' log-level */
    LOG_DEBUG,		/**< debug log-level */
    LOG_DETAIL,		/**< detailed log-level */
    LOG_LAST		/**< internal: don't use */
  } LogLevel_t;

/*---------- GLOBALs ----------*/

/*---------- PROTOTYPES [API] ----------*/
LogLevel_t LogLevel(void);

void LogPrintf (LogLevel_t, const char* format, ...) _LAL_GCC_PRINTF_FORMAT_(2,3);
void LogPrintfVerbatim (LogLevel_t, const char* format, ...) _LAL_GCC_PRINTF_FORMAT_(2,3);

int XLALfprintfGSLmatrix ( FILE *fp, const char *fmt, const gsl_matrix *gij ) _LAL_GCC_VPRINTF_FORMAT_(2);
int XLALfprintfGSLvector ( FILE *fp, const char *fmt, const gsl_vector *vect ) _LAL_GCC_VPRINTF_FORMAT_(2);
int XLALfprintfGSLvector_int ( FILE *fp, const char *fmt, const gsl_vector_int *vect ) _LAL_GCC_VPRINTF_FORMAT_(2);

REAL8 XLALGetTimeOfDay(void);
REAL8 XLALGetCPUTime ( void );

REAL8 XLALGetPeakHeapUsageMB ( void );

char * XLALClearLinebreaks ( const char *str );


int XLALdumpREAL4TimeSeries (const char *fname, const REAL4TimeSeries *series);
int XLALdumpREAL8TimeSeries (const char *fname, const REAL8TimeSeries *series);
int XLALdumpCOMPLEX8TimeSeries (const char *fname, const COMPLEX8TimeSeries *series );

/*@}*/

#ifdef  __cplusplus
}
#endif
/* C++ protection. */

#endif  /* Double-include protection. */

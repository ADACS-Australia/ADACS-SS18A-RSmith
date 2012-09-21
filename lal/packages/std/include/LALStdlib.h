/*
*  Copyright (C) 2007 Jolien Creighton
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

/**
\author J. D. E. Creighton, T. D. Creighton
\addtogroup LALStdlib_h

\brief Includes the standard LAL header files.

\heading{Synopsis}
\code
#include <lal/LALStdlib.h>
\endcode

This header is the overall header for the \c std
package.  It provides the datatypes, constants, and macros required by
most LAL functions, by including the following header files in the
\c std package:

\code
#include <lal/LALDatatypes.h>
#include <lal/LALStatusMacros.h>
\endcode

This header also includes function prototype headers for certain standard modules used
by many LAL routines:

\code
#include <stdio.h>
#include <stdarg.h>
#include <lal/LALMalloc.h>
\endcode

*/

#ifndef _LALSTDLIB_H
#define _LALSTDLIB_H

#include <lal/LALDatatypes.h>
#include <lal/LALStatusMacros.h>

#include <stdio.h>
#include <stdarg.h>
#include <lal/LALMalloc.h>

/* Redefine the restict keyword when compiling with C++ */
#ifdef  __cplusplus
#ifdef __GNUC__
#define restrict __restrict__
#else
#define restrict
#endif
#endif

#ifdef  __cplusplus
extern "C" {
#endif

/* These are non-ANSI standard routines that will be allowed in LAL */
#ifndef SWIG /* exclude from SWIG interface */
int getopt( int, char * const *, const char * );
FILE *popen( const char *, const char * );
int pclose( FILE * );
#endif /* SWIG */

#ifdef  __cplusplus
}
#endif

#endif /* _LALSTDLIB_H */

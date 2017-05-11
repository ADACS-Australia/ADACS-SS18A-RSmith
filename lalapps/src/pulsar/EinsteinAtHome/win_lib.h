/*
*  Copyright (C) 2004-2008 Bernd Machenschalk
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

/* Header file for MS-Windows specific functions */

#ifdef _WIN32
#include <time.h>

/* chdir mapped to the hidden _chdir */
#include <direct.h>
#define chdir _chdir

/* some more functions with just different names */
#define fsync _commit
#define putenv _putenv

/* functions actually implemented in win_lib.cpp */
#ifdef __cplusplus
extern "C" {
#endif
  /* functions implemented based on Windows API counterparts */
  extern unsigned int sleep(unsigned int s);
  extern char *index(const char *s, int c);
  extern struct tm *gmtime_r(const time_t *t, struct tm *s);
  extern char *asctime_r(const struct tm *t, char *s);
  extern char *strsep (char **, const char *);
  /* quick fix for LAL now using finite without checking if we have it... */
  extern int finite(double x);
  /* get a "signaling" NaN for testing FPEs & signal(-handler) */
  extern float get_float_snan(void);
  /* a different, more "atomic" implementation of boinc_rename() */
  extern int eah_rename(const char* oldf, const char* newf);
#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */

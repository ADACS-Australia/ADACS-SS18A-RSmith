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

#ifndef _LALSTRING_H
#define _LALSTRING_H

#include <stddef.h>
#include <lal/LALAtomicDatatypes.h>

#ifdef  __cplusplus
extern "C" {
#elif 0
}       /* so that editors will match preceding brace */
#endif

/**
 * \defgroup LALString_h Header LALString.h
 * \ingroup lal_std
 * \author Creighton, J. D. E.
 * \brief XLAL string manipulation routines.
 *//*@{*/

char *XLALStringAppend(char *s, const char *append);
char *XLALStringDuplicate(const char *s);
size_t XLALStringCopy(char *dst, const char *src, size_t size);
size_t XLALStringConcatenate(char *dst, const char *src, size_t size);
int XLALStringToLowerCase(char * string);
int XLALStringToUpperCase(char * string);
int XLALStringCaseCompare(const char *s1, const char *s2);
int XLALStringNCaseCompare(const char *s1, const char *s2, size_t n);

/*@}*/

#if 0
{       /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif
#endif /* _LALSTRING_H */

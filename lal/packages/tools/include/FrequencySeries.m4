/*
 * $Id$
 *
 * Copyright (C) 2007  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#ifndef _FREQUENCYSERIES_H
#define _FREQUENCYSERIES_H


#include <stddef.h>
#include <lal/LALDatatypes.h>
#include <lal/LALRCSID.h>

#ifdef  __cplusplus
extern "C" {
#pragma }
#endif

NRCSID(FREQUENCYSERIESH, "$Id:");

define(`DATATYPE',COMPLEX8)
include(FrequencySeriesH.m4)

define(`DATATYPE',COMPLEX16)
include(FrequencySeriesH.m4)

define(`DATATYPE',REAL4)
include(FrequencySeriesH.m4)

define(`DATATYPE',REAL8)
include(FrequencySeriesH.m4)

define(`DATATYPE',INT2)
include(FrequencySeriesH.m4)

define(`DATATYPE',INT4)
include(FrequencySeriesH.m4)

define(`DATATYPE',INT8)
include(FrequencySeriesH.m4)

define(`DATATYPE',UINT2)
include(FrequencySeriesH.m4)

define(`DATATYPE',UINT4)
include(FrequencySeriesH.m4)

define(`DATATYPE',UINT8)
include(FrequencySeriesH.m4)

#ifdef __cplusplus
#pragma {
}
#endif

#endif  /* _FREQUENCYSERIES_H */

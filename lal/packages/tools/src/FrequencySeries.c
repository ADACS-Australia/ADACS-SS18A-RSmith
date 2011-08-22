/*
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


#include <math.h>
#include <string.h>
#include <lal/Date.h>
#include <lal/LALComplex.h>
#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/FrequencySeries.h>
#include <lal/Sequence.h>
#include <lal/Units.h>
#include <lal/XLALError.h>


#include <lal/LALRCSID.h>
NRCSID(FREQUENCYSERIESC, "$Id$");

#define DATATYPE REAL4
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE REAL8
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE COMPLEX8
#define ARG arg1->data->data[i].re += arg2->data->data[j].re * unit_ratio; \
            arg1->data->data[i].im += arg2->data->data[j].im * unit_ratio;
#define ARG1 arg1->data->data[i] = LAL_COMPLEX8_ZERO;
#define ARG2 REAL4 re = arg2->data->data[j].re * unit_ratio; \
             REAL4 im = arg2->data->data[j].im * unit_ratio; \
             arg1->data->data[i].re = arg1->data->data[i].re * re - arg1->data->data[i].im * im; \
             arg1->data->data[i].im = arg1->data->data[i].re * im + arg1->data->data[i].im * re;
#include "FrequencySeries_source.c"
#include "FrequencySeriesComplex_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE COMPLEX16
#define ARG arg1->data->data[i].re += arg2->data->data[j].re * unit_ratio; \
            arg1->data->data[i].im += arg2->data->data[j].im * unit_ratio;
#define ARG1 arg1->data->data[i] = LAL_COMPLEX16_ZERO;
#define ARG2 REAL8 re = arg2->data->data[j].re * unit_ratio; \
             REAL8 im = arg2->data->data[j].im * unit_ratio; \
             arg1->data->data[i].re = arg1->data->data[i].re * re - arg1->data->data[i].im * im; \
             arg1->data->data[i].im = arg1->data->data[i].re * im + arg1->data->data[i].im * re;
#include "FrequencySeries_source.c"
#include "FrequencySeriesComplex_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE INT2
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE UINT2
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE INT4
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE UINT4
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE INT8
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

#define DATATYPE UINT8
#define ARG arg1->data->data[i] += arg2->data->data[j] * unit_ratio;
#define ARG1 arg1->data->data[i] = 0.0;
#define ARG2 ARG;
#include "FrequencySeries_source.c"
#undef DATATYPE
#undef ARG
#undef ARG1
#undef ARG2

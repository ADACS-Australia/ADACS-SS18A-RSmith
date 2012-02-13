/*
*  Copyright (C) 2007 Jolien Creighton, Teviet Creighton
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
\author Creighton, T. D.
\file

\heading{Header \ref StreamInput.h}
\latexonly\label{s_StreamInput_h}\endlatexonly

Provides routines to read data from an open stream and store it in LAL
data structures.

\heading{Synopsis}
\code
#include "StreamInput.h"
\endcode

This header provides prototypes for routines that construct
LAL data structures using the data from a file (or other I/O) stream.
The routines do not provide a system-level interface to create files
and open or close file streams; they simply assume that they have been
passed an open, readable stream.  Nonetheless, because they involve
I/O stream manipulation, these routines are placed in the
\c lalsupport library rather than in \c lal proper.

The routines in \ref StreamVectorInput.c and
\ref StreamVectorSequenceInput.c are compartmentalized in such a way
that they can easily be converted if the LAL specification later
changes the way in which I/O streams are handled.  In partucular, the
only file I/O commands used are <tt>fgets()</tt> and <tt>feof()</tt>.
Thus the upgrade would involve only the following global changes:
<ol>
<li> Replace all occurrences of <tt>FILE *</tt> with the name of the
LAL I/O stream pointer type.</li>
<li> Replace all occurrences of <tt>fgets()</tt> and <tt>feof()</tt> with
equivalent LAL functions.</li>
</ol>
In particular, there is no need to translate routines such as
<tt>fscanf()</tt>; one should simply read data into a LAL
\c CHARVector and then use <tt>sscanf()</tt> to format the input.
This is the approach used in the numerical input routines in
\ref StreamVectorInput.c and \ref StreamVectorSequenceInput.c.

The routines in \ref StreamSequenceInput.c are less robust but much
more efficient: they use <tt>fscanf()</tt> to parse the input stream
directly.  They are intended primarily for test programs that may need
to read large datafiles of undetermined length.  The routines in
\ref StreamSeriesInput.c and \ref StreamGridInput.c also parse the
input stream directly using <tt>fscanf()</tt>, to avoid potentially
crippling computational overhead.

*/

#ifndef _STREAMINPUT_H
#define _STREAMINPUT_H

#include <lal/LALStdlib.h>
#include <lal/Grid.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
\heading{Error conditions}
 \name Error Codes */ /*@{*/
#define STREAMINPUTH_ENUL  1
#define STREAMINPUTH_EOUT  2
#define STREAMINPUTH_EMEM  3
#define STREAMINPUTH_ELEN  4
#define STREAMINPUTH_ESLEN 5
#define STREAMINPUTH_EVLEN 6
#define STREAMINPUTH_EDLEN 7
#define STREAMINPUTH_EDIM  8
#define STREAMINPUTH_EFMT  9
#define STREAMINPUTH_EBUF 10

#define STREAMINPUTH_MSGENUL  "Unexpected null pointer in arguments"
#define STREAMINPUTH_MSGEOUT  "Output handle points to a non-null pointer"
#define STREAMINPUTH_MSGEMEM  "Memory allocation error"
#define STREAMINPUTH_MSGELEN  "No numbers were read"
#define STREAMINPUTH_MSGESLEN "Not enough numbers read to fill sequence"
#define STREAMINPUTH_MSGEVLEN "Could not determine complex vectorLength"
#define STREAMINPUTH_MSGEDLEN "Dimension lengths inconsistent or not given"
#define STREAMINPUTH_MSGEDIM  "Inconsistent or non-positive arrayDim value"
#define STREAMINPUTH_MSGEFMT  "Badly formatted number"
#define STREAMINPUTH_MSGEBUF  "BUFFSIZE not a multiple of largest complex type size"
/*@}*//**

\heading{Types}

*/





/* Function prototypes. */




void
LALCHARReadVector( LALStatus  *status, CHARVector **vector, FILE *stream );

void
LALI2ReadVector( LALStatus  *status, INT2Vector **vector, FILE *stream, BOOLEAN strict );

void
LALI4ReadVector( LALStatus  *status, INT4Vector **vector, FILE *stream, BOOLEAN strict );

void
LALI8ReadVector( LALStatus  *status, INT8Vector **vector, FILE *stream, BOOLEAN strict );

void
LALU2ReadVector( LALStatus  *status, UINT2Vector **vector, FILE *stream, BOOLEAN strict );

void
LALU4ReadVector( LALStatus  *status, UINT4Vector **vector, FILE *stream, BOOLEAN strict );

void
LALU8ReadVector( LALStatus  *status, UINT8Vector **vector, FILE *stream, BOOLEAN strict );

void
LALSReadVector( LALStatus  *status, REAL4Vector **vector, FILE *stream, BOOLEAN strict );

void
LALDReadVector( LALStatus  *status, REAL8Vector **vector, FILE *stream, BOOLEAN strict );




void
LALCHARReadVectorSequence( LALStatus  *status, CHARVectorSequence **sequence, FILE *stream );

void
LALI2ReadVectorSequence( LALStatus  *status, INT2VectorSequence **sequence, FILE *stream );

void
LALI4ReadVectorSequence( LALStatus  *status, INT4VectorSequence **sequence, FILE *stream );

void
LALI8ReadVectorSequence( LALStatus  *status, INT8VectorSequence **sequence, FILE *stream );

void
LALU2ReadVectorSequence( LALStatus  *status, UINT2VectorSequence **sequence, FILE *stream );

void
LALU4ReadVectorSequence( LALStatus  *status, UINT4VectorSequence **sequence, FILE *stream );

void
LALU8ReadVectorSequence( LALStatus  *status, UINT8VectorSequence **sequence, FILE *stream );

void
LALSReadVectorSequence( LALStatus  *status, REAL4VectorSequence **sequence, FILE *stream );

void
LALDReadVectorSequence( LALStatus  *status, REAL8VectorSequence **sequence, FILE *stream );




void
LALCHARReadSequence( LALStatus *status, CHARSequence **sequence, FILE *stream );

void
LALI2ReadSequence( LALStatus *status, INT2Sequence **sequence, FILE *stream );

void
LALI4ReadSequence( LALStatus *status, INT4Sequence **sequence, FILE *stream );

void
LALI8ReadSequence( LALStatus *status, INT8Sequence **sequence, FILE *stream );

void
LALU2ReadSequence( LALStatus *status, UINT2Sequence **sequence, FILE *stream );

void
LALU4ReadSequence( LALStatus *status, UINT4Sequence **sequence, FILE *stream );

void
LALU8ReadSequence( LALStatus *status, UINT8Sequence **sequence, FILE *stream );

void
LALSReadSequence( LALStatus *status, REAL4Sequence **sequence, FILE *stream );

void
LALDReadSequence( LALStatus *status, REAL8Sequence **sequence, FILE *stream );

void
LALCReadSequence( LALStatus *status, COMPLEX8Sequence **sequence, FILE *stream );

void
LALZReadSequence( LALStatus *status, COMPLEX16Sequence **sequence, FILE *stream );




void
LALI2ReadTSeries( LALStatus *status, INT2TimeSeries *series, FILE *stream );
void
LALI4ReadTSeries( LALStatus *status, INT4TimeSeries *series, FILE *stream );
void
LALI8ReadTSeries( LALStatus *status, INT8TimeSeries *series, FILE *stream );
void
LALU2ReadTSeries( LALStatus *status, UINT2TimeSeries *series, FILE *stream );
void
LALU4ReadTSeries( LALStatus *status, UINT4TimeSeries *series, FILE *stream );
void
LALU8ReadTSeries( LALStatus *status, UINT8TimeSeries *series, FILE *stream );
void
LALSReadTSeries( LALStatus *status, REAL4TimeSeries *series, FILE *stream );
void
LALDReadTSeries( LALStatus *status, REAL8TimeSeries *series, FILE *stream );
void
LALCReadTSeries( LALStatus *status, COMPLEX8TimeSeries *series, FILE *stream );
void
LALZReadTSeries( LALStatus *status, COMPLEX16TimeSeries *series, FILE *stream );

void
LALI2ReadTVectorSeries( LALStatus *status, INT2TimeVectorSeries *series, FILE *stream );
void
LALI4ReadTVectorSeries( LALStatus *status, INT4TimeVectorSeries *series, FILE *stream );
void
LALI8ReadTVectorSeries( LALStatus *status, INT8TimeVectorSeries *series, FILE *stream );
void
LALU2ReadTVectorSeries( LALStatus *status, UINT2TimeVectorSeries *series, FILE *stream );
void
LALU4ReadTVectorSeries( LALStatus *status, UINT4TimeVectorSeries *series, FILE *stream );
void
LALU8ReadTVectorSeries( LALStatus *status, UINT8TimeVectorSeries *series, FILE *stream );
void
LALSReadTVectorSeries( LALStatus *status, REAL4TimeVectorSeries *series, FILE *stream );
void
LALDReadTVectorSeries( LALStatus *status, REAL8TimeVectorSeries *series, FILE *stream );
void
LALCReadTVectorSeries( LALStatus *status, COMPLEX8TimeVectorSeries *series, FILE *stream );
void
LALZReadTVectorSeries( LALStatus *status, COMPLEX16TimeVectorSeries *series, FILE *stream );

void
LALI2ReadTArraySeries( LALStatus *status, INT2TimeArraySeries *series, FILE *stream );
void
LALI4ReadTArraySeries( LALStatus *status, INT4TimeArraySeries *series, FILE *stream );
void
LALI8ReadTArraySeries( LALStatus *status, INT8TimeArraySeries *series, FILE *stream );
void
LALU2ReadTArraySeries( LALStatus *status, UINT2TimeArraySeries *series, FILE *stream );
void
LALU4ReadTArraySeries( LALStatus *status, UINT4TimeArraySeries *series, FILE *stream );
void
LALU8ReadTArraySeries( LALStatus *status, UINT8TimeArraySeries *series, FILE *stream );
void
LALSReadTArraySeries( LALStatus *status, REAL4TimeArraySeries *series, FILE *stream );
void
LALDReadTArraySeries( LALStatus *status, REAL8TimeArraySeries *series, FILE *stream );
void
LALCReadTArraySeries( LALStatus *status, COMPLEX8TimeArraySeries *series, FILE *stream );
void
LALZReadTArraySeries( LALStatus *status, COMPLEX16TimeArraySeries *series, FILE *stream );

void
LALI2ReadFSeries( LALStatus *status, INT2FrequencySeries *series, FILE *stream );
void
LALI4ReadFSeries( LALStatus *status, INT4FrequencySeries *series, FILE *stream );
void
LALI8ReadFSeries( LALStatus *status, INT8FrequencySeries *series, FILE *stream );
void
LALU2ReadFSeries( LALStatus *status, UINT2FrequencySeries *series, FILE *stream );
void
LALU4ReadFSeries( LALStatus *status, UINT4FrequencySeries *series, FILE *stream );
void
LALU8ReadFSeries( LALStatus *status, UINT8FrequencySeries *series, FILE *stream );
void
LALSReadFSeries( LALStatus *status, REAL4FrequencySeries *series, FILE *stream );
void
LALDReadFSeries( LALStatus *status, REAL8FrequencySeries *series, FILE *stream );
void
LALCReadFSeries( LALStatus *status, COMPLEX8FrequencySeries *series, FILE *stream );
void
LALZReadFSeries( LALStatus *status, COMPLEX16FrequencySeries *series, FILE *stream );




void
LALI2ReadGrid( LALStatus *status, INT2Grid **grid, FILE *stream );
void
LALI4ReadGrid( LALStatus *status, INT4Grid **grid, FILE *stream );
void
LALI8ReadGrid( LALStatus *status, INT8Grid **grid, FILE *stream );
void
LALU2ReadGrid( LALStatus *status, UINT2Grid **grid, FILE *stream );
void
LALU4ReadGrid( LALStatus *status, UINT4Grid **grid, FILE *stream );
void
LALU8ReadGrid( LALStatus *status, UINT8Grid **grid, FILE *stream );
void
LALSReadGrid( LALStatus *status, REAL4Grid **grid, FILE *stream );
void
LALDReadGrid( LALStatus *status, REAL8Grid **grid, FILE *stream );
void
LALCReadGrid( LALStatus *status, COMPLEX8Grid **grid, FILE *stream );
void
LALZReadGrid( LALStatus *status, COMPLEX16Grid **grid, FILE *stream );






#ifdef __cplusplus
}
#endif

#endif /* _STREAMINPUT_H */

/*
*  Copyright (C) 2007 Alexander Dietz, Drew Keppel, Duncan Brown, Eirini Messaritaki, Gareth Jones, George Birthisel, Jolien Creighton, Kipp Cannon, Lisa M. Goggin, Patrick Brady, Peter Shawhan, Saikat Ray-Majumder, Stephen Fairhurst, Xavier Siemens, Craig Robinson , Thomas Cokelaer
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

/*-----------------------------------------------------------------------
 *
 * File Name: LIGOMetadataUtils.h
 *
 * Author: Brown, D. A. and Fairhurst, S.
 *
 * Revision: $Id$
 *
 *-----------------------------------------------------------------------
 */

#if 0
<lalVerbatim file="LIGOMetadataUtilsHV">
Author: Brown, D. A. and Fairhurst, S.
$Id$
</lalVerbatim>
<lalLaTeX>
\section{Header \texttt{LIGOMetadataUtils.h}}
\label{s:LIGOMetadataItils.h}

Provides functions for manipulating the LAL structures that correspond
to the LIGO metadata database tables defined in \texttt{LIGOMetadataTables.h}.

\subsection*{Synopsis}
\begin{verbatim}
#include <lal/LIGOMetadataUtils.h>
\end{verbatim}

\noindent This header provides prototypes for routines that perform processing
on the LAL structures that correspond to the LIGO metadata database tables
defined in \texttt{LIGOMetadataTables.h}, such as sorting and eliminating
duplictaes. The functions specific to a particular metadata table (e.g.
\texttt{sngl\_inspiral}, \texttt{sngl\_burst}, etc.) are all prototyped in
this header.

\subsection*{Types}

\noindent None.

</lalLaTeX>
#endif

#ifndef _LIGOMETADATAUTILS_H
#define _LIGOMETADATAUTILS_H

#ifdef  __cplusplus
extern "C" {
#pragma }
#endif

#include <lal/LIGOMetadataTables.h>
#include <lal/LALDetectors.h>
#include <lal/Segments.h>

NRCSID( LIGOMETADATAUTILSH, "$Id$" );

#if 0
<lalLaTeX>
\subsection*{Error codes}
</lalLaTeX>
#endif
/* <lalErrTable> */
#define LIGOMETADATAUTILSH_ENULL 1
#define LIGOMETADATAUTILSH_ENNUL 2
#define LIGOMETADATAUTILSH_ETIME 3
#define LIGOMETADATAUTILSH_ECOOR 4
#define LIGOMETADATAUTILSH_ESGAP 5
#define LIGOMETADATAUTILSH_ESDUB 6
#define LIGOMETADATAUTILSH_ETEST 7
#define LIGOMETADATAUTILSH_EDET 8
#define LIGOMETADATAUTILSH_EDIST 9
#define LIGOMETADATAUTILSH_MSGENULL "Null pointer"
#define LIGOMETADATAUTILSH_MSGENNUL "Non-null pointer"
#define LIGOMETADATAUTILSH_MSGETIME "Invalid GPS Time"
#define LIGOMETADATAUTILSH_MSGECOOR "Invalid Coordinate System"
#define LIGOMETADATAUTILSH_MSGESGAP "Gap in Search Summary Input"
#define LIGOMETADATAUTILSH_MSGESDUB "Repeated data in Search Summary Input"
#define LIGOMETADATAUTILSH_MSGETEST "Unknown parameter test for sorting events"
#define LIGOMETADATAUTILSH_MSGEDET "Unknown detector"
#define LIGOMETADATAUTILSH_MSGEDIST "No horizon distance for consistency cut"

/* </lalErrTable> */

#if 0
<lalLaTeX>
\subsection*{Types}
</lalLaTeX>
\idx[Type]{LALPlaygroundDataMask}
\idx[Type]{SnglInspiralParameterTest}
\idx[Type]{SnglInspiralAccuracy}
\idx[Type]{SnglInspiralClusterChoice}





\subsubsection*{Type \texttt{LALPlaygroundDataMask}}
#endif
/* <lalVerbatim> */
typedef enum
{
  unspecified_data_type,
  playground_only,
  exclude_play,
  all_data
}
LALPlaygroundDataMask;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{LALPlaygroundDataMask} contains an enum type for describing the
subset of data to be used, \texttt{playground\_only}, \texttt{exclude\_play}
and \texttt{all\_data}.
\subsubsection*{Type \texttt{LALPlaygroundDataMask}}
</lalLaTeX>
#endif


/*
 *
 * general manipulation functions
 *
 */

ProcessTable *XLALCreateProcessTableRow(void);
void XLALDestroyProcessTableRow(ProcessTable *);
void XLALDestroyProcessTable(ProcessTable *);

ProcessParamsTable *XLALCreateProcessParamsTableRow(const ProcessTable *);
void XLALDestroyProcessParamsTableRow(ProcessParamsTable *);
void XLALDestroyProcessParamsTable(ProcessParamsTable *);

SearchSummaryTable *XLALCreateSearchSummaryTableRow(const ProcessTable *);
void XLALDestroySearchSummaryTableRow(SearchSummaryTable *);
void XLALDestroySearchSummaryTable(SearchSummaryTable *);

int
XLALCountProcessTable(
    ProcessTable *head
    );

int
XLALCountProcessParamsTable(
    ProcessParamsTable *head
    );

int
XLALCountMultiInspiralTable(
    MultiInspiralTable *head
    );

int
XLALIFONumber(
    const char *ifo
    );

void
XLALReturnIFO(
    char                *ifo,
    InterferometerNumber IFONumber
    );

void
XLALReturnDetector(
    LALDetector           *det,
    InterferometerNumber   IFONumber
    );


int
XLALPlaygroundInSearchSummary (
    SearchSummaryTable *ssTable,
    LIGOTimeGPS        *inPlayTime,
    LIGOTimeGPS        *outPlayTime
    );

void
LALPlaygroundInSearchSummary (
    LALStatus          *status,
    SearchSummaryTable *ssTable,
    LIGOTimeGPS        *inPlayTime,
    LIGOTimeGPS        *outPlayTime
    );


void
LALTimeCheckSearchSummary (
    LALStatus          *status,
    SearchSummaryTable *ssTable,
    LIGOTimeGPS        *startTime,
    LIGOTimeGPS        *endTime
    );

int
LALCompareSearchSummaryByInTime (
    const void *a,
    const void *b
    );

int
XLALCompareCoincInspiralByStat (
    const void *a,
    const void *b
    );

int
LALCompareSearchSummaryByOutTime (
    const void *a,
    const void *b
    );

int
XLALTimeSortSearchSummary(
    SearchSummaryTable  **summHead,
    int(*comparfunc)    (const void *, const void *)
    );

void
LALTimeSortSearchSummary (
    LALStatus            *status,
    SearchSummaryTable  **summHead,
    int(*comparfunc)    (const void *, const void *)
    );

void
LALIfoScanSearchSummary(
    LALStatus                  *status,
    SearchSummaryTable        **output,
    SearchSummaryTable         *input,
    CHAR                       *ifo
    );

void
LALDistanceScanSummValueTable (
    LALStatus            *status,
    SummValueTable       *summValueList,
    LIGOTimeGPS          gps,
    const CHAR           *ifo,
    REAL4                *distance
    );

void
LALCheckOutTimeFromSearchSummary (
    LALStatus            *status,
    SearchSummaryTable   *summList,
    CHAR                 *ifo,
    LIGOTimeGPS          *startTime,
    LIGOTimeGPS          *endTime
    );

SearchSummaryTable *
XLALIfoScanSearchSummary(
    SearchSummaryTable         *input,
    CHAR                       *ifos
    );

void
LALIfoScanSummValue(
    LALStatus                  *status,
    SummValueTable            **output,
    SummValueTable             *input,
    CHAR                       *ifo
    );

int
LALCompareSummValueByTime (
    const void *a,
    const void *b
    );

int
XLALTimeSortSummValue (
    SummValueTable      **summHead,
    int(*comparfunc)    (const void *, const void *)
    );

void
LALTimeSortSummValue (
    LALStatus            *status,
    SummValueTable      **summHead,
    int(*comparfunc)    (const void *, const void *)
    );

#if 0
<lalLaTeX>
\vfill{\footnotesize\input{LIGOMetadataUtilsHV}}

\newpage\input{LIGOMetadataUtilsC}
\newpage\input{SnglInspiralUtilsC}
\newpage\input{CoincInspiralUtilsC}
\newpage\input{SimInspiralUtilsC}
</lalLaTeX>
#endif

#ifdef  __cplusplus
#pragma {
}
#endif

#endif /* _LIGOMETADATAUTILS_H */


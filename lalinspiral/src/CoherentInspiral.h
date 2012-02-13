/*
*  Copyright (C) 2007 Sukanta Bose, Jolien Creighton, Sean Seader, Thomas Cokelaer
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
 * File Name: CoherentInspiral.h
 *
 * Author: Bose, S., Seader, S. E.
 *
 *-----------------------------------------------------------------------
 */

/**

\author Bose, S., Seader, S. E.
\file

\brief Provides core prototypes, structures and functions to filter
data from multiple interferometers coherently for binary inspiral chirps.

\section sec_ci_stat Coherent search statistic for binary neutron stars

The coherent statistic will be defined here.

\heading{Synopsis}
\code
#include <lal/CoherentInspiral.h>
\endcode

*/

#ifndef _COHERENTINSPIRALH_H
#define _COHERENTINSPIRALH_H

#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/LALDatatypes.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LALInspiral.h>
#include <lal/FindChirp.h>
#include <lal/LALInspiralBank.h>

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

/**\name Error Codes */ /*@{*/
#define COHERENTINSPIRALH_ENULL 1
#define COHERENTINSPIRALH_ENNUL 2
#define COHERENTINSPIRALH_EALOC 3
#define COHERENTINSPIRALH_ENUMZ 4
#define COHERENTINSPIRALH_ESEGZ 5
#define COHERENTINSPIRALH_ECHIZ 6
#define COHERENTINSPIRALH_EDTZO 7
#define COHERENTINSPIRALH_EFREE 8
#define COHERENTINSPIRALH_ERHOT 9
#define COHERENTINSPIRALH_ECHIT 10
#define COHERENTINSPIRALH_ESMSM 11
#define COHERENTINSPIRALH_EZDET 12
#define COHERENTINSPIRALH_MSGENULL "Null pointer"
#define COHERENTINSPIRALH_MSGENNUL "Non-null pointer"
#define COHERENTINSPIRALH_MSGEALOC "Memory allocation error"
#define COHERENTINSPIRALH_MSGENUMZ "Invalid number of points in segment"
#define COHERENTINSPIRALH_MSGESEGZ "Invalid number of segments"
#define COHERENTINSPIRALH_MSGECHIZ "Invalid number of chi squared bins"
#define COHERENTINSPIRALH_MSGEDTZO "deltaT is zero or negative"
#define COHERENTINSPIRALH_MSGEFREE "Error freeing memory"
#define COHERENTINSPIRALH_MSGERHOT "coherentSNR threshold is negative"
#define COHERENTINSPIRALH_MSGECHIT "Chisq threshold is negative"
#define COHERENTINSPIRALH_MSGESMSM "Size mismatch between vectors"
#define COHERENTINSPIRALH_MSGEZDET "Number of detectors should be greater than 1 and less than 5"
/*@}*/

/* --- parameter structure for the coherent inspiral filtering function ---- */
typedef struct
tagCoherentInspiralInitParams
{
  UINT4                         numDetectors;
  UINT4                         numSegments;
  UINT4                         numPoints;
  UINT4                         numBeamPoints;
  UINT4                         cohSNROut;
  UINT4                         cohH1H2SNROut;
  UINT4                         nullStatH1H2Out;
  UINT4                         nullStatOut;
  UINT4                         threeSiteCase;
}
CoherentInspiralInitParams;


typedef struct
tagDetectorVector
{
  UINT4                   numDetectors;
  LALDetector            *detector;
}
DetectorVector;


typedef struct
tagDetectorBeamArray
{
  UINT4                    numBeamPoints;
  REAL4TimeSeries         *thetaPhiVs;/* 4D array: theta,phi,v+,v- */
}
DetectorBeamArray;


typedef struct
tagCoherentInspiralBeamVector
{
  UINT4                   numDetectors;
  DetectorBeamArray      *detBeamArray;
}
CoherentInspiralBeamVector;


/** This structure provides the parameters used by the CoherentInspiralFilter() function.

<dl>
<dt><tt>UINT4 numPoints</tt></dt><dd> Number of time-points in the \f$c\f$ series
from each detector. This determines the number of time-points in the
\c cohSNRVec time-series.</dd>

<dt><tt>REAL4 cohSNRThresh</tt></dt><dd> The value to threshold the multi-detector
coherent signal to noise
ratio square, \f$\rho^2\f$, on. If the signal to noise exceeds this value, then a
candidate event is generated. Must be \f$\ge 0\f$ on entry.</dd>

<dt><tt>DetectorVector   detectors</tt></dt><dd> This structure is defined below. It specifies the detectors on which
the coherent search is being performed.</dd>

<dt><tt>REAL4Vector *cohSNRVec</tt></dt><dd> Pointer to a vector that is set to
\f$\rho^2(t_j)\f$ on exit. If NULL \f$\rho^2(t_j)\f$ is not stored.
</dd>
</dl>

*/
typedef struct
tagCoherentInspiralFilterParams
{
  INT4                          numTmplts;
  UINT4                         maximizeOverChirp;
  UINT4                         numDetectors;
  UINT4                         numSegments;
  INT4                          numPoints;
  UINT4                         numBeamPoints;
  UINT4                         threeSiteCase;
  REAL4                         fLow;
  REAL8                         deltaT;
  REAL4                         cohSNRThresh;
  REAL8Vector                  *sigmasqVec;
  REAL4Vector                  *chisqVec;
  REAL4                         templateNorm;
  INT4                          segmentLength; /* time points */
  UINT4                         cohSNROut;
  UINT4                         cohH1H2SNROut;
  UINT4                         nullStatH1H2Out;
  UINT4                         nullStatOut;
  UINT2Vector                  *detIDVec; /* Note: H1, H2 are from same site, but are different detectors */
  DetectorVector               *detectorVec; /*stores detectors' site info */
  REAL4TimeSeries              *cohSNRVec;
  REAL4TimeSeries              *cohH1H2SNRVec;
  REAL4TimeSeries              *nullStatH1H2Vec;
  REAL4TimeSeries              *nullStatVec;
  REAL4TimeSeries              *cohSNRVec3Sites;
  REAL4TimeSeries              *nullStatVec3Sites;
  REAL4                         chirpTime;
  double                        decStep;
  double                        raStep;
  UINT4                         estimParams;
  UINT4                         followup;
  UINT4                         exttrig;
}
CoherentInspiralFilterParams;

/* --- input to the CoherentInspiral filtering functions --------- */

/** This structure groups the \f$c = x+iy\f$ outputs of \f$M\f$ detectors
into an ordered set. The FindChirpFilter code, when separately run on the
data from multiple detectors, outputs a \c COMPLEX8TimeSeries, \f$c\f$, for
each detector. If a coherent search is to be performed on the data from
these \f$M\f$ detectors, one of the inputs required is the
\c CoherentInspiralCVector structure with a default vector
\c length of \f$M=6\f$ and with the vector index ordered as 0=H1, 1=L1,
2=V (Virgo), 3=G (GEO), 4=T (Tama), (just like the lalcached detector siteIDs)
and 5=H2. If a coherent search is to be performed on, say, the data from
H1, L1, Virgo, and GEO, then the \c length
member above will be set to 6 (by default), but the pointers to the fourth and
fifth \c COMPLEX8TimeSeries will be set to NULL; the remainder will
point to the \f$c\f$ outputs from the above 4 detectors, in that order.

<dl>
<dt><tt>UINT4  length</tt></dt><dd> Length of the vector; set to 6 (by default)
for the total number of operating (or nearly so) interferometers.</dd>

<dt><tt>COMPLEX8TimeSeries  *cData</tt></dt><dd> Pointer to the c outputs of
the 6 interferometers.</dd>
</dl>

*/
typedef struct
tagCoherentInspiralCVector
{
  UINT4                   numDetectors;
  COMPLEX8TimeSeries     *cData[4];
}
CoherentInspiralCVector;




/** This structure provides the essential information for
computing the coherent SNR from the \f$c\f$ outputs of multiple detectors.
In addition to this, the code requires the beam-pattern coefficients
for the different detectors. These coefficients are currently
computed by a Mathematica code and are read in as ascii files directly
by the coherent code. But there are plans for the future where a new member
will be added to this structure to store these coefficients.

<dl>
<dt><tt>CoherentInspiralCVector   *multiCData</tt></dt><dd> Pointer to the
 vector of COMPLEX8TimeSeries, namely, \c CoherentInspiralCVector.</dd>
</dl>
*/
typedef struct
tagCoherentInspiralFilterInput
{
  InspiralTemplate            *tmplt;
  CoherentInspiralCVector     *multiCData;
  CoherentInspiralBeamVector  *beamVec;
}
CoherentInspiralFilterInput;


typedef struct tagSkyGrid {
  REAL8 Alpha;
  REAL8 Delta;
  struct tagSkyGrid *next;
} SkyGrid;

/*
 *
 * function prototypes for memory management functions
 .*
 */

void
LALCoherentInspiralFilterInputInit (
    LALStatus                       *status,
    CoherentInspiralFilterInput    **input,
    CoherentInspiralInitParams      *params
    );

void
LALCoherentInspiralFilterInputFinalize (
    LALStatus                       *status,
    CoherentInspiralFilterInput    **input
    );

void
LALCoherentInspiralFilterParamsInit (
    LALStatus                       *status,
    CoherentInspiralFilterParams   **output,
    CoherentInspiralInitParams      *params
    );

void
LALCoherentInspiralFilterParamsFinalize (
    LALStatus                       *status,
    CoherentInspiralFilterParams   **output
    );

/*
 *
 * function prototypes for coherent inspiral filter function
 *
 */

void
LALCoherentInspiralEstimatePsiEpsilonCoaPhase (
    LALStatus                             *status,
    INT4                                   caseID[LAL_NUM_IFO],
    REAL8                                 *sigmasq,
    REAL4                                  theta,
    REAL4                                  phi,
    COMPLEX8                               cData[4],
    REAL4                                 *inclination,
    REAL4                                 *polarization,
    REAL4                                 *coaPhase
    );

void
LALCoherentInspiralEstimateDistance (
    LALStatus                             *status,
    REAL8                                 *sigmasq,
    REAL4                                  templateNorm,
    REAL8                                  deltaT,
    INT4                                   segmentLength,  /* time pts */
    REAL4                                  coherentSNR,
    REAL4                                 *distance
    );

void
XLALCoherentInspiralFilterSegment (
    LALStatus                             *status,
    MultiInspiralTable                    **eventList,
    CoherentInspiralFilterInput           *input,
    CoherentInspiralFilterParams          *params,
    const SkyGrid                         *skyGridPtr,
    REAL4                                 chisq[4],
    REAL4                                 chisq_dof[4],
    REAL4                                 eff_snr_denom_fac,
    REAL4                                 nullStatRegul
    );


#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _COHERENTINSPIRALH_H */

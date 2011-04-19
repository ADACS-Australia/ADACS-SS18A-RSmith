/*  
 *  Copyright (C) 2009-2010 Holger Pletsch.
 *
 *  Based on HierarchicalSearch.h by
 *  Copyright (C) 2005-2008 Badri Krishnan, Alicia Sintes, Bernd Machenschalk.
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
 * 
 */

#ifndef _HIERARCHSEARCHGCTH  /* Double-include protection. */
#define _HIERARCHSEARCHGCTH

/* standard includes */
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h> 

/* lal includes */
#include <lal/UserInput.h>
#include <lal/LALStdlib.h>
#include <lal/PulsarDataTypes.h>
#include <lal/SFTfileIO.h>
#include <lal/AVFactories.h>
#include <lal/RngMedBias.h>
#include <lal/LALComputeAM.h>
#include <lal/ComputeSky.h>
#include <lal/LALInitBarycenter.h>
#include <lal/Velocity.h>
#include <lal/LALDemod.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/Date.h>
#include <lal/LALHough.h> 
#include <lal/NormalizeSFTRngMed.h>
#include <lal/ComputeFstat.h>
#include <lal/Statistics.h>
#include <lal/GeneratePulsarSignal.h> 
#include <lal/LogPrintf.h>
#include <lal/DopplerScan.h>
#include <lal/UniversalDopplerMetric.h>

/* lalapps includes */
#include <lalapps.h>

/* more efficient toplist using heaps */
#include "GCTtoplist.h"
#include "ComputeFstat_RS.h"


/******************************************************
 *   Protection against C++ name mangling
 */

#ifdef  __cplusplus
extern "C" {
#endif


/******************************************************
 *  Assignment of Id string using NRCSID()
 */

NRCSID( HIERARCHICALSEARCHH, "$Id: HierarchicalSearchGC.h,v 1.9 2009/10/07 08:14:37 hpletsch Exp $" );

/******************************************************
 *  Error codes and messages.
 */
 
#define HIERARCHICALSEARCH_ENORM 0
#define HIERARCHICALSEARCH_ESUB  1
#define HIERARCHICALSEARCH_EARG  2
#define HIERARCHICALSEARCH_EBAD  3
#define HIERARCHICALSEARCH_EFILE 4
#define HIERARCHICALSEARCH_ENULL 5
#define HIERARCHICALSEARCH_EVAL  6
#define HIERARCHICALSEARCH_ENONULL 7
#define HIERARCHICALSEARCH_EDLOPEN 8
#define HIERARCHICALSEARCH_EWORKER 9
#define HIERARCHICALSEARCH_ECHECKPT 10
#define HIERARCHICALSEARCH_EMEM 11
#define HIERARCHICALSEARCH_ESFT 12
#define HIERARCHICALSEARCH_ECG 13
#define HIERARCHICALSEARCH_EXLAL 14

#define HIERARCHICALSEARCH_MSGENORM    "Normal exit"
#define HIERARCHICALSEARCH_MSGESUB     "Subroutine failed"
#define HIERARCHICALSEARCH_MSGEARG     "Error parsing arguments"
#define HIERARCHICALSEARCH_MSGEBAD     "Bad argument values"
#define HIERARCHICALSEARCH_MSGEFILE    "Could not create output file"
#define HIERARCHICALSEARCH_MSGENULL    "Null pointer"
#define HIERARCHICALSEARCH_MSGEVAL     "Invalid value"
#define HIERARCHICALSEARCH_MSGENONULL  "Pointer not null"
#define HIERARCHICALSEARCH_MSGECHECKPT "Could not resume from checkpoint"
#define HIERARCHICALSEARCH_MSGEMEM     "Out of memory"
#define HIERARCHICALSEARCH_MSGESFT     "SFT validity check failed"
#define HIERARCHICALSEARCH_MSGEXLAL    "XLAL function call failed"


/* ******************************************************************
 *  Structure, enum, union, etc., typdefs.
 */

  /** type describing one coherent segment of data: ( start-time + duration ) */
  typedef struct {
    UINT4 startTime;	/**< gps start-time of segment, in seconds */
    UINT4 duration;	/**< duration of segment in seconds */
  } Segment_t;

  /** a standard vector of data-segments */
  typedef struct {
    UINT4 length;	/**< number of segments */
    Segment_t *data;	/**< array of segments */
  } SegmentVector_t;

  /** sequence of MultiSFT vectors -- for each stack */
  typedef struct tagMultiSFTVectorSequence {
    UINT4 length;     /**< number of stacks */
    MultiSFTVector **data; /**< the SFT vectors */
  } MultiSFTVectorSequence;

  /** sequence of Multi-noise weights vectors -- for each stack */
  typedef struct tagMultiNoiseWeightsSequence {
    UINT4 length;     /**< number of stacks */
    MultiNoiseWeights **data; /**< the noise weights */
  } MultiNoiseWeightsSequence;

  /** sequence of Multi-detector state vectors -- for each stack */
  typedef struct tagMultiDetectorStateSeriesSequence {
    UINT4 length;     /**< number of stacks */
    MultiDetectorStateSeries **data; /**< the detector state series */
  } MultiDetectorStateSeriesSequence;

  /* sequence of SFT catalogs -- for each stack */
  typedef struct tagSFTCatalogSequence {
    UINT4 length;     /**< the number of stacks */
    SFTCatalog *data; /**< the catalogs */
  } SFTCatalogSequence;


  /** parameters for the semicoherent stage */
  typedef struct tagSemiCoherentParams {
    LIGOTimeGPSVector *tsMid;  /**< timestamps of mid points of segments */
    LIGOTimeGPS refTime;       /**< reference time for f, fdot definition */
    REAL8VectorSequence *pos;  /**< Earth orbital position for each segment */
    REAL8VectorSequence *vel;  /**< Earth orbital velocity for each segment */
    REAL8VectorSequence *acc;  /**< Earth orbital acceleration for each segment (new) */ 
    CHAR *outBaseName;         /**< file for writing output -- if chosen */
    REAL8  threshold;          /**< Threshold for candidate selection */
    UINT4 extraBinsFstat;      /**< Extra bins required for Fstat calculation */
  } SemiCoherentParams;


  /* ------------------------------------------------------------------------- */


  /** structure for storing fine-grid points */
#ifdef GC_SSE2_OPT
#define FINEGRID_NC_T UCHAR
#else
#define FINEGRID_NC_T UINT4
#endif
  typedef struct tagFineGrid {
    REAL8 freqmin_fg;       /**< fine-grid start in frequency */
    REAL8 f1dotmin_fg;      /**< fine-grid start in 1st spindown */
    REAL8 dfreq_fg;         /**< fine-grid spacing in frequency */
    REAL8 df1dot_fg;        /**< fine-grid spacing in 1st spindown */
    REAL8 alpha;            /**< right ascension */
    REAL8 delta;            /**< declination */
    LIGOTimeGPS refTime;    /**< reference time for candidates */
    UINT4 length;           /**< maximum allowed length of vectors */
    UINT4 freqlength;       /**< number of fine-grid points in frequency */
    UINT4 f1dotlength;      /**< number of fine-grid points in 1st spindown */
    REAL4 * sumTwoF;        /**< sum of 2F-values */
    FINEGRID_NC_T * nc;     /**< number count */
  } FineGrid;


  /** one coarse-grid point */
  typedef struct tagCoarseGridPoint {
    UINT4 Uindex;      /**< U index */
    REAL4 TwoF;       /**< 2F-value */
  } CoarseGridPoint;  


  /** structure for storing coarse-grid points */
  typedef struct tagCoarseGrid {
    UINT4 length;             /**< maximum allowed length of vectors */
    UINT4 * Uindex;      /**< U index */
    REAL4 * TwoF;       /**< 2F-value */
  } CoarseGrid;

 /* ------------------------------------------------------------------------- */

 /* function prototypes */

  void SetUpStacks(LALStatus *status, 
		 SFTCatalogSequence  *out,  
		 REAL8 tStack,
		 SFTCatalog  *in,
		 UINT4 nStacks);

  void GetChkPointIndex( LALStatus *status,
			 INT4 *loopindex, 
			 const CHAR *fnameChkPoint);

#ifdef  __cplusplus
}                /* Close C++ protection */
#endif

#endif  /* Double-include protection. */

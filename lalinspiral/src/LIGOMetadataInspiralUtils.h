/*
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

#ifndef _LIGOMETADATAINSPIRALUTILS_H
#define _LIGOMETADATAINSPIRALUTILS_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <lal/LIGOMetadataUtils.h>
#include <lal/LALInspiral.h>
#include <lal/GeneratePPNInspiral.h>

/*
 *
 * inspiral specific structures
 *
 */

#if 0
<lalLaTeX>
\subsubsection*{Type \texttt{SnglInspiralParameterTest}}
</lalLaTeX>
#endif
/* <lalVerbatim> */
typedef enum
{
  unspecified_test,
  no_test,
  m1_and_m2,
  psi0_and_psi3,
  mchirp_and_eta,
  tau0_and_tau3,
  ellipsoid
}
SnglInspiralParameterTest;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{SnglInspiralParameterTest} contains an enum type for each of the
tests of mass parameters which are used.
\subsubsection*{Type \texttt{SnglInspiralAccuracy}}
</lalLaTeX>
#endif
/* <lalVerbatim> */
typedef struct
tagSnglInspiralAccuracy
{
  INT4        match;
  REAL4       epsilon;
  REAL4       kappa;
  INT8        dt;
  REAL4       dm;
  REAL4       deta;
  REAL4       dmchirp;
  REAL4	      dmchirpHi;
  REAL4       highMass;
  REAL4       dpsi0;
  REAL4       dpsi3;
  REAL4       dtau0;
  REAL4       dtau3;
  SnglInspiralParameterTest test;
  INT4        exttrig;
}
SnglInspiralAccuracy;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{SnglInspiralAccuracy} structure contains parameters used for
testing coincidence between two or more single inspiral tables.  These include
a timing accuracy \texttt{dt}, five mass accuracies \texttt{dm} (used for
testing \texttt{mass1} and \texttt{mass2}), \texttt{deta}, \texttt{dmchirp},
\texttt{dpsi0} and \texttt{dpsi3}.  It also includes the parameters
\texttt{kappa} and \texttt{epsilon} which are used for testing consistency of
effective distance.
\subsubsection*{Type \texttt{SnglInspiralClusterChoice}}
</lalLaTeX>
#endif
/* <lalVerbatim> */
typedef struct
tagInspiralAccuracyList
{
  INT4                      match;
  SnglInspiralParameterTest test;
  SnglInspiralAccuracy      ifoAccuracy[LAL_NUM_IFO];
  INT8                      lightTravelTime[LAL_NUM_IFO][LAL_NUM_IFO];
  REAL4                     iotaCutH1H2;
  REAL4                     iotaCutH1L1;
  REAL8                     eMatch;
  INT4                      exttrig;
}
InspiralAccuracyList;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{InspiralAccuracyList} structure contains parameter accuracies for
each of the six global interferometers.  These are stored in the
\texttt{SnglInspiralAccuracy} structure.  The accuracies stored should be the
accuracy with which each instrument can determine the given parameter.  It also
contains a \texttt{match} which is set to 1 to signify that coincidence
criteria are satisfied and 0 otherwise.  Finally, the
\texttt{SnglInspiralParameterTest} must be specified.
\subsubsection*{Type \texttt{SnglInspiralClusterChoice}}
</lalLaTeX>
#endif

/* <lalVerbatim> */
typedef struct
tagCoincInspiralStatParams
{
  REAL4    param_a[LAL_NUM_IFO];
  REAL4    param_b[LAL_NUM_IFO];
  REAL4    eff_snr_denom_fac;
}
CoincInspiralStatParams;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{CoincInspiralStatParams} structure contains the bitten L parameter for
each of the six global interferometers.  These are stored in the
\texttt{param\_a} and \texttt{param\_b} structure.
</lalLaTeX>
#endif

/* <lalVerbatim> */
typedef enum
{
  none,
  snr_and_chisq,
  snrsq_over_chisq,
  snr
}
SnglInspiralClusterChoice;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{SnglInspiralClusterChoice} provides three choices for clustering
a single inspiral table.  The\texttt{snr} clustering returns the trigger
with the greatest signal to noise ratio; \texttt{snr\_and\_chisq} replaces
the existing trigger if the new trigger has \textit{both} a greater snr and
a smaller chi squared value; \texttt{snrsq\_over\_chisq} selects the trigger
with the largest value of snr squared divided by the chi squared.
</lalLaTeX>
#endif
/* <lalVerbatim> */
typedef enum
{
  no_stat,
  snrsq,
  effective_snrsq,
  s3_snr_chi_stat,
  bitten_l,
  bitten_lsq,
  ifar
}
CoincInspiralStatistic;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{CoincInspiralStatistic} provides two choices for clustering
a single inspiral table.  The\texttt{snrsq} clustering returns the trigger
with the greatest summed snr$^{2}$ from all instruments.  The
\texttt{snr\_chi\_stat} replaces selects the trigger
with the largest value of the snr and chisq statistic and the \texttt{bitten\_l}
returns the minimum among the summed snr$^{2}$ from all instruments and the
$a\times snr_i - b$ in each detector. The parameters  $a$ and $b$ must be
provided by the user.
</lalLaTeX>
#endif
/* <lalVerbatim> */
typedef enum
{
  no_statistic,
  nullstat,
  cohsnr,
  effCohSnr,
  snrByNullstat,
  autoCorrCohSqByNullstat,
  crossCorrCohSqByNullstat,
  autoCorrNullSqByNullstat,
  crossCorrNullSqByNullstat
}
MultiInspiralClusterChoice;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{MultiInspiralClusterChoice} provides choices for clustering
a multi inspiral table.  The \texttt{cohsnr} clustering returns the trigger
with the greatest coherent signal to noise ratio; the \texttt{nullstat}
clustering returns the trigger with the smallest null-statistic value.
</lalLaTeX>
#endif
/* <lalVerbatim> */
typedef struct
tagSnglInspiralBCVCalphafCut
{
  REAL4       h1_lo;
  REAL4       h1_hi;
  REAL4       h2_lo;
  REAL4       h2_hi;
  REAL4       l1_lo;
  REAL4       l1_hi;
  REAL4       psi0cut;
}
SnglInspiralBCVCalphafCut;
/*</lalVerbatim> */
#if 0
<lalLaTeX>
The \texttt{SnglInspiralBCVCalphafCut} provides entries for cutting single IFO triggers generated with the BCVC code. For each LSC IFO there is a field \texttt{lo} and \texttt{hi} which corresponds to the area allowing triggers.
</lalLaTeX>
#endif


/*
 *
 * inspiral specific functions
 *
 */

/* sngl inspiral */
void
LALFreeSnglInspiral (
    LALStatus          *status,
    SnglInspiralTable **eventHead
    );

int
XLALFreeSnglInspiral (
    SnglInspiralTable **eventHead
    );

void
LALSortSnglInspiral (
    LALStatus          *status,
    SnglInspiralTable **eventHead,
    int(*comparfunc)    (const void *, const void *)
    );

SnglInspiralTable *
XLALSortSnglInspiral (
    SnglInspiralTable  *eventHead,
    int(*comparfunc)    (const void *, const void *)
    );

int
LALCompareSnglInspiralByMass (
    const void *a,
    const void *b
    );

int
LALCompareSnglInspiralByPsi (
    const void *a,
    const void *b
    );

int
LALCompareSnglInspiralByTime (
    const void *a,
    const void *b
    );

int
LALCompareSnglInspiralByID (
     const void *a,
     const void *b
     );

void
LALCompareSnglInspiral (
    LALStatus                *status,
    SnglInspiralTable        *aPtr,
    SnglInspiralTable        *bPtr,
    SnglInspiralAccuracy     *params
    );

int
XLALCompareInspirals (
    SnglInspiralTable        *aPtr,
    SnglInspiralTable        *bPtr,
    InspiralAccuracyList     *params
    );

void
LALCompareInspirals (
    LALStatus                *status,
    SnglInspiralTable        *aPtr,
    SnglInspiralTable        *bPtr,
    InspiralAccuracyList     *params
    );

void
LALClusterSnglInspiralTable (
    LALStatus                  *status,
    SnglInspiralTable         **inspiralEvent,
    INT8                        dtimeNS,
    SnglInspiralClusterChoice   clusterchoice
    );

REAL4
XLALSnglInspiralStat(
    SnglInspiralTable         *snglInspiral,
    SnglInspiralClusterChoice  snglStat
    );

int
XLALClusterSnglInspiralTable (
    SnglInspiralTable         **inspiralList,
    INT8                        dtimeNS,
    SnglInspiralClusterChoice   clusterchoice
    );

SnglInspiralTable *
XLALTimeCutSingleInspiral(
    SnglInspiralTable          *eventList,
    LIGOTimeGPS                *startTime,
    LIGOTimeGPS                *endTime
    );

void
LALTimeCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    LIGOTimeGPS                *startTime,
    LIGOTimeGPS                *endTime
    );


void
LALSNRCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    REAL4                       snrCut
    );

SnglInspiralTable *
XLALSNRCutSingleInspiral(
    SnglInspiralTable          *eventHead,
    REAL4                       snrCut
    );

SnglInspiralTable *
XLALRsqCutSingleInspiral (
    SnglInspiralTable          *eventHead,
    REAL4                       rsqVetoTimeThresh,
    REAL4                       rsqMaxSnr,
    REAL4                       rsqAboveSnrCoeff,
    REAL4                       rsqAboveSnrPow
    );

SnglInspiralTable *
XLALVetoSingleInspiral (
    SnglInspiralTable          *eventHead,
    LALSegList                 *vetoSegs,
    const CHAR                 *ifo
    );

SnglInspiralTable *
XLALalphaFTemp (
    SnglInspiralTable          *eventHead
    );

void
LALBCVCVetoSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    SnglInspiralBCVCalphafCut   alphafParams
    );

void
LALalphaFCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    REAL4                       alphaFhi,
    REAL4                       alphaFlo
    );

void
LALIfoCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    CHAR                       *ifo
    );

SnglInspiralTable *
XLALIfoCutSingleInspiral(
    SnglInspiralTable         **eventHead,
    char                       *ifo
    );

void
LALIfoCountSingleInspiral(
    LALStatus                  *status,
    UINT4                      *numTrigs,
    SnglInspiralTable          *input,
    InterferometerNumber        ifoNumber
    );

int
XLALTimeSlideSegList(
    LALSegList                 *seglist,
    const LIGOTimeGPS          *ringStartTime,
    const LIGOTimeGPS          *ringEndTime,
    const LIGOTimeGPS          *slideTimes
    );

void
XLALTimeSlideSingleInspiral(
    SnglInspiralTable          *input,
    const LIGOTimeGPS          *ringStartTime,
    const LIGOTimeGPS          *ringEndTime,
    const LIGOTimeGPS           slideTimes[]
    );

SnglInspiralTable *
XLALPlayTestSingleInspiral(
    SnglInspiralTable          *eventHead,
    LALPlaygroundDataMask      *dataType
    );

void
LALPlayTestSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    LALPlaygroundDataMask      *dataType
    );


void
LALCreateTrigBank(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    SnglInspiralParameterTest  *test
    );

void
LALIncaCoincidenceTest(
    LALStatus                  *status,
    SnglInspiralTable         **ifoAOutput,
    SnglInspiralTable         **ifoBOutput,
    SnglInspiralTable          *ifoAInput,
    SnglInspiralTable          *ifoBInput,
    SnglInspiralAccuracy       *errorParams
    );

void
LALTamaCoincidenceTest(
    LALStatus                  *status,
    SnglInspiralTable         **ifoAOutput,
    SnglInspiralTable         **ifoBOutput,
    SnglInspiralTable          *ifoAInput,
    SnglInspiralTable          *ifoBInput,
    SnglInspiralAccuracy       *errorParams,
    SnglInspiralClusterChoice   clusterchoice
    );

int
XLALMaxSnglInspiralOverIntervals(
    SnglInspiralTable         **eventHead,
    INT4                       deltaT
    );

INT4
XLALCountSnglInspiral(
    SnglInspiralTable *head
    );

INT4
XLALCountCoincInspiral(
    CoincInspiralTable *head
    );

/* coinc inspiral */
void
LALCreateTwoIFOCoincList(
    LALStatus                  *status,
    CoincInspiralTable        **coincOutput,
    SnglInspiralTable          *snglInput,
    InspiralAccuracyList       *accuracyParams
    );

void
LALCreateNIFOCoincList(
    LALStatus                  *status,
    CoincInspiralTable        **coincHead,
    InspiralAccuracyList       *accuracyParams,
    INT4                        N
    );

void
LALRemoveRepeatedCoincs(
    LALStatus                  *status,
    CoincInspiralTable        **coincHead
    );

void
LALFreeCoincInspiral(
    LALStatus                  *status,
    CoincInspiralTable        **coincPtr
    );

int
XLALFreeCoincInspiral(
    CoincInspiralTable        **coincPtr
    );


void
LALAddSnglInspiralToCoinc(
    LALStatus                  *status,
    CoincInspiralTable        **coincPtr,
    SnglInspiralTable          *snglInspiral
    );

CoincInspiralTable *
XLALAddSnglInspiralToCoinc(
    CoincInspiralTable         *coincInspiral,
    SnglInspiralTable          *snglInspiral
    );

void
XLALSnglInspiralCoincTest(
    CoincInspiralTable         *coincInspiral,
    SnglInspiralTable          *snglInspiral,
    InspiralAccuracyList       *accuracyParams
    );

void
LALSnglInspiralCoincTest(
    LALStatus                  *status,
    CoincInspiralTable         *coincInspiral,
    SnglInspiralTable          *snglInspiral,
    InspiralAccuracyList       *accuracyParams
    );

void
XLALInspiralPsi0Psi3CutBCVC(
    CoincInspiralTable        **coincInspiral
    );

void
XLALInspiralIotaCutBCVC(
    CoincInspiralTable        **coincInspiral,
    InspiralAccuracyList       *accuracyParams
    );

void
LALInspiralDistanceCutCleaning(
    LALStatus                  *status,
    CoincInspiralTable        **coincInspiral,
    InspiralAccuracyList       *accuracyParams,
    REAL4 			snrThreshold,
    SummValueTable             *summValueList,
    LALSegList                 *vetoSegsH1,
    LALSegList                 *vetoSegsH2
    );

void
XLALInspiralDistanceCutBCVC(
    CoincInspiralTable        **coincInspiral,
    InspiralAccuracyList       *accuracyParams
    );

void
XLALInspiralDistanceCut(
    CoincInspiralTable        **coincInspiral,
    InspiralAccuracyList       *accuracyParams
    );

void
XLALInspiralSNRCutBCV2(
    CoincInspiralTable        **coincInspiral
    );

SnglInspiralTable *
XLALExtractSnglInspiralFromCoinc(
    CoincInspiralTable         *coincInspiral,
    LIGOTimeGPS                *gpsStartTime,
    INT4                        slideNum
    );

void
LALExtractSnglInspiralFromCoinc(
    LALStatus                  *status,
    SnglInspiralTable         **snglPtr,
    CoincInspiralTable         *coincInspiral,
    LIGOTimeGPS                *gpsStartTime,
    INT4                        slideNum
    );

int
XLALCreateCoincSlideTable(
    CoincInspiralSlideTable   **slideTableHead,
    INT4                        numSlides
    );

REAL4
XLALSetupCoincSlideTable(
    CoincInspiralSlideTable    *slideTableHead,
    CoincInspiralTable         *coincSlideHead,
    char                       *timeAnalyzedFileName,
    REAL4                       timeModifier,
    INT4                        numSlides
    );

int
XLALRecreateCoincFromSngls(
    CoincInspiralTable        **coincPtr,
    SnglInspiralTable         **snglInspiral
    );

void
LALCoincCutSnglInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **snglPtr
    );

int
XLALGenerateCoherentBank(
    SnglInspiralTable         **coherentBank,
    CoincInspiralTable         *coincInput,
    CHAR                       *ifos
    );

INT8
XLALCoincInspiralTimeNS (
    const CoincInspiralTable         *coincInspiral
    );

REAL4
XLALCoincInspiralStat(
    const CoincInspiralTable   *coincInspiral,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams    *bittenLParams
    );

int XLALComputeAndStoreEffectiveSNR(   	CoincInspiralTable *head,
					CoincInspiralStatistic *stat,
					CoincInspiralStatParams *par
    );

int
XLALClusterCoincInspiralTable (
    CoincInspiralTable        **coincList,
    INT8                        dtimeNS,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams *bittenLParams
    );

int
XLALCompareCoincInspiralByTime (
    const void *a,
    const void *b
    );

int
XLALCompareCoincInspiralByEffectiveSnr (
    const void *a,
    const void *b
    );

CoincInspiralTable *
XLALSortCoincInspiral (
    CoincInspiralTable  *eventHead,
    int(*comparfunc)    (const void *, const void *)
    );

CoincInspiralTable *
XLALSortCoincInspiralByStat (
    CoincInspiralTable  *eventHead,
    int(*comparfunc)    (const void *, const void *),
    CoincInspiralStatParams *statParams,
    CoincInspiralStatistic *stat
    );


int
XLALCoincInspiralIfos (
    CoincInspiralTable  *coincInspiral,
    char                *ifos
    );

int
XLALCoincInspiralIfosCut(
    CoincInspiralTable **coincHead,
    char                *ifos
    );

int
XLALCoincInspiralIfosDiscard(
    CoincInspiralTable **coincHead,
    char                *ifos
    );

UINT8
XLALCoincInspiralIdNumber (
    CoincInspiralTable  *coincInspiral
    );

CoincInspiralTable *
XLALCoincInspiralSlideCut(
    CoincInspiralTable **coincHead,
    int                  slideNum
    );

CoincInspiralTable *
XLALStatCutCoincInspiral (
    CoincInspiralTable         *eventHead,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams    *bittenLParams,
    REAL4                       statCut
    );

int
XLALCalcExpFitNLoudestBackground (
    CoincInspiralTable         *coincSlideHead,
    int                         fitNum,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams    *bittenLParams,
    REAL4                      *fitStat,
    REAL4                      *fitA,
    REAL4                      *fitB
    );

REAL4
XLALRateCalcCoincInspiral (
    CoincInspiralTable         *eventZeroHead,
    CoincInspiralTable         *eventSlideHead,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams    *bittenLParams,
    REAL4                       timeAnalyzed,
    REAL4                       fitStat,
    REAL4                       fitA,
    REAL4                       fitB
    );

REAL4
XLALRateErrorCalcCoincInspiral (
    CoincInspiralTable         *eventZeroHead,
    CoincInspiralSlideTable    *eventSlideHead,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams    *bittenLParams,
    int                         numSlides,
    REAL4                       timeAnalyzed,
    REAL4                       fitStat,
    REAL4                       fitA,
    REAL4                       fitB
    );

CoincInspiralTable *
XLALRateStatCutCoincInspiral (
    CoincInspiralTable         *eventZeroHead,
    CoincInspiralStatistic      coincStat,
    CoincInspiralStatParams    *bittenLParams,
    REAL4                       statCut,
    REAL4                       rateCut
    );

SnglInspiralTable *
XLALCompleteCoincInspiral (
    CoincInspiralTable         *eventHead,
    int                         ifoList[LAL_NUM_IFO]
    );

CoincInspiralTable *
XLALPlayTestCoincInspiral(
    CoincInspiralTable         *eventHead,
    LALPlaygroundDataMask      *dataType
    );

CoincInspiralTable *
XLALMeanMassCut(
    CoincInspiralTable         *eventHead,
    char                       *massCut,
    REAL4                       massRangeLow,
    REAL4                       massRangeHigh,
    REAL4                       mass2RangeLow,
    REAL4                       mass2RangeHigh
    );

SnglInspiralTable *
XLALMassCut(
    SnglInspiralTable          *eventHead,
    char                       *massCut,
    REAL4                       massRangeLow,
    REAL4                       massRangeHigh,
    REAL4                       mass2RangeLow,
    REAL4                       mass2RangeHigh
    );

/* sim inspiral */

void
LALGalacticInspiralParamsToSimInspiralTable(
    LALStatus                  *status,
    SimInspiralTable           *output,
    GalacticInspiralParamStruc *input,
    RandomParams               *params
    );

void
LALInspiralSiteTimeAndDist(
    LALStatus         *status,
    SimInspiralTable  *output,
    LALDetector       *detector,
    LIGOTimeGPS       *endTime,
    REAL4             *effDist,
    SkyPosition       *skyPos
    );

void
LALPopulateSimInspiralSiteInfo(
    LALStatus                  *status,
    SimInspiralTable           *output
    );

void
XLALSortSimInspiral(
    SimInspiralTable **head,
    int (*comparefunc)(const SimInspiralTable * const *,
      const SimInspiralTable * const *)
    );

int
XLALCompareSimInspiralByGeocentEndTime(
	const SimInspiralTable * const *a,
	const SimInspiralTable * const *b
    );

int
XLALFreeSimInspiral (
    SimInspiralTable **eventHead
    );

void
XLALPlayTestSimInspiral(
    SimInspiralTable          **eventHead,
    LALPlaygroundDataMask      *dataType
    );

int
XLALSimInspiralInSearchedData(
    SimInspiralTable         **eventHead,
    SearchSummaryTable       **summList
    );

int
XLALSimInspiralChirpMassCut(
    SimInspiralTable   **eventHead,
    REAL4                minChirpMass,
    REAL4                maxChirpMass
    );

int
XLALSimInspiralCompMassCut(
    SimInspiralTable   **eventHead,
    REAL4                minCompMass,
    REAL4                maxCompMass,
    REAL4                minCompMass2,
    REAL4                maxCompMass2
    );

int
XLALSimInspiralTotalMassCut(
    SimInspiralTable   **eventHead,
    REAL4                minTotalMass,
    REAL4                maxTotalMass
    );

INT8
XLALReturnSimInspiralEndTime (
    SimInspiralTable *event,
    CHAR             *ifo
    );

int
XLALSnglSimInspiralTest (
    SimInspiralTable  **simHead,
    SnglInspiralTable **eventHead,
    SimInspiralTable  **missedSimHead,
    SnglInspiralTable **missedSnglHead,
    INT8                injectWindowNS
    );

int
XLALCoincSimInspiralTest (
    SimInspiralTable   **simHead,
    CoincInspiralTable **coincHead,
    SimInspiralTable   **missedSimHead,
    CoincInspiralTable **missedCoincHead
   );

/* multi inspiral */

void
LALFreeMultiInspiral (
    LALStatus          *status,
    MultiInspiralTable **eventHead
    );

int
XLALFreeMultiInspiral (
    MultiInspiralTable **eventHead
    );

MultiInspiralTable *
XLALSortMultiInspiral (
    MultiInspiralTable *eventHead,
    int(*comparfunc)   (const void *, const void *)
    );

REAL4
XLALMultiInspiralStat(
    MultiInspiralTable         *multiInspiral,
    MultiInspiralClusterChoice  multiStat
    );

int
XLALClusterMultiInspiralTable (
    MultiInspiralTable         **inspiralList,
    INT8                         dtimeNS,
    MultiInspiralClusterChoice   clusterchoice
    );

MultiInspiralTable *
XLALTimeCutMultiInspiral(
    MultiInspiralTable          *eventHead,
    LIGOTimeGPS                *startTime,
    LIGOTimeGPS                *endTime
    );

MultiInspiralTable *
XLALSNRCutMultiInspiral (
    MultiInspiralTable          *eventHead,
    REAL4                       snrCut
    );

MultiInspiralTable *
XLALPlayTestMultiInspiral(
    MultiInspiralTable          *eventHead,
    LALPlaygroundDataMask      *dataType
    );

INT4 XLALCountMultiInspiral( MultiInspiralTable *head );

int
LALCompareMultiInspiralByTime (
    const void *a,
    const void *b
    );

int
XLALMultiSimInspiralTest (
    SimInspiralTable  **simHead,
    MultiInspiralTable **eventHead,
    SimInspiralTable  **missedSimHead,
    MultiInspiralTable **missedMultiHead,
    INT8                injectWindowNS
    );

int
XLALMultiInspiralIfos (
    MultiInspiralTable  *multiInspiral,
    char                *ifos
    );

int
XLALMultiInspiralIfosCut(
    MultiInspiralTable **multiHead,
    char                *ifos
    );

/* inspiral param accuracy */

void
XLALPopulateAccuracyParams(
       InspiralAccuracyList  *accuracyParams
);

void
XLALPopulateAccuracyParamsExt(
       InspiralAccuracyList  *accuracyParams,
       const LIGOTimeGPS     *gpstime,
       const REAL8            ra_deg,
       const REAL8            dec_deg
);

#ifdef  __cplusplus
}
#endif

#endif /* _LIGOMETADATAINSPIRALUTILS_H */

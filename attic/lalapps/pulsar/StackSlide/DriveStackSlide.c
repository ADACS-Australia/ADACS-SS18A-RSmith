/*
*  Copyright (C) 2007 Gregory Mendell, Virginia Re
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

/*********************************************************************************/
/*                                                                               */
/* File: DriveStackSlide.c                                                       */
/* Purpose: Contains driver functions for StackSlide search                      */
/* Author: Mendell, G. and Landry, M.                                            */
/* Started: 03 December 2003                                                     */
/*                                                                               */
/*********************************************************************************/

/* REVISIONS: */
/* 01/05/04 gam; format ascii file output of frequency and SUMs */
/* 01/06/04 gam; remove extraneous code that is now in StackSlide.c */
/* 01/14/04 gam; Change threshold1 and threshold2 from type REAL8 to type REAL4 */  
/* 01/14/04 gam; Add code that searches SUMs for peaks */
/* 01/20/04 gam; Change findStackSlidePeaks to LALFindStackSlidePeaks; put params into struct */
/* 01/20/04 gam; Ignore peaks within one maxWidth of f0SUM or f0SUM + bandSUM */
/* 01/20/04 gam; Correct code that calculates peak width */
/* 01/20/04 gam; Change unusedFlag2 to threshold3. */
/*               When thresholdFlag == 1 then threshold1 and threshold2 = thresholds on power and threshold3 = maxWidth. */
/* 01/21/04 gam; Call LALFindStackSlidePeaks only if thresholdFlag > 0 */
/* 01/21/04 gam; if params->outputFlag == 1 then output just SUM power; if params->outputFlag == 2 output frequency and power. */
/* 01/27/04 gam; Change unusedFlag1, unusedParam1 and unusedParam2 to threshold4, threshold5, and threshold6. */
/*               When thresholdFlag == 1 then threshold4 = fraction minimum power in a cluster must be below subpeak to count as new peak */
/*               threshold5 = false alarm rate associated with threshold1; threshold6 could used for a time based cut. */
/* 01/27/04 gam; Fix bugs in LALFindStackSlidePeaks; make peak finding algorithm more sophisticated. */
/* 01/28/04 gam; Fix bugs with multiple outputs; output a separate file for each SUM */
/* 01/28/04 mrl; Removed patchnumber as was previously removed from StackSlideParams */
/* 01/31/04 gam; If (params->parameterSpaceFlag == 0) then set up sky with deltaRA(DEC) = deltaRA(0)/cos(DEC), where deltaRA(0) = params->deltaRA */
/* 02/02/04 gam; Fix bugs with handling case numSpinDown = 0 and numFreqDerivTotal = 0. */
/* 02/02/04 gam; Make code clear that only currently supporting the case params->parameterSpaceFlag == 0. */
/* 02/04/04 gam; Add code to store peaks in the SnglStackSlidePeriodicTable struct and write these out in xml format. */
/* 02/09/04 gam; Clean up SnglStackSlidePeriodicTable and xml storage to remove unused or repetative columns */
/* 02/09/04 gam; Add xmlStream to params and output other tables into xml file */
/* 02/11/04 gam; In process table, replace UNKNOWN with BLANK = " ". Comment out cvs REVISION, DATE, etc..; just need CVS_ID_STRING and CVS_SOURCE. */
/* 02/11/04 gam; Comment out ComputePowerStats, SUMStats, and FreqSeriesPowerStats */
/* 02/11/04 gam; Remove obsolete code associate with INCLUDE_BUILDSUMFRAME_CODE (BuildSUMFrameOutput) and INCLUDE_BUILDOUTPUT_CODE (BuildOutput). */
/* 02/11/04 gam; LALInitBarycenter lal support package rather than internal version */
/* 02/11/04 gam; Use power rather than amplitudes when finding peaks */
/* 02/11/04 gam; Change code to process 1 SUM at a time to allow processing of millions of SUMs per job. */
/* 02/12/04 gam; Add num_sums = numSUMsTotal, freq_index =f0SUM*params->tEffSUM, and num_bins = nBinsPerSUM to StackSlideSFTsSearchSummaryTable */
/*               Add freq_index to SnglStackSlidePeriodicTable.  Each of these are integers that can be stored exactly */
/* 02/17/04 gam; Implement new rules for thresholdFlag. */
/* 02/17/04 gam; Change outputFlag to outputSUMFlag; add outputEventFlag and keepThisNumber  */
/*               (Remove windowFilterParam2 and windowFilterParam3)                          */
/*               Implement new rules for outputSUMFlag, outputEventFlag. and keepThisNumber. */
/*               Add loudestPeaksArray:                                                      */
/*               Keep the loudest event for every eventBandWidth = params->bandSUM/keepThisNumber, which is the same as */
/*               keep the loudest event for every nBinsPerOutputEvent = params->nBinsPerSUM/keepThisNumber. */
/* 02/19/04 gam; make maxWidth type REAL8. */

/* 02/20/04 gam; Fix bugs; make sure outputSUMFlag, thresholdFlag, and outputEventFlag are used correctly */
/* 02/20/04 gam; Make sure maxPower and totalEventCount gets set correctly for search summary; remove maxEventNum */
/* 02/20/04 gam; Since (outputEventFlag & 2) > 0 and keepThisNumber = 1 will keep loudest event, remove code that kept loudestPeak. */
/* 02/20/04 gam; In InitializePeaksArray, initialize frequency to frequency of first bin for each event */
/* 02/20/04 gam; change threshold6 to INT4 maxWidthBin; remove maxWidth; replace fMax and fMin with ifMax and ifMin */
/*               use threshold 3 as fracPwDrop; threshold 4 and 5 are currently unused */
/* 02/20/04 gam; if (thresholdFlag & 4) > 0 then do not include overlap events in output */
/* 02/20/04 gam; Replace thresholdFlag with vetoWidePeaks and vetoOverlapPeaks in struct LALFindStackSlidePeakParams */
/* 02/23/04 gam; Fix bug; allow thresholdFlag < 1 and outputEventFlag < 1, which means do not set thresholds and do not output events. */
/* 02/23/04 gam; When DEBUG_PARAMETERS, format for params->maxWidthBins is %i */
/* 02/24/04 gam; Before LALInitBarycenter set edat->leap = 13; Was NOT getting initialized. Note 13 is OK for 2000; Check for current date! */
/* 03/01/04 gam; Call LALLeapSecs to get edat->leap */
/* 03/01/04 gam; if (params->normalizationFlag & 2) > 0 normalize BLKs else normalize STKs */
/* 03/02/04 gam; if (params->thresholdFlag & 8 > 0) update pw_mean_thissum, pw_stddev_thissum ignoring peak bins; update pwr_snr. */
/* 03/02/04 gam; add returnPeaks to LALFindStackSlidePeakParams; if false return info about SUM and peaks without returning peaks themselves. */
/* 03/02/04 gam; Add LALFindStackSlidePeakOutputs to LALFindStackSlidePeaks to output pwMeanWithoutPeaks, pwStdDevWithoutPeaks, binsWithoutPeaks, acceptedEventCount, rejectedEventCount */
/* 03/03/04 gam; change windowFilterParam1 to normalizationThreshold */
/* 03/03/04 gam; if (params->normalizationFlag & 8) > 0 normalize with veto on power above normalizationThreshold = max_power_allowed/mean_power */
/* 03/03/04 gam; When DEBUG_PARAMETERS output parameters in syntax that tcl could use to set these with documentation. */
/* 03/03/04 gam; If params->testFlag == 1 output the Hough number count instead of power using threshold5 as the cutoff after normalizing. */
/* 04/14/04 gam; if (params->normalizationFlag & 4) > 0 normalize STKs using running median */
/* 04/15/04 gam; if (params->normalizationFlag & 16) > 0 then output into .Sh file GPS startTime and PSD estimate for each SFT. */
/* 04/15/04 gam; Add INT2 params->debugOptionFlag; if (params->debugOptionFlag & 1) > 0 print commandline args to stdout */
/* 04/15/04 gam; Change DEBUG_PARAMETERS to INCLUDE_DEBUG_PARAMETERS_CODE; allow printing of commandline args to stdout if (params->debugOptionFlag & 1) > 0 */
/* 04/15/04 gam; if ((params->debugOptionFlag & 2) > 0 ) then print table if INCLUDE_PRINT_PEAKS_TABLE_CODE (change DEBUG_FIND_PEAKS_TABLE to INCLUDE_PRINT_PEAKS_TABLE_CODE) */
/* 04/15/04 gam; if ((params->debugOptionFlag & 4) > 0 ) then print sky positions INCLUDE_DEBUG_SKY_POSITIONS_CODE (change DEBUG_SKY_POSITIONS to INCLUDE_DEBUG_SKY_POSITIONS_CODE) */
/* 04/15/04 gam; Add debugOptionFlag to struct StackSlideSkyPatchParams */
/* 04/26/04 gam; Change LALStackSlide to StackSlide and LALSTACKSLIDE to STACKSLIDE for initial entry to LALapps. */
/* 05/05/04 gam; Change params->normalizationThreshold to params->normalizationParameter.  If normalizing with running median use this to correct bias in median to get mean. */
/* 05/11/04 gam; Add code to software inject signals into the SFTs for Monte Carlo simulations */
/* 05/11/04 gam; Move SECTION: set up parameter space into StackSlideInitSearch; free memory in StackSlideFinalizeSearch */
/* 05/11/04 gam; Free edata ephemeris data in StackSlideFinalizeSearch */
/* 05/11/04 gam; If (params->testFlag & 1) > 0 output the Hough number count */
/* 05/11/04 gam; If (params->testFlag & 2) > 0 inject fake signals and run Monte Carlo Simulation; use threshold4 for h_0*/
/* 05/26/04 gam; Move writing to stackslide search summary table to StackSlideConditionData */
/* 05/26/04 gam; Change finishedSUMs to finishSUMs; add startSUMs; defaults are TRUE; use to control I/O during Monte Carlo */
/* 05/26/04 gam; Add whichMCSUM = which Monte Carlo SUM; default is -1. */
/* 07/09/04 gam; If using running median, use LALRngMedBias to set params->normalizationParameter to correct bias in the median. */
/* 08/02/04 gam; if (params->testFlag & 4) > 0 ComputeSky uses reference time: params->timeStamps[0].gpsSeconds, params->timeStamps[0].gpsNanoSeconds */
/* 08/30/04 gam; if (outputEventFlag & 4) > 0 set returnOneEventPerSUM to TRUE; only the loudest event from each SUM is then returned. */
/* 09/27/04 gam; if numBLKsPerSTK == 1 then want to ensure 1 to 1 correspondence between BLKs and STKs */
/*               WARNING: also, should avoid other cases for now until we decide how to handle         */
/*                        overlapping SFTs and gaps that are not multiples of tBLK!                    */
/* 10/28/04 gam; if (params->weightFlag & 1) > 0 then use PowerFlux weights from running median. Must have (params->normalizationFlag & 4) > 0 */
/* 10/28/04 gam; if (params->weightFlag & 2) > 0 then include beam pattern F_+ in PowerFlux weights from running median. Must have (params->normalizationFlag & 4) > 0 */
/* 10/28/04 gam; if (params->weightFlag & 4) > 0 then include beam pattern F_x in PowerFlux weights from running median. Must have (params->normalizationFlag & 4) > 0 */
/* 10/28/04 gam; change unused params->windowFilterFlag to REAL8 params->orientationAngle used to find F_+ and F_x with weightFlag or MC with fixed polarization angle */
/* 11/01/04 gam; if (params->weightFlag & 8) > 0 rescale STKs with threshold5 to prevent dynamic range issues. */
/* 11/18/04 gam; in GetDetResponseTStampMidPts latitute is DEC and longitude is RA, or course! */
/* 12/03/04 gam; add parameter: BOOLEAN divideSUMsByNumSTKs; default is TRUE; FALSE if Hough Test or PowerFlux weighting is done. */
/* 12/06/04 gam; get params->sampleRate, = effective sample rate, from the SFTs; calculate params->deltaT after reading SFTs. */
/* 12/06/04 gam; add params->gpsEpochStartTimeNan; get gpsEpochStartTime, gpsEpochStartTimeNan, and gpsStartTime from command line; */
/* 12/06/04 gam; change calibrationFlag to cosInclinationAngle */
/* 12/06/04 gam; if (params->testFlag & 8) > 0 use fixed values for psi and cosIota during Monte Carlo simulations */
/* 12/18/04 gam; Change LALSRunningMedian to LALSRunningMedian2 */
/* 12/18/04 gam; Remove unused CVS_REVISION, CVS_DATE, and UNKNOWN. */
/* 02/25/05 gam; if params->debugOptionFlag == 1 then exit after printing parameters */
/* 02/25/05 gam; revise SECTION make SUMs:                                                           */
/*               1. remove obsolete code                                                             */
/*               2. clean up indentation                                                             */
/*               3. break into smaller functions                                                     */
/*               4. move loops for isolated case (params->binaryFlag == 0) into StackSlideIsolated.c */
/*               5. use new StackSlide function for isolated case.                                   */
/* 02/28/05 gam; add extra parameters needed by loop code to StackSlideSearchParams struct */
/* 04/12/05 gam; LIGOLW_XML_TABLE_FOOTER removed from lal, so add as STACKSLIDE_XML_TABLE_FOOTER. */
/* 04/12/05 gam; Add StackSlideSearchParams *params to StackSlideBinary. */
/* 04/12/05 gam; Remove from StackSlideParams *stksldParams, those already in StackSlideSearchParams *params */
/* 04/12/05 gam; if ((params->debugOptionFlag & 8) > 0 ) find maxPwr each SFT, replace bin with 1, all other bins with 0 */
/* 05/06/05 gam; if ((params->debugOptionFlag & 16) > 0 ) also replace one bin to either side of bin with maxPwr with 1 */
/* 05/06/05 gam; If params->debugOptionFlag & 128 > 0 and isolated case, just creates a SUM from the STKs without sliding */
/* 05/13/05 gam; Add function FindAveEarthAcc that finds aveEarthAccVec, the Earth's average acceleration vector during the analysis time. */
/* 05/13/05 gam; Add function FindLongLatFromVec that find for a vector that points from the center to a position on a sphere, the latitude and longitude of this position */
/* 05/13/05 gam; Add function RotateSkyCoordinates that transforms longIn and latIn to longOut, latOut as related by three rotations. */
/* 05/13/05 gam; Add function RotateSkyPosData that rotates skyPosData using RotateSkyCoordinates */
/* 05/13/05 gam; if (params->parameterSpaceFlag & 1) > 0 rotate skyPosData into coordinates with Earth's average acceleration at the pole. */
/* 05/13/05 gam; if (params->parameterSpaceFlag & 2) > 0 rotate skyPosData into galactic plane */
/* 05/14/05 gam; Change unused numSUMsPerCall to linesAndHarmonicsFile; file with instrument line and harmonic spectral disturbances data. */
/* 05/14/05 gam; if (params->normalizationFlag & 32) > 0 then clean SFTs using info in linesAndHarmonicsFile */
/* 05/19/05 gam; Add INT4 *sumBinMask; params->sumBinMask == 0 if bin should be excluded from search or Monte Carlo due to cleaning */
/* 05/19/05 gam; In LALFindStackSlidePeaks set binFlag = 0 if bin excluded; initialize binFlag with sumBinMask. */
/* 05/19/05 gam; In LALUpdateLoudestFromSUMs exclude bins with sumBinMask == 0. */
/* 05/22/05 gam; Fix bug in FindLongLatFromVec; when longitude positive, then OK as is, when negative need to add 2pi not pi! */
/* 05/24/05 gam; make maxPower and totalEventCount part of params; change finishSUMs to finishPeriodicTable; end xml in FinalizeSearch */
/* 05/24/05 gam; change tDomainChannel to priorResultsFile; change fDomainChannel to parameterSpaceFile */
/* 05/25/05 gam; change patchNumber to maxMCinterations; change inputDataTypeFlag to maxMCfracErr */
/* 05/24/05 gam; if (params->testFlag & 16 > 0) use results from prior jobs in the pipeline and report on current MC results */
/* 05/24/05 gam; if (params->testFlag & 32 > 0) iterate MC up to 10 times to converge on desired confidence */
/* 05/24/05 gam; if (params->debugOptionFlag & 32 > 0) print Monte Carlo Simulation results to stdout */
/* 05/24/05 gam; add StackSlideMonteCarloResultsTable */
/* 07/13/05 gam; if (params->normalizationFlag & 32) > 0 then ignor bins with params->sumBinMask == 0 */
/* 07/13/05 gam; if (params->normalizationFlag & 64) > 0 then clean SFTs using info in linesAndHarmonicsFile */
/* 07/13/05 gam; make RandomParams *randPar a parameter for CleanCOMPLEX8SFT; initialze RandomParams *randPar once to avoid repeatly opening /dev/urandom */
/* 07/17/05 gam; Change ...Deriv5 command line arguments to ones that control new Monte Carlo (MC) options */
/* 07/29/05 gam; if (params->testFlag & 64) > 0 set searchSurroundingPts == 1 and  */
/*               search surrounding parameters space pts; else search nearest only */
/* 08/24/05 gam; Fix off by one error when computing tmpNumRA in CountOrAssignSkyPosData */
/* 08/31/05 gam; In StackSlideComputeSky set ssbT0 to gpsStartTime, which is gpsEpochStartTime in this code; this now gives the epoch that defines T_0 at the SSB! */
/* 09/06/05 gam; Change params->maxMCfracErr to params->maxMCErr, the absolute error in confidence for convergence. */
/* 09/09/05 gam; Use SFT cleaning function in LAL SFTClean.h rather than in SFTbin.h */
/* 09/12/05 gam; if ( (params->weightFlag & 16) > 0 ) save inverseMedians and weight STKs with these. */
/* 09/12/05 gam; if (params->testFlag & 128) > 0 make BLKs and STKs narrower band based on extra bins */
/* 09/14/05 gam; add more vetting of command line arguments and ABORTs */
/* 09/16/05 gam; In CountOrAssignSkyPosData and MC code, for each DEC adjust deltaRA and numRA to evenly  */
/*               space grid points so that deltaRA used is <= input deltaRA from the command line divided */
/*               by cos(DEC). This fixes a problem where the last grid point could wrap around the sphere */
/*               and be closer to the first grid point that the spacing between other grid points.        */
/* 09/16/05 gam; Add some error checking and clean up parameter space assignment code. */
/* 09/23/05 gam; if ((params->debugOptionFlag & 4) > 0 ) also print out the spindown grid. */
/* 09/23/05 gam; Besides checking that startRA and startDEC are in range, also check stopRA and stopDEc. */
/* 09/23/05 gam; In addition to maxSpindownFreqShift add maxSpinupFreqShift */
/* 10/20/05 gam; if ( (params->weightFlag & 8) > 0 ) renorm the BLK (SFT) data up front with the inverse mean absolute value of the BLK data = params->blkRescaleFactor */
/* 10/20/05 gam; if using running median and params->normalizationParameter > 0, then use to correct the bias. */
/* 11/09/05 gam; if normalizationFlag & 1 > 0 normalize STKs using mean (but if normalizationFlag & 4 > 0, running median takes precedence) */
/* 12/07/05 gam; fix bug in StackSlideCleanSFTs: need to initialize nBinsPerBLK before allocating memory. */
/* 01/12/06 gam; Add function WriteStackSlideLEsToPriorResultsFile; if ( (outputEventFlag & 8) > 0 ) && !( (params->testFlag & 2) > 0 ) write loudest events to params->priorResultsFile */
/* 01/12/06 gam; Always set maxMC = params->maxMCinterations; if NOT ( (params->testFlag & 32) > 0 ) then run MC maxMCinterations times; linearly interpolate to get final result */
/* 02/20/06 gam; if (debugOptionFlag & 64) > 0 generated SUMs with an estimate of StackSlide Power for a given source; also get F_+, F_x, sqrt(Sn) the bin mismatch, and binoffset if (debugOptionFlag & 2) > 0 */
/* 04/03/06 gam; to save disk space, if running an MC do not output rows with loudest event from each injection unless (params->outputEventFlag & 16) > 0 */

/*********************************************/
/*                                           */
/* START SECTION: define preprocessor flags  */
/*                                           */
/*********************************************/
/* #define INCLUDE_DEMOD_CODE */ /* GET CODE WORKING WITH STKS = PSDs FIRST */
/* #define INCLUDE_TIMEFLOAT_CODE */
#define INCLUDE_DEBUG_PARAMETERS_CODE
#define INCLUDE_PRINT_PEAKS_TABLE_CODE
#define INCLUDE_DEBUG_SKYANDSPINDOWNGRID_CODE
/* #define DEBUG_DRIVESTACKSLIDE */
/* #define DEBUG_SUM_TEMPLATEPARAMS */
/* #define DEBUG_EPHEMERISDATA */
/* #define DEBUG_ROTATESKYCOORDINATES */
/* #define DEBUG_POWER_STATS */
/* #define DEBUG_FIND_PEAKS */
/* #define DEBUG_STACKSLIDEARGS */
/* #define DEBUG_COMPUTED_PARAMETERS */
/* #define DEBUG_BLK_CODE */
/* #define DEBUG_DEMOD_CODE */
/* #define DEBUG_INPUTBLK_CODE */
/* #define DEBUG_NORMALIZEBLKS */
/* #define DEBUG_CLEANLINESANDHARMONICS */
/* #define DEBUG_CLEANED_SFTS */
/* #define DEBUG_SUMBINMASK */
/* #define DEBUG_CALIBRATEBLKS */
/* #define DEBUG_TESTCASE */
/* #define DEBUG_CHECKDYNAMICRANGEANDRESCALEBLKDATA */
/* #define INCLUDE_INTERNALLALINITBARYCENTER */
/*********************************************/
/*                                           */
/* END SECTION: define preprocessor flags    */
/*                                           */
/*********************************************/
/*********************************************/
/*                                           */
/* START SECTION: include header files       */
/* (Note that most include statements        */
/*  are in the header files below.)          */
/*                                           */
/*********************************************/
#include "DriveStackSlide.h"
#include "StackSlideIsolated.h"
#include "StackSlideBinary.h"
/*********************************************/
/*                                           */
/* END SECTION: include header files         */
/*                                           */
/*********************************************/

/**********************************************/
/*                                            */
/* START SECTION: more prototype declarations */
/*                                            */
/**********************************************/
#ifdef INCLUDE_INTERNALLALINITBARYCENTER
 void InternalLALInitBarycenter(LALStatus *stat, EphemerisData *edat);
#endif
#ifdef INCLUDE_TIMEFLOAT_CODE
 static void TimeToFloat(REAL8 *f, LIGOTimeGPS *tgps); /* Copied from LALDemodTest.c */
 static void FloatToTime(LIGOTimeGPS *tgps, REAL8 *f); /* Copied from LALDemodTest.c */
#endif
/**********************************************/
/*                                            */
/* END SECTION: more prototype declarations   */
/*                                            */
/**********************************************/

#define PROGRAM_NAME "stackslide"
#define CVS_ID_STRING "$Id$"
#define CVS_SOURCE "$Source$"
#define BLANK " "

/******************************************/
/*                                        */
/* START FUNCTION: StackSlideInitSearch   */
/*                                        */
/******************************************/
void StackSlideInitSearch(
    LALStatus              *status,
    StackSlideSearchParams *params,
    int                     argc,
    char                   *argv[]    
    )
{
  
  /* UINT2 i; */
  INT4 i;
  INT4 k;  /* 05/11/04 gam */
  
 INITSTATUS(status);
 ATTATCHSTATUSPTR (status);

  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nStackSlideInitSearch\n");
  	fflush(stdout);
  #endif

  #ifdef DEBUG_STACKSLIDEARGS
  	printf("\nargc = %i\n",argc);
  #endif

/******************************************/
/*                                        */
/* START SECTION: parameters passed as    */
/* arguments.  Those indented are         */
/* computed from these.                   */
/*                                        */
/******************************************/

  #ifdef DEBUG_STACKSLIDEARGS
  	fprintf(stdout, "\nSTART SECTION: parameters passed as arguments\n");
  	fflush(stdout);
  #endif

 /* Initialize */
 params->ifoNickName = NULL;
 params->IFO = NULL;
 params->patchName = NULL;
 params->priorResultsFile = NULL;   /* 05/24/05 gam */
 params->parameterSpaceFile = NULL; /* 05/24/05 gam */
 params->sunEdatFile = NULL;
 params->earthEdatFile = NULL;
 params->sftDirectory = NULL;
 params->outputFile = NULL;
 params->linesAndHarmonicsFile = NULL; /* 05/14/05 gam */
 params->randPar = NULL; /* 07/13/05 gam */
 
 params->stksldSkyPatchData = (StackSlideSkyPatchParams *)LALMalloc(sizeof(StackSlideSkyPatchParams)); /* 01/31/04 gam */
 
 params->debugOptionFlag = 0; /* 04/15/04 gam; default is to print no debugging information */
 
 params->sampleRate = -1.0; /* 12/06/04 gam; these will get updated when BLKs are read. */
 params->deltaT = -1.0;
 
 params->gpsStartTimeNan = 0; /* 12/06/04 gam; search startTime is whole number; however gpsEpochStartTimeNan can be nonzero */
 
/* 05/02/14 vir: initialize binary params */
params->binaryFlag = 0;          /*the isolated case is the default*/
params->OrbitalEccentricity = 0;
params->alphaSX1=0;
params->deltaSX1=0;
params->ArgPeriapse = 0;
params->TperiapseSSBSec = 0;
params->TperiapseSSBNanoSec = 0;
params->SMAcentral = 0;
params->deltaSMA = 0;
params->nMaxSMA=0;
params->nMaxTperi=0;

/* 07/17/05 gam; currently unused and fixed as zero: */
params->startFDeriv5 = 0.0;
params->stopFDeriv5  = 0.0;
params->deltaFDeriv5 = 0.0;
params->numFDeriv5   =   0;
 
 for (i = 1; i < argc; i++) {

	#ifdef DEBUG_STACKSLIDEARGS
		printf("\nargv[%i] = %s\n",i,argv[i]);
	#endif

	switch(i) {
		case  1: params->gpsEpochStartTimeSec = (UINT4)atol(argv[i]); break;
		case  2: params->gpsEpochStartTimeNan = (UINT4)atol(argv[i]); break;
		case  3: params->gpsStartTimeSec = (UINT4)atol(argv[i]); break;
		case  4: params->duration = (REAL8)atof(argv[i]); break;

		case  5: params->numBLKs = (INT4)atol(argv[i]); break;
		case  6: params->tBLK = (REAL8)atof(argv[i]); break;
		case  7: params->tEffBLK = (REAL8)atof(argv[i]); break;
		case  8: params->f0BLK = (REAL8)atof(argv[i]); break;
		case  9: params->bandBLK = (REAL8)atof(argv[i]); break;
		case 10: params->nBinsPerBLK = (INT4)atol(argv[i]); break;

		case 11: params->numBLKsPerSTK = (INT4)atol(argv[i]); break;
		case 12: params->tEffSTK = (REAL8)atof(argv[i]); break;
		case 13: params->f0STK = (REAL8)atof(argv[i]); break;
		case 14: params->bandSTK = (REAL8)atof(argv[i]); break;
		case 15: params->nBinsPerSTK = (INT4)atol(argv[i]); break;

		case 16: params->numSTKsPerSUM = (INT4)atol(argv[i]); break;
		case 17: params->tEffSUM = (REAL8)atof(argv[i]); break;
		case 18: params->f0SUM = (REAL8)atof(argv[i]); break;
		case 19: params->bandSUM = (REAL8)atof(argv[i]); break;
		case 20: params->nBinsPerSUM = (INT4)atol(argv[i]); break;

		case 21: params->ifoNickName = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->ifoNickName, argv[i]); break;
		case 22: params->IFO = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->IFO, argv[i]); break;
		case 23: params->patchName = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->patchName, argv[i]); break;

		case 24: params->maxMCinterations = (INT4)atol(argv[i]); break;

		case 25: params->priorResultsFile = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->priorResultsFile, argv[i]); break;
		case 26: params->parameterSpaceFile = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->parameterSpaceFile, argv[i]); break;

		case 27: params->maxMCErr = (REAL8)atof(argv[i]); break;

		case 28: params->parameterSpaceFlag = (INT2)atoi(argv[i]); break;

		case 29: params->stackTypeFlag = (INT2)atoi(argv[i]); break;
		case 30: params->Dterms = (INT4)atoi(argv[i]); break;

		case 31: params->thresholdFlag = (INT2)atoi(argv[i]); break;
		case 32: params->threshold1 = (REAL4)atof(argv[i]); break;
		case 33: params->threshold2 = (REAL4)atof(argv[i]); break;
		case 34: params->threshold3 = (REAL4)atof(argv[i]); break;
		case 35: params->threshold4 = (REAL4)atof(argv[i]); break;
		case 36: params->threshold5 = (REAL4)atof(argv[i]); break;
		case 37: params->maxWidthBins = (INT4)atof(argv[i]); break;
		
		case 38: params->weightFlag = (INT2)atoi(argv[i]); break;

		case 39: params->orientationAngle = (REAL8)atof(argv[i]); break;
		case 40: params->cosInclinationAngle = (REAL8)atof(argv[i]); break;

		case 41: params->normalizationFlag = (INT2)atoi(argv[i]); break;
		case 42: params->f0NRM = (REAL8)atof(argv[i]); break;
		case 43: params->bandNRM = (REAL8)atof(argv[i]); break;
		case 44: params->nBinsPerNRM = (INT4)atol(argv[i]); break;
		case 45: params->normalizationParameter = (REAL4)atof(argv[i]); break;
		
		case 46: params->testFlag = (INT2)atoi(argv[i]); break;
		
		case 47: params->linesAndHarmonicsFile = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->linesAndHarmonicsFile, argv[i]); break;
		
		case 48: params->outputSUMFlag = (INT2)atoi(argv[i]); break;
		case 49: params->outputEventFlag = (INT2)atoi(argv[i]); break;
		case 50: params->keepThisNumber = (INT4)atoi(argv[i]); break;

		case 51: params->stksldSkyPatchData->startRA = (REAL8)atof(argv[i]); break;
		case 52: params->stksldSkyPatchData->stopRA = (REAL8)atof(argv[i]); break;
		case 53: params->stksldSkyPatchData->deltaRA = (REAL8)atof(argv[i]); break;
		case 54: params->stksldSkyPatchData->numRA = (INT4)atol(argv[i]); break;

		case 55: params->stksldSkyPatchData->startDec = (REAL8)atof(argv[i]); break;
		case 56: params->stksldSkyPatchData->stopDec = (REAL8)atof(argv[i]); break;
		case 57: params->stksldSkyPatchData->deltaDec = (REAL8)atof(argv[i]); break;
		case 58: params->stksldSkyPatchData->numDec = (INT4)atol(argv[i]); break;

		case 59: params->numSpinDown = (INT4)atol(argv[i]); break;

		case 60: params->startFDeriv1 = (REAL8)atof(argv[i]); break;
		case 61: params->stopFDeriv1 = (REAL8)atof(argv[i]); break;
		case 62: params->deltaFDeriv1 = (REAL8)atof(argv[i]); break;
		case 63: params->numFDeriv1 = (INT4)atol(argv[i]); break;

		case 64: params->startFDeriv2 = (REAL8)atof(argv[i]); break;
		case 65: params->stopFDeriv2 = (REAL8)atof(argv[i]); break;
		case 66: params->deltaFDeriv2 = (REAL8)atof(argv[i]); break;
		case 67: params->numFDeriv2 = (INT4)atol(argv[i]); break;

		case 68: params->startFDeriv3 = (REAL8)atof(argv[i]); break;
		case 69: params->stopFDeriv3 = (REAL8)atof(argv[i]); break;
		case 70: params->deltaFDeriv3 = (REAL8)atof(argv[i]); break;
		case 71: params->numFDeriv3 = (INT4)atol(argv[i]); break;

		case 72: params->startFDeriv4 = (REAL8)atof(argv[i]); break;
		case 73: params->stopFDeriv4 = (REAL8)atof(argv[i]); break;
		case 74: params->deltaFDeriv4 = (REAL8)atof(argv[i]); break;
		case 75: params->numFDeriv4 = (INT4)atol(argv[i]); break;

		case 76: params->numMCInjections = (INT4)atol(argv[i]); break;
		case 77: params->numMCRescalings = (INT4)atol(argv[i]); break;
		case 78: params->rescaleMCFraction = (REAL8)atof(argv[i]); break;
		case 79: params->parameterMC = (REAL8)atof(argv[i]); break;

		case 80: params->sunEdatFile = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->sunEdatFile, argv[i]); break;
		case 81: params->earthEdatFile = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->earthEdatFile, argv[i]); break;
		case 82: params->sftDirectory = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->sftDirectory, argv[i]); break;
		case 83: params->outputFile = (CHAR *) LALMalloc( (strlen( argv[i] ) + 1) * sizeof(CHAR) );
		         strcpy(params->outputFile, argv[i]); break;
		case 84: params->debugOptionFlag = (INT2)atoi(argv[i]); break;
	        case 85: params->binaryFlag=(INT2)atoi(argv[i]); break;/*must be =1 for binaries */
        /*added a few entries regarding the orbital motion and if binary alpha and delta*/
                case 86: params->OrbitalEccentricity=(REAL8)atof(argv[i]); break;
		case 87: params->alphaSX1=(REAL8)atof(argv[i]); break;
		case 88: params->deltaSX1=(REAL8)atof(argv[i]); break;
		case 89: params->ArgPeriapse=(REAL8)atof(argv[i]); break;
		case 90: params->TperiapseSSBSec=(UINT4)atol(argv[i]); break;
		case 91: params->TperiapseSSBNanoSec=(UINT4)atol(argv[i]); break;	 
		case 92: params->SMAcentral=(REAL8)atof(argv[i]); break;
		case 93: params->deltaSMA=(REAL8)atof(argv[i]); break;/*put this =0 if you don't want a random SMA*/
                case 94: params->nMaxSMA=(INT4)atoi(argv[i]); break;	/*05/02/18 vir:added entry*/		 
		case 95: params->nMaxTperi=(INT4)atoi(argv[i]); break; /*05/02/18 vir: added entry*/

	}
  } /* end for (i=1; i<argc; i++)  */

#ifdef DEBUG_INPUTBLK_CODE
    fprintf(stdout,"\n\nparams->numBLKs = %i \n", params->numBLKs);
    fprintf(stdout,"params->nBinsPerBLK = %i \n", params->nBinsPerBLK);
    fprintf(stdout,"params->f0BLK = %g \n", params->f0BLK);
    fprintf(stdout,"params->bandBLK = %g \n", params->bandBLK);
    fflush(stdout);
 #endif
  
  params->stksldSkyPatchData->debugOptionFlag = params->debugOptionFlag;
  
  /* params->deltaT = 1.0/params->sampleRate; */ /* 12/06/04 gam */
  params->dfBLK = 1.0/params->tEffBLK;

  /* params->numSTKs is not computed here. Not duration/tSTK if gaps are present */
  params->tSTK = params->tBLK*params->numBLKsPerSTK;       /* duration in seconds of STKs = tBLK*numBLKsPerSTK */
  params->dfSTK = 1.0/params->tEffSTK;                     /* Freq resolution of STK = 1.0/tEffSTK. Could be different from 1/tSTK due to oversampling  */

  params->tSUM = params->tSTK*params->numSTKsPerSUM; /* duration in seconds of output SUMs = tSTK*numSTKsPerSUM. (Usually = duration) */
  params->numSUMsPerParamSpacePt = params->duration/params->tSUM; /* Number of output SUMs per parameter space point = params->duration/params->tSUM. (Usually will == 1) */
  if (params->numSUMsPerParamSpacePt != 1) {
     ABORT( status, DRIVESTACKSLIDEH_ENSUMPERPARAMPT, DRIVESTACKSLIDEH_MSGENSUMPERPARAMPT); /* 09/16/06 gam; currently must == 1 */
  }
  params->dfSUM = 1.0/params->tEffSUM;  /* Freq resolution of output SUMs.  dfSUM = 1.0/tEffSUM. Could be different from dfSTK in some algorithms?  */

  /* Initialize and set up number of points in the parameter space */
  params->numSkyPosTotal = 0;     /* Total number of Sky positions to cover */
  params->numFreqDerivTotal = 0;  /* Total number of Frequency evolution models to cover */
  params->numParamSpacePts = 0;   /* Total number of points in the parameter space to cover */

  /* 05/13/05 gam; add options */
  if (params->parameterSpaceFlag >= 0) {

     /* The third argument is 0; this just counts the number of sky postions: */
     CountOrAssignSkyPosData(params->skyPosData,&(params->numSkyPosTotal),0,params->stksldSkyPatchData);
     if (params->numSkyPosTotal <= 0) {
        ABORT( status, DRIVESTACKSLIDEH_ENUMSKYPOS, DRIVESTACKSLIDEH_MSGENUMSKYPOS); /* 09/16/06 gam */
     }

     for (i=0;i<params->numSpinDown;i++) {
         switch(i) {
           case  0: params->numFreqDerivTotal = params->numFDeriv1; break;
           case  1: params->numFreqDerivTotal *= params->numFDeriv2; break;
           case  2: params->numFreqDerivTotal *= params->numFDeriv3; break;
           case  3: params->numFreqDerivTotal *= params->numFDeriv4; break;
           case  4: params->numFreqDerivTotal *= params->numFDeriv5; break;
         }
     }

     /* 02/28/05 gam; moved here */
     if (params->numFreqDerivTotal != 0) {
        params->numFreqDerivIncludingNoSpinDown = params->numFreqDerivTotal;
     } else {
        params->numFreqDerivIncludingNoSpinDown = 1;  /* Even if numSpinDown = 0 still need to count case of zero spindown. */
     }

     /*05/02/18 vir:*/
     if (params->binaryFlag==0) {
        params->numParamSpacePts = params->numSkyPosTotal*params->numFreqDerivIncludingNoSpinDown;
     } else { 
        params->numParamSpacePts = params->numSkyPosTotal*params->numFreqDerivIncludingNoSpinDown*params->nMaxSMA*params->nMaxTperi;
     }

     /* 09/16/05 gam; moved here and revised */
     params->maxSpindownFreqShift = 0; /* Used when "cleaning" SFTs to decide how many bins to ignore near a line due to spindown */
     params->maxSpinupFreqShift = 0;   /* 09/23/05 gam; used when "cleaning" SFTs to decide how many bins to ignore near a line due to spinup */
     for(i=0;i<params->numSpinDown;i++) {
       /* This estimates the maximum the frequency of a signal can change due to spindown or spinup during the analysis, */
       /* Here we approximate the maximum value of T - T0 to an accuracy of 8 minutes. If the frequency changes by more  */
       /* than 0.5 bins during 8 minutes due to spindown you are looking for very young pulsars!                         */
       REAL8 maxDeltaT = MAX(fabs((REAL8)(params->gpsStartTimeSec + params->duration - params->gpsEpochStartTimeSec)),fabs((REAL8)(params->gpsStartTimeSec - params->gpsEpochStartTimeSec)));
       /* Typically maxDeltaT is about equal to the duration; if gpsEpochStartTimeSec is larger than gpsStartTimeSec + duration/2 then we over-estimate the effects of spindown and spinup here. */
       switch(i) {
        case  0: params->maxSpindownFreqShift += STKSLDMIN(params->startFDeriv1, params->stopFDeriv1)*maxDeltaT; break;
        case  1: params->maxSpindownFreqShift += STKSLDMIN(params->startFDeriv2, params->stopFDeriv2)*pow(maxDeltaT,2.0); break;
        case  2: params->maxSpindownFreqShift += STKSLDMIN(params->startFDeriv3, params->stopFDeriv3)*pow(maxDeltaT,3.0); break;
        case  3: params->maxSpindownFreqShift += STKSLDMIN(params->startFDeriv4, params->stopFDeriv4)*pow(maxDeltaT,4.0); break;
        case  4: params->maxSpindownFreqShift += STKSLDMIN(params->startFDeriv5, params->stopFDeriv5)*pow(maxDeltaT,5.0); break;
       }
       /* 09/23/05 gam */
       switch(i) {
        case  0: params->maxSpinupFreqShift += STKSLDMAX(params->startFDeriv1, params->stopFDeriv1)*maxDeltaT; break;
        case  1: params->maxSpinupFreqShift += STKSLDMAX(params->startFDeriv2, params->stopFDeriv2)*pow(maxDeltaT,2.0); break;
        case  2: params->maxSpinupFreqShift += STKSLDMAX(params->startFDeriv3, params->stopFDeriv3)*pow(maxDeltaT,3.0); break;
        case  3: params->maxSpinupFreqShift += STKSLDMAX(params->startFDeriv4, params->stopFDeriv4)*pow(maxDeltaT,4.0); break;
        case  4: params->maxSpinupFreqShift += STKSLDMAX(params->startFDeriv5, params->stopFDeriv5)*pow(maxDeltaT,5.0); break;
       }
     }

  } else {
     ABORT( status, DRIVESTACKSLIDEH_EPARAMSPACEFLAG, DRIVESTACKSLIDEH_MSGEPARAMSPACEFLAG); /* 02/02/04 gam */
  } /* END if (params->parameterSpaceFlag >= 0) else ...*/  
  params->numSUMsTotal = params->numSUMsPerParamSpacePt*params->numParamSpacePts;  /* Total Number of Sums = numSUMsPerParamPt*numParamSpacePts */
  #ifdef DEBUG_COMPUTED_PARAMETERS
    fprintf(stdout,"numSUMstotal is %d\n",params->numSUMsTotal);
    fflush(stdout);
  #endif
  
  #ifdef INCLUDE_DEBUG_PARAMETERS_CODE
   if ((params->debugOptionFlag & 1) > 0 ) {
        /* if above defined then the code prints out documentation for the command line arguments */
    	fprintf(stdout,"\n");
    	fprintf(stdout,"#Key to terminology: \n");
    	fprintf(stdout,"#BLKs: The input blocks of frequency domain data (e.g., from SFTs). \n");
    	fprintf(stdout,"#STKs: The BLK data is stacked up and turned in STKs. \n");
    	fprintf(stdout,"#SUMs: The STKs are slid and then summed to produce SUMs; \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"#For historical reasons, parameters for controlling Monte Carlo Simulations occur in several places.\n");
    	fprintf(stdout,"#See: maxMCinterations, maxMCErr, orientationAngle, cosInclinationAngle, weightFlag, testFlag, outputEventFlag, numMCInjections, numMCRescalings, rescaleMCFraction, parameterMC, and debugOptionFlag.\n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"#Example command line arguments for ComputeStackSlide: \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set gpsEpochStartTimeSec %23d; #1  UINT4 GPS seconds at the SSB giving reference time that defines the start of the epoch at the SSB.\n", params->gpsEpochStartTimeSec);
    	fprintf(stdout,"set gpsEpochStartTimeNan %23d; #2  UINT4 GPS nanoseconds at the SSB giving reference time that defines the start of the epoch at the SSB.\n", params->gpsEpochStartTimeNan);
    	fprintf(stdout,"set gpsStartTimeSec      %23d; #3  UINT4 analysis GPS start-time seconds at the detector. \n", params->gpsStartTimeSec);
    	fprintf(stdout,"#Note that for the isolated search case, to make Monte Carlo simulations with spindown work properly and self-consistently with the search one needs to:\n");
    	fprintf(stdout,"# 1. Use lal/packages/pulsar/src/ComputeSky.c version 1.11 or higher.\n");
    	fprintf(stdout,"# 2. Use lal/packages/inject/src/GeneratePulsarSignal.c version 1.40 or higher.\n");
    	fprintf(stdout,"# 3. Make sure gpsEpochStartTimeSec == gpsStartTimeSec = start time of the first BLK of data (i.e., the first SFT) and that gpsEpochStartTimeNan == 0.\n");
    	fprintf(stdout,"#When not running a Monte Carlo simulation set gpsEpochStartTimeSec and gpsEpochStartTimeNan to any T_0 desired, giving the start of the epoch at the SSB.\n");
    	fprintf(stdout,"set duration     %23.16e; #4  REAL8 analysis duration \n", params->duration);
    	fprintf(stdout,"#Note that duration is used to find SFTs with start time in the\n");
    	fprintf(stdout,"#interval [gpsStartTimeSec, gpsStartTimeSec+duration-tBLK] until \n");
    	fprintf(stdout,"#numBLKs SFTs are found. Thus, the actual duration of the\n");
    	fprintf(stdout,"#search is numBLKs*tBLK. See the next two parameters.\n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set numBLKs            %23d; #5  INT4 input number of blocks of data (e.g., actual number of SFTs used in this job). \n", params->numBLKs);
    	fprintf(stdout,"set tBLK               %23.16e; #6  REAL8 time baseline of input BLKs (e.g., of SFTs). \n", params->tBLK);
    	fprintf(stdout,"set tEffBLK            %23.16e; #7  REAL8 effective time baseline of input BLKs (e.g., not tBLK if oversampling used). \n", params->tEffBLK);
    	fprintf(stdout,"set f0BLK              %23.16e; #8  REAL8 start frequency of the input BLKs. \n", params->f0BLK);
    	fprintf(stdout,"set bandBLK            %23.16e; #9  REAL8 frequency band of input BLKs. \n", params->bandBLK);
    	fprintf(stdout,"set nBinsPerBLK        %23d; #10 INT4 number of frequency bins one BLKs. \n", params->nBinsPerBLK);
    	fprintf(stdout,"#Note that nBinsPerBLK takes precedence. An error is thrown if this does not correspond to tEffBLK*bandBLK rounded to the nearest integer. \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set numBLKsPerSTK      %23d; #11 INT4 number BLKs used to make one STK. \n", params->numBLKsPerSTK);
    	fprintf(stdout,"set tEffSTK            %23.16e; #12 REAL8 effective time baseline of STKs (deltaF = 1/tEffSTK). \n", params->tEffSTK);
    	fprintf(stdout,"set f0STK              %23.16e; #13 REAL8 start frequency of STKs. \n", params->f0STK);
    	fprintf(stdout,"set bandSTK            %23.16e; #14 REAL8 frequency band of STKs. \n", params->bandSTK);
    	fprintf(stdout,"set nBinsPerSTK        %23d; #15 INT4 number of frequency bins in one STK. \n", params->nBinsPerSTK);
    	fprintf(stdout,"#Note that nBinsPerSTK takes precedence. An error is thrown if this does not correspond to tEffSTK*bandSTK rounded to the nearest integer. \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set numSTKsPerSUM      %23d; #16 INT4 number of STKs used to make one SUM. \n", params->numSTKsPerSUM);
    	fprintf(stdout,"set tEffSUM            %23.16e; #17 REAL8 effective time baseline of SUMs (deltaF = 1/tEffSUM). \n", params->tEffSUM);
    	fprintf(stdout,"set f0SUM              %23.16e; #18 REAL8 start frequency of SUMs. \n", params->f0SUM);
    	fprintf(stdout,"set bandSUM            %23.16e; #19 REAL8 frequency band of SUMs. \n", params->bandSUM);
    	fprintf(stdout,"set nBinsPerSUM        %23d; #20 INT4 number of frequency bins in one SUM. \n", params->nBinsPerSUM);
    	fprintf(stdout,"#Note that nBinsPerSUM takes precedence. An error is thrown if this does not correspond to tEffSUM*bandSUM rounded to the nearest integer. \n");
    	fprintf(stdout,"#Since the entire frequency band slides together, bandSUM cannot exceed 1.0/((v_Earth/c)_max*tEffSTK),\n");
    	fprintf(stdout,"#where (v_Earth/c)_max = %23.16e. This keeps the maximum error in the number of bins to slide less than or equal to 0.5 bins. \n", ((REAL8)STACKSLIDEMAXV));
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set ifoNickName                             %s; #21 CHAR* H2, H1, L1, or G1. \n", params->ifoNickName);
    	fprintf(stdout,"set IFO                                    %s; #22 CHAR* LHO, LLO, or GEO. \n", params->IFO);
    	fprintf(stdout,"set patchName                       %s; #23 CHAR* a name to identify this search (e.g., S2 Galactic Center).\n", params->patchName);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set maxMCinterations   %23d; #24 INT4 maximum number of times to iterate entire Monte Carlo Simulation. \n", params->maxMCinterations);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set priorResultsFile     %s; #25 CHAR* file with the loudest event and estimated UL from a prior step in the pipeline. \n", params->priorResultsFile);
    	fprintf(stdout,"set parameterSpaceFile   %s; #26 CHAR* file with parameter space data \n", params->parameterSpaceFile);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set maxMCErr    %23.16e; #27 REAL8 maximum absolute error allowed when testing for convergence of confidence when iterating Monte Carlo. \n", params->maxMCErr);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set parameterSpaceFlag %23d; #28 INT2 how to generate parameter space. \n", params->parameterSpaceFlag);
    	fprintf(stdout,"#The parameterSpaceFlag options are: \n");
    	fprintf(stdout,"# if parameterSpaceFlag >= 0 generate sky positions uniformly on the sphere and spindowns without using parameter space metric.\n");
    	fprintf(stdout,"# if (parameterSpaceFlag & 1 > 0) rotate skyPosData into coordinates with Earth's average acceleration at the pole.\n");
    	fprintf(stdout,"# if (parameterSpaceFlag & 2) > 0 rotate skyPosData into galactic plane.\n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set stackTypeFlag      %23d; #29 INT2 how to generate STKs from BLKs. \n", params->stackTypeFlag);
    	fprintf(stdout,"set Dterms             %23d; #30 INT4 number of terms for Dirichlet kernel (when generating the F-stat or fake SFTs with a signal). \n", params->Dterms);
    	fprintf(stdout,"# Note that 0 means stacks are PSDs from SFTs (the only option currently supported); 1 means stacks are F-statistic from SFTs. \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set thresholdFlag      %23d; #31 INT2 how to apply the thresholds. \n", params->thresholdFlag);
    	fprintf(stdout,"set threshold1         %23.16e; #32 REAL4 peak found if power is above this. \n", params->threshold1);
    	fprintf(stdout,"set threshold2         %23.16e; #33 REAL4 peak ends if power drops below this. \n", params->threshold2);
    	fprintf(stdout,"set threshold3         %23.16e; #34 REAL4 ratio peak height to valley depth that indicates a new peak rather than subpeak in a cluster. \n", params->threshold3); /* 01/20/04 gam */
    	fprintf(stdout,"set threshold4         %23.16e; #35 REAL4 unused (except when testFlag & 2 > 0; see below).\n", params->threshold4); /* 05/11/04 gam */ /* 01/27/04 gam */
    	fprintf(stdout,"set threshold5         %23.16e; #36 REAL4 unused (except when testFlag & 1 > 0).\n", params->threshold5); /* 05/11/04 gam */ /* 01/27/04 gam */ /* 11/01/04 gam */ /* 10/20/05 gam */
    	fprintf(stdout,"#Also see below how threshold3, threhold4, and threshold5 are used when (debugOptionFlag & 64) > 0 \n");
    	fprintf(stdout,"set maxWidthBins       %23d; #37 REAL4 maximum width in bins. \n", params->maxWidthBins); /* 02/20/04 gam */ /* 02/23/04 gam */
    	fprintf(stdout,"#The thresholdFlag rules are: \n");
    	fprintf(stdout,"# if (thresholdFlag <= 0) do not analyze SUMs for peaks about threshold,\n");
    	fprintf(stdout,"# if (thresholdFlag > 0) analyze SUMs for peaks above threshold,\n");
    	fprintf(stdout,"# if (thresholdFlag & 2 > 0) find peaks above threshold with width less than maxWidthBins; else find peaks above threshold of any width,\n");
    	fprintf(stdout,"# if (thresholdFlag & 4 > 0) ignore overlap events within maxWidthBins of edge of SUM band (useful when overlapping adjacent search bands by 2*maxWidthBins),\n");
    	fprintf(stdout,"# if (thresholdFlag & 8 > 0) then update pw_mean_thissum, pw_stddev_thissum, pwr_snr ignoring peak bins.\n");
    	fprintf(stdout,"\n");
    	/* fprintf(stdout,"set calibrationFlag    %23d; #38 INT2 whether to calibrate input data (< 0 already done, 0 no, 1 yes). \n", params->calibrationFlag);
    	fprintf(stdout,"# Note that < 0 and 0 are only options currently supported. \n"); 
    	fprintf(stdout,"\n"); */ /* 12/06/04 gam */
    	fprintf(stdout,"set weightFlag         %23d; #38 INT2 how to weight STKs. \n", params->weightFlag);
    	fprintf(stdout,"#The weightFlag rules are: \n");        
    	fprintf(stdout,"# if (weightFlag & 1 > 0) use powerFlux style weights; must using running median (see normalizationFlag rules),\n");
    	fprintf(stdout,"# if (weightFlag & 2 > 0) include beam pattern F_+ in calculation of weights,\n");
    	fprintf(stdout,"# if (weightFlag & 4 > 0) include beam pattern F_x in calculation of weights,\n");
    	fprintf(stdout,"# if (weightFlag & 8 > 0) rescale the input BLK data (SFTs) with the inverse mean of the absolute value of this data, [fabs(Re(BLKData)) + fabs(Im(BLKData))]/2, to prevent dynamic range issues.\n"); /* 11/01/04 gam */
    	fprintf(stdout,"# if (weightFlag & 16 > 0) save medians and weight SFTs with inverse medians; must using running median (see normalizationFlag rules).\n"); /* 09/12/05 gam */
    	fprintf(stdout,"# This last option will reuse the medians which can speed up Monte Carlo Simulations. However one must test what bias this introduces by also running the MC with this option off.\n"); /* 09/12/05 gam */
    	fprintf(stdout,"\n");
    	/* fprintf(stdout,"set windowFilterFlag   %23d; #40 INT2 whether to window or filter data (< 0 already done, 0 no, 1 yes) \n", params->windowFilterFlag); */ /* 10/28/04 gam */
    	/* fprintf(stdout,"# Note that < 0 and 0 are only options currently supported. \n"); */ /* 10/28/04 gam */
    	fprintf(stdout,"set orientationAngle    %23.16e; #39 REAL8 orientation angle in radians used to find F_+ and F_x when used to weight STKs or if using fixed polarization for a Monte Carlo Simulation. \n", params->orientationAngle);
    	fprintf(stdout,"set cosInclinationAngle %23.16e; #40 REAL8 cosine inclination angle if using fixed value for a Monte Carlo Simulation. \n", params->cosInclinationAngle); /* 12/06/04 gam */
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set normalizationFlag  %23d; #41 INT2 what normalization to do. \n", params->normalizationFlag);
    	fprintf(stdout,"set f0NRM              %23.16e; #42 REAL8 frequency to start with when finding norms. \n", params->f0NRM);
    	fprintf(stdout,"set bandNRM            %23.16e; #43 REAL8 frequency band to use when finding norms. \n", params->bandNRM);
    	fprintf(stdout,"set nBinsPerNRM        %23d; #44 INT4 number of frequency bins to use when finding norms. \n", params->nBinsPerNRM);
    	fprintf(stdout,"set normalizationParameter %23.16e; #45 REAL4 see uses below. \n", params->normalizationParameter);
    	fprintf(stdout,"#The normalizationFlag rules are: \n");
    	fprintf(stdout,"# if (normalizationFlag & 1) > 0 normalize STKs using mean, but if (normalizationFlag & 4) > 0, running median takes precedence,\n"); /* 11/09/05 gam */
    	fprintf(stdout,"# if (normalizationFlag & 2) > 0 normalize BLKs else normalize STKs, \n");
    	fprintf(stdout,"# if (normalizationFlag & 4) > 0 normalize STKs using running median (or use medians in weights when weightFlag > 0), \n"); /* 04/14/04 gam; now implemented */ /* 10/28/04 gam */
    	fprintf(stdout,"# The running median block size is given by nBinsPerNRM.\n");
    	fprintf(stdout,"# If normalizing with the running median and normalizationParameter > 0, this is used to correct bias in the median to get the mean; otherwise LALRngMedBias is used to correct this bias.\n"); /* 07/09/04 gam */ /* 05/05/04 gam; */ /* 10/20/05 gam */
    	fprintf(stdout,"# if (normalizationFlag & 8) > 0 normalize with veto on power above normalizationParameter = max_power_allowed/mean_power.\n");
    	fprintf(stdout,"# if (normalizationFlag & 16) > 0 then output into .Sh file GPS startTime and PSD estimate for each SFT.\n"); /* 04/15/04 gam */
    	fprintf(stdout,"# if (normalizationFlag & 32) > 0 then ignore bins using info in linesAndHarmonicsFile.\n"); /* 05/14/05 gam */
    	fprintf(stdout,"# if (normalizationFlag & 64) > 0 then clean SFTs using info in linesAndHarmonicsFile before normalizing.\n"); /* 07/13/05 gam */
    	fprintf(stdout,"# Note that the (normalizationFlag & 32) > 0 and (normalizationFlag & 64) > 0 options can be set independently.\n"); /* 07/13/05 gam */
    	fprintf(stdout,"# WARNING: if searching for very young pulsars with frequencies that change significantly in less that 8 minutes due to spindown or spinup, or if gpsEpochStartTimeSec is larger than gpsStartTimeSec + duration/2, check how maxSpindownFreqShift and maxSpinupFreqShift are used in the code before cleaning SFTs.\n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set testFlag           %23d; #46 INT2 specify test case.\n", params->testFlag); /* 05/11/04 gam */
    	fprintf(stdout,"# if ((testFlag & 1) > 0) output Hough number counts instead of power; use threshold5 for Hough cutoff.\n"); /* 05/11/04 gam */
    	fprintf(stdout,"# if ((testFlag & 2) > 0) inject fake signals and run Monte Carlo Simulation; use threshold4 for h_0.\n");   /* 05/11/04 gam */
    	fprintf(stdout,"# if ((testFlag & 4) > 0) use LALComputeSkyAndZeroPsiAMResponse and LALFastGeneratePulsarSFTs instead of LALGeneratePulsarSignal and LALSignalToSFTs during Monte Carlo Simulations. See LAL inject package.\n"); /* 08/02/04 gam */
    	fprintf(stdout,"# if ((testFlag & 8) > 0) use fixed orientationAngle and cosInclinationAngle set above during Monte Carlo Simulations.\n"); /* 12/06/04 gam */
    	fprintf(stdout,"# if ((testFlag & 16) > 0) use results from prior jobs in the pipeline and report on current Monte Carlo results.\n");
    	fprintf(stdout,"# if ((testFlag & 32) > 0) break out of iterated Monte Carlo Simulation if desired confidence found;\n");
    	fprintf(stdout,"#  else will run all iterations using rescaleMCFraction to rescale injected amplitudes and linearly interpolate to find UL.\n");
    	fprintf(stdout,"# if ((testFlag & 64) > 0) search surrounding parameters space pts during Monte Carlo Simulations; else search nearest only.\n");
    	fprintf(stdout,"# if ((testFlag & 128) > 0) speed up Monte Carlo Simulations by injecting into middle of a band with nBinsPerBLK - nBinsPerSUM bins only.\n");
    	fprintf(stdout,"# The prior results must be given in the priorResultsFile set above.\n");
    	fprintf(stdout,"# The maximum number of iterations is given by maxMCinterations set above.\n");
    	fprintf(stdout,"# The maximum absolute error allowed when testing for convergence of confidence when iterating the Monte Carlo is set by maxMCErr above.\n");
    	fprintf(stdout,"# The loudest event, upper limits and confidence are reported in the searchresults_stackslidemontecarlo table in the xml file.\n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set linesAndHarmonicsFile %s; #47 CHAR* file with instrument line and harmonic spectral disturbances data.\n", params->linesAndHarmonicsFile);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set outputSUMFlag      %23d; #48 INT2 whether to output SUMs e.g., in ascii. \n", params->outputSUMFlag);
    	fprintf(stdout,"#The outputSUMFlag rules are: \n");
    	fprintf(stdout,"# if (outputSUMFlag <= 0) do not output any SUMs,\n");
    	fprintf(stdout,"# if (outputSUMFlag > 0) output ascii files with sums (if ComputeStackSlide sums compiled with ascii output enabled; see preprocessor flags),\n");
    	fprintf(stdout,"# if ((outputSUMFlag & 2) > 0) output frequency and power into ascii sum files; else output power only. \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set outputEventFlag    %23d; #49 INT2 determines xml output of process, event, and summary tables. \n", params->outputEventFlag);
    	fprintf(stdout,"set keepThisNumber     %23d; #50 INT4 how many events to keep (used when keeping loudest events). \n", params->keepThisNumber);
    	fprintf(stdout,"#The outputEventFlag rules: \n");
    	fprintf(stdout,"# if (outputEventFlag <= 0) do not output xml file,\n");
    	fprintf(stdout,"# if (outputEventFlag > 0) output xml file.\n");
    	fprintf(stdout,"# if (((outputEventFlag & 2) > 0) && (keepThisNumber > 0)) keep only keepThisNumber loudest events based on this criteria: \n");
    	fprintf(stdout,"#  keep the loudest event for every eventBandWidth = bandSUM/keepThisNumber, which is the same as, \n");
    	fprintf(stdout,"#  keep the loudest event for every nBinsPerOutputEvent = nBinsPerSUM/keepThisNumber; \n");
    	fprintf(stdout,"#  thus if keepThisNumber == 1 then we only output the loudest event; if keepThisNumber == nBinsPerSUM we output the loudest event for every bin.\n");
    	fprintf(stdout,"# if ((outputEventFlag & 4) > 0) the loudest event from each template (i.e., each sky position and set of spindown parameters) is output. \n"); /* 08/30/04 gam */
    	fprintf(stdout,"# if ((outputEventFlag & 8) > 0) and not running a Monte Carlo Simulation, write the loudest event to the priorResultsFile for use by later Monte Carlo Simulation; if running MC, produce estimated UL based on loudest event from priorResultsFile.\n"); /* 01/12/06 gam */
    	fprintf(stdout,"#  In this case the parameterMC must be set to the desired confidence, threshold4 to the first guess for the UL, and rescaleMCFraction will be used as the initial uncertainty in the UL.\n"); /* 01/12/06 gam */
    	fprintf(stdout,"# if ((outputEventFlag & 16) > 0) if running a Monte Carlo Simulation, write loudest event for each injection to xml file. (The default is not write these during an MC to save disk space.)\n"); /* 04/03/06 gam */
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set startRA            %23.16e; #51 REAL8 start right ascension in radians. \n", params->stksldSkyPatchData->startRA);
    	fprintf(stdout,"set stopRA             %23.16e; #52 REAL8 end right ascension in radians. \n", params->stksldSkyPatchData->stopRA);
    	fprintf(stdout,"set deltaRA            %23.16e; #53 REAL8 delta right ascension in radians. \n", params->stksldSkyPatchData->deltaRA);
    	fprintf(stdout,"set numRA              %23d; #54 INT4 number of right ascensions for DEC = 0.  \n", params->stksldSkyPatchData->numRA);
    	fprintf(stdout,"#Note that deltaRA >= 0.0 must be true, otherwise an error is thrown! \n");
    	fprintf(stdout,"#An error is thrown if deltaRA is not consistent with numRA and [startRA stopRA) for DEC = 0.\n");
    	fprintf(stdout,"#For each declination, deltaRA = input deltaRA/cos(DEC); numRA = ceil((stopRA-startRA)/deltaRA); deltaRA = (stopRA-startRA)/numRA is used to cover interval [startRA stopRA). \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set startDec           %23.16e; #55 REAL8 start declination in radians. \n", params->stksldSkyPatchData->startDec);
    	fprintf(stdout,"set stopDec            %23.16e; #56 REAL8 end declination in radians. \n", params->stksldSkyPatchData->stopDec);
    	fprintf(stdout,"set deltaDec           %23.16e; #57 REAL8 delta declination in radians. \n", params->stksldSkyPatchData->deltaDec);
    	fprintf(stdout,"set numDec             %23d; #58 INT4 number of declinations. \n", params->stksldSkyPatchData->numDec);
    	fprintf(stdout,"#Note that deltaDec >= 0.0 must be true, otherwise an error is thrown! \n");
    	fprintf(stdout,"#Note that numDec takes precedence; an error is thrown if this is not consistent with deltaDec and either the interval [startDec, stopDec) or [startDec, stopDec].\n");
    	fprintf(stdout,"#The DECs are generated by DEC = startDec + i*deltaDec for i = 0 to numDec - 1. \n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set numSpinDown        %23d; #59 INT4 number of spindown parameters to include in the search. \n", params->numSpinDown);
    	fprintf(stdout,"#Note when numSpinDown > 0 that derivs are generated by FDerivN = startFDerivN + i*deltaFDerivN for i = 0 to numFDerivN - 1. \n");
    	fprintf(stdout,"#In this case deltaFDerivN <= 0.0 must be true for N odd and deltaFDerivN >= 0.0 must be true for N even, otherwise an error is thrown! \n");
    	fprintf(stdout,"#Note that numFDerivN takes precedence; an error is thrown if this is not consistent with deltaFDerivN and either the interval [startFDerivN, stopFDerivN) or [startFDerivN, stopFDerivN].\n");
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set startFDeriv1       %23.16e; #60 REAL8 start 1st deriv of freq in Hz/s. \n", params->startFDeriv1);
    	fprintf(stdout,"set stopFDeriv1        %23.16e; #61 REAL8 end 1st deriv of freq in Hz/s. \n", params->stopFDeriv1);
    	fprintf(stdout,"set deltaFDeriv1       %23.16e; #62 REAL8 delta 1st deriv of freq in Hz/s. \n", params->deltaFDeriv1);
	fprintf(stdout,"set numFDeriv1         %23d; #63 INT4 number 1st derivs of freq. \n", params->numFDeriv1);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set startFDeriv2       %23.16e; #64 REAL8 start 2nd deriv of freq in Hz/s^2. \n", params->startFDeriv2);
    	fprintf(stdout,"set stopFDeriv2        %23.16e; #65 REAL8 end 2nd deriv of freq in Hz/s^2. \n", params->stopFDeriv2);
    	fprintf(stdout,"set deltaFDeriv2       %23.16e; #66 REAL8 delta 2nd deriv of freq in Hz/s^2. \n", params->deltaFDeriv2);
    	fprintf(stdout,"set numFDeriv2         %23d; #67 INT4 number 2nd derivs of freq. \n", params->numFDeriv2);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set startFDeriv3       %23.16e; #68 REAL8 start 3rd deriv of freq in Hz/s^3. \n", params->startFDeriv3);
    	fprintf(stdout,"set stopFDeriv3        %23.16e; #69 REAL8 end 3rd deriv of freq in Hz/s^3. \n", params->stopFDeriv3);
    	fprintf(stdout,"set deltaFDeriv3       %23.16e; #70 REAL8 delta 3rd deriv of freq in Hz/s^3. \n", params->deltaFDeriv3);
	fprintf(stdout,"set numFDeriv3         %23d; #71 INT4 number of 3rd derivs of freq. \n", params->numFDeriv3);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set startFDeriv4       %23.16e; #72 REAL8 start 4th deriv of freq in Hz/s^4. \n", params->startFDeriv4);
    	fprintf(stdout,"set stopFDeriv4        %23.16e; #73 REAL8 end 4th deriv of freq in Hz/s^4. \n", params->stopFDeriv4);
    	fprintf(stdout,"set deltaFDeriv4       %23.16e; #74 REAL8 delta 4th deriv of freq in Hz/s^4. \n", params->deltaFDeriv4);
	fprintf(stdout,"set numFDeriv4         %23d; #75 INT4 number of 4th derivs of freq. \n", params->numFDeriv4);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set numMCInjections    %23d; #76 INT4 number of Monte Carlo injections to do when (testFlag & 2) > 0.\n", params->numMCInjections);
    	fprintf(stdout,"set numMCRescalings    %23d; #77 INT4 number of times to rescale signal injection to run multiple Monte Carlo simulations in parallel. \n", params->numMCRescalings);
    	fprintf(stdout,"set rescaleMCFraction  %23.16e; #78 REAL8 fraction to change injected amplitude by with each rescaling. \n", params->rescaleMCFraction);
	fprintf(stdout,"set parameterMC        %23.16e; #79 REAL8 if (outputEventFlag & 8) > 0 the parameterMC must be set to the desired confidence. \n", params->parameterMC);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set sunEdatFile        %s; #80 CHAR* name of ascii file with sun ephemeris data. \n", params->sunEdatFile);
    	fprintf(stdout,"set earthEdatFile      %s; #81 CHAR* name of ascii file with earth ephemeris data. \n", params->earthEdatFile);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set sftDirectory       %s; #82 CHAR* path and pattern to match to find input data. \n", params->sftDirectory);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set outputFile         %s; #83 CHAR* name of output file (.xml is appended to this name for xml output). \n", params->outputFile);
    	fprintf(stdout,"\n");
    	fprintf(stdout,"set debugOptionFlag    %23d; #84 INT2 debugging information to print to stdout. \n", params->debugOptionFlag);
    	fprintf(stdout,"# if (debugOptionFlag & 1) > 0 then print command line arguments.\n");
    	fprintf(stdout,"# if (debugOptionFlag == 1) then print command line arguments and abort!\n"); /* 02/25/05 gam */
    	fprintf(stdout,"# if (debugOptionFlag & 2) > 0 then print table with events (isolated case only).\n");
    	fprintf(stdout,"# if (debugOptionFlag & 4) > 0 then print sky grid and spindown grid with debugging information.\n");
    	fprintf(stdout,"# if (debugOptionFlag & 8) > 0 then the STK bin with max power is set to 1, all other to 0.\n");
    	fprintf(stdout,"# if (debugOptionFlag & 16) > 0 also set to 1 one bin to either side of the bin with maxPwr.\n");
    	fprintf(stdout,"# if (debugOptionFlag & 32) > 0 print Monte Carlo Simulation results to stdout.\n");
    	fprintf(stdout,"# if (debugOptionFlag & 64) > 0 generated SUMs with an estimate of StackSlide Power for a given source.\n");
    	fprintf(stdout,"#Must also use the (weightFlag & 16) > 0 option; the (weightFlag & 8) > 0 option is recommended.\n");
    	fprintf(stdout,"#Must set threshold3 to the value of 2.0/(f_s*f_s*tBLK), where f_s is the effective sample rate of the input data.\n");
    	fprintf(stdout,"#Must set threshold4 = A_+, threhold5 = A_x, and orientationAngle = polarization angle of the source for which to estimate the power.\n");
    	fprintf(stdout,"#In addition, if (debugOptionFlag & 2) > 0, these are output to stdout for the time stamps of the input BLKs: A_+, F_+(midpoint time), A_x, F_x(midpoint time), sqrt(Sn) estimated from the running median, the bin mismatch, the SNR squared d2, and the binoffset.\n");
    	fprintf(stdout,"#The estimated power is output for each template based on the other options for sky position, spindown values, and frequency.\n");
    	fprintf(stdout,"# if (debugOptionFlag & 128) > 0 creates SUMs from the STKs without sliding (isolated case only).\n");
    	fprintf(stdout,"# if (debugOptionFlag & 256) > 0 then print the factor used to rescale BLKs to prevent dynamic range issues; see discussion of weightFlag & 8 > 0 above.\n");
    	fprintf(stdout,"# Use of the debugOptionFlag provides an easy way to validate the StackSlide code! \n");
    	fprintf(stdout,"# For fake data with a signal and no noise, run on the exact template for the signal with the \n");
    	fprintf(stdout,"# debugOptionFlag bit for 8, or 8 and 16, set. The StackSlide power should equal the number of SFTs,\n");
    	fprintf(stdout,"# to within ~4 percent, for debugOptionFlag & 8 > 0 and exactly equal this for debugOptionFlag & 24 > 0.\n");
        /* 02/25/05 gam; if params->debugOptionFlag == 1 then exit after printing parameters */
        if (params->debugOptionFlag == 1) {
           ABORT( status, DRIVESTACKSLIDEH_EUSERREQUESTEXIT, DRIVESTACKSLIDEH_MSGEUSERREQUESTEXIT);
        }
   } /* 04/15/04 gam; END if ((params->debugOptionFlag & 1) > 0 ) */
 #endif
 #ifdef DEBUG_COMPUTED_PARAMETERS
	/* fprintf(stdout,"params->deltaT = %g \n", params->deltaT); */ /* 12/06/04 gam */
	fprintf(stdout,"params->gpsStartTimeNan = %23d \n", params->gpsStartTimeNan); /* 12/06/04 gam */
	fprintf(stdout,"params->dfBLK = %g \n", params->dfBLK);
	fprintf(stdout,"params->tSTK = %g \n", params->tSTK);
	fprintf(stdout,"params->dfSTK = %g \n", params->dfSTK);
	fprintf(stdout,"params->tSUM = %g \n", params->tSUM);
	fprintf(stdout,"params->numSUMsPerParamSpacePt = %i \n", params->numSUMsPerParamSpacePt);
	fprintf(stdout,"params->dfSUM = %g \n", params->dfSUM);
	fprintf(stdout,"params->numSkyPosTotal = %i \n", params->numSkyPosTotal);
	fprintf(stdout,"params->numFreqDerivTotal = %i \n", params->numFreqDerivTotal);
	fprintf(stdout,"params->numParamSpacePts = %i \n", params->numParamSpacePts);
	fprintf(stdout,"params->numSUMsTotal = %i \n", params->numSUMsTotal);
    	fprintf(stdout,"\n");
        fflush(stdout);
 #endif

/******************************************/
/*                                        */
/* END SECTION: parameters passed as      */
/* arguments.  Those indented are         */
/* computed from these.                   */
/*                                        */
/******************************************/

/**********************************************/
/*                                            */
/* START SECTION: initialize other parameters */
/*                                            */
/**********************************************/

  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART SECTION: initialize other parameters\n");
  	fflush(stdout);
  #endif


  /* Set up name of this DSO */
  params->dsoName = NULL;
  params->dsoName = (CHAR *) LALMalloc( (strlen( "DRIVESTACKSLIDE" ) + 1) * sizeof(CHAR) );
  strcpy(params->dsoName, "DRIVESTACKSLIDE");

  /* params->searchMaster = initSearchParams->nodeClass; */
  params->rank         = 0;
  params->numSlaves    = 0;
  params->curSlaves    = 0;
  params->numNodes     = 0;
  /* params->numDBOutput  = 0; */ /* 02/11/04 gam */ /* Total number of rows going into all databases is calculated. */

  /* parameters for keeping track of which BLKs, STKs, SUMs to work on; which are done. */
  params->finishedBLKs = 0;  /* Set equal to true when all BLKS for this job have been found in input data */
  params->finishedSTKs = 0;  /* Set equal to true when all STKs for this job have been created */
  /* params->finishedSUMs = 0; */ /* 05/26/04 gam */ /* Set equal to true when all BLKS for this job have been created */
  params->startSUMs = 1;            /* 05/26/04 gam; use to control I/O during Monte Carlo. Default is TRUE. */
  params->finishPeriodicTable = 1;  /* 05/24/05 gam */ /* 05/26/04 gam; use to control I/O during Monte Carlo. Default is TRUE. */
  params->whichMCSUM = -1;          /* 05/26/04 gam; which SUM the Monte Carlo Simulation is running on. Default is -1. */

  params->maxPower = 0;        /* 05/25/05 gam; power in loudest event */
  params->totalEventCount = 0; /* 05/25/05 gam; total number of peaks found */

  params->iMinBLK = 0;                         /* Index of minimum frequency in BLK band */
  params->iMaxBLK = params->nBinsPerBLK - 1;   /* Index of maximum frequency in BLK band */

  params->edat = NULL;       /* Initialize pointer to ephemeris data */

  /* 02/28/05 gam; moved here */
  params->outputLoudestFromPeaks = 0;  /* default value */
  params->outputLoudestFromSUMs = 0;   /* default value */
  if ((params->outputEventFlag & 2) > 0) {
      if (params->thresholdFlag > 0) {
         params->outputLoudestFromPeaks = 1; /* Output loudest based on thresholds */
      } else {
         params->outputLoudestFromSUMs = 1;  /* Output loudest directly from SUMs, without thresholds */
      }
  }

  /* 02/28/05 gam; moved here */
  params->weightSTKsIncludingBeamPattern = 0;  /* default value */
  params->weightSTKsWithInverseMedians = 0;    /* 09/12/05 gam; default value */
  params->inverseMediansSaved = 0;             /* 09/12/05 gam; default value */
  params->inverseMedians = NULL;               /* 09/12/05 gam; default value */
  params->plusOrCross = 1;                     /* default value */
  if ( (params->weightFlag & 1) > 0 )  {
     if ( ( (params->weightFlag & 2) > 0 ) || ( (params->weightFlag & 4) > 0 ) ) {
          params->weightSTKsIncludingBeamPattern = 1;
          if ( (params->weightFlag & 2) > 0 ) {
             params->plusOrCross = 1; /* get F_+ */
          } else {
             params->plusOrCross = 0; /* get F_x */
          }               
     }
  } else if ( (params->weightFlag & 16) > 0 ) {
     params->weightSTKsWithInverseMedians = 1; /* 09/12/05 gam */
  }

  /* 02/28/05 gam; moved here */
  params->nBinsPerOutputEvent = 0;
  
/**********************************************/
/*                                            */
/* END SECTION: initialize other parameters   */
/*                                            */
/**********************************************/

/**********************************************/
/*                                            */
/* START SECTION: validate parameters         */
/*                                            */
/**********************************************/

  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART SECTION: validate parameters\n");
  	fflush(stdout);
  #endif

  if (params->duration <= 0.0) {
       ABORT( status, DRIVESTACKSLIDEH_EDURATION, DRIVESTACKSLIDEH_MSGEDURATION );
  }

  /* vet BKL parameters */
  if (params->numBLKs <= 0.0) {
       ABORT( status, DRIVESTACKSLIDEH_ENBLKS, DRIVESTACKSLIDEH_MSGENBLKS );
  }
  if (params->tBLK <= 0.0 || params->tBLK > params->duration || params->tEffBLK <= 0.0) {
    ABORT( status, DRIVESTACKSLIDEH_ETBLK, DRIVESTACKSLIDEH_MSGETBLK);
  }
  if ( (params->bandBLK <= 0) || (floor(params->bandBLK*params->tEffBLK + 0.5) != params->nBinsPerBLK) ) {
         ABORT( status, DRIVESTACKSLIDEH_EBANDBLK, DRIVESTACKSLIDEH_MSGEBANDBLK );
  }

  /* vet STK parameters */
  if (params->numBLKsPerSTK <= 0) {
       ABORT( status, DRIVESTACKSLIDEH_ENBLKSPERSTK, DRIVESTACKSLIDEH_MSGENBLKSPERSTK );
  }
  if (params->tEffSTK <= 0.0) {
       ABORT( status, DRIVESTACKSLIDEH_ETEFFSTK, DRIVESTACKSLIDEH_MSGETEFFSTK );
  }
  if ( (params->bandSTK <= 0) || (floor(params->bandSTK*params->tEffSTK + 0.5) != params->nBinsPerSTK) ) {
         ABORT( status, DRIVESTACKSLIDEH_EBANDSTK, DRIVESTACKSLIDEH_MSGEBANDSTK );
  }

  /* vet SUM parameters */
  if (params->numSTKsPerSUM <= 0.0) {
       ABORT( status, DRIVESTACKSLIDEH_ENSTKSPERSUM, DRIVESTACKSLIDEH_MSGENSTKSPERSUM );
  }  
  if (params->tEffSUM <= 0.0) {
       ABORT( status, DRIVESTACKSLIDEH_ETEFFSUM, DRIVESTACKSLIDEH_MSGETEFFSUM );
  }
  if ( (params->bandSUM <= 0) || (floor(params->bandSUM*params->tEffSUM + 0.5) != params->nBinsPerSUM) ) {
         ABORT( status, DRIVESTACKSLIDEH_EBANDSUM, DRIVESTACKSLIDEH_MSGEBANDSUM );
  }

  /* 07/17/05 gam; since entire frequency band slides together, band cannot exceed (c/v_Earth)_max/tEffSTK  */
  /* This keeps the maximum error in the number of bins to slide less than or equal to 0.5 bins.            */
  if ( params->bandSUM > 1.0/(params->tEffSTK*((REAL8)STACKSLIDEMAXV)) ){
    if ((params->debugOptionFlag & 128) > 0 ) {
      /* continue; sliding is turned off */
    } else {
      ABORT( status, DRIVESTACKSLIDEH_EBANDTOOWIDE, DRIVESTACKSLIDEH_MSGEBANDTOOWIDE);
    }
  }

  /* "Invalid or null ifoNickName" */
  if ( !(params->ifoNickName) ){
    ABORT( status, DRIVESTACKSLIDEH_EIFONICKNAME, DRIVESTACKSLIDEH_MSGEIFONICKNAME);
  }

  /* "Invalid or null IFO" */
  if ( !(params->IFO) ){
    ABORT( status, DRIVESTACKSLIDEH_EIFO, DRIVESTACKSLIDEH_MSGEIFO);
  }

  /* "Invalid or null Target Name" */
  if ( !(params->patchName) ){
    ABORT( status, DRIVESTACKSLIDEH_ETARGETNAME, DRIVESTACKSLIDEH_MSGETARGETNAME);
  }

  if ( params->parameterSpaceFlag < 0 ) {
    ABORT( status, DRIVESTACKSLIDEH_EPARAMSPACEFLAG, DRIVESTACKSLIDEH_MSGEPARAMSPACEFLAG );
  }

  if ( params->stackTypeFlag !=0 ) {
    /* 0 is the only currently available option */
    ABORT( status, DRIVESTACKSLIDEH_ESTKTYPEFLAG, DRIVESTACKSLIDEH_MSGESTKTYPEFLAG );
  }

  if ( params->weightFlag < 0 ) {
    ABORT( status, DRIVESTACKSLIDEH_EWEIGHTFLAG, DRIVESTACKSLIDEH_MSGEWEIGHTFLAG );
  }

  if ( params->orientationAngle < -1.0*((REAL8)LAL_TWOPI) || params->orientationAngle > ((REAL8)LAL_TWOPI) ) {
    ABORT( status, DRIVESTACKSLIDEH_EBADORIANGLE, DRIVESTACKSLIDEH_MSGEBADORIANGLE );
  }

  if ( (params->cosInclinationAngle < -1.0) || (params->cosInclinationAngle > 1.0) ) {
    ABORT( status, DRIVESTACKSLIDEH_EBADCOSINC, DRIVESTACKSLIDEH_MSGEBADCOSINC );
  }

  if ( params->normalizationFlag < 0 ) {
    ABORT( status, DRIVESTACKSLIDEH_ENORMFLAG, DRIVESTACKSLIDEH_MSGENORMFLAG );
  }

  /* Normalization of BLKs not currently supported; it is the STKs that get normalized */
  if ((params->normalizationFlag & 2) > 0) {
    ABORT( status, DRIVESTACKSLIDEH_ENORMBLKs, DRIVESTACKSLIDEH_MSGENORMBLKs );
  }

  /* Check that params->nBinsPerNRM makes sense */
  if ( (params->normalizationFlag & 4) > 0 ) {
    /* Using running median; blocksize = params->nBinsPerNRM */
    if ( (params->nBinsPerNRM <= 0) || (params->nBinsPerNRM > params->nBinsPerSTK) ) {
       ABORT( status, DRIVESTACKSLIDEH_EBANDNRM, DRIVESTACKSLIDEH_MSGEBANDNRM );
    }
  } else if ( (params->normalizationFlag & 1) > 0 ) {
    if ( (params->bandNRM <= 0) || (floor(params->bandNRM*params->tEffSTK + 0.5) != params->nBinsPerNRM) ) {
       ABORT( status, DRIVESTACKSLIDEH_EBANDNRM, DRIVESTACKSLIDEH_MSGEBANDNRM );
    }
  }

  /* Bit 8 vetoes power above the normalizationParameter when finding mean power to normalize STKs. */
  /* Cannot do this when bit 4 is set and using the running median. (Instead should use cleaning options.) */
  if ( ((params->normalizationFlag & 8) > 0) && ((params->normalizationFlag & 4) > 0) ) {
      ABORT( status, DRIVESTACKSLIDEH_ENORMBIT4AND8, DRIVESTACKSLIDEH_MSGENORMBIT4AND8 );
  }
  
  if ( params->testFlag < 0 ) {
    ABORT( status, DRIVESTACKSLIDEH_ETESTFLAG, DRIVESTACKSLIDEH_MSGETESTFLAG );
  }

  /* Cannot set weightFlag bit 16 and testFlag bit 128; these options are incompatible. */
  /* The first reuses the medians from the first injection with each injection; the     */
  /* second uses a narrow band for injections, changing this band with each injection.  */
  if ( ((params->weightFlag & 16) > 0) && ((params->testFlag & 128) > 0) ) {
      ABORT( status, DRIVESTACKSLIDEH_EBADWEIGHTTEST, DRIVESTACKSLIDEH_MSGEBADWEIGHTTEST );
  }

  /* 02/28/05 gam; move into initStackSlide */ 
  if ( ( (params->outputEventFlag & 2) <= 0 ) && (params->thresholdFlag < 1) && (params->outputEventFlag > 0)  ) {
       ABORT( status, DRIVESTACKSLIDEH_EOUTPUTREQUEST, DRIVESTACKSLIDEH_MSGEOUTPUTREQUEST ); /* Cannot output all events if thresholds are not used */
  }

  /* 2nd bit in outputEventFlag set to keep loudest but keepThisNumber was < 1! */
  if ( (params->outputLoudestFromPeaks || params->outputLoudestFromSUMs) && (params->keepThisNumber < 1) ) {
       ABORT( status, DRIVESTACKSLIDEH_EKEEPTHISNEVENTS, DRIVESTACKSLIDEH_MSGEKEEPTHISNEVENTS );
  }

  /* vet sky RA parameters */
  if (params->stksldSkyPatchData->startRA < 0 || params->stksldSkyPatchData->startRA > (REAL8)LAL_TWOPI) {
    ABORT( status, DRIVESTACKSLIDEH_ERA, DRIVESTACKSLIDEH_MSGERA);
  }
  if (params->stksldSkyPatchData->stopRA < 0 || params->stksldSkyPatchData->stopRA > (REAL8)LAL_TWOPI) {
    ABORT( status, DRIVESTACKSLIDEH_ERA, DRIVESTACKSLIDEH_MSGERA);
  }  
  if (params->stksldSkyPatchData->deltaRA < 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_EDELTARA, DRIVESTACKSLIDEH_MSGEDELTARA );
  }
  if (params->stksldSkyPatchData->numRA <= 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMRA, DRIVESTACKSLIDEH_MSGENUMRA );
  }  
  if (params->stksldSkyPatchData->deltaRA > 0.0) {
    /* numRA and deltaRA must be consistent with the interval [startRA, stopRA) for DEC = 0. For each declination,   */
    /* deltaRA = input deltaRA/cos(DEC); numRA = ceil((stopRA-startRA)/deltaRA); deltaRA = (stopRA-startRA)/numRA is used. */
    if ( floor( (params->stksldSkyPatchData->stopRA - params->stksldSkyPatchData->startRA)
           /params->stksldSkyPatchData->deltaRA + 0.5 ) != params->stksldSkyPatchData->numRA ) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMRA, DRIVESTACKSLIDEH_MSGENUMRA );
    }  
  } else {
    if ( (params->stksldSkyPatchData->stopRA != params->stksldSkyPatchData->startRA) || (params->stksldSkyPatchData->numRA != 1) ) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMRA, DRIVESTACKSLIDEH_MSGENUMRA );
    }
  }

  /* vet sky DEC parameters */
  if (params->stksldSkyPatchData->startDec < -1.0*(REAL8)LAL_PI_2 || params->stksldSkyPatchData->startDec > (REAL8)LAL_PI_2) {
    ABORT( status, DRIVESTACKSLIDEH_EDEC, DRIVESTACKSLIDEH_MSGEDEC );
  }
  if (params->stksldSkyPatchData->stopDec < -1.0*(REAL8)LAL_PI_2 || params->stksldSkyPatchData->stopDec > (REAL8)LAL_PI_2) {
    ABORT( status, DRIVESTACKSLIDEH_EDEC, DRIVESTACKSLIDEH_MSGEDEC );
  }  
  if (params->stksldSkyPatchData->deltaDec < 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_EDELTADEC, DRIVESTACKSLIDEH_MSGEDELTADEC );
  }
  if (params->stksldSkyPatchData->numDec <= 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDEC, DRIVESTACKSLIDEH_MSGENUMDEC );
  }  
  if (params->stksldSkyPatchData->deltaDec > 0.0) {
    /* Interval covered must be [startDEC, stopDEC) or [startDEC, stopDEC] */
    if ( ( floor( (params->stksldSkyPatchData->stopDec - params->stksldSkyPatchData->startDec)
           /params->stksldSkyPatchData->deltaDec + 0.5 ) != params->stksldSkyPatchData->numDec )      
         &&
         ( floor( (params->stksldSkyPatchData->stopDec - params->stksldSkyPatchData->startDec)
           /params->stksldSkyPatchData->deltaDec + 0.5 ) != (params->stksldSkyPatchData->numDec - 1) )
     ) 
    {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDEC, DRIVESTACKSLIDEH_MSGENUMDEC );
    }
  } else {
    if ( (params->stksldSkyPatchData->stopDec != params->stksldSkyPatchData->startDec) || (params->stksldSkyPatchData->numDec != 1) ) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDEC, DRIVESTACKSLIDEH_MSGENUMDEC );
    }
  }

  /* 07/17/05 gam; currently only support a maximum of 4 spindown parameters */
  if ( params->numSpinDown > 4 ) {
      ABORT( status, DRIVESTACKSLIDEH_ETOOMANYSPINDOWN, DRIVESTACKSLIDEH_MSGETOOMANYSPINDOWN);
  }

  if ( params->numSpinDown > 0 ) {
    /* vet FDeriv1 parameters; note that deltaFDeriv1 cannot be positive */
    if (params->deltaFDeriv1 > 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_EDELTADERIV1, DRIVESTACKSLIDEH_MSGEDELTADERIV1 );
    }
    if (params->numFDeriv1 <= 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV1, DRIVESTACKSLIDEH_MSGENUMDERIV1 );
    }  
    if (params->deltaFDeriv1 < 0.0) {
      /* Interval covered must be [startFDeriv1, stopFDeriv1) or [startFDeriv1, stopFDeriv1] */
      if ( ( floor( (params->stopFDeriv1 - params->startFDeriv1)
           /params->deltaFDeriv1 + 0.5 ) != params->numFDeriv1 )
         &&
         ( floor( (params->stopFDeriv1 - params->startFDeriv1)
           /params->deltaFDeriv1 + 0.5 ) != (params->numFDeriv1 - 1) )
       ) 
      {
         ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV1, DRIVESTACKSLIDEH_MSGENUMDERIV1 );
      }
    } else {
      if ( (params->stopFDeriv1 != params->startFDeriv1) || (params->numFDeriv1 != 1) ) {
        ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV1, DRIVESTACKSLIDEH_MSGENUMDERIV1 );
      }
    }
  }

  if ( params->numSpinDown > 1 ) {
    /* vet FDeriv2 parameters; note that deltaFDeriv2 cannot be negative */
    if (params->deltaFDeriv2 < 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_EDELTADERIV2, DRIVESTACKSLIDEH_MSGEDELTADERIV2 );
    }
    if (params->numFDeriv2 <= 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV2, DRIVESTACKSLIDEH_MSGENUMDERIV2 );
    }  
    if (params->deltaFDeriv2 > 0.0) {
      /* Interval covered must be [startFDeriv2, stopFDeriv2) or [startFDeriv2, stopFDeriv2] */
      if ( ( floor( (params->stopFDeriv2 - params->startFDeriv2)
           /params->deltaFDeriv2 + 0.5 ) != params->numFDeriv2 )
         &&
         ( floor( (params->stopFDeriv2 - params->startFDeriv2)
           /params->deltaFDeriv2 + 0.5 ) != (params->numFDeriv2 - 1) )
       ) 
      {
         ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV2, DRIVESTACKSLIDEH_MSGENUMDERIV2 );
      }
    } else {
      if ( (params->stopFDeriv2 != params->startFDeriv2) || (params->numFDeriv2 != 1) ) {
        ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV2, DRIVESTACKSLIDEH_MSGENUMDERIV2 );
      }
    }
  }
  
  if ( params->numSpinDown > 2 ) {
    /* vet FDeriv3 parameters; note that deltaFDeriv3 cannot be positive */
    if (params->deltaFDeriv3 > 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_EDELTADERIV3, DRIVESTACKSLIDEH_MSGEDELTADERIV3 );
    }
    if (params->numFDeriv3 <= 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV3, DRIVESTACKSLIDEH_MSGENUMDERIV3 );
    }  
    if (params->deltaFDeriv3 < 0.0) {
      /* Interval covered must be [startFDeriv3, stopFDeriv3) or [startFDeriv3, stopFDeriv3] */
      if ( ( floor( (params->stopFDeriv3 - params->startFDeriv3)
           /params->deltaFDeriv3 + 0.5 ) != params->numFDeriv3 )
         &&
         ( floor( (params->stopFDeriv3 - params->startFDeriv3)
           /params->deltaFDeriv3 + 0.5 ) != (params->numFDeriv3 - 1) )
       ) 
      {
         ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV3, DRIVESTACKSLIDEH_MSGENUMDERIV3 );
      }
    } else {
      if ( (params->stopFDeriv3 != params->startFDeriv3) || (params->numFDeriv3 != 1) ) {
        ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV3, DRIVESTACKSLIDEH_MSGENUMDERIV3 );
      }
    }
  }

  if ( params->numSpinDown > 3 ) {
    /* vet FDeriv4 parameters; note that deltaFDeriv4 cannot be negative */
    if (params->deltaFDeriv4 < 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_EDELTADERIV4, DRIVESTACKSLIDEH_MSGEDELTADERIV4 );
    }
    if (params->numFDeriv4 <= 0.0) {
      ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV4, DRIVESTACKSLIDEH_MSGENUMDERIV4 );
    }  
    if (params->deltaFDeriv4 > 0.0) {
      /* Interval covered must be [startFDeriv4, stopFDeriv4) or [startFDeriv4, stopFDeriv4] */
      if ( ( floor( (params->stopFDeriv4 - params->startFDeriv4)
           /params->deltaFDeriv4 + 0.5 ) != params->numFDeriv4 )
         &&
         ( floor( (params->stopFDeriv4 - params->startFDeriv4)
           /params->deltaFDeriv4 + 0.5 ) != (params->numFDeriv4 - 1) )
       ) 
      {
         ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV4, DRIVESTACKSLIDEH_MSGENUMDERIV4 );
      }
    } else {
      if ( (params->stopFDeriv4 != params->startFDeriv4) || (params->numFDeriv4 != 1) ) {
        ABORT( status, DRIVESTACKSLIDEH_ENUMDERIV4, DRIVESTACKSLIDEH_MSGENUMDERIV4 );
      }
    }
  }

  /* 09/16/05 gam */
  if ( (params->outputSUMFlag > 0) && (params->numSUMsTotal > 100) ) {
     ABORT( status, DRIVESTACKSLIDEH_EOUTPUTSUMS, DRIVESTACKSLIDEH_MSGEOUTPUTSUMS );
  }
  /* Note that the Monte Carlo parameters are vetting in the modules with the Monto Carlo code. */

/**********************************************/
/*                                            */
/* END SECTION: validate parameters           */
/*                                            */
/**********************************************/

/* 05/11/04 gam; Move SECTION: set up parameter space into StackSlideInitSearch; free memory in StackSlideFinalizeSearch */
/**********************************************/
/*                                            */
/* START SECTION: set up parameter space      */
/*                                            */
/**********************************************/
  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART SECTION: set up parameter space\n");
  	fflush(stdout);
  #endif
  
 params->skyPosData=(REAL8 **)LALMalloc(params->numSkyPosTotal*sizeof(REAL8 *));
 for(i=0;i<params->numSkyPosTotal;i++)
 {
        params->skyPosData[i] = (REAL8 *)LALMalloc(2*sizeof(REAL8));
 }

 if (params->numSpinDown > 0) {
   params->freqDerivData=(REAL8 **)LALMalloc(params->numFreqDerivTotal*sizeof(REAL8 *));
   for(i=0;i<params->numFreqDerivTotal;i++)
   {
        params->freqDerivData[i] = (REAL8 *)LALMalloc(params->numSpinDown*sizeof(REAL8));
   }
 }

 /* 05/13/05 gam; add options */
 if (params->parameterSpaceFlag >= 0) { 

   INT4 iFDeriv1 = 0;
   INT4 iFDeriv2 = 0;
   INT4 iFDeriv3 = 0;
   INT4 iFDeriv4 = 0;
   INT4 iFDeriv5 = 0;

   INT4 repeatFDeriv1every = 0;
   INT4 repeatFDeriv2every = 0;
   INT4 repeatFDeriv3every = 0;
   INT4 repeatFDeriv4every = 0;
   INT4 repeatFDeriv5every = 0;

   /* The third argument is 1; this assigns params->skyPosData isotropically on the sphere: */
   CountOrAssignSkyPosData(params->skyPosData,&(params->numSkyPosTotal),1,params->stksldSkyPatchData);

   for(i=0;i<params->numSpinDown;i++)
   {
        switch(i) {
	   case  0: repeatFDeriv1every = params->numFDeriv1; break;
	   case  1: repeatFDeriv2every = params->numFDeriv2*repeatFDeriv1every; break;
	   case  2: repeatFDeriv3every = params->numFDeriv3*repeatFDeriv2every; break;
	   case  3: repeatFDeriv4every = params->numFDeriv4*repeatFDeriv3every; break;
	   case  4: repeatFDeriv5every = params->numFDeriv5*repeatFDeriv4every; break;
	}
   }

   for(i=0;i<params->numFreqDerivTotal;i++)
   { 
        for(k=0;k<params->numSpinDown;k++)
	{
		if (k == 0) {
		   iFDeriv1 = i % params->numFDeriv1;
		   params->freqDerivData[i][k] = params->startFDeriv1 + iFDeriv1*params->deltaFDeriv1;
		} else if (k == 1) {
		   iFDeriv2 = floor((i % repeatFDeriv2every)/repeatFDeriv1every);
		   params->freqDerivData[i][k] = params->startFDeriv2 + iFDeriv2*params->deltaFDeriv2;
		} else if (k == 2) {
		   iFDeriv3 = floor((i % repeatFDeriv3every)/repeatFDeriv2every);
		   params->freqDerivData[i][k] = params->startFDeriv3 + iFDeriv3*params->deltaFDeriv3;
		} else if (k == 3) {
		   iFDeriv4 = floor((i % repeatFDeriv4every)/repeatFDeriv3every);
		   params->freqDerivData[i][k] = params->startFDeriv4 + iFDeriv4*params->deltaFDeriv4;
		} else if (k == 4) {
		   iFDeriv5 = floor((i % repeatFDeriv5every)/repeatFDeriv4every);
		   params->freqDerivData[i][k] = params->startFDeriv5 + iFDeriv4*params->deltaFDeriv5;
		} /* END if (k == 0) ELSE ... */
        } /* END for(k=0;k<params->numSpinDown;k++) */
   } /* END for(i=0;i<params->numFreqDerivTotal;i++) */
   
   #ifdef INCLUDE_DEBUG_SKYANDSPINDOWNGRID_CODE
    /* 09/23/05 gam */
    if ((params->debugOptionFlag & 4) > 0 ) {
      fprintf(stdout,"\nSpindown Grid Info:\n");
      fprintf(stdout,"numFreqDerivIncludingNoSpinDown = %i, numFreqDerivTotal = %i, numSpinDown = %i\n",params->numFreqDerivIncludingNoSpinDown,params->numFreqDerivTotal,params->numSpinDown);
      fflush(stdout);
      for(i=0;i<params->numFreqDerivTotal;i++) {
        for(k=0;k<params->numSpinDown;k++) {
           fprintf(stdout,"fDeriv%i = params->freqDerivData[%i][%i] = %23.16e ",(k+1),i,k,params->freqDerivData[i][k]);
           fflush(stdout);
        }
        fprintf(stdout,"\n");
      }
    }    
   #endif
 } /* END if (params->parameterSpaceFlag >= 0) */
/**********************************************/
/*                                            */
/* END SECTION: set up parameter space        */
/*                                            */
/**********************************************/

/**************************************************/
/*                                                */
/* START SECTION: open xml file; write to process */
/*                and process params tables       */
/*                                                */
/**************************************************/
                        
       /* 02/09/04 gam;  prepare to output results in xml.  Note that if outputEventFlag < 1 then do not output xml. */ 
       if (params->outputEventFlag > 0) { 
	                  CHAR *xmlFile;
            MetadataTable         proctable;
            MetadataTable         procparams;

            params->xmlStream = (LIGOLwXMLStream *) LALMalloc(sizeof(LIGOLwXMLStream));
	    xmlFile = (CHAR *) LALMalloc( (strlen(params->outputFile) + 5) * sizeof(CHAR) );
	    strcpy(xmlFile,params->outputFile);
	    strcat(xmlFile,".xml");
	    
	    /* Open xml file; also prints xml file header. */
	    LALOpenLIGOLwXMLFile (status->statusPtr,params->xmlStream,xmlFile);
	    CHECKSTATUSPTR (status);
            LALFree(xmlFile);

            /* write the process table */
	    proctable.processTable = (ProcessTable *) LALCalloc( 1, sizeof(ProcessTable) );
            /* JUST SET SOME VALUES TO 0 OR BLANK FOR NOW */
            snprintf( proctable.processTable->program, LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
            snprintf( proctable.processTable->version, LIGOMETA_VERSION_MAX, "%s", CVS_ID_STRING );
            snprintf( proctable.processTable->cvs_repository, LIGOMETA_CVS_REPOSITORY_MAX, "%s", CVS_SOURCE );
            proctable.processTable->cvs_entry_time.gpsSeconds = 0;   
            proctable.processTable->cvs_entry_time.gpsNanoSeconds = 0;
	    snprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX, "%s", BLANK );
	    proctable.processTable->is_online = 0;
	    snprintf( proctable.processTable->node, LIGOMETA_NODE_MAX, "%s", BLANK );
	    snprintf( proctable.processTable->username, LIGOMETA_USERNAME_MAX, "%s", BLANK );
	    XLALGPSTimeNow(&(proctable.processTable->start_time));
	    XLALGPSTimeNow(&(proctable.processTable->end_time));
	    proctable.processTable->jobid = 0;
            snprintf( proctable.processTable->domain, LIGOMETA_DOMAIN_MAX, "%s", BLANK );
	    proctable.processTable->unix_procid = 0;
            snprintf( proctable.processTable->ifos, LIGOMETA_IFOS_MAX, "%s", params->ifoNickName );
	    proctable.processTable->next = NULL;

            LALBeginLIGOLwXMLTable( status->statusPtr, params->xmlStream, process_table );
	    CHECKSTATUSPTR (status);
            LALWriteLIGOLwXMLTable( status->statusPtr, params->xmlStream, proctable, process_table );
	    CHECKSTATUSPTR (status);
            LALEndLIGOLwXMLTable (  status->statusPtr, params->xmlStream );
	    CHECKSTATUSPTR (status);
            LALFree( proctable.processTable );
	    
            /* write the process params table */
	    procparams.processParamsTable = (ProcessParamsTable *) LALCalloc( 1, sizeof(ProcessParamsTable) );
            snprintf( procparams.processParamsTable->program, LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
            snprintf( procparams.processParamsTable->param, LIGOMETA_PARAM_MAX, "%s", "cmd line args" );
            snprintf( procparams.processParamsTable->type, LIGOMETA_TYPE_MAX, "%s", "string" );
	    strcpy(procparams.processParamsTable->value," ");
            for (i = 1; i < argc; i++) {	    
	       strcat(procparams.processParamsTable->value,argv[i]);
	       strcat(procparams.processParamsTable->value," ");
	    }
            procparams.processParamsTable->next = NULL;
	    
            LALBeginLIGOLwXMLTable( status->statusPtr, params->xmlStream, process_params_table );
	    CHECKSTATUSPTR (status);
            LALWriteLIGOLwXMLTable( status->statusPtr, params->xmlStream, procparams, process_params_table );
	    CHECKSTATUSPTR (status);
            LALEndLIGOLwXMLTable (  status->statusPtr, params->xmlStream );
	    CHECKSTATUSPTR (status);
            LALFree( procparams.processParamsTable );
       }
/**************************************************/
/*                                                */
/* END SECTION: open xml file; write to process   */
/*                and process params tables       */
/*                                                */
/**************************************************/

  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status); 

}
/******************************************/
/*                                        */
/* END FUNCTION: StackSlideInitSearch     */
/*                                        */
/******************************************/
/********************************************/
/*                                          */
/* START FUNCTION: StackSlideConditionData  */
/*                                          */
/********************************************/
void StackSlideConditionData(
    LALStatus              *status,
    StackSlideSearchParams *params
    )
{

 INITSTATUS(status);
 ATTATCHSTATUSPTR (status);

  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART FUNCTION: StackSlideConditionData\n");
  	fflush(stdout);
  #endif

 params->searchMaster = 1; /* Used under LDAS; just set = TRUE here. */
 if (params->searchMaster) {

  /* UINT4 i = 0;  */    /* all purpose index */
  /* UINT4 k = 0;  */    /* another all purpose index */
  INT4 k = 0;      /* another all purpose index */  

/*********************************************************/
/*                                                       */
/* START SECTION: validate parameters after reading BLKs */
/*                                                       */
/*********************************************************/
    
  /* Must read in BLK data before calling StackSlideConditionData */
  if (!params->finishedBLKs) {
     ABORT( status, DRIVESTACKSLIDEH_EMISSINGBLKDATA, DRIVESTACKSLIDEH_MSGEMISSINGBLKDATA);
  }
  
  /* 12/06/04 gam */
  #ifdef DEBUG_COMPUTED_PARAMETERS
    fprintf(stdout,"params->sampleRate = %23.16e \n", params->sampleRate); 
    fprintf(stdout,"params->deltaT = %23.16e \n", params->deltaT);
  #endif
 
  if (params->deltaT <= 0.0) {
    ABORT( status, DRIVESTACKSLIDEH_EDELTAT, DRIVESTACKSLIDEH_MSGEDELTAT);
  } 

/*********************************************************/
/*                                                       */
/* END SECTION: validate parameters after reading BLKs   */
/*                                                       */
/*********************************************************/

/************************************************************************************/
/*                                                                                  */
/* START SECTION: check BLK Data for dynamic range issues and rescale if indicated. */
/*                                                                                  */
/************************************************************************************/
      /* 10/20/05 gam */
      CheckDynamicRangeAndRescaleBLKData(status->statusPtr, &(params->blkRescaleFactor), params->BLKData, params->numBLKs, params->weightFlag);
      INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
      if ((params->debugOptionFlag & 256) > 0 ) {
         fprintf(stdout, "\nAll input BLK (SFT) data has been rescaled using params->blkRescaleFactor = %23.16e\n",params->blkRescaleFactor);
         fflush(stdout);
      }
      if (params->blkRescaleFactor <= 0.0) {
              ABORT( status, DRIVESTACKSLIDEH_EBLKRESCALEFACT , DRIVESTACKSLIDEH_MSGEBLKRESCALEFACT );
      }
      /* Always rescale params->threshold4, which is guess for h_0 when running MC, but note default value for params->blkRescaleFactor is 1.0 unless (params->weightFlag & 8) > 0 */
      params->threshold4 = params->blkRescaleFactor*params->threshold4;
/************************************************************************************/
/*                                                                                  */
/* END SECTION: check BLK Data for dynamic range issues and rescale if indicated.   */
/*                                                                                  */
/************************************************************************************/

/*****************************************************/
/*                                                   */
/* START SECTION: find correction for bias in median */
/*                                                   */
/*****************************************************/  
  /* 07/09/04 gam; Use LALRngMedBias to correct bias in the median. */
  if ( (params->normalizationFlag & 4) > 0 )  {
     if (params->normalizationParameter <= 0) {
       /* 10/20/05 gam; get this value from LAL only if not entered from command line */
       REAL8 tmpBiasFactor;
       LALRngMedBias(status->statusPtr, &tmpBiasFactor, params->nBinsPerNRM);
       CHECKSTATUSPTR (status);
       params->normalizationParameter = (REAL4)tmpBiasFactor;
       /* 05/05/04 gam; If normalizing with running median use normalizationParameter to correct bias in median to get mean. */
       if (params->normalizationParameter < ((REAL4)LAL_LN2) || params->normalizationParameter > 1.0) {
        ABORT( status, DRIVESTACKSLIDEH_ENORMPARAM, DRIVESTACKSLIDEH_MSGENORMPARAM );
       }
     }
  }
/*****************************************************/
/*                                                   */
/* END SECTION: find correction for bias in median   */
/*                                                   */
/*****************************************************/  

/* 05/14/05 gam; if (params->normalizationFlag & 32) > 0 then ignore bins using info in linesAndHarmonicsFile */
/* 07/13/05 gam; if (params->normalizationFlag & 64) > 0 then clean SFTs using info in linesAndHarmonicsFile */
/*************************************************/
/*                                               */
/* START SECTION: if cleaning SFTs, read in info */
/*                about lines and harmonics      */
/*                                               */
/*************************************************/
  params->sumBinMask = NULL; /* 05/19/05 gam; params->sumBinMask == 0 if bin should be excluded from search or Monte Carlo due to cleaning */
  params->sumBinMask=(INT4 *)LALMalloc(params->nBinsPerSUM*sizeof(INT4));
  for(k=0;k<params->nBinsPerSUM;k++) {
     params->sumBinMask[k] = 1; /* always set default case: no bins are excluded! */
  }
  params->percentBinsExcluded = 0.0; /* default case; 0 percent of bins are excluded! */
  
  /* If option is set, get info about lines and harmonics */
  params->infoHarmonics = NULL;
  params->infoLines = NULL;
  /* if ( (params->normalizationFlag & 32) > 0 ) */ /* 07/13/05 gam */
  if ( ((params->normalizationFlag & 32) > 0) || ((params->normalizationFlag & 64) > 0)  )  {
    params->infoHarmonics = (LineHarmonicsInfo *)LALMalloc(sizeof(LineHarmonicsInfo));
    params->infoLines = (LineNoiseInfo *)LALMalloc(sizeof(LineNoiseInfo));

    params->infoHarmonics->nHarmonicSets = 0;
    params->infoHarmonics->startFreq = NULL;
    params->infoHarmonics->gapFreq = NULL;
    params->infoHarmonics->numHarmonics = NULL;
    params->infoHarmonics->leftWing = NULL;
    params->infoHarmonics->rightWing = NULL;
    params->infoLines->nLines = 0;
    params->infoLines->lineFreq = NULL;
    params->infoLines->leftWing = NULL;
    params->infoLines->rightWing = NULL;

    StackSlideGetLinesAndHarmonics(status->statusPtr, params->infoHarmonics, params->infoLines, params->f0BLK, params->bandBLK, params->linesAndHarmonicsFile);
    INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);

    /* 07/13/05 gam */  
    if ( (params->normalizationFlag & 32) > 0 ) {
      /* 05/19/05 gam; set up params->sumBinMask with bins to exclude from search or Monte Carlo due to cleaning */
      StackSlideGetBinMask(status->statusPtr, params->sumBinMask, &(params->percentBinsExcluded), params->infoLines,
         ((REAL8)STACKSLIDEMAXV), params->maxSpindownFreqShift, params->maxSpinupFreqShift, params->f0SUM, params->tEffSUM, params->nBinsPerSUM);
      INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
    } 

    /* 07/13/05 gam; make RandomParams *randPar a parameter for CleanCOMPLEX8SFT; initialze RandomParams *randPar once to avoid repeatly opening /dev/urandom */
    if ( (params->normalizationFlag & 64) > 0 ) {
      INT4 seed=0;
      FILE *fpRandom;
      INT4 rndCount;
      fpRandom = fopen("/dev/urandom","r");
      rndCount = fread(&seed, sizeof(INT4),1, fpRandom);
      fclose(fpRandom);
      LALCreateRandomParams(status->statusPtr, &(params->randPar), seed);
      CHECKSTATUSPTR (status);
    }
    
    #ifdef DEBUG_CLEANLINESANDHARMONICS
      for(k=0;k<params->infoHarmonics->nHarmonicSets;k++) {
        fprintf(stdout,"params->infoHarmonics->startFreq[%i] = %g\n",k,params->infoHarmonics->startFreq[k]);
        fprintf(stdout,"params->infoHarmonics->gapFreq[%i] = %g\n",k,params->infoHarmonics->gapFreq[k]);
        fprintf(stdout,"params->infoHarmonics->numHarmonics[%i] = %d\n",k,params->infoHarmonics->numHarmonics[k]);
        fprintf(stdout,"params->infoHarmonics->leftWing[%i] = %g\n",k,params->infoHarmonics->leftWing[k]);
        fprintf(stdout,"params->infoHarmonics->rightWing[%i] = %g\n",k,params->infoHarmonics->rightWing[k]);
        fflush(stdout);
      }        
      for(k=0;k<params->infoLines->nLines;k++) {
        fprintf(stdout,"params->infoLines->lineFreq[%i] = %g\n",k,params->infoLines->lineFreq[k]);
        fprintf(stdout,"params->infoLines->leftWing[%i] = %g\n",k,params->infoLines->leftWing[k]);
        fprintf(stdout,"params->infoLines->rightWing[%i] = %g\n",k,params->infoLines->rightWing[k]);
        fflush(stdout);    
      }
    #endif

  } /* END if ( ((params->normalizationFlag & 32) > 0) || ((params->normalizationFlag & 64) > 0)  ) */

  #ifdef DEBUG_SUMBINMASK
    for(k=0;k<params->nBinsPerSUM;k++) {
        fprintf(stdout,"f = %g, sumBinMask[%i] = %i\n",params->f0SUM + k/params->tEffSUM,k,params->sumBinMask[k]);
        fflush(stdout);
    }
    fprintf(stdout,"params->maxSpindownFreqShift = %g\n",params->maxSpindownFreqShift);
    fprintf(stdout,"params->maxSpinupFreqShift = %g\n",params->maxSpinupFreqShift);
    fprintf(stdout,"params->percentBinsExcluded = %g\n",params->percentBinsExcluded);
    fflush(stdout);
  #endif
/*************************************************/
/*                                               */
/* END SECTION: if cleaning SFTs, read in info   */
/*                about lines and harmonics      */
/*                                               */
/*************************************************/
   
/**********************************************/
/*                                            */
/* START SECTION: set units                   */
/*                                            */
/**********************************************/
        /* Input to code should be calibrated SFTS */
        /* HARD CODE units as strain/sqrt(Hz) */
        {
           LALUnitPair  unitPair;   /* A pair of LALUnits */
           RAT4         unitPower;  /* Rational number and demoninator minus one to compute power on a LALUnit */
           LALUnit      unitTmp;     /* Temporary place holder for units. */

           params->unitBLK = lalStrainUnit;  /* Assume input data is already calibrated */
           unitPower.numerator = -1;
           unitPower.denominatorMinusOne = 1;
           LALUnitRaise(status->statusPtr, &unitTmp, &lalHertzUnit, &unitPower );
           CHECKSTATUSPTR (status);
           unitPair.unitOne = &(params->unitBLK);
           unitPair.unitTwo = &unitTmp;
           LALUnitMultiply(status->statusPtr, &params->unitBLK, &unitPair);
           CHECKSTATUSPTR (status);
        }
/**********************************************/
/*                                            */
/* END SECTION: set units                     */
/*                                            */
/**********************************************/

/**********************************************************/
/*                                                        */
/* START SECTION: init params from BLKData and timeStamps */
/*                                                        */
/**********************************************************/
     #ifdef DEBUG_DRIVESTACKSLIDE
       fprintf(stdout, "\nSTART SECTION: init params from BLKData and timeStamps\n");
       fflush(stdout);
     #endif

     params->whichSTK = 0;  /* which STK does BLK go with; Will be needed when more that one BLK per STK  */
     params->lastWhichSTK = -1;  /* Last value of params->whichSTK does BLK go with  */
     params->numSTKs = 0; /* Initialize to 0; will count the actual number of stacks below. Not duration/tSTK if gaps are present */
     /* Loop through BLK data to find number of STKs */
     for(k=0;k<params->numBLKs;k++) { 
         /* Which STK does this BKL go with */
         params->whichSTK = (params->BLKData[k]->fft->epoch.gpsSeconds - params->gpsStartTimeSec)/params->tSTK;
         /* if (params->whichSTK > params->lastWhichSTK) */ /* 09/27/04 gam; if numBLKsPerSTK == 1 ensure 1 to 1 correspondence between BLKs and STKs */
         if ( (params->numBLKsPerSTK == 1) || (params->whichSTK != params->lastWhichSTK) ) {
                     params->numSTKs++;
         }
         #ifdef DEBUG_DRIVESTACKSLIDE
            fprintf(stdout,"Found BLK # %i, starts at %i; belongs to STK # %i; total STKs so far = %i \n",k,params->BLKData[k]->fft->epoch.gpsSeconds,params->whichSTK,params->numSTKs);
            fflush(stdout);
         #endif
         params->lastWhichSTK = params->whichSTK;
      }    
      
      /* 02/09/04 gam; use to hold times for search summary table */
      params->actualStartTime.gpsSeconds = params->timeStamps[0].gpsSeconds;
      params->actualStartTime.gpsNanoSeconds = params->timeStamps[0].gpsNanoSeconds;
      params->actualEndTime.gpsSeconds = params->timeStamps[params->numBLKs - 1].gpsSeconds + (UINT4)params->tBLK;
      params->actualEndTime.gpsNanoSeconds = params->timeStamps[params->numBLKs - 1].gpsNanoSeconds;
        
      #ifdef DEBUG_DRIVESTACKSLIDE
         fprintf(stdout,"\nCheck that additional parameters are set up OK: \n");
         fprintf(stdout,"\nparams->numSTKs = %i \n", params->numSTKs);
         fflush(stdout);
         fprintf(stdout,"\n END SECTION: init params from BLKData and timeStamps \n\n");
         fflush(stdout);
      #endif
/**********************************************************/
/*                                                        */
/* END SECTION: init params from BLKData and timeStamps   */
/*                                                        */
/**********************************************************/     

/**********************************************/
/*                                            */
/* START SECTION: unpack ephemeris data       */
/*                                            */
/**********************************************/
  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART SECTION: unpack ephemeris data\n");
  	fflush(stdout);
  #endif
  
  /* 02/11/04 gam; reorganize to use LALInitBarycenter from LAL support package */
  
  params->edat = (EphemerisData *)LALMalloc(sizeof(EphemerisData));

  /* NOTE: hard code input earth and sun file names here. */
  /* (*params->edat).ephiles.earthEphemeris = "/home/gmendell/searchcode/share/lal/earth00-04.dat";
     (*params->edat).ephiles.sunEphemeris = "/home/gmendell/searchcode/share/lal/sun00-04.dat"; */

  params->edat->ephiles.sunEphemeris = params->sunEdatFile ;
  params->edat->ephiles.earthEphemeris = params->earthEdatFile;

  #ifdef INCLUDE_INTERNALLALINITBARYCENTER	  
    /* Use internal copy of LALInitBarycenter.c from the LAL support package.  Author: Curt Cutler */
    InternalLALInitBarycenter(status->statusPtr, params->edat);
  #else
    LALInitBarycenter(status->statusPtr, params->edat);
  #endif
  INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
     
/**********************************************/
/*                                            */
/* END SECTION: unpack ephemeris data         */
/*                                            */
/**********************************************/

/************************************************/
/*                                              */
/* START SECTION: Find Earth's ave acceleration */
/*                and rotate sky coordinates to */
/*                form sky bands if requested   */
/*                                              */
/************************************************/

  /* 05/13/05 gam; Add function FindAveEarthAcc that finds aveEarthAccVec, the Earth's average acceleration vector during the analysis time. */
  /* 05/13/05 gam; Add function FindLongLatFromVec that find for a vector that points from the center to a position on a sphere, the latitude and longitude of this position */
  FindAveEarthAcc(status->statusPtr, params->aveEarthAccVec, ((REAL8)(params->gpsStartTimeSec)),((REAL8)(params->gpsStartTimeSec) + params->duration), params->edat);
  INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
  FindLongLatFromVec(status->statusPtr, &(params->aveEarthAccRA), &(params->aveEarthAccDEC), params->aveEarthAccVec);
  INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);

  if ( (params->parameterSpaceFlag & 1) >  0) {
    /* 05/13/05 gam; if (params->parameterSpaceFlag & 1) > 0 rotate skyPosData into coordinates with Earth's average acceleration at the pole. */
    RotateSkyPosData(status->statusPtr, params->skyPosData, params->numSkyPosTotal, params->aveEarthAccRA, params->aveEarthAccDEC, 0.0);
    INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
  } else if ( (params->parameterSpaceFlag & 2) >  0) {
    /* 05/13/05 gam; if (params->parameterSpaceFlag & 2) > 0 rotate skyPosData into galactic plane */
    RotateSkyPosData(status->statusPtr, params->skyPosData, params->numSkyPosTotal, 4.65,((REAL8)LAL_PI_2)-0.4, 0.0);
    INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
  }

  #ifdef DEBUG_ROTATESKYCOORDINATES
    fprintf(stdout,"params->aveEarthAccVec[0] = %23.16e \n",params->aveEarthAccVec[0]);
    fprintf(stdout,"params->aveEarthAccVec[1] = %23.16e \n",params->aveEarthAccVec[1]);
    fprintf(stdout,"params->aveEarthAccVec[2] = %23.16e \n",params->aveEarthAccVec[2]);
    fprintf(stdout,"params->aveEarthAccRA = %23.16f \n",params->aveEarthAccRA);
    fprintf(stdout,"params->aveEarthAccDEC = %23.16f \n",params->aveEarthAccDEC);
    fflush(stdout);
    for(k=0;k<params->numSkyPosTotal;k++) {
       fprintf(stdout, "params->skyPosData[%i][0] = %23.16f\n",k,params->skyPosData[k][0]);
       fprintf(stdout, "params->skyPosData[%i][1] = %23.16f\n",k,params->skyPosData[k][1]);
       fflush(stdout);
    }
  #endif

/************************************************/
/*                                              */
/* END SECTION: Find Earth's ave acceleration   */
/*                and rotate sky coordinates to */
/*                form sky bands if requested   */
/*                                              */
/************************************************/

/* 05/26/04 gam; Move writing to stackslide search summary table to StackSlideConditionData */
/***********************************************************/
/*                                                         */
/* START SECTION: write to stackslide search summary table */
/*                                                         */
/***********************************************************/

    /* 02/11/04 gam; moved here so not in loop over SUMs */
    /* 02/04/04 gam; note that if outputEventFlag < 1 then do not output xml. */ 
    if (params->outputEventFlag > 0) {
            StackSlideSFTsSearchSummaryTable *stksldSFTsSearchSummaryTable;
	    
	    /* write to the stackslide search summary table. */    
	    stksldSFTsSearchSummaryTable = (StackSlideSFTsSearchSummaryTable *) LALCalloc( 1, sizeof(StackSlideSFTsSearchSummaryTable) );
            
	    /* 02/12/04 gam; Add num_sums = numSUMsTotal, freq_index =f0SUM*params->tEffSUM, and num_bins = nBinsPerSUM to StackSlideSFTsSearchSummaryTable */
	    snprintf( stksldSFTsSearchSummaryTable->ifo, LIGOMETA_IFOS_MAX, "%s", params->ifoNickName );
	    snprintf( stksldSFTsSearchSummaryTable->data_directory, LIGOMETA_STRING_MAX, "%s", params->sftDirectory );
	    snprintf( stksldSFTsSearchSummaryTable->comment, LIGOMETA_COMMENT_MAX, "%s", params->patchName );  /* Let patchName serve as the comment */
            stksldSFTsSearchSummaryTable->start_time = params->gpsStartTimeSec;
            stksldSFTsSearchSummaryTable->start_time_ns =params->gpsStartTimeNan;
            stksldSFTsSearchSummaryTable->duration = params->duration;
            stksldSFTsSearchSummaryTable->sft_baseline = params->tEffBLK;
            stksldSFTsSearchSummaryTable->num_sfts = params->numBLKs;
            stksldSFTsSearchSummaryTable->start_freq = params->f0SUM;
            stksldSFTsSearchSummaryTable->band = params->bandSUM;
            stksldSFTsSearchSummaryTable->sum_baseline = params->tEffSUM;
            stksldSFTsSearchSummaryTable->freq_index = floor(params->f0SUM*params->tEffSUM + 0.5);
            stksldSFTsSearchSummaryTable->num_bins = params->nBinsPerSUM;
            stksldSFTsSearchSummaryTable->num_sums = params->numSUMsTotal;
            stksldSFTsSearchSummaryTable->next = NULL;

            fprintf( params->xmlStream->fp, LIGOLW_XML_LOCAL_SEARCHSUMMARY_STACKSLIDESFTS );
            params->xmlStream->first = 0;
            /* print out the row */
            fprintf( params->xmlStream->fp, LOCAL_SEARCHSUMMARY_STACKSLIDESFTS_ROW,
                stksldSFTsSearchSummaryTable->ifo,
                stksldSFTsSearchSummaryTable->data_directory,
                stksldSFTsSearchSummaryTable->comment,
                stksldSFTsSearchSummaryTable->start_time,
                stksldSFTsSearchSummaryTable->start_time_ns,
                stksldSFTsSearchSummaryTable->duration,
                stksldSFTsSearchSummaryTable->sft_baseline,
                stksldSFTsSearchSummaryTable->num_sfts,
                stksldSFTsSearchSummaryTable->start_freq,
                stksldSFTsSearchSummaryTable->band,
                stksldSFTsSearchSummaryTable->sum_baseline,
                stksldSFTsSearchSummaryTable->freq_index,
                stksldSFTsSearchSummaryTable->num_bins,
                stksldSFTsSearchSummaryTable->num_sums
           );
           /* End the table. */
           fprintf( params->xmlStream->fp, STACKSLIDE_XML_TABLE_FOOTER ); /* 04/12/05 gam */
           params->xmlStream->table = no_table;
           LALFree(stksldSFTsSearchSummaryTable);
    }

/***********************************************************/
/*                                                         */
/* END SECTION: write to stackslide search summary table   */
/*                                                         */
/***********************************************************/

 }
 /* END if (params->searchMaster) */

 CHECKSTATUSPTR (status);
 DETATCHSTATUSPTR (status);
}
/********************************************/
/*                                          */
/* END FUNCTION: StackSlideConditionData    */
/*                                          */
/********************************************/

/******************************************/
/*                                        */
/* START FUNCTION: StackSlideApplySearch  */
/*                                        */
/******************************************/
void StackSlideApplySearch(
    LALStatus              *status,
    StackSlideSearchParams *params
    )
{

  INT4 i = 0; /* all purpose index */
  INT4 k = 0; /* another all purpose index */

  INITSTATUS(status);
  ATTATCHSTATUSPTR(status);

  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART FUNCTION: StackSlideApplySearch\n");
  	fflush(stdout);
  #endif

  /* params->searchMaster is used in LDAS; set to TRUE in this code. */
  if (params->searchMaster) {

    StackSlideParams *stksldParams;  
    LALFindStackSlidePeakParams *pLALFindStackSlidePeakParams;
    LALFindStackSlidePeakOutputs *pLALFindStackSlidePeakOutputs; /* 03/02/04 gam; this and next 5 lines are LALFindStackSlidePeaks outputs */
    REAL4 pwMeanWithoutPeaks;
    REAL4 pwStdDevWithoutPeaks;
    INT4  binsWithoutPeaks;
    INT4  acceptedEventCount;
    INT4  rejectedEventCount;    
    LALUpdateLoudestStackSlideParams *pLALUpdateLoudestStackSlideParams; /* 02/17/04 gam */
    INT4  istkNRM = 0;    /* 03/03/04 gam */
    REAL4 stkNRM = 0.0;   /* 03/03/04 gam */
    INT4 nrmBinCount = 0; /* 03/03/04 gam */
    FILE *fpPSD; /* 04/15/04 gam */
    LALDetector cachedDetector;
    SnglStackSlidePeriodicTable *loudestPeaksArray;  /* 02/17/04 gam; keep track of the loudest events */

/***********************************************************/
/*                                                         */
/* START SECTION: clean SFTs if option set and lines exist */
/*                                                         */
/***********************************************************/
  /* if ( (params->normalizationFlag & 32) > 0 ) */ /* 07/13/05 gam */
  if ( (params->normalizationFlag & 64) > 0 )  {  
    if (params->infoLines->nLines > 0) {
      /* 05/14/05 gam; cleans SFTs using CleanCOMPLEX8SFT by Sintes, A.M., Krishnan, B. */ /* 07/13/05 gam; add params->randPar */
      /* 07/13/05 gam NOTE THAT THE MAXIMUM NUMBER OF BINS TO CLEAN IS NOW params->nBinsPerBLK, i.e., COULD CLEAN ENTIRE BAND! */
      StackSlideCleanSFTs(status->statusPtr,params->BLKData,params->infoLines,params->numBLKs,params->nBinsPerNRM,params->nBinsPerBLK,params->randPar);
      INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
    }
  }
/***********************************************************/
/*                                                         */
/* END SECTION: clean SFTs if option set and lines exist   */
/*                                                         */
/***********************************************************/

/**********************************************/
/*                                            */
/* START SECTION: normalize BLKs              */
/*                                            */
/**********************************************/
   #ifdef DEBUG_DRIVESTACKSLIDE
     fprintf(stdout, "\nSTART SECTION: normalize BLKs\n");
     fflush(stdout);
   #endif
   /* if (params->normalizationFlag == 1) */ /* 03/01/04 gam; normalize STKs not BLKs unless second bit is set. */
   if ((params->normalizationFlag & 2) > 0) {
        /* Input data is SFTs */

        /* UINT4 iNRM = 0; */
        INT4 iNRM = 0;	
        REAL4 pwSumNRM= 0.0;
        REAL4 blkNRM = 1.0; /* Default value: does not change BLKs. */

        params->iMinNRM = floor((params->f0NRM - params->f0BLK)*params->tBLK + 0.5);    /* Index of mimimum frequency to include when normalizing BLKs */
        params->iMaxNRM = params->iMinNRM + params->nBinsPerNRM - 1;  /* Index of maximum frequency to include when normalizing BLKs */

        #ifdef DEBUG_NORMALIZEBLKS
               fprintf(stdout, "Normalizing BLKData:  params->iMinNRM, params->iMaxNRM = %i, %i\n",params->iMinNRM,params->iMaxNRM);
               fflush(stdout);
        #endif

	for(k=0;k<params->numBLKs;k++) {
            pwSumNRM= 0.0;

	    for(iNRM=params->iMinNRM;iNRM<=params->iMaxNRM;iNRM++)
            {
	        pwSumNRM += params->BLKData[k]->fft->data->data[iNRM].re*params->BLKData[k]->fft->data->data[iNRM].re
		       + params->BLKData[k]->fft->data->data[iNRM].im*params->BLKData[k]->fft->data->data[iNRM].im;
            }
            blkNRM = sqrt(pwSumNRM/((REAL4)(params->nBinsPerNRM))); /* Note NO factor of 2; makes mean BLK power = 1, but NRM is not the one-sided power spectral density */

	    #ifdef DEBUG_NORMALIZEBLKS
                  fprintf(stdout, "Normalizing BLKData[%i], with MeanSpectralAmplitude = %g \n",k, blkNRM);
                  fflush(stdout);
            #endif

	    if (blkNRM > 0.0) {
		blkNRM = 1.0/blkNRM; /* Divide just once; then multiply below. */
		for(i=0;i<params->nBinsPerBLK;i++)
		{
			params->BLKData[k]->fft->data->data[i].re = blkNRM*params->BLKData[k]->fft->data->data[i].re;
			params->BLKData[k]->fft->data->data[i].im = blkNRM*params->BLKData[k]->fft->data->data[i].im;
		}
            } else {
	        /* BLK could be all zeros; just continue */
	    }
	} /* END for(k=0;k<params->numBLKs;k++) */
        params->unitBLK = lalDimensionlessUnit; /* BLKData is now dimensionless */
   } /* END if ((params->normalizationFlag & 2) > 0) */
/**********************************************/
/*                                            */
/* END SECTION: normalize BLKs                */
/*                                            */
/**********************************************/

/* 05/11/04 gam; SECTION: set up parameter space was here; moved into StackSlideInitSearch */ 

/**********************************************/
/*                                            */
/* START SECTION: make STKs                   */
/*                                            */
/**********************************************/

  #ifdef DEBUG_DRIVESTACKSLIDE
  	fprintf(stdout, "\nSTART SECTION: make STKs\n");
  	fflush(stdout);
  #endif

 /* Allocate memory for the STKData structure */
 params->STKData=(REAL4FrequencySeries **)LALMalloc(params->numSTKs*sizeof(REAL4FrequencySeries *));
 for(i=0;i<params->numSTKs;i++)
 {
	params->STKData[i]=(REAL4FrequencySeries *)LALMalloc(sizeof(REAL4FrequencySeries));
	params->STKData[i]->data=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
	params->STKData[i]->data->data=(REAL4 *)LALMalloc(params->nBinsPerSTK*sizeof(REAL4));
 }

 /* 10/28/04 gam; allocate memory for parameters needed if weighting of STKs is done */
 if ( (params->weightFlag & 1) > 0 )  {    
    /* 10/28/04 gam; params->inverseSquareMedians is a container with inverse square medians for each STK for each frequency bin; for use with powerFlux style weighting of STKs */
    /* 10/28/04 gam; params->sumInverseSquareMedians is container with sum of inverse square medians for each frequency bin; for use with powerFlux style weighting of STKs. */    
    params->inverseSquareMedians=(REAL4Vector **)LALMalloc(params->numSTKs*sizeof(REAL4Vector *));
    for(i=0;i<params->numSTKs;i++) {
        params->inverseSquareMedians[i]=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
        params->inverseSquareMedians[i]->data=(REAL4 *)LALMalloc(params->nBinsPerSTK*sizeof(REAL4));
    }
    params->sumInverseSquareMedians=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
    params->sumInverseSquareMedians->data=(REAL4 *)LALMalloc(params->nBinsPerSTK*sizeof(REAL4));
    for(i=0;i<params->nBinsPerSTK;i++) {
        params->sumInverseSquareMedians->data[i] = 0.0; /* initialize */
    }
    if ( ( (params->weightFlag & 2) > 0 ) || ( (params->weightFlag & 4) > 0 ) ) {
       /* 10/28/04 gam; savSTKDATA is for reuse with powerFlux style weighting of STKs for each sky position */     
       params->savSTKData=(REAL4Vector **)LALMalloc(params->numSTKs*sizeof(REAL4Vector *));    
       for(i=0;i<params->numSTKs;i++) {
           params->savSTKData[i]=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
           params->savSTKData[i]->data=(REAL4 *)LALMalloc(params->nBinsPerSTK*sizeof(REAL4));
       }
       /* 10/28/04 gam; container for squared detector response F_+^2 or F_x^2 for one sky position, one polarization angle, for midpoints of a timeStamps */
       params->detResponseTStampMidPts=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
       params->detResponseTStampMidPts->data=(REAL4 *)LALMalloc(params->numSTKs*sizeof(REAL4));
    } else {
       params->savSTKData = NULL;
       params->detResponseTStampMidPts = NULL;
    }
 } else { 
    if ( params->weightSTKsWithInverseMedians && !(params->inverseMediansSaved) ) {
       /* 09/12/05 gam; allocate memory for params->inverseMedians; was set to NULL in StackSlideInit function */
       params->inverseMedians=(REAL4Vector **)LALMalloc(params->numSTKs*sizeof(REAL4Vector *));
       for(i=0;i<params->numSTKs;i++) {
          params->inverseMedians[i]=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
          params->inverseMedians[i]->data=(REAL4 *)LALMalloc(params->nBinsPerSTK*sizeof(REAL4));
       }
    }
    /* not saving square medians */
    params->savSTKData = NULL;
    params->detResponseTStampMidPts = NULL;
    params->inverseSquareMedians = NULL;
    params->sumInverseSquareMedians = NULL;
 }

 if (params->stackTypeFlag == 0) {

    /************************************************/
    /*                                              */
    /* START SUBSECTION: Stacks are PSDs from SFTs  */
    /*                                              */
    /************************************************/

        /* UINT4 kSTK = -1; */ /* index give which STK  */
        /* UINT4 iSTK = 0;  */ /* index gives which STK bin */       
	INT4 kSTK = -1;  /* index give which STK  */
        INT4 iSTK = 0;  /* index gives which STK bin */

        #ifdef DEBUG_DRIVESTACKSLIDE
  	    fprintf(stdout, "\nSTART SUBSECTION: Stacks are PSDs from SFTs\n");
  	    fflush(stdout);
        #endif

        params->whichSTK = 0;  /* which STK does BLK go with  */
        params->lastWhichSTK = -1;  /* Last value of params->whichSTK does BLK go with  */

	params->iMinBLK = floor((params->f0STK-params->f0BLK)*params->tBLK + 0.5); /* Index of mimimum frequency to include when making STKs from BLKs */
        params->iMaxBLK = params->iMinBLK + params->nBinsPerSTK - 1;               /* Index of maximum frequency to include when making STKs from BLKs */

	for(k=0;k<params->numBLKs;k++) {

	    /* Which STK does this BKL go with */
	    params->whichSTK = (params->BLKData[k]->fft->epoch.gpsSeconds - params->gpsStartTimeSec)/params->tSTK;
	    /* if (params->whichSTK > params->lastWhichSTK) */ /* 09/27/04 gam; if numBLKsPerSTK == 1 ensure 1 to 1 correspondence between BLKs and STKs */
            if ( (params->numBLKsPerSTK == 1) || (params->whichSTK != params->lastWhichSTK) ) {
                kSTK++; /* increment which stack we are building */
                /* 09/27/04 gam; initialize STK here: */
                if (params->numBLKsPerSTK == 1) {
                  params->STKData[kSTK]->epoch.gpsSeconds = params->BLKData[k]->fft->epoch.gpsSeconds;
                  params->STKData[kSTK]->epoch.gpsNanoSeconds = params->BLKData[k]->fft->epoch.gpsNanoSeconds;
                } else {
                  params->STKData[kSTK]->epoch.gpsSeconds = params->gpsStartTimeSec + params->whichSTK*params->tSTK;
                  params->STKData[kSTK]->epoch.gpsNanoSeconds = params->gpsStartTimeNan;
                }
                params->STKData[kSTK]->f0=params->f0STK;
                params->STKData[kSTK]->deltaF=params->dfSTK;
                params->STKData[kSTK]->data->length=params->nBinsPerSTK;
	    }
	    for(i=params->iMinBLK; i <= params->iMaxBLK; i++)
            {
                iSTK = i - params->iMinBLK;
                params->STKData[kSTK]->data->data[iSTK] = params->BLKData[k]->fft->data->data[i].re*params->BLKData[k]->fft->data->data[i].re
                        + params->BLKData[k]->fft->data->data[i].im*params->BLKData[k]->fft->data->data[i].im;
                /* 09/27/04 gam; initialization of STK should occur above */
	        /* if (params->whichSTK > params->lastWhichSTK) {
	           params->STKData[kSTK]->data->data[iSTK] = params->BLKData[k]->fft->data->data[i].re*params->BLKData[k]->fft->data->data[i].re
		       + params->BLKData[k]->fft->data->data[i].im*params->BLKData[k]->fft->data->data[i].im;
                   params->STKData[kSTK]->epoch.gpsSeconds = params->gpsStartTimeSec + params->whichSTK*params->tSTK;
		   params->STKData[kSTK]->epoch.gpsNanoSeconds = params->gpsStartTimeNan;
		   params->STKData[kSTK]->f0=params->f0STK;
		   params->STKData[kSTK]->deltaF=params->dfSTK;
		   params->STKData[kSTK]->data->length=params->nBinsPerSTK;
	        } else {
	           params->STKData[kSTK]->data->data[iSTK] += params->BLKData[k]->fft->data->data[i].re*params->BLKData[k]->fft->data->data[i].re
		       + params->BLKData[k]->fft->data->data[i].im*params->BLKData[k]->fft->data->data[i].im;
		} */
            }
            params->lastWhichSTK = params->whichSTK;

	} /* END for(k=0;k<params->numBLKs;k++) */

        /* 11/01/04 gam; if (params->weightFlag & 8) > 0 rescale STKs with threshold5 to prevent dynamic range issues. */
        /* if ( (params->weightFlag & 8) > 0 )  {
           RescaleSTKData(params->STKData, params->numSTKs, params->nBinsPerSTK, params->threshold5);
        } */ /* 10/20/05 gam */

        /* 04/15/04 gam; open file for output of PSD estimates for each SFT */
        if ( ((params->normalizationFlag & 16) > 0) && !((params->normalizationFlag & 2) > 0) ) {
            CHAR *psdFile;
            psdFile = (CHAR *) LALMalloc( (strlen(params->outputFile) + 4) * sizeof(CHAR) );
            strcpy(psdFile,params->outputFile);
            strcat(psdFile,".Sh");    
            /* Open file for PSD estimates */
            fpPSD = fopen(psdFile, "w");
            LALFree(psdFile);
        }

        /* 03/01/04 gam; Add option to normalizationFlag to normalize STKs rather than BLKs; this should be the default action. */
        /* if ( (params->normalizationFlag > 0) && !((params->normalizationFlag & 2) > 0) ) */ /* 11/09/05 gam */
        if ( ( ((params->normalizationFlag & 1) > 0) || ((params->normalizationFlag & 4) > 0) ) && !((params->normalizationFlag & 2) > 0) ) {

         /* INT4  istkNRM = 0;  */ /* 03/03/04 gam; moved above */
         /* REAL4 stkNRM = 0.0; */ /* 03/03/04 gam; moved above */
           
         if ( (params->normalizationFlag & 4) > 0 )  {
           /* 04/14/04 gam; if (params->normalizationFlag & 4) > 0 normalize STKs using running median */
           REAL4Vector *medians=NULL;
           UINT4 mediansLength;
           INT4 mediansLengthm1;
           INT4 mediansOffset1;
           INT4 mediansOffset2;
           LALRunningMedianPar runningMedianParams;

           runningMedianParams.blocksize = (UINT4)params->nBinsPerNRM;  /* block size used in running median code */
           mediansOffset1 = params->nBinsPerNRM/2 - 1;
           mediansOffset2 = mediansOffset1 + params->nBinsPerSTK - params->nBinsPerNRM;
           mediansLength = ((UINT4)params->nBinsPerSTK) - runningMedianParams.blocksize + 1; /* length of the medians vector */
           mediansLengthm1 = ((INT4)mediansLength) - 1;
           LALSCreateVector(status->statusPtr, &medians, mediansLength);
           CHECKSTATUSPTR (status);
   
           #ifdef DEBUG_NORMALIZEBLKS
               fprintf(stdout, "Normalizing STKs with running median; blocksize = %i,\n",runningMedianParams.blocksize);
               fprintf(stdout, "Normalizing STKs with running median; mediansOffset1 = %i,\n",mediansOffset1);
               fprintf(stdout, "Normalizing STKs with running median; mediansOffset2 = %i,\n",mediansOffset2);
               fprintf(stdout, "Normalizing STKs with running median; mediansLength = %i,\n",mediansLength);
               fprintf(stdout, "Normalizing STKs with running median; mediansLengthm1 = %i,\n",mediansLengthm1);
               fprintf(stdout, "Correction to bias in median set to params->normalizationParameter = %23.16e \n",params->normalizationParameter);
               fflush(stdout);
           #endif
           
           /* 05/05/04 gam; If normalizing with running median use normalizationParameter to correct bias in median to get mean. */
           /* if (params->normalizationParameter < ((REAL4)LAL_LN2) || params->normalizationParameter > 1.0) {
              ABORT( status, DRIVESTACKSLIDEH_ENORMPARAM, DRIVESTACKSLIDEH_MSGENORMPARAM );
           } */ /* 07/09/04 gam; moved into StackSlideConditionData */
              
           /* For each STK, call LALSRunningMedian; use medians to normalize the STK. */
           for(k=0;k<params->numSTKs;k++) {
              /* LALSRunningMedian(status->statusPtr, medians, params->STKData[k]->data, runningMedianParams); */
              LALSRunningMedian2(status->statusPtr, medians, params->STKData[k]->data, runningMedianParams); /* 12/18/04 gam */
              INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);

              /* 05/05/04 gam; If normalizing with running median use normalizationParameter to correct bias in median to get mean. */
              for(istkNRM=0;istkNRM<mediansLength;istkNRM++) {
                  medians->data[istkNRM] = medians->data[istkNRM]/params->normalizationParameter;
              }
        
              /* 04/15/04 gam */
              if ( (params->normalizationFlag & 16) > 0 )  {      
                /* Output into .Sh file GPS startTime and PSD estimate for each SFT. */
                stkNRM = 0.0;
                for(istkNRM=0;istkNRM<mediansLength;istkNRM++) {
                  stkNRM += medians->data[istkNRM];
                }
                stkNRM = stkNRM/((REAL4)mediansLength);
                /* fprintf(fpPSD,"%d %22.16e \n",params->STKData[k]->epoch.gpsSeconds,2.0*stkNRM/((REAL4)LAL_LN2)); */
                fprintf(fpPSD,"%d %22.16e \n",params->STKData[k]->epoch.gpsSeconds,2.0*stkNRM);
                #ifdef DEBUG_NORMALIZEBLKS
                  fprintf(stdout, "Normalizing STKData[%i], with running median; average median = %g \n",k, stkNRM);
                  fflush(stdout);
                #endif
              } else {
                #ifdef DEBUG_NORMALIZEBLKS
                  fprintf(stdout, "Normalizing STKData[%i], with running median; typical median = %g \n",k,medians->data[mediansLengthm1/2]);
                  fflush(stdout);
                #endif
              }

              /* 10/28/04 gam; if weightFlag set save inverse square medians for later weighting of STKs; else normalize now with medians */
              if ( (params->weightFlag & 1) > 0 )  {
                 /* Call function that save inverse square medians for weighting STKs */
                 SaveInverseSquareMedians(params->inverseSquareMedians, params->sumInverseSquareMedians, params->weightFlag, medians, k, mediansOffset1, mediansOffset2, mediansLengthm1, params->nBinsPerSTK);
              } else if ( params->weightSTKsWithInverseMedians ) {
                /* 09/12/05 gam; save params->inverseMedians first time through;
                else do nothing here; will weight STKs with inverseMedians below. */
                if ( !(params->inverseMediansSaved) ) {
                  SaveInverseMedians(params->inverseMedians, medians, k, mediansOffset1, mediansOffset2, mediansLengthm1, params->nBinsPerSTK);
                  if ( k==(params->numSTKs-1) ) {
                     params->inverseMediansSaved = 1;  /* All inverse medians have been saved */
                  }
                } 
              } else {              
                 /* Note NO factor of 2; makes mean STK power = 1, but NRM is not the one-sided power spectral density */
                 for(istkNRM=0;istkNRM<mediansOffset1;istkNRM++) {
                     params->STKData[k]->data->data[istkNRM] = params->STKData[k]->data->data[istkNRM]/medians->data[0];
                 }

                 for(istkNRM=mediansOffset1;istkNRM<mediansOffset2;istkNRM++) {
                     params->STKData[k]->data->data[istkNRM] = params->STKData[k]->data->data[istkNRM]/medians->data[istkNRM-mediansOffset1];
                 }

                 for(istkNRM=mediansOffset2;istkNRM<params->nBinsPerSTK;istkNRM++) {
                     params->STKData[k]->data->data[istkNRM] = params->STKData[k]->data->data[istkNRM]/medians->data[mediansLengthm1];
                 }              
              } /* END if ( (params->weightFlag & 1) > 0 )  */
      
           } /* END for(k=0;k<params->numSTKs;k++) */

           LALSDestroyVector(status->statusPtr, &medians);
           CHECKSTATUSPTR (status);

         } else {
           /* 04/14/04 gam; if (params->normalizationFlag & 4) > 0 normalize STKs using running median; else do as before: */

           params->iMinNRM = floor((params->f0NRM - params->f0STK)*params->tEffSTK + 0.5);    /* Index of mimimum frequency to include when normalizing BLKs */
           params->iMaxNRM = params->iMinNRM + params->nBinsPerNRM - 1;  /* Index of maximum frequency to include when normalizing BLKs */

           #ifdef DEBUG_NORMALIZEBLKS
               fprintf(stdout, "Normalizing STKs not BLKs:  params->iMinNRM, params->iMaxNRM = %i, %i\n",params->iMinNRM,params->iMaxNRM);
               fflush(stdout);
           #endif

           for(k=0;k<params->numSTKs;k++) {
              stkNRM = 0.0;

              for(istkNRM=params->iMinNRM;istkNRM<=params->iMaxNRM;istkNRM++) {
                   stkNRM += params->STKData[k]->data->data[istkNRM];
              }
              stkNRM = stkNRM/((REAL4)(params->nBinsPerNRM)); /* Note NO factor of 2; makes mean BLK power = 1, but NRM is not the one-sided power spectral density */

              /* 04/15/04 gam; output to file estimates of PSD for each SFT */
              if ( ((params->normalizationFlag & 16) > 0) && !((params->normalizationFlag & 8) > 0) ) {
                fprintf(fpPSD,"%d %22.16e \n",params->STKData[k]->epoch.gpsSeconds,2.0*stkNRM);
              }
      
              #ifdef DEBUG_NORMALIZEBLKS
                  fprintf(stdout, "Normalizing STKData[%i], with Mean = %g \n",k, stkNRM);
                  fflush(stdout);
              #endif

              if (stkNRM > 0.0) {
                  stkNRM = 1.0/stkNRM; /* Divide just once; then multiply below. */
                  for(i=0;i<params->nBinsPerSTK;i++) {
                    params->STKData[k]->data->data[i] = stkNRM*params->STKData[k]->data->data[i];
                  }  
              } else {
                  /* STK could be all zeros; just continue */
              }
           } /* END for(k=0;k<params->numSTKs;k++) */
         } /* 04/14/04 gam; END if ( (params->normalizationFlag & 4) > 0 ) */
         params->unitBLK = lalDimensionlessUnit; /* BLKData is now effectively dimensionless; really this is the dimension of STKs */
        }/* END if ( (params->normalizationFlag > 0) && !((params->normalizationFlag & 2) > 0) ) */

        /* 03/03/04 gam; if (params->normalizationFlag & 8) > 0 normalize STKs with veto on power above normalizationThreshold = max_power_allowed/mean_power */
        /* if ((params->normalizationFlag & 8) > 0) */
        /* 04/14/04 gam; Do not apply cutoff if using running median: */
        if ( ((params->normalizationFlag & 8) > 0) && !((params->normalizationFlag & 4) > 0) ) {
           for(k=0;k<params->numSTKs;k++) {
              stkNRM = 0.0;
              nrmBinCount = 0;
              for(istkNRM=params->iMinNRM;istkNRM<=params->iMaxNRM;istkNRM++) {
                 if (params->STKData[k]->data->data[istkNRM] > params->normalizationParameter) {
                    /* veto this bin from the norm */
                 } else {
                     nrmBinCount++;
                     stkNRM += params->STKData[k]->data->data[istkNRM];
                 }
              }
              if (nrmBinCount > 0) {
                 stkNRM = stkNRM/((REAL4)(nrmBinCount)); /* Adjusted norm; Note NO factor of 2 as above. */
              } else {
                 stkNRM = 0.0;   /* Will just continue below */
              }
              
              /* 04/15/04 gam; output to file estimates of PSD for each SFT */
              if ( (params->normalizationFlag & 16) > 0 )  {
                fprintf(fpPSD,"%d %22.16e \n",params->STKData[k]->epoch.gpsSeconds,2.0*stkNRM);
              }

              #ifdef DEBUG_NORMALIZEBLKS
                  fprintf(stdout, "Adjusted mean for STKData[%i] = %g \n",k, stkNRM);
                  fflush(stdout);
              #endif

              if (stkNRM > 0.0) {
                  stkNRM = 1.0/stkNRM; /* Divide just once; then multiply below. */
                  for(i=0;i<params->nBinsPerSTK;i++) {
                    params->STKData[k]->data->data[i] = stkNRM*params->STKData[k]->data->data[i];
                  }
              } else {
                  /* STK could be all zeros; just continue */
              }
           } /* END for(k=0;k<params->numSTKs;k++) */
        } /* END if ((params->normalizationFlag & 8) > 0) */

        /* 04/15/04 gam; close file for output of PSD estimates for each SFT */
        if ( ((params->normalizationFlag & 16) > 0) && !((params->normalizationFlag & 2) > 0) ) {
            fclose(fpPSD);
        }
        
        /* 03/03/04 gam; If params->testFlag == 1 output the Hough number count instead of power using threshold5 as the cutoff after normalizing. */
        /* if (params->testFlag == 1) */ /* 05/11/04 gam; If (params->testFlag & 1) > 0 output the Hough number count */
        if ( (params->testFlag & 1) > 0 ) {
           #ifdef DEBUG_TESTCASE
               fprintf(stdout, "Using threshold5 = %g as cutoff to output Hough number count! \n",params->threshold5);
               fflush(stdout);
           #endif
           for(k=0;k<params->numSTKs;k++) {
              for(i=0;i<params->nBinsPerSTK;i++) {
                 if (params->STKData[k]->data->data[i] > params->threshold5) {
                    params->STKData[k]->data->data[i] = 1; /* set this bin = 1 */
                    /* fprintf(stdout, "Set bin to 1 for STK %i bin %i; params->STKData[k]->data->data[i] = %g\n",k,i,params->STKData[k]->data->data[i]); */
                 } else {
                    params->STKData[k]->data->data[i] = 0; /* set this bin = 0 */
                    /* fprintf(stdout, "Set bin to 0 for STK %i bin %i; params->STKData[k]->data->data[i] = %g\n",k,i,params->STKData[k]->data->data[i]); */
                 }
              }
           } /* END for(k=0;k<params->numSTKs;k++) */
        } /* END if ( (params->testFlag & 1) > 0 ) */

        /* 04/12/05 gam; if ((params->debugOptionFlag & 8) > 0 ) find maxPwr each SFT, replace bin with 1, all other bins with 0 */
        if ((params->debugOptionFlag & 8) > 0 ) {
           /* The STK bin with maximum power is replaced with 1, all
              other bins are set to zero. For an injected signal into
              the middle of the SUM band the output should be the 
              the number of SFTs if StackSlide is working properly. */
           INT4 iPwrMax = 0;
           REAL4 pwrMax = -1.0;
           for(k=0;k<params->numSTKs;k++) {
              /* Find the bin with maximum power */
              iPwrMax = 0;
              pwrMax = -1.0;
              for(i=0;i<params->nBinsPerSTK;i++) {
                if (params->STKData[k]->data->data[i] > pwrMax) {
                  iPwrMax = i;
                  pwrMax = params->STKData[k]->data->data[i];
                }
              }
              for(i=0;i<params->nBinsPerSTK;i++) {
                 if (i == iPwrMax) {
                    params->STKData[k]->data->data[i] = 1;
                 } else if (i == (iPwrMax-1)) {
                    if ((params->debugOptionFlag & 16) > 0 ) {
                       params->STKData[k]->data->data[i] = 1;
                    } else {
                       params->STKData[k]->data->data[i] = 0;
                    }
                 } else if (i == (iPwrMax+1)) {
                    if ((params->debugOptionFlag & 16) > 0 ) {
                       params->STKData[k]->data->data[i] = 1;
                    } else {
                       params->STKData[k]->data->data[i] = 0;
                    }
                 } else {
                    params->STKData[k]->data->data[i] = 0;
                 }
              }
           } /* END for(k=0;k<params->numSTKs;k++) */
        } /* END if ((params->debugOptionFlag & 8) > 0 ) */
        
        params->finishedSTKs = 1;  /* Set equal to true when all STKs for this job have been created */

    /************************************************/
    /*                                              */
    /* END SUBSECTION: Stacks are PSDs from SFTs    */
    /*                                              */
    /************************************************/

 } else if (params->stackTypeFlag == 1) {

    /**************************************************/
    /*                                                */
    /* START SUBSECTION: Stacks are F-stat from SFTs  */
    /*                                                */
    /**************************************************/
    
    /* NOT YET IMPLEMENTED*/
    
    /* ALLOCATE MEMORY ETC... FOR PARAMETERS USED WITH LALDEMOD */
    #ifdef DEBUG_DRIVESTACKSLIDE
           fprintf(stdout, "\nSTART SUBSECTION: stacks are F-statistic from SFTs\n");
           fflush(stdout);
    #endif

    #ifdef INCLUDE_DEMOD_CODE
        /* STKs are F-statistic from BLKs */
     #endif
    /**************************************************/
    /*                                                */
    /* END SUBSECTION: Stacks are F-stat from SFTs    */
    /*                                                */
    /**************************************************/

 } /* END if (params->stackTypeFlag ...) */

/**********************************************/
/*                                            */
/* END SECTION: make STKs                     */
/*                                            */
/**********************************************/

/***********************************************************/
/*                                                         */
/* START SECTION: write to stackslide search summary table */
/*                                                         */
/***********************************************************/

/* 05/26/04 gam; Move writing to stackslide search summary table to StackSlideConditionData */

/***********************************************************/
/*                                                         */
/* END SECTION: write to stackslide search summary table   */
/*                                                         */
/***********************************************************/

/******************************/
/*                            */
/* START SECTION: make SUMs   */
/*                            */
/******************************/

  #ifdef DEBUG_DRIVESTACKSLIDE
      fprintf(stdout, "\nSTART SECTION: make SUMs\n");
      fflush(stdout);
  #endif

  /* 02/11/04 gam; Change code to process 1 SUM at a time to allow processing of millions of SUMs per job. */
  params->SUMData=(REAL4FrequencySeries **)LALMalloc(sizeof(REAL4FrequencySeries *));
  params->SUMData[0]=(REAL4FrequencySeries *)LALMalloc(sizeof(REAL4FrequencySeries));
  params->SUMData[0]->data=(REAL4Vector *)LALMalloc(sizeof(REAL4Vector));
  params->SUMData[0]->data->data=(REAL4 *)LALMalloc(params->nBinsPerSUM*sizeof(REAL4));
  
  /* 02/17/04 gam; check whether we are outputing just the loudest events */
  if (params->outputLoudestFromPeaks || params->outputLoudestFromSUMs) {
       loudestPeaksArray = (SnglStackSlidePeriodicTable *)LALMalloc(params->keepThisNumber*sizeof(SnglStackSlidePeriodicTable));
       params->nBinsPerOutputEvent = params->nBinsPerSUM/params->keepThisNumber;
       InitializePeaksArray(loudestPeaksArray,params->keepThisNumber,params->f0SUM,params->dfSUM,params->nBinsPerOutputEvent); /* 02/20/04 gam */
       #ifdef INCLUDE_PRINT_PEAKS_TABLE_CODE
         if ( ((params->debugOptionFlag & 2) > 0) && params->startSUMs ) {
           fprintf(stdout,"\nOnly kept one loudest event per %i bins for a total of %i events.\n",params->nBinsPerOutputEvent, params->keepThisNumber);
           fflush(stdout);
         }
       #endif         
  }
  #ifdef INCLUDE_PRINT_PEAKS_TABLE_CODE
     if ( ((params->debugOptionFlag & 2) > 0) && params->startSUMs ) {
       fprintf(stdout,"\n   Event_no   SUM_index         RA        DEC    freq_deriv1   frequency      power  pwr_snr   width_bins\n");
       fflush(stdout);
     }
  #endif    

  /* Allocate memory and initialize StackSlideParams *stksldParams used by Isolated and Binary code*/
  /* Other stksldParams are set up in the StackSlideIsolated and StackSlideBinary functions */
  stksldParams = (StackSlideParams *)LALMalloc(1*sizeof(StackSlideParams));
  stksldParams->f0STK = params->f0STK;
  stksldParams->f0SUM = params->f0SUM;
  stksldParams->tSTK = params->tSTK;
  stksldParams->tSUM = params->tSUM;
  stksldParams->nBinsPerSUM = params->nBinsPerSUM;
  stksldParams->numSTKs = params->numSTKs;
  stksldParams->dfSUM = params->dfSUM;
  stksldParams->numSpinDown = params->numSpinDown;
  stksldParams->freqDerivData = NULL;
  if ( ( (params->testFlag & 1) > 0 ) || ( (params->weightFlag & 1) > 0 ) || ((params->debugOptionFlag & 8) > 0 ) ) {
    stksldParams->divideSUMsByNumSTKs = 0; /* FALSE if Hough Test or PowerFlux weighting is done. */
  } else {
    stksldParams->divideSUMsByNumSTKs = 1; /* default is TRUE */
  }
  
  /* Set up the IFO for the barycentering routines */
  if (strstr(params->IFO, "LHO")) {
             cachedDetector = lalCachedDetectors[LALDetectorIndexLHODIFF];
  } else if (strstr(params->IFO, "LLO")) {
             cachedDetector = lalCachedDetectors[LALDetectorIndexLLODIFF];
  } else if (strstr(params->IFO, "GEO")) {
             cachedDetector = lalCachedDetectors[LALDetectorIndexGEO600DIFF];
  } else if (strstr(params->IFO, "VIRGO")) {
             cachedDetector = lalCachedDetectors[LALDetectorIndexVIRGODIFF];
  } else if (strstr(params->IFO, "TAMA")) {
             cachedDetector = lalCachedDetectors[LALDetectorIndexTAMA300DIFF];
  } else {
             /* "Invalid or null IFO" */
             ABORT( status, DRIVESTACKSLIDEH_EIFO, DRIVESTACKSLIDEH_MSGEIFO);
  }
  (params->baryinput).site.location[0]=cachedDetector.location[0]/LAL_C_SI;
  (params->baryinput).site.location[1]=cachedDetector.location[1]/LAL_C_SI;
  (params->baryinput).site.location[2]=cachedDetector.location[2]/LAL_C_SI;
  (params->baryinput).alpha=params->stksldSkyPatchData->startRA;
  (params->baryinput).delta=params->stksldSkyPatchData->startDec;
  (params->baryinput).dInv=0.e0;  /* Inverse distance to the source.  Not sure what 0.0 does. */

  /* Set up parameters for finding peaks */
  pLALFindStackSlidePeakParams = (LALFindStackSlidePeakParams *)LALMalloc(sizeof(LALFindStackSlidePeakParams));
  pLALFindStackSlidePeakParams->returnPeaks = 1; /* 03/02/04 gam; default case is true */
  /* pLALFindStackSlidePeakParams->updateMeanStdDev = updateMeanStdDev; */ /* 02/28/05 gam */ /* 03/02/04 gam */
  if ( (params->thresholdFlag & 8) > 0 ) {
    pLALFindStackSlidePeakParams->updateMeanStdDev = 1;
  } else {
    pLALFindStackSlidePeakParams->updateMeanStdDev = 0;
  }  
  /* pLALFindStackSlidePeakParams->vetoWidePeaks = vetoWidePeaks;   */ /* /02/28/05 gam */ /* 02/20/04 gam */
  if ( (params->thresholdFlag & 2) > 0 ) {
    pLALFindStackSlidePeakParams->vetoWidePeaks = 1;
  } else {
    pLALFindStackSlidePeakParams->vetoWidePeaks = 0;
  }
  /* pLALFindStackSlidePeakParams->vetoOverlapPeaks = vetoOverlapPeaks; */ /* 02/28/05 gam */ /* 02/20/04 gam */
  if ( (params->thresholdFlag & 4) > 0 ) {
    pLALFindStackSlidePeakParams->vetoOverlapPeaks = 1;
  } else {
    pLALFindStackSlidePeakParams->vetoOverlapPeaks = 0;
  }
  /* 08/30/04 gam; if (outputEventFlag & 4) > 0 set returnOneEventPerSUM to TRUE; only the loudest event from each SUM is then returned. */
  if ( (params->outputEventFlag & 4) > 0 ) {
     pLALFindStackSlidePeakParams->returnOneEventPerSUM = 1;
  } else {
     pLALFindStackSlidePeakParams->returnOneEventPerSUM = 0;
  }
  pLALFindStackSlidePeakParams->threshold1 = params->threshold1;
  pLALFindStackSlidePeakParams->threshold2 = params->threshold2;
  pLALFindStackSlidePeakParams->fracPwDrop = params->threshold3;    /* 02/20/04 gam; fraction minimum power in a cluster must be below subpeak to count as new peak */
  pLALFindStackSlidePeakParams->maxWidthBins = params->maxWidthBins;                /* 02/20/04 gam */
  pLALFindStackSlidePeakParams->ifMin = params->maxWidthBins;                       /* 02/20/04 gam */
  pLALFindStackSlidePeakParams->ifMax = params->nBinsPerSUM - params->maxWidthBins; /* 02/20/04 gam */
  pLALFindStackSlidePeakParams->skyPosData = params->skyPosData;  /* 02/04/04 gam; added this and next 2 lines */
  pLALFindStackSlidePeakParams->freqDerivData = params->freqDerivData;
  pLALFindStackSlidePeakParams->numSpinDown = params->numSpinDown;
  pLALFindStackSlidePeakParams->binMask = params->sumBinMask; /* 05/19/05 gam; initialize binFlag with sumBinMask (called binMask in the struct). */
      
  /* 03/02/04 gam; allocate and initialize output struct for LALFindStackSlidePeaks  */    
  pLALFindStackSlidePeakOutputs = (LALFindStackSlidePeakOutputs *)LALMalloc(sizeof(LALFindStackSlidePeakOutputs));
  pLALFindStackSlidePeakOutputs->pwMeanWithoutPeaks = &pwMeanWithoutPeaks;
  pLALFindStackSlidePeakOutputs->pwStdDevWithoutPeaks = &pwStdDevWithoutPeaks;
  pLALFindStackSlidePeakOutputs->binsWithoutPeaks = &binsWithoutPeaks;
  pLALFindStackSlidePeakOutputs->acceptedEventCount = &acceptedEventCount;
  pLALFindStackSlidePeakOutputs->rejectedEventCount = &rejectedEventCount;
                        
  /* 02/17/04 gam */
  if (params->outputLoudestFromSUMs) {
     pLALUpdateLoudestStackSlideParams = (LALUpdateLoudestStackSlideParams *)LALMalloc(sizeof(LALUpdateLoudestStackSlideParams));
     pLALUpdateLoudestStackSlideParams->arraySize = params->keepThisNumber;
     pLALUpdateLoudestStackSlideParams->nBinsPerOutputEvent = params->nBinsPerOutputEvent;
     pLALUpdateLoudestStackSlideParams->ifMin = params->maxWidthBins;                       /* 02/20/04 gam */
     pLALUpdateLoudestStackSlideParams->ifMax = params->nBinsPerSUM - params->maxWidthBins; /* 02/20/04 gam */
     pLALUpdateLoudestStackSlideParams->skyPosData = params->skyPosData;
     pLALUpdateLoudestStackSlideParams->freqDerivData = params->freqDerivData;
     pLALUpdateLoudestStackSlideParams->numSpinDown = params->numSpinDown;
     pLALUpdateLoudestStackSlideParams->binMask = params->sumBinMask; /* 05/19/05 gam; In LALUpdateLoudestFromSUMs exclude bins with sumBinMask == 0 (binMask in struct). */
     params->totalEventCount = params->keepThisNumber; /* 02/20/04 gam; this is fixed when output is from SUMs */
  }
    
  if ((params->outputEventFlag > 0) && params->startSUMs) {
     /* 04/03/06 gam; to save disk space, if running an MC do not output rows with loudest event from each injection unless (params->outputEventFlag & 16) > 0 */
     if ( ( (params->testFlag & 2) > 0 )  && !( (params->outputEventFlag & 16) > 0 ) ) {
       /* Just continue! */
     } else {
       /* Begin the stackslide periodic table. */     
       fprintf( params->xmlStream->fp, LIGOLW_XML_SNGL_LOCAL_STACKSLIDEPERIODIC );
       params->xmlStream->first = 1;
     }
  }       

  /* 10/28/04 gam; weight STKs depending on value of params->weightFlag */
  if ( (params->weightFlag & 1) > 0 )  {
     if ( ( (params->weightFlag & 2) > 0 ) || ( (params->weightFlag & 4) > 0 ) ) {
          /* 10/28/04 gam; savSTKDATA for reuse with powerFlux style weighting of STKs for each sky position */ 
          SaveSTKData(params->savSTKData, params->STKData, params->numSTKs, params->nBinsPerSTK);
     } else {
          /* Just need to apply weights once */       
          WeightSTKsWithoutBeamPattern(params->STKData, params->inverseSquareMedians, params->sumInverseSquareMedians, params->numSTKs, params->nBinsPerSTK);
     }
  } else if (params->weightSTKsWithInverseMedians) {
          /* 09/12/05 gam; weight STKs with saved inverse medians for StackSlide style weighting */
          WeightSTKsWithInverseMedians(params->STKData, params->inverseMedians, params->numSTKs, params->nBinsPerSTK);
  }

  /* Check whether this is the isolated or binary case */
  if (params->binaryFlag==0) {
   
   /* 02/28/05 gam; All loops and output of isolated events has moved to StackSlideIsolated */
   StackSlideIsolated (
      status->statusPtr,
      loudestPeaksArray,
      pLALFindStackSlidePeakOutputs,
      pLALFindStackSlidePeakParams,
      pLALUpdateLoudestStackSlideParams,
      &cachedDetector,
      stksldParams,
      params );
   INTERNAL_SHOWERRORFROMSUB (status); CHECKSTATUSPTR (status);
    
  } else if ((params->binaryFlag & 1) == 1) {
   
   /* 04/12/05 gam; Move everthing but call to StackSlideBinary into StackSlideBinary.c */
   StackSlideBinary(status->statusPtr, stksldParams, params);
   CHECKSTATUSPTR (status);

  } /* end of if binaryFlag==1 else ...*/

  /* 02/17/04 gam */
  if (params->outputLoudestFromPeaks || params->outputLoudestFromSUMs) {
     LALFree(loudestPeaksArray);
  }

/******************************/
/*                            */
/* END SECTION: make SUMs     */
/*                            */
/******************************/

/*****************************************/
/*                                       */
/* START SECTION: Deallocate some memory */
/*                                       */
/*****************************************/

         #ifdef DEBUG_DRIVESTACKSLIDE
             fprintf(stdout, "\nSTART SECTION: Deallocate some memory\n");
             fflush(stdout);
         #endif

         /* 02/11/04 gam; Change code to process 1 SUM at a time to allow processing of millions of SUMs per job. */
         LALFree(params->SUMData[0]->data->data);
         LALFree(params->SUMData[0]->data);
         LALFree(params->SUMData[0]);
         LALFree(params->SUMData);

         /* Deallocate memory for the STKData structure */
         for(i=0;i<params->numSTKs;i++)
         {
             LALFree(params->STKData[i]->data->data);
             LALFree(params->STKData[i]->data);
             LALFree(params->STKData[i]);
         }
         LALFree(params->STKData);
         
         /* 10/28/04 gam; deallocate memory for parameters needed if weighting of STKs is done */
         if ( (params->weightFlag & 1) > 0 )  {    
             for(i=0;i<params->numSTKs;i++) {
                 LALFree(params->inverseSquareMedians[i]->data);
                 LALFree(params->inverseSquareMedians[i]);

             }         
             LALFree(params->inverseSquareMedians);
             LALFree(params->sumInverseSquareMedians->data);
             LALFree(params->sumInverseSquareMedians);
             
             if ( ( (params->weightFlag & 2) > 0 ) || ( (params->weightFlag & 4) > 0 ) ) {
                 for(i=0;i<params->numSTKs;i++) {
                    LALFree(params->savSTKData[i]->data);
                    LALFree(params->savSTKData[i]);

                 }
                 LALFree(params->savSTKData);
                 LALFree(params->detResponseTStampMidPts->data);
                 LALFree(params->detResponseTStampMidPts);
            } 
        } /* END if ( (params->weightFlag & 1) > 0 ) */         

         /* 11/08/03 gam; Add in first version of StackSlide written by Mike Landry */
         LALFree(stksldParams);
         
         /* 01/20/04 gam; Change findStackSlidePeaks to LALFindStackSlidePeaks; put params into struct */
         LALFree(pLALFindStackSlidePeakParams);
         
         LALFree(pLALFindStackSlidePeakOutputs); /* 03/02/04 gam; delete LALFindStackSlidePeaks output struct */

         if (params->outputLoudestFromSUMs) {
           LALFree(pLALUpdateLoudestStackSlideParams);  /* 02/17/04 gam */
         }
      
/*****************************************/
/*                                       */
/* END SECTION: Deallocate some memory   */
/*                                       */
/*****************************************/
      
  } else {

  } /* END if (params->searchMaster) else ... */

  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);
}
/******************************************/
/*                                        */
/* END FUNCTION: StackSlideApplySearch    */
/*                                        */
/******************************************/

/********************************************/
/*                                          */
/* START FUNCTION: StackSlideFinalizeSearch */
/*                                          */
/********************************************/
void StackSlideFinalizeSearch(
    LALStatus              *status,
    StackSlideSearchParams *params
    )
{
  INT4          i = 0; /* 05/11/04 gam */

  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);

  #ifdef DEBUG_DRIVESTACKSLIDE
      fprintf(stdout, "\nSTART FUNCTION: StackSlideFinalizeSearch\n");
      fflush(stdout);
  #endif

  /* params->searchMaster is used by LDAS; set to TRUE in this code */
  if (params->searchMaster) {

    /* 05/24/05 gam; finish stackslide periodic table in not done previously; move to FinializeSearch */
    if ((params->outputEventFlag > 0) && params->finishPeriodicTable) {
      /* 04/03/06 gam; to save disk space, if running an MC do not output rows with loudest event from each injection unless (params->outputEventFlag & 16) > 0 */
      if ( ( (params->testFlag & 2) > 0 )  && !( (params->outputEventFlag & 16) > 0 ) ) {
         /* Just continue! */
      } else {
        fprintf( params->xmlStream->fp, STACKSLIDE_XML_TABLE_FOOTER ); /* 04/12/05 gam */
        params->xmlStream->table = no_table;
      }
    }
    if (params->outputEventFlag > 0) {
      MetadataTable         searchsumm;       
      MetadataTable         searchsummvars;

      /* write the search summary table */
      searchsumm.searchSummaryTable = (SearchSummaryTable *) LALCalloc( 1, sizeof(SearchSummaryTable) );
      snprintf( searchsumm.searchSummaryTable->comment, LIGOMETA_COMMENT_MAX, "%s", params->patchName );  /* Let patchName serve as the comment */
      searchsumm.searchSummaryTable->in_start_time.gpsSeconds = params->gpsStartTimeSec;
      searchsumm.searchSummaryTable->in_start_time.gpsNanoSeconds = params->gpsStartTimeNan;
      searchsumm.searchSummaryTable->out_start_time.gpsSeconds = params->actualStartTime.gpsSeconds;   
      searchsumm.searchSummaryTable->out_start_time.gpsNanoSeconds = params->actualStartTime.gpsNanoSeconds;
      searchsumm.searchSummaryTable->in_end_time.gpsSeconds = params->gpsStartTimeSec + params->duration;
      searchsumm.searchSummaryTable->in_end_time.gpsNanoSeconds = params->gpsStartTimeNan;
      searchsumm.searchSummaryTable->out_end_time.gpsSeconds = params->actualEndTime.gpsSeconds;
      searchsumm.searchSummaryTable->out_end_time.gpsNanoSeconds = params->actualEndTime.gpsNanoSeconds;
      searchsumm.searchSummaryTable->nevents = params->totalEventCount;
      searchsumm.searchSummaryTable->nnodes = 0;
      searchsumm.searchSummaryTable->next = NULL;

      LALBeginLIGOLwXMLTable( status->statusPtr, params->xmlStream, search_summary_table );
      CHECKSTATUSPTR (status);
      LALWriteLIGOLwXMLTable( status->statusPtr, params->xmlStream, searchsumm, search_summary_table );
      CHECKSTATUSPTR (status);
      LALEndLIGOLwXMLTable (  status->statusPtr, params->xmlStream );
      CHECKSTATUSPTR (status);
      LALFree(searchsumm.searchSummaryTable);

      /* write the search summ vars table */
      searchsummvars.searchSummvarsTable = (SearchSummvarsTable *) LALCalloc( 1, sizeof(SearchSummvarsTable) );
      snprintf( searchsummvars.searchSummvarsTable->name,LIGOMETA_NAME_MAX, "%s","max_power");
      snprintf( searchsummvars.searchSummvarsTable->string,LIGOMETA_STRING_MAX, "%s"," ");
      searchsummvars.searchSummvarsTable->value = (REAL8)(params->maxPower);
      searchsummvars.searchSummvarsTable->next = NULL;

      LALBeginLIGOLwXMLTable( status->statusPtr, params->xmlStream, search_summvars_table );
      CHECKSTATUSPTR (status);
      LALWriteLIGOLwXMLTable( status->statusPtr, params->xmlStream, searchsummvars, search_summvars_table );
      CHECKSTATUSPTR (status);
      LALEndLIGOLwXMLTable (  status->statusPtr, params->xmlStream );
      CHECKSTATUSPTR (status);
      LALFree(searchsummvars.searchSummvarsTable);
    
      /* Print xml file footer and close the file. */
      LALCloseLIGOLwXMLFile (status->statusPtr,params->xmlStream);
      CHECKSTATUSPTR (status);
    
      /* LALFree(xmlFile); */ /* 02/09/04 gam; Already freed in StackSlideInitSearch */
      LALFree(params->xmlStream);
    } /* end if (params->outputEventFlag > 0) */

  /**********************************************/
  /*                                            */
  /* START SECTION: Deallocate remaining memory */
  /*                                            */
  /**********************************************/

   #ifdef DEBUG_DRIVESTACKSLIDE
        fprintf(stdout, "\nSTART SECTION: Deallocate remaining memory\n");
        fflush(stdout);
   #endif
   
  /* Deallocate memory for the freqDerivData structure */
  /* 05/11/04 gam; free memory in StackSlideFinalizeSearch */
  if (params->numSpinDown > 0) {
    for(i=0;i<params->numFreqDerivTotal;i++)
    {
        LALFree(params->freqDerivData[i]);
    }
    LALFree(params->freqDerivData);
  }

  /* Deallocate memory for the skyPosData structure */
  /* 05/11/04 gam; free memory in StackSlideFinalizeSearch */
  for(i=0;i<params->numSkyPosTotal;i++)
  {
       LALFree(params->skyPosData[i]);
  }
  LALFree(params->skyPosData);
   
  /* 05/11/04 gam; free memory in StackSlideFinalizeSearch */
  LALFree(params->edat->ephemS);
  LALFree(params->edat->ephemE);
  LALFree(params->edat);
         
  /* 05/11/04 gam; free memory in StackSlideFinalizeSearch */
  LALFree(params->stksldSkyPatchData);

  LALFree(params->ifoNickName);
  LALFree(params->IFO);
  LALFree(params->patchName);

  LALFree(params->priorResultsFile);
  LALFree(params->parameterSpaceFile);

  LALFree(params->dsoName);
  
  LALFree(params->sunEdatFile);
  LALFree(params->earthEdatFile);
  LALFree(params->sftDirectory);
  LALFree(params->outputFile);
  
  /* 05/14/05 gam */
  LALFree(params->linesAndHarmonicsFile);
  /* if ( (params->normalizationFlag & 32) > 0 ) */ /* 07/13/05 gam */
  if ( ((params->normalizationFlag & 32) > 0) || ((params->normalizationFlag & 64) > 0)  )  {
    if (params->infoLines->nLines > 0) {
      LALFree(params->infoLines->lineFreq);
      LALFree(params->infoLines->leftWing);
      LALFree(params->infoLines->rightWing);
    } 
    if (params->infoHarmonics->nHarmonicSets > 0) {
      LALFree(params->infoHarmonics->startFreq);
      LALFree(params->infoHarmonics->gapFreq);
      LALFree(params->infoHarmonics->numHarmonics);
      LALFree(params->infoHarmonics->leftWing);
      LALFree(params->infoHarmonics->rightWing);
    }
    LALFree(params->infoHarmonics);
    LALFree(params->infoLines);

    /* 07/13/05 gam; make RandomParams *randPar a parameter for CleanCOMPLEX8SFT; initialze RandomParams *randPar once to avoid repeatly opening /dev/urandom */    
    if ( (params->normalizationFlag & 64) > 0 ) {    
      LALDestroyRandomParams(status->statusPtr, &(params->randPar));
      CHECKSTATUSPTR (status);
    }
  } /* END if ( ((params->normalizationFlag & 32) > 0) || ((params->normalizationFlag & 64) > 0)  ) */

  LALFree(params->sumBinMask);  /* 05/19/05 gam */

  /* 09/12/05 gam; deallocate memory for parameters needed if weighting of STKs with inverse medians is done */
  if ( params->weightSTKsWithInverseMedians && (params->inverseMedians != NULL) ) {
     for(i=0;i<params->numSTKs;i++) {
        LALFree(params->inverseMedians[i]->data);
        LALFree(params->inverseMedians[i]);
     }         
     LALFree(params->inverseMedians);
  }

  /* LALFree(params); */ /*Declared and freed in the calling function */

  /**********************************************/
  /*                                            */
  /* END SECTION: Deallocate remaining memory   */
  /*                                            */
  /**********************************************/
  }

 CHECKSTATUSPTR (status);
 DETATCHSTATUSPTR (status);
}
/********************************************/
/*                                          */
/* END FUNCTION: StackSlideFinalizeSearch   */
/*                                          */
/********************************************/

/******************************************/
/*                                        */
/* START SECTION: internal functions      */
/*                                        */
/******************************************/

#ifdef INCLUDE_TIMEFLOAT_CODE
/* Copied from LALDemodTest.c */
static void TimeToFloat(REAL8 *f, LIGOTimeGPS *tgps)
{
  INT4 x, y;

  x=tgps->gpsSeconds;
  y=tgps->gpsNanoSeconds;
  *f=(REAL8)x+(REAL8)y*1.e-9;
}

/* Copied from LALDemodTest.c */
static void FloatToTime(LIGOTimeGPS *tgps, REAL8 *f)
{
  REAL8 temp0, temp2, temp3;
  REAL8 temp1, temp4;

  temp0 = floor(*f);     /* this is tgps.S */
  temp1 = (*f) * 1.e10;
  temp2 = fmod(temp1, 1.e10);
  temp3 = fmod(temp1, 1.e2);
  temp4 = (temp2-temp3) * 0.1;

  tgps->gpsSeconds = (INT4)temp0;
  tgps->gpsNanoSeconds = (INT4)temp4;
}
#endif

/* Calculate stats associated with frequency series */ /* 02/11/04 gam; comment out */
/*void ComputePowerStats( const REAL4Vector *fsVec, FreqSeriesPowerStats *stats, REAL8 fStart, REAL8 df )
{
    UINT4 i = 0;
    INT4 ipwMax = 0;
    REAL8 pw = 0.0;
    REAL8 pwSum= 0.0;
    REAL8 pwSum2 = 0.0;

    stats->pwMax = 0.0;

    for(i=0;i<fsVec->length;i++) {
		pw = (REAL8)fsVec->data[i];

	        pwSum += pw;
	        pwSum2 += pw*pw;

		if(stats->pwMax < pw)
		{
			stats->pwMax=pw;
			ipwMax=i;
		}
    }
    stats->freqPWMax = fStart + ((REAL8)ipwMax)*df;
    stats->pwMean = pwSum/((REAL8)fsVec->length);

    stats->pwStdDev = pwSum2/((REAL8)fsVec->length) - stats->pwMean*stats->pwMean;
    if ( stats->pwStdDev > 0.0 ) {
    	stats->pwStdDev = sqrt( stats->pwStdDev );
    } else {
        stats->pwStdDev = -1.0;
    }
}*/

/* void InitializePeaksArray( SnglStackSlidePeriodicTable *loudestPeaksArray, INT4 arraySize ) */ /* 02/20/04 gam */
void InitializePeaksArray( SnglStackSlidePeriodicTable *loudestPeaksArray, INT4 arraySize, REAL8 f0, REAL8 df, INT4 nBinsPerOutputEvent )
{
    INT4 i = 0;           /* all purpose index */
    REAL8 freq = 0;
    for (i = 0; i < arraySize; i++) {
            freq = f0 + i*nBinsPerOutputEvent*df; /* 02/20/04 gam; default frequency */
            loudestPeaksArray[i].sum_no=-1;
            loudestPeaksArray[i].event_no_thissum=-1;
            loudestPeaksArray[i].overlap_event=-1;
            loudestPeaksArray[i].frequency=freq;  /* 02/20/04 gam; set default */
            loudestPeaksArray[i].freq_index=-1;
            loudestPeaksArray[i].power=0;         /* default power is 0 */
            loudestPeaksArray[i].width_bins=-1;
            loudestPeaksArray[i].num_subpeaks=-1;
            loudestPeaksArray[i].pwr_snr=-1;
            loudestPeaksArray[i].false_alarm_prob=-1;
            loudestPeaksArray[i].goodness_of_fit=-1;
            loudestPeaksArray[i].sky_ra=-1;
            loudestPeaksArray[i].sky_dec=-1;
            loudestPeaksArray[i].fderiv_1=-1;
            loudestPeaksArray[i].pw_mean_thissum=-1;
            loudestPeaksArray[i].pw_stddev_thissum=-1;
    }
} /* end InitializePeaksAndPowerArrays */

void LALUpdateLoudestFromPeaks( SnglStackSlidePeriodicTable *loudestPeaksArray, const SnglStackSlidePeriodicTable *peaks, INT4 nBinsPerOutputEvent )
{
    INT4 i = 0;           /* all purpose index */
    
    i = peaks->freq_index/nBinsPerOutputEvent;  /* Determine which loudestPeaksArray bin this frequency belongs to */
    
    if (peaks->power > loudestPeaksArray[i].power) {
            loudestPeaksArray[i].sum_no=peaks->sum_no;
            loudestPeaksArray[i].event_no_thissum=peaks->event_no_thissum;
            loudestPeaksArray[i].overlap_event=peaks->overlap_event;
            loudestPeaksArray[i].frequency=peaks->frequency;
            loudestPeaksArray[i].freq_index=peaks->freq_index;
            loudestPeaksArray[i].power=peaks->power;
            loudestPeaksArray[i].width_bins=peaks->width_bins;
            loudestPeaksArray[i].num_subpeaks=peaks->num_subpeaks;
            loudestPeaksArray[i].pwr_snr=peaks->pwr_snr;
            loudestPeaksArray[i].false_alarm_prob=peaks->false_alarm_prob;
            loudestPeaksArray[i].goodness_of_fit=peaks->goodness_of_fit;
            loudestPeaksArray[i].sky_ra=peaks->sky_ra;
            loudestPeaksArray[i].sky_dec=peaks->sky_dec;
            loudestPeaksArray[i].fderiv_1=peaks->fderiv_1;
            loudestPeaksArray[i].pw_mean_thissum=peaks->pw_mean_thissum;
            loudestPeaksArray[i].pw_stddev_thissum=peaks->pw_stddev_thissum;
   }
} /* end LALUpdateLoudestFromPeaks */

void LALUpdateLoudestFromSUMs( SnglStackSlidePeriodicTable *loudestPeaksArray, const REAL4FrequencySeries *oneSUM, const LALUpdateLoudestStackSlideParams *params )
{

    INT4 i = 0;           /* all purpose index */
    INT4 j = 0;           /* all purpose index */
    INT4 k = 0;           /* all purpose index */
    REAL4 pwr = 0;
    /* REAL8 fMin = oneSUM->f0 + params->maxWidth;
    REAL8 fMax = oneSUM->f0 + ((REAL8)oneSUM->data->length)*oneSUM->deltaF - params->maxWidth; */ /* 02/20/04 gam */
    REAL8 fPeak = 0;      /* The actual frequency that the max pwr is at */
    BOOLEAN findMeanAndStdDev = 1;
    REAL4 pwrSum= 0.0;    /* sum of power */
    REAL4 pwrSum2 = 0.0;  /* sum of power squared = sum of power*/
    REAL4 pwrMean = 0.0;
    REAL4 pwrStdDev = 0.0;        
    INT4 eventCount = 0;
    INT4 binCount = 0; /* 05/19/05 gam */

    for(j=0;j<oneSUM->data->length;j++) {
       
       /* 05/19/05 gam; In LALUpdateLoudestFromSUMs exclude bins with sumBinMask == 0 (binMask in struct). */
       if (params->binMask[j] != 0) {
         pwr = oneSUM->data->data[j];
         i = j/params->nBinsPerOutputEvent;  /* Determine which loudestPeaksArray bin this frequency belongs to */                   
       
         if (pwr > loudestPeaksArray[i].power) {
            eventCount++;
            if (findMeanAndStdDev) {
               findMeanAndStdDev = 0;  /* Find mean and std dev only once */
               for(k=0;k<oneSUM->data->length;k++) {
                   /* 05/19/05 gam; In LALUpdateLoudestFromSUMs exclude bins with sumBinMask == 0 (binMask in struct). */
                   if (params->binMask[k] != 0) {
                     pwrSum += oneSUM->data->data[k];
                     pwrSum2 += oneSUM->data->data[k]*oneSUM->data->data[k];
                     binCount++;
                   }
               }
               /* pwrMean = pwrSum/((REAL4)oneSUM->data->length); */  /* Mean pwr */ /* 05/19/05 gam; */
               /* pwrStdDev = pwrSum2/((REAL4)oneSUM->data->length) - pwrMean*pwrMean; */ /* 05/19/05 gam; */
               if (binCount > 0) {
                 pwrMean = pwrSum/((REAL4)binCount);   /* Mean pwr */
                 pwrStdDev = pwrSum2/((REAL4)binCount) - pwrMean*pwrMean;
               } else {
                 pwrMean = -1.0;
                 pwrStdDev = -1.0;
               }
               if ( pwrStdDev > 0.0 ) {
                 pwrStdDev = sqrt( pwrStdDev );
               } else {
                 pwrStdDev = -1.0; /* Should not happen */
               }
            }
            fPeak = oneSUM->f0+(REAL8)j*oneSUM->deltaF;
            loudestPeaksArray[i].sum_no=params->iSUM;
            loudestPeaksArray[i].event_no_thissum=eventCount;
            /* if (fPeak < fMin || fPeak >= fMax) */ /* 02/20/04 gam */
            if (j < params->ifMin || j >= params->ifMax) {
              loudestPeaksArray[i].overlap_event = 1;
            } else {
              loudestPeaksArray[i].overlap_event = 0;
            }
            loudestPeaksArray[i].frequency=fPeak;
            loudestPeaksArray[i].freq_index=j;
            loudestPeaksArray[i].power=pwr;
            /* loudestPeaksArray[i].width_bins=-1; */  /* This is already the default */
            /* loudestPeaksArray[i].num_subpeaks=-1; */ /* This is already the default */
            loudestPeaksArray[i].pwr_snr=pwr/pwrStdDev;
            /* loudestPeaksArray[i].false_alarm_prob=-1; */ /* This is already the default */
            /* loudestPeaksArray[i].goodness_of_fit=-1;  */ /* This is already the default */
            loudestPeaksArray[i].sky_ra = params->skyPosData[params->iSky][0];
            loudestPeaksArray[i].sky_dec = params->skyPosData[params->iSky][1];
            for(k=0;k<params->numSpinDown;k++)
            {
                if (k == 0) {
                   loudestPeaksArray[i].fderiv_1 = params->freqDerivData[params->iDeriv][k];
                } else if (k == 1) {
                   /* loudestPeaksArray[i].fderiv_2 = params->freqDerivData[params->iDeriv][k]; */
                } else if (k == 2) {
                   /* loudestPeaksArray[i].fderiv_3 = params->freqDerivData[params->iDeriv][k]; */
                } else if (k == 3) {
                   /* loudestPeaksArray[i].fderiv_4 = params->freqDerivData[params->iDeriv][k]; */
                } else if (k == 4) {
                   /* loudestPeaksArray[i].fderiv_5 = params->freqDerivData[params->iDeriv][k]; */
                }
            }
            for(k=params->numSpinDown;k<5;k++)
            {
                if (k == 0) {
                   loudestPeaksArray[i].fderiv_1 = 0;
                } else if (k == 1) {
                   /* loudestPeaksArray[i].fderiv_2 = 0; */
                } else if (k == 2) {
                   /* loudestPeaksArray[i].fderiv_3 = 0; */
                } else if (k == 3) {
                   /* loudestPeaksArray[i].fderiv_4 = 0; */
                } else if (k == 4) {
                   /* loudestPeaksArray[i].fderiv_5 = 0; */
                }
            }
            loudestPeaksArray[i].pw_mean_thissum = pwrMean;
            loudestPeaksArray[i].pw_stddev_thissum = pwrStdDev;
         } /* end if (pwr > loudestPeaksArray[i]->power) */
       } /* END if (params->binMask[j] != 0) */
    } /* endfor(j=0;j<oneSUM->data->length;j++) */
} /* end LALUpdateLoudestFromSUMs */

/* 01/14/04 gam; Function for finding peaks in StackSlide SUMs */
/* void findStackSlidePeaks( const REAL4FrequencySeries *oneSUM, StackSlidePeak *peaks, REAL4 threshold1, REAL4 threshold2 ) */
/* 01/20/04 gam; Change findStackSlidePeaks to LALFindStackSlidePeaks; put params into struct */
/* void LALFindStackSlidePeaks( StackSlidePeak *peaks, const REAL4FrequencySeries *oneSUM, LALFindStackSlidePeakParams *params ) */ /* 02/04/04 gam */
/* void LALFindStackSlidePeaks( SnglStackSlidePeriodicTable *peaks, const REAL4FrequencySeries *oneSUM, LALFindStackSlidePeakParams *params ) */ /* 02/04/04 gam */
/* 02/04/04 gam; Note need pntrToPeaksPntr to be pointer to a pointer so that we can change BOTH the data of peak and the pointer to the current peak in the linked list */
/* 02/11/04 gam; Use power rather than amplitudes when finding peaks */
/* void LALFindStackSlidePeaks( SnglStackSlidePeriodicTable **pntrToPeaksPntr, const REAL4FrequencySeries *oneSUM, LALFindStackSlidePeakParams *params ) */
/* 03/02/04 gam; Add LALFindStackSlidePeakOutputs; change pntrToPeaksPntr to (outputs->pntrToPeaksPntr) */
void LALFindStackSlidePeaks( LALFindStackSlidePeakOutputs *outputs, const REAL4FrequencySeries *oneSUM, LALFindStackSlidePeakParams *params )
{
    INT4 i = 0;           /* all purpose index */
    INT4 j = 0;           /* all purpose index */
    INT4 k = 0;           /* all purpose index */ /* 02/04/04 gam */
    INT4 iPwrMax = 0;     /* index max power */ /* 01/27/04 gam; renamed from ipwrMax */
    REAL4 pwrMax = 0.0;   /* maximum power */
    INT4  iPwrMin = 0;       /* 01/27/04 gam; index of local minimum power */
    REAL4 pwrMin = 0.0;   /* 01/27/04 gam; local minimum power */
    /* REAL4 pwr = 0.0;  */ /* 01/27/04 gam; Not used. */ /* power */
    REAL4 currentPwr = 0.0; /* 02/11/04 gam; current power */
    REAL4 lastPwr = 0.0;  /* last power */
    REAL4 secondToLastPwr = 0.0;  /* 02/04/04 gam; one before the last one */
    REAL4 pwrSum= 0.0;    /* sum of power */
    REAL4 pwrSum2 = 0.0;  /* sum of power squared = sum of power*/
    REAL4 pwrMean = 0.0;
    REAL4 pwrStdDev = 0.0;
    INT4 *binFlag;        /* Flag indicating if bin already belongs to a peak */
    /* REAL4 *powers; */  /* 02/11/04 gam; used to be amplitudes; no longer needed */
    INT4 peakWidth = 0;
    INT4 jMin = 0;           /* 01/27/04 gam; minimum index of a peak */
    INT4 jMax = 0;           /* 01/27/04 gam; maximum index of a peak */
    INT4 eventCount = 0;     /* 02/04/04 gam; found a peak or a cluster of peaks; keep track of how many */
    INT4 rejectedCount = 0;  /* 03/02/04 gam; keep track of how many events were rejected */
    INT4 binCount = 0;       /* 03/02/04 gam; keep track of how many bins do not have peaks */
    INT4 subPeakCount = 0;   /* 01/27/04 gam; Count number of subpeaks in a cluster */
    BOOLEAN findMorePeaks = 1;
    BOOLEAN findWidth = 1;
    BOOLEAN outputThisPeak = 1; /* 02/20/04 gam */
    INT4 overlapEvent = 0;      /* 02/20/04 gam */
    /* REAL8 fMin = oneSUM->f0 + params->maxWidth;
    REAL8 fMax = oneSUM->f0 + ((REAL8)oneSUM->data->length)*oneSUM->deltaF - params->maxWidth; */ /* 02/20/04 gam */
    REAL8 fPeak = 0.0;
    SnglStackSlidePeriodicTable *peaks = *(outputs->pntrToPeaksPntr); /* 02/04/04 gam; peaks is the pointer to current peak */
    SnglStackSlidePeriodicTable *previousPeak;    /* 02/04/04 gam; needed to remember pointer to previous peak */
    
    binFlag=(INT4 *)LALMalloc(oneSUM->data->length*sizeof(INT4));
    /* powers=(REAL4 *)LALMalloc(oneSUM->data->length*sizeof(REAL4)); */ /* 02/11/04 gam; used to be amplitudes; no longer needed */
             
    for(i=0;i<oneSUM->data->length;i++) {
        /* binFlag[i] = 0; */ /* 05/19/05 gam */
        binFlag[i] = params->binMask[i];
        if (binFlag[i] != 0) {
          pwrSum += oneSUM->data->data[i];
          pwrSum2 += oneSUM->data->data[i]*oneSUM->data->data[i];
          binCount++;
        }
    }
    /* pwrMean = pwrSum/((REAL4)oneSUM->data->length); */ /* 05/19/05 gam */ /* Mean pwr */
    /* Standard Deviation of the power = sqrt of the variance */
    /* pwrStdDev = pwrSum2/((REAL4)oneSUM->data->length) - pwrMean*pwrMean; */ /* 05/19/05 gam */
    if (binCount > 0) {    
      pwrMean = pwrSum/((REAL4)binCount); /* Mean pwr */
      pwrStdDev = pwrSum2/((REAL4)binCount) - pwrMean*pwrMean;      
    } else {
      pwrMean = -1.0;
      pwrStdDev = -1.0;
    }
    if ( pwrStdDev > 0.0 ) {
        pwrStdDev = sqrt( pwrStdDev );
    } else {
        pwrStdDev = -1.0; /* Should not happen except due to round off error in test data */
    }
    binCount = oneSUM->data->length - binCount; /* 05/19/05 gam; binCount now number of excluded bins */
    
    while (findMorePeaks) {
        pwrMax = 0.0;
        iPwrMax = 0;
        for(i=0;i<oneSUM->data->length;i++) {
               /* if (binFlag[i] != 1) */ /* 05/19/05 gam */
                if (binFlag[i] != 0) {
                  if (oneSUM->data->data[i] > pwrMax) {
                     pwrMax = oneSUM->data->data[i];
                     iPwrMax = i;
                  }
                }
        }
	if (pwrMax > params->threshold1) {
		/* Found a peak; analyze it */
		subPeakCount = 0; /* 01/27/04 gam; initialize */
                secondToLastPwr = pwrMax;  /* 02/04/04 gam */
		lastPwr = pwrMax;
		iPwrMin = iPwrMax;  /* 01/27/04 gam */
                pwrMin = pwrMax; /* 01/27/04 gam */
		j = iPwrMax -1;
		findWidth = 1;
		jMin  = j; /* 01/27/04 gam; initialize with default value */
		while (j > -1 && findWidth) {
                     currentPwr = oneSUM->data->data[j];
		     /* 01/27/04 gam; keep track of minimum */
		     if (currentPwr < pwrMin) {
		         pwrMin = currentPwr;
		         iPwrMin = j;
		     }
		     /* if ( binFlag[j] == 1 || currentPwr < params->threshold2 ) */ /* 05/19/05 gam */
		     if ( binFlag[j] == 0 || currentPwr < params->threshold2 ) {
		        jMin = j; /* 01/27/04 gam; save mimimum j */
		        findWidth = 0;
		     } else if ( currentPwr > lastPwr && params->fracPwDrop*currentPwr > pwrMin ) {
                        /* 01/27/04 gam; Power has gone up and pwrMin was < fracPwDrop times current power; */
		        jMin = iPwrMin-1; /* Set jMin back to point where power was minimum minus 1 to include in width */
		        findWidth = 0;
		     } else {
		       /* if ( powers[j] > params->threshold1 && powers[j] > lastPwr ) */ /* 02/04/04 gam */
		       if ( lastPwr > secondToLastPwr && currentPwr < lastPwr ) {
		           subPeakCount++;  /* 01/27/04 gam; a significant subpeak is included in one peak event. */
		       }
                       secondToLastPwr = lastPwr;  /* 02/04/04 gam */
		       lastPwr = currentPwr;
		       j--;
		     }
		     /* lastPwr = powers[j];
		     j--; */ /* 01/20/04 gam; Correct code that calculates peak width; move into else to avoid extra j-- */
		}
                secondToLastPwr = pwrMax;  /* 02/04/04 gam; Need to reinitialize */
		lastPwr = pwrMax;  /* 01/27/04 gam; Need to reinitialize */
		iPwrMin = iPwrMax;  /* 01/27/04 gam */
                pwrMin = pwrMax; /* 01/27/04 gam */
		j = iPwrMax + 1;
		findWidth = 1;
		jMax = j; /* 01/27/04 gam; initialize with default value */
		while (j < oneSUM->data->length && findWidth) {
                     currentPwr = oneSUM->data->data[j];
		     /* 01/27/04 gam; keep track of minimum */
		     if (currentPwr < pwrMin) {
		         pwrMin = currentPwr;
			 iPwrMin = j;
		     }
		     /* if ( binFlag[j] == 1 || currentPwr < params->threshold2 ) */ /* 05/19/05 gam */
		     if ( binFlag[j] == 0 || currentPwr < params->threshold2 ) {
		        jMax = j; /* 01/27/04 gam; save maximum j */
		        findWidth = 0;
		     } else if ( currentPwr > lastPwr && params->fracPwDrop*currentPwr > pwrMin ) {
                        /* 01/27/04 gam; Power has gone up and pwrMin was < fracPwDrop times current power; */
		        jMax = iPwrMin+1; /* Set jMax back to point where power was minimum plus 1 to include in width. */
		        findWidth = 0;
		     } else {
		       /* if ( powers[j] > params->threshold1 && powers[j] > lastPwr ) */ /* 02/04/04 gam */
		       if ( lastPwr > secondToLastPwr && currentPwr < lastPwr ) {
		           subPeakCount++;  /* 01/27/04 gam; a significant subpeak is included in one peak event. */
		       }
                       secondToLastPwr = lastPwr;  /* 02/04/04 gam */
		       lastPwr = currentPwr;
		       j++;
		     }
		     /* lastPwr = powers[j];
		     j++; */ /* 01/20/04 gam; Correct code that calculates peak width; move into else to avoid extra j++ */
		}
		peakWidth = jMax - jMin - 1;
                /* 02/17/04 gam; always set this to avoid infinite loop of finding same peak over and over, even if not recording this peak. */
                for(i=jMin+1;i<jMax;i++) {
		           /* binFlag[i] = 1; */ /* set bin as belonging to this peak */ /* 05/19/05 gam */
		           binFlag[i] = 0; /* set bin as belonging to this peak */
                           binCount++;     /* 03/02/04 gam; keep track of number of bins belonging to a peak */
		}
                /* 02/20/04 gam; check for vetos on peaks in overlap zone or width greater than maxWidthBins */
                if (iPwrMax < params->ifMin || iPwrMax >= params->ifMax) {
                   overlapEvent = 1;
                } else {
                   overlapEvent = 0;
                }
                /* if ( (overlapEvent > 0) && params->vetoOverlapPeaks ) */ /* 03/02/04 gam; add check on params->returnPeaks */
                if (!(params->returnPeaks)) {
                    outputThisPeak = 0; /* flag is set to not return any peaks; the number of potential peaks is counted by rejectedCount*/
                } else if ( (overlapEvent > 0) && params->vetoOverlapPeaks ) {
                    outputThisPeak = 0; /* do not output peak in overlap zone */
                } else if ( (peakWidth > params->maxWidthBins) && params->vetoWidePeaks ) {
                    outputThisPeak = 0; /* do not output wide peak */
                } else {
                    outputThisPeak = 1; /* default */
                }
                if (outputThisPeak) {
                  eventCount++; /* 02/04/04 gam; found a peak or a cluster of peaks; keep track of how many */
                  if (eventCount == 1) {
                    if (params->returnOneEventPerSUM) {
                       findMorePeaks = 0;  /* 08/30/04 gam; first event found = loudest event; return after setting this peak below. */
                    }
                    #ifdef DEBUG_FIND_PEAKS
                       fprintf(stdout,"\nFound peak(s) in SUM #%i for template parameters:\n",params->iSUM);
                       fprintf(stdout,"RA = %18.10f\n",params->skyPosData[params->iSky][0]);
                       fprintf(stdout,"DEC = %18.10f\n",params->skyPosData[params->iSky][1]);
                       for(k=0;k<params->numSpinDown;k++)
                       {
                         fprintf(stdout,"FREQDERIV%i = %18.10f\n",k+1,params->freqDerivData[params->iDeriv][k]);
                       }  
                       fflush(stdout);
                    #endif
		    if (!peaks) {
                       *(outputs->pntrToPeaksPntr) = peaks = (SnglStackSlidePeriodicTable *)LALMalloc(sizeof(SnglStackSlidePeriodicTable));
                       peaks->next = NULL;
		    } else {
                       previousPeak = peaks;
		       peaks = peaks->next;  /* Move peaks pointer to next peak */
                       *(outputs->pntrToPeaksPntr) = peaks = (SnglStackSlidePeriodicTable *)LALMalloc(sizeof(SnglStackSlidePeriodicTable));
                       peaks->next = NULL;
                       previousPeak->next = peaks; /* Make previous peak next element point to the current peak */
		    }
		  } else {
                    previousPeak = peaks;
		    peaks = peaks->next;  /* Move peaks pointer to next peak */
                    *(outputs->pntrToPeaksPntr) = peaks = (SnglStackSlidePeriodicTable *)LALMalloc(sizeof(SnglStackSlidePeriodicTable));
                    peaks->next = NULL;
                    previousPeak->next = peaks; /* Make previous peak next element point to the current peak */
		  }
                  fPeak = oneSUM->f0+(REAL8)iPwrMax*oneSUM->deltaF;
                  #ifdef DEBUG_FIND_PEAKS
	            fprintf(stdout,"\nFound peak at f = %18.10f \n",fPeak);
		    /* if (fPeak < fMin || fPeak >= fMax) */ /* 02/20/04 gam */
		    if (overlapEvent > 0) {
			fprintf(stdout,"Peak is in an overlap zone!  Should set overlap_event to 1.\n");
		    }
		    fprintf(stdout,"power = %g \n",pwrMax);
		    fprintf(stdout,"width = %i \n",peakWidth);
		    fprintf(stdout,"SNR = %g\n",pwrMax/pwrStdDev);
		    fprintf(stdout,"Number sub-peaks above threshold1 found with this peak = %i \n",subPeakCount);
	            fflush(stdout);
		  #endif
                  /* 02/04/04 gam; assign values for the current peak to the linked list */
                  /* 02/12/04 gam; Add freq_index to SnglStackSlidePeriodicTable */
		  peaks->sum_no = params->iSUM;
		  peaks->event_no_thissum = eventCount;
		  /* if (fPeak < fMin || fPeak >= fMax) */ /* 02/20/04 gam */
		  /* if (iPwrMax < params->ifMin || iPwrMax >= params->ifMax) {
		     peaks->overlap_event = 1;
		  } else {
		     peaks->overlap_event = 0;
		  } */
                  peaks->overlap_event = overlapEvent;  /* 02/20/04 gam */
		  peaks->frequency = fPeak;
		  peaks->freq_index = iPwrMax;
		  peaks->power = pwrMax;
		  peaks->width_bins = peakWidth;
		  peaks->num_subpeaks = subPeakCount;
		  peaks->pwr_snr = pwrMax/pwrStdDev;
		  peaks->false_alarm_prob = -1; /* 02/09/04 gam; change false_alarm_prob_upperlimit to false_alarm_prob */
		  peaks->goodness_of_fit = -1;
		  peaks->sky_ra = params->skyPosData[params->iSky][0];
		  peaks->sky_dec = params->skyPosData[params->iSky][1];
                  /* 02/09/04 gam; remove fderiv_2-5 */
                  for(k=0;k<params->numSpinDown;k++)
                  {
		     if (k == 0) {
			peaks->fderiv_1 = params->freqDerivData[params->iDeriv][k];
		     } else if (k == 1) {
			/* peaks->fderiv_2 = params->freqDerivData[params->iDeriv][k]; */
		     } else if (k == 2) {
			/* peaks->fderiv_3 = params->freqDerivData[params->iDeriv][k]; */
		     } else if (k == 3) {
			/* peaks->fderiv_4 = params->freqDerivData[params->iDeriv][k]; */
		     } else if (k == 4) {
			/* peaks->fderiv_5 = params->freqDerivData[params->iDeriv][k]; */
		     }
                  }
                  for(k=params->numSpinDown;k<5;k++)
                  {
		     if (k == 0) {
			peaks->fderiv_1 = 0;
		     } else if (k == 1) {
			/* peaks->fderiv_2 = 0; */
		     } else if (k == 2) {
			/* peaks->fderiv_3 = 0; */
		     } else if (k == 3) {
			/* peaks->fderiv_4 = 0; */
		     } else if (k == 4) {
			/* peaks->fderiv_5 = 0; */
		     }
                  }
		  peaks->pw_mean_thissum = pwrMean;     /* 02/11/04 gam; set this and next value. */
		  peaks->pw_stddev_thissum = pwrStdDev;
		} else {
                  rejectedCount++; /* 03/02/04 gam */
		} /* END if (outputThisPeak) */
	} else {
	  findMorePeaks = 0; /* No more peaks to find */
	}
    }

    /* 03/02/04 gam; set outputs; check if we updated values are wanted and can be computed */
    binCount = oneSUM->data->length - binCount; /* 03/02/04 gam; change to number of bins that do not belong to a peak */
    if (params->updateMeanStdDev && binCount > 0) {
      pwrSum = 0;
      pwrSum2 = 0;
      for(i=0;i<oneSUM->data->length;i++) {
        /* if (binFlag[i] != 1) */ /* 05/19/05 gam */
        if (binFlag[i] != 0) {
          pwrSum += oneSUM->data->data[i];
          pwrSum2 += oneSUM->data->data[i]*oneSUM->data->data[i];
        }
      }
      pwrMean = pwrSum/((REAL4)binCount);   /* Mean pwr */
      /* Standard Deviation of the power = sqrt of the variance */
      pwrStdDev = pwrSum2/((REAL4)binCount) - pwrMean*pwrMean;
      if ( pwrStdDev > 0.0 ) {
        pwrStdDev = sqrt( pwrStdDev );
      } else {
          pwrStdDev = -1.0; /* Should not happen except due to round off error in test data */
      }
      *(outputs->pwMeanWithoutPeaks) = pwrMean;
      *(outputs->pwStdDevWithoutPeaks) = pwrStdDev;      
    } else {
      *(outputs->pwMeanWithoutPeaks) = -1;   /* indicates no meaningful value could be found */
      *(outputs->pwStdDevWithoutPeaks) = -1; /* indicates no meaningful value could be found */
    }
    *(outputs->binsWithoutPeaks) = binCount;
    *(outputs->acceptedEventCount) = eventCount;
    *(outputs->rejectedEventCount) = rejectedCount;
    
    LALFree(binFlag);
    /* LALFree(powers); */ /* 02/11/04 gam; used to be amplitudes; no longer needed */
}

/* 01/31/04 gam; Loop over DECs. For each DEC set up deltaRA(DEC) = deltaRA(0)/cos(DEC), where deltaRA(0) = params->deltaRA. */
/*               Handle the Celestial Pole and other cases so at least one sky position is done at each DEC */
void CountOrAssignSkyPosData(REAL8 **skyPosData, INT4 *numSkyPosTotal, BOOLEAN returnData, StackSlideSkyPatchParams *params)
{
   INT4 i = 0;
   INT4 iRA = 0;
   INT4 iDec = 0;
   REAL8 tmpDEC = 0.0;
   REAL8 cosTmpDEC = 0.0;
   REAL8 tmpDeltaRA = 0.0;
   INT4  tmpNumRA = 0;

   #ifdef INCLUDE_DEBUG_SKYANDSPINDOWNGRID_CODE
    if ((params->debugOptionFlag & 4) > 0 ) {
      fprintf(stdout,"\nSky Grid Info:\n");
      fprintf(stdout,"returnData = %i \n",returnData);
      fflush(stdout);
    }
   #endif

   for(iDec=0;iDec<params->numDec;iDec++) {
        tmpDEC = params->startDec + iDec*params->deltaDec;
	cosTmpDEC = cos(tmpDEC);
	if (cosTmpDEC != 0.0) {
            tmpDeltaRA = params->deltaRA/cosTmpDEC;
	    if ( (tmpDeltaRA != 0.0) && (params->stopRA > params->startRA) ) {
	       tmpNumRA = ceil((params->stopRA - params->startRA)/tmpDeltaRA);
	       tmpDeltaRA = (params->stopRA - params->startRA)/((REAL8)tmpNumRA); /* 09/16/05 gam */
	       if (tmpNumRA < 1) {
	          tmpNumRA = 1;  /* Always do at least one point in the Sky for each DEC */
	       }
	    } else {
	       tmpNumRA = 1; /* Always do at least one point in the Sky for each DEC */
	    }
	} else {
	    tmpDeltaRA = 0.0; /* We are at the North or South Celestial Pole */
	    tmpNumRA = 1;    /* Always do at least one point in the Sky for each DEC including the poles. */
	}
        #ifdef INCLUDE_DEBUG_SKYANDSPINDOWNGRID_CODE
          if ((params->debugOptionFlag & 4) > 0 ) {
	    fprintf(stdout,"tmpDeltaRA = %23.16f, tmpNumRA = %i \n",tmpDeltaRA,tmpNumRA);
	    fflush(stdout);
          }
        #endif
        for(iRA=0;iRA<tmpNumRA;iRA++) {
            if (returnData) {
	      skyPosData[i][0] = params->startRA + iRA*tmpDeltaRA;
	      skyPosData[i][1] = tmpDEC;
              #ifdef INCLUDE_DEBUG_SKYANDSPINDOWNGRID_CODE
                if ((params->debugOptionFlag & 4) > 0 ) {
	          fprintf(stdout,"RA = skyPosData[%i][0] = %23.16f, DEC = skyPosData[%i][1] = %23.16f \n",i,skyPosData[i][0],i,skyPosData[i][1]);
	          fflush(stdout);
                }
              #endif
	    }
	    i++; /* increment up to number of params->numSkyPosTotal total. */
	}
   }
   *numSkyPosTotal = i; /* Always return number of sky positions */
   #ifdef INCLUDE_DEBUG_SKYANDSPINDOWNGRID_CODE
     if ((params->debugOptionFlag & 4) > 0 ) {
       fprintf(stdout,"numSkyPosTotal= %i \n",*numSkyPosTotal);
       fflush(stdout);
     }
   #endif   
}

#ifdef INCLUDE_INTERNALLALINITBARYCENTER
 /* Copied from LALInitBarycenter.c in the LAL support package.  Author: Curt Cutler */
 void InternalLALInitBarycenter(LALStatus *stat, EphemerisData *edat)
 {

    FILE *fp1, *fp2; /* fp1 is table of Earth location; fp2 is for Sun*/

    INT4 j; /*dummy index*/
    INT4 gpsYr;  /*gpsYr + leap is the time on the GPS clock
                          at first instant of new year, UTC; equivalently
                          leap is # of leap secs added between Jan.6, 1980 and
                          Jan. 2 of given year */

/*  07/01/02 gam Comment out since LALINITBARYCENTERC not defined */
    INITSTATUS(stat);
    ATTATCHSTATUSPTR(stat);       

/*  07/01/02 gam  Change for local stand-alone code */
    /* fp1 = LALOpenDataFile(edat->ephiles.earthEphemeris);
    fp2 = LALOpenDataFile(edat->ephiles.sunEphemeris); */
    fp1 = fopen(edat->ephiles.earthEphemeris, "r");
    fp2 = fopen(edat->ephiles.sunEphemeris, "r");

    /*
    fp1 = fopen(edat->ephiles.earthEphemeris,"r");
    fp2 = fopen(edat->ephiles.sunEphemeris,"r");
    */

    /* CHECK THAT fp1 and fp2 are not NULL: */
    if ( ( fp1 == NULL ) || ( fp2 == NULL ) ) {
      /*  07/01/02 gam Comment out since LALINITBARYCENTERC not defined */
      /* ABORT (stat, LALINITBARYCENTERH_EOPEN, LALINITBARYCENTERH_MSGEOPEN); */
      return;
    }

/*reading first line of each file */

	 fscanf(fp1,"%d %le %d\n", &gpsYr, &edat->dtEtable, &edat->nentriesE);
	 fscanf(fp2,"%d %le %d\n", &gpsYr, &edat->dtStable, &edat->nentriesS);

     edat->ephemE  = (PosVelAcc *)LALMalloc(edat->nentriesE*sizeof(PosVelAcc));
     edat->ephemS  = (PosVelAcc *)LALMalloc(edat->nentriesS*sizeof(PosVelAcc));

/*first column in earth.dat or sun.dat is gps time--one long integer
  giving the number of secs that have ticked since start of GPS epoch
  +  on 1980 Jan. 6 00:00:00 UTC
*/
       for (j=0; j < edat->nentriesE; ++j){

	 /* check return value of fscanf */
	 fscanf(fp1,"%le %le %le %le %le %le %le %le %le %le\n",
		&edat->ephemE[j].gps,&edat->ephemE[j].pos[0],
		&edat->ephemE[j].pos[1],&edat->ephemE[j].pos[2],
		&edat->ephemE[j].vel[0],&edat->ephemE[j].vel[1],
		&edat->ephemE[j].vel[2],&edat->ephemE[j].acc[0],
		&edat->ephemE[j].acc[1],
		&edat->ephemE[j].acc[2] );
       }

       for (j=0; j < edat->nentriesS; ++j){
	 fscanf(fp2,"%le %le %le %le %le %le %le %le %le %le\n",
		&edat->ephemS[j].gps,&edat->ephemS[j].pos[0],
		&edat->ephemS[j].pos[1],&edat->ephemS[j].pos[2],
		&edat->ephemS[j].vel[0],&edat->ephemS[j].vel[1],
		&edat->ephemS[j].vel[2],&edat->ephemS[j].acc[0],
		&edat->ephemS[j].acc[1],
		&edat->ephemS[j].acc[2] );
       }


/*Test to make sure last entries for gpsE,S are
  reasonable numbers; specifically, checking that they are
  of same order as t2000
  --within factor e
Machine dependent: to be fixed!

    if ( (fabs(log(1.e0*(edat->ephemE[edat->nentriesE -1].gps)/t2000)) > 1.e0)
        ||(fabs(log(1.e0*(edat->ephemS[edat->nentriesS -1].gps)/t2000)) > 1.e0) ){
      LALFree(edat->ephemE);
      LALFree(edat->ephemS);
      ABORT(stat,LALINITBARYCENTERH_EEPHFILE, LALINITBARYCENTERH_MSGEEPHFILE);
    }
*/
	fclose(fp1);
        fclose(fp2);

	/*curt: is below the right return???*/
        /*  07/01/02 gam Comment out since LALINITBARYCENTERC not defined */
	
	DETATCHSTATUSPTR(stat);
	RETURN(stat);
	
 }
#endif

/* 09/12/05 gam; function that saves inverse medians for StackSlide style weighting of STKs */
void SaveInverseMedians(REAL4Vector **inverseMedians, REAL4Vector *medians, INT4 k, INT4 mediansOffset1, INT4 mediansOffset2, INT4 mediansLengthm1, INT4 nBinsPerSTK)
{
          INT4 i;

          for(i=0;i<mediansOffset1;i++) {
                inverseMedians[k]->data[i] = 1.0/medians->data[0];
          }

          for(i=mediansOffset1;i<mediansOffset2;i++) {
               inverseMedians[k]->data[i] = 1.0/medians->data[i-mediansOffset1];
          }
          
          for(i=mediansOffset2;i<nBinsPerSTK;i++) {
              inverseMedians[k]->data[i] = 1.0/medians->data[mediansLengthm1];
          }
} /* END SaveInverseMedians */

/* 09/12/04 gam; weight STKs with inverse mediasl for StackSlide style weighting of STKs */
void WeightSTKsWithInverseMedians(REAL4FrequencySeries **STKData, REAL4Vector **inverseMedians, INT4 numSTKs, INT4 nBinsPerSTK)
{
    INT4 i,k;
    for(k=0;k<numSTKs;k++) {
        for(i=0;i<nBinsPerSTK;i++) {
            STKData[k]->data->data[i] = inverseMedians[k]->data[i] * STKData[k]->data->data[i];
        }
    }
} /* END WeightSTKsWithoutBeamPattern */

/* 10/28/04 gam; function that saves inverse square medians for weighting STKs */
void SaveInverseSquareMedians(REAL4Vector **inverseSquareMedians, REAL4Vector *sumInverseSquareMedians, INT2 weightFlag, REAL4Vector *medians, INT4 k, INT4 mediansOffset1, INT4 mediansOffset2, INT4 mediansLengthm1, INT4 nBinsPerSTK)
{

          INT4 i;

          for(i=0;i<mediansOffset1;i++) {
                inverseSquareMedians[k]->data[i] = 1.0/(medians->data[0] * medians->data[0]);
          }

          for(i=mediansOffset1;i<mediansOffset2;i++) {
               inverseSquareMedians[k]->data[i] = 1.0/(medians->data[i-mediansOffset1] * medians->data[i-mediansOffset1]);
          }
          
          for(i=mediansOffset2;i<nBinsPerSTK;i++) {
              inverseSquareMedians[k]->data[i] = 1.0/(medians->data[mediansLengthm1] * medians->data[mediansLengthm1]);
          }
                    
          /* NOTE: it is assumed that sumInverseSquareMedians have been initialized to zero */
          if ( ( (weightFlag & 2) > 0 ) || ( (weightFlag & 4) > 0 ) ) {
             /* Just continue; will have to compute later using F_+ or F_x for each point on the sky */
          } else {
             for(i=0;i<nBinsPerSTK;i++) {                    
                   sumInverseSquareMedians->data[i] += inverseSquareMedians[k]->data[i];
             }                      
          }
} /* END SaveInverseSquareMedians */

/* 10/28/04 gam; savSTKDATA for reuse with powerFlux style weighting of STKs for each sky position */                        
void SaveSTKData(REAL4Vector ** savSTKData, REAL4FrequencySeries **STKData, INT4 numSTKs, INT4 nBinsPerSTK)
{
     INT4 i,k;
     for(k=0;k<numSTKs;k++) {
         for(i=0;i<nBinsPerSTK;i++) {
             savSTKData[k]->data[i] = STKData[k]->data->data[i];
         }
     }
} /* END SaveSTKData */

/* 10/28/04 gam; apply powerFlux style weights */
void WeightSTKsWithoutBeamPattern(REAL4FrequencySeries **STKData, REAL4Vector **inverseSquareMedians, REAL4Vector *sumInverseSquareMedians, INT4 numSTKs, INT4 nBinsPerSTK)
{
    INT4 i,k;
    for(k=0;k<numSTKs;k++) {
        for(i=0;i<nBinsPerSTK;i++) {
            STKData[k]->data->data[i] = inverseSquareMedians[k]->data[i] * STKData[k]->data->data[i] / sumInverseSquareMedians->data[i];
        }
    }
} /* END WeightSTKsWithoutBeamPattern */

/* 10/28/04 gam; apply powerFlux style weights including detector beam pattern response */
void WeightSTKsIncludingBeamPattern(REAL4FrequencySeries **STKData,
           REAL4Vector ** savSTKData,
           REAL4Vector **inverseSquareMedians,
           REAL4Vector *sumInverseSquareMedians,
           REAL4Vector *detResponseTStampMidPts,
           INT4 numSTKs, INT4 nBinsPerSTK, REAL8 tSTK)
{
    INT4 i,k;
    REAL4 detRes4 = 0; /* 4th power of detector response; note that detResponseTStampMidPts is aleady the square of F_+ or F_x*/
        
    for(i=0;i<nBinsPerSTK;i++) {
        sumInverseSquareMedians->data[i] = 0.0; /* initialize */
    }
           
    for(k=0;k<numSTKs;k++) {
        detRes4 = detResponseTStampMidPts->data[k] * detResponseTStampMidPts->data[k];
        for(i=0;i<nBinsPerSTK;i++) {
            sumInverseSquareMedians->data[i] += detRes4*inverseSquareMedians[k]->data[i];
        }
    }
    
    for(k=0;k<numSTKs;k++) {
        for(i=0;i<nBinsPerSTK;i++) {
            STKData[k]->data->data[i] = detResponseTStampMidPts->data[k] *
                   inverseSquareMedians[k]->data[i] * savSTKData[k]->data[i] / sumInverseSquareMedians->data[i];
        }
    }
} /* END WeightSTKsIncludingBeamPattern */


/* 10/28/04 gam; get squared detector response F_+^2 or F_x^2 for one sky position, one polarization angle, for midpoints of a timeStamps */
void GetDetResponseTStampMidPts(LALStatus *stat, REAL4Vector *detResponseTStampMidPts, LIGOTimeGPS *timeStamps, INT4 numSTKs, REAL8 tSTK,
     LALDetector *cachedDetector, REAL8 *skyPosData, REAL8 orientationAngle, CoordinateSystem coordSystem, INT2 plusOrCross)
{
    INT4 i;
    LALDetAMResponse response;  /* output of LALComputeDetAMResponse */
    LALDetAndSource      *das;  /* input for LALComputeDetAMResponse */
    LALGPSandAcc   timeAndAcc;  /* input for LALComputeDetAMResponse */
    REAL8 halftSTK = 0.5*tSTK;  /* half the time of one STK */    
    LIGOTimeGPS midTS;          /* midpoint time for an STK */
    
    INITSTATUS(stat);
    ATTATCHSTATUSPTR(stat);       
    
    /* Set up das, the Detector and Source info */
    das = (LALDetAndSource *)LALMalloc(sizeof(LALDetAndSource));
    das->pSource = (LALSource *)LALMalloc(sizeof(LALSource));
    das->pDetector = cachedDetector;    
    /* das->pSource->equatorialCoords.latitude = skyPosData[0];
    das->pSource->equatorialCoords.longitude = skyPosData[1]; */
    /* 11/18/04 gam; in GetDetResponseTStampMidPts latitute is DEC and longitude is RA, or course! */
    das->pSource->equatorialCoords.longitude = skyPosData[0];
    das->pSource->equatorialCoords.latitude = skyPosData[1];

    das->pSource->orientation = orientationAngle;  
    das->pSource->equatorialCoords.system = coordSystem;

    /* loop that calls LALComputeDetAMResponse to find F_+ and F_x at the midpoint of each SFT for ZERO Psi */
    for(i=0; i<numSTKs; i++) {
      /* Find mid point from timestamp, half way through SFT. */
      TRY ( LALAddFloatToGPS (stat->statusPtr, &midTS, &(timeStamps[i]), halftSTK), stat);
      timeAndAcc.gps=midTS;
      TRY ( LALComputeDetAMResponse(stat->statusPtr, &response, das, &timeAndAcc), stat);
      if (plusOrCross > 0) {
         detResponseTStampMidPts->data[i] = response.plus * response.plus;
      } else {
         detResponseTStampMidPts->data[i] = response.cross * response.cross;
      }
    }
    LALFree(das->pSource);
    LALFree(das);
  
    DETATCHSTATUSPTR(stat);
    RETURN(stat);
}  

/* 11/01/04 gam; if (params->weightFlag & 8) > 0 rescale STKs with threshold5 to prevent dynamic range issues. */
/* void RescaleSTKData(REAL4FrequencySeries **STKData, INT4 numSTKs, INT4 nBinsPerSTK,REAL4 RescaleFactor)
{
     INT4 i,k;
     for(k=0;k<numSTKs;k++) {
         for(i=0;i<nBinsPerSTK;i++) {
             STKData[k]->data->data[i] *= RescaleFactor;
         }
     }     
} */ /* END RescaleSTKData */ /* 10/20/05 gam */

/* 10/20/05 gam */
void CheckDynamicRangeAndRescaleBLKData(LALStatus *status, REAL4 *blkRescaleFactor, FFT **BLKData, INT4 numBLKs, INT2 weightFlag)
{

  INT4 i,j, nBinsPerBLK, count;
  REAL4 minFabsBLKData, maxFabsBLKData, sumFabsBLKData, meanFabsBLKData, temp;

  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);

  nBinsPerBLK = BLKData[0]->fft->data->length;
  count = 0;
  minFabsBLKData = 1.0;
  maxFabsBLKData = 0.0;
  sumFabsBLKData = 0.0;
  for(i=0;i<numBLKs;i++) {
      for(j=0;j<nBinsPerBLK;j++) {
          temp = ( fabs(BLKData[i]->fft->data->data[j].re) + fabs(BLKData[i]->fft->data->data[j].im) ) /2.0;
          if ( (temp > 0.0) && (temp < minFabsBLKData) ) {
              minFabsBLKData = temp;
          }
          if ( temp > maxFabsBLKData ) {
              maxFabsBLKData = temp;
          }
          sumFabsBLKData += temp;
          count++;
      }
  }
  meanFabsBLKData = sumFabsBLKData/((REAL4) count);

  #ifdef DEBUG_CHECKDYNAMICRANGEANDRESCALEBLKDATA
     fprintf(stdout, "minimum nonzero absolute value of BKL data = %23.16e\n",minFabsBLKData);
     fprintf(stdout, "maximum nonzero absolute value of BKL data = %23.16e\n",maxFabsBLKData);
     fprintf(stdout, "mean absolute value of BKL data = %23.16e\n",meanFabsBLKData);
     fflush(stdout);
  #endif

  if ( (weightFlag & 8) > 0 )  {
     *blkRescaleFactor = 1.0/meanFabsBLKData;
     /* Rescale the BLK data with *blkRescaleFactor */
     for(i=0;i<numBLKs;i++) {
         for(j=0;j<nBinsPerBLK;j++) {
            BLKData[i]->fft->data->data[j].re = (*blkRescaleFactor) * BLKData[i]->fft->data->data[j].re;
            BLKData[i]->fft->data->data[j].im = (*blkRescaleFactor) * BLKData[i]->fft->data->data[j].im;
         }
     }     
  } else {
    *blkRescaleFactor = 1.0;
     /* Since we work with absolute squares of BKL data, make sure we are not in danger of underflows or overflows */
     if ( (meanFabsBLKData <= STACKSLIDEUNDERFLOWDANGER) || (maxFabsBLKData >=  STACKSLIDEOVERFLOWDANGER) ) {
        ABORT( status, DRIVESTACKSLIDEH_EDYNAMICRANGE , DRIVESTACKSLIDEH_MSGEDYNAMICRANGE );
     }
  }

  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);
}

/* 02/25/05 gam; utility for printing one SUM to a file */
void printOneStackSlideSUM( const REAL4FrequencySeries *oneSUM,
                  INT2                       outputSUMFlag,
                  CHAR                       *outputFile,
                  INT4                       kSUM,
                  REAL8                      f0SUM,
                  REAL8                      dfSUM,
                  INT4                       nBinsPerSUM,
                  INT4                       numSUMsTotal
)
{   
    INT4 k;
    FILE *outfp = NULL;
    CHAR *baseOutputFile = NULL;  /* Used only if numSUMsTotal > 1 */
    CHAR fileNumber[7];           /* Fixed size string that holds .00000, .00001, .00002, etc... */

    /* set up the file name and open the file */
    if (numSUMsTotal > 1) {
       baseOutputFile = (CHAR *) LALMalloc( (strlen(outputFile) + 7) * sizeof(CHAR) );
       strcpy(baseOutputFile,outputFile);
       sprintf(fileNumber,".%05d",kSUM);
       strcat(baseOutputFile,fileNumber);
       outfp = fopen(baseOutputFile, "w");
    } else {
       outfp = fopen(outputFile, "w");
    }

    /* print out the data based on the outputSUMFlag option */
    if ((outputSUMFlag & 1) > 0) {
       for(k=0;k<nBinsPerSUM; k++) {
           fprintf(outfp,"%g \n",oneSUM->data->data[k]);
           fflush(outfp);
       }
    } else {
       for(k=0;k<nBinsPerSUM; k++) {
           fprintf(outfp,"%18.10f %g \n",f0SUM+(REAL8)k*dfSUM, oneSUM->data->data[k]);
           fflush(outfp);
       }
    }
   
    /* close the file and clean up */
    fclose(outfp);   
    if (numSUMsTotal > 1) {
       LALFree(baseOutputFile);
    }
} /* END void printOneSUM */

/* 05/13/05 gam; Add function FindAveEarthAcc that finds aveEarthAccVec, the Earth's average acceleration vector during the analysis time. */
void FindAveEarthAcc(LALStatus *status, REAL8 *aveEarthAccVec, REAL8 startTime, REAL8 endTime, const EphemerisData *edat)
{

  INT4 i, iStart, iEnd;
  REAL8 count;

  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);
  
  iStart = floor( ( (startTime - edat->ephemE[0].gps)/edat->dtEtable ) + 0.5); /* finding starting Earth table entry */
  iEnd = floor( ( (endTime - edat->ephemE[0].gps)/edat->dtEtable ) + 0.5);     /* finding ending Earth table entry   */
  count = (REAL8)(iEnd - iStart + 1);

  if ( (count <= 0) || (iStart < 0) || (iEnd < 0) || (iEnd >= edat->nentriesE) ) {
     ABORT( status, DRIVESTACKSLIDEH_EAVEEARTHACCVEC, DRIVESTACKSLIDEH_MSGEAVEEARTHACCVEC);
  } 

  aveEarthAccVec[0] = 0.0;
  aveEarthAccVec[1] = 0.0;
  aveEarthAccVec[2] = 0.0;
  for (i=iStart; i<=iEnd; i++) {
     aveEarthAccVec[0] += edat->ephemE[i].acc[0];
     aveEarthAccVec[1] += edat->ephemE[i].acc[1];
     aveEarthAccVec[2] += edat->ephemE[i].acc[2];
  }
  aveEarthAccVec[0] = aveEarthAccVec[0]/count;
  aveEarthAccVec[1] = aveEarthAccVec[1]/count;
  aveEarthAccVec[2] = aveEarthAccVec[2]/count;
  
  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);

} /* END void FindAveEarthAcc */

/* 05/13/05 gam; Add function FindLongLatFromVec that find for a vector that points from the center to a position on a sphere, the latitude and longitude of this position */
void FindLongLatFromVec(LALStatus *status, REAL8 *longitude, REAL8 *latitude, const REAL8 *vec)
{

  REAL8 vLength;
  
  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);
  
  vLength = sqrt( vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2] );
  if (vLength <= 0.0) {
     ABORT( status, DRIVESTACKSLIDEH_ELONGLATFROMVEC, DRIVESTACKSLIDEH_MSGELONGLATFROMVEC);
  }
  
  *latitude = asin( vec[2] / vLength );
  /* *longitude = atan2(vec[1],vec[0]) + ((REAL8)LAL_PI); */ /* 05/22/05 gam; Not correct; fixed below */
  *longitude = atan2(vec[1],vec[0]);
  if (*longitude < 0.0) {
     *longitude += ((REAL8)LAL_TWOPI);
  }
  
  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);
}

/* 05/13/05 gam; Add function RotateSkyCoordinates that transforms longIn and latIn to longOut, latOut as related by three rotations. */
void RotateSkyCoordinates(LALStatus *status, REAL8 *longOut, REAL8 *latOut, REAL8 longIn, REAL8 latIn, REAL8 longPole, REAL8 latPole, REAL8 longOffset)
{
  /*  Transforms longIn and latIn to longOut, latOut as related by three rotations.
      The North pole of the input coordinate frame lies at
      (longPole, latPole) in the output coordinate frame. 
      The input coordinate system has axes: x_in, y_in, z_in.
      The output coordinate system has axes: x_out, y_out, z_out.      
      The input coordinate system is rotated with respect output
      coordinate system by the following Euler Angle rotations:
        1. Rotate by longPole - 3*pi/2 about the z_out-axis.
        2. Rotate by pi/2 -latPol about the x_out-axis.
           That is rotate about the line of nodes; the x_out-axis
           points to the ascending nodes as per the right-hand rule.
        3. Rotate about the z_in-axis by longOffset.
  */

  REAL8 longInFromAscendingNode, Xtmp, Ytmp, cosLatIn, sinLatIn, cosLatPole, sinLatPole, cosLatOut;
  
  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);

  cosLatIn   = cos(latIn);
  sinLatIn   = sin(latIn);
  cosLatPole = cos(latPole);
  sinLatPole = sin(latPole);

  /* Invert the last rotation; this moves the x_in-axis back to the ascending node */  
  longInFromAscendingNode = longIn + longOffset;

  /* Compute the output latitude using spherical trig */
  *latOut = asin ( sinLatPole * sinLatIn + cosLatPole * cosLatIn * sin(longInFromAscendingNode) );
  
  cosLatOut  = cos(*latOut);

  if ( cosLatOut == 0.0 ) {
     *longOut = 0.0;  /* Output latitude is at a pole; longitude can be anything, so chose zero */
  } else {
     if (cosLatPole == 0.0) {
       /* The output longitude is just a simple rotation about the z_out-axis */
       *longOut = longIn + longPole - 3.0*LAL_PI/2.0;
     } else {
       /* Compute the output longitude using spherical trig */
       Xtmp = ( sinLatIn - sinLatPole * sin(*latOut) )  / ( cosLatPole * cosLatOut );
       Ytmp = cosLatIn * cos(longInFromAscendingNode) / cosLatOut;
       *longOut = longPole + atan2(Ytmp,Xtmp);
     }
     if (*longOut < 0.0) {
        *longOut += LAL_TWOPI;  /* Make sure longOut is positive */
     }
     *longOut = fmod(*longOut,LAL_TWOPI);  /* Make sure longOut is between 0 and 2pi */
  }

  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);

} /* END void RotateSkyCoordinates */
  
/* 05/13/05 gam; Add function RotateSkyPosData that rotates skyPosData using RotateSkyCoordinates */
void RotateSkyPosData(LALStatus *status, REAL8 **skyPosData, INT4 numSkyPosTotal, REAL8 longPole, REAL8 latPole, REAL8 longOffset)
{
     INT4 iSky;
     REAL8 tmpRA, tmpDEC;
     
     INITSTATUS(status);
     ATTATCHSTATUSPTR (status);

     for(iSky=0;iSky<numSkyPosTotal;iSky++) {
       RotateSkyCoordinates(status->statusPtr, &tmpRA, &tmpDEC, skyPosData[iSky][0], skyPosData[iSky][1], longPole, latPole, longOffset);
       CHECKSTATUSPTR (status);
       skyPosData[iSky][0] = tmpRA;
       skyPosData[iSky][1] = tmpDEC;
     }

     CHECKSTATUSPTR (status);
     DETATCHSTATUSPTR (status);

} /* END void RotateSkyPosData */

/* 05/14/05 gam; function that reads in line and harmonics info from file; based on SFTclean.c by Krishnan, B. */
void StackSlideGetLinesAndHarmonics(LALStatus *status, LineHarmonicsInfo *infoHarmonics, LineNoiseInfo *infoLines, REAL8 fStart, REAL8 fBand, CHAR *linesAndHarmonicsFile)
{
  LineNoiseInfo lines;
  INT4 nLines, count1, nHarmonicSets;  

  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);

  lines.lineFreq  = NULL;
  lines.leftWing  = NULL;
  lines.rightWing = NULL;

  /* FindNumberHarmonics(status->statusPtr, infoHarmonics, linesAndHarmonicsFile); */ /* 09/09/05 gam */
  LALFindNumberHarmonics(status->statusPtr, infoHarmonics, linesAndHarmonicsFile);
  
  CHECKSTATUSPTR (status);
  nHarmonicSets = infoHarmonics->nHarmonicSets;

  if (nHarmonicSets > 0) {
      infoHarmonics->startFreq = (REAL8 *)LALMalloc(nHarmonicSets * sizeof(REAL8));
      infoHarmonics->gapFreq = (REAL8 *)LALMalloc(nHarmonicSets * sizeof(REAL8));
      infoHarmonics->numHarmonics = (INT4 *)LALMalloc(nHarmonicSets * sizeof(INT4));
      infoHarmonics->leftWing = (REAL8 *)LALMalloc(nHarmonicSets * sizeof(REAL8));
      infoHarmonics->rightWing = (REAL8 *)LALMalloc(nHarmonicSets * sizeof(REAL8));

      /* ReadHarmonicsInfo(status->statusPtr, infoHarmonics, linesAndHarmonicsFile); */ /* 09/09/05 gam */
      LALReadHarmonicsInfo(status->statusPtr, infoHarmonics, linesAndHarmonicsFile);
      CHECKSTATUSPTR (status);
      
      nLines = 0;
      for (count1=0; count1 < nHarmonicSets; count1++) {
        nLines += infoHarmonics->numHarmonics[count1];
      }

      lines.nLines = nLines;
      lines.lineFreq = (REAL8 *)LALMalloc(nLines * sizeof(REAL8));
      lines.leftWing = (REAL8 *)LALMalloc(nLines * sizeof(REAL8));
      lines.rightWing = (REAL8 *)LALMalloc(nLines * sizeof(REAL8));

      /* Harmonics2Lines(status->statusPtr, &lines, infoHarmonics); */ /* 09/09/05 gam */
      LALHarmonics2Lines(status->statusPtr, &lines, infoHarmonics);
      CHECKSTATUSPTR (status);
      
      infoLines->nLines = nLines;
      infoLines->lineFreq = (REAL8 *)LALMalloc(nLines * sizeof(REAL8));
      infoLines->leftWing = (REAL8 *)LALMalloc(nLines * sizeof(REAL8));
      infoLines->rightWing = (REAL8 *)LALMalloc(nLines * sizeof(REAL8));

      /* ChooseLines(status->statusPtr, infoLines, &lines, fStart, fStart + fBand); */ /* 09/09/05 gam */
      LALChooseLines(status->statusPtr, infoLines, &lines, fStart, fStart + fBand);
      CHECKSTATUSPTR (status);

      if (nLines > 0) {
        LALFree(lines.lineFreq);
        LALFree(lines.leftWing);
        LALFree(lines.rightWing);
      }
  }

  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);

} /* END StackSlideGetLinesAndHarmonics */

/* 05/19/05 gam; set up params->sumBinMask with bins to exclude from search or Monte Carlo due to cleaning */
void StackSlideGetBinMask(LALStatus *status, INT4 *binMask, REAL8 *percentBinsExcluded, LineNoiseInfo *infoLines,
     REAL8 maxDopplerVOverC, REAL8 maxSpindownFreqShift, REAL8 maxSpinupFreqShift, REAL8 f0, REAL8 tBase, INT4 nBins)
{
  INT4 j,k,f0Bin,spindownBins,spinupBins,minBin,maxBin,binCount;
  REAL8 tBaseOverOneMinusMaxDoppler, tBaseOverOnePlusMaxDoppler;
  
  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);
  
  f0Bin = floor(f0*tBase + 0.5);
  /* 09/23/05 gam; the floor and ceil are used here to be conservative regardless of the sign of the spindown or spinup shifts */
  spindownBins = floor(maxSpindownFreqShift*tBase);  /* Usually maxSpindownFreqShift is negative */
  spinupBins = ceil(maxSpinupFreqShift*tBase);       /* Usually maxSpinupFreqShift is zero or positive */
  tBaseOverOneMinusMaxDoppler = tBase/(1.0 - maxDopplerVOverC);
  tBaseOverOnePlusMaxDoppler  = tBase/(1.0 + maxDopplerVOverC);

  for(k=0;k<infoLines->nLines;k++) {

        /* Find min and max bin that could be affected by this line */
        /* Assuming templates do not spinup, but spindown only */
        minBin = floor( (infoLines->lineFreq[k] - infoLines->leftWing[k])*tBaseOverOnePlusMaxDoppler ) - spinupBins - f0Bin;
        if (minBin < 0) minBin = 0;
        maxBin = ceil(  (infoLines->lineFreq[k] + infoLines->rightWing[k])*tBaseOverOneMinusMaxDoppler ) - spindownBins - f0Bin;
        if (maxBin >= nBins) maxBin = nBins - 1;
        
        /* Exclude bins that could be affected by this line; note sum from minBin to maxBin inclusive */
        for(j=minBin;j<=maxBin;j++) {
            binMask[j] = 0; /* exclude this bin */
        }
  }
  binCount = 0;
  for(j=0;j<nBins;j++) {
         if (binMask[j] == 0) binCount++;
  }
  *percentBinsExcluded = 100.0*((REAL8)binCount)/((REAL8)nBins);

  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);

} /* END StackSlideGetBinMask */

/* 05/14/05 gam; cleans SFTs using CleanCOMPLEX8SFT by Sintes, A.M., Krishnan, B. */  /* 07/13/05 gam; add RandomParams *randPar */
void StackSlideCleanSFTs(LALStatus *status, FFT **BLKData, LineNoiseInfo *infoLines, INT4 numBLKs, INT4 nBinsPerNRM, INT4 maxBins, RandomParams *randPar)
{
  INT4 i,j, nBinsPerBLK;
  SFTtype *oneSFT;

  INITSTATUS(status);
  ATTATCHSTATUSPTR (status);

  nBinsPerBLK = BLKData[0]->fft->data->length; /* 12/07/05 gam; moved here */

  oneSFT = NULL;
  LALCreateSFTtype (status->statusPtr, &oneSFT, ((UINT4)nBinsPerBLK));
  CHECKSTATUSPTR (status);

  for(i=0;i<numBLKs;i++) {
      /* copy the data for this BLK to oneSFT */
      oneSFT->epoch = BLKData[i]->fft->epoch;
      oneSFT->f0 = BLKData[i]->fft->f0;
      oneSFT->deltaF = BLKData[i]->fft->deltaF;
      oneSFT->data->length = BLKData[i]->fft->data->length;
      for(j=0;j<nBinsPerBLK;j++) {
           oneSFT->data->data[j].re = BLKData[i]->fft->data->data[j].re;
           oneSFT->data->data[j].im = BLKData[i]->fft->data->data[j].im;
      }
      /* CleanCOMPLEX8SFT(status->statusPtr, oneSFT, maxBins, nBinsPerNRM, infoLines, randPar); */ /* 09/09/05 gam */
      LALCleanCOMPLEX8SFT(status->statusPtr, oneSFT, maxBins, nBinsPerNRM, infoLines, randPar); /* clean this SFT */
      CHECKSTATUSPTR (status);

      #ifdef DEBUG_CLEANED_SFTS
        /* debug which bins get cleaned */
        for(j=0;j<nBinsPerBLK;j++) {
           if ( (BLKData[i]->fft->data->data[j].re != oneSFT->data->data[j].re) || (BLKData[i]->fft->data->data[j].im != oneSFT->data->data[j].im) ) {
              fprintf(stdout,"%23.16e %23.16e %23.16e %23.16e %23.16e \n",BLKData[i]->fft->f0 + ((REAL8)j)*BLKData[i]->fft->deltaF,BLKData[i]->fft->data->data[j].re,BLKData[i]->fft->data->data[j].im, oneSFT->data->data[j].re,oneSFT->data->data[j].im);
           }
        }
      #endif

      /* copy the clean SFT data back to this BLK */
      for(j=0;j<nBinsPerBLK;j++) {
           BLKData[i]->fft->data->data[j].re = oneSFT->data->data[j].re;
           BLKData[i]->fft->data->data[j].im = oneSFT->data->data[j].im;
      }
  }
  LALDestroySFTtype (status->statusPtr,&oneSFT);
  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR (status);

} /* END StackSlideCleanSFTs */

/******************************************/
/*                                        */
/* END SECTION: internal functions        */
/*                                        */
/******************************************/

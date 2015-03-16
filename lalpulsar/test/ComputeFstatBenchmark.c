/*
*  Copyright (C) 2015 Reinhard Prix
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

#include <lal/XLALError.h>
#include <lal/LALBarycenter.h>
#include <lal/LALInitBarycenter.h>
#include <lal/LogPrintf.h>
#include <lal/CWMakeFakeData.h>
#include <lal/LALConstants.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/ComputeFstat.h>
#include <lal/DetectorStates.h>
#include <lal/LFTandTSutils.h>
#include <lal/LALString.h>
#include <lal/UserInput.h>

// benchmark ComputeFstat() functions for performance and memory usage

typedef struct
{
  BOOLEAN help;			//!< output help-string */
  CHAR *FstatMethod;		//!< select which method/algorithm to use to compute the F-statistic
  REAL8 Freq;
  REAL8 f1dot;
  REAL8 FreqResolution;
  INT4 numFreqBins;
  REAL8 Tseg;
  REAL8 Tsft;
  INT4 numSegments;
} UserInput_t;

// ---------- main ----------
int
main ( int argc, char *argv[] )
{
  // ---------- handle user input ----------
  UserInput_t XLAL_INIT_DECL(uvar_s);
  UserInput_t *uvar = &uvar_s;

  uvar->FstatMethod = XLALStringDuplicate("ResampBest");
  uvar->Freq = 100;
  uvar->f1dot = -3e-9;
  uvar->FreqResolution = 3;
  uvar->numFreqBins = 50000;
  uvar->Tseg = 60 * 3600;
  uvar->numSegments = 90;
  uvar->Tsft = 1800;

  XLALregBOOLUserStruct   ( help,               'h', UVAR_HELP,     "Print this message");
  XLALregSTRINGUserStruct ( FstatMethod,          0, UVAR_OPTIONAL,  XLALFstatMethodHelpString() );
  XLALregREALUserStruct   ( Freq,                 0, UVAR_OPTIONAL,  "Search frequency in Hz" );
  XLALregREALUserStruct   ( f1dot,                0, UVAR_OPTIONAL,  "Search spindown f1dot in Hz/s" );
  XLALregREALUserStruct   ( FreqResolution,       0, UVAR_OPTIONAL,  "Frequency resolution factor 'r' such that dFreq = 1/(r*T)" );
  XLALregREALUserStruct   ( Tseg,                 0, UVAR_OPTIONAL,  "Coherent segment length" );
  XLALregINTUserStruct    ( numSegments,          0, UVAR_OPTIONAL,  "number of frequency bins to search" );
  XLALregINTUserStruct    ( numFreqBins,          0, UVAR_OPTIONAL,  "number of frequency bins to search" );
  XLALregREALUserStruct   ( Tsft,                 0, UVAR_DEVELOPER, "SFT length" );

  XLAL_CHECK ( XLALUserVarReadAllInput(argc, argv) == XLAL_SUCCESS, XLAL_EFUNC );
  if (uvar->help) {	// if help was requested, we're done here
    return XLAL_SUCCESS;
  }
  FstatMethodType FstatMethod;
  XLAL_CHECK ( XLALParseFstatMethodString ( &FstatMethod, uvar->FstatMethod ) == XLAL_SUCCESS, XLAL_EFUNC );
  REAL8 dFreq = 1.0 / ( uvar->FreqResolution * uvar->Tseg );
  REAL8 FreqBand = uvar->numFreqBins * dFreq;
  XLAL_CHECK ( uvar->numSegments >= 1, XLAL_EINVAL );
  XLAL_CHECK ( uvar->FreqResolution > 0, XLAL_EINVAL );
  XLAL_CHECK ( uvar->Freq > 0, XLAL_EINVAL );
  XLAL_CHECK ( uvar->Tseg > uvar->Tsft, XLAL_EINVAL );
  XLAL_CHECK ( uvar->Tsft > 1, XLAL_EINVAL );
  XLAL_CHECK ( uvar->numFreqBins >= 1, XLAL_EINVAL );
  fprintf ( stderr, "Tseg = %.1f d, numSegments = %" LAL_INT4_FORMAT ", Freq = %.1f Hz, f1dot = %.1e Hz/s, dFreq = %.1e Hz, numFreqBins = %" LAL_INT4_FORMAT ", FreqBand = %.2f Hz, Tsft = %.0f s\n",
            uvar->Tseg / 86400.0, uvar->numSegments, uvar->Freq, uvar->f1dot, dFreq, uvar->numFreqBins, FreqBand, uvar->Tsft );
  // ---------- end: handle user input ----------
  EphemerisData *ephem;
  REAL8 maxMem0 = XLALGetPeakHeapUsageMB();
  XLAL_CHECK ( (ephem = XLALInitBarycenter ( TEST_DATA_DIR "earth00-19-DE405.dat.gz", TEST_DATA_DIR "sun00-19-DE405.dat.gz" )) != NULL, XLAL_EFUNC );
  REAL8 memBase = XLALGetPeakHeapUsageMB();
  REAL8 memEphem = memBase - maxMem0;

  XLALPrintInfo ("mem(ephemeris) = %.1f MB\n", memEphem );

  // ----- setup injection and data parameters
  LALStringVector *detNames = NULL;
  XLAL_CHECK ( (detNames = XLALCreateStringVector ( "H1", "L1", NULL )) != NULL, XLAL_EFUNC );
  UINT4 numDetectors = detNames->length;

  LIGOTimeGPS startTime = {711595934, 0};
  LIGOTimeGPS startTime_l = startTime;
  LIGOTimeGPS endTime_l;
  SFTCatalog **catalogs;
  XLAL_CHECK ( (catalogs = XLALCalloc ( uvar->numSegments, sizeof( catalogs[0] ))) != NULL, XLAL_ENOMEM );

  for ( INT4 l = 0; l < uvar->numSegments; l ++ )
    {
      endTime_l = startTime_l;
      XLALGPSAdd( &endTime_l, uvar->Tseg );
      MultiLIGOTimeGPSVector *multiTimestamps;
      XLAL_CHECK ( (multiTimestamps = XLALMakeMultiTimestamps ( startTime_l, uvar->Tseg, uvar->Tsft, 0, numDetectors )) != NULL, XLAL_EFUNC );
      XLAL_CHECK ( (catalogs[l] = XLALMultiAddToFakeSFTCatalog ( NULL, detNames, multiTimestamps )) != NULL, XLAL_EFUNC );
      XLALDestroyMultiTimestamps ( multiTimestamps );
      startTime_l = endTime_l;
    } // for l < numSegments
  LIGOTimeGPS endTime = endTime_l;

  PulsarSpinRange XLAL_INIT_DECL(spinRange);
  LIGOTimeGPS refTime = { startTime.gpsSeconds - 2.3 * uvar->Tseg, 0 };
  spinRange.refTime = refTime;
  spinRange.fkdot[0] = uvar->Freq;
  spinRange.fkdot[1] = uvar->f1dot;
  spinRange.fkdotBand[0] = FreqBand;
  spinRange.fkdotBand[1] = 0;
  REAL8 asini = 0, Period = 0, ecc = 0;
  REAL8 minCoverFreq, maxCoverFreq;
  XLAL_CHECK ( XLALCWSignalCoveringBand ( &minCoverFreq, &maxCoverFreq, &startTime, &endTime, &spinRange, asini, Period, ecc ) == XLAL_SUCCESS, XLAL_EFUNC );

  UINT4 numBinsSFT = ceil ( (maxCoverFreq - minCoverFreq) * uvar->Tsft + 2 * 8 );
  UINT4 numSFTsPerSeg = catalogs[0]->length;
  REAL8 memSFTs = uvar->numSegments * numSFTsPerSeg * ( sizeof(SFTtype) + numBinsSFT * sizeof(COMPLEX8)) / 1e6;

  PulsarDopplerParams XLAL_INIT_DECL(Doppler);
  Doppler.refTime = refTime;
  Doppler.Alpha = 0.5;
  Doppler.Delta = 0.5;
  memcpy ( &Doppler.fkdot, &spinRange.fkdot, sizeof(Doppler.fkdot) );;
  Doppler.period = Period;
  Doppler.ecc = ecc;
  Doppler.asini = asini;

  // ----- setup optional Fstat arguments
  FstatOptionalArgs optionalArgs = FstatOptionalArgsDefaults;

  MultiNoiseFloor XLAL_INIT_DECL(injectSqrtSX);
  injectSqrtSX.length = numDetectors;
  for ( UINT4 X=0; X < numDetectors; X ++ ) {
    injectSqrtSX.sqrtSn[X] = 1;
  }
  optionalArgs.injectSqrtSX = &injectSqrtSX;
  optionalArgs.FstatMethod = FstatMethod;

  FstatWorkspace *sharedWorkspace = NULL;
  FstatInputVector *inputs;
  XLAL_CHECK ( (inputs = XLALCreateFstatInputVector ( uvar->numSegments )) != NULL, XLAL_EFUNC );
  for ( INT4 l = 0; l < uvar->numSegments; l ++ )
    {
      XLAL_CHECK ( (inputs->data[l] = XLALCreateFstatInput ( catalogs[l], minCoverFreq, maxCoverFreq, dFreq, ephem, &optionalArgs )) != NULL, XLAL_EFUNC );
      if ( l == 0 ) {
        sharedWorkspace = XLALGetSharedFstatWorkspace ( inputs->data[0] );
      }
      optionalArgs.sharedWorkspace = sharedWorkspace;
    }

  // ----- compute Fstatistics over segments
  FstatQuantities whatToCompute = (FSTATQ_2F | FSTATQ_2F_PER_DET);

  FstatResults **results;
  XLAL_CHECK ( (results = XLALCalloc ( uvar->numSegments, sizeof(results[0]))) != NULL, XLAL_ENOMEM );
  REAL8 tauFSumBuffered = 0, tauFSumUnbuffered = 0;
  for ( INT4 l = 0; l < uvar->numSegments; l ++ )
    {
      // call it once to initialize buffering, don't count this time
      REAL8 tic = XLALGetTimeOfDay();
      XLAL_CHECK ( XLALComputeFstat ( &results[l], inputs->data[l], &Doppler, uvar->numFreqBins, whatToCompute ) == XLAL_SUCCESS, XLAL_EFUNC );
      REAL8 toc = XLALGetTimeOfDay();
      tauFSumUnbuffered += ( toc - tic );
      // now call it with full buffering to get converged runtime per template (assuming many templates per skypoint or per binary params)
      tic = XLALGetTimeOfDay();
      XLAL_CHECK ( XLALComputeFstat ( &results[l], inputs->data[l], &Doppler, uvar->numFreqBins, whatToCompute ) == XLAL_SUCCESS, XLAL_EFUNC );
      toc = XLALGetTimeOfDay();
      tauFSumBuffered += (toc - tic);
    } // for l < numSegments
  REAL8 tauF1Buffered   = tauFSumBuffered / ( uvar->numSegments * uvar->numFreqBins * numDetectors );
  REAL8 tauF1Unbuffered = tauFSumUnbuffered / ( uvar->numSegments * uvar->numFreqBins * numDetectors );
  REAL8 memMaxCompute = XLALGetPeakHeapUsageMB() - memBase;

  fprintf (stderr, "%-15s: tauF1Buffered = %.1e s (=>tauF0SFT = %1.e s), tauF1Unbuffered = %.1e s, memSFTs = %.1f MB, memMaxCompute = %.1f MB\n",
           XLALGetFstatMethodName ( FstatMethod ), tauF1Buffered, numDetectors * tauF1Buffered / numSFTsPerSeg, tauF1Unbuffered, memSFTs, memMaxCompute  );

  // ----- free memory ----------
  XLALDestroyFstatInputVector ( inputs );
  for ( INT4 l = 0; l < uvar->numSegments; l ++ )
    {
      XLALDestroySFTCatalog ( catalogs[l] );
      XLALDestroyFstatResults ( results[l] );
    }
  XLALFree ( catalogs );
  XLALFree ( results );
  XLALDestroyFstatWorkspace ( sharedWorkspace );
  XLALDestroyUserVars();
  XLALDestroyStringVector ( detNames );
  XLALDestroyEphemerisData ( ephem );

  LALCheckMemoryLeaks();

  return XLAL_SUCCESS;

} // main()

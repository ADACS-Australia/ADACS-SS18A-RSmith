/*
*  Copyright (C) 2014 Reinhard Prix
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
#include <lal/UniversalDopplerMetric.h>
#include <lal/LogPrintf.h>
#include <lal/CWMakeFakeData.h>
#include <lal/LALConstants.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/ComputeFstat.h>
#include <lal/DetectorStates.h>
#include <lal/LFTandTSutils.h>
#include <lal/LALString.h>

// basic consistency checks of the ComputeFstat module: compare F-stat results for all
// *available* Fstat methods against each other

static int compareFstatResults ( const FstatResults *result1, const FstatResults *result2 );

// ---------- main ----------
int
main ( int argc, char *argv[] )
{
  XLAL_CHECK ( argc == 1, XLAL_EINVAL, "No input arguments allowed.\n" );
  XLAL_CHECK ( argv != NULL, XLAL_EINVAL );

  // ----- load ephemeris
  EphemerisData *ephem;
  XLAL_CHECK ( (ephem = XLALInitBarycenter ( TEST_DATA_DIR "earth00-19-DE405.dat.gz", TEST_DATA_DIR "sun00-19-DE405.dat.gz" )) != NULL, XLAL_EFUNC );

  // ----- setup injection and data parameters
  LALStringVector *detNames = NULL;
  XLAL_CHECK ( (detNames = XLALCreateStringVector ( "H1", "L1", NULL )) != NULL, XLAL_EFUNC );
  UINT4 numDetectors = detNames->length;

  // generate and assume some gaussian noise floors
  MultiNoiseFloor XLAL_INIT_DECL(injectSqrtSX);
  MultiNoiseFloor XLAL_INIT_DECL(assumeSqrtSX);
  injectSqrtSX.length = numDetectors;
  assumeSqrtSX.length = numDetectors;
  for ( UINT4 X = 0; X < numDetectors; X ++ )
    {
      injectSqrtSX.sqrtSn[X] = 1.0 + 2.0*X;
      assumeSqrtSX.sqrtSn[X] = 1.0 + 2.0*X;
    }

  LIGOTimeGPS startTime = {711595934, 0};
  REAL8 Tspan = 20 * 3600;
  LIGOTimeGPS endTime = startTime;
  XLALGPSAdd( &endTime, Tspan );
  REAL8 Tsft = 1800;

  LIGOTimeGPS refTime = { startTime.gpsSeconds - 2.3 * Tspan, 0 };

  MultiLIGOTimeGPSVector *multiTimestamps;
  XLAL_CHECK ( ( multiTimestamps = XLALCalloc ( 1, sizeof(*multiTimestamps))) != NULL, XLAL_ENOMEM );
  XLAL_CHECK ( ( multiTimestamps->data = XLALCalloc ( numDetectors, sizeof(multiTimestamps->data[0]) )) != NULL, XLAL_ENOMEM );
  multiTimestamps->length = numDetectors;
  LIGOTimeGPS startTimeX = startTime;
  for ( UINT4 X=0; X < numDetectors; X ++ )
    {
      XLAL_CHECK ( (multiTimestamps->data[X] = XLALMakeTimestamps ( startTimeX, Tspan, Tsft, 0 ) ) != NULL, XLAL_EFUNC );
      XLALGPSAdd ( &startTimeX, 0.5 * Tspan );	// shift start-times by 1/2 Tspan for each detector
      Tspan *= 2.0;
    } // for X < numDetectors

  // shift a few timestamps around to create gaps
  UINT4 numSFTsPerDet = multiTimestamps->data[0]->length;
  multiTimestamps->data[0]->data[numSFTsPerDet-1].gpsSeconds += 10000;
  multiTimestamps->data[0]->data[numSFTsPerDet-2].gpsSeconds += 5000;
  multiTimestamps->data[1]->data[0].gpsSeconds -= 10000;
  multiTimestamps->data[1]->data[1].gpsSeconds -=  5000;

  SFTCatalog *catalog;
  XLAL_CHECK ( (catalog = XLALMultiAddToFakeSFTCatalog ( NULL, detNames, multiTimestamps )) != NULL, XLAL_EFUNC );

  // ----- CW sources to injet ----------
  REAL8 Freq = 100.0;

  PulsarParamsVector *injectSources;
  XLAL_CHECK ( (injectSources = XLALCreatePulsarParamsVector(1)) != NULL, XLAL_EFUNC );

  injectSources->data[0].Amp.h0   = 1;
  injectSources->data[0].Amp.cosi = 0.5;
  injectSources->data[0].Amp.psi  = 0.1;
  injectSources->data[0].Amp.phi0 = 1.2;

  REAL8 asini = 0; // 1.4;	// sco-X1 like
  REAL8 Period = 0; // 19 * 3600;// sco-X1 like
  REAL8 ecc = 0; // 0.1;	// much larger than ScoX1
  PulsarDopplerParams XLAL_INIT_DECL(Doppler);
  Doppler.Alpha = 0.5;
  Doppler.Delta = -0.5;
  Doppler.fkdot[0] = Freq;
  Doppler.fkdot[1] = -1e-9;
  Doppler.refTime = refTime;

  Doppler.asini = asini;
  Doppler.ecc = ecc;
  Doppler.tp = startTime;
  Doppler.period = Period;
  Doppler.argp = 0.5;

  injectSources->data[0].Doppler = Doppler;

  REAL8 dFreq = 0.1 / Tspan;		// 10x finer than native FFT resolution
  REAL8 mis = 0.5;
  REAL8 df1dot = sqrt( 720.0 * mis ) / (LAL_PI * Tspan * Tspan);	// metric (f-projected) stepsize for given mismatch mis
  REAL8 dSky = 1e4 / (Freq * Tspan);	// rough estimate of a 'metric' sky step, eg. Eq.(118) in \cite Prix07

  REAL8 dPeriod = 3600;
  UINT4 numFreqBins = 1000;

  UINT4 numf1dotPoints  = 2;
  UINT4 numSkyPoints    = 2;
  UINT4 numPeriodPoints = 2;

  PulsarSpinRange XLAL_INIT_DECL(spinRange);
  spinRange.refTime = refTime;
  memcpy ( &spinRange.fkdot, &injectSources->data[0].Doppler.fkdot, sizeof(spinRange.fkdot) );
  spinRange.fkdotBand[0] = (numFreqBins - 1)*dFreq - 10*LAL_REAL8_EPS;
  spinRange.fkdotBand[1] = (numf1dotPoints - 1)*df1dot - 10*LAL_REAL8_EPS;

  Doppler.fkdot[0] -= 0.4 * spinRange.fkdotBand[0];

  REAL8 minCoverFreq, maxCoverFreq;
  XLAL_CHECK ( XLALCWSignalCoveringBand ( &minCoverFreq, &maxCoverFreq, &startTime, &endTime, &spinRange, asini, Period, ecc ) == XLAL_SUCCESS, XLAL_EFUNC );

  // ----- setup extra Fstat method params
  FstatExtraParams XLAL_INIT_DECL(extraParams);
  extraParams.randSeed  = 1;
  extraParams.SSBprec = SSBPREC_RELATIVISTICOPT;
  extraParams.Dterms = 8;	// constant value that works for all Demod methods

  // ----- prepare input data with injection for all available methods
  FstatInput *input[FMETHOD_END];
  FstatResults *results[FMETHOD_END];
  for ( UINT4 iMethod = FMETHOD_START; iMethod < FMETHOD_END; iMethod ++ )
    {
      results[iMethod] = NULL;
      if ( !XLALFstatMethodIsAvailable(iMethod) ) {
        continue;
      }
      input[iMethod] = XLALCreateFstatInput ( catalog, minCoverFreq, maxCoverFreq, injectSources, &injectSqrtSX, &assumeSqrtSX, 50, ephem, iMethod, &extraParams );
      XLAL_CHECK ( input[iMethod] != NULL, XLAL_EFUNC );
    }

  FstatQuantities whatToCompute = (FSTATQ_2F | FSTATQ_FAFB);
  // ----- loop over all templates {sky, f1dot, period}
  for ( UINT4 iSky = 0; iSky < numSkyPoints; iSky ++ )
    {
      for ( UINT4 if1dot = 0; if1dot < numf1dotPoints; if1dot ++ )
        {
          for ( UINT4 iPeriod = 0; iPeriod < numPeriodPoints; iPeriod ++ )
            {
              // ----- loop over all available methods and compare Fstat results
              FstatMethodType firstMethod = FMETHOD_START;
              for ( UINT4 iMethod = FMETHOD_START; iMethod < FMETHOD_END; iMethod ++ )
                {
                  if ( !XLALFstatMethodIsAvailable(iMethod) ) {
                    continue;
                  }
                  if ( firstMethod == FMETHOD_START ) {	// keep track of first available method found
                    firstMethod = iMethod;
                  }

                  XLAL_CHECK ( XLALComputeFstat ( &results[iMethod], input[iMethod], &Doppler, dFreq, numFreqBins, whatToCompute ) == XLAL_SUCCESS, XLAL_EFUNC );

                  if ( lalDebugLevel & LALINFOBIT )
                    {
                      FILE *fp;
                      char fname[1024]; XLAL_INIT_MEM ( fname );
                      snprintf ( fname, sizeof(fname)-1, "twoF%s-iSky%02d-if1dot%02d-iPeriod%02d.dat", XLALGetFstatMethodName(iMethod), iSky, if1dot, iPeriod );
                      XLAL_CHECK ( (fp = fopen ( fname, "wb" )) != NULL, XLAL_EFUNC );
                      for ( UINT4 k = 0; k < results[iMethod]->numFreqBins; k ++ )
                        {
                          REAL8 Freq0 = results[iMethod]->doppler.fkdot[0];
                          REAL8 Freq_k = Freq0 + k * results[iMethod]->dFreq;
                          if ( whatToCompute & FSTATQ_FAFB ) {
                            fprintf ( fp, "%20.16g %10.4g   %10.4g %10.4g   %10.4g %10.4g\n",
                                      Freq_k, results[iMethod]->twoF[k],
                                      crealf(results[iMethod]->Fa[k]), cimagf(results[iMethod]->Fa[k]),
                                      crealf(results[iMethod]->Fb[k]), cimagf(results[iMethod]->Fb[k])
                                      );
                          } else {
                            fprintf ( fp, "%20.16g %10.4g\n",
                                      Freq_k, results[iMethod]->twoF[k] );
                          }
                        } // for k < numFreqBins
                      fclose(fp);
                    } // if info

                  // compare to first result
                  XLALPrintInfo ("Comparing results between method '%s' and '%s'\n", XLALGetFstatMethodName(firstMethod), XLALGetFstatMethodName(iMethod) );
                  if ( compareFstatResults ( results[firstMethod], results[iMethod] ) != XLAL_SUCCESS )
                    {
                      XLALPrintError ("Comparison between method '%s' and '%s' failed\n", XLALGetFstatMethodName(firstMethod), XLALGetFstatMethodName(iMethod) );
                      XLAL_ERROR ( XLAL_EFUNC );
                    }

                }  // for i < FMETHOD_END

              Doppler.period += dPeriod;

            } // for iPeriod < numPeriodPoints

          Doppler.fkdot[1] += df1dot;

        } // for if1dot < numf1dotPoints

      Doppler.Alpha += dSky;

    } // for iSky < numSkyPoints

  // free remaining memory
  for ( UINT4 iMethod=FMETHOD_START; iMethod < FMETHOD_END; iMethod ++ )
    {
      if ( !XLALFstatMethodIsAvailable(iMethod) ) {
        continue;
      }
      XLALDestroyFstatInput ( input[iMethod] );
      XLALDestroyFstatResults ( results[iMethod] );
    } // for i < FMETHOD_END

  XLALDestroyPulsarParamsVector ( injectSources );
  XLALDestroySFTCatalog ( catalog );
  XLALDestroyMultiTimestamps ( multiTimestamps );
  XLALDestroyStringVector ( detNames );
  XLALDestroyEphemerisData ( ephem );

  LALCheckMemoryLeaks();

  return XLAL_SUCCESS;

} // main()

static int
compareFstatResults ( const FstatResults *result1, const FstatResults *result2 )
{
  XLAL_CHECK ( (result1 != NULL) && (result2 != NULL), XLAL_EINVAL );

  XLAL_CHECK ( result1->whatWasComputed == result2->whatWasComputed, XLAL_EINVAL );
  XLAL_CHECK ( result1->dFreq == result2->dFreq, XLAL_EINVAL );
  XLAL_CHECK ( result1->numFreqBins == result2->numFreqBins, XLAL_EINVAL );
  XLAL_CHECK ( result1->numDetectors == result2->numDetectors, XLAL_EINVAL );

  // ----- set tolerance levels for comparisons ----------
  VectorComparison XLAL_INIT_DECL(tol);
  tol.relErr_L1 	= 8e-2;
  tol.relErr_L2		= 8e-2;
  tol.angleV 		= 0.08;  // rad
  tol.relErr_atMaxAbsx	= 8e-2;
  tol.relErr_atMaxAbsy  = 8e-2;

  UINT4 numFreqBins = result1->numFreqBins;
  VectorComparison XLAL_INIT_DECL(cmp);

  if ( result1->whatWasComputed & FSTATQ_2F )
    {
      // ----- package twoF arrays into REAL4Vectors
      REAL4Vector XLAL_INIT_DECL(v1);
      REAL4Vector XLAL_INIT_DECL(v2);
      v1.length = numFreqBins;
      v2.length = numFreqBins;
      v1.data = result1->twoF;
      v2.data = result2->twoF;

      XLALPrintInfo ("Comparing 2F values:\n");
      XLAL_CHECK ( XLALCompareREAL4Vectors ( &cmp, &v1, &v2, &tol ) == XLAL_SUCCESS, XLAL_EFUNC );
    }

  if ( result1->whatWasComputed & FSTATQ_FAFB )
    {
      // ----- package Fa,Fb arrays int COMPLEX8Vectors
      COMPLEX8Vector XLAL_INIT_DECL(c1);
      COMPLEX8Vector XLAL_INIT_DECL(c2);
      c1.length = numFreqBins;
      c2.length = numFreqBins;
      // Fa
      c1.data = result1->Fa;
      c2.data = result2->Fa;
      XLALPrintInfo ("Comparing Fa values:\n");
      XLAL_CHECK ( XLALCompareCOMPLEX8Vectors ( &cmp, &c1, &c2, &tol ) == XLAL_SUCCESS, XLAL_EFUNC );	 // FIXME: deactivated test
      // Fb
      c1.data = result1->Fb;
      c2.data = result2->Fb;
      XLALPrintInfo ("Comparing Fb values:\n");
      XLAL_CHECK ( XLALCompareCOMPLEX8Vectors ( &cmp, &c1, &c2, &tol ) == XLAL_SUCCESS, XLAL_EFUNC );	 // FIXME: deactivated test
    }

  return XLAL_SUCCESS;

} // compareFstatResults()

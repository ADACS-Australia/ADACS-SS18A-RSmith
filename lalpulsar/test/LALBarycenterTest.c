/*
 * Copyright (C) 2012 Miroslav Shaltev, R Prix
*  Copyright (C) 2007 Curt Cutler, David Chin, Jolien Creighton, Reinhard Prix, Teviet Creighton
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
 * \author Curt Cutler
 * \date 2001
 * \file
 * \ingroup LALBarycenter_h
 * \brief  Tests the routine LALBarycenter(): exercises some of the error
 * conditions and makes sure that they work.
 *
 * ### Program <tt>LALBarycenterTest.c</tt> ###
 *
 *
 * ### Usage ###
 *
 * \code
 * LALBarycenterTest
 * \endcode
 *
 * ### Description ###
 *
 * This program demonstrates the use of LALBarycenter.c.
 * The two ephemeris files specified in the EphemerisFilenames
 * structure (e.g., for data taken in 1998, <tt>sun98.dat</tt> and <tt>earth98.dat</tt>)
 * are assumed to be in the same directory as the program as
 * the test program.
 *
 */

#include <lal/LALBarycenter.h>
#include <lal/LALInitBarycenter.h>
#include <lal/DetectorSite.h>
#include <lal/Date.h>
#include <lal/LogPrintf.h>

/** \name Error codes */
/*@{*/
#define LALBARYCENTERTESTC_ENOM 0
#define LALBARYCENTERTESTC_EOPEN 1
#define LALBARYCENTERTESTC_EOUTOFRANGEE  4
#define LALBARYCENTERTESTC_EOUTOFRANGES  8
#define LALBARYCENTERTESTC_EBADSOURCEPOS 16
#define LALBARYCENTERTESTC_EEPHFILE 32

#define LALBARYCENTERTESTC_MSGENOM "Nominal exit"
#define LALBARYCENTERTESTC_MSGEOPEN "Error checking failed to catch missing ephemeris file."
#define LALBARYCENTERTESTC_MSGEOUTOFRANGEE "Failed to catch that tgps not in range of earth.dat file"
#define LALBARYCENTERTESTC_MSGEOUTOFRANGES "Failed to catch that tgps not in range of sun.dat file"
#define LALBARYCENTERTESTC_MSGEBADSOURCEPOS "Failed to catch bad source position"
#define LALBARYCENTERTESTC_MSGEEPHFILE "Failed to catch error reading ephemeris file."
/*@}*/

/** \cond DONT_DOXYGEN */

/* ----- internal prototype ---------- */
int compare_ephemeris ( const EphemerisData *edat1, const EphemerisData *edat2 );
REAL8 relerr(REAL8 x, REAL8 xapprox);

inline REAL8 relerr ( REAL8 x, REAL8 xapprox )
{
  REAL8 abserr = fabs ( x - xapprox );
  REAL8 absmean = 0.5 * fabs( x + xapprox );
  if ( absmean > 10 * LAL_REAL8_EPS )
    return abserr / absmean;
  else
    return abserr;
}

int diffEmissionTime  ( EmissionTime  *diff, const EmissionTime *emit1, const EmissionTime *emit2 );
int absmaxEmissionTime ( EmissionTime *absmax, const EmissionTime *demit1, const EmissionTime *demit2 );
REAL8 maxErrInEmissionTime ( const EmissionTime *demit );

const INT4 t2000 = 630720013; 		/* gps time at Jan 1, 2000 00:00:00 UTC */
const INT4 t1998 = 630720013-730*86400-1;	/* gps at Jan 1,1998 00:00:00 UTC*/

// ----------------------------------------------------------------------

int
main( void )
{
  static LALStatus status;

  char eEphFileBad[] = TEST_DATA_DIR "earth47.dat";
  char eEphFile[] = TEST_DATA_DIR "earth98.dat";
  char sEphFile[] = TEST_DATA_DIR "sun98.dat";

  /* Checking response if data files not present */
  EphemerisData edat;
  edat.ephiles.earthEphemeris = eEphFileBad;
  edat.ephiles.sunEphemeris   = sEphFile;
  LALInitBarycenter(&status, &edat);
  if ( status.statusCode != LALINITBARYCENTERH_EOPEN)
    {
      XLALPrintError( "Got error code %d and message '%s', but expected error code %d\n", status.statusCode, status.statusDescription, LALINITBARYCENTERH_EOPEN);
      return LALBARYCENTERTESTC_EOPEN;
    }
  else
    {
      // XLALPrintError ("==================== this error is as expected and OK!! ==================== \n");
      xlalErrno = 0;
    }

  /* Now inputting kosher ephemeris. files and leap sec, to illustrate
   * proper usage. The real, serious TEST of the code is a script written
   * by Rejean Dupuis comparing LALBarycenter to TEMPO for thousands
   * of source positions and times.
   */
  edat.ephiles.earthEphemeris = eEphFile;
  edat.ephiles.sunEphemeris = sEphFile;
  LALInitBarycenter(&status, &edat);
  if ( status.statusCode ) {
    XLALPrintError ("LALInitBarycenter() failed with code %d\n", status.statusCode);
    return XLAL_EFAILED;
  }

  /* ===== now test equivalence of new XLALInitBarycenter() function ========== */
  EphemerisData *edat_xlal;
  if ( ( edat_xlal = XLALInitBarycenter ( eEphFile, sEphFile )) == NULL ) {
    XLALPrintError ("Something failed in XLALInitBarycenter(), errno =%d\n", xlalErrno );
    return XLAL_EFAILED;
  }
  if ( compare_ephemeris ( &edat, edat_xlal ) != XLAL_SUCCESS ) {
    XLALPrintError ("Equivalence test failed between XLALInitEphemeris() and LALInitEphemeris()\n" );
    return XLAL_EFAILED;
  }
  XLALDestroyEphemerisData ( edat_xlal );

  /* ========================================================================== */


 /* The routines using LALBarycenter package, the code above, leading
    up LALInitBarycenter call, should be near top of main. The idea is
    that ephemeris data is read into RAM once, at the beginning.

    NOTE that the only part of the piece of the LALDetector structure
    baryinput.site that has to be filled in by the driver code is
    the 3-vector: baryinput.site.location[] .

    NOTE that the driver code that calls LALInitBarycenter must
    LALFree(edat->ephemE) and LALFree(edat->ephemS).
    The driver code that calls LALBarycenter must LALFree(edat).
 */

  /* Now getting coords for detector */
  LALDetector cachedDetector;
  cachedDetector = lalCachedDetectors[LALDetectorIndexGEO600DIFF];

  BarycenterInput XLAL_INIT_DECL(baryinput);
  baryinput.site.location[0]=cachedDetector.location[0]/LAL_C_SI;
  baryinput.site.location[1]=cachedDetector.location[1]/LAL_C_SI;
  baryinput.site.location[2]=cachedDetector.location[2]/LAL_C_SI;

  EarthState earth;
  EarthState earth_xlal;
  EmissionTime  emit, emit_xlal, emit_opt;

  /* ----- Checking error messages when the timestamp is not within the 1-yr ephemeris files */
  LIGOTimeGPS tGPS = {t1998+5e7, 0 };
  LALBarycenterEarth ( &status, &earth, &tGPS, &edat );
  if ( status.statusCode == 0 ) {
    XLALPrintError ( "LALBarycenterEarth() succeeded but expected to get error\n");
    return LALBARYCENTERTESTC_EOUTOFRANGEE;
  } else {
    XLALPrintError ("==================== this error is as expected and OK!! ==================== \n");
    xlalErrno = 0;
  }

  /* next try calling for bad choice of RA,DEC (e.g., something sensible in degrees, but radians)*/
  tGPS.gpsSeconds = t1998+3600;
  LALBarycenterEarth ( &status, &earth, &tGPS, &edat );

  baryinput.alpha= 120;
  baryinput.delta = 60;
  baryinput.dInv = 0;

  LALBarycenter ( &status, &emit, &baryinput, &earth );
  if ( status.statusCode == 0 ) {
    XLALPrintError( "LALBarycenter() succeeded but expected to get error\n" );
    return LALBARYCENTERTESTC_EBADSOURCEPOS;
  } else {
    XLALPrintError ("==================== this error is as expected and OK!! ==================== \n");
    xlalErrno = 0;
  }

  /* ---------- Now running program w/o errors, to illustrate proper use. ---------- */
  EmissionTime XLAL_INIT_DECL(maxDiff);
  EmissionTime XLAL_INIT_DECL(maxDiffOpt);
  REAL8 tic, toc;
  UINT4 NRepeat = 1;
  UINT4 counter = 0;
  REAL8 tau_lal = 0, tau_xlal = 0, tau_opt = 0;
  BarycenterBuffer *buffer = NULL;

  unsigned int seed = XLALGetTimeOfDay();
  srand ( seed );

  /* Outer loop over different sky positions */
  for ( UINT4 k=0; k < 300; k++)
    {
      baryinput.alpha = ( 1.0 * rand() / RAND_MAX ) * LAL_TWOPI;	// in [0, 2pi]
      baryinput.delta = ( 1.0 * rand() / RAND_MAX ) * LAL_PI - LAL_PI_2;// in [-pi/2, pi/2]
      baryinput.dInv = 0.e0;

      /* inner loop over pulse arrival times */
      for ( UINT4 i=0; i < 100; i++ )
        {
          REAL8 tPulse = t1998 + ( 1.0 * rand() / RAND_MAX ) * LAL_YRSID_SI;	// t in [1998, 1999]
          XLALGPSSetREAL8( &tGPS, tPulse );
          baryinput.tgps = tGPS;

          /* ----- old LAL interface ---------- */
          LALBarycenterEarth ( &status, &earth, &tGPS, &edat);
          if ( status.statusCode ) {
            XLALPrintError ("LALBarycenterEarth() failed with code %d\n", status.statusCode);
            return XLAL_EFAILED;
          }

          tic = XLALGetTimeOfDay();
          for ( UINT4 l = 0; l < NRepeat; l++ )
            LALBarycenter ( &status, &emit, &baryinput, &earth );
          toc = XLALGetTimeOfDay();
          tau_lal += ( toc - tic ) / NRepeat;
          if ( status.statusCode ) {
            XLALPrintError ("LALBarycenter() failed with code %d\n", status.statusCode);
            return XLAL_EFAILED;
          }

          /* ----- new XLAL interface ---------- */
          XLAL_CHECK ( XLALBarycenterEarth ( &earth_xlal, &tGPS, &edat ) == XLAL_SUCCESS, XLAL_EFAILED );
          tic = XLALGetTimeOfDay();
          for ( UINT4 l = 0; l < NRepeat; l ++ )
            XLAL_CHECK ( XLALBarycenter ( &emit_xlal, &baryinput, &earth_xlal ) == XLAL_SUCCESS, XLAL_EFAILED );
          toc = XLALGetTimeOfDay();
          tau_xlal += ( toc - tic ) / NRepeat;

          /* collect maximal deviations over all struct-fields of 'emit' */
          EmissionTime thisDiff;
          diffEmissionTime ( &thisDiff, &emit, &emit_xlal );
          absmaxEmissionTime ( &maxDiff, &maxDiff, &thisDiff );

          /* ----- optimized XLAL version with buffering ---------- */
          tic = XLALGetTimeOfDay();
          for ( UINT4 l = 0; l < NRepeat; l ++ )
            XLAL_CHECK ( XLALBarycenterOpt ( &emit_opt, &baryinput, &earth_xlal, &buffer ) == XLAL_SUCCESS, XLAL_EFAILED );
          toc = XLALGetTimeOfDay();
          tau_opt += ( toc - tic ) / NRepeat;

          /* collect maximal deviations over all struct-fields of 'emit' */
          diffEmissionTime ( &thisDiff, &emit, &emit_opt );
          absmaxEmissionTime ( &maxDiffOpt, &maxDiffOpt, &thisDiff );

          counter ++;
        } /* for i */

    } /* for k */

  XLALFree ( buffer );
  buffer = NULL;

  /* ----- check differences in results ---------- */
  REAL8 tolerance = 1e-9;	// in seconds: can't go beyond nanosecond precision due to GPS limitation
  REAL8 maxEmitDiff = maxErrInEmissionTime ( &maxDiff );
  REAL8 maxEmitDiffOpt = maxErrInEmissionTime ( &maxDiffOpt );
  XLALPrintInfo ( "Max error (in seconds) between LALBarycenter() and XLALBarycenter()     = %g s (tolerance = %g s)\n", maxEmitDiff, tolerance );
  XLAL_CHECK ( maxEmitDiff < tolerance, XLAL_EFAILED,
               "Max error (in seconds) between LALBarycenter() and XLALBarycenter()  = %g s, exceeding tolerance of %g s\n", maxEmitDiff, tolerance );

  XLALPrintInfo ( "Max error (in seconds) between LALBarycenter() and XLALBarycenterOpt()  = %g s (tolerance = %g s)\n", maxEmitDiffOpt, tolerance );
  XLAL_CHECK ( maxEmitDiffOpt < tolerance, XLAL_EFAILED,
               "Max error (in seconds) between LALBarycenter() and XLALBarycenterOpt()  = %g s, exceeding tolerance of %g s\n",
               maxEmitDiffOpt, tolerance );
  printf ( "%g	%g %d %d %g	%g %g %g	%g %g %g\n",
           maxEmitDiffOpt,
           maxDiffOpt.deltaT, maxDiffOpt.te.gpsSeconds, maxDiffOpt.te.gpsNanoSeconds, maxDiffOpt.tDot,
           maxDiffOpt.rDetector[0], maxDiffOpt.rDetector[1], maxDiffOpt.rDetector[2],
           maxDiffOpt.vDetector[0], maxDiffOpt.vDetector[1], maxDiffOpt.vDetector[2]
           );

  /* ----- output runtimes ---------- */
  XLALPrintError ("Runtimes per function-call, averaged over %g calls\n", 1.0 * NRepeat * counter );
  XLALPrintError ("LALBarycenter() 	%g s\n", tau_lal / counter );
  XLALPrintError ("XLALBarycenter()	%g s (= %.1f %%)\n", tau_xlal / counter, - 100 * (tau_lal - tau_xlal ) / tau_lal );
  XLALPrintError ("XLALBarycenterOpt()	%g s (= %.1f %%)\n", tau_opt / counter,  - 100 * (tau_lal - tau_opt ) / tau_lal );

  /* ===== test XLALRestrictEphemerisData() ===== */
  XLALPrintInfo("\n\nTesting XLALRestrictEphemerisData() ... ");
  {
    XLAL_CHECK( edat.nentriesS >= 100, XLAL_EFAILED );
    const INT4 orig_nentriesS = edat.nentriesS;
    for (INT4 i = 1; i <= 4; ++i) {
      REAL8 start, end;
      LIGOTimeGPS startGPS, endGPS;
      INT4 diff_nentriesS;

      start = edat.ephemS[2*i].gps;
      end = edat.ephemS[edat.nentriesS - 1 - 3*i].gps;
      XLAL_CHECK( XLALGPSSetREAL8(&startGPS, start) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALGPSSetREAL8(&endGPS, end) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALRestrictEphemerisData(&edat, &startGPS, &endGPS) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK( edat.ephemS[0].gps == start, XLAL_EFAILED, "\nTest S%dA FAILED: %0.9f != start %0.9f\n", i, edat.ephemS[0].gps, start );
      XLAL_CHECK( edat.ephemS[edat.nentriesS - 1].gps == end, XLAL_EFAILED, "\nTest S%dA FAILED: end %0.9f != %0.9f\n", i, edat.ephemS[edat.nentriesS - 1].gps, end );
      diff_nentriesS = ((i*i + i) * 5) / 2 + 2*(i-1);
      XLAL_CHECK( orig_nentriesS - edat.nentriesS == diff_nentriesS, XLAL_EFAILED, "\nTest S%dA FAILED: nentries %d != %d\n", i, orig_nentriesS - edat.nentriesS, diff_nentriesS );

      XLAL_CHECK( XLALGPSSetREAL8(&startGPS, start + 0.5*edat.dtStable) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALGPSSetREAL8(&endGPS, end - 0.5*edat.dtStable) != NULL, XLAL_EFUNC );
      start = edat.ephemS[0].gps;
      end = edat.ephemS[edat.nentriesS - 1].gps;
      XLAL_CHECK( XLALRestrictEphemerisData(&edat, &startGPS, &endGPS) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK( edat.ephemS[0].gps == start, XLAL_EFAILED, "\nTest S%dB FAILED: start %0.9f != %0.9f\n", i, edat.ephemS[0].gps, start );
      XLAL_CHECK( edat.ephemS[edat.nentriesS - 1].gps == end, XLAL_EFAILED, "\nTest S%dB FAILED: end %0.9f != %0.9f\n", i, edat.ephemS[edat.nentriesS - 1].gps, end );
      diff_nentriesS = ((i*i + i) * 5) / 2 + 2*(i-1);
      XLAL_CHECK( orig_nentriesS - edat.nentriesS == diff_nentriesS, XLAL_EFAILED, "\nTest S%dB FAILED: nentries %d != %d\n", i, orig_nentriesS - edat.nentriesS, diff_nentriesS );

      XLAL_CHECK( XLALGPSSetREAL8(&startGPS, start + 1.5*edat.dtStable) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALGPSSetREAL8(&endGPS, end - 1.5*edat.dtStable) != NULL, XLAL_EFUNC );
      start = edat.ephemS[1].gps;
      end = edat.ephemS[edat.nentriesS - 2].gps;
      XLAL_CHECK( XLALRestrictEphemerisData(&edat, &startGPS, &endGPS) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK( edat.ephemS[0].gps == start, XLAL_EFAILED, "\nTest S%dC FAILED: start %0.9f != %0.9f\n", i, edat.ephemS[0].gps, start );
      XLAL_CHECK( edat.ephemS[edat.nentriesS - 1].gps == end, XLAL_EFAILED, "\nTest S%dC FAILED: end %0.9f != %0.9f\n", i, edat.ephemS[edat.nentriesS - 1].gps, end );
      diff_nentriesS = ((i*i + i) * 5) / 2 + 2*i;
      XLAL_CHECK( orig_nentriesS - edat.nentriesS == diff_nentriesS, XLAL_EFAILED, "\nTest S%dC FAILED: nentries %d != %d\n", i, orig_nentriesS - edat.nentriesS, diff_nentriesS );
    }

    XLAL_CHECK( edat.nentriesE >= 100, XLAL_EFAILED );
    const INT4 orig_nentriesE = edat.nentriesE;
    for (INT4 i = 1; i <= 4; ++i) {
      REAL8 start, end;
      LIGOTimeGPS startGPS, endGPS;
      INT4 diff_nentriesE;

      start = edat.ephemE[2*i].gps;
      end = edat.ephemE[edat.nentriesE - 1 - 3*i].gps;
      XLAL_CHECK( XLALGPSSetREAL8(&startGPS, start) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALGPSSetREAL8(&endGPS, end) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALRestrictEphemerisData(&edat, &startGPS, &endGPS) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK( edat.ephemE[0].gps == start, XLAL_EFAILED, "\nTest E%dA FAILED: %0.9f != start %0.9f\n", i, edat.ephemE[0].gps, start );
      XLAL_CHECK( edat.ephemE[edat.nentriesE - 1].gps == end, XLAL_EFAILED, "\nTest E%dA FAILED: end %0.9f != %0.9f\n", i, edat.ephemE[edat.nentriesE - 1].gps, end );
      diff_nentriesE = ((i*i + i) * 5) / 2 + 2*(i-1);
      XLAL_CHECK( orig_nentriesE - edat.nentriesE == diff_nentriesE, XLAL_EFAILED, "\nTest E%dA FAILED: nentries %d != %d\n", i, orig_nentriesE - edat.nentriesE, diff_nentriesE );

      XLAL_CHECK( XLALGPSSetREAL8(&startGPS, start + 0.5*edat.dtEtable) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALGPSSetREAL8(&endGPS, end - 0.5*edat.dtEtable) != NULL, XLAL_EFUNC );
      start = edat.ephemE[0].gps;
      end = edat.ephemE[edat.nentriesE - 1].gps;
      XLAL_CHECK( XLALRestrictEphemerisData(&edat, &startGPS, &endGPS) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK( edat.ephemE[0].gps == start, XLAL_EFAILED, "\nTest E%dB FAILED: start %0.9f != %0.9f\n", i, edat.ephemE[0].gps, start );
      XLAL_CHECK( edat.ephemE[edat.nentriesE - 1].gps == end, XLAL_EFAILED, "\nTest E%dB FAILED: end %0.9f != %0.9f\n", i, edat.ephemE[edat.nentriesE - 1].gps, end );
      diff_nentriesE = ((i*i + i) * 5) / 2 + 2*(i-1);
      XLAL_CHECK( orig_nentriesE - edat.nentriesE == diff_nentriesE, XLAL_EFAILED, "\nTest E%dB FAILED: nentries %d != %d\n", i, orig_nentriesE - edat.nentriesE, diff_nentriesE );

      XLAL_CHECK( XLALGPSSetREAL8(&startGPS, start + 1.5*edat.dtEtable) != NULL, XLAL_EFUNC );
      XLAL_CHECK( XLALGPSSetREAL8(&endGPS, end - 1.5*edat.dtEtable) != NULL, XLAL_EFUNC );
      start = edat.ephemE[1].gps;
      end = edat.ephemE[edat.nentriesE - 2].gps;
      XLAL_CHECK( XLALRestrictEphemerisData(&edat, &startGPS, &endGPS) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK( edat.ephemE[0].gps == start, XLAL_EFAILED, "\nTest E%dC FAILED: start %0.9f != %0.9f\n", i, edat.ephemE[0].gps, start );
      XLAL_CHECK( edat.ephemE[edat.nentriesE - 1].gps == end, XLAL_EFAILED, "\nTest E%dC FAILED: end %0.9f != %0.9f\n", i, edat.ephemE[edat.nentriesE - 1].gps, end );
      diff_nentriesE = ((i*i + i) * 5) / 2 + 2*i;
      XLAL_CHECK( orig_nentriesE - edat.nentriesE == diff_nentriesE, XLAL_EFAILED, "\nTest E%dC FAILED: nentries %d != %d\n", i, orig_nentriesE - edat.nentriesE, diff_nentriesE );
    }
  }
  XLALPrintInfo("PASSED\n\n");

  LALFree(edat.ephemE);
  LALFree(edat.ephemS);

  LALCheckMemoryLeaks();

  XLALPrintError ("==> OK. All tests successful!\n\n");

  return 0;

} /* main() */


/**
 * Function to test equivalence between two loaded ephemeris-data structs.
 * Note: we compare everything *except* the ephiles fields, which are actually useless.
 */
int
compare_ephemeris ( const EphemerisData *edat1, const EphemerisData *edat2 )
{
  if ( !edat1 || !edat2 ) {
    XLALPrintError ("%s: invalid NULL input edat1=%p, edat2=%p\n", __func__, edat1, edat2 );
    XLAL_ERROR ( XLAL_EINVAL );
  }

  if ( edat1->nentriesE != edat2->nentriesE ) {
    XLALPrintError ("%s: different nentriesE (%d != %d)\n", __func__, edat1->nentriesE, edat2->nentriesE );
    XLAL_ERROR ( XLAL_EFAILED );
  }
  if ( edat1->nentriesS != edat2->nentriesS ) {
    XLALPrintError ("%s: different nentriesS (%d != %d)\n", __func__, edat1->nentriesS, edat2->nentriesS );
    XLAL_ERROR ( XLAL_EFAILED );
  }
  if ( edat1->dtEtable != edat2->dtEtable ) {
    XLALPrintError ("%s: different dtEtable (%g != %g)\n", __func__, edat1->dtEtable, edat2->dtEtable );
    XLAL_ERROR ( XLAL_EFAILED );
  }
  if ( edat1->dtStable != edat2->dtStable ) {
    XLALPrintError ("%s: different dtStable (%g != %g)\n", __func__, edat1->dtStable, edat2->dtStable );
    XLAL_ERROR ( XLAL_EFAILED );
  }

  /* compare earth ephemeris data */
  if ( !edat1->ephemE || !edat2->ephemE ) {
    XLALPrintError ("%s: invalid NULL ephemE pointer edat1 (%p), edat2 (%p)\n", __func__, edat1->ephemE, edat2->ephemE );
    XLAL_ERROR ( XLAL_EINVAL );
  }
  INT4 i;
  for ( i=0; i < edat1->nentriesE; i ++ )
    {
      if ( edat1->ephemE[i].gps != edat2->ephemE[i].gps ||

           edat1->ephemE[i].pos[0] != edat2->ephemE[i].pos[0] ||
           edat1->ephemE[i].pos[1] != edat2->ephemE[i].pos[1] ||
           edat1->ephemE[i].pos[2] != edat2->ephemE[i].pos[2] ||

           edat1->ephemE[i].vel[0] != edat2->ephemE[i].vel[0] ||
           edat1->ephemE[i].vel[1] != edat2->ephemE[i].vel[1] ||
           edat1->ephemE[i].vel[2] != edat2->ephemE[i].vel[2] ||

           edat1->ephemE[i].acc[0] != edat2->ephemE[i].acc[0] ||
           edat1->ephemE[i].acc[1] != edat2->ephemE[i].acc[1] ||
           edat1->ephemE[i].acc[2] != edat2->ephemE[i].acc[2]
           )
        {
          XLALPrintError ("%s: Inconsistent earth-entry %d:\n", __func__, i );
          XLALPrintError ("    edat1 = %g, (%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
                          edat1->ephemE[i].gps,
                          edat1->ephemE[i].pos[0], edat1->ephemE[i].pos[1], edat1->ephemE[i].pos[2],
                          edat1->ephemE[i].vel[0], edat1->ephemE[i].vel[1], edat1->ephemE[i].vel[2],
                          edat1->ephemE[i].acc[0], edat1->ephemE[i].acc[1], edat1->ephemE[i].acc[2]
                          );
          XLALPrintError ("    edat2 = %g, (%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
                          edat2->ephemE[i].gps,
                          edat2->ephemE[i].pos[0], edat2->ephemE[i].pos[1], edat2->ephemE[i].pos[2],
                          edat2->ephemE[i].vel[0], edat2->ephemE[i].vel[1], edat2->ephemE[i].vel[2],
                          edat2->ephemE[i].acc[0], edat2->ephemE[i].acc[1], edat2->ephemE[i].acc[2]
                          );
          XLAL_ERROR ( XLAL_EFAILED );
        } /* if difference in data-set i */

    } /* for i < nentriesE */

  /* compare sun ephemeris data */
  if ( !edat1->ephemS || !edat2->ephemS ) {
    XLALPrintError ("%s: invalid NULL ephemS pointer edat1 (%p), edat2 (%p)\n", __func__, edat1->ephemS, edat2->ephemS );
    XLAL_ERROR ( XLAL_EINVAL );
  }
  for ( i=0; i < edat1->nentriesS; i ++ )
    {
      if ( edat1->ephemS[i].gps != edat2->ephemS[i].gps ||

           edat1->ephemS[i].pos[0] != edat2->ephemS[i].pos[0] ||
           edat1->ephemS[i].pos[1] != edat2->ephemS[i].pos[1] ||
           edat1->ephemS[i].pos[2] != edat2->ephemS[i].pos[2] ||

           edat1->ephemS[i].vel[0] != edat2->ephemS[i].vel[0] ||
           edat1->ephemS[i].vel[1] != edat2->ephemS[i].vel[1] ||
           edat1->ephemS[i].vel[2] != edat2->ephemS[i].vel[2] ||

           edat1->ephemS[i].acc[0] != edat2->ephemS[i].acc[0] ||
           edat1->ephemS[i].acc[1] != edat2->ephemS[i].acc[1] ||
           edat1->ephemS[i].acc[2] != edat2->ephemS[i].acc[2]
           )
        {
          XLALPrintError ("%s: Inconsistent sun-entry %d:\n", __func__, i );
          XLALPrintError ("    edat1 = %g, (%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
                          edat1->ephemS[i].gps,
                          edat1->ephemS[i].pos[0], edat1->ephemS[i].pos[1], edat1->ephemS[i].pos[2],
                          edat1->ephemS[i].vel[0], edat1->ephemS[i].vel[1], edat1->ephemS[i].vel[2],
                          edat1->ephemS[i].acc[0], edat1->ephemS[i].acc[1], edat1->ephemS[i].acc[2]
                          );
          XLALPrintError ("    edat2 = %g, (%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
                          edat2->ephemS[i].gps,
                          edat2->ephemS[i].pos[0], edat2->ephemS[i].pos[1], edat2->ephemS[i].pos[2],
                          edat2->ephemS[i].vel[0], edat2->ephemS[i].vel[1], edat2->ephemS[i].vel[2],
                          edat2->ephemS[i].acc[0], edat2->ephemS[i].acc[1], edat2->ephemS[i].acc[2]
                          );
          XLAL_ERROR ( XLAL_EFAILED );
        } /* if difference in data-set i */

    } /* for i < nentriesE */

  /* everything seems fine */
  return XLAL_SUCCESS;

} /* compare_ephemeris() */

/* return differences in all fields from EmissionTime struct */
int
diffEmissionTime ( EmissionTime *diff, const EmissionTime *emit1, const EmissionTime *emit2 )
{
  if ( diff == NULL )
    return -1;

  diff->deltaT = emit1->deltaT - emit2->deltaT;
  diff->te.gpsSeconds = emit1->te.gpsSeconds - emit2->te.gpsSeconds;
  diff->te.gpsNanoSeconds = emit1->te.gpsNanoSeconds - emit2->te.gpsNanoSeconds;
  diff->tDot = emit1->tDot - emit2->tDot;
  for ( UINT4 i = 0; i < 3; i ++ )
    {
      diff->rDetector[i] = emit1->rDetector[i] - emit2->rDetector[i];
      diff->vDetector[i] = emit1->vDetector[i] - emit2->vDetector[i];
    }

  return 0;

} /* DiffEmissionTime() */

/* return max ( fabs ( de1, e2 ) ) on all struct fields of 'de1' and 'de2' respectively */
int
absmaxEmissionTime ( EmissionTime *absmax, const EmissionTime *demit1, const EmissionTime *demit2 )
{
  if ( absmax == NULL )
    return -1;

  absmax->deltaT 		= fmax ( fabs ( demit1->deltaT ) , fabs ( demit2->deltaT ) );
  absmax->te.gpsSeconds 	= fmax ( abs ( demit1->te.gpsSeconds) , abs ( demit2->te.gpsSeconds ) );
  absmax->te.gpsNanoSeconds 	= fmax ( abs ( demit1->te.gpsNanoSeconds ), abs ( demit2->te.gpsNanoSeconds ) );
  absmax->tDot 			= fmax ( fabs ( demit1->tDot ) , fabs ( demit2->tDot ) );
  for ( UINT4 i = 0; i < 3; i ++ )
    {
      absmax->rDetector[i] 	= fmax ( fabs ( demit1->rDetector[i] ) , fabs ( demit2->rDetector[i] ) );
      absmax->vDetector[i] 	= fmax ( fabs ( demit1->vDetector[i] ), fabs ( demit2->vDetector[i] ) );
    }

  return 0;

} /* absmaxEmissionTime() */

/* return maximal struct entry from 'demit' */
REAL8
maxErrInEmissionTime ( const EmissionTime *demit )
{
  REAL8 maxdiff = 0;
  maxdiff 		= fmax ( maxdiff, fabs ( demit->deltaT ) );
  maxdiff		= fmax ( maxdiff, abs ( demit->te.gpsSeconds ));
  maxdiff 		= fmax ( maxdiff, 1e-9 * abs ( demit->te.gpsNanoSeconds ) );
  maxdiff 		= fmax ( maxdiff, fabs ( demit->tDot ) );
  for ( UINT4 i = 0; i < 3; i ++ )
    {
      maxdiff  		= fmax ( maxdiff, fabs ( demit->rDetector[i] ) );
      maxdiff  		= fmax ( maxdiff, fabs ( demit->vDetector[i] ) );
    }
  return maxdiff;
}

/** \endcond */

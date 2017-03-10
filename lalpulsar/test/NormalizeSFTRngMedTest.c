/*
*  Copyright (C) 2012 Reinhard Prix
*  Copyright (C) 2005,2006 Badri Krishnan
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

#include <lal/FrequencySeries.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/Units.h>

#define REL_ERR(x,y) ( fabs((x) - (y)) / fabs( (x) ) )

/* Default parameters. */

REAL8 tol = LAL_REAL4_EPS;

int main ( void )
{
  const char *fn = __func__;

  SFTtype *mySFT;
  LIGOTimeGPS epoch = { 731210229, 0 };
  REAL8 dFreq = 1.0 / 1800.0;
  REAL8 f0 = 150.0 - 2.0 * dFreq;

  /* init data array */
  COMPLEX8 vals[] = {
    crectf( -1.249241e-21,   1.194085e-21 ),
    crectf(  2.207420e-21,   2.472366e-22 ),
    crectf(  1.497939e-21,   6.593609e-22 ),
    crectf(  3.544089e-20,  -9.365807e-21 ),
    crectf(  1.292773e-21,  -1.402466e-21 )
  };
  UINT4 numBins = sizeof ( vals ) / sizeof(vals[0] );

  if ( (mySFT = XLALCreateSFT ( numBins )) == NULL ) {
    XLALPrintError ("%s: Failed to create test-SFT using XLALCreateSFT(), xlalErrno = %d\n", fn, xlalErrno );
    return XLAL_EFAILED;
  }
  /* init header */
  strcpy ( mySFT->name, "H1;testSFTRngmed" );
  mySFT->epoch = epoch;
  mySFT->f0 = f0;
  mySFT->deltaF = dFreq;

  /* we simply copy over these data-values into the SFT */
  UINT4 iBin;
  for ( iBin = 0; iBin < numBins; iBin ++ )
    mySFT->data->data[iBin] = vals[iBin];

  /* get memory for running-median vector */
  REAL8FrequencySeries XLAL_INIT_DECL(rngmed);
  XLAL_CHECK ( (rngmed.data = XLALCreateREAL8Vector ( numBins )) != NULL, XLAL_EFUNC, "Failed  XLALCreateREAL8Vector ( %d )", numBins );

  // ---------- Test running-median PSD estimation in simple blocksize cases
  // ------------------------------------------------------------
  // TEST 1: odd blocksize = 3
  // ------------------------------------------------------------
  UINT4 blockSize3 = 3;

  /* reference result for 3-bin block running-median computed in octave:
octave> sft = [ \
        -1.249241e-21 +  1.194085e-21i, \
         2.207420e-21 +  2.472366e-22i, \
         1.497939e-21 +  6.593609e-22i, \
         3.544089e-20 -  9.365807e-21i, \
         1.292773e-21 -  1.402466e-21i  \
         ];
octave> periodo = abs(sft).^2;
octave> m1 = median ( periodo(1:3) ); m2 = median ( periodo(2:4) ); m3 = median ( periodo (3:5 ) );
octave> rngmed = [ m1, m1, m2, m3, m3 ];
octave> printf ("rngmedREF3 = { %.16g, %.16g, %.16g, %.16g, %.16g };\n", rngmed );
        rngmedREF3[] = { 2.986442063306e-42, 2.986442063306e-42, 4.933828992779561e-42, 3.638172910684999e-42, 3.638172910684999e-42 };
  */
  REAL8 rngmedREF3[] = { 2.986442063306e-42, 2.986442063306e-42, 4.933828992779561e-42, 3.638172910684999e-42, 3.638172910684999e-42 };

  /* compute running median */
  XLAL_CHECK ( XLALSFTtoRngmed ( &rngmed, mySFT, blockSize3 ) == XLAL_SUCCESS, XLAL_EFUNC, "XLALSFTtoRngmed() failed.");

  /* get median->mean bias correction, needed for octave-reference results, to make
   * them comparable to the bias-corrected results from XLALSFTtoRngmed()
   */
  REAL8 medianBias3 = XLALRngMedBias ( blockSize3 );
  XLAL_CHECK ( xlalErrno == 0, XLAL_EFUNC, "XLALRngMedBias() failed.");

  BOOLEAN pass = 1;
  const CHAR *passStr;
  printf ("%4s %22s %22s %8s    <%g\n", "Bin", "rngmed(LAL)", "rngmed(Octave)", "relError", tol);
  for (iBin=0; iBin < numBins; iBin ++ )
    {
      REAL8 rngmedVAL = rngmed.data->data[iBin];
      REAL8 rngmedREF = rngmedREF3[iBin] / medianBias3;	// apply median-bias correction
      REAL8 relErr = REL_ERR ( rngmedREF, rngmedVAL );
      if ( relErr > tol ) {
        pass = 0;
        passStr = "fail";
      } else {
        passStr = "OK.";
      }

      printf ("%4d %22.16g %22.16g %8.1g    %s\n", iBin, rngmedVAL, rngmedREF, relErr, passStr );

    } /* for iBin < numBins */

  // ------------------------------------------------------------
  // TEST 2: even blocksize = 4
  // ------------------------------------------------------------
  UINT4 blockSize4 = 4;

  /* reference result for 4-bin block running-median computed in octave:
octave> m1 = median ( periodo(1:4) ); m2 = median ( periodo(2:5) );
octave> rngmed = [ m1, m1, m1, m2, m2 ];
octave> printf ("rngmedREF4[] = { %.16g, %.16g, %.16g, %.16g, %.16g };\n", rngmed );
rngmedREF4[] = { 3.96013552804278e-42, 3.96013552804278e-42, 3.96013552804278e-42, 4.28600095173228e-42, 4.28600095173228e-42 };
  */
  REAL8 rngmedREF4[] = { 3.96013552804278e-42, 3.96013552804278e-42, 3.96013552804278e-42, 4.28600095173228e-42, 4.28600095173228e-42 };

  /* compute running median */
  XLAL_CHECK ( XLALSFTtoRngmed ( &rngmed, mySFT, blockSize4 ) == XLAL_SUCCESS, XLAL_EFUNC, "XLALSFTtoRngmed() failed.");

  /* get median->mean bias correction, needed for octave-reference results, to make
   * them comparable to the bias-corrected results from XLALSFTtoRngmed()
   */
  REAL8 medianBias4 = XLALRngMedBias ( blockSize4 );
  XLAL_CHECK ( xlalErrno == 0, XLAL_EFUNC, "XLALRngMedBias() failed.");

  printf ("%4s %22s %22s %8s    <%g\n", "Bin", "rngmed(LAL)", "rngmed(Octave)", "relError", tol);
  for (iBin=0; iBin < numBins; iBin ++ )
    {
      REAL8 rngmedVAL = rngmed.data->data[iBin];
      REAL8 rngmedREF = rngmedREF4[iBin] / medianBias4;	// apply median-bias correction
      REAL8 relErr = REL_ERR ( rngmedREF, rngmedVAL );
      if ( relErr > tol ) {
        pass = 0;
        passStr = "fail";
      } else {
        passStr = "OK.";
      }

      printf ("%4d %22.16g %22.16g %8.1g    %s\n", iBin, rngmedVAL, rngmedREF, relErr, passStr );

    } /* for iBin < numBins */

  /* free memory */
  XLALDestroyREAL8Vector ( rngmed.data );
  XLALDestroySFT ( mySFT );

  LALCheckMemoryLeaks();

  if ( !pass )
    {
      printf ("Test failed! Difference exceeded tolerance.\n");
      return XLAL_EFAILED;
    }
  else
    {
      printf ("Test passed.\n");
      return XLAL_SUCCESS;
    }

} /* main() */

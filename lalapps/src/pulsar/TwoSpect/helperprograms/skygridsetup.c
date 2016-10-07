/*
 *  Copyright (C) 2011, 2013, 2014 Evan Goetz
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


#include <stdio.h>
#include <math.h>

#include <lal/UserInput.h>
#include <lal/LALString.h>
#include <lal/DopplerScan.h>
#include <lal/LALInitBarycenter.h>

#include "../antenna.h"

typedef struct
{
   REAL8 Tsft;
   REAL8 SFToverlap;
   REAL8 t0;
   REAL8 Tobs;
   REAL8 fmin;
   REAL8 fspan;
   CHAR *IFO;
   CHAR *outfilename;
   CHAR *ephemEarth;
   CHAR *ephemSun;
   CHAR *skyRegion;
   BOOLEAN v1;
} UserVariables_t;

INT4 InitUserVars(UserVariables_t *uvar, int argc, char *argv[]);

//Main program
int main(int argc, char *argv[])
{
   
   FILE *OUTPUT;
   LALDetector det;
   LALStatus XLAL_INIT_DECL(status);

   UserVariables_t XLAL_INIT_DECL(uvar);
   XLAL_CHECK ( InitUserVars(&uvar, argc, argv) == XLAL_SUCCESS, XLAL_EFUNC );

   XLAL_CHECK( (OUTPUT = fopen(uvar.outfilename,"w")) != NULL, XLAL_EIO, "Output file %s could not be opened\n", uvar.outfilename );
   
   //Interferometer
   if (strcmp("L1", uvar.IFO)==0) {
      fprintf(stderr,"IFO = %s\n", uvar.IFO);
      det = lalCachedDetectors[LAL_LLO_4K_DETECTOR]; //L1
   } else if (strcmp("H1", uvar.IFO)==0) {
      fprintf(stderr,"IFO = %s\n", uvar.IFO);
      det = lalCachedDetectors[LAL_LHO_4K_DETECTOR]; //H1
   } else if (strcmp("V1", uvar.IFO)==0) {
      fprintf(stderr,"IFO = %s\n", uvar.IFO);
      det = lalCachedDetectors[LAL_VIRGO_DETECTOR]; //V1
   } else {
      XLAL_ERROR( XLAL_EINVAL, "Not using valid interferometer! Expected 'H1', 'L1', or 'V1' not %s.\n", uvar.IFO);
   }
   
   //Parameters for the sky-grid
   fprintf(stderr, "Sky region = %s\n", uvar.skyRegion);
   DopplerSkyScanInit XLAL_INIT_DECL(scanInit);
   DopplerSkyScanState XLAL_INIT_DECL(scan);
   PulsarDopplerParams dopplerpos;
   scanInit.gridType = GRID_ISOTROPIC;     //Default value for an approximate-isotropic grid
   scanInit.skyRegionString = uvar.skyRegion;      //"allsky" = Default value for all-sky search
   scanInit.numSkyPartitions = 1;   //Default value so sky is not broken into chunks
   scanInit.Freq = uvar.fmin+0.5*uvar.fspan;  //Mid-point of the frequency band
   
   //Initialize ephemeris data structure
   EphemerisData *edat = NULL;
   XLAL_CHECK( (edat = XLALInitBarycenter(uvar.ephemEarth, uvar.ephemSun)) != NULL, XLAL_EFUNC );
   
   //v1: Maximum orbital earth speed in units of c from start of S6 TwoSpect data for 104 weeks total time
   //else: maximum detector velocity around the time of observation
   REAL4 detectorVmax = 0.0;
   if (uvar.v1) detectorVmax = CompDetectorVmax(931081500.0+uvar.SFToverlap, uvar.Tsft, uvar.SFToverlap, 62899200.0-uvar.SFToverlap, det, edat);
   else detectorVmax = CompDetectorVmax(uvar.t0-uvar.Tsft, uvar.Tsft, uvar.SFToverlap, uvar.Tobs+2.0*uvar.Tsft, det, edat);
   XLAL_CHECK( xlalErrno == 0, XLAL_EFUNC, "CompDetectorVmax() failed\n" );
   
   //Initialize the sky-grid
   scanInit.dAlpha = 0.5/((uvar.fmin+0.5*uvar.fspan) * uvar.Tsft * detectorVmax);
   scanInit.dDelta = scanInit.dAlpha;
   InitDopplerSkyScan(&status, &scan, &scanInit);
   XLAL_CHECK( status.statusCode == 0, XLAL_EFUNC );
   
   //Start at first location
   XLAL_CHECK( XLALNextDopplerSkyPos(&dopplerpos, &scan) == XLAL_SUCCESS, XLAL_EFUNC );
   
   //loop through and output to the specified file
   while (scan.state != STATE_FINISHED) {
      fprintf(OUTPUT, "%.6f %.6f\n", dopplerpos.Alpha, dopplerpos.Delta);
      
      //Iterate to next sky location
      XLAL_CHECK( XLALNextDopplerSkyPos(&dopplerpos, &scan) == XLAL_SUCCESS, XLAL_EFUNC );
   }

   //Destroy
   XLALDestroyUserVars();
   XLALDestroyEphemerisData(edat);
   fclose(OUTPUT);
   
   return 0;
   
}


INT4 InitUserVars(UserVariables_t *uvar, int argc, char *argv[])
{
   XLAL_CHECK ( uvar != NULL, XLAL_EINVAL, "Invalid NULL input 'uvar'\n");
   XLAL_CHECK ( argv != NULL, XLAL_EINVAL, "Invalid NULL input 'argv'\n");

   uvar->ephemEarth = XLALStringDuplicate("earth00-19-DE405.dat.gz");
   uvar->ephemSun = XLALStringDuplicate("sun00-19-DE405.dat.gz");
   uvar->outfilename = XLALStringDuplicate("skygrid.dat");
   uvar->skyRegion = XLALStringDuplicate("allsky");
   uvar->Tsft = 1800;
   uvar->SFToverlap = 900;
   uvar->fspan = 0.25;

   XLALRegisterUvarMember(  Tsft,        REAL8, 0 , OPTIONAL , "SFT coherence time");
   XLALRegisterUvarMember(  SFToverlap,  REAL8, 0 , OPTIONAL , "SFT overlap in seconds, usually Tsft/2");
   XLALRegisterUvarMember(  t0,          REAL8, 0 , OPTIONAL , "GPS start time of the search; not needed if --v1 is specified");
   XLALRegisterUvarMember(  Tobs,        REAL8, 0 , OPTIONAL , "Duration of the search (in seconds); not needed if --v1 is specified");
   XLALRegisterUvarMember(  fmin,        REAL8, 0 , OPTIONAL , "Minimum frequency of band");
   XLALRegisterUvarMember(  fspan,       REAL8, 0 , OPTIONAL , "Frequency span of band");
   XLALRegisterUvarMember(IFO,         STRING, 0 , REQUIRED , "Interferometer whose data is being analyzed");
   XLALRegisterUvarMember(outfilename, STRING, 0 , OPTIONAL , "Output filename");
   XLALRegisterUvarMember(ephemEarth,  STRING, 0 , OPTIONAL , "Earth ephemeris file");
   XLALRegisterUvarMember(ephemSun,    STRING, 0 , OPTIONAL , "Sun ephemeris file");
   XLALRegisterUvarMember(skyRegion,   STRING, 0 , OPTIONAL , "Region of the sky to search (e.g. (ra1,dec1),(ra2,dec2),(ra3,dec3)...) or allsky");
   XLALRegisterUvarMember(  v1,          BOOLEAN, 0 , DEVELOPER, "Flag to use older style of CompDetectorVmax (for S6/VSR2-3 analysis)");

   BOOLEAN should_exit = 0;
   XLAL_CHECK( XLALUserVarReadAllInput( &should_exit, argc, argv ) == XLAL_SUCCESS, XLAL_EFUNC );
   if ( should_exit ) exit (1);

   if (!uvar->v1 && !XLALUserVarWasSet(&uvar->t0)) XLAL_ERROR( XLAL_EINVAL, "Must set t0" );
   if (!uvar->v1 && !XLALUserVarWasSet(&uvar->Tobs)) XLAL_ERROR( XLAL_EINVAL, "Must set Tobs" );

   return XLAL_SUCCESS;
}

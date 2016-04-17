/*
*  Copyright (C) 2009, 2016 Matt Pitkin
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

/* Calculate the frequency evolution of a pulsar over time */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <lal/LALStdlib.h>
#include <lal/LALAtomicDatatypes.h>
#include <lal/LALDatatypes.h>
#include <lal/AVFactories.h>
#include <lal/SkyCoordinates.h>
#include <lal/DetectorSite.h>
#include <lal/DetResponse.h>
#include <lal/ReadPulsarParFile.h>
#include <lal/BinaryPulsarTiming.h>
#include <lal/LALString.h>
#include <lal/SFTutils.h>
#include <lal/LALBarycenter.h>
#include <lal/LALInitBarycenter.h>
#include <lal/LALConstants.h>
#include <lal/XLALError.h>

#include <getopt.h>

#include "ppe_models.h"
#include "ppe_utils.h"

#define CODEUSAGE \
"Usage: %s [options]\n\n"\
" --help              display this message\n"\
" --detector          IFOs to use e.g. H1\n"\
" --par-file          pulsar parameter (.par) file (full path)\n"\
" --start             start time GPS\n"\
" --timespan          time span to calculate over (seconds)\n"\
" --deltat            length of each step (seconds)\n"\
" --output-dir        output directory\n"\
"\n"

typedef struct tagInputParams{
  CHAR *det;
  CHAR *parfile;

  INT4 start;
  INT4 timespan;
  INT4 deltat;

  CHAR *outputdir;
}InputParams;

void get_input_args(InputParams *inputParams, INT4 argc, CHAR *argv[]);

void get_freq( double start, double deltaT, PulsarParameters *params, BarycenterInput bary,
               EphemerisData *edat, TimeCorrectionData *tdat, TimeCorrectionType ttype, REAL8Vector *freqs,
               REAL8Vector *dfsolar, REAL8Vector *dfbinary, REAL8Vector *dftotal );

int main(int argc, char *argv[]){
  InputParams inputs;

  FILE *fp = NULL;

  PulsarParameters *params;

  UINT4 i = 0, npoints = 0;

  EphemerisData *edat = NULL;
  TimeCorrectionData *tdat = NULL;
  LALDetector det;
  BarycenterInput baryinput;

  CHAR outputfile[256];

  get_input_args(&inputs, argc, argv);

  npoints = floor((REAL8)inputs.timespan/(REAL8)inputs.deltat);

  params = XLALReadTEMPOParFileNew( inputs.parfile );

  /* set up ephemerises */
  det = *XLALGetSiteInfo( inputs.det ); /* just set site as LHO */
  baryinput.site = det;
  baryinput.site.location[0] /= LAL_C_SI;
  baryinput.site.location[1] /= LAL_C_SI;
  baryinput.site.location[2] /= LAL_C_SI;

  CHAR *earthfile = NULL, *sunfile = NULL, *timefile = NULL;
  TimeCorrectionType ttype;

  ttype = XLALAutoSetEphemerisFiles( &earthfile, &sunfile, &timefile, params, inputs.start, inputs.start + inputs.timespan );

  edat = XLALInitBarycenter( earthfile, sunfile );
  tdat = XLALInitTimeCorrections( timefile );

  /* vector to hold frequency time series and Doppler shift time series' */
  REAL8Vector *freqs = NULL, *dopplerss = NULL, *dopplerbs = NULL, *doppler = NULL;
  freqs = XLALCreateREAL8Vector( npoints );
  dopplerss = XLALCreateREAL8Vector( npoints );
  dopplerbs = XLALCreateREAL8Vector( npoints );
  doppler = XLALCreateREAL8Vector( npoints );

  /* calculate the frequency every minute for the mean values */
  get_freq( inputs.start, (double)inputs.deltat, params, baryinput, edat, tdat, ttype, freqs, dopplerss, dopplerbs, doppler );

  sprintf(outputfile, "%s/frequency_evolution_%s.txt", inputs.outputdir, inputs.det);

  fp = fopen(outputfile, "w");
  for( i=0; i < freqs->length; i++ ){
    fprintf(fp, "%lf\t%.15lf\t%.10e\t%.10e\t%.10e\n", inputs.start + (double)(i*inputs.deltat), freqs->data[i],
            dopplerss->data[i], dopplerbs->data[i], doppler->data[i]);
  }
  fclose(fp);

  XLALDestroyREAL8Vector(freqs);
  XLALDestroyREAL8Vector(dopplerss);
  XLALDestroyREAL8Vector(dopplerbs);
  XLALDestroyREAL8Vector(doppler);

  PulsarFreeParams( params );

  return 0;
}

void get_input_args(InputParams *inputParams, INT4 argc, CHAR *argv[]){
  struct option long_options[] =
  {
    { "help",       no_argument,       0, 'h' },
    { "detector",   required_argument, 0, 'D' },
    { "par-file",   required_argument, 0, 'P' },
    { "start",      required_argument, 0, 's' },
    { "timespan",   required_argument, 0, 't' },
    { "deltat",     required_argument, 0, 'd' },
    { "output-dir", required_argument, 0, 'o' },
    { 0, 0, 0, 0 }
  };

  CHAR args[] = "hD:P:s:t:d:o:";
  CHAR *program = argv[0];

  while( 1 ){
    INT4 option_index = 0;
    INT4 c;

    c = getopt_long_only( argc, argv, args, long_options, &option_index );
    if( c == -1 ) /* end of options */
      break;

    switch( c ){
      case 0:
        if( long_options[option_index].flag )
          break;
        else
          fprintf(stderr, "Error passing option %s with argument %s\n",
            long_options[option_index].name, optarg);
      case 'h': /* help message */
        fprintf(stderr, CODEUSAGE, program);
        exit(0);
      case 'D':
        inputParams->det = XLALStringDuplicate(optarg);
        break;
      case 'P':
        inputParams->parfile = XLALStringDuplicate(optarg);
        break;
      case 's':
        inputParams->start = atoi(optarg);
        break;
      case 't':
        inputParams->timespan = atoi(optarg);
        break;
      case 'd':
        inputParams->deltat = atoi(optarg);
        break;
      case 'o':
        inputParams->outputdir = XLALStringDuplicate(optarg);
        break;
      case '?':
        fprintf(stderr, "Unknown error while parsing options\n");
      default:
        fprintf(stderr, "Unknown error while parsing options\n");
    }
  }
}

/* function to return a vector of the pulsar frequency for each data point */
void get_freq( double start, double deltaT,
               PulsarParameters *params, BarycenterInput bary, EphemerisData *edat,
               TimeCorrectionData *tdat, TimeCorrectionType ttype,
               REAL8Vector *freqs, REAL8Vector *dfsolar, REAL8Vector *dfbinary,
               REAL8Vector *dftotal ){
  UINT4 i = 0;

  REAL8 T0 = 0., DT = 0.;
  REAL8 time0 = 0., timep1 = 0.;

  EarthState earth, earth2;
  EmissionTime emit, emit2;
  BinaryPulsarInput binput;
  BinaryPulsarOutput boutput, boutput2;

  /* if edat is NULL then show error */
  if( edat == NULL){
    XLALPrintError ("%s: Ephemeris does not appear to be initialised!", __func__ );
    XLAL_ERROR_VOID( XLAL_EINVAL );
  }

  /* get right ascension and declination */
  REAL8 ra = 0.;
  if ( PulsarCheckParam( params, "RA" ) ) { ra = PulsarGetREAL8Param( params, "RA" ); }
  else if ( PulsarCheckParam( params, "RAJ" ) ) { ra = PulsarGetREAL8Param( params, "RAJ" ); }
  else {
    XLALPrintError ("%s: No source right ascension specified!", __func__ );
    XLAL_ERROR_VOID( XLAL_EINVAL );
  }
  REAL8 dec = 0.;
  if ( PulsarCheckParam( params, "DEC" ) ) { dec = PulsarGetREAL8Param( params, "DEC" ); }
  else if ( PulsarCheckParam( params, "DECJ" ) ) { dec = PulsarGetREAL8Param( params, "DECJ" ); }
  else {
    XLALPrintError ("%s: No source declination specified!", __func__ );
    XLAL_ERROR_VOID( XLAL_EINVAL );
  }

  REAL8 pmra = PulsarGetREAL8ParamOrZero( params, "PMRA" );
  REAL8 pmdec = PulsarGetREAL8ParamOrZero( params, "PMDEC" );
  REAL8 pepoch = PulsarGetREAL8ParamOrZero( params, "PEPOCH" );
  REAL8 posepoch = PulsarGetREAL8ParamOrZero( params, "POSEPOCH" );
  REAL8 px = PulsarGetREAL8ParamOrZero( params, "PX" );     /* parallax */
  REAL8 dist = PulsarGetREAL8ParamOrZero( params, "DIST" ); /* distance */

  /* set 1/distance if parallax or distance value is given (1/sec) */
  if( px != 0. ) { bary.dInv = px*1e-3*LAL_C_SI/LAL_PC_SI; }
  else if( dist != 0. ) { bary.dInv = LAL_C_SI/(dist*1e3*LAL_PC_SI); }
  else { bary.dInv = 0.; }

   /* set the position and frequency epochs if not already set */
  if( pepoch == 0. && posepoch != 0.) { pepoch = posepoch; }
  else if( posepoch == 0. && pepoch != 0. ) { posepoch = pepoch; }

  /* get frequencies */
  REAL8 f0, f1, f2, f3, f4, f5;

  f0 = PulsarGetREAL8ParamOrZero(params, "F0");
  f1 = PulsarGetREAL8ParamOrZero(params, "F1");
  f2 = PulsarGetREAL8ParamOrZero(params, "F2");
  f3 = PulsarGetREAL8ParamOrZero(params, "F3");
  f4 = PulsarGetREAL8ParamOrZero(params, "F4");
  f5 = PulsarGetREAL8ParamOrZero(params, "F5");

  for( i=0; i< freqs->length; i++ ){
    LIGOTimeGPS tgps;

    T0 = pepoch;
    time0 = start + deltaT*(double)i;
    timep1 = time0 + 1.;

    DT = time0 - T0;

    bary.tgps.gpsSeconds = (UINT8)floor(time0);
    bary.tgps.gpsNanoSeconds = (UINT8)floor((fmod(time0,1.)*1e9));

    bary.delta = dec + DT*pmdec;
    bary.alpha = ra + DT*pmra/cos(bary.delta);

    /* call barycentring routines */
    XLALBarycenterEarthNew( &earth, &bary.tgps, edat, tdat, ttype );
    XLALBarycenter( &emit, &bary, &earth );

    /* add 1 sec so we can get doppler shift */
    bary.tgps.gpsSeconds = (UINT8)floor(timep1);
    bary.tgps.gpsNanoSeconds = (UINT8)floor((fmod(timep1,1.)*1e9));

    XLALBarycenterEarthNew( &earth2, &bary.tgps, edat, tdat, ttype );
    XLALBarycenter( &emit2, &bary, &earth2 );

    /* work out frequency (assuming stationary at barycentre) */
    freqs->data[i] = 2.*(f0 + f1*DT
      + 0.5*f2*DT*DT
      + (1./6.)*f3*DT*DT*DT
      + (1./24.)*f4*DT*DT*DT*DT
      + (1./120.)*f5*DT*DT*DT*DT*DT);

    /* get solar system doppler shift and add it on */
    dfsolar->data[i] = (emit2.deltaT-emit.deltaT)*freqs->data[i];

    /* get binary system doppler shift */
    if ( PulsarCheckParam( params, "BINARY" ) ){
      binput.tb = time0 + emit.deltaT;
      XLALGPSSetREAL8(&tgps, time0);
      get_earth_pos_vel( &earth, edat, &tgps );
      binput.earth = earth;
      XLALBinaryPulsarDeltaTNew( &boutput, &binput, params );

      binput.tb = timep1 + emit2.deltaT;
      XLALGPSSetREAL8(&tgps, timep1);
      get_earth_pos_vel( &earth, edat, &tgps );
      binput.earth = earth2;
      XLALBinaryPulsarDeltaTNew( &boutput2, &binput, params );

      dfbinary->data[i] = (boutput2.deltaT-boutput.deltaT)*freqs->data[i];
    }
    else{ dfbinary->data[i] = 0.; }

    dftotal->data[i] = dfbinary->data[i] + dfsolar->data[i];

    freqs->data[i] += dftotal->data[i];
  }
}

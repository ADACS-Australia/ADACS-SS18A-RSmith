/*
*  Copyright (C) 2007 Bernd Machenschalk, Thomas Essinger-Hileman, Benjamin Owen, Robert Adam Mercer, Reinhard Prix
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
 * \file
 * \ingroup pulsarApps
 * \author Thomas Essinger-Hileman, Benjamin Owen
 */


#include <getopt.h>
#include <math.h>
#include <strings.h>
#include <stdio.h>

#include <lalapps.h>

#include <lal/AVFactories.h>
#include <lal/LALBarycenter.h>
#include <lal/LALConfig.h>
#include <lal/LALDatatypes.h>
#include <lal/LALInitBarycenter.h>
#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/LALXMGRInterface.h>
#include <lal/PtoleMetric.h>
#include <lal/StackMetric.h>
#include <lal/TwoDMesh.h>


RCSID( "$Id$" );
 
/* type of parameter space metric to use  */
enum {
  undefined,     /* duh */
  ptolemetric,   /* PtoleMetric() */
  baryptolemaic, /* CoherentMetric() plus TBaryPtolemaic()  */
  ephemeris     /* CoherentMetric() plus TEphemeris() */
} metric_type = undefined;


enum {
  undetermined,
  hanford,
  livingston,
  geo,
  tama,
  virgo
} detector = undetermined;


/* Limits of sky search  */
REAL4 ra_min  = 0.0;
REAL4 ra_max  = LAL_TWOPI; /* BEN: need to fix this */
REAL4 dec_min = -LAL_PI_2;
REAL4 dec_max = LAL_PI_2;
REAL4 MAX_NODES = 8e6; /* limit on number of nodes for TwoDMesh  */

REAL8Vector *tevlambda;


int main( int argc, char *argv[] )
{

  LALStatus stat = blank_status;  /* status structure */
  static REAL8Vector *outputMetric; /* output argument for PulsarMetric */
  PtoleMetricIn search; /* input structure for PulsarMetric() */


  /* Define input variables and set default values */
  int begin            = 731265908;  /* start time of integration */
  REAL4 duration       = 1.8e5;      /* duration of integration (seconds) */
  // currently unused: REAL4 min_spindown   = 1e10;       /* minimum spindown age (seconds) */
  // currently unused: int spindown_order   = 0;          /* minimum spindown order */
  // currently unused: REAL4 mismatch       = 0.05;       /* mismatch threshold of mesh */
  REAL4 max_frequency  = 1e3;        /* maximum frequency of search (Hz) */

  /* other useful variables */
  int option_index = 0;  /* getopt_long option index */
  int opt;               /* Argument for switch statement with getopt_long */
  int detector_argument; /* setting of detector location */
  int j,k;
  CHAR earth[] = "earth00-04.dat";
  CHAR sun[] = "sun00-04.dat";
  LALDetector this_detector;

  /* Set getopt_long option arguments */
  static struct option long_options[] = {
    {"metric-type",       1, 0, 1},
    {"start-gps-seconds", 1, 0, 2},
    {"detector",          1, 0, 3},
    {"debug-level",       1, 0, 4},
    {"integration-time",  1, 0, 5},
    {"min-spindown-age",  1, 0 ,6},
    {"spindown-order",    1, 0, 7},
    {"mismatch",          1, 0, 8},
    {"max-frequency",     1, 0, 9},
    {0, 0, 0, 0}
  };
  EphemerisData *eph = (EphemerisData *)LALMalloc(sizeof(EphemerisData));
  eph->ephiles.earthEphemeris = earth;
  eph->ephiles.sunEphemeris = sun;

  /* Set up. */
  lal_errhandler = LAL_ERR_EXIT;
  search.epoch.gpsSeconds = begin;
  search.duration = duration;
  search.maxFreq = max_frequency;
  search.ephemeris = eph;
  search.position.system = COORDINATESYSTEM_EQUATORIAL;

  /* Parse command-line options. */
  while( (opt = getopt_long( argc, argv, "a:bc:defghjk", long_options, &option_index )) != -1 )
  {
    
    switch ( opt ) {

    case 1: /* Set type of metric for LALMetricWrapper */
      if( !strcmp( optarg, "ptolemetric" ) )
        metric_type = ptolemetric;
      if( !strcmp( optarg, "baryptolemaic" ) )
	metric_type = baryptolemaic;
      if( !strcmp( optarg, "ephemeris" ) )
	metric_type = ephemeris;
      break;

    case 2: /* start-gps-seconds option */
      begin = atoi ( optarg );
      break;

    case 3: /* Set detector site for LALMetricWrapper */
      if( !strcmp( optarg, "hanford" ) )
	detector = hanford;
      if( !strcmp( optarg, "livingston" ) )
	detector = livingston;
      if( !strcmp( optarg, "geo" ) )
	detector = geo;
      if( !strcmp( optarg, "tama" ) )
	detector = tama;
      if( !strcmp( optarg, "virgo" ) )
	detector = virgo;
      break;

    case 4:
      set_debug_level( optarg );
      break;

    case 5:
      duration = atoi (optarg);
      break;

    case 6:
      // currently unused: min_spindown = atoi ( optarg );
      break;

    case 7:
      // currently unused: spindown_order = atoi ( optarg );
      break;

    case 8:
      // currently unused: mismatch = atof( optarg );
      break;

    case 9:
      max_frequency = atoi( optarg );
      break;

    }/*switch( opt )*/

  }/*while ( getopt... )*/
printf( "parsed options...\n" );


 /* Set metric type  */
 switch( metric_type ) {

 case ptolemetric: 
   search.metricType = LAL_PMETRIC_COH_PTOLE_ANALYTIC;
   break;

 case baryptolemaic:
   search.metricType = LAL_PMETRIC_COH_PTOLE_NUMERIC;
   break;

 case ephemeris:
   search.metricType = LAL_PMETRIC_COH_EPHEM;
   break;

 default:
   printf( "Invalid metric type\n" );
   exit(1);

 } 
 printf( "Set metric type\n" );


 /* set detector location  */
 switch( detector ) {

 case hanford:
   detector_argument = LALDetectorIndexLHODIFF;
   break;

 case livingston:
   detector_argument = LALDetectorIndexLLODIFF;
   break;

 case geo:
   detector_argument = LALDetectorIndexGEO600DIFF;
   break;

 case tama:
   detector_argument = LALDetectorIndexTAMA300DIFF;
   break;

 case virgo:
   detector_argument = LALDetectorIndexVIRGODIFF;
   break;

 default:
   printf( "Invalid detector argument\n" );
   exit(1);

 }
 printf( "Set detector location\n" );

 this_detector = lalCachedDetectors[detector_argument];
 search.site = &this_detector;

 LAL_CALL( LALPulsarMetric( &stat, &outputMetric, &search ), &stat );

 /* Print metric  */
 printf( "\nmetric at the requested point\n" );
   for (j=0; j<=2; j++) {
     for (k=0; k<=j; k++) 
       printf(" %f ", outputMetric->data[k+j*(j+1)/2] );
     printf("\n");
   }


   /* clean up and leave */
   LALFree( search.ephemeris->ephemE );
   LALFree( search.ephemeris->ephemS );

  LALCheckMemoryLeaks();
  return 0;
} /* main() */

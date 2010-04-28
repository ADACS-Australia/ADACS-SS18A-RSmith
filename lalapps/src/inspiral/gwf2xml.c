/*
*  Copyright (C) 2007 Stephen Fairhurst
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

/*----------------------------------------------------------------------- 
 * 
 * File Name: gwf2xml.c
 *
 * Author: Fairhurst, S
 *
 * Revision: $Id$
 * 
 *-----------------------------------------------------------------------
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <lal/LALFrameL.h>
#include <lal/Date.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOMetadataTables.h>

#define SNR_MIN 6.
#define SNR_MAX 1.e+10  
#define SIM_MIN 1.0e-30  
#define SIM_MAX 1.0e+10


#define USAGE \
"Usage: gwf2xml [options]\n"\
  "\n"\
  "  --help                 display this message\n"\
  "  --input FILE           read input data from frame FILE\n"\
  "  --output FILE          write output data to xml FILE\n"\
  "  --snr-threshold SNR    set the minimum SNR of triggers (default 6)\n"\
  "  --ifo IFO              set the IFO from which the triggers have come\n"\

char *ifo = NULL;

/* FIXME: work around for FrameL const string issue */
static char * ignore_const(const char *s)
{
  union {const char *c; char *s; } u = {s};
  return u.s;
}

static int frEvent2snglInspiral(SnglInspiralTable **snglInspiralEvent, 
    FrEvent *frameEvent )
{
  FrEvent              *frEvt   = NULL;
  SnglInspiralTable    *snglEvt = NULL;
  int                   numEvt  = 0;
  double                timeAfter = 0;

  /* If we already have events in snglInspiralEvent, 
   * wind on to the end of the list */
  for( snglEvt = *snglInspiralEvent; snglEvt; snglEvt=snglEvt->next);

  /* store the frameEvents in the snglInspiral linked list */
  for(frEvt=frameEvent; frEvt; frEvt=frEvt->next, ++numEvt) 
  {
    if ( !(*snglInspiralEvent) )
    {
      *snglInspiralEvent = snglEvt = (SnglInspiralTable * ) 
        LALCalloc( 1, sizeof(SnglInspiralTable) );
    }
    else
    {
      snglEvt = snglEvt->next = (SnglInspiralTable * ) 
        LALCalloc( 1, sizeof(SnglInspiralTable) );
    }

    /* read data from the frEvt */
    snprintf(snglEvt->search, LIGOMETA_SEARCH_MAX, "%s", frEvt->name);
    snglEvt->snr = frEvt->amplitude;
    snglEvt->end_time.gpsSeconds = frEvt->GTimeS;
    snglEvt->end_time.gpsNanoSeconds = frEvt->GTimeN;
    timeAfter = frEvt->timeAfter;
    XLALGPSAdd(&snglEvt->end_time,timeAfter);
    snglEvt->eff_distance = FrEventGetParam ( frEvt, ignore_const("distance (Mpc)") );
    snglEvt->mass1 = FrEventGetParam ( frEvt, ignore_const("mass1") );
    snglEvt->mass2 = FrEventGetParam ( frEvt, ignore_const("mass2") );
    snglEvt->tau0 =FrEventGetParam ( frEvt, ignore_const("tau0") );
    snglEvt->tau3 = FrEventGetParam ( frEvt, ignore_const("tau1p5") );
    snglEvt->coa_phase = FrEventGetParam ( frEvt, ignore_const("phase") );
    snglEvt->chisq = FrEventGetParam ( frEvt, ignore_const("chi2") );

    /* populate additional colums */
    snglEvt->mtotal = snglEvt->mass1 + snglEvt->mass2;
    snglEvt->eta = (snglEvt->mass1 * snglEvt->mass2) /
      (snglEvt->mtotal * snglEvt->mtotal);
    snglEvt->mchirp = pow( snglEvt->eta, 0.6) * snglEvt->mtotal;
    snprintf(snglEvt->ifo, LIGOMETA_IFO_MAX, "%s", ifo);
  }
  return( numEvt );
}


static int frSimEvent2simInspiral (SimInspiralTable **simInspiralEvent,
    FrSimEvent       *frSimEvent )
{
  FrSimEvent           *frSimEvt   = NULL;
  SimInspiralTable     *simEvt     = NULL;
  int                   numSim     = 0;

  /* If we already have events in snglInspiralEvent, 
   * wind on to the end of the list */
  for( simEvt = *simInspiralEvent; simEvt; simEvt = simEvt->next);

  /* store the frameEvents in the snglInspiral linked list */
  for( frSimEvt = frSimEvent; frSimEvt; frSimEvt = frSimEvt->next, ++numSim) 
  {
    if ( !(*simInspiralEvent) )
    {
      *simInspiralEvent = simEvt = (SimInspiralTable * ) 
        LALCalloc( 1, sizeof(SimInspiralTable) );
    }
    else
    {
      simEvt = simEvt->next = (SimInspiralTable * ) 
        LALCalloc( 1, sizeof(SimInspiralTable) );
    }

    /* read data from the frSimEvt */
    snprintf(simEvt->waveform, LIGOMETA_SEARCH_MAX, "%s", frSimEvt->name);
    simEvt->geocent_end_time.gpsSeconds = frSimEvt->GTimeS;
    simEvt->geocent_end_time.gpsNanoSeconds = frSimEvt->GTimeN;
    simEvt->v_end_time = simEvt->geocent_end_time;


    simEvt->distance = FrSimEventGetParam ( frSimEvt, ignore_const("distance") );
    simEvt->eff_dist_v = simEvt->distance;

    simEvt->mass1 = FrSimEventGetParam ( frSimEvt, ignore_const("m1") );
    simEvt->mass2 = FrSimEventGetParam ( frSimEvt, ignore_const("m2") );
  }

  return ( numSim );
}

int main( int argc, char *argv[] )
{
  /* lal initialization variables */
  static LALStatus stat;

  /*  program option variables */
  char *inputFileName = NULL;
  char *outputFileName = NULL;


  /* frame data structures */
  FrFile *iFile;
  FrEvent *frameEvent = NULL;
  FrSimEvent *frSimEvent = NULL;

  double tStart;
  double tEnd;
  double duration;
  double snrMin = SNR_MIN;
  double snrMax = SNR_MAX;

  double simMin = SIM_MIN;
  double simMax = SIM_MAX;

  int numEvt = 0;
  int numSim = 0;
  /* xml data structures */

  SnglInspiralTable    *snglInspiralEvent = NULL;
  SnglInspiralTable    *snglEvt = NULL;

  SimInspiralTable     *simInspiralEvent = NULL;
  SimInspiralTable     *simEvt = NULL;

  LIGOLwXMLStream       xmlStream;
  MetadataTable         outputTable;
  MetadataTable         searchsumm;

  /*
   *
   * parse command line arguments
   *
   */


  while (1)
  {
    /* getopt arguments */
    static struct option long_options[] = 
    {
      {"help",                    no_argument,            0,              'h'},
      {"input",                   required_argument,      0,              'i'},
      {"output",                  required_argument,      0,              'o'},
      {"snr-threshold",           required_argument,      0,              's'},
      {"ifo",                     required_argument,      0,              'd'},
      {0, 0, 0, 0}
    };
    int c;

    /* getopt_long stores the option index here. */
    int option_index = 0;
    size_t optarg_len;

    c = getopt_long_only ( argc, argv, "hi:o:s:", 
        long_options, &option_index );

    /* detect the end of the options */
    if ( c == - 1 )
      break;

    switch ( c )
    {
      case 0:
        /* if this option set a flag, do nothing else now */
        if ( long_options[option_index].flag != 0 )
        {
          break;
        }
        else
        {
          fprintf( stderr, "error parsing option %s with argument %s\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'h':
        fprintf( stdout, USAGE );
        exit( 0 );
        break;

      case 'i':
        /* create storage for the input file name */
        optarg_len = strlen( optarg ) + 1;
        inputFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( inputFileName, optarg, optarg_len );
        break;

      case 'o':
        /* create storage for the output file name */
        optarg_len = strlen( optarg ) + 1;
        outputFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( outputFileName, optarg, optarg_len );
        break;

    case 'd':
        /* create storage for the output file name */
        optarg_len = strlen( optarg ) + 1;
        ifo = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( ifo, optarg, optarg_len );
        break;

      case 's':
        snrMin = (double) atof( optarg );
        if ( snrMin < 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "threshold must be >= 0: "
              "(%f specified)\n",
              long_options[option_index].name, snrMin );
          exit( 1 );
        }
        break;

      case '?':
        exit( 1 );
        break;

      default:
        fprintf( stderr, "unknown error while parsing options\n" );
        exit( 1 );
    }   
  }

  if ( optind < argc )
  {
    fprintf( stderr, "extraneous command line arguments:\n" );
    while ( optind < argc )
    {
      fprintf ( stderr, "%s\n", argv[optind++] );
    }
    exit( 1 );
  }

  /*
   *
   * read in the triggers from the input frame file
   *
   */

  iFile = FrFileINew(inputFileName);

  /* set start time, duration and amplitude window 
     this should be turned into command line arguments */

  tStart   = FrFileITStart(iFile);
  tEnd     = FrFileITEnd(iFile);
  duration = tEnd - tStart;

  /* read in the events */
  frameEvent = FrEventReadT(iFile, ignore_const("*clustered"), tStart, duration, 
      snrMin, snrMax);


  /*Write out details of events to SnglInspiralTable*/
  numEvt = frEvent2snglInspiral( &snglInspiralEvent, frameEvent); 

  fprintf( stdout, "Read in %d triggers from frEvent structure in %s\n",
      numEvt, inputFileName );
  
  /* free the frame events */
  FrEventFree(frameEvent);


  /*
   *
   * read in the simEvent from the input frame file
   *
   */

  frSimEvent  = FrSimEventReadT (iFile, ignore_const("cb*"), tStart, duration, 
      simMin, simMax);

  /*Write out details of events to SnglInspiralTable*/
  numSim = frSimEvent2simInspiral( &simInspiralEvent, frSimEvent); 

  fprintf( stdout, "Read in %d injections from frEvent structure in %s\n",
      numSim, inputFileName );

  /* free the frame events */
  FrSimEventFree(frSimEvent);


        /* 
         *
         * write a search summary table
         *
         */

        /* create the search summary and zero out the summvars table */
  searchsumm.searchSummaryTable = (SearchSummaryTable *)
    calloc( 1, sizeof(SearchSummaryTable) );

          
        /* create the search summary and zero out the summvars table */
  searchsumm.searchSummaryTable = (SearchSummaryTable *)
    calloc( 1, sizeof(SearchSummaryTable) );

        searchsumm.searchSummaryTable->in_start_time.gpsSeconds = tStart;
  searchsumm.searchSummaryTable->in_end_time.gpsSeconds = tEnd;
  
  searchsumm.searchSummaryTable->out_start_time.gpsSeconds = tStart;
        searchsumm.searchSummaryTable->out_end_time.gpsSeconds = tEnd;
  searchsumm.searchSummaryTable->nnodes = 1;
        if (numEvt)
        {
          searchsumm.searchSummaryTable->nevents = numEvt;
        }
        else if (numSim)
        {
          searchsumm.searchSummaryTable->nevents = numSim;
        }
        

  /*
   *
   * write output data to xml file
   *
   */

  /* write xml output file */
  memset( &xmlStream, 0, sizeof(LIGOLwXMLStream) );
  LALOpenLIGOLwXMLFile( &stat, &xmlStream, outputFileName );

        /* Write search_summary table */
    LALBeginLIGOLwXMLTable( &stat, &xmlStream, 
          search_summary_table );
    LALWriteLIGOLwXMLTable( &stat, &xmlStream, searchsumm, 
          search_summary_table );
    LALEndLIGOLwXMLTable ( &stat, &xmlStream );


  /* Write the results to the inspiral table */
  if ( snglInspiralEvent )
  {
    outputTable.snglInspiralTable = snglInspiralEvent;
    LALBeginLIGOLwXMLTable( &stat, &xmlStream, sngl_inspiral_table );
    LALWriteLIGOLwXMLTable( &stat, &xmlStream, outputTable, 
        sngl_inspiral_table );
    LALEndLIGOLwXMLTable( &stat, &xmlStream );
  }

  /* Write the results to the sim inspiral table */
  if ( simInspiralEvent )
  {
    outputTable.simInspiralTable = simInspiralEvent;
    LALBeginLIGOLwXMLTable( &stat, &xmlStream, sim_inspiral_table );
    LALWriteLIGOLwXMLTable( &stat, &xmlStream, outputTable, 
        sim_inspiral_table );
    LALEndLIGOLwXMLTable( &stat, &xmlStream );
  }

  /* close the output file */
  LALCloseLIGOLwXMLFile(&stat, &xmlStream);


  /*
   *
   * free memory and exit
   *
   */


  /* free the inspiral events we saved */
  while ( snglInspiralEvent )
  {
    snglEvt = snglInspiralEvent;
    snglInspiralEvent = snglInspiralEvent->next;
    LALFree( snglEvt );
  }

  /* free the sim inspiral events we saved */
  while ( simInspiralEvent )
  {
    simEvt = simInspiralEvent;
    simInspiralEvent = simInspiralEvent->next;
    LALFree( simEvt );
  }

  LALCheckMemoryLeaks();
  exit( 0 );
}

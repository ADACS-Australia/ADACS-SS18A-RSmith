/*
*  Copyright (C) 2007 Duncan Brown, Kipp Cannon, Lisa M. Goggin
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
 * File Name: ringread.c
 *
 * Author: Goggin, L. M. based on sire.c by Brady, P. R, Brown, D. A., and Fairhurst, S
 * 
 * Revision: $Id$
 * 
 *-----------------------------------------------------------------------
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <regex.h>
#include <time.h>
#include <glob.h>
#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/Date.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataRingdownUtils.h>
#include <lal/LIGOLwXMLRingdownRead.h>
#include <lalapps.h>
#include <processtable.h>
#include <LALAppsVCSInfo.h>

RCSID("$Id$");

#define PROGRAM_NAME "ringread.c"
#define CVS_ID_STRING "$Id$"
#define CVS_REVISION "$Revision$"
#define CVS_SOURCE "$Source$"
#define CVS_DATE "$Date$"

#define USAGE \
  "Usage: lalapps_ringread [options]\n"\
"\n"\
"  --help                       display this message\n"\
"  --verbose                    print progress information\n"\
"  --debug-level LEVEL          set the LAL debug level to LEVEL\n"\
"  --user-tag STRING            set the process_params usertag to STRING\n"\
"  --comment STRING             set the process table comment to STRING\n"\
"  --version                    print the CVS version string\n"\
"\n"\
"Input data source:\n"\
"  --glob GLOB                  use pattern GLOB to determine the input files\n"\
"  --input FILE                 read list of input XML files from FILE\n"\
"\n"\
"Output data destination:\n"\
"  --output FILE                write output data to FILE\n"\
"  --tama-output FILE           write out text triggers for tama\n"\
"  --summary-file FILE          write trigger analysis summary to FILE\n"\
"  --data-type                   specify the data type, must be one of\n"\
"                               (playground_only|exclude_play|all_data)\n"\
"\n"\
"Clustering and Sorting:\n"\
"  --sort-triggers              time sort the ringdown triggers\n"\
"  --snr-threshold RHO          discard all triggers with snr less than RHO\n"\
"  --cluster-algorithm CHOICE   use trigger clustering algorithm CHOICE\n"\
"                               [ snr_and_chisq | snrsq_over_chisq | snr ]\n"\
"  --cluster-time T             cluster triggers with T ms window\n"\
"  --ifo-cut IFO                only keep triggers from IFO\n"\
"\n"\
"Injection analysis:\n"\
"  --injection-file FILE        read injection parameters from FILE\n"\
"  --injection-coincidence T    trigger and injection coincidence window (ms)\n"\
"  --missed-injections FILE     write sim_ringdown for missed injections to FILE\n"\
"  --hardware-injections GPS    assume hardware injections starting at GPS\n"\
"\n"\
"Maintainer flags:\n"\
"  --disable-trig-start-time    don't modify the search summary table\n"

#define ADD_PROCESS_PARAM( pptype, format, ppvalue ) \
  this_proc_param = this_proc_param->next = (ProcessParamsTable *) \
calloc( 1, sizeof(ProcessParamsTable) ); \
snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", \
    PROGRAM_NAME ); \
snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--%s", \
    long_options[option_index].name ); \
snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "%s", pptype ); \
snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, format, ppvalue );

#define MAX_PATH 4096

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

/* function to read the next line of data from the input file list */
char *get_next_line( char *line, size_t size, FILE *fp );
char *get_next_line( char *line, size_t size, FILE *fp )
{
  char *s;
  do
    s = fgets( line, size, fp );
  while ( ( line[0] == '#' || line[0] == '%' ) && s );
  return s;
}

int sortTriggers = 0;
LALPlaygroundDataMask dataType;
extern int vrbflg;

int main( int argc, char *argv[] )
{
  /* lal initialization variables */
  LALStatus status = blank_status;

  /*  program option variables */
  CHAR *userTag = NULL;
  CHAR comment[LIGOMETA_COMMENT_MAX];
  char *ifoName = NULL;
  char *inputGlob = NULL;
  char *inputFileName = NULL;
  char *outputFileName = NULL;
  char *tamaFileName = NULL;
  char *summFileName = NULL;
  REAL4 snrStar = -1;
  SnglInspiralClusterChoice clusterchoice = none;
  INT8 cluster_dt = -1;
  char *injectFileName = NULL;
  INT8 inject_dt = -1;
  char *missedFileName = NULL;
  INT4 hardware = 0;
  int  enableTrigStartTime = 1;
  int j;
  FILE *fp = NULL;
  glob_t globbedFiles;
  int numInFiles = 0;
  char **inFileNameList;
  char line[MAX_PATH];
  int  errnum;
  static const char *func = "XLALSnglRingdownTableFromLIGOLw"; 

  UINT8 triggerInputTimeNS = 0;

  MetadataTable         proctable;
  MetadataTable         procparams;
  ProcessParamsTable   *this_proc_param;

  UINT4                 numSimEvents = 0;
  UINT4                 numSimInData = 0;
  UINT4                 numSimFound  = 0;
  UINT4                 numSimMissed = 0;
  UINT4                 numSimDiscard = 0;
  UINT4                 numSimProcessed = 0;

  SimRingdownTable     *simEventHead = NULL;
  SimRingdownTable     *thisSimEvent = NULL;
  SimRingdownTable     *missedSimHead = NULL;
  SimRingdownTable     *thisMissedSim = NULL;
  SimRingdownTable     *tmpSimEvent = NULL;
  SimRingdownTable     *prevSimEvent = NULL;

  SearchSummaryTable   *searchSummaryTable = NULL;

  UINT4                 numEvents = 0;
  UINT4                 numEventsKept = 0;
  UINT4                 numEventsInIFO = 0;
  UINT4                 numEventsCoinc = 0;
  UINT4                 numEventsDiscard = 0;
  UINT4                 numEventsProcessed = 0;
  UINT4                 numClusteredEvents = 0;

  SnglRingdownTable   **eventHandle = NULL;      
  SnglRingdownTable    *eventHead = NULL;
  SnglRingdownTable    *thisEvent = NULL;
  SnglRingdownTable    *tmpEvent = NULL;
  SnglRingdownTable    *prevEvent = NULL;

  LIGOLwXMLStream       xmlStream;
  MetadataTable         outputTable;


  /*
   *
   * initialization
   *
   */


  /* set up inital debugging values */
  lal_errhandler = LAL_ERR_EXIT;
  set_debug_level( "33" );

  /* create the process and process params tables */
  proctable.processTable = (ProcessTable *) 
    calloc( 1, sizeof(ProcessTable) );
  XLALGPSTimeNow(&(proctable.processTable->start_time));

  XLALPopulateProcessTable(proctable.processTable, PROGRAM_NAME,
      LALAPPS_VCS_IDENT_ID, LALAPPS_VCS_IDENT_STATUS, LALAPPS_VCS_IDENT_DATE, 0);

  this_proc_param = procparams.processParamsTable = (ProcessParamsTable *) 
    calloc( 1, sizeof(ProcessParamsTable) );
  memset( comment, 0, LIGOMETA_COMMENT_MAX * sizeof(CHAR) );


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
      {"verbose",             no_argument,           &vrbflg,              1 },
      {"sort-triggers",       no_argument,     &sortTriggers,              1 },
      {"help",                    no_argument,            0,              'h'},
      {"debug-level",             required_argument,      0,              'z'},
      {"user-tag",                required_argument,      0,              'Z'},
      {"userTag",                 required_argument,      0,              'Z'},
      {"comment",                 required_argument,      0,              'c'},
      {"version",                 no_argument,            0,              'V'},
      {"glob",                    required_argument,      0,              'g'},
      {"input",                   required_argument,      0,              'i'},
      {"output",                  required_argument,      0,              'o'},
      {"data-type",               required_argument,      0,              'k'},
      {"tama-output",             required_argument,      0,              'j'},
      {"summary-file",            required_argument,      0,              'S'},
      {"snr-threshold",           required_argument,      0,              's'},
      {"cluster-algorithm",       required_argument,      0,              'C'},
      {"cluster-time",            required_argument,      0,              't'},
      {"ifo-cut",                 required_argument,      0,              'd'},
      {"injection-file",          required_argument,      0,              'I'},
      {"injection-coincidence",   required_argument,      0,              'T'},
      {"missed-injections",       required_argument,      0,              'm'},
      {"hardware-injections",     required_argument,      0,              'H'},
      {"disable-trig-start-time", no_argument,            0,              'D'},
      {0, 0, 0, 0}
    };
    int c;

    /* getopt_long stores the option index here. */
    int option_index = 0;
    size_t optarg_len;

    c = getopt_long_only ( argc, argv, "hzZ:c:d:g:i:o:j:S:s:C:Vt:I:T:m:H:D", 
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

      case 'z':
        set_debug_level( optarg );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'Z':
        /* create storage for the usertag */
        optarg_len = strlen( optarg ) + 1;
        userTag = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
        memcpy( userTag, optarg, optarg_len );

        this_proc_param = this_proc_param->next = (ProcessParamsTable *)
          calloc( 1, sizeof(ProcessParamsTable) );
        snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "-userTag" );
        snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );
        break;

      case 'c':
        if ( strlen( optarg ) > LIGOMETA_COMMENT_MAX - 1 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "comment must be less than %d characters\n",
              long_options[option_index].name, LIGOMETA_COMMENT_MAX );
          exit( 1 );
        }
        else
        {
          snprintf( comment, LIGOMETA_COMMENT_MAX, "%s", optarg);
        }
        break;

      case 'V':
        fprintf( stdout, "Single Ringdown Reader and Injection Analysis\n"
            "Patrick Brady, Duncan Brown and Steve Fairhurst\n");
        XLALOutputVersionString(stderr, 0);
        exit( 0 );
        break;

      case 'g':
        /* create storage for the input file glob */
        optarg_len = strlen( optarg ) + 1;
        inputGlob = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( inputGlob, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "'%s'", optarg );
        break;

      case 'i':
        /* create storage for the input file name */
        optarg_len = strlen( optarg ) + 1;
        inputFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( inputFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'o':
        /* create storage for the output file name */
        optarg_len = strlen( optarg ) + 1;
        outputFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( outputFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'j':
        /* create storage of the TAMA file name */
        optarg_len = strlen( optarg ) + 1;
        tamaFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( tamaFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'S':
        /* create storage for the summ file name */
        optarg_len = strlen( optarg ) + 1;
        summFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( summFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 's':
        snrStar = (REAL4) atof( optarg );
        if ( snrStar < 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "threshold must be >= 0: "
              "(%f specified)\n",
              long_options[option_index].name, snrStar );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", snrStar );
        break;

      case 'k':
        /* type of data to analyze */
        if ( ! strcmp( "playground_only", optarg ) )
        {
          dataType = playground_only;
        }
        else if ( ! strcmp( "exclude_play", optarg ) )
        {
          dataType = exclude_play;
        }
        else if ( ! strcmp( "all_data", optarg ) )
        {
          dataType = all_data;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown data type, %s, specified: "
              "(must be playground_only, exclude_play or all_data)\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'C':
        /* choose the clustering algorithm */
        {        
          if ( ! strcmp( "snr_and_chisq", optarg ) )
          {
            clusterchoice = snr_and_chisq;
          }
          else if ( ! strcmp( "snrsq_over_chisq", optarg) )
          {
            clusterchoice = snrsq_over_chisq;
          }
          else if ( ! strcmp( "snr", optarg) )
          {
            clusterchoice = snr;
          }        
          else
          {
            fprintf( stderr, "invalid argument to  --%s:\n"
                "unknown clustering specified:\n "
                "%s (must be one of: snr_and_chisq, \n"
                "   snrsq_over_chisq or snr)\n",
                long_options[option_index].name, optarg);
            exit( 1 );
          }
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        break;

      case 't':
        /* cluster time is specified on command line in ms */
        cluster_dt = (INT8) atoi( optarg );
        if ( cluster_dt <= 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "custer window must be > 0: "
              "(%" LAL_INT8_FORMAT " specified)\n",
              long_options[option_index].name, cluster_dt );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%" LAL_INT8_FORMAT "", cluster_dt );
        /* convert cluster time from ms to ns */
        cluster_dt *= LAL_INT8_C(1000000);
        break;

      case 'I':
        /* create storage for the injection file name */
        optarg_len = strlen( optarg ) + 1;
        injectFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( injectFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'd':
        optarg_len = strlen( optarg ) + 1;
        ifoName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( ifoName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'T':
        /* injection coincidence time is specified on command line in ms */
        inject_dt = (INT8) atoi( optarg );
        if ( inject_dt < 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "injection coincidence window must be >= 0: "
              "(%" LAL_INT8_FORMAT " specified)\n",
              long_options[option_index].name, inject_dt );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%" LAL_INT8_FORMAT " ", inject_dt );
        /* convert inject time from ms to ns */
        inject_dt *= LAL_INT8_C(1000000);
        break;

      case 'm':
        /* create storage for the missed injection file name */
        optarg_len = strlen( optarg ) + 1;
        missedFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( missedFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'H':
        hardware = (INT4) atoi( optarg );
        if ( hardware <= 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "GPS start time of hardware injections must be > 0: "
              "(%d specified)\n",
              long_options[option_index].name, hardware );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%" LAL_INT4_FORMAT " ", hardware );
        break;

      case 'D':
        enableTrigStartTime = 0;
        ADD_PROCESS_PARAM( "string", "%s", " " );
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
   * can use LALCalloc() / LALMalloc() from here
   *
   */


  /* don't buffer stdout if we are in verbose mode */
  if ( vrbflg ) setvbuf( stdout, NULL, _IONBF, 0 );

  /* fill the comment, if a user has specified it, or leave it blank */
  if ( ! *comment )
  {
    snprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX, " " );
  }
  else
  {
    snprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX,
        "%s", comment );
  }

  /* check that the input and output file names have been specified */
  if ( (! inputGlob && ! inputFileName) || (inputGlob && inputFileName) )
  {
    fprintf( stderr, "exactly one of --glob or --input must be specified\n" );
    exit( 1 );
  }
  if ( ! outputFileName )
  {
    fprintf( stderr, "--output must be specified\n" );
    exit( 1 );
  }

  /* check that Data Type has been specified */
  if ( dataType == unspecified_data_type )
  {
    fprintf( stderr, "Error: --data-type must be specified\n");
    exit(1);
  }

  /* check that if clustering is being done that we have all the options */
  if ( clusterchoice && cluster_dt < 0 )
  {
    fprintf( stderr, "--cluster-time must be specified if --cluster-algorithm "
        "is given\n" );
    exit( 1 );
  }
  else if ( ! clusterchoice && cluster_dt >= 0 )
  {
    fprintf( stderr, "--cluster-algorithm must be specified if --cluster-time "
        "is given\n" );
    exit( 1 );
  }

  /* check that we have all the options to do injections */
  if ( injectFileName && inject_dt < 0 )
  {
    fprintf( stderr, "--injection-coincidence must be specified if "
        "--injection-file is given\n" );
    exit( 1 );
  }
  else if ( ! injectFileName && inject_dt >= 0 )
  {
    fprintf( stderr, "--injection-file must be specified if "
        "--injection-coincidence is given\n" );
    exit( 1 );
  }

  /* save the sort triggers flag */
  if ( sortTriggers )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *) 
      calloc( 1, sizeof(ProcessParamsTable) ); 
    snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s",
        PROGRAM_NAME ); 
    snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--sort-triggers" );
    snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" ); 
    snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }

  switch ( dataType )
  {
    case playground_only:
      if ( vrbflg )
        fprintf( stdout, "using data from playground times only\n" );
      snprintf( procparams.processParamsTable->program, 
          LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
      snprintf( procparams.processParamsTable->param,
          LIGOMETA_PARAM_MAX, "--playground-only" );
      snprintf( procparams.processParamsTable->type, 
          LIGOMETA_TYPE_MAX, "string" );
      snprintf( procparams.processParamsTable->value, 
          LIGOMETA_TYPE_MAX, " " );
      break;

    case exclude_play:
      if ( vrbflg )
        fprintf( stdout, "excluding all triggers in playground times\n" );
      snprintf( procparams.processParamsTable->program, 
          LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
      snprintf( procparams.processParamsTable->param,
          LIGOMETA_PARAM_MAX, "--exclude-play" );
      snprintf( procparams.processParamsTable->type, 
          LIGOMETA_TYPE_MAX, "string" );
      snprintf( procparams.processParamsTable->value, 
          LIGOMETA_TYPE_MAX, " " );
      break;

    case all_data:
      if ( vrbflg )
        fprintf( stdout, "using all input data\n" );
      snprintf( procparams.processParamsTable->program, 
          LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
      snprintf( procparams.processParamsTable->param,
          LIGOMETA_PARAM_MAX, "--all-data" );
      snprintf( procparams.processParamsTable->type, 
          LIGOMETA_TYPE_MAX, "string" );
      snprintf( procparams.processParamsTable->value, 
          LIGOMETA_TYPE_MAX, " " );
      break;

    default:
      fprintf( stderr, "data set not defined\n" );
      exit( 1 );
  }


  /*
   *
   * read in the injection XML file, if we are doing an injection analysis
   *
   */


  if ( injectFileName )
  {
    if ( vrbflg ) 
      fprintf( stdout, "reading injections from %s... ", injectFileName );

    simEventHead = XLALSimRingdownTableFromLIGOLw( injectFileName, 0, 0 );

    if ( vrbflg ) fprintf( stdout, "got %d injections\n", numSimEvents );

    if ( ! simEventHead )
    {
      fprintf( stderr, "error: unable to read sim_ringdown table from %s\n", 
          injectFileName );
      exit( 1 );
    }

    /* if we are doing hardware injections, increment all the start times */
    if ( hardware )
    {
      if ( vrbflg ) fprintf( stdout, 
          "incrementing GPS times of injections by %d seconds\n", hardware );

      for ( thisSimEvent = simEventHead; 
          thisSimEvent; thisSimEvent = thisSimEvent->next )
      {
        thisSimEvent->geocent_start_time.gpsSeconds += hardware;
        thisSimEvent->h_start_time.gpsSeconds       += hardware;
        thisSimEvent->l_start_time.gpsSeconds       += hardware;
      }
    }

    /* discard all injection events that are not in the data we want */
    if ( dataType != all_data )
    {
      numSimDiscard = 0;

      thisSimEvent = simEventHead;
      simEventHead = NULL;
      prevSimEvent = NULL;

      if ( vrbflg ) fprintf( stdout, "discarding injections not in data\n" );

      while ( thisSimEvent )
      {
        INT4 isPlayground = XLALINT8NanoSecIsPlayground(XLALGPSToINT8NS(&(thisSimEvent->geocent_start_time)));

        if ( (dataType == playground_only && isPlayground) || 
            (dataType == exclude_play && ! isPlayground) )
        {
          /* store the head of the linked list */
          if ( ! simEventHead ) simEventHead = thisSimEvent;

          /* keep this event */
          prevSimEvent = thisSimEvent;
          thisSimEvent = thisSimEvent->next;
          ++numSimInData;
          if ( vrbflg ) fprintf( stdout, "+" );
        }
        else
        {
          /* throw this event away */
          tmpSimEvent = thisSimEvent;
          if ( prevSimEvent ) prevSimEvent->next = thisSimEvent->next;
          thisSimEvent = thisSimEvent->next;
          LALFree( tmpSimEvent );
          ++numSimDiscard;
          if ( vrbflg ) fprintf( stdout, "-" );
        }
      }

      if ( vrbflg ) 
        fprintf( stdout, "\nusing %d (discarded %d) of %d injections\n",
            numSimInData, numSimDiscard, numSimEvents );
    }
    else
    {
      if ( vrbflg ) 
        fprintf( stdout, "using all %d injections\n", numSimInData );
      numSimInData = numSimEvents;
    }
  }


  /*
   *
   * read in the input triggers from the xml files
   *
   */


  if ( inputGlob )
  {
    /* use glob() to get a list of the input file names */
    if ( glob( inputGlob, GLOB_ERR, NULL, &globbedFiles ) )
    {
      fprintf( stderr, "error globbing files from %s\n", inputGlob );
      perror( "error:" );
      exit( 1 );
    }

    numInFiles = globbedFiles.gl_pathc;
    inFileNameList = (char **) LALCalloc( numInFiles, sizeof(char *) );

    for ( j = 0; j < numInFiles; ++j )
    {
      inFileNameList[j] = globbedFiles.gl_pathv[j];
    }
  }
  else if ( inputFileName )
  {
    /* read the list of input filenames from a file */
    fp = fopen( inputFileName, "r" );
    if ( ! fp )
    {
      fprintf( stderr, "could not open file containing list of xml files\n" );
      perror( "error:" );
      exit( 1 );
    }

    /* count the number of lines in the file */
    while ( get_next_line( line, sizeof(line), fp ) )
    {
      ++numInFiles;
    }
    rewind( fp );

    /* allocate memory to store the input file names */
    inFileNameList = (char **) LALCalloc( numInFiles, sizeof(char *) );

    /* read in the input file names */
    for ( j = 0; j < numInFiles; ++j )
    {
      inFileNameList[j] = (char *) LALCalloc( MAX_PATH, sizeof(char) );
      get_next_line( line, sizeof(line), fp );
      strncpy( inFileNameList[j], line, strlen(line) - 1);
    }

    fclose( fp );
  }
  else
  {
    fprintf( stderr, "no input file mechanism specified\n" );
    exit( 1 );
  }

  if ( vrbflg )
  {
    fprintf( stdout, "reading input triggers from:\n" );
    for ( j = 0; j < numInFiles; ++j )
    {
      fprintf( stdout, "%s\n", inFileNameList[j] );
    }
  }


  /*
   *
   * read in the triggers from the input xml files
   *
   */


  if ( injectFileName )
  {
    thisSimEvent = simEventHead;
    simEventHead = NULL;
    prevSimEvent = NULL;
    numSimDiscard = 0;
    numSimInData = 0;

    if ( vrbflg ) 
      fprintf( stdout, "discarding injections not in input data\n" );
  }

  for ( j = 0; j < numInFiles; ++j )
  {
    LIGOTimeGPS inPlay, outPlay;
    UINT8 outPlayNS, outStartNS, outEndNS, triggerTimeNS;
    INT4 trigStartTimeArg = 0;

    searchSummaryTable = XLALSearchSummaryTableFromLIGOLw( inFileNameList[j] );
    if ( ( ! searchSummaryTable ) || searchSummaryTable->next )
    {
      fprintf( stderr, 
          "error: zero or multiple search_summary tables in %s\n",
          inFileNameList[j] );
      exit( 1 );
    }

    if ( enableTrigStartTime )
    {
      /* override the value of out_start_time if there is a non-zero */
      /* --trig-start-time option in the process_params table        */
      /* this is necessary to get round a bug in early versions of   */
      /* the ringdown code                                           */

      int mioStatus;
      int pParParam;
      int pParValue;
      struct MetaioParseEnvironment parseEnv;
      const  MetaioParseEnv env = &parseEnv;

      /* open the procress_params table from the input file */
      mioStatus = MetaioOpenTable( env, inFileNameList[j], "process_params" );
      if ( mioStatus )
      {
        fprintf( stderr, "error opening process_params table from file %s\n", 
            inFileNameList[j] );
        exit( 1 );
      }

      /* figure out where the param and value columns are */
      if ( (pParParam = MetaioFindColumn( env, "param" )) < 0 )
      {
        fprintf( stderr, "unable to find column param in process_params\n" );
        MetaioClose(env);
        exit( 1 );
      }
      if ( (pParValue = MetaioFindColumn( env, "value" )) < 0 )
      {
        fprintf( stderr, "unable to find column value in process_params\n" );
        MetaioClose(env);
        exit( 1 );
      }

      /* get the trigger start time from the process params */
      while ( (mioStatus = MetaioGetRow(env)) == 1 )
      {
        if ( ! strcmp( env->ligo_lw.table.elt[pParParam].data.lstring.data, 
              "--trig-start-time" ) )
        {
          trigStartTimeArg = (INT4) 
            atoi( env->ligo_lw.table.elt[pParValue].data.lstring.data );
        }
      }

      MetaioClose( env );

      if ( trigStartTimeArg )
      {
        searchSummaryTable->out_start_time.gpsSeconds = trigStartTimeArg;
        searchSummaryTable->out_start_time.gpsNanoSeconds = 0;
        if ( vrbflg ) fprintf( stdout, "file %s has --trig-start-time %d\n",
            inFileNameList[j], trigStartTimeArg );
      }
    }

    /* compute the out time from the search summary table */
    outStartNS = XLALGPSToINT8NS ( &(searchSummaryTable->out_start_time) );
    outEndNS = XLALGPSToINT8NS ( &(searchSummaryTable->out_end_time) );
    triggerTimeNS = outEndNS - outStartNS;

    /* check for events and playground */
    if ( dataType != all_data )
    {
      LAL_CALL( LALPlaygroundInSearchSummary( &status, searchSummaryTable,
            &inPlay, &outPlay ), &status );
      outPlayNS = XLALGPSToINT8NS ( &outPlay );

      if ( dataType == playground_only )
      {
        if ( outPlayNS )
        {
          /* increment the total trigger time by the amount of playground */
          triggerInputTimeNS += outPlayNS;
        }
        else
        {
          /* skip this file as it does not contain any playground data */
          if ( vrbflg )
          {
            fprintf( stdout, "file %s not in playground, continuing\n", 
                inFileNameList[j] );
          }
          LALFree( searchSummaryTable );
          searchSummaryTable = NULL;
          continue;
        }
      }
      else if ( dataType == exclude_play )
      {
        /* increment the total trigger time by the out time minus */
        /* the time that is in the playground                     */
        triggerInputTimeNS += triggerTimeNS - outPlayNS;
      }
    }
    else
    {
      /* increment the total trigger time by the out time minus */
      triggerInputTimeNS += triggerTimeNS;
    }

    if ( injectFileName )
    {
      if ( vrbflg ) fprintf( stdout, "discarding injections not in file: " );

      /* throw away injections that are outside analyzed times */
      while ( thisSimEvent && thisSimEvent->geocent_start_time.gpsSeconds < 
          searchSummaryTable->out_end_time.gpsSeconds )
      {
        /* check if injection is before file start time */
        if ( thisSimEvent->geocent_start_time.gpsSeconds < 
            searchSummaryTable->out_start_time.gpsSeconds )
        {
          /* discard the current injection */
          if ( prevSimEvent ) prevSimEvent->next = thisSimEvent->next;
          tmpSimEvent = thisSimEvent;
          thisSimEvent = thisSimEvent->next;
          LALFree( tmpSimEvent );
          ++numSimDiscard;
          if ( vrbflg ) fprintf( stdout, "-" );
        }
        else
        {
          /* store the head of the linked list */
          if ( ! simEventHead ) simEventHead = thisSimEvent;

          /* keep this injection */
          prevSimEvent = thisSimEvent;
          thisSimEvent = thisSimEvent->next;
          ++numSimInData;
          if ( vrbflg ) fprintf( stdout, "+" );
        }
      }
      if ( vrbflg ) fprintf( stdout, "\n" );
    }


    /*
     *
     * if there are any events in the file, read them in
     *
     */


    if ( searchSummaryTable->nevents )
    {
      INT4 isPlay;

      if ( vrbflg ) fprintf( stdout, "file %s contains %d events, processing\n",
          inFileNameList[j], searchSummaryTable->nevents );

      if ( ! prevEvent )
      {
        eventHandle = &thisEvent;
      }
      else
      {
        eventHandle = &(prevEvent->next);
      }

      /* read the events from the file into a temporary list */
      XLAL_TRY( *eventHandle = XLALSnglRingdownTableFromLIGOLw( inFileNameList[j] ), errnum);
      if ( ! *eventHandle )
        switch ( errnum )
        {
          case XLAL_EDATA:
            XLALPrintError("Unable to read sngl_ringdown table from %s\n", inFileNameList[j] );
            /*LALFree(thisInputFile);*/
            XLALClearErrno();
            break;
          default:
            XLALSetErrno( errnum );
            XLAL_ERROR( func, XLAL_EFUNC );
        }
      
      /* only keep triggers from the data that we want to analyze */
      thisEvent = *eventHandle;
      while ( thisEvent )
      {
        numEvents++;

        isPlay = XLALINT8NanoSecIsPlayground( XLALGPSToINT8NS( &(thisEvent->start_time) ) );

        if ( (dataType == all_data || 
              (dataType == playground_only && isPlay) ||
              (dataType == exclude_play && ! isPlay))
            && ( snrStar < 0 || thisEvent->snr > snrStar) )
        {
          /* keep the trigger and increment the count of triggers */
          if ( ! eventHead ) eventHead = thisEvent;
          prevEvent = thisEvent;
          thisEvent = thisEvent->next;
          ++numEventsKept;
        }
        else
        {
          /* discard the trigger and move to the next one */
          if ( prevEvent ) prevEvent->next = thisEvent->next;
          tmpEvent = thisEvent;
          thisEvent = thisEvent->next;
          LAL_CALL ( LALFreeSnglRingdown ( &status, &tmpEvent ), &status);
        }
      }

      /* make sure that the linked list is properly terminated */
      if ( prevEvent && prevEvent->next ) prevEvent->next->next = NULL;
    }
    else
    {
      if ( vrbflg ) fprintf( stdout, "file %s contains no events, skipping\n",
          inFileNameList[j] );
    }

    LALFree( searchSummaryTable );
    searchSummaryTable = NULL;
  }

  /* discard the remaining injections which occured after the last file */
  if ( injectFileName )
  {
    if ( vrbflg ) fprintf( stdout, "kept %d injections, discarded %d\n",
        numSimInData, numSimDiscard );

    if ( prevSimEvent ) prevSimEvent->next = NULL;

    numSimDiscard = 0;
    while ( thisSimEvent )
    {
      tmpSimEvent = thisSimEvent;
      thisSimEvent = thisSimEvent->next;
      LALFree( tmpSimEvent );
      ++numSimDiscard;
      if ( vrbflg ) fprintf( stdout, "-" );
    }

    if ( vrbflg ) fprintf( stdout, "\ndiscarded %d injections at end of list\n",
        numSimDiscard );
  }


  /*
   *
   * sort the ringdown events by time
   *
   */


  if ( injectFileName || sortTriggers )
  {
    if ( vrbflg ) fprintf( stdout, "sorting ringdown trigger list..." );
    LAL_CALL( LALSortSnglRingdown( &status, &eventHead, 
          *LALCompareSnglRingdownByTime ), &status );
    if ( vrbflg ) fprintf( stdout, "done\n" );
  }


  /*
   *
   * keep only event from requested ifo
   *
   */

  if ( ifoName )
  {
    if ( vrbflg ) fprintf( stdout, 
        "keeping only triggers from %s, discarding others...", ifoName );
    LAL_CALL( LALIfoCutSingleRingdown( &status, &eventHead, ifoName ), &status );
    LALIfoCountSingleRingdown( &status, &numEventsInIFO, eventHead, 
        XLALIFONumber(ifoName) );

    if ( vrbflg ) fprintf( stdout, "done\n" );
  }

  /*
   *
   * check for events that are coincident with injections
   *
   */


  if ( injectFileName )
  {
    int coincidence = 0;
    UINT8 simTime, ringdownTime;

    if ( vrbflg ) fprintf( stdout, 
        "checking for events that are coincident with injections\n" );

    /* Note: we are assuming that both the ringdown and */
    /* injection events are time sorted                 */
    thisSimEvent = simEventHead;
    thisEvent    = eventHead;

    simEventHead = NULL;
    eventHead    = NULL;
    prevSimEvent = NULL;
    prevEvent    = NULL;

    numSimFound      = 0;
    numSimDiscard    = 0;
    numEventsDiscard = 0;
    numEventsCoinc   = 0;

    if ( ! thisEvent )
    {
      /* no triggers in the input data, so all injections are missed */
      if ( vrbflg ) fprintf( stdout, "no triggers in input data\n" );

      thisMissedSim = missedSimHead = thisSimEvent;

      while ( thisMissedSim )
      {
        /* count the number of injections just stuck in the missed list */
        if ( vrbflg ) fprintf( stdout, "M" );
        ++numSimMissed;
        ++numSimProcessed;
        thisMissedSim = thisMissedSim->next;
      }
    }
    else
    {
      /* begin loop over the sim_ringdown events */
      while ( thisSimEvent )
      {
        /* compute the end time in nanosec for the injection */
        /* at the relevant detector                          */
        if ( ! strcmp( "L1", thisEvent->ifo ) )
        {
          simTime = XLALGPSToINT8NS ( &(thisSimEvent->l_start_time) );
        }
        else if ( ! strcmp( "H1", thisEvent->ifo ) || 
            ! strcmp( "H2", thisEvent->ifo ) )
        {
          simTime = XLALGPSToINT8NS ( &(thisSimEvent->h_start_time) );
        }
        else
        {
          fprintf( stderr, "unknown detector found in event list: %s\n", 
              thisEvent->ifo );
          fprintf( stderr, "Detector must be one of (G1|H1|H2|L1|T1|V1)\n");
          exit( 1 );
        }

        /* find the first ringdown event after the current sim event */
        while ( thisEvent )
        {
          coincidence = 0;

          /* compute the time in nanosec for the ringdown */
          ringdownTime = XLALGPSToINT8NS ( &(thisEvent->start_time) );

          if ( ringdownTime < (simTime - inject_dt) )
          {
            /* discard this event and move on to the next one */
            if ( prevEvent ) prevEvent->next = thisEvent->next;
            tmpEvent = thisEvent;
            thisEvent = thisEvent->next;
            LAL_CALL ( LALFreeSnglRingdown ( &status, &tmpEvent ), &status);
            ++numEventsProcessed;
            ++numEventsDiscard;
            if ( vrbflg ) fprintf( stdout, "-" );
          }
          else
          {
            /* we have reached the negative coincincidence window */
            break;
          }
        }

        while ( thisEvent )
        {
          /* compute the time in nanosec for the ringdown */
          ringdownTime = XLALGPSToINT8NS ( &(thisEvent->start_time) );

          if ( ringdownTime < (simTime + inject_dt) )
          {
            /* this event is within the coincidence window  */
            /* store this event and move on to the next one */
            if ( ! eventHead ) eventHead = thisEvent;
            prevEvent = thisEvent;
            thisEvent = thisEvent->next;
            coincidence = 1;
            ++numEventsProcessed;
            ++numEventsCoinc;
            if ( vrbflg ) fprintf( stdout, "+" );
          }
          else
          {
            /* we have reached the end of the positive coincincidence window */
            break;
          }
        }

        if ( coincidence )
        {
          /* keep this event in the list and move to the next sim event */
          if ( ! simEventHead ) simEventHead = thisSimEvent;
          prevSimEvent = thisSimEvent;
          ++numSimFound;
          ++numSimProcessed;
          thisSimEvent = thisSimEvent->next;
          if ( vrbflg ) fprintf( stdout, "F" );
        }
        else
        {
          /* save this sim event in the list of missed events... */
          if ( ! missedSimHead )
          {
            missedSimHead = thisMissedSim = thisSimEvent;
          }
          else
          {
            thisMissedSim = thisMissedSim->next = thisSimEvent;
          }

          /* ...and remove it from the list of found events */
          if ( prevSimEvent ) prevSimEvent->next = thisSimEvent->next;
          ++numSimMissed;
          if ( vrbflg ) fprintf( stdout, "M" );

          /* move to the next sim in the list */
          ++numSimProcessed;
          thisSimEvent = thisSimEvent->next;

          /* make sure the missed sim list is terminated */
          thisMissedSim->next = NULL;
        }

        if ( ! thisEvent )
        {
          /* these are no more events to process so all the rest of the */
          /* injections must be put in the missed injections list       */
          if ( ! missedSimHead )
          {
            /* this and any subsequent events are in the missed sim list */
            if ( thisSimEvent ) thisMissedSim = missedSimHead = thisSimEvent;
          }
          else
          {
            if ( thisSimEvent )
            {
              /* append the rest of the list to the list of missed injections */
              thisMissedSim = thisMissedSim->next = thisSimEvent;
            }
            else
            {
              /* there are no injections after this one */
              thisMissedSim = thisMissedSim->next = NULL;
            }
          }

          /* terminate the list of found injections correctly */
          if ( prevSimEvent ) prevSimEvent->next = NULL;

          while ( thisMissedSim )
          {
            /* count the number of injections just stuck in the missed list */
            if ( vrbflg ) fprintf( stdout, "M" );
            ++numSimMissed;
            ++numSimProcessed;
            thisMissedSim = thisMissedSim->next;
          }
          thisSimEvent = NULL;
          break;
        }
      }

      if ( thisEvent )
      {
        /* discard any remaining ringdown triggers -- including thisEvent */
        /* as we have run out of injections */
        tmpEvent = thisEvent;
        if ( prevEvent ) prevEvent->next = NULL;
        while ( tmpEvent )
        {
          thisEvent = tmpEvent;
          tmpEvent = tmpEvent->next;
          LAL_CALL ( LALFreeSnglRingdown ( &status, &thisEvent ), &status);
          ++numEventsDiscard;
          ++numEventsProcessed;
          if ( vrbflg ) fprintf( stdout, "-" );
        }
      }
    }

    if ( vrbflg )
    {
      fprintf( stdout, "\nfound %d injections, missed %d injections "
          "(%d injections processed)\n",
          numSimFound, numSimMissed, numSimProcessed );

      fprintf( stdout, "found %d coincident events, %d events discarded "
          "(%d events processed)\n",
          numEventsCoinc, numEventsDiscard, numEventsProcessed );
    }

  } /* end if ( injectFileName ) */


  /*
   *
   * cluster the remaining events
   *
   */


  if ( eventHead && clusterchoice )
  {
    if ( vrbflg ) fprintf( stdout, "clustering remaining triggers... " );
    LAL_CALL( LALClusterSnglRingdownTable( &status, eventHead,
          cluster_dt, clusterchoice ), &status );
    if ( vrbflg ) fprintf( stdout, "done\n" );

    /* count the number of triggers surviving the clustering */
    thisEvent = eventHead;
    numClusteredEvents = 0;
    while ( thisEvent )
    {
      ++numClusteredEvents;
      thisEvent = thisEvent->next;
    }
  }


  /*
   *
   * write output data
   *
   */


  /* write the main output file containing found injections */
  if ( vrbflg ) fprintf( stdout, "writing output xml files... " );
  memset( &xmlStream, 0, sizeof(LIGOLwXMLStream) );
  LAL_CALL( LALOpenLIGOLwXMLFile( &status, &xmlStream, outputFileName ), &status );

  /* write out the process and process params tables */
  if ( vrbflg ) fprintf( stdout, "process... " );
  XLALGPSTimeNow(&(proctable.processTable->start_time));
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream, process_table ), 
      &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, proctable, 
        process_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlStream ), &status );
  free( proctable.processTable );

  /* write the process params table */
  if ( vrbflg ) fprintf( stdout, "process_params... " );
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream, 
        process_params_table ), &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, procparams, 
        process_params_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlStream ), &status );

  /* Write the found injections to the sim table */
  if ( simEventHead )
  {
    if ( vrbflg ) fprintf( stdout, "sim_ringdown... " );
    outputTable.simRingdownTable = simEventHead;
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream, 
          sim_ringdown_table ), &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, outputTable, 
          sim_ringdown_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable( &status, &xmlStream ), &status );
  }

  /* Write the results to the ringdown table */
  if ( eventHead )
  {
    if ( vrbflg ) fprintf( stdout, "sngl_ringdown... " );
    outputTable.snglRingdownTable = eventHead;
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream, 
          sngl_ringdown_table ), &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, outputTable, 
          sngl_ringdown_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable( &status, &xmlStream ), &status);
  }

  /* close the output file */
  LAL_CALL( LALCloseLIGOLwXMLFile(&status, &xmlStream), &status);
  if ( vrbflg ) fprintf( stdout, "done\n" );

  /* write out the TAMA file if it is requested */
  if ( tamaFileName )
  {
    /* FIXME */
    REAL8 UNUSED trigtime;

    fp = fopen( tamaFileName, "w" );
    if ( ! fp )
    {
      perror( "TAMA file" );
      exit( 1 );
    }

    fprintf( fp, "IFO   trigger time       snr         chisq       "
        " total mass     eta       eff dist (kpc)\n" );

    for ( thisEvent = eventHead; thisEvent; thisEvent = thisEvent->next )
    {
      trigtime = XLALGPSGetREAL8(&(thisEvent->start_time));
    }

    fclose( fp );
  }

  if ( missedFileName )
  {
    /* open the missed injections file and write the missed injections to it */
    if ( vrbflg ) fprintf( stdout, "writing missed injections... " );
    memset( &xmlStream, 0, sizeof(LIGOLwXMLStream) );
    LAL_CALL( LALOpenLIGOLwXMLFile( &status, &xmlStream, missedFileName ), 
        &status );

    if ( missedSimHead )
    {
      outputTable.simRingdownTable = missedSimHead;
      LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream, sim_ringdown_table ),
          &status );
      LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, outputTable, 
            sim_ringdown_table ), &status );
      LAL_CALL( LALEndLIGOLwXMLTable( &status, &xmlStream ), &status );
    }

    LAL_CALL( LALCloseLIGOLwXMLFile( &status, &xmlStream ), &status );
    if ( vrbflg ) fprintf( stdout, "done\n" );
  }

  if ( summFileName )
  {
    LIGOTimeGPS triggerTime;

    /* write out a summary file */
    fp = fopen( summFileName, "w" );

    switch ( dataType )
    {
      case playground_only:
        fprintf( fp, "using data from playground times only\n" );
        break;
      case exclude_play:
        fprintf( fp, "excluding all triggers in playground times\n" );
        break;
      case all_data:
        fprintf( fp, "using all input data\n" );
        break;
      default:
        fprintf( stderr, "data set not defined\n" );
        exit( 1 );
    }

    fprintf( fp, "read triggers from %d files\n", numInFiles );
    fprintf( fp, "number of triggers in input files: %d \n", numEvents );
    if ( snrStar >= 0 )
    {
      fprintf( fp, "number of triggers in input data with snr above %f: %d \n",
          snrStar, numEventsKept );
    }
    else
    {
      fprintf( fp, "number of triggers in input data %d \n", numEventsKept );
    }

    if ( ifoName )
    {
      fprintf( fp, "number of triggers from %s ifo %d \n", ifoName, 
          numEventsInIFO );
    }

    XLALINT8NSToGPS( &triggerTime, triggerInputTimeNS );
    fprintf( fp, "amount of time analysed for triggers %d sec %d ns\n", 
        triggerTime.gpsSeconds, triggerTime.gpsNanoSeconds );

    if ( injectFileName )
    {
      fprintf( fp, "read %d injections from file %s\n", 
          numSimEvents, injectFileName );

      fprintf( fp, "number of injections in input data: %d\n", numSimInData );
      fprintf( fp, "number of injections found in input data: %d\n", 
          numSimFound );
      fprintf( fp, 
          "number of triggers found within %" LAL_INT8_FORMAT "msec of injection: %d\n",
          (inject_dt / LAL_INT8_C(1000000) ), numEventsCoinc );

      fprintf( fp, "efficiency: %f \n", 
          (REAL4) numSimFound / (REAL4) numSimInData );
    }

    if ( clusterchoice )
    {
      fprintf( fp, "number of event clusters with %" LAL_INT8_FORMAT " msec window: %d\n",
          cluster_dt/ LAL_INT8_C(1000000), numClusteredEvents ); 
    }

    fclose( fp ); 
  }


  /*
   *
   * free memory and exit
   *
   */


  /* free the ringdown events we saved */
  while ( eventHead )
  {
    thisEvent = eventHead;
    eventHead = eventHead->next;
    LAL_CALL ( LALFreeSnglRingdown ( &status, &thisEvent ), &status);
  }

  /* free the process params */
  while( procparams.processParamsTable )
  {
    this_proc_param = procparams.processParamsTable;
    procparams.processParamsTable = this_proc_param->next;
    free( this_proc_param );
  }

  /* free the found injections */
  while ( simEventHead )
  {
    thisSimEvent = simEventHead;
    simEventHead = simEventHead->next;
    LALFree( thisSimEvent );
  }

  /* free the temporary memory containing the missed injections */
  while ( missedSimHead )
  {
    tmpSimEvent = missedSimHead;
    missedSimHead = missedSimHead->next;
    LALFree( tmpSimEvent );
  }

  /* free the input file name data */
  if ( inputGlob )
  {
    LALFree( inFileNameList ); 
    globfree( &globbedFiles );
  }
  else
  {
    for ( j = 0; j < numInFiles; ++j )
    {
      LALFree( inFileNameList[j] );
    }
    LALFree( inFileNameList );
  }

  if ( vrbflg ) fprintf( stdout, "checking memory leaks and exiting\n" );
  LALCheckMemoryLeaks();
  exit( 0 );
}


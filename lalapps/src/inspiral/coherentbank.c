/*
*  Copyright (C) 2007 Alexander Dietz, Stephen Fairhurst, Sean Seader
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
 * File Name: coherent_bank.c
 *
 * Author: Fairhust, S. and Seader, S.E.
 *
 *-----------------------------------------------------------------------
 */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/types.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <lal/LIGOMetadataInspiralUtils.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOMetadataUtils.h>
#include <lalapps.h>
#include <processtable.h>
#include <LALAppsVCSInfo.h>

RCSID("$Id$");

#define CVS_ID_STRING "$Id$"
#define CVS_NAME_STRING "$Name$"
#define CVS_REVISION "$Revision$"
#define CVS_SOURCE "$Source$"
#define CVS_DATE "$Date$"
#define PROGRAM_NAME "lalapps_coherentbank"

#define ADD_PROCESS_PARAM( pptype, format, ppvalue ) \
  this_proc_param = this_proc_param->next = (ProcessParamsTable *) \
calloc( 1, sizeof(ProcessParamsTable) ); \
snprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", \
    PROGRAM_NAME ); \
snprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--%s", \
    long_options[option_index].name ); \
snprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "%s", pptype ); \
snprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, format, ppvalue );

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

extern int vrbflg;
int allIFO = -1;

/*
 *
 * USAGE
 *
 */

static void print_usage(char *program)
{
  fprintf(stderr,
      "Usage: %s [options] [LIGOLW XML input files]\n"\
      "The following options are recognized.  Options not surrounded in []\n"\
      "are required.\n", program );
  fprintf(stderr,
      " [--help]                      display this message\n"\
      " [--verbose]                   print progress information\n"\
      " [--version]                   print version information and exit\n"\
      " [--debug-level]   level       set the LAL debug level to LEVEL\n"\
      " [--user-tag]      usertag     set the process_params usertag\n"\
      " [--comment]       string      set the process table comment\n"\
      " [--write-compress]            write a compressed xml file\n"\
      "  --ifos           ifos        list of ifos for which we have data\n"\
      "  --gps-start-time start_time  start time of the job\n"\
      "  --gps-end-time   end_time    end time of the job\n"\
      "  --enable-all-ifo             generate bank with templates for all ifos\n"\
      "  --disable-all-ifo            only generate bank for triggers in coinc\n"\
      " [--coinc-stat]        stat     use coinc statistic for cluster/cut\n"\
      " [--sngl-stat]   clusterchoice use single-ifo statistic for cluster/cut\n"\
      "                     [ none (default) | snr_and_chisq | snrsq_over_chisq | snr ]\n"\
      "                     [ snrsq | effective_snrsq ]\n"\
      " [--run-type]      runType     create trigger bank on coincs or\n"\
      "                               inspiral-coherent triggers\n"\
      "                               [ cohbank (default) | cohinspbank ]\n"\
      " [--stat-threshold]    thresh   discard all triggers with stat less than thresh\n"\
      " [--cluster-time]      time     cluster triggers with time ms window\n"\
      "\n");
}

int main( int argc, char *argv[] )
{
  static LALStatus      status;

  INT4 i;
  INT4 numTriggers = 0;
  INT4 numCoincs = 0;
  INT4 numTmplts = 0;
  INT4 UNUSED numCoincSegCutTrigs = 0;
  CoincInspiralStatistic coincstat = no_stat;
  SnglInspiralClusterChoice clusterchoice = none;
  CohbankRunType runType = cohbank;
  REAL4 statThreshold = 0;
  INT8 cluster_dt = 0;
  int  numSlides = 0;
  int  numClusteredEvents = 0;
  CoincInspiralStatParams    bittenLParams;
  INT4        startTime = -1;
  LIGOTimeGPS startTimeGPS = {0,0};
  INT4        endTime = -1;
  LIGOTimeGPS endTimeGPS = {0,0};
  INT8        UNUSED startTimeNS = 0;
  INT8        UNUSED endTimeNS = 0;
  CHAR  ifos[LIGOMETA_IFOS_MAX];

  CHAR  comment[LIGOMETA_COMMENT_MAX];
  CHAR *userTag = NULL;

  CHAR  fileName[FILENAME_MAX];

  SnglInspiralTable    *inspiralEventList=NULL;
  SnglInspiralTable    *currentTrigger = NULL;
  SnglInspiralTable    *newEventList = NULL;

  CoincInspiralTable   *coincHead = NULL;
  CoincInspiralTable   *thisCoinc = NULL;

  SearchSummvarsTable  *inputFiles = NULL;
  SearchSummvarsTable  *thisInputFile = NULL;

  SearchSummaryTable   *searchSummList = NULL;
  SearchSummaryTable   *thisSearchSumm = NULL;

  MetadataTable         proctable;
  MetadataTable         processParamsTable;
  MetadataTable         searchsumm;
  MetadataTable         inspiralTable;
  ProcessParamsTable   *this_proc_param = NULL;
  LIGOLwXMLStream       xmlStream;
  INT4                  outCompress = 0;


  /* getopt arguments */
  struct option long_options[] =
  {
    {"verbose",                no_argument,     &vrbflg,                  1 },
    {"enable-all-ifo",         no_argument,     &allIFO,                  1 },
    {"disable-all-ifo",        no_argument,     &allIFO,                  0 },
    {"write-compress",         no_argument,     &outCompress,             1 },
    {"comment",                required_argument,     0,                 'x'},
    {"user-tag",               required_argument,     0,                 'Z'},
    {"help",                   no_argument,           0,                 'h'},
    {"debug-level",            required_argument,     0,                 'z'},
    {"version",                no_argument,           0,                 'V'},
    {"gps-start-time",         required_argument,     0,                 's'},
    {"gps-end-time",           required_argument,     0,                 't'},
    {"ifos",                   required_argument,     0,                 'i'},
    {"coinc-stat",             required_argument,     0,                 'C'},
    {"sngl-stat",              required_argument,     0,                 'S'},
    {"run-type",               required_argument,     0,                 'r'},
    {"stat-threshold",         required_argument,     0,                 'E'},
    {"cluster-time",           required_argument,     0,                 'T'},
    {"num-slides",             required_argument,     0,                 'N'},
    {"eff-snr-denom-fac",      required_argument,     0,                 'A'},
    {0, 0, 0, 0}
  };
  int c;

  /*
   *
   * initialize things
   *
   */

  lal_errhandler = LAL_ERR_EXIT;
  set_debug_level( "1" );
  /*  setvbuf( stdout, NULL, _IONBF, 0 );*/

  /* create the process and process params tables */
  proctable.processTable = (ProcessTable *) calloc( 1, sizeof(ProcessTable) );
  XLALGPSTimeNow(&(proctable.processTable->start_time));
  XLALPopulateProcessTable(proctable.processTable, PROGRAM_NAME, LALAPPS_VCS_IDENT_ID,
      LALAPPS_VCS_IDENT_STATUS, LALAPPS_VCS_IDENT_DATE, 0);
  this_proc_param = processParamsTable.processParamsTable =
    (ProcessParamsTable *) calloc( 1, sizeof(ProcessParamsTable) );

  /* initialize variables */
  memset( comment, 0, LIGOMETA_COMMENT_MAX * sizeof(CHAR) );
  memset( ifos, 0, LIGOMETA_IFOS_MAX * sizeof(CHAR) );

  /* create the search summary and zero out the summvars table */
  searchsumm.searchSummaryTable = (SearchSummaryTable *)
    calloc( 1, sizeof(SearchSummaryTable) );

  memset( &bittenLParams, 0, sizeof(CoincInspiralStatParams   ) );
  /* default value from traditional effective snr formula */
  bittenLParams.eff_snr_denom_fac = 250.0;

  /* parse the arguments */
  while ( 1 )
  {
    /* getopt_long stores long option here */
    int option_index = 0;
    size_t optarg_len;
    long int gpstime;

    c = getopt_long_only( argc, argv,
        "hi:r:s:t:x:z:C:E:T:N:A:VZ:", long_options,
        &option_index );

    /* detect the end of the options */
    if ( c == -1 )
    {
      break;
    }

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
          fprintf( stderr, "Error parsing option %s with argument %s\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'i':
        /* set ifos */
        strncpy( ifos, optarg, LIGOMETA_IFOS_MAX );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 's':
        /* start time coincidence window */
        gpstime = atol( optarg );
        if ( gpstime < 441417609 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is prior to "
              "Jan 01, 1994  00:00:00 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpstime );
          exit( 1 );
        }
        if ( gpstime > 999999999 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is after "
              "Sep 14, 2011  01:46:26 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpstime );
          exit( 1 );
        }
        startTime = (INT4) gpstime;
        startTimeGPS.gpsSeconds = startTime;
        ADD_PROCESS_PARAM( "int", "%" LAL_INT4_FORMAT, startTime );
        break;

      case 't':
        /* end time coincidence window */
        gpstime = atol( optarg );
        if ( gpstime < 441417609 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is prior to "
              "Jan 01, 1994  00:00:00 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpstime );
          exit( 1 );
        }
        if ( gpstime > 999999999 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "GPS start time is after "
              "Sep 14, 2011  01:46:26 UTC:\n"
              "(%ld specified)\n",
              long_options[option_index].name, gpstime );
          exit( 1 );
        }
        endTime = (INT4) gpstime;
        endTimeGPS.gpsSeconds = endTime;
        ADD_PROCESS_PARAM( "int", "%" LAL_INT4_FORMAT, endTime );
        break;

      case 'x':
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

      case 'h':
        /* help message */
        print_usage(argv[0]);
        exit( 1 );
        break;

      case 'z':
        set_debug_level( optarg );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'C':
        /* choose the coinc statistic */
        {
          if ( ! strcmp( "snrsq", optarg ) )
          {
            coincstat = snrsq;
          }
          else if ( ! strcmp( "effective_snrsq", optarg) )
          {
            coincstat = effective_snrsq;
          }
          else
          {
            fprintf( stderr, "invalid argument to  --%s:\n"
                "unknown coinc statistic:\n "
                "%s (must be one of:\n"
                "snrsq, effective_snrsq\n",
                long_options[option_index].name, optarg);
            exit( 1 );
          }
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        break;

      case 'S':
        /* choose the single-ifo cluster statistic */
        {
          if ( ! strcmp( "snr", optarg ) )
          {
            clusterchoice = snr;
          }
          else if ( ! strcmp( "snr_and_chisq", optarg) )
          {
            clusterchoice = snr_and_chisq;
          }
          else if ( ! strcmp( "snrsq_over_chisq", optarg) )
          {
            clusterchoice = snrsq_over_chisq;
          }
          else
          {
            fprintf( stderr, "invalid argument to  --%s:\n"
                "unknown coinc statistic:\n "
                "%s (must be one of:\n"
		"snr, snr_and_chisq, or snrsq_over_chisq\n",
                long_options[option_index].name, optarg);
            exit( 1 );
          }
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        break;

      case 'r':
        /* choose the run-type for constructing a bank on either
           coinc or inspiral-coherent triggers */
        {
          if ( ! strcmp( "cohbank", optarg ) )
          {
            runType = cohbank;
          }
          else if ( ! strcmp( "cohinspbank", optarg) )
          {
            runType = cohinspbank;
          }
          else
          {
            fprintf( stderr, "invalid argument to  --%s:\n"
                "unknown run-type:\n "
                "%s (must be one of:\n"
                "cohbank, cohinspbank\n",
                long_options[option_index].name, optarg);
            exit( 1 );
          }
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        break;

      case 'E':
        /* store the stat threshold for a cut */
        statThreshold = atof( optarg );
        if ( statThreshold <= 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "statThreshold must be positive: (%f specified)\n",
              long_options[option_index].name, statThreshold );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%f", statThreshold );
        break;


      case 'T':
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
        ADD_PROCESS_PARAM( "int", "%" LAL_INT8_FORMAT, cluster_dt );
        /* convert cluster time from ms to ns */
        cluster_dt *= 1000000LL;
        break;

      case 'N':
        /* store the number of slides */
        numSlides = atoi( optarg );
        if ( numSlides < 0 )
        {
          fprintf( stdout, "invalid argument to --%s:\n"
              "numSlides >= 0: "
              "(%d specified)\n",
              long_options[option_index].name, numSlides );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", numSlides );
        break;

      case 'A':
        bittenLParams.eff_snr_denom_fac = atof(optarg);
        ADD_PROCESS_PARAM( "float", "%s", optarg);
        break;

      case 'Z':
        /* create storage for the usertag */
        optarg_len = strlen(optarg) + 1;
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

      case 'V':
        /* print version information and exit */
        fprintf( stdout, "Coherent Bank Generator\n"
            "Steve Fairhurst and Shawn Seader\n");
        XLALOutputVersionString(stderr, 0);
        exit( 0 );
        break;

      case '?':
        print_usage(argv[0]);
        exit( 1 );
        break;

      default:
        fprintf( stderr, "Error: Unknown error while parsing options\n" );
        print_usage(argv[0]);
        exit( 1 );
    }
  }
  if ( ! *comment )
  {
    snprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX, " " );
    snprintf( searchsumm.searchSummaryTable->comment, LIGOMETA_COMMENT_MAX,
        " " );
  }
  else
  {
    snprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX,
        "%s", comment );
    snprintf( searchsumm.searchSummaryTable->comment, LIGOMETA_COMMENT_MAX,
        "%s", comment );
  }

  /* enable/disable-all-ifo is stored in the first process param row */
  if ( allIFO == 1 )
  {
    snprintf( processParamsTable.processParamsTable->program,
        LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
    snprintf( processParamsTable.processParamsTable->param,
        LIGOMETA_PARAM_MAX, "--enable-all-ifo" );
    snprintf( processParamsTable.processParamsTable->type,
        LIGOMETA_TYPE_MAX, "string" );
    snprintf( processParamsTable.processParamsTable->value,
        LIGOMETA_TYPE_MAX, " " );
  }
  else if ( allIFO == 0 )
  {
    snprintf( processParamsTable.processParamsTable->program,
        LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
    snprintf( processParamsTable.processParamsTable->param,
        LIGOMETA_PARAM_MAX, "--disable-all-ifo" );
    snprintf( processParamsTable.processParamsTable->type,
        LIGOMETA_TYPE_MAX, "string" );
    snprintf( processParamsTable.processParamsTable->value,
        LIGOMETA_TYPE_MAX, " " );
  }
  else
  {
    fprintf( stderr, "--enable-all-ifo or --disable-all-ifo "
        "argument must be specified\n" );
    exit( 1 );
  }

  /*
   *
   * check the values of the arguments
   *
   */

  startTimeNS = XLALGPSToINT8NS( &startTimeGPS );
  endTimeNS = XLALGPSToINT8NS( &endTimeGPS );

  if ( startTime < 0 )
  {
    fprintf( stderr, "Error: --gps-start-time must be specified\n" );
    exit( 1 );
  }

  if ( endTime < 0 )
  {
    fprintf( stderr, "Error: --gps-end-time must be specified\n" );
    exit( 1 );
  }

  if ( !strlen(ifos) )
  {
    fprintf(stderr,"You must specify a list of ifos with --ifos. Exiting.\n");
    exit(1);
  }

  /* check that if clustering is being done that we have all the options */
  if ( cluster_dt && (coincstat == no_stat) )
  {
    fprintf( stderr,
        "--coinc-stat must be specified if --cluster-time is given\n" );
    exit( 1 );
  }

  /*
   *
   * read in the input data from the rest of the arguments
   *
   */

  if ( optind < argc )
  {
    if ( !(runType == cohinspbank ) ) {
      for( i = optind; i < argc; ++i )
      {
	INT4 numFileTriggers = 0;
	numFileTriggers = XLALReadInspiralTriggerFile( &inspiralEventList,
	     &currentTrigger, &searchSummList, &inputFiles, argv[i] );
	if (numFileTriggers < 0)
	{
	  fprintf(stderr, "Error reading triggers from file %s",
            argv[i]);
	  exit( 1 );
	}

	numTriggers += numFileTriggers;
      }
    }
    else {
      InterferometerNumber  ifoNumber = LAL_UNKNOWN_IFO;
      INT4                  numCoincTrigs = 0;
      for ( ifoNumber = 0; ifoNumber< LAL_NUM_IFO; ifoNumber++) {
	//SnglInspiralTable    *inspiralEventTmpList=NULL;
	INT4                  numIfoTriggers = 0;

	for( i = optind; i < argc; ++i )
	{
	  INT4 numFileTriggers = 0;
	  SearchSummaryTable *inputSummary = NULL;

	  /* read in the search summary and store */
	  XLALPrintInfo(
		"XLALReadInspiralTriggerFile(): Reading search_summary table\n");

	  inputSummary = XLALSearchSummaryTableFromLIGOLw(argv[i]);

	  if ( ! inputSummary )
	  {
	    fprintf(stderr,"No valid search_summary table in %s, exiting\n",
		    argv[i] );
	    exit( 1 );
	  }
	  else
	  {
	    fprintf(stdout,"IFOs in input summary table is %s\n",inputSummary->ifos);
	    if ( (ifoNumber == XLALIFONumber(inputSummary->ifos) ) ) {
	      fprintf(stdout,"Reading triggers from IFO %s\n",inputSummary->ifos);
	      numFileTriggers = XLALReadInspiralTriggerFile( &inspiralEventList,
			&currentTrigger, &searchSummList, &inputFiles, argv[i] );
	      if (numFileTriggers < 0)
	      {
		fprintf(stderr, "Error reading triggers from file %s",
			argv[i]);
		exit( 1 );
	      }

	      numIfoTriggers += numFileTriggers;

	    }
	  }
	}/* Loop to read all trigger files from a single ifo */

      	if( vrbflg )
	  {
	    fprintf( stdout,
		     "Number of triggers found in ifo number %d is %d\n",
		     ifoNumber , numIfoTriggers);
	  }

        numCoincTrigs += numIfoTriggers;
        if( vrbflg )
        {
          fprintf( stdout,
                   "Number of unclustered triggers found in all ifos in this coinc-segment is %d\n",
                  numCoincTrigs);
        }
      }/* Closes for loop over ifoNumber */

      /* keep only triggers within the requested interval */
      if ( vrbflg ) fprintf( stdout,
			     "Discarding triggers outside requested interval\n" );
      LAL_CALL( LALTimeCutSingleInspiral( &status, &inspiralEventList,
				  &startTimeGPS, &endTimeGPS), &status );

      if ( vrbflg ) fprintf( stdout,
                             "Removing triggers with event-ids from outside this coinc-segment\n" );
      if ( vrbflg ) fprintf( stdout, "GPS start time of this coinc-segment is %d\n",
                   startTime);

      numCoincSegCutTrigs = XLALCoincSegCutSnglInspiral( startTime, &inspiralEventList);

      if ( vrbflg ) fprintf( stdout,
                             "Sorting triggers within requested interval\n" );
      /* sort single inspiral trigger list according to event_id */
      inspiralEventList = XLALSortSnglInspiral( inspiralEventList,
                                        LALCompareSnglInspiralByID);

      if ( vrbflg ) fprintf( stdout,
                             "Clustering triggers in event-id\n" );
      numTriggers = XLALClusterInEventID(&inspiralEventList,clusterchoice);
      if( vrbflg )
	{
	  fprintf( stdout,
		   "Number of CLUSTERED triggers found in coinc-segment is %d\n",
		   numTriggers);
	}
	//numTriggers += numIfoTriggers;

    } /*Closes if runType is not cohinspbank */
  } /* Closes if optind < argc */
  else
  {
    fprintf( stderr, "Error: No trigger files specified.\n" );
    exit( 1 );
  }

  if ( numTriggers == 0 )
  {
    if( vrbflg )
    {
      fprintf( stdout,
         "No triggers found - the coherent bank will be empty.\n");
    }
  }
  else
  {
    if( vrbflg )
    {
      fprintf( stdout,
          "Read in a total of %d triggers.\n", numTriggers);
    }

    /* reconstruct the coincs */
    numCoincs = XLALRecreateCoincFromSngls( &coincHead, &inspiralEventList );
    if( numCoincs < 0 )
    {
      fprintf(stderr, "Unable to reconstruct coincs from single ifo triggers");
      exit( 1 );
    }
    else if ( vrbflg )
    {
      fprintf( stdout,
          "Recreated %d coincs from the %d triggers\n", numCoincs,
          numTriggers );
    }

  /*
   *
   * sort the inspiral events by time
   *
   */


    if ( coincHead && cluster_dt )
    {
      if ( vrbflg ) fprintf( stdout, "sorting coinc inspiral trigger list..." );
      coincHead = XLALSortCoincInspiral( coincHead,
                          *XLALCompareCoincInspiralByTime );
      if ( vrbflg ) fprintf( stdout, "done\n" );

      if ( vrbflg ) fprintf( stdout, "clustering remaining triggers... " );

      if ( !numSlides )
      {
        numClusteredEvents = XLALClusterCoincInspiralTable( &coincHead,
                      cluster_dt, coincstat , &bittenLParams);
      }
      else
      {
        int slide = 0;
        int numClusteredSlide = 0;
        CoincInspiralTable *slideCoinc = NULL;
        CoincInspiralTable *slideClust = NULL;

        if ( vrbflg ) fprintf( stdout, "splitting events by slide\n" );

        for( slide = -numSlides; slide < (numSlides + 1); slide++)
        {
          if ( vrbflg ) fprintf( stdout, "slide number %d; ", slide );
          /* extract the slide */
          slideCoinc = XLALCoincInspiralSlideCut( &coincHead, slide );
          /* run clustering */
          numClusteredSlide = XLALClusterCoincInspiralTable( &slideCoinc,
                       cluster_dt, coincstat, &bittenLParams);

          if ( vrbflg ) fprintf( stdout, "%d clustered events \n",
                                 numClusteredSlide );
          numClusteredEvents += numClusteredSlide;

          /* add clustered triggers */
          if( slideCoinc )
          {
            if( slideClust )
            {
              thisCoinc = thisCoinc->next = slideCoinc;
            }
            else
            {
              slideClust = thisCoinc = slideCoinc;
            }
            /* scroll to end of list */
            for( ; thisCoinc->next; thisCoinc = thisCoinc->next);
          }
        }

        /* free coincHead -- although we expect it to be empty */
        while ( coincHead )
        {
          thisCoinc = coincHead;
          coincHead = coincHead->next;
          XLALFreeCoincInspiral( &thisCoinc );
        }

        /* move events to coincHead */
        coincHead = slideClust;
        slideClust = NULL;
      }

      if ( vrbflg ) fprintf( stdout, "done\n" );
      if ( vrbflg ) fprintf( stdout, "%d clustered events \n",
                             numClusteredEvents );
    }

    /*
     *
     *  Create the coherent bank
     *
     */

    numTmplts = XLALGenerateCoherentBank( &newEventList, coincHead );

    if ( numTmplts < 0 )
    {
      fprintf(stderr, "Unable to generate coherent bank\n");
      exit( 1 );
    }
    else if ( vrbflg )
    {
      fprintf(stdout, "Generated a coherent bank with %d templates\n",
          numTmplts);
    }
  }
  /*
   *
   * write the output xml file
   *
   */

  /* search summary entries: */
  searchsumm.searchSummaryTable->in_start_time = startTimeGPS;
  searchsumm.searchSummaryTable->in_end_time = endTimeGPS;
  searchsumm.searchSummaryTable->out_start_time = startTimeGPS;
  searchsumm.searchSummaryTable->out_end_time = endTimeGPS;
  searchsumm.searchSummaryTable->nevents = numTmplts;

  startTimeNS = XLALGPSToINT8NS( &startTimeGPS );
  endTimeNS = XLALGPSToINT8NS( &endTimeGPS );

  if ( vrbflg ) fprintf( stdout, "writing output file... " );

  /* set the file name correctly */
  if ( userTag && !outCompress )
  {
    if ( runType == cohinspbank ) {
      snprintf( fileName, FILENAME_MAX, "%s-COHINSPBANK_%s-%d-%d.xml",
        ifos, userTag, startTime, endTime - startTime );
    }
    else {
      snprintf( fileName, FILENAME_MAX, "%s-COHBANK_%s-%d-%d.xml",
        ifos, userTag, startTime, endTime - startTime );
    }
  }
  else if ( userTag && outCompress )
  {
    if ( runType == cohinspbank ) {
      snprintf( fileName, FILENAME_MAX, "%s-COHINSPBANK_%s-%d-%d.xml.gz",
        ifos, userTag, startTime, endTime - startTime );
    }
    else {
      snprintf( fileName, FILENAME_MAX, "%s-COHBANK_%s-%d-%d.xml.gz",
        ifos, userTag, startTime, endTime - startTime );
    }
  }
  else if ( !userTag && outCompress )
  {
    if ( runType == cohinspbank ) {
      snprintf( fileName, FILENAME_MAX, "%s-COHINSPBANK-%d-%d.xml.gz",
        ifos, startTime, endTime - startTime );
    }
    else {
      snprintf( fileName, FILENAME_MAX, "%s-COHBANK-%d-%d.xml.gz",
        ifos, startTime, endTime - startTime );
    }
  }
  else
  {
    if ( runType == cohinspbank ) {
      snprintf( fileName, FILENAME_MAX, "%s-COHINSPBANK-%d-%d.xml",
        ifos, startTime, endTime - startTime );
    }
    else {
      snprintf( fileName, FILENAME_MAX, "%s-COHBANK-%d-%d.xml",
        ifos, startTime, endTime - startTime );
    }
  }
  memset( &xmlStream, 0, sizeof(LIGOLwXMLStream) );
  LAL_CALL( LALOpenLIGOLwXMLFile( &status , &xmlStream, fileName ),
      &status );

  /* write process table */
  XLALGPSTimeNow(&(proctable.processTable->end_time));
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream, process_table ),
      &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, proctable,
        process_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlStream ), &status );

  /* write process_params table */
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream,
        process_params_table ), &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, processParamsTable,
        process_params_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlStream ), &status );

  /* write search_summary table */
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &xmlStream,
        search_summary_table ), &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, searchsumm,
        search_summary_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &xmlStream ), &status );

  /* write the search_summvars tabls */
  /* XXX not necessary as bank file specified in arguements XXX
  LAL_CALL( LALBeginLIGOLwXMLTable( &status ,&xmlStream,
        search_summvars_table), &status );
  searchSummvarsTable.searchSummvarsTable = inputFiles;
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, searchSummvarsTable,
        search_summvars_table), &status );
  LAL_CALL( LALEndLIGOLwXMLTable( &status, &xmlStream), &status ); */

  /* write the sngl_inspiral table if we have one*/
  if ( newEventList )
  {
    LAL_CALL( LALBeginLIGOLwXMLTable( &status ,&xmlStream,
          sngl_inspiral_table), &status );
    inspiralTable.snglInspiralTable = newEventList;
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &xmlStream, inspiralTable,
          sngl_inspiral_table), &status );
    LAL_CALL( LALEndLIGOLwXMLTable( &status, &xmlStream), &status );
  }

  LAL_CALL( LALCloseLIGOLwXMLFile( &status, &xmlStream), &status );


  if ( vrbflg ) fprintf( stdout, "done\n" );


  /*
   *
   * clean up the memory that has been allocated
   *
   */


  if ( vrbflg ) fprintf( stdout, "freeing memory... " );

  free( proctable.processTable );
  free( searchsumm.searchSummaryTable );

  while ( processParamsTable.processParamsTable )
  {
    this_proc_param = processParamsTable.processParamsTable;
    processParamsTable.processParamsTable = this_proc_param->next;
    free( this_proc_param );
  }

  while ( inputFiles )
  {
    thisInputFile = inputFiles;
    inputFiles = thisInputFile->next;
    LALFree( thisInputFile );
  }

  while ( searchSummList )
  {
    thisSearchSumm = searchSummList;
    searchSummList = searchSummList->next;
    LALFree( thisSearchSumm );
  }

  while ( inspiralEventList )
  {
    currentTrigger = inspiralEventList;
    inspiralEventList = inspiralEventList->next;
    LAL_CALL( LALFreeSnglInspiral( &status, &currentTrigger ), &status );
  }

  /* while ( coincEventList )
  {
    thisCoinc = coincEventList;
    coincEventList = coincEventList->next;
    LALFree( thisCoinc );
  }
  */

  while ( newEventList )
  {
    currentTrigger = newEventList;
    newEventList = newEventList->next;
    LAL_CALL( LALFreeSnglInspiral( &status, &currentTrigger ), &status );
  }

  while ( coincHead )
  {
    thisCoinc = coincHead;
    coincHead = coincHead->next;
    LALFree( thisCoinc );
  }

  if ( userTag ) free( userTag );

  if ( vrbflg ) fprintf( stdout, "done\n" );

  LALCheckMemoryLeaks();

  exit( 0 );
}


int XLALCoincSegCutSnglInspiral(
    INT4                        startTime,
    SnglInspiralTable         **inspiralList
    )

{
  SnglInspiralTable     *thisEvent=NULL;
  SnglInspiralTable     *prevEvent=NULL;
  SnglInspiralTable     *nextEvent=NULL;

  int                    numSnglClust = 0;
  UINT8                  timeExtract = 1000000000;
  INT4 timeCheck=0;
  if ( !inspiralList )
  {
    XLALPrintInfo(
      "XLALClusterInEventID: Empty trigger list passed as input\n" );
    return( 0 );
  }

  if ( ! *inspiralList )
  {
    XLALPrintInfo(
      "XLALClusterInEventID: Empty trigger list passed as input\n" );
    return( 0 );
  }

  thisEvent = *inspiralList;
  //nextEvent = (*inspiralList)->next;
  *inspiralList = NULL;

  while ( thisEvent ) {
    fprintf(stdout, "This event's END TIME NS is %" LAL_INT4_FORMAT "\n",thisEvent->end_time.gpsNanoSeconds);
    fprintf(stdout, "This event's id is %" LAL_UINT8_FORMAT "\n",thisEvent->event_id->id);
    timeCheck = floor(thisEvent->event_id->id/timeExtract);
    fprintf(stdout, "This event's gps-start time is %" LAL_INT4_FORMAT "\n",
            timeCheck);

    /* find events in the same coinc-segment */
    if ( !( timeCheck == startTime) ) {
      /* displace this event in cluster */
      nextEvent = thisEvent->next;
      if( prevEvent )
      {
         prevEvent->next = nextEvent;
      }
      XLALFreeSnglInspiral( &thisEvent );
      thisEvent = nextEvent;
      nextEvent = thisEvent->next;
    }
    else {
      /* otherwise we keep this unique event trigger */
      if ( ! *inspiralList )
	{
	  *inspiralList = thisEvent;
	}
      prevEvent = thisEvent;
      thisEvent = thisEvent->next;
      if ( !thisEvent )
        nextEvent = thisEvent->next;
      ++numSnglClust;
    }
  }

  fprintf(stdout, "This last time-check is %" LAL_INT4_FORMAT "\n", timeCheck);
  /* store the last event */
  //if ( ! (*inspiralList) )
   // {
    //  *inspiralList = thisEvent;
   // }
  //++numSnglClust;

  return(numSnglClust);
}


int XLALClusterInEventID(
    SnglInspiralTable         **inspiralList,
    SnglInspiralClusterChoice   clusterchoice
    )

{
  SnglInspiralTable     *thisEvent=NULL;
  SnglInspiralTable     *prevEvent=NULL;
  SnglInspiralTable     *nextEvent=NULL;

  int                    numSnglClust = 0;

  if ( !inspiralList )
  {
    XLALPrintInfo(
      "XLALClusterInEventID: Empty trigger list passed as input\n" );
    return( 0 );
  }

  if ( ! *inspiralList )
  {
    XLALPrintInfo(
      "XLALClusterInEventID: Empty trigger list passed as input\n" );
    return( 0 );
  }

  thisEvent = *inspiralList;
  nextEvent = (*inspiralList)->next;
  *inspiralList = NULL;

  while ( nextEvent ) {
    /* find events with the same eventID in the same IFO */
    if ( (thisEvent->event_id->id == nextEvent->event_id->id )
	 && !strcmp(thisEvent->ifo,nextEvent->ifo ) ) {
      REAL4 thisStat = XLALSnglInspiralStat( thisEvent, clusterchoice );
      REAL4 nextStat = XLALSnglInspiralStat( nextEvent, clusterchoice );

      fprintf(stdout, "Next event's id is %" LAL_UINT8_FORMAT "\n",nextEvent->event_id->id);
      fprintf(stdout, "Next-statistic is %e\n",nextStat);
      fprintf(stdout, "Next IFO is %s\n",nextEvent->ifo);


      if ( nextStat > thisStat ) {
        /* displace previous event in cluster */
        if( prevEvent )
	  {
	    prevEvent->next = nextEvent;
	  }
        XLALFreeSnglInspiral( &thisEvent );
        thisEvent = nextEvent;
        nextEvent = thisEvent->next;
      }
      else
	{
	  /* otherwise just dump next event from cluster */
	  thisEvent->next = nextEvent->next;
	  XLALFreeSnglInspiral ( &nextEvent );
	  nextEvent = thisEvent->next;
	}
    }
    else
      {
	/* otherwise we keep this unique event trigger */
	if ( ! *inspiralList )
	  {
	    *inspiralList = thisEvent;
	  }
	prevEvent = thisEvent;
	thisEvent = thisEvent->next;
	nextEvent = thisEvent->next;
	++numSnglClust;
      }
  }

  /* store the last event */
  if ( ! (*inspiralList) )
    {
      *inspiralList = thisEvent;
  }
  ++numSnglClust;

  return(numSnglClust);
}

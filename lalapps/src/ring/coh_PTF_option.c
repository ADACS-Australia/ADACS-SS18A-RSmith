/*
*  Copyright (C) 2007 Duncan Brown, Jolien Creighton, Lisa M. Goggin, Matt Pitkin
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#include "coh_PTF.h"

#include "lalapps.h"
#include "errutil.h"
#include "gpstime.h"
#include "injsgnl.h"
#include "processtable.h"

RCSID( "$Id$" );

extern int vrbflg;
static int coh_PTF_usage( const char *program );
static int coh_PTF_default_params( struct coh_PTF_params *params );


/* parse command line arguments using getopt_long to get ring params */
int coh_PTF_parse_options(struct coh_PTF_params *params,int argc,char **argv )
{
  static struct coh_PTF_params localparams;
  memset( &localparams.haveTrig, 0, LAL_NUM_IFO * sizeof(int) );
  struct option long_options[] =
  {
    { "verbose", no_argument, &vrbflg, 1 },
    { "strain-data", no_argument, &localparams.strainData, 1 },
    { "simulated-data", no_argument, &localparams.simData, 1 },
    { "zero-data", no_argument, &localparams.zeroData, 1 },
    { "white-spectrum", no_argument, &localparams.whiteSpectrum, 1 },
    { "write-raw-data",     no_argument, &localparams.writeRawData, 1 },
    { "write-data",         no_argument, &localparams.writeProcessedData, 1 },
    { "write-inv-spectrum", no_argument, &localparams.writeInvSpectrum, 1 },
    { "write-segment",      no_argument, &localparams.writeSegment, 1 },
    { "write-filter-output",no_argument, &localparams.writeFilterOutput, 1 },
    { "analyze-inj-segs-only",no_argument, &localparams.analyzeInjSegsOnly, 1},
    { "do-null-stream"     ,no_argument, &localparams.doNullStream,1},
    { "do-trace-snr"       ,no_argument, &localparams.doTraceSNR,1},
    { "do-bank-veto"       ,no_argument, &localparams.doBankVeto,1},
    { "do-auto-veto"       ,no_argument, &localparams.doAutoVeto,1},
/*    {"g1-data",         no_argument,   &(haveTrig[LAL_IFO_G1]),   1 },*/
    {"h1-data",      no_argument,   &(localparams.haveTrig[LAL_IFO_H1]),   1 },
    {"h2-data",         no_argument,&(localparams.haveTrig[LAL_IFO_H2]),   1 },
    {"l1-data",      no_argument,   &(localparams.haveTrig[LAL_IFO_L1]),   1 },
/*    {"t1-data",         no_argument,   &(haveTrig[LAL_IFO_T1]),   1 },*/
    {"v1-data",      no_argument,   &(localparams.haveTrig[LAL_IFO_V1]),   1 },
    { "help",                    no_argument,       0, 'h' },
    { "version",                 no_argument,       0, 'V' },
    { "gps-start-time",          required_argument, 0, 'a' },
    { "gps-start-time-ns",       required_argument, 0, 'A' },
    { "gps-end-time",            required_argument, 0, 'b' },
    { "gps-end-time-ns",         required_argument, 0, 'B' },
    { "h1-channel-name",            required_argument, 0, 'c' },
    { "h1-frame-cache",             required_argument, 0, 'D' },
    { "h2-channel-name",            required_argument, 0, 'x' },
    { "h2-frame-cache",             required_argument, 0, 'X' },
    { "l1-channel-name",            required_argument, 0, 'y' },
    { "l1-frame-cache",             required_argument, 0, 'Y' },
    { "v1-channel-name",            required_argument, 0, 'z' },
    { "v1-frame-cache",             required_argument, 0, 'Z' },
    { "debug-level",             required_argument, 0, 'd' },
    { "cutoff-frequency",        required_argument, 0, 'e' },
    { "highpass-frequency",      required_argument, 0, 'E' },
    { "injection-file",          required_argument, 0, 'i' },
    { "snr-threshold",           required_argument, 0, 'j' },
    { "trig-time-window",        required_argument, 0, 'J' },
    { "user-tag",                required_argument, 0, 'k' },
    { "ifo-tag",                 required_argument, 0, 'K' },
    { "non-spin-snr2-threshold", required_argument, 0, 'l' },
    { "spin-snr2-threshold",     required_argument, 0, 'L' },
    { "spin-bank",               required_argument, 0, 'm' },
    { "non-spin-bank",           required_argument, 0, 'M' },
    { "only-segment-numbers",    required_argument, 0, 'n' },
    { "only-template-numbers",   required_argument, 0, 'N' },
    { "output-file",             required_argument, 0, 'o' },
    { "bank-file",               required_argument, 0, 'O' },
    { "num-auto-chisq-points",   required_argument, 0, 'p' },
    { "auto-veto-time-step",     required_argument, 0, 'P' },
    { "random-seed",             required_argument, 0, 'r' },
    { "dynamic-range-factor",    required_argument, 0, 'R' },
    { "sample-rate",             required_argument, 0, 's' },
    { "segment-duration",        required_argument, 0, 'S' },
    { "bank-veto-templates",     required_argument, 0, 't' },
    { "inverse-spec-length",     required_argument, 0, 'T' },
    { "trig-start-time",         required_argument, 0, 'u' },
    { "trig-end-time",           required_argument, 0, 'U' },
    { "block-duration",          required_argument, 0, 'w' },
    { "pad-data",                required_argument, 0, 'W' },
    { "right-ascension",         required_argument, 0, 'f' },
    { "declination",             required_argument, 0, 'F' },
    { 0, 0, 0, 0 }
  };
  char args[] = "a:A:b:B:c:d:D:e:E:f:F:h:i:j:J:k:K:l:L:m:M:n:N:o:O:r:R:s:S:t:T:u:U:V:w:W:x:X:y:Y:z:Z";
  char *program = argv[0];

  /* set default values for parameters before parsing arguments */
  coh_PTF_default_params( &localparams );

  while ( 1 )
  {
    int option_index = 0;
    int c;

    c = getopt_long_only( argc, argv, args, long_options, &option_index );
    if ( c == -1 ) /* end of options */
      break;

    switch ( c )
    {
      case 0: /* if option set a flag, nothing else to do */
        if ( long_options[option_index].flag )
          break;
        else
          error( "error parsing option %s with argument %s\n",
              long_options[option_index].name, optarg );
      case 'a': /* gps-start-time */
        localparams.startTime.gpsSeconds = atol( optarg );
        break;
      case 'A': /* gps-start-time-ns */
        localparams.startTime.gpsNanoSeconds = atol( optarg );
        break;
      case 'b': /* gps-end-time */
        localparams.endTime.gpsSeconds = atol( optarg );
        break;
      case 'B': /* gps-end-time-ns */
        localparams.endTime.gpsNanoSeconds = atol( optarg );
        break;
      case 'c': /* h1 channel-name */
        localparams.channel[LAL_IFO_H1] = optarg;
        break;
      case 'D': /* h1 frame-cache */
        localparams.dataCache[LAL_IFO_H1] = optarg;
        break;
      case 'y': /* l1 channel-name */
        localparams.channel[LAL_IFO_L1] = optarg;
        break;
      case 'Y': /* l1 frame-cache */
        localparams.dataCache[LAL_IFO_L1] = optarg;
        break;
      case 'z': /* v1 channel-name */
        localparams.channel[LAL_IFO_V1] = optarg;
        break;
      case 'Z': /* v1 frame-cache */
        localparams.dataCache[LAL_IFO_V1] = optarg;
        break;
      case 'x': /* h2 channel-name */
        localparams.channel[LAL_IFO_H2] = optarg;
        break;
      case 'X': /* h2 frame-cache */
        localparams.dataCache[LAL_IFO_H2] = optarg;
        break;
      case 'd': /* debug-level */
        set_debug_level( optarg );
        break;
      case 'e': /* cutoff-frequency */
        localparams.lowCutoffFrequency = atof( optarg );
        break;
      case 'E': /* highpass-frequency */
        localparams.highpassFrequency = atof( optarg );
        break;
      case 'f': /* right-ascension */
        localparams.rightAscension = atof( optarg ) * LAL_PI_180;
        break;
      case 'F': /* Declination */
        localparams.declination = atof( optarg ) * LAL_PI_180;
        break;
      case 'h': /* help */
        coh_PTF_usage( program );
        exit( 0 );
      case 'i': /* injection-file */
        localparams.injectFile = optarg;
        break;
      case 'j':
        localparams.threshold = atof(optarg); 
        break;
      case 'J':
        localparams.timeWindow = atof(optarg);
        break;
      case 'k': /* user-tag */
        strncpy( localparams.userTag, optarg, sizeof( localparams.userTag ) - 1 );
        break;
      case 'K': /* ifo-tag */
        strncpy( localparams.ifoTag, optarg, sizeof( localparams.ifoTag ) - 1 );
        break;
      case 'l':
        localparams.nonspinSNR2threshold = atof(optarg);
        break;
      case 'L':
        localparams.spinSNR2threshold = atof(optarg);
        break;
      case 'm': /* spin bank */
        localparams.spinBank = optarg;
        break;
      case 'M': /* non spin bank */
        localparams.noSpinBank = optarg;
        break;
      case 'n': /* only-segment-numbers */
        localparams.segmentsToDoList = optarg;
        break;
      case 'N': /* only-template-number */
        localparams.templatesToDoList = optarg;
        break;
      case 'o': /* output-file */
        strncpy( localparams.outputFile, optarg, sizeof( localparams.outputFile ) - 1 );
        break;
      case 'O': /* bank-file */
        localparams.bankFile = optarg;
        break;
      case 'p': /* num auto chisq points */
        localparams.numAutoPoints = atoi( optarg );
        break;
      case 'P': /* Auto veto time step */
        localparams.autoVetoTimeStep = atof( optarg );
        break;
      case 'r': /* random seed */
        localparams.randomSeed = atoi( optarg );
        break;
      case 'R': /* dynamic range factor */
        localparams.dynRangeFac = atof( optarg );
        break;
      case 's': /* sample rate */
        localparams.sampleRate = atof( optarg );
        break;
      case 'S': /* segment-duration */
        localparams.segmentDuration = atof( optarg );
        break;
      case 't': /* bank veto template bank */
        localparams.bankVetoBankName = optarg;
        break;
      case 'T': /* inverse-spec-length */
        localparams.invSpecLen = atof( optarg );
        break;
      case 'u': /* trig-start-time */
        localparams.trigStartTimeNS = (INT8) atol( optarg ) * LAL_INT8_C(1000000000);
        break;
      case 'U': /* trig-end-time */
        localparams.trigEndTimeNS = (INT8) atol( optarg ) * LAL_INT8_C(1000000000);
        break;
      case 'w': /* block-duration */
        localparams.duration = atof( optarg );
        break;
      case 'W': /* pad-data */
        localparams.padData = atof( optarg );
        break;
      case 'V': /* version */
        PRINT_VERSION( "ring" );
        exit( 0 );
      case '?':
        error( "unknown error while parsing options\n" );
      default:
        error( "unknown error while parsing options\n" );
    }
  }

  if ( optind < argc )
  {
    fprintf( stderr, "extraneous command line arguments:\n" );
    while ( optind < argc )
      fprintf( stderr, "%s\n", argv[optind++] );
    exit( 1 );
  }

  *params = localparams;

  return 0;
}

/* sets default values for parameters */
static int coh_PTF_default_params( struct coh_PTF_params *params )
{
  /* overall, default values are zero */
  memset( params, 0, sizeof( *params ) );

  /* Right Ascension and declination must be provided */
  params->rightAscension = -1000.;
  params->declination = -1000.;

  /* dynamic range factor must be greater than zero */
  params->dynRangeFac = 1.0;

  /* negative value means use the "default" values */
  params->highpassFrequency     = -1.0; /* use low-frequency cutoff */

  /* segments and templates to do: all of them */
  params->segmentsToDoList  = "^-$";
  params->templatesToDoList = "^-$";

  /* flags specifying what to do: default is to do everything */
  params->getBank     = 1;
  params->getData     = 1;
  params->getSpectrum = 1;
  params->doFilter    = 1;

  return 0;
}

/* macro for testing validity of a condition that prints an error if invalid */
#define sanity_check( condition ) \
  ( condition ? 0 : ( fputs( #condition " not satisfied\n", stderr ), error( "sanity check failed\n" ) ) ) 

/* check sanity of parameters and sets appropriate values of unset parameters */
int coh_PTF_params_sanity_check( struct coh_PTF_params *params )
{
  UINT4 recordLength = 0;
  UINT4 segmentLength = 0;
  UINT4 segmentStride = 0;
  UINT4 truncateLength = 0;
  UINT4 ifoNumber;
  INT8 startTime;
  INT8 endTime;


  if ( params->getSpectrum ) /* need data and response if not strain data */
    sanity_check( params->getData && (params->strainData) );

  /* parameters required to get data */
  if ( params->getData )
  {
    /* checks on data duration */
    startTime = epoch_to_ns( &params->startTime );
    endTime   = epoch_to_ns( &params->endTime );
    sanity_check( startTime > 0 );
    sanity_check( endTime > startTime );
    sanity_check( params->duration > 0 );
    sanity_check( 1e9*params->duration == ((endTime - startTime)) );

    /* checks on size of data record */
    sanity_check( params->sampleRate > 0 );
    recordLength = params->duration * params->sampleRate;
    sanity_check( recordLength > 0 );
    for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
    {
      if ( params->haveTrig[ifoNumber] )
      {
        sanity_check(params->channel[ifoNumber]);
        sanity_check(params->dataCache[ifoNumber]);
      }
    }
  }

  /* parameters required to get spectrum */
  if ( params->getSpectrum )
  {
    /* checks on size of data segments and stride */
    sanity_check( params->segmentDuration > 0 );
    segmentLength = floor(params->segmentDuration * params->sampleRate + 0.5);
    sanity_check( recordLength / segmentLength > 0 );
    params->strideDuration = 0.5 * params->segmentDuration;
    segmentStride = floor(params->strideDuration * params->sampleRate + 0.5);
    sanity_check( segmentStride > 0 );
    params->truncateDuration = 0.25 * params->strideDuration;
    truncateLength = floor(params->truncateDuration * params->sampleRate + 0.5);
    sanity_check( truncateLength > 0 );
    /* record length, segment length and stride need to be commensurate */
    sanity_check( !( (recordLength - segmentLength) % segmentStride ) );
    params->numOverlapSegments = 1 + (recordLength - segmentLength)/segmentStride;
    sanity_check( ! (params->numOverlapSegments % 2) ); /* required to be even for median-mean method */

    /* checks on data input information */
    /*sanity_check( params->channel );*/
    sanity_check( params->dynRangeFac > 0.0 );
  }
  sanity_check( params->rightAscension >= 0. && params->rightAscension <= 2.*LAL_PI);
  sanity_check( params->declination >= -LAL_PI/2. && params->declination <= LAL_PI/2.);

// This needs fixing. Need a check on whether segmentsToDoList and 
// analyzeInjSegsOnly have been given.
//  sanity_check( ! ((params->segmentsToDoList  != "^-$") && (params->analyzeInjSegsOnly)));

  return 0;
}

/* Sanity check for coh_PTF_inspiral specific */
int coh_PTF_params_inspiral_sanity_check( struct coh_PTF_params *params )
{
  sanity_check( params->threshold );
  sanity_check( params->timeWindow );
  sanity_check( params->outputFile );
  if ( params->bankFile )
  {
    fprintf(stderr,"Please use --spin-bank and/or --non-spin-bank with this ");
    fprintf(stderr,"code and not --bank-file.\n");
    sanity_check(! params->bankFile );
  }
  if ( params->doBankVeto && (! params->bankVetoBankName) )
  {
    fprintf(stderr, "When using --do-bank-veto you must also supply ");
    fprintf(stderr, "--bank-veto-templates. \n" );
    sanity_check(!( params->doBankVeto && (! params->bankVetoBankName)));
  }
  if ( params->bankVetoBankName && (! params->doBankVeto) )
  {
    fprintf(stderr, "Supplying --bank-veto-templates will do nothing if ");
    fprintf(stderr, "--do-bank-veto is not given. \n" );
  }
  if ( params->doAutoVeto && (! (params->autoVetoTimeStep && params->numAutoPoints)))
  {
    fprintf(stderr, "When using --do-auto-veto you must also supply ");
    fprintf(stderr, "--num-auto-chisq-points and --auto-veto-time-step\n");
    sanity_check(params->doAutoVeto && params->autoVetoTimeStep && params->numAutoPoints);
  }
  sanity_check(params->spinBank || params->noSpinBank);
  return 0;
}

/* Sanity check for coh_PTF_spin_checker specific */
int coh_PTF_params_spin_checker_sanity_check( struct coh_PTF_params *params )
{
  sanity_check( params->spinSNR2threshold > 0 );
  sanity_check( params->nonspinSNR2threshold > 0 );
  sanity_check( params->spinBank );
  sanity_check( params->noSpinBank);

  return 0;
}


/* prints a help message */
static int coh_PTF_usage( const char *program )
{
  fprintf( stderr, "usage: %s options\n", program );
  fprintf( stderr, "\ngeneral options:\n" );
  fprintf( stderr, "--help                     print this message\n" );
  fprintf( stderr, "--version                  print the version of the code\n" );
  fprintf( stderr, "--verbose                  print verbose messages while running\n" );
  fprintf( stderr, "--debug-level=dbglvl       set the LAL debug level\n" );

  fprintf( stderr, "\ndata reading options:\n" );
  fprintf( stderr, "--h1-data                  Analyze h1 data \n" );
  fprintf( stderr, "--h2-data                  Analyze h2 data \n" );
  fprintf( stderr, "--l1-data                  Analyze l1 data \n" );
  fprintf( stderr, "--v1-data                  Analyze v1 data \n" );
  fprintf( stderr, "--h1-frame-cache=cachefile    name of the frame cache file\n" );
  fprintf( stderr, "--h2-frame-cache=cachefile    name of the frame cache file\n" );
  fprintf( stderr, "--l1-frame-cache=cachefile    name of the frame cache file\n" );
  fprintf( stderr, "--v1-frame-cache=cachefile    name of the frame cache file\n" );
  fprintf( stderr, "--h1-channel-name             data channel to analyze\n" );
  fprintf( stderr, "--h2-channel-name             data channel to analyze\n" );
  fprintf( stderr, "--l1-channel-name             data channel to analyze\n" );
  fprintf( stderr, "--v1-channel-name             data channel to analyze\n" );
  fprintf( stderr, "--gps-start-time=tstart    GPS start time of data to analyze (sec)\n" );
  fprintf( stderr, "--gps-start-time-ns=tstartns  nanosecond residual of start time\n" );
  fprintf( stderr, "--gps-end-time=tstop       GPS stop time of data to analyze (sec)\n" );
  fprintf( stderr, "--gps-end-time-ns=tstopns  nanosecond residual of stop time\n" );
  fprintf( stderr, "\nsimulated data options:\n" );
  fprintf( stderr, "--simulated-data           create simulated white Gaussian noise\n" );
  fprintf( stderr, "--random-seed=seed         random number seed for simulated data\n" );
  fprintf( stderr, "--sample-rate=srate        sampling rate of simulated data (Hz)\n" );
  fprintf( stderr, "--zero-data                create a time series of zeros\n" );

  fprintf( stderr, "\ndata conditioning options:\n" );
  fprintf( stderr, "--highpass-frequency=fhi   high-pass filter data at frequency fhi (Hz)\n" );
  fprintf( stderr, "--sample-rate=srate        decimate data to be at sample rate srate (Hz)\n" );

  fprintf( stderr, "\ncalibration options:\n" );
  fprintf( stderr, "--strain-data              data is strain (already calibrated)\n" );
  fprintf( stderr, "--dynamic-range-factor=dynfac  scale calibration by factor dynfac\n" );

  fprintf( stderr, "\ndata segmentation options:\n" );
  fprintf( stderr, "--segment-duration=duration  duration of a data segment (sec)\n" );
  fprintf( stderr, "--block-duration=duration    duration of an analysis block (sec)\n" );
  fprintf( stderr, "--pad-data=duration          input data padding (sec)\n" );

  fprintf( stderr, "\npower spectrum options:\n" );
  fprintf( stderr, "--white-spectrum           use uniform white power spectrum\n" );
  fprintf( stderr, "--cutoff-frequency=fcut    low frequency spectral cutoff (Hz)\n" );
  fprintf( stderr, "--inverse-spec-length=t    set length of inverse spectrum to t seconds\n" );
  fprintf( stderr, "\nbank generation options:\n" );
  fprintf( stderr, "--bank-file=name           Location of tmpltbank xml file\n" );
  fprintf( stderr, "--spin-bank=name   Location of output spin bank for spin checker or input spin bank for cohPTF_inspiral \n");
  fprintf( stderr, "--non-spin-bank=name   Location of output non spin bank for spin checker or input non spin bank for cohPTF_inspiral \n");
  fprintf( stderr, "\nfiltering options:\n" );
  fprintf( stderr, "--only-segment-numbers=seglist  list of segment numbers to compute\n" );
  fprintf( stderr, "--analyze-inj-segs-only  Only analyze times when injections have been made\n" );
  fprintf( stderr, "--only-template-numbers=tmpltlist  list of filter templates to use\n" );
  fprintf( stderr, "--right-ascension=ra right ascension of external trigger in degrees\n" );
  fprintf( stderr, "--declination=dec declination of external trigger in degrees\n" );
  fprintf( stderr, "--injection-file=file list of software injections to make into the data. If this option is not given injections are not made\n");

  fprintf( stderr, "\nTrigger extraction options:\n" );
  fprintf( stderr, "--snr-threshold=threshold Only keep triggers with a snr above threshold\n" );
  fprintf( stderr, "--non-spin-snr2-threshold=value SNR squared value over which a non spin trigger is considered found for spin checker program\n" );
  fprintf( stderr, "--spin-snr2-threshold=value SNR squared value over which a spin trigger is considered found for spin checker program\n" );
  fprintf( stderr, "--trig-time-window=window Keep loudest trigger within window seconds\n" );
  fprintf( stderr, "--do-null-stream Calculate Null SNR for potential triggers\n");
  fprintf( stderr, "--do-trace-snr Calculate Trace SNR for potential triggers \n");
  fprintf( stderr, "--do-bank-veto Calculate Bank Veto for potential triggers \n");
  fprintf( stderr, "--bank-veto-templates File containing templates to use for bank veto \n");
  fprintf( stderr, "--do-auto-veto Calculate Auto Veto for potential triggers \n");
  fprintf( stderr, "--num-auto-chisq-points Number of points to use in calculating auto veto \n");
  fprintf( stderr, "--auto-veto-time-step Seperation between points for auto veto \n");
  fprintf( stderr, "\ntrigger output options:\n" );
  fprintf( stderr, "--output-file=outfile      output triggers to file outfile\n" );
  fprintf( stderr, "--trig-start-time=sec      output only triggers after GPS time sec. CURRENTLY NONFUNCTIONAL\n" );
  fprintf( stderr, "--trig-end-time=sec        output only triggers before GPS time sec. CURRENTLY NONFUNCTIONAL\n" );
  fprintf( stderr, "--ifo-tag=string           set ifotag to string for file naming\n" );
  fprintf( stderr, "--user-tag=string          set the process_params usertag to string\n" );

  fprintf( stderr, "\nintermediate data output options:\n" );
  fprintf( stderr, "--write-raw-data           write raw data before injection or conditioning\n" );
  fprintf( stderr, "--write-data               write data after injection and conditioning\n" );
  fprintf( stderr, "--write-inv-spectrum       write inverse power spectrum\n" );
  fprintf( stderr, "--write-segment            write overwhitened data segments\n" );
  fprintf( stderr, "--write-filter-output      write filtered data segments\n" );

  return 0;
}

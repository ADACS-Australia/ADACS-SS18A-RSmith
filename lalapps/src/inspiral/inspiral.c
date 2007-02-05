/*----------------------------------------------------------------------- 
 * 
 * File Name: inspiral.c
 *
 * Author: Brown, D. A.
 * 
 * Revision: $Id$
 * 
 *-----------------------------------------------------------------------
 */

#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <regex.h>
#include <time.h>
#include <math.h>

#include <FrameL.h>

#include <lalapps.h>
#include <series.h>
#include <processtable.h>
#include <lalappsfrutils.h>

#include <lal/LALConfig.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALDatatypes.h>
#include <lal/AVFactories.h>
#include <lal/LALConstants.h>
#include <lal/FrameStream.h>
#include <lal/ResampleTimeSeries.h>
#include <lal/Calibration.h>
#include <lal/FrameCalibration.h>
#include <lal/Window.h>
#include <lal/TimeFreqFFT.h>
#include <lal/IIRFilter.h>
#include <lal/BandPassTimeSeries.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/Date.h>
#include <lal/Units.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpSP.h>
#include <lal/FindChirpTD.h>
#include <lal/FindChirpBCV.h>
#include <lal/FindChirpBCVSpin.h>
#include <lal/FindChirpChisq.h>
#include <lal/LALTrigScanCluster.h>

RCSID( "$Id$" );

#define CVS_ID_STRING "$Id$"
#define CVS_NAME_STRING "$Name$"
#define CVS_REVISION "$Revision$"
#define CVS_ID_STRING "$Id$"
#define CVS_NAME_STRING "$Name$"
#define CVS_REVISION "$Revision$"
#define CVS_SOURCE "$Source$"
#define CVS_DATE "$Date$"
#define PROGRAM_NAME "inspiral"

/* define the parameters for a 1.4,1.4 sloar mass standard candle with snr 8 */
#define CANDLE_MASS1 1.4
#define CANDLE_MASS2 1.4
#define CANDLE_RHOSQ 64.0


#define ADD_SUMM_VALUE( sv_name, sv_comment, val, intval ) \
if ( this_summ_value ) \
{ \
  this_summ_value = this_summ_value->next = (SummValueTable *) \
  LALCalloc( 1, sizeof(SummValueTable) ); \
} \
else \
{ \
  summvalue.summValueTable = this_summ_value = (SummValueTable *) \
  LALCalloc( 1, sizeof(SummValueTable) ); \
} \
LALSnprintf( this_summ_value->program, LIGOMETA_PROGRAM_MAX, "%s", \
  PROGRAM_NAME ); \
this_summ_value->version = 0; \
this_summ_value->start_time = searchsumm.searchSummaryTable->out_start_time; \
this_summ_value->end_time = searchsumm.searchSummaryTable->out_end_time; \
this_summ_value->value = (REAL4) val; \
this_summ_value->intvalue = (INT4) intval; \
LALSnprintf( this_summ_value->name, LIGOMETA_SUMMVALUE_NAME_MAX, "%s", \
    sv_name ); \
LALSnprintf( this_summ_value->ifo, LIGOMETA_IFO_MAX, "%s", ifo ); \
LALSnprintf( this_summ_value->comment, LIGOMETA_SUMMVALUE_COMM_MAX, \
    "%s", sv_comment ); \

double rint(double x);
int arg_parse_check( int argc, char *argv[], MetadataTable procparams );

#ifdef LALAPPS_CONDOR
extern int condor_compress_ckpt;
void init_image_with_file_name( char *ckpt_file_name );
void ckpt_and_exit( void );
#endif

typedef struct
tagFrameHNode
{
  FrameH *frHeader;
  struct tagFrameHNode *next;
}
FrameHNode;

/*
 *
 * variables that control program behaviour
 *
 */


/* debugging */
extern int vrbflg;                      /* verbocity of lal function    */

/* checkpointing */
INT4  dataCheckpoint = 0;               /* condor checkpoint after data */
CHAR  ckptPath[FILENAME_MAX];           /* input and ckpt file path     */
CHAR  outputPath[FILENAME_MAX];         /* output data file path        */

/* input data parameters */
INT8  gpsStartTimeNS   = 0;             /* input data GPS start time ns */
LIGOTimeGPS gpsStartTime;               /* input data GPS start time    */
INT8  gpsEndTimeNS     = 0;             /* input data GPS end time ns   */
LIGOTimeGPS gpsEndTime;                 /* input data GPS end time      */
INT4  padData = 0;                      /* saftety margin on input data */
CHAR  *fqChanName       = NULL;         /* name of data channel         */
INT4  globFrameData     = 0;            /* glob *.gwf to get frame data */
CHAR  *frInCacheName    = NULL;         /* cache file containing frames */
CHAR  *frInType         = NULL;         /* type of data frames          */
INT4  numPoints         = -1;           /* points in a segment          */
INT4  numSegments       = -1;           /* number of segments           */
INT4  ovrlap            = -1;           /* overlap between segments     */
CHAR  ifo[3];                           /* two character ifo code       */
CHAR *channelName = NULL;               /* channel string               */
UINT4 inputDataLength = 0;              /* number of points in input    */
REAL4 strainHighPassFreq = -1;          /* h(t) high pass frequency     */
INT4  strainHighPassOrder = -1;         /* h(t) high pass filter order  */
REAL4 strainHighPassAtten = -1;         /* h(t) high pass attenuation   */
enum { undefined, real_4, real_8 } calData = undefined; /* cal data type*/

/* data conditioning parameters */
LIGOTimeGPS slideData   = {0,0};        /* slide data for time shifting */
INT4   resampFiltType   = -1;           /* low pass filter used for res */
INT4   sampleRate       = -1;           /* sample rate of filter data   */
INT4   highPass         = -1;           /* enable high pass on raw data */
REAL4  highPassFreq     = 0;            /* high pass frequency          */
INT4   highPassOrder    = -1;           /* order of the td iir filter   */
REAL4  highPassAtten    = -1;           /* attenuation of the td filter */

REAL4  fLow             = -1;           /* low frequency cutoff         */
INT4   specType         = -1;           /* use median or mean psd       */
INT4   badMeanPsd       = 0;            /* use a mean with no overlap   */
INT4   invSpecTrunc     = -1;           /* length of inverse spec (s)   */
REAL4  dynRangeExponent = -1;           /* exponent of dynamic range    */
CHAR  *calCacheName     = NULL;         /* location of calibration data */
INT4   globCalData      = 0;            /* glob for calibration frames  */
INT4   pointCal         = 0;            /* don't average cal over chunk */
CHAR  *injectionFile    = NULL;         /* name of file containing injs */
CHAR **tdFollowUpFiles  = NULL;         /* name of file containing td f */
INT4   numTDFiles       = 0;            /* Number of files to follow up */
int    injectOverhead   = 0;            /* inject h+ into detector      */
REAL4  mmFast           = -1.0;         /* match for the --fast option  */

/* matched filter parameters */
CHAR *bankFileName      = NULL;         /* name of input template bank  */
INT4  startTemplate     = -1;           /* index of first template      */
INT4  stopTemplate      = -1;           /* index of last template       */
INT4  numChisqBins      = -1;           /* number of chisq bins         */
REAL4 chisqDelta        = -1;           /* set chisq delta param        */
REAL4 snrThresh         = -1;           /* signal to noise thresholds   */
REAL4 chisqThresh       = -1;           /* chisq veto thresholds        */
Clustering clusterMethod;               /* chosen clustering algorithm  */  
REAL4 clusterWindow     = -1;           /* cluster over time window     */  
Approximant approximant;                /* waveform approximant         */
CHAR *approximantName   = NULL;         /* waveform approximant name    */
INT4 bcvConstraint      = 0;            /* constraint BCV filter        */
INT4 flagFilterInjOnly  = -1;           /* flag for filtering inj. only */ 

/* rsq veto params */
INT4 enableRsqVeto      = -1;           /* enable the r^2 veto          */
REAL4 rsqVetoWindow     = -1;           /* r^2 veto time window         */ 
REAL4 rsqVetoThresh     = -1;           /* r^2 veto threshold           */
INT4 doRsqVeto          = 0;            /* do the r^2 veto              */
REAL4 rsqVetoTimeThresh = -1;           /* r^2 veto time threshold      */
REAL4 rsqVetoMaxSNR     = -1;           /* r^2 veto maximum snr         */
REAL4 rsqVetoCoeff      = -1;           /* r^2 veto coefficient         */
REAL4 rsqVetoPow        = -1;           /* r^2 veto power               */

/* generic simulation parameters */
enum { unset, urandom, user } randSeedType = unset;    /* sim seed type */
INT4  randomSeed        = 0;            /* value of sim rand seed       */
REAL4 gaussVar          = 64.0;         /* variance of gaussian noise   */
INT4  gaussianNoise     = 0;            /* make input data gaussian     */
INT4  unitResponse      = 0;            /* set the response to unity    */

/* template bank simulation params */
INT4  bankSim           = 0;            /* number of template bank sims */
CHAR *bankSimFileName   = NULL;         /* file contining sim_inspiral  */
FindChirpBankSimParams bankSimParams = { 0, 0, -1, -1, NULL, -1, NULL, NULL };
                                        /* template bank sim params     */

/* reverse chirp bank option */
INT4 reverseChirpBank      = 0;           /* enable the reverse chirp     */
                                        /* template bank option         */

/* output parameters */
CHAR  *userTag          = NULL;         /* string the user can tag with */
CHAR  *ifoTag           = NULL;         /* string to tag parent IFOs    */
CHAR   fileName[FILENAME_MAX];          /* name of output files         */
INT4   maximizationInterval = 0;        /* Max over template in this    */ 
                                        /* maximizationInterval Nanosec */ 
                                        /* interval                     */
trigScanType trigScanMethod = trigScanNone;    
                                        /* Switch for clustering        */
                                        /* triggers in template         */
                                        /* parameters and end time      */
REAL8  trigScanDeltaEndTime = 0.0;      /* Use this interval (msec)     */
                                        /* over trigger end time while  */
                                        /* using trigScanCluster        */
REAL8  trigScanMetricScalingFac = -1.0;     
/* Use this scaling factor for the volume spanned by a trigger in the   */
/* parameter space. When set to x, the volume is taken to be that of the*/
/* ambiguity ellipsoid at a 'minimal match' of (1.0-x).                 */ 
                                        /* original bank entered at the */
                                        /* command line                 */
INT2  trigScanAppendStragglers = -1;    /* Switch to append cluster     */
                                        /* out-liers (stragglers)       */

INT8   trigStartTimeNS  = 0;            /* write triggers only after    */
INT8   trigEndTimeNS    = 0;            /* write triggers only before   */
INT8   outTimeNS        = 0;            /* search summ out time         */
int    enableOutput     = -1;           /* write out inspiral events    */
MetadataTableType outputMask = sngl_inspiral_table; /* default to all   */
int    writeRawData     = 0;            /* write the raw data to a file */
int    writeFilterData  = 0;            /* write post injection data    */
int    writeResponse    = 0;            /* write response function used */
int    writeSpectrum    = 0;            /* write computed psd to file   */
int    writeRhosq       = 0;            /* write rhosq time series      */
int    writeChisq       = 0;            /* write chisq time series      */
int    writeCData       = 0;            /* write complex time series c  */

/* other command line args */
CHAR comment[LIGOMETA_COMMENT_MAX];     /* process param comment        */

int main( int argc, char *argv[] )
{
  /* lal function variables */
  LALStatus             status = blank_status;
  LALLeapSecAccuracy    accuracy = LALLEAPSEC_LOOSE;

  /* frame input data */
  FrCache      *frInCache = NULL;
  FrCache      *frGlobCache = NULL;
  FrCache      *calCache = NULL;
  FrStream     *frStream = NULL;
  FrChanIn      frChan;
  FrCacheSieve  sieve;
  const size_t  calGlobLen = FILENAME_MAX;
  CHAR         *calGlobPattern;

  /* frame output data */
  struct FrFile *frOutFile  = NULL;
  struct FrameH *outFrame   = NULL;
  FrameHNode *coherentFrames = NULL;
  FrameHNode *thisCoherentFrame = NULL;
  UINT4          nRhosqFr = 0;
  UINT4          nChisqFr = 0;

  /* raw input data storage */
  REAL4TimeSeries               chan;
  REAL8TimeSeries               strainChan;
  REAL4FrequencySeries          spec;
  COMPLEX8FrequencySeries       resp;
  DataSegmentVector            *dataSegVec = NULL;
  COMPLEX8TimeSeries           *coherentInputData = NULL;

  /* structures for preconditioning */
  COMPLEX8FrequencySeries       injResp;
  COMPLEX8FrequencySeries      *injRespPtr;
  ResampleTSParams              resampleParams;
  LALWindowParams               wpars;
  AverageSpectrumParams         avgSpecParams;

  /* findchirp data structures */
  FindChirpInitParams          *fcInitParams   = NULL;
  FindChirpSegmentVector       *fcSegVec       = NULL;
  FindChirpDataParams          *fcDataParams   = NULL;
  FindChirpTmpltParams         *fcTmpltParams  = NULL;
  FindChirpFilterParams        *fcFilterParams = NULL;
  FindChirpFilterInput         *fcFilterInput  = NULL;
  FindChirpStandardCandle       candle;

  /* inspiral template structures */
  INT4                          numTmplts    = 0;
  InspiralTemplate             *bankHead     = NULL;
  InspiralTemplate             *bankCurrent  = NULL;
  InspiralTemplateNode         *tmpltHead    = NULL;
  InspiralTemplateNode         *tmpltCurrent = NULL;
  InspiralTemplateNode         *tmpltInsert  = NULL;

  /* inspiral events */
  INT4                          numEvents   = 0;
  SnglInspiralTable            *event       = NULL;
  SnglInspiralTable            *eventList   = NULL;
  SnglInspiralTable            *tempTmplt   = NULL;
  MetadataTable                 savedEvents;

  /* output data */
  MetadataTable         proctable;
  MetadataTable         procparams;
  MetadataTable         searchsumm;
  MetadataTable         searchsummvars;
  MetadataTable         siminspiral;
  MetadataTable         siminstparams;
  MetadataTable         summvalue;
  MetadataTable         filtertable;
  SearchSummvarsTable  *this_search_summvar = NULL;
  SummValueTable       *this_summ_value = NULL;
  ProcessParamsTable   *this_proc_param = NULL;
  LIGOLwXMLStream       results;

  /* counters and other variables */
  const LALUnit strainPerCount = {0,{0,0,0,0,0,1,-1},{0,0,0,0,0,0,0}};
  UINT4 i, j, k;
  INT4  inserted;
  INT4  currentLevel;
  INT4  cDataForFrame = 0;
  CHAR  fname[FILENAME_MAX];
  CHAR  cdataStr[LALNameLength];
  REAL8 inputLengthNS;
  UINT4 numInputPoints;
  const REAL8 epsilon = 1.0e-8;
  UINT4 resampleChan = 0;
  REAL8 tsLength;
  INT8  durationNS = 0;
  CalibrationUpdateParams calfacts, inj_calfacts;
  REAL4 alpha = 0;
  REAL4 alphabeta = 0;
  REAL4 inj_alpha = 0;
  REAL4 inj_alphabeta = 0;
  CHAR tmpChName[LALNameLength];
  REAL8 inputDeltaT;
  REAL8 dynRange = 1.0;
  REAL8 trigTime = 0.0;
  REAL8 lowerBound = 0.0;
  REAL8 upperBound = 0.0;

  /* random number generator parameters */
  RandomParams *randParams = NULL;

  /* template bank simulation variables */
  UINT4 bankSimCutLowIndex = 0;
  UINT4 bankSimCutHighIndex = 0;
  INT4  bankSimCount = 0;
  REAL4 matchNorm = 0;
  SnglInspiralTable  *loudestEventHead = NULL;
  SnglInspiralTable  *thisLoudestEvent = NULL;
  SimInspiralTable   *thisSimInspiral = NULL;
  SimInstParamsTable *thisSimInstParams = NULL;
  SimInspiralTable   *bankSimHead = NULL;
  SimInspiralTable   *thisBankSim = NULL;
  FindChirpBankSimParams *bankSimParamsPtr = NULL;

  /* injection information */
  int                  numInjections = 0;
  SimInspiralTable    *injections = NULL;
  SimInspiralTable    *thisInj = NULL;

  /* Time domain follow-up events */
  SnglInspiralTable    *tdFollowUpEvents = NULL;
  SnglInspiralTable    *thisFollowUpEvent  = NULL;

  /* --fast option related variables */
  UINT4  *analyseThisTmplt = NULL;
  INT4    thisTemplateIndex = 0;
  UINT4   analyseTag;
  
  /* trigScan clustering input parameters */
  trigScanClusterIn  *condenseIn=NULL; 


  /*
   *
   * initialization
   *
   */


  /* set up inital debugging values */
  lal_errhandler = LAL_ERR_EXIT;
  set_debug_level( "1" );

  /* create the process and process params tables */
  proctable.processTable = (ProcessTable *) calloc( 1, sizeof(ProcessTable) );
  LAL_CALL( LALGPSTimeNow ( &status, &(proctable.processTable->start_time),
        &accuracy ), &status );
  LAL_CALL( populate_process_table( &status, proctable.processTable, 
        PROGRAM_NAME, CVS_REVISION, CVS_SOURCE, CVS_DATE ), &status );
  this_proc_param = procparams.processParamsTable = (ProcessParamsTable *) 
    calloc( 1, sizeof(ProcessParamsTable) );
  memset( comment, 0, LIGOMETA_COMMENT_MAX * sizeof(CHAR) );

  /* create the search summary and zero out the summvars table */
  searchsumm.searchSummaryTable = (SearchSummaryTable *)
    calloc( 1, sizeof(SearchSummaryTable) );
  searchsummvars.searchSummvarsTable = NULL;

  /* create the filter table */
  filtertable.filterTable = (FilterTable *)
    calloc( 1, sizeof(FilterTable) );

  /* zero out the checkpoint and output paths */
  memset( ckptPath, 0, FILENAME_MAX * sizeof(CHAR) );
  memset( outputPath, 0, FILENAME_MAX * sizeof(CHAR) );

  /* call the argument parse and check function */
  arg_parse_check( argc, argv, procparams );

  /* wind to the end of the process params table */
  for ( this_proc_param = procparams.processParamsTable; this_proc_param->next;
      this_proc_param = this_proc_param->next );

  /* can use LALMalloc() and LALCalloc() from here onwards */

  /* populate the filter table */
  LALSnprintf( filtertable.filterTable->program, LIGOMETA_PROGRAM_MAX, "%s", 
      PROGRAM_NAME );
  filtertable.filterTable->start_time = gpsStartTime.gpsSeconds;
  LALSnprintf( filtertable.filterTable->filter_name, LIGOMETA_COMMENT_MAX,
      "%stwoPN", approximantName );

  /* fill the comment, if a user has specified on, or leave it blank */
  if ( ! *comment )
  {
    LALSnprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX, " " );
    LALSnprintf( filtertable.filterTable->comment, LIGOMETA_COMMENT_MAX, " " );
  } 
  else 
  {
    LALSnprintf( proctable.processTable->comment, LIGOMETA_COMMENT_MAX,
        "%s", comment );
    LALSnprintf( filtertable.filterTable->comment, LIGOMETA_COMMENT_MAX,
        "%s", comment );
  }

  /* put the name of the search in the search_summary comment */ 
  /* XXX note twoPN is hardwires and should be changed if we modify the    */
  /* code to allow different PN orders to be specified on the command line */
  LALSnprintf( searchsumm.searchSummaryTable->comment, LIGOMETA_COMMENT_MAX,
      "%stwoPN", approximantName );

  /* set the name of the output file */
  if ( userTag && ifoTag )
  {
    LALSnprintf( fileName, FILENAME_MAX, "%s-INSPIRAL_%s_%s-%d-%d", ifo, 
        ifoTag, userTag, gpsStartTime.gpsSeconds, 
        gpsEndTime.gpsSeconds - gpsStartTime.gpsSeconds ); 
  }
  else if ( userTag && !ifoTag )
  {
    LALSnprintf( fileName, FILENAME_MAX, "%s-INSPIRAL_%s-%d-%d", ifo, 
        userTag,  gpsStartTime.gpsSeconds, 
        gpsEndTime.gpsSeconds - gpsStartTime.gpsSeconds );
  }
  else if ( !userTag && ifoTag )
  {
    LALSnprintf( fileName, FILENAME_MAX, "%s-INSPIRAL_%s-%d-%d", ifo, 
        ifoTag,  gpsStartTime.gpsSeconds, 
        gpsEndTime.gpsSeconds - gpsStartTime.gpsSeconds ); 
  }
  else
  {
    LALSnprintf( fileName, FILENAME_MAX, "%s-INSPIRAL-%d-%d", ifo,
        gpsStartTime.gpsSeconds, 
        gpsEndTime.gpsSeconds - gpsStartTime.gpsSeconds );     
  }

  /* the number of nodes for a standalone job is always 1 */
  searchsumm.searchSummaryTable->nnodes = 1;

  /* fill the ifos field of the search summary table */
  LALSnprintf( searchsumm.searchSummaryTable->ifos, LIGOMETA_IFOS_MAX, ifo );

  /* make sure all the output table pointers are null */
  savedEvents.snglInspiralTable = NULL;
  siminspiral.simInspiralTable = NULL;
  siminstparams.simInstParamsTable = NULL;

  /* create the standard candle and database table */
  memset( &candle, 0, sizeof(FindChirpStandardCandle) );
  strncpy( candle.ifo, ifo, 2 * sizeof(CHAR) );
  candle.tmplt.mass1 = CANDLE_MASS1;
  candle.tmplt.mass2 = CANDLE_MASS2;
  candle.rhosq       = CANDLE_RHOSQ;
  candle.tmplt.totalMass = candle.tmplt.mass1 + candle.tmplt.mass2;
  candle.tmplt.mu = candle.tmplt.mass1 * candle.tmplt.mass2 / 
    candle.tmplt.totalMass;
  candle.tmplt.eta = candle.tmplt.mu / candle.tmplt.totalMass;

  /* create the dynamic range exponent */
  dynRange = pow( 2.0, dynRangeExponent );


  /*
   *
   * create a (possibly heirarcical) template bank
   *
   */


  /* read in the template bank from a ligo lw xml file */
  numTmplts = InspiralTmpltBankFromLIGOLw( &bankHead, bankFileName,
      startTemplate, stopTemplate );
  if ( numTmplts < 0 )
  {
    fprintf( stderr, "error: unable to read templates from %s\n", 
        bankFileName );
    exit( 1 );
  }
  else if ( numTmplts == 0 )
  {
    /* if there are no tmplts, store the time we would have analyzed and exit */
    fprintf( stdout, "no templates found in template bank file: %s\n"
        "exiting without searching for events.\n", bankFileName );

    searchsumm.searchSummaryTable->out_start_time.gpsSeconds = 
      gpsStartTime.gpsSeconds + (numPoints / (4 * sampleRate));

    LAL_CALL( LALGPStoINT8( &status, &outTimeNS,
          &(searchsumm.searchSummaryTable->out_start_time) ), &status );

    if ( ! bankSim && ( trigStartTimeNS && (trigStartTimeNS > outTimeNS) ) )
    {
      LAL_CALL( LALINT8toGPS( &status, 
            &(searchsumm.searchSummaryTable->out_start_time), 
            &trigStartTimeNS ), &status );
    }

    searchsumm.searchSummaryTable->out_end_time.gpsSeconds = 
      gpsEndTime.gpsSeconds - (numPoints / (4 * sampleRate));

    LAL_CALL( LALGPStoINT8( &status, &outTimeNS,
          &(searchsumm.searchSummaryTable->out_end_time) ), &status );

    if ( ! bankSim && ( trigEndTimeNS && (trigEndTimeNS < outTimeNS) ) )
    {
      LAL_CALL( LALINT8toGPS( &status, 
            &(searchsumm.searchSummaryTable->out_end_time), 
            &trigEndTimeNS ), &status );
    }
  }

  if ( vrbflg ) fprintf( stdout, "parsed %d templates from %s\n", 
      numTmplts, bankFileName );

  if ( numTmplts )
  {
    /* save the minimal match of the bank in the process params: create */
    /* the table entry with calloc() since it will be freed with free() */
    this_proc_param = this_proc_param->next = (ProcessParamsTable *) 
      calloc( 1, sizeof(ProcessParamsTable) ); 
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
        PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--minimal-match" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "float" ); 
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%e", 
        bankHead->minMatch );
  }

  /* create the linked list of template nodes for coarse templates */
  for ( bankCurrent = bankHead; bankCurrent; bankCurrent = bankCurrent->next )
  {
    LAL_CALL( LALFindChirpCreateTmpltNode( &status, 
          bankCurrent, &tmpltCurrent ), &status );
    if ( !tmpltHead ) tmpltHead = tmpltCurrent;
  }


  /*
   *
   * read in the input data channel
   *
   */


  /* set the time series parameters of the input data and resample params */
  memset( &resampleParams, 0, sizeof(ResampleTSParams) );
  resampleParams.deltaT = 1.0 / (REAL8) sampleRate;

  /* set the params of the input data time series */
  memset( &chan, 0, sizeof(REAL4TimeSeries) );
  memset( &strainChan, 0, sizeof(REAL8TimeSeries) );
  chan.epoch = gpsStartTime;
  chan.epoch.gpsSeconds -= padData;   /* subtract pad seconds from start */
  /* subtract slide from start */
  chan.epoch.gpsSeconds -= slideData.gpsSeconds; 
  chan.epoch.gpsNanoSeconds -= slideData.gpsNanoSeconds;
  /* copy the start time into the REAL8 h(t) time series */
  strainChan.epoch = chan.epoch;

  if ( globFrameData )
  {
    CHAR ifoRegExPattern[6];

    if ( vrbflg ) fprintf( stdout, "globbing for *.gwf frame files from %c "
        "of type %s in current directory\n", fqChanName[0], frInType );

    frGlobCache = NULL;

    /* create a frame cache by globbing all *.gwf files in the pwd */
    LAL_CALL( LALFrCacheGenerate( &status, &frGlobCache, NULL, NULL ), 
        &status );

    /* check we globbed at least one frame file */
    if ( ! frGlobCache->numFrameFiles )
    {
      fprintf( stderr, "error: no frame file files of type %s found\n",
          frInType );
      exit( 1 );
    }

    /* sieve out the requested data type */
    memset( &sieve, 0, sizeof(FrCacheSieve) );
    LALSnprintf( ifoRegExPattern, 
        sizeof(ifoRegExPattern) / sizeof(*ifoRegExPattern), ".*%c.*", 
        fqChanName[0] );
    sieve.srcRegEx = ifoRegExPattern;
    sieve.dscRegEx = frInType;
    LAL_CALL( LALFrCacheSieve( &status, &frInCache, frGlobCache, &sieve ), 
        &status );

    /* check we got at least one frame file back after the sieve */
    if ( ! frInCache->numFrameFiles )
    {
      fprintf( stderr, "error: no frame files of type %s globbed as input\n",
          frInType );
      exit( 1 );
    }

    LAL_CALL( LALDestroyFrCache( &status, &frGlobCache ), &status );
  }
  else
  {
    if ( vrbflg ) fprintf( stdout, 
        "reading frame file locations from cache file: %s\n", frInCacheName );

    /* read a frame cache from the specified file */
    LAL_CALL( LALFrCacheImport( &status, &frInCache, frInCacheName), &status );
  }

  /* open the input data frame stream from the frame cache */
  LAL_CALL( LALFrCacheOpen( &status, &frStream, frInCache ), &status );

  /* set the mode of the frame stream to fail on gaps or time errors */
  frStream->mode = LAL_FR_VERBOSE_MODE;

  /* seek to required epoch and set chan name */
  LAL_CALL( LALFrSeek( &status, &(chan.epoch), frStream ), &status );
  frChan.name = fqChanName;

  if ( calData == real_8 )
  {
    /* determine the sample rate of the raw data */
    LAL_CALL( LALFrGetREAL8TimeSeries( &status, &strainChan, &frChan, 
          frStream ), &status );

    /* copy the data paramaters from the h(t) channel to input data channel */
    LALSnprintf( chan.name, LALNameLength * sizeof(CHAR), "%s", 
        strainChan.name );
    chan.epoch          = strainChan.epoch;
    chan.deltaT         = strainChan.deltaT;
    chan.f0             = strainChan.f0;
    chan.sampleUnits    = strainChan.sampleUnits;
  }
  else
  {
    /* determine the sample rate of the raw data */
    LAL_CALL( LALFrGetREAL4TimeSeries( &status, &chan, &frChan, frStream ),
        &status );
  }

  /* store the input sample rate */
  this_search_summvar = searchsummvars.searchSummvarsTable = 
    (SearchSummvarsTable *) LALCalloc( 1, sizeof(SearchSummvarsTable) );
  LALSnprintf( this_search_summvar->name, LIGOMETA_NAME_MAX * sizeof(CHAR),
      "raw data sample rate" );
  this_search_summvar->value = inputDeltaT = chan.deltaT;

  /* determine if we need to resample the channel */
  if ( vrbflg )
  {
    fprintf( stdout, "resampleParams.deltaT = %e\n", resampleParams.deltaT );
    fprintf( stdout, "chan.deltaT = %e\n", chan.deltaT );
  }
  if ( ! ( fabs( resampleParams.deltaT - chan.deltaT ) < epsilon ) )
  {
    resampleChan = 1;
    if ( vrbflg )
      fprintf( stdout, "input channel will be resampled\n" );

    if ( resampFiltType == 0 )
    {
      resampleParams.filterType = LDASfirLP;
    }
    else if ( resampFiltType == 1 )
    {
      resampleParams.filterType = defaultButterworth;
    }
  }

  /* determine the number of points to get and create storage for the data */
  inputLengthNS = 
    (REAL8) ( gpsEndTimeNS - gpsStartTimeNS + 2000000000LL * padData );
  numInputPoints = (UINT4) floor( inputLengthNS / (chan.deltaT * 1.0e9) + 0.5 );
  if ( calData == real_8 )
  {
    /* create storage for the REAL8 h(t) input data */
    LAL_CALL( LALDCreateVector( &status, &(strainChan.data), numInputPoints ), 
        &status );
  }
  LAL_CALL( LALSCreateVector( &status, &(chan.data), numInputPoints ), 
      &status );

  if ( vrbflg ) fprintf( stdout, "input channel %s has sample interval "
      "(deltaT) = %e\nreading %d points from frame stream\n", fqChanName, 
      chan.deltaT, numInputPoints );

  if ( calData == real_8 )
  {
    /* read in the REAL8 h(t) data here */
    PassBandParamStruc strainHighpassParam;

    /* read the REAL8 h(t) data from the time series into strainChan      */
    /* which already has the correct amount of memory allocated */
    if ( vrbflg ) fprintf( stdout, "reading REAL8 h(t) data from frames... " );

    LAL_CALL( LALFrGetREAL8TimeSeries( &status, &strainChan, &frChan, 
          frStream ), &status);

    if ( vrbflg ) fprintf( stdout, "done\n" );

    /* high pass the h(t) data using the parameters specified on the cmd line*/
    strainHighpassParam.nMax = strainHighPassOrder;
    strainHighpassParam.f1 = -1.0;
    strainHighpassParam.f2 = (REAL8) strainHighPassFreq;
    strainHighpassParam.a1 = -1.0;
    strainHighpassParam.a2 = (REAL8)(1.0 - strainHighPassAtten);
    if ( vrbflg ) fprintf( stdout, 
        "applying %d order high pass to REAL8 h(t) data: "
        "%3.2f of signal passes at %4.2f Hz\n", 
        strainHighpassParam.nMax, strainHighpassParam.a2, 
        strainHighpassParam.f2 );

    LAL_CALL( LALButterworthREAL8TimeSeries( &status, &strainChan, 
          &strainHighpassParam ), &status );

    /* cast the REAL8 h(t) data to REAL4 in the chan time series       */
    /* which already has the correct amount of memory allocated */
    for ( j = 0 ; j < numInputPoints ; ++j )
    {
      chan.data->data[j] = (REAL4) ( strainChan.data->data[j] * dynRange );
    }

    /* re-copy the data paramaters from the h(t) channel to input data channel*/
    LALSnprintf( chan.name, LALNameLength * sizeof(CHAR), "%s", 
        strainChan.name );
    chan.epoch          = strainChan.epoch;
    chan.deltaT         = strainChan.deltaT;
    chan.f0             = strainChan.f0;
    chan.sampleUnits    = strainChan.sampleUnits;

    /* free the REAL8 h(t) input data */
    LAL_CALL( LALDDestroyVector( &status, &(strainChan.data) ), &status );
    strainChan.data = NULL;
  }
  else
  {
    /* read the data channel time series from frames */
    LAL_CALL( LALFrGetREAL4TimeSeries( &status, &chan, &frChan, frStream ),
        &status );
    if ( calData == real_4 )
    {
      /* multiply the input data by dynRange */
      for ( j = 0 ; j < numInputPoints ; ++j )
      {
        chan.data->data[j] *= dynRange;
      }
    } 
  }
  memcpy( &(chan.sampleUnits), &lalADCCountUnit, sizeof(LALUnit) );

  /* store the start and end time of the raw channel in the search summary */
  searchsumm.searchSummaryTable->in_start_time = chan.epoch;
  LAL_CALL( LALGPStoFloat( &status, &tsLength, &(chan.epoch) ), 
      &status );
  tsLength += chan.deltaT * (REAL8) chan.data->length;
  LAL_CALL( LALFloatToGPS( &status, 
        &(searchsumm.searchSummaryTable->in_end_time), &tsLength ), &status );

  /* close the frame file stream and destroy the cache */
  LAL_CALL( LALFrClose( &status, &frStream ), &status );
  LAL_CALL( LALDestroyFrCache( &status, &frInCache ), &status );

  /* write the raw channel data as read in from the frame files */
  if ( writeRawData ) outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
      &chan, "ct", "RAW" );

  if ( vrbflg ) fprintf( stdout, "read channel %s from frame stream\n"
      "got %d points with deltaT %e\nstarting at GPS time %d sec %d ns\n", 
      chan.name, chan.data->length, chan.deltaT, 
      chan.epoch.gpsSeconds, chan.epoch.gpsNanoSeconds );


  /*
   *
   * create the random seed if it will be needed
   *
   */


  if ( randSeedType != unset )
  {
    /* store the seed in the search summvars table */
    this_search_summvar = this_search_summvar->next = 
      (SearchSummvarsTable *) LALCalloc( 1, sizeof(SearchSummvarsTable) );
    LALSnprintf( this_search_summvar->name, LIGOMETA_NAME_MAX * sizeof(CHAR),
        "template bank simulation seed" );

    if ( randSeedType == urandom )
    {
      FILE   *fpRand = NULL;
      INT4    randByte;

      if ( vrbflg ) 
        fprintf( stdout, "obtaining random seed from /dev/urandom: " );

      randomSeed = 0;
      fpRand = fopen( "/dev/urandom", "r" );
      if ( fpRand )
      {
        for ( randByte = 0; randByte < 4 ; ++randByte )
        {
          INT4 tmpSeed = (INT4) fgetc( fpRand );
          randomSeed += tmpSeed << ( randByte * 8 );
        }
        fclose( fpRand );
      }
      else
      {
        perror( "error obtaining random seed from /dev/urandom" );
        exit( 1 );
      }
    }
    else if ( randSeedType == user )
    {
      if ( vrbflg ) 
        fprintf( stdout, "using user specified random seed: " );
    }
    else
    {
      /* should never get here */
      fprintf( stderr, "error obtaining random seed\n" );
      exit( 1 );
    }

    this_search_summvar->value = randomSeed;
    if ( vrbflg ) fprintf( stdout, "%d\n", randomSeed );

    /* create the tmplt bank random parameter structure */
    LAL_CALL( LALCreateRandomParams( &status, &randParams, randomSeed ),
        &status );
  }

  /* replace the input data with gaussian noise if necessary */
  if ( gaussianNoise )
  {
    if ( vrbflg ) fprintf( stdout, 
        "setting input data to gaussian noise with variance %e... ", gaussVar );
    memset( chan.data->data, 0, chan.data->length * sizeof(REAL4) );
    LAL_CALL( LALNormalDeviates( &status, chan.data, randParams ), &status );
    for ( j = 0; j < chan.data->length; ++j )
    {
      chan.data->data[j] *= gaussVar;
    }
    if ( vrbflg ) fprintf( stdout, "done\n" );

    /* write the raw channel data as read in from the frame files */
    if ( writeRawData ) outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
        &chan, "ct", "RAW_GAUSSIAN" );
  }


  /*
   *
   * generate the response function for the requested time
   *
   */


  /* create storage for the response function */
  memset( &resp, 0, sizeof(COMPLEX8FrequencySeries) );
  LAL_CALL( LALCCreateVector( &status, &(resp.data), numPoints / 2 + 1 ), 
      &status );

  /* set the parameters of the response to match the data */
  resp.epoch.gpsSeconds = chan.epoch.gpsSeconds + padData;
  resp.epoch.gpsNanoSeconds = chan.epoch.gpsNanoSeconds;
  resp.deltaF = (REAL8) sampleRate / (REAL8) numPoints;
  resp.sampleUnits = strainPerCount;
  strcpy( resp.name, chan.name );

  /* generate the response function for the current time */
  if ( vrbflg ) fprintf( stdout, "generating response at time %d sec %d ns\n",
      resp.epoch.gpsSeconds, resp.epoch.gpsNanoSeconds );

  /* initialize the calfacts */
  memset( &calfacts, 0, sizeof(CalibrationUpdateParams) );
  calfacts.ifo = ifo;

  /* determine length of chunk */
  if ( pointCal )
  {
    calfacts.duration.gpsSeconds = 1;
    calfacts.duration.gpsNanoSeconds = 0;
  }
  else
  {
    durationNS = gpsEndTimeNS - gpsStartTimeNS;
    LAL_CALL( LALINT8toGPS( &status, &(calfacts.duration), 
          &durationNS ), &status );
  }

  if ( calData )
  {
    /* if we are using calibrated data set the response to unity */
    for( k = 0; k < resp.data->length; ++k )
    {
      resp.data->data[k].re = (REAL4) (1.0 / dynRange);
      resp.data->data[k].im = 0.0;
    }
    if ( writeResponse ) outFrame = fr_add_proc_COMPLEX8FrequencySeries( 
        outFrame, &resp, "strain/ct", "RESPONSE_h(t)" );
  }
  else
  {
    /* create the lal calibration frame cache */
    if ( globCalData )
    {
      calGlobPattern = (CHAR *) LALCalloc( calGlobLen, sizeof(CHAR) );
      LALSnprintf( calGlobPattern, calGlobLen * sizeof(CHAR), 
          "*CAL*%s*.gwf", ifo );
      if ( vrbflg ) fprintf( stdout, "globbing for %s calibration frame files "
          "in current directory\n", calGlobPattern );
    }
    else
    {
      calGlobPattern = NULL;
      if ( vrbflg ) fprintf( stdout, 
          "reading calibration data from cache: %s\n", calCacheName );
    }

    LAL_CALL( LALCreateCalibFrCache( &status, &calCache, calCacheName, 
          NULL, calGlobPattern ), &status );

    if ( calGlobPattern ) LALFree( calGlobPattern );

    /* store the name of the calibration files used */
    for ( i = 0; i < calCache->numFrameFiles; ++i )
    {
      this_search_summvar = this_search_summvar->next = 
        (SearchSummvarsTable *) LALCalloc( 1, sizeof(SearchSummvarsTable) );
      LALSnprintf( this_search_summvar->name, LIGOMETA_NAME_MAX * sizeof(CHAR),
          "calibration frame %d", i );
      LALSnprintf( this_search_summvar->string, 
          LIGOMETA_STRING_MAX * sizeof(CHAR), "%s", 
          calCache->frameFiles[i].url );
    }

    /* get the response from the frame data */
    LAL_CALL( LALExtractFrameResponse( &status, &resp, calCache, 
          &calfacts), &status );
    LAL_CALL( LALDestroyFrCache( &status, &calCache), &status );
    alpha = (REAL4) calfacts.alpha.re;
    alphabeta = (REAL4) calfacts.alphabeta.re;
    if ( vrbflg ) fprintf( stdout, 
        "for calibration of data, alpha = %f and alphabeta = %f\n",
        alpha, alphabeta);

    if ( writeResponse ) 
      outFrame = fr_add_proc_COMPLEX8FrequencySeries( outFrame, &resp, 
          "strain/ct", "RESPONSE" );
  }

  if ( unitResponse )
  {
    /* replace the response function with unity if */
    /* we are filtering gaussian noise             */
    if ( vrbflg ) fprintf( stdout, "setting response to unity... " );
    for ( k = 0; k < resp.data->length; ++k )
    {
      resp.data->data[k].re = 1.0;
      resp.data->data[k].im = 0;
    }
    if ( vrbflg ) fprintf( stdout, "done\n" );

    if ( writeResponse ) outFrame = fr_add_proc_COMPLEX8FrequencySeries( 
        outFrame, &resp, "strain/ct", "RESPONSE_UNITY" );
  }

  /* slide the channel back to the fake time for background studies */
  chan.epoch.gpsSeconds += slideData.gpsSeconds;
  chan.epoch.gpsNanoSeconds += slideData.gpsNanoSeconds;


  if ( tdFollowUpFiles )
  {
    INT4 injSafety = 10;
    LIGOTimeGPS startKeep = gpsStartTime;
    LIGOTimeGPS endKeep   = gpsEndTime;

    startKeep.gpsSeconds -= injSafety;
    endKeep.gpsSeconds   += injSafety;

    /* read in the time domain follow-up data from XML */
    for (i = 0; i < numTDFiles; i++ )
    {
      INT4 thisTDNum = 0;
      if ( !tdFollowUpEvents )
      {
        thisTDNum = LALSnglInspiralTableFromLIGOLw(&tdFollowUpEvents, 
             tdFollowUpFiles[i], 0, -1);
        thisFollowUpEvent = tdFollowUpEvents;
      }
      else
      {
        thisTDNum = LALSnglInspiralTableFromLIGOLw(&(thisFollowUpEvent->next),
             tdFollowUpFiles[i], 0, -1);
      }
      if ( thisTDNum < 0 )
      {
        fprintf( stderr, "Error reading file %s\n", tdFollowUpFiles[i] );
        exit( 1 );
      }

      if ( thisTDNum )
      {
        while ( thisFollowUpEvent->next )
        {
          thisFollowUpEvent = thisFollowUpEvent->next;
        }
      }
    }
    
    tdFollowUpEvents = XLALTimeCutSingleInspiral( tdFollowUpEvents,
        &startKeep, &endKeep);
  }

  /*
   *
   * inject signals into the raw, unresampled data
   *
   */


  if ( injectionFile )
  {
    /* get injections within 500 seconds of either end of the segment.   */
    /* a 0.4,0.4 MACHO starting at 30.0 Hz has length 435.374683 seconds */
    /* so this should be plenty of safety. better to waste cpu than miss */
    /* injected signals...                                               */
    INT4 injSafety = 500;

    /* read in the injection data from XML */
    numInjections = SimInspiralTableFromLIGOLw( &injections, injectionFile,
        gpsStartTime.gpsSeconds - injSafety, 
        gpsEndTime.gpsSeconds + injSafety );

    if ( numInjections < 0 )
    {
      fprintf( stderr, "error: cannot read injection file" );
      exit( 1 );
    }
    else if ( numInjections )
    {
      /* see if we need a higher resolution response to do the injections */
      if ( resampleChan )
      {
        /* we need a different resolution of response function for injections */
        UINT4 rateRatio = floor( resampleParams.deltaT / chan.deltaT + 0.5 );
        UINT4 rawNumPoints = rateRatio * numPoints;

        if ( vrbflg ) fprintf( stdout, "rateRatio = %d\nrawNumPoints = %d\n"
            "chan.deltaT = %e\n", rateRatio, rawNumPoints, chan.deltaT );

        memset( &injResp, 0, sizeof(COMPLEX8FrequencySeries) );
        LAL_CALL( LALCCreateVector( &status, &(injResp.data), 
              rawNumPoints / 2 + 1 ), &status );
        injResp.epoch = resp.epoch;
        injResp.deltaF = 1.0 / ( rawNumPoints * chan.deltaT );
        injResp.sampleUnits = strainPerCount;
        strcpy( injResp.name, chan.name );

        if ( calData )
        {
          /* if we are using calibrated data set the response to unity */
          if ( vrbflg ) fprintf( stdout, 
              "setting injection response to inverse dynRange... " );
          for ( k = 0; k < injResp.data->length; ++k )
          {
            injResp.data->data[k].re = (REAL4)(1.0/dynRange);
            injResp.data->data[k].im = 0.0;
          }
          injRespPtr = &injResp;
          if ( writeResponse ) 
            outFrame = fr_add_proc_COMPLEX8FrequencySeries( outFrame, 
                &injResp, "strain/ct", "RESPONSE_INJ_CAL" );
        }
        else
        {
          /* generate the response function for the current time */
          if ( vrbflg ) fprintf( stdout, 
              "generating high resolution response at time %d sec %d ns\n"
              "length = %d points, deltaF = %e Hz\n",
              resp.epoch.gpsSeconds, resp.epoch.gpsNanoSeconds,
              injResp.data->length, injResp.deltaF );

          /* initialize the inj_calfacts */
          memset( &inj_calfacts, 0, sizeof(CalibrationUpdateParams) );
          inj_calfacts.ifo = ifo;
          LAL_CALL( LALINT8toGPS( &status, &(inj_calfacts.duration), 
                &durationNS ), &status );

          /* create the lal calibration frame cache */
          if ( globCalData )
          {
            calGlobPattern = (CHAR *) LALCalloc( calGlobLen, sizeof(CHAR) );
            LALSnprintf( calGlobPattern, calGlobLen * sizeof(CHAR), 
                "*CAL*%s*.gwf", ifo );
            if ( vrbflg ) fprintf( stdout, 
                "globbing for %s calibration frame files "
                "in current directory\n", calGlobPattern );
          }
          else
          {
            calGlobPattern = NULL;
            if ( vrbflg ) fprintf( stdout, 
                "reading calibration data from cache: %s\n", calCacheName );
          }

          LAL_CALL( LALCreateCalibFrCache( &status, &calCache, calCacheName, 
                NULL, calGlobPattern ), &status );

          if ( calGlobPattern ) LALFree( calGlobPattern );

          /* store the name of the calibration files used */
          for ( i = 0; i < calCache->numFrameFiles; ++i )
          {
            this_search_summvar = this_search_summvar->next = 
              (SearchSummvarsTable *) 
              LALCalloc( 1, sizeof(SearchSummvarsTable) );
            LALSnprintf( this_search_summvar->name, 
                LIGOMETA_NAME_MAX * sizeof(CHAR), 
                "injection calibration frame %d", i );
            LALSnprintf( this_search_summvar->string, 
                LIGOMETA_STRING_MAX * sizeof(CHAR), "%s", 
                calCache->frameFiles[i].url );
          }

          /* extract the calibration from frames */
          LAL_CALL( LALExtractFrameResponse( &status, &injResp, calCache, 
                &inj_calfacts ), &status );
          LAL_CALL( LALDestroyFrCache( &status, &calCache), &status );

          inj_alpha = (REAL4) calfacts.alpha.re;
          inj_alphabeta = (REAL4) calfacts.alphabeta.re;
          if ( vrbflg ) fprintf( stdout, 
              "for injections, alpha = %f and alphabeta = %f\n",
              inj_alpha, inj_alphabeta);

          injRespPtr = &injResp;

          if ( writeResponse ) 
            outFrame = fr_add_proc_COMPLEX8FrequencySeries( 
                outFrame, &injResp, "strain/ct", "RESPONSE_INJ" );
        }

        if ( gaussianNoise )
        {
          /* replace the response function with unity if */
          /* we are filtering gaussian noise             */
          if ( vrbflg ) fprintf( stdout, "setting response to unity... " );
          for ( k = 0; k < injResp.data->length; ++k )
          {
            injResp.data->data[k].re = 1.0;
            injResp.data->data[k].im = 0;
          }
          if ( vrbflg ) fprintf( stdout, "done\n" );

          if ( writeResponse ) outFrame = fr_add_proc_COMPLEX8FrequencySeries( 
              outFrame, &injResp, "strain/ct", "RESPONSE_INJ_GAUSSIAN" );
        }
      }
      else
      {
        /* the data is already at the correct sample rate, just do injections */
        injRespPtr = &resp;
        memset( &injResp, 0, sizeof(COMPLEX8FrequencySeries) );
      }

      /* inject the signals, preserving the channel name (Tev mangles it) */
      LALSnprintf( tmpChName, LALNameLength * sizeof(CHAR), "%s", chan.name );

      /* if injectOverhead option, then set chan.name to "ZENITH".  
       * This causes no detector site to be found in the injection code so
       * that the injection is done directly overhead (i.e. with a response 
       * function of F+ = 1; Fx = 0) */
      if ( injectOverhead )
      {
        LALSnprintf( chan.name, LALNameLength * sizeof(CHAR), "ZENITH" );
      }

      LAL_CALL( LALFindChirpInjectSignals( &status, &chan, injections, 
            injRespPtr ), &status );
      LALSnprintf( chan.name,  LALNameLength * sizeof(CHAR), "%s", tmpChName );

      if ( vrbflg ) fprintf( stdout, "injected %d signals from %s into %s\n", 
          numInjections, injectionFile, chan.name );

      /* write the raw channel data plus injections to the output frame file */
      if ( writeRawData ) outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
          &chan, "ct", "RAW_INJ" );

      if ( injResp.data )
        LAL_CALL( LALCDestroyVector( &status, &(injResp.data) ), &status );
    }
    else
    {
      if ( vrbflg ) fprintf( stdout, "no injections in this chunk\n" );
    }
  }


  /*
   *
   * resample the data to the requested rate
   *
   */


  if ( resampleChan )
  {
    if (vrbflg) fprintf( stdout, "resampling input data from %e to %e\n",
        chan.deltaT, resampleParams.deltaT );

    LAL_CALL( LALResampleREAL4TimeSeries( &status, &chan, &resampleParams ),
        &status );

    if ( vrbflg ) fprintf( stdout, "channel %s resampled:\n"
        "%d points with deltaT %e\nstarting at GPS time %d sec %d ns\n", 
        chan.name, chan.data->length, chan.deltaT, 
        chan.epoch.gpsSeconds, chan.epoch.gpsNanoSeconds );

    /* write the resampled channel data as read in from the frame files */
    if ( writeRawData ) outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
        &chan, "ct", "RAW_RESAMP" );
  }

  /* store the filter data sample rate */
  this_search_summvar = this_search_summvar->next = 
    (SearchSummvarsTable *) LALCalloc( 1, sizeof(SearchSummvarsTable) );
  LALSnprintf( this_search_summvar->name, LIGOMETA_NAME_MAX * sizeof(CHAR),
      "filter data sample rate" );
  this_search_summvar->value = chan.deltaT;


  /*
   *
   * we have all the input data that we need, so checkpoint if requested
   *
   */


  if ( dataCheckpoint )
  {
#ifdef LALAPPS_CONDOR
    condor_compress_ckpt = 1;
    if ( ckptPath[0] )
    {
      LALSnprintf( fname, FILENAME_MAX * sizeof(CHAR), "%s/%s.ckpt", 
          ckptPath, fileName );
    }
    else
    {
      LALSnprintf( fname, FILENAME_MAX * sizeof(CHAR), "%s.ckpt", fileName );
    }
    if ( vrbflg ) fprintf( stdout, "checkpointing to file %s\n", fname );
    init_image_with_file_name( fname );
    ckpt_and_exit();
#else
    fprintf( stderr, "--data-checkpoint cannot be used unless "
        "lalapps is condor compiled\n" );
    exit( 1 );
#endif
  }


  /* 
   *
   * high pass the data, removed pad from time series and check length of data
   *
   */


  /* iir filter to remove low frequencies from data channel */
  if ( highPass )
  {
    PassBandParamStruc highpassParam;
    highpassParam.nMax = highPassOrder;
    highpassParam.f1 = -1.0;
    highpassParam.f2 = (REAL8) highPassFreq;
    highpassParam.a1 = -1.0;
    highpassParam.a2 = (REAL8)(1.0 - highPassAtten); /* a2 is not attenuation */

    if ( vrbflg ) fprintf( stdout, "applying %d order high pass: "
        "%3.2f of signal passes at %4.2f Hz\n", 
        highpassParam.nMax, highpassParam.a2, highpassParam.f2 );

    LAL_CALL( LALDButterworthREAL4TimeSeries( &status, &chan, &highpassParam ),
        &status );
  }

  /* remove pad from requested data from start and end of time series */
  memmove( chan.data->data, chan.data->data + padData * sampleRate, 
      (chan.data->length - 2 * padData * sampleRate) * sizeof(REAL4) );
  LALRealloc( chan.data->data, 
      (chan.data->length - 2 * padData * sampleRate) * sizeof(REAL4) );
  chan.data->length -= 2 * padData * sampleRate;
  chan.epoch.gpsSeconds += padData;

  if ( vrbflg ) fprintf( stdout, "after removal of %d second padding at "
      "start and end:\ndata channel sample interval (deltaT) = %e\n"
      "data channel length = %d\nstarting at %d sec %d ns\n", 
      padData , chan.deltaT , chan.data->length, 
      chan.epoch.gpsSeconds, chan.epoch.gpsNanoSeconds );

  if ( writeFilterData ) outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
      &chan, "ct", "FILTER" );

  /* check data length */
  if ( chan.data->length != inputDataLength )
  {
    fprintf( stderr, "error: computed channel length and requested\n"
        "input data length do not match:\nchan.data->length = %d\n"
        "inputDataLength = %d\nyou have found a bug in the code.\n"
        "please report this to <duncan@gravity.phys.uwm.edu>\n",
        chan.data->length, inputDataLength );
    exit( 1 );
  }

  /* store the start and end time of the filter channel in the search summ */
  /* noting that we don't look for events in the first and last quarter    */
  /* of each findchirp segment of the input data                           */
  LAL_CALL( LALGPStoFloat( &status, &tsLength, &(chan.epoch) ), 
      &status );
  tsLength += (REAL8) (numPoints / 4) * chan.deltaT;
  LAL_CALL( LALFloatToGPS( &status, 
        &(searchsumm.searchSummaryTable->out_start_time), &tsLength ), 
      &status );

  LAL_CALL( LALGPStoINT8( &status, &outTimeNS,
        &(searchsumm.searchSummaryTable->out_start_time) ), &status );

  if ( ! bankSim && ( trigStartTimeNS && (trigStartTimeNS > outTimeNS) ) )
  {
    /* override with trigger start time */
    LAL_CALL( LALINT8toGPS( &status,
          &(searchsumm.searchSummaryTable->out_start_time), 
          &trigStartTimeNS ), &status );
  }

  LAL_CALL( LALGPStoFloat( &status, &tsLength, &(chan.epoch) ), 
      &status );
  tsLength += chan.deltaT * ((REAL8) chan.data->length - (REAL8) (numPoints/4));
  LAL_CALL( LALFloatToGPS( &status, 
        &(searchsumm.searchSummaryTable->out_end_time), &tsLength ), 
      &status );

  LAL_CALL( LALGPStoINT8( &status, &outTimeNS,
        &(searchsumm.searchSummaryTable->out_end_time) ), &status );

  if ( ! bankSim && ( trigEndTimeNS && (trigEndTimeNS < outTimeNS) ) )
  {
    /* override with trigger end time */
    LAL_CALL( LALINT8toGPS( &status, 
          &(searchsumm.searchSummaryTable->out_end_time), 
          &trigEndTimeNS ), &status );
  }

  /* 
   *
   * create and populate findchip initialization structure 
   *
   */


  if ( ! ( fcInitParams = (FindChirpInitParams *) 
        LALCalloc( 1, sizeof(FindChirpInitParams) ) ) )
  {
    fprintf( stderr, "could not allocate memory for findchirp init params\n" );
    exit( 1 );
  }

  fcInitParams->numPoints      = numPoints;
  fcInitParams->numSegments    = numSegments;
  fcInitParams->numChisqBins   = numChisqBins;
  fcInitParams->createRhosqVec = writeRhosq;
  fcInitParams->ovrlap         = ovrlap;
  fcInitParams->approximant    = approximant;
  fcInitParams->createCVec     = writeCData;


  /*
   *
   * power spectrum estimation
   *
   */


  /* initialize findchirp data conditioning routine */
  LAL_CALL( LALFindChirpDataInit( &status, &fcDataParams, fcInitParams ), 
      &status );
  fcDataParams->invSpecTrunc = invSpecTrunc * sampleRate;
  fcDataParams->fLow = fLow;

  /* create storage for the power spectral estimate */
  memset( &spec, 0, sizeof(REAL4FrequencySeries) );
  LAL_CALL( LALSCreateVector( &status, &(spec.data), numPoints / 2 + 1 ), 
      &status );

  /* compute the windowed power spectrum for the data channel */
  avgSpecParams.window = NULL;
  switch ( specType )
  {
    case 0:
      avgSpecParams.method = useMean;
      if ( vrbflg ) fprintf( stdout, "computing mean psd" );
      break;
    case 1:
      avgSpecParams.method = useMedian;
      if ( vrbflg ) fprintf( stdout, "computing median psd" );
      break;
    case 2:
    case 3:
    case 4:
      avgSpecParams.method = useUnity;
      if ( vrbflg ) fprintf( stdout, "computing constant psd with unit value" );
      break;
    default:
      fprintf( stderr, "Error: unknown PSD estimation type: %d\n", specType );
      exit( 1 );
  }

  /* use the fft plan created by findchirp */
  avgSpecParams.plan = fcDataParams->fwdPlan;

  wpars.type = Hann;
  wpars.length = numPoints;
  if ( badMeanPsd )
  {
    avgSpecParams.overlap = 0;
    if ( vrbflg ) fprintf( stdout, " without overlap\n" );
  }
  else
  {
    avgSpecParams.overlap = numPoints / 2;
    if ( vrbflg ) 
      fprintf( stdout, " with overlap %d\n", avgSpecParams.overlap );
  }

  LAL_CALL( LALCreateREAL4Window( &status, &(avgSpecParams.window),
        &wpars ), &status );
  LAL_CALL( LALREAL4AverageSpectrum( &status, &spec, &chan, &avgSpecParams ),
      &status );
  LAL_CALL( LALDestroyREAL4Window( &status, &(avgSpecParams.window) ), 
      &status );
  strcpy( spec.name, chan.name );

  if ( specType == 2 )
  {
    /* multiply the unit power spectrum to get a gaussian psd */
    REAL4 gaussVarSq = gaussVar * gaussVar;
    if ( resampleChan )
    {
      /* reduce the variance as we have resampled the data */
      gaussVarSq *= inputDeltaT / chan.deltaT;
    }
    for ( k = 0; k < spec.data->length; ++k )
    {
      spec.data->data[k] *= 2.0 * gaussVarSq * (REAL4) chan.deltaT;
    }

    if ( vrbflg ) 
      fprintf( stdout, "set psd to constant value = %e\n", spec.data->data[0] );
  }
  else if ( specType == 3 )
  {
    /* replace the spectrum with the Initial LIGO design noise curve */
    for ( k = 0; k < spec.data->length; ++k )
    {
      REAL8 sim_psd_freq = (REAL8) k * spec.deltaF;
      REAL8 sim_psd_value;
      LALLIGOIPsd( NULL, &sim_psd_value, sim_psd_freq );
      spec.data->data[k] = 9.0e-46 * sim_psd_value;
    }
    
    if ( vrbflg ) fprintf( stdout, "set psd to Initial LIGO design\n" );
  }
  else if ( specType == 4 )
  {
    /* replace the spectrum with the Advanced LIGO design noise curve */
    for ( k = 0; k < spec.data->length; ++k )
    {
      REAL8 sim_psd_freq = (REAL8) k * spec.deltaF;
      REAL8 sim_psd_value;
      LALAdvLIGOPsd( NULL, &sim_psd_value, sim_psd_freq );
      spec.data->data[k] = 9.0e-46 * sim_psd_value;
    }
    
    if ( vrbflg ) fprintf( stdout, "set psd to Advanced LIGO design\n" );
  }

  /* write the spectrum data to a file */
  if ( writeSpectrum ) outFrame = fr_add_proc_REAL4FrequencySeries( outFrame, 
      &spec, "ct^2/Hz", "PSD" );


  /*
   *
   * initialize the template bank simulation
   *
   */


  if ( bankSim )
  {
    if ( vrbflg ) fprintf( stdout, "initializing template bank simulation\n" );

    /* use the created random parameters */
    bankSimParams.randParams = randParams;

    /* override the initialization parameters for a bank simulation */
    fcInitParams->numSegments  = 1;
    fcInitParams->ovrlap       = 0;
    fcInitParams->numChisqBins = 0;
    snrThresh                  = 0;
    chisqThresh                = LAL_REAL4_MAX;
    chisqDelta                 = LAL_REAL4_MAX;

    if ( bankSimFileName )
    {
      /* read in the simuation parameters from a sim_inspiral_table */
      bankSim = SimInspiralTableFromLIGOLw( &bankSimHead, bankSimFileName,
          0, 0 );
      for ( thisBankSim = bankSimHead; thisBankSim; 
          thisBankSim = thisBankSim->next )
      {
        /* set the time of the injection to zero so it is injected in  */
        /* the middle of the data segment: the other times are ignored */
        thisBankSim->geocent_end_time.gpsSeconds = 0;
        thisBankSim->geocent_end_time.gpsNanoSeconds = 0;
      }
      thisBankSim = bankSimHead;
      if ( vrbflg ) fprintf( stdout, "Read %d bank sim parameters from %s\n",
            bankSim, bankSimFileName );
    }
    else
    {
      /* use the bank sim params to generate the simulated signals */
      bankSimParamsPtr = &bankSimParams;
      if ( vrbflg ) 
        fprintf( stdout, "internally generating bank sim siganls\n" );
    }

    bankSimCutLowIndex = XLALFindChirpBankSimInitialize( &spec, &resp, fLow );

    if ( vrbflg ) 
      fprintf( stdout, "psd low frequency cutoff index = %d\n", bankSimCutLowIndex );
    if ( writeSpectrum ) outFrame = fr_add_proc_REAL4FrequencySeries( outFrame, 
        &spec, "strain^2/Hz", "PSD_SIM" );
    if ( writeResponse ) outFrame = fr_add_proc_COMPLEX8FrequencySeries( 
        outFrame, &resp, "strain/ct", "RESPONSE_SIM" );
  }


  /*
   *
   * create the data structures needed for findchirp
   *
   */


  if ( vrbflg ) fprintf( stdout, "initializing findchirp... " );

  /* create the data segment vector from the input data */
  LAL_CALL( LALInitializeDataSegmentVector( &status, &dataSegVec,
        &chan, &spec, &resp, fcInitParams ), &status );

  /* set the analyze flag according to what we are doing */
  if ( tdFollowUpFiles || injectionFile )
  {
    if (tdFollowUpFiles)
    {
      /* Only analyze segments containing coincident BCV triggers */
      XLALFindChirpSetFollowUpSegment (dataSegVec, &tdFollowUpEvents);
    }
    else
    {
      /* set the analyzeSegment flag only on segments with injections */
      XLALFindChirpSetAnalyzeSegment (dataSegVec, injections);
    }
  }

  /* create the findchirp data storage */
  LAL_CALL( LALCreateFindChirpSegmentVector( &status, &fcSegVec, 
        fcInitParams ), &status );

  /* initialize the template functions */
  LAL_CALL( LALFindChirpTemplateInit( &status, &fcTmpltParams, 
        fcInitParams ), &status );

  fcDataParams->dynRange = fcTmpltParams->dynRange = dynRange;
  fcTmpltParams->deltaT = chan.deltaT;
  fcTmpltParams->fLow = fLow;
  fcTmpltParams->reverseChirpBank = reverseChirpBank;

  /* initialize findchirp filter functions */
  LAL_CALL( LALFindChirpFilterInit( &status, &fcFilterParams, fcInitParams ), 
      &status );
  fcFilterParams->deltaT = chan.deltaT;
  fcFilterParams->chisqParams->approximant = approximant;

  /* set up parameters for the filter output veto */
  if ( enableRsqVeto )
  {
    fcFilterParams->filterOutputVetoParams = (FindChirpFilterOutputVetoParams *)
      LALCalloc( 1, sizeof(FindChirpFilterOutputVetoParams) );
    fcFilterParams->filterOutputVetoParams->rsqvetoWindow = rsqVetoWindow;
    fcFilterParams->filterOutputVetoParams->rsqvetoThresh = rsqVetoThresh;
    if ( doRsqVeto )
    {
      fcFilterParams->filterOutputVetoParams->rsqvetoTimeThresh = rsqVetoTimeThresh;
      fcFilterParams->filterOutputVetoParams->rsqvetoMaxSNR = rsqVetoMaxSNR;
      fcFilterParams->filterOutputVetoParams->rsqvetoCoeff = rsqVetoCoeff;
      fcFilterParams->filterOutputVetoParams->rsqvetoPow = rsqVetoPow;
    }
  }
  else
  {
    fcFilterParams->filterOutputVetoParams = NULL;
  }

  LAL_CALL( LALCreateFindChirpInput( &status, &fcFilterInput, fcInitParams ), 
      &status );

  LAL_CALL( LALFindChirpChisqVetoInit( &status, fcFilterParams->chisqParams, 
        fcInitParams->numChisqBins, fcInitParams->numPoints ), 
      &status );

  /* initialize findchirp clustering params */
  fcFilterParams->clusterMethod = clusterMethod;
  fcFilterParams->clusterWindow = clusterWindow;

  /* parse the thresholds */
  fcFilterParams->rhosqThresh = snrThresh * snrThresh;
  fcFilterParams->chisqThresh = chisqThresh;
  fcFilterParams->chisqDelta  = chisqDelta;

  if ( vrbflg ) fprintf( stdout, "done\n" );


  /*
   *
   * matched filtering engine
   *
   */


  /* begin loop over number of template bank simulation */
  /* if we are not doing a template bank simulation,    */
  /* this exectues the main part of the filtering code  */
  /* just one (which is what we want to do)             */
  bankSimCount = 0;
  do
  {
    if ( bankSim )
    {
      if ( vrbflg ) 
        fprintf( stdout, "bank simulation %d/%d\n", bankSimCount, bankSim );

      /* zero out the input data segment and the injection params */
      if ( vrbflg ) fprintf( stdout, 
          "zeroing data stream and adding random injection for bank sim... " );
      memset( chan.data->data, 0, chan.data->length * sizeof(REAL4) );

      /* inject a random signal if we are doing a template bank simulation */ 
      if ( ! siminspiral.simInspiralTable )
        thisSimInspiral = siminspiral.simInspiralTable =
          XLALFindChirpBankSimInjectSignal( dataSegVec, &resp, thisBankSim, 
              bankSimParamsPtr );
      else
        thisSimInspiral = thisSimInspiral->next = 
          XLALFindChirpBankSimInjectSignal( dataSegVec, &resp, thisBankSim, 
              bankSimParamsPtr );

      /* write the channel data plus injection to the output frame file */
      if ( writeRawData ) outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
          &chan, "ct", "SIM" );
      if ( vrbflg ) fprintf( stdout, "done\n" );
    }


    /*
     *
     * condition data segments for filtering
     *
     */


    switch ( approximant )
    {
      case TaylorT1:
      case TaylorT2:
      case TaylorT3:
      case GeneratePPN:
      case PadeT1:
      case EOB:
        if ( vrbflg ) 
          fprintf( stdout, "findchirp conditioning data for TD\n" );
        LAL_CALL( LALFindChirpTDData( &status, fcSegVec, dataSegVec, 
              fcDataParams ), &status );
        break;

      case FindChirpSP:
        if ( vrbflg ) 
          fprintf( stdout, "findchirp conditioning data for SP\n" );
        LAL_CALL( LALFindChirpSPData( &status, fcSegVec, dataSegVec, 
              fcDataParams ), &status );
        break;
       
      case BCV:
        if ( vrbflg ) 
          fprintf( stdout, "findchirp conditioning data for BCV\n" );
        LAL_CALL( LALFindChirpBCVData( &status, fcSegVec, dataSegVec, 
              fcDataParams ), &status );
        break;

      case BCVSpin:
        if ( vrbflg ) 
          fprintf( stdout, "findchirp conditioning data for BCVSpin\n" );
        LAL_CALL( LALFindChirpTDData( &status, fcSegVec, dataSegVec, 
              fcDataParams ), &status );
        break;
      default:
        fprintf( stderr, "error: unknown waveform approximant for data\n" );
        exit( 1 );
        break;
    }

    if ( bankSim )
    {
      if ( vrbflg ) fprintf( stdout,
          "computing minimal match normalization... " );
      /* compute the bankSimCutHighIndex based on the fFinal variable stored in bankSim. */
      if ( (UINT4)(thisSimInspiral->f_final / spec.deltaF) < fcDataParams->wtildeVec->length )
      {
	bankSimCutHighIndex = (UINT4)(thisSimInspiral->f_final / spec.deltaF);
      }
      else
      {
	bankSimCutHighIndex = fcDataParams->wtildeVec->length;
      }
      matchNorm = XLALFindChirpBankSimSignalNorm( fcDataParams, fcSegVec, 
          bankSimCutLowIndex, bankSimCutHighIndex );
	
      if ( vrbflg ) fprintf( stdout, "%e\n", matchNorm );
    }
    else
    {
      if ( approximant == FindChirpSP ) 
      {
        /* compute the standard candle */
        REAL4 cannonDist = 1.0; /* Mpc */
        REAL4 m  = (REAL4) candle.tmplt.totalMass;
        REAL4 mu = (REAL4) candle.tmplt.mu;
        REAL4 distNorm = 2.0 * LAL_MRSUN_SI / (cannonDist * 1e6 * LAL_PC_SI);
        REAL4 candleTmpltNorm = sqrt( (5.0*mu) / 96.0 ) *
          pow( m / (LAL_PI*LAL_PI) , 1.0/3.0 ) *
          pow( LAL_MTSUN_SI / (REAL4) chan.deltaT, -1.0/6.0 );

        distNorm *= fcTmpltParams->dynRange;
        candleTmpltNorm *= candleTmpltNorm;
        candleTmpltNorm *= distNorm * distNorm;

        candle.sigmasq = 4.0 * ( (REAL4) chan.deltaT / (REAL4) numPoints );
        candle.sigmasq *= candleTmpltNorm * 
          fcSegVec->data->segNorm->data[fcSegVec->data->segNorm->length-1];

        candle.distance = sqrt( candle.sigmasq / candle.rhosq );

        if ( vrbflg ) 
        {
          fprintf( stdout, "candle m = %e\ncandle mu = %e\n"
              "candle.rhosq = %e\nchan.deltaT = %e\nnumPoints = %d\n"
              "fcSegVec->data->segNorm->data[fcSegVec->data->segNorm->length-1]"
              " = %e\ncandleTmpltNorm = %e\ncandle.distance = %e Mpc\n"
              "candle.sigmasq = %e\n",
              m, mu, candle.rhosq, chan.deltaT, numPoints, 
              fcSegVec->data->segNorm->data[fcSegVec->data->segNorm->length-1], 
              candleTmpltNorm, candle.distance, candle.sigmasq );
          fflush( stdout );
        }
      }
      else if ( approximant == BCV )
      {
        /* compute the standard candle for a  5-5 Msun inspiral*/
        REAL4 cannonDist = 1.0; /* Mpc */
        REAL4 m  = 10.0;
        REAL4 mu = 2.5;
        REAL4 k1 = numPoints * chan.deltaT ;
        UINT4 kmax = 432 * k1 ; /* 432 = fISCO for 5-5 */
        REAL4 distNorm = 2.0 * LAL_MRSUN_SI / (cannonDist * 1e6 * LAL_PC_SI);
        REAL4 candleTmpltNorm = sqrt( (5.0*mu) / 96.0 ) *
          pow( m/(LAL_PI*LAL_PI) , 1.0/3.0 ) * 
          pow(LAL_MTSUN_SI / (REAL4) chan.deltaT, -1.0/6.0);
        distNorm *= fcTmpltParams->dynRange;
        candleTmpltNorm *= candleTmpltNorm;
        candleTmpltNorm *= distNorm * distNorm;
        candle.sigmasq = 4.0 * ( (REAL4) chan.deltaT / (REAL4) numPoints );
        candle.sigmasq *= candleTmpltNorm *
          fcSegVec->data->segNorm->data[kmax];
        candle.distance = sqrt( candle.sigmasq / candle.rhosq );

        /* for 5-5 Msun... */
        candle.tmplt.mass1 = 5.0;
        candle.tmplt.mass2 = 5.0;
        candle.tmplt.totalMass = 10.0;
        candle.tmplt.mu = 2.5;
        candle.tmplt.eta = 0.25;

        if ( vrbflg ) 
        {
          fprintf( stdout, "candle m = %e\ncandle mu = %e\n"
              "candle.rhosq = %e\nchan.deltaT = %e\nnumPoints = %d\n"
              "fcSegVec->data->segNorm->data[kmax] = %e\n"
              "kmax = %d\ncandleTmpltNorm = %e\ncandle.distance = %e Mpc \n"
              "candle.sigmasq=%e\n",
              m,mu,candle.rhosq,chan.deltaT,
              numPoints,fcSegVec->data->segNorm->data[kmax],kmax,
              candleTmpltNorm,candle.distance,candle.sigmasq);
          fflush(stdout);
        }
      }
      else 
      {
        if ( vrbflg )
        {
          fprintf( stdout, "standard candle not calculated; \n"
              "chan.deltaT = %e\nnumPoints = %d\n",
              chan.deltaT, numPoints );
          fflush( stdout );
        }
      }  
    }


    /*================================================================
     * This is a dense bit of logical constructs and basically 
     * implements the following truth table
     *
     *   Injection   Follow-Up    --fast      Action
     *   ---------   ---------    --------  ----------
     *      T           F           F        SetAnalyseTemplate()
     *      T           F           T              "
     *      T           T           F              "
     *      T           T           T        TagTemplateAndSegment()
     *      F           T           T              "
     *      F           T           F        Do Nothing
     *      F           F           F        Do Nothing
     *      F           F           T        Irrelevant (do nothing)
     *================================================================*/
    if ( numTmplts > 0 && ( injectionFile  || (tdFollowUpFiles && mmFast >= 0.0) ))
    {
      /* Make space for analyseThisTmplt */
      analyseThisTmplt = (UINT4 *) LALCalloc (numTmplts, sizeof(UINT4));

      /* If we are performing a follow up, we only want to follow up templates
       * near coincident events, regardless of the injection file used
       */
      if ( injectionFile && (!tdFollowUpFiles || mmFast < 0) )
      {
          
      /* set the analyseThisTmplt flag on templates     */
      /* that are 'close' to the injections             */
        LAL_CALL( LALFindChirpSetAnalyseTemplate( &status, analyseThisTmplt,
            mmFast, fcSegVec->data[0].data->deltaF, sampleRate, fcDataParams,
            numTmplts, tmpltHead, numInjections, injections ), &status );

      }
      else
      {
        XLALFindChirpTagTemplateAndSegment(dataSegVec, tmpltHead, &tdFollowUpEvents,
                ifo, mmFast, analyseThisTmplt );
      }
    }  

    /* If using trigScan clustering, then init the clustering */
    /* input parameters before they are freed up in the code. */
    /* Anand :: We had to put the numTmplts > 0 condition as  */
    /* trig2tmplt would sometime return an empty template     */
    /* bank file in the absence of coinc triggers.            */
    if ( trigScanMethod && numTmplts > 0 )
    { 
        XLALPopulateTrigScanInput( &condenseIn, fcDataParams, 
                fcTmpltParams, fcFilterParams, bankHead );
    }


    /*
     *
     * hierarchial search engine
     *
     */


    for ( tmpltCurrent = tmpltHead, inserted = 0, thisTemplateIndex = 0; 
        tmpltCurrent; 
        tmpltCurrent = tmpltCurrent->next, inserted = 0, thisTemplateIndex++ )
    {

      /* If we are injecting or in td-follow-up mode and the    */
      /* analyseThisTmpltFlag is down look no further:          */
      /* simply continue to the next template                   */
      if ( injectionFile )
      {
        if ( ! analyseThisTmplt[thisTemplateIndex] )
          continue;
        else
          if ( vrbflg ) fprintf( stdout,
              "\n\n === Template %d going through \n\n", thisTemplateIndex );
      }

      /*  generate template */
      switch ( approximant )
      {
        case TaylorT1:
        case TaylorT2:
        case TaylorT3:
        case GeneratePPN:
        case PadeT1:
        case EOB:
          LAL_CALL( LALFindChirpTDTemplate( &status, fcFilterInput->fcTmplt, 
                tmpltCurrent->tmpltPtr, fcTmpltParams ), &status );
          break;

        case FindChirpSP:
          LAL_CALL( LALFindChirpSPTemplate( &status, fcFilterInput->fcTmplt, 
                tmpltCurrent->tmpltPtr, fcTmpltParams ), &status );
          break;

        case BCV:
          LAL_CALL( LALFindChirpBCVTemplate( &status, fcFilterInput->fcTmplt, 
                tmpltCurrent->tmpltPtr, fcTmpltParams ), &status );
          break;

        case BCVSpin:
          LAL_CALL( LALFindChirpBCVSpinTemplate( &status, 
                fcFilterInput->fcTmplt, tmpltCurrent->tmpltPtr, 
                fcTmpltParams, fcDataParams ), &status );
          break;

        default:
          fprintf( stderr, "error: unknown waveform template approximant \n" );
          exit( 1 );
          break;
      }

      /* loop over data segments */
      for ( i = 0; i < fcSegVec->length ; ++i )
      {
        INT8 fcSegStartTimeNS;
        INT8 fcSegEndTimeNS;

        LAL_CALL( LALGPStoINT8( &status, &fcSegStartTimeNS, 
              &(fcSegVec->data[i].data->epoch) ), &status );
        fcSegEndTimeNS = fcSegStartTimeNS + (INT8)
          ( (REAL8) numPoints * 1e9 * fcSegVec->data[i].deltaT );

        /* skip segment if it is not contained in the trig start or end times */
        if ( ! bankSim && 
            ( (trigStartTimeNS && (trigStartTimeNS > fcSegEndTimeNS)) || 
            (trigEndTimeNS && (trigEndTimeNS < fcSegStartTimeNS)) ) )
        { 
          if ( vrbflg ) fprintf( stdout, 
              "skipping segment %d/%d [%lld-%lld] (outside trig time)\n", 
              fcSegVec->data[i].number, fcSegVec->length, 
              fcSegStartTimeNS, fcSegEndTimeNS );

          continue;
        }

        /* By default assume that we are going to analyse the   */
        /* segment with the template (analyseTag = 1)           */
        analyseTag = 1;

        /* If injections are being done or if in td follow-up mode */
        /* check if for any reason the analyseTag flag needs to be */
        /* brought down                                            */
        if ( tdFollowUpFiles || ( injectionFile  && flagFilterInjOnly ))
        {
          if ( tdFollowUpFiles )
          {
            if ( mmFast >= 0.0 )
            {
                /* Compare tmplt flag with segment number using bitwise and */
                if ( !(analyseThisTmplt[thisTemplateIndex] & ( 1u << i )))
                {
                    analyseTag = 0;
                }
            }
            else if ( !fcSegVec->data[i].analyzeSegment )
            {
              analyseTag = 0;
            }

          }
          else
          {
            analyseTag = XLALCmprSgmntTmpltFlags( numInjections,
                analyseThisTmplt[thisTemplateIndex],
                fcSegVec->data[i].analyzeSegment );
          }
        }

        /* filter data segment */ 
        if ( fcSegVec->data[i].level == tmpltCurrent->tmpltPtr->level && 
            analyseTag )
        {
          if ( vrbflg ) fprintf( stdout, 
              "filtering segment %d/%d [%lld-%lld] "
              "against template %d/%d (%e,%e)\n", 
              fcSegVec->data[i].number,  fcSegVec->length,
              fcSegStartTimeNS, fcSegEndTimeNS,
              tmpltCurrent->tmpltPtr->number, numTmplts,
              fcFilterInput->fcTmplt->tmplt.mass1, 
              fcFilterInput->fcTmplt->tmplt.mass2 );

          fcFilterInput->segment = fcSegVec->data + i;

          /* decide which filtering routine to use */
          switch ( approximant )
          {
            case TaylorT1:
            case TaylorT2:
            case TaylorT3:
            case GeneratePPN:
            case PadeT1:
            case EOB:
              /* construct normalization for time domain templates... */
              LAL_CALL( LALFindChirpTDNormalize( &status, 
                    fcFilterInput->fcTmplt, fcFilterInput->segment, 
                    fcDataParams ), &status );
              /* ...and fall through to FindChirpFilterSegment() */
            case FindChirpSP:
              LAL_CALL( LALFindChirpFilterSegment( &status, 
                    &eventList, fcFilterInput, fcFilterParams ), &status ); 
              break;

            case BCV:
              if (!bcvConstraint)
              {
                LAL_CALL( LALFindChirpBCVFilterSegment( &status,
                      &eventList, fcFilterInput, fcFilterParams ), &status );
              }
              else
              { 
                LAL_CALL( LALFindChirpBCVCFilterSegment( &status,
                      &eventList, fcFilterInput, fcFilterParams ), &status );
              }
              break;

            case BCVSpin:
              LAL_CALL( LALFindChirpBCVSpinFilterSegment( &status,
                    &eventList, fcFilterInput, fcFilterParams, fcDataParams 
                    ), &status );
              break;

            default:
              fprintf( stderr, 
                  "error: unknown waveform approximant for filter\n" );
              exit( 1 );
              break;
          }

          if ( writeRhosq )
          {
            CHAR snrsqStr[LALNameLength];
            LALSnprintf( snrsqStr, LALNameLength*sizeof(CHAR), 
                "SNRSQ_%d", nRhosqFr++ );
            strcpy( fcFilterParams->rhosqVec->name, chan.name );
            outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
                fcFilterParams->rhosqVec, "none", snrsqStr );
          }

          if ( writeCData && ! strcmp(ifo,tmpltCurrent->tmpltPtr->ifo) )
          {
            trigTime = tmpltCurrent->tmpltPtr->end_time.gpsSeconds + 1e-9 * 
              tmpltCurrent->tmpltPtr->end_time.gpsNanoSeconds;
            lowerBound = gpsStartTime.gpsSeconds + numPoints/(4 * sampleRate );
            upperBound = gpsEndTime.gpsSeconds - numPoints/(4 * sampleRate );

            if ( trigTime >= lowerBound && trigTime <= upperBound )
            {
              tempTmplt = (SnglInspiralTable *) 
                LALCalloc(1, sizeof(SnglInspiralTable) );
              tempTmplt->event_id = (EventIDColumn *) 
                LALCalloc(1, sizeof(EventIDColumn) );
              tempTmplt->mass1 = tmpltCurrent->tmpltPtr->mass1;
              tempTmplt->end_time.gpsSeconds = 
                tmpltCurrent->tmpltPtr->end_time.gpsSeconds;
              tempTmplt->end_time.gpsNanoSeconds = 
                tmpltCurrent->tmpltPtr->end_time.gpsNanoSeconds;
              tempTmplt->event_id->id = tmpltCurrent->tmpltPtr->event_id->id;

              LAL_CALL( LALFindChirpCreateCoherentInput( &status,
                  &coherentInputData, fcFilterParams->cVec, 
                  tempTmplt, 2.0, numPoints / 4 ), &status );

              LALFree( tempTmplt->event_id );
              LALFree( tempTmplt );

              if ( coherentInputData )
              {
                cDataForFrame = 1;
		LALSnprintf( cdataStr, LALNameLength*sizeof(CHAR),
			     "CData_%Ld", tmpltCurrent->tmpltPtr->event_id->id );
                strcpy( coherentInputData->name, chan.name );
                if ( ! coherentFrames )
                {
                  thisCoherentFrame = coherentFrames = (FrameHNode *) 
                    LALCalloc( 1, sizeof(FrameHNode) );
                }
                else
                {
                  thisCoherentFrame = thisCoherentFrame->next = (FrameHNode *) 
                    LALCalloc( 1, sizeof(FrameHNode) );
                }
                thisCoherentFrame->frHeader = fr_add_proc_COMPLEX8TimeSeries( 
                    outFrame, coherentInputData, "none", cdataStr );
                LAL_CALL( LALCDestroyVector( &status, 
                      &(coherentInputData->data) ), &status );
                LALFree( coherentInputData );
                coherentInputData = NULL;
              }
            }
          }

          if ( writeChisq )
          {
            CHAR chisqStr[LALNameLength];
            REAL4TimeSeries chisqts;
            LALSnprintf( chisqStr, LALNameLength*sizeof(CHAR), 
                "CHISQ_%d", nChisqFr++ );
            chisqts.epoch = fcFilterInput->segment->data->epoch;
            memcpy( &(chisqts.name), fcFilterInput->segment->data->name,
                LALNameLength * sizeof(CHAR) );
            chisqts.deltaT = fcFilterInput->segment->deltaT;
            chisqts.data = fcFilterParams->chisqVec;
            outFrame = fr_add_proc_REAL4TimeSeries( outFrame, 
                &chisqts, "none", chisqStr );
          }
          if ( vrbflg ) 
            fprintf( stdout, "epoch = %d\n",
                fcFilterInput->segment->data->epoch.gpsSeconds );
        }
        else
        {
          if ( vrbflg ) fprintf( stdout, "skipping segment %d/%d [%lld-%lld] "
              "(segment level %d, template level %d)\n", 
              fcSegVec->data[i].number, fcSegVec->length, 
              fcSegStartTimeNS, fcSegEndTimeNS,
              fcSegVec->data[i].level, tmpltCurrent->tmpltPtr->level );
        }

        /*  test if filter returned any events */
        if ( eventList )
        {
          if ( vrbflg ) fprintf( stdout, 
              "segment %d rang template [m (%e,%e)] [psi (%e,%e)]\n",
              fcSegVec->data[i].number,
              fcFilterInput->fcTmplt->tmplt.mass1, 
              fcFilterInput->fcTmplt->tmplt.mass2,
              fcFilterInput->fcTmplt->tmplt.psi0, 
              fcFilterInput->fcTmplt->tmplt.psi3 );

          if ( tmpltCurrent->tmpltPtr->fine != NULL && inserted == 0 )
          {
            if ( vrbflg ) fprintf( stdout, 
                "inserting fine templates into list\n" );

            tmpltInsert = tmpltCurrent;
            inserted = 1;
            fcSegVec->data[i].level += 1;

            for ( bankCurrent = tmpltCurrent->tmpltPtr->fine ; 
                bankCurrent; bankCurrent = bankCurrent->next )
            {
              LAL_CALL( LALFindChirpCreateTmpltNode( &status, 
                    bankCurrent, &tmpltInsert ), &status );
            }

          }
          else if ( ! tmpltCurrent->tmpltPtr->fine )
          {
            if ( vrbflg ) fprintf( stdout, "***>  dumping events  <***\n" );

            if ( ! savedEvents.snglInspiralTable )
            {
              savedEvents.snglInspiralTable = eventList;
            }
            else
            {
              event->next = eventList;
            }
          }
          else
          {
            if ( vrbflg ) fprintf( stdout, 
                "already inserted fine templates, skipping\n" ); 

            fcSegVec->data[i].level += 1;
          } 

          /* save a ptr to the last event in the list and count the events */
          ++numEvents;
          while ( eventList->next )
          {
            eventList = eventList->next;
            ++numEvents;
          }
          event = eventList;
          eventList = NULL;
        } /* end if ( events ) */

        /* if going up a level, remove inserted nodes, reset segment levels */ 
        if ( tmpltCurrent->next && (tmpltCurrent->next->tmpltPtr->level < 
              tmpltCurrent->tmpltPtr->level) )
        {
          /* record the current number */
          currentLevel = tmpltCurrent->tmpltPtr->level;

          /* decrease segment filter levels if the have been increased */
          for ( i = 0 ; i < fcSegVec->length; i++ )
          {
            if ( fcSegVec->data[i].level == currentLevel )
            {
              fcSegVec->data[i].level -= 1;
            }
          }

          if ( vrbflg ) fprintf( stdout, "removing inserted fine templates\n" );

          while ( tmpltCurrent->tmpltPtr->level == currentLevel )
          {
            LAL_CALL( LALFindChirpDestroyTmpltNode( &status, &tmpltCurrent ),
                &status );
          }          
        } /* end if up a level */

      } /* end loop over data segments */

      /* delete the chisq bins for templates */
      switch ( approximant )
      {
        case TaylorT1:
        case TaylorT2:
        case TaylorT3:
        case GeneratePPN:
        case PadeT1:
        case EOB:
	case FindChirpSP:
          /* the chisq bins need to be re-computed for the next template */
          for ( i = 0; i < fcSegVec->length ; ++i )
          {
            if ( fcSegVec->data[i].chisqBinVec->data )
            {
              LALFree( fcSegVec->data[i].chisqBinVec->data );
              fcSegVec->data[i].chisqBinVec->data = NULL;
            }
          }
        default:
          break;
      }

    } /* end loop over linked list */

    if ( bankSim  )
    {
      if ( bankSimParams.maxMatch )
      {
        /* compute the best match over the template bank */
        if ( ! siminstparams.simInstParamsTable )
          thisSimInstParams = siminstparams.simInstParamsTable =
            XLALFindChirpBankSimMaxMatch( &(savedEvents.snglInspiralTable), 
                matchNorm );
        else
          thisSimInstParams = thisSimInstParams->next =
            XLALFindChirpBankSimMaxMatch( &(savedEvents.snglInspiralTable), 
                matchNorm );

        /* save only the template that gives the loudest event */
        if ( ! loudestEventHead )
          thisLoudestEvent = loudestEventHead = 
            savedEvents.snglInspiralTable;
        else
          thisLoudestEvent = thisLoudestEvent->next = 
            savedEvents.snglInspiralTable;
      }
      else
      {
        /* compute the match for all the templates in the bank */
        for ( event = savedEvents.snglInspiralTable; event; 
            event = event->next )
        {
          if ( ! siminstparams.simInstParamsTable )
            thisSimInstParams = siminstparams.simInstParamsTable =
              XLALFindChirpBankSimComputeMatch( event, matchNorm );
          else
            thisSimInstParams = thisSimInstParams->next =
              XLALFindChirpBankSimComputeMatch( event, matchNorm );
        }

        /* append all the inspiral templates to the loudest event list */
        if ( ! loudestEventHead )
          thisLoudestEvent = loudestEventHead = savedEvents.snglInspiralTable;
        else
          thisLoudestEvent = thisLoudestEvent->next = 
            savedEvents.snglInspiralTable;

        while ( thisLoudestEvent && thisLoudestEvent->next )
          thisLoudestEvent = thisLoudestEvent->next;
      }

      /* null out the list of inspiral events */
      savedEvents.snglInspiralTable = NULL;
    }

    /* increment the bank sim sim_inspiral table if necessary */
    if ( bankSimHead )
    {
      thisBankSim = thisBankSim->next;
    }

  } while ( ++bankSimCount < bankSim ); /* end loop over bank simulations */

  if ( bankSim )
  {
    /* save the number of bank simulation in the search summary table */
    searchsumm.searchSummaryTable->nevents = bankSim;

    /* point the saved events to the linked list of loudest events */
    savedEvents.snglInspiralTable = loudestEventHead;
  }
  else
  {
    /* save the number of events in the search summary table */
    searchsumm.searchSummaryTable->nevents = numEvents;
  }


  /*
   *
   * free memory used by filtering code
   *
   */


  if ( vrbflg ) fprintf( stdout, "freeing memory\n" );

  /* free memory used by findchirp */
  LAL_CALL( LALFindChirpChisqVetoFinalize( &status, 
        fcFilterParams->chisqParams, fcInitParams->numChisqBins ), 
      &status );
  LAL_CALL( LALDestroyFindChirpInput( &status, &fcFilterInput ), 
      &status );
  if ( fcFilterParams->filterOutputVetoParams ) 
  {
    LALFree( fcFilterParams->filterOutputVetoParams );
  }
  LAL_CALL( LALFindChirpFilterFinalize( &status, &fcFilterParams ), 
      &status );
  LAL_CALL( LALFindChirpTemplateFinalize( &status, &fcTmpltParams ), 
      &status );
  LAL_CALL( LALFindChirpDataFinalize( &status, &fcDataParams ),
      &status );
  LAL_CALL( LALDestroyFindChirpSegmentVector( &status, &fcSegVec ),
      &status );
  LALFree( fcInitParams );

  /* free the template bank */
  while ( bankHead )
  {
    bankCurrent = bankHead;
    bankHead = bankHead->next;
    if ( bankCurrent->event_id )
    {
      LALFree( bankCurrent->event_id );
    }
    LALFree( bankCurrent );
    bankCurrent = NULL;
  }

  /* destroy linked list of template nodes */
  while ( tmpltHead )
  {
    LAL_CALL( LALFindChirpDestroyTmpltNode( &status, &tmpltHead ), &status );
  }

  /* free any bank sim parameters */
  if ( bankSimHead )
  {
    while ( bankSimHead )
    {
      thisBankSim = bankSimHead;
      bankSimHead = bankSimHead->next;
      LALFree( thisBankSim );
    }
  }

  /* free the data storage */
  LAL_CALL( LALFinalizeDataSegmentVector( &status, &dataSegVec ), &status );
  LAL_CALL( LALSDestroyVector( &status, &(chan.data) ), &status );
  LAL_CALL( LALSDestroyVector( &status, &(spec.data) ), &status );
  LAL_CALL( LALCDestroyVector( &status, &(resp.data) ), &status );

  /* free the random parameters structure */
  if ( randSeedType != unset )
  {
    LAL_CALL( LALDestroyRandomParams( &status, &randParams ), &status );
  }


  /*
   *
   * write the result results to disk
   *
   */


  /* write the output frame */
  if ( writeRawData || writeFilterData || writeResponse || writeSpectrum ||
      writeRhosq || writeChisq || (writeCData && cDataForFrame) )
  {
    if ( outputPath[0] )
    {
      LALSnprintf( fname, FILENAME_MAX * sizeof(CHAR), "%s/%s.gwf", 
          outputPath, fileName );
    }
    else
    {
      LALSnprintf( fname, FILENAME_MAX * sizeof(CHAR), "%s.gwf", fileName );
    }
    if ( vrbflg ) fprintf( stdout, "writing frame data to %s... ", fname );
    frOutFile = FrFileONew( fname, 0 );
    if ( writeRawData || writeFilterData || writeResponse || writeSpectrum ||
        writeRhosq || writeChisq )
    {
      FrameWrite( outFrame, frOutFile );
    }
    while( coherentFrames )
    {
      thisCoherentFrame = coherentFrames;
      FrameWrite( coherentFrames->frHeader, frOutFile );
      coherentFrames = coherentFrames->next;
      LALFree( thisCoherentFrame );
      thisCoherentFrame = NULL;
    }
    FrFileOEnd( frOutFile );
    if ( vrbflg ) fprintf( stdout, "done\n" );
  }

  /* open the output xml file */
  memset( &results, 0, sizeof(LIGOLwXMLStream) );
  if ( outputPath[0] )
  {
    LALSnprintf( fname, FILENAME_MAX * sizeof(CHAR), "%s/%s.xml", 
        outputPath, fileName );
  }
  else
  {
    LALSnprintf( fname, FILENAME_MAX * sizeof(CHAR), "%s.xml", fileName );
  }
  if ( vrbflg ) fprintf( stdout, "writing XML data to %s...\n", fname );
  LAL_CALL( LALOpenLIGOLwXMLFile( &status, &results, fname), &status );

  /* write the process table */
  if ( vrbflg ) fprintf( stdout, "  process table...\n" );
  LALSnprintf( proctable.processTable->ifos, LIGOMETA_IFOS_MAX, "%s", ifo );
  LAL_CALL( LALGPSTimeNow ( &status, &(proctable.processTable->end_time),
        &accuracy ), &status );
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, process_table ), 
      &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, proctable, 
        process_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );
  free( proctable.processTable );

  /* write the process params table */
  if ( vrbflg ) fprintf( stdout, "  process_params table...\n" );
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, process_params_table ), 
      &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, procparams, 
        process_params_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );
  while( procparams.processParamsTable )
  {
    this_proc_param = procparams.processParamsTable;
    procparams.processParamsTable = this_proc_param->next;
    free( this_proc_param );
  }

  /* write the search summary table */
  if ( vrbflg ) fprintf( stdout, "  search_summary table...\n" );
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, 
        search_summary_table ), &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, searchsumm, 
        search_summary_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );

  /* write the filter table */
  if ( vrbflg ) fprintf( stdout, "  filter table...\n" );
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, 
        filter_table ), &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, filtertable, 
        filter_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );
  free( filtertable.filterTable );

  /* write the search summvars table */
  if ( numTmplts )
  {
    if ( vrbflg ) fprintf( stdout, "  search_summvars table...\n" );
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, 
          search_summvars_table ), &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, searchsummvars, 
          search_summvars_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );
  }
  while( searchsummvars.searchSummvarsTable )
  {
    this_search_summvar = searchsummvars.searchSummvarsTable;
    searchsummvars.searchSummvarsTable = this_search_summvar->next;
    LALFree( this_search_summvar );
  }

  /* write the summ_value table with the standard candle distance */
  if ( ! bankSim )
  {
    if ( approximant == FindChirpSP )
    {
      if ( vrbflg ) fprintf( stdout, "  summ_value table...\n" );
      ADD_SUMM_VALUE( "inspiral_effective_distance", "1.4_1.4_8", 
          candle.distance, 0);
    }
    else if ( approximant == BCV )
    {
      if ( vrbflg ) fprintf( stdout, "  summ_value table...\n" );
      ADD_SUMM_VALUE( "inspiral_effective_distance", "5.0_5.0_8",
          candle.distance, 0);
    }
  }

  /* store calibration information */
  ADD_SUMM_VALUE( "calibration alpha", "analysis", alpha, 0 );
  ADD_SUMM_VALUE( "calibration alphabeta", "analysis", alphabeta, 0 );
  if (injectionFile) 
  {
    ADD_SUMM_VALUE( "calibration alpha", "injection", inj_alpha, 0 );
    ADD_SUMM_VALUE( "calibration alphabeta", "injection", inj_alphabeta, 0 );
  }
  LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, summ_value_table ), 
      &status );
  LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, summvalue, 
        summ_value_table ), &status );
  LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );

  while ( summvalue.summValueTable )
  {
    this_summ_value = summvalue.summValueTable;
    summvalue.summValueTable = summvalue.summValueTable->next;
    LALFree( this_summ_value );
  }

  /* free the search summary table after the summ_value table is written */
  free( searchsumm.searchSummaryTable );

  /* write the sngl_inspiral triggers to the output xml */
  if ( savedEvents.snglInspiralTable )
  {
    SnglInspiralTable *tmpEventHead = NULL;
    SnglInspiralTable *lastEvent = NULL;

    /* sort the inspiral events by time */
    if ( vrbflg ) fprintf( stdout, "  sorting events by time... " );
    LAL_CALL( LALSortSnglInspiral( &status, &(savedEvents.snglInspiralTable),
          LALCompareSnglInspiralByTime), &status );
    if ( vrbflg ) fprintf( stdout, "done\n" );

    /* discard any triggers outside the trig start/end time window */
    event = savedEvents.snglInspiralTable;
    if ( ! bankSim && ( trigStartTimeNS || trigEndTimeNS ) )
    {
      if ( vrbflg ) fprintf( stdout, 
          "  discarding triggers outside trig start/end time... " );

      while ( event )
      {
        INT8 trigTimeNS;
        LAL_CALL( LALGPStoINT8( &status, &trigTimeNS, &(event->end_time) ), 
            &status );

        if ( trigTimeNS &&
            ((trigStartTimeNS && (trigTimeNS < trigStartTimeNS)) ||
             (trigEndTimeNS && (trigTimeNS >= trigEndTimeNS))) )
        {
          /* throw this trigger away */
          SnglInspiralTable *tmpEvent = event;

          if ( lastEvent )
          {
            lastEvent->next = event->next;
          }

          /* increment the linked list by one and free the event */
          event = event->next;
          LALFree( tmpEvent );
        }
        else 
        {
          /* store the first event as the head of the new linked list */
          if ( ! tmpEventHead ) tmpEventHead = event;

          /* save the last event and increment the linked list by one */
          lastEvent = event;
          event = event->next;
        }
      }

      savedEvents.snglInspiralTable = tmpEventHead;

      if ( vrbflg ) fprintf( stdout, "done\n" );
    }

    if (maximizationInterval)
    {
      XLALMaxSnglInspiralOverIntervals( &(savedEvents.snglInspiralTable),
          maximizationInterval);
    }

    /* trigScanClustering */ 
    if ( trigScanMethod ) 
    { 
        if ( condenseIn && (savedEvents.snglInspiralTable) ) 
        { 
            condenseIn->bin_time   = trigScanDeltaEndTime; 
            condenseIn->ts_scaling = trigScanMetricScalingFac;
            condenseIn->scanMethod = trigScanMethod; 
            condenseIn->n          = XLALCountSnglInspiral ( (savedEvents.snglInspiralTable) ); 
            condenseIn->vrbflag    = vrbflg;
            condenseIn->appendStragglers = trigScanAppendStragglers; 
            
            /* Call the clustering routine */ 
            LAL_CALL( LALClusterSnglInspiralOverTemplatesAndEndTime ( &status, 
                        &(savedEvents.snglInspiralTable), condenseIn ), &status );
        }
        else
        {
            if ( vrbflg )
                  fprintf (stderr, 
                          "The event head appears to be null containing %d triggers \n", 
                          XLALCountSnglInspiral ( (savedEvents.snglInspiralTable) ));
        }
    }

    /* if we haven't thrown all the triggers away, write sngl_inspiral table */
    if ( savedEvents.snglInspiralTable )
    {
      if ( vrbflg ) fprintf( stdout, "  sngl_inspiral table...\n" );
      LAL_CALL( LALBeginLIGOLwXMLTable( &status, 
            &results, outputMask ), &status );
      LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, savedEvents, 
            outputMask ), &status );
      LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );
    }
  }
  
  while ( savedEvents.snglInspiralTable )
  {
    event = savedEvents.snglInspiralTable;
    savedEvents.snglInspiralTable = savedEvents.snglInspiralTable->next;
    LALFree( event );
  }

  /* write the signals injected in a template bank simulation */
  if ( bankSim && siminspiral.simInspiralTable )
  {
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, &results, sim_inspiral_table ), 
        &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, siminspiral, 
          sim_inspiral_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );

    while ( siminspiral.simInspiralTable )
    {
      thisSimInspiral = siminspiral.simInspiralTable;
      siminspiral.simInspiralTable = siminspiral.simInspiralTable->next;
      LALFree( thisSimInspiral );
    }
  }

  /* write the template bank simulation results to the xml */
  if ( bankSim && siminstparams.simInstParamsTable )
  {
    if ( vrbflg ) fprintf( stdout, "  sim_inst table...\n" );
    LAL_CALL( LALBeginLIGOLwXMLTable( &status, 
          &results, sim_inst_params_table ), &status );
    LAL_CALL( LALWriteLIGOLwXMLTable( &status, &results, siminstparams, 
          sim_inst_params_table ), &status );
    LAL_CALL( LALEndLIGOLwXMLTable ( &status, &results ), &status );

    while ( siminstparams.simInstParamsTable )
    {
      thisSimInstParams = siminstparams.simInstParamsTable;
      siminstparams.simInstParamsTable = siminstparams.simInstParamsTable->next;
      LALFree( thisSimInstParams );
    }
  }

  /* close the output xml file */
  LAL_CALL( LALCloseLIGOLwXMLFile ( &status, &results ), &status );
  if ( vrbflg ) fprintf( stdout, "done. XML file closed\n" );

  /* free the rest of the memory, check for memory leaks and exit */
  if ( tdFollowUpFiles )
  {
    free ( tdFollowUpFiles );
    while ( tdFollowUpEvents )
    {
      thisFollowUpEvent = tdFollowUpEvents;
      tdFollowUpEvents = tdFollowUpEvents->next;
      XLALFreeSnglInspiral( &thisFollowUpEvent );
    }
  }

  if ( injectionFile ) 
  {
    free ( injectionFile );     
    while ( injections )
    {
      thisInj = injections;
      injections = injections->next;
      LALFree( thisInj );
    }
  }
  if ( condenseIn )  LALFree( condenseIn );
  if ( approximantName) free( approximantName );
  if ( calCacheName )  free( calCacheName );
  if ( frInCacheName ) free( frInCacheName );
  if ( frInType )      free( frInType );
  if ( bankFileName )  free( bankFileName );
  if ( channelName )   free( channelName );
  if ( fqChanName )    free( fqChanName );

  /* Free the analyseThisTmplt vector */
  if ( analyseThisTmplt) LALFree (analyseThisTmplt);

  if ( vrbflg ) fprintf( stdout, "checking memory leaks and exiting\n" );
  LALCheckMemoryLeaks();

  /* print a success message to stdout for parsing by exitcode */
  fprintf( stdout, "%s: EXITCODE0\n", argv[0] );
  exit( 0 );
}

/* ------------------------------------------------------------------------- */

#define ADD_PROCESS_PARAM( pptype, format, ppvalue ) \
  this_proc_param = this_proc_param->next = (ProcessParamsTable *) \
calloc( 1, sizeof(ProcessParamsTable) ); \
LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", \
    PROGRAM_NAME ); \
LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "--%s", \
    long_options[option_index].name ); \
LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "%s", pptype ); \
LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, format, ppvalue );

#define USAGE \
  "lalapps_inspiral [options]\n\n"\
"  --help                       display this message\n"\
"  --verbose                    print progress information\n"\
"  --version                    print version information and exit\n"\
"  --debug-level LEVEL          set the LAL debug level to LEVEL\n"\
"  --user-tag STRING            set the process_params usertag to STRING\n"\
"  --ifo-tag STRING             set the ifotag to STRING - for file naming\n"\
"  --comment STRING             set the process table comment to STRING\n"\
"\n"\
"  --gps-start-time SEC         GPS second of data start time\n"\
"  --gps-start-time-ns NS       GPS nanosecond of data start time\n"\
"  --gps-end-time SEC           GPS second of data end time\n"\
"  --gps-end-time-ns NS         GPS nanosecond of data end time\n"\
"  --pad-data T                 pad the data start and end time by T seconds\n"\
"  --slide-time T               slide data start epoch by T seconds\n"\
"  --slide-time-ns T            slide data start epoch by T nanoseconds\n"\
"\n"\
"  --glob-frame-data            glob *.gwf files in the pwd to obtain frame data\n"\
"  --frame-type TAG             input data is contained in frames of type TAG\n"\
"  --frame-cache                obtain frame data from LAL frame cache FILE\n"\
"  --calibration-cache FILE     obtain calibration from LAL frame cache FILE\n"\
"  --glob-calibration-data      obtain calibration by globbing in working dir\n"\
"\n"\
"  --channel-name CHAN          read data from interferometer channel CHAN\n"\
"  --calibrated-data TYPE       calibrated data of TYPE real_4 or real_8\n"\
"  --strain-high-pass-freq F    high pass REAL8 h(t) data above F Hz\n"\
"  --strain-high-pass-order O   set the order of the h(t) high pass filter to O\n"\
"  --strain-high-pass-atten A   set the attenuation of the high pass filter to A\n"\
"  --point-calibration          use the first point in the chunk to calibrate\n"\
"\n"\
"  --injection-file FILE        inject simulated inspiral signals from FILE\n"\
"  --fast F                     analyse injections templates within a match F\n"\
"  --inject-overhead            inject signals from overhead detector\n"\
"  --enable-filter-inj-only     filter only segments with injections\n"\
"  --disable-filter-inj-only    filter all segments when doing injections\n"\
"                               All segments are filtered.\n"\
"\n"\
"  --td-follow-up FILE          Follow up coincident BCV events in FILE\n"\
"  --bank-file FILE             read template bank parameters from FILE\n"\
"  --start-template N           start filtering at template number N in bank\n"\
"  --stop-template N            stop filtering at template number N in bank\n"\
"  --reverse-chirp-bank         filters data using a reverse chirp template bank\n"\
"\n"\
"  --sample-rate F              filter data at F Hz, downsampling if necessary\n"\
"  --resample-filter TYPE       set resample filter to TYPE (ldas|butterworth)\n"\
"\n"\
"  --disable-high-pass          turn off the IIR highpass filter\n"\
"  --enable-high-pass F         high pass data above F Hz using an IIR filter\n"\
"  --high-pass-order O          set the order of the high pass filter to O\n"\
"  --high-pass-attenuation A    set the attenuation of the high pass filter to A\n"\
"  --spectrum-type TYPE         use PSD estimator TYPE\n"\
"                                 (mean|median|gaussian|LIGO|AdvLIGO)\n"\
"\n"\
"  --segment-length N           set data segment length to N points\n"\
"  --number-of-segments N       set number of data segments to N\n"\
"  --segment-overlap N          overlap data segments by N points\n"\
"\n"\
"  --low-frequency-cutoff F     do not filter below F Hz\n"\
"  --inverse-spec-length T      set length of inverse spectrum to T seconds\n"\
"  --dynamic-range-exponent X   set dynamic range scaling to 2^X\n"\
"\n"\
"  --approximant APPROX         set approximant of the waveform to APPROX\n"\
"                               (FindChirpSP|BCV|BCVC|BCVSpin|TaylorT1|TaylorT2|\n"\
"                                  TaylorT3|PadeT1|EOB|GeneratePPN)\n"\
"  --snr-threshold RHO          set signal-to-noise threshold to RHO\n"\
"  --chisq-bins P               set number of chisq veto bins to P\n"\
"  --chisq-delta DELTA          set chisq delta parameter to DELTA\n"\
"  --chisq-threshold X          threshold on chi^2 < X * ( p + DELTA *rho^2 )\n"\
"  --cluster-method MTHD        max over chirp MTHD (tmplt|window|none)\n"\
"  --cluster-window SEC         set length of clustering time window if required\n"\
"\n"\
"  --enable-rsq-veto            enable the r^2 veto test\n"\
"  --disable-rsq-veto           disable the r^2 veto test\n"\
"  --rsq-veto-window SEC        set the r^2 veto window to SEC\n"\
"  --rsq-veto-threshold RSQ     set r^2 veto threshold to RSQ\n"\
"\n"\
"  --do-rsq-veto                do the r^2 veto\n"\
"  --rsq-veto-time-thresh SEC   set the r^2 veto window to SEC\n"\
"  --rsq-veto-max-snr MAXSNR    set the r^2 veto maximum snr to MAXSNR\n"\
"  --rsq-veto-coeff COEFF       set the r^2 veto coefficient to COEFF\n"\
"  --rsq-veto-pow POW           set the r^2 veto power to POW\n"\
"\n"\
"  --maximization-interval msec set length of maximization interval\n"\
"\n"\
"  --ts-cluster   MTHD          max over template and end time MTHD \n"\
"                                 (T0T3Tc|T0T3TcAS|Psi0Psi3Tc|Psi0Psi3TcAS)\n"\
"  --ts-endtime-interval msec   set end-time interval for TrigScan clustering\n"\
"  --ts-metric-scaling fac      scale the metric which defines the ellipsoids for TrigScan\n"\
"                               Scaling must be > 0\n"\
"\n"\
"  --enable-output              write the results to a LIGO LW XML file\n"\
"  --output-mask MASK           write the output sngl_inspiral table\n"\
"                                 with optional MASK (bns|bcv)\n"\
"  --disable-output             do not write LIGO LW XML output file\n"\
"  --trig-start-time SEC        only output triggers after GPS time SEC\n"\
"  --trig-end-time SEC          only output triggers before GPS time SEC\n"\
"\n"\
"  --gaussian-noise VAR         replace data with gaussian noise of variance VAR\n"\
"  --random-seed SEED           set random number seed for injections to SEED\n"\
"                                 (urandom|integer)\n"\
"\n"\
"  --bank-simulation N          perform N injections to test the template bank\n"\
"                                (sim_inspiral.xml|integer)\n"\
"  --enable-bank-sim-max        compute the maximum match over the bank\n"\
"  --disable-bank-sim-max       do not maximize the match over the bank\n"\
"  --sim-approximant APX        set approximant of the injected waveform to APX\n"\
"                                 (TaylorT1|TaylorT2|TaylorT3|PadeT1|EOB|\n"\
"                                  GeneratePPN|FrameFile)\n"\
"  --sim-frame-file F           read the bank sim waveform from frame named F\n"\
"  --sim-frame-channel C        read the bank sim waveform from frame channel C\n"\
"  --sim-minimum-mass M         set minimum mass of bank injected signal to M\n"\
"  --sim-maximum-mass M         set maximum mass of bank injected signal to M\n"\
"\n"\
"  --data-checkpoint            checkpoint and exit after data is read in\n"\
"  --checkpoint-path PATH       write checkpoint file under PATH\n"\
"  --output-path PATH           write output data to PATH\n"\
"\n"\
"  --write-raw-data             write raw data to a frame file\n"\
"  --write-filter-data          write data that is passed to filter to a frame\n"\
"  --write-response             write the computed response function to a frame\n"\
"  --write-spectrum             write the uncalibrated psd to a frame\n"\
"  --write-snrsq                write the snr time series for each data segment\n"\
"  --write-chisq                write the r^2 time series for each data segment\n"\
"  --write-cdata                write the complex filter output\n"\
"\n"

int arg_parse_check( int argc, char *argv[], MetadataTable procparams )
{
  /* getopt arguments */
  struct option long_options[] =
  {
    /* these options set a flag */
    {"verbose",                 no_argument,       &vrbflg,           1 },
    {"enable-output",           no_argument,       &enableOutput,     1 },
    {"disable-output",          no_argument,       &enableOutput,     0 },
    {"disable-high-pass",       no_argument,       &highPass,         0 },
    {"inject-overhead",         no_argument,       &injectOverhead,   1 },
    {"data-checkpoint",         no_argument,       &dataCheckpoint,   1 },
    {"glob-frame-data",         no_argument,       &globFrameData,    1 },
    {"glob-calibration-data",   no_argument,       &globCalData,      1 },
    {"point-calibration",       no_argument,       &pointCal,         1 },
    {"enable-rsq-veto",         no_argument,       &enableRsqVeto,    1 },
    {"disable-rsq-veto",        no_argument,       &enableRsqVeto,    0 },
    {"enable-filter-inj-only",  no_argument,       &flagFilterInjOnly,1 },
    {"disable-filter-inj-only", no_argument,       &flagFilterInjOnly,0 },
    {"reverse-chirp-bank",      no_argument,       &reverseChirpBank, 1 },
    {"do-rsq-veto",             no_argument,       &doRsqVeto,        1 },
    /* these options don't set a flag */
    {"gps-start-time",          required_argument, 0,                'a'},
    {"gps-start-time-ns",       required_argument, 0,                'A'},
    {"gps-end-time",            required_argument, 0,                'b'},
    {"gps-end-time-ns",         required_argument, 0,                'B'},
    {"channel-name",            required_argument, 0,                'c'},
    {"segment-length",          required_argument, 0,                'd'},
    {"trig-start-time",         required_argument, 0,                'C'},
    {"trig-end-time",           required_argument, 0,                'D'},
    {"number-of-segments",      required_argument, 0,                'e'},
    {"segment-overlap",         required_argument, 0,                'f'},
    {"sample-rate",             required_argument, 0,                'g'},
    {"calibrated-data",         required_argument, 0,                'y'},
    {"strain-high-pass-freq",   required_argument, 0,                'E'},
    {"strain-high-pass-order",  required_argument, 0,                'P'},
    {"strain-high-pass-atten",  required_argument, 0,                'Q'},
    {"help",                    no_argument,       0,                'h'},
    {"low-frequency-cutoff",    required_argument, 0,                'i'},
    {"spectrum-type",           required_argument, 0,                'j'},
    {"inverse-spec-length",     required_argument, 0,                'k'},
    {"dynamic-range-exponent",  required_argument, 0,                'l'},
    {"start-template",          required_argument, 0,                'm'},
    {"chisq-delta",             required_argument, 0,                'M'},
    {"stop-template",           required_argument, 0,                'n'},
    {"chisq-bins",              required_argument, 0,                'o'},
    {"calibration-cache",       required_argument, 0,                'p'},
    {"approximant",             required_argument, 0,                'F'},
    {"snr-threshold",           required_argument, 0,                'q'},
    {"chisq-threshold",         required_argument, 0,                'r'},
    {"resample-filter",         required_argument, 0,                'R'},
    {"comment",                 required_argument, 0,                's'},
    {"enable-high-pass",        required_argument, 0,                't'},
    {"high-pass-order",         required_argument, 0,                'H'},
    {"high-pass-attenuation",   required_argument, 0,                'T'},
    {"frame-cache",             required_argument, 0,                'u'},
    {"frame-type",              required_argument, 0,                'S'},
    {"bank-file",               required_argument, 0,                'v'},
    {"injection-file",          required_argument, 0,                'w'},
    {"pad-data",                required_argument, 0,                'x'},
    {"slide-time",              required_argument, 0,                'X'},
    {"slide-time-ns",           required_argument, 0,                'Y'},
    {"bank-simulation",         required_argument, 0,                'K'},
    {"sim-approximant",         required_argument, 0,                'L'},
    {"random-seed",             required_argument, 0,                'J'},
    {"sim-minimum-mass",        required_argument, 0,                'U'},
    {"sim-maximum-mass",        required_argument, 0,                'W'},
    {"gaussian-noise",          required_argument, 0,                'G'},
    {"checkpoint-path",         required_argument, 0,                'N'},
    {"output-path",             required_argument, 0,                'O'},
    {"debug-level",             required_argument, 0,                'z'},
    {"user-tag",                required_argument, 0,                'Z'},
    {"userTag",                 required_argument, 0,                'Z'},
    {"ifo-tag",                 required_argument, 0,                'I'},
    {"version",                 no_argument,       0,                'V'},
    {"maximization-interval",   required_argument, 0,                '@'},  
    {"cluster-window",          required_argument, 0,                '#'},  
    {"cluster-method",          required_argument, 0,                '0'},  
    {"output-mask",             required_argument, 0,                '1'},  
    {"fast",                    required_argument, 0,                '2'},  
    {"rsq-veto-window",         required_argument, 0,                '3'},   
    {"rsq-veto-threshold",      required_argument, 0,                '4'},  
    {"enable-bank-sim-max",     no_argument,       0,                '5'},  
    {"disable-bank-sim-max",    no_argument,       0,                '6'},  
    {"sim-frame-file",          required_argument, 0,                '7'},
    {"sim-frame-channel",       required_argument, 0,                '8'},
    {"td-follow-up",            required_argument, 0,                '9'},
    {"ts-cluster",              required_argument, 0,                '*'},
    {"ts-endtime-interval",     required_argument, 0,                '<'},
    {"ts-metric-scaling",       required_argument, 0,                '>'},
    {"rsq-veto-time-thresh",    required_argument, 0,                '('},
    {"rsq-veto-max-snr",        required_argument, 0,                ')'},
    {"rsq-veto-coeff",          required_argument, 0,                '['},
    {"rsq-veto-pow",            required_argument, 0,                ']'},
    /* frame writing options */
    {"write-raw-data",          no_argument,       &writeRawData,     1 },
    {"write-filter-data",       no_argument,       &writeFilterData,  1 },
    {"write-response",          no_argument,       &writeResponse,    1 },
    {"write-spectrum",          no_argument,       &writeSpectrum,    1 },
    {"write-snrsq",             no_argument,       &writeRhosq,       1 },
    {"write-chisq",             no_argument,       &writeChisq,       1 },
    {"write-cdata",             no_argument,       &writeCData,       1 },
    {0, 0, 0, 0}
  };
  int c;
  INT4 haveDynRange = 0;
  INT4 haveApprox = 0;
  INT4 haveClusterMethod = 0;
  INT4 haveBankSimApprox = 0;
  ProcessParamsTable *this_proc_param = procparams.processParamsTable;
  LALStatus             status = blank_status;


  /*
   *
   * parse command line arguments
   *
   */


  while ( 1 )
  {
    /* getopt_long stores long option here */
    int option_index = 0;
    size_t optarg_len;

    c = getopt_long_only( argc, argv, 
        "-A:B:C:D:E:F:G:H:I:J:K:L:M:N:O:P:Q:R:S:T:U:VW:X:Y:Z:"
        "a:b:c:d:e:f:g:hi:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:"
        "0:1::2:3:4:567:8:9:*:>:<:(:):[:]",
        long_options, &option_index );

    /* detect the end of the options */
    if ( c == - 1 )
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
          fprintf( stderr, "error parsing option %s with argument %s\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        break;

      case 'a':
        {
          long int gstartt = atol( optarg );
          if ( gstartt < 441417609 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS start time is prior to " 
                "Jan 01, 1994  00:00:00 UTC:\n"
                "(%ld specified)\n",
                long_options[option_index].name, gstartt );
            exit( 1 );
          }
          if ( gstartt > 999999999 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS start time is after " 
                "Sep 14, 2011  01:46:26 UTC:\n"
                "(%ld specified)\n", 
                long_options[option_index].name, gstartt );
            exit( 1 );
          }
          gpsStartTimeNS += (INT8) gstartt * 1000000000LL;
          ADD_PROCESS_PARAM( "int", "%ld", gstartt );
        }
        break;

      case 'A':
        {
          long int gstarttns = atol( optarg );
          if ( gstarttns < 0 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS start time nanoseconds is negative\n",
                long_options[option_index].name );
            exit( 1 );
          }
          if ( gstarttns > 999999999 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS start time nanoseconds is greater than unity:\n" 
                "Must be <= 999999999 (%ld specified)\n", 
                long_options[option_index].name, gstarttns );
            exit( 1 );
          }
          gpsStartTimeNS += (INT8) gstarttns;
          ADD_PROCESS_PARAM( "int", "%ld", gstarttns );
        }
        break;

      case 'b':
        {
          long int gendt = atol( optarg );
          if ( gendt > 999999999 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS end time is after " 
                "Sep 14, 2011  01:46:26 UTC:\n"
                "(%ld specified)\n", 
                long_options[option_index].name, gendt );
            exit( 1 );
          }
          else if ( gendt < 441417609 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS end time is prior to " 
                "Jan 01, 1994  00:00:00 UTC:\n"
                "(%ld specified)\n", 
                long_options[option_index].name, gendt );
            exit( 1 );
          }            
          gpsEndTimeNS += (INT8) gendt * 1000000000LL;
          ADD_PROCESS_PARAM( "int", "%ld", gendt );
        }
        break;

      case 'B':
        {
          long int gendtns = atol( optarg );
          if ( gendtns < 0 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS end time nanoseconds is negative\n",
                long_options[option_index].name );
            exit( 1 );
          }
          else if ( gendtns > 999999999 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "GPS end time nanoseconds is greater than unity:\n" 
                "Must be <= 999999999:\n"
                "(%ld specified)\n", 
                long_options[option_index].name, gendtns );
            exit( 1 );
          }            
          gpsEndTimeNS += (INT8) gendtns;
          ADD_PROCESS_PARAM( "int", "%ld", gendtns );
        }
        break;

      case 'c':
        {
          /* create storage for the channel name and copy it */
          char *channamptr = NULL;
          optarg_len = strlen( optarg ) + 1;
          fqChanName = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
          memcpy( fqChanName, optarg, optarg_len );
          ADD_PROCESS_PARAM( "string", "%s", optarg );

          /* check that we have a proper channel name */
          if ( ! (channamptr = strstr( fqChanName, ":" ) ) )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "channel name must be a full LIGO channel name "
                "e.g. L1:LSC-AS_Q\n(%s specified)\n",
                long_options[option_index].name, optarg );
            exit( 1 );
          }
          optarg_len = strlen( ++channamptr ) + 1;
          channelName = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
          memcpy( channelName, channamptr, optarg_len );

          /* copy the first two characters to the ifo name */
          memset( ifo, 0, sizeof(ifo) );
          memcpy( ifo, optarg, sizeof(ifo) - 1 );
        }
        break;

      case 'd':
        numPoints = (INT4) atoi( optarg );
        if ( numPoints < 2 || numPoints % 2 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "number of points must be a non-zero power of 2: "
              "(%d specified) \n", 
              long_options[option_index].name, numPoints );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", numPoints );
        break;

      case 'C':
        {
          long int gstartt = atol( optarg );
          /* ignore a value of zero */
          if ( gstartt )
          {
            if ( gstartt < 441417609 )
            {
              fprintf( stderr, "invalid argument to --%s:\n"
                  "GPS start time is prior to " 
                  "Jan 01, 1994  00:00:00 UTC:\n"
                  "(%ld specified)\n",
                  long_options[option_index].name, gstartt );
              exit( 1 );
            }
            if ( gstartt > 999999999 )
            {
              fprintf( stderr, "invalid argument to --%s:\n"
                  "GPS start time is after " 
                  "Sep 14, 2011  01:46:26 UTC:\n"
                  "(%ld specified)\n", 
                  long_options[option_index].name, gstartt );
              exit( 1 );
            }
            trigStartTimeNS = (INT8) gstartt * 1000000000LL;
          }
          ADD_PROCESS_PARAM( "int", "%ld", gstartt );
        }
        break;

      case 'D':
        {
          long int gendt = atol( optarg );
          /* ignore a value of zero */
          if ( gendt )
          {
            if ( gendt > 999999999 )
            {
              fprintf( stderr, "invalid argument to --%s:\n"
                  "GPS end time is after " 
                  "Sep 14, 2011  01:46:26 UTC:\n"
                  "(%ld specified)\n", 
                  long_options[option_index].name, gendt );
              exit( 1 );
            }
            else if ( gendt < 441417609 )
            {
              fprintf( stderr, "invalid argument to --%s:\n"
                  "GPS end time is prior to " 
                  "Jan 01, 1994  00:00:00 UTC:\n"
                  "(%ld specified)\n", 
                  long_options[option_index].name, gendt );
              exit( 1 );
            }            
            trigEndTimeNS = (INT8) gendt * 1000000000LL;
          }
          ADD_PROCESS_PARAM( "int", "%ld", gendt );
        }
        break;

      case 'e':
        numSegments = (INT4) atoi( optarg );
        if ( numSegments < 1 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "number of data segment must be greater than 0: "
              "(%d specified)\n", 
              long_options[option_index].name, numSegments );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", numSegments );
        break;

      case 'f':
        ovrlap = (INT4) atoi( optarg );
        if ( ovrlap < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "data segment overlap must be positive: "
              "(%d specified)\n", 
              long_options[option_index].name, ovrlap );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", ovrlap );
        break;

      case 'g':
        sampleRate = (INT4) atoi( optarg );
        if ( sampleRate < 2 || sampleRate > 16384 || sampleRate % 2 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "rate must be power of 2 between 2 and 16384 inclusive: "
              "(%d specified)\n", 
              long_options[option_index].name, sampleRate );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", sampleRate );
        break;

      case 'y':
        /* specify which type of calibrated data */
        {
          if ( ! strcmp( "real_4", optarg ) )
          {
            calData = real_4;
          }
          else if ( ! strcmp( "real_8", optarg ) )
          {
            calData = real_8;
          }
          else
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "unknown data type specified;\n"
                "%s (must be one of: real_4, real_8)\n",
                long_options[option_index].name, optarg);
          }
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        break;

      case 'E':
        strainHighPassFreq = (REAL4) atof( optarg );
        if ( strainHighPassFreq <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "REAL8 h(t) high pass filter frequency must be greater than 0 Hz:"
              "(%f Hz specified)\n",
              long_options[option_index].name, strainHighPassFreq );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", strainHighPassFreq );
        break;

      case 'P':
        strainHighPassOrder = (INT4) atoi( optarg );
        if ( strainHighPassOrder <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "REAL8 h(t) high pass filter order must be greater than 0: "
              "(%d specified)\n",
              long_options[option_index].name, strainHighPassOrder );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", strainHighPassOrder );
        break;

      case 'Q':
        strainHighPassAtten = (REAL4) atof( optarg );
        if ( strainHighPassAtten < 0.0 || strainHighPassAtten > 1.0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "REAL8 h(t) high pass attenuation must be in the range [0:1]: "
              "(%f specified)\n",
              long_options[option_index].name, strainHighPassAtten );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", strainHighPassAtten );
        break;

      case 'h':
        fprintf( stdout, USAGE );
        exit( 0 );
        break;

      case 'i':
        fLow = (REAL4) atof( optarg );
        if ( fLow < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "low frequency cutoff is less than 0 Hz: "
              "(%f Hz specified)\n",
              long_options[option_index].name, fLow );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", fLow );
        break;

      case 'j':
        if ( ! strcmp( "mean", optarg ) )
        {
          specType = 0;
        }
        else if ( ! strcmp( "median", optarg ) )
        {
          specType = 1;
        }
        else if ( ! strcmp( "bad-mean", optarg ) )
        {
          specType = 0;
          badMeanPsd = 1;
        }
        else if ( ! strcmp( "gaussian", optarg ) )
        {
          specType = 2;
          fprintf( stderr,
              "WARNING: replacing psd with white gaussian spectrum\n" );
        }
        else if ( ! strcmp( "LIGO", optarg ) )
        {
          specType = 3;
          unitResponse = 1;
          fprintf( stderr,
              "WARNING: replacing psd with Initial LIGO design spectrum\n"
              "WARNING: replacing response function with unity\n" );
        }
        else if ( ! strcmp( "AdvLIGO", optarg ) )
        {
          specType = 4;
          unitResponse = 1;
          fprintf( stderr,
              "WARNING: replacing psd with Advanced LIGO design spectrum\n"
              "WARNING: replacing response function with unity\n" );
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown power spectrum type: "
              "%s (must be mean, median or gaussian)\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'k':
        invSpecTrunc = (INT4) atoi( optarg );
        if ( invSpecTrunc < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "inverse spectrum length must be positive or zero: "
              "(%d specified)\n", 
              long_options[option_index].name, invSpecTrunc );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", invSpecTrunc );
        break;

      case 'l':
        dynRangeExponent = (REAL4) atof( optarg );
        haveDynRange = 1;
        ADD_PROCESS_PARAM( "float", "%e", dynRangeExponent );
        break;

      case 'm':
        startTemplate = (INT4) atoi( optarg );
        if ( startTemplate < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "template bank start index must be positive: "
              "(%d specified)\n", 
              long_options[option_index].name, startTemplate );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", startTemplate );
        break;

      case 'M':
        chisqDelta = (REAL4) atof( optarg );
        if ( chisqDelta < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "chi squared delta parameter must be positive: "          
              "(%e specified)\n", 
              long_options[option_index].name, chisqDelta );
        }
        ADD_PROCESS_PARAM( "float", "%e", chisqDelta );
        break;

      case 'n':
        stopTemplate = (INT4) atoi( optarg );
        if ( stopTemplate < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "template bank stop index must be positive: "
              "(%d specified)\n", 
              long_options[option_index].name, stopTemplate );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", stopTemplate );
        break;

      case 'o':
        numChisqBins = (INT4) atoi( optarg );
        if ( numChisqBins < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "number of chisq veto bins must be positive: "
              "(%d specified)\n", 
              long_options[option_index].name, numChisqBins );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", numChisqBins );
        break;

      case 'p':
        /* create storage for the calibration frame cache name */
        optarg_len = strlen( optarg ) + 1;
        calCacheName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( calCacheName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'F':
        /* create storage for the approximant name */
        optarg_len = strlen( optarg ) + 1;
        approximantName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( approximantName, optarg, optarg_len );
        if ( ! strcmp( "TaylorT1", optarg ) )
        {
          approximant = TaylorT1;
        }
        else if ( ! strcmp( "TaylorT2", optarg ) )
        {
          approximant = TaylorT2;
        }
        else if ( ! strcmp( "TaylorT3", optarg ) )
        {
          approximant = TaylorT3;
        }
        else if ( ! strcmp( "GeneratePPN", optarg ) )
        {
          approximant = GeneratePPN;
        }
        else if ( ! strcmp( "PadeT1", optarg ) )
        {
          approximant = PadeT1;
        }
        else if ( ! strcmp( "EOB", optarg ) )
        {
          approximant = EOB;
        }
        else if ( ! strcmp( "FindChirpSP", optarg ) )
        {
          approximant = FindChirpSP;
        }
        else if ( ! strcmp( "BCV", optarg ) )
        {
          approximant = BCV;
        }
        else if ( ! strcmp( "BCVC", optarg ) )
        {
          approximant = BCV;
          bcvConstraint = 1;
        }
        else if ( ! strcmp( "BCVSpin", optarg ) )
        {
          approximant = BCVSpin;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown order specified: "
              "%s (must be either FindChirpSP or BCV or BCVC or BCVSpin)\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        haveApprox = 1;
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'q':
        snrThresh = atof( optarg );
        if ( snrThresh < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "signal to noise threshold must be positive: "
              "(%f specified)\n", 
              long_options[option_index].name, snrThresh );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%s", optarg );
        break;

      case 'r':
        chisqThresh = atof( optarg );
        if ( chisqThresh < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "chi squared threshold must be positive: "
              "(%f specified)\n", 
              long_options[option_index].name, chisqThresh );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%s", optarg );
        break;

      case 'R':
        if ( ! strcmp( "ldas", optarg ) )
        {
          resampFiltType = 0;
        }
        else if ( ! strcmp( "butterworth", optarg ) )
        {
          resampFiltType = 1;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown resampling filter type: "
              "%s (must be ldas or butterworth)\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 's':
        if ( strlen( optarg ) > LIGOMETA_COMMENT_MAX - 1 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "comment must be less than %d characters\n",
              long_options[option_index].name, LIGOMETA_COMMENT_MAX );
          exit( 1 );
        }
        else
        {
          LALSnprintf( comment, LIGOMETA_COMMENT_MAX, "%s", optarg);
        }
        break;

      case 't':
        highPass = 1;
        highPassFreq = (REAL4) atof( optarg );
        if ( highPassFreq <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "high pass filter frequency must be greater than 0 Hz: "
              "(%f Hz specified)\n",
              long_options[option_index].name, highPassFreq );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", highPassFreq );
        break;

      case 'H':
        highPassOrder = (INT4) atoi( optarg );
        if ( highPassOrder <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "high pass filter order must be greater than 0: "
              "(%d specified)\n",
              long_options[option_index].name, highPassOrder );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", highPassOrder );
        break;

      case 'T':
        highPassAtten = (REAL4) atof( optarg );
        if ( highPassAtten < 0.0 || highPassAtten > 1.0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "high pass attenuation must be in the range [0:1]: "
              "(%f specified)\n",
              long_options[option_index].name, highPassAtten );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", highPassAtten );
        break;

      case 'u':
        /* create storage for the input frame cache name */
        optarg_len = strlen( optarg ) + 1;
        frInCacheName = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
        memcpy( frInCacheName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'S':
        optarg_len = strlen( optarg ) + 1;
        frInType = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
        memcpy( frInType, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'v':
        /* create storage for the calibration frame cache name */
        optarg_len = strlen( optarg ) + 1;
        bankFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( bankFileName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'w':
        /* create storage for the injection file name */
        optarg_len = strlen( optarg ) + 1;
        injectionFile = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( injectionFile, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'x':
        padData = (INT4) atoi( optarg );
        if ( padData < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "number of seconds to pad from input data"
              "must be greater than 0: (%d specified)\n", 
              long_options[option_index].name, padData );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "int", "%d", padData );
        break;

      case 'X':
        slideData.gpsSeconds = (INT4) atoi( optarg );
        ADD_PROCESS_PARAM( "int", "%d", slideData.gpsSeconds );
        break;

      case 'Y':
        slideData.gpsNanoSeconds = (INT4) atoi( optarg );
        ADD_PROCESS_PARAM( "int", "%d", slideData.gpsNanoSeconds );
        break;

      case 'K':
        if ( strstr( optarg, "xml" ) )
        {
          /* we have been passed an xml file name */
          optarg_len = strlen( optarg ) + 1;
          bankSimFileName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
          memcpy( bankSimFileName, optarg, optarg_len );
          bankSim = 1;
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        else
        {
          /* parse the agument as an integer numer of simulations */
          bankSim = (INT4) atoi( optarg );
          if ( bankSim < 1 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "number of template bank simulations"
                "must be greater than 1: (%d specified)\n", 
                long_options[option_index].name, bankSim );
            exit( 1 );
          }
          ADD_PROCESS_PARAM( "int", "%d", bankSim );
        }
        break;

      case 'L':
        if ( ! strcmp( "TaylorT1", optarg ) )
        {
          bankSimParams.approx = TaylorT1;
        }
        else if ( ! strcmp( "TaylorT2", optarg ) )
        {
          bankSimParams.approx = TaylorT2;
        }
        else if ( ! strcmp( "TaylorT3", optarg ) )
        {
          bankSimParams.approx = TaylorT3;
        }
        else if ( ! strcmp( "PadeT1", optarg ) )
        {
          bankSimParams.approx = PadeT1;
        }
        else if ( ! strcmp( "EOB", optarg ) )
        {
          bankSimParams.approx = EOB;
        }
        else if ( ! strcmp( "GeneratePPN", optarg ) )
        {
          bankSimParams.approx = GeneratePPN;
        }
        else if ( ! strcmp( "FrameFile", optarg ) )
        {
          bankSimParams.approx = FrameFile;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown order specified: %s\n(must be one of TaylorT1, "
              "TaylorT2, TaylorT3, PadeT1, EOB, GeneratePPN, FrameFile)\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        haveBankSimApprox = 1;
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'J':
        if ( ! strcmp( "urandom", optarg ) )
        {
          randSeedType = urandom;
          ADD_PROCESS_PARAM( "string", "%s", optarg );
        }
        else
        {
          randSeedType = user;
          randomSeed = (INT4) atoi( optarg );
          ADD_PROCESS_PARAM( "int", "%d", randomSeed );
        }
        break;

      case 'U':
        bankSimParams.minMass = (REAL4) atof( optarg );
        if ( bankSimParams.minMass <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "miniumum component mass must be > 0: "
              "(%f solar masses specified)\n",
              long_options[option_index].name, bankSimParams.minMass );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", bankSimParams.minMass );
        break;

      case 'W':
        bankSimParams.maxMass = (REAL4) atof( optarg );
        if ( bankSimParams.maxMass <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "maxiumum component mass must be > 0: "
              "(%f solar masses specified)\n",
              long_options[option_index].name, bankSimParams.maxMass );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", bankSimParams.maxMass );
        break;

      case 'G':
        gaussVar = (REAL4) atof( optarg );
        if ( gaussVar < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "variance of gaussian noise must be >= 0: "
              "(%f specified)\n",
              long_options[option_index].name, gaussVar );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", gaussVar );
        gaussianNoise = 1;
        unitResponse = 1;
        fprintf( stderr,
            "WARNING: replacing input data with white gaussian noise\n"
            "WARNING: replacing response function with unity\n" );
        break;

      case 'N':
        if ( LALSnprintf( ckptPath, FILENAME_MAX * sizeof(CHAR), 
              "%s", optarg ) < 0 )
        {
          fprintf( stderr, "invalid argument to --%s\n"
              "local path %s too long: string truncated\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );

      case 'O':
        if ( LALSnprintf( outputPath, FILENAME_MAX * sizeof(CHAR), 
              "%s", optarg ) < 0 )
        {
          fprintf( stderr, "invalid argument to --%s\n"
              "output path %s too long: string truncated\n",
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );

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
        LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, "%s", 
            PROGRAM_NAME );
        LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, "-userTag" );
        LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
        LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, "%s",
            optarg );
        break;

      case 'I':
        /* create storaged for the ifo-tag */
        optarg_len = strlen( optarg ) + 1;
        ifoTag = (CHAR *) calloc( optarg_len, sizeof(CHAR) );
        memcpy( ifoTag, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case 'V':
        /* print version information and exit */
        fprintf( stdout, "LIGO/LSC Standalone Inspiral Search Engine\n" 
            "Duncan Brown <duncan@gravity.phys.uwm.edu>\n"
            "CVS Version: " CVS_ID_STRING "\n"
            "CVS Tag: " CVS_NAME_STRING "\n" );
        exit( 0 );
        break;

      case '#':
        clusterWindow = (REAL4) atof( optarg );
        if ( clusterWindow <= 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "clustering time window is less than or equal to 0 secs: "
              "(%f secs specified)\n",
              long_options[option_index].name, clusterWindow );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%e", clusterWindow );
        break;

      case '@':
        {
          long int maxms = atol( optarg );
          if ( maxms < 0 )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "maximization interval is less than 0 msecs: "
                "(%ld msecs specified)\n",
                long_options[option_index].name, maxms );
            exit( 1 );
          }
	  /* internally we require maximizationInterval to be in nano seconds */
 	  /* This will be passed as an argument in the call to                */
          /* XLALMaxSnglInspiralOverIntervals (). Therefore multiply by       */
          /* 1000000 to convert msec to nano seconds                          */ 
          maximizationInterval = (INT4) maxms * 1000000;
          ADD_PROCESS_PARAM( "int", "%ld", maxms );
        }
        break;

      case '0':
        if ( ! strcmp( "none", optarg ) )
        {
          clusterMethod = noClustering;
        }
        else if ( ! strcmp( "template", optarg ) )
        {
          clusterMethod = tmplt;
        } 
        else if ( ! strcmp( "window", optarg ) )
        {
          clusterMethod = window;
        } 
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown clustering method: "
              "%s (must be 'none', 'template' or 'window')\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        haveClusterMethod = 1;
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case '1':
        if ( ! strcmp( "bns", optarg ) )
        {
          outputMask = sngl_inspiral_table_bns;
        }
        else if ( ! strcmp( "bcv", optarg ) )
        {
          outputMask = sngl_inspiral_table_bcv;
        }
        else
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "must be either bns or bcv: (%s specified)\n",
              long_options[option_index].name, optarg);
          exit (1);
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case '2':
        {
          float  mmFastArg = atof (optarg);
          if ( mmFastArg < (float)(0.0) || mmFastArg > (float)(1.0) )
          {
            fprintf( stderr, "invalid argument to --%s:\n"
                "should be between 0.0 and 1.0: (%f specified)\n",
                long_options[option_index].name, mmFastArg);
            exit (1);
          }
          mmFast = (REAL4) mmFastArg;
          ADD_PROCESS_PARAM( "float", "%e", mmFast );
        }
        break;

      case '3': 
        rsqVetoWindow = atol( optarg ); 
        if ( rsqVetoWindow < 0 ) 
        { 
          fprintf( stderr, "invalid argument to --%s:\n" 
              "r^2 veto time window is less than or equal to 0 msec: " 
              "(%f msecs specified)\n", 
              long_options[option_index].name, rsqVetoWindow ); 
          exit( 1 ); 
        } 
        ADD_PROCESS_PARAM( "float", "%s", optarg ); 
        break; 

      case '4': 
        rsqVetoThresh = atof( optarg ); 
        if ( rsqVetoThresh < 0 ) 
        { 
          fprintf( stderr, "invalid argument to --%s:\n" 
              " r^2 veto threshold must be positive: " 
              "(%f specified)\n",  
              long_options[option_index].name, rsqVetoThresh ); 
          exit( 1 ); 
        } 
        ADD_PROCESS_PARAM( "float", "%s", optarg ); 
        break; 

      case '5':
        bankSimParams.maxMatch = 1;
        ADD_PROCESS_PARAM( "string", "%s", " " );
        break;

      case '6':
        bankSimParams.maxMatch = 0;
        ADD_PROCESS_PARAM( "string", "%s", " " );
        break;

      case '7':
        /* create storage for the bank sim frame file name */
        optarg_len = strlen( optarg ) + 1;
        bankSimParams.frameName = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( bankSimParams.frameName, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case '8':
        /* create storage for the bank sim frame file channel */
        optarg_len = strlen( optarg ) + 1;
        bankSimParams.frameChan = (CHAR *) calloc( optarg_len, sizeof(CHAR));
        memcpy( bankSimParams.frameChan, optarg, optarg_len );
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;  

      case '9':
        numTDFiles = 1;
        /* Count the number of thinca files to follow up */
        while ( !strstr( argv[optind], "--" ) )
        {
          numTDFiles++;
          optind++;
        }
        optind = optind - numTDFiles;

        /* Set pointers to the relevant filenames */
        tdFollowUpFiles = (CHAR **) calloc( numTDFiles, sizeof(CHAR *));

        numTDFiles = 0;
        while ( !strstr( argv[optind], "--" ) )
        {
          tdFollowUpFiles[numTDFiles++] = argv[optind];
          ADD_PROCESS_PARAM( "string", "%s", argv[optind] );
          optind++;

        }
        break;

      case '*':
        /* store trigSanClustering method */
        if ( ! strcmp( "T0T3Tc", optarg ) )
        {
            trigScanMethod = T0T3Tc;
            trigScanAppendStragglers = 0;
        }
        else if ( ! strcmp( "T0T3TcAS", optarg ) )
        {
            trigScanMethod = T0T3Tc;
            trigScanAppendStragglers = 1;
        }
        else if ( ! strcmp( "Psi0Psi3Tc", optarg ) )
        {
            trigScanMethod = Psi0Psi3Tc;
            trigScanAppendStragglers = 0;
        }
        else if ( ! strcmp( "Psi0Psi3TcAS", optarg ) )
        {
            trigScanMethod = Psi0Psi3Tc;
            trigScanAppendStragglers = 1;
        }
        else 
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              "unknown scan method specified: %s\n"
              "(Must be one of T0T3Tc, T0T3TcAS, Psi0Psi3Tc, Psi0Psi3TcAS)\n", 
              long_options[option_index].name, optarg );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "string", "%s", optarg );
        break;

      case '<':
        /* TrigScan Delta End Time */ 
        trigScanDeltaEndTime = atof( optarg ); 
        if ( trigScanDeltaEndTime < 0.0L ) 
        { 
          fprintf( stderr, "invalid argument to --%s:\n" 
              "ts-endtime-interval must be positive: " 
              "(%f specified)\n",  
              long_options[option_index].name, trigScanDeltaEndTime ); 
          exit( 1 ); 
        } 
        ADD_PROCESS_PARAM( "float", "%s", optarg ); 
        break; 
      
      case '>':
        /* TrigScan Template Metric Scaling Factor */ 
        trigScanMetricScalingFac = atof( optarg ); 
        if ( trigScanMetricScalingFac <= 0.0 ) 
        { 
          fprintf( stderr, "invalid argument to --%s:\n" 
              "ts-volume-safety must be > 0.0 : " 
              "(%f specified)\n",  
              long_options[option_index].name, trigScanMetricScalingFac ); 
          exit( 1 ); 
        } 
        ADD_PROCESS_PARAM( "float", "%s", optarg ); 
        break; 

      case '?':
        exit( 1 );
        break;

      case '(':
        rsqVetoTimeThresh = atof( optarg );
        if ( rsqVetoTimeThresh < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              " r^2 veto time threshold must be positive: "
              "(%f specified)\n",
              long_options[option_index].name, rsqVetoTimeThresh );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%s", optarg );
        break;

      case ')':
        rsqVetoMaxSNR = atof( optarg );
        if ( rsqVetoMaxSNR < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              " r^2 veto maximum snr must be positive: "
              "(%f specified)\n",
              long_options[option_index].name, rsqVetoMaxSNR );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%s", optarg );
        break;

      case '[':
        rsqVetoCoeff = atof( optarg );
        if ( rsqVetoCoeff < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              " r^2 veto coefficient must be positive: "
              "(%f specified)\n",
              long_options[option_index].name, rsqVetoCoeff );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%s", optarg );
        break;

      case ']':
        rsqVetoPow = atof( optarg );
        if ( rsqVetoPow < 0 )
        {
          fprintf( stderr, "invalid argument to --%s:\n"
              " r^2 veto power must be positive: "
              "(%f specified)\n",
              long_options[option_index].name, rsqVetoPow );
          exit( 1 );
        }
        ADD_PROCESS_PARAM( "float", "%s", optarg );
        break;

      default:
        fprintf( stderr, "unknown error while parsing options (%d)\n", c );
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

  /* enable output is stored in the first process param row */
  if ( enableOutput == 1 )
  {
    LALSnprintf( procparams.processParamsTable->program, 
        LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
    LALSnprintf( procparams.processParamsTable->param,
        LIGOMETA_PARAM_MAX, "--enable-output" );
    LALSnprintf( procparams.processParamsTable->type, 
        LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( procparams.processParamsTable->value, 
        LIGOMETA_VALUE_MAX, " " );
  }
  else if ( enableOutput == 0 )
  {
    LALSnprintf( procparams.processParamsTable->program, 
        LIGOMETA_PROGRAM_MAX, "%s", PROGRAM_NAME );
    LALSnprintf( procparams.processParamsTable->param,
        LIGOMETA_PARAM_MAX, "--disable-output" );
    LALSnprintf( procparams.processParamsTable->type, 
        LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( procparams.processParamsTable->value, 
        LIGOMETA_VALUE_MAX, " " );
  }
  else
  {
    fprintf( stderr, "--enable-output or --disable-output "
        "argument must be specified\n" );
    exit( 1 );
  } 

  /* check inject-overhead option */
  if ( injectOverhead )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--inject-overhead" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }

  /* store point calibration option */
  if ( pointCal )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX,
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX,
        "--point-calibration" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }


  /*
   *
   * check validity of arguments
   *
   *
   */


  /* check validity of input data time */
  if ( ! gpsStartTimeNS )
  {
    fprintf( stderr, "--gps-start-time must be specified\n" );
    exit( 1 );
  }
  LAL_CALL( LALINT8toGPS( &status, &gpsStartTime, &gpsStartTimeNS ), 
      &status );
  if ( ! gpsEndTimeNS )
  {
    fprintf( stderr, "--gps-end-time must be specified\n" );
    exit( 1 );
  }
  LAL_CALL( LALINT8toGPS( &status, &gpsEndTime, &gpsEndTimeNS ), 
      &status );
  if ( gpsEndTimeNS <= gpsStartTimeNS )
  {
    fprintf( stderr, "invalid gps time range: "
        "start time: %d, end time %d\n",
        gpsStartTime.gpsSeconds, gpsEndTime.gpsSeconds );
    exit( 1 );
  }

  /* check trigger generation time is within input time */
  if ( trigStartTimeNS )
  {
    if ( trigStartTimeNS < gpsStartTimeNS )
    {
      fprintf( stderr, 
          "trigStartTimeNS = %lld\nis less than gpsStartTimeNS = %lld", 
          trigStartTimeNS, gpsStartTimeNS );
    }
  }
  if ( trigEndTimeNS )
  {
    if ( trigEndTimeNS > gpsEndTimeNS )
    {
      fprintf( stderr, 
          "trigEndTimeNS = %lld\nis greater than gpsEndTimeNS = %lld", 
          trigEndTimeNS, gpsEndTimeNS );
    }
  }

  /* check validity of data length parameters */
  if ( numPoints < 0 )
  {
    fprintf( stderr, "--segment-length must be specified\n" );
    exit( 1 );
  }
  if ( numSegments < 0 )
  {
    fprintf( stderr, "--number-of-segments must be specified\n" );
    exit( 1 );
  }
  if ( ovrlap < 0 )
  {
    fprintf( stderr, "--segment-overlap must be specified\n" );
    exit( 1 );
  }

  /* check sample rate has been given */
  if ( sampleRate < 0 )
  {
    fprintf( stderr, "--sample-rate must be specified\n" );
    exit( 1 );
  }

  /* check high pass option has been given */
  if ( highPass < 0 )
  {
    fprintf( stderr, "--disable-high-pass or --enable-high-pass (freq)"
        " must be specified\n" );
    exit( 1 );
  }
  else if ( ! highPass )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--disable-high-pass" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }
  else
  {
    /* check that all the high pass parameters have been specified */
    if ( highPassOrder < 0 )
    {
      fprintf( stderr, "--high-pass-order must be specified\n" );
      exit( 1 );
    }
    if ( highPassAtten < 0 )
    {
      fprintf( stderr, "--high-pass-attenuation must be specified\n" );
      exit( 1 );
    }
  }

  if ( calData == real_8 )
  {
    /* check that strain high pass parameters have been specified */
    if ( strainHighPassFreq < 0 )
    {
      fprintf( stderr, 
          "--strain-high-pass-freq must be specified for REAL8 h(t) data\n" );
      exit( 1 );
    }
    if ( strainHighPassOrder < 0 )
    {
      fprintf( stderr, 
          "--strain-high-pass-order must be specified for REAL8 h(t) data\n" );
      exit( 1 );
    }
    if ( strainHighPassAtten < 0 )
    {
      fprintf( stderr, 
          "--strain-high-pass-atten must be specified for REAL8 h(t) data\n" );
      exit( 1 );
    }
  }

  /* check validity of input data length */
  inputDataLength = numPoints * numSegments - ( numSegments - 1 ) * ovrlap;
  {
    INT8 gpsChanIntervalNS = gpsEndTimeNS - gpsStartTimeNS;
    INT8 inputDataLengthNS = (INT8) inputDataLength * 1000000000LL / 
      (INT8) sampleRate;

    if ( inputDataLengthNS != gpsChanIntervalNS )
    {
      fprintf( stderr, "length of input data and data chunk do not match\n" );
      fprintf( stderr, "start time: %lld, end time %lld\n",
          gpsStartTimeNS / 1000000000LL, gpsEndTimeNS / 1000000000LL );
      fprintf( stderr, "gps channel time interval: %lld ns\n"
          "computed input data length: %lld ns\n", 
          gpsChanIntervalNS, inputDataLengthNS );
      exit( 1 );
    }
  }

  /* check filter parameters have been specified */
  if ( numChisqBins < 0 )
  {
    fprintf( stderr, "--chisq-bins must be specified\n" );
    exit( 1 );
  }
  if ( fLow < 0 )
  {
    fprintf( stderr, "--low-frequency-cutoff must be specified\n" );
    exit( 1 );
  }
  if ( resampFiltType < 0 )
  {
    fprintf( stderr, "--resample-filter must be specified\n" );
    exit( 1 );
  }
  if ( specType < 0 )
  {
    fprintf( stderr, "--spectrum-type must be specified\n" );
    exit( 1 );
  }
  if ( clusterWindow * sampleRate > numPoints )				
  {
    fprintf( stderr, "--cluster-window must be less than "
        "--segment-length\n" );
    exit( 1 );
  }
  if ( ! haveClusterMethod )
  {
    fprintf( stderr, "--cluster-method must be specified\n" );
    exit( 1 );
  }
  if ( clusterMethod == window && clusterWindow == -1 )
  {
    fprintf( stderr, "--cluster-window must be specified "
        "if --clustering method 'window' chosen\n" );
    exit( 1 );
  }
  if ( clusterMethod != window && clusterWindow != -1 )
  {
    fprintf( stderr, "--cluster-window specified "
        "but --clustering method 'window' not chosen\n" );
    exit( 1 );
  }
  if ( invSpecTrunc < 0 )
  {
    fprintf( stderr, "--inverse-spec-length must be specified\n" );
    exit( 1 );
  }
  else if ( invSpecTrunc * sampleRate > numPoints )
  {
    fprintf( stderr, "--inverse-spec-length must be less than "
        "--segment-length\n" );
    exit( 1 );
  }

  if ( ! haveDynRange )
  {
    fprintf( stderr, "--dynamic-range-exponent must be specified\n" );
    exit( 1 );
  }
  if ( ! haveApprox )
  {
    fprintf( stderr, "--approximant must be specified\n" );
    exit( 1 );
  }

  /* check that a channel has been requested and fill the ifo */
  if ( ! fqChanName )
  {
    fprintf( stderr, "--channel-name must be specified\n" );
    exit( 1 );
  }

  if ( ! bankSim )
  {
    /* check that the thresholds have been specified */
    if ( snrThresh < 0 )
    {
      fprintf( stderr, "--snr-threshold must be specified\n" );
      exit( 1 );
    }
    if ( chisqThresh < 0 )
    {
      fprintf( stderr, "--chisq-threshold must be specified\n" );
      exit( 1 );
    }
    if ( chisqDelta < 0 )
    {
      fprintf( stderr, "--chisq-delta must be specified\n" );
      exit( 1 );
    }
  }

  /* check that we can correctly obtain the input frame data */
  if ( globFrameData )
  {
    if ( frInCacheName )
    {
      fprintf( stderr, 
          "--frame-cache must not be specified when globbing frame data\n" );
      exit( 1 );
    }

    if ( ! frInType )
    {
      fprintf( stderr, 
          "--frame-type must be specified when globbing frame data\n" );
      exit( 1 );
    }
  }
  else
  {
    if ( ! frInCacheName )
    {
      fprintf( stderr, 
          "--frame-cache must be specified when not globbing frame data\n" );
      exit( 1 );
    }

    if ( frInType )
    {
      fprintf( stderr, "--frame-type must not be specified when obtaining "
          "frame data from a cache file\n" );
      exit( 1 );
    }
  }

  /* record the glob frame data option in the process params */
  if ( globFrameData )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--glob-frame-data" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }

  /* check we can calibrate the data if it's not h(t) */
  if ( ! calData )
  {
    if ( ! ( calCacheName || globCalData ) )
    {
      fprintf( stderr, "either --calibration-cache or "
          "--glob-calibration-data must be specified\n" );
      exit( 1 );
    }
    else if ( calCacheName && globCalData )
    {
      fprintf( stderr, "only one of --calibration-cache or "
          "--glob-calibration-data can be specified\n" );
      exit( 1 );
    }
  }
  else
  {
    if ( calCacheName || globCalData )
    {
      fprintf( stderr, "neither --calibration-cache nor "
          "--glob-calibration-data\nshould be given for calibrated data\n" );
      exit( 1 );
    }
  }

  /* record the glob calibration data option in the process params */
  if ( globCalData )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--glob-calibration-data" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }

  /* check that a template bank has been specified */
  if ( ! bankFileName )
  {
    fprintf( stderr, "--bank-file must be specified\n" );
    exit( 1 );
  }

  /* check FindChirpSP is used if reverse chirp bank is specified */
  if ( reverseChirpBank )
  {
    if ( reverseChirpBank && ! ( approximant == FindChirpSP ) )
    {
      fprintf( stderr, "--approximant must be FindChirpSP if "
          "--reverse-chirp-bank is specified\n" );
      exit( 1 );
    }
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX,
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX,
        "--reverse-chirp-bank" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }

  /* check that a random seed for gaussian noise generation has been given */
  if ( gaussianNoise && randSeedType == unset )
  {
    fprintf( stderr, "--random-seed must be specified if "
        "--gaussian-noise is given\n" );
    exit( 1 );
  }

  /* check the bank simulation parameters */
  if ( bankSim )
  {
    if ( bankSimParams.maxMatch < 0 )
    {
      fprintf( stderr, "--enable-bank-sim-max or --disable-bank-sim-max "
          "must be specified\n" );
      exit( 1 );
    }

    if ( ! bankSimFileName )
    {
      if ( ! haveBankSimApprox )
      {
        fprintf( stderr, "--sim-approximant must be specified if "
            "--bank-simulation is not a sim_inspiral file\n" );
        exit( 1 );
      }

      if ( bankSimParams.approx == FrameFile )
      {
        if ( ! bankSimParams.frameName || ! bankSimParams.frameChan )
        {
          fprintf( stderr, "--sim-frame-file and --sim-frame-channel\n"
              "must be specified if --sim-approximant is FrameFile\n" );
          exit( 1 );
        }
      }
      else
      {
        if ( randSeedType == unset )
        {
          fprintf( stderr, "--random-seed must be specified if\n"
              "--bank-simulation is given and approx is not FrameFile\n" );
          exit( 1 );
        }
        if ( bankSimParams.minMass < 0 )
        {
          fprintf( stderr, "--sim-minimum-mass must be specified if\n"
              "--bank-simulation is given and approx is not FrameFile\n" );
          exit( 1 );
        }
        if ( bankSimParams.maxMass < 0 )
        {
          fprintf( stderr, "--sim-maximum-mass must be specified if\n"
              "--bank-simulation is given and approx is not FrameFile\n" );
          exit( 1 );
        }
      }
    }
  }

  /* check rsq veto flags */
  if ( enableRsqVeto == 0 )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--disable-rsq-veto" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }
  else if ( enableRsqVeto == 1 )
  {
    if ( ( rsqVetoWindow < 0 ) || ( rsqVetoThresh < 0 ) )
    {
      fprintf( stderr, "both --rsq-veto-window and --rsq-veto-thresh must\n"
          "be specified if the --enable-rsq-veto argument is given\n" );
      exit( 1 );
    }

    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--enable-rsq-veto" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }
  else
  {
    fprintf( stderr, 
        "one of --enable-rsq-veto or --disable-rsq-veto must be specified\n" );
    exit( 1 );
  }

  if ( doRsqVeto == 1 )
  {
    if ( enableRsqVeto == 0 )
    {
      fprintf( stderr, "--enable-rsq-veto must be specified \n"
          "if the --do-rsq-veto argument is specified\n" );
      exit( 1 );
    }
    else if ( ( rsqVetoTimeThresh < 0 ) || ( rsqVetoMaxSNR < 0 ) )
    {
      fprintf( stderr, "--rsq-veto-time-thresh and --rsq-veto-max-snr must be \n"
          "specified if the --do-rsq-veto argument is specified\n" );
      exit( 1 );      
    }
    else if ( ( rsqVetoCoeff > 0 ) && ( ( rsqVetoTimeThresh < 0 ) || ( rsqVetoMaxSNR < 0 ) || ( rsqVetoPow < 0 ) ) )
    {
      fprintf( stderr, "--rsq-veto-time-thresh --rsq-veto-max-snr and --rsq-veto-pow \n"
          "must be specified if --rsq-veto-coeff is specified\n" );
      exit( 1 );
    }
   else if ( ( rsqVetoPow > 0 ) && ( ( rsqVetoTimeThresh < 0 ) || ( rsqVetoMaxSNR < 0 ) || ( rsqVetoCoeff < 0 ) ) )
    {
      fprintf( stderr, "--rsq-veto-time-thresh --rsq-veto-max-snr and --rsq-veto-coeff \n"
          "must be specified if --rsq-veto-pow is specified\n" );
      exit( 1 );
    }

    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX,
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX,
        "--do-rsq-veto" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }

  /* check to filter injection segments only */
  if ( flagFilterInjOnly == 1 )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--enable-filter-inj-only" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  }
  else if ( flagFilterInjOnly == 0 )
  {
    this_proc_param = this_proc_param->next = (ProcessParamsTable *)
      calloc( 1, sizeof(ProcessParamsTable) );
    LALSnprintf( this_proc_param->program, LIGOMETA_PROGRAM_MAX, 
        "%s", PROGRAM_NAME );
    LALSnprintf( this_proc_param->param, LIGOMETA_PARAM_MAX, 
        "--disable-filter-inj-only" );
    LALSnprintf( this_proc_param->type, LIGOMETA_TYPE_MAX, "string" );
    LALSnprintf( this_proc_param->value, LIGOMETA_VALUE_MAX, " " );
  } 
  else if ( flagFilterInjOnly == -1 && injectionFile ) 
  {
    fprintf( stderr, "one of --enable-filter-inj-only or "
        "--disable-filter-inj-only must be specified\n" );
    exit( 1 );
  }

  /* Check the trigScan input parameters */
  if ( trigScanMethod )
  {
      if ( trigScanMetricScalingFac <= 0.0 )
      {
          fprintf ( stderr, "You must specify --ts-metric-scaling\n" );
          exit(1);
      }

      if ( maximizationInterval )
      {
          fprintf ( stderr, "Cannot specify both --maximization-interval"
                  " and --ts-cluster \nChoose any one of the two methods"
                  " for clustering raw inspiral triggers.\n" );
          exit(1);
      }
  }
  /* If the trigScan parameters are reasonable, set trigScanDeltaEndTime*/
  /* Here we change trigScanDeltaEndTime to msec                        */
  trigScanDeltaEndTime /= 1000.L;

  return 0;
}

#undef ADD_PROCESS_PARAM

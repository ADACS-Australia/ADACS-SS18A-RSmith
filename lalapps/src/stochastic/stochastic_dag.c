/*
 * stochastic.c - SGWB Standalone Analysis Pipeline
 *
 * Tania Regimbau <Tania.Regimbau@astro.cf.ac.uk>
 * Adam Mercer <ram@star.sr.bham.ac.uk>
 *
 *
 * $Id$
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include <unistd.h>
#include <getopt.h>

#include <FrameL.h>

#include <lal/AVFactories.h>
#include <lal/BandPassTimeSeries.h>
#include <lal/Calibration.h>
#include <lal/ComplexFFT.h>
#include <lal/CoarseGrainFrequencySeries.h>
#include <lal/Date.h>
#include <lal/DetectorSite.h>
#include <lal/IIRFilter.h>
#include <lal/FrameCache.h>
#include <lal/FrameCalibration.h>
#include <lal/FrameStream.h>
#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/LALRCSID.h>
#include <lal/LALStatusMacros.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/PrintFTSeries.h>
#include <lal/PrintVector.h>
#include <lal/Random.h>
#include <lal/RealFFT.h>
#include <lal/ResampleTimeSeries.h>
#include <lal/SimulateSB.h>
#include <lal/StreamInput.h>
#include <lal/StochasticCrossCorrelation.h>
#include <lal/TimeFreqFFT.h>
#include <lal/Units.h>
#include <lal/Window.h>

#include <lalapps.h>

#include "stochastic.h"

NRCSID (STOCHASTICC, "$Id$");
RCSID ("$Id$");

/* cvs info */
#define CVS_ID "$Id$"
#define CVS_REVISION "$Revision$"
#define CVS_DATE "$Date$"
#define PROGRAM_NAME "lalapps_stochastic_dag"

/* variables for getopt options parsing */
char *optarg;
int optind;

/* flags for getopt_long */
static int inject_flag;
static int apply_mask_flag;
static int high_pass_flag;
static int overlap_hann_flag;
static int test_flag;
static int post_analysis_flag;
extern int vrbflg;

/* parameters for the stochastic search */

/* sampling parameters */
INT4 sampleRate = 16384;
INT4 resampleRate = 1024;
REAL8 deltaF = 0.25;

/* data parameters */
LIGOTimeGPS gpsStartTime;
UINT8 startTime;
UINT8 stopTime;
INT4 intervalDuration = 180;
INT4 segmentDuration = 60;
INT4 calibDuration = 60;
INT4 calibOffset = 30;
CHAR *frameCache1;
CHAR *frameCache2;
CHAR *calCache1;
CHAR *calCache2;
CHAR channel1[LALNameLength];
CHAR channel2[LALNameLength];
CHAR ifo1[LALNameLength];
CHAR ifo2[LALNameLength];
INT4 site1;
INT4 site2;

/* frequency band */
INT4 fMin = 50;
INT4 fMax = 300;

/* omegaGW parameters */
REAL4 alpha = 0.0;
REAL4 fRef = 100.0;
REAL4 omegaRef = 1.;

/* monte carlo parameters */
/* at the moment the code cannot do monte carlo with overlapped Hann window */
REAL4 scaleFactor = 1.;
INT4 seed = 1;
INT4 NLoop = 1;

/* window parameters */
/* 60 s for pure Hann, 1 s for Tukey, 0s for rectangular window */
INT4 hannDuration = 1;

/* high pass filtering parameters */
REAL4 highPassFreq = 40.;
REAL4 highPassAt;
INT4  highPassOrder = 6;

/* number of bins for frequency masking */ 
INT4 maskBin = 1;

/* arguments associated with test flag */
INT4 testInter = 0;
INT4 testSeg = 0;
INT4 testTrial = 0;

/* output file */
CHAR outputFilePath[200];

INT4 main(INT4 argc, CHAR *argv[])
{
  /* variable declarations */

  /* status pointer */
  LALStatus status;

  /* output file */
  FILE *out1;
  FILE *out2;
  CHAR outputFilename1[200];
  CHAR outputFilename2[200];

  /* counters */
  INT4 i, j, n, segLoop, jobLoop, N;

  /* results parameters */
  REAL8 y;
  REAL8 yOpt;
  REAL8 varTheo;
  REAL8 inVarTheoSum;
  REAL8 ptEst;
  REAL8 error;

  /* input data segment */
  INT4 numSegments;
  INT4 numJobs;
  INT4 segMiddle;
  INT4 segmentLength;
  INT4 segmentPadLength;
  INT4 intervalLength;
  INT4 segmentShift;
  INT4 padData;
  LIGOTimeGPS gpsStartPadTime;
  LIGOTimeGPS gpsCalibTime;
  ReadDataPairParams streamParams;
  StreamPair streamPair;
  REAL4TimeSeries segment1;
  REAL4TimeSeries segment2;
  REAL4TimeSeries segmentPad1;
  REAL4TimeSeries segmentPad2;
  REAL4TimeSeries interval1;
  REAL4TimeSeries interval2;
  REAL4Vector *seg1[100];
  REAL4Vector *segPad1[100];
  REAL4Vector *seg2[100];
  REAL4Vector *segPad2[100];

  /* simulated signal structures */
  StochasticOmegaGWParameters parametersOmega;
  SSSimStochBGParams SBParams;
  SSSimStochBGInput SBInput;
  SSSimStochBGOutput SBOutput;
  REAL4TimeSeries SimStochBG1;
  REAL4TimeSeries SimStochBG2;
  REAL4FrequencySeries MComegaGW;
  COMPLEX8FrequencySeries MCresponse1;
  COMPLEX8FrequencySeries MCresponse2;
  COMPLEX8Vector *MCresp1[100];
  COMPLEX8Vector *MCresp2[100];

  INT4 MCLoop;
  INT4 MCfreqLength;
  REAL8 MCdeltaF;
  REAL8 MCdeltaT;
  LALUnit countPerStrain = {0,{0,0,0,0,0,-1,1},{0,0,0,0,0,0,0}};

  /* window for segment data streams */
  REAL4TimeSeries dataWindow;

  /* hann window */
  INT4 hannLength;
  LALWindowParams hannParams;
  REAL4Vector *hannWindow;

  /* high pass filtering */
  PassBandParamStruc highpassParam;

  /* response functions */
  COMPLEX8FrequencySeries responseTemp1;
  COMPLEX8FrequencySeries responseTemp2;
  COMPLEX8FrequencySeries response1;
  COMPLEX8FrequencySeries response2;
  COMPLEX8Vector *resp1[100];
  COMPLEX8Vector *resp2[100];
  INT4 respLength;
  CalibrationUpdateParams calfacts;
  LALUnit countPerAttoStrain = {18,{0,0,0,0,0,-1,1},{0,0,0,0,0,0,0}};

  /* data structures for PSDs */
  INT4 overlapPSDLength;
  INT4 psdTempLength;
  INT4 windowPSDLength;
  INT4 filterLength;
  INT4 numPointInf, numPointSup;
  LALWindowParams winparPSD;
  AverageSpectrumParams specparPSD;
  REAL4FrequencySeries psdTemp1;
  REAL4FrequencySeries psdTemp2;
  REAL4FrequencySeries psd1;
  REAL4FrequencySeries psd2;
  REAL4Vector *calPsd1;
  REAL4Vector *calPsd2;
  LALUnit psdUnits = {0,{0,0,1,0,0,0,2},{0,0,0,0,0,0,0}};

  /* calibrated inverse noise data structures */
  REAL4FrequencySeries calInvPsd1;
  REAL4FrequencySeries calInvPsd2;

  /* units for inverse noise */
  LALUnit calPSDUnit = {36,{0,0,-1,0,0,-2,0},{0,0,0,0,0,0,0}};

  /* structures for LALInverseNoise */
  StochasticInverseNoiseInput inverseNoiseIn1;
  StochasticInverseNoiseInput inverseNoiseIn2;
  StochasticInverseNoiseCalOutput inverseNoiseOut1;
  StochasticInverseNoiseCalOutput inverseNoiseOut2;

  /* zeropad and fft structures */
  SZeroPadAndFFTParameters zeroPadParams;
  RealFFTPlan *fftDataPlan = NULL;
  COMPLEX8FrequencySeries hBarTilde1;
  COMPLEX8FrequencySeries hBarTilde2;
  UINT4 zeroPadLength;
  UINT4 fftDataLength;

  /* overlap reduction function */
  LALDetectorPair detectors;
  REAL4FrequencySeries overlap;
  OverlapReductionFunctionParameters ORFparams;
  LALUnit overlapUnit = {0,{0,0,0,0,0,2,0},{0,0,0,0,0,0,0}};

  /* frequency mask structures */
  REAL4FrequencySeries mask;
  REAL4Vector *maskTemp;
  INT4 Nbin;

  /* structures for optimal filter normalisation */
  StochasticOptimalFilterNormalizationInput normInput;
  StochasticOptimalFilterNormalizationOutput normOutput;
  StochasticOptimalFilterNormalizationParameters normParams;
  REAL4WithUnits normLambda;
  REAL4WithUnits normSigma;
  REAL8 lambda;

  /* structures for optimal filter */
  REAL4FrequencySeries optFilter;
  StochasticOptimalFilterCalInput optFilterIn;

  /* spectrum structures */
  StochasticOmegaGWParameters omegaGWParams;
  REAL4FrequencySeries omegaGW;

  /* structures for CC spectrum and CC statistics */
  StochasticCrossCorrelationCalInput ccIn;
  BOOLEAN epochsMatch = 1;
  REAL4WithUnits ccStat;
  COMPLEX8FrequencySeries ccSpectrum;

  /* error handler */
  status.statusPtr = NULL;

  /* set default error behaviour */
  lal_errhandler = LAL_ERR_EXIT;
  set_debug_level("33");

  /* parse command line options */
  parseOptions(argc, argv);

  /* initialize gps time structure */
  gpsStartTime.gpsSeconds = startTime;
  gpsStartTime.gpsNanoSeconds = 0.;
  gpsStartPadTime.gpsSeconds = startTime - padData;
  gpsStartPadTime.gpsNanoSeconds = 0.;
  gpsCalibTime.gpsSeconds = startTime + calibOffset;
  gpsCalibTime.gpsNanoSeconds = 0.;

  if (vrbflg)
  {
    fprintf(stdout,"number of jobs:");
  }

  /* get number of segments */
  numSegments = (INT4)(intervalDuration / segmentDuration);
  numJobs = (INT4)((stopTime - startTime) / segmentDuration ) - \
    numSegments + 1;
  segMiddle = (INT4) ((numSegments - 1) / 2);
  segmentShift = segmentDuration;

  if (vrbflg)
  {
    fprintf(stdout, "%d\n", numJobs);
  }

  if (overlap_hann_flag)
  {
    /* numSegments = 2 * numSegments - 1; */
    segmentShift = segmentDuration / 2;
  }

  if ((sampleRate == resampleRate)&&(high_pass_flag == 0))
  {
    padData = 0;
  }
  else
  {
    padData = 1;
  }

  /* set length for data segments */
  intervalLength = (intervalDuration + (2 * padData)) * resampleRate;
  segmentPadLength = (segmentDuration + (2 * padData)) * resampleRate;
  segmentLength = segmentDuration * resampleRate;

  /* set metadata fields for data segments */
  strncpy(interval1.name, "interval1", LALNameLength);
  strncpy(interval2.name, "interval2", LALNameLength);
  interval1.sampleUnits = interval2.sampleUnits = lalADCCountUnit;
  interval1.epoch = interval2.epoch = gpsStartTime;
  interval1.deltaT = interval2.deltaT = 1./(REAL8)resampleRate;
  interval1.f0 = interval2.f0 = 0;

  strncpy(segmentPad1.name, "segmentPad1", LALNameLength);
  strncpy(segmentPad2.name, "segmentPad2", LALNameLength);
  segmentPad1.sampleUnits = segmentPad2.sampleUnits = lalADCCountUnit;
  segmentPad1.epoch = segmentPad2.epoch = gpsStartTime;
  segmentPad1.deltaT = segmentPad2.deltaT = 1./(REAL8)resampleRate;
  segmentPad1.f0 = segmentPad2.f0 = 0;

  strncpy(segment1.name, "segment1", LALNameLength);
  strncpy(segment2.name, "segment2", LALNameLength);
  segment1.sampleUnits = segment2.sampleUnits = lalADCCountUnit;
  segment1.epoch = segment2.epoch = gpsStartTime;
  segment1.deltaT = segment2.deltaT = 1./(REAL8)resampleRate;
  segment1.f0 = segment2.f0 = 0;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for data segments...\n");
  }

  /* allocate memory for data segments */

  interval1.data = interval2.data = NULL;
  LAL_CALL( LALSCreateVector(&status, &(interval1.data), intervalLength), \
      &status );
  LAL_CALL( LALSCreateVector(&status, &(interval2.data), intervalLength), \
      &status );
  memset( interval1.data->data, 0, \
      interval1.data->length * sizeof(*interval1.data->data));
  memset( interval2.data->data, 0, \
      interval2.data->length * sizeof(*interval2.data->data));

  segmentPad1.data = segmentPad2.data = NULL;
  LAL_CALL( LALSCreateVector(&status, &(segmentPad1.data), \
        segmentPadLength), &status );
  LAL_CALL( LALSCreateVector(&status, &(segmentPad2.data), \
        segmentPadLength), &status );
  memset( segmentPad1.data->data, 0, \
      segmentPad1.data->length * sizeof(*segmentPad1.data->data));
  memset( segmentPad2.data->data, 0, \
      segmentPad2.data->length * sizeof(*segmentPad2.data->data));

  segment1.data = segment2.data = NULL;
  LAL_CALL( LALSCreateVector(&status, &(segment1.data), segmentLength), \
      &status );
  LAL_CALL( LALSCreateVector(&status, &(segment2.data), segmentLength), \
      &status );
  memset( segment1.data->data, 0, \
      segment1.data->length * sizeof(*segment1.data->data));
  memset( segment2.data->data, 0, \
      segment2.data->length * sizeof(*segment2.data->data));


  for (i = 0; i < numSegments; i++)
  {
    segPad1[i]= segPad2[i] = NULL;
    LAL_CALL( LALCreateVector(&status, &(segPad1[i]), segmentPadLength), \
        &status);
    LAL_CALL( LALCreateVector(&status, &(segPad2[i]), segmentPadLength), \
        &status);

    memset( segPad1[i]->data, 0, \
        segPad1[i]->length * sizeof(*segPad1[i]->data));
    memset( segPad2[i]->data, 0, \
        segPad2[i]->length * sizeof(*segPad2[i]->data));
  }

  for (i = 0; i < numSegments; i++)
  {
    seg1[i]= seg2[i] = NULL;
    LAL_CALL( LALCreateVector(&status, &(seg1[i]), segmentLength), \
        &status);
    LAL_CALL( LALCreateVector(&status, &(seg2[i]), segmentLength), \
        &status);

    memset( seg1[i]->data, 0, \
        seg1[i]->length * sizeof(*seg1[i]->data));
    memset( seg2[i]->data, 0, \
        seg2[i]->length * sizeof(*seg2[i]->data));
  }

  /* set segment input parameters */
  streamParams.frameCache1 = frameCache1;
  streamParams.frameCache2 = frameCache2;
  streamParams.ifo1 = ifo1;
  streamParams.ifo2 = ifo2;
  streamParams.channel1 = channel1;
  streamParams.channel2 = channel2;
  streamParams.buffer = 0;
  streamParams.sampleRate = sampleRate;
  streamParams.resampleRate = resampleRate;

  /* set stream data structures */
  streamPair.stream1 = &interval1;
  streamPair.stream2 = &interval2;

  if (inject_flag)
  {
    if (vrbflg)
    {
      fprintf(stdout, "Allocating memory for MC...\n");
    }

    MCdeltaT = 1.0 / resampleRate;
    MCdeltaF = (REAL8)resampleRate / (REAL8)segmentPadLength;
    MCfreqLength = (segmentPadLength / 2) + 1;

    /* create vectors to store the simulated signal */
    strncpy(SimStochBG1.name, "Whitened-SimulatedSB1", LALNameLength);
    strncpy(SimStochBG2.name, "Whitened-SimulatedSB2", LALNameLength);
    SimStochBG1.f0 = SimStochBG2.f0 = 0.;
    SimStochBG1.epoch = SimStochBG2.epoch = gpsStartPadTime;
    SimStochBG1.deltaT = SimStochBG2.deltaT = 1./(REAL8)resampleRate;
    SimStochBG1.sampleUnits = SimStochBG2.sampleUnits = lalADCCountUnit;
    SimStochBG1.data = SimStochBG2.data = NULL;
    LAL_CALL( LALSCreateVector(&status, &(SimStochBG1.data), \
          segmentPadLength), &status );
    LAL_CALL( LALSCreateVector(&status, &(SimStochBG2.data), \
          segmentPadLength), &status );

    memset(SimStochBG1.data->data, 0, \
        SimStochBG1.data->length *sizeof(*SimStochBG1.data->data));
    memset(SimStochBG2.data->data, 0, \
        SimStochBG2.data->length *sizeof(*SimStochBG2.data->data));

    /* define parameters for SimulateSB */
    SBParams.length = segmentPadLength;
    SBParams.deltaT = 1. / resampleRate;
    SBParams.detectorOne = lalCachedDetectors[site1];
    SBParams.detectorTwo = lalCachedDetectors[site2];
    SBParams.SSimStochBGTimeSeries1Unit = lalADCCountUnit;
    SBParams.SSimStochBGTimeSeries2Unit = lalADCCountUnit;

    /* omegaGW */
    parametersOmega.length = MCfreqLength;
    parametersOmega.f0 = 0.;
    parametersOmega.deltaF = MCdeltaF;
    parametersOmega.alpha = alpha;
    parametersOmega.fRef = fRef;
    parametersOmega.omegaRef = omegaRef;

    /* allocate memory */
    MComegaGW.data = NULL;
    LAL_CALL( LALSCreateVector(&status, &(MComegaGW.data), MCfreqLength), \
        &status );

    memset(MComegaGW.data->data, 0,
        MComegaGW.data->length * sizeof(*MComegaGW.data->data));

    /* generate omegaGW */
    LAL_CALL( LALStochasticOmegaGW( &status, &MComegaGW, &parametersOmega), \
        &status);

    /* response functions */
    memset(&calfacts, 0, sizeof(CalibrationUpdateParams));
    strncpy(MCresponse1.name,"MCresponse1", LALNameLength);
    strncpy(MCresponse2.name,"MCresponse2", LALNameLength);
    MCresponse1.sampleUnits = MCresponse2.sampleUnits = countPerStrain;
    MCresponse1.epoch = MCresponse2.epoch = gpsCalibTime;
    MCresponse1.deltaF = MCresponse2.deltaF = MCdeltaF;
    MCresponse1.f0 = MCresponse2.f0 = 0;

    MCresponse1.data = MCresponse2.data = NULL;
    LAL_CALL( LALCCreateVector(&status, &(MCresponse1.data), MCfreqLength), \
        &status);
    LAL_CALL( LALCCreateVector(&status, &(MCresponse2.data), MCfreqLength), \
        &status);

    memset( MCresponse1.data->data, 0, \
        MCresponse1.data->length * sizeof(*MCresponse1.data->data));
    memset( MCresponse2.data->data, 0, \
        MCresponse2.data->length * sizeof(*MCresponse2.data->data));

    for (i = 0; i < numSegments; i++)
    {
      MCresp1[i]= MCresp2[i] = NULL;
      LAL_CALL( LALCCreateVector( &status, &(MCresp1[i]), MCfreqLength), \
          &status);
      LAL_CALL( LALCCreateVector( &status, &(MCresp2[i]), MCfreqLength), \
          &status);

      memset( MCresp1[i]->data, 0, \
          MCresp1[i]->length * sizeof(*MCresp1[i]->data));
      memset( MCresp2[i]->data, 0, \
          MCresp2[i]->length * sizeof(*MCresp2[i]->data));
    }
  }

  /* set parameters for PSD estimation */
  windowPSDLength = (UINT4)(resampleRate / deltaF);
  overlapPSDLength = windowPSDLength / 2;
  psdTempLength = (windowPSDLength / 2) + 1;
  numPointInf = (UINT4)(fMin / deltaF);
  numPointSup = (UINT4)(fMax / deltaF);
  filterLength = numPointSup - numPointInf + 1;

  specparPSD.method = useMean;
  specparPSD.overlap = overlapPSDLength;
  specparPSD.plan = NULL;
  specparPSD.window = NULL;

  /* set window parameters for PSD estimation */
  winparPSD.length = windowPSDLength;
  winparPSD.type = Hann;

  /* set metadata fields for PSDs */
  strncpy(psdTemp1.name, "psdTemp1", LALNameLength);
  strncpy(psdTemp2.name, "psdTemp2", LALNameLength);
  psdTemp1.sampleUnits = psdTemp2.sampleUnits = psdUnits;
  psdTemp1.deltaF = psdTemp2.deltaF = deltaF;
  psdTemp1.f0 = psdTemp2.f0 = 0;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for PSDs...\n");
  }

  /* allocate memory for PSDs */
  psdTemp1.data = psdTemp2.data = NULL;
  LAL_CALL( LALCreateVector(&status, &(psdTemp1.data), psdTempLength), \
      &status );
  LAL_CALL( LALCreateVector(&status, &(psdTemp2.data), psdTempLength), \
      &status );
  memset( psdTemp1.data->data, 0, \
      psdTemp1.data->length * sizeof(*psdTemp1.data->data));
  memset( psdTemp2.data->data, 0, \
      psdTemp2.data->length * sizeof(*psdTemp2.data->data));

  /* reduced frequency band PSDs */
  /* set metadata fields for reduced frequency band PSDs */
  strncpy(psd1.name, "psd1", LALNameLength);
  strncpy(psd2.name, "psd2", LALNameLength);
  psd1.deltaF = psd2.deltaF = deltaF;
  psd1.f0 = psd2.f0 = fMin;
  psd1.sampleUnits = psd2.sampleUnits = psdUnits;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for reduced frequency band PSDs...\n");
  }

  /* allocate memory for reduced frequency band PSDs */
  psd1.data = psd2.data = NULL;
  LAL_CALL( LALCreateVector(&status, &psd1.data, filterLength), &status );
  LAL_CALL( LALCreateVector(&status, &psd2.data, filterLength), &status );
  memset(psd1.data->data, 0, psd1.data->length * sizeof(*psd1.data->data));
  memset(psd2.data->data, 0, psd2.data->length * sizeof(*psd2.data->data));

  /* allocate memory for calibrated PSDs */
  calPsd1 = calPsd2 = NULL;
  LAL_CALL( LALCreateVector(&status, &calPsd1, filterLength), &status );
  LAL_CALL( LALCreateVector(&status, &calPsd2, filterLength), &status );
  memset(calPsd1->data, 0, calPsd1->length * sizeof(*calPsd1->data));
  memset(calPsd2->data, 0, calPsd2->length * sizeof(*calPsd2->data));

  /* set parameters for response functions */
  respLength = (UINT4)(fMax / deltaF) + 1;

  /* set metadata fields for response functions */
  strncpy(responseTemp1.name, "responseTemp1", LALNameLength);
  strncpy(responseTemp2.name, "responseTemp2", LALNameLength);
  responseTemp1.sampleUnits = responseTemp2.sampleUnits = countPerAttoStrain;
  responseTemp1.epoch = responseTemp2.epoch = gpsCalibTime;
  responseTemp1.deltaF = responseTemp2.deltaF = deltaF;
  responseTemp1.f0 = responseTemp2.f0 = 0;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for response functions...\n");
  }

  /* allocate memory for response functions */
  responseTemp1.data = responseTemp2.data = NULL;
  LAL_CALL( LALCCreateVector(&status, &(responseTemp1.data), respLength), \
      &status );
  LAL_CALL( LALCCreateVector(&status, &(responseTemp2.data), respLength), \
      &status );
  memset( responseTemp1.data->data, 0, \
      responseTemp1.data->length * sizeof(*responseTemp1.data->data));
  memset( responseTemp2.data->data, 0, \
      responseTemp2.data->length * sizeof(*responseTemp2.data->data));

  /* reduced frequency band response functions */
  /* set metadata fields for reduced frequency band response functions */
  strncpy(response1.name, "response1", LALNameLength);
  strncpy(response2.name, "response2", LALNameLength);
  response1.sampleUnits = response2.sampleUnits = countPerAttoStrain;
  response1.epoch = response2.epoch = gpsCalibTime;
  response1.deltaF = response2.deltaF = deltaF;
  response1.f0 = response2.f0 = fMin;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for reduced frequency band "
        "response functions...\n");
  }

  /* allocate memory for reduced frequency band response functions */
  response1.data = response2.data = NULL;
  LAL_CALL( LALCCreateVector(&status, &(response1.data), filterLength), \
      &status );
  LAL_CALL( LALCCreateVector(&status, &(response2.data), filterLength), \
      &status );
  memset( response1.data->data, 0, \
      response1.data->length * sizeof(*response1.data->data));
  memset( response2.data->data, 0, \
      response2.data->length * sizeof(*response2.data->data));

  for (i = 0; i < numSegments; i++)
  {
    resp1[i]= resp2[i] = NULL;
    LAL_CALL( LALCCreateVector( &status, &(resp1[i]), filterLength), \
        &status);
    LAL_CALL( LALCCreateVector( &status, &(resp2[i]), filterLength), \
        &status);

    memset( resp1[i]->data, 0, \
        resp1[i]->length * sizeof(*resp1[i]->data));
    memset( resp2[i]->data, 0, \
        resp2[i]->length * sizeof(*resp2[i]->data));
  }

  if (vrbflg)
  {
    fprintf(stdout, "Creating FFT plan for PSD estimation...\n");
  }

  /* create fft plan */
  LAL_CALL ( LALCreateForwardRealFFTPlan(&status, &specparPSD.plan, \
        windowPSDLength, 0), &status );

  if (vrbflg)
  {
    fprintf(stdout, "Creating window for PSD estimation...\n");
  }

  /* create window for PSD estimation */
  LAL_CALL( LALCreateREAL4Window(&status, &specparPSD.window, &winparPSD), \
      &status );

  /* set metadata fields for inverse noise structures */
  strncpy(calInvPsd1.name, "calInvPsd1", LALNameLength);
  strncpy(calInvPsd2.name, "calInvPsd2", LALNameLength);
  calInvPsd1.sampleUnits = calInvPsd2.sampleUnits = calPSDUnit;
  calInvPsd1.deltaF = calInvPsd2.deltaF = deltaF;
  calInvPsd1.f0 = calInvPsd2.f0 = fMin;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for inverse noise...\n");
  }

  /* allocate memory for inverse noise */
  calInvPsd1.data = calInvPsd2.data = NULL;
  LAL_CALL( LALCreateVector( &status, &(calInvPsd1.data), filterLength), \
      &status );
  LAL_CALL( LALCreateVector(&status, &(calInvPsd2.data), filterLength), \
      &status );
  memset( calInvPsd1.data->data, 0, \
      calInvPsd1.data->length * sizeof(*calInvPsd1.data->data));
  memset( calInvPsd2.data->data, 0, \
      calInvPsd2.data->length * sizeof(*calInvPsd2.data->data));

  /* set inverse noise inputs */
  inverseNoiseIn1.unCalibratedNoisePSD = &psd1;
  inverseNoiseIn1.responseFunction = &response1;
  inverseNoiseIn2.unCalibratedNoisePSD = &psd2;
  inverseNoiseIn2.responseFunction = &response2;

  /* set inverse noise outputs */
  inverseNoiseOut1.calibratedInverseNoisePSD = &calInvPsd1;
  inverseNoiseOut2.calibratedInverseNoisePSD = &calInvPsd2;

  /* set window parameters for segment data streams */
  strncpy(dataWindow.name, "dataWindow", LALNameLength);
  dataWindow.sampleUnits = lalDimensionlessUnit;
  dataWindow.deltaT = 1./resampleRate;
  dataWindow.f0 = 0;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for data segment window...\n");
  }

  /* allocate memory for segment window */
  dataWindow.data = NULL;
  LAL_CALL( LALSCreateVector(&status, &(dataWindow.data), segmentLength), \
      &status );
  memset( dataWindow.data->data, 0, \
      dataWindow.data->length * sizeof(*dataWindow.data->data));

  if (vrbflg)
  {
    fprintf(stdout, "Generating data segment window...\n");
  }

  /* generate window */
  for (i = 0; i < segmentLength; i++)
  {
    dataWindow.data->data[i] = 1.;
  }
  if (overlap_hann_flag)
  {
    hannDuration = segmentLength;
  }

  if (hannDuration != 0)
  {
    /* generate pure Hann window */
    hannLength = hannDuration * resampleRate;
    hannParams.length = hannLength;
    hannParams.type = Hann;

    /* allocate memory for hann window */
    hannWindow = NULL;
    LAL_CALL( LALSCreateVector(&status, &hannWindow, hannLength), &status );
    memset( hannWindow->data, 0, \
        hannWindow->length * sizeof(*hannWindow->data));

    /* generate hann window */
    LAL_CALL( LALWindow(&status, hannWindow, &hannParams), &status );

    /* construct Tukey window */
    for (i = 0; i < hannLength / 2; i++)
    {
      dataWindow.data->data[i] = hannWindow->data[i];
    }
    for (i = segmentLength - (hannLength / 2); i < segmentLength; i++)
    {
      dataWindow.data->data[i] = \
        hannWindow->data[i - segmentLength + hannLength];
    }
  }

  /* print window */
  if (test_flag)
  {
    LALSPrintTimeSeries(&dataWindow, "dataWindow.dat");
  }

  /* structure for high pass filtering */
  if (high_pass_flag)
  {
    highpassParam.nMax = highPassOrder;
    highpassParam.f1 = -1.0;
    highpassParam.f2 = highPassFreq;
    highpassParam.a1 = -1.0;
    highpassParam.a2 = highPassAt;
  }

  /* zeropad lengths */
  zeroPadLength = 2 * segmentLength;
  fftDataLength = (zeroPadLength / 2) + 1;

  /* create fft plan */
  LAL_CALL( LALCreateForwardRealFFTPlan(&status, &fftDataPlan, \
        zeroPadLength, 0), &status );

  /* set metadata fields for zeropad ffts */
  strncpy(hBarTilde1.name, "hBarTilde1", LALNameLength);
  strncpy(hBarTilde2.name, "hBarTilde2", LALNameLength);

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for zeropad...\n");
  }

  /* allocate memory for zeropad */
  hBarTilde1.data = hBarTilde2.data = NULL;
  LAL_CALL( LALCCreateVector(&status, &(hBarTilde1.data), fftDataLength), \
      &status );
  LAL_CALL( LALCCreateVector(&status, &(hBarTilde2.data), fftDataLength), \
      &status );
  memset( hBarTilde1.data->data, 0, \
      hBarTilde1.data->length * sizeof(*hBarTilde1.data->data));
  memset( hBarTilde2.data->data, 0, \
      hBarTilde2.data->length * sizeof(*hBarTilde2.data->data));

  /* set zeropad parameters */
  zeroPadParams.fftPlan = fftDataPlan;
  zeroPadParams.window = dataWindow.data;
  zeroPadParams.length = zeroPadLength;

  /* quantities needed to build the optimal filter */

  /* set metadata fields for overlap reduction function */
  strncpy(overlap.name, "overlap", LALNameLength);
  overlap.sampleUnits = overlapUnit;
  overlap.deltaF = deltaF;
  overlap.f0 = fMin;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for the overlap reduction " \
        "function...\n");
  }

  /* allocate memory for overlap reduction function */
  overlap.data = NULL;
  LAL_CALL( LALCreateVector(&status, &(overlap.data), filterLength), &status );
  memset( overlap.data->data, 0, \
      overlap.data->length * sizeof(*overlap.data->data));

  /* set parameters for overlap reduction function */
  ORFparams.length = filterLength;
  ORFparams.f0 = fMin;
  ORFparams.deltaF = deltaF;

  /* set overlap reduction function detector pair */
  detectors.detectorOne = lalCachedDetectors[site1];
  detectors.detectorTwo = lalCachedDetectors[site2];

  if (vrbflg)
  {
    fprintf(stdout, "Generating the overlap reduction function...\n");
  }

  /* generate overlap reduction function */
  LAL_CALL ( LALOverlapReductionFunction( &status, &overlap, &detectors, \
        &ORFparams), &status );

  /* print */
  if (test_flag)
  {
    LALSPrintFrequencySeries(&overlap, "overlap.dat");
  }

  /* set metadata fields for spectrum */
  strncpy(omegaGW.name, "omegaGW", LALNameLength);
  omegaGW.sampleUnits = lalDimensionlessUnit;
  omegaGW.deltaF = deltaF;
  omegaGW.f0 = fMin;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for spectrum...\n");
  }

  /* allocate memory for spectrum */
  omegaGW.data = NULL;
  LAL_CALL( LALCreateVector(&status, &(omegaGW.data), filterLength), &status );
  memset( omegaGW.data->data, 0, \
      omegaGW.data->length * sizeof(*omegaGW.data->data));

  /* set omegaGW parameters */
  omegaGWParams.alpha = alpha;
  omegaGWParams.fRef = fRef;
  omegaGWParams.omegaRef = omegaRef;
  omegaGWParams.length = filterLength;
  omegaGWParams.f0 = fMin;
  omegaGWParams.deltaF = deltaF;

  if (vrbflg)
  {
    fprintf(stdout, "Generating spectrum for optimal filter...\n");
  }

  /* generage omegaGW */
  LAL_CALL( LALStochasticOmegaGW(&status, &omegaGW, &omegaGWParams), \
      &status );

  /* frequency mask */
  if (apply_mask_flag)
  {
    /* extra bins */
    Nbin = (maskBin - 1) / 2;

    /* set metadata fields for frequency mask */
    strncpy(mask.name, "mask", LALNameLength);
    mask.deltaF = deltaF;
    mask.f0 = fMin;
    mask.sampleUnits = lalDimensionlessUnit;

    if (vrbflg)
    {
      fprintf(stdout, "Allocating memory for frequency mask...\n");
    }

    /* allocate memory for frequency mask */
    mask.data = maskTemp = NULL;
    LAL_CALL( LALCreateVector(&status, &(mask.data), filterLength), &status );
    LAL_CALL( LALCreateVector(&status, &maskTemp, respLength), &status );
    memset(mask.data->data, 0, mask.data->length * sizeof(*mask.data->data));
    memset(maskTemp->data, 0, maskTemp->length * sizeof(*maskTemp->data));

    if (vrbflg)
    {
      fprintf(stdout, "Generating frequency mask...\n");
    }

    /* set all values to 1 */
    for (i = 0; i < respLength; i++)
    {
      maskTemp->data[i] = 1.;
    }

    if (vrbflg)
    {
      fprintf(stdout, "Masking multiples of 16 Hz...\n");
    }

    /* remove multiples of 16 Hz */
    for (i = 0; i < respLength; i += (UINT4)(16 / deltaF))
    {
      maskTemp->data[i]= 0.;
      for (j = 0; j < Nbin; j ++)
      {
        if ((i + 1 + j) < respLength)
        {
          maskTemp->data[i + 1 + j] = 0.;
        }
        if ((i - 1 - j) > 0 )
        {
          maskTemp->data[i - 1 - j] = 0.;
        }
      }
    }

    if (vrbflg)
    {
      fprintf(stdout, "Masking multiples of 60 Hz...\n");
    }

    /* remove multiples of 60 Hz */
    for (i = 0; i < respLength; i += (UINT4)(60 / deltaF))
    {
      maskTemp->data[i] = 0.;
      for (j = 0; j < Nbin; j ++)
      {
        if ((i + 1 + j) < respLength)
        {
          maskTemp->data[i + 1 + j] = 0.;
        }
        if ((i - 1 - j) > 0 )
        {
          maskTemp->data[i - 1 - j] = 0.;
        }
      }
    }

    if (vrbflg)
    {
      fprintf(stdout, "Getting appropriate frequency band for mask...\n");
    }

    /* get appropriate band */
    for (i = 0; i < filterLength; i++)
    {
      mask.data->data[i] = maskTemp->data[i + numPointInf];
    }

    if (test_flag)
    {
      LALSPrintFrequencySeries(&mask, "mask.dat");
    }

    if (vrbflg)
    {
      fprintf(stdout, "Applying frequency mask to spectrum..\n");
    }

    /* apply mask to omegaGW */
    for (i = 0; i < filterLength; i++)
    {
      omegaGW.data->data[i] *= mask.data->data[i];
    }
  }

  /* print */
  if (test_flag)
  {
    LALSPrintFrequencySeries(&omegaGW, "omegaGW.dat");
  }

  /* set normalisation parameters */
  normParams.fRef = fRef;
  normParams.heterodyned = 0;
  normParams.window1 = normParams.window2 = dataWindow.data;

  /* set normalisation input */
  normInput.overlapReductionFunction = &overlap;
  normInput.omegaGW = &omegaGW;
  normInput.inverseNoisePSD1 = &calInvPsd1;
  normInput.inverseNoisePSD2 = &calInvPsd2;

  /* set normalisation output */
  normOutput.normalization = &normLambda;
  normOutput.variance = &normSigma;

  /* set metadata fields for optimal filter */
  strncpy(optFilter.name, "optFilter", LALNameLength);
  optFilter.epoch = gpsStartTime;
  optFilter.deltaF = deltaF;
  optFilter.f0 = fMin;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for optimal filter...\n");
  }

  /* allocate memory for optimal filter */
  optFilter.data = NULL;
  LAL_CALL( LALCreateVector(&status, &(optFilter.data), filterLength), \
      &status );
  memset( optFilter.data->data, 0, \
      optFilter.data->length * sizeof(*optFilter.data->data));

  /* set optimal filter inputs */
  optFilterIn.overlapReductionFunction = &overlap;
  optFilterIn.omegaGW = &omegaGW;
  optFilterIn.calibratedInverseNoisePSD1 = &calInvPsd1;
  optFilterIn.calibratedInverseNoisePSD2 = &calInvPsd2;

  /* set metadata fields for CC spectrum */
  strncpy(ccSpectrum.name, "ccSpectrum", LALNameLength);
  ccSpectrum.epoch = gpsStartTime;
  ccSpectrum.deltaF = deltaF;
  ccSpectrum.f0 = fMin;

  /* allocate memory for CC spectrum*/
  ccSpectrum.data = NULL;
  LAL_CALL( LALCCreateVector(&status, &(ccSpectrum.data), filterLength), \
      &status );
  memset( ccSpectrum.data->data, 0, \
      ccSpectrum.data->length * sizeof(*ccSpectrum.data->data));

  /* set CC inputs */
  ccIn.hBarTildeOne = &hBarTilde1;
  ccIn.hBarTildeTwo = &hBarTilde2;
  ccIn.responseFunctionOne = &response1;
  ccIn.responseFunctionTwo = &response2;
  ccIn.optimalFilter = &optFilter;

  /*** DONE HERE WITH ALLOCATION ***/

  if (overlap_hann_flag)
  {
    N = 2;
  }
  else
  {
    N = 1;
  }

  for (n = 0; n < N; n++)
  {
    /* initialize parameters for post analysis */
    yOpt = 0.;
    inVarTheoSum = 0.;

    /** read first interval and get response functions **/

    /* lal_errhandler = LAL_ERR_RTRN; */

    /* define segment epoch */
    startTime = startTime + (n * segmentShift);
    gpsStartTime.gpsSeconds = startTime;
    gpsStartPadTime.gpsSeconds = gpsStartTime.gpsSeconds - padData;
    interval1.epoch = interval2.epoch = gpsStartPadTime;

    if (vrbflg)
    {
      fprintf( stdout, "read first interval starting at %d... \n", \
          gpsStartPadTime.gpsSeconds);
    }

    /* read data */
    streamParams.duration = intervalDuration + (2 * padData);
    streamParams.startTime = gpsStartPadTime.gpsSeconds;
    LAL_CALL(readDataPair(&status, &streamPair, &streamParams), &status);

    /* skip segment if data not found or corrupted with 0 values */
    /*
    if ((status.statusCode != 0) || (interval1.data == NULL) \
        || (interval2.data == NULL))
    {
      clear_status(&status);
      if (jobLoop < (numSegments - 1)) continue;
      else break;
    }
    */

    /* print */
    if (test_flag)
    {
      LALSPrintTimeSeries(&interval1, "interval1_0.dat");
      LALSPrintTimeSeries(&interval2, "interval2_0.dat");
    }

    if (vrbflg)
    {
      fprintf( stdout, "split into %d segments\n", numSegments);
    }

    /* split into segments and store */
    for (segLoop = 0; segLoop < numSegments; segLoop++)
    {
      for (i = 0; i < segmentPadLength; i ++)
      {
        segPad1[segLoop]->data[i] = 
          interval1.data->data[i + segLoop * segmentLength];
        segPad2[segLoop]->data[i] = 
          interval2.data->data[i + segLoop * segmentLength];
      }
    }

    /* compute response functions */

    if (vrbflg)
    {
      fprintf(stdout, "Getting appropriate frequency response functions " \
          "for segments of the first interval...\n");
    }

    for (segLoop = 0; segLoop < numSegments; segLoop++)
    {
      gpsCalibTime.gpsSeconds = startTime + \
        (segLoop * segmentDuration) + calibOffset;
      responseTemp1.epoch = responseTemp2.epoch = gpsCalibTime;

      if (vrbflg)
      {
        fprintf(stdout, "request GPS time %d\n", gpsCalibTime.gpsSeconds);
      }

      calfacts.ifo = ifo1;
      LAL_CALL( LALExtractFrameResponse(&status, &responseTemp1, calCache1, \
            &calfacts), &status );
      calfacts.ifo = ifo2;
      LAL_CALL( LALExtractFrameResponse(&status, &responseTemp2, calCache2, \
            &calfacts), &status );

      /* exit loop if calibration not found */
      /*
      if ((status.statusCode !=0 ) || (responseTemp1.data == NULL))
      {
        break;
      }
      if ((status.statusCode !=0 ) || (responseTemp2.data == NULL))
      {
        break;
      }
      */

      /* reduce to the optimal filter frequency range */
      if (vrbflg)
      {
        fprintf(stdout, "reduce to optimal filter range\n");
      }

      response1.epoch = response2.epoch = gpsCalibTime;
      for (i = 0; i < filterLength; i++)
      {
        response1.data->data[i] = responseTemp1.data->data[i + numPointInf];
        response2.data->data[i] = responseTemp2.data->data[i + numPointInf];
      }

      /* print */
      if ((test_flag) && (segLoop == testSeg))
      {
        LALCPrintFrequencySeries(&response1, "response1_0.dat");
        LALCPrintFrequencySeries(&response2, "response2_0.dat");
      }

      /* store in memory */
      for (i = 0; i < filterLength; i++)
      {
        resp1[segLoop]->data[i] = response1.data->data[i];
        resp2[segLoop]->data[i] = response2.data->data[i];
      }

      /* convert response function for use in the MC routine */
      if (inject_flag)
      {
        MCresponse1.epoch = MCresponse2.epoch = gpsCalibTime;

        LAL_CALL( LALResponseConvert(&status, &MCresponse1, \
              &responseTemp1), &status );
        LAL_CALL( LALResponseConvert(&status, &MCresponse2, \
              &responseTemp2), &status );
        calfacts.ifo = ifo1;
        LAL_CALL( LALExtractFrameResponse(&status, &MCresponse1, calCache1, \
              &calfacts), &status );
        calfacts.ifo = ifo2;
        LAL_CALL( LALExtractFrameResponse(&status, &MCresponse2, calCache2, \
              &calfacts), &status );

        /* force DC to be 0 and nyquist to be real */
        MCresponse1.data->data[0].re = MCresponse2.data->data[0].re = 0.;
        MCresponse1.data->data[0].im = MCresponse2.data->data[0].im = 0.;
        MCresponse1.data->data[MCfreqLength-1].im = 0.;
        MCresponse2.data->data[MCfreqLength-1].im = 0.;

        /* print */
        if ((test_flag) && (segLoop == testSeg))
        {
          LALCPrintFrequencySeries(&MCresponse1, "MCresponse1_0.dat");
          LALCPrintFrequencySeries(&MCresponse2, "MCresponse2_0.dat");
        }

        /* store in memory */
        for (i = 0; i < MCfreqLength; i++)
        {
          MCresp1[segLoop]->data[i] = MCresponse1.data->data[i];
          MCresp2[segLoop]->data[i] = MCresponse2.data->data[i];
        }
      }
    }

    for (MCLoop = 0; MCLoop < NLoop; MCLoop ++)
    {
      /* initialize parameters for post analysis */
      yOpt = 0.;
      inVarTheoSum = 0.;

      /* open output file */
      LALSnprintf(outputFilename1, LALNameLength, \
          "%s/stat-%s%s-%lld-%lld-%d.dat", outputFilePath, ifo1, ifo2, \
          startTime, stopTime, MCLoop);
      /* open output file */
      LALSnprintf( outputFilename2, LALNameLength, \
          "%s/post-%s%s-%lld-%lld-%d.dat", outputFilePath, ifo1, ifo2, \
          startTime, stopTime, MCLoop);

      /* set stream data structures to read data*/
      streamParams.duration = segmentDuration + 2 * padData;
      streamPair.stream1 = &segmentPad1;
      streamPair.stream2 = &segmentPad2;

      if (vrbflg)
      {
        fprintf(stdout, "loop over %d jobs\n", numJobs);
      }

      /* loop over jobs (or segments to analyse) */
      /* need more work to include overlapping Hann */

      for (jobLoop = 0; jobLoop < numJobs; jobLoop++)
      {
        /* initialize average PSDs */
        for (i = 0; i < filterLength; i++)
        {
          calPsd1->data[i] = 0.;calPsd2->data[i] = 0.;
        }

        if (jobLoop!=0)
        {
          if (vrbflg)
          {
            fprintf( stdout, "shift segments\n");
          }

          for (segLoop = 0; segLoop < numSegments - 1; segLoop++)
          {
            for (i = 0; i < segmentPadLength; i++)
            {
              segPad1[segLoop]->data[i] = segPad1[segLoop + 1]->data[i];
              segPad2[segLoop]->data[i] = segPad2[segLoop + 1]->data[i];
            }
            for (i = 0; i < filterLength; i++)
            {
              resp1[segLoop]->data[i] = resp1[segLoop + 1]->data[i];
              resp2[segLoop]->data[i] = resp2[segLoop + 1]->data[i];

              if (inject_flag)
              {
                MCresp1[segLoop]->data[i] = MCresp1[segLoop + 1]->data[i];
                MCresp2[segLoop]->data[i] = MCresp2[segLoop + 1]->data[i];
              }
            }
          }

          /* read extra segment */
          gpsStartTime.gpsSeconds = startTime + \
            (jobLoop + numSegments - 1) * segmentDuration;
          gpsStartPadTime.gpsSeconds = gpsStartTime.gpsSeconds - padData;
          streamParams.startTime = gpsStartPadTime.gpsSeconds;

          if (vrbflg)
          {
            fprintf( stdout, "read end segment at GPS %d... \n", \
                gpsStartPadTime.gpsSeconds);
          }

          LAL_CALL(readDataPair(&status, &streamPair, &streamParams), &status);

          /* store in memory */
          for (i = 0; i < segmentPadLength; i++)
          {
            segPad1[numSegments-1]->data[i] = segmentPad1.data->data[i];
            segPad2[numSegments-1]->data[i] = segmentPad2.data->data[i];
          }

          /* compute extra response function */
          gpsCalibTime.gpsSeconds = gpsStartTime.gpsSeconds + calibOffset;
          responseTemp1.epoch = responseTemp2.epoch = gpsCalibTime;

          if (vrbflg)
          {
            fprintf( stdout, "compute end response function at GPS %d...\n", \
                gpsCalibTime.gpsSeconds);
          }

          calfacts.ifo = ifo1;
          LAL_CALL( LALExtractFrameResponse(&status, &responseTemp1, \
                calCache1, &calfacts), &status );
          calfacts.ifo = ifo2;
          LAL_CALL( LALExtractFrameResponse(&status, &responseTemp2, \
                calCache2, &calfacts), &status );

          /* reduce to the optimal filter frequency range */
          response1.epoch = response2.epoch = gpsCalibTime;
          for (i = 0; i < filterLength; i++)
          {
            response1.data->data[i] = responseTemp1.data->data[i + numPointInf];
            response2.data->data[i] = responseTemp2.data->data[i + numPointInf];
          }

          /* print */
          if ((test_flag) && (segLoop == testSeg) && (jobLoop == testInter))
          {
            LALCPrintFrequencySeries(&response1, "response1.dat");
            LALCPrintFrequencySeries(&response2, "response2.dat");
          }

          /* store in memory */
          for (i = 0; i < filterLength ; i++)
          {
            resp1[numSegments-1]->data[i] = response1.data->data[i];
            resp2[numSegments-1]->data[i] = response2.data->data[i];
          }

          /* convert response function for use in the MC routine */
          if (inject_flag)
          {
            MCresponse1.epoch = MCresponse2.epoch = gpsCalibTime;

            LAL_CALL( LALResponseConvert(&status, &MCresponse1,
                  &responseTemp1), &status );
            LAL_CALL( LALResponseConvert(&status, &MCresponse2,
                  &responseTemp2), &status );
            calfacts.ifo = ifo1;
            LAL_CALL( LALExtractFrameResponse(&status, &MCresponse1, \
                  calCache1, &calfacts), &status );
            calfacts.ifo = ifo2;
            LAL_CALL( LALExtractFrameResponse(&status, &MCresponse2, \
                  calCache2, &calfacts), &status );

            /* force DC to be 0 and nyquist to be real */
            MCresponse1.data->data[0].re = MCresponse2.data->data[0].re = 0.;
            MCresponse1.data->data[0].im = MCresponse2.data->data[0].im = 0.;
            MCresponse1.data->data[MCfreqLength-1].im = 0.;
            MCresponse2.data->data[MCfreqLength-1].im = 0.;

            /* store in memory */
            for (i = 0; i < MCfreqLength ; i++)
            {
              MCresp1[numSegments]->data[i] = MCresponse1.data->data[i];
              MCresp2[numSegments]->data[i] = MCresponse2.data->data[i];
            }
          }
        }

        for (segLoop = 0; segLoop < numSegments; segLoop++)
        {
          gpsStartTime.gpsSeconds = startTime +
            (jobLoop + segLoop) * segmentDuration;

          for (i = 0; i < filterLength; i++)
          {
            response1.data->data[i] = resp1[segLoop]->data[i];
            response2.data->data[i] = resp2[segLoop]->data[i];
          }

          /* simulate signal */
          if (inject_flag)
          {
            for (i = 0; i < MCfreqLength; i++)
            {
              MCresponse1.data->data[i] = MCresp1[segLoop]->data[i];
              MCresponse2.data->data[i] = MCresp2[segLoop]->data[i];
            }

            /* set parameters for monte carlo */
            SimStochBG1.epoch = SimStochBG2.epoch = gpsStartPadTime;
            SBParams.seed = seed;

            /* define input structure for SimulateSB */
            SBInput.omegaGW = &MComegaGW;
            SBInput.whiteningFilter1 = &MCresponse1;
            SBInput.whiteningFilter2 = &MCresponse2;

            /* define output structure for SimulateSB */
            SBOutput.SSimStochBG1 = &SimStochBG1;
            SBOutput.SSimStochBG2 = &SimStochBG2;

            /* perform monte carlo */
            LALSSSimStochBGTimeSeries(&status, &SBOutput, &SBInput, &SBParams);

            /* print */
            if ((test_flag) && (jobLoop == testInter) && \
                (segLoop == testSeg) && (MCLoop == testTrial))
            {
              LALSPrintTimeSeries(&SimStochBG1, "simStochBG1.dat");
              LALSPrintTimeSeries(&SimStochBG2, "simStochBG2.dat");
            }

            /* multiply by scale factor and inject into real data */
            for (i = 0; i < segmentPadLength; i++)
            {
              segmentPad1.data->data[i] = segPad1[segLoop]->data[i] + \
                (scaleFactor * SimStochBG1.data->data[i]);
              segmentPad2.data->data[i] = segPad2[segLoop]->data[i] + \
                (scaleFactor * SimStochBG2.data->data[i]);
            }

            /* increase seed */
            seed = seed + 2;
          }
          else
          {
            for (i = 0; i < segmentPadLength; i++)
            {
              segmentPad1.data->data[i] = segPad1[segLoop]->data[i];
              segmentPad2.data->data[i] = segPad2[segLoop]->data[i];
            }
          }

          /* print */
          if ((test_flag) && (jobLoop == testInter) &&
              (segLoop == testSeg) && (MCLoop == testTrial))
          {
            LALSPrintTimeSeries(&segmentPad1, "segmentPad1.dat");
            LALSPrintTimeSeries(&segmentPad2, "segmentPad2.dat");
          }

          /* high pass fitering */
          if (high_pass_flag)
          {
            LAL_CALL( LALButterworthREAL4TimeSeries( &status, &segmentPad1, \
                  &highpassParam ), &status );
            LAL_CALL( LALButterworthREAL4TimeSeries( &status, &segmentPad2, \
                  &highpassParam ), &status );
          }

          /* throw away pad data on each side of the segment */
          for (i = 0; i < segmentLength; i ++)
          {
            segment1.data->data[i] = \
              segmentPad1.data->data[i + (padData * resampleRate)];
            segment2.data->data[i] = \
              segmentPad2.data->data[i + (padData * resampleRate)];
          }

          /* print */
          if ((test_flag) && (jobLoop == testInter) && \
              (segLoop == testSeg) && (MCLoop == testTrial))
          {
            LALSPrintTimeSeries(&segment1, "segment1.dat");
            LALSPrintTimeSeries(&segment2, "segment2.dat");
          }

          /* store in memory */
          for (i = 0; i < segmentLength; i ++)
          {
            seg1[segLoop]->data[i] = segment1.data->data[i];
            seg2[segLoop]->data[i] = segment2.data->data[i];
          }

          if (segLoop != segMiddle)
          {
            if (vrbflg)
            {
              fprintf(stdout, "Estimating PSDs...\n");
            }

            /* compute uncalibrated PSDs */
            LAL_CALL( LALREAL4AverageSpectrum(&status, &psdTemp1, &segment1, \
                  &specparPSD), &status );
            LAL_CALL( LALREAL4AverageSpectrum(&status, &psdTemp2, &segment2, \
                  &specparPSD), &status );

            if (vrbflg)
            {
              fprintf(stdout, "Getting appropriate frequency band "
                  "for PSDs..\n");
            }

            /* reduce to the optimal filter frequency range */
            for (i = 0; i < filterLength; i++)
            {
              psd1.data->data[i] = psdTemp1.data->data[i + numPointInf];
              psd2.data->data[i] = psdTemp2.data->data[i + numPointInf];
            }

            /* print */
            if ((test_flag) && (jobLoop == testInter) && \
                (segLoop == testSeg) && (MCLoop == testTrial))
            {
              LALSPrintFrequencySeries(&psd1, "psd1.dat");
              LALSPrintFrequencySeries(&psd2, "psd2.dat");
            }

            if (vrbflg)
            {
              fprintf(stdout, "Generating inverse noise...\n");
            }

            /* compute inverse calibrate PSDs */
            LAL_CALL( LALStochasticInverseNoiseCal(&status, \
                  &inverseNoiseOut1, &inverseNoiseIn1), &status );
            LAL_CALL( LALStochasticInverseNoiseCal(&status, \
                  &inverseNoiseOut2, &inverseNoiseIn2), &status );

            /* print */
            if ((test_flag) && (jobLoop == testInter) && \
                (segLoop == testSeg) && (MCLoop == testTrial))
            {
              LALSPrintFrequencySeries(&calInvPsd1, "calInvPsd1.dat");
              LALSPrintFrequencySeries(&calInvPsd2, "calInvPsd2.dat");
            }

            /* sum over calibrated PSDs for average */
            for (i = 0; i < filterLength; i++)
            {
              calPsd1->data[i] = calPsd1->data[i] + 1./calInvPsd1.data->data[i];
              calPsd2->data[i] = calPsd2->data[i] + 1./calInvPsd2.data->data[i];
            }
          }
        }

        /* average calibrated PSDs and take inverse */
        for (i = 0; i < filterLength; i++)
        {
          calPsd1->data[i] = calPsd1->data[i] / (REAL4)(numSegments-1);
          calPsd2->data[i] = calPsd2->data[i] / (REAL4)(numSegments-1);
          calInvPsd1.data->data[i] = 1./calPsd1->data[i];
          calInvPsd2.data->data[i] = 1./calPsd2->data[i];
        }

        /* print */
        if ((test_flag) && (jobLoop == testInter) && (MCLoop == testTrial))
        {
          LALSPrintFrequencySeries(&calInvPsd1, "calInvPsdAvg1.dat");
          LALSPrintFrequencySeries(&calInvPsd2, "calInvPsdAvg2.dat");
        }

        if (vrbflg)
        {
          fprintf(stdout, "Normalising optimal filter...\n");
        }

        /* compute variance and normalisation for optimal filter */
        LAL_CALL( LALStochasticOptimalFilterNormalization(&status, \
              &normOutput, &normInput, &normParams), &status );
        lambda = (REAL8)(normLambda.value * \
            pow(10.,normLambda.units.powerOfTen));
        varTheo = (REAL8)(segmentDuration * normSigma.value * \
            pow(10.,normSigma.units.powerOfTen));

        if (vrbflg)
        {
          fprintf(stdout, "Generating optimal filter...\n");
        }

        /* build optimal filter */
        optFilter.epoch = gpsStartTime;
        LAL_CALL( LALStochasticOptimalFilterCal(&status, &optFilter, \
              &optFilterIn, &normLambda), &status );

        /* print */
        if ((test_flag) && (jobLoop == testInter) && (MCLoop == testTrial))
        {
          LALPrintFrequencySeries(&optFilter, "optFilter.dat");
        }

        /* save */
        if (vrbflg)
        {
          fprintf(stdout, "job %d:", jobLoop);
          fprintf(stdout, "varTheo = %e s\n", varTheo);
        }

        gpsStartTime.gpsSeconds = startTime + \
          (jobLoop + segMiddle) * segmentDuration;

        /* analyse middle segment */
        if (vrbflg)
        {
          fprintf(stdout, "analysing segment at GPS %d\n", \
              gpsStartTime.gpsSeconds);
        }

        for (i = 0; i < segmentLength; i++)
        {
          segment1.data->data[i] = seg1[segMiddle]->data[i];
          segment2.data->data[i] = seg2[segMiddle]->data[i];
        }

        /* print */
        if ((test_flag) && (jobLoop == testInter) && (MCLoop == testTrial))
        {
          LALSPrintTimeSeries(&segment1, "segmentMiddle1.dat");
          LALSPrintTimeSeries(&segment2, "segmentMiddle2.dat");
        }

        /* zero pad and fft */
        LAL_CALL( LALSZeroPadAndFFT(&status, &hBarTilde1, &segment1, \
              &zeroPadParams), &status );
        LAL_CALL( LALSZeroPadAndFFT(&status, &hBarTilde2, &segment2, \
              &zeroPadParams), &status );

        /* print */
        if ((test_flag) && (jobLoop == testInter) && \
            (segLoop == testSeg) && (MCLoop == testTrial))
        {
          LALCPrintFrequencySeries(&hBarTilde1, "hBarTilde1.dat");
          LALCPrintFrequencySeries(&hBarTilde2, "hBarTilde2.dat");
        }

        if (vrbflg)
        {
          fprintf(stdout, "Generating cross correlation spectrum...\n");
        }

        /* cc spectrum */
        for (i = 0; i < filterLength; i++)
        {
          response1.data->data[i] = resp1[segMiddle]->data[i];
          response2.data->data[i] = resp2[segMiddle]->data[i];
        }

        LAL_CALL( LALStochasticCrossCorrelationSpectrumCal(&status, \
              &ccSpectrum, &ccIn, epochsMatch), &status );

        /* save */
        if (vrbflg)
        {
          LALCPrintFrequencySeries(&ccSpectrum, "ccSpectrum.dat");
        }

        /* cc statistic */
        LAL_CALL( LALStochasticCrossCorrelationStatisticCal(&status, &ccStat, \
              &ccIn,epochsMatch), &status );

        /* print */
        if ((test_flag) && (jobLoop == testInter) && (MCLoop == testTrial))
        {
          LALCPrintFrequencySeries(&ccSpectrum, "ccSpectrum.dat");
        }

        y = (REAL8)(ccStat.value * pow(10.,ccStat.units.powerOfTen));

        if (vrbflg)
        {
          fprintf(stdout, "y = %e\n", y);
        }

        if (post_analysis_flag)
        {
          yOpt = yOpt + (y/varTheo);
          inVarTheoSum = inVarTheoSum + (1./varTheo);
        }

        /* output to file */
        out1 = fopen(outputFilename1, "a");
        fprintf(out1,"%d %e %e\n", gpsStartTime.gpsSeconds, y, varTheo);
        fclose(out1);
      }
    }

    lal_errhandler = LAL_ERR_EXIT;
    if (post_analysis_flag)
    {
      ptEst = (yOpt / inVarTheoSum) /(REAL8)segmentDuration;
      error = sqrt(1./inVarTheoSum) /(REAL8)segmentDuration;
      out2 = fopen(outputFilename2, "a");
      fprintf(out2,"%lld %lld %e %e\n", startTime, stopTime, ptEst, error);
      fclose(out2);
      if (vrbflg)
      {
        fprintf(stdout,"ptEst = %e error = %e\n", ptEst, error);
      }
    }
  }

  /* cleanup */
  LAL_CALL( LALDestroyRealFFTPlan(&status, &(specparPSD.plan)), &status );
  LAL_CALL( LALDestroyRealFFTPlan(&status, &fftDataPlan), &status );
  LAL_CALL( LALDestroyVector(&status, &(interval1.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(interval2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(segmentPad1.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(segmentPad2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(segment1.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(segment2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(psdTemp1.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(psdTemp2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(psd1.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(psd2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(calPsd1)), &status );
  LAL_CALL( LALDestroyVector(&status, &(calPsd2)), &status );
  LAL_CALL( LALCDestroyVector(&status, &(responseTemp1.data)), &status );
  LAL_CALL( LALCDestroyVector(&status, &(responseTemp2.data)), &status );
  LAL_CALL( LALCDestroyVector(&status, &(response1.data)), &status );
  LAL_CALL( LALCDestroyVector(&status, &(response2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(optFilter.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(calInvPsd1.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(calInvPsd2.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(overlap.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(omegaGW.data)), &status );
  LAL_CALL( LALDestroyVector(&status, &(dataWindow.data)), &status );
  if (apply_mask_flag)
  {
    LAL_CALL( LALDestroyVector(&status, &(mask.data)), &status );
    LAL_CALL( LALDestroyVector(&status, &maskTemp), &status );
  }
  if (hannDuration != 0)
  {
    LAL_CALL( LALDestroyVector(&status, &hannWindow), &status );
  }
  LAL_CALL( LALCDestroyVector(&status, &(hBarTilde1.data)), &status );
  LAL_CALL( LALCDestroyVector(&status, &(hBarTilde2.data)), &status );
  if (inject_flag)
  {
    LAL_CALL( LALDestroyVector(&status, &SimStochBG1.data), &status );
    LAL_CALL( LALDestroyVector(&status, &SimStochBG2.data), &status );
    LAL_CALL( LALCDestroyVector(&status, &(MCresponse1.data)), &status );
    LAL_CALL( LALCDestroyVector(&status, &(MCresponse2.data)), &status );
    LAL_CALL( LALDestroyVector(&status, &(MComegaGW.data)), &status );
  }
  /*
  for (i = 0; i <numSegments; i++)
  {
    LAL_CALL( LALCDestroyVector(&status, &(resp1[i])), &status );
    LAL_CALL( LALCDestroyVector(&status, &(resp2[i])), &status );
    LAL_CALL( LALDestroyVector(&status, &(segPad1[i])), &status );
    LAL_CALL( LALDestroyVector(&status, &(segPad2[i])), &status );
    LAL_CALL( LALDestroyVector(&status, &(seg1[i])), &status );
    LAL_CALL( LALDestroyVector(&status, &(seg2[i])), &status );
    if (inject_flag)
    {
      LAL_CALL( LALCDestroyVector(&status, &(MCresp1[i])), &status );
      LAL_CALL( LALCDestroyVector(&status, &(MCresp2[i])), &status );
    }
  }
  */

  return 0;
}

#define USAGE \
  "Usage: lalapps_stochastic [options]\n"\
  " -h, --help                          print this message\n"\
  " -v, --version                       display version\n"\
  " --verbose                           verbose mode\n"\
  " --post-analysis                     post analysis\n"\
  " -z, --debug-level N                 set lalDebugLevel\n"\
  " -t, --gps-start-time N              GPS start time\n"\
  " -T, --gps-stop-time N               GPS stop time\n"\
  " -L, --interval-duration N           interval duration\n"\
  " -l, --segment-duration N            segment duration\n"\
  " -A, --sample-rate N                 sample rate\n"\
  " -a, --resample-rate N               resample rate\n"\
  " -f, --f-min N                       minimum frequency\n"\
  " -F, --f-max N                       maximum frequency\n"\
  " --high-pass-filter                  apply high pass filter\n"\
  " -k, --hpf-frequency N               high pass filter knee frequency\n"\
  " -p, --hpf-attenuation N             high pass filter attenuation\n"\
  " -P, --hpf-order N                   high pass filter order\n"\
  " --overlap-hann                      use overlap window\n"\
  " -w, --hann-duration N               hann duration\n"\
  " -i, --ifo-one IFO                   ifo for first stream\n"\
  " -I, --ifo-two IFO                   ifo for second stream\n"\
  " -d, --frame-cache-one FILE          cache file for first stream\n"\
  " -D, --frame-cache-two FILE          cache file for second stream\n"\
  " -r, --calibration-cache-one FILE    first stream calibration cache\n"\
  " -R, --calibration-cache-two FILE    second stream calibration cache\n"\
  " -c, --calibration-offset N          offset for calibration time\n"\
  " --apply-mask                        apply frequency masking\n"\
  " -b, --mask-bin N                    number of bin for frequency mask\n"\
  " --inject                            inject a signal into the data\n"\
  " -o, --scale-factor N                scale factor for injection\n"\
  " -g, --seed N                        seed for injections\n"\
  " -N, --trials N                      number of trial for MC\n"\
  " -S, --output-dir DIR                directory for output files\n"\
  " --test                              print intermediate results\n"\
  " -U, --test-interval N               interval number for test\n"\
  " -V, --test-segment N                segment number test\n"\
  " -W, --test-trial N                  trial number for test\n"\
  " -z, --debug-level N                 debugging level\n"\
  "\n"

/* parse command line options */
void parseOptions(INT4 argc, CHAR *argv[])
{
  int c = -1;

  while(1)
  {
    static struct option long_options[] =
    {
      /* options that set a flag */
      {"inject", no_argument, &inject_flag, 1},
      {"apply-mask", no_argument, &apply_mask_flag, 1},
      {"high-pass-filter", no_argument, &high_pass_flag, 1},
      {"overlap-hann", no_argument, &overlap_hann_flag, 1},
      {"post-analysis", no_argument, &post_analysis_flag,1},
      {"verbose", no_argument, &vrbflg, 1},
      {"test", no_argument, &test_flag, 1},
      /* options that don't set a flag */
      {"help", no_argument, 0, 'h'},
      {"gps-start-time", required_argument, 0, 't'},
      {"gps-end-time", required_argument, 0, 'T'},
      {"interval-duration", required_argument, 0, 'L'},
      {"segment-duration", required_argument, 0, 'l'},
      {"sample-rate", required_argument, 0, 'A'},
      {"resample-rate", required_argument, 0, 'a'},
      {"f-min", required_argument, 0, 'f'},
      {"f-max", required_argument, 0, 'F'},
      {"hann-duration", required_argument, 0, 'w'},
      {"hpf-frequency", required_argument, 0, 'k'},
      {"hpf-attenuation", required_argument, 0, 'p'},
      {"hpf-order", required_argument, 0, 'P'},
      {"ifo-one", required_argument, 0, 'i'},
      {"ifo-two", required_argument, 0, 'I'},
      {"frame-cache-one", required_argument, 0, 'd'},
      {"frame-cache-two", required_argument, 0, 'D'},
      {"calibration-cache-one", required_argument, 0, 'r'},
      {"calibration-cache-two", required_argument, 0, 'R'},
      {"calibration-offset", required_argument, 0, 'c'},
      {"mask-bin", required_argument, 0, 'b'},
      {"scale-factor", required_argument, 0, 'o'},
      {"seed", required_argument, 0, 'g'},
      {"trials", required_argument, 0, 'N'},
      {"output-dir", required_argument, 0, 'S'},
      {"test-interval", required_argument, 0, 'U'},
      {"test-segment", required_argument, 0, 'V'},
      {"test-trial", required_argument, 0, 'W'},
      {"debug-level", required_argument, 0, 'z'},
      {"version", no_argument, 0, 'v'},
      {0, 0, 0, 0}
    };

    /* getopt_long stores the option here */
    int option_index = 0;
    size_t optarg_len;

    c = getopt_long(argc, argv, \
        "ht:T:L:l:A:a:f:F:w:k:p:P:i:I:d:D:r:R:c:b:o:g:N:S:U:V:W:z:v", \
        long_options, &option_index);

    if (c == -1)
    {
      /* end of options, break loop */
      break;
    }

    switch(c)
    {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        printf ("option %s", long_options[option_index].name);
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;

      case 'h':
        /* HELP!!! */
        fprintf(stdout, USAGE);
        exit(0);
        break;

      case 't':
        /* start time */
        startTime = atoi(optarg);
        break;

      case 'T':
        /* stop time */
        stopTime = atoi(optarg);
        break;

      case 'L':
        /* duration */
        intervalDuration = atoi(optarg);
        break;

      case 'l':
        /* duration */
        segmentDuration = atoi(optarg);
        break;

      case 'A':
        /* sample rate */
        sampleRate = atoi(optarg);
        break;

      case 'a':
        /* resampling */
        resampleRate = atoi(optarg);
        break;

      case 'f':
        /* minimal frequency */
        fMin = atoi(optarg);
        break;

      case 'F':
        /* maximal frequency */
        fMax = atoi(optarg);
        break;

      case 'w':
        /* hann window duration */
        hannDuration = atoi(optarg);
        break;

      case 'k':
        /* high pass knee filter frequency */
        highPassFreq= atof(optarg);
        break;

      case 'p':
        /* high pass filter attenuation */
        highPassAt = atof(optarg);
        break;

      case 'P':
        /* high pass filter order */
        highPassOrder = atoi(optarg);
        break;

      case 'i':
        /* ifo for first stream */
        strncpy(ifo1, optarg, LALNameLength);

        /* set site and channel */
        if (strncmp(ifo1, "H1", 2) == 0)
        {
          site1 = 0;
          strncpy(channel1, "H1:LSC-AS_Q", LALNameLength);
        }
        else if (strncmp(ifo1, "H2", 2) == 0)
        {
          site1 = 0;
          strncpy(channel1, "H2:LSC-AS_Q", LALNameLength);
        }
        else if (strncmp(ifo1, "L1", 2) == 0)
        {
          site1 = 1;
          strncpy(channel1, "L1:LSC-AS_Q", LALNameLength);
        }
        else
        {
          fprintf(stderr, "First IFO not recognised...\n");
          exit(1);
        }

        break;

      case 'I':
        /* ifo for second stream */
        strncpy(ifo2, optarg, LALNameLength);

        /* set site and channel */
        if (strncmp(ifo2, "H1", 2) == 0)
        {
          site2 = 0;
          strncpy(channel2, "H1:LSC-AS_Q", LALNameLength);
        }
        else if (strncmp(ifo2, "H2", 2) == 0)
        {
          site2 = 0;
          strncpy(channel2, "H2:LSC-AS_Q", LALNameLength);
        }
        else if (strncmp(ifo2, "L1", 2) == 0)
        {
          site2 = 1;
          strncpy(channel2, "L1:LSC-AS_Q", LALNameLength);
        }
        else
        {
          fprintf(stderr, "Second IFO not recognised...\n");
          exit(1);
        }

        break;

      case 'd':
        /* data cache one */
        optarg_len = strlen(optarg) + 1;
        frameCache1 = (CHAR*)calloc(optarg_len, sizeof(CHAR));
        memcpy(frameCache1, optarg, optarg_len);
        break;

      case 'D':
        /* data cache two */
        optarg_len = strlen(optarg) + 1;
        frameCache2 = (CHAR*)calloc(optarg_len, sizeof(CHAR));
        memcpy(frameCache2, optarg, optarg_len);
        break;

      case 'r':
        /* calibration cache one */
        optarg_len = strlen(optarg) + 1;
        calCache1 = (CHAR*)calloc(optarg_len, sizeof(CHAR));
        memcpy(calCache1, optarg, optarg_len);
        break;

      case 'R':
        /* calibration cache two */
        optarg_len = strlen(optarg) + 1;
        calCache2 = (CHAR*)calloc(optarg_len, sizeof(CHAR));
        memcpy(calCache2, optarg, optarg_len);
        break;

      case 'c':
        /* calibration time offset */
        calibOffset = atoi(optarg);
        break;

      case 'b':
        /* bin for frequency mask */
        maskBin = atoi(optarg);
        break;

      case 'o':
        /* scale factor */
        scaleFactor = atof(optarg);
        break;

      case 'g':
        /* seed */
        seed = atoi(optarg);
        break;

      case 'N':
        /* number of injection */
        NLoop = atoi(optarg);
        break;

      case 'S':
        /* directory for output files */
        strncpy(outputFilePath, optarg, LALNameLength);
        break;

      case 'U':
        /* interval number for test */
        testInter = atoi(optarg);
        break;

      case 'V':
        /* segment number for test */
        testSeg = atoi(optarg);
        break;

      case 'W':
        /* trial  number for test */
        testTrial = atoi(optarg);
        break;

      case 'z':
        /* set debug level */
        set_debug_level( optarg );
        break;

      case 'v':
        /* display version info and exit */
        fprintf(stdout, "Standalone SGWB Search Engine\n" CVS_ID "\n");
        exit(0);
        break;

      case '?':
        exit(1);
        break;

      default:
        fprintf(stderr, "unknown error while parsing options\n");
        exit(1);
    }
  }

  if (optind < argc)
  {
    fprintf(stderr, "extraneous command line arguments:\n");
    while(optind < argc)
    {
      fprintf(stderr, "%s\n", argv[optind++]);
    }
    exit(1);
  }

  return;
}

/* function to read data from frames */
void readDataPair(LALStatus *status,
    StreamPair *streamPair,
    ReadDataPairParams *params)
{
  /* counters */
  INT4 i;

  /* variables */
  FrCache *frCache1 = NULL;
  FrStream *frStream1 = NULL;
  FrCache *frCache2 = NULL;
  FrStream *frStream2 = NULL;
  FrChanIn frChanIn1, frChanIn2;
  REAL4TimeSeries dataStream1;
  REAL4TimeSeries dataStream2;
  ResampleTSParams resampleParams;
  LIGOTimeGPS bufferStartTime;
  UINT8 startTime;
  INT4 buffer;
  INT4 resampleRate;
  INT4 sampleRate;

  /* read parameters */
  startTime = params->startTime;
  buffer = params->buffer;
  resampleRate = params->resampleRate;
  sampleRate = params->sampleRate;

  /* initialise status pointer */
  INITSTATUS( status, "readDataPair", STOCHASTICC );
  ATTATCHSTATUSPTR( status );

  /* buffer start time */
  bufferStartTime.gpsSeconds = startTime - buffer;
  bufferStartTime.gpsNanoSeconds = 0;

  /* set channels */
  frChanIn1.name = params->channel1;
  frChanIn2.name = params->channel2;
  frChanIn2.type = ADCDataChannel;
  frChanIn1.type = ADCDataChannel;

  /* initial data structures */
  dataStream1.epoch =  dataStream2.epoch = bufferStartTime;

  /* set resample parameters */
  resampleParams.deltaT = 1.0 / (REAL8)resampleRate;
  resampleParams.filterType = defaultButterworth;

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for raw data streams...\n");
  }

  /* allocate memory for first raw stream*/
  dataStream1.data = NULL;
  LALSCreateVector( status->statusPtr, &(dataStream1.data), \
      sampleRate * (params->duration + (2 * buffer)));
  CHECKSTATUSPTR (status);
  memset( dataStream1.data->data, 0, \
      dataStream1.data->length * sizeof(*dataStream1.data->data));

  if (vrbflg)
  {
    fprintf(stdout, "Opening first frame cache...\n");
  }

  /* open first frame cache */
  LALFrCacheImport(status->statusPtr, &frCache1, params->frameCache1);
  CHECKSTATUSPTR (status);
  LALFrCacheOpen(status->statusPtr, &frStream1, frCache1);
  CHECKSTATUSPTR (status);

  if (vrbflg)
  {
    fprintf(stdout, "Reading in channel \"%s\"...\n", frChanIn1.name);
  }

  /* read first channel */
  LALFrSeek(status->statusPtr, &(bufferStartTime), frStream1);
  CHECKSTATUSPTR (status);
  LALFrGetREAL4TimeSeries(status->statusPtr, &dataStream1, &frChanIn1, \
      frStream1);
  CHECKSTATUSPTR (status);

  /* resample */
  if (resampleRate != sampleRate)
  {
    if (vrbflg)
    {
      fprintf(stdout, "Resampling to %d Hz...\n", resampleRate);
    }

    /* resample */
    LALResampleREAL4TimeSeries(status->statusPtr, &dataStream1, \
        &resampleParams);
    CHECKSTATUSPTR( status );
  }

  if (vrbflg)
  {
    fprintf(stdout, "Allocating memory for second raw data stream...\n");
  }

  /* allocate memory for second raw stream */
  dataStream2.data = NULL;
  LALSCreateVector(status->statusPtr, &(dataStream2.data), \
      sampleRate * (params->duration + (2 * buffer)));
  CHECKSTATUSPTR( status );
  memset(dataStream2.data->data, 0, \
      dataStream2.data->length * sizeof(*dataStream2.data->data));

  if (strcmp(params->frameCache1, params->frameCache2) == 0)
  {
    if (vrbflg)
    {
      fprintf(stdout, "Reading in channel \"%s\" from same cache...\n", \
        frChanIn2.name);
    }

    /* read in second channel */
    LALFrSeek(status->statusPtr, &(bufferStartTime), frStream1);
    CHECKSTATUSPTR (status);
    LALFrGetREAL4TimeSeries(status->statusPtr, &dataStream2, &frChanIn2, \
        frStream1);
    CHECKSTATUSPTR (status);

    if (vrbflg)
    {
      fprintf(stdout, "Closing frame cache...\n");
    }

    /* close frame cache */
    LALFrClose(status->statusPtr, &frStream1);
    CHECKSTATUSPTR (status);
  }
  else
  {
    if (vrbflg)
    {
      fprintf(stdout, "Closing first frame cache...\n");
    }

    /* close first frame cache */
    LALFrClose(status->statusPtr, &frStream1);
    CHECKSTATUSPTR (status);

    if (vrbflg)
    {
      fprintf(stdout, "Opening second frame cache...\n");
    }

    /* open second frame cache and read in second channel */
    LALFrCacheImport(status->statusPtr, &frCache2, params->frameCache2);
    CHECKSTATUSPTR (status);
    LALFrCacheOpen(status->statusPtr, &frStream2, frCache2);
    CHECKSTATUSPTR (status);

    if (vrbflg)
    {
      fprintf(stdout, "Reading in channel \"%s\"...\n", frChanIn2.name);
    }

    /* read in second channel */
    LALFrSeek(status->statusPtr, &(bufferStartTime), frStream2);
    CHECKSTATUSPTR (status);
    LALFrGetREAL4TimeSeries(status->statusPtr, &dataStream2,
        &frChanIn2, frStream2);
    CHECKSTATUSPTR (status);

    if (vrbflg)
    {
      fprintf(stdout, "Closing second frame cache...\n");
    }

    /* close second frame stream */
    LALFrClose(status->statusPtr, &frStream2);
    CHECKSTATUSPTR (status);
  }

  /* resample */
  if (resampleRate != sampleRate)
  {
    if (vrbflg)
    {
      fprintf(stdout, "Resampling to %d Hz...\n", resampleRate);
    }

    /* resample */
    LALResampleREAL4TimeSeries(status->statusPtr, &dataStream2, \
        &resampleParams);
    CHECKSTATUSPTR (status);
  }

  /* build output */
  strncpy(streamPair->stream1->name,dataStream1.name, LALNameLength);
  strncpy(streamPair->stream2->name,dataStream2.name, LALNameLength);
  streamPair->stream1->epoch.gpsSeconds = startTime;
  streamPair->stream2->epoch.gpsSeconds = startTime;
  streamPair->stream1->epoch.gpsNanoSeconds = 0;
  streamPair->stream2->epoch.gpsNanoSeconds = 0;
  streamPair->stream1->deltaT = 1./(REAL8)resampleRate;
  streamPair->stream2->deltaT = 1./(REAL8)resampleRate;
  streamPair->stream1->f0 = streamPair->stream2->f0 = 0;
  streamPair->stream1->sampleUnits = dataStream1.sampleUnits;
  streamPair->stream2->sampleUnits = dataStream2.sampleUnits;

  /* remove buffer, and hence corruption due to resampling */
  for (i = 0; i < params->duration * resampleRate; i++)
  {
    streamPair->stream1->data->data[i] = \
      dataStream1.data->data[i + (resampleRate * buffer)];
    streamPair->stream2->data->data[i] = \
      dataStream2.data->data[i + (resampleRate * buffer)];
  }

  /* clean up */
  LALSDestroyVector(status->statusPtr, &(dataStream1.data));
  CHECKSTATUSPTR (status);
  LALSDestroyVector(status->statusPtr, &(dataStream2.data));
  CHECKSTATUSPTR (status);

  /* return status */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}

/* 
 * vim: et
 */

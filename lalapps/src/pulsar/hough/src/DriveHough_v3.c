/*
 *  Copyright (C) 2005 Badri Krishnan, Alicia Sintes  
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
 * \file DriveHough_v3.c
 * \author Badri Krishnan, Alicia Sintes 
 * \brief Driver code for performing Hough transform search on non-demodulated
   data.

   Revision: $Id$
 
   History:   Created by Sintes July 04, 2003
   
   This is the main driver for the Hough transform routines.
   It takes as input a set of SFTs and search parameters and
   outputs the number counts using the Hough transform routines 

*/


#include "./DriveHoughColor.h"

#ifdef TIMING
#include "./timer/cycle_counter/Intel/GCC/cycle_counter.h"
#endif

RCSID( "$Id$");


/* ***************************************************************
 * Constant Declarations.  Default parameters.
 *****************************************************************/

extern int lalDebugLevel;

#define EARTHEPHEMERIS "./earth00-04.dat"
#define SUNEPHEMERIS "./sun00-04.dat"

#define ACCURACY 0.00000001 /* of the velocity calculation */
#define MAXFILES 3000 /* maximum number of files to read in a directory */
#define MAXFILENAMELENGTH 256 /* maximum # of characters  of a SFT filename */
#define SFTDIRECTORY "/home/badkri/fakesfts"  
/* #define SFTDIRECTORY "/nfs/morbo/geo600/hannover/sft/S2-LIGO/S2_L1_Funky-v3Calv5DQ30MinSFTs" */
#define DIROUT "./outHM1/"      /* prefix file output */
#define BASENAMEOUT "HM1"
#define FILEVELOCITY "./velocity.data"  /* name: file with time-velocity info */
#define FILETIME "./Ts" /* name: file with timestamps */
#define IFO 2         /*  detector, 1:GEO, 2:LLO, 3:LHO */
#define THRESHOLD 1.6 /* thresold for peak selection, with respect to the
                              the averaged power in the search band */
#define FALSEALARM 0.00000001 /* Hough false alarm for candidate selection */
#define SKYFILE "./sky1"      
#define F0 255.0          /*  frequency to build the LUT and start search */
#define FBAND 0.2          /* search frequency band  (in Hz) */
#define NFSIZE  21 /* n-freq. span of the cylinder, to account for spin-down
                          search */
#define BLOCKSRNGMED 101 /* Running median window size */

#define TRUE (1==1)
#define FALSE (1==0)

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv------------------------------------ */
int main(int argc, char *argv[]){

  /* LALStatus pointer */
  static LALStatus           status;  
  
  /* detector */
  static LALDetector         detector;

  /* time and velocity vectors and files*/
  static LIGOTimeGPSVector   timeV;
  static REAL8Cart3CoorVector velV;
  /*   static REAL8Cart3CoorVector velVmid; */
  static REAL8Vector         timeDiffV;

  /* standard pulsar sft types */ 
  SFTVector *inputSFTs=NULL;

  /* vector of weights */
  REAL8Vector weightsV;

  /* ephemeris */
  EphemerisData    *edat=NULL;

  /* hough structures */
  static HOUGHptfLUTVector   lutV; /* the Look Up Table vector*/
  static HOUGHPeakGramVector pgV;  /* vector of peakgrams */
  static PHMDVectorSequence  phmdVS;  /* the partial Hough map derivatives */
  static UINT8FrequencyIndexVector freqInd; /* for trajectory in time-freq plane */
  static HOUGHResolutionPar parRes;   /* patch grid information */
  static HOUGHPatchGrid  patch;   /* Patch description */ 
  static HOUGHParamPLUT  parLut;  /* parameters needed to build lut  */
  static HOUGHDemodPar   parDem;  /* demodulation parameters or  */
  static HOUGHSizePar    parSize; 
  static HOUGHMapTotal   ht;   /* the total Hough map */
  static UINT4Vector     hist; /* histogram of number counts for a single map */
  static UINT4Vector     histTotal; /* number count histogram for all maps */
  HoughStats      stats;

  /* skypatch info */
  REAL8  *skyAlpha, *skyDelta, *skySizeAlpha, *skySizeDelta; 
  INT4   nSkyPatches, skyCounter=0; 

  /* log file and strings */
  FILE   *fpLog=NULL;
  CHAR   *fnameLog=NULL; 
  CHAR   *logstr=NULL; 
  CHAR   filehisto[256]; 
  CHAR   filestats[256]; 
  CHAR   filestar[256];

  /* miscellaneous */
  INT4   houghThreshold, nfSizeCylinder, iHmap, nSpin1Max;
  /* the maximum number count */
  UINT4  *nStar = NULL; 
  /* where the max occurs */
  REAL8  *freqStar=NULL, *alphaStar=NULL, *deltaStar=NULL, *fdotStar=NULL; 
  UINT4  mObsCoh;
  INT8   f0Bin, fLastBin, fBin;
  REAL8  alpha, delta, timeBase, deltaF, f1jump;
  REAL8  normalizeThr, patchSizeX, patchSizeY;
  UINT2  xSide, ySide;
  UINT2  maxNBins, maxNBorders;
  FILE   *fp1 = NULL;
  FILE   *fpStar = NULL;  

  /* user input variables */
  BOOLEAN uvar_help, uvar_weighAM, uvar_weighNoise;
  INT4 uvar_ifo, uvar_blocksRngMed;
  REAL8 uvar_f0, uvar_peakThreshold, uvar_houghFalseAlarm, uvar_fSearchBand;
  CHAR *uvar_earthEphemeris=NULL;
  CHAR *uvar_sunEphemeris=NULL;
  CHAR *uvar_sftDir=NULL;
  CHAR *uvar_dirnameOut=NULL;
  CHAR *uvar_fbasenameOut=NULL;
  CHAR *uvar_skyfile=NULL;

  /* variables for various preprocessor flags */
#ifdef PRINTEVENTS
  FILE   *fpEvents = NULL;
  CHAR   fileEvents[256];
#endif

#ifdef PRINTTEMPLATES
  FILE *fpTemplates=NULL;
  CHAR fileTemplates[256];
#endif

#ifdef PRINTMAPS
  CHAR fileMaps[256];
#endif

#ifdef TIMING
  unsigned long long start, stop;
#endif
  
#ifdef TIMING
  start = realcc();
#endif

  
  /******************************************************************/
  /*    Set up the default parameters.      */
  /*****************************************************************/

  lalDebugLevel = 0;
  /* LALDebugLevel must be called before anything else */
  SUB( LALGetDebugLevel( &status, argc, argv, 'd'), &status);

  uvar_help = FALSE;
  uvar_weighAM = FALSE;
  uvar_weighNoise = FALSE;
  uvar_ifo = IFO;
  uvar_blocksRngMed = BLOCKSRNGMED;
  uvar_f0 = F0;
  uvar_fSearchBand = FBAND;
  uvar_peakThreshold = THRESHOLD;
  uvar_houghFalseAlarm = FALSEALARM;
  
  uvar_earthEphemeris = (CHAR *)LALMalloc(512*sizeof(CHAR));
  strcpy(uvar_earthEphemeris,EARTHEPHEMERIS);

  uvar_sunEphemeris = (CHAR *)LALMalloc(512*sizeof(CHAR));
  strcpy(uvar_sunEphemeris,SUNEPHEMERIS);

  uvar_sftDir = (CHAR *)LALMalloc(512*sizeof(CHAR));
  strcpy(uvar_sftDir,SFTDIRECTORY);

  uvar_dirnameOut = (CHAR *)LALMalloc(512*sizeof(CHAR));
  strcpy(uvar_dirnameOut,DIROUT);

  uvar_fbasenameOut = (CHAR *)LALMalloc(512*sizeof(CHAR));
  strcpy(uvar_fbasenameOut,BASENAMEOUT);

  uvar_skyfile = (CHAR *)LALMalloc(512*sizeof(CHAR));
  strcpy(uvar_skyfile,SKYFILE);

  /* register user input variables */
  SUB( LALRegisterBOOLUserVar(   &status, "help",            'h', UVAR_HELP,     "Print this message",            &uvar_help),            &status);  
  SUB( LALRegisterBOOLUserVar(   &status, "weighAM",          0,  UVAR_OPTIONAL, "Print this message",            &uvar_weighAM),         &status);  
  SUB( LALRegisterBOOLUserVar(   &status, "weighNoise",       0,  UVAR_OPTIONAL, "Print this message",            &uvar_weighNoise),      &status);  
  SUB( LALRegisterINTUserVar(    &status, "ifo",             'i', UVAR_OPTIONAL, "Detector GEO(1) LLO(2) LHO(3)", &uvar_ifo ),            &status);
  SUB( LALRegisterINTUserVar(    &status, "blocksRngMed",    'w', UVAR_OPTIONAL, "RngMed block size",             &uvar_blocksRngMed),    &status);
  SUB( LALRegisterREALUserVar(   &status, "f0",              'f', UVAR_OPTIONAL, "Start search frequency",        &uvar_f0),              &status);
  SUB( LALRegisterREALUserVar(   &status, "fSearchBand",     'b', UVAR_OPTIONAL, "Search frequency band",         &uvar_fSearchBand),     &status);
  SUB( LALRegisterREALUserVar(   &status, "peakThreshold",   't', UVAR_OPTIONAL, "Peak selection threshold",      &uvar_peakThreshold),   &status);
  SUB( LALRegisterREALUserVar(   &status, "houghFalseAlarm", 'a', UVAR_OPTIONAL, "Hough false alarm",             &uvar_houghFalseAlarm), &status);
  SUB( LALRegisterSTRINGUserVar( &status, "earthEphemeris",  'E', UVAR_OPTIONAL, "Earth Ephemeris file",          &uvar_earthEphemeris),  &status);
  SUB( LALRegisterSTRINGUserVar( &status, "sunEphemeris",    'S', UVAR_OPTIONAL, "Sun Ephemeris file",            &uvar_sunEphemeris),    &status);
  SUB( LALRegisterSTRINGUserVar( &status, "sftDir",          'D', UVAR_OPTIONAL, "SFT Directory",                 &uvar_sftDir),          &status);
  SUB( LALRegisterSTRINGUserVar( &status, "dirnameOut",      'o', UVAR_OPTIONAL, "Output directory",              &uvar_dirnameOut),      &status);
  SUB( LALRegisterSTRINGUserVar( &status, "fbasenameOut",    'B', UVAR_OPTIONAL, "Output file basename",          &uvar_fbasenameOut),    &status);
  SUB( LALRegisterSTRINGUserVar( &status, "skyfile",         'P', UVAR_OPTIONAL, "Input skypatch file",           &uvar_skyfile),         &status);

  /* read all command line variables */
  SUB( LALUserVarReadAllInput(&status, argc, argv), &status);

  /* set value of bias which might have been changed from default */  
  SUB( LALRngMedBias( &status, &normalizeThr, uvar_blocksRngMed ), &status ); 
  
  /* set value of nfsizecylinder */
  nfSizeCylinder = NFSIZE;
  
  /* set detector */
  if (uvar_ifo ==1) detector=lalCachedDetectors[LALDetectorIndexGEO600DIFF];
  if (uvar_ifo ==2) detector=lalCachedDetectors[LALDetectorIndexLLODIFF];
  if (uvar_ifo ==3) detector=lalCachedDetectors[LALDetectorIndexLHODIFF];

  /* exit if help was required */
  if (uvar_help)
    exit(0); 
 
  /* open log file for writing */
  fnameLog = (CHAR *)LALMalloc( 512*sizeof(CHAR));
  strcpy(fnameLog,uvar_dirnameOut);
  strcat(fnameLog, "/logfiles/");
  /* now create directory fdirOut/logfiles using mkdir */
  errno = 0;
  {
    /* check whether file can be created or if it exists already 
       if not then exit */
    INT4 mkdir_result;
    mkdir_result = mkdir(fnameLog, S_IRWXU | S_IRWXG | S_IRWXO);
    if ( (mkdir_result == -1) && (errno != EEXIST) )
      {
	fprintf(stderr, "unable to create logfiles directory %d\n", skyCounter);
        LALFree(fnameLog);
	return 1;  /* stop the program */
      }
  }
  /* create the logfilename in the logdirectory */
  strcat(fnameLog, uvar_fbasenameOut);
  strcat(fnameLog,".log");
  /* open the log file for writing */
  if ((fpLog = fopen(fnameLog, "w")) == NULL) {
    fprintf(stderr, "Unable to open file %s for writing\n", fnameLog);
    LALFree(fnameLog);
    exit(1);
  }
  
  /* get the log string */
  SUB( LALUserVarGetLog(&status, &logstr, UVAR_LOGFMT_CFGFILE), &status);  

  fprintf( fpLog, "## LOG FILE FOR Hough Driver\n\n");
  fprintf( fpLog, "# User Input:\n");
  fprintf( fpLog, "#-------------------------------------------\n");
  fprintf( fpLog, logstr);
  LALFree(logstr);

  /* copy contents of skypatch file into logfile */
  fprintf(fpLog, "\n\n# Contents of skypatch file:\n");
  fclose(fpLog);
  {
    CHAR command[1024] = "";
    sprintf(command, "cat %s >> %s", uvar_skyfile, fnameLog);
    system(command);
  }

  /* append an ident-string defining the exact CVS-version of the code used */
  fpLog = fopen(fnameLog, "a");
  {
    CHAR command[1024] = "";
    fprintf (fpLog, "\n\n# CVS-versions of executable:\n");
    fprintf (fpLog, "# -----------------------------------------\n");
    fclose (fpLog);
    
    sprintf (command, "ident %s | sort -u >> %s", argv[0], fnameLog);
    system (command);	/* we don't check this. If it fails, we assume that */
    			/* one of the system-commands was not available, and */
    			/* therefore the CVS-versions will not be logged */

    LALFree(fnameLog); 
  }



  /*****************************************************************/
  /* read skypatch info */
  /*****************************************************************/
  {
    FILE   *fpsky = NULL; 
    INT4   r;
    REAL8  temp1, temp2, temp3, temp4;
    
    fpsky = fopen(uvar_skyfile, "r");
    if ( !fpsky )
      {
	fprintf(stderr, "Unable to find skyfile %s\n", uvar_skyfile);
	return DRIVEHOUGHCOLOR_EFILE;
      }
    
    
    nSkyPatches = 0;
    do 
      {
	r=fscanf(fpsky,"%lf%lf%lf%lf\n", &temp1, &temp2, &temp3, &temp4);
	/* make sure the line has the right number of entries or is EOF */
	if (r==4) nSkyPatches++;
      } while ( r != EOF);
    rewind(fpsky);
    
    skyAlpha = (REAL8 *)LALMalloc(nSkyPatches*sizeof(REAL8));
    skyDelta = (REAL8 *)LALMalloc(nSkyPatches*sizeof(REAL8));     
    skySizeAlpha = (REAL8 *)LALMalloc(nSkyPatches*sizeof(REAL8));
    skySizeDelta = (REAL8 *)LALMalloc(nSkyPatches*sizeof(REAL8));     
    
    
    for (skyCounter = 0; skyCounter < nSkyPatches; skyCounter++)
      {
	r=fscanf(fpsky,"%lf%lf%lf%lf\n", skyAlpha + skyCounter, skyDelta + skyCounter, 
		 skySizeAlpha + skyCounter,  skySizeDelta + skyCounter);
      }
    
    fclose(fpsky);     
  }


  /* read sft files */
  {
    CHAR *tempDir;
    REAL8 doppWings, fmin, fmax;
    INT4 length;

    /* copy pattern to be searched */
    tempDir = (CHAR *)LALMalloc(512*sizeof(CHAR));
    strcpy(tempDir, uvar_sftDir);
    strcat(tempDir, "/*SFT*.*");

    /* add Doppler wings */
    doppWings = (uvar_f0 + uvar_fSearchBand) * VTOT;    
    fmin = uvar_f0 - doppWings;
    fmax = uvar_f0 + uvar_fSearchBand + doppWings;

    /* read sft files making sure to add extra bins for running median */
    LAL_CALL( LALReadSFTfiles ( &status, &inputSFTs, fmin, fmax, 
				uvar_blocksRngMed, tempDir), &status); 
    LALFree(tempDir);

    /* set other parameters based on sfts */
    mObsCoh = inputSFTs->length; /* number of sfts */
    deltaF = inputSFTs->data->deltaF;  /* frequency resolution */
    timeBase= 1.0/deltaF; /* coherent integration time */

    f0Bin = floor( uvar_f0 * timeBase + 0.5); /* initial search frequency */
    length =  uvar_fSearchBand * timeBase; /* total number of search bins - 1 */
    fLastBin = f0Bin+length;   /* final frequency bin to be analyzed */

    /* set up weights -- this should be done before normalizing the sfts */
    weightsV.length = mObsCoh;
    weightsV.data = (REAL8 *)LALMalloc(mObsCoh*sizeof(REAL8));

    /* calculate sft noise weights if required by user */
    SUB( LALHOUGHInitializeWeights( &status, &weightsV), &status);
    if (uvar_weighNoise ) {
      SUB( LALHOUGHComputeNoiseWeights( &status, &weightsV, inputSFTs, uvar_blocksRngMed), &status); 
      SUB( LALHOUGHNormalizeWeights( &status, &weightsV), &status);
    }

    /* normalize sfts */
    LAL_CALL( LALNormalizeSFTVect (&status, inputSFTs, uvar_blocksRngMed, 0), &status);

    /* use above value of length to allocate memory for nstar, fstar etc. */
    nStar = (UINT4 *)LALMalloc((length + 1)*sizeof(UINT4));
    freqStar = (REAL8 *)LALMalloc((length + 1)*sizeof(REAL8));
    alphaStar = (REAL8 *)LALMalloc((length + 1)*sizeof(REAL8));
    deltaStar = (REAL8 *)LALMalloc((length + 1)*sizeof(REAL8));
    fdotStar = (REAL8 *)LALMalloc((length + 1)*sizeof(REAL8));

    /* initialize nstar values */
    {
      INT4 starIndex;
      for( starIndex = 0; starIndex < length + 1; starIndex++)
	nStar[starIndex] = 0;
    } 
  } /* end of sft reading block */
  
    
  /* computing the Hough threshold for a given false alarm  */
  /* HoughThreshold = N*alpha +sqrt(2N alpha (1-alpha))*erfcinv(2 alpha_h) */
  {
    REAL8  alphaPeak, meanN, sigmaN, erfcInv;
    
    alphaPeak = exp(- uvar_peakThreshold);
    
    meanN = mObsCoh* alphaPeak;
    sigmaN = sqrt(meanN *(1.0 - alphaPeak));
    /* this should be  erfcInv =erfcinv(2.0 *uvar_houghFalseAlarm) */
    /* the function used is basically the inverse of the CDF for the 
       Gaussian distribution with unit variance and
       erfcinv(x) = gsl_cdf_ugaussian_Qinv (0.5*x)/sqrt(2) */
    /* First check that false alarm is within bounds 
       and set it to something reasonable if not */
    if ( (uvar_houghFalseAlarm > 0.999)&&(uvar_houghFalseAlarm < 0.0) ) 
      uvar_houghFalseAlarm =  FALSEALARM;
    erfcInv = gsl_cdf_ugaussian_Qinv (uvar_houghFalseAlarm)/sqrt(2);    
    houghThreshold = meanN + sigmaN*sqrt(2.0)*erfcInv;    
  }


  /* ****************************************************************/
  /* reading from SFT, times and generating peakgrams  */
  /* ****************************************************************/

  timeV.length = mObsCoh;
  timeV.data = NULL;  
  timeV.data = (LIGOTimeGPS *)LALMalloc(mObsCoh*sizeof(LIGOTimeGPS));
  
  pgV.length = mObsCoh;
  pgV.pg = NULL;
  pgV.pg = (HOUGHPeakGram *)LALMalloc(mObsCoh*sizeof(HOUGHPeakGram));

  { 
    SFTtype  sft;
    UCHARPeakGram     pg1;
    INT4   nPeaks, length;
    UINT4  j; 
  
    length = inputSFTs->data->data->length;
    pg1.length = length;
    pg1.data = NULL;
    pg1.data = (UCHAR *)LALMalloc(length* sizeof(UCHAR));

    /* loop over sfts and select peaks */
    for (j=0; j < mObsCoh; j++){

      sft = inputSFTs->data[j];

      SUB (SFTtoUCHARPeakGram( &status, &pg1, &sft, uvar_peakThreshold), &status);
      
      nPeaks = pg1.nPeaks;
      timeV.data[j].gpsSeconds = pg1.epoch.gpsSeconds;
      timeV.data[j].gpsNanoSeconds = pg1.epoch.gpsNanoSeconds;

      /* compress peakgram */      
      pgV.pg[j].length = nPeaks;
      pgV.pg[j].peak = NULL;
      pgV.pg[j].peak = (INT4 *)LALMalloc(nPeaks* sizeof(INT4));

      SUB( LALUCHAR2HOUGHPeak( &status, &(pgV.pg[j]), &pg1), &status );
    } /* end loop over sfts */


    /* we are done with the sfts and ucharpeakgram now */
    LAL_CALL (LALDestroySFTVector(&status, &inputSFTs), &status );
    LALFree(pg1.data);

  }/* end block for selecting peaks */

  /******************************************************************/
  /* compute detector velocity for those time stamps  */
  /******************************************************************/
  velV.length = mObsCoh;
  velV.data = NULL;
  velV.data = (REAL8Cart3Coor *)LALMalloc(mObsCoh*sizeof(REAL8Cart3Coor));

  /*   velVmid.length = mObsCoh; */
  /*   velVmid.data = NULL; */
  /*   velVmid.data = (REAL8Cart3Coor *)LALMalloc(mObsCoh*sizeof(REAL8Cart3Coor)); */

  {  
    VelocityPar   velPar;
    REAL8     vel[3], tempVel[3]; 
    UINT4     j; 
    
    LALLeapSecFormatAndAcc lsfas = {LALLEAPSEC_GPSUTC, LALLEAPSEC_STRICT};
    INT4 tmpLeap; /* need this because Date pkg defines leap seconds as
                   INT4, while EphemerisData defines it to be INT2. This won't
                   cause problems before, oh, I don't know, the Earth has been
                   destroyed in nuclear holocaust. -- dwchin 2004-02-29 */

    velPar.detector = detector;
    velPar.tBase = timeBase;
    velPar.vTol = ACCURACY;
    velPar.edat = NULL; 

    /*  ephemeris info */
    edat = (EphemerisData *)LALMalloc(sizeof(EphemerisData));
   (*edat).ephiles.earthEphemeris = uvar_earthEphemeris;
   (*edat).ephiles.sunEphemeris = uvar_sunEphemeris;

    /* Leap seconds for the start time of the run */
    SUB( LALLeapSecs(&status, &tmpLeap, &(timeV.data[0]), &lsfas), &status);
    (*edat).leap = (INT2)tmpLeap;
    /* (*edat).leap = 13;   <<<<<<< Correct this */

    /* read in ephemeris data */
    SUB( LALInitBarycenter( &status, edat), &status);
    velPar.edat = edat;
    
    for(j=0; j< velV.length; ++j){
      velPar.startTime.gpsSeconds     = timeV.data[j].gpsSeconds;
      velPar.startTime.gpsNanoSeconds = timeV.data[j].gpsNanoSeconds;
      
      SUB( LALAvgDetectorVel ( &status, vel, &velPar), &status ); 
      
      {
	REAL8  midTimeBase; 
	LIGOTimeGPS midTimeBaseGPS, tempGPS; 

	/* convert half of timebase to GPS structure */	
        midTimeBase = 0.5 * timeBase; 
        SUB( LALFloatToGPS ( &status, &midTimeBaseGPS, &midTimeBase), &status); 

	/* in case time to calculate velocity i smid time of sft */
	/* add mid time base to start time of sft */
 	tempGPS.gpsSeconds = midTimeBaseGPS.gpsSeconds + timeV.data[j].gpsSeconds; 
 	tempGPS.gpsNanoSeconds = midTimeBaseGPS.gpsNanoSeconds + timeV.data[j].gpsNanoSeconds;  
	
	/* for debugging purposes */
	/*      in case time to calculate velocity is start time */
	/* 	tempGPS.gpsSeconds = timeV.data[j].gpsSeconds;  */
	/* 	tempGPS.gpsNanoSeconds = timeV.data[j].gpsNanoSeconds;  */
        SUB( LALDetectorVel ( &status, tempVel, &tempGPS, detector, edat), &status ); 
      }

      /*       velV.data[j].x= vel[0]; */
      /*       velV.data[j].y= vel[1]; */
      /*       velV.data[j].z= vel[2];    */

      /* for debugging purposes */
      velV.data[j].x= tempVel[0];
      velV.data[j].y= tempVel[1];
      velV.data[j].z= tempVel[2];   

      /*       velVmid.data[j].x = tempVel[0]; */
      /*       velVmid.data[j].y = tempVel[1]; */
      /*       velVmid.data[j].z = tempVel[2]; */
    }
  }

  /******************************************************************/
  /* compute the time difference relative to startTime for all SFT */
  /******************************************************************/
  timeDiffV.length = mObsCoh;
  timeDiffV.data = NULL; 
  timeDiffV.data = (REAL8 *)LALMalloc(mObsCoh*sizeof(REAL8));

  {
    REAL8   t0, ts, tn, midTimeBase;
    UINT4   j; 

    midTimeBase=0.5*timeBase;
    ts = timeV.data[0].gpsSeconds;
    tn = timeV.data[0].gpsNanoSeconds * 1.00E-9;
    t0=ts+tn;
    timeDiffV.data[0] = midTimeBase;

    for(j=1; j< velV.length; ++j){
      ts = timeV.data[j].gpsSeconds;
      tn = timeV.data[j].gpsNanoSeconds * 1.00E-9;  
      timeDiffV.data[j] = ts+tn - t0 + midTimeBase; 
    }  
  }

  /* ****************************************************************/
  /*  Writing the time & detector-velocity corresponding to each SFT  */
  /* ****************************************************************/
  
  /*     {  */
  /*       UINT4   j;  */
  /*       FILE   *fp = NULL;  */
  
  /*       fp = fopen( "velocityOut.c", "w");  */
  /*       if (fp==NULL){  */
  /* 	fprintf(stderr,"Unable to open velocity file for writing\n");  */
  /* 	return 1;  */
  /*       }  */
  /*       /\*     read data format:  INT4 INT4  REAL8 REAL8 REAL8 *\/ */
  /*       for (j=0; j<mObsCoh;++j){  */
  /* 	fprintf(fp, "%d %d %g %g %g %g %g %g\n",  */
  /* 		timeV.data[j].gpsSeconds, timeV.data[j].gpsNanoSeconds, velV.data[j].x, velV.data[j].y,  */
  /* 		velV.data[j].z, velVmid.data[j].x, velVmid.data[j].y, velVmid.data[j].z );  */
  
  /* 	fflush(fp);  */
  /*       }  */
  /*       fclose(fp);  */
  
  /*     fp = fopen( fnameTime, "w");  */
  /*       if (fp==NULL){  */
  /* 	fprintf(stderr,"Unable to open file %s\n for writing",fnameTime);  */
  /*         return 1; /\* stop the program *\/ */
  /*       }  */
  /*       /\* read data format:  INT4 INT4 *\/ */
  /*       for (j=0; j<mObsCoh;++j){  */
  /* 	fprintf(fp, "%d %d \n",  */
  /* 		(timeV.data[j].gpsSeconds),  */
  /* 		(timeV.data[j].gpsNanoSeconds) );  */
  
  /* 	fflush(fp);  */
  /*       }  */
  /*       fclose(fp);  */
  /*  }  */
  



  
  /* loop over sky patches */
  for (skyCounter = 0; skyCounter < nSkyPatches; skyCounter++)
    {
      /* set sky positions and skypatch sizes */
      alpha = skyAlpha[skyCounter];
      delta = skyDelta[skyCounter];
      patchSizeX = skySizeDelta[skyCounter];
      patchSizeY = skySizeAlpha[skyCounter];


      /* calculate amplitude modulation weights */
      if (uvar_weighAM) {
	SUB( LALHOUGHComputeAMWeights( &status, &weightsV, &timeV, &detector, 
				       edat, alpha, delta), &status);
	SUB( LALHOUGHNormalizeWeights( &status, &weightsV), &status);
      }
      
      /******************************************************************/  
      /* opening the output statistic, and event files */
      /******************************************************************/  
      
      /* create the directory name uvar_dirnameOut/skypatch_$j */
      strcpy(  filestats, uvar_dirnameOut);
      strcat( filestats, "/skypatch_");
      {
	CHAR tempstr[16];
	sprintf(tempstr, "%d", skyCounter+1);
	strcat( filestats, tempstr);
      }
      strcat( filestats, "/");

      /* now create directory fnameout/skypatch_$j using mkdir */
      errno = 0;
      {
	/* check whether file can be created or if it exists already 
	   if not then exit */
	INT4 mkdir_result;
	mkdir_result = mkdir(filestats, S_IRWXU | S_IRWXG | S_IRWXO);
	if ( (mkdir_result == -1) && (errno != EEXIST) )
	  {
	    fprintf(stderr, "unable to create skypatch directory %d\n", skyCounter);
	    return 1;  /* stop the program */
	  }
      }


      /* create the base filenames for the stats, histo and event files and template files*/
      strcat( filestats, uvar_fbasenameOut);
      strcpy( filehisto, filestats);

#ifdef PRINTEVENTS
      strcpy( fileEvents, filestats);
#endif

#ifdef PRINTTEMPLATES
      strcpy(fileTemplates, filestats);
#endif

#ifdef PRINTMAPS
      strcpy(fileMaps, filestats);
#endif

      /* create and open the stats file for writing */
      strcat(  filestats, "stats");
      fp1=fopen(filestats,"w");
      if ( !fp1 ){
	fprintf(stderr,"Unable to find file %s for writing\n", filestats);
	return DRIVEHOUGHCOLOR_EFILE;
      }
      /*setlinebuf(fp1);*/ /*line buffered on */  
      setvbuf(fp1, (char *)NULL, _IOLBF, 0);      


#ifdef PRINTEVENTS
      /* create and open the events list file */
      strcat(  fileEvents, "events");
      fpEvents=fopen(fileEvents,"w");
      if ( !fpEvents ){
	fprintf(stderr,"Unable to find file %s\n", fileEvents);
	return DRIVEHOUGHCOLOR_EFILE;
      }
      setvbuf(fp1, (char *)NULL, _IOLBF, 0);      
      /*setlinebuf(fpEvents);*/ /*line buffered on */  
#endif


#ifdef PRINTTEMPLATES
      /* create and open templates file */
      strcat( fileTemplates, "templates");
      fpTemplates = fopen(fileTemplates, "w");
      if ( !fpTemplates ){
	fprintf(stderr, "Unable to create file %s\n", fileTemplates);
	return DRIVEHOUGHCOLOR_EFILE;
      }
      setvbuf(fp1, (char *)NULL, _IOLBF, 0);      
      /*setlinebuf(fpTemplates);*/ /*line buffered on */   
#endif 

      /* ****************************************************************/
      /*  general parameter settings and 1st memory allocation */
      /* ****************************************************************/
      
      lutV.length    = mObsCoh;
      lutV.lut = NULL;
      lutV.lut = (HOUGHptfLUT *)LALMalloc(mObsCoh*sizeof(HOUGHptfLUT));
      
      phmdVS.length  = mObsCoh;
      phmdVS.nfSize  = nfSizeCylinder;
      phmdVS.deltaF  = deltaF;
      phmdVS.phmd = NULL;
      phmdVS.phmd=(HOUGHphmd *)LALMalloc(mObsCoh*nfSizeCylinder*sizeof(HOUGHphmd));
      
      freqInd.deltaF = deltaF;
      freqInd.length = mObsCoh;
      freqInd.data = NULL;
      freqInd.data =  ( UINT8 *)LALMalloc(mObsCoh*sizeof(UINT8));
      
      /* ****************************************************************/
      /* Case: no spins-demodulaton  or Non demodulation (SFT input)*/
      parDem.deltaF = deltaF;
      parDem.skyPatch.alpha = alpha;
      parDem.skyPatch.delta = delta;
      parDem.timeDiff = 0.0;
      parDem.spin.length = 0;
      parDem.spin.data = NULL;
      parDem.positC.x = 0.0;
      parDem.positC.y = 0.0;
      parDem.positC.z = 0.0;
      
      /*****************************************************************/
      parRes.deltaF = deltaF;
      parRes.patchSkySizeX  = patchSizeX;
      parRes.patchSkySizeY  = patchSizeY;
      parRes.pixelFactor = PIXELFACTOR;
      parRes.pixErr = PIXERR;
      parRes.linErr = LINERR;
      parRes.vTotC = VTOT;
      /******************************************************************/  
      /* ************* histogram of the number-counts in the Hough maps */
      hist.length = mObsCoh+1;
      histTotal.length = mObsCoh+1;
      hist.data = NULL;
      histTotal.data = NULL;
      hist.data = (UINT4 *)LALMalloc((mObsCoh+1)*sizeof(UINT4));
      histTotal.data = (UINT4 *)LALMalloc((mObsCoh+1)*sizeof(UINT4));
      { 
	UINT4   j;
	for(j=0; j< histTotal.length; ++j){ histTotal.data[j]=0; }
      }
      
      /******************************************************************/  
      /******************************************************************/  
      
      fBin= f0Bin;
      iHmap = 0;
      
      /* ***** for spin-down case ****/
      nSpin1Max = floor(nfSizeCylinder/2.0);
      f1jump = 1./timeDiffV.data[mObsCoh- 1];
      
      /******************************************************************/
      /******************************************************************/
      /* starting the search f0-fLastBin.
	 Note one set LUT might not cover all the interval.
	 This is taken into account.
	 Memory allocation changes */
      /******************************************************************/
      /******************************************************************/
      
      while( fBin <= fLastBin){
	INT8 fBinSearch, fBinSearchMax;
	UINT4 i,j; 
	REAL8UnitPolarCoor sourceLocation;
	
	
	parRes.f0Bin =  fBin;      
	SUB( LALHOUGHComputeNDSizePar( &status, &parSize, &parRes ),  &status );
	xSide = parSize.xSide;
	ySide = parSize.ySide;
	maxNBins = parSize.maxNBins;
	maxNBorders = parSize.maxNBorders;
	
	/* *******************create patch grid at fBin ****************  */
	patch.xSide = xSide;
	patch.ySide = ySide;
	patch.xCoor = NULL;
	patch.yCoor = NULL;
	patch.xCoor = (REAL8 *)LALMalloc(xSide*sizeof(REAL8));
	patch.yCoor = (REAL8 *)LALMalloc(ySide*sizeof(REAL8));
	SUB( LALHOUGHFillPatchGrid( &status, &patch, &parSize ), &status );
	
	/*************** other memory allocation and settings************ */
	for(j=0; j<lutV.length; ++j){
	  lutV.lut[j].maxNBins = maxNBins;
	  lutV.lut[j].maxNBorders = maxNBorders;
	  lutV.lut[j].border =
	    (HOUGHBorder *)LALMalloc(maxNBorders*sizeof(HOUGHBorder));
	  lutV.lut[j].bin =
	    (HOUGHBin2Border *)LALMalloc(maxNBins*sizeof(HOUGHBin2Border));
	  for (i=0; i<maxNBorders; ++i){
	    lutV.lut[j].border[i].ySide = ySide;
	    lutV.lut[j].border[i].xPixel =
	      (COORType *)LALMalloc(ySide*sizeof(COORType));
	  }
	}
	for(j=0; j<phmdVS.length * phmdVS.nfSize; ++j){
	  phmdVS.phmd[j].maxNBorders = maxNBorders;
	  phmdVS.phmd[j].leftBorderP =
	    (HOUGHBorder **)LALMalloc(maxNBorders*sizeof(HOUGHBorder *));
	  phmdVS.phmd[j].rightBorderP =
	    (HOUGHBorder **)LALMalloc(maxNBorders*sizeof(HOUGHBorder *));
	  phmdVS.phmd[j].ySide = ySide;
	  phmdVS.phmd[j].firstColumn = NULL;
	  phmdVS.phmd[j].firstColumn = (UCHAR *)LALMalloc(ySide*sizeof(UCHAR));
	}
	
	/* ************* create all the LUTs at fBin ********************  */  
	for (j=0;j< mObsCoh;++j){  /* create all the LUTs */
	  parDem.veloC.x = velV.data[j].x;
	  parDem.veloC.y = velV.data[j].y;
	  parDem.veloC.z = velV.data[j].z;      
	  /* calculate parameters needed for buiding the LUT */
	  SUB( LALNDHOUGHParamPLUT( &status, &parLut, &parSize, &parDem),&status );
	  /* build the LUT */
	  SUB( LALHOUGHConstructPLUT( &status, &(lutV.lut[j]), &patch, &parLut ),
	       &status );
	}
        
	/************* build the set of  PHMD centered around fBin***********/     
	phmdVS.fBinMin = fBin-floor(nfSizeCylinder/2.);
	SUB( LALHOUGHConstructSpacePHMD(&status, &phmdVS, &pgV, &lutV), &status );
	if (uvar_weighAM || uvar_weighNoise) {
	  SUB( LALHOUGHWeighSpacePHMD(&status, &phmdVS, &weightsV), &status);
	}
	
	/* ************ initializing the Total Hough map space *********** */   
	ht.xSide = xSide;
	ht.ySide = ySide;
	ht.mObsCoh = mObsCoh;
	ht.deltaF = deltaF;
	ht.map   = NULL;
	ht.map   = (HoughTT *)LALMalloc(xSide*ySide*sizeof(HoughTT));
	SUB( LALHOUGHInitializeHT( &status, &ht, &patch), &status); /*not needed */
	
	/******************************************************************/
	/*  Search frequency interval possible using the same LUTs */
	fBinSearch = fBin;
	fBinSearchMax= fBin + parSize.nFreqValid -1-floor( (nfSizeCylinder-1)/2.);
	
	/** >>>>>>>>>>>>>>>>>>>>>>>>>>>>> * <<<<<<<<<<<<<<<<<<<<<<<<<<<< **/
	/* Study all possible frequencies with one set of LUT */
	/** >>>>>>>>>>>>>>>>>>>>>>>>>>>>> * <<<<<<<<<<<<<<<<<<<<<<<<<<<< **/
	
	while ( (fBinSearch <= fLastBin) && (fBinSearch < fBinSearchMax) )  { 
	  
	  /**********************************************/
	  /* Case: No spin-down.  Study the fBinSearch */
	  /**********************************************/
	  ht.f0Bin = fBinSearch;
	  ht.spinRes.length =0;
	  ht.spinRes.data = NULL;
	  for (j=0;j< mObsCoh;++j){	freqInd.data[j]= fBinSearch; } 
	  if (uvar_weighAM || uvar_weighNoise) {
	    SUB( LALHOUGHConstructHMT_W( &status, &ht, &freqInd, &phmdVS ), &status );
	  }
	  else {
	    SUB( LALHOUGHConstructHMT( &status, &ht, &freqInd, &phmdVS ), &status );	  
	  }

	  /* ********************* perfom stat. analysis on the maps ****************** */
	  SUB( LALHoughStatistics ( &status, &stats, &ht), &status );
	  SUB( LALStereo2SkyLocation (&status, &sourceLocation, 
				   stats.maxIndex[0], stats.maxIndex[1], 
				   &patch, &parDem), &status);
	  SUB( LALHoughHistogram ( &status, &hist, &ht), &status);
	  for(j=0; j< histTotal.length; ++j){ histTotal.data[j]+=hist.data[j]; }

	  if (stats.maxCount > nStar[fBinSearch-f0Bin])
	    {
	      nStar[fBinSearch-f0Bin] = stats.maxCount;
	      freqStar[fBinSearch-f0Bin] = fBinSearch*deltaF;
	      alphaStar[fBinSearch-f0Bin] = sourceLocation.alpha;
	      deltaStar[fBinSearch-f0Bin] = sourceLocation.delta;
	      fdotStar[fBinSearch-f0Bin] = 0;
	    }
       	  
	  /* ********************* print results *********************** */
	  
#ifdef PRINTMAPS
	  
	  if( PrintHmap2m_file( &ht, fileMaps, iHmap ) ) return 5;
#endif 
	  
 
	  fprintf(fp1, "%d %f %f %f %f %f %f %f 0.0 \n",
		  iHmap, sourceLocation.alpha, sourceLocation.delta,
		  (REAL4)stats.maxCount, (REAL4)stats.minCount, stats.avgCount,stats.stdDev,
		  (fBinSearch*deltaF) );
          
#ifdef PRINTEVENTS
	  SUB( PrintHoughEvents (&status, fpEvents, houghThreshold, &ht,
				 &patch, &parDem), &status );
#endif      

#ifdef PRINTTEMPLATES
	  SUB( PrintHoughEvents (&status, fpTemplates, 0.0, &ht, &patch, &parDem), &status);
#endif

	  ++iHmap;
	  
	  
	  /********************************************/
	  /* Case: 1 spin-down. at  fBinSearch */
	  /********************************************/
	  {
	    INT4   n;
	    REAL8  f1dis;
	    
	    ht.spinRes.length = 1;
	    ht.spinRes.data = NULL;
	    ht.spinRes.data = (REAL8 *)LALMalloc(ht.spinRes.length*sizeof(REAL8));
	    
	    for( n=1; n<= nSpin1Max; ++n){ /*loop over all values of f1 */
	      f1dis = - n*f1jump;
	      ht.spinRes.data[0] =  f1dis*deltaF;
	      
	      for (j=0;j< mObsCoh;++j){
		freqInd.data[j] = fBinSearch + floor(timeDiffV.data[j]*f1dis+0.5);
	      }
	      
	      if (uvar_weighAM || uvar_weighNoise) {
		SUB( LALHOUGHConstructHMT_W( &status, &ht, &freqInd, &phmdVS ), &status );
	      }
	      else {
		SUB( LALHOUGHConstructHMT( &status, &ht, &freqInd, &phmdVS ), &status );	  
	      }

	      /* ********************* perfom stat. analysis on the maps ****************** */
	      SUB( LALHoughStatistics ( &status, &stats, &ht), &status );
	      SUB( LALStereo2SkyLocation (&status, &sourceLocation, 
				       stats.maxIndex[0], stats.maxIndex[1], &patch, &parDem), &status);
	      SUB( LALHoughHistogram ( &status, &hist, &ht), &status);
	      for(j=0; j< histTotal.length; ++j){ histTotal.data[j]+=hist.data[j]; }
	      

	      if (stats.maxCount > nStar[fBinSearch-f0Bin])
		{
		  nStar[fBinSearch-f0Bin] = stats.maxCount;
		  freqStar[fBinSearch-f0Bin] = fBinSearch*deltaF;
		  alphaStar[fBinSearch-f0Bin] = sourceLocation.alpha;
		  deltaStar[fBinSearch-f0Bin] = sourceLocation.delta;
		  fdotStar[fBinSearch-f0Bin] = ht.spinRes.data[0];
		}


	      /* ***** print results *********************** */
	      
#ifdef PRINTMAPS
	      if( PrintHmap2m_file( &ht, fileMaps, iHmap ) ) return 5;
#endif
	      
	      fprintf(fp1, "%d %f %f %f %f %f %f %f %g\n",
		      iHmap, sourceLocation.alpha, sourceLocation.delta,
		      (REAL4)stats.maxCount, (REAL4)stats.minCount, stats.avgCount,stats.stdDev,
		      (fBinSearch*deltaF), ht.spinRes.data[0]);
#ifdef PRINTEVENTS
	      SUB( PrintHoughEvents (&status, fpEvents, houghThreshold, &ht,
				     &patch, &parDem), &status );
#endif    

#ifdef PRINTTEMPLATES
	  SUB( PrintHoughEvents (&status, fpTemplates, 0.0, &ht, &patch, &parDem), &status);
#endif

	      ++iHmap;
	      
	      /* what else with output, equal to non-spin case */
	    }
	    LALFree(ht.spinRes.data);
	  }
	  
	  /********************************************/
	  /* *** shift the search freq. & PHMD structure 1 freq.bin ****** */
	  ++fBinSearch;
	  SUB( LALHOUGHupdateSpacePHMDup(&status, &phmdVS, &pgV, &lutV), &status );
	  if (uvar_weighAM || uvar_weighNoise) {
	    SUB( LALHOUGHWeighSpacePHMD( &status, &phmdVS, &weightsV), &status);
	  }
	}   /* ********>>>>>>  closing second while  <<<<<<<<**********<  */
	
	fBin = fBinSearch;
	
	/* ********************  Free partial memory ******************* */
	LALFree(patch.xCoor);
	LALFree(patch.yCoor);
	LALFree(ht.map);
	
	for (j=0; j<lutV.length ; ++j){
	  for (i=0; i<maxNBorders; ++i){
	    LALFree( lutV.lut[j].border[i].xPixel);
	  }
	  LALFree( lutV.lut[j].border);
	  LALFree( lutV.lut[j].bin);
	}
	for(j=0; j<phmdVS.length * phmdVS.nfSize; ++j){
	  LALFree( phmdVS.phmd[j].leftBorderP);
	  LALFree( phmdVS.phmd[j].rightBorderP);
	  LALFree( phmdVS.phmd[j].firstColumn);
	}
	
      } /* closing while */
      
      /******************************************************************/
      /* printing total histogram */
      /******************************************************************/
      if( PrintHistogram( &histTotal, filehisto) ) return 7;
      
      /******************************************************************/
      /* closing files with statistics results and events */
      /******************************************************************/  
      fclose(fp1);
#ifdef PRINTEVENTS
      fclose(fpEvents);
#endif

#ifdef PRINTTEMPLATES
      fclose(fpTemplates);
#endif
     

      /******************************************************************/
      /* Free memory allocated inside skypatches loop */
      /******************************************************************/
      
      LALFree(lutV.lut);  

      
      LALFree(phmdVS.phmd);
      LALFree(freqInd.data);
      LALFree(hist.data);
      LALFree(histTotal.data);
    
      
#ifdef TIMING
      stop = realcc();
      printf(" All: %llu\n", stop-start);
#endif
  
    } /* finish loop over skypatches */


  /* create the directory for writing nstar */
  strcpy( filestar, uvar_dirnameOut);
  strcat( filestar, "/nstarfiles/");
  errno = 0;
  {
    /* check whether file can be created or if it exists already 
       if not then exit */
    INT4 mkdir_result;
    mkdir_result = mkdir(filestar, S_IRWXU | S_IRWXG | S_IRWXO);
    if ( (mkdir_result == -1) && (errno != EEXIST) )
      {
	fprintf(stderr, "unable to create nstar directory\n");
	return 1;  /* stop the program */
      }
  }
  strcat( filestar, uvar_fbasenameOut);
  strcat( filestar, "nstar");


  /* open the nstar file for writing */
  fpStar=fopen(filestar,"w");
  if ( !fpStar ){
    fprintf(stderr,"Unable to find file %s for writing\n", filestar);
    return DRIVEHOUGHCOLOR_EFILE;
  }
  /*setlinebuf(fp1);*/ /*line buffered on */  
  setvbuf(fpStar, (char *)NULL, _IOLBF, 0);      


  /* write the nstar resulta */
  {
    INT4 starIndex;
    /* we don't record the nstar for the last bin */
    for(starIndex = 0; starIndex < fLastBin - f0Bin; starIndex++)
      {
	fprintf(fpStar, "%d %f %f %f %g \n", nStar[starIndex], freqStar[starIndex], 
		alphaStar[starIndex], deltaStar[starIndex], fdotStar[starIndex] );
      }
  }


  /* close nstar file */
  fclose(fpStar);

  /* free memory allocated outside skypatches loop */ 
  {
    UINT4 j;
    for (j=0;j< mObsCoh;++j) LALFree( pgV.pg[j].peak); 
  }
  LALFree(pgV.pg);
  
  LALFree(timeDiffV.data);
  LALFree(timeV.data);
  LALFree(velV.data);

  LALFree(weightsV.data);
  
  LALFree(edat->ephemE);
  LALFree(edat->ephemS);
  LALFree(edat);

  LALFree(skyAlpha);
  LALFree(skyDelta);
  LALFree(skySizeAlpha);
  LALFree(skySizeDelta);

  LALFree(nStar);
  LALFree(alphaStar);
  LALFree(deltaStar);
  LALFree(freqStar);
  LALFree(fdotStar);

  SUB (LALDestroyUserVars(&status), &status);
	
  LALCheckMemoryLeaks();
  
  INFO( DRIVEHOUGHCOLOR_MSGENORM );
  return DRIVEHOUGHCOLOR_ENORM;
}



  
/******************************************************************/
/* printing the Histogram of all maps into a file                    */
/******************************************************************/
  
int PrintHistogram(UINT4Vector *hist, CHAR *fnameOut){

  FILE  *fp=NULL;   /* Output file */
  char filename[256];
  UINT4  i ;
 
  strcpy(  filename, fnameOut);
  strcat(  filename, "histo");
  fp=fopen(filename,"w");

  if ( !fp )
    {  
      fprintf(stderr,"Unable to find file %s\n",filename);
      return DRIVEHOUGHCOLOR_EFILE; 
    }

  for (i=0; i < hist->length; i++){
    fprintf(fp,"%d  %d\n", i, hist->data[i]);
  }
  
  fclose( fp );  
  return 0;
}



/******************************************************************/
/* printing the HM into a file                    */
/******************************************************************/

int PrintHmap2file(HOUGHMapTotal *ht, CHAR *fnameOut, INT4 iHmap){

  FILE  *fp=NULL;   /* Output file */
  char filename[256], filenumber[16]; 
  INT4  k, i ;
  UINT2 xSide, ySide;
   
  strcpy(  filename, fnameOut);
  sprintf( filenumber, ".%06d",iHmap); 
  strcat(  filename, filenumber);
  fp=fopen(filename,"w");

  if ( !fp ){  
    fprintf(stderr,"Unable to find file %s\n",filename);
    return DRIVEHOUGHCOLOR_EFILE; 
  }

  ySide= ht->ySide;
  xSide= ht->xSide;

  for(k=ySide-1; k>=0; --k){
    for(i=0;i<xSide;++i){
      fprintf( fp ," %f", (REAL4)ht->map[k*xSide +i]);
      fflush( fp );
    }
    fprintf( fp ," \n");
    fflush( fp );
  }

  fclose( fp );  
  return 0;
}

/* >>>>>>>>>>>>>>>>>>>>>*************************<<<<<<<<<<<<<<<<<<<< */
/* >>>>>>>>>>>>>>>>>>>>>*************************<<<<<<<<<<<<<<<<<<<< */

/******************************************************************/
/* printing the HM into a m_file                    */
/******************************************************************/

int PrintHmap2m_file(HOUGHMapTotal *ht, CHAR *fnameOut, INT4 iHmap){

  FILE  *fp=NULL;   /* Output file */
  char filename[256], filenumber[16]; 
  INT4  k, i ;
  UINT2 xSide, ySide;
  INT4 mObsCoh;
  REAL8 f0,f1;
   
  strcpy(  filename, fnameOut);
  sprintf( filenumber, "%06d.m",iHmap); 
  strcat(  filename, filenumber);
  fp=fopen(filename,"w");

  if ( !fp ){  
    fprintf(stderr,"Unable to find file %s\n",filename);
    return DRIVEHOUGHCOLOR_EFILE; 
  }

  ySide= ht->ySide;
  xSide= ht->xSide;
  f0=ht->f0Bin* ht->deltaF;
  mObsCoh = ht->mObsCoh;
  f1=0.0;
  if( ht->spinRes.length ){ f1=ht->spinRes.data[0]; }
  
  /* printing into matlab format */
  
  fprintf( fp ,"f0= %f ; \n", f0);
  fprintf( fp ,"f1= %g ; \n", f1);
  fprintf( fp ,"map= [ \n");

  for(k=ySide-1; k>=0; --k){
    for(i=0;i<xSide;++i){
      fprintf( fp ," %f", (REAL4)ht->map[k*xSide +i]);
      fflush( fp );
    }
    fprintf( fp ," \n");
    fflush( fp );
  }
  
  fprintf( fp ,"    ]; \n");
  fclose( fp );  
  return 0;
}

/* >>>>>>>>>>>>>>>>>>>>>*************************<<<<<<<<<<<<<<<<<<<< */
/* >>>>>>>>>>>>>>>>>>>>>*************************<<<<<<<<<<<<<<<<<<<< */


/* >>>>>>>>>>>>>>>>>>>>>*************************<<<<<<<<<<<<<<<<<<<< */
/******************************************************************/
/*  Find and print events to a given open file */
/******************************************************************/
void PrintHoughEvents (LALStatus       *status,
        	      FILE            *fpEvents,
	  	      INT4            houghThreshold,
		      HOUGHMapTotal   *ht,
	    	      HOUGHPatchGrid  *patch,
	 	      HOUGHDemodPar   *parDem)
{

  REAL8UnitPolarCoor sourceLocation;
  UINT2    xPos, yPos, xSide, ySide;
  INT4     temp;
  REAL8    f0;
  /* --------------------------------------------- */
  INITSTATUS (status, "PrintHoughEvents", rcsid);
  ATTATCHSTATUSPTR (status);
  
 /* make sure arguments are not null */
  ASSERT (patch , status, DRIVEHOUGHCOLOR_ENULL,DRIVEHOUGHCOLOR_MSGENULL);
  ASSERT (parDem, status, DRIVEHOUGHCOLOR_ENULL,DRIVEHOUGHCOLOR_MSGENULL);
  ASSERT (ht, status, DRIVEHOUGHCOLOR_ENULL,DRIVEHOUGHCOLOR_MSGENULL);

 /* make sure input hough map is ok*/
  ASSERT (ht->xSide > 0, status, DRIVEHOUGHCOLOR_EBAD,DRIVEHOUGHCOLOR_MSGEBAD);
  ASSERT (ht->ySide > 0, status, DRIVEHOUGHCOLOR_EBAD,DRIVEHOUGHCOLOR_MSGEBAD);
  
  /* read input parameters */
  xSide = ht->xSide;
  ySide = ht->ySide; 
  
  f0=(ht->f0Bin)*(ht->deltaF);
  
  for(yPos =0; yPos<ySide; yPos++){
    for(xPos =0; xPos<xSide; xPos++){
      /* read the current number count */
      temp = ht->map[yPos*xSide + xPos];
      if(temp > houghThreshold){
        TRY( LALStereo2SkyLocation(status->statusPtr, 
				&sourceLocation,xPos,yPos,patch, parDem), status);
	if (ht->spinRes.length) {
	  fprintf(fpEvents, "%d %g %g %g %g \n", 
		  temp, sourceLocation.alpha, sourceLocation.delta, 
		  f0, ht->spinRes.data[0]);
	}
	else {
	  fprintf(fpEvents, "%d %g %g %g %g \n", 
		  temp, sourceLocation.alpha, sourceLocation.delta, 
		  f0,0.00);
	}
      }      
    }
  }
  	 
  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}    
/* >>>>>>>>>>>>>>>>>>>>>*************************<<<<<<<<<<<<<<<<<<<< */












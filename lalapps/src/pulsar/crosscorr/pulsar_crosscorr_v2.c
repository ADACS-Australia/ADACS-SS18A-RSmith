/*
 *  Copyright (C) 2013 Badri Krishnan, Shane Larson, John Whelan
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
 * \author B.Krishnan, S.Larson, J.T.Whelan
 * \date 2013
 * \file pulsar_crosscorr_v2.c
 * \ingroup pulsarApps
 * \brief Perform CW cross-correlation search - version 2
 *
 */


/*lalapps includes */
#include <lalapps.h>
#include <lal/UserInput.h>
#include <lal/SFTfileIO.h>
#include <lal/LogPrintf.h>
#include <lal/DopplerScan.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/LALInitBarycenter.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/PulsarCrossCorr_v2.h>

/* user input variables */
typedef struct{
  BOOLEAN help; /**< if the user wants a help message */
  INT4    startTime;          /**< desired start GPS time of search */ 
  INT4    endTime;            /**< desired end GPS time */
  REAL8   maxLag;             /**< maximum lag time in seconds between SFTs in correlation */
  BOOLEAN inclAutoCorr;       /**< include auto-correlation terms (an SFT with itself) */
  REAL8   fStart;             /**< start frequency in Hz */
  REAL8   fBand;              /**< frequency band to search over in Hz */
  /* REAL8   fdotStart;          /\**< starting value for first spindown in Hz/s*\/ */
  /* REAL8   fdotBand;           /\**< range of first spindown to search over in Hz/s *\/ */
  REAL8   alphaRad;           /**< right ascension in radians */
  REAL8   deltaRad;           /**< declination in radians */
  REAL8   refTime;            /**< reference time for pulsar phase definition */
  REAL8   orbitAsiniSec;      /**< start projected semimajor axis in seconds */
  REAL8   orbitAsiniSecBand;  /**< band for projected semimajor axis in seconds */
  REAL8   orbitPSec;          /**< binary orbital period in seconds */
  REAL8   orbitTimeAsc;       /**< start time of ascension for binary orbit */ 
  REAL8   orbitTimeAscBand;   /**< band for time of ascension for binary orbit */ 
  CHAR    *sftLocation;       /**< location of SFT data */
  CHAR    *ephemYear;         /**< range of years for ephemeris file */
  INT4    rngMedBlock;        /**< running median block size */
} UserInput_t;

/* struct to store useful variables */
typedef struct{
  SFTCatalog *catalog; /**< catalog of SFTs */  
  EphemerisData *edat; /**< ephemeris data */
} ConfigVariables;


#define DEFAULT_EPHEMDIR "env LAL_DATA_PATH"
#define EPHEM_YEARS "00-19-DE405"

#define TRUE (1==1)
#define FALSE (1==0)
#define MAXFILENAMELENGTH 512

/* empty user input struct for initialization */
UserInput_t empty_UserInput;

/* local function prototypes */
int XLALInitUserVars ( UserInput_t *uvar );
int XLALInitializeConfigVars (ConfigVariables *config, const UserInput_t *uvar);

int main(int argc, char *argv[]){

  UserInput_t uvar = empty_UserInput;
  static ConfigVariables config;

  /* sft related variables */ 
  MultiSFTVector *inputSFTs = NULL;
  MultiPSDVector *multiPSDs = NULL;
  MultiNoiseWeights *multiWeights = NULL;
  MultiLIGOTimeGPSVector *multiTimes = NULL;
  MultiLALDetector *multiDetectors = NULL;
  MultiDetectorStateSeries *multiStates = NULL;
  MultiAMCoeffs *multiCoeffs = NULL;
  SFTIndexList *sftIndices = NULL;
  SFTPairIndexList *sftPairs = NULL;

  SkyPosition skyPos = empty_SkyPosition;

  REAL8 fMin, fMax; /* min and max frequencies read from SFTs */
  REAL8 deltaF; /* frequency resolution associated with time baseline of SFTs */

  REAL8Vector *curlyGUnshifted = NULL;

  /* initialize and register user variables */
  if ( XLALInitUserVars( &uvar ) != XLAL_SUCCESS ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALInitUserVars() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* read user input from the command line or config file */
  if ( XLALUserVarReadAllInput ( argc, argv ) != XLAL_SUCCESS ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALUserVarReadAllInput() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  if (uvar.help)	/* if help was requested, then exit */
    return 0;
 
  /* configure useful variables based on user input */
  if ( XLALInitializeConfigVars ( &config, &uvar) != XLAL_SUCCESS ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALInitUserVars() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  deltaF = config.catalog->data[0].header.deltaF;

  /* now read the data */
  /* FIXME: need to correct fMin and fMax for Doppler shift, rngmedian bins and spindown range */
  /* this is essentially just a place holder for now */
  /* FIXME: this running median buffer is overkill, since the running median block need not be centered on the search frequency */
  fMin = uvar.fStart - 0.5 * uvar.rngMedBlock * deltaF;
  fMax = uvar.fStart + 0.5 * uvar.rngMedBlock * deltaF + uvar.fBand;

  /* read the SFTs*/
  if ((inputSFTs = XLALLoadMultiSFTs ( config.catalog, fMin, fMax)) == NULL){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALLoadMultiSFTs() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* calculate the psd and normalize the SFTs */
  if (( multiPSDs =  XLALNormalizeMultiSFTVect ( inputSFTs, uvar.rngMedBlock )) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALNormalizeMultiSFTVect() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* compute the noise weights for the AM coefficients */
  if (( multiWeights = XLALComputeMultiNoiseWeights ( multiPSDs, uvar.rngMedBlock, 0 )) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALComputeMultiNoiseWeights() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* read the timestamps from the SFTs */
  if ((multiTimes = XLALExtractMultiTimestampsFromSFTs ( inputSFTs )) == NULL){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALExtractMultiTimestampsFromSFTs() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* read the detector information from the SFTs */
  if ((multiDetectors = XLALExtractMultiLALDetectorFromSFTs ( inputSFTs )) == NULL){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALExtractMultiLALDetectorFromSFTs() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Find the detector state for each SFT */
  if ((multiStates = XLALGetMultiDetectorStates ( multiTimes, multiDetectors, config.edat, 0.0 )) == NULL){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALGetMultiDetectorStates() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Note this is specialized to a single sky position */
  /* This might need to be moved into the config variables */
  skyPos.system = COORDINATESYSTEM_EQUATORIAL;
  skyPos.longitude = uvar.alphaRad;
  skyPos.latitude  = uvar.deltaRad;

  /* Calculate the AM coefficients (a,b) for each SFT */
  if ((multiCoeffs = XLALComputeMultiAMCoeffs ( multiStates, multiWeights, skyPos )) == NULL){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALGetMultiDetectorStates() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

 /* Call XLALWeightMultiAMCoffs, replace AM-coeffs by weighted AM-coeffs*/
  if ( ( XLALWeightMultiAMCoeffs (multiCoeffs, multiWeights ))!= XLAL_SUCCESS){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALWeightMultiAMCoeffs() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Construct the flat list of SFTs (this sort of replicates the
     catalog, but there's not an obvious way to get the information
     back) */

  if ( ( XLALCreateSFTIndexListFromMultiSFTVect( &sftIndices, inputSFTs ) != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCreateSFTIndexListFromMultiSFTVect() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Construct the list of SFT pairs */

  if ( ( XLALCreateSFTPairIndexList( &sftPairs, sftIndices, inputSFTs, uvar.maxLag, uvar.inclAutoCorr ) != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCreateSFTPairIndexList() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Get weighting factors for calculation of metric */
  /* note that the sigma-squared is now absorbed into the curly G
     because the AM coefficients are noise-weighted. */
  if ( ( XLALCalculateAveCurlyGAmpUnshifted( &curlyGUnshifted, sftPairs, sftIndices, multiCoeffs)  != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCalculateAveCurlyGUnshifted() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* /\* get SFT parameters so that we can initialise search frequency resolutions *\/ */
  /* /\* calculate deltaF_SFT *\/ */
  /* deltaF_SFT = catalog->data[0].header.deltaF;  /\* frequency resolution *\/ */
  /* timeBase= 1.0/deltaF_SFT; /\* sft baseline *\/ */

  /* /\* catalog is ordered in time so we can get start, end time and tObs *\/ */
  /* firstTimeStamp = catalog->data[0].header.epoch; */
  /* lastTimeStamp = catalog->data[catalog->length - 1].header.epoch; */
  /* tObs = XLALGPSDiff( &lastTimeStamp, &firstTimeStamp ) + timeBase; */

  /* /\*set pulsar reference time *\/ */
  /* if (LALUserVarWasSet ( &uvar_refTime )) { */
  /*   XLALGPSSetREAL8(&refTime, uvar_refTime); */
  /* }  */
  /* else {	/\*if refTime is not set, set it to midpoint of sfts*\/ */
  /*   XLALGPSSetREAL8(&refTime, (0.5*tObs) + XLALGPSGetREAL8(&firstTimeStamp));  */
  /* } */

  /* /\* set frequency resolution defaults if not set by user *\/ */
  /* if (!(LALUserVarWasSet (&uvar_fResolution))) { */
  /*   uvar_fResolution = 1/tObs; */
  /* } */



  /* { */
  /*   /\* block for calculating frequency range to read from SFTs *\/ */
  /*   /\* user specifies freq and fdot range at reftime */
  /*      we translate this range of fdots to start and endtime and find */
  /*      the largest frequency band required to cover the  */
  /*      frequency evolution  *\/ */
  /*   PulsarSpinRange spinRange_startTime; /\**< freq and fdot range at start-time of observation *\/ */
  /*   PulsarSpinRange spinRange_endTime;   /\**< freq and fdot range at end-time of observation *\/ */
  /*   PulsarSpinRange spinRange_refTime;   /\**< freq and fdot range at the reference time *\/ */

  /*   REAL8 startTime_freqLo, startTime_freqHi, endTime_freqLo, endTime_freqHi, freqLo, freqHi; */

  /*   REAL8Vector *fdotsMin=NULL; */
  /*   REAL8Vector *fdotsMax=NULL; */

  /*   UINT4 k; */

  /*   fdotsMin = (REAL8Vector *)LALCalloc(1, sizeof(REAL8Vector)); */
  /*   fdotsMin->length = N_SPINDOWN_DERIVS; */
  /*   fdotsMin->data = (REAL8 *)LALCalloc(fdotsMin->length, sizeof(REAL8)); */

  /*   fdotsMax = (REAL8Vector *)LALCalloc(1, sizeof(REAL8Vector)); */
  /*   fdotsMax->length = N_SPINDOWN_DERIVS; */
  /*   fdotsMax->data = (REAL8 *)LALCalloc(fdotsMax->length, sizeof(REAL8)); */

  /*   INIT_MEM(spinRange_startTime); */
  /*   INIT_MEM(spinRange_endTime); */
  /*   INIT_MEM(spinRange_refTime); */

  /*   spinRange_refTime.refTime = refTime; */
  /*   spinRange_refTime.fkdot[0] = uvar_f0; */
  /*   spinRange_refTime.fkdotBand[0] = uvar_fBand; */
  /* } */


  /* XLALDestroySFTPairIndexList(sftPairs) */
  /* XLALDestroySFTIndexList(sftIndices) */
  XLALDestroyMultiSFTVector ( inputSFTs ); 
  XLALDestroyMultiPSDVector ( multiPSDs );

  XLALDestroySFTCatalog (config.catalog );
  XLALFree( config.edat->ephemE );
  XLALFree( config.edat->ephemS );
  XLALFree( config.edat );

  /* de-allocate memory for user input variables */
  XLALDestroyUserVars();

  /* check memory leaks if we forgot to de-allocate anything */
  LALCheckMemoryLeaks();

  return 0;

} /* main */


/* initialize and register user variables */
int XLALInitUserVars (UserInput_t *uvar)
{

  /* initialize with some defaults */
  uvar->help = FALSE;
  uvar->startTime = 814838413;	/* 1 Nov 2005, ~ start of S5 */
  uvar->endTime = uvar->startTime + (INT4) round ( LAL_YRSID_SI ) ;	/* 1 year of data */
  uvar->maxLag = 0.0;
  uvar->inclAutoCorr = FALSE;
  uvar->fStart = 100.0; 
  uvar->fBand = 0.1;
  /* uvar->fdotStart = 0.0; */
  /* uvar->fdotBand = 0.0; */
  uvar->alphaRad = 0.0;
  uvar->deltaRad = 0.0;
  uvar->rngMedBlock = 50;

  /* default for reftime is in the middle */
  uvar->refTime = 0.5*(uvar->startTime + uvar->endTime);

  /* zero binary orbital parameters means not a binary */
  uvar->orbitAsiniSec = 0.0;
  uvar->orbitAsiniSecBand = 0.0;
  uvar->orbitPSec = 0.0;
  uvar->orbitTimeAsc = 0;
  uvar->orbitTimeAscBand = 0;

  uvar->ephemYear = XLALCalloc (1, strlen(EPHEM_YEARS)+1);
  strcpy (uvar->ephemYear, EPHEM_YEARS);

  uvar->sftLocation = XLALCalloc(1, MAXFILENAMELENGTH+1);

  /* register  user-variables */
  XLALregBOOLUserStruct ( help, 	 'h',  UVAR_HELP, "Print this message");  
  
  XLALregINTUserStruct   ( startTime,     0,  UVAR_OPTIONAL, "Desired start time of analysis in GPS seconds");
  XLALregINTUserStruct   ( endTime,       0,  UVAR_OPTIONAL, "Desired end time of analysis in GPS seconds");
  XLALregREALUserStruct  ( maxLag,        0,  UVAR_OPTIONAL, "Maximum lag time in seconds between SFTs in correlation");
  XLALregBOOLUserStruct  ( inclAutoCorr,  0,  UVAR_OPTIONAL, "Include auto-correlation terms (an SFT with itself)");
  XLALregREALUserStruct  ( fStart,        0,  UVAR_OPTIONAL, "Start frequency in Hz");
  XLALregREALUserStruct  ( fBand,         0,  UVAR_OPTIONAL, "Frequency band to search over in Hz ");
  /* XLALregREALUserStruct  ( fdotStart,     0,  UVAR_OPTIONAL, "Start value of spindown in Hz/s"); */
  /* XLALregREALUserStruct  ( fdotBand,      0,  UVAR_OPTIONAL, "Band for spindown values in Hz/s"); */
  XLALregREALUserStruct  ( alphaRad,      0,  UVAR_OPTIONAL, "Right ascension for directed search (radians)");
  XLALregREALUserStruct  ( deltaRad,      0,  UVAR_OPTIONAL, "Declination for directed search (radians)");
  XLALregREALUserStruct  ( refTime,       0,  UVAR_OPTIONAL, "SSB reference time for pulsar-parameters [Default: midPoint]");
  XLALregREALUserStruct  ( orbitAsiniSec, 0,  UVAR_OPTIONAL, "Start of search band for projected semimajor axis (seconds) [0 means not a binary]");
  XLALregREALUserStruct  ( orbitAsiniSecBand, 0,  UVAR_OPTIONAL, "Width of search band for projected semimajor axis (seconds)");
  XLALregREALUserStruct  ( orbitPSec,     0,  UVAR_OPTIONAL, "Binary orbital period (seconds) [0 means not a binary]");
  XLALregREALUserStruct  ( orbitTimeAsc,  0,  UVAR_OPTIONAL, "Start of orbital time-of-ascension band in GPS seconds");
  XLALregREALUserStruct  ( orbitTimeAscBand, 0,  UVAR_OPTIONAL, "Width of orbital time-of-ascension band (seconds)");
  XLALregSTRINGUserStruct( ephemYear,     0,  UVAR_OPTIONAL, "String Ephemeris year range");
  XLALregSTRINGUserStruct( sftLocation,   0,  UVAR_REQUIRED, "Filename pattern for locating SFT data");
  XLALregINTUserStruct   ( rngMedBlock,   0,  UVAR_OPTIONAL, "Running median block size for PSD estimation");

  if ( xlalErrno ) {
    XLALPrintError ("%s: user variable initialization failed with errno = %d.\n", __func__, xlalErrno );
    XLAL_ERROR ( XLAL_EFUNC );
  }

  return XLAL_SUCCESS;
}



/* initialize and register user variables */
int XLALInitializeConfigVars (ConfigVariables *config, const UserInput_t *uvar)
{

  static SFTConstraints constraints;
  LIGOTimeGPS startTime, endTime;
  CHAR EphemEarth[MAXFILENAMELENGTH]; /* file with earth-ephemeris data */
  CHAR EphemSun[MAXFILENAMELENGTH];	/* file with sun-ephemeris data */


  /* set sft catalog constraints */
  constraints.detector = NULL;
  constraints.timestamps = NULL;
  constraints.startTime = &startTime;
  constraints.endTime = &endTime;
  XLALGPSSet( constraints.startTime, uvar->startTime, 0);
  XLALGPSSet( constraints.endTime, uvar->endTime,0); 

  /* This check doesn't seem to work, since XLALGPSSet doesn't set its
     first argument.

  if ( (constraints.startTime == NULL)&& (constraints.endTime == NULL) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALGPSSet() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  */

  /* get catalog of SFTs */
  if ((config->catalog = XLALSFTdataFind (uvar->sftLocation, &constraints)) == NULL){ 
    LogPrintf ( LOG_CRITICAL, "%s: XLALSFTdataFind() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* initialize ephemeris data*/
  /* first check input consistency */
  if ( uvar->ephemYear == NULL) {
    XLALPrintError ("%s: invalid NULL input for 'ephemYear'\n", __func__ );
    XLAL_ERROR ( XLAL_EINVAL );
  }

  /* construct ephemeris file names from ephemeris year input*/
  snprintf(EphemEarth, MAXFILENAMELENGTH, "earth%s.dat", uvar->ephemYear);
  snprintf(EphemSun, MAXFILENAMELENGTH, "sun%s.dat",  uvar->ephemYear);
  EphemEarth[MAXFILENAMELENGTH-1]=0;
  EphemSun[MAXFILENAMELENGTH-1]=0;

  /* now call initbarycentering routine */
  if ( (config->edat = XLALInitBarycenter ( EphemEarth, EphemSun)) == NULL ) {
    XLALPrintError ("%s: XLALInitBarycenter() failed.\n", __func__ );
    XLAL_ERROR ( XLAL_EFUNC );
  }
  
  return XLAL_SUCCESS;

} /* XLALInitializeConfigVars() */

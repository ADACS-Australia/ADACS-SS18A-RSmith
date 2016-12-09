/*
 *  Copyright (C) 2013 Badri Krishnan, Shane Larson, John Whelan
 *  Copyright (C) 2013, 2014 Badri Krishnan, John Whelan, Yuanhao Zhang
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

/*lalapps includes */
#include <lalapps.h>
#include <lal/UserInput.h>
#include <lal/SFTfileIO.h>
#include <lal/SFTutils.h>
#include <lal/LogPrintf.h>
#include <lal/DopplerScan.h>
#include <lal/DetectorStates.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/LALInitBarycenter.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/LALString.h>
#include <lal/PulsarCrossCorr_v2.h>
#include "CrossCorrToplist.h"

/**
 * \author B.Krishnan, S.Larson, J.T.Whelan, Y.Zhang
 * \date 2013, 2014, 2015
 * \file
 * \ingroup lalapps_pulsar_CrossCorr
 * \brief Perform CW cross-correlation search - version 2
 *
 * This carries out the cross-correlation search defined in
 * \cite Dhurandhar2007, and specifically the implementation detailed
 * in \cite Whelan2015 .  The SFT-normalization routines described in
 * \cite T0900149-v5 are leveraged.
 */
/** @{ */

/* introduce mismatch in f and all 5 binary parameters */

/* user input variables */
typedef struct{
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
  REAL8   orbitPSecBand;      /**< band for binary orbital period in seconds*/
  REAL8   orbitTimeAsc;       /**< start time of ascension for binary orbit */
  REAL8   orbitTimeAscBand;   /**< band for time of ascension for binary orbit */
  CHAR    *sftLocation;       /**< location of SFT data */
  CHAR    *ephemEarth;	      /**< Earth ephemeris file to use */
  CHAR    *ephemSun;	      /**< Sun ephemeris file to use */
  INT4    rngMedBlock;        /**< running median block size */
  INT4    numBins;            /**< number of frequency bins to include in sum */
  REAL8   mismatchF;          /**< mismatch for frequency spacing */
  REAL8   mismatchA;          /**< mismatch for spacing in semi-major axis */
  REAL8   mismatchT;          /**< mismatch for spacing in time of periapse passage */
  REAL8   mismatchP;          /**< mismatch for spacing in period */
  REAL8   spacingF;           /**< spacing in frequency */
  REAL8   spacingA;           /**< spacing in semi-major axis*/
  REAL8   spacingT;           /**< spacing in time of periapse passage*/
  REAL8   spacingP;           /**< spacing in period*/
  INT4    numCand;            /**< number of candidates to keep in output toplist */
  CHAR    *linesToCleanFilenames; /**< comma-separated list of filenames with known lines for each ifo */
  CHAR    *pairListInputFilename;  /**< input filename containing list of sft index pairs (if not provided, determine list of pairs */
  CHAR    *pairListOutputFilename; /**< output filename to write list of sft index pairs */
  CHAR    *sftListOutputFilename;  /**< output filename to write list of sfts */
  CHAR    *sftListInputFilename;   /**< input filename to read in the  list of sfts and check the order of SFTs */
  CHAR    *gammaAveOutputFilename; /**< output filename to write Gamma_ave = (aa+bb)/10 */
  CHAR    *gammaCircOutputFilename; /**< output filename to write Gamma_circ = (ab-ba)/10 */
  CHAR    *toplistFilename;   /**< output filename containing candidates in toplist */
  CHAR    *logFilename;       /**< name of log file*/
} UserInput_t;

/* struct to store useful variables */
typedef struct{
  SFTCatalog *catalog; /**< catalog of SFTs */
  EphemerisData *edat; /**< ephemeris data */
  LALStringVector *lineFiles; /**< list of line files */
  REAL8   refTime;     /**< reference time for pulsar phase definition */
} ConfigVariables;

#define TRUE (1==1)
#define FALSE (1==0)
#define MAXFILENAMELENGTH 512
#define MAXLINELENGTH 1024

/* local function prototypes */
int XLALInitUserVars ( UserInput_t *uvar );
int XLALInitializeConfigVars (ConfigVariables *config, const UserInput_t *uvar);
int XLALDestroyConfigVars (ConfigVariables *config);
int GetNextCrossCorrTemplate(BOOLEAN *binaryParamsFlag, BOOLEAN *firstPoint, PulsarDopplerParams *dopplerpos, PulsarDopplerParams *binaryTemplateSpacings, PulsarDopplerParams *minBinaryTemplate, PulsarDopplerParams *maxBinaryTemplate, UINT8 *fCount, UINT8 *aCount, UINT8 *tCount, UINT8 *pCount, UINT8 fSpacingNum, UINT8 aSpacingNum, UINT8 tSpacingNum, UINT8 pSpacingNum);
UINT4 pcc_count_csv( CHAR *csvline );
INT4 XLALFindBadBins ( UINT4Vector *badBinData, INT4 binCount, REAL8 flo, REAL8 fhi, REAL8 f0, REAL8 deltaF, UINT4 length) ;
/** @} */

int main(int argc, char *argv[]){

  UserInput_t XLAL_INIT_DECL(uvar);
  static ConfigVariables config;

  /* sft related variables */
  MultiSFTVector *inputSFTs = NULL;
  MultiPSDVector *multiPSDs = NULL;
  MultiNoiseWeights *multiWeights = NULL;
  MultiLIGOTimeGPSVector *multiTimes = NULL;
  MultiLALDetector multiDetectors;
  MultiDetectorStateSeries *multiStates = NULL;
  MultiAMCoeffs *multiCoeffs = NULL;
  SFTIndexList *sftIndices = NULL;
  SFTPairIndexList *sftPairs = NULL;
  REAL8Vector *shiftedFreqs = NULL;
  UINT4Vector *lowestBins = NULL;
  COMPLEX8Vector *expSignalPhases = NULL;
  REAL8VectorSequence *sincList = NULL;
  PulsarDopplerParams XLAL_INIT_DECL(dopplerpos);
  PulsarDopplerParams thisBinaryTemplate, binaryTemplateSpacings;
  PulsarDopplerParams minBinaryTemplate, maxBinaryTemplate;
  SkyPosition XLAL_INIT_DECL(skyPos);
  MultiSSBtimes *multiBinaryTimes = NULL;

  INT4  k;
  UINT4 j;
  REAL8 fMin, fMax; /* min and max frequencies read from SFTs */
  REAL8 deltaF; /* frequency resolution associated with time baseline of SFTs */

  BOOLEAN dopplerShiftFlag = TRUE;
  toplist_t *ccToplist=NULL;
  CrossCorrBinaryOutputEntry thisCandidate;
  UINT4 checksum;
  CHARVector *dimName = NULL;

  LogPrintf (LOG_CRITICAL, "Starting time\n"); /*for debug convenience to record calculating time*/
  /* initialize and register user variables */
  LIGOTimeGPS computingStartGPSTime, computingEndGPSTime;
  XLALGPSTimeNow (&computingStartGPSTime); /* record the rough starting GPS time*/

  if ( XLALInitUserVars( &uvar ) != XLAL_SUCCESS ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALInitUserVars() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* read user input from the command line or config file */
  BOOLEAN should_exit = 0;
  if ( XLALUserVarReadAllInput ( &should_exit, argc, argv, lalAppsVCSInfoList ) != XLAL_SUCCESS ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALUserVarReadAllInput() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }
  if (should_exit)
    return EXIT_FAILURE;

  CHAR *VCSInfoString = XLALGetVersionString(0);     /**<LAL + LALapps Vsersion string*/

  /* configure useful variables based on user input */
  if ( XLALInitializeConfigVars ( &config, &uvar) != XLAL_SUCCESS ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALInitUserVars() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  deltaF = config.catalog->data[0].header.deltaF;
  REAL8 Tsft = 1.0 / deltaF;

  if (XLALUserVarWasSet(&uvar.spacingF) && XLALUserVarWasSet(&uvar.mismatchF))
    LogPrintf (LOG_CRITICAL, "spacingF and mismatchF are both set, use spacingF %.9g by default\n\n", uvar.spacingF);
  if (XLALUserVarWasSet(&uvar.spacingA) && XLALUserVarWasSet(&uvar.mismatchA))
    LogPrintf (LOG_CRITICAL, "spacingA and mismatchA are both set, use spacingA %.9g by default\n\n", uvar.spacingA);
  if (XLALUserVarWasSet(&uvar.spacingT) && XLALUserVarWasSet(&uvar.mismatchT))
    LogPrintf (LOG_CRITICAL, "spacingT and mismatchT are both set, use spacingT %.9g by default\n\n", uvar.spacingT);
  if (XLALUserVarWasSet(&uvar.spacingP) && XLALUserVarWasSet(&uvar.mismatchP))
    LogPrintf (LOG_CRITICAL, "spacingP and mismatchP are both set, use spacingP %.9g by default\n\n", uvar.spacingP);

  /* create the toplist */
  create_crossCorrBinary_toplist( &ccToplist, uvar.numCand);
  /* now read the data */

  /* /\* get SFT parameters so that we can initialise search frequency resolutions *\/ */
  /* /\* calculate deltaF_SFT *\/ */
  /* deltaF_SFT = catalog->data[0].header.deltaF;  /\* frequency resolution *\/ */
  /* timeBase= 1.0/deltaF_SFT; /\* sft baseline *\/ */

  /* /\* catalog is ordered in time so we can get start, end time and tObs *\/ */
  /* firstTimeStamp = catalog->data[0].header.epoch; */
  /* lastTimeStamp = catalog->data[catalog->length - 1].header.epoch; */
  /* tObs = XLALGPSDiff( &lastTimeStamp, &firstTimeStamp ) + timeBase; */

  /* /\*set pulsar reference time *\/ */
  /* if (XLALUserVarWasSet ( &uvar_refTime )) { */
  /*   XLALGPSSetREAL8(&refTime, uvar_refTime); */
  /* }  */
  /* else {	/\*if refTime is not set, set it to midpoint of sfts*\/ */
  /*   XLALGPSSetREAL8(&refTime, (0.5*tObs) + XLALGPSGetREAL8(&firstTimeStamp));  */
  /* } */

  /* /\* set frequency resolution defaults if not set by user *\/ */
  /* if (!(XLALUserVarWasSet (&uvar_fResolution))) { */
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

  /*   XLAL_INIT_MEM(spinRange_startTime); */
  /*   XLAL_INIT_MEM(spinRange_endTime); */
  /*   XLAL_INIT_MEM(spinRange_refTime); */

  /*   spinRange_refTime.refTime = refTime; */
  /*   spinRange_refTime.fkdot[0] = uvar_f0; */
  /*   spinRange_refTime.fkdotBand[0] = uvar_fBand; */
  /* } */

  /* FIXME: need to correct fMin and fMax for Doppler shift, rngmedian bins and spindown range */
  /* this is essentially just a place holder for now */
  /* FIXME: this running median buffer is overkill, since the running median block need not be centered on the search frequency */
  REAL8 vMax = LAL_TWOPI * (uvar.orbitAsiniSec + uvar.orbitAsiniSecBand) / uvar.orbitPSec + LAL_TWOPI * LAL_REARTH_SI / (LAL_DAYSID_SI * LAL_C_SI) + LAL_TWOPI * LAL_AU_SI/(LAL_YRSID_SI * LAL_C_SI); /*calculate the maximum relative velocity in speed of light*/
  fMin = uvar.fStart * (1 - vMax) - 0.5 * uvar.rngMedBlock * deltaF;
  fMax = (uvar.fStart + uvar.fBand) * (1 + vMax) + 0.5 * uvar.rngMedBlock * deltaF;

  /* read the SFTs*/
  if ((inputSFTs = XLALLoadMultiSFTs ( config.catalog, fMin, fMax)) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALLoadMultiSFTs() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* calculate the psd and normalize the SFTs */
  if (( multiPSDs =  XLALNormalizeMultiSFTVect ( inputSFTs, uvar.rngMedBlock, NULL )) == NULL){
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
  if ( XLALMultiLALDetectorFromMultiSFTs ( &multiDetectors, inputSFTs ) != XLAL_SUCCESS){
    LogPrintf ( LOG_CRITICAL, "%s: XLALMultiLALDetectorFromMultiSFTs() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Find the detector state for each SFT */
  /* Offset by Tsft/2 to get midpoint as timestamp */
  if ((multiStates = XLALGetMultiDetectorStates ( multiTimes, &multiDetectors, config.edat, 0.5 * Tsft )) == NULL){
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
    LogPrintf ( LOG_CRITICAL, "%s: XLALComputeMultiAMCoeffs() failed with errno=%d\n", __func__, xlalErrno );
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
#define PCC_SFTPAIR_HEADER "# The length of SFT-pair list is %u #\n"
#define PCC_SFTPAIR_BODY "%u %u\n"
#define PCC_SFT_HEADER "# The length of SFT list is %u #\n"
#define PCC_SFT_BODY "%s %d %d\n"
  FILE *fp = NULL;

  if (XLALUserVarWasSet(&uvar.pairListInputFilename)) { /* If the user provided a list for reading, use it */
    if((sftPairs = XLALCalloc(1, sizeof(sftPairs))) == NULL){
      XLAL_ERROR(XLAL_ENOMEM);
    }
    if((fp = fopen(uvar.pairListInputFilename, "r")) == NULL){
      LogPrintf ( LOG_CRITICAL, "didn't find SFT-pair list file with given input name\n");
      XLAL_ERROR( XLAL_EFUNC );
    }
    if(fscanf(fp,PCC_SFTPAIR_HEADER,&sftPairs->length)==EOF){
      LogPrintf ( LOG_CRITICAL, "can't read the length of SFT-pair list from the header\n");
      XLAL_ERROR( XLAL_EFUNC );
    }

    if((sftPairs->data = XLALCalloc(sftPairs->length, sizeof(*sftPairs->data)))==NULL){
      XLALFree(sftPairs);
      XLAL_ERROR(XLAL_ENOMEM);
    }

    for(j = 0; j < sftPairs->length; j++){ /*read in  the SFT-pair list */
      if(fscanf(fp,PCC_SFTPAIR_BODY, &sftPairs->data[j].sftNum[0], &sftPairs->data[j].sftNum[1])==EOF){
	LogPrintf ( LOG_CRITICAL, "The length of SFT-pair list doesn't match!");
	XLAL_ERROR( XLAL_EFUNC );
      }
    }
    fclose(fp);

  }

  else { /* if not, construct the list of pairs */
    if ( ( XLALCreateSFTPairIndexList( &sftPairs, sftIndices, inputSFTs, uvar.maxLag, uvar.inclAutoCorr ) != XLAL_SUCCESS ) ) {
      LogPrintf ( LOG_CRITICAL, "%s: XLALCreateSFTPairIndexList() failed with errno=%d\n", __func__, xlalErrno );
      XLAL_ERROR( XLAL_EFUNC );
    }
  }

  if (XLALUserVarWasSet(&uvar.pairListOutputFilename)) { /* Write the list of pairs to a file, if a name was provided */
    if((fp = fopen(uvar.pairListOutputFilename, "w")) == NULL){
      LogPrintf ( LOG_CRITICAL, "Can't write in SFT-pair list \n");
      XLAL_ERROR( XLAL_EFUNC );
    }
    fprintf(fp,PCC_SFTPAIR_HEADER, sftPairs->length ); /*output the length of SFT-pair list to the header*/
    for(j = 0; j < sftPairs->length; j++){
      fprintf(fp,PCC_SFTPAIR_BODY, sftPairs->data[j].sftNum[0], sftPairs->data[j].sftNum[1]);
    }
    fclose(fp);
  }

  if (XLALUserVarWasSet(&uvar.sftListOutputFilename)) { /* Write the list of SFTs to a file for sanity-checking purposes */
    if((fp = fopen(uvar.sftListOutputFilename, "w")) == NULL){
      LogPrintf ( LOG_CRITICAL, "Can't write in flat SFT list \n");
      XLAL_ERROR( XLAL_EFUNC );
    }
    fprintf(fp,PCC_SFT_HEADER, sftIndices->length ); /*output the length of SFT list to the header*/
    for(j = 0; j < sftIndices->length; j++){ /*output the SFT list */
      fprintf(fp,PCC_SFT_BODY, inputSFTs->data[sftIndices->data[j].detInd]->data[sftIndices->data[j].sftInd].name, inputSFTs->data[sftIndices->data[j].detInd]->data[sftIndices->data[j].sftInd].epoch.gpsSeconds, inputSFTs->data[sftIndices->data[j].detInd]->data[sftIndices->data[j].sftInd].epoch.gpsNanoSeconds);
    }
    fclose(fp);
  }

  else if(XLALUserVarWasSet(&uvar.sftListInputFilename)){ /*do a sanity check of the order of SFTs list if the name of input SFT list is given*/
    UINT4 numofsft=0;
    if((fp = fopen(uvar.sftListInputFilename, "r")) == NULL){
      LogPrintf ( LOG_CRITICAL, "Can't read in flat SFT list \n");
      XLAL_ERROR( XLAL_EFUNC );
    }
    if (fscanf(fp, PCC_SFT_HEADER, &numofsft)==EOF){
      LogPrintf ( LOG_CRITICAL, "can't read in the length of SFT list from header\n");
      XLAL_ERROR( XLAL_EFUNC );
    }

    CHARVectorSequence *checkDet=NULL;
    if ((checkDet = XLALCreateCHARVectorSequence (numofsft, LALNameLength) ) == NULL){
      LogPrintf ( LOG_CRITICAL, "%s: XLALCreateCHARVector() failed with errno=%d\n", __func__, xlalErrno );
      XLAL_ERROR( XLAL_EFUNC );
    }
    INT4 checkGPS[numofsft], checkGPSns[numofsft];
    if(numofsft == sftIndices->length){
      for (j=0; j<numofsft; j++){
	if( fscanf(fp,PCC_SFT_BODY,&checkDet->data[j * LALNameLength], &checkGPS[j], &checkGPSns[j])==EOF){
	  LogPrintf ( LOG_CRITICAL, "The length of SFT list doesn't match\n");
	  XLAL_ERROR( XLAL_EFUNC );
	}
	if(strcmp( inputSFTs->data[sftIndices->data[j].detInd]->data[sftIndices->data[j].sftInd].name, &checkDet->data[j * LALNameLength] ) != 0
	   ||inputSFTs->data[sftIndices->data[j].detInd]->data[sftIndices->data[j].sftInd].epoch.gpsSeconds != checkGPS[j]
	   ||inputSFTs->data[sftIndices->data[j].detInd]->data[sftIndices->data[j].sftInd].epoch.gpsNanoSeconds != checkGPSns[j] ){
	  LogPrintf ( LOG_CRITICAL, "The order of SFTs has been changed, it's the end of civilization\n");
	  XLAL_ERROR( XLAL_EFUNC );
	}
      }
      fclose(fp);
      XLALDestroyCHARVectorSequence(checkDet);
    }
    else{
      LogPrintf ( LOG_CRITICAL, "Run for your life, the length of SFT list doesn't match");
      XLAL_ERROR( XLAL_EFUNC );
    }
  }
  else
    {

    }

  /* Parse the list of lines to avoid (if given) */

  MultiUINT4Vector *badBins = NULL;

#define PCC_LINEFILE_HEADER "%% %2s lines cleaning file for O1\n"\
"%%\n"\
"%% File contains %d (non-comment) lines\n"\
"%%\n"\
"%% Column 1 - frequency spacing (Hz) of comb (or frequency of single line)\n"\
"%% Column 2 - comb type (0 - singlet, 1 - comb with fixed width, 2 - comb with scaling width)\n"\
"%% Column 3 - frequency offset of 1st visible harmonic (Hz)\n"\
"%% Column 4 - index of first visible harmonic\n"\
"%% Column 5 - index of last visible harmonic\n"\
"%% Column 6 - width of left band (Hz)\n"\
"%% Column 7 - width of right band (Hz)\n"\
"%%\n"\
"%% For fixed-width combs, veto the band:\n"\
"%%     [offset+index*spacing-leftwidth, offset+index*spacing+rightwidth]\n"\
"%% For scaling-width combs, veto the band:\n"\
"%%     [offset+index*spacing-index*leftwidth, offset+index*spacing+index*rightwidth]\n"\
"%%"
#define PCC_LINEFILE_BODY "%lf %d %lf %d %d %lf %lf"

  if ( config.lineFiles != NULL) {
    if((badBins = XLALCalloc(1, sizeof(badBins))) == NULL){
      XLAL_ERROR(XLAL_ENOMEM);
    }
    UINT4 numDets = inputSFTs->length;
    badBins->length = numDets;
    if((badBins->data = XLALCalloc(numDets, sizeof(*badBins->data))) == NULL){
      XLAL_ERROR(XLAL_ENOMEM);
    }

    for ( UINT4 i_f=0 ; i_f < config.lineFiles->length ; i_f++ ) {
      /* printf("i_f=%d\n",i_f); */
      if((fp = fopen(config.lineFiles->data[i_f], "r")) == NULL){
	LogPrintf ( LOG_CRITICAL, "%s: didn't find line file with name %s\n", __func__, config.lineFiles->data[i_f]);
	XLAL_ERROR( XLAL_EINVAL );
      }
      CHAR ifo[3];
      UINT4 numLines;
      if(fscanf(fp,PCC_LINEFILE_HEADER,&ifo[0],&numLines)==EOF){
	LogPrintf ( LOG_CRITICAL, "can't parse header of line file %s\n",config.lineFiles->data[i_f]);
	XLAL_ERROR( XLAL_EINVAL );
      }
      fclose(fp);

      UINT4 detInd = 0;
      for ( ; detInd < numDets ; detInd++ ){
	if ( strcmp(ifo,inputSFTs->data[detInd]->data[0].name) == 0 ) break;
      } /*find the detctor index of the ifo and break the loop*/
      if (detInd >= numDets){
	LogPrintf ( LOG_CRITICAL, "%s: didn't find index for IFO %s\n", __func__, ifo);
	XLAL_ERROR( XLAL_EINVAL );
      }
      UINT4 numSFTFreqs = inputSFTs->data[detInd]->data[0].data->length;
      REAL8 f0 = inputSFTs->data[detInd]->data[0].f0;
      if ( deltaF != inputSFTs->data[detInd]->data[0].deltaF ){
	LogPrintf ( LOG_CRITICAL, "%s: deltaF = %f disagrees with SFT deltaF = %f", __func__, deltaF, inputSFTs->data[detInd]->data[0].deltaF );
	XLAL_ERROR( XLAL_EINVAL );
      }

      if ((badBins->data[detInd] = XLALCreateUINT4Vector ( numSFTFreqs ) ) == NULL){
	LogPrintf ( LOG_CRITICAL, "%s: XLALCreateUINT4Vector() failed with errno=%d\n", __func__, xlalErrno );
	XLAL_ERROR( XLAL_EFUNC );
      }
      /* printf("%d\n",badBins->data[detInd]->length); */
      INT4 binCount = 0;

      if((fp = fopen(config.lineFiles->data[i_f], "r")) == NULL){
	LogPrintf ( LOG_CRITICAL, "%s: didn't find line file with name %s\n", __func__, config.lineFiles->data[i_f]);
	XLAL_ERROR( XLAL_EINVAL );
      }
      CHAR thisline[MAXLINELENGTH];
      UINT4 linesRead = 0;
      while (fgets(thisline, sizeof thisline, fp)) {
	if ( thisline[0] == '%' ) continue;
	REAL8 spacing;
	UINT4 combtype;
	REAL8 offset;
	UINT4 firstindex, lastindex;
	REAL8 leftwidth, rightwidth;
	UINT4 numRead = sscanf(thisline,PCC_LINEFILE_BODY, &spacing, &combtype, &offset, &firstindex, &lastindex, &leftwidth, &rightwidth);
	if(numRead!=7){
	  LogPrintf ( LOG_CRITICAL, "Failed to read data out of line file %s: needed 7, got %d\n", config.lineFiles->data[i_f], numRead );
	  XLAL_ERROR( XLAL_EINVAL );
	}
	/*	printf(PCC_LINEFILE_BODY "\n", spacing, combtype, offset, firstindex, lastindex, leftwidth, rightwidth); */
	switch (combtype){
	case 0: /* singlet line */
	  if ( firstindex != lastindex ){
	    LogPrintf ( LOG_CRITICAL, "Error in file %s: singlet line with firstindex=%d,lastindex=%d\n", config.lineFiles->data[i_f], firstindex, lastindex );
	    XLAL_ERROR( XLAL_EINVAL );
	  }
	  binCount = XLALFindBadBins( badBins->data[detInd], binCount, offset+firstindex*spacing-leftwidth, offset+lastindex*spacing+rightwidth, f0, deltaF, numSFTFreqs);
	  if (binCount < 0){
	    LogPrintf ( LOG_CRITICAL, "%s: XLALFindBadBins() failed with errno=%d\n", __func__, xlalErrno );
	    XLAL_ERROR( XLAL_EFUNC );
	  }
	  /* printf ("veto %f to %f\n", offset+firstindex*spacing-leftwidth, offset+lastindex*spacing+rightwidth); */
	  break;
	case 1: /* fixed-width comb */
	  if ( firstindex > lastindex ){
	    LogPrintf ( LOG_CRITICAL, "Error in file %s: comb with firstindex=%d,lastindex=%d\n", config.lineFiles->data[i_f], firstindex, lastindex );
	    XLAL_ERROR( XLAL_EINVAL );
	  }
	  for ( UINT4 index0 = firstindex ; index0 <= lastindex; index0++ ) {
	    binCount = XLALFindBadBins( badBins->data[detInd], binCount, offset+index0*spacing-leftwidth, offset+index0*spacing+rightwidth, f0, deltaF, numSFTFreqs);
	    if (binCount < 0){
	      LogPrintf ( LOG_CRITICAL, "%s: XLALFindBadBins() failed with errno=%d\n", __func__, xlalErrno );
	      XLAL_ERROR( XLAL_EFUNC );
	    }
	    /* printf ("veto %f to %f\n", offset+index0*spacing-leftwidth, offset+index0*spacing+rightwidth); */
	  }
	  break;
	case 2: /* scaling-width comb */
	  if ( firstindex > lastindex ){
	    LogPrintf ( LOG_CRITICAL, "Error in file %s: comb with firstindex=%d,lastindex=%d\n", config.lineFiles->data[i_f], firstindex, lastindex );
	    XLAL_ERROR( XLAL_EINVAL );
	  }
	  for ( UINT4 index0 = firstindex ; index0 <= lastindex; index0++ ) {
	    binCount = XLALFindBadBins( badBins->data[detInd], binCount, offset+index0*spacing-index0*leftwidth, offset+index0*spacing+index0*rightwidth, f0, deltaF, numSFTFreqs);
	    if (binCount < 0){
	      LogPrintf ( LOG_CRITICAL, "%s: XLALFindBadBins() failed with errno=%d\n", __func__, xlalErrno );
	      XLAL_ERROR( XLAL_EFUNC );
	    }
	    /* printf ("veto %f to %f\n", offset+index0*spacing-index0*leftwidth, offset+index0*spacing+index0*rightwidth); */
	  }
	  break;
	default:
	  LogPrintf ( LOG_CRITICAL, "Unrecognized combtype %d\n", combtype );
	  XLAL_ERROR( XLAL_EINVAL );
	}

	linesRead++;
      } /* while (fgets(thisline, sizeof thisline, fp)) */
      if (linesRead != numLines){
	LogPrintf ( LOG_CRITICAL, "Read %d lines out of %s but expected %d", linesRead, config.lineFiles->data[i_f], numLines );
	XLAL_ERROR( XLAL_EFUNC );
      }
      if ( binCount == 0 ){
	XLALDestroyUINT4Vector( badBins->data[detInd] );
	badBins->data[detInd] = NULL;
      } else {
	if ( (badBins->data[detInd]->data = XLALRealloc(badBins->data[detInd]->data, binCount*sizeof(UINT4)) ) == NULL){
	  XLAL_ERROR(XLAL_ENOMEM);
	}
	badBins->data[detInd]->length = binCount;
      }
    } /* for ( UINT4 i_f=0 ; i_f < config.lineFiles->length ; i_f++ ) */
  } /* if ( config->lineFiles ) */

  /* Get weighting factors for calculation of metric */
  /* note that the sigma-squared is now absorbed into the curly G
     because the AM coefficients are noise-weighted. */
  REAL8Vector *GammaAve = NULL;
  REAL8Vector *GammaCirc = NULL;
  if ( ( XLALCalculateCrossCorrGammas( &GammaAve, &GammaCirc, sftPairs, sftIndices, multiCoeffs)  != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCalculateCrossCorrGammas() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

#define PCC_GAMMA_HEADER "# The normalization Sinv_Tsft is %g #\n"
#define PCC_GAMMA_BODY "%.10g\n"
  if (XLALUserVarWasSet(&uvar.gammaAveOutputFilename)) { /* Write the aa+bb weight for each pair to a file, if a name was provided */
    if((fp = fopen(uvar.gammaAveOutputFilename, "w")) == NULL) {
      LogPrintf ( LOG_CRITICAL, "Can't write in Gamma_ave list \n");
      XLAL_ERROR( XLAL_EFUNC );
    }
    fprintf(fp,PCC_GAMMA_HEADER, multiWeights->Sinv_Tsft); /*output the normalization factor to the header*/
    for(j = 0; j < sftPairs->length; j++){
      fprintf(fp,PCC_GAMMA_BODY, GammaAve->data[j]);
    }
    fclose(fp);
  }
  if (XLALUserVarWasSet(&uvar.gammaCircOutputFilename)) { /* Write the ab-ba weight for each pair to a file, if a name was provided */
    if((fp = fopen(uvar.gammaCircOutputFilename, "w")) == NULL) {
      LogPrintf ( LOG_CRITICAL, "Can't write in Gamma_circ list \n");
      XLAL_ERROR( XLAL_EFUNC );
    }
    fprintf(fp,PCC_GAMMA_HEADER, multiWeights->Sinv_Tsft); /*output the normalization factor to the header*/
    for(j = 0; j < sftPairs->length; j++){
      fprintf(fp,PCC_GAMMA_BODY, GammaCirc->data[j]);
    }
    fclose(fp);
  }

  /*initialize binary parameters structure*/
  XLAL_INIT_MEM(minBinaryTemplate);
  XLAL_INIT_MEM(maxBinaryTemplate);
  XLAL_INIT_MEM(thisBinaryTemplate);
  XLAL_INIT_MEM(binaryTemplateSpacings);
  /*fill in minbinaryOrbitParams*/
  XLALGPSSetREAL8( &minBinaryTemplate.tp, uvar.orbitTimeAsc);
  minBinaryTemplate.argp = 0.0;
  minBinaryTemplate.asini = uvar.orbitAsiniSec;
  minBinaryTemplate.ecc = 0.0;
  minBinaryTemplate.period = uvar.orbitPSec;
  minBinaryTemplate.fkdot[0] = uvar.fStart;
  /*fill in maxBinaryParams*/
  XLALGPSSetREAL8( &maxBinaryTemplate.tp, uvar.orbitTimeAsc + uvar.orbitTimeAscBand);
  maxBinaryTemplate.argp = 0.0;
  maxBinaryTemplate.asini = uvar.orbitAsiniSec + uvar.orbitAsiniSecBand;
  maxBinaryTemplate.ecc = 0.0;
  maxBinaryTemplate.period = uvar.orbitPSec + uvar.orbitPSecBand;
  maxBinaryTemplate.fkdot[0] = uvar.fStart + uvar.fBand;
  /*fill in thisBinaryTemplate*/
  XLALGPSSetREAL8( &thisBinaryTemplate.tp, uvar.orbitTimeAsc + 0.5 * uvar.orbitTimeAscBand);
  thisBinaryTemplate.argp = 0.0;
  thisBinaryTemplate.asini = 0.5*(minBinaryTemplate.asini + maxBinaryTemplate.asini);
  thisBinaryTemplate.ecc = 0.0;
  thisBinaryTemplate.period =0.5*(minBinaryTemplate.period + maxBinaryTemplate.period);
  thisBinaryTemplate.fkdot[0]=0.5*(minBinaryTemplate.fkdot[0] + maxBinaryTemplate.fkdot[0]);

  REAL8 old_diagff = 0; /*diagonal metric components*/
  REAL8 old_diagaa = 0;
  REAL8 old_diagTT = 0;
  REAL8 old_diagpp = 0;
  REAL8 ccStat = 0;
  REAL8 evSquared = 0;
  REAL8 estSens = 0; /*estimated sensitivity(4.13)*/
  REAL8 weightedMuTAve = 0;

  /*Get metric diagonal components, also estimate sensitivity i.e. E[rho]/(h0)^2 (4.13)*/
  if ( (XLALCalculateLMXBCrossCorrDiagMetric(&estSens, &old_diagff, &old_diagaa, &old_diagTT, &old_diagpp, &weightedMuTAve, thisBinaryTemplate, GammaAve, sftPairs, sftIndices, inputSFTs, multiWeights /*, kappaValues*/)  != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCalculateLMXBCrossCorrDiagMetric() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* initialize the doppler scan struct which stores the current template information */
  XLALGPSSetREAL8(&dopplerpos.refTime, config.refTime);
  dopplerpos.Alpha = uvar.alphaRad;
  dopplerpos.Delta = uvar.deltaRad;
  dopplerpos.fkdot[0] = minBinaryTemplate.fkdot[0];
  /* set all spindowns to zero */
  for (k=1; k < PULSAR_MAX_SPINS; k++)
    dopplerpos.fkdot[k] = 0.0;
  dopplerpos.asini = minBinaryTemplate.asini;
  dopplerpos.period = minBinaryTemplate.period;
  dopplerpos.tp = minBinaryTemplate.tp;
  dopplerpos.ecc = minBinaryTemplate.ecc;
  dopplerpos.argp = minBinaryTemplate.argp;

  /* now set the initial values of binary parameters */
  /*  thisBinaryTemplate.asini = uvar.orbitAsiniSec;
  thisBinaryTemplate.period = uvar.orbitPSec;
  XLALGPSSetREAL8( &thisBinaryTemplate.tp, uvar.orbitTimeAsc);
  thisBinaryTemplate.ecc = 0.0;
  thisBinaryTemplate.argp = 0.0;*/
  /* copy to dopplerpos */

  /* Calculate SSB times (can do this once since search is currently only for one sky position, and binary doppler shift is added later) */
  MultiSSBtimes *multiSSBTimes = NULL;
  if ((multiSSBTimes = XLALGetMultiSSBtimes ( multiStates, skyPos, dopplerpos.refTime, SSBPREC_RELATIVISTICOPT )) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALGetMultiSSBtimes() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Allocate structure for binary doppler-shifting information */
  if ((multiBinaryTimes = XLALDuplicateMultiSSBtimes ( multiSSBTimes )) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALDuplicateMultiSSBtimes() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  UINT8 numSFTs = sftIndices->length;
  if ((shiftedFreqs = XLALCreateREAL8Vector ( numSFTs ) ) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALCreateREAL8Vector() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }
  if ((lowestBins = XLALCreateUINT4Vector ( numSFTs ) ) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALCreateUINT4Vector() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  if ((expSignalPhases = XLALCreateCOMPLEX8Vector ( numSFTs ) ) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALCreateREAL8Vector() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }
  if ((sincList = XLALCreateREAL8VectorSequence ( numSFTs, uvar.numBins ) ) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALCreateREAL8VectorSequence() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

   /* "New" general metric computation */
  /* For now hard-code circular parameter space */

  const DopplerCoordinateSystem coordSys = {
    .dim = 4,
    .coordIDs = { DOPPLERCOORD_FREQ,
		  DOPPLERCOORD_ASINI,
		  DOPPLERCOORD_TASC,
		  DOPPLERCOORD_PORB, },
  };

  REAL8VectorSequence *phaseDerivs = NULL;
  if ( ( XLALCalculateCrossCorrPhaseDerivatives ( &phaseDerivs, &thisBinaryTemplate, config.edat, sftIndices, multiSSBTimes, &coordSys )  != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCalculateCrossCorrPhaseDerivatives() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* fill in metric and parameter offsets */
  gsl_matrix *g_ij = NULL;
  gsl_vector *eps_i = NULL;
  REAL8 sumGammaSq = 0;
  if ( ( XLALCalculateCrossCorrPhaseMetric ( &g_ij, &eps_i, &sumGammaSq, phaseDerivs, sftPairs, GammaAve, GammaCirc, &coordSys ) != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALCalculateCrossCorrPhaseMetric() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }
  XLALDestroyREAL8VectorSequence ( phaseDerivs );
  XLALDestroyREAL8Vector ( GammaCirc );

  /*  if ((fp = fopen("gsldata.dat","w"))==NULL){
    LogPrintf ( LOG_CRITICAL, "Can't write in gsl matrix file");
    XLAL_ERROR( XLAL_EFUNC );
  }

  XLALfprintfGSLvector(fp, "%g", eps_i);
  XLALfprintfGSLmatrix(fp, "%g", g_ij);*/

  REAL8 diagff = gsl_matrix_get(g_ij, 0, 0);
  REAL8 diagaa = gsl_matrix_get(g_ij, 1, 1);
  REAL8 diagTT = gsl_matrix_get(g_ij, 2, 2);
  REAL8 diagpp = gsl_matrix_get(g_ij, 3, 3);

  dimName = XLALCreateCHARVector(coordSys.dim);
  dimName->data[0] = 'f';
  dimName->data[1] = 'a';
  dimName->data[2] = 'T';
  dimName->data[3] = 'p';

  /* spacing in frequency from diagff */ /* set spacings in new dopplerparams struct */
  if (XLALUserVarWasSet(&uvar.spacingF)) /* If spacing was given by CMD line, use it, else calculate spacing by mismatch*/
    binaryTemplateSpacings.fkdot[0] = uvar.spacingF;
  else
    binaryTemplateSpacings.fkdot[0] = sqrt(uvar.mismatchF / diagff);

  if (XLALUserVarWasSet(&uvar.spacingA))
    binaryTemplateSpacings.asini = uvar.spacingA;
  else
    binaryTemplateSpacings.asini = sqrt(uvar.mismatchA / diagaa);
  /* this is annoying: tp is a GPS time while we want a difference
     in time which should be just REAL8 */
  if (XLALUserVarWasSet(&uvar.spacingT))
    XLALGPSSetREAL8( &binaryTemplateSpacings.tp, uvar.spacingT);
  else
    XLALGPSSetREAL8( &binaryTemplateSpacings.tp, sqrt(uvar.mismatchT / diagTT));

  if (XLALUserVarWasSet(&uvar.spacingP))
    binaryTemplateSpacings.period = uvar.spacingP;
  else
    binaryTemplateSpacings.period = sqrt(uvar.mismatchP / diagpp);

  /* metric elements for eccentric case not considered? */

  UINT8 fCount = 0, aCount = 0, tCount = 0 , pCount = 0;
  const UINT8 fSpacingNum = floor( uvar.fBand / binaryTemplateSpacings.fkdot[0]);
  const UINT8 aSpacingNum = floor( uvar.orbitAsiniSecBand / binaryTemplateSpacings.asini);
  const UINT8 tSpacingNum = floor( uvar.orbitTimeAscBand / XLALGPSGetREAL8(&binaryTemplateSpacings.tp));
  const UINT8 pSpacingNum = floor( uvar.orbitPSecBand / binaryTemplateSpacings.period);

  /*reset minbinaryOrbitParams to shift the first point a factor so as to make the center of all seaching points centers at the center of searching band*/
  minBinaryTemplate.fkdot[0] = uvar.fStart + 0.5 * (uvar.fBand - fSpacingNum * binaryTemplateSpacings.fkdot[0]);
  minBinaryTemplate.asini = uvar.orbitAsiniSec + 0.5 * (uvar.orbitAsiniSecBand - aSpacingNum * binaryTemplateSpacings.asini);
  XLALGPSSetREAL8( &minBinaryTemplate.tp, uvar.orbitTimeAsc + 0.5 * (uvar.orbitTimeAscBand - tSpacingNum * XLALGPSGetREAL8(&binaryTemplateSpacings.tp)));
  minBinaryTemplate.period = uvar.orbitPSec + 0.5 * (uvar.orbitPSecBand - pSpacingNum * binaryTemplateSpacings.period);
  /*also reset dopplerpos orbital parameters and frequency*/
  dopplerpos.fkdot[0] = minBinaryTemplate.fkdot[0];
  dopplerpos.asini = minBinaryTemplate.asini;
  dopplerpos.tp = minBinaryTemplate.tp;
  dopplerpos.period = minBinaryTemplate.period;

  /* args should be : spacings, min and max doppler params */
  BOOLEAN firstPoint = TRUE; /* a boolean to help to search at the beginning point in parameter space, after the search it is set to be FALSE to end the loop*/
  if ( (XLALAddMultiBinaryTimes( &multiBinaryTimes, multiSSBTimes, &dopplerpos )  != XLAL_SUCCESS ) ) {
    LogPrintf ( LOG_CRITICAL, "%s: XLALAddMultiBinaryTimes() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  } /*Need to apply additional doppler shifting before the loop, or the first point in parameter space will be lost and return a wrong SNR when fBand!=0*/

  while ( GetNextCrossCorrTemplate(&dopplerShiftFlag, &firstPoint, &dopplerpos, &binaryTemplateSpacings, &minBinaryTemplate, &maxBinaryTemplate, &fCount, &aCount, &tCount, &pCount, fSpacingNum, aSpacingNum, tSpacingNum, pSpacingNum) == 0)
    {
      /* do useful stuff here*/

      /* Apply additional Doppler shifting using current binary orbital parameters */
      /* Might want to be clever about checking whether we've changed the orbital parameters or only the frequency */
      if (dopplerShiftFlag == TRUE)
	{
	  if ( (XLALAddMultiBinaryTimes( &multiBinaryTimes, multiSSBTimes, &dopplerpos )  != XLAL_SUCCESS ) ) {
	    LogPrintf ( LOG_CRITICAL, "%s: XLALAddMultiBinaryTimes() failed with errno=%d\n", __func__, xlalErrno );
	    XLAL_ERROR( XLAL_EFUNC );
	  }
	}

      if ( (XLALGetDopplerShiftedFrequencyInfo( shiftedFreqs, lowestBins, expSignalPhases, sincList, uvar.numBins, &dopplerpos, sftIndices, inputSFTs, multiBinaryTimes, badBins, Tsft )  != XLAL_SUCCESS ) ) {
	LogPrintf ( LOG_CRITICAL, "%s: XLALGetDopplerShiftedFrequencyInfo() failed with errno=%d\n", __func__, xlalErrno );
	XLAL_ERROR( XLAL_EFUNC );
      }

      if ( (XLALCalculatePulsarCrossCorrStatistic( &ccStat, &evSquared, GammaAve, expSignalPhases, lowestBins, sincList, sftPairs, sftIndices, inputSFTs, multiWeights, uvar.numBins)  != XLAL_SUCCESS ) ) {
	LogPrintf ( LOG_CRITICAL, "%s: XLALCalculatePulsarCrossCorrStatistic() failed with errno=%d\n", __func__, xlalErrno );
	XLAL_ERROR( XLAL_EFUNC );
      }

      /* fill candidate struct and insert into toplist if necessary */
      thisCandidate.freq = dopplerpos.fkdot[0];
      thisCandidate.tp = XLALGPSGetREAL8( &dopplerpos.tp );
      thisCandidate.argp = dopplerpos.argp;
      thisCandidate.asini = dopplerpos.asini;
      thisCandidate.ecc = dopplerpos.ecc;
      thisCandidate.period = dopplerpos.period;
      thisCandidate.rho = ccStat;
      thisCandidate.evSquared = evSquared;
      thisCandidate.estSens = estSens;

      insert_into_crossCorrBinary_toplist(ccToplist, thisCandidate);

    } /* end while loop over templates */

  /* write candidates to file */
  sort_crossCorrBinary_toplist( ccToplist );
  /* add error checking */

  final_write_crossCorrBinary_toplist_to_file( ccToplist, uvar.toplistFilename, &checksum);

  REAL8 h0Sens = sqrt((10 / sqrt(estSens))); /*for a SNR=10 signal, the h0 we can detect*/

  XLALGPSTimeNow (&computingEndGPSTime); /*record the rough end time*/
  UINT4 computingTime = computingEndGPSTime.gpsSeconds - computingStartGPSTime.gpsSeconds;
  /* make a meta-data file*/
  if(XLALUserVarWasSet(&uvar.logFilename)){
    CHAR *CMDInputStr = XLALUserVarGetLog ( UVAR_LOGFMT_CFGFILE );
    if ((fp = fopen(uvar.logFilename,"w"))==NULL){
    LogPrintf ( LOG_CRITICAL, "Can't write in logfile");
    XLAL_ERROR( XLAL_EFUNC );
    }
    fprintf(fp, "[UserInput]\n\n");
    fprintf(fp, "%s\n", CMDInputStr);
    fprintf(fp, "[CalculatedValues]\n\n");
    REAL8 g_lm, eps_n;
    for (UINT4 l = 0; l < coordSys.dim; l++){
      for (UINT4 m = 0; m < coordSys.dim; m++){
	if (l == m){
	  g_lm = gsl_matrix_get(g_ij, l, m);
	  fprintf(fp, "g_%c%c = %.9"LAL_REAL8_FORMAT"\n", dimName->data[l], dimName->data[m], g_lm);
	}
      }
    }
    for (UINT4 l = 0; l < coordSys.dim; l++){
      for (UINT4 m = 0; m < coordSys.dim; m++){
	if (l < m){
	  g_lm = gsl_matrix_get(g_ij, l, m);
	  fprintf(fp, "g_%c%c = %.9"LAL_REAL8_FORMAT"\n", dimName->data[l], dimName->data[m], g_lm);
	}
      }
    }
    for (UINT4 n = 0; n < coordSys.dim; n++){
      eps_n = gsl_vector_get(eps_i, n);
      fprintf(fp, "eps_%c = %.9"LAL_REAL8_FORMAT"\n", dimName->data[n], eps_n);
    }
    /* old metric for debugging */
    fprintf(fp, "old_diagff = %.9g\n", old_diagff);
    fprintf(fp, "old_diagaa = %.9g\n", old_diagaa);
    fprintf(fp, "old_diagTT = %.9g\n", old_diagTT);
    fprintf(fp, "old_diagpp = %.9g\n", old_diagpp);
    fprintf(fp, "FSpacing = %.9g\n", binaryTemplateSpacings.fkdot[0]);
    fprintf(fp, "ASpacing = %.9g\n", binaryTemplateSpacings.asini);
    fprintf(fp, "TSpacing = %.9g\n", XLALGPSGetREAL8(&binaryTemplateSpacings.tp));
    fprintf(fp, "PSpacing = %.9g\n", binaryTemplateSpacings.period);
    fprintf(fp, "TemplatenumF = %" LAL_UINT8_FORMAT "\n", (fSpacingNum + 1));
    fprintf(fp, "TemplatenumA = %" LAL_UINT8_FORMAT "\n", (aSpacingNum + 1));
    fprintf(fp, "TemplatenumT = %" LAL_UINT8_FORMAT "\n", (tSpacingNum + 1));
    fprintf(fp, "TemplatenumP = %" LAL_UINT8_FORMAT "\n", (pSpacingNum + 1));
    fprintf(fp, "TemplatenumTotal = %" LAL_UINT8_FORMAT "\n",(fSpacingNum + 1) * (aSpacingNum + 1) * (tSpacingNum + 1) * (pSpacingNum + 1));
    fprintf(fp, "Sens = %.9g\n", estSens); /*(E[rho]/h0^2)^2*/
    fprintf(fp, "h0_min_SNR10 = %.9g\n", h0Sens); /*for rho = 10 in our pipeline*/
    fprintf(fp, "weightedMutAve = %.9f\n", weightedMuTAve); /*weighted average of mean SFT from each pair of SFT*/
    fprintf(fp, "jobStartTime = %" LAL_INT4_FORMAT "\n", computingStartGPSTime.gpsSeconds); /*job start time in GPS-time*/
    fprintf(fp, "jobEndTime = %" LAL_INT4_FORMAT "\n", computingEndGPSTime.gpsSeconds); /*job end time in GPS-time*/
    fprintf(fp, "computingTime = %" LAL_UINT4_FORMAT "\n", computingTime); /*total time in sec*/
    fprintf(fp, "SFTnum = %" LAL_UINT4_FORMAT "\n", sftIndices->length); /*total number of SFT*/
    fprintf(fp, "pairnum = %" LAL_UINT4_FORMAT "\n", sftPairs->length); /*total number of pair of SFT*/
    fprintf(fp, "Tsft = %.6g\n", Tsft); /*SFT duration*/
    fprintf(fp, "\n[Version]\n\n");
    fprintf(fp, "%s",  VCSInfoString);
    fclose(fp);
    XLALFree(CMDInputStr);
  }

  /* FIXME: Need to destroy badBins */

  XLALFree(VCSInfoString);
  XLALDestroyCOMPLEX8Vector ( expSignalPhases );
  XLALDestroyUINT4Vector ( lowestBins );
  XLALDestroyREAL8Vector ( shiftedFreqs );
  XLALDestroyREAL8VectorSequence ( sincList );
  XLALDestroyMultiSSBtimes ( multiBinaryTimes );
  XLALDestroyMultiSSBtimes ( multiSSBTimes );
  XLALDestroyREAL8Vector ( GammaAve );
  XLALDestroySFTPairIndexList( sftPairs );
  XLALDestroySFTIndexList( sftIndices );
  XLALDestroyMultiAMCoeffs ( multiCoeffs );
  XLALDestroyMultiDetectorStateSeries ( multiStates );
  XLALDestroyMultiTimestamps ( multiTimes );
  XLALDestroyMultiNoiseWeights ( multiWeights );
  XLALDestroyMultiPSDVector ( multiPSDs );
  XLALDestroyMultiSFTVector ( inputSFTs );
  XLALDestroyCHARVector ( dimName );
  /* de-allocate memory for configuration variables */
  XLALDestroyConfigVars ( &config );

  /* de-allocate memory for user input variables */
  XLALDestroyUserVars();

  /* free toplist memory */
  free_crossCorr_toplist(&ccToplist);

  /* check memory leaks if we forgot to de-allocate anything */
  LALCheckMemoryLeaks();

  LogPrintf (LOG_CRITICAL, "End time\n");/*for debug convenience to record calculating time*/

  return 0;


} /* main */


/* initialize and register user variables */
int XLALInitUserVars (UserInput_t *uvar)
{

  /* initialize with some defaults */
  uvar->maxLag = 0.0;
  uvar->inclAutoCorr = FALSE;
  uvar->fStart = 100.0;
  uvar->fBand = 0.1;
  /* uvar->fdotStart = 0.0; */
  /* uvar->fdotBand = 0.0; */
  uvar->alphaRad = 0.0;
  uvar->deltaRad = 0.0;
  uvar->refTime = 0.0;
  uvar->rngMedBlock = 50;
  uvar->numBins = 1;

  /* zero binary orbital parameters means not a binary */
  uvar->orbitAsiniSec = 0.0;
  uvar->orbitAsiniSecBand = 0.0;
  uvar->orbitPSec = 0.0;
  uvar->orbitPSecBand = 0.0;
  uvar->orbitTimeAsc = 0;
  uvar->orbitTimeAscBand = 0;

  /*default mismatch values */
  /* set to 0.1 by default -- for no real reason */
  /* make 0.1 a macro? */
  uvar->mismatchF = 0.1;
  uvar->mismatchA = 0.1;
  uvar->mismatchT = 0.1;
  uvar->mismatchP = 0.1;

  uvar->ephemEarth = XLALStringDuplicate("earth00-19-DE405.dat.gz");
  uvar->ephemSun = XLALStringDuplicate("sun00-19-DE405.dat.gz");

  uvar->sftLocation = XLALCalloc(1, MAXFILENAMELENGTH+1);

  /* initialize number of candidates in toplist -- default is just to return the single best candidate */
  uvar->numCand = 1;
  uvar->toplistFilename = XLALStringDuplicate("toplist_crosscorr.dat");

  /* register  user-variables */
  XLALRegisterUvarMember( startTime,       INT4, 0,  REQUIRED, "Desired start time of analysis in GPS seconds");
  XLALRegisterUvarMember( endTime,         INT4, 0,  REQUIRED, "Desired end time of analysis in GPS seconds");
  XLALRegisterUvarMember( maxLag,          REAL8, 0,  OPTIONAL, "Maximum lag time in seconds between SFTs in correlation");
  XLALRegisterUvarMember( inclAutoCorr,    BOOLEAN, 0,  OPTIONAL, "Include auto-correlation terms (an SFT with itself)");
  XLALRegisterUvarMember( fStart,          REAL8, 0,  OPTIONAL, "Start frequency in Hz");
  XLALRegisterUvarMember( fBand,           REAL8, 0,  OPTIONAL, "Frequency band to search over in Hz ");
  /* XLALRegisterUvarMember( fdotStart,     REAL8, 0,  OPTIONAL, "Start value of spindown in Hz/s"); */
  /* XLALRegisterUvarMember( fdotBand,      REAL8, 0,  OPTIONAL, "Band for spindown values in Hz/s"); */
  XLALRegisterUvarMember( alphaRad,        REAL8, 0,  OPTIONAL, "Right ascension for directed search (radians)");
  XLALRegisterUvarMember( deltaRad,        REAL8, 0,  OPTIONAL, "Declination for directed search (radians)");
  XLALRegisterUvarMember( refTime,         REAL8, 0,  OPTIONAL, "SSB reference time for pulsar-parameters [Default: midPoint]");
  XLALRegisterUvarMember( orbitAsiniSec,   REAL8, 0,  OPTIONAL, "Start of search band for projected semimajor axis (seconds) [0 means not a binary]");
  XLALRegisterUvarMember( orbitAsiniSecBand, REAL8, 0,  OPTIONAL, "Width of search band for projected semimajor axis (seconds)");
  XLALRegisterUvarMember( orbitPSec,       REAL8, 0,  OPTIONAL, "Binary orbital period (seconds) [0 means not a binary]");
  XLALRegisterUvarMember( orbitPSecBand,       REAL8, 0,  OPTIONAL, "Band for binary orbital period (seconds) ");
  XLALRegisterUvarMember( orbitTimeAsc,    REAL8, 0,  OPTIONAL, "Start of orbital time-of-ascension band in GPS seconds");
  XLALRegisterUvarMember( orbitTimeAscBand, REAL8, 0,  OPTIONAL, "Width of orbital time-of-ascension band (seconds)");
  XLALRegisterUvarMember( ephemEarth,      STRING, 0,  OPTIONAL, "Earth ephemeris file to use");
  XLALRegisterUvarMember( ephemSun,        STRING, 0,  OPTIONAL, "Sun ephemeris file to use");
  XLALRegisterUvarMember( sftLocation,     STRING, 0,  REQUIRED, "Filename pattern for locating SFT data");
  XLALRegisterUvarMember( rngMedBlock,     INT4, 0,  OPTIONAL, "Running median block size for PSD estimation");
  XLALRegisterUvarMember( numBins,         INT4, 0,  OPTIONAL, "Number of frequency bins to include in calculation");
  XLALRegisterUvarMember( mismatchF,       REAL8, 0,  OPTIONAL, "Desired mismatch for frequency spacing");
  XLALRegisterUvarMember( mismatchA,       REAL8, 0,  OPTIONAL, "Desired mismatch for asini spacing");
  XLALRegisterUvarMember( mismatchT,       REAL8, 0,  OPTIONAL, "Desired mismatch for periapse passage time spacing");
  XLALRegisterUvarMember( mismatchP,       REAL8, 0,  OPTIONAL, "Desired mismatch for period spacing");
  XLALRegisterUvarMember( spacingF,       REAL8, 0,  OPTIONAL, "Desired frequency spacing");
  XLALRegisterUvarMember( spacingA,       REAL8, 0,  OPTIONAL, "Desired asini spacing");
  XLALRegisterUvarMember( spacingT,       REAL8, 0,  OPTIONAL, "Desired periapse passage time spacing");
  XLALRegisterUvarMember( spacingP,       REAL8, 0,  OPTIONAL, "Desired period spacing");
  XLALRegisterUvarMember( numCand,         INT4, 0,  OPTIONAL, "Number of candidates to keep in toplist");
  XLALRegisterUvarMember( linesToCleanFilenames, STRING, 0,  OPTIONAL, "Comma-separated list of line files");
  XLALRegisterUvarMember( pairListInputFilename, STRING, 0,  OPTIONAL, "Name of file from which to read list of SFT pairs");
  XLALRegisterUvarMember( pairListOutputFilename, STRING, 0,  OPTIONAL, "Name of file to which to write list of SFT pairs");
  XLALRegisterUvarMember( sftListOutputFilename, STRING, 0,  OPTIONAL, "Name of file to which to write list of SFTs (for sanity checks)");
  XLALRegisterUvarMember( sftListInputFilename, STRING, 0,  OPTIONAL, "Name of file to which to read in list of SFTs (for sanity checks)");
  XLALRegisterUvarMember( gammaAveOutputFilename, STRING, 0,  OPTIONAL, "Name of file to which to write aa+bb weights (for e.g., false alarm estimation)");
  XLALRegisterUvarMember( gammaCircOutputFilename, STRING, 0,  OPTIONAL, "Name of file to which to write ab-ba weights (for e.g., systematic error)");
  XLALRegisterUvarMember( toplistFilename, STRING, 0,  OPTIONAL, "Output filename containing candidates in toplist");
  XLALRegisterUvarMember( logFilename, STRING, 0,  OPTIONAL, "Output a meta-data file for the search");
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

  /* set sft catalog constraints */
  constraints.detector = NULL;
  constraints.timestamps = NULL;
  constraints.minStartTime = &startTime;
  constraints.maxStartTime = &endTime;
  XLALGPSSet( constraints.minStartTime, uvar->startTime, 0 );
  XLALGPSSet( constraints.maxStartTime, uvar->endTime, 0 );

  if (XLALUserVarWasSet(&(uvar->refTime)))
    config->refTime = uvar->refTime;
  else
    config->refTime = uvar->startTime + 0.5 * ( (REAL8) (uvar->endTime - uvar->startTime) );


  /* This check doesn't seem to work, since XLALGPSSet doesn't set its
     first argument.

     if ( (constraints.minStartTime == NULL)&& (constraints.maxStartTime == NULL) ) {
     LogPrintf ( LOG_CRITICAL, "%s: XLALGPSSet() failed with errno=%d\n", __func__, xlalErrno );
     XLAL_ERROR( XLAL_EFUNC );
     }

  */

  /* get catalog of SFTs */
  if ((config->catalog = XLALSFTdataFind (uvar->sftLocation, &constraints)) == NULL){
    LogPrintf ( LOG_CRITICAL, "%s: XLALSFTdataFind() failed with errno=%d\n", __func__, xlalErrno );
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* initialize ephemeris data */
  XLAL_CHECK ( (config->edat = XLALInitBarycenter ( uvar->ephemEarth, uvar->ephemSun )) != NULL, XLAL_EFUNC );

  /* parse comma-separated list of lines files */
  config->lineFiles = NULL;

  if (XLALUserVarWasSet(&uvar->linesToCleanFilenames))
    {
      CHAR *tmpstring = NULL;
      XLAL_CHECK ( (tmpstring = XLALStringDuplicate( uvar->linesToCleanFilenames )) != NULL, XLAL_EFUNC );

      UINT4 numfiles = pcc_count_csv( tmpstring );

      LALFree( tmpstring );
      XLAL_CHECK ( (tmpstring = XLALStringDuplicate( uvar->linesToCleanFilenames )) != NULL, XLAL_EFUNC );

      for ( UINT4 i = 0 ; i < numfiles ; i++ ){
	CHAR *pcc_tmpfile = NULL;
	XLAL_CHECK ( (pcc_tmpfile = XLALStringToken( &tmpstring, ",", 0))!= NULL, XLAL_EFUNC );
	XLAL_CHECK ( (config->lineFiles = XLALAppendString2Vector( config->lineFiles, pcc_tmpfile ))!= NULL, XLAL_EFUNC );

      }
    }

  return XLAL_SUCCESS;

}
/* XLALInitializeConfigVars() */

/* deallocate memory associated with config variables */
int XLALDestroyConfigVars (ConfigVariables *config)
{
  XLALDestroySFTCatalog(config->catalog);
  XLALDestroyEphemerisData(config->edat);
  XLALDestroyStringVector(config->lineFiles);
  return XLAL_SUCCESS;
}
/* XLALDestroyConfigVars() */

/* getting the next template */
/** FIXME: spacings and min, max values of binary parameters are not used yet */


int GetNextCrossCorrTemplate(BOOLEAN *binaryParamsFlag, BOOLEAN *firstPoint, PulsarDopplerParams *dopplerpos, PulsarDopplerParams *binaryTemplateSpacings, PulsarDopplerParams *minBinaryTemplate, PulsarDopplerParams *maxBinaryTemplate, UINT8 *fCount, UINT8 *aCount, UINT8 *tCount, UINT8 *pCount, UINT8 fSpacingNum, UINT8 aSpacingNum, UINT8 tSpacingNum, UINT8 pSpacingNum)
{

  /* basic sanity checks */
  if (binaryTemplateSpacings == NULL)
    return -1;

  if (minBinaryTemplate == NULL)
    return -1;

  if (maxBinaryTemplate == NULL)
    return -1;

  /* check spacings not negative */

  if ( *fCount < fSpacingNum)    /*loop over f at first*/
    {
      dopplerpos->fkdot[0] = minBinaryTemplate->fkdot[0] + (*fCount + 1) * binaryTemplateSpacings->fkdot[0];
      *binaryParamsFlag = FALSE;
      *fCount += 1;
      return 0;
    }
  else
    {
      if ( *aCount < aSpacingNum )  /*after looping all f, initialize f and loop over a_p*/
	{
	  dopplerpos->asini = minBinaryTemplate->asini + (*aCount + 1) * binaryTemplateSpacings->asini;
	  dopplerpos->fkdot[0] = minBinaryTemplate->fkdot[0];
	  *fCount = 0;
	  *binaryParamsFlag = TRUE;
	  *aCount += 1;
	  return 0;
	}
      else
	{
	  if ( *pCount < pSpacingNum )  /*after looping the plane of f and a_p, initialize f, a_p and loop over P*/
	    {
	      dopplerpos->period = minBinaryTemplate->period + (*pCount + 1) * binaryTemplateSpacings->period;
	      dopplerpos->fkdot[0] =  minBinaryTemplate->fkdot[0];
	      *fCount = 0;
	      dopplerpos->asini = minBinaryTemplate->asini;
	      *aCount = 0;
	      *binaryParamsFlag = TRUE;
	      *pCount += 1;
	      return 0;
	    }

	  else
	    {
	      if ( *tCount < tSpacingNum ) /*after looping f, a_p and P, initialize f, a_p and P, then loop over T*/
		{
		  REAL8 nextGPSTime = XLALGPSGetREAL8(&minBinaryTemplate->tp) + (*tCount + 1) *  XLALGPSGetREAL8(&binaryTemplateSpacings->tp);
		  XLALGPSSetREAL8( &dopplerpos->tp, nextGPSTime);
		  dopplerpos->fkdot[0] =  minBinaryTemplate->fkdot[0];
		  *fCount = 0;
		  dopplerpos->asini = minBinaryTemplate->asini;
		  *aCount = 0;
		  dopplerpos->period = minBinaryTemplate->period;
		  *pCount = 0;
		  *binaryParamsFlag = TRUE;
		  *tCount += 1;
		  return 0;
		}

	      else
		{
		  if (*firstPoint == TRUE) /*go back to search at the beginning point in parameter space*/
		    {
		      dopplerpos->fkdot[0] = minBinaryTemplate->fkdot[0];
		      dopplerpos->asini = minBinaryTemplate->asini;
		      dopplerpos->period = minBinaryTemplate->period;
		      dopplerpos->tp = minBinaryTemplate->tp;
		      *firstPoint = FALSE;
		      *binaryParamsFlag = TRUE;
		      return 0;
		    }
		  else
		    return 1;
		}
	    }
	}
    }
}

/* Copied from ppe_utils.c by Matt Pitkin */

/**
 * \brief Counts the number of comma separated values in a string
 *
 * This function counts the number of comma separated values in a given input string.
 *
 * \param csvline [in] Any string
 *
 * \return The number of comma separated value in the input string
 */
UINT4 pcc_count_csv( CHAR *csvline ){
  CHAR *inputstr = NULL;
  UINT4 count = 0;

  inputstr = XLALStringDuplicate( csvline );

  /* count number of commas */
  while(1){
    if( XLALStringToken(&inputstr, ",", 0) == NULL ){ XLAL_ERROR( XLAL_EFUNC, "Error... problem counting number of commas!" ); }

    if ( inputstr == NULL ) { break; }

    count++;
  }

  return count+1;
}

/** Convert a range of contaminated frequencies into a set of bins to zero out */
/* Returns the running number of zeroed bins */
INT4 XLALFindBadBins
  (
   UINT4Vector *badBinData, /* Modified: running list of bad bins */
   INT4 binCount, /* Input: number of bins already in list */
   REAL8 flo, /* Input: Lower end of contaminated frequency range */
   REAL8 fhi, /* Input: Upper end of contaminated frequency range */
   REAL8 f0, /* Input: Base frequency of frequency series */
   REAL8 deltaF, /* Input: Frequency step of frequency series */
   UINT4 length /* Input: Size of frequency series */
   ) 
{

  /* printf ("veto %f to %f\n", flo, fhi); */

  /* printf ("Series has %d bins starting at %f with %f spacing\n", length, f0, deltaF); */
  /* printf ("Last bin is %f\n", f0 + length * deltaF); */

  INT4 newBinCount = binCount;
  INT4 firstBadBin = (INT4) floor ( ( flo - f0 ) / deltaF ); /*use floor to get the lowest contaminated bin*/
  /* printf ("firstBadBin = %d\n",firstBadBin); */
  if ( firstBadBin < 0 ) firstBadBin = 0;
  INT4 lastBadBin = (INT4) ceil ( ( fhi - f0 ) / deltaF ); /*use ceil to get the highest contaminated bin make sure to extend the boundary*/
  /* printf ("lastBadBin = %d\n",lastBadBin); */
  if ( lastBadBin >= (INT4) length ) lastBadBin = (INT4) (length-1);

  /* printf ("%d %d\n", firstBadBin, lastBadBin); */

  for ( INT4 badBin = firstBadBin ; badBin <= lastBadBin ; badBin++ ){
    /* printf ("%d %d %d\n", badBin, firstBadBin, lastBadBin); */
    if (newBinCount > (INT4) length) {
      LogPrintf ( LOG_CRITICAL, "%s: requested bin %d longer than series length %d\n", __func__, newBinCount, length);
      XLAL_ERROR( XLAL_EINVAL );
    }
    UINT4 checkIfBinAppeared = 0;
    for (INT4 checkExistBadBin = 0; checkExistBadBin < binCount; checkExistBadBin++)
      {
	if (badBin < 0)
	  {
	    LogPrintf ( LOG_CRITICAL, "badBin %d negative", badBin);
	    XLAL_ERROR(XLAL_ETYPE);
	  }
	if (badBinData->data[checkExistBadBin] == (UINT4) badBin)
	  checkIfBinAppeared++;
      }
    if (checkIfBinAppeared == 0)
      {
	badBinData->data[newBinCount] = badBin;
	newBinCount++;
      }
  }

  return newBinCount;

}

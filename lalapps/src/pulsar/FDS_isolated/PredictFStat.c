/*
 * Copyright (C) 2006 Iraj Gholami, Reinhard Prix
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

/*********************************************************************************/
/** \author I. Gholami, R. Prix
 * \file 
 * \brief
 * Calculate the F-statistic Semi-Analytically of pulsar GW signals.
 * Implements the so-called "F-statistic" as introduced in \ref JKS98 and Cutler-Schutz 2005.
 *                                                                          
 *********************************************************************************/
#include "config.h"

/* System includes */
#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

/* LAL-includes */
#include <lal/AVFactories.h>
#include <lal/LALInitBarycenter.h>
#include <lal/UserInput.h>
#include <lal/SFTfileIO.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/ComputeFstat.h>
#include <lal/LALHough.h>

#include <lalapps.h>

#include "LogPrintf.h"

/* local includes */

RCSID( "$Id$");

/*---------- DEFINES ----------*/

#define MAXFILENAMELENGTH 256   /* Maximum # of characters of a SFT filename */

#define EPHEM_YEARS  "00-04"	/**< default range: override with --ephemYear */

#define TRUE (1==1)
#define FALSE (1==0)

/*----- Error-codes -----*/
#define PREDICTFSTAT_ENULL 	1
#define PREDICTFSTAT_ESYS     	2
#define PREDICTFSTAT_EINPUT   	3
#define PREDICTFSTAT_EMEM   	4
#define PREDICTFSTAT_ENONULL 	5
#define PREDICTFSTAT_EXLAL	6

#define PREDICTFSTAT_MSGENULL 	"Arguments contained an unexpected null pointer"
#define PREDICTFSTAT_MSGESYS	"System call failed (probably file IO)"
#define PREDICTFSTAT_MSGEINPUT  "Invalid input"
#define PREDICTFSTAT_MSGEMEM   	"Out of memory. Bad."
#define PREDICTFSTAT_MSGENONULL "Output pointer is non-NULL"
#define PREDICTFSTAT_MSGEXLAL	"XLALFunction-call failed"

/** convert GPS-time to REAL8 */
#define GPS2REAL8(gps) (1.0 * (gps).gpsSeconds + 1.e-9 * (gps).gpsNanoSeconds )

#define MYMAX(x,y) ( (x) > (y) ? (x) : (y) )
#define MYMIN(x,y) ( (x) < (y) ? (x) : (y) )

#define SQ(x) ((x)*(x))

#define LAL_INT4_MAX 2147483647

/** Configuration settings required for and defining a coherent pulsar search.
 * These are 'pre-processed' settings, which have been derived from the user-input.
 */
typedef struct {
  CHAR *dataSummary;            /**< descriptive string describing the data */
  REAL8 aPlus, aCross;		/**< internally always use Aplus, Across */
  REAL8 A, B, C;		/**< antenna-pattern integrals */
  REAL8 TsftShat;		/**< noise normalization */
} ConfigVariables;

/*---------- Global variables ----------*/
extern int vrbflg;		/**< defined in lalapps.c */

ConfigVariables GV;		/**< global container for various derived configuration settings */

/* ----- User-variables: can be set from config-file or command-line */
BOOLEAN uvar_help;

INT4 uvar_RngMedWindow;

REAL8 uvar_aPlus;
REAL8 uvar_aCross;
REAL8 uvar_psi;
REAL8 uvar_h0;
REAL8 uvar_cosi;
REAL8 uvar_cosiota;	/* DEPRECATED in favor of cosi */
REAL8 uvar_Freq;
REAL8 uvar_Alpha;
REAL8 uvar_Delta;

CHAR *uvar_IFO;
CHAR *uvar_ephemDir;
CHAR *uvar_ephemYear;
CHAR *uvar_DataFiles;
CHAR *uvar_outputFstat;
INT4 uvar_minStartTime;
INT4 uvar_maxEndTime;

/* ---------- local prototypes ---------- */
int main(int argc,char *argv[]);

void initUserVars (LALStatus *);
void InitPFS ( LALStatus *, ConfigVariables *cfg );
void InitEphemeris (LALStatus *, EphemerisData *edat, const CHAR *ephemDir, const CHAR *ephemYear, LIGOTimeGPS epoch);

/*---------- empty initializers ---------- */
static const SFTConstraints empty_SFTConstraints;

/*----------------------------------------------------------------------*/
/* Main Function starts here */
/*----------------------------------------------------------------------*/
/** 
 * MAIN function of PredictFStat code.
 * Calculates the F-statistic for a given position in the sky and detector 
 * semi-analytically and outputs the final 2F value.
 */
int main(int argc,char *argv[]) 
{
  LALStatus status = blank_status;	/* initialize status */
  REAL8 rho2;	/* SNR^2 */

  lalDebugLevel = 0;  
  vrbflg = 1;	/* verbose error-messages */
  
  /* set LAL error-handler */
  lal_errhandler = LAL_ERR_EXIT;

  /* register all user-variable */
  LAL_CALL (LALGetDebugLevel(&status, argc, argv, 'v'), &status);
  LAL_CALL (initUserVars(&status), &status); 	

  /* do ALL cmdline and cfgfile handling */
  LAL_CALL (LALUserVarReadAllInput(&status, argc,argv), &status);	

  if (uvar_help)	/* if help was requested, we're done here */
    exit (0);

  /* Initialize code-setup */
  LAL_CALL ( InitPFS(&status, &GV), &status);
  
  { /* Calculating the F-Statistic */
    REAL8 al1, al2, al3;
    REAL8 Ap2 = SQ(GV.aPlus);
    REAL8 Ac2 = SQ(GV.aCross);
    REAL8 cos2psi2 = SQ( cos(2*uvar_psi) );
    REAL8 sin2psi2 = SQ( sin(2*uvar_psi) );

    al1 = Ap2 * cos2psi2 + Ac2 * sin2psi2;	/* A1^2 + A3^2 */
    al2 = Ap2 * sin2psi2 + Ac2 * cos2psi2;	/* A2^2 + A4^2 */
    al3 = ( Ap2 - Ac2 ) * sin(2.0*uvar_psi) * cos(2.0*uvar_psi);	/* A1 A2 + A3 A4 */
    
    /* SNR^2 */
    rho2 = 0.5 * GV.TsftShat * (GV.A * al1 + GV.B * al2 + 2.0 * GV.C * al3 );
  }

  fprintf(stdout, "\n%.1f\n", 4.0 + rho2);

  /* output predicted Fstat-value into file, if requested */
  if (uvar_outputFstat)
    {
      FILE *fpFstat = NULL;
      CHAR *logstr = NULL;
      
      if ( (fpFstat = fopen (uvar_outputFstat, "wb")) == NULL)
	{
	  LALPrintError ("\nError opening file '%s' for writing..\n\n", uvar_outputFstat);
	  return (PREDICTFSTAT_ESYS);
	}
      
      /* log search-footprint at head of output-file */
      LAL_CALL( LALUserVarGetLog (&status, &logstr,  UVAR_LOGFMT_CMDLINE ), &status );

      fprintf(fpFstat, "## %s\n## %s\n",
	      "$Id$",
	      logstr );
      LALFree ( logstr );
      /* append 'dataSummary' */
      fprintf (fpFstat, "%s", GV.dataSummary );
      fprintf (fpFstat, "## E[2F]   sigma[2F] \n");
      fprintf (fpFstat, "  %g    %g\n", 4.0 + rho2,  sqrt( 4.0 * ( 2.0 + rho2 )  ) );
      fclose (fpFstat);
    } /* if outputFstat */   

  /* Free config-Variables and userInput stuff */
  LAL_CALL (LALDestroyUserVars (&status), &status);
  LALFree ( GV.dataSummary );
  
  /* did we forget anything ? */
  LALCheckMemoryLeaks();
  
  return 0;
  
} /* main() */

/** 
 * Register all our "user-variables" that can be specified from cmd-line and/or config-file.
 * Here we set defaults for some user-variables and register them with the UserInput module.
 */
void
initUserVars (LALStatus *status)
{

  INITSTATUS( status, "initUserVars", rcsid );
  ATTATCHSTATUSPTR (status);

  /* set a few defaults */
  uvar_RngMedWindow = 50;	/* for running-median */
  
  uvar_ephemYear = LALCalloc (1, strlen(EPHEM_YEARS)+1);
  strcpy (uvar_ephemYear, EPHEM_YEARS);
  
#define DEFAULT_EPHEMDIR "env LAL_DATA_PATH"
  uvar_ephemDir = LALCalloc (1, strlen(DEFAULT_EPHEMDIR)+1);
  strcpy (uvar_ephemDir, DEFAULT_EPHEMDIR);
  
  uvar_help = FALSE;
  uvar_outputFstat = NULL;
  
  uvar_minStartTime = 0;
  uvar_maxEndTime = LAL_INT4_MAX;
  
  /* register all our user-variables */
  LALregBOOLUserVar(status,	help, 		'h', UVAR_HELP,     "Print this message"); 
  
  LALregREALUserVar(status,	h0,		's', UVAR_OPTIONAL, "Signal amplitude h_0");
  LALregREALUserVar(status,	cosi,		'i', UVAR_OPTIONAL, "Inclination of rotation-axis Cos(iota)");
  LALregREALUserVar(status,	cosiota,	 0 , UVAR_DEVELOPER,"[DEPRECATED] Use --cosi instead!");
  LALregREALUserVar(status, 	aPlus,	 	 0 , UVAR_OPTIONAL, "Alternative to {h0,cosi}: A_+ amplitude");
  LALregREALUserVar(status,	aCross,  	 0 , UVAR_OPTIONAL, "Alternative to {h0,cosi}: A_x amplitude");

  LALregREALUserVar(status,	psi,		'Y', UVAR_REQUIRED, "Polarisation in rad");
  LALregREALUserVar(status,	Alpha,		'a', UVAR_REQUIRED, "Sky position alpha (equatorial coordinates) in radians");
  LALregREALUserVar(status,	Delta,		'd', UVAR_REQUIRED, "Sky position delta (equatorial coordinates) in radians");
  LALregREALUserVar(status,	Freq,		'F', UVAR_REQUIRED, "Signal frequency (for noise-estimation)");
  
  LALregSTRINGUserVar(status,	DataFiles, 	'D', UVAR_REQUIRED, "File-pattern specifying (multi-IFO) input SFT-files"); 
  LALregSTRINGUserVar(status,	IFO, 		'I', UVAR_OPTIONAL, "Detector-constraint: 'G1', 'L1', 'H1', 'H2' ...(useful for single-IFO v1-SFTs only!)");
  LALregSTRINGUserVar(status,	ephemDir, 	'E', UVAR_OPTIONAL, "Directory where Ephemeris files are located");
  LALregSTRINGUserVar(status,	ephemYear, 	'y', UVAR_OPTIONAL, "Year (or range of years) of ephemeris files to be used");
  LALregSTRINGUserVar(status,	outputFstat,	  0,  UVAR_OPTIONAL, "Output-file for predicted F-stat value" );
  
  LALregINTUserVar ( status,	minStartTime, 	 0,  UVAR_OPTIONAL, "Earliest SFT-timestamp to include");
  LALregINTUserVar ( status,	maxEndTime, 	 0,  UVAR_OPTIONAL, "Latest SFT-timestamps to include");

  LALregINTUserVar(status,	RngMedWindow,	'k', UVAR_DEVELOPER, "Running-Median window size");  
  
  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* initUserVars() */

/** Load Ephemeris from ephemeris data-files  */
void
InitEphemeris (LALStatus * status,   
	       EphemerisData *edat,	/**< [out] the ephemeris-data */
	       const CHAR *ephemDir,	/**< directory containing ephems */
	       const CHAR *ephemYear,	/**< which years do we need? */
	       LIGOTimeGPS epoch	/**< epoch of observation */
	       )
{
#define FNAME_LENGTH 1024
  CHAR EphemEarth[FNAME_LENGTH];	/* filename of earth-ephemeris data */
  CHAR EphemSun[FNAME_LENGTH];	/* filename of sun-ephemeris data */
  LALLeapSecFormatAndAcc formatAndAcc = {LALLEAPSEC_GPSUTC, LALLEAPSEC_STRICT};
  INT4 leap;

  INITSTATUS( status, "InitEphemeris", rcsid );
  ATTATCHSTATUSPTR (status);
  
  ASSERT ( edat, status, PREDICTFSTAT_ENULL, PREDICTFSTAT_MSGENULL );
  ASSERT ( ephemYear, status, PREDICTFSTAT_ENULL, PREDICTFSTAT_MSGENULL );
  
  if ( ephemDir )
    {
      LALSnprintf(EphemEarth, FNAME_LENGTH, "%s/earth%s.dat", ephemDir, ephemYear);
      LALSnprintf(EphemSun, FNAME_LENGTH, "%s/sun%s.dat", ephemDir, ephemYear);
    }
  else
    {
      LALSnprintf(EphemEarth, FNAME_LENGTH, "earth%s.dat", ephemYear);
      LALSnprintf(EphemSun, FNAME_LENGTH, "sun%s.dat",  ephemYear);
    }
  EphemEarth[FNAME_LENGTH-1]=0;
  EphemSun[FNAME_LENGTH-1]=0;
  
  /* NOTE: the 'ephiles' are ONLY ever used in LALInitBarycenter, which is
   * why we can use local variables (EphemEarth, EphemSun) to initialize them.
   */
  edat->ephiles.earthEphemeris = EphemEarth;
  edat->ephiles.sunEphemeris = EphemSun;
  
  TRY (LALLeapSecs (status->statusPtr, &leap, &epoch, &formatAndAcc), status);
  edat->leap = (INT2) leap;
  
  TRY (LALInitBarycenter(status->statusPtr, edat), status);
  
  DETATCHSTATUSPTR ( status );
  RETURN ( status );
  
} /* InitEphemeris() */

/** Initialized Fstat-code: handle user-input and set everything up. */
void
InitPFS ( LALStatus *status, ConfigVariables *cfg )
{
  SFTCatalog *catalog = NULL;
  SFTConstraints constraints = empty_SFTConstraints;
  SkyPosition skypos;
  REAL8 S_hat;

  LIGOTimeGPS startTime, endTime;
  REAL8 duration, Tsft;
  LIGOTimeGPS minStartTimeGPS, maxEndTimeGPS;

  EphemerisData *edat = NULL;		    	/* ephemeris data */
  MultiAMCoeffs *multiAMcoef = NULL;
  MultiPSDVector *multiPSDs = NULL;
  MultiNoiseWeights *multiNoiseWeights = NULL;
  MultiSFTVector *multiSFTs = NULL;	    	/* multi-IFO SFT-vectors */
  MultiDetectorStateSeries *multiDetStates = NULL; /* pos, vel and LMSTs for detector at times t_i */

  INITSTATUS (status, "InitPFS", rcsid);
  ATTATCHSTATUSPTR (status);

  { /* Check user-input consistency */
    BOOLEAN have_h0, have_cosi, have_cosiota, have_Ap, have_Ac;
    REAL8 cosi;

    have_h0 = LALUserVarWasSet ( &uvar_h0 );
    have_cosi = LALUserVarWasSet ( &uvar_cosi );
    have_cosiota = LALUserVarWasSet ( &uvar_cosiota );
    have_Ap = LALUserVarWasSet ( &uvar_aPlus );
    have_Ac = LALUserVarWasSet ( &uvar_aCross );

    /* ----- handle cosi/cosiota ambiguity */
    if ( (have_cosi && have_cosiota) || ( !have_cosi && !have_cosiota) ) {
      LogPrintf (LOG_CRITICAL, "Need EITHER --cosi [preferred] OR --cosiota [deprecated]!\n");
      ABORT ( status, PREDICTFSTAT_EINPUT, PREDICTFSTAT_MSGEINPUT );
    }
    if ( have_cosiota )
      cosi = uvar_cosiota;
    else
      cosi = uvar_cosi;
    have_cosi = TRUE;
    /* ----- handle {h0,cosi} || {aPlus,aCross} freedom ----- */
    if ( ( have_h0 && !have_cosi ) || ( !have_h0 && have_cosi ) )
      {
	LogPrintf (LOG_CRITICAL, "Need both (h0, cosi) to specify signal!\n");
	ABORT ( status, PREDICTFSTAT_EINPUT, PREDICTFSTAT_MSGEINPUT );
      }
    if ( ( have_Ap && !have_Ac) || ( !have_Ap && have_Ac ) )
      {
	LogPrintf (LOG_CRITICAL, "Need both (aPlus, aCross) to specify signal!\n");
	ABORT ( status, PREDICTFSTAT_EINPUT, PREDICTFSTAT_MSGEINPUT );
      }
    if ( have_h0 && have_Ap )
      {
	LogPrintf (LOG_CRITICAL, "Overdetermined: specify EITHER (h0,cosi) OR (aPlus,aCross)!\n");
	ABORT ( status, PREDICTFSTAT_EINPUT, PREDICTFSTAT_MSGEINPUT );
      }
    /* ----- internally we always use Aplus, Across */
    if ( have_h0 )
      {
	cfg->aPlus = 0.5 * uvar_h0 * ( 1.0 + SQ( cosi) );
	cfg->aCross = uvar_h0 * uvar_cosi;
      }
    else
      {
	cfg->aPlus = uvar_aPlus;
	cfg->aCross = uvar_aCross;
      }
  }/* check user-input */


  /* ----- prepare SFT-reading ----- */
  if ( LALUserVarWasSet ( &uvar_IFO ) )
    if ( (constraints.detector = XLALGetChannelPrefix ( uvar_IFO )) == NULL ) {
      ABORT ( status,  PREDICTFSTAT_EINPUT,  PREDICTFSTAT_MSGEINPUT);
    }

  minStartTimeGPS.gpsSeconds = uvar_minStartTime;
  minStartTimeGPS.gpsNanoSeconds = 0;
  maxEndTimeGPS.gpsSeconds = uvar_maxEndTime;
  maxEndTimeGPS.gpsNanoSeconds = 0;
  constraints.startTime = &minStartTimeGPS;
  constraints.endTime = &maxEndTimeGPS;

  /* ----- get full SFT-catalog of all matching (multi-IFO) SFTs */
  LogPrintf (LOG_DEBUG, "Finding all SFTs to load ... ");
  TRY ( LALSFTdataFind ( status->statusPtr, &catalog, uvar_DataFiles, &constraints ), status);    
  LogPrintfVerbatim (LOG_DEBUG, "done. (found %d SFTs)\n", catalog->length);
  if ( constraints.detector ) 
    LALFree ( constraints.detector );

  if ( catalog->length == 0 ) 
    {
      LogPrintf (LOG_CRITICAL, "No matching SFTs for pattern '%s'!\n", uvar_DataFiles );
      ABORT ( status,  PREDICTFSTAT_EINPUT,  PREDICTFSTAT_MSGEINPUT);
    }

  /* ----- deduce start- and end-time of the observation spanned by the data */
  {
    UINT4 numSFTs = catalog->length;
    Tsft = 1.0 / catalog->data[0].header.deltaF;
    startTime = catalog->data[0].header.epoch;
    endTime   = catalog->data[numSFTs-1].header.epoch;
    LALAddFloatToGPS(status->statusPtr, &endTime, &endTime, Tsft );	/* can't fail */
    duration = GPS2REAL8(endTime) - GPS2REAL8 (startTime);
  }

  {/* ----- load the multi-IFO SFT-vectors ----- */
    UINT4 wings = uvar_RngMedWindow/2 + 10;   /* extra frequency-bins needed for rngmed */
    REAL8 fMax = uvar_Freq + 1.0 * wings / Tsft;
    REAL8 fMin = uvar_Freq - 1.0 * wings / Tsft;

    LogPrintf (LOG_DEBUG, "Loading SFTs ... ");
    TRY ( LALLoadMultiSFTs ( status->statusPtr, &multiSFTs, catalog, fMin, fMax ), status );
    LogPrintfVerbatim (LOG_DEBUG, "done.\n");
    TRY ( LALDestroySFTCatalog ( status->statusPtr, &catalog ), status );
  }

  /* ----- produce a log-string describing the data-specific setup ----- */
  {
    struct tm utc;
    time_t tp;
    CHAR dateStr[512], line[512], summary[1024];
    UINT4 i, numDet;
    numDet = multiSFTs->length;
    tp = time(NULL);
    sprintf (summary, "## Date: %s", asctime( gmtime( &tp ) ) );
    strcat (summary, "## Loaded SFTs: [ " );
    for ( i=0; i < numDet; i ++ ) {
      sprintf (line, "%s:%d%s",  multiSFTs->data[i]->data->name, multiSFTs->data[i]->length,
	       (i < numDet - 1)?", ":" ]\n");
      strcat ( summary, line );
    }
    utc = *XLALGPSToUTC( &utc, (INT4)GPS2REAL8(startTime) );
    strcpy ( dateStr, asctime(&utc) );
    dateStr[ strlen(dateStr) - 1 ] = 0;
    sprintf (line, "## Start GPS time tStart = %12.3f    (%s GMT)\n", GPS2REAL8(startTime), dateStr);
    strcat ( summary, line );
    sprintf (line, "## Total time spanned    = %12.3f s  (%.1f hours)\n", duration, duration/3600 );
    strcat ( summary, line );

    if ( (cfg->dataSummary = LALCalloc(1, strlen(summary) + 1 )) == NULL ) {
      ABORT (status, PREDICTFSTAT_EMEM, PREDICTFSTAT_MSGEMEM);
    }
    strcpy ( cfg->dataSummary, summary );

    LogPrintfVerbatim( LOG_DEBUG, cfg->dataSummary );
  } /* write dataSummary string */

  { /* ----- load ephemeris-data ----- */
    CHAR *ephemDir;

    edat = LALCalloc(1, sizeof(EphemerisData));
    if ( LALUserVarWasSet ( &uvar_ephemDir ) )
      ephemDir = uvar_ephemDir;
    else
      ephemDir = NULL;
    TRY( InitEphemeris (status->statusPtr, edat, ephemDir, uvar_ephemYear, startTime ), status);
  }

  /* ----- obtain the (multi-IFO) 'detector-state series' for all SFTs ----- */
  TRY (LALGetMultiDetectorStates( status->statusPtr, &multiDetStates, multiSFTs, edat), status );

  /* normalize skyposition: correctly map into [0,2pi]x[-pi/2,pi/2] */
  skypos.longitude = uvar_Alpha;
  skypos.latitude = uvar_Delta;
  skypos.system = COORDINATESYSTEM_EQUATORIAL;
  TRY (LALNormalizeSkyPosition ( status->statusPtr, &skypos, &skypos), status);

  TRY ( LALGetMultiAMCoeffs ( status->statusPtr, &multiAMcoef, multiDetStates, skypos ), status);
  TRY ( LALNormalizeMultiSFTVect(status->statusPtr, &multiPSDs, multiSFTs, uvar_RngMedWindow ), status);
  TRY ( LALComputeMultiNoiseWeights (status->statusPtr, &multiNoiseWeights, &S_hat, multiPSDs, 
				     uvar_RngMedWindow, 0 ), status );

  cfg->TsftShat = Tsft * S_hat;	/* overall inverse noise-norm for Fstat */

  /* noise-weighting of Antenna-patterns and compute A,B,C */
  if ( XLALWeighMultiAMCoeffs ( multiAMcoef, multiNoiseWeights ) != XLAL_SUCCESS ) {
    LogPrintf (LOG_CRITICAL, "XLALWeighMultiAMCoeffs() failed with error = %d\n\n", xlalErrno );
    ABORT ( status, COMPUTEFSTATC_EXLAL, COMPUTEFSTATC_MSGEXLAL );
  }

  /* OK: we only need the integrated antenna-pattern coefficients A,B,C, D */
  cfg->A = multiAMcoef->A;
  cfg->B = multiAMcoef->B;
  cfg->C = multiAMcoef->C;

  /* free everything not needed any more */
  TRY ( LALDestroyMultiPSDVector (status->statusPtr, &multiPSDs ), status );
  TRY ( LALDestroyMultiNoiseWeights (status->statusPtr, &multiNoiseWeights ), status );
  TRY ( LALDestroyMultiSFTVector (status->statusPtr, &multiSFTs ), status );
  XLALDestroyMultiDetectorStateSeries ( multiDetStates );
  XLALDestroyMultiAMCoeffs ( multiAMcoef );

  /* Free ephemeris data */
  LALFree(edat->ephemE);
  LALFree(edat->ephemS);
  LALFree(edat);


  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* InitPFS() */

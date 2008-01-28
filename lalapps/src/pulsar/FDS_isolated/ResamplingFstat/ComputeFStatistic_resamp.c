/*
 * Copyright (C) 2005, 2006, 2007 Reinhard Prix, Iraj Gholami
 * Copyright (C) 2004 Reinhard Prix
 * Copyright (C) 2002, 2003, 2004 M.A. Papa, X. Siemens, Y. Itoh
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
/** \author R. Prix, I. Gholami, Y. Ioth, Papa, X. Siemens
 * \file
 * \brief
 * Calculate the F-statistic for a given parameter-space of pulsar GW signals.
 * Implements the so-called "F-statistic" as introduced in \ref JKS98.
 *
 *********************************************************************************/
#include "config.h"

/* System includes */
#include <math.h>
#include <stdio.h>
#include <time.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif


int finite(double);

/* LAL-includes */
#include <lal/AVFactories.h>
#include <lal/LALInitBarycenter.h>
#include <lal/UserInput.h>
#include <lal/SFTfileIO.h>
#include <lal/ExtrapolatePulsarSpins.h>
#include <lal/FrequencySeries.h>

#include <lal/NormalizeSFTRngMed.h>
#include <lal/ComputeFstat.h>
#include <lal/LALHough.h>

#include <lal/LogPrintf.h>
#include <lal/DopplerFullScan.h>

#include <lalapps.h>

/* local includes */

#include "../HeapToplist.h"


RCSID( "$Id$");

/*---------- DEFINES ----------*/

#define MAXFILENAMELENGTH 256   /* Maximum # of characters of a SFT filename */

#define EPHEM_YEARS  "00-04"	/**< default range: override with --ephemYear */

#define TRUE (1==1)
#define FALSE (1==0)

/*----- SWITCHES -----*/
#define NUM_SPINS 4		/* number of spin-values to consider: {f, fdot, f2dot, ... } */

/*----- Error-codes -----*/
#define COMPUTEFSTATISTIC_ENULL 	1
#define COMPUTEFSTATISTIC_ESYS     	2
#define COMPUTEFSTATISTIC_EINPUT   	3
#define COMPUTEFSTATISTIC_EMEM   	4
#define COMPUTEFSTATISTIC_ENONULL 	5
#define COMPUTEFSTATISTIC_EXLAL		6

#define COMPUTEFSTATISTIC_MSGENULL 	"Arguments contained an unexpected null pointer"
#define COMPUTEFSTATISTIC_MSGESYS	"System call failed (probably file IO)"
#define COMPUTEFSTATISTIC_MSGEINPUT   	"Invalid input"
#define COMPUTEFSTATISTIC_MSGEMEM   	"Out of memory. Bad."
#define COMPUTEFSTATISTIC_MSGENONULL 	"Output pointer is non-NULL"
#define COMPUTEFSTATISTIC_MSGEXLAL	"XLALFunction-call failed"

/*----- Macros -----*/

/** convert GPS-time to REAL8 */
#define GPS2REAL8(gps) (1.0 * (gps).gpsSeconds + 1.e-9 * (gps).gpsNanoSeconds )
#define SQ(x) ( (x) * (x) )

#define MYMAX(x,y) ( (x) > (y) ? (x) : (y) )
#define MYMIN(x,y) ( (x) < (y) ? (x) : (y) )

#define LAL_INT4_MAX 2147483647

/*---------- internal types ----------*/

/** What info do we want to store in our toplist? */
typedef struct {
  PulsarDopplerParams doppler;		/**< Doppler params of this 'candidate' */
  Fcomponents  Fstat;			/**< the Fstat-value (plus Fa,Fb) for this candidate */
  CmplxAntennaPatternMatrix Mmunu;		/**< antenna-pattern matrix Mmunu = 0.5* Sinv*Tsft * [ Ad, Cd; Cd; Bd ] */
} FstatCandidate;


/** moving 'Scanline window' of candidates on the scan-line,
 * which is used to find local 1D maxima.
 */
typedef struct
{
  UINT4 length;
  FstatCandidate *window; 		/**< array holding candidates */
  FstatCandidate *center;		/**< pointer to middle candidate in window */
} scanlineWindow_t;

/** Configuration settings required for and defining a coherent pulsar search.
 * These are 'pre-processed' settings, which have been derived from the user-input.
 */
typedef struct {
  REAL8 Tsft;                               /**< length of one SFT in seconds */
  LIGOTimeGPS refTime;			    /**< reference-time for pulsar-parameters in SBB frame */
  /* -------------------- Resampling -------------------- */
  REAL8 FFTFreqBand;                        /**< treated outside of DopplerScan for resampling-technique */
  /* ---------------------------------------------------- */
  DopplerRegion searchRegion;		    /**< parameter-space region (at *internalRefTime*) to search over */
  DopplerFullScanState *scanState;          /**< current state of the Doppler-scan */
  PulsarDopplerParams stepSizes;	    /**< user-preferences on Doppler-param step-sizes */
  EphemerisData *ephemeris;		    /**< ephemeris data (from LALInitBarycenter()) */
  MultiSFTVector *multiSFTs;		    /**< multi-IFO SFT-vectors */
  MultiDetectorStateSeries *multiDetStates; /**< pos, vel and LMSTs for detector at times t_i */
  MultiNoiseWeights *multiNoiseWeights;	    /**< normalized noise-weights of those SFTs */
  ComputeFParams CFparams;		    /**< parameters for Fstat (e.g Dterms, SSB-prec,...) */
  CHAR *logstring;                          /**< log containing max-info on this search setup */
  toplist_t* FstatToplist;		    /**< sorted 'toplist' of the NumCandidatesToKeep loudest candidates */
  scanlineWindow_t *scanlineWindow;         /**< moving window of candidates on scanline to find local maxima */
} ConfigVariables;

LALUnit empty_Unit;

/*---------- Global variables ----------*/
extern int vrbflg;		/**< defined in lalapps.c */

/* ----- User-variables: can be set from config-file or command-line */
INT4 uvar_Dterms;
CHAR *uvar_IFO;
BOOLEAN uvar_SignalOnly;
BOOLEAN uvar_UseNoiseWeights;
REAL8 uvar_Freq;
REAL8 uvar_FreqBand;
/* REAL8 uvar_dFreq;   deactivated for resampling version */
REAL8 uvar_Alpha;
REAL8 uvar_dAlpha;
REAL8 uvar_AlphaBand;
REAL8 uvar_Delta;
REAL8 uvar_dDelta;
REAL8 uvar_DeltaBand;
/* 1st spindown */
REAL8 uvar_f1dot;
REAL8 uvar_df1dot;
REAL8 uvar_f1dotBand;
/* 2nd spindown */
REAL8 uvar_f2dot;
REAL8 uvar_df2dot;
REAL8 uvar_f2dotBand;
/* 3rd spindown */
REAL8 uvar_f3dot;
REAL8 uvar_df3dot;
REAL8 uvar_f3dotBand;
/* --- */
REAL8 uvar_TwoFthreshold;
CHAR *uvar_ephemDir;
CHAR *uvar_ephemYear;
INT4  uvar_gridType;
INT4  uvar_metricType;
BOOLEAN uvar_projectMetric;
REAL8 uvar_metricMismatch;
CHAR *uvar_skyRegion;
CHAR *uvar_DataFiles;
BOOLEAN uvar_help;
CHAR *uvar_outputLogfile;
CHAR *uvar_outputFstat;
CHAR *uvar_outputLoudest;

INT4 uvar_NumCandidatesToKeep;
INT4 uvar_clusterOnScanline;

CHAR *uvar_gridFile;
REAL8 uvar_dopplermax;
INT4 uvar_RngMedWindow;
REAL8 uvar_refTime;
REAL8 uvar_internalRefTime;
INT4 uvar_SSBprecision;

INT4 uvar_minStartTime;
INT4 uvar_maxEndTime;
CHAR *uvar_workingDir;
REAL8 uvar_timerCount;
INT4 uvar_upsampleSFTs;

/* Defining a multi-IFO complex time series. Keeping the Real and Imaginary parts as seperate vectors, since it is necessary to interpolate them seperately */

typedef struct
{
  UINT4 length;                    /* Number of IFOs */
  REAL8TimeSeries* Real;           /* Real part of the time series */
  REAL8TimeSeries* Imag;           /* Imaginary part of the time series */
}MultiCOMPLEX8TimeSeries;

/* Using a temporary Time Series for now */
MultiCOMPLEX8TimeSeries *temp_TimeSeries;

/* A contiguity structure required by the preprocessing function in order to store the information pertaining to the contiguity of SFTs and the gaps between them */

typedef struct
{
  UINT4  length;                    /* Number of Contiguous blocks */
  UINT4* NumContinuous;             /* Number of Contiguous SFTs in each block */
  REAL8* Gap;                       /* Gap between two Contiguous blocks in seconds */
}Contiguity;


/* ---------- local prototypes ---------- */
void CalcTimeSeries(LALStatus *, MultiSFTVector *multiSFTs, MultiCOMPLEX8TimeSeries* Tseries);
int main(int argc,char *argv[]);
void initUserVars (LALStatus *);
void InitFStat ( LALStatus *, ConfigVariables *cfg );
void Freemem(LALStatus *,  ConfigVariables *cfg);

void WriteFStatLog (LALStatus *, CHAR *argv[], const CHAR *log_fname);
void checkUserInputConsistency (LALStatus *);
int outputBeamTS( const CHAR *fname, const AMCoeffs *amcoe, const DetectorStateSeries *detStates );
void InitEphemeris (LALStatus *, EphemerisData *edat, const CHAR *ephemDir, const CHAR *ephemYear, LIGOTimeGPS epoch, BOOLEAN isLISA);
void getUnitWeights ( LALStatus *, MultiNoiseWeights **multiWeights, const MultiSFTVector *multiSFTs );

int write_FstatCandidate_to_fp ( FILE *fp, const FstatCandidate *thisFCand );
int write_PulsarCandidate_to_fp ( FILE *fp,  const PulsarCandidate *pulsarParams, const FstatCandidate *Fcand );

int compareFstatCandidates ( const void *candA, const void *candB );
void getLogString ( LALStatus *status, CHAR **logstr, const ConfigVariables *cfg );

const char *va(const char *format, ...);	/* little var-arg string helper function */

/* ---------- scanline window functions ---------- */
scanlineWindow_t *XLALCreateScanlineWindow ( UINT4 windowWings );
void XLALDestroyScanlineWindow ( scanlineWindow_t *scanlineWindow );
int XLALAdvanceScanlineWindow ( const FstatCandidate *nextCand, scanlineWindow_t *scanWindow );
BOOLEAN XLALCenterIsLocalMax ( const scanlineWindow_t *scanWindow );


/*---------- empty initializers ---------- */
static const ConfigVariables empty_ConfigVariables;
static const FstatCandidate empty_FstatCandidate;

/*----------------------------------------------------------------------*/
/* Function definitions start here */
/*----------------------------------------------------------------------*/

/* CalcTimeSeries calculates a heterodyned downsampled time series.
   It heterodynes the middle of the band to zero and downsamples
   appropriately. The resulting time series is complex and is stored
   in the MultiComplex8TimesSeries structure.
*/

void CalcTimeSeries(LALStatus *Status, MultiSFTVector *multiSFTs, MultiCOMPLEX8TimeSeries* Tseries)
{
  Contiguity C;            
  UINT4 i,j,k,p,q;         /* Counters */

  printf("Here for now :) %d \n",multiSFTs->length);
  /*loop over IFOs*/
  for(i=0;i<multiSFTs->length;i++)
    {
      SFTVector *SFT_Vect = multiSFTs->data[i]; /* Copy local  SFTVect */
      BOOLEAN IsFirst = TRUE;                   /* Bookkeeping Variable */
      UINT4 NumCount = 1;                       /* Number of SFTs in each block */
      /* Initialize C, length = 0 to begin with. But we need to assign memory to Gap and NumContinuous. The maximum number of continuous blocks is the total number of SFTs, therefore it is appropriate to assign that much memory */
      C.length = 0;                    
      C.Gap = (REAL8*)LALMalloc(sizeof(REAL8)*SFT_Vect->length); 
      C.NumContinuous = (UINT4*)LALMalloc(sizeof(UINT4)*SFT_Vect->length);

      REAL8 SFT_TimeBaseline = 0;               /* Time Baseline of SFTs */

      printf("The length of the SFT_Vect = %g\n",SFT_Vect->data[0].deltaF);

      /* In order to avoid Seg Faults */
      /* Time_Baseline = 1/deltaF */
      if(SFT_Vect->length)                      
	SFT_TimeBaseline = 1.0/SFT_Vect->data[0].deltaF; 

      /* Another Bookkeeping variable */
      UINT4 NumofBlocks = 0;

      /* Loop over all SFTs in this SFTVector */
      for(j=0;j<SFT_Vect->length;j++)
	{
	  /* Stores difference in times between two consecutive SFTs */
	  REAL8 TimeDiff;             

	  if(IsFirst)
	    {
	      IsFirst = FALSE;        /* No Longer First */
	      NumCount = 1;           /* Minimum SFTs in each block is 1 */
	    }
	  else
	    {
	        /* Calculate the difference in start times between this SFT and the one before it, since this one isnt the first */
	      TimeDiff = GPS2REAL8(SFT_Vect->data[i].epoch)-GPS2REAL8(SFT_Vect->data[i-1].epoch);                   

	      /* If true, it means that these two SFTs are next to each other in time and hence add 1 to the Number continuous */
	      if(TimeDiff == SFT_TimeBaseline) 
		NumCount++;           
	      
	      /* Now we are dealing with a new block */
	      else                    
		{
		  IsFirst = TRUE;

		  /* Restart Cycle with this SFT being first */
		  j--;      

		  /* Record the Gap between these two blocks */
		  C.Gap[NumofBlocks] = TimeDiff;

		  /* Also Record how many SFTs in this block */
		  C.NumContinuous[NumofBlocks] = NumCount;
		  
		  /* One more in this Block */
		  NumofBlocks += 1;
		}
	    }/*Top most else() */
	}
      
      /* Record information for the last block */
      C.Gap[NumofBlocks] = 0;
      C.NumContinuous[NumofBlocks] = NumCount;
      C.length = NumofBlocks + 1;
      for(k=0;k<C.length;k++)
	{
	  printf("Number Continuous = %d , Gap = %g\n",C.NumContinuous[k],C.Gap[k]);
	}
    }/*Loop over Multi-IFOs */
}/*CalctimeSeries()

/**
 * MAIN function of ComputeFStatistic code.
 * Calculate the F-statistic over a given portion of the parameter-space
 * and write a list of 'candidates' into a file(default: 'Fstats').
 */
int main(int argc,char *argv[])
{
  LALStatus status = blank_status;	/* initialize status */

  FILE *fpFstat = NULL;
  ComputeFBuffer cfBuffer = empty_ComputeFBuffer;
  REAL8 numTemplates, templateCounter;
  REAL8 tickCounter;
  time_t clock0;
  REAL8FrequencySeries *fstatVector = NULL;
  PulsarDopplerParams dopplerpos = empty_PulsarDopplerParams;		/* current search-parameters */
  FstatCandidate loudestFCand = empty_FstatCandidate, thisFCand = empty_FstatCandidate;
  UINT4 k;
  ConfigVariables GV = empty_ConfigVariables;		/**< global container for various derived configuration settings */

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

  /* set log-level */
  LogSetLevel ( lalDebugLevel );

  /* keep a log-file recording all relevant parameters of this search-run */
  if ( uvar_outputLogfile ) {
    LAL_CALL (WriteFStatLog ( &status, argv, uvar_outputLogfile ), &status );
  }

  /* do some sanity checks on the user-input before we proceed */
  LAL_CALL ( checkUserInputConsistency(&status), &status);

  /* Initialization the common variables of the code, */
  /* like ephemeries data and template grids: */
  LAL_CALL ( InitFStat(&status, &GV), &status);

  /* if a complete output of the F-statistic file was requested,
   * we open and prepare the output-file here */
  if (uvar_outputFstat)
    {
      if ( (fpFstat = fopen (uvar_outputFstat, "wb")) == NULL)
	{
	  LALPrintError ("\nError opening file '%s' for writing..\n\n", uvar_outputFstat);
	  return (COMPUTEFSTATISTIC_ESYS);
	}

      fprintf (fpFstat, "%s", GV.logstring );
    } /* if outputFstat */

  /* count number of templates */
  numTemplates = XLALNumDopplerTemplates ( GV.scanState );

  /* prepare Fstat-vector over frequencies to hold output results */
  {
    REAL8 dFreq = 1.0 / GV.multiDetStates->Tspan;
    UINT4 numFreqBins = floor( GV.FFTFreqBand / dFreq +1e-6) + 1;
    REAL8 Freq0 = GV.searchRegion.fkdot[0];
    fstatVector = XLALCreateREAL8FrequencySeries ("Fstat vector", &GV.searchRegion.refTime, Freq0, dFreq, &empty_Unit, numFreqBins );
    if ( fstatVector == NULL ) {
      fprintf ( stderr, "Oops, out of memory!\n");
      return  COMPUTEFSTATISTIC_EMEM;
    }
  } /* setup Fstat-vector */

  /*----------------------------------------------------------------------
   * main loop: demodulate data for each point in the sky-position grid
   * and for each value of the frequency-spindown
   */
  templateCounter = 0.0;
  tickCounter = 0;
  clock0 = time(NULL);

  /*Call the CalcTimeSeries Function Here*/
  CalcTimeSeries(&status, GV.multiSFTs,temp_TimeSeries);

  while ( XLALNextDopplerPos( &dopplerpos, GV.scanState ) == 0 )
    {
      /* main function call: compute F-statistic over frequency-band  */
      LAL_CALL( ComputeFStatFreqBand ( &status, fstatVector, &dopplerpos, GV.multiSFTs, GV.multiNoiseWeights,
				       GV.multiDetStates, &GV.CFparams ), &status );

      /* Progress meter */
      templateCounter += 1.0;
      if ( lalDebugLevel && ( ++tickCounter > uvar_timerCount) )
	{
	  REAL8 diffSec = time(NULL) - clock0 ;  /* seconds since start of loop*/
	  REAL8 taup = diffSec / templateCounter ;
	  REAL8 timeLeft = (numTemplates - templateCounter) *  taup;
	  tickCounter = 0.0;
	  LogPrintf (LOG_DEBUG, "Progress: %g/%g = %.2f %% done, Estimated time left: %.0f s\n",
		     templateCounter, numTemplates, templateCounter/numTemplates * 100.0, timeLeft);
	}

      for ( k=0; k < fstatVector->data->length; k++)
	{
	  REAL8 thisF = fstatVector->data->data[k];
	  REAL8 thisFreq = fstatVector->f0 + k * fstatVector->deltaF;

	  /* sanity check on the result */
	  if ( !finite ( thisF ) )
	    {
	      LogPrintf(LOG_CRITICAL, "non-finite F = %.16g\n", thisF );
	      LogPrintf (LOG_CRITICAL, "[Alpha,Delta] = [%.16g,%.16g],\nfkdot=[%.16g,%.16g,%.16g,%16.g]\n",
			 dopplerpos.Alpha, dopplerpos.Delta,
			 thisFreq, dopplerpos.fkdot[1], dopplerpos.fkdot[2], dopplerpos.fkdot[3] );
	  return -1;
	    }

	  /* propagate fkdot from internalRefTime back to refTime for outputting results */
	  /* FIXE: only do this for candidates we're going to write out */
	  dopplerpos.fkdot[0] = thisFreq;
	  LAL_CALL ( LALExtrapolatePulsarSpins ( &status, dopplerpos.fkdot, GV.refTime, dopplerpos.fkdot, GV.searchRegion.refTime ), &status );
	  dopplerpos.refTime = GV.refTime;

	  /* correct normalization in --SignalOnly case:
	   * we didn't normalize data by 1/sqrt(Tsft * 0.5 * Sh) in terms of
	   * the single-sided PSD Sh: the SignalOnly case is characterized by
	   * setting Sh->1, so we need to divide Fa,Fb by sqrt(0.5*Tsft) and F by (0.5*Tsft)
	   */
	  if ( uvar_SignalOnly )
	    {
	      REAL8 norm = 1.0 / sqrt( 0.5 * GV.Tsft );
	      thisF *= norm * norm;
	      thisF += 2;		/* compute E[2F]:= 4 + SNR^2 */
	    }
	  thisFCand.Fstat.F = thisF;
	  thisFCand.doppler = dopplerpos;

	  /* push new value onto scan-line buffer */
	  XLALAdvanceScanlineWindow ( &thisFCand, GV.scanlineWindow );

	  /* two types of threshold: fixed (TwoFThreshold) and dynamic (NumCandidatesToKeep) */
	  if ( XLALCenterIsLocalMax ( GV.scanlineWindow ) 					/* must be 1D local maximum */
	       && (2.0 * GV.scanlineWindow->center->Fstat.F >= uvar_TwoFthreshold) )	/* fixed threshold */
	    {
	      FstatCandidate *writeCand = GV.scanlineWindow->center;

	      /* insert this into toplist if requested */
	      if ( GV.FstatToplist  )			/* dynamic threshold */
		{
		  if ( insert_into_toplist(GV.FstatToplist, (void*)writeCand ) )
		    LogPrintf ( LOG_DETAIL, "Added new candidate into toplist: 2F = %f\n", 2.0 * writeCand->Fstat.F );
		  else
		    LogPrintf ( LOG_DETAIL, "NOT added the candidate into toplist: 2F = %f\n", 2 * writeCand->Fstat.F );
		}
	      else if ( fpFstat ) 				/* no toplist :write out immediately */
		{
		  if ( write_FstatCandidate_to_fp ( fpFstat, writeCand ) != 0 )
		    {
		      LogPrintf (LOG_CRITICAL, "Failed to write candidate to file.\n");
		      return -1;
		    }
		} /* if outputFstat */

	    } /* if 2F > threshold */

	  /* separately keep track of loudest candidate (for --outputLoudest) */
	  if ( thisFCand.Fstat.F > loudestFCand.Fstat.F )
	    loudestFCand = thisFCand;

	} /* inner loop about frequency-bins from resampling frequ-band */

    } /* while more Doppler positions to scan */

  /* ----- if using toplist: sort and write it out to file now ----- */
  if ( fpFstat && GV.FstatToplist )
    {
      UINT4 el;

      /* sort toplist */
      LogPrintf ( LOG_DEBUG, "Sorting toplist ... ");
      qsort_toplist ( GV.FstatToplist, compareFstatCandidates );
      LogPrintfVerbatim ( LOG_DEBUG, "done.\n");

      for ( el=0; el < GV.FstatToplist->elems; el ++ )
	{
	  const FstatCandidate *candi;
	  if ( ( candi = (const FstatCandidate *) toplist_elem ( GV.FstatToplist, el )) == NULL ) {
	    LogPrintf ( LOG_CRITICAL, "Internal consistency problems with toplist: contains fewer elements than expected!\n");
	    return -1;
	  }
	  if ( write_FstatCandidate_to_fp ( fpFstat, candi ) != 0 )
	    {
	      LogPrintf (LOG_CRITICAL, "Failed to write candidate to file.\n");
	      return -1;
	    }
	} /* for el < elems in toplist */

    } /* if fpFstat && toplist */

  if ( fpFstat )
    {
      fprintf (fpFstat, "%%DONE\n");
      fclose (fpFstat);
      fpFstat = NULL;
    }

  /* ----- estimate amplitude-parameters for the loudest canidate and output into separate file ----- */
  if ( uvar_outputLoudest )
    {
      FILE *fpLoudest;
      PulsarCandidate pulsarParams = empty_PulsarCandidate;
      pulsarParams.Doppler = loudestFCand.doppler;

      LAL_CALL(LALEstimatePulsarAmplitudeParams (&status, &pulsarParams, &loudestFCand.Fstat, &GV.searchRegion.refTime, &loudestFCand.Mmunu ),
	       &status );

      if ( (fpLoudest = fopen (uvar_outputLoudest, "wb")) == NULL)
	{
	  LALPrintError ("\nError opening file '%s' for writing..\n\n", uvar_outputLoudest);
	  return COMPUTEFSTATISTIC_ESYS;
	}

      /* write header with run-info */
      fprintf (fpLoudest, "%s", GV.logstring );

      /* write this 'candidate' to disc */
      if ( write_PulsarCandidate_to_fp ( fpLoudest,  &pulsarParams, &loudestFCand) != XLAL_SUCCESS )
	{
	  LogPrintf(LOG_CRITICAL, "call to write_PulsarCandidate_to_fp() failed!\n");
	  return COMPUTEFSTATISTIC_ESYS;
	}

      fclose (fpLoudest);

      gsl_matrix_free ( pulsarParams.AmpFisherMatrix );

    } /* write loudest candidate to file */

  LogPrintf (LOG_DEBUG, "Search finished.\n");

  /* Free memory */
  LogPrintf (LOG_DEBUG, "Freeing Doppler grid ... ");
  LAL_CALL ( FreeDopplerFullScan(&status, &GV.scanState), &status);
  LogPrintfVerbatim ( LOG_DEBUG, "done.\n");

  XLALDestroyREAL8FrequencySeries ( fstatVector );

  XLALEmptyComputeFBuffer ( &cfBuffer );

  LAL_CALL ( Freemem(&status, &GV), &status);

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
  uvar_upsampleSFTs = 1;
  uvar_Dterms 	= 16;
  uvar_FreqBand = 0.0;
  uvar_Alpha 	= 0.0;
  uvar_Delta 	= 0.0;
  uvar_AlphaBand = 0;
  uvar_DeltaBand = 0;
  uvar_skyRegion = NULL;

  uvar_ephemYear = LALCalloc (1, strlen(EPHEM_YEARS)+1);
  strcpy (uvar_ephemYear, EPHEM_YEARS);

#define DEFAULT_EPHEMDIR "env LAL_DATA_PATH"
  uvar_ephemDir = LALCalloc (1, strlen(DEFAULT_EPHEMDIR)+1);
  strcpy (uvar_ephemDir, DEFAULT_EPHEMDIR);

  uvar_SignalOnly = FALSE;
  uvar_UseNoiseWeights = TRUE;

  uvar_f1dot     = 0.0;
  uvar_f1dotBand = 0.0;

  /* default step-sizes for GRID_FLAT */
  uvar_dAlpha 	= 0.001;
  uvar_dDelta 	= 0.001;
  /*   uvar_dFreq 	 = 0.0;	 deactivated for resampling version */
  uvar_df1dot    = 0.0;
  uvar_df2dot    = 0.0;
  uvar_df3dot    = 0.0;


  uvar_TwoFthreshold = 10.0;
  uvar_NumCandidatesToKeep = 0;
  uvar_clusterOnScanline = 0;

  uvar_metricType =  LAL_PMETRIC_NONE;
  uvar_projectMetric = TRUE;
  uvar_gridType = GRID_FLAT;

  uvar_metricMismatch = 0.02;

  uvar_help = FALSE;
  uvar_outputLogfile = NULL;

  uvar_outputFstat = NULL;

  uvar_gridFile = NULL;

  uvar_dopplermax =  1.05e-4;
  uvar_RngMedWindow = 50;	/* for running-median */

  uvar_SSBprecision = SSBPREC_RELATIVISTIC;

  uvar_minStartTime = 0;
  uvar_maxEndTime = LAL_INT4_MAX;

  uvar_workingDir = (CHAR*)LALMalloc(512);
  strcpy(uvar_workingDir, ".");

  uvar_timerCount = 1e5;	/* output a timer/progress count every N templates */

  /* ---------- register all user-variables ---------- */
  LALregBOOLUserVar(status, 	help, 		'h', UVAR_HELP,     "Print this message");

  LALregREALUserVar(status, 	Alpha, 		'a', UVAR_OPTIONAL, "Sky position alpha (equatorial coordinates) in radians");
  LALregREALUserVar(status, 	Delta, 		'd', UVAR_OPTIONAL, "Sky position delta (equatorial coordinates) in radians");
  LALregREALUserVar(status, 	Freq, 		'f', UVAR_REQUIRED, "Starting search frequency in Hz");
  LALregREALUserVar(status, 	f1dot, 		's', UVAR_OPTIONAL, "First spindown parameter  dFreq/dt");
  LALregREALUserVar(status, 	f2dot, 		 0 , UVAR_OPTIONAL, "Second spindown parameter d^2Freq/dt^2");
  LALregREALUserVar(status, 	f3dot, 		 0 , UVAR_OPTIONAL, "Third spindown parameter  d^3Freq/dt^2");

  LALregREALUserVar(status, 	AlphaBand, 	'z', UVAR_OPTIONAL, "Band in alpha (equatorial coordinates) in radians");
  LALregREALUserVar(status, 	DeltaBand, 	'c', UVAR_OPTIONAL, "Band in delta (equatorial coordinates) in radians");
  LALregREALUserVar(status, 	FreqBand, 	'b', UVAR_OPTIONAL, "Search frequency band in Hz");
  LALregREALUserVar(status, 	f1dotBand, 	'm', UVAR_OPTIONAL, "Search-band for f1dot");
  LALregREALUserVar(status, 	f2dotBand, 	 0 , UVAR_OPTIONAL, "Search-band for f2dot");
  LALregREALUserVar(status, 	f3dotBand, 	 0 , UVAR_OPTIONAL, "Search-band for f3dot");

  LALregREALUserVar(status, 	dAlpha, 	'l', UVAR_OPTIONAL, "Resolution in alpha (equatorial coordinates) in radians");
  LALregREALUserVar(status, 	dDelta, 	'g', UVAR_OPTIONAL, "Resolution in delta (equatorial coordinates) in radians");
  /*   LALregREALUserVar(status,     dFreq,          'r', UVAR_OPTIONAL, "Frequency resolution in Hz [Default: 1/(2T)]");
       --> deactivated for resampling version
   */
  LALregREALUserVar(status, 	df1dot, 	'e', UVAR_OPTIONAL, "Stepsize for f1dot [Default: 1/(2T^2)");
  LALregREALUserVar(status, 	df2dot, 	 0 , UVAR_OPTIONAL, "Stepsize for f2dot [Default: 1/(2T^3)");
  LALregREALUserVar(status, 	df3dot, 	 0 , UVAR_OPTIONAL, "Stepsize for f3dot [Default: 1/(2T^4)");

  LALregSTRINGUserVar(status,	skyRegion, 	'R', UVAR_OPTIONAL, "ALTERNATIVE: Specify sky-region by polygon (or use 'allsky')");
  LALregSTRINGUserVar(status,	DataFiles, 	'D', UVAR_REQUIRED, "File-pattern specifying (multi-IFO) input SFT-files");
  LALregSTRINGUserVar(status, 	IFO, 		'I', UVAR_OPTIONAL, "Detector: 'G1', 'L1', 'H1', 'H2' ...(useful for single-IFO v1-SFTs only!)");
  LALregSTRINGUserVar(status,	ephemDir, 	'E', UVAR_OPTIONAL, "Directory where Ephemeris files are located");
  LALregSTRINGUserVar(status,	ephemYear, 	'y', UVAR_OPTIONAL, "Year (or range of years) of ephemeris files to be used");
  LALregBOOLUserVar(status, 	SignalOnly, 	'S', UVAR_OPTIONAL, "Signal only flag");
  LALregBOOLUserVar(status, 	UseNoiseWeights,'W', UVAR_OPTIONAL, "Use SFT-specific noise weights");

  LALregREALUserVar(status, 	TwoFthreshold,	'F', UVAR_OPTIONAL, "Set the threshold for selection of 2F");
  LALregINTUserVar(status, 	gridType,	 0 , UVAR_OPTIONAL, "Grid: 0=flat, 1=isotropic, 2=metric, 3=skygrid-file, 6=grid-file, 7=An*lattice");
  LALregINTUserVar(status, 	metricType,	'M', UVAR_OPTIONAL, "Metric: 0=none,1=Ptole-analytic,2=Ptole-numeric, 3=exact");
  LALregREALUserVar(status, 	metricMismatch,	'X', UVAR_OPTIONAL, "Maximal allowed mismatch for metric tiling");
  LALregSTRINGUserVar(status,	outputLogfile,	 0,  UVAR_OPTIONAL, "Name of log-file identifying the code + search performed");
  LALregSTRINGUserVar(status,	gridFile,	 0,  UVAR_OPTIONAL, "Load grid from this file: sky-grid or full-grid depending on --gridType.");
  LALregREALUserVar(status,	refTime,	 0,  UVAR_OPTIONAL, "SSB reference time for pulsar-paramters [Default: startTime]");
  LALregREALUserVar(status, 	dopplermax, 	'q', UVAR_OPTIONAL, "Maximum doppler shift expected");

  LALregSTRINGUserVar(status,	outputFstat,	 0,  UVAR_OPTIONAL, "Output-file for F-statistic field over the parameter-space");
  LALregSTRINGUserVar(status,   outputLoudest,	 0,  UVAR_OPTIONAL, "Loudest F-statistic candidate + estimated MLE amplitudes");

  LALregINTUserVar(status,      NumCandidatesToKeep,0, UVAR_OPTIONAL, "Number of Fstat 'candidates' to keep. (0 = All)");
  LALregINTUserVar(status,      clusterOnScanline, 0, UVAR_OPTIONAL, "Neighbors on each side for finding 1D local maxima on scanline");


  LALregINTUserVar ( status, 	minStartTime, 	 0,  UVAR_OPTIONAL, "Earliest SFT-timestamp to include");
  LALregINTUserVar ( status, 	maxEndTime, 	 0,  UVAR_OPTIONAL, "Latest SFT-timestamps to include");

  /* ----- more experimental/expert options ----- */
  LALregINTUserVar (status, 	SSBprecision,	 0,  UVAR_DEVELOPER, "Precision to use for time-transformation to SSB: 0=Newtonian 1=relativistic");
  LALregINTUserVar(status, 	RngMedWindow,	'k', UVAR_DEVELOPER, "Running-Median window size");
  LALregINTUserVar(status,	Dterms,		't', UVAR_DEVELOPER, "Number of terms to keep in Dirichlet kernel sum");

  LALregSTRINGUserVar(status,   workingDir,     'w', UVAR_DEVELOPER, "Directory to use as work directory.");
  LALregREALUserVar(status, 	timerCount, 	 0,  UVAR_DEVELOPER, "N: Output progress/timer info every N templates");
  LALregREALUserVar(status,	internalRefTime, 0,  UVAR_DEVELOPER, "internal reference time to use for Fstat-computation [Default: startTime]");

  LALregINTUserVar(status,	upsampleSFTs,	 0,  UVAR_DEVELOPER, "(integer) Factor to up-sample SFTs by");
  LALregBOOLUserVar(status, 	projectMetric, 	 0,  UVAR_DEVELOPER, "Use projected metric on Freq=const subspact");

  DETATCHSTATUSPTR (status);
  RETURN (status);
} /* initUserVars() */

/** Load Ephemeris from ephemeris data-files  */
void
InitEphemeris (LALStatus * status,
	       EphemerisData *edat,	/**< [out] the ephemeris-data */
	       const CHAR *ephemDir,	/**< directory containing ephems */
	       const CHAR *ephemYear,	/**< which years do we need? */
	       LIGOTimeGPS epoch,	/**< epoch of observation */
	       BOOLEAN isLISA		/**< hack this function for LISA ephemeris */
	       )
{
#define FNAME_LENGTH 1024
  CHAR EphemEarth[FNAME_LENGTH];	/* filename of earth-ephemeris data */
  CHAR EphemSun[FNAME_LENGTH];	/* filename of sun-ephemeris data */
  LALLeapSecFormatAndAcc formatAndAcc = {LALLEAPSEC_GPSUTC, LALLEAPSEC_LOOSE};
  INT4 leap;

  INITSTATUS( status, "InitEphemeris", rcsid );
  ATTATCHSTATUSPTR (status);

  ASSERT ( edat, status, COMPUTEFSTATISTIC_ENULL, COMPUTEFSTATISTIC_MSGENULL );
  ASSERT ( ephemYear, status, COMPUTEFSTATISTIC_ENULL, COMPUTEFSTATISTIC_MSGENULL );

  if ( ephemDir )
    {
      if ( isLISA )
	LALSnprintf(EphemEarth, FNAME_LENGTH, "%s/ephemMLDC.dat", ephemDir);
      else
	LALSnprintf(EphemEarth, FNAME_LENGTH, "%s/earth%s.dat", ephemDir, ephemYear);

      LALSnprintf(EphemSun, FNAME_LENGTH, "%s/sun%s.dat", ephemDir, ephemYear);
    }
  else
    {
      if ( isLISA )
	LALSnprintf(EphemEarth, FNAME_LENGTH, "ephemMLDC.dat");
      else
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



/** Initialized Fstat-code: handle user-input and set everything up.
 * NOTE: the logical *order* of things in here is very important, so be careful
 */
void
InitFStat ( LALStatus *status, ConfigVariables *cfg )
{
  REAL8 fCoverMin, fCoverMax;	/* covering frequency-band to read from SFTs */
  SFTCatalog *catalog = NULL;
  SFTConstraints constraints = empty_SFTConstraints;
  LIGOTimeGPS minStartTimeGPS = empty_LIGOTimeGPS;
  LIGOTimeGPS maxEndTimeGPS = empty_LIGOTimeGPS;
  PulsarSpinRange spinRangeRef = empty_PulsarSpinRange;

  UINT4 numSFTs;
  LIGOTimeGPS startTime, endTime;

  INITSTATUS (status, "InitFStat", rcsid);
  ATTATCHSTATUSPTR (status);

  /* set the current working directory */
  if(chdir(uvar_workingDir) != 0)
    {
      LogPrintf (LOG_CRITICAL,  "Unable to change directory to workinDir '%s'\n", uvar_workingDir);
      ABORT (status, COMPUTEFSTATC_EINPUT, COMPUTEFSTATC_MSGEINPUT);
    }

  /* use IFO-contraint if one given by the user */
  if ( LALUserVarWasSet ( &uvar_IFO ) )
    if ( (constraints.detector = XLALGetChannelPrefix ( uvar_IFO )) == NULL ) {
      ABORT ( status,  COMPUTEFSTATISTIC_EINPUT,  COMPUTEFSTATISTIC_MSGEINPUT);
    }
  minStartTimeGPS.gpsSeconds = uvar_minStartTime;
  maxEndTimeGPS.gpsSeconds = uvar_maxEndTime;
  constraints.startTime = &minStartTimeGPS;
  constraints.endTime = &maxEndTimeGPS;

  /* get full SFT-catalog of all matching (multi-IFO) SFTs */
  LogPrintf (LOG_DEBUG, "Finding all SFTs to load ... ");
  TRY ( LALSFTdataFind ( status->statusPtr, &catalog, uvar_DataFiles, &constraints ), status);
  LogPrintfVerbatim (LOG_DEBUG, "done. (found %d SFTs)\n", catalog->length);

  if ( constraints.detector )
    LALFree ( constraints.detector );

  if ( !catalog || catalog->length == 0 )
    {
      LALPrintError ("\nSorry, didn't find any matching SFTs with pattern '%s'!\n\n", uvar_DataFiles );
      ABORT ( status,  COMPUTEFSTATISTIC_EINPUT,  COMPUTEFSTATISTIC_MSGEINPUT);
    }

  /* deduce start- and end-time of the observation spanned by the data */
  numSFTs = catalog->length;
  cfg->Tsft = 1.0 / catalog->data[0].header.deltaF;
  startTime = catalog->data[0].header.epoch;
  endTime   = catalog->data[numSFTs-1].header.epoch;
  XLALAddFloatToGPS(&endTime, cfg->Tsft);	/* add on Tsft to last SFT start-time */

  /* ----- get reference-times (from user if given, use startTime otherwise): ----- */
  if ( LALUserVarWasSet(&uvar_refTime)) {
    TRY ( LALFloatToGPS (status->statusPtr, &(cfg->refTime), &uvar_refTime), status);
  }
  else
    cfg->refTime = startTime;

  { /* ----- prepare spin-range at refTime (in *canonical format*, ie all Bands >= 0) ----- */
    REAL8 fMin = MYMIN ( uvar_Freq, uvar_Freq + uvar_FreqBand );
    REAL8 fMax = MYMAX ( uvar_Freq, uvar_Freq + uvar_FreqBand );

    REAL8 f1dotMin = MYMIN ( uvar_f1dot, uvar_f1dot + uvar_f1dotBand );
    REAL8 f1dotMax = MYMAX ( uvar_f1dot, uvar_f1dot + uvar_f1dotBand );

    REAL8 f2dotMin = MYMIN ( uvar_f2dot, uvar_f2dot + uvar_f2dotBand );
    REAL8 f2dotMax = MYMAX ( uvar_f2dot, uvar_f2dot + uvar_f2dotBand );

    REAL8 f3dotMin = MYMIN ( uvar_f3dot, uvar_f3dot + uvar_f3dotBand );
    REAL8 f3dotMax = MYMAX ( uvar_f3dot, uvar_f3dot + uvar_f3dotBand );

    spinRangeRef.refTime = cfg->refTime;
    spinRangeRef.fkdot[0] = fMin;
    spinRangeRef.fkdot[1] = f1dotMin;
    spinRangeRef.fkdot[2] = f2dotMin;
    spinRangeRef.fkdot[3] = f3dotMin;

    spinRangeRef.fkdotBand[0] = fMax - fMin;
    spinRangeRef.fkdotBand[1] = f1dotMax - f1dotMin;
    spinRangeRef.fkdotBand[2] = f2dotMax - f2dotMin;
    spinRangeRef.fkdotBand[3] = f3dotMax - f3dotMin;
  } /* spin-range at refTime */

  { /* ----- What frequency-band do we need to read from the SFTs?
     * propagate spin-range from refTime to startTime and endTime of observation
     */
    PulsarSpinRange spinRangeStart, spinRangeEnd;	/* temporary only */
    REAL8 fmaxStart, fmaxEnd, fminStart, fminEnd;

    /* compute spin-range at startTime of observation */
    TRY ( LALExtrapolatePulsarSpinRange (status->statusPtr, &spinRangeStart, startTime, &spinRangeRef ), status );
    /* compute spin-range at endTime of these SFTs */
    TRY ( LALExtrapolatePulsarSpinRange (status->statusPtr, &spinRangeEnd, endTime, &spinRangeStart ), status );

    fminStart = spinRangeStart.fkdot[0];
    /* ranges are in canonical format! */
    fmaxStart = fminStart + spinRangeStart.fkdotBand[0];
    fminEnd   = spinRangeEnd.fkdot[0];
    fmaxEnd   = fminEnd + spinRangeEnd.fkdotBand[0];

    /*  get covering frequency-band  */
    fCoverMax = MYMAX ( fmaxStart, fmaxEnd );
    fCoverMin = MYMIN ( fminStart, fminEnd );

  } /* extrapolate spin-range */

  {/* ----- load the multi-IFO SFT-vectors ----- */
    UINT4 wings = MYMAX(uvar_Dterms, uvar_RngMedWindow/2 +1);	/* extra frequency-bins needed for rngmed, and Dterms */
    REAL8 fMax = (1.0 + uvar_dopplermax) * fCoverMax + wings / cfg->Tsft; /* correct for doppler-shift and wings */
    REAL8 fMin = (1.0 - uvar_dopplermax) * fCoverMin - wings / cfg->Tsft;

    LogPrintf (LOG_DEBUG, "Loading SFTs ... ");
    TRY ( LALLoadMultiSFTs ( status->statusPtr, &(cfg->multiSFTs), catalog, fMin, fMax ), status );
    LogPrintfVerbatim (LOG_DEBUG, "done.\n");
    TRY ( LALDestroySFTCatalog ( status->statusPtr, &catalog ), status );
  }
  { /* ----- load ephemeris-data ----- */
    CHAR *ephemDir;
    BOOLEAN isLISA = FALSE;

    cfg->ephemeris = LALCalloc(1, sizeof(EphemerisData));
    if ( LALUserVarWasSet ( &uvar_ephemDir ) )
      ephemDir = uvar_ephemDir;
    else
      ephemDir = NULL;

    /* hack: if first detector is LISA, we load MLDC-ephemeris instead of 'earth' files */
    if ( cfg->multiSFTs->data[0]->data[0].name[0] == 'Z' )
      isLISA = TRUE;

    TRY( InitEphemeris (status->statusPtr, cfg->ephemeris, ephemDir, uvar_ephemYear, startTime, isLISA ), status);
  }

  /* ----- obtain the (multi-IFO) 'detector-state series' for all SFTs ----- */
  TRY ( LALGetMultiDetectorStates ( status->statusPtr, &(cfg->multiDetStates), cfg->multiSFTs, cfg->ephemeris ), status );

  /* ----- normalize SFTs and calculate noise-weights ----- */
  if ( uvar_SignalOnly )
      cfg->multiNoiseWeights = NULL;   /* noiseWeights == NULL is equivalent to unit noise-weights in ComputeFstat() */
  else
    {
      UINT4 X, alpha;
      MultiPSDVector *rngmed = NULL;
      cfg->multiNoiseWeights = NULL;
      TRY ( LALNormalizeMultiSFTVect (status->statusPtr, &rngmed, cfg->multiSFTs, uvar_RngMedWindow ), status );
      TRY ( LALComputeMultiNoiseWeights  (status->statusPtr, &(cfg->multiNoiseWeights), rngmed, uvar_RngMedWindow, 0 ), status );
      TRY ( LALDestroyMultiPSDVector (status->statusPtr, &rngmed ), status );
      if ( !uvar_UseNoiseWeights )	/* in that case simply set weights to 1.0 */
	for ( X = 0; X < cfg->multiNoiseWeights->length; X ++ )
	  for ( alpha = 0; alpha < cfg->multiNoiseWeights->data[X]->length; alpha ++ )
	    cfg->multiNoiseWeights->data[X]->data[alpha] = 1.0;
    } /* if ! SignalOnly */

  /* ----- upsample SFTs ----- */
  if ( (lalDebugLevel >= 2) && (uvar_upsampleSFTs > 1) )
  {
    UINT4 X, numDet = cfg->multiSFTs->length;
    LogPrintf (LOG_DEBUG, "Writing original SFTs for debugging ... ");
    for (X=0; X < numDet ; X ++ )
      {
	TRY ( LALWriteSFTVector2Dir ( status->statusPtr, cfg->multiSFTs->data[X], "./", "original", "orig"), status );
      }
    LogPrintfVerbatim ( LOG_DEBUG, "done.\n");
  }

  LogPrintf (LOG_DEBUG, "Upsampling SFTs by factor %d ... ", uvar_upsampleSFTs );
  TRY ( upsampleMultiSFTVector ( status->statusPtr, cfg->multiSFTs, uvar_upsampleSFTs, 16 ), status );
  LogPrintfVerbatim (LOG_DEBUG, "done.\n");

  if ( lalDebugLevel >= 2 && (uvar_upsampleSFTs > 1) )
  {
    UINT4 X, numDet = cfg->multiSFTs->length;
    CHAR tag[60];
    sprintf (tag, "upsampled%02d", uvar_upsampleSFTs );
    LogPrintf (LOG_DEBUG, "Writing upsampled SFTs for debugging ... ");
    for (X=0; X < numDet ; X ++ )
      {
	TRY ( LALWriteSFTVector2Dir ( status->statusPtr, cfg->multiSFTs->data[X], "./", tag, tag), status );
      }
    LogPrintfVerbatim ( LOG_DEBUG, "done.\n");
  }


  { /* ----- set up Doppler region (at internalRefTime) to scan ----- */
    LIGOTimeGPS internalRefTime = empty_LIGOTimeGPS;
    PulsarSpinRange spinRangeInt = empty_PulsarSpinRange;
    BOOLEAN haveAlphaDelta = LALUserVarWasSet(&uvar_Alpha) && LALUserVarWasSet(&uvar_Delta);

    if (uvar_skyRegion)
      {
	cfg->searchRegion.skyRegionString = (CHAR*)LALCalloc(1, strlen(uvar_skyRegion)+1);
	if ( cfg->searchRegion.skyRegionString == NULL ) {
	  ABORT (status, COMPUTEFSTATC_EMEM, COMPUTEFSTATC_MSGEMEM);
	}
	strcpy (cfg->searchRegion.skyRegionString, uvar_skyRegion);
      }
    else if (haveAlphaDelta)    /* parse this into a sky-region */
      {
	TRY ( SkySquare2String( status->statusPtr, &(cfg->searchRegion.skyRegionString),
				uvar_Alpha, uvar_Delta,	uvar_AlphaBand, uvar_DeltaBand), status);
      }

    if ( LALUserVarWasSet ( &uvar_internalRefTime ) ) {
      TRY ( LALFloatToGPS (status->statusPtr, &(internalRefTime), &uvar_internalRefTime), status);
    }
    else
      internalRefTime = startTime;

    /* spin searchRegion defined by spin-range at *internal* reference-time */
    TRY ( LALExtrapolatePulsarSpinRange (status->statusPtr, &spinRangeInt, internalRefTime, &spinRangeRef ), status );
    cfg->searchRegion.refTime = spinRangeInt.refTime;
    memcpy ( &cfg->searchRegion.fkdot, &spinRangeInt.fkdot, sizeof(spinRangeInt.fkdot) );
    memcpy ( &cfg->searchRegion.fkdotBand, &spinRangeInt.fkdotBand, sizeof(spinRangeInt.fkdotBand) );

    /* special treatment of frequency band: take out of Doppler search-region for resampling technique */
    cfg->FFTFreqBand = cfg->searchRegion.fkdotBand[0];
    cfg->searchRegion.fkdotBand[0] = 0;		/* Doppler region contains no frequency-band */

  } /* get DopplerRegion */

  /* ----- set computational parameters for F-statistic from User-input ----- */
  cfg->CFparams.Dterms = uvar_Dterms;
  cfg->CFparams.SSBprec = uvar_SSBprecision;
  cfg->CFparams.upsampling = 1.0 * uvar_upsampleSFTs;

  /* ----- set fixed grid step-sizes from user-input for GRID_FLAT ----- */
  cfg->stepSizes.Alpha = uvar_dAlpha;
  cfg->stepSizes.Delta = uvar_dDelta;
  cfg->stepSizes.fkdot[0] = 0;  	/* set default stepsize to FFT spacing: 1/Tspan */
  cfg->stepSizes.fkdot[1] = uvar_df1dot;
  cfg->stepSizes.fkdot[2] = uvar_df2dot;
  cfg->stepSizes.fkdot[3] = uvar_df3dot;
  cfg->stepSizes.orbit = NULL;

  /* ----- set up toplist if requested ----- */
  if ( uvar_NumCandidatesToKeep > 0 )
    if ( create_toplist( &(cfg->FstatToplist), uvar_NumCandidatesToKeep, sizeof(FstatCandidate), compareFstatCandidates) != 0 ) {
      ABORT (status, COMPUTEFSTATISTIC_EMEM, COMPUTEFSTATISTIC_MSGEMEM );
    }

  /* ----- set up scanline-window if requested for 1D local-maximum clustering on scanline ----- */
  if ( (cfg->scanlineWindow = XLALCreateScanlineWindow ( uvar_clusterOnScanline )) == NULL ) {
    ABORT (status, COMPUTEFSTATISTIC_EMEM, COMPUTEFSTATISTIC_MSGEMEM );
  }

  /* initialize full multi-dimensional Doppler-scanner */
  {
    DopplerFullScanInit scanInit;			/* init-structure for DopperScanner */

    scanInit.searchRegion = cfg->searchRegion;
    scanInit.gridType = uvar_gridType;
    scanInit.gridFile = uvar_gridFile;
    scanInit.metricType = uvar_metricType;
    scanInit.projectMetric = uvar_projectMetric;
    scanInit.metricMismatch = uvar_metricMismatch;
    scanInit.stepSizes = cfg->stepSizes;
    scanInit.ephemeris = cfg->ephemeris;		/* used by Ephemeris-based metric */
    scanInit.startTime = cfg->multiDetStates->startTime;
    scanInit.Tspan     = cfg->multiDetStates->Tspan;
    scanInit.Detector  = &(cfg->multiDetStates->data[0]->detector);	/* just use first IFO for metric */

    LogPrintf (LOG_DEBUG, "Setting up template grid ... ");
    TRY ( InitDopplerFullScan ( status->statusPtr, &cfg->scanState, &scanInit), status);
    LogPrintf (LOG_DEBUG, "template grid ready: %.0f templates.\n", XLALNumDopplerTemplates ( cfg->scanState ) );
  }

  /* ----- produce a log-string describing the data-specific setup ----- */
  TRY ( getLogString ( status->statusPtr, &(cfg->logstring), cfg ), status );
  LogPrintfVerbatim( LOG_DEBUG, cfg->logstring );


  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* InitFStat() */

/** Produce a log-string describing the present run-setup
 */
void
getLogString ( LALStatus *status, CHAR **logstr, const ConfigVariables *cfg )
{
  struct tm utc;
  time_t tp;
  CHAR dateStr[512], line[512], summary[4096];
  CHAR *cmdline = NULL;
  UINT4 i, numDet, numSpins = PULSAR_MAX_SPINS;
  const CHAR *codeID = "$Id$";
  CHAR *ret = NULL;

  INITSTATUS( status, "getLogString", rcsid );
  ATTATCHSTATUSPTR (status);

  /* first get full commandline describing search*/
  TRY ( LALUserVarGetLog (status->statusPtr, &cmdline,  UVAR_LOGFMT_CMDLINE ), status );
  sprintf (summary, "%%%% %s\n%%%% %s\n", codeID, cmdline );
  LALFree ( cmdline );

  numDet = cfg->multiSFTs->length;
  tp = time(NULL);
  sprintf (line, "%%%% Started search: %s", asctime( gmtime( &tp ) ) );
  strcat ( summary, line );
  strcat (summary, "%% Loaded SFTs: [ " );
  for ( i=0; i < numDet; i ++ )
    {
      sprintf (line, "%s:%d%s",  cfg->multiSFTs->data[i]->data->name,
	       cfg->multiSFTs->data[i]->length,
	       (i < numDet - 1)?", ":" ]\n");
      strcat ( summary, line );
    }
  utc = *XLALGPSToUTC( &utc, (INT4)GPS2REAL8(cfg->multiDetStates->startTime) );
  strcpy ( dateStr, asctime(&utc) );
  dateStr[ strlen(dateStr) - 1 ] = 0;
  sprintf (line, "%%%% Start GPS time tStart = %12.3f    (%s GMT)\n",
	   GPS2REAL8(cfg->multiDetStates->startTime), dateStr);
  strcat ( summary, line );
  sprintf (line, "%%%% Total time spanned    = %12.3f s  (%.1f hours)\n",
	   cfg->multiDetStates->Tspan, cfg->multiDetStates->Tspan/3600 );
  strcat ( summary, line );
  sprintf (line, "%%%% Pulsar-params refTime = %12.3f \n", GPS2REAL8(cfg->refTime) );
  strcat ( summary, line );
  sprintf (line, "%%%% InternalRefTime       = %12.3f \n", GPS2REAL8(cfg->searchRegion.refTime) );
  strcat ( summary, line );
  sprintf (line, "%%%% Spin-range at internalRefTime: " );
  strcat ( summary, line );

  strcat (summary, "fkdot = [ " );
  for (i=0; i < numSpins; i ++ )
    {
      sprintf (line, "%.16g:%.16g%s",
	       cfg->searchRegion.fkdot[i],
	       cfg->searchRegion.fkdot[i] + cfg->searchRegion.fkdotBand[i],
	       (i < numSpins - 1)?", ":" ]\n");
      strcat ( summary, line );
    }

  if ( (ret = LALCalloc(1, strlen(summary) + 1 )) == NULL ) {
    ABORT (status, COMPUTEFSTATISTIC_EMEM, COMPUTEFSTATISTIC_MSGEMEM);
  }

  strcpy ( ret, summary );

  /* return result */
  (*logstr) = ret;

  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* getLogString() */



/***********************************************************************/
/** Log the all relevant parameters of the present search-run to a log-file.
 * The name of the log-file is log_fname
 * <em>NOTE:</em> Currently this function only logs the user-input and code-versions.
 */
void
WriteFStatLog (LALStatus *status, char *argv[], const CHAR *log_fname )
{
  CHAR *logstr = NULL;
  CHAR command[512] = "";
  FILE *fplog;

  INITSTATUS (status, "WriteFStatLog", rcsid);
  ATTATCHSTATUSPTR (status);

  if ( !log_fname )	/* no logfile given */
    return;

  /* prepare log-file for writing */
  if ( (fplog = fopen(log_fname, "wb" )) == NULL) {
    LogPrintf ( LOG_CRITICAL , "Failed to open log-file '%s' for writing.\n\n", log_fname );
    ABORT (status, COMPUTEFSTATISTIC_ESYS, COMPUTEFSTATISTIC_MSGESYS);
  }

  /* write out a log describing the complete user-input (in cfg-file format) */
  TRY (LALUserVarGetLog (status->statusPtr, &logstr,  UVAR_LOGFMT_CFGFILE), status);

  fprintf (fplog, "%%%% LOG-FILE of ComputeFStatistic run\n\n");
  fprintf (fplog, "%% User-input:\n");
  fprintf (fplog, "%%----------------------------------------------------------------------\n\n");

  fprintf (fplog, logstr);
  LALFree (logstr);

  /* append an ident-string defining the exact CVS-version of the code used */
  fprintf (fplog, "\n\n%% CVS-versions of executable:\n");
  fprintf (fplog, "%% ----------------------------------------------------------------------\n");
  fclose (fplog);

  sprintf (command, "ident %s 2> /dev/null | sort -u >> %s", argv[0], log_fname);
  system (command);	/* we don't check this. If it fails, we assume that */
    			/* one of the system-commands was not available, and */
    			/* therefore the CVS-versions will not be logged */

  DETATCHSTATUSPTR (status);
  RETURN(status);

} /* WriteFStatLog() */


/** Free all globally allocated memory. */
void
Freemem(LALStatus *status,  ConfigVariables *cfg)
{
  INITSTATUS (status, "Freemem", rcsid);
  ATTATCHSTATUSPTR (status);


  /* Free SFT data */
  TRY ( LALDestroyMultiSFTVector (status->statusPtr, &(cfg->multiSFTs) ), status );
  /* and corresponding noise-weights */
  TRY ( LALDestroyMultiNoiseWeights (status->statusPtr, &(cfg->multiNoiseWeights) ), status );

  /* destroy DetectorStateSeries */
  XLALDestroyMultiDetectorStateSeries ( cfg->multiDetStates );

  /* destroy FstatToplist if any */
  if ( cfg->FstatToplist )
    free_toplist( &(cfg->FstatToplist) );

  if ( cfg->scanlineWindow )
    XLALDestroyScanlineWindow ( cfg->scanlineWindow );

  /* Free config-Variables and userInput stuff */
  TRY (LALDestroyUserVars (status->statusPtr), status);

  if ( cfg->searchRegion.skyRegionString )
    LALFree ( cfg->searchRegion.skyRegionString );

  /* Free ephemeris data */
  LALFree(cfg->ephemeris->ephemE);
  LALFree(cfg->ephemeris->ephemS);
  LALFree(cfg->ephemeris);

  if ( cfg->logstring )
    LALFree ( cfg->logstring );

  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* Freemem() */


/*----------------------------------------------------------------------*/
/** Some general consistency-checks on user-input.
 * Throws an error plus prints error-message if problems are found.
 */
void
checkUserInputConsistency (LALStatus *status)
{

  INITSTATUS (status, "checkUserInputConsistency", rcsid);

  if (uvar_ephemYear == NULL)
    {
      LALPrintError ("\nNo ephemeris year specified (option 'ephemYear')\n\n");
      ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
    }

  /* check for negative stepsizes in Freq, Alpha, Delta */
  if ( LALUserVarWasSet(&uvar_dAlpha) && (uvar_dAlpha < 0) )
    {
      LALPrintError ("\nNegative value of stepsize dAlpha not allowed!\n\n");
      ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
    }
  if ( LALUserVarWasSet(&uvar_dDelta) && (uvar_dDelta < 0) )
    {
      LALPrintError ("\nNegative value of stepsize dDelta not allowed!\n\n");
      ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
    }
  /* deactivated in resampling version 
  if ( LALUserVarWasSet(&uvar_dFreq) && (uvar_dFreq < 0) )
    {
      LALPrintError ("\nNegative value of stepsize dFreq not allowed!\n\n");
      ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
    }
  */

  /* grid-related checks */
  {
    BOOLEAN haveAlphaBand = LALUserVarWasSet( &uvar_AlphaBand );
    BOOLEAN haveDeltaBand = LALUserVarWasSet( &uvar_DeltaBand );
    BOOLEAN haveSkyRegion, haveAlphaDelta, haveGridFile;
    BOOLEAN useSkyGridFile, useFullGridFile, haveMetric, useMetric;

    haveSkyRegion  	= (uvar_skyRegion != NULL);
    haveAlphaDelta 	= (LALUserVarWasSet(&uvar_Alpha) && LALUserVarWasSet(&uvar_Delta) );
    haveGridFile      	= (uvar_gridFile != NULL);
    useSkyGridFile   	= (uvar_gridType == GRID_FILE_SKYGRID);
    useFullGridFile	= (uvar_gridType == GRID_FILE_FULLGRID);
    haveMetric     	= (uvar_metricType > LAL_PMETRIC_NONE);
    useMetric     	= (uvar_gridType == GRID_METRIC);

    if ( !useFullGridFile && !useSkyGridFile && haveGridFile )
      {
        LALWarning (status, "\nWARNING: gridFile was specified but not needed ... will be ignored\n\n");
      }
    if ( useSkyGridFile && !haveGridFile )
      {
        LALPrintError ("\nERROR: gridType=SKY-FILE, but no --gridFile specified!\n\n");
        ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
      }
    if ( useFullGridFile && !haveGridFile )
      {
	LALPrintError ("\nERROR: gridType=GRID-FILE, but no --gridFile specified!\n\n");
        ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
      }

    if ( (haveAlphaBand && !haveDeltaBand) || (haveDeltaBand && !haveAlphaBand) )
      {
	LALPrintError ("\nERROR: Need either BOTH (AlphaBand, DeltaBand) or NONE.\n\n");
        ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
      }

    if ( haveSkyRegion && haveAlphaDelta )
      {
        LALPrintError ("\nOverdetermined sky-region: only use EITHER (Alpha,Delta) OR skyRegion!\n\n");
        ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
      }
    if ( !useMetric && haveMetric)
      {
        LALWarning (status, "\nWARNING: Metric was specified for non-metric grid... will be ignored!\n");
      }
    if ( useMetric && !haveMetric)
      {
        LALPrintError ("\nERROR: metric grid-type selected, but no metricType selected\n\n");
        ABORT (status, COMPUTEFSTATISTIC_EINPUT, COMPUTEFSTATISTIC_MSGEINPUT);
      }

  } /* Grid-related checks */

  RETURN (status);
} /* checkUserInputConsistency() */

/* debug-output a(t) and b(t) into given file.
 * return 0 = OK, -1 on error
 */
int
outputBeamTS( const CHAR *fname, const AMCoeffs *amcoe, const DetectorStateSeries *detStates )
{
  FILE *fp;
  UINT4 i, len;

  if ( !fname || !amcoe || !amcoe->a || !amcoe->b || !detStates)
    return -1;

  len = amcoe->a->length;
  if ( (len != amcoe->b->length) || ( len != detStates->length ) )
    return -1;

  if ( (fp = fopen(fname, "wb")) == NULL )
    return -1;

  for (i=0; i < len; i ++ )
    {
      INT4 ret;
      ret = fprintf (fp, "%9d %f %f %f \n",
		     detStates->data[i].tGPS.gpsSeconds, detStates->data[i].LMST, amcoe->a->data[i], amcoe->b->data[i] );
      if ( ret < 0 )
	{
	  fprintf (fp, "ERROR\n");
	  fclose(fp);
	  return -1;
	}
    }

  fclose(fp);
  return 0;
} /* outputBeamTS() */

/*
============
va ['stolen' from Quake2 (GPL'ed)]

does a varargs printf into a temp buffer, so I don't need to have
varargs versions of all text functions.
FIXME: make this buffer size safe someday
============
*/
const char *va(const char *format, ...)
{
        va_list         argptr;
        static char     string[1024];

        va_start (argptr, format);
        vsprintf (string, format,argptr);
        va_end (argptr);

        return string;
}

/** write full 'PulsarCandidate' (i.e. Doppler params + Amplitude params + error-bars + Fa,Fb, F, + A,B,C,D
 * RETURN 0 = OK, -1 = ERROR
 */
int
write_PulsarCandidate_to_fp ( FILE *fp,  const PulsarCandidate *pulsarParams, const FstatCandidate *Fcand )
{
  if ( !fp || !pulsarParams || !Fcand  )
    return -1;

  fprintf (fp, "\n");

  fprintf (fp, "refTime  = % 9d;\n", pulsarParams->Doppler.refTime.gpsSeconds );   /* forget about ns... */

  fprintf (fp, "\n");

  /* Amplitude parameters with error-estimates */
  fprintf (fp, "h0       = % .6g;\n", pulsarParams->Amp.h0 );
  fprintf (fp, "dh0      = % .6g;\n", pulsarParams->dAmp.h0 );
  fprintf (fp, "cosi     = % .6g;\n", pulsarParams->Amp.cosi );
  fprintf (fp, "dcosi    = % .6g;\n", pulsarParams->dAmp.cosi );
  fprintf (fp, "phi0     = % .6g;\n", pulsarParams->Amp.phi0 );
  fprintf (fp, "dphi0    = % .6g;\n", pulsarParams->dAmp.phi0 );
  fprintf (fp, "psi      = % .6g;\n", pulsarParams->Amp.psi );
  fprintf (fp, "dpsi     = % .6g;\n", pulsarParams->dAmp.psi );

  fprintf (fp, "\n");

  /* Doppler parameters */
  fprintf (fp, "Alpha    = % .16g;\n", pulsarParams->Doppler.Alpha );
  fprintf (fp, "Delta    = % .16g;\n", pulsarParams->Doppler.Delta );
  fprintf (fp, "Freq     = % .16g;\n", pulsarParams->Doppler.fkdot[0] );
  fprintf (fp, "f1dot    = % .16g;\n", pulsarParams->Doppler.fkdot[1] );
  fprintf (fp, "f2dot    = % .16g;\n", pulsarParams->Doppler.fkdot[2] );
  fprintf (fp, "f3dot    = % .16g;\n", pulsarParams->Doppler.fkdot[3] );

  fprintf (fp, "\n");

  /* Amplitude Modulation Coefficients */
  fprintf (fp, "Ad       = % .6g;\n", Fcand->Mmunu.Ad );
  fprintf (fp, "Bd       = % .6g;\n", Fcand->Mmunu.Bd );
  fprintf (fp, "Cd       = % .6g;\n", Fcand->Mmunu.Cd );
  fprintf (fp, "Sinv_Tsft= % .6g;\n", Fcand->Mmunu.Sinv_Tsft );
  fprintf (fp, "\n");

  /* Fstat-values */
  fprintf (fp, "Fa       = % .6g  %+.6gi;\n", Fcand->Fstat.Fa.re, Fcand->Fstat.Fa.im );
  fprintf (fp, "Fb       = % .6g  %+.6gi;\n", Fcand->Fstat.Fb.re, Fcand->Fstat.Fb.im );
  fprintf (fp, "twoF     = % .6g;\n", 2.0 * Fcand->Fstat.F );

  fprintf (fp, "\nAmpFisher = \\\n" );
  XLALfprintfGSLmatrix ( fp, "%.9g",pulsarParams->AmpFisherMatrix );

  return 0;

} /* write_PulsarCandidate_to_fp() */

/** comparison function for our candidates toplist */
int
compareFstatCandidates ( const void *candA, const void *candB )
{
  if ( ((const FstatCandidate *)candA)->Fstat.F < ((const FstatCandidate *)candB)->Fstat.F )
    return 1;
  else
    return -1;

} /* compareFstatCandidates() */

/** write one 'FstatCandidate' (i.e. only Doppler-params + Fstat) into file 'fp'.
 * Return: 0 = OK, -1 = ERROR
 */
int
write_FstatCandidate_to_fp ( FILE *fp, const FstatCandidate *thisFCand )
{

  if ( !fp || !thisFCand )
    return -1;

  fprintf (fp, "%.16g %.16g %.16g %.6g %.5g %.5g %.9g\n",
	   thisFCand->doppler.fkdot[0], thisFCand->doppler.Alpha, thisFCand->doppler.Delta,
	   thisFCand->doppler.fkdot[1], thisFCand->doppler.fkdot[2], thisFCand->doppler.fkdot[3],
	   2.0 * thisFCand->Fstat.F );

  return 0;

} /* write_candidate_to_fp() */

/* --------------------------------------------------------------------------------
 * Scanline window functions
 * FIXME: should go into a separate file once implementation is settled down ...
 *
 * --------------------------------------------------------------------------------*/

/** Create a scanline window, with given windowWings >= 0.
 * Note: the actual window-size is 1 + 2 * windowWings
 */
scanlineWindow_t *
XLALCreateScanlineWindow ( UINT4 windowWings ) /**< number of neighbors on each side in scanlineWindow */
{
  const CHAR *fn = "XLALCreateScanlineWindow()";
  scanlineWindow_t *ret = NULL;
  UINT4 windowLen = 1 + 2 * windowWings;

  if ( ( ret = LALCalloc ( 1, sizeof(*ret)) ) == NULL ) {
    XLAL_ERROR_NULL( fn, COMPUTEFSTATISTIC_EMEM );
  }

  ret->length = windowLen;

  if ( (ret->window = LALCalloc ( windowLen, sizeof( ret->window[0] ) )) == NULL ) {
    LALFree ( ret );
    XLAL_ERROR_NULL( fn, COMPUTEFSTATISTIC_EMEM );
  }

  ret->center = &(ret->window[ windowWings ]);	/* points to central bin */

  return ret;

} /* XLALCreateScanlineWindow() */

void
XLALDestroyScanlineWindow ( scanlineWindow_t *scanlineWindow )
{
  if ( !scanlineWindow )
    return;

  if ( scanlineWindow->window )
    LALFree ( scanlineWindow->window );

  LALFree ( scanlineWindow );

  return;

} /* XLALDestroyScanlineWindow() */

/** Advance by pushing a new candidate into the scanline-window
 */
int
XLALAdvanceScanlineWindow ( const FstatCandidate *nextCand, scanlineWindow_t *scanWindow )
{
  const CHAR *fn = "XLALAdvanceScanlineWindow()";
  UINT4 i;

  if ( !nextCand || !scanWindow || !scanWindow->window ) {
    XLAL_ERROR (fn, XLAL_EINVAL );
  }

  for ( i=1; i < scanWindow->length; i ++ )
    scanWindow->window[i - 1] = scanWindow->window[i];

  scanWindow->window[ scanWindow->length - 1 ] = *nextCand;	/* struct-copy */

  return XLAL_SUCCESS;

} /* XLALAdvanceScanlineWindow() */

/** check wether central candidate in Scanline-window is a local maximum
 */
BOOLEAN
XLALCenterIsLocalMax ( const scanlineWindow_t *scanWindow )
{
  UINT4 i;
  REAL8 F0;

  if ( !scanWindow || !scanWindow->center )
    return FALSE;

  F0 = scanWindow->center->Fstat.F;

  for ( i=0; i < scanWindow->length; i ++ )
    if ( scanWindow->window[i].Fstat.F > F0 )
      return FALSE;

  return TRUE;

} /* XLALCenterIsLocalMax() */

/**
 * \file getMesh.c
 *
 * \author{Reinhard Prix}
 *
 * Standalone code to produce a 'mesh' in the parameter-space 
 * Currently this is just some code to use InitDopplerScan()
 * directly outside of ComputeFStatistic, and can't do much more
 * than generating sky-grids. 
 *
 * UserInput-parameters a compatible with ComputeFStatistic.
 *
 * Revision: $Id$
 *           
 *-----------------------------------------------------------------------*/

/* ---------- includes ---------- */
#include <math.h>

#include <lalapps.h>

#include <lal/UserInput.h>
#include <lal/LALConstants.h>
#include <lal/Date.h>
#include <lal/LALInitBarycenter.h>
#include <lal/AVFactories.h>
#include <lal/SFTutils.h>

#include "DopplerScan.h"

RCSID ("$Id$");

/* Error codes and messages */
#define GETMESH_ENORM 	0
#define GETMESH_ESUB  	1
#define GETMESH_EARG  	2
#define GETMESH_EBAD  	3
#define GETMESH_EFILE 	4
#define GETMESH_ENOARG 	5
#define GETMESH_EINPUT 	6
#define GETMESH_EMEM 	7
#define GETMESH_ENULL 	8

#define GETMESH_MSGENORM 	"Normal exit"
#define GETMESH_MSGESUB  	"Subroutine failed"
#define GETMESH_MSGEARG  	"Error parsing arguments"
#define GETMESH_MSGEBAD  	"Bad argument values"
#define GETMESH_MSGEFILE 	"File IO error"
#define GETMESH_MSGENOARG 	"Missing argument"
#define GETMESH_MSGEINPUT 	"Invalid user input"
#define GETMESH_MSGEMEM 	"Out of memory"
#define GETMESH_MSGENULL 	"Illegal NULL pointer in input"

/*---------- local defines ---------- */
#define TRUE (1==1)
#define FALSE (1==0)

/* ---------- some local types ---------- */
typedef struct {
  CHAR EphemEarth[512];		/**< filename of earth-ephemeris data */
  CHAR EphemSun[512];		/**< filename of sun-ephemeris data */
  LALDetector *Detector;        /**< detector of data to be searched */
  EphemerisData *ephemeris;	/**< ephemeris data (from LALInitBarycenter()) */
  LIGOTimeGPS startTimeGPS;	/**< starttime of observation */
  REAL8 duration;		/**< total observation-time spanned */
  DopplerRegion searchRegion;	/**< Doppler parameter-space to search */
} ConfigVariables;

/*---------- empty structs for initializations ----------*/
static const ConfigVariables empty_ConfigVariables;
static const PtoleMetricIn empty_metricpar;

/* ---------- local prototypes ---------- */
void initUserVars (LALStatus *);
void initGeneral (LALStatus *, ConfigVariables *cfg);
void checkUserInputConsistency (LALStatus *);
void getSearchRegion (LALStatus *, DopplerRegion *searchRegion, const DopplerSkyScanInit *params);
void setTrueRandomSeed(void);

/* ---------- User variables ---------- */
BOOLEAN uvar_help;

CHAR* uvar_IFO;			/**< name of detector (LHO, LLO, GEO, VIRGO,..)*/
REAL8 uvar_Alpha;		/**< skyposition Alpha: radians, equatorial coords. */
REAL8 uvar_Delta;		/**< skyposition Delta: radians, equatorial coords. */
REAL8 uvar_Freq;		/**< target-frequency */
REAL8 uvar_f1dot;		/**< target 1. spindown-value df/dt */
CHAR *uvar_ephemDir;		/**< directory of ephemeris-files */
CHAR *uvar_ephemYear;		/**< year-range of ephemeris-file to use */
INT4  uvar_metricType;		/**< Metric function to use: ptole_analytic, ptole-numeric, ... */
REAL8 uvar_metricMismatch;	/**< metric mismatch */
REAL8 uvar_startTime;		/**< GPS start time of observation */
REAL8 uvar_endTime;		/**< GPS end time of observation */
REAL8 uvar_duration;		/**< OR alternatively: length of observation in seconds */
BOOLEAN uvar_projectMetric;		/**< project out frequency-component of metric */

REAL8 uvar_dAlpha;		/**< Alpha-resolution if GRID_FLAT or GRID_ISOTROPIC */
REAL8 uvar_dDelta;		/**< Delta-resolution if ... */
REAL8 uvar_AlphaBand;		/**< Sky-region interval in Alpha */
REAL8 uvar_DeltaBand;		/**< Sky-region interval in Delta */
REAL8 uvar_FreqBand;		/**< Frequency-band */
REAL8 uvar_f1dotBand;		/**< spindown-band for f1dot */
REAL8 uvar_dFreq;
REAL8 uvar_df1dot;
INT4  uvar_gridType;		/**< GRID_FLAT, GRID_ISOTROPIC, GRID_METRIC or GRID_FILE */
INT4  uvar_metricType;		/**< if GRID_METRIC: what type of metric to use? */
REAL8 uvar_metricMismatch;	/**< what's the maximal mismatch of the grid? */
CHAR *uvar_skyRegion;		/**< alternative input: polygon describing skyregion */

CHAR *uvar_skyGridFile;		/**< read in sky-grid if GRID_FILE */
CHAR *uvar_outputSkyGrid;	/**< write sky-grid into this file */

INT4 uvar_searchNeighbors;	/**< number of desired gridpoints/dimension around central point*/
INT4 uvar_randomSeed;		/**< random-seed for searchNeighbors grid randomization */

CHAR *uvar_mergedSFTFile;	/**< dummy for CFS-compatibility */

extern int vrbflg;

/*----------------------------------------------------------------------
 * main function 
 *----------------------------------------------------------------------*/
int
main(int argc, char *argv[]) 
{
  LALStatus status = blank_status;
  ConfigVariables config = empty_ConfigVariables;
  DopplerSkyScanInit scanInit = empty_DopplerSkyScanInit; /* init-structure for DopperScanner */
  DopplerSkyScanState thisScan = empty_DopplerSkyScanState; /* current state of the Doppler-scan */
  UINT4 nFreq, nf1dot;

  lalDebugLevel = 0;  
  vrbflg = 1;	/* verbose error-messages */

  /* set LAL error-handler */
  lal_errhandler = LAL_ERR_EXIT;

  /* register user-variables */
  LAL_CALL (LALGetDebugLevel (&status, argc, argv, 'v'), &status);
  LAL_CALL (initUserVars (&status), &status);	  

  /* read cmdline & cfgfile  */	
  LAL_CALL (LALUserVarReadAllInput (&status, argc,argv), &status);  

  if (uvar_help) 	/* help requested: we're done */
    exit (0);

  /* check if user-input acutally made sense... */
  LAL_CALL ( checkUserInputConsistency(&status), &status);

  /* do some general setup of the code */
  LAL_CALL (initGeneral (&status, &config), &status);


  /* ---------- main-task: run InitDopplerSkyScan() ---------- */
  if (lalDebugLevel) printf ("\nSetting up template grid ...");
  scanInit.metricType = (LALPulsarMetricType) uvar_metricType;
  scanInit.dAlpha = uvar_dAlpha;
  scanInit.dDelta = uvar_dDelta;
  scanInit.gridType = (DopplerGridType) uvar_gridType;
  scanInit.metricMismatch = uvar_metricMismatch;
  scanInit.obsBegin = config.startTimeGPS;
  scanInit.obsDuration = config.duration;
  scanInit.projectMetric = uvar_projectMetric;
  scanInit.Detector = config.Detector;
  scanInit.ephemeris = config.ephemeris;       /* used by Ephemeris-based metric */
  scanInit.skyGridFile = uvar_skyGridFile;      /* if applicable */
  
  /* figure out searchRegion from UserInput and possibly --searchNeighbors */
  config.searchRegion = empty_DopplerRegion;	/* set to empty first */
  LAL_CALL ( getSearchRegion(&status, &(config.searchRegion), &scanInit ), &status);
  scanInit.skyRegionString = config.searchRegion.skyRegionString;
  scanInit.Freq = config.searchRegion.fkdot[0] + config.searchRegion.fkdotBand[0];

  /* the following call generates a skygrid plus determines dFreq, df1dot,..
   * Currently the complete template-grid is more like a 'foliation' of
   * the parameter-space by skygrids, stacked along f and f1dot, 
   * not a full 4D metric template-grid
   */
  LAL_CALL ( InitDopplerSkyScan( &status, &thisScan, &scanInit), &status); 

  /* ---------- overload Frequency- and spindown-resolution if input by user ----------*/
  if ( LALUserVarWasSet( &uvar_dFreq ) )
    thisScan.dfkdot[0] = uvar_dFreq;

  if( LALUserVarWasSet( &uvar_df1dot) ) 
    thisScan.dfkdot[1] = uvar_df1dot;

  /* we write the sky-grid to disk? */
  if ( uvar_outputSkyGrid ) 
    {
      printf ("\nNow writing sky-grid into file '%s' ...", uvar_outputSkyGrid);
      LAL_CALL(writeSkyGridFile(&status, thisScan.skyGrid, uvar_outputSkyGrid ), &status);
      printf (" done.\n\n");
    }
  

  /* ----- output grid-info ----- */
  nFreq = (UINT4)(config.searchRegion.fkdotBand[0] / thisScan.dfkdot[0] + 1e-6) + 1;  
  nf1dot =(UINT4)(config.searchRegion.fkdotBand[1]/ thisScan.dfkdot[1] + 1e-6) + 1;  

  /* debug output */
  if ( lalDebugLevel )
    {
      printf ("DEBUG: Search-region:\n");
      printf ("       skyRegion = \"%s\"\n", scanInit.skyRegionString);
      printf ("       Freq in  = [%.16g, %.16g]\n", 
	      config.searchRegion.fkdot[0], 
	      config.searchRegion.fkdot[0] + config.searchRegion.fkdotBand[0]);
      printf ("       f1dot in = [%.16g, %.16g]\n",
	      config.searchRegion.fkdot[1], 
	      config.searchRegion.fkdot[1] + config.searchRegion.fkdotBand[1]);

      printf ("\nDEBUG: actual grid-spacings: dFreq = %g, df1dot = %g\n\n",
	      thisScan.dfkdot[0], thisScan.dfkdot[1]);

      printf ("Templates: sky x Freq x f1dot = %d x %d x %d\n\n",
	      thisScan.numSkyGridPoints, nFreq, nf1dot );
    } /* debug-output */


  /* "official output": if NOT --searchNeighbors, just output the \
   * Total Number of templates:
   */
  if ( ! LALUserVarWasSet (&uvar_searchNeighbors) )
    {
      printf ("\n%g\n\n", (REAL8)thisScan.numSkyGridPoints * nFreq *  nf1dot );
    }
  /* output octave-compatible 'search-range' as a matrix: 
   * [ Alpha, AlphaBand; Delta, DeltaBand; Freq, FreqBand; f1dot, f1dotBand ]
   */
  else 
    {
      SkyRegion skyregion;
      DopplerRegion *searchRegion = &(config.searchRegion);
      REAL8 Alpha, AlphaBand, Delta, DeltaBand;
      REAL8 Freq, FreqBand, f1dot, f1dotBand;

      /* we need to parse the skyRegion string into a SkyRegion, which is easier to handle */
      LAL_CALL (ParseSkyRegionString(&status, &skyregion, searchRegion->skyRegionString), &status);
      if ( skyregion.vertices) 
	LALFree ( skyregion.vertices );

      Alpha = skyregion.lowerLeft.longitude; 
      AlphaBand = skyregion.upperRight.longitude - Alpha;
      Delta = skyregion.lowerLeft.latitude;
      DeltaBand = skyregion.upperRight.latitude - Delta;
      
      Freq = searchRegion->fkdot[0];
      FreqBand = searchRegion->fkdotBand[0];
      f1dot = searchRegion->fkdot[1];
      f1dotBand = searchRegion->fkdotBand[1];
      
      /* now write octave-compatible matrix containing the search-region */
      printf ("[%.12g, %.12g; %.12g, %.12g; %.12g, %.12g; %.12g, %.12g ]\n", 
	      Alpha, AlphaBand,
	      Delta, DeltaBand,
	      Freq, FreqBand,
	      f1dot, f1dotBand );
      
    } /* if uvar_searchNeighbors */

  /* ----- clean up and exit ----- */
  /* Free DopplerSkyScan-stuff (grid) */
  thisScan.state = STATE_FINISHED;
  LAL_CALL ( FreeDopplerSkyScan(&status, &thisScan), &status);

  /* Free User-variables and contents */
  LAL_CALL ( LALDestroyUserVars (&status), &status);

  if ( config.searchRegion.skyRegionString )
    LALFree ( config.searchRegion.skyRegionString );

  LALFree ( config.Detector );

  LALCheckMemoryLeaks(); 

  return 0;
} /* main */


/*----------------------------------------------------------------------*/
/** register all our "user-variables" */
void
initUserVars (LALStatus *stat)
{
  INITSTATUS( stat, "initUserVars", rcsid );
  ATTATCHSTATUSPTR (stat);

  /* set a few defaults */
#define EPHEM_YEARS  "00-04"
  uvar_ephemYear = (CHAR*)LALCalloc (1, strlen(EPHEM_YEARS)+1);
  strcpy (uvar_ephemYear, EPHEM_YEARS);

#define DEFAULT_EPHEMDIR "env LAL_DATA_PATH"
  uvar_ephemDir = (CHAR*)LALCalloc (1, strlen(DEFAULT_EPHEMDIR)+1);
  strcpy (uvar_ephemDir, DEFAULT_EPHEMDIR);

  uvar_f1dot = 0.0;
  uvar_metricType = LAL_PMETRIC_COH_PTOLE_ANALYTIC;
  uvar_metricMismatch = 0.02;
  uvar_help = FALSE;

  uvar_startTime = 714180733;
  uvar_endTime = 0;

  uvar_projectMetric = TRUE;

  uvar_dAlpha = 0.001;
  uvar_dDelta = 0.001;

  uvar_AlphaBand = 0;
  uvar_DeltaBand = 0;
  uvar_FreqBand = 0;
  uvar_f1dotBand = 0;

  uvar_gridType = GRID_FLAT;
  uvar_metricType = LAL_PMETRIC_COH_PTOLE_ANALYTIC;

  uvar_metricMismatch = 0.02;
  uvar_skyRegion = NULL;
  uvar_skyGridFile = NULL;
  uvar_outputSkyGrid = NULL;

  uvar_searchNeighbors = 0;

  /* now register all our user-variable */

  LALregBOOLUserVar(stat,	help,		'h', UVAR_HELP,     
		    "Print this help/usage message");
  LALregSTRINGUserVar(stat,	IFO,		'I', UVAR_REQUIRED, 
		      "Detector: H1, H2, L1, G1, ... ");

  LALregREALUserVar(stat,	Alpha,		'a', UVAR_OPTIONAL,
		    "skyposition Alpha in radians, equatorial coords.");
  LALregREALUserVar(stat,	Delta, 		'd', UVAR_OPTIONAL,
		    "skyposition Delta in radians, equatorial coords.");

  LALregREALUserVar(stat,       AlphaBand,      'z', UVAR_OPTIONAL, "Band in alpha (equatorial coordinates) in radians");
  LALregREALUserVar(stat,       DeltaBand,      'c', UVAR_OPTIONAL, "Band in delta (equatorial coordinates) in radians");
  LALregSTRINGUserVar(stat,     skyRegion,      'R', UVAR_OPTIONAL, "ALTERNATIVE: specify sky-region by polygon. Format: \"(a1,d1), (a2,d2),..(aN,dN)\"");
  LALregREALUserVar(stat,	Freq, 		'f', UVAR_REQUIRED, 
		    "target frequency");
  LALregREALUserVar(stat,	f1dot, 		's', UVAR_OPTIONAL, 
		    "first spindown-value df/dt");
  LALregREALUserVar(stat,       FreqBand,       'b', UVAR_OPTIONAL, "Search frequency band in Hz");
  LALregREALUserVar(stat,       f1dotBand,      'm', UVAR_OPTIONAL, "Search-band for f1dot");

  LALregREALUserVar(stat,       dFreq,          'r', UVAR_OPTIONAL, "Frequency resolution in Hz (default: 1/(2*Tsft*Nsft)");
  LALregREALUserVar(stat,       df1dot,         'e', UVAR_OPTIONAL, "Resolution for f1dot (default: use metric or 1/(2*T^2))");

  LALregINTUserVar(stat,        metricType,     'M', UVAR_OPTIONAL, 
		   "Metric: 0=none,1=Ptole-analytic,2=Ptole-numeric, 3=exact");
  LALregBOOLUserVar(stat,	projectMetric,	 0,  UVAR_OPTIONAL,
		    "Project metric onto frequency-surface");
  LALregREALUserVar(stat,       startTime,      't', UVAR_OPTIONAL, 
		    "GPS start time of observation");
  LALregREALUserVar(stat,       endTime,      't',   UVAR_OPTIONAL, 
		    "GPS end time of observation");
  LALregREALUserVar(stat,	duration,	'T', UVAR_OPTIONAL,
		    "Alternative: Duration of observation in seconds");

  LALregSTRINGUserVar(stat,     ephemDir,       'E', UVAR_OPTIONAL, 
		      "Directory where Ephemeris files are located");
  LALregSTRINGUserVar(stat,     ephemYear,      'y', UVAR_OPTIONAL, 
		      "Year (or range of years) of ephemeris files to be used");

  /* ---------- */
  LALregINTUserVar(stat,        gridType,        0 , UVAR_OPTIONAL, "Template SKY-grid: 0=flat, 1=isotropic, 2=metric, 3=file, 4=SkyGridFILE+metric");
  LALregINTUserVar(stat,        metricType,     'M', UVAR_OPTIONAL, "Metric: 0=none,1=Ptole-analytic,2=Ptole-numeric, 3=exact");
  LALregREALUserVar(stat,       metricMismatch, 'X', UVAR_OPTIONAL, "Maximal mismatch for SKY-grid (adjust value for more dimensions)");
  LALregREALUserVar(stat,       dAlpha,         'l', UVAR_OPTIONAL, "Resolution in alpha (equatorial coordinates) in radians");
  LALregREALUserVar(stat,       dDelta,         'g', UVAR_OPTIONAL, "Resolution in delta (equatorial coordinates) in radians");
  LALregSTRINGUserVar(stat,     skyGridFile,     0,  UVAR_OPTIONAL, "Load sky-grid from this file.");
  LALregSTRINGUserVar(stat,     outputSkyGrid,   0,  UVAR_OPTIONAL, "Write sky-grid into this file.");
  LALregINTUserVar(stat, 	searchNeighbors, 0,  UVAR_OPTIONAL, "Determine search-params with resulting in a grid with roughly this many points/dimension");
  LALregINTUserVar(stat, 	randomSeed, 	 0,  UVAR_OPTIONAL, "Random-seed to use for searchNeighbors grid-randomization");

  LALregSTRINGUserVar(stat, 	mergedSFTFile, 	 0,  UVAR_DEVELOPER, "Dummy for CFS-compatibility ");

  DETATCHSTATUSPTR (stat);
  RETURN (stat);

} /* initUserVars() */

/*----------------------------------------------------------------------*/
/** do some general initializations, 
 * e.g. load ephemeris-files (if required), setup detector etc
 */
void
initGeneral (LALStatus *status, ConfigVariables *cfg)
{

  INITSTATUS( status, "initGeneral", rcsid );
  ATTATCHSTATUSPTR (status);

  /* ----- set up Tspan */
  TRY ( LALFloatToGPS (status->statusPtr, &(cfg->startTimeGPS), &uvar_startTime), status);

  if ( LALUserVarWasSet (&uvar_endTime) )
    cfg->duration = uvar_endTime - uvar_startTime;
  else
    cfg->duration = uvar_duration;
  

  /* ---------- init ephemeris if needed ---------- */
  if ( uvar_metricType ==  LAL_PMETRIC_COH_EPHEM )
    {
      LALLeapSecFormatAndAcc formatAndAcc = {LALLEAPSEC_GPSUTC, LALLEAPSEC_STRICT};
      INT4 leap;

      if (LALUserVarWasSet (&uvar_ephemDir) )
	{
	  sprintf(cfg->EphemEarth, "%s/earth%s.dat", uvar_ephemDir, uvar_ephemYear);
	  sprintf(cfg->EphemSun, "%s/sun%s.dat", uvar_ephemDir, uvar_ephemYear);
	}
      else
	{
	  sprintf(cfg->EphemEarth, "earth%s.dat", uvar_ephemYear);
	  sprintf(cfg->EphemSun, "sun%s.dat", uvar_ephemYear);
	}

      cfg->ephemeris = (EphemerisData*) LALCalloc( 1, sizeof(EphemerisData) );
      cfg->ephemeris->ephiles.earthEphemeris = cfg->EphemEarth;
      cfg->ephemeris->ephiles.sunEphemeris = cfg->EphemSun;

      TRY (LALLeapSecs(status->statusPtr, &leap, &(cfg->startTimeGPS), &formatAndAcc), status);
      cfg->ephemeris->leap = leap;

      TRY (LALInitBarycenter (status->statusPtr, cfg->ephemeris), status);

  } /* end: init ephemeris data */


  /* ---------- initialize detector ---------- */
  if ( (cfg->Detector = XLALGetSiteInfo ( uvar_IFO )) == NULL ) {
    ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
  }

  DETATCHSTATUSPTR(status);
  RETURN(status);

} /* initGeneral() */

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
      ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
    }      
  /* don't allow negative bands (for safty in griding-routines) */
  if ( (uvar_AlphaBand < 0) ||  (uvar_DeltaBand < 0) )
    {
      LALPrintError ("\nNegative value of sky-bands not allowed (alpha or delta)!\n\n");
      ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
    }
  /* check for negative stepsizes in Freq, Alpha, Delta */
  if ( LALUserVarWasSet(&uvar_dAlpha) && (uvar_dAlpha < 0) )
    {
      LALPrintError ("\nNegative value of stepsize dAlpha not allowed!\n\n");
      ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
    }
  if ( LALUserVarWasSet(&uvar_dDelta) && (uvar_dDelta < 0) )
    {
      LALPrintError ("\nNegative value of stepsize dDelta not allowed!\n\n");
      ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
    }

  /* grid-related checks */
  {
    BOOLEAN haveAlphaBand = LALUserVarWasSet( &uvar_AlphaBand );
    BOOLEAN haveDeltaBand = LALUserVarWasSet( &uvar_DeltaBand );

    BOOLEAN haveSkyRegion, haveAlphaDelta, haveSkyGridFile, useSkyGridFile, haveMetric, useMetric;

    haveSkyRegion  = (uvar_skyRegion != NULL);
    haveAlphaDelta = (LALUserVarWasSet(&uvar_Alpha) && LALUserVarWasSet(&uvar_Delta) );
    haveSkyGridFile   = (uvar_skyGridFile != NULL);
    useSkyGridFile   = (uvar_gridType == GRID_FILE_SKYGRID) || (uvar_gridType == GRID_METRIC_SKYFILE);
    haveMetric     = (uvar_metricType > LAL_PMETRIC_NONE);
    useMetric     = (uvar_gridType == GRID_METRIC);

    if ( (haveAlphaBand && !haveDeltaBand) || (haveDeltaBand && !haveAlphaBand) )
      {
	LALPrintError ("\nERROR: Need either BOTH (AlphaBand, DeltaBand) or NONE.\n\n"); 
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
      }

    if ( !useSkyGridFile && !(haveSkyRegion || haveAlphaDelta) )
      {
        LALPrintError ("\nNeed sky-region: either use (Alpha,Delta) or skyRegion!\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
      }
    if ( haveSkyRegion && haveAlphaDelta )
      {
        LALPrintError ("\nOverdetermined sky-region: only use EITHER (Alpha,Delta)"
		       " OR skyRegion!\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);
      }
    if ( useSkyGridFile && !haveSkyGridFile )
      {
        LALPrintError ("\nERROR: gridType=FILE, but no skyGridFile specified!\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);  
      }
    if ( !useSkyGridFile && haveSkyGridFile )
      {
        LALWarning (status, "\nWARNING: skyGridFile was specified but not needed ..."
		    " will be ignored\n");
      }
    if ( useSkyGridFile && (haveSkyRegion || haveAlphaDelta) )
      {
        LALWarning (status, "\nWARNING: We are using skyGridFile, but sky-region was"
		    " also specified ... will be ignored!\n");
      }
    if ( !useMetric && haveMetric) 
      {
        LALWarning (status, "\nWARNING: Metric was specified for non-metric grid..."
		    " will be ignored!\n");
      }
    if ( useMetric && !haveMetric) 
      {
        LALPrintError ("\nERROR: metric grid-type selected, but no metricType selected\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);      
      }

  } /* grid-related checks */


  /* check that observation start+duration were specified correctly */
  {
    BOOLEAN haveEndTime = LALUserVarWasSet(&uvar_endTime);
    BOOLEAN haveDuration =LALUserVarWasSet(&uvar_duration);
    if ( !haveEndTime && !haveDuration )
      {
	LALPrintError ("\nERROR: need either --endTime or --duration!\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);      
      }
    if ( haveEndTime && haveDuration )
      {
	LALPrintError ("\nERROR: can specify only one of --endTime or --duration!\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);      
      }

    if ( haveEndTime && (uvar_endTime < uvar_startTime) )
      {
	LALPrintError ("\nERROR: endTime must be later than startTime!\n\n");
        ABORT (status, GETMESH_EINPUT, GETMESH_MSGEINPUT);      
      }
      
  } /* check Tspan */

  RETURN (status);
} /* checkUserInputConsistency() */


/** Determine the DopplerRegion in parameter-space to search over.
 * 
 * Normally this is just given directly by the user and therefore trivial. 
 *
 * However, for Monte-Carlo-runs testing the metric-grid we want to search 
 * a (randomized) 'small' region around a given parameter-space point, 
 * specifying only the (approx.) number of grid-points desired per dimension. 
 * This is done by getMCDopplerCube(), which determines the appropriate
 * Bands in each direction using the metric. 
 * This behavior is triggered by the user-input  --searchNeighbors.
 *
 * This function allows the user additionally to manually override these Bands.
 *
 */
void
getSearchRegion (LALStatus *status, 
		 DopplerRegion *searchRegion,	/**< OUT: the DopplerRegion to search over */
		 const DopplerSkyScanInit *params)	/**< IN: DopplerSkyScan params might be needed */
{

  DopplerRegion ret = empty_DopplerRegion;

  INITSTATUS (status, "getSearchRegion", rcsid);  
  ATTATCHSTATUSPTR (status);

  ASSERT ( searchRegion, status, GETMESH_ENULL, GETMESH_MSGENULL);
  ASSERT ( searchRegion->skyRegionString == NULL, status, 
	   GETMESH_EINPUT, GETMESH_MSGEINPUT);

  /* set defaults */
  ret.fkdot[0] = uvar_Freq;
  ret.fkdotBand[0] = uvar_FreqBand;

  ret.fkdot[1] = uvar_f1dot;
  ret.fkdotBand[1] = uvar_f1dotBand;

  /* 'normalize' if negative bands where given */
  if ( ret.fkdotBand[0] < 0 )
    {
      ret.fkdotBand[0] *= -1.0;
      ret.fkdot[0] -= ret.fkdotBand[0];
    }
  if ( ret.fkdotBand[1] < 0 )
    {
      ret.fkdotBand[1] *= -1.0;
      ret.fkdot[1] -= ret.fkdotBand[1];
    }



  /* if user specified the option -searchNeighbors=N, we generate an 
   * automatic search-region of N grid-steps in each dimension
   * around the given search-point
   */
  if ( LALUserVarWasSet(&uvar_searchNeighbors) ) 
    {
      DopplerRegion cube = empty_DopplerRegion;
      PulsarDopplerParams signal = empty_PulsarDopplerParams;
      
      signal.Alpha = uvar_Alpha;
      signal.Delta = uvar_Delta;
      signal.fkdot[0]  = uvar_Freq;
      signal.fkdot[1] = uvar_f1dot;

      /* set random-seed for MC grid-randomization */
      if ( LALUserVarWasSet(&uvar_randomSeed) )
	srand(uvar_randomSeed);
      else
	setTrueRandomSeed();

      /* construct MC doppler-cube around signal-location */
      TRY ( getMCDopplerCube(status->statusPtr, 
			     &cube, signal, uvar_searchNeighbors, params), status);

      /* free previous skyRegionString */
      if ( ret.skyRegionString )
	LALFree (ret.skyRegionString);

      /* overload defaults with automatic search-region */
      ret = cube;
      
    } /* if searchNeighbors */

  /* ---------- finally, the user can override the 'neighborhood' search-Region
   * if he explicitly specified search-bands in some (or all) directions.
   * 
   * The motivation is to allow more selective control over the search-region,
   * e.g. by setting certain bands to zero.
   */ 
  {
    BOOLEAN haveSkyBands = LALUserVarWasSet(&uvar_AlphaBand) && LALUserVarWasSet(&uvar_DeltaBand);

    if ( haveSkyBands )     /* manually set skyRegion */
      {
	CHAR *str = NULL;
	TRY ( SkySquare2String( status->statusPtr, &str,
				uvar_Alpha, uvar_Delta, 
				uvar_AlphaBand, uvar_DeltaBand), status);
	if ( ret.skyRegionString) 
	  LALFree ( ret.skyRegionString );

	ret.skyRegionString = str;
      }
    else if ( uvar_skyRegion )
      {
	if ( ret.skyRegionString) 
	  LALFree ( ret.skyRegionString );
	ret.skyRegionString = LALCalloc(1, strlen(uvar_skyRegion) + 1);
	strcpy ( ret.skyRegionString, uvar_skyRegion);
      }

    if ( LALUserVarWasSet(&uvar_FreqBand) )	/* manually set Frequency-interval */
      {
	ret.fkdot[0] = uvar_Freq;
	ret.fkdotBand[0] = uvar_FreqBand;
      }
    if ( LALUserVarWasSet(&uvar_f1dotBand) )	/* manually set spindown-interval */
      {
	ret.fkdot[1] = uvar_f1dot;
	ret.fkdotBand[1] = uvar_f1dotBand;
      }

  } /* user-override of search-bands */


  /* 'normalize' all spin-bands to be positive */
  if ( ret.fkdotBand[0] < 0 )
    {
      ret.fkdotBand[0]  *= -1.0;
      ret.fkdot[0]  -= ret.fkdotBand[0];
    }
  if ( ret.fkdotBand[1] < 0 )
    {
      ret.fkdotBand[1] *= -1.0;
      ret.fkdot[1] -= ret.fkdot[1];
    }

  *searchRegion = ret;	/* return the result */

  DETATCHSTATUSPTR (status);
  RETURN(status);

} /* getSearchRegion() */

/** set random-seed from /dev/urandom if possible, otherwise
 * from uninitialized local-var ;) 
 */
void
setTrueRandomSeed(void)
{
  FILE *fpRandom;
  INT4 seed;		/* NOTE: possibly used initialized! that's ok!! */

  fpRandom = fopen("/dev/urandom", "r");	/* read Linux random-pool for seed */
  if ( fpRandom == NULL ) 
    {
      LALPrintError ("\nCould not read from /dev/urandom ... using default seed.\n\n");
    }
  else
    {
      fread(&seed, sizeof(INT4),1, fpRandom);
      fclose(fpRandom);
    }

  srand(seed);

  return;
} /* setTrueRandomSeed() */


/*
 * Copyright (C) 2006, 2008 Reinhard Prix
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
 * \file
 *
 * \author{Reinhard Prix}
 *
 * New code to calculated various F-statistic metrics and mismatches
 *
 * Revision: $Id$
 *
 */

/* ---------- includes ---------- */
#include <math.h>


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


#include <lal/UserInput.h>
#include <lal/LALConstants.h>
#include <lal/Date.h>
#include <lal/LALInitBarycenter.h>
#include <lal/AVFactories.h>
#include <lal/SkyCoordinates.h>
#include <lal/ComputeFstat.h>
#include <lal/PulsarTimes.h>
#include <lal/SFTutils.h>

#include <lalapps.h>
#include <lal/ComputeFstat.h>
#include <lal/LogPrintf.h>
#include <lal/StringVector.h>

#include <lal/lalGitID.h>
#include <lalappsGitID.h>

#include <lal/FlatPulsarMetric.h>
#include <lal/UniversalDopplerMetric.h>

RCSID("$Id");

/* ---------- Error codes and messages ---------- */
#define FSTATMETRIC_EMEM 	1
#define FSTATMETRIC_EINPUT	2
#define FSTATMETRIC_ENULL	3
#define FSTATMETRIC_ENONULL	4
#define FSTATMETRIC_EFILE	5
#define FSTATMETRIC_EXLAL	6


#define FSTATMETRIC_MSGEMEM	"Out of memory."
#define FSTATMETRIC_MSGEINPUT	"Invalid input."
#define FSTATMETRIC_MSGENULL	"Illegal NULL input"
#define FSTATMETRIC_MSGENONULL	"Output field not initialized to NULL"
#define FSTATMETRIC_MSGEFILE	"File IO error"
#define FSTATMETRIC_MSGEXLAL	"XLAL function call failed"

/* ----- compile switches ----- */
/* uncomment the following to turn off range-checking in GSL vector-functions */
/* #define GSL_RANGE_CHECK_OFF 1*/

#define METRIC_FORMAT	"%.16g"		/* fprintf-format used for printing metric components */

/*---------- local defines ---------- */
#define TRUE 		(1==1)
#define FALSE 		(1==0)

#define OneBillion 	1.0e9
#define ORB_V0		1e-4

/* ----- Macros ----- */
/** convert GPS-time to REAL8 */
#define GPS2REAL8(gps) (1.0 * (gps).gpsSeconds + 1.e-9 * (gps).gpsNanoSeconds )

/** Simple Euklidean scalar product for two 3-dim vectors in cartesian coords */
#define SCALAR(u,v) ((u)[0]*(v)[0] + (u)[1]*(v)[1] + (u)[2]*(v)[2])

/** copy 3 components of Euklidean vector */
#define COPY_VECT(dst,src) do { (dst)[0] = (src)[0]; (dst)[1] = (src)[1]; (dst)[2] = (src)[2]; } while(0)

#define SQ(x) ((x) * (x))

/* ---------- local types ---------- */
typedef struct
{
  EphemerisData *edat;			/**< ephemeris data (from LALInitBarycenter()) */
  LIGOTimeGPS startTime;		/**< start time of observation */
  PulsarDopplerParams dopplerPoint;	/**< doppler point */
  MultiDetectorInfo detInfo;		/**< (multi-)detector info */
} ConfigVariables;


typedef struct
{
  BOOLEAN help;

  LALStringVector* IFOs;	/**< list of detector-names "H1,H2,L1,.." or single detector*/
  LALStringVector* IFOweights; /**< list of relative detector-weights "w1, w2, w3, .." */

  REAL8 Freq;		/**< target-frequency */
  REAL8 Alpha;		/**< skyposition Alpha: radians, equatorial coords. */
  REAL8 Delta;		/**< skyposition Delta: radians, equatorial coords. */
  REAL8 f1dot;		/**< target 1. spindown-value df/dt */

  CHAR *ephemEarth;	/**< filename of Earth ephemeris */
  CHAR *ephemSun;	/**< filename of Sun ephemeris */

  REAL8 startTime;	/**< GPS start time of observation */
  REAL8 duration;	/**< length of observation in seconds */

  REAL8 cosi;		/**< cos(iota) */
  REAL8 psi;		/**< polarization-angle psi */

  CHAR* outputMetric;	/**< filename to write metrics into */

  BOOLEAN fullFmetric;	/**< compute the 'full' F-metric including antenna-patterns, otherwise == phaseMetric */

  INT4 detMotionType;	/**< enum-value DetectorMotionType specifying type of detector-motion to use */

  INT4 projection;     /**< project metric onto surface */

  LALStringVector* coords; /**< list of Doppler-coordinates to compute metric in, see --coordsHelp for possible values */
  BOOLEAN coordsHelp;	/**< output help-string explaining all the possible Doppler-coordinate names for --cords */

} UserVariables_t;


/*---------- empty structs for initializations ----------*/
ConfigVariables empty_ConfigVariables;
PulsarTimesParamStruc empty_PulsarTimesParamStruc;
UserVariables_t empty_UserVariables;
/* ---------- global variables ----------*/
extern int vrbflg;

/* ---------- local prototypes ---------- */
void initUserVars (LALStatus *status, UserVariables_t *uvar);
void InitCode (LALStatus *status, ConfigVariables *cfg, const UserVariables_t *uvar);

EphemerisData* InitEphemeris (const CHAR *ephemEarth, const CHAR *ephemSun );

int project_metric( gsl_matrix *ret_ij, gsl_matrix *g_ij, const UINT4 coordinate );
int outer_product ( gsl_matrix *ret_ij, const gsl_vector *u_i, const gsl_vector *v_j );
int symmetrize ( gsl_matrix *mat );
REAL8 quad_form ( const gsl_matrix *mat, const gsl_vector *vec );


void FreeMem ( LALStatus *status, ConfigVariables *cfg );

/*============================================================
 * FUNCTION definitions
 *============================================================*/

int
main(int argc, char *argv[])
{
  LALStatus status = blank_status;
  ConfigVariables config = empty_ConfigVariables;
  UserVariables_t uvar = empty_UserVariables;
  FILE *fpMetric = 0;
  CHAR dummy[512];
  DopplerMetric *metric;

  sprintf (dummy, "%s", lalGitID );
  sprintf (dummy, "%s", lalappsGitID );

  lalDebugLevel = 0;
  vrbflg = 1;	/* verbose error-messages */

  /* set LAL error-handler */
  lal_errhandler = LAL_ERR_EXIT;

  /* register user-variables */
  LAL_CALL (LALGetDebugLevel (&status, argc, argv, 'v'), &status);

  /* set log-level */
  LogSetLevel ( lalDebugLevel );

  LAL_CALL (initUserVars (&status, &uvar), &status);

  /* read cmdline & cfgfile  */
  LAL_CALL (LALUserVarReadAllInput (&status, argc,argv), &status);

  if (uvar.help) 	/* help requested: we're done */
    return 0;

  if ( uvar.coordsHelp )
    {
      CHAR *helpstr;
      if ( (helpstr = XLALDopplerCoordinateHelpAll()) == NULL )
	{
	  LogPrintf ( LOG_CRITICAL, "XLALDopplerCoordinateHelpAll() failed!\n\n");
	  return -1;
	}
      printf ( "\n%s\n", helpstr );
      LALFree ( helpstr );
      return 0;
    } /* if coordsHelp */

  /* basic setup and initializations */
  LAL_CALL ( InitCode(&status, &config, &uvar), &status);

  /* ----- setup metric computation and call XLALDopplerMetric() ---------- */
  if ( ! uvar.fullFmetric )
    {
      metric = XLALDopplerPhaseMetric ( uvar.coords,
					uvar.detMotionType,
					&config.dopplerPoint,
					config.startTime,
					uvar.duration,
					config.edat,
					uvar.IFOs->data[0]
					);
      if ( !metric ) {
	LogPrintf (LOG_CRITICAL, "Something failed in XLALDopplerPhaseMetric()\n\n");
	return -1;
      }
    } /* if Phase metric */
  else
    {
      MultiDetectorInfo detInfo = empty_MultiDetectorInfo;
      if ( XLALParseMultiDetectorInfo ( &detInfo, uvar.IFOs, uvar.IFOweights ) != XLAL_SUCCESS ) {
	LogPrintf (LOG_CRITICAL, "Failed to parse detector names and/or weights\n\n");
	return -1;
      }

      metric = XLALDopplerFstatMetric ( uvar.coords,
					uvar.detMotionType,
					&config.dopplerPoint,
					config.startTime,
					uvar.duration,
					config.edat,
					&detInfo,
					uvar.cosi,
					uvar.psi
					);
      if ( !metric ) {
	LogPrintf (LOG_CRITICAL, "Something failed in XLALDopplerFstatMetric()\n\n");
	return -1;
      }

    } /* if full Fstat Metric */

  /* ---------- output results ---------- */
  if ( uvar.outputMetric )
    {
      UINT4 i;
      const DopplerMetricParams* meta = &(metric->meta);

      if ( (fpMetric = fopen ( uvar.outputMetric, "wb" )) == NULL )
	return FSTATMETRIC_EFILE;

      fprintf ( fpMetric, "%%%% History: %s\n%%%%%s\n", lalGitID, lalappsGitID );
      fprintf ( fpMetric, "%%%% DopplerCoordinates = [ " );
      for ( i=0; i < meta->coordSys.dim; i ++ )
	{
	  if ( i > 0 ) fprintf ( fpMetric, ", " );
	  fprintf ( fpMetric, "%s", XLALDopplerCoordinateName(meta->coordSys.coordIDs[i]));
	}
      fprintf ( fpMetric, "];\n");
      fprintf ( fpMetric, "%%%% DetectorMotionType = '%s'\n", XLALDetectorMotionName(meta->detMotionType) );
      fprintf ( fpMetric, "%%%% DopplerPoint = {\n");
      fprintf ( fpMetric, "%%%% 	refTime = {%d, %d}\n",
		meta->dopplerPoint.refTime.gpsSeconds, meta->dopplerPoint.refTime.gpsNanoSeconds );
      fprintf ( fpMetric, "%%%% 	Alpha = %f rad; Delta = %f rad\n", meta->dopplerPoint.Alpha, meta->dopplerPoint.Delta );
      fprintf ( fpMetric, "%%%% 	fkdot = [%f, %g, %g, %g ]\n",
		meta->dopplerPoint.fkdot[0], meta->dopplerPoint.fkdot[1], meta->dopplerPoint.fkdot[2], meta->dopplerPoint.fkdot[3] );
      if ( meta->dopplerPoint.orbit )
	{
	  const BinaryOrbitParams *orbit = meta->dopplerPoint.orbit;
	  fprintf ( fpMetric, "%%%% 	   orbit = { \n");
	  fprintf ( fpMetric, "%%%% 		tp = {%d, %d}\n", orbit->tp.gpsSeconds, orbit->tp.gpsNanoSeconds );
	  fprintf ( fpMetric, "%%%% 		argp  = %g\n", orbit->argp );
	  fprintf ( fpMetric, "%%%% 		asini = %g\n", orbit->asini );
	  fprintf ( fpMetric, "%%%% 		ecc = %g\n", orbit->ecc );
	  fprintf ( fpMetric, "%%%% 		period = %g\n", orbit->period );
	  fprintf ( fpMetric, "%%%% 	   }\n");
	} /* if doppler->orbit */
      fprintf ( fpMetric, "%%%% }\n");

      fprintf ( fpMetric, "%%%% startTime = {%d, %d}\n", meta->startTime.gpsSeconds, meta->startTime.gpsNanoSeconds );
      fprintf ( fpMetric, "%%%% duration  = %f\n", meta->Tspan );
      fprintf ( fpMetric, "%%%% detectors = [");
      for ( i=0; i < meta->detInfo.length; i ++ )
	{
	  if ( i > 0 ) fprintf ( fpMetric, ", ");
	  fprintf ( fpMetric, "%s", meta->detInfo.sites[i].frDetector.name );
	}
      fprintf ( fpMetric, "];\n");
      fprintf ( fpMetric, "%%%% detectorWeights = [");
      for ( i=0; i < meta->detInfo.length; i ++ )
	{
	  if ( i > 0 ) fprintf ( fpMetric, ", ");
	  fprintf ( fpMetric, "%f", meta->detInfo.detWeights[i] );
	}
      fprintf ( fpMetric, "];\n");

      if ( ! metric->meta.fullFmetric )
	{
	  fprintf ( fpMetric, "\ng_ij = \\\n" ); XLALfprintfGSLmatrix ( fpMetric, METRIC_FORMAT,  metric->g_ij );
	} /* simple phase metric */
      else
	{
	  REAL4 D = metric->A * metric->B - SQ(metric->C);
	  fprintf ( fpMetric, "%%%% cosi = %g; psi = %g\n", meta->cosi, meta->psi );
	  fprintf ( fpMetric, "\ng_ij = \\\n" ); XLALfprintfGSLmatrix ( fpMetric, METRIC_FORMAT,  metric->g_ij );

	  fprintf ( fpMetric, "\nA = %.16g; B = %.16g; C = %.16g; D = %.16g;\n", metric->A, metric->B, metric->C, D );
	  fprintf ( fpMetric, "\ngFav_ij = \\\n" ); XLALfprintfGSLmatrix ( fpMetric, METRIC_FORMAT,  metric->gFav_ij );
	  fprintf ( fpMetric, "\nm1_ij = \\\n" ); XLALfprintfGSLmatrix ( fpMetric, METRIC_FORMAT,  metric->m1_ij );
	  fprintf ( fpMetric, "\nm2_ij = \\\n" ); XLALfprintfGSLmatrix ( fpMetric, METRIC_FORMAT,  metric->m2_ij );
	  fprintf ( fpMetric, "\nm3_ij = \\\n" ); XLALfprintfGSLmatrix ( fpMetric, METRIC_FORMAT,  metric->m3_ij );
	} /* full Fstat-metric */

      fclose ( fpMetric );

    } /* if outputMetric */

  /* ----- done: free all memory */
  XLALDestroyDopplerMetric ( metric );
  LAL_CALL (FreeMem(&status, &config), &status);

  LALCheckMemoryLeaks();

  return 0;
} /* main */


/** register all "user-variables" */
void
initUserVars (LALStatus *status, UserVariables_t *uvar)
{
  INITSTATUS( status, "initUserVars", rcsid );
  ATTATCHSTATUSPTR (status);

  /* set a few defaults */
  uvar->help = FALSE;

#define EPHEM_EARTH  "earth00-04.dat"
#define EPHEM_SUN    "sun00-04.dat"
  uvar->ephemEarth = LALCalloc (1, strlen(EPHEM_EARTH)+1);
  strcpy (uvar->ephemEarth, EPHEM_EARTH);
  uvar->ephemSun = LALCalloc (1, strlen(EPHEM_SUN)+1);
  strcpy (uvar->ephemSun, EPHEM_SUN);

  uvar->Freq = 100;
  uvar->f1dot = 0.0;

  uvar->startTime = 714180733;
  uvar->duration = 10 * 3600;

  uvar->projection = 0;
  if ( (uvar->IFOs = XLALCreateStringVector ( "H1", NULL )) == NULL ) {
    LogPrintf (LOG_CRITICAL, "Call to XLALCreateStringVector() failed with xlalErrno = %d\n", xlalErrno );
    ABORT ( status, FSTATMETRIC_EXLAL, FSTATMETRIC_MSGEXLAL );
  }

  uvar->IFOweights = NULL;

  uvar->fullFmetric = FALSE;
  uvar->detMotionType = DETMOTION_SPIN_ORBIT;

  if ( (uvar->coords = XLALCreateStringVector ( "Freq_Nat", "Alpha", "Delta", "f1dot_Nat", NULL )) == NULL ) {
    LogPrintf (LOG_CRITICAL, "Call to XLALCreateStringVector() failed with xlalErrno = %d\n", xlalErrno );
    ABORT ( status, FSTATMETRIC_EXLAL, FSTATMETRIC_MSGEXLAL );
  }

  /* register all our user-variables */

  LALregBOOLUserStruct(status,	help,		'h', UVAR_HELP,		"Print this help/usage message");
  LALregLISTUserStruct(status,	IFOs,		'I', UVAR_OPTIONAL, 	"Comma-separated list of detectors, eg. \"H1,H2,L1,G1, ...\" ");
  LALregLISTUserStruct(status,	IFOweights,	 0,  UVAR_OPTIONAL, 	"Comma-separated list of relative noise-weights, eg. \"w1,w2,w3,..\" ");
  LALregREALUserStruct(status,	Alpha,		'a', UVAR_OPTIONAL,	"skyposition Alpha in radians, equatorial coords.");
  LALregREALUserStruct(status,	Delta, 		'd', UVAR_OPTIONAL,	"skyposition Delta in radians, equatorial coords.");
  LALregREALUserStruct(status,	Freq, 		'f', UVAR_OPTIONAL, 	"target frequency");
  LALregREALUserStruct(status,	f1dot, 		's', UVAR_OPTIONAL, 	"first spindown-value df/dt");
  LALregREALUserStruct(status, 	startTime,      't', UVAR_OPTIONAL, 	"GPS start time of observation");
  LALregREALUserStruct(status,  duration,	'T', UVAR_OPTIONAL,	"Alternative: Duration of observation in seconds");
  LALregSTRINGUserStruct(status, ephemEarth,    'E', UVAR_OPTIONAL, 	"Filename of Earth-ephemeris");
  LALregSTRINGUserStruct(status, ephemSun,      'S', UVAR_OPTIONAL, 	"Filename of Sun-ephemeris");
  LALregREALUserStruct(status, 	cosi,	 	 0, UVAR_OPTIONAL,	"Pulsar orientation-angle cos(iota) [-1,1]" );
  LALregREALUserStruct(status,	psi,		 0, UVAR_OPTIONAL,	"Wave polarization-angle psi [0, pi]" );
  LALregSTRINGUserStruct(status, outputMetric,	'o', UVAR_OPTIONAL,	"Output the metric components (in octave format) into this file.");
  LALregINTUserStruct(status,   projection,      0,  UVAR_OPTIONAL,     "Coordinate of metric projection: 0=none, 1=f, 2=Alpha, 3=Delta, 4=f1dot");

  LALregLISTUserStruct(status,	coords,		'c', UVAR_OPTIONAL, 	"Doppler-coordinates to compute metric in (see --coordsHelp)");
  LALregBOOLUserStruct(status,	coordsHelp,      0,  UVAR_OPTIONAL,     "output help-string explaining all the possible Doppler-coordinate names for --coords");

  LALregBOOLUserStruct(status,	fullFmetric,     0,  UVAR_OPTIONAL,     "Compute the 'full' F-metric including antenna-patterns; if FALSE: phaseMetric");
  LALregINTUserStruct(status,  	detMotionType,	 0,  UVAR_DEVELOPER,	"Detector-motion: 0=spin+orbit, 1=orbit, 2=spin, 3=spin+ptoleorbit, 4=ptoleorbit");

  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* initUserVars() */


/** basic initializations: set-up 'ConfigVariables'
 */
void
InitCode (LALStatus *status, ConfigVariables *cfg, const UserVariables_t *uvar)
{

  INITSTATUS (status, "InitCode", rcsid);
  ATTATCHSTATUSPTR (status);

  /* ----- determine start-time from user-input */
  XLALGPSSetREAL8( &(cfg->startTime), uvar->startTime );

  if ( (cfg->edat = InitEphemeris ( uvar->ephemEarth, uvar->ephemSun )) == NULL ) {
    LogPrintf (LOG_CRITICAL, "Failed to initialize ephemeris data!\n");
    ABORT ( status, FSTATMETRIC_EINPUT, FSTATMETRIC_MSGEINPUT);
  }

  /* ----- get parameter-space point from user-input) */
  cfg->dopplerPoint = empty_PulsarDopplerParams;
  cfg->dopplerPoint.refTime = cfg->startTime;
  cfg->dopplerPoint.Alpha = uvar->Alpha;
  cfg->dopplerPoint.Delta = uvar->Delta;
  cfg->dopplerPoint.fkdot[0] = uvar->Freq;
  cfg->dopplerPoint.fkdot[1] = uvar->f1dot;
  cfg->dopplerPoint.orbit = NULL;

  /* ----- initialize IFOs and (Multi-)DetectorStateSeries  ----- */
  {
    UINT4 numDet = uvar->IFOs->length;

    if ( numDet > DOPPLERMETRIC_MAX_DETECTORS ) {
      LogPrintf (LOG_CRITICAL, "More detectors (%d) specified than can be handled maximally (%d)\n", numDet, DOPPLERMETRIC_MAX_DETECTORS );
      ABORT (status, FSTATMETRIC_EINPUT, FSTATMETRIC_MSGEINPUT);
    }
    if ( uvar->IFOweights && (uvar->IFOweights->length != numDet ) )
      {
	LogPrintf (LOG_CRITICAL, "If specified, the number of IFOweights (%d) must agree with the number of IFOs!\n",
		   uvar->IFOweights->length, numDet );
	ABORT (status, FSTATMETRIC_EINPUT, FSTATMETRIC_MSGEINPUT);
      }

    if ( ! uvar->fullFmetric && (numDet > 1) ) {
      LogPrintf ( LOG_CRITICAL, "PhaseMetric (fullFmetric==FALSE) allows specifying only a single detector!\n");
      ABORT (status, FSTATMETRIC_EINPUT, FSTATMETRIC_MSGEINPUT);
    }

  } /* handle detector input */

  DETATCHSTATUSPTR (status);
  RETURN (status);

} /* InitCode() */


/** Free all memory */
void
FreeMem ( LALStatus *status, ConfigVariables *cfg )
{

  INITSTATUS (status, "FreeMem", rcsid);
  ATTATCHSTATUSPTR (status);

  ASSERT ( cfg, status, FSTATMETRIC_ENULL, FSTATMETRIC_MSGENULL );

  TRY ( LALDestroyUserVars ( status->statusPtr ), status );

  LALFree ( cfg->edat->ephemE );
  LALFree ( cfg->edat->ephemS );
  LALFree ( cfg->edat );

  DETATCHSTATUSPTR (status);
  RETURN(status);

} /* FreeMem() */


/** Calculate the projected metric onto the subspace of 'c' given by
 * ret_ij = g_ij - ( g_ic * g_jc / g_cc ) , where c is the value of the projected coordinate
 * The output-matrix ret must be allocated
 *
 * return 0 = OK, -1 on error.
 */
int
project_metric( gsl_matrix *ret_ij, gsl_matrix * g_ij, const UINT4 c )
{
  UINT4 i,j;

  if ( !ret_ij )
    return -1;
  if ( !g_ij )
    return -1;
  if ( (ret_ij->size1 != ret_ij->size2) )
    return -1;
  if ( (g_ij->size1 != g_ij->size2) )
    return -1;

  for ( i=0; i < ret_ij->size1; i ++) {
    for ( j=0; j < ret_ij->size2; j ++ ) {
      if ( i==c || j==c ) {
	gsl_matrix_set ( ret_ij, i, j, 0.0 );
      }
      else {
	gsl_matrix_set ( ret_ij, i, j, ( gsl_matrix_get(g_ij, i, j) - (gsl_matrix_get(g_ij, i, c) * gsl_matrix_get(g_ij, j, c) / gsl_matrix_get(g_ij, c, c)) ));
      }
    }
  }
  return 0;
}


/** Calculate the outer product ret_ij of vectors u_i and v_j, given by
 * ret_ij = u_i v_j
 * The output-matrix ret must be allocated and have dimensions len(u) x len(v)
 *
 * return 0 = OK, -1 on error.
 */
int
outer_product (gsl_matrix *ret_ij, const gsl_vector *u_i, const gsl_vector *v_j )
{
  UINT4 i, j;

  if ( !ret_ij || !u_i || !v_j )
    return -1;

  if ( (ret_ij->size1 != u_i->size) || ( ret_ij->size2 != v_j->size) )
    return -1;


  for ( i=0; i < ret_ij->size1; i ++)
    for ( j=0; j < ret_ij->size2; j ++ )
      gsl_matrix_set ( ret_ij, i, j, gsl_vector_get(u_i, i) * gsl_vector_get(v_j, j) );

  return 0;

} /* outer_product() */


/* symmetrize the input-matrix 'mat' (which must be quadratic)
 */
int
symmetrize ( gsl_matrix *mat )
{
  gsl_matrix *tmp;

  if ( !mat )
    return -1;
  if ( mat->size1 != mat->size2 )
    return -1;

  tmp = gsl_matrix_calloc ( mat->size1, mat->size2 );

  gsl_matrix_transpose_memcpy ( tmp, mat );

  gsl_matrix_add (mat, tmp );

  gsl_matrix_scale ( mat, 0.5 );

  gsl_matrix_free ( tmp );

  return 0;

} /* symmetrize() */


/* compute the quadratic form = vec.mat.vec
 */
REAL8
quad_form ( const gsl_matrix *mat, const gsl_vector *vec )
{
  UINT4 i, j;
  REAL8 ret = 0;

  if ( !mat || !vec )
    return 0;
  if ( (mat->size1 != mat->size2) || ( mat->size1 != vec->size ) )
    return 0;

  for (i=0; i < mat->size1; i ++ )
    for (j=0; j < mat->size2; j ++ )
      ret += gsl_vector_get(vec, i) * gsl_matrix_get (mat, i, j) * gsl_vector_get(vec,j);

  return ret;

} /* quad_form() */


/** Load Ephemeris from ephemeris data-files  */
EphemerisData *
InitEphemeris (const CHAR *ephemEarth,	/**< filename of Earth-ephemeris */
	       const CHAR *ephemSun	/**< filename of Sun-ephemeris */
	       )
{
#define FNAME_LENGTH 1024
  const CHAR *fn = "InitEphemeris()";
  LALStatus status = blank_status;
  EphemerisData *edat;

  if ( !ephemEarth || !ephemSun ) {
    XLALPrintError ("\n%s: NULL pointer passed as ephemeris-filename!\n", fn);
    return NULL;
  }

  if ( (strlen ( ephemEarth ) > FNAME_LENGTH) || (strlen ( ephemSun ) > FNAME_LENGTH) ) {
    XLALPrintError ( "\n%s: Ephemeris-filenames must not be longer than %d!\n", fn, FNAME_LENGTH );
    return NULL;
  }

  /* allocate memory for ephemeris-data to be returned */
  if ( (edat = LALCalloc ( 1, sizeof(*edat))) == NULL ) {
    XLALPrintError("\n%s: Out of memory!\n", fn );
    return NULL;
  }

  /* NOTE: the 'ephiles' are ONLY ever used in LALInitBarycenter, which is
   * why we don't need to store copies of these filenames
   */
  edat->ephiles.earthEphemeris = ephemEarth;
  edat->ephiles.sunEphemeris   = ephemSun;

  LALInitBarycenter( &status, edat);
  if ( status.statusCode ) {
    XLALPrintError("\n%s: call to LALInitBarycenter failed with code %d\n", fn, status.statusCode );
    return NULL;
  }

  edat->leap = 0;	/* NOTE! we don't care about leap-seconds for the metric! */

  return edat;

} /* InitEphemeris() */

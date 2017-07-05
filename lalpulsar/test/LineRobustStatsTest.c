/*
 * Copyright (C) 2011, 2014 David Keitel
 * Copyright (C) 2017 Reinhard Prix
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

/* ---------- Includes -------------------- */
#include <lal/AVFactories.h>
#include <lal/LineRobustStats.h>
#include <lal/LogPrintf.h>

/******************************************************
 *  Error codes and messages.
 */

/* ---------- Defines -------------------- */
#define TRUE (1==1)
#define FALSE (1==0)

/**
 * Enum for LR-statistic variants
 */
typedef enum {
  LRS_BSGL  = 0,
  LRS_BSGLtL = 1,
  LRS_BtSGLtL = 2,
  LRS_BStSGLtL = 3
} LRstat_variant_t;


/*---------- internal prototypes ----------*/
int
XLALCompareBSGLComputations ( const REAL4 TwoF,
			      const UINT4 numDetectors,
			      const REAL4Vector *TwoFX,
			      const REAL4 cohFstar0,
			      const REAL4 *oLtLGX,
			      const UINT4 numSegs,
			      const REAL4 tolerance,
			      const LRstat_variant_t LRstat_variant,
			      const REAL4 maxTwoF,
			      const REAL4Vector *maxTwoFX
			   );

REAL4
XLALComputePedestrianLRStat ( const REAL4 TwoF,
			      const UINT4 numDetectors,
			      const REAL4Vector *TwoFX,
			      const REAL4 cohFstar0,
			      const REAL4 *oLtLGX,
			      const BOOLEAN useLogCorrection,
			      const LRstat_variant_t LRstat_variant,
			      const REAL4 maxTwoF,
			      const REAL4Vector *maxTwoFX,
			      const UINT4 numSegs
			   );

int
XLALCheckBSGLDifferences ( const REAL4 diff,
			   const REAL4 tolerance,
			   const CHAR *casestring
			);

int
XLALCheckBSGLVectorFunctions (void);

/* ###################################  MAIN  ################################### */

int main( int argc, char *argv[]) {

  /* sanity check for input arguments */
  XLAL_CHECK ( argc == 1, XLAL_EINVAL, "The executable '%s' doesn't support any input arguments right now.\n", argv[0] );

  printf ("Starting coherent test...\n");
  UINT4 numSegs = 1;

  /* set up single- and multi-IFO F-stat input */
  REAL4 TwoF = 7.0;
  UINT4 numDetectors = 2;
  REAL4Vector *TwoFX = NULL;
  XLAL_CHECK ( (TwoFX = XLALCreateREAL4Vector ( numDetectors )) != NULL, XLAL_EFUNC );
  TwoFX->data[0] = 4.0;
  TwoFX->data[1] = 12.0;
  /* maximum allowed difference between recalculated and XLAL result */
  REAL4 tolerance = 1e-06;

  /* compute and compare the results for one set of Fstar0, oLtLGX values */
  REAL4 cohFstar0 = -LAL_REAL4_MAX; /* prior from BSGL derivation, -Inf means pure line veto, +Inf means pure multi-Fstat */
  REAL4 oLtLGXarray[numDetectors]; /* per-IFO prior odds ratio for line vs. Gaussian noise */
  oLtLGXarray[0] = 0.5; /* pick some arbitrary values first */
  oLtLGXarray[1] = 0.8;
  REAL4 *oLtLGX = oLtLGXarray;
  printf ("Computing BSGL for TwoF_multi=%f, TwoFX=(%f,%f), priors cohF*0=%f and oLtLGX=(%f,%f)...\n", TwoF, TwoFX->data[0], TwoFX->data[1], cohFstar0, oLtLGX[0], oLtLGX[1] );
  XLAL_CHECK ( XLALCompareBSGLComputations( TwoF, numDetectors, TwoFX, cohFstar0, oLtLGX, numSegs, tolerance, LRS_BSGL, 0.0, NULL ) == XLAL_SUCCESS, XLAL_EFUNC );

  /* change the priors to catch more possible problems */
  cohFstar0 = 10.0;
  oLtLGX = NULL; /* NULL is interpreted as oLtLGX[X]=1 for all X (in most general case includes both L and tL) */

  printf ("Computing BSGL for TwoF_multi=%f, TwoFX=(%f,%f), priors cohF*0=%f and oLtLGX=NULL...\n", TwoF, TwoFX->data[0], TwoFX->data[1], cohFstar0 );
  XLAL_CHECK ( XLALCompareBSGLComputations( TwoF, numDetectors, TwoFX, cohFstar0, oLtLGX, numSegs, tolerance, LRS_BSGL, 0.0, NULL ) == XLAL_SUCCESS, XLAL_EFUNC );

  XLALDestroyREAL4Vector(TwoFX);

  printf ("Starting semi-coherent test...\n");
  numSegs = 3;

  /* initialize some per-segment F-stats */
  REAL4Vector *TwoFl = NULL;
  REAL4Vector *TwoFXl = NULL;
  XLAL_CHECK ( (TwoFl = XLALCreateREAL4Vector ( numSegs )) != NULL, XLAL_EFUNC );
  XLAL_CHECK ( (TwoFXl = XLALCreateREAL4Vector ( numSegs*numDetectors )) != NULL, XLAL_EFUNC );
  TwoFl->data[0] = 4.0;
  TwoFl->data[1] = 8.0;
  TwoFl->data[2] = 4.0;
  for ( UINT4 X = 0; X < numDetectors; X++ ) {
    for ( UINT4 l = 0; l < numSegs; l++ ) {
      TwoFXl->data[X*numSegs+l] = TwoFl->data[l]/sqrt(numDetectors);
    }
  }
  TwoFXl->data[0+1] = 32.0;

  REAL4 sumTwoF = 0.0;
  REAL4Vector *sumTwoFX = NULL;
  XLAL_CHECK ( (sumTwoFX = XLALCreateREAL4Vector ( numDetectors )) != NULL, XLAL_EFUNC );
  for ( UINT4 l = 0; l < numSegs; l++ ) {
    sumTwoF += TwoFl->data[l];
    for ( UINT4 X = 0; X < numDetectors; X++ ) {
      sumTwoFX->data[X] += TwoFXl->data[X*numSegs+l];
    }
  }

  printf ("Computing semi-coherent BSGL for sum_TwoF_multi=%f, sum_TwoFX=(%f,%f), priors cohF*0=%f and oLtLGX=NULL...\n", sumTwoF, sumTwoFX->data[0], sumTwoFX->data[1], cohFstar0 );
  XLAL_CHECK ( XLALCompareBSGLComputations( sumTwoF, numDetectors, sumTwoFX, cohFstar0, oLtLGX, numSegs, tolerance, LRS_BSGL, 0.0, NULL ) == XLAL_SUCCESS, XLAL_EFUNC );

  REAL4 maxTwoF = 0.0;
  REAL4Vector *maxTwoFX = NULL;
  XLAL_CHECK ( (maxTwoFX = XLALCreateREAL4Vector ( numDetectors )) != NULL, XLAL_EFUNC );
  for ( UINT4 l = 0; l < numSegs; l++ ) {
    maxTwoF = fmax(maxTwoF,TwoFl->data[l]);
    for ( UINT4 X = 0; X < numDetectors; X++ ) {
      maxTwoFX->data[X] = fmax(maxTwoFX->data[X],TwoFXl->data[X*numSegs+l]);
    }
  }

  printf ("Computing semi-coherent BSGLtL for sum_TwoF_multi=%f, sum_TwoFX=(%f,%f), max_TwoF_multi=%f, max_TwoFX=(%f,%f), priors cohF*0=%f and oLtLGX=NULL...\n", sumTwoF, sumTwoFX->data[0], sumTwoFX->data[1], maxTwoF, maxTwoFX->data[0], maxTwoFX->data[1], cohFstar0 );
  XLAL_CHECK ( XLALCompareBSGLComputations( sumTwoF, numDetectors, sumTwoFX, cohFstar0, oLtLGX, numSegs, tolerance, LRS_BSGLtL, maxTwoF, maxTwoFX  ) == XLAL_SUCCESS, XLAL_EFUNC );

  printf ("Computing semi-coherent BtSGLtL for sum_TwoF_multi=%f, sum_TwoFX=(%f,%f), max_TwoF_multi=%f, max_TwoFX=(%f,%f), priors cohF*0=%f and oLtLGX=NULL...\n", sumTwoF, sumTwoFX->data[0], sumTwoFX->data[1], maxTwoF, maxTwoFX->data[0], maxTwoFX->data[1], cohFstar0 );
  XLAL_CHECK ( XLALCompareBSGLComputations( sumTwoF, numDetectors, sumTwoFX, cohFstar0, oLtLGX, numSegs, tolerance, LRS_BtSGLtL, maxTwoF, maxTwoFX  ) == XLAL_SUCCESS, XLAL_EFUNC );

  printf ("Computing semi-coherent BStSGLtL for sum_TwoF_multi=%f, sum_TwoFX=(%f,%f), max_TwoF_multi=%f, max_TwoFX=(%f,%f), priors cohF*0=%f and oLtLGX=NULL...\n", sumTwoF, sumTwoFX->data[0], sumTwoFX->data[1], maxTwoF, maxTwoFX->data[0], maxTwoFX->data[1], cohFstar0 );
  XLAL_CHECK ( XLALCompareBSGLComputations( sumTwoF, numDetectors, sumTwoFX, cohFstar0, oLtLGX, numSegs, tolerance, LRS_BStSGLtL, maxTwoF, maxTwoFX  ) == XLAL_SUCCESS, XLAL_EFUNC );

  /* free memory */
  XLALDestroyREAL4Vector(TwoFl);
  XLALDestroyREAL4Vector(TwoFXl);
  XLALDestroyREAL4Vector(sumTwoFX);
  XLALDestroyREAL4Vector(maxTwoFX);


  // check vector versions of B*SGL* function
  XLAL_CHECK_MAIN ( XLALCheckBSGLVectorFunctions() == XLAL_SUCCESS, XLAL_EFUNC );

  LALCheckMemoryLeaks();

  return XLAL_SUCCESS;

} /* main */


/**
 * Test function to compute BSGL values from XLALComputeBSGL
 * against those recomputed from scratch ("pedestrian" formula);
 * compare the results and exit if tolerance is violated.
 */
int
XLALCompareBSGLComputations ( const REAL4 TwoF,			/**< multi-detector  Fstat */
			     const UINT4 numDetectors,		/**< number of detectors */
			     const REAL4Vector *TwoFX,		/**< vector of single-detector Fstats */
			     const REAL4 cohFstar0,		/**< amplitude prior normalization for lines */
			     const REAL4 *oLtLGX,		/**< array of single-detector prior line odds ratio, can be NULL */
			     const UINT4 numSegs,		/**< number of segments */
			     const REAL4 tolerance,		/**< tolerance for comparisons */
			     const LRstat_variant_t LRstat_variant, /**< which statistic variant to use */
			     const REAL4 maxTwoF,		/**< multi-detector maximum Fstat over segments*/
			     const REAL4Vector *maxTwoFX	/**< vector of single-detector maximum Fstats over segments*/
                          )
{

  /* pedestrian version, with and without log corrections */
  REAL4 log10LRS_extcomp_notallterms = XLALComputePedestrianLRStat ( TwoF, numDetectors, TwoFX, cohFstar0, oLtLGX, FALSE, LRstat_variant, maxTwoF, maxTwoFX, numSegs );
  REAL4 log10LRS_extcomp_allterms    = XLALComputePedestrianLRStat ( TwoF, numDetectors, TwoFX, cohFstar0, oLtLGX, TRUE, LRstat_variant, maxTwoF, maxTwoFX, numSegs );

  /* faster version: use only the leading term of the BSGL denominator sum */
  BSGLSetup *setup_noLogCorrection;
  XLAL_CHECK ( (setup_noLogCorrection = XLALCreateBSGLSetup ( numDetectors, numSegs*cohFstar0, oLtLGX, FALSE, numSegs )) != NULL, XLAL_EFUNC );
  REAL4 log10_LRS_XLAL_notallterms = 0.0;
  char funcname[64] = "";
  switch ( LRstat_variant ) {
    case LRS_BSGL :
      log10_LRS_XLAL_notallterms = XLALComputeBSGL ( TwoF, TwoFX->data, setup_noLogCorrection );
      snprintf(funcname, sizeof(funcname), "%s", "XLALComputeBSGL");
      break;
    case LRS_BSGLtL  :
      log10_LRS_XLAL_notallterms = XLALComputeBSGLtL ( TwoF, TwoFX->data, maxTwoFX->data, setup_noLogCorrection );
      snprintf(funcname, sizeof(funcname), "%s", "XLALComputeBSGLtL");
      break;
    case LRS_BtSGLtL  :
      log10_LRS_XLAL_notallterms = XLALComputeBtSGLtL ( maxTwoF, TwoFX->data, maxTwoFX->data, setup_noLogCorrection );
      snprintf(funcname, sizeof(funcname), "%s", "XLALComputeBtSGLtL");
      break;
    case LRS_BStSGLtL  :
      log10_LRS_XLAL_notallterms = XLALComputeBStSGLtL ( TwoF, maxTwoF, TwoFX->data, maxTwoFX->data, setup_noLogCorrection );
      snprintf(funcname, sizeof(funcname), "%s", "XLALComputeBStSGLtL");
      break;
  }
  XLAL_CHECK ( xlalErrno == 0, XLAL_EFUNC, "XLALComputeBSGL() failed with xlalErrno = %d\n", xlalErrno );
  XLALDestroyBSGLSetup ( setup_noLogCorrection ); setup_noLogCorrection = NULL;

  /* more precise version: use all terms of the BSGL denominator sum */
  BSGLSetup *setup_withLogCorrection;
  XLAL_CHECK ( (setup_withLogCorrection = XLALCreateBSGLSetup ( numDetectors, numSegs*cohFstar0, oLtLGX, TRUE, numSegs )) != NULL, XLAL_EFUNC );
  REAL4 log10_LRS_XLAL_allterms = 0.0;
  switch ( LRstat_variant ) {
    case LRS_BSGL :
      log10_LRS_XLAL_allterms = XLALComputeBSGL ( TwoF, TwoFX->data, setup_withLogCorrection );
      break;
//     case LRS_BSGLtL  :
//       log10_LRS_XLAL_allterms = XLALComputeBSGLtL ( TwoF, TwoFX->data, maxTwoFX->data, setup_withLogCorrection );
//       break;
//     case LRS_BtSGLtL  :
//       log10_LRS_XLAL_allterms = XLALComputeBtSGLtL ( maxTwoF, TwoFX->data, maxTwoFX->data, setup_withLogCorrection );
//       break;
//     case LRS_BStSGLtL  :
//       log10_LRS_XLAL_allterms = XLALComputeBStSGLtL ( TwoF, maxTwoF, TwoFX->data, maxTwoFX->data, setup_withLogCorrection );
//       break;
      default :
        printf("log correction not implemented for GLtL denominator, skipping full-precision test for this LRS variant.\n");
  }
  XLAL_CHECK ( xlalErrno == 0, XLAL_EFUNC, "%s() failed with xlalErrno = %d\n", funcname, xlalErrno );
  XLALDestroyBSGLSetup ( setup_withLogCorrection ); setup_withLogCorrection = NULL;

  /* compute relative deviations */
  REAL4 diff_allterms = 0.0;
  if ( LRstat_variant == 0 ) {
    diff_allterms    = fabs( log10_LRS_XLAL_allterms    - log10LRS_extcomp_allterms    ) / ( 0.5 * ( log10_LRS_XLAL_allterms    + log10LRS_extcomp_allterms    ));
  }
  REAL4 diff_notallterms = fabs( log10_LRS_XLAL_notallterms - log10LRS_extcomp_notallterms ) / ( 0.5 * ( log10_LRS_XLAL_notallterms + log10LRS_extcomp_notallterms ));

  /* output results and deviations and return with error when tolerances are violated */
  printf ( "Externally recomputed     with  allterms: log10LRS=%f\n",               log10LRS_extcomp_allterms );
  printf ( "Externally recomputed     with !allterms: log10LRS=%f\n",               log10LRS_extcomp_notallterms );
  char casestring[256] = "";
  if ( LRstat_variant == 0 ) {
    printf ( "%s()         with  allterms: log10LRS=%f (rel. dev.: %f)", funcname, log10_LRS_XLAL_allterms,    diff_allterms );
    snprintf(casestring, sizeof(casestring), "%s() with useAllTerms=TRUE", funcname);
    XLAL_CHECK ( XLALCheckBSGLDifferences ( diff_allterms,    tolerance, casestring ) == XLAL_SUCCESS, XLAL_EFUNC );
  }
  printf ( "%s()         with !allterms: log10LRS=%f (rel. dev.: %f)", funcname, log10_LRS_XLAL_notallterms, diff_notallterms );
  snprintf(casestring, sizeof(casestring), "%s() with useAllTerms=FALSE", funcname);
  XLAL_CHECK ( XLALCheckBSGLDifferences ( diff_notallterms, tolerance, casestring ) == XLAL_SUCCESS, XLAL_EFUNC );

  return XLAL_SUCCESS;

} /* XLALCompareBSGLComputations() */

/**
 * compute BSGL "the pedestrian way", from Eq. (40) of Keitel, Prix, Papa, Leaci, Siddiqi, PR D 89, 064023 (2014),
 * explicit formula for numDet=2:
 * log10 BSGL = F - log ( e^F* + e^{F1}*oLtLG1/oLtLG + e^{F2}*oLtLG2/oLtLG )
 */
REAL4
XLALComputePedestrianLRStat ( const REAL4 TwoF,				/**< multi-detector  Fstat */
			      const UINT4 numDetectors,			/**< number of detectors */
			      const REAL4Vector *TwoFX,			/**< vector of single-detector Fstats */
			      const REAL4 cohFstar0,			/**< amplitude prior normalization for lines */
			      const REAL4 *oLtLGX,			/**< array of single-detector prior line odds ratio, can be NULL (in most general case includes both L and tL) */
			      const BOOLEAN useLogCorrection,		/**< include log-term correction or not */
			      const LRstat_variant_t LRstat_variant, 	/**< which statistic variant to compute */
			      const REAL4 maxTwoF,			/**< multi-detector maximum Fstat over segments*/
			      const REAL4Vector *maxTwoFX,		/**< vector of single-detector maximum Fstats over segments*/
			      const UINT4 numSegs			/**< number of segments */
                          )
{

  REAL4 log10LRS_extcomp = 0;

  /* sum up overall prior odds */
  REAL4 oLtLGXarray[numDetectors];
  REAL4 oLtLG = 0.0;
  if ( oLtLGX ) {
    for ( UINT4 X = 0; X < numDetectors; X++ ) {
      oLtLG += oLtLGX[X];
    }
  }
  else { /* if oLtLGX == NULL, assume oLtLGX=1/numDetectors for all X  ==> oLtLG = sumX oLtLGX = 1*/
    oLtLG = 1.0;
    for ( UINT4 X = 0; X < numDetectors; X++ ) {
      oLtLGXarray[X] = 1.0/numDetectors;
    }
    oLtLGX = oLtLGXarray;
  } // if ( oLtLGX == NULL )

  REAL4 numerator = 0.0;
  if ( LRstat_variant != LRS_BtSGLtL ) {
    numerator += exp(0.5*TwoF);
  }
  if ( ( LRstat_variant == LRS_BtSGLtL ) || ( LRstat_variant == LRS_BStSGLtL ) ) {
    numerator += exp(0.5*maxTwoF+(numSegs-1)*cohFstar0)/numSegs; /* extra 1/numSegs frome qual splitting of prior odds over segments */
  }

  if ( LRstat_variant == LRS_BStSGLtL ) {
    numerator *= 0.5; /* assume equal splitting of prior odds between S and tS */
  }

  log10LRS_extcomp = log(numerator);

  REAL4 *denomterms = NULL;
  REAL4 denomterms_GL[1+numDetectors];
  REAL4 denomterms_GLtL[1+2*numDetectors];
  if ( LRstat_variant == LRS_BSGL ) {
   denomterms = denomterms_GL;
  }
  else {
   denomterms = denomterms_GLtL;
  }

  denomterms[0] = exp(numSegs*cohFstar0)/oLtLG;
  REAL4 maxterm = denomterms[0];
  for (UINT4 X = 0; X < numDetectors; X++) {
    denomterms[1+X] = oLtLGX[X]*exp(0.5*TwoFX->data[X])/oLtLG;
    if ( LRstat_variant != LRS_BSGL ) {
     denomterms[1+X] *= 0.5; /* assume equal splitting of prior odds between L and tL */
    }
    maxterm = fmax(maxterm,denomterms[1+X]);
  }
  if ( LRstat_variant != LRS_BSGL ) { /* NOTE: this is the loudest-only (per detector) version only! */
    for (UINT4 X = 0; X < numDetectors; X++) {
      denomterms[1+numDetectors+X] = 0.5*oLtLGX[X]*exp(0.5*maxTwoFX->data[X]+(numSegs-1)*cohFstar0)/(oLtLG*numSegs); /* 0.5 from assuming equal splitting of prior odds between L and tL and 1/numSegs from equal splitting over segments */
      maxterm = fmax(maxterm,denomterms[1+numDetectors+X]);
    }
  }

  if ( !useLogCorrection ) {
    log10LRS_extcomp -= log(maxterm);
  }
  else {
    REAL4 BSGL_extcomp_denom = 0.0;
    for (UINT4 X = 0; X < 1+numDetectors; X++) {
      BSGL_extcomp_denom += denomterms[X];
    }
    log10LRS_extcomp -= log( BSGL_extcomp_denom );
  }

  /* this is not yet the log-Bayes-factor, as computed by XLALComputeBSGL(), so need to correct by log(1+1/oLtLG) */
  log10LRS_extcomp += log(1+1/oLtLG);
  /* and actually switch to log10 */
  log10LRS_extcomp *= LAL_LOG10E;

  return log10LRS_extcomp;

} /* XLALComputePedestrianLRStat() */

int
XLALCheckBSGLDifferences ( const REAL4 diff,
			  const REAL4 tolerance,
			  const CHAR *casestring
			)
{

  if ( fabs(diff) <= tolerance ) {
    printf (" ==> OK!\n");
  }
  else {
    printf (" ==> BAD!\n");
    XLAL_ERROR ( XLAL_EFAILED, "\nTolerance %f exceeded for %s!\n", tolerance, casestring );
    return XLAL_FAILURE;
  }

  return XLAL_SUCCESS;

} /* XLALCheckBSGLDifferences() */

int
XLALCheckBSGLVectorFunctions ( void )
{
#define VEC_LEN 5
#define NUM_SEGS 3

  REAL4 cohFstar0 = 4.37; // 10% false-alarm: invFalseAlarm_chi2 ( 0.1, 4 * 3 ) / 3
  UINT4 numDet = 3;
  REAL4 oLtLGX[3] = {0.1, 0.8, 1.5}; /* per-IFO prior odds ratio for line vs. Gaussian noise */

  BSGLSetup *setup_noLogCorr;
  BSGLSetup *setup_withLogCorr;
  XLAL_CHECK ( (setup_noLogCorr   = XLALCreateBSGLSetup ( numDet, NUM_SEGS*cohFstar0, oLtLGX, FALSE, NUM_SEGS )) != NULL, XLAL_EFUNC );
  XLAL_CHECK ( (setup_withLogCorr = XLALCreateBSGLSetup ( numDet, NUM_SEGS*cohFstar0, oLtLGX, TRUE,  NUM_SEGS )) != NULL, XLAL_EFUNC );

  // generate some 'reasonable' fake Fstat numbers
  // trying to span some range of 'favoring noise' to 'favoring signal'
  REAL4 twoFPerDet0[PULSAR_MAX_DETECTORS][VEC_LEN] = { 				 // printf ( "{%f, %f, %f, %f, %f},\n", 3 * unifrnd ( 4, 8, [3,5] ) )
    {14.333578, 18.477274, 19.765395, 18.754920, 22.048903},
    {14.180436, 15.754341, 17.745453, 14.727934, 18.460679},
    {21.988492, 15.632081, 18.439836, 19.492070, 16.174584},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
  };
  REAL4 twoF[VEC_LEN]; XLAL_INIT_MEM ( twoF );
  REAL4 attenuate[VEC_LEN] = { 0.1, 0.3, 0.5, 0.7, 0.9 };
  for ( UINT4 i = 0; i < VEC_LEN; i ++ ) {
    for ( UINT4 X = 0; X < PULSAR_MAX_DETECTORS; X ++ ) {
      twoF[i] += twoFPerDet0[X][i];
    }
    twoF[i] *= attenuate[i];
  }

  const REAL4 *twoFPerDet[PULSAR_MAX_DETECTORS];
  for ( UINT4 X = 0; X < PULSAR_MAX_DETECTORS; X ++ ) {
    twoFPerDet[X] = twoFPerDet0[X];
  }

  REAL4 maxTwoFSeg[VEC_LEN] = {4.585174, 13.221874, 13.005110, 34.682062, 48.968980}; 	// maxTwoFSeg = unifrnd ( twoF ./ 3, twoF ); printf ( "{%f, %f, %f, %f, %f};\n", maxTwoFSeg );
  REAL4 maxTwoFSegPerDet0[PULSAR_MAX_DETECTORS][VEC_LEN] = { 				// maxTwoFSegPerDet = unifrnd ( twoFPerDet ./ 3, twoFPerDet ); printf ( "{%f, %f, %f, %f, %f},\n", maxTwoFSegPerDet );
    {12.341489, 9.090180, 16.084636, 14.056626, 11.450930},
    {14.006959, 8.774348, 9.584360, 18.157686, 11.551223},
    {7.490141, 18.599300, 9.724949, 15.562138, 10.512018},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
  };
  const REAL4 *maxTwoFSegPerDet[PULSAR_MAX_DETECTORS];
  for ( UINT4 X = 0; X < PULSAR_MAX_DETECTORS; X ++ ) {
    maxTwoFSegPerDet[X] = maxTwoFSegPerDet0[X];
  }

  REAL4 outBSGL_log[VEC_LEN];
  REAL4 outBSGL_nolog[VEC_LEN];
  REAL4 outBSGLtL[VEC_LEN];
  REAL4 outBtSGLtL[VEC_LEN];

  XLAL_CHECK ( XLALVectorComputeBSGL ( outBSGL_log, twoF, twoFPerDet, VEC_LEN, setup_withLogCorr ) == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALVectorComputeBSGL ( outBSGL_nolog, twoF, twoFPerDet, VEC_LEN, setup_noLogCorr ) == XLAL_SUCCESS, XLAL_EFUNC );

  XLAL_CHECK ( XLALVectorComputeBSGLtL ( outBSGLtL, twoF, twoFPerDet, maxTwoFSegPerDet, VEC_LEN, setup_noLogCorr ) == XLAL_SUCCESS, XLAL_EFUNC );
  XLAL_CHECK ( XLALVectorComputeBtSGLtL (outBtSGLtL, maxTwoFSeg, twoFPerDet, maxTwoFSegPerDet, VEC_LEN, setup_noLogCorr ) == XLAL_SUCCESS, XLAL_EFUNC );

  printf ("\nBSGL_log = {" );
  for ( UINT4 i = 0; i < VEC_LEN; i ++ ) { printf ( "%f%s ", outBSGL_log[i], (i<VEC_LEN-1)? "," : "}" ); }
  printf ("\n");
  printf ("\nBSGL_nolog = {" );
  for ( UINT4 i = 0; i < VEC_LEN; i ++ ) { printf ( "%f%s ", outBSGL_nolog[i], (i<VEC_LEN-1)? "," : "}" ); }
  printf ("\n");
  printf ("\nBSGLtL = {" );
  for ( UINT4 i = 0; i < VEC_LEN; i ++ ) { printf ( "%f%s ", outBSGLtL[i], (i<VEC_LEN-1)? "," : "}" ); }
  printf ("\n");
  printf ("\nBtSGLtL = {" );
  for ( UINT4 i = 0; i < VEC_LEN; i ++ ) { printf ( "%f%s ", outBtSGLtL[i], (i<VEC_LEN-1)? "," : "}" ); }
  printf ("\n");

  // compute these alternatively with the single-value function versions
  REAL4 errBSGL_log = 0;
  REAL4 errBSGL_nolog = 0;
  REAL4 errBSGLtL = 0;
  REAL4 errBtSGLtL = 0;

  for ( UINT4 i = 0; i < VEC_LEN; i ++ )
    {
      REAL4 twoFX[PULSAR_MAX_DETECTORS];
      REAL4 maxTwoFXl[PULSAR_MAX_DETECTORS];
      for ( UINT4 X = 0; X < PULSAR_MAX_DETECTORS; X ++ )
        {
          twoFX[X] = twoFPerDet[X][i];
          maxTwoFXl[X] = maxTwoFSegPerDet[X][i];
        }
      REAL4 cmp;
      // BSGL_log
      cmp = XLALComputeBSGL ( twoF[i], twoFX, setup_withLogCorr );
      XLAL_CHECK ( xlalErrno == XLAL_SUCCESS, XLAL_EFUNC, "XLALComputeBSGL() failed.\n" );
      errBSGL_log = fmaxf ( errBSGL_log, fabsf ( cmp - outBSGL_log[i] ) );
      printf ("i = %d: err = %g\n", i, errBSGL_log );

      // BSGL_nolog
      cmp = XLALComputeBSGL ( twoF[i], twoFX, setup_noLogCorr );
      XLAL_CHECK ( xlalErrno == XLAL_SUCCESS, XLAL_EFUNC, "XLALComputeBSGL() failed.\n" );
      errBSGL_nolog = fmaxf ( errBSGL_nolog, fabsf ( cmp - outBSGL_nolog[i] ) );
      printf ("i = %d: err = %g\n", i, errBSGL_nolog );

      // BSGLtL
      cmp = XLALComputeBSGLtL ( twoF[i], twoFX, maxTwoFXl, setup_noLogCorr );
      XLAL_CHECK ( xlalErrno == XLAL_SUCCESS, XLAL_EFUNC, "XLALComputeBSGLtL() failed.\n" );
      errBSGLtL = fmaxf ( errBSGLtL, fabsf ( cmp - outBSGLtL[i] ) );
      printf ("i = %d: err = %g\n", i,errBSGLtL );

      // BtSGLtL
      cmp = XLALComputeBtSGLtL ( maxTwoFSeg[i], twoFX, maxTwoFXl, setup_noLogCorr );
      XLAL_CHECK ( xlalErrno == XLAL_SUCCESS, XLAL_EFUNC, "XLALComputeBtSGLtL() failed.\n" );
      errBtSGLtL = fmaxf ( errBtSGLtL, fabsf ( cmp - outBtSGLtL[i] ) );
      printf ("i = %d: err = %g\n", i, errBtSGLtL );

    } // for i < VECL_LEN

  REAL4 tolerance = 5 * LAL_REAL4_EPS;
  XLAL_CHECK ( errBSGL_log <= tolerance, XLAL_ETOL, "Error in vector BSGL with log-correction exceeds tolerance, %g > %g\n", errBSGL_log, tolerance );
  XLAL_CHECK ( errBSGL_nolog <= tolerance, XLAL_ETOL, "Error in vector BSGL without log-correction exceeds tolerance, %g > %g\n", errBSGL_nolog, tolerance );
  XLAL_CHECK ( errBSGLtL <= tolerance, XLAL_ETOL, "Error in vector BSGLtL exceeds tolerance, %g > %g\n", errBSGLtL, tolerance );
  XLAL_CHECK ( errBtSGLtL <= tolerance, XLAL_ETOL, "Error in vector BtSGLtL exceeds tolerance, %g > %g\n", errBtSGLtL, tolerance );

  printf ("%s: success!\n", __func__ );

  XLALDestroyBSGLSetup ( setup_noLogCorr );
  XLALDestroyBSGLSetup ( setup_withLogCorr );

  return XLAL_SUCCESS;

} // XLALCheckBSGLVectorFunctions()

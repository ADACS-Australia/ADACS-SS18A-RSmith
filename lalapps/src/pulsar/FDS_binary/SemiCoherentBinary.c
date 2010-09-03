/*
 * Copyright (C) 2010 Chris Messenger
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

/** \author C.Messenger
 * \ingroup pulsarCoherent
 * \file
 * \brief
 * This code is designed to compute the Bayes factor for a semi-coherent analysis
 * of input SFT data specific to searching for continuous signals in a binary system. 
 *
 * It generates likelihood samples from a coarse grid of templates placed on each SFT and 
 * combines them using a fine binary template band.  The parameter space is integrated over 
 * and a Bayes factor is produced.
 *
 */

/***********************************************************************************************/
/* includes */
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <gsl/gsl_interp.h>        /* needed for the gsl interpolation */
#include <gsl/gsl_spline.h>        /* needed for the gsl interpolation */
#include <gsl/gsl_rng.h>           /* for random number generation */ 
#include <gsl/gsl_randist.h>       /* for random number generation */ 
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <lal/TimeSeries.h>
#include <lal/LALDatatypes.h>
#include <lal/Units.h>
#include <lal/SFTutils.h>
#include <lal/SFTfileIO.h>
#include <lal/ComplexFFT.h>
#include <lal/UserInput.h>
#include <lal/LogPrintf.h>
#include <lal/LALFrameIO.h>
#include <lal/FrameStream.h>
#include <lalappsfrutils.h>
#include <lalapps.h>

/***********************************************************************************************/
/* some global constants */

#define STRINGLENGTH 256              /* the length of general string */
#define LONGSTRINGLENGTH 1024         /* the length of general string */
#define NFREQMAX 4                    /* the max dimensionality of the frequency derivitive grid */
#define NBINMAX 4                     /* the number of binary parameter dimensions */
#define WINGS_FACTOR 1.05             /* the safety factor in reading extra frequency from SFTs */
#define PCU_AREA 0.13                 /* the collecting area of a single PCU in square metres */
#define DEFAULT_SOURCE "SCOX1"        /* the default source name */

/***********************************************************************************************/
/* some useful macros */

#define MYMAX(x,y) ( (x) > (y) ? (x) : (y) )
#define MYMIN(x,y) ( (x) < (y) ? (x) : (y) )

/***********************************************************************************************/
/* define internal structures */

/** A single parameter prior pdf
 */
typedef struct { 
  REAL8Vector *logpriors;           /**< vector that stores the log of the prior pdf */
  REAL8 logdelta;                   /**< the log of the spacing */
  BOOLEAN gaussian;                 /**< are we using a Gaussian prior on this parameter */
} REAL8Priors;

/** A vector of prior pdfs for many dimensions 
 */
typedef struct { 
  REAL8Priors *data;                /**< points to the prior data */
  UINT4 ndim;                       /**< the dimensionality of the prior space */
} REAL8PriorsVector;

/** A single parameter dimensions boundaries
 */
typedef struct { 
  REAL8 min;                        /**< the parameter space minimum */
  REAL8 max;                        /**< the parameter space maximium */
  REAL8 mid;                        /**< the parameter space mid point (where a Gaussian prior is centered) */
  REAL8 sig;                        /**< the one-sigma uncertainty on the parameter */
  REAL8 span;                       /**< the parameter space span */
  BOOLEAN gaussian;                 /**< are we using a Gaussian prior on this parameter */
  CHAR name[LALNameLength];         /**< string containing the name of the dimension */
} REAL8Dimension;

/** A vector of parameter space boundary information 
 */
typedef struct { 
  REAL8Dimension *data;             /**< the boundaries, span, etc for a single dimension */
  UINT4 ndim;                       /**< the number of dimensions */
} REAL8Space;

/** Stores the gridding parameters for a single dimension 
 */
typedef struct { 
  REAL8 min;                        /**< the starting points of the grid */
  REAL8 delta;                      /**< the grid spacings */
  UINT4 length;                     /**< the number of templates in each dimension */
  CHAR name[LALNameLength];         /**< string containing the name of the dimension */
} Grid;

/** Stores the current location in a hyper-cubic parameter space
 */
typedef struct { 
  REAL8 *x;                         /**< the location in parameter space */
  INT4 *idx;                        /**< the index of each dimension for this template */
  UINT4 ndim;                       /**< the dimension of the parameter space */
  UINT4 currentidx;                 /**< the current index value of the template */
} Template;

/** Stores the gridding parameters for a hypercubic grid of templates
 */
typedef struct { 
  Grid *grid;                       /**< stores the parameters defining a single dimension */
  UINT4 ndim;                       /**< the number of dimensions */
  UINT4 *prod;                      /**< internal variable used to store the size of sub-dimensions */
  UINT4 max;                        /**< the maximum (total) number of templates */
  REAL8 mismatch;                   /**< the mismatch */
} GridParameters;

/** Stores the parameters of an injection
 */
typedef struct { 
  Template temp;                    /**< stores the parameters of the signal */
  REAL8 amp;                        /**< the injected amplitude */
} InjectionParameters;

/** contains information regarding the search parameter space
 */
typedef struct { 
  REAL8Space *space;                /**< stores the parameter space boundaries */
  GridParameters *gridparams;       /**< stores the grid */
  REAL8PriorsVector *priors;        /**< stores the priors on the paramaters */ 
  LIGOTimeGPS epoch;                /**< the start time of the entire observation */
  REAL8 span;                       /**< the span of the entire observation */
  REAL8 tseg;                       /**< the coherent time */
  CHAR source[LALNameLength];       /**< the name of the source */
  InjectionParameters *inj;         /**< stores the injected signal parameters (if any) */
} ParameterSpace;

/*****************************************************************************************/

/** Stores the gridding parameters for a hypercubic grid of templates
 */
typedef struct { 
  GridParameters **segment;         /**< stores the parameters defining a single dimension */
  UINT4 length;                     /**< the number of segments */
} GridParametersVector;

/** Stores segment parameters
 */
typedef struct { 
  INT4Vector *npcus;                /**< a vector of PCUs */
  REAL8Vector *dt;                  /**< a vector of sampling times */
} SegmentParams;

/** Stores parameters useful for the efficient calcualtion of the likelihood
 */
typedef struct { 
  REAL8 logsqrtP;                   /**< intermediate variable for the phase and amp marginalised likelihood calculation */
  REAL8 PQ;                         /**< intermediate variable for the phase and amp marginalised likelihood calculation */
} LikelihoodParams;

/** Stores parameters useful for the efficient calcualtion of the likelihood
 */
typedef struct { 
  LikelihoodParams *data;           /**< a vector of likelihood parameter structures */
  UINT4 length;                     /**< the length of the vector */
} LikelihoodParamsVector;

/** Stores the results of a Bayesian posterior integration (Bayes factor, evidence, posteriors, etc...)
 */
typedef struct { 
  REAL8 logBayesFactor_phaseamp;                /**< the log Bayes factor for phase and amplitude marginalised per segment */
  REAL8 logBayesFactor_phase;                   /**< the log Bayes factor for phase marginalised per segment */
  REAL8Vector *logBayesFactor_phaseamp_vector;  /**< the log Bayes factor for each segment individually */
  REAL8Vector **logposteriors_phaseamp;         /**< the output log posteriors for phase and amplitude marginalised per segment */
  REAL8Vector **logposteriors_phase;            /**< the output log posteriors for phase marginalised per segment */
  GridParameters *gridparams;                   /**< the grid used for the marginalisation */
  UINT4 ndim;                                   /**< the dimensionality of the space */
  LIGOTimeGPS *epoch;                           /**< the epochs of each segment */
  UINT4 nsegments;                              /**< the number of segments used */
} BayesianProducts;

/** Storage for the demodulated power from a single segment 
 */
typedef struct { 
  REAL4Vector *data;                /**< pointer to the power data stored sequentially */
  REAL8 r;                          /**< the estimated noise background */
  UINT4 npcus;                      /**< the number of PCUs */
  REAL8 dt;                         /**< the original sampling time */
  LIGOTimeGPS epoch;                /**< the epoch of the segment */
  REAL8 duration;                   /**< the duration of the segment */
  GridParameters *gridparams;       /**< the grid on which the power was computed */
} REAL4DemodulatedPower;

/** Storage for the demodulated power 
 */
typedef struct { 
  REAL4DemodulatedPower **segment;  /**< pointer to a set of REAL4VectorArrays */
  UINT4 length;                     /**< the number of segments */
} REAL4DemodulatedPowerVector;

/** An array of COMPLEX8TimeSeries 
 */
typedef struct { 
  COMPLEX8TimeSeries **data;        /**< pointer to a set of COMPLEX8TimeSeries */
  UINT4 length;                     /**< the number of vectors */
} COMPLEX8TimeSeriesArray;

/** A structure that stores user input variables 
 */
typedef struct { 
  BOOLEAN help;		            /**< trigger output of help string */
  CHAR *sftbasename;                /**< basename of input SFT files */
  CHAR *outputdir;                  /**< the output directory */
  CHAR *source;                     /**< the name of the source */
  REAL8 freq;                       /**< the starting frequency */
  REAL8 freqband;                   /**< the search band width */
  REAL8 orbperiod;                  /**< the central orbital period value */
  REAL8 deltaorbperiod;             /**< the uncertainty on the orbital period */
  REAL8 asini;                      /**< the central orbital semi-major axis value */
  REAL8 deltaasini;                 /**< the uncertainty on the orbital semi-major axis value */
  REAL8 tasc;                       /**< the central time of ascension value */
  REAL8 deltatasc;                  /**< the uncertainty on the central time of ascension value */
  REAL8 nsig;                       /**< the width (in sigmas) of the Gaussian priors */
  REAL8 mismatch;                   /**< the grid mismatch */      
  REAL8 sigalpha;                   /**< the amplitude prior sigma */
  INT4 gpsstart;                    /**< the min GPS time to include */
  INT4 gpsend;                      /**< the max GPS time to include */
  BOOLEAN gaussianpriors;           /**< flag for using Gaussian priors on the orbital parameters */
  CHAR *obsid_pattern;              /**< the OBS ID substring */
  INT4 seed;                        /**< fix the random number generator seed */
  REAL8 inject_amplitude;           /**< the amplitude of the injected signal */
  BOOLEAN version;	            /**< output version-info */
} UserInput_t;

/***********************************************************************************************/
/* global variables */
extern int vrbflg;	 	/**< defined in lalapps.c */
RCSID( "$Id$");		        /* FIXME: use git-ID instead to set 'rcsid' */

/* parameters for bessel function calculation (taken from Abramowitz and Stegun P.367) */
UINT4 LEN_BESSCO_HIGH = 9;
REAL8 BESSCO_HIGH[] = {0.39894228,0.01328592,0.00225319,-0.00157565,0.00916281,-0.02057706,0.02635537,-0.01647633,0.00392377};
UINT4 LEN_BESSCO_LOW = 7;
REAL8 BESSCO_LOW[] = {1.0,3.5156229,3.0899424,1.2067492,0.2659732,0.0360768,0.0045813};

/***********************************************************************************************/
/* define functions */
int main(int argc,char *argv[]);
int XLALReadUserVars(int argc,char *argv[],UserInput_t *uvar, CHAR **clargs);
int XLALDefineBinaryParameterSpace(REAL8Space **space,UserInput_t *uvar); 
int XLALReadSFTs(SFTVector **sfts,SegmentParams **segparams,CHAR *sftbasename, REAL8 freq, REAL8 freqband, INT4 gpsstart, INT4 gpsend,CHAR *obsid_pattern);
int XLALComputeFreqGridParamsVector(GridParametersVector **freqgridparams,REAL8Space *pspace, SFTVector *sftvec, REAL8 mu);
int XLALComputeFreqGridParams(GridParameters **freqgridparams,REAL8Space *pspace, REAL8 tmid,REAL8 tsft, REAL8 mu);
int XLALSFTVectorToCOMPLEX8TimeSeriesArray(COMPLEX8TimeSeriesArray **dstimevec, SFTVector *sftvec);
int XLALComputeDemodulatedPower(REAL4DemodulatedPower **power,COMPLEX8TimeSeries *time,GridParameters *gridparams);
int XLALComputeDemodulatedPowerVector(REAL4DemodulatedPowerVector **power,COMPLEX8TimeSeriesArray *time,GridParametersVector *gridparams);
int XLALEstimateBackgroundFlux(REAL8Vector **background, SegmentParams *segparams, SFTVector *sftvec);
int XLALComputeBinaryGridParams(GridParameters **binarygridparams,REAL8Space *space,REAL8 T,REAL8 DT,REAL8 mu);
int XLALComputeBinaryPriors(REAL8PriorsVector **priors,REAL8Space *space,GridParameters *gridparams);
int XLALComputeBayesFactor(BayesianProducts **Bayes,REAL4DemodulatedPowerVector *power,ParameterSpace *pspace,REAL8 sigalpha);
int XLALGetNextBinaryTemplate(Template **temp,GridParameters *gridparams);
int XLALComputeBinaryFreqDerivitives(Template *fdots,Template *bintemp,REAL8 tmid);
REAL8 XLALComputePhaseAmpMargLogLRatio(REAL8 X,LikelihoodParams *Lparams);
int XLALSetupLikelihood(LikelihoodParamsVector **Lparamsvec,BayesianProducts **Bayes,REAL4DemodulatedPowerVector *power,GridParameters *binarygrid,REAL8 sigalpha);
REAL8 XLALLogBesselI0(REAL8 z);
REAL8 XLALLogSumExp(REAL8 logx,REAL8 logy);
int XLALFreeParameterSpace(ParameterSpace *pspace);
int XLALFreeREAL4DemodulatedPowerVector(REAL4DemodulatedPowerVector *power);
int XLALFreeBayesianProducts(BayesianProducts *Bayes);
int XLALOutputBayesResults(CHAR *outputdir,BayesianProducts *Bayes,ParameterSpace *pspace,CHAR *clargs);
int XLALAddBinarySignalToSFTVector(SFTVector **sftvec,ParameterSpace *pspace,REAL8 inject_amplitude,INT4 seed);
int XLALInitgslrand(gsl_rng **gslrnd,INT8 seed);

/***********************************************************************************************/
/* empty initializers */
UserInput_t empty_UserInput;
ParameterSpace empty_ParameterSpace;

/** The main function of semicoherentbinary.c
 *
 */
int main( int argc, char *argv[] )  {

  static const char *fn = __func__;             /* store function name for log output */
  UserInput_t uvar = empty_UserInput;           /* user input variables */
  CHAR *clargs = NULL;                          /* store the command line args */
  SFTVector *sftvec = NULL;                     /* stores the input SFTs */
  SegmentParams *segparams = NULL;              /* stores the number of PCUs and sampling time per SFT */
  REAL8Vector *background = NULL;               /* estimates of the background for each SFT */
  ParameterSpace pspace = empty_ParameterSpace; /* the search parameter space */
  COMPLEX8TimeSeriesArray *dstimevec = NULL;    /* contains the downsampled inverse FFT'd SFTs */
  REAL4DemodulatedPowerVector *power = NULL;    /* contains the demodulated power for all SFTs */
  GridParametersVector *freqgridparams = NULL;  /* the coherent grid on the frequency derivitive parameter space */
  BayesianProducts *Bayes = NULL;               /* the Bayesian results */
  REAL8 fmin_read,fmax_read,fband_read;         /* the range of frequencies to be read from SFTs */
  UINT4 i;                                      /* counter */

  lalDebugLevel = 1;
  vrbflg = 1;	                        /* verbose error-messages */

  /* turn off default GSL error handler */
  gsl_set_error_handler_off();

  /* setup LAL debug level */
  if (XLALGetDebugLevel(argc, argv, 'v')) {
    LogPrintf(LOG_CRITICAL,"%s : XLALGetDebugLevel() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogSetLevel(lalDebugLevel);

  /* register and read all user-variables */
  if (XLALReadUserVars(argc,argv,&uvar,&clargs)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALReadUserVars() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : read in uservars\n",fn);
 
  /**********************************************************************************/
  /* DEFINE THE BINARY PARAMETER SPACE */
  /**********************************************************************************/

  /* register and read all user-variables */
  if (XLALDefineBinaryParameterSpace(&(pspace.space),&uvar)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALDefineBinaryParameterSpace() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : defined binary parameter prior space\n",fn);

  /* make crude but safe estimate of the bandwidth required for the source */
  fmin_read = pspace.space->data[0].min - WINGS_FACTOR*pspace.space->data[0].min*pspace.space->data[1].max*pspace.space->data[3].max;
  fmax_read = pspace.space->data[0].max + WINGS_FACTOR*pspace.space->data[0].max*pspace.space->data[1].max*pspace.space->data[3].max;
  fband_read = fmax_read - fmin_read;
  LogPrintf(LOG_DEBUG,"%s : reading in SFT frequency band [%f -> %f]\n",fn,fmin_read,fmax_read);
 
  /**********************************************************************************/
  /* READ THE SFT DATA */
  /**********************************************************************************/
  
  /* load in the SFTs - also fill in the segment parameters structure */
  if (XLALReadSFTs(&sftvec,&segparams,uvar.sftbasename,fmin_read,fband_read,uvar.gpsstart,uvar.gpsend,uvar.obsid_pattern)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALReadSFTs() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : read in SFTs\n",fn);
  
  /* define SFT length and the start and span of the observations plus the definitive segment time */
  pspace.tseg = 1.0/sftvec->data[0].deltaF;
  memcpy(&(pspace.epoch),&(sftvec->data[0].epoch),sizeof(LIGOTimeGPS));
  pspace.span = XLALGPSDiff(&(sftvec->data[sftvec->length-1].epoch),&(sftvec->data[0].epoch)) + pspace.tseg;
  sprintf(pspace.source,"%s",uvar.source);
  LogPrintf(LOG_DEBUG,"%s : SFT length = %f seconds\n",fn,pspace.tseg);
  LogPrintf(LOG_DEBUG,"%s : entire dataset starts at GPS time %d contains %d SFTS and spans %.0f seconds\n",fn,pspace.epoch.gpsSeconds,sftvec->length,pspace.span);

  /**********************************************************************************/
  /* COMPUTE THE FINE GRID PARAMETERS */
  /**********************************************************************************/
  
  /* compute the fine grid on the binary parameters */
  if (XLALComputeBinaryGridParams(&(pspace.gridparams),pspace.space,pspace.span,pspace.tseg,uvar.mismatch)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALComputeBinaryGridParams() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : computed the binary parameter space grid\n",fn);
 
  /**********************************************************************************/
  /* COMPUTE THE BINARY PARAMETER SPACE PRIORS */
  /**********************************************************************************/
  
  /* compute the priors on the binary parameters */
  if (XLALComputeBinaryPriors(&(pspace.priors),pspace.space,pspace.gridparams)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALComputeBinaryPriors() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : computed the binary parameter space priors\n",fn);
  
  /**********************************************************************************/
  /* ADD A SIMULATED SIGNAL TO THE SFT DATA */
  /**********************************************************************************/

  if (XLALUserVarWasSet(&(uvar.inject_amplitude))) {
    if (XLALAddBinarySignalToSFTVector(&sftvec,&pspace,uvar.inject_amplitude,uvar.seed)) {
      LogPrintf(LOG_CRITICAL,"%s : XLALAddBinarySignalToSFTVector() failed with error = %d\n",fn,xlalErrno);
      return 1;
    }
    LogPrintf(LOG_DEBUG,"%s : added a simulated signal to the SFTs\n",fn);
  }

  /**********************************************************************************/
  /* ESTIMATE BACKGROUND NOISE FROM SFTS */
  /**********************************************************************************/
  
  /* compute the background noise using the sfts */
  if (XLALEstimateBackgroundFlux(&background,segparams,sftvec)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALEstimateBackgroundFlux() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : estimated the background noise from the SFTs\n",fn);
 
  /**********************************************************************************/
  /* CONVERT ALL SFTS TO DOWNSAMPLED TIMESERIES */
  /**********************************************************************************/
  
  /* convert sfts to downsample dtimeseries */
  if (XLALSFTVectorToCOMPLEX8TimeSeriesArray(&dstimevec,sftvec)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALSFTVectorToCOMPLEX8TimeSeriesArray() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : converted SFTs to downsampled timeseries\n",fn);

  /**********************************************************************************/
  /* COMPUTE THE COARSE GRID ON FREQUENCY DERIVITIVE */
  /**********************************************************************************/

  /* compute the grid parameters for all SFTs */
  if (XLALComputeFreqGridParamsVector(&freqgridparams,pspace.space,sftvec,uvar.mismatch)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALComputeFreqGridParams() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }

  /* free un-needed original SFT vector */
  XLALDestroySFTVector(sftvec);
  LogPrintf(LOG_DEBUG,"%s : Freed the SFT memory\n",fn);

  /**********************************************************************************/
  /* COMPUTE THE STATISTICS ON THE COARSE GRID */
  /**********************************************************************************/

  /* compute the statistic on the grid */
  if (XLALComputeDemodulatedPowerVector(&power,dstimevec,freqgridparams)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALComputeDemodulatedPower() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : computed the demodulated power\n",fn);
  
  /* fill in segment parameters */
  for (i=0;i<power->length;i++) {
    power->segment[i]->r = background->data[i];
    power->segment[i]->dt = segparams->dt->data[i];
    power->segment[i]->npcus = segparams->npcus->data[i];
    power->segment[i]->duration = pspace.tseg; 
  }

  /* free un-needed downsampled timeseries */
  for (i=0;i<dstimevec->length;i++) {
    XLALDestroyCOMPLEX8TimeSeries(dstimevec->data[i]);
  }
  XLALFree(dstimevec->data);
  XLALFree(dstimevec);
  LogPrintf(LOG_DEBUG,"%s : freed the downsampled timeseries memory\n",fn);

  /* free frequency grid - the contents of each segment have been moved to the power structure and are freed later */
  XLALFree(freqgridparams->segment);
  XLALFree(freqgridparams);

  /* free the background estimate */
  XLALDestroyREAL8Vector(background);
  LogPrintf(LOG_DEBUG,"%s : freed the background noise estimate\n",fn);

  /* free the segment params */
  XLALDestroyREAL8Vector(segparams->dt);
  XLALDestroyINT4Vector(segparams->npcus);
  XLALFree(segparams);
  LogPrintf(LOG_DEBUG,"%s : freed the segment parameters\n",fn);

  /**********************************************************************************/
  /* INTEGRATE OVER THE FINE GRID TO COMPUTE THE BAYES FACTOR */
  /**********************************************************************************/

  /* compute the Bayes factor and the posterior distributions */
  if (XLALComputeBayesFactor(&Bayes,power,&pspace,uvar.sigalpha)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALComputeBayesFactor() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : computed the BayesFactor and posteriors\n",fn);
 
  /**********************************************************************************/
  /* OUTPUT RESULTS TO FILE */
  /**********************************************************************************/

  if (XLALOutputBayesResults(uvar.outputdir,Bayes,&pspace,clargs)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALOutputBayesResults() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : output results to file.\n",fn);
 
  /**********************************************************************************/
  /* CLEAN UP */
  /**********************************************************************************/

  /* clean up the parameter space */
  if (XLALFreeParameterSpace(&pspace)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALFreeParameterSpace() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : freed the parameter space\n",fn);
 
  /* clean up the demodulated power */
  if (XLALFreeREAL4DemodulatedPowerVector(power)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALFreeREAL4DemodulatedPowerVector() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }  
  LogPrintf(LOG_DEBUG,"%s : freed the demodulated power\n",fn);

  /* clean up the demodulated power */
  if (XLALFreeBayesianProducts(Bayes)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALFreeBayesianProducts() failed with error = %d\n",fn,xlalErrno);
    return 1;
  }  
  LogPrintf(LOG_DEBUG,"%s : freed the Bayesian results\n",fn);

  /* Free config-Variables and userInput stuff */
  XLALDestroyUserVars();
  XLALFree(clargs);

  /* did we forget anything ? */
  LALCheckMemoryLeaks();
  LogPrintf(LOG_DEBUG,"%s : successfully checked memory leaks.\n",fn);

  LogPrintf(LOG_DEBUG,"%s : successfully completed.\n",fn);
  return 0;
  
} /* end of main */

/** Read in input user arguments
 *
 */
int XLALReadUserVars(int argc,            /**< [in] the command line argument counter */ 
		     char *argv[],        /**< [in] the command line arguments */
		     UserInput_t *uvar,   /**< [out] the user input structure */
		     CHAR **clargs        /**< [out] the command line args string */
		     )
{

  const CHAR *fn = __func__;   /* store function name for log output */
  CHAR *version_string;
  INT4 i;

  /* initialise user variables */
  uvar->sftbasename = NULL; 
  uvar->obsid_pattern = NULL;
  uvar->gpsstart = -1;
  uvar->gpsend = -1;
  uvar->mismatch = 0.2;
  uvar->nsig = 0;
  uvar->gaussianpriors = 0;
  uvar->seed = 0;

  /* initialise default source as SCOX1 */
  {
    UINT4 n = strlen(DEFAULT_SOURCE) + 1;
    uvar->source = XLALCalloc(n,sizeof(CHAR));
    snprintf(uvar->source,n,"%s",DEFAULT_SOURCE);
  }

  /* initialise all parameter space ranges to zero */
  uvar->freqband = 0;
  uvar->deltaorbperiod = 0.0;
  uvar->deltaasini = 0.0;
  uvar->deltatasc = 0.0;

  /* ---------- register all user-variables ---------- */
  XLALregBOOLUserStruct(help, 		        'h', UVAR_HELP,     "Print this message");
  XLALregSTRINGUserStruct(sftbasename, 	        'i', UVAR_REQUIRED, "The basename of the input SFT files"); 
  XLALregSTRINGUserStruct(outputdir, 	        'o', UVAR_REQUIRED, "The output directory name"); 
  XLALregSTRINGUserStruct(source, 	        'x', UVAR_OPTIONAL, "The source name (default SCOX1)"); 
  XLALregREALUserStruct(freq,                   'f', UVAR_REQUIRED, "The starting frequency (Hz)");
  XLALregREALUserStruct(freqband,   	        'b', UVAR_OPTIONAL, "The frequency band (Hz)");
  XLALregREALUserStruct(orbperiod,              'P', UVAR_REQUIRED, "The central orbital period value (sec)");
  XLALregREALUserStruct(deltaorbperiod,   	'p', UVAR_OPTIONAL, "The orbital period uncertainty (sec)");
  XLALregREALUserStruct(asini,                  'A', UVAR_REQUIRED, "The central orbital semi-major axis (sec)");
  XLALregREALUserStruct(deltaasini,       	'a', UVAR_OPTIONAL, "The orbital semi-major axis uncertainty (sec)");
  XLALregREALUserStruct(tasc,                   'T', UVAR_REQUIRED, "The central orbital time of ascension (GPS)");
  XLALregREALUserStruct(deltatasc,      	't', UVAR_OPTIONAL, "The orbital time of ascension uncertainty (GPS)");
  XLALregREALUserStruct(nsig,            	'g', UVAR_OPTIONAL, "The width of the Gaussian priors (in sigmas).  If unset flat 1-sig priors are used.");
  XLALregREALUserStruct(mismatch,        	'm', UVAR_OPTIONAL, "The grid mismatch (0->1)");
  XLALregREALUserStruct(sigalpha,        	'z', UVAR_REQUIRED, "The stdev of the zero mean amplitude prior in cnts/s/m^2");
  XLALregINTUserStruct(gpsstart,                's', UVAR_OPTIONAL, "The minimum start time (GPS sec)");
  XLALregINTUserStruct(gpsend,          	'e', UVAR_OPTIONAL, "The maximum end time (GPS sec)");
  XLALregSTRINGUserStruct(obsid_pattern,        'O', UVAR_OPTIONAL, "The observation ID substring to match"); 
  XLALregINTUserStruct(seed,                   'd', UVAR_SPECIAL,  "Fix the random number generator seed");
  XLALregREALUserStruct(inject_amplitude,     	'J', UVAR_SPECIAL,  "The amplitude of the injected signal (in cnts/s/m^2)");
  XLALregBOOLUserStruct(version,                'V', UVAR_SPECIAL,  "Output code version");

  /* do ALL cmdline and cfgfile handling */
  if (XLALUserVarReadAllInput(argc, argv)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALUserVarReadAllInput() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }

  /* if help was requested, we're done here */
  if (uvar->help) exit(0);

  if ((version_string = XLALGetVersionString(0)) == NULL) {
    XLALPrintError("XLALGetVersionString(0) failed.\n");
    exit(1);
  }
  
  if (uvar->version) {
    printf("%s\n",version_string);
    exit(0);
  }
  XLALFree(version_string);

  /* set priors flag if the sigma width has been set */
  if (XLALUserVarWasSet(&(uvar->nsig))) {
    uvar->gaussianpriors = 1;
    LogPrintf(LOG_DEBUG,"%s : using Gaussian priors on orbital parameters.\n",fn);
  }
  else uvar->nsig = 1.0;

  if (uvar->nsig < 0.0) {
    LogPrintf(LOG_CRITICAL,"%s : the user input nsig must > 0.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }

  /* put clargs into string */
  *clargs = XLALCalloc(1,sizeof(CHAR));
  for (i=0;i<argc;i++) {
    INT4 len = 2 + strlen(argv[i]) + strlen(*clargs);
    *clargs = XLALRealloc(*clargs,len*sizeof(CHAR));
    strcat(*clargs,argv[i]);
    strcat(*clargs," ");
  }

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

/** Computes the binary parameter space boundaries given the user input args
 *
 * For each search dimension we define the min, max, mid, and span of that dimension 
 * plus we give each dimension a name, define whether it is to be given a flat or 
 * Gaussian prior and specify the sigma of that prior.
 *
 */
int XLALDefineBinaryParameterSpace(REAL8Space **space,                 /**< [out] the parameter space  */ 
				   UserInput_t *uvar                   /**< [in] the user input variables */
				   )
{
  
  const CHAR *fn = __func__;   /* store function name for log output */
  
  /* validate input variables */
  if ((*space) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input REAL8Space boundary structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if (uvar == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input UserInput_t structure = NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  
  /* allocate memory for the parameter space */
  if ( ((*space) = XLALCalloc(1,sizeof(REAL8Space))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ( ((*space)->data = XLALCalloc(NBINMAX,sizeof(REAL8Dimension))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }

  /* this represents a hyper-cubic parameter space */
  /* we make sure that parameter ranges are consistent i.e asini > 0 etc.. */
  (*space)->ndim = NBINMAX;

  /* frequency */
  snprintf((*space)->data[0].name,LALNameLength,"nu");
  (*space)->data[0].min = uvar->freq;
  (*space)->data[0].max = uvar->freq + uvar->freqband;
  (*space)->data[0].mid = uvar->freq + 0.5*uvar->freqband;
  (*space)->data[0].sig = 0.5*uvar->freqband;
  (*space)->data[0].span = uvar->freqband;
  (*space)->data[0].gaussian = 0;

  /* asini */
  snprintf((*space)->data[1].name,LALNameLength,"asini");
  (*space)->data[1].min = uvar->asini - fabs(uvar->nsig*uvar->deltaasini) > 0.0 ? uvar->asini - fabs(uvar->nsig*uvar->deltaasini) : 0.0;
  (*space)->data[1].max = uvar->asini + fabs(uvar->nsig*uvar->deltaasini);
  (*space)->data[1].mid = uvar->asini;
  (*space)->data[1].sig = uvar->deltaasini;
  (*space)->data[1].span = (*space)->data[1].max - (*space)->data[1].min;
  (*space)->data[1].gaussian = uvar->gaussianpriors;
  
  /* tasc */
  snprintf((*space)->data[2].name,LALNameLength,"tasc");
  (*space)->data[2].min = uvar->tasc - fabs(uvar->nsig*uvar->deltatasc);
  (*space)->data[2].max = uvar->tasc + fabs(uvar->nsig*uvar->deltatasc);
  (*space)->data[2].mid = uvar->tasc;
  (*space)->data[2].sig = uvar->deltatasc;
  (*space)->data[2].span = (*space)->data[2].max - (*space)->data[2].min;
  (*space)->data[2].gaussian = uvar->gaussianpriors;
  
  /* omega */
  snprintf((*space)->data[3].name,LALNameLength,"omega");
  (*space)->data[3].min = LAL_TWOPI/(uvar->orbperiod + fabs(uvar->nsig*uvar->deltaorbperiod));
  (*space)->data[3].max = LAL_TWOPI/(uvar->orbperiod - fabs(uvar->nsig*uvar->deltaorbperiod));
  (*space)->data[3].mid = LAL_TWOPI/uvar->orbperiod;
  (*space)->data[3].sig = (*space)->data[3].mid - (*space)->data[3].min;
  if ((*space)->data[3].max < 0.0) {
    LogPrintf(LOG_CRITICAL,"%s: max boundary on omega is < 0.  Exiting.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  (*space)->data[3].gaussian = uvar->gaussianpriors;
  (*space)->data[3].span = (*space)->data[3].max - (*space)->data[3].min;

  /* output boundaries to screen */
  if (uvar->gaussianpriors) LogPrintf(LOG_DEBUG,"%s : using Gaussian priors on the following %.2f sigma ranges (except nu)\n",fn,uvar->nsig);
  else LogPrintf(LOG_DEBUG,"%s : using flat priors on the following ranges\n",fn);
  LogPrintf(LOG_DEBUG,"%s : parameter space, %s = [%e -> %e]\n",fn,(*space)->data[0].name,(*space)->data[0].min,(*space)->data[0].max);
  LogPrintf(LOG_DEBUG,"%s : parameter space, %s = [%e -> %e]\n",fn,(*space)->data[1].name,(*space)->data[1].min,(*space)->data[1].max);
  LogPrintf(LOG_DEBUG,"%s : parameter space, %s = [%e -> %e]\n",fn,(*space)->data[2].name,(*space)->data[2].min,(*space)->data[2].max);
  LogPrintf(LOG_DEBUG,"%s : parameter space, %s = [%e -> %e]\n",fn,(*space)->data[3].name,(*space)->data[3].min,(*space)->data[3].max);

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

/** Compute the prior probability density functions on the search parameters
 *
 * The priors are computed on the search grid and correctly normalised
 *
 */
int XLALComputeBinaryPriors(REAL8PriorsVector **priors,        /**< [out] the priors on each search parameter */
			    REAL8Space *space,                /**< [in] the parameter space */
			    GridParameters *gridparams        /**< [in] the grid on this parameter space */
			    )
{
  const CHAR *fn = __func__;   /* store function name for log output */
  UINT4 i,j;                   /* counters */

  /* validate input */
  if ((*priors) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input REAL8PriorsVector structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if (space == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input REAL8Space structure = NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if (gridparams == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input GridParameters structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  
  /* allocate memory for the priors */
  if (((*priors) = (REAL8PriorsVector*)XLALCalloc(1,sizeof(REAL8PriorsVector))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if (((*priors)->data = (REAL8Priors *)XLALCalloc(gridparams->ndim,sizeof(REAL8Priors))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  for (i=0;i<gridparams->ndim;i++) {
    if (((*priors)->data[i].logpriors = XLALCreateREAL8Vector(gridparams->grid[i].length)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s : XLALCrateREAL8Vector() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
  }
  
  /* loop over each dimension */
  for (i=0;i<gridparams->ndim;i++) {
       
    /* if we're using Gaussian priors on this parameter */
    if (space->data[i].gaussian) {
      
      REAL8 x0 = space->data[i].mid;
      REAL8 sig = space->data[i].sig;
      LogPrintf(LOG_DEBUG,"%s : computing Gaussian priors for parameter %s\n",fn,space->data[i].name);
      
      /* account for single template situations, i.e known parameters */ 
      if (gridparams->grid[i].length>1) {
	for (j=0;j<gridparams->grid[i].length;j++) {
	  REAL8 x = gridparams->grid[i].min + j*gridparams->grid[i].delta;
	  REAL8 norm = (-0.5)*log(LAL_TWOPI) - log(sig);
	  (*priors)->data[i].logpriors->data[j] = norm - 0.5*pow((x-x0)/sig,2.0);
	}
	(*priors)->data[i].logdelta = log(gridparams->grid[i].delta);
      }
      else {
	(*priors)->data[i].logpriors->data[0] = 0.0;
	(*priors)->data[i].logdelta = 0.0;
      }
      
    }
    /* otherwise we use flat priors */
    else {
      
      LogPrintf(LOG_DEBUG,"%s : computing Flat priors for parameter %s\n",fn,space->data[i].name);
      
      /* set flat prior such that for a single template the prior has no effect once multiplied by deltax */
      /* account for single template situations, i.e known parameters */ 
      if (gridparams->grid[i].length>1) {
	for (j=0;j<gridparams->grid[i].length;j++) {
	  REAL8 flat = log(1.0/(gridparams->grid[i].delta*gridparams->grid[i].length));
	  (*priors)->data[i].logpriors->data[j] = flat;
	}
	(*priors)->data[i].logdelta = log(gridparams->grid[i].delta);
      }
      else {
	(*priors)->data[i].logpriors->data[0] = 0.0;
	(*priors)->data[i].logdelta = 0.0;
      }
      
    }
       
    /* record gaussian flag */
    (*priors)->data[i].gaussian = space->data[i].gaussian;

  }

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

/** Adds a simulated signal to the existing SFTs
 *
 * The parameters of the injection are drawn from the binary parameter priors
 * with the exception of the amplitude.  We cannot do this exactly since the 
 * signal is not additive but we approximate this by drawing the signal component
 * from a Poisson distribution and adding it to the existing noise.
 *
 */
int XLALAddBinarySignalToSFTVector(SFTVector **sftvec,           /**< [in/out] the input SFTs into which we add a signal */ 
				   ParameterSpace *pspace,       /**< [in] the parameter space */
				   REAL8 inject_amplitude,       /**< [in] the injection amplitude in cnts/s/m^2 */
				   INT4 seed                     /**< [in] the random number seed */
				   )
{
  const CHAR *fn = __func__;            /* store function name for log output */
  COMPLEX8FFTPlan *plan = NULL;         /* FFT plan */
  COMPLEX8Vector *xt = NULL;            /* the downsampled timeseries */
  COMPLEX8Vector *xf = NULL;            /* the fft'd timeseries */
  REAL8 nu,a,tasc,omega,phi0;           /* randomly selected injection parameters */
  UINT4 i,j;                            /* counters */

  /* allocate memory for the injection parameters */
  if ((pspace->inj = XLALCalloc(1,sizeof(InjectionParameters))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ((pspace->inj->temp.x = XLALCalloc(pspace->space->ndim,sizeof(REAL8))) == NULL)  {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  pspace->inj->temp.ndim = pspace->space->ndim;

  /* randomly select values from the priors */
  {
    gsl_rng * r;
    REAL8PriorsVector *priors = pspace->priors;
    REAL8Space *space = pspace->space;

    /* initialise the random number generator */
    if (XLALInitgslrand(&r,seed)) {
      LogPrintf(LOG_CRITICAL,"%s: XLALinitgslrand() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_EFAULT);
    }
    
    if (priors->data[0].gaussian) nu = space->data[0].mid + gsl_ran_gaussian(r,space->data[0].sig);
    else nu = gsl_ran_flat(r,space->data[0].min,space->data[0].max);
    if (priors->data[0].gaussian) a = space->data[1].mid + gsl_ran_gaussian(r,space->data[1].sig);
    else a = gsl_ran_flat(r,space->data[1].min,space->data[1].max);
    if (priors->data[0].gaussian) tasc = space->data[2].mid + gsl_ran_gaussian(r,space->data[2].sig);
    else tasc = gsl_ran_flat(r,space->data[2].min,space->data[2].max);
    if (priors->data[0].gaussian) omega = space->data[3].mid + gsl_ran_gaussian(r,space->data[3].sig);
    else omega = gsl_ran_flat(r,space->data[3].min,space->data[3].max);

    LogPrintf(LOG_DEBUG,"%s: the injected signal parameters are :\n",fn);
    LogPrintf(LOG_DEBUG,"%s: injected nu = %6.12f\n",fn,nu);
    LogPrintf(LOG_DEBUG,"%s: injected asini = %6.12f\n",fn,a);
    LogPrintf(LOG_DEBUG,"%s: injected tasc = %6.12f\n",fn,tasc);
    LogPrintf(LOG_DEBUG,"%s: injected omega = %6.12e\n",fn,omega);
     
    /* nuiseance parameters */
    phi0 = gsl_ran_flat(r,0,LAL_TWOPI);

    gsl_rng_free (r);
  }

  {
   
    /* define some parameters common to all SFTs */
    REAL8 tsft = 1.0/(*sftvec)->data[0].deltaF;
    UINT4 numBins = (*sftvec)->data[0].data->length;
    REAL8 f0 = (*sftvec)->data[0].f0;
    REAL8 deltaf = (*sftvec)->data[0].deltaF;
    REAL8 fhet = f0 + deltaf*floor((numBins+1)/2);
    REAL8 deltat = tsft/numBins;
    REAL8 tref = 0.0;

    /* allocate memory for the timeseries and the frequency domain output */
    if ( (xt = XLALCreateCOMPLEX8Vector(numBins)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateCOMPLEX8Vector() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    if ( (xf = XLALCreateCOMPLEX8Vector(numBins)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateCOMPLEX8Vector() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    
    /* make the reverse plan */
    if ((plan = XLALCreateForwardCOMPLEX8FFTPlan(numBins,1)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateCOMPLEX8FFTPlan() failed with error = %d\n",fn,xlalErrno);
      return XLAL_EINVAL;
    }
    LogPrintf(LOG_DEBUG,"%s : created the COMPLEX8 FFT plan for signal injection\n",fn);
    
    /* we generate a down-sampled time series for each SFT */
    for (i=0;i<(*sftvec)->length;i++) {
      
      /* initialise input and output */
      memset(xt->data,0.0,numBins*sizeof(COMPLEX8));
      memset(xf->data,0.0,numBins*sizeof(COMPLEX8));
      
      /* generate the heterodyned timeseries */
      for (j=0;j<numBins;j++) {
	REAL8 t = j*deltat + XLALGPSGetREAL8(&((*sftvec)->data[i].epoch));
	
	REAL8 phase = phi0 + LAL_TWOPI*((t-tref)*(nu-fhet) - nu*a*sin(omega*(t-tasc))) - LAL_PI/2.0;
	xt->data[j].re = 0.5*inject_amplitude*cos(phase)*deltat;
	xt->data[j].im = 0.5*inject_amplitude*sin(phase)*deltat;
      }
      
      /* perform fft */
      if (XLALCOMPLEX8VectorFFT(xf,xt,plan)) {
	LogPrintf(LOG_CRITICAL,"%s: XLALCOMPLEX8VectorFFT() failed with error = %d\n",fn,xlalErrno);
	return XLAL_EINVAL;
      }
      LogPrintf(LOG_DEBUG,"%s : performed %d/%d FFT for signal injection\n",fn,i+1,(*sftvec)->length);
    
      /* now add directly to the input sft - making sure to flip negative frequencies */
      for (j=0;j<(UINT4)floor((numBins+1)/2);j++) {
	(*sftvec)->data[i].data->data[j].re += xf->data[j+(UINT4)floor((numBins)/2)].re;
	(*sftvec)->data[i].data->data[j].im += xf->data[j+(UINT4)floor((numBins)/2)].im;
      }
      for (j=(UINT4)floor((numBins+1)/2);j<numBins;j++) {
	(*sftvec)->data[i].data->data[j].re += xf->data[j-(UINT4)floor((numBins+1)/2)].re;
	(*sftvec)->data[i].data->data[j].im += xf->data[j-(UINT4)floor((numBins+1)/2)].im;
      }
      
      /* TESTING */
     /*  { */
/* 	FILE *fp = NULL; */
/* 	CHAR name[256]; */
/* 	sprintf(name,"/home/chmess/temp/injtest_%d.txt",i); */
/* 	fp = fopen(name,"w"); */
/* 	for (j=0;j<(UINT4)floor((numBins+1)/2);j++) { */
/* 	  printf("i = %d j = %d\n",j,j+(UINT4)floor((numBins)/2)); */
/* 	  fprintf(fp,"%6.12f %6.12f %6.12f\n",f0+j*deltaf,xf->data[j+(UINT4)floor((numBins)/2)].re,xf->data[j+(UINT4)floor((numBins)/2)].im); */
/* 	} */
/* 	for (j=(UINT4)floor((numBins+1)/2);j<numBins;j++) { */
/* 	  printf("i = %d j = %d\n",j,j-(UINT4)floor((numBins+1)/2)); */
/* 	  fprintf(fp,"%6.12f %6.12f %6.12f\n",f0+j*deltaf,xf->data[j-(UINT4)floor((numBins+1)/2)].re,xf->data[j-(UINT4)floor((numBins+1)/2)].im); */
/* 	} */
/* 	fclose(fp); */
/*       } */
     
    }
    
  }

  /* fill in injection parameters */
  pspace->inj->amp = inject_amplitude;
  pspace->inj->temp.x[0] = nu;
  pspace->inj->temp.x[1] = a;
  pspace->inj->temp.x[2] = tasc;
  pspace->inj->temp.x[3] = omega;

  /* free memeory */
  XLALDestroyCOMPLEX8FFTPlan(plan);
  XLALDestroyCOMPLEX8Vector(xt);
  XLALDestroyCOMPLEX8Vector(xf);
  
  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

/** Read in SFTs to an SFTVector
 *
 */
int XLALReadSFTs(SFTVector **sftvec,        /**< [out] the input SFT data */
		 SegmentParams **segparams, /**< [out] the segment parameters (noise, sampling time, etc..) */
		 CHAR *sftbasename,         /**< [in] the SFT file basename to read in */
		 REAL8 freq,                /**< [in] the starting frequency to read in */
		 REAL8 freqband,            /**< [in] the bandwidth to read */
		 INT4 start,                /**< [in] the min GPS time of the input data */
		 INT4 end,                  /**< [in] the max GPS time of the input data*/
		 CHAR *obsid_pattern        /**< [in] the OBS-ID pattern */
  		 )
{

  const CHAR *fn = __func__;   /* store function name for log output */
  static SFTConstraints constraints;
  SFTCatalog *catalog = NULL;
  SFTCatalog *newcat = NULL;
  INT4 sft_check_result = 0;
  REAL8 freqmin,freqmax;
  LIGOTimeGPS *dummy_gpsstart = NULL;
  LIGOTimeGPS *dummy_gpsend = NULL;
  LIGOTimeGPS gpsstart, gpsend;
  UINT4 i;                                    /* counters */
  INT4 count = 0;
  LALStatus status = blank_status;              /* for use wih non-XLAL functions */
  
  /* validate input variables */
  if (*sftvec != NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input SFTVector structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if (*segparams != NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input INT4Vector structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if (sftbasename == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, input SFT basename string == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if (freqband < 0 ) {
    LogPrintf(LOG_CRITICAL,"%s : Invalid input, frequency band must be > 0.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }  
  if ((start > 0) && (end > 0)) {
    if (start - end >= 0) {
      LogPrintf(LOG_CRITICAL,"%s: Invalid input, the start time %d >= %d end time.\n",fn,start,end);
      XLAL_ERROR(fn,XLAL_EINVAL);
    }
  }

  /* get sft catalog */
  /* if the input gps times are negative i.e. not set, then we pass null pointers to LALLALSFTDataFind */
  if ((start > 0) && (obsid_pattern == NULL)) {
    XLALGPSSetREAL8(&gpsstart,(REAL8)start);
    dummy_gpsstart = &gpsstart;
  }
  if ((end > 0) && (obsid_pattern == NULL)) {
    XLALGPSSetREAL8(&gpsend,(REAL8)end);
    dummy_gpsend = &gpsend;
  }  
  constraints.startTime = dummy_gpsstart;
  constraints.endTime = dummy_gpsend;
  LAL_CALL( LALSFTdataFind( &status, &catalog, sftbasename, &constraints), &status);
  LogPrintf(LOG_DEBUG,"%s : found %d SFTs\n",fn,catalog->length);
  
  /* define actual frequency range to read in */
  freqmin = freq;
  freqmax = freqmin + freqband;

  /* allocate memory for the temporary catalog */
  if (obsid_pattern != NULL) { 
    if ( (newcat = XLALCalloc(1,sizeof(SFTCatalog))) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    if ( (newcat->data = (SFTDescriptor *)XLALCalloc(catalog->length,sizeof(SFTDescriptor))) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    newcat->length = 0;
   
    /* manually restrict the catalog to the correct OBS ID pattern */
    /* we also check the frequency range manually because it's a lot quicker than the LAL functions */
    for (i=0;i<catalog->length;i++) {
      
      CHAR *c = NULL;
      CHAR *s_obsid,*e_obsid;
      CHAR obsid_string[STRINGLENGTH];
      REAL8 sft_fmin = catalog->data[i].header.f0;
      REAL8 sft_fmax = sft_fmin + catalog->data[i].numBins*catalog->data[i].header.deltaF;

      if ((c = strstr(catalog->data[i].comment,"Additional comment")) == NULL) {
	LogPrintf(LOG_CRITICAL,"%s: Error, couldn't find required header information in SFT files.\n",fn);
	XLAL_ERROR(fn,XLAL_EINVAL);
      }
      
      /* extract the OBS ID string from the comment field */
      s_obsid = strstr(c,"OBS_ID") + 9;
      e_obsid = strstr(s_obsid,"\n");
      snprintf(obsid_string,strlen(s_obsid) - strlen(e_obsid) + 1,"%s",s_obsid);

      /* if the obsid is not consistent with the requested pattern then we ignore this SFT */
      if (strstr(obsid_string,obsid_pattern) != NULL) {
	if (!((sft_fmin>freqmax) || (sft_fmax<freqmin))) {
	  memcpy(&(newcat->data[newcat->length]),&(catalog->data[i]),sizeof(SFTDescriptor));
	  newcat->length++;
	}
      }
      
    }
    LogPrintf(LOG_DEBUG,"%s : found %d SFTs matching %s pattern\n",fn,newcat->length,obsid_pattern);
  }
  else newcat = catalog;
  
  /* check CRC sums of SFTs */
  LAL_CALL ( LALCheckSFTCatalog ( &status, &sft_check_result, newcat ), &status );
  if (sft_check_result) {
    LogPrintf(LOG_CRITICAL,"%s : LALCheckSFTCatalogSFT() validity check failed with error = %d\n", sft_check_result);
    return 1;
  }
  LogPrintf(LOG_DEBUG,"%s : checked the SFTs\n",fn);

  /* allocate memory for the output pcu vector (allocate full possible length) */
  if ( ((*segparams) = XLALCalloc(1,sizeof(SegmentParams))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }

  /* allocate memory for the output pcu and dt vectors (allocate full possible length) */
  if ( ((*segparams)->npcus = XLALCreateINT4Vector(catalog->length)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCreateINT4Vector() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ( ((*segparams)->dt = XLALCreateREAL8Vector(catalog->length)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCreateREAL8Vector() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  
  /* load the SFT-vectors */
  LAL_CALL( LALLoadSFTs ( &status, sftvec, newcat, freqmin, freqmax ), &status);
  LogPrintf(LOG_DEBUG,"%s : loaded the sfts\n",fn);

  /* associate each SFT with the corresponding number of PCUs, sampling time and OBS ID */
  for (i=0;i<(*sftvec)->length;i++) {

    /* find the corresponding catalog extry and extract the number of PCUs, sampling time and OBS ID */
    UINT4 j = 0;
    INT4 notfound = 1;
    while ((j<newcat->length) && notfound ) {
      
      /* if the epoch matches the catalog epoch then get info from catalog comment field */
      if (XLALGPSCmp(&(newcat->data[j].header.epoch),&((*sftvec)->data[i].epoch)) == 0) {
	CHAR *c = NULL;
	CHAR *s_pcus,*s_dt,*e_dt;
	CHAR npcus_string[STRINGLENGTH];
	CHAR dt_string[STRINGLENGTH];
	
	if ((c = strstr(newcat->data[j].comment,"Additional comment")) == NULL) {
	  LogPrintf(LOG_CRITICAL,"%s: Error, couldn't find required header information in SFT files.\n",fn);
	  XLAL_ERROR(fn,XLAL_EINVAL);
	}
	
	/* extract the relavent strings from the comment field */
	s_pcus = strstr(c,"NPCUS") + 8;	
	snprintf(npcus_string,2,"%s",s_pcus);
	s_dt = strstr(c,"DELTAT") + 9;
	e_dt = strstr(s_dt,",");
	snprintf(dt_string,strlen(s_dt) - strlen(e_dt) + 1,"%s",s_dt);
	(*segparams)->npcus->data[count] = (INT4)atoi(npcus_string);
	(*segparams)->dt->data[count] = (REAL8)atof(dt_string);
	
	/* unset the not found flag and increment the count */
	notfound = 0;
	count++;
      }
      /* increment the count over catalog entries */
      j++;
    }

  }
 
  /* resize the segment params vectors */
  if ( ((*segparams)->npcus = XLALResizeINT4Vector((*segparams)->npcus,count)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALResizeINT4Vector() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ( ((*segparams)->dt = XLALResizeREAL8Vector((*segparams)->dt,count)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALResizeREAL8Vector() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
 
  /* we don't need the original catalog anymore -  also free the new catalog */
  LAL_CALL( LALDestroySFTCatalog( &status, &catalog ), &status);
  if (obsid_pattern) {
    XLALFree(newcat->data);
    XLALFree(newcat);
  }
  LogPrintf(LOG_DEBUG,"%s : destroyed the catalogue(s)\n",fn);
  
  /* check if we found any SFTs */
  if ((*sftvec)->length == 0) {
    LogPrintf(LOG_CRITICAL,"%s : No SFTs found in specified frequency range.  Exiting.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Inverse FFT all narrowband SFTs 
 *
 * In order to apply the frequency derivitive corrections we must work in the time domain
 * so here we convert all SFTs to the complex time domain.
 *
 */
int XLALSFTVectorToCOMPLEX8TimeSeriesArray(COMPLEX8TimeSeriesArray **dstimevec,      /**< [out] the downsampled timeseries */
					   SFTVector *sftvec                        /**< [in] the input SFT vector */
					   )
{

  const CHAR *fn = __func__;             /* store function name for log output */
  INT4 i;                                /* counter */
  COMPLEX8FFTPlan *plan = NULL;         /* inverse FFT plan */
  UINT4 N;                               /* the length of the SFTs */

  /* validate input arguments */
  if ((*dstimevec) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output COMPLEX8TimeSeriesArray structure != NULL.\n",fn);
    return XLAL_EINVAL;
  }
  if (sftvec == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input SFTVector structure == NULL.\n",fn);
    return XLAL_EINVAL;
  }

  /* we check that all input SFTs are of identical length so we make a single plan */
  N = sftvec->data[0].data->length;
  for (i=0;i<(INT4)sftvec->length;i++) {
    if (sftvec->data[0].data->length != N) {
      LogPrintf(LOG_CRITICAL,"%s: Invalid input, input SFTs have different lengths.\n",fn);
      return XLAL_EINVAL;
    }
  }
  LogPrintf(LOG_DEBUG,"%s : checked that all SFTs have length %d.\n",fn,N);

  /* make the reverse plan */
  if ((plan = XLALCreateReverseCOMPLEX8FFTPlan(N,1)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCreateReverseCOMPLEX8FFTPlan() failed with error = %d\n",fn,xlalErrno);
    return XLAL_EINVAL;
  }
  LogPrintf(LOG_DEBUG,"%s : created the inverse FFT plan\n",fn);

  /* allocate memory for output */
  if (((*dstimevec) = (COMPLEX8TimeSeriesArray*)XLALCalloc(1,sizeof(COMPLEX8TimeSeriesArray))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for a COMPLEX8TimeSeriesArray structure\n",fn,xlalErrno);
    return XLAL_ENOMEM;
  }
  if (((*dstimevec)->data = (COMPLEX8TimeSeries**)XLALCalloc(sftvec->length,sizeof(COMPLEX8TimeSeries *))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for a vector of COMPLEX8TimeSeries pointers\n",fn,xlalErrno);
    return XLAL_ENOMEM;
  }
  (*dstimevec)->length = sftvec->length;
  LogPrintf(LOG_DEBUG,"%s : allocated memory for the output data structure\n",fn);

  /* loop over each SFT */
  for (i=0;i<(INT4)sftvec->length;i++) {
  
    COMPLEX8Vector temp_output;
    REAL8 Tsft = 1.0/sftvec->data[i].deltaF;
    REAL8 deltaT = Tsft/N;

    /* point to input */
    COMPLEX8Vector temp_input;
    temp_input.length = sftvec->data[i].data->length;
    temp_input.data = (COMPLEX8*)sftvec->data[i].data->data;
   
    /* allocate output memory - create a COMPLEX8TimeSeries */
    if (((*dstimevec)->data[i] = XLALCreateCOMPLEX8TimeSeries("DS",&(sftvec->data[i].epoch),sftvec->data[i].f0,deltaT,&lalDimensionlessUnit,N)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateCOMPLEX8TimeSeries() failed to allocate memory for inverse FFT output.\n",fn);
      return XLAL_ENOMEM;
    }
    LogPrintf(LOG_DEBUG,"%s : allocated memory for the %d/%d inverse FFT\n",fn,i+1,sftvec->length);

    /* point temp output to timeseries */
    temp_output.length = N;
    temp_output.data = (COMPLEX8*)(*dstimevec)->data[i]->data->data;

    /* perform inverse FFT */
    if (XLALCOMPLEX8VectorFFT(&temp_output, &temp_input, plan)) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCOMPLEX16VectorFFT() failed with error = %d\n",fn,xlalErrno);
      return XLAL_EINVAL;
    }
    LogPrintf(LOG_DEBUG,"%s : performed %d/%d inverse FFT\n",fn,i+1,sftvec->length);

  }
  LogPrintf(LOG_DEBUG,"%s : performed inverse FFT on all %d SFTs\n",fn,sftvec->length);

  /* free memeory */
  XLALDestroyCOMPLEX8FFTPlan(plan);

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Compute the gridding parameters on spin derivitives for all segments
 *
 * This is simply a wrapper for the single segment function 
 *
 */
int XLALComputeFreqGridParamsVector(GridParametersVector **freqgridparams,    /**< [out] the gridding parameters */
				    REAL8Space *space,                        /**< [in] the orbital parameter space */
				    SFTVector *sftvec,                        /**< [in] the input SFTs */
				    REAL8 mu                                  /**< [in] the required mismatch */ 
				    )
{
  
  const CHAR *fn = __func__;            /* store function name for log output */
  UINT4 i;                              /* counter */

  /* validate input arguments */
  if ((*freqgridparams) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output GridParamsVector structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (space == NULL) {
     LogPrintf(LOG_CRITICAL,"%s: Invalid input, input REAL8Space structure == NULL.\n",fn);
     XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (sftvec == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input SFTVector structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if ((mu < 0)||(mu>1)) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input mismatch parameter, not in range 0 -> 1.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  
  /* allocate memory for each set of grid parameters */
  if (((*freqgridparams) = (GridParametersVector*)XLALCalloc(1,sizeof(GridParametersVector))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() falied with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if (((*freqgridparams)->segment = (GridParameters**)XLALCalloc(sftvec->length,sizeof(GridParameters*))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for a COMPLEX8TimeSeriesArray structure\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  (*freqgridparams)->length = sftvec->length;

  /* loop over each SFT */
  for (i=0;i<sftvec->length;i++) {
    
    REAL8 t0 = XLALGPSGetREAL8(&(sftvec->data[i].epoch));
    REAL8 tsft = 1.0/sftvec->data[i].deltaF;
    REAL8 tmid = t0 + 0.5*tsft;

    if (XLALComputeFreqGridParams(&((*freqgridparams)->segment[i]),space,tmid,tsft,mu)) {
      LogPrintf(LOG_CRITICAL,"%s: XLALComputeFreqGridParams() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_EINVAL);
    }
    LogPrintf(LOG_DEBUG,"%s : computed frequency grid for SFT %d/%d\n",fn,i+1,sftvec->length);
    
  }

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Compute the gridding parameters on spin derivitives
 *
 * The circular orbit binary phase model is phi = 2*pi*nu*( (t-tref) - a*sin( W*(t-tasc) )
 * from which we compute the min and maximum instantaneous spin derivitives.
 *
 */
int XLALComputeFreqGridParams(GridParameters **gridparams,              /**< [out] the gridding parameters */
			      REAL8Space *space,                        /**< [in] the orbital parameter space */
			      REAL8 tmid,                               /**< [in] the segment mid point */
			      REAL8 Tseg,                               /**< [in] the segment length */
			      REAL8 mu                                  /**< [in] the required mismatch */ 
			      )
{
  
  const CHAR *fn = __func__;            /* store function name for log output */
  UINT4 i,j,k,l;                         /* counters */
  INT4 n;                                /* counter */
 /*  REAL8 nu,asini,omega,tasc;    */          /* temporary orbital parameters */
 /*  REAL8 fn[NFREQMAX];    */                 /* temporary values of spin derivitives */
  REAL8 fnmin[NFREQMAX],fnmax[NFREQMAX]; /* min and max values of spin derivitives */
  INT4 dim[NFREQMAX];                    /* flag indicating whether a dimension has width */
  INT4 ndim = -1;                        /* the number of spin derivitive dimensions required */      
  Template fdots;                        /* template for an instance of spin parameters */
  Template bintemp;                      /* template for instance of binary parameters */              
  UINT4 ngrid = 100;                     /* the number of grid points per omega and tasc to search */

  /* validate input arguments */
  if ((*gridparams) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output GridParams structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (space == NULL) {
     LogPrintf(LOG_CRITICAL,"%s: Invalid input, input ParameterSpace structure == NULL.\n",fn);
     XLAL_ERROR(fn,XLAL_EINVAL);
   }
   if (tmid < 0) {
     LogPrintf(LOG_CRITICAL,"%s: Invalid input, input GPS time < 0.\n",fn);
     XLAL_ERROR(fn,XLAL_EINVAL);
   }
   if (Tseg < 0) {
     LogPrintf(LOG_CRITICAL,"%s: Invalid input, input Tseg parameter < 0.\n",fn);
     XLAL_ERROR(fn,XLAL_EINVAL);
   }
   if ((mu < 0)||(mu>1)) {
     LogPrintf(LOG_CRITICAL,"%s: Invalid input, input mismatch parameter, not in range 0 -> 1.\n",fn);
     XLAL_ERROR(fn,XLAL_EINVAL);
   }

   /* allocte memory */
   if (((*gridparams) = (GridParameters*)XLALCalloc(1,sizeof(GridParameters))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }

   /* allocate memory for the fdots */
   if ((fdots.x = XLALCalloc(NFREQMAX,sizeof(REAL8))) == NULL) {
     LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
     XLAL_ERROR(fn,XLAL_ENOMEM);
   }
   if ((bintemp.x = XLALCalloc(NBINMAX,sizeof(REAL8))) == NULL) {
     LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
     XLAL_ERROR(fn,XLAL_ENOMEM);
   }
   fdots.ndim = NFREQMAX;
   bintemp.ndim = NBINMAX;
   
   /* initialise the min and max spin derivitives */
   for (n=0;n<NFREQMAX;n++) {
     fnmin[n] = 1e38;
     fnmax[n] = -1e38;
   }

   /* loop over each parameter in turn and compute the spin derivitives at the corners of the parameter space */
   for (i=0;i<2;i++) {     /* nu */
     if (i==0) bintemp.x[0] = space->data[0].min;
     else bintemp.x[0] = space->data[0].max;
     
     for (j=0;j<2;j++) {    /* a */
       if (j==0) bintemp.x[1] = space->data[1].min;
       else bintemp.x[1] = space->data[1].max;
       
       /* tasc and omega are the problematic ones so we'll perform a fine grid search over them */
       for (k=0;k<ngrid;k++) {   /* tasc */
	 bintemp.x[2] = space->data[2].min + k*(space->data[2].max-space->data[2].min)/(ngrid-1);
	 
	 for (l=0;l<ngrid;l++) {  /* omega */
	   bintemp.x[3] = space->data[3].min + l*(space->data[3].max-space->data[3].min)/(ngrid-1);
	   
	   if (XLALComputeBinaryFreqDerivitives(&fdots,&bintemp,tmid)) {
	     LogPrintf(LOG_CRITICAL,"%s : XLALComputeBinaryFreqDerivitives() failed with error = %d\n",fn,xlalErrno);
	     XLAL_ERROR(fn,XLAL_EFAULT);
	   }

	 /*   /\* the instantanous frequency is therefore f0 = nu - a*nu*W*cos(W*(t-tasc) ) *\/ */
/* 	   fn[0] = nu - asini*nu*omega*cos(omega*(tmid-tasc)); */
	   
/* 	   /\* the instantanous nth frequency derivitive is therefore fn = - a * nu * W^(n+1) * cos ( W*(t-tasc) + n*pi/2 ) *\/ */
/* 	   for (n=1;n<NFREQMAX;n++) { */
/* 	     fn[n] = (-1.0)*asini*nu*pow(omega,n+1)*cos( omega*(tmid-tasc) + 0.5*n*LAL_PI); */
/* 	   } */

	   /* find min and max values */
	   for (n=0;n<NFREQMAX;n++) {
	     if (fdots.x[n] < fnmin[n]) fnmin[n] = fdots.x[n];
	     if (fdots.x[n] > fnmax[n]) fnmax[n] = fdots.x[n];
	   }

	 }

       }

     }

   }
   for (n=0;n<NFREQMAX;n++) {
     LogPrintf(LOG_DEBUG,"%s : determined f%d range as [%6.12e -> %6.12e].\n",fn,n,fnmin[n],fnmax[n]);
   }
   LogPrintf(LOG_DEBUG,"%s : midpoint epoch for this SFT is %6.12f\n",fn,tmid);

   /* free templates */
   XLALFree(fdots.x);
   XLALFree(bintemp.x);

   /* compute the required dimensionality of the frequency derivitive grid */
   /* we check the width of a 1-D template across each dimension span */ 
   for (n=0;n<NFREQMAX;n++) {
     REAL8 gnn = pow(LAL_PI,2.0)*pow(Tseg,2*n+2)/(pow(2.0,2*n)*(2*n+3.0));
     REAL8 deltafn = 2.0*sqrt(mu/gnn);
     REAL8 span = fnmax[n] - fnmin[n];
     dim[n] = 0;
     if (span > deltafn) dim[n] = 1;
     LogPrintf(LOG_DEBUG,"%s : single template span for %d'th derivitive = %e.\n",fn,n,deltafn);
   }
   n = NFREQMAX-1;
   while ( (n>=0) && (ndim == -1) ) {
     if (dim[n] > 0) ndim = n+1;
     n--;
   }
   if (ndim < 0) {
      LogPrintf(LOG_CRITICAL,"%s: dimensionality of frequency space < 0.  No templates required.\n",fn);
      return XLAL_EINVAL;
   }
   LogPrintf(LOG_DEBUG,"%s : determined dimensionality of frequency space = %d.\n",fn,ndim);

   /* allocate memory to the output */
   if ( ((*gridparams)->grid = XLALCalloc(ndim,sizeof(Grid))) == NULL) {
     LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for gridparams->grid.\n",fn);
     return XLAL_ENOMEM;
   }
   (*gridparams)->ndim = ndim;
   LogPrintf(LOG_DEBUG,"%s : allocated memory for the output grid parameters.\n",fn);

   /* Compute the grid spacing, grid start and span for each spin derivitive dimension */
   for (n=0;n<ndim;n++) {

     /* compute diagonal metric element and corresponding spacing */
     REAL8 gnn = pow(LAL_PI,2.0)*pow(Tseg,2*n+2)/(pow(2.0,2*n)*(2*n+3.0));
     REAL8 deltafn = 2.0*sqrt(mu/(ndim*gnn));

     /* compute number of grid points in this dimension and enforce a grid centered on the middle of the parameter space */
     INT4 length = (INT4)ceil((fnmax[n]-fnmin[n])/deltafn);
     REAL8 minfn = 0.5*(fnmin[n]+fnmax[n]) - 0.5*(length-1)*deltafn;
     
     (*gridparams)->grid[n].delta = deltafn;					 
     (*gridparams)->grid[n].length = length;
     (*gridparams)->grid[n].min = minfn;
     snprintf((*gridparams)->grid[n].name,LALNameLength,"f%d",n);
     
     LogPrintf(LOG_DEBUG,"%s : %s -> [%e - %e] (%e) %d grid points.\n",fn,(*gridparams)->grid[n].name,(*gridparams)->grid[n].min,
	       (*gridparams)->grid[n].min+((*gridparams)->grid[n].length-1)*(*gridparams)->grid[n].delta,
	       (*gridparams)->grid[n].delta,(*gridparams)->grid[n].length);
   }
   LogPrintf(LOG_DEBUG,"%s : computed output grid parameters.\n",fn);

   /* compute some internally required parameters for the grid */
   if ( ((*gridparams)->prod = XLALCalloc(ndim,sizeof(UINT4))) == NULL) {
     LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for Template structure.\n",fn);
     return XLAL_ENOMEM;
   }
   (*gridparams)->ndim = ndim;
   (*gridparams)->mismatch = mu;
   (*gridparams)->max = 1;
   for (k=0;k<(*gridparams)->ndim;k++) (*gridparams)->max *= (*gridparams)->grid[k].length;
   
   (*gridparams)->prod[0] = 1;
   for (k=1;k<(*gridparams)->ndim;k++) (*gridparams)->prod[k] = (*gridparams)->prod[k-1]*(*gridparams)->grid[k-1].length;
   
   LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
   return XLAL_SUCCESS;

 } 

/** Compute the demodulated power for all downsampled timeseries 
 *
 * This function is simply a wrapper for XLALComputeDemodulatedPower 
 *
 */
int XLALComputeDemodulatedPowerVector(REAL4DemodulatedPowerVector **power,     /**< [out] the spin derivitive demodulated power */ 
				      COMPLEX8TimeSeriesArray *dsdata,         /**< [in] the downsampled SFT data */
				      GridParametersVector *gridparams         /**< [in/out] the spin derivitive gridding parameters */
				      )
{
  
  const CHAR *fn = __func__;             /* store function name for log output */
  UINT4 i;

  /* validate input */
  if ((*power) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output REAL4DemodulatedPowerVector structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (dsdata == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input COMPLEX8TimeSeriesArray structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (gridparams == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input GridParametersVector structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (dsdata->length != gridparams->length) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, length of downsampled data vector and grid parameters vector not equal.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }

  /* allocate memory */
  if ( ((*power) = (REAL4DemodulatedPowerVector*)XLALCalloc(1,sizeof(REAL4DemodulatedPowerVector))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for REAL4DemodulatedPowerVector structure.\n",fn);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ( ((*power)->segment = (REAL4DemodulatedPower**)XLALCalloc(dsdata->length,sizeof(REAL4DemodulatedPower*))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for REAL4DemodulatedPower structure.\n",fn);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  (*power)->length = dsdata->length;

  /* loop over each segment */
  for (i=0;i<dsdata->length;i++) {
    
    if (XLALComputeDemodulatedPower(&((*power)->segment[i]),dsdata->data[i],gridparams->segment[i])) {
      LogPrintf(LOG_CRITICAL,"%s: XLALComputeDemodulatedPwer() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_EINVAL);
    }
    LogPrintf(LOG_DEBUG,"%s : computed demodulated power for SFT %d/%d\n",fn,i+1,dsdata->length);

  }
  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Compute the demodulated power for a single SFT 
 *
 * This involves taking the downsampled SFT timeseries and multiplying by the
 * timeseries spin-derivitive templates in turn.  Then we inverse FFT the result 
 * and square to obtain the power at a given set of freq derivitive parameters. 
 *
 */
int XLALComputeDemodulatedPower(REAL4DemodulatedPower **power,     /**< [out] the spin derivitive demodulated power */ 
				COMPLEX8TimeSeries *dsdata,        /**< [in] the downsampled SFT data */
				GridParameters *gridparams             /**< [in/out] the spin derivitive gridding parameters */
				)
{
  
  const CHAR *fnc = __func__;             /* store function name for log output */
  COMPLEX8FFTPlan *plan = NULL;           /* plan for the inverse FFT */
  UINT4 i,j,k,n;                          /* counters */
  REAL8 freqoffset = 0.0;                 /* the offset between the desired starting freq and the closest bin */
  INT4 binoffset = 0;                     /* the offset in output frequency bins from the start of the sft */
  COMPLEX8Vector *temp_input = NULL;      /* the temporary input of the inverse FFT = data*template */
  COMPLEX8Vector *temp_output = NULL;     /* the temporary output of the inverse FFT */
  REAL8Vector *fn = NULL;                 /* used to store the current frequency derivitive values (0'th element not used) */
  REAL8 norm = 1.0;                       /* the normalisation factor required to return to original scaling */
  UINT4 Nvec = 1;                         /* the number of spin derivitives on which to perform FFT */
  UINT4 cnt = 0;                          /* indexes the output vector */

  /* validate input arguments */
  if ((*power) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output REAL4DemodulatedPower structure != NULL.\n",fnc);
    return XLAL_EINVAL;
  }
  if (dsdata == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input COMPLEX8TimeSeries structure == NULL.\n",fnc);
    return XLAL_EINVAL;
  }
  if (gridparams == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input GridParameters structure == NULL.\n",fnc);
    return XLAL_EINVAL;
  }
  
  /* allocate memory for the output structure */
  if ( (*power = XLALCalloc(1,sizeof(REAL4DemodulatedPower))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for REAL4DemodulatedPower structure.\n",fnc);
    return XLAL_ENOMEM;
  }

  /* allocate memory for sequentially stored power */
  if ( ((*power)->data = XLALCreateREAL4Vector(gridparams->max)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCreateREAL4Vector() failed with error = %d\n",fnc,xlalErrno);
    return XLAL_ENOMEM;
  }
   
  /* compute the number of spin-derivitive templates (not including freq) */
  for (n=1;n<gridparams->ndim;n++) {
    Nvec *= gridparams->grid[n].length;
  }
  LogPrintf(LOG_DEBUG,"%s : computed number of spin derivitives as = %d.\n",fnc,Nvec);

  /* compute timeseries parameters given the requested frequency resolution */
  {
    REAL8 Teff = 1.0/gridparams->grid[0].delta;
    UINT4 N = floor(0.5 + Teff/dsdata->deltaT);
    Teff = (REAL8)N*dsdata->deltaT;
    gridparams->grid[0].delta = 1.0/Teff;
    norm = pow(1.0/(REAL8)dsdata->data->length,2.0);
   /*  LogPrintf(LOG_DEBUG,"%s : length of FFT input = %d\n",fnc,N); */
/*     LogPrintf(LOG_DEBUG,"%s : computed effective length for inverse FFT as %f sec.\n",fnc,Teff); */
/*     LogPrintf(LOG_DEBUG,"%s : computed modified frequency resolution as %e Hz.\n",fnc,gridparams->grid[0].delta); */

    /* allocate memory for the temporary zero-padded input data */
    if ( (temp_input = XLALCreateCOMPLEX8Vector(N)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateCOMPLEX8Vector() failed with error = %d.\n",fnc,xlalErrno);
      return XLAL_ENOMEM;
    }
    
    /* allocate memory for the time domain zero-padded phase correction vector */
    if ( (temp_output = XLALCreateCOMPLEX8Vector(N)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateCOMPLEX8Vector() failed with error = %d\n",fnc,xlalErrno);
      return XLAL_ENOMEM;
    }
    
    /* create a forward complex fft plan */
    if ((plan = XLALCreateForwardCOMPLEX8FFTPlan(N,0)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateForwardCOMPLEX8FFTPlan() failed with error = %d\n",fnc,xlalErrno);
      return XLAL_ENOMEM;
    }
    
    /* create a vector to store the frequency derivitive values */
    if ((fn = XLALCreateREAL8Vector(gridparams->ndim)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCreateREAL8Vector() failed with error = %d\n",fnc,xlalErrno);
      return XLAL_ENOMEM;
    }

  }

  /* compute frequency offset needed for heterodyne so that desired frequency lies at an exact bin */
  {
    REAL8 newdf = gridparams->grid[0].delta;
    REAL8 closest;
    binoffset = (INT4)floor(0.5 + (gridparams->grid[0].min - dsdata->f0)/newdf);
    closest = dsdata->f0 + newdf*binoffset;
    freqoffset = gridparams->grid[0].min - closest;

    /* LogPrintf(LOG_DEBUG,"%s : requested start frequency = %e Hz -> closest frequency = %e Hz.\n",fnc,gridparams->grid[0].min,closest); */
/*     LogPrintf(LOG_DEBUG,"%s : frequency offset from closets bin = %e Hz\n",fnc,freqoffset); */
/*     LogPrintf(LOG_DEBUG,"%s : offset from first frequency = %d bins\n",fnc,binoffset); */
  }

  /* loop over each value of the spin derivitive grid (not including the frequency dimension itself) */
  for (i=0;i<Nvec;i++) {
    
    /* define current spin derivitive values - since the vectors are stored linearly we need to be able */
    /* to access the i'th element and know what spin derivitives it is for */
    UINT4 idx = i;
    for (j=gridparams->ndim-1;j>0;j--) {
      UINT4 prod = 1;
      for (k=j;k>0;k--) prod *= gridparams->grid[k].length;
      idx = idx - (UINT4)floor(idx/prod)*prod; 
      fn->data[j] = gridparams->grid[j].min + idx*gridparams->grid[j].delta;
      LogPrintf(LOG_DEBUG,"%s : for derivitive index %d -> f%d index = %d value = %e\n",fnc,i,j,idx,fn->data[j]);
    }
  
    /* initialise the input data */
    memset(temp_input->data,0.0,temp_input->length*sizeof(COMPLEX8));

    /* apply time domain phase correction - first loop over time and then spin derivitive */
    for (j=0;j<dsdata->data->length;j++) {
      
      /* compute phase correction including heterodyne to shift frequencies to match up with grid */
      REAL8 tn = j*dsdata->deltaT - 0.5*dsdata->deltaT*dsdata->data->length;
      REAL8 arg = (-1.0)*LAL_TWOPI*freqoffset*tn;
      REAL8 xr, xi;
      UINT4 fac = 1; 

      /* loop over each spin derivitive and add to phase contribution for current time sample */
      for (k=1;k<gridparams->ndim;k++) {
	tn *= tn;
	fac *= k+1;
	arg += (-1.0)*LAL_TWOPI*fn->data[k]*tn/fac;
      }

      /* compute real and imaginary parts of phase correction timeseries */
      xr = cos(arg);
      xi = sin(arg);
    
      /* multiply data by phase correction - leave the zero-padding */
      temp_input->data[j].re = dsdata->data->data[j].re*xr - dsdata->data->data[j].im*xi;
      temp_input->data[j].im = dsdata->data->data[j].re*xi + dsdata->data->data[j].im*xr;
				     
    }  

    /* FFT and square to get power */
    if (XLALCOMPLEX8VectorFFT(temp_output,temp_input,plan)) {
      LogPrintf(LOG_CRITICAL,"%s: XLALCOMPLEX8VectorFFT() failed with error = %d\n",fnc,xlalErrno);
      return XLAL_ENOMEM;
    }
    /* LogPrintf(LOG_DEBUG,"%s : computed the FFT\n",fnc); */

    /* check that we will not overrun the output vector */
    if ( (binoffset < 0) || (binoffset + (INT4)gridparams->grid[0].length > (INT4)temp_output->length) ) {
      LogPrintf(LOG_CRITICAL,"%s: strange, required bins from demodulated power out of range of result.\n",fnc,xlalErrno);
      return XLAL_EFAULT;
    }
    
    /* fill in the actual output making sure that the frequency bins of interest are used and the result is normalised */
    for (j=0;j<gridparams->grid[0].length;j++) {
      k = j + binoffset;
      (*power)->data->data[cnt] = norm*(temp_output->data[k].re*temp_output->data[k].re + temp_output->data[k].im*temp_output->data[k].im); 
      cnt++;
    }
    
  }

  /* point results gridparams pointer to the actual gridparams structure */
  (*power)->gridparams = gridparams;
   memcpy(&((*power)->epoch),&(dsdata->epoch),sizeof(LIGOTimeGPS));

  /* free memory */
  XLALDestroyCOMPLEX8Vector(temp_input);
  XLALDestroyCOMPLEX8Vector(temp_output);
  XLALDestroyREAL8Vector(fn);
  XLALDestroyCOMPLEX8FFTPlan(plan);

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fnc);
  return XLAL_SUCCESS;

}

/** Compute the background photon flux for each SFT
 *
 * This uses the median of the power to obtain the quantity r (the background 
 * photon flux). 
 *
 */
int XLALEstimateBackgroundFlux(REAL8Vector **background,     /**< [out] the background flux estimate */
			       SegmentParams *segparams,     /**< [in] the number of operational PCUs for each SFT */ 
			       SFTVector *sftvec             /**< [in] the SFTs */
 			       )
{
  
  const CHAR *fn = __func__;             /* store function name for log output */
  LALStatus status = blank_status;        /* for use wih non-XLAL functions */
  UINT4 i,j;                               /* counters */

  /* validate input arguments */
  if ((*background) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output REAL8Vector structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (segparams == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input INT4Vector structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (sftvec == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input SFTVector structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (segparams->npcus->length != sftvec->length) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, length of sftvector != length of npcus vector.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }

  /* allocate memory for background estaimte results */
  if (((*background) = XLALCreateREAL8Vector(sftvec->length)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: XLALCreateREAL8Vector() failed with error = %d.\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }

  /* loop over each SFT */
  for (i=0;i<sftvec->length;i++) {

    COMPLEX8Sequence *sft = sftvec->data[i].data;
    REAL8 *P = NULL;
    REAL8 medianbias;                       /* the bias from computing the median */
    REAL8 median;                           /* the median of the power */
    REAL8 T = 1.0/sftvec->data[i].deltaF;

    /* allocate temporary memory */
    if ((P = XLALCalloc(sft->length,sizeof(REAL8))) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for gridparams->grid.\n",fn);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }

    /* loop over each element in the SFT and record the power */
    for (j=0;j<sft->length;j++) P[j] = (sft->data[j].re*sft->data[j].re + sft->data[j].im*sft->data[j].im);

    /* sort the data */
    gsl_sort(P,1,sft->length);
 
    /* compute median */
    median = gsl_stats_median_from_sorted_data(P,1,sft->length);
   
    /* compute the median bias */
    LAL_CALL ( LALRngMedBias( &status, &medianbias, sft->length ), &status);
   
    /* record estimate */
    (*background)->data[i] = (segparams->npcus->data[i]*PCU_AREA/T)*median/medianbias;
    LogPrintf(LOG_DEBUG,"%s : Estimated the background for SFT starting at %d as %f cnts/s/m^2.\n",fn,sftvec->data[i].epoch.gpsSeconds,(*background)->data[i]);
    /* free the power */
    XLALFree(P);
    
  }

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Compute the grid on binary parameters based on the semi-coherent metric 
 *
 * We use this grid to perform the integration of the posterior and to ultimately 
 * compute the Bayes factor.
 *
 */
int XLALComputeBinaryGridParams(GridParameters **binarygridparams,  /**< [out] the binary parameter grid */
				REAL8Space *space,                  /**< [in] the signal parameter space */
				REAL8 T,                            /**< [in] the duration of the observation */
				REAL8 DT,                           /**< [in] the length of the coherent segments */
				REAL8 mu                            /**< [in] the mismatch */
				)
{
  
  const CHAR *fn = __func__;             /* store function name for log output */
  REAL8 gnn[NBINMAX];                    /* stores the diagonal metric elements */ 
  INT4 ndim = 0;                         /* the number of actual search dimensions */      
  INT4 n,k;                              /* counters */

  /* validate input arguments */
  if ((*binarygridparams) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output GridParameters structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (space == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input ParameterSpace structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (T < 0) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input T parameter < 0.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if ((DT < 0) || (DT > T)) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input DT parameter < 0 or < T.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if ( (mu < 0) || (mu>1) ) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input mismatch parameter, not in range 0 -> 1.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
 
  /* compute the semi-coherent binary metric diagonal elements */
  {
    REAL8 numax = space->data[0].max;
    REAL8 amax = space->data[1].max;
    REAL8 omegamax = space->data[3].max;
    gnn[0] = (pow(LAL_PI,2.0)/6.0)*pow(DT,2.0);                                  /* nu */
    gnn[1] = (pow(LAL_PI,2.0)/6.0)*pow(numax*omegamax*DT,2.0);                   /* a */
    gnn[2] = (pow(LAL_PI,2.0)/6.0)*pow(numax*amax*omegamax*omegamax*DT,2.0);     /* tasc */
    gnn[3] = (pow(LAL_PI,2.0)/6.0)*pow(numax*amax*omegamax*DT*T,2.0)/12.0;       /* W */
    
    /* add eccentricity parameters at some point */
    /*  gnn->data[4] = (pow(LAL_PI,2.0)/6.0)*pow(numax*amax*omegamax*DT,2.0);       /\* kappa *\/ */
    /*  gnn->data[5] = (pow(LAL_PI,2.0)/6.0)*pow(numax*amax*omegamax*DT,2.0);       /\* eta *\/ */
  }    
  
   /* allocate memory to the output */
  if ( ((*binarygridparams) = (GridParameters*)XLALCalloc(1,sizeof(GridParameters))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for gridparams->grid.\n",fn);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ( ((*binarygridparams)->grid = (Grid*)XLALCalloc(NBINMAX,sizeof(Grid))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for gridparams->grid.\n",fn);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if ( ((*binarygridparams)->prod = (UINT4*)XLALCalloc(NBINMAX,sizeof(UINT4))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for Template structure.\n",fn);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  (*binarygridparams)->ndim = NBINMAX;
  LogPrintf(LOG_DEBUG,"%s : allocated memory for the output grid parameters.\n",fn);
  
  /* we need to determine the true number of searchable dimensions */
  /* we check the width of a 1-D template across each dimension span */ 
  for (n=0;n<NBINMAX;n++) {
    REAL8 deltax = 2.0*sqrt(mu/gnn[n]);
    if (space->data[n].span > deltax) ndim++;
  }
  LogPrintf(LOG_DEBUG,"%s : determined true dimensionality of binary space = %d.\n",fn,ndim);

  /* Compute the grid spacing, grid start and span for each spin derivitive dimension */
  for (n=0;n<NBINMAX;n++) {
    
    REAL8 deltax;
    UINT4 length;
    REAL8 xmin;

    /* only if we have a non-zero span */
    if (space->data[n].span > 0) {
      
      /* compute spacing for this parameter given the total number of true search dimensions and the mismatch */
      deltax = 2.0*sqrt(mu/(ndim*gnn[n]));
      
      /* compute number of grid points in this dimension and enforce a grid centered on the middle of the parameter space */
      length = MYMAX((UINT4)ceil((space->data[n].span)/deltax),1);
      xmin = 0.5*(space->data[n].min + space->data[n].max) - 0.5*(length-1)*deltax;
      
    }
    else {
      
      /* otherwise set the space boundaries accordingly */
      deltax = 0.0;
      length = 1;
      xmin = space->data[n].min;

    }
    
    (*binarygridparams)->grid[n].delta = deltax;					 
    (*binarygridparams)->grid[n].length = length;
    (*binarygridparams)->grid[n].min = xmin;
    strncpy((*binarygridparams)->grid[n].name,space->data[n].name,LALNameLength*sizeof(CHAR));

    LogPrintf(LOG_DEBUG,"%s : %s -> [%e - %e] (%e) %d grid points.\n",fn,(*binarygridparams)->grid[n].name,(*binarygridparams)->grid[n].min,
	      (*binarygridparams)->grid[n].min+(*binarygridparams)->grid[n].length*(*binarygridparams)->grid[n].delta,
	      (*binarygridparams)->grid[n].delta,(*binarygridparams)->grid[n].length);
  }
  LogPrintf(LOG_DEBUG,"%s : computed output grid parameters.\n",fn);

  /* compute some internally required parameters for the grid */  
  (*binarygridparams)->mismatch = mu;
  (*binarygridparams)->max = 1;
  for (k=0;k<(INT4)(*binarygridparams)->ndim;k++) (*binarygridparams)->max *= (*binarygridparams)->grid[k].length;
  
  (*binarygridparams)->prod[0] = 1;
  for (k=1;k<(INT4)(*binarygridparams)->ndim;k++) (*binarygridparams)->prod[k] = (*binarygridparams)->prod[k-1]*(*binarygridparams)->grid[k-1].length;
  
  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Compute the Bayes factor for the semi-coherent search
 *
 * This function performs the integral over the binary parameter space on 
 * the posterior probability distribution (likelihood*prior).
 *
 */
int XLALComputeBayesFactor(BayesianProducts **Bayes,                /**< [out] the Bayesian analysis products */
			   REAL4DemodulatedPowerVector *power,      /**< [in] the input data in the form of power */
			   ParameterSpace *pspace,                  /**< [in] the parameter space */
			   REAL8 sigalpha                           /**< [in] the signal amplitude prior sigma */
			   )
{  
  const CHAR *fn = __func__;                          /* store function name for log output */
  LikelihoodParamsVector *Lparamsvec = NULL;          /* stores parameters required for the likelihood calculation */
  Template *bintemp = NULL;                           /* the binary parameter space template */
  Template fdots;                                     /* the freq derivitive template for each segment */
  UINT4 i,j;                                          /* counters */
  REAL8 logBayesfactor = -1e200;                      /* the final BayesFactor result */
  REAL8PriorsVector *priors = pspace->priors;         /* shortcut pointer to priors */
  UINT4 percent = 0;                                  /* counter for status update */

  /* validate input parameters */
  if ((*Bayes) != NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output BayesianProducts structure != NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (power == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input REAL4DemodulatedPowerVector structure = NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (pspace == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input GridParameters structure = NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (sigalpha < 0.0) {
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, input sigalphs must be > 0.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  
  /* setup parameters for the likelihood computation */
  if (XLALSetupLikelihood(&Lparamsvec,Bayes,power,pspace->gridparams,sigalpha)) {
    LogPrintf(LOG_CRITICAL,"%s : XLALSetupLikelihood() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_EFAULT);
  }

  /* allocate memory for the fdots */
  if ((fdots.x = XLALCalloc(power->segment[0]->gridparams->ndim,sizeof(REAL8))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  fdots.ndim = power->segment[0]->gridparams->ndim;

  /* single loop over all templates */
  while (XLALGetNextBinaryTemplate(&bintemp,pspace->gridparams)) {

    REAL8 logLratiosum = 0.0;                       /* initialise likelihood ratio */

    /* LogPrintf(LOG_DEBUG,"%s : current template %d/%d = [%f %f %f %e]\n",fn,bintemp->currentidx,pspace->gridparams->max,bintemp->x[0],bintemp->x[1],bintemp->x[2],bintemp->x[3]); */

    /* loop over segments */
    for (i=0;i<power->length;i++) {
      
      REAL4DemodulatedPower *currentpower = power->segment[i];
      GridParameters *fdotgrid = power->segment[i]->gridparams;
      REAL8 tmid = XLALGPSGetREAL8(&(power->segment[i]->epoch)) + 0.5*pspace->tseg;
      UINT4 idx = 0;
      REAL8 logLratio = 0.0;
      REAL4 X;
      
      /* compute instantaneous frequency derivitives corresponding to the current template for this segment */
      if (XLALComputeBinaryFreqDerivitives(&fdots,bintemp,tmid)) {
	LogPrintf(LOG_CRITICAL,"%s : XLALFindBinaryFreqDerivitives() failed with error = %d\n",fn,xlalErrno);
	return XLAL_EFAULT;
      }
     /*  LogPrintf(LOG_DEBUG,"%s : segment %d/%d -> (tmid = %6.0f) spin derivitives = [%6.12f]\n",fn,i,tmid,power->length,fdots.x[0]); */
      
      /* find indices corresponding to the spin derivitive values for the segment power */
      for (j=0;j<fdots.ndim;j++) {
	UINT4 tempidx = floor(0.5 + (fdots.x[j] - fdotgrid->grid[j].min)/fdotgrid->grid[j].delta);
	idx += tempidx*fdotgrid->prod[j];
      }
      X = currentpower->data->data[idx];

      /* compute the likelihood for this location given the power value */
      /* inside loop over segments we compute the product of likelihood ratios */
      /* this is the sum of log-likelihood ratios */
      logLratio = XLALComputePhaseAmpMargLogLRatio(X,&(Lparamsvec->data[i]));      
      logLratiosum += logLratio;

      /* apply priors - this is a multiplication of likelihoods OR a sum in log-likelihoods */
      for (j=0;j<pspace->gridparams->ndim;j++) {
	logLratio += priors->data[j].logpriors->data[bintemp->idx[j]];
      }

      /* record the log BayesFactor for each SFT */
      (*Bayes)->logBayesFactor_phaseamp_vector->data[i] = XLALLogSumExp((*Bayes)->logBayesFactor_phaseamp_vector->data[i],logLratio);

    } /* end loop over segments */
    
    /* apply priors - this is a multiplication of likelihoods OR a sum in log-likelihoods */
    for (i=0;i<pspace->gridparams->ndim;i++) {
      logLratiosum += priors->data[i].logpriors->data[bintemp->idx[i]];
    }
    
    /* for this template we contribute to each posterior vector */
    /* we sum likelihood-ratios NOT log-likelihood-ratios */
    for (i=0;i<pspace->gridparams->ndim;i++) {
      REAL8 temp = (*Bayes)->logposteriors_phaseamp[i]->data[bintemp->idx[i]];
      (*Bayes)->logposteriors_phaseamp[i]->data[bintemp->idx[i]] = XLALLogSumExp(temp,logLratiosum); 
    }

    /* we also sum likelihood-ratios to compute the overall Bayes-factor */
    logBayesfactor = XLALLogSumExp(logBayesfactor,logLratiosum);
    
    /* output status to screen */
    if ((UINT4)floor(0.5 + 100*bintemp->currentidx/pspace->gridparams->max) > percent) {
      percent = (UINT4)floor(0.5 + 100*bintemp->currentidx/pspace->gridparams->max);
      LogPrintf(LOG_DEBUG,"%s : completed %d%%\n",fn,percent);
    }

  } /* end loop over templates */

  /* normalise the Bayesfactors */
  for (i=0;i<pspace->gridparams->ndim;i++) {
    logBayesfactor += priors->data[i].logdelta;
     for (j=0;j<power->length;j++) (*Bayes)->logBayesFactor_phaseamp_vector->data[j] += priors->data[i].logdelta;
  }
  LogPrintf(LOG_DEBUG,"%s : computed log(B) = %e\n",fn,logBayesfactor);

  /* point the Bayesfactor results grid to the grid used and the result obtained */
  (*Bayes)->gridparams = pspace->gridparams;
  (*Bayes)->logBayesFactor_phaseamp = logBayesfactor;

  /* free template memory */
  XLALFree(fdots.x);

  /* free likelihood params */
  XLALFree(Lparamsvec->data);
  XLALFree(Lparamsvec);

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Compute some repeatedly used parameters in the likelihood computation 
 *
 * To make the likelihood computation efficient we compute some parameters before 
 * cycling over templates.  We also compute the priors.
 *
 */
int XLALSetupLikelihood(LikelihoodParamsVector **Lparamsvec,       /**< [out] set of likelihood params for each segment */
			BayesianProducts **Bayes,                  /**< [out] the output products of the Bayesian search */
			REAL4DemodulatedPowerVector *power,        /**< [in] the data in the form of power */
			GridParameters *binarygrid,                    /**< [in] the binary parameter grid */
			REAL8 sigalpha                             /**< [in] the amplitude sigma prior */
			)
{
  const CHAR *fn = __func__;            /* store function name for log output */
  UINT4 i,j;                            /* counters */

  /* validate input */
  

  /* allocate memory for the likelihood params */
  if (((*Lparamsvec) = XLALCalloc(1,sizeof(LikelihoodParamsVector))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if (((*Lparamsvec)->data = XLALCalloc(power->length,sizeof(LikelihoodParams))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }

  /* allocate memory for the Bayesian output products */
  if (((*Bayes) = XLALCalloc(1,sizeof(BayesianProducts))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if (((*Bayes)->logBayesFactor_phaseamp_vector = XLALCreateREAL8Vector(power->length)) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCrateREAL8Vector() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if (((*Bayes)->epoch = XLALCalloc(power->length,sizeof(LIGOTimeGPS))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  if (((*Bayes)->logposteriors_phaseamp = XLALCalloc(binarygrid->ndim,sizeof(REAL8Vector *))) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s : XLALCalloc() failed with error = %d\n",fn,xlalErrno);
    XLAL_ERROR(fn,XLAL_ENOMEM);
  }
  for (i=0;i<binarygrid->ndim;i++) { 
    if (((*Bayes)->logposteriors_phaseamp[i] = XLALCreateREAL8Vector(binarygrid->grid[i].length)) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s : XLALCrateREAL8Vector() failed with error = %d\n",fn,xlalErrno);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    /* initialise results */
    for (j=0;j<binarygrid->grid[i].length;j++) (*Bayes)->logposteriors_phaseamp[i]->data[j] = -1e200;
    
  }
  (*Bayes)->gridparams = binarygrid;
  (*Bayes)->ndim = binarygrid->ndim;
  (*Bayes)->nsegments = power->length;

   /* initialise results - we add using XLALlogsumexp so we initialise to the log of a very low number */
  for (j=0;j<power->length;j++) (*Bayes)->logBayesFactor_phaseamp_vector->data[j] = -1e200;
  
  /**************************************************************************************************/
  /* parameters for phase and amplitude marginalisation */

  /* loop over each segment and compute parameters required for the likelihood computation */
  for (i=0;i<power->length;i++) {

    /* define commonly used individual sft quantities */
    REAL8 T = power->segment[i]->duration;
    REAL8 nV = (REAL8)power->segment[i]->npcus*PCU_AREA;
    REAL8 r = power->segment[i]->r;
   
    /* compute the parameters needed for the computation of the */
    /* phase and amplitude marginalised log-likelihood ratio */ 
    REAL8 Y = 0.25*T*nV/r;
    REAL8 X = nV/r;
    REAL8 P = 1.0/(2.0*Y*sigalpha*sigalpha + 1.0);
    (*Lparamsvec)->data[i].logsqrtP = 0.5*log(P);
    (*Lparamsvec)->data[i].PQ = P*0.25*sigalpha*sigalpha*X*X;
    LogPrintf(LOG_DEBUG,"%s : computed X = %e Y = %e P = %e PQ = %e for SFT %d/%d\n",fn,X,Y,P,(*Lparamsvec)->data[i].PQ,i+1,power->length);
    
    /* record epoch */
    memcpy(&((*Bayes)->epoch[i]),&(power->segment[i]->epoch),sizeof(LIGOTimeGPS));
    
  }
 
  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

/** Compute the next binary template in the grid
 *
 * the templates are generated sequentially from the grid parameters file.  The n-dimensional
 * virtual indices of each dimension are also generated  
 *
 */
int XLALGetNextBinaryTemplate(Template **temp,                        /**< [out] the signal model template parameters */
			      GridParameters *gridparams              /**< [in] the parameter space grid params */
			      )
{  
  const CHAR *fn = __func__;             /* store function name for log output */
  UINT4 idx;                             /* the index variable */ 
  INT4 j;                                /* counters */

  /* if the input template is null then we allocate memory and assume we are on the first template */
  if ((*temp) == NULL) {
    if ( ((*temp) = XLALCalloc(1,sizeof(Template))) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for Template structure.\n",fn);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    if ( ((*temp)->x = XLALCalloc(gridparams->ndim,sizeof(REAL8))) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for Template structure.\n",fn);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    if ( ((*temp)->idx = XLALCalloc(gridparams->ndim,sizeof(UINT4))) == NULL) {
      LogPrintf(LOG_CRITICAL,"%s: unable to allocate memory for Template structure.\n",fn);
      XLAL_ERROR(fn,XLAL_ENOMEM);
    }
    (*temp)->currentidx = 0;
    (*temp)->ndim = gridparams->ndim;
  }
  else if ((*temp)->currentidx == gridparams->max) {
    
    /* free binary template memory */
    XLALFree((*temp)->x);
    XLALFree((*temp)->idx);
    XLALFree(*temp);
    
    LogPrintf(LOG_DEBUG,"%s: at last template.\n",fn);
    return 0;
  }
    
  /* initialise index */
  idx = (*temp)->currentidx;

  /* loop over each dimension and obtain the index for that dimension (store both) */
  for (j=gridparams->ndim-1;j>=0;j--) {

    /* compute the index for the j'th dimension and compute the actual value */
    UINT4 q = floor(idx/gridparams->prod[j]);
    (*temp)->x[j] = gridparams->grid[j].min + q*gridparams->grid[j].delta;
    (*temp)->idx[j] = q;

    /* update the index variable for the next dimension */
    idx = idx - q*gridparams->prod[j];
    
  }
  
  /* update index */
  (*temp)->currentidx++;

  return 1;

}

/** Compute the instantaneous frequency derivitives for a given binary template and segment
 *
 */
int XLALComputeBinaryFreqDerivitives(Template *fdots,                        /**< [out] the frequency derivitives */
				     Template *bintemp,                      /**< [in] the binary template */
				     REAL8 tmid                              /**< [in] the midpoint time of the segment */
				     )
{  
  UINT4 n;                               /* counters */
  REAL8 nu = bintemp->x[0];              /* define nu */
  REAL8 asini = bintemp->x[1];           /* define asini */
  REAL8 tasc = bintemp->x[2];            /* define tasc */
  REAL8 omega = bintemp->x[3];           /* define omega */

  /* the instantanous frequency is therefore f0 = nu - a*nu*W*cos(W*(t-tasc) ) */
  fdots->x[0] = nu - nu*asini*omega*cos(omega*(tmid-tasc));

  /* the instantanous nth frequency derivitive is therefore fn = - a * nu * W^(n+1) * cos ( W*(t-tasc) + n*pi/2 ) */
  for (n=1;n<fdots->ndim;n++) {
    fdots->x[n] = (-1.0)*nu*asini*pow(omega,n+1)*cos(omega*(tmid-tasc) + 0.5*n*LAL_PI);
  }
  
  return XLAL_SUCCESS;

}

/** Compute the phase and amplitude marginalised log-likelihood for a signal in Poisson noise 
 *
 * This function computes (as efficiently as possible) the log-likelihood of obtaining a particular
 * power value given Poisson noise and marginalising over an unknown phase and amplitude.
 *
 */
REAL8 XLALComputePhaseAmpMargLogLRatio(REAL8 X,                       /**< [in] the Fourier power */ 
				       LikelihoodParams *Lparams      /**< [in] pre-computed parameters useful in the likelihood */
				       ) 
{
  REAL8 arg = Lparams->PQ*X;    /* define the argument to the bessel function */

  /* compute the log-likelihood ratio */
  return Lparams->logsqrtP + arg +  XLALLogBesselI0(arg);
  
}

/** Output the results to file 
 *
 * We choose to output all results from a specific analysis to a single file 
 *
 */
int XLALOutputBayesResults(CHAR *outputdir,            /**< [in] the output directory name */
			   BayesianProducts *Bayes,    /**< [in] the results structure */
			   ParameterSpace *pspace,     /**< [in] the parameter space */ 
			   CHAR *clargs                /**< [in] the command line args */
			   )
{
  const CHAR *fn = __func__;            /* store function name for log output */
  CHAR outputfile[LONGSTRINGLENGTH];    /* the output filename */
  time_t curtime = time(NULL);          /* get the current time */
  CHAR *time_string = NULL;             /* stores the current time */
  CHAR *version_string = NULL;           /* pointer to a string containing the git version information */
  FILE *fp = NULL;                      /* pointer to the output file */
  UINT4 i,j;                            /* counters */

  /* validate input */
  if (outputdir == NULL) { 
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, output directory string == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (Bayes == NULL) { 
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, results BayesProducts structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  if (pspace == NULL) { 
    LogPrintf(LOG_CRITICAL,"%s: Invalid input, ParameterSpace structure == NULL.\n",fn);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  
  /* define the output filename */
  /* the format we adopt is the following BayesianResults-<SOURCE>-<START>_<END>-<MIN_FREQ_INT>_<MIN_FREQ_mHZ>_ <MAX_FREQ_INT>_<MAX_FREQ_mHZ>.txt */
  {
    UINT4 min_freq_int = floor(pspace->space->data[0].min);
    UINT4 max_freq_int = floor(pspace->space->data[0].max);
    UINT4 min_freq_mhz = (UINT4)floor(0.5 + (pspace->space->data[0].min - (REAL8)min_freq_int)*1e3);
    UINT4 max_freq_mhz = (UINT4)floor(0.5 + (pspace->space->data[0].max - (REAL8)max_freq_int)*1e3);
    UINT4 end = (UINT4)ceil(XLALGPSGetREAL8(&(pspace->epoch)) + pspace->span);
    snprintf(outputfile,LONGSTRINGLENGTH,"%s/BayesianResults-%s-%d_%d-%04d_%03d_%04d_%03d.txt",
	     outputdir,pspace->source,pspace->epoch.gpsSeconds,end,min_freq_int,min_freq_mhz,max_freq_int,max_freq_mhz); 
  }
  LogPrintf(LOG_DEBUG,"%s : output to %s\n",fn,outputfile);

  /* open the output file */
  if ((fp = fopen(outputfile,"w")) == NULL) {
    LogPrintf(LOG_CRITICAL,"%s: Error, failed to open file %s for writing.  Exiting.\n",fn,outputfile);
    XLAL_ERROR(fn,XLAL_EINVAL);
  }
  
  /* Convert time to local time representation */
  {
    struct tm *loctime = localtime(&curtime);
    CHAR *temp_time = asctime(loctime);    
    UINT4 n = strlen(temp_time);
    time_string = XLALCalloc(n,sizeof(CHAR));
    snprintf(time_string,n-1,"%s",temp_time);
  }
  
  /* get GIT version information */
  {
    CHAR *temp_version = XLALGetVersionString(0); 
    UINT4 n = strlen(temp_version);
    version_string = XLALCalloc(n,sizeof(CHAR));
    snprintf(version_string,n-1,"%s",temp_version);
    XLALFree(temp_version);
  }

  /* output header information */
  fprintf(fp,"%s \n",version_string);
  fprintf(fp,"%%%% command line args\t\t= %s\n",clargs);
  fprintf(fp,"%%%% filename\t\t\t= %s\n",outputfile);
  fprintf(fp,"%%%% date\t\t\t\t= %s\n",time_string);
  fprintf(fp,"%%%% start time (GPS sec)\t\t= %d\n",pspace->epoch.gpsSeconds);
  fprintf(fp,"%%%% observation span (sec)\t= %d\n",(UINT4)pspace->span);
  fprintf(fp,"%%%% coherent time (sec)\t\t= %d\n",(UINT4)pspace->tseg);
  fprintf(fp,"%%%% number of segments\t\t= %d\n",Bayes->nsegments);
  fprintf(fp,"%%%% number of dimensions\t\t= %d\n",Bayes->gridparams->ndim);
  fprintf(fp,"%%%% mismatch\t\t\t= %6.12f\n",Bayes->gridparams->mismatch);
  fprintf(fp,"%%%%\n");

  /* if an injection has been performed we output the injection parameters */
  if (pspace->inj) {
    fprintf(fp,"%%%% injection parameters --------------------------------------------------------------------------------\n");
    fprintf(fp,"%%%% inj_amp\t\t\t\t= %6.12f\n",pspace->inj->amp);
    fprintf(fp,"%%%% inj_nu\t\t\t\t= %6.12f\n",pspace->inj->temp.x[0]);
    fprintf(fp,"%%%% inj_asini\t\t\t= %6.12f\n",pspace->inj->temp.x[1]);
    fprintf(fp,"%%%% inj_tasc\t\t\t\t= %6.12f\n",pspace->inj->temp.x[2]);
    fprintf(fp,"%%%% inj_omega\t\t\t= %6.12e\n",pspace->inj->temp.x[3]);
    fprintf(fp,"%%%% -----------------------------------------------------------------------------------------------------\n");
    fprintf(fp,"%%%%\n");
  }

  /* output the main Bayes factor results */
  fprintf(fp,"%%%% log Bayes Factor (phase and amplitude marginalised per segment)\t= %6.12e\n",Bayes->logBayesFactor_phaseamp);
  fprintf(fp,"%%%% log Bayes Factor (phase marginalised per segment)\t\t\t= not implemented\n");
  fprintf(fp,"%%%%\n");
  fprintf(fp,"%%%% log Bayes Factor (phase and amplitude marginalised per segment)\n");
  fprintf(fp,"%%%%\n");

  /* output the Bayes factor for each segment */
  for (i=0;i<Bayes->nsegments;i++) {
    fprintf(fp,"%d\t%d\t%6.12e\n",Bayes->epoch[i].gpsSeconds,Bayes->epoch[i].gpsSeconds+(UINT4)pspace->tseg,Bayes->logBayesFactor_phaseamp_vector->data[i]);
  }
  fprintf(fp,"%%%%\n");

  /* loop over each search dimension and output the grid parameters and posteriors */
  for (i=0;i<Bayes->gridparams->ndim;i++) {
    fprintf(fp,"%%%% -------------------------------------------------------------------------------------------------------\n%%%%\n");
    fprintf(fp,"%%%% name_%d\t= %s\n",i,Bayes->gridparams->grid[i].name);
    fprintf(fp,"%%%% min_%d\t= %6.12e\n",i,pspace->space->data[i].min);
    fprintf(fp,"%%%% max_%d\t= %6.12e\n",i,pspace->space->data[i].max);
    fprintf(fp,"%%%% sig_%d\t= %6.12e\n",i,pspace->space->data[i].sig);
    fprintf(fp,"%%%% start_%d\t= %6.12e\n",i,Bayes->gridparams->grid[i].min);
    fprintf(fp,"%%%% delta_%d\t= %6.12e\n",i,Bayes->gridparams->grid[i].delta);
    fprintf(fp,"%%%% length_%d\t= %d\n",i,Bayes->gridparams->grid[i].length);
    if (pspace->priors->data[i].gaussian) fprintf(fp,"%%%% prior_%d\t= GAUSSIAN\n",i);
    else fprintf(fp,"%%%% prior_%d\t= FLAT\n",i);
    fprintf(fp,"%%%%\n%%%%\t%s\t\tlog_post(%s)\t\tnorm_post(%s)\t\tnorm_prior(%s)\n%%%%\n",
	    Bayes->gridparams->grid[i].name,Bayes->gridparams->grid[i].name,Bayes->gridparams->grid[i].name,Bayes->gridparams->grid[i].name);

    /* output posteriors - we output un-normalised and normalised posteriors plus priors */
    {
      REAL8 sum = 0.0;
      REAL8 mx = Bayes->logposteriors_phaseamp[i]->data[0];
      for (j=0;j<Bayes->gridparams->grid[i].length;j++) if (Bayes->logposteriors_phaseamp[i]->data[j] > mx) mx = Bayes->logposteriors_phaseamp[i]->data[j];
      
      /* compute normalising constant for the posteriors */
      for (j=0;j<Bayes->gridparams->grid[i].length;j++) {
	sum += exp(Bayes->logposteriors_phaseamp[i]->data[j]-mx)*Bayes->gridparams->grid[i].delta;
      }

      /* output posteriors and priors to file */
      for (j=0;j<Bayes->gridparams->grid[i].length;j++) {
	REAL8 x = Bayes->gridparams->grid[i].min + j*Bayes->gridparams->grid[i].delta;
	REAL8 log_post = Bayes->logposteriors_phaseamp[i]->data[j];
	REAL8 norm_post = exp(Bayes->logposteriors_phaseamp[i]->data[j]-mx)/sum;
	REAL8 norm_prior = exp(pspace->priors->data[i].logpriors->data[j]);
	fprintf(fp,"%6.12e\t%6.12e\t%6.12e\t%6.12e\n",x,log_post,norm_post,norm_prior);
      }
    
    }
  
  }

  /* close the file */
  fclose(fp);

  /* free memory */
  XLALFree(time_string);
  XLALFree(version_string);

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** function to compute the log of the bessel function without computing the bessel function directly
 *
 * We compute the log of the Bessel function I0(z) bypassing the computation of the 
 * function and then taking the log.  This avoids numerical problems.  The expansion
 * used is taken from Abramowitz and Stegun P.378
 *
 */
REAL8 XLALLogBesselI0(REAL8 z            /**< [in] the argument of the Bessel function */
		      )
{
  UINT4 i;                /* counter */
  REAL8 tn = 1.0;         /* initialise the variable storing t^n */

  /* for large input args */
  if (z>3.75) {

    REAL8 invt = 3.75/z;
    REAL8 y = BESSCO_HIGH[0];
       
    /* compute expansion */
    for (i=1;i<LEN_BESSCO_HIGH;i++) {
      tn = tn*invt;
      y += BESSCO_HIGH[i]*tn;
    }
    
    /* compute log of bessel function */
    return log(y) + z - 0.5*log(z);
    
  }
  /* for small input args */
  else {
    
    REAL8 I0 = BESSCO_LOW[0];
    REAL8 tnsq = z*z/14.0625;
   
    for (i=1;i<LEN_BESSCO_LOW;i++) {
      tn *= tnsq;
      I0 += BESSCO_LOW[i]*tn;
    }

    return log(I0);

  }
  
}

/* function to compute the log of the sum of the arguments of two logged quantities
 *
 * Eg. input log(x) and log(y) -> output log(x+y)
 *
 * If you do this by exponentiating first, then summing and then logging again you
 * can easily gat overflow errors.  We use a trick to avoid this.
 */
REAL8 XLALLogSumExp(REAL8 logx,      /**< [in] the log of x */  
		    REAL8 logy       /**< [in] the log of y */
		    )
{
  
  /* initially set max arg as logx */
  REAL8 logmax = logx; 
  REAL8 logmin = logy;

  /* if logy > logx then switch the max and min */
  if (logy>logx) {
    logmax = logy;
    logmin = logx;
  }
 
  /* compute log(x + y) = logmax + log(1.0 + exp(logmin - logmax)) */
  /* this way the argument to the exponential is always negative */
  return logmax + log(1.0 + exp(logmin - logmax));     
  
 }

/** Free the memory allocated within a ParameterSpace structure
 *
 */
int XLALFreeParameterSpace(ParameterSpace *pspace            /**< [in] the parameter space to be freed */
			   )
{
  
  const CHAR *fn = __func__;   /* store function name for log output */
  UINT4 i;                     /* counter */

  /* free parameter space */
  XLALFree(pspace->space->data);
  XLALFree(pspace->space);
  
  /* free prior params */
  for (i=0;i<pspace->gridparams->ndim;i++) {
    XLALDestroyREAL8Vector(pspace->priors->data[i].logpriors);
  }
  XLALFree(pspace->priors->data);
  XLALFree(pspace->priors);
  LogPrintf(LOG_DEBUG,"%s : freed the prior parameters\n",fn);
  
  /* free binary grid params */
  XLALFree(pspace->gridparams->grid);
  XLALFree(pspace->gridparams->prod);
  XLALFree(pspace->gridparams);
  LogPrintf(LOG_DEBUG,"%s : freed the binary grid parameters\n",fn);

  /* free the injection parameters if used */
  if (pspace->inj) {
    XLALFree(pspace->inj->temp.x);
    XLALFree(pspace->inj);
  }

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Free the memory allocated within a REAL4DemodulatedPowerVector structure
 *
 */
int XLALFreeREAL4DemodulatedPowerVector(REAL4DemodulatedPowerVector *power            /**< [in] the data to be freed */
					)
{
  
  const CHAR *fn = __func__;   /* store function name for log output */
  UINT4 i;                     /* counter */

  /* free each segment */
  for (i=0;i<power->length;i++) {  

    /* if there is a non-null gridparams structure then free it aswell */
    if (power->segment[i]->gridparams) {
      XLALFree(power->segment[i]->gridparams->grid);
      XLALFree(power->segment[i]->gridparams->prod);
      XLALFree(power->segment[i]->gridparams);
    }
    XLALDestroyREAL4Vector(power->segment[i]->data);
    XLALFree(power->segment[i]);
  }
  XLALFree(power->segment);
  XLALFree(power);

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;

}

/** Free the memory allocated within a BayesianProducts structure 
 *
 */
int XLALFreeBayesianProducts(BayesianProducts *Bayes            /**< [in] the data to be freed */
			     )
{
  const CHAR *fn = __func__;   /* store function name for log output */
  UINT4 i;                     /* counter */

  /* free results */
  XLALDestroyREAL8Vector(Bayes->logBayesFactor_phaseamp_vector);
 
  for (i=0;i<Bayes->ndim;i++) {
    XLALDestroyREAL8Vector(Bayes->logposteriors_phaseamp[i]);
  }
  XLALFree(Bayes->logposteriors_phaseamp);
  XLALFree(Bayes->epoch);
  XLALFree(Bayes);
  LogPrintf(LOG_DEBUG,"%s : freed the Bayesian results\n",fn); 

  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

/** this function initialises the gsl random number generation 
 *
 * If the input seed is zero then a random seed is drawn from 
 * /dev/urandom.
 *
 */
int XLALInitgslrand(gsl_rng **gslrnd,     /**< [out] the gsl random number generator */
		    INT8 seed             /**< [in] the random number generator seed */
		    )
{  
  
  const CHAR *fn = __func__;   /* store function name for log output */
  FILE *devrandom = NULL;      /* pointer to the /dev/urandom file */
  
  /* if the seed is 0 then we draw a random seed from /dev/urandom */
  if (seed == 0) {
    
    /* open /dev/urandom */
    if ((devrandom=fopen("/dev/urandom","r")) == NULL)  {
      LogPrintf(LOG_CRITICAL,"%s: Error, unable to open device /dev/random\n",fn);
      XLAL_ERROR(fn,XLAL_EINVAL);
    }
    
    /* read a random seed */
    if (fread((void*)&seed,sizeof(INT8),1,devrandom) != 1) {
      LogPrintf(LOG_CRITICAL,"%s: Error, unable to read /dev/random\n",fn);
      XLAL_ERROR(fn,XLAL_EINVAL);
    }
    fclose(devrandom);
    
  }
  
  /* setup gsl random number generation */   
  *gslrnd = gsl_rng_alloc(gsl_rng_taus2); 
  gsl_rng_set(*gslrnd,seed);
 
  LogPrintf(LOG_DEBUG,"%s : leaving.\n",fn);
  return XLAL_SUCCESS;
  
}

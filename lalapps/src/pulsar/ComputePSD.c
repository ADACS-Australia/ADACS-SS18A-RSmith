/*
*  Copyright (C) 2007, 2010 Karl Wette
*  Copyright (C) 2007 Badri Krishnan, Iraj Gholami, Reinhard Prix, Alicia Sintes
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
 * \ingroup pulsarApps
 * \author Badri Krishnan, Iraj Gholami, Reinhard Prix, Alicia Sintes, Karl Wette
 * \brief Compute power spectral densities
 */

#include <glob.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_math.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <lal/SFTfileIO.h>
#include <lal/Random.h>
#include <lal/PulsarDataTypes.h>
#include <lal/UserInput.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/LALInitBarycenter.h>
#include <lal/SFTClean.h>

#include <lal/LogPrintf.h>

#include <lalapps.h>

RCSID( "$Id$");

/* ---------- Error codes and messages ---------- */
#define COMPUTEPSDC_ENORM 0
#define COMPUTEPSDC_ESUB  1
#define COMPUTEPSDC_EARG  2
#define COMPUTEPSDC_EBAD  3
#define COMPUTEPSDC_EFILE 4
#define COMPUTEPSDC_ENULL 5
#define COMPUTEPSDC_EMEM  6

#define COMPUTEPSDC_MSGENORM "Normal exit"
#define COMPUTEPSDC_MSGESUB  "Subroutine failed"
#define COMPUTEPSDC_MSGEARG  "Error parsing arguments"
#define COMPUTEPSDC_MSGEBAD  "Bad argument values"
#define COMPUTEPSDC_MSGEFILE "Could not create output file"
#define COMPUTEPSDC_MSGENULL "Null Pointer"
#define COMPUTEPSDC_MSGEMEM  "Out of memory"

/*---------- local defines ---------- */

#define TRUE (1==1)
#define FALSE (1==0)

/* ----- Macros ----- */

/* ---------- local types ---------- */

/** types of mathematical operations */
enum tagMATH_OP_TYPE {
  MATH_OP_ARITHMETIC_SUM = 0,   /**< sum(x)     */
  MATH_OP_ARITHMETIC_MEAN,      /**< sum(x) / n */
  MATH_OP_ARITHMETIC_MEDIAN,    /**< x_1 <= ... x_{n/2} <= .. <= x_n */
  MATH_OP_HARMONIC_SUM,         /**< 1 / sum(1/x) */
  MATH_OP_HARMONIC_MEAN,        /**< n / sum(1/x) */
  MATH_OP_POWERMINUS2_SUM,      /**< 1 / sqrt( sum(1/x/x) )*/
  MATH_OP_POWERMINUS2_MEAN,     /**< 1 / sqrt( sum(1/x/x) /n )*/
  MATH_OP_MINIMUM,              /**< x_1 <= ... */
  MATH_OP_MAXIMUM,              /**< ... <= x_n */
  MATH_OP_LAST
};

/** user input variables */
typedef struct
{
  BOOLEAN help;

  CHAR *inputData;    	/**< directory for input sfts */
  CHAR *outputPSD;    	/**< directory for output sfts */
  CHAR *outputSpectBname;

  REAL8 Freq;		/**< *physical* start frequency to compute PSD for (excluding rngmed wings) */
  REAL8 FreqBand;	/**< *physical* frequency band to compute PSD for (excluding rngmed wings) */

  REAL8 startTime;
  REAL8 endTime;
  CHAR *IFO;
  CHAR  *timeStampsFile;
  LALStringVector *linefiles;
  INT4 blocksRngMed;	/**< number of running-median bins to use */
  INT4 maxBinsClean;

  INT4 PSDmthopSFTs;     /**< for PSD, type of math. operation over SFTs */
  INT4 PSDmthopIFOs;     /**< for PSD, type of math. operation over IFOs */
  BOOLEAN outputNormSFT; /**< output normalised SFT power? */
  INT4 nSFTmthopSFTs;    /**< for norm. SFT, type of math. operation over SFTs */
  INT4 nSFTmthopIFOs;    /**< for norm. SFT, type of math. operation over IFOs */

  REAL8 binSizeHz;       /**< output PSD bin size in Hz */
  INT4  binSize;         /**< output PSD bin size in no. of bins */
  INT4  PSDmthopBins;    /**< for PSD, type of math. operation over bins */
  INT4  nSFTmthopBins;   /**< for norm. SFT, type of math. operation over bins */
  REAL8 binStepHz;       /**< output PSD bin step in Hz */
  INT4  binStep;         /**< output PSD bin step in no. of bins */
  BOOLEAN outFreqBinEnd; /**< output the end frequency of each bin? */

  BOOLEAN dumpMultiPSDVector; /**< output multi-PSD vector over IFOs, timestamps, and frequencies into file(s) */

  REAL8 fStart;		/**< Start Frequency to load from SFT and compute PSD, including wings (it is RECOMMENDED to use --Freq instead) */
  REAL8 fBand;		/**< Frequency Band to load from SFT and compute PSD, including wings (it is RECOMMENDED to use --FreqBand instead) */


} UserVariables_t;

/*---------- empty structs for initializations ----------*/
UserVariables_t empty_UserVariables;
/* ---------- global variables ----------*/

extern int vrbflg;
extern INT4 lalDebugLevel;


/* ---------- local prototypes ---------- */
int initUserVars (int argc, char *argv[], UserVariables_t *uvar);
void LALfwriteSpectrograms ( LALStatus *status, const CHAR *bname, const MultiPSDVector *multiPSD );
static REAL8 math_op(REAL8*, size_t, INT4);
int XLALDumpMultiPSDVector ( const CHAR *outbname, const MultiPSDVector *multiPSDVect );
MultiSFTVector *XLALReadSFTs ( UINT4 *firstBin, UINT4 *lastBin, const UserVariables_t *uvar );

int XLALCropMultiPSDVector ( MultiPSDVector *multiPSDVect, UINT4 firstBin, UINT4 lastBin );

/*============================================================
 * FUNCTION definitions
 *============================================================*/
int
main(int argc, char *argv[])
{
  const char *fn = __func__;
  static LALStatus       status;  /* LALStatus pointer */
  UserVariables_t uvar = empty_UserVariables;

  UINT4 k, numBins, numIFOs, maxNumSFTs, X, alpha;
  REAL8 Freq0, dFreq, normPSD;
  UINT4 finalBinSize, finalBinStep, finalNumBins;

  REAL8Vector *overSFTs = NULL; /* one frequency bin over SFTs */
  REAL8Vector *overIFOs = NULL; /* one frequency bin over IFOs */
  REAL8Vector *finalPSD = NULL; /* math. operation PSD over SFTs and IFOs */
  REAL8Vector *finalNormSFT = NULL; /* normalised SFT power */

  /* LALDebugLevel must be called before anything else */
  lalDebugLevel = 0;
  vrbflg = 1;	/* verbose error-messages */

  /* set LAL error-handler */
  lal_errhandler = LAL_ERR_EXIT;
  if (XLALGetDebugLevel(argc, argv, 'v') != XLAL_SUCCESS)
    return EXIT_FAILURE;

  /* set log-level */
  LogSetLevel ( lalDebugLevel );

  /* register and read user variables */
  if (initUserVars(argc, argv, &uvar) != XLAL_SUCCESS)
    return EXIT_FAILURE;

  /* exit if help was required */
  if (uvar.help)
    return EXIT_SUCCESS;

  UINT4 firstBin, lastBin;
  MultiSFTVector *inputSFTs = NULL;
  if ( ( inputSFTs = XLALReadSFTs ( &firstBin, &lastBin, &uvar ) ) == NULL )
    {
      XLALPrintError ("Call to XLALReadSFTs() failed with xlalErrno = %d\n", xlalErrno );
      return EXIT_FAILURE;
    }

  /* clean sfts if required */
  if ( XLALUserVarWasSet( &uvar.linefiles ) )
    {
      RandomParams *randPar=NULL;
      FILE *fpRand=NULL;
      INT4 seed, ranCount;

      if ( (fpRand = fopen("/dev/urandom", "r")) == NULL ) {
	fprintf(stderr,"Error in opening /dev/urandom" );
	return EXIT_FAILURE;
      }

      if ( (ranCount = fread(&seed, sizeof(seed), 1, fpRand)) != 1 ) {
	fprintf(stderr,"Error in getting random seed" );
	return EXIT_FAILURE;
      }

      LAL_CALL ( LALCreateRandomParams (&status, &randPar, seed), &status );

      LAL_CALL( LALRemoveKnownLinesInMultiSFTVector ( &status, inputSFTs, uvar.maxBinsClean, uvar.blocksRngMed, uvar.linefiles, randPar), &status);
      LAL_CALL ( LALDestroyRandomParams (&status, &randPar), &status);
      fclose(fpRand);
    } /* end cleaning */

  LogPrintf (LOG_DEBUG, "Computing spectrogram and PSD ... ");

  /* get power running-median rngmed[ |data|^2 ] from SFTs */
  MultiPSDVector *multiPSD = NULL;
  LAL_CALL( LALNormalizeMultiSFTVect (&status, &multiPSD, inputSFTs, uvar.blocksRngMed), &status);
  /* restrict this PSD to just the "physical" band if requested using {--Freq, --FreqBand} */
  if ( ( XLALCropMultiPSDVector ( multiPSD, firstBin, lastBin )) != XLAL_SUCCESS ) {
    XLALPrintError ("%s: XLALCropMultiPSDVector (inputPSD, %d, %d) failed with xlalErrno = %d\n", fn, firstBin, lastBin, xlalErrno );
    return EXIT_FAILURE;
  }

  /* start frequency and frequency spacing */
  Freq0 = multiPSD->data[0]->data[0].f0;
  dFreq = multiPSD->data[0]->data[0].deltaF;

  /* number of raw bins in final PSD */
  numBins = multiPSD->data[0]->data[0].data->length;
  if ( (finalPSD = XLALCreateREAL8Vector ( numBins )) == NULL ) {
    LogPrintf (LOG_CRITICAL, "Out of memory!\n");
    return EXIT_FAILURE;
  }

  /* number of IFOs */
  numIFOs = multiPSD->length;
  if ( (overIFOs = XLALCreateREAL8Vector ( numIFOs )) == NULL ) {
    LogPrintf (LOG_CRITICAL, "Out of memory!\n");
    return EXIT_FAILURE;
  }

  /* maximum number of SFTs */
  maxNumSFTs = 0;
  for (X = 0; X < numIFOs; ++X) {
    maxNumSFTs = GSL_MAX(maxNumSFTs, multiPSD->data[X]->length);
  }
  if ( (overSFTs = XLALCreateREAL8Vector ( maxNumSFTs )) == NULL ) {
    LogPrintf (LOG_CRITICAL, "Out of memory!\n");
    return EXIT_FAILURE;
  }

  /* normalize rngmd(power) to get proper *single-sided* PSD: Sn = (2/Tsft) rngmed[|data|^2]] */
  normPSD = 2.0 * dFreq;

  /* loop over frequency bins in final PSD */
  for (k = 0; k < numBins; ++k) {

    /* loop over IFOs */
    for (X = 0; X < numIFOs; ++X) {

      /* number of SFTs for this IFO */
      UINT4 numSFTs = multiPSD->data[X]->length;

      /* copy PSD frequency bins and normalise multiPSD for later use */
      for (alpha = 0; alpha < numSFTs; ++alpha) {
	multiPSD->data[X]->data[alpha].data->data[k] *= normPSD;
	overSFTs->data[alpha] = multiPSD->data[X]->data[alpha].data->data[k];
      }

      /* compute math. operation over SFTs for this IFO */
      overIFOs->data[X] = math_op(overSFTs->data, numSFTs, uvar.PSDmthopSFTs);
      if (XLALIsREAL8FailNaN( overIFOs->data[X] ))
	return EXIT_FAILURE;

    } /* for IFOs X */

    /* compute math. operation over IFOs for this frequency */
    finalPSD->data[k] = math_op(overIFOs->data, numIFOs, uvar.PSDmthopIFOs);
    if (XLALIsREAL8FailNaN( finalPSD->data[k] ))
      return EXIT_FAILURE;

  } /* for freq bins k */
  LogPrintfVerbatim ( LOG_DEBUG, "done.\n");

  /* compute normalised SFT power */
  if (uvar.outputNormSFT) {
    LogPrintf (LOG_DEBUG, "Computing normalised SFT power ... ");

    if ( (finalNormSFT = XLALCreateREAL8Vector ( numBins )) == NULL ) {
      LogPrintf (LOG_CRITICAL, "Out of memory!\n");
      return EXIT_FAILURE;
    }

    /* loop over frequency bins in SFTs */
    for (k = 0; k < numBins; ++k) {

      /* loop over IFOs */
      for (X = 0; X < numIFOs; ++X) {

	/* number of SFTs for this IFO */
	UINT4 numSFTs = inputSFTs->data[X]->length;

	/* compute SFT power */
	for (alpha = 0; alpha < numSFTs; ++alpha) {
	  COMPLEX8 bin = inputSFTs->data[X]->data[alpha].data->data[k];
	  overSFTs->data[alpha] = bin.re*bin.re + bin.im*bin.im;
	}

	/* compute math. operation over SFTs for this IFO */
	overIFOs->data[X] = math_op(overSFTs->data, numSFTs, uvar.nSFTmthopSFTs);
	if (XLALIsREAL8FailNaN( overIFOs->data[X] ))
	  return EXIT_FAILURE;

      } /* over IFOs */

      /* compute math. operation over IFOs for this frequency */
      finalNormSFT->data[k] = math_op(overIFOs->data, numIFOs, uvar.nSFTmthopIFOs);
      if (XLALIsREAL8FailNaN( finalPSD->data[k] ))
	return EXIT_FAILURE;

    } /* over freq bins */
    LogPrintfVerbatim ( LOG_DEBUG, "done.\n");
  }

  /* output spectrograms */
  if ( uvar.outputSpectBname ) {
    LAL_CALL ( LALfwriteSpectrograms ( &status, uvar.outputSpectBname, multiPSD ), &status );
  }

  /* ---------- if user requested it, output complete MultiPSDVector over IFOs X, timestamps and freq-bins into ASCI file(s) */
  if ( uvar.dumpMultiPSDVector ) {
    if ( XLALDumpMultiPSDVector ( uvar.outputPSD, multiPSD ) != XLAL_SUCCESS ) {
      XLALPrintError ("%s: XLALDumpMultiPSDVector() failed, xlalErrnor = %d\n", fn, xlalErrno );
      return EXIT_FAILURE;
    }
  } /* if uvar.dumpMultiPSDVector */


  /* ---------- BINNING if requested ---------- */
  /* work out bin size */
  if (XLALUserVarWasSet(&uvar.binSize)) {
    finalBinSize = uvar.binSize;
  }
  else if (XLALUserVarWasSet(&uvar.binSizeHz)) {
    finalBinSize = (UINT4)floor(uvar.binSizeHz / dFreq + 0.5); /* round to nearest bin */
  }
  else {
    finalBinSize = 1;
  }

  /* work out bin step */
  if (XLALUserVarWasSet(&uvar.binStep)) {
    finalBinStep = uvar.binStep;
  }
  else if (XLALUserVarWasSet(&uvar.binStepHz)) {
    finalBinStep = (UINT4)floor(uvar.binStepHz / dFreq + 0.5); /* round to nearest bin */
  }
  else {
    finalBinStep = finalBinSize;
  }

  /* work out total number of bins */
  finalNumBins = (UINT4)floor((numBins - finalBinSize) / finalBinStep) + 1;

  /* write final PSD to file */
  if (XLALUserVarWasSet(&uvar.outputPSD)) {

    FILE *fpOut = NULL;

    if ((fpOut = fopen(uvar.outputPSD, "wb")) == NULL) {
      LogPrintf ( LOG_CRITICAL, "Unable to open output file %s for writing...exiting \n", uvar.outputPSD );
      return EXIT_FAILURE;
    }

    LogPrintf(LOG_DEBUG, "Printing PSD to file ... ");
    for (k = 0; k < finalNumBins; ++k) {
      UINT4 b = k * finalBinStep;

      REAL8 f0 = Freq0 + b * dFreq;
      REAL8 f1 = f0 + finalBinStep * dFreq;
      fprintf(fpOut, "%f", f0);
      if (uvar.outFreqBinEnd)
	fprintf(fpOut, "   %f", f1);

      REAL8 psd = math_op(&(finalPSD->data[b]), finalBinSize, uvar.PSDmthopBins);
      if (XLALIsREAL8FailNaN( psd ))
	return EXIT_FAILURE;
      fprintf(fpOut, "   %e", psd);

      if (uvar.outputNormSFT) {
	REAL8 nsft = math_op(&(finalNormSFT->data[b]), finalBinSize, uvar.nSFTmthopBins);
	if (XLALIsREAL8FailNaN( nsft ))
	  return EXIT_FAILURE;
	fprintf(fpOut, "   %f", nsft);
      }

      fprintf(fpOut, "\n");
    }
    LogPrintfVerbatim ( LOG_DEBUG, "done.\n");

    fclose(fpOut);

  }

  /* we are now done with the psd */
  LAL_CALL ( LALDestroyMultiPSDVector  ( &status, &multiPSD), &status);
  LAL_CALL ( LALDestroyMultiSFTVector  (&status, &inputSFTs), &status);

  LAL_CALL (LALDestroyUserVars(&status), &status);

  XLALDestroyREAL8Vector ( overSFTs );
  XLALDestroyREAL8Vector ( overIFOs );
  XLALDestroyREAL8Vector ( finalPSD );
  XLALDestroyREAL8Vector ( finalNormSFT );

  LALCheckMemoryLeaks();

  return EXIT_SUCCESS;

} /* main() */

/** compute the various kinds of math. operation */
REAL8 math_op(REAL8* data, size_t length, INT4 type) {

  UINT4 i;
  REAL8 res = 0.0;

  switch (type) {

  case MATH_OP_ARITHMETIC_SUM: /* sum(data) */

    for (i = 0; i < length; ++i) res += *(data++);

    break;

  case MATH_OP_ARITHMETIC_MEAN: /* sum(data)/length  */

    for (i = 0; i < length; ++i) res += *(data++);
    res /= (REAL8)length;

    break;

  case MATH_OP_ARITHMETIC_MEDIAN: /* middle element of sort(data) */

    gsl_sort(data, 1, length);
    if (length/2 == (length+1)/2) /* length is even */ {
      res = (data[length/2] + data[length/2+1])/2;
    }
    else /* length is odd */ {
      res = data[length/2];
    }

    break;

  case MATH_OP_HARMONIC_SUM: /* 1 / sum(1 / data) */

    for (i = 0; i < length; ++i) res += 1.0 / *(data++);
    res = 1.0 / res;

    break;

  case MATH_OP_HARMONIC_MEAN: /* length / sum(1 / data) */

    for (i = 0; i < length; ++i) res += 1.0 / *(data++);
    res = (REAL8)length / res;

    break;

  case MATH_OP_POWERMINUS2_SUM: /*   1 / sqrt ( sum(1 / data/data) )*/

    for (i = 0; i < length; ++i) res += 1.0 / (data[i]*data[i]);
    res = 1.0 / sqrt(res);

    break;

   case MATH_OP_POWERMINUS2_MEAN: /*   1 / sqrt ( sum(1/data/data) / length )*/

    for (i = 0; i < length; ++i) res += 1.0 / (data[i]*data[i]);
    res = 1.0 / sqrt(res / (REAL8)length);

    break;

  case MATH_OP_MINIMUM: /* first element of sort(data) */

    gsl_sort(data, 1, length);
    res = data[0];
    break;

  case MATH_OP_MAXIMUM: /* first element of sort(data) */

    gsl_sort(data, 1, length);
    res = data[length-1];
    break;

  default:

    XLALPrintError("'%i' is not a valid math. operation", type);
    XLAL_ERROR_REAL8(__func__, XLAL_EINVAL);

  }

  return res;

}


/** register all "user-variables" */
int
initUserVars (int argc, char *argv[], UserVariables_t *uvar)
{

  /* set a few defaults */
  uvar->help = FALSE;

  uvar->maxBinsClean = 100;
  uvar->blocksRngMed = 101;

  uvar->startTime = 0.0;
  uvar->endTime = 0.0;

  uvar->inputData = NULL;

  uvar->IFO = NULL;

  /* default: read all SFT bins */
  uvar->fStart = -1;
  uvar->fBand = 0;

  uvar->outputPSD = NULL;
  uvar->outputNormSFT = FALSE;
  uvar->outFreqBinEnd = FALSE;

  uvar->PSDmthopSFTs = MATH_OP_HARMONIC_MEAN;
  uvar->PSDmthopIFOs = MATH_OP_HARMONIC_SUM;

  uvar->nSFTmthopSFTs = MATH_OP_ARITHMETIC_MEAN;
  uvar->nSFTmthopIFOs = MATH_OP_MAXIMUM;
  uvar->dumpMultiPSDVector = FALSE;

  uvar->binSizeHz = 0.0;
  uvar->binSize   = 1;
  uvar->PSDmthopBins  = MATH_OP_ARITHMETIC_MEDIAN;
  uvar->nSFTmthopBins = MATH_OP_MAXIMUM;
  uvar->binStep   = 0.0;
  uvar->binStep   = 1;

  /* register user input variables */
  XLALregBOOLUserStruct  (help,             'h', UVAR_HELP,     "Print this message" );
  XLALregSTRINGUserStruct(inputData,        'i', UVAR_REQUIRED, "Input SFT pattern");
  XLALregSTRINGUserStruct(outputPSD,        'o', UVAR_OPTIONAL, "Output PSD into this file");
  XLALregSTRINGUserStruct(outputSpectBname,  0 , UVAR_OPTIONAL, "Filename-base for (binary) spectrograms (one per IFO)");

  XLALregREALUserStruct  (Freq,              0,  UVAR_OPTIONAL, "physical start frequency to compute PSD for (excluding rngmed wings)");
  XLALregREALUserStruct  (FreqBand,          0,  UVAR_OPTIONAL, "physical frequency band to compute PSD for (excluding rngmed wings)");

  XLALregREALUserStruct  (startTime,        's', UVAR_OPTIONAL, "GPS start time");
  XLALregREALUserStruct  (endTime,          'e', UVAR_OPTIONAL, "GPS end time");
  XLALregSTRINGUserStruct(timeStampsFile,   't', UVAR_OPTIONAL, "Time-stamps file");
  XLALregSTRINGUserStruct(IFO,               0 , UVAR_OPTIONAL, "Detector filter");

  XLALregINTUserStruct   (blocksRngMed,     'w', UVAR_OPTIONAL, "Running Median window size");

  XLALregINTUserStruct   (PSDmthopSFTs,     'S', UVAR_OPTIONAL, "For PSD, type of math. operation over SFTs: "
                                                                "0=arith-sum, 1=arith-mean, 2=arith-median, "
                                                                "3=harm-sum, 4=harm-mean, "
                                                                "5=power-2-sum, 6=power-2-mean, "
                                                                "7=min, 8=max");
  XLALregINTUserStruct   (PSDmthopIFOs,     'I', UVAR_OPTIONAL, "For PSD, type of math. op. over IFOs: "
                                                                "see --PSDmthopSFTs");
  XLALregBOOLUserStruct  (outputNormSFT,    'n', UVAR_OPTIONAL, "Output normalised SFT power to PSD file");
  XLALregINTUserStruct   (nSFTmthopSFTs,    'N', UVAR_OPTIONAL, "For norm. SFT, type of math. op. over SFTs: "
                                                                "see --PSDmthopSFTs");
  XLALregINTUserStruct   (nSFTmthopIFOs,    'J', UVAR_OPTIONAL, "For norm. SFT, type of math. op. over IFOs: "
                                                                "see --PSDmthopSFTs");

  XLALregINTUserStruct   (binSize,          'z', UVAR_OPTIONAL, "Bin the output into bins of size (in number of bins)");
  XLALregREALUserStruct  (binSizeHz,        'Z', UVAR_OPTIONAL, "Bin the output into bins of size (in Hz)");
  XLALregINTUserStruct   (PSDmthopBins,     'A', UVAR_OPTIONAL, "If binning, for PSD type of math. op. over bins: "
                                                                "see --PSDmthopSFTs");
  XLALregINTUserStruct   (nSFTmthopBins,    'B', UVAR_OPTIONAL, "If binning, for norm. SFT type of math. op. over bins: "
                                                                "see --PSDmthopSFTs");
  XLALregINTUserStruct   (binStep,          'p', UVAR_OPTIONAL, "If binning, step size to move bin along "
                                                                "(in number of bins, default is bin size)");
  XLALregREALUserStruct  (binStepHz,        'P', UVAR_OPTIONAL, "If binning, step size to move bin along "
                                                                "(in Hz, default is bin size)");
  XLALregBOOLUserStruct  (outFreqBinEnd,    'E', UVAR_OPTIONAL, "Output the end frequency of each bin");

  XLALregINTUserStruct   (maxBinsClean,     'm', UVAR_OPTIONAL, "Maximum Cleaning Bins");
  XLALregLISTUserStruct  (linefiles,         0 , UVAR_OPTIONAL, "Comma separated list of linefiles "
								"(names must contain IFO name)");

  XLALregBOOLUserStruct  (dumpMultiPSDVector,'d',UVAR_OPTIONAL, "Output multi-PSD vector over IFOs, timestamps, and frequencies into file(s) '<outputPSD>-IFO'");

  /* ----- developer options ---------- */
  XLALregREALUserStruct  (fStart,           'f', UVAR_DEVELOPER, "Start Frequency to load from SFT and compute PSD, including rngmed wings (BETTER: use --Freq instead)");
  XLALregREALUserStruct  (fBand,            'b', UVAR_DEVELOPER, "Frequency Band to load from SFT and compute PSD, including rngmed wings (BETTER: use --FreqBand instead)");



  /* read all command line variables */
  if (XLALUserVarReadAllInput(argc, argv) != XLAL_SUCCESS)
    return XLAL_FAILURE;

  /* check user-input consistency */
  if (XLALUserVarWasSet(&(uvar->PSDmthopSFTs)) && !(0 <= uvar->PSDmthopSFTs && uvar->PSDmthopSFTs < MATH_OP_LAST)) {
    XLALPrintError("ERROR: --PSDmthopSFTs(-S) must be between 0 and %i", MATH_OP_LAST - 1);
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->PSDmthopIFOs)) && !(0 <= uvar->PSDmthopIFOs && uvar->PSDmthopIFOs < MATH_OP_LAST)) {
    XLALPrintError("ERROR: --PSDmthopIFOs(-I) must be between 0 and %i", MATH_OP_LAST - 1);
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->nSFTmthopSFTs)) && !(0 <= uvar->nSFTmthopSFTs && uvar->nSFTmthopSFTs < MATH_OP_LAST)) {
    XLALPrintError("ERROR: --nSFTmthopSFTs(-N) must be between 0 and %i", MATH_OP_LAST - 1);
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->nSFTmthopIFOs)) && !(0 <= uvar->nSFTmthopIFOs && uvar->nSFTmthopIFOs < MATH_OP_LAST)) {
    XLALPrintError("ERROR: --nSFTmthopIFOs(-J) must be between 0 and %i", MATH_OP_LAST - 1);
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->PSDmthopBins)) && !(0 <= uvar->PSDmthopBins && uvar->PSDmthopBins < MATH_OP_LAST)) {
    XLALPrintError("ERROR: --PSDmthopBins(-A) must be between 0 and %i", MATH_OP_LAST - 1);
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->nSFTmthopBins)) && !(0 <= uvar->nSFTmthopBins && uvar->nSFTmthopBins < MATH_OP_LAST)) {
    XLALPrintError("ERROR: --nSFTmthopBins(-B) must be between 0 and %i", MATH_OP_LAST - 1);
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->binSize)) && XLALUserVarWasSet(&(uvar->binSizeHz))) {
    XLALPrintError("ERROR: --binSize(-z) and --binSizeHz(-Z) are mutually exclusive");
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->binSize)) && uvar->binSize <= 0) {
    XLALPrintError("ERROR: --binSize(-z) must be strictly positive");
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->binSizeHz)) && uvar->binSizeHz <= 0.0) {
    XLALPrintError("ERROR: --binSizeHz(-Z) must be strictly positive");
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->binStep)) && XLALUserVarWasSet(&(uvar->binStepHz))) {
    XLALPrintError("ERROR: --binStep(-p) and --binStepHz(-P) are mutually exclusive");
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->binStep)) && uvar->binStep <= 0) {
    XLALPrintError("ERROR: --binStep(-p) must be strictly positive");
    return XLAL_FAILURE;
  }
  if (XLALUserVarWasSet(&(uvar->binStepHz)) && uvar->binStepHz <= 0.0) {
    XLALPrintError("ERROR: --binStepHz(-P) must be strictly positive");
    return XLAL_FAILURE;
  }

  return XLAL_SUCCESS;

} /* initUserVars() */


/** Write a multi-PSD into spectrograms for each IFO.
 * Using gnuplot 'binary' matrix format
 * The filename for each IFO is generated as 'bname-IFO'
 */
void
LALfwriteSpectrograms ( LALStatus *status, const CHAR* bname, const MultiPSDVector *multiPSD )
{
  UINT4 X;
  CHAR *fname;
  float num, *row_data;		/* cast to float for writing (gnuplot binary format) */
  FILE *fp;

  INITSTATUS( status, "LALfwriteSpectrograms", rcsid );
  ATTATCHSTATUSPTR (status);

  if ( !bname || !multiPSD || multiPSD->length == 0 ) {
    ABORT ( status, COMPUTEPSDC_ENULL, COMPUTEPSDC_MSGENULL );
  }

  /* loop over IFOs */
  for ( X = 0; X < multiPSD->length ; X ++ )
    {
      UINT4 len = strlen ( bname ) + 4;	/* append '-XN' to get IFO-specific filename */
      UINT4 numSFTs, numBins;
      UINT4 j, k;
      const CHAR *tmp;
      REAL8 f0, df;

      numSFTs = multiPSD->data[X]->length;
      numBins = multiPSD->data[X]->data[0].data->length;

      /* allocate memory for data row-vector */
      if ( ( row_data = LALMalloc ( numBins * sizeof(float) )) == NULL ) {
	ABORT ( status, COMPUTEPSDC_EMEM, COMPUTEPSDC_MSGEMEM );
      }

      if ( ( fname = LALMalloc ( len * sizeof(CHAR) )) == NULL ) {
	LALFree ( row_data );
	ABORT ( status, COMPUTEPSDC_EMEM, COMPUTEPSDC_MSGEMEM );
      }
      tmp = multiPSD->data[X]->data[0].name;
      sprintf ( fname, "%s-%c%c", bname, tmp[0], tmp[1] );

      if ( ( fp = fopen( fname, "wb" ))  == NULL ) {
	LogPrintf (LOG_CRITICAL, "Failed to open spectrogram file '%s' for writing!\n", fname );
	goto failed;

      }

      /* write number of columns: i.e. frequency-bins */
      num = (float)numBins;
      if ((fwrite((char *) &num, sizeof(float), 1, fp)) != 1) {
	LogPrintf (LOG_CRITICAL, "Failed to fwrite() to spectrogram file '%s'\n", fname );
	goto failed;
      }

      /* write frequencies as column-titles */
      f0 = multiPSD->data[X]->data[0].f0;
      df = multiPSD->data[X]->data[0].deltaF;
      for ( k=0; k < numBins; k ++ )
	row_data[k] = (float) ( f0 + 1.0 * k * df );
      if ( fwrite((char *) row_data, sizeof(float), numBins, fp) != numBins ) {
	LogPrintf (LOG_CRITICAL, "Failed to fwrite() to spectrogram file '%s'\n", fname );
	goto failed;
      }

      /* write PSDs of successive SFTs in rows, first column is GPS-time in seconds */
      for ( j = 0; j < numSFTs ; j ++ )
	{
	  num = (float) multiPSD->data[X]->data[j].epoch.gpsSeconds;
	  for ( k = 0; k < numBins; k ++ )
	    row_data[k] = (float) sqrt ( multiPSD->data[X]->data[j].data->data[k] );

	  if ( ( fwrite((char *) &num, sizeof(float), 1, fp) != 1 ) ||
	       ( fwrite((char *) row_data, sizeof(float), numBins, fp) != numBins ) ) {
	    LogPrintf (LOG_CRITICAL, "Failed to fwrite() to spectrogram file '%s'\n", fname );
	    goto failed;
	  }

	} /* for j < numSFTs */

      fclose ( fp );
      LALFree ( fname );
      LALFree ( row_data );

    } /* for X < numIFOs */

  DETATCHSTATUSPTR (status);
  RETURN (status);

  /* cleanup and exit on write-error */
 failed:
  if ( fname ) LALFree ( fname );
  if ( row_data ) LALFree ( row_data );
  if ( fp ) fclose ( fp );
  ABORT ( status, COMPUTEPSDC_EFILE, COMPUTEPSDC_MSGEFILE );

} /* LALfwriteSpectrograms() */

/** Dump complete multi-PSDVector over IFOs, timestamps and frequency-bins into
 *  per-IFO ASCII output-files <outbname-IFO>
 *
 */
int
XLALDumpMultiPSDVector ( const CHAR *outbname,			/**< output basename <outbname> */
                         const MultiPSDVector *multiPSDVect	/**< multi-psd vector to output */
                  )
{
  const char *fn = __func__;

  /* check input consistency */
  if ( outbname == NULL ) {
    XLALPrintError ("%s: NULL input 'outbname'\n", fn );
    XLAL_ERROR ( fn, XLAL_EINVAL );
  }
  if ( multiPSDVect == NULL ) {
    XLALPrintError ("%s: NULL input 'multiPSDVect'\n", fn );
    XLAL_ERROR ( fn, XLAL_EINVAL );
  }
  if ( multiPSDVect->length == 0 || multiPSDVect->data==0 ) {
    XLALPrintError ("%s: invalid multiPSDVect input (length=0 or data=NULL)\n", fn );
    XLAL_ERROR ( fn, XLAL_EINVAL );
  }

  CHAR *fname;
  FILE *fp;

  UINT4 len = strlen ( outbname ) + 4;
  if ( ( fname = XLALMalloc ( len * sizeof(CHAR) )) == NULL ) {
    XLALPrintError ("%s: XLALMalloc(%d) failed.\n", fn, len );
    XLAL_ERROR ( fn, XLAL_ENOMEM);
  }

  UINT4 numIFOs = multiPSDVect->length;
  UINT4 X;
  for ( X = 0; X < numIFOs; X ++ )
    {
      PSDVector *thisPSDVect = multiPSDVect->data[X];
      char buf[100];

      sprintf ( fname, "%s-%c%c", outbname, thisPSDVect->data[0].name[0], thisPSDVect->data[0].name[1] );

      if ( ( fp = fopen( fname, "wb" ))  == NULL ) {
        XLALPrintError ("%s: Failed to open PSDperSFT file '%s' for writing!\n", fn, fname );
        XLALFree ( fname );
        XLAL_ERROR ( fn, XLAL_ESYS );
      }

      REAL8 f0       = thisPSDVect->data[0].f0;
      REAL8 dFreq    = thisPSDVect->data[0].deltaF;
      UINT4 numFreqs = thisPSDVect->data[0].data->length;
      UINT4 iFreq;

      /* write comment header line into this output file */
      /* FIXME: output code-version/cmdline/history info */
      fprintf(fp,"%%%% first line holds frequencies [Hz] of PSD-Columns\n");
      /* loop over frequency and output comnment-header markers */
      fprintf(fp,"%%%%%-17s", "dummy");
      for ( iFreq = 0; iFreq < numFreqs; iFreq ++ )
        {
          sprintf (buf, "f%d [Hz]", iFreq + 1 );
          fprintf (fp, " %-21s", buf );
        }
      fprintf (fp, "\n");

      /* write parseable header-line giving bin frequencies for PSDs */
      fprintf (fp, "%-19d", -1 );
      for (iFreq = 0; iFreq < numFreqs; iFreq++ )
        fprintf (fp, " %-21.16g", f0 + iFreq * dFreq );
      fprintf (fp, "\n\n\n");

      /* output another header line describing the following format "ti[GPS] PSD(f1) ... " */
      fprintf(fp,"%%%%%-17s", "ti[GPS]");
      for ( iFreq = 0; iFreq < numFreqs; iFreq ++ )
        {
          sprintf (buf, "PSD(f%d)", iFreq + 1 );
          fprintf (fp, " %-21s", buf );
        }
      fprintf (fp, "\n");

      /* loop over timestamps: dump all PSDs over frequencies into one line */
      UINT4 numTS = thisPSDVect->length;
      UINT4 iTS;
      for ( iTS = 0; iTS < numTS; iTS++ )
        {
          REAL8FrequencySeries *thisPSD = &thisPSDVect->data[iTS];

          /* first output timestamp GPS time for this line */
          REAL8 tGPS = XLALGPSGetREAL8( &thisPSD->epoch );
          fprintf (fp, "%-19.18g", tGPS );

          /* some internal consistency/paranoia checks */
          if ( ( f0 != thisPSD->f0) || ( dFreq != thisPSD->deltaF ) || (numFreqs != thisPSD->data->length ) ) {
            XLALPrintError ("%s: %d-th timestamp %f: inconsistent PSDVector: f0 = %g : %g,  dFreq = %g : %g, numFreqs = %d : %d \n",
                            fn, iTS, tGPS, f0, thisPSD->f0, dFreq, thisPSD->deltaF, numFreqs, thisPSD->data->length );
            XLALFree ( fname );
            fclose ( fp );
            XLAL_ERROR ( fn, XLAL_EDOM );
          }

          /* loop over all frequencies and dump PSD-value */
          for ( iFreq = 0; iFreq < numFreqs; iFreq ++ )
            fprintf (fp, " %-21.16g", thisPSD->data->data[iFreq] );

          fprintf (fp, "\n");

        } /* for iTS < numTS */

      fclose ( fp );

    } /* for X < numIFOs */

  XLALFree ( fname );

  return XLAL_SUCCESS;

} /* XLALDumpMultiPSDVector() */


/** Load all SFTs according to user-input, returns multi-SFT vector.
 *
 * Returns 'effective' range of SFT-bins [firstBin, lastBin], which which the PSD will be estimated:
 *   - if the user input {fStart, fBand} then these are loaded from SFTs and directly translated into bins
 *   - if user input {Freq, FreqBand}, we load a wider frequency-band ADDING running-median/2 on either side
 *     from the SFTs, and firstBind, lastBin correspond to {Freq,FreqBand} (rounded to closest bins)
 *
 */
MultiSFTVector *
XLALReadSFTs ( UINT4 *firstBin,			/**< [out] first PSD bin for output */
               UINT4 *lastBin,			/**< [out] last PSD bin for output */
               const UserVariables_t *uvar	/**< [in] complete user-input */
               )
{
  const char *fn = __func__;

  SFTCatalog *catalog = NULL;
  SFTConstraints constraints = empty_SFTConstraints;
  LIGOTimeGPS startTimeGPS, endTimeGPS;
  LIGOTimeGPSVector *inputTimeStampsVector = NULL;

  /* check input */
  if ( !uvar || !uvar->inputData ) {
    XLALPrintError ("%s: invalid NULL input 'uvar' or 'uvar->inputData'\n", fn );
    XLAL_ERROR_NULL ( fn, XLAL_EINVAL );
  }

  /* set detector constraint */
  if ( XLALUserVarWasSet ( &uvar->IFO ) )
    constraints.detector = uvar->IFO;
  else
    constraints.detector = NULL;

  if ( XLALUserVarWasSet( &uvar->startTime ) ) {
    XLALGPSSetREAL8 ( &startTimeGPS, uvar->startTime);
    constraints.startTime = &startTimeGPS;
  }

  if ( XLALUserVarWasSet( &uvar->endTime ) ) {
    XLALGPSSetREAL8 ( &endTimeGPS, uvar->endTime);
    constraints.endTime = &endTimeGPS;
  }

  if ( XLALUserVarWasSet( &uvar->timeStampsFile ) ) {
    if ( (inputTimeStampsVector = XLALReadTimestampsFile ( uvar->timeStampsFile )) == NULL )
      XLAL_ERROR_NULL ( fn, XLAL_EFUNC );

    constraints.timestamps = inputTimeStampsVector;
  }

  /* get sft catalog */
  LogPrintf ( LOG_DEBUG, "Finding all SFTs to load ... ");
  LALStatus status = blank_status;
  LALSFTdataFind ( &status, &catalog, uvar->inputData, &constraints);
  if ( status.statusCode != 0 ) {
    XLALPrintError ("%s: LALSFTdataFind() failed with statusCode = %d\n", fn, status.statusCode );
    XLAL_ERROR_NULL ( fn, XLAL_EFAILED );
  }
  if ( (catalog == NULL) || (catalog->length == 0) ) {
    XLALPrintError ("%s: Unable to match any SFTs with pattern '%s'\n", fn, uvar->inputData );
    XLAL_ERROR_NULL ( fn, XLAL_EFAILED );
  }
  LogPrintfVerbatim ( LOG_DEBUG, "done (found %i SFTs).\n", catalog->length);

  /* now we can free the inputTimeStampsVector */
  if ( inputTimeStampsVector )
    XLALDestroyTimestampVector ( inputTimeStampsVector );

  /* ----- some user-input consistency checks */
  BOOLEAN have_fStart   = XLALUserVarWasSet ( &uvar->fStart );
  BOOLEAN have_Freq     = XLALUserVarWasSet ( &uvar->Freq );
  BOOLEAN have_fBand    = XLALUserVarWasSet ( &uvar->fBand );
  BOOLEAN have_FreqBand = XLALUserVarWasSet ( &uvar->FreqBand );
  if ( have_fStart && have_Freq ) {
    XLALPrintError ("%s: use only one of --fStart OR --Freq (see --help)\n", fn );
    XLAL_ERROR_NULL ( fn, XLAL_EINVAL );
  }
  if ( have_fBand && have_FreqBand ) {
    XLALPrintError ("%s: use only one of --fBand OR --FreqBand (see --help)\n", fn );
    XLAL_ERROR_NULL ( fn, XLAL_EINVAL );
  }
  if ( ( have_fStart && have_FreqBand ) || ( have_Freq && have_fBand ) ) {
    XLALPrintError ("%s: don't mix {--fStart,--fBand} with {--Freq,--FreqBand} inputs (see --help)\n", fn );
    XLAL_ERROR_NULL ( fn, XLAL_EINVAL );
  }
  /* ---------- figure out the right frequency-band to read from the SFTs, depending on user-input ----- */
  REAL8 fMin, fMax;
  UINT4 binsOffset; /* rngmed bin offset from start and end */
  UINT4 binsBand;   /* width of physical FreqBand in bins */
  if ( have_Freq )
    {
      REAL8 dFreq = catalog->data[0].header.deltaF;
      binsOffset = uvar->blocksRngMed / 2 + 1;	/* truncates down plus add one bin extra safety! */
      binsBand   = ceil ( (uvar->FreqBand - 1e-9) / dFreq ) + 1; /* round up ! */

      REAL8 rngmedSideBand = binsOffset * dFreq;

      fMin = uvar->Freq - rngmedSideBand;
      fMax = uvar->Freq + uvar->FreqBand + rngmedSideBand;
    }
  else	/* NOTE: if no user-input on freq-band, we fall back to defaults on {fStart, fBand} */
    {
      fMin = uvar->fStart;
      fMax = uvar->fStart + uvar->fBand;
      binsOffset = 0;	/* no truncation of rngmed sidebands */
    }

  /* ---------- read the sfts ---------- */
  LogPrintf (LOG_DEBUG, "Loading all SFTs ... ");
  MultiSFTVector *multi_sfts;
  if ( ( multi_sfts = XLALLoadMultiSFTs ( catalog, fMin, fMax ) ) == NULL ) {
    XLALPrintError ("%s: XLALLoadMultiSFTs( %f, %f ) failed with xlalErrno = %d\n", fn, fMin, fMax, xlalErrno );
    XLAL_ERROR_NULL ( fn, XLAL_EFUNC );
  }
  XLALDestroySFTCatalog ( &catalog );
  LogPrintfVerbatim ( LOG_DEBUG, "done.\n");
  /* ---------- end loading SFTs ---------- */

  /* figure out effective PSD bin-boundaries for user */
  UINT4 numBins = multi_sfts->data[0]->data[0].data->length;
  INT4 bin0, bin1;
  if ( have_Freq )
    {
      bin0 = 0 + binsOffset;
      bin1 = bin0 + binsBand - 1;
    }
  else	/* output all bins loaded from SFTs (includes rngmed-sidebands) */
    {
      bin0 = 0;
      bin1 = numBins - 1;
    }

  /* return results */
  (*firstBin) = (UINT4) bin0;
  (*lastBin ) = (UINT4) bin1;

  XLALPrintInfo ("%s: loaded SFTs have %d bins, effective PSD output band is [%d, %d]\n", fn, numBins, bin0, bin1 );

  return multi_sfts;

} /* XLALReadSFTs() */

/** Function that *truncates the PSD in place* to the requested frequency-bin interval [firstBin, lastBin] for the given multiPSDVector.
 */
int
XLALCropMultiPSDVector ( MultiPSDVector *multiPSDVect,
                         UINT4 firstBin,
                         UINT4 lastBin
                         )
{
  const char *fn = __func__;

  /* check user input */
  if ( !multiPSDVect ) {
    XLALPrintError ("%s: invalid NULL input 'multiPSDVect'\n", fn );
    XLAL_ERROR ( fn, XLAL_EINVAL );
  }
  if ( lastBin < firstBin ) {
    XLALPrintError ("%s: empty bin interval requested [%d, %d]\n", fn, firstBin, lastBin );
    XLAL_ERROR ( fn, XLAL_EDOM );
  }

  UINT4 numIFOs = multiPSDVect->length;
  UINT4 numBins = multiPSDVect->data[0]->data[0].data->length;

  if ( (firstBin >= numBins) || (lastBin >= numBins ) ) {
    XLALPrintError ("%s: requested bin-interval [%d, %d] outside of PSD bins [0, %d]\n", fn, firstBin, lastBin, 0, numBins - 1 );
    XLAL_ERROR ( fn, XLAL_EDOM );
  }

  /* ----- check if there's anything to do at all? ----- */
  if ( (firstBin == 0)  && (lastBin == numBins - 1) )
    return XLAL_SUCCESS;

  REAL8 f0    = multiPSDVect->data[0]->data[0].f0;
  REAL8 dFreq = multiPSDVect->data[0]->data[0].deltaF;

  /* ----- loop over detectors, timestamps, then crop each PSD ----- */
  UINT4 X;
  for ( X=0; X < numIFOs; X ++ )
    {
      PSDVector *thisPSDVect = multiPSDVect->data[X];
      UINT4 numTS   = thisPSDVect->length;

      UINT4 iTS;
      for ( iTS = 0; iTS < numTS; iTS ++ )
        {
          REAL8FrequencySeries *thisPSD = &thisPSDVect->data[iTS];

          if ( numBins != thisPSD->data->length ) {
            XLALPrintError ("%s: inconsistent number of frequency-bins across multiPSDVector: X=%d, iTS=%d: numBins = %d != %d\n",
                            fn, X, iTS, numBins, thisPSD->data->length );
            XLAL_ERROR ( fn, XLAL_EDOM );
          }

          UINT4 numNewBins = lastBin - firstBin + 1;
          /* now do some clever memory-sane cropping on this PSD */
          if ( firstBin > 0 )
            {
              void *dest = thisPSD->data->data;	/* always copy to the beginning of array */
              void *src  = thisPSD->data->data + firstBin;	/* we're copying from firstBin on ... */

              if ( dest != memmove(dest, src, numNewBins * sizeof(*thisPSD->data->data)) ) { /* memmove() handles overlapping memory correctly */
                XLALPrintError ("%s: something failed in moving PSD data with memmove()\n", fn );
                XLAL_ERROR ( fn, XLAL_EFAILED );
              }
            }
          /* truncate array to new size */
          thisPSD->data->length = numNewBins;

          /* set correct start-frequency in cropped PSD */
          thisPSD->f0 = f0 + firstBin * dFreq;

        } /* for iTS < numTS */

    } /* for X < numIFOs */

  /* that should be all ... */
  return XLAL_SUCCESS;

} /* XLALCropMultiPSDVector() */

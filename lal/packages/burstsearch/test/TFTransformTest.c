/*----------------------------------------------------------------------- 
 * 
 * File Name: TFTransformTest.c
 * 
 * Author: Eanna Flanagan
 * 
 * Revision: $Id$
 * 
 *----------------------------------------------------------------------- 
 * 
 * NAME 
 * main()
 *
 * SYNOPSIS 
 * 
 * DESCRIPTION 
 * Test suite for functions in TFTransform.c
 * 
 * DIAGNOSTICS
 * Writes PASS or FAIL to stdout as tests are passed or failed.
 *
 * CALLS
 *
 * NOTES
 *
 *-----------------------------------------------------------------------
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "LALStdlib.h"
#include "SeqFactories.h"
#include "VectorOps.h"
#include "TFTransform.h"
#include "PrintVector.h"


#define _CODES(x) #x
#define CODES(x) _CODES(x)


NRCSID (MAIN, "$Id$");


extern char *optarg;
extern int   optind;

INT4 lalDebugLevel = 1;   /* set to 2 to get full status information for tests */
INT4 verbose    = 1;

static void
Usage (const char *program, int exitflag);

static void
ParseOptions (int argc, char *argv[]);

static void
TestStatus (LALStatus *status, const char *expectedCodes, int exitCode);

static REAL4 ff(REAL4 w);   /* simple function used to construct a waveform */


int
main (int argc, char *argv[])
{
  const INT4 ntot   = 1000;   /* total number of points in time domain */
  const REAL8 alpha = 0.27;   /* ln(nt)/ln(ntot) */
  const REAL8 beta  = 0.2;    /* nf_actual / nf_total */

  static LALStatus                 status; 
  TFPlaneParams                 params;
  VerticalTFTransformIn         transformparams;
  HorizontalTFTransformIn       transformparams1;
  REAL4TimeSeries               tseries;
  COMPLEX8FrequencySeries       fseries;
  COMPLEX8TimeFrequencyPlane    *tfp=NULL;
  RealDFTParams                 *dftparams1=NULL;


  INT4                          i;
  INT4                          tseglength;
  INT4                          fseglength;
  INT4                          nt;
  INT4                          nf;
  INT4                          nforig;

  /*
   *
   * Parse the command line options
   *
   */

  ParseOptions (argc, argv);



  /* compute parameters */

  nt = (INT4)(exp( alpha * log( (REAL8)(ntot))));
  nforig = ntot / (2 * nt);
  nf = (INT4)(beta * (REAL8)(nforig));

  if(verbose)
    {
      printf("Total number of data points :    %d\n",ntot);
      printf("Number of time bins         :    %d\n",nt);
      printf("Number of frequency bins    :    %d\n",nf);
      printf("Original # of freq bins     :    %d\n",nforig);
    }


    

  /* 
   *  
   *  Set up input time series
   *
   */

  tseries.epoch.gpsSeconds=0;
  tseries.epoch.gpsNanoSeconds=0;
  tseries.deltaT = 0.001;  /* 1 kHz sampling */
  tseries.f0 = 0.0;
  tseries.name = NULL;
  tseries.sampleUnits=NULL;
  tseries.data=NULL;


  LALSCreateVector (&status, &(tseries.data), ntot);
  TestStatus (&status, CODES(0), 1);

  for(i=0; i< tseries.data->length; i++)
    {
      tseries.data->data[i] = ff( (REAL4)(i)/ (REAL4)(ntot));
    };

  /*
   *
   *   Set up time-frequency plane structure
   *
   */

  /* setup parameters structure for creating TF plane */
  params.timeBins = nt;
  params.freqBins = nf;
  params.deltaT = tseries.deltaT * 2.0 * (REAL8)(nforig);
  params.flow = 0.0;

  /* Create TF plane  */
  LALCreateTFPlane( &status, &tfp, &params);
  TestStatus (&status, CODES(0), 1);


  /*  
   *
   *
   *  Test of vertical TFTransform
   *
   *
   */
  
  {
    LALWindowParams winParams;
    winParams.type=Rectangular;

    tseglength = 2 * (INT4)( (params.deltaT) / (2.0*(tseries.deltaT)) );
    winParams.length=tseglength;

    /* setup input structure for computing TF transform */
    transformparams.startT=0;
    transformparams.dftParams=NULL;
    LALCreateRealDFTParams( &status, &(transformparams.dftParams), &winParams, 1); 
    TestStatus (&status, CODES(0), 1);


    /* Compute TF transform */
    if(verbose)
      {
	printf("Computing vertical time-frequency plane\n");
      }
    LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
    TestStatus (&status, CODES(0), 1);


    /* Destroy stuff */
    LALDestroyRealDFTParams( &status, &(transformparams.dftParams));
    TestStatus (&status, CODES(0), 1);
  }





  /*  
   *
   *
   *  Test of horizontal TFTransform
   *
   *
   */

  {
    LALWindowParams winParams;
    LALWindowParams winParams1;
    winParams.type=Rectangular;
    winParams.length=ntot;
    
    fseries.data=NULL;
    LALCCreateVector( &status, &(fseries.data), ntot/2+1);
    TestStatus (&status, CODES(0), 1);
      
    LALCreateRealDFTParams( &status, &dftparams1, &winParams, 1);
    TestStatus (&status, CODES(0), 1);
    
    if(verbose)
      {
	printf("Computing FFT of time domain data\n");
      }
    LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
    TestStatus (&status, CODES(0), 1);

    fseglength = (INT4)( 0.5+1/(tfp->params->deltaT * fseries.deltaF));

    /* setup input structure for computing TF transform */
    transformparams1.startT=0;  /* not used for horizontal transforms */
    transformparams1.dftParams=NULL;

    winParams1.type=Rectangular;
    winParams1.length=fseglength;

    LALCreateComplexDFTParams( &status, &(transformparams1.dftParams),
                            &winParams1,-1); 
    TestStatus (&status, CODES(0), 1);


    
    if(verbose)
      {
	printf("Computing horizontal time-frequency plane\n");
      }
    /* Compute TF transform */
    LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
    TestStatus (&status, CODES(0), 1);


    /* Destroy stuff */
    
    LALDestroyComplexDFTParams( &status, &(transformparams1.dftParams));
    TestStatus (&status, CODES(0), 1);
    
    LALDestroyRealDFTParams( &status, &dftparams1);
    TestStatus (&status, CODES(0), 1);

    LALCDestroyVector( &status, &(fseries.data));
    TestStatus (&status, CODES(0), 1);
  }


  
  LALDestroyTFPlane( &status, &tfp);
  TestStatus (&status, CODES(0), 1);

  LALSDestroyVector (&status, &(tseries.data) );
  TestStatus (&status, CODES(0), 1);





  /*************************************************************************
   *                                                                       * 
   *                                                                       *
   *  Now check to make sure that correct error codes are generated.       *
   *                                                                       *
   *                                                                       * 
   *************************************************************************/


  if (verbose || lalDebugLevel)
  {
    printf ("\n===== Check Errors =====\n");
  }

  /* 
   *
   *  Test functions LALCreateRealDFTParams() and LALDestroyRealDFTParams()
   *
   */
  {
    LALWindowParams winParams;

    if (verbose)
      {
	printf ("\n--- Testing LALCreateRealDFTParams() and LALDestroyRealDFTParams() \n\n");
      }

    winParams.type=Rectangular;
    winParams.length=tseglength;

    LALCreateRealDFTParams( &status, NULL, &winParams, 1); 
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    winParams.length=0;
    LALCreateRealDFTParams( &status, &(transformparams.dftParams), &winParams, 1);  
    TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
    winParams.length=tseglength;
    
    transformparams.dftParams=NULL;
    LALDestroyRealDFTParams( &status, &(transformparams.dftParams)); 
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    /* next few tests require a valid DFTParams */
    
    LALCreateRealDFTParams( &status, &(transformparams.dftParams), &winParams, 1);
    TestStatus (&status, CODES(0), 1);

    LALCreateRealDFTParams( &status, &(transformparams.dftParams), &winParams, 1);
    TestStatus (&status, CODES(TFTRANSFORM_EALLOCP), 1);

    LALDestroyRealDFTParams( &status, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    {
      RealFFTPlan *p;
      p = transformparams.dftParams->plan;
      transformparams.dftParams->plan = NULL;
      LALDestroyRealDFTParams( &status, &(transformparams.dftParams));
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams->plan = p;
    }

    {
      REAL4Vector *p;
      p = transformparams.dftParams->window;
      transformparams.dftParams->window = NULL;
      LALDestroyRealDFTParams( &status, &(transformparams.dftParams));
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams->window = p;
    }

    LALDestroyRealDFTParams( &status, &(transformparams.dftParams));
    TestStatus (&status, CODES(0), 1);
  }


  /* 
   *
   *  Test functions LALCreateComplexDFTParams() and LALDestroyComplexDFTParams()
   *
   */
  {
    LALWindowParams winParams;

    if (verbose)
      {
	printf ("\n--- Testing LALCreateComplexDFTParams() and LALDestroyComplexDFTParams() \n\n");
      }

    winParams.type=Rectangular;
    winParams.length=tseglength;

    LALCreateComplexDFTParams( &status, NULL, &winParams, 1); 
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    winParams.length=0;
    LALCreateComplexDFTParams( &status, &(transformparams1.dftParams), &winParams, 1);  
    TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
    winParams.length=tseglength;
    
    transformparams.dftParams=NULL;
    LALDestroyComplexDFTParams( &status, &(transformparams1.dftParams)); 
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    /* next few tests require a valid DFTParams */
    
    LALCreateComplexDFTParams( &status, &(transformparams1.dftParams), &winParams, 1);
    TestStatus (&status, CODES(0), 1);

    LALCreateComplexDFTParams( &status, &(transformparams1.dftParams), &winParams, 1);
    TestStatus (&status, CODES(TFTRANSFORM_EALLOCP), 1);


    LALDestroyComplexDFTParams( &status, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);


    {
      ComplexFFTPlan *p;
      p = transformparams1.dftParams->plan;
      transformparams1.dftParams->plan = NULL;
      LALDestroyComplexDFTParams( &status, &(transformparams1.dftParams));
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams->plan = p;
    }

    {
      REAL4Vector *p;
      p = transformparams1.dftParams->window;
      transformparams1.dftParams->window = NULL;
      LALDestroyComplexDFTParams( &status, &(transformparams1.dftParams));
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams->window = p;
    }

    LALDestroyComplexDFTParams( &status, &(transformparams1.dftParams));
    TestStatus (&status, CODES(0), 1);

  }




  /* 
   *
   *  Test function LALComputeFrequencySeries() 
   *
   */
  {
    LALWindowParams winParams;
    winParams.type=Rectangular;
    winParams.length=ntot;

    if (verbose)
      {
	printf("\n--- Testing LALComputeFrequencySeries()\n\n");
      }

    LALCCreateVector( &status, &(fseries.data), ntot/2+1);
    TestStatus (&status, CODES(0), 1);
      
    LALSCreateVector (&status, &(tseries.data), ntot);
    TestStatus (&status, CODES(0), 1);

    LALCreateRealDFTParams( &status, &dftparams1, &winParams, 1);
    TestStatus (&status, CODES(0), 1);
    
    LALComputeFrequencySeries ( &status, &fseries, &tseries, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    LALComputeFrequencySeries ( &status, &fseries, NULL, dftparams1);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    LALComputeFrequencySeries ( &status, NULL, &tseries, dftparams1);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    {
      COMPLEX8Vector *p;
      p = fseries.data;
      fseries.data=NULL;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      fseries.data=p;
    }

    {
      REAL4Vector *p;
      p = tseries.data;
      tseries.data=NULL;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tseries.data=p;

      p = dftparams1->window;
      dftparams1->window=NULL;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      dftparams1->window=p;
    }

    {
      RealFFTPlan *p;
      p = dftparams1->plan;
      dftparams1->plan=NULL;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      dftparams1->plan=p;
    }

    {
      INT4 n;
      n = dftparams1->plan->size;
      dftparams1->plan->size=0;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      dftparams1->plan->size=n;
    }

    tseries.data->length--;
    LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    tseries.data->length++;
    
    fseries.data->length--;
    LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    fseries.data->length++;

    dftparams1->window->length--;
    LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    dftparams1->window->length++;

    {
      REAL8 p;
      p = tseries.deltaT;
      tseries.deltaT=0.0;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tseries.deltaT=p;

      p = dftparams1->sumofsquares;
      dftparams1->sumofsquares=0;
      LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      dftparams1->sumofsquares=p;
    }

    LALDestroyVector (&status, &(tseries.data));
    TestStatus (&status, CODES(0), 1);

    LALDestroyRealDFTParams( &status, &dftparams1);
    TestStatus (&status, CODES(0), 1);

    LALCDestroyVector( &status, &(fseries.data));
    TestStatus (&status, CODES(0), 1);
  }



  /* 
   *
   *  Test functions LALCreateTFPlane() and LALDestroyTFPlane()
   *
   */
  {
    if (verbose)
      {
	printf("\n--- Testing LALCreateTFPlane() and LALDestroyTFPlane()\n\n");
      }

    LALCreateTFPlane( &status, &tfp, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    LALCreateTFPlane( &status, NULL, &params);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    LALDestroyTFPlane( &status, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    LALDestroyTFPlane( &status, &tfp);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    {
      INT4 n;
      n = params.timeBins;
      params.timeBins=0;
      LALCreateTFPlane( &status, &tfp, &params);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      params.timeBins=n;

      n = params.freqBins;
      params.freqBins=0;
      LALCreateTFPlane( &status, &tfp, &params);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      params.freqBins=n;
    }

    {
      REAL8 p;
      p = params.deltaT;
      params.deltaT = 0.0;
      LALCreateTFPlane( &status, &tfp, &params);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      params.deltaT=p;
    }

      
  
    /* next few tests require a valid TF plane */
  
    LALCreateTFPlane( &status, &tfp, &params);
    TestStatus (&status, CODES(0), 1);

    LALCreateTFPlane( &status, &tfp, &params);
    TestStatus (&status, CODES(TFTRANSFORM_EALLOCP), 1);
    
    {
      COMPLEX8 *p;
      p = tfp->data;
      tfp->data = NULL;
      LALDestroyTFPlane( &status, &tfp);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tfp->data = p;
    }

    {
      TFPlaneParams *p;
      p = tfp->params;
      tfp->params=NULL;
      LALDestroyTFPlane( &status, &tfp);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tfp->params = p;
    }
    
    LALDestroyTFPlane( &status, &tfp);
    TestStatus (&status, CODES(0), 1);
  }






  /* 
   *
   *  Test function LALTimeSeriesToTFPlane() 
   *
   */
  {
    LALWindowParams winParams;
    winParams.type=Rectangular;
    winParams.length=tseglength;

    if (verbose)
      {
	printf("\n--- Testing LALTimeSeriesToTFPlane()\n\n");
      }

    LALCreateRealDFTParams( &status, &(transformparams.dftParams), &winParams, 1);    TestStatus (&status, CODES(0), 1);

    LALSCreateVector (&status, &(tseries.data), ntot);
    TestStatus (&status, CODES(0), 1);

    for(i=0; i< tseries.data->length; i++)
      {
	tseries.data->data[i] = ff( (REAL4)(i)/ (REAL4)(ntot));
      };

    LALCreateTFPlane( &status, &tfp, &params);
    TestStatus (&status, CODES(0), 1);


    /* Now start checking errors */

    LALTimeSeriesToTFPlane( &status, tfp, &tseries, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
    LALTimeSeriesToTFPlane( &status, tfp, NULL, &transformparams);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
    LALTimeSeriesToTFPlane( &status, NULL, &tseries, &transformparams);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    {
      REAL4Vector *p;
      p = tseries.data;
      tseries.data=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tseries.data=p;

      p = transformparams.dftParams->window;
      transformparams.dftParams->window=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams->window=p;

    }

    {
      REAL4 *p;
      p = tseries.data->data;
      tseries.data->data=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tseries.data->data=p;

      p = transformparams.dftParams->window->data;
      transformparams.dftParams->window->data=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams->window->data=p;
    }

    {
      COMPLEX8 *p;
      p = tfp->data;
      tfp->data=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tfp->data=p;
    }

    {
      RealDFTParams *p;
      p = transformparams.dftParams;
      transformparams.dftParams=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams=p;
    }

    {
      RealFFTPlan *p;
      p = transformparams.dftParams->plan;
      transformparams.dftParams->plan=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams->plan=p;
    }

    {
      void *p;
      p = transformparams.dftParams->plan->plan;
      transformparams.dftParams->plan->plan=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams.dftParams->plan->plan=p;
    }

    {
      TFPlaneParams *p;
      p = tfp->params;
      tfp->params=NULL;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tfp->params=p;
    }

    {
      INT4 p;
      p = tfp->params->timeBins;
      tfp->params->timeBins=0;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tfp->params->timeBins=p;

      p = tfp->params->freqBins;
      tfp->params->freqBins=0;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);

      tfp->params->freqBins=10000000;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
      tfp->params->freqBins=p;

      p = transformparams.startT;
      transformparams.startT = 1000000;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
      transformparams.startT = p;
    }

    {
      REAL4 p;
      p = transformparams.dftParams->sumofsquares;
      transformparams.dftParams->sumofsquares=0;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      transformparams.dftParams->sumofsquares=p;
    }

    {
      REAL8 p;
      p = tseries.deltaT;
      tseries.deltaT=0.0;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tseries.deltaT=p;

      p = tfp->params->deltaT;
      tfp->params->deltaT=0.0;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tfp->params->deltaT = 0.1 * tseries.deltaT;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
      tfp->params->deltaT=p;

      p = tseries.f0;
      tseries.f0 = -1.0;
      LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tseries.f0=p;
    }

    transformparams.dftParams->plan->size--;
    LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    transformparams.dftParams->plan->size++;

    transformparams.dftParams->window->length--;
    LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    transformparams.dftParams->window->length++;

    transformparams.dftParams->plan->sign=-1;
    LALTimeSeriesToTFPlane( &status, tfp, &tseries, &transformparams);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    transformparams.dftParams->plan->sign=1;

    
    /* clean up */

    LALDestroyTFPlane( &status, &tfp);
    TestStatus (&status, CODES(0), 1);

    LALSDestroyVector (&status, &(tseries.data) );
    TestStatus (&status, CODES(0), 1);

    LALDestroyRealDFTParams( &status, &(transformparams.dftParams));
    TestStatus (&status, CODES(0), 1);
  }






  /* 
   *
   *  Test function LALFreqSeriesToTFPlane() 
   *
   */
  {
    LALWindowParams winParams;
    LALWindowParams winParams1; 

    winParams.type=Rectangular;
    winParams.length=ntot;
    winParams1.type=Rectangular;
    winParams1.length=fseglength;

    if (verbose)
      {
	printf("\n--- Testing LALFreqSeriesToTFPlane()\n\n");
      }


    LALSCreateVector (&status, &(tseries.data), ntot);
    TestStatus (&status, CODES(0), 1);

    for(i=0; i< tseries.data->length; i++)
      {
	tseries.data->data[i] = ff( (REAL4)(i)/ (REAL4)(ntot));
      };

    LALCreateTFPlane( &status, &tfp, &params);
    TestStatus (&status, CODES(0), 1);

    LALCCreateVector( &status, &(fseries.data), ntot/2+1);
    TestStatus (&status, CODES(0), 1);
      

    LALCreateRealDFTParams( &status, &dftparams1, &winParams,1);
    TestStatus (&status, CODES(0), 1);
    
    LALComputeFrequencySeries ( &status, &fseries, &tseries, dftparams1);
    TestStatus (&status, CODES(0), 1);

    LALCreateComplexDFTParams( &status, &(transformparams1.dftParams), 
                            &winParams1, -1); 
    TestStatus (&status, CODES(0), 1);

    LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
    TestStatus (&status, CODES(0), 1);



    /* Now start checking errors */

    LALFreqSeriesToTFPlane( &status, tfp, &fseries, NULL);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
    LALFreqSeriesToTFPlane( &status, tfp, NULL, &transformparams1);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
    LALFreqSeriesToTFPlane( &status, NULL, &fseries, &transformparams1);
    TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);

    {
      COMPLEX8Vector *p;
      p = fseries.data;
      fseries.data=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      fseries.data=p;
    }
    
    {
      REAL4Vector *p;
      p = transformparams1.dftParams->window;
      transformparams1.dftParams->window=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams->window=p;
    }

    {
      COMPLEX8 *p;
      p = fseries.data->data;
      fseries.data->data=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      fseries.data->data=p;

      p = tfp->data;
      tfp->data=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tfp->data=p;
    }

    {
      REAL4 *p;
      p = transformparams1.dftParams->window->data;
      transformparams1.dftParams->window->data=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams->window->data=p;
    }

    {
      ComplexDFTParams *p;
      p = transformparams1.dftParams;
      transformparams1.dftParams=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams=p;
    }


    {
      ComplexFFTPlan *p;
      p = transformparams1.dftParams->plan;
      transformparams1.dftParams->plan=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams->plan=p;
    }

    {
      void *p;
      p = transformparams1.dftParams->plan->plan;
      transformparams1.dftParams->plan->plan=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      transformparams1.dftParams->plan->plan=p;
    }

    {
      TFPlaneParams *p;
      p = tfp->params;
      tfp->params=NULL;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_ENULLP), 1);
      tfp->params=p;
    }

    {
      INT4 p;
      p = tfp->params->timeBins;
      tfp->params->timeBins=0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tfp->params->timeBins=p;

      p = tfp->params->freqBins;
      tfp->params->freqBins=0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);

      tfp->params->freqBins=10000000;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
      tfp->params->freqBins=p;
    }

    {
      REAL4 p;
      p = transformparams1.dftParams->sumofsquares;
      transformparams1.dftParams->sumofsquares=0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      transformparams1.dftParams->sumofsquares=p;
    }

    {
      REAL8 p;
      p = fseries.deltaF;
      fseries.deltaF=0.0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      fseries.deltaF=p;

      p = tfp->params->deltaT;
      tfp->params->deltaT=0.0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      tfp->params->deltaT = 20.0 / fseries.deltaF;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
      tfp->params->deltaT=p;

      p = fseries.f0;
      fseries.f0 = -1.0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EPOSARG), 1);
      fseries.f0 = 10000000.0;
      LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
      TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
      fseries.f0=p;
    }

    transformparams1.dftParams->plan->size--;
    LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    transformparams1.dftParams->plan->size++;

    transformparams1.dftParams->window->length--;
    LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    transformparams1.dftParams->window->length++;

    transformparams1.dftParams->plan->sign=1;
    LALFreqSeriesToTFPlane( &status, tfp, &fseries, &transformparams1);
    TestStatus (&status, CODES(TFTRANSFORM_EINCOMP), 1);
    transformparams1.dftParams->plan->sign=-1;




    /* Now clean up */

    LALDestroyComplexDFTParams( &status, &(transformparams1.dftParams));
    TestStatus (&status, CODES(0), 1);
    
    LALDestroyRealDFTParams( &status, &dftparams1);
    TestStatus (&status, CODES(0), 1);

    LALCDestroyVector( &status, &(fseries.data));
    TestStatus (&status, CODES(0), 1);

    LALDestroyTFPlane( &status, &tfp);
    TestStatus (&status, CODES(0), 1);

    LALSDestroyVector (&status, &(tseries.data) );
    TestStatus (&status, CODES(0), 1);

  }


  LALCheckMemoryLeaks ();

  if(verbose)  printf("PASS: all tests\n");
  
  return 0;
}


static REAL4 ff(REAL4 w)
{
  /* simple waveform function used for testing */
  REAL4 t,s,f;
  REAL4 sigma = 0.4;
  t = 2.0* (w - 0.5);
  f = 70.0 + 30.0*t;
  s = sin(f*t)*exp(-t*t/( 2.0 * sigma * sigma));
  return(s);
}





/*
 * TestStatus ()
 *
 * Routine to check that the status code status->statusCode agrees with one of
 * the codes specified in the space-delimited string ignored; if not,
 * exit to the system with code exitcode.
 *
 */
static void
TestStatus (LALStatus *status, const char *ignored, int exitcode)
{
  char  str[64];
  char *tok;

  if (verbose)
  {
    /*REPORTSTATUS (status);*/
  }

  if (strncpy (str, ignored, sizeof (str)))
  {
    if ((tok = strtok (str, " ")))
    {
      do
      {
        if (status->statusCode == atoi (tok))
        {
          return;
        }
      }
      while ((tok = strtok (NULL, " ")));
    }
    else
    {
      if (status->statusCode == atoi (tok))
      {
        return;
      }
    }
  }

  fprintf (stderr, "\nExiting to system with code %d\n", exitcode);
  exit (exitcode);
}


/*
 * Usage ()
 *
 * Prints a usage message for program program and exits with code exitcode.
 *
 */
static void
Usage (const char *program, int exitcode)
{
  fprintf (stderr, "Usage: %s [options]\n", program);
  fprintf (stderr, "Options:\n");
  fprintf (stderr, "  -h         print this message\n");
  fprintf (stderr, "  -q         quiet: run silently\n");
  fprintf (stderr, "  -v         verbose: print extra information\n");
  fprintf (stderr, "  -d level   set lalDebugLevel to level\n");
  exit (exitcode);
}


/*
 * ParseOptions ()
 *
 * Parses the argc - 1 option strings in argv[].
 *
 */
static void
ParseOptions (int argc, char *argv[])
{
  while (1)
  {
    int c = -1;

    c = getopt (argc, argv, "hqvd:");
    if (c == -1)
    {
      break;
    }

    switch (c)
    {
      case 'd': /* set debug level */
        lalDebugLevel = atoi (optarg);
        break;

      case 'v': /* verbose */
        ++verbose;
        break;

      case 'q': /* quiet: run silently (ignore error messages) */
        freopen ("/dev/null", "w", stderr);
        freopen ("/dev/null", "w", stdout);
        break;

      case 'h':
        Usage (argv[0], 0);
        break;

      default:
        Usage (argv[0], 1);
    }

  }

  if (optind < argc)
  {
    Usage (argv[0], 1);
  }

  return;
}






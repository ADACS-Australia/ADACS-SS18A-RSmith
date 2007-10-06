 /*
  * Copyright (C) 2004, 2005 Cristina V. Torres
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

/*----------------------------------------------------------------------- 
 * 
 * File Name: TSData.c
 *
 * Author: Torres C  (Univ of TX at Brownsville)
 * 
 * Revision: $Id$
 * 
 *-----------------------------------------------------------------------
 */
 
#include <lal/TSData.h>
#include <lal/TSSearch.h>
#include <lal/FrameCache.h>
#include <lal/LALStdlib.h>

NRCSID (TSDATAC,"$Id$");

/*
 *Extra diagnostic code
 */
/* Non Compliant code taken from EPSearch.c */
static void print_real4fseries(const REAL4FrequencySeries *fseries,const char *file)
{
#if 0
  /* FIXME: why can't the linker find this function? */
  LALSPrintFrequencySeries(fseries, file);
#else
  FILE *fp = fopen(file, "w");
  size_t i;

  if(fp) {
    for(i = 0; i < fseries->data->length; i++)
      fprintf(fp, "%f\t%g\n", i * fseries->deltaF, fseries->data->data[i]);
    fclose(fp);
  }
#endif
}
static void print_complex8fseries(const COMPLEX8FrequencySeries *fseries, const char *file)
{
#if 0
  /* FIXME: why can't the linker find this function? */
  LALCPrintFrequencySeries(fseries, file);
#else
  FILE *fp = fopen(file, "w");
  size_t i;

  if(fp) {
    for(i = 0; i < fseries->data->length; i++)
      fprintf(fp, "%f\t%g\n", i * fseries->deltaF, sqrt(fseries->data->data[i].re * fseries->data->data[i].re + fseries->data->data[i].im * fseries->data->data[i].im));
    fclose(fp);
  }
#endif
}
static void print_real4tseries(const REAL4TimeSeries *fseries, const char *file)
{
#if 0
  /* FIXME: why can't the linker find this function? */
  LALSPrintTimeSeries(fseries, file);
#else
  FILE *fp = fopen(file, "w");
  size_t i;

  if(fp) {
    for(i = 0; i < fseries->data->length; i++)
      fprintf(fp, "%f\t%g\n", i * fseries->deltaT, fseries->data->data[i]);
    fclose(fp);
  }
#endif
}
/*
 * End diagnostic code
 */

void
LALCreateTSDataSegmentVector (
			      LALStatus                    *status,
			      TSSegmentVector             **vector,
			      TSCreateParams               *params
			      )
{
  INT4                           i;
  UINT4                  segmentLength=0;
  const LIGOTimeGPS      gps_zero = LIGOTIMEGPSZERO;

  INITSTATUS (status, "LALCreateTSSegmentVector", TSDATAC);
  ATTATCHSTATUSPTR (status);

  ASSERT (!*vector, status, TSDATA_ENNUL, TSDATA_MSGENNUL);
  ASSERT (params, status, TSDATA_ENULL, TSDATA_MSGENULL);
  ASSERT (params->numberDataSegments > 0, 
	  status, TSDATA_ESEGZ, TSDATA_MSGESEGZ);
  ASSERT (params->dataSegmentPoints > 0,
	  status, TSDATA_ENUMZ, TSDATA_MSGENUMZ);
  
  *vector = (TSSegmentVector *) LALMalloc(sizeof(TSSegmentVector));


  (*vector)->length = params->numberDataSegments;
  (*vector)->dataSeg  = (REAL4TimeSeries **) 
    LALMalloc(params->numberDataSegments*sizeof(REAL4TimeSeries*));
  if ( !((*vector)->dataSeg) )
    {
      LALFree(*vector);
      ABORT( status, TSDATA_EALOC, TSDATA_MSGEALOC );
    }
  /*
   * Intialize structure to empty state
   */
  segmentLength=params->dataSegmentPoints;
  for (i = 0; i < (INT4)((*vector)->length); i++)
    {
      LALCreateREAL4TimeSeries(status->statusPtr,
			       &((*vector)->dataSeg[i]),
			       "Uninitialized",
			       gps_zero,
			       0,
			       1,
			       lalDimensionlessUnit,
			       segmentLength);
      CHECKSTATUSPTR (status);
    }
  DETATCHSTATUSPTR (status);
  RETURN (status);
}
/*END LALCreateTSDataSegmentVector*/


void
LALDestroyTSDataSegmentVector (
			       LALStatus                  *status,
			       TSSegmentVector            *vector
			       )
{
  INT4                    i;

  INITSTATUS (status, "LALDestroyTSDataSegmentVector", TSDATAC);
  ATTATCHSTATUSPTR (status);
  ASSERT (vector, 
	  status, 
	  TSDATA_ENULL, 
	  TSDATA_MSGENULL);
  for (i = 0; i < (INT4)(vector->length) ; i++)
    {
      LALDestroyREAL4TimeSeries(status->statusPtr,(vector->dataSeg[i]));
      CHECKSTATUSPTR (status);
    }
  if (vector->dataSeg)
    LALFree(vector->dataSeg);
  if (vector)
    LALFree(vector);
  DETATCHSTATUSPTR (status);
  RETURN (status);
}
/*END LALDestroyTSDataSegmentVector */


void
LALTrackSearchConnectSigma(
			   LALStatus                     *status,
			   TrackSearchOut                *curveinfo,
			   TimeFreqRep                    map,
			   TrackSearchParams              params
			   )
{
  INT4 i,j;
  Curve *curveA;
  Curve *curveB;

  INITSTATUS(status,"LALTrackSearchConnectSigma",TSDATAC);
  ATTATCHSTATUSPTR (status);


  /* Outer Loop trys to establish joints */
  for (i=0;i<curveinfo->numberOfCurves;i++)
    {
      /*Ignore curves labeled with marker 'D' */
      if (curveinfo->curves[i].trash != 'D' )
	{
	  for (j=0;j<curveinfo->numberOfCurves;j++)
	    {
	      /*Ignore 'D' label curves*/
	      if (curveinfo->curves[j].trash != 'D' )
		{
		  /*Inside Loop test this track to all other*/
		  if (i!=j)
		    {
		      curveA=&(curveinfo->curves[i]);
		      curveB=&(curveinfo->curves[j]);
		      if ((abs(curveA->row[curveA->n]-curveB->row[0]) <
			   params.sigma+1)
			  &&
			  (abs(curveA->col[curveA->n]-curveB->row[0]) < params.sigma+1))
			{
			  /* Following function is iffy */
			  /* Ok they touch join them */
			  connect2Segments(map,curveA,curveB);
			  /* Reset interior loop to try and check new longer track*/
			  j=0;
			}
		    }
		}
	    }
	}
    }
  /* Done joing useful curve candidates */
  /* Cleanup linked list */

  DETATCHSTATUSPTR(status);
  RETURN(status);
}


void
LALTrackSearchApplyThreshold(
			     LALStatus         *status,
			     TrackSearchOut    *curveinfo,
			     TrackSearchOut    *dataProduct,
			     TSSearchParams     params
			     )
{
  INT4 UsefulCurves;
  INT4 i;
  INT4 j;
  INT4 cicn;

  INITSTATUS(status,"LALTrackSearchApplyThreshold",TSDATAC);
  ATTATCHSTATUSPTR (status);
  ASSERT(dataProduct->curves==NULL,status,TS_NON_NULL_POINTER,TS_MSGNON_NULL_POINTER);
  /*Trackstore struct field not copied! */
  UsefulCurves=0; /*Array numbering starts with zero */
  dataProduct->numberOfCurves=0;/*Default value*/
  if (curveinfo->numberOfCurves > 0)
    {
      /* Count up number of useful curves to keep */
      for(i=0;i<curveinfo->numberOfCurves;i++)
	{
	  if ((curveinfo->curves[i].n >= params.MinLength) && 
	      (curveinfo->curves[i].totalPower >= params.MinPower))
	    {
	      /*UsefulCurve Var is not count but curve index so add 1*/
	      /* Expand the structure to take in another curve */
	      UsefulCurves=dataProduct->numberOfCurves;
	      dataProduct->curves = (Curve*)LALRealloc(dataProduct->curves,(sizeof(Curve) * (UsefulCurves+1)));

	      /*Length of curve to copy*/
	      cicn=curveinfo->curves[i].n;
	      dataProduct->curves[UsefulCurves].row=(INT4*)LALMalloc(sizeof(INT4) * cicn);
	      dataProduct->curves[UsefulCurves].col=(INT4*)LALMalloc(sizeof(INT4) * cicn);
	      dataProduct->curves[UsefulCurves].depth=(REAL4*)LALMalloc(sizeof(REAL4) * cicn);
	      dataProduct->curves[UsefulCurves].fBinHz=(REAL4*)LALMalloc(sizeof(REAL4) * cicn);
	      dataProduct->curves[UsefulCurves].gpsStamp=(LIGOTimeGPS*)LALMalloc(sizeof(LIGOTimeGPS) * cicn);
	      dataProduct->curves[UsefulCurves].n = cicn;
	      dataProduct->curves[UsefulCurves].totalPower = curveinfo->curves[i].totalPower;
	      /* Copy the data over */
	      for(j=0;j<dataProduct->curves[UsefulCurves].n;j++)
		{
		  dataProduct->curves[UsefulCurves].row[j]=curveinfo->curves[i].row[j];
		  dataProduct->curves[UsefulCurves].col[j]=curveinfo->curves[i].col[j];
		  dataProduct->curves[UsefulCurves].depth[j]=
		    curveinfo->curves[i].depth[j];
		  dataProduct->curves[UsefulCurves].fBinHz[j]=
		    curveinfo->curves[i].fBinHz[j];
		  dataProduct->curves[UsefulCurves].gpsStamp[j].gpsSeconds=
		    curveinfo->curves[i].gpsStamp[j].gpsSeconds;
		  dataProduct->curves[UsefulCurves].gpsStamp[j].gpsNanoSeconds=
		    curveinfo->curves[i].gpsStamp[j].gpsNanoSeconds;
		}
	      /*Increase index recall count is +1 of index*/
	      dataProduct->numberOfCurves++;
	      /* Move on to check next curve */
	    }
	}
      /*If nothing passes the threshold deallocate the pointer*/
      if ((dataProduct->numberOfCurves == 0) && 
	  (dataProduct->curves != NULL))
	{
	  LALFree(dataProduct->curves);
	  dataProduct->curves=NULL;
	}
    }
  /* Done moving useful curve candidates */
  DETATCHSTATUSPTR(status);
  RETURN(status);
} /* End LALTrackSearchApplyThreshold */


/* Routine to whiten the data stream */
/* We will overwhiten by calling 2 times in order */
/* Currently unused and not thoroughly tested */
void
LALTrackSearchWhitenREAL4TimeSeries(
				    LALStatus              *status,
				    REAL4TimeSeries        *signal,
				    REAL4FrequencySeries   *signalPSD,
				    TSWhitenParams          params
				    )
{
  UINT4                      i=0;
  RealFFTPlan               *forwardPlan=NULL;
  RealFFTPlan               *reversePlan=NULL;
  COMPLEX8FrequencySeries   *signalFFT=NULL;
  const LIGOTimeGPS              gps_zero = LIGOTIMEGPSZERO;
  UINT4                      planLength=0;
  REAL8                     factor=0;
  LALUnit                   tmpUnit1=lalDimensionlessUnit;
  LALUnitPair               tmpUnitPair;
  RAT4                      exponent;
  INITSTATUS(status,"LALTrackSearchWhitenREAL4TimeSeries",TSDATAC);
  ATTATCHSTATUSPTR (status);

  /*Setup FFT Plans*/
  planLength=signal->data->length;
  LALCreateForwardREAL4FFTPlan(status->statusPtr,
			       &forwardPlan,
			       planLength,
			       0);
  CHECKSTATUSPTR (status);
  LALCreateReverseREAL4FFTPlan(status->statusPtr,
			       &reversePlan,
			       planLength,
			       0);
  CHECKSTATUSPTR (status);

  /* Allocate space for FFT */
  LALCreateCOMPLEX8FrequencySeries(status->statusPtr,
				   &signalFFT,
				   "tmpSegPSD",
				   gps_zero,
				   0,
				   1/(signal->deltaT*signal->data->length),
				   lalDimensionlessUnit,
				   planLength/2+1);
  /* FFT the time series */
  LALForwardREAL4FFT(status->statusPtr,
		     signalFFT->data,
		     signal->data,
		     forwardPlan);
  CHECKSTATUSPTR (status);
   
  /*
   * Diagnostic code
   * Temporary
   */
  print_complex8fseries(signalFFT,"dataFFTComplex.txt");
  /*
   * Perform whitening
   * Look at Tech Doc T010095-00  Sec3 
   */
  for (i=0;i<signalFFT->data->length;i++)
    {
      if ( signalPSD->data->data[i] == 0.0 )
	factor=0;
      else
	/*
	 * This whitening filter pulled from EPsearch code
	 */
	factor=2*sqrt(signalFFT->deltaF/signalPSD->data->data[i]);

      signalFFT->data->data[i].re = signalFFT->data->data[i].re * factor;
      signalFFT->data->data[i].im = signalFFT->data->data[i].im * factor;
    }
  /* 
   * Manipulate the LALUnits structure to reflect above operation
   */
  exponent.numerator=-1;
  exponent.denominatorMinusOne=1;/*2*/
  LALUnitRaise(status->statusPtr,
	       &tmpUnit1,
	       &(signalPSD->sampleUnits),
	       &exponent);
  CHECKSTATUSPTR (status);
  tmpUnitPair.unitOne=&tmpUnit1;
  tmpUnitPair.unitTwo=&(signalFFT->sampleUnits);
  LALUnitMultiply(status->statusPtr,
		  &(signalFFT->sampleUnits),
		  &tmpUnitPair);
  CHECKSTATUSPTR (status);
  /* 
   * Diagnostic code
   */
  print_complex8fseries(signalFFT,"dataFFTComplexPOST.txt");
   
  /* 
   * Transform back to time domain 
   */
  LALReverseREAL4FFT(status->statusPtr,
		     signal->data,
		     signalFFT->data,
		     reversePlan);
  CHECKSTATUSPTR (status);
  /* 
   * The 1/n factor need to be applied
   * See lsd-5 p 259 10.1 
   */
  for (i=0;i<signal->data->length;i++)
    signal->data->data[i]= signal->data->data[i]/signal->data->length;

  /* 
   * Diagnostic code
   */
  print_real4tseries(signal,"dataSegi.txt");

  /* 
   *Release the temporary memory 
   */
  if (signalFFT)
    {
      LALDestroyCOMPLEX8FrequencySeries(status->statusPtr,signalFFT);
      CHECKSTATUSPTR (status);
    }
  if (forwardPlan)
    {
      LALDestroyREAL4FFTPlan(status->statusPtr,&forwardPlan);
      CHECKSTATUSPTR (status);
    }
  if (reversePlan)
    {
      LALDestroyREAL4FFTPlan(status->statusPtr,&reversePlan);
      CHECKSTATUSPTR (status);
    }
  DETATCHSTATUSPTR(status);
  RETURN(status);
}

/* End whiten routine */


/* Begin Fourier Domain whitening routine */
void
LALTrackSearchWhitenCOMPLEX8FrequencySeries(
					    LALStatus                *status,
					    COMPLEX8FrequencySeries  *fSeries,
					    REAL4FrequencySeries     *PSD,
					    UINT4                     level
					    )
{
  UINT4         i=0;
  REAL8         factor=0;
  LALUnit       tmpUnit1=lalDimensionlessUnit;
  LALUnitPair   tmpUnitPair;
  RAT4          exponent;

  INITSTATUS(status,"LALTrackSearchWhitenCOMPLEX8FrequencySeries",TSDATAC);
  ATTATCHSTATUSPTR (status);
  /*
   * Error checking 
   */
  ASSERT(level > 0, status,TSDATA_EINVA,TSDATA_MSGEINVA);

  for (i=0;i<fSeries->data->length;i++)
    {
      if ( PSD->data->data[i] == 0.0 )
	factor=0;
      else
	/*
	 * This whitening filter pulled from EPsearch code
	 */
	factor=2*sqrt(fSeries->deltaF/PSD->data->data[i]);

      fSeries->data->data[i].re = fSeries->data->data[i].re * factor;
      fSeries->data->data[i].im = fSeries->data->data[i].im * factor;
    }
 /*
  * LALUnits manipulation
  */
   exponent.numerator=-1;
  exponent.denominatorMinusOne=1;/*2*/
  LALUnitRaise(status->statusPtr,
	       &tmpUnit1,
	       &(PSD->sampleUnits),
	       &exponent);
  CHECKSTATUSPTR (status);
  tmpUnitPair.unitOne=&tmpUnit1;
  tmpUnitPair.unitTwo=&(fSeries->sampleUnits);
  LALUnitMultiply(status->statusPtr,
		  &(fSeries->sampleUnits),
		  &tmpUnitPair);
  CHECKSTATUSPTR (status);

  DETATCHSTATUSPTR(status);
  RETURN(status);
}
/* End Fourier domain whitening */

/* Begin calibration routine */
void
LALTrackSearchCalibrateREAL4TimeSeries(LALStatus               *status,
				       REAL4TimeSeries         *signal,
				       COMPLEX8FrequencySeries *response)
{
  UINT4                      i=0;
  RealFFTPlan               *forwardPlan=NULL;
  RealFFTPlan               *reversePlan=NULL;
  COMPLEX8FrequencySeries   *signalFFT=NULL;
  UINT4                      planLength=0;

  INITSTATUS(status,"LALTrackSearchCalibrateREAL4TimeSeries",TSDATAC);
  ATTATCHSTATUSPTR (status);
  /* Need consistency checks for inputs so that the df of each match */
  /*
   * Setup FFT plans for FFTing the data segment
   */
  planLength=signal->data->length;
  LALCreateForwardREAL4FFTPlan(status->statusPtr,
			       &forwardPlan,
			       planLength,
			       0);
  CHECKSTATUSPTR (status);
  LALCreateReverseREAL4FFTPlan(status->statusPtr,
			       &reversePlan,
			       planLength,
			       0);
  CHECKSTATUSPTR (status);
  /*
   * Allocate RAM for temp Freq series
   */
  LALCreateCOMPLEX8FrequencySeries(status->statusPtr,
				   &signalFFT,
				   "tmpSignalFFT",
				   signal->epoch,
				   0,
				   1/signal->deltaT,
				   lalDimensionlessUnit,
				   planLength/2+1);
  /*
   * FFT the data segment for calibration 
   */
  LALForwardREAL4FFT(status->statusPtr,
		     signalFFT->data,
		     signal->data,
		     forwardPlan);
  CHECKSTATUSPTR (status);
  /*
   * Perform the frequency basis calibration as defined in 
   * LSD Conventions Eq 23.1 p 601
   */
  for (i=0;i<signal->data->length;i++)
    {
      signalFFT->data->data[i].re=
	response->data->data[i].re*signalFFT->data->data[i].re;
      signalFFT->data->data[i].im=
	response->data->data[i].im*signalFFT->data->data[i].im;
    }
  /*
   * Bring this back to the time domain
   * this is the calibrated data set
   */
  LALReverseREAL4FFT(status->statusPtr,
		     signal->data,
		     signalFFT->data,
		     reversePlan);
  CHECKSTATUSPTR (status);
  /* 
   * The 1/n factor need to be applied
   * See lsd-5 p 259 10.1 
   */
  for (i=0;i<signal->data->length;i++)
    signal->data->data[i]= signal->data->data[i]/signal->data->length;

  /*
   * Destroy signalFFT Temp variable
   */
  if (signalFFT)
    {
      LALDestroyCOMPLEX8FrequencySeries(status->statusPtr,signalFFT);
      CHECKSTATUSPTR (status);
    }
  /*
   * Destroy the FFT plans
   */
  if (forwardPlan)
    {
      LALDestroyREAL4FFTPlan(status->statusPtr,&forwardPlan);
      CHECKSTATUSPTR (status);
    }
  if (reversePlan)
    {
      LALDestroyREAL4FFTPlan(status->statusPtr,&reversePlan);
      CHECKSTATUSPTR (status);
    }
  DETATCHSTATUSPTR(status);
  RETURN(status);
    
}
/* End calibration routine */


/* Begin Fourier Domain calibration routine */
void
LALTrackSearchCalibrateCOMPLEX8FrequencySeries(
					       LALStatus                 *status,
					       COMPLEX8FrequencySeries   *fSeries,
					       COMPLEX8FrequencySeries   *response
					       )
{
  UINT4          i=0;
  LALUnitPair    tmpUnitPair;
  REAL4          a=0;
  REAL4          b=0;
  REAL4          c=0;
  REAL4          d=0;

  INITSTATUS(status,"LALTrackSearchCalibrateCOMPLEX8FrequencySeries",TSDATAC);
  ATTATCHSTATUSPTR (status);
  /*
   * Error checking
   */
  ASSERT(fSeries != NULL,status,TSDATA_ENULL, TSDATA_MSGENULL);
  ASSERT(response != NULL,status,TSDATA_ENULL, TSDATA_MSGENULL);
  /*
   * Calibration is done via applying expression
   * 23.1 Conventions
   * s(f) = R(f;t)v(f)
   * Unit field is adjust appropriately and we return a 
   * calibrated data solution
   */
  for(i=0;i<fSeries->data->length;i++)
    {
      a=fSeries->data->data[i].re;
      b=fSeries->data->data[i].im;
      c=response->data->data[i].re;
      d=response->data->data[i].im;
      /*(a+bi)*(c+di)*/
      fSeries->data->data[i].re=(a*c - b*d);
      fSeries->data->data[i].im=(a*d + b*c);
    }
  /* 
   * Unit manipulation
   */
  tmpUnitPair.unitOne=&(fSeries->sampleUnits);
  tmpUnitPair.unitTwo=&(response->sampleUnits);
  LALUnitMultiply(status->statusPtr,
		  &(fSeries->sampleUnits),
		  &tmpUnitPair);
  CHECKSTATUSPTR (status);
  DETATCHSTATUSPTR(status);
  RETURN(status);
}
/* End Fourier Domain calibration routine */

/*
 * This is the function to break up long stretch of input data
 * into the requested chunks accounting for overlap
 */
void
LALTrackSearchDataSegmenter(
			    LALStatus           *status,
			    REAL4TimeSeries     *TSSearchData,
			    TSSegmentVector     *PreparedData,
			    TSSearchParams       params)
{
  UINT4         k=0;
  UINT4         l=0;
  UINT4         j=0;
  REAL8         kTime;
  LIGOTimeGPS   timeInterval;

  INITSTATUS (status, "LALTrackSearchDataSegmenter", TSDATAC);

  /*
   * Error checking
   */
  ASSERT(PreparedData != NULL,status,TSDATA_ENULL,TSDATA_MSGENULL);
  ASSERT(TSSearchData != NULL,status,TSDATA_ENULL,TSDATA_MSGENULL);
  ASSERT((params.SegLengthPoints == PreparedData->dataSeg[0]->data->length),
	 status,
	 TSDATA_ENUMZ,
	 TSDATA_MSGENUMZ);
  ASSERT(PreparedData->length == params.NumSeg,
	 status,
	 TSDATA_ESEGZ,
	 TSDATA_MSGESEGZ);
  /*
   * We want to fill up our TSDataVector structure accounding for
   * desired number of segments and overlaps
   */
  ATTATCHSTATUSPTR (status);
  for (l=0;l<PreparedData->length;l++)
    {
      /*Determlne Segment Epoch*/
      kTime=TSSearchData->deltaT*k;
      LALFloatToGPS(status->statusPtr,&(timeInterval),&kTime);
      CHECKSTATUSPTR (status);
      for (j=0;j<PreparedData->dataSeg[l]->data->length;j++)
	{
	  PreparedData->dataSeg[l]->data->data[j]=TSSearchData->data->data[k];
	  /*	  printf("%d\n",j);*/
	  k++;
	};
      /*Ajust for segment overlap*/
      k = k - params.overlapFlag;
      PreparedData->dataSeg[l]->data->length = params.SegLengthPoints;
      PreparedData->dataSeg[l]->deltaT=TSSearchData->deltaT;
      PreparedData->dataSeg[l]->sampleUnits=TSSearchData->sampleUnits;
      PreparedData->dataSeg[l]->epoch.gpsSeconds=
	TSSearchData->epoch.gpsSeconds+timeInterval.gpsSeconds;
      PreparedData->dataSeg[l]->epoch.gpsNanoSeconds=
	TSSearchData->epoch.gpsNanoSeconds+timeInterval.gpsNanoSeconds;
      sprintf(PreparedData->dataSeg[l]->name,"%s","Initialized");
    };

  /*
   * End segment setup
   */
 DETATCHSTATUSPTR (status);
 RETURN (status);
}
/* End the data segmenter */

void
LALSVectorPolynomialInterpolation(
				  LALStatus         *status,
				  REAL4Sequence     *newDomain,
				  REAL4Sequence     *newRange,
				  REAL4Sequence     *domain,
				  REAL4Sequence     *range
				  )
{
  SInterpolatePar               interpolateParams;
  SInterpolateOut               interpolateResult;
  REAL4Vector                   *xfrag=NULL;
  REAL4Vector                   *yfrag=NULL;
  UINT4                         i=0;
  INT4                          j=0;
  INT4                          k=0;
  INT4                         bottomElement,topElement,currentElement;
  BOOLEAN                       cut;

  INITSTATUS(status,"LALSVectorPolynomialInterpolation",TSDATAC);
  ATTATCHSTATUSPTR (status);
  ASSERT(range->length > 5,status,TSDATA_EINTP,TSDATA_MSGEINTP);
  LALCreateVector(status->statusPtr,&xfrag,5);
  CHECKSTATUSPTR(status);
  LALCreateVector(status->statusPtr,&yfrag,5);
  CHECKSTATUSPTR(status);
  for (i=0;i<newRange->length;i++)
    {
      /*
       *Setup params
       */
      
      /* 
       * Domain is ordered so use bisection technique to extract domain
       * point of interest.
       * Since we are interpolating using 5 points from the orignal range
       */
      currentElement=((UINT4) (domain->length/2));
      topElement=domain->length;
      bottomElement=0;
      cut=1;
      while (cut)
	{  
	  if (newDomain->data[i] >= domain->data[currentElement])
	    {
	      bottomElement=currentElement;
	      currentElement=((UINT4) (topElement-bottomElement)/2)+bottomElement;
	    }
	  else
	    {
	      topElement=currentElement;
	      currentElement=topElement-((UINT4) (topElement-bottomElement)/2);
	    }
	  if ((topElement-bottomElement) == 1) cut=0;
	}
      if (currentElement < 2) currentElement=2;
      if ((domain->length - currentElement) < 2)
	currentElement=domain->length - 2;
      for (j=0,k=currentElement-2;k<currentElement+3;k++,j++)
	{
	  xfrag->data[j]=domain->data[k];
	  yfrag->data[j]=range->data[k];
	}
      interpolateParams.n=5;
      interpolateParams.x=xfrag->data;
      interpolateParams.y=yfrag->data;
      LALSPolynomialInterpolation(status->statusPtr,
				  &interpolateResult,
				  newDomain->data[i],
				  &interpolateParams);
      CHECKSTATUSPTR(status);
      newRange->data[i]=interpolateResult.y;
    }
  if (xfrag)
    {
      LALDestroyVector(status->statusPtr,&xfrag);
      CHECKSTATUSPTR(status);
    }
  if (yfrag)
    {
      LALDestroyVector(status->statusPtr,&yfrag);
      CHECKSTATUSPTR(status);     
    }
  DETATCHSTATUSPTR(status);
  RETURN(status);
}/* End interpolate vector like Matlab interp1 */

/* 
 * Noncompliant code 
 * Local SUB-routines
 * Non complient test code 
 */

/* 
 * As written this routine most likely wont work 
 * I think there is variable scope problems
 */
void
connect2Segments(
		 TimeFreqRep    map,
		 Curve          *curveA,
		 Curve          *curveB
		 )
{
  INT4         i,j,k,m;
  INT4         deltaY;
  INT4         deltaX;
  INT4         newLength;
  INT4         intermediateSegment;
  Curve        tempCurve;
  /* Determine DeltaY */
  /* Determine DeltaX */
  /* Connect via greater of two */
  
  /* Determine points between head-tail to include */
  /* Calculate straight line segement between two points */

  deltaY=curveA->row[curveA->n]-curveB->row[0];
  deltaX=curveA->row[curveA->n]-curveB->row[0];
  newLength=0;
  intermediateSegment=(int)floor(sqrt(deltaY*deltaY+deltaX*deltaX));
  
  newLength=curveA->n+curveB->n+intermediateSegment;
  /* Allocate concatenated curve */
  tempCurve.row=LALMalloc(sizeof(INT4) * newLength);
  tempCurve.col=LALMalloc(sizeof(INT4) * newLength);
  tempCurve.fBinHz=LALMalloc(sizeof(REAL4) * newLength);
  tempCurve.gpsStamp=LALMalloc(sizeof(LIGOTimeGPS) * newLength);
  tempCurve.depth=LALMalloc(sizeof(REAL4) * newLength);

  /* Copy in segments information use map also */
  i=0;
  /* Copy in curveA */
  for (j=0;j<curveA->n;j++)
    {
      tempCurve.row[j]=curveA->row[j];
      tempCurve.col[j]=curveA->col[j];
      tempCurve.fBinHz[j]=curveA->fBinHz[j];
      tempCurve.gpsStamp[j].gpsSeconds=curveA->gpsStamp[j].gpsSeconds;
      tempCurve.gpsStamp[j].gpsNanoSeconds=curveA->gpsStamp[j].gpsNanoSeconds;
      tempCurve.depth[j]=curveA->depth[j];
      i++;
    };
  /* Copy intermediate curve information */
  /* X direction loop */
  m=0;
  for (j=curveA->n;j<(curveA->n+deltaX);j++)
    {
      for (k=0;k<deltaY;k++)
	{
	  tempCurve.row[curveA->n+m]=curveA->row[curveA->n]+j*k;
	  tempCurve.col[curveA->n+m]=curveA->col[curveA->n]+k;
	  tempCurve.fBinHz[curveA->n+m]=0;
	  tempCurve.gpsStamp[curveA->n+m].gpsSeconds=0;
	  tempCurve.gpsStamp[curveA->n+m].gpsNanoSeconds=0;
	  tempCurve.depth[curveA->n+m]=map.map[tempCurve.col[curveA->n+m]][tempCurve.row[curveA->n+m]];
	  if ( m >= intermediateSegment)
	    {
	      k=deltaY+1;
	      j=deltaX+1;
	    }
	}
    }
  /* Copy End segment information */
  for (k=curveA->n+deltaX;k<curveA->n+curveB->n;k++)
    {
      tempCurve.row[k]=curveB->row[k];
      tempCurve.col[k]=curveB->col[k];
      tempCurve.fBinHz[k]=curveB->fBinHz[k];
      tempCurve.gpsStamp[k].gpsSeconds=curveB->gpsStamp[k].gpsSeconds;
      tempCurve.gpsStamp[k].gpsNanoSeconds=curveB->gpsStamp[k].gpsNanoSeconds;
      tempCurve.depth[k]=curveB->depth[k];
      k++;
    };
  /*Set total sum of power field */ 
  for (i=0;i<tempCurve.n;i++)
    {
      tempCurve.totalPower=tempCurve.totalPower+tempCurve.depth[i];
    };
  /* Set Used Marker on second curve when returned to calling function
   */
  curveB->trash='D';
  /* Set 'K'-keep Marker on tempCurve so we know is a joined track */
  tempCurve.trash='K';
  /* Set curveA to tempCurve free original curveA struct first!*/
  /* Freeing curveA */
  LALFree(curveA->row);
  LALFree(curveA->col);
  LALFree(curveA->gpsStamp);
  LALFree(curveA->fBinHz);
  LALFree(curveA->depth);
  LALFree(curveA);
  /*Set pointer to curveA to show joined product */
  curveA=&tempCurve;
  /*Deallocate curveB memory also*/
  LALFree(curveB->row);
  LALFree(curveB->col);
  LALFree(curveB->gpsStamp);
  LALFree(curveB->fBinHz);
  LALFree(curveB->depth);
  curveB->n=0;
  curveB->totalPower=0;
  return;
}

/*
 * Meant as companion function to connect track function
 */
void cleanLinkedList(
		     TrackSearchOut      *inList,
		     TrackSearchOut      *outList
		     )
{/*Will be used to clean up linked list structure and organize it */
  INT4 UsefulCurves;
  INT4 i;
  INT4 j;
  INT4 noc2;
  
  UsefulCurves=0; /*Array numbering starts with zero */
  outList->curves = LALMalloc(sizeof(outList->curves));
  /* Count up number of useful curves to keep */
  for(i=0;i<inList->numberOfCurves;i++)
    {
      if ((inList->curves[i].trash = 'K'))
	{
	  UsefulCurves++;
	  /* Expand the structure to take in another curve */
	  outList->curves = LALRealloc(outList->curves,
				       (sizeof(Curve) * (UsefulCurves)));
	  outList->numberOfCurves = UsefulCurves;
	  /* Numbering starts at zero */
	  noc2 = outList->numberOfCurves-1;
	  outList->curves[noc2].row=LALMalloc(sizeof(INT4) * 
					      inList->curves[i].n);
	  outList->curves[noc2].col=LALMalloc(sizeof(INT4) * 
					      inList->curves[i].n);
	  outList->curves[noc2].depth=LALMalloc(sizeof(REAL4) * 
						inList->curves[i].n);
	  outList->curves[noc2].n = inList->curves[i].n;
	  outList->curves[noc2].totalPower = inList->curves[i].totalPower;
	  /* Copy the data over */
	  for(j=0;j<outList->curves[noc2].n;j++)
	    {
	      outList->curves[noc2].row[j]=inList->curves[i].row[j];
	      outList->curves[noc2].col[j]=inList->curves[i].col[j];
	      outList->curves[noc2].depth[j]=inList->curves[i].depth[j];
	    }
	  outList->curves[noc2].trash='K';
	  /* Move on to check next curve */
	}
    }
  return;
}

/*
 * Diagnostic function to write a data file of map values and the
 * corresponding time series values to build that map
 */
void WriteMap(
	      TimeFreqRep       map,
	      REAL4Vector       signal
	      )

{
  INT4         i;
  INT4         j;
  INT4         k;
  FILE        *fp;
  FILE        *fp2;
  FILE        *fp3;

  fp = fopen("OutMap.dat","w");
  fp2 = fopen("OutMap2.dat","w");
  fp3 = fopen("OutSignal.dat","w");
  fprintf(fp,"Number of Time Columns: %i\n",map.tCol);
  fprintf(fp,"Number of Freq Rows:    %i\n",map.fRow);
  fprintf(fp,"Time Instants to follow\n");
  for (i = 0;i < map.tCol;i++)
    {
      fprintf(fp,"%d\n",map.timeInstant[i]);
    }
  fprintf(fp,"Freq Instants to follow\n");
  for (i = 0;i < (map.fRow/2+1);i++)
    {
      fprintf(fp,"%e\n",map.freqBin[i]);
    }
  fprintf(fp,"Raw Map ouput to follow space delimited carriage return marks end of row\n\n");
  k = 0;
  i = 0;
  j = 0;
  for (i = 0;i < map.tCol;i++)
    { 
      for (j = 0;j <(map.fRow/2+1);j++)
	{
	  fprintf(fp2,"%f\n",(&map)->map[i][j]);
	  k++;
	};
    };
  if (&signal != NULL)
    {
      for (i=0;i < ((INT4) signal.length);i++)
	{
	  fprintf(fp3,"%f\n",signal.data[i]);
	};
    };
  fclose(fp);
  fclose(fp2);
}

/* 
 * Diagnostic code to create greyscale pgms of the map
 */
void DumpTFImage(
		 REAL4        **image,
		 const CHAR    *filename,
		 INT4           height,
		 INT4           width,
		 BOOLEAN        killNeg
		 )
{
  INT4       i;
  INT4       j;
  FILE      *fp;
  REAL4      maxval=0;
  REAL4      maxnum;
  REAL4      minval=0;
  REAL8      meanval=0;
  INT4       counter=0;
  REAL8      stdDev=0;
  REAL8      sumXsqr=0;
  REAL8      sumX=0;
  INT4       pgmval;
  REAL4      temppgm;
  REAL4      currentval;
  REAL4      currentabsval;
  REAL4      temppoint;
  CHAR*      pgmfilename;
  CHAR*      datfilename;
  CHAR*      auxfilename;
  CHAR       PGM[5]=".pgm";
  CHAR       DAT[5]=".dat";
  CHAR       AUX[5]=".aux";
  CHAR       newFilename[256]="";
  INT4       scale=255;
  static INT4 callCounter=0;

  /* Alloc for two filenames dat and pgm */
  sprintf(newFilename,"%s_%i_",filename,callCounter++);
  pgmfilename = (CHAR*)LALCalloc((strlen(newFilename)+strlen(PGM)+1),sizeof(CHAR));
  datfilename = (CHAR*)LALCalloc((strlen(newFilename)+strlen(DAT)+1),sizeof(CHAR));
  auxfilename = (CHAR*)LALCalloc((strlen(newFilename)+strlen(DAT)+1),sizeof(CHAR));
  strcat(pgmfilename,newFilename);
  strcat(pgmfilename,PGM);
  strcat(datfilename,newFilename);
  strcat(datfilename,DAT);
  strcat(auxfilename,newFilename);
  strcat(auxfilename,AUX);

  /* Getting max image value to normalize to scale */
  /* Write Space Delimited table */
  temppoint=0;
  fp = fopen(datfilename,"w");
  maxnum = image[0][0]; /* Max Value no abs taken on value */
  for (i=(width-1);i>-1;i--)
    { 
      for (j=0;j<height;j++)
	{
	  temppoint = image[j][i];
	  if (killNeg)
	    {
	      if (image[j][i] < 0)
		{
		  temppoint = 0;
		}
	    };
	  currentval = temppoint;
	  currentabsval = fabs(temppoint);
	  /* To figure out mean and stddev*/
	  sumX=sumX+currentval;
	  sumXsqr=sumXsqr+(currentval*currentval);
	  counter++;
	  if (maxval < currentabsval)
	    {
	      maxval = currentabsval;
	    }
	  if (maxnum < currentval)
	    {
	      maxnum = currentval;
	    }
	  if (minval > currentval)
	    {
	      minval = currentval;
	    }
	  fprintf(fp,"%6.18f ",currentval);
	}
      fprintf(fp,"\n");
    }
  fclose(fp);
  /* PGM File Creation */
  fp = fopen(pgmfilename,"w");
  fprintf(fp,"P2\n");
  fprintf(fp,"#Written by ImageDump\n");
  fprintf(fp,"%i %i\n",height,width);
  fprintf(fp,"%i\n",scale);
  for (i=(width-1);i>-1;i--)
    {
      for (j=0;j<height;j++)
	{
	  temppoint = image[j][i];
	  if (killNeg)
	    {
	      if (image[j][i] < 0)
		{
		  temppoint = 0;
		}
	    };
	  currentval = temppoint;
	  temppgm = ((currentval-minval)*(scale/(maxnum-minval)));
	  pgmval = floor(temppgm);
	  /*pgmval = floor(((currentval-minval)*255)/(maxval-minval));*/
	  /*pgmval = floor((currentval-minval)*(255/(maxval-minval)));*/
          fprintf(fp,"%i\n",pgmval);
	}
    }
  fclose(fp);
  /* Writing Aux file for information purpose only */
  meanval=sumX/counter;
  stdDev=sqrt((sumXsqr-(sumX*meanval))/counter-1);
  fp = fopen(auxfilename,"w");
  fprintf(fp,"Aux Data information\n");
  fprintf(fp,"Data ABS Max Value Found:%e\n",maxval);
  fprintf(fp,"Data Max Value Found    :%e\n",maxnum);
  fprintf(fp,"Data Min Value Found    :%e\n",minval);
  fprintf(fp,"Data Dim Height %i\n",height);
  fprintf(fp,"Data Dim Width  %i\n",width);
  fprintf(fp,"Data Mean    : %e\n",meanval);
  fprintf(fp,"Data STDDEV  : %e\n",stdDev);
  fprintf(fp,"Data Elements: %i\n",counter);
  fclose(fp);
  LALFree(auxfilename);
  LALFree(pgmfilename);
  LALFree(datfilename);
  return;
}
/*
 * Same as above function but can write to specified filename scheme
 */
void DumpTFImageCHAR(
		     CHAR         **image,
		     const CHAR           *filename,
		     INT4           height,
		     INT4           width,
		     BOOLEAN        killNeg
		     )
{
  INT4       i;
  INT4       j;
  FILE      *fp;
  REAL4      maxval;
  REAL4      maxnum;
  REAL4      minval;
  INT4       pgmval;
  REAL4      temppgm;
  REAL4      currentval;
  REAL4      currentabsval;
  REAL4      temppoint;
  CHAR*      pgmfilename;
  CHAR*      datfilename;
  CHAR*      auxfilename;
  CHAR       PGM[5]=".pgm";
  CHAR       DAT[5]=".dat";
  CHAR       AUX[5]=".aux";
  INT4       scale=255;
  
  maxval=0; /* Hold max abs (magnitude) value */
  minval=0;
  /* Alloc for two filenames dat and pgm */
  pgmfilename = (CHAR*)LALCalloc((strlen(filename)+strlen(PGM)+1),sizeof(CHAR));
  datfilename = (CHAR*)LALCalloc((strlen(filename)+strlen(DAT)+1),sizeof(CHAR));
  auxfilename = (CHAR*)LALCalloc((strlen(filename)+strlen(DAT)+1),sizeof(CHAR));
  strcat(pgmfilename,filename);
  strcat(pgmfilename,PGM);
  strcat(datfilename,filename);
  strcat(datfilename,DAT);
  strcat(auxfilename,filename);
  strcat(auxfilename,AUX);

  /* Getting max image value to normalize to scale */
  /* Write Space Delimited table */
  temppoint=0;
  fp = fopen(datfilename,"w");
  maxnum = atof(&image[0][0]); /* Max Value no abs taken on value */
  for (i=(width-1);i>-1;i--)
    { 
      for (j=0;j<height;j++)
	{
	  temppoint = atof(&image[j][i]);
	  if (killNeg)
	    {
	      if (atof(&image[j][i]) < 0)
		{
		  temppoint = 0;
		}
	    };
	  currentval = temppoint;
	  currentabsval = fabs(temppoint);
	  if (maxval < currentabsval)
	    {
	      maxval = currentabsval;
	    }
	  if (maxnum < currentval)
	    {
	      maxnum = currentval;
	    }
	  if (minval > currentval)
	    {
	      minval = currentval;
	    }
	  fprintf(fp,"%f ",currentval);
	}
      fprintf(fp,"\n");
    }
  fclose(fp);
  /* PGM File Creation */
  fp = fopen(pgmfilename,"w");
  fprintf(fp,"P2\n");
  fprintf(fp,"#Written by ImageDump\n");
  fprintf(fp,"%i %i\n",height,width);
  fprintf(fp,"%i\n",scale);
  for (i=(width-1);i>-1;i--)
    {
      for (j=0;j<height;j++)
	{
	  temppoint = (atof(&image[j][i]));
	  if (killNeg)
	    {
	      if (atof(&image[j][i]) < 0)
		{
		  temppoint = 0;
		}
	    };
	  currentval = temppoint;
	  temppgm = ((currentval-minval)*(scale/(maxnum-minval)));
	  pgmval = floor(temppgm);
	  /*pgmval = floor(((currentval-minval)*255)/(maxval-minval));*/
	  /*pgmval = floor((currentval-minval)*(255/(maxval-minval)));*/
          fprintf(fp,"%i\n",pgmval);
	}
    }
  fclose(fp);
  /* Writing Aux file for information purpose only */
  fp = fopen(auxfilename,"w");
  fprintf(fp,"Aux Data information\n");
  fprintf(fp,"Data ABS Max Value Found:%e\n",maxval);
  fprintf(fp,"Data Max Value Found    :%e\n",maxnum);
  fprintf(fp,"Data Min Value Found    :%e\n",minval);
  fprintf(fp,"Data Dim Height %i\n",height);
  fprintf(fp,"Data Dim Width  %i\n",width);
  fclose(fp);
  LALFree(auxfilename);
  LALFree(pgmfilename);
  LALFree(datfilename);
  return;
}

/*
 * Creates small text files with a profile of the gaussian kernels and
 * corresponding derivatives
 */
void DumpREAL8KernelMask(
			 REAL8      *kernel,
			 const CHAR       *filename,
			 INT4        ksize
			 )
{
  INT4       i;
  CHAR       DAT[5]=".dat";
  CHAR       AUX[5]=".aux";
  REAL8      minvalue=0;
  REAL8      maxvalue=0;
  CHAR*      datfilename;
  CHAR*      auxfilename;
  FILE      *fp;

  datfilename = (CHAR*)LALCalloc((strlen(filename)+strlen(DAT)+1),sizeof(CHAR));
  auxfilename = (CHAR*)LALCalloc((strlen(filename)+strlen(DAT)+1),sizeof(CHAR));
  strcat(datfilename,filename);
  strcat(datfilename,DAT);
  strcat(auxfilename,filename);
  strcat(auxfilename,AUX);
  minvalue=kernel[0];
  maxvalue=kernel[0];
  fp = fopen(datfilename,"w");
  for (i=0;i<ksize;i++)
    {
      if (maxvalue < kernel[i])
	{
	  maxvalue = kernel[i];
	}
      if (minvalue > kernel[i])
	{
	  minvalue = kernel[i];
	}
      fprintf(fp,"%e\n",kernel[i]);
    }
  fclose(fp);
  /* Write Aux file also */
  fp =fopen(auxfilename,"w");
  fprintf(fp,"Aux Data for Kernel\n");
  fprintf(fp,"Data Min Value .....%e\n",minvalue);
  fprintf(fp,"Data Max Value .....%e\n",maxvalue);
  fprintf(fp,"Kernel Length ......%i\n",ksize);
  fclose(fp);
  LALFree(auxfilename);
  LALFree(datfilename);
  return;
}


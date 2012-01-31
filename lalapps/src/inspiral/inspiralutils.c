/*
*  Copyright (C) 2007 Patrick Brady
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
 * File Name: inspiralutils.c
 *
 * Author: Brown, D. A., Krishnan, B
 *
 * Revision: $Id$
 *
 *-----------------------------------------------------------------------
 */

#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <regex.h>
#include <time.h>
#include <math.h>

#include <lalapps.h>
#include <series.h>
#include <processtable.h>
#include <lalappsfrutils.h>

#include <lal/LALConfig.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALDatatypes.h>
#include <lal/AVFactories.h>
#include <lal/LALConstants.h>
#include <lal/PrintFTSeries.h>
#include <lal/FrameStream.h>
#include <lal/ResampleTimeSeries.h>
#include <lal/Calibration.h>
#include <lal/FrameCalibration.h>
#include <lal/Window.h>
#include <lal/TimeFreqFFT.h>
#include <lal/IIRFilter.h>
#include <lal/BandPassTimeSeries.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/Date.h>
#include <lal/Units.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpSP.h>
#include <lal/FindChirpTD.h>
#include <lal/FindChirpBCV.h>
#include <lal/FindChirpBCVSpin.h>
#include <lal/FindChirpChisq.h>
#include <lal/LALFrameL.h>

#include <lal/LALSimulation.h>

#include "inspiral.h"


RCSID( "$Id$" );


REAL4 compute_candle_distance(REAL4 candleM1, REAL4 candleM2,
    REAL4 thissnr, REAL8 chanDeltaT, INT4 nPoints,
    REAL8FrequencySeries *spec, UINT4 cut)
{
  UINT4 k;
  REAL8 sigmaSqSum = 0;
  REAL8 distance = 0;
  REAL8 negativeSevenOverThree = -7.0/3.0;
  REAL8 totalMass = candleM1 + candleM2;
  REAL8 mu = candleM1 * candleM2 / totalMass;
  REAL8 distNorm = 2.0 * LAL_MRSUN_SI / (1.0e6 * LAL_PC_SI );
  REAL8 a = sqrt( (5.0 * mu) / 96.0 ) *
    pow( totalMass / ( LAL_PI * LAL_PI ), 1.0/3.0 ) *
    pow( LAL_MTSUN_SI / chanDeltaT, -1.0/6.0 );
  REAL8 sigmaSq = 4.0 * ( chanDeltaT / (REAL8) nPoints ) *
    distNorm * distNorm * a * a;
  REAL8 f_max = 1.0 / (6.0 * sqrt(6.0) * LAL_PI * totalMass * LAL_MTSUN_SI);
  REAL8 f = 0;

  for ( k = cut, f = spec->deltaF * cut;
      k < spec->data->length && f < f_max;
      ++k, f = spec->deltaF * k )
  {
    sigmaSqSum +=
      pow( (REAL8) k / (REAL8) nPoints, negativeSevenOverThree )
      / spec->data->data[k];
  }

  sigmaSq *= sigmaSqSum;

  distance = sqrt( sigmaSq ) / thissnr;

  return distance;
}

REAL4 XLALCandleDistanceTD(
    Approximant approximant,
    REAL4 candleM1,
    REAL4 candleM2,
    REAL4 candlesnr,
    REAL8 chanDeltaT,
    INT4 nPoints,
    REAL8FrequencySeries *spec,
    UINT4 cut)
{

  LALStatus      status   = blank_status;

  InspiralTemplate  tmplt;
  REAL4Vector    *waveform = NULL;
  COMPLEX8Vector *waveFFT  = NULL;
  REAL4FFTPlan   *fwdPlan  = NULL;

  REAL8          sigmaSq;
  REAL8          distance;
  UINT4          i;

  memset( &tmplt, 0, sizeof(tmplt) );

  /* Create storage for TD and FD template */
  waveform = XLALCreateREAL4Vector( nPoints );
  waveFFT  = XLALCreateCOMPLEX8Vector( spec->data->length );
  fwdPlan  = XLALCreateForwardREAL4FFTPlan( nPoints, 0 );

  /* Populate the template parameters */
  tmplt.mass1 = candleM1;
  tmplt.mass2 = candleM2;
  tmplt.ieta  = 1;
  tmplt.approximant = approximant;
  tmplt.tSampling   = 1.0/chanDeltaT;
  tmplt.order       = LAL_PNORDER_PSEUDO_FOUR; /* Hardcode for EOBNR for now */
  tmplt.fLower      = spec->deltaF *cut;
  tmplt.distance    = 1.0e6 * LAL_PC_SI; /* Mpc */
  tmplt.massChoice  = m1Andm2;
  tmplt.fCutoff     = tmplt.tSampling / 2.0 - spec->deltaF;

  /* From this, calculate the other parameters */
  LAL_CALL( LALInspiralParameterCalc( &status, &tmplt ), &status );

  /* Generate the waveform */
  LAL_CALL( LALInspiralWave( &status, waveform, &tmplt ), &status );

  XLALREAL4ForwardFFT( waveFFT, waveform, fwdPlan );

  sigmaSq = 0.0;
  for ( i = cut; i < waveFFT->length; i++ )
  {
    sigmaSq += ( waveFFT->data[i].re * waveFFT->data[i].re
            + waveFFT->data[i].im * waveFFT->data[i].im )
            / spec->data->data[i];
  }

  sigmaSq *= 4.0 * chanDeltaT / nPoints;

  /* Now calculate the distance */
  distance = sqrt( sigmaSq ) / (REAL8)candlesnr;

  /* Clean up! */
  XLALDestroyREAL4Vector( waveform );
  XLALDestroyCOMPLEX8Vector( waveFFT );
  XLALDestroyREAL4FFTPlan( fwdPlan );

  return (REAL4)distance;
}

SummValueTable **add_summvalue_table(SummValueTable **newTable,
    LIGOTimeGPS gpsStartTime, LIGOTimeGPS gpsEndTime,
    const CHAR *programName, const CHAR *ifoName,
    const CHAR *summValueName, const CHAR *comment, REAL8 value
    )
{
  /* add value to summ_value table */
  *newTable = (SummValueTable *) LALCalloc( 1, sizeof(SummValueTable) );
  snprintf( (*newTable)->program, LIGOMETA_PROGRAM_MAX,
      "%s", programName );
  (*newTable)->version = 0;
  (*newTable)->start_time = gpsStartTime;
  (*newTable)->end_time = gpsEndTime;
  snprintf( (*newTable)->ifo, LIGOMETA_IFO_MAX, "%s", ifoName );
  snprintf( (*newTable)->name, LIGOMETA_SUMMVALUE_NAME_MAX,
      "%s", summValueName );
  snprintf( (*newTable)->comment, LIGOMETA_SUMMVALUE_COMM_MAX,
      "%s", comment );
  (*newTable)->value = value;

  return (newTable);
}



void AddNumRelStrainModes(  LALStatus              *status,     /**< pointer to LALStatus structure */
                            REAL4TimeVectorSeries  **outStrain, /**< [out]  h+, hx data    */
                            SimInspiralTable *thisinj           /**< [in]   injection data */)
{
  INT4 modeL, modeM, modeLlo, modeLhi;
  INT4 len, lenPlus, lenCross, k;
  CHAR *channel_name_plus;
  CHAR *channel_name_cross;
  FrStream  *frStream = NULL;
  FrCache frCache;
  LIGOTimeGPS epoch;
  REAL4TimeSeries  *seriesPlus=NULL;
  REAL4TimeSeries  *seriesCross=NULL;
  REAL8 massMpc;
  REAL4TimeVectorSeries *sumStrain=NULL;
  REAL4TimeVectorSeries *tempStrain=NULL;
  /*   NRWaveMetaData thisMetaData; */

  INITSTATUS (status, "LALAddStrainModes", rcsid);
  ATTATCHSTATUSPTR (status);

  modeLlo = thisinj->numrel_mode_min;
  modeLhi = thisinj->numrel_mode_max;

  /* create a frame cache and open the frame stream */
  frCache.numFrameFiles = 1;
  frCache.frameFiles = LALCalloc(1, sizeof(frCache.frameFiles[0]));
  frCache.frameFiles[0].url = thisinj->numrel_data;
  frStream = XLALFrCacheOpen( &frCache );

  /* the total mass of the binary in Mpc */
  massMpc = (thisinj->mass1 + thisinj->mass2) * LAL_MRSUN_SI / ( LAL_PC_SI * 1.0e6);

  /* start time of waveform -- set it to something */
  epoch.gpsSeconds     = 0;
  epoch.gpsNanoSeconds = 0;

  /* loop over l values */
  for ( modeL = modeLlo; modeL <= modeLhi; modeL++ ) {

    /* loop over m values */
    for ( modeM = -modeL; modeM <= modeL; modeM++ ) {
      /* read numrel waveform */
      /* first the plus polarization */
      channel_name_plus = XLALGetNinjaChannelName("plus", modeL, modeM);
      /*get number of data points */
      lenPlus = XLALFrGetVectorLength ( channel_name_plus, frStream );

      /* now the cross polarization */
      channel_name_cross = XLALGetNinjaChannelName("cross", modeL, modeM);
      /*get number of data points */
      lenCross = XLALFrGetVectorLength ( channel_name_cross, frStream );

      /* skip on to next mode if mode doesn't exist */
      if ( (lenPlus <= 0) || (lenCross <= 0) || (lenPlus != lenCross) ) {
        XLALClearErrno();
        LALFree(channel_name_plus);
        LALFree(channel_name_cross);
        continue;
      }

      /* note: lenPlus and lenCross must be equal if we got this far*/
      /* len = lenPlus; */
      len = lenPlus;

      /* allocate and read the plus/cross time series */
      seriesPlus = XLALCreateREAL4TimeSeries ( channel_name_plus, &epoch, 0, 0, &lalDimensionlessUnit, len);
      memset(seriesPlus->data->data, 0, seriesPlus->data->length*sizeof(REAL4));
      XLALFrGetREAL4TimeSeries ( seriesPlus, frStream );
      XLALFrRewind( frStream );
      LALFree(channel_name_plus);

      seriesCross = XLALCreateREAL4TimeSeries ( channel_name_cross, &epoch, 0, 0, &lalDimensionlessUnit, len);
      memset(seriesCross->data->data, 0, seriesCross->data->length*sizeof(REAL4));
      XLALFrGetREAL4TimeSeries ( seriesCross, frStream );
      XLALFrRewind( frStream );
      LALFree(channel_name_cross);

      /* allocate memory for tempStrain */
      tempStrain = LALCalloc(1, sizeof(*tempStrain));
      tempStrain->data = XLALCreateREAL4VectorSequence(2, len);
      tempStrain->deltaT = LAL_MTSUN_SI * (thisinj->mass1 + thisinj->mass2) * seriesPlus->deltaT ;
      tempStrain->f0 = seriesPlus->f0;
      tempStrain->sampleUnits = seriesPlus->sampleUnits;
      memset(tempStrain->data->data, 0, tempStrain->data->length * tempStrain->data->vectorLength*sizeof(REAL4));

      /* now copy the data and scale amplitude corresponding to distance of 1Mpc*/
      for (k = 0; k < len; k++) {
        tempStrain->data->data[k] = massMpc * seriesPlus->data->data[k];
        tempStrain->data->data[len + k] = massMpc * seriesCross->data->data[k];            }

        /* we are done with seriesPlus and Cross for this iteration */
        XLALDestroyREAL4TimeSeries (seriesPlus);
        XLALDestroyREAL4TimeSeries (seriesCross);
        seriesPlus = NULL;
        seriesCross = NULL;

        /* compute the h+ and hx for given inclination and coalescence phase*/
        XLALOrientNRWave( tempStrain, modeL, modeM, thisinj->inclination, thisinj->coa_phase );
        if (sumStrain == NULL) {

          sumStrain = LALCalloc(1, sizeof(*sumStrain));
          sumStrain->data =  XLALCreateREAL4VectorSequence(2, tempStrain->data->vectorLength);
          sumStrain->deltaT = tempStrain->deltaT;
          sumStrain->f0 = tempStrain->f0;
          sumStrain->sampleUnits = tempStrain->sampleUnits;

          memset(sumStrain->data->data,0.0,2*tempStrain->data->vectorLength*sizeof(REAL4));

          sumStrain = XLALSumStrain( sumStrain, tempStrain );

        } else {
          sumStrain = XLALSumStrain( sumStrain, tempStrain );
        }

        /* clear memory for strain */
        if (tempStrain->data != NULL) {
          XLALDestroyREAL4VectorSequence ( tempStrain->data );
          LALFree( tempStrain );
          tempStrain = NULL;
      }
    } /* end loop over modeM values */
  } /* end loop over modeL values */

  XLALFrClose( frStream );
  LALFree(frCache.frameFiles);
  *outStrain = sumStrain;
  DETATCHSTATUSPTR(status);
  RETURN(status);

}

void AddNumRelStrainModesREAL8(LALStatus      *status,  /**< pointer to LALStatus structure */
                            REAL8TimeSeries   **seriesPlus, /**< [out]  h+, hx data    */
                            REAL8TimeSeries   **seriesCross, /**< [out]  h+, hx data    */
                            SimInspiralTable  *thisinj  /**< [in]   injection data */)
{
  INT4 modeL, modeM, modeLlo, modeLhi;
  INT4 len, lenPlus, lenCross, k;
  CHAR *channel_name_plus;
  CHAR *channel_name_cross;
  FrStream  *frStream = NULL;
  FrCache frCache;
  LIGOTimeGPS epoch;
  REAL8TimeSeries  *modePlus=NULL;
  REAL8TimeSeries  *modeCross=NULL;
  REAL8 massMpc, timeStep;
  INITSTATUS (status, "AddNumRelStrainModesREAL8", rcsid);
  ATTATCHSTATUSPTR (status);

  modeLlo = thisinj->numrel_mode_min;
  modeLhi = thisinj->numrel_mode_max;

  /* create a frame cache and open the frame stream */
  frCache.numFrameFiles     = 1;
  frCache.frameFiles        = LALCalloc(1, sizeof(frCache.frameFiles[0]));
  frCache.frameFiles[0].url = thisinj->numrel_data;
  frStream                  = XLALFrCacheOpen( &frCache );

  /* the total mass of the binary in Mpc */
  massMpc = (thisinj->mass1 + thisinj->mass2) * LAL_MRSUN_SI / ( LAL_PC_SI * 1.0e6);

  /* Time step in dimensionful units */
  timeStep = (thisinj->mass1 + thisinj->mass2) * LAL_MTSUN_SI;

  /* start time of waveform -- set it to something */
  epoch.gpsSeconds     = thisinj->geocent_end_time.gpsSeconds;
  epoch.gpsNanoSeconds = thisinj->geocent_end_time.gpsNanoSeconds;

  /* loop over l values */
  for ( modeL = modeLlo; modeL <= modeLhi; modeL++ ) {

    /* loop over m values */
    for ( modeM = -modeL; modeM <= modeL; modeM++ ) {
      /* read numrel waveform */
      /* first the plus polarization */
      channel_name_plus = XLALGetNinjaChannelName("plus", modeL, modeM);
      /*get number of data points */
      lenPlus = XLALFrGetVectorLength ( channel_name_plus, frStream );

      /* now the cross polarization */
      channel_name_cross = XLALGetNinjaChannelName("cross", modeL, modeM);
      /*get number of data points */
      lenCross = XLALFrGetVectorLength ( channel_name_cross, frStream );

      /* skip on to next mode if mode doesn't exist */
      if ( (lenPlus <= 0) || (lenCross <= 0) || (lenPlus != lenCross) ) {
        XLALClearErrno();
        LALFree(channel_name_plus);
        LALFree(channel_name_cross);
        continue;
      }

      /* note: lenPlus and lenCross must be equal if we got this far*/
      len = lenPlus;

      /* allocate and read the plus/cross time series */
      modePlus = XLALCreateREAL8TimeSeries ( channel_name_plus, &epoch, 0, 0, &lalDimensionlessUnit, len);
      memset(modePlus->data->data, 0, modePlus->data->length*sizeof(REAL8));
      XLALFrGetREAL8TimeSeries ( modePlus, frStream );
      XLALFrRewind( frStream );
      LALFree(channel_name_plus);

      modeCross = XLALCreateREAL8TimeSeries ( channel_name_cross, &epoch, 0, 0, &lalDimensionlessUnit, len);
      memset(modeCross->data->data, 0, modeCross->data->length*sizeof(REAL8));
      XLALFrGetREAL8TimeSeries ( modeCross, frStream );
      XLALFrRewind( frStream );
      LALFree(channel_name_cross);

      /* scale and add */
      if (*seriesPlus == NULL) {
          *seriesPlus = XLALCreateREAL8TimeSeries ( "hplus", &epoch, 0, 0, &lalDimensionlessUnit, len);
          memset((*seriesPlus)->data->data, 0, (*seriesPlus)->data->length*sizeof(REAL8));
          (*seriesPlus)->deltaT = modePlus->deltaT;
      }

      if (*seriesCross == NULL) {
          *seriesCross = XLALCreateREAL8TimeSeries ( "hcross", &epoch, 0, 0, &lalDimensionlessUnit, len);
          memset((*seriesCross)->data->data, 0, (*seriesCross)->data->length*sizeof(REAL8));
          (*seriesCross)->deltaT = modeCross->deltaT;
      }

      XLALOrientNRWaveTimeSeriesREAL8( modePlus, modeCross, modeL, modeM, thisinj->inclination, thisinj->coa_phase );

      for (k = 0; k < len; k++) {
        (*seriesPlus)->data->data[k]  += massMpc * modePlus->data->data[k];
        (*seriesCross)->data->data[k] += massMpc * modeCross->data->data[k];
      }

      /* we are done with seriesPlus and Cross for this iteration */
      XLALDestroyREAL8TimeSeries (modePlus);
      XLALDestroyREAL8TimeSeries (modeCross);
    } /* end loop over modeM values */
  } /* end loop over modeL values */
  (*seriesPlus)->deltaT  *= timeStep;
  (*seriesCross)->deltaT *= timeStep;
  XLALFrClose( frStream );
  LALFree(frCache.frameFiles);
  DETATCHSTATUSPTR(status);
  RETURN(status);
}

/** Main function for injecting numetrical relativity waveforms.
    Takes as input a list of injections, and adds h(t) to a given
    timeseries for a specified ifo and a dynamic range factor.
*/
void InjectNumRelWaveforms (LALStatus           *status,       /**< pointer to LALStatus structure */
                            REAL4TimeSeries     *chan,         /**< [out] the output time series */
                            SimInspiralTable    *injections,   /**< [in] list of injections */
                            CHAR                ifo[3],        /**< [in] 2 char code for interferometer */
                            REAL8               dynRange,      /**< [in] dynamic range factor for scaling time series */
                            REAL8               freqLowCutoff, /**< [in] Lower cutoff frequency */
                            REAL8               snrLow,        /**< [in] lower cutoff value of snr */
                            REAL8               snrHigh,       /**< TO BE DOCUMENTED */
                            CHAR                *fname)        /**< [in] higher cutoff value of snr */
{

  REAL4TimeVectorSeries *tempStrain=NULL;
  SimInspiralTable    *thisInj = NULL;
  REAL8 startFreq, startFreqHz, massTotal;
  REAL8 thisSNR;
  SimInspiralTable *simTableOut=NULL;
  SimInspiralTable *thisInjOut=NULL;

  INITSTATUS (status, "InjectNumRelWaveforms", rcsid);
  ATTATCHSTATUSPTR (status);
  ASSERT( chan, status, INSPIRALH_ENULL, INSPIRALH_MSGENULL );
  ASSERT( ifo, status, INSPIRALH_ENULL, INSPIRALH_MSGENULL );


  /* loop over injections */
  for ( thisInj = injections; thisInj; thisInj = thisInj->next )
    {

      startFreq = start_freq_from_frame_url(thisInj->numrel_data);
      massTotal = (thisInj->mass1 + thisInj->mass2) * LAL_MTSUN_SI;
      startFreqHz = startFreq / ( LAL_TWOPI * massTotal);

       if (startFreqHz < freqLowCutoff)
       {
          TRY( AddNumRelStrainModes( status->statusPtr, &tempStrain, thisInj),
               status);
          thisSNR = calculate_ligo_snr_from_strain( tempStrain, thisInj, ifo);

          /* set channel name */
          snprintf( chan->name, LALNameLength * sizeof( CHAR ),
                    "%s:STRAIN", ifo );

          if ((thisSNR < snrHigh) && (thisSNR > snrLow))
            {
              /* simTableOut will be null only the first time */
              if ( simTableOut == NULL) {
                simTableOut = (SimInspiralTable *)LALCalloc( 1, sizeof(SimInspiralTable) );
                memcpy(simTableOut, thisInj, sizeof(*thisInj));
                simTableOut->next = NULL;
                thisInjOut = simTableOut;
              }
              else {
                thisInjOut->next = (SimInspiralTable *)LALCalloc( 1, sizeof(SimInspiralTable) );
                memcpy(thisInjOut->next, thisInj, sizeof(*thisInj));
                thisInjOut->next->next = NULL;
                thisInjOut = thisInjOut->next;
              }

              TRY( LALInjectStrainGW( status->statusPtr, chan, tempStrain, thisInj,
                                      ifo, dynRange), status);
            }

          XLALDestroyREAL4VectorSequence ( tempStrain->data);
          tempStrain->data = NULL;
          LALFree(tempStrain);
          tempStrain = NULL;
        }

    } /* loop over injectionsj */


  /* write and free the output simInspiral table */
  if ( simTableOut ) {

    LIGOLwXMLStream xmlfp;
    MetadataTable dummyTable;
      dummyTable.simInspiralTable = simTableOut;

    /* write the xml table of actual injections */
    memset( &xmlfp, 0, sizeof(LIGOLwXMLStream) );
    TRY( LALOpenLIGOLwXMLFile( status->statusPtr, &xmlfp, fname ), status );
    TRY( LALBeginLIGOLwXMLTable( status->statusPtr, &xmlfp, sim_inspiral_table ),
              status );

    TRY( LALWriteLIGOLwXMLTable( status->statusPtr, &xmlfp, dummyTable,
                                      sim_inspiral_table ), status );

    TRY( LALEndLIGOLwXMLTable ( status->statusPtr, &xmlfp ), status );
    TRY( LALCloseLIGOLwXMLFile ( status->statusPtr, &xmlfp ), status );

  }

  while (simTableOut) {
    thisInjOut = simTableOut;
    simTableOut = simTableOut->next;
    LALFree(thisInjOut);
  }

  DETATCHSTATUSPTR(status);
  RETURN(status);

}

/** Main function for injecting numetrical relativity waveforms.
    Takes as input a list of injections, and adds h(t) to a given
    timeseries for a specified ifo and a dynamic range factor.
*/
void InjectNumRelWaveformsREAL8 (LALStatus      *status,       /**< pointer to LALStatus structure */
                            REAL8TimeSeries     *chan,         /**< [out] the output time series */
                            SimInspiralTable    *injections,   /**< [in] list of injections */
                            CHAR                ifo[3],        /**< [in] 2 char code for interferometer */
                            REAL8               freqLowCutoff, /**< [in] Lower cutoff frequency */
                            REAL8               snrLow,        /**< [in] lower cutoff value of snr */
                            REAL8               snrHigh,       /**< TO BE DOCUMENTED */
                            CHAR                *fname)       /**< [in] higher cutoff value of snr */
{
  SimInspiralTable *thisInj = NULL;
  REAL8 startFreq, startFreqHz, massTotal;
  REAL8 thisSNR;
  SimInspiralTable *simTableOut=NULL;
  SimInspiralTable *thisInjOut=NULL;

  INITSTATUS (status, "InjectNumRelWaveformsREAL8", rcsid);
  ATTATCHSTATUSPTR (status);
  ASSERT( chan, status, INSPIRALH_ENULL, INSPIRALH_MSGENULL );
  ASSERT( ifo, status, INSPIRALH_ENULL, INSPIRALH_MSGENULL );


  /* loop over injections */
  for ( thisInj = injections; thisInj; thisInj = thisInj->next )
    {

      startFreq = start_freq_from_frame_url(thisInj->numrel_data);
      massTotal = (thisInj->mass1 + thisInj->mass2) * LAL_MTSUN_SI;
      startFreqHz = startFreq / ( LAL_TWOPI * massTotal);

      if (startFreqHz < freqLowCutoff)
        {
          REAL8TimeSeries *strain = NULL;
          strain  = XLALNRInjectionStrain(ifo, thisInj);
          thisSNR = calculate_ligo_snr_from_strain_real8(strain, ifo);

           /* set channel name */
           snprintf( chan->name, LALNameLength * sizeof( CHAR ),
                    "%s:STRAIN", ifo );

          if ((thisSNR < snrHigh) && (thisSNR > snrLow))
            {
              /* simTableOut will be null only the first time */
              if ( simTableOut == NULL) {
                simTableOut = (SimInspiralTable *)LALCalloc( 1, sizeof(SimInspiralTable) );
                memcpy(simTableOut, thisInj, sizeof(*thisInj));
                simTableOut->next = NULL;
                thisInjOut = simTableOut;
              }
              else {
                thisInjOut->next = (SimInspiralTable *)LALCalloc( 1, sizeof(SimInspiralTable) );
                memcpy(thisInjOut->next, thisInj, sizeof(*thisInj));
                thisInjOut->next->next = NULL;
                thisInjOut = thisInjOut->next;
              }

              XLALSimAddInjectionREAL8TimeSeries( chan, strain, NULL);
            }

          XLALDestroyREAL8TimeSeries (strain);
        }
    } /* loop over injectionsj */


  /* write and free the output simInspiral table */
  if ( simTableOut ) {

    LIGOLwXMLStream xmlfp;
    MetadataTable dummyTable;
    dummyTable.simInspiralTable = simTableOut;

    /* write the xml table of actual injections */
    memset( &xmlfp, 0, sizeof(LIGOLwXMLStream) );
    TRY( LALOpenLIGOLwXMLFile( status->statusPtr, &xmlfp, fname ), status );
    TRY( LALBeginLIGOLwXMLTable( status->statusPtr, &xmlfp, sim_inspiral_table ),
              status );

    TRY( LALWriteLIGOLwXMLTable( status->statusPtr, &xmlfp, dummyTable,
                                      sim_inspiral_table ), status );

    TRY( LALEndLIGOLwXMLTable ( status->statusPtr, &xmlfp ), status );
    TRY( LALCloseLIGOLwXMLFile ( status->statusPtr, &xmlfp ), status );

  }

  while (simTableOut) {
    thisInjOut = simTableOut;
    simTableOut = simTableOut->next;
    LALFree(thisInjOut);
  }

  DETATCHSTATUSPTR(status);
  RETURN(status);

}

REAL8 start_freq_from_frame_url(CHAR  *url)
{

  FrameH *frame=NULL;
  FrFile *frFile=NULL;
  FrHistory *frHist=NULL;
  FrHistory *thisHist;
  CHAR *comment=NULL;
  CHAR *token=NULL;
  REAL8 ret=0;

  frFile =  XLALFrOpenURL( url );
  frame = FrameRead (frFile);
  frHist = frame->history;
  thisHist = frHist;
  while (thisHist) {

    /* get history comment string and parse it */
    comment = LALCalloc(1, (strlen(thisHist->comment)+1)*sizeof(CHAR));
    strcpy(comment, thisHist->comment);

    token = strtok(comment,":");

    if (strstr(token,"freqStart22") || strstr(token,"freq_start_22")) {
      token = strtok(NULL,":");
      ret = atof(token);
    }

    LALFree(comment);
    comment = NULL;

    thisHist = thisHist->next;
  }

  FrFileIEnd( frFile );
  return ret;
}


REAL8 calculate_ligo_snr_from_strain_real8(  REAL8TimeSeries *strain,
                                       const CHAR            ifo[3])
{

  REAL8 ret = -1, snrSq, freq, psdValue;
  REAL8 deltaF;
  REAL8FFTPlan *pfwd;
  COMPLEX16FrequencySeries *fftData;
  UINT4 k;

  /* create the time series */
  deltaF  = strain->deltaT * strain->data->length;
  fftData = XLALCreateCOMPLEX16FrequencySeries( strain->name,  &(strain->epoch),
                                               0, deltaF, &lalDimensionlessUnit,
                                               strain->data->length/2 + 1 );

  /* perform the fft */
  pfwd = XLALCreateForwardREAL8FFTPlan( strain->data->length, 0 );
  XLALREAL8TimeFreqFFT( fftData, strain, pfwd );

  /* compute the SNR for initial LIGO at design */
  for ( snrSq = 0, k = 0; k < fftData->data->length; k++ )
    {
      freq = fftData->deltaF * k;

      if ( ifo[0] == 'V' )
        {
          if (freq < 35)
            continue;

          LALVIRGOPsd( NULL, &psdValue, freq );
          psdValue /= 9e-46;
        }
      else
        {
          if (freq < 40)
            continue;

          LALLIGOIPsd( NULL, &psdValue, freq );
        }

      fftData->data->data[k].re /= 3e-23;
      fftData->data->data[k].im /= 3e-23;
      snrSq += fftData->data->data[k].re * fftData->data->data[k].re / psdValue;
      snrSq += fftData->data->data[k].im * fftData->data->data[k].im / psdValue;
    }
  snrSq *= 4*fftData->deltaF;

  XLALDestroyREAL8FFTPlan( pfwd );
  XLALDestroyCOMPLEX16FrequencySeries( fftData );

  ret = sqrt(snrSq);
  return ret;
}


REAL8 calculate_ligo_snr_from_strain(  REAL4TimeVectorSeries *strain,
                                       SimInspiralTable      *thisInj,
                                       const CHAR            ifo[3])
{

  REAL8 ret = -1, snrSq, freq, psdValue;
  REAL8 sampleRate = 4096, deltaF;
  REAL4TimeSeries *chan = NULL;
  REAL4FFTPlan *pfwd;
  COMPLEX8FrequencySeries *fftData;
  UINT4 k;

  /* create the time series */
  chan    = XLALCalculateNRStrain( strain, thisInj, ifo, sampleRate );
  deltaF  = chan->deltaT * strain->data->vectorLength;
  fftData = XLALCreateCOMPLEX8FrequencySeries( chan->name,  &(chan->epoch),
                                               0, deltaF, &lalDimensionlessUnit,
                                               chan->data->length/2 + 1 );

  /* perform the fft */
  pfwd = XLALCreateForwardREAL4FFTPlan( chan->data->length, 0 );
  XLALREAL4TimeFreqFFT( fftData, chan, pfwd );

  /* compute the SNR for initial LIGO at design */
  for ( snrSq = 0, k = 0; k < fftData->data->length; k++ )
    {
      freq = fftData->deltaF * k;

      if ( ifo[0] == 'V' )
        {
          if (freq < 50)
            continue;

          LALVIRGOPsd( NULL, &psdValue, freq );
          psdValue /= 9e-46;
        }
      else
        {
          if (freq < 40)
            continue;

          LALLIGOIPsd( NULL, &psdValue, freq );
        }

      fftData->data->data[k].re /= 3e-23;
      fftData->data->data[k].im /= 3e-23;
      snrSq += fftData->data->data[k].re * fftData->data->data[k].re / psdValue;
      snrSq += fftData->data->data[k].im * fftData->data->data[k].im / psdValue;
    }
  snrSq *= 4*fftData->deltaF;

  XLALDestroyREAL4FFTPlan( pfwd );
  XLALDestroyCOMPLEX8FrequencySeries( fftData );

  XLALDestroyREAL4Vector ( chan->data);
  LALFree(chan);

  ret = sqrt(snrSq);     return ret;

}

REAL8TimeSeries *
XLALNRInjectionStrain(const char *ifo, SimInspiralTable *inj)
{
  static LALStatus status;

  REAL8TimeSeries *plus      = NULL;
  REAL8TimeSeries *plus_int  = NULL;
  REAL8TimeSeries *cross     = NULL;
  REAL8TimeSeries *cross_int = NULL;
  REAL8TimeSeries *strain    = NULL;

  INT4 sampleRate                = 16384;
  InterferometerNumber ifoNumber = LAL_UNKNOWN_IFO;
  LALDetector det;

  /* Add the modes together */
  AddNumRelStrainModesREAL8(&status, &plus, &cross, inj);
  /* Place at distance */
  for (uint j = 0; j < plus->data->length; j++)
  {
    plus->data->data[j]  /= inj->distance;
    cross->data->data[j] /= inj->distance;
  }

  plus->sampleUnits  = lalADCCountUnit;
  cross->sampleUnits = lalADCCountUnit;

  /* Interpolate to desired sample rate */
  plus_int  = XLALInterpolateNRWaveREAL8(plus, sampleRate);
  cross_int = XLALInterpolateNRWaveREAL8(cross, sampleRate);

  /* look up detector */
  memset( &det, 0, sizeof(LALDetector) );
  ifoNumber = XLALIFONumber( ifo );
  XLALReturnDetector( &det, ifoNumber );

  /* Use Jolien's method to place on the sky */
  strain = XLALSimDetectorStrainREAL8TimeSeries(plus_int, cross_int,
           inj->longitude, inj->latitude, inj->polarization, &det);

  /* We want the end time to be the time of largest amplitude */
  REAL8 max_value = 0;
  REAL8 offset    = 0;
  uint  max_idx   = 0;
  
  for (uint j = 0; j < strain->data->length; j++)
  {
    if (fabs(strain->data->data[j]) > max_value)
    {
      max_value = abs(strain->data->data[j]);
      max_idx   = j;
    }
  }

  offset = max_idx * strain->deltaT;

  XLALGPSAdd( &(strain->epoch), -offset);

  XLALDestroyREAL8TimeSeries (plus);
  XLALDestroyREAL8TimeSeries (plus_int);
  XLALDestroyREAL8TimeSeries (cross);
  XLALDestroyREAL8TimeSeries (cross_int);

  return strain;
}

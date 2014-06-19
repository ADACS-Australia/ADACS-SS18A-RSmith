//
// Copyright (C) 2012, 2013, 2014 Karl Wette
// Copyright (C) 2009 Chris Messenger, Reinhard Prix, Pinkesh Patel, Xavier Siemens, Holger Pletsch
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with with program; see the file COPYING. If not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA  02111-1307  USA
//

// This file implements the F-statistic resampling algorithm. It is not compiled directly, but
// included from ComputeFstat.c

#include <lal/LogPrintf.h>

/* Struct holding buffered ComputeFStat()-internal quantities to avoid unnecessarily
 * recomputing things that depend ONLY on the skyposition and detector-state series (but not on the spins).
 * For the first call of ComputeFStatFreqBand_RS() the pointer-entries should all be NULL.
 */
typedef struct tagComputeFBuffer_RS {
  MultiDetectorStateSeries *multiDetStates;             	/* buffer for each detStates (store pointer) and skypos */
  REAL8 Alpha, Delta;                                         	/* skyposition of candidate */
  LIGOTimeGPS segstart;                                       	/* the start time of the first SFT of the first detector (used to check if the segment has changed) */
  MultiSSBtimes *multiSSB;
  MultiAMCoeffs *multiAMcoef;
  MultiCOMPLEX8TimeSeries *multiTimeseries;                   /* the buffered unweighted multi-detector timeseries */
  MultiCOMPLEX8TimeSeries *multiFa_resampled;                 /* the buffered multi-detector resampled timeseries weighted by a(t) */
  MultiCOMPLEX8TimeSeries *multiFb_resampled;                 /* the buffered multi-detector resampled timeseries weighted by b(t) */
} ComputeFBuffer_RS;;

// internal Resampling-specific parameters
struct tagFstatInput_Resamp {
  MultiSFTVector *multiSFTs;            // Input multi-detector SFTs
  ComputeFBuffer_RS buffer; 		/* buffer for storing pre-resampled timeseries (used for resampling implementation) */
};

/* Destruction of a ComputeFBuffer *contents*,
 * i.e. the multiSSB and multiAMcoeff, while the
 * buffer-container is not freed (which is why it's passed
 * by value and not by reference...) */
static void
XLALEmptyComputeFBuffer_RS ( ComputeFBuffer_RS *buffer )
{
  if ( buffer->multiSSB ) XLALDestroyMultiSSBtimes( buffer->multiSSB );
  buffer->multiSSB = NULL;
  if ( buffer->multiAMcoef) XLALDestroyMultiAMCoeffs( buffer->multiAMcoef );
  buffer->multiAMcoef = NULL;
  if ( buffer->multiTimeseries) XLALDestroyMultiCOMPLEX8TimeSeries( buffer->multiTimeseries );
  buffer->multiTimeseries = NULL;
  if ( buffer->multiFa_resampled) XLALDestroyMultiCOMPLEX8TimeSeries( buffer->multiFa_resampled );
  buffer->multiFa_resampled = NULL;
  if ( buffer->multiFb_resampled) XLALDestroyMultiCOMPLEX8TimeSeries( buffer->multiFb_resampled );
  buffer->multiFb_resampled = NULL;
  if ( buffer->multiDetStates) XLALDestroyMultiDetectorStateSeries( buffer->multiDetStates);
  buffer->multiDetStates = NULL;
  /* if ( buffer ) XLALFree(buffer); */

  return;

} /* XLALEmptyComputeFBuffer_RS() */


static void
DestroyFstatInput_Resamp ( FstatInput_Resamp* resamp )
{
  XLALDestroyMultiSFTVector (resamp->multiSFTs );
  XLALEmptyComputeFBuffer_RS ( &resamp->buffer );
  XLALFree ( resamp );
  return;
} // DestroyFstatInput_Resamp()

static int
SetupFstatInput_Resamp ( FstatInput_Resamp *resamp,
                         const FstatInput_Common *common,
                         MultiSFTVector *multiSFTs
                         )
{
  // Check input
  XLAL_CHECK(common != NULL, XLAL_EFAULT);
  XLAL_CHECK(resamp != NULL, XLAL_EFAULT);
  XLAL_CHECK(multiSFTs != NULL, XLAL_EFAULT);

  // Save pointer to SFTs
  resamp->multiSFTs = multiSFTs;

  // clear buffer
  XLAL_INIT_MEM ( resamp->buffer );

  return XLAL_SUCCESS;

} // SetupFstatInput_Resamp()


static int
GetFstatExtraBins_Resamp ( FstatInput_Resamp* resamp )
{
  XLAL_CHECK(resamp != NULL, XLAL_EFAULT);
  return 8;	// use 8 extra bins to give better agreement with LALDemod(w Dterms=8) near the boundaries
} // GetFstatExtraBins_Resamp()


static int
ComputeFstat_Resamp ( FstatResults* Fstats,
                      const FstatInput_Common *common,
                      FstatInput_Resamp* resamp
                      )
{
  // Check input
  XLAL_CHECK ( Fstats != NULL, XLAL_EFAULT );
  XLAL_CHECK ( common != NULL, XLAL_EFAULT );
  XLAL_CHECK ( resamp != NULL, XLAL_EFAULT );

  // Get which F-statistic quantities to compute
  const FstatQuantities whatToCompute = Fstats->whatWasComputed;

  // Check which quantities can be computed
  XLAL_CHECK ( !(whatToCompute & FSTATQ_ATOMS_PER_DET), XLAL_EINVAL, "Resampling does not currently support atoms per detector" );

  const PulsarDopplerParams *thisPoint = &Fstats->doppler;
  MultiSFTVector *multiSFTs = resamp->multiSFTs;
  const MultiNoiseWeights *multiWeights = common->noiseWeights;
  ComputeFBuffer_RS *cfBuffer = &resamp->buffer;                      /* set local pointer to the buffer location */
  {// ================================================================================
    // Call ComputeFStatFreqBand_RS()
    UINT4 numDetectors;
    MultiDetectorStateSeries *multiDetStates = NULL;
    MultiSSBtimes *multiSSB = NULL;
    MultiAMCoeffs *multiAMcoef = NULL;
    MultiCOMPLEX8TimeSeries *multiTimeseries = NULL;
    SkyPosition skypos;
    MultiCOMPLEX8TimeSeries *multiFa_resampled = NULL;
    MultiCOMPLEX8TimeSeries *multiFb_resampled = NULL;
    UINT4 numSamples;
    REAL8 dt;
    ComplexFFTPlan *pfwd = NULL;  /* this will store the FFT plan */

    numDetectors = multiSFTs->length;               /* set the number of detectors to the number of sets of SFTs */
    // unused: SFTtype * firstSFT = &(multiSFTs->data[0]->data[0]);      /* use data from the first SFT from the first detector to set other params */

    /* check that the multidetector noise weights have the same length as the multiSFTs */
    XLAL_CHECK ( (multiWeights == NULL) || (multiWeights->length == numDetectors), XLAL_EINVAL );

    /* Dealing with the input SFT -> timeseries conversion and whether it has already been done and is buffered */

    /* generate bandpassed and downsampled timeseries for each detector                         */
    /* we only ever do this once for a given dataset so we read it from the buffer if it exists */
    /* in future implementations we will pass this directly to the function instead of SFTs     */

    /* check if there is an not existing timeseries and if the start time in the buffer does not match the start time of the SFTs */
    if ( !cfBuffer->multiTimeseries || ( XLALGPSCmp(&cfBuffer->segstart,&multiSFTs->data[0]->data[0].epoch) != 0) )
      {
        XLALPrintInfo ("*** New segment : recomputing timeseries and detstates\n");
        if ( !cfBuffer->multiTimeseries) { XLALPrintInfo("timeseries pointer was null\n"); }
        if ( XLALGPSCmp ( &cfBuffer->segstart, &multiSFTs->data[0]->data[0].epoch) != 0 ) {
          XLALPrintInfo("segstart changed from %d to %d\n",cfBuffer->segstart.gpsSeconds,multiSFTs->data[0]->data[0].epoch.gpsSeconds);
        }

        /* if there was no existing timeseries we need to recompute the timeseries from the SFTs */
        /* generate multiple coincident timeseries - one for each detector spanning start -> end */
        /* we need each timeseries to span the exact same amount of time and to start at the same time */
        /* because for the multi-detector Fstat we need frequency bins to be coincident */
        /* The memory allocated here is freed when the buffer is cleared in the calling program */

        /* generate a new timeseries from the input SFTs */
        XLAL_CHECK ( (multiTimeseries = XLALMultiSFTVectorToCOMPLEX8TimeSeries(multiSFTs)) != NULL, XLAL_EFUNC );

        /* recompute the multidetector states for the possibly time shifted SFTs */
        /* the function XLALMultiSFTVectorToCOMPLEX8TimeSeries may have shifted the SFT start times around */
        /* and since these times will be used later on for the resampling we also need to recompute the */
        /* MultiDetectorStates because their timestamps are used later on to compute SSB times which we */
        /* need to be accurate at the midpoints of the SFTs.  Understand ? */

        /* recompute the multiDetStates for the new SFT start times */
        MultiLALDetector multiIFO;
        XLAL_CHECK ( XLALMultiLALDetectorFromMultiSFTs ( &multiIFO, multiSFTs ) == XLAL_SUCCESS, XLAL_EFUNC );
        MultiLIGOTimeGPSVector *multiTS;
        XLAL_CHECK ( (multiTS = XLALExtractMultiTimestampsFromSFTs ( multiSFTs )) != NULL, XLAL_EFUNC );

        REAL8 Tsft = 1.0 / multiSFTs->data[0]->data[0].deltaF;
        REAL8 tOffset = 0.5 * Tsft;
        XLAL_CHECK ( (multiDetStates = XLALGetMultiDetectorStates ( multiTS, &multiIFO, common->ephemerides, tOffset )) != NULL, XLAL_EFUNC );
        XLALDestroyMultiTimestamps ( multiTS );

        /* set all other segment dependent quantity pointers to NULL */
        /* this will basically mean that we will have to compute all sky dependent quantities again */
        XLALEmptyComputeFBuffer_RS( cfBuffer );

        /* buffer the multitimeseries, detstates and the current start time of the input data */
        cfBuffer->multiTimeseries = multiTimeseries;
        cfBuffer->multiDetStates = multiDetStates;
        cfBuffer->segstart.gpsSeconds = multiSFTs->data[0]->data[0].epoch.gpsSeconds;
        cfBuffer->segstart.gpsNanoSeconds = multiSFTs->data[0]->data[0].epoch.gpsNanoSeconds;

      }  /* if (!cfBuffer->multiTimeseries || (buffered-start != SFT-start) ) */
    /* End of the SFT -> timeseries buffering checks */


    /* compute the fractional bin offset between the user requested initial frequency */
    /* and the closest output frequency bin */
    REAL8 diff = cfBuffer->multiTimeseries->data[0]->f0 - thisPoint->fkdot[0]; /* the difference between the new timeseries heterodyne frequency and the user requested lowest frequency */

    // use given frequency resolution or exactly 'diff' if dFreq=0 // FIXME: temporary fix until we properly figure out 1-bin resampling efficiently
    REAL8 dFreq = (Fstats->dFreq > 0) ? Fstats->dFreq : diff;
    INT4  diff_bins = (INT4)lround( diff / dFreq );           /* the rounded number of output frequency bins difference */
    REAL8 shift = diff - dFreq * diff_bins;                       /* the fractional bin frequency offset */

    /* Dealing with sky position dependent quantities and buffering them */

    /* if the sky position has changed or if any of the sky position dependent quantities are not buffered
       i.e the multiDetstates, the multiAMcoefficients, the multiSSB times and the resampled multiTimeSeries Fa and Fb,
       then we need to recompute these and buffer them */
    if ( (cfBuffer->Alpha != thisPoint->Alpha )                                 /* and alpha hasn't changed */
         || ( cfBuffer->Delta != thisPoint->Delta )                             /* and delta hasn't changed */
         || ( cfBuffer->multiAMcoef == NULL )                                 /* and we have a buffered multiAMcoefficents */
         || ( cfBuffer->multiSSB == NULL )                                    /* and we have buffered multiSSB times */
         || ( cfBuffer->multiFa_resampled == NULL )                           /* and we have buffered multiFa_resampled  */
         || ( cfBuffer->multiFb_resampled == NULL )                           /* and we have multiFb_resampled */
         || ( thisPoint->asini > 0 )						// no buffering in binary-CW case for now
         )
      {
        XLALPrintInfo("*** New sky position : recomputing SSB times, AM coefficients and Fa and Fb\n");

        /* compute the SSB times corresponding to the midpoints of each SFT for the current sky position for all detectors */
        skypos.system = COORDINATESYSTEM_EQUATORIAL;
        skypos.longitude = thisPoint->Alpha;
        skypos.latitude  = thisPoint->Delta;
        XLAL_CHECK ( (multiSSB = XLALGetMultiSSBtimes ( cfBuffer->multiDetStates, skypos, thisPoint->refTime, common->SSBprec )) != NULL, XLAL_EFUNC );

        MultiSSBtimes *multiBinary = NULL;
        MultiSSBtimes *multiSSBTotal = NULL;
        // handle binary-orbital timing corrections, if applicable
        if ( thisPoint->asini > 0 )
          {
            // compute binary time corrections to the SSB time delays and SSB time derivitive
            XLAL_CHECK ( XLALAddMultiBinaryTimes ( &multiBinary, multiSSB, thisPoint ) == XLAL_SUCCESS, XLAL_EFUNC );
            multiSSBTotal = multiBinary;
          }
        else
          {
            multiSSBTotal = multiSSB;
          }

        /* compute the AM parameters for each detector */
        XLAL_CHECK ( (multiAMcoef = XLALComputeMultiAMCoeffs ( cfBuffer->multiDetStates, multiWeights, skypos )) != NULL, XLAL_EFUNC );

        /* Generate a(t) and b(t) weighted heterodyned downsampled timeseries */
        MultiCOMPLEX8TimeSeries *multiFa = NULL;
        MultiCOMPLEX8TimeSeries *multiFb = NULL;
        XLAL_CHECK ( XLALAntennaWeightMultiCOMPLEX8TimeSeries ( &multiFa, &multiFb, cfBuffer->multiTimeseries, multiAMcoef, multiSFTs) == XLAL_SUCCESS, XLAL_EFUNC );

        /* Perform barycentric resampling on the multi-detector timeseries */
        XLAL_CHECK ( XLALBarycentricResampleMultiCOMPLEX8TimeSeries ( &multiFa_resampled, &multiFb_resampled, multiFa, multiFb, multiSSBTotal, multiSFTs, dFreq) == XLAL_SUCCESS, XLAL_EFUNC );

        XLALDestroyMultiSSBtimes ( multiBinary );

        /* free multiFa and MultiFb - we won't need them again since we're storing the resampled versions */
        XLALDestroyMultiCOMPLEX8TimeSeries ( multiFa );
        XLALDestroyMultiCOMPLEX8TimeSeries ( multiFb );

        /* buffer all new sky position dependent values - after clearing them */
        cfBuffer->Alpha = thisPoint->Alpha;
        cfBuffer->Delta = thisPoint->Delta;

        XLALDestroyMultiSSBtimes( cfBuffer->multiSSB );
        cfBuffer->multiSSB = multiSSB;

        XLALDestroyMultiAMCoeffs( cfBuffer->multiAMcoef );
        cfBuffer->multiAMcoef = multiAMcoef;

        XLALDestroyMultiCOMPLEX8TimeSeries( cfBuffer->multiFa_resampled );
        XLALDestroyMultiCOMPLEX8TimeSeries( cfBuffer->multiFb_resampled );
        cfBuffer->multiFa_resampled = multiFa_resampled;
        cfBuffer->multiFb_resampled = multiFb_resampled;

      } /* could not reuse previously buffered quantities */

    /* End of the sky position dependent quantity buffering */

    /* store AM coefficient integrals in local variables */
    REAL4 Ad = cfBuffer->multiAMcoef->Mmunu.Ad;
    REAL4 Bd = cfBuffer->multiAMcoef->Mmunu.Bd;
    REAL4 Cd = cfBuffer->multiAMcoef->Mmunu.Cd;
    REAL4 Ed = cfBuffer->multiAMcoef->Mmunu.Ed;
    REAL4 Dd = cfBuffer->multiAMcoef->Mmunu.Dd;
    REAL4 Dd_inv = 1.0f / Dd;

    // *copy* complete resampled multi-complex8 timeseries so we can apply spindown-corrections to it
    MultiCOMPLEX8TimeSeries *multiFa_spin, *multiFb_spin;
    XLAL_CHECK ( (multiFa_spin = XLALDuplicateMultiCOMPLEX8TimeSeries ( cfBuffer->multiFa_resampled )) != NULL, XLAL_EFUNC );
    XLAL_CHECK ( (multiFb_spin = XLALDuplicateMultiCOMPLEX8TimeSeries ( cfBuffer->multiFb_resampled )) != NULL, XLAL_EFUNC );

    /* shift the timeseries by a fraction of a frequency bin so that user requested frequency is exactly resolved */
    if (shift != 0.0)
      {
        XLAL_CHECK ( XLALFrequencyShiftMultiCOMPLEX8TimeSeries ( &multiFa_spin, shift ) == XLAL_SUCCESS, XLAL_EFUNC );
        XLAL_CHECK ( XLALFrequencyShiftMultiCOMPLEX8TimeSeries ( &multiFb_spin, shift ) == XLAL_SUCCESS, XLAL_EFUNC );
      }

    /* apply spin derivitive correction to resampled timeseries */
    /* this function only applies a correction if there are any non-zero spin derivitives */
    XLAL_CHECK ( XLALSpinDownCorrectionMultiFaFb ( &multiFa_spin, &multiFb_spin, thisPoint ) == XLAL_SUCCESS, XLAL_EFUNC );

    /* we now compute the FFTs of the resampled functions Fa and Fb for each detector */
    /* and combine them into the multi-detector F-statistic */

    /* we use the first detector Fa time series to obtain the number of time samples and the sampling time */
    /* these should be the same for all Fa and Fb timeseries */
    numSamples = multiFa_spin->data[0]->data->length;
    dt = multiFa_spin->data[0]->deltaT;

    /* allocate memory for individual-detector FFT outputs */
    COMPLEX8Vector *outaX, *outbX;
    XLAL_CHECK ( (outaX = XLALCreateCOMPLEX8Vector(numSamples)) != NULL, XLAL_EFUNC );
    XLAL_CHECK ( (outbX = XLALCreateCOMPLEX8Vector(numSamples)) != NULL, XLAL_EFUNC );

    /* make forwards FFT plan - this will be re-used for each detector */
    XLAL_CHECK ( (pfwd = XLALCreateCOMPLEX8FFTPlan ( numSamples, 1, 0) ) != NULL, XLAL_EFUNC );

    UINT4 numFreqBins = Fstats->numFreqBins;

    /* define new initial frequency of the frequency domain representations of Fa and Fb */
    /* before the shift the zero bin was the heterodyne frequency */
    /* now we've shifted it by N - NhalfPosDC(N) bins */
    REAL8 f0_shifted = multiFa_spin->data[0]->f0 - NhalfNeg(numSamples) * dFreq;
    /* define number of bins offset from the internal start frequency bin to the user requested bin */
    UINT4 offset_bins = (UINT4) lround ( ( thisPoint->fkdot[0] - f0_shifted ) / dFreq );

    COMPLEX8 *Fa_k, *Fb_k;
    XLAL_CHECK ( (Fa_k = XLALCalloc ( numFreqBins, sizeof(*Fa_k))) != NULL, XLAL_ENOMEM );
    XLAL_CHECK ( (Fb_k = XLALCalloc ( numFreqBins, sizeof(*Fa_k))) != NULL, XLAL_ENOMEM );

    /* loop over detectors */
    for ( UINT4 X=0; X < numDetectors; X++ )
      {
        COMPLEX8Vector *ina = multiFa_spin->data[X]->data; /* we point the input to the current detector Fa timeseries */
        COMPLEX8Vector *inb = multiFb_spin->data[X]->data; /* we point the input to the current detector Fb timeseries */

        /* Fourier transform the resampled Fa(t) and Fb(t) */
        XLAL_CHECK ( XLALCOMPLEX8VectorFFT ( outaX, ina, pfwd ) == XLAL_SUCCESS, XLAL_EFUNC );
        XLAL_CHECK ( XLALCOMPLEX8VectorFFT ( outbX, inb, pfwd ) == XLAL_SUCCESS, XLAL_EFUNC );

        /* the complex FFT output is shifted such that the heterodyne frequency is at DC */
        /* we need to shift the negative frequencies to before the positive ones */
        XLAL_CHECK ( XLALFFTShiftCOMPLEX8Vector ( &outaX ) == XLAL_SUCCESS, XLAL_EFUNC );
        XLAL_CHECK ( XLALFFTShiftCOMPLEX8Vector ( &outbX ) == XLAL_SUCCESS, XLAL_EFUNC );

        REAL4 AdX = cfBuffer->multiAMcoef->data[X]->A;
        REAL4 BdX = cfBuffer->multiAMcoef->data[X]->B;
        REAL4 CdX = cfBuffer->multiAMcoef->data[X]->C;
        REAL4 EdX = 0; // FIXME
        REAL4 DdX_inv = 1.0 / cfBuffer->multiAMcoef->data[X]->D;

        /* compute final Fa,Fb and Fstats (per-detector and combined) */
        for ( UINT4 k = 0; k < numFreqBins; k++ )
          {
            UINT4 idy = k + offset_bins;
            COMPLEX8 FaX_k = dt * outaX->data[idy];
            COMPLEX8 FbX_k = dt * outbX->data[idy];

            Fa_k[k] += FaX_k;
            Fb_k[k] += FbX_k;

            if ( whatToCompute & FSTATQ_FAFB_PER_DET )
              {
                Fstats->FaPerDet[X][k] = FaX_k;
                Fstats->FbPerDet[X][k] = FbX_k;
              }

            if ( whatToCompute & FSTATQ_2F_PER_DET )
              {
                Fstats->twoFPerDet[X][k] = XLALComputeFstatFromFaFb ( FaX_k, FbX_k, AdX, BdX, CdX, EdX, DdX_inv );
              }
          } // for k < numFreqBins

      } // for X < numDetectors

    if ( whatToCompute & FSTATQ_FAFB )
      {
        for ( UINT4 k=0; k < numFreqBins; k ++ )
          {
            Fstats->Fa[k] = Fa_k[k];
            Fstats->Fb[k] = Fb_k[k];
          } // for k < numFreqBins
      } // if FSTATQ_FAFB


    if ( whatToCompute & FSTATQ_2F )
      {
        for ( UINT4 k=0; k < numFreqBins; k++ )
          {
            Fstats->twoF[k] = XLALComputeFstatFromFaFb ( Fa_k[k], Fb_k[k], Ad, Bd, Cd, Ed, Dd_inv );
          } // for k < numFreqBins
      } // if FSTATQ_2F

    /* free memory not stored in the buffer */
    XLALFree ( Fa_k );
    XLALFree ( Fb_k );
    XLALDestroyCOMPLEX8Vector ( outaX );
    XLALDestroyCOMPLEX8Vector ( outbX );

    XLALDestroyCOMPLEX8FFTPlan ( pfwd );

    XLALDestroyMultiCOMPLEX8TimeSeries ( multiFa_spin );
    XLALDestroyMultiCOMPLEX8TimeSeries ( multiFb_spin );

  }// ================================================================================

  // Return F-atoms per detector
  if (whatToCompute & FSTATQ_ATOMS_PER_DET) {
    XLAL_ERROR(XLAL_EFAILED, "Unimplemented!");
  }

  Fstats->Mmunu = cfBuffer->multiAMcoef->Mmunu;

  return XLAL_SUCCESS;

} // ComputeFstat_Resamp()

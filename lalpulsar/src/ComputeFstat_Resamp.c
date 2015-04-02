//
// Copyright (C) 2014 Reinhard Prix
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

#include <complex.h>
#include <fftw3.h>
#include <lal/FFTWMutex.h>
#include <lal/Units.h>
#include <lal/Factorial.h>

// ========== Resamp internals ==========

// ----- local macros ----------
#define MYMAX(x,y) ( (x) > (y) ? (x) : (y) )
#define MYMIN(x,y) ( (x) < (y) ? (x) : (y) )

// local macro versions of library functions to avoid calling external functions in GPU-ready code
#define GPSDIFF(x,y) (1.0*((x).gpsSeconds - (y).gpsSeconds) + ((x).gpsNanoSeconds - (y).gpsNanoSeconds)*1e-9)
#define GPSGETREAL8(x) ( (x)->gpsSeconds + ( (x)->gpsNanoSeconds / XLAL_BILLION_REAL8 ) );
#define GPSSETREAL8(gps,r8) do {                                        \
    (gps).gpsSeconds     = (UINT4)floor(r8);                            \
    (gps).gpsNanoSeconds = (UINT4)round ( ((r8) - (gps).gpsSeconds) * XLAL_BILLION_REAL8 ); \
    if ( (gps).gpsNanoSeconds == XLAL_BILLION_INT4 ) {                  \
      (gps).gpsSeconds += 1;                                            \
      (gps).gpsNanoSeconds = 0;                                         \
    }                                                                   \
  } while(0)


// ----- local constants
#define COLLECT_TIMING 1

// ----- local types ----------
typedef struct tagMultiUINT4Vector
{
  UINT4 length;
  UINT4Vector **data;
} MultiUINT4Vector;

// ----- workspace ----------
typedef struct tagResampTimingInfo
{ // NOTE: all times refer to a single-detector timing case
  REAL8 tauTotal;		// total time spent in ComputeFstat_Resamp()
  REAL8 tauBary;		// time spent in barycentric resampling
  REAL8 tauSpin;		// time spent in spindown+frequency correction
  REAL8 tauFFT;			// time spent in FFT
  REAL8 tauNorm;		// time spent normalizing the final Fa,Fb
  REAL8 tauFab2F;		// time to compute Fstat from {Fa,Fb}
  REAL8 tauMem;			// time to realloc and memset-0 arrays
  REAL8 tauSumFabX;		// time to sum_X Fab^X
  REAL8 tauF1Buf;		// Resampling timing 'constant': Fstat time per template per detector for a 'buffered' case (same skypos, same numFreqBins)
  REAL8 tauF1NoBuf;		// Resampling timing 'constant': Fstat time per template per detector for an 'unbuffered' usage (different skypos and numFreqBins)
} ResampTimingInfo;

struct tagFstatWorkspace
{
  // intermediate quantities to interpolate and operate on SRC-frame timeseries
  COMPLEX8Vector *TStmp1_SRC;	// can hold a single-detector SRC-frame spindown-corrected timeseries [without zero-padding]
  COMPLEX8Vector *TStmp2_SRC;	// can hold a single-detector SRC-frame spindown-corrected timeseries [without zero-padding]
  REAL8Vector *SRCtimes_DET;	// holds uniformly-spaced SRC-frame timesteps translated into detector frame [for interpolation]

  // input padded timeseries ts(t) and output Fab(f) of length 'numSamplesFFT' and corresponding fftw plan
  UINT4 numSamplesFFT;		// allocated number of zero-padded SRC-frame time samples (related to dFreq)
  fftwf_plan fftplan;		// buffer FFT plan for given numSamplesOut length
  COMPLEX8 *TS_FFT;		// zero-padded, spindown-corr SRC-frame TS
  COMPLEX8 *FabX_Raw;		// raw full-band FFT result Fa,Fb

  // arrays of size numFreqBinsOut over frequency bins f_k:
  UINT4 numFreqBinsOut;		// number of output frequency bins {f_k}
  COMPLEX8 *FaX_k;		// properly normalized F_a^X(f_k) over output bins
  COMPLEX8 *FbX_k;		// properly normalized F_b^X(f_k) over output bins
  COMPLEX8 *Fa_k;		// properly normalized F_a(f_k) over output bins
  COMPLEX8 *Fb_k;		// properly normalized F_b(f_k) over output bins
  UINT4 numFreqBinsAlloc;	// internal: keep track of allocated length of frequency-arrays

  ResampTimingInfo timingInfo;	// temporary storage for collecting timing data
};

struct tagFstatInput_Resamp
{
  MultiCOMPLEX8TimeSeries  *multiTimeSeries_DET;	// input SFTs converted into a heterodyned timeseries
  // ----- buffering -----
  PulsarDopplerParams prev_doppler;			// buffering: previous phase-evolution ("doppler") parameters

  AntennaPatternMatrix Mmunu;				// combined multi-IFO antenna-pattern coefficients {A,B,C,E}
  AntennaPatternMatrix MmunuX[PULSAR_MAX_DETECTORS];	// per-IFO antenna-pattern coefficients {AX,BX,CX,EX}

  MultiCOMPLEX8TimeSeries *multiTimeSeries_SRC_a;	// multi-detector SRC-frame timeseries, multiplied by AM function a(t)
  MultiCOMPLEX8TimeSeries *multiTimeSeries_SRC_b;	// multi-detector SRC-frame timeseries, multiplied by AM function b(t)

  // ----- workspace -----
  FstatWorkspace *ws;					// 'workspace': pre-allocated vectors used to store intermediate results
  BOOLEAN ownThisWorkspace;				// flag whether we 'own' or share this workspace (ie who is responsible for freeing it)
};


// ----- local prototypes ----------
static int
XLALApplySpindownAndFreqShift ( COMPLEX8 *xOut,
                                const COMPLEX8TimeSeries *xIn,
                                const PulsarDopplerParams *doppler,
                                REAL8 freqShift
                                );

static int
XLALBarycentricResampleMultiCOMPLEX8TimeSeries ( FstatInput_Resamp *resamp,
                                                 const PulsarDopplerParams *thisPoint,
                                                 const FstatInput_Common *common
                                                 );

static int
XLALComputeFaFb_Resamp ( FstatWorkspace *ws,
                         const PulsarDopplerParams thisPoint,
                         REAL8 dFreq,
                         const COMPLEX8TimeSeries *TimeSeries_SRC_a,
                         const COMPLEX8TimeSeries *TimeSeries_SRC_b
                         );

static FstatWorkspace *XLALCreateFstatWorkspace ( UINT4 numSamplesSRC, UINT4 numSamplesFFT );

// pseudo-internal: don't export API but allow using them from test/benchmark codes
int XLALAppendResampInfo2File ( FILE *fp, const FstatInput *input );
int XLALGetResampTimingInfo ( REAL8 *tauF1NoBuf, REAL8 *tauF1Buf, const FstatInput *input );

// ==================== function definitions ====================

// ---------- exported API functions ----------
///
/// Create a new workspace with given time samples in SRC frame 'numSamplesSRC' (holds time-series for spindown-correction)
/// and given total number of time-samples for FFTing (includes zero-padding for frequency-resolution)
///
static FstatWorkspace *
XLALCreateFstatWorkspace ( UINT4 numSamplesSRC,
                           UINT4 numSamplesFFT
                           )
{
  FstatWorkspace *ws;

  XLAL_CHECK_NULL ( (ws = XLALCalloc ( 1, sizeof(*ws))) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL ( (ws->TStmp1_SRC   = XLALCreateCOMPLEX8Vector ( numSamplesSRC )) != NULL, XLAL_EFUNC );
  XLAL_CHECK_NULL ( (ws->TStmp2_SRC   = XLALCreateCOMPLEX8Vector ( numSamplesSRC )) != NULL, XLAL_EFUNC );
  XLAL_CHECK_NULL ( (ws->SRCtimes_DET = XLALCreateREAL8Vector ( numSamplesSRC )) != NULL, XLAL_EFUNC );

  XLAL_CHECK_NULL ( (ws->FabX_Raw = fftw_malloc ( numSamplesFFT * sizeof(COMPLEX8) )) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL ( (ws->TS_FFT   = fftw_malloc ( numSamplesFFT * sizeof(COMPLEX8) )) != NULL, XLAL_ENOMEM );

  LAL_FFTW_WISDOM_LOCK;
  XLAL_CHECK_NULL ( (ws->fftplan = fftwf_plan_dft_1d ( numSamplesFFT, ws->TS_FFT, ws->FabX_Raw, FFTW_FORWARD, FFTW_MEASURE )) != NULL, XLAL_EFAILED, "fftwf_plan_dft_1d() failed\n");
  LAL_FFTW_WISDOM_UNLOCK;
  ws->numSamplesFFT = numSamplesFFT;

  return ws;
} // XLALCreateFstatWorkspace()

///
/// Function to extract a workspace from a resampling setup, which can be passed in FstatOptionalArgs to be shared by various setups
/// in order to save memory. Note, when using this, you need to free this workspace yourself at the end using XLALDestroyFstatWorkspace().
/// Note: Demod methods don't use a workspace, so NULL (without error) is returned in this case.
///
FstatWorkspace *
XLALGetSharedFstatWorkspace ( FstatInput *input		//!< [in,out] Fstat input structure to extract shared workspace from
                              )
{
  XLAL_CHECK_NULL ( input != NULL, XLAL_EINVAL );

  if ( input->resamp == NULL ) {
    return NULL;
  }

  input->resamp->ownThisWorkspace = 0;	// the caller now owns the workspace and has to free it
  return input->resamp->ws;

} // XLALGetSharedFstatWorkspace()


void
XLALDestroyFstatWorkspace ( FstatWorkspace *ws )
{
  if ( ws == NULL ) {
    return;
  }

  XLALDestroyCOMPLEX8Vector ( ws->TStmp1_SRC );
  XLALDestroyCOMPLEX8Vector ( ws->TStmp2_SRC );
  XLALDestroyREAL8Vector ( ws->SRCtimes_DET );

  LAL_FFTW_WISDOM_LOCK;
  fftwf_destroy_plan ( ws->fftplan );
  LAL_FFTW_WISDOM_UNLOCK;

  fftw_free ( ws->FabX_Raw );
  fftw_free ( ws->TS_FFT );

  XLALFree ( ws->FaX_k );
  XLALFree ( ws->FbX_k );
  XLALFree ( ws->Fa_k );
  XLALFree ( ws->Fb_k );

  XLALFree ( ws );
  return;

} // XLALDestroyFstatWorkspace()

// return resampling timing coefficients 'tauF1' for buffered and unbuffered calls
int
XLALGetResampTimingInfo ( REAL8 *tauF1NoBuf, REAL8 *tauF1Buf, const FstatInput *input )
{
  XLAL_CHECK ( input != NULL, XLAL_EINVAL );
  XLAL_CHECK ( input->resamp != NULL, XLAL_EINVAL );
  XLAL_CHECK ( input->resamp->ws != NULL, XLAL_EINVAL );
  const ResampTimingInfo *ti = &(input->resamp->ws->timingInfo);

  (*tauF1NoBuf) = ti->tauF1NoBuf;
  (*tauF1Buf) = ti->tauF1Buf;

  return XLAL_SUCCESS;

} // XLALGetResampTimingInfo()

/// debug/optimizer helper function: dump internal info from resampling code into a file
/// if called with input==NULL, output a header-comment-line
int
XLALAppendResampInfo2File ( FILE *fp, const FstatInput *input )
{
  XLAL_CHECK ( fp != NULL, XLAL_EINVAL );

  if ( input == NULL ) {
    fprintf (fp, "%%%%%8s %10s %6s %10s %10s ",
             "Nfreq", "NsFFT", "Nsft0", "Ns_DET0", "Ns_SRC0" );
    fprintf (fp, "%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
             "tauTotal", "tauFFT", "tauBary", "tauSpin", "tauAM", "tauNorm", "tauFab2F", "tauMem", "tauSumFabX", "tauF1NoBuf", "tauF1Buf" );
    return XLAL_SUCCESS;
  }
  XLAL_CHECK ( input->resamp != NULL, XLAL_EINVAL );
  const FstatInput_Resamp *resamp = input->resamp;
  const FstatInput_Common *common = input->common;
  const FstatWorkspace *ws = resamp->ws;

  fprintf (fp, "%10d %10d", ws->numFreqBinsOut, ws->numSamplesFFT );
  UINT4 numSamples_DETX0 = resamp->multiTimeSeries_DET->data[0]->data->length;
  UINT4 numSFTs_X0 = common->multiTimestamps->data[0]->length;
  COMPLEX8TimeSeries *ts_SRCX0 = resamp->multiTimeSeries_SRC_a->data[0];
  UINT4 numSamples_SRCX0 = ts_SRCX0->data->length;
  fprintf (fp, " %6d %10d %10d ", numSFTs_X0, numSamples_DETX0, numSamples_SRCX0 );

  const ResampTimingInfo *ti = &(ws->timingInfo);
  fprintf (fp, "%10.1e %10.1e %10.1e %10.1e %10.1e %10.1e %10.1e %10.1e %10.1e %10.1e %10.1e\n",
           ti->tauTotal, ti->tauFFT, ti->tauBary, ti->tauSpin, 0.0, ti->tauNorm, ti->tauFab2F, ti->tauMem, ti->tauSumFabX, ti->tauF1NoBuf, ti->tauF1Buf );

  return XLAL_SUCCESS;

} // XLALAppendResampInfo2File()

// ---------- internal functions ----------
static void
DestroyFstatInput_Resamp ( FstatInput_Resamp* resamp )
{
  XLALDestroyMultiCOMPLEX8TimeSeries (resamp->multiTimeSeries_DET );

  // ----- free buffer
  XLALDestroyMultiCOMPLEX8TimeSeries ( resamp->multiTimeSeries_SRC_a );
  XLALDestroyMultiCOMPLEX8TimeSeries ( resamp->multiTimeSeries_SRC_b );

  // ----- free workspace
  if ( resamp->ownThisWorkspace ) {
    XLALDestroyFstatWorkspace ( resamp->ws );
  }

  XLALFree ( resamp );

  return;

} // DestroyFstatInput_Resamp()

static int
SetupFstatInput_Resamp ( FstatInput_Resamp *resamp,
                         const FstatInput_Common *common,
                         MultiSFTVector *multiSFTs,
                         FstatWorkspace *sharedWorkspace
                         )
{
  // Check input
  XLAL_CHECK(common != NULL, XLAL_EFAULT);
  XLAL_CHECK(resamp != NULL, XLAL_EFAULT);
  XLAL_CHECK(multiSFTs != NULL, XLAL_EFAULT);

  // Convert SFTs into heterodyned complex timeseries [in detector frame]
  XLAL_CHECK ( (resamp->multiTimeSeries_DET = XLALMultiSFTVectorToCOMPLEX8TimeSeries ( multiSFTs )) != NULL, XLAL_EFUNC );

  XLALDestroyMultiSFTVector ( multiSFTs );	// don't need them SFTs any more ...

  UINT4 numDetectors = resamp->multiTimeSeries_DET->length;
  REAL8 dt_DET       = resamp->multiTimeSeries_DET->data[0]->deltaT;
  REAL8 fHet         = resamp->multiTimeSeries_DET->data[0]->f0;
  REAL8 Tsft         = common->multiTimestamps->data[0]->deltaT;

  // determine resampled timeseries parameters
  REAL8 TspanFFT = 1.0 / common->dFreq;
  UINT4 numSamplesFFT = (UINT4) ceil ( TspanFFT / dt_DET );      // we use ceil() so that we artificially widen the band rather than reduce it
  // round numSamplesFFT to next power of 2
  numSamplesFFT = (UINT4) pow ( 2, ceil(log2(numSamplesFFT)));
  REAL8 dt_SRC = TspanFFT / numSamplesFFT;			// adjust sampling rate to allow achieving exact requested dFreq=1/TspanFFT !

  // ----- allocate buffer memory ----------

  // header for SRC-frame resampled timeseries buffer
  XLAL_CHECK ( (resamp->multiTimeSeries_SRC_a = XLALCalloc ( 1, sizeof(MultiCOMPLEX8TimeSeries)) ) != NULL, XLAL_ENOMEM );
  XLAL_CHECK ( (resamp->multiTimeSeries_SRC_a->data = XLALCalloc ( numDetectors, sizeof(COMPLEX8TimeSeries) )) != NULL, XLAL_ENOMEM );
  resamp->multiTimeSeries_SRC_a->length = numDetectors;

  XLAL_CHECK ( (resamp->multiTimeSeries_SRC_b = XLALCalloc ( 1, sizeof(MultiCOMPLEX8TimeSeries)) ) != NULL, XLAL_ENOMEM );
  XLAL_CHECK ( (resamp->multiTimeSeries_SRC_b->data = XLALCalloc ( numDetectors, sizeof(COMPLEX8TimeSeries) )) != NULL, XLAL_ENOMEM );
  resamp->multiTimeSeries_SRC_b->length = numDetectors;

  LIGOTimeGPS XLAL_INIT_DECL(epoch0);	// will be set to corresponding SRC-frame epoch when barycentering
  UINT4 numSamplesMax_SRC = 0;
  for ( UINT4 X = 0; X < numDetectors; X ++ )
    {
      // ----- check input consistency ----------
      REAL8 dt_DETX = resamp->multiTimeSeries_DET->data[X]->deltaT;
      XLAL_CHECK ( dt_DET == dt_DETX, XLAL_EINVAL, "Input timeseries must have identical 'deltaT(X=%d)' (%.16g != %.16g)\n", X, dt_DET, dt_DETX);

      REAL8 fHetX = resamp->multiTimeSeries_DET->data[X]->f0;
      XLAL_CHECK ( fHet == fHetX, XLAL_EINVAL, "Input timeseries must have identical heterodyning frequency 'f0(X=%d)' (%.16g != %.16g)\n", X, fHet, fHetX );

      REAL8 TsftX = common->multiTimestamps->data[X]->deltaT;
      XLAL_CHECK ( Tsft == TsftX, XLAL_EINVAL, "Input timestamps must have identical stepsize 'Tsft(X=%d)' (%.16g != %.16g)\n", X, Tsft, TsftX );

      // ----- prepare memory fo SRC-frame timeseries and AM coefficients
      const char *nameX = resamp->multiTimeSeries_DET->data[X]->name;
      UINT4 numSamples_DETX = resamp->multiTimeSeries_DET->data[X]->data->length;
      UINT4 numSamples_SRCX = (UINT4)ceil ( numSamples_DETX * dt_DET / dt_SRC );

      XLAL_CHECK ( (resamp->multiTimeSeries_SRC_a->data[X] = XLALCreateCOMPLEX8TimeSeries ( nameX, &epoch0, fHet, dt_SRC, &lalDimensionlessUnit, numSamples_SRCX )) != NULL, XLAL_EFUNC );
      XLAL_CHECK ( (resamp->multiTimeSeries_SRC_b->data[X] = XLALCreateCOMPLEX8TimeSeries ( nameX, &epoch0, fHet, dt_SRC, &lalDimensionlessUnit, numSamples_SRCX )) != NULL, XLAL_EFUNC );

      numSamplesMax_SRC = MYMAX ( numSamplesMax_SRC, numSamples_SRCX );
    } // for X < numDetectors

  // ---- re-use shared workspace, or allocate here ----------
  if ( sharedWorkspace != NULL )
    {
      XLAL_CHECK ( numSamplesFFT == sharedWorkspace->numSamplesFFT, XLAL_EINVAL, "Shared workspace of different frequency resolution: numSamplesFFT = %d != %d\n",
                   sharedWorkspace->numSamplesFFT, numSamplesFFT );

      // adjust maximal SRC-frame timeseries length, if necessary
      if ( numSamplesMax_SRC > sharedWorkspace->TStmp1_SRC->length ) {
        XLAL_CHECK ( (sharedWorkspace->TStmp1_SRC->data = XLALRealloc ( sharedWorkspace->TStmp1_SRC->data,   numSamplesMax_SRC * sizeof(COMPLEX8) )) != NULL, XLAL_ENOMEM );
        sharedWorkspace->TStmp1_SRC->length = numSamplesMax_SRC;
        XLAL_CHECK ( (sharedWorkspace->TStmp2_SRC->data = XLALRealloc ( sharedWorkspace->TStmp2_SRC->data,   numSamplesMax_SRC * sizeof(COMPLEX8) )) != NULL, XLAL_ENOMEM );
        sharedWorkspace->TStmp2_SRC->length = numSamplesMax_SRC;
        XLAL_CHECK ( (sharedWorkspace->SRCtimes_DET->data = XLALRealloc ( sharedWorkspace->SRCtimes_DET->data, numSamplesMax_SRC * sizeof(REAL8) )) != NULL, XLAL_ENOMEM );
        sharedWorkspace->SRCtimes_DET->length = numSamplesMax_SRC;
      }
      resamp->ws = sharedWorkspace;
      resamp->ownThisWorkspace = 0;
    } // end: if shared workspace given
  else
    {
      XLAL_CHECK ( ( resamp->ws = XLALCreateFstatWorkspace ( numSamplesMax_SRC, numSamplesFFT )) != NULL, XLAL_EFUNC );
      resamp->ownThisWorkspace = 1;
    } // end: if we create our own workspace

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

  const FstatQuantities whatToCompute = Fstats->whatWasComputed;
  XLAL_CHECK ( !(whatToCompute & FSTATQ_ATOMS_PER_DET), XLAL_EINVAL, "Resampling does not currently support atoms per detector" );

#ifdef COLLECT_TIMING
  // collect internal timing info
  XLAL_INIT_MEM ( resamp->ws->timingInfo );
  ResampTimingInfo *ti = &(resamp->ws->timingInfo);
  REAL8 ticStart,tocEnd;
  ticStart = XLALGetCPUTime();
  REAL8 tic,toc;
#endif

  // ----- handy shortcuts ----------
  PulsarDopplerParams thisPoint = Fstats->doppler;
  const MultiCOMPLEX8TimeSeries *multiTimeSeries_DET = resamp->multiTimeSeries_DET;
  UINT4 numDetectors = multiTimeSeries_DET->length;

  // ============================== BEGIN: handle buffering =============================
  BOOLEAN same_skypos = (resamp->prev_doppler.Alpha == thisPoint.Alpha) && (resamp->prev_doppler.Delta == thisPoint.Delta);
  BOOLEAN same_refTime = ( GPSDIFF ( resamp->prev_doppler.refTime, thisPoint.refTime ) == 0 );
  BOOLEAN same_binary = \
    (resamp->prev_doppler.asini == thisPoint.asini) &&
    (resamp->prev_doppler.period == thisPoint.period) &&
    (resamp->prev_doppler.ecc == thisPoint.ecc) &&
    (GPSDIFF( resamp->prev_doppler.tp, thisPoint.tp ) == 0 ) &&
    (resamp->prev_doppler.argp == thisPoint.argp);

  FstatWorkspace *ws = resamp->ws;

  // ----- not same skypos+binary+refTime? --> re-compute SRC-frame timeseries, AM-coeffs and store in buffer
#ifdef COLLECT_TIMING
  tic = XLALGetCPUTime();
#endif
  if ( ! ( same_skypos && same_refTime && same_binary) )
    {
      XLAL_CHECK ( XLALBarycentricResampleMultiCOMPLEX8TimeSeries ( resamp, &thisPoint, common ) == XLAL_SUCCESS, XLAL_EFUNC );
    }
#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauBary = (toc-tic);
#endif
  MultiCOMPLEX8TimeSeries *multiTimeSeries_SRC_a = resamp->multiTimeSeries_SRC_a;
  MultiCOMPLEX8TimeSeries *multiTimeSeries_SRC_b = resamp->multiTimeSeries_SRC_b;

  // ============================== check workspace is properly allocated and initialized ===========

  // ----- workspace that depends on number of output frequency bins 'numFreqBins' ----------
  UINT4 numFreqBins = Fstats->numFreqBins;

#ifdef COLLECT_TIMING
  tic = XLALGetCPUTime();
#endif

  // NOTE: we try to use as much existing memory as possible in FstatResults, so we only
  // allocate local 'workspace' storage in case there's not already a vector allocated in FstatResults for it
  // this also avoid having to copy these results in case the user asked for them to be returned
  if ( whatToCompute & FSTATQ_FAFB )
    {
      XLALFree ( ws->Fa_k ); // avoid memory leak if allocated in previous call
      ws->Fa_k = Fstats->Fa;
      XLALFree ( ws->Fb_k ); // avoid memory leak if allocated in previous call
      ws->Fb_k = Fstats->Fb;
    } // end: if returning FaFb we can use that return-struct as 'workspace'
  else	// otherwise: we (re)allocate it locally
    {
      if ( numFreqBins > ws->numFreqBinsAlloc )
        {
          XLAL_CHECK ( (ws->Fa_k = XLALRealloc ( ws->Fa_k, numFreqBins * sizeof(COMPLEX8))) != NULL, XLAL_ENOMEM );
          XLAL_CHECK ( (ws->Fb_k = XLALRealloc ( ws->Fb_k, numFreqBins * sizeof(COMPLEX8))) != NULL, XLAL_ENOMEM );
        } // only increase workspace arrays
    }

  if ( whatToCompute & FSTATQ_FAFB_PER_DET )
    {
      XLALFree ( ws->FaX_k ); // avoid memory leak if allocated in previous call
      ws->FaX_k = NULL;	// will be set in loop over detectors X
      XLALFree ( ws->FbX_k ); // avoid memory leak if allocated in previous call
      ws->FbX_k = NULL;	// will be set in loop over detectors X
    } // end: if returning FaFbPerDet we can use that return-struct as 'workspace'
  else	// otherwise: we (re)allocate it locally
    {
      if ( numFreqBins > ws->numFreqBinsAlloc )
        {
          XLAL_CHECK ( (ws->FaX_k = XLALRealloc ( ws->FaX_k, numFreqBins * sizeof(COMPLEX8))) != NULL, XLAL_ENOMEM );
          XLAL_CHECK ( (ws->FbX_k = XLALRealloc ( ws->FbX_k, numFreqBins * sizeof(COMPLEX8))) != NULL, XLAL_ENOMEM );
        } // only increase workspace arrays
    }
  if ( numFreqBins > ws->numFreqBinsAlloc ) {
    ws->numFreqBinsAlloc = numFreqBins;	// keep track of allocated array length
  }
  ws->numFreqBinsOut = numFreqBins;
  // ====================================================================================================

#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauMem = (toc-tic);	// this one doesn't scale with number of detector!
#endif

  // loop over detectors
  for ( UINT4 X=0; X < numDetectors; X++ )
    {
      // if return-struct contains memory for holding FaFbPerDet: use that directly instead of local memory
      if ( whatToCompute & FSTATQ_FAFB_PER_DET )
        {
          ws->FaX_k = Fstats->FaPerDet[X];
          ws->FbX_k = Fstats->FbPerDet[X];
        }
      const COMPLEX8TimeSeries *TimeSeriesX_SRC_a = multiTimeSeries_SRC_a->data[X];
      const COMPLEX8TimeSeries *TimeSeriesX_SRC_b = multiTimeSeries_SRC_b->data[X];

      // compute {Fa^X(f_k), Fb^X(f_k)}: results returned via workspace resamp->ws
      XLAL_CHECK ( XLALComputeFaFb_Resamp ( resamp->ws, thisPoint, common->dFreq, TimeSeriesX_SRC_a, TimeSeriesX_SRC_b ) == XLAL_SUCCESS, XLAL_EFUNC );

#ifdef COLLECT_TIMING
      tic = XLALGetCPUTime();
#endif
      if ( X == 0 )
        { // avoid having to memset this array: for the first detector we *copy* results
          for ( UINT4 k = 0; k < numFreqBins; k++ )
            {
              ws->Fa_k[k] = ws->FaX_k[k];
              ws->Fb_k[k] = ws->FbX_k[k];
            }
        } // end: if X==0
      else
        { // for subsequent detectors we *add to* them
          for ( UINT4 k = 0; k < numFreqBins; k++ )
            {
              ws->Fa_k[k] += ws->FaX_k[k];
              ws->Fb_k[k] += ws->FbX_k[k];
            }
        } // end:if X>0
#ifdef COLLECT_TIMING
      toc = XLALGetCPUTime();
      ti->tauSumFabX += (toc-tic);
      tic = toc;
#endif
      // ----- if requested: compute per-detector Fstat_X_k
      if ( whatToCompute & FSTATQ_2F_PER_DET )
        {
          const REAL4 AdX = resamp->MmunuX[X].Ad;
          const REAL4 BdX = resamp->MmunuX[X].Bd;
          const REAL4 CdX = resamp->MmunuX[X].Cd;
          const REAL4 EdX = resamp->MmunuX[X].Ed;
          const REAL4 DdX_inv = 1.0f / resamp->MmunuX[X].Dd;
          for ( UINT4 k = 0; k < numFreqBins; k ++ )
            {
              Fstats->twoFPerDet[X][k] = XLALComputeFstatFromFaFb ( ws->FaX_k[k], ws->FbX_k[k], AdX, BdX, CdX, EdX, DdX_inv );
            }  // for k < numFreqBins
        } // end: if compute F_X
#ifdef COLLECT_TIMING
      toc = XLALGetCPUTime();
      ti->tauFab2F += ( toc - tic );
#endif

    } // for X < numDetectors

#ifdef COLLECT_TIMING
  ti->tauSumFabX /= numDetectors;
  ti->tauFab2F /= numDetectors;
  tic = XLALGetCPUTime();
#endif
  if ( whatToCompute & FSTATQ_2F )
    {
      const REAL4 Ad = resamp->Mmunu.Ad;
      const REAL4 Bd = resamp->Mmunu.Bd;
      const REAL4 Cd = resamp->Mmunu.Cd;
      const REAL4 Ed = resamp->Mmunu.Ed;
      const REAL4 Dd_inv = 1.0f / resamp->Mmunu.Dd;
      for ( UINT4 k=0; k < numFreqBins; k++ )
        {
          Fstats->twoF[k] = XLALComputeFstatFromFaFb ( ws->Fa_k[k], ws->Fb_k[k], Ad, Bd, Cd, Ed, Dd_inv );
        }
    } // if FSTATQ_2F
#ifdef COLLECT_TIMING
      toc = XLALGetCPUTime();
      ti->tauFab2F += ( toc - tic );
#endif

  // Return F-atoms per detector
  if (whatToCompute & FSTATQ_ATOMS_PER_DET) {
    XLAL_ERROR(XLAL_EFAILED, "NOT implemented!");
  }

  Fstats->Mmunu = resamp->Mmunu;

  // ----- workspace memory management:
  // if we used the return struct directly to store Fa,Fb results,
  // make sure to wipe those pointers to avoid mistakenly considering them as 'local' memory
  // and re-allocing it in another call to this function
  if ( whatToCompute & FSTATQ_FAFB )
    {
      ws->Fa_k = NULL;
      ws->Fb_k = NULL;
    }
  if ( whatToCompute & FSTATQ_FAFB_PER_DET )
    {
      ws->FaX_k = NULL;
      ws->FbX_k = NULL;
    }

#ifdef COLLECT_TIMING
  // timings are per-detector
  tocEnd = XLALGetCPUTime();
  ti->tauTotal = (tocEnd - ticStart);
  // rescale all relevant timings to single-IFO case
  ti->tauTotal /= numDetectors;
  ti->tauBary  /= numDetectors;
  ti->tauSpin  /= numDetectors;
  ti->tauFFT   /= numDetectors;
  ti->tauNorm  /= numDetectors;

  // compute 'fundamental' timing numbers per template per detector
  ti->tauF1NoBuf = ti->tauTotal / numFreqBins;
  ti->tauF1Buf   = (ti->tauTotal - ti->tauBary - ti->tauMem) / numFreqBins;
#endif

  return XLAL_SUCCESS;

} // ComputeFstat_Resamp()


static int
XLALComputeFaFb_Resamp ( FstatWorkspace *restrict ws,				//!< [in,out] pre-allocated 'workspace' for temporary and output quantities
                         const PulsarDopplerParams thisPoint,			//!< [in] Doppler point to compute {FaX,FbX} for
                         REAL8 dFreq,						//!< [in] output frequency resolution
                         const COMPLEX8TimeSeries * restrict TimeSeries_SRC_a,	//!< [in] SRC-frame single-IFO timeseries * a(t)
                         const COMPLEX8TimeSeries * restrict TimeSeries_SRC_b	//!< [in] SRC-frame single-IFO timeseries * b(t)
                         )
{
  XLAL_CHECK ( (ws != NULL) && (TimeSeries_SRC_a != NULL) && (TimeSeries_SRC_b != NULL), XLAL_EINVAL );
  XLAL_CHECK ( dFreq > 0, XLAL_EINVAL );

  REAL8 FreqOut0 = thisPoint.fkdot[0];

  // compute frequency shift to align heterodyne frequency with output frequency bins
  REAL8 fHet   = TimeSeries_SRC_a->f0;
  REAL8 dt_SRC = TimeSeries_SRC_a->deltaT;

  REAL8 freqShift = remainder ( FreqOut0 - fHet, dFreq ); // frequency shift to closest bin
  REAL8 fMinFFT = fHet + freqShift - dFreq * (ws->numSamplesFFT/2);	// we'll shift DC into the *middle bin* N/2  [N always even!]
  UINT4 offset_bins = (UINT4) lround ( ( FreqOut0 - fMinFFT ) / dFreq );

#ifdef COLLECT_TIMING
  // collect some internal timing info
  ResampTimingInfo *ti = &(ws->timingInfo);
  REAL8 tic,toc;
  tic = XLALGetCPUTime();
#endif

  memset ( ws->TS_FFT, 0, ws->numSamplesFFT * sizeof(ws->TS_FFT[0]) );
  // ----- compute FaX_k
  // apply spindown phase-factors, store result in zero-padded timeseries for 'FFT'ing
  XLAL_CHECK ( XLALApplySpindownAndFreqShift ( ws->TS_FFT, TimeSeries_SRC_a, &thisPoint, freqShift ) == XLAL_SUCCESS, XLAL_EFUNC );

#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauSpin += ( toc - tic);
  tic = toc;
#endif

  // Fourier transform the resampled Fa(t)
  fftwf_execute ( ws->fftplan );

  for ( UINT4 k = 0; k < ws->numFreqBinsOut; k++ ) {
    ws->FaX_k[k] = ws->FabX_Raw [ offset_bins + k ];
  }

#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauFFT += ( toc - tic);
  tic = toc;
#endif

  // ----- compute FbX_k
  // apply spindown phase-factors, store result in zero-padded timeseries for 'FFT'ing
  XLAL_CHECK ( XLALApplySpindownAndFreqShift ( ws->TS_FFT, TimeSeries_SRC_b, &thisPoint, freqShift ) == XLAL_SUCCESS, XLAL_EFUNC );

#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauSpin += ( toc - tic);
  tic = toc;
#endif

  // Fourier transform the resampled Fa(t)
  fftwf_execute ( ws->fftplan );

  for ( UINT4 k = 0; k < ws->numFreqBinsOut; k++ ) {
    ws->FbX_k[k] = ws->FabX_Raw [ offset_bins + k ];
  }

#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauFFT += ( toc - tic);
  tic = toc;
#endif

  // ----- normalization factors to be applied to Fa and Fb:
  const REAL8 dtauX = GPSDIFF ( TimeSeries_SRC_a->epoch, thisPoint.refTime );
  for ( UINT4 k = 0; k < ws->numFreqBinsOut; k++ )
    {
      REAL8 f_k = FreqOut0 + k * dFreq;
      REAL8 cycles = - f_k * dtauX;
      REAL4 sinphase, cosphase;
      XLALSinCos2PiLUT ( &sinphase, &cosphase, cycles );
      COMPLEX8 normX_k = dt_SRC * crectf ( cosphase, sinphase );
      ws->FaX_k[k] *= normX_k;
      ws->FbX_k[k] *= normX_k;
    } // for k < numFreqBinsOut

#ifdef COLLECT_TIMING
  toc = XLALGetCPUTime();
  ti->tauNorm += ( toc - tic);
  tic = toc;
#endif

  return XLAL_SUCCESS;

} // XLALComputeFaFb_Resamp()

static int
XLALApplySpindownAndFreqShift ( COMPLEX8 *restrict xOut,      			///< [out] the spindown-corrected SRC-frame timeseries
                                const COMPLEX8TimeSeries *restrict xIn,		///< [in] the input SRC-frame timeseries
                                const PulsarDopplerParams *restrict doppler,	///< [in] containing spindown parameters
                                REAL8 freqShift					///< [in] frequency-shift to apply, sign is "new - old"
                                )
{
  // input sanity checks
  XLAL_CHECK ( xOut != NULL, XLAL_EINVAL );
  XLAL_CHECK ( xIn != NULL, XLAL_EINVAL );
  XLAL_CHECK ( doppler != NULL, XLAL_EINVAL );

  // determine number of spin downs to include
  UINT4 s_max = PULSAR_MAX_SPINS - 1;
  while ( (s_max > 0) && (doppler->fkdot[s_max] == 0) ) {
    s_max --;
  }

  REAL8 dt = xIn->deltaT;
  UINT4 numSamplesIn  = xIn->data->length;

  LIGOTimeGPS epoch = xIn->epoch;
  REAL8 Dtau0 = GPSDIFF ( epoch, doppler->refTime );

  // loop over time samples
  for ( UINT4 j = 0; j < numSamplesIn; j ++ )
    {
      REAL8 taup_j = j * dt;
      REAL8 Dtau_alpha_j = Dtau0 + taup_j;

      REAL8 cycles = - freqShift * taup_j;

      REAL8 Dtau_pow_kp1 = Dtau_alpha_j;
      for ( UINT4 k = 1; k <= s_max; k++ )
        {
          Dtau_pow_kp1 *= Dtau_alpha_j;
          cycles += - LAL_FACT_INV[k+1] * doppler->fkdot[k] * Dtau_pow_kp1;
        } // for k = 1 ... s_max

      REAL4 cosphase, sinphase;
      XLAL_CHECK( XLALSinCos2PiLUT ( &sinphase, &cosphase, cycles ) == XLAL_SUCCESS, XLAL_EFUNC );
      COMPLEX8 em2piphase = crectf ( cosphase, sinphase );

      // weight the complex timeseries by the antenna patterns
      xOut[j] = em2piphase * xIn->data->data[j];

    } // for j < numSamplesIn

  return XLAL_SUCCESS;

} // XLALApplySpindownAndFreqShift()

///
/// Performs barycentric resampling on a multi-detector timeseries, updates resampling buffer with results
///
/// NOTE: this function does NOT check whether the previously-buffered solution can be reused, it assumes the
/// caller has already done so, and simply computes the requested resampled time-series, and AM-coefficients
///
static int
XLALBarycentricResampleMultiCOMPLEX8TimeSeries ( FstatInput_Resamp *resamp,		// [in/out] resampling input and buffer (to store resampling TS)
                                                 const PulsarDopplerParams *thisPoint,	// [in] current skypoint and reftime
                                                 const FstatInput_Common *common	// [in] various input quantities and parameters used here
                                                 )
{
  // check input sanity
  XLAL_CHECK ( thisPoint != NULL, XLAL_EINVAL );
  XLAL_CHECK ( common != NULL, XLAL_EINVAL );
  XLAL_CHECK ( resamp != NULL, XLAL_EINVAL );
  XLAL_CHECK ( resamp->multiTimeSeries_DET != NULL, XLAL_EINVAL );
  XLAL_CHECK ( resamp->multiTimeSeries_SRC_a != NULL, XLAL_EINVAL );
  XLAL_CHECK ( resamp->multiTimeSeries_SRC_b != NULL, XLAL_EINVAL );

  UINT4 numDetectors = resamp->multiTimeSeries_DET->length;
  XLAL_CHECK ( resamp->multiTimeSeries_SRC_a->length == numDetectors, XLAL_EINVAL, "Inconsistent number of detectors tsDET(%d) != tsSRC(%d)\n", numDetectors, resamp->multiTimeSeries_SRC_a->length );
  XLAL_CHECK ( resamp->multiTimeSeries_SRC_b->length == numDetectors, XLAL_EINVAL, "Inconsistent number of detectors tsDET(%d) != tsSRC(%d)\n", numDetectors, resamp->multiTimeSeries_SRC_b->length );

  SkyPosition skypos;
  skypos.system = COORDINATESYSTEM_EQUATORIAL;
  skypos.longitude = thisPoint->Alpha;
  skypos.latitude  = thisPoint->Delta;

  MultiAMCoeffs *multiAMcoef;
  XLAL_CHECK ( (multiAMcoef = XLALComputeMultiAMCoeffs ( common->multiDetectorStates, common->multiNoiseWeights, skypos )) != NULL, XLAL_EFUNC );
  resamp->Mmunu = multiAMcoef->Mmunu;
  for ( UINT4 X = 0; X < numDetectors; X ++ )
    {
      resamp->MmunuX[X].Ad = multiAMcoef->data[X]->A;
      resamp->MmunuX[X].Bd = multiAMcoef->data[X]->B;
      resamp->MmunuX[X].Cd = multiAMcoef->data[X]->C;
      resamp->MmunuX[X].Ed = 0;
      resamp->MmunuX[X].Dd = multiAMcoef->data[X]->D;
    }

  MultiSSBtimes *multiSRCtimes;
  XLAL_CHECK ( (multiSRCtimes = XLALGetMultiSSBtimes ( common->multiDetectorStates, skypos, thisPoint->refTime, common->SSBprec )) != NULL, XLAL_EFUNC );
  if ( thisPoint->asini > 0 ) {
    XLAL_CHECK ( XLALAddMultiBinaryTimes ( &multiSRCtimes, multiSRCtimes, thisPoint ) == XLAL_SUCCESS, XLAL_EFUNC );
  }

  // shorthands
  REAL8 fHet = resamp->multiTimeSeries_DET->data[0]->f0;
  REAL8 Tsft = common->multiTimestamps->data[0]->deltaT;
  REAL8 dt_SRC = resamp->multiTimeSeries_SRC_a->data[0]->deltaT;

  const REAL4 signumLUT[2] = {1, -1};

  // loop over detectors X
  for ( UINT4 X = 0; X < numDetectors; X++)
    {
      // shorthand pointers: input
      const COMPLEX8TimeSeries *TimeSeries_DETX = resamp->multiTimeSeries_DET->data[X];
      const LIGOTimeGPSVector  *Timestamps_DETX = common->multiTimestamps->data[X];
      const SSBtimes *SRCtimesX                 = multiSRCtimes->data[X];
      const AMCoeffs *AMcoefX			= multiAMcoef->data[X];

      // shorthand pointers: output
      COMPLEX8TimeSeries *TimeSeries_SRCX_a     = resamp->multiTimeSeries_SRC_a->data[X];
      COMPLEX8TimeSeries *TimeSeries_SRCX_b     = resamp->multiTimeSeries_SRC_b->data[X];
      REAL8Vector *ti_DET = resamp->ws->SRCtimes_DET;

      // useful shorthands
      REAL8 refTime8        = GPSGETREAL8 ( &SRCtimesX->refTime );
      UINT4 numSFTsX        = Timestamps_DETX->length;
      UINT4 numSamples_DETX = TimeSeries_DETX->data->length;
      UINT4 numSamples_SRCX = TimeSeries_SRCX_a->data->length;

      // sanity checks on input data
      XLAL_CHECK ( numSamples_SRCX == TimeSeries_SRCX_b->data->length, XLAL_EINVAL );
      XLAL_CHECK ( dt_SRC == TimeSeries_SRCX_a->deltaT, XLAL_EINVAL );
      XLAL_CHECK ( dt_SRC == TimeSeries_SRCX_b->deltaT, XLAL_EINVAL );
      XLAL_CHECK ( numSamples_DETX > 0, XLAL_EINVAL, "Input timeseries for detector X=%d has zero samples. Can't handle that!\n", X );
      XLAL_CHECK ( (SRCtimesX->DeltaT->length == numSFTsX) && (SRCtimesX->Tdot->length == numSFTsX), XLAL_EINVAL );
      XLAL_CHECK ( fHet == resamp->multiTimeSeries_DET->data[X]->f0, XLAL_EINVAL );
      XLAL_CHECK ( Tsft == common->multiTimestamps->data[X]->deltaT, XLAL_EINVAL );

      TimeSeries_SRCX_a->f0 = fHet;
      TimeSeries_SRCX_b->f0 = fHet;
      // set SRC-frame time-series start-time
      REAL8 tStart_SRC_0 = refTime8 + SRCtimesX->DeltaT->data[0] - (0.5*Tsft) * SRCtimesX->Tdot->data[0];
      LIGOTimeGPS epoch;
      GPSSETREAL8 ( epoch, tStart_SRC_0 );
      TimeSeries_SRCX_a->epoch = epoch;
      TimeSeries_SRCX_b->epoch = epoch;

      // make sure all output samples are initialized to zero first, in case of gaps
      memset ( TimeSeries_SRCX_a->data->data, 0, TimeSeries_SRCX_a->data->length * sizeof(TimeSeries_SRCX_a->data->data[0]) );
      memset ( TimeSeries_SRCX_b->data->data, 0, TimeSeries_SRCX_b->data->length * sizeof(TimeSeries_SRCX_b->data->data[0]) );
      // make sure detector-frame timesteps to interpolate to are initialized to 0, in case of gaps
      memset ( resamp->ws->SRCtimes_DET->data, 0, resamp->ws->SRCtimes_DET->length * sizeof(resamp->ws->SRCtimes_DET->data[0]) );

      REAL8 tStart_DET_0 = GPSGETREAL8 ( &(Timestamps_DETX->data[0]) );// START time of the SFT at the detector

      // loop over SFT timestamps and compute the detector frame time samples corresponding to uniformly sampled SRC time samples
      for ( UINT4 alpha = 0; alpha < numSFTsX; alpha ++ )
        {
          // define some useful shorthands
          REAL8 Tdot_al       = SRCtimesX->Tdot->data [ alpha ];		// the instantaneous time derivitive dt_SRC/dt_DET at the MID-POINT of the SFT
          REAL8 tMid_SRC_al   = refTime8 + SRCtimesX->DeltaT->data[alpha];	// MID-POINT time of the SFT at the SRC
          REAL8 tStart_SRC_al = tMid_SRC_al - 0.5 * Tsft * Tdot_al;		// approximate START time of the SFT at the SRC
          REAL8 tEnd_SRC_al   = tMid_SRC_al + 0.5 * Tsft * Tdot_al;		// approximate END time of the SFT at the SRC

          REAL8 tStart_DET_al = GPSGETREAL8 ( &(Timestamps_DETX->data[alpha]) );// START time of the SFT at the detector
          REAL8 tMid_DET_al   = tStart_DET_al + 0.5 * Tsft;			// MID-POINT time of the SFT at the detector

          // indices of first and last SRC-frame sample corresponding to this SFT
          UINT4 iStart_SRC_al = lround ( (tStart_SRC_al - tStart_SRC_0) / dt_SRC );	// the index of the resampled timeseries corresponding to the start of the SFT
          UINT4 iEnd_SRC_al   = lround ( (tEnd_SRC_al - tStart_SRC_0) / dt_SRC );	// the index of the resampled timeseries corresponding to the end of the SFT

          // truncate to actual SRC-frame timeseries
          iStart_SRC_al = MYMIN ( iStart_SRC_al, numSamples_SRCX - 1);
          iEnd_SRC_al   = MYMIN ( iEnd_SRC_al, numSamples_SRCX - 1);
          UINT4 numSamplesSFT_SRC_al = iEnd_SRC_al - iStart_SRC_al + 1;		// the number of samples in the SRC-frame for this SFT

          REAL4 a_al = AMcoefX->a->data[alpha];
          REAL4 b_al = AMcoefX->b->data[alpha];
          XLAL_CHECK ( a_al != 0 && b_al != 0, XLAL_EINVAL );
          for ( UINT4 j = 0; j < numSamplesSFT_SRC_al; j++ )
            {
              UINT4 iSRC_al_j  = iStart_SRC_al + j;

              // for each time sample in the SRC frame, we estimate the corresponding detector time,
              // using a linear approximation expanding around the midpoint of each SFT
              REAL8 t_SRC = tStart_SRC_0 + iSRC_al_j * dt_SRC;
              ti_DET->data [ iSRC_al_j ] = tMid_DET_al + ( t_SRC - tMid_SRC_al ) / Tdot_al;

              // pre-compute correction factors due to non-zero heterodyne frequency of input
              REAL8 tDiff = iSRC_al_j * dt_SRC + (tStart_DET_0 - ti_DET->data [ iSRC_al_j ]); 	// tSRC_al_j - tDET(tSRC_al_j)
              REAL8 cycles = fmod ( fHet * tDiff, 1.0 );				// the accumulated heterodyne cycles

              // use a look-up-table for speed to compute real and imaginary phase
              REAL4 cosphase, sinphase;                                   // the real and imaginary parts of the phase correction
              XLAL_CHECK( XLALSinCos2PiLUT ( &sinphase, &cosphase, -cycles ) == XLAL_SUCCESS, XLAL_EFUNC );
              COMPLEX8 ei2piphase = crectf ( cosphase, sinphase );

              // apply AM coefficients a(t), b(t) to SRC frame timeseries [alternate sign to get final FFT return DC in the middle]
              REAL4 signum = signumLUT [ (iSRC_al_j % 2) ];	// alternating sign, avoid branching
              ei2piphase *= signum;
              resamp->ws->TStmp1_SRC->data [ iSRC_al_j ] = ei2piphase * a_al;
              resamp->ws->TStmp2_SRC->data [ iSRC_al_j ] = ei2piphase * b_al;
            } // for j < numSamples_SRC_al

        } // for  alpha < numSFTsX

      const UINT4 Dterms = 8;
      XLAL_CHECK ( ti_DET->length >= TimeSeries_SRCX_a->data->length, XLAL_EINVAL );
      UINT4 bak_length = ti_DET->length;
      ti_DET->length = TimeSeries_SRCX_a->data->length;
      XLAL_CHECK ( XLALSincInterpolateCOMPLEX8TimeSeries ( TimeSeries_SRCX_a->data, ti_DET, TimeSeries_DETX, Dterms ) == XLAL_SUCCESS, XLAL_EFUNC );
      ti_DET->length = bak_length;

      // apply heterodyne correction and AM-functions a(t) and b(t) to interpolated timeseries
      for ( UINT4 j = 0; j < numSamples_SRCX; j ++ )
        {
          TimeSeries_SRCX_b->data->data[j] = TimeSeries_SRCX_a->data->data[j] * resamp->ws->TStmp2_SRC->data[j];
          TimeSeries_SRCX_a->data->data[j] *= resamp->ws->TStmp1_SRC->data[j];
        } // for j < numSamples_SRCX

    } // for X < numDetectors

  XLALDestroyMultiAMCoeffs ( multiAMcoef );
  XLALDestroyMultiSSBtimes ( multiSRCtimes );

  return XLAL_SUCCESS;

} // XLALBarycentricResampleMultiCOMPLEX8TimeSeries()

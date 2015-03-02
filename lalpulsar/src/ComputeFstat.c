//
// Copyright (C) 2014 Reinhard Prix
// Copyright (C) 2012, 2013, 2014 Karl Wette
// Copyright (C) 2007 Chris Messenger
// Copyright (C) 2006 John T. Whelan, Badri Krishnan
// Copyright (C) 2005, 2006, 2007, 2010 Reinhard Prix
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

#include <config.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <lal/ComputeFstat.h>
#include <lal/Factorial.h>
#include <lal/TimeSeries.h>
#include <lal/LALComputeAM.h>
#include <lal/LFTandTSutils.h>
#include <lal/SinCosLUT.h>
#include <lal/NormalizeSFTRngMed.h>

// ----- macro definitions
#define SQ(x) ( (x) * (x) )

// ---------- Internal struct definitions ---------- //

// Common input data for F-statistic methods
typedef struct {
  MultiLALDetector detectors;                           // List of detectors
  MultiLIGOTimeGPSVector *multiTimestamps;              // Multi-detector list of SFT timestamps
  MultiNoiseWeights *multiNoiseWeights;                 // Multi-detector noise weights
  MultiDetectorStateSeries *multiDetectorStates;        // Multi-detector state series
  const EphemerisData *ephemerides;                     // Ephemerides for the time-span of the SFTs
  SSBprecision SSBprec;                                 // Barycentric transformation precision
  FstatMethodType FstatMethod;                          // Method to use for computing the F-statistic
} FstatInput_Common;

// Input data specific to F-statistic methods
typedef struct tagFstatInput_Demod FstatInput_Demod;
typedef struct tagFstatInput_Resamp FstatInput_Resamp;

// Internal definition of input data structure
struct tagFstatInput {
  FstatInput_Common* common;                        // Common input data
  FstatInput_Demod* demod;                          // Demodulation input data
  FstatInput_Resamp* resamp;                        // Resampling input data
};

// ----- internal prototypes
// ---------- Check for various computer architectures ---------- //

#if defined(HAVE_SSE) || defined(__SSE__)
#define CFS_HAVE_SSE 1
#else
#define CFS_HAVE_SSE 0
#endif

#if defined(HAVE_ALTIVEC) || defined(__ALTIVEC__)
#define CFS_HAVE_ALTIVEC 1
#else
#define CFS_HAVE_ALTIVEC 0
#endif

// ---------- Include F-statistic method implementations ---------- //

#if CFS_HAVE_SSE
const int FMETHOD_DEMOD_BEST = FMETHOD_DEMOD_SSE;
#elif CFS_HAVE_ALTIVEC
const int FMETHOD_DEMOD_BEST = FMETHOD_DEMOD_ALTIVEC;
#else
const int FMETHOD_DEMOD_BEST = FMETHOD_DEMOD_OPTC;
#endif

const int FMETHOD_RESAMP_BEST = FMETHOD_RESAMP_GENERIC;

static const struct {
  const char *const name;
  BOOLEAN available;
} FstatMethodNames[FMETHOD_END] = {
  [FMETHOD_DEMOD_GENERIC]	= {"DemodGeneric",	1 },
  [FMETHOD_DEMOD_OPTC] 		= {"DemodOptC", 	1 },
  [FMETHOD_DEMOD_SSE] 		= {"DemodSSE", 		CFS_HAVE_SSE},
  [FMETHOD_DEMOD_ALTIVEC] 	= {"DemodAltivec", 	CFS_HAVE_ALTIVEC},

  [FMETHOD_RESAMP_GENERIC]	= {"ResampGeneric", 	1 }
} ;

#include "ComputeFstat_Demod.c"
#include "ComputeFstat_Resamp.c"

// ==================== Function definitions =================== //

///
/// Create a #FstatInputVector of the given length, for example for setting up
/// F-stat searches over several segments.
///
FstatInputVector*
XLALCreateFstatInputVector ( const UINT4 length            ///< [in] Length of the #FstatInputVector.
                             )
{
  // Allocate and initialise vector container
  FstatInputVector* inputs;
  XLAL_CHECK_NULL ( (inputs = XLALCalloc ( 1, sizeof(*inputs))) != NULL, XLAL_ENOMEM );
  inputs->length = length;

  // Allocate and initialise vector data
  if (inputs->length > 0) {
    XLAL_CHECK_NULL ( (inputs->data = XLALCalloc ( inputs->length, sizeof(inputs->data[0]) )) != NULL, XLAL_ENOMEM );
  }

  return inputs;

} // XLALCreateFstatInputVector()

///
/// Free all memory associated with a #FstatInputVector structure.
///
void
XLALDestroyFstatInputVector ( FstatInputVector* inputs        ///< [in] #FstatInputVector structure to be freed.
                              )
{
  if ( inputs == NULL ) {
    return;
  }

  if ( inputs->data )
    {
      for ( UINT4 i = 0; i < inputs->length; ++i ) {
        XLALDestroyFstatInput ( inputs->data[i] );
      }
      XLALFree ( inputs->data );
    }

  XLALFree ( inputs );

  return;

} // XLALDestroyFstatInputVector()

///
/// Create a #FstatAtomVector of the given length.
///
FstatAtomVector*
XLALCreateFstatAtomVector ( const UINT4 length ///< [in] Length of the #FstatAtomVector.
                            )
{
  // Allocate and initialise vector container
  FstatAtomVector* atoms;
  XLAL_CHECK_NULL ( (atoms = XLALCalloc ( 1, sizeof(*atoms) )) != NULL, XLAL_ENOMEM );
  atoms->length = length;

  // Allocate and initialise vector data
  if (atoms->length > 0) {
    XLAL_CHECK_NULL ( (atoms->data = XLALCalloc (atoms->length, sizeof(atoms->data[0]) )) != NULL, XLAL_ENOMEM );
  }

  return atoms;

} // XLALCreateFstatAtomVector()

///
/// Free all memory associated with a #FstatAtomVector structure.
///
void
XLALDestroyFstatAtomVector ( FstatAtomVector *atoms      ///< [in] #FstatAtomVector structure to be freed.
                             )
{
  if ( atoms == NULL ) {
    return;
  }

  if ( atoms->data ) {
    XLALFree ( atoms->data );
  }
  XLALFree ( atoms );

  return;

} // XLALDestroyFstatAtomVector()

///
/// Create a #MultiFstatAtomVector of the given length.
///
MultiFstatAtomVector*
XLALCreateMultiFstatAtomVector ( const UINT4 length   ///< [in] Length of the #MultiFstatAtomVector.
                                 )
{
  // Allocate and initialise vector container
  MultiFstatAtomVector* multiAtoms;
  XLAL_CHECK_NULL ( (multiAtoms = XLALCalloc(1, sizeof(*multiAtoms))) != NULL, XLAL_ENOMEM );
  multiAtoms->length = length;

  // Allocate and initialise vector data
  if ( multiAtoms->length > 0 ) {
    XLAL_CHECK_NULL ( (multiAtoms->data = XLALCalloc ( multiAtoms->length, sizeof(multiAtoms->data[0]) )) != NULL, XLAL_ENOMEM );
  }

  return multiAtoms;

} // XLALCreateMultiFstatAtomVector()

///
/// Free all memory associated with a #MultiFstatAtomVector structure.
///
void
XLALDestroyMultiFstatAtomVector ( MultiFstatAtomVector *multiAtoms  ///< [in] #MultiFstatAtomVector structure to be freed.
                                  )
{
  if ( multiAtoms == NULL ) {
    return;
  }

  for ( UINT4 X = 0; X < multiAtoms->length; ++X ) {
    XLALDestroyFstatAtomVector ( multiAtoms->data[X] );
  }
  XLALFree ( multiAtoms->data );
  XLALFree ( multiAtoms );

  return;

} // XLALDestroyMultiFstatAtomVector()

///
/// Create a fully-setup \c FstatInput structure for computing the \f$\mathcal{F}\f$-statistic using XLALComputeFstat().
///
FstatInput *
XLALCreateFstatInput ( const SFTCatalog *SFTcatalog,		  ///< [in] Catalog of SFTs to either load from files, or generate in memory.
                                                                  ///< The \c locator field of each ::SFTDescriptor must be \c !NULL for SFT loading,
                                                                  ///< and \c NULL for SFT generation.

                       const REAL8 minCoverFreq,		  ///< [in] Minimum instantaneous frequency which will be covered over the SFT time span.
                       const REAL8 maxCoverFreq,		  ///< [in] Maximum instantaneous frequency which will be covered over the SFT time span.
                       const PulsarParamsVector *injectSources,	  ///< [in] Optional vector of parameters of CW signals to simulate and inject.

                       const MultiNoiseFloor *injectSqrtSX,	  ///< [in] Optional array of single-sided PSD values governing fake Gaussian noise generation.
                                                                  ///< If supplied, then fake Gaussian noise with the given PSD values will be added to the SFTs.

                       const MultiNoiseFloor *assumeSqrtSX,	  ///< [in] Optional array of single-sided PSD values governing the calculation of SFT noise weights.
                                                                  ///< If supplied, then SFT noise weights are calculated from constant spectra with the given PSD
                                                                  ///< values; otherwise, SFT noise weights are calculated from PSDs computed from a running median
                                                                  ///< of the SFTs themselves.
                       const UINT4 runningMedianWindow,		  ///< [in] If SFT noise weights are calculated from the SFTs, the running median window length to use.

                       const EphemerisData *ephemerides,	  ///< [in] Ephemerides for the time-span of the SFTs.

                       const FstatMethodType FstatMethod,	  ///< [in] Method to use for computing the \f$\mathcal{F}\f$-statistic.
                       const FstatExtraParams *extraParams        ///< [in] Minor tuning or method-specific parameters.
                       )
{
  // Check catalog
  XLAL_CHECK_NULL ( SFTcatalog != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( SFTcatalog->length > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL ( SFTcatalog->data != NULL, XLAL_EINVAL );
  for ( UINT4 i = 1; i < SFTcatalog->length; ++i ) {
    XLAL_CHECK_NULL ( (SFTcatalog->data[0].locator == NULL) == (SFTcatalog->data[i].locator == NULL), XLAL_EINVAL,
                      "All 'locator' fields of SFTDescriptors in 'SFTcatalog' must be either NULL or !NULL." );
  }

  // Check remaining parameters
  XLAL_CHECK_NULL ( isfinite(minCoverFreq) && minCoverFreq > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL ( isfinite(maxCoverFreq) && maxCoverFreq > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL ( maxCoverFreq > minCoverFreq, XLAL_EINVAL );
  XLAL_CHECK_NULL ( injectSources == NULL || injectSources->length > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL ( injectSources == NULL || injectSources->data != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( ephemerides != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( extraParams != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( extraParams->SSBprec < SSBPREC_LAST, XLAL_EINVAL );

  // Determine whether to load and/or generate SFTs
  const BOOLEAN loadSFTs = (SFTcatalog->data[0].locator != NULL);
  const BOOLEAN generateSFTs = (injectSources != NULL) || (injectSqrtSX != NULL);
  XLAL_CHECK_NULL ( loadSFTs || generateSFTs, XLAL_EINVAL, "Can neither load nor generate SFTs with given parameters" );

  // Create top-level input data struct
  FstatInput* input;
  XLAL_CHECK_NULL ( (input = XLALCalloc ( 1, sizeof(*input) )) != NULL, XLAL_ENOMEM );

  // create common input data
  XLAL_CHECK_NULL ( (input->common = XLALCalloc ( 1, sizeof(*input->common))) != NULL, XLAL_ENOMEM );
  FstatInput_Common *const common = input->common;	// handy shortcut

  // create method-specific input data
  if ( XLALFstatMethodClassIsDemod ( FstatMethod ) )
    {
      XLAL_CHECK_NULL ( (input->demod = XLALCalloc ( 1, sizeof(FstatInput_Demod) )) != NULL, XLAL_ENOMEM );
      input->demod->Dterms = extraParams->Dterms;
    }
  else if ( XLALFstatMethodClassIsResamp ( FstatMethod ) )
    {
      XLAL_CHECK_NULL ( (input->resamp = XLALCalloc(1, sizeof(FstatInput_Resamp))) != NULL, XLAL_ENOMEM );
    }
  else
    {
      XLAL_ERROR_NULL ( XLAL_EINVAL, "Received invalid Fstat method enum '%d'\n", FstatMethod );
    }
  common->FstatMethod = FstatMethod;

  // Determine the time baseline of an SFT
  const REAL8 Tsft = 1.0 / SFTcatalog->data[0].header.deltaF;

  // Determine the frequency band required by each method 'minFreqMethod',
  // as well as the frequency band required to load or generate initial SFTs for 'minFreqFull'
  // the difference being that for noise-floor estimation, we need extra frequency-bands for the
  // running median
  REAL8 minFreqMethod, maxFreqMethod;
  REAL8 minFreqFull, maxFreqFull;
  {
    // Determine whether the method being used requires extra SFT frequency bins
    int extraBinsMethod = 0;
    if ( input->demod != NULL )
      {
        extraBinsMethod = GetFstatExtraBins_Demod ( input->demod );
      }
    else if ( input->resamp != NULL )
      {
        extraBinsMethod = GetFstatExtraBins_Resamp ( input->resamp );
      }
    else
      {
        XLAL_ERROR_NULL ( XLAL_EFAILED, "Invalid FstatInput struct passed to %s()", __func__);
      }
    XLAL_CHECK_NULL ( extraBinsMethod >= 0, XLAL_EFAILED );

    // Add number of extra frequency bins required by running median
    int extraBinsFull = extraBinsMethod + runningMedianWindow/2 + 1; // NOTE: running-median window needed irrespective of assumeSqrtSX!

    // Extend frequency range by number of extra bins times SFT bin width
    const REAL8 extraFreqMethod = extraBinsMethod / Tsft;
    minFreqMethod = minCoverFreq - extraFreqMethod;
    maxFreqMethod = maxCoverFreq + extraFreqMethod;

    const REAL8 extraFreqFull = extraBinsFull / Tsft;
    minFreqFull = minCoverFreq - extraFreqFull;
    maxFreqFull = maxCoverFreq + extraFreqFull;

  } // end: block to determine frequency-bins range

  // Load SFTs, if required, and extract detectors and timestamps
  MultiSFTVector *multiSFTs = NULL;
  if (loadSFTs)
    {
      // Load all SFTs at once
      XLAL_CHECK_NULL ( ( multiSFTs = XLALLoadMultiSFTs(SFTcatalog, minFreqFull, maxFreqFull) ) != NULL, XLAL_EFUNC );

      // Extract detectors and timestamps from SFTs
      XLAL_CHECK_NULL ( XLALMultiLALDetectorFromMultiSFTs ( &common->detectors, multiSFTs ) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK_NULL ( ( common->multiTimestamps = XLALExtractMultiTimestampsFromSFTs ( multiSFTs ) ) != NULL,  XLAL_EFUNC );

    }
  else
    {
      // Create a multi-view of SFT catalog
      MultiSFTCatalogView *multiSFTcatalog;
      XLAL_CHECK_NULL ( (multiSFTcatalog = XLALGetMultiSFTCatalogView(SFTcatalog)) != NULL, XLAL_EFUNC );

      // Extract detectors and timestamps from multi-view of SFT catalog
      XLAL_CHECK_NULL ( XLALMultiLALDetectorFromMultiSFTCatalogView ( &common->detectors, multiSFTcatalog ) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK_NULL ( ( common->multiTimestamps = XLALTimestampsFromMultiSFTCatalogView ( multiSFTcatalog ) ) != NULL,  XLAL_EFUNC );

      // Cleanup
      XLALDestroyMultiSFTCatalogView ( multiSFTcatalog );
    } // end: if !loadSFTs

  // Check length of multi-noise floor arrays
  XLAL_CHECK_NULL ( injectSqrtSX == NULL || injectSqrtSX->length == common->detectors.length, XLAL_EINVAL );
  XLAL_CHECK_NULL ( assumeSqrtSX == NULL || assumeSqrtSX->length == common->detectors.length, XLAL_EINVAL );

  // Generate SFTs with injections and noise, if required
  if (generateSFTs)
    {
      // Initialise parameters struct for XLALCWMakeFakeMultiData()
      CWMFDataParams XLAL_INIT_DECL(MFDparams);
      MFDparams.fMin = minFreqFull;
      MFDparams.Band = maxFreqFull - minFreqFull;
      MFDparams.multiIFO = common->detectors;
      MFDparams.multiTimestamps = *(common->multiTimestamps);
      MFDparams.randSeed = extraParams->randSeed;

      // Set noise floors if sqrtSX is given; otherwise noise floors are zero
      if ( injectSqrtSX != NULL ) {
        MFDparams.multiNoiseFloor = (*injectSqrtSX);
      } else {
        MFDparams.multiNoiseFloor.length = common->detectors.length;
      }

      // Generate SFTs with injections
      MultiSFTVector *fakeMultiSFTs = NULL;
      XLAL_CHECK_NULL ( XLALCWMakeFakeMultiData ( &fakeMultiSFTs, NULL, injectSources, &MFDparams, ephemerides ) == XLAL_SUCCESS, XLAL_EFUNC );

      // If SFTs were loaded, add generated SFTs to then, otherwise just used generated SFTs
      if (multiSFTs != NULL) {
        XLAL_CHECK_NULL ( XLALMultiSFTVectorAdd ( multiSFTs, fakeMultiSFTs ) == XLAL_SUCCESS, XLAL_EFUNC );
        XLALDestroyMultiSFTVector ( fakeMultiSFTs );
      } else {
        multiSFTs = fakeMultiSFTs;
      }

    } // if generateSFTs

  // Check that no single-SFT input vectors are given to avoid returning singular results
  for ( UINT4 X = 0; X < common->detectors.length; ++X ) {
    XLAL_CHECK_NULL ( multiSFTs->data[X]->length > 1, XLAL_EINVAL, "Need more than 1 SFTs per Detector!\n" );
  }

  // Normalise SFTs using either running median or assumed PSDs
  MultiPSDVector *runningMedian;
  XLAL_CHECK_NULL ( (runningMedian = XLALNormalizeMultiSFTVect ( multiSFTs, runningMedianWindow, assumeSqrtSX )) != NULL, XLAL_EFUNC );

  // Calculate SFT noise weights from PSD
  XLAL_CHECK_NULL ( (common->multiNoiseWeights = XLALComputeMultiNoiseWeights ( runningMedian, runningMedianWindow, 0 )) != NULL, XLAL_EFUNC );

  // at this point we're done with running-median noise estimation and can 'trim' the SFTs back to
  // the width actually required by the Fstat-methods *methods*.
  // NOTE: this is especially relevant for resampling, where the frequency-band determines the sampling
  // rate, and the number of samples that need to be FFT'ed
  XLAL_CHECK_NULL ( XLALMultiSFTVectorResizeBand ( multiSFTs, minFreqMethod, maxFreqMethod - minFreqMethod ) == XLAL_SUCCESS, XLAL_EFUNC );

  // Get detector states, with a timestamp shift of Tsft/2
  const REAL8 tOffset = 0.5 * Tsft;
  XLAL_CHECK_NULL ( (common->multiDetectorStates = XLALGetMultiDetectorStates ( common->multiTimestamps, &common->detectors, ephemerides, tOffset )) != NULL, XLAL_EFUNC );

  // Save ephemerides and SSB precision
  common->ephemerides = ephemerides;
  common->SSBprec = extraParams->SSBprec;

  // Call the appropriate method function to setup their input data structures
  // - The method input data structures are expected to take ownership of the
  //   SFTs, which is why 'input->common' does not retain a pointer to them
  if ( input->demod != NULL )
    {
      XLAL_CHECK_NULL ( SetupFstatInput_Demod ( input->demod, common, multiSFTs ) == XLAL_SUCCESS, XLAL_EFUNC );
    }
  else if ( input->resamp != NULL )
    {
      XLAL_CHECK_NULL ( SetupFstatInput_Resamp ( input->resamp, common, multiSFTs ) == XLAL_SUCCESS, XLAL_EFUNC );
    }
  else
    {
      XLAL_ERROR_NULL ( XLAL_EFAILED, "Invalid FstatInput struct passed to %s()", __func__ );
    }
  multiSFTs = NULL;

  // Cleanup
  XLALDestroyMultiPSDVector ( runningMedian );

  return input;

} // XLALCreateFstatInput()

///
/// Returns the detector information stored in a \c FstatInput structure.
///
const MultiLALDetector*
XLALGetFstatInputDetectors ( const FstatInput* input    ///< [in] \c FstatInput structure.
                             )
{
  // Check input
  XLAL_CHECK_NULL ( input != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( input->common != NULL, XLAL_EINVAL, "'input' has not yet been set up" );

  return &input->common->detectors;

} // XLALGetFstatInputDetectors()

///
/// Returns the SFT timestamps stored in a \c FstatInput structure.
///
const MultiLIGOTimeGPSVector*
XLALGetFstatInputTimestamps ( const FstatInput* input	///< [in] \c FstatInput structure.
                              )
{
  // Check input
  XLAL_CHECK_NULL ( input != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( input->common != NULL, XLAL_EINVAL, "'input' has not yet been set up" );

  return input->common->multiTimestamps;

} // XLALGetFstatInputTimestamps()

///
/// Returns the multi-detector noise weights stored in a \c FstatInput structure.
///
const MultiNoiseWeights*
XLALGetFstatInputNoiseWeights ( const FstatInput* input     ///< [in] \c FstatInput structure.
                                )
{
  // Check input
  XLAL_CHECK_NULL ( input != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( input->common != NULL, XLAL_EINVAL, "'input' has not yet been set up" );

  return input->common->multiNoiseWeights;

} // XLALGetFstatInputNoiseWeights()

///
/// Returns the multi-detector state series stored in a \c FstatInput structure.
///
const MultiDetectorStateSeries*
XLALGetFstatInputDetectorStates ( const FstatInput* input	///< [in] \c FstatInput structure.
                                  )
{
  // Check input
  XLAL_CHECK_NULL ( input != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( input->common != NULL, XLAL_EINVAL, "'input' has not yet been set up" );

  return input->common->multiDetectorStates;

} // XLALGetFstatInputDetectorStates()

///
/// Compute the \f$\mathcal{F}\f$-statistic over a band of frequencies.
///
int
XLALComputeFstat ( FstatResults **Fstats,	  	///< [in/out] Address of a pointer to a #FstatResults results structure; if \c NULL, allocate here.
                   FstatInput *input,		  	///< [in] Input data structure created by one of the setup functions.
                   const PulsarDopplerParams *doppler,  ///< [in] Doppler parameters, including starting frequency, at which to compute \f$2\mathcal{F}\f$
                   const REAL8 dFreq,	  		///< [in] Required spacing in frequency between each \f$\mathcal{F}\f$-statistic.
                   const UINT4 numFreqBins,		///< [in] Number of frequencies at which the \f$2\mathcal{F}\f$ are to be computed.
                   const FstatQuantities whatToCompute	///< [in] Bit-field of which \f$\mathcal{F}\f$-statistic quantities to compute.
                   )
{
  // Check input
  XLAL_CHECK ( Fstats != NULL, XLAL_EINVAL);
  XLAL_CHECK ( input != NULL, XLAL_EINVAL);
  XLAL_CHECK ( input->common != NULL, XLAL_EINVAL, "'input' has not yet been set up");
  XLAL_CHECK ( doppler != NULL, XLAL_EINVAL);
  XLAL_CHECK ( doppler->asini >= 0, XLAL_EINVAL);
  XLAL_CHECK ( dFreq > 0 || (numFreqBins == 1 && dFreq >= 0), XLAL_EINVAL);
  XLAL_CHECK ( numFreqBins > 0, XLAL_EINVAL);
  XLAL_CHECK ( 0 < whatToCompute && whatToCompute < FSTATQ_LAST, XLAL_EINVAL);

  // Allocate results struct, if needed
  if ( (*Fstats) == NULL ) {
    XLAL_CHECK ( ((*Fstats) = XLALCalloc ( 1, sizeof(**Fstats) )) != NULL, XLAL_ENOMEM );
  }

  // Get constant pointer to common input data
  const FstatInput_Common *common = input->common;
  const UINT4 numDetectors = common->detectors.length;

  // Enlarge result arrays if they are too small
  const BOOLEAN moreFreqBins = (numFreqBins > (*Fstats)->internalalloclen);
  const BOOLEAN moreDetectors = (numDetectors > (*Fstats)->numDetectors);
  if (moreFreqBins || moreDetectors)
    {
      // Enlarge multi-detector 2F array
      if ( (whatToCompute & FSTATQ_2F) && moreFreqBins )
        {
          (*Fstats)->twoF = XLALRealloc ( (*Fstats)->twoF, numFreqBins*sizeof((*Fstats)->twoF[0]) );
          XLAL_CHECK ( (*Fstats)->twoF != NULL, XLAL_EINVAL, "Failed to (re)allocate (*Fstats)->twoF to length %u", numFreqBins );
        }

      // Enlarge multi-detector Fa & Fb array
      if ( (whatToCompute & FSTATQ_FAFB) && moreFreqBins )
        {
          (*Fstats)->Fa = XLALRealloc( (*Fstats)->Fa, numFreqBins * sizeof((*Fstats)->Fa[0]) );
          XLAL_CHECK ( (*Fstats)->Fa != NULL, XLAL_EINVAL, "Failed to (re)allocate (*Fstats)->Fa to length %u", numFreqBins );
          (*Fstats)->Fb = XLALRealloc( (*Fstats)->Fb, numFreqBins * sizeof((*Fstats)->Fb[0]) );
          XLAL_CHECK ( (*Fstats)->Fb != NULL, XLAL_EINVAL, "Failed to (re)allocate (*Fstats)->Fb to length %u", numFreqBins );
        }

      // Enlarge 2F per detector arrays
      if ( (whatToCompute & FSTATQ_2F_PER_DET) && (moreFreqBins || moreDetectors) )
        {
          for ( UINT4 X = 0; X < numDetectors; ++X )
            {
              (*Fstats)->twoFPerDet[X] = XLALRealloc ( (*Fstats)->twoFPerDet[X], numFreqBins * sizeof((*Fstats)->twoFPerDet[X][0]) );
              XLAL_CHECK ( (*Fstats)->twoFPerDet[X] != NULL, XLAL_EINVAL, "Failed to (re)allocate (*Fstats)->twoFPerDet[%u] to length %u", X, numFreqBins );
            }
        }

      // Enlarge Fa & Fb per detector arrays
      if ( ( whatToCompute & FSTATQ_FAFB_PER_DET) && (moreFreqBins || moreDetectors) )
        {
          for ( UINT4 X = 0; X < numDetectors; ++X )
            {
              (*Fstats)->FaPerDet[X] = XLALRealloc ( (*Fstats)->FaPerDet[X], numFreqBins*sizeof((*Fstats)->FaPerDet[X][0]) );
              XLAL_CHECK( (*Fstats)->FaPerDet[X] != NULL, XLAL_EINVAL, "Failed to (re)allocate (*Fstats)->FaPerDet[%u] to length %u", X, numFreqBins );
              (*Fstats)->FbPerDet[X] = XLALRealloc ( (*Fstats)->FbPerDet[X], numFreqBins*sizeof((*Fstats)->FbPerDet[X][0]) );
              XLAL_CHECK( (*Fstats)->FbPerDet[X] != NULL, XLAL_EINVAL, "Fbiled to (re)allocate (*Fstats)->FbPerDet[%u] to length %u", X, numFreqBins );
            }
        }

      // Enlarge F-atoms per detector arrays, and initialise to NULL
      if ( (whatToCompute & FSTATQ_ATOMS_PER_DET) && (moreFreqBins || moreDetectors) )
        {
              (*Fstats)->multiFatoms = XLALRealloc ( (*Fstats)->multiFatoms, numFreqBins*sizeof((*Fstats)->multiFatoms[0]) );
              XLAL_CHECK ( (*Fstats)->multiFatoms != NULL, XLAL_EINVAL, "Failed to (re)allocate (*Fstats)->multiFatoms to length %u", numFreqBins );

              // If more detectors are needed, destroy multi-F-atom vectors so they can be re-allocated later
              if ( moreDetectors )
                {
                  for ( UINT4 k = 0; k < numFreqBins; ++k )
                    {
                      XLALDestroyMultiFstatAtomVector ( (*Fstats)->multiFatoms[k] );
                      (*Fstats)->multiFatoms[k] = NULL;
                    }
                }
              else
                {
                  for ( UINT4 k = (*Fstats)->internalalloclen; k < numFreqBins; ++k ) {
                    (*Fstats)->multiFatoms[k] = NULL;
                  }
                }

        } // if Atoms_per_det to enlarge

      // Update allocated length of arrays
      (*Fstats)->internalalloclen = numFreqBins;

    } // if (moreFreqBins || moreDetectors)

  // Initialise result struct parameters
  (*Fstats)->doppler = *doppler;
  (*Fstats)->dFreq = dFreq;
  (*Fstats)->numFreqBins = numFreqBins;
  (*Fstats)->numDetectors = numDetectors;
  memset ( (*Fstats)->detectorNames, 0, sizeof((*Fstats)->detectorNames) );
  for (UINT4 X = 0; X < numDetectors; ++X) {
    strncpy ( (*Fstats)->detectorNames[X], common->detectors.sites[X].frDetector.prefix, 2 );
  }
  (*Fstats)->whatWasComputed = whatToCompute;

  // Call the appropriate method function to compute the F-statistic
  if ( input->demod != NULL )
    {
      XLAL_CHECK ( ComputeFstat_Demod(*Fstats, common, input->demod) == XLAL_SUCCESS, XLAL_EFUNC );
    }
  else if ( input->resamp != NULL )
    {
      XLAL_CHECK ( ComputeFstat_Resamp(*Fstats, common, input->resamp) == XLAL_SUCCESS, XLAL_EFUNC );
    }
  else
    {
      XLAL_ERROR(XLAL_EFAILED, "Invalid FstatInput struct passed to %s()", __func__);
    }

  return XLAL_SUCCESS;

} // XLALComputeFstat()

///
/// Free all memory associated with a \c FstatInput structure.
///
void
XLALDestroyFstatInput ( FstatInput* input	///< [in] \c FstatInput structure to be freed.
                        )
{
  if ( input == NULL ) {
    return;
  }

  if (input->common != NULL)
    {
      XLALDestroyMultiTimestamps ( input->common->multiTimestamps );
      XLALDestroyMultiNoiseWeights ( input->common->multiNoiseWeights );
      XLALDestroyMultiDetectorStateSeries ( input->common->multiDetectorStates );
      XLALFree ( input->common );
    }
  if (input->demod != NULL)
    {
      DestroyFstatInput_Demod ( input->demod );
    }
  else if ( input->resamp != NULL )
    {
      DestroyFstatInput_Resamp ( input->resamp );
    }

  XLALFree ( input );

  return;
} // XLALDestroyFstatInput()

///
/// Free all memory associated with a #FstatResults structure.
///
void
XLALDestroyFstatResults ( FstatResults* Fstats  ///< [in] #FstatResults structure to be freed.
                          )
{
  if ( Fstats == NULL ) {
    return;
  }

  XLALFree ( Fstats->twoF );
  XLALFree ( Fstats->Fa );
  XLALFree ( Fstats->Fb );
  for ( UINT4 X = 0; X < PULSAR_MAX_DETECTORS; ++X )
    {
      XLALFree ( Fstats->twoFPerDet[X] );
      XLALFree ( Fstats->FaPerDet[X] );
      XLALFree ( Fstats->FbPerDet[X] );
      if ( Fstats->multiFatoms != NULL )
        {
          for ( UINT4 n = 0; n < Fstats->internalalloclen; ++n )
            {
              XLALDestroyMultiFstatAtomVector ( Fstats->multiFatoms[n] );
            }
          XLALFree ( Fstats->multiFatoms );
        }
    }

  XLALFree ( Fstats );

  return;
} // XLALDestroyFstatResults()

///
/// Add +4 to any multi-detector or per-detector 2F values computed by XLALComputeFstat().
/// This is for compatibility with programs which expect this normalisation if SFTs do not
/// contain noise, e.g. \c lalapps_ComputeFstatistic with the \c --SignalOnly option.
///
int
XLALAdd4ToFstatResults ( FstatResults* Fstats    ///< [in/out] #FstatResults structure.
                         )
{
  // Check input
  XLAL_CHECK( Fstats != NULL, XLAL_EINVAL );

  // Add +4 to multi-detector 2F array
  if ( Fstats->whatWasComputed & FSTATQ_2F )
    {
      for ( UINT4 k = 0; k < Fstats->numFreqBins; ++k ) {
        Fstats->twoF[k] += 4;
      }
    }

  // Add +4 to 2F per detector arrays
  if ( Fstats->whatWasComputed & FSTATQ_2F_PER_DET )
    {
      for ( UINT4 X = 0; X < Fstats->numDetectors; ++X ) {
        for ( UINT4 k = 0; k < Fstats->numFreqBins; ++k ) {
          Fstats->twoFPerDet[X][k] += 4;
        }
      }
    }

  return XLAL_SUCCESS;

} // XLALAdd4ToFstatResults()

///
/// Compute single-or multi-IFO Fstat '2F' from multi-IFO 'atoms'
///
REAL4
XLALComputeFstatFromAtoms ( const MultiFstatAtomVector *multiFstatAtoms,   ///< [in] Multi-detector atoms
                            const INT4                 X                   ///< [in] Detector number, give -1 for multi-Fstat
                            )
{
  // ----- check input parameters and report errors
  XLAL_CHECK_REAL4 ( multiFstatAtoms && multiFstatAtoms->data && multiFstatAtoms->data[0]->data, XLAL_EINVAL, "Empty pointer as input parameter." );
  XLAL_CHECK_REAL4 ( multiFstatAtoms->length > 0, XLAL_EBADLEN, "Input MultiFstatAtomVector has zero length. (i.e., no detectors)" );
  XLAL_CHECK_REAL4 ( X >= -1, XLAL_EDOM, "Invalid detector number X=%d. Only nonnegative numbers, or -1 for multi-F, are allowed.", X );
  XLAL_CHECK_REAL4 ( ( X < 0 ) || ( (UINT4)(X) <= multiFstatAtoms->length-1 ), XLAL_EDOM, "Requested X=%d, but FstatAtoms only have length %d.", X, multiFstatAtoms->length );

  // internal detector index Y to do both single- and multi-F case
  UINT4 Y, Ystart, Yend;
  if ( X == -1 ) { /* loop through all detectors to get multi-Fstat */
    Ystart = 0;
    Yend   = multiFstatAtoms->length-1;
  }
  else { /* just compute single-Fstat for 1 IFO */
    Ystart = X;
    Yend   = X;
  }

  // set up temporary Fatoms and matrix elements for summations
  REAL4 mmatrixA = 0.0, mmatrixB = 0.0, mmatrixC = 0.0;
  REAL4 twoF = 0.0;
  COMPLEX8 Fa, Fb;
  Fa = 0.0;
  Fb = 0.0;

  for (Y = Ystart; Y <= Yend; Y++ ) /* loop through detectors */
    {
      UINT4 alpha, numSFTs;
      numSFTs = multiFstatAtoms->data[Y]->length;
      XLAL_CHECK_REAL4 ( numSFTs > 0, XLAL_EDOM, "Input FstatAtomVector has zero length. (i.e., no timestamps for detector X=%d)", Y );

      for ( alpha = 0; alpha < numSFTs; alpha++ ) /* loop through SFTs */
        {
          FstatAtom *thisAtom = &multiFstatAtoms->data[Y]->data[alpha];
          /* sum up matrix elements and Fa, Fb */
          mmatrixA += thisAtom->a2_alpha;
          mmatrixB += thisAtom->b2_alpha;
          mmatrixC += thisAtom->ab_alpha;
          Fa += thisAtom->Fa_alpha;
          Fb += thisAtom->Fb_alpha;
        } /* loop through SFTs */

    } // loop through detectors

  // compute determinant and final Fstat (not twoF!)
  REAL4 Dinv = 1.0 / ( mmatrixA * mmatrixB - SQ(mmatrixC) );

  twoF = XLALComputeFstatFromFaFb ( Fa, Fb, mmatrixA, mmatrixB, mmatrixC, 0, Dinv );

  return twoF;

} // XLALComputeFstatFromAtoms()


///
/// Simple helper function which computes \f$2\mathcal{F}\f$ from given \f$F_a\f$ and \f$F_b\f$, and antenna-pattern
/// coefficients \f$(A,B,C,E)\f$ with inverse determinant \f$\text{Dinv} = 1/D\f$ where \f$D = A * B - C^2 - E^2\f$.
///
REAL4
XLALComputeFstatFromFaFb ( COMPLEX8 Fa, COMPLEX8 Fb, REAL4 A, REAL4 B, REAL4 C, REAL4 E, REAL4 Dinv )
{
  REAL4 Fa_re = creal(Fa);
  REAL4 Fa_im = cimag(Fa);
  REAL4 Fb_re = creal(Fb);
  REAL4 Fb_im = cimag(Fb);

  REAL4 F = Dinv * (  B * ( SQ(Fa_re) + SQ(Fa_im) )
                      + A * ( SQ(Fb_re) + SQ(Fb_im) )
                      - 2.0 * C * (   Fa_re * Fb_re + Fa_im * Fb_im )
                      - 2.0 * E * ( - Fa_re * Fb_im + Fa_im * Fb_re )		// nonzero only in RAA case where Ed!=0
                      );
  return 2*F;

} // XLALComputeFstatFromFaFb()

///
/// Return true if given #FstatMethodType corresponds to a valid and *available* Fstat method, false otherwise
///
int
XLALFstatMethodIsAvailable ( FstatMethodType i )
{
  if ( (i <= FMETHOD_START) || (i >= FMETHOD_END) ) {
    return 0;
  }
  if ( (FstatMethodNames[i].name == NULL) || !FstatMethodNames[i].available ) {
    return 0;
  }

  return 1;
} // XLALFstatMethodIsAvailable()


///
/// Provide human-readable names for the different \f$\mathcal{F}\f$-statistic method variants in #FstatMethodType.
///
const CHAR *
XLALGetFstatMethodName ( FstatMethodType i )
{
  XLAL_CHECK_NULL ( (i > FMETHOD_START) && (i < FMETHOD_END) && FstatMethodNames[i].name!=NULL, XLAL_EDOM, "Invalid FstatMethodType = %d\n", i );
  return FstatMethodNames[i].name;
} // XLALGetFstatMethodName()

///
/// Return pointer to a static help string enumerating all (available) #FstatMethodType options.
/// Also indicates which is the (guessed) available 'best' method available.
///
const CHAR *
XLALFstatMethodHelpString ( void )
{
  static int firstCall = 1;
  static CHAR helpstr[1024];
  if ( firstCall )
    {
      CHAR buf[1024];
      strncpy (helpstr, "Available methods: (", sizeof(helpstr));
      UINT4 len = strlen(helpstr);
      const CHAR *separator = "";
      for (int i = FMETHOD_START + 1; i < FMETHOD_END; i++ )
        {
          if ( ! FstatMethodNames[i].available ) {
            continue;
          }
          snprintf ( buf, sizeof(buf), "%s%s", separator, FstatMethodNames[i].name );
          separator="|";
          if ( i == FMETHOD_DEMOD_BEST ) {
            strncat ( buf, "=DemodBest", sizeof(buf) - strlen(buf) - 1 );
          }
          if ( i == FMETHOD_RESAMP_BEST ) {
            strncat ( buf, "=ResampBest", sizeof(buf) - strlen(buf) - 1 );
          }
          len += strlen(buf);
          XLAL_CHECK_NULL ( len < sizeof(helpstr), XLAL_EBADLEN, "FstatMethod help-string exceeds buffer length (%lu)\n", sizeof(helpstr) );
          strcat ( helpstr, buf );
        } // for i < FMETHOD_LAST

      strcat(helpstr, ") ");
      firstCall = 0;

    } // if firstCall

  return helpstr;
} // XLALFstatMethodHelpString()

///
/// Parse a given string into an #FstatMethodType number if valid and available,
/// return error otherwise.
///
int
XLALParseFstatMethodString ( FstatMethodType *Fmethod, 	//!< [out] Parsed #FstatMethodType enum
                             const char *s		//!< [in] String to parse
                             )
{
  XLAL_CHECK ( s != NULL, XLAL_EINVAL );
  XLAL_CHECK ( Fmethod != NULL, XLAL_EINVAL );

  // handle special user-input strings to select respective (guessed) best method
  if ( strcmp ( s, "DemodBest" ) == 0 )
    {
      (*Fmethod) = FMETHOD_DEMOD_BEST;
      return XLAL_SUCCESS;
    }
  if ( strcmp ( s, "ResampBest" ) == 0 )
    {
      (*Fmethod) = FMETHOD_RESAMP_BEST;
      return XLAL_SUCCESS;
    }

  // find matching FstatMethod string
  for (int i = FMETHOD_START + 1; i < FMETHOD_END; i++ )
    {
      if ( (FstatMethodNames[i].name != NULL) && (strcmp ( s, FstatMethodNames[i].name ) == 0) )
        {
          if ( FstatMethodNames[i].available )
            {
              (*Fmethod) = i;
              return XLAL_SUCCESS;
            }
          else
            {
              XLAL_ERROR ( XLAL_EINVAL, "Chosen FstatMethod '%s' valid but unavailable in this binary\n", s );
            }
        } // if found matching FstatMethod
    } // for i < FMETHOD_LAST

  XLAL_ERROR ( XLAL_EINVAL, "Unknown FstatMethod '%s'\n", s );

} // XLALParseFstatMethodString()

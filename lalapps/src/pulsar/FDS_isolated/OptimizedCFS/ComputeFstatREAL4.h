/*
*  Copyright (C) 2009 Reinhard Prix, Oleg Korobkin, Bernd Machenschalk
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

#ifndef _COMPUTEFSTATREAL4_H
#define _COMPUTEFSTATREAL4_H

/* C++ protection. */
#ifdef  __cplusplus
extern "C" {
#endif

/* ---------- includes ---------- */
#include <lal/LALDatatypes.h>
#include <lal/DetectorStates.h>
#include <lal/ComputeFstat.h>

/* for OpenCL */
#ifndef USE_OPENCL_KERNEL
#define USE_OPENCL_KERNEL 0
#endif
#ifndef USE_OPENCL_KERNEL_CPU
#define USE_OPENCL_KERNEL_CPU 0
#endif
#define USE_OPENCL (USE_OPENCL_KERNEL || USE_OPENCL_KERNEL_CPU)

/* Map functions in HierarchicalSearch.c for CUDA / OpenCL use */
#ifdef USE_CUDA
#define GPUREADY_DEFAULT 1
#define INITIALIZE_COPROCESSOR_DEVICE   if(InitializeCUDADevice(&stackMultiSFT, &fstatVector)) return -1;
#define UNINITIALIZE_COPROCESSOR_DEVICE UnInitializeCUDADevice();
#define REARRANGE_SFT_DATA              RearrangeSFTData4CUDA( &fstatVector, &thisPoint, &stackMultiSFT, &stackMultiNoiseWeights, &stackMultiDetStates, CFparams.Dterms, &cfvBuffer);
#elif USE_OPENCL
#define GPUREADY_DEFAULT 1
#define INITIALIZE_COPROCESSOR_DEVICE   clW = empty_CLWorkspace; XLALInitCLWorkspace (&clW, &stackMultiSFT);
#define UNINITIALIZE_COPROCESSOR_DEVICE XLALDestroyCLWorkspace ( &clW, &stackMultiSFT );
#define REARRANGE_SFT_DATA              XLALRearrangeSFTData ( &clW, &fstatVector );
#endif /*  USE_CUDA */

/** ----- switch between different versions of XLALComputeFStatFreqBandVector() ----- */
#ifdef USE_CUDA
#define XLALComputeFStatFreqBandVector XLALComputeFStatFreqBandVectorCUDA
#elif USE_OPENCL
#define XLALComputeFStatFreqBandVector XLALComputeFStatFreqBandVectorOpenCL
#else
#define XLALComputeFStatFreqBandVector XLALComputeFStatFreqBandVectorCPU
#endif

/* ---------- exported types ---------- */
// FIXME: the following types might be useful to move into LAL at some point
/** sequence of MultiSFT vectors -- for each segment */
typedef struct tagMultiSFTVectorSequence {
  UINT4 length;     		/**< number of segments */
  MultiSFTVector **data; 	/**< the SFT vectors */
} MultiSFTVectorSequence;

/** sequence of Multi-noise weights vectors -- for each segment */
typedef struct tagMultiNoiseWeightsSequence {
  UINT4 length;     		/**< number of segments */
  MultiNoiseWeights **data; 	/**< the noise weights */
} MultiNoiseWeightsSequence;

/** sequence of Multi-detector state vectors -- for each segment */
typedef struct tagMultiDetectorStateSeriesSequence {
  UINT4 length;     		/**< number of segments */
  MultiDetectorStateSeries **data; /**< the detector state series */
} MultiDetectorStateSeriesSequence;


/** REAL4 version of pulsar spins fkdot[] */
typedef struct {
  REAL4 FreqMain;			/**< "main" part of frequency fkdot[0], normally just the integral part */
  REAL4 fkdot[PULSAR_MAX_SPINS];	/**< remaining spin-parameters, including *fractional* part of Freq = fkdot[0] */
  UINT4 spdnOrder;			/**< highest non-zero spindown order (ie maximal value of 'k' in fkdot = df^(k)/dt^k */
} PulsarSpinsREAL4;


/**
 * Simple container for REAL4-vectors, holding the SSB-timings DeltaT_alpha  and Tdot_alpha,
 * with one entry per SFT-timestamp. We also store the SSB reference-time tau0.
 * NOTE: this is a REAL4 version of SSBtimes, preserving the required precision by appropriate
 * 'splitting' of REAL8's into pairs of REAL4s.
 */
typedef struct {
  LIGOTimeGPS refTime;		/**< Reference time wrt to which the time-differences DeltaT are computed */
  REAL4Vector *DeltaT_int;	/**< Integral part of time-difference of SFT-alpha - tau0 in SSB-frame */
  REAL4Vector *DeltaT_rem;	/**< Remainder of time-difference of SFT-alpha - tau0 in SSB-frame */
  REAL4Vector *TdotM1;		/**< dT/dt - 1 : time-derivative of SSB-time wrt local time for SFT-alpha, of order O(1e-4) */
} SSBtimesREAL4;


/** Multi-IFO container for SSB timings in REAL4-representation */
typedef struct {
  UINT4 length;			/**< number of IFOs */
  SSBtimesREAL4 **data;		/**< array of SSBtimes (pointers) */
} MultiSSBtimesREAL4;


/**
 * Type containing F-statistic proper plus the two complex amplitudes Fa and Fb (for ML-estimators).
 * NOTE: this is simply a REAL4 version of Fcomponents.
 */
typedef struct {
  REAL4 F;		/**< F-statistic value */
  COMPLEX8 Fa;		/**< complex amplitude Fa */
  COMPLEX8 Fb;		/**< complex amplitude Fb */
} FcomponentsREAL4;

/**
 * Struct holding buffered XLALDriverFstatREAL4()-internal quantities
 * to avoid unnecessarily recomputing things that depend ONLY on the skyposition and detector-state series
 * (but not on the spins).
 */
typedef struct {
  REAL8 Alpha, Delta;					/**< target skyposition of previous search */
  const MultiDetectorStateSeries *multiDetStates;	/**< input detStates series used in previous search */
  MultiSSBtimesREAL4 *multiSSB;				/**< SSB timings computed in previous search */
  MultiAMCoeffs *multiAMcoef;				/**< antenna-pattern coeffs computed in previous search */
} ComputeFBufferREAL4;

/**
 * Struct holding buffered XLALComputeFStatFreqBandVector()-internal quantities
 * to avoid unnecessarily recomputing things that depend ONLY on the skyposition and detector-state series
 * (but not on the spins).
 */
typedef struct {
  REAL8 Alpha, Delta;						/**< target skyposition of previous search */
  const MultiDetectorStateSeriesSequence *multiDetStatesV;	/**< input detStates series used in previous search */
  UINT4 numSegments;						/**< number of segments */
  MultiSSBtimesREAL4 **multiSSB4V;				/**< array[numSegments] of SSB timings computed in previous search */
  MultiAMCoeffs **multiAMcoefV;					/**< array[numSegments] of antenna-pattern coeffs computed in previous search */
} ComputeFBufferREAL4V;

/* ---------- exported global variables ---------- */
extern const ComputeFBufferREAL4 empty_ComputeFBufferREAL4;
extern const ComputeFBufferREAL4V empty_ComputeFBufferREAL4V;

/* ---------- exported API prototypes ---------- */
int
XLALComputeFStatFreqBandVector ( REAL4FrequencySeriesVector *fstatBandV,
				 const PulsarDopplerParams *doppler,
				 const MultiSFTVectorSequence *multiSFTsV,
				 const MultiNoiseWeightsSequence *multiWeightsV,
				 const MultiDetectorStateSeriesSequence *multiDetStatesV,
				 UINT4 Dterms,
				 ComputeFBufferREAL4V *cfvBuffer
				 );

int
XLALDriverFstatREAL4 ( REAL4 *Fstat,
                     const PulsarDopplerParams *doppler,
                     const MultiSFTVector *multiSFTs,
                     const MultiNoiseWeights *multiWeights,
                     const MultiDetectorStateSeries *multiDetStates,
                     UINT4 Dterms,
                     ComputeFBufferREAL4 *cfBuffer
                     );

void
XLALCoreFstatREAL4 (REAL4 *Fstat,
                  PulsarSpinsREAL4 *fkdot4,
                  const MultiSFTVector *multiSFTs,
                  MultiSSBtimesREAL4 *multiSSB4,
                  MultiAMCoeffs *multiAMcoef,
                  UINT4 Dterms
                  );


int
XLALComputeFaFbREAL4 ( FcomponentsREAL4 *FaFb,
                       const SFTVector *sfts,
                       const PulsarSpinsREAL4 *fkdot4,
                       const SSBtimesREAL4 *tSSB,
                       const AMCoeffs *amcoe,
                       UINT4 Dterms);


MultiSSBtimesREAL4 *
XLALGetMultiSSBtimesREAL4 ( const MultiDetectorStateSeries *multiDetStates,
                            REAL8 Alpha, REAL8 Delta,
                            LIGOTimeGPS refTime
                            );

SSBtimesREAL4 *
XLALGetSSBtimesREAL4 ( const DetectorStateSeries *DetectorStates,
                       REAL8 Alpha, REAL8 Delta,
                       LIGOTimeGPS refTime
                       );

void XLALDestroySSBtimesREAL4 ( SSBtimesREAL4 *tSSB );
void XLALDestroyMultiSSBtimesREAL4 ( MultiSSBtimesREAL4 *multiSSB );

void XLALEmptyComputeFBufferREAL4 ( ComputeFBufferREAL4 *cfb);
void XLALEmptyComputeFBufferREAL4V ( ComputeFBufferREAL4V *cfbv);

#ifdef USE_CUDA
  extern int InitializeCUDADevice(MultiSFTVectorSequence *stackMultiSFT, REAL4FrequencySeriesVector *fstatVector);
  extern void UnInitializeCUDADevice(void);
  extern void RearrangeSFTData4CUDA(REAL4FrequencySeriesVector *fstatBandV,
				 const PulsarDopplerParams *doppler,
				 const MultiSFTVectorSequence *multiSFTsV,
				 const MultiNoiseWeightsSequence *multiWeightsV,
				 const MultiDetectorStateSeriesSequence *multiDetStatesV,
				 UINT4 Dterms,
				 ComputeFBufferREAL4V *cfvBuffer);
#elif USE_OPENCL
#include "ComputeFstatREAL4OpenCL.h"
#endif

#ifdef  __cplusplus
}
#endif
/* C++ protection. */

#endif  /* Double-include protection. */

/*
 *  Copyright (C) 2013,2014,2015 Michael Puerrer, Alejandro Bohe
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

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_trig.h>

#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALAtomicDatatypes.h>
#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/LALSimInspiral.h>
#include <lal/Units.h>
#include <lal/XLALError.h>
#include <lal/SphericalHarmonics.h>
#include <lal/Sequence.h>
#include <lal/LALStdlib.h>
#include <lal/LALStddef.h>

#include "LALSimIMR.h"

#include "LALSimIMRPhenomP.h"
#include "LALSimIMRPhenomP_shared.hh"

#if defined(LALSIMULATION_CUDA_ENABLED)

#define N_LALSIM_CUDA_STREAMS_DEFAULT 0

static void set_streams(UINT4 L_fCut, UINT4 n_streams_alloc, UINT4 *n_streams, UINT4 *stream_size, UINT4 *stream_offset){

    // Make sure that every stream has some work
    (*n_streams) = n_streams_alloc; 
    if((*n_streams)>L_fCut)
        (*n_streams)=L_fCut;

    if(*n_streams>0){
    // Compute the stream array offsets and sizes
        UINT4  i_stream        = 0;
        UINT4  stream_offset_i = 0;
        UINT4  stream_size_i   = L_fCut/(*n_streams);
        for(i_stream=0;i_stream<(*n_streams);i_stream++,stream_offset_i+=stream_size_i){
            // Set stream size
            if(i_stream==((*n_streams)-1))
                stream_size_i=L_fCut-stream_offset_i;
            else
                stream_size_i=(L_fCut-stream_offset_i)/((*n_streams)-i_stream);
            stream_size[i_stream]  =stream_size_i;

            // Set stream offset
            stream_offset[i_stream]=stream_offset_i;
        }
    }
}

static void alloc_streams(UINT4 L_fCut_max, UINT4 n_streams_alloc, UINT4 *n_streams, cudaStream_t **stream, UINT4 **stream_size,UINT4 **stream_offset){
    if(n_streams_alloc>0){
        (*stream)        = (cudaStream_t *)malloc(n_streams_alloc*sizeof(cudaStream_t));
        (*stream_size)   = (UINT4 *)malloc(n_streams_alloc*sizeof(UINT4));
        (*stream_offset) = (UINT4 *)malloc(n_streams_alloc*sizeof(UINT4));

        // Create the streams
        UINT4  i_stream = 0;
        for(i_stream=0;i_stream<n_streams_alloc;i_stream++)
            throw_on_cuda_error(cudaStreamCreate(&((*stream)[i_stream])),lalsimulation_cuda_exception::STREAM_CREATE);

        // Set the streams to the simplest initial state of L_fCut=L_fCut_max
        set_streams(L_fCut_max, n_streams_alloc, n_streams, (*stream_size), (*stream_offset));
    }
    else{
        (*n_streams)     = 0;
        (*stream)        = NULL;
        (*stream_size)   = NULL;
        (*stream_offset) = NULL;
    }
}

static void free_streams(UINT4 *n_streams_alloc,UINT4 *n_streams,cudaStream_t **stream, UINT4 **stream_size,UINT4 **stream_offset){
    if((*n_streams_alloc)>0){
        UINT4 i_stream = 0;
        for(i_stream=0;i_stream<(*n_streams_alloc);i_stream++)
           throw_on_cuda_error(cudaStreamDestroy((*stream)[i_stream]),lalsimulation_cuda_exception::STREAM_DESTROY);
        free((*stream));
        free((*stream_size));
        free((*stream_offset));
    }
    (*n_streams)      =0;
    (*n_streams_alloc)=0;
    (*stream)         =NULL;
    (*stream_size)    =NULL;
    (*stream_offset)  =NULL;
}

// If n_streams_alloc==0, then a synchronous calculation will be performed.  Otherwise, the calculation
// will be conducted with n_streams_alloc asynchronous streams.
LALSIMULATION_CUDA_HOST PhenomPCore_buffer_info *XLALPhenomPCore_buffer(const UINT4 L_fCut_max, const INT4 n_streams_alloc_in, const bool auto_offload){

    PhenomPCore_buffer_info *buf = (PhenomPCore_buffer_info *)malloc(sizeof(PhenomPCore_buffer_info));

    // This bool indicates whether results should be automatically offloaded after they are generated
    // The user may not want to do this because they might want to do some on-device processing
    // afterwards.  This saves the performance hit from an unnecessary device->host communication.
    buf->auto_offload = auto_offload;

    // Set the number of streams that we will allocate.
    UINT4 n_streams_alloc = N_LALSIM_CUDA_STREAMS_DEFAULT;
    if(n_streams_alloc_in<0)
        n_streams_alloc = N_LALSIM_CUDA_STREAMS_DEFAULT;
    else
        n_streams_alloc = n_streams_alloc_in;

    buf->init        =true;
    buf->L_fCut_alloc=L_fCut_max;
    buf->offset      =0;

    // Establish a device context
    throw_on_cuda_error(cudaFree(0),lalsimulation_cuda_exception::INIT);

    // Initialize cuda streams
    buf->n_streams_alloc = n_streams_alloc;
    alloc_streams(L_fCut_max,buf->n_streams_alloc,&(buf->n_streams),&(buf->stream),&(buf->stream_size),&(buf->stream_offset));

    // Initialize pinned memory
    if(buf->n_streams_alloc>0){
        throw_on_cuda_error(cudaHostAlloc(&(buf->freqs_pinned),  L_fCut_max*sizeof(REAL8),    cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
        throw_on_cuda_error(cudaHostAlloc(&(buf->hctilde_pinned),L_fCut_max*sizeof(COMPLEX16),cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
        throw_on_cuda_error(cudaHostAlloc(&(buf->hptilde_pinned),L_fCut_max*sizeof(COMPLEX16),cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
        throw_on_cuda_error(cudaHostAlloc(&(buf->phis_pinned),   L_fCut_max*sizeof(REAL8),    cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
    }
    else{
        buf->freqs_pinned   = NULL;
        buf->hctilde_pinned = NULL;
        buf->hptilde_pinned = NULL;
        buf->phis_pinned    = NULL;
    }

    // Initialize offload arrays if auto-offload is off
    if(!buf->auto_offload){
        // This is a bit of an ugly kludge; no error checking and use of XLALCreate...FrequencySeries is overkill
        // ... however, it's what I had to do to get the SWIG wrappers to yeild an object that could be accessed by Python
        // Ideally, we would just use COMPLEX16 and REAL8 arrays
        LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; 
        buf->hptilde_offload = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &ligotimegps_zero, 0.0, 0., &lalStrainUnit, L_fCut_max);
        buf->hctilde_offload = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &ligotimegps_zero, 0.0, 0., &lalStrainUnit, L_fCut_max);
        buf->phis_offload    = XLALCreateREAL8FrequencySeries    ("phi: FD waveform",     &ligotimegps_zero, 0.0, 0., &lalStrainUnit, L_fCut_max);
    }
    else{
        buf->hctilde_offload = NULL;
        buf->hptilde_offload = NULL;
        buf->phis_offload    = NULL;
    }

    // Initialize device memory for input/output arrays
    throw_on_cuda_error(cudaMalloc(&(buf->freqs),  L_fCut_max*sizeof(REAL8)),    lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->hptilde),L_fCut_max*sizeof(COMPLEX16)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->hctilde),L_fCut_max*sizeof(COMPLEX16)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->phis),   L_fCut_max*sizeof(REAL8)),    lalsimulation_cuda_exception::MALLOC);

    // Initialize device memory for other inputs
    throw_on_cuda_error(cudaMalloc(&(buf->pAmp),          sizeof(IMRPhenomDAmplitudeCoefficients)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->pPhi),          sizeof(IMRPhenomDPhaseCoefficients)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->PCparams),      sizeof(BBHPhenomCParams)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->pn),            sizeof(PNPhasingSeries)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->angcoeffs),     sizeof(NNLOanglecoeffs)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->Y2m),           sizeof(SpinWeightedSphericalHarmonic_l2)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->amp_prefactors),sizeof(AmpInsPrefactors)),lalsimulation_cuda_exception::MALLOC);
    throw_on_cuda_error(cudaMalloc(&(buf->phi_prefactors),sizeof(PhiInsPrefactors)),lalsimulation_cuda_exception::MALLOC);

    return(buf);
}

LALSIMULATION_CUDA_HOST void XLALfree_PhenomPCore_buffer(PhenomPCore_buffer_info *buf){
    if(buf!=NULL){
        if(buf->init){
            // Free streams
            free_streams(&(buf->n_streams_alloc),&(buf->n_streams),&(buf->stream),&(buf->stream_size),&(buf->stream_offset));

            // Free inputs/outputs
            if(buf->n_streams_alloc>0){
                throw_on_cuda_error(cudaFreeHost(buf->freqs_pinned),  lalsimulation_cuda_exception::FREE);
                throw_on_cuda_error(cudaFreeHost(buf->hptilde_pinned),lalsimulation_cuda_exception::FREE);
                throw_on_cuda_error(cudaFreeHost(buf->hctilde_pinned),lalsimulation_cuda_exception::FREE);
                throw_on_cuda_error(cudaFreeHost(buf->phis_pinned),   lalsimulation_cuda_exception::FREE);
            }
            else{
                buf->freqs_pinned  =NULL;
                buf->hptilde_pinned=NULL;
                buf->hctilde_pinned=NULL;
                buf->phis_pinned   =NULL;
            }
            // Free offload arrays
            if(!buf->auto_offload){
                XLALDestroyCOMPLEX16FrequencySeries(buf->hptilde_offload);
                XLALDestroyCOMPLEX16FrequencySeries(buf->hctilde_offload);
                XLALDestroyREAL8FrequencySeries(buf->phis_offload);
            }
            else{
                buf->hctilde_offload=NULL;
                buf->hptilde_offload=NULL;
                buf->phis_offload   =NULL;
            }
            throw_on_cuda_error(cudaFree(buf->phis),              lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->hctilde),           lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->hptilde),           lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->phi_prefactors),    lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->amp_prefactors),    lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->Y2m),               lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->angcoeffs),         lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->pn),                lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->PCparams),          lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->pPhi),              lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->pAmp),              lalsimulation_cuda_exception::FREE);
            throw_on_cuda_error(cudaFree(buf->freqs),             lalsimulation_cuda_exception::FREE);
            buf->phis          =NULL;
            buf->hctilde       =NULL;
            buf->hptilde       =NULL;
            buf->phi_prefactors=NULL;
            buf->amp_prefactors=NULL;
            buf->Y2m           =NULL;
            buf->angcoeffs     =NULL;
            buf->pn            =NULL;
            buf->PCparams      =NULL;
            buf->pPhi          =NULL;
            buf->pAmp          =NULL;
            buf->freqs         =NULL;
            buf->init          =false;
        }
    }
}

LALSIMULATION_CUDA_HOST void XLALoffload_PhenomPCore_buffer(PhenomPCore_buffer_info *buf){
    if(buf!=NULL){
        throw_on_cuda_error(cudaMemcpy(buf->hptilde_offload->data->data,buf->hptilde,buf->L_fCut_alloc*sizeof(COMPLEX16),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
        throw_on_cuda_error(cudaMemcpy(buf->hctilde_offload->data->data,buf->hctilde,buf->L_fCut_alloc*sizeof(COMPLEX16),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
        throw_on_cuda_error(cudaMemcpy(buf->phis_offload->data->data,   buf->phis,   buf->L_fCut_alloc*sizeof(REAL8),    cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
        if(buf->offset!=0){
            INT4 j=buf->L_fCut-1;
            INT4 i=j-buf->offset;
            for(;i>=0;i--,j--){
                buf->hptilde_offload->data->data[j]=buf->hptilde_offload->data->data[i];
                buf->hctilde_offload->data->data[j]=buf->hctilde_offload->data->data[i];
                buf->phis_offload->data[j]         =buf->phis_offload->data[i];
            }
        }
    }
}

void PhenomPCoreAllFrequencies_cuda(UINT4 L_fCut,
        REAL8Sequence *freqs_host,
        UINT4 offset,
        const REAL8 eta,
        const REAL8 chi1_l,
        const REAL8 chi2_l,
        const REAL8 chip,
        const REAL8 distance,
        const REAL8 M,
        const REAL8 phic,
        IMRPhenomDAmplitudeCoefficients *pAmp_host,
        IMRPhenomDPhaseCoefficients *pPhi_host,
        BBHPhenomCParams *PCparams_host,
        PNPhasingSeries *pn_host,
        NNLOanglecoeffs *angcoeffs_host,
        SpinWeightedSphericalHarmonic_l2 *Y2m_host,
        const REAL8 alphaNNLOoffset,
        const REAL8 alpha0,
        const REAL8 epsilonNNLOoffset,
        IMRPhenomP_version_type IMRPhenomP_version,
        AmpInsPrefactors *amp_prefactors_host,
        PhiInsPrefactors *phi_prefactors_host,
        COMPLEX16FrequencySeries *hptilde_host,
        COMPLEX16FrequencySeries *hctilde_host,
        REAL8 *phis_host,
        const void *buf_in,
        int   *errcode){

  // Arg!  Circular dependancies galore when I tried to pass this by type
  // correctly, so we do this trick instead (for now at least)
  PhenomPCore_buffer_info *buf = (PhenomPCore_buffer_info *)buf_in;

  // Check if the results should be offloaded
  bool offload = true;
  if(buf!=NULL)
     offload = buf->auto_offload;

  try{

    // Initialize streams
    cudaStream_t *stream=NULL;
    UINT4        *stream_size=NULL;
    UINT4        *stream_offset=NULL;
    UINT4         n_streams=0;
    UINT4         n_streams_alloc=0;
    if(buf!=NULL){
       n_streams      =buf->n_streams;
       n_streams_alloc=buf->n_streams_alloc;
       stream         =buf->stream;
       stream_size    =buf->stream_size;
       stream_offset  =buf->stream_offset;

       // Because L_fCut can vary in size between each run, we need
       // to recompute the stream offsets, sizes, etc every time
       set_streams(L_fCut,n_streams_alloc,&n_streams,stream_size,stream_offset);
    }
    else{
       n_streams_alloc = N_LALSIM_CUDA_STREAMS_DEFAULT;
       alloc_streams(L_fCut,n_streams_alloc,&n_streams,&stream,&stream_size,&stream_offset);
    }

    // We use pinned versions of the input & output arrays
    // to enable asynchronous implementations
    REAL8     *freqs_pinned  =NULL;
    COMPLEX16 *hptilde_pinned=NULL;
    COMPLEX16 *hctilde_pinned=NULL;
    REAL8     *phis_pinned   =NULL;
    REAL8     *freqs         =NULL;
    COMPLEX16 *hptilde       =NULL;
    COMPLEX16 *hctilde       =NULL;
    REAL8     *phis          =NULL;
    if(buf!=NULL){
        // Check that the buffer is sufficiently allocated
        if(buf->L_fCut_alloc<L_fCut){
            printf("Error: buf->L_fCut_alloc<L_fCut (ie %d<%d).",buf->L_fCut_alloc,L_fCut);
            exit(1);
        }

        // Store the L_fCut used for this call in the buffer
        buf->L_fCut=L_fCut;

        // Fetch pointers from buffer
        if(n_streams>0){
            freqs_pinned  =buf->freqs_pinned;
            hptilde_pinned=buf->hptilde_pinned;
            hctilde_pinned=buf->hctilde_pinned;
            phis_pinned   =buf->phis_pinned;
        }
        freqs  =buf->freqs;
        hptilde=buf->hptilde;
        hctilde=buf->hctilde;
        phis   =buf->phis;
    }
    else{
        // Establish a device context
        throw_on_cuda_error(cudaFree(0),lalsimulation_cuda_exception::INIT);

        // Allocate buffer arrays
        if(n_streams>0){
            throw_on_cuda_error(cudaHostAlloc(&freqs_pinned,  L_fCut*sizeof(REAL8),    cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
            throw_on_cuda_error(cudaHostAlloc(&hctilde_pinned,L_fCut*sizeof(COMPLEX16),cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
            throw_on_cuda_error(cudaHostAlloc(&hptilde_pinned,L_fCut*sizeof(COMPLEX16),cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
            throw_on_cuda_error(cudaHostAlloc(&phis_pinned,   L_fCut*sizeof(REAL8),    cudaHostAllocDefault),lalsimulation_cuda_exception::MALLOC);
        }
        throw_on_cuda_error(cudaMalloc(&freqs,  L_fCut*sizeof(REAL8)),    lalsimulation_cuda_exception::MALLOC);
        throw_on_cuda_error(cudaMalloc(&hptilde,L_fCut*sizeof(COMPLEX16)),lalsimulation_cuda_exception::MALLOC);
        throw_on_cuda_error(cudaMalloc(&hctilde,L_fCut*sizeof(COMPLEX16)),lalsimulation_cuda_exception::MALLOC);
        throw_on_cuda_error(cudaMalloc(&phis,   L_fCut*sizeof(REAL8)),    lalsimulation_cuda_exception::MALLOC);
    }

    // ----------------------------------
    // Copy input parameters host->device
    IMRPhenomDAmplitudeCoefficients *pAmp=NULL;
    if(pAmp_host!=NULL){
      if(buf!=NULL)
          pAmp=buf->pAmp;
      else
          throw_on_cuda_error(cudaMalloc(&pAmp,sizeof(IMRPhenomDAmplitudeCoefficients)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(pAmp,pAmp_host,sizeof(IMRPhenomDAmplitudeCoefficients),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    IMRPhenomDPhaseCoefficients *pPhi=NULL;
    if(pPhi_host!=NULL){
      if(buf!=NULL)
          pPhi=buf->pPhi;
      else
          throw_on_cuda_error(cudaMalloc(&pPhi,sizeof(IMRPhenomDPhaseCoefficients)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(pPhi,pPhi_host,sizeof(IMRPhenomDPhaseCoefficients),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    BBHPhenomCParams *PCparams=NULL;
    if(PCparams_host!=NULL){
      if(buf!=NULL)
          PCparams=buf->PCparams;
      else
          throw_on_cuda_error(cudaMalloc(&PCparams,sizeof(BBHPhenomCParams)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(PCparams,PCparams_host,sizeof(BBHPhenomCParams),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    PNPhasingSeries *pn=NULL;
    if(pn_host!=NULL){
      if(buf!=NULL)
          pn=buf->pn;
      else
          throw_on_cuda_error(cudaMalloc(&pn,sizeof(PNPhasingSeries)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(pn, pn_host,sizeof(PNPhasingSeries), cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    NNLOanglecoeffs *angcoeffs=NULL;
    if(angcoeffs_host!=NULL){
      if(buf!=NULL)
          angcoeffs=buf->angcoeffs;
      else
          throw_on_cuda_error(cudaMalloc(&angcoeffs,sizeof(NNLOanglecoeffs)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(angcoeffs,angcoeffs_host,sizeof(NNLOanglecoeffs),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    SpinWeightedSphericalHarmonic_l2 *Y2m=NULL;
    if(Y2m_host!=NULL){
      if(buf!=NULL)
          Y2m=buf->Y2m;
      else
          throw_on_cuda_error(cudaMalloc(&Y2m,sizeof(SpinWeightedSphericalHarmonic_l2)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(Y2m,Y2m_host,sizeof(SpinWeightedSphericalHarmonic_l2),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    AmpInsPrefactors *amp_prefactors=NULL;
    if(amp_prefactors_host!=NULL){
      if(buf!=NULL)
          amp_prefactors=buf->amp_prefactors;
      else
          throw_on_cuda_error(cudaMalloc(&amp_prefactors,sizeof(AmpInsPrefactors)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(amp_prefactors,amp_prefactors_host,sizeof(AmpInsPrefactors),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }

    PhiInsPrefactors *phi_prefactors=NULL;
    if(phi_prefactors_host!=NULL){
      if(buf!=NULL)
          phi_prefactors=buf->phi_prefactors;
      else
          throw_on_cuda_error(cudaMalloc(&phi_prefactors,sizeof(PhiInsPrefactors)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMemcpy(phi_prefactors,phi_prefactors_host,sizeof(PhiInsPrefactors),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
    }
    // ----------------------------------

    // Set the number of threads per block.  Profiling
    // has revealed that this kernel uses 72-registers.
    // This number of threads is appropriate in that
    // case for a P100.
    int n_threads=16;

    // Perform the GPU work in chunks using asynchronous streams
    if(n_streams>0){
        // Load the pinned input buffer
        throw_on_cuda_error(cudaMemcpy(freqs_pinned,freqs_host->data,L_fCut*sizeof(REAL8),cudaMemcpyHostToHost),lalsimulation_cuda_exception::MEMCPY);

        UINT4 i_stream = 0;
        for(i_stream=0;i_stream<n_streams;i_stream++){
            // If we are using one-or-more stream(s), we need to
            // work with the pinned arrays and the appropriate
            // stream sizes and stream offsets
            UINT4        stream_offset_i = stream_offset[i_stream];
            UINT4        stream_size_i   = stream_size[i_stream];
            cudaStream_t stream_i        = stream[i_stream];

            // Perform host->device transfer (via the pinned array)
            throw_on_cuda_error(cudaMemcpyAsync(&(freqs[stream_offset_i]),&(freqs_pinned[stream_offset_i]),stream_size_i*sizeof(REAL8),cudaMemcpyHostToDevice,stream_i),lalsimulation_cuda_exception::MEMCPY);

            // Run kernel
            int grid_size=(stream_size_i+(n_threads-1))/n_threads;
            throw_on_kernel_error((PhenomPCoreOneFrequency_cuda<<<grid_size,n_threads,0,stream_i>>>(
                  L_fCut,
                  freqs,
                  stream_offset_i,
                  eta,
                  chi1_l,
                  chi2_l,
                  chip,
                  distance,
                  M,
                  phic,
                  pAmp,
                  pPhi,
                  PCparams,
                  pn,
                  angcoeffs,
                  Y2m,
                  alphaNNLOoffset,
                  alpha0,
                  epsilonNNLOoffset,
                  IMRPhenomP_version,
                  amp_prefactors,
                  phi_prefactors,
                  hptilde,
                  hctilde,
                  phis)),lalsimulation_cuda_exception::KERNEL_PHENOMPCOREONEFREQUENCY);

            // Perform device->host transfer of results
            if(offload){
                throw_on_cuda_error(cudaMemcpyAsync(&(hptilde_pinned[stream_offset_i]),&(hptilde[stream_offset_i]),stream_size_i*sizeof(COMPLEX16),cudaMemcpyDeviceToHost,stream_i),lalsimulation_cuda_exception::MEMCPY);
                throw_on_cuda_error(cudaMemcpyAsync(&(hctilde_pinned[stream_offset_i]),&(hctilde[stream_offset_i]),stream_size_i*sizeof(COMPLEX16),cudaMemcpyDeviceToHost,stream_i),lalsimulation_cuda_exception::MEMCPY);
                throw_on_cuda_error(cudaMemcpyAsync(&(phis_pinned[stream_offset_i]),   &(phis[stream_offset_i]),   stream_size_i*sizeof(REAL8),    cudaMemcpyDeviceToHost,stream_i),lalsimulation_cuda_exception::MEMCPY);        
            }
        }

        // Unload the pinned output buffers
        if(offload){
            throw_on_cuda_error(cudaMemcpy(hptilde_host->data->data,hptilde_pinned,L_fCut*sizeof(COMPLEX16),cudaMemcpyHostToHost),lalsimulation_cuda_exception::MEMCPY);
            throw_on_cuda_error(cudaMemcpy(hctilde_host->data->data,hctilde_pinned,L_fCut*sizeof(COMPLEX16),cudaMemcpyHostToHost),lalsimulation_cuda_exception::MEMCPY);
            throw_on_cuda_error(cudaMemcpy(phis_host,               phis_pinned,   L_fCut*sizeof(REAL8),    cudaMemcpyHostToHost),lalsimulation_cuda_exception::MEMCPY);
        }

    }
    // ... else, we can bypass the pinned arrays
    else{
        // Perform host->device transfer
        throw_on_cuda_error(cudaMemcpy(freqs,freqs_host->data,L_fCut*sizeof(REAL8),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);

        // Run kernel
        int grid_size=(L_fCut+(n_threads-1))/n_threads;
        throw_on_kernel_error((PhenomPCoreOneFrequency_cuda<<<grid_size,n_threads>>>(
              L_fCut,
              freqs,
              0,
              eta,
              chi1_l,
              chi2_l,
              chip,
              distance,
              M,
              phic,
              pAmp,
              pPhi,
              PCparams,
              pn,
              angcoeffs,
              Y2m,
              alphaNNLOoffset,
              alpha0,
              epsilonNNLOoffset,
              IMRPhenomP_version,
              amp_prefactors,
              phi_prefactors,
              hptilde,
              hctilde,
              phis)),lalsimulation_cuda_exception::KERNEL_PHENOMPCOREONEFREQUENCY);

        // Perform device->host transfer of results
        if(offload){
            throw_on_cuda_error(cudaMemcpy(hptilde_host->data->data,hptilde,L_fCut*sizeof(COMPLEX16),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
            throw_on_cuda_error(cudaMemcpy(hctilde_host->data->data,hctilde,L_fCut*sizeof(COMPLEX16),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
            throw_on_cuda_error(cudaMemcpy(phis_host,               phis,   L_fCut*sizeof(REAL8),    cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
        }
    }

    // Shift index for frequency series if needed.  This wouldn't work in the
    // kernel due to async implementation, so it needs to be done here.
    if(buf!=NULL)
        buf->offset=offset;
    if(offload){
        if(offset!=0){
            INT4 j=L_fCut-1;
            INT4 i=j-offset;
            for(;i>=0;i--,j--){
                hptilde_host->data->data[j]=hptilde_host->data->data[i];
                hctilde_host->data->data[j]=hctilde_host->data->data[i];
                phis_host[j]               =phis_host[i];
            }
        }
    }

    // Clean-up
    if(buf==NULL){
        if(freqs_pinned!=NULL)   throw_on_cuda_error(cudaFreeHost(freqs_pinned),  lalsimulation_cuda_exception::FREE);
        if(hptilde_pinned!=NULL) throw_on_cuda_error(cudaFreeHost(hptilde_pinned),lalsimulation_cuda_exception::FREE);
        if(hctilde_pinned!=NULL) throw_on_cuda_error(cudaFreeHost(hctilde_pinned),lalsimulation_cuda_exception::FREE);
        if(phis_pinned!=NULL)    throw_on_cuda_error(cudaFreeHost(phis_pinned),   lalsimulation_cuda_exception::FREE);
        if(phis!=NULL)           throw_on_cuda_error(cudaFree(phis),              lalsimulation_cuda_exception::FREE);
        if(hctilde!=NULL)        throw_on_cuda_error(cudaFree(hctilde),           lalsimulation_cuda_exception::FREE);
        if(hptilde!=NULL)        throw_on_cuda_error(cudaFree(hptilde),           lalsimulation_cuda_exception::FREE);
        if(phi_prefactors!=NULL) throw_on_cuda_error(cudaFree(phi_prefactors),    lalsimulation_cuda_exception::FREE);
        if(amp_prefactors!=NULL) throw_on_cuda_error(cudaFree(amp_prefactors),    lalsimulation_cuda_exception::FREE);
        if(Y2m!=NULL)            throw_on_cuda_error(cudaFree(Y2m),               lalsimulation_cuda_exception::FREE);
        if(angcoeffs!=NULL)      throw_on_cuda_error(cudaFree(angcoeffs),         lalsimulation_cuda_exception::FREE);
        if(pn!=NULL)             throw_on_cuda_error(cudaFree(pn),                lalsimulation_cuda_exception::FREE);
        if(PCparams!=NULL)       throw_on_cuda_error(cudaFree(PCparams),          lalsimulation_cuda_exception::FREE);
        if(pPhi!=NULL)           throw_on_cuda_error(cudaFree(pPhi),              lalsimulation_cuda_exception::FREE);
        if(pAmp!=NULL)           throw_on_cuda_error(cudaFree(pAmp),              lalsimulation_cuda_exception::FREE);
        if(freqs!=NULL)          throw_on_cuda_error(cudaFree(freqs),             lalsimulation_cuda_exception::FREE);
        UINT4 i_stream;
        for(i_stream=0;i_stream<n_streams_alloc;i_stream++)
           throw_on_cuda_error(cudaStreamDestroy(stream[i_stream]),lalsimulation_cuda_exception::STREAM_DESTROY);
        if(stream_size  !=NULL) free(stream_size);
        if(stream_offset!=NULL) free(stream_offset);
        if(stream       !=NULL) free(stream);
        n_streams       = 0;
        n_streams_alloc = 0;
    }
  }
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }
}

// Call this function in kernels to put the GPU in an error state that can be caught after as an exception
//    This is not necessarily the best way, but it will do the job for now.  This is based on:
// https://devtalk.nvidia.com/default/topic/418479/how-to-trigger-a-cuda-error-from-inside-a-kernel/
// We accept an error code and do something useless with so that the functionality is there
// for the day when we come-up with a better way of throwing and reporting CUDA kernel errors.
__device__ void inline cause_cuda_error(const int error_code){
   int *adr = (int*)0xffffffff;
   *adr = 12; // This should induce an error state
   *adr = error_code; // This is to prevent a compiler warning
}

// These functions deal with any GPU exceptions, but should be called with the macros defined in the corresponding .hh file
__host__ void _throw_on_generic_error(bool check_failure,int implementation_code, const std::string file, const std::string func, int line)
{
  if(check_failure) throw(lalsimulation_cuda_exception(GENERIC_CUDA_ERROR_CODE,implementation_code,file,func,line));
}
__host__ void _throw_on_cuda_error(cudaError_t cuda_code, int implementation_code, const std::string file, const std::string func, int line)
{
  if(cuda_code != cudaSuccess) throw(lalsimulation_cuda_exception((int)cuda_code,implementation_code,file,func,line));
}
__host__ void _check_for_cuda_error(int implementation_code,const std::string file, const std::string func, int line)
{
  try{
    cudaError_t cuda_code = cudaPeekAtLastError();
    if(cuda_code != cudaSuccess)
        throw(lalsimulation_cuda_exception((int)cuda_code,implementation_code,"CUDA error detected after ",file,func,line));
  }
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }
}
__host__ void _check_thread_sync(int implementation_code,const std::string file, const std::string func, int line)
{
  try{
    cudaError_t cuda_code = cudaDeviceSynchronize();
    if(cuda_code != cudaSuccess)
        throw(lalsimulation_cuda_exception((int)cuda_code,implementation_code,"Threads not synchronised after ",file,func,line));
  }
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }
}
// This is here for any future MPI implementations, if needed
__host__ void _throw_on_global_error(const std::string file, const std::string func, int line)
{
  (void)line; // to avoid a compiler warning
  //int error_code=0;
  //MPI_Allreduce(MPI_IN_PLACE,&error_code,1,MPI_INT,MPI_MAX,run_globals.mpi_comm);
  //if(error_code!=0) throw(meraxes_cuda_exception(0,meraxes_cuda_exception::GLOBAL,file,func,line));
}
__host__ void notify_of_global_error(int error_code)
{
  (void)error_code; // to avoid a compiler warning
  //int result=(int)error_code;
  //MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,run_globals.mpi_comm);
}

__global__ void PhenomPCoreOneFrequency_cuda(
        UINT4 L_fCut,
        REAL8 *freqs,
        const UINT4 stream_offset,
        const REAL8 eta,
        const REAL8 chi1_l,
        const REAL8 chi2_l,
        const REAL8 chip,
        const REAL8 distance,
        const REAL8 M,
        const REAL8 phic,
        IMRPhenomDAmplitudeCoefficients *pAmp,
        IMRPhenomDPhaseCoefficients *pPhi,
        BBHPhenomCParams *PCparams,
        PNPhasingSeries *pn,
        NNLOanglecoeffs *angcoeffs,
        SpinWeightedSphericalHarmonic_l2 *Y2m,
        const REAL8 alphaNNLOoffset,
        const REAL8 alpha0,
        const REAL8 epsilonNNLOoffset,
        IMRPhenomP_version_type IMRPhenomP_version,
        AmpInsPrefactors *amp_prefactors,
        PhiInsPrefactors *phi_prefactors,
        COMPLEX16 *hptilde,
        COMPLEX16 *hctilde,
        REAL8 *phis){
    UINT4 i = (UINT4)(blockIdx.x*blockDim.x + threadIdx.x + stream_offset);
    if(i < L_fCut){

      COMPLEX16 hp_val;
      COMPLEX16 hc_val;
      REAL8     phasing;
      REAL8     f = freqs[i];
  
      // Generate the waveform
      int per_thread_errcode = PhenomPCoreOneFrequency(f, eta, chi1_l, chi2_l, chip, distance, M, phic,
                                                       pAmp, pPhi, PCparams, pn, angcoeffs, Y2m,
                                                       alphaNNLOoffset - alpha0, epsilonNNLOoffset,
                                                       &hp_val, &hc_val, &phasing, IMRPhenomP_version, amp_prefactors, phi_prefactors);
  
      // Throw execption if necessary
      if (per_thread_errcode != XLAL_SUCCESS)
        cause_cuda_error(101010101);

      hptilde[i] = hp_val;
      hctilde[i] = hc_val;
      phis[i]    = phasing;

  }
}
#else
void *XLALPhenomPCore_buffer(const UINT4 L_fCut_max,const INT4 n_streams){
    (void)L_fCut_max; // to avoid a compiler warning
    (void)n_streams;  // to avoid a compiler warning
}
void XLALfree_PhenomPCore_buffer(void *buf){
    (void)(buf); // to avoid a compiler warning
}
#endif

/* Macro functions to rotate the components of a vector about an axis */
#define ROTATEZ(angle, vx, vy, vz)\
tmp1 = vx*cos(angle) - vy*sin(angle);\
tmp2 = vx*sin(angle) + vy*cos(angle);\
vx = tmp1;\
vy = tmp2

#define ROTATEY(angle, vx, vy, vz)\
tmp1 = vx*cos(angle) + vz*sin(angle);\
tmp2 = - vx*sin(angle) + vz*cos(angle);\
vx = tmp1;\
vz = tmp2

const double sqrt_6 = 2.44948974278317788;

/**
 * \f[
 * \newcommand{\hP}{h^\mathrm{P}}
 * \newcommand{\PAmp}{A^\mathrm{P}}
 * \newcommand{\PPhase}{\phi^\mathrm{P}}
 * \newcommand{\chieff}{\chi_\mathrm{eff}}
 * \newcommand{\chip}{\chi_\mathrm{p}}
 * \f]
 * Internal core function to calculate
 * plus and cross polarizations of the PhenomP model
 * for a single frequency.
 *
 * The general expression for the modes \f$\hP_{2m}(t)\f$
 * is given by Eq. 1 of arXiv:1308.3271.
 * We calculate the frequency domain l=2 plus and cross polarizations separately
 * for each m = -2, ... , 2.
 *
 * The expression of the polarizations times the \f$Y_{lm}\f$
 * in code notation are:
 * \f{equation*}{
 * \left(\tilde{h}_{2m}\right)_+ = e^{-2i \epsilon}
 * \left(e^{-i m \alpha} d^2_{-2,m} (-2Y_{2m})
 * + e^{+i m \alpha} d^2_{2,m} (-2Y_{2m})^*\right) \cdot \hP / 2 \,,
 * \f}
 * \f{equation*}{
 * \left(\tilde{h}_{2m}\right)_x = e^{-2i \epsilon}
 * \left(e^{-i m \alpha} d^2_{-2,m} (-2Y_{2m})
 * - e^{+i m \alpha} d^2_{2,m} (-2Y_{2m})^*\right) \cdot \hP / 2 \,,
 * \f}
 * where the \f$d^l_{m',m}\f$ are Wigner d-matrices evaluated at \f$-\beta\f$,
 * and \f$\hP\f$ is the Phenom[C,D] frequency domain model:
 * \f{equation*}{
 * \hP(f) = \PAmp(f) e^{-i \PPhase(f)} \,.
 * \f}
 *
 * Note that in arXiv:1308.3271, the angle \f$\beta\f$ (beta) is called iota.
 *
 * For IMRPhenomP(v1) we put all spin on the larger BH,
 * convention: \f$m_2 \geq m_1\f$.
 * Hence:
 * \f{eqnarray*}{
 * \chieff      &=& \left( m_1 \cdot \chi_1 + m_2 \cdot \chi_2 \right)/M \,,\\
 * \chi_l       &=& \chieff / m_2 \quad (\text{for } M=1) \,,\\
 * S_L          &=& m_2^2 \chi_l = m_2 \cdot M \cdot \chieff
 *               = \frac{q}{1+q} \cdot \chieff \quad (\text{for } M=1) \,.
 * \f}
 *
 * For IMRPhenomPv2 we use both aligned spins:
 * \f{equation*}{
 * S_L = \chi_1 \cdot m_1^2 + \chi_2 \cdot m_2^2 \,.
 * \f}
 *
 * For both IMRPhenomP(v1) and IMRPhenomPv2 we put the in-plane spin on the larger BH:
 * \f{equation*}{
 * S_\mathrm{perp} = \chip \cdot m_2^2
 * \f}
 * (perpendicular spin).
 */
LALSIMULATION_CUDA_HOST_DEVICE int PhenomPCoreOneFrequency(
  const REAL8 fHz,                            /**< Frequency (Hz) */
  const REAL8 eta,                            /**< Symmetric mass ratio */
  const REAL8 chi1_l,                         /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l,                         /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                           /**< Dimensionless spin in the orbital plane */
  const REAL8 distance,                       /**< Distance of source (m) */
  const REAL8 M,                              /**< Total mass (Solar masses) */
  const REAL8 phic,                           /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  IMRPhenomDAmplitudeCoefficients *pAmp,      /**< Internal IMRPhenomD amplitude coefficients */
  IMRPhenomDPhaseCoefficients *pPhi,          /**< Internal IMRPhenomD phase coefficients */
  BBHPhenomCParams *PCparams,                 /**< Internal PhenomC parameters */
  PNPhasingSeries *PNparams,                  /**< PN inspiral phase coefficients */
  NNLOanglecoeffs *angcoeffs,                 /**< Struct with PN coeffs for the NNLO angles */
  SpinWeightedSphericalHarmonic_l2 *Y2m,      /**< Struct of l=2 spherical harmonics of spin weight -2 */
  const REAL8 alphaoffset,                    /**< f_ref dependent offset for alpha angle (azimuthal precession angle) */
  const REAL8 epsilonoffset,                  /**< f_ref dependent offset for epsilon angle */
  COMPLEX16 *hp,                              /**< [out] plus polarization \f$\tilde h_+\f$ */
  COMPLEX16 *hc,                              /**< [out] cross polarization \f$\tilde h_x\f$ */
  REAL8 *phasing,                             /**< [out] overall phasing */
  IMRPhenomP_version_type IMRPhenomP_version, /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
  AmpInsPrefactors *amp_prefactors,           /**< pre-calculated (cached for saving runtime) coefficients for amplitude. See LALSimIMRPhenomD_internals.c*/
  PhiInsPrefactors *phi_prefactors            /**< pre-calculated (cached for saving runtime) coefficients for phase. See LALSimIMRPhenomD_internals.*/)
{
  XLAL_CHECK_CUDA(angcoeffs != NULL, XLAL_EFAULT);
  XLAL_CHECK_CUDA(hp != NULL, XLAL_EFAULT);
  XLAL_CHECK_CUDA(hc != NULL, XLAL_EFAULT);
  XLAL_CHECK_CUDA(Y2m != NULL, XLAL_EFAULT);
  XLAL_CHECK_CUDA(phasing != NULL, XLAL_EFAULT);

  REAL8 f = fHz*LAL_MTSUN_SI*M; /* Frequency in geometric units */

  REAL8 aPhenom = 0.0;
  REAL8 phPhenom = 0.0;
  int errcode = XLAL_SUCCESS;
  UsefulPowers powers_of_f;

  const REAL8 q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta)/(2.0*eta);
  const REAL8 m1 = 1.0/(1.0+q);       /* Mass of the smaller BH for unit total mass M=1. */
  const REAL8 m2 = q/(1.0+q);         /* Mass of the larger BH for unit total mass M=1. */
  const REAL8 Sperp = chip*(m2*m2);   /* Dimensionfull spin component in the orbital plane. S_perp = S_2_perp */
  REAL8 SL;                           /* Dimensionfull aligned spin. */
  const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l); /* effective spin for M=1 */

  /* Calculate Phenom amplitude and phase for a given frequency. */
  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      XLAL_CHECK_CUDA(PCparams != NULL, XLAL_EFAULT);
      errcode = IMRPhenomCGenerateAmpPhase( &aPhenom, &phPhenom, fHz, eta, PCparams );
      if( errcode != XLAL_SUCCESS ) XLAL_ERROR_CUDA(XLAL_EFUNC);
      SL = chi_eff*m2;        /* Dimensionfull aligned spin of the largest BH. SL = m2^2 chil = m2*M*chi_eff */
      break;
    case IMRPhenomPv2_V:
      XLAL_CHECK_CUDA(pAmp != NULL, XLAL_EFAULT);
      XLAL_CHECK_CUDA(pPhi != NULL, XLAL_EFAULT);
      XLAL_CHECK_CUDA(PNparams != NULL, XLAL_EFAULT);
      XLAL_CHECK_CUDA(amp_prefactors != NULL, XLAL_EFAULT);
      XLAL_CHECK_CUDA(phi_prefactors != NULL, XLAL_EFAULT);
      errcode = init_useful_powers(&powers_of_f, f);
      XLAL_CHECK_CUDA(errcode == XLAL_SUCCESS, errcode, "init_useful_powers failed for f");
      aPhenom = IMRPhenDAmplitude(f, pAmp, &powers_of_f, amp_prefactors);
      phPhenom = IMRPhenDPhase(f, pPhi, PNparams, &powers_of_f, phi_prefactors);
      SL = chi1_l*m1*m1 + chi2_l*m2*m2;        /* Dimensionfull aligned spin. */
      break;
  }

  const COMPLEX16 MY_I = complex<double>(0,1);
  phPhenom -= 2.*phic; /* Note: phic is orbital phase */
  REAL8 amp0 = M * LAL_MRSUN_SI * M * LAL_MTSUN_SI / distance;
  COMPLEX16 hP = amp0 * aPhenom * (cos(phPhenom) - MY_I * sin(phPhenom));//cexp(-I*phPhenom); /* Assemble IMRPhenom waveform. */

  /* Compute PN NNLO angles */
  const REAL8 omega = LAL_PI * f;
  const REAL8 logomega = log(omega);
  const REAL8 omega_cbrt = cbrt(omega);
  const REAL8 omega_cbrt2 = omega_cbrt*omega_cbrt;

  REAL8 alpha = (angcoeffs->alphacoeff1/omega
              + angcoeffs->alphacoeff2/omega_cbrt2
              + angcoeffs->alphacoeff3/omega_cbrt
              + angcoeffs->alphacoeff4*logomega
              + angcoeffs->alphacoeff5*omega_cbrt) - alphaoffset;

  REAL8 epsilon = (angcoeffs->epsiloncoeff1/omega
                + angcoeffs->epsiloncoeff2/omega_cbrt2
                + angcoeffs->epsiloncoeff3/omega_cbrt
                + angcoeffs->epsiloncoeff4*logomega
                + angcoeffs->epsiloncoeff5*omega_cbrt) - epsilonoffset;

  /* Calculate intermediate expressions cos(beta/2), sin(beta/2) and powers thereof for Wigner d's. */
  REAL8 cBetah, sBetah; /* cos(beta/2), sin(beta/2) */
  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      WignerdCoefficients_SmallAngleApproximation(&cBetah, &sBetah, omega_cbrt, SL, eta, Sperp);
      break;
    case IMRPhenomPv2_V:
      WignerdCoefficients(&cBetah, &sBetah, omega_cbrt, SL, eta, Sperp);
      break;
  }

  const REAL8 cBetah2 = cBetah*cBetah;
  const REAL8 cBetah3 = cBetah2*cBetah;
  const REAL8 cBetah4 = cBetah3*cBetah;
  const REAL8 sBetah2 = sBetah*sBetah;
  const REAL8 sBetah3 = sBetah2*sBetah;
  const REAL8 sBetah4 = sBetah3*sBetah;

  /* Compute Wigner d coefficients
    The expressions below agree with refX [Goldstein?] and Mathematica
    d2  = Table[WignerD[{2, mp, 2}, 0, -\[Beta], 0], {mp, -2, 2}]
    dm2 = Table[WignerD[{2, mp, -2}, 0, -\[Beta], 0], {mp, -2, 2}]
  */
  COMPLEX16 d2[5]   = {sBetah4, 2*cBetah*sBetah3, sqrt_6*sBetah2*cBetah2, 2*cBetah3*sBetah, cBetah4};
  COMPLEX16 dm2[5]  = {d2[4], -d2[3], d2[2], -d2[1], d2[0]}; /* Exploit symmetry d^2_{-2,-m} = (-1)^m d^2_{2,m} */

  COMPLEX16 Y2mA[5] = {Y2m->Y2m2, Y2m->Y2m1, Y2m->Y20, Y2m->Y21, Y2m->Y22};
  COMPLEX16 hp_sum = 0;
  COMPLEX16 hc_sum = 0;

  /* Sum up contributions to \tilde h+ and \tilde hx */
  /* Precompute powers of e^{i m alpha} */
  COMPLEX16 cexp_i_alpha =  cos(alpha) + MY_I*sin(alpha);//cexp(+MY_I*alpha);
  COMPLEX16 cexp_2i_alpha = cexp_i_alpha*cexp_i_alpha;
  COMPLEX16 cexp_mi_alpha = 1.0/cexp_i_alpha;
  COMPLEX16 cexp_m2i_alpha = cexp_mi_alpha*cexp_mi_alpha;
  COMPLEX16 cexp_im_alpha[5] = {cexp_m2i_alpha, cexp_mi_alpha, 1.0, cexp_i_alpha, cexp_2i_alpha};
  for(int m=-2; m<=2; m++) {
    COMPLEX16 T2m   = cexp_im_alpha[-m+2] * dm2[m+2] *      Y2mA[m+2];  /*  = cexp(-I*m*alpha) * dm2[m+2] *      Y2mA[m+2] */
    COMPLEX16 Tm2m  = cexp_im_alpha[m+2]  * d2[m+2]  * conj(Y2mA[m+2]); /*  = cexp(+I*m*alpha) * d2[m+2]  * conj(Y2mA[m+2]) */
    hp_sum +=     T2m + Tm2m;
    hc_sum += +MY_I*(T2m - Tm2m);
  }

  COMPLEX16 eps_phase_hP = (cos(2*epsilon) - MY_I*sin(2*epsilon)) *hP /2.0;//cexp(-2*I*epsilon) * hP / 2.0;
  *hp = eps_phase_hP * hp_sum;
  *hc = eps_phase_hP * hc_sum;

  // Return phasing for time-shift correction
  *phasing = -phPhenom; // ignore alpha and epsilon contributions

  return XLAL_SUCCESS;
}

/**
 * Simple 2PN version of the orbital angular momentum L,
 * without any spin terms expressed as a function of v.
 * For IMRPhenomP(v2).
 *
 *  Reference:
 *  - Boh&eacute; et al, 1212.5520v2 Eq 4.7 first line
 */
LALSIMULATION_CUDA_HOST_DEVICE REAL8 L2PNR(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta) /**< Symmetric mass-ratio */
{
  const REAL8 eta2 = eta*eta;
  const REAL8 x = v*v;
  const REAL8 x2 = x*x;
  return (eta*(1.0 + (1.5 + eta/6.0)*x + (3.375 - (19.0*eta)/8. - eta2/24.0)*x2)) / sqrt(x);
}

/**
 * Simple 2PN version of the orbital angular momentum L,
 * without any spin terms expressed as a function of v.
 * For IMRPhenomP(v1).
 *
 * Reference:
 *  - Kidder, Phys. Rev. D 52, 821–847 (1995), Eq. 2.9
 */
LALSIMULATION_CUDA_HOST_DEVICE REAL8 L2PNR_v1(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta) /**< Symmetric mass-ratio */
{
  const REAL8 mu = eta; /* M=1 */
  const REAL8 v2 = v*v;
  const REAL8 v3 = v2*v;
  const REAL8 v4 = v3*v;
  const REAL8 eta2 = eta*eta;
  const REAL8 b = (4.75 + eta/9.)*eta*v4;


  return mu*sqrt((1 - ((3 - eta)*v2)/3. + b)/v2)*
    (1 + ((1 - 3*eta)*v2)/2. + (3*(1 - 7*eta + 13*eta2)*v4)/8. +
      ((14 - 41*eta + 4*eta2)*v4)/(4.*pow_2_of(1 - ((3 - eta)*v2)/3. + b)) +
      ((3 + eta)*v2)/(1 - ((3 - eta)*v2)/3. + b) +
      ((7 - 10*eta - 9*eta2)*v4)/(2.*(1 - ((3 - eta)*v2)/3. + b)));
}

/** Expressions used for the WignerD symbol
  * with full expressions for the angles.
  * Used for IMRPhenomP(v2):
  * \f{equation}{
  * \cos(\beta) = \hat J . \hat L
  *             = \left( 1 + \left( S_\mathrm{p} / (L + S_L) \right)^2 \right)^{-1/2}
  *             = \left( L + S_L \right) / \sqrt{ \left( L + S_L \right)^2 + S_p^2 }
  *             = \mathrm{sign}\left( L + S_L \right) \cdot \left( 1 + \left( S_p / \left(L + S_L\right)\right)^2 \right)^{-1/2}
  * \f}
 */
LALSIMULATION_CUDA_HOST_DEVICE void WignerdCoefficients(
  REAL8 *cos_beta_half, /**< [out] cos(beta/2) */
  REAL8 *sin_beta_half, /**< [out] sin(beta/2) */
  const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,       /**< Dimensionfull aligned spin */
  const REAL8 eta,      /**< Symmetric mass-ratio */
  const REAL8 Sp)       /**< Dimensionfull spin component in the orbital plane */
{
  XLAL_CHECK_CUDA_VOID(cos_beta_half != NULL, XLAL_EFAULT);
  XLAL_CHECK_CUDA_VOID(sin_beta_half != NULL, XLAL_EFAULT);
  /* We define the shorthand s := Sp / (L + SL) */
  const REAL8 L = L2PNR(v, eta);
    // We ignore the sign of L + SL below.
  REAL8 s = Sp / (L + SL);  /* s := Sp / (L + SL) */
  REAL8 s2 = s*s;
  REAL8 cos_beta = 1.0 / sqrt(1.0 + s2);
  *cos_beta_half = + sqrt( (1.0 + cos_beta) / 2.0 );  /* cos(beta/2) */
  *sin_beta_half = + sqrt( (1.0 - cos_beta) / 2.0 );  /* sin(beta/2) */
}

/** Expressions used for the WignerD symbol
  * with small angle approximation.
  * Used for IMRPhenomP(v1):
  * \f{equation}{
  * \cos(\beta) = \hat J . \hat L
  *             = \left(1 + \left( S_\mathrm{p} / (L + S_L)\right)^2 \right)^{-1/2}
  * \f}
  * We use the expression
  * \f{equation}{
  * \cos(\beta/2) \approx (1 + s^2 / 4 )^{-1/2} \,,
  * \f}
  * where \f$s := S_p / (L + S_L)\f$.
 */
LALSIMULATION_CUDA_HOST_DEVICE void WignerdCoefficients_SmallAngleApproximation(
  REAL8 *cos_beta_half, /**< Output: cos(beta/2) */
  REAL8 *sin_beta_half, /**< Output: sin(beta/2) */
  const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,       /**< Dimensionfull aligned spin */
  const REAL8 eta,      /**< Symmetric mass-ratio */
  const REAL8 Sp)       /**< Dimensionfull spin component in the orbital plane */
{
  XLAL_CHECK_CUDA_VOID(cos_beta_half != NULL, XLAL_EFAULT);
  XLAL_CHECK_CUDA_VOID(sin_beta_half != NULL, XLAL_EFAULT);
  REAL8 s = Sp / (L2PNR_v1(v, eta) + SL);  /* s := Sp / (L + SL) */
  REAL8 s2 = s*s;
  *cos_beta_half = 1.0/sqrt(1.0 + s2/4.0);           /* cos(beta/2) */
  *sin_beta_half = sqrt(1.0 - 1.0/(1.0 + s2/4.0));   /* sin(beta/2) */
}

/***********************************************************************************/
/* The following private function generates the complete amplitude and phase of    */
/* PhenomC waveform, at a given frequency.                                         */
/* Eq. (5.3), (5.9) of the Main paper.                                             */
/***********************************************************************************/
LALSIMULATION_CUDA_HOST_DEVICE int IMRPhenomCGenerateAmpPhase(
    REAL8 *amplitude, /**< pointer to memory for phenomC amplitude */
    REAL8 *phasing,   /**< pointer to memory for phenomC phase */
    REAL8 f,          /**< frequency (Hz) */
    REAL8 eta,        /**< dimensionless mass-ratio */
    const BBHPhenomCParams *params /**< pointer to Object storing coefficients and constants */
    )
{
  *amplitude = 0.0;
  *phasing = 0.0;

  /* Get the phase */
  REAL8 v =  cbrt(params->piM * f);
  REAL8 Mf = params->m_sec * f;

  if( v >= 1.0 )
    XLAL_ERROR_CUDA(XLAL_EDOM);

  REAL8 v2 = v*v;
  REAL8 v3 = v*v2;
  REAL8 v4 = v2*v2;
  REAL8 v5 = v3*v2;
  REAL8 v6 = v3*v3;
  REAL8 v7 = v4*v3;
  REAL8 v10 = v5*v5;

  /* SPA part of the phase */
  REAL8 phSPA = 1. + params->pfa1 * v + params->pfa2 * v2 + params->pfa3 * v3 + params->pfa4 * v4 +
    (1. + log(v3)) * params->pfa5 * v5 + (params->pfa6  + params->pfa6log * log(v3))*v6 +
    params->pfa7 * v7;
  phSPA *= (params->pfaN / v5);

  // Taking t0 = phi0 = 0
  phSPA -= (LAL_PI / 4.);

  REAL8 w = cbrt( Mf );
  REAL8 w2 = w*w;
  REAL8 w3 = w2*w;
  REAL8 w5 = w3*w2;

  /* The Pre-Merger (PM) phase */
  REAL8 phPM = (params->a1/w5) + (params->a2/w3) + (params->a3/w) + params->a4 +
    (params->a5*w2) +(params->a6*w3);
  phPM /= eta;

  /* Ring-down phase */
  REAL8 phRD = params->b1 + params->b2 * params->m_sec * f;

  REAL8 wPlusf1 = wPlus( f, params->f1, params->d1, params );
  REAL8 wPlusf2 = wPlus( f, params->f2, params->d2, params );
  REAL8 wMinusf1 = wMinus( f, params->f1, params->d1, params );
  REAL8 wMinusf2 = wMinus( f, params->f2, params->d2, params );

  *phasing = phSPA * wMinusf1 + phPM * wPlusf1 * wMinusf2 + phRD * wPlusf2;

  /* Get the amplitude */
  REAL8 xdot = 1. + params->xdota2*v2 + params->xdota3*v3 + params->xdota4*v4 +
    params->xdota5*v5 + (params->xdota6 + params->xdota6log*log(v2))*v6 +
    params->xdota7 * v7;
  xdot *= (params->xdotaN * v10);

  if( xdot < 0.0 && f < params->f1 )
  {
    XLALPrintError_CUDA("omegaDot < 0, while frequency is below SPA-PM matching freq.");
    XLAL_ERROR_CUDA( XLAL_EDOM );
  }

  REAL8 aPM = 0.0;

  /* Following Emma's code, take only the absolute value of omegaDot, when
   * computing the amplitude */
  REAL8 omgdot = 1.5*v*xdot;
  REAL8 ampfac = sqrt(fabs(LAL_PI/omgdot));

  /* Get the real and imaginary part of the PM amplitude */
  REAL8 AmpPMre = ampfac * params->AN * v2 * (1. + params->A2*v2 + params->A3*v3 +
      params->A4*v4 + params->A5*v5 + (params->A6 + params->A6log*log(v2))*v6);
  REAL8 AmpPMim = ampfac * params->AN * v2 * (params->A5imag * v5 + params->A6imag * v6);

  /* Following Emma's code, we take the absolute part of the complex SPA
   * amplitude, and keep that as the amplitude */
  aPM = sqrt( AmpPMre * AmpPMre + AmpPMim * AmpPMim );

  aPM += (params->g1 * pow(Mf,(5./6.)));

  /* The Ring-down aamplitude */
  REAL8 Mfrd = params->MfRingDown;

  /* From Emma's code, use sigma = fRD * del2 / Qual */
  REAL8 sig = Mfrd * params->del2 / params->Qual;
  REAL8 sig2 = sig*sig;
  REAL8 L = sig2 / ((Mf - Mfrd)*(Mf - Mfrd) + sig2/4.);

  REAL8 aRD = params->del1 * L * pow(Mf, (-7./6.));

  REAL8 wPlusf0 = wPlus( f, params->f0, params->d0, params );
  REAL8 wMinusf0 = wMinus( f, params->f0, params->d0, params );

  *amplitude = - (aPM * wMinusf0 + aRD * wPlusf0);

  return XLAL_SUCCESS;
}

LALSIMULATION_CUDA_HOST_DEVICE int init_useful_powers(UsefulPowers * p, REAL8 number)
{
    XLAL_CHECK_CUDA(0 != p, XLAL_EFAULT, "p is NULL");
    XLAL_CHECK_CUDA(number >= 0 , XLAL_EDOM, "number must be non-negative");

    // consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
    p->sixth = pow(number, 1/6.0);
    p->third = p->sixth * p->sixth;
    p->two_thirds = number / p->third;
    p->four_thirds = number * (p->third);
    p->five_thirds = p->four_thirds * (p->third);
    p->two = number * number;
    p->seven_thirds = p->third * p->two;
    p->eight_thirds = p->two_thirds * p->two;

    return XLAL_SUCCESS;
}

// Call ComputeIMRPhenomDAmplitudeCoefficients() first!
/**
 * This function computes the IMR amplitude given phenom coefficients.
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double IMRPhenDAmplitude(double f, IMRPhenomDAmplitudeCoefficients *p, UsefulPowers *powers_of_f, AmpInsPrefactors * prefactors) {
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermediate and merger-ringdown amplitude parts

  // Transition frequencies
  p->fInsJoin = AMP_fJoin_INS;
  p->fMRDJoin = p->fmaxCalc;

  double f_seven_sixths = f * powers_of_f->sixth;
  double AmpPreFac = prefactors->amp0 / f_seven_sixths;

  // split the calculation to just 1 of 3 possible mutually exclusive ranges

  if (!StepFunc_boolean(f, p->fInsJoin))    // Inspiral range
  {
      double AmpIns = AmpPreFac * AmpInsAnsatz(f, powers_of_f, prefactors);
      return AmpIns;
  }

  if (StepFunc_boolean(f, p->fMRDJoin)) // MRD range
  {
      double AmpMRD = AmpPreFac * AmpMRDAnsatz(f, p);
      return AmpMRD;
  }

  //    Intermediate range
  double AmpInt = AmpPreFac * AmpIntAnsatz(f, p);
  return AmpInt;
}

/**
 * This function computes the IMR phase given phenom coefficients.
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double IMRPhenDPhase(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, UsefulPowers *powers_of_f, PhiInsPrefactors * prefactors)
{
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermendiate and merger-ringdown phase parts

  // split the calculation to just 1 of 3 possible mutually exclusive ranges

  if (!StepFunc_boolean(f, p->fInsJoin))    // Inspiral range
  {
      double PhiIns = PhiInsAnsatzInt(f, powers_of_f, prefactors, p, pn);
      return PhiIns;
  }

  if (StepFunc_boolean(f, p->fMRDJoin)) // MRD range
  {
      double PhiMRD = 1.0/p->eta * PhiMRDAnsatzInt(f, p) + p->C1MRD + p->C2MRD * f;
      return PhiMRD;
  }

  //    Intermediate range
  double PhiInt = 1.0/p->eta * PhiIntAnsatz(f, p) + p->C1Int + p->C2Int * f;
  return PhiInt;
}

/*********************************************************************/
/* The following function return the hyperbolic-Tan+ windows used   */
/* in Eq.(5.9), (5.13) of the Main paper                             */
/*********************************************************************/
LALSIMULATION_CUDA_HOST_DEVICE REAL8 wPlus( const REAL8 f, const REAL8 f0, const REAL8 d, const BBHPhenomCParams *params )
{

  REAL8 Mf = params->m_sec * f;
  REAL8 Mf0 = params->m_sec * f0;

  return ( 0.5 * (1. + tanh(4.*(Mf - Mf0)/d) ) );

}

/*********************************************************************/
/* The following function return the hyperbolic-Tan- windows used   */
/* in Eq.(5.9), (5.13) of the Main paper                             */
/*********************************************************************/
LALSIMULATION_CUDA_HOST_DEVICE REAL8 wMinus( const REAL8 f, const REAL8 f0, const REAL8 d, const BBHPhenomCParams *params )
{

  REAL8 Mf = params->m_sec * f;
  REAL8 Mf0 = params->m_sec * f0;

  return ( 0.5 * (1. - tanh(4.*(Mf - Mf0)/d) ) );

}

/**
 * Step function in boolean version
 */
LALSIMULATION_CUDA_HOST_DEVICE bool StepFunc_boolean(const double t, const double t1) {
    return (t >= t1);
}

// The Newtonian term in LAL is fine and we should use exactly the same (either hardcoded or call).
// We just use the Mathematica expression for convenience.
/**
 * Inspiral amplitude plus rho phenom coefficents. rho coefficients computed
 * in rho1_fun, rho2_fun, rho3_fun functions.
 * Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details
 */
LALSIMULATION_CUDA_HOST_DEVICE double AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors) {
  double Mf2 = powers_of_Mf->two;
  double Mf3 = Mf*Mf2;

  return 1 + powers_of_Mf->two_thirds * prefactors->two_thirds
            + Mf * prefactors->one + powers_of_Mf->four_thirds * prefactors->four_thirds
            + powers_of_Mf->five_thirds * prefactors->five_thirds + Mf2 * prefactors->two
            + powers_of_Mf->seven_thirds * prefactors->seven_thirds + powers_of_Mf->eight_thirds * prefactors->eight_thirds
            + Mf3 * prefactors->three;
}

/**
 * Take the AmpInsAnsatz expression and compute the first derivative
 * with respect to frequency to get the expression below.
 */
LALSIMULATION_CUDA_HOST_DEVICE double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double eta = p->eta;
  double chi1 = p->chi1;
  double chi2 = p->chi2;
  double rho1 = p->rho1;
  double rho2 = p->rho2;
  double rho3 = p->rho3;

  double chi12 = chi1*chi1;
  double chi22 = chi2*chi2;
  double eta2 = eta*eta;
  double eta3 = eta*eta2;
  double Mf2 = Mf*Mf;
  double Pi = LAL_PI;
#if defined(LALSIMULATION_CUDA_ENABLED)
  double Pi2 = LAL_PI*LAL_PI;
#else
  double Pi2 = powers_of_pi.two;
#endif
  double Seta = sqrt(1.0 - 4.0*eta);

   return ((-969 + 1804*eta)*pow(Pi,2.0/3.0))/(1008.*pow(Mf,1.0/3.0))
   + ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.
   + ((-27312085 - 10287648*chi22 - 10287648*chi12*(1 + Seta)
   + 10287648*chi22*Seta + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
   + 35371056*eta2)*pow(Mf,1.0/3.0)*pow(Pi,4.0/3.0))/6.096384e6
   + (5*pow(Mf,2.0/3.0)*pow(Pi,5.0/3.0)*(chi2*(-285197*(-1 + Seta)
   + 4*(-91902 + 1579*Seta)*eta - 35632*eta2) + chi1*(285197*(1 + Seta)
   - 4*(91902 + 1579*Seta)*eta - 35632*eta2) + 42840*(-1 + 4*eta)*Pi))/96768.
   - (Mf*Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2 - 324322727232*eta3
   - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta) - 107414046432*chi22*Seta
   + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi)
   + 12*eta*(-545384828789.0 - 176491177632*chi1*chi2 + 202603761360*chi22 + 77616*chi12*(2610335 + 995766*Seta)
   - 77287373856*chi22*Seta + 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2)))/3.0042980352e10
   + (7.0/3.0)*pow(Mf,4.0/3.0)*rho1 + (8.0/3.0)*pow(Mf,5.0/3.0)*rho2 + 3*Mf2*rho3;
}

/**
 * Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;
  double fDMgamma3 = fDM*gamma3;
  double fminfRD = f - fRD;
  return exp( -(fminfRD)*gamma2 / (fDMgamma3) )
    * (fDMgamma3*gamma1) / (pow_2_of(fminfRD) + pow_2_of(fDMgamma3));
}

/**
 * first frequency derivative of AmpMRDAnsatz
 */
LALSIMULATION_CUDA_HOST_DEVICE double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  double fDMgamma3 = fDM * gamma3;
  double pow2_fDMgamma3 = pow_2_of(fDMgamma3);
  double fminfRD = f - fRD;
  double expfactor = exp(((fminfRD)*gamma2)/(fDMgamma3));
  double pow2pluspow2 = pow_2_of(fminfRD) + pow2_fDMgamma3;

   return (-2*fDM*(fminfRD)*gamma3*gamma1) / ( expfactor * pow_2_of(pow2pluspow2)) -
     (gamma2*gamma1) / ( expfactor * (pow2pluspow2)) ;
}

/**
 * Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double AmpIntAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double Mf2 = Mf*Mf;
  double Mf3 = Mf*Mf2;
  double Mf4 = Mf*Mf3;
  return p->delta0 + p->delta1*Mf + p->delta2*Mf2 + p->delta3*Mf3 + p->delta4*Mf4;
}

/**
 * Ansatz for the inspiral phase.
 * We call the LAL TF2 coefficients here.
 * The exact values of the coefficients used are given
 * as comments in the top of this file
 * Defined by Equation 27 and 28 arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double PhiInsAnsatzInt(double Mf, UsefulPowers * powers_of_Mf, PhiInsPrefactors * prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn)
{
    XLAL_CHECK_CUDA(0 != pn, XLAL_EFAULT, "pn is NULL");

  // Assemble PN phasing series
#if defined(LALSIMULATION_CUDA_ENABLED)
  const double sixth = pow(LAL_PI, 1/6.0);
  const double third = sixth*sixth;
  const double v = powers_of_Mf->third * third;
#else
  const double v = powers_of_Mf->third * powers_of_pi.third;
#endif
  const double logv = log(v);

  double phasing = prefactors->initial_phasing;

  phasing += prefactors->two_thirds * powers_of_Mf->two_thirds;
  phasing += prefactors->third * powers_of_Mf->third;
  phasing += prefactors->third_with_logv * logv * powers_of_Mf->third;
  phasing += prefactors->logv * logv;
  phasing += prefactors->minus_third / powers_of_Mf->third;
  phasing += prefactors->minus_two_thirds / powers_of_Mf->two_thirds;
  phasing += prefactors->minus_one / Mf;
  phasing += prefactors->minus_five_thirds / powers_of_Mf->five_thirds; // * v^0

  // Now add higher order terms that were calibrated for PhenomD
  phasing += ( prefactors->one * Mf + prefactors->four_thirds * powers_of_Mf->four_thirds
               + prefactors->five_thirds * powers_of_Mf->five_thirds
               + prefactors->two * powers_of_Mf->two
             ) / p->eta;

  return phasing;
}

/**
 * Ansatz for the merger-ringdown phase Equation 14 arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double PhiMRDAnsatzInt(double f, IMRPhenomDPhaseCoefficients *p)
{
  double sqrootf = sqrt(f);
  double fpow1_5 = f * sqrootf;
  // check if this is any faster: 2 sqrts instead of one pow(x,0.75)
  double fpow0_75 = sqrt(fpow1_5); // pow(f,0.75);

  return -(p->alpha2/f)
         + (4.0/3.0) * (p->alpha3 * fpow0_75)
         + p->alpha1 * f
         + p->alpha4 * atan((f - p->alpha5 * p->fRD) / p->fDM);
}

/**
 * ansatz for the intermediate phase defined by Equation 16 arXiv:1508.07253
 */
LALSIMULATION_CUDA_HOST_DEVICE double PhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
  // 1./eta in paper omitted and put in when need in the functions:
  // ComputeIMRPhenDPhaseConnectionCoefficients
  // IMRPhenDPhase
  return  p->beta1*Mf - p->beta3/(3.*pow_3_of(Mf)) + p->beta2*log(Mf);
}


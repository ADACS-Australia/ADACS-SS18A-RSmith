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
#include "LALSimIMRPhenomP_cuda.h"

void PhenomPCoreAllFrequencies_cuda(UINT4 L_fCut,
        REAL8Sequence *freqs,
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
        int   *errcode){
fprintf(stderr,"Entered cuda code.\n");

  // Copy inputs to device
  IMRPhenomDAmplitudeCoefficients pAmp;
  IMRPhenomDPhaseCoefficients pPhi;
  BBHPhenomCParams PCparams;
  PNPhasingSeries pn;
  NNLOanglecoeffs angcoeffs;
  SpinWeightedSphericalHarmonic_l2 Y2m;
  AmpInsPrefactors amp_prefactors;
  PhiInsPrefactors phi_prefactors;
  COMPLEX16FrequencySeries *hptilde;
  COMPLEX16FrequencySeries *hctilde;
  REAL8 *phis;
  try{
      throw_on_cuda_error(cudaMemcpy(&pAmp,          pAmp_host,          sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&pPhi,          pPhi_host,          sizeof(IMRPhenomDPhaseCoefficients),     cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&PCparams,      PCparams_host,      sizeof(BBHPhenomCParams),                cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&pn,            pn_host,            sizeof(PNPhasingSeries),                 cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&angcoeffs,     angcoeffs_host,     sizeof(NNLOanglecoeffs),                 cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&Y2m,           Y2m_host,           sizeof(SpinWeightedSphericalHarmonic_l2),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&amp_prefactors,amp_prefactors_host,sizeof(AmpInsPrefactors),                cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(&phi_prefactors,phi_prefactors_host,sizeof(PhiInsPrefactors),                cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMalloc((void **)&hptilde,L_fCut*sizeof(COMPLEX16)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMalloc((void **)&hctilde,L_fCut*sizeof(COMPLEX16)),lalsimulation_cuda_exception::MALLOC);
      throw_on_cuda_error(cudaMalloc((void **)&phis,   L_fCut*sizeof(REAL8)),    lalsimulation_cuda_exception::MALLOC);
  }
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }

  // Run kernel
  try{
    int n_threads=256;
    int grid_size=(L_fCut+(n_threads-1))/n_threads;
    /*
    PhenomPCoreOneFrequency_cuda<<<grid_size,n_threads>>>(L_fCut,
          freqs,
          offset,
          eta,
          chi1_l,
          chi2_l,
          chip,
          distance,
          M,
          phic,
          &pAmp,
          &pPhi,
          &PCparams,
          &pn,
          &angcoeffs,
          &Y2m,
          alphaNNLOoffset,
          alpha0,
          epsilonNNLOoffset,
          IMRPhenomP_version,
          &amp_prefactors,
          &phi_prefactors,
          hptilde,
          hctilde,
          phis);
          */
  }
  // Alter this to return an error code on kernel errorcode exception
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }

  // Offload results
  try{
      throw_on_cuda_error(cudaMemcpy(hptilde_host,hptilde,L_fCut*sizeof(COMPLEX16FrequencySeries),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(hctilde_host,hctilde,L_fCut*sizeof(COMPLEX16FrequencySeries),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
      throw_on_cuda_error(cudaMemcpy(phis_host,   phis,   L_fCut*sizeof(REAL8),                   cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
  }
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }

  // Clean-up
  try{
      throw_on_cuda_error(cudaFree(hptilde),lalsimulation_cuda_exception::FREE);
      throw_on_cuda_error(cudaFree(hctilde),lalsimulation_cuda_exception::FREE);
      throw_on_cuda_error(cudaFree(phis),   lalsimulation_cuda_exception::FREE);
  }
  catch(const lalsimulation_cuda_exception e){
      e.process_exception();
  }
fprintf(stderr,"Leaving cuda code.\n");
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
__host__ void _throw_on_global_error(const std::string file, const std::string func, int line)
{
  int error_code=0;
  //MPI_Allreduce(MPI_IN_PLACE,&error_code,1,MPI_INT,MPI_MAX,run_globals.mpi_comm);
  //if(error_code!=0) throw(meraxes_cuda_exception(0,meraxes_cuda_exception::GLOBAL,file,func,line));
}
__host__ void notify_of_global_error(int error_code)
{
  int result=(int)error_code;
  //MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,run_globals.mpi_comm);
}

__device__ void PhenomPCoreOneFrequency_cuda(UINT4 L_fCut,
        REAL8Sequence *freqs,
        UINT4 offset,
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
        COMPLEX16FrequencySeries *hptilde,
        COMPLEX16FrequencySeries *hctilde,
        REAL8 *phis){

    UINT4 i = (UINT4)(blockIdx.x*blockDim.x + threadIdx.x);
    if(i < L_fCut){

      COMPLEX16 hp_val;
      COMPLEX16 hc_val;
      REAL8     phasing;
      double    f = freqs->data[i];
      int       j = i + offset; // shift index for frequency series if needed

      // Generate the waveform
      //int per_thread_errcode = PhenomPCoreOneFrequency(f,
      //                                                 eta, chi1_l, chi2_l, chip, distance, M, phic,
      //                                                 pAmp, pPhi, PCparams, pn, angcoeffs, Y2m,
      //                                                 alphaNNLOoffset - alpha0, epsilonNNLOoffset,
      //                                                 &hp_val, &hc_val, &phasing, IMRPhenomP_version, amp_prefactors, phi_prefactors);

      // THROW EXCEPTION HERE INSTEAD
      //if (per_thread_errcode != XLAL_SUCCESS) {
      //  (*errcode) = per_thread_errcode;
      //}

      (hptilde->data->data)[j] = hp_val;
      (hctilde->data->data)[j] = hc_val;

      phis[i] = phasing;

  }
}

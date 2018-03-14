//==============================================================================
//
// This code was developed as part of the Astronomy Data and Computing Services
// (ADACS; https://adacs.org.au) 2017B Software Support program.
//
// Written by: Gregory B. Poole
// Date:       September 2017
//
// It is distributed under the MIT (Expat) License (see https://opensource.org/):
//
// Copyright (c) 2017 Astronomy Data and Computing Services (ADACS)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//==============================================================================

//#include <assert.h>
//#include <signal.h>
//#include <limits.h>
//
//#include <fstream>
//#include <vector>
//#include <algorithm>
//#include <string>
//
//#include <hdf5.h>
//#include <hdf5_hl.h>
//
#include <cuda_runtime.h>
#include <cufft.h>
//
//#ifdef __GNUC__
//#define UNUSED __attribute__ ((unused))
//#else
//#define UNUSED
//#endif
//
//#include <stdlib.h>
//#include <float.h>
////#include <gsl/gsl_errno.h>
////#include <gsl/gsl_spline.h>
////#include <gsl/gsl_math.h>
////#include <gsl/gsl_sf_trig.h>
//
//#include <lal/Date.h>
//#include <lal/FrequencySeries.h>
//#include <lal/LALAtomicDatatypes.h>
//#include <lal/LALConstants.h>
//#include <lal/LALDatatypes.h>
//#include <lal/LALSimInspiral.h>
//#include <lal/Units.h>
//#include <lal/XLALError.h>
//#include <lal/SphericalHarmonics.h>
//#include <lal/Sequence.h>
//#include <lal/LALStdlib.h>
//#include <lal/LALStddef.h>
//
////#include "LALSimIMR.h"
//// This is ugly, but allows us to reuse internal IMRPhenomC and IMRPhenomD functions without making those functions XLAL 
////#include "LALSimIMRPhenomC_internals.c"
////#include "LALSimIMRPhenomD_internals.c"
//#include "LALSimIMRPhenomD_internals.h"
//
////#include "LALSimIMRPhenomP.h"
//#include "LALSimIMRPhenomP_cuda.h"
//
//#ifndef _OPENMP
//#define omp ignore
//#endif
//
//// Macro functions to rotate the components of a vector about an axis 
//#define ROTATEZ(angle, vx, vy, vz)\
//tmp1 = vx*cos(angle) - vy*sin(angle);\
//tmp2 = vx*sin(angle) + vy*cos(angle);\
//vx = tmp1;\
//vy = tmp2
//
//#define ROTATEY(angle, vx, vy, vz)\
//tmp1 = vx*cos(angle) + vz*sin(angle);\
//tmp2 = - vx*sin(angle) + vz*cos(angle);\
//vx = tmp1;\
//vz = tmp2
//
////const double sqrt_6 = 2.44948974278317788;
//
//// These functions deal with any GPU exceptions, but should be called with the macros defined in the corresponding .hh file
//__host__ void _throw_on_generic_error(bool check_failure,int implementation_code, const std::string file, const std::string func, int line)
//{
//  if(check_failure) throw(lalsimulation_cuda_exception(GENERIC_CUDA_ERROR_CODE,implementation_code,file,func,line));
//}
//__host__ void _throw_on_cuda_error(cudaError_t cuda_code, int implementation_code, const std::string file, const std::string func, int line)
//{
//  if(cuda_code != cudaSuccess) throw(lalsimulation_cuda_exception((int)cuda_code,implementation_code,file,func,line));
//}
//__host__ void _throw_on_cuFFT_error(cufftResult cufft_code, int implementation_code, const std::string file, const std::string func, int line)
//{
//  if(cufft_code != CUFFT_SUCCESS) throw(lalsimulation_cuda_exception((int)cufft_code,implementation_code,file,func,line));
//}
//__host__ void _check_for_cuda_error(int implementation_code,const std::string file, const std::string func, int line)
//{
//  try{
//    cudaError_t cuda_code = cudaPeekAtLastError();
//    if(cuda_code != cudaSuccess)
//        throw(lalsimulation_cuda_exception((int)cuda_code,implementation_code,"CUDA error detected after ",file,func,line));
//  }
//  catch(const lalsimulation_cuda_exception e){
//      e.process_exception();
//  }
//}
//__host__ void _check_thread_sync(int implementation_code,const std::string file, const std::string func, int line)
//{
//  try{
//    cudaError_t cuda_code = cudaDeviceSynchronize();
//    if(cuda_code != cudaSuccess)
//        throw(lalsimulation_cuda_exception((int)cuda_code,implementation_code,"Threads not synchronised after ",file,func,line));
//  }
//  catch(const lalsimulation_cuda_exception e){
//      e.process_exception();
//  }
//}
//__host__ void _throw_on_global_error(const std::string file, const std::string func, int line)
//{
//  int error_code=0;
////  MPI_Allreduce(MPI_IN_PLACE,&error_code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
//  if(error_code!=0) throw(lalsimulation_cuda_exception(0,lalsimulation_cuda_exception::GLOBAL,file,func,line));
//}
//__host__ void notify_of_global_error(int error_code)
//{
////  int result=(int)error_code;
////  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
//}
//
//// Initialize device.  Called by init_gpu().
//void init_CUDA(){
//    try{
//        // Check if the environment variable PBS_GPUFILE is defined
//        //    If it is, we assume that it lists the devices available
//        //    to the job.  Use this list to select a device number.
//        /*
//        char *filename_gpu_list=std::getenv("PBS_GPUFILE");
//        if(filename_gpu_list!=NULL){
//            // To determine which device to use, we need 4 steps:
//            //   1) Determine what node each rank is on by reading the
//            //      PBS_NODEFILE file.
//            //   2) Determine the rank-order-position of each mpi rank
//            //      in the list of cores on the local node (i_rank_node),
//            //      as well as the number of ranks on the local node (n_rank_node)
//            //   3) Determine the list of available device numbers
//            //      on the local node
//            //   4) Select a device number of the rank from the list
//            //      of available devices using i_rank_node & n_rank_node.
//
//            // First, parse PBS_NODEFILE and communicate the name of each rank's node
//            char *filename_node_name_list=std::getenv("PBS_NODEFILE");
//            throw_on_generic_error(filename_node_name_list==NULL,lalsimulation_cuda_exception::INIT_PBS_GPUFILE);
//            char node_name[MPI_MAX_PROCESSOR_NAME]; // this will hold the rank's node name
//            if(run_globals.mpi_rank==0){
//                // Open the file
//                std::ifstream infile(filename_node_name_list);
//                // Read each line
//                int i_rank=0;
//                std::string node_name_i;
//                while (std::getline(infile,node_name_i)){
//                    char node_name_i_c_ctr[MPI_MAX_PROCESSOR_NAME];
//                    strcpy(node_name_i_c_ctr,node_name_i.c_str());
//                    // Store this device number on the i_rank'th rank
//                    if(i_rank==0)
//                        strcpy(node_name,node_name_i_c_ctr);
//                    else if(i_rank<run_globals.mpi_size)
//                        MPI_Send(node_name_i_c_ctr,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,i_rank,110100100,run_globals.mpi_comm);
//                    i_rank++;
//                }
//                infile.close();
//                // Check that the length of the file matches the size of the communicator
//                throw_on_generic_error(i_rank!=run_globals.mpi_size,lalsimulation_cuda_exception::INIT_PBS_GPUFILE);
//            }
//            else{
//                MPI_Status mpi_status; // ignored, at the moment
//                MPI_Recv(node_name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,110100100,run_globals.mpi_comm,&mpi_status);
//            }
//            throw_on_global_error();
//
//            // Second, generate a count of ranks on the local node and determine each rank's position in that count
//            int n_rank_node=0;
//
//            // Open the file and loop over the entries
//            std::ifstream infile;
//            if(run_globals.mpi_rank==0)
//                infile.open(filename_node_name_list);
//            int i_rank     =0;
//            int i_rank_node=-1; // this will hold the rank-order of this mpi rank on the local node
//            for(;i_rank<run_globals.mpi_size;i_rank++){ // we've already checked above that the size is as expected
//                char node_test[MPI_MAX_PROCESSOR_NAME];
//                if(run_globals.mpi_rank==0){
//                    std::string node_name_i;
//                    std::getline(infile,node_name_i);
//                    strcpy(node_test,node_name_i.c_str());
//                }
//                MPI_Bcast(node_test,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,run_globals.mpi_comm);
//                bool flag_test_is_local=(!strcmp(node_name,node_test));
//                // Determine the rank-order for this node.
//                // Set an integer, defaulting to the end of an ascending list ...
//                int i_rank_node_test=run_globals.mpi_size+run_globals.mpi_rank;
//                // ... unless this item matches the local node name, and we haven't set a value yet ...
//                if(flag_test_is_local && i_rank_node<0)
//                    i_rank_node_test=run_globals.mpi_rank;
//                // Find the earliest local-node rank that isn't set yet
//                MPI_Allreduce(MPI_IN_PLACE,&i_rank_node_test,1,MPI_INT,MPI_MIN,run_globals.mpi_comm);
//                // And set the index for that node
//                if(i_rank_node_test==run_globals.mpi_rank)
//                    i_rank_node=n_rank_node;
//                // Lastly, generate a count
//                if(flag_test_is_local)
//                    n_rank_node++;
//            }
//            if(run_globals.mpi_rank==0)
//                infile.close();
//
//            // Sanity check on the local node's rank count
//            throw_on_generic_error(n_rank_node<=0 || n_rank_node>run_globals.mpi_size,lalsimulation_cuda_exception::INIT_PBS_GPUFILE);
//            throw_on_global_error();
//
//            // Read the GPU list, keeping a list of device numbers on this rank's node.
//            //    The file is assumed to list the gpus with the following form:
//            //    <node_name>-gpu<device_number>
//
//            // Read the device file into a vector of strings on the 0'th rank
//            std::vector<std::string> device_file_lines;
//            if(run_globals.mpi_rank==0){
//                std::ifstream infile(filename_gpu_list);
//                std::string line;
//                while (std::getline(infile,line)) device_file_lines.push_back(line);
//                infile.close();
//            }
//            int n_device_global=device_file_lines.size();
//            MPI_Bcast(&n_device_global,1,MPI_INT,0,run_globals.mpi_comm);
//
//            // Loop over the file again, communicating host names and device numbers to each rank
//            std::vector<int> device_number_list;
//            int i_device=0;
//            for(;i_device<n_device_global;i_device++){ // we've already checked above that the size is as expected
//                char host_name_i[MPI_MAX_PROCESSOR_NAME];
//                int  device_number_i;
//                // Read the line and parse it into host name and device number
//                if(run_globals.mpi_rank==0){
//                    // Parse the i_rank'th node name
//                    std::string line=device_file_lines[i_device];
//                    std::string host_name_temp;
//                    int  i_char=0;
//                    char char_i=line[i_char];
//                    int  l_size=line.size();
//                    while(char_i!='-' && i_char<l_size){
//                        host_name_temp+=char_i;
//                        if((++i_char)>=l_size) break;
//                        char_i=line[i_char];
//                    }
//                    strcpy(host_name_i,host_name_temp.c_str());
//                    // Skip '-gpu'
//                    i_char+=4;
//                    // Parse i_rank'th device number
//                    std::string device_number_str;
//                    char_i=line[i_char];
//                    while(i_char<l_size){
//                        device_number_str+=char_i;
//                        if((++i_char)>=l_size) break;
//                        char_i=line[i_char];
//                    }
//                    device_number_i=std::stoi(device_number_str);
//                }
//                // Communicate result
//                MPI_Bcast(host_name_i,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,run_globals.mpi_comm);
//                MPI_Bcast(&device_number_i,1,MPI_INT,0,run_globals.mpi_comm);
//
//                // Accumulate a list of device numbers on the local node
//                if(!strcmp(node_name,host_name_i)){
//                    if(std::find(device_number_list.begin(),device_number_list.end(),device_number_i)==device_number_list.end()){
//                       device_number_list.push_back(device_number_i);
//                    }
//                }
//            }
//            throw_on_generic_error(device_number_list.size()<=0,lalsimulation_cuda_exception::INIT_PBS_GPUFILE);
//
//            // Finally, set the device number
//            int device_number=device_number_list[i_rank_node%device_number_list.size()];
//
//            // Set the device to establish a context
//            throw_on_cuda_error(cudaSetDevice(device_number),lalsimulation_cuda_exception::INIT);
//        }
//        // If a GPU file is not specified, assume that CUDA can establish a default context.
//        else
//        */
//            throw_on_cuda_error(cudaFree(0),lalsimulation_cuda_exception::INIT);
//        throw_on_global_error();
//
//        // Get the device assigned to this context
//        //throw_on_cuda_error(cudaGetDevice(&(gpu->device)),lalsimulation_cuda_exception::INIT);
//
//        // Get the properties of the device assigned to this context
//        //throw_on_cuda_error(cudaGetDeviceProperties(&(gpu->properties),gpu->device),lalsimulation_cuda_exception::INIT);
//
//        // Throw an exception if another rank has thrown one
//        throw_on_global_error();
//    }
//    catch(const lalsimulation_cuda_exception e){
//        e.process_exception();
//    }
//
//    // Set the number of threads to use.  Perhaps something
//    //    more sophisticated should be done here eventually ...
//    //int n_threads=256;
//
//    // Send a status message to mlog
//    //if(run_globals.mpi_size==1){
//    //    mlog("GPU context established on device %d (%s; %.1fGBs of global memory).",MLOG_MESG,
//    //            run_globals.gpu->device,run_globals.gpu->properties.name,(float)(run_globals.gpu->properties.totalGlobalMem/(1024*1024*1024)));
//    //}
//    //else{
//    //    // Report device details for each rank
//    //    int i_rank=0;
//    //    for(;i_rank<run_globals.mpi_size;i_rank++){
//    //        if(i_rank==run_globals.mpi_rank)
//    //            mlog("Context established on GPU device %d (%s; %.1fGBs of global memory).",MLOG_MESG|MLOG_ALLRANKS|MLOG_FLUSH,
//    //                    run_globals.gpu->device,run_globals.gpu->properties.name,(float)(run_globals.gpu->properties.totalGlobalMem/(1024*1024*1024)));
//    //        MPI_Barrier(run_globals.mpi_comm);
//    //    }
//    //}
//}
//
//// Call this function in kernels to put the GPU in an error state that can be caught after as an exception
////    This is not necessarily the best way, but it will do the job for now.  This is based on:
//// https://devtalk.nvidia.com/default/topic/418479/how-to-trigger-a-cuda-error-from-inside-a-kernel/
//__device__
//void inline cause_cuda_error(){
//   int *adr = (int*)0xffffffff;
//   *adr = 12;
//}
//
//__host__
//void PhenomPCoreAllFrequencies_cuda(UINT4 L_fCut,
//        REAL8Sequence *freqs,
//        UINT4 offset,
//        const REAL8 eta,
//        const REAL8 chi1_l,
//        const REAL8 chi2_l,
//        const REAL8 chip,
//        const REAL8 distance,
//        const REAL8 M,
//        const REAL8 phic,
//        IMRPhenomDAmplitudeCoefficients *pAmp_host,
//        IMRPhenomDPhaseCoefficients *pPhi_host,
//        BBHPhenomCParams *PCparams_host,
//        PNPhasingSeries *pn_host,
//        NNLOanglecoeffs *angcoeffs_host,
//        SpinWeightedSphericalHarmonic_l2 *Y2m_host,
//        const REAL8 alphaNNLOoffset,
//        const REAL8 alpha0,
//        const REAL8 epsilonNNLOoffset,
//        IMRPhenomP_version_type IMRPhenomP_version,
//        AmpInsPrefactors *amp_prefactors_host,
//        PhiInsPrefactors *phi_prefactors_host,
//        COMPLEX16FrequencySeries **hptilde_host,
//        COMPLEX16FrequencySeries **hctilde_host,
//        REAL8 *phis_host,
//        int   *errcode){
//
//  // Copy inputs to device
//  IMRPhenomDAmplitudeCoefficients pAmp;
//  IMRPhenomDPhaseCoefficients pPhi;
//  BBHPhenomCParams PCparams;
//  PNPhasingSeries pn;
//  NNLOanglecoeffs angcoeffs;
//  SpinWeightedSphericalHarmonic_l2 Y2m;
//  AmpInsPrefactors amp_prefactors;
//  PhiInsPrefactors phi_prefactors;
//  try{
//      throw_on_cuda_error(cudaMemcpy(&pAmp,          pAmp_host,          sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&pPhi,          pPhi_host,          sizeof(IMRPhenomDPhaseCoefficients),     cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&PCparams,      PCparams_host,      sizeof(BBHPhenomCParams),                cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&pn,            pn_host,            sizeof(PNPhasingSeries),                 cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&angcoeffs,     angcoeffs_host,     sizeof(NNLOanglecoeffs),                 cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&Y2m,           Y2m_host,           sizeof(SpinWeightedSphericalHarmonic_l2),cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&amp_prefactors,amp_prefactors_host,sizeof(AmpInsPrefactors),                cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(&phi_prefactors,phi_prefactors_host,sizeof(PhiInsPrefactors),                cudaMemcpyHostToDevice),lalsimulation_cuda_exception::MEMCPY);
//  }
//  catch(const lalsimulation_cuda_exception e){
//      e.process_exception();
//  }
//
//  // Run kernel
//  try{
//    PhenomPCoreOneFrequency_cuda(L_fCut,
//          freqs,
//          offset,
//          eta,
//          chi1_l,
//          chi2_l,
//          chip,
//          distance,
//          M,
//          phic,
//          pAmp,
//          pPhi,
//          PCparams,
//          pn,
//          angcoeffs,
//          Y2m,
//          alphaNNLOoffset,
//          alpha0,
//          epsilonNNLOoffset,
//          IMRPhenomP_version,
//          amp_prefactors,
//          phi_prefactors,
//          hptilde,
//          hctilde,
//          phis);
//  }
//  // Alter this to return an error code on kernel errorcode exception
//  catch(const lalsimulation_cuda_exception e){
//      e.process_exception();
//  }
//
//  // Offload results
//  try{
//      throw_on_cuda_error(cudaMemcpy(hptilde_host,hptilde,L_fCut*sizeof(COMPLEX16FrequencySeries),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(hctilde_host,hctilde,L_fCut*sizeof(COMPLEX16FrequencySeries),cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
//      throw_on_cuda_error(cudaMemcpy(phis_host,   phis,   L_fCut*sizeof(REAL8),                   cudaMemcpyDeviceToHost),lalsimulation_cuda_exception::MEMCPY);
//  }
//  catch(const lalsimulation_cuda_exception e){
//      e.process_exception();
//  }
//}
//
//__global__
//void PhenomPCoreOneFrequency_cuda(UINT4 L_fCut,
//        REAL8Sequence *freqs,
//        UINT4 offset,
//        const REAL8 eta,
//        const REAL8 chi1_l,
//        const REAL8 chi2_l,
//        const REAL8 chip,
//        const REAL8 distance,
//        const REAL8 M,
//        const REAL8 phic,
//        IMRPhenomDAmplitudeCoefficients *pAmp,
//        IMRPhenomDPhaseCoefficients *pPhi,
//        BBHPhenomCParams *PCparams,
//        PNPhasingSeries *pn,
//        NNLOanglecoeffs *angcoeffs,
//        SpinWeightedSphericalHarmonic_l2 *Y2m,
//        const REAL8 alphaNNLOoffset,
//        const REAL8 alpha0,
//        const REAL8 epsilonNNLOoffset,
//        IMRPhenomP_version_type IMRPhenomP_version,
//        AmpInsPrefactors *amp_prefactors,
//        PhiInsPrefactors *phi_prefactors,
//        COMPLEX16FrequencySeries **hptilde,
//        COMPLEX16FrequencySeries **hctilde,
//        REAL8 *phis){
//
//    UINT4 i = (UINT4)(blockIdx.x*blockDim.x + threadIdx.x);
//    if(i < L_fCut){
//
//      COMPLEX16 hp_val = 0.0;
//      COMPLEX16 hc_val = 0.0;
//      REAL8 phasing = 0;
//      double f = freqs->data[i];
//      int j = i + offset; // shift index for frequency series if needed
//
//      // Generate the waveform 
//      int per_thread_errcode = PhenomPCoreOneFrequency(f,
//                                                       eta, chi1_l, chi2_l, chip, distance, M, phic,
//                                                       pAmp, pPhi, PCparams, pn, angcoeffs, Y2m,
//                                                       alphaNNLOoffset - alpha0, epsilonNNLOoffset,
//                                                       &hp_val, &hc_val, &phasing, IMRPhenomP_version, amp_prefactors, phi_prefactors);
//
//      // THROW EXCEPTION HERE INSTEAD
//      //if (per_thread_errcode != XLAL_SUCCESS) {
//      //  (*errcode) = per_thread_errcode;
//      //}
//
//      ((*hptilde)->data->data)[j] = hp_val;
//      ((*hctilde)->data->data)[j] = hc_val;
//
//      phis[i] = phasing;
//
//  }
//}

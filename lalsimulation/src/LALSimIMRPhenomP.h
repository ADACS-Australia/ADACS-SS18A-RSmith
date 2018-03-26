#ifndef _LALSIM_IMR_PHENOMP_H
#define _LALSIM_IMR_PHENOMP_H

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

#include <lal/LALStdlib.h>
#include <lal/LALSimIMR.h>
#include <lal/LALConstants.h>

#include "LALSimIMRPhenomC_internals.h"
#include "LALSimIMRPhenomD_internals.h"

#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>


/* CONSTANTS */

/**
 * Tolerance used below which numbers are treated as zero for the calculation of atan2
 */
#define MAX_TOL_ATAN 1.0e-15


/* ************************** PhenomP internal function prototypes *****************************/
/* atan2 wrapper that returns 0 when both magnitudes of x and y are below tol, otherwise it returns
   atan2(x, y) */
REAL8 atan2tol(REAL8 x, REAL8 y, REAL8 tol);

/* PhenomC parameters for modified ringdown: Uses final spin formula of Barausse & Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009 */
BBHPhenomCParams *ComputeIMRPhenomCParamsRDmod(
  const REAL8 m1,   /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,   /**< Mass of companion 2 (solar masses) */
  const REAL8 chi,  /**< Reduced aligned spin of the binary chi = (m1*chi1 + m2*chi2)/M */
  const REAL8 chip,  /**< Dimensionless spin in the orbital plane */
  LALDict *extraParams /**< linked list containing the extra testing GR parameters */
);

typedef struct tagNNLOanglecoeffs {
    REAL8 alphacoeff1; /* Coefficient of omega^(-1)   in alphaNNLO */
    REAL8 alphacoeff2; /* Coefficient of omega^(-2/3) in alphaNNLO */
    REAL8 alphacoeff3; /* Coefficient of omega^(-1/3) in alphaNNLO */
    REAL8 alphacoeff4; /* Coefficient of log(omega)   in alphaNNLO */
    REAL8 alphacoeff5; /* Coefficient of omega^(1/3)  in alphaNNLO */

    REAL8 epsiloncoeff1; /* Coefficient of omega^(-1)   in epsilonNNLO */
    REAL8 epsiloncoeff2; /* Coefficient of omega^(-2/3) in epsilonNNLO */
    REAL8 epsiloncoeff3; /* Coefficient of omega^(-1/3) in epsilonNNLO */
    REAL8 epsiloncoeff4; /* Coefficient of log(omega)   in epsilonNNLO */
    REAL8 epsiloncoeff5; /* Coefficient of omega^(1/3)  in epsilonNNLO */
} NNLOanglecoeffs;

void ComputeNNLOanglecoeffs(
  NNLOanglecoeffs *angcoeffs,  /**< Output: Structure to store results */
  const REAL8 q,               /**< Mass-ratio (convention q>1) */
  const REAL8 chil,            /**< Dimensionless aligned spin of the largest BH */
  const REAL8 chip             /**< Dimensionless spin component in the orbital plane */
);

typedef struct tagSpinWeightedSphericalHarmonic_l2 {
  COMPLEX16 Y2m2, Y2m1, Y20, Y21, Y22;
} SpinWeightedSphericalHarmonic_l2;

/* Internal core function to calculate PhenomP polarizations for a sequence of frequences. */
int PhenomPCore(
  COMPLEX16FrequencySeries **hptilde,   /**< Output: Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,   /**< Output: Frequency-domain waveform hx */
  const REAL8 chi1_l_in,                /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l_in,                /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                     /**< Effective spin in the orbital plane */
  const REAL8 thetaJ,                   /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 m1_SI_in,                 /**< Mass of companion 1 (kg) */
  const REAL8 m2_SI_in,                 /**< Mass of companion 2 (kg) */
  const REAL8 distance,                 /**< Distance of source (m) */
  const REAL8 alpha0,                   /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                     /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 f_ref,                    /**< Reference frequency */
  const REAL8Sequence *freqs,           /**< Frequency points at which to evaluate the waveform (Hz) */
  double deltaF,                        /**< Sampling frequency (Hz).
   * If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
  IMRPhenomP_version_type IMRPhenomP_version, /**< IMRPhenomPv1 uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
  LALDict *extraParams /**< linked list containing the extra testing GR parameters */
);

/* Internal core function to calculate PhenomP polarizations for a single frequency. */
int PhenomPCoreOneFrequency(
  const REAL8 fHz,                        /**< Frequency (Hz) */
  const REAL8 eta,                        /**< Symmetric mass ratio */
  const REAL8 chi1_l,                     /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l,                     /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                       /**< Dimensionless spin in the orbital plane */
  const REAL8 distance,                   /**< Distance of source (m) */
  const REAL8 M,                          /**< Total mass (Solar masses) */
  const REAL8 phic,                       /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  IMRPhenomDAmplitudeCoefficients *pAmp,  /**< Internal IMRPhenomD amplitude coefficients */
  IMRPhenomDPhaseCoefficients *pPhi,      /**< Internal IMRPhenomD phase coefficients */
  BBHPhenomCParams *PCparams,             /**< Internal PhenomC parameters */
  PNPhasingSeries *PNparams,              /**< PN inspiral phase coefficients */
  NNLOanglecoeffs *angcoeffs,             /**< Struct with PN coeffs for the NNLO angles */
  SpinWeightedSphericalHarmonic_l2 *Y2m,  /**< Struct of l=2 spherical harmonics of spin weight -2 */
  const REAL8 alphaoffset,                /**< f_ref dependent offset for alpha angle (azimuthal precession angle) */
  const REAL8 epsilonoffset,              /**< f_ref dependent offset for epsilon angle */
  COMPLEX16 *hp,                          /**< Output: tilde h_+ */
  COMPLEX16 *hc,                          /**< Output: tilde h_+ */
  REAL8 *phasing,                         /**< Output: overall phasing */
  const UINT4 IMRPhenomP_version,         /**< Version number: 1 uses IMRPhenomC, 2 uses IMRPhenomD */
  AmpInsPrefactors *amp_prefactors,       /**< pre-calculated (cached for saving runtime) coefficients for amplitude. See LALSimIMRPhenomD_internals.c*/
  PhiInsPrefactors *phi_prefactors        /**< pre-calculated (cached for saving runtime) coefficients for phase. See LALSimIMRPhenomD_internals.*/
);

void PhenomPCoreAllFrequencies(UINT4 L_fCut,
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
        REAL8 *phis,
        int   *errcode);

/* Simple 2PN version of L, without any spin terms expressed as a function of v */
REAL8 L2PNR(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta  /**< Symmetric mass-ratio */
);

REAL8 L2PNR_v1(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta  /**< Symmetric mass-ratio */
);

void WignerdCoefficients(
  REAL8 *cos_beta_half,   /**< Output: cos(beta/2) */
  REAL8 *sin_beta_half,   /**< Output: sin(beta/2) */
  const REAL8 v,          /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,         /**< Dimensionfull aligned spin */
  const REAL8 eta,        /**< Symmetric mass-ratio */
  const REAL8 Sp          /**< Dimensionfull spin component in the orbital plane */
);

void WignerdCoefficients_SmallAngleApproximation(
  REAL8 *cos_beta_half, /**< Output: cos(beta/2) */
  REAL8 *sin_beta_half, /**< Output: sin(beta/2) */
  const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,       /**< Dimensionfull aligned spin */
  const REAL8 eta,      /**< Symmetric mass-ratio */
  const REAL8 Sp        /**< Dimensionfull spin component in the orbital plane */
);

void CheckMaxOpeningAngle(
  const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
  const REAL8 chi1_l, /**< Aligned spin of BH 1 */
  const REAL8 chi2_l, /**< Aligned spin of BH 2 */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
);

REAL8 FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(
  const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
  const REAL8 chi1_l, /**< Aligned spin of BH 1 */
  const REAL8 chi2_l, /**< Aligned spin of BH 2 */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
);

REAL8 FinalSpinBarausse2009_all_spin_on_larger_BH(
  const REAL8 nu,     /**< Symmetric mass-ratio */
  const REAL8 chi,    /**< Effective aligned spin of the binary:  chi = (m1*chi1 + m2*chi2)/M  */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
);

REAL8 FinalSpinBarausse2009(  /* Barausse & Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009, arXiv:0904.2577 */
  const REAL8 nu,               /**< Symmetric mass-ratio */
  const REAL8 a1,               /**< |a_1| norm of dimensionless spin vector for BH 1 */
  const REAL8 a2,               /**< |a_2| norm of dimensionless spin vector for BH 2 */
  const REAL8 cos_alpha,        /**< cos(alpha) = \\hat a_1 . \\hat a_2 (Eq. 7) */
  const REAL8 cos_beta_tilde,   /**< cos(\\tilde beta)  = \\hat a_1 . \\hat L (Eq. 9) */
  const REAL8 cos_gamma_tilde   /**< cos(\\tilde gamma) = \\hat a_2 . \\hat L (Eq. 9)*/
);

bool approximately_equal(REAL8 x, REAL8 y, REAL8 epsilon);
void nudge(REAL8 *x, REAL8 X, REAL8 epsilon);

#if defined(__cplusplus) && defined(__NVCC__)
#include <string>
#include <cuda_runtime.h>

__global__
void PhenomPCoreOneFrequency_cuda(UINT4 L_fCut,
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
        REAL8 *phis);

// Host-side exception-handling routines
__host__ void  _throw_on_generic_error(bool check_failure,int implementation_code,  const std::string file, const std::string func, int line);
__host__ void  _throw_on_cuda_error   (cudaError_t cuda_code, int implementation_code,  const std::string file, const std::string func, int line);
__host__ void  _throw_on_kernel_error (int implementation_code, const std::string file, const std::string func, int line);
__host__ void  _check_for_cuda_error  (int implementation_code, const std::string file, const std::string func, int line);
__host__ void  _check_thread_sync     (int implementation_code, const std::string file, const std::string func, int line);
__host__ void  _throw_on_global_error (const std::string file, const std::string func, int line);
__host__ void  notify_of_global_error (int error_code);

// Wrap exception-handling calls with these macros to add exception location information to the error messages
// N.B.: The ',' in the execution configuration part of a CUDA kernel call confuses the pre-processor ... so
//       make sure to add ()'s around the kernel call when using throw_on_kernel_error()
#define throw_on_generic_error(check_failure,implementation_code){ _throw_on_generic_error(check_failure,implementation_code, __FILE__, __func__, __LINE__); }
#define throw_on_cuda_error(cuda_code,implementation_code)       { _throw_on_cuda_error ((cuda_code),implementation_code, __FILE__, __func__, __LINE__); }
#define throw_on_kernel_error(kernel_call,implementation_code)   { (kernel_call);_check_for_cuda_error(implementation_code, __FILE__, __func__, __LINE__); }
#define check_thread_sync(implementation_code)                   { _check_thread_sync(implementation_code,__FILE__, __func__, __LINE__); }
#define throw_on_global_error()                                  { _throw_on_global_error(__FILE__, __func__, __LINE__);}

// Define base exception classes
#include <sstream>
#include <string>
#define  GENERIC_CUDA_ERROR_CODE 160614
class lalsimulation_exception_base : public std::exception {
    protected:
        int         _error_code;
        std::string _message;
        std::string _file;
        std::string _func;
        int         _line;
        std::string _composition;
    public:
        // Constructor (C++ STL strings).
        explicit lalsimulation_exception_base(int code,const std::string& message,const std::string& file,const std::string& func,const int line):
          _error_code(code),
          _message(message),
          _file(file),
          _func(func),
          _line(line)
          {
            // Create the error message ...
            std::stringstream s;
            // ... add the error description to the exception message
            s << _message << " (code=";
            // ... add the error location to the exception message
            s << _error_code << ")";
            // Convert stream to string
            this->_composition = s.str();
          }

        // Destructor.  Virtual to allow for subclassing.
        virtual ~lalsimulation_exception_base() noexcept {}

        // The following functions return pointers.
        //    The underlying memory is possessed by the
        //    lalsimulation_exception object. Callers must
        //    not attempt to free the memory.
        virtual const char* what() const noexcept {
            return(_composition.c_str());
        }
        virtual const char* file() const noexcept {
            return(_file.c_str());
        }
        virtual const char* func() const noexcept {
            return(_func.c_str());
        }

        // Returns an integer expressing the error code.
        virtual int error_code() const noexcept {
           return(this->_error_code);
        }

        // Returns an integer expressing the line number of the error.
        virtual int line() const noexcept {
           return(this->_line);
        }

        // Call this method inside catch blocks to process the exception
        // THIS IS THE CODE TO MODIFY IF YOU WANT TO ADJUST THE WAY
        //    LALSIMULATION RESPONDS TO GPU ERRORS
        virtual void process_exception() const{
            // This loop aids in making the error reporting a bit cleaner in the log
            //for(int i_rank=0;i_rank<run_globals.mpi_size;i_rank++){
            //    if(i_rank==run_globals.mpi_rank){
            //        if(this->error_code()!=0)
            //            _mlog_error(this->what(),this->file(),this->func(),this->line());
            //    }
            //    MPI_Barrier(run_globals.mpi_comm);
            //}
            //if(this->error_code()!=0)
            //    ABORT(EXIT_FAILURE);
            //else
            //    ABORT(EXIT_SUCCESS);
        }
};

// Derived lalsimulation exception class for CUDA exceptions
//    Define all implementation error codes and associated error messages here
class lalsimulation_cuda_exception : public lalsimulation_exception_base {
    public:
        // List all the implementation exception codes here
        enum _code_list{
            INVALID_FILTER,
            GLOBAL,
            INIT,
            INIT_PBS_GPUFILE,
            MALLOC,
            FREE,
            MEMCPY,
            SYNC,
            KERNEL_CMPLX_AX,
            KERNEL_SET_ARRAY,
            KERNEL_FILTER,
            KERNEL_CHECK,
            KERNEL_MAIN_LOOP
            };
    private:
        std::string _set_message(int code){
            // List the exception message information for each implementation exception code here
            switch (code){
            case INVALID_FILTER:
                return("Invalid filter specified in run_globals.");
            case GLOBAL:
                return("Notified of error on differing rank.");
            case INIT:
                return("CUDA error while initializing device.");
            case INIT_PBS_GPUFILE:
                return("CUDA error while using PBS_GPUFILE to set device number");
            case MALLOC:
                return("CUDA error while calling cudaMalloc()");
            case FREE:
                return("CUDA error while calling cudaFree()");
            case MEMCPY:
                return("CUDA error while calling cudaMemcpy()");
            case SYNC:
                return("CUDA error while syncing device");
            // The following kernel error messages are meant to have
            //   modifiers placed in front of them to descriminate
            //   between CUDA errors and thread-sync errors.
            case KERNEL_CMPLX_AX:
                return("scalar-times-complex-vector kernel execution");
            case KERNEL_SET_ARRAY:
                return("initialize-vector-to-constant kernel execution");
            case KERNEL_FILTER:
                return("filter/convolution kernel execution");
            case KERNEL_CHECK:
                return("post-convolution-sanity-check kernel execution");
            case KERNEL_MAIN_LOOP:
                return("main find_HII_bubbles() kernel execution");
            // This should never happen.
            default:
                return("Undocumented CUDA error.");
            }
        }
    public:
        // This is the constructor used for most standard exception handling
        explicit lalsimulation_cuda_exception(int cuda_code,int implementation_code,const std::string file,const std::string func,const int line) :
                 lalsimulation_exception_base((cuda_code),_set_message(implementation_code),file,func,line) {
                     if(implementation_code!=GLOBAL) notify_of_global_error(cuda_code);

        }
        // This constructor deals with the case when we want to modify the _set_message() string.  This is
        //    used for specifying whether kernel errors are CUDA errors or thread-sync errors, for example.
        explicit lalsimulation_cuda_exception(int cuda_code,int implementation_code,const std::string& modifier,const std::string file,const std::string func,const int line) :
                 lalsimulation_exception_base((cuda_code),modifier+_set_message(implementation_code),file,func,line) {
                     if(implementation_code!=GLOBAL) notify_of_global_error(cuda_code);
        }
};
#endif

#endif	// of #ifndef _LALSIM_IMR_PHENOMP_H

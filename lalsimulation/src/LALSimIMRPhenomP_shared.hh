#ifndef LALSIMIMRPHENOMP_SHARED_H
#define LALSIMIMRPHENOMP_SHARED_H

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

#include <lal/LALStdlib.h>
#include <lal/LALSimIMR.h>
#include <lal/LALConstants.h>

#include <lal/LALSimIMRPhenomP.h>
#include <lal/LALSimIMRPhenomC_internals.h>
#include <lal/LALSimIMRPhenomD_internals.h>

#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>

//#if defined(LALSIMULATION_CUDA_ENABLED)
#if defined(__CUDACC__)
#define LALSIMULATION_CUDA_HOST   __host__
#define LALSIMULATION_CUDA_DEVICE __device__
#define LALSIMULATION_CUDA_GLOBAL __global__
#else
#define LALSIMULATION_CUDA_HOST  
#define LALSIMULATION_CUDA_DEVICE 
#define LALSIMULATION_CUDA_GLOBAL 
#endif
#define LALSIMULATION_CUDA_HOST_DEVICE LALSIMULATION_CUDA_HOST LALSIMULATION_CUDA_DEVICE
#define LALSIMULATION_CUDA_DEVICE_HOST LALSIMULATION_CUDA_HOST_DEVICE

#if !defined(LALSIMULATION_CUDA_ENABLED)
void *XLALPhenomPCore_buffer(UINT4 L_fCut_max);
void XLALfree_PhenomPCore_buffer(void *buf);
#else
#include <cuda_runtime.h>
typedef struct tagPhenomPCore_buffer_info{
   bool       init;
   REAL8     *freqs;
   COMPLEX16 *hptilde;
   COMPLEX16 *hctilde;
   REAL8     *phis;
   REAL8     *freqs_pinned;
   COMPLEX16 *hctilde_pinned;
   COMPLEX16 *hptilde_pinned;
   REAL8     *phis_pinned;
   IMRPhenomDAmplitudeCoefficients  *pAmp;
   IMRPhenomDPhaseCoefficients      *pPhi;
   BBHPhenomCParams                 *PCparams;
   PNPhasingSeries                  *pn;
   NNLOanglecoeffs                  *angcoeffs;
   SpinWeightedSphericalHarmonic_l2 *Y2m;
   AmpInsPrefactors                 *amp_prefactors;
   PhiInsPrefactors                 *phi_prefactors;
   UINT4         L_fCut_alloc;
   UINT4         n_streams;
   UINT4         n_streams_alloc;
   cudaStream_t *stream;
   UINT4        *stream_size;
   UINT4        *stream_offset;
} PhenomPCore_buffer_info;
LALSIMULATION_CUDA_HOST PhenomPCore_buffer_info *XLALPhenomPCore_buffer(UINT4 L_fCut_max);
LALSIMULATION_CUDA_HOST void XLALfree_PhenomPCore_buffer(PhenomPCore_buffer_info *buf);
#endif

#if defined(LALSIMULATION_CUDA_ENABLED)
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
        const void  *buf,
        int   *errcode);
#endif

#if defined(__CUDACC__)
#include <string>
#include <cuda_runtime.h>
__global__ void PhenomPCoreOneFrequency_cuda(UINT4 L_fCut,
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
        REAL8 *phis);

// Device-side exception-handling 
__device__ void inline cause_cuda_error();

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
            fprintf(stderr,"%s, reported from %s() [%s:%d]\n",this->what(),this->func(),this->file(),this->line());
            exit(1);
            //XLAL_ERROR_CUDA(XLAL_EFAILED,msg);
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
            STREAM_CREATE,
            STREAM_SYNC,
            STREAM_DESTROY,
            KERNEL_PHENOMPCOREONEFREQUENCY
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
            case STREAM_CREATE:
                return("CUDA error while creating device stream");
            case STREAM_SYNC:
                return("CUDA error while syncing device stream");
            case STREAM_DESTROY:
                return("CUDA error while freeing device stream");
            // The following kernel error messages are meant to have
            //   modifiers placed in front of them to descriminate
            //   between CUDA errors and thread-sync errors.
            case KERNEL_PHENOMPCOREONEFREQUENCY:
                return("PhenomPCoreOneFrequency_cuda kernel execution");
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

// Redefine LAL error handling macros for CUDA code
#if defined(__CUDA_ARCH__)
#define _XLAL_CHECK_CUDA(assertion,error_code,...) \
    do { \
        if (!(assertion)) { \
            cause_cuda_error(error_code); \
        } \
    } while (0)
#define _XLAL_ERROR_CUDA(error_code,...) \
    do { \
        cause_cuda_error(error_code); \
    } while (0)
#define _XLALPrintError(error_code,...) \
    do { } while (0)
#define XLAL_CHECK_CUDA      _XLAL_CHECK_CUDA
#define XLAL_CHECK_CUDA_VOID _XLAL_CHECK_CUDA
#define XLAL_ERROR_CUDA      _XLAL_ERROR_CUDA
#define XLALPrintError_CUDA  _XLALPrintError
#else
#define XLAL_CHECK_CUDA      XLAL_CHECK
#define XLAL_CHECK_CUDA_VOID XLAL_CHECK_VOID
#define XLAL_ERROR_CUDA      XLAL_ERROR
#define XLALPrintError_CUDA  XLALPrintError
#endif

LALSIMULATION_CUDA_HOST_DEVICE int PhenomPCoreOneFrequency(
  const REAL8 fHz,                            
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
  PNPhasingSeries *PNparams,                  
  NNLOanglecoeffs *angcoeffs,                 
  SpinWeightedSphericalHarmonic_l2 *Y2m,      
  const REAL8 alphaoffset,                    
  const REAL8 epsilonoffset,                  
  COMPLEX16 *hp,                              
  COMPLEX16 *hc,                              
  REAL8 *phasing,                             
  IMRPhenomP_version_type IMRPhenomP_version, 
  AmpInsPrefactors *amp_prefactors,           
  PhiInsPrefactors *phi_prefactors);

/* Simple 2PN version of L, without any spin terms expressed as a function of v */
LALSIMULATION_CUDA_HOST_DEVICE REAL8 L2PNR(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta  /**< Symmetric mass-ratio */
);

LALSIMULATION_CUDA_HOST_DEVICE REAL8 L2PNR_v1(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta  /**< Symmetric mass-ratio */
);
LALSIMULATION_CUDA_HOST_DEVICE void WignerdCoefficients(
  REAL8 *cos_beta_half,   /**< Output: cos(beta/2) */
  REAL8 *sin_beta_half,   /**< Output: sin(beta/2) */
  const REAL8 v,          /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,         /**< Dimensionfull aligned spin */
  const REAL8 eta,        /**< Symmetric mass-ratio */
  const REAL8 Sp          /**< Dimensionfull spin component in the orbital plane */
);

LALSIMULATION_CUDA_HOST_DEVICE void WignerdCoefficients_SmallAngleApproximation(
  REAL8 *cos_beta_half, /**< Output: cos(beta/2) */
  REAL8 *sin_beta_half, /**< Output: sin(beta/2) */
  const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,       /**< Dimensionfull aligned spin */
  const REAL8 eta,      /**< Symmetric mass-ratio */
  const REAL8 Sp        /**< Dimensionfull spin component in the orbital plane */
);

LALSIMULATION_CUDA_HOST_DEVICE int IMRPhenomCGenerateAmpPhase( REAL8 *amplitude, REAL8 *phasing, REAL8 f, REAL8 eta, const BBHPhenomCParams *params);

/**
 * must be called before the first usage of *p
 */
LALSIMULATION_CUDA_HOST_DEVICE LALSIMULATION_CUDA_HOST int init_useful_powers(UsefulPowers * p, REAL8 number);
LALSIMULATION_CUDA_HOST_DEVICE double IMRPhenDAmplitude(double f, IMRPhenomDAmplitudeCoefficients *p, UsefulPowers *powers_of_f, AmpInsPrefactors * prefactors);
LALSIMULATION_CUDA_HOST_DEVICE double IMRPhenDPhase(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, UsefulPowers *powers_of_f, PhiInsPrefactors * prefactors);
LALSIMULATION_CUDA_HOST_DEVICE REAL8 wPlus( const REAL8 f, const REAL8 f0, const REAL8 d, const BBHPhenomCParams *params );
LALSIMULATION_CUDA_HOST_DEVICE REAL8 wMinus( const REAL8 f, const REAL8 f0, const REAL8 d, const BBHPhenomCParams *params );
LALSIMULATION_CUDA_HOST_DEVICE bool StepFunc_boolean(const double t, const double t1);
LALSIMULATION_CUDA_HOST_DEVICE double AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors);
LALSIMULATION_CUDA_HOST_DEVICE double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p);
LALSIMULATION_CUDA_HOST_DEVICE double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
LALSIMULATION_CUDA_HOST_DEVICE double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
LALSIMULATION_CUDA_HOST_DEVICE double AmpIntAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
LALSIMULATION_CUDA_HOST_DEVICE double PhiInsAnsatzInt(double f, UsefulPowers * powers_of_Mf, PhiInsPrefactors * prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn);
LALSIMULATION_CUDA_HOST_DEVICE double PhiMRDAnsatzInt(double f, IMRPhenomDPhaseCoefficients *p);
LALSIMULATION_CUDA_HOST_DEVICE double PhiIntAnsatz(double f, IMRPhenomDPhaseCoefficients *p);

/**
 * calc square of number without floating point 'pow'
 */
LALSIMULATION_CUDA_HOST_DEVICE inline double pow_2_of(double number)
{
    return (number*number);
}

/**
 * calc cube of number without floating point 'pow'
 */
LALSIMULATION_CUDA_HOST_DEVICE inline double pow_3_of(double number)
{
    return (number*number*number);
}

/**
 * calc fourth power of number without floating point 'pow'
 */
LALSIMULATION_CUDA_HOST_DEVICE inline double pow_4_of(double number)
{
    double pow2 = pow_2_of(number);
    return pow2 * pow2;
}

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif

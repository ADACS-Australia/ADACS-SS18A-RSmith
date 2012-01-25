# 1 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.cpp"
# 1 "/home/kawies/dev-dbg/src/lalsuite/lalinspiral/src//"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.cpp"
# 1 "Chisq_GPU.cu"
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_types.h"
# 149 "/usr/lib/gcc/x86_64-linux-gnu/4.4.5/include/stddef.h" 3
typedef long ptrdiff_t;
# 211 "/usr/lib/gcc/x86_64-linux-gnu/4.4.5/include/stddef.h" 3
typedef unsigned long size_t;
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h" 1 3
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 1 3
# 42 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_types.h" 1 3
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_types.h" 3
enum cudaRoundMode
{
  cudaRoundNearest,
  cudaRoundZero,
  cudaRoundPosInf,
  cudaRoundMinInf
};
# 43 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 1 3
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
enum cudaError
{





  cudaSuccess = 0,





  cudaErrorMissingConfiguration = 1,





  cudaErrorMemoryAllocation = 2,





  cudaErrorInitializationError = 3,
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorLaunchFailure = 4,
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorPriorLaunchFailure = 5,
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorLaunchTimeout = 6,
# 159 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorLaunchOutOfResources = 7,





  cudaErrorInvalidDeviceFunction = 8,
# 174 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorInvalidConfiguration = 9,





  cudaErrorInvalidDevice = 10,





  cudaErrorInvalidValue = 11,





  cudaErrorInvalidPitchValue = 12,





  cudaErrorInvalidSymbol = 13,




  cudaErrorMapBufferObjectFailed = 14,




  cudaErrorUnmapBufferObjectFailed = 15,





  cudaErrorInvalidHostPointer = 16,





  cudaErrorInvalidDevicePointer = 17,





  cudaErrorInvalidTexture = 18,





  cudaErrorInvalidTextureBinding = 19,






  cudaErrorInvalidChannelDescriptor = 20,





  cudaErrorInvalidMemcpyDirection = 21,
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorAddressOfConstant = 22,
# 264 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorTextureFetchFailed = 23,
# 273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorTextureNotBound = 24,
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorSynchronizationError = 25,





  cudaErrorInvalidFilterSetting = 26,





  cudaErrorInvalidNormSetting = 27,







  cudaErrorMixedDeviceExecution = 28,







  cudaErrorCudartUnloading = 29,




  cudaErrorUnknown = 30,





  cudaErrorNotYetImplemented = 31,
# 330 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorMemoryValueTooLarge = 32,






  cudaErrorInvalidResourceHandle = 33,







  cudaErrorNotReady = 34,






  cudaErrorInsufficientDriver = 35,
# 365 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorSetOnActiveProcess = 36,





  cudaErrorInvalidSurface = 37,





  cudaErrorNoDevice = 38,





  cudaErrorECCUncorrectable = 39,




  cudaErrorSharedObjectSymbolNotFound = 40,




  cudaErrorSharedObjectInitFailed = 41,





  cudaErrorUnsupportedLimit = 42,





  cudaErrorDuplicateVariableName = 43,





  cudaErrorDuplicateTextureName = 44,





  cudaErrorDuplicateSurfaceName = 45,
# 426 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorDevicesUnavailable = 46,




  cudaErrorInvalidKernelImage = 47,







  cudaErrorNoKernelImageForDevice = 48,
# 448 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
  cudaErrorIncompatibleDriverContext = 49,




  cudaErrorStartupFailure = 0x7f,





  cudaErrorApiFailureBase = 10000
};





enum cudaChannelFormatKind
{
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3
};





struct cudaChannelFormatDesc
{
  int x;
  int y;
  int z;
  int w;
  enum cudaChannelFormatKind f;
};





struct cudaArray;





enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};






struct cudaPitchedPtr
{
  void *ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
};






struct cudaExtent
{
  size_t width;
  size_t height;
  size_t depth;
};






struct cudaPos
{
  size_t x;
  size_t y;
  size_t z;
};





struct cudaMemcpy3DParms
{
  struct cudaArray *srcArray;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;

  struct cudaArray *dstArray;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;

  struct cudaExtent extent;
  enum cudaMemcpyKind kind;
};





struct cudaGraphicsResource;





enum cudaGraphicsRegisterFlags
{
  cudaGraphicsRegisterFlagsNone = 0
};





enum cudaGraphicsMapFlags
{
  cudaGraphicsMapFlagsNone = 0,
  cudaGraphicsMapFlagsReadOnly = 1,
  cudaGraphicsMapFlagsWriteDiscard = 2
};





enum cudaGraphicsCubeFace {
  cudaGraphicsCubeFacePositiveX = 0x00,
  cudaGraphicsCubeFaceNegativeX = 0x01,
  cudaGraphicsCubeFacePositiveY = 0x02,
  cudaGraphicsCubeFaceNegativeY = 0x03,
  cudaGraphicsCubeFacePositiveZ = 0x04,
  cudaGraphicsCubeFaceNegativeZ = 0x05
};





struct cudaFuncAttributes
{





   size_t sharedSizeBytes;





   size_t constSizeBytes;




   size_t localSizeBytes;






   int maxThreadsPerBlock;




   int numRegs;






   int ptxVersion;






   int binaryVersion;

   int __cudaReserved[6];
};





enum cudaFuncCache
{
  cudaFuncCachePreferNone = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1 = 2
};





enum cudaComputeMode
{
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2
};





enum cudaLimit
{
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02
};





struct cudaDeviceProp
{
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture2D[2];
  int maxTexture3D[3];
  int maxTexture2DArray[3];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int tccDriver;
  int __cudaReserved[21];
};
# 768 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h" 3
typedef enum cudaError cudaError_t;





typedef struct CUstream_st *cudaStream_t;





typedef struct CUevent_st *cudaEvent_t;





typedef struct cudaGraphicsResource *cudaGraphicsResource_t;





typedef struct CUuuid_st cudaUUID_t;
# 44 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_types.h" 1 3
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_types.h" 3
enum cudaSurfaceBoundaryMode
{
  cudaBoundaryModeZero = 0,
  cudaBoundaryModeClamp = 1,
  cudaBoundaryModeTrap = 2
};





enum cudaSurfaceFormatMode
{
  cudaFormatModeForced = 0,
  cudaFormatModeAuto = 1
};





struct surfaceReference
{



  struct cudaChannelFormatDesc channelDesc;
};
# 45 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_types.h" 1 3
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_types.h" 3
enum cudaTextureAddressMode
{
  cudaAddressModeWrap = 0,
  cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2,
  cudaAddressModeBorder = 3
};





enum cudaTextureFilterMode
{
  cudaFilterModePoint = 0,
  cudaFilterModeLinear = 1
};





enum cudaTextureReadMode
{
  cudaReadModeElementType = 0,
  cudaReadModeNormalizedFloat = 1
};





struct textureReference
{



  int normalized;



  enum cudaTextureFilterMode filterMode;



  enum cudaTextureAddressMode addressMode[3];



  struct cudaChannelFormatDesc channelDesc;
  int __cudaReserved[16];
};
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 2 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 1 3
# 45 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 1 3
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 1 3
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 2 3
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 2 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/host_defines.h" 1 3
# 47 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 2 3
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 3
struct char1
{
  signed char x;
};


struct uchar1
{
  unsigned char x;
};


struct __attribute__((aligned(2))) char2
{
  signed char x, y;
};


struct __attribute__((aligned(2))) uchar2
{
  unsigned char x, y;
};


struct char3
{
  signed char x, y, z;
};


struct uchar3
{
  unsigned char x, y, z;
};


struct __attribute__((aligned(4))) char4
{
  signed char x, y, z, w;
};


struct __attribute__((aligned(4))) uchar4
{
  unsigned char x, y, z, w;
};


struct short1
{
  short x;
};


struct ushort1
{
  unsigned short x;
};


struct __attribute__((aligned(4))) short2
{
  short x, y;
};


struct __attribute__((aligned(4))) ushort2
{
  unsigned short x, y;
};


struct short3
{
  short x, y, z;
};


struct ushort3
{
  unsigned short x, y, z;
};


struct __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };


struct __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };


struct int1
{
  int x;
};


struct uint1
{
  unsigned int x;
};


struct __attribute__((aligned(8))) int2 { int x; int y; };


struct __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };


struct int3
{
  int x, y, z;
};


struct uint3
{
  unsigned int x, y, z;
};


struct __attribute__((aligned(16))) int4
{
  int x, y, z, w;
};


struct __attribute__((aligned(16))) uint4
{
  unsigned int x, y, z, w;
};


struct long1
{
  long int x;
};


struct ulong1
{
  unsigned long x;
};
# 229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 3
struct __attribute__((aligned(2*sizeof(long int)))) long2
{
  long int x, y;
};


struct __attribute__((aligned(2*sizeof(unsigned long int)))) ulong2
{
  unsigned long int x, y;
};




struct long3
{
  long int x, y, z;
};


struct ulong3
{
  unsigned long int x, y, z;
};


struct __attribute__((aligned(16))) long4
{
  long int x, y, z, w;
};


struct __attribute__((aligned(16))) ulong4
{
  unsigned long int x, y, z, w;
};


struct float1
{
  float x;
};


struct __attribute__((aligned(8))) float2 { float x; float y; };


struct float3
{
  float x, y, z;
};


struct __attribute__((aligned(16))) float4
{
  float x, y, z, w;
};


struct longlong1
{
  long long int x;
};


struct ulonglong1
{
  unsigned long long int x;
};


struct __attribute__((aligned(16))) longlong2
{
  long long int x, y;
};


struct __attribute__((aligned(16))) ulonglong2
{
  unsigned long long int x, y;
};


struct longlong3
{
  long long int x, y, z;
};


struct ulonglong3
{
  unsigned long long int x, y, z;
};


struct __attribute__((aligned(16))) longlong4
{
  long long int x, y, z ,w;
};


struct __attribute__((aligned(16))) ulonglong4
{
  unsigned long long int x, y, z, w;
};


struct double1
{
  double x;
};


struct __attribute__((aligned(16))) double2
{
  double x, y;
};


struct double3
{
  double x, y, z;
};


struct __attribute__((aligned(16))) double4
{
  double x, y, z, w;
};
# 366 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 3
typedef struct char1 char1;

typedef struct uchar1 uchar1;

typedef struct char2 char2;

typedef struct uchar2 uchar2;

typedef struct char3 char3;

typedef struct uchar3 uchar3;

typedef struct char4 char4;

typedef struct uchar4 uchar4;

typedef struct short1 short1;

typedef struct ushort1 ushort1;

typedef struct short2 short2;

typedef struct ushort2 ushort2;

typedef struct short3 short3;

typedef struct ushort3 ushort3;

typedef struct short4 short4;

typedef struct ushort4 ushort4;

typedef struct int1 int1;

typedef struct uint1 uint1;

typedef struct int2 int2;

typedef struct uint2 uint2;

typedef struct int3 int3;

typedef struct uint3 uint3;

typedef struct int4 int4;

typedef struct uint4 uint4;

typedef struct long1 long1;

typedef struct ulong1 ulong1;

typedef struct long2 long2;

typedef struct ulong2 ulong2;

typedef struct long3 long3;

typedef struct ulong3 ulong3;

typedef struct long4 long4;

typedef struct ulong4 ulong4;

typedef struct float1 float1;

typedef struct float2 float2;

typedef struct float3 float3;

typedef struct float4 float4;

typedef struct longlong1 longlong1;

typedef struct ulonglong1 ulonglong1;

typedef struct longlong2 longlong2;

typedef struct ulonglong2 ulonglong2;

typedef struct longlong3 longlong3;

typedef struct ulonglong3 ulonglong3;

typedef struct longlong4 longlong4;

typedef struct ulonglong4 ulonglong4;

typedef struct double1 double1;

typedef struct double2 double2;

typedef struct double3 double3;

typedef struct double4 double4;
# 469 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h" 3
struct dim3
{
    unsigned int x, y, z;

    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }

};


typedef struct dim3 dim3;
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/builtin_types.h" 2 3
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h" 2 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/storage_class.h" 1 3
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h" 2 3
# 213 "/usr/lib/gcc/x86_64-linux-gnu/4.4.5/include/stddef.h" 2 3
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 466 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 478 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 491 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 497 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 510 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 523 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 535 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 546 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 564 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 570 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 579 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 590 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 603 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 656 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 667 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 678 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 689 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 768 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 774 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 780 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 786 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 792 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_types.h"
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_types.h"
# 74 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_types.h"
# 84 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_types.h"
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_types.h"
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_types.h"
# 85 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_types.h"
# 95 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_types.h"
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 87 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 93 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 99 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 111 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 123 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 129 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 141 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 153 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 159 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 171 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 189 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 201 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 213 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 249 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 261 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 267 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 276 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 288 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 294 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 300 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 306 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 312 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 318 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 324 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 330 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 336 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 342 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 348 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 354 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 366 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 368 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 370 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 372 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 374 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 376 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 378 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 380 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 382 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 384 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 386 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 388 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 390 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 392 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 394 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 396 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 398 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 400 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 402 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 404 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 406 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 408 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 410 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 412 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 414 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 416 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 418 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 420 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 422 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 424 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 426 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 428 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 430 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 432 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 434 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 436 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 438 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 440 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 442 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 444 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 446 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 448 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 450 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 452 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 454 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 456 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 458 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 460 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 469 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 480 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_types.h"
# 115 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadExit();
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSynchronize();
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSetLimit(cudaLimit, size_t);
# 207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadGetLimit(size_t *, cudaLimit);
# 237 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadGetCacheConfig(cudaFuncCache *);
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSetCacheConfig(cudaFuncCache);
# 330 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetLastError();
# 373 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaPeekAtLastError();
# 387 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" const char *cudaGetErrorString(cudaError_t);
# 418 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceCount(int *);
# 536 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
# 555 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaChooseDevice(int *, const cudaDeviceProp *);
# 579 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDevice(int);
# 597 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDevice(int *);
# 626 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetValidDevices(int *, int);
# 677 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDeviceFlags(unsigned);
# 703 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *);
# 719 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t);
# 753 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
# 771 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t);
# 789 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamQuery(cudaStream_t);
# 821 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *);
# 852 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
# 885 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
# 914 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventQuery(cudaEvent_t);
# 946 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t);
# 966 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t);
# 1007 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
# 1046 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = (0), cudaStream_t = 0);
# 1073 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetupArgument(const void *, size_t, size_t);
# 1119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFuncSetCacheConfig(const char *, cudaFuncCache);
# 1154 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaLaunch(const char *);
# 1187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, const char *);
# 1209 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForDevice(double *);
# 1231 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForHost(double *);
# 1263 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc(void **, size_t);
# 1292 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocHost(void **, size_t);
# 1331 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
# 1370 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocArray(cudaArray **, const cudaChannelFormatDesc *, size_t, size_t = (0), unsigned = (0));
# 1394 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFree(void *);
# 1414 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFreeHost(void *);
# 1436 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFreeArray(cudaArray *);
# 1495 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaHostAlloc(void **, size_t, unsigned);
# 1522 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaHostGetDevicePointer(void **, void *, unsigned);
# 1541 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaHostGetFlags(unsigned *, void *);
# 1576 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3D(cudaPitchedPtr *, cudaExtent);
# 1626 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3DArray(cudaArray **, const cudaChannelFormatDesc *, cudaExtent, unsigned = (0));
# 1723 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *);
# 1828 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t = 0);
# 1847 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemGetInfo(size_t *, size_t *);
# 1880 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
# 1913 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToArray(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind);
# 1946 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromArray(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind);
# 1981 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
# 2023 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
# 2064 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
# 2105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DFromArray(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind);
# 2144 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
# 2179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyHostToDevice);
# 2213 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyDeviceToHost);
# 2256 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2298 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2340 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2391 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2441 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2491 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2535 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2578 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2600 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset(void *, int, size_t);
# 2626 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t);
# 2665 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset3D(cudaPitchedPtr, int, cudaExtent);
# 2692 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = 0);
# 2724 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset2DAsync(void *, size_t, int, size_t, size_t, cudaStream_t = 0);
# 2769 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset3DAsync(cudaPitchedPtr, int, cudaExtent, cudaStream_t = 0);
# 2796 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolAddress(void **, const char *);
# 2819 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolSize(size_t *, const char *);
# 2865 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t);
# 2897 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t, unsigned);
# 2932 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsMapResources(int, cudaGraphicsResource_t *, cudaStream_t = 0);
# 2963 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsUnmapResources(int, cudaGraphicsResource_t *, cudaStream_t = 0);
# 2992 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(void **, size_t *, cudaGraphicsResource_t);
# 3026 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray **, cudaGraphicsResource_t, unsigned, unsigned);
# 3059 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *, const cudaArray *);
# 3094 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind);
# 3136 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTexture(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t = (((2147483647) * 2U) + 1U));
# 3179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTexture2D(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t, size_t, size_t);
# 3207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTextureToArray(const textureReference *, const cudaArray *, const cudaChannelFormatDesc *);
# 3228 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaUnbindTexture(const textureReference *);
# 3253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *, const textureReference *);
# 3277 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetTextureReference(const textureReference **, const char *);
# 3310 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindSurfaceToArray(const surfaceReference *, const cudaArray *, const cudaChannelFormatDesc *);
# 3328 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSurfaceReference(const surfaceReference **, const char *);
# 3355 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaDriverGetVersion(int *);
# 3372 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaRuntimeGetVersion(int *);
# 3377 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetExportTable(const void **, const cudaUUID_t *);
# 93 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc()
# 94 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 95 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf()
# 99 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 102 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 103 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1()
# 106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 109 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2()
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 114 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 116 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
# 117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4()
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 121 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 123 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
# 124 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> ()
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 128 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(char)) * 8);
# 133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 137 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> ()
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 139 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 141 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 142 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 144 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> ()
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 146 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 149 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 151 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> ()
# 152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 153 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 156 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> ()
# 159 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> ()
# 166 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 169 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 172 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> ()
# 173 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 174 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 176 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> ()
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 181 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 184 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 186 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> ()
# 187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 188 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 191 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 193 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> ()
# 194 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 197 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 198 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> ()
# 201 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 204 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 205 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> ()
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 209 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 211 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 212 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 214 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> ()
# 215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 216 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 219 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 221 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> ()
# 222 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 223 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 225 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 226 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 228 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> ()
# 229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 232 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 233 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> ()
# 236 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 237 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 239 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> ()
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 244 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 246 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 249 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> ()
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 251 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 254 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 256 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> ()
# 257 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 258 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 261 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 263 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> ()
# 264 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 265 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 267 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 268 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 270 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> ()
# 271 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 272 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 274 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 275 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 277 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> ()
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 279 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 281 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 284 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> ()
# 285 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 286 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 288 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 289 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 291 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> ()
# 292 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 293 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 295 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 296 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 298 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> ()
# 299 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 300 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 302 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 303 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 365 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> ()
# 366 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 367 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 369 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 370 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 372 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> ()
# 373 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 374 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 376 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 377 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 379 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> ()
# 380 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 381 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 383 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
# 384 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 386 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> ()
# 387 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
{
# 388 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 390 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
# 391 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/channel_descriptor.h"
}
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
{
# 67 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
cudaPitchedPtr s;
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(s.ptr) = d;
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(s.pitch) = p;
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(s.xsize) = xsz;
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(s.ysize) = ysz;
# 74 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
return s;
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
}
# 92 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z)
# 93 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
{
# 94 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
cudaPos p;
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(p.x) = x;
# 97 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(p.y) = y;
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(p.z) = z;
# 100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
return p;
# 101 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
}
# 118 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
# 119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
{
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
cudaExtent e;
# 122 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(e.width) = w;
# 123 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(e.height) = h;
# 124 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
(e.depth) = d;
# 126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
return e;
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/driver_functions.h"
}
# 55 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline char1 make_char1(signed char x)
# 56 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 57 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
char1 t; (t.x) = x; return t;
# 58 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 60 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uchar1 make_uchar1(unsigned char x)
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 62 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uchar1 t; (t.x) = x; return t;
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline char2 make_char2(signed char x, signed char y)
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 67 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
char2 t; (t.x) = x; (t.y) = y; return t;
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uchar2 make_uchar2(unsigned char x, unsigned char y)
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uchar2 t; (t.x) = x; (t.y) = y; return t;
# 73 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline char3 make_char3(signed char x, signed char y, signed char z)
# 76 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 77 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 78 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 80 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 82 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 83 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 85 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w)
# 86 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 87 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 88 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 90 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
# 91 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 92 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 93 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 95 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline short1 make_short1(short x)
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 97 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
short1 t; (t.x) = x; return t;
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ushort1 make_ushort1(unsigned short x)
# 101 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 102 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ushort1 t; (t.x) = x; return t;
# 103 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline short2 make_short2(short x, short y)
# 106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
short2 t; (t.x) = x; (t.y) = y; return t;
# 108 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ushort2 make_ushort2(unsigned short x, unsigned short y)
# 111 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ushort2 t; (t.x) = x; (t.y) = y; return t;
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 115 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline short3 make_short3(short x, short y, short z)
# 116 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 118 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
# 121 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 122 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 123 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline short4 make_short4(short x, short y, short z, short w)
# 126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 128 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline int1 make_int1(int x)
# 136 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 137 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
int1 t; (t.x) = x; return t;
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uint1 make_uint1(unsigned x)
# 141 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 142 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uint1 t; (t.x) = x; return t;
# 143 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline int2 make_int2(int x, int y)
# 146 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
int2 t; (t.x) = x; (t.y) = y; return t;
# 148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uint2 make_uint2(unsigned x, unsigned y)
# 151 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uint2 t; (t.x) = x; (t.y) = y; return t;
# 153 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline int3 make_int3(int x, int y, int z)
# 156 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 157 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z)
# 161 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline int4 make_int4(int x, int y, int z, int w)
# 166 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 168 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w)
# 171 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 172 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 173 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 175 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline long1 make_long1(long x)
# 176 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
long1 t; (t.x) = x; return t;
# 178 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulong1 make_ulong1(unsigned long x)
# 181 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 182 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulong1 t; (t.x) = x; return t;
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 185 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline long2 make_long2(long x, long y)
# 186 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
long2 t; (t.x) = x; (t.y) = y; return t;
# 188 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulong2 make_ulong2(unsigned long x, unsigned long y)
# 191 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 192 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulong2 t; (t.x) = x; (t.y) = y; return t;
# 193 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline long3 make_long3(long x, long y, long z)
# 196 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 197 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 198 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z)
# 201 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 205 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline long4 make_long4(long x, long y, long z, long w)
# 206 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 210 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w)
# 211 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 212 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 213 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline float1 make_float1(float x)
# 216 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 217 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
float1 t; (t.x) = x; return t;
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 220 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline float2 make_float2(float x, float y)
# 221 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 222 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
float2 t; (t.x) = x; (t.y) = y; return t;
# 223 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 225 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline float3 make_float3(float x, float y, float z)
# 226 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 228 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline float4 make_float4(float x, float y, float z, float w)
# 231 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 232 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 233 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline longlong1 make_longlong1(long long x)
# 236 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 237 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
longlong1 t; (t.x) = x; return t;
# 238 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulonglong1 make_ulonglong1(unsigned long long x)
# 241 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulonglong1 t; (t.x) = x; return t;
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 245 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline longlong2 make_longlong2(long long x, long long y)
# 246 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
longlong2 t; (t.x) = x; (t.y) = y; return t;
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y)
# 251 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 252 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulonglong2 t; (t.x) = x; (t.y) = y; return t;
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline longlong3 make_longlong3(long long x, long long y, long long z)
# 256 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 257 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 258 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z)
# 261 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 262 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 263 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 265 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w)
# 266 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 267 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 268 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 270 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w)
# 271 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 272 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 275 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline double1 make_double1(double x)
# 276 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 277 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
double1 t; (t.x) = x; return t;
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 280 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline double2 make_double2(double x, double y)
# 281 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
double2 t; (t.x) = x; (t.y) = y; return t;
# 283 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 285 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline double3 make_double3(double x, double y, double z)
# 286 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 287 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 288 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 290 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
static inline double4 make_double4(double x, double y, double z, double w)
# 291 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
{
# 292 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 293 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/vector_functions.h"
}
# 43 "/usr/include/string.h" 3
extern "C" __attribute__((weak)) void *memcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 48 "/usr/include/string.h" 3
extern "C" void *memmove(void *, const void *, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 56 "/usr/include/string.h" 3
extern "C" void *memccpy(void *__restrict__, const void *__restrict__, int, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 64 "/usr/include/string.h" 3
extern "C" __attribute__((weak)) void *memset(void *, int, size_t) throw() __attribute__((nonnull(1)));
# 67 "/usr/include/string.h" 3
extern "C" int memcmp(const void *, const void *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 74 "/usr/include/string.h" 3
extern void *memchr(void *, int, size_t) throw() __asm__("memchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 76 "/usr/include/string.h" 3
extern const void *memchr(const void *, int, size_t) throw() __asm__("memchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 103 "/usr/include/string.h" 3
void *rawmemchr(void *, int) throw() __asm__("rawmemchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 105 "/usr/include/string.h" 3
const void *rawmemchr(const void *, int) throw() __asm__("rawmemchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 114 "/usr/include/string.h" 3
void *memrchr(void *, int, size_t) throw() __asm__("memrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 116 "/usr/include/string.h" 3
const void *memrchr(const void *, int, size_t) throw() __asm__("memrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 127 "/usr/include/string.h" 3
extern "C" char *strcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 130 "/usr/include/string.h" 3
extern "C" char *strncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 135 "/usr/include/string.h" 3
extern "C" char *strcat(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 138 "/usr/include/string.h" 3
extern "C" char *strncat(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 142 "/usr/include/string.h" 3
extern "C" int strcmp(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 145 "/usr/include/string.h" 3
extern "C" int strncmp(const char *, const char *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 149 "/usr/include/string.h" 3
extern "C" int strcoll(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 152 "/usr/include/string.h" 3
extern "C" size_t strxfrm(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(2)));
# 40 "/usr/include/xlocale.h" 3
extern "C" { typedef
# 28 "/usr/include/xlocale.h" 3
struct __locale_struct {
# 31 "/usr/include/xlocale.h" 3
struct locale_data *__locales[13];
# 34 "/usr/include/xlocale.h" 3
const unsigned short *__ctype_b;
# 35 "/usr/include/xlocale.h" 3
const int *__ctype_tolower;
# 36 "/usr/include/xlocale.h" 3
const int *__ctype_toupper;
# 39 "/usr/include/xlocale.h" 3
const char *__names[13];
# 40 "/usr/include/xlocale.h" 3
} *__locale_t; }
# 43 "/usr/include/xlocale.h" 3
extern "C" { typedef __locale_t locale_t; }
# 164 "/usr/include/string.h" 3
extern "C" int strcoll_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 167 "/usr/include/string.h" 3
extern "C" size_t strxfrm_l(char *, const char *, size_t, __locale_t) throw() __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 173 "/usr/include/string.h" 3
extern "C" char *strdup(const char *) throw() __attribute__((__malloc__)) __attribute__((nonnull(1)));
# 181 "/usr/include/string.h" 3
extern "C" char *strndup(const char *, size_t) throw() __attribute__((__malloc__)) __attribute__((nonnull(1)));
# 213 "/usr/include/string.h" 3
extern char *strchr(char *, int) throw() __asm__("strchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 215 "/usr/include/string.h" 3
extern const char *strchr(const char *, int) throw() __asm__("strchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 240 "/usr/include/string.h" 3
extern char *strrchr(char *, int) throw() __asm__("strrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 242 "/usr/include/string.h" 3
extern const char *strrchr(const char *, int) throw() __asm__("strrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 269 "/usr/include/string.h" 3
char *strchrnul(char *, int) throw() __asm__("strchrnul") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 271 "/usr/include/string.h" 3
const char *strchrnul(const char *, int) throw() __asm__("strchrnul") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 282 "/usr/include/string.h" 3
extern "C" size_t strcspn(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 286 "/usr/include/string.h" 3
extern "C" size_t strspn(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 292 "/usr/include/string.h" 3
extern char *strpbrk(char *, const char *) throw() __asm__("strpbrk") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 294 "/usr/include/string.h" 3
extern const char *strpbrk(const char *, const char *) throw() __asm__("strpbrk") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 319 "/usr/include/string.h" 3
extern char *strstr(char *, const char *) throw() __asm__("strstr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 321 "/usr/include/string.h" 3
extern const char *strstr(const char *, const char *) throw() __asm__("strstr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 346 "/usr/include/string.h" 3
extern "C" char *strtok(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(2)));
# 352 "/usr/include/string.h" 3
extern "C" char *__strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 357 "/usr/include/string.h" 3
extern "C" char *strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 365 "/usr/include/string.h" 3
char *strcasestr(char *, const char *) throw() __asm__("strcasestr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 367 "/usr/include/string.h" 3
const char *strcasestr(const char *, const char *) throw() __asm__("strcasestr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 380 "/usr/include/string.h" 3
extern "C" void *memmem(const void *, size_t, const void *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 386 "/usr/include/string.h" 3
extern "C" void *__mempcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 389 "/usr/include/string.h" 3
extern "C" void *mempcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 397 "/usr/include/string.h" 3
extern "C" size_t strlen(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 404 "/usr/include/string.h" 3
extern "C" size_t strnlen(const char *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 411 "/usr/include/string.h" 3
extern "C" char *strerror(int) throw();
# 436 "/usr/include/string.h" 3
extern "C" char *strerror_r(int, char *, size_t) throw() __attribute__((nonnull(2)));
# 443 "/usr/include/string.h" 3
extern "C" char *strerror_l(int, __locale_t) throw();
# 449 "/usr/include/string.h" 3
extern "C" void __bzero(void *, size_t) throw() __attribute__((nonnull(1)));
# 453 "/usr/include/string.h" 3
extern "C" void bcopy(const void *, void *, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 457 "/usr/include/string.h" 3
extern "C" void bzero(void *, size_t) throw() __attribute__((nonnull(1)));
# 460 "/usr/include/string.h" 3
extern "C" int bcmp(const void *, const void *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 467 "/usr/include/string.h" 3
extern char *index(char *, int) throw() __asm__("index") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 469 "/usr/include/string.h" 3
extern const char *index(const char *, int) throw() __asm__("index") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 495 "/usr/include/string.h" 3
extern char *rindex(char *, int) throw() __asm__("rindex") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 497 "/usr/include/string.h" 3
extern const char *rindex(const char *, int) throw() __asm__("rindex") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 521 "/usr/include/string.h" 3
extern "C" int ffs(int) throw() __attribute__((__const__));
# 526 "/usr/include/string.h" 3
extern "C" int ffsl(long) throw() __attribute__((__const__));
# 528 "/usr/include/string.h" 3
extern "C" int ffsll(long long) throw() __attribute__((__const__));
# 534 "/usr/include/string.h" 3
extern "C" int strcasecmp(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 538 "/usr/include/string.h" 3
extern "C" int strncasecmp(const char *, const char *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 545 "/usr/include/string.h" 3
extern "C" int strcasecmp_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 549 "/usr/include/string.h" 3
extern "C" int strncasecmp_l(const char *, const char *, size_t, __locale_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 557 "/usr/include/string.h" 3
extern "C" char *strsep(char **__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 564 "/usr/include/string.h" 3
extern "C" char *strsignal(int) throw();
# 567 "/usr/include/string.h" 3
extern "C" char *__stpcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 569 "/usr/include/string.h" 3
extern "C" char *stpcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 574 "/usr/include/string.h" 3
extern "C" char *__stpncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 577 "/usr/include/string.h" 3
extern "C" char *stpncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 584 "/usr/include/string.h" 3
extern "C" int strverscmp(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 588 "/usr/include/string.h" 3
extern "C" char *strfry(char *) throw() __attribute__((nonnull(1)));
# 591 "/usr/include/string.h" 3
extern "C" void *memfrob(void *, size_t) throw() __attribute__((nonnull(1)));
# 599 "/usr/include/string.h" 3
char *basename(char *) throw() __asm__("basename") __attribute__((nonnull(1)));
# 601 "/usr/include/string.h" 3
const char *basename(const char *) throw() __asm__("basename") __attribute__((nonnull(1)));
# 31 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned char __u_char; }
# 32 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned short __u_short; }
# 33 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __u_int; }
# 34 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __u_long; }
# 37 "/usr/include/bits/types.h" 3
extern "C" { typedef signed char __int8_t; }
# 38 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned char __uint8_t; }
# 39 "/usr/include/bits/types.h" 3
extern "C" { typedef signed short __int16_t; }
# 40 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned short __uint16_t; }
# 41 "/usr/include/bits/types.h" 3
extern "C" { typedef signed int __int32_t; }
# 42 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __uint32_t; }
# 44 "/usr/include/bits/types.h" 3
extern "C" { typedef signed long __int64_t; }
# 45 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __uint64_t; }
# 53 "/usr/include/bits/types.h" 3
extern "C" { typedef long __quad_t; }
# 54 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __u_quad_t; }
# 134 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __dev_t; }
# 135 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __uid_t; }
# 136 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __gid_t; }
# 137 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __ino_t; }
# 138 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __ino64_t; }
# 139 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __mode_t; }
# 140 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __nlink_t; }
# 141 "/usr/include/bits/types.h" 3
extern "C" { typedef long __off_t; }
# 142 "/usr/include/bits/types.h" 3
extern "C" { typedef long __off64_t; }
# 143 "/usr/include/bits/types.h" 3
extern "C" { typedef int __pid_t; }
# 144 "/usr/include/bits/types.h" 3
extern "C" { typedef struct { int __val[2]; } __fsid_t; }
# 145 "/usr/include/bits/types.h" 3
extern "C" { typedef long __clock_t; }
# 146 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __rlim_t; }
# 147 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __rlim64_t; }
# 148 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __id_t; }
# 149 "/usr/include/bits/types.h" 3
extern "C" { typedef long __time_t; }
# 150 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __useconds_t; }
# 151 "/usr/include/bits/types.h" 3
extern "C" { typedef long __suseconds_t; }
# 153 "/usr/include/bits/types.h" 3
extern "C" { typedef int __daddr_t; }
# 154 "/usr/include/bits/types.h" 3
extern "C" { typedef long __swblk_t; }
# 155 "/usr/include/bits/types.h" 3
extern "C" { typedef int __key_t; }
# 158 "/usr/include/bits/types.h" 3
extern "C" { typedef int __clockid_t; }
# 161 "/usr/include/bits/types.h" 3
extern "C" { typedef void *__timer_t; }
# 164 "/usr/include/bits/types.h" 3
extern "C" { typedef long __blksize_t; }
# 169 "/usr/include/bits/types.h" 3
extern "C" { typedef long __blkcnt_t; }
# 170 "/usr/include/bits/types.h" 3
extern "C" { typedef long __blkcnt64_t; }
# 173 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsblkcnt_t; }
# 174 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsblkcnt64_t; }
# 177 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsfilcnt_t; }
# 178 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned long __fsfilcnt64_t; }
# 180 "/usr/include/bits/types.h" 3
extern "C" { typedef long __ssize_t; }
# 184 "/usr/include/bits/types.h" 3
extern "C" { typedef __off64_t __loff_t; }
# 185 "/usr/include/bits/types.h" 3
extern "C" { typedef __quad_t *__qaddr_t; }
# 186 "/usr/include/bits/types.h" 3
extern "C" { typedef char *__caddr_t; }
# 189 "/usr/include/bits/types.h" 3
extern "C" { typedef long __intptr_t; }
# 192 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned __socklen_t; }
# 60 "/usr/include/time.h" 3
extern "C" { typedef __clock_t clock_t; }
# 76 "/usr/include/time.h" 3
extern "C" { typedef __time_t time_t; }
# 92 "/usr/include/time.h" 3
extern "C" { typedef __clockid_t clockid_t; }
# 104 "/usr/include/time.h" 3
extern "C" { typedef __timer_t timer_t; }
# 120 "/usr/include/time.h" 3
extern "C" { struct timespec {
# 122 "/usr/include/time.h" 3
__time_t tv_sec;
# 123 "/usr/include/time.h" 3
long tv_nsec;
# 124 "/usr/include/time.h" 3
}; }
# 133 "/usr/include/time.h" 3
extern "C" { struct tm {
# 135 "/usr/include/time.h" 3
int tm_sec;
# 136 "/usr/include/time.h" 3
int tm_min;
# 137 "/usr/include/time.h" 3
int tm_hour;
# 138 "/usr/include/time.h" 3
int tm_mday;
# 139 "/usr/include/time.h" 3
int tm_mon;
# 140 "/usr/include/time.h" 3
int tm_year;
# 141 "/usr/include/time.h" 3
int tm_wday;
# 142 "/usr/include/time.h" 3
int tm_yday;
# 143 "/usr/include/time.h" 3
int tm_isdst;
# 146 "/usr/include/time.h" 3
long tm_gmtoff;
# 147 "/usr/include/time.h" 3
const char *tm_zone;
# 152 "/usr/include/time.h" 3
}; }
# 161 "/usr/include/time.h" 3
extern "C" { struct itimerspec {
# 163 "/usr/include/time.h" 3
timespec it_interval;
# 164 "/usr/include/time.h" 3
timespec it_value;
# 165 "/usr/include/time.h" 3
}; }
# 168 "/usr/include/time.h" 3
struct sigevent;
# 174 "/usr/include/time.h" 3
extern "C" { typedef __pid_t pid_t; }
# 183 "/usr/include/time.h" 3
extern "C" __attribute__((weak)) clock_t clock() throw();
# 186 "/usr/include/time.h" 3
extern "C" time_t time(time_t *) throw();
# 189 "/usr/include/time.h" 3
extern "C" double difftime(time_t, time_t) throw() __attribute__((__const__));
# 193 "/usr/include/time.h" 3
extern "C" time_t mktime(tm *) throw();
# 199 "/usr/include/time.h" 3
extern "C" size_t strftime(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__) throw();
# 207 "/usr/include/time.h" 3
extern "C" char *strptime(const char *__restrict__, const char *__restrict__, tm *) throw();
# 217 "/usr/include/time.h" 3
extern "C" size_t strftime_l(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__, __locale_t) throw();
# 224 "/usr/include/time.h" 3
extern "C" char *strptime_l(const char *__restrict__, const char *__restrict__, tm *, __locale_t) throw();
# 233 "/usr/include/time.h" 3
extern "C" tm *gmtime(const time_t *) throw();
# 237 "/usr/include/time.h" 3
extern "C" tm *localtime(const time_t *) throw();
# 243 "/usr/include/time.h" 3
extern "C" tm *gmtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 248 "/usr/include/time.h" 3
extern "C" tm *localtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 255 "/usr/include/time.h" 3
extern "C" char *asctime(const tm *) throw();
# 258 "/usr/include/time.h" 3
extern "C" char *ctime(const time_t *) throw();
# 266 "/usr/include/time.h" 3
extern "C" char *asctime_r(const tm *__restrict__, char *__restrict__) throw();
# 270 "/usr/include/time.h" 3
extern "C" char *ctime_r(const time_t *__restrict__, char *__restrict__) throw();
# 276 "/usr/include/time.h" 3
extern "C" { extern char *__tzname[2]; }
# 277 "/usr/include/time.h" 3
extern "C" { extern int __daylight; }
# 278 "/usr/include/time.h" 3
extern "C" { extern long __timezone; }
# 283 "/usr/include/time.h" 3
extern "C" { extern char *tzname[2]; }
# 287 "/usr/include/time.h" 3
extern "C" void tzset() throw();
# 291 "/usr/include/time.h" 3
extern "C" { extern int daylight; }
# 292 "/usr/include/time.h" 3
extern "C" { extern long timezone; }
# 298 "/usr/include/time.h" 3
extern "C" int stime(const time_t *) throw();
# 313 "/usr/include/time.h" 3
extern "C" time_t timegm(tm *) throw();
# 316 "/usr/include/time.h" 3
extern "C" time_t timelocal(tm *) throw();
# 319 "/usr/include/time.h" 3
extern "C" int dysize(int) throw() __attribute__((__const__));
# 328 "/usr/include/time.h" 3
extern "C" int nanosleep(const timespec *, timespec *);
# 333 "/usr/include/time.h" 3
extern "C" int clock_getres(clockid_t, timespec *) throw();
# 336 "/usr/include/time.h" 3
extern "C" int clock_gettime(clockid_t, timespec *) throw();
# 339 "/usr/include/time.h" 3
extern "C" int clock_settime(clockid_t, const timespec *) throw();
# 347 "/usr/include/time.h" 3
extern "C" int clock_nanosleep(clockid_t, int, const timespec *, timespec *);
# 352 "/usr/include/time.h" 3
extern "C" int clock_getcpuclockid(pid_t, clockid_t *) throw();
# 357 "/usr/include/time.h" 3
extern "C" int timer_create(clockid_t, sigevent *__restrict__, timer_t *__restrict__) throw();
# 362 "/usr/include/time.h" 3
extern "C" int timer_delete(timer_t) throw();
# 365 "/usr/include/time.h" 3
extern "C" int timer_settime(timer_t, int, const itimerspec *__restrict__, itimerspec *__restrict__) throw();
# 370 "/usr/include/time.h" 3
extern "C" int timer_gettime(timer_t, itimerspec *) throw();
# 374 "/usr/include/time.h" 3
extern "C" int timer_getoverrun(timer_t) throw();
# 390 "/usr/include/time.h" 3
extern "C" { extern int getdate_err; }
# 399 "/usr/include/time.h" 3
extern "C" tm *getdate(const char *);
# 413 "/usr/include/time.h" 3
extern "C" int getdate_r(const char *__restrict__, tm *__restrict__);
# 57 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((weak)) clock_t clock() throw();
# 59 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((weak)) void *memset(void *, int, size_t) throw() __attribute__((nonnull(1)));
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/common_functions.h"
extern "C" __attribute__((weak)) void *memcpy(void *, const void *, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int abs(int) throw() __attribute__((__const__));
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long labs(long) throw() __attribute__((__const__));
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llabs(long long) throw() __attribute__((__const__));
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fabs(double) throw() __attribute__((__const__));
# 74 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fabsf(float) throw() __attribute__((__const__));
# 77 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int min(int, int);
# 79 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned umin(unsigned, unsigned);
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llmin(long long, long long);
# 83 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned long long ullmin(unsigned long long, unsigned long long);
# 85 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fminf(float, float) throw();
# 87 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fmin(double, double) throw();
# 90 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int max(int, int);
# 92 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned umax(unsigned, unsigned);
# 94 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llmax(long long, long long);
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned long long ullmax(unsigned long long, unsigned long long);
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fmaxf(float, float) throw();
# 100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fmax(double, double) throw();
# 103 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sin(double) throw();
# 105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sinf(float) throw();
# 108 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double cos(double) throw();
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float cosf(float) throw();
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) void sincos(double, double *, double *) throw();
# 115 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) void sincosf(float, float *, float *) throw();
# 118 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double tan(double) throw();
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float tanf(float) throw();
# 123 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sqrt(double) throw();
# 125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sqrtf(float) throw();
# 128 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double rsqrt(double);
# 130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float rsqrtf(float);
# 133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double exp2(double) throw();
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float exp2f(float) throw();
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double exp10(double) throw();
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float exp10f(float) throw();
# 143 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double expm1(double) throw();
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float expm1f(float) throw();
# 148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log2(double) throw();
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float log2f(float) throw();
# 153 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log10(double) throw();
# 155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float log10f(float) throw();
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log(double) throw();
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float logf(float) throw();
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log1p(double) throw();
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float log1pf(float) throw();
# 168 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double floor(double) throw() __attribute__((__const__));
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float floorf(float) throw() __attribute__((__const__));
# 173 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double exp(double) throw();
# 175 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float expf(float) throw();
# 178 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double cosh(double) throw();
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float coshf(float) throw();
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sinh(double) throw();
# 185 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sinhf(float) throw();
# 188 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double tanh(double) throw();
# 190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float tanhf(float) throw();
# 193 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double acosh(double) throw();
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float acoshf(float) throw();
# 198 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double asinh(double) throw();
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float asinhf(float) throw();
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double atanh(double) throw();
# 205 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float atanhf(float) throw();
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double ldexp(double, int) throw();
# 210 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float ldexpf(float, int) throw();
# 213 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double logb(double) throw();
# 215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float logbf(float) throw();
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int ilogb(double) throw();
# 220 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int ilogbf(float) throw();
# 223 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double scalbn(double, int) throw();
# 225 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float scalbnf(float, int) throw();
# 228 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double scalbln(double, long) throw();
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float scalblnf(float, long) throw();
# 233 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double frexp(double, int *) throw();
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float frexpf(float, int *) throw();
# 238 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double round(double) throw() __attribute__((__const__));
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float roundf(float) throw() __attribute__((__const__));
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lround(double) throw();
# 245 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lroundf(float) throw();
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llround(double) throw();
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llroundf(float) throw();
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double rint(double) throw();
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float rintf(float) throw();
# 258 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lrint(double) throw();
# 260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lrintf(float) throw();
# 263 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llrint(double) throw();
# 265 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llrintf(float) throw();
# 268 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double nearbyint(double) throw();
# 270 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float nearbyintf(float) throw();
# 273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double ceil(double) throw() __attribute__((__const__));
# 275 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float ceilf(float) throw() __attribute__((__const__));
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double trunc(double) throw() __attribute__((__const__));
# 280 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float truncf(float) throw() __attribute__((__const__));
# 283 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fdim(double, double) throw();
# 285 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fdimf(float, float) throw();
# 288 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double atan2(double, double) throw();
# 290 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float atan2f(float, float) throw();
# 293 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double atan(double) throw();
# 295 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float atanf(float) throw();
# 298 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double asin(double) throw();
# 300 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float asinf(float) throw();
# 303 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double acos(double) throw();
# 305 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float acosf(float) throw();
# 308 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double hypot(double, double) throw();
# 310 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float hypotf(float, float) throw();
# 313 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double cbrt(double) throw();
# 315 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float cbrtf(float) throw();
# 318 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double rcbrt(double);
# 320 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float rcbrtf(float);
# 323 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sinpi(double);
# 325 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sinpif(float);
# 328 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double pow(double, double) throw();
# 330 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float powf(float, float) throw();
# 333 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double modf(double, double *) throw();
# 335 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float modff(float, float *) throw();
# 338 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fmod(double, double) throw();
# 340 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fmodf(float, float) throw();
# 343 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double remainder(double, double) throw();
# 345 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float remainderf(float, float) throw();
# 348 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double remquo(double, double, int *) throw();
# 350 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float remquof(float, float, int *) throw();
# 353 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erf(double) throw();
# 355 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erff(float) throw();
# 358 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erfinv(double);
# 360 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erfinvf(float);
# 363 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erfc(double) throw();
# 365 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erfcf(float) throw();
# 368 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erfcinv(double);
# 370 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erfcinvf(float);
# 373 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double lgamma(double) throw();
# 375 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float lgammaf(float) throw();
# 378 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double tgamma(double) throw();
# 380 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float tgammaf(float) throw();
# 383 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double copysign(double, double) throw() __attribute__((__const__));
# 385 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float copysignf(float, float) throw() __attribute__((__const__));
# 388 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double nextafter(double, double) throw() __attribute__((__const__));
# 390 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float nextafterf(float, float) throw() __attribute__((__const__));
# 393 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double nan(const char *) throw() __attribute__((__const__));
# 395 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float nanf(const char *) throw() __attribute__((__const__));
# 398 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isinf(double) throw() __attribute__((__const__));
# 400 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isinff(float) throw() __attribute__((__const__));
# 403 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isnan(double) throw() __attribute__((__const__));
# 405 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isnanf(float) throw() __attribute__((__const__));
# 419 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __finite(double) throw() __attribute__((__const__));
# 421 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __finitef(float) throw() __attribute__((__const__));
# 423 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __signbit(double) throw() __attribute__((__const__));
# 428 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __signbitf(float) throw() __attribute__((__const__));
# 431 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fma(double, double, double) throw();
# 433 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fmaf(float, float, float) throw();
# 441 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __signbitl(long double) throw() __attribute__((__const__));
# 443 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isinfl(long double) throw() __attribute__((__const__));
# 445 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isnanl(long double) throw() __attribute__((__const__));
# 455 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __finitel(long double) throw() __attribute__((__const__));
# 31 "/usr/include/bits/mathdef.h" 3
extern "C" { typedef float float_t; }
# 32 "/usr/include/bits/mathdef.h" 3
extern "C" { typedef double double_t; }
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double acos(double) throw(); extern "C" double __acos(double) throw();
# 57 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double asin(double) throw(); extern "C" double __asin(double) throw();
# 59 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double atan(double) throw(); extern "C" double __atan(double) throw();
# 61 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double atan2(double, double) throw(); extern "C" double __atan2(double, double) throw();
# 64 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double cos(double) throw(); extern "C" double __cos(double) throw();
# 66 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double sin(double) throw(); extern "C" double __sin(double) throw();
# 68 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double tan(double) throw(); extern "C" double __tan(double) throw();
# 73 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double cosh(double) throw(); extern "C" double __cosh(double) throw();
# 75 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double sinh(double) throw(); extern "C" double __sinh(double) throw();
# 77 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double tanh(double) throw(); extern "C" double __tanh(double) throw();
# 82 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) void sincos(double, double *, double *) throw(); extern "C" void __sincos(double, double *, double *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double acosh(double) throw(); extern "C" double __acosh(double) throw();
# 91 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double asinh(double) throw(); extern "C" double __asinh(double) throw();
# 93 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double atanh(double) throw(); extern "C" double __atanh(double) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double exp(double) throw(); extern "C" double __exp(double) throw();
# 104 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double frexp(double, int *) throw(); extern "C" double __frexp(double, int *) throw();
# 107 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double ldexp(double, int) throw(); extern "C" double __ldexp(double, int) throw();
# 110 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log(double) throw(); extern "C" double __log(double) throw();
# 113 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log10(double) throw(); extern "C" double __log10(double) throw();
# 116 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double modf(double, double *) throw(); extern "C" double __modf(double, double *) throw();
# 121 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double exp10(double) throw(); extern "C" double __exp10(double) throw();
# 123 "/usr/include/bits/mathcalls.h" 3
extern "C" double pow10(double) throw(); extern "C" double __pow10(double) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double expm1(double) throw(); extern "C" double __expm1(double) throw();
# 132 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log1p(double) throw(); extern "C" double __log1p(double) throw();
# 135 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double logb(double) throw(); extern "C" double __logb(double) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double exp2(double) throw(); extern "C" double __exp2(double) throw();
# 145 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log2(double) throw(); extern "C" double __log2(double) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double pow(double, double) throw(); extern "C" double __pow(double, double) throw();
# 157 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double sqrt(double) throw(); extern "C" double __sqrt(double) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double hypot(double, double) throw(); extern "C" double __hypot(double, double) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double cbrt(double) throw(); extern "C" double __cbrt(double) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double ceil(double) throw() __attribute__((__const__)); extern "C" double __ceil(double) throw() __attribute__((__const__));
# 182 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fabs(double) throw() __attribute__((__const__)); extern "C" double __fabs(double) throw() __attribute__((__const__));
# 185 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double floor(double) throw() __attribute__((__const__)); extern "C" double __floor(double) throw() __attribute__((__const__));
# 188 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fmod(double, double) throw(); extern "C" double __fmod(double, double) throw();
# 193 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isinf(double) throw() __attribute__((__const__));
# 196 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __finite(double) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinf(double) throw() __attribute__((__const__));
# 205 "/usr/include/bits/mathcalls.h" 3
extern "C" int finite(double) throw() __attribute__((__const__));
# 208 "/usr/include/bits/mathcalls.h" 3
extern "C" double drem(double, double) throw(); extern "C" double __drem(double, double) throw();
# 212 "/usr/include/bits/mathcalls.h" 3
extern "C" double significand(double) throw(); extern "C" double __significand(double) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double copysign(double, double) throw() __attribute__((__const__)); extern "C" double __copysign(double, double) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double nan(const char *) throw() __attribute__((__const__)); extern "C" double __nan(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isnan(double) throw() __attribute__((__const__));
# 235 "/usr/include/bits/mathcalls.h" 3
extern "C" int isnan(double) throw() __attribute__((__const__));
# 238 "/usr/include/bits/mathcalls.h" 3
extern "C" double j0(double) throw(); extern "C" double __j0(double) throw();
# 239 "/usr/include/bits/mathcalls.h" 3
extern "C" double j1(double) throw(); extern "C" double __j1(double) throw();
# 240 "/usr/include/bits/mathcalls.h" 3
extern "C" double jn(int, double) throw(); extern "C" double __jn(int, double) throw();
# 241 "/usr/include/bits/mathcalls.h" 3
extern "C" double y0(double) throw(); extern "C" double __y0(double) throw();
# 242 "/usr/include/bits/mathcalls.h" 3
extern "C" double y1(double) throw(); extern "C" double __y1(double) throw();
# 243 "/usr/include/bits/mathcalls.h" 3
extern "C" double yn(int, double) throw(); extern "C" double __yn(int, double) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double erf(double) throw(); extern "C" double __erf(double) throw();
# 251 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double erfc(double) throw(); extern "C" double __erfc(double) throw();
# 252 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double lgamma(double) throw(); extern "C" double __lgamma(double) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double tgamma(double) throw(); extern "C" double __tgamma(double) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" double gamma(double) throw(); extern "C" double __gamma(double) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" double lgamma_r(double, int *) throw(); extern "C" double __lgamma_r(double, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double rint(double) throw(); extern "C" double __rint(double) throw();
# 283 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double nextafter(double, double) throw() __attribute__((__const__)); extern "C" double __nextafter(double, double) throw() __attribute__((__const__));
# 285 "/usr/include/bits/mathcalls.h" 3
extern "C" double nexttoward(double, long double) throw() __attribute__((__const__)); extern "C" double __nexttoward(double, long double) throw() __attribute__((__const__));
# 289 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double remainder(double, double) throw(); extern "C" double __remainder(double, double) throw();
# 293 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double scalbn(double, int) throw(); extern "C" double __scalbn(double, int) throw();
# 297 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int ilogb(double) throw(); extern "C" int __ilogb(double) throw();
# 302 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double scalbln(double, long) throw(); extern "C" double __scalbln(double, long) throw();
# 306 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double nearbyint(double) throw(); extern "C" double __nearbyint(double) throw();
# 310 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double round(double) throw() __attribute__((__const__)); extern "C" double __round(double) throw() __attribute__((__const__));
# 314 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double trunc(double) throw() __attribute__((__const__)); extern "C" double __trunc(double) throw() __attribute__((__const__));
# 319 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double remquo(double, double, int *) throw(); extern "C" double __remquo(double, double, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lrint(double) throw(); extern "C" long __lrint(double) throw();
# 327 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llrint(double) throw(); extern "C" long long __llrint(double) throw();
# 331 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lround(double) throw(); extern "C" long __lround(double) throw();
# 332 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llround(double) throw(); extern "C" long long __llround(double) throw();
# 336 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fdim(double, double) throw(); extern "C" double __fdim(double, double) throw();
# 339 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fmax(double, double) throw(); extern "C" double __fmax(double, double) throw();
# 342 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fmin(double, double) throw(); extern "C" double __fmin(double, double) throw();
# 346 "/usr/include/bits/mathcalls.h" 3
extern "C" int __fpclassify(double) throw() __attribute__((__const__));
# 350 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __signbit(double) throw() __attribute__((__const__));
# 355 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fma(double, double, double) throw(); extern "C" double __fma(double, double, double) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" double scalb(double, double) throw(); extern "C" double __scalb(double, double) throw();
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float acosf(float) throw(); extern "C" float __acosf(float) throw();
# 57 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float asinf(float) throw(); extern "C" float __asinf(float) throw();
# 59 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float atanf(float) throw(); extern "C" float __atanf(float) throw();
# 61 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float atan2f(float, float) throw(); extern "C" float __atan2f(float, float) throw();
# 64 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float cosf(float) throw();
# 66 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float sinf(float) throw();
# 68 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float tanf(float) throw();
# 73 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float coshf(float) throw(); extern "C" float __coshf(float) throw();
# 75 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float sinhf(float) throw(); extern "C" float __sinhf(float) throw();
# 77 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float tanhf(float) throw(); extern "C" float __tanhf(float) throw();
# 82 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) void sincosf(float, float *, float *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float acoshf(float) throw(); extern "C" float __acoshf(float) throw();
# 91 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float asinhf(float) throw(); extern "C" float __asinhf(float) throw();
# 93 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float atanhf(float) throw(); extern "C" float __atanhf(float) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float expf(float) throw();
# 104 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float frexpf(float, int *) throw(); extern "C" float __frexpf(float, int *) throw();
# 107 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float ldexpf(float, int) throw(); extern "C" float __ldexpf(float, int) throw();
# 110 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float logf(float) throw();
# 113 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float log10f(float) throw();
# 116 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float modff(float, float *) throw(); extern "C" float __modff(float, float *) throw();
# 121 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float exp10f(float) throw();
# 123 "/usr/include/bits/mathcalls.h" 3
extern "C" float pow10f(float) throw(); extern "C" float __pow10f(float) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float expm1f(float) throw(); extern "C" float __expm1f(float) throw();
# 132 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float log1pf(float) throw(); extern "C" float __log1pf(float) throw();
# 135 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float logbf(float) throw(); extern "C" float __logbf(float) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float exp2f(float) throw(); extern "C" float __exp2f(float) throw();
# 145 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float log2f(float) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float powf(float, float) throw();
# 157 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float sqrtf(float) throw(); extern "C" float __sqrtf(float) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float hypotf(float, float) throw(); extern "C" float __hypotf(float, float) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float cbrtf(float) throw(); extern "C" float __cbrtf(float) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float ceilf(float) throw() __attribute__((__const__)); extern "C" float __ceilf(float) throw() __attribute__((__const__));
# 182 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fabsf(float) throw() __attribute__((__const__)); extern "C" float __fabsf(float) throw() __attribute__((__const__));
# 185 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float floorf(float) throw() __attribute__((__const__)); extern "C" float __floorf(float) throw() __attribute__((__const__));
# 188 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fmodf(float, float) throw(); extern "C" float __fmodf(float, float) throw();
# 193 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isinff(float) throw() __attribute__((__const__));
# 196 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __finitef(float) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinff(float) throw() __attribute__((__const__));
# 205 "/usr/include/bits/mathcalls.h" 3
extern "C" int finitef(float) throw() __attribute__((__const__));
# 208 "/usr/include/bits/mathcalls.h" 3
extern "C" float dremf(float, float) throw(); extern "C" float __dremf(float, float) throw();
# 212 "/usr/include/bits/mathcalls.h" 3
extern "C" float significandf(float) throw(); extern "C" float __significandf(float) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float copysignf(float, float) throw() __attribute__((__const__)); extern "C" float __copysignf(float, float) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float nanf(const char *) throw() __attribute__((__const__)); extern "C" float __nanf(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isnanf(float) throw() __attribute__((__const__));
# 235 "/usr/include/bits/mathcalls.h" 3
extern "C" int isnanf(float) throw() __attribute__((__const__));
# 238 "/usr/include/bits/mathcalls.h" 3
extern "C" float j0f(float) throw(); extern "C" float __j0f(float) throw();
# 239 "/usr/include/bits/mathcalls.h" 3
extern "C" float j1f(float) throw(); extern "C" float __j1f(float) throw();
# 240 "/usr/include/bits/mathcalls.h" 3
extern "C" float jnf(int, float) throw(); extern "C" float __jnf(int, float) throw();
# 241 "/usr/include/bits/mathcalls.h" 3
extern "C" float y0f(float) throw(); extern "C" float __y0f(float) throw();
# 242 "/usr/include/bits/mathcalls.h" 3
extern "C" float y1f(float) throw(); extern "C" float __y1f(float) throw();
# 243 "/usr/include/bits/mathcalls.h" 3
extern "C" float ynf(int, float) throw(); extern "C" float __ynf(int, float) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float erff(float) throw(); extern "C" float __erff(float) throw();
# 251 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float erfcf(float) throw(); extern "C" float __erfcf(float) throw();
# 252 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float lgammaf(float) throw(); extern "C" float __lgammaf(float) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float tgammaf(float) throw(); extern "C" float __tgammaf(float) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" float gammaf(float) throw(); extern "C" float __gammaf(float) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" float lgammaf_r(float, int *) throw(); extern "C" float __lgammaf_r(float, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float rintf(float) throw(); extern "C" float __rintf(float) throw();
# 283 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float nextafterf(float, float) throw() __attribute__((__const__)); extern "C" float __nextafterf(float, float) throw() __attribute__((__const__));
# 285 "/usr/include/bits/mathcalls.h" 3
extern "C" float nexttowardf(float, long double) throw() __attribute__((__const__)); extern "C" float __nexttowardf(float, long double) throw() __attribute__((__const__));
# 289 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float remainderf(float, float) throw(); extern "C" float __remainderf(float, float) throw();
# 293 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float scalbnf(float, int) throw(); extern "C" float __scalbnf(float, int) throw();
# 297 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int ilogbf(float) throw(); extern "C" int __ilogbf(float) throw();
# 302 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float scalblnf(float, long) throw(); extern "C" float __scalblnf(float, long) throw();
# 306 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float nearbyintf(float) throw(); extern "C" float __nearbyintf(float) throw();
# 310 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float roundf(float) throw() __attribute__((__const__)); extern "C" float __roundf(float) throw() __attribute__((__const__));
# 314 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float truncf(float) throw() __attribute__((__const__)); extern "C" float __truncf(float) throw() __attribute__((__const__));
# 319 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float remquof(float, float, int *) throw(); extern "C" float __remquof(float, float, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lrintf(float) throw(); extern "C" long __lrintf(float) throw();
# 327 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llrintf(float) throw(); extern "C" long long __llrintf(float) throw();
# 331 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lroundf(float) throw(); extern "C" long __lroundf(float) throw();
# 332 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llroundf(float) throw(); extern "C" long long __llroundf(float) throw();
# 336 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fdimf(float, float) throw(); extern "C" float __fdimf(float, float) throw();
# 339 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fmaxf(float, float) throw(); extern "C" float __fmaxf(float, float) throw();
# 342 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fminf(float, float) throw(); extern "C" float __fminf(float, float) throw();
# 346 "/usr/include/bits/mathcalls.h" 3
extern "C" int __fpclassifyf(float) throw() __attribute__((__const__));
# 350 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __signbitf(float) throw() __attribute__((__const__));
# 355 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fmaf(float, float, float) throw(); extern "C" float __fmaf(float, float, float) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" float scalbf(float, float) throw(); extern "C" float __scalbf(float, float) throw();
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acosl(long double) throw(); extern "C" long double __acosl(long double) throw();
# 57 "/usr/include/bits/mathcalls.h" 3
extern "C" long double asinl(long double) throw(); extern "C" long double __asinl(long double) throw();
# 59 "/usr/include/bits/mathcalls.h" 3
extern "C" long double atanl(long double) throw(); extern "C" long double __atanl(long double) throw();
# 61 "/usr/include/bits/mathcalls.h" 3
extern "C" long double atan2l(long double, long double) throw(); extern "C" long double __atan2l(long double, long double) throw();
# 64 "/usr/include/bits/mathcalls.h" 3
extern "C" long double cosl(long double) throw(); extern "C" long double __cosl(long double) throw();
# 66 "/usr/include/bits/mathcalls.h" 3
extern "C" long double sinl(long double) throw(); extern "C" long double __sinl(long double) throw();
# 68 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tanl(long double) throw(); extern "C" long double __tanl(long double) throw();
# 73 "/usr/include/bits/mathcalls.h" 3
extern "C" long double coshl(long double) throw(); extern "C" long double __coshl(long double) throw();
# 75 "/usr/include/bits/mathcalls.h" 3
extern "C" long double sinhl(long double) throw(); extern "C" long double __sinhl(long double) throw();
# 77 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tanhl(long double) throw(); extern "C" long double __tanhl(long double) throw();
# 82 "/usr/include/bits/mathcalls.h" 3
extern "C" void sincosl(long double, long double *, long double *) throw(); extern "C" void __sincosl(long double, long double *, long double *) throw();
# 89 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acoshl(long double) throw(); extern "C" long double __acoshl(long double) throw();
# 91 "/usr/include/bits/mathcalls.h" 3
extern "C" long double asinhl(long double) throw(); extern "C" long double __asinhl(long double) throw();
# 93 "/usr/include/bits/mathcalls.h" 3
extern "C" long double atanhl(long double) throw(); extern "C" long double __atanhl(long double) throw();
# 101 "/usr/include/bits/mathcalls.h" 3
extern "C" long double expl(long double) throw(); extern "C" long double __expl(long double) throw();
# 104 "/usr/include/bits/mathcalls.h" 3
extern "C" long double frexpl(long double, int *) throw(); extern "C" long double __frexpl(long double, int *) throw();
# 107 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ldexpl(long double, int) throw(); extern "C" long double __ldexpl(long double, int) throw();
# 110 "/usr/include/bits/mathcalls.h" 3
extern "C" long double logl(long double) throw(); extern "C" long double __logl(long double) throw();
# 113 "/usr/include/bits/mathcalls.h" 3
extern "C" long double log10l(long double) throw(); extern "C" long double __log10l(long double) throw();
# 116 "/usr/include/bits/mathcalls.h" 3
extern "C" long double modfl(long double, long double *) throw(); extern "C" long double __modfl(long double, long double *) throw();
# 121 "/usr/include/bits/mathcalls.h" 3
extern "C" long double exp10l(long double) throw(); extern "C" long double __exp10l(long double) throw();
# 123 "/usr/include/bits/mathcalls.h" 3
extern "C" long double pow10l(long double) throw(); extern "C" long double __pow10l(long double) throw();
# 129 "/usr/include/bits/mathcalls.h" 3
extern "C" long double expm1l(long double) throw(); extern "C" long double __expm1l(long double) throw();
# 132 "/usr/include/bits/mathcalls.h" 3
extern "C" long double log1pl(long double) throw(); extern "C" long double __log1pl(long double) throw();
# 135 "/usr/include/bits/mathcalls.h" 3
extern "C" long double logbl(long double) throw(); extern "C" long double __logbl(long double) throw();
# 142 "/usr/include/bits/mathcalls.h" 3
extern "C" long double exp2l(long double) throw(); extern "C" long double __exp2l(long double) throw();
# 145 "/usr/include/bits/mathcalls.h" 3
extern "C" long double log2l(long double) throw(); extern "C" long double __log2l(long double) throw();
# 154 "/usr/include/bits/mathcalls.h" 3
extern "C" long double powl(long double, long double) throw(); extern "C" long double __powl(long double, long double) throw();
# 157 "/usr/include/bits/mathcalls.h" 3
extern "C" long double sqrtl(long double) throw(); extern "C" long double __sqrtl(long double) throw();
# 163 "/usr/include/bits/mathcalls.h" 3
extern "C" long double hypotl(long double, long double) throw(); extern "C" long double __hypotl(long double, long double) throw();
# 170 "/usr/include/bits/mathcalls.h" 3
extern "C" long double cbrtl(long double) throw(); extern "C" long double __cbrtl(long double) throw();
# 179 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ceill(long double) throw() __attribute__((__const__)); extern "C" long double __ceill(long double) throw() __attribute__((__const__));
# 182 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fabsl(long double) throw() __attribute__((__const__)); extern "C" long double __fabsl(long double) throw() __attribute__((__const__));
# 185 "/usr/include/bits/mathcalls.h" 3
extern "C" long double floorl(long double) throw() __attribute__((__const__)); extern "C" long double __floorl(long double) throw() __attribute__((__const__));
# 188 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fmodl(long double, long double) throw(); extern "C" long double __fmodl(long double, long double) throw();
# 193 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isinfl(long double) throw() __attribute__((__const__));
# 196 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __finitel(long double) throw() __attribute__((__const__));
# 202 "/usr/include/bits/mathcalls.h" 3
extern "C" int isinfl(long double) throw() __attribute__((__const__));
# 205 "/usr/include/bits/mathcalls.h" 3
extern "C" int finitel(long double) throw() __attribute__((__const__));
# 208 "/usr/include/bits/mathcalls.h" 3
extern "C" long double dreml(long double, long double) throw(); extern "C" long double __dreml(long double, long double) throw();
# 212 "/usr/include/bits/mathcalls.h" 3
extern "C" long double significandl(long double) throw(); extern "C" long double __significandl(long double) throw();
# 218 "/usr/include/bits/mathcalls.h" 3
extern "C" long double copysignl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __copysignl(long double, long double) throw() __attribute__((__const__));
# 225 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nanl(const char *) throw() __attribute__((__const__)); extern "C" long double __nanl(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isnanl(long double) throw() __attribute__((__const__));
# 235 "/usr/include/bits/mathcalls.h" 3
extern "C" int isnanl(long double) throw() __attribute__((__const__));
# 238 "/usr/include/bits/mathcalls.h" 3
extern "C" long double j0l(long double) throw(); extern "C" long double __j0l(long double) throw();
# 239 "/usr/include/bits/mathcalls.h" 3
extern "C" long double j1l(long double) throw(); extern "C" long double __j1l(long double) throw();
# 240 "/usr/include/bits/mathcalls.h" 3
extern "C" long double jnl(int, long double) throw(); extern "C" long double __jnl(int, long double) throw();
# 241 "/usr/include/bits/mathcalls.h" 3
extern "C" long double y0l(long double) throw(); extern "C" long double __y0l(long double) throw();
# 242 "/usr/include/bits/mathcalls.h" 3
extern "C" long double y1l(long double) throw(); extern "C" long double __y1l(long double) throw();
# 243 "/usr/include/bits/mathcalls.h" 3
extern "C" long double ynl(int, long double) throw(); extern "C" long double __ynl(int, long double) throw();
# 250 "/usr/include/bits/mathcalls.h" 3
extern "C" long double erfl(long double) throw(); extern "C" long double __erfl(long double) throw();
# 251 "/usr/include/bits/mathcalls.h" 3
extern "C" long double erfcl(long double) throw(); extern "C" long double __erfcl(long double) throw();
# 252 "/usr/include/bits/mathcalls.h" 3
extern "C" long double lgammal(long double) throw(); extern "C" long double __lgammal(long double) throw();
# 259 "/usr/include/bits/mathcalls.h" 3
extern "C" long double tgammal(long double) throw(); extern "C" long double __tgammal(long double) throw();
# 265 "/usr/include/bits/mathcalls.h" 3
extern "C" long double gammal(long double) throw(); extern "C" long double __gammal(long double) throw();
# 272 "/usr/include/bits/mathcalls.h" 3
extern "C" long double lgammal_r(long double, int *) throw(); extern "C" long double __lgammal_r(long double, int *) throw();
# 280 "/usr/include/bits/mathcalls.h" 3
extern "C" long double rintl(long double) throw(); extern "C" long double __rintl(long double) throw();
# 283 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nextafterl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nextafterl(long double, long double) throw() __attribute__((__const__));
# 285 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nexttowardl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nexttowardl(long double, long double) throw() __attribute__((__const__));
# 289 "/usr/include/bits/mathcalls.h" 3
extern "C" long double remainderl(long double, long double) throw(); extern "C" long double __remainderl(long double, long double) throw();
# 293 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalbnl(long double, int) throw(); extern "C" long double __scalbnl(long double, int) throw();
# 297 "/usr/include/bits/mathcalls.h" 3
extern "C" int ilogbl(long double) throw(); extern "C" int __ilogbl(long double) throw();
# 302 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalblnl(long double, long) throw(); extern "C" long double __scalblnl(long double, long) throw();
# 306 "/usr/include/bits/mathcalls.h" 3
extern "C" long double nearbyintl(long double) throw(); extern "C" long double __nearbyintl(long double) throw();
# 310 "/usr/include/bits/mathcalls.h" 3
extern "C" long double roundl(long double) throw() __attribute__((__const__)); extern "C" long double __roundl(long double) throw() __attribute__((__const__));
# 314 "/usr/include/bits/mathcalls.h" 3
extern "C" long double truncl(long double) throw() __attribute__((__const__)); extern "C" long double __truncl(long double) throw() __attribute__((__const__));
# 319 "/usr/include/bits/mathcalls.h" 3
extern "C" long double remquol(long double, long double, int *) throw(); extern "C" long double __remquol(long double, long double, int *) throw();
# 326 "/usr/include/bits/mathcalls.h" 3
extern "C" long lrintl(long double) throw(); extern "C" long __lrintl(long double) throw();
# 327 "/usr/include/bits/mathcalls.h" 3
extern "C" long long llrintl(long double) throw(); extern "C" long long __llrintl(long double) throw();
# 331 "/usr/include/bits/mathcalls.h" 3
extern "C" long lroundl(long double) throw(); extern "C" long __lroundl(long double) throw();
# 332 "/usr/include/bits/mathcalls.h" 3
extern "C" long long llroundl(long double) throw(); extern "C" long long __llroundl(long double) throw();
# 336 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fdiml(long double, long double) throw(); extern "C" long double __fdiml(long double, long double) throw();
# 339 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fmaxl(long double, long double) throw(); extern "C" long double __fmaxl(long double, long double) throw();
# 342 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fminl(long double, long double) throw(); extern "C" long double __fminl(long double, long double) throw();
# 346 "/usr/include/bits/mathcalls.h" 3
extern "C" int __fpclassifyl(long double) throw() __attribute__((__const__));
# 350 "/usr/include/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __signbitl(long double) throw() __attribute__((__const__));
# 355 "/usr/include/bits/mathcalls.h" 3
extern "C" long double fmal(long double, long double, long double) throw(); extern "C" long double __fmal(long double, long double, long double) throw();
# 364 "/usr/include/bits/mathcalls.h" 3
extern "C" long double scalbl(long double, long double) throw(); extern "C" long double __scalbl(long double, long double) throw();
# 161 "/usr/include/math.h" 3
extern "C" { extern int signgam; }
# 203 "/usr/include/math.h" 3
enum {
# 204 "/usr/include/math.h" 3
FP_NAN,
# 206 "/usr/include/math.h" 3
FP_INFINITE,
# 208 "/usr/include/math.h" 3
FP_ZERO,
# 210 "/usr/include/math.h" 3
FP_SUBNORMAL,
# 212 "/usr/include/math.h" 3
FP_NORMAL
# 214 "/usr/include/math.h" 3
};
# 302 "/usr/include/math.h" 3
extern "C" { typedef
# 296 "/usr/include/math.h" 3
enum {
# 297 "/usr/include/math.h" 3
_IEEE_ = (-1),
# 298 "/usr/include/math.h" 3
_SVID_ = 0,
# 299 "/usr/include/math.h" 3
_XOPEN_,
# 300 "/usr/include/math.h" 3
_POSIX_,
# 301 "/usr/include/math.h" 3
_ISOC_
# 302 "/usr/include/math.h" 3
} _LIB_VERSION_TYPE; }
# 307 "/usr/include/math.h" 3
extern "C" { extern _LIB_VERSION_TYPE _LIB_VERSION; }
# 318 "/usr/include/math.h" 3
extern "C" { struct __exception {
# 323 "/usr/include/math.h" 3
int type;
# 324 "/usr/include/math.h" 3
char *name;
# 325 "/usr/include/math.h" 3
double arg1;
# 326 "/usr/include/math.h" 3
double arg2;
# 327 "/usr/include/math.h" 3
double retval;
# 328 "/usr/include/math.h" 3
}; }
# 331 "/usr/include/math.h" 3
extern "C" int matherr(__exception *) throw();
# 67 "/usr/include/bits/waitstatus.h" 3
extern "C" { union wait {
# 69 "/usr/include/bits/waitstatus.h" 3
int w_status;
# 71 "/usr/include/bits/waitstatus.h" 3
struct {
# 73 "/usr/include/bits/waitstatus.h" 3
unsigned __w_termsig:7;
# 74 "/usr/include/bits/waitstatus.h" 3
unsigned __w_coredump:1;
# 75 "/usr/include/bits/waitstatus.h" 3
unsigned __w_retcode:8;
# 76 "/usr/include/bits/waitstatus.h" 3
unsigned:16;
# 84 "/usr/include/bits/waitstatus.h" 3
} __wait_terminated;
# 86 "/usr/include/bits/waitstatus.h" 3
struct {
# 88 "/usr/include/bits/waitstatus.h" 3
unsigned __w_stopval:8;
# 89 "/usr/include/bits/waitstatus.h" 3
unsigned __w_stopsig:8;
# 90 "/usr/include/bits/waitstatus.h" 3
unsigned:16;
# 97 "/usr/include/bits/waitstatus.h" 3
} __wait_stopped;
# 98 "/usr/include/bits/waitstatus.h" 3
}; }
# 102 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 99 "/usr/include/stdlib.h" 3
struct {
# 100 "/usr/include/stdlib.h" 3
int quot;
# 101 "/usr/include/stdlib.h" 3
int rem;
# 102 "/usr/include/stdlib.h" 3
} div_t; }
# 110 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 107 "/usr/include/stdlib.h" 3
struct {
# 108 "/usr/include/stdlib.h" 3
long quot;
# 109 "/usr/include/stdlib.h" 3
long rem;
# 110 "/usr/include/stdlib.h" 3
} ldiv_t; }
# 122 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 119 "/usr/include/stdlib.h" 3
struct {
# 120 "/usr/include/stdlib.h" 3
long long quot;
# 121 "/usr/include/stdlib.h" 3
long long rem;
# 122 "/usr/include/stdlib.h" 3
} lldiv_t; }
# 140 "/usr/include/stdlib.h" 3
extern "C" size_t __ctype_get_mb_cur_max() throw();
# 145 "/usr/include/stdlib.h" 3
extern "C" double atof(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 148 "/usr/include/stdlib.h" 3
extern "C" int atoi(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 151 "/usr/include/stdlib.h" 3
extern "C" long atol(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 158 "/usr/include/stdlib.h" 3
extern "C" long long atoll(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 165 "/usr/include/stdlib.h" 3
extern "C" double strtod(const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1)));
# 173 "/usr/include/stdlib.h" 3
extern "C" float strtof(const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1)));
# 176 "/usr/include/stdlib.h" 3
extern "C" long double strtold(const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1)));
# 184 "/usr/include/stdlib.h" 3
extern "C" long strtol(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 188 "/usr/include/stdlib.h" 3
extern "C" unsigned long strtoul(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 196 "/usr/include/stdlib.h" 3
extern "C" long long strtoq(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 201 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtouq(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 210 "/usr/include/stdlib.h" 3
extern "C" long long strtoll(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 215 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 240 "/usr/include/stdlib.h" 3
extern "C" long strtol_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 244 "/usr/include/stdlib.h" 3
extern "C" unsigned long strtoul_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 250 "/usr/include/stdlib.h" 3
extern "C" long long strtoll_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 256 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 261 "/usr/include/stdlib.h" 3
extern "C" double strtod_l(const char *__restrict__, char **__restrict__, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 265 "/usr/include/stdlib.h" 3
extern "C" float strtof_l(const char *__restrict__, char **__restrict__, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 269 "/usr/include/stdlib.h" 3
extern "C" long double strtold_l(const char *__restrict__, char **__restrict__, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 311 "/usr/include/stdlib.h" 3
extern "C" char *l64a(long) throw();
# 314 "/usr/include/stdlib.h" 3
extern "C" long a64l(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 35 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_char u_char; }
# 36 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_short u_short; }
# 37 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_int u_int; }
# 38 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_long u_long; }
# 39 "/usr/include/sys/types.h" 3
extern "C" { typedef __quad_t quad_t; }
# 40 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_quad_t u_quad_t; }
# 41 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsid_t fsid_t; }
# 46 "/usr/include/sys/types.h" 3
extern "C" { typedef __loff_t loff_t; }
# 50 "/usr/include/sys/types.h" 3
extern "C" { typedef __ino_t ino_t; }
# 57 "/usr/include/sys/types.h" 3
extern "C" { typedef __ino64_t ino64_t; }
# 62 "/usr/include/sys/types.h" 3
extern "C" { typedef __dev_t dev_t; }
# 67 "/usr/include/sys/types.h" 3
extern "C" { typedef __gid_t gid_t; }
# 72 "/usr/include/sys/types.h" 3
extern "C" { typedef __mode_t mode_t; }
# 77 "/usr/include/sys/types.h" 3
extern "C" { typedef __nlink_t nlink_t; }
# 82 "/usr/include/sys/types.h" 3
extern "C" { typedef __uid_t uid_t; }
# 88 "/usr/include/sys/types.h" 3
extern "C" { typedef __off_t off_t; }
# 95 "/usr/include/sys/types.h" 3
extern "C" { typedef __off64_t off64_t; }
# 105 "/usr/include/sys/types.h" 3
extern "C" { typedef __id_t id_t; }
# 110 "/usr/include/sys/types.h" 3
extern "C" { typedef __ssize_t ssize_t; }
# 116 "/usr/include/sys/types.h" 3
extern "C" { typedef __daddr_t daddr_t; }
# 117 "/usr/include/sys/types.h" 3
extern "C" { typedef __caddr_t caddr_t; }
# 123 "/usr/include/sys/types.h" 3
extern "C" { typedef __key_t key_t; }
# 137 "/usr/include/sys/types.h" 3
extern "C" { typedef __useconds_t useconds_t; }
# 141 "/usr/include/sys/types.h" 3
extern "C" { typedef __suseconds_t suseconds_t; }
# 151 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned long ulong; }
# 152 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned short ushort; }
# 153 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned uint; }
# 195 "/usr/include/sys/types.h" 3
extern "C" { typedef signed char int8_t; }
# 196 "/usr/include/sys/types.h" 3
extern "C" { typedef short int16_t; }
# 197 "/usr/include/sys/types.h" 3
extern "C" { typedef int int32_t; }
# 198 "/usr/include/sys/types.h" 3
extern "C" { typedef long int64_t; }
# 201 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned char u_int8_t; }
# 202 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned short u_int16_t; }
# 203 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned u_int32_t; }
# 204 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned long u_int64_t; }
# 206 "/usr/include/sys/types.h" 3
extern "C" { typedef long register_t; }
# 24 "/usr/include/bits/sigset.h" 3
extern "C" { typedef int __sig_atomic_t; }
# 32 "/usr/include/bits/sigset.h" 3
extern "C" { typedef
# 30 "/usr/include/bits/sigset.h" 3
struct {
# 31 "/usr/include/bits/sigset.h" 3
unsigned long __val[((1024) / ((8) * sizeof(unsigned long)))];
# 32 "/usr/include/bits/sigset.h" 3
} __sigset_t; }
# 38 "/usr/include/sys/select.h" 3
extern "C" { typedef __sigset_t sigset_t; }
# 69 "/usr/include/bits/time.h" 3
extern "C" { struct timeval {
# 71 "/usr/include/bits/time.h" 3
__time_t tv_sec;
# 72 "/usr/include/bits/time.h" 3
__suseconds_t tv_usec;
# 73 "/usr/include/bits/time.h" 3
}; }
# 55 "/usr/include/sys/select.h" 3
extern "C" { typedef long __fd_mask; }
# 78 "/usr/include/sys/select.h" 3
extern "C" { typedef
# 68 "/usr/include/sys/select.h" 3
struct {
# 72 "/usr/include/sys/select.h" 3
__fd_mask fds_bits[(1024 / (8 * ((int)sizeof(__fd_mask))))];
# 78 "/usr/include/sys/select.h" 3
} fd_set; }
# 85 "/usr/include/sys/select.h" 3
extern "C" { typedef __fd_mask fd_mask; }
# 109 "/usr/include/sys/select.h" 3
extern "C" int select(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, timeval *__restrict__);
# 121 "/usr/include/sys/select.h" 3
extern "C" int pselect(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, const timespec *__restrict__, const __sigset_t *__restrict__);
# 31 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_major(unsigned long long) throw();
# 34 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_minor(unsigned long long) throw();
# 37 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned long long gnu_dev_makedev(unsigned, unsigned) throw();
# 228 "/usr/include/sys/types.h" 3
extern "C" { typedef __blksize_t blksize_t; }
# 235 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt_t blkcnt_t; }
# 239 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsblkcnt_t fsblkcnt_t; }
# 243 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsfilcnt_t fsfilcnt_t; }
# 262 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt64_t blkcnt64_t; }
# 263 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsblkcnt64_t fsblkcnt64_t; }
# 264 "/usr/include/sys/types.h" 3
extern "C" { typedef __fsfilcnt64_t fsfilcnt64_t; }
# 50 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned long pthread_t; }
# 57 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 54 "/usr/include/bits/pthreadtypes.h" 3
union {
# 55 "/usr/include/bits/pthreadtypes.h" 3
char __size[56];
# 56 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 57 "/usr/include/bits/pthreadtypes.h" 3
} pthread_attr_t; }
# 65 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 61 "/usr/include/bits/pthreadtypes.h" 3
struct __pthread_internal_list {
# 63 "/usr/include/bits/pthreadtypes.h" 3
__pthread_internal_list *__prev;
# 64 "/usr/include/bits/pthreadtypes.h" 3
__pthread_internal_list *__next;
# 65 "/usr/include/bits/pthreadtypes.h" 3
} __pthread_list_t; }
# 104 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 77 "/usr/include/bits/pthreadtypes.h" 3
union {
# 78 "/usr/include/bits/pthreadtypes.h" 3
struct __pthread_mutex_s {
# 80 "/usr/include/bits/pthreadtypes.h" 3
int __lock;
# 81 "/usr/include/bits/pthreadtypes.h" 3
unsigned __count;
# 82 "/usr/include/bits/pthreadtypes.h" 3
int __owner;
# 84 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nusers;
# 88 "/usr/include/bits/pthreadtypes.h" 3
int __kind;
# 90 "/usr/include/bits/pthreadtypes.h" 3
int __spins;
# 91 "/usr/include/bits/pthreadtypes.h" 3
__pthread_list_t __list;
# 101 "/usr/include/bits/pthreadtypes.h" 3
} __data;
# 102 "/usr/include/bits/pthreadtypes.h" 3
char __size[40];
# 103 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 104 "/usr/include/bits/pthreadtypes.h" 3
} pthread_mutex_t; }
# 110 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 107 "/usr/include/bits/pthreadtypes.h" 3
union {
# 108 "/usr/include/bits/pthreadtypes.h" 3
char __size[4];
# 109 "/usr/include/bits/pthreadtypes.h" 3
int __align;
# 110 "/usr/include/bits/pthreadtypes.h" 3
} pthread_mutexattr_t; }
# 130 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 116 "/usr/include/bits/pthreadtypes.h" 3
union {
# 118 "/usr/include/bits/pthreadtypes.h" 3
struct {
# 119 "/usr/include/bits/pthreadtypes.h" 3
int __lock;
# 120 "/usr/include/bits/pthreadtypes.h" 3
unsigned __futex;
# 121 "/usr/include/bits/pthreadtypes.h" 3
__extension__ unsigned long long __total_seq;
# 122 "/usr/include/bits/pthreadtypes.h" 3
__extension__ unsigned long long __wakeup_seq;
# 123 "/usr/include/bits/pthreadtypes.h" 3
__extension__ unsigned long long __woken_seq;
# 124 "/usr/include/bits/pthreadtypes.h" 3
void *__mutex;
# 125 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nwaiters;
# 126 "/usr/include/bits/pthreadtypes.h" 3
unsigned __broadcast_seq;
# 127 "/usr/include/bits/pthreadtypes.h" 3
} __data;
# 128 "/usr/include/bits/pthreadtypes.h" 3
char __size[48];
# 129 "/usr/include/bits/pthreadtypes.h" 3
__extension__ long long __align;
# 130 "/usr/include/bits/pthreadtypes.h" 3
} pthread_cond_t; }
# 136 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 133 "/usr/include/bits/pthreadtypes.h" 3
union {
# 134 "/usr/include/bits/pthreadtypes.h" 3
char __size[4];
# 135 "/usr/include/bits/pthreadtypes.h" 3
int __align;
# 136 "/usr/include/bits/pthreadtypes.h" 3
} pthread_condattr_t; }
# 140 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned pthread_key_t; }
# 144 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef int pthread_once_t; }
# 189 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 151 "/usr/include/bits/pthreadtypes.h" 3
union {
# 154 "/usr/include/bits/pthreadtypes.h" 3
struct {
# 155 "/usr/include/bits/pthreadtypes.h" 3
int __lock;
# 156 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nr_readers;
# 157 "/usr/include/bits/pthreadtypes.h" 3
unsigned __readers_wakeup;
# 158 "/usr/include/bits/pthreadtypes.h" 3
unsigned __writer_wakeup;
# 159 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nr_readers_queued;
# 160 "/usr/include/bits/pthreadtypes.h" 3
unsigned __nr_writers_queued;
# 161 "/usr/include/bits/pthreadtypes.h" 3
int __writer;
# 162 "/usr/include/bits/pthreadtypes.h" 3
int __shared;
# 163 "/usr/include/bits/pthreadtypes.h" 3
unsigned long __pad1;
# 164 "/usr/include/bits/pthreadtypes.h" 3
unsigned long __pad2;
# 167 "/usr/include/bits/pthreadtypes.h" 3
unsigned __flags;
# 168 "/usr/include/bits/pthreadtypes.h" 3
} __data;
# 187 "/usr/include/bits/pthreadtypes.h" 3
char __size[56];
# 188 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 189 "/usr/include/bits/pthreadtypes.h" 3
} pthread_rwlock_t; }
# 195 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 192 "/usr/include/bits/pthreadtypes.h" 3
union {
# 193 "/usr/include/bits/pthreadtypes.h" 3
char __size[8];
# 194 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 195 "/usr/include/bits/pthreadtypes.h" 3
} pthread_rwlockattr_t; }
# 201 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef volatile int pthread_spinlock_t; }
# 210 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 207 "/usr/include/bits/pthreadtypes.h" 3
union {
# 208 "/usr/include/bits/pthreadtypes.h" 3
char __size[32];
# 209 "/usr/include/bits/pthreadtypes.h" 3
long __align;
# 210 "/usr/include/bits/pthreadtypes.h" 3
} pthread_barrier_t; }
# 216 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef
# 213 "/usr/include/bits/pthreadtypes.h" 3
union {
# 214 "/usr/include/bits/pthreadtypes.h" 3
char __size[4];
# 215 "/usr/include/bits/pthreadtypes.h" 3
int __align;
# 216 "/usr/include/bits/pthreadtypes.h" 3
} pthread_barrierattr_t; }
# 327 "/usr/include/stdlib.h" 3
extern "C" long random() throw();
# 330 "/usr/include/stdlib.h" 3
extern "C" void srandom(unsigned) throw();
# 336 "/usr/include/stdlib.h" 3
extern "C" char *initstate(unsigned, char *, size_t) throw() __attribute__((nonnull(2)));
# 341 "/usr/include/stdlib.h" 3
extern "C" char *setstate(char *) throw() __attribute__((nonnull(1)));
# 349 "/usr/include/stdlib.h" 3
extern "C" { struct random_data {
# 351 "/usr/include/stdlib.h" 3
int32_t *fptr;
# 352 "/usr/include/stdlib.h" 3
int32_t *rptr;
# 353 "/usr/include/stdlib.h" 3
int32_t *state;
# 354 "/usr/include/stdlib.h" 3
int rand_type;
# 355 "/usr/include/stdlib.h" 3
int rand_deg;
# 356 "/usr/include/stdlib.h" 3
int rand_sep;
# 357 "/usr/include/stdlib.h" 3
int32_t *end_ptr;
# 358 "/usr/include/stdlib.h" 3
}; }
# 360 "/usr/include/stdlib.h" 3
extern "C" int random_r(random_data *__restrict__, int32_t *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 363 "/usr/include/stdlib.h" 3
extern "C" int srandom_r(unsigned, random_data *) throw() __attribute__((nonnull(2)));
# 366 "/usr/include/stdlib.h" 3
extern "C" int initstate_r(unsigned, char *__restrict__, size_t, random_data *__restrict__) throw() __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 371 "/usr/include/stdlib.h" 3
extern "C" int setstate_r(char *__restrict__, random_data *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 380 "/usr/include/stdlib.h" 3
extern "C" int rand() throw();
# 382 "/usr/include/stdlib.h" 3
extern "C" void srand(unsigned) throw();
# 387 "/usr/include/stdlib.h" 3
extern "C" int rand_r(unsigned *) throw();
# 395 "/usr/include/stdlib.h" 3
extern "C" double drand48() throw();
# 396 "/usr/include/stdlib.h" 3
extern "C" double erand48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 399 "/usr/include/stdlib.h" 3
extern "C" long lrand48() throw();
# 400 "/usr/include/stdlib.h" 3
extern "C" long nrand48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 404 "/usr/include/stdlib.h" 3
extern "C" long mrand48() throw();
# 405 "/usr/include/stdlib.h" 3
extern "C" long jrand48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 409 "/usr/include/stdlib.h" 3
extern "C" void srand48(long) throw();
# 410 "/usr/include/stdlib.h" 3
extern "C" unsigned short *seed48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 412 "/usr/include/stdlib.h" 3
extern "C" void lcong48(unsigned short [7]) throw() __attribute__((nonnull(1)));
# 418 "/usr/include/stdlib.h" 3
extern "C" { struct drand48_data {
# 420 "/usr/include/stdlib.h" 3
unsigned short __x[3];
# 421 "/usr/include/stdlib.h" 3
unsigned short __old_x[3];
# 422 "/usr/include/stdlib.h" 3
unsigned short __c;
# 423 "/usr/include/stdlib.h" 3
unsigned short __init;
# 424 "/usr/include/stdlib.h" 3
unsigned long long __a;
# 425 "/usr/include/stdlib.h" 3
}; }
# 428 "/usr/include/stdlib.h" 3
extern "C" int drand48_r(drand48_data *__restrict__, double *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 430 "/usr/include/stdlib.h" 3
extern "C" int erand48_r(unsigned short [3], drand48_data *__restrict__, double *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 435 "/usr/include/stdlib.h" 3
extern "C" int lrand48_r(drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 438 "/usr/include/stdlib.h" 3
extern "C" int nrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 444 "/usr/include/stdlib.h" 3
extern "C" int mrand48_r(drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 447 "/usr/include/stdlib.h" 3
extern "C" int jrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 453 "/usr/include/stdlib.h" 3
extern "C" int srand48_r(long, drand48_data *) throw() __attribute__((nonnull(2)));
# 456 "/usr/include/stdlib.h" 3
extern "C" int seed48_r(unsigned short [3], drand48_data *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 459 "/usr/include/stdlib.h" 3
extern "C" int lcong48_r(unsigned short [7], drand48_data *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 471 "/usr/include/stdlib.h" 3
extern "C" void *malloc(size_t) throw() __attribute__((__malloc__));
# 473 "/usr/include/stdlib.h" 3
extern "C" void *calloc(size_t, size_t) throw() __attribute__((__malloc__));
# 485 "/usr/include/stdlib.h" 3
extern "C" void *realloc(void *, size_t) throw() __attribute__((__warn_unused_result__));
# 488 "/usr/include/stdlib.h" 3
extern "C" void free(void *) throw();
# 493 "/usr/include/stdlib.h" 3
extern "C" void cfree(void *) throw();
# 33 "/usr/include/alloca.h" 3
extern "C" void *alloca(size_t) throw();
# 502 "/usr/include/stdlib.h" 3
extern "C" void *valloc(size_t) throw() __attribute__((__malloc__));
# 507 "/usr/include/stdlib.h" 3
extern "C" int posix_memalign(void **, size_t, size_t) throw() __attribute__((nonnull(1)));
# 513 "/usr/include/stdlib.h" 3
extern "C" void abort() throw() __attribute__((__noreturn__));
# 517 "/usr/include/stdlib.h" 3
extern "C" int atexit(void (*)(void)) throw() __attribute__((nonnull(1)));
# 524 "/usr/include/stdlib.h" 3
int at_quick_exit(void (*)(void)) throw() __asm__("at_quick_exit") __attribute__((nonnull(1)));
# 535 "/usr/include/stdlib.h" 3
extern "C" int on_exit(void (*)(int, void *), void *) throw() __attribute__((nonnull(1)));
# 543 "/usr/include/stdlib.h" 3
extern "C" void exit(int) throw() __attribute__((__noreturn__));
# 551 "/usr/include/stdlib.h" 3
extern "C" void quick_exit(int) throw() __attribute__((__noreturn__));
# 559 "/usr/include/stdlib.h" 3
extern "C" void _Exit(int) throw() __attribute__((__noreturn__));
# 566 "/usr/include/stdlib.h" 3
extern "C" char *getenv(const char *) throw() __attribute__((nonnull(1)));
# 571 "/usr/include/stdlib.h" 3
extern "C" char *__secure_getenv(const char *) throw() __attribute__((nonnull(1)));
# 578 "/usr/include/stdlib.h" 3
extern "C" int putenv(char *) throw() __attribute__((nonnull(1)));
# 584 "/usr/include/stdlib.h" 3
extern "C" int setenv(const char *, const char *, int) throw() __attribute__((nonnull(2)));
# 588 "/usr/include/stdlib.h" 3
extern "C" int unsetenv(const char *) throw();
# 595 "/usr/include/stdlib.h" 3
extern "C" int clearenv() throw();
# 604 "/usr/include/stdlib.h" 3
extern "C" char *mktemp(char *) throw() __attribute__((nonnull(1)));
# 615 "/usr/include/stdlib.h" 3
extern "C" int mkstemp(char *) __attribute__((nonnull(1)));
# 625 "/usr/include/stdlib.h" 3
extern "C" int mkstemp64(char *) __attribute__((nonnull(1)));
# 637 "/usr/include/stdlib.h" 3
extern "C" int mkstemps(char *, int) __attribute__((nonnull(1)));
# 647 "/usr/include/stdlib.h" 3
extern "C" int mkstemps64(char *, int) __attribute__((nonnull(1)));
# 658 "/usr/include/stdlib.h" 3
extern "C" char *mkdtemp(char *) throw() __attribute__((nonnull(1)));
# 669 "/usr/include/stdlib.h" 3
extern "C" int mkostemp(char *, int) __attribute__((nonnull(1)));
# 679 "/usr/include/stdlib.h" 3
extern "C" int mkostemp64(char *, int) __attribute__((nonnull(1)));
# 689 "/usr/include/stdlib.h" 3
extern "C" int mkostemps(char *, int, int) __attribute__((nonnull(1)));
# 701 "/usr/include/stdlib.h" 3
extern "C" int mkostemps64(char *, int, int) __attribute__((nonnull(1)));
# 712 "/usr/include/stdlib.h" 3
extern "C" int system(const char *);
# 719 "/usr/include/stdlib.h" 3
extern "C" char *canonicalize_file_name(const char *) throw() __attribute__((nonnull(1)));
# 729 "/usr/include/stdlib.h" 3
extern "C" char *realpath(const char *__restrict__, char *__restrict__) throw();
# 737 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_fn_t)(const void *, const void *); }
# 740 "/usr/include/stdlib.h" 3
extern "C" { typedef __compar_fn_t comparison_fn_t; }
# 744 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_d_fn_t)(const void *, const void *, void *); }
# 750 "/usr/include/stdlib.h" 3
extern "C" void *bsearch(const void *, const void *, size_t, size_t, __compar_fn_t) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(5)));
# 756 "/usr/include/stdlib.h" 3
extern "C" void qsort(void *, size_t, size_t, __compar_fn_t) __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 759 "/usr/include/stdlib.h" 3
extern "C" void qsort_r(void *, size_t, size_t, __compar_d_fn_t, void *) __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 766 "/usr/include/stdlib.h" 3
extern "C" __attribute__((weak)) int abs(int) throw() __attribute__((__const__));
# 767 "/usr/include/stdlib.h" 3
extern "C" __attribute__((weak)) long labs(long) throw() __attribute__((__const__));
# 771 "/usr/include/stdlib.h" 3
extern "C" __attribute__((weak)) long long llabs(long long) throw() __attribute__((__const__));
# 780 "/usr/include/stdlib.h" 3
extern "C" div_t div(int, int) throw() __attribute__((__const__));
# 782 "/usr/include/stdlib.h" 3
extern "C" ldiv_t ldiv(long, long) throw() __attribute__((__const__));
# 788 "/usr/include/stdlib.h" 3
extern "C" lldiv_t lldiv(long long, long long) throw() __attribute__((__const__));
# 802 "/usr/include/stdlib.h" 3
extern "C" char *ecvt(double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 808 "/usr/include/stdlib.h" 3
extern "C" char *fcvt(double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 814 "/usr/include/stdlib.h" 3
extern "C" char *gcvt(double, int, char *) throw() __attribute__((nonnull(3)));
# 820 "/usr/include/stdlib.h" 3
extern "C" char *qecvt(long double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 823 "/usr/include/stdlib.h" 3
extern "C" char *qfcvt(long double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 826 "/usr/include/stdlib.h" 3
extern "C" char *qgcvt(long double, int, char *) throw() __attribute__((nonnull(3)));
# 832 "/usr/include/stdlib.h" 3
extern "C" int ecvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 835 "/usr/include/stdlib.h" 3
extern "C" int fcvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 839 "/usr/include/stdlib.h" 3
extern "C" int qecvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 843 "/usr/include/stdlib.h" 3
extern "C" int qfcvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 854 "/usr/include/stdlib.h" 3
extern "C" int mblen(const char *, size_t) throw();
# 857 "/usr/include/stdlib.h" 3
extern "C" int mbtowc(wchar_t *__restrict__, const char *__restrict__, size_t) throw();
# 861 "/usr/include/stdlib.h" 3
extern "C" int wctomb(char *, wchar_t) throw();
# 865 "/usr/include/stdlib.h" 3
extern "C" size_t mbstowcs(wchar_t *__restrict__, const char *__restrict__, size_t) throw();
# 868 "/usr/include/stdlib.h" 3
extern "C" size_t wcstombs(char *__restrict__, const wchar_t *__restrict__, size_t) throw();
# 879 "/usr/include/stdlib.h" 3
extern "C" int rpmatch(const char *) throw() __attribute__((nonnull(1)));
# 890 "/usr/include/stdlib.h" 3
extern "C" int getsubopt(char **__restrict__, char *const *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 899 "/usr/include/stdlib.h" 3
extern "C" void setkey(const char *) throw() __attribute__((nonnull(1)));
# 907 "/usr/include/stdlib.h" 3
extern "C" int posix_openpt(int);
# 915 "/usr/include/stdlib.h" 3
extern "C" int grantpt(int) throw();
# 919 "/usr/include/stdlib.h" 3
extern "C" int unlockpt(int) throw();
# 924 "/usr/include/stdlib.h" 3
extern "C" char *ptsname(int) throw();
# 931 "/usr/include/stdlib.h" 3
extern "C" int ptsname_r(int, char *, size_t) throw() __attribute__((nonnull(2)));
# 935 "/usr/include/stdlib.h" 3
extern "C" int getpt();
# 942 "/usr/include/stdlib.h" 3
extern "C" int getloadavg(double [], int) throw() __attribute__((nonnull(1)));
# 69 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 71 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Iterator, class _Container> class __normal_iterator;
# 74 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
}
# 76 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
namespace std __attribute__((visibility("default"))) {
# 78 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __true_type { };
# 79 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __false_type { };
# 81 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< bool __T0>
# 82 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __truth_type {
# 83 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type; };
# 86 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __truth_type< true> {
# 87 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type; };
# 91 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Sp, class _Tp>
# 92 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __traitor {
# 94 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = (((bool)_Sp::__value) || ((bool)_Tp::__value))};
# 95 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef typename __truth_type< __value> ::__type __type;
# 96 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 99 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class , class >
# 100 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __are_same {
# 102 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 103 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 104 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 106 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 107 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __are_same< _Tp, _Tp> {
# 109 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 110 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 111 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 114 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 115 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_void {
# 117 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 118 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 119 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 122 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_void< void> {
# 124 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 125 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 126 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 131 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 132 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_integer {
# 134 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 135 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 136 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 142 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< bool> {
# 144 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 145 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 146 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 149 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< char> {
# 151 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 152 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 153 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 156 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< signed char> {
# 158 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 159 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 160 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 163 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned char> {
# 165 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 166 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 167 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 171 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< wchar_t> {
# 173 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 174 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 175 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 195 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< short> {
# 197 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 198 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 199 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 202 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned short> {
# 204 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 205 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 206 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 209 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< int> {
# 211 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 212 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 213 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 216 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned> {
# 218 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 219 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 220 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 223 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< long> {
# 225 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 226 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 227 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 230 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned long> {
# 232 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 233 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 234 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 237 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< long long> {
# 239 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 240 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 241 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 244 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned long long> {
# 246 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 247 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 248 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 253 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 254 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_floating {
# 256 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 257 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 258 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 262 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_floating< float> {
# 264 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 265 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 266 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 269 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_floating< double> {
# 271 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 272 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 273 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 276 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_floating< long double> {
# 278 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 279 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 280 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 285 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 286 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_pointer {
# 288 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 289 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 290 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 292 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 293 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_pointer< _Tp *> {
# 295 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 296 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 297 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 302 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 303 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_normal_iterator {
# 305 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 306 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 307 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 309 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Iterator, class _Container>
# 310 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> > {
# 313 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 314 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 315 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 320 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 321 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> > {
# 323 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 328 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 329 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> > {
# 331 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 336 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 337 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> > {
# 339 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 344 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 345 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_char {
# 347 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 348 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 349 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 352 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_char< char> {
# 354 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 355 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 356 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 360 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_char< wchar_t> {
# 362 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 363 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 364 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 367 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 368 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_byte {
# 370 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 371 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 372 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 375 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_byte< char> {
# 377 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 378 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 379 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 382 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_byte< signed char> {
# 384 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 385 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 386 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 389 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_byte< unsigned char> {
# 391 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 392 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 393 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 398 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 399 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_move_iterator {
# 401 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 402 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 403 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 417 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
}
# 37 "/usr/include/c++/4.4/ext/type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 40 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< bool __T1, class >
# 41 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __enable_if {
# 42 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 44 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 45 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __enable_if< true, _Tp> {
# 46 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Tp __type; };
# 50 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< bool _Cond, class _Iftrue, class _Iffalse>
# 51 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __conditional_type {
# 52 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Iftrue __type; };
# 54 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Iftrue, class _Iffalse>
# 55 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __conditional_type< false, _Iftrue, _Iffalse> {
# 56 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Iffalse __type; };
# 60 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 61 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __add_unsigned {
# 64 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;
# 67 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
# 68 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 71 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< char> {
# 72 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned char __type; };
# 75 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< signed char> {
# 76 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned char __type; };
# 79 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< short> {
# 80 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned short __type; };
# 83 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< int> {
# 84 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned __type; };
# 87 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< long> {
# 88 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned long __type; };
# 91 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< long long> {
# 92 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned long long __type; };
# 96 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< bool> ;
# 99 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< wchar_t> ;
# 103 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 104 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __remove_unsigned {
# 107 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;
# 110 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
# 111 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 114 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< char> {
# 115 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef signed char __type; };
# 118 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned char> {
# 119 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef signed char __type; };
# 122 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned short> {
# 123 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef short __type; };
# 126 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned> {
# 127 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef int __type; };
# 130 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned long> {
# 131 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef long __type; };
# 134 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned long long> {
# 135 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef long long __type; };
# 139 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< bool> ;
# 142 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< wchar_t> ;
# 146 "/usr/include/c++/4.4/ext/type_traits.h" 3
template < typename _Type >
    inline bool
    __is_null_pointer ( _Type * __ptr )
    { return __ptr == 0; }
# 151 "/usr/include/c++/4.4/ext/type_traits.h" 3
template < typename _Type >
    inline bool
    __is_null_pointer ( _Type )
    { return false; }
# 158 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, bool __T2 = std::__is_integer< _Tp> ::__value>
# 159 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote {
# 160 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef double __type; };
# 162 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 163 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote< _Tp, false> {
# 164 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Tp __type; };
# 166 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, class _Up>
# 167 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote_2 {
# 170 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 171 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 174 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef __typeof__(__type1() + __type2()) __type;
# 175 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 177 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, class _Up, class _Vp>
# 178 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote_3 {
# 181 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 182 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 183 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
# 186 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef __typeof__((__type1() + __type2()) + __type3()) __type;
# 187 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 189 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, class _Up, class _Vp, class _Wp>
# 190 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote_4 {
# 193 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 194 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 195 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
# 196 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Wp, std::__is_integer< _Wp> ::__value> ::__type __type4;
# 199 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef __typeof__(((__type1() + __type2()) + __type3()) + __type4()) __type;
# 200 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 202 "/usr/include/c++/4.4/ext/type_traits.h" 3
}
# 77 "/usr/include/c++/4.4/cmath" 3
namespace std __attribute__((visibility("default"))) {
# 81 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    _Tp __cmath_power ( _Tp, unsigned int );
# 84 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline _Tp
    __pow_helper ( _Tp __x, int __n )
    {
      return __n < 0
        ? _Tp ( 1 ) / __cmath_power ( __x, - __n )
        : __cmath_power ( __x, __n );
    }
# 94 "/usr/include/c++/4.4/cmath" 3
inline double abs(double __x)
# 95 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabs(__x); }
# 98 "/usr/include/c++/4.4/cmath" 3
inline float abs(float __x)
# 99 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsf(__x); }
# 102 "/usr/include/c++/4.4/cmath" 3
inline long double abs(long double __x)
# 103 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsl(__x); }
# 105 "/usr/include/c++/4.4/cmath" 3
using ::acos;
# 108 "/usr/include/c++/4.4/cmath" 3
inline float acos(float __x)
# 109 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_acosf(__x); }
# 112 "/usr/include/c++/4.4/cmath" 3
inline long double acos(long double __x)
# 113 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_acosl(__x); }
# 115 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    acos ( _Tp __x )
    { return __builtin_acos ( __x ); }
# 121 "/usr/include/c++/4.4/cmath" 3
using ::asin;
# 124 "/usr/include/c++/4.4/cmath" 3
inline float asin(float __x)
# 125 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_asinf(__x); }
# 128 "/usr/include/c++/4.4/cmath" 3
inline long double asin(long double __x)
# 129 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_asinl(__x); }
# 131 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    asin ( _Tp __x )
    { return __builtin_asin ( __x ); }
# 137 "/usr/include/c++/4.4/cmath" 3
using ::atan;
# 140 "/usr/include/c++/4.4/cmath" 3
inline float atan(float __x)
# 141 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atanf(__x); }
# 144 "/usr/include/c++/4.4/cmath" 3
inline long double atan(long double __x)
# 145 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atanl(__x); }
# 147 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    atan ( _Tp __x )
    { return __builtin_atan ( __x ); }
# 153 "/usr/include/c++/4.4/cmath" 3
using ::atan2;
# 156 "/usr/include/c++/4.4/cmath" 3
inline float atan2(float __y, float __x)
# 157 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atan2f(__y, __x); }
# 160 "/usr/include/c++/4.4/cmath" 3
inline long double atan2(long double __y, long double __x)
# 161 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atan2l(__y, __x); }
# 163 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp, typename _Up >
    inline
    typename __gnu_cxx :: __promote_2 <
    typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value
        && __is_arithmetic < _Up > :: __value,
        _Tp > :: __type, _Up > :: __type
    atan2 ( _Tp __y, _Up __x )
    {
      typedef typename __gnu_cxx :: __promote_2 < _Tp, _Up > :: __type __type;
      return atan2 ( __type ( __y ), __type ( __x ) );
    }
# 175 "/usr/include/c++/4.4/cmath" 3
using ::ceil;
# 178 "/usr/include/c++/4.4/cmath" 3
inline float ceil(float __x)
# 179 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ceilf(__x); }
# 182 "/usr/include/c++/4.4/cmath" 3
inline long double ceil(long double __x)
# 183 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ceill(__x); }
# 185 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    ceil ( _Tp __x )
    { return __builtin_ceil ( __x ); }
# 191 "/usr/include/c++/4.4/cmath" 3
using ::cos;
# 194 "/usr/include/c++/4.4/cmath" 3
inline float cos(float __x)
# 195 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_cosf(__x); }
# 198 "/usr/include/c++/4.4/cmath" 3
inline long double cos(long double __x)
# 199 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_cosl(__x); }
# 201 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    cos ( _Tp __x )
    { return __builtin_cos ( __x ); }
# 207 "/usr/include/c++/4.4/cmath" 3
using ::cosh;
# 210 "/usr/include/c++/4.4/cmath" 3
inline float cosh(float __x)
# 211 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_coshf(__x); }
# 214 "/usr/include/c++/4.4/cmath" 3
inline long double cosh(long double __x)
# 215 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_coshl(__x); }
# 217 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    cosh ( _Tp __x )
    { return __builtin_cosh ( __x ); }
# 223 "/usr/include/c++/4.4/cmath" 3
using ::exp;
# 226 "/usr/include/c++/4.4/cmath" 3
inline float exp(float __x)
# 227 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_expf(__x); }
# 230 "/usr/include/c++/4.4/cmath" 3
inline long double exp(long double __x)
# 231 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_expl(__x); }
# 233 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    exp ( _Tp __x )
    { return __builtin_exp ( __x ); }
# 239 "/usr/include/c++/4.4/cmath" 3
using ::fabs;
# 242 "/usr/include/c++/4.4/cmath" 3
inline float fabs(float __x)
# 243 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsf(__x); }
# 246 "/usr/include/c++/4.4/cmath" 3
inline long double fabs(long double __x)
# 247 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsl(__x); }
# 249 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    fabs ( _Tp __x )
    { return __builtin_fabs ( __x ); }
# 255 "/usr/include/c++/4.4/cmath" 3
using ::floor;
# 258 "/usr/include/c++/4.4/cmath" 3
inline float floor(float __x)
# 259 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_floorf(__x); }
# 262 "/usr/include/c++/4.4/cmath" 3
inline long double floor(long double __x)
# 263 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_floorl(__x); }
# 265 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    floor ( _Tp __x )
    { return __builtin_floor ( __x ); }
# 271 "/usr/include/c++/4.4/cmath" 3
using ::fmod;
# 274 "/usr/include/c++/4.4/cmath" 3
inline float fmod(float __x, float __y)
# 275 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fmodf(__x, __y); }
# 278 "/usr/include/c++/4.4/cmath" 3
inline long double fmod(long double __x, long double __y)
# 279 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fmodl(__x, __y); }
# 281 "/usr/include/c++/4.4/cmath" 3
using ::frexp;
# 284 "/usr/include/c++/4.4/cmath" 3
inline float frexp(float __x, int *__exp)
# 285 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_frexpf(__x, __exp); }
# 288 "/usr/include/c++/4.4/cmath" 3
inline long double frexp(long double __x, int *__exp)
# 289 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_frexpl(__x, __exp); }
# 291 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    frexp ( _Tp __x, int * __exp )
    { return __builtin_frexp ( __x, __exp ); }
# 297 "/usr/include/c++/4.4/cmath" 3
using ::ldexp;
# 300 "/usr/include/c++/4.4/cmath" 3
inline float ldexp(float __x, int __exp)
# 301 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ldexpf(__x, __exp); }
# 304 "/usr/include/c++/4.4/cmath" 3
inline long double ldexp(long double __x, int __exp)
# 305 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ldexpl(__x, __exp); }
# 307 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
  ldexp ( _Tp __x, int __exp )
  { return __builtin_ldexp ( __x, __exp ); }
# 313 "/usr/include/c++/4.4/cmath" 3
using ::log;
# 316 "/usr/include/c++/4.4/cmath" 3
inline float log(float __x)
# 317 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_logf(__x); }
# 320 "/usr/include/c++/4.4/cmath" 3
inline long double log(long double __x)
# 321 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_logl(__x); }
# 323 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    log ( _Tp __x )
    { return __builtin_log ( __x ); }
# 329 "/usr/include/c++/4.4/cmath" 3
using ::log10;
# 332 "/usr/include/c++/4.4/cmath" 3
inline float log10(float __x)
# 333 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_log10f(__x); }
# 336 "/usr/include/c++/4.4/cmath" 3
inline long double log10(long double __x)
# 337 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_log10l(__x); }
# 339 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    log10 ( _Tp __x )
    { return __builtin_log10 ( __x ); }
# 345 "/usr/include/c++/4.4/cmath" 3
using ::modf;
# 348 "/usr/include/c++/4.4/cmath" 3
inline float modf(float __x, float *__iptr)
# 349 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_modff(__x, __iptr); }
# 352 "/usr/include/c++/4.4/cmath" 3
inline long double modf(long double __x, long double *__iptr)
# 353 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_modfl(__x, __iptr); }
# 355 "/usr/include/c++/4.4/cmath" 3
using ::pow;
# 358 "/usr/include/c++/4.4/cmath" 3
inline float pow(float __x, float __y)
# 359 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powf(__x, __y); }
# 362 "/usr/include/c++/4.4/cmath" 3
inline long double pow(long double __x, long double __y)
# 363 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powl(__x, __y); }
# 369 "/usr/include/c++/4.4/cmath" 3
inline double pow(double __x, int __i)
# 370 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powi(__x, __i); }
# 373 "/usr/include/c++/4.4/cmath" 3
inline float pow(float __x, int __n)
# 374 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powif(__x, __n); }
# 377 "/usr/include/c++/4.4/cmath" 3
inline long double pow(long double __x, int __n)
# 378 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powil(__x, __n); }
# 381 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp, typename _Up >
    inline
    typename __gnu_cxx :: __promote_2 <
    typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value
        && __is_arithmetic < _Up > :: __value,
        _Tp > :: __type, _Up > :: __type
    pow ( _Tp __x, _Up __y )
    {
      typedef typename __gnu_cxx :: __promote_2 < _Tp, _Up > :: __type __type;
      return pow ( __type ( __x ), __type ( __y ) );
    }
# 393 "/usr/include/c++/4.4/cmath" 3
using ::sin;
# 396 "/usr/include/c++/4.4/cmath" 3
inline float sin(float __x)
# 397 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinf(__x); }
# 400 "/usr/include/c++/4.4/cmath" 3
inline long double sin(long double __x)
# 401 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinl(__x); }
# 403 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sin ( _Tp __x )
    { return __builtin_sin ( __x ); }
# 409 "/usr/include/c++/4.4/cmath" 3
using ::sinh;
# 412 "/usr/include/c++/4.4/cmath" 3
inline float sinh(float __x)
# 413 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinhf(__x); }
# 416 "/usr/include/c++/4.4/cmath" 3
inline long double sinh(long double __x)
# 417 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinhl(__x); }
# 419 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sinh ( _Tp __x )
    { return __builtin_sinh ( __x ); }
# 425 "/usr/include/c++/4.4/cmath" 3
using ::sqrt;
# 428 "/usr/include/c++/4.4/cmath" 3
inline float sqrt(float __x)
# 429 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sqrtf(__x); }
# 432 "/usr/include/c++/4.4/cmath" 3
inline long double sqrt(long double __x)
# 433 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sqrtl(__x); }
# 435 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sqrt ( _Tp __x )
    { return __builtin_sqrt ( __x ); }
# 441 "/usr/include/c++/4.4/cmath" 3
using ::tan;
# 444 "/usr/include/c++/4.4/cmath" 3
inline float tan(float __x)
# 445 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanf(__x); }
# 448 "/usr/include/c++/4.4/cmath" 3
inline long double tan(long double __x)
# 449 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanl(__x); }
# 451 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    tan ( _Tp __x )
    { return __builtin_tan ( __x ); }
# 457 "/usr/include/c++/4.4/cmath" 3
using ::tanh;
# 460 "/usr/include/c++/4.4/cmath" 3
inline float tanh(float __x)
# 461 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanhf(__x); }
# 464 "/usr/include/c++/4.4/cmath" 3
inline long double tanh(long double __x)
# 465 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanhl(__x); }
# 467 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    tanh ( _Tp __x )
    { return __builtin_tanh ( __x ); }
# 473 "/usr/include/c++/4.4/cmath" 3
}
# 492 "/usr/include/c++/4.4/cmath" 3
namespace std __attribute__((visibility("default"))) {
# 494 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    fpclassify ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_fpclassify ( FP_NAN, FP_INFINITE, FP_NORMAL,
      FP_SUBNORMAL, FP_ZERO, __type ( __f ) );
    }
# 504 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isfinite ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isfinite ( __type ( __f ) );
    }
# 513 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isinf ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isinf ( __type ( __f ) );
    }
# 522 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isnan ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isnan ( __type ( __f ) );
    }
# 531 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isnormal ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isnormal ( __type ( __f ) );
    }
# 540 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    signbit ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_signbit ( __type ( __f ) );
    }
# 549 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isgreater ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isgreater ( __type ( __f1 ), __type ( __f2 ) );
    }
# 558 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isgreaterequal ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isgreaterequal ( __type ( __f1 ), __type ( __f2 ) );
    }
# 567 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isless ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isless ( __type ( __f1 ), __type ( __f2 ) );
    }
# 576 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    islessequal ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_islessequal ( __type ( __f1 ), __type ( __f2 ) );
    }
# 585 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    islessgreater ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_islessgreater ( __type ( __f1 ), __type ( __f2 ) );
    }
# 594 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isunordered ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isunordered ( __type ( __f1 ), __type ( __f2 ) );
    }
# 603 "/usr/include/c++/4.4/cmath" 3
}
# 35 "/usr/include/c++/4.4/bits/cmath.tcc" 3
namespace std __attribute__((visibility("default"))) {
# 37 "/usr/include/c++/4.4/bits/cmath.tcc" 3
template < typename _Tp >
    inline _Tp
    __cmath_power ( _Tp __x, unsigned int __n )
    {
      _Tp __y = __n % 2 ? __x : _Tp ( 1 );

      while ( __n >>= 1 )
        {
          __x = __x * __x;
          if ( __n % 2 )
            __y = __y * __x;
        }

      return __y;
    }
# 53 "/usr/include/c++/4.4/bits/cmath.tcc" 3
}
# 49 "/usr/include/c++/4.4/cstddef" 3
namespace std __attribute__((visibility("default"))) {
# 51 "/usr/include/c++/4.4/cstddef" 3
using ::ptrdiff_t;
# 52 "/usr/include/c++/4.4/cstddef" 3
using ::size_t;
# 54 "/usr/include/c++/4.4/cstddef" 3
}
# 100 "/usr/include/c++/4.4/cstdlib" 3
namespace std __attribute__((visibility("default"))) {
# 102 "/usr/include/c++/4.4/cstdlib" 3
using ::div_t;
# 103 "/usr/include/c++/4.4/cstdlib" 3
using ::ldiv_t;
# 105 "/usr/include/c++/4.4/cstdlib" 3
using ::abort;
# 106 "/usr/include/c++/4.4/cstdlib" 3
using ::abs;
# 107 "/usr/include/c++/4.4/cstdlib" 3
using ::atexit;
# 108 "/usr/include/c++/4.4/cstdlib" 3
using ::atof;
# 109 "/usr/include/c++/4.4/cstdlib" 3
using ::atoi;
# 110 "/usr/include/c++/4.4/cstdlib" 3
using ::atol;
# 111 "/usr/include/c++/4.4/cstdlib" 3
using ::bsearch;
# 112 "/usr/include/c++/4.4/cstdlib" 3
using ::calloc;
# 113 "/usr/include/c++/4.4/cstdlib" 3
using ::div;
# 114 "/usr/include/c++/4.4/cstdlib" 3
using ::exit;
# 115 "/usr/include/c++/4.4/cstdlib" 3
using ::free;
# 116 "/usr/include/c++/4.4/cstdlib" 3
using ::getenv;
# 117 "/usr/include/c++/4.4/cstdlib" 3
using ::labs;
# 118 "/usr/include/c++/4.4/cstdlib" 3
using ::ldiv;
# 119 "/usr/include/c++/4.4/cstdlib" 3
using ::malloc;
# 121 "/usr/include/c++/4.4/cstdlib" 3
using ::mblen;
# 122 "/usr/include/c++/4.4/cstdlib" 3
using ::mbstowcs;
# 123 "/usr/include/c++/4.4/cstdlib" 3
using ::mbtowc;
# 125 "/usr/include/c++/4.4/cstdlib" 3
using ::qsort;
# 126 "/usr/include/c++/4.4/cstdlib" 3
using ::rand;
# 127 "/usr/include/c++/4.4/cstdlib" 3
using ::realloc;
# 128 "/usr/include/c++/4.4/cstdlib" 3
using ::srand;
# 129 "/usr/include/c++/4.4/cstdlib" 3
using ::strtod;
# 130 "/usr/include/c++/4.4/cstdlib" 3
using ::strtol;
# 131 "/usr/include/c++/4.4/cstdlib" 3
using ::strtoul;
# 132 "/usr/include/c++/4.4/cstdlib" 3
using ::system;
# 134 "/usr/include/c++/4.4/cstdlib" 3
using ::wcstombs;
# 135 "/usr/include/c++/4.4/cstdlib" 3
using ::wctomb;
# 139 "/usr/include/c++/4.4/cstdlib" 3
inline long abs(long __i) { return labs(__i); }
# 142 "/usr/include/c++/4.4/cstdlib" 3
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); }
# 144 "/usr/include/c++/4.4/cstdlib" 3
}
# 157 "/usr/include/c++/4.4/cstdlib" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 160 "/usr/include/c++/4.4/cstdlib" 3
using ::lldiv_t;
# 166 "/usr/include/c++/4.4/cstdlib" 3
using ::_Exit;
# 170 "/usr/include/c++/4.4/cstdlib" 3
inline long long abs(long long __x) { return (__x >= (0)) ? __x : (-__x); }
# 173 "/usr/include/c++/4.4/cstdlib" 3
using ::llabs;
# 176 "/usr/include/c++/4.4/cstdlib" 3
inline lldiv_t div(long long __n, long long __d)
# 177 "/usr/include/c++/4.4/cstdlib" 3
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; }
# 179 "/usr/include/c++/4.4/cstdlib" 3
using ::lldiv;
# 190 "/usr/include/c++/4.4/cstdlib" 3
using ::atoll;
# 191 "/usr/include/c++/4.4/cstdlib" 3
using ::strtoll;
# 192 "/usr/include/c++/4.4/cstdlib" 3
using ::strtoull;
# 194 "/usr/include/c++/4.4/cstdlib" 3
using ::strtof;
# 195 "/usr/include/c++/4.4/cstdlib" 3
using ::strtold;
# 197 "/usr/include/c++/4.4/cstdlib" 3
}
# 199 "/usr/include/c++/4.4/cstdlib" 3
namespace std __attribute__((visibility("default"))) {
# 202 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::lldiv_t;
# 204 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::_Exit;
# 205 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::abs;
# 207 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::llabs;
# 208 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::div;
# 209 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::lldiv;
# 211 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::atoll;
# 212 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtof;
# 213 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtoll;
# 214 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtoull;
# 215 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtold;
# 217 "/usr/include/c++/4.4/cstdlib" 3
}
# 497 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
namespace __gnu_cxx {
# 499 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline long long abs(long long) __attribute__((visibility("default")));
# 500 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 502 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
namespace std {
# 504 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
template< class T> extern inline T __pow_helper(T, int);
# 505 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
template< class T> extern inline T __cmath_power(T, unsigned);
# 506 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 508 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::abs;
# 509 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::fabs;
# 510 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::ceil;
# 511 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::floor;
# 512 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::sqrt;
# 513 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::pow;
# 514 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::log;
# 515 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::log10;
# 516 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::fmod;
# 517 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::modf;
# 518 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::exp;
# 519 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::frexp;
# 520 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::ldexp;
# 521 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::asin;
# 522 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::sin;
# 523 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::sinh;
# 524 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::acos;
# 525 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::cos;
# 526 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::cosh;
# 527 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::atan;
# 528 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::atan2;
# 529 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::tan;
# 530 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
using std::tanh;
# 584 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
namespace std {
# 587 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline long abs(long) __attribute__((visibility("default")));
# 588 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float abs(float) __attribute__((visibility("default")));
# 589 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline double abs(double) __attribute__((visibility("default")));
# 590 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float fabs(float) __attribute__((visibility("default")));
# 591 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float ceil(float) __attribute__((visibility("default")));
# 592 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float floor(float) __attribute__((visibility("default")));
# 593 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float sqrt(float) __attribute__((visibility("default")));
# 594 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float pow(float, float) __attribute__((visibility("default")));
# 595 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float pow(float, int) __attribute__((visibility("default")));
# 596 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline double pow(double, int) __attribute__((visibility("default")));
# 597 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float log(float) __attribute__((visibility("default")));
# 598 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float log10(float) __attribute__((visibility("default")));
# 599 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float fmod(float, float) __attribute__((visibility("default")));
# 600 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float modf(float, float *) __attribute__((visibility("default")));
# 601 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float exp(float) __attribute__((visibility("default")));
# 602 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float frexp(float, int *) __attribute__((visibility("default")));
# 603 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float ldexp(float, int) __attribute__((visibility("default")));
# 604 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float asin(float) __attribute__((visibility("default")));
# 605 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float sin(float) __attribute__((visibility("default")));
# 606 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float sinh(float) __attribute__((visibility("default")));
# 607 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float acos(float) __attribute__((visibility("default")));
# 608 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float cos(float) __attribute__((visibility("default")));
# 609 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float cosh(float) __attribute__((visibility("default")));
# 610 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float atan(float) __attribute__((visibility("default")));
# 611 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float atan2(float, float) __attribute__((visibility("default")));
# 612 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float tan(float) __attribute__((visibility("default")));
# 613 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
extern inline float tanh(float) __attribute__((visibility("default")));
# 616 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 619 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float logb(float a)
# 620 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 621 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return logbf(a);
# 622 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 624 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline int ilogb(float a)
# 625 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 626 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ilogbf(a);
# 627 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 629 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float scalbn(float a, int b)
# 630 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 631 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return scalbnf(a, b);
# 632 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 634 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float scalbln(float a, long b)
# 635 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 636 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return scalblnf(a, b);
# 637 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 639 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float exp2(float a)
# 640 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 641 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return exp2f(a);
# 642 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 644 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float exp10(float a)
# 645 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 646 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return exp10f(a);
# 647 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 649 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float expm1(float a)
# 650 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 651 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return expm1f(a);
# 652 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 654 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float log2(float a)
# 655 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 656 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return log2f(a);
# 657 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 659 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float log1p(float a)
# 660 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 661 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return log1pf(a);
# 662 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 664 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float rsqrt(float a)
# 665 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 666 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return rsqrtf(a);
# 667 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 669 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float acosh(float a)
# 670 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 671 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return acoshf(a);
# 672 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 674 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float asinh(float a)
# 675 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 676 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return asinhf(a);
# 677 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 679 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float atanh(float a)
# 680 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 681 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return atanhf(a);
# 682 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 684 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float hypot(float a, float b)
# 685 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 686 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return hypotf(a, b);
# 687 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 689 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float cbrt(float a)
# 690 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 691 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return cbrtf(a);
# 692 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 694 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float rcbrt(float a)
# 695 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 696 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return rcbrtf(a);
# 697 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 699 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float sinpi(float a)
# 700 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 701 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return sinpif(a);
# 702 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 704 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline void sincos(float a, float *sptr, float *cptr)
# 705 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 706 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
sincosf(a, sptr, cptr);
# 707 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 709 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float erf(float a)
# 710 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 711 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return erff(a);
# 712 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 714 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float erfinv(float a)
# 715 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 716 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return erfinvf(a);
# 717 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 719 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float erfc(float a)
# 720 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 721 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return erfcf(a);
# 722 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 724 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float erfcinv(float a)
# 725 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 726 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return erfcinvf(a);
# 727 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 729 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float lgamma(float a)
# 730 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 731 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return lgammaf(a);
# 732 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 734 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float tgamma(float a)
# 735 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 736 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return tgammaf(a);
# 737 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 739 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float copysign(float a, float b)
# 740 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 741 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return copysignf(a, b);
# 742 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 744 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double copysign(double a, float b)
# 745 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 746 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return copysign(a, (double)b);
# 747 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 749 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float copysign(float a, double b)
# 750 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 751 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return copysignf(a, (float)b);
# 752 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 754 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float nextafter(float a, float b)
# 755 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 756 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return nextafterf(a, b);
# 757 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 759 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float remainder(float a, float b)
# 760 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 761 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return remainderf(a, b);
# 762 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 764 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float remquo(float a, float b, int *quo)
# 765 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 766 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return remquof(a, b, quo);
# 767 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 769 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float round(float a)
# 770 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 771 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return roundf(a);
# 772 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 774 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline long lround(float a)
# 775 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 776 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return lroundf(a);
# 777 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 779 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline long long llround(float a)
# 780 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 781 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return llroundf(a);
# 782 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 784 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float trunc(float a)
# 785 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 786 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return truncf(a);
# 787 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 789 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float rint(float a)
# 790 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 791 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return rintf(a);
# 792 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 794 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline long lrint(float a)
# 795 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 796 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return lrintf(a);
# 797 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 799 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline long long llrint(float a)
# 800 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 801 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return llrintf(a);
# 802 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 804 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float nearbyint(float a)
# 805 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 806 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return nearbyintf(a);
# 807 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 809 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float fdim(float a, float b)
# 810 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 811 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fdimf(a, b);
# 812 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 814 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float fma(float a, float b, float c)
# 815 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 816 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmaf(a, b, c);
# 817 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 819 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float fmax(float a, float b)
# 820 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 821 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmaxf(a, b);
# 822 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 824 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float fmin(float a, float b)
# 825 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 826 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fminf(a, b);
# 827 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 829 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned min(unsigned a, unsigned b)
# 830 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 831 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return umin(a, b);
# 832 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 834 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned min(int a, unsigned b)
# 835 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 836 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return umin((unsigned)a, b);
# 837 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 839 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned min(unsigned a, int b)
# 840 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 841 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return umin(a, (unsigned)b);
# 842 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 844 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline long long min(long long a, long long b)
# 845 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 846 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return llmin(a, b);
# 847 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 849 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned long long min(unsigned long long a, unsigned long long b)
# 850 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 851 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ullmin(a, b);
# 852 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 854 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned long long min(long long a, unsigned long long b)
# 855 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 856 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ullmin((unsigned long long)a, b);
# 857 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 859 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned long long min(unsigned long long a, long long b)
# 860 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 861 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ullmin(a, (unsigned long long)b);
# 862 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 864 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float min(float a, float b)
# 865 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 866 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fminf(a, b);
# 867 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 869 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double min(double a, double b)
# 870 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 871 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmin(a, b);
# 872 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 874 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double min(float a, double b)
# 875 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 876 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmin((double)a, b);
# 877 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 879 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double min(double a, float b)
# 880 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 881 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmin(a, (double)b);
# 882 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 884 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned max(unsigned a, unsigned b)
# 885 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 886 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return umax(a, b);
# 887 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 889 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned max(int a, unsigned b)
# 890 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 891 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return umax((unsigned)a, b);
# 892 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 894 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned max(unsigned a, int b)
# 895 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 896 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return umax(a, (unsigned)b);
# 897 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 899 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline long long max(long long a, long long b)
# 900 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 901 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return llmax(a, b);
# 902 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 904 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned long long max(unsigned long long a, unsigned long long b)
# 905 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 906 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ullmax(a, b);
# 907 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 909 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned long long max(long long a, unsigned long long b)
# 910 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 911 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ullmax((unsigned long long)a, b);
# 912 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 914 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline unsigned long long max(unsigned long long a, long long b)
# 915 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 916 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return ullmax(a, (unsigned long long)b);
# 917 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 919 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline float max(float a, float b)
# 920 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 921 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmaxf(a, b);
# 922 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 924 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double max(double a, double b)
# 925 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 926 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmax(a, b);
# 927 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 929 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double max(float a, double b)
# 930 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 931 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmax((double)a, b);
# 932 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 934 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
static inline double max(double a, float b)
# 935 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
{
# 936 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
return fmax(a, (double)b);
# 937 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h"
}
# 60 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
template< class T, int dim = 1>
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
struct surface : public surfaceReference {
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
surface()
# 64 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
{
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
(channelDesc) = cudaCreateChannelDesc< T> ();
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
}
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
surface(cudaChannelFormatDesc desc)
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
{
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
(channelDesc) = desc;
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
}
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
};
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
template< int dim>
# 76 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
struct surface< void, dim> : public surfaceReference {
# 78 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
surface()
# 79 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
{
# 80 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
(channelDesc) = cudaCreateChannelDesc< void> ();
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
}
# 82 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_surface_types.h"
};
# 60 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
template< class T, int dim = 1, cudaTextureReadMode mode = cudaReadModeElementType>
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
struct texture : public textureReference {
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
texture(int norm = 0, cudaTextureFilterMode
# 64 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
fMode = cudaFilterModePoint, cudaTextureAddressMode
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
aMode = cudaAddressModeClamp)
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
{
# 67 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
(normalized) = norm;
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
(filterMode) = fMode;
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
((addressMode)[0]) = aMode;
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
((addressMode)[1]) = aMode;
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
((addressMode)[2]) = aMode;
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
(channelDesc) = cudaCreateChannelDesc< T> ();
# 73 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
}
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
texture(int norm, cudaTextureFilterMode
# 76 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
fMode, cudaTextureAddressMode
# 77 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
aMode, cudaChannelFormatDesc
# 78 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
desc)
# 79 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
{
# 80 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
(normalized) = norm;
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
(filterMode) = fMode;
# 82 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
((addressMode)[0]) = aMode;
# 83 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
((addressMode)[1]) = aMode;
# 84 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
((addressMode)[2]) = aMode;
# 85 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
(channelDesc) = desc;
# 86 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
}
# 87 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_texture_types.h"
};
# 324 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline int mulhi(int a, int b)
# 325 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 327 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 329 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b)
# 330 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 332 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 334 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b)
# 335 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 337 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 339 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b)
# 340 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 342 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 344 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline long long mul64hi(long long a, long long b)
# 345 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 347 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 349 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b)
# 350 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 352 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 354 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b)
# 355 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 357 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 359 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b)
# 360 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 362 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 364 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline int float_as_int(float a)
# 365 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 367 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 369 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline float int_as_float(int a)
# 370 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 372 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 374 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline float saturate(float a)
# 375 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 377 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 379 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline int mul24(int a, int b)
# 380 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 382 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 384 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b)
# 385 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 387 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 389 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline void trap()
# 390 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 392 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 394 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline void brkpt(int c)
# 395 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 397 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 399 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline void syncthreads()
# 400 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 402 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 404 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline void prof_trigger(int e)
# 405 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 422 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 424 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline void threadfence(bool global = true)
# 425 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 427 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 429 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode = cudaRoundZero)
# 430 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 435 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 437 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero)
# 438 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 443 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 445 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest)
# 446 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 451 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 453 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest)
# 454 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 459 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_functions.h"
exit(___);}
# 102 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val)
# 103 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val)
# 108 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val)
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 115 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val)
# 118 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 122 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val)
# 123 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val)
# 128 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val)
# 133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 137 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val)
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 142 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val)
# 143 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val)
# 148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val)
# 153 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 157 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val)
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val)
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val)
# 168 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 172 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val)
# 173 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 175 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val)
# 178 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 182 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val)
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 185 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val)
# 188 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 192 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val)
# 193 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 197 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val)
# 198 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val)
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 205 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 75 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val)
# 76 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 78 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 80 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val)
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 83 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 85 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val)
# 86 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 88 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 90 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline bool any(bool cond)
# 91 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 93 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 95 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline bool all(bool cond)
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode)
# 171 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 176 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 178 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest)
# 179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 184 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 186 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest)
# 187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 192 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 194 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero)
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero)
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 210 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero)
# 211 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 216 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero)
# 219 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 224 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 226 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest)
# 227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 232 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 234 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest)
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest)
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 245 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest)
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 252 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest)
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_13_double_functions.h"
exit(___);}
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val)
# 67 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_atomic_functions.h"
{int volatile ___ = 1;
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_atomic_functions.h"
exit(___);}
# 124 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned ballot(bool pred)
# 125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 129 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred)
# 130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 134 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred)
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 137 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 139 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred)
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 142 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 97 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf1Dread(T *res, surface< void, 1> surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 99 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 108 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline T
# 109 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 116 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 118 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf1Dread(T *res, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 122 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 128 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline signed char surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 134 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 137 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned char surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 143 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 144 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 146 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 149 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 156 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 164 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 166 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 169 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 174 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 178 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 184 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 186 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 189 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned short surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 192 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 196 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 198 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 201 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 204 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 212 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 216 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 221 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 222 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 226 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 232 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 236 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 238 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 241 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 244 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 254 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 256 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 259 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 264 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 267 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 268 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 270 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 274 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 281 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 284 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 287 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline long long surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 288 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 290 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 293 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned long long surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 294 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 296 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 299 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 300 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 302 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 305 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 306 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 308 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 311 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 312 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 316 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 319 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 320 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 322 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 385 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 386 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 388 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 391 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 392 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 394 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 397 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 398 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 402 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 405 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 406 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 410 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 457 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 458 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf2Dread(T *res, surface< void, 2> surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 459 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 466 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 468 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline T
# 469 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 470 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 476 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 478 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 479 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf2Dread(T *res, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 480 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 482 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 485 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 486 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 488 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 491 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline signed char surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 492 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 494 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 497 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned char surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 498 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 500 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 503 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 504 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 506 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 509 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 510 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 512 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 515 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 516 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 520 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 523 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 524 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 526 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 529 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 530 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 534 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 537 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 538 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 540 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 543 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 544 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 546 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 549 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned short surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 550 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 552 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 555 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 556 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 558 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 561 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 562 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 564 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 567 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 568 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 572 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 575 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 576 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 578 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 581 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 582 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 586 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 589 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 590 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 592 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 595 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 596 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 598 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 601 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 602 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 604 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 607 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 608 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 610 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 613 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 614 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 616 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 619 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 620 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 624 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 627 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 628 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 630 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 633 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 634 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 638 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 641 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 642 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 644 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 647 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline long long surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 648 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 650 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 653 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned long long surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 654 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 656 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 659 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 660 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 662 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 665 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 666 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 668 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 671 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 672 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 676 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 679 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 680 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 682 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 745 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 746 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 748 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 751 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 752 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 754 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 757 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 758 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 762 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 765 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 766 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 770 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 817 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 818 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf1Dwrite(T val, surface< void, 1> surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 819 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 837 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 839 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 840 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf1Dwrite(T val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 841 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 843 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 846 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 847 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 849 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 851 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(signed char val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 852 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 854 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 856 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned char val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 857 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 859 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 861 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 862 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 864 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 866 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uchar1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 867 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 869 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 871 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 872 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 874 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 876 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uchar2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 877 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 879 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 881 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 882 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 884 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 886 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uchar4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 887 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 889 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 891 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 892 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 894 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 896 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned short val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 897 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 899 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 901 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 902 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 904 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 906 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ushort1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 907 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 909 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 911 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 912 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 914 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 916 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ushort2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 917 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 919 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 921 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 922 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 924 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 926 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ushort4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 927 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 929 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 931 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 932 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 934 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 936 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 937 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 939 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 941 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 942 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 944 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 946 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uint1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 947 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 949 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 951 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 952 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 954 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 956 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uint2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 957 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 959 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 961 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 962 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 964 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 966 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uint4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 967 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 969 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 971 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(long long val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 972 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 974 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 976 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned long long val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 977 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 979 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 981 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(longlong1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 982 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 984 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 986 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ulonglong1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 987 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 989 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 991 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(longlong2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 992 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 994 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 996 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ulonglong2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 997 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 999 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1045 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1046 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1048 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1050 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1051 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1053 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1055 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1056 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1058 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1060 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1061 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1063 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 1111 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf2Dwrite(T val, surface< void, 2> surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 1133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
surf2Dwrite(T val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1134 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1136 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1139 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1142 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1144 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(signed char val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1149 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned char val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1154 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1157 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1159 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uchar1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1164 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1169 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uchar2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1172 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1174 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1175 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uchar4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1182 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1184 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1185 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1189 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned short val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1192 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1194 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1197 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1199 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ushort1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1204 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1205 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1209 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ushort2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1210 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1212 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1214 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1217 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1219 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ushort4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1220 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1222 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1224 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1225 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1232 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1234 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1237 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1239 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uint1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1244 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1245 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1249 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uint2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1252 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1254 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1257 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1259 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uint4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1262 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1264 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(long long val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1265 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1267 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1269 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned long long val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1270 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1272 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1274 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(longlong1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1275 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1277 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1279 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ulonglong1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1280 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1284 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(longlong2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1285 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1287 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1289 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ulonglong2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1290 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1292 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1338 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1339 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1341 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1343 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1344 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1346 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1348 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1349 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1351 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 1353 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1354 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1356 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/surface_functions.h"
exit(___);}
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< class T, cudaTextureReadMode readMode> __attribute__((unused)) extern uint4 __utexfetchi(texture< T, 1, readMode> , int4);
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< class T, cudaTextureReadMode readMode> __attribute__((unused)) extern int4 __itexfetchi(texture< T, 1, readMode> , int4);
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< class T, cudaTextureReadMode readMode> __attribute__((unused)) extern float4 __ftexfetchi(texture< T, 1, readMode> , int4);
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< class T, int dim, cudaTextureReadMode readMode> __attribute__((unused)) extern uint4 __utexfetch(texture< T, dim, readMode> , float4, int = dim);
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< class T, int dim, cudaTextureReadMode readMode> __attribute__((unused)) extern int4 __itexfetch(texture< T, dim, readMode> , float4, int = dim);
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< class T, int dim, cudaTextureReadMode readMode> __attribute__((unused)) extern float4 __ftexfetch(texture< T, dim, readMode> , float4, int = dim);
# 80 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex1Dfetch(texture< char, 1, cudaReadModeElementType> t, int x)
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 89 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 91 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex1Dfetch(texture< signed char, 1, cudaReadModeElementType> t, int x)
# 92 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex1Dfetch(texture< unsigned char, 1, cudaReadModeElementType> t, int x)
# 99 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 103 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex1Dfetch(texture< char1, 1, cudaReadModeElementType> t, int x)
# 106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex1Dfetch(texture< uchar1, 1, cudaReadModeElementType> t, int x)
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex1Dfetch(texture< char2, 1, cudaReadModeElementType> t, int x)
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 124 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex1Dfetch(texture< uchar2, 1, cudaReadModeElementType> t, int x)
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex1Dfetch(texture< char4, 1, cudaReadModeElementType> t, int x)
# 134 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex1Dfetch(texture< uchar4, 1, cudaReadModeElementType> t, int x)
# 141 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 153 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex1Dfetch(texture< short, 1, cudaReadModeElementType> t, int x)
# 154 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex1Dfetch(texture< unsigned short, 1, cudaReadModeElementType> t, int x)
# 161 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex1Dfetch(texture< short1, 1, cudaReadModeElementType> t, int x)
# 168 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 172 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 174 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex1Dfetch(texture< ushort1, 1, cudaReadModeElementType> t, int x)
# 175 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 181 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex1Dfetch(texture< short2, 1, cudaReadModeElementType> t, int x)
# 182 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 186 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 188 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex1Dfetch(texture< ushort2, 1, cudaReadModeElementType> t, int x)
# 189 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 193 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 195 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex1Dfetch(texture< short4, 1, cudaReadModeElementType> t, int x)
# 196 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex1Dfetch(texture< ushort4, 1, cudaReadModeElementType> t, int x)
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 207 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex1Dfetch(texture< int, 1, cudaReadModeElementType> t, int x)
# 216 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 220 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 222 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex1Dfetch(texture< unsigned, 1, cudaReadModeElementType> t, int x)
# 223 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex1Dfetch(texture< int1, 1, cudaReadModeElementType> t, int x)
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 234 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 236 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex1Dfetch(texture< uint1, 1, cudaReadModeElementType> t, int x)
# 237 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 241 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex1Dfetch(texture< int2, 1, cudaReadModeElementType> t, int x)
# 244 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex1Dfetch(texture< uint2, 1, cudaReadModeElementType> t, int x)
# 251 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 257 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex1Dfetch(texture< int4, 1, cudaReadModeElementType> t, int x)
# 258 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 262 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 264 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex1Dfetch(texture< uint4, 1, cudaReadModeElementType> t, int x)
# 265 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 269 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 343 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< float, 1, cudaReadModeElementType> t, int x)
# 344 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 348 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 350 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< float1, 1, cudaReadModeElementType> t, int x)
# 351 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 355 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 357 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< float2, 1, cudaReadModeElementType> t, int x)
# 358 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 362 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 364 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< float4, 1, cudaReadModeElementType> t, int x)
# 365 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 369 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 377 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< char, 1, cudaReadModeNormalizedFloat> t, int x)
# 378 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 387 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 389 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< signed char, 1, cudaReadModeNormalizedFloat> t, int x)
# 390 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 395 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 397 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< unsigned char, 1, cudaReadModeNormalizedFloat> t, int x)
# 398 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 403 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 405 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< char1, 1, cudaReadModeNormalizedFloat> t, int x)
# 406 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 411 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 413 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< uchar1, 1, cudaReadModeNormalizedFloat> t, int x)
# 414 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 419 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 421 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< char2, 1, cudaReadModeNormalizedFloat> t, int x)
# 422 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 427 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 429 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< uchar2, 1, cudaReadModeNormalizedFloat> t, int x)
# 430 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 435 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 437 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< char4, 1, cudaReadModeNormalizedFloat> t, int x)
# 438 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 443 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 445 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< uchar4, 1, cudaReadModeNormalizedFloat> t, int x)
# 446 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 451 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 459 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< short, 1, cudaReadModeNormalizedFloat> t, int x)
# 460 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 465 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 467 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< unsigned short, 1, cudaReadModeNormalizedFloat> t, int x)
# 468 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 473 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 475 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< short1, 1, cudaReadModeNormalizedFloat> t, int x)
# 476 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 481 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 483 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< ushort1, 1, cudaReadModeNormalizedFloat> t, int x)
# 484 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 489 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 491 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< short2, 1, cudaReadModeNormalizedFloat> t, int x)
# 492 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 497 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 499 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< ushort2, 1, cudaReadModeNormalizedFloat> t, int x)
# 500 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 505 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 507 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< short4, 1, cudaReadModeNormalizedFloat> t, int x)
# 508 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 513 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 515 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< ushort4, 1, cudaReadModeNormalizedFloat> t, int x)
# 516 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 521 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 529 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex1D(texture< char, 1, cudaReadModeElementType> t, float x)
# 530 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 538 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 540 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex1D(texture< signed char, 1, cudaReadModeElementType> t, float x)
# 541 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 545 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 547 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex1D(texture< unsigned char, 1, cudaReadModeElementType> t, float x)
# 548 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 552 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 554 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex1D(texture< char1, 1, cudaReadModeElementType> t, float x)
# 555 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 559 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 561 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex1D(texture< uchar1, 1, cudaReadModeElementType> t, float x)
# 562 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 566 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 568 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex1D(texture< char2, 1, cudaReadModeElementType> t, float x)
# 569 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 573 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 575 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex1D(texture< uchar2, 1, cudaReadModeElementType> t, float x)
# 576 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 580 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 582 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex1D(texture< char4, 1, cudaReadModeElementType> t, float x)
# 583 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 587 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 589 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex1D(texture< uchar4, 1, cudaReadModeElementType> t, float x)
# 590 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 594 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 602 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex1D(texture< short, 1, cudaReadModeElementType> t, float x)
# 603 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 607 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 609 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex1D(texture< unsigned short, 1, cudaReadModeElementType> t, float x)
# 610 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 614 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 616 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex1D(texture< short1, 1, cudaReadModeElementType> t, float x)
# 617 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 621 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 623 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex1D(texture< ushort1, 1, cudaReadModeElementType> t, float x)
# 624 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 628 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 630 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex1D(texture< short2, 1, cudaReadModeElementType> t, float x)
# 631 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 635 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 637 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex1D(texture< ushort2, 1, cudaReadModeElementType> t, float x)
# 638 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 642 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 644 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex1D(texture< short4, 1, cudaReadModeElementType> t, float x)
# 645 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 649 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 651 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex1D(texture< ushort4, 1, cudaReadModeElementType> t, float x)
# 652 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 656 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 664 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex1D(texture< int, 1, cudaReadModeElementType> t, float x)
# 665 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 669 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 671 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex1D(texture< unsigned, 1, cudaReadModeElementType> t, float x)
# 672 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 676 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 678 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex1D(texture< int1, 1, cudaReadModeElementType> t, float x)
# 679 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 683 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 685 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex1D(texture< uint1, 1, cudaReadModeElementType> t, float x)
# 686 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 690 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 692 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex1D(texture< int2, 1, cudaReadModeElementType> t, float x)
# 693 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 697 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 699 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex1D(texture< uint2, 1, cudaReadModeElementType> t, float x)
# 700 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 704 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 706 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex1D(texture< int4, 1, cudaReadModeElementType> t, float x)
# 707 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 711 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 713 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex1D(texture< uint4, 1, cudaReadModeElementType> t, float x)
# 714 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 718 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 798 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< float, 1, cudaReadModeElementType> t, float x)
# 799 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 803 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 805 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< float1, 1, cudaReadModeElementType> t, float x)
# 806 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 810 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 812 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< float2, 1, cudaReadModeElementType> t, float x)
# 813 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 817 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 819 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< float4, 1, cudaReadModeElementType> t, float x)
# 820 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 824 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 832 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< char, 1, cudaReadModeNormalizedFloat> t, float x)
# 833 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 842 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 844 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< signed char, 1, cudaReadModeNormalizedFloat> t, float x)
# 845 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 850 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 852 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< unsigned char, 1, cudaReadModeNormalizedFloat> t, float x)
# 853 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 858 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 860 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< char1, 1, cudaReadModeNormalizedFloat> t, float x)
# 861 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 866 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 868 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< uchar1, 1, cudaReadModeNormalizedFloat> t, float x)
# 869 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 874 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 876 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< char2, 1, cudaReadModeNormalizedFloat> t, float x)
# 877 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 882 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 884 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< uchar2, 1, cudaReadModeNormalizedFloat> t, float x)
# 885 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 890 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 892 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< char4, 1, cudaReadModeNormalizedFloat> t, float x)
# 893 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 898 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 900 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< uchar4, 1, cudaReadModeNormalizedFloat> t, float x)
# 901 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 906 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 914 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< short, 1, cudaReadModeNormalizedFloat> t, float x)
# 915 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 920 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 922 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< unsigned short, 1, cudaReadModeNormalizedFloat> t, float x)
# 923 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 928 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 930 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< short1, 1, cudaReadModeNormalizedFloat> t, float x)
# 931 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 936 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 938 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< ushort1, 1, cudaReadModeNormalizedFloat> t, float x)
# 939 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 944 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 946 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< short2, 1, cudaReadModeNormalizedFloat> t, float x)
# 947 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 952 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 954 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< ushort2, 1, cudaReadModeNormalizedFloat> t, float x)
# 955 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 960 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 962 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< short4, 1, cudaReadModeNormalizedFloat> t, float x)
# 963 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 968 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 970 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< ushort4, 1, cudaReadModeNormalizedFloat> t, float x)
# 971 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 976 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 984 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex2D(texture< char, 2, cudaReadModeElementType> t, float x, float y)
# 985 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 993 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 995 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex2D(texture< signed char, 2, cudaReadModeElementType> t, float x, float y)
# 996 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1000 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1002 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex2D(texture< unsigned char, 2, cudaReadModeElementType> t, float x, float y)
# 1003 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1007 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1009 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex2D(texture< char1, 2, cudaReadModeElementType> t, float x, float y)
# 1010 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1014 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1016 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex2D(texture< uchar1, 2, cudaReadModeElementType> t, float x, float y)
# 1017 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1021 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1023 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex2D(texture< char2, 2, cudaReadModeElementType> t, float x, float y)
# 1024 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1028 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1030 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex2D(texture< uchar2, 2, cudaReadModeElementType> t, float x, float y)
# 1031 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1035 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1037 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2D(texture< char4, 2, cudaReadModeElementType> t, float x, float y)
# 1038 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1042 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1044 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2D(texture< uchar4, 2, cudaReadModeElementType> t, float x, float y)
# 1045 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1049 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1057 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex2D(texture< short, 2, cudaReadModeElementType> t, float x, float y)
# 1058 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1062 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1064 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex2D(texture< unsigned short, 2, cudaReadModeElementType> t, float x, float y)
# 1065 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1069 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1071 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex2D(texture< short1, 2, cudaReadModeElementType> t, float x, float y)
# 1072 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1076 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1078 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex2D(texture< ushort1, 2, cudaReadModeElementType> t, float x, float y)
# 1079 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1083 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1085 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex2D(texture< short2, 2, cudaReadModeElementType> t, float x, float y)
# 1086 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1090 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1092 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex2D(texture< ushort2, 2, cudaReadModeElementType> t, float x, float y)
# 1093 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1097 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1099 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2D(texture< short4, 2, cudaReadModeElementType> t, float x, float y)
# 1100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1104 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2D(texture< ushort4, 2, cudaReadModeElementType> t, float x, float y)
# 1107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1111 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex2D(texture< int, 2, cudaReadModeElementType> t, float x, float y)
# 1120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1124 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex2D(texture< unsigned, 2, cudaReadModeElementType> t, float x, float y)
# 1127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex2D(texture< int1, 2, cudaReadModeElementType> t, float x, float y)
# 1134 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex2D(texture< uint1, 2, cudaReadModeElementType> t, float x, float y)
# 1141 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex2D(texture< int2, 2, cudaReadModeElementType> t, float x, float y)
# 1148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1154 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex2D(texture< uint2, 2, cudaReadModeElementType> t, float x, float y)
# 1155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1159 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1161 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2D(texture< int4, 2, cudaReadModeElementType> t, float x, float y)
# 1162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1166 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1168 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2D(texture< uint4, 2, cudaReadModeElementType> t, float x, float y)
# 1169 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1173 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< float, 2, cudaReadModeElementType> t, float x, float y)
# 1248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1252 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1254 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< float1, 2, cudaReadModeElementType> t, float x, float y)
# 1255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1259 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1261 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< float2, 2, cudaReadModeElementType> t, float x, float y)
# 1262 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1266 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1268 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< float4, 2, cudaReadModeElementType> t, float x, float y)
# 1269 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1281 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< char, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1291 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1293 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< signed char, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1294 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1299 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1301 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< unsigned char, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1302 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1307 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1309 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< char1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1310 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1315 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1317 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< uchar1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1318 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1323 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1325 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< char2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1326 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1331 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1333 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< uchar2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1334 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1339 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1341 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< char4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1342 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1347 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1349 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< uchar4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1350 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1355 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1363 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< short, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1364 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1369 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1371 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< unsigned short, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1372 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1377 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1379 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< short1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1380 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1385 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1387 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< ushort1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1388 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1393 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1395 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< short2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1396 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1401 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1403 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< ushort2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1404 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1409 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1411 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< short4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1412 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1417 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1419 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< ushort4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1420 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1425 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1433 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex3D(texture< char, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1434 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1442 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1444 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex3D(texture< signed char, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1445 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1449 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1451 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex3D(texture< unsigned char, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1452 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1456 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1458 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex3D(texture< char1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1459 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1463 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1465 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex3D(texture< uchar1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1466 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1470 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1472 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex3D(texture< char2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1473 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1477 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1479 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex3D(texture< uchar2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1480 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1484 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1486 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex3D(texture< char4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1487 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1491 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1493 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex3D(texture< uchar4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1494 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1498 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1506 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex3D(texture< short, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1507 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1511 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1513 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex3D(texture< unsigned short, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1514 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1518 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1520 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex3D(texture< short1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1521 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1525 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1527 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex3D(texture< ushort1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1528 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1532 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1534 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex3D(texture< short2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1535 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1539 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1541 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex3D(texture< ushort2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1542 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1546 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1548 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex3D(texture< short4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1549 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1553 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1555 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex3D(texture< ushort4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1556 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1560 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1568 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex3D(texture< int, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1569 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1573 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1575 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex3D(texture< unsigned, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1576 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1580 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1582 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex3D(texture< int1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1583 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1587 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1589 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex3D(texture< uint1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1590 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1594 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1596 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex3D(texture< int2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1597 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1601 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1603 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex3D(texture< uint2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1604 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1608 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1610 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex3D(texture< int4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1611 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1615 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1617 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex3D(texture< uint4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1618 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1622 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1696 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< float, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1697 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1701 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1703 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< float1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1704 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1708 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1710 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< float2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1711 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1715 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1717 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< float4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1718 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1722 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1730 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< char, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1731 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1740 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1742 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< signed char, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1743 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1748 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1750 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< unsigned char, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1751 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1756 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1758 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< char1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1759 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1764 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1766 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< uchar1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1767 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1772 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1774 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< char2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1775 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1780 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1782 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< uchar2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1783 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1788 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1790 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< char4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1791 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1796 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1798 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< uchar4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1799 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1804 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1812 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< short, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1813 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1818 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1820 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< unsigned short, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1821 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1826 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1828 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< short1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1829 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1834 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1836 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< ushort1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1837 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1842 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1844 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< short2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1845 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1850 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1852 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< ushort2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1853 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1858 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1860 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< short4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1861 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1866 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1868 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< ushort4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1869 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1874 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1930 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< int comp, class T> __attribute__((unused)) extern int4 __itex2Dgather(texture< T, 2, cudaReadModeElementType> , float2, int = comp);
# 1932 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< int comp, class T> __attribute__((unused)) extern uint4 __utex2Dgather(texture< T, 2, cudaReadModeElementType> , float2, int = comp);
# 1934 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
template< int comp, class T> __attribute__((unused)) extern float4 __ftex2Dgather(texture< T, 2, cudaReadModeElementType> , float2, int = comp);
# 1954 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1955 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1957 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1959 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< signed char, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1960 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1962 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1964 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< unsigned char, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1965 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1967 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1969 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1970 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1972 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1974 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1975 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1977 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1979 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1980 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1982 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1984 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1985 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1987 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1989 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1990 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1992 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1994 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1995 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1997 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1999 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2000 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2002 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2004 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2005 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2007 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2009 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2010 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2012 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2014 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< unsigned short, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2015 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2017 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2019 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2020 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2022 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2024 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2025 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2027 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2029 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2030 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2032 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2034 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2035 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2037 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2039 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2040 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2042 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2044 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2045 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2047 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2049 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2050 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2052 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2054 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2055 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2057 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2059 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2060 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2062 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2064 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< unsigned, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2065 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2067 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2069 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2070 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2072 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2074 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2075 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2077 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2079 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2080 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2082 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2084 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2085 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2087 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2089 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2090 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2092 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2094 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2095 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2097 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2099 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2102 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2104 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2105 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2109 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2114 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2115 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2119 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2122 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2124 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2129 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/texture_fetch_functions.h"
exit(___);}
# 53 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_launch_parameters.h"
extern "C" { extern const uint3 threadIdx; }
# 55 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_launch_parameters.h"
extern "C" { extern const uint3 blockIdx; }
# 57 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_launch_parameters.h"
extern "C" { extern const dim3 blockDim; }
# 59 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_launch_parameters.h"
extern "C" { extern const dim3 gridDim; }
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/device_launch_parameters.h"
extern "C" { extern const int warpSize; }
# 106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaSetupArgument(T
# 108 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
arg, size_t
# 109 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset)
# 111 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaSetupArgument((const void *)(&arg), sizeof(T), offset);
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 146 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
event, unsigned
# 147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
flags)
# 149 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaEventCreateWithFlags(event, 0);
# 151 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 209 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
ptr, size_t
# 210 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size, unsigned
# 211 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
flags)
# 213 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 214 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaHostAlloc(ptr, size, flags);
# 215 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 217 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaHostAlloc(T **
# 219 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
ptr, size_t
# 220 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size, unsigned
# 221 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
flags)
# 223 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 224 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaHostAlloc((void **)((void *)ptr), size, flags);
# 225 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 228 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaHostGetDevicePointer(T **
# 229 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
pDevice, void *
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
pHost, unsigned
# 231 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
flags)
# 233 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 234 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags);
# 235 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 237 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 238 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMalloc(T **
# 239 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, size_t
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size)
# 242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMalloc((void **)((void *)devPtr), size);
# 244 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 246 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMallocHost(T **
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
ptr, size_t
# 249 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size, unsigned
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
flags = (0))
# 252 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMallocHost((void **)((void *)ptr), size, flags);
# 254 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 256 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 257 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMallocPitch(T **
# 258 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, size_t *
# 259 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
pitch, size_t
# 260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
width, size_t
# 261 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
height)
# 263 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 264 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMallocPitch((void **)((void *)devPtr), pitch, width, height);
# 265 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 275 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbol(char *
# 276 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 277 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 279 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 280 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice)
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 283 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbol((const char *)symbol, src, count, offset, kind);
# 284 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 286 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 287 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyToSymbol(const T &
# 288 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 289 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 290 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 291 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 292 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice)
# 294 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 295 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbol((const char *)(&symbol), src, count, offset, kind);
# 296 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 298 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbolAsync(char *
# 299 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 300 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 301 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 302 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 303 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice, cudaStream_t
# 304 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
stream = 0)
# 306 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 307 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbolAsync((const char *)symbol, src, count, offset, kind, stream);
# 308 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 310 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 311 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyToSymbolAsync(const T &
# 312 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, const void *
# 313 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
src, size_t
# 314 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 315 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 316 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice, cudaStream_t
# 317 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
stream = 0)
# 319 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 320 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbolAsync((const char *)(&symbol), src, count, offset, kind, stream);
# 321 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 329 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbol(void *
# 330 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
dst, char *
# 331 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 332 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 333 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 334 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost)
# 336 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 337 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbol(dst, (const char *)symbol, count, offset, kind);
# 338 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 340 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 341 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyFromSymbol(void *
# 342 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
dst, const T &
# 343 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 344 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 345 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 346 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost)
# 348 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 349 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbol(dst, (const char *)(&symbol), count, offset, kind);
# 350 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 352 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbolAsync(void *
# 353 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
dst, char *
# 354 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 355 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 356 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 357 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost, cudaStream_t
# 358 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
stream = 0)
# 360 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 361 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbolAsync(dst, (const char *)symbol, count, offset, kind, stream);
# 362 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 364 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 365 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaMemcpyFromSymbolAsync(void *
# 366 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
dst, const T &
# 367 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol, size_t
# 368 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
count, size_t
# 369 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 370 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost, cudaStream_t
# 371 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
stream = 0)
# 373 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 374 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbolAsync(dst, (const char *)(&symbol), count, offset, kind, stream);
# 375 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 377 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolAddress(void **
# 378 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, char *
# 379 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol)
# 381 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 382 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolAddress(devPtr, (const char *)symbol);
# 383 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 410 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 411 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaGetSymbolAddress(void **
# 412 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, const T &
# 413 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol)
# 415 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 416 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolAddress(devPtr, (const char *)(&symbol));
# 417 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 425 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolSize(size_t *
# 426 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size, char *
# 427 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol)
# 429 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 430 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolSize(size, (const char *)symbol);
# 431 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 458 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 459 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaGetSymbolSize(size_t *
# 460 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size, const T &
# 461 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
symbol)
# 463 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 464 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaGetSymbolSize(size, (const char *)(&symbol));
# 465 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 507 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 508 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindTexture(size_t *
# 509 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 510 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex, const void *
# 511 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, const cudaChannelFormatDesc &
# 512 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
desc, size_t
# 513 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size = (((2147483647) * 2U) + 1U))
# 515 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 516 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaBindTexture(offset, &tex, devPtr, &desc, size);
# 517 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 552 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 553 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindTexture(size_t *
# 554 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 555 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex, const void *
# 556 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, size_t
# 557 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
size = (((2147483647) * 2U) + 1U))
# 559 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 560 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size);
# 561 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 608 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 609 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindTexture2D(size_t *
# 610 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 611 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex, const void *
# 612 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, const cudaChannelFormatDesc &
# 613 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
desc, size_t
# 614 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
width, size_t
# 615 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
height, size_t
# 616 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
pitch)
# 618 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 619 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
# 620 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 666 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 667 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindTexture2D(size_t *
# 668 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 669 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex, const void *
# 670 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
devPtr, size_t
# 671 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
width, size_t
# 672 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
height, size_t
# 673 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
pitch)
# 675 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 676 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaBindTexture2D(offset, &tex, devPtr, &(tex.texture< T, dim, readMode> ::channelDesc), width, height, pitch);
# 677 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 708 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 709 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindTextureToArray(const texture< T, dim, readMode> &
# 710 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex, const cudaArray *
# 711 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
array, const cudaChannelFormatDesc &
# 712 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
desc)
# 714 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 715 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaBindTextureToArray(&tex, array, &desc);
# 716 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 746 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 747 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindTextureToArray(const texture< T, dim, readMode> &
# 748 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex, const cudaArray *
# 749 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
array)
# 751 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 752 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaChannelFormatDesc desc;
# 753 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaError_t err = cudaGetChannelDesc(&desc, array);
# 755 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err;
# 756 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 785 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 786 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaUnbindTexture(const texture< T, dim, readMode> &
# 787 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex)
# 789 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 790 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaUnbindTexture(&tex);
# 791 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 825 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 826 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaGetTextureAlignmentOffset(size_t *
# 827 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 828 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
tex)
# 830 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 831 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaGetTextureAlignmentOffset(offset, &tex);
# 832 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 886 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 887 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaFuncSetCacheConfig(T *
# 888 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
func, cudaFuncCache
# 889 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cacheConfig)
# 891 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 892 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaFuncSetCacheConfig((const char *)func, cacheConfig);
# 893 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 930 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 931 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaLaunch(T *
# 932 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
entry)
# 934 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 935 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaLaunch((const char *)entry);
# 936 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 970 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 971 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaFuncGetAttributes(cudaFuncAttributes *
# 972 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
attr, T *
# 973 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
entry)
# 975 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 976 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaFuncGetAttributes(attr, (const char *)entry);
# 977 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 999 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim> inline cudaError_t
# 1000 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindSurfaceToArray(const surface< T, dim> &
# 1001 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
surf, const cudaArray *
# 1002 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
array, const cudaChannelFormatDesc &
# 1003 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
desc)
# 1005 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 1006 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return cudaBindSurfaceToArray(&surf, array, &desc);
# 1007 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 1028 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
template< class T, int dim> inline cudaError_t
# 1029 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaBindSurfaceToArray(const surface< T, dim> &
# 1030 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
surf, const cudaArray *
# 1031 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
array)
# 1033 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
{
# 1034 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaChannelFormatDesc desc;
# 1035 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
cudaError_t err = cudaGetChannelDesc(&desc, array);
# 1037 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err;
# 1038 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuda_runtime.h"
}
# 45 "/usr/include/stdio.h" 3
struct _IO_FILE;
# 49 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE FILE; }
# 65 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE __FILE; }
# 95 "/usr/include/wchar.h" 3
extern "C" { typedef
# 84 "/usr/include/wchar.h" 3
struct {
# 85 "/usr/include/wchar.h" 3
int __count;
# 87 "/usr/include/wchar.h" 3
union {
# 89 "/usr/include/wchar.h" 3
unsigned __wch;
# 93 "/usr/include/wchar.h" 3
char __wchb[4];
# 94 "/usr/include/wchar.h" 3
} __value;
# 95 "/usr/include/wchar.h" 3
} __mbstate_t; }
# 26 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 23 "/usr/include/_G_config.h" 3
struct {
# 24 "/usr/include/_G_config.h" 3
__off_t __pos;
# 25 "/usr/include/_G_config.h" 3
__mbstate_t __state;
# 26 "/usr/include/_G_config.h" 3
} _G_fpos_t; }
# 31 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 28 "/usr/include/_G_config.h" 3
struct {
# 29 "/usr/include/_G_config.h" 3
__off64_t __pos;
# 30 "/usr/include/_G_config.h" 3
__mbstate_t __state;
# 31 "/usr/include/_G_config.h" 3
} _G_fpos64_t; }
# 53 "/usr/include/_G_config.h" 3
extern "C" { typedef short _G_int16_t; }
# 54 "/usr/include/_G_config.h" 3
extern "C" { typedef int _G_int32_t; }
# 55 "/usr/include/_G_config.h" 3
extern "C" { typedef unsigned short _G_uint16_t; }
# 56 "/usr/include/_G_config.h" 3
extern "C" { typedef unsigned _G_uint32_t; }
# 40 "/usr/lib/gcc/x86_64-linux-gnu/4.4.5/include/stdarg.h" 3
extern "C" { typedef __builtin_va_list __gnuc_va_list; }
# 170 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3
extern "C" { typedef void _IO_lock_t; }
# 186 "/usr/include/libio.h" 3
extern "C" { struct _IO_marker {
# 187 "/usr/include/libio.h" 3
_IO_marker *_next;
# 188 "/usr/include/libio.h" 3
_IO_FILE *_sbuf;
# 192 "/usr/include/libio.h" 3
int _pos;
# 203 "/usr/include/libio.h" 3
}; }
# 206 "/usr/include/libio.h" 3
enum __codecvt_result {
# 208 "/usr/include/libio.h" 3
__codecvt_ok,
# 209 "/usr/include/libio.h" 3
__codecvt_partial,
# 210 "/usr/include/libio.h" 3
__codecvt_error,
# 211 "/usr/include/libio.h" 3
__codecvt_noconv
# 212 "/usr/include/libio.h" 3
};
# 271 "/usr/include/libio.h" 3
extern "C" { struct _IO_FILE {
# 272 "/usr/include/libio.h" 3
int _flags;
# 277 "/usr/include/libio.h" 3
char *_IO_read_ptr;
# 278 "/usr/include/libio.h" 3
char *_IO_read_end;
# 279 "/usr/include/libio.h" 3
char *_IO_read_base;
# 280 "/usr/include/libio.h" 3
char *_IO_write_base;
# 281 "/usr/include/libio.h" 3
char *_IO_write_ptr;
# 282 "/usr/include/libio.h" 3
char *_IO_write_end;
# 283 "/usr/include/libio.h" 3
char *_IO_buf_base;
# 284 "/usr/include/libio.h" 3
char *_IO_buf_end;
# 286 "/usr/include/libio.h" 3
char *_IO_save_base;
# 287 "/usr/include/libio.h" 3
char *_IO_backup_base;
# 288 "/usr/include/libio.h" 3
char *_IO_save_end;
# 290 "/usr/include/libio.h" 3
_IO_marker *_markers;
# 292 "/usr/include/libio.h" 3
_IO_FILE *_chain;
# 294 "/usr/include/libio.h" 3
int _fileno;
# 298 "/usr/include/libio.h" 3
int _flags2;
# 300 "/usr/include/libio.h" 3
__off_t _old_offset;
# 304 "/usr/include/libio.h" 3
unsigned short _cur_column;
# 305 "/usr/include/libio.h" 3
signed char _vtable_offset;
# 306 "/usr/include/libio.h" 3
char _shortbuf[1];
# 310 "/usr/include/libio.h" 3
_IO_lock_t *_lock;
# 319 "/usr/include/libio.h" 3
__off64_t _offset;
# 328 "/usr/include/libio.h" 3
void *__pad1;
# 329 "/usr/include/libio.h" 3
void *__pad2;
# 330 "/usr/include/libio.h" 3
void *__pad3;
# 331 "/usr/include/libio.h" 3
void *__pad4;
# 332 "/usr/include/libio.h" 3
size_t __pad5;
# 334 "/usr/include/libio.h" 3
int _mode;
# 336 "/usr/include/libio.h" 3
char _unused2[((((15) * sizeof(int)) - ((4) * sizeof(void *))) - sizeof(size_t))];
# 338 "/usr/include/libio.h" 3
}; }
# 344 "/usr/include/libio.h" 3
struct _IO_FILE_plus;
# 346 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stdin_; }
# 347 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stdout_; }
# 348 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stderr_; }
# 364 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_read_fn(void *, char *, size_t); }
# 372 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_write_fn(void *, const char *, size_t); }
# 381 "/usr/include/libio.h" 3
extern "C" { typedef int __io_seek_fn(void *, __off64_t *, int); }
# 384 "/usr/include/libio.h" 3
extern "C" { typedef int __io_close_fn(void *); }
# 389 "/usr/include/libio.h" 3
extern "C" { typedef __io_read_fn cookie_read_function_t; }
# 390 "/usr/include/libio.h" 3
extern "C" { typedef __io_write_fn cookie_write_function_t; }
# 391 "/usr/include/libio.h" 3
extern "C" { typedef __io_seek_fn cookie_seek_function_t; }
# 392 "/usr/include/libio.h" 3
extern "C" { typedef __io_close_fn cookie_close_function_t; }
# 401 "/usr/include/libio.h" 3
extern "C" { typedef
# 396 "/usr/include/libio.h" 3
struct {
# 397 "/usr/include/libio.h" 3
__io_read_fn *read;
# 398 "/usr/include/libio.h" 3
__io_write_fn *write;
# 399 "/usr/include/libio.h" 3
__io_seek_fn *seek;
# 400 "/usr/include/libio.h" 3
__io_close_fn *close;
# 401 "/usr/include/libio.h" 3
} _IO_cookie_io_functions_t; }
# 402 "/usr/include/libio.h" 3
extern "C" { typedef _IO_cookie_io_functions_t cookie_io_functions_t; }
# 404 "/usr/include/libio.h" 3
struct _IO_cookie_file;
# 407 "/usr/include/libio.h" 3
extern "C" void _IO_cookie_init(_IO_cookie_file *, int, void *, _IO_cookie_io_functions_t);
# 416 "/usr/include/libio.h" 3
extern "C" int __underflow(_IO_FILE *);
# 417 "/usr/include/libio.h" 3
extern "C" int __uflow(_IO_FILE *);
# 418 "/usr/include/libio.h" 3
extern "C" int __overflow(_IO_FILE *, int);
# 460 "/usr/include/libio.h" 3
extern "C" int _IO_getc(_IO_FILE *);
# 461 "/usr/include/libio.h" 3
extern "C" int _IO_putc(int, _IO_FILE *);
# 462 "/usr/include/libio.h" 3
extern "C" int _IO_feof(_IO_FILE *) throw();
# 463 "/usr/include/libio.h" 3
extern "C" int _IO_ferror(_IO_FILE *) throw();
# 465 "/usr/include/libio.h" 3
extern "C" int _IO_peekc_locked(_IO_FILE *);
# 471 "/usr/include/libio.h" 3
extern "C" void _IO_flockfile(_IO_FILE *) throw();
# 472 "/usr/include/libio.h" 3
extern "C" void _IO_funlockfile(_IO_FILE *) throw();
# 473 "/usr/include/libio.h" 3
extern "C" int _IO_ftrylockfile(_IO_FILE *) throw();
# 490 "/usr/include/libio.h" 3
extern "C" int _IO_vfscanf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list, int *__restrict__);
# 492 "/usr/include/libio.h" 3
extern "C" int _IO_vfprintf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 494 "/usr/include/libio.h" 3
extern "C" __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t);
# 495 "/usr/include/libio.h" 3
extern "C" size_t _IO_sgetn(_IO_FILE *, void *, size_t);
# 497 "/usr/include/libio.h" 3
extern "C" __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int);
# 498 "/usr/include/libio.h" 3
extern "C" __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int);
# 500 "/usr/include/libio.h" 3
extern "C" void _IO_free_backup_area(_IO_FILE *) throw();
# 80 "/usr/include/stdio.h" 3
extern "C" { typedef __gnuc_va_list va_list; }
# 91 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos_t fpos_t; }
# 97 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos64_t fpos64_t; }
# 145 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdin; }
# 146 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdout; }
# 147 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stderr; }
# 155 "/usr/include/stdio.h" 3
extern "C" int remove(const char *) throw();
# 157 "/usr/include/stdio.h" 3
extern "C" int rename(const char *, const char *) throw();
# 162 "/usr/include/stdio.h" 3
extern "C" int renameat(int, const char *, int, const char *) throw();
# 172 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile();
# 182 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile64();
# 186 "/usr/include/stdio.h" 3
extern "C" char *tmpnam(char *) throw();
# 192 "/usr/include/stdio.h" 3
extern "C" char *tmpnam_r(char *) throw();
# 204 "/usr/include/stdio.h" 3
extern "C" char *tempnam(const char *, const char *) throw() __attribute__((__malloc__));
# 214 "/usr/include/stdio.h" 3
extern "C" int fclose(FILE *);
# 219 "/usr/include/stdio.h" 3
extern "C" int fflush(FILE *);
# 229 "/usr/include/stdio.h" 3
extern "C" int fflush_unlocked(FILE *);
# 239 "/usr/include/stdio.h" 3
extern "C" int fcloseall();
# 249 "/usr/include/stdio.h" 3
extern "C" FILE *fopen(const char *__restrict__, const char *__restrict__);
# 255 "/usr/include/stdio.h" 3
extern "C" FILE *freopen(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 274 "/usr/include/stdio.h" 3
extern "C" FILE *fopen64(const char *__restrict__, const char *__restrict__);
# 276 "/usr/include/stdio.h" 3
extern "C" FILE *freopen64(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 283 "/usr/include/stdio.h" 3
extern "C" FILE *fdopen(int, const char *) throw();
# 289 "/usr/include/stdio.h" 3
extern "C" FILE *fopencookie(void *__restrict__, const char *__restrict__, _IO_cookie_io_functions_t) throw();
# 296 "/usr/include/stdio.h" 3
extern "C" FILE *fmemopen(void *, size_t, const char *) throw();
# 302 "/usr/include/stdio.h" 3
extern "C" FILE *open_memstream(char **, size_t *) throw();
# 309 "/usr/include/stdio.h" 3
extern "C" void setbuf(FILE *__restrict__, char *__restrict__) throw();
# 313 "/usr/include/stdio.h" 3
extern "C" int setvbuf(FILE *__restrict__, char *__restrict__, int, size_t) throw();
# 320 "/usr/include/stdio.h" 3
extern "C" void setbuffer(FILE *__restrict__, char *__restrict__, size_t) throw();
# 324 "/usr/include/stdio.h" 3
extern "C" void setlinebuf(FILE *) throw();
# 333 "/usr/include/stdio.h" 3
extern "C" int fprintf(FILE *__restrict__, const char *__restrict__, ...);
# 339 "/usr/include/stdio.h" 3
extern "C" int printf(const char *__restrict__, ...);
# 341 "/usr/include/stdio.h" 3
extern "C" int sprintf(char *__restrict__, const char *__restrict__, ...) throw();
# 348 "/usr/include/stdio.h" 3
extern "C" int vfprintf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 354 "/usr/include/stdio.h" 3
extern "C" int vprintf(const char *__restrict__, __gnuc_va_list);
# 356 "/usr/include/stdio.h" 3
extern "C" int vsprintf(char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 363 "/usr/include/stdio.h" 3
extern "C" int snprintf(char *__restrict__, size_t, const char *__restrict__, ...) throw();
# 367 "/usr/include/stdio.h" 3
extern "C" int vsnprintf(char *__restrict__, size_t, const char *__restrict__, __gnuc_va_list) throw();
# 376 "/usr/include/stdio.h" 3
extern "C" int vasprintf(char **__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 379 "/usr/include/stdio.h" 3
extern "C" int __asprintf(char **__restrict__, const char *__restrict__, ...) throw();
# 382 "/usr/include/stdio.h" 3
extern "C" int asprintf(char **__restrict__, const char *__restrict__, ...) throw();
# 394 "/usr/include/stdio.h" 3
extern "C" int vdprintf(int, const char *__restrict__, __gnuc_va_list);
# 397 "/usr/include/stdio.h" 3
extern "C" int dprintf(int, const char *__restrict__, ...);
# 407 "/usr/include/stdio.h" 3
extern "C" int fscanf(FILE *__restrict__, const char *__restrict__, ...);
# 413 "/usr/include/stdio.h" 3
extern "C" int scanf(const char *__restrict__, ...);
# 415 "/usr/include/stdio.h" 3
extern "C" int sscanf(const char *__restrict__, const char *__restrict__, ...) throw();
# 453 "/usr/include/stdio.h" 3
extern "C" int vfscanf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 461 "/usr/include/stdio.h" 3
extern "C" int vscanf(const char *__restrict__, __gnuc_va_list);
# 465 "/usr/include/stdio.h" 3
extern "C" int vsscanf(const char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 513 "/usr/include/stdio.h" 3
extern "C" int fgetc(FILE *);
# 514 "/usr/include/stdio.h" 3
extern "C" int getc(FILE *);
# 520 "/usr/include/stdio.h" 3
extern "C" int getchar();
# 532 "/usr/include/stdio.h" 3
extern "C" int getc_unlocked(FILE *);
# 533 "/usr/include/stdio.h" 3
extern "C" int getchar_unlocked();
# 543 "/usr/include/stdio.h" 3
extern "C" int fgetc_unlocked(FILE *);
# 555 "/usr/include/stdio.h" 3
extern "C" int fputc(int, FILE *);
# 556 "/usr/include/stdio.h" 3
extern "C" int putc(int, FILE *);
# 562 "/usr/include/stdio.h" 3
extern "C" int putchar(int);
# 576 "/usr/include/stdio.h" 3
extern "C" int fputc_unlocked(int, FILE *);
# 584 "/usr/include/stdio.h" 3
extern "C" int putc_unlocked(int, FILE *);
# 585 "/usr/include/stdio.h" 3
extern "C" int putchar_unlocked(int);
# 592 "/usr/include/stdio.h" 3
extern "C" int getw(FILE *);
# 595 "/usr/include/stdio.h" 3
extern "C" int putw(int, FILE *);
# 604 "/usr/include/stdio.h" 3
extern "C" char *fgets(char *__restrict__, int, FILE *__restrict__);
# 612 "/usr/include/stdio.h" 3
extern "C" char *gets(char *);
# 622 "/usr/include/stdio.h" 3
extern "C" char *fgets_unlocked(char *__restrict__, int, FILE *__restrict__);
# 638 "/usr/include/stdio.h" 3
extern "C" __ssize_t __getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 641 "/usr/include/stdio.h" 3
extern "C" __ssize_t getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 651 "/usr/include/stdio.h" 3
extern "C" __ssize_t getline(char **__restrict__, size_t *__restrict__, FILE *__restrict__);
# 662 "/usr/include/stdio.h" 3
extern "C" int fputs(const char *__restrict__, FILE *__restrict__);
# 668 "/usr/include/stdio.h" 3
extern "C" int puts(const char *);
# 675 "/usr/include/stdio.h" 3
extern "C" int ungetc(int, FILE *);
# 682 "/usr/include/stdio.h" 3
extern "C" size_t fread(void *__restrict__, size_t, size_t, FILE *__restrict__);
# 688 "/usr/include/stdio.h" 3
extern "C" size_t fwrite(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 699 "/usr/include/stdio.h" 3
extern "C" int fputs_unlocked(const char *__restrict__, FILE *__restrict__);
# 710 "/usr/include/stdio.h" 3
extern "C" size_t fread_unlocked(void *__restrict__, size_t, size_t, FILE *__restrict__);
# 712 "/usr/include/stdio.h" 3
extern "C" size_t fwrite_unlocked(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 722 "/usr/include/stdio.h" 3
extern "C" int fseek(FILE *, long, int);
# 727 "/usr/include/stdio.h" 3
extern "C" long ftell(FILE *);
# 732 "/usr/include/stdio.h" 3
extern "C" void rewind(FILE *);
# 746 "/usr/include/stdio.h" 3
extern "C" int fseeko(FILE *, __off_t, int);
# 751 "/usr/include/stdio.h" 3
extern "C" __off_t ftello(FILE *);
# 771 "/usr/include/stdio.h" 3
extern "C" int fgetpos(FILE *__restrict__, fpos_t *__restrict__);
# 776 "/usr/include/stdio.h" 3
extern "C" int fsetpos(FILE *, const fpos_t *);
# 791 "/usr/include/stdio.h" 3
extern "C" int fseeko64(FILE *, __off64_t, int);
# 792 "/usr/include/stdio.h" 3
extern "C" __off64_t ftello64(FILE *);
# 793 "/usr/include/stdio.h" 3
extern "C" int fgetpos64(FILE *__restrict__, fpos64_t *__restrict__);
# 794 "/usr/include/stdio.h" 3
extern "C" int fsetpos64(FILE *, const fpos64_t *);
# 799 "/usr/include/stdio.h" 3
extern "C" void clearerr(FILE *) throw();
# 801 "/usr/include/stdio.h" 3
extern "C" int feof(FILE *) throw();
# 803 "/usr/include/stdio.h" 3
extern "C" int ferror(FILE *) throw();
# 808 "/usr/include/stdio.h" 3
extern "C" void clearerr_unlocked(FILE *) throw();
# 809 "/usr/include/stdio.h" 3
extern "C" int feof_unlocked(FILE *) throw();
# 810 "/usr/include/stdio.h" 3
extern "C" int ferror_unlocked(FILE *) throw();
# 819 "/usr/include/stdio.h" 3
extern "C" void perror(const char *);
# 27 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern int sys_nerr; }
# 28 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern const char *const sys_errlist[]; }
# 31 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern int _sys_nerr; }
# 32 "/usr/include/bits/sys_errlist.h" 3
extern "C" { extern const char *const _sys_errlist[]; }
# 831 "/usr/include/stdio.h" 3
extern "C" int fileno(FILE *) throw();
# 836 "/usr/include/stdio.h" 3
extern "C" int fileno_unlocked(FILE *) throw();
# 846 "/usr/include/stdio.h" 3
extern "C" FILE *popen(const char *, const char *);
# 852 "/usr/include/stdio.h" 3
extern "C" int pclose(FILE *);
# 858 "/usr/include/stdio.h" 3
extern "C" char *ctermid(char *) throw();
# 864 "/usr/include/stdio.h" 3
extern "C" char *cuserid(char *);
# 869 "/usr/include/stdio.h" 3
struct obstack;
# 872 "/usr/include/stdio.h" 3
extern "C" int obstack_printf(obstack *__restrict__, const char *__restrict__, ...) throw();
# 875 "/usr/include/stdio.h" 3
extern "C" int obstack_vprintf(obstack *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 886 "/usr/include/stdio.h" 3
extern "C" void flockfile(FILE *) throw();
# 890 "/usr/include/stdio.h" 3
extern "C" int ftrylockfile(FILE *) throw();
# 893 "/usr/include/stdio.h" 3
extern "C" void funlockfile(FILE *) throw();
# 46 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { typedef float2 cuFloatComplex; }
# 48 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline float cuCrealf(cuFloatComplex x)
# 49 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 50 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return x.x;
# 51 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 53 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline float cuCimagf(cuFloatComplex x)
# 54 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 55 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return x.y;
# 56 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 58 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuFloatComplex make_cuFloatComplex(float
# 59 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
r, float i)
# 60 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
cuFloatComplex res;
# 62 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
(res.x) = r;
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
(res.y) = i;
# 64 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return res;
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 67 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuFloatComplex cuConjf(cuFloatComplex x)
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuFloatComplex(cuCrealf(x), -cuCimagf(x));
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 73 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 74 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuFloatComplex(cuCrealf(x) + cuCrealf(y), cuCimagf(x) + cuCimagf(y));
# 76 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 78 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex
# 79 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 80 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuFloatComplex(cuCrealf(x) - cuCrealf(y), cuCimagf(x) - cuCimagf(y));
# 83 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 90 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex
# 91 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 92 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 93 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
cuFloatComplex prod;
# 94 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
prod = make_cuFloatComplex((cuCrealf(x) * cuCrealf(y)) - (cuCimagf(x) * cuCimagf(y)), (cuCrealf(x) * cuCimagf(y)) + (cuCimagf(x) * cuCrealf(y)));
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return prod;
# 99 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 106 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex
# 107 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 108 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 109 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
cuFloatComplex quot;
# 110 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float s = (fabsf(cuCrealf(y)) + fabsf(cuCimagf(y)));
# 111 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float oos = ((1.0F) / s);
# 112 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float ars = (cuCrealf(x) * oos);
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float ais = (cuCimagf(x) * oos);
# 114 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float brs = (cuCrealf(y) * oos);
# 115 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float bis = (cuCimagf(y) * oos);
# 116 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
s = ((brs * brs) + (bis * bis));
# 117 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
oos = ((1.0F) / s);
# 118 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
quot = make_cuFloatComplex(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
# 120 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return quot;
# 121 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline float cuCabsf(cuFloatComplex x)
# 132 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 133 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float a = cuCrealf(x);
# 134 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float b = cuCimagf(x);
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float v, w, t;
# 136 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
a = fabsf(a);
# 137 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
b = fabsf(b);
# 138 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
if (a > b) {
# 139 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
v = a;
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
w = b;
# 141 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} else {
# 142 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
v = b;
# 143 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
w = a;
# 144 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 145 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = (w / v);
# 146 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = ((1.0F) + (t * t));
# 147 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = (v * sqrtf(t));
# 148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
if (((v == (0.0F)) || (v > (3.402823466e+38F))) || (w > (3.402823466e+38F))) {
# 149 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = (v + w);
# 150 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 151 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return t;
# 152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 155 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { typedef double2 cuDoubleComplex; }
# 157 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline double cuCreal(cuDoubleComplex x)
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 159 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return x.x;
# 160 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 162 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline double cuCimag(cuDoubleComplex x)
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 164 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return x.y;
# 165 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuDoubleComplex make_cuDoubleComplex(double
# 168 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
r, double i)
# 169 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 170 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
cuDoubleComplex res;
# 171 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
(res.x) = r;
# 172 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
(res.y) = i;
# 173 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return res;
# 174 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 176 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuDoubleComplex cuConj(cuDoubleComplex x)
# 177 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 178 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuDoubleComplex(cuCreal(x), -cuCimag(x));
# 179 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 181 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex
# 182 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 183 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 184 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuDoubleComplex(cuCreal(x) + cuCreal(y), cuCimag(x) + cuCimag(y));
# 186 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 188 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex
# 189 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 190 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 191 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuDoubleComplex(cuCreal(x) - cuCreal(y), cuCimag(x) - cuCimag(y));
# 193 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 200 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex
# 201 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 202 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
cuDoubleComplex prod;
# 204 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
prod = make_cuDoubleComplex((cuCreal(x) * cuCreal(y)) - (cuCimag(x) * cuCimag(y)), (cuCreal(x) * cuCimag(y)) + (cuCimag(x) * cuCreal(y)));
# 208 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return prod;
# 209 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 216 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex
# 217 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 218 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 219 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
cuDoubleComplex quot;
# 220 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double s = (fabs(cuCreal(y)) + fabs(cuCimag(y)));
# 221 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double oos = ((1.0) / s);
# 222 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double ars = (cuCreal(x) * oos);
# 223 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double ais = (cuCimag(x) * oos);
# 224 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double brs = (cuCreal(y) * oos);
# 225 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double bis = (cuCimag(y) * oos);
# 226 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
s = ((brs * brs) + (bis * bis));
# 227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
oos = ((1.0) / s);
# 228 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
quot = make_cuDoubleComplex(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
# 230 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return quot;
# 231 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 239 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
extern "C" { static inline double cuCabs(cuDoubleComplex x)
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 241 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double a = cuCreal(x);
# 242 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double b = cuCimag(x);
# 243 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double v, w, t;
# 244 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
a = fabs(a);
# 245 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
b = fabs(b);
# 246 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
if (a > b) {
# 247 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
v = a;
# 248 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
w = b;
# 249 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} else {
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
v = b;
# 251 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
w = a;
# 252 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 253 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = (w / v);
# 254 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = ((1.0) + (t * t));
# 255 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = (v * sqrt(t));
# 256 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
if (((v == (0.0)) || (v > (1.797693134862315708e+308))) || (w > (1.797693134862315708e+308)))
# 257 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 258 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
t = (v + w);
# 259 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 260 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return t;
# 261 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
} }
# 268 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
typedef cuFloatComplex cuComplex;
# 269 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
static inline cuComplex make_cuComplex(float x, float
# 270 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
y)
# 271 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 272 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuFloatComplex(x, y);
# 273 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 276 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
static inline cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex
# 277 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
c)
# 278 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 279 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuDoubleComplex((double)cuCrealf(c), (double)cuCimagf(c));
# 280 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 282 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
static inline cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex
# 283 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
c)
# 284 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 285 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuFloatComplex((float)cuCreal(c), (float)cuCimag(c));
# 286 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 289 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
static inline cuComplex cuCfmaf(cuComplex x, cuComplex y, cuComplex d)
# 290 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 291 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float real_res;
# 292 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
float imag_res;
# 294 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
real_res = ((cuCrealf(x) * cuCrealf(y)) + cuCrealf(d));
# 295 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
imag_res = ((cuCrealf(x) * cuCimagf(y)) + cuCimagf(d));
# 297 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
real_res = ((-(cuCimagf(x) * cuCimagf(y))) + real_res);
# 298 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
imag_res = ((cuCimagf(x) * cuCrealf(y)) + imag_res);
# 300 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuComplex(real_res, imag_res);
# 301 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 303 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
static inline cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d)
# 304 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
{
# 305 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double real_res;
# 306 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
double imag_res;
# 308 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
real_res = ((cuCreal(x) * cuCreal(y)) + cuCreal(d));
# 309 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
imag_res = ((cuCreal(x) * cuCimag(y)) + cuCimag(d));
# 311 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
real_res = ((-(cuCimag(x) * cuCimag(y))) + real_res);
# 312 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
imag_res = ((cuCimag(x) * cuCreal(y)) + imag_res);
# 314 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
return make_cuDoubleComplex(real_res, imag_res);
# 315 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cuComplex.h"
}
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef
# 61 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
enum cufftResult_t {
# 62 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_SUCCESS,
# 63 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_INVALID_PLAN,
# 64 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_ALLOC_FAILED,
# 65 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_INVALID_TYPE,
# 66 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_INVALID_VALUE,
# 67 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_INTERNAL_ERROR,
# 68 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_EXEC_FAILED,
# 69 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_SETUP_FAILED,
# 70 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_INVALID_SIZE,
# 71 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_UNALIGNED_DATA
# 72 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
} cufftResult; }
# 77 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef unsigned cufftHandle; }
# 81 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef float cufftReal; }
# 82 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef double cufftDoubleReal; }
# 87 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef cuComplex cufftComplex; }
# 88 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef cuDoubleComplex cufftDoubleComplex; }
# 102 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef
# 95 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
enum cufftType_t {
# 96 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_R2C = 42,
# 97 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_C2R = 44,
# 98 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_C2C = 41,
# 99 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_D2Z = 106,
# 100 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_Z2D = 108,
# 101 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_Z2Z = 105
# 102 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
} cufftType; }
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" { typedef
# 126 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
enum cufftCompatibility_t {
# 127 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_COMPATIBILITY_NATIVE,
# 128 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_COMPATIBILITY_FFTW_PADDING,
# 129 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC,
# 130 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
CUFFT_COMPATIBILITY_FFTW_ALL
# 131 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
} cufftCompatibility; }
# 135 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftPlan1d(cufftHandle *, int, cufftType, int);
# 140 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftPlan2d(cufftHandle *, int, int, cufftType);
# 144 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftPlan3d(cufftHandle *, int, int, int, cufftType);
# 148 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftPlanMany(cufftHandle *, int, int *, int *, int, int, int *, int, int, cufftType, int);
# 156 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftDestroy(cufftHandle);
# 158 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftExecC2C(cufftHandle, cufftComplex *, cufftComplex *, int);
# 163 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftExecR2C(cufftHandle, cufftReal *, cufftComplex *);
# 167 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftExecC2R(cufftHandle, cufftComplex *, cufftReal *);
# 171 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftExecZ2Z(cufftHandle, cufftDoubleComplex *, cufftDoubleComplex *, int);
# 176 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftExecD2Z(cufftHandle, cufftDoubleReal *, cufftDoubleComplex *);
# 180 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftExecZ2D(cufftHandle, cufftDoubleComplex *, cufftDoubleReal *);
# 184 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftSetStream(cufftHandle, cudaStream_t);
# 187 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/cufft.h"
extern "C" cufftResult cufftSetCompatibilityMode(cufftHandle, cufftCompatibility);
# 209 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef char CHAR; }
# 210 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef unsigned char UCHAR; }
# 211 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef unsigned char BOOLEAN; }
# 49 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned char uint8_t; }
# 50 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned short uint16_t; }
# 52 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned uint32_t; }
# 56 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uint64_t; }
# 66 "/usr/include/stdint.h" 3
extern "C" { typedef signed char int_least8_t; }
# 67 "/usr/include/stdint.h" 3
extern "C" { typedef short int_least16_t; }
# 68 "/usr/include/stdint.h" 3
extern "C" { typedef int int_least32_t; }
# 70 "/usr/include/stdint.h" 3
extern "C" { typedef long int_least64_t; }
# 77 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned char uint_least8_t; }
# 78 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned short uint_least16_t; }
# 79 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned uint_least32_t; }
# 81 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uint_least64_t; }
# 91 "/usr/include/stdint.h" 3
extern "C" { typedef signed char int_fast8_t; }
# 93 "/usr/include/stdint.h" 3
extern "C" { typedef long int_fast16_t; }
# 94 "/usr/include/stdint.h" 3
extern "C" { typedef long int_fast32_t; }
# 95 "/usr/include/stdint.h" 3
extern "C" { typedef long int_fast64_t; }
# 104 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned char uint_fast8_t; }
# 106 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uint_fast16_t; }
# 107 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uint_fast32_t; }
# 108 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uint_fast64_t; }
# 120 "/usr/include/stdint.h" 3
extern "C" { typedef long intptr_t; }
# 123 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uintptr_t; }
# 135 "/usr/include/stdint.h" 3
extern "C" { typedef long intmax_t; }
# 136 "/usr/include/stdint.h" 3
extern "C" { typedef unsigned long uintmax_t; }
# 114 "/home/kawies/dev/include/lal/LALRCSID.h"
static const volatile char *LALRCSIDH __attribute__((__unused__)) = ("$Id$");
# 216 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
static const volatile char *LALATOMICDATATYPESH __attribute__((__unused__)) = ("$Id$");
# 234 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef int16_t INT2; }
# 235 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef int32_t INT4; }
# 236 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef int64_t INT8; }
# 237 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef uint16_t UINT2; }
# 238 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef uint32_t UINT4; }
# 239 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef uint64_t UINT8; }
# 246 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef float REAL4; }
# 247 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef double REAL8; }
# 291 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef
# 286 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
struct tagCOMPLEX8 {
# 288 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
REAL4 re;
# 289 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
REAL4 im;
# 291 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
} COMPLEX8; }
# 300 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
extern "C" { typedef
# 295 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
struct tagCOMPLEX16 {
# 297 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
REAL8 re;
# 298 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
REAL8 im;
# 300 "/home/kawies/dev/include/lal/LALAtomicDatatypes.h"
} COMPLEX16; }
# 35 "Chisq_GPU.cu"
static void CudaError(cudaError_t error, const char *file, int line)
# 36 "Chisq_GPU.cu"
{
# 37 "Chisq_GPU.cu"
if (error != (cudaSuccess))
# 38 "Chisq_GPU.cu"
{
# 39 "Chisq_GPU.cu"
printf("%s:%d %s\n", file, line, cudaGetErrorString(error));
# 40 "Chisq_GPU.cu"
exit(-1);
# 41 "Chisq_GPU.cu"
}
# 42 "Chisq_GPU.cu"
}
# 56 "Chisq_GPU.cu"
 __attribute__((noinline)) void chisqKernel(REAL4 *g_chisq, COMPLEX8 *g_q, COMPLEX8 *g_data, UINT4
# 57 "Chisq_GPU.cu"
numPoints, UINT4 numChisqBins, REAL4 chisqNorm) ;
# 84 "Chisq_GPU.cu"
extern "C" void Chisq_GPU(REAL4 *chisq, COMPLEX8 *q, COMPLEX8 *qtilde, UINT4 *chisqBin, UINT4
# 85 "Chisq_GPU.cu"
numPoints, UINT4 numChisqBins, REAL4 chisqNorm)
# 86 "Chisq_GPU.cu"
{
# 90 "Chisq_GPU.cu"
COMPLEX8 *d_q;
# 91 "Chisq_GPU.cu"
CudaError(cudaMalloc((void **)(&d_q), numPoints * sizeof(COMPLEX8)), "Chisq_GPU.cu", 91);
# 93 "Chisq_GPU.cu"
CudaError(cudaMemcpy(d_q, q, numPoints * sizeof(COMPLEX8), cudaMemcpyHostToDevice), "Chisq_GPU.cu", 93);
# 96 "Chisq_GPU.cu"
REAL4 *d_chisq;
# 97 "Chisq_GPU.cu"
CudaError(cudaMalloc((void **)(&d_chisq), numPoints * sizeof(REAL4)), "Chisq_GPU.cu", 97);
# 98 "Chisq_GPU.cu"
CudaError(cudaMemset(d_chisq, 0, numPoints * sizeof(REAL4)), "Chisq_GPU.cu", 98);
# 103 "Chisq_GPU.cu"
COMPLEX8 *d_data;
# 104 "Chisq_GPU.cu"
CudaError(cudaMalloc((void **)(&d_data), (numPoints * numChisqBins) * sizeof(COMPLEX8)), "Chisq_GPU.cu", 104);
# 105 "Chisq_GPU.cu"
CudaError(cudaMemset(d_data, 0, (numPoints * numChisqBins) * sizeof(COMPLEX8)), "Chisq_GPU.cu", 105);
# 108 "Chisq_GPU.cu"
COMPLEX8 *d_qtildeBin;
# 109 "Chisq_GPU.cu"
CudaError(cudaMalloc((void **)(&d_qtildeBin), (numPoints * numChisqBins) * sizeof(COMPLEX8)), "Chisq_GPU.cu", 109);
# 110 "Chisq_GPU.cu"
CudaError(cudaMemset(d_qtildeBin, 0, (numPoints * numChisqBins) * sizeof(COMPLEX8)), "Chisq_GPU.cu", 110);
# 117 "Chisq_GPU.cu"
for (unsigned i = (0); i < numChisqBins; i++)
# 118 "Chisq_GPU.cu"
{
# 124 "Chisq_GPU.cu"
CudaError(cudaMemcpy((&(d_qtildeBin[i * numPoints])) + (chisqBin[i]), qtilde + (chisqBin[i]), ((chisqBin[i + (1)]) - (chisqBin[i])) * sizeof(COMPLEX8), cudaMemcpyHostToDevice), "Chisq_GPU.cu", 127);
# 128 "Chisq_GPU.cu"
}
# 132 "Chisq_GPU.cu"
cufftHandle batchPlan;
# 133 "Chisq_GPU.cu"
cufftPlan1d(&batchPlan, numPoints, CUFFT_C2C, numChisqBins);
# 135 "Chisq_GPU.cu"
cudaEvent_t start, stop;
# 136 "Chisq_GPU.cu"
CudaError(cudaEventCreate(&start), "Chisq_GPU.cu", 136);
# 137 "Chisq_GPU.cu"
CudaError(cudaEventCreate(&stop), "Chisq_GPU.cu", 137);
# 138 "Chisq_GPU.cu"
CudaError(cudaEventRecord(start, 0), "Chisq_GPU.cu", 138);
# 140 "Chisq_GPU.cu"
cufftExecC2C(batchPlan, (cufftComplex *)d_qtildeBin, (cufftComplex *)d_data, 1);
# 144 "Chisq_GPU.cu"
cufftDestroy(batchPlan);
# 147 "Chisq_GPU.cu"
unsigned numThreadsX = ((unsigned)512);
# 148 "Chisq_GPU.cu"
unsigned numBlocksX = (numPoints / numThreadsX);
# 150 "Chisq_GPU.cu"
dim3 grid(numBlocksX, 1, 1);
# 151 "Chisq_GPU.cu"
dim3 threads(numThreadsX, 1, 1);
# 155 "Chisq_GPU.cu"
cudaConfigureCall(grid, threads) ? ((void)0) : chisqKernel(d_chisq, d_q, d_data, numPoints, numChisqBins, chisqNorm);
# 157 "Chisq_GPU.cu"
cudaThreadSynchronize();
# 161 "Chisq_GPU.cu"
CudaError(cudaMemcpy(chisq, d_chisq, numPoints * sizeof(REAL4), cudaMemcpyDeviceToHost), "Chisq_GPU.cu", 161);
# 164 "Chisq_GPU.cu"
CudaError(cudaFree(d_q), "Chisq_GPU.cu", 164);
# 165 "Chisq_GPU.cu"
CudaError(cudaFree(d_chisq), "Chisq_GPU.cu", 165);
# 166 "Chisq_GPU.cu"
CudaError(cudaFree(d_data), "Chisq_GPU.cu", 166);
# 167 "Chisq_GPU.cu"
CudaError(cudaFree(d_qtildeBin), "Chisq_GPU.cu", 167);
# 169 "Chisq_GPU.cu"
}
# 1 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.stub.c"
# 1 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.stub.c" 1
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h" 1
# 91 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h"
extern "C" {

extern void** __cudaRegisterFatBinary(
  void *fatCubin
);

extern void __cudaUnregisterFatBinary(
  void **fatCubinHandle
);

extern void __cudaRegisterVar(
        void **fatCubinHandle,
        char *hostVar,
        char *deviceAddress,
  const char *deviceName,
        int ext,
        int size,
        int constant,
        int global
);

extern void __cudaRegisterTexture(
        void **fatCubinHandle,
  const struct textureReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int norm,
        int ext
);

extern void __cudaRegisterSurface(
        void **fatCubinHandle,
  const struct surfaceReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int ext
);

extern void __cudaRegisterFunction(
        void **fatCubinHandle,
  const char *hostFun,
        char *deviceFun,
  const char *deviceName,
        int thread_limit,
        uint3 *tid,
        uint3 *bid,
        dim3 *bDim,
        dim3 *gDim,
        int *wSize
);



extern int atexit(void(*)(void)) throw();







}

static void **__cudaFatCubinHandle;

static void __cudaUnregisterBinaryUtil(void)
{
  __cudaUnregisterFatBinary(__cudaFatCubinHandle);
}

# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/common_functions.h" 1
# 90 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/common_functions.h"
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 1 3
# 948 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_constants.h" 1 3
# 949 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 2 3
# 2973 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/func_macro.h" 1 3
# 2974 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 2 3
# 4683 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) double rsqrt(double a); double rsqrt(double a)
{
  return 1.0 / sqrt(a);
}

extern __attribute__((__weak__)) double rcbrt(double a); double rcbrt(double a)
{
  double s, t;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return 1.0 / a;
  }
  s = fabs(a);
  t = exp2(-3.3333333333333333e-1 * log2(s));
  t = ((t*t) * (-s*t) + 1.0) * (3.3333333333333333e-1*t) + t;



  if (__signbit(a))

  {
    t = -t;
  }
  return t;
}

extern __attribute__((__weak__)) double sinpi(double a); double sinpi(double a)
{
  int n;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return sin (a);
  }
  if (a == floor(a)) {
    return ((a / 1.0e308) / 1.0e308) / 1.0e308;
  }
  a = remquo (a, 0.5, &n);
  a = a * 3.1415926535897931e+0;
  if (n & 1) {
    a = cos (a);
  } else {
    a = sin (a);
  }
  if (n & 2) {
    a = -a;
  }
  return a;
}

extern __attribute__((__weak__)) double erfinv(double a); double erfinv(double a)
{
  double p, q, t, fa;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  fa = fabs(a);
  if (fa >= 1.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;
    if (fa == 1.0) {
      t = a * exp(1000.0);
    }
  } else if (fa >= 0.9375) {




    t = log1p(-fa);
    t = 1.0 / sqrt(-t);
    p = 2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q = t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
    if (a < 0.0) t = -t;
  } else if (fa >= 0.75) {




    t = a * a - .87890625;
    p = .21489185007307062000e+0;
    p = p * t - .64200071507209448655e+1;
    p = p * t + .29631331505876308123e+2;
    p = p * t - .47644367129787181803e+2;
    p = p * t + .34810057749357500873e+2;
    p = p * t - .12954198980646771502e+2;
    p = p * t + .25349389220714893917e+1;
    p = p * t - .24758242362823355486e+0;
    p = p * t + .94897362808681080020e-2;
    q = t - .12831383833953226499e+2;
    q = q * t + .41409991778428888716e+2;
    q = q * t - .53715373448862143349e+2;
    q = q * t + .33880176779595142685e+2;
    q = q * t - .11315360624238054876e+2;
    q = q * t + .20369295047216351160e+1;
    q = q * t - .18611650627372178511e+0;
    q = q * t + .67544512778850945940e-2;
    p = p / q;
    t = a * p;
  } else {




    t = a * a - .5625;
    p = - .23886240104308755900e+2;
    p = p * t + .45560204272689128170e+3;
    p = p * t - .22977467176607144887e+4;
    p = p * t + .46631433533434331287e+4;
    p = p * t - .43799652308386926161e+4;
    p = p * t + .19007153590528134753e+4;
    p = p * t - .30786872642313695280e+3;
    q = t - .83288327901936570000e+2;
    q = q * t + .92741319160935318800e+3;
    q = q * t - .35088976383877264098e+4;
    q = q * t + .59039348134843665626e+4;
    q = q * t - .48481635430048872102e+4;
    q = q * t + .18997769186453057810e+4;
    q = q * t - .28386514725366621129e+3;
    p = p / q;
    t = a * p;
  }
  return t;
}

extern __attribute__((__weak__)) double erfcinv(double a); double erfcinv(double a)
{
  double t;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  if (__isnan(a)) {
    return a + a;
  }
  if (a <= 0.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;
    if (a == 0.0) {
        t = (1.0 - a) * exp(1000.0);
    }
  }
  else if (a >= 0.0625) {
    t = erfinv (1.0 - a);
  }
  else if (a >= 1e-100) {




    double p, q;
    t = log(a);
    t = 1.0 / sqrt(-t);
    p = 2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q = t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
  }
  else {




    double p, q;
    t = log(a);
    t = 1.0 / sqrt(-t);
    p = 6.9952990607058154858e-1;
    p = p * t + 1.9507620287580568829e+0;
    p = p * t + 8.2810030904462690216e-1;
    p = p * t + 1.1279046353630280005e-1;
    p = p * t + 6.0537914739162189689e-3;
    p = p * t + 1.3714329569665128933e-4;
    p = p * t + 1.2964481560643197452e-6;
    p = p * t + 4.6156006321345332510e-9;
    p = p * t + 4.5344689563209398450e-12;
    q = t + 1.5771922386662040546e+0;
    q = q * t + 2.1238242087454993542e+0;
    q = q * t + 8.4001814918178042919e-1;
    q = q * t + 1.1311889334355782065e-1;
    q = q * t + 6.0574830550097140404e-3;
    q = q * t + 1.3715891988350205065e-4;
    q = q * t + 1.2964671850944981713e-6;
    q = q * t + 4.6156017600933592558e-9;
    q = q * t + 4.5344687377088206783e-12;
    t = p / (q * t);
  }
  return t;
}

extern __attribute__((__weak__)) float rsqrtf(float a); float rsqrtf(float a)
{
  return (float)rsqrt((double)a);
}

extern __attribute__((__weak__)) float rcbrtf(float a); float rcbrtf(float a)
{
  return (float)rcbrt((double)a);
}

extern __attribute__((__weak__)) float sinpif(float a); float sinpif(float a)
{
  return (float)sinpi((double)a);
}

extern __attribute__((__weak__)) float erfinvf(float a); float erfinvf(float a)
{
  return (float)erfinv((double)a);
}

extern __attribute__((__weak__)) float erfcinvf(float a); float erfcinvf(float a)
{
  return (float)erfcinv((double)a);
}







extern __attribute__((__weak__)) int min(int a, int b); int min(int a, int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umin(unsigned int a, unsigned int b); unsigned int umin(unsigned int a, unsigned int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) long long int llmin(long long int a, long long int b); long long int llmin(long long int a, long long int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) unsigned long long int ullmin(unsigned long long int a, unsigned long long int b); unsigned long long int ullmin(unsigned long long int a, unsigned long long int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) int max(int a, int b); int max(int a, int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umax(unsigned int a, unsigned int b); unsigned int umax(unsigned int a, unsigned int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) long long int llmax(long long int a, long long int b); long long int llmax(long long int a, long long int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) unsigned long long int ullmax(unsigned long long int a, unsigned long long int b); unsigned long long int ullmax(unsigned long long int a, unsigned long long int b)
{
  return a > b ? a : b;
}
# 5000 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 3
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions_dbl_ptx1.h" 1 3
# 5001 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/math_functions.h" 2 3
# 91 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/common_functions.h" 2
# 164 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/crt/host_runtime.h" 2







#pragma pack()
# 2 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.stub.c" 2
# 1 "/tmp/tmpxft_000076db_00000000-3_Chisq_GPU.fatbin.c" 1
# 1 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h" 1
# 83 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
extern "C" {
# 97 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;
# 113 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* ptx;
} __cudaFatPtxEntry;
# 125 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
typedef struct __cudaFatDebugEntryRec {
    char* gpuProfileName;
    char* debug;
    struct __cudaFatDebugEntryRec *next;
    unsigned int size;
} __cudaFatDebugEntry;

typedef struct __cudaFatElfEntryRec {
    char* gpuProfileName;
    char* elf;
    struct __cudaFatElfEntryRec *next;
    unsigned int size;
} __cudaFatElfEntry;

typedef enum {
      __cudaFatDontSearchFlag = (1 << 0),
      __cudaFatDontCacheFlag = (1 << 1),
      __cudaFatSassDebugFlag = (1 << 2)
} __cudaFatCudaBinaryFlag;
# 152 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* name;
} __cudaFatSymbol;
# 166 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int characteristic;
    __cudaFatElfEntry *elf;
} __cudaFatCudaBinary;
# 203 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
    typedef enum {
        __cudaFatAvoidPTX,
        __cudaFatPreferBestCode,
        __cudaFatForcePTX
    } __cudaFatCompilationPolicy;
# 227 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
void fatGetCubinForGpuWithPolicy( __cudaFatCudaBinary *binary, __cudaFatCompilationPolicy policy, char* gpuName, char* *cubin, char* *dbgInfoFile );
# 240 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
unsigned char fatCheckJitForGpuWithPolicy( __cudaFatCudaBinary *binary, __cudaFatCompilationPolicy policy, char* gpuName, char* *ptx );
# 250 "/usr/local/nvidia/sdk-3.2/cuda/bin/../include/__cudaFatFormat.h"
void fatFreeCubin( char* cubin, char* dbgInfoFile );





void __cudaFatFreePTX( char* ptx );


}
# 2 "/tmp/tmpxft_000076db_00000000-3_Chisq_GPU.fatbin.c" 2

asm(
".section .rodata\n"
".align 32\n"
"__deviceText_$sm_10$:\n"
".quad 0x33010102464c457f,0x0000000000000002,0x0000000100be0002,0x0000000000000000\n"
".quad 0x000000000000481c,0x0000000000000040,0x00380040000a010a,0x0001001700400003\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000600\n"
".quad 0x000000000000020a,0x0000000000000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x000000000000080a\n"
".quad 0x0000000000000181,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x000000000000098b\n"
".quad 0x0000000000000300,0x0000001f00000002,0x0000000000000001,0x0000000000000018\n"
".quad 0x0000000100000036,0x0000000000000006,0x0000000000000000,0x0000000000000c8b\n"
".quad 0x00000000000008f8,0x0c00000600000003,0x0000000000000004,0x0000000000000000\n"
".quad 0x00000001000000da,0x0000000000000002,0x0000000000000000,0x0000000000001583\n"
".quad 0x000000000000000c,0x0000000400000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x0000000100000062,0x0000000000000002,0x0000000000000000,0x000000000000158f\n"
".quad 0x0000000000000080,0x0000000400000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000008000000a9,0x0000000000000003,0x0000000000000000,0x000000000000160f\n"
".quad 0x0000000000000034,0x0000000400000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x00000001000001e6,0x0000000000000000,0x0000000000000000,0x000000000000160f\n"
".quad 0x000000000000003c,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000009000001f6,0x0000000000000000,0x0000000000000000,0x000000000000164b\n"
".quad 0x0000000000000010,0x0000000800000003,0x0000000000000004,0x0000000000000010\n"
".quad 0x0000000100000193,0x0000000000000000,0x0000000000000000,0x000000000000165b\n"
".quad 0x0000000000000134,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000009000001a7,0x0000000000000000,0x0000000000000000,0x000000000000178f\n"
".quad 0x0000000000000010,0x0000000a00000003,0x0000000000000004,0x0000000000000010\n"
".quad 0x0000000100000177,0x0000000000000000,0x0000000000000000,0x000000000000179f\n"
".quad 0x00000000000003e0,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000900000183,0x0000000000000000,0x0000000000000000,0x0000000000001b7f\n"
".quad 0x0000000000000010,0x0000000c00000003,0x0000000000000004,0x0000000000000010\n"
".quad 0x000000010000014d,0x0000000000000000,0x0000000000000000,0x0000000000001b8f\n"
".quad 0x00000000000003c5,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000900000160,0x0000000000000000,0x0000000000000000,0x0000000000001f54\n"
".quad 0x0000000000000010,0x0000000e00000003,0x0000000000000004,0x0000000000000010\n"
".quad 0x000000010000013b,0x0000000000000000,0x0000000000000000,0x0000000000001f64\n"
".quad 0x00000000000010fb,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000010000012d,0x0000000000000000,0x0000000000000000,0x000000000000305f\n"
".quad 0x0000000000000143,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000010000011a,0x0000000000000000,0x0000000000000000,0x00000000000031a2\n"
".quad 0x000000000000053c,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000009000001cf,0x0000000000000000,0x0000000000000000,0x00000000000036de\n"
".quad 0x0000000000000020,0x0000001200000003,0x0000000000000004,0x0000000000000010\n"
".quad 0x000000010000010e,0x0000000000000000,0x0000000000000000,0x00000000000036fe\n"
".quad 0x000000000000053c,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000009000001bf,0x0000000000000000,0x0000000000000000,0x0000000000003c3a\n"
".quad 0x00000000000000c0,0x0000001400000003,0x0000000000000004,0x0000000000000010\n"
".quad 0x0000000100000091,0x0000000000000000,0x0000000000000000,0x0000000000003cfa\n"
".quad 0x0000000000000b20,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x672e766e2e006261\n"
".quad 0x6e692e6c61626f6c,0x672e766e2e007469,0x742e006c61626f6c,0x31315a5f2e747865\n"
".quad 0x72654b7173696863,0x31315066506c656e,0x4c504d4f43676174,0x6a6a5f3153385845\n"
".quad 0x6e692e766e2e0066,0x6331315a5f2e6f66,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x6265645f766e2e00,0x5f6f666e695f6775\n"
".quad 0x737361735f676572,0x6168732e766e2e00,0x31315a5f2e646572,0x72654b7173696863\n"
".quad 0x31315066506c656e,0x4c504d4f43676174,0x6a6a5f3153385845,0x6f632e766e2e0066\n"
".quad 0x2e31746e6174736e,0x7369686331315a5f,0x506c656e72654b71,0x4367617431315066\n"
".quad 0x533858454c504d4f,0x642e00666a6a5f31,0x666e695f67756265,0x65645f766e2e006f\n"
".quad 0x6f666e695f677562,0x65642e007874705f,0x726262615f677562,0x645f766e2e007665\n"
".quad 0x7874705f67756265,0x766e2e007478745f,0x6c5f67756265645f,0x007874705f656e69\n"
".quad 0x5f766e2e6c65722e,0x696c5f6775626564,0x2e007874705f656e,0x696c5f6775626564\n"
".quad 0x2e6c65722e00656e,0x696c5f6775626564,0x645f766e2e00656e,0x6e696c5f67756265\n"
".quad 0x2e00737361735f65,0x645f766e2e6c6572,0x6e696c5f67756265,0x2e00737361735f65\n"
".quad 0x756265642e6c6572,0x2e006f666e695f67,0x645f766e2e6c6572,0x666e695f67756265\n"
".quad 0x642e007874705f6f,0x6275705f67756265,0x722e0073656d616e,0x67756265642e6c65\n"
".quad 0x656d616e6275705f,0x6331315a5f000073,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x70616475635f5f00,0x31315a5f5f6d7261\n"
".quad 0x72654b7173696863,0x31315066506c656e,0x4c504d4f43676174,0x6a6a5f3153385845\n"
".quad 0x4e71736968635f66,0x75635f5f006d726f,0x5f5f6d7261706164,0x717369686331315a\n"
".quad 0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d,0x6d756e5f666a6a5f\n"
".quad 0x6e69427173696843,0x616475635f5f0073,0x315a5f5f6d726170,0x654b717369686331\n"
".quad 0x315066506c656e72,0x504d4f4367617431,0x6a5f31533858454c,0x6968635f675f666a\n"
".quad 0x6475635f5f007173,0x5a5f5f6d72617061,0x4b71736968633131,0x5066506c656e7265\n"
".quad 0x4d4f436761743131,0x5f31533858454c50,0x00715f675f666a6a,0x6170616475635f5f\n"
".quad 0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761\n"
".quad 0x666a6a5f31533858,0x6e696f506d756e5f,0x6475635f5f007374,0x5a5f5f6d72617061\n"
".quad 0x4b71736968633131,0x5066506c656e7265,0x4d4f436761743131,0x5f31533858454c50\n"
".quad 0x61645f675f666a6a,0x0000000000006174,0x0000000000000000,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000000100,0x0000000000000000,0x0300000000000000\n"
".quad 0x0000000000000200,0x0000000000000000,0x0300000000000000,0x0000000000000300\n"
".quad 0x0000000000000000,0x0300000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000000000,0x0000000000000000,0x0300000000000000\n"
".quad 0x0000000000000400,0x00000008f8000000,0x0300000000000000,0x0000000000000600\n"
".quad 0x0000000000000000,0x0300000000000000,0x0000000000001600,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000000700,0x0000000000000000,0x0100000027000000\n"
".quad 0x0000000030000700,0x0000000004000000,0x0100000062000000,0x000000002c000700\n"
".quad 0x0000000004000000,0x01000000a0000000,0x0000000010000700,0x0000000008000000\n"
".quad 0x01000000d9000000,0x0000000018000700,0x0000000008000000,0x010000010e000000\n"
".quad 0x0000000028000700,0x0000000004000000,0x0100000149000000,0x0000000020000700\n"
".quad 0x0000000008000000,0x0300000000000000,0x0000000000000500,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000001400,0x0000000000000000,0x0300000000000000\n"
".quad 0x0000000000001200,0x0000000000000000,0x0300000000000000,0x0000000000001100\n"
".quad 0x0000000000000000,0x0300000000000000,0x0000000000001000,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000000e00,0x0000000000000000,0x0300000000000000\n"
".quad 0x0000000000000f00,0x0000000000000000,0x0300000000000000,0x0000000000000c00\n"
".quad 0x0000000000000000,0x0300000000000000,0x0000000000000d00,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000000a00,0x0000000000000000,0x0300000000000000\n"
".quad 0x0000000000000b00,0x0000000000000000,0x0300000000000000,0x0000000000001500\n"
".quad 0x0000000000000000,0x0300000000000000,0x0000000000001300,0x0000000000000000\n"
".quad 0x0300000000000000,0x0000000000000800,0x0000000000000000,0x0300000000000000\n"
".quad 0x0000000000000900,0x0000000000000000,0x1200000001000000,0x0000000000000410\n"
".quad 0x0000000000000000,0x041000be04000000,0x03102c8009100082,0x8000000405000000\n"
".quad 0x801400c009c00007,0x80100002050423c7,0x80300104090403c7,0xc8a00005fd6400c7\n"
".quad 0x80100002050c0147,0x80101190030403c7,0x8010000011000002,0x80a000080d0003c7\n"
".quad 0x8010004c11040007,0x80100008110023c7,0x80a00008110003c7,0x8010004211040007\n"
".quad 0x80100008110023c7,0x80a00008090003c7,0x1440051014040007,0x8030100a15600412\n"
".quad 0x8060041009c41007,0x0820028608000147,0x031020800d100084,0x8000000605000000\n"
".quad 0x801400c019c00007,0x801400c21d0423c7,0x1810008c140423c7,0x801000040910008e\n"
".quad 0x031028800d0403c7,0x8000000605000000,0x801400c00dc00007,0x10100082040423c7\n"
".quad 0x8060020e1140030c,0x8030100811000107,0x8060020c0dc41007,0x0c2003841c000107\n"
".quad 0x0360081e0d403f1c,0x8030100611000000,0x0360081c11c41007,0x80403f1c0d000000\n"
".quad 0x8060801e21000007,0x80301010250040c7,0x8030101021c41007,0x0320009029e41007\n"
".quad 0xc860801ffd000010,0x801000100d0040c7,0x801000140d0403c7,0xc860801dfd0403c8\n"
".quad 0x20707f1e0c004247,0x0c10008624100088,0x101000921c100090,0xc820000bfd20038a\n"
".quad 0x0c30478c140400c7,0xc8200009fd201f88,0x0c305f8a100407c7,0x80301f060d100086\n"
".quad 0x802000060dc40007,0x80d00e060d0407c7,0x801000060d80c007,0x03102080110403c7\n"
".quad 0x8000000805000000,0x801400c011c00007,0x801400c2150423c7,0x1c100088180423c7\n"
".quad 0x801000040910008a,0x03102880110403c7,0x8000000805000000,0x801400c011c00007\n"
".quad 0x14100082040423c7,0x8060021215400310,0x8030100a15000147,0x8060021011c41007\n"
".quad 0x1020048420000147,0x0360082211403f20,0x8030100815000000,0x0360082015c41007\n"
".quad 0x80403f2011000000,0x8060802229000007,0x8030101425004107,0x803010142dc41007\n"
".quad 0x0320009629e41007,0xc8608023fd000010,0x8010001611004107,0x80100014110403c7\n"
".quad 0xc8608021fd0403c8,0x80603f2211004247,0x80d0040a150c0107,0x80d0040a11040087\n"
".quad 0x80d0040a15040087,0x1010008820040087,0x8020000c1510008a,0xc820000dfd040207\n"
".quad 0x8030400e19040207,0x0320048a11040107,0xc821000bfd000000,0x10305f8c14044047\n"
".quad 0x80301f0811100088,0x8020000811c40007,0x80d00e08110407c7,0x801000081180c007\n"
".quad 0x03101880150403c7,0x8000000a05000000,0x801400c021c00007,0x801400c2250423c7\n"
".quad 0x201000901c0423c7,0x80403f0815100092,0x0360080a15000007,0x8030100a19000000\n"
".quad 0x0360080819c41007,0x80403f0815000000,0x8060800a29000007,0x8030101425004147\n"
".quad 0x803010142dc41007,0x0320009629e41007,0xc860800bfd000010,0x8010001615004147\n"
".quad 0x80100014150403c7,0xc8608009fd0403c8,0x28707f0a14004247,0x1410008a2c10008c\n"
".quad 0x1810009624100094,0xc820000ffd20058e,0x143049901c040147,0xc820000dfd201f8c\n"
".quad 0x14305f8e180407c7,0x80301f0a1510008a,0x8020000a15c40007,0x80d00e0a150407c7\n"
".quad 0x031030801980c007,0x8000000c05000000,0x801400c019c00007,0x18c0060a180423c7\n"
".quad 0x03102c801510008c,0x8000000a05000000,0x801400c015c00007,0x80a0000a150423c7\n"
".quad 0x0810008a1c440047,0x0c10008614100084,0x1c10008c18100088,0x1c10008c1010008e\n"
".quad 0x80a0000e1910008e,0x80b0820c19c41147,0xc8a0000dfd604107,0x1c100088100c0147\n"
".quad 0x001009300310008e,0x03c0000811000001,0x03c0000e1d03e800,0x1c1000881003e800\n"
".quad 0x1090000e1810008e,0x8010000811c00608,0x03103080190403c7,0x8000000c05000000\n"
".quad 0x801400c019c00007,0x1410008a140423c7,0x10b0440a10c0050c,0x0310188015100088\n"
".quad 0x8000000a05000000,0x801400c021c00007,0x801400c2250423c7,0x201000901c0423c7\n"
".quad 0x80403f0815100092,0x0360080a15000007,0x8030100a19000000,0x0360080819c41007\n"
".quad 0x80403f0815000000,0x8060800a29000007,0x8030101425004147,0x803010142dc41007\n"
".quad 0x0320009629e41007,0xc860800bfd000010,0x8010001615004147,0x80100014150403c7\n"
".quad 0xc8608009fd0403c8,0x28707f0a14004247,0x1410008a2c10008c,0x1810009624100094\n"
".quad 0xc820000ffd20058e,0x803040101d040147,0x0320048c15040247,0xc821000dfd000000\n"
".quad 0x14305f8e18044047,0x80301f0a1510008a,0x8020000a15c40007,0x80d00e0a150407c7\n"
".quad 0x031030801980c007,0x8000000c05000000,0x801400c019c00007,0x14c0060a140423c7\n"
".quad 0x03102c801910008a,0x8000000c05000000,0x801400c019c00007,0x80a0000c190423c7\n"
".quad 0x1010008c18440047,0x1810008a14100088,0x1810008a1410008c,0x80a0000c1d10008c\n"
".quad 0x80b0820e1dc41147,0xc8a0000ffd604107,0x1810008a140c0147,0x00100ca00310008c\n"
".quad 0x03c0000a15000001,0x03c0000c1903e800,0x1810008a1403e800,0x1490000c1810008c\n"
".quad 0x8010000a15c0060a,0x03103080190403c7,0x8000000c05000000,0x801400c019c00007\n"
".quad 0x0c1000860c0423c7,0x0cb045060cc0030c,0x0310108015100086,0x8000000a05000000\n"
".quad 0x801400c021c00007,0x801400c2250423c7,0x201000901c0423c7,0x80403f0815100092\n"
".quad 0x0360040a15000007,0x8030100a19000000,0x0360040819c41007,0x80403f0815000000\n"
".quad 0x8060820a29000007,0x8030101425004147,0x803010142dc41007,0x0320009629e41007\n"
".quad 0xc860820bfd000010,0x8010001615004147,0x80100014150403c7,0xc8608209fd0403c8\n"
".quad 0x28707f0a14004247,0x1410008a2c10008c,0x1810009624100094,0xc820000ffd20058e\n"
".quad 0x143049901c040147,0xc820000dfd201f8c,0x14305f8e180407c7,0x80301f0a1510008a\n"
".quad 0x8020000a15c40007,0x80d00e0a190407c7,0x141000860c80c007,0x10c0050614100086\n"
".quad 0x801000080d100088,0x80e003080d0403c7,0x80b0000c15000147,0x031010800d0000c7\n"
".quad 0x8000000605000000,0x801400c021c00007,0x801400c2250423c7,0x1c100090180423c7\n"
".quad 0x80403f080d100092,0x0360040a0d000007,0x8030100611000000,0x0360040811c41007\n"
".quad 0x80403f080d000000,0x8060820a21000007,0x80301010250040c7,0x8030101021c41007\n"
".quad 0x0320009029e41007,0xc860820bfd000010,0x801000100d0040c7,0x801000140d0403c7\n"
".quad 0xc8608209fd0403c8,0x80603f0a09004247,0x081000840c0c00c7,0x0810008410100088\n"
".quad 0x8020000c0d100086,0xc820000dfd040107,0x0830428e10040107,0xc8200007fd201f86\n"
".quad 0x08305f880c0407c7,0x80301f0409100084,0x8020000409c40007,0x80d00e04150407c7\n"
".quad 0x8010000205a0c007,0x03200182050403c7,0x0410008204000000,0x03102c8009100082\n"
".quad 0x8000000405000000,0x801400c009c00007,0x80100002050423c7,0x80300104090403c7\n"
".quad 0xc8a00005fd640107,0x80100090030c0147,0x80861ffe03000002,0x00f0000001000007\n"
".quad 0x00f0000001e00000,0x00f0000001e00000,0x8030000003e00000,0x01f0000001000007\n"
".quad 0x0400000008e00000,0x047e800000000000,0x080000000000180b,0x1800000010000000\n"
".quad 0x200000001c000000,0x0400241803000000,0x05ffffffff000c17,0x040013f000002000\n"
".quad 0x04ffffffff000c17,0x040013f000001c00,0x03ffffffff000c17,0x040013f000001800\n"
".quad 0x02ffffffff000c17,0x040023f000001000,0x01ffffffff000c17,0x040023f000000800\n"
".quad 0x00ffffffff000c17,0x380023f000000000,0x0000000002000000,0x00032c0000053c00\n"
".quad 0x69686331315a5f00,0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43\n"
".quad 0x0000666a6a5f3153,0x0000000006000000,0x1100000001000000,0x0200000130000000\n"
".quad 0xfb01010000001000,0x0001010101000a0e,0x0209000000010000,0x0000000000000000\n"
".quad 0x48480100c2030004,0xf3f0800118020103,0x01030118020103f0,0x0118020103011802\n"
".quad 0x8001280201034948,0x0103480118020103,0xf802010349011c02,0x012c020103f00100\n"
".quad 0x0380012802010381,0x0201034801180201,0x018c02010349011c,0x0103011802010301\n"
".quad 0x2802020381013002,0xf00100fc02010301,0x020103012c020103,0x1802010348480118\n"
".quad 0x4801180203038001,0x030100d002030348,0x4948484801180201,0x0201030128020203\n"
".quad 0x01140201030100fc,0x0201030130020103,0x1802010348480118,0x0203034848f28001\n"
".quad 0x01180201030100d0,0x2802020349484848,0xf00100fc02010301,0x484848012c020103\n"
".quad 0x2802020380808048,0x0301018002010301,0x2c02030301140201,0x1802010381808001\n"
".quad 0x2002020383f08001,0x1d01010001080201,0x0200000000000000,0xdc0000001f000000\n"
".quad 0x0003690002000003,0x01000a0efb010100,0x2f01000000010101,0x6d6f682f00706d74\n"
".quad 0x73656977616b2f65,0x636e692f7665642f,0x6c616c2f6564756c,0x696c2f7273752f00\n"
".quad 0x38782f6363672f62,0x6e696c2d34365f36,0x342f756e672d7875,0x636e692f352e342e\n"
".quad 0x73752f006564756c,0x2f6c61636f6c2f72,0x732f61696469766e,0x632f322e332d6b64\n"
".quad 0x2f6e69622f616475,0x756c636e692f2e2e,0x2f007472632f6564,0x61636f6c2f727375\n"
".quad 0x61696469766e2f6c,0x322e332d6b64732f,0x69622f616475632f,0x636e692f2e2e2f6e\n"
".quad 0x73752f006564756c,0x64756c636e692f72,0x2f00737469622f65,0x6c636e692f727375\n"
".quad 0x6f633c0000656475,0x696c2d646e616d6d,0x74000000003e656e,0x30305f746678706d\n"
".quad 0x305f626436373030,0x2d30303030303030,0x5f71736968435f36,0x616475632e555047\n"
".quad 0x007570672e326566,0x74414c414c000001,0x6174614463696d6f,0x00682e7365707974\n"
".quad 0x6564647473000002,0x6400000300682e66,0x75725f6563697665,0x00682e656d69746e\n"
".quad 0x5f74736f68000004,0x2e73656e69666564,0x6975620000050068,0x7079745f6e69746c\n"
".quad 0x00000500682e7365,0x745f656369766564,0x0500682e73657079,0x7265766972640000\n"
".quad 0x682e73657079745f,0x6672757300000500,0x657079745f656361,0x7400000500682e73\n"
".quad 0x745f657275747865,0x0500682e73657079,0x726f746365760000,0x682e73657079745f\n"
".quad 0x6976656400000500,0x636e75616c5f6563,0x656d617261705f68,0x0500682e73726574\n"
".quad 0x6761726f74730000,0x2e7373616c635f65,0x7079740000040068,0x00000600682e7365\n"
".quad 0x0700682e656d6974,0x746e696474730000,0x6f6300000700682e,0x6e75665f6e6f6d6d\n"
".quad 0x682e736e6f697463,0x6874616d00000500,0x6f6974636e75665f,0x00000500682e736e\n"
".quad 0x6e6f635f6874616d,0x682e73746e617473,0x6976656400000500,0x74636e75665f6563\n"
".quad 0x0500682e736e6f69,0x5f31315f6d730000,0x665f63696d6f7461,0x736e6f6974636e75\n"
".quad 0x6d7300000500682e,0x6d6f74615f32315f,0x74636e75665f6369,0x0500682e736e6f69\n"
".quad 0x5f33315f6d730000,0x665f656c62756f64,0x736e6f6974636e75,0x6d7300000500682e\n"
".quad 0x6d6f74615f30325f,0x74636e75665f6369,0x0500682e736e6f69,0x5f30325f6d730000\n"
".quad 0x69736e6972746e69,0x00000500682e7363,0x5f65636166727573,0x6e6f6974636e7566\n"
".quad 0x7400000500682e73,0x665f657275747865,0x6e75665f68637465,0x682e736e6f697463\n"
".quad 0x6874616d00000500,0x6f6974636e75665f,0x705f6c62645f736e,0x000500682e317874\n"
".quad 0x475f717369684300,0x00000075632e5550,0x0000000209000000,0x031e040000000000\n"
".quad 0x0100c8020203013b,0x01030100e0020203,0xc80202030102a802,0x0203d00315040102\n"
".quad 0x7cb0031e040102b8,0x280202030100d802,0xb80203ce03150401,0x027cb2031e040102\n"
".quad 0x01280203030100d8,0x10030103f0027403,0x200202030100d002,0x7601010001080201\n"
".quad 0x0200000000000003,0xc10000001f000000,0x0003690002000003,0x01000a0efb010100\n"
".quad 0x2f01000000010101,0x6d6f682f00706d74,0x73656977616b2f65,0x636e692f7665642f\n"
".quad 0x6c616c2f6564756c,0x696c2f7273752f00,0x38782f6363672f62,0x6e696c2d34365f36\n"
".quad 0x342f756e672d7875,0x636e692f352e342e,0x73752f006564756c,0x2f6c61636f6c2f72\n"
".quad 0x732f61696469766e,0x632f322e332d6b64,0x2f6e69622f616475,0x756c636e692f2e2e\n"
".quad 0x2f007472632f6564,0x61636f6c2f727375,0x61696469766e2f6c,0x322e332d6b64732f\n"
".quad 0x69622f616475632f,0x636e692f2e2e2f6e,0x73752f006564756c,0x64756c636e692f72\n"
".quad 0x2f00737469622f65,0x6c636e692f727375,0x6f633c0000656475,0x696c2d646e616d6d\n"
".quad 0x74000000003e656e,0x30305f746678706d,0x305f626436373030,0x2d30303030303030\n"
".quad 0x5f71736968435f36,0x616475632e555047,0x007570672e326566,0x74414c414c000001\n"
".quad 0x6174614463696d6f,0x00682e7365707974,0x6564647473000002,0x6400000300682e66\n"
".quad 0x75725f6563697665,0x00682e656d69746e,0x5f74736f68000004,0x2e73656e69666564\n"
".quad 0x6975620000050068,0x7079745f6e69746c,0x00000500682e7365,0x745f656369766564\n"
".quad 0x0500682e73657079,0x7265766972640000,0x682e73657079745f,0x6672757300000500\n"
".quad 0x657079745f656361,0x7400000500682e73,0x745f657275747865,0x0500682e73657079\n"
".quad 0x726f746365760000,0x682e73657079745f,0x6976656400000500,0x636e75616c5f6563\n"
".quad 0x656d617261705f68,0x0500682e73726574,0x6761726f74730000,0x2e7373616c635f65\n"
".quad 0x7079740000040068,0x00000600682e7365,0x0700682e656d6974,0x746e696474730000\n"
".quad 0x6f6300000700682e,0x6e75665f6e6f6d6d,0x682e736e6f697463,0x6874616d00000500\n"
".quad 0x6f6974636e75665f,0x00000500682e736e,0x6e6f635f6874616d,0x682e73746e617473\n"
".quad 0x6976656400000500,0x74636e75665f6563,0x0500682e736e6f69,0x5f31315f6d730000\n"
".quad 0x665f63696d6f7461,0x736e6f6974636e75,0x6d7300000500682e,0x6d6f74615f32315f\n"
".quad 0x74636e75665f6369,0x0500682e736e6f69,0x5f33315f6d730000,0x665f656c62756f64\n"
".quad 0x736e6f6974636e75,0x6d7300000500682e,0x6d6f74615f30325f,0x74636e75665f6369\n"
".quad 0x0500682e736e6f69,0x5f30325f6d730000,0x69736e6972746e69,0x00000500682e7363\n"
".quad 0x5f65636166727573,0x6e6f6974636e7566,0x7400000500682e73,0x665f657275747865\n"
".quad 0x6e75665f68637465,0x682e736e6f697463,0x6874616d00000500,0x6f6974636e75665f\n"
".quad 0x705f6c62645f736e,0x000500682e317874,0x475f717369684300,0x00000075632e5550\n"
".quad 0x0000000209000000,0x001e040000000000,0x0000000000430209,0xb8738f013b030000\n"
".quad 0x0d0203d0031504b9,0x05027cb0031e0401,0x0203ce0315046501,0x027cb2031e04010d\n"
".quad 0x0113027403660105,0x01022d010b021003,0x0000037601010001,0x0000000200000000\n"
".quad 0x7265762e0000001f,0x342e31206e6f6973,0x7465677261742e00,0x202c30315f6d7320\n"
".quad 0x5f3436665f70616d,0x00003233665f6f74,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x2e00000000000000\n"
".quad 0x5a5f207972746e65,0x4b71736968633131,0x5066506c656e7265,0x4d4f436761743131\n"
".quad 0x5f31533858454c50,0x702e002820666a6a,0x36752e206d617261,0x616475635f5f2034\n"
".quad 0x315a5f5f6d726170,0x654b717369686331,0x315066506c656e72,0x504d4f4367617431\n"
".quad 0x6a5f31533858454c,0x6968635f675f666a,0x7261702e002c7173,0x203436752e206d61\n"
".quad 0x6170616475635f5f,0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x702e002c715f675f,0x36752e206d617261\n"
".quad 0x616475635f5f2034,0x315a5f5f6d726170,0x654b717369686331,0x315066506c656e72\n"
".quad 0x504d4f4367617431,0x6a5f31533858454c,0x7461645f675f666a,0x617261702e002c61\n"
".quad 0x5f203233752e206d,0x726170616475635f,0x686331315a5f5f6d,0x656e72654b717369\n"
".quad 0x617431315066506c,0x58454c504d4f4367,0x5f666a6a5f315338,0x746e696f506d756e\n"
".quad 0x617261702e002c73,0x5f203233752e206d,0x726170616475635f,0x686331315a5f5f6d\n"
".quad 0x656e72654b717369,0x617431315066506c,0x58454c504d4f4367,0x5f666a6a5f315338\n"
".quad 0x71736968436d756e,0x702e002c736e6942,0x33662e206d617261,0x616475635f5f2032\n"
".quad 0x315a5f5f6d726170,0x654b717369686331,0x315066506c656e72,0x504d4f4367617431\n"
".quad 0x6a5f31533858454c,0x71736968635f666a,0x007b00296d726f4e,0x33752e206765722e\n"
".quad 0x3e38323c72252032,0x2e206765722e003b,0x3c64722520343675,0x65722e003b3e3632\n"
".quad 0x25203233662e2067,0x2e003b3e32343c66,0x6572702e20676572,0x3b3e343c70252064\n"
".quad 0x656257444c240000,0x31315a5f5f6e6967,0x72654b7173696863,0x31315066506c656e\n"
".quad 0x4c504d4f43676174,0x6a6a5f3153385845,0x6257444c24003a66,0x636f6c626e696765\n"
".quad 0x3a315f3330325f6b,0x33752e766f6d0000,0x202c317225092032,0x732e766f6d003b30\n"
".quad 0x2c32722509203233,0x646c003b31722520,0x752e6d617261702e,0x2c33722509203233\n"
".quad 0x616475635f5f5b20,0x315a5f5f6d726170,0x654b717369686331,0x315066506c656e72\n"
".quad 0x504d4f4367617431,0x6a5f31533858454c,0x68436d756e5f666a,0x5d736e6942717369\n"
".quad 0x33732e766f6d003b,0x202c347225092032,0x746573003b327225,0x3233752e656c2e70\n"
".quad 0x25202c3170250920,0x3b347225202c3372,0x7262203170254000,0x5f305f4c24092061\n"
".quad 0x4c24003b30333333,0x3a343730335f305f,0x67656257444c2400,0x5f6b636f6c626e69\n"
".quad 0x00003a335f333032,0x2e3233752e747663,0x3572250920363175,0x782e64697425202c\n"
".quad 0x33752e747663003b,0x2509203631752e32,0x61746325202c3672,0x7663003b782e6469\n"
".quad 0x31752e3233752e74,0x202c377225092036,0x3b782e6469746e25,0x2e6f6c2e6c756d00\n"
".quad 0x3872250920323375,0x25202c367225202c,0x2e646461003b3772,0x3972250920323375\n"
".quad 0x25202c357225202c,0x2e766f6d003b3872,0x3172250920323373,0x003b397225202c30\n"
".quad 0x617261702e646c00,0x2509203436752e6d,0x5f5f5b202c316472,0x6d72617061647563\n"
".quad 0x69686331315a5f5f,0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43\n"
".quad 0x675f666a6a5f3153,0x003b5d617461645f,0x203233732e766f6d,0x25202c3131722509\n"
".quad 0x2e646c003b303172,0x33752e6d61726170,0x2c32317225092032,0x616475635f5f5b20\n"
".quad 0x315a5f5f6d726170,0x654b717369686331,0x315066506c656e72,0x504d4f4367617431\n"
".quad 0x6a5f31533858454c,0x6f506d756e5f666a,0x6d003b5d73746e69,0x09203233732e766f\n"
".quad 0x7225202c33317225,0x6c2e6c756d003b32,0x2509203233752e6f,0x317225202c343172\n"
".quad 0x3b33317225202c32,0x3233752e64646100,0x202c353172250920,0x7225202c31317225\n"
".quad 0x2e747663003b3431,0x203233752e343675,0x25202c3264722509,0x6c756d003b353172\n"
".quad 0x33752e656469772e,0x2c33647225092032,0x38202c3531722520,0x36752e646461003b\n"
".quad 0x2c34647225092034,0x25202c3164722520,0x2e646c003b336472,0x662e6c61626f6c67\n"
".quad 0x2c31662509203233,0x302b346472255b20,0x662e766f6d003b5d,0x2c32662509203233\n"
".quad 0x6c00003b31662520,0x2e6d617261702e64,0x6472250920343675,0x75635f5f5b202c35\n"
".quad 0x5f5f6d7261706164,0x717369686331315a,0x66506c656e72654b,0x4f43676174313150\n"
".quad 0x31533858454c504d,0x645f675f666a6a5f,0x6f6d003b5d617461,0x2509203233732e76\n"
".quad 0x317225202c363172,0x61702e646c003b30,0x203233752e6d6172,0x5b202c3731722509\n"
".quad 0x6170616475635f5f,0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x6e696f506d756e5f,0x766f6d003b5d7374\n"
".quad 0x722509203233732e,0x3b327225202c3831,0x2e6f6c2e6c756d00,0x3172250920323375\n"
".quad 0x2c37317225202c39,0x61003b3831722520,0x09203233752e6464,0x7225202c30327225\n"
".quad 0x39317225202c3631,0x36752e747663003b,0x2509203233752e34,0x327225202c366472\n"
".quad 0x772e6c756d003b30,0x203233752e656469,0x25202c3764722509,0x003b38202c303272\n"
".quad 0x203436752e646461,0x25202c3864722509,0x647225202c356472,0x6c672e646c003b37\n"
".quad 0x3233662e6c61626f,0x5b202c3366250920,0x3b5d342b38647225,0x3233662e766f6d00\n"
".quad 0x25202c3466250920,0x2e646c00003b3366,0x36752e6d61726170,0x2c39647225092034\n"
".quad 0x616475635f5f5b20,0x315a5f5f6d726170,0x654b717369686331,0x315066506c656e72\n"
".quad 0x504d4f4367617431,0x6a5f31533858454c,0x3b5d715f675f666a,0x3436752e74766300\n"
".quad 0x722509203233752e,0x317225202c303164,0x772e6c756d003b30,0x203233752e656469\n"
".quad 0x202c313164722509,0x3b38202c30317225,0x3436752e64646100,0x2c32316472250920\n"
".quad 0x25202c3964722520,0x646c003b31316472,0x2e6c61626f6c672e,0x3566250920323366\n"
".quad 0x32316472255b202c,0x2e646c003b5d302b,0x33662e6d61726170,0x202c366625092032\n"
".quad 0x70616475635f5f5b,0x31315a5f5f6d7261,0x72654b7173696863,0x31315066506c656e\n"
".quad 0x4c504d4f43676174,0x6a6a5f3153385845,0x4e71736968635f66,0x756d003b5d6d726f\n"
".quad 0x2509203233662e6c,0x2c356625202c3766,0x6f6d003b36662520,0x2509203233662e76\n"
".quad 0x3b376625202c3866,0x617261702e646c00,0x2509203233752e6d,0x5f5f5b202c313272\n"
".quad 0x6d72617061647563,0x69686331315a5f5f,0x6c656e72654b7173,0x6761743131506650\n"
".quad 0x3858454c504d4f43,0x6e5f666a6a5f3153,0x4271736968436d75,0x7663003b5d736e69\n"
".quad 0x3233662e6e722e74,0x662509203233752e,0x3b31327225202c39,0x3233662e766f6d00\n"
".quad 0x202c303166250920,0x444c24003b396625,0x5f696e6967656257,0x6665646976696466\n"
".quad 0x003a395f3330325f,0x3233662e766f6d00,0x202c313166250920,0x766f6d003b386625\n"
".quad 0x662509203233662e,0x30316625202c3231,0x75662e766964003b,0x09203233662e6c6c\n"
".quad 0x6625202c33316625,0x32316625202c3131,0x6e6557444c24003b,0x69766964665f6964\n"
".quad 0x5f3330325f666564,0x702e646c00003a38,0x3233662e6d617261,0x202c343166250920\n"
".quad 0x70616475635f5f5b,0x31315a5f5f6d7261,0x72654b7173696863,0x31315066506c656e\n"
".quad 0x4c504d4f43676174,0x6a6a5f3153385845,0x4e71736968635f66,0x6f6d003b5d6d726f\n"
".quad 0x2509203233662e76,0x326625202c353166,0x33662e6c756d003b,0x2c36316625092032\n"
".quad 0x25202c3431662520,0x627573003b353166,0x662509203233662e,0x36316625202c3731\n"
".quad 0x003b33316625202c,0x203233662e766f6d,0x25202c3831662509,0x646c00003b373166\n"
".quad 0x752e6d617261702e,0x3164722509203436,0x75635f5f5b202c33,0x5f5f6d7261706164\n"
".quad 0x717369686331315a,0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d\n"
".quad 0x715f675f666a6a5f,0x752e747663003b5d,0x09203233752e3436,0x25202c3431647225\n"
".quad 0x6c756d003b303172,0x33752e656469772e,0x3531647225092032,0x202c30317225202c\n"
".quad 0x752e646461003b38,0x3164722509203436,0x3331647225202c36,0x3b3531647225202c\n"
".quad 0x626f6c672e646c00,0x09203233662e6c61,0x255b202c39316625,0x3b5d342b36316472\n"
".quad 0x617261702e646c00,0x2509203233662e6d,0x5f5f5b202c303266,0x6d72617061647563\n"
".quad 0x69686331315a5f5f,0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43\n"
".quad 0x635f666a6a5f3153,0x6d726f4e71736968,0x662e6c756d003b5d,0x3132662509203233\n"
".quad 0x202c39316625202c,0x6f6d003b30326625,0x2509203233662e76,0x326625202c323266\n"
".quad 0x61702e646c003b31,0x203233752e6d6172,0x5b202c3232722509,0x6170616475635f5f\n"
".quad 0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761\n"
".quad 0x666a6a5f31533858,0x736968436d756e5f,0x003b5d736e694271,0x662e6e722e747663\n"
".quad 0x09203233752e3233,0x7225202c33326625,0x2e766f6d003b3232,0x3266250920323366\n"
".quad 0x3b33326625202c34,0x67656257444c2400,0x766964665f696e69,0x3330325f66656469\n"
".quad 0x766f6d00003a375f,0x662509203233662e,0x32326625202c3532,0x33662e766f6d003b\n"
".quad 0x2c36326625092032,0x64003b3432662520,0x2e6c6c75662e7669,0x3266250920323366\n"
".quad 0x2c35326625202c37,0x24003b3632662520,0x5f69646e6557444c,0x6665646976696466\n"
".quad 0x003a365f3330325f,0x617261702e646c00,0x2509203233662e6d,0x5f5f5b202c383266\n"
".quad 0x6d72617061647563,0x69686331315a5f5f,0x6c656e72654b7173,0x6761743131506650\n"
".quad 0x3858454c504d4f43,0x635f666a6a5f3153,0x6d726f4e71736968,0x662e766f6d003b5d\n"
".quad 0x3932662509203233,0x6d003b346625202c,0x09203233662e6c75,0x6625202c30336625\n"
".quad 0x39326625202c3832,0x33662e627573003b,0x2c31336625092032,0x25202c3033662520\n"
".quad 0x766f6d003b373266,0x662509203233662e,0x31336625202c3233,0x61702e646c00003b\n"
".quad 0x203436752e6d6172,0x202c373164722509,0x70616475635f5f5b,0x31315a5f5f6d7261\n"
".quad 0x72654b7173696863,0x31315066506c656e,0x4c504d4f43676174,0x6a6a5f3153385845\n"
".quad 0x736968635f675f66,0x2e747663003b5d71,0x203233752e343675,0x202c383164722509\n"
".quad 0x756d003b30317225,0x752e656469772e6c,0x3164722509203233,0x2c30317225202c39\n"
".quad 0x2e646461003b3420,0x6472250920343675,0x31647225202c3032,0x3931647225202c37\n"
".quad 0x6f6c672e646c003b,0x203233662e6c6162,0x5b202c3333662509,0x5d302b3032647225\n"
".quad 0x33662e766f6d003b,0x2c34336625092032,0x6d003b3233662520,0x09203233662e766f\n"
".quad 0x6625202c35336625,0x2e6c756d003b3233,0x3366250920323366,0x2c34336625202c36\n"
".quad 0x6d003b3533662520,0x09203233662e766f,0x6625202c37336625,0x2e766f6d003b3831\n"
".quad 0x3366250920323366,0x3b38316625202c38,0x3233662e64616d00,0x202c393366250920\n"
".quad 0x6625202c37336625,0x36336625202c3833,0x33662e646461003b,0x2c30346625092032\n"
".quad 0x25202c3333662520,0x2e646c003b393366,0x36752e6d61726170,0x3132647225092034\n"
".quad 0x6475635f5f5b202c,0x5a5f5f6d72617061,0x4b71736968633131,0x5066506c656e7265\n"
".quad 0x4d4f436761743131,0x5f31533858454c50,0x68635f675f666a6a,0x7663003b5d717369\n"
".quad 0x33752e3436752e74,0x3232647225092032,0x003b30317225202c,0x656469772e6c756d\n"
".quad 0x722509203233752e,0x317225202c333264,0x6461003b34202c30,0x2509203436752e64\n"
".quad 0x7225202c34326472,0x647225202c313264,0x672e7473003b3332,0x33662e6c61626f6c\n"
".quad 0x326472255b092032,0x6625202c5d302b34,0x57444c24003b3034,0x6b636f6c62646e65\n"
".quad 0x003a335f3330325f,0x3233732e766f6d00,0x202c333272250920,0x646461003b327225\n"
".quad 0x722509203233752e,0x33327225202c3432,0x766f6d003b31202c,0x722509203233732e\n"
".quad 0x3b34327225202c32,0x315f305f744c2400,0x2e646c003a323832,0x33752e6d61726170\n"
".quad 0x2c35327225092032,0x616475635f5f5b20,0x315a5f5f6d726170,0x654b717369686331\n"
".quad 0x315066506c656e72,0x504d4f4367617431,0x6a5f31533858454c,0x68436d756e5f666a\n"
".quad 0x5d736e6942717369,0x33732e766f6d003b,0x2c36327225092032,0x6573003b32722520\n"
".quad 0x33752e74672e7074,0x202c327025092032,0x7225202c35327225,0x32702540003b3632\n"
".quad 0x4c24092061726220,0x3b343730335f305f,0x33335f305f4c2400,0x57444c24003a3033\n"
".quad 0x6b636f6c62646e65,0x003a315f3330325f,0x6e79732e72616200,0x6500003b30092063\n"
".quad 0x444c24003b746978,0x315a5f5f646e6557,0x654b717369686331,0x315066506c656e72\n"
".quad 0x504d4f4367617431,0x6a5f31533858454c,0x2f2f207d003a666a,0x69686331315a5f20\n"
".quad 0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43,0x0100666a6a5f3153\n"
".quad 0x0825081b08030111,0x000006100b420b13,0x0b0b3e0803002402,0x0b3a01130300000b\n"
".quad 0x13010b0b08030b3b,0x3b0b3a000d040000,0x320a38134908030b,0x0b3a00160500000b\n"
".quad 0x000013490803053b,0x0700001349002606,0x0803053b0b3a0113,0x0d08000013010b0b\n"
".quad 0x490803053b0b3a00,0x0900000b320a3813,0x0b330b0b1349000f,0x3b0b3a01040a0000\n"
".quad 0x0013010b0b08030b,0x0b3b0b3a00280b00,0x010c0000061c0803,0x0013010c3c134901\n"
".quad 0x00000b2f00210d00,0x03053b0b3a00340e,0x00350f0000134908,0x3a00161000001349\n"
".quad 0x00134908030b3b0b,0x0b3b0b3a012e1100,0x408b0c270c3f0803,0x01011201110a400c\n"
".quad 0x0b3a000512000013,0x0a02134908030b3b,0x11010b1300000b33,0x0034140000011201\n"
".quad 0x134908030b3b0b3a,0x1d1500000b330a02,0x0013310112011101,0x053b0b3a00051600\n"
".quad 0x0b330a0213490803,0x3a002e1700001331,0x2713490803053b0b,0x00000c408b0b200c\n"
".quad 0x0002000005380000,0x742f010800000000,0x6678706d742f706d,0x3637303030305f74\n"
".quad 0x30303030305f6264,0x68435f372d303030,0x2e5550475f717369,0x2f00692e33707063\n"
".quad 0x77616b2f656d6f68,0x2d7665642f736569,0x2f6372732f676264,0x65746975736c616c\n"
".quad 0x70736e696c616c2f,0x6372732f6c617269,0x2063636e65706f00,0x0000000400322e33\n"
".quad 0x6769736e75020000,0x00746e692064656e,0x6e6975be0c030407,0x000000c60c003374\n"
".quad 0x0000800078bf0c04,0xbf0c040100230200,0x2302000000800079,0x80007abf0c040104\n"
".quad 0x0001082302000000,0x746e697501980c05,0xc606000000900033,0x6401d60c07000000\n"
".quad 0x0001120c00336d69,0x80007801d70c0800,0x0801002302000000,0x000080007901d70c\n"
".quad 0xd70c080104230200,0x0200000080007a01,0x01e00c0500010823,0x0000d900336d6964\n"
".quad 0x6902000001120600,0x012406040500746e,0x74616f6c66020000,0x0000013009040400\n"
".quad 0x6c62756f64020508,0x6e6f6c0208040065,0x7520676e6f6c2067,0x2064656e6769736e\n"
".quad 0x6c02080700746e69,0x676e6f6c20676e6f,0x0a080500746e6920,0x6f52616475632f08\n"
".quad 0x0065646f4d646e75,0x30080b000001e704,0x6e756f5261647563,0x7473657261654e64\n"
".quad 0x31080b0000000000,0x6e756f5261647563,0x0001006f72655a64,0x64756332080b0000\n"
".quad 0x6f50646e756f5261,0x00000200666e4973,0x6164756334080b00,0x6e694d646e756f52\n"
".quad 0x0000000300666e49,0x6920676e6f6c0200,0x686302080500746e,0x01f3060106007261\n"
".quad 0x08000001fb090000,0x6f6c6601110c0705,0x0002340800327461,0x30007801110c0800\n"
".quad 0x0801002302000001,0x000130007901110c,0x0c05000104230200,0x3274616f6c6601b0\n"
".quad 0x00800c0000020700,0x0d00000250010000,0x5f5f0563130e0005,0x695f747261647563\n"
".quad 0x4300665f69706f32,0x0000012409000002,0x01000000800c0508,0x0600060d0000027d\n"
".quad 0x0001300f00000130,0x0508000001400900,0x344c414552f60310,0x028e090000013000\n"
".quad 0x011e030705080000,0x4c504d4f43676174,0x0002d60800385845,0x0065720120030800\n"
".quad 0x010023020000028e,0x8e006d6901210308,0x0001042302000002,0x504d4f4301230305\n"
".quad 0x0002a2003858454c,0x0508000002d60900,0x33746e6975341110,0x0000008000745f32\n"
".quad 0x34544e4955ee0310,0x6f6c02000002ee00,0x6769736e7520676e,0x00746e692064656e\n"
".quad 0x0800000112090807,0x1e11000003200605,0x69686331315a5f39,0x6c656e72654b7173\n"
".quad 0x6761743131506650,0x3858454c504d4f43,0x0100666a6a5f3153,0x0000000043000101\n"
".quad 0x00000000bc000000,0x1200000526000000,0x736968635f67391e,0x03090000029b0071\n"
".quad 0x0000000000000010,0x00715f67391e1207,0x00180309000002e7,0x1207000000000000\n"
".quad 0x617461645f67391e,0x200309000002e700,0x0700000000000000,0x6f506d756e391e12\n"
".quad 0x0002fe0073746e69,0x0000000028030900,0x6e391e1207000000,0x4271736968436d75\n"
".quad 0x000002fe00736e69,0x00000000002c0309,0x6863391e12070000,0x006d726f4e717369\n"
".quad 0x003003090000028e,0x1307000000000000,0x0000000000000043,0x00000000000000b7\n"
".quad 0x0002ee006c3c1e14,0x020195e4b2900500,0x0000000000004c13,0x000000000000ac00\n"
".quad 0x02ee006a3e1e1400,0xabc8e2b090060000,0x006c58401e140202,0xccb290050000028e\n"
".quad 0x6c59411e14020195,0xb490050000028e00,0x64431e14020195cc,0x8e006c5861746c65\n"
".quad 0x98e2b89006000002,0x6564451e140202ab,0x028e006c5961746c,0xab98e6b290060000\n"
".quad 0x0000901500000202,0x0000930000000000,0x0005260000000000,0x300061020f151600\n"
".quad 0x98e4b29006000001,0x16000000000202ab,0x0001300062020f15,0x02ab98e4b4900600\n"
".quad 0x7815000000000002,0x7b00000000000000,0x2600000000000000,0x61020f1516000005\n"
".quad 0xb890050000013000,0x00000000020195cc,0x01300062020f1516,0xab98e2b090060000\n"
".quad 0x0000000000000202,0x76696466020f1517,0x0001300066656469,0x007c000001010100\n"
".quad 0x0001000000000000,0x0006000000170000,0x0001000000000000,0x0538000000130000\n"
".quad 0x0000000000020000,0x742f706d742f0108,0x30305f746678706d,0x305f626436373030\n"
".quad 0x2d30303030303030,0x5f71736968435f37,0x337070632e555047,0x656d6f682f00692e\n"
".quad 0x2f73656977616b2f,0x2f6762642d766564,0x736c616c2f637273,0x6c616c2f65746975\n"
".quad 0x6c61726970736e69,0x65706f006372732f,0x00322e332063636e,0x7502000000000004\n"
".quad 0x2064656e6769736e,0x0c03040700746e69,0x0c0033746e6975be,0x78bf0c04000000c6\n"
".quad 0x0023020000008000,0x00800079bf0c0401,0x0c04010423020000,0x0200000080007abf\n"
".quad 0x01980c0500010823,0x00900033746e6975,0x07000000c6060000,0x00336d696401d60c\n"
".quad 0xd70c08000001120c,0x0200000080007801,0x7901d70c08010023,0x0423020000008000\n"
".quad 0x80007a01d70c0801,0x0001082302000000,0x336d696401e00c05,0x011206000000d900\n"
".quad 0x0500746e69020000,0x6602000001240604,0x0904040074616f6c,0x6402050800000130\n"
".quad 0x080400656c62756f,0x6f6c20676e6f6c02,0x6769736e7520676e,0x00746e692064656e\n"
".quad 0x20676e6f6c020807,0x746e6920676e6f6c,0x75632f080a080500,0x4d646e756f526164\n"
".quad 0x0001e7040065646f,0x6164756330080b00,0x61654e646e756f52,0x0000000074736572\n"
".quad 0x6164756331080b00,0x72655a646e756f52,0x080b00000001006f,0x756f526164756332\n"
".quad 0x666e49736f50646e,0x34080b0000000200,0x6e756f5261647563,0x00666e496e694d64\n"
".quad 0x6f6c020000000003,0x0500746e6920676e,0x0600726168630208,0xfb09000001f30601\n"
".quad 0x110c070508000001,0x003274616f6c6601,0x110c080000023408,0x0200000130007801\n"
".quad 0x7901110c08010023,0x0423020000013000,0x6c6601b00c050001,0x000207003274616f\n"
".quad 0x5001000000800c00,0x130e00050d000002,0x616475635f5f0563,0x69706f32695f7472\n"
".quad 0x090000024300665f,0x800c050800000124,0x0000027d01000000,0x000001300600060d\n"
".quad 0x014009000001300f,0x52f6031005080000,0x00013000344c4145,0x05080000028e0900\n"
".quad 0x43676174011e0307,0x003858454c504d4f,0x200308000002d608,0x0000028e00657201\n"
".quad 0x0121030801002302,0x020000028e006d69,0x0123030500010423,0x3858454c504d4f43\n"
".quad 0x02d609000002a200,0x7534111005080000,0x00745f3233746e69,0x55ee031000000080\n"
".quad 0x0002ee0034544e49,0x7520676e6f6c0200,0x2064656e6769736e,0x1209080700746e69\n"
".quad 0x0320060508000001,0x315a5f391e110000,0x654b717369686331,0x315066506c656e72\n"
".quad 0x504d4f4367617431,0x6a5f31533858454c,0x000001010100666a,0xf000000000000000\n"
".quad 0x2600000000000008,0x5f67391e12000005,0x029b007173696863,0x0000001003090000\n"
".quad 0x391e120700000000,0x000002e700715f67,0x0000000000180309,0x5f67391e12070000\n"
".quad 0x0002e70061746164,0x0000000020030900,0x6e391e1207000000,0x73746e696f506d75\n"
".quad 0x280309000002fe00,0x0700000000000000,0x68436d756e391e12,0x00736e6942717369\n"
".quad 0x002c0309000002fe,0x1207000000000000,0x4e7173696863391e,0x0000028e006d726f\n"
".quad 0x0000000000300309,0x0000000013070000,0x000008c800000000,0x6c3c1e1400000000\n"
".quad 0xb29005000002ee00,0x00004813020195e4,0x0008780000000000,0x3e1e140000000000\n"
".quad 0x9006000002ee006a,0x1e140202abc8e2b0,0x0000028e006c5840,0x14020195ccb29005\n"
".quad 0x00028e006c59411e,0x020195ccb4900500,0x61746c6564431e14,0x060000028e006c58\n"
".quad 0x140202ab98e2b890,0x5961746c6564451e,0x90060000028e006c,0x00000202ab98e6b2\n"
".quad 0x0000000000060815,0x0000000000066000,0x0f15160000052600,0x0600000130006102\n"
".quad 0x000202ab98e4b290,0x62020f1516000000,0xb490060000013000,0x0000000202ab98e4\n"
".quad 0x0000000450150000,0x00000004a8000000,0x1600000526000000,0x0001300061020f15\n"
".quad 0x020195ccb8900500,0x020f151600000000,0x9006000001300062,0x00000202ab98e2b0\n"
".quad 0x020f151700000000,0x6665646976696466,0x0101010000013000,0x0000000004ed0000\n"
".quad 0x001f000000020000,0x0000000004e50000,0x001f000000020000,0x0000000004ab0000\n"
".quad 0x001f000000020000,0x0000000004a30000,0x001f000000020000,0x0000000004370000\n"
".quad 0x001f000000020000,0x00000000042f0000,0x001f000000020000,0x0000000004160000\n"
".quad 0x001f000000020000,0x00000000040e0000,0x001f000000020000,0x0000000003610000\n"
".quad 0x001f000000020000,0x0000000003590000,0x001f000000020000,0x00000000007c0000\n"
".quad 0x0017000000010000,0x0000000000060000,0x0013000000010000,0x686331315a5f0000\n"
".quad 0x656e72654b717369,0x617431315066506c,0x58454c504d4f4367,0x00666a6a5f315338\n"
".quad 0x0000000000000085,0x0500000100317225,0x0000004000000000,0x0032722500000000\n"
".quad 0x0000000405000001,0x0000000000000880,0x0500000200337225,0x0000002800000018\n"
".quad 0x0034722500000000,0x0000002005000001,0x0000000000000040,0x0100000000317025\n"
".quad 0x0000004000000030,0x3764722500000001,0x0000480500000500,0x0000000000004000\n"
".quad 0x0004003764722500,0x0040000000480500,0x7225000000000000,0x0050050000030035\n"
".quad 0x0000000000a80000,0x0004003672250000,0x00f4000000680500,0x7225000000000000\n"
".quad 0x0080050000020037,0x0000000000980000,0x0002003872250000,0x00a0000000980500\n"
".quad 0x7225000000000000,0x00a0050000020039,0x00000000044c0000,0x0200303172250000\n"
".quad 0x20000000a4050000,0x2500000000000008,0x0500000600316472,0x000000cc000000b8\n"
".quad 0x3164722500000001,0x0000c00500000700,0x0000000000011000,0x0005003164722500\n"
".quad 0x0198000000c80500,0x7225000000010000,0xcc05000006003164,0x00000001f0000000\n"
".quad 0x0031317225000000,0x000000d005000002,0x000000000000044c,0x0000030032317225\n"
".quad 0x000108000000e805,0x3172250000000000,0x00f0050000010033,0x00000000044c0000\n"
".quad 0x0300343172250000,0x1400000108050000,0x2500000000000001,0x0500000700353172\n"
".quad 0x0000018800000110,0x3364722500000000,0x00017c0500000800,0x0000010000023800\n"
".quad 0x0009003364722500,0x0268000001800500,0x7225000000000000,0x8405000003003364\n"
".quad 0x010000019c000001,0x0033647225000000,0x0000018805000007,0x00000000000001f4\n"
".quad 0x0000040034647225,0x0001a80000018c05,0x6472250000000100,0x0198050000050034\n"
".quad 0x0000000001e80000,0x0003003166250000,0x0444000001c00500,0x6625000000000000\n"
".quad 0x01c8050000030032,0x0000000004440000,0x0400356472250000,0x00000001e0050000\n"
".quad 0x2500000001000002,0x0500000500356472,0x0000021c000001e8,0x3564722500000000\n"
".quad 0x0001f00500000600,0x000001000002d800,0x0007003564722500,0x0338000001f40500\n"
".quad 0x7225000000000000,0xf805000002003631,0x000000044c000001,0x0037317225000000\n"
".quad 0x0000021005000004,0x0000000000000230,0x0000010038317225,0x00044c0000021805\n"
".quad 0x3172250000000000,0x0230050000040039,0x00000000023c0000,0x0800303272250000\n"
".quad 0xc000000238050000,0x2500000000000002,0x0500000400376472,0x000002c4000002b0\n"
".quad 0x3764722500000001,0x0002b80500000500,0x000000000002c800,0x0008003764722500\n"
".quad 0x0328000002c00500,0x7225000000010000,0xc405000004003764,0x00000002e0000002\n"
".quad 0x0038647225000000,0x000002c805000005,0x00000001000002f0,0x0000060038647225\n"
".quad 0x000350000002d805,0x3366250000000000,0x0003080500000400,0x0000000000044c00\n"
".quad 0x0000040034662500,0x00044c0000031005,0x6472250000000000,0x0328050000080039\n"
".quad 0x00010000033c0000,0x0900396472250000,0x7000000330050000,0x2500000000000003\n"
".quad 0x0500000700396472,0x000003c800000338,0x3964722500000001,0x00033c0500000800\n"
".quad 0x0000000000044c00,0x0a00313164722500,0x4c000003ac050000,0x2500000001000004\n"
".quad 0x00000b0031316472,0x00044c000003b005,0x6472250000000000,0xb405000005003131\n"
".quad 0x01000003cc000003,0x3131647225000000,0x0003b80500000900,0x0000000000044c00\n"
".quad 0x0600323164722500,0xd8000003bc050000,0x2500000001000003,0x0000070032316472\n"
".quad 0x000438000003c805,0x3566250000000000,0x0003f00500000500,0x0000000000041800\n"
".quad 0x0000060036662500,0x0004100000040805,0x3766250000000000,0x0004100500000600\n"
".quad 0x0000000000044c00,0x0000060038662500,0x0004580000041405,0x3272250000000000\n"
".quad 0x0428050000050031,0x0000000004300000,0x0005003966250000,0x0440000004300500\n"
".quad 0x6625000000000000,0x3805000007003031,0x0000000478000004,0x0500326625000000\n"
".quad 0xc400000440050000,0x2500000000000004,0x4405000003003466,0x000000067c000004\n"
".quad 0x0031316625000000,0x0000045005000004,0x0000000000000478,0x0000070032316625\n"
".quad 0x0004780000045405,0x3166250000000000,0x049c050000040033,0x0000000004c80000\n"
".quad 0x0600343166250000,0x08000004b8050000,0x2500000000000005,0x0500000500353166\n"
".quad 0x000004c4000004c0,0x3631662500000000,0x0004c40500000500,0x000000000004d000\n"
".quad 0x0004003731662500,0x0604000004c80500,0x6625000000000000,0xcc05000004003831\n"
".quad 0x00000007c8000004,0x3331647225000000,0x0004e00500000800,0x000001000004f400\n"
".quad 0x0900333164722500,0x28000004e8050000,0x2500000000000005,0x0000070033316472\n"
".quad 0x000580000004f005,0x6472250000000100,0xf405000008003331,0x0000000604000004\n"
".quad 0x3531647225000000,0x0005640500000a00,0x0000010000060400,0x0b00353164722500\n"
".quad 0x0400000568050000,0x2500000000000006,0x0000050035316472,0x0005880000056c05\n"
".quad 0x6472250000000100,0x7005000009003531,0x0000000604000005,0x3631647225000000\n"
".quad 0x0005740500000600,0x0000010000059800,0x0700363164722500,0x0400000580050000\n"
".quad 0x2500000000000006,0x0500000500393166,0x000005d0000005b0,0x3032662500000000\n"
".quad 0x0005c80500000600,0x000000000005d800,0x0005003132662500,0x0604000005d00500\n"
".quad 0x6625000000000000,0xd405000005003232,0x0000000630000005,0x0032327225000000\n"
".quad 0x000005e805000006,0x00000000000005f0,0x0000060033326625,0x000604000005f005\n"
".quad 0x3266250000000000,0x05f8050000060034,0x0000000006300000,0x0500353266250000\n"
".quad 0x3000000608050000,0x2500000000000006,0x0500000600363266,0x000006300000060c\n"
".quad 0x3732662500000000,0x0006540500000500,0x0000000000068800,0x0006003832662500\n"
".quad 0x06c0000006700500,0x6625000000000000,0x7805000003003932,0x000000067c000006\n"
".quad 0x0030336625000000,0x0000067c05000003,0x0000000000000680,0x0000030031336625\n"
".quad 0x0007780000068005,0x3366250000000000,0x0684050000030032,0x0000000007780000\n"
".quad 0x0037316472250000,0x0000069805000008,0x00000001000006ac,0x0009003731647225\n"
".quad 0x06e0000006a00500,0x7225000000000000,0x0500000700373164,0x00000738000006a8\n"
".quad 0x3164722500000001,0x06ac050000080037,0x0000000007a00000,0x0039316472250000\n"
".quad 0x0000071c0500000a,0x00000001000007f8,0x000b003931647225,0x0870000007200500\n"
".quad 0x7225000000000000,0x0500000500393164,0x0000073c00000724,0x3164722500000001\n"
".quad 0x0728050000090039,0x0000000007a80000,0x0030326472250000,0x0000072c05000006\n"
".quad 0x0000000100000748,0x0007003032647225,0x07b4000007380500,0x6625000000000000\n"
".quad 0x6005000006003333,0x00000007b0000007,0x0034336625000000,0x0000076805000003\n"
".quad 0x0000000000000778,0x0000050035336625,0x0007700000076c05,0x3366250000000000\n"
".quad 0x0770050000050036,0x0000000007880000,0x0400373366250000,0xc800000774050000\n"
".quad 0x2500000000000007,0x0500000300383366,0x0000078000000778,0x3933662500000000\n"
".quad 0x0007800500000300,0x0000000000079000,0x0005003034662500,0x0870000007880500\n"
".quad 0x7225000000000000,0x0500000800313264,0x000007e0000007a0,0x3264722500000001\n"
".quad 0x07a8050000090031,0x0000000007e80000,0x0031326472250000,0x000007b005000006\n"
".quad 0x0000000100000870,0x0007003132647225,0x0870000007b40500,0x7225000000010000\n"
".quad 0x0500000300333264,0x0000083800000828,0x3264722500000000,0x082c050000020033\n"
".quad 0x0000000008340000,0x0033326472250000,0x0000083005000004,0x0000000100000848\n"
".quad 0x0002003332647225,0x084c000008340500,0x7225000000000000,0x0500000300343264\n"
".quad 0x0000085800000838,0x3264722500000001,0x0848050000040034,0x0000000008700000\n"
".quad 0x0100333272250000,0x8000000878050000,0x2500000000000008,0x0500000100343272\n"
".quad 0x0000088c00000880,0x0032722500000000,0x0000088805000001,0x00000000000008c0\n"
".quad 0x0000020035327225,0x0008b0000008a005,0x3272250000000000,0x08a8050000010036\n"
".quad 0x0000000008c00000,0x0000003270250000,0x08c0000008b80100,0x0000000600000000\n"
".quad 0x0000481c00000005,0x0000000000000000,0x0000000000000000,0x000000a800000000\n"
".quad 0x000000a800000000,0x0000000400000000,0x6000000000000000,0x00000c8b00001f05\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000098400000000,0x0000098400000000\n"
".quad 0x0000000400000000,0x6000000000000000,0x0000160f00001f06,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000003400000000,0x0000000400000000\n"
".quad 0x0000000000000000\n"
".text");

extern "C" {

extern const unsigned long long __deviceText_$sm_10$[2329];

}

asm(
".section .rodata\n"
".align 32\n"
"__deviceText_$compute_10$:\n"
".quad 0x6f69737265762e09,0x2e090a342e31206e,0x7320746567726174,0x616d202c30315f6d\n"
".quad 0x6f745f3436665f70,0x2f2f090a3233665f,0x656c69706d6f6320,0x2f20687469772064\n"
".quad 0x61636f6c2f727375,0x61696469766e2f6c,0x322e332d6b64732f,0x706f2f616475632f\n"
".quad 0x62696c2f34366e65,0x2f2f090a65622f2f,0x636e65706f766e20,0x756220322e332063\n"
".quad 0x32206e6f20746c69,0x302d31312d303130,0x2d2d2f2f090a0a33,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x6f43202f2f090a2d,0x20676e696c69706d\n"
".quad 0x706d742f706d742f,0x303030305f746678,0x3030305f62643637,0x5f372d3030303030\n"
".quad 0x50475f7173696843,0x692e337070632e55,0x632f706d742f2820,0x444f652e23494263\n"
".quad 0x2f2f090a2956446c,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2f2f090a0a2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x202f2f090a2d2d2d,0x3a736e6f6974704f,0x2d2d2d2d2f2f090a,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x0a2d2d2d2d2d2d2d,0x72615420202f2f09,0x2c7874703a746567\n"
".quad 0x5f6d733a41534920,0x69646e45202c3031,0x6c7474696c3a6e61,0x746e696f50202c65\n"
".quad 0x3a657a6953207265,0x20202f2f090a3436,0x74704f2809304f2d,0x6f6974617a696d69\n"
".quad 0x296c6576656c206e,0x672d20202f2f090a,0x6775626544280932,0x0a296c6576656c20\n"
".quad 0x326d2d20202f2f09,0x74726f7065522809,0x726f736976646120,0x2f2f090a29736569\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x662e090a0a2d2d2d\n"
".quad 0x3c22093109656c69,0x2d646e616d6d6f63,0x090a223e656e696c,0x093209656c69662e\n"
".quad 0x6d742f706d742f22,0x3030305f74667870,0x30305f6264363730,0x362d303030303030\n"
".quad 0x475f71736968435f,0x66616475632e5550,0x0a227570672e3265,0x3309656c69662e09\n"
".quad 0x2f656d6f682f2209,0x642f73656977616b,0x756c636e692f7665,0x4c2f6c616c2f6564\n"
".quad 0x63696d6f74414c41,0x6570797461746144,0x662e090a22682e73,0x2f22093409656c69\n"
".quad 0x2f62696c2f727375,0x5f3638782f636367,0x78756e696c2d3436,0x342e342f756e672d\n"
".quad 0x756c636e692f352e,0x65646474732f6564,0x662e090a22682e66,0x2f22093509656c69\n"
".quad 0x61636f6c2f727375,0x61696469766e2f6c,0x322e332d6b64732f,0x69622f616475632f\n"
".quad 0x636e692f2e2e2f6e,0x7472632f6564756c,0x5f6563697665642f,0x2e656d69746e7572\n"
".quad 0x6c69662e090a2268,0x73752f2209360965,0x2f6c61636f6c2f72,0x732f61696469766e\n"
".quad 0x632f322e332d6b64,0x2f6e69622f616475,0x756c636e692f2e2e,0x5f74736f682f6564\n"
".quad 0x2e73656e69666564,0x6c69662e090a2268,0x73752f2209370965,0x2f6c61636f6c2f72\n"
".quad 0x732f61696469766e,0x632f322e332d6b64,0x2f6e69622f616475,0x756c636e692f2e2e\n"
".quad 0x746c6975622f6564,0x73657079745f6e69,0x69662e090a22682e,0x752f22093809656c\n"
".quad 0x6c61636f6c2f7273,0x2f61696469766e2f,0x2f322e332d6b6473,0x6e69622f61647563\n"
".quad 0x6c636e692f2e2e2f,0x697665642f656475,0x73657079745f6563,0x69662e090a22682e\n"
".quad 0x752f22093909656c,0x6c61636f6c2f7273,0x2f61696469766e2f,0x2f322e332d6b6473\n"
".quad 0x6e69622f61647563,0x6c636e692f2e2e2f,0x766972642f656475,0x73657079745f7265\n"
".quad 0x69662e090a22682e,0x2f2209303109656c,0x61636f6c2f727375,0x61696469766e2f6c\n"
".quad 0x322e332d6b64732f,0x69622f616475632f,0x636e692f2e2e2f6e,0x7275732f6564756c\n"
".quad 0x7079745f65636166,0x2e090a22682e7365,0x09313109656c6966,0x6f6c2f7273752f22\n"
".quad 0x6469766e2f6c6163,0x332d6b64732f6169,0x2f616475632f322e,0x692f2e2e2f6e6962\n"
".quad 0x742f6564756c636e,0x745f657275747865,0x0a22682e73657079,0x3109656c69662e09\n"
".quad 0x2f7273752f220932,0x766e2f6c61636f6c,0x6b64732f61696469,0x6475632f322e332d\n"
".quad 0x2e2e2f6e69622f61,0x6564756c636e692f,0x5f726f746365762f,0x22682e7365707974\n"
".quad 0x09656c69662e090a,0x7273752f22093331,0x6e2f6c61636f6c2f,0x64732f6169646976\n"
".quad 0x75632f322e332d6b,0x2e2f6e69622f6164,0x64756c636e692f2e,0x6563697665642f65\n"
".quad 0x5f68636e75616c5f,0x6574656d61726170,0x2e090a22682e7372,0x09343109656c6966\n"
".quad 0x6f6c2f7273752f22,0x6469766e2f6c6163,0x332d6b64732f6169,0x2f616475632f322e\n"
".quad 0x692f2e2e2f6e6962,0x632f6564756c636e,0x61726f74732f7472,0x7373616c635f6567\n"
".quad 0x69662e090a22682e,0x2f2209353109656c,0x6c636e692f727375,0x737469622f656475\n"
".quad 0x682e73657079742f,0x656c69662e090a22,0x73752f2209363109,0x64756c636e692f72\n"
".quad 0x682e656d69742f65,0x656c69662e090a22,0x73752f2209373109,0x64756c636e692f72\n"
".quad 0x746e696474732f65,0x69662e090a22682e,0x2f2209383109656c,0x61636f6c2f727375\n"
".quad 0x61696469766e2f6c,0x322e332d6b64732f,0x69622f616475632f,0x636e692f2e2e2f6e\n"
".quad 0x6d6f632f6564756c,0x636e75665f6e6f6d,0x22682e736e6f6974,0x09656c69662e090a\n"
".quad 0x7273752f22093931,0x6e2f6c61636f6c2f,0x64732f6169646976,0x75632f322e332d6b\n"
".quad 0x2e2f6e69622f6164,0x64756c636e692f2e,0x665f6874616d2f65,0x736e6f6974636e75\n"
".quad 0x69662e090a22682e,0x2f2209303209656c,0x61636f6c2f727375,0x61696469766e2f6c\n"
".quad 0x322e332d6b64732f,0x69622f616475632f,0x636e692f2e2e2f6e,0x74616d2f6564756c\n"
".quad 0x6174736e6f635f68,0x090a22682e73746e,0x313209656c69662e,0x6c2f7273752f2209\n"
".quad 0x69766e2f6c61636f,0x2d6b64732f616964,0x616475632f322e33,0x2f2e2e2f6e69622f\n"
".quad 0x2f6564756c636e69,0x665f656369766564,0x736e6f6974636e75,0x69662e090a22682e\n"
".quad 0x2f2209323209656c,0x61636f6c2f727375,0x61696469766e2f6c,0x322e332d6b64732f\n"
".quad 0x69622f616475632f,0x636e692f2e2e2f6e,0x5f6d732f6564756c,0x696d6f74615f3131\n"
".quad 0x6974636e75665f63,0x090a22682e736e6f,0x333209656c69662e,0x6c2f7273752f2209\n"
".quad 0x69766e2f6c61636f,0x2d6b64732f616964,0x616475632f322e33,0x2f2e2e2f6e69622f\n"
".quad 0x2f6564756c636e69,0x74615f32315f6d73,0x6e75665f63696d6f,0x682e736e6f697463\n"
".quad 0x656c69662e090a22,0x73752f2209343209,0x2f6c61636f6c2f72,0x732f61696469766e\n"
".quad 0x632f322e332d6b64,0x2f6e69622f616475,0x756c636e692f2e2e,0x33315f6d732f6564\n"
".quad 0x5f656c62756f645f,0x6e6f6974636e7566,0x662e090a22682e73,0x2209353209656c69\n"
".quad 0x636f6c2f7273752f,0x696469766e2f6c61,0x2e332d6b64732f61,0x622f616475632f32\n"
".quad 0x6e692f2e2e2f6e69,0x6d732f6564756c63,0x6d6f74615f30325f,0x74636e75665f6369\n"
".quad 0x0a22682e736e6f69,0x3209656c69662e09,0x2f7273752f220936,0x766e2f6c61636f6c\n"
".quad 0x6b64732f61696469,0x6475632f322e332d,0x2e2e2f6e69622f61,0x6564756c636e692f\n"
".quad 0x695f30325f6d732f,0x6369736e6972746e,0x662e090a22682e73,0x2209373209656c69\n"
".quad 0x636f6c2f7273752f,0x696469766e2f6c61,0x2e332d6b64732f61,0x622f616475632f32\n"
".quad 0x6e692f2e2e2f6e69,0x75732f6564756c63,0x75665f6563616672,0x2e736e6f6974636e\n"
".quad 0x6c69662e090a2268,0x752f220938320965,0x6c61636f6c2f7273,0x2f61696469766e2f\n"
".quad 0x2f322e332d6b6473,0x6e69622f61647563,0x6c636e692f2e2e2f,0x747865742f656475\n"
".quad 0x637465665f657275,0x6974636e75665f68,0x090a22682e736e6f,0x393209656c69662e\n"
".quad 0x6c2f7273752f2209,0x69766e2f6c61636f,0x2d6b64732f616964,0x616475632f322e33\n"
".quad 0x2f2e2e2f6e69622f,0x2f6564756c636e69,0x6e75665f6874616d,0x645f736e6f697463\n"
".quad 0x2e317874705f6c62,0x6c69662e090a2268,0x6843220930330965,0x2e5550475f717369\n"
".quad 0x2e090a0a0a227563,0x5a5f207972746e65,0x4b71736968633131,0x5066506c656e7265\n"
".quad 0x4d4f436761743131,0x5f31533858454c50,0x09090a2820666a6a,0x2e206d617261702e\n"
".quad 0x75635f5f20343675,0x5f5f6d7261706164,0x717369686331315a,0x66506c656e72654b\n"
".quad 0x4f43676174313150,0x31533858454c504d,0x635f675f666a6a5f,0x09090a2c71736968\n"
".quad 0x2e206d617261702e,0x75635f5f20343675,0x5f5f6d7261706164,0x717369686331315a\n"
".quad 0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d,0x715f675f666a6a5f\n"
".quad 0x7261702e09090a2c,0x203436752e206d61,0x6170616475635f5f,0x6331315a5f5f6d72\n"
".quad 0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761,0x666a6a5f31533858\n"
".quad 0x2c617461645f675f,0x617261702e09090a,0x5f203233752e206d,0x726170616475635f\n"
".quad 0x686331315a5f5f6d,0x656e72654b717369,0x617431315066506c,0x58454c504d4f4367\n"
".quad 0x5f666a6a5f315338,0x746e696f506d756e,0x61702e09090a2c73,0x3233752e206d6172\n"
".quad 0x70616475635f5f20,0x31315a5f5f6d7261,0x72654b7173696863,0x31315066506c656e\n"
".quad 0x4c504d4f43676174,0x6a6a5f3153385845,0x6968436d756e5f66,0x0a2c736e69427173\n"
".quad 0x6d617261702e0909,0x5f5f203233662e20,0x6d72617061647563,0x69686331315a5f5f\n"
".quad 0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43,0x635f666a6a5f3153\n"
".quad 0x6d726f4e71736968,0x722e090a7b090a29,0x203233752e206765,0x0a3b3e38323c7225\n"
".quad 0x752e206765722e09,0x323c647225203436,0x65722e090a3b3e36,0x25203233662e2067\n"
".quad 0x090a3b3e32343c66,0x72702e206765722e,0x3e343c7025206465,0x09636f6c2e090a3b\n"
".quad 0x0a30093735093033,0x6967656257444c24,0x686331315a5f5f6e,0x656e72654b717369\n"
".quad 0x617431315066506c,0x58454c504d4f4367,0x3a666a6a5f315338,0x67656257444c240a\n"
".quad 0x5f6b636f6c626e69,0x090a3a315f333032,0x09303309636f6c2e,0x6f6d090a30093036\n"
".quad 0x2509203233752e76,0x090a3b30202c3172,0x203233732e766f6d,0x7225202c32722509\n"
".quad 0x702e646c090a3b31,0x3233752e6d617261,0x5b202c3372250920,0x6170616475635f5f\n"
".quad 0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761\n"
".quad 0x666a6a5f31533858,0x736968436d756e5f,0x0a3b5d736e694271,0x3233732e766f6d09\n"
".quad 0x25202c3472250920,0x746573090a3b3272,0x3233752e656c2e70,0x25202c3170250920\n"
".quad 0x3b347225202c3372,0x622031702540090a,0x305f4c2409206172,0x240a3b303333335f\n"
".quad 0x343730335f305f4c,0x656257444c240a3a,0x6b636f6c626e6967,0x0a3a335f3330325f\n"
".quad 0x303309636f6c2e09,0x63090a3009323609,0x752e3233752e7476,0x2c35722509203631\n"
".quad 0x3b782e6469742520,0x33752e747663090a,0x2509203631752e32,0x61746325202c3672\n"
".quad 0x63090a3b782e6469,0x752e3233752e7476,0x2c37722509203631,0x782e6469746e2520\n"
".quad 0x6c2e6c756d090a3b,0x2509203233752e6f,0x2c367225202c3872,0x61090a3b37722520\n"
".quad 0x09203233752e6464,0x357225202c397225,0x090a3b387225202c,0x203233732e766f6d\n"
".quad 0x25202c3031722509,0x6f6c2e090a3b3972,0x0934360930330963,0x61702e646c090a30\n"
".quad 0x203436752e6d6172,0x5b202c3164722509,0x6170616475635f5f,0x6331315a5f5f6d72\n"
".quad 0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761,0x666a6a5f31533858\n"
".quad 0x5d617461645f675f,0x732e766f6d090a3b,0x3131722509203233,0x0a3b30317225202c\n"
".quad 0x617261702e646c09,0x2509203233752e6d,0x5f5f5b202c323172,0x6d72617061647563\n"
".quad 0x69686331315a5f5f,0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43\n"
".quad 0x6e5f666a6a5f3153,0x73746e696f506d75,0x2e766f6d090a3b5d,0x3172250920323373\n"
".quad 0x0a3b327225202c33,0x2e6f6c2e6c756d09,0x3172250920323375,0x2c32317225202c34\n"
".quad 0x090a3b3331722520,0x203233752e646461,0x25202c3531722509,0x317225202c313172\n"
".quad 0x2e747663090a3b34,0x203233752e343675,0x25202c3264722509,0x756d090a3b353172\n"
".quad 0x752e656469772e6c,0x3364722509203233,0x202c35317225202c,0x2e646461090a3b38\n"
".quad 0x6472250920343675,0x2c31647225202c34,0x090a3b3364722520,0x61626f6c672e646c\n"
".quad 0x2509203233662e6c,0x6472255b202c3166,0x6d090a3b5d302b34,0x09203233662e766f\n"
".quad 0x316625202c326625,0x09636f6c2e090a3b,0x0a30093536093033,0x617261702e646c09\n"
".quad 0x2509203436752e6d,0x5f5f5b202c356472,0x6d72617061647563,0x69686331315a5f5f\n"
".quad 0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43,0x675f666a6a5f3153\n"
".quad 0x0a3b5d617461645f,0x3233732e766f6d09,0x202c363172250920,0x6c090a3b30317225\n"
".quad 0x2e6d617261702e64,0x3172250920323375,0x75635f5f5b202c37,0x5f5f6d7261706164\n"
".quad 0x717369686331315a,0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d\n"
".quad 0x6d756e5f666a6a5f,0x3b5d73746e696f50,0x33732e766f6d090a,0x2c38317225092032\n"
".quad 0x6d090a3b32722520,0x33752e6f6c2e6c75,0x2c39317225092032,0x25202c3731722520\n"
".quad 0x6461090a3b383172,0x2509203233752e64,0x317225202c303272,0x3b39317225202c36\n"
".quad 0x36752e747663090a,0x2509203233752e34,0x327225202c366472,0x2e6c756d090a3b30\n"
".quad 0x3233752e65646977,0x202c376472250920,0x3b38202c30327225,0x36752e646461090a\n"
".quad 0x2c38647225092034,0x25202c3564722520,0x646c090a3b376472,0x2e6c61626f6c672e\n"
".quad 0x3366250920323366,0x2b386472255b202c,0x766f6d090a3b5d34,0x662509203233662e\n"
".quad 0x0a3b336625202c34,0x303309636f6c2e09,0x6c090a3009373609,0x2e6d617261702e64\n"
".quad 0x6472250920343675,0x75635f5f5b202c39,0x5f5f6d7261706164,0x717369686331315a\n"
".quad 0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d,0x715f675f666a6a5f\n"
".quad 0x2e747663090a3b5d,0x203233752e343675,0x202c303164722509,0x6d090a3b30317225\n"
".quad 0x2e656469772e6c75,0x6472250920323375,0x30317225202c3131,0x6461090a3b38202c\n"
".quad 0x2509203436752e64,0x7225202c32316472,0x31647225202c3964,0x672e646c090a3b31\n"
".quad 0x33662e6c61626f6c,0x202c356625092032,0x302b32316472255b,0x702e646c090a3b5d\n"
".quad 0x3233662e6d617261,0x5b202c3666250920,0x6170616475635f5f,0x6331315a5f5f6d72\n"
".quad 0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761,0x666a6a5f31533858\n"
".quad 0x6f4e71736968635f,0x756d090a3b5d6d72,0x2509203233662e6c,0x2c356625202c3766\n"
".quad 0x6d090a3b36662520,0x09203233662e766f,0x376625202c386625,0x61702e646c090a3b\n"
".quad 0x203233752e6d6172,0x5b202c3132722509,0x6170616475635f5f,0x6331315a5f5f6d72\n"
".quad 0x6e72654b71736968,0x7431315066506c65,0x454c504d4f436761,0x666a6a5f31533858\n"
".quad 0x736968436d756e5f,0x0a3b5d736e694271,0x2e6e722e74766309,0x203233752e323366\n"
".quad 0x7225202c39662509,0x766f6d090a3b3132,0x662509203233662e,0x3b396625202c3031\n"
".quad 0x67656257444c240a,0x766964665f696e69,0x3330325f66656469,0x6f6c2e090a3a395f\n"
".quad 0x3133350931320963,0x2e766f6d090a3009,0x3166250920323366,0x0a3b386625202c31\n"
".quad 0x3233662e766f6d09,0x202c323166250920,0x64090a3b30316625,0x2e6c6c75662e7669\n"
".quad 0x3166250920323366,0x2c31316625202c33,0x240a3b3231662520,0x5f69646e6557444c\n"
".quad 0x6665646976696466,0x0a3a385f3330325f,0x303309636f6c2e09,0x6c090a3009373609\n"
".quad 0x2e6d617261702e64,0x3166250920323366,0x75635f5f5b202c34,0x5f5f6d7261706164\n"
".quad 0x717369686331315a,0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d\n"
".quad 0x6968635f666a6a5f,0x3b5d6d726f4e7173,0x33662e766f6d090a,0x2c35316625092032\n"
".quad 0x6d090a3b32662520,0x09203233662e6c75,0x6625202c36316625,0x35316625202c3431\n"
".quad 0x662e627573090a3b,0x3731662509203233,0x202c36316625202c,0x6d090a3b33316625\n"
".quad 0x09203233662e766f,0x6625202c38316625,0x6f6c2e090a3b3731,0x0939360930330963\n"
".quad 0x61702e646c090a30,0x203436752e6d6172,0x202c333164722509,0x70616475635f5f5b\n"
".quad 0x31315a5f5f6d7261,0x72654b7173696863,0x31315066506c656e,0x4c504d4f43676174\n"
".quad 0x6a6a5f3153385845,0x0a3b5d715f675f66,0x3436752e74766309,0x722509203233752e\n"
".quad 0x317225202c343164,0x2e6c756d090a3b30,0x3233752e65646977,0x2c35316472250920\n"
".quad 0x38202c3031722520,0x752e646461090a3b,0x3164722509203436,0x3331647225202c36\n"
".quad 0x3b3531647225202c,0x6f6c672e646c090a,0x203233662e6c6162,0x5b202c3931662509\n"
".quad 0x5d342b3631647225,0x61702e646c090a3b,0x203233662e6d6172,0x5b202c3032662509\n"
".quad 0x6170616475635f5f,0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x6f4e71736968635f,0x756d090a3b5d6d72\n"
".quad 0x2509203233662e6c,0x316625202c313266,0x3b30326625202c39,0x33662e766f6d090a\n"
".quad 0x2c32326625092032,0x090a3b3132662520,0x6d617261702e646c,0x722509203233752e\n"
".quad 0x635f5f5b202c3232,0x5f6d726170616475,0x7369686331315a5f,0x506c656e72654b71\n"
".quad 0x4367617431315066,0x533858454c504d4f,0x756e5f666a6a5f31,0x694271736968436d\n"
".quad 0x7663090a3b5d736e,0x3233662e6e722e74,0x662509203233752e,0x32327225202c3332\n"
".quad 0x662e766f6d090a3b,0x3432662509203233,0x0a3b33326625202c,0x6967656257444c24\n"
".quad 0x69766964665f696e,0x5f3330325f666564,0x636f6c2e090a3a37,0x0931333509313209\n"
".quad 0x662e766f6d090a30,0x3532662509203233,0x0a3b32326625202c,0x3233662e766f6d09\n"
".quad 0x202c363266250920,0x64090a3b34326625,0x2e6c6c75662e7669,0x3266250920323366\n"
".quad 0x2c35326625202c37,0x240a3b3632662520,0x5f69646e6557444c,0x6665646976696466\n"
".quad 0x0a3a365f3330325f,0x303309636f6c2e09,0x6c090a3009393609,0x2e6d617261702e64\n"
".quad 0x3266250920323366,0x75635f5f5b202c38,0x5f5f6d7261706164,0x717369686331315a\n"
".quad 0x66506c656e72654b,0x4f43676174313150,0x31533858454c504d,0x6968635f666a6a5f\n"
".quad 0x3b5d6d726f4e7173,0x33662e766f6d090a,0x2c39326625092032,0x6d090a3b34662520\n"
".quad 0x09203233662e6c75,0x6625202c30336625,0x39326625202c3832,0x662e627573090a3b\n"
".quad 0x3133662509203233,0x202c30336625202c,0x6d090a3b37326625,0x09203233662e766f\n"
".quad 0x6625202c32336625,0x6f6c2e090a3b3133,0x0932370930330963,0x61702e646c090a30\n"
".quad 0x203436752e6d6172,0x202c373164722509,0x70616475635f5f5b,0x31315a5f5f6d7261\n"
".quad 0x72654b7173696863,0x31315066506c656e,0x4c504d4f43676174,0x6a6a5f3153385845\n"
".quad 0x736968635f675f66,0x747663090a3b5d71,0x3233752e3436752e,0x2c38316472250920\n"
".quad 0x090a3b3031722520,0x656469772e6c756d,0x722509203233752e,0x317225202c393164\n"
".quad 0x61090a3b34202c30,0x09203436752e6464,0x25202c3032647225,0x7225202c37316472\n"
".quad 0x646c090a3b393164,0x2e6c61626f6c672e,0x3366250920323366,0x326472255b202c33\n"
".quad 0x6d090a3b5d302b30,0x09203233662e766f,0x6625202c34336625,0x766f6d090a3b3233\n"
".quad 0x662509203233662e,0x32336625202c3533,0x662e6c756d090a3b,0x3633662509203233\n"
".quad 0x202c34336625202c,0x6d090a3b35336625,0x09203233662e766f,0x6625202c37336625\n"
".quad 0x766f6d090a3b3831,0x662509203233662e,0x38316625202c3833,0x662e64616d090a3b\n"
".quad 0x3933662509203233,0x202c37336625202c,0x6625202c38336625,0x646461090a3b3633\n"
".quad 0x662509203233662e,0x33336625202c3034,0x0a3b39336625202c,0x617261702e646c09\n"
".quad 0x2509203436752e6d,0x5f5b202c31326472,0x726170616475635f,0x686331315a5f5f6d\n"
".quad 0x656e72654b717369,0x617431315066506c,0x58454c504d4f4367,0x5f666a6a5f315338\n"
".quad 0x5d71736968635f67,0x752e747663090a3b,0x09203233752e3436,0x25202c3232647225\n"
".quad 0x756d090a3b303172,0x752e656469772e6c,0x3264722509203233,0x2c30317225202c33\n"
".quad 0x646461090a3b3420,0x722509203436752e,0x647225202c343264,0x32647225202c3132\n"
".quad 0x672e7473090a3b33,0x33662e6c61626f6c,0x326472255b092032,0x6625202c5d302b34\n"
".quad 0x57444c240a3b3034,0x6b636f6c62646e65,0x0a3a335f3330325f,0x303309636f6c2e09\n"
".quad 0x6d090a3009303609,0x09203233732e766f,0x7225202c33327225,0x2e646461090a3b32\n"
".quad 0x3272250920323375,0x2c33327225202c34,0x766f6d090a3b3120,0x722509203233732e\n"
".quad 0x3b34327225202c32,0x315f305f744c240a,0x646c090a3a323832,0x752e6d617261702e\n"
".quad 0x3532722509203233,0x6475635f5f5b202c,0x5a5f5f6d72617061,0x4b71736968633131\n"
".quad 0x5066506c656e7265,0x4d4f436761743131,0x5f31533858454c50,0x436d756e5f666a6a\n"
".quad 0x736e694271736968,0x2e766f6d090a3b5d,0x3272250920323373,0x0a3b327225202c36\n"
".quad 0x74672e7074657309,0x702509203233752e,0x2c35327225202c32,0x090a3b3632722520\n"
".quad 0x6172622032702540,0x335f305f4c240920,0x5f4c240a3b343730,0x0a3a303333335f30\n"
".quad 0x62646e6557444c24,0x3330325f6b636f6c,0x6f6c2e090a3a315f,0x0936370930330963\n"
".quad 0x732e726162090a30,0x0a3b300920636e79,0x303309636f6c2e09,0x65090a3009383709\n"
".quad 0x444c240a3b746978,0x315a5f5f646e6557,0x654b717369686331,0x315066506c656e72\n"
".quad 0x504d4f4367617431,0x6a5f31533858454c,0x2f207d090a3a666a,0x686331315a5f202f\n"
".quad 0x656e72654b717369,0x617431315066506c,0x58454c504d4f4367,0x0a666a6a5f315338\n"
".quad 0x415744404009200a,0x746365732e204652,0x6265642e206e6f69,0x2c6f666e695f6775\n"
".quad 0x6f7270402c222220,0x40090a7374696267,0x2e20465241574440,0x3378300965747962\n"
".quad 0x2c35307830202c38,0x30202c3030783020,0x307830202c303078,0x0a30307830202c32\n"
".quad 0x4652415744404009,0x0965747962342e20,0x615f67756265642e,0x40090a7665726262\n"
".quad 0x2e20465241574440,0x7830096574796234,0x3830313066323437,0x663234377830202c\n"
".quad 0x7830202c64363037,0x6436303738373636,0x303330337830202c,0x4040090a34376635\n"
".quad 0x342e204652415744,0x3378300965747962,0x2c30333033373336,0x3666353033783020\n"
".quad 0x337830202c343632,0x2c30333033303330,0x3330336432783020,0x444040090a303330\n"
".quad 0x62342e2046524157,0x3836783009657479,0x202c373366353334,0x3337313766357830\n"
".quad 0x65327830202c3936,0x202c373430353535,0x3037303733337830,0x57444040090a3336\n"
".quad 0x7962342e20465241,0x3066327830096574,0x30202c6532393630,0x3666366436353678\n"
".quad 0x3637377830202c38,0x30202c6632623631,0x3635363337663278,0x4157444040090a39\n"
".quad 0x747962342e204652,0x3637643278300965,0x7830202c34363536,0x3436323637366632\n"
".quad 0x333666327830202c,0x7830202c33373237,0x6336313663363337,0x524157444040090a\n"
".quad 0x65747962342e2046,0x3634373536783009,0x367830202c353739,0x2c66326336313663\n"
".quad 0x3633373037783020,0x367830202c393665,0x0a39363237313663,0x4652415744404009\n"
".quad 0x0965747962342e20,0x3337323733367830,0x35367830202c6632,0x202c303066363037\n"
".quad 0x3336333630327830,0x30307830202c6536,0x090a333365323233,0x2046524157444040\n"
".quad 0x783009657479622e,0x30307830202c3430,0x524157444040090a,0x65747962342e2046\n"
".quad 0x5f67756265642e09,0x4040090a656e696c,0x342e204652415744,0x3778300965747962\n"
".quad 0x2c32303537653633,0x3665363536783020,0x367830202c393637,0x2c34363032393665\n"
".quad 0x3037303430783020,0x444040090a343730,0x62342e2046524157,0x3537783009657479\n"
".quad 0x202c333063306562,0x6536343733337830,0x30307830202c3936,0x202c303063303663\n"
".quad 0x3030343063307830,0x57444040090a3030,0x7962342e20465241,0x3030387830096574\n"
".quad 0x30202c6662383730,0x3030303030323078,0x3034307830202c30,0x30202c3332303031\n"
".quad 0x3066623937303078,0x4157444040090a63,0x747962342e204652,0x3030303078300965\n"
".quad 0x7830202c30383030,0x3230333234303130,0x666261377830202c,0x7830202c34306330\n"
".quad 0x3030303830303030,0x524157444040090a,0x65747962342e2046,0x3033323830783009\n"
".quad 0x307830202c303032,0x2c31303030353063,0x3035373936783020,0x307830202c383931\n"
".quad 0x0a65363437333330,0x4652415744404009,0x0965747962342e20,0x3030303030307830\n"
".quad 0x30307830202c3039,0x202c363036633030,0x3730633036647830,0x64367830202c3030\n"
".quad 0x090a313034363936,0x2046524157444040,0x300965747962342e,0x3330306330323178\n"
".quad 0x3038307830202c33,0x30202c3130303030,0x3037643130383778,0x3030307830202c63\n"
".quad 0x40090a3030303830,0x2e20465241574440,0x7830096574796234,0x3030323033323030\n"
".quad 0x633037647830202c,0x7830202c31303830,0x3130393730303038,0x303032307830202c\n"
".quad 0x4040090a30303030,0x342e204652415744,0x3078300965747962,0x2c33323430313038\n"
".quad 0x6431306137783020,0x307830202c633037,0x2c30303038303030,0x3033323830783020\n"
".quad 0x444040090a303032,0x62342e2046524157,0x6330783009657479,0x202c313030303530\n"
".quad 0x3130343639367830,0x39647830202c3065,0x202c643633333030,0x3030303036307830\n"
".quad 0x57444040090a3030,0x7962342e20465241,0x3030307830096574,0x30202c3231313030\n"
".quad 0x3039366536343778,0x3036307830202c32,0x30202c3030353034,0x3231303030303078\n"
".quad 0x4157444040090a34,0x747962342e204652,0x6336663678300965,0x7830202c32303636\n"
".quad 0x3136343730303430,0x303331307830202c,0x7830202c34303930,0x3030303038303530\n"
".quad 0x524157444040090a,0x65747962342e2046,0x3666363537783009,0x307830202c323034\n"
".quad 0x2c32366336353630,0x3032306336783020,0x327830202c343038,0x0a66366536373630\n"
".quad 0x4652415744404009,0x0965747962342e20,0x6636653637367830,0x33377830202c6336\n"
".quad 0x202c303235376536,0x3736653635367830,0x65367830202c3936,0x090a343630323936\n"
".quad 0x2046524157444040,0x300965747962342e,0x3730303730383078,0x3665367830202c34\n"
".quad 0x30202c3230633666,0x3630326336663678,0x3239367830202c37,0x40090a6536373630\n"
".quad 0x2e20465241574440,0x7830096574796234,0x6536343730303530,0x383066327830202c\n"
".quad 0x7830202c38306130,0x3336353734363136,0x353765367830202c,0x4040090a32356636\n"
".quad 0x342e204652415744,0x3678300965747962,0x2c34366434663634,0x3034303765783020\n"
".quad 0x307830202c353630,0x2c31303030303062,0x3333363537783020,0x444040090a383030\n"
".quad 0x62342e2046524157,0x6636783009657479,0x202c343631363235,0x6536343665347830\n"
".quad 0x35367830202c3537,0x202c353631363237,0x3437303030307830,0x57444040090a3337\n"
".quad 0x7962342e20465241,0x3062307830096574,0x30202c3030303030,0x3031333336353778\n"
".quad 0x3566367830202c38,0x30202c3436313632,0x3765363436613578,0x4157444040090a35\n"
".quad 0x747962342e204652,0x6636303078300965,0x7830202c35363237,0x3130303030303030\n"
".quad 0x323333367830202c,0x7830202c62303830,0x3537343631363235,0x524157444040090a\n"
".quad 0x65747962342e2046,0x3765363436783009,0x347830202c663635,0x2c30356636333739\n"
".quad 0x3630303230783020,0x307830202c653636,0x0a30303030303062,0x4652415744404009\n"
".quad 0x0965747962342e20,0x3433333635377830,0x66367830202c3830,0x202c343631363235\n"
".quad 0x6536343664347830,0x65367830202c3537,0x090a393665363934,0x2046524157444040\n"
".quad 0x300965747962342e,0x3630303330303078,0x3032307830202c36,0x30202c3030303030\n"
".quad 0x3666366536373678,0x3634377830202c63,0x40090a3032393665,0x2e20465241574440\n"
".quad 0x7830096574796234,0x3030353038303230,0x313632377830202c,0x7830202c33363836\n"
".quad 0x3030363031303630,0x303030307830202c,0x4040090a33663130,0x342e204652415744\n"
".quad 0x3078300965747962,0x2c39306266313030,0x3035303730783020,0x367830202c303038\n"
".quad 0x2c63303131313036,0x3631363437783020,0x444040090a633666,0x62342e2046524157\n"
".quad 0x3433783009657479,0x202c323330303830,0x3030303038307830,0x38377830202c3230\n"
".quad 0x202c633031313130,0x3033313030307830,0x57444040090a3030,0x7962342e20465241\n"
".quad 0x3230307830096574,0x30202c3030323033,0x3038306330313178,0x3030337830202c31\n"
".quad 0x30202c3130393730,0x3030303030323078,0x4157444040090a31,0x747962342e204652\n"
".quad 0x3130303078300965,0x7830202c33323430,0x3530633030623130,0x663631367830202c\n"
".quad 0x7830202c36366336,0x3437323330303730,0x524157444040090a,0x65747962342e2046\n"
".quad 0x3030306330783009,0x307830202c323030,0x2c30383030303030,0x3532303030783020\n"
".quad 0x307830202c313030,0x0a30306430353030,0x4652415744404009,0x0965747962342e20\n"
".quad 0x3331333635307830,0x35377830202c6530,0x202c663566353336,0x3136323734377830\n"
".quad 0x66367830202c3436,0x090a663539363233,0x2046524157444040,0x300965747962342e\n"
".quad 0x3739366635363678,0x3030307830202c30,0x30202c3030333432,0x3039303432313078\n"
".quad 0x3035307830202c30,0x40090a3030303038,0x2e20465241574440,0x7830096574796234\n"
".quad 0x6330303830303030,0x643732307830202c,0x7830202c30303130,0x3030303064303630\n"
".quad 0x303331307830202c,0x4040090a30303630,0x342e204652415744,0x3378300965747962\n"
".quad 0x2c30303030663030,0x3030303930783020,0x307830202c313030,0x2c30343130303030\n"
".quad 0x3030313330783020,0x444040090a383035,0x62342e2046524157,0x3134783009657479\n"
".quad 0x202c366632353534,0x3433303030337830,0x39307830202c6334,0x202c313030303030\n"
".quad 0x3230303030307830,0x57444040090a6538,0x7962342e20465241,0x3033307830096574\n"
".quad 0x30202c3830353037,0x3131303437313678,0x3464347830202c65,0x30202c3736333466\n"
".quad 0x3563343534383578,0x4157444040090a30,0x747962342e204652,0x3830366478300965\n"
".quad 0x7830202c38333030,0x3230303030303830,0x313032377830202c,0x7830202c33303032\n"
".quad 0x3536303065383230,0x524157444040090a,0x65747962342e2046,0x3032303332783009\n"
".quad 0x307830202c303030,0x2c30303130383033,0x3039366436783020,0x307830202c313231\n"
".quad 0x0a30306538323030,0x4652415744404009,0x0965747962342e20,0x3230333234307830\n"
".quad 0x33307830202c3030,0x202c313030303530,0x3130333466347830,0x35347830202c3332\n"
".quad 0x090a643430356334,0x2046524157444040,0x300965747962342e,0x3538333030326178\n"
".quad 0x3039307830202c38,0x30202c3230303030,0x6432303030303078,0x3131317830202c36\n"
".quad 0x40090a3830353030,0x2e20465241574440,0x7830096574796234,0x3433353739366536\n"
".quad 0x323366357830202c,0x7830202c34373333,0x3437303030383030,0x303133307830202c\n"
".quad 0x4040090a30303030,0x342e204652415744,0x3478300965747962,0x2c65653535393465\n"
".quad 0x3330306565783020,0x307830202c343534,0x2c32303030303032,0x3665363736783020\n"
".quad 0x444040090a633666,0x62342e2046524157,0x3337783009657479,0x202c303235376536\n"
".quad 0x3736653635367830,0x65367830202c3936,0x202c343630323936,0x3030373038307830\n"
".quad 0x57444040090a3437,0x7962342e20465241,0x3030307830096574,0x30202c3930323131\n"
".quad 0x3038303530363078,0x3030307830202c30,0x30202c3032333030,0x3165313933663578\n"
".quad 0x4157444040090a31,0x747962342e204652,0x3133333678300965,0x7830202c61353133\n"
".quad 0x3836393633373137,0x323765367830202c,0x7830202c62343536,0x3536633630353636\n"
".quad 0x524157444040090a,0x65747962342e2046,0x3331333437783009,0x347830202c303531\n"
".quad 0x2c31363736333466,0x3563343534783020,0x337830202c643430,0x0a38353833333531\n"
".quad 0x4652415744404009,0x3009657479622e20,0x367830202c663578,0x2c61367830202c61\n"
".quad 0x30202c3636783020,0x307830202c303078,0x2c31307830202c31,0x40090a3130783020\n"
".quad 0x2e20465241574440,0x3078300965747962,0x4157444040090a30,0x646175712e204652\n"
".quad 0x67656257444c2409,0x6331315a5f5f6e69,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x524157444040090a,0x09646175712e2046\n"
".quad 0x5f646e6557444c24,0x7369686331315a5f,0x506c656e72654b71,0x4367617431315066\n"
".quad 0x533858454c504d4f,0x40090a666a6a5f31,0x2e20465241574440,0x7830096574796234\n"
".quad 0x3632353030303030,0x393337367830202c,0x7830202c32316531,0x6635333638363936\n"
".quad 0x303062397830202c,0x4040090a33373137,0x622e204652415744,0x3230783009657479\n"
".quad 0x202c30307830202c,0x7830202c30307830,0x33307830202c3930,0x524157444040090a\n"
".quad 0x09646175712e2046,0x6170616475635f5f,0x6331315a5f5f6d72,0x6e72654b71736968\n"
".quad 0x7431315066506c65,0x454c504d4f436761,0x666a6a5f31533858,0x71736968635f675f\n"
".quad 0x524157444040090a,0x09657479622e2046,0x7830202c37307830,0x65317830202c3231\n"
".quad 0x202c39337830202c,0x7830202c37367830,0x31377830202c6635,0x090a30307830202c\n"
".quad 0x2046524157444040,0x783009657479622e,0x32307830202c3765,0x202c30307830202c\n"
".quad 0x7830202c30307830,0x33307830202c3930,0x524157444040090a,0x09646175712e2046\n"
".quad 0x6170616475635f5f,0x6331315a5f5f6d72,0x6e72654b71736968,0x7431315066506c65\n"
".quad 0x454c504d4f436761,0x666a6a5f31533858,0x4040090a715f675f,0x342e204652415744\n"
".quad 0x3378300965747962,0x2c37303231653139,0x3534363136783020,0x657830202c373666\n"
".quad 0x2c34373136303037,0x3030303930783020,0x444040090a323030,0x79622e2046524157\n"
".quad 0x0a33307830096574,0x4652415744404009,0x5f09646175712e20,0x726170616475635f\n"
".quad 0x686331315a5f5f6d,0x656e72654b717369,0x617431315066506c,0x58454c504d4f4367\n"
".quad 0x5f666a6a5f315338,0x090a617461645f67,0x2046524157444040,0x300965747962342e\n"
".quad 0x3032316531393378,0x3630357830202c37,0x30202c6536353764,0x3639366536343778\n"
".quad 0x6632307830202c66,0x40090a3337303065,0x2e20465241574440,0x3078300965747962\n"
".quad 0x2c30307830202c30,0x30202c3930783020,0x444040090a333078,0x75712e2046524157\n"
".quad 0x6475635f5f096461,0x5a5f5f6d72617061,0x4b71736968633131,0x5066506c656e7265\n"
".quad 0x4d4f436761743131,0x5f31533858454c50,0x506d756e5f666a6a,0x40090a73746e696f\n"
".quad 0x2e20465241574440,0x7830096574796234,0x3730323165313933,0x643633347830202c\n"
".quad 0x7830202c65363537,0x3836393633373137,0x653633377830202c,0x4040090a32343936\n"
".quad 0x622e204652415744,0x3030783009657479,0x202c65667830202c,0x7830202c32307830\n"
".quad 0x30307830202c3030,0x202c39307830202c,0x4040090a33307830,0x712e204652415744\n"
".quad 0x75635f5f09646175,0x5f5f6d7261706164,0x717369686331315a,0x66506c656e72654b\n"
".quad 0x4f43676174313150,0x31533858454c504d,0x6d756e5f666a6a5f,0x6e69427173696843\n"
".quad 0x4157444040090a73,0x747962342e204652,0x6531393378300965,0x7830202c37303231\n"
".quad 0x3336383639363337,0x663632377830202c,0x7830202c31376534,0x6436303065383230\n"
".quad 0x524157444040090a,0x09657479622e2046,0x7830202c30307830,0x39307830202c3030\n"
".quad 0x090a33307830202c,0x2046524157444040,0x5f5f09646175712e,0x6d72617061647563\n"
".quad 0x69686331315a5f5f,0x6c656e72654b7173,0x6761743131506650,0x3858454c504d4f43\n"
".quad 0x635f666a6a5f3153,0x6d726f4e71736968,0x524157444040090a,0x09657479622e2046\n"
".quad 0x7830202c37307830,0x57444040090a3331,0x6175712e20465241,0x656257444c240964\n"
".quad 0x6b636f6c626e6967,0x090a315f3330325f,0x2046524157444040,0x4c2409646175712e\n"
".quad 0x6f6c62646e655744,0x315f3330325f6b63,0x524157444040090a,0x65747962342e2046\n"
".quad 0x3163336336783009,0x307830202c343165,0x2c30306565323030,0x3030393262783020\n"
".quad 0x307830202c303035,0x0a34653539313032,0x4652415744404009,0x3009657479622e20\n"
".quad 0x444040090a333178,0x75712e2046524157,0x6257444c24096461,0x636f6c626e696765\n"
".quad 0x0a335f3330325f6b,0x4652415744404009,0x2409646175712e20,0x6c62646e6557444c\n"
".quad 0x5f3330325f6b636f,0x4157444040090a33,0x747962342e204652,0x6533613678300965\n"
".quad 0x7830202c34316531,0x3030656532303030,0x303930627830202c,0x7830202c30303630\n"
".quad 0x3265386362613230,0x524157444040090a,0x65747962342e2046,0x3165313034783009\n"
".quad 0x387830202c323034,0x2c38356336303065,0x3030303530783020,0x397830202c323030\n"
".quad 0x0a30393262636335,0x4652415744404009,0x0965747962342e20,0x3230343165317830\n"
".quad 0x30307830202c3130,0x202c313439356336,0x3230303030307830,0x63637830202c6538\n"
".quad 0x090a353030393462,0x2046524157444040,0x300965747962342e,0x3931303230343178\n"
".quad 0x3635367830202c35,0x30202c6531333434,0x3634373136383578,0x3832307830202c63\n"
".quad 0x40090a6336303065,0x2e20465241574440,0x7830096574796234,0x3030303036303039\n"
".quad 0x383962617830202c,0x7830202c38623265,0x3230323034316531,0x353663367830202c\n"
".quad 0x4040090a35343436,0x342e204652415744,0x3678300965747962,0x2c34373136393563\n"
".quad 0x3832303030783020,0x627830202c303065,0x2c30303630303932,0x3962613230783020\n"
".quad 0x444040090a366538,0x79622e2046524157,0x2c32307830096574,0x30202c3030783020\n"
".quad 0x317830202c303078,0x4157444040090a35,0x646175712e204652,0x67656257444c2409\n"
".quad 0x766964665f696e69,0x3330325f66656469,0x57444040090a375f,0x6175712e20465241\n"
".quad 0x6e6557444c240964,0x69766964665f6964,0x5f3330325f666564,0x4157444040090a36\n"
".quad 0x747962342e204652,0x3030303078300965,0x7830202c36323530,0x3631353166303230\n"
".quad 0x303331307830202c,0x7830202c31363030,0x3030303036303039,0x524157444040090a\n"
".quad 0x65747962342e2046,0x6538396261783009,0x307830202c326234,0x2c32303230303030\n"
".quad 0x3036313531783020,0x307830202c303030,0x0a66303230323630,0x4652415744404009\n"
".quad 0x0965747962342e20,0x3130303030307830,0x34657830202c3033,0x202c363030393462\n"
".quad 0x6261323032307830,0x30307830202c3839,0x090a303030303030,0x2046524157444040\n"
".quad 0x783009657479622e,0x35317830202c3030,0x524157444040090a,0x09646175712e2046\n"
".quad 0x6967656257444c24,0x69766964665f696e,0x5f3330325f666564,0x4157444040090a39\n"
".quad 0x646175712e204652,0x646e6557444c2409,0x6469766964665f69,0x385f3330325f6665\n"
".quad 0x524157444040090a,0x65747962342e2046,0x3030303030783009,0x307830202c363235\n"
".quad 0x2c36313531663032,0x3030333130783020,0x397830202c313630,0x0a30303030353030\n"
".quad 0x4652415744404009,0x0965747962342e20,0x6363353931307830,0x30307830202c3862\n"
".quad 0x202c323030303030,0x3631353166307830,0x30337830202c3030,0x090a323032363030\n"
".quad 0x2046524157444040,0x300965747962342e,0x3030303030363078,0x6538397830202c31\n"
".quad 0x30202c3039306232,0x6132303230303078,0x3030307830202c62,0x40090a3030303030\n"
".quad 0x2e20465241574440,0x7830096574796234,0x3030373135316630,0x343639367830202c\n"
".quad 0x7830202c32303636,0x3637393634363536,0x303331307830202c,0x4040090a36363030\n"
".quad 0x622e204652415744,0x3030783009657479,0x202c30307830202c,0x7830202c31307830\n"
".quad 0x31307830202c3130,0x202c30307830202c,0x09200a0a30307830,0x2046524157444040\n"
".quad 0x6e6f69746365732e,0x5f67756265642e20,0x73656d616e627570,0x7270402c2222202c\n"
".quad 0x090a73746962676f,0x2046524157444040,0x783009657479622e,0x30307830202c3833\n"
".quad 0x202c30307830202c,0x7830202c30307830,0x30307830202c3230,0x524157444040090a\n"
".quad 0x65747962342e2046,0x5f67756265642e09,0x4040090a6f666e69,0x342e204652415744\n"
".quad 0x3078300965747962,0x2c63333530303030,0x3030303030783020,0x337830202c633233\n"
".quad 0x2c66356135313331,0x3639363337783020,0x444040090a333638,0x62342e2046524157\n"
".quad 0x3237783009657479,0x202c313762343536,0x3536633630357830,0x31337830202c6536\n"
".quad 0x202c363630353133,0x3136373633347830,0x57444040090a3437,0x7962342e20465241\n"
".quad 0x3563347830096574,0x30202c6634643430,0x3438353833333578,0x3661367830202c35\n"
".quad 0x30202c3133663561,0x3630303030303078,0x4157444040090a36,0x657479622e204652\n"
".quad 0x30202c3030783009,0x4009200a0a303078,0x2e20465241574440,0x206e6f6974636573\n"
".quad 0x615f67756265642e,0x22202c7665726262,0x62676f7270402c22,0x444040090a737469\n"
".quad 0x62342e2046524157,0x3330783009657479,0x202c313031313130,0x6231383035327830\n"
".quad 0x32347830202c3830,0x202c383033316230,0x3031363030307830,0x57444040090a6230\n"
".quad 0x7962342e20465241,0x3230307830096574,0x30202c3030323034,0x3038306533623078\n"
".quad 0x3030307830202c33,0x30202c6230623030,0x3033313130613378,0x4157444040090a33\n"
".quad 0x747962342e204652,0x6230333078300965,0x7830202c62306233,0x3830623062303130\n"
".quad 0x303034307830202c,0x7830202c33313030,0x6430303061336230,0x524157444040090a\n"
".quad 0x65747962342e2046,0x3033303830783009,0x307830202c623362,0x2c39343331383361\n"
".quad 0x3030303030783020,0x337830202c323362,0x0a35303631303061,0x4652415744404009\n"
".quad 0x0965747962342e20,0x6233353033307830,0x30307830202c6230,0x202c383039343331\n"
".quad 0x3630363230307830,0x30307830202c3030,0x090a393433313030,0x2046524157444040\n"
".quad 0x300965747962342e,0x3033313130613378,0x3033307830202c37,0x30202c6230623335\n"
".quad 0x3062306230313078,0x3038307830202c38,0x40090a3331303030,0x2e20465241574440\n"
".quad 0x7830096574796234,0x6430303061336230,0x333038307830202c,0x7830202c62333530\n"
".quad 0x3934333138336130,0x303030307830202c,0x4040090a32336230,0x342e204652415744\n"
".quad 0x3478300965747962,0x2c39306630303039,0x3062303333783020,0x307830202c333162\n"
".quad 0x2c62303030303061,0x3061336230783020,0x444040090a343031,0x62342e2046524157\n"
".quad 0x3830783009657479,0x202c623362303330,0x6230313033317830,0x38327830202c6230\n"
".quad 0x202c303030306230,0x6133623062337830,0x57444040090a3030,0x7962342e20465241\n"
".quad 0x3063317830096574,0x30202c6230333038,0x3030303030633078,0x3433317830202c36\n"
".quad 0x30202c3130313039,0x3363303130333178,0x4157444040090a63,0x747962342e204652\n"
".quad 0x6430313278300965,0x7830202c30303030,0x3030663262303030,0x343330307830202c\n"
".quad 0x7830202c30306530,0x6133623062333530,0x524157444040090a,0x65747962342e2046\n"
".quad 0x3039343331783009,0x337830202c333038,0x2c30303030663035,0x3433313030783020\n"
".quad 0x307830202c303039,0x0a30303031363130,0x4652415744404009,0x0965747962342e20\n"
".quad 0x6230623362307830,0x33317830202c6133,0x202c333038303934,0x3030313165327830\n"
".quad 0x62337830202c3030,0x090a313061336230,0x2046524157444040,0x300965747962342e\n"
".quad 0x3033303830663378,0x3062387830202c62,0x30202c6330373263,0x3463303034613078\n"
".quad 0x3131307830202c30,0x40090a3131313032,0x2e20465241574440,0x7830096574796234\n"
".quad 0x3130333130303030,0x303061337830202c,0x7830202c32313530,0x6230623362303330\n"
".quad 0x333132307830202c,0x4040090a38303934,0x342e204652415744,0x3078300965747962\n"
".quad 0x2c61303333623030,0x3162303130783020,0x307830202c303033,0x2c31313130323131\n"
".quad 0x3034313433783020,0x444040090a303030,0x62342e2046524157,0x6233783009657479\n"
".quad 0x202c303061336230,0x3330383039347830,0x33337830202c6230,0x202c333132306130\n"
".quad 0x3030303035317830,0x57444040090a6230,0x7962342e20465241,0x3131307830096574\n"
".quad 0x30202c6431313031,0x3131303133333178,0x3135307830202c32,0x30202c3030303036\n"
".quad 0x3061336230623378,0x4157444040090a30,0x747962342e204652,0x3830393478300965\n"
".quad 0x7830202c35303330,0x3331323061303333,0x333130307830202c,0x7830202c62303133\n"
".quad 0x3030373165323030,0x524157444040090a,0x65747962342e2046,0x3062333530783009\n"
".quad 0x317830202c613362,0x2c33303830393433,0x3030326230783020,0x307830202c373263\n"
".quad 0x0a62383034633030,0x4652415744404009,0x3009657479622e20,0x307830202c303078\n"
".quad 0x0a30307830202c30,0x000000000000000a\n"
".text");

extern "C" {

extern const unsigned long long __deviceText_$compute_10$[1886];

}

static __cudaFatPtxEntry __ptxEntries [] = {{(char*)"compute_10",(char*)__deviceText_$compute_10$},{0,0}};
static __cudaFatCubinEntry __cubinEntries[] = {{0,0}};
static __cudaFatDebugEntry __debugEntries0 = {0, 0, 0, 0} ;
static __cudaFatElfEntry __elfEntries0 = {0, 0, 0, 0} ;
static __cudaFatElfEntry __elfEntries1 = {(char*)"sm_10", (char*)__deviceText_$sm_10$, &__elfEntries0, (unsigned int)sizeof(__deviceText_$sm_10$)};



static __cudaFatCudaBinary __fatDeviceText __attribute__ ((section (".nvFatBinSegment")))= {0x1ee55a01,0x00000004,0x2e00b786,(char*)"b3eed0eccb295c09",(char*)"Chisq_GPU.cu",(char*)"-v  -g --dont-merge-basicblocks --return-at-end ",__ptxEntries,__cubinEntries,&__debugEntries0,0,0,0,0,0,0x0b5e8760,&__elfEntries1};
# 3 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.stub.c" 2
struct __T20 {REAL4 *__par0;COMPLEX8 *__par1;COMPLEX8 *__par2;UINT4 __par3;UINT4 __par4;REAL4 __par5;int __dummy_field;};
extern void __device_stub__Z11chisqKernelPfP11tagCOMPLEX8S1_jjf(REAL4 *, COMPLEX8 *, COMPLEX8 *, UINT4, UINT4, REAL4);
static void __sti____cudaRegisterAll_44_tmpxft_000076db_00000000_4_Chisq_GPU_cpp1_ii_33f50c82(void) __attribute__((__constructor__));
void __device_stub__Z11chisqKernelPfP11tagCOMPLEX8S1_jjf(REAL4 *__par0, COMPLEX8 *__par1, COMPLEX8 *__par2, UINT4 __par3, UINT4 __par4, REAL4 __par5){ struct __T20 *__T21 = 0;
if (cudaSetupArgument((void*)(char*)&__par0, sizeof(__par0), (size_t)&__T21->__par0) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par1, sizeof(__par1), (size_t)&__T21->__par1) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par2, sizeof(__par2), (size_t)&__T21->__par2) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par3, sizeof(__par3), (size_t)&__T21->__par3) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par4, sizeof(__par4), (size_t)&__T21->__par4) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par5, sizeof(__par5), (size_t)&__T21->__par5) != cudaSuccess) return;{ volatile static char *__f; __f = ((char *)((void ( *)(REAL4 *, COMPLEX8 *, COMPLEX8 *, UINT4, UINT4, REAL4))chisqKernel)); (void)cudaLaunch(((char *)((void ( *)(REAL4 *, COMPLEX8 *, COMPLEX8 *, UINT4, UINT4, REAL4))chisqKernel))); };}
void chisqKernel( REAL4 *__cuda_0,COMPLEX8 *__cuda_1,COMPLEX8 *__cuda_2,UINT4 __cuda_3,UINT4 __cuda_4,REAL4 __cuda_5)
# 58 "Chisq_GPU.cu"
{__device_stub__Z11chisqKernelPfP11tagCOMPLEX8S1_jjf( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 78 "Chisq_GPU.cu"
}
# 1 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.stub.c"
static void __sti____cudaRegisterAll_44_tmpxft_000076db_00000000_4_Chisq_GPU_cpp1_ii_33f50c82(void) { __cudaFatCubinHandle = __cudaRegisterFatBinary((void*)&__fatDeviceText); atexit(__cudaUnregisterBinaryUtil); __cudaRegisterFunction(__cudaFatCubinHandle, (const char*)((void ( *)(REAL4 *, COMPLEX8 *, COMPLEX8 *, UINT4, UINT4, REAL4))chisqKernel), (char*)"_Z11chisqKernelPfP11tagCOMPLEX8S1_jjf", "_Z11chisqKernelPfP11tagCOMPLEX8S1_jjf", -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0); }
# 1 "/tmp/tmpxft_000076db_00000000-1_Chisq_GPU.cudafe1.stub.c" 2

/*
*  Copyright (C) 2007 Jolien Creighton
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

#include <string.h>

#include <lal/LALDatatypes.h>
#include <lal/LALMalloc.h>
#include <lal/LALStatusMacros.h>
#include <lal/XLALError.h>
#include <lal/ComplexFFT.h>
#include <lal/CudaPlan.h>
#include <lal/FFTWMutex.h>
#include <fftw3.h>

#include "CudaFunctions.h"
#include "CudaFFT.h"

NRCSID( COMPLEXFFTC, "$Id$" );

/*
 *
 * XLAL COMPLEX8 functions
 *
 */


COMPLEX8FFTPlan * XLALCreateCOMPLEX8FFTPlan( UINT4 size, int fwdflg, int measurelvl )
{
  static const char *func = "XLALCreateCOMPLEX8FFTPlan";
  COMPLEX8FFTPlan *plan;
  UINT4 createSize;

  if ( ! size )
    XLAL_ERROR_NULL( func, XLAL_EBADLEN );

  /* "use" measurelvl */
  measurelvl = 0;

  /* allocate memory for the plan and the temporary arrays */
  plan = XLALMalloc( sizeof( *plan ) );
  if ( ! plan )
  {
    XLALFree( plan );
    XLAL_ERROR_NULL( func, XLAL_ENOMEM );
  }

  /* create the plan */
  /* LAL_FFTW_PTHREAD_MUTEX_LOCK; */
  /*
   * Change the size to avoid CUDA bug with FFT of size 1
   */
  if( size == 1 ) createSize = 2;
  else createSize = size;
  cufftPlan1d( &plan->plan, createSize, CUFFT_C2C, 1 );
  /* LAL_FFTW_PTHREAD_MUTEX_UNLOCK; */
  /* check to see success of plan creation */
  if ( ! plan->plan )
  {
    XLALFree( plan );
    XLAL_ERROR_NULL( func, XLAL_EFAILED );
  }

  plan->d_input = XLALCudaMallocComplex(size);
  plan->d_output = XLALCudaMallocComplex(size);
  /* now set remaining plan fields */
  plan->size = size;
  plan->sign = ( fwdflg ? -1 : 1 );

  return plan;
}


COMPLEX8FFTPlan * XLALCreateForwardCOMPLEX8FFTPlan( UINT4 size, int measurelvl )
{
  static const char *func = "XLALCreateForwardCOMPLEX8FFTPlan";
  COMPLEX8FFTPlan *plan;
  plan = XLALCreateCOMPLEX8FFTPlan( size, 1, measurelvl );
  if ( ! plan )
    XLAL_ERROR_NULL( func, XLAL_EFUNC );
  return plan;
}


COMPLEX8FFTPlan * XLALCreateReverseCOMPLEX8FFTPlan( UINT4 size, int measurelvl )
{
  static const char *func = "XLALCreateReverseCOMPLEX8FFTPlan";
  COMPLEX8FFTPlan *plan;
  plan = XLALCreateCOMPLEX8FFTPlan( size, 0, measurelvl );
  if ( ! plan )
    XLAL_ERROR_NULL( func, XLAL_EFUNC );
  return plan;
}


void XLALDestroyCOMPLEX8FFTPlan( COMPLEX8FFTPlan *plan )
{
  static const char *func = "XLALDestroyCOMPLEX8FFTPlan";
  if ( ! plan )
    XLAL_ERROR_VOID( func, XLAL_EFAULT );
  if ( ! plan->plan )
    XLAL_ERROR_VOID( func, XLAL_EINVAL );
  //LAL_FFTW_PTHREAD_MUTEX_LOCK;
  cufftDestroy( plan->plan );
  XLALCudaFree(plan->d_input);
  XLALCudaFree(plan->d_output);
  //LAL_FFTW_PTHREAD_MUTEX_UNLOCK;
  memset( plan, 0, sizeof( *plan ) );
  XLALFree( plan );
  return;
}


int XLALCOMPLEX8VectorFFT( COMPLEX8Vector *output, COMPLEX8Vector *input,
    const COMPLEX8FFTPlan *plan )
{
  static const char *func = "XLALCOMPLEX8VectorFFT";
  if ( ! output || ! input || ! plan )
    XLAL_ERROR( func, XLAL_EFAULT );
  if ( ! plan->plan || ! plan->size )
    XLAL_ERROR( func, XLAL_EINVAL );
  if ( ! output->data || ! input->data || output->data == input->data )
    XLAL_ERROR( func, XLAL_EINVAL ); /* note: must be out-of-place */
  if ( output->length != plan->size || input->length != plan->size )
    XLAL_ERROR( func, XLAL_EBADLEN );

  /* do the fft */
  if( plan->size == 1 )
  {
    output->data[0].re = input->data[0].re;
    output->data[0].im = input->data[0].im;
  }
  else
    cudafft_execute_c2c( plan->plan,
	(cufftComplex *)(output->data), (cufftComplex *)(input->data),
	(cufftComplex *)plan->d_output, (cufftComplex *)plan->d_input,
	plan->sign, plan->size );
  return 0;
}



/*
 *
 * XLAL COMPLEX16 functions
 *
 */


COMPLEX16FFTPlan * XLALCreateCOMPLEX16FFTPlan( UINT4 size, int fwdflg, int measurelvl )
{
  static const char *func = "XLALCreateCOMPLEX16FFTPlan";
  COMPLEX16FFTPlan *plan;
  COMPLEX16 *tmp1;
  COMPLEX16 *tmp2;
  int flags = FFTW_UNALIGNED;

  if ( ! size )
    XLAL_ERROR_NULL( func, XLAL_EBADLEN );

  /* based on measurement level, set fftw3 flags to perform
   * requested degree of measurement */
  switch ( measurelvl )
  {
    case 0: /* estimate */
      flags |= FFTW_ESTIMATE;
      break;
    default: /* exhaustive measurement */
      flags |= FFTW_EXHAUSTIVE;
      /* fall-through */
    case 2: /* lengthy measurement */
      flags |= FFTW_PATIENT;
      /* fall-through */
    case 1: /* measure the best plan */
      flags |= FFTW_MEASURE;
      break;
  }

  /* allocate memory for the plan and the temporary arrays */
  plan = XLALMalloc( sizeof( *plan ) );
  tmp1 = XLALMalloc( size * sizeof( *tmp1 ) );
  tmp2 = XLALMalloc( size * sizeof( *tmp2 ) );
  if ( ! plan || ! tmp1 || ! tmp2 )
  {
    XLALFree( plan );
    XLALFree( tmp1 );
    XLALFree( tmp2 );
    XLAL_ERROR_NULL( func, XLAL_ENOMEM );
  }

  /* create the plan */
  LAL_FFTW_PTHREAD_MUTEX_LOCK;
  plan->plan = fftw_plan_dft_1d( size,
      (fftw_complex *)tmp1, (fftw_complex *)tmp2,
      fwdflg ? FFTW_FORWARD : FFTW_BACKWARD, flags );
  LAL_FFTW_PTHREAD_MUTEX_UNLOCK;

  /* free temporary arrays */
  XLALFree( tmp2 );
  XLALFree( tmp1 );

  /* check to see success of plan creation */
  if ( ! plan->plan )
  {
    XLALFree( plan );
    XLAL_ERROR_NULL( func, XLAL_EFAILED );
  }

  /* now set remaining plan fields */
  plan->size = size;
  plan->sign = ( fwdflg ? -1 : 1 );

  return plan;
}


COMPLEX16FFTPlan * XLALCreateForwardCOMPLEX16FFTPlan( UINT4 size, int measurelvl )
{
  static const char *func = "XLALCreateForwardCOMPLEX16FFTPlan";
  COMPLEX16FFTPlan *plan;
  plan = XLALCreateCOMPLEX16FFTPlan( size, 1, measurelvl );
  if ( ! plan )
    XLAL_ERROR_NULL( func, XLAL_EFUNC );
  return plan;
}


COMPLEX16FFTPlan * XLALCreateReverseCOMPLEX16FFTPlan( UINT4 size, int measurelvl )
{
  static const char *func = "XLALCreateReverseCOMPLEX16FFTPlan";
  COMPLEX16FFTPlan *plan;
  plan = XLALCreateCOMPLEX16FFTPlan( size, 0, measurelvl );
  if ( ! plan )
    XLAL_ERROR_NULL( func, XLAL_EFUNC );
  return plan;
}


void XLALDestroyCOMPLEX16FFTPlan( COMPLEX16FFTPlan *plan )
{
  static const char *func = "XLALDestroyCOMPLEX16FFTPlan";
  if ( ! plan )
    XLAL_ERROR_VOID( func, XLAL_EFAULT );
  if ( ! plan->plan )
    XLAL_ERROR_VOID( func, XLAL_EINVAL );
  LAL_FFTW_PTHREAD_MUTEX_LOCK;
  fftw_destroy_plan( plan->plan );
  LAL_FFTW_PTHREAD_MUTEX_UNLOCK;
  memset( plan, 0, sizeof( *plan ) );
  XLALFree( plan );
  return;
}


int XLALCOMPLEX16VectorFFT( COMPLEX16Vector *output, COMPLEX16Vector *input,
    const COMPLEX16FFTPlan *plan )
{
  static const char *func = "XLALCOMPLEX16VectorFFT";
  if ( ! output || ! input || ! plan )
    XLAL_ERROR( func, XLAL_EFAULT );
  if ( ! plan->plan || ! plan->size )
    XLAL_ERROR( func, XLAL_EINVAL );
  if ( ! output->data || ! input->data || output->data == input->data )
    XLAL_ERROR( func, XLAL_EINVAL ); /* note: must be out-of-place */
  if ( output->length != plan->size || input->length != plan->size )
    XLAL_ERROR( func, XLAL_EBADLEN );

  /* do the fft */
  fftw_execute_dft(
      plan->plan,
      (fftw_complex *)input->data,
      (fftw_complex *)output->data
      );
  return 0;
}



/*
 *
 * LAL COMPLEX8 functions
 *
 */



void
LALCreateForwardCOMPLEX8FFTPlan(
    LALStatus       *status,
    COMPLEX8FFTPlan **plan,
    UINT4            size,
    INT4             measure
    )
{
  INITSTATUS( status, "LALCreateForwardCOMPLEX8FFTPlan", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALCreateForwardCOMPLEX8FFTPlan", "XLALCreateForwardCOMPLEX8FFTPlan");

  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( ! *plan, status, COMPLEXFFTH_ENNUL, COMPLEXFFTH_MSGENNUL );
  ASSERT( size > 0, status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );

  *plan = XLALCreateCOMPLEX8FFTPlan( size, 1, measure );
  if ( ! *plan )
  {
    int code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EBADLEN:
        ABORT( status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
      case XLAL_ENOMEM:
        ABORT( status, COMPLEXFFTH_EALOC, COMPLEXFFTH_MSGEALOC );
      case XLAL_EFAILED:
        ABORT( status, COMPLEXFFTH_EFFTW, COMPLEXFFTH_MSGEFFTW );
      default:
        ABORTXLAL( status );
    }
  }

  RETURN( status );
}



void
LALCreateReverseCOMPLEX8FFTPlan(
    LALStatus       *status,
    COMPLEX8FFTPlan **plan,
    UINT4            size,
    INT4             measure
    )
{
  INITSTATUS( status, "LALCreateReverseCOMPLEX8FFTPlan", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALCreateReverseCOMPLEX8FFTPlan", "XLALCreateReverseCOMPLEX8FFTPlan");

  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( ! *plan, status, COMPLEXFFTH_ENNUL, COMPLEXFFTH_MSGENNUL );
  ASSERT( size > 0, status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );

  *plan = XLALCreateCOMPLEX8FFTPlan( size, 0, measure );
  if ( ! *plan )
  {
    int code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EBADLEN:
        ABORT( status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
      case XLAL_ENOMEM:
        ABORT( status, COMPLEXFFTH_EALOC, COMPLEXFFTH_MSGEALOC );
      case XLAL_EFAILED:
        ABORT( status, COMPLEXFFTH_EFFTW, COMPLEXFFTH_MSGEFFTW );
      default:
        ABORTXLAL( status );
    }
  }

  RETURN( status );
}



void
LALDestroyCOMPLEX8FFTPlan (
    LALStatus       *status,
    COMPLEX8FFTPlan **plan
    )
{
  INITSTATUS( status, "LALDestroyCOMPLEX8FFTPlan", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALDestroyCOMPLEX8FFTPlan", "XLALDestroyCOMPLEX8FFTPlan");
  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( *plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  XLALDestroyCOMPLEX8FFTPlan( *plan );
  if ( xlalErrno )
  {
    int code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EINVAL:
        ABORT( status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
      default:
        ABORTXLAL( status );
    }
  }
  RETURN( status );
}



void
LALCOMPLEX8VectorFFT (
    LALStatus      *status,
    COMPLEX8Vector *output,
    COMPLEX8Vector *input,
    COMPLEX8FFTPlan *plan
    )
{
  int code;
  INITSTATUS( status, "LALCOMPLEX8VectorFFT", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALCOMPLEX8VectorFFT", "XLALCOMPLEX8VectorFFT");

  ASSERT( output, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( input, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );

  ASSERT( output->data, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( input->data, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );

  /* make sure that it is not the same data! */
  ASSERT( output->data != input->data, status,
          COMPLEXFFTH_ESAME, COMPLEXFFTH_MSGESAME );

  /* make sure that the lengths agree */
  ASSERT( plan->size > 0, status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
  ASSERT( output->length == plan->size, status,
          COMPLEXFFTH_ESZMM, COMPLEXFFTH_MSGESZMM );
  ASSERT( input->length == plan->size, status,
          COMPLEXFFTH_ESZMM, COMPLEXFFTH_MSGESZMM );

  code = XLALCOMPLEX8VectorFFT( output, input, plan );
  if ( code )
  {
    code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EINVAL:
        if ( ! plan->size ) /* plan size was invalid */
        {
          ABORT( status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
        }
        else if ( output->data == input->data ) /* same data pointers */
        {
          ABORT( status, COMPLEXFFTH_ESAME, COMPLEXFFTH_MSGESAME );
        }
        else /* one of the data pointers was NULL */
        {
          ABORT( status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
        }
      case XLAL_EBADLEN: /* size mismatch */
        ABORT( status, COMPLEXFFTH_ESZMM, COMPLEXFFTH_MSGESZMM );
      default:
        ABORTXLAL( status );
    }
  }

  RETURN( status );
}


/*
 *
 * LAL COMPLEX16 functions
 *
 */



void
LALCreateForwardCOMPLEX16FFTPlan(
    LALStatus       *status,
    COMPLEX16FFTPlan **plan,
    UINT4            size,
    INT4             measure
    )
{
  INITSTATUS( status, "LALCreateForwardCOMPLEX16FFTPlan", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALCreateForwardCOMPLEX16FFTPlan", "XLALCreateForwardCOMPLEX16FFTPlan");

  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( ! *plan, status, COMPLEXFFTH_ENNUL, COMPLEXFFTH_MSGENNUL );
  ASSERT( size > 0, status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );

  *plan = XLALCreateCOMPLEX16FFTPlan( size, 1, measure );
  if ( ! *plan )
  {
    int code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EBADLEN:
        ABORT( status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
      case XLAL_ENOMEM:
        ABORT( status, COMPLEXFFTH_EALOC, COMPLEXFFTH_MSGEALOC );
      case XLAL_EFAILED:
        ABORT( status, COMPLEXFFTH_EFFTW, COMPLEXFFTH_MSGEFFTW );
      default:
        ABORTXLAL( status );
    }
  }

  RETURN( status );
}



void
LALCreateReverseCOMPLEX16FFTPlan(
    LALStatus       *status,
    COMPLEX16FFTPlan **plan,
    UINT4            size,
    INT4             measure
    )
{
  INITSTATUS( status, "LALCreateReverseCOMPLEX16FFTPlan", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALCreateReverseCOMPLEX16FFTPlan", "XLALCreateReverseCOMPLEX16FFTPlan");

  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( ! *plan, status, COMPLEXFFTH_ENNUL, COMPLEXFFTH_MSGENNUL );
  ASSERT( size > 0, status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );

  *plan = XLALCreateCOMPLEX16FFTPlan( size, 0, measure );
  if ( ! *plan )
  {
    int code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EBADLEN:
        ABORT( status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
      case XLAL_ENOMEM:
        ABORT( status, COMPLEXFFTH_EALOC, COMPLEXFFTH_MSGEALOC );
      case XLAL_EFAILED:
        ABORT( status, COMPLEXFFTH_EFFTW, COMPLEXFFTH_MSGEFFTW );
      default:
        ABORTXLAL( status );
    }
  }
  RETURN( status );
}



void
LALDestroyCOMPLEX16FFTPlan (
    LALStatus       *status,
    COMPLEX16FFTPlan **plan
    )
{
  INITSTATUS( status, "LALDestroyCOMPLEX16FFTPlan", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALDestroyCOMPLEX16FFTPlan", "XLALDestroyCOMPLEX16FFTPlan");
  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( *plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  XLALDestroyCOMPLEX16FFTPlan( *plan );
  if ( xlalErrno )
  {
    int code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EINVAL:
        ABORT( status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
      default:
        ABORTXLAL( status );
    }
  }
  RETURN( status );
}



void
LALCOMPLEX16VectorFFT (
    LALStatus      *status,
    COMPLEX16Vector *output,
    COMPLEX16Vector *input,
    COMPLEX16FFTPlan *plan
    )
{
  int code;
  INITSTATUS( status, "LALCOMPLEX16VectorFFT", COMPLEXFFTC );
  XLALPrintDeprecationWarning("LALCOMPLEX16VectorFFT", "XLALCOMPLEX16VectorFFT");

  ASSERT( output, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( input, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( plan, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );

  ASSERT( output->data, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
  ASSERT( input->data, status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );

  /* make sure that it is not the same data! */
  ASSERT( output->data != input->data, status,
          COMPLEXFFTH_ESAME, COMPLEXFFTH_MSGESAME );

  /* make sure that the lengths agree */
  ASSERT( plan->size > 0, status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
  ASSERT( output->length == plan->size, status,
          COMPLEXFFTH_ESZMM, COMPLEXFFTH_MSGESZMM );
  ASSERT( input->length == plan->size, status,
          COMPLEXFFTH_ESZMM, COMPLEXFFTH_MSGESZMM );

  code = XLALCOMPLEX16VectorFFT( output, input, plan );
  if ( code )
  {
    code = xlalErrno;
    XLALClearErrno();
    switch ( code )
    {
      case XLAL_EINVAL:
        if ( ! plan->size ) /* plan size was invalid */
        {
          ABORT( status, COMPLEXFFTH_ESIZE, COMPLEXFFTH_MSGESIZE );
        }
        else if ( output->data == input->data ) /* same data pointers */
        {
          ABORT( status, COMPLEXFFTH_ESAME, COMPLEXFFTH_MSGESAME );
        }
        else /* one of the data pointers was NULL */
        {
          ABORT( status, COMPLEXFFTH_ENULL, COMPLEXFFTH_MSGENULL );
        }
      case XLAL_EBADLEN: /* size mismatch */
        ABORT( status, COMPLEXFFTH_ESZMM, COMPLEXFFTH_MSGESZMM );
      default:
        ABORTXLAL( status );
    }
  }
  RETURN( status );
}

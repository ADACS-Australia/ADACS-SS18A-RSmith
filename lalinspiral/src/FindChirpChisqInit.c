/*
*  Copyright (C) 2007 Duncan Brown, Gareth Jones, Craig Robinson
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

/*-----------------------------------------------------------------------
 *
 * File Name: FindChirpChisqInit.c
 *
 * Author: Anderson, W. G., and Brown, D. A., BCV-Modifications: Messaritaki E.
 *
 *-----------------------------------------------------------------------
 */

/**

\author Anderson, W. G., and Brown D. A., BCV-Modifications: Messaritaki E.
\file
\ingroup FindChirpChisq_h

\brief Module to initialize the \f$\chi^2\f$ veto for the various templates (SP, BCV,etc.)

\heading{Description}

The function <tt>LALFindChirpChisqVetoInit()</tt> takes as input the number of
bins required to contruct the \f$\chi^2\f$ veto and the number of points a data
segment as a parameter. The pointer <tt>*params</tt> must contain the
address of a structure of type \c FindChirpChisqParams for which storage
has already been allocated.  On exit this structure will be populated with the
correct values for execution of the function <tt>LALFindChirpChisqVeto()</tt>.
The workspace arrays and the inverse FFTW plan used by the veto will be
created.

The function <tt>LALFindChirpChisqVetoFinalize()</tt> takes the address of a
structure of type \c FindChirpChisqParams which has been populated by
<tt>LALFindChirpChisqVetoInit()</tt> as input. It takes the number of bins
required to contruct the \f$\chi^2\f$ veto and as a parameter. On exit all memory
allocated by the <tt>LALFindChirpChisqVetoInit()</tt> will be freed.

\heading{Algorithm}

chisq algorithm here

\heading{Uses}
\code
LALCreateReverseComplexFFTPlan()
LALDestroyComplexFFTPlan()
LALCCreateVector()
LALCDestroyVector()
LALCOMPLEX8VectorFFT()
\endcode

\heading{Notes}

*/

#include <stdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/ComplexFFT.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpChisq.h>

void
LALFindChirpChisqVetoInit (
    LALStatus                  *status,
    FindChirpChisqParams       *params,
    UINT4                       numChisqBins,
    UINT4                       numPoints
    )

{
  UINT4                         l, m;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  ASSERT( params, status,
      FINDCHIRPCHISQH_ENULL, FINDCHIRPCHISQH_MSGENULL );

  ASSERT( numPoints > 0, status,
      FINDCHIRPCHISQH_ENUMZ, FINDCHIRPCHISQH_MSGENUMZ );

  ASSERT( ! params->plan, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
  ASSERT( ! params->qtildeBinVec, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
  ASSERT( ! params->qtildeBinVec, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
  ASSERT( ! params->qtildeBinVecBCV, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );

  /* check that we are using a known approximant */
  switch ( params->approximant )
  {
    case TaylorT1:
    case TaylorT2:
    case TaylorT3:
    case TaylorF2:
    case GeneratePPN:
    case FindChirpSP:
    case FindChirpPTF:
    case PadeT1:
    case EOB:
    case EOBNR:
    case EOBNRv2:
    case BCV:
    case BCVSpin:
    case AmpCorPPN:
    case IMRPhenomB:
      break;
    default:
      ABORT( status, FINDCHIRPCHISQH_EUAPX, FINDCHIRPCHISQH_MSGEUAPX );
      break;
  }


  /*
   *
   * if numChisqBins is zero, don't initialize anything
   *
   */


  if ( numChisqBins == 0 )
  {
    DETATCHSTATUSPTR( status );
    RETURN( status );
  }


  /*
   *
   * create storage
   *
   */


  /* create plan for chisq filter */
  LALCreateReverseComplexFFTPlan( status->statusPtr,
      &(params->plan), numPoints, 1 );
  CHECKSTATUSPTR( status );

  /* create one vector for the fourier domain data */
  LALCCreateVector( status->statusPtr,
      &(params->qtildeBinVec), numPoints );
  BEGINFAIL( status )
  {
    TRY( LALDestroyComplexFFTPlan( status->statusPtr,
          &(params->plan) ), status );
  }
  ENDFAIL( status );


  /* create one vector for the additional BCV fourier domain data */
  if ( params->approximant == BCV)
  {
    LALCCreateVector( status->statusPtr,
        &(params->qtildeBinVecBCV), numPoints );
    BEGINFAIL( status )
    {
      TRY( LALDestroyComplexFFTPlan( status->statusPtr,
            &(params->plan) ), status );
      TRY( LALCDestroyVector( status->statusPtr,
            &(params->qtildeBinVec) ), status );
    }
    ENDFAIL( status );
  }


  /* create numBins vectors for the time domain data */
  params->qBinVecPtr = (COMPLEX8Vector **)
    LALCalloc( 1, numChisqBins * sizeof(COMPLEX8Vector*) );
  if ( ! params->qBinVecPtr )
  {
    TRY( LALCDestroyVector( status->statusPtr,
          &(params->qtildeBinVec) ), status );
    if( params->qtildeBinVecBCV )
    {
      TRY( LALCDestroyVector( status->statusPtr,
            &(params->qtildeBinVecBCV) ), status );
    }
    TRY( LALDestroyComplexFFTPlan( status->statusPtr,
          &(params->plan) ), status );
    ABORT( status, FINDCHIRPCHISQH_EALOC, FINDCHIRPCHISQH_MSGEALOC );
  }

  for ( l = 0; l < numChisqBins; ++l )
  {
    LALCCreateVector( status->statusPtr, params->qBinVecPtr + l, numPoints );
    BEGINFAIL( status )
    {
      for ( m = 0; m < l ; ++m )
      {
        TRY( LALCDestroyVector( status->statusPtr,
              params->qBinVecPtr + m ), status );
      }
      LALFree( params->qBinVecPtr );
      TRY( LALCDestroyVector( status->statusPtr,
            &(params->qtildeBinVec) ), status );
      if (params->qtildeBinVecBCV)
      {
        TRY( LALCDestroyVector( status->statusPtr,
              &(params->qtildeBinVecBCV) ), status );
      }
      TRY( LALDestroyComplexFFTPlan( status->statusPtr,
            &(params->plan) ), status );
    }
    ENDFAIL( status );
  }

  /* create additional numBins vectors for the BCV time domain data */
  if ( params->approximant == BCV )
  {
    params->qBinVecPtrBCV = (COMPLEX8Vector **)
      LALCalloc( 1, numChisqBins * sizeof(COMPLEX8Vector*) );
    if ( ! params->qBinVecPtrBCV )
    {
      TRY( LALCDestroyVector( status->statusPtr,
            &(params->qtildeBinVec) ), status );

      if ( params->qtildeBinVecBCV )
      {
        TRY( LALCDestroyVector( status->statusPtr,
              &(params->qtildeBinVecBCV) ), status );
      }
      TRY( LALDestroyComplexFFTPlan( status->statusPtr,
            &(params->plan) ), status );
      ABORT( status, FINDCHIRPCHISQH_EALOC, FINDCHIRPCHISQH_MSGEALOC );
    }

    for ( l = 0; l < numChisqBins; ++l )
    {
      LALCCreateVector(status->statusPtr,params->qBinVecPtrBCV + l, numPoints );
      BEGINFAIL( status )
      {
        for ( m = 0; m < l ; ++m )
        {
          TRY( LALCDestroyVector( status->statusPtr,
                params->qBinVecPtrBCV + m ), status );
        }
        LALFree( params->qBinVecPtrBCV );
        for ( m = 0; m < l ; ++m )
        {
          TRY( LALCDestroyVector( status->statusPtr,
                params->qBinVecPtr + m ), status );
        }
        LALFree( params->qBinVecPtr );
        TRY( LALCDestroyVector( status->statusPtr,
              &(params->qtildeBinVec) ), status );
        if ( params->qtildeBinVecBCV )
        {
          TRY( LALCDestroyVector( status->statusPtr,
                &(params->qtildeBinVecBCV) ), status );
        }
        TRY( LALDestroyComplexFFTPlan( status->statusPtr,
              &(params->plan) ), status );
      }
      ENDFAIL( status );
    }
  }

  /* normal exit */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}




void
LALFindChirpChisqVetoFinalize (
    LALStatus                  *status,
    FindChirpChisqParams       *params,
    UINT4                       numChisqBins
    )

{
  UINT4                         l;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  /* check that we are using a known approximant */
  switch ( params->approximant )
  {
    case TaylorT1:
    case TaylorT2:
    case TaylorT3:
    case TaylorF2:
    case GeneratePPN:
    case FindChirpSP:
    case FindChirpPTF:
    case PadeT1:
    case EOB:
    case EOBNR:
    case EOBNRv2:
    case BCV:
    case BCVSpin:
    case AmpCorPPN:
    case IMRPhenomB:
      break;
    default:
      ABORT( status, FINDCHIRPCHISQH_EUAPX, FINDCHIRPCHISQH_MSGEUAPX );
      break;
  }


  /*
   *
   * if numChisqBins is zero, don't finalize anything
   *
   */


  if ( numChisqBins == 0 )
  {
    DETATCHSTATUSPTR( status );
    RETURN( status );
  }


  /*
   *
   * check arguments
   *
   */


  ASSERT( params, status,
      FINDCHIRPCHISQH_ENULL, FINDCHIRPCHISQH_MSGENULL );
  ASSERT( params->plan, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
  ASSERT( params->qtildeBinVec, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
  ASSERT( params->qtildeBinVec, status,
      FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );

  if ( params->approximant == BCV )
  {
    ASSERT( params->qtildeBinVecBCV, status,
        FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
    ASSERT( params->qtildeBinVecBCV, status,
        FINDCHIRPCHISQH_ENNUL, FINDCHIRPCHISQH_MSGENNUL );
  }


  /*
   *
   * destroy storage
   *
   */


  for ( l = 0; l < numChisqBins; ++l )
  {
    LALCDestroyVector( status->statusPtr, (params->qBinVecPtr + l) );
    CHECKSTATUSPTR( status );
  }

  LALFree( params->qBinVecPtr );

  LALCDestroyVector( status->statusPtr, &(params->qtildeBinVec) );
  CHECKSTATUSPTR( status );

  if ( params->approximant == BCV )
  {
    for ( l = 0; l < numChisqBins; ++l )
    {
      LALCDestroyVector( status->statusPtr, (params->qBinVecPtrBCV + l) );
      CHECKSTATUSPTR( status );
    }

    LALFree( params->qBinVecPtrBCV );

    LALCDestroyVector( status->statusPtr, &(params->qtildeBinVecBCV) );
    CHECKSTATUSPTR( status );
  }

  /* destroy plan for chisq filter */
  LALDestroyComplexFFTPlan( status->statusPtr, &(params->plan) );
  CHECKSTATUSPTR( status );


  /* normal exit */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}

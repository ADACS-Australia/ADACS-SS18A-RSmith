/*
*  Copyright (C) 2007 Andres C. Rodriguez, Sukanta Bose, Chad Hanna, Darren Woods, Diego Fazi, Duncan Brown, Eirini Messaritaki, Gareth Jones, Jolien Creighton, Stephen Fairhurst, Craig Robinson , Sean Seader, Thomas Cokelaer
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
 * File Name: FindChirpFilter.c
 *
 * Author: Brown, D. A.
 *
 *-----------------------------------------------------------------------
 */

/**
 * \author Brown D. A.
 * \file
 * \ingroup FindChirp_h
 *
 * \brief This module provides the core of the matched filter for binary inspiral chirps.
 *
 * \section sec_fcf_match Matched Filtering Using Post-Newtonian Templates
 *
 * The gravitational wave strain induced in an interferometer by a binary
 * inspiral may be written as
 * \f{equation}{
 * \label{eq_rootwaveform}
 * h(t) = \frac{A(t)}{\mathcal{D}} \cos\left( 2 \phi(t) - \theta \right),
 * \f}
 * where
 * \f{equation}{
 * A(t) = - \frac{2G\mu}{c^4} \left[ \pi GM f(t) \right]^\frac{2}{3}
 * \f}
 * and \f$\mathcal{D}\f$ is the <em>effective distance</em>, given by
 * \f{equation}{
 * \mathcal{D} = \frac{r}{\sqrt{F_+^2 (1 + \cos^2 \iota)^2 + F_\times^2 4 \cos^2 \iota}}.
 * \f}
 * The phase angle \f$\theta\f$ is
 * \f{equation}{
 * \tan \theta = \frac{F_\times 2\cos \iota}{F_+(1 + \cos^2 \iota)}
 * \f}
 * and \f$\phi(t)\f$ is the phase evolution of the inspiral waveform.
 *
 */

#include <math.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALConstants.h>
#include <lal/Date.h>
#include <lal/AVFactories.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpChisq.h>
/*#include <lal/FindChirpFilterOutputVeto.h>*/

/* debugging */
extern int vrbflg;                      /* verbocity of lal function    */

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

void
LALFindChirpFilterSegment (
    LALStatus                  *status,
    SnglInspiralTable         **eventList,
    FindChirpFilterInput       *input,
    FindChirpFilterParams      *params
    )

{
  UINT4                 j, k, kmax;
  UINT4                 numPoints;
  UINT4                 deltaEventIndex;
  UINT4                 ignoreIndex;
  REAL8                 deltaT;
  REAL8                 deltaF;
  REAL4                 norm;
  REAL4                 UNUSED modqsqThresh;
  /*BOOLEAN               haveEvent     = 0;*/
  COMPLEX8             *qtilde        = NULL;
  COMPLEX8             *q             = NULL;
  COMPLEX8             *inputData     = NULL;
  COMPLEX8             *tmpltSignal   = NULL;
  /*SnglInspiralTable    *thisEvent     = NULL;*/

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );


  /*
   *
   * check that the arguments are reasonable
   *
   */


  /* make sure the output handle exists, but points to a null pointer */
  ASSERT( eventList, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( !*eventList, status, FINDCHIRPH_ENNUL, FINDCHIRPH_MSGENNUL );

  /* make sure that the parameter structure exists */
  ASSERT( params, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check that the filter parameters are reasonable */
  ASSERT( params->deltaT > 0, status,
      FINDCHIRPH_EDTZO, FINDCHIRPH_MSGEDTZO );
  ASSERT( params->rhosqThresh >= 0, status,
      FINDCHIRPH_ERHOT, FINDCHIRPH_MSGERHOT );
  ASSERT( params->chisqThresh >= 0, status,
      FINDCHIRPH_ECHIT, FINDCHIRPH_MSGECHIT );
  ASSERT( params->chisqDelta >= 0, status,
      FINDCHIRPH_ECHIT, FINDCHIRPH_MSGECHIT );

  /* check that the fft plan exists */
  ASSERT( params->invPlan, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check that the workspace vectors exist */
  ASSERT( params->qVec, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->qVec->data, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->qtildeVec, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->qtildeVec->data, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check that the chisq parameter and input structures exist */
  ASSERT( params->chisqParams, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->chisqInput, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* if a rhosqVec vector has been created, check we can store data in it */
  if ( params->rhosqVec )
  {
    ASSERT( params->rhosqVec->data->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
    ASSERT( params->rhosqVec->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  }

  if ( params->cVec )
  {
    ASSERT( params->cVec->data->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL  );
    ASSERT( params->cVec->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  }

  /* if we are doing a chisq, check we can store the data */
  if ( input->segment->chisqBinVec->length )
  {
    ASSERT( params->chisqVec, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
    ASSERT( params->chisqVec->data, status,
        FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  }

  /* make sure that the input structure exists */
  ASSERT( input, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* make sure that the input structure contains some input */
  ASSERT( input->fcTmplt, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( input->segment, status, FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  /* check the allowed approximants */
  switch ( params->approximant )
  {
    case TaylorT1:
    case TaylorT2:
    case TaylorT3:
    case TaylorF2:
    case GeneratePPN:
    case PadeT1:
    case EOB:
    case EOBNR:
    case EOBNRv2:
    case FindChirpSP:
    case IMRPhenomB:
      break;

    default:
      ABORT( status, FINDCHIRPH_EUAPX, FINDCHIRPH_MSGEUAPX );
      break;
  }

  /* make sure the approximant in the tmplt and segment agree */
  if ( params->approximant != input->fcTmplt->tmplt.approximant ||
      params->approximant != input->segment->approximant )
  {
    ABORT( status, FINDCHIRPH_EAPRX, FINDCHIRPH_MSGEAPRX );
  }


  /*
   *
   * point local pointers to input and output pointers
   *
   */


  /* workspace vectors */
  q = params->qVec->data;
  qtilde = params->qtildeVec->data;

  /* number of points in a segment */
  numPoints = params->qVec->length;

  /* template and data */
  inputData = input->segment->data->data->data;
  tmpltSignal = input->fcTmplt->data->data;
  deltaT = params->deltaT;
  deltaF = 1.0 / ( params->deltaT * (REAL8) numPoints );
  kmax = input->fcTmplt->tmplt.fFinal / deltaF < numPoints/2 ?
    input->fcTmplt->tmplt.fFinal / deltaF : numPoints/2;


  /*
   *
   * compute viable search regions in the snrsq vector
   *
   */


  if ( input->fcTmplt->tmplt.tC <= 0 )
  {
    ABORT( status, FINDCHIRPH_ECHTZ, FINDCHIRPH_MSGECHTZ );
  }

  deltaEventIndex = (UINT4) rint( (input->fcTmplt->tmplt.tC / deltaT) + 1.0 );

  /* ignore corrupted data at start and end */
  params->ignoreIndex = ignoreIndex =
    ( input->segment->invSpecTrunc / 2 ) + deltaEventIndex;

  if ( lalDebugLevel & LALINFO )
  {
    CHAR infomsg[256];

    snprintf( infomsg, XLAL_NUM_ELEM(infomsg),
        "m1 = %e m2 = %e => %e seconds => %d points\n"
        "invSpecTrunc = %d => ignoreIndex = %d\n",
        input->fcTmplt->tmplt.mass1, input->fcTmplt->tmplt.mass2,
        input->fcTmplt->tmplt.tC , deltaEventIndex,
        input->segment->invSpecTrunc, ignoreIndex );
    LALInfo( status, infomsg );
  }

  /* XXX check that we are not filtering corrupted data XXX */
  /* XXX this is hardwired to 1/4 segment length        XXX */
  if ( ignoreIndex > numPoints / 4 )
  {
    ABORT( status, FINDCHIRPH_ECRUP, FINDCHIRPH_MSGECRUP );
  }
  /* XXX reset ignoreIndex to one quarter of a segment XXX */
  params->ignoreIndex = ignoreIndex = numPoints / 4;

  if ( lalDebugLevel & LALINFO )
  {
    CHAR infomsg[256];

    snprintf( infomsg, XLAL_NUM_ELEM(infomsg),
        "filtering from %d to %d\n",
        ignoreIndex, numPoints - ignoreIndex );
    LALInfo( status, infomsg );
  }


  /*
   *
   * compute qtilde and q
   *
   */


  memset( qtilde, 0, numPoints * sizeof(COMPLEX8) );
  /* qtilde positive frequency, not DC or nyquist */
  for ( k = 1; k < kmax; ++k )
  {
    REAL4 r = crealf(inputData[k]);
    REAL4 s = cimagf(inputData[k]);
    REAL4 x = crealf(tmpltSignal[k]);
    REAL4 y = 0 - cimagf(tmpltSignal[k]);       /* note complex conjugate */

    qtilde[k] = crectf( r*x - s*y, r*y + s*x );
  }


  /* inverse fft to get q */
  LALCOMPLEX8VectorFFT( status->statusPtr, params->qVec, params->qtildeVec,
      params->invPlan );
  CHECKSTATUSPTR( status );


  /*
   *
   * calculate signal to noise squared
   *
   */


  /* if full snrsq vector is required, set it to zero */
  if ( params->rhosqVec )
    memset( params->rhosqVec->data->data, 0, numPoints * sizeof( REAL4 ) );

  if (params->cVec )
    memset( params->cVec->data->data, 0, numPoints * sizeof( COMPLEX8 ) );

  /* normalisation */
  input->fcTmplt->norm = norm =
    4.0 * (deltaT / (REAL4)numPoints) / input->segment->segNorm->data[kmax];

  /* normalised snr threhold */
  modqsqThresh = params->rhosqThresh / norm;

  /* if full snrsq vector is required, store the snrsq */
  if ( params->rhosqVec )
  {
    memcpy( params->rhosqVec->name, input->segment->data->name,
        LALNameLength * sizeof(CHAR) );
    memcpy( &(params->rhosqVec->epoch), &(input->segment->data->epoch),
        sizeof(LIGOTimeGPS) );
    params->rhosqVec->deltaT = input->segment->deltaT;

    for ( j = 0; j < numPoints; ++j )
    {
      REAL4 modqsq = crealf(q[j]) * crealf(q[j]) + cimagf(q[j]) * cimagf(q[j]);
      params->rhosqVec->data->data[j] = norm * modqsq;
    }
  }

  if ( params->cVec )
  {
    memcpy( params->cVec->name, input->segment->data->name,
        LALNameLength * sizeof(CHAR) );
    memcpy( &(params->cVec->epoch), &(input->segment->data->epoch),
        sizeof(LIGOTimeGPS) );
    params->cVec->deltaT = input->segment->deltaT;

    for ( j = 0; j < numPoints; ++j )
    {
      params->cVec->data->data[j] = (((REAL4) sqrt(norm)) * q[j]);
    }
  }


  #if 0
  /* This is done in FindChirpClusterEvents now!!*/
  /* determine if we need to compute the chisq vector */
  if ( input->segment->chisqBinVec->length )
  {
    /* look for an event in the filter output */
    for ( j = ignoreIndex; j < numPoints - ignoreIndex; ++j )
    {
      REAL4 modqsq = q[j].re * q[j].re + q[j].im * q[j].im;

      /* if snrsq exceeds threshold at any point */
      if ( modqsq > modqsqThresh )
      {
        haveEvent = 1;        /* mark segment to have events    */
        break;
      }
    }
    if ( haveEvent )
    {
      /* compute the chisq vector for this segment */
      memset( params->chisqVec->data, 0,
          params->chisqVec->length * sizeof(REAL4) );

      /* pointers to chisq input */
      params->chisqInput->qtildeVec = params->qtildeVec;
      params->chisqInput->qVec      = params->qVec;

      /* pointer to the chisq bin vector in the segment */
      params->chisqParams->chisqBinVec = input->segment->chisqBinVec;
      params->chisqParams->norm        = norm;

      /* compute the chisq bin boundaries for this template */
      if ( ! params->chisqParams->chisqBinVec->data )
      {
        LALFindChirpComputeChisqBins( status->statusPtr,
            params->chisqParams->chisqBinVec, input->segment, kmax );
        CHECKSTATUSPTR( status );
      }

      /* compute the chisq threshold: this is slow! */
      LALFindChirpChisqVeto( status->statusPtr, params->chisqVec,
          params->chisqInput, params->chisqParams );
      CHECKSTATUSPTR (status);
    }
  }
  #endif

  (void)eventList;

  /* normal exit */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}

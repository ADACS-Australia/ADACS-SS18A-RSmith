#if 0 
<lalVerbatim file="FindChirpACTDTemplateCV">
Author: Brown, D. A., Creighton, J. D. E. and Mckechan, D. J. A.
$Id$
</lalVerbatim> 

<lalLaTeX>
\subsection{Module \texttt{FindChirpACTDTemplate.c}}
\label{ss:FindChirpACTDTemplate.c}

Provides functions to create time domain inspiral templates in a
form that can be used by the \texttt{FindChirpACTDFilter()} function.

\subsubsection*{Prototypes}
\vspace{0.1in}
\input{FindChirpACTDTemplateCP}
\idx{LALFindChirpACTDTemplate()}
\idx{LALFindChirpACTDNormalize()}

The function \texttt{LALFindChirpACTDTemplate()} creates a time 
domain template using LALGeneratePPNAmpCorInspiral().

\subsubsection*{Algorithm}

Blah.

\subsubsection*{Uses}
\begin{verbatim}
LALCalloc()
LALFree()
LALCreateVector()
LALDestroyVector()
\end{verbatim}

\subsubsection*{Notes}

\vfill{\footnotesize\input{FindChirpACTDTemplateCV}}
</lalLaTeX> 
#endif

#include <math.h>
#include <lal/LALStdlib.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <lal/DataBuffer.h>
#include <lal/LALInspiral.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpTD.h>
#include <lal/FindChirpACTD.h>

NRCSID (FINDCHIRPACTDTEMPLATEC, "$Id$");

/* <lalVerbatim file="FindChirpACTDTemplateCP"> */
void
LALFindChirpACTDTemplate (
    LALStatus                  *status,
    FindChirpTemplate          *fcTmplt,
    InspiralTemplate           *tmplt,
    FindChirpTmpltParams       *params
    )
/* </lalVerbatim> */
{
  UINT4          i, j;
  UINT4          shift;
  UINT4          numPoints;
  REAL4Vector    ACTDVecs[NACTDVECS];
  COMPLEX8Vector ACTDtilde[NACTDVECS];
  REAL8          deltaT;
  REAL8          sampleRate;
  const REAL4    cannonDist = 1.0; /* Mpc */
  CHAR           infomsg[512];
  PPNParamStruc  ppnParams;
  CoherentGW     waveform;

  REAL4Vector  *tmpACTDVec = NULL; /* Used for band-passing */


  INITSTATUS( status, "LALFindChirpACTDTemplate", FINDCHIRPACTDTEMPLATEC );
  ATTATCHSTATUSPTR( status );


  /*
   *
   * check that the arguments are reasonable
   *
   */
  

  /* check that the output structures exist */
  ASSERT( fcTmplt, status, 
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( fcTmplt->ACTDtilde, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( fcTmplt->ACTDtilde->length == NACTDVECS, status, 
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( fcTmplt->ACTDtilde->data, status, 
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  /* check that the parameter structure exists */
  ASSERT( params, status, 
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( params->ACTDVecs, status, 
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( params->ACTDVecs->length == NACTDVECS, status, 
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( params->ACTDVecs->data, status, 
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );


  /* check we have an fft plan for the template */
  ASSERT( params->fwdPlan, status, 
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  /* check that the timestep is positive */
  ASSERT( params->deltaT > 0, status, 
      FINDCHIRPTDH_EDELT, FINDCHIRPTDH_MSGEDELT );

  /* check that the input exists */
  ASSERT( tmplt, status, FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  /* check that the template masses are not equal */
  if( tmplt->mass1 == tmplt->mass2 )
  {
    ABORT( status, FINDCHIRPACTDH_EQMAS, FINDCHIRPACTDH_MSGEQMAS );
  }

  /* check that the  approximant is AmpCorPPN */
  if( params->approximant != AmpCorPPN )
  {
    ABORT( status, FINDCHIRPTDH_EMAPX, FINDCHIRPTDH_MSGEMAPX );
  }  

  /* store deltaT and zero out the time domain waveform vector */
  deltaT = params->deltaT;
  sampleRate = 1.0 / deltaT;
  numPoints =  params->ACTDVecs->vectorLength;

  ASSERT( numPoints == (2 * (fcTmplt->ACTDtilde->vectorLength - 1)), status,
      FINDCHIRPTDH_EMISM, FINDCHIRPTDH_MSGEMISM );


  /*
   *
   * generate the waveform using LALGeneratePPNAmpCorInspiral() from inject
   *
   */



  /* input parameters */
  memset( &ppnParams, 0, sizeof(PPNParamStruc) );
  ppnParams.deltaT = deltaT;
  ppnParams.mTot = tmplt->mass1 + tmplt->mass2;
  ppnParams.eta = tmplt->mass1 * tmplt->mass2 /
    ( ppnParams.mTot * ppnParams.mTot );
  /* Set distance at 20Mpc for testing, will be normalised anyway */
  ppnParams.d = 1.0 * 1000. * 1000. * LAL_PC_SI;
  ppnParams.fStartIn = params->fLow;
  ppnParams.fStopIn = - 1.0 /
                    (6.0 * sqrt(6.0) * LAL_PI * ppnParams.mTot * LAL_MTSUN_SI);

  /* ACTD specific */
  ppnParams.inc = LAL_PI_4; 
  ppnParams.ampOrder = ( INT4 )( tmplt->ampOrder );

  /* generate waveform amplitude and phase */
  memset( &waveform, 0, sizeof(CoherentGW) );
  LALGeneratePPNAmpCorInspiral( status->statusPtr, &waveform, &ppnParams );
  CHECKSTATUSPTR( status );

  /* print termination information and check sampling */
  LALInfo( status, ppnParams.termDescription );
  if ( ppnParams.dfdt > 2.0 )
  {
  /* ABORT( status, FINDCHIRPTDH_ESMPL, FINDCHIRPTDH_MSGESMPL ); */
  }
  if ( waveform.a->data->length > numPoints )
  {
    ABORT( status, FINDCHIRPTDH_ELONG, FINDCHIRPTDH_MSGELONG );
  }

  memset( params->ACTDVecs->data, 0, NACTDVECS * numPoints * sizeof( REAL4 ) );
  memset( fcTmplt->ACTDtilde->data, 0, 
                        NACTDVECS * (numPoints / 2 + 1) * sizeof( COMPLEX8 ) ); 

  for( j=0; j<NACTDVECS; ++j )
  {
    ACTDVecs[j].length  = numPoints;
    ACTDtilde[j].length = numPoints / 2 + 1;
    ACTDVecs[j].data  = params->ACTDVecs->data + (j * numPoints);
    ACTDtilde[j].data = fcTmplt->ACTDtilde->data + (j * (numPoints / 2 + 1 ) );
  }

  /* compute h(t) */
  /* legacy - length is the lentgh of the vectors */
  for ( i = 0; i < waveform.a->data->length; ++i )
  {
    for ( j = 0; j < NACTDVECS; ++j )
    {
      ACTDVecs[j].data[i] = waveform.a->data->data[3*i + j]
         * cos( ( ( REAL4 )( j ) + 1.0 ) / 2.0 * waveform.phi->data->data[i] );
    }
  }


  /* free the memory allocated by LALGeneratePPNAmpCorInspiral() */
  LALSDestroyVectorSequence( status->statusPtr, &(waveform.h->data) );
  CHECKSTATUSPTR( status );

  LALSDestroyVectorSequence( status->statusPtr, &(waveform.a->data) );
  CHECKSTATUSPTR( status );

  LALSDestroyVector( status->statusPtr, &(waveform.f->data) );
  CHECKSTATUSPTR( status );

  LALDDestroyVector( status->statusPtr, &(waveform.phi->data) );
  CHECKSTATUSPTR( status );

  LALFree( waveform.h );
  LALFree( waveform.a );
  LALFree( waveform.f );
  LALFree( waveform.phi );
    
  /* waveform parameters needed for findchirp filter */
  tmplt->approximant = params->approximant;
  tmplt->tC = ppnParams.tc;
  tmplt->fFinal = ppnParams.fStop;
    
  fcTmplt->tmpltNorm = params->dynRange / ( cannonDist * 1.0e6 * LAL_PC_SI );
  fcTmplt->tmpltNorm *= fcTmplt->tmpltNorm;

  /* 
   *
   * Loop over each template and apply tapering/band passing etc
   *
   */
    
  /* Create a temporary vector to avoid mishaps */
  if( ( tmpACTDVec = XLALCreateREAL4Vector( numPoints ) ) == NULL )
  {
    ABORTXLAL( status );
  }
  
  for( i = 0; i < NACTDVECS; i++ )
  {
    memcpy( tmpACTDVec->data, ACTDVecs[i].data,  
               numPoints * sizeof( ACTDVecs[i].data ) );

    /* Taper the waveform if required */
    if ( params->taperTmplt != INSPIRAL_TAPER_NONE )
    {
      if ( XLALInspiralWaveTaper( tmpACTDVec, params->taperTmplt )
             == XLAL_FAILURE )
      {
        ABORTXLAL( status );
      }
    }


    /* Find the end of the chirp */
    j = numPoints - 1;
    while ( tmpACTDVec->data[j] == 0 )
    {
      /* search for the end of the chirp but don't fall off the array */
      if ( --j == 0 )
      {
        ABORT( status, FINDCHIRPTDH_EEMTY, FINDCHIRPTDH_MSGEEMTY );
      }
    }
    ++j;

    if ( params->bandPassTmplt )
    {
      REAL4Vector bpVector; /*Used to save time */
      REAL4 bpfLow;
      REAL4 bpfFinal;

      /* We want to shift the template to the middle of the vector so */
      /* that band-passing will work properly */
      shift = ( numPoints - j ) / 2;
      memmove( tmpACTDVec->data + shift, tmpACTDVec->data, 
                                       j * sizeof( tmpACTDVec->data ) );

      memset( tmpACTDVec->data, 0, shift * sizeof( tmpACTDVec->data ) );
      memset( tmpACTDVec->data + ( numPoints + j ) / 2, 0,
         ( numPoints - ( numPoints + j ) / 2 ) * sizeof( tmpACTDVec->data ) );

      /* Select an appropriate part of the vector to band pass. */
      /* band passing the whole thing takes a lot of time */
      if ( j > 2 * sampleRate && 2 * j <= numPoints )
      {
        bpVector.length = 2 * j;
        bpVector.data   = tmpACTDVec->data + numPoints / 2 - j;
      }
      else if ( j <= 2 * sampleRate && j + 2 * sampleRate <= numPoints )
      {
        bpVector.length = j + 2 * sampleRate;
        bpVector.data   = tmpACTDVec->data
                   + ( numPoints - j ) / 2 - (INT4)sampleRate;
      }
      else
      {
        bpVector.length = numPoints;
        bpVector.data   = tmpACTDVec->data;
      }
      
     /* Adjust frequencies according to the harmonic */
      bpfLow = 0.98 * tmplt->fLower ;
      bpfFinal = 1.02 * tmplt->fFinal * ( ( REAL4 )( i ) + 1. ) / 2.;

     /* If the harmonic is not in our bandwidth we mega band pass it */
     if( bpfLow >= bpfFinal )
     {
        bpfLow = 0.98 * tmplt->fLower;
        bpfFinal = 1.02* ( tmplt->fLower + 1. );
     }


      if ( XLALBandPassInspiralTemplate( &bpVector, bpfLow,
                   tmplt->fFinal, sampleRate ) == XLAL_FAILURE )
      {
        ABORTXLAL( status );
      }

      /* Now we need to do the shift to the end. */
      memcpy( tmpACTDVec->data, tmpACTDVec->data + ( numPoints + j ) / 2,
         ( numPoints - ( numPoints + j ) / 2 )  * sizeof( tmpACTDVec->data ) );
      memcpy( tmpACTDVec->data + numPoints - ( numPoints + j ) / 2,
         tmpACTDVec->data, ( numPoints + j ) /2 * sizeof( tmpACTDVec->data ) );

    }
    else
    {
      /* No need for so much shifting around if not band passing */
      /* shift chirp to end of vector so it is the correct place for filter */
        memmove( tmpACTDVec->data + numPoints - j, tmpACTDVec->data, 
                                             j * sizeof( tmpACTDVec->data ) );
        memset( tmpACTDVec->data, 0, 
                             ( numPoints - j ) * sizeof( tmpACTDVec->data ) );
    }

    memcpy( ACTDVecs[i].data, tmpACTDVec->data, 
                            numPoints * sizeof( tmpACTDVec->data ) );

  }

  XLALDestroyREAL4Vector( tmpACTDVec );
  tmpACTDVec = NULL;


  /*
   *
   * create the frequency domain findchirp templates
   *
   */

  /* fft harmonics */
  for( i = 0; i < NACTDVECS; ++i)
  {
    if ( XLALREAL4ForwardFFT( &ACTDtilde[i], &ACTDVecs[i], 
         params->fwdPlan ) == XLAL_FAILURE )
    {
      ABORTXLAL( status );
    }
  }

  /* copy the template parameters to the findchirp template structure */
  memcpy( &(fcTmplt->tmplt), tmplt, sizeof(InspiralTemplate) );

  /* print the template normalization constant */
  if ( lalDebugLevel & LALINFO )
  {
    LALSnprintf( infomsg, sizeof(infomsg) / sizeof(*infomsg), 
        "tmpltNorm = %e\n", fcTmplt->tmpltNorm );
    LALInfo( status, infomsg );
  }


  /* normal exit */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}


/* <lalVerbatim file="FindChirpACTDTemplate"> */
void
LALFindChirpACTDNormalize(
    LALStatus                  *status,
    FindChirpTemplate          *fcTmplt,
    FindChirpTmpltParams       *tmpltParams,
    FindChirpDataParams        *params
    )
/* </lalVerbatim> */
{
  UINT4          k;
  UINT4          numPoints;
  REAL8          deltaF; 
  COMPLEX8Vector ACTDtilde[NACTDVECS];
  COMPLEX8      *wtilde;
 
  /* Inner Products */
  /* Dominant 2nd Harmonic */
  REAL4   H2H2;
  /* 1st Harmonic */
  REAL4   H1H1, h1H2;
  /* 3rd Harmonic */
  REAL4   H3H3, h3H1, h3H2;

  REAL4   norm;

  INITSTATUS( status, "LALFindChirpACTDNormalize", FINDCHIRPACTDTEMPLATEC );

  /* check the required input exists */
  ASSERT( fcTmplt, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( tmpltParams, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  ASSERT( params, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  ASSERT( params->wtildeVec, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( params->wtildeVec->data, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  ASSERT( params->tmpltPowerVec, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );
  ASSERT( params->tmpltPowerVec->data, status,
      FINDCHIRPTDH_ENULL, FINDCHIRPTDH_MSGENULL );

  /* check that we have the correct approximant */
  if( params->approximant != AmpCorPPN )
  {
    ABORT( status, FINDCHIRPTDH_EMAPX, FINDCHIRPTDH_MSGEMAPX );
  }


  wtilde = params->wtildeVec->data;

  numPoints = fcTmplt->ACTDtilde->vectorLength;
 
  deltaF = 1. / ( (REAL8)(numPoints) * tmpltParams->deltaT );

  for( k=0; k<NACTDVECS; ++k )
  {
    ACTDtilde[k].length = numPoints;
    ACTDtilde[k].data  = fcTmplt->ACTDtilde->data + (k * numPoints );
  }



   
  /* 
   * Start with dominant harmonic. 
   * Calculate H2 
   */
  H2H2 = XLALFindChirpACTDInnerProduct( &ACTDtilde[1], &ACTDtilde[1], 
                                        wtilde, tmpltParams->fLow, deltaF );
  norm = pow( H2H2, -0.5 );
  for( k = 0; k < numPoints; ++k )
  {
    ACTDtilde[1].data[k].re *= norm; 
    ACTDtilde[1].data[k].im *= norm; 
  }


  /* 
   * Calculate H1 
   */
  h1H2 = XLALFindChirpACTDInnerProduct( &ACTDtilde[1], &ACTDtilde[0],
                                      wtilde, tmpltParams->fLow, deltaF );
  for( k = 0; k < numPoints; ++k )
  {
    ACTDtilde[0].data[k].re -=  h1H2 * ACTDtilde[1].data[k].re;
    ACTDtilde[0].data[k].im -=  h1H2 * ACTDtilde[1].data[k].im;
  }
  H1H1 = XLALFindChirpACTDInnerProduct( &ACTDtilde[0], &ACTDtilde[0],
                                      wtilde, tmpltParams->fLow, deltaF );
  norm = pow( H1H1, -0.5 );
  for( k = 0; k < numPoints; ++k )
  {
    ACTDtilde[0].data[k].re *=  norm;                  
    ACTDtilde[0].data[k].im *=  norm;                  
  }

  /*
   * Calculate H3
   */
  h3H1 = XLALFindChirpACTDInnerProduct( &ACTDtilde[0], &ACTDtilde[2],
                                      wtilde, tmpltParams->fLow, deltaF );
  h3H2 = XLALFindChirpACTDInnerProduct( &ACTDtilde[1], &ACTDtilde[2],
                                      wtilde, tmpltParams->fLow, deltaF );
  for( k = 0; k < numPoints; ++k )
  {
    ACTDtilde[2].data[k].re -=  h3H1 * ACTDtilde[0].data[k].re; 
    ACTDtilde[2].data[k].re -=  h3H2 * ACTDtilde[1].data[k].re; 
    ACTDtilde[2].data[k].im -=  h3H1 * ACTDtilde[0].data[k].im;
    ACTDtilde[2].data[k].im -=  h3H2 * ACTDtilde[1].data[k].im; 
  }
  H3H3 = XLALFindChirpACTDInnerProduct( &ACTDtilde[2], &ACTDtilde[2],
                                      wtilde, tmpltParams->fLow, deltaF );
  norm = pow( H3H3, -0.5 );
  for( k = 0; k < numPoints; ++k )
  {
    ACTDtilde[2].data[k].re *=  norm;                  
    ACTDtilde[2].data[k].im *=  norm;                  
  }

 /* XXX UNCOMMENT BELOW TO TEST ORTHONORMALISATION XXX */

  H1H1 = XLALFindChirpACTDInnerProduct( &ACTDtilde[0], &ACTDtilde[0],
                                      wtilde, tmpltParams->fLow, deltaF );
  H2H2 = XLALFindChirpACTDInnerProduct( &ACTDtilde[1], &ACTDtilde[1],
                                      wtilde, tmpltParams->fLow, deltaF );
  H3H3 = XLALFindChirpACTDInnerProduct( &ACTDtilde[2], &ACTDtilde[2],
                                      wtilde, tmpltParams->fLow, deltaF );

  fprintf( stderr, "\n\n  H1H1 = %.4f  H2H2 = %.4f  H3H3 = %.4f\n", 
                                          H1H1, H2H2, H3H3 );
  
  h1H2 = XLALFindChirpACTDInnerProduct( &ACTDtilde[1], &ACTDtilde[0],
                                      wtilde, tmpltParams->fLow, deltaF );
  h3H1 = XLALFindChirpACTDInnerProduct( &ACTDtilde[2], &ACTDtilde[0],
                                      wtilde, tmpltParams->fLow, deltaF );
  h3H2 = XLALFindChirpACTDInnerProduct( &ACTDtilde[2], &ACTDtilde[1],
                                      wtilde, tmpltParams->fLow, deltaF );

  fprintf( stderr, "  h1H2 = %.4f  h3H1 = %.4f  h3H2 = %.4f\n", 
                                          h1H2, h3H1, h3H2 );
  fprintf( stderr, "                                        " );
  /* XXX UNCOMMENT ABOVE TO TEST ORTHONORMALIZATION XXX */

  /* normal exit */
  RETURN( status );

}

REAL4  XLALFindChirpACTDInnerProduct(
               COMPLEX8Vector  *a,
               COMPLEX8Vector  *b,
               COMPLEX8        *wtilde,
               REAL4            flower,
               REAL4            deltaF
                                    )
{
  INT4  k;
  REAL4 innerProduct;
  REAL8 sum = 0.0;
  flower = 0.0;

  for( k = 0; k < a->length; ++k )
  {
    if(  k * deltaF  >= flower )
    {
      REAL4 power;
      power = a->data[k].re * b->data[k].re;
      power += a->data[k].im * b->data[k].im;
      sum += power * wtilde[k].re;
    }
  }

  innerProduct = 4.0 * (REAL4)(sum) * deltaF;

  return innerProduct;

}

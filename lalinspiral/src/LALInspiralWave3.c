/*
*  Copyright (C) 2007 Stas Babak, David Churches, B.S. Sathyaprakash, Craig Robinson , Thomas Cokelaer
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

/**

\author Sathyaprakash, B. S.
\file
\ingroup LALInspiral_h

\brief These modules generate a time-domain chirp waveform of type #TaylorT3.

\heading{Prototypes}

<tt>LALInspiralWave3()</tt>
<ul>
<li> \c output: Output containing the inspiral waveform.</li>
<li> \c params: Input containing binary chirp parameters.</li>
</ul>


<tt>LALInspiralWave3Templates()</tt>
<ul>
<li> \c output1: Output containing the 0-phase inspiral waveform.</li>
<li> \c output2: Output containing the \f$\pi/2\f$-phase inspiral waveform.</li>
<li> \c params: Input containing binary chirp parameters.</li>
</ul>


\heading{Description}
LALInspiralWave3() generates #TaylorT3 approximant which
corresponds to the case wherein
the phase of the waveform is given as an explicit function of time
as in Equation.\eqref{eq_InspiralWavePhase3}.

LALInspiralWave3Templates() simultaneously generates
two inspiral waveforms and the two differ in
phase by \f$\pi/2\f$.

\heading{Algorithm}

\heading{Uses}
\code
LALInspiralParameterCalc()
LALInspiralChooseModel()
LALInspiralSetup()
LALInspiralPhasing3 (via expnFunc)()
LALInspiralFrequency3 (via expnFunc)()
\endcode

*/

#include <lal/LALStdlib.h>
#include <lal/LALInspiral.h>
#include <lal/FindRoot.h>
#include <lal/Units.h>
#include <lal/SeqFactories.h>


typedef struct
{
	REAL8 (*func)(REAL8 tC, expnCoeffs *ak);
	expnCoeffs ak;
}
ChirptimeFromFreqIn;

static REAL8 XLALInspiralFrequency3Wrapper(REAL8 tC, void *pars);

static int
XLALInspiralWave3Engine(
                REAL4Vector      *output1,
                REAL4Vector      *output2,
                REAL4Vector      *h,
                REAL4Vector      *a,
                REAL4Vector      *ff,
                REAL8Vector      *phi,
                InspiralTemplate *params,
                InspiralInit     *paramsInit
                );

NRCSID (LALINSPIRALWAVE3C, "$Id$");



void
LALInspiralWave3 (
   LALStatus        *status,
   REAL4Vector      *output,
   InspiralTemplate *params
   )

{


  INT4 count;
  InspiralInit paramsInit;

  INITSTATUS (status, "LALInspiralWave3", LALINSPIRALWAVE3C);
  ATTATCHSTATUSPTR(status);

  ASSERT(output, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(output->data, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(params, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(params->nStartPad >= 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT(params->fLower > 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT(params->tSampling > 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);

  LALInspiralSetup (status->statusPtr, &(paramsInit.ak), params);
  CHECKSTATUSPTR(status);
  LALInspiralChooseModel(status->statusPtr, &(paramsInit.func),
					 &(paramsInit.ak), params);
  CHECKSTATUSPTR(status);


  LALInspiralParameterCalc (status->statusPtr, params);
  CHECKSTATUSPTR(status);

  ASSERT(params->totalMass >0., status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT(params->eta >= 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);

  memset( output->data, 0, output->length * sizeof(REAL4) );

  /* Call the engine function */
  count = XLALInspiralWave3Engine(output, NULL, NULL,
			NULL, NULL, NULL, params, &paramsInit);
  if (count < 0)
    ABORTXLAL(status);

  CHECKSTATUSPTR(status);

  DETATCHSTATUSPTR(status);
  RETURN(status);
}

static REAL8 XLALInspiralFrequency3Wrapper(REAL8 tC, void *pars)
{
  static const char *func = "XLALInspiralFrequency3Wrapper";

  ChirptimeFromFreqIn *in;
  REAL8 freq, f;

  in = (ChirptimeFromFreqIn *) pars;
  freq = in->func(tC, &(in->ak));
  if (XLAL_IS_REAL8_FAIL_NAN(freq))
    XLAL_ERROR_REAL8(func, XLAL_EFUNC);
  f = freq - in->ak.f0;

  /*
  fprintf(stderr, "Here freq=%e f=%e tc=%e f0=%e\n", freq, *f, tC, in->ak.f0);
   */

  return f;
}

NRCSID (LALINSPIRALWAVE3TEMPLATESC, "$Id$");



void
LALInspiralWave3Templates (
   LALStatus        *status,
   REAL4Vector      *output1,
   REAL4Vector      *output2,
   InspiralTemplate *params
   )

{

  INT4 count;

  InspiralInit paramsInit;

  INITSTATUS (status, "LALInspiralWave3Templates", LALINSPIRALWAVE3TEMPLATESC);
  ATTATCHSTATUSPTR(status);

  ASSERT(output1, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(output2, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(output1->data, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(output2->data, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(params, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT(params->nStartPad >= 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT(params->fLower > 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT(params->tSampling > 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);

  LALInspiralSetup (status->statusPtr, &(paramsInit.ak), params);
  CHECKSTATUSPTR(status);
  LALInspiralChooseModel(status->statusPtr, &(paramsInit.func),
					&(paramsInit.ak), params);
  CHECKSTATUSPTR(status);

/* Calculate the three unknown paramaters in (m1,m2,M,eta,mu) from the two
   which are given.  */

  LALInspiralParameterCalc (status->statusPtr, params);
  CHECKSTATUSPTR(status);

  ASSERT(params->totalMass >0., status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT(params->eta >= 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);

  /* Initialise the waveforms to zero */
  memset(output1->data, 0, output1->length * sizeof(REAL4));
  memset(output2->data, 0, output2->length * sizeof(REAL4));

  /* Call the engine function */
  count = XLALInspiralWave3Engine(output1, output2, NULL,
			    NULL, NULL, NULL, params, &paramsInit);
  if (count < 0)
    ABORTXLAL(status);
  CHECKSTATUSPTR(status);

  DETATCHSTATUSPTR(status);
  RETURN(status);
}





NRCSID (LALINSPIRALWAVE3FORINJECTIONC, "$Id$");



void
LALInspiralWave3ForInjection (
			      LALStatus        *status,
			      CoherentGW       *waveform,
			      InspiralTemplate *params,
			      PPNParamStruc  *ppnParams)


{

  INT4 count;
  UINT4 i;
  REAL4Vector *h=NULL;
  REAL4Vector *a=NULL;
  REAL4Vector *ff=NULL ;
  REAL8Vector *phiv=NULL;
  CreateVectorSequenceIn in;

  REAL8 phiC;/* phase at coalescence */
  CHAR message[256];


  InspiralInit paramsInit;

 /** -- -- */
  INITSTATUS (status, "LALInspiralWave3ForInjection", LALINSPIRALWAVE3FORINJECTIONC);
  ATTATCHSTATUSPTR(status);

  /* Make sure parameter and waveform structures exist. */
  ASSERT( params, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL );
  ASSERT(waveform, status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT( !( waveform->h ), status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL );
  ASSERT( !( waveform->a ), status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL );
  ASSERT( !( waveform->f ), status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL );
  ASSERT( !( waveform->phi ), status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL );
  ASSERT( !( waveform->shift ), status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL );

  params->ampOrder = 0;
  sprintf(message, "WARNING: Amp Order has been reset to %d", params->ampOrder);
  LALInfo(status, message);
  /* Compute some parameters*/
  LALInspiralInit(status->statusPtr, params, &paramsInit);
  CHECKSTATUSPTR(status);

  if (paramsInit.nbins==0)
    {
      DETATCHSTATUSPTR(status);
      RETURN (status);

    }

  /* Now we can allocate memory and vector for coherentGW structure*/
  LALSCreateVector(status->statusPtr, &ff, paramsInit.nbins);
  CHECKSTATUSPTR(status);
  LALSCreateVector(status->statusPtr, &a, 2*paramsInit.nbins);
  CHECKSTATUSPTR(status);
  LALDCreateVector(status->statusPtr, &phiv, paramsInit.nbins);
  CHECKSTATUSPTR(status);

 /* By default the waveform is empty */

  memset(ff->data, 0, ff->length * sizeof(REAL4));
  memset(a->data,  0, a->length * sizeof(REAL4));
  memset(phiv->data, 0, phiv->length * sizeof(REAL8));

  if( params->ampOrder )
  {
     LALSCreateVector(status->statusPtr, &h, 2*paramsInit.nbins);
     CHECKSTATUSPTR(status);
     memset(h->data,  0, h->length * sizeof(REAL4));
  }

  /* Call the engine function */
  count = XLALInspiralWave3Engine(NULL, NULL, h, a, ff, phiv, params, &paramsInit);
  if (count < 0)
    ABORTXLAL(status);
  BEGINFAIL( status ) {
     LALSDestroyVector(status->statusPtr, &ff);
     CHECKSTATUSPTR(status);
     LALSDestroyVector(status->statusPtr, &a);
     CHECKSTATUSPTR(status);
     LALDDestroyVector(status->statusPtr, &phiv);
     CHECKSTATUSPTR(status);
     if( params->ampOrder )
     {
       LALSDestroyVector(status->statusPtr, &h);
       CHECKSTATUSPTR(status);
     }
  }
  ENDFAIL(status);

  /* Check an empty waveform hasn't been returned */
  for (i = 0; i < phiv->length; i++)
  {
    if (phiv->data[i] != 0.0) break;
    /* If the waveform returned is empty, return now */
    if (i == phiv->length - 1)
    {
      LALSDestroyVector(status->statusPtr, &ff);
      CHECKSTATUSPTR(status);
      LALSDestroyVector(status->statusPtr, &a);
      CHECKSTATUSPTR(status);
      LALDDestroyVector(status->statusPtr, &phiv);
      CHECKSTATUSPTR(status);

      DETATCHSTATUSPTR(status);
      RETURN(status);
    }
  }

  /*  if ( (phase/2./LAL_PI) < 2. ){
    sprintf(message, "The waveform has only %lf cycles; we don't keep waveform with less than 2 cycles.",
	       (double)phase/2./(double)LAL_PI );
    LALWarning(status, message);


  }
  else*/
    {

      /*wrap the phase vector*/
      phiC =  phiv->data[count-1] ;
      for (i=0; i<(UINT4)count;i++)
	{
	  phiv->data[i] =  phiv->data[i] -phiC + ppnParams->phi;
	}
      /* Allocate the waveform structures. */
      if ( ( waveform->a = (REAL4TimeVectorSeries *)
	     LALMalloc( sizeof(REAL4TimeVectorSeries) ) ) == NULL ) {
	ABORT( status, LALINSPIRALH_EMEM,
	       LALINSPIRALH_MSGEMEM );
      }
      memset( waveform->a, 0, sizeof(REAL4TimeVectorSeries) );
      if ( ( waveform->f = (REAL4TimeSeries *)
	     LALMalloc( sizeof(REAL4TimeSeries) ) ) == NULL ) {
	LALFree( waveform->a ); waveform->a = NULL;
	ABORT( status, LALINSPIRALH_EMEM,
	       LALINSPIRALH_MSGEMEM );
      }
      memset( waveform->f, 0, sizeof(REAL4TimeSeries) );
      if ( ( waveform->phi = (REAL8TimeSeries *)
	     LALMalloc( sizeof(REAL8TimeSeries) ) ) == NULL ) {
	LALFree( waveform->a ); waveform->a = NULL;
	LALFree( waveform->f ); waveform->f = NULL;
	ABORT( status, LALINSPIRALH_EMEM,
	       LALINSPIRALH_MSGEMEM );
      }
      memset( waveform->phi, 0, sizeof(REAL8TimeSeries) );



      in.length = (UINT4)count;
      in.vectorLength = 2;
      LALSCreateVectorSequence( status->statusPtr,
				&( waveform->a->data ), &in );
      CHECKSTATUSPTR(status);
      LALSCreateVector( status->statusPtr,
			&( waveform->f->data ), count);
      CHECKSTATUSPTR(status);
      LALDCreateVector( status->statusPtr,
			&( waveform->phi->data ), count );
      CHECKSTATUSPTR(status);




      memcpy(waveform->f->data->data , ff->data, count*(sizeof(REAL4)));
      memcpy(waveform->a->data->data , a->data, 2*count*(sizeof(REAL4)));
      memcpy(waveform->phi->data->data ,phiv->data, count*(sizeof(REAL8)));




      waveform->a->deltaT = waveform->f->deltaT = waveform->phi->deltaT
	= 1./params->tSampling;

      waveform->a->sampleUnits = lalStrainUnit;
      waveform->f->sampleUnits = lalHertzUnit;
      waveform->phi->sampleUnits = lalDimensionlessUnit;
      waveform->position = ppnParams->position;
      waveform->psi = ppnParams->psi;

      snprintf( waveform->a->name, LALNameLength, "T3 inspiral amplitudes" );
      snprintf( waveform->f->name, LALNameLength, "T3 inspiral frequency" );
      snprintf( waveform->phi->name, LALNameLength, "T3  inspiral phase" );

      /* --- fill some output ---*/

      ppnParams->tc     = (double)(count-1) / params->tSampling ;
      ppnParams->length = count;
      ppnParams->dfdt   = ((REAL4)(waveform->f->data->data[count-1]
			- waveform->f->data->data[count-2])) * ppnParams->deltaT;
      ppnParams->fStop  = params->fFinal;
      ppnParams->termCode        = GENERATEPPNINSPIRALH_EFSTOP;
      ppnParams->termDescription = GENERATEPPNINSPIRALH_MSGEFSTOP;

      ppnParams->fStart   = ppnParams->fStartIn;

      if( params->ampOrder )
      {
        if ( ( waveform->h = (REAL4TimeVectorSeries *)
	       LALMalloc( sizeof(REAL4TimeVectorSeries) ) ) == NULL )
        {
	  ABORT( status, LALINSPIRALH_EMEM, LALINSPIRALH_MSGEMEM );
        }
        memset( waveform->h, 0, sizeof(REAL4TimeVectorSeries) );
        LALSCreateVectorSequence( status->statusPtr,
				  &( waveform->h->data ), &in );
        CHECKSTATUSPTR(status);
        memcpy(waveform->h->data->data , h->data, 2*count*(sizeof(REAL4)));
        waveform->h->deltaT = 1./params->tSampling;
        waveform->h->sampleUnits = lalStrainUnit;
        snprintf( waveform->h->name, LALNameLength, "T3 inspiral polarizations" );
        LALSDestroyVector(status->statusPtr, &h);
        CHECKSTATUSPTR(status);
      }
    }    /*end of coherentGW storage */


  /* --- free memory --- */
  LALSDestroyVector(status->statusPtr, &ff);
  CHECKSTATUSPTR(status);
  LALSDestroyVector(status->statusPtr, &a);
  CHECKSTATUSPTR(status);
  LALDDestroyVector(status->statusPtr, &phiv);
  CHECKSTATUSPTR(status);



  DETATCHSTATUSPTR(status);
  RETURN(status);
}


/* Engine function used to generate the waveforms */
static int
XLALInspiralWave3Engine(
                REAL4Vector      *output1,
                REAL4Vector      *output2,
                REAL4Vector      *h,
                REAL4Vector      *a,
                REAL4Vector      *ff,
                REAL8Vector      *phiv,
                InspiralTemplate *params,
                InspiralInit     *paramsInit
                )

{
  INT4 i, startShift, count;
  REAL8 dt, fu, eta, tc, totalMass, t, td, c1, phi0, phi1, phi;
  REAL8 v, v2, f, fHigh, tmax, fOld, phase, omega;
  REAL8 xmin,xmax,xacc;
  REAL8 (*frequencyFunction)(REAL8, void *);
  expnFunc func;
  expnCoeffs ak;
  ChirptimeFromFreqIn timeIn;
  void *pars;

  REAL8 temp, tempMax=0, tempMin = 0;

  /* Only used in injection case */
  REAL8 unitHz = 0.;
  REAL8 cosI = 0.;/* cosine of system inclination */
  REAL8 apFac = 0., acFac = 0.;/* extra factor in plus and cross amplitudes */


  ak   = paramsInit->ak;
  func = paramsInit->func;

  if (output2 || a)
      params->nStartPad = 0; /* must be zero for templates and injections */

  eta = params->eta;   /* Symmetric mass ratio  */
  /* Only in injection case.. */
  unitHz = params->totalMass*LAL_MTSUN_SI*(REAL8)LAL_PI;
  cosI   = cos( params->inclination );
  apFac  = acFac = -2.0 * eta *params->totalMass * LAL_MRSUN_SI/params->distance;
  apFac *= 1.0 + cosI*cosI;
  acFac *= 2.0 * cosI;

  dt = 1.0/(params->tSampling);    /* sampling rate  */
  fu = params->fCutoff;            /* upper frequency cutoff  */
  phi = params->startPhase;        /* initial phase  */
  startShift = params->nStartPad;  /* number of zeros at the start of the wave  */


  tc=params->tC;       /* Instant of coalescence of the compact objects */
  totalMass = (params->totalMass)*LAL_MTSUN_SI; /* mass of the system in seconds */


/*
   If flso is less than the user inputted upper frequency cutoff fu,
   then the waveforn is truncated at f=flso.
*/

  if (fu)
     fHigh = (fu < ak.flso) ? fu : ak.flso;
  else
     fHigh = ak.flso;

/*
   Check that the highest frequency is less than half the sampling frequency -
   the Nyquist theorum
*/

  if (fHigh >= 0.5/dt)
  {
    XLALPrintError("fHigh must be less than Nyquist frequency\n");
    XLAL_ERROR(__func__, XLAL_EDOM);
  }

  if (fHigh <= params->fLower)
  {
    XLALPrintError("fHigh must be larger than fLower\n");
    XLAL_ERROR(__func__, XLAL_EDOM);
  }

/* Here's the part which calculates the waveform */

  c1 = eta/(5.*totalMass);

  i = startShift;

  /*
   * In Jan 2003 we realized that the tC determined as a sum of chirp times is
   * not quite the tC that should enter the definition of Theta in the expression
   * for the frequency as a function of time (see DIS3, 2000). This is because
   * chirp times are obtained by inverting t(f). Rather tC should be obtained by
   * solving the equation f0 - f(tC) = 0. This is what is implemented below.
   */

  timeIn.func = func.frequency3;
  timeIn.ak = ak;
  frequencyFunction = &XLALInspiralFrequency3Wrapper;
  xmin = c1*params->tC/2.;
  xmax = c1*params->tC*2.;
  xacc = 1.e-6;
  pars = (void*) &timeIn;
  /* tc is the instant of coalescence */

  xmax = c1*params->tC*3 + 5.; /* we add 5 so that if tC is small then xmax
                                  is always greater than a given value (here 5)*/

  /* for x in [xmin, xmax], we search the value which gives the max frequency.
   and keep the corresponding rootIn.xmin. */



  for (tc = c1*params->tC/1000.; tc < xmax; tc+=c1*params->tC/1000.){
    temp = XLALInspiralFrequency3Wrapper(tc , pars);
    if (XLAL_IS_REAL8_FAIL_NAN(temp))
      XLAL_ERROR(__func__, XLAL_EFUNC);
    if (temp > tempMax) {
      xmin = tc;
      tempMax = temp;
    }
    if (temp < tempMin) {
      tempMin = temp;
    }
  }

  /* if we have found a value positive then everything should be fine in the
     BissectionFindRoot function */
  if (tempMax > 0  &&  tempMin < 0){
    tc = XLALDBisectionFindRoot (frequencyFunction, xmin, xmax, xacc, pars);
    if (XLAL_IS_REAL8_FAIL_NAN(tc))
      XLAL_ERROR(__func__, XLAL_EFUNC);
  }
  else if (a)
  {
    /* Otherwise we return an empty waveform for injection */
    return 0;
  }
  else
  {
    /* Or abort if not injection */
    XLALPrintError("Can't find good bracket for BisectionFindRoot\n");
    XLAL_ERROR(__func__, XLAL_EFAILED);
  }

  tc /= c1;

  tc += params->startTime;       /* Add user given startTime to instant of
                                     coalescence of the compact objects */

  t=0.0;
  td = c1*(tc-t);
  phase = func.phasing3(td, &ak);
  if (XLAL_IS_REAL8_FAIL_NAN(phase))
    XLAL_ERROR(__func__, XLAL_EFUNC);
  f = func.frequency3(td, &ak);
  if (XLAL_IS_REAL8_FAIL_NAN(f))
    XLAL_ERROR(__func__, XLAL_EFUNC);
  phi0=-phase+phi;
  phi1=phi0+LAL_PI_2;

  count = 0;
  tmax = tc - dt;
  fOld = 0.0;

/* We stop if any of the following conditions fail */

  while (f < fHigh && t < tmax && f > fOld)
  {
    /* Check we don't write past the end of the vector */
    if ((output1 && ((UINT4)i >= output1->length)) || (ff && ((UINT4)count >= ff->length)))
    {
      XLALPrintError("Attempting to write beyond the end of vector\n");
      XLAL_ERROR(__func__, XLAL_EBADLEN);
    }

    fOld = f;
    v = pow(f*LAL_PI*totalMass, oneby3);
    v2 = v*v;
    if (output1)
    {


      /*
      output1->data[i]   = LALInspiralHPlusPolarization( phase+phi0, v, params );
      */
      output1->data[i]   = (REAL4) (apFac * v2) * cos(phase+phi0);
      if (output2)
      /*
        output2->data[i] = LALInspiralHCrossPolarization( phase+phi1, v, params );
      */
        output2->data[i] = (REAL4) (apFac * v2) * cos(phase+phi1);
    }
    else if (a)
    {
      int ice, ico;
      ice = 2*count;
      ico = ice + 1;
      omega = v*v*v;

      ff->data[count]     = (REAL4)(omega/unitHz);
      a->data[ice]        = (REAL4)(apFac * v2 );
      a->data[ico]        = (REAL4)(acFac * v2 );
      phiv->data[count]   = (REAL8)(phase );

      if (h)
      {
        h->data[ice] = LALInspiralHPlusPolarization( phase, v, params );
        h->data[ico] = LALInspiralHCrossPolarization( phase, v, params );
      }
    }
    ++i;
    ++count;
    t=count*dt;
    td = c1*(tc-t);
    phase = func.phasing3(td, &ak);
    if (XLAL_IS_REAL8_FAIL_NAN(phase))
      XLAL_ERROR(__func__, XLAL_EFUNC);
    f = func.frequency3(td, &ak);
    if (XLAL_IS_REAL8_FAIL_NAN(f))
      XLAL_ERROR(__func__, XLAL_EFUNC);
  }
  params->fFinal = fOld;
  if (output1 && !output2) params->tC = t;
/*
  fprintf(stderr, "%e %e\n", f, fHigh);
*/

  return count;
}

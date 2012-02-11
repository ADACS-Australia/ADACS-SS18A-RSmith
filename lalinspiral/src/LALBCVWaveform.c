/*
*  Copyright (C) 2007 B.S. Sathyaprakash, Thomas Cokelaer
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
 * \author B.S. Sathyaprakash
 * \file
 * \ingroup LALInspiral_h


 * \brief This module contains a single function <tt>LALBCVWaveform()</tt>.
 *
 * \heading{Prototypes}
 *
 * <tt>LALLALBCVWaveform():</tt>
 * <ul>
 * <li> \c signalvec: Output containing the <em>Fourier transform</em> of the inspiral waveform.
 * </li><li> \c params: Input containing binary chirp parameters;
 * it is necessary and sufficent to specify the following parameters
 * of the \c params structure:
 * <tt>psi0, psi3, alpha, fendBCV(fFinal), nStartPad, fLower, tSampling</tt>.
 * All other parameters in \c params are ignored.  </li></ul>
 *
 * <tt>LALLALBCVSpinWaveform()</tt>
 * <ul>
 * <li> \c signalvec: Output containing the <em>Fourier transform</em> of the inspiral waveform.
 * </li><li> \c params: Input containing binary chirp parameters;
 * it is necessary and sufficent to specify the following parameters
 * of the \c params structure:
 * <tt>psi0, psi3, alpha1, alpha2, beta, fendBCV(fFinal), nStartPad, fLower, tSampling</tt>.
 * All other parameters in \c params are ignored.  </li></ul> *
 *
 * \heading{Description}
 * This module can be used to generate <em>detection template
 * family</em> of Buonanno, Chen and Vallisneri [\ref BCV03,BCV03b].
 * There are two modules: <tt>LALBCVWaveform.</tt> and <tt>LALBCVSpinWaveform.</tt>
 * The former can be used to generate non-spinning waveforms and the DTF
 * it implements is given in Eq.\eqref{eq_BCV_NonSpinning} and the latter
 * to generate spinning waveforms (Eq.\eqref{eq_BCV_Spinning}.
 *
 * \heading{Algorithm}
 * A straightforward implementation of the formula. Note that the routine returns
 * <tt>Fourier transform</tt> of the signal as opposed to most other modules in this
 * package which return time-domain signals. Also, the amplitude is quite arbitrary.
 *
 * \heading{Uses}
 * \code
 * ASSERT
 * ATTATCHSTATUSPTR
 * DETATCHSTATUSPTR
 * INITSTATUS
 * RETURN
 * \endcode
 *
 * \heading{Notes}
 *
 * %% Any relevant notes.
 *
 *
 *
*/

#include <lal/LALInspiral.h>

void
LALBCVWaveform(
   LALStatus        *status,
   REAL4Vector      *signalvec,
   InspiralTemplate *params
   )
 {

  REAL8 f, df;
  REAL8 shift, phi, psi, amp0, amp;
  REAL8 Sevenby6, Fiveby3, Twoby3, alpha;
  INT4 n, i;

  INITSTATUS(status);
  ATTATCHSTATUSPTR(status);

  ASSERT (signalvec,  status,       LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT (signalvec->data,  status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT (params,  status,       LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT (params->nStartPad >= 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT (params->fLower > 0, status,     LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT (params->tSampling > 0, status,  LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);

  n = signalvec->length;
  Twoby3 = 2.L/3.L;
  Sevenby6 = 7.L/6.L;
  Fiveby3 = 5.L/3.L;

  df = params->tSampling/(REAL8)n;
  params->fFinal = params->fCutoff;
  alpha = params->alpha / pow(params->fCutoff, Twoby3);


  /* to do : check that a_f in [0,1]??*/

  /*
  totalMass = params->totalMass * LAL_MTSUN_SI;
  shift = LAL_TWOPI * (params->tC - (float)n /params->tSampling
		  - params->startTime - params->nStartPad/params->tSampling);
  */
  /*
  shift = LAL_TWOPI * ((double) params->nStartPad/params->tSampling + params->startTime);
  phi = params->startPhase + params->psi0 * pow(params->fLower, -Fiveby3)
	  + params->psi3 * pow(params->fLower, -Twoby3);
   */
  shift = -LAL_TWOPI * ((double)params->nStartPad/params->tSampling);
  phi = - params->startPhase;

  /*
  amp0 = params->signalAmplitude * pow(5./(384.*params->eta), 0.5) *
	   totalMass * pow(LAL_PI * totalMass,-Sevenby6) *
	   params->tSampling * (2. / signalvec->length);

   */

  amp0 = params->signalAmplitude;
  /*  Computing BCV waveform */

  signalvec->data[0] = 0.0;
  signalvec->data[n/2] = 0.0;

  for(i=1;i<n/2;i++)
  {
	  f = i*df;

	  if (f < params->fLower || f > params->fFinal)
	  {
	  /*
	   * All frequency components below params->fLower and above fn are set to zero
	   */
              signalvec->data[i] = 0.;
              signalvec->data[n-i] = 0.;

	  }
       	  else
	  {
          /* What shall we put for sign phi? for uspa it must be "-" */
              psi =  (shift*f + phi + params->psi0*pow(f,-Fiveby3) + params->psi3*pow(f,-Twoby3));
	      amp = amp0 * (1. - alpha * pow(f,Twoby3)) * pow(f,-Sevenby6);

              signalvec->data[i] = (REAL4) (amp * cos(psi));
              signalvec->data[n-i] = (REAL4) (-amp * sin(psi));
          }
  }
  DETATCHSTATUSPTR(status);
  RETURN(status);
}






void
LALBCVSpinWaveform(
   LALStatus        *status,
   REAL4Vector      *signalvec,
   InspiralTemplate *params
   )
 {

  REAL8 f, df;
  REAL8 shift, phi, psi, amp0, ampRe, ampIm, modphase;
  REAL8 Sevenby6, Fiveby3, Twoby3;
  INT4 n, i;

  INITSTATUS(status);
  ATTATCHSTATUSPTR(status);

  ASSERT (signalvec,  status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT (signalvec->data,  status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT (params,  status, LALINSPIRALH_ENULL, LALINSPIRALH_MSGENULL);
  ASSERT (params->nStartPad >= 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT (params->fLower > 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);
  ASSERT (params->tSampling > 0, status, LALINSPIRALH_ESIZE, LALINSPIRALH_MSGESIZE);

  n = signalvec->length;
  Twoby3 = 2.L/3.L;
  Sevenby6 = 7.L/6.L;
  Fiveby3 = 5.L/3.L;

  df = params->tSampling/(REAL8)n;

  /*
  totalMass = params->totalMass * LAL_MTSUN_SI;
  shift = LAL_TWOPI * (params->tC - (float)n /params->tSampling
		  - params->startTime - params->nStartPad/params->tSampling);
  */
  shift = -LAL_TWOPI * (params->nStartPad/params->tSampling);

  /* use this to shift waveform start time by hand... */
  /*shift = -LAL_TWOPI * (131072/params->tSampling);*/

  phi = - params->startPhase;

  /*
  amp0 = params->signalAmplitude * pow(5./(384.*params->eta), 0.5) *
	   totalMass * pow(LAL_PI * totalMass,-Sevenby6) *
	   params->tSampling * (2. / signalvec->length);

   */
  amp0 = params->signalAmplitude;
  /*  Computing BCV waveform */

  signalvec->data[0] = 0.0;
  signalvec->data[n/2] = 0.0;

  for(i=1;i<n/2;i++)
  {
	  f = i*df;
	  if (f < params->fLower || f > params->fCutoff)
	  {
	  /*
	   * All frequency components below params->fLower and above fn are set to zero
	   */
              signalvec->data[i] = 0.;
              signalvec->data[n-i] = 0.;

	  }
       	  else
	  {
          /* What shall we put for sign phi? for uspa it must be "-" */

	    psi = (shift*f + phi + params->psi0*pow(f,-Fiveby3) + params->psi3*pow(f,-Twoby3));
	    modphase = params->beta * pow(f,-Twoby3);

	    ampRe = amp0 * pow(f,-Sevenby6)
                         * (params->alpha1
                         + (params->alpha2 * cos(modphase))
                         + (params->alpha3 * sin(modphase)));
	    ampIm = amp0 * pow(f,-Sevenby6)
                         * (params->alpha4
                         + (params->alpha5 * cos(modphase))
                         + (params->alpha6 * sin(modphase)));

            signalvec->data[i]   = (REAL4) ((ampRe * cos(psi)) - (ampIm * sin(psi)));
            signalvec->data[n-i] = (REAL4) -1.*((ampRe * sin(psi)) + (ampIm * cos(psi)));
          }
  }

  params->fFinal = params->fCutoff;

  DETATCHSTATUSPTR(status);
  RETURN(status);
}

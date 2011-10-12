/*
 * Copyright (C) 2011 Drew Keppel, J. Creighton, S. Fairhurst, B. Krishnan, L. Santamaria, Stas Babak, David Churches, B.S. Sathyaprakash, Craig Robinson , Thomas Cokelaer
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

#include <math.h>

#include <lal/LALSimInspiral.h>
#define LAL_USE_COMPLEX_SHORT_MACROS
#include <lal/FindRoot.h>
#include <lal/LALComplex.h>
#include <lal/LALConstants.h>
#include <lal/LALStdlib.h>
#include <lal/TimeSeries.h>
#include <lal/Units.h>
#include <lal/LALSimInspiraldEnergyFlux.h>
#include <LALSimInspiralPNCoefficients.c>

#include "check_series_macros.h"

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

NRCSID(LALSIMINSPIRALTAYLORT3C, "$Id$");

typedef struct
tagexpnCoeffsTaylorT3 {
   int ieta;
   /* Taylor expansion coefficents in phi(t)*/
   REAL8 ptaN, pta2, pta3, pta4, pta5, pta6, pta7, ptl6;
   /* Taylor expansion coefficents in f(t)*/
   REAL8 ftaN, fta2, fta3, fta4, fta5, fta6, fta7, ftl6;

   /* sampling interval*/
   REAL8 samplinginterval;
   /* symmetric mass ratio, total mass, component masses*/
   REAL8 eta, totalmass, m1, m2;

   /* initial and final values of frequency, time, velocity; lso
    values of velocity and frequency; final phase.*/
   REAL8 f0, fn, t0, theta_lso, v0, vn, vf, vlso, flso, phiC;

   /* last stable orbit and pole defined by various Taylor and P-approximants*/
   REAL8 vlsoT0, vlsoT2, vlsoT4, vlsoT6;
}  expnCoeffsTaylorT3;

typedef REAL8 (SimInspiralPhasing3)(
   REAL8 td,
   expnCoeffsTaylorT3 *ak);

typedef REAL8 (SimInspiralFrequency3)(
   REAL8 td,
   expnCoeffsTaylorT3 *ak);

typedef struct
tagexpnFuncTaylorT3
{
   SimInspiralPhasing3 *phasing3;
   SimInspiralFrequency3 *frequency3;
} expnFuncTaylorT3;

typedef struct
{
	REAL8 (*func)(REAL8 tC, expnCoeffsTaylorT3 *ak);
	expnCoeffsTaylorT3 ak;
}
FreqInFromChirptime;

static REAL8 XLALInspiralFrequency3Wrapper(REAL8 tC, void *pars)
{
  FreqInFromChirptime *in;
  REAL8 freq, f;

  in = (FreqInFromChirptime *) pars;
  freq = in->func(tC, &(in->ak));
  if (XLAL_IS_REAL8_FAIL_NAN(freq))
    XLAL_ERROR_REAL8(XLAL_EFUNC);
  f = freq - in->ak.f0;

  /*
  fprintf(stderr, "Here freq=%e f=%e tc=%e f0=%e\n", freq, *f, tC, in->ak.f0);
   */

  return f;
}

static REAL8
XLALSimInspiralFrequency3_0PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta3;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta3 = theta*theta*theta;

  frequency = theta3*ak->ftaN;

  return frequency;
}


static REAL8
XLALSimInspiralFrequency3_2PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;

  frequency = theta3*ak->ftaN * (1.
             + ak->fta2*theta2);

  return frequency;
}


static REAL8
XLALSimInspiralFrequency3_3PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;

  frequency = theta3*ak->ftaN * (1.
             + ak->fta2*theta2
             + ak->fta3*theta3);

  return frequency;
}


static REAL8
XLALSimInspiralFrequency3_4PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;

  frequency = theta3*ak->ftaN * (1.
             + ak->fta2*theta2
             + ak->fta3*theta3
             + ak->fta4*theta4);

  return frequency;
}


static REAL8
XLALSimInspiralFrequency3_5PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;

  frequency = theta3*ak->ftaN * (1.
             + ak->fta2*theta2
             + ak->fta3*theta3
             + ak->fta4*theta4
             + ak->fta5*theta5);

  return frequency;
}


static REAL8
XLALSimInspiralFrequency3_6PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5,theta6;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;
  theta6 = theta5*theta;

  frequency = theta3*ak->ftaN * (1.
             + ak->fta2*theta2
             + ak->fta3*theta3
             + ak->fta4*theta4
             + ak->fta5*theta5
             + (ak->fta6 + ak->ftl6*log(2.*theta))*theta6);

  return frequency;
}


static REAL8
XLALSimInspiralFrequency3_7PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5,theta6,theta7;
  REAL8 frequency;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;
  theta6 = theta5*theta;
  theta7 = theta6*theta;

  frequency = theta3*ak->ftaN * (1.
             + ak->fta2*theta2
             + ak->fta3*theta3
             + ak->fta4*theta4
             + ak->fta5*theta5
             + (ak->fta6 + ak->ftl6*log(2.*theta))*theta6
             + ak->fta7*theta7);

  return frequency;
}


static REAL8
XLALSimInspiralPhasing3_0PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta5;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta5 = pow(td,-0.625);
  phase = (ak->ptaN/theta5);

  return phase;
}


static REAL8
XLALSimInspiralPhasing3_2PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta5;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta5 = theta2*theta2*theta;

  phase = (ak->ptaN/theta5) * (1.
         + ak->pta2*theta2);

  return phase;
}


static REAL8
XLALSimInspiralPhasing3_3PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta5;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta5 = theta2*theta3;

  phase = (ak->ptaN/theta5) * (1.
         + ak->pta2*theta2
         + ak->pta3*theta3);

  return phase;
}


static REAL8
XLALSimInspiralPhasing3_4PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;

  phase = (ak->ptaN/theta5) * (1.
         + ak->pta2*theta2
         + ak->pta3*theta3
         + ak->pta4*theta4);

  return phase;
}


static REAL8
XLALSimInspiralPhasing3_5PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;

  phase = (ak->ptaN/theta5) * (1.
         + ak->pta2*theta2
         + ak->pta3*theta3
         + ak->pta4*theta4
         + ak->pta5 * log(theta/ak->theta_lso) * theta5);

  return phase;
}


static REAL8
XLALSimInspiralPhasing3_6PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5,theta6;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;
  theta6 = theta5*theta;

  phase = (ak->ptaN/theta5) * (1.
         + ak->pta2*theta2
         + ak->pta3*theta3
         + ak->pta4*theta4
         + ak->pta5*log(theta/ak->theta_lso)*theta5
         +(ak->ptl6*log(2.*theta) + ak->pta6)*theta6);

  return phase;
}


static REAL8
XLALSimInspiralPhasing3_7PN (
   REAL8       td,
   expnCoeffsTaylorT3 *ak
   )
{
  REAL8 theta,theta2,theta3,theta4,theta5,theta6,theta7;
  REAL8 phase;

  if (ak == NULL)
    XLAL_ERROR_REAL8(XLAL_EFAULT);

  theta = pow(td,-0.125);
  theta2 = theta*theta;
  theta3 = theta2*theta;
  theta4 = theta3*theta;
  theta5 = theta4*theta;
  theta6 = theta5*theta;
  theta7 = theta6*theta;

  phase = (ak->ptaN/theta5) * (1.
         + ak->pta2*theta2
         + ak->pta3*theta3
         + ak->pta4*theta4
         + ak->pta5*log(theta/ak->theta_lso)*theta5
         +(ak->ptl6*log(2.*theta) + ak->pta6)*theta6
         + ak->pta7*theta7);

  return phase;
}


/**
 * Returns the sum of chirp times to a specified order.
 *
 * Computes the sum of the chirp times to a specified order. Inputs given in SI
 * units.
 */
static REAL8 XLALSimInspiralChirpLength(
		REAL8 m1,		/**< mass of companion 1 */
		REAL8 m2,		/**< mass of companion 2 */
		REAL8 f_min,		/**< start frequency */
		int O			/**< twice post-Newtonian order */
		)
{
	REAL8 tN, t2, t3 ,t4, t5, t6, t6l, t7, tC;
	REAL8 v, v2, v3, v4, v5, v6, v7, v8;
	REAL8 piFl = LAL_PI * f_min;
	REAL8 m = m1 + m2;
	REAL8 eta = m1 * m2 / m / m;
	m *= LAL_G_SI / pow(LAL_C_SI, 3.0); /* convert m from kilograms to seconds */

	/* Should use the coefficients from LALInspiraSetup.c to avoid errors.
	 */
	v = cbrt(piFl * m);
	v2 = v*v;
	v3 = v*v2;
	v4 = v*v3;
	v5 = v*v4;
	v6 = v*v5;
	v7 = v*v6;
	v8 = v*v7;

	tN = XLALSimInspiralTaylorT2Timing_0PNCoeff(m1+m2, eta);
	t2 = XLALSimInspiralTaylorT2Timing_2PNCoeff(eta);
	t3 = XLALSimInspiralTaylorT2Timing_3PNCoeff(eta);
	t4 = XLALSimInspiralTaylorT2Timing_4PNCoeff(eta);
	t5 = XLALSimInspiralTaylorT2Timing_5PNCoeff(eta);
	t6 = XLALSimInspiralTaylorT2Timing_6PNCoeff(eta);
	t7 = XLALSimInspiralTaylorT2Timing_7PNCoeff(eta);
	t6l = XLALSimInspiralTaylorT2Timing_6PNLogCoeff(eta);

	switch (O) {
		case 0:
		case 1:
			t2 = 0.;
		case 2:
			t3 = 0.;
		case 3:
			t4 = 0.;
		case 4:
			t5 = 0.;
		case 5:
			t6 = 0.;
			t6l = 0.;
		case 6:
			t7 = 0.;
		case 7:
			break;
		case 8:
			XLALPrintError("XLAL Error - %s: Not supported for requested PN order\n", __func__);
			XLAL_ERROR_REAL8(XLAL_EINVAL);
			break;
		default:
			XLALPrintError("XLAL Error - %s: Unknown PN order in switch\n", __func__);
			XLAL_ERROR_REAL8(XLAL_EINVAL);
	}

	tC = -tN / v8 * (1.
		+ t2 * v2
		+ t3 * v3
		+ t4 * v4
		+ t5 * v5
		+ (t6 + t6l*log(16*v2)) * v6
		+ t7 * v7);

	return tC;
}


/**
 * Set up the expnCoeffsTaylorT3 and expnFuncTaylorT3 structures for
 * generating a TaylorT3 waveform.
 *
 * Inputs given in SI units.
 */
static int XLALSimInspiralTaylorT3Setup(
		expnCoeffsTaylorT3 *ak,	/**< coefficients for TaylorT3 evolution [modified] */
		expnFuncTaylorT3 *f,	/**< functions for TaylorT3 evolution [modified] */
	       	REAL8 deltaT,		/**< sampling interval */
		REAL8 m1,		/**< mass of companion 1 */
		REAL8 m2,		/**< mass of companion 2 */
		REAL8 f_min,		/**< start frequency */
		int O			/**< twice post-Newtonian order */
		)
{
  REAL8 eta, tn;

  ak->t0 = 0;
  ak->m1 = m1;
  ak->m2 = m2;
  ak->totalmass = ak->m1 + ak->m2;
  eta = ak->eta = m1 * m2 / (ak->totalmass * ak->totalmass);
  ak->totalmass *= LAL_G_SI / pow(LAL_C_SI, 3.0); /* convert totalmass from kilograms to seconds */

  ak->f0 = f_min;
  ak->samplinginterval = deltaT;
  ak->fn = 1. / (2. * ak->samplinginterval);
  ak->v0 = cbrt(LAL_PI * ak->totalmass * f_min);

  ak->ptaN = XLALSimInspiralTaylorT3Phasing_0PNCoeff(eta);
  ak->pta2 = XLALSimInspiralTaylorT3Phasing_2PNCoeff(eta);
  ak->pta3 = XLALSimInspiralTaylorT3Phasing_3PNCoeff(eta);
  ak->pta4 = XLALSimInspiralTaylorT3Phasing_4PNCoeff(eta);
  ak->pta5 = XLALSimInspiralTaylorT3Phasing_5PNCoeff(eta);
  ak->pta6 = XLALSimInspiralTaylorT3Phasing_6PNCoeff(eta);
  ak->pta7 = XLALSimInspiralTaylorT3Phasing_7PNCoeff(eta);
  ak->ptl6 = XLALSimInspiralTaylorT3Phasing_6PNLogCoeff(eta);

  ak->ftaN = XLALSimInspiralTaylorT3Frequency_0PNCoeff(m1+m2);
  ak->fta2 = XLALSimInspiralTaylorT3Frequency_2PNCoeff(eta);
  ak->fta3 = XLALSimInspiralTaylorT3Frequency_3PNCoeff(eta);
  ak->fta4 = XLALSimInspiralTaylorT3Frequency_4PNCoeff(eta);
  ak->fta5 = XLALSimInspiralTaylorT3Frequency_5PNCoeff(eta);
  ak->fta6 = XLALSimInspiralTaylorT3Frequency_6PNCoeff(eta);
  ak->fta7 = XLALSimInspiralTaylorT3Frequency_7PNCoeff(eta);
  ak->ftl6 = XLALSimInspiralTaylorT3Frequency_6PNLogCoeff(eta);

  switch (O)
  {
     case 0:
           f->phasing3 = &XLALSimInspiralPhasing3_0PN;
           f->frequency3 = &XLALSimInspiralFrequency3_0PN;
           break;
     case 1:
           XLALPrintError("XLAL Error - %s: PN approximant not supported for requested PN order\n", __func__);
           XLAL_ERROR(XLAL_EINVAL);
           break;
     case 2:
           f->phasing3 = &XLALSimInspiralPhasing3_2PN;
           f->frequency3 = &XLALSimInspiralFrequency3_2PN;
           break;
     case 3:
           f->phasing3 = &XLALSimInspiralPhasing3_3PN;
           f->frequency3 = &XLALSimInspiralFrequency3_3PN;
           break;
     case 4:
           f->phasing3 = &XLALSimInspiralPhasing3_4PN;
           f->frequency3 = &XLALSimInspiralFrequency3_4PN;
           break;
     case 5:
           f->phasing3 = &XLALSimInspiralPhasing3_5PN;
           f->frequency3 = &XLALSimInspiralFrequency3_5PN;
           break;
     case 6:
           f->phasing3 = &XLALSimInspiralPhasing3_6PN;
           f->frequency3 = &XLALSimInspiralFrequency3_6PN;
           break;
     case 7:
           f->phasing3 = &XLALSimInspiralPhasing3_7PN;
           f->frequency3 = &XLALSimInspiralFrequency3_7PN;
           break;
     case 8:
           XLALPrintError("XLAL Error - %s: PN approximant not supported for requested PN order\n", __func__);
           XLAL_ERROR(XLAL_EINVAL);
           break;
     default:
        XLALPrintError("XLAL Error - %s: Unknown PN order in switch\n", __func__);
        XLAL_ERROR(XLAL_EINVAL);
  }

  tn = XLALSimInspiralTaylorLength(deltaT, m1, m2, f_min, O);
  ak->theta_lso = pow(tn, -0.125);

  return XLAL_SUCCESS;
}


/**
 * Computes a post-Newtonian orbit using the Taylor T3 method.
 */
int XLALSimInspiralTaylorT3PNEvolveOrbit(
		REAL8TimeSeries **V,   /**< post-Newtonian parameter [returned] */
	       	REAL8TimeSeries **phi, /**< orbital phase [returned] */
	       	LIGOTimeGPS *t_end,    /**< time at end of waveform */
	       	REAL8 phi_end,         /**< GW phase at end of waveform */
	       	REAL8 deltaT,          /**< sampling interval */
		REAL8 m1,              /**< mass of companion 1 */
		REAL8 m2,              /**< mass of companion 2 */
		REAL8 f_min,           /**< start frequency */
		int O                  /**< twice post-Newtonian order */
		)
{
	const UINT4 blocklen = 1024;
	const REAL8 visco = sqrt(1.0/6.0);
	REAL8 m = m1 + m2;
	REAL8 nu = m1 * m2 / m / m;
	m *= LAL_G_SI / pow(LAL_C_SI, 3.0); /* convert m from kilograms to seconds */
	REAL8 tmptC, tC, c1, xmin, xmax, xacc, v, phase, fOld, t, td, temp, tempMin = 0, tempMax = 0;
	REAL8 (*freqfunction)(REAL8, void *);
	UINT4 j;
	REAL8 f;
	void *pars;

	expnFuncTaylorT3 expnfunc;
	expnCoeffsTaylorT3 ak;
	FreqInFromChirptime timeIn;

	/* allocate memory */

	*V = XLALCreateREAL8TimeSeries("ORBITAL_FREQUENCY_PARAMETER", t_end, 0.0, deltaT, &lalDimensionlessUnit,
		blocklen);
	*phi = XLALCreateREAL8TimeSeries("ORBITAL_PHASE", t_end, 0.0, deltaT, &lalDimensionlessUnit, blocklen);
	if (!V || !phi)
		XLAL_ERROR(XLAL_EFUNC);


	/* initialize expnCoeffsTaylorT3 and expnFuncTaylorT3 structures */
	if (XLALSimInspiralTaylorT3Setup(&ak, &expnfunc, deltaT, m1, m2, f_min, O))
		XLAL_ERROR(XLAL_EFUNC);

	tC = XLALSimInspiralChirpLength(m1, m2, f_min, O);
	c1 = nu/(5.*m);

	/*
	 * In Jan 2003 we realized that the tC determined as a sum of chirp
	 * times is not quite the tC that should enter the definition of Theta
	 * in the expression for the frequency as a function of time (see DIS3,
	 * 2000). This is because chirp times are obtained by inverting t(f).
	 * Rather tC should be obtained by solving the equation f0 - f(tC) = 0.
	 * This is what is implemented below.
	 */

	timeIn.func = expnfunc.frequency3;
	timeIn.ak = ak;
	freqfunction = &XLALInspiralFrequency3Wrapper;
	xmin = c1*tC/2.;
	xmax = c1*tC*2.;
	xacc = 1.e-6;
	pars = (void*) &timeIn;
	/* tc is the instant of coalescence */

	/* we add 5 so that if tC is small then xmax
	 * is always greater than a given value (here 5)*/
	xmax = c1*tC*3 + 5.;

	/* for x in [xmin, xmax], we search the value which gives the max
	 * frequency.  and keep the corresponding rootIn.xmin. */

	for (tmptC = c1*tC/1000.; tmptC < xmax; tmptC+=c1*tC/1000.){
		temp = XLALInspiralFrequency3Wrapper(tmptC , pars);
		if (XLAL_IS_REAL8_FAIL_NAN(temp))
			XLAL_ERROR(XLAL_EFUNC);
		if (temp > tempMax) {
			xmin = tmptC;
			tempMax = temp;
		}
		if (temp < tempMin) {
			tempMin = temp;
		}
	}

	/* if we have found a value positive then everything should be fine in
	 * the BissectionFindRoot function */
	if (tempMax > 0  &&  tempMin < 0){
		tC = XLALDBisectionFindRoot (freqfunction, xmin, xmax, xacc, pars);
		if (XLAL_IS_REAL8_FAIL_NAN(tC))
			XLAL_ERROR(XLAL_EFUNC);
	}
	else{
		XLALPrintError("Can't find good bracket for BisectionFindRoot");
		XLAL_ERROR(XLAL_EMAXITER);
	}

	tC /= c1;

	/* start waveform generation */

	t = 0.;
	td = c1 * (tC - t);
	phase = expnfunc.phasing3(td, &ak);
	if (XLAL_IS_REAL8_FAIL_NAN(phase))
		XLAL_ERROR(XLAL_EFUNC);
	f = expnfunc.frequency3(td, &ak);
	if (XLAL_IS_REAL8_FAIL_NAN(f))
		XLAL_ERROR(XLAL_EFUNC);

	v = cbrt(f * LAL_PI * m);
	(*V)->data->data[0] = v;
	(*phi)->data->data[0] = phase;

	j = 0;
	while (1) {

		/* make one step */

		j++;
		fOld = f;
		t = j * deltaT;
		td = c1 * (tC - t);
		phase = expnfunc.phasing3(td, &ak);
		if (XLAL_IS_REAL8_FAIL_NAN(phase))
			XLAL_ERROR(XLAL_EFUNC);
		f = expnfunc.frequency3(td, &ak);
		if (XLAL_IS_REAL8_FAIL_NAN(f))
			XLAL_ERROR(XLAL_EFUNC);
		v = cbrt(f * LAL_PI * m);

		/* check termination conditions */

		if (t >= tC) {
			XLALPrintInfo("XLAL Info - %s: PN inspiral terminated at coalesence time\n", __func__);
			break;
		}
		if (v >= visco) {
			XLALPrintInfo("XLAL Info - %s: PN inspiral terminated at ISCO\n", __func__);
			break;
		}
		if (f <= fOld) {
			XLALPrintInfo("XLAL Info - %s: PN inspiral terminated when frequency stalled\n", __func__);
			break;
		}
	
		/* save current values in vectors but first make sure we don't write past end of vectors */

		if ( j >= (*V)->data->length ) {
			if ( ! XLALResizeREAL8TimeSeries(*V, 0, (*V)->data->length + blocklen) )
				XLAL_ERROR(XLAL_EFUNC);
			if ( ! XLALResizeREAL8TimeSeries(*phi, 0, (*phi)->data->length + blocklen) )
				XLAL_ERROR(XLAL_EFUNC);
		}
		(*V)->data->data[j] = v;
		(*phi)->data->data[j] = phase;
	}

	/* make the correct length */

	if ( ! XLALResizeREAL8TimeSeries(*V, 0, j) )
		XLAL_ERROR(XLAL_EFUNC);
	if ( ! XLALResizeREAL8TimeSeries(*phi, 0, j) )
		XLAL_ERROR(XLAL_EFUNC);

	/* adjust to correct tc and phic */

	XLALGPSAdd(&(*phi)->epoch, -1.0*j*deltaT);
	XLALGPSAdd(&(*V)->epoch, -1.0*j*deltaT);

	/* phi here is the orbital phase = 1/2 * GW phase.
	 * End GW phase specified on command line.
	 * Adjust phase so phi = phi_end/2 at the end */

	phi_end /= 2.;
	phi_end -= (*phi)->data->data[j-1];
	for (j = 0; j < (*phi)->data->length; ++j)
		(*phi)->data->data[j] += phi_end;
	return (int)(*V)->data->length;
}


/**
 * Driver routine to compute the post-Newtonian inspiral waveform.
 *
 * This routine allows the user to specify different pN orders
 * for phasing calcuation vs. amplitude calculations.
 */
int XLALSimInspiralTaylorT3PNGenerator(
		REAL8TimeSeries **hplus,  /**< +-polarization waveform */
	       	REAL8TimeSeries **hcross, /**< x-polarization waveform */
	       	LIGOTimeGPS *t_end,       /**< time at end of waveform */
	       	REAL8 phi_end,            /**< GW phase at end of waveform */
	       	REAL8 x0,                 /**< tail-term gauge choice thing (if you don't know, just set it to zero) */
	       	REAL8 deltaT,             /**< sampling interval */
	       	REAL8 m1,                 /**< mass of companion 1 */
	       	REAL8 m2,                 /**< mass of companion 2 */
	       	REAL8 f_min,              /**< start frequency */
	       	REAL8 r,                  /**< distance of source */
	       	REAL8 i,                  /**< inclination of source (rad) */
	       	int amplitudeO,           /**< twice post-Newtonian amplitude order */
	       	int phaseO                /**< twice post-Newtonian phase order */
		)
{
	REAL8TimeSeries *V;
	REAL8TimeSeries *phi;
	int status;
	int n;
	n = XLALSimInspiralTaylorT3PNEvolveOrbit(&V, &phi, t_end, phi_end, deltaT, m1, m2, f_min, phaseO);
	if ( n < 0 )
		XLAL_ERROR(XLAL_EFUNC);
	status = XLALSimInspiralPNPolarizationWaveforms(hplus, hcross, V, phi, x0, m1, m2, r, i, amplitudeO);
	XLALDestroyREAL8TimeSeries(phi);
	XLALDestroyREAL8TimeSeries(V);
	if ( status < 0 )
		XLAL_ERROR(XLAL_EFUNC);
	return n;
}


/**
 * Driver routine to compute the post-Newtonian inspiral waveform.
 *
 * This routine uses the same pN order for phasing and amplitude
 * (unless the order is -1 in which case the highest available
 * order is used for both of these -- which might not be the same).
 *
 * Log terms in amplitudes are ignored.  This is a gauge choice.
 */
int XLALSimInspiralTaylorT3PN(
		REAL8TimeSeries **hplus,  /**< +-polarization waveform */
	       	REAL8TimeSeries **hcross, /**< x-polarization waveform */
	       	LIGOTimeGPS *t_end,       /**< time at end of waveform */
	       	REAL8 phi_end,            /**< GW phase at end of waveform */
	       	REAL8 deltaT,             /**< sampling interval */
	       	REAL8 m1,                 /**< mass of companion 1 */
	       	REAL8 m2,                 /**< mass of companion 2 */
	       	REAL8 f_min,              /**< start frequency */
	       	REAL8 r,                  /**< distance of source */
	       	REAL8 i,                  /**< inclination of source (rad) */
	       	int O                     /**< twice post-Newtonian order */
		)
{
	/* set x0=0 to ignore log terms */
	return XLALSimInspiralTaylorT3PNGenerator(hplus, hcross, t_end, phi_end, 0.0, deltaT, m1, m2, f_min, r, i, O, O);
}


/**
 * Driver routine to compute the restricted post-Newtonian inspiral waveform.
 *
 * This routine computes the phasing to the specified order, but
 * only computes the amplitudes to the Newtonian (quadrupole) order.
 *
 * Log terms in amplitudes are ignored.  This is a gauge choice.
 */
int XLALSimInspiralTaylorT3PNRestricted(
		REAL8TimeSeries **hplus,  /**< +-polarization waveform */
	       	REAL8TimeSeries **hcross, /**< x-polarization waveform */
	       	LIGOTimeGPS *t_end,       /**< time at end of waveform */
	       	REAL8 phi_end,            /**< GW phase at end of waveform */
	       	REAL8 deltaT,             /**< sampling interval */
	       	REAL8 m1,                 /**< mass of companion 1 */
	       	REAL8 m2,                 /**< mass of companion 2 */
	       	REAL8 f_min,              /**< start frequency */
	       	REAL8 r,                  /**< distance of source */
	       	REAL8 i,                  /**< inclination of source (rad) */
	       	int O                     /**< twice post-Newtonian phase order */
		)
{
	/* use Newtonian order for amplitude */
	/* set x0=0 to ignore log terms */
	return XLALSimInspiralTaylorT3PNGenerator(hplus, hcross, t_end, phi_end, 0.0, deltaT, m1, m2, f_min, r, i, 0, O);
}

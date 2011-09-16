/*
 * Copyright (C) 2011 P. Ajith, Nickolas Fotopoulos
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

#include <lal/LALStdlib.h>
#include <lal/LALSimIMR.h>
#include <lal/LALComplex.h>
#include <lal/LALConstants.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeSeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/Units.h>

typedef struct tagBBHPhenomParams{
  REAL8 fMerger;
  REAL8 fRing;
  REAL8 fCut;
  REAL8 sigma;
  REAL8 psi0;
  REAL8 psi1;
  REAL8 psi2;
  REAL8 psi3;
  REAL8 psi4;
  REAL8 psi5;
  REAL8 psi6;
  REAL8 psi7;
  REAL8 psi8;
}
BBHPhenomParams;

/**
 *
 * private function prototypes; all internal functions use solar masses.
 *
 */

static BBHPhenomParams *ComputeIMRPhenomAParams(REAL8 m1, REAL8 m2);
static BBHPhenomParams *ComputeIMRPhenomBParams(REAL8 m1, REAL8 m2, REAL8 chi);

static REAL8 EstimateSafeFMinForTD(REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 deltaT);
static REAL8 EstimateSafeFMaxForTD(REAL8 f_max, REAL8 dt);
static REAL8 ComputeTau0(REAL8 m1, REAL8 m2, REAL8 f_min);
static ssize_t EstimateIMRLength(REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 deltaT);
static ssize_t NextPow2(ssize_t n);

static REAL8 LorentzianFn(REAL8 freq, REAL8 fRing, REAL8 sigma);

static int IMRPhenomAGenerateFD(COMPLEX16FrequencySeries **htilde, LIGOTimeGPS *tRef, REAL8 phiRef, REAL8 fRef, REAL8 deltaF, REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 f_max, REAL8 distance, BBHPhenomParams *params);
static int IMRPhenomBGenerateFD(COMPLEX16FrequencySeries **htilde, LIGOTimeGPS *tRef, REAL8 phiRef, REAL8 fRef, REAL8 deltaF, REAL8 m1, REAL8 m2, REAL8 chi, REAL8 f_min, REAL8 f_max, REAL8 distance, BBHPhenomParams *params);
static int IMRPhenomAGenerateTD(REAL8TimeSeries **h, LIGOTimeGPS *tRef, REAL8 phiRef, REAL8 fRef, REAL8 deltaT, REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 f_max, REAL8 distance, BBHPhenomParams *params);
static int IMRPhenomBGenerateTD(REAL8TimeSeries **h, LIGOTimeGPS *tRef, REAL8 phiRef, REAL8 fRef, REAL8 deltaT, REAL8 m1, REAL8 m2, REAL8 chi, REAL8 f_min, REAL8 f_max, REAL8 distance, BBHPhenomParams *params);
static int FDToTD(REAL8TimeSeries **signalTD, COMPLEX16FrequencySeries *signalFD, LIGOTimeGPS *tRef, REAL8 totalMass, REAL8 deltaT, REAL8 f_min, REAL8 f_max, REAL8 f_min_wide, REAL8 f_max_wide);
static ssize_t find_instant_freq(REAL8TimeSeries *hp, REAL8TimeSeries *hc, REAL8 target, ssize_t start);
static int apply_inclination(REAL8TimeSeries **hplus, REAL8TimeSeries **hcross, REAL8 inclination);


/**
 *
 * main functions
 *
 */

/**
 * Driver routine to compute the non-spinning, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomA in the frequency domain.
 *
 * Reference:
 *   - Waveform: Eq.(4.13) and (4.16) of http://arxiv.org/pdf/0710.2335
 *   - Coefficients: Eq.(4.18) of http://arxiv.org/pdf/0710.2335 and
 *                   Table I of http://arxiv.org/pdf/0712.0343
 *
 * All input parameters should be SI units.
 */
int XLALSimIMRPhenomAGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    LIGOTimeGPS *tRef,                 /**< time at fRef */
    REAL8 phiRef,                      /**< phase at fRef */
    REAL8 fRef,                        /**< reference frequency */
    REAL8 deltaF,                      /**< frequency resolution */
    REAL8 m1,                          /**< mass of companion 1 */
    REAL8 m2,                          /**< mass of companion 2 */
    REAL8 f_min,                       /**< start frequency */
    REAL8 f_max,                       /**< end frequency; if 0, set to fCut */
    REAL8 distance                     /**< distance of source */
) {
  BBHPhenomParams *params;

  /* check inputs for sanity */
  if (*htilde) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (!tRef) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (fRef <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (deltaF <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m1 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m2 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_min <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_max < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (distance <= 0) XLAL_ERROR(__func__, XLAL_EDOM);

  /* external: SI; internal: solar masses */
  m1 /= LAL_MSUN_SI;
  m2 /= LAL_MSUN_SI;

  /* phenomenological parameters*/
  params = ComputeIMRPhenomAParams(m1, m2);
  if (!params) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* default f_max to params->fCut */
  if (f_max == 0.) f_max = params->fCut;

  return IMRPhenomAGenerateFD(htilde, tRef, phiRef, fRef, deltaF, m1, m2, f_min, f_max, distance, params);
}

/**
 * Driver routine to compute the non-spinning, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomA in the time domain.
 *
 * Reference:
 *   - Waveform: Eq.(4.13) and (4.16) of http://arxiv.org/pdf/0710.2335
 *   - Coefficients: Eq.(4.18) of http://arxiv.org/pdf/0710.2335 and
 *                   Table I of http://arxiv.org/pdf/0712.0343
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomAGenerateTD(
    REAL8TimeSeries **hplus,  /**< +-polarization waveform */
    REAL8TimeSeries **hcross, /**< x-polarization waveform */
    LIGOTimeGPS *tRef,        /**< time at fRef */
    REAL8 phiRef,             /**< phase at fRef */
    REAL8 fRef,               /**< reference frequency */
    REAL8 deltaT,             /**< sampling interval */
    REAL8 m1,                 /**< mass of companion 1 */
    REAL8 m2,                 /**< mass of companion 2 */
    REAL8 f_min,              /**< start frequency */
    REAL8 f_max,              /**< end frequency */
    REAL8 distance,           /**< distance of source */
    REAL8 inclination         /**< inclination of source */
) {
  BBHPhenomParams *params;
  ssize_t cut_ind;

  /* check inputs for sanity */
  if (*hplus) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (*hcross) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (!tRef) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (fRef <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (deltaT <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m1 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m2 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_min <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_max < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (distance <= 0) XLAL_ERROR(__func__, XLAL_EDOM);

  /* external: SI; internal: solar masses */
  m1 /= LAL_MSUN_SI;
  m2 /= LAL_MSUN_SI;

  /* phenomenological parameters*/
  params = ComputeIMRPhenomAParams(m1, m2);
  if (!params) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* default f_max to params->fCut */
  if (f_max == 0.) f_max = params->fCut;

  /* generate hplus */
  IMRPhenomAGenerateTD(hplus, tRef, phiRef, fRef, deltaT, m1, m2, f_min, f_max, distance, params);
  if (!(*hplus)) {
      XLALFree(params);
      XLAL_ERROR(__func__, XLAL_EFUNC);
  }

  /* generate hcross, which is hplus phase-shifted by pi/2 */
  IMRPhenomAGenerateTD(hcross, tRef, phiRef + LAL_PI / 2, fRef, deltaT, m1, m2, f_min, f_max, distance, params);
  XLALFree(params);
  if (!(*hcross)) {
      XLALDestroyREAL8TimeSeries(*hplus);
      *hplus = NULL;
      XLAL_ERROR(__func__, XLAL_EFUNC);
  }

  /* clip the parts below f_min */
  cut_ind = find_instant_freq(*hplus, *hcross, f_min, (*hplus)->data->length - EstimateIMRLength(m1, m2, f_min, deltaT) + EstimateIMRLength(m1, m2, f_max, deltaT));
  *hplus = XLALResizeREAL8TimeSeries(*hplus, cut_ind, (*hplus)->data->length - cut_ind);
  *hcross = XLALResizeREAL8TimeSeries(*hcross, cut_ind, (*hcross)->data->length - cut_ind);
  if (!(*hplus) || !(*hcross))
    XLAL_ERROR(__func__, XLAL_EFUNC);

  /* apply inclination */
  return apply_inclination(hplus, hcross, inclination);
}

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomB in the time domain.
 *
 * Reference: http://arxiv.org/pdf/0909.2867
 *   - Waveform: Eq.(1)
 *   - Coefficients: Eq.(2) and Table I
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomBGenerateTD(
    REAL8TimeSeries **hplus,  /**< +-polarization waveform */
    REAL8TimeSeries **hcross, /**< x-polarization waveform */
    LIGOTimeGPS *tRef,        /**< time at fRef */
    REAL8 phiRef,             /**< phase at fRef */
    REAL8 fRef,               /**< reference frequency */
    REAL8 deltaT,             /**< sampling interval */
    REAL8 m1,                 /**< mass of companion 1 */
    REAL8 m2,                 /**< mass of companion 2 */
    REAL8 chi,                /**< mass-weighted aligned-spin parameter */
    REAL8 f_min,              /**< start frequency */
    REAL8 f_max,              /**< end frequency */
    REAL8 distance,           /**< distance of source */
    REAL8 inclination         /**< inclination of source */
) {
  BBHPhenomParams *params;
  ssize_t cut_ind;

  /* check inputs for sanity */
  if (*hplus) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (*hcross) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (!tRef) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (fRef <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (deltaT <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m1 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m2 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (fabs(chi) > 1) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_min <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_max < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (distance <= 0) XLAL_ERROR(__func__, XLAL_EDOM);

  /* external: SI; internal: solar masses */
  m1 /= LAL_MSUN_SI;
  m2 /= LAL_MSUN_SI;

  /* phenomenological parameters*/
  params = ComputeIMRPhenomBParams(m1, m2, chi);
  if (!params) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* default f_max to params->fCut */
  if (f_max == 0.) f_max = params->fCut;

  /* generate plus */
  IMRPhenomBGenerateTD(hplus, tRef, phiRef, fRef, deltaT, m1, m2, chi, f_min, f_max, distance, params);
  if (!(*hplus)) {
      XLALFree(params);
      XLAL_ERROR(__func__, XLAL_EFUNC);
  }

  /* generate cross, phase-shifted by pi/2 */
  IMRPhenomBGenerateTD(hcross, tRef, phiRef + LAL_PI / 2, fRef, deltaT, m1, m2, chi, f_min, f_max, distance, params);
  XLALFree(params);
  if (!(*hcross)) {
      XLALDestroyREAL8TimeSeries(*hplus);
      *hplus = NULL;
      XLAL_ERROR(__func__, XLAL_EFUNC);
  }

  /* clip the parts below f_min */
  cut_ind = find_instant_freq(*hplus, *hcross, f_min, (*hplus)->data->length - EstimateIMRLength(m1, m2, f_min, deltaT) + EstimateIMRLength(m1, m2, f_max, deltaT));
  *hplus = XLALResizeREAL8TimeSeries(*hplus, cut_ind, (*hplus)->data->length - cut_ind);
  *hcross = XLALResizeREAL8TimeSeries(*hcross, cut_ind, (*hcross)->data->length - cut_ind);
  if (!(*hplus) || !(*hcross))
    XLAL_ERROR(__func__, XLAL_EFUNC);

  /* apply inclination */
  return apply_inclination(hplus, hcross, inclination);
}


/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomB in the frequency domain.
 *
 * Reference: http://arxiv.org/pdf/0909.2867
 *   - Waveform: Eq.(1)
 *   - Coefficients: Eq.(2) and Table I
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomBGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    LIGOTimeGPS *tRef,                 /**< time at fRef */
    REAL8 phiRef,                      /**< phase at fRef */
    REAL8 fRef,                        /**< reference frequency */
    REAL8 deltaF,                      /**< sampling interval */
    REAL8 m1,                          /**< mass of companion 1 */
    REAL8 m2,                          /**< mass of companion 2 */
    REAL8 chi,                         /**< mass-weighted aligned-spin parameter */
    REAL8 f_min,                       /**< start frequency */
    REAL8 f_max,                       /**< end frequency */
    REAL8 distance                     /**< distance of source */
) {
  BBHPhenomParams *params;
  int status;

  /* check inputs for sanity */
  if (*htilde) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (!tRef) XLAL_ERROR(__func__, XLAL_EFAULT);
  if (fRef <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (deltaF <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m1 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (m2 < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (fabs(chi) > 1) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_min <= 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (f_max < 0) XLAL_ERROR(__func__, XLAL_EDOM);
  if (distance <= 0) XLAL_ERROR(__func__, XLAL_EDOM);

  /* external: SI; internal: solar masses */
  m1 /= LAL_MSUN_SI;
  m2 /= LAL_MSUN_SI;

  /* phenomenological parameters*/
  params = ComputeIMRPhenomBParams(m1, m2, chi);
  if (!params) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* default f_max to params->fCut */
  if (f_max == 0.) f_max = params->fCut;

  status = IMRPhenomBGenerateFD(htilde, tRef, phiRef, fRef, deltaF, m1, m2, chi, f_min, f_max, distance, params);
  LALFree(params);
  return status;
}

/*********************************************************************/
/* Compute phenomenological parameters for non-spinning binaries     */
/* Ref. Eq.(4.18) of http://arxiv.org/pdf/0710.2335 and              */
/* Table I of http://arxiv.org/pdf/0712.0343                         */
/*                                                                   */
/* Takes solar masses.                                               */
/*********************************************************************/
static BBHPhenomParams *ComputeIMRPhenomAParams(REAL8 m1, REAL8 m2) {
  REAL8 totalMass, piM, eta, fMerg_a, fMerg_b, fMerg_c, fRing_a, fRing_b, etap2;
  REAL8 fRing_c, sigma_a, sigma_b, sigma_c, fCut_a, fCut_b, fCut_c;
  REAL8 psi0_a, psi0_b, psi0_c, psi2_a, psi2_b, psi2_c, psi3_a, psi3_b, psi3_c;
  REAL8 psi4_a, psi4_b, psi4_c, psi6_a, psi6_b, psi6_c, psi7_a, psi7_b, psi7_c;
  BBHPhenomParams *phenParams;

  phenParams = (BBHPhenomParams *) XLALMalloc(sizeof(BBHPhenomParams));
  if (!phenParams) XLAL_ERROR_NULL(__func__, XLAL_EFUNC);
  memset(phenParams, 0, sizeof(BBHPhenomParams));

  /* calculate the total mass and symmetric mass ratio */
  totalMass = m1 + m2;
  eta = m1 * m2 / (totalMass * totalMass);
  piM = totalMass * LAL_PI * LAL_MTSUN_SI;

  fMerg_a = 6.6389e-01;
  fMerg_b = -1.0321e-01;
  fMerg_c = 1.0979e-01;

  fRing_a = 1.3278e+00;
  fRing_b = -2.0642e-01;
  fRing_c = 2.1957e-01;

  sigma_a = 1.1383e+00;
  sigma_b = -1.7700e-01;
  sigma_c = 4.6834e-02;

  fCut_a = 1.7086e+00;
  fCut_b = -2.6592e-01;
  fCut_c = 2.8236e-01;

  psi0_a = -1.5829e-01;
  psi0_b = 8.7016e-02;
  psi0_c = -3.3382e-02;

  psi2_a = 3.2967e+01;
  psi2_b = -1.9000e+01;
  psi2_c = 2.1345e+00;

  psi3_a = -3.0849e+02;
  psi3_b = 1.8211e+02;
  psi3_c = -2.1727e+01;

  psi4_a = 1.1525e+03;
  psi4_b = -7.1477e+02;
  psi4_c = 9.9692e+01;

  psi6_a = 1.2057e+03;
  psi6_b = -8.4233e+02;
  psi6_c = 1.8046e+02;

  psi7_a = 0.;
  psi7_b = 0.;
  psi7_c = 0.;

  /* Evaluate the polynomials. See Eq. (4.18) of P. Ajith et al
   * arXiv:0710.2335 [gr-qc] */
  etap2 = eta*eta;
  phenParams->fCut = (fCut_a*etap2  + fCut_b*eta  + fCut_c)/piM;
  phenParams->fMerger = (fMerg_a*etap2  + fMerg_b*eta  + fMerg_c)/piM;
  phenParams->fRing = (fRing_a*etap2 + fRing_b*eta + fRing_c)/piM;
  phenParams->sigma = (sigma_a*etap2 + sigma_b*eta + sigma_c)/piM;

  phenParams->psi0 = (psi0_a*etap2 + psi0_b*eta + psi0_c)/(eta*pow(piM, 5./3.));
  phenParams->psi1 = 0.;
  phenParams->psi2 = (psi2_a*etap2 + psi2_b*eta + psi2_c)/(eta*pow(piM, 3./3.));
  phenParams->psi3 = (psi3_a*etap2 + psi3_b*eta + psi3_c)/(eta*pow(piM, 2./3.));
  phenParams->psi4 = (psi4_a*etap2 + psi4_b*eta + psi4_c)/(eta*pow(piM, 1./3.));
  phenParams->psi5 = 0.;
  phenParams->psi6 = (psi6_a*etap2 + psi6_b*eta + psi6_c)/(eta*pow(piM, -1./3.));
  phenParams->psi7 = (psi7_a*etap2 + psi7_b*eta + psi7_c)/(eta*pow(piM, -2./3.));

  return phenParams;
}

/*********************************************************************/
/* Compute phenomenological parameters for aligned-spin binaries     */
/* Ref. Eq.(2) and Table I of http://arxiv.org/pdf/0909.2867         */
/*                                                                   */
/* Takes solar masses. Populates and returns a new BBHPhenomParams   */
/* structure.                                                        */
/*********************************************************************/
static BBHPhenomParams *ComputeIMRPhenomBParams(REAL8 m1, REAL8 m2, REAL8 chi) {
  REAL8 totalMass, piM, eta;
  REAL8 etap2, chip2, etap3, etap2chi, etachip2, etachi;
  BBHPhenomParams *phenParams;

  phenParams = (BBHPhenomParams *) XLALMalloc(sizeof(BBHPhenomParams));
  if (!phenParams) XLAL_ERROR_NULL(__func__, XLAL_EFUNC);
  memset(phenParams, 0, sizeof(BBHPhenomParams));

  /* calculate the total mass and symmetric mass ratio */
  totalMass = m1 + m2;
  eta = m1 * m2 / (totalMass * totalMass);
  piM = totalMass * LAL_PI * LAL_MTSUN_SI;

  /* spinning phenomenological waveforms */
  etap2 = eta*eta;
  chip2 = chi*chi;
  etap3 = etap2*eta;
  etap2chi = etap2*chi;
  etachip2 = eta*chip2;
  etachi = eta*chi;

  phenParams->psi0 = 3./(128.*eta);

  phenParams->psi2 = 3715./756. +
  -9.2091e+02*eta + 4.9213e+02*etachi + 1.3503e+02*etachip2 +
  6.7419e+03*etap2 + -1.0534e+03*etap2chi +
  -1.3397e+04*etap3 ;

  phenParams->psi3 = -16.*LAL_PI + 113.*chi/3. +
  1.7022e+04*eta + -9.5659e+03*etachi + -2.1821e+03*etachip2 +
  -1.2137e+05*etap2 + 2.0752e+04*etap2chi +
  2.3859e+05*etap3 ;

  phenParams->psi4 = 15293365./508032. - 405.*chip2/8. +
  -1.2544e+05*eta + 7.5066e+04*etachi + 1.3382e+04*etachip2 +
  8.7354e+05*etap2 + -1.6573e+05*etap2chi +
  -1.6936e+06*etap3 ;

  phenParams->psi6 = -8.8977e+05*eta + 6.3102e+05*etachi + 5.0676e+04*etachip2 +
  5.9808e+06*etap2 + -1.4148e+06*etap2chi +
  -1.1280e+07*etap3 ;

  phenParams->psi7 = 8.6960e+05*eta + -6.7098e+05*etachi + -3.0082e+04*etachip2 +
  -5.8379e+06*etap2 + 1.5145e+06*etap2chi +
  1.0891e+07*etap3 ;

  phenParams->psi8 = -3.6600e+05*eta + 3.0670e+05*etachi + 6.3176e+02*etachip2 +
  2.4265e+06*etap2 + -7.2180e+05*etap2chi +
  -4.5524e+06*etap3;

  phenParams->fMerger =  1. - 4.4547*pow(1.-chi,0.217) + 3.521*pow(1.-chi,0.26) +
  6.4365e-01*eta + 8.2696e-01*etachi + -2.7063e-01*etachip2 +
  -5.8218e-02*etap2 + -3.9346e+00*etap2chi +
  -7.0916e+00*etap3 ;

  phenParams->fRing = (1. - 0.63*pow(1.-chi,0.3))/2. +
  1.4690e-01*eta + -1.2281e-01*etachi + -2.6091e-02*etachip2 +
  -2.4900e-02*etap2 + 1.7013e-01*etap2chi +
  2.3252e+00*etap3 ;

  phenParams->sigma = (1. - 0.63*pow(1.-chi,0.3))*pow(1.-chi,0.45)/4. +
  -4.0979e-01*eta + -3.5226e-02*etachi + 1.0082e-01*etachip2 +
  1.8286e+00*etap2 + -2.0169e-02*etap2chi +
  -2.8698e+00*etap3 ;

  phenParams->fCut = 3.2361e-01 + 4.8935e-02*chi + 1.3463e-02*chip2 +
  -1.3313e-01*eta + -8.1719e-02*etachi + 1.4512e-01*etachip2 +
  -2.7140e-01*etap2 + 1.2788e-01*etap2chi +
  4.9220e+00*etap3 ;

  phenParams->fCut   /= piM;
  phenParams->fMerger/= piM;
  phenParams->fRing  /= piM;
  phenParams->sigma  /= piM;

  phenParams->psi1    = 0.;
  phenParams->psi5    = 0.;

  return phenParams;
}

/**
 * Return tau0, the Newtonian chirp length estimate.
 */
static REAL8 ComputeTau0(REAL8 m1, REAL8 m2, REAL8 f_min) {
  REAL8 totalMass, eta;

  totalMass = m1 + m2;
  eta = m1 * m2 / (totalMass * totalMass);
  return 5. * totalMass * LAL_MTSUN_SI / (256. * eta * pow(LAL_PI * totalMass * LAL_MTSUN_SI * f_min, 8./3.));
}

/**
 * Estimate the length of a TD vector that can hold the waveform as the Newtonian
 * chirp time tau0 plus 1000 M.
 */
static ssize_t EstimateIMRLength(REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 deltaT) {
  return (ssize_t) floor((ComputeTau0(m1, m2, f_min) + 1000 * (m1 + m2) * LAL_MTSUN_SI) / deltaT);
}

static ssize_t NextPow2(ssize_t n) {
  return 1 << (ssize_t) ceil(log2(n));
}

/**
 * Find a lower value for f_min (using the definition of Newtonian chirp
 * time) such that the waveform has a minimum length of tau0. This is
 * necessary to avoid FFT artifacts.
 */
static REAL8 EstimateSafeFMinForTD(REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 deltaT) {
  REAL8 temp_f_min, totalMass, eta, tau0;

  totalMass = m1 + m2;
  eta = m1 * m2 / (totalMass * totalMass);
  tau0 = deltaT * NextPow2(1.025 * EstimateIMRLength(m1, m2, f_min, deltaT));
  temp_f_min = pow((tau0 * 256. * eta * pow(totalMass * LAL_MTSUN_SI, 5./3.) / 5.), -3./8.) / LAL_PI;
  if (temp_f_min > f_min) temp_f_min = f_min;
  if (temp_f_min < 0.5) temp_f_min = 0.5;
  return temp_f_min;
}

/**
 * Find a higher value of f_max so that we can safely apply a window later.
 */
static REAL8 EstimateSafeFMaxForTD(REAL8 f_max, REAL8 deltaT) {
  REAL8 temp_f_max;
  temp_f_max = 1.025 * f_max;

  /* make sure that these frequencies are not too out of range */
  if (temp_f_max > 2. / deltaT - 100.) temp_f_max = 2. / deltaT - 100.;
  return temp_f_max;
}

static REAL8 LorentzianFn (
    REAL8 freq,
    REAL8 fRing,
    REAL8 sigma) {
  return sigma / (2 * LAL_PI * ((freq - fRing)*(freq - fRing)
    + sigma*sigma / 4.0));
}


/**
 * Private function to generate IMRPhenomA frequency-domain waveforms given coefficients
 */
static int IMRPhenomAGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    LIGOTimeGPS *tRef,                 /**< time at fRef */
    REAL8 phiRef,                      /**< phase at fRef */
    REAL8 fRef,                        /**< reference frequency */
    REAL8 deltaF,                      /**< frequency resolution */
    REAL8 m1,                          /**< mass of companion 1 [solar masses] */
    REAL8 m2,                          /**< mass of companion 2 [solar masses] */
    REAL8 f_min,                       /**< start frequency */
    REAL8 f_max,                       /**< end frequency */
    REAL8 distance,                    /**< distance of source */
    BBHPhenomParams *params            /**< from ComputeIMRPhenomAParams */
) {
  REAL8 shft, amp0, fMerg, fRing, sigma, totalMass, eta;
  ssize_t i, n;

  fMerg = params->fMerger;
  fRing = params->fRing;
  sigma = params->sigma;
  totalMass = m1 + m2;
  eta = m1 * m2 / (totalMass * totalMass);

  /* compute the amplitude pre-factor */
  amp0 = pow(LAL_MTSUN_SI*totalMass, 5./6.) * pow(fMerg, -7./6.)
    / pow(LAL_PI, 2./3.) * sqrt(5. * eta / 24.) / (distance / LAL_C_SI);

  /* allocate htilde */
  n = NextPow2(f_max / deltaF) + 1;
  *htilde = XLALCreateCOMPLEX16FrequencySeries("htilde: FD waveform", tRef, 0.0, deltaF, &lalStrainUnit, n);
  memset((*htilde)->data->data, 0, n * sizeof(COMPLEX16));
  XLALUnitDivide(&((*htilde)->sampleUnits), &((*htilde)->sampleUnits), &lalSecondUnit);
  if (!(*htilde)) XLAL_ERROR(__func__, XLAL_EFUNC);

  shft = LAL_TWOPI * (tRef->gpsSeconds + 1e-9 * tRef->gpsNanoSeconds);

  /* now generate the waveform at all frequency bins except DC and Nyquist */
  for (i=1; i < n - 1; i++) {
    REAL8 ampEff, psiEff;
    /* Fourier frequency corresponding to this bin */
    REAL8 f = i * deltaF;
    REAL8 fNorm = f / fMerg;

    /* compute the amplitude */
    if ((f < f_min) || (f > f_max)) continue;
    else if (f <= fMerg) ampEff = amp0 * pow(fNorm, -7./6.);
    else if ((f > fMerg) & (f <= fRing)) ampEff = amp0 * pow(fNorm, -2./3.);
    else if (f > fRing)
      ampEff = amp0 * LAL_PI_2 * pow(fRing / fMerg, -2./3.) * sigma
        * LorentzianFn(f, fRing, sigma);
    else {
      XLALDestroyCOMPLEX16FrequencySeries(*htilde);
      *htilde = NULL;
      XLAL_ERROR(__func__, XLAL_EDOM);
    }

    /* now compute the phase */
    psiEff = shft * (f - fRef) + phiRef  /* use reference freq. and phase */
      + params->psi0 * pow(f, -5./3.)
      + params->psi1 * pow(f, -4./3.)
      + params->psi2 * pow(f, -3./3.)
      + params->psi3 * pow(f, -2./3.)
      + params->psi4 * pow(f, -1./3.)
      + params->psi5 // * pow(f, 0.)
      + params->psi6 * cbrt(f)
      + params->psi7 * pow(f, 2./3.);

    /* generate the waveform */
    ((*htilde)->data->data)[i].re = ampEff * cos(psiEff);
    ((*htilde)->data->data)[i].im = ampEff * sin(psiEff);
  }

  return XLAL_SUCCESS;
}

/**
 * Private function to generate IMRPhenomB frequency-domain waveforms given coefficients
 */
static int IMRPhenomBGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    LIGOTimeGPS *tRef,                 /**< time at fRef */
    REAL8 phiRef,                      /**< phase at fRef */
    REAL8 fRef,                        /**< reference frequency */
    REAL8 deltaF,                      /**< frequency resolution */
    REAL8 m1,                          /**< mass of companion 1 [solar masses] */
    REAL8 m2,                          /**< mass of companion 2 [solar masses] */
    REAL8 chi,                         /**< mass-weighted aligned-spin parameter */
    REAL8 f_min,                       /**< start frequency */
    REAL8 f_max,                       /**< end frequency; if 0 */
    REAL8 distance,                    /**< distance of source */
    BBHPhenomParams *params            /**< from ComputeIMRPhenomBParams */
) {
  REAL8 shft, amp0, fMerg, fRing, sigma, totalMass, eta;
  REAL8 alpha2, alpha3, mergPower, epsilon_1, epsilon_2, vMerg, vRing, w1, w2;
  ssize_t i, n;

  fMerg = params->fMerger;
  fRing = params->fRing;
  sigma = params->sigma;
  totalMass = m1 + m2;
  eta = m1 * m2 / (totalMass * totalMass);

  /* compute the amplitude pre-factor */
  amp0 = pow(LAL_MTSUN_SI*totalMass, 5./6.) * pow(fMerg, -7./6.)
    / pow(LAL_PI, 2./3.) * sqrt(5. * eta / 24.) / (distance / LAL_C_SI);

  /* allocate htilde */
  n = NextPow2(f_max / deltaF) + 1;
  *htilde = XLALCreateCOMPLEX16FrequencySeries("htilde: FD waveform", tRef, 0.0, deltaF, &lalStrainUnit, n);
  memset((*htilde)->data->data, 0, n * sizeof(COMPLEX16));
  XLALUnitDivide(&((*htilde)->sampleUnits), &((*htilde)->sampleUnits), &lalSecondUnit);
  if (!(*htilde)) XLAL_ERROR(__func__, XLAL_EFUNC);

  shft = LAL_TWOPI * (tRef->gpsSeconds + 1e-9 * tRef->gpsNanoSeconds);

  /***********************************************************************/
  /* these are the parameters required for the "new" phenomenological IMR
   * waveforms*/
  /***********************************************************************/

  /* PN corrections to the frequency domain amplitude of the (2,2) mode */
  alpha2   = -323./224. + 451.*eta/168.;
  alpha3   = (27./8. - 11.*eta/6.)*chi;

  /* leading order power law of the merger amplitude */
  mergPower = -2./3.;

  /* spin-dependent corrections to the merger amplitude */
  epsilon_1 =  1.4547*chi - 1.8897;
  epsilon_2 = -1.8153*chi + 1.6557;

  /* normalisation constant of the inspiral amplitude */
  vMerg = cbrt(LAL_PI * totalMass * LAL_MTSUN_SI * fMerg);
  vRing = cbrt(LAL_PI * totalMass * LAL_MTSUN_SI * fRing);

  w1 = 1. + alpha2 * vMerg * vMerg + alpha3*pow(vMerg, 3.);
  w1 = w1/(1. + epsilon_1 * vMerg + epsilon_2 * vMerg * vMerg);
  w2 = w1 * (LAL_PI * sigma / 2.) * pow(fRing / fMerg, mergPower)
          * (1. + epsilon_1 * vRing + epsilon_2 * vRing * vRing);

  /* now generate the waveform at all frequency bins except DC and Nyquist */
  for (i=1; i < n - 1; i++) {
    REAL8 ampEff, psiEff;
    REAL8 v, v2, v3, v4, v5, v6, v7, v8;

    /* Fourier frequency corresponding to this bin */
    REAL8 f = i * deltaF;
    REAL8 fNorm = f / fMerg;

    /* PN expansion parameter */
    v = cbrt(LAL_PI * totalMass * LAL_MTSUN_SI * f);
    v2 = v*v; v3 = v2*v; v4 = v2*v2; v5 = v4*v; v6 = v3*v3; v7 = v6*v, v8 = v7*v;

    /* compute the amplitude */
    if ((f < f_min) || (f > f_max))
      continue;
    else if (f <= fMerg)
      ampEff = pow(fNorm, -7./6.)*(1. + alpha2 * v2 + alpha3 * v3);
    else if ((f > fMerg) & (f <= fRing))
      ampEff = w1 * pow(fNorm, mergPower) * (1. + epsilon_1 * v + epsilon_2 * v2);
    else if (f > fRing)
      ampEff = w2 * LorentzianFn(f, fRing, sigma);
    else {
      XLALDestroyCOMPLEX16FrequencySeries(*htilde);
      *htilde = NULL;
      XLAL_ERROR(__func__, XLAL_EDOM);
    }

    /* now compute the phase */
    psiEff = shft * (f - fRef) - phiRef  /* use reference freq. and phase; phi is flipped relative to IMRPhenomA */
      + 3./(128.*eta*v5)*(1 + params->psi2*v2
      + params->psi3*v3 + params->psi4*v4
      + params->psi5*v5 + params->psi6*v6
      + params->psi7*v7 + params->psi8*v8);

    /* generate the waveform */
    ((*htilde)->data->data)[i].re = amp0 * ampEff * cos(psiEff);
    ((*htilde)->data->data)[i].im = -amp0 * ampEff * sin(psiEff);
  }

  return XLAL_SUCCESS;
}

/**
 * Private function to generate time-domain waveforms given coefficients
 */
static int IMRPhenomAGenerateTD(REAL8TimeSeries **h, LIGOTimeGPS *tRef, REAL8 phiRef, REAL8 fRef, REAL8 deltaT, REAL8 m1, REAL8 m2, REAL8 f_min, REAL8 f_max, REAL8 distance, BBHPhenomParams *params) {
  REAL8 f_min_wide, f_max_wide, deltaF;
  COMPLEX16FrequencySeries *htilde;
  /* We will generate the waveform from a frequency which is lower than the
   * f_min chosen. Also the cutoff frequency may be higher than the f_max. We
   * will later apply a window function, and truncate the time-domain waveform
   * below an instantaneous frequency f_min. */
  f_min_wide = EstimateSafeFMinForTD(m1, m2, f_min, deltaT);
  f_max_wide = 0.5 / deltaT;
  if (EstimateSafeFMaxForTD(f_max, deltaT) > f_max_wide)
    XLALPrintWarning("Warning: sampling rate too low to capture chosen f_max\n");
  deltaF = 1. / (deltaT * NextPow2(EstimateIMRLength(m1, m2, f_min_wide, deltaT)));

  /* generate in frequency domain */
  if (IMRPhenomAGenerateFD(&htilde, tRef, phiRef, fRef, deltaF, m1, m2, f_min_wide, f_max_wide, distance, params)) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* convert to time domain */
  FDToTD(h, htilde, tRef, m1 + m2, deltaT, f_min, f_max, f_min_wide, f_max_wide);
  XLALDestroyCOMPLEX16FrequencySeries(htilde);
  if (!*h) XLAL_ERROR(__func__, XLAL_EFUNC);

  return XLAL_SUCCESS;
}

/**
 * Private function to generate time-domain waveforms given coefficients
 */
static int IMRPhenomBGenerateTD(REAL8TimeSeries **h, LIGOTimeGPS *tRef, REAL8 phiRef, REAL8 fRef, REAL8 deltaT, REAL8 m1, REAL8 m2, REAL8 chi, REAL8 f_min, REAL8 f_max, REAL8 distance, BBHPhenomParams *params) {
  REAL8 f_min_wide, f_max_wide, deltaF;
  COMPLEX16FrequencySeries *htilde;
  /* We will generate the waveform from a frequency which is lower than the
   * f_min chosen. Also the cutoff frequency is higher than the f_max. We
   * will later apply a window function, and truncate the time-domain waveform
   * below an instantaneous frequency f_min. */
  f_min_wide = EstimateSafeFMinForTD(m1, m2, f_min, deltaT);
  f_max_wide = 0.5 / deltaT;
  if (EstimateSafeFMaxForTD(f_max, deltaT) > f_max_wide)
    XLALPrintWarning("Warning: sampling rate too low for expected spectral content\n");
  deltaF = 1. / (deltaT * NextPow2(EstimateIMRLength(m1, m2, f_min_wide, deltaT)));

  /* generate in frequency domain */
  if (IMRPhenomBGenerateFD(&htilde, tRef, phiRef, fRef, deltaF, m1, m2, chi, f_min_wide, f_max_wide, distance, params)) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* convert to time domain */
  FDToTD(h, htilde, tRef, m1 + m2, deltaT, f_min, f_max, f_min_wide, f_max_wide);
  XLALDestroyCOMPLEX16FrequencySeries(htilde);
  if (!*h) XLAL_ERROR(__func__, XLAL_EFUNC);

  return XLAL_SUCCESS;
}

/**
 * Window and IFFT a FD waveform to TD, then window in TD.
 * Requires that the FD waveform be generated outside of f_min and f_max.
 * FD waveform is modified.
 */
static int FDToTD(REAL8TimeSeries **signalTD, COMPLEX16FrequencySeries *signalFD, LIGOTimeGPS *tRef, REAL8 totalMass, REAL8 deltaT, REAL8 f_min, REAL8 f_max, REAL8 f_min_wide, REAL8 f_max_wide) {
  REAL8 f, deltaF, winFLo, winFHi, softWin, windowLength;
  REAL8FFTPlan *revPlan;
  ssize_t nf, nt, k;

  /* check inputs */
  if (f_min_wide >= f_min) XLAL_ERROR(__func__, XLAL_EDOM);

  /* apply the softening window function */
  nf = signalFD->data->length;
  nt = 2 * (nf - 1);
  deltaF = 1. / (deltaT * nt);

  winFLo = (f_min + f_min_wide) / 2.;
  winFHi = (f_max + f_max_wide) / 2.;
  if (winFHi > 0.5 / deltaT) winFHi = 0.5 / deltaT;

  for (k = 0; k < nf; k++) {
    f = k * deltaF;
    softWin = (1. + tanh(f - winFLo))
            * (1. - tanh(f - winFHi)) / 4.;
    signalFD->data->data[k].re *= softWin;
    signalFD->data->data[k].im *= softWin;
  }

  /* allocate output */
  *signalTD = XLALCreateREAL8TimeSeries("h", tRef, 0.0, deltaT, &lalStrainUnit, nt);

  /* Inverse Fourier transform */
  revPlan = XLALCreateReverseREAL8FFTPlan(nt, 1);
  if (!revPlan) {
    XLALDestroyREAL8TimeSeries(*signalTD);
    *signalTD = NULL;
    XLAL_ERROR(__func__, XLAL_EFUNC);
  }
  XLALREAL8FreqTimeFFT(*signalTD, signalFD, revPlan);
  XLALDestroyREAL8FFTPlan(revPlan);
  if (!(*signalTD)) XLAL_ERROR(__func__, XLAL_EFUNC);

  /* apply a linearly decreasing window at the end
   * of the waveform in order to avoid edge effects. */
  windowLength = 20. * totalMass * LAL_MTSUN_SI / deltaT;
  if (windowLength > (*signalTD)->data->length) XLAL_ERROR(__func__, XLAL_ERANGE);
  for (k = 0; k < windowLength; k++)
    (*signalTD)->data->data[nt-k-1] *= k / windowLength;

  return XLAL_SUCCESS;
}

/* return the index before the instantaneous frequency rises past target */
static ssize_t find_instant_freq(REAL8TimeSeries *hp, REAL8TimeSeries *hc, REAL8 target, ssize_t start) {
  ssize_t k;

  /* Use second order differencing to find the instantaneous frequency as
   * h = A e^(2 pi i f t) ==> f = d/dt(h) / (2*pi*h) */
  for (k = start; k < hp->data->length - 1; k++) {
    REAL8 hpDot, hcDot, f;
    hpDot = (hp->data->data[k+1] - hp->data->data[k-1]) / (2 * hp->deltaT);
    hcDot = (hc->data->data[k+1] - hc->data->data[k-1]) / (2 * hc->deltaT);
    f = -hcDot * hp->data->data[k] + hpDot * hc->data->data[k];
    f /= LAL_TWOPI;
    f /= hp->data->data[k] * hp->data->data[k] + hc->data->data[k] * hc->data->data[k];
    if (f >= target) return k - 1;
  }
  XLAL_ERROR(__func__, XLAL_EDOM);
}

static int apply_inclination(REAL8TimeSeries **hplus, REAL8TimeSeries **hcross, REAL8 inclination) {
  REAL8 inclFacPlus, inclFacCross, cosI;
  ssize_t k;

  cosI = cos(inclination);

  inclFacPlus = -0.5 * (1. + cosI * cosI);
  inclFacCross = -cosI;
  for (k = 0; k < (*hplus)->data->length; k++) {
      (*hplus)->data->data[k] *= inclFacPlus;
      (*hcross)->data->data[k] *= inclFacCross;
  }

  return XLAL_SUCCESS;
}

/*
 * Copyright (C) 2012 Prayush Kumar, Frank Ohme, 2015 Michael Puerrer
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


/* The paper refered to here as the Main paper, is Phys. Rev. D 82, 064016 (2010)
 * */

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

#include <math.h>
#include <complex.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <lal/LALStdlib.h>
#include <lal/LALSimIMR.h>
#include <lal/LALConstants.h>
#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/StringInput.h>
#include <lal/TimeSeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/Units.h>

#include "LALSimIMRPhenomC_internals.c"

#ifndef _OPENMP
#define omp ignore
#endif

/*
 * private function prototypes; all internal functions use solar masses.
 *
 */

static int IMRPhenomCGenerateFD(COMPLEX16FrequencySeries **htilde, const REAL8 phi0, const REAL8 deltaF, const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 f_max, const REAL8 distance, const BBHPhenomCParams *params);

static int IMRPhenomCGenerateFDForTD(COMPLEX16FrequencySeries **htilde, const REAL8 t0, const REAL8 phi0, const REAL8 deltaF, const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 f_max, const REAL8 distance, const BBHPhenomCParams *params, const size_t nf);
static int IMRPhenomCGenerateTD(REAL8TimeSeries **h, const REAL8 phiPeak, size_t *ind_t0, const REAL8 deltaT, const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 f_max, const REAL8 distance, const BBHPhenomCParams *params);

static REAL8 EstimateSafeFMinForTD(const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 deltaT);
static REAL8 EstimateSafeFMaxForTD(const REAL8 f_max, const REAL8 dt);
static REAL8 ComputeTau0(const REAL8 m1, const REAL8 m2, const REAL8 f_min);
static size_t EstimateIMRLength(const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 deltaT);

static int FDToTD(REAL8TimeSeries **signalTD, const COMPLEX16FrequencySeries *signalFD, const REAL8 totalMass, const REAL8 deltaT, const REAL8 f_min, const REAL8 f_max, const REAL8 f_min_wide, const REAL8 f_max_wide);
static size_t find_instant_freq(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc, const REAL8 target, const size_t start);
static size_t find_peak_amp(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc);
static int apply_phase_shift(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc, const REAL8 shift);
static int apply_inclination(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc, const REAL8 inclination);

static REAL8 PlanckTaper(const REAL8 t, const REAL8 t1, const REAL8 t2);

/**
 * @addtogroup LALSimIMRPhenom_c
 * @{
 * @name Routines for IMR Phenomenological Model "C"
 * @{
 */

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomC in the frequency domain.
 *
 * Reference: http://arxiv.org/pdf/1005.3306v3.pdf
 * - Waveform: Eq.(5.3)-(5.13)
 * - Coefficients: Eq.(5.14) and Table II
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomCGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    const REAL8 phi0,                  /**< orbital phase at peak (rad) */
    const REAL8 deltaF,                /**< sampling interval (Hz) */
    const REAL8 m1_SI,                 /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,                 /**< mass of companion 2 (kg) */
    const REAL8 chi,                   /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,                 /**< starting GW frequency (Hz) */
    const REAL8 f_max,                 /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance,              /**< distance of source (m) */
    const LALSimInspiralTestGRParam *extraParams /**< linked list containing the extra testing GR parameters */
) {
  BBHPhenomCParams *params;
  int status;
  REAL8 f_max_prime;

  /* external: SI; internal: solar masses */
  const REAL8 m1 = m1_SI / LAL_MSUN_SI;
  const REAL8 m2 = m2_SI / LAL_MSUN_SI;

  /* check inputs for sanity */
  if (*htilde) XLAL_ERROR(XLAL_EFAULT);
  if (deltaF <= 0) XLAL_ERROR(XLAL_EDOM);
  if (m1 <= 0) XLAL_ERROR(XLAL_EDOM);
  if (m2 <= 0) XLAL_ERROR(XLAL_EDOM);
  if (fabs(chi) > 1) XLAL_ERROR(XLAL_EDOM);
  if (f_min <= 0) XLAL_ERROR(XLAL_EDOM);
  if (f_max < 0) XLAL_ERROR(XLAL_EDOM);
  if (distance <= 0) XLAL_ERROR(XLAL_EDOM);

  /* If spins are above 0.9 or below -0.9, throw an error */
  if (chi > 0.9 || chi < -0.9)
      XLAL_ERROR(XLAL_EDOM, "Spins outside the range [-0.9,0.9] are not supported\n");

  /* If mass ratio is above 4 and below 20, give a warning, and if it is above
   * 20, throw an error */
  REAL8 q = (m1 > m2) ? (m1 / m2) : (m2 / m1);

  if (q > 20.0)
      XLAL_ERROR(XLAL_EDOM, "Mass ratio is way outside the calibration range. m1/m2 should be <= 20.\n");
  else if (q > 4.0)
      XLAL_PRINT_WARNING("Warning: The model is only calibrated for m1/m2 <= 4.\n");

  /* phenomenological parameters*/
  params = ComputeIMRPhenomCParams(m1, m2, chi, extraParams);
  if (!params) XLAL_ERROR(XLAL_EFUNC);
  if (params->fCut <= f_min)
      XLAL_ERROR(XLAL_EDOM, "(fCut = 0.15M) <= f_min\n");

  /* default f_max to params->fCut */
  f_max_prime = f_max ? f_max : params->fCut;
  f_max_prime = (f_max_prime > params->fCut) ? params->fCut : f_max_prime;
  if (f_max_prime <= f_min)
      XLAL_ERROR(XLAL_EDOM, "f_max <= f_min\n");

  status = IMRPhenomCGenerateFD(htilde, phi0, deltaF, m1, m2, f_min, f_max_prime, distance, params);

  if (f_max_prime < f_max) {
    // The user has requested a higher f_max than Mf=params->fCut.
    // Resize the frequency series to fill with zeros to fill with zeros beyond the cutoff frequency.
        size_t n_full = NextPow2_PC(f_max / deltaF) + 1; // we actually want to have the length be a power of 2 + 1 power of 2 + 1
    *htilde = XLALResizeCOMPLEX16FrequencySeries(*htilde, 0, n_full);
  }

  LALFree(params);
  return status;
}

/**
 * Convenience function to quickly find the default final
 * frequency.
 */
double XLALSimIMRPhenomCGetFinalFreq(
    const REAL8 m1,
    const REAL8 m2,
    const REAL8 chi
) {
    BBHPhenomCParams *phenomParams;
    const LALSimInspiralTestGRParam *extraParams = NULL;
    phenomParams = ComputeIMRPhenomCParams(m1, m2, chi, extraParams);
    return phenomParams->fCut;
}

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomC in the time domain.
 * (Note that this approximant was constructed as a smooth function in
 * the frequency domain, so there might be spurious effects after
 * transforming into the time domain. One example are small amplitude
 * oscillations just before merger.)
 *
 * Reference: http://arxiv.org/pdf/1005.3306v3.pdf
 *   - Waveform: Eq.(5.3)-(5.13)
 *   - Coefficients: Eq.(5.14) and Table II
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomCGenerateTD(
    REAL8TimeSeries **hplus,  /**< +-polarization waveform */
    REAL8TimeSeries **hcross, /**< x-polarization waveform */
    const REAL8 phiPeak,      /**< orbital phase at peak (rad) */
    const REAL8 deltaT,       /**< sampling interval (s) */
    const REAL8 m1_SI,        /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,        /**< mass of companion 2 (kg) */
    const REAL8 chi,          /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,        /**< starting GW frequency (Hz) */
    const REAL8 f_max,        /**< end GW frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance,     /**< distance of source (m) */
    const REAL8 inclination,   /**< inclination of source (rad) */
    const LALSimInspiralTestGRParam *extraParams /**< linked list containing the extra testing GR parameters */ 
) {
	BBHPhenomCParams *params;
	size_t cut_ind, peak_ind, ind_t0;
	REAL8 peak_phase;  /* measured, not intended */
	REAL8 f_max_prime;

	/* external: SI; internal: solar masses */
	const REAL8 m1 = m1_SI / LAL_MSUN_SI;
	const REAL8 m2 = m2_SI / LAL_MSUN_SI;
	const REAL8 fISCO = 0.022 / ((m1 + m2) * LAL_MTSUN_SI);

	/* check inputs for sanity */
	if (*hplus) XLAL_ERROR(XLAL_EFAULT);
	if (*hcross) XLAL_ERROR(XLAL_EFAULT);
	if (deltaT <= 0) XLAL_ERROR(XLAL_EDOM);
	if (m1 < 0) XLAL_ERROR(XLAL_EDOM);
	if (m2 < 0) XLAL_ERROR(XLAL_EDOM);
	if (fabs(chi) > 1) XLAL_ERROR(XLAL_EDOM);
	if (f_min <= 0) XLAL_ERROR(XLAL_EDOM);
	if (f_max < 0) XLAL_ERROR(XLAL_EDOM);
	if (distance <= 0) XLAL_ERROR(XLAL_EDOM);

	/* If spins are above 0.9 or below -0.9, throw an error */
	if (chi > 0.9 || chi < -0.9)
		XLAL_ERROR(XLAL_EDOM, "Spins outside the range [-0.9,0.9] are not supported\n");

	/* If mass ratio is above 4 and below 20, give a warning, and if it is above
	 * 20, throw an error */
	REAL8 q = (m1 > m2) ? (m1 / m2) : (m2 / m1);

	if (q > 20.0)
		XLAL_ERROR(XLAL_EDOM, "Mass ratio is way outside the calibration range. m1/m2 should be <= 20.\n");
	else if (q > 4.0)
		XLAL_PRINT_WARNING("Warning: The model is only calibrated for m1/m2 <= 4.\n");

	/* phenomenological parameters*/
	params = ComputeIMRPhenomCParams(m1, m2, chi, extraParams);
	if (!params) XLAL_ERROR(XLAL_EFUNC);
	if (params->fCut <= f_min)
		XLAL_ERROR(XLAL_EDOM, "(fCut = 0.15M) <= f_min\n");

	/* default f_max to params->fCut */
	f_max_prime = f_max ? f_max : params->fCut;
	f_max_prime = (f_max_prime > params->fCut) ? params->fCut : f_max_prime;
	if (f_max_prime <= f_min)
		XLAL_ERROR(XLAL_EDOM, "f_max <= f_min\n");

	/* generate plus */

	IMRPhenomCGenerateTD(hplus, 0., &ind_t0, deltaT, m1, m2, f_min, f_max_prime, distance, params);
	if (!(*hplus)) {
		XLALFree(params);
		XLAL_ERROR(XLAL_EFUNC);
	}

	/* generate hcross, which is hplus w/ GW phase shifted by -pi/2
	 * <==> orb. phase shifted by -pi/4 */
	IMRPhenomCGenerateTD(hcross, -LAL_PI_4, &ind_t0, deltaT, m1, m2, f_min, f_max_prime, distance, params);
	if (!(*hcross)) {
		XLALDestroyREAL8TimeSeries(*hplus);
		*hplus = NULL;
		XLAL_ERROR(XLAL_EFUNC);
	}

	/* clip the parts below f_min */
	//const size_t start_ind = ((*hplus)->data->length + EstimateIMRLength(m1, m2, f_max_prime, deltaT)) > EstimateIMRLength(m1, m2, f_min, deltaT) ? ((*hplus)->data->length + EstimateIMRLength(m1, m2, f_max_prime, deltaT) - EstimateIMRLength(m1, m2, f_min, deltaT)) : 0;

	//const size_t start_ind = ((*hplus)->data->length
	//	      - EstimateIMRLength(m1, m2, 0.95 * f_min + 0.05 * f_max_prime, deltaT));


	peak_ind = find_peak_amp(*hplus, *hcross);

	cut_ind =find_instant_freq(*hplus, *hcross, f_min < fISCO/2. ? f_min : fISCO/2., peak_ind);
	*hplus = XLALResizeREAL8TimeSeries(*hplus, cut_ind, (*hplus)->data->length - cut_ind);
	*hcross = XLALResizeREAL8TimeSeries(*hcross, cut_ind, (*hcross)->data->length - cut_ind);

	if (!(*hplus) || !(*hcross))
		XLAL_ERROR(XLAL_EFUNC);

	/* set phase and time at peak */
	peak_ind = find_peak_amp(*hplus, *hcross);
	peak_phase = atan2((*hcross)->data->data[peak_ind], (*hplus)->data->data[peak_ind]);
	// NB: factor of 2 b/c phiPeak is *orbital* phase, and we're shifting GW phase
	apply_phase_shift(*hplus, *hcross, 2.*phiPeak - peak_phase);
	XLALGPSSetREAL8(&((*hplus)->epoch), -(peak_ind * deltaT));
	XLALGPSSetREAL8(&((*hcross)->epoch), -(peak_ind * deltaT));

	/* apply inclination */
	XLALFree(params);
	return apply_inclination(*hplus, *hcross, inclination);
}

/** @} */
/** @} */


/* *********************************************************************************/
/* The following private function generates IMRPhenomC frequency-domain waveforms  */
/* given coefficients */
/* *********************************************************************************/

static int IMRPhenomCGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    const REAL8 phi0,                  /**< phase at peak */
    const REAL8 deltaF,                /**< frequency resolution */
    const REAL8 m1,                    /**< mass of companion 1 [solar masses] */
    const REAL8 m2,                    /**< mass of companion 2 [solar masses] */
    //const REAL8 chi,                   /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,                 /**< start frequency */
    const REAL8 f_max,                 /**< end frequency */
    const REAL8 distance,              /**< distance to source (m) */
    const BBHPhenomCParams *params      /**< from ComputeIMRPhenomCParams */
) {
  LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; // = {0, 0}

  int errcode = XLAL_SUCCESS;
  /*
   We can't call XLAL_ERROR() directly with OpenMP on.
   Keep track of return codes for each thread and in addition use flush to get out of
   the parallel for loop as soon as possible if something went wrong in any thread.
  */

  const REAL8 M = m1 + m2;
  const REAL8 eta = m1 * m2 / (M * M);

  /* Memory to temporarily store components of amplitude and phase */
  /*
  REAL8 phSPA, phPM, phRD, aPM, aRD;
  REAL8 wPlusf1, wPlusf2, wMinusf1, wMinusf2, wPlusf0, wMinusf0;
  */

  /* compute the amplitude pre-factor */
  REAL8 amp0 = 2. * sqrt(5. / (64.*LAL_PI)) * M * LAL_MRSUN_SI * M * LAL_MTSUN_SI / distance;

  /* allocate htilde */
  size_t n = NextPow2_PC(f_max / deltaF) + 1;
  /* coalesce at t=0 */
  XLALGPSAdd(&ligotimegps_zero, -1. / deltaF); // shift by overall length in time
  *htilde = XLALCreateCOMPLEX16FrequencySeries("htilde: FD waveform", &ligotimegps_zero, 0.0,
      deltaF, &lalStrainUnit, n);
  memset((*htilde)->data->data, 0, n * sizeof(COMPLEX16));
  XLALUnitMultiply(&((*htilde)->sampleUnits), &((*htilde)->sampleUnits), &lalSecondUnit);
  if (!(*htilde)) XLAL_ERROR(XLAL_EFUNC);

  size_t ind_min = (size_t) (f_min / deltaF);
  size_t ind_max = (size_t) (f_max / deltaF);

  /* Set up spline for phase */
  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  size_t L =  ind_max - ind_min;
  gsl_spline *phiI = gsl_spline_alloc(gsl_interp_cspline, L);
  REAL8 *freqs = XLALMalloc(L*sizeof(REAL8));
  REAL8 *phis = XLALMalloc(L*sizeof(REAL8));

  /* now generate the waveform */
  #pragma omp parallel for
  for (size_t i = ind_min; i < ind_max; i++)
  {

    REAL8 phPhenomC = 0.0;
    REAL8 aPhenomC = 0.0;
    REAL8 f = i * deltaF;

    int per_thread_errcode;
    #pragma omp flush(errcode)
    if (errcode != XLAL_SUCCESS)
      goto skip;

    per_thread_errcode = IMRPhenomCGenerateAmpPhase( &aPhenomC, &phPhenomC, f, eta, params );
    if (per_thread_errcode != XLAL_SUCCESS) {
      errcode = per_thread_errcode;
      #pragma omp flush(errcode)
    }

    phPhenomC -= 2.*phi0; // factor of 2 b/c phi0 is orbital phase

    freqs[i-ind_min] = f;
    phis[i-ind_min] = -phPhenomC; // PhenomP uses cexp(-I*phPhenomC); want to use same phase adjustment code, so we will flip the sign of the phase

    /* generate the waveform */
    ((*htilde)->data->data)[i] = amp0 * aPhenomC * cos(phPhenomC);
    ((*htilde)->data->data)[i] += -I * amp0 * aPhenomC * sin(phPhenomC);

  skip: /* this statement intentionally left blank */;

  }

  if( errcode != XLAL_SUCCESS )
    XLAL_ERROR(errcode);

  /* Correct phasing so we coalesce at t=0 (with the definition of the epoch=-1/deltaF above) */
  gsl_spline_init(phiI, freqs, phis, L);

  REAL8 f_final = params->fRingDown;
  XLAL_PRINT_INFO("f_ringdown = %g\n", f_final);

  // Prevent gsl interpolation errors
  if (f_final > freqs[L-1])
    f_final = freqs[L-1];
  if (f_final < freqs[0])
    XLAL_ERROR(XLAL_EDOM, "f_ringdown <= f_min\n");

  /* Time correction is t(f_final) = 1/(2pi) dphi/df (f_final) */
  REAL8 t_corr = gsl_spline_eval_deriv(phiI, f_final, acc) / (2*LAL_PI);
  XLAL_PRINT_INFO("t_corr = %g\n", t_corr);
  /* Now correct phase */
  for (size_t i = ind_min; i < ind_max; i++) {
    REAL8 f = i * deltaF;
    ((*htilde)->data->data)[i] *= cexp(-2*LAL_PI * I * f * t_corr);
  }

  gsl_spline_free(phiI);
  gsl_interp_accel_free(acc);
  XLALFree(freqs);
  XLALFree(phis);

  return XLAL_SUCCESS;
}

static int IMRPhenomCGenerateFDForTD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    const REAL8 t0,					   /**< time of coalescence */
    const REAL8 phi0,                  /**< phase at peak */
    const REAL8 deltaF,                /**< frequency resolution */
    const REAL8 m1,                    /**< mass of companion 1 [solar masses] */
    const REAL8 m2,                    /**< mass of companion 2 [solar masses] */
    //const REAL8 chi,                   /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,                 /**< start frequency */
    const REAL8 f_max,                 /**< end frequency */
    const REAL8 distance,              /**< distance to source (m) */
    const BBHPhenomCParams *params,    /**< from ComputeIMRPhenomCParams */
    const size_t nf                    /**< Length of frequency vector required */
) {
  static LIGOTimeGPS ligotimegps_zero = {0, 0};
  size_t i;
  INT4 errcode;

  const REAL8 M = m1 + m2;
  const REAL8 eta = m1 * m2 / (M * M);

  /* Memory to temporarily store components of amplitude and phase */
  /*
  REAL8 phSPA, phPM, phRD, aPM, aRD;
  REAL8 wPlusf1, wPlusf2, wMinusf1, wMinusf2, wPlusf0, wMinusf0;
  */
  REAL8 phPhenomC = 0.0;
  REAL8 aPhenomC = 0.0;

  /* compute the amplitude pre-factor */
  REAL8 amp0 = 2. * sqrt(5. / (64.*LAL_PI)) * M * LAL_MRSUN_SI * M * LAL_MTSUN_SI / distance;

  /* allocate htilde */
  size_t n = NextPow2_PC(f_max / deltaF) + 1;
  if ( n > nf )
    XLAL_ERROR(XLAL_EDOM, "The required length passed as input wont fit the FD waveform\n");
  else
    n = nf;
  *htilde = XLALCreateCOMPLEX16FrequencySeries("htilde: FD waveform", &ligotimegps_zero, 0.0,
      deltaF, &lalStrainUnit, n);
  memset((*htilde)->data->data, 0, n * sizeof(COMPLEX16));
  XLALUnitMultiply(&((*htilde)->sampleUnits), &((*htilde)->sampleUnits), &lalSecondUnit);
  if (!(*htilde)) XLAL_ERROR(XLAL_EFUNC);

  /* now generate the waveform */
  size_t ind_max = (size_t) (f_max / deltaF);
  for (i = (size_t) (f_min / deltaF); i < ind_max; i++)
  {
    REAL8 f = i * deltaF;

    errcode = IMRPhenomCGenerateAmpPhase( &aPhenomC, &phPhenomC, f, eta, params );
    if( errcode != XLAL_SUCCESS )
      XLAL_ERROR(XLAL_EFUNC);
    phPhenomC -= 2.*phi0; // factor of 2 b/c phi0 is orbital phase
    phPhenomC += 2.*LAL_PI*f*t0; //shifting the coalescence to t=t0

    /* generate the waveform */
    ((*htilde)->data->data)[i] = amp0 * aPhenomC * cos(phPhenomC);
    ((*htilde)->data->data)[i] += -I * amp0 * aPhenomC * sin(phPhenomC);
  }

  return XLAL_SUCCESS;
}

/*
 * Private function to generate time-domain waveforms given coefficients
 */
static int IMRPhenomCGenerateTD(
	REAL8TimeSeries **h,
    const REAL8 phi0,
    size_t *ind_t0,
	const REAL8 deltaT,
	const REAL8 m1,
	const REAL8 m2,
	//const REAL8 chi,
	const REAL8 f_min,
	const REAL8 f_max,
	const REAL8 distance,
	const BBHPhenomCParams *params
) {
	REAL8 deltaF;
	COMPLEX16FrequencySeries *htilde=NULL;
	/* We will generate the waveform from a frequency which is lower than the
	 * f_min chosen. Also the cutoff frequency is higher than the f_max. We
	 * will later apply a window function, and truncate the time-domain waveform
	 * below an instantaneous frequency f_min. */
	REAL8 f_min_wide = EstimateSafeFMinForTD(m1, m2, f_min, deltaT);
	const REAL8 f_max_wide = params->fCut; //0.5 / deltaT;
	if (EstimateSafeFMaxForTD(f_max, deltaT) > f_max_wide)
		XLAL_PRINT_WARNING("Warning: sampling rate (%" LAL_REAL8_FORMAT " Hz) too low for expected spectral content (%" LAL_REAL8_FORMAT " Hz) \n", deltaT, EstimateSafeFMaxForTD(f_max, deltaT));

    const size_t nt = NextPow2_PC(EstimateIMRLength(m1, m2, f_min_wide, deltaT));
    const size_t ne = EstimateIMRLength(m1, m2, params->fRingDown, deltaT);
    *ind_t0 = ((nt - EstimateIMRLength(m1, m2, f_min_wide, deltaT)) > (4 * ne)) ? (nt -(2* ne)) : (nt - (1 * ne));
	deltaF = 1. / (deltaT * (REAL8) nt);


	/* Get the length in time for the entire IMR waveform, so the coalesence
	 * can be positioned (in time) accordingly, to avoid wrap-around */
	REAL8 t0 = deltaT * ((REAL8) *ind_t0);


	/* generate in frequency domain */
	if (IMRPhenomCGenerateFDForTD(&htilde, t0, phi0, deltaF, m1, m2, //chi,
	    f_min_wide, f_max_wide, distance, params, nt/2 + 1)) XLAL_ERROR(XLAL_EFUNC);


	/* convert to time domain */
	FDToTD(h, htilde, m1 + m2, deltaT, f_min, f_max, f_min_wide, f_max_wide);

	XLALDestroyCOMPLEX16FrequencySeries(htilde);
	if (!*h) XLAL_ERROR(XLAL_EFUNC);

	return XLAL_SUCCESS;
}

/* *********************************************************************************/
/* The following functions are borrowed as-it-is from LALSimIMRPhenom.c, and used  */
/* to generate the time-domain version of the frequency domain PhenomC approximant */
/* *********************************************************************************/

/*
 * Return tau0, the Newtonian chirp length estimate.
 */
static REAL8 ComputeTau0(const REAL8 m1, const REAL8 m2, const REAL8 f_min) {
	const REAL8 totalMass = m1 + m2;
	const REAL8 eta = m1 * m2 / (totalMass * totalMass);
	return 5. * totalMass * LAL_MTSUN_SI / (256. * eta * pow(LAL_PI * totalMass * LAL_MTSUN_SI * f_min, 8./3.));
}

/*
 * Estimate the length of a TD vector that can hold the waveform as the Newtonian
 * chirp time tau0 plus 1000 M.
 */
static size_t EstimateIMRLength(const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 deltaT) {
	return (size_t) floor((ComputeTau0(m1, m2, f_min) + 1000 * (m1 + m2) * LAL_MTSUN_SI) / deltaT);
}

/*
 * Find a lower value for f_min (using the definition of Newtonian chirp
 * time) such that the waveform has a minimum length of tau0. This is
 * necessary to avoid FFT artifacts.
 */
static REAL8 EstimateSafeFMinForTD(const REAL8 m1, const REAL8 m2, const REAL8 f_min, const REAL8 deltaT) {
	const REAL8 totalMass = m1 + m2;
	const REAL8 eta = m1 * m2 / (totalMass * totalMass);
	const REAL8 fISCO = 0.022 / (totalMass * LAL_MTSUN_SI);
	REAL8 tau0 = deltaT * NextPow2_PC(1.5 * EstimateIMRLength(m1, m2, f_min, deltaT));
	REAL8 temp_f_min = pow((tau0 * 256. * eta * pow(totalMass * LAL_MTSUN_SI, 5./3.) / 5.), -3./8.) / LAL_PI;
	if (temp_f_min > f_min) temp_f_min = f_min;
	if (temp_f_min < 0.5) temp_f_min = 0.5;
	if (temp_f_min > fISCO/2.) temp_f_min = fISCO/2.;
	return temp_f_min;
}

/*
 * Find a higher value of f_max so that we can safely apply a window later.
 */
static REAL8 EstimateSafeFMaxForTD(const REAL8 f_max, const REAL8 deltaT) {
	REAL8 temp_f_max = 1.025 * f_max;

	/* make sure that these frequencies are not too out of range */
	if (temp_f_max > 2. / deltaT - 100.) temp_f_max = 2. / deltaT - 100.;
	return temp_f_max;
}


/*
 * Window and IFFT a FD waveform to TD, then window in TD.
 * Requires that the FD waveform be generated outside of f_min and f_max.
 * FD waveform is modified.
 */
static int FDToTD(REAL8TimeSeries **signalTD, const COMPLEX16FrequencySeries *signalFD, const REAL8 totalMass, const REAL8 deltaT, const REAL8 f_min, const REAL8 f_max, const REAL8 f_min_wide, const REAL8 f_max_wide) {
	const LIGOTimeGPS gpstime_zero = {0, 0};
	const size_t nf = signalFD->data->length;
	const size_t nt = 2 * (nf - 1);

        /* Calculate the expected lengths of the FD and TD vectors from the
         * deltaF and deltaT passed in to this function */
        //const size_t exp_nt = (size_t) 1. / (signalFD->deltaF * deltaT);
        //const size_t exp_nf = (exp_nt / 2) + 1;

	const REAL8 windowLength = 20. * totalMass * LAL_MTSUN_SI / deltaT;
	const REAL8 winFLo = 0.2*f_min_wide + 0.8*f_min;
	/* frequency used for tapering, slightly less than f_min to minimize FFT artifacts
	 * equivalent to winFLo = f_min_wide + 0.8*(f_min - f_min_wide) */
	REAL8 winFHi = f_max_wide;
	COMPLEX16 *FDdata = signalFD->data->data;
	REAL8FFTPlan *revPlan;
	REAL8 *TDdata;
	size_t k;

	/* check inputs */
	if (f_min_wide >= f_min) XLAL_ERROR(XLAL_EDOM);

	/* apply the softening window function */
	if (winFHi > 0.5 / deltaT) winFHi = 0.5 / deltaT;
	for (k = nf;k--;) {
		const REAL8 f = k / (deltaT * nt);
		REAL8 softWin = PlanckTaper(f, f_min_wide, winFLo) * (1.0 - PlanckTaper(f, f_max, winFHi));
		FDdata[k] *= softWin;
	}


	/* allocate output */
	*signalTD = XLALCreateREAL8TimeSeries("h", &gpstime_zero, 0.0, deltaT, &lalStrainUnit, nt);


	/* Inverse Fourier transform */
	revPlan = XLALCreateReverseREAL8FFTPlan(nt, 1);
	if (!revPlan) {
		XLALDestroyREAL8TimeSeries(*signalTD);
		*signalTD = NULL;
		XLAL_ERROR(XLAL_EFUNC);
	}
	XLALREAL8FreqTimeFFT(*signalTD, signalFD, revPlan);
	XLALDestroyREAL8FFTPlan(revPlan);
	if (!(*signalTD)) XLAL_ERROR(XLAL_EFUNC);


	/* apply a linearly decreasing window at the end
	 * of the waveform in order to avoid edge effects. */
	if (windowLength > (*signalTD)->data->length) XLAL_ERROR(XLAL_ERANGE);
	TDdata = (*signalTD)->data->data;
	for (k = windowLength; k--;)
		TDdata[nt-k-1] *= k / windowLength;

	return XLAL_SUCCESS;
}

/* return the index before the instantaneous frequency rises past target */
static size_t find_instant_freq(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc, const REAL8 target, const size_t start) {
	/* const size_t n = hp->data->length - 1; */
	/* size_t k = (start + 1) > 0 ? (start + 1) : 0; */
	size_t k = start;
	size_t target_ind = 0;

	/* Use second order differencing to find the instantaneous frequency as
	 * h = A e^(2 pi i f t) ==> f = d/dt(h) / (2*pi*h) */
	for (; k > 0 /*< n*/; k--) {
		const REAL8 hpDot = (hp->data->data[k+1] - hp->data->data[k-1]) / (2 * hp->deltaT);
		const REAL8 hcDot = (hc->data->data[k+1] - hc->data->data[k-1]) / (2 * hc->deltaT);
		REAL8 f = hcDot * hp->data->data[k] - hpDot * hc->data->data[k];
		f /= LAL_TWOPI;
		f /= hp->data->data[k] * hp->data->data[k] + hc->data->data[k] * hc->data->data[k];
		if (f <= target){
			target_ind = k;
			break;
			//return k;// - 1;
		}
	}


	return target_ind;
	XLAL_ERROR(XLAL_EDOM);
}

/* Return the index of the sample at with the peak amplitude */
static size_t find_peak_amp(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc) {
	const REAL8 *hpdata = hp->data->data;
	const REAL8 *hcdata = hc->data->data;
	size_t k = hp->data->length;
	size_t peak_ind = -1;
	REAL8 peak_amp_sq = 0.;

	for (;k--;) {
		const REAL8 amp_sq = hpdata[k] * hpdata[k] + hcdata[k] * hcdata[k];
		if (amp_sq > peak_amp_sq) {
			peak_ind = k;
			peak_amp_sq = amp_sq;
		}
	}
	return peak_ind;
}

static int apply_phase_shift(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc, const REAL8 shift) {
    REAL8 *hpdata = hp->data->data;
    REAL8 *hcdata = hc->data->data;
    size_t k = hp->data->length;
    const double cs = cos(shift);
    const double ss = sin(shift);

    for (;k--;) {
        const REAL8 temp_hpdata = hpdata[k] * cs - hcdata[k] * ss;
        hcdata[k] = hpdata[k] * ss + hcdata[k] * cs;
        hpdata[k] = temp_hpdata;
    }
    return 0;
}

static int apply_inclination(const REAL8TimeSeries *hp, const REAL8TimeSeries *hc, const REAL8 inclination) {
	REAL8 inclFacPlus, inclFacCross;
	REAL8 *hpdata = hp->data->data;
	REAL8 *hcdata = hc->data->data;
	size_t k = hp->data->length;

	inclFacCross = cos(inclination);
	inclFacPlus = 0.5 * (1. + inclFacCross * inclFacCross);
	for (;k--;) {
		hpdata[k] *= inclFacPlus;
		hcdata[k] *= inclFacCross;
	}

	return XLAL_SUCCESS;
}

static REAL8 PlanckTaper(const REAL8 t, const REAL8 t1, const REAL8 t2)
	{
	REAL8 taper;
	if (t <= t1) {
		taper = 0.;
	}
	else if (t >= t2) {
		taper = 1.;
	}
	else {
		taper = 1. / (exp((t2 - t1)/(t - t1) + (t2 - t1)/(t - t2)) + 1.);
	}

	return taper;
	}

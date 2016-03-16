/*
 *  Copyright (C) 2013,2014,2015 Michael Puerrer, Alejandro Bohe
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

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALAtomicDatatypes.h>
#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/LALSimInspiral.h>
#include <lal/Units.h>
#include <lal/XLALError.h>
#include <lal/SphericalHarmonics.h>
#include <lal/Sequence.h>
#include <lal/LALStdlib.h>
#include <lal/LALStddef.h>

#include "LALSimIMR.h"
/* This is ugly, but allows us to reuse internal IMRPhenomC and IMRPhenomD functions without making those functions XLAL */
#include "LALSimIMRPhenomC_internals.c"
#include "LALSimIMRPhenomD_internals.c"

#include "LALSimIMRPhenomP.h"

#ifndef _OPENMP
#define omp ignore
#endif

const double sqrt_6 = 2.44948974278317788;

/* ************************************* Implementation ****************************************/

/**
 * @addtogroup LALSimIMRPhenom_c
 * @{
 *
 * @name Routines for IMR Phenomenological Model "P"
 * @{
 *
 * @author Michael Puerrer, Alejandro Bohe
 *
 * @brief Functions for producing IMRPhenomP waveforms for precessing binaries,
 * as described in Hannam et al., arXiv:1308.3271 [gr-qc].
 *
 * @note Two versions of IMRPhenomP are available (selected by IMRPhenomP_version):
 *    * version 1 ("IMRPhenomP"): based on IMRPhenomC
 *      (outdated, not reviewed!)
 *    * version 2 ("IMRPhenomPv2"): based on IMRPhenomD
 *      (to be used, currently under review as of Dec 2015)
 *
 * Each IMRPhenomP version inherits its range of validity
 * over the parameter space from the respective aligned-spin waveform.
 *
 * @attention A time-domain implementation of IMRPhenomPv2 is available in XLALChooseTDWaveform(). 
 * This is based on a straight-forward inverse Fourier transformation via XLALSimInspiralTDfromFD(),
 * but it was not included in the IMRPhenomPv2 review. Use it at your own risk. 
 */

/**
 * Function to map LAL parameters
 * (masses, 6 spin components and Lhat at f_ref)
 * into IMRPhenomP intrinsic parameters
 * (chi1_l, chi2_l, chip, thetaJ, alpha0).
 *
 * All input masses and frequencies should be in SI units.
 *
 * See Fig. 1. in arxiv:1408.1810 for a diagram of the angles.
 */
int XLALSimIMRPhenomPCalculateModelParameters(
    REAL8 *chi1_l,                  /**< [out] Dimensionless aligned spin on companion 1 */
    REAL8 *chi2_l,                  /**< [out] Dimensionless aligned spin on companion 2 */
    REAL8 *chip,                    /**< [out] Effective spin in the orbital plane */
    REAL8 *thetaJ,                  /**< [out] Angle between J0 and line of sight (z-direction) */
    REAL8 *alpha0,                  /**< [out] Initial value of alpha angle (azimuthal precession angle) */
    const REAL8 m1_SI,              /**< Mass of companion 1 (kg) */
    const REAL8 m2_SI,              /**< Mass of companion 2 (kg) */
    const REAL8 f_ref,              /**< Reference GW frequency (Hz) */
    const REAL8 lnhatx,             /**< Initial value of LNhatx: orbital angular momentum unit vector */
    const REAL8 lnhaty,             /**< Initial value of LNhaty */
    const REAL8 lnhatz,             /**< Initial value of LNhatz */
    const REAL8 s1x,                /**< Initial value of s1x: dimensionless spin of BH 1 */
    const REAL8 s1y,                /**< Initial value of s1y: dimensionless spin of BH 1 */
    const REAL8 s1z,                /**< Initial value of s1z: dimensionless spin of BH 1 */
    const REAL8 s2x,                /**< Initial value of s2x: dimensionless spin of BH 2 */
    const REAL8 s2y,                /**< Initial value of s2y: dimensionless spin of BH 2 */
    const REAL8 s2z,                /**< Initial value of s2z: dimensionless spin of BH 2 */
    IMRPhenomP_version_type IMRPhenomP_version /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
)
{
  // Note that the angle phiJ defined below and alpha0 are degenerate. Therefore we do not output phiJ.

  /* Check arguments for sanity */
  XLAL_CHECK(chi1_l != NULL, XLAL_EFAULT);
  XLAL_CHECK(chi2_l != NULL, XLAL_EFAULT);
  XLAL_CHECK(chip != NULL, XLAL_EFAULT);
  XLAL_CHECK(thetaJ != NULL, XLAL_EFAULT);
  XLAL_CHECK(alpha0 != NULL, XLAL_EFAULT);

  XLAL_CHECK(f_ref > 0, XLAL_EDOM, "Reference frequency must be positive.\n");
  XLAL_CHECK(m1_SI > 0, XLAL_EDOM, "m1 must be positive.\n");
  XLAL_CHECK(m2_SI > 0, XLAL_EDOM, "m2 must be positive.\n");
  XLAL_CHECK(fabs(lnhatx*lnhatx + lnhaty*lnhaty + lnhatz*lnhatz - 1) < 1e-10, XLAL_EDOM, "Lnhat must be a unit vector.\n");
  XLAL_CHECK(fabs(s1x*s1x + s1y*s1y + s1z*s1z) <= 1.0, XLAL_EDOM, "|S1/m1^2| must be <= 1.\n");
  XLAL_CHECK(fabs(s2x*s2x + s2y*s2y + s2z*s2z) <= 1.0, XLAL_EDOM, "|S2/m2^2| must be <= 1.\n");

  const REAL8 m1 = m1_SI / LAL_MSUN_SI;   /* Masses in solar masses */
  const REAL8 m2 = m2_SI / LAL_MSUN_SI;
  const REAL8 M = m1+m2;
  const REAL8 m1_2 = m1*m1;
  const REAL8 m2_2 = m2*m2;
  const REAL8 eta = m1 * m2 / (M*M);    /* Symmetric mass-ratio */

  /* Aligned spins */
  *chi1_l = lnhatx*s1x + lnhaty*s1y + lnhatz*s1z; /* Dimensionless aligned spin on BH 1 */
  *chi2_l = lnhatx*s2x + lnhaty*s2y + lnhatz*s2z; /* Dimensionless aligned spin on BH 2 */

  /* Spin components orthogonal to lnhat */
  const REAL8 S1_perp_x = (s1x - *chi1_l*lnhatx) * m1_2;
  const REAL8 S1_perp_y = (s1y - *chi1_l*lnhaty) * m1_2;
  const REAL8 S1_perp_z = (s1z - *chi1_l*lnhatz) * m1_2;
  const REAL8 S2_perp_x = (s2x - *chi2_l*lnhatx) * m2_2;
  const REAL8 S2_perp_y = (s2y - *chi2_l*lnhaty) * m2_2;
  const REAL8 S2_perp_z = (s2z - *chi2_l*lnhatz) * m2_2;
  const REAL8 S1_perp = sqrt(S1_perp_x*S1_perp_x + S1_perp_y*S1_perp_y + S1_perp_z*S1_perp_z);
  const REAL8 S2_perp = sqrt(S2_perp_x*S2_perp_x + S2_perp_y*S2_perp_y + S2_perp_z*S2_perp_z);

  const REAL8 A1 = 2 + (3*m2) / (2*m1);
  const REAL8 A2 = 2 + (3*m1) / (2*m2);
  const REAL8 ASp1 = A1*S1_perp;
  const REAL8 ASp2 = A2*S2_perp;
  const REAL8 num = (ASp2 > ASp1) ? ASp2 : ASp1;
  const REAL8 den = (m2 > m1) ? A2*m2_2 : A1*m1_2;
  *chip = num / den; /*  chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of larger BH (See Eqn. 32 in technical document) */

  /* Compute L, J0 and orientation angles */
  const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
  const REAL8 piM = LAL_PI * m_sec;
  const REAL8 v_ref = cbrt(piM * f_ref);

  REAL8 L0 = 0.0;
  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      L0 = M*M * L2PNR_v1(v_ref, eta); /* Use 2PN approximation for L. */
      break;
    case IMRPhenomPv2_V:
      L0 = M*M * L2PNR(v_ref, eta);   /* Use 2PN approximation for L. */
      break;
    default:
      XLAL_ERROR( XLAL_EINVAL, "Unknown IMRPhenomP version!\nAt present only v1 and v2 are available." );
      break;
    }

  const REAL8 Jx0 = L0 * lnhatx + m1_2*s1x + m2_2*s2x;
  const REAL8 Jy0 = L0 * lnhaty + m1_2*s1y + m2_2*s2y;
  const REAL8 Jz0 = L0 * lnhatz + m1_2*s1z + m2_2*s2z;
  const REAL8 J0 = sqrt(Jx0*Jx0 + Jy0*Jy0 + Jz0*Jz0);

  /* Compute thetaJ, the angle between J0 and line of sight (z-direction) */
  if (J0 < 1e-10) {
    XLAL_PRINT_WARNING("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n");
    *thetaJ = 0;
  } else {
    *thetaJ = acos(Jz0 / J0);
  }

  REAL8 phiJ; // We only use this angle internally since it is degenerate with alpha0.
  if (Jx0 < DBL_MIN && Jy0 < DBL_MIN)
    phiJ = 0;
  else
    phiJ = atan2(Jy0, Jx0); /* Angle of J0 in the plane of the sky */
    /* Note: Compared to the similar code in SpinTaylorF2 we have defined phiJ as the angle between the positive
    (rather than the negative) x-axis and the projection of J0, since this is a more natural definition of the angle.
    We have also renamed the angle from psiJ to phiJ. */

  /* Rotate Lnhat back to frame where J is along z and the line of sight in the Oxz plane with >0 projection in x, to figure out initial alpha */
  /* The rotation matrix is
    {
      {-cos(thetaJ)*cos(phiJ), -cos(thetaJ)*sin(phiJ), sin(thetaJ)},
      {sin(phiJ), -cos(phiJ), 0},
      {cos(phiJ)*sin(thetaJ), sin(thetaJ)*sin(phiJ),cos(thetaJ)}
    }
  */
  const REAL8 rotLx = -lnhatx*cos(*thetaJ)*cos(phiJ) - lnhaty*cos(*thetaJ)*sin(phiJ) + lnhatz*sin(*thetaJ);
  const REAL8 rotLy = lnhatx*sin(phiJ) - lnhaty*cos(phiJ);
  if (rotLx == 0.0 && rotLy == 0.0)
    *alpha0 = 0.0;
  else
    *alpha0 = atan2(rotLy, rotLx);

  return XLAL_SUCCESS;
}

/**
 * Driver routine to compute the precessing inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomP in the frequency domain.
 *
 * Reference:
 * - Hannam et al., arXiv:1308.3271 [gr-qc]
 *
 * \ref XLALSimIMRPhenomPCalculateModelParameters should be called first
 * to map LAL parameters into IMRPhenomP intrinsic parameters
 * (chi1_l, chi2_l, chip, thetaJ, alpha0).
 *
 * This function can be used for equally-spaced frequency series.
 * For unequal spacing, use \ref XLALSimIMRPhenomPFrequencySequence instead.
 */
int XLALSimIMRPhenomP(
  COMPLEX16FrequencySeries **hptilde,         /**< [out] Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,         /**< [out] Frequency-domain waveform hx */
  const REAL8 chi1_l,                         /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l,                         /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                           /**< Effective spin in the orbital plane */
  const REAL8 thetaJ,                         /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 m1_SI,                          /**< Mass of companion 1 (kg) */
  const REAL8 m2_SI,                          /**< Mass of companion 2 (kg) */
  const REAL8 distance,                       /**< Distance of source (m) */
  const REAL8 alpha0,                         /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                           /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 deltaF,                         /**< Sampling frequency (Hz) */
  const REAL8 f_min,                          /**< Starting GW frequency (Hz) */
  const REAL8 f_max,                          /**< End frequency; 0 defaults to ringdown cutoff freq */
  const REAL8 f_ref,                          /**< Reference frequency */
  IMRPhenomP_version_type IMRPhenomP_version) /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
{
  // See Fig. 1. in arxiv:1408.1810 for diagram of the angles.
  // Note that the angles phiJ which is calculated internally in XLALSimIMRPhenomPCalculateModelParameters
  // and alpha0 are degenerate. Therefore phiJ is not passed to this function.

  // Use f_min, f_max, deltaF to compute freqs sequence
  // Instead of building a full sequency we only transfer the boundaries and let
  // the internal core function do the rest (and properly take care of corner cases).
  XLAL_CHECK (f_min > 0, XLAL_EDOM, "Minimum frequency must be positive.");
  XLAL_CHECK (f_max >= 0, XLAL_EDOM, "Maximum frequency must be non-negative.");
  XLAL_CHECK ( ( f_max == 0 ) || ( f_max > f_min ), XLAL_EDOM, "f_max <= f_min");
  REAL8Sequence *freqs = XLALCreateREAL8Sequence(2);
  XLAL_CHECK(freqs != NULL, XLAL_EFAULT);
  freqs->data[0] = f_min;
  freqs->data[1] = f_max;

  int retcode = PhenomPCore(hptilde, hctilde,
      chi1_l, chi2_l, chip, thetaJ, m1_SI, m2_SI, distance, alpha0, phic, f_ref, freqs, deltaF, IMRPhenomP_version);
  XLAL_CHECK(retcode == XLAL_SUCCESS, XLAL_EFUNC, "Failed to generate IMRPhenomP waveform.");
  XLALDestroyREAL8Sequence(freqs);
  return (retcode);
}

/**
 * Driver routine to compute the precessing inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomP in the frequency domain.
 *
 * Reference:
 * - Hannam et al., arXiv:1308.3271 [gr-qc]
 *
 * \ref XLALSimIMRPhenomPCalculateModelParameters should be called first
 * to map LAL parameters into IMRPhenomP intrinsic parameters
 * (chi1_l, chi2_l, chip, thetaJ, alpha0).
 *
 * This function can be used for user-specified,
 * potentially unequally-spaced frequency series.
 * For equal spacing with a given deltaF, use \ref XLALSimIMRPhenomP instead.
 */
int XLALSimIMRPhenomPFrequencySequence(
  COMPLEX16FrequencySeries **hptilde,         /**< [out] Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,         /**< [out] Frequency-domain waveform hx */
  const REAL8Sequence *freqs,                 /**< Frequency points at which to evaluate the waveform (Hz) */
  const REAL8 chi1_l,                         /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l,                         /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                           /**< Effective spin in the orbital plane */
  const REAL8 thetaJ,                         /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 m1_SI,                          /**< Mass of companion 1 (kg) */
  const REAL8 m2_SI,                          /**< Mass of companion 2 (kg) */
  const REAL8 distance,                       /**< Distance of source (m) */
  const REAL8 alpha0,                         /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                           /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 f_ref,                          /**< Reference frequency */
  IMRPhenomP_version_type IMRPhenomP_version) /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
{
  // See Fig. 1. in arxiv:1408.1810 for diagram of the angles.
  // Note that the angles phiJ which is calculated internally in XLALSimIMRPhenomPCalculateModelParameters
  // and alpha0 are degenerate. Therefore phiJ is not passed to this function.

  // Call the internal core function with deltaF = 0 to indicate that freqs is non-uniformly
  // spaced and we want the strain only at these frequencies
  int retcode = PhenomPCore(hptilde, hctilde,
      chi1_l, chi2_l, chip, thetaJ, m1_SI, m2_SI, distance, alpha0, phic, f_ref, freqs, 0, IMRPhenomP_version);
  XLAL_CHECK(retcode == XLAL_SUCCESS, XLAL_EFUNC, "Failed to generate IMRPhenomP waveform.");
  return(retcode);
}

/** @} */
/** @} */


/**
 * Internal core function to calculate
 * plus and cross polarizations of the PhenomP model
 * for a set of frequencies.
 * This can handle either user-specified frequency points
 * or create an equally-spaced frequency series.
 */
static int PhenomPCore(
  COMPLEX16FrequencySeries **hptilde,        /**< [out] Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,        /**< [out] Frequency-domain waveform hx */
  const REAL8 chi1_l_in,                     /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l_in,                     /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                          /**< Effective spin in the orbital plane */
  const REAL8 thetaJ,                        /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 m1_SI_in,                      /**< Mass of companion 1 (kg) */
  const REAL8 m2_SI_in,                      /**< Mass of companion 2 (kg) */
  const REAL8 distance,                      /**< Distance of source (m) */
  const REAL8 alpha0,                        /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                          /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 f_ref,                         /**< Reference frequency */
  const REAL8Sequence *freqs_in,             /**< Frequency points at which to evaluate the waveform (Hz) */
  double deltaF,                             /**< Sampling frequency (Hz).
   * If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
  IMRPhenomP_version_type IMRPhenomP_version /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
  )
{
  /* Check inputs for sanity */
  XLAL_CHECK(NULL != hptilde, XLAL_EFAULT);
  XLAL_CHECK(NULL != hctilde, XLAL_EFAULT);
  XLAL_CHECK(*hptilde == NULL, XLAL_EFAULT);
  XLAL_CHECK(*hctilde == NULL, XLAL_EFAULT);
  XLAL_CHECK(deltaF >= 0, XLAL_EDOM, "deltaF must be non-negative.\n");
  XLAL_CHECK(m1_SI_in > 0, XLAL_EDOM, "m1 must be positive.\n");
  XLAL_CHECK(m2_SI_in > 0, XLAL_EDOM, "m2 must be positive.\n");
  XLAL_CHECK(f_ref > 0, XLAL_EDOM, "Reference frequency must be non-negative.\n");
  XLAL_CHECK(distance > 0, XLAL_EDOM, "distance must be positive.\n");
  XLAL_CHECK(fabs(chi1_l_in) <= 1.0, XLAL_EDOM, "Aligned spin chi1_l=%g must be <= 1 in magnitude!\n", chi1_l_in);
  XLAL_CHECK(fabs(chi2_l_in) <= 1.0, XLAL_EDOM, "Aligned spin chi2_l=%g must be <= 1 in magnitude!\n", chi2_l_in);
  XLAL_CHECK(fabs(chip) <= 1.0, XLAL_EDOM, "In-plane spin chip =%g must be <= 1 in magnitude!\n", chip);

  // See Fig. 1. in arxiv:1408.1810 for diagram of the angles.
  // Note that the angles phiJ which is calculated internally in XLALSimIMRPhenomPCalculateModelParameters
  // and alpha0 are degenerate. Therefore phiJ is not passed to this function.
  /* Phenomenological parameters */
  IMRPhenomDAmplitudeCoefficients *pAmp = NULL;
  IMRPhenomDPhaseCoefficients *pPhi = NULL;
  BBHPhenomCParams *PCparams = NULL;
  PNPhasingSeries *pn = NULL;
  gsl_interp_accel *acc = NULL;
  gsl_spline *phiI = NULL;
  REAL8Sequence *freqs = NULL;
  REAL8 *phis=NULL;
  int errcode = XLAL_SUCCESS;

  // Enforce convention m2 >= m1
  REAL8 chi1_l, chi2_l;
  REAL8 m1_SI, m2_SI;
  if (m2_SI_in >= m1_SI_in) {
    m1_SI = m1_SI_in;
    m2_SI = m2_SI_in;
    chi1_l = chi1_l_in;
    chi2_l = chi2_l_in;
  }
  else { // swap bodies 1 <-> 2
    m1_SI = m2_SI_in;
    m2_SI = m1_SI_in;
    chi1_l = chi2_l_in;
    chi2_l = chi1_l_in;
  }

  errcode = init_useful_powers(&powers_of_pi, LAL_PI);
  XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_useful_powers() failed.");

  /* Find frequency bounds */
  if (!freqs_in || !freqs_in->data) XLAL_ERROR(XLAL_EFAULT);
  double f_min = freqs_in->data[0];
  double f_max = freqs_in->data[freqs_in->length - 1];
  XLAL_CHECK(f_min > 0, XLAL_EDOM, "Minimum frequency must be positive.\n");
  XLAL_CHECK(f_max >= 0, XLAL_EDOM, "Maximum frequency must be non-negative.\n");

  /* External units: SI; internal units: solar masses */
  const REAL8 m1 = m1_SI / LAL_MSUN_SI;
  const REAL8 m2 = m2_SI / LAL_MSUN_SI;
  const REAL8 M = m1 + m2;
  const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
  const REAL8 q = m2 / m1; /* q >= 1 */
  const REAL8 eta = m1 * m2 / (M*M);    /* Symmetric mass-ratio */
  const REAL8 piM = LAL_PI * m_sec;

  LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; // = {0, 0}

  // Note:
  // * IMRPhenomP uses chi_eff both in the aligned part and the twisting
  // * IMRPhenomPv2 uses chi1_l, chi2_l in the aligned part and chi_eff in the twisting
  const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l) / M; /* Effective aligned spin */
  const REAL8 chil = (1.0+q)/q * chi_eff; /* dimensionless aligned spin of the largest BH */

  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      XLAL_PRINT_WARNING("Warning: IMRPhenomP(v1) is unreviewed.\n");
      if (eta < 0.0453515) /* q = 20 */
          XLAL_ERROR(XLAL_EDOM, "IMRPhenomP(v1): Mass ratio is way outside the calibration range. m1/m2 should be <= 20.\n");
      else if (eta < 0.16)  /* q = 4 */
          XLAL_PRINT_WARNING("IMRPhenomP(v1): Warning: The model is only calibrated for m1/m2 <= 4.\n");
      /* If spins are above 0.9 or below -0.9, throw an error. */
      /* The rationale behind this is given at this page: https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomCdevel-SanityCheck01 */
      if (chi_eff > 0.9 || chi_eff < -0.9)
          XLAL_ERROR(XLAL_EDOM, "IMRPhenomP(v1): Effective spin chi_eff = %g outside the range [-0.9,0.9] is not supported!\n", chi_eff);
      break;
    case IMRPhenomPv2_V:
      if (q > 18.0)
        XLAL_PRINT_WARNING("IMRPhenomPv2: Warning: The underlying non-precessing model is calibrated up to m1/m2 <= 18.\n");
      else if (q > 100.0)
          XLAL_ERROR(XLAL_EDOM, "IMRPhenomPv2: Mass ratio q > 100 which is way outside the calibration range q <= 18.\n");
      CheckMaxOpeningAngle(m1, m2, chi1_l, chi2_l, chip);
      break;
    default:
      XLAL_ERROR( XLAL_EINVAL, "Unknown IMRPhenomP version!\nAt present only v1 and v2 are available." );
      break;
    }

  NNLOanglecoeffs angcoeffs; /* Next-to-next-to leading order PN coefficients for Euler angles alpha and epsilon */
  ComputeNNLOanglecoeffs(&angcoeffs,q,chil,chip);

  /* Compute the offsets due to the choice of integration constant in alpha and epsilon PN formula */
  const REAL8 omega_ref = piM * f_ref;
  const REAL8 logomega_ref = log(omega_ref);
  const REAL8 omega_ref_cbrt = cbrt(piM * f_ref); // == v0
  const REAL8 omega_ref_cbrt2 = omega_ref_cbrt*omega_ref_cbrt;
  const REAL8 alphaNNLOoffset = (angcoeffs.alphacoeff1/omega_ref
                              + angcoeffs.alphacoeff2/omega_ref_cbrt2
                              + angcoeffs.alphacoeff3/omega_ref_cbrt
                              + angcoeffs.alphacoeff4*logomega_ref
                              + angcoeffs.alphacoeff5*omega_ref_cbrt);

  const REAL8 epsilonNNLOoffset = (angcoeffs.epsiloncoeff1/omega_ref
                                + angcoeffs.epsiloncoeff2/omega_ref_cbrt2
                                + angcoeffs.epsiloncoeff3/omega_ref_cbrt
                                + angcoeffs.epsiloncoeff4*logomega_ref
                                + angcoeffs.epsiloncoeff5*omega_ref_cbrt);

  /* Compute Ylm's only once and pass them to PhenomPCoreOneFrequency() below. */
  SpinWeightedSphericalHarmonic_l2 Y2m;
  const REAL8 ytheta  = thetaJ;
  const REAL8 yphi    = 0;
  Y2m.Y2m2 = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -2);
  Y2m.Y2m1 = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -1);
  Y2m.Y20  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  0);
  Y2m.Y21  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  1);
  Y2m.Y22  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  2);


  REAL8 fCut = 0.0;
  REAL8 finspin = 0.0;
  REAL8 f_final = 0.0;

  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      XLAL_PRINT_INFO("*** IMRPhenomP version 1: based on IMRPhenomC ***");
      // PhenomC with ringdown using Barausse 2009 formula for final spin
      PCparams = ComputeIMRPhenomCParamsRDmod(m1, m2, chi_eff, chip);
      if (!PCparams) {
        errcode = XLAL_EFUNC;
        goto cleanup;
      }
      fCut = PCparams->fCut;
      f_final = PCparams->fRingDown;
      break;
    case IMRPhenomPv2_V:
      XLAL_PRINT_INFO("*** IMRPhenomP version 2: based on IMRPhenomD ***");
      // PhenomD uses FinalSpin0815() to calculate the final spin if the spins are aligned.
      // We use a generalized version of FinalSpin0815() that includes the in-plane spin chip.
      finspin = FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(m1, m2, chi1_l, chi2_l, chip);
      if( fabs(finspin) > 1.0 ) {
        XLAL_PRINT_WARNING("Warning: final spin magnitude %g > 1. Setting final spin magnitude = 1.", finspin);
        finspin = copysign(1.0, finspin);
      }

      // IMRPhenomD assumes that m1 >= m2.
      pAmp = ComputeIMRPhenomDAmplitudeCoefficients(eta, chi2_l, chi1_l, finspin);
      pPhi = ComputeIMRPhenomDPhaseCoefficients(eta, chi2_l, chi1_l, finspin);
      XLALSimInspiralTaylorF2AlignedPhasing(&pn, m1, m2, chi1_l, chi2_l, 1.0, 1.0, LAL_SIM_INSPIRAL_SPIN_ORDER_35PN);
      if (!pAmp || !pPhi || !pn) {
        errcode = XLAL_EFUNC;
        goto cleanup;
      }

      // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
      // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
      // was not available when PhenomD was tuned.
      pn->v[6] -= (Subtract3PNSS(m1, m2, M, chi1_l, chi2_l) * pn->v[0]);

      PhiInsPrefactors phi_prefactors;
      errcode = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
      XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_phi_ins_prefactors failed");

      ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors);
      // This should be the same as the ending frequency in PhenomD
      fCut = f_CUT / m_sec;
      f_final = pAmp->fRD / m_sec;
      break;
    default:
      XLALPrintError( "XLAL Error - %s: Unknown IMRPhenomP version!\nAt present only v1 and v2 are available.\n", __func__);
      errcode = XLAL_EINVAL;
      goto cleanup;
      break;
  }

  XLAL_CHECK ( fCut > f_min, XLAL_EDOM, "fCut = %.2g/M <= f_min", fCut );

  /* Default f_max to params->fCut */
  REAL8 f_max_prime = f_max ? f_max : fCut;
  f_max_prime = (f_max_prime > fCut) ? fCut : f_max_prime;
  if (f_max_prime <= f_min){
    XLALPrintError("XLAL Error - %s: f_max <= f_min\n", __func__);
    errcode = XLAL_EDOM;
    goto cleanup;
  }

  /* Allocate hp, hc */

  UINT4 L_fCut = 0; // number of frequency points before we hit fCut
  size_t n = 0;
  UINT4 offset = 0; // Index shift between freqs and the frequency series
  if (deltaF > 0)  { // freqs contains uniform frequency grid with spacing deltaF; we start at frequency 0
    /* Set up output array with size closest power of 2 */
    if (f_max_prime < f_max)  /* Resize waveform if user wants f_max larger than cutoff frequency */
      n = NextPow2(f_max / deltaF) + 1;
    else
      n = NextPow2(f_max_prime / deltaF) + 1;

    /* coalesce at t=0 */
    XLAL_CHECK(XLALGPSAdd(&ligotimegps_zero, -1. / deltaF), XLAL_EFUNC,
    "Failed to shift coalescence time by -1.0/deltaF with deltaF=%g.", deltaF); // shift by overall length in time

    *hptilde = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &ligotimegps_zero, 0.0, deltaF, &lalStrainUnit, n);
    if(!*hptilde) {
      errcode = XLAL_ENOMEM;
      goto cleanup;
    }
    *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &ligotimegps_zero, 0.0, deltaF, &lalStrainUnit, n);
    if(!*hctilde) {
      errcode = XLAL_ENOMEM;
      goto cleanup;
    }

    // Recreate freqs using only the lower and upper bounds
    size_t i_min = (size_t) (f_min / deltaF);
    size_t i_max = (size_t) (f_max_prime / deltaF);
    freqs = XLALCreateREAL8Sequence(i_max - i_min);
    if (!freqs) {
      errcode = XLAL_EFUNC;
      XLALPrintError("XLAL Error - %s: Frequency array allocation failed.", __func__);
      goto cleanup;
    }
    for (UINT4 i=i_min; i<i_max; i++)
      freqs->data[i-i_min] = i*deltaF;
    L_fCut = freqs->length;
    offset = i_min;
  } else { // freqs contains frequencies with non-uniform spacing; we start at lowest given frequency
    n = freqs_in->length;

    *hptilde = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &ligotimegps_zero, f_min, 0, &lalStrainUnit, n);
    if(!*hptilde) {
      XLALPrintError("XLAL Error - %s: Failed to allocate frequency series for hptilde polarization with f_min=%f and %tu bins.", __func__, f_min, n);
      errcode = XLAL_ENOMEM;
      goto cleanup;
    }
    *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &ligotimegps_zero, f_min, 0, &lalStrainUnit, n);
    if(!*hctilde) {
      XLALPrintError("XLAL Error - %s: Failed to allocate frequency series for hctilde polarization with f_min=%f and %tu bins.", __func__, f_min, n);
      errcode = XLAL_ENOMEM;
      goto cleanup;
    }
    offset = 0;

    // Enforce that FS is strictly increasing
    // (This is needed for phase correction towards the end of this function.)
    for (UINT4 i=1; i<n; i++)
    {
      if (!(freqs_in->data[i] > freqs_in->data[i-1]))
      {
        XLALPrintError("XLAL Error - %s: Frequency sequence must be strictly increasing!\n",  __func__);
        errcode = XLAL_EDOM;
        goto cleanup;
      }
    }

    freqs = XLALCreateREAL8Sequence(n);
    if (!freqs) {
      XLALPrintError("XLAL Error - %s: Frequency array allocation failed.",  __func__);
      errcode = XLAL_ENOMEM;
      goto cleanup;
    }
    // Restrict sequence to frequencies <= fCut
    for (UINT4 i=0; i<n; i++)
      if (freqs_in->data[i] <= fCut) {
        freqs->data[i] = freqs_in->data[i];
        L_fCut++;
      }
  }


  memset((*hptilde)->data->data, 0, n * sizeof(COMPLEX16));
  memset((*hctilde)->data->data, 0, n * sizeof(COMPLEX16));
  XLALUnitMultiply(&((*hptilde)->sampleUnits), &((*hptilde)->sampleUnits), &lalSecondUnit);
  XLALUnitMultiply(&((*hctilde)->sampleUnits), &((*hctilde)->sampleUnits), &lalSecondUnit);

  phis = XLALMalloc(L_fCut*sizeof(REAL8)); // array for waveform phase
  if(!phis) {
    errcode = XLAL_ENOMEM;
    goto cleanup;
  }
  REAL8 phasing = 0;

  AmpInsPrefactors amp_prefactors;
  PhiInsPrefactors phi_prefactors;

  if (IMRPhenomP_version == IMRPhenomPv2_V) {
    errcode = init_amp_ins_prefactors(&amp_prefactors, pAmp);
    XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_amp_ins_prefactors() failed.");
    errcode = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
    XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_phi_ins_prefactors() failed.");
  }

  /*
    We can't call XLAL_ERROR() directly with OpenMP on.
    Keep track of return codes for each thread and in addition use flush to get out of
    the parallel for loop as soon as possible if something went wrong in any thread.
  */
  #pragma omp parallel for
  for (UINT4 i=0; i<L_fCut; i++) { // loop over frequency points in sequence
    COMPLEX16 hp_val = 0.0;
    COMPLEX16 hc_val = 0.0;
    double f = freqs->data[i];
    int j = i + offset; // shift index for frequency series if needed

    int per_thread_errcode;

    #pragma omp flush(errcode)
    if (errcode != XLAL_SUCCESS)
      goto skip;

    /* Generate the waveform */
    per_thread_errcode = PhenomPCoreOneFrequency(f, eta, chi1_l, chi2_l, chip, distance, M, phic,
                              pAmp, pPhi, PCparams, pn, &angcoeffs, &Y2m,
                              alphaNNLOoffset - alpha0, epsilonNNLOoffset,
                              &hp_val, &hc_val, &phasing, IMRPhenomP_version, &amp_prefactors, &phi_prefactors);

    if (per_thread_errcode != XLAL_SUCCESS) {
      errcode = per_thread_errcode;
      #pragma omp flush(errcode)
    }

    ((*hptilde)->data->data)[j] = hp_val;
    ((*hctilde)->data->data)[j] = hc_val;

    phis[i] = phasing;

    skip: /* this statement intentionally left blank */;
  }

  /* Correct phasing so we coalesce at t=0 (with the definition of the epoch=-1/deltaF above) */
  /* We apply the same time shift to hptilde and hctilde based on the overall phasing returned by PhenomPCoreOneFrequency */
  /* Set up spline for phase */
  acc = gsl_interp_accel_alloc();
  phiI = gsl_spline_alloc(gsl_interp_cspline, L_fCut);
  XLAL_CHECK(phiI, XLAL_ENOMEM, "Failed to allocate GSL spline with %d points for phase.", L_fCut);

  gsl_spline_init(phiI, freqs->data, phis, L_fCut);

  // Prevent gsl interpolation errors
  if (f_final > freqs->data[L_fCut-1])
    f_final = freqs->data[L_fCut-1];
  if (f_final < freqs->data[0])
  {
    XLALPrintError("XLAL Error - %s: f_ringdown = %f < f_min\n", __func__, f_final);
    errcode = XLAL_EDOM;
    goto cleanup;
  }

  /* Time correction is t(f_final) = 1/(2pi) dphi/df (f_final) */
  REAL8 t_corr = gsl_spline_eval_deriv(phiI, f_final, acc) / (2*LAL_PI);
  /* Now correct phase */
  for (UINT4 i=0; i<L_fCut; i++) { // loop over frequency points in sequence
    double f = freqs->data[i];
    COMPLEX16 phase_corr = cexp(-2*LAL_PI * I * f * t_corr);
    int j = i + offset; // shift index for frequency series if needed
    ((*hptilde)->data->data)[j] *= phase_corr;
    ((*hctilde)->data->data)[j] *= phase_corr;
  }

  cleanup:
  if(phis) XLALFree(phis);
  if(phiI) gsl_spline_free(phiI);
  if(acc) gsl_interp_accel_free(acc);

  if(PCparams) XLALFree(PCparams);
  if(pAmp) XLALFree(pAmp);
  if(pPhi) XLALFree(pPhi);
  if(pn) XLALFree(pn);

  if(freqs) XLALDestroyREAL8Sequence(freqs);

  if( errcode != XLAL_SUCCESS ) {
    if(*hptilde)
      XLALDestroyCOMPLEX16FrequencySeries(*hptilde);
    if(*hctilde)
      XLALDestroyCOMPLEX16FrequencySeries(*hctilde);
    XLAL_ERROR(errcode);
  }
  else {
    return XLAL_SUCCESS;
  }
}

/* ***************************** PhenomP internal functions *********************************/

/**
 * \f[
 * \newcommand{\hP}{h^\mathrm{P}}
 * \newcommand{\PAmp}{A^\mathrm{P}}
 * \newcommand{\PPhase}{\phi^\mathrm{P}}
 * \newcommand{\chieff}{\chi_\mathrm{eff}}
 * \newcommand{\chip}{\chi_\mathrm{p}}
 * \f]
 * Internal core function to calculate
 * plus and cross polarizations of the PhenomP model
 * for a single frequency.
 *
 * The general expression for the modes \f$\hP_{2m}(t)\f$
 * is given by Eq. 1 of arXiv:1308.3271.
 * We calculate the frequency domain l=2 plus and cross polarizations separately
 * for each m = -2, ... , 2.
 *
 * The expression of the polarizations times the \f$Y_{lm}\f$
 * in code notation are:
 * \f{equation*}{
 * \left(\tilde{h}_{2m}\right)_+ = e^{-2i \epsilon}
 * \left(e^{-i m \alpha} d^2_{-2,m} (-2Y_{2m})
 * + e^{+i m \alpha} d^2_{2,m} (-2Y_{2m})^*\right) \cdot \hP / 2 \,,
 * \f}
 * \f{equation*}{
 * \left(\tilde{h}_{2m}\right)_x = e^{-2i \epsilon}
 * \left(e^{-i m \alpha} d^2_{-2,m} (-2Y_{2m})
 * - e^{+i m \alpha} d^2_{2,m} (-2Y_{2m})^*\right) \cdot \hP / 2 \,,
 * \f}
 * where the \f$d^l_{m',m}\f$ are Wigner d-matrices evaluated at \f$-\beta\f$,
 * and \f$\hP\f$ is the Phenom[C,D] frequency domain model:
 * \f{equation*}{
 * \hP(f) = \PAmp(f) e^{-i \PPhase(f)} \,.
 * \f}
 *
 * Note that in arXiv:1308.3271, the angle \f$\beta\f$ (beta) is called iota.
 *
 * For IMRPhenomP(v1) we put all spin on the larger BH,
 * convention: \f$m_2 \geq m_1\f$.
 * Hence:
 * \f{eqnarray*}{
 * \chieff      &=& \left( m_1 \cdot \chi_1 + m_2 \cdot \chi_2 \right)/M \,,\\
 * \chi_l       &=& \chieff / m_2 \quad (\text{for } M=1) \,,\\
 * S_L          &=& m_2^2 \chi_l = m_2 \cdot M \cdot \chieff
 *               = \frac{q}{1+q} \cdot \chieff \quad (\text{for } M=1) \,.
 * \f}
 *
 * For IMRPhenomPv2 we use both aligned spins:
 * \f{equation*}{
 * S_L = \chi_1 \cdot m_1^2 + \chi_2 \cdot m_2^2 \,.
 * \f}
 *
 * For both IMRPhenomP(v1) and IMRPhenomPv2 we put the in-plane spin on the larger BH:
 * \f{equation*}{
 * S_\mathrm{perp} = \chip \cdot m_2^2
 * \f}
 * (perpendicular spin).
 */
static int PhenomPCoreOneFrequency(
  const REAL8 fHz,                            /**< Frequency (Hz) */
  const REAL8 eta,                            /**< Symmetric mass ratio */
  const REAL8 chi1_l,                         /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l,                         /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                           /**< Dimensionless spin in the orbital plane */
  const REAL8 distance,                       /**< Distance of source (m) */
  const REAL8 M,                              /**< Total mass (Solar masses) */
  const REAL8 phic,                           /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  IMRPhenomDAmplitudeCoefficients *pAmp,      /**< Internal IMRPhenomD amplitude coefficients */
  IMRPhenomDPhaseCoefficients *pPhi,          /**< Internal IMRPhenomD phase coefficients */
  BBHPhenomCParams *PCparams,                 /**< Internal PhenomC parameters */
  PNPhasingSeries *PNparams,                  /**< PN inspiral phase coefficients */
  NNLOanglecoeffs *angcoeffs,                 /**< Struct with PN coeffs for the NNLO angles */
  SpinWeightedSphericalHarmonic_l2 *Y2m,      /**< Struct of l=2 spherical harmonics of spin weight -2 */
  const REAL8 alphaoffset,                    /**< f_ref dependent offset for alpha angle (azimuthal precession angle) */
  const REAL8 epsilonoffset,                  /**< f_ref dependent offset for epsilon angle */
  COMPLEX16 *hp,                              /**< [out] plus polarization \f$\tilde h_+\f$ */
  COMPLEX16 *hc,                              /**< [out] cross polarization \f$\tilde h_x\f$ */
  REAL8 *phasing,                             /**< [out] overall phasing */
  IMRPhenomP_version_type IMRPhenomP_version, /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
  AmpInsPrefactors *amp_prefactors,           /**< pre-calculated (cached for saving runtime) coefficients for amplitude. See LALSimIMRPhenomD_internals.c*/
  PhiInsPrefactors *phi_prefactors            /**< pre-calculated (cached for saving runtime) coefficients for phase. See LALSimIMRPhenomD_internals.*/)
{
  XLAL_CHECK(angcoeffs != NULL, XLAL_EFAULT);
  XLAL_CHECK(hp != NULL, XLAL_EFAULT);
  XLAL_CHECK(hc != NULL, XLAL_EFAULT);
  XLAL_CHECK(Y2m != NULL, XLAL_EFAULT);
  XLAL_CHECK(phasing != NULL, XLAL_EFAULT);

  REAL8 f = fHz*LAL_MTSUN_SI*M; /* Frequency in geometric units */

  REAL8 aPhenom = 0.0;
  REAL8 phPhenom = 0.0;
  int errcode = XLAL_SUCCESS;
  UsefulPowers powers_of_f;

  const REAL8 q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta)/(2.0*eta);
  const REAL8 m1 = 1.0/(1.0+q);       /* Mass of the smaller BH for unit total mass M=1. */
  const REAL8 m2 = q/(1.0+q);         /* Mass of the larger BH for unit total mass M=1. */
  const REAL8 Sperp = chip*(m2*m2);   /* Dimensionfull spin component in the orbital plane. S_perp = S_2_perp */
  REAL8 SL;                           /* Dimensionfull aligned spin. */
  const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l); /* effective spin for M=1 */

  /* Calculate Phenom amplitude and phase for a given frequency. */
  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      XLAL_CHECK(PCparams != NULL, XLAL_EFAULT);
      errcode = IMRPhenomCGenerateAmpPhase( &aPhenom, &phPhenom, fHz, eta, PCparams );
      if( errcode != XLAL_SUCCESS ) XLAL_ERROR(XLAL_EFUNC);
      SL = chi_eff*m2;        /* Dimensionfull aligned spin of the largest BH. SL = m2^2 chil = m2*M*chi_eff */
      break;
    case IMRPhenomPv2_V:
      XLAL_CHECK(pAmp != NULL, XLAL_EFAULT);
      XLAL_CHECK(pPhi != NULL, XLAL_EFAULT);
      XLAL_CHECK(PNparams != NULL, XLAL_EFAULT);
      XLAL_CHECK(amp_prefactors != NULL, XLAL_EFAULT);
      XLAL_CHECK(phi_prefactors != NULL, XLAL_EFAULT);
      errcode = init_useful_powers(&powers_of_f, f);
      XLAL_CHECK(errcode == XLAL_SUCCESS, errcode, "init_useful_powers failed for f");
      aPhenom = IMRPhenDAmplitude(f, pAmp, &powers_of_f, amp_prefactors);
      phPhenom = IMRPhenDPhase(f, pPhi, PNparams, &powers_of_f, phi_prefactors);
      SL = chi1_l*m1*m1 + chi2_l*m2*m2;        /* Dimensionfull aligned spin. */
      break;
    default:
      XLAL_ERROR( XLAL_EINVAL, "Unknown IMRPhenomP version!\nAt present only v1 and v2 are available." );
      break;
  }

  phPhenom -= 2.*phic; /* Note: phic is orbital phase */
  REAL8 amp0 = M * LAL_MRSUN_SI * M * LAL_MTSUN_SI / distance;
  COMPLEX16 hP = amp0 * aPhenom * cexp(-I*phPhenom); /* Assemble IMRPhenom waveform. */

  /* Compute PN NNLO angles */
  const REAL8 omega = LAL_PI * f;
  const REAL8 logomega = log(omega);
  const REAL8 omega_cbrt = cbrt(omega);
  const REAL8 omega_cbrt2 = omega_cbrt*omega_cbrt;

  REAL8 alpha = (angcoeffs->alphacoeff1/omega
              + angcoeffs->alphacoeff2/omega_cbrt2
              + angcoeffs->alphacoeff3/omega_cbrt
              + angcoeffs->alphacoeff4*logomega
              + angcoeffs->alphacoeff5*omega_cbrt) - alphaoffset;

  REAL8 epsilon = (angcoeffs->epsiloncoeff1/omega
                + angcoeffs->epsiloncoeff2/omega_cbrt2
                + angcoeffs->epsiloncoeff3/omega_cbrt
                + angcoeffs->epsiloncoeff4*logomega
                + angcoeffs->epsiloncoeff5*omega_cbrt) - epsilonoffset;

  /* Calculate intermediate expressions cos(beta/2), sin(beta/2) and powers thereof for Wigner d's. */
  REAL8 cBetah, sBetah; /* cos(beta/2), sin(beta/2) */
  switch (IMRPhenomP_version) {
    case IMRPhenomPv1_V:
      WignerdCoefficients_SmallAngleApproximation(&cBetah, &sBetah, omega_cbrt, SL, eta, Sperp);
      break;
    case IMRPhenomPv2_V:
      WignerdCoefficients(&cBetah, &sBetah, omega_cbrt, SL, eta, Sperp);
      break;
  default:
    XLAL_ERROR( XLAL_EINVAL, " Unknown IMRPhenomP version!\nAt present only v1 and v2 are available." );
    break;
  }

  const REAL8 cBetah2 = cBetah*cBetah;
  const REAL8 cBetah3 = cBetah2*cBetah;
  const REAL8 cBetah4 = cBetah3*cBetah;
  const REAL8 sBetah2 = sBetah*sBetah;
  const REAL8 sBetah3 = sBetah2*sBetah;
  const REAL8 sBetah4 = sBetah3*sBetah;

  /* Compute Wigner d coefficients
    The expressions below agree with refX [Goldstein?] and Mathematica
    d2  = Table[WignerD[{2, mp, 2}, 0, -\[Beta], 0], {mp, -2, 2}]
    dm2 = Table[WignerD[{2, mp, -2}, 0, -\[Beta], 0], {mp, -2, 2}]
  */
  COMPLEX16 d2[5]   = {sBetah4, 2*cBetah*sBetah3, sqrt_6*sBetah2*cBetah2, 2*cBetah3*sBetah, cBetah4};
  COMPLEX16 dm2[5]  = {d2[4], -d2[3], d2[2], -d2[1], d2[0]}; /* Exploit symmetry d^2_{-2,-m} = (-1)^m d^2_{2,m} */

  COMPLEX16 Y2mA[5] = {Y2m->Y2m2, Y2m->Y2m1, Y2m->Y20, Y2m->Y21, Y2m->Y22};
  COMPLEX16 hp_sum = 0;
  COMPLEX16 hc_sum = 0;

  /* Sum up contributions to \tilde h+ and \tilde hx */
  /* Precompute powers of e^{i m alpha} */
  COMPLEX16 cexp_i_alpha = cexp(+I*alpha);
  COMPLEX16 cexp_2i_alpha = cexp_i_alpha*cexp_i_alpha;
  COMPLEX16 cexp_mi_alpha = 1.0/cexp_i_alpha;
  COMPLEX16 cexp_m2i_alpha = cexp_mi_alpha*cexp_mi_alpha;
  COMPLEX16 cexp_im_alpha[5] = {cexp_m2i_alpha, cexp_mi_alpha, 1.0, cexp_i_alpha, cexp_2i_alpha};
  for(int m=-2; m<=2; m++) {
    COMPLEX16 T2m   = cexp_im_alpha[-m+2] * dm2[m+2] *      Y2mA[m+2];  /*  = cexp(-I*m*alpha) * dm2[m+2] *      Y2mA[m+2] */
    COMPLEX16 Tm2m  = cexp_im_alpha[m+2]  * d2[m+2]  * conj(Y2mA[m+2]); /*  = cexp(+I*m*alpha) * d2[m+2]  * conj(Y2mA[m+2]) */
    hp_sum +=     T2m + Tm2m;
    hc_sum += +I*(T2m - Tm2m);
  }

  COMPLEX16 eps_phase_hP = cexp(-2*I*epsilon) * hP / 2.0;
  *hp = eps_phase_hP * hp_sum;
  *hc = eps_phase_hP * hc_sum;

  // Return phasing for time-shift correction
  *phasing = -phPhenom; // ignore alpha and epsilon contributions

  return XLAL_SUCCESS;
}

/**
 * Next-to-next-to-leading order PN coefficients
 * for Euler angles \f$\alpha\f$ and \f$\epsilon\f$.
 */
static void ComputeNNLOanglecoeffs(
  NNLOanglecoeffs *angcoeffs, /**< [out] Structure to store results */
  const REAL8 q,              /**< Mass-ratio (convention q>1) */
  const REAL8 chil,           /**< Dimensionless aligned spin of the largest BH */
  const REAL8 chip)           /**< Dimensionless spin component in the orbital plane */
{
  const REAL8 m2 = q/(1. + q);
  const REAL8 m1 = 1./(1. + q);
  const REAL8 dm = m1 - m2;
  const REAL8 mtot = 1.;
  const REAL8 eta = m1*m2; /* mtot = 1 */
  const REAL8 eta2 = eta*eta;
  const REAL8 eta3 = eta2*eta;
  const REAL8 eta4 = eta3*eta;
  const REAL8 mtot2 = mtot*mtot;
  const REAL8 mtot4 = mtot2*mtot2;
  const REAL8 mtot6 = mtot4*mtot2;
  const REAL8 mtot8 = mtot6*mtot2;
  const REAL8 chil2 = chil*chil;
  const REAL8 chip2 = chip*chip;
  const REAL8 chip4 = chip2*chip2;
  const REAL8 dm2 = dm*dm;
  const REAL8 dm3 = dm2*dm;
  const REAL8 m2_2 = m2*m2;
  const REAL8 m2_3 = m2_2*m2;
  const REAL8 m2_4 = m2_3*m2;
  const REAL8 m2_5 = m2_4*m2;
  const REAL8 m2_6 = m2_5*m2;
  const REAL8 m2_7 = m2_6*m2;
  const REAL8 m2_8 = m2_7*m2;

  XLAL_CHECK_VOID(angcoeffs != NULL, XLAL_EFAULT);

  angcoeffs->alphacoeff1 = (-0.18229166666666666 - (5*dm)/(64.*m2));

  angcoeffs->alphacoeff2 = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta));

  angcoeffs->alphacoeff3 = (-1.7952473958333333 - (4555*dm)/(7168.*m2) -
        (15*chip2*dm*m2_3)/(128.*mtot4*eta2) -
        (35*chip2*m2_4)/(128.*mtot4*eta2) - (515*eta)/384. - (15*dm2*eta)/(256.*m2_2) -
        (175*dm*eta)/(256.*m2));

  angcoeffs->alphacoeff4 = - (35*LAL_PI)/48. - (5*dm*LAL_PI)/(16.*m2) +
     (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) +
     (2545*m2_2*chil)/(1152.*mtot2) -
     (5*chip2*dm*m2_5*chil)/(128.*mtot6*eta3) -
     (35*chip2*m2_6*chil)/(384.*mtot6*eta3) + (2035*dm*m2*chil)/(21504.*mtot2*eta) +
     (2995*m2_2*chil)/(9216.*mtot2*eta);

  angcoeffs->alphacoeff5 = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) -
        (15*chip4*dm*m2_7)/(512.*mtot8*eta4) -
        (35*chip4*m2_8)/(512.*mtot8*eta4) -
        (485*chip2*dm*m2_3)/(14336.*mtot4*eta2) +
        (475*chip2*m2_4)/(6144.*mtot4*eta2) +
        (15*chip2*dm2*m2_2)/(256.*mtot4*eta) + (145*chip2*dm*m2_3)/(512.*mtot4*eta) +
        (575*chip2*m2_4)/(1536.*mtot4*eta) + (39695*eta)/86016. + (1615*dm2*eta)/(28672.*m2_2) -
        (265*dm*eta)/(14336.*m2) + (955*eta2)/576. + (15*dm3*eta2)/(1024.*m2_3) +
        (35*dm2*eta2)/(256.*m2_2) + (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*LAL_PI*chil)/(16.*mtot2*eta) -
        (35*m2_2*LAL_PI*chil)/(16.*mtot2*eta) + (15*chip2*dm*m2_7*chil2)/(128.*mtot8*eta4) +
        (35*chip2*m2_8*chil2)/(128.*mtot8*eta4) +
        (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
        (1645*m2_4*chil2)/(192.*mtot4*eta));

  angcoeffs->epsiloncoeff1 = (-0.18229166666666666 - (5*dm)/(64.*m2));

  angcoeffs->epsiloncoeff2 = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta));

  angcoeffs->epsiloncoeff3 = (-1.7952473958333333 - (4555*dm)/(7168.*m2) - (515*eta)/384. -
        (15*dm2*eta)/(256.*m2_2) - (175*dm*eta)/(256.*m2));

  angcoeffs->epsiloncoeff4 = - (35*LAL_PI)/48. - (5*dm*LAL_PI)/(16.*m2) +
     (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) +
     (2545*m2_2*chil)/(1152.*mtot2) + (2035*dm*m2*chil)/(21504.*mtot2*eta) +
     (2995*m2_2*chil)/(9216.*mtot2*eta);

  angcoeffs->epsiloncoeff5 = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) + (39695*eta)/86016. +
        (1615*dm2*eta)/(28672.*m2_2) - (265*dm*eta)/(14336.*m2) + (955*eta2)/576. +
        (15*dm3*eta2)/(1024.*m2_3) + (35*dm2*eta2)/(256.*m2_2) +
        (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*LAL_PI*chil)/(16.*mtot2*eta) - (35*m2_2*LAL_PI*chil)/(16.*mtot2*eta) +
        (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
        (1645*m2_4*chil2)/(192.*mtot4*eta));
}

/**
 * Simple 2PN version of the orbital angular momentum L,
 * without any spin terms expressed as a function of v.
 * For IMRPhenomP(v2).
 *
 *  Reference:
 *  - Boh&eacute; et al, 1212.5520v2 Eq 4.7 first line
 */
static REAL8 L2PNR(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta) /**< Symmetric mass-ratio */
{
  const REAL8 eta2 = eta*eta;
  const REAL8 x = v*v;
  const REAL8 x2 = x*x;
  return (eta*(1.0 + (1.5 + eta/6.0)*x + (3.375 - (19.0*eta)/8. - eta2/24.0)*x2)) / sqrt(x);
}

/**
 * Simple 2PN version of the orbital angular momentum L,
 * without any spin terms expressed as a function of v.
 * For IMRPhenomP(v1).
 *
 * Reference:
 *  - Kidder, Phys. Rev. D 52, 821–847 (1995), Eq. 2.9
 */
static REAL8 L2PNR_v1(
  const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 eta) /**< Symmetric mass-ratio */
{
  const REAL8 mu = eta; /* M=1 */
  const REAL8 v2 = v*v;
  const REAL8 v3 = v2*v;
  const REAL8 v4 = v3*v;
  const REAL8 eta2 = eta*eta;
  const REAL8 b = (4.75 + eta/9.)*eta*v4;


  return mu*sqrt((1 - ((3 - eta)*v2)/3. + b)/v2)*
    (1 + ((1 - 3*eta)*v2)/2. + (3*(1 - 7*eta + 13*eta2)*v4)/8. +
      ((14 - 41*eta + 4*eta2)*v4)/(4.*pow_2_of(1 - ((3 - eta)*v2)/3. + b)) +
      ((3 + eta)*v2)/(1 - ((3 - eta)*v2)/3. + b) +
      ((7 - 10*eta - 9*eta2)*v4)/(2.*(1 - ((3 - eta)*v2)/3. + b)));
}

/** Expressions used for the WignerD symbol
  * with full expressions for the angles.
  * Used for IMRPhenomP(v2):
  * \f{equation}{
  * \cos(\beta) = \hat J . \hat L
  *             = \left( 1 + \left( S_\mathrm{p} / (L + S_L) \right)^2 \right)^{-1/2}
  *             = \left( L + S_L \right) / \sqrt{ \left( L + S_L \right)^2 + S_p^2 }
  *             = \mathrm{sign}\left( L + S_L \right) \cdot \left( 1 + \left( S_p / \left(L + S_L\right)\right)^2 \right)^{-1/2}
  * \f}
 */
static void WignerdCoefficients(
  REAL8 *cos_beta_half, /**< [out] cos(beta/2) */
  REAL8 *sin_beta_half, /**< [out] sin(beta/2) */
  const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,       /**< Dimensionfull aligned spin */
  const REAL8 eta,      /**< Symmetric mass-ratio */
  const REAL8 Sp)       /**< Dimensionfull spin component in the orbital plane */
{
  XLAL_CHECK_VOID(cos_beta_half != NULL, XLAL_EFAULT);
  XLAL_CHECK_VOID(sin_beta_half != NULL, XLAL_EFAULT);
  /* We define the shorthand s := Sp / (L + SL) */
  const REAL8 L = L2PNR(v, eta);
    // We ignore the sign of L + SL below.
  REAL8 s = Sp / (L + SL);  /* s := Sp / (L + SL) */
  REAL8 s2 = s*s;
  REAL8 cos_beta = 1.0 / sqrt(1.0 + s2);
  *cos_beta_half = + sqrt( (1.0 + cos_beta) / 2.0 );  /* cos(beta/2) */
  *sin_beta_half = + sqrt( (1.0 - cos_beta) / 2.0 );  /* sin(beta/2) */
}

/** Expressions used for the WignerD symbol
  * with small angle approximation.
  * Used for IMRPhenomP(v1):
  * \f{equation}{
  * \cos(\beta) = \hat J . \hat L
  *             = \left(1 + \left( S_\mathrm{p} / (L + S_L)\right)^2 \right)^{-1/2}
  * \f}
  * We use the expression
  * \f{equation}{
  * \cos(\beta/2) \approx (1 + s^2 / 4 )^{-1/2} \,,
  * \f}
  * where \f$s := S_p / (L + S_L)\f$.
 */
static void WignerdCoefficients_SmallAngleApproximation(
  REAL8 *cos_beta_half, /**< Output: cos(beta/2) */
  REAL8 *sin_beta_half, /**< Output: sin(beta/2) */
  const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
  const REAL8 SL,       /**< Dimensionfull aligned spin */
  const REAL8 eta,      /**< Symmetric mass-ratio */
  const REAL8 Sp)       /**< Dimensionfull spin component in the orbital plane */
{
  XLAL_CHECK_VOID(cos_beta_half != NULL, XLAL_EFAULT);
  XLAL_CHECK_VOID(sin_beta_half != NULL, XLAL_EFAULT);
  REAL8 s = Sp / (L2PNR_v1(v, eta) + SL);  /* s := Sp / (L + SL) */
  REAL8 s2 = s*s;
  *cos_beta_half = 1.0/sqrt(1.0 + s2/4.0);           /* cos(beta/2) */
  *sin_beta_half = sqrt(1.0 - 1.0/(1.0 + s2/4.0));   /* sin(beta/2) */
}

/**
 * In this helper function we check whether the maximum opening angle during the evolution
 * becomes larger than pi/2 or pi/4, in which case a warning is issued.
 */
static void CheckMaxOpeningAngle(
  const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
  const REAL8 chi1_l, /**< Aligned spin of BH 1 */
  const REAL8 chi2_l, /**< Aligned spin of BH 2 */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
) {
  REAL8 M = m1+m2;
  REAL8 m1_normalized = m1/M;
  REAL8 m2_normalized = m2/M;
  REAL8 eta = m1_normalized*m2_normalized;
  REAL8 v_at_max_beta = sqrt( 2*(9. + eta - sqrt(1539 - 1008*eta - 17*eta*eta)) / 3. / (eta*eta + 57.*eta - 81.) );
  // The coefficients above come from finding the roots of the derivative of L2PN
  REAL8 SL = m1_normalized*m1_normalized*chi1_l + m2_normalized*m2_normalized*chi2_l;
  REAL8 Sperp = m2_normalized*m2_normalized * chip;
  REAL8 cBetah, sBetah;
  WignerdCoefficients(&cBetah, &sBetah, v_at_max_beta, SL, eta, Sperp);
  REAL8 L_min = L2PNR(v_at_max_beta,eta);
  REAL8 max_beta = 2*acos(cBetah);
    /** If L+SL becomes <0, WignerdCoefficients does not track the angle between J and L anymore (see tech doc, choice of + sign
     * so that the Wigner coefficients are OK in the aligned spin limit) and the model may become pathological as one moves away from
     * the aligned spin limit.
     * If this does not happen, then max_beta is the actual maximum opening angle as predicted by the model.
     */
  if (L_min + SL < 0. && chip > 0.)
    XLAL_PRINT_WARNING("The maximum opening angle exceeds Pi/2.\nThe model may be pathological in this regime.");
  else if (max_beta > LAL_PI_4)
    XLAL_PRINT_WARNING("The maximum opening angle %g is larger than Pi/4.\nThe model has not been tested against NR in this regime.", max_beta);
}

/**
 * Wrapper for final-spin formula based on:
 * - IMRPhenomD's FinalSpin0815() for aligned spins.
 *
 * We use their convention m1>m2
 * and put <b>all in-plane spin on the larger BH</b>.
 *
 * In the aligned limit return the FinalSpin0815 value.
 */
static REAL8 FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(
  const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
  const REAL8 chi1_l, /**< Aligned spin of BH 1 */
  const REAL8 chi2_l, /**< Aligned spin of BH 2 */
  const REAL8 chip)   /**< Dimensionless spin in the orbital plane */
{
  const REAL8 M = m1+m2;
  const REAL8 eta = m1*m2/(M*M);

  REAL8 af_parallel, q_factor;
  if (m1 >= m2) {
    q_factor = m1/M;
    af_parallel = FinalSpin0815(eta, chi1_l, chi2_l);
  }
  else {
    q_factor = m2/M;
    af_parallel = FinalSpin0815(eta, chi2_l, chi1_l);
  }  

  REAL8 Sperp = chip * q_factor*q_factor;
  REAL8 af = copysign(1.0, af_parallel) * sqrt(Sperp*Sperp + af_parallel*af_parallel);
  return af;
}

/**
 * Wrapper for final-spin formula based on:
 * - Barausse \& Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009,
 * arXiv:0904.2577
 *
 * We use their convention m1>m2
 * and put <b>all spin on the larger BH</b>:
 *
 * a1 = (chip, 0, chi), a2 = (0,0,0), L = (0,0,1)
 */
static REAL8 FinalSpinBarausse2009_all_spin_on_larger_BH(
  const REAL8 nu,     /**< Symmetric mass-ratio */
  const REAL8 chi,    /**< Effective aligned spin of the binary:  chi = (m1*chi1 + m2*chi2)/M  */
  const REAL8 chip)   /**< Dimensionless spin in the orbital plane */
{

  const REAL8 a1_x = chip;
  const REAL8 a1_y = 0;
  const REAL8 a1_z = chi;
  const REAL8 a2_x = 0;
  const REAL8 a2_y = 0;
  const REAL8 a2_z = 0;

  const REAL8 a1 = sqrt(a1_x*a1_x + a1_y*a1_y + a1_z*a1_z);
  const REAL8 a2 = sqrt(a2_x*a2_x + a2_y*a2_y + a2_z*a2_z);

  const REAL8 cos_alpha = (a1*a2 == 0) ? 0.0 : a1_z*a2_z/(a1*a2); /* cos(alpha) = \hat a1 . \hat a2 (Eq. 7) */
  const REAL8 cos_beta_tilde  = (a1 == 0) ? 0.0 : a1_z/a1;  /* \cos(\tilde \beta)  = \hat a1 . \hat L  (Eq. 9) */
  const REAL8 cos_gamma_tilde = (a2 == 0) ? 0.0 : a2_z/a2;  /* \cos(\tilde \gamma) = \hat a2 . \hat L (Eq. 9) */

  return FinalSpinBarausse2009(nu, a1, a2, cos_alpha, cos_beta_tilde, cos_gamma_tilde);
}

/**
 * Final-spin formula based on:
 * - Barausse \& Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009,
 * arXiv:0904.2577
 *
 * We use their convention m1>m2.
 */
static REAL8 FinalSpinBarausse2009(
  const REAL8 nu,               /**< Symmetric mass-ratio */
  const REAL8 a1,               /**< |a_1| norm of dimensionless spin vector for BH 1 */
  const REAL8 a2,               /**< |a_2| norm of dimensionless spin vector for BH 2 */
  const REAL8 cos_alpha,        /**< \f$\cos(\alpha) = \hat a_1 . \hat a_2\f$ (Eq. 7) */
  const REAL8 cos_beta_tilde,   /**< \f$\cos(\tilde \beta)  = \hat a_1 . \hat L\f$ (Eq. 9) */
  const REAL8 cos_gamma_tilde)  /**< \f$\cos(\tilde \gamma) = \hat a_2 . \hat L\f$ (Eq. 9)*/
{
  REAL8 q = (2*nu)/(1 + sqrt(1 - 4*nu) - 2*nu);

  /* These parameters are defined in eq. 3. */
  const REAL8 s4 = -0.1229;
  const REAL8 s5 = 0.4537;
  const REAL8 t0 = -2.8904;
  const REAL8 t2 = -3.5171;
  const REAL8 t3 = 2.5763;

  /* shorthands */
  const REAL8 nu2 = nu*nu;
  const REAL8 q2 = q*q;
  const REAL8 q4 = q2*q2;
  const REAL8 q2p = 1 + q2;
  const REAL8 q2p2 = q2p*q2p;
  const REAL8 qp = 1 + q;
  const REAL8 qp2 = qp*qp;
  const REAL8 a1_2 = a1*a1;
  const REAL8 a2_2 = a2*a2;

  /* l = \tilde l/(m1*m2), where \tilde l = S_fin - (S1 + S2) = L - J_rad (Eq. 4) */
  const REAL8 l = 2*sqrt(3.0) + t2*nu + t3*nu2
                + (s4 / q2p2) * (a1_2 + a2_2*q4 + 2*a1*a2*q2*cos_alpha)
                + ((s5*nu + t0 + 2)/q2p) * (a1*cos_beta_tilde + a2*cos_gamma_tilde*q2); /* |l| (Eq. 10) */
  const REAL8 l2 = l*l;

  /* a_fin = S_fin/M^2  (Eq. 6) */
  const REAL8 a_fin = (1.0 / qp2) * sqrt(a1_2 + a2_2*q4 + 2*a1*a2*q2*cos_alpha + 2*(a1*cos_beta_tilde + a2*q2*cos_gamma_tilde)*l*q + l2*q2);
  return a_fin;
}


/**
 * PhenomC parameters for modified ringdown,
 * uses final spin formula of:
 * - Barausse \& Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009,
 * arXiv:0904.2577
 */
UNUSED static BBHPhenomCParams *ComputeIMRPhenomCParamsRDmod(
  const REAL8 m1,   /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,   /**< Mass of companion 2 (solar masses) */
  const REAL8 chi,  /**< Reduced aligned spin of the binary chi = (m1*chi1 + m2*chi2)/M */
  const REAL8 chip) /**< Dimensionless spin in the orbital plane */
{

  BBHPhenomCParams *p = NULL;
  p = ComputeIMRPhenomCParams(m1, m2, chi); /* populate parameters with the original PhenomC setup */
  if( !p )
    XLAL_ERROR_NULL(XLAL_EFUNC);

  const REAL8 M = m1 + m2;
  const REAL8 eta = m1 * m2 / (M * M);

  REAL8 finspin = FinalSpinBarausse2009_all_spin_on_larger_BH(eta, chi, chip);
  if( fabs(finspin) > 1.0 ) {
    XLAL_PRINT_WARNING("Warning: final spin magnitude %g > 1. Setting final spin magnitude = 1.", finspin);
    finspin = copysign(1.0, finspin);
  }

  p->afin = finspin;

  /* Get the Ringdown frequency */
  REAL8 prefac = (1./(2.*LAL_PI)) * LAL_C_SI * LAL_C_SI * LAL_C_SI / (LAL_G_SI * M * LAL_MSUN_SI);
  REAL8 k1 = 1.5251;
  REAL8 k2 = -1.1568;
  REAL8 k3 = 0.1292;

  p->fRingDown = (prefac * (k1 + k2 * pow(1. - fabs(finspin), k3)));
  p->MfRingDown = p->m_sec * p->fRingDown;

  /* Get the quality factor of ring-down, using Eq (5.6) of Main paper (arxiv.org/pdf/1005.3306v3.pdf) */
  p->Qual = (0.7000 + (1.4187 * pow(1.0 - fabs(finspin), -0.4990)) );

  /* Get the transition frequencies, at which the model switches phase and
   * amplitude prescriptions, as used in Eq.(5.9), (5.13) of the Main paper */
  p->f1 = 0.1 * p->fRingDown;
  p->f2 = p->fRingDown;
  p->f0 = 0.98 * p->fRingDown;
  p->d1 = 0.005;
  p->d2 = 0.005;
  p->d0 = 0.015;

  /* Get the coefficients beta1, beta2, defined in Eq 5.7 of the main paper */
  REAL8 Mfrd = p->MfRingDown;

  p->b2 = ((-5./3.)* p->a1 * pow(Mfrd,(-8./3.)) - p->a2/(Mfrd*Mfrd) -
      (p->a3/3.)*pow(Mfrd,(-4./3.)) + (2./3.)* p->a5 * pow(Mfrd,(-1./3.)) + p->a6)/eta;

  REAL8 psiPMrd = IMRPhenomCGeneratePhasePM( p->fRingDown, eta, p );

  p->b1 = psiPMrd - (p->b2 * Mfrd);

  /* Taking the upper cut-off frequency as 0.15M */
  /*p->fCut = (1.7086 * eta * eta - 0.26592 * eta + 0.28236) / p->piM;*/
  p->fCut = 0.15 / p->m_sec;

  return p;
}

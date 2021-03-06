#ifndef _LALSIM_IMR_PHENOMP_H
#define _LALSIM_IMR_PHENOMP_H

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

#include <lal/LALSimulationConfig.h>
#include <lal/LALStdlib.h>
#include <lal/LALSimIMR.h>
#include <lal/LALConstants.h>

#include <lal/LALSimIMRPhenomC_internals.h>
#include <lal/LALSimIMRPhenomD_internals.h>

#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>


/* CONSTANTS */

/**
 * Tolerance used below which numbers are treated as zero for the calculation of atan2
 */
#define MAX_TOL_ATAN 1.0e-15


typedef struct tagNNLOanglecoeffs {
    REAL8 alphacoeff1; /* Coefficient of omega^(-1)   in alphaNNLO */
    REAL8 alphacoeff2; /* Coefficient of omega^(-2/3) in alphaNNLO */
    REAL8 alphacoeff3; /* Coefficient of omega^(-1/3) in alphaNNLO */
    REAL8 alphacoeff4; /* Coefficient of log(omega)   in alphaNNLO */
    REAL8 alphacoeff5; /* Coefficient of omega^(1/3)  in alphaNNLO */

    REAL8 epsiloncoeff1; /* Coefficient of omega^(-1)   in epsilonNNLO */
    REAL8 epsiloncoeff2; /* Coefficient of omega^(-2/3) in epsilonNNLO */
    REAL8 epsiloncoeff3; /* Coefficient of omega^(-1/3) in epsilonNNLO */
    REAL8 epsiloncoeff4; /* Coefficient of log(omega)   in epsilonNNLO */
    REAL8 epsiloncoeff5; /* Coefficient of omega^(1/3)  in epsilonNNLO */
} NNLOanglecoeffs;

typedef struct tagSpinWeightedSphericalHarmonic_l2 {
  COMPLEX16 Y2m2, Y2m1, Y20, Y21, Y22;
} SpinWeightedSphericalHarmonic_l2;

#ifdef __cplusplus
extern "C" {
#endif

/* PhenomC parameters for modified ringdown: Uses final spin formula of Barausse & Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009 */
BBHPhenomCParams *ComputeIMRPhenomCParamsRDmod(
  const REAL8 m1,   /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,   /**< Mass of companion 2 (solar masses) */
  const REAL8 chi,  /**< Reduced aligned spin of the binary chi = (m1*chi1 + m2*chi2)/M */
  const REAL8 chip,  /**< Dimensionless spin in the orbital plane */
  LALDict *extraParams /**< linked list containing the extra testing GR parameters */
);

void ComputeNNLOanglecoeffs(
  NNLOanglecoeffs *angcoeffs,  /**< Output: Structure to store results */
  const REAL8 q,               /**< Mass-ratio (convention q>1) */
  const REAL8 chil,            /**< Dimensionless aligned spin of the largest BH */
  const REAL8 chip             /**< Dimensionless spin component in the orbital plane */
);

/* Internal core function to calculate PhenomP polarizations for a sequence of frequences. */
int PhenomPCore(
  COMPLEX16FrequencySeries **hptilde,   /**< Output: Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,   /**< Output: Frequency-domain waveform hx */
  const REAL8 chi1_l_in,                /**< Dimensionless aligned spin on companion 1 */
  const REAL8 chi2_l_in,                /**< Dimensionless aligned spin on companion 2 */
  const REAL8 chip,                     /**< Effective spin in the orbital plane */
  const REAL8 thetaJ,                   /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 m1_SI_in,                 /**< Mass of companion 1 (kg) */
  const REAL8 m2_SI_in,                 /**< Mass of companion 2 (kg) */
  const REAL8 distance,                 /**< Distance of source (m) */
  const REAL8 alpha0,                   /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                     /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 f_ref,                    /**< Reference frequency */
  const REAL8Sequence *freqs,           /**< Frequency points at which to evaluate the waveform (Hz) */
  double deltaF,                        /**< Sampling frequency (Hz).
   * If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
  IMRPhenomP_version_type IMRPhenomP_version, /**< IMRPhenomPv1 uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD */
  LALDict *extraParams,                 /**< linked list containing the extra testing GR parameters */
  const void *buf                       /**< A pointer for passing preallocated buffers; useful for GPU runs (set to NULL otherwise) */
);

/* ************************** PhenomP internal function prototypes *****************************/
/* atan2 wrapper that returns 0 when both magnitudes of x and y are below tol, otherwise it returns
   atan2(x, y) */
REAL8 atan2tol(REAL8 x, REAL8 y, REAL8 tol);

void CheckMaxOpeningAngle(
  const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
  const REAL8 chi1_l, /**< Aligned spin of BH 1 */
  const REAL8 chi2_l, /**< Aligned spin of BH 2 */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
);

REAL8 FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(
  const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
  const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
  const REAL8 chi1_l, /**< Aligned spin of BH 1 */
  const REAL8 chi2_l, /**< Aligned spin of BH 2 */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
);

REAL8 FinalSpinBarausse2009_all_spin_on_larger_BH(
  const REAL8 nu,     /**< Symmetric mass-ratio */
  const REAL8 chi,    /**< Effective aligned spin of the binary:  chi = (m1*chi1 + m2*chi2)/M  */
  const REAL8 chip    /**< Dimensionless spin in the orbital plane */
);

REAL8 FinalSpinBarausse2009(  /* Barausse & Rezzolla, Astrophys.J.Lett.704:L40-L44, 2009, arXiv:0904.2577 */
  const REAL8 nu,               /**< Symmetric mass-ratio */
  const REAL8 a1,               /**< |a_1| norm of dimensionless spin vector for BH 1 */
  const REAL8 a2,               /**< |a_2| norm of dimensionless spin vector for BH 2 */
  const REAL8 cos_alpha,        /**< cos(alpha) = \\hat a_1 . \\hat a_2 (Eq. 7) */
  const REAL8 cos_beta_tilde,   /**< cos(\\tilde beta)  = \\hat a_1 . \\hat L (Eq. 9) */
  const REAL8 cos_gamma_tilde   /**< cos(\\tilde gamma) = \\hat a_2 . \\hat L (Eq. 9)*/
);

#ifdef __cplusplus
}
#endif

#if !defined(LALSIMULATION_CUDA_ENABLED)
void PhenomPCoreAllFrequencies_cpu(UINT4 L_fCut,
        REAL8Sequence *freqs,
        UINT4 offset,
        const REAL8 eta,
        const REAL8 chi1_l,
        const REAL8 chi2_l,
        const REAL8 chip,
        const REAL8 distance,
        const REAL8 M,
        const REAL8 phic,
        IMRPhenomDAmplitudeCoefficients *pAmp,
        IMRPhenomDPhaseCoefficients *pPhi,
        BBHPhenomCParams *PCparams,
        PNPhasingSeries *pn,
        NNLOanglecoeffs *angcoeffs,
        SpinWeightedSphericalHarmonic_l2 *Y2m,
        const REAL8 alphaNNLOoffset,
        const REAL8 alpha0,
        const REAL8 epsilonNNLOoffset,
        IMRPhenomP_version_type IMRPhenomP_version,
        AmpInsPrefactors *amp_prefactors,
        PhiInsPrefactors *phi_prefactors,
        COMPLEX16FrequencySeries *hptilde,
        COMPLEX16FrequencySeries *hctilde,
        REAL8 *phis,
        const void *buf,
        int   *errcode);
#endif

#endif	// of #ifndef _LALSIM_IMR_PHENOMP_H

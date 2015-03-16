/*
 * Copyright (C) 2011 N. Fotopoulos <nickolas.fotopoulos@ligo.org>
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

#ifndef _LALSIMIMR_H
#define _LALSIMIMR_H

#include <lal/LALDatatypes.h>
#include <lal/LALSimInspiral.h>

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

/**
 * The number of e-folds of ringdown which should be attached for
 * EOBNR models
 */
#define EOB_RD_EFOLDS 10.0

/**
 * Driver routine to compute the non-spinning, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomA in the frequency domain.
 *
 * Reference:
 * - Waveform: Eq.(4.13) and (4.16) of http://arxiv.org/pdf/0710.2335
 * - Coefficients: Eq.(4.18) of http://arxiv.org/pdf/0710.2335 and
 * Table I of http://arxiv.org/pdf/0712.0343
 *
 * All input parameters should be SI units.
 */
int XLALSimIMRPhenomAGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    const REAL8 phiPeak,               /**< orbital phase at peak (rad) */
    const REAL8 deltaF,                /**< sampling interval (Hz) */
    const REAL8 m1_SI,                 /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,                 /**< mass of companion 2 (kg) */
    const REAL8 f_min,                 /**< starting GW frequency (Hz) */
    const REAL8 f_max,                 /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance               /**< distance of source (m) */
);

/**
 * Driver routine to compute the non-spinning, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomA in the time domain.
 *
 * Reference:
 * - Waveform: Eq.(4.13) and (4.16) of http://arxiv.org/pdf/0710.2335
 * - Coefficients: Eq.(4.18) of http://arxiv.org/pdf/0710.2335 and
 * Table I of http://arxiv.org/pdf/0712.0343
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomAGenerateTD(
    REAL8TimeSeries **hplus,  /**< +-polarization waveform */
    REAL8TimeSeries **hcross, /**< x-polarization waveform */
    const REAL8 phiPeak,      /**< orbital phase at peak (rad) */
    const REAL8 deltaT,       /**< sampling interval (s) */
    const REAL8 m1_SI,        /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,        /**< mass of companion 2 (kg) */
    const REAL8 f_min,        /**< starting GW frequency (Hz) */
    const REAL8 f_max,        /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance,     /**< distance of source (m) */
    const REAL8 inclination   /**< inclination of source (rad) */
);

/**
 * Compute the dimensionless, spin-aligned parameter chi as used in the
 * IMRPhenomB waveform. This is different from chi in SpinTaylorRedSpin!
 * Reference: http://arxiv.org/pdf/0909.2867, paragraph 3.
 */
double XLALSimIMRPhenomBComputeChi(
    const REAL8 m1,                          /**< mass of companion 1 */
    const REAL8 m2,                          /**< mass of companion 2 */
    const REAL8 s1z,                         /**< spin of companion 1 */
    const REAL8 s2z                          /**< spin of companion 2 */
);

/**
 * Compute the default final frequency 
 */
double XLALSimIMRPhenomAGetFinalFreq(
    const REAL8 m1,
    const REAL8 m2
);

double XLALSimIMRPhenomBGetFinalFreq(
    const REAL8 m1,
    const REAL8 m2,
    const REAL8 chi
);

double XLALSimIMRPhenomCGetFinalFreq(
    const REAL8 m1,
    const REAL8 m2,
    const REAL8 chi
);

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomB in the frequency domain.
 *
 * Reference: http://arxiv.org/pdf/0909.2867
 * - Waveform: Eq.(1)
 * - Coefficients: Eq.(2) and Table I
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomBGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    const REAL8 phiPeak,               /**< orbital phase at peak (rad) */
    const REAL8 deltaF,                /**< sampling interval (Hz) */
    const REAL8 m1_SI,                 /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,                 /**< mass of companion 2 (kg) */
    const REAL8 chi,                   /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,                 /**< starting GW frequency (Hz) */
    const REAL8 f_max,                 /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance               /**< distance of source (m) */
);

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomB in the time domain.
 *
 * Reference: http://arxiv.org/pdf/0909.2867
 * - Waveform: Eq.(1)
 * - Coefficients: Eq.(2) and Table I
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomBGenerateTD(
    REAL8TimeSeries **hplus,  /**< +-polarization waveform */
    REAL8TimeSeries **hcross, /**< x-polarization waveform */
    const REAL8 phiPeak,      /**< orbital phase at peak (rad) */
    const REAL8 deltaT,       /**< sampling interval (s) */
    const REAL8 m1_SI,        /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,        /**< mass of companion 2 (kg) */
    const REAL8 chi,          /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,        /**< starting GW frequency (Hz) */
    const REAL8 f_max,        /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance,     /**< distance of source (m) */
    const REAL8 inclination   /**< inclination of source (rad) */
);

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomC in the frequency domain.
 *
 * Reference: http://arxiv.org/abs/1005.3306
 * - Waveform: Eq.(5.1) - (5.13)
 * - Coefficients: Eq.(5.14) and Table II
 *
 * All input parameters should be in SI units. Angles should be in radians.
 */
int XLALSimIMRPhenomCGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< FD waveform */
    const REAL8 phiPeak,               /**< orbital phase at peak (rad) */
    const REAL8 deltaF,                /**< sampling interval (Hz) */
    const REAL8 m1_SI,                 /**< mass of companion 1 (kg) */
    const REAL8 m2_SI,                 /**< mass of companion 2 (kg) */
    const REAL8 chi,                   /**< mass-weighted aligned-spin parameter */
    const REAL8 f_min,                 /**< starting GW frequency (Hz) */
    const REAL8 f_max,                 /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance               /**< distance of source (m) */
);

/**
 * Driver routine to compute the plus and cross polarizations for the IMRPhenomP model
 * for precessing binaries in the frequency domain.
 *
 * This function takes effective model parameters that should be computed with
 * XLALSimIMRPhenomPCalculateModelParameters().
 *
 * Reference: http://arxiv.org/abs/1308.3271
 *
 */

int XLALSimIMRPhenomP(
  COMPLEX16FrequencySeries **hptilde,   /**< Output: Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,   /**< Output: Frequency-domain waveform hx */
  const REAL8 chi_eff,                  /**< Effective aligned spin */
  const REAL8 chip,                     /**< Effective spin in the orbital plane */
  const REAL8 eta,                      /**< Symmetric mass-ratio */
  const REAL8 thetaJ,                   /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 Mtot_SI,                  /**< Total mass of binary (kg) */
  const REAL8 distance,                 /**< Distance of source (m) */
  const REAL8 alpha0,                   /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                     /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 deltaF,                   /**< Sampling frequency (Hz) */
  const REAL8 f_min,                    /**< Starting GW frequency (Hz) */
  const REAL8 f_max,                   	/**< End frequency; 0 defaults to ringdown cutoff freq */
  const REAL8 f_ref                     /**< Reference frequency */
);

/**
 * Driver routine to compute the plus and cross polarizations for the IMRPhenomP model
 * for precessing binaries in the frequency domain for a specified sequence of frequencies.
 *
 * This function takes effective model parameters that should be computed with
 * XLALSimIMRPhenomPCalculateModelParameters().
 *
 * Reference: http://arxiv.org/abs/1308.3271
 *
 */

int XLALSimIMRPhenomPFrequencySequence(
  COMPLEX16FrequencySeries **hptilde,   /**< Output: Frequency-domain waveform h+ */
  COMPLEX16FrequencySeries **hctilde,   /**< Output: Frequency-domain waveform hx */
  const REAL8Sequence *freqs,           /**< Frequency points at which to evaluate the waveform (Hz) */
  const REAL8 chi_eff,                  /**< Effective aligned spin */
  const REAL8 chip,                     /**< Effective spin in the orbital plane */
  const REAL8 eta,                      /**< Symmetric mass-ratio */
  const REAL8 thetaJ,                   /**< Angle between J0 and line of sight (z-direction) */
  const REAL8 Mtot_SI,                  /**< Total mass of binary (kg) */
  const REAL8 distance,                 /**< Distance of source (m) */
  const REAL8 alpha0,                   /**< Initial value of alpha angle (azimuthal precession angle) */
  const REAL8 phic,                     /**< Orbital phase at the peak of the underlying non precessing model (rad) */
  const REAL8 f_ref                     /**< Reference frequency */
);


/**
 * Function that transforms from the LAL frame to model effective parameters for the IMRPhenomP model.
 *
 * Reference: http://arxiv.org/abs/1308.3271
 *
 */

int XLALSimIMRPhenomPCalculateModelParameters(
    REAL8 *chi_eff,                 /**< Output: Effective aligned spin */
    REAL8 *chip,                    /**< Output: Effective spin in the orbital plane */
    REAL8 *eta,                     /**< Output: Symmetric mass-ratio */
    REAL8 *thetaJ,                  /**< Output: Angle between J0 and line of sight (z-direction) */
    REAL8 *alpha0,                  /**< Output: Initial value of alpha angle (azimuthal precession angle) */
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
    const REAL8 s2z                	/**< Initial value of s2z: dimensionless spin of BH 2 */
);

/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomC in the time domain.
 *
 * Reference: http://arxiv.org/abs/1005.3306
 *   - Waveform: Eq.(5.1) - (5.13)
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
    const REAL8 f_max,        /**< end frequency; 0 defaults to ringdown cutoff freq */
    const REAL8 distance,     /**< distance of source (m) */
    const REAL8 inclination   /**< inclination of source (rad) */
);

/**
 * This function generates the plus and cross polarizations for the dominant
 * (2,2) mode of the EOBNRv2 approximant. This model is defined in Pan et al,
 * arXiv:1106.1021v1 [gr-qc].
 */
int XLALSimIMREOBNRv2DominantMode(
    REAL8TimeSeries **hplus,      /**<< The +-polarization waveform (returned) */
    REAL8TimeSeries **hcross,     /**<< The x-polarization waveform (returned) */
    const REAL8       phiC,       /**<< The phase at the coalescence time */
    const REAL8       deltaT,     /**<< Sampling interval (in seconds) */
    const REAL8       m1SI,       /**<< First component mass (in kg) */
    const REAL8       m2SI,       /**<< Second component mass (in kg) */
    const REAL8       fLower,     /**<< Starting frequency (in Hz) */
    const REAL8       distance,   /**<< Distance to source (in metres) */
    const REAL8       inclination /**<< Inclination of the source (in radians) */
);

/**
 * This function generates the plus and cross polarizations for the EOBNRv2 approximant
 * with all available modes included. This model is defined in Pan et al,
 * arXiv:1106.1021v1 [gr-qc].
 */
int XLALSimIMREOBNRv2AllModes(
    REAL8TimeSeries **hplus,      /**<< The +-polarization waveform (returned) */
    REAL8TimeSeries **hcross,     /**<< The x-polarization waveform (returned) */
    const REAL8       phiC,       /**<< The phase at the time of peak amplitude */
    const REAL8       deltaT,     /**<< Sampling interval (in seconds) */
    const REAL8       m1SI,       /**<< First component mass (in kg) */
    const REAL8       m2SI,       /**<< Second component mass (in kg) */
    const REAL8       fLower,     /**<< Starting frequency (in Hz) */
    const REAL8       distance,   /**<< Distance to source (in metres) */
    const REAL8       inclination /**<< Inclination of the source (in radians) */
);

SphHarmTimeSeries *XLALSimIMREOBNRv2Modes(
        const REAL8 phiRef,  /**< Orbital phase at coalescence (radians) */
        const REAL8 deltaT,  /**< Sampling interval (s) */
        const REAL8 m1,      /**< First component mass (kg) */
        const REAL8 m2,      /**< Second component mass (kg) */
        const REAL8 fLower,  /**< Starting GW frequency (Hz) */
        const REAL8 distance /**< Distance to sources (m) */
        );

double XLALSimIMRSpinAlignedEOBPeakFrequency( 
    REAL8 m1SI,            /**< mass of companion 1 (kg) */
    REAL8 m2SI,            /**< mass of companion 2 (kg) */
    const REAL8 spin1z,    /**< z-component of the dimensionless spin of object 1 */
    const REAL8 spin2z,    /**< z-component of the dimensionless spin of object 2 */
    UINT4 SpinAlignedEOBversion   /**< 1 for SEOBNRv1, 2 for SEOBNRv2 */
    );

int XLALSimIMRSpinAlignedEOBWaveform(
        REAL8TimeSeries **hplus,
        REAL8TimeSeries **hcross,
        const REAL8     phiC,
        REAL8           deltaT,
        const REAL8     m1SI,
        const REAL8     m2SI,
        const REAL8     fMin,
        const REAL8     r,
        const REAL8     inc,
        const REAL8     spin1z,
        const REAL8     spin2z,
        UINT4           SpinAlignedEOBversion
     );

int XLALSimIMRSpinEOBWaveform(
        REAL8TimeSeries **hplus,
        REAL8TimeSeries **hcross,
        //LIGOTimeGPS     *tc,
        const REAL8     phiC,
        const REAL8     deltaT,
        const REAL8     m1SI,
        const REAL8     m2SI,
        const REAL8     fMin,
        const REAL8     r,
        const REAL8     inc,
        //const REAL8     spin1z,
        //const REAL8     spin2z,
        //UINT4           SpinAlignedEOBversion
        const REAL8     spin1[],
        const REAL8     spin2[]
     );

/*
 * SEOBNRv1 reduced order models
 * See CQG 31 195010, 2014, arXiv:1402.4146 for details.
 */

int XLALSimIMRSEOBNRv1ROMSingleSpin(
    struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
    struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
    REAL8 phiRef,                                 /**< Phase at reference frequency */
    REAL8 deltaF,                                 /**< Sampling frequency (Hz) */
    REAL8 fLow,                                   /**< Starting GW frequency (Hz) */
    REAL8 fHigh,                                  /**< End frequency; 0 defaults to ringdown cutoff freq */
    REAL8 fRef,                                   /**< Reference frequency; 0 defaults to fLow */
    REAL8 distance,                               /**< Distance of source (m) */
    REAL8 inclination,                            /**< Inclination of source (rad) */
    REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
    REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
    REAL8 chi                                     /**< Effective aligned spin */
);

/** Compute waveform in LAL format at specified frequencies */
int XLALSimIMRSEOBNRv1ROMSingleSpinFrequencySequence(
    struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
    struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
    const REAL8Sequence *freqs,                   /**< Frequency points at which to evaluate the waveform (Hz) */
    REAL8 phiRef,                                 /**< Phase at reference frequency */
    REAL8 fRef,                                   /**< Reference frequency; 0 defaults to fLow */
    REAL8 distance,                               /**< Distance of source (m) */
    REAL8 inclination,                            /**< Inclination of source (rad) */
    REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
    REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
    REAL8 chi                                     /**< Effective aligned spin */
);

int XLALSimIMRSEOBNRv1ROMDoubleSpin(
    struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
    struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
    REAL8 phiRef,                                 /**< Phase at reference frequency */
    REAL8 deltaF,                                 /**< Sampling frequency (Hz) */
    REAL8 fLow,                                   /**< Starting GW frequency (Hz) */
    REAL8 fHigh,                                  /**< End frequency; 0 defaults to ringdown cutoff freq */
    REAL8 fRef,                                   /**< Reference frequency; 0 defaults to fLow */
    REAL8 distance,                               /**< Distance of source (m) */
    REAL8 inclination,                            /**< Inclination of source (rad) */
    REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
    REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
    REAL8 chi1,                                   /**< Dimensionless aligned component spin 1 */
    REAL8 chi2                                    /**< Dimensionless aligned component spin 2 */
);

/** Compute waveform in LAL format at specified frequencies */
int XLALSimIMRSEOBNRv1ROMDoubleSpinFrequencySequence(
    struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
    struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
    const REAL8Sequence *freqs,                   /**< Frequency points at which to evaluate the waveform (Hz) */
    REAL8 phiRef,                                 /**< Phase at reference frequency */
    REAL8 fRef,                                   /**< Reference frequency; 0 defaults to fLow */
    REAL8 distance,                               /**< Distance of source (m) */
    REAL8 inclination,                            /**< Inclination of source (rad) */
    REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
    REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
    REAL8 chi1,                                   /**< Dimensionless aligned component spin 1 */
    REAL8 chi2                                    /**< Dimensionless aligned component spin 2 */
);


/*
 * SEOBNRv2 reduced order models PRELIMINARY!
 * See CQG 31 195010, 2014, arXiv:1402.4146 for details.
 */

int XLALSimIMRSEOBNRv2ROMSingleSpin(
    struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
    struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
    REAL8 phiRef,                                 /**< Phase at reference frequency */
    REAL8 deltaF,                                 /**< Sampling frequency (Hz) */
    REAL8 fLow,                                   /**< Starting GW frequency (Hz) */
    REAL8 fHigh,                                  /**< End frequency; 0 defaults to ringdown cutoff freq */
    REAL8 fRef,                                   /**< Reference frequency; 0 defaults to fLow */
    REAL8 distance,                               /**< Distance of source (m) */
    REAL8 inclination,                            /**< Inclination of source (rad) */
    REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
    REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
    REAL8 chi                                     /**< Effective aligned spin */
);

/** Compute waveform in LAL format at specified frequencies */
int XLALSimIMRSEOBNRv2ROMSingleSpinFrequencySequence(
  struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
  struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
  const REAL8Sequence *freqs,                   /**< Frequency points at which to evaluate the waveform (Hz) */
  REAL8 phiRef,                                 /**< Phase at reference time */
  REAL8 fRef,                                   /**< Reference frequency (Hz); 0 defaults to fLow */
  REAL8 distance,                               /**< Distance of source (m) */
  REAL8 inclination,                            /**< Inclination of source (rad) */
  REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
  REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
  REAL8 chi                                     /**< Effective aligned spin */
);

/**
 * Compute the time at a given frequency. The origin of time is at the merger.
 * The allowed frequency range for the input is Mf \in [0.0001, 0.3].
 */
int XLALSimIMRSEOBNRv2ROMSingleSpinTimeOfFrequency(
  REAL8 *t,         /**< Output: time (s) at frequency */
  REAL8 frequency,  /**< Frequency (Hz) */
  REAL8 m1SI,       /**< Mass of companion 1 (kg) */
  REAL8 m2SI,       /**< Mass of companion 2 (kg) */
  REAL8 chi         /**< Effective aligned spin */
);

/**
 * Compute the frequency at a given time. The origin of time is at the merger.
 * The frequency range for the output is Mf \in [0.0001, 0.3].
 */
int XLALSimIMRSEOBNRv2ROMSingleSpinFrequencyOfTime(
  REAL8 *frequency,   /**< Output: Frequency (Hz) */
  REAL8 t,            /**< Time (s) at frequency */
  REAL8 m1SI,         /**< Mass of companion 1 (kg) */
  REAL8 m2SI,         /**< Mass of companion 2 (kg) */
  REAL8 chi           /**< Effective aligned spin */
);

int XLALSimIMRSEOBNRv2ROMDoubleSpin(
    struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
    struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
    REAL8 phiRef,                                 /**< Phase at reference frequency */
    REAL8 deltaF,                                 /**< Sampling frequency (Hz) */
    REAL8 fLow,                                   /**< Starting GW frequency (Hz) */
    REAL8 fHigh,                                  /**< End frequency; 0 defaults to ringdown cutoff freq */
    REAL8 fRef,                                   /**< Reference frequency; 0 defaults to fLow */
    REAL8 distance,                               /**< Distance of source (m) */
    REAL8 inclination,                            /**< Inclination of source (rad) */
    REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
    REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
    REAL8 chi1,                                   /**< Dimensionless aligned component spin 1 */
    REAL8 chi2                                    /**< Dimensionless aligned component spin 2 */
);

/** Compute waveform in LAL format at specified frequencies */
int XLALSimIMRSEOBNRv2ROMDoubleSpinFrequencySequence(
  struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
  struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
  const REAL8Sequence *freqs,                   /**< Frequency points at which to evaluate the waveform (Hz) */
  REAL8 phiRef,                                 /**< Phase at reference time */
  REAL8 fRef,                                   /**< Reference frequency (Hz); 0 defaults to fLow */
  REAL8 distance,                               /**< Distance of source (m) */
  REAL8 inclination,                            /**< Inclination of source (rad) */
  REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
  REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
  REAL8 chi1,                                   /**< Dimensionless aligned component spin 1 */
  REAL8 chi2                                    /**< Dimensionless aligned component spin 2 */
);

 /**
  * Compute the time at a given frequency. The origin of time is at the merger.
  * The allowed frequency range for the input is from Mf = 0.00053 to half the ringdown frequency.
  */
 int XLALSimIMRSEOBNRv2ROMDoubleSpinTimeOfFrequency(
   REAL8 *t,         /**< Output: time (s) at frequency */
   REAL8 frequency,  /**< Frequency (Hz) */
   REAL8 m1SI,       /**< Mass of companion 1 (kg) */
   REAL8 m2SI,       /**< Mass of companion 2 (kg) */
   REAL8 chi1,       /**< Dimensionless aligned component spin 1 */
   REAL8 chi2        /**< Dimensionless aligned component spin 2 */
 );

 /**
  * Compute the frequency at a given time. The origin of time is at the merger.
  * The frequency range for the output is from Mf = 0.00053 to half the ringdown frequency.
  */
 int XLALSimIMRSEOBNRv2ROMDoubleSpinFrequencyOfTime(
   REAL8 *frequency,   /**< Output: Frequency (Hz) */
   REAL8 t,            /**< Time (s) at frequency */
   REAL8 m1SI,         /**< Mass of companion 1 (kg) */
   REAL8 m2SI,         /**< Mass of companion 2 (kg) */
   REAL8 chi1,         /**< Dimensionless aligned component spin 1 */
   REAL8 chi2          /**< Dimensionless aligned component spin 2 */
 );


/**
 * Compute SEOBNRv2 chirp time from interpolant assuming a single-spin.
 */
REAL8 XLALSimIMRSEOBNRv2ChirpTimeSingleSpin(
  const REAL8 m1_SI,    /**< Mass of companion 1 [kg] */
  const REAL8 m2_SI,    /**< Mass of companion 2 [kg] */
  const REAL8 chi,      /**< Effective aligned spin */
  const REAL8 f_min     /**< Starting frequency [Hz] */
);

  
/**
 * Routine to compute the mass and spin of the final black hole given
 * the masses, spins, binding energy, and orbital angular momentum vector.
 */
int XLALSimIMRPhenSpinFinalMassSpin(REAL8 *finalMass,
				    REAL8 *finalSpin,
				    REAL8 m1,
				    REAL8 m2,
				    REAL8 s1s1,
				    REAL8 s2s2,
				    REAL8 s1L,
				    REAL8 s2L,
				    REAL8 s1s2,
				    REAL8 energy);

int XLALSimSpinInspiralGenerator(REAL8TimeSeries **hPlus,	        /**< +-polarization waveform [returned] */
				 REAL8TimeSeries **hCross,	        /**< x-polarization waveform [returned] */
				 REAL8 phi_start,                       /**< start phase */
				 REAL8 deltaT,                          /**< sampling interval */
				 REAL8 m1,                              /**< mass of companion 1 */
				 REAL8 m2,                              /**< mass of companion 2 */
				 REAL8 f_min,                           /**< start frequency */
				 REAL8 f_ref,                           /**< reference frequency */
				 REAL8 r,                               /**< distance of source */
				 REAL8 iota,                            /**< inclination of source (rad) */
				 REAL8 s1x,                             /**< x-component of dimensionless spin for object 1 */
				 REAL8 s1y,                             /**< y-component of dimensionless spin for object 1 */
				 REAL8 s1z,                             /**< z-component of dimensionless spin for object 1 */
				 REAL8 s2x,                             /**< x-component of dimensionless spin for object 2 */
				 REAL8 s2y,                             /**< y-component of dimensionless spin for object 2 */
				 REAL8 s2z,                             /**< z-component of dimensionless spin for object 2 */
				 int phaseO,                            /**< twice post-Newtonian phase order */
				 int ampO,                              /**< twice post-Newtonian amplitude order */
				 LALSimInspiralWaveformFlags *waveFlags,/**< Choice of axis for input spin params */
				 LALSimInspiralTestGRParam *testGRparams/**< Choice of axis for input spin params */
				 );

/**
 * Driver routine to compute a precessing post-Newtonian inspiral-merger-ringdown waveform
 */

int XLALSimIMRPhenSpinInspiralRDGenerator(
    REAL8TimeSeries **hplus,    /**< +-polarization waveform */
    REAL8TimeSeries **hcross,   /**< x-polarization waveform */
    REAL8 phi0,                 /**< phase at time of peak amplitude*/
    REAL8 deltaT,               /**< sampling interval */
    REAL8 m1,                   /**< mass of companion 1 */
    REAL8 m2,                   /**< mass of companion 2 */
    REAL8 f_min,                /**< start frequency */
    REAL8 f_ref,                /**< reference frequency */
    REAL8 r,                    /**< distance of source */
    REAL8 iota,                 /**< inclination of source (rad) */
    REAL8 s1x,                  /**< x-component of dimensionless spin for object 1 */
    REAL8 s1y,                  /**< y-component of dimensionless spin for object 1 */
    REAL8 s1z,                  /**< z-component of dimensionless spin for object 1 */
    REAL8 s2x,                  /**< x-component of dimensionless spin for object 2 */
    REAL8 s2y,                  /**< y-component of dimensionless spin for object 2 */
    REAL8 s2z,                  /**< z-component of dimensionless spin for object 2 */
    int phaseO,                 /**< twice post-Newtonian phase order */
    int ampO,                   /**< twice post-Newtonian amplitude order */
    LALSimInspiralWaveformFlags *waveFlag,/**< Choice of axis for input spin params */
    LALSimInspiralTestGRParam *testGRparam  /**< Choice of axis for input spin params */
					  );

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _LALSIMIMR_H */

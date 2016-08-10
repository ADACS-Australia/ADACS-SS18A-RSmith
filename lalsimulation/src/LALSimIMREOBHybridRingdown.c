/*
*  Copyright (C) 2008 Yi Pan, B.S. Sathyaprakash (minor modificaitons), Prayush
*  Kumar (some additions)
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
 * \author Yi Pan, Craig Robinson
 * \file
 *
 * \brief Module to compute the ring-down waveform as linear combination
 * of quasi-normal-modes decaying waveforms, which can be attached to
 * the inspiral part of the compat binary coalescing waveform.
 * The method is describe in Sec. II C of Pan et al. PRD 84, 124052 (2011),
 * specifically Eqs. 30 - 32.
 * Eqs. 30 and 31 are written in explicity linear equation systems in
 * DCC document T1100433.
 * This method is currently used for EOBNRv2 and SEOBNRv1 models. The only difference
 * between the two models in ring-down waveform is the pseudo-QNM introduced
 * in the latter (see Taracchini et al. PRD 86, 024011 (2012) for more details).
 */
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <lal/LALStdlib.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include "LALSimIMREOBNRv2.h"
#include "LALSimBlackHoleRingdown.h"
#include "LALSimIMREOBNQCCorrection.c"

#ifndef _LALSIMIMREOBHYBRIDRINGDOWN_C
#define _LALSIMIMREOBHYBRIDRINGDOWN_C

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

/**
 * Generates the ringdown wave associated with the given real
 * and imaginary parts of the inspiral waveform. The parameters of
 * the ringdown, such as amplitude and phase offsets, are determined
 * by solving the linear equations defined in the DCC document T1100433.
 * In the linear equations Ax=y,
 * A is a 16-by-16 matrix depending on QNM (complex) frequencies,
 * x is a 16-d vector of the 8 unknown complex QNM amplitudes,
 * y is a 16-d vector depending on inspiral-plunge waveforms and their derivatives near merger.
 */
static INT4 XLALSimIMREOBHybridRingdownWave(REAL8Vector * rdwave1,
                                   /**<< OUTPUT, Real part of ringdown waveform */
    REAL8Vector * rdwave2,         /**<< OUTPUT, Imag part of ringdown waveform */
    const REAL8 dt,                /**<< Sampling interval */
    const REAL8 mass1,             /**<< First component mass (in Solar masses) */
    const REAL8 mass2,             /**<< Second component mass (in Solar masses) */
    REAL8VectorSequence * inspwave1,
                                   /**<< Values and derivs of real part inspiral waveform */
    REAL8VectorSequence * inspwave2,
                                   /**<< Values and derivs of imag part inspiral waveform */
    COMPLEX16Vector * modefreqs,   /**<< Complex freqs of ringdown (scaled by total mass) */
    REAL8Vector * matchrange       /**<< Times which determine the comb of ringdown attachment */
    )
{

    /* XLAL error handling */
    INT4 errcode = XLAL_SUCCESS;

    /* For checking GSL return codes */
    INT4 gslStatus;

    UINT4 i, j, k, nmodes = 8;

    /* Sampling rate from input */
    REAL8 t1, t2, t3, t4, t5, rt;
    gsl_matrix *coef;
    gsl_vector *hderivs;
    gsl_vector *x;
    gsl_permutation *p;
    REAL8Vector *modeamps;
    int s;
    REAL8 tj;
    REAL8 m;

    /* mass in geometric units */
    m = (mass1 + mass2) * LAL_MTSUN_SI;
    t5 = (matchrange->data[0] - matchrange->data[1]) * m;
    rt = -t5 / 5.;

    t4 = t5 + rt;
    t3 = t4 + rt;
    t2 = t3 + rt;
    t1 = t2 + rt;

    if (inspwave1->length != 3 || inspwave2->length != 3 || modefreqs->length != nmodes) {
        XLAL_ERROR(XLAL_EBADLEN);
    }

    /* Solving the linear system for QNMs amplitude coefficients using gsl routine */
    /* Initiate matrices and supporting variables */
    XLAL_CALLGSL(coef = (gsl_matrix *) gsl_matrix_alloc(2 * nmodes, 2 * nmodes));
    XLAL_CALLGSL(hderivs = (gsl_vector *) gsl_vector_alloc(2 * nmodes));
    XLAL_CALLGSL(x = (gsl_vector *) gsl_vector_alloc(2 * nmodes));
    XLAL_CALLGSL(p = (gsl_permutation *) gsl_permutation_alloc(2 * nmodes));

    /* Check all matrices and variables were allocated */
    if (!coef || !hderivs || !x || !p) {
        if (coef)
            gsl_matrix_free(coef);
        if (hderivs)
            gsl_vector_free(hderivs);
        if (x)
            gsl_vector_free(x);
        if (p)
            gsl_permutation_free(p);

        XLAL_ERROR(XLAL_ENOMEM);
    }

    /* Define the linear system Ax=y */
    /* Matrix A (2*n by 2*n) has block symmetry. Define half of A here as "coef" */
    /* The half of A defined here corresponds to matrices M1 and -M2 in the DCC document T1100433 */
    /* Define y here as "hderivs" */
    for (i = 0; i < nmodes; ++i) {
        gsl_matrix_set(coef, 0, i, 1);
        gsl_matrix_set(coef, 1, i, -cimag(modefreqs->data[i]));
        gsl_matrix_set(coef, 2, i, exp(-cimag(modefreqs->data[i]) * t1) * cos(creal(modefreqs->data[i]) * t1));
        gsl_matrix_set(coef, 3, i, exp(-cimag(modefreqs->data[i]) * t2) * cos(creal(modefreqs->data[i]) * t2));
        gsl_matrix_set(coef, 4, i, exp(-cimag(modefreqs->data[i]) * t3) * cos(creal(modefreqs->data[i]) * t3));
        gsl_matrix_set(coef, 5, i, exp(-cimag(modefreqs->data[i]) * t4) * cos(creal(modefreqs->data[i]) * t4));
        gsl_matrix_set(coef, 6, i, exp(-cimag(modefreqs->data[i]) * t5) * cos(creal(modefreqs->data[i]) * t5));
        gsl_matrix_set(coef, 7, i, exp(-cimag(modefreqs->data[i]) * t5) * (-cimag(modefreqs->data[i]) * cos(creal(modefreqs->data[i]) * t5)
                - creal(modefreqs->data[i]) * sin(creal(modefreqs->data[i]) * t5)));
        gsl_matrix_set(coef, 8, i, 0);
        gsl_matrix_set(coef, 9, i, -creal(modefreqs->data[i]));
        gsl_matrix_set(coef, 10, i, -exp(-cimag(modefreqs->data[i]) * t1) * sin(creal(modefreqs->data[i]) * t1));
        gsl_matrix_set(coef, 11, i, -exp(-cimag(modefreqs->data[i]) * t2) * sin(creal(modefreqs->data[i]) * t2));
        gsl_matrix_set(coef, 12, i, -exp(-cimag(modefreqs->data[i]) * t3) * sin(creal(modefreqs->data[i]) * t3));
        gsl_matrix_set(coef, 13, i, -exp(-cimag(modefreqs->data[i]) * t4) * sin(creal(modefreqs->data[i]) * t4));
        gsl_matrix_set(coef, 14, i, -exp(-cimag(modefreqs->data[i]) * t5) * sin(creal(modefreqs->data[i]) * t5));
        gsl_matrix_set(coef, 15, i, exp(-cimag(modefreqs->data[i]) * t5) * (cimag(modefreqs->data[i]) * sin(creal(modefreqs->data[i]) * t5)
                - creal(modefreqs->data[i]) * cos(creal(modefreqs->data[i]) * t5)));
    }
    for (i = 0; i < 2; ++i) {
        gsl_vector_set(hderivs, i, inspwave1->data[(i + 1) * inspwave1->vectorLength - 1]);
        gsl_vector_set(hderivs, i + nmodes, inspwave2->data[(i + 1) * inspwave2->vectorLength - 1]);
        gsl_vector_set(hderivs, i + 6, inspwave1->data[i * inspwave1->vectorLength]);
        gsl_vector_set(hderivs, i + 6 + nmodes, inspwave2->data[i * inspwave2->vectorLength]);
    }
    gsl_vector_set(hderivs, 2, inspwave1->data[4]);
    gsl_vector_set(hderivs, 2 + nmodes, inspwave2->data[4]);
    gsl_vector_set(hderivs, 3, inspwave1->data[3]);
    gsl_vector_set(hderivs, 3 + nmodes, inspwave2->data[3]);
    gsl_vector_set(hderivs, 4, inspwave1->data[2]);
    gsl_vector_set(hderivs, 4 + nmodes, inspwave2->data[2]);
    gsl_vector_set(hderivs, 5, inspwave1->data[1]);
    gsl_vector_set(hderivs, 5 + nmodes, inspwave2->data[1]);

    /* Complete the definition for the rest half of A */
    for (i = 0; i < nmodes; ++i) {
        for (k = 0; k < nmodes; ++k) {
            gsl_matrix_set(coef, i, k + nmodes, -gsl_matrix_get(coef, i + nmodes, k));
            gsl_matrix_set(coef, i + nmodes, k + nmodes, gsl_matrix_get(coef, i, k));
        }
    }

#if 0
    /* print ringdown-matching linear system: coefficient matrix and RHS vector */
    printf("\nRingdown matching matrix:\n");
    for (i = 0; i < 16; ++i) {
        for (j = 0; j < 16; ++j) {
            printf("%.12e ", gsl_matrix_get(coef, i, j));
        }
        printf("\n");
    }
    printf("RHS:  ");
    for (i = 0; i < 16; ++i) {
        printf("%.12e   ", gsl_vector_get(hderivs, i));
    }
    printf("\n");
#endif

    /* Call gsl LU decomposition to solve the linear system */
    XLAL_CALLGSL(gslStatus = gsl_linalg_LU_decomp(coef, p, &s));
    if (gslStatus == GSL_SUCCESS) {
        XLAL_CALLGSL(gslStatus = gsl_linalg_LU_solve(coef, p, hderivs, x));
    }
    if (gslStatus != GSL_SUCCESS) {
        gsl_matrix_free(coef);
        gsl_vector_free(hderivs);
        gsl_vector_free(x);
        gsl_permutation_free(p);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /* Putting solution to an XLAL vector */
    modeamps = XLALCreateREAL8Vector(2 * nmodes);

    if (!modeamps) {
        gsl_matrix_free(coef);
        gsl_vector_free(hderivs);
        gsl_vector_free(x);
        gsl_permutation_free(p);
        XLAL_ERROR(XLAL_ENOMEM);
    }

    for (i = 0; i < nmodes; ++i) {
        modeamps->data[i] = gsl_vector_get(x, i);
        modeamps->data[i + nmodes] = gsl_vector_get(x, i + nmodes);
    }

    /* Free all gsl linear algebra objects */
    gsl_matrix_free(coef);
    gsl_vector_free(hderivs);
    gsl_vector_free(x);
    gsl_permutation_free(p);

    /* Build ring-down waveforms */

    REAL8 timeOffset = fmod(matchrange->data[1], dt / m) * dt;

    for (j = 0; j < rdwave1->length; ++j) {
        tj = j * dt - timeOffset;
        rdwave1->data[j] = 0;
        rdwave2->data[j] = 0;
        for (i = 0; i < nmodes; ++i) {
            rdwave1->data[j] += exp(-tj * cimag(modefreqs->data[i]))
                * (modeamps->data[i] * cos(tj * creal(modefreqs->data[i]))
                + modeamps->data[i + nmodes] * sin(tj * creal(modefreqs->data[i])));
            rdwave2->data[j] += exp(-tj * cimag(modefreqs->data[i]))
                * (-modeamps->data[i] * sin(tj * creal(modefreqs->data[i]))
                + modeamps->data[i + nmodes] * cos(tj * creal(modefreqs->data[i])));
        }
    }

    XLALDestroyREAL8Vector(modeamps);
    return errcode;
}

/**
 * Function which calculates the value of the waveform, plus its
 * first and second derivatives, for the points which will be required
 * in the hybrid comb attachment of the ringdown.
 */
static INT4 XLALGenerateHybridWaveDerivatives(REAL8Vector * rwave,
                                     /**<< OUTPUT, values of the waveform at comb points */
    REAL8Vector * dwave,             /**<< OUTPUT, 1st deriv of the waveform at comb points */
    REAL8Vector * ddwave,            /**<< OUTPUT, 2nd deriv of the waveform at comb points */
    REAL8Vector * timeVec,           /**<< Vector containing the time */
    REAL8Vector * wave,              /**<< Last part of inspiral waveform */
    REAL8Vector * matchrange,        /**<< Times which determine the size of the comb */
    REAL8 dt,                        /**<< Sample time step */
    REAL8 mass1,                     /**<< First component mass (in Solar masses) */
    REAL8 mass2                      /**<< Second component mass (in Solar masses) */
    )
{

    /* XLAL error handling */
    INT4 errcode = XLAL_SUCCESS;

    /* For checking GSL return codes */
    INT4 gslStatus;

    UINT4 j;
    UINT4 vecLength;
    REAL8 m;
    double *y;
    double ry, dy, dy2;
    double rt;
    double *tlist;
    gsl_interp_accel *acc;
    gsl_spline *spline;

    /* Total mass in geometric units */
    m = (mass1 + mass2) * LAL_MTSUN_SI;

    tlist = (double *)LALMalloc(6 * sizeof(double));
    rt = (matchrange->data[1] - matchrange->data[0]) / 5.;
    tlist[0] = matchrange->data[0];
    tlist[1] = tlist[0] + rt;
    tlist[2] = tlist[1] + rt;
    tlist[3] = tlist[2] + rt;
    tlist[4] = tlist[3] + rt;
    tlist[5] = matchrange->data[1];

    /* Set the length of the interpolation vectors */
    vecLength = (m * matchrange->data[2] / dt) + 1;

    /* Getting interpolation and derivatives of the waveform using gsl spline routine */
    /* Initiate arrays and supporting variables for gsl */
    y = (double *)LALMalloc(vecLength * sizeof(double));

    if (!y) {
        XLAL_ERROR(XLAL_ENOMEM);
    }

    for (j = 0; j < vecLength; ++j) {
        y[j] = wave->data[j];
    }

    XLAL_CALLGSL(acc = (gsl_interp_accel *) gsl_interp_accel_alloc());
    XLAL_CALLGSL(spline = (gsl_spline *) gsl_spline_alloc(gsl_interp_cspline, vecLength));
    if (!acc || !spline) {
        if (acc)
            gsl_interp_accel_free(acc);
        if (spline)
            gsl_spline_free(spline);
        LALFree(y);
        XLAL_ERROR(XLAL_ENOMEM);
    }

    /* Gall gsl spline interpolation */
    gslStatus = gsl_spline_init(spline, timeVec->data, y, vecLength);
    if (gslStatus != GSL_SUCCESS) {
        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);
        LALFree(y);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /* Getting first and second order time derivatives from gsl interpolations */
    for (j = 0; j < 6; ++j) {
        gslStatus = gsl_spline_eval_e(spline, tlist[j], acc, &ry);
        if (gslStatus == GSL_SUCCESS) {
            gslStatus = gsl_spline_eval_deriv_e(spline, tlist[j], acc, &dy);
            gslStatus = gsl_spline_eval_deriv2_e(spline, tlist[j], acc, &dy2);
        }
        if (gslStatus != GSL_SUCCESS) {
            gsl_spline_free(spline);
            gsl_interp_accel_free(acc);
            LALFree(y);
            XLAL_ERROR(XLAL_EFUNC);
        }
        rwave->data[j] = (REAL8) (ry);
        dwave->data[j] = (REAL8) (dy / m);
        ddwave->data[j] = (REAL8) (dy2 / m / m);

    }

    /* Free gsl variables */
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    LALFree(tlist);
    LALFree(y);

    return errcode;
}

/**
 * The main workhorse function for performing the ringdown attachment for EOB
 * models EOBNRv2 and SEOBNRv1. This is the function which gets called by the
 * code generating the full IMR waveform once generation of the inspiral part
 * has been completed.
 * The ringdown is attached using the hybrid comb matching detailed in
 * The method is describe in Sec. II C of Pan et al. PRD 84, 124052 (2011),
 * specifically Eqs. 30 - 32.. Further details of the
 * implementation of the found in the DCC document T1100433.
 * In SEOBNRv1, the last physical overtone is replace by a pseudoQNM. See
 * Taracchini et al. PRD 86, 024011 (2012) for details.
 * STEP 1) Get mass and spin of the final black hole and the complex ringdown frequencies
 * STEP 2) Based on least-damped-mode decay time, allocate memory for rigndown waveform
 * STEP 3) Get values and derivatives of inspiral waveforms at matching comb points
 * STEP 4) Solve QNM coefficients and generate ringdown waveforms
 * STEP 5) Stitch inspiral and ringdown waveoforms
 */
//static INT4 XLALSimIMREOBHybridAttachRingdown(
static UNUSED INT4 XLALSimIMREOBHybridAttachRingdown(REAL8Vector * signal1,
                           /**<< OUTPUT, Real of inspiral waveform to which we attach ringdown */
    REAL8Vector * signal2, /**<< OUTPUT, Imag of inspiral waveform to which we attach ringdown */
    const INT4 l,          /**<< Current mode l */
    const INT4 m,          /**<< Current mode m */
    const REAL8 dt,        /**<< Sample time step (in seconds) */
    const REAL8 mass1,     /**<< First component mass (in Solar masses) */
    const REAL8 mass2,     /**<< Second component mass (in Solar masses) */
    const REAL8 spin1x,    /**<<The spin of the first object; only needed for spin waveforms */
    const REAL8 spin1y,    /**<<The spin of the first object; only needed for spin waveforms */
    const REAL8 spin1z,    /**<<The spin of the first object; only needed for spin waveforms */
    const REAL8 spin2x,    /**<<The spin of the second object; only needed for spin waveforms */
    const REAL8 spin2y,    /**<<The spin of the second object; only needed for spin waveforms */
    const REAL8 spin2z,    /**<<The spin of the second object; only needed for spin waveforms */
    REAL8Vector * timeVec, /**<< Vector containing the time values */
    REAL8Vector * matchrange,
                           /**<< Time values chosen as points for performing comb matching */
    Approximant approximant/**<<The waveform approximant being used */
    ) {

    COMPLEX16Vector *modefreqs;
    //COMPLEX16       freq7sav;
    UINT4 Nrdwave;
    UINT4 j;

    UINT4 nmodes;
    REAL8Vector *rdwave1;
    REAL8Vector *rdwave2;
    REAL8Vector *rinspwave;
    REAL8Vector *dinspwave;
    REAL8Vector *ddinspwave;
    REAL8VectorSequence *inspwaves1;
    REAL8VectorSequence *inspwaves2;
    REAL8 eta, a, chi, NRPeakOmega22;   /* To generate pQNM frequency */
    REAL8 sh, kk, kt1, kt2;     /* To generate pQNM frequency */
    REAL8 mTot; /* In geometric units */
    REAL8 spin1[3] = { spin1x, spin1y, spin1z };
    REAL8 spin2[3] = { spin2x, spin2y, spin2z };
    REAL8 finalMass, finalSpin;

    mTot = (mass1 + mass2) * LAL_MTSUN_SI;
    eta = mass1 * mass2 / ((mass1 + mass2) * (mass1 + mass2));

    /*
     * STEP 1) Get mass and spin of the final black hole and the complex ringdown frequencies
     */

    /* Create memory for the QNM frequencies */
    nmodes = 8;
    modefreqs = XLALCreateCOMPLEX16Vector(nmodes);
    if (!modefreqs) {
        XLAL_ERROR(XLAL_ENOMEM);
    }

    if (XLALSimIMREOBGenerateQNMFreqV2(modefreqs, mass1, mass2, spin1, spin2, l, m, nmodes, approximant) == XLAL_FAILURE) {
        XLALDestroyCOMPLEX16Vector(modefreqs);
        XLAL_ERROR(XLAL_EFUNC);
    }
    //UINT4 i;
    //for (i=0; i<nmodes; i++){
    //    printf("Stas mode %d: %f + i %f \n", i, creal(modefreqs->data[i]), cimag(modefreqs->data[i]));
    //}

    /* Call XLALSimIMREOBFinalMassSpin() to get mass and spin of the final black hole */
    if (XLALSimIMREOBFinalMassSpin(&finalMass, &finalSpin, mass1, mass2, spin1, spin2, approximant) == XLAL_FAILURE) {
        XLAL_ERROR(XLAL_EFUNC);
    }

    if (approximant == SEOBNRv1) {
        /* Replace the last QNM with pQNM */
        /* We assume aligned/antialigned spins here */
        a = (spin1[2] + spin2[2]) / 2. * (1.0 - 2.0 * eta) + (spin1[2] - spin2[2]) / 2. * (mass1 - mass2) / (mass1 + mass2);
        NRPeakOmega22 = GetNRSpinPeakOmega(l, m, eta, a) / mTot;
        /*printf("a and NRomega in QNM freq: %.16e %.16e %.16e %.16e %.16e\n",spin1[2],spin2[2],
         * mTot/LAL_MTSUN_SI,a,NRPeakOmega22*mTot); */
        modefreqs->data[7] = (NRPeakOmega22 / finalMass + creal(modefreqs->data[0])) / 2.;
        modefreqs->data[7] += I * 10. / 3. * cimag(modefreqs->data[0]);
    }

    if (approximant == SEOBNRv2)        //See pages 6 to 12 of the dcc document T1400476-v3 for expressions in this block.
    {
        /* Replace the last two QNMs with pQNMs */
        /* We assume aligned/antialigned spins here */
        /* Definitions of a, chi and NRPeakOmega22, where the last one is an approximation of \phi'[tmatch] in T1400476-v3. */
        a = (spin1[2] + spin2[2]) / 2. * (1.0 - 2.0 * eta) + (spin1[2] - spin2[2]) / 2. * (mass1 - mass2) / (mass1 + mass2);
        NRPeakOmega22 = GetNRSpinPeakOmegav2(l, m, eta, a) / mTot;

        /* Define chi */
        chi = (spin1[2] + spin2[2]) / 2. + (spin1[2] - spin2[2]) / 2. * ((mass1 - mass2) / (mass1 + mass2)) / (1. - 2. * eta);

        /* For extreme chi (>= 0.8), there are scale factors in both complex
         * pseudo-QNM frequencies. kk, kt1, kt2 describe those factors. */
        // Below definitions of kk, kt1 and kt2 are likely obsolete
        kk = kt1 = kt2 = 1.;
        if (chi >= 0.8) {
            kk = 0.7 + 0.3 * exp(100. * (eta - 0.25));
            kt1 = 0.5 * sqrt(1. + 800.0 * eta * eta / 3.0) - 0.125;
            kt2 = 0.5 * pow(1. + 0.5 * eta * sqrt(eta) / 0.0225, 2. / 3.) - 0.2;
        }
        // Above definitions of kk, kt1 and kt2 are likely obsolete
        /*printf("a, chi and NRomega in QNM freq: %.16e %.16e %.16e %.16e %.16e %.16e\n",
         * spin1[2],spin2[2],mTot/LAL_MTSUN_SI,a,chi,NRPeakOmega22*mTot); */
        sh = 0.;
        //freq7sav = modefreqs->data[7];

        /* Cases 1, 2 and 3 in T1400476-v3. Note that the difference between the
         * chi1=chi2=0 case and the chi<0.7 cases is only in Dtcomb,
         * which is not specified or used in this file.
         */
        modefreqs->data[7] = (2. / 3. * NRPeakOmega22 / finalMass) + (1. / 3. * creal(modefreqs->data[0]));
        modefreqs->data[7] += I * 3.5 / 0.9 * cimag(modefreqs->data[0]);
        modefreqs->data[6] = (3. / 4. * NRPeakOmega22 / finalMass) + (1. / 4. * creal(modefreqs->data[0]));
        modefreqs->data[6] += I * 3.5 * cimag(modefreqs->data[0]);

        /* For extreme chi (>= 0.8), the ringdown attachment should end
         * slightly before the value given passed to this function. sh
         * gives exactly how long before. */
        if (chi >= 0.7 && chi < 0.8) {
            sh = -9. * (eta - 0.25);
        }
        if ((eta > 30. / 31. / 31. && eta <= 10. / 121. && chi >= 0.8) || (eta <= 30. / 31. / 31. && chi >= 0.8 && chi < 0.9)) {        // This is case 4 in T1400476-v3
            sh = -9. * (eta - 0.25) * (1. + 2. * exp(-(chi - 0.85) * (chi - 0.85) / 0.05 / 0.05)) * (1. + 1. / (1. + exp((eta - 0.01) / 0.001)));
            kk = 0.7 + 0.3 * exp(100. * (eta - 0.25));
            kt1 = 0.5 * sqrt(1. + 800.0 * eta * eta / 3.0) - 0.125;
            kt2 = 0.5 * pow(1. + 0.5 * eta * sqrt(eta) / 0.0225, 2. / 3.) - 0.2;
            modefreqs->data[4] = 0.4 * (1. + kk) * creal(modefreqs->data[6])
                + I * cimag(modefreqs->data[6]) / (2.5 * kt2 * exp(-(eta - 0.005) / 0.03));
            modefreqs->data[5] = 0.4 * (1. + kk) * creal(modefreqs->data[7])
                + I * cimag(modefreqs->data[7]) / (1.5 * kt1 * exp(-(eta - 0.005) / 0.03));
            modefreqs->data[6] = kk * creal(modefreqs->data[6]) + I * cimag(modefreqs->data[6]) / kt2;
            modefreqs->data[7] = kk * creal(modefreqs->data[7]) + I * cimag(modefreqs->data[7]) / kt1;
        }
        if (eta < 30. / 31. / 31. && chi >= 0.9) {      // This is case 5 in T1400476-v3
            sh = 0.55 - 9. * (eta - 0.25) * (1. + 2. * exp(-(chi - 0.85) * (chi - 0.85) / 0.05 / 0.05)) * (1. + 1. / (1. + exp((eta - 0.01) / 0.001)));
            kk = 0.7 + 0.3 * exp(100. * (eta - 0.25));
            kt1 = 0.5 * sqrt(1. + 800.0 * eta * eta / 3.0) - 0.125;
            kt2 = 0.5 * pow(1. + 0.5 * eta * sqrt(eta) / 0.0225, 2. / 3.) - 0.2;
            modefreqs->data[4] = 1.1 * 0.4 * (1. + kk) * creal(modefreqs->data[6])
                + I * cimag(modefreqs->data[6]) / (1.05 * 2.5 * kt2 * exp(-(eta - 0.005) / 0.03));
            modefreqs->data[5] = 0.4 * (1. + kk) * creal(modefreqs->data[7])
                + I * cimag(modefreqs->data[7]) / (1.05 * 1.5 * kt1 * exp(-(eta - 0.005) / 0.03));
            modefreqs->data[6] = kk * creal(modefreqs->data[6]) + I * cimag(modefreqs->data[6]) / kt2;
            modefreqs->data[7] = kk * creal(modefreqs->data[7]) + I * cimag(modefreqs->data[7]) / kt1;
        }
        if (eta > 10. / 121. && chi >= 0.8) {   // This is case 6 in T1400476-v3
            sh = 1. - 9. * (eta - 0.25) * (1. + 2. * exp(-(chi - 0.85) * (chi - 0.85) / 0.05 / 0.05)) * (1. + 1. / (1. + exp((eta - 0.01) / 0.001)));
            kk = 0.7 + 0.3 * exp(100. * (eta - 0.25));
            kt1 = 0.45 * sqrt(1. + 200.0 * eta * eta / 3.0) - 0.125;
            kt2 = 0.5 * pow(1. + 0.5 * eta * sqrt(eta) / 0.0225, 2. / 3.) - 0.2;
            modefreqs->data[6] = kk * creal(modefreqs->data[6]) + I * cimag(modefreqs->data[6]) / 0.95 / kt2;
            modefreqs->data[7] = kk * creal(modefreqs->data[7]) + I * cimag(modefreqs->data[7]) / kt1;
        }
        // The last line of T1400476-v3
        matchrange->data[0] -= sh;
        matchrange->data[1] -= sh;
/*
modefreqs->data[7] = 0.38068371/mTot + I/1.4677128/mTot;
modefreqs->data[6] = 0.37007703/mTot + I/1.3359367/mTot;
modefreqs->data[5] = 0.36980703/mTot + I/1.7791212/mTot;
modefreqs->data[4] = 0.3595034/mTot + I/2.6989764/mTot;
printf("sh = %f\n",sh);
printf("PeakOmega = %f, mTot = %f\n",NRPeakOmega22,mTot);
printf("w0 = %f, t0 = %f\n",creal(modefreqs->data[0])*mTot, 1./cimag(modefreqs->data[0])/mTot);
printf("w1 = %f, t1 = %f\n",creal(modefreqs->data[6])*mTot, 1./cimag(modefreqs->data[6])/mTot);
printf("w2 = %f, t2 = %f\n",creal(modefreqs->data[7])*mTot, 1./cimag(modefreqs->data[7])/mTot);
printf("w3 = %f, t3 = %f\n",creal(modefreqs->data[4])*mTot, 1./cimag(modefreqs->data[4])/mTot);
printf("w4 = %f, t4 = %f\n",creal(modefreqs->data[5])*mTot, 1./cimag(modefreqs->data[5])/mTot);
*/
    }
    // Move ringdown comb boundaries to sampling points to avoid numerical artifacts.
    matchrange->data[0] -= fmod(matchrange->data[0], dt / mTot);
    matchrange->data[1] -= fmod(matchrange->data[1], dt / mTot);
    /*for (j = 0; j < nmodes; j++)
     * {
     * printf("QNM frequencies: %d %d %d %f %f\n",l,m,j,creal(modefreqs->data[j])*mTot,1./cimag(modefreqs->data[j])/mTot);
     * } */

    /* Ringdown signal length: 10 times the decay time of the n=0 mode */
    Nrdwave = (INT4) (EOB_RD_EFOLDS / cimag(modefreqs->data[0]) / dt);

    /* Check the value of attpos, to prevent memory access problems later */
    if (matchrange->data[0] * mTot / dt < 5 || matchrange->data[1] * mTot / dt > matchrange->data[2] * mTot / dt - 2) {
        XLALPrintError("More inspiral points needed for ringdown matching.\n");
        printf("%.16e,%.16e,%.16e\n", matchrange->data[0] * mTot / dt, matchrange->data[1] * mTot / dt, matchrange->data[2] * mTot / dt - 2);
        XLALDestroyCOMPLEX16Vector(modefreqs);
        XLAL_ERROR(XLAL_EFAILED);
    }

    /*
     * STEP 2) Based on least-damped-mode decay time, allocate memory for rigndown waveform
     */

    /* Create memory for the ring-down and full waveforms, and derivatives of inspirals */

    rdwave1 = XLALCreateREAL8Vector(Nrdwave);
    rdwave2 = XLALCreateREAL8Vector(Nrdwave);
    rinspwave = XLALCreateREAL8Vector(6);
    dinspwave = XLALCreateREAL8Vector(6);
    ddinspwave = XLALCreateREAL8Vector(6);
    inspwaves1 = XLALCreateREAL8VectorSequence(3, 6);
    inspwaves2 = XLALCreateREAL8VectorSequence(3, 6);

    /* Check memory was allocated */
    if (!rdwave1 || !rdwave2 || !rinspwave || !dinspwave || !ddinspwave || !inspwaves1 || !inspwaves2) {
        XLALDestroyCOMPLEX16Vector(modefreqs);
        if (rdwave1)
            XLALDestroyREAL8Vector(rdwave1);
        if (rdwave2)
            XLALDestroyREAL8Vector(rdwave2);
        if (rinspwave)
            XLALDestroyREAL8Vector(rinspwave);
        if (dinspwave)
            XLALDestroyREAL8Vector(dinspwave);
        if (ddinspwave)
            XLALDestroyREAL8Vector(ddinspwave);
        if (inspwaves1)
            XLALDestroyREAL8VectorSequence(inspwaves1);
        if (inspwaves2)
            XLALDestroyREAL8VectorSequence(inspwaves2);
        XLAL_ERROR(XLAL_ENOMEM);
    }

    memset(rdwave1->data, 0, rdwave1->length * sizeof(REAL8));
    memset(rdwave2->data, 0, rdwave2->length * sizeof(REAL8));

    /*
     * STEP 3) Get values and derivatives of inspiral waveforms at matching comb points
     */

    /* Generate derivatives of the last part of inspiral waves */
    /* Get derivatives of signal1 */
    if (XLALGenerateHybridWaveDerivatives(rinspwave, dinspwave, ddinspwave, timeVec, signal1, matchrange, dt, mass1, mass2) == XLAL_FAILURE) {
        XLALDestroyCOMPLEX16Vector(modefreqs);
        XLALDestroyREAL8Vector(rdwave1);
        XLALDestroyREAL8Vector(rdwave2);
        XLALDestroyREAL8Vector(rinspwave);
        XLALDestroyREAL8Vector(dinspwave);
        XLALDestroyREAL8Vector(ddinspwave);
        XLALDestroyREAL8VectorSequence(inspwaves1);
        XLALDestroyREAL8VectorSequence(inspwaves2);
        XLAL_ERROR(XLAL_EFUNC);
    }
    for (j = 0; j < 6; j++) {
        inspwaves1->data[j] = rinspwave->data[j];
        inspwaves1->data[j + 6] = dinspwave->data[j];
        inspwaves1->data[j + 12] = ddinspwave->data[j];
    }

    /* Get derivatives of signal2 */
    if (XLALGenerateHybridWaveDerivatives(rinspwave, dinspwave, ddinspwave, timeVec, signal2, matchrange, dt, mass1, mass2) == XLAL_FAILURE) {
        XLALDestroyCOMPLEX16Vector(modefreqs);
        XLALDestroyREAL8Vector(rdwave1);
        XLALDestroyREAL8Vector(rdwave2);
        XLALDestroyREAL8Vector(rinspwave);
        XLALDestroyREAL8Vector(dinspwave);
        XLALDestroyREAL8Vector(ddinspwave);
        XLALDestroyREAL8VectorSequence(inspwaves1);
        XLALDestroyREAL8VectorSequence(inspwaves2);
        XLAL_ERROR(XLAL_EFUNC);
    }
    for (j = 0; j < 6; j++) {
        inspwaves2->data[j] = rinspwave->data[j];
        inspwaves2->data[j + 6] = dinspwave->data[j];
        inspwaves2->data[j + 12] = ddinspwave->data[j];
    }

    /*
     * STEP 4) Solve QNM coefficients and generate ringdown waveforms
     */

    /* Generate ring-down waveforms */
    if (XLALSimIMREOBHybridRingdownWave(rdwave1, rdwave2, dt, mass1, mass2, inspwaves1, inspwaves2, modefreqs, matchrange) == XLAL_FAILURE) {
        XLALDestroyCOMPLEX16Vector(modefreqs);
        XLALDestroyREAL8Vector(rdwave1);
        XLALDestroyREAL8Vector(rdwave2);
        XLALDestroyREAL8Vector(rinspwave);
        XLALDestroyREAL8Vector(dinspwave);
        XLALDestroyREAL8Vector(ddinspwave);
        XLALDestroyREAL8VectorSequence(inspwaves1);
        XLALDestroyREAL8VectorSequence(inspwaves2);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /*
     * STEP 5) Stitch inspiral and ringdown waveoforms
     */

    /* Generate full waveforms, by stitching inspiral and ring-down waveforms */
    //UINT4 attachIdx = matchrange->data[1] * mTot / dt;
    UINT4 attachIdx = round(matchrange->data[1] * mTot / dt);
    for (j = 1; j < Nrdwave; ++j) {
        signal1->data[j + attachIdx] = rdwave1->data[j];
        signal2->data[j + attachIdx] = rdwave2->data[j];
    }

    memset(signal1->data + Nrdwave + attachIdx, 0, (signal1->length - Nrdwave - attachIdx) * sizeof(REAL8));
    memset(signal2->data + Nrdwave + attachIdx, 0, (signal2->length - Nrdwave - attachIdx) * sizeof(REAL8));

    /* Free memory */
    XLALDestroyCOMPLEX16Vector(modefreqs);
    XLALDestroyREAL8Vector(rdwave1);
    XLALDestroyREAL8Vector(rdwave2);
    XLALDestroyREAL8Vector(rinspwave);
    XLALDestroyREAL8Vector(dinspwave);
    XLALDestroyREAL8Vector(ddinspwave);
    XLALDestroyREAL8VectorSequence(inspwaves1);
    XLALDestroyREAL8VectorSequence(inspwaves2);

    return XLAL_SUCCESS;
}

/**
 * The main  function for performing the ringdown attachment for SEOBNRv4 (and beyond)
 * This is the function which gets called by the code generating the full IMR waveform once
 * generation of the inspiral part has been completed.
 * The ringdown is attached by factoring the less damped harmonics and apply tanh fit to the rest.
 * STEP 1) Get mass and spin of the final black hole and the complex ringdown frequencies
 * STEP 2) Apply the fit function from the attachment time
 * STEP 3) Construct the full RD by applying (factor) of less damped 220 mode
 * STEP 4) Constructing the RD stitched to the inspiral-merger
 */
static UNUSED INT4 XLALSimIMREOBAttachFitRingdown(
//static INT4 XLALSimIMREOBAttachFitRingdown(
    REAL8Vector * signal1, /**<< OUTPUT, Real of inspiral waveform to which we attach ringdown */
    REAL8Vector * signal2, /**<< OUTPUT, Imag of inspiral waveform to which we attach ringdown */
    const INT4 l,          /**<< Current mode l */
    const INT4 m,          /**<< Current mode m */
    const REAL8 dt,        /**<< Sample time step (in seconds) */
    const REAL8 mass1,     /**<< First component mass (in Solar masses) */
    const REAL8 mass2,     /**<< Second component mass (in Solar masses) */
    const REAL8 spin1x,    /**<<The spin of the first object;  */
    const REAL8 spin1y,    /**<<The spin of the first object;  */
    const REAL8 spin1z,    /**<<The spin of the first object;  */
    const REAL8 spin2x,    /**<<The spin of the second object; */
    const REAL8 spin2y,    /**<<The spin of the second object; */
    const REAL8 spin2z,    /**<<The spin of the second object; */
    REAL8Vector * timeVec, /**<< Vector containing the time values */
    REAL8Vector * matchrange,
                           /**<< Time values chosen as points for performing comb matching */
    Approximant approximant/**<<The waveform approximant being used */
    ) {
    INT4 debugSB = 0;
    UINT4 nmodes;
    UINT4 i;
    UNUSED INT4 phasecount;
    REAL8 finalMass, finalSpin;
    REAL8Vector *ampWave;
    REAL8Vector *phWave;
    REAL8Vector *rdtime;
    COMPLEX16Vector *modefreqs;

    ampWave = XLALCreateREAL8Vector(signal1->length);
    phWave = XLALCreateREAL8Vector(signal1->length);

    REAL8 mtot = mass1 + mass2;
    REAL8 eta = mass1 * mass2 / (mtot * mtot);

    /* Here I assume that the spins were properly projected (is precessing) and only spin1z, spin2z
     * are relevant, if not we need to add extra function to define what we call chi1, chi2 */
    REAL8 chi1 = spin1z;
    REAL8 chi2 = spin2z;
    REAL8 spin1[3] = { spin1x, spin1y, spin1z };
    REAL8 spin2[3] = { spin2x, spin2y, spin2z };

    Approximant appr;
    appr = approximant;
//    if ( approximant != SEOBNRv2 ){
//        // We use here approximant = SEOBNRv2 for computation of the final mass final spin and frequencies
//        // This should change if other method for final mass/spin are available
//        appr = SEOBNRv2;
//    }

    if (debugSB) {
        printf("RDfit: we use spin1 %f, spin2 %f, and it should be dimensionless [-1,1] \n", chi1, chi2);
        printf("We use approximant = %d \n", appr);
    }
    REAL8 chis = 0.5 * (chi1 + chi2);
    REAL8 chia = 0.5 * (chi1 - chi2);
    REAL8 chi = chis + chia * sqrt(fabs(1 - 4. * eta)) / (1.0 - 2.0 * eta);

    /** getting modes */

    nmodes = 8;
    modefreqs = XLALCreateCOMPLEX16Vector(nmodes);
    if (XLALSimIMREOBGenerateQNMFreqV2(modefreqs, mass1, mass2, spin1, spin2, l, m, nmodes, appr) == XLAL_FAILURE) {
        XLALDestroyCOMPLEX16Vector(modefreqs);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /** Compute final mass final spin */
    if (XLALSimIMREOBFinalMassSpin(&finalMass, &finalSpin, mass1, mass2, spin1, spin2, appr) == XLAL_FAILURE) {
        XLAL_ERROR(XLAL_EFUNC);
    }

    COMPLEX16 sigma220;
    //sigma220 = -0.0609 - I*0.8326;
    sigma220 = (-cimag(modefreqs->data[0]) - I * creal(modefreqs->data[0])) * (mtot * LAL_MTSUN_SI);

    if (debugSB) {
        printf("Final mass = %f, final spin = %f\n", finalMass, finalSpin);
        printf("matchpoints are: %f,  %f,   %f \n", matchrange->data[0], matchrange->data[1], matchrange->data[2]);
        printf("the 0-overtone is: %f + i %f \n", creal(sigma220), cimag(sigma220));
    }
    /*  Compute amplitude and the phase of the data */
    phasecount = 0;
    for (i = 0; i < signal1->length; i++) {
        ampWave->data[i] = 0.0;
        phWave->data[i] = 0.0;
    }
    REAL8 prev, now, corph;
    prev = atan2(signal2->data[0], signal1->data[0]);
    phWave->data[0] = prev;
    for (i = 0; i < timeVec->length; i++) {
        ampWave->data[i] = sqrt(signal1->data[i] * signal1->data[i] + signal2->data[i] * signal2->data[i]);
        now = atan2(signal2->data[i], signal1->data[i]);
        if (i > 0) {
            corph = now - prev;
            corph = corph > LAL_PI ? corph - LAL_TWOPI : (corph < -LAL_PI ? corph + LAL_TWOPI : corph);

            phWave->data[i] = phWave->data[i - 1] + corph;
            prev = now;
        }
        //phWave->data[i] = now;
    }
    FILE *fout = NULL;
    if (debugSB) {
        fout = fopen("CheckStasAmplPhase.dat", "w");
        printf("Check the length: time %d, signal %d \n", timeVec->length, signal1->length);
        for (i = 0; i < timeVec->length; i++) {
            fprintf(fout, "%f   %.16e   %.16e   %.16e   %.16e  \n", timeVec->data[i], ampWave->data[i], phWave->data[i], signal1->data[i], signal2->data[i]);
        }

        fclose(fout);
    }

     /** Search for the maximum of the amplitude, which is not necessary (I think), since the rdMatch
      * tells us where maximum is */
    UINT4 indAmax;
    REAL8 valAmax;
    REAL8 tofAmax;

    indAmax = 0;
    valAmax = ampWave->data[0];
    //tofAmax = timeVec->data[0];
    tofAmax = matchrange->data[1];
    //for (i=1; i<ampWave->length-1; i++){
    INT4 found = 0;
    for (i = 1; i < timeVec->length - 1; i++) {

        if (timeVec->data[i] == tofAmax) {
            found = 1;
            indAmax = i;
            valAmax = ampWave->data[i];
        }
        //if (ampWave->data[i-1] <= ampWave->data[i] && ampWave->data[i] > ampWave->data[i+1]){
        //    indAmax = i;
        //    valAmax = ampWave->data[i];
        //    tofAmax= timeVec->data[i];
        //    printf("max i=%d  t= %f, amp = %.16e \n", i, timeVec->data[i], ampWave->data[i]);
        //    break;
        //}
    }
    if (found == 0) {
        printf(" time of maximum amplitude is not found ");
        exit(1);
    } else {
        if (debugSB) {
            printf(" found maximum times: %f,   %f \n", timeVec->data[indAmax], matchrange->data[1]);
        }
    }
    if (debugSB) {
        printf("Check: The maximum of amplitude is %.16e found at t=%f, index = %d (out of %d) \n", valAmax, tofAmax, indAmax, timeVec->length);
        //printf("compare it to the supplied time of peak: %f %f \n", matchrange->data[0], matchrange->data[1]);
    }

    /* Ringdown signal length: 10 times the decay time of the n=0 mode */
    //UINT4 Nrdwave = (INT4) (EOB_RD_EFOLDS / (cimag(modefreqs->data[0])*(mtot * LAL_MTSUN_SI)) / dt);
    UINT4 Nrdwave = (INT4) (EOB_RD_EFOLDS / cimag(modefreqs->data[0]) / dt);

    //printf("Stas Nrdwave %d,  dt = %f", Nrdwave, dt);
    // FIXME Alejandro started from dt not from 0, is it important?
    REAL8 dtM = dt / (mtot * LAL_MTSUN_SI);     // go to geometrica units
    rdtime = XLALCreateREAL8Vector(Nrdwave);
    for (i = 0; i < Nrdwave; i++) {
        rdtime->data[i] = i * dtM;      // this time for RD and it starts with 0 (Alejandro's time starts with
    }

    /** Computing fit coefficients */

    REAL8 A1coeff00 = 0.0830664;
    REAL8 A1coeff01 = -0.0196758;
    REAL8 A1coeff02 = -0.0136459;
    REAL8 A1coeff10 = 0.0612892;
    REAL8 A1coeff11 = 0.00146142;
    REAL8 A1coeff20 = -0.0893454;
    REAL8 ampcf1;
    ampcf1 = A1coeff00 + A1coeff01 * chi + A1coeff02 * chi * chi + A1coeff10 * eta + A1coeff11 * eta * chi + A1coeff20 * eta * eta;

    REAL8 A2coeff00 = -0.623953;
    REAL8 A2coeff01 = -0.371365;
    REAL8 A2coeff10 = 1.39777;
    REAL8 A2coeff11 = 2.40203;
    REAL8 A2coeff20 = -1.82173;
    REAL8 A2coeff21 = -5.25339;
    REAL8 ampcf2;
    ampcf2 = A2coeff00 + A2coeff01 * chi + A2coeff10 * eta + A2coeff11 * eta * chi + A2coeff20 * eta * eta + A2coeff21 * eta * eta * chi;

//    printf("creal(sigma220), 2.*ampcf1*tanh(ampcf2) = %.16e %.16e\n",1./creal(sigma220), 2.*ampcf1*tanh(ampcf2));
    if (creal(sigma220) > 2. * ampcf1 * tanh(ampcf2)) {
        ampcf1 = creal(sigma220) / (2. * tanh(ampcf2));
    }

    REAL8 P1coeff00 = 0.147584;
    REAL8 P1coeff01 = 0.00779176;
    REAL8 P1coeff02 = -0.0244358;
    REAL8 P1coeff10 = 0.263456;
    REAL8 P1coeff11 = -0.120853;
    REAL8 P1coeff20 = -0.808987;

    REAL8 phasecf1;
    phasecf1 = P1coeff00 + P1coeff01 * chi + P1coeff02 * chi * chi + P1coeff10 * eta + P1coeff11 * eta * chi + P1coeff20 * eta * eta;

    REAL8 P2coeff00 = 2.46654;
    REAL8 P2coeff01 = 3.13067;
    REAL8 P2coeff02 = 0.581626;
    REAL8 P2coeff10 = -6.99396;
    REAL8 P2coeff11 = -9.61861;
    REAL8 P2coeff20 = 17.5646;
    REAL8 phasecf2;

    phasecf2 = P2coeff00 + P2coeff01 * chi + P2coeff02 * chi * chi + P2coeff10 * eta + P2coeff11 * eta * chi + P2coeff20 * eta * eta;

    REAL8 tcons = 0.;   //attachment point relative to peak; possibility for non zero values not fully implemented

    REAL8 Arescaledtcons = valAmax * exp(-creal(sigma220) * tcons) / eta;
    REAL8 dtArescaledtcons = (0. - creal(sigma220) * valAmax) * exp(-creal(sigma220) * tcons) / eta;    // 0 - assumes extermum (max) of peak amplitude
    REAL8 ampcc1 = dtArescaledtcons * pow(cosh(ampcf2), 2) / ampcf1;
    REAL8 ampcc2 = (Arescaledtcons * ampcf1 - dtArescaledtcons * cosh(ampcf2) * sinh(ampcf2)) / ampcf1;

    REAL8Vector *ampRD;
    REAL8Vector *phRD;
    ampRD = XLALCreateREAL8Vector(Nrdwave);
    phRD = XLALCreateREAL8Vector(Nrdwave);

    for (i = 0; i < Nrdwave; i++) {
        ampRD->data[i] = eta * exp(creal(sigma220) * rdtime->data[i]) * (ampcc1 * tanh(ampcf1 * rdtime->data[i] + ampcf2) + ampcc2);

    }
    //if (debugSB){
    //    fout = fopen("StasAmpRD.dat", "w");
    //    for (i=0; i<Nrdwave; i++){
    //        fprintf(fout, "%.16e   %.16e   %.16e   %.16e \n", rdtime->data[i], ampRD->data[i],
    //                eta*exp( creal(sigma220)* rdtime->data[i] ),
    //                (ampcc1*tanh(ampcf1*rdtime->data[i]+ampcf2)+ampcc2) );
    //    }
    //    fclose(fout);
    //}
    if (debugSB) {
        fout = fopen("StasAmpRD_full.dat", "w");
        for (i = 0; i < indAmax; i++) {
            fprintf(fout, "%.16e    %.16e \n", timeVec->data[i], ampWave->data[i]);
        }
        for (i = 0; i < Nrdwave; i++) {
            fprintf(fout, "%.16e    %.16e \n", rdtime->data[i] + tofAmax, ampRD->data[i]);
        }
        fclose(fout);
    }

    // check that we have indAmax-1
    REAL8 omegarescaledtcons = (phWave->data[indAmax] - phWave->data[indAmax - 1]) / dtM - cimag(sigma220);
    //    (phasecutatpeak[-1]-phasecutatpeak[-2])/dt - sigma220.imag
    REAL8 phasecc1 = omegarescaledtcons * (phasecf2 + 1.0) / phasecf2 / phasecf1;
    REAL8 phi0 = phWave->data[indAmax];
    REAL8 logargnum, logargden;

    for (i = 0; i < Nrdwave; i++) {
        logargnum = 1. + phasecf2 * exp(-phasecf1 * rdtime->data[i]);
        logargden = 1. + phasecf2;
        phRD->data[i] = phi0 - phasecc1 * log(logargnum / logargden) + cimag(sigma220) * rdtime->data[i];
    }

    if (debugSB) {
        fout = fopen("StasPhRD_full.dat", "w");
        for (i = 0; i < indAmax; i++) {
            fprintf(fout, "%.16e    %.16e \n", timeVec->data[i], phWave->data[i]);
        }
        for (i = 0; i < Nrdwave; i++) {
            fprintf(fout, "%.16e    %.16e \n", rdtime->data[i] + tofAmax, phRD->data[i]);
        }
        fclose(fout);

        // Compute the frequency of the ful signal
        UINT4 totSz = indAmax + Nrdwave;
        REAL8Vector *PhFull;
        PhFull = XLALCreateREAL8Vector(totSz);
        REAL8Vector *tFull;
        tFull = XLALCreateREAL8Vector(totSz);
        REAL8Vector *frFull;
        frFull = XLALCreateREAL8Vector(totSz);

        for (i = 0; i < indAmax; i++) {
            tFull->data[i] = timeVec->data[i];
            PhFull->data[i] = phWave->data[i];
        }
        for (i = 0; i < Nrdwave; i++) {
            tFull->data[i + indAmax] = rdtime->data[i] + tofAmax;
            PhFull->data[i + indAmax] = phRD->data[i];
        }
        fout = fopen("StasPhRD_full2.dat", "w");
        for (i = 0; i < totSz; i++) {
            fprintf(fout, "%.16e   %.16e \n", tFull->data[i], PhFull->data[i]);
        }
        fclose(fout);

        gsl_spline *spline = NULL;
        gsl_interp_accel *acc = NULL;
        spline = gsl_spline_alloc(gsl_interp_cspline, totSz);
        acc = gsl_interp_accel_alloc();
        gsl_spline_init(spline, tFull->data, PhFull->data, totSz);
        for (i = 0; i < totSz; i++) {
            frFull->data[i] = gsl_spline_eval_deriv(spline, tFull->data[i], acc);
        }
        fout = fopen("StasFrRD_full.dat", "w");
        for (i = 0; i < totSz; i++) {
            fprintf(fout, "%.16e   %.16e \n", tFull->data[i], frFull->data[i]);
        }
        fclose(fout);

        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);

        XLALDestroyREAL8Vector(PhFull);
        XLALDestroyREAL8Vector(tFull);
        XLALDestroyREAL8Vector(frFull);

    }

    //FIXME check the size of signal1, signal2
    //FIXME check continuity at the attachment point (max of amplitude)
    for (i = 0; i < Nrdwave; i++) {
        signal1->data[i + indAmax] = ampRD->data[i] * cos(phRD->data[i]);
        signal2->data[i + indAmax] = ampRD->data[i] * sin(phRD->data[i]);
    }

    XLALDestroyREAL8Vector(ampWave);
    XLALDestroyREAL8Vector(phWave);
    XLALDestroyCOMPLEX16Vector(modefreqs);
    XLALDestroyREAL8Vector(rdtime);
    XLALDestroyREAL8Vector(ampRD);
    XLALDestroyREAL8Vector(phRD);

    return XLAL_SUCCESS;

}

#endif /*_LALSIMIMREOBHYBRIDRINGDOWN_C*/

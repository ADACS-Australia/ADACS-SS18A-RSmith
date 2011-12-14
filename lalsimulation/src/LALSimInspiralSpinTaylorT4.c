/*
 * Copyright (C) 2011 E. Ochsner
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
#include <lal/Units.h>
#include <lal/LALConstants.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALAdaptiveRungeKutta4.h>
#include <lal/TimeSeries.h>
#include "check_series_macros.h"

#define UNUSED(expr) do { (void)(expr); } while (0)
/*
#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif
*/

/* use error codes above 1024 to avoid conflicts with GSL */
#define LALSIMINSPIRAL_ST4_TEST_ENERGY 			1025
#define LALSIMINSPIRAL_ST4_TEST_OMEGADOT 		1026
#define LALSIMINSPIRAL_ST4_TEST_COORDINATE 		1027
#define LALSIMINSPIRAL_ST4_TEST_OMEGANAN 		1028
#define LALSIMINSPIRAL_ST4_DERIVATIVE_OMEGANONPOS 	1030

/* (2x) Highest available PN order - UPDATE IF NEW ORDERS ADDED!!*/
#define LAL_MAX_PN_ORDER 8
/* Number of variables used for precessing waveforms */
#define LAL_NUM_ST4_VARIABLES 14
/* absolute and relative tolerance for adaptive Runge-Kutta ODE integrator */
/* 1.e-06 is too large for end of 1.4--1.4 M_sun BNS inspiral */
/* (phase difference at end will be ~10% of GW cycle). */
/* 1.e-12 is used so last data point isn't nan for 6PN tidal, */
/* since larger values probably cause larger step sizes. */
#define LAL_ST4_ABSOLUTE_TOLERANCE 1.e-12
#define LAL_ST4_RELATIVE_TOLERANCE 1.e-12

NRCSID(LALSIMINSPIRALSPINTAYLORT4C, "$Id$");

/**
 * Struct containing all of the non-dynamical coefficients needed
 * to evolve a spinning, precessing binary and produce a waveform.
 * This struct is passed to the 2 static functions below
 */
typedef struct tagXLALSimInspiralSpinTaylorT4Coeffs
{
	REAL8 eta; 			// symmetric mass ratio
	REAL8 wdotnewt; // leading order coefficient of wdot = \dot{\omega}
	REAL8 wdotcoeff[LAL_MAX_PN_ORDER]; // coeffs. of PN corrections to wdot
	REAL8 wdotlogcoeff; 		// coefficient of log term in wdot
	REAL8 Ecoeff[LAL_MAX_PN_ORDER]; // coeffs. of PN corrections to energy
	REAL8 wdotSO15s1, wdotSO15s2; 	// non-dynamical 1.5PN SO corrections
	REAL8 wdotSS2; 			// non-dynamical 2PN SS correction
	REAL8 wdotSelfSS2; 	// non-dynamical 2PN self-spin correction
	REAL8 wdotQM2; 	// non-dynamical 2PN quadrupole-monopole correction
	REAL8 wdotSO25s1, wdotSO25s2; 	// non-dynamical 2.5PN SO corrections
	REAL8 ESO15s1, ESO15s2; 	// non-dynamical 1.5PN SO corrections
	REAL8 ESS2; 			// non-dynamical 2PN SS correction
	REAL8 ESelfSS2; 	// non-dynamical 2PN self-spin correction
	REAL8 EQM2; 	// non-dynamical 2PN quadrupole-monopole correction
	REAL8 ESO25s1, ESO25s2; 	// non-dynamical 2.5PN SO corrections 
	REAL8 LNhatSO15s1, LNhatSO15s2; // non-dynamical 1.5PN SO corrections
	REAL8 LNhatSS2; 		// non-dynamical 2PN SS correction 
	REAL8 wdottidal5pn;		// leading order tidal correction 
	REAL8 wdottidal6pn;		// next to leading order tidal correction
	REAL8 Etidal5pn;		// leading order tidal correction to energy
	REAL8 Etidal6pn;		// next to leading order tidal correction to energy
} XLALSimInspiralSpinTaylorT4Coeffs;

/* Declarations of static functions - defined below */
static int XLALSimInspiralSpinTaylorT4StoppingTest(double t, 
	const double values[], double dvalues[], void *mparams);
static int XLALSimInspiralSpinTaylorT4Derivatives(double t, 
	const double values[], double dvalues[], void *mparams);



/**
 * This function evolves the orbital equations for a precessing binary using 
 * the "TaylorT4" approximant for solving the orbital dynamics 
 * (see arXiv:0907.0700 for a review of the various PN approximants).
 *
 * It returns time series of the "orbital velocity", orbital phase, 
 * and components for both individual spin vectors, the "Newtonian"
 * orbital angular momentum (which defines the instantaneous plane)
 * and "E1", a basis vector in the instantaneous orbital plane.
 * Note that LNhat and E1 completely specify the instantaneous orbital plane.
 * It also returns the time and phase of the final time step
 *
 * For input, the function takes the two masses, the initial orbital phase, 
 * Values of S1, S2, LNhat, E1 vectors at starting time,
 * the desired time step size, the starting GW frequency, 
 * and PN order at which to evolve the phase,
 * 
 * NOTE: All vectors are given in the so-called "radiation frame", 
 * where the direction of propagation is the z-axis, the principal "+" 
 * polarization axis is the x-axis, and the y-axis is given by the RH rule.
 * You must give the initial values in this frame, and the time series of the
 * vector components will also be returned in this frame
 */
int XLALSimInspiralPNEvolveOrbitSpinTaylorT4(
	REAL8TimeSeries **V,          /**< post-Newtonian parameter [returned]*/
	REAL8TimeSeries **Phi,        /**< orbital phase            [returned]*/
	REAL8TimeSeries **S1x,	      /**< Spin1 vector x component [returned]*/
	REAL8TimeSeries **S1y,	      /**< "    "    "  y component [returned]*/
	REAL8TimeSeries **S1z,	      /**< "    "    "  z component [returned]*/
	REAL8TimeSeries **S2x,	      /**< Spin2 vector x component [returned]*/
	REAL8TimeSeries **S2y,	      /**< "    "    "  y component [returned]*/
	REAL8TimeSeries **S2z,	      /**< "    "    "  z component [returned]*/
	REAL8TimeSeries **LNhatx,     /**< unit orbital ang. mom. x [returned]*/
	REAL8TimeSeries **LNhaty,     /**< "    "    "  y component [returned]*/
	REAL8TimeSeries **LNhatz,     /**< "    "    "  z component [returned]*/
	REAL8TimeSeries **E1x,	      /**< orb. plane basis vector x[returned]*/
	REAL8TimeSeries **E1y,	      /**< "    "    "  y component [returned]*/
	REAL8TimeSeries **E1z,	      /**< "    "    "  z component [returned]*/
	REAL8 phi_end,                /**< orbital phase at last sample */
	REAL8 deltaT,          	      /**< sampling interval (s) */
	REAL8 m1,              	      /**< mass of companion 1 (kg) */
	REAL8 m2,              	      /**< mass of companion 2 (kg) */
	REAL8 fStart,                 /**< start frequency */
	REAL8 s1x,                    /**< initial value of S1x */
	REAL8 s1y,                    /**< initial value of S1y */
	REAL8 s1z,                    /**< initial value of S1z */
	REAL8 s2x,                    /**< initial value of S2x */
	REAL8 s2y,                    /**< initial value of S2y */
	REAL8 s2z,                    /**< initial value of S2z */
	REAL8 lnhatx,                 /**< initial value of LNhatx */
	REAL8 lnhaty,                 /**< initial value of LNhaty */
	REAL8 lnhatz,                 /**< initial value of LNhatz */
	REAL8 e1x,                    /**< initial value of E1x */
	REAL8 e1y,                    /**< initial value of E1y */
	REAL8 e1z,                    /**< initial value of E1z */
	REAL8 lambda1,                /**< (tidal deformability of mass 1) / (total mass)^5 (dimensionless) */
	REAL8 lambda2,                /**< (tidal deformability of mass 2) / (total mass)^5 (dimensionless) */
	LALSimInspiralInteraction interactionFlags, /**< flag to control spin and tidal effects */
	INT4 phaseO                   /**< twice post-Newtonian order */
	)
{
    INT4 intreturn;
    XLALSimInspiralSpinTaylorT4Coeffs params;/* Frequently used coefficients */
    ark4GSLIntegrator *integrator = NULL;     /* GSL integrator object */
    REAL8 yinit[LAL_NUM_ST4_VARIABLES];       /* initial values of parameters */
    REAL8Array *yout;	 /* time series of variables returned from integrator */
    /* intermediate variables */
    UINT4 i, lengths, len;
    REAL8 m1m2, m2m1, M, eta, Mchirp, norm, phiShift;
    LIGOTimeGPS tStart = LIGOTIMEGPSZERO;
	REAL8 m1M, m2M; /* m1/M, m2/M */

    /* Zero the coefficients */
    memset(&params, 0, sizeof(XLALSimInspiralSpinTaylorT4Coeffs));

    /* Define mass variables and other coefficients */
    m1m2 = m1 / m2;
    m2m1 = m2 / m1;
    m1 *= LAL_G_SI / pow(LAL_C_SI, 3.0); /* convert m1 from kg to seconds */
    m2 *= LAL_G_SI / pow(LAL_C_SI, 3.0); /* convert m2 from kg to seconds */
    M = m1 + m2;
	m1M = m1 / M;
	m2M = m2 / M;
    eta = m1 * m2 / M / M;
    Mchirp = M * pow(eta, 3./5.);
    params.wdotnewt = (96.0/5.0) * eta;
    params.eta = eta;
	
    /** 
     * Set coefficients up to PN order phaseO. 
     * epnorb is the binary energy and
     * wdotorb is the derivative of the orbital frequency \dot{\omega}.
     * These are just the non-spinning contributions.
     * Spin corrections must be recomputed at every step
     * because the relative orientations of S, L can change
     *
     * The values can be found in Buonanno, Iyer, Ochsner, Pan and Sathyaprakash
     * Phys. Rev. D 80, 084043 (2009) arXiv:0907.0700 (aka "BIOPS")
     * Eq. 3.1 for the energy and Eq. 3.6 for \dot{\omega}
     *
     * Note that Eq. 3.6 actually gives dv/dt, but this relates to \omega by
     * d (M \omega)/dt = d (v^3)/dt = 3 v^2 dv/dt
     * so the PN corrections are the same 
     * but the leading order is 3 v^2 times Eq. 3.6
     */
    switch( phaseO )
    {
        case -1: /* highest available PN order */
        case 8:
        /* case LAL_PNORDER_THREE_POINT_FIVE: */
        case 7:
            params.wdotcoeff[7] = (LAL_PI/12096.0) 
                    * (-13245.0 + 717350.0*eta + 731960.0*eta*eta);
            params.Ecoeff[7] = 0.;
        /* case LAL_PNORDER_THREE: */
        case 6:
            params.wdotcoeff[6] = 16447322263./139708800. - 1712./105. 
                    * LAL_GAMMA - 56198689./217728. * eta + LAL_PI * LAL_PI 
                    * (16./3. + 451./48. * eta) + 541./896. * eta * eta 
                    - 5605./2592. * eta * eta * eta - 856./105. * log(16.);
            params.wdotlogcoeff = - 1712./315.;
            params.Ecoeff[6] = - 675./64. + ( 34445./576. 
                    - 205./96. * LAL_PI * LAL_PI ) * eta
                    - (155./96.) *eta * eta - 35./5184. * eta * eta * eta;
        /* case LAL_PNORDER_TWO_POINT_FIVE: */
        case 5:
            params.wdotcoeff[5] = -(1./672.) * LAL_PI * (4159. + 15876.*eta);
            params.Ecoeff[5] = 0.;
        /* case LAL_PNORDER_TWO: */
        case 4:
            params.wdotcoeff[4] = (34103. + 122949.*eta 
                    + 59472.*eta*eta)/18144.;
            params.Ecoeff[4] = (-81. + 57.*eta - eta*eta)/24.;
        /*case LAL_PNORDER_ONE_POINT_FIVE:*/
        case 3:
            params.wdotcoeff[3] = 4. * LAL_PI;
            params.Ecoeff[3] = 0.;
        /*case LAL_PNORDER_ONE:*/
        case 2:
            params.wdotcoeff[2] = -(1./336.) * (743. + 924.*eta);
            params.Ecoeff[2] = -(1.0/12.0) * (9.0 + eta);
        /*case LAL_PNORDER_HALF:*/
        case 1:
            params.wdotcoeff[1] = 0.;
            params.Ecoeff[1] = 0.;
        /*case LAL_PNORDER_NEWTONIAN:*/
        case 0:
            params.wdotcoeff[0] = 1.;
            params.Ecoeff[0] = 1.;
            break;
        default: 
            XLALPrintError("XLAL Error - %s: Invalid phase. PN order %s\n", 
                    __func__, phaseO );
            XLAL_ERROR(XLAL_EINVAL);
            break;
    }

    /**
     * Compute the non-dynamical coefficients of spin corrections 
     * to the evolution equations for omega, L, S1 and S2 and binary energy E.
     * Flags control which spin corrections are included
     */
    params.LNhatSO15s1 	= 0.;
    params.LNhatSO15s2 	= 0.;
    params.wdotSO15s1 	= 0.;
    params.wdotSO15s2 	= 0.;
    params.ESO15s1 	= 0.;
    params.ESO15s2 	= 0.;
    params.LNhatSS2 	= 0.;
    params.wdotSS2 	= 0.;
    params.ESS2 	= 0.;
    params.wdotSelfSS2 	= 0.;
    params.ESelfSS2 	= 0.;
    params.wdotQM2 	= 0.;
    params.EQM2 	= 0.;
    params.wdotSO25s1 	= 0.;
    params.wdotSO25s2 	= 0.;
    params.ESO25s1 	= 0.;
    params.ESO25s2 	= 0.;
    if( (interactionFlags & LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_15PN) == LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_15PN )
    {
        params.LNhatSO15s1 	= 2. + 3./2. * m2m1;
        params.LNhatSO15s2	= 2. + 3./2. * m1m2;
        params.wdotSO15s1 	= - ( 113. + 75. * m2m1 ) / 12.;
        params.wdotSO15s2 	= - ( 113. + 75. * m1m2 ) / 12.;
        params.ESO15s1 		= 8./3. + 2. * m2m1;
        params.ESO15s2 		= 8./3. + 2. * m1m2;
    }
    if( (interactionFlags & LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_2PN) == LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_2PN )
    {
        params.LNhatSS2 	= -1.5 / eta;
        params.wdotSS2 		= - 1. / 48. / eta;
        params.ESS2 		= 1. / eta;
    }
    if( (interactionFlags & LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_SELF_2PN) == LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_SELF_2PN ) /* ADD ME!! */
    {
        params.wdotSelfSS2 	= 0.;
        params.ESelfSS2 	= 0.;
    }
    if( (interactionFlags & LAL_SIM_INSPIRAL_INTERACTION_QUAD_MONO_2PN) == LAL_SIM_INSPIRAL_INTERACTION_QUAD_MONO_2PN ) /* ADD ME!! */
    {
        params.wdotQM2 		= 0.;
        params.EQM2 		= 0.;
    }
    if( (interactionFlags & LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_25PN) == LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_25PN ) /* ADD ME!! */
    {
        params.wdotSO25s1 	= 0.;
        params.wdotSO25s2 	= 0.;	
        params.ESO25s1 		= 0.;
        params.ESO25s2 		= 0.;	
    }
	
    /**
     * Compute the coefficients of tidal corrections 
     * to the evolution equations for omega and binary energy E.
     * Flags control which tidal corrections are included.
     * Coefficients found from Eqs. 2.11 and 3.10 of 
     * Vines, Flanagan, Hinderer, PRD 83, 084051 (2011).
     */
    params.wdottidal5pn = 0.;
    params.wdottidal6pn = 0.;
    params.Etidal5pn = 0.;
    params.Etidal6pn = 0.;
    if( interactionFlags >= LAL_SIM_INSPIRAL_INTERACTION_TIDAL_5PN)
    {
        params.wdottidal5pn = lambda1 * 6. * (1. + 11. * m2M) / m1M
                + lambda2 * 6. * (1. + 11. * m1M) / m2M;
        params.Etidal5pn = - 9. * m2m1 * lambda1 - 9. * m1m2 * lambda2;
    }
    if( interactionFlags >= LAL_SIM_INSPIRAL_INTERACTION_TIDAL_6PN )
    {
        params.wdottidal6pn = lambda1 * (4421./28. - 12263./28. * m1M 
                + 1893./2. * m1M * m1M - 661 * m1M * m1M * m1M) / (2 * m1M)
                + lambda2 * (4421./28. - 12263./28. * m2M + 1893./2. * m2M * m2M
                - 661 * m2M * m2M * m2M) / (2 * m2M);
        params.Etidal6pn = - 11./2. * m2m1 
                * (3. + 2. * m1M + 3. * m1M * m1M) * lambda1 
                - 11./2. * m1m2 * (3. + 2. * m2M + 3. * m2M * m2M) * lambda2;
    }
	   
    /* length estimation (Newtonian) */
    /* since integration is adaptive, we could use a better estimate */
    lengths = (5.0/256.0) * pow(LAL_PI,-8.0/3.0) 
            * pow(Mchirp * fStart,-5.0/3.0) / fStart;

    /* Put initial values into a single array for the integrator */
    yinit[0] = 0.; /* set initial orbital phase to 0 - will adjust later */
    yinit[1] = LAL_PI * M * fStart;  /* \hat{omega} = (pi M f) */
    /* LNh(x,y,z) */
    yinit[2] = lnhatx;
    yinit[3] = lnhaty;
    yinit[4] = lnhatz;
    /* S1(x,y,z) */
    norm = m1 * m1 / M / M;
    yinit[5] = norm * s1x;
    yinit[6] = norm * s1y;
    yinit[7] = norm * s1z;
    /* S2(x,y,z) */
    norm = m2 * m2 / M / M;
    yinit[8] = norm * s2x;
    yinit[9] = norm * s2y;
    yinit[10]= norm * s2z;
    /* E1(x,y,z) */
    yinit[11] = e1x;
    yinit[12] = e1y;
    yinit[13] = e1z;

    /* initialize the integrator */
    integrator = XLALAdaptiveRungeKutta4Init(LAL_NUM_ST4_VARIABLES,
            XLALSimInspiralSpinTaylorT4Derivatives,
            XLALSimInspiralSpinTaylorT4StoppingTest,
            LAL_ST4_ABSOLUTE_TOLERANCE, LAL_ST4_RELATIVE_TOLERANCE);
    if( !integrator )
    {
        XLALPrintError("XLAL Error - %s: Cannot allocate integrator\n", 
                __func__);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /* stop the integration only when the test is true */
    integrator->stopontestonly = 1;

    /* run the integration; note: time is measured in \hat{t} = t / M */
    len = XLALAdaptiveRungeKutta4(integrator, (void *) &params, yinit,
            0.0, lengths/M, deltaT/M, &yout);

    intreturn = integrator->returncode;
    XLALAdaptiveRungeKutta4Free(integrator);

    if (!len) 
    {
        XLALPrintError("XLAL Error - %s: integration failed with errorcode %d.\n", __func__, intreturn);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /* Print warning about abnormal termination */
    if (intreturn != 0 && intreturn != LALSIMINSPIRAL_ST4_TEST_ENERGY 
            && intreturn != LALSIMINSPIRAL_ST4_TEST_OMEGADOT) 
    {
        XLALPrintWarning("XLAL Warning - %s: integration terminated with code %d.\n Waveform parameters were m1 = %e, m2 = %e, s1 = (%e,%e,%e), s2 = (%e,%e,%e), inc = %e.\n", __func__, intreturn, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, acos(lnhatz));
    }

    /* Adjust tStart so last sample is at time=0 */
    XLALGPSAdd(&tStart, -1.0*(len-1)*deltaT);

    /* allocate memory for output vectors */
    *V = XLALCreateREAL8TimeSeries( "PN_EXPANSION_PARAMETER", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *Phi = XLALCreateREAL8TimeSeries( "ORBITAL_PHASE", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *S1x = XLALCreateREAL8TimeSeries( "SPIN1_X_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *S1y = XLALCreateREAL8TimeSeries( "SPIN1_Y_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *S1z = XLALCreateREAL8TimeSeries( "SPIN1_Z_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *S2x = XLALCreateREAL8TimeSeries( "SPIN2_X_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *S2y = XLALCreateREAL8TimeSeries( "SPIN2_Y_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *S2z = XLALCreateREAL8TimeSeries( "SPIN2_Z_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *LNhatx = XLALCreateREAL8TimeSeries( "LNHAT_X_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *LNhaty = XLALCreateREAL8TimeSeries( "LNHAT_Y_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *LNhatz = XLALCreateREAL8TimeSeries( "LNHAT_Z_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *E1x = XLALCreateREAL8TimeSeries( "E1_BASIS_X_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *E1y = XLALCreateREAL8TimeSeries( "E1_BASIS_Y_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    *E1z = XLALCreateREAL8TimeSeries( "E1_BASIS_Z_COMPONENT", &tStart, 0., 
            deltaT, &lalDimensionlessUnit, len); 
    if ( !V || !Phi || !S1x || !S1y || !S1z || !S2x || !S2y || !S2z 
            || !LNhatx || !LNhaty || !LNhatz || !E1x || !E1y || !E1z )
    {
        XLALDestroyREAL8Array(yout);
        XLAL_ERROR(XLAL_EFUNC);
    }

    /* Compute phase shift to get desired value phi_end in last sample */
    phiShift = phi_end - yout->data[2*len-1];

    /* Copy time series of dynamical variables */
    /* from yout array returned by integrator to output time series */
    /* Note the first 'len' members of yout are the time steps */
    for( i = 0; i < len; i++ )
    {	
        (*Phi)->data->data[i] 		= yout->data[len+i] + phiShift;
        (*V)->data->data[i] 		= cbrt(yout->data[2*len+i]);
        (*LNhatx)->data->data[i] 	= yout->data[3*len+i];
        (*LNhaty)->data->data[i] 	= yout->data[4*len+i];
        (*LNhatz)->data->data[i] 	= yout->data[5*len+i];
        (*S1x)->data->data[i] 		= yout->data[6*len+i];
        (*S1y)->data->data[i] 		= yout->data[7*len+i];
        (*S1z)->data->data[i] 		= yout->data[8*len+i];
        (*S2x)->data->data[i] 		= yout->data[9*len+i];
        (*S2y)->data->data[i] 		= yout->data[10*len+i];
        (*S2z)->data->data[i] 		= yout->data[11*len+i];
        (*E1x)->data->data[i] 		= yout->data[12*len+i];
        (*E1y)->data->data[i] 		= yout->data[13*len+i];
        (*E1z)->data->data[i] 		= yout->data[14*len+i];
    }

    XLALDestroyREAL8Array(yout);

    return XLAL_SUCCESS;
}

/**
 * Internal function called by the integration routine.
 * Stops the integration if
 * 		1) The energy decreases with increasing orbital frequency
 *		2) The orbital frequency begins decreasing
 *		3) The orbital frequency becomes infinite 
 */
static int XLALSimInspiralSpinTaylorT4StoppingTest(
	double t, 
	const double values[],
	double dvalues[], 
	void *mparams
	)
{
    REAL8 omega, v, test;
    XLALSimInspiralSpinTaylorT4Coeffs *params 
            = (XLALSimInspiralSpinTaylorT4Coeffs*) mparams;
    /* Spin-corrections to energy (including dynamical terms) */
    REAL8 Espin15 = 0., Espin2 = 0., Espin25 = 0.;

    UNUSED(t);

    omega = values[1];
    v = pow(omega,1./3.);

    if( params->ESO15s1 != 0. || params->ESO15s2 != 0. || params->ESS2 != 0. )
    { 	/* Some spin terms are non-zero, so compute spins dot L */ 
        REAL8 LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z;
        REAL8 LNdotS1, LNdotS2;

        LNhx = values[2]; LNhy  = values[3]; LNhz = values[4] ;
        S1x  = values[5]; S1y   = values[6]; S1z  = values[7] ;
        S2x  = values[8]; S2y   = values[9]; S2z  = values[10];

        LNdotS1 = (LNhx*S1x + LNhy*S1y + LNhz*S1z);
        LNdotS2 = (LNhx*S2x + LNhy*S2y + LNhz*S2z);

        if( params->ESO15s1 != 0. || params->ESO15s2 != 0. )
        {   /* Compute 1.5PN SO correction to energy */
            Espin15 += params->ESO15s1 * LNdotS1 + params->ESO15s2 * LNdotS2;
        }

        if( params->ESS2 != 0. )
        {   /* Compute 2PN SS correction to energy */
            REAL8 S1dotS2;
            S1dotS2 = (S1x*S2x + S1y*S2y + S1z*S2z);
            Espin2 += params->ESS2  * (S1dotS2 - 3. * LNdotS1 * LNdotS2);
        }

        if( params->ESelfSS2 != 0. )
        {   /* Compute 2PN self-spin correction to energy */
            Espin2 += 0.; /* ADD ME!! */
        }

        if( params->ESO25s1 != 0. || params->wdotSO25s2 != 0. )
        {   /* Compute 2.5PN SO correction to energy */
            Espin25 += 0.; /* ADD ME!! */
        }
    }

    /**
     * We are testing if the orbital energy increases with \omega. 
     * We should be losing energy to GW flux, so if E increases 
     * we stop integration because the dynamics are becoming unphysical. 
     * 'test' is the PN expansion of dE/d\omega without the prefactor, 
     * i.e. dE/d\omega = dE/dv * dv/d\omega = - (M^2*eta/6) * test
     * Therefore, the energy is increasing with \omega iff. test < 0.
     */
    test = 2. + v * v * ( 4. * params->Ecoeff[2] 
            + v * ( 5. * (params->Ecoeff[3] + Espin15) 
            + v * ( 6. * (params->Ecoeff[4] + Espin2)
            + v * ( 7. * (params->Ecoeff[5] + Espin25)
            + v * ( 8. *  params->Ecoeff[6]
            + v * ( 9. *  params->Ecoeff[7]
			+ v * v * v * ( 12. * params->Etidal5pn
			+ v * v * ( 14. * params->Etidal6pn ) ) ) ) ) ) ) );
	
    if (test < 0.0) /* energy test fails! */
        return LALSIMINSPIRAL_ST4_TEST_ENERGY;
    else if (dvalues[1] < 0.0) /* omegadot < 0! */
        return LALSIMINSPIRAL_ST4_TEST_OMEGADOT;
    else if isnan(omega) /* omega is nan! */
        return LALSIMINSPIRAL_ST4_TEST_OMEGANAN;
    else /* Step successful, continue integrating */
        return GSL_SUCCESS;
}

/**
 * Internal function called by the integration routine.
 * Given the values of all the dynamical variables 'values' at time 't',
 * This function computes their derivatives 'dvalues'
 * so the ODE integrator can take a step
 * All non-dynamical quantities (masses, etc.) are passed in "mparams"
 *
 * The derivatives for \omega, L_N, S1, S2 can be found 
 * as Eqs. (1), (8) - (10) of gr-qc/0405090
 * The derivative of E1 is Eq. (15)-(16) of gr-qc/0310034
 */
static int XLALSimInspiralSpinTaylorT4Derivatives(
	double t, 
	const double values[],
	double dvalues[], 
	void *mparams
	) 
{
    /* coordinates and derivatives */
    REAL8 LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z, E1x, E1y, E1z;
    REAL8 omega, ds, domega, dLNhx, dLNhy, dLNhz;
    REAL8 dS1x, dS1y, dS1z, dS2x, dS2y, dS2z, dE1x, dE1y, dE1z;

    /* auxiliary variables */
    REAL8 v, v2, v3, v4, v5, v7, v11, omega2, LNdotS1, LNdotS2, S1dotS2;
    REAL8 OmegaLx, OmegaLy, OmegaLz, OmegaLdotLN;
    REAL8 OmegaEx, OmegaEy, OmegaEz, OmegaSx, OmegaSy, OmegaSz;
    REAL8 wspin15 = 0., wspin2 = 0., wspin25 = 0.;

    XLALSimInspiralSpinTaylorT4Coeffs *params 
            = (XLALSimInspiralSpinTaylorT4Coeffs*) mparams;

    UNUSED(t);

    /* copy variables */
    // UNUSED!!: s    = values[0] ;
    omega   	= values[1] ;
    LNhx = values[2] ; LNhy    	= values[3] ; LNhz 	= values[4] ;
    S1x  = values[5] ; S1y     	= values[6] ; S1z 	= values[7] ;
    S2x  = values[8] ; S2y     	= values[9] ; S2z 	= values[10];
    E1x  = values[11]; E1y    	= values[12]; E1z 	= values[13];

    if (omega <= 0.0) /* orbital frequency must be positive! */
    {
        return LALSIMINSPIRAL_ST4_DERIVATIVE_OMEGANONPOS;
    }

    v = pow(omega,1./3.);
    v2  = v * v; v3 = v2 * v; v4 = v3 * v; 
    v5 = v * v4; v7 = v4 * v3; v11 = v7 * v4;

    LNdotS1 = (LNhx*S1x + LNhy*S1y + LNhz*S1z);
    LNdotS2 = (LNhx*S2x + LNhy*S2y + LNhz*S2z);
    S1dotS2 = (S1x*S2x  + S1y*S2y  + S1z*S2z );

    /** 
     * domega
     * 
     * Note we are actually computing d \hat{\omega} / d \hat{t}
     * where \hat{\omega} = M \omega and \hat{t} = t / M
     * Therefore 'domega' = M^2 * d\omega / dt
     *
     * See Eqs. (1)-(7) of gr-qc/0405090 But note that our spin variables 
     * are scaled by component masses relative to that paper.
     * i.e. S_i = (m_i/M)^2 * \hat{S_i}
     *
     * non-spinning coefficients of \dot{\omega} (params->wdotcoeff[i])
     * should have been set before this function was called
     */
    if( params->wdotSO15s1 != 0. || params->wdotSO15s2 != 0. )
    {	/* Compute 1.5PN SO correction to omega derivative */
        wspin15 = params->wdotSO15s1 * LNdotS1 + params->wdotSO15s2 * LNdotS2;
    }
    if( params->wdotSS2 != 0. )
    {	/* Compute 2PN SS correction to omega derivative */
        wspin2 = params->wdotSS2 * (247. * S1dotS2 - 721. * LNdotS1 * LNdotS2);
    }
    if( params->wdotSelfSS2 != 0. )
    {	/* Compute 2PN self-spin correction to omega derivative */
        wspin2 += 0.; /* ADDME!! */	
    }
    if( params->wdotSO25s1 != 0. || params->wdotSO25s2 != 0. )
    {	/* Compute 2.5PN SO correction to omega derivative */
        wspin25 = 0.; /* ADDME!! */
    }

    domega  = params->wdotnewt * v11 * ( params->wdotcoeff[0] 
            + v * ( params->wdotcoeff[1] 
			+ v * ( params->wdotcoeff[2]
            + v * ( params->wdotcoeff[3] + wspin15 
            + v * ( params->wdotcoeff[4] + wspin2 
            + v * ( params->wdotcoeff[5] + wspin25 
            + v * ( params->wdotcoeff[6] + params->wdotlogcoeff * log(omega)
            + v * ( params->wdotcoeff[7] 
			+ v3 * ( params->wdottidal5pn
			+ v2 * ( params->wdottidal6pn ) ) ) ) ) ) ) ) ) );

    /**
     * dLN
     * 
     * d \hat{L_N}/d \hat{t} = M * d\hat{L_N} / dt = \Omega_L x \hat{L_N}
     * This is Eq. (10) of gr-qc/0405090 ( times M b/c we use \hat{t})
     */
    omega2 = omega * omega;
    /* \Omega_L vector */
    OmegaLx = omega2 * (params->LNhatSO15s1 * S1x + params->LNhatSO15s2 * S2x)
            + v7 * params->LNhatSS2 * (LNdotS2 * S1x + LNdotS1 * S2x);
    OmegaLy = omega2 * (params->LNhatSO15s1 * S1y + params->LNhatSO15s2 * S2y)
            + v7 * params->LNhatSS2 * (LNdotS2 * S1y + LNdotS1 * S2y);
    OmegaLz = omega2 * (params->LNhatSO15s1 * S1z + params->LNhatSO15s2 * S2z)
            + v7 * params->LNhatSS2 * (LNdotS2 * S1z + LNdotS1 * S2z);

    /* Take cross product of \Omega_L with \hat{L_N} */
    dLNhx = (-OmegaLz*LNhy + OmegaLy*LNhz);
    dLNhy = (-OmegaLx*LNhz + OmegaLz*LNhx);
    dLNhz = (-OmegaLy*LNhx + OmegaLx*LNhy);

    /**
     * dE1
     * 
     * d E_1 / d \hat{t} = M * d E_1 / dt
     * Computed from \Omega_L and \hat{L_N} with Eq. (15)-(16) of gr-qc/0310034
     */
    OmegaLdotLN = OmegaLx * LNhx + OmegaLy * LNhy + OmegaLz * LNhz;
    /* \Omega_E vector */
    OmegaEx = OmegaLx - OmegaLdotLN * LNhx;
    OmegaEy = OmegaLy - OmegaLdotLN * LNhy;
    OmegaEz = OmegaLz - OmegaLdotLN * LNhz;

    /* Take cross product of \Omega_E with E_1 */
    dE1x = (-OmegaEz*E1y + OmegaEy*E1z);
    dE1y = (-OmegaEx*E1z + OmegaEz*E1x);
    dE1z = (-OmegaEy*E1x + OmegaEx*E1y);

    /**
     * dS1
     * 
     * d S_1 / d \hat{t} = M * d S_1 / dt = \Omega_{S1} x S_1
     * This is Eq. (8) of gr-qc/0405090.
     * However, that paper uses spin variables which are M^2 times our spins
     */
    /* \Omega_{S1} vector */
    OmegaSx = v5 * params->eta * params->LNhatSO15s1 * LNhx
            + omega2 * 0.5 * (S2x - 3. * LNdotS2 * LNhx);
    OmegaSy = v5 * params->eta * params->LNhatSO15s1 * LNhy
            + omega2 * 0.5 * (S2y - 3. * LNdotS2 * LNhy);
    OmegaSz = v5 * params->eta * params->LNhatSO15s1 * LNhz
            + omega2 * 0.5 * (S2z - 3. * LNdotS2 * LNhz);

    /* Take cross product of \Omega_{S1} with S_1 */
    dS1x = (-OmegaSz*S1y + OmegaSy*S1z);
    dS1y = (-OmegaSx*S1z + OmegaSz*S1x);
    dS1z = (-OmegaSy*S1x + OmegaSx*S1y);

    /**
     * dS2
     * 
     * d S_2 / d \hat{t} = M * d S_2 / dt = \Omega_{S2} x S_2
     * This is Eq. (9) of gr-qc/0405090.
     * However, that paper uses spin variables which are M^2 times our spins
     */
    /* \Omega_{S2} vector */
    OmegaSx = v5 * params->eta * params->LNhatSO15s2 * LNhx
            + omega2 * 0.5 * (S1x - 3. * LNdotS1 * LNhx);
    OmegaSy = v5 * params->eta * params->LNhatSO15s2 * LNhy
            + omega2 * 0.5 * (S1y - 3. * LNdotS1 * LNhy);
    OmegaSz = v5 * params->eta * params->LNhatSO15s2 * LNhz
            + omega2 * 0.5 * (S1z - 3. * LNdotS1 * LNhz);

    /* Take cross product of \Omega_{S2} with S_2 */
    dS2x = (-OmegaSz*S2y + OmegaSy*S2z);
    dS2y = (-OmegaSx*S2z + OmegaSz*S2x);
    dS2z = (-OmegaSy*S2x + OmegaSx*S2y);

    /* dphi = d \phi / d \hat{t} = M d \phi /dt = M \omega = \hat{\omega} */
    ds = omega;

    dvalues[0]    = ds   ; dvalues[1]     = domega;
    dvalues[2]    = dLNhx; dvalues[3]     = dLNhy ; dvalues[4]    = dLNhz;
    dvalues[5]    = dS1x ; dvalues[6]     = dS1y  ; dvalues[7]    = dS1z ;
    dvalues[8]    = dS2x ; dvalues[9]     = dS2y  ; dvalues[10]   = dS2z ;
    dvalues[11]   = dE1x ; dvalues[12]    = dE1y  ; dvalues[13]   = dE1z ;

    return GSL_SUCCESS;
}

/**
 * Driver routine to compute a precessing post-Newtonian inspiral waveform
 * with phasing computed from energy balance using the so-called "T4" method.
 *
 * This routine allows the user to specify different pN orders
 * for phasing calcuation vs. amplitude calculations.
 */
int XLALSimInspiralSpinTaylorT4(
	REAL8TimeSeries **hplus,        /**< +-polarization waveform */
	REAL8TimeSeries **hcross,       /**< x-polarization waveform */
	REAL8 phi_end,                  /**< GW phase of final sample (rad) */
	REAL8 v0,                       /**< tail gauge term (default = 1) */
	REAL8 deltaT,                   /**< sampling interval (s) */
	REAL8 m1,                       /**< mass of companion 1 (kg) */
	REAL8 m2,                       /**< mass of companion 2 (kg) */
	REAL8 fStart,                   /**< start GW frequency (Hz) */
	REAL8 r,                        /**< distance of source (m) */
	REAL8 s1x,                      /**< initial value of S1x */
	REAL8 s1y,                      /**< initial value of S1y */
	REAL8 s1z,                      /**< initial value of S1z */
	REAL8 s2x,                      /**< initial value of S2x */
	REAL8 s2y,                      /**< initial value of S2y */
	REAL8 s2z,                      /**< initial value of S2z */
	REAL8 lnhatx,                   /**< initial value of LNhatx */
	REAL8 lnhaty,                   /**< initial value of LNhaty */
	REAL8 lnhatz,                   /**< initial value of LNhatz */
	REAL8 e1x,                      /**< initial value of E1x */
	REAL8 e1y,                      /**< initial value of E1y */
	REAL8 e1z,                      /**< initial value of E1z */
	REAL8 lambda1,                  /**< (tidal deformability of mass 1) / (total mass)^5 (dimensionless) */
	REAL8 lambda2,                  /**< (tidal deformability of mass 2) / (total mass)^5 (dimensionless) */
	LALSimInspiralInteraction interactionFlags, /**< flag to control spin and tidal effects */
	int phaseO,                     /**< twice PN phase order */
	int amplitudeO                  /**< twice PN amplitude order */
	)
{
    REAL8TimeSeries *V, *Phi, *S1x, *S1y, *S1z, *S2x, *S2y, *S2z;
    REAL8TimeSeries *LNhatx, *LNhaty, *LNhatz, *E1x, *E1y, *E1z;
    int status, n;

    /* Evolve the dynamical variables */
    /* Note we use phi_end/2 b/c the orbit evolver wants an orbital phase */
    /* but this waveform driver is supplied a reference GW phase */
    n = XLALSimInspiralPNEvolveOrbitSpinTaylorT4(&V, &Phi, &S1x, &S1y, &S1z, 
            &S2x, &S2y, &S2z, &LNhatx, &LNhaty, &LNhatz, &E1x, &E1y, &E1z,
            phi_end/2., deltaT, m1, m2, fStart, s1x, s1y, s1z, s2x, s2y,
            s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1, lambda2, interactionFlags, phaseO);
    if( n < 0 )
        XLAL_ERROR(XLAL_EFUNC);

    /* Use the dynamical variables to build the polarizations */
    status = XLALSimInspiralPrecessingPolarizationWaveforms(hplus, hcross,
            V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, 
            E1x, E1y, E1z, m1, m2, r, v0, amplitudeO);

    /* Destroy vectors of dynamical variables, check for errors then exit */
    XLALDestroyREAL8TimeSeries(V);
    XLALDestroyREAL8TimeSeries(Phi);
    XLALDestroyREAL8TimeSeries(S1x);
    XLALDestroyREAL8TimeSeries(S1y);
    XLALDestroyREAL8TimeSeries(S1z);
    XLALDestroyREAL8TimeSeries(S2x);
    XLALDestroyREAL8TimeSeries(S2y);
    XLALDestroyREAL8TimeSeries(S2z);
    XLALDestroyREAL8TimeSeries(LNhatx);
    XLALDestroyREAL8TimeSeries(LNhaty);
    XLALDestroyREAL8TimeSeries(LNhatz);
    XLALDestroyREAL8TimeSeries(E1x);
    XLALDestroyREAL8TimeSeries(E1y);
    XLALDestroyREAL8TimeSeries(E1z);
    if( status < 0 )
        XLAL_ERROR(XLAL_EFUNC);

    return n;
}

/**
 * Driver routine to compute a precessing post-Newtonian inspiral waveform
 * with phasing computed from energy balance using the so-called "T4" method.
 *
 * This routine assumes leading-order amplitude dependence (restricted waveform)
 * but allows the user to specify the phase PN order
 */
int XLALSimInspiralRestrictedSpinTaylorT4(
	REAL8TimeSeries **hplus,        /**< +-polarization waveform */
	REAL8TimeSeries **hcross,       /**< x-polarization waveform */
	REAL8 phi_end,                  /**< GW phase of final sample (rad) */
	REAL8 v0,                       /**< tail gauge term (default = 1) */
	REAL8 deltaT,                   /**< sampling interval (s) */
	REAL8 m1,                       /**< mass of companion 1 (kg) */
	REAL8 m2,                       /**< mass of companion 2 (kg) */
	REAL8 fStart,                   /**< start GW frequency (Hz) */
	REAL8 r,                        /**< distance of source (m) */
	REAL8 s1x,                      /**< initial value of S1x */
	REAL8 s1y,                      /**< initial value of S1y */
	REAL8 s1z,                      /**< initial value of S1z */
	REAL8 s2x,                      /**< initial value of S2x */
	REAL8 s2y,                      /**< initial value of S2y */
	REAL8 s2z,                      /**< initial value of S2z */
	REAL8 lnhatx,                   /**< initial value of LNhatx */
	REAL8 lnhaty,                   /**< initial value of LNhaty */
	REAL8 lnhatz,                   /**< initial value of LNhatz */
	REAL8 e1x,                      /**< initial value of E1x */
	REAL8 e1y,                      /**< initial value of E1y */
	REAL8 e1z,                      /**< initial value of E1z */
	REAL8 lambda1,                  /**< (tidal deformability of mass 1) / (total mass)^5 (dimensionless) */
	REAL8 lambda2,                  /**< (tidal deformability of mass 2) / (total mass)^5 (dimensionless) */
	LALSimInspiralInteraction interactionFlags, /**< flag to control spin and tidal effects */					  
	int phaseO                      /**< twice PN phase order */
	)
{
    REAL8TimeSeries *V, *Phi, *S1x, *S1y, *S1z, *S2x, *S2y, *S2z;
    REAL8TimeSeries *LNhatx, *LNhaty, *LNhatz, *E1x, *E1y, *E1z;
    int status, n, ampO = 0;

    /* Evolve the dynamical variables */
    /* Note we use phi_end/2 b/c the orbit evolver wants an orbital phase */
    /* but this waveform driver is supplied a reference GW phase */
    n = XLALSimInspiralPNEvolveOrbitSpinTaylorT4(&V, &Phi, &S1x, &S1y, &S1z, 
            &S2x, &S2y, &S2z, &LNhatx, &LNhaty, &LNhatz, &E1x, &E1y, &E1z,
            phi_end/2., deltaT, m1, m2, fStart, s1x, s1y, s1z, s2x, s2y,
            s2z, lnhatx, lnhaty, lnhatz, e1x, e1y, e1z, lambda1, lambda2, interactionFlags, phaseO);
    if( n < 0 )
        XLAL_ERROR(XLAL_EFUNC);

    /* Use the dynamical variables to build the polarizations */
    status = XLALSimInspiralPrecessingPolarizationWaveforms(hplus, hcross,
            V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNhatx, LNhaty, LNhatz, 
            E1x, E1y, E1z, m1, m2, r, v0, ampO);

    /* Destroy vectors of dynamical variables, check for errors then exit */
    XLALDestroyREAL8TimeSeries(V);
    XLALDestroyREAL8TimeSeries(Phi);
    XLALDestroyREAL8TimeSeries(S1x);
    XLALDestroyREAL8TimeSeries(S1y);
    XLALDestroyREAL8TimeSeries(S1z);
    XLALDestroyREAL8TimeSeries(S2x);
    XLALDestroyREAL8TimeSeries(S2y);
    XLALDestroyREAL8TimeSeries(S2z);
    XLALDestroyREAL8TimeSeries(LNhatx);
    XLALDestroyREAL8TimeSeries(LNhaty);
    XLALDestroyREAL8TimeSeries(LNhatz);
    XLALDestroyREAL8TimeSeries(E1x);
    XLALDestroyREAL8TimeSeries(E1y);
    XLALDestroyREAL8TimeSeries(E1z);
    if( status < 0 )
        XLAL_ERROR(XLAL_EFUNC);

    return n;
}


/*
 *  Copyright (C) 2012 Chris Pankow
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

#include <lal/LALSimInspiralPrecess.h>
#include <lal/LALAtomicDatatypes.h>

/**
 * Takes in the h_lm spherical harmonic decomposed modes and rotates the modes
 * by Euler angles alpha, beta, and gamma using the Wigner D matricies.
 * 
 * e.g.
 *
 * \f$\tilde{h}_{l,m}(t) = D^l_{m,m'} h_{l,m'}(t)\f$
 */
int XLALSimInspiralPrecessionRotateModes(
                SphHarmTimeSeries* h_lm, /**< spherical harmonic decomposed modes, modified in place */
                REAL8TimeSeries* alpha, /**< alpha Euler angle time series */
                REAL8TimeSeries* beta, /**< beta Euler angle time series */
                REAL8TimeSeries* gam /**< gamma Euler angle time series */
){

	unsigned int i;
	int l, lmax, m, mp;
	lmax = XLALSphHarmTimeSeriesGetMaxL( h_lm );
	// Temporary holding variables
	complex double *x_lm = XLALCalloc( 2*lmax+1, sizeof(complex double) );
	COMPLEX16TimeSeries **h_xx = XLALCalloc( 2*lmax+1, sizeof(COMPLEX16TimeSeries) );

	for(i=0; i<alpha->data->length; i++){
		for(l=0; l<=lmax; l++){
			for(m=0; m<2*l+1; m++){
				h_xx[m] = XLALSphHarmTimeSeriesGetMode(h_lm, l, m-l);
				if( !h_xx[m] ){
					x_lm[m] = 0;
				} else {
					x_lm[m] = h_xx[m]->data->data[i];
					h_xx[m]->data->data[i] = 0;
				}
			}

			for(m=0; m<2*l+1; m++){
				for(mp=0; mp<2*l+1; mp++){
					if( !h_xx[m] ) continue;
					h_xx[m]->data->data[i] += 
						x_lm[mp] * XLALWignerDMatrix( l, mp-l, m-l, alpha->data->data[i], beta->data->data[i], gam->data->data[i] );
				}
			}
		}
	}

	XLALFree( x_lm );
	XLALFree( h_xx );
	return XLAL_SUCCESS;
}

/**
 * Takes in the l=2, abs(m)=2 decomposed modes as a strain time series and
 * imposes the effect of a constant cone of precession. This is accomplished
 * by taking the axis of the binary rotational plane and rotating the Y_lm
 * such that it appears to "precess" around a fixed J direction.
 * Note that h_2_2 and h_22 are modified in place.
 *
 * Future revisions will change the first two pointers to this:
 * COMPLEX16TimeSeries** h_lm
 *
 * and add 
 * unsigned int l
 *
 * Thus the input h_lm will be considered a list of pointers to the h_lm for a
 * given l and the appropriate action will be taken for *all* of the submodes.
 */
int XLALSimInspiralConstantPrecessionConeWaveformModes(
				SphHarmTimeSeries* h_lm, /**< (l,m) modes, modified in place */
				double precess_freq, /**< Precession frequency in Hz */
				double a, /**< Opening angle of precession cone in rads  */
				double phi_precess, /**< initial phase in cone of L around J */
				double alpha_0, /**< azimuth btwn center of cone and line of sight */
				double beta_0 /**< zenith btwn center of cone and line of sight */
) {
		/*
		 * Since the h_22 modes are the most likely to exist, we'll use those
		 * to do our error checking
		 */

		COMPLEX16TimeSeries *h_22, *h_2_2;
		h_22 = XLALSphHarmTimeSeriesGetMode( h_lm, 2, 2 );
		h_2_2 = XLALSphHarmTimeSeriesGetMode( h_lm, 2, -2 );

		if( !h_22 && !h_2_2 ){
			XLALPrintError( "XLAL Error - %s: Currently, ConstanntPrecessionConeWaveformModes requires the l=2 m=+/-2 modes to exist to continue.", __func__);
			XLAL_ERROR( XLAL_EINVAL );
		}

		// Error checking
		// Since we need at least three points to do any of the numerical work
		// we intend to, if the waveform is two points or smaller, we're in
		// trouble. I don't expect this to be a problem.
		if( h_2_2->data->length <= 2 ){
			XLALPrintError( "XLAL Error - %s: Waveform length is too small to evolve accurately.", __func__);
			XLAL_ERROR( XLAL_EBADLEN );
		}
        if( h_2_2->data->length != h_22->data->length ){
            XLALPrintError( "XLAL Error - %s: Input (2,2) and (2,-2) modes have different length.", __func__);
            XLAL_ERROR( XLAL_EBADLEN );
        }

		unsigned int i;
		double omg_p = 2*LAL_PI*precess_freq;
		double t=0;

		// time evolved Euler angles
		REAL8TimeSeries* alpha = XLALCreateREAL8TimeSeries(
			"euler angle alpha",
			&(h_22->epoch),
			h_22->f0,
			h_22->deltaT,
			&(h_22->sampleUnits),
			h_22->data->length
		);
		REAL8TimeSeries* beta = XLALCreateREAL8TimeSeries(
			"euler angle beta",
			&(h_22->epoch),
			h_22->f0,
			h_22->deltaT,
			&(h_22->sampleUnits),
			h_22->data->length
		);
		REAL8TimeSeries* gam = XLALCreateREAL8TimeSeries(
			"euler angle gamma",
			&(h_22->epoch),
			h_22->f0,
			h_22->deltaT,
			&(h_22->sampleUnits),
			h_22->data->length
		);

		// Minimal rotation constraint
		// \gamma(t) = \int \cos(\beta(t)) \alpha'(t) dt
		// Uses the second order finite difference to estimate dalpha/dt
		// Then the trapezoid rule for the integration
		for(i=0; i<alpha->data->length; i++){
			t = h_22->deltaT*i;
			alpha->data->data[i] = a*sin(omg_p * t + phi_precess) + alpha_0;
			beta->data->data[i] = a*cos(omg_p * t + phi_precess) + beta_0;
		}

		// NOTE: The step size cancels out between the difference and sum and
		// thus does not appear below.
		// two point forward difference
		double dalpha_0 = alpha->data->data[1] - alpha->data->data[0];
		// three point central difference
		double dalpha_1 = 0.5*(alpha->data->data[2] - alpha->data->data[0]);
		gam->data->data[0] = 0.;
		gam->data->data[1] =
				cos(beta->data->data[0])*dalpha_0 +
				cos(beta->data->data[1])*dalpha_1;
		for(i=2; i<gam->data->length-1; i++){
			// three point central difference
			dalpha_0 = dalpha_1;
			dalpha_1 = 0.5*(alpha->data->data[i+1] - alpha->data->data[i-1]);
			// Two point numerical integration over the interval
			gam->data->data[i] = 
				gam->data->data[i-1] +
				cos(beta->data->data[i-1])*dalpha_0 +
				cos(beta->data->data[i])*dalpha_1;
		}
		// Use two point backward difference for last point
        dalpha_0 = dalpha_1;
        dalpha_1 = alpha->data->data[i] - alpha->data->data[i-1];
		gam->data->data[i] = gam->data->data[i-1] +
                cos(beta->data->data[i-1])*dalpha_0 +
                cos(beta->data->data[i])*dalpha_1;

		/*************************************************************/
		// Rotate waveform
		//SphHarmTimeSeries *h_lm;

		//h_lm = XLALSphHarmTimeSeriesAddMode( NULL, *h_22, 2, 2 );
		//unsigned int data_length = (*h_22)->data->length;
		//XLALDestroyCOMPLEX16TimeSeries( *h_22 );
		//h_lm = XLALSphHarmTimeSeriesAddMode( h_lm, *h_2_2, 2, -2 );
		//XLALDestroyCOMPLEX16TimeSeries( *h_2_2 );

		XLALSimInspiralPrecessionRotateModes( h_lm, alpha, beta, gam );

		//*h_22 = XLALSphHarmTimeSeriesGetMode( h_lm, 2, 2 );
		//*h_22 = XLALCutCOMPLEX16TimeSeries( *h_22, 0, data_length );
		//*h_2_2 = XLALSphHarmTimeSeriesGetMode( h_lm, 2, -2 );
		//*h_2_2 = XLALCutCOMPLEX16TimeSeries( *h_2_2, 0, data_length );

		//XLALDestroySphHarmTimeSeries( h_lm );
		/*************************************************************/

		XLALDestroyREAL8TimeSeries( alpha );
		XLALDestroyREAL8TimeSeries( beta );
		XLALDestroyREAL8TimeSeries( gam );

		return XLAL_SUCCESS;
}

/**
 * Takes in the spherical harmonic decomposed modes as a strain time series and
 * imposes the effect of a constant cone of precession. The result is returned
 * in the physical waveforms hp, hx, after they have been resummed from the 
 * modified h_lm waveforms.
 *
 * NOTE: the h_lm modes will be modified in place
 */
int XLALSimInspiralConstantPrecessionConeWaveform(
				REAL8TimeSeries** hp, /**< Output precessing plus polarization */
				REAL8TimeSeries** hx, /**< Output precessing cross polarization*/
				SphHarmTimeSeries* h_lm, /**< Input non-precessing (l,m) modes */
				double precess_freq, /**< Precession frequency in Hz */
				double a, /**< Opening angle of precession cone in rads  */
				double phi_precess, /**< initial phase in cone of L around J */
				double alpha_0, /**< azimuth btwn center of cone and line of sight */
				double beta_0 /**< zenith btwn center of cone and line of sight */
) {
		int ret = XLALSimInspiralConstantPrecessionConeWaveformModes( 
						h_lm,
						precess_freq, a, phi_precess,
						alpha_0, beta_0 );
        if( ret != XLAL_SUCCESS )
            XLAL_ERROR( XLAL_EFUNC );

        /*
         * Since the h_22 modes are the most likely to exist, we'll use those
         * to do our error checking
         */
        COMPLEX16TimeSeries *h_22;
        h_22 = XLALSphHarmTimeSeriesGetMode( h_lm, 2, 2 );

		if( !(*hp) ){
			XLALDestroyREAL8TimeSeries( *hp );
		}
		*hp = XLALCreateREAL8TimeSeries(
			"h_+ precessed waveform",
			&(h_22->epoch),
			h_22->f0,
			h_22->deltaT,
			&(h_22->sampleUnits),
			h_22->data->length
		);
		if( !(*hx) ){
			XLALDestroyREAL8TimeSeries( *hx );
		}
		*hx = XLALCreateREAL8TimeSeries(
			"h_x precessed waveform",
			&(h_22->epoch),
			h_22->f0,
			h_22->deltaT,
			&(h_22->sampleUnits),
			h_22->data->length
		);

		unsigned int i, l, l_max;
		int m;
		// FIXME: Should these be fixed?
		double view_th = 0.0, view_ph = 0.0;
		l_max = XLALSphHarmTimeSeriesGetMaxL( h_lm );
		COMPLEX16TimeSeries* h_xx;
		// Reconstitute the waveform from the h_lm
		for(i=0; i<h_22->data->length; i++){
			COMPLEX16 x_t = 0.I;
			for(l=2; l <= l_max; l++){
				for(m=(int)-l; m<=(int)l; m++){
					h_xx = XLALSphHarmTimeSeriesGetMode( h_lm, l, m );
					if( h_xx ){
						x_t += h_xx->data->data[i] * XLALSpinWeightedSphericalHarmonic( view_th, view_ph, -2, l, m );
					}
				}
			}
			(*hp)->data->data[i] = crealf( x_t );
			(*hx)->data->data[i] = cimagf( x_t );
		}

		// User should do this.
		//XLALDestroyCOMPLEX16TimeSeries( h_22 );
		//XLALDestroyCOMPLEX16TimeSeries( h_2_2 );

        return XLAL_SUCCESS;
}

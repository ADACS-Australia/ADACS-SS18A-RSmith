/*
*  Copyright (C) 2007 David M. Whitbeck, Thomas Essinger-Hileman, Jolien Creighton, Ian Jones, Benjamin Owen, Reinhard Prix, Karl Wette
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
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <lal/AVFactories.h>
#include <lal/DetectorSite.h>
#include <lal/LALStdlib.h>
#include <lal/PtoleMetric.h>
#include <lal/GetEarthTimes.h>
#include <lal/Factorial.h>

/* Bounds on acceptable parameters, may be somewhat arbitrary */
#define MIN_DURATION (LAL_DAYSID_SI/LAL_TWOPI) /* Metric acts funny if duration too short */
#define MIN_MAXFREQ  1.                        /* Arbitrary */

/* A private factorial function */
/* static int factrl( int ); */

/**
 * \author Jones D. I., Owen, B. J., and Whitbeck, D. M.
 * \date 2001-2005
 * \brief Computes metric components for a pulsar search in the ``Ptolemaic''
 * approximation; both the Earth's spin and orbit are included.
 *
 * ### Description ###
 *
 * This function computes metric components in a way that yields results
 * very similar to those of LALCoherentMetric() called with the
 * output of LALTBaryPtolemaic(). The CPU demand, however, is less,
 * and the metric components can be expressed analytically, lending
 * themselves to better understanding of the behavior of the parameter
 * space.  For short durations (less than about 70000 seconds or 20 hours) a
 * Taylor series expansion is used to improve the accuracy with which several
 * terms are computed.
 *
 * ### Algorithm ###
 *
 * For speed and checking reasons, a minimum of numerical computation is
 * involved. The metric components can be expressed analytically (though not
 * tidily) in terms of trig functions and are much too large to be worth
 * writing down here.  More comprehensive documentation on the derivation of
 * the metric components can be found in the pulgroup CVS archive as
 * docs/S2/FDS/Isolated/ptolemetric.tex.  Jones, Owen, and Whitbeck will write
 * up the calculation and some tests as a journal article.
 *
 * The function XLALGetEarthTimes() is used to calculate the spin and
 * rotational phase of the Earth at the beginning of the observation.
 *
 * On output, the \a metric->data is arranged with the same indexing
 * scheme as in LALCoherentMetric(). The order of the parameters is
 * \f$(f_0, \alpha, \delta)\f$.
 *
 * ### Notes ###
 *
 * The analytic metric components were derived separately by Jones and
 * Whitbeck (and partly by Owen) and found to agree.  Also, the output of this
 * function has been compared against that of the function combination
 * (CoherentMetric + TDBaryPtolemaic), which is a numerical implementation of
 * the Ptolemaic approximation, and found to agree up to the fourth
 * significant figure or better.  Even using DTEphemeris.c for the true
 * Earth's orbit only causes errors in the metric components of order 10\%,
 * with (so far) no noticeable effect on the sky coverage.
 *
 * At present, only one spindown parameter can be included with the sky
 * location.  The code contains (commented out) expressions for
 * spindown-spindown metric components for an arbitrary number of spindowns,
 * but the (commented out) spindown-sky components neglect orbital motion.
 *
 * A separate routine, XLALSpindownMetric(), has been added to compute the
 * metric for multiple spindowns but for fixed sky position, suitable for
 * e.g. directed searches.
 */
void LALPtoleMetric( LALStatus *status,
                     REAL8Vector *metric,
                     PtoleMetricIn *input )
{
  INT2 j;              /* Loop counters */
  /*REAL8 temp1, temp2, temp3, temp4; dummy variables */
  REAL8 R_o, R_s;         /* Amplitude of daily/yearly modulation, s */
  REAL8 lat, lon;         /* latitude and longitude of detector site */
  REAL8 omega_s, omega_o; /* Ang freq of daily/yearly modulation, rad/s */
  REAL8 phi_o_i;          /* Phase of Earth's orbit at begining (epoch) */
  REAL8 phi_o_f;          /* Phase of Earth's orbit at end (epoch+duration) */
  REAL8 phi_s_i;          /* Phase of Earth's spin at beginning (epoch) */
  REAL8 phi_s_f;          /* Phase of Earth's spin at end (epoch+duration) */
  REAL8 cos_l, sin_l, sin_2l;  /* Trig fns for detector lattitude */
  REAL8 cos_i, sin_i;     /* Trig fns for inclination of Earth's spin */
  REAL8 cos_a, sin_a, sin_2a;  /* Trig fns for source RA */
  REAL8 cos_d, sin_d, sin_2d;  /* Trig fns for source declination */
  REAL8 A[22];     /* Array of intermediate quantities */
  REAL8 B[10];     /* Array of intermediate quantities */
  REAL8 Is[5];      /* Array of integrals needed for spindown.*/
  UINT2 dim;         /* Dimension of parameter space */
  REAL8 tMidnight, tAutumn; /* Needed to calculate phases of spin*/
  /* and orbit at t_gps =0             */
  REAL8Vector *big_metric; /* 10-dim metric in (phi,f,a,d,f1) for internal use */
  REAL8 T;  /* Duration of observation */
  REAL8 f; /* Frequency */
  /* Some useful short-hand notations involving the orbital phase: */
  REAL8 D_p_o;
  REAL8 sin_p_o;
  REAL8 cos_p_o;
  REAL8 D_sin_p_o;
  REAL8 D_cos_p_o;
  REAL8 D_sin_2p_o;
  REAL8 D_cos_2p_o;
  /* Some useful short-hand notations involving the spin phase: */
  REAL8 D_p_s;
  REAL8 sin_p_s;
  REAL8 cos_p_s;
  REAL8 D_sin_p_s;
  REAL8 D_cos_p_s;
  REAL8 D_sin_2p_s;
  REAL8 D_cos_2p_s;
  /* Some useful short-hand notations involving spin and orbital phases: */
  REAL8 D_sin_p_o_plus_s;
  REAL8 D_sin_p_o_minus_s;
  REAL8 D_cos_p_o_plus_s;
  REAL8 D_cos_p_o_minus_s;
  /* Quantities related to the short-time Tatlor expansions : */
  REAL8 D_p_o_crit;       /* The orbital phase BELOW which series used.  */
  REAL8 sum1, sum2, sum3;
  REAL8 next1, next2, next3;
  INT2 j_max;  /* Number of terms in series */


  INITSTATUS(status);

  /* Check for valid input structure. */
  ASSERT( input != NULL, status, PTOLEMETRICH_ENULL,
	  PTOLEMETRICH_MSGENULL );

  /* Check for valid sky position. */
  ASSERT( input->position.system == COORDINATESYSTEM_EQUATORIAL, status,
	  PTOLEMETRICH_EPARM, PTOLEMETRICH_MSGEPARM );
  ASSERT( input->position.longitude >= 0, status, PTOLEMETRICH_EPARM,
	  PTOLEMETRICH_MSGEPARM );
  ASSERT( input->position.longitude <= LAL_TWOPI, status,
          PTOLEMETRICH_EPARM, PTOLEMETRICH_MSGEPARM );
  ASSERT( fabs(input->position.latitude) <= LAL_PI_2, status,
	  PTOLEMETRICH_EPARM, PTOLEMETRICH_MSGEPARM );

  /* Check for valid maximum frequency. */
  ASSERT( input->maxFreq > MIN_MAXFREQ, status, PTOLEMETRICH_EPARM,
          PTOLEMETRICH_MSGEPARM );

  /* Check for valid detector location. */
  ASSERT( fabs(input->site->frDetector.vertexLatitudeRadians) <= LAL_PI_2, status,
	  PTOLEMETRICH_EPARM, PTOLEMETRICH_MSGEPARM );
  ASSERT( fabs(input->site->frDetector.vertexLongitudeRadians) <= LAL_PI, status,
	  PTOLEMETRICH_EPARM, PTOLEMETRICH_MSGEPARM );

  if( input->spindown )
    dim = 2+input->spindown->length;
  else
    dim = 2;
  ASSERT( metric != NULL, status, PTOLEMETRICH_ENULL,
	  PTOLEMETRICH_MSGENULL );
  ASSERT( metric->data != NULL, status, PTOLEMETRICH_ENULL,
          PTOLEMETRICH_MSGENULL );
  ASSERT( metric->length == (UINT4)(dim+1)*(dim+2)/2, status, PTOLEMETRICH_EDIM,
          PTOLEMETRICH_MSGEDIM );

  /* A bigger metric that includes phase for internal use only:  */
  /* Apart from normalization, this is just the information matrix. */
  big_metric = NULL;
  LALDCreateVector( status, &big_metric, (dim+2)*(dim+3)/2);

  /* Detector location: */
  lat = input->site->frDetector.vertexLatitudeRadians;
  lon = input->site->frDetector.vertexLongitudeRadians;
  cos_l = cos(lat);
  sin_l = sin(lat);
  sin_2l = sin(2*lat);

  /* Inclination of Earth's spin-axis to ecliptic */
  sin_i = sin(LAL_IEARTH);
  cos_i = cos(LAL_IEARTH);

  /* Radii of circular motions in seconds:  */
  R_s = LAL_REARTH_SI / LAL_C_SI;
  R_o = LAL_AU_SI / LAL_C_SI;

  /* To switch off orbital motion uncomment this: */
  /* R_o = 0.0; */

  /* To switch off spin motion uncomment this: */
  /* R_s = 0.0; */

  /* Angular velocities: */
  omega_s = LAL_TWOPI / LAL_DAYSID_SI;
  omega_o = LAL_TWOPI / LAL_YRSID_SI;

  /* Duration of observation */
  T = input->duration;

  /* Frequency: */
  f = input->maxFreq;

  /* Source RA: */
  sin_a  = sin(input->position.longitude);
  cos_a  = cos(input->position.longitude);
  sin_2a = sin(2*(input->position.longitude));

  /* Source declination: */
  sin_d  = sin(input->position.latitude);
  cos_d  = cos(input->position.latitude);
  sin_2d = sin(2*(input->position.latitude));

  /* Calculation of phases of spin and orbit at start: */
  XLAL_CHECK_LAL( status, XLALGetEarthTimes(&input->epoch, &tMidnight, &tAutumn) == XLAL_SUCCESS, XLAL_EFUNC );
  phi_o_i = -tAutumn/LAL_YRSID_SI*LAL_TWOPI;
  phi_s_i = -tMidnight/LAL_DAYSID_SI*LAL_TWOPI + lon;


  /* Quantities involving the orbital phase: */
  phi_o_f   = phi_o_i + omega_o*T;
  D_p_o     = omega_o*T;
  sin_p_o   = sin(phi_o_f);
  cos_p_o   = cos(phi_o_f);
  D_sin_p_o = (sin(phi_o_f) - sin(phi_o_i))/D_p_o;
  D_cos_p_o = (cos(phi_o_f) - cos(phi_o_i))/D_p_o;
  D_sin_2p_o = (sin(2*phi_o_f) - sin(2*phi_o_i))/2.0/D_p_o;
  D_cos_2p_o = (cos(2*phi_o_f) - cos(2*phi_o_i))/2.0/D_p_o;

  /* Quantities involving the spin phase: */
  phi_s_f    = phi_s_i + omega_s*T;
  D_p_s      = omega_s*T;
  sin_p_s    = sin(phi_s_f);
  cos_p_s    = cos(phi_s_f);
  D_sin_p_s  = (sin(phi_s_f) - sin(phi_s_i))/D_p_s;
  D_cos_p_s  = (cos(phi_s_f) - cos(phi_s_i))/D_p_s;
  D_sin_2p_s = (sin(2*phi_s_f) - sin(2*phi_s_i))/2.0/D_p_s;
  D_cos_2p_s = (cos(2*phi_s_f) - cos(2*phi_s_i))/2.0/D_p_s;

  /* Some mixed quantities: */
  D_sin_p_o_plus_s
    = (sin(phi_o_f+phi_s_f) - sin(phi_o_i+phi_s_i))/(D_p_o + D_p_s);
  D_sin_p_o_minus_s
    = (sin(phi_o_f-phi_s_f) - sin(phi_o_i-phi_s_i))/(D_p_o - D_p_s);
  D_cos_p_o_plus_s
    = (cos(phi_o_f+phi_s_f) - cos(phi_o_i+phi_s_i))/(D_p_o + D_p_s);
  D_cos_p_o_minus_s
    = (cos(phi_o_f-phi_s_f) - cos(phi_o_i-phi_s_i))/(D_p_o - D_p_s);



  /***************************************************************/
  /* Small D_p_o overwrite:                                      */
  /***************************************************************/
  j_max = 7;
  D_p_o_crit = 1.4e-2; /* This corresponds to about 70000 seconds */

  sum1  = next1 = D_p_o/2.0;
  sum2  = next2 = D_p_o/3.0/2.0;
  sum3  = next3 = D_p_o;

  for(j=1; j<=j_max; j++)
    {
      next1 *= -pow(D_p_o,2.0)/(2.0*j+1.0)/(2.0*j+2.0);
      sum1  += next1;
      next2 *= -pow(D_p_o,2.0)/(2.0*j+2.0)/(2.0*j+3.0);
      sum2  += next2;
      next3 *= -pow(2.0*D_p_o,2.0)/(2.0*j+1.0)/(2.0*j+2.0);
      sum3  += next3;
    }

  if(D_p_o < D_p_o_crit)
    {
      D_sin_p_o = sin(phi_o_f)*sum1 + cos(phi_o_f)*sin(D_p_o)/D_p_o;
      D_cos_p_o = cos(phi_o_f)*sum1 - sin(phi_o_f)*sin(D_p_o)/D_p_o;
      D_sin_2p_o
	= sin(2.0*phi_o_f)*sum3 + cos(2.0*phi_o_f)*sin(2.0*D_p_o)/2.0/D_p_o;
      D_cos_2p_o
	= cos(2.0*phi_o_f)*sum3 - sin(2.0*phi_o_f)*sin(2.0*D_p_o)/2.0/D_p_o;
   }
  /****************************************************************/


  /* The A[i] quantities: */
  A[1] =
    R_o*D_sin_p_o + R_s*cos_l*D_sin_p_s;

  A[2] =
    R_o*cos_i*D_cos_p_o + R_s*cos_l*D_cos_p_s;

  A[3] =
    -R_o*sin_i*D_cos_p_o + R_s*sin_l;

  A[4] =
    R_o*(sin_p_o/D_p_o + D_cos_p_o/D_p_o);

  A[5] =
    R_s*(sin_p_s/D_p_s + D_cos_p_s/D_p_s);

  A[6] =
    R_o*(-cos_p_o/D_p_o + D_sin_p_o/D_p_o);

  /* Special overwrite for A4 and A6: *********************/
  if(D_p_o < D_p_o_crit)
    {
      A[4] = R_o*(cos(phi_o_f)*sum1/D_p_o + sin(phi_o_f)*sum2);
      A[6] = R_o*(sin(phi_o_f)*sum1/D_p_o - cos(phi_o_f)*sum2);
    }
  /***********************************************************/

  A[7] =
    R_s*(-cos_p_s/D_p_s + D_sin_p_s/D_p_s);

  A[8] =
    R_o*R_o*(1 + D_sin_2p_o);

  A[9] =
    R_o*R_s*(D_sin_p_o_minus_s + D_sin_p_o_plus_s);

  A[10] =
    R_s*R_s*(1 + D_sin_2p_s);

  A[11] =
    R_o*R_o*D_cos_2p_o;

  A[12] =
    R_o*R_s*(-D_cos_p_o_minus_s + D_cos_p_o_plus_s);

  A[13] =
    R_o*R_s*(D_cos_p_o_minus_s + D_cos_p_o_plus_s);

  A[14] =
    R_s*R_s*D_cos_2p_s;

  A[15] =
    R_o*R_o*(1 - D_sin_2p_o);

  A[16] =
    R_o*R_s*(D_sin_p_o_minus_s - D_sin_p_o_plus_s);

  A[17] =
    R_s*R_s*(1 - D_sin_2p_s);

  A[18] =
    R_o*R_s*D_sin_p_o;

  A[19] =
    R_s*R_s*D_sin_p_s;

  A[20] =
    R_o*R_s*D_cos_p_o;

  A[21] =
    R_s*R_s*D_cos_p_s;


  /* The B[i] quantities: */
  B[1] =
    A[4] + A[5]*cos_l;

  B[2] =
    A[6]*cos_i + A[7]*cos_l;

  B[3] =
    A[6]*sin_i + R_s*sin_l/2;

  B[4] =
    A[8] + 2*A[9]*cos_l + A[10]*cos_l*cos_l;

  B[5] =
    A[11]*cos_i + A[12]*cos_l + A[13]*cos_i*cos_l + A[14]*cos_l*cos_l;

  B[6] =
    A[15]*cos_i*cos_i +2*A[16]*cos_i*cos_l + A[17]*cos_l*cos_l;

  B[7] =
    -A[11]*sin_i + 2*A[18]*sin_l - A[13]*sin_i*cos_l + A[19]*sin_2l;

  B[8] =
    A[15]*sin_i*cos_i - 2*A[20]*cos_i*sin_l + A[16]*sin_i*cos_l - A[21]*sin_2l;

  B[9] =
    A[15]*sin_i*sin_i - 4*A[20]*sin_i*sin_l + 2*R_s*R_s*sin_l*sin_l;

 /* The spindown integrals. */

  /* orbital t^2 cos */

  Is[1] = (2*sin(phi_o_i) + 2*omega_o*T*cos_p_o + (-2 + pow(omega_o*T,2))*sin_p_o)/pow(omega_o*T,3);

  /* spin t^2 cos */
  Is[2] = (2*sin(phi_s_i) + 2*omega_s*T*cos_p_s + (-2 + pow(omega_s*T,2))*sin_p_s)/pow(omega_s*T,3);

  /* orbital t^2 sin */
  Is[3] = (-2*cos(phi_o_i) + 2*omega_o*T*sin_p_o - (-2 + pow(omega_o*T,2))*cos_p_o)/pow(omega_o*T,3);

  /*spin t^2 sin */

  Is[4] = (-2*cos(phi_s_i) + 2*omega_s*T*sin_p_s - (-2 + pow(omega_s*T,2))*cos_p_s)/pow(omega_s*T,3);



  /* The 4-dim metric components: */
  /* g_pp = */
  big_metric->data[0] =
    1;

  /* g_pf = */
  big_metric->data[1] =
    LAL_PI*T;

  /* g_pa = */
  big_metric->data[3] =
    -LAL_TWOPI*f*cos_d*(A[1]*sin_a + A[2]*cos_a);

  /* g_pd = */
  big_metric->data[6] =
    LAL_TWOPI*f*(-A[1]*sin_d*cos_a + A[2]*sin_d*sin_a + A[3]*cos_d);

  /* g_ff = */
  big_metric->data[2] =
    pow(LAL_TWOPI*T, 2)/3;

  /* g_fa = */
  big_metric->data[4] =
    pow(LAL_TWOPI,2)*f*cos_d*T*(-B[1]*sin_a + B[2]*cos_a);

  /* g_fd = */
  big_metric->data[7] =
    pow(LAL_TWOPI,2)*f*T*(-B[1]*sin_d*cos_a - B[2]*sin_d*sin_a + B[3]*cos_d);

  /* g_aa = */
  big_metric->data[5] =
    2*pow(LAL_PI*f*cos_d,2) * (B[4]*sin_a*sin_a + B[5]*sin_2a + B[6]*cos_a*cos_a);

  /* g_ad =  */
  big_metric->data[8] =
    2*pow(LAL_PI*f,2)*cos_d*(B[4]*sin_a*cos_a*sin_d - B[5]*sin_a*sin_a*sin_d
			     -B[7]*sin_a*cos_d + B[5]*cos_a*cos_a*sin_d - B[6]*sin_a*cos_a*sin_d
			     +B[8]*cos_a*cos_d);

  /* g_dd = */
  big_metric->data[9] =
    2*pow(LAL_PI*f,2)*(B[4]*pow(cos_a*sin_d,2) + B[6]*pow(sin_a*sin_d,2)
		       +B[9]*pow(cos_d,2) - B[5]*sin_2a*pow(sin_d,2) - B[8]*sin_a*sin_2d
		       -B[7]*cos_a*sin_2d);

  /*The spindown components*/
  if(dim==3) {
    /* g_p1 = */
    big_metric->data[10] =
      T*LAL_PI*f*T/3;

    /* g_f1= */
    big_metric->data[11]=
      T*pow(LAL_PI*T,2)*f/2;

    /* g_a1 = */
    big_metric->data[12] = T*2*pow(LAL_PI*f,2)*T*(-cos_d*sin_a*(R_o*Is[1] + R_s*cos_l*Is[2])+ cos_d*cos_a*(R_o*cos_i*Is[3] + R_s*cos_l*Is[4]));

    /* g_d1 = */
    big_metric->data[13] = T*2*pow(LAL_PI*f,2)*T*(-sin_d*cos_a*(R_o*Is[1] + R_s*cos_l*Is[2])- sin_d*sin_a*(R_o*cos_i*Is[3] + R_s*cos_l*Is[4]) + cos_d*(R_o*sin_i*Is[3] + R_s*sin_l/3));

    /* g_11 = */
    big_metric->data[14] = T*T*pow(LAL_PI*f*T,2)/5;
  }


  /**********************************************************/
  /* Spin-down stuff not consistent with rest of code - don't uncomment! */
  /* Spindown-spindown metric components, before projection
     if( input->spindown )
     for (j=1; j<=dim-2; j++)
     for (k=1; k<=j; k++)
     metric->data[(k+2)+(j+2)*(j+3)/2] = pow(LAL_TWOPI*input->maxFreq
     *input->duration,2)/(j+2)/(k+2)/(j+k+3);

     Spindown-angle metric components, before projection
     if( input->spindown )
     for (j=1; j<=(INT4)input->spindown->length; j++) {

     Spindown-RA: 1+(j+2)*(j+3)/2
     metric->data[1+(j+2)*(j+3)/2] = 0;
     temp1=0;
     temp2=0;
     for (k=j+1; k>=0; k--) {
     metric->data[1+(j+2)*(j+3)/2] += pow(-1,(k+1)/2)*factrl(j+1)
     /factrl(j+1-k)/pow(D_p_s,k)*((k%2)?sin_p_s:cos_p_s);
     metric->data[1+(j+2)*(j+3)/2] += pow(-1,j/2)/pow(D_p_s,j+1)*factrl(j+1)
     *((j%2)?cos(phi_s_i):sin(phi_s_i));
     metric->data[1+(j+2)*(j+3)/2] -= (cos_p_s-cos(phi_s_i))/(j+2);
     metric->data[1+(j+2)*(j+3)/2] *= -pow(LAL_TWOPI*input->maxFreq,2)*R_s*cos_l
     *cos_d/omega_s/(j+1);
     temp1+=pow(-1,(k+1)/2)*factrl(j+1)
     /factrl(j+1-k)/pow(D_p_s,k)*((k%2)?sin_p_s:cos_p_s);
     temp2+=pow(-1,(k+1)/2)*factrl(j+1)
     /factrl(j+1-k)/pow(D_p_o,k)*((k%2)?sin_p_o:cos_p_o);
     }
     temp1+=pow(-1,j/2)/pow(D_p_s,j+1)*factrl(j+1)
     *((j%2)?cos(phi_s_i):sin(phi_s_i));
     temp2+=pow(-1,j/2)/pow(D_p_o,j+1)*factrl(j+1)
     *((j%2)?cos(phi_o_i):sin(phi_o_i));
     temp1-=(cos_p_s-cos(phi_s_i))/(j+2);
     temp2-=(cos_p_o-cos(phi_o_i))/(j+2);
     temp1*=-pow(LAL_TWOPI*input->maxFreq,2)*R_s*cos_l
     *cos_d/omega_s/(j+1);
     temp2*=-pow(LAL_TWOPI*input->maxFreq,2)*R_o*cos_i
     *cos_d/omega_o/(j+1);
     metric->data[1+(j+2)*(j+3)/2]+=temp1+temp2;
     Spindown-dec: 2+(j+2)*(j+3)/2
     metric->data[2+(j+2)*(j+3)/2] = 0;
     temp3=0;
     temp4=0;
     for (k=j+1; k>=0; k--) {
     metric->data[2+(j+2)*(j+3)/2] -= pow(-1,k/2)*factrl(j+1)/factrl(j+1-k)
     /pow(D_p_s,k)*((k%2)?cos_p_s:sin_p_s);
     metric->data[2+(j+2)*(j+3)/2] += pow(-1,(j+1)/2)/pow(D_p_s,j+1)
     *factrl(j+1)*((j%2)?sin(phi_s_i):cos(phi_s_i));
     metric->data[2+(j+2)*(j+3)/2] += (sin_p_s-sin(phi_s_i))/(j+2);
     metric->data[2+(j+2)*(j+3)/2] *= pow(LAL_TWOPI*input->maxFreq,2)*R_s*cos_l
     *sin_d/omega_s/(j+1);
     }    for( j... )
     temp3-=pow(-1,k/2)*factrl(j+1)/factrl(j+1-k)
     /pow(D_p_s,k)*((k%2)?cos_p_s:sin_p_s);
     temp4-=pow(-1,k/2)*factrl(j+1)/factrl(j+1-k)
     /pow(D_p_o,k)*((k%2)?cos_p_o:sin_p_o);
     }
     temp3+=pow(-1,(j+1)/2)/pow(D_p_s,j+1)
     *factrl(j+1)*((j%2)?sin(phi_s_i):cos(phi_s_i));
     temp4+=pow(-1,(j+1)/2)/pow(D_p_o,j+1)
     *factrl(j+1)*((j%2)?sin(phi_o_i):cos(phi_o_i));
     temp3+=(sin_p_s-sin(phi_s_i))/(j+2);
     temp4+=(sin_p_o-sin(phi_o_i))/(j+2);
     temp3*=pow(LAL_TWOPI*input->maxFreq,2)*R_s*cos_l
     *sin_d/omega_s/(j+1);
     temp4*=pow(LAL_TWOPI*input->maxFreq,2)*R_s*cos_i
     *sin_d/omega_o/(j+1);
     metric->data[2+(j+2)*(j+3)/2]=temp3+temp4;
     }
     f0-spindown : 0+(j+2)*(j+3)/2
     if( input->spindown )
     for (j=1; j<=dim-2; j++)
     metric->data[(j+2)*(j+3)/2] = 2*pow(LAL_PI,2)*input->maxFreq
     *pow(input->duration,2)/(j+2)/(j+3);*/
/*************************************************************/


/* Project down to 4-dim metric: */

/*f-f component */
  metric->data[0] =  big_metric->data[2]
    - big_metric->data[1]*big_metric->data[1]/big_metric->data[0];
  /*f-a component */
  metric->data[1] =  big_metric->data[4]
    - big_metric->data[1]*big_metric->data[3]/big_metric->data[0];
  /*a-a component */
  metric->data[2] =  big_metric->data[5]
    - big_metric->data[3]*big_metric->data[3]/big_metric->data[0];
  /*f-d component */
  metric->data[3] =  big_metric->data[7]
    - big_metric->data[6]*big_metric->data[1]/big_metric->data[0];
  /*a-d component */
  metric->data[4] =  big_metric->data[8]
    - big_metric->data[6]*big_metric->data[3]/big_metric->data[0];
  /*d-d component */
  metric->data[5] =  big_metric->data[9]
    - big_metric->data[6]*big_metric->data[6]/big_metric->data[0];
  if(dim==3) {

    /*f-f1 component */
    metric->data[6] = big_metric->data[11]
      - big_metric->data[1]*big_metric->data[10]/big_metric->data[0];

    /*a-f1 component */
    metric->data[7] = big_metric->data[12]
      - big_metric->data[3]*big_metric->data[10]/big_metric->data[0];

    /*d-f1 component */
    metric->data[8] = big_metric->data[13]
      - big_metric->data[6]*big_metric->data[10]/big_metric->data[0];

    /* f1-f1 component */
    metric->data[9] = big_metric->data[14]
      - big_metric->data[10]*big_metric->data[10]/big_metric->data[0];
  }

  LALDDestroyVector( status, &big_metric );
  /* All done */
  RETURN( status );
} /* LALPtoleMetric() */


/* This is a dead simple, no error-checking, private factorial function. */
/* static int factrl( int arg ) */
/* { */
/*   int ans = 1; */

/*   if (arg==0) return 1; */
/*   do { */
/*     ans *= arg; */
/*   } */
/*   while(--arg>0); */
/*   return ans; */
/* } */ /* factrl() */


/**
 * \brief Unified "wrapper" to provide a uniform interface to LALPtoleMetric() and LALCoherentMetric().
 * \author Reinhard Prix
 *
 * The parameter structure of LALPtoleMetric() was used, because it's more compact.
 */
void LALPulsarMetric ( LALStatus *stat,
		       REAL8Vector **metric,
		       PtoleMetricIn *input )
{
  UINT4 nSpin, dim;

  INITSTATUS(stat);
  ATTATCHSTATUSPTR (stat);

  ASSERT ( input, stat, PTOLEMETRICH_ENULL, PTOLEMETRICH_MSGENULL );
  ASSERT ( metric != NULL, stat, PTOLEMETRICH_ENULL, PTOLEMETRICH_MSGENULL );
  ASSERT ( *metric == NULL, stat, PTOLEMETRICH_ENONULL, PTOLEMETRICH_MSGENONULL );

  if ( input->spindown )
    nSpin = input->spindown->length;
  else
    nSpin = 0;


  /* allocate the output-metric */
  dim = 3 + nSpin;	/* dimensionality of parameter-space: Alpha,Delta,f + spindowns */

  TRY ( LALDCreateVector (stat->statusPtr, metric, dim * (dim+1)/2), stat);

  switch (input->metricType)
    {
    case LAL_PMETRIC_COH_PTOLE_ANALYTIC: /* use Ben&Ian's analytic ptolemaic metric */
      LALPtoleMetric (stat->statusPtr, *metric, input);
      BEGINFAIL(stat) {
	LALDDestroyVector (stat->statusPtr, metric);
      }ENDFAIL(stat);
      break;

    default:
      XLALPrintError ("Unknown metric type `%d`\n", input->metricType);
      ABORT (stat, PTOLEMETRICH_EMETRIC,  PTOLEMETRICH_MSGEMETRIC);
      break;

    } /* switch type */

  DETATCHSTATUSPTR (stat);
  RETURN (stat);

} /* LALPulsarMetric() */


/**
 * \brief Project out the zeroth dimension of a metric.
 * \author Creighton, T. D.
 * \date 2000
 *
 * ### Description ###
 *
 * This function takes a metric \f$g_{\alpha\beta}\f$, where
 * \f$\alpha,\beta=0,1,\ldots,n\f$, and computes the projected metric
 * \f$\gamma_{ij}\f$ on the subspace \f$i,j=1,\ldots,n\f$, as described in the
 * header StackMetric.h.
 *
 * The argument \a *metric stores the metric components in the manner
 * used by the functions LALCoherentMetric() and
 * LALStackMetric(), and \a errors indicates whether error
 * estimates are included in \a *metric.  Thus \a *metric is a
 * vector of length \f$(n+1)(n+2)/2\f$ if \a errors is zero, or of length
 * \f$(n+1)(n+2)\f$ if \a errors is nonzero; see LALCoherentMetric()
 * for the indexing scheme.
 *
 * Upon return, \a *metric stores the components of \f$\gamma_{ij}\f$ in
 * the same manner as above, with the physically meaningless components
 * \f$\gamma_{\alpha0} = \gamma_{0\alpha}\f$ (and their uncertainties) set
 * identically to zero.
 *
 * ### Algorithm ###
 *
 * The function simply implements \eqref{eq_gij_gab} in
 * StackMetric.h.  The formula used to convert uncertainties
 * \f$s_{\alpha\beta}\f$ in the metric components \f$g_{\alpha\beta}\f$ into
 * uncertainties \f$\sigma_{ij}\f$ in \f$\gamma_{ij}\f$ is:
 * \f[
 * \sigma_{ij} = s_{ij}
 * + s_{0i}\left|\frac{g_{0j}}{g_{00}}\right|
 * + s_{0j}\left|\frac{g_{0i}}{g_{00}}\right|
 * + s_{00}\left|\frac{g_{0i}g_{0j}}{(g_{00})^2}\right| \; .
 * \f]
 * Note that if the metric is highly degenerate, one may find that one or
 * more projected components are comparable in magnitude to their
 * estimated numerical uncertainties.  This can occur when the
 * observation times are very short or very long compared to the
 * timescales over which the timing derivatives are varying.  In the
 * former case, one is advised to use analytic approximations or a
 * different parameter basis.  In the latter case, the degenerate
 * components are often not relevant for data analysis, and can be
 * effectively set to zero.
 *
 * Technically, starting from a full metric
 * \f$g_{\alpha\beta}(\mathbf{\lambda})\f$, the projection
 * \f$\gamma_{ij}(\vec\lambda)\f$ is the metric of a subspace
 * \f$\{\vec\lambda\}\f$ passing through the point \f$\mathbf{\lambda}\f$ on a plane
 * orthogonal to the \f$\lambda^0\f$ axis.  In order for \f$\gamma_{ij}\f$ to
 * measure the \em maximum distance between points \f$\vec\lambda\f$, it
 * is important to evaluate \f$g_{\alpha\beta}\f$ at the value of \f$\lambda^0\f$
 * that gives the largest possible separations.  For the pulsar search
 * formalism discussed in StackMetric.h, this is always
 * achieved by choosing the largest value of \f$\lambda^0=f_\mathrm{max}\f$
 * that is to be covered in the search.
 */
void
LALProjectMetric( LALStatus *stat, REAL8Vector *metric, BOOLEAN errors )
{
  UINT4 s;     /* The number of parameters before projection. */
  UINT4 i, j;  /* Indecies. */
  REAL8 *data; /* The metric data array. */

  INITSTATUS(stat);

  /* Check that data exist. */
  ASSERT(metric,stat,PTOLEMETRICH_ENULL,PTOLEMETRICH_MSGENULL);
  ASSERT(metric->data,stat,PTOLEMETRICH_ENULL,PTOLEMETRICH_MSGENULL);
  data=metric->data;

  /* Make sure the metric's length is compatible with some
     dimensionality s. */
  for(s=0,i=1;i<metric->length;s++){
    i=s*(s+1);
    if(!errors)
      i=i>>1;
  }
  s--; /* counteract the final s++ */
  ASSERT(i==metric->length,stat,PTOLEMETRICH_EPARM,
	 PTOLEMETRICH_MSGEPARM);

  /* Now project out the zeroth component of the metric. */
  for(i=1;i<s;i++)
    for(j=1;j<=i;j++){
      INT4 i0 = (i*(i+1))>>1;
      INT4 myj0 = (j*(j+1))>>1;
      INT4 ij = i0+j;
      if(errors){
	data[2*ij]-=data[2*i0]*data[2*myj0]/data[0];
	data[2*ij+1]+=data[2*i0+1]*fabs(data[2*myj0]/data[0])
	  + data[2*myj0+1]*fabs(data[2*i0]/data[0])
	  + data[1]*fabs(data[2*i0]/data[0])*fabs(data[2*myj0]/data[0]);
      } else
	data[ij]-=data[i0]*data[myj0]/data[0];
    }

  /* Set all irrelevant metric coefficients to zero. */
  for(i=0;i<s;i++){
    INT4 i0 = i*(i+1)>>1;
    if(errors)
      data[2*i0]=data[2*i0+1]=0.0;
    else
      data[i0]=0.0;
  }

  /* And that's it! */
  RETURN(stat);
}


/**
 * \brief Figure out dimension of a REAL8Vector -encoded metric (see PMETRIC_INDEX() ).
 * Return error if input-vector is NULL or not consistent with a quadratic matrix.
 */
int
XLALFindMetricDim ( const REAL8Vector *metric )
{
  UINT4 dim;
  UINT4 length;
  UINT4 trylength;

  if ( !metric )
    {
      XLALPrintError ("\nNULL Input received!\n\n");
      XLAL_ERROR ( XLAL_EINVAL);
    }

  length = metric->length;
  if ( length == 0 )
    return 0;

  dim=1;
  while ( (trylength = dim * (dim + 1)/2 ) <= length )
    {
      if ( length == trylength )
	return dim;
      else
	dim ++;
    }

  /* no fitting dimension found ==> error */
  XLALPrintError ("\nInput vector is inconsisten with symmetric quadratic matrix!\n\n");
  XLAL_ERROR ( XLAL_EINVAL);

}/* XLALFindMetricDim() */

/**
 * Frequency and frequency derivative components of the metric, suitable for a directed
 * search with only one fixed sky position. The units are those expected by ComputeFstat.
 */
int XLALSpindownMetric(
  gsl_matrix* metric,	/**< [in] Matrix containing the metric */
  double Tspan		/**< [in] Time span of the data set */
  )
{

  // Check input
  XLAL_CHECK(metric != NULL, XLAL_EFAULT);
  XLAL_CHECK(metric->size1 == metric->size2, XLAL_ESIZE);
  XLAL_CHECK(Tspan > 0, XLAL_EINVAL);

  // Calculate metric
  for (size_t i = 0; i < metric->size1; ++i) {
    for (size_t j = i; j < metric->size2; ++j) {
      gsl_matrix_set(metric, i, j, (
                       4.0 * LAL_PI * LAL_PI * pow(Tspan, i + j + 2) * (i + 1) * (j + 1)
                       ) / (
                         LAL_FACT[i + 1] * LAL_FACT[j + 1] * (i + 2) * (j + 2) * (i + j + 3)
                         ));
      gsl_matrix_set(metric, j, i, gsl_matrix_get(metric, i, j));
    }
  }

  return XLAL_SUCCESS;

} // XLALSpindownMetric

/*@}*/

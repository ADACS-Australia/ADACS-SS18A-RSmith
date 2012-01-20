/*
 *  Copyright (C) 2011 Evan Goetz
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


#ifndef __FASTCHISQINV_H__
#define __FASTCHISQINV_H__

#include <lal/LALStdlib.h>

struct cheb_series_struct {
   double * c;   /* coefficients                */
   int order;    /* order of expansion          */
   double a;     /* lower interval point        */
   double b;     /* upper interval point        */
   int order_sp; /* effective single precision order */
};
typedef struct cheb_series_struct cheb_series;

REAL8 cdf_chisq_Pinv(REAL8 P, REAL8 nu);
REAL8 cdf_chisq_Qinv(REAL8 Q, REAL8 nu);
REAL8 cdf_gamma_Pinv(REAL8 P, REAL8 a, REAL8 b);
REAL8 cdf_gamma_Qinv(REAL8 Q, REAL8 a, REAL8 b);
REAL8 cdf_ugaussian_Pinv(REAL8 P);
REAL8 cdf_ugaussian_Qinv(REAL8 Q);
REAL8 twospect_small(REAL8 q);
REAL8 twospect_intermediate(REAL8 r);
REAL8 twospect_tail(REAL8 r);
REAL8 rat_eval(const REAL8 a[], const size_t na, const REAL8 b[], const size_t nb, const REAL8 x);
REAL8 cdf_gamma_P(REAL8 x, REAL8 a, REAL8 b);
REAL8 cdf_gamma_P_usingmatlab(REAL8 x, REAL8 a, REAL8 b);
REAL8 cdf_gamma_Q(REAL8 x, REAL8 a, REAL8 b);
REAL8 cdf_gamma_Q_usingmatlab(REAL8 x, REAL8 a, REAL8 b);
REAL8 ran_gamma_pdf(REAL8 x, REAL8 a, REAL8 b);
REAL8 sf_gamma_inc_P(REAL8 a, REAL8 x);
REAL8 sf_gamma_inc_Q(REAL8 a, REAL8 x);
REAL8 matlab_gamma_inc(REAL8 x, REAL8 a, INT4 upper);
REAL8 gamma_inc_P_series(REAL8 a, REAL8 x);
REAL8 gamma_inc_Q_series(REAL8 a, REAL8 x);
REAL8 gamma_inc_D(REAL8 a, REAL8 x);
REAL8 twospect_sf_gammastar(REAL8 x);
REAL8 twospect_cheb_eval(const cheb_series * cs, REAL8 x);
REAL8 gammastar_ser(REAL8 x);
REAL8 sf_exprel_n_CF(REAL8 N, REAL8 x);
REAL8 gamma_inc_Q_asymp_unif(REAL8 a, REAL8 x);
REAL8 gamma_inc_Q_CF(REAL8 a, REAL8 x);
REAL8 gamma_inc_F_CF(REAL8 a, REAL8 x);
REAL8 gamma_inc_Q_large_x(REAL8 a, REAL8 x);



#endif

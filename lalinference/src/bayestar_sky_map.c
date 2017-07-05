/*                                           >y#
                                            ~'#o+
                                           '~~~md~
                '|+>#!~'':::::....        .~'~'cY#
            .+oy+>|##!~~~''':::......     ~:'':md! .
          #rcmory+>|#~''':::'::...::.::. :..'''Yr:...
        'coRRaamuyb>|!~'''::::':...........  .+n|.::..
       !maMMNMRYmuybb|!~'''':.........::::::: ro'..::..
      .cODDMYouuurub!':::...........:::~'.. |o>::...:..
      >BDNCYYmroyb>|#~:::::::::::::~':.:: :ob::::::::..
      uOCCNAa#'''''||':::.                :oy':::::::::.
    :rRDn!  :~::'y+::':  ... ...:::.     :ob':::::::::::.
   yMYy:   :>yooCY'.':.   .:'':......    ~u+~::::::::::::.
  >>:'. .~>yBDMo!.'': . .:'':.   .      >u|!:::::::::::::.
    ':'~|mYu#:'~'''. :.~:':...         yy>|~:::::::::::::..
    :!ydu>|!rDu::'. +'#~::!#'.~:     |r++>#':::::::::::::..
    mn>>>>>YNo:'': !# >'::::...  ..:cyb++>!:::::::::..:::...
    :ouooyodu:'': .!:.!:::.       yobbbb+>~::::::::....:....
     'cacumo~''' .'~ :~'.::.    :aybbbbbb>':::'~''::::....
      .mamd>'''. :~' :':'.:.   om>bbbyyyb>'.#b>|#~~~'':..
      .yYYo''': .:~' .'::'   .ny>+++byyoao!b+|||#!~~~''''''::.
      .#RUb:''. .:'' .:':   |a#|>>>>yBMdb #yb++b|':::::''':'::::::.
      .'CO!'''  .:'' .'    uu~##|+mMYy>+:|yyo+:::'::.         .::::::
      .:RB~''' ..::'.':   o>~!#uOOu>bby'|yB>.'::  '~!!!!!~':. ..  .::::
       :Rm''': ..:~:!:  'c~~+YNnbyyybb~'mr.':  !+yoy+>||!~'::.       :::.
      ..Oo''': .'' ~:  !+|BDCryuuuuub|#B!::  !rnYaocob|#!~'':.  ..    .::.
      . nB''': :  .'  |dNNduroomnddnuun::.  ydNAMMOary+>#~:.:::...      .:
       .uC~'''    :. yNRmmmadYUROMMBmm.:   bnNDDDMRBoy>|#~':....:.      .:
                 :' ymrmnYUROMAAAAMYn::. .!oYNDDMYmub|!~'::....:..     :
                 !'#booBRMMANDDDNNMO!:. !~#ooRNNAMMOOmuy+#!':::.......    :.
                .!'!#>ynCMNDDDDDNMRu.. '|:!raRMNAMOOdooy+|!~:::........   .:
                 : .'rdbcRMNNNNAMRB!:  |!:~bycmdYYBaoryy+|!~':::.::::.  ..
                 ..~|RMADnnONAMMRdy:. .>#::yyoroccruuybb>#!~'::':...::.
                  :'oMOMOYNMnybyuo!.  :>#::b+youuoyyy+>>|!~':.    :::::
                  ''YMCOYYNMOCCCRdoy##~~~: !b>bb+>>>||#~:..:::     ::::.
                  .:OMRCoRNAMOCROYYUdoy|>~:.~!!~!~~':...:'::::.   :::::.
                  ''oNOYyMNAMMMRYnory+|!!!:.....     ::.  :'::::::::::::
                 .:..uNabOAMMCOdcyb+|!~':::.          !!'.. :~:::::'''':.
                  .   +Y>nOORYauyy>!!'':....           !#~..  .~:''''''':.

****************  ____  _____  ______________________    ____     **************
***************  / __ )/   \ \/ / ____/ ___/_  __/   |  / __ \   ***************
**************  / __  / /| |\  / __/  \__ \ / / / /| | / /_/ /  ****************
*************  / /_/ / ___ |/ / /___ ___/ // / / ___ |/ _, _/  *****************
************  /_____/_/  |_/_/_____//____//_/ /_/  |_/_/ |_|  ******************
*/


/*
 * Copyright (C) 2013-2017  Leo Singer
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with with program; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA  02111-1307  USA
 */

#include "config.h"
#include "bayestar_cosmology.h"
#include "bayestar_sky_map.h"
#include "bayestar_distance.h"
#include "bayestar_moc.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <lal/cubic_interp.h>
#include <lal/DetResponse.h>
#include <lal/InspiralInjectionParams.h>
#include <lal/FrequencySeries.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALSimulation.h>

#include <chealpix.h>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_test.h>

#include "logaddexp.h"

#ifndef _OPENMP
#define omp ignore
#endif


/* Compute |z|^2. Hopefully a little faster than gsl_pow_2(cabs(z)), because no
 * square roots are necessary. */
static double cabs2(double complex z) {
    return gsl_pow_2(creal(z)) + gsl_pow_2(cimag(z));
}


/*
 * Catmull-Rom cubic spline interpolant of x(t) for regularly gridded
 * samples x_i(t_i), assuming:
 *
 *     t_0 = -1, x_0 = x[0],
 *     t_1 = 0,  x_1 = x[1],
 *     t_2 = 1,  x_2 = x[2],
 *     t_3 = 2,  x_3 = x[3].
 */
static double complex complex_catrom(
    float complex x0,
    float complex x1,
    float complex x2,
    float complex x3,
    double t
) {
    return x1
        + t*(-0.5*x0 + 0.5*x2
        + t*(x0 - 2.5*x1 + 2.*x2 - 0.5*x3
        + t*(-0.5*x0 + 1.5*x1 - 1.5*x2 + 0.5*x3)));
}


/* Evaluate a complex time series using cubic spline interpolation, assuming
 * that the vector x gives the samples of the time series at times
 * 0, 1, ..., nsamples-1. */
static double complex eval_snr(
    const float complex *x,
    size_t nsamples,
    double t
) {
    ssize_t i;
    double f;
    double complex y;

    /* Break |t| into integer and fractional parts. */
    {
        double dbl_i;
        f = modf(t, &dbl_i);
        i = dbl_i;
    }

    if (i >= 1 && i < (ssize_t)nsamples - 2)
        y = complex_catrom(x[i-1], x[i], x[i+1], x[i+2], f);
    else
        y = 0;

    return y;
}


typedef struct {
    bicubic_interp *region0;
    cubic_interp *region1;
    cubic_interp *region2;
    double xmax, ymax, vmax, r1, r2;
    int k;
} log_radial_integrator;


typedef struct {
    double scale;
    double p;
    double b;
    int k, cosmology;
} radial_integrand_params;


/* Uniform-in-comoving volume prior for the WMAP9 cosmology.
 * This is implemented as a cubic spline interpolant.
 *
 * The following static variables are defined in bayestar_cosmology.h, which
 * is automatically generated by bayestar_cosmology.py:
 *     - dVC_dVL_data
 *     - dVC_dVL_tmin
 *     - dVC_dVL_dt
 *     - dVC_dVL_high_z_slope
 *     - dVC_dVL_high_z_intercept
 */
static gsl_spline *dVC_dVL_interp = NULL;
static void dVC_dVL_init(void)
{
    const size_t len = sizeof(dVC_dVL_data) / sizeof(*dVC_dVL_data);
    dVC_dVL_interp = gsl_spline_alloc(gsl_interp_cspline, len);
    assert(dVC_dVL_interp);
    double x[len];
    for (size_t i = 0; i < len; i ++)
        x[i] = dVC_dVL_tmin + i * dVC_dVL_dt;
    int ret = gsl_spline_init(dVC_dVL_interp, x, dVC_dVL_data, len);
    assert(ret == GSL_SUCCESS);
}


static double log_dVC_dVL(double DL)
{
    const double log_DL = log(DL);
    if (log_DL <= dVC_dVL_tmin)
    {
        return 0.0;
    } else if (log_DL >= dVC_dVL_tmax) {
        return dVC_dVL_high_z_slope * log_DL + dVC_dVL_high_z_intercept;
    } else {
        return gsl_spline_eval(dVC_dVL_interp, log_DL, NULL);
    }
}


static double radial_integrand(double r, void *params)
{
    const radial_integrand_params *integrand_params = params;
    const double scale = integrand_params->scale;
    const double p = integrand_params->p;
    const double b = integrand_params->b;
    const int k = integrand_params->k;
    double ret = scale - gsl_pow_2(p / r - 0.5 * b / p);
    if (integrand_params->cosmology)
        ret += log_dVC_dVL(r);
    return gsl_sf_exp_mult(
        ret, gsl_sf_bessel_I0_scaled(b / r) * gsl_pow_int(r, k));
}


static double log_radial_integrand(double r, void *params)
{
    const radial_integrand_params *integrand_params = params;
    const double scale = integrand_params->scale;
    const double p = integrand_params->p;
    const double b = integrand_params->b;
    const int k = integrand_params->k;
    double ret = log(gsl_sf_bessel_I0_scaled(b / r) * gsl_pow_int(r, k))
        + scale - gsl_pow_2(p / r - 0.5 * b / p);
    if (integrand_params->cosmology)
        ret += log_dVC_dVL(r);
    return ret;
}


static double log_radial_integral(double r1, double r2, double p, double b, int k, int cosmology)
{
    radial_integrand_params params = {0, p, b, k, cosmology};
    double breakpoints[5];
    unsigned char nbreakpoints = 0;
    double result = 0, abserr, log_offset = -INFINITY;
    int ret;

    if (b != 0) {
        /* Calculate the approximate distance at which the integrand attains a
         * maximum (middle) and a fraction eta of the maximum (left and right).
         * This neglects the scaled Bessel function factors and the power-law
         * distance prior. It assumes that the likelihood is approximately of
         * the form
         *
         *    -p^2/r^2 + B/r.
         *
         * Then the middle breakpoint occurs at 1/r = -B/2A, and the left and
         * right breakpoints occur when
         *
         *   A/r^2 + B/r = log(eta) - B^2/4A.
         */

        static const double eta = 0.01;
        const double middle = 2 * gsl_pow_2(p) / b;
        const double left = 1 / (1 / middle + sqrt(-log(eta)) / p);
        const double right = 1 / (1 / middle - sqrt(-log(eta)) / p);

        /* Use whichever of the middle, left, and right points lie within the
         * integration limits as initial subdivisions for the adaptive
         * integrator. */

        breakpoints[nbreakpoints++] = r1;
        if(left > breakpoints[nbreakpoints-1] && left < r2)
            breakpoints[nbreakpoints++] = left;
        if(middle > breakpoints[nbreakpoints-1] && middle < r2)
            breakpoints[nbreakpoints++] = middle;
        if(right > breakpoints[nbreakpoints-1] && right < r2)
            breakpoints[nbreakpoints++] = right;
        breakpoints[nbreakpoints++] = r2;
    } else {
        /* Inner breakpoints are undefined because b = 0. */
        breakpoints[nbreakpoints++] = r1;
        breakpoints[nbreakpoints++] = r2;
    }

    /* Re-scale the integrand so that the maximum value at any of the
     * breakpoints is 1. Note that the initial value of the constant term
     * is overwritten. */

    for (unsigned char i = 0; i < nbreakpoints; i++)
    {
        double new_log_offset = log_radial_integrand(breakpoints[i], &params);
        if (new_log_offset > log_offset)
            log_offset = new_log_offset;
    }

    /* If the largest value of the log integrand was -INFINITY, then the
     * integrand is 0 everywhere. Set log_offset to 0, because subtracting
     * -INFINITY would make the integrand infinite. */
    if (log_offset == -INFINITY)
        log_offset = 0;

    params.scale = -log_offset;

    {
        /* Maximum number of subdivisions for adaptive integration. */
        static const size_t n = 64;

        /* Allocate workspace on stack. Hopefully, a little bit faster than
         * using the heap in multi-threaded code. */

        double alist[n];
        double blist[n];
        double rlist[n];
        double elist[n];
        size_t order[n];
        size_t level[n];
        gsl_integration_workspace workspace = {
            .alist = alist,
            .blist = blist,
            .rlist = rlist,
            .elist = elist,
            .order = order,
            .level = level,
            .limit = n
        };

        /* Set up integrand data structure. */
        const gsl_function func = {radial_integrand, &params};

        /* Perform adaptive Gaussian quadrature. */
        ret = gsl_integration_qagp(&func, breakpoints, nbreakpoints,
            DBL_MIN, 1e-8, n, &workspace, &result, &abserr);

        /* FIXME: do we care to keep the error estimate around? */
    }

    /* FIXME: do something with ret */
    (void)ret;

    /* Done! */
    return log_offset + log(result);
}


static const size_t default_log_radial_integrator_size = 400;


static log_radial_integrator *log_radial_integrator_init(double r1, double r2, int k, int cosmology, double pmax, size_t size)
{
    log_radial_integrator *integrator;
    bicubic_interp *region0;
    cubic_interp *region1, *region2;
    const double alpha = 4;
    const double p0 = 0.5 * (k >= 0 ? r2 : r1);
    const double xmax = log(pmax);
    const double x0 = GSL_MIN_DBL(log(p0), xmax);
    const double xmin = x0 - (1 + M_SQRT2) * alpha;
    const double ymax = x0 + alpha;
    const double ymin = 2 * x0 - M_SQRT2 * alpha - xmax;
    const double d = (xmax - xmin) / (size - 1); /* dx = dy = du */
    const double umin = - (1 + M_SQRT1_2) * alpha;
    const double vmax = x0 - M_SQRT1_2 * alpha;
    double z0[size][size], z1[size], z2[size];
    /* const double umax = xmax - vmax; */ /* unused */

    integrator = malloc(sizeof(*integrator));

    #pragma omp parallel for
    for (size_t i = 0; i < size * size; i ++)
    {
        const size_t ix = i / size;
        const size_t iy = i % size;
        const double x = xmin + ix * d;
        const double y = ymin + iy * d;
        const double p = exp(x);
        const double r0 = exp(y);
        const double b = 2 * gsl_pow_2(p) / r0;
        /* Note: using this where p > r0; could reduce evaluations by half */
        z0[ix][iy] = log_radial_integral(r1, r2, p, b, k, cosmology);
    }
    region0 = bicubic_interp_init(*z0, size, size, xmin, ymin, d, d);

    for (size_t i = 0; i < size; i ++)
        z1[i] = z0[i][size - 1];
    region1 = cubic_interp_init(z1, size, xmin, d);

    for (size_t i = 0; i < size; i ++)
        z2[i] = z0[i][size - 1 - i];
    region2 = cubic_interp_init(z2, size, umin, d);

    if (!(integrator && region0 && region1 && region2))
    {
        free(integrator);
        free(region0);
        free(region1);
        free(region2);
        XLAL_ERROR_NULL(XLAL_ENOMEM, "not enough memory to allocate integrator");
    }

    integrator->region0 = region0;
    integrator->region1 = region1;
    integrator->region2 = region2;
    integrator->xmax = xmax;
    integrator->ymax = ymax;
    integrator->vmax = vmax;
    integrator->r1 = r1;
    integrator->r2 = r2;
    integrator->k = k;
    return integrator;
}


static void log_radial_integrator_free(log_radial_integrator *integrator)
{
    if (integrator)
    {
        bicubic_interp_free(integrator->region0);
        integrator->region0 = NULL;
        cubic_interp_free(integrator->region1);
        integrator->region1 = NULL;
        cubic_interp_free(integrator->region2);
        integrator->region2 = NULL;
    }
    free(integrator);
}


static double log_radial_integrator_eval(const log_radial_integrator *integrator, double p, double b, double log_p, double log_b)
{
    const double x = log_p;
    const double y = M_LN2 + 2 * log_p - log_b;
    double result;
    assert(x <= integrator->xmax);

    if (p == 0) {
        /* note: p2 == 0 implies b == 0 */
        assert(b == 0);
        int k1 = integrator->k + 1;

        if (k1 == 0)
        {
            result = log(log(integrator->r2 / integrator->r1));
        } else {
            result = log((gsl_pow_int(integrator->r2, k1) - gsl_pow_int(integrator->r1, k1)) / k1);
        }
    } else {
        if (y >= integrator->ymax) {
            result = cubic_interp_eval(integrator->region1, x);
        } else {
            const double v = 0.5 * (x + y);
            if (v <= integrator->vmax)
            {
                const double u = 0.5 * (x - y);
                result = cubic_interp_eval(integrator->region2, u);
            } else {
                result = bicubic_interp_eval(integrator->region0, x, y);
            }
        }
        result += gsl_pow_2(0.5 * b / p);
    }

    return result;
}


/* Find error in time of arrival. */
static void toa_errors(
    double *dt,
    double theta,
    double phi,
    double gmst,
    int nifos,
    const double **locs, /* Input: detector position. */
    const double *toas /* Input: time of arrival. */
) {
    /* Convert to Cartesian coordinates. */
    double n[3];
    ang2vec(theta, phi - gmst, n);

    for (int i = 0; i < nifos; i ++)
    {
        double dot = 0;
        for (int j = 0; j < 3; j ++)
        {
            dot += locs[i][j] * n[j];
        }
        dt[i] = toas[i] + dot / LAL_C_SI;
    }
}


/* Compute antenna factors from the detector response tensor and source
 * sky location, and return as a complex number F_plus + i F_cross. */
static double complex complex_antenna_factor(
    const float response[3][3],
    double ra,
    double dec,
    double gmst
) {
    double complex F;
    XLALComputeDetAMResponse(
        (double *)&F,     /* Type-punned real part */
        1 + (double *)&F, /* Type-punned imag part */
        response, ra, dec, 0, gmst);
    return F;
}


/* Expression for complex amplitude on arrival (without 1/distance factor) */
static double complex signal_amplitude_model(
    double complex F,               /* Antenna factor */
    double complex exp_i_twopsi,    /* e^(i*2*psi), for polarization angle psi */
    double u,                       /* cos(inclination) */
    double u2                       /* cos^2(inclination */
) {
    const double complex tmp = F * conj(exp_i_twopsi);
    return 0.5 * (1 + u2) * creal(tmp) - I * u * cimag(tmp);
}


#define nu 10
static const unsigned int ntwopsi = 10;
static double u_points_weights[nu][2];


static void u_points_weights_init(void)
{
    /* Look up Gauss-Legendre quadrature rule for integral over cos(i). */
    gsl_integration_glfixed_table *gltable
        = gsl_integration_glfixed_table_alloc(nu);

    /* Don't bother checking the return value. GSL has static, precomputed
     * values for certain orders, and for the order I have picked it will
     * return a pointer to one of these. See:
     *
     * http://git.savannah.gnu.org/cgit/gsl.git/tree/integration/glfixed.c
     */
    assert(gltable);
    assert(gltable->precomputed); /* We don't have to free it. */

    for (unsigned int iu = 0; iu < nu; iu++)
    {
        double weight;

        /* Look up Gauss-Legendre abscissa and weight. */
        int ret = gsl_integration_glfixed_point(
            -1, 1, iu, &u_points_weights[iu][0], &weight, gltable);

        /* Don't bother checking return value; the only
         * possible failure is in index bounds checking. */
        assert(ret == GSL_SUCCESS);

        u_points_weights[iu][1] = log(weight);
    }
}


static double complex exp_i(double phi) {
    return cos(phi) + I * sin(phi);
}


/* Compare two pixels by contained probability. */
static int bayestar_pixel_compare_prob(const void *a, const void *b)
{
    const bayestar_pixel *apix = a;
    const bayestar_pixel *bpix = b;

    const double delta_logp = (apix->value[0] - bpix->value[0])
        - 2 * M_LN2 * (uniq2order64(apix->uniq) - uniq2order64(bpix->uniq));

    if (delta_logp < 0)
        return -1;
    else if (delta_logp > 0)
        return 1;
    else
        return 0;
}


static void bayestar_pixels_sort_prob(bayestar_pixel *pixels, size_t len)
{
    qsort(pixels, len, sizeof(bayestar_pixel), bayestar_pixel_compare_prob);
}


/* Compare two pixels by contained probability. */
static int bayestar_pixel_compare_uniq(const void *a, const void *b)
{
    const bayestar_pixel *apix = a;
    const bayestar_pixel *bpix = b;
    const unsigned long long auniq = apix->uniq;
    const unsigned long long buniq = bpix->uniq;

    if (auniq < buniq)
        return -1;
    else if (auniq > buniq)
        return 1;
    else
        return 0;
}


static void bayestar_pixels_sort_uniq(bayestar_pixel *pixels, size_t len)
{
    qsort(pixels, len, sizeof(bayestar_pixel), bayestar_pixel_compare_uniq);
}


static void *realloc_or_free(void *ptr, size_t size)
{
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr)
    {
        free(ptr);
        XLAL_ERROR_NULL(XLAL_ENOMEM, "not enough memory to resize array");
    }
    return new_ptr;
}


/* Subdivide the final last_n pixels of an adaptively refined sky map. */
static bayestar_pixel *bayestar_pixels_refine(
    bayestar_pixel *pixels, size_t *len, size_t last_n
) {
    assert(last_n <= *len);

    /* New length: adding 4*last_n new pixels, removing last_n old pixels. */
    const size_t new_len = *len + 3 * last_n;
    const size_t new_size = new_len * sizeof(bayestar_pixel);

    pixels = realloc_or_free(pixels, new_size);
    if (pixels)
    {
        for (size_t i = 0; i < last_n; i ++)
        {
            const uint64_t uniq = 4 * pixels[*len - i - 1].uniq;
            for (unsigned char j = 0; j < 4; j ++)
                pixels[new_len - (4 * i + j) - 1].uniq = j + uniq;
        }
        *len = new_len;
    }
    return pixels;
}


static bayestar_pixel *bayestar_pixels_alloc(size_t *len, unsigned char order)
{
    const uint64_t nside = (uint64_t)1 << order;
    const uint64_t npix = nside2npix64(nside);
    const size_t size = npix * sizeof(bayestar_pixel);

    bayestar_pixel *pixels = malloc(size);
    if (!pixels)
        XLAL_ERROR_NULL(XLAL_ENOMEM, "not enough memory to allocate sky map");

    *len = npix;
    for (unsigned long long ipix = 0; ipix < npix; ipix ++)
        pixels[ipix].uniq = nest2uniq64(order, ipix);
    return pixels;
}


static void logsumexp(const double *accum, double log_weight, double *result, unsigned long ni, unsigned long nj)
{
    double max_accum[nj];
    for (unsigned long j = 0; j < nj; j ++)
        max_accum[j] = -INFINITY;
    for (unsigned long i = 0; i < ni; i ++)
        for (unsigned long j = 0; j < nj; j ++)
            if (accum[i * nj + j] > max_accum[j])
                max_accum[j] = accum[i * nj + j];
    double sum_accum[nj];
    for (unsigned long j = 0; j < nj; j ++)
        sum_accum[j] = 0;
    for (unsigned long i = 0; i < ni; i ++)
        for (unsigned long j = 0; j < nj; j ++)
            sum_accum[j] += exp(accum[i * nj + j] - max_accum[j]);
    for (unsigned long j = 0; j < nj; j ++)
        result[j] = log(sum_accum[j]) + max_accum[j] + log_weight;
}


/* Fudge factor for excess estimation error in gstlal_inspiral. */
static const double FUDGE = 0.83;


static void bayestar_sky_map_toa_phoa_snr_pixel(
    log_radial_integrator *integrators[],
    unsigned char nint,
    uint64_t uniq,
    double *const value,
    double gmst,
    unsigned int nifos,
    unsigned long nsamples,
    double sample_rate,
    const double *epochs,
    const float complex **snrs,
    const float (**responses)[3],
    const double **locations,
    const double *horizons
) {
    double complex F[nifos];
    double dt[nifos];
    double accum[nint];
    for (unsigned char k = 0; k < nint; k ++)
        accum[k] = -INFINITY;

    {
        double theta, phi;
        pix2ang_uniq64(uniq, &theta, &phi);

        /* Look up antenna factors */
        for (unsigned int iifo = 0; iifo < nifos; iifo++)
            F[iifo] = complex_antenna_factor(
                responses[iifo], phi, M_PI_2-theta, gmst) * horizons[iifo];

        toa_errors(dt, theta, phi, gmst, nifos, locations, epochs);
    }

    /* Integrate over 2*psi */
    for (unsigned int itwopsi = 0; itwopsi < ntwopsi; itwopsi++)
    {
        const double twopsi = (2 * M_PI / ntwopsi) * itwopsi;
        const double complex exp_i_twopsi = exp_i(twopsi);
        double accum1[nint];
        for (unsigned char k = 0; k < nint; k ++)
            accum1[k] = -INFINITY;

        /* Integrate over u from -1 to 1. */
        for (unsigned int iu = 0; iu < nu; iu++)
        {
            const double u = u_points_weights[iu][0];
            const double log_weight = u_points_weights[iu][1];
            double accum2[nsamples][nint];

            const double u2 = gsl_pow_2(u);
            double complex z_times_r[nifos];
            double p2 = 0;

            for (unsigned int iifo = 0; iifo < nifos; iifo ++)
            {
                p2 += cabs2(
                    z_times_r[iifo] = signal_amplitude_model(
                        F[iifo], exp_i_twopsi, u, u2));
            }
            p2 *= 0.5;
            p2 *= gsl_pow_2(FUDGE);
            const double p = sqrt(p2);
            const double log_p = log(p);

            for (unsigned long isample = 0; isample < nsamples; isample++)
            {
                double b;
                {
                    double complex I0arg_complex_times_r = 0;
                    for (unsigned int iifo = 0; iifo < nifos; iifo ++)
                    {
                        I0arg_complex_times_r += conj(z_times_r[iifo])
                            * eval_snr(snrs[iifo], nsamples,
                                isample - dt[iifo] * sample_rate - 0.5 * (nsamples - 1));
                    }
                    b = cabs(I0arg_complex_times_r);
                    b *= gsl_pow_2(FUDGE);
                }
                const double log_b = log(b);

                for (unsigned char k = 0; k < nint; k ++)
                {
                    accum2[isample][k] = log_radial_integrator_eval(
                        integrators[k], p, b, log_p, log_b);
                }
            }

            double log_sum_accum2[nint];
            logsumexp(*accum2, log_weight, log_sum_accum2, nsamples, nint);
            for (unsigned char k = 0; k < nint; k ++)
                accum1[k] = logaddexp(accum1[k], log_sum_accum2[k]);
        }

        for (unsigned char k = 0; k < nint; k ++)
        {
            accum[k] = logaddexp(accum[k], accum1[k]);
        }
    }

    /* Record logarithm of posterior. */
    for (unsigned char k = 0; k < nint; k ++)
    {
        value[k] = accum[k];
    }
}


static pthread_once_t bayestar_init_once = PTHREAD_ONCE_INIT;
static void bayestar_init_func(void)
{
    dVC_dVL_init();
    u_points_weights_init();
}
static void bayestar_init(void)
{
    int ret = pthread_once(&bayestar_init_once, bayestar_init_func);
    assert(ret == 0);
}


bayestar_pixel *bayestar_sky_map_toa_phoa_snr(
    size_t *out_len,                /* Number of returned pixels */
    double *out_log_bci,            /* log Bayes factor: coherent vs. incoherent */
    double *out_log_bsn,            /* log Bayes factor: signal vs. noise */
    /* Prior */
    double min_distance,            /* Minimum distance */
    double max_distance,            /* Maximum distance */
    int prior_distance_power,       /* Power of distance in prior */
    int cosmology,                  /* Set to nonzero to include comoving volume correction */
    /* Data */
    double gmst,                    /* GMST (rad) */
    unsigned int nifos,             /* Number of detectors */
    unsigned long nsamples,         /* Length of SNR series */
    double sample_rate,             /* Sample rate in seconds */
    const double *epochs,           /* Timestamps of SNR time series */
    const float complex **snrs,     /* Complex SNR series */
    const float (**responses)[3],   /* Detector responses */
    const double **locations,       /* Barycentered Cartesian geographic detector positions (m) */
    const double *horizons          /* SNR=1 horizon distances for each detector */
) {
    /* Initialize precalculated tables. */
    bayestar_init();

    if (cosmology && prior_distance_power != 2)
    {
        XLAL_ERROR_NULL(XLAL_EINVAL,
            "BAYESTAR supports cosmological priors only for for prior_distance_power=2");
    }
    log_radial_integrator *integrators[] = {NULL, NULL, NULL};
    {
        double pmax = 0;
        for (unsigned int iifo = 0; iifo < nifos; iifo ++)
        {
            pmax += gsl_pow_2(horizons[iifo]);
        }
        pmax = sqrt(0.5 * pmax);
        pmax *= FUDGE;
        for (unsigned char k = 0; k < 3; k ++)
        {
            integrators[k] = log_radial_integrator_init(
                min_distance, max_distance, prior_distance_power + k, cosmology,
                pmax, default_log_radial_integrator_size);
            if (!integrators[k])
            {
                for (unsigned char kk = 0; kk < k; kk ++)
                    log_radial_integrator_free(integrators[kk]);
                return NULL;
            }
        }
    }

    static const unsigned char order0 = 4;
    size_t len;
    bayestar_pixel *pixels = bayestar_pixels_alloc(&len, order0);
    if (!pixels)
    {
        for (unsigned char k = 0; k < 3; k ++)
            log_radial_integrator_free(integrators[k]);
        return NULL;
    }
    const unsigned long npix0 = len;

    /* At the lowest order, compute both the coherent probability map and the
     * incoherent evidence. */
    const double log_norm = -log(2 * (2 * M_PI) * (4 * M_PI) * ntwopsi * nsamples) - log_radial_integrator_eval(integrators[0], 0, 0, -INFINITY, -INFINITY);
    double log_evidence_coherent, log_evidence_incoherent[nifos];
    {
        double accum[npix0][nifos];

        #pragma omp parallel for
        for (unsigned long i = 0; i < npix0; i ++)
        {
            bayestar_sky_map_toa_phoa_snr_pixel(integrators, 1, pixels[i].uniq,
                pixels[i].value, gmst, nifos, nsamples, sample_rate, epochs,
                snrs, responses, locations, horizons);

            for (unsigned int iifo = 0; iifo < nifos; iifo ++)
            {
                bayestar_sky_map_toa_phoa_snr_pixel(integrators, 1,
                    pixels[i].uniq, &accum[i][iifo], gmst, 1, nsamples,
                    sample_rate, &epochs[iifo], &snrs[iifo], &responses[iifo],
                    &locations[iifo], &horizons[iifo]);
            }
        }

        const double log_weight = log_norm + log(uniq2pixarea64(pixels[0].uniq));

        logsumexp(*accum, log_weight, log_evidence_incoherent, npix0, nifos);
    }

    /* Sort pixels by ascending posterior probability. */
    bayestar_pixels_sort_prob(pixels, len);

    /* Adaptively refine until order=11 (nside=2048). */
    for (unsigned char level = order0; level < 11; level ++)
    {
        /* Adaptively refine the pixels that contain the most probability. */
        pixels = bayestar_pixels_refine(pixels, &len, npix0 / 4);
        if (!pixels)
        {
            for (unsigned char k = 0; k < 3; k ++)
                log_radial_integrator_free(integrators[k]);
            return NULL;
        }

        #pragma omp parallel for
        for (unsigned long i = len - npix0; i < len; i ++)
        {
            bayestar_sky_map_toa_phoa_snr_pixel(integrators, 1, pixels[i].uniq,
                pixels[i].value, gmst, nifos, nsamples, sample_rate, epochs,
                snrs, responses, locations, horizons);
        }

        /* Sort pixels by ascending posterior probability. */
        bayestar_pixels_sort_prob(pixels, len);
    }

    /* Evaluate distance layers. */
    #pragma omp parallel for
    for (unsigned long i = 0; i < len; i ++)
    {
        bayestar_sky_map_toa_phoa_snr_pixel(&integrators[1], 2, pixels[i].uniq,
            &pixels[i].value[1], gmst, nifos, nsamples, sample_rate, epochs,
            snrs, responses, locations, horizons);
    }

    for (unsigned char k = 0; k < 3; k ++)
        log_radial_integrator_free(integrators[k]);

    /* Rescale so that log(max) = 0. */
    const double max_logp = pixels[len - 1].value[0];
    for (ssize_t i = (ssize_t)len - 1; i >= 0; i --)
        for (unsigned char k = 0; k < 3; k ++)
            pixels[i].value[k] -= max_logp;

    /* Determine normalization of map. */
    double norm = 0;
    for (ssize_t i = (ssize_t)len - 1; i >= 0; i --)
    {
        const double dA = uniq2pixarea64(pixels[i].uniq);
        const double dP = gsl_sf_exp_mult(pixels[i].value[0], dA);
        if (dP <= 0)
            break; /* We have reached underflow. */
        norm += dP;
    }
    log_evidence_coherent = log(norm) + max_logp + log_norm;
    norm = 1 / norm;

    /* Rescale, normalize, and prepare output. */
    for (ssize_t i = (ssize_t)len - 1; i >= 0; i --)
    {
        const double prob = gsl_sf_exp_mult(pixels[i].value[0], norm);
        double rmean = exp(pixels[i].value[1] - pixels[i].value[0]);
        double rstd = exp(pixels[i].value[2] - pixels[i].value[0]) - gsl_pow_2(rmean);
        if (rstd >= 0)
        {
            rstd = sqrt(rstd);
        } else {
            rmean = INFINITY;
            rstd = 1;
        }
        pixels[i].value[0] = prob;
        pixels[i].value[1] = rmean;
        pixels[i].value[2] = rstd;
    }

    /* Sort pixels by ascending NUNIQ index. */
    bayestar_pixels_sort_uniq(pixels, len);

    /* Calculate log Bayes factor. */
    *out_log_bci = *out_log_bsn = log_evidence_coherent;
    for (unsigned int i = 0; i < nifos; i ++)
        *out_log_bci -= log_evidence_incoherent[i];

    /* Done! */
    *out_len = len;
    return pixels;
}


double bayestar_log_likelihood_toa_phoa_snr(
    /* Parameters */
    double ra,                      /* Right ascension (rad) */
    double sin_dec,                 /* Sin(declination) */
    double distance,                /* Distance */
    double u,                       /* Cos(inclination) */
    double twopsi,                  /* Twice polarization angle (rad) */
    double t,                       /* Barycentered arrival time (s) */
    /* Data */
    double gmst,                    /* GMST (rad) */
    unsigned int nifos,             /* Number of detectors */
    unsigned long nsamples,         /* Lengths of SNR series */
    double sample_rate,             /* Sample rate in seconds */
    const double *epochs,           /* Timestamps of SNR time series */
    const float complex **snrs,     /* Complex SNR series */
    const float (**responses)[3],   /* Detector responses */
    const double **locations,       /* Barycentered Cartesian geographic detector positions (m) */
    const double *horizons          /* SNR=1 horizon distances for each detector */
) {
    const double dec = asin(sin_dec);
    const double u2 = gsl_pow_2(u);
    const double complex exp_i_twopsi = exp_i(twopsi);
    const double one_by_r = 1 / distance;

    /* Compute time of arrival errors */
    double dt[nifos];
    toa_errors(dt, M_PI_2 - dec, ra, gmst, nifos, locations, epochs);

    double complex i0arg_complex_times_r = 0;
    double A = 0;

    /* Loop over detectors */
    for (unsigned int iifo = 0; iifo < nifos; iifo++)
    {
        const double complex F = complex_antenna_factor(
            responses[iifo], ra, dec, gmst) * horizons[iifo];

        const double complex z_times_r =
             signal_amplitude_model(F, exp_i_twopsi, u, u2);

        i0arg_complex_times_r += conj(z_times_r)
            * eval_snr(snrs[iifo], nsamples, (t - dt[iifo]) * sample_rate - 0.5 * (nsamples - 1));
        A += cabs2(z_times_r);
    }
    A *= -0.5;

    double i0arg_times_r = cabs(i0arg_complex_times_r);

    A *= gsl_pow_2(FUDGE);
    i0arg_times_r *= gsl_pow_2(FUDGE);

    return (A * one_by_r + i0arg_times_r) * one_by_r
        + log(gsl_sf_bessel_I0_scaled(i0arg_times_r * one_by_r));
}


/*
 * Unit tests
 */


static void test_cabs2(double complex z)
{
    const double result = cabs2(z);
    const double expected = gsl_pow_2(cabs(z));
    gsl_test_abs(result, expected, 2 * GSL_DBL_EPSILON,
        "testing cabs2(%g + %g j)", creal(z), cimag(z));
}


static void test_complex_catrom(void)
{
    for (double t = 0; t <= 1; t += 0.01)
    {
        const double complex result = complex_catrom(0, 0, 0, 0, t);
        const double complex expected = 0;
        gsl_test_abs(creal(result), creal(expected), 0,
            "testing complex Catmull-rom interpolant for zero input");
        gsl_test_abs(cimag(result), cimag(expected), 0,
            "testing complex Catmull-rom interpolant for zero input");
    }

    for (double t = 0; t <= 1; t += 0.01)
    {
        const double complex result = complex_catrom(1, 1, 1, 1, t);
        const double complex expected = 1;
        gsl_test_abs(creal(result), creal(expected), 0,
            "testing complex Catmull-rom interpolant for unit input");
        gsl_test_abs(cimag(result), cimag(expected), 0,
            "testing complex Catmull-rom interpolant for unit input");
    }

    for (double t = 0; t <= 1; t += 0.01)
    {
        const double complex result = complex_catrom(1.0j, 1.0j, 1.0j, 1.0j, t);
        const double complex expected = 1.0j;
        gsl_test_abs(creal(result), creal(expected), 0,
            "testing complex Catmull-rom interpolant for unit imaginary input");
        gsl_test_abs(cimag(result), cimag(expected), 1,
            "testing complex Catmull-rom interpolant for unit imaginary input");
    }

    for (double t = 0; t <= 1; t += 0.01)
    {
        const double complex result = complex_catrom(1.0+1.0j, 1.0+1.0j, 1.0+1.0j, 1.0+1.0j, t);
        const double complex expected = 1.0+1.0j;
        gsl_test_abs(creal(result), creal(expected), 0,
            "testing complex Catmull-rom interpolant for unit real + imaginary input");
        gsl_test_abs(cimag(result), cimag(expected), 0,
            "testing complex Catmull-rom interpolant for unit real + imaginary input");
    }

    for (double t = 0; t <= 1; t += 0.01)
    {
        const double complex result = complex_catrom(1, 0, 1, 4, t);
        const double complex expected = gsl_pow_2(t);
        gsl_test_abs(creal(result), creal(expected), 0,
            "testing complex Catmull-rom interpolant for quadratic real input");
        gsl_test_abs(cimag(result), cimag(expected), 0,
            "testing complex Catmull-rom interpolant for quadratic real input");
    }
}


static void test_eval_snr(void)
{
    static const size_t nsamples = 64;
    float complex x[nsamples];

    /* Populate data with samples of x(t) = t^2 + t j */
    for (size_t i = 0; i < nsamples; i ++)
        x[i] = gsl_pow_2(i) + i * 1.0j;

    for (double t = 0; t <= nsamples; t += 0.1)
    {
        const double result = eval_snr(x, nsamples, t);
        const double expected = (t > 1 && t < nsamples - 2) ? (gsl_pow_2(t) + t*1.0j) : 0;
        gsl_test_abs(creal(result), creal(expected), 1e4 * GSL_DBL_EPSILON,
            "testing real part of eval_snr(%g) for x(t) = t^2 + t j", t);
        gsl_test_abs(cimag(result), cimag(expected), 1e4 * GSL_DBL_EPSILON,
            "testing imaginary part of eval_snr(%g) for x(t) = t^2 + t j", t);
    }
}


static void test_signal_amplitude_model(
    double ra,
    double dec,
    double inclination,
    double polarization,
    unsigned long epoch_nanos,
    const char *instrument
) {
    const LALDetector *detector = XLALDetectorPrefixToLALDetector(
        instrument);
    LIGOTimeGPS epoch;
    double gmst;
    XLALINT8NSToGPS(&epoch, epoch_nanos);
    gmst = XLALGreenwichMeanSiderealTime(&epoch);

    double complex result;
    {
        double complex F;
        const double complex exp_i_twopsi = exp_i(2 * polarization);
        const double u = cos(inclination);
        const double u2 = gsl_pow_2(u);
        XLALComputeDetAMResponse(
            (double *)&F,     /* Type-punned real part */
            1 + (double *)&F, /* Type-punned imag part */
            detector->response, ra, dec, 0, gmst);

        result = signal_amplitude_model(F, exp_i_twopsi, u, u2);
    }

    float abs_expected;
    {
        SimInspiralTable sim_inspiral;
        memset(&sim_inspiral, 0, sizeof(sim_inspiral));
        sim_inspiral.distance = 1;
        sim_inspiral.longitude = ra;
        sim_inspiral.latitude = dec;
        sim_inspiral.inclination = inclination;
        sim_inspiral.polarization = polarization;
        sim_inspiral.geocent_end_time = epoch;
        sim_inspiral.end_time_gmst = gmst;

        XLALInspiralSiteTimeAndDist(
            &sim_inspiral, detector, &epoch, &abs_expected);
        abs_expected = 1 / abs_expected;
    }

    /* This is the *really* slow way of working out the signal amplitude:
     * generate a frequency-domain waveform and inject it. */
    double complex expected = 0;
    {
        COMPLEX16FrequencySeries
            *Htemplate = NULL, *Hsignal = NULL, *Hcross = NULL;
        double h = 0, Fplus, Fcross;
        int ret;
        LALDict *params = XLALCreateDict();
        assert(params);
        XLALSimInspiralWaveformParamsInsertPNPhaseOrder(params, LAL_PNORDER_NEWTONIAN);
        XLALSimInspiralWaveformParamsInsertPNAmplitudeOrder(params, LAL_PNORDER_NEWTONIAN);

        /* Calculate antenna factors */
        XLALComputeDetAMResponse(
            &Fplus, &Fcross, detector->response, ra, dec, polarization, gmst);

        /* Check that the way I compute the antenna factors matches */
        {
            double complex F;
            XLALComputeDetAMResponse(
                (double *)&F, 1 + (double *)&F, detector->response, ra, dec, 0, gmst);
            F *= exp_i(-2 * polarization);

            gsl_test_abs(creal(F), Fplus, 4 * GSL_DBL_EPSILON,
                "testing Fplus(ra=%0.03f, dec=%0.03f, "
                "inc=%0.03f, pol=%0.03f, epoch_nanos=%lu, detector=%s)",
                ra, dec, inclination, polarization, epoch_nanos, instrument);

            gsl_test_abs(cimag(F), Fcross, 4 * GSL_DBL_EPSILON,
                "testing Fcross(ra=%0.03f, dec=%0.03f, "
                "inc=%0.03f, pol=%0.03f, epoch_nanos=%lu, detector=%s)",
                ra, dec, inclination, polarization, epoch_nanos, instrument);
        }

        /* "Template" waveform with inclination angle of zero */
        ret = XLALSimInspiralFD(
            &Htemplate, &Hcross, 1.4 * LAL_MSUN_SI, 1.4 * LAL_MSUN_SI,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 100, 101, 100, params,
            TaylorF2);
        assert(ret == XLAL_SUCCESS);

        /* Discard any non-quadrature phase component of "template" */
        for (size_t i = 0; i < Htemplate->data->length; i ++)
            Htemplate->data->data[i] += I * Hcross->data->data[i];

        /* Free Hcross */
        XLALDestroyCOMPLEX16FrequencySeries(Hcross);
        Hcross = NULL;

        /* Normalize "template" */
        for (size_t i = 0; i < Htemplate->data->length; i ++)
            h += cabs2(Htemplate->data->data[i]);
        h = 2 / h;
        for (size_t i = 0; i < Htemplate->data->length; i ++)
            Htemplate->data->data[i] *= h;

        /* "Signal" waveform with requested inclination angle */
        ret = XLALSimInspiralFD(
            &Hsignal, &Hcross, 1.4 * LAL_MSUN_SI, 1.4 * LAL_MSUN_SI,
            0, 0, 0, 0, 0, 0, 1, inclination, 0, 0, 0, 0, 1, 100, 101, 100,
            params, TaylorF2);
        assert(ret == XLAL_SUCCESS);

        /* Project "signal" using antenna factors */
        for (size_t i = 0; i < Hsignal->data->length; i ++)
            Hsignal->data->data[i] = Fplus * Hsignal->data->data[i]
                                   + Fcross * Hcross->data->data[i];

        /* Free Hcross */
        XLALDestroyCOMPLEX16FrequencySeries(Hcross);
        Hcross = NULL;

        /* Work out complex amplitude by comparing with "template" waveform */
        for (size_t i = 0; i < Htemplate->data->length; i ++)
            expected += conj(Htemplate->data->data[i]) * Hsignal->data->data[i];

        XLALDestroyDict(params);
        XLALDestroyCOMPLEX16FrequencySeries(Hsignal);
        XLALDestroyCOMPLEX16FrequencySeries(Htemplate);
        Hsignal = Htemplate = NULL;
    }

    /* Test to nearly float (32-bit) precision because
     * XLALInspiralSiteTimeAndDist returns result as float. */
    gsl_test_abs(cabs(result), abs_expected, 1.5 * GSL_FLT_EPSILON,
        "testing abs(signal_amplitude_model(ra=%0.03f, dec=%0.03f, "
        "inc=%0.03f, pol=%0.03f, epoch_nanos=%lu, detector=%s))",
        ra, dec, inclination, polarization, epoch_nanos, instrument);

    gsl_test_abs(creal(result), creal(expected), 4 * GSL_DBL_EPSILON,
        "testing re(signal_amplitude_model(ra=%0.03f, dec=%0.03f, "
        "inc=%0.03f, pol=%0.03f, epoch_nanos=%lu, detector=%s))",
        ra, dec, inclination, polarization, epoch_nanos, instrument);

    gsl_test_abs(cimag(result), cimag(expected), 4 * GSL_DBL_EPSILON,
        "testing im(signal_amplitude_model(ra=%0.03f, dec=%0.03f, "
        "inc=%0.03f, pol=%0.03f, epoch_nanos=%lu, detector=%s))",
        ra, dec, inclination, polarization, epoch_nanos, instrument);
}


static void test_log_radial_integral(
    double expected, double tol, double r1, double r2, double p2, double b, int k)
{
    const double p = sqrt(p2);
    log_radial_integrator *integrator = log_radial_integrator_init(
        r1, r2, k, 0, p + 0.5, default_log_radial_integrator_size);

    gsl_test(!integrator, "testing that integrator object is non-NULL");
    if (integrator)
    {
        const double result = log_radial_integrator_eval(integrator, p, b, log(p), log(b));

        gsl_test_rel(
            result, expected, tol,
            "testing toa_phoa_snr_log_radial_integral("
            "r1=%g, r2=%g, p2=%g, b=%g, k=%d)", r1, r2, p2, b, k);
        free(integrator);
    }
}


static void test_distance_moments_to_parameters_round_trip(double mean, double std)
{
    static const double min_mean_std = M_SQRT3 + 0.1;
    double mu, sigma, norm, mean2, std2, norm2;

    bayestar_distance_moments_to_parameters(
        mean, std, &mu, &sigma, &norm);
    bayestar_distance_parameters_to_moments(
        mu, sigma, &mean2, &std2, &norm2);

    if (gsl_finite(mean / std) && mean / std >= min_mean_std)
    {
        gsl_test_rel(norm2, norm, 1e-9,
            "testing round-trip conversion of normalization for mean=%g, std=%g",
            mean, std);
        gsl_test_rel(mean2, mean, 1e-9,
            "testing round-trip conversion of mean for mean=%g, std=%g",
            mean, std);
        gsl_test_rel(std2, std, 1e-9,
            "testing round-trip conversion of mean for mean=%g, std=%g",
            mean, std);
    } else {
        gsl_test_int(gsl_isinf(mu), 1,
            "testing that out-of-bounds value gives mu=+inf for mean=%g, std=%g",
            mean, std);
        gsl_test_abs(sigma, 1, 0,
            "testing that out-of-bounds value gives sigma=1 for mean=%g, std=%g",
            mean, std);
        gsl_test_abs(norm, 0, 0,
            "testing that out-of-bounds value gives norm=0 for mean=%g, std=%g",
            mean, std);
        gsl_test_int(gsl_isinf(mean2), 1,
            "testing that out-of-bounds value gives mean=+inf for mean=%g, std=%g",
            mean, std);
        gsl_test_abs(std2, 1, 0,
            "testing that out-of-bounds value gives std=1 for mean=%g, std=%g",
            mean, std);
        gsl_test_abs(norm2, 0, 0,
            "testing that out-of-bounds value gives norm=0 for mean=%g, std=%g",
            mean, std);
    }
}


static void test_nest2uniq64(uint8_t order, uint64_t nest, uint64_t uniq)
{
    const uint64_t uniq_result = nest2uniq64(order, nest);
    gsl_test(!(uniq_result == uniq),
        "expected nest2uniq64(%u, %llu) = %llu, got %llu",
        (unsigned) order, nest, uniq, uniq_result);

    uint64_t nest_result;
    const uint8_t order_result = uniq2nest64(uniq, &nest_result);
    gsl_test(!(nest_result == nest && order_result == order),
        "expected uniq2nest64(%llu) = (%u, %llu), got (%u, %llu)",
        uniq, (unsigned) order, nest, order_result, nest_result);
}


static void test_cosmology(void)
{
    static const int n = sizeof(dVC_dVL_test_x) / sizeof(*dVC_dVL_test_x);
    for (int i = 0; i < n; i ++)
    {
        const double DL = dVC_dVL_test_x[i];
        const double result = exp(log_dVC_dVL(DL));
        const double expected = dVC_dVL_test_y[i];
        gsl_test_rel(result, expected, 2e-3,
            "testing cosmological prior for DL=%g", DL);
    }
}


int bayestar_test(void)
{
    /* Initialize precalculated tables. */
    bayestar_init();

    for (double re = -1; re < 1; re += 0.1)
        for (double im = -1; im < 1; im += 0.1)
            test_cabs2(re + im * 1.0j);

    test_complex_catrom();
    test_eval_snr();

    for (double ra = -M_PI; ra <= M_PI; ra += 0.4 * M_PI)
        for (double dec = -M_PI_2; dec <= M_PI_2; dec += 0.4 * M_PI)
            for (double inc = -M_PI; inc <= M_PI; inc += 0.4 * M_PI)
                for (double pol = -M_PI; pol <= M_PI; pol += 0.4 * M_PI)
                    for (unsigned long epoch = 1000000000000000000;
                         epoch <= 1000086400000000000;
                         epoch += 7200000000000)
                        test_signal_amplitude_model(ra, dec, inc, pol, epoch, "H1");

    /* Tests of radial integrand with p2=0, b=0. */
    test_log_radial_integral(0, 0, 0, 1, 0, 0, 0);
    test_log_radial_integral(0, 0, exp(1), exp(2), 0, 0, -1);
    test_log_radial_integral(log(63), 0, 3, 6, 0, 0, 2);
    /* Te integrand with p2>0, b=0 (from Mathematica). */
    test_log_radial_integral(-0.480238, 1e-3, 1, 2, 1, 0, 0);
    test_log_radial_integral(0.432919, 1e-3, 1, 2, 1, 0, 2);
    test_log_radial_integral(-2.76076, 1e-3, 0, 1, 1, 0, 2);
    test_log_radial_integral(61.07118, 1e-3, 0, 1e9, 1, 0, 2);
    test_log_radial_integral(-112.23053, 5e-2, 0, 0.1, 1, 0, 2);
    /* Note: this test underflows, so we test that the log is -inf. */
    /* test_log_radial_integral(-1.00004e6, 1e-8, 0, 1e-3, 1, 0, 2); */
    test_log_radial_integral(-INFINITY, 1e-3, 0, 1e-3, 1, 0, 2);

    /* Tests of radial integrand with p2>0, b>0 with ML peak outside
     * of integration limits (true values from Mathematica NIntegrate). */
    test_log_radial_integral(2.94548, 1e-4, 0, 4, 1, 1, 2);
    test_log_radial_integral(2.94545, 1e-4, 0.5, 4, 1, 1, 2);
    test_log_radial_integral(2.94085, 1e-4, 1, 4, 1, 1, 2);
    /* Tests of radial integrand with p2>0, b>0 with ML peak outside
     * of integration limits (true values from Mathematica NIntegrate). */
    test_log_radial_integral(-2.43264, 1e-5, 0, 1, 1, 1, 2);
    test_log_radial_integral(-2.43808, 1e-5, 0.5, 1, 1, 1, 2);
    test_log_radial_integral(-0.707038, 1e-5, 1, 1.5, 1, 1, 2);

    {
        const double r1 = 0.0, r2 = 0.25, pmax = 1.0;
        const int k = 2;
        const double tol = 1e-5;
        log_radial_integrator *integrator = log_radial_integrator_init(
            r1, r2, k, 0, pmax, default_log_radial_integrator_size);

        gsl_test(!integrator, "testing that integrator object is non-NULL");
        if (integrator)
        {
            for (double p = 0.01; p <= pmax; p += 0.01)
            {
                for (double b = 0.0; b <= 2 * pmax; b += 0.01)
                {
                    const double r0 = 2 * gsl_pow_2(p) / b;
                    const double x = log(p);
                    const double y = log(r0);
                    const double expected = exp(log_radial_integral(r1, r2, p, b, k, 0));
                    const double result = exp(log_radial_integrator_eval(integrator, p, b, log(p), log(b)) - gsl_pow_2(0.5 * b / p));
                    gsl_test_abs(
                        result, expected, tol, "testing log_radial_integrator_eval("
                        "r1=%g, r2=%g, p=%g, b=%g, k=%d, x=%g, y=%g)", r1, r2, p, b, k, x, y);
                }
            }
            free(integrator);
        }
    }

    for (double mean = 0; mean < 100; mean ++)
        for (double std = 0; std < 100; std ++)
            test_distance_moments_to_parameters_round_trip(mean, std);

    test_nest2uniq64(0, 0, 4);
    test_nest2uniq64(0, 1, 5);
    test_nest2uniq64(0, 2, 6);
    test_nest2uniq64(0, 3, 7);
    test_nest2uniq64(0, 4, 8);
    test_nest2uniq64(0, 5, 9);
    test_nest2uniq64(0, 6, 10);
    test_nest2uniq64(0, 7, 11);
    test_nest2uniq64(0, 8, 12);
    test_nest2uniq64(0, 9, 13);
    test_nest2uniq64(0, 10, 14);
    test_nest2uniq64(0, 11, 15);
    test_nest2uniq64(1, 0, 16);
    test_nest2uniq64(1, 1, 17);
    test_nest2uniq64(1, 2, 18);
    test_nest2uniq64(1, 47, 63);
    test_nest2uniq64(12, 0, 0x4000000ull);
    test_nest2uniq64(12, 1, 0x4000001ull);
    test_nest2uniq64(29, 0, 0x1000000000000000ull);
    test_nest2uniq64(29, 1, 0x1000000000000001ull);
    test_nest2uniq64(29, 0x2FFFFFFFFFFFFFFFull, 0x3FFFFFFFFFFFFFFFull);

    test_cosmology();

    return gsl_test_summary();
}

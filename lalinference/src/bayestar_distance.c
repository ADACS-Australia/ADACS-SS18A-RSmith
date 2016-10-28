/*
 * Copyright (C) 2015-2016  Leo Singer
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
#include "bayestar_distance.h"

#include <gsl/gsl_cblas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics.h>

#include <chealpix.h>


double bayestar_distance_conditional_pdf(
    double r, double mu, double sigma, double norm)
{
    if (!isfinite(mu))
        return 0;

    const double x = -0.5 * gsl_pow_2((r - mu) / sigma);
    const double y = norm * gsl_pow_2(r) / (sqrt(2 * M_PI) * sigma);
    return gsl_sf_exp_mult(x, y);
}


static double ugaussian_integral(double x1, double x2)
{
    if (GSL_SIGN(x1) != GSL_SIGN(x2))
    {
        return gsl_cdf_ugaussian_P(x2) - gsl_cdf_ugaussian_P(x1);
    } else if (x1 > 0) {
        const double logerfc1 = gsl_sf_log_erfc(x1 * M_SQRT1_2);
        const double logerfc2 = gsl_sf_log_erfc(x2 * M_SQRT1_2);
        return 0.5 * (exp(logerfc1) - exp(logerfc2));
    } else {
        const double logerfc1 = gsl_sf_log_erfc(-x1 * M_SQRT1_2);
        const double logerfc2 = gsl_sf_log_erfc(-x2 * M_SQRT1_2);
        return 0.5 * (exp(logerfc2) - exp(logerfc1));
    }
}


double bayestar_distance_conditional_cdf(
    double r, double mu, double sigma, double norm)
{
    if (!isfinite(mu))
        return 0;

    const double mu2 = gsl_pow_2(mu);
    const double sigma2 = gsl_pow_2(sigma);
    const double arg1 = -mu / sigma;
    const double arg2 = (r - mu) / sigma;

    return (
        (mu2 + sigma2) * ugaussian_integral(arg1, arg2)
        + sigma / sqrt(2 * M_PI) * (gsl_sf_exp_mult(-0.5 * gsl_pow_2(arg1), mu)
        - gsl_sf_exp_mult(-0.5 * gsl_pow_2(arg2), r + mu))
    ) * norm;
}


static void conditional_ppf_fdf(double r, void *params, double *f, double *df)
{
    const double p = ((double*) params)[0];
    const double mu = ((double *) params)[1];
    const double norm = ((double *) params)[2];
    *f = bayestar_distance_conditional_cdf(r, mu, 1, norm) - p;
    *df = bayestar_distance_conditional_pdf(r, mu, 1, norm);
}


static double conditional_ppf_f(double r, void *params)
{
    double f, df;
    conditional_ppf_fdf(r, params, &f, &df);
    return f;
}


static double conditional_ppf_df(double r, void *params)
{
    double f, df;
    conditional_ppf_fdf(r, params, &f, &df);
    return df;
}


double bayestar_distance_conditional_ppf(
    double p, double mu, double sigma, double norm)
{
    if (p <= 0)
        return 0;
    else if (p >= 1)
        return GSL_POSINF;
    else if (!(isfinite(p) && isfinite(mu)
            && isfinite(sigma) && isfinite(norm)))
        return GSL_NAN;

    /* Convert to standard distribution with sigma = 1. */
    mu /= sigma;
    norm *= gsl_pow_2(sigma);

    /* Set up variables for tracking progress toward the solution. */
    static const int max_iter = 50;
    double params[] = {p, mu, norm};
    int iter = 0;
    double z = mu > 0 ? mu : 0.5; /* FIXME: better initial guess? */
    int status;

    /* Set up solver (on stack). */
    const gsl_root_fdfsolver_type *algo = gsl_root_fdfsolver_steffenson;
    char state[algo->size];
    gsl_root_fdfsolver solver = {algo, NULL, 0, state};
    gsl_function_fdf fun = {
        conditional_ppf_f, conditional_ppf_df, conditional_ppf_fdf,
        &params};
    gsl_root_fdfsolver_set(&solver, &fun, z);

    do
    {
        const double zold = z;
        status = gsl_root_fdfsolver_iterate(&solver);
        z = gsl_root_fdfsolver_root(&solver);
        status = gsl_root_test_delta (z, zold, 0, GSL_SQRT_DBL_EPSILON);
        iter++;
    } while (status == GSL_CONTINUE && iter < max_iter);
    /* FIXME: do something with status? */

    /* Rescale to original value of sigma. */
    z *= sigma;

    return z;
}


static void integrals(
    double z,
    double *x2, double *x3, double *x4,
    double *dx2, double *dx3, double *dx4)
{
    const double H = gsl_sf_hazard(- z);
    const double Hp = - H * (z + H);
    const double z2 = gsl_pow_2(z);
    *x2 = z2 + 1 + z * H;
    *x3 = z * (z2 + 3) + (z2 + 2) * H;
    *x4 = z2 * (z2 + 6) + 3 + z * (z2 + 5) * H;
    *dx2 = 2 * z + H + z * Hp;
    *dx3 = 3 * (z2 + 1) + 2 * z * H + (z2 + 2) * Hp;
    *dx4 = 4 * z * (z2 + 3) + (3 * z2 + 5) * H + z * (z2 + 5) * Hp;
}


static void moments_to_parameters_fdf(
    double z, void *params, double *fval, double *dfval)
{
    const double mean_std = *(double *)params;
    const double target = 1 / gsl_pow_2(mean_std) + 1;
    double x2, x3, x4, dx2, dx3, dx4;
    integrals(z, &x2, &x3, &x4, &dx2, &dx3, &dx4);
    *fval = target * gsl_pow_2(x3) - x4 * x2;
    *dfval = target * 2 * x3 * dx3 - x4 * dx2 - dx4 * x2;
}


static double moments_to_parameters_f(double z, void *params)
{
    double fval, dfval;
    moments_to_parameters_fdf(z, params, &fval, &dfval);
    return fval;
}


static double moments_to_parameters_df(double z, void *params)
{
    double fval, dfval;
    moments_to_parameters_fdf(z, params, &fval, &dfval);
    return dfval;
}


static int solve_z(double mean_std, double *result)
{
    /* Set up variables for tracking progress toward the solution. */
    static const int max_iter = 50;
    int iter = 0;
    double z = mean_std;
    int status;

    /* Set up solver (on stack). */
    const gsl_root_fdfsolver_type *algo = gsl_root_fdfsolver_steffenson;
    char state[algo->size];
    gsl_root_fdfsolver solver = {algo, NULL, 0, state};
    gsl_function_fdf fun = {
        moments_to_parameters_f,
        moments_to_parameters_df,
        moments_to_parameters_fdf,
        &mean_std};
    gsl_root_fdfsolver_set(&solver, &fun, z);

    do
    {
        const double zold = z;
        status = gsl_root_fdfsolver_iterate(&solver);
        z = gsl_root_fdfsolver_root(&solver);
        status = gsl_root_test_delta (z, zold, 0, GSL_SQRT_DBL_EPSILON);
        iter++;
    } while (status == GSL_CONTINUE && iter < max_iter);

    *result = z;
    return status;
}


int bayestar_distance_moments_to_parameters(
    double mean, double std, double *mu, double *sigma, double *norm)
{
    /* Set up function to solve. */
    double mean_std = mean / std;
    /* Minimum value of (mean/std) for a quadratically weighted
     * normal distribution. The limit of (mean/std) as (mu/sigma) goes to -inf
     * is sqrt(3). We limit (mean/std) to a little bit more than sqrt(3),
     * because as (mu/sigma) becomes more and more negative the normalization
     * has to get very large.
     */
    static const double min_mean_std = M_SQRT3 + 0.1;
    int status;

    if (gsl_finite(mean_std) && mean_std >= min_mean_std)
    {
        double z, x2, x3, x4, dx2, dx3, dx4;
        status = solve_z(mean_std, &z);
        integrals(z, &x2, &x3, &x4, &dx2, &dx3, &dx4);
        *sigma = mean * x2 / x3;
        *mu = *sigma * z;
        *norm = 1 / (gsl_pow_2(*sigma) * x2 * gsl_sf_erf_Q(-z));
    } else {
        status = GSL_SUCCESS;
        *mu = INFINITY;
        *sigma = 1;
        *norm = 0;
    }

    return status;
}


void bayestar_distance_parameters_to_moments(
    double mu, double sigma, double *mean, double *std, double *norm)
{
    if (gsl_finite(mu / sigma))
    {
        const double z = mu / sigma;
        double x2, x3, x4, dx2, dx3, dx4;

        integrals(z, &x2, &x3, &x4, &dx2, &dx3, &dx4);

        *mean = sigma * x3 / x2;
        *std = *mean * sqrt(x4 * x2 / gsl_pow_2(x3) - 1);
        *norm = 1 / (gsl_pow_2(sigma) * x2 * gsl_sf_erf_Q(-z));
    } else {
        *mean = INFINITY;
        *std = 1;
        *norm = 0;
    }
}


double bayestar_volume_render(
    double x, double y, double max_distance, int axis0, int axis1,
    const double *R, long nside, int nest,
    const double *prob, const double *mu,
    const double *sigma, const double *norm)
{
    /* Determine which axis to integrate over
     * (the one that isn't in the args) */
    int axis2;
    int axes[] = {0, 0, 0};
    axes[axis0] = 1;
    axes[axis1] = 1;
    for (axis2 = 0; axes[axis2]; axis2++)
        ; /* loop body intentionally no-op */

    /* Construct grid in theta, the elevation angle from the
     * spatial origin to the plane of the screen. */

    /* Transverse distance from origin to point on screen */
    const double a = sqrt(gsl_pow_2(x) + gsl_pow_2(y));

    /* Maximum value of theta (at edge of screen-aligned cube) */
    const double theta_max = atan2(max_distance, a);
    const double dtheta = 0.5 * M_PI / nside / 4;

    /* Construct regular grid from -theta_max to +theta_max */
    double ret = 0;
    for (double theta = -theta_max; theta <= theta_max; theta += dtheta)
    {
        /* Differential z = a tan(theta),
         * dz = dz/dtheta dtheta = a tan'(theta) dtheta = a sec^2(theta) dtheta,
         * and dtheta = const */
        const double dz_dtheta = a / gsl_pow_2(cos(theta));
        const double z = a * tan(theta);
        double xyz[3];
        xyz[axis0] = x;
        xyz[axis1] = y;
        xyz[axis2] = z;

       /* Transform from screen-aligned cube to celestial coordinates before
        * looking up pixel indices. */
        double vec[3];
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, 3, 3, 1, R, 3, xyz, 1, 0, vec, 1);
        long ipix;
        if (nest)
            vec2pix_nest(nside, vec, &ipix);
        else
            vec2pix_ring(nside, vec, &ipix);
        double r = sqrt(gsl_pow_2(x) + gsl_pow_2(y) + gsl_pow_2(z));

        if (isfinite(mu[ipix]))
            ret += gsl_sf_exp_mult(
                -0.5 * gsl_pow_2((r - mu[ipix]) / sigma[ipix]),
                prob[ipix] / sigma[ipix] * norm[ipix] * dz_dtheta);
    }

    return ret / (M_SQRT2 * M_SQRTPI);
}


double bayestar_distance_marginal_pdf(
    double r, long npix,
    const double *prob, const double *mu,
    const double *sigma, const double *norm)
{
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < npix; i ++)
        sum += prob[i] * bayestar_distance_conditional_pdf(
            r, mu[i], sigma[i], norm[i]);
    return sum;
}


double bayestar_distance_marginal_cdf(
    double r, long npix,
    const double *prob, const double *mu,
    const double *sigma, const double *norm)
{
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < npix; i ++)
        sum += prob[i] * bayestar_distance_conditional_cdf(
            r, mu[i], sigma[i], norm[i]);
    return sum;
}


typedef struct {
    double p;
    long npix;
    const double *prob;
    const double *mu;
    const double *sigma;
    const double *norm;
} marginal_ppf_params;


static void marginal_ppf_fdf(double r, void *params, double *f, double *df)
{
    const marginal_ppf_params *p = (marginal_ppf_params *)params;
    const double _f = bayestar_distance_marginal_cdf(
        r, p->npix, p->prob, p->mu, p->sigma, p->norm);
    const double _df = bayestar_distance_marginal_pdf(
        r, p->npix, p->prob, p->mu, p->sigma, p->norm);
    if (p->p > 0.5)
    {
        *f = log(1 - _f) - log(1 - p->p);
        *df = -_df / (1 - _f);
    } else {
        *f = log(_f) - log(p->p);
        *df = _df / _f;
    }
}


static double marginal_ppf_f(double r, void *params)
{
    double f, df;
    marginal_ppf_fdf(r, params, &f, &df);
    return f;
}


static double marginal_ppf_df(double r, void *params)
{
    double f, df;
    marginal_ppf_fdf(r, params, &f, &df);
    return df;
}


double bayestar_distance_marginal_ppf(
    double p, long npix,
    const double *prob, const double *mu,
    const double *sigma, const double *norm)
{
    if (p <= 0)
        return 0;
    else if (p >= 1)
        return GSL_POSINF;
    else if (!isfinite(p))
        return GSL_NAN;

    /* Set up variables for tracking progress toward the solution. */
    static const int max_iter = 50;
    marginal_ppf_params params = {p, npix, prob, mu, sigma, norm};
    int iter = 0;
    const size_t max_ipix = gsl_stats_max_index(prob, 1, npix);
    double r = bayestar_distance_conditional_ppf(
        p, mu[max_ipix], sigma[max_ipix], norm[max_ipix]);
    int status;

    /* Set up solver (on stack). */
    const gsl_root_fdfsolver_type *algo = gsl_root_fdfsolver_steffenson;
    char state[algo->size];
    gsl_root_fdfsolver solver = {algo, NULL, 0, state};
    gsl_function_fdf fun = {
        marginal_ppf_f, marginal_ppf_df, marginal_ppf_fdf,
        &params};
    gsl_root_fdfsolver_set(&solver, &fun, r);

    do
    {
        const double rold = r;
        status = gsl_root_fdfsolver_iterate(&solver);
        r = gsl_root_fdfsolver_root(&solver);
        status = gsl_root_test_delta (r, rold, 0, GSL_SQRT_DBL_EPSILON);
        iter++;
    } while (status == GSL_CONTINUE && iter < max_iter);
    /* FIXME: do something with status? */

    return r;
}

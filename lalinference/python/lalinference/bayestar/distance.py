#
# Copyright (C) 2016  Leo Singer
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
"""
Distance ansatz functions.
"""
from __future__ import division
__author__ = "Leo Singer <leo.singer@ligo.org>"


import numpy as np
import healpy as hp
import scipy.special
from . import _distance
from ._distance import *


__all__ = tuple(_ for _ in _distance.__dict__ if not _.startswith('_')) + (
    'ud_grade', 'conditional_kde', 'cartesian_kde_to_moments', 'principal_axes',
    'parameters_to_moments', 'find_injection_distance')


def _add_newdoc_ufunc(func, doc):
    # The function `np.lib.add_newdoc_ufunc` can only change a ufunc's
    # docstring if it is `NULL`. This workaround avoids an exception
    # when the user tries to `reload()` this module.
    try:
        np.lib.add_newdoc_ufunc(func, doc)
    except ValueError as e:
        if e.message == 'Cannot change docstring of ufunc with non-NULL docstring':
            pass


_add_newdoc_ufunc(conditional_pdf, """\
Conditional distance probability density function (ansatz).

Parameters
----------
r : `numpy.ndarray`
    Distance (Mpc)
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)

Returns
-------
pdf : `numpy.ndarray`
    Conditional probability density according to ansatz.
""")


_add_newdoc_ufunc(conditional_cdf, """\
Cumulative conditional distribution of distance (ansatz).

Parameters
----------
r : `numpy.ndarray`
    Distance (Mpc)
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)

Returns
-------
pdf : `numpy.ndarray`
    Conditional probability density according to ansatz.

Test against numerical integral of pdf:
>>> import scipy.integrate
>>> distmu = 10.0
>>> distsigma = 5.0
>>> distnorm = 1.0
>>> r = 8.0
>>> expected, _ = scipy.integrate.quad(
...     conditional_pdf, 0, r,
...     (distmu, distsigma, distnorm))
>>> result = conditional_cdf(
...     r, distmu, distsigma, distnorm)
>>> np.testing.assert_almost_equal(expected, result)
""")


_add_newdoc_ufunc(conditional_ppf, """\
Point percent function (inverse cdf) of distribution of distance (ansatz).

Parameters
----------
p : `numpy.ndarray`
    The cumulative distribution function
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)

Returns
-------
r : `numpy.ndarray`
    Distance at which the cdf is equal to `p`.

Test against numerical estimate:
>>> import scipy.optimize
>>> distmu = 10.0
>>> distsigma = 5.0
>>> distnorm = 1.0
>>> p = 0.16  # "one-sigma" lower limit
>>> expected_r16 = scipy.optimize.brentq(lambda r: conditional_cdf(r, distmu, distsigma, distnorm) - p, 0.0, 100.0)
>>> r16 = conditional_ppf(p, distmu, distsigma, distnorm)
>>> np.testing.assert_almost_equal(expected_r16, r16)
""")


_add_newdoc_ufunc(moments_to_parameters, """\
Convert ansatz moments to parameters.
This function is the inverse of `parameters_to_moments`.

Parameters
----------
distmean : `numpy.ndarray`
    Conditional mean of distance (Mpc)
diststd : `numpy.ndarray`
    Conditional standard deviation of distance (Mpc)

Returns
-------
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)
""")


_add_newdoc_ufunc(parameters_to_moments, """\
Convert ansatz parameters to moments.
This function is the inverse of `moments_to_parameters`.

Parameters
----------
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)

Returns
-------
distmean : `numpy.ndarray`
    Conditional mean of distance (Mpc)
diststd : `numpy.ndarray`
    Conditional standard deviation of distance (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)

For mu=0, sigma=1, the ansatz is a chi distribution with 3 degrees of
freedom, and the moments have simple expressions.
>>> mean, std, norm = parameters_to_moments(0, 1)
>>> expected_mean = 2 * np.sqrt(2 / np.pi)
>>> expected_std = np.sqrt(3 - expected_mean**2)
>>> expected_norm = 2.0
>>> np.testing.assert_allclose(mean, expected_mean)
>>> np.testing.assert_allclose(std, expected_std)
>>> np.testing.assert_allclose(norm, expected_norm)

Check that the moments scale as expected when we vary sigma.
>>> sigma = np.logspace(-8, 8)
>>> mean, std, norm = parameters_to_moments(0, sigma)
>>> np.testing.assert_allclose(mean, expected_mean * sigma)
>>> np.testing.assert_allclose(std, expected_std * sigma)
>>> np.testing.assert_allclose(norm, expected_norm / sigma**2)

Check some more arbitrary values using numerical quadrature:
>>> sigma = 1.0
>>> for mu in np.linspace(-10, 10):
...     mean, std, norm = parameters_to_moments(mu, sigma)
...     moments = np.empty(3)
...     for k in range(3):
...         moments[k], _ = scipy.integrate.quad(
...             lambda r: r**k * conditional_pdf(r, mu, sigma, 1.0),
...             0, np.inf)
...     expected_norm = 1 / moments[0]
...     expected_mean, r2 = moments[1:] * expected_norm
...     expected_std = np.sqrt(r2 - np.square(expected_mean))
...     np.testing.assert_approx_equal(mean, expected_mean, 5)
...     np.testing.assert_approx_equal(std, expected_std, 5)
...     np.testing.assert_approx_equal(norm, expected_norm, 5)
""")


_add_newdoc_ufunc(volume_render, """\
Perform volumetric rendering of a 3D sky map.

Parameters
----------
x : `numpy.ndarray`
    X-coordinate in rendered image
y : `numpy.ndarray`
    Y-coordinate in rendered image
max_distance : float
    Limit of integration from `-max_distance` to `+max_distance`
axis0 : int
    Index of axis to assign to x-coordinate
axis1 : int
    Index of axis to assign to y-coordinate
R : `numpy.ndarray`
    Rotation matrix as provided by `principal_axes`
nest : bool
    HEALPix ordering scheme
prob : `numpy.ndarray`
    Marginal probability (pix^-2)
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)

Returns
-------
image : `numpy.ndarray`
    Rendered image
""")


_add_newdoc_ufunc(marginal_pdf, """\
Calculate all-sky marginal pdf (ansatz).

Parameters
----------
r : `numpy.ndarray`
    Distance (Mpc)
prob : `numpy.ndarray`
    Marginal probability (pix^-2)
distmu : `numpy.ndarray`
    Distance location parameter (Mpc)
distsigma : `numpy.ndarray`
    Distance scale parameter (Mpc)
distnorm : `numpy.ndarray`
    Distance normalization factor (Mpc^-2)

Returns
-------
pdf : `numpy.ndarray`
    Conditional probability density according to ansatz.
""")


def ud_grade(prob, distmu, distsigma, *args, **kwargs):
    """
    Upsample or downsample a distance-resolved sky map.

    Parameters
    ----------
    prob : `numpy.ndarray`
        Marginal probability (pix^-2)
    distmu : `numpy.ndarray`
        Distance location parameter (Mpc)
    distsigma : `numpy.ndarray`
        Distance scale parameter (Mpc)
    distnorm : `numpy.ndarray`
        Distance normalization factor (Mpc^-2)
    *args, **kwargs :
        Additional arguments to `healpy.ud_grade` (e.g.,
        `nside`, `order_in`, `order_out`).

    Returns
    -------
    prob : `numpy.ndarray`
        Resampled marginal probability (pix^-2)
    distmu : `numpy.ndarray`
        Resampled distance location parameter (Mpc)
    distsigma : `numpy.ndarray`
        Resampled distance scale parameter (Mpc)
    distnorm : `numpy.ndarray`
        Resampled distance normalization factor (Mpc^-2)
    """
    bad = ~(np.isfinite(distmu) & np.isfinite(distsigma))
    distmean, diststd, _ = parameters_to_moments(distmu, distsigma)
    distmean[bad] = 0
    diststd[bad] = 0
    distmean = hp.ud_grade(prob * distmu, *args, power=-2, **kwargs)
    diststd = hp.ud_grade(prob * np.square(diststd), *args, power=-2, **kwargs)
    prob = hp.ud_grade(prob, *args, power=-2, **kwargs)
    distmean /= prob
    diststd = np.sqrt(diststd / prob)
    bad = ~hp.ud_grade(~bad, *args, power=-2, **kwargs)
    distmean[bad] = np.inf
    diststd[bad] = 1
    distmu, distsigma, distnorm = moments_to_parameters(distmean, diststd)
    return prob, distmu, distsigma, distnorm


def _conditional_kde(n, X, Cinv, W):
    Cinv_n = np.dot(Cinv, n)
    cinv = np.dot(n, Cinv_n)
    x = np.dot(Cinv_n, X) / cinv
    w = W * (0.5 / np.pi) * np.sqrt(np.linalg.det(Cinv) / cinv) * np.exp(
        0.5 * (np.square(x) * cinv - (np.dot(Cinv, X) * X).sum(0)))
    return x, cinv, w


def conditional_kde(n, datasets, inverse_covariances, weights):
    return [_conditional_kde(n, X, Cinv, W)
        for X, Cinv, W in zip(datasets, inverse_covariances, weights)]


def cartesian_kde_to_moments(n, datasets, inverse_covariances, weights):
    """
    Calculate the marginal probability, conditional mean, and conditional
    standard deviation of a mixture of three-dimensional kernel density
    estimators (KDEs), in a given direction specified by a unit vector.

    Parameters
    ----------
    n : `numpy.ndarray`
        A unit vector; an array of length 3.
    datasets : list of `numpy.ndarray`
        A list 2D Numpy arrays specifying the sample points of the KDEs.
        The first dimension of each array is 3.
    inverse_covariances: list of `numpy.ndarray`
        An array of 3x3 matrices specifying the inverses of the covariance
        matrices of the KDEs. The list has the same length as the datasets
        parameter.
    weights : list
        A list of floating-point weights.

    Returns
    -------
    prob : float
        The marginal probability in direction n, integrated over all distances.
    mean : float
        The conditional mean in direction n.
    std : float
        The conditional standard deviation in direction n.

    >>> # Some imports
    >>> import scipy.stats
    >>> import scipy.integrate
    >>> # Construct random dataset for KDE
    >>> np.random.seed(0)
    >>> nclusters = 5
    >>> ndata = np.random.randint(0, 1000, nclusters)
    >>> covs = [np.random.uniform(0, 1, size=(3, 3)) for _ in range(nclusters)]
    >>> covs = [_ + _.T + 3 * np.eye(3) for _ in covs]
    >>> means = np.random.uniform(-1, 1, size=(nclusters, 3))
    >>> datasets = [np.random.multivariate_normal(m, c, n).T
    ...     for m, c, n in zip(means, covs, ndata)]
    >>> weights = ndata / float(np.sum(ndata))
    >>>
    >>> # Construct set of KDEs
    >>> kdes = [scipy.stats.gaussian_kde(_) for _ in datasets]
    >>>
    >>> # Random unit vector n
    >>> n = np.random.normal(size=3)
    >>> n /= np.sqrt(np.sum(np.square(n)))
    >>>
    >>> # Analytically evaluate conditional mean and std. dev. in direction n
    >>> datasets = [_.dataset for _ in kdes]
    >>> inverse_covariances = [_.inv_cov for _ in kdes]
    >>> result_prob, result_mean, result_std = cartesian_kde_to_moments(
    ...     n, datasets, inverse_covariances, weights)
    >>>
    >>> # Numerically integrate conditional distance moments
    >>> def rkbar(k):
    ...     def integrand(r):
    ...         return r ** k * np.sum([kde(r * n) * weight
    ...             for kde, weight in zip(kdes, weights)])
    ...     integral, err = scipy.integrate.quad(integrand, 0, np.inf)
    ...     return integral
    ...
    >>> r0bar = rkbar(2)
    >>> r1bar = rkbar(3)
    >>> r2bar = rkbar(4)
    >>>
    >>> # Extract conditional mean and std. dev.
    >>> r1bar /= r0bar
    >>> r2bar /= r0bar
    >>> expected_prob = r0bar
    >>> expected_mean = r1bar
    >>> expected_std = np.sqrt(r2bar - np.square(r1bar))
    >>>
    >>> # Check that the two methods give almost the same result
    >>> np.testing.assert_almost_equal(result_prob, expected_prob)
    >>> np.testing.assert_almost_equal(result_mean, expected_mean)
    >>> np.testing.assert_almost_equal(result_std, expected_std)
    >>>
    >>> # Check that KDE is normalized over unit sphere.
    >>> nside = 32
    >>> npix = hp.nside2npix(nside)
    >>> prob, _, _ = np.transpose([cartesian_kde_to_moments(
    ...     np.asarray(hp.pix2vec(nside, ipix)),
    ...     datasets, inverse_covariances, weights)
    ...     for ipix in range(npix)])
    >>> result_integral = prob.sum() * hp.nside2pixarea(nside)
    >>> np.testing.assert_almost_equal(result_integral, 1.0, decimal=4)
    """
    # Initialize moments of conditional KDE.
    r0bar = 0
    r1bar = 0
    r2bar = 0

    # Loop over KDEs.
    for X, Cinv, W in zip(datasets, inverse_covariances, weights):
        x, cinv, w = _conditional_kde(n, X, Cinv, W)

        # Accumulate moments of conditional KDE.
        c = 1 / cinv
        x2 = np.square(x)
        a = scipy.special.ndtr(x * np.sqrt(cinv))
        b = np.sqrt(0.5 / np.pi * c) * np.exp(-0.5 * cinv * x2)
        r0bar_ = (x2 + c) * a + x * b
        r1bar_ = x * (x2 + 3 * c) * a + (x2 + 2 * c) * b
        r2bar_ = (x2 * x2 + 6 * x2 * c + 3 * c * c) * a + x * (x2 + 5 * c) * b
        r0bar += np.mean(w * r0bar_)
        r1bar += np.mean(w * r1bar_)
        r2bar += np.mean(w * r2bar_)

    # Normalize moments.
    r1bar /= r0bar
    r2bar /= r0bar
    var = r2bar - np.square(r1bar)

    # Handle invalid values.
    if var >= 0:
        mean = r1bar
        std = np.sqrt(var)
    else:
        mean = np.inf
        std = 1.0
    prob = r0bar

    # Done!
    return prob, mean, std


def principal_axes(prob, distmu, distsigma, nest=False):
    npix = len(prob)
    nside = hp.npix2nside(npix)
    bad = ~(np.isfinite(prob) & np.isfinite(distmu) & np.isfinite(distsigma))
    distmean, diststd, _ = parameters_to_moments(distmu, distsigma)
    mass = prob * (np.square(diststd) + np.square(distmean))
    mass[bad] = 0.0
    xyz = np.asarray(hp.pix2vec(nside, np.arange(npix), nest=nest))
    cov = np.dot(xyz * mass, xyz.T)
    L, V = np.linalg.eigh(cov)
    if np.linalg.det(V) < 0:
        V = -V
    return V


def parameters_to_marginal_moments(prob, distmu, distsigma):
    """Calculate the marginal (integrated all-sky) mean and standard deviation
    of distance from the ansatz parameters.

    Parameters
    ----------
    prob : `numpy.ndarray`
        Marginal probability (pix^-2)
    distmu : `numpy.ndarray`
        Distance location parameter (Mpc)
    distsigma : `numpy.ndarray`
        Distance scale parameter (Mpc)

    Returns
    -------
    distmean : float
        Mean distance (Mpc)
    diststd : float
        Std. deviation of distance (Mpc)
    """
    good = np.isfinite(prob) & np.isfinite(distmu) & np.isfinite(distsigma)
    prob = prob[good]
    distmu = distmu[good]
    distsigma = distsigma[good]
    distmean, diststd, _ = parameters_to_moments(distmu, distsigma)
    rbar = (prob * distmean).sum()
    r2bar = (prob * (np.square(diststd) + np.square(distmean))).sum()
    return rbar, np.sqrt(r2bar - np.square(rbar))


def find_injection_distance(true_dist, prob, distmu, distsigma, distnorm):
    return np.sum(prob * conditional_cdf(
        true_dist, distmu, distsigma, distnorm))

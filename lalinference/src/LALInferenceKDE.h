/*
 *  LALInferenceKDE.h:  Bayesian Followup, kernel density estimator.
 *
 *  Copyright (C) 2013 Ben Farr
 *
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
#ifndef LALInferenceKDE_h
#define LALInferenceKDE_h

#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/LALDatatypes.h>
#include <lal/LALInference.h>

struct tagkmeans;

/**
 * Structure containing the Guassian kernel density of a set of samples.
 */
typedef struct
tagKDE
{
    gsl_matrix *data;                       /**< Data to estimate the underlying distribution of */
    INT4 dim;                               /**< Dimension of points in \a data. */
    INT4 npts;                              /**< Number of points in \a data. */
    REAL8 bandwidth;                        /**< Bandwidth of kernels. */
    REAL8 log_norm_factor;                  /**< Normalization factor of the KDE. */
    gsl_vector * mean;                      /**< The mean of \a data */
    gsl_matrix * cov;                       /**< The covariance matrix of \a data */
    gsl_matrix * cholesky_decomp_cov;       /**< The Cholesky decomposition of the
                                                  covariance matrix, containing both terms
                                                  as returned by gsl_linalg_cholesky_decomp(). */
    gsl_matrix * cholesky_decomp_cov_lower; /**< Just the lower portion of \a cholesky_decomp_cov. */

    LALInferenceParamVaryType * lower_bound_types; /**< Array of param boundary types */
    LALInferenceParamVaryType * upper_bound_types; /**< Array of param boundary types */
    REAL8 * lower_bounds;              /**< Lower param bounds */
    REAL8 * upper_bounds;              /**< Upper param bounds */
} LALInferenceKDE;

/* Allocate, fill, and tune a Gaussian kernel density estimate given an array of points. */
LALInferenceKDE *LALInferenceNewKDE(REAL8 *pts, INT4 npts, INT4 dim, INT4 *mask);

/* Allocate, fill, and tune a Gaussian kernel density estimate given a matrix of points. */
LALInferenceKDE *LALInferenceNewKDEfromMat(gsl_matrix *data, INT4 *mask);

/* Allocate, fill, and tune a Gaussian kernel density estimate given a matrix of points. */
LALInferenceKDE *LALInferenceInitKDE(INT4 npts, INT4 dim);

/* Free an allocated KDE structure. */
void LALInferenceDestroyKDE(LALInferenceKDE *kde);

/* Calculate the bandwidth and normalization factor for a KDE. */
void LALInferenceSetKDEBandwidth(LALInferenceKDE *kde);

/* Evaluate the (log) PDF from a KDE at a single point. */
REAL8 LALInferenceKDEEvaluatePoint(LALInferenceKDE *kde, REAL8 *point);

/* Draw a sample from a kernel density estimate. */
REAL8 *LALInferenceDrawKDESample(LALInferenceKDE *kde, gsl_rng *rng);

/* Compute the Cholesky decomposition of a matrix. */
INT4 LALInferenceCholeskyDecompose(gsl_matrix *mat);

/* Calculate the covariance matrix of a data set. */
void LALInferenceComputeCovariance(gsl_matrix *cov, gsl_matrix *pts);

/* Calculate the mean of a data set. */
void LALInferenceComputeMean(gsl_vector *mean, gsl_matrix *points);

/* Calculate the determinant of a matrix. */
REAL8 LALInferenceMatrixDet(gsl_matrix *mat);

/* Determine the log of the sum of an array of exponentials. */
REAL8 log_add_exps(REAL8 *vals, INT4 size);

#endif

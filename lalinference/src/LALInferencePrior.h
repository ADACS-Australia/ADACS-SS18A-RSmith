/*
 *
 *  LALInference:          LAL Inference library
 *  LALInferencePrior.h:   Collection of common priors
 *
 *  Copyright (C) 2009 Ilya Mandel, Vivien Raymond, Christian Roever, Marc van der Sluys and John Veitch
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

/**
 * \file LALInferencePrior.h
 * \brief Collection of commonly used Prior functions and utilities
 * \ingroup LALInference
 * 
 * This file contains 
 * 
 */

#ifndef LALInferencePrior_h
#define LALInferencePrior_h


#include <lal/LALInference.h>
#include <lal/LALInferenceNestedSampler.h>
/** Return the logarithmic prior density of the variables specified, for the non-spinning/spinning inspiral signal case.
 */
REAL8 LALInferenceInspiralPrior(LALInferenceRunState *runState, LALInferenceVariables *variables);

/** Apply cyclic and reflective boundaries to \c parameter to bring it
 *  back within the allowed prior ranges that are specified in \c
 *  priorArgs.  LALInferenceCyclicReflectiveBound() should not be
 *  called after any multi-parameter update step in a jump proposal,
 *  as this violates detailed balance.
 *
 *  \param parameter [in] Pointer to an array of parameters
 *  \param priorArgs [in] Pointer to an array of prior ranges
 */
void LALInferenceCyclicReflectiveBound(LALInferenceVariables *parameter, LALInferenceVariables *priorArgs);

/** \brief Rotate initial phase if polarisation angle is cyclic around ranges
 * 
 *  If the polarisation angle parameter \f$\psi\f$ is cyclic about its upper and
 *  lower ranges of \f$-\pi/4\f$ to \f$\pi/4\f$ then the transformation for
 *  crossing a boundary requires the initial phase parameter \f$\phi_0\f$ to be
 *  rotated through \f$\pi\f$ radians. The function assumes the value of
 *  \f$\psi\f$ has been rescaled to be between 0 and \f$2\pi\f$ - this is a
 *  requirement of the covariance matrix routine \c LALInferenceNScalcCVM
 *  function.  
 * 
 *  This is particularly relevant for pulsar analyses.
 * 
 *  \param parameter [in] Pointer to an array of parameters
 */
void LALInferenceRotateInitialPhase( LALInferenceVariables *parameter );

/** Return the logarithmic prior density of the variables as specified for the sky localisation project 
 *  (see: https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/SkyLocComparison#priors ), 
 *  for the non-spinning/spinning inspiral signal case.
 */
REAL8 LALInferenceInspiralSkyLocPrior(LALInferenceRunState *runState, LALInferenceVariables *params);

/** Return the logarithmic prior density of the variables specified, 
 *  for the non-spinning/spinning inspiral signal case.
 */
REAL8 LALInferenceInspiralPriorNormalised(LALInferenceRunState *runState, LALInferenceVariables *params);

/** Function to add the minimum and maximum values for the uniform prior onto the \c priorArgs. 
 */
void LALInferenceAddMinMaxPrior(LALInferenceVariables *priorArgs, const char *name, REAL8 *min, REAL8 *max, LALInferenceVariableType type);

/** Get the minimum and maximum values of the uniform prior from the \c priorArgs list, given a name. 
 */
void LALInferenceGetMinMaxPrior(LALInferenceVariables *priorArgs, const char *name, REAL8 *min, REAL8 *max);

/** Function to remove the mininum and maximum values for the uniform prior onto the \c priorArgs. 
 */
void LALInferenceRemoveMinMaxPrior(LALInferenceVariables *priorArgs, const char *name);

/** Function to add the mu and sigma values for the Gaussian prior onto the \c priorArgs. 
 */
void LALInferenceAddGaussianPrior(LALInferenceVariables *priorArgs, 
                                  const char *name, REAL8 *mu, REAL8 *sigma,
                                  LALInferenceVariableType type);

/** Get the mu and sigma values of the Gaussian prior from the \c priorArgs list, given a name. 
 */
void LALInferenceGetGaussianPrior(LALInferenceVariables *priorArgs, 
                                  const char *name, REAL8 *mu, REAL8 *sigma);


/** Function to remove the mu and sigma values for the Gaussian prior onto the \c priorArgs. 
 */
void LALInferenceRemoveGaussianPrior(LALInferenceVariables *priorArgs, const char *name);

/** Check for types of standard prior */
/** Check for a uniform prior (with mininum and maximum) */
int LALInferenceCheckMinMaxPrior(LALInferenceVariables *priorArgs, const char *name);
/** Check for a Gaussian prior (with a mean and variance) */
int LALInferenceCheckGaussianPrior(LALInferenceVariables *priorArgs, const char *name);

/** Function to add a correlation matrix and parameter index for a prior
 * defined as part of a multivariate Gaussian distribution onto the \c
 * priorArgs. The correlation coefficient matrix must be a gsl_matrix and the
 * index for the given parameter in the matrix must be supplied. 
 */
void LALInferenceAddCorrelatedPrior( LALInferenceVariables *priorArgs, 
                                     const char *name, gsl_matrix **cor, 
                                     UINT4 *idx );

/** Get the correlation coefficient matrix and index for a parameter from the
 * \c priorArgs list.
 */ 
void LALInferenceGetCorrelatedPrior( LALInferenceVariables *priorArgs, 
                                     const char *name, gsl_matrix **cor, 
                                     UINT4 *idx );

/** Remove the correlation coefficient matrix and index for a parameter from the
 * \c priorArgs list.
 */ 
void LALInferenceRemoveCorrelatedPrior( LALInferenceVariables *priorArgs, 
                                        const char *name );

/** Check for the existance of a correlation coefficient matrix and index for
 * a parameter from the \c priorArgs list.
 */ 
int LALInferenceCheckCorrelatedPrior( LALInferenceVariables *priorArgs, 
                                      const char *name );

/** Draw variables from the prior ranges */
void LALInferenceDrawFromPrior( LALInferenceVariables *output, 
                                LALInferenceVariables *priorArgs, 
                                gsl_rng *rdm );

/** Draw an individual variable from its prior range */
void LALInferenceDrawNameFromPrior( LALInferenceVariables *output, 
                                    LALInferenceVariables *priorArgs, 
                                    char *name, LALInferenceVariableType type, 
                                    gsl_rng *rdm );

/** Prior that is 1 everywhere in component mass space. */
REAL8 LALInferenceAnalyticNullPrior(LALInferenceRunState *runState, LALInferenceVariables *params);

/** Prior that is 1 everywhere. */
REAL8 LALInferenceNullPrior(LALInferenceRunState *runState, LALInferenceVariables *params);

/** Computes the numerical normalization of the mass prior \f$p(\mathcal{M}) \sim
    \mathcal{M}^{-11/6}\f$ applying all cuts in the mass plane implied by the
    various component, total, and chirp mass limits, and the mass
    ratio limits.  Returns the integral of \f$\mathcal{M}^{-11/6}\f$ over the allowed
    ranges in mass. */
REAL8 LALInferenceComputePriorMassNorm(const double MMin, const double MMax, const double MTotMax, 
                    const double McMin, const double McMax,
                    const double massRatioMin, const double massRatioMax, const char *massRatioName);

/** Prior that checks for minimum and maximum prior range specified in runState->priorArgs
    and returns 0.0 if sample lies inside the boundaries, -DBL_MAX otherwise.
    Can be used with MinMaxPrior functions.
    Ignores variables which are not REAL8 or do not have min and max values set.
*/
REAL8 LALInferenceFlatBoundedPrior(LALInferenceRunState *runState, LALInferenceVariables *params);


#endif

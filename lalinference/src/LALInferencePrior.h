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
 */

#ifndef LALInferencePrior_h
#define LALInferencePrior_h


#include <lal/LALInference.h>

REAL8 LALInferenceInspiralPrior(LALInferenceRunState *runState, LALInferenceVariables *variables);
void LALInferenceCyclicReflectiveBound(LALInferenceVariables *parameter, LALInferenceVariables *priorArgs);
void LALInferenceRotateInitialPhase( LALInferenceVariables *parameter );

REAL8 LALInferenceInspiralSkyLocPrior(LALInferenceRunState *runState, LALInferenceVariables *params);
REAL8 LALInferenceInspiralPriorNormalised(LALInferenceRunState *runState,
LALInferenceVariables *params);

void LALInferenceAddMinMaxPrior(LALInferenceVariables *priorArgs, const char *name, void *min, void *max, LALInferenceVariableType type);
void LALInferenceGetMinMaxPrior(LALInferenceVariables *priorArgs, const char *name, void *min, void *max);
void LALInferenceRemoveMinMaxPrior(LALInferenceVariables *priorArgs, const char *name);

void LALInferenceAddGaussianPrior(LALInferenceVariables *priorArgs, const char *name, void *mu,
  void *sigma, LALInferenceVariableType type);
void LALInferenceGetGaussianPrior(LALInferenceVariables *priorArgs, const char *name, void *mu,
  void *sigma);
void LALInferenceRemoveGaussianPrior(LALInferenceVariables *priorArgs, const char *name);

/** Check for types of standard prior */
/** Check for a flat prior with a min and max */
int LALInferenceCheckMinMaxPrior(LALInferenceVariables *priorArgs, const char *name);
/** Check for a Gaussian prior (with a mean and variance) */
int LALInferenceCheckGaussianPrior(LALInferenceVariables *priorArgs, const char *name);


#endif

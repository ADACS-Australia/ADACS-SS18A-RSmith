/*******************************************************************************
  Matt Pitkin, Colin Gill, John Veitch - 2011

  ppe_likelihood.h

  Header file for ppe_likelihood.c

*******************************************************************************/

/*
  Author:
*/

/**
 * \file
 * \ingroup lalapps_pulsar
 * \author Matthew Pitkin, John Veitch, Colin Gill
 *
 * \brief Header file for the likelihood and prior functions used in parameter
 * estimation code for known pulsar searches using the nested sampling
 * algorithm.
 */

#ifndef _PPE_LIKELIHOOD_H
#define _PPE_LIKELIHOOD_H

#include "pulsar_parameter_estimation_nested.h"

#ifdef __cplusplus
extern "C" {
#endif

/* likelihood function */
REAL8 pulsar_log_likelihood( LALInferenceVariables *vars,
                             LALInferenceIFOData *data,
                             LALInferenceModel *get_pulsar_model );

/* noise only likelihood */
REAL8 noise_only_likelihood( LALInferenceRunState *runState );

/* prior function */
REAL8 priorFunction( LALInferenceRunState *runState,
		     LALInferenceVariables *params,
		     LALInferenceModel *mode );

/* check params are within prior range */
UINT4 in_range( LALInferenceVariables *priors, LALInferenceVariables *params );

/* convert nested samples to posterior samples */
void ns_to_posterior( LALInferenceRunState *runState );

/* create kd-tree prior */
void create_kdtree_prior( LALInferenceRunState *runState );

#ifdef __cplusplus
}
#endif

#endif /* _PPE_LIKELIHOOD_H */

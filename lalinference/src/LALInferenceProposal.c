/* 
 *  LALInferenceProposal.c:  Bayesian Followup, jump proposals.
 *
 *  Copyright (C) 2011 Ilya Mandel, Vivien Raymond, Christian Roever,
 *  Marc van der Sluys, John Veitch, Will M. Farr
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

#include <stdio.h>
#include <stdlib.h>
#include <lal/LALInspiral.h>
#include <lal/DetResponse.h>
#include <lal/SeqFactories.h>
#include <lal/Date.h>
#include <lal/VectorOps.h>
#include <lal/TimeFreqFFT.h>
#include <lal/GenerateInspiral.h>
#include <lal/TimeDelay.h>
#include <lal/SkyCoordinates.h>
#include <lal/LALInference.h>
#include <lal/LALInferencePrior.h>
#include <lal/LALInferenceLikelihood.h>
#include <lal/LALInferenceTemplate.h>
#include <lal/LALInferenceProposal.h>
#include <lal/XLALError.h>

#include <lal/LALStdlib.h>

const char *cycleArrayName = "Proposal Cycle";
const char *cycleArrayLengthName = "Proposal Cycle Length";
const char *cycleArrayCounterName = "Proposal Cycle Counter";

const char *LALInferenceSigmaJumpName = "sigmaJump";
const char *LALInferenceCurrentProposalName = "Current Proposal";

/* Proposal Names */
const char *singleAdaptProposalName = "Single";
const char *singleProposalName = "Single";
const char *orbitalPhaseJumpName = "OrbitalPhase";
const char *inclinationDistanceName = "InclinationDistance";
const char *covarianceEigenvectorJumpName = "CovarianceEigenvector";
const char *skyLocWanderJumpName = "SkyLocWander";
const char *differentialEvolutionFullName = "DifferentialEvolutionFull";
const char *differentialEvolutionMassesName = "DifferentialEvolutionMasses";
const char *differentialEvolutionAmpName = "DifferentialEvolutionAmp";
const char *differentialEvolutionSpinsName = "DifferentialEvolutionSpins";
const char *differentialEvolutionSkyName = "DifferentialEvolutionSky";
const char *drawApproxPriorName = "DrawApproxPrior";
const char *skyReflectDetPlaneName = "SkyReflectDetPlane";
const char *rotateSpinsName = "RotateSpins";
const char *polarizationPhaseJumpName = "PolarizationPhase";
const char *distanceQuasiGibbsProposalName = "DistanceQuasiGibbs";
const char *orbitalPhaseQuasiGibbsProposalName = "OrbitalPhaseQuasiGibbs";
const char *KDNeighborhoodProposalName = "KDNeighborhood";

/* Mode hopping fraction for the differential evoultion proposals. */
static const REAL8 modeHoppingFrac = 0.1;

static int
same_detector_location(LALInferenceIFOData *d1, LALInferenceIFOData *d2) {
  UINT4 i;

  for (i = 0; i < 3; i++) {
    if (d1->detector->location[i] != d2->detector->location[i]) return 0;
  }

  return 1;
}

static UINT4 
numDetectorsUniquePositions(LALInferenceRunState *runState) {
  UINT4 nIFO = 0;
  UINT4 nCollision = 0;
  LALInferenceIFOData *currentIFO = NULL;

  for (currentIFO = runState->data; currentIFO; currentIFO = currentIFO->next) {
    LALInferenceIFOData *subsequentIFO = NULL;
    nIFO++;
    for (subsequentIFO = currentIFO->next; subsequentIFO; subsequentIFO = subsequentIFO->next) {
      if (same_detector_location(subsequentIFO, currentIFO)) {
        nCollision++;
        break;
      }
    }
  }

  return nIFO - nCollision;
}


static void
LALInferenceSetLogProposalRatio(LALInferenceRunState *runState, REAL8 logP) {
  if (LALInferenceCheckVariable(runState->proposalArgs, "logProposalRatio")) {
    LALInferenceSetVariable(runState->proposalArgs, "logProposalRatio", &logP);
  } else {
    LALInferenceAddVariable(runState->proposalArgs, "logProposalRatio", &logP, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_LINEAR);
  }
}

void
LALInferenceAddProposalToCycle(LALInferenceRunState *runState, const char *propName, LALInferenceProposalFunction *prop, UINT4 weight) {
  const char *fname = "LALInferenceAddProposalToCycle";

  UINT4 length = 0;
  LALInferenceProposalFunction **cycle = NULL;
  LALInferenceVariables *propArgs = runState->proposalArgs;
  LALInferenceVariables *propStats = runState->proposalStats;

  /* Quit without doing anything if weight = 0. */
  if (weight == 0) {
    return;
  }

  if (LALInferenceCheckVariable(propArgs, cycleArrayName) && LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    /* Have all the data in proposal args. */
    UINT4 i;

    length = *((UINT4 *)LALInferenceGetVariable(propArgs, cycleArrayLengthName));
    cycle = *((LALInferenceProposalFunction ***)LALInferenceGetVariable(propArgs, cycleArrayName));

    cycle = XLALRealloc(cycle, (length+weight)*sizeof(LALInferenceProposalFunction *));
    if (cycle == NULL) {
      XLALError(fname, __FILE__, __LINE__, XLAL_ENOMEM);
      exit(1);
    }

    for (i = length; i < length + weight; i++) {
      cycle[i] = prop;
    }

    length += weight;

    LALInferenceSetVariable(propArgs, cycleArrayLengthName, &length);
    LALInferenceSetVariable(propArgs, cycleArrayName, (void *)&cycle);
  } else {
    /* There are no data in proposal args.  Set some. */
    UINT4 i;
    
    length = weight;

    cycle = XLALMalloc(length*sizeof(LALInferenceProposalFunction *));
    if (cycle == NULL) {
      XLALError(fname, __FILE__, __LINE__, XLAL_ENOMEM);
      exit(1);
    }

    for (i = 0; i < length; i++) {
      cycle[i] = prop;
    }

    LALInferenceAddVariable(propArgs, cycleArrayLengthName, &length,
LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_LINEAR);
    LALInferenceAddVariable(propArgs, cycleArrayName, (void *)&cycle,
LALINFERENCE_void_ptr_t, LALINFERENCE_PARAM_LINEAR);
  }

  /* If propStats is not NULL, add counters for proposal function if they aren't already there */
  if(propStats){
    if(!LALInferenceCheckVariable(propStats, propName)){
      LALInferenceProposalStatistics propStat = {
        .weight = weight,
        .proposed = 0,
        .accepted = 0};
      LALInferenceAddVariable(propStats, propName, (void *)&propStat, LALINFERENCE_void_ptr_t, LALINFERENCE_PARAM_LINEAR);
    }
  }
}

void
LALInferenceRandomizeProposalCycle(LALInferenceRunState *runState) {
  const char *fname = "LALInferenceRandomizeProposalCycle";
  UINT4 length = 0;
  LALInferenceProposalFunction **cycle = NULL;
  LALInferenceVariables *propArgs = runState->proposalArgs;

  UINT4 i;

  if (!LALInferenceCheckVariable(propArgs, cycleArrayName) || !LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    XLALError(fname, __FILE__, __LINE__, XLAL_FAILURE);
    exit(1);
  }

  cycle = *((LALInferenceProposalFunction ***)LALInferenceGetVariable(propArgs, cycleArrayName));
  length = *((UINT4 *)LALInferenceGetVariable(propArgs, cycleArrayLengthName));

  for (i = length - 1; i > 0; i--) {
    /* Fill in array from right to left, chosen randomly from remaining proposals. */
    UINT4 j;
    LALInferenceProposalFunction *prop;

    j = gsl_rng_uniform_int(runState->GSLrandom, i+1);
    prop = cycle[j];
    cycle[j] = cycle[i];
    cycle[i] = prop;
  }
}

/* Convert NS to MCMC variables (call before calling MCMC proposal from NS) */
void NSFillMCMCVariables(LALInferenceVariables *proposedParams, LALInferenceVariables *priorArgs)
{
  REAL8 distance=0.0,mc=0.0,dmin,dmax,mmin,mmax;
  if(LALInferenceCheckVariable(proposedParams,"logdistance"))
  {
    distance=exp(*(REAL8*)LALInferenceGetVariable(proposedParams,"logdistance"));
    LALInferenceAddVariable(proposedParams,"distance",&distance,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }
  if(!LALInferenceCheckMinMaxPrior(priorArgs,"distance") &&
     LALInferenceCheckMinMaxPrior(priorArgs,"logdistance"))
  {
    LALInferenceGetMinMaxPrior(priorArgs,"logdistance",&dmin,&dmax);
    dmin=exp(dmin); dmax=exp(dmax);
    LALInferenceAddMinMaxPrior(priorArgs,"distance",&dmin,&dmax,LALINFERENCE_REAL8_t);
  }
  if(LALInferenceCheckVariable(proposedParams,"logmc")){
    mc=exp(*(REAL8 *)LALInferenceGetVariable(proposedParams,"logmc"));
    LALInferenceAddVariable(proposedParams,"chirpmass",&mc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }
  if(!LALInferenceCheckMinMaxPrior(priorArgs,"chirpmass") && 
     LALInferenceCheckMinMaxPrior(priorArgs,"logmc"))
  {
    LALInferenceGetMinMaxPrior(priorArgs,"logmc",&mmin,&mmax);
    mmin=exp(mmin); mmax=exp(mmax);
    LALInferenceAddMinMaxPrior(priorArgs,"chirpmass",&mmin,&mmax,LALINFERENCE_REAL8_t);
  }
  return;
}

void NSWrapMCMCLALProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams)
{
  /* PTMCMC likes to read currentParams directly, whereas NS expects proposedParams
   to be modified by the proposal. Back up currentParams and then restore it after
   calling the MCMC proposal function. */
  REAL8 oldlogdist=-1.0,oldlogmc=-1.0;
  REAL8 newdist,newmc;
  LALInferenceVariables *currentParamsBackup=runState->currentParams;
  
  /* PTMCMC expects some variables that NS doesn't use by default, so create them */
  
  if(LALInferenceCheckVariable(proposedParams,"logdistance"))
    oldlogdist=*(REAL8 *)LALInferenceGetVariable(proposedParams,"logdistance");
  if(LALInferenceCheckVariable(proposedParams,"logmc"))
    oldlogmc=*(REAL8*)LALInferenceGetVariable(proposedParams,"logmc");
  
  NSFillMCMCVariables(proposedParams,runState->priorArgs);

  runState->currentParams=proposedParams; 
  LALInferenceDefaultProposal(runState,proposedParams);
  /* Restore currentParams */
  runState->currentParams=currentParamsBackup;
  
  /* If the remapped variables are not updated do it here */
  if(oldlogdist!=-1.0)
    if(oldlogdist==*(REAL8*)LALInferenceGetVariable(proposedParams,"logdistance"))
      {
	newdist=*(REAL8*)LALInferenceGetVariable(proposedParams,"distance");
	newdist=log(newdist);
	LALInferenceSetVariable(proposedParams,"logdistance",&newdist);
      }
  if(oldlogmc!=-1.0)
    if(oldlogmc==*(REAL8*)LALInferenceGetVariable(proposedParams,"logmc"))
    {
      newmc=*(REAL8*)LALInferenceGetVariable(proposedParams,"chirpmass");
      newmc=log(newmc);
      LALInferenceSetVariable(proposedParams,"logmc",&newmc);
    }
  
}

void 
LALInferenceCyclicProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *fname = "LALInferenceCyclicProposal";
  UINT4 length = 0;
  UINT4 i = 0;
  LALInferenceProposalFunction **cycle = NULL;
  LALInferenceVariables *propArgs = runState->proposalArgs;

  /* Must have cycle array and cycle array length in propArgs. */
  if (!LALInferenceCheckVariable(propArgs, cycleArrayName) || !LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    XLALError(fname, __FILE__, __LINE__, XLAL_FAILURE);
    exit(1);
  }

  length = *((UINT4 *)LALInferenceGetVariable(propArgs, cycleArrayLengthName));
  cycle = *((LALInferenceProposalFunction ***)LALInferenceGetVariable(propArgs, cycleArrayName));

  /* If there is not a proposal counter, put one into the variables, initialized to zero. */
  if (!LALInferenceCheckVariable(propArgs, cycleArrayCounterName)) {
    i = 0;
    LALInferenceAddVariable(propArgs, cycleArrayCounterName, &i, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_CIRCULAR);
  }

  i = *((UINT4 *)LALInferenceGetVariable(propArgs, cycleArrayCounterName));

  if (i >= length) {
    XLALError(fname, __FILE__, __LINE__, XLAL_FAILURE);
    exit(1);
  }

  /* Call proposal. */
  (cycle[i])(runState, proposedParams);

  /* Increment counter for the next time around. */
  i = (i+1) % length;
  LALInferenceSetVariable(propArgs, cycleArrayCounterName, &i);
}

void
LALInferenceDeleteProposalCycle(LALInferenceRunState *runState) {
  LALInferenceVariables *propArgs = runState->proposalArgs;
  
  if (LALInferenceCheckVariable(propArgs, cycleArrayName)) {
    LALInferenceProposalFunction **cycle = *((LALInferenceProposalFunction ***)LALInferenceGetVariable(propArgs, cycleArrayName));
    XLALFree(cycle);
    LALInferenceRemoveVariable(propArgs, cycleArrayName);
  }

  if (LALInferenceCheckVariable(propArgs, cycleArrayCounterName)) {
    LALInferenceRemoveVariable(propArgs, cycleArrayCounterName);
  }

  if (LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    LALInferenceRemoveVariable(propArgs, cycleArrayLengthName);
  }
}

static void
SetupDefaultProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const UINT4 BIGWEIGHT = 20;
  const UINT4 SMALLWEIGHT = 5;
  const UINT4 TINYWEIGHT = 1;
  const char defaultPropName[]="none";
  ProcessParamsTable *ppt;
  if(!LALInferenceCheckVariable(runState->proposalArgs,LALInferenceCurrentProposalName))
      LALInferenceAddVariable(runState->proposalArgs,LALInferenceCurrentProposalName, (void*)&defaultPropName, LALINFERENCE_string_t, LALINFERENCE_PARAM_OUTPUT);

  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  /* The default, single-parameter updates. */
  if(!LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-singleadapt"))
      LALInferenceAddProposalToCycle(runState, singleAdaptProposalName, &LALInferenceSingleAdaptProposal, BIGWEIGHT);

  if(!LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-psiphi"))
      LALInferenceAddProposalToCycle(runState, polarizationPhaseJumpName, &LALInferencePolarizationPhaseJump, TINYWEIGHT);
  if(!LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-skywander"))
      LALInferenceAddProposalToCycle(runState, skyLocWanderJumpName, &LALInferenceSkyLocWanderJump, SMALLWEIGHT);

  UINT4 nDet = numDetectorsUniquePositions(runState);
  if (nDet == 3 && !LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-skyreflect")) {
    LALInferenceAddProposalToCycle(runState, skyReflectDetPlaneName, &LALInferenceSkyReflectDetPlane, TINYWEIGHT);
  }
  if(!LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-drawprior"))
    LALInferenceAddProposalToCycle(runState, drawApproxPriorName, &LALInferenceDrawApproxPrior, TINYWEIGHT);

  /* Now add various special proposals that are conditional on
     command-line arguments or variables in the params. */
  if (LALInferenceCheckVariable(proposedParams, "theta_spin1")&&!LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-rotate-spins")) {
    LALInferenceAddProposalToCycle(runState, rotateSpinsName, &LALInferenceRotateSpins, SMALLWEIGHT);
  }

  ppt=LALInferenceGetProcParamVal(runState->commandLine, "--covariancematrix");
  if(!ppt){
 	ppt=LALInferenceGetProcParamVal(runState->commandLine, "--covarianceMatrix");
 	if(ppt) XLALPrintWarning("WARNING: Deprecated --covarianceMatrix option will be removed, please change to --covariancematrix");
  }
  if (ppt) {
    LALInferenceAddProposalToCycle(runState, covarianceEigenvectorJumpName, &LALInferenceCovarianceEigenvectorJump, BIGWEIGHT);
  }

  if (!LALInferenceGetProcParamVal(runState->commandLine, "--noDifferentialEvolution")
      && !LALInferenceGetProcParamVal(runState->commandLine, "--nodifferentialevolution") && !LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-differentialevolution")) {
    LALInferenceAddProposalToCycle(runState, differentialEvolutionFullName, &LALInferenceDifferentialEvolutionFull, BIGWEIGHT);
    LALInferenceAddProposalToCycle(runState, differentialEvolutionMassesName, &LALInferenceDifferentialEvolutionMasses, SMALLWEIGHT);
    LALInferenceAddProposalToCycle(runState, differentialEvolutionAmpName, &LALInferenceDifferentialEvolutionAmp, SMALLWEIGHT);
    LALInferenceAddProposalToCycle(runState, differentialEvolutionSkyName, &LALInferenceDifferentialEvolutionSky, SMALLWEIGHT);
    if (LALInferenceCheckVariable(proposedParams, "theta_spin1")) {
      LALInferenceAddProposalToCycle(runState, differentialEvolutionSpinsName, &LALInferenceDifferentialEvolutionSpins, SMALLWEIGHT);
    }
  }

  if (LALInferenceGetProcParamVal(runState->commandLine, "--kDTree") || LALInferenceGetProcParamVal(runState->commandLine,"--kdtree")) {
    LALInferenceAddProposalToCycle(runState, KDNeighborhoodProposalName, &LALInferenceKDNeighborhoodProposal, SMALLWEIGHT);
  }

  if(!LALInferenceGetProcParamVal(runState->commandLine,"--nogibbsproposal") && !LALInferenceGetProcParamVal(runState->commandLine,"--proposal-no-gibbs")){
    LALInferenceAddProposalToCycle(runState, distanceQuasiGibbsProposalName, &LALInferenceDistanceQuasiGibbsProposal, SMALLWEIGHT);
    LALInferenceAddProposalToCycle(runState, orbitalPhaseQuasiGibbsProposalName, &LALInferenceOrbitalPhaseQuasiGibbsProposal, SMALLWEIGHT);
  }
  LALInferenceRandomizeProposalCycle(runState);
}

static void
SetupRapidSkyLocProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  LALInferenceCopyVariables(runState->currentParams, proposedParams);
  LALInferenceAddProposalToCycle(runState, singleAdaptProposalName, &LALInferenceSingleAdaptProposal, 100);
  //LALInferenceAddProposalToCycle(runState, skyLocWanderJumpName, &LALInferenceSkyLocWanderJump, 1);
  //LALInferenceAddProposalToCycle(runState, inclinationDistanceName, &LALInferenceInclinationDistance, 1);
  LALInferenceAddProposalToCycle(runState, polarizationPhaseJumpName, &LALInferencePolarizationPhaseJump, 1);

  /*
  UINT4 nDet = numDetectorsUniquePositions(runState);
  if (nDet == 3) {
    LALInferenceAddProposalToCycle(runState, skyReflectDetPlaneName, &LALInferenceSkyReflectDetPlane, 1);
  }
  */

  if (!LALInferenceGetProcParamVal(runState->commandLine, "--noDifferentialEvolution")) {
    LALInferenceAddProposalToCycle(runState, differentialEvolutionFullName, &LALInferenceDifferentialEvolutionFull, 10);
    LALInferenceAddProposalToCycle(runState, differentialEvolutionAmpName, &LALInferenceDifferentialEvolutionAmp, 1);
    LALInferenceAddProposalToCycle(runState, differentialEvolutionSkyName, &LALInferenceDifferentialEvolutionSky, 5);
  }

  if(!LALInferenceGetProcParamVal(runState->commandLine,"--nogibbsproposal")){
    LALInferenceAddProposalToCycle(runState, distanceQuasiGibbsProposalName, &LALInferenceDistanceQuasiGibbsProposal, 1);
    LALInferenceAddProposalToCycle(runState, orbitalPhaseQuasiGibbsProposalName, &LALInferenceOrbitalPhaseQuasiGibbsProposal, 1);
  }

  LALInferenceRandomizeProposalCycle(runState);
}

static void
SetupPTTempTestProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  LALInferenceCopyVariables(runState->currentParams, proposedParams);
  LALInferenceAddProposalToCycle(runState, drawApproxPriorName, &LALInferenceDrawApproxPrior, 1);
  LALInferenceRandomizeProposalCycle(runState);
}

void LALInferencePTTempTestProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  LALInferenceVariables *propArgs = runState->proposalArgs;
  if (!LALInferenceCheckVariable(propArgs, cycleArrayName) || !LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    /* In case there is a partial cycle set up already, delete it. */
    LALInferenceDeleteProposalCycle(runState);
    SetupPTTempTestProposal(runState, proposedParams);
  }

  LALInferenceCyclicProposal(runState, proposedParams);
}

void LALInferenceRapidSkyLocProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  LALInferenceVariables *propArgs = runState->proposalArgs;

  if (!LALInferenceCheckVariable(propArgs, cycleArrayName) || !LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    /* In case there is a partial cycle set up already, delete it. */
    LALInferenceDeleteProposalCycle(runState);
    SetupRapidSkyLocProposal(runState, proposedParams);
  }

  LALInferenceCyclicProposal(runState, proposedParams);
}


void LALInferenceDefaultProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams)
{
  LALInferenceVariables *propArgs = runState->proposalArgs;

  /* If the cyclic proposal is not yet set up, set it up.  Note that
     this means that you can set up your own proposal cycle and it
     will be used in this function. */
  if (!LALInferenceCheckVariable(propArgs, cycleArrayName) || !LALInferenceCheckVariable(propArgs, cycleArrayLengthName)) {
    /* In case there is a partial cycle set up already, delete it. */
    LALInferenceDeleteProposalCycle(runState);
    SetupDefaultProposal(runState, proposedParams);
  }

  LALInferenceCyclicProposal(runState, proposedParams);
}

void LALInferenceSingleAdaptProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = singleAdaptProposalName;
  LALInferenceVariables *args = runState->proposalArgs;
  LALInferenceSetVariable(args, LALInferenceCurrentProposalName, &propName);
  ProcessParamsTable *ppt = LALInferenceGetProcParamVal(runState->commandLine, "--adapt");
  
  if (!LALInferenceCheckVariable(args, LALInferenceSigmaJumpName) || !ppt) {
    /* We are not adaptive, or for some reason don't have a sigma
       vector---fall back on old proposal. */
    LALInferenceSingleProposal(runState, proposedParams);
  } else {
    gsl_rng *rng = runState->GSLrandom;
    LALInferenceVariableItem *param = NULL, *dummyParam = NULL;
    REAL8 T = *(REAL8 *)LALInferenceGetVariable(args, "temperature");
    REAL8 sqrtT = sqrt(T);
    UINT4 dim;
    UINT4 i;
    UINT4 varNr;
    REAL8Vector *sigmas = *(REAL8Vector **) LALInferenceGetVariable(args, LALInferenceSigmaJumpName);

    LALInferenceCopyVariables(runState->currentParams, proposedParams);

    dim = proposedParams->dimension;

    do {
      varNr = 1+gsl_rng_uniform_int(rng, dim);
      param = LALInferenceGetItemNr(proposedParams, varNr);
    } while (param->vary == LALINFERENCE_PARAM_FIXED || param->vary == LALINFERENCE_PARAM_OUTPUT);

    for (dummyParam = proposedParams->head, i = 0; dummyParam != NULL; dummyParam = dummyParam->next) {
      if (!strcmp(dummyParam->name, param->name)) {
        /* Found it; i = index into sigma vector. */
        break;
      } else if (dummyParam->vary == LALINFERENCE_PARAM_FIXED || dummyParam->vary == LALINFERENCE_PARAM_OUTPUT) {
        /* Don't increment i, since we're not dealing with a "real" parameter. */
        continue;
      } else {
        i++;
        continue;
      }
    }

    if (param->type != LALINFERENCE_REAL8_t) {
      fprintf(stderr, "Attempting to set non-REAL8 parameter with numerical sigma (in %s, %d)\n",
              __FILE__, __LINE__);
      exit(1);
    } 

    if (i >= sigmas->length) {
      fprintf(stderr, "Attempting to draw single-parameter jump %d past the end of sigma array %d.\n(Maybe you used a non-spinning correlation matrix for a spinning run?)\nError in %s, line %d.\n",
              i,sigmas->length,__FILE__, __LINE__);
      exit(1);
    }

    *((REAL8 *)param->value) += gsl_ran_ugaussian(rng)*sigmas->data[i]*sqrtT;

    LALInferenceCyclicReflectiveBound(proposedParams, runState->priorArgs);

    /* Set the log of the proposal ratio to zero, since this is a
       symmetric proposal. */
    LALInferenceSetLogProposalRatio(runState, 0.0);

    INT4 as = 1;
    LALInferenceSetVariable(args, "adaptableStep", &as);

    LALInferenceSetVariable(args, "proposedVariableNumber", &varNr);
    
    LALInferenceSetVariable(args, "proposedArrayNumber", &i);
  }
}

void LALInferenceSingleProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams)
{
  const char *propName = singleProposalName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  gsl_rng * GSLrandom=runState->GSLrandom;
  LALInferenceVariableItem *param=NULL, *dummyParam=NULL;
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  REAL8 T = *(REAL8 *)LALInferenceGetVariable(runState->proposalArgs, "temperature");
	
  REAL8 sigma = 0.1*sqrt(T); /* Adapt step to temperature. */
  REAL8 big_sigma = 1.0;
  UINT4 dim;
  UINT4 i;
  UINT4 varNr;
  
  if(gsl_ran_ugaussian(GSLrandom) < 1.0e-3) big_sigma = 1.0e1;    //Every 1e3 iterations, take a 10x larger jump in a parameter
  if(gsl_ran_ugaussian(GSLrandom) < 1.0e-4) big_sigma = 1.0e2;    //Every 1e4 iterations, take a 100x larger jump in a parameter

  dim = proposedParams->dimension;
  
  do {
    varNr = 1+gsl_rng_uniform_int(GSLrandom, dim);
    param = LALInferenceGetItemNr(proposedParams, varNr);
  } while (param->vary == LALINFERENCE_PARAM_FIXED || param->vary == LALINFERENCE_PARAM_OUTPUT);
  
  for (dummyParam = proposedParams->head, i = 0; dummyParam != NULL; dummyParam = dummyParam->next) {
    if (!strcmp(dummyParam->name, param->name)) {
      /* Found it; i = index into sigma vector. */
      break;
    } else if (dummyParam->vary == LALINFERENCE_PARAM_FIXED || dummyParam->vary == LALINFERENCE_PARAM_OUTPUT) {
      /* Don't increment i, since we're not dealing with a "real" parameter. */
      continue;
    } else {
      i++;
      continue;
    }
  }	//printf("%s\n",param->name);
		
  if (LALInferenceGetProcParamVal(runState->commandLine, "--zeroLogLike") || LALInferenceGetProcParamVal(runState->commandLine,"--zerologlike")) {
    if (!strcmp(param->name, "massratio")) {
      sigma = 0.02;
    } else if (!strcmp(param->name, "asym_massratio")) {
      sigma = 0.08;
    } else if (!strcmp(param->name, "chirpmass")) {
      sigma = 1.0;
    } else if (!strcmp(param->name, "time")) {
      sigma = 0.02;
    } else if (!strcmp(param->name, "phase")) {
      sigma = 0.6;
    } else if (!strcmp(param->name, "distance")) {
      sigma = 10.0;
    } else if (!strcmp(param->name, "declination")) {
      sigma = 0.3;
    } else if (!strcmp(param->name, "rightascension")) {
      sigma = 0.6;
    } else if (!strcmp(param->name, "polarisation")) {
      sigma = 0.6;
    } else if (!strcmp(param->name, "inclination")) {
      sigma = 0.3;
    } else if (!strcmp(param->name, "a_spin1")) {
      sigma = 0.1;
    } else if (!strcmp(param->name, "theta_spin1")) {
      sigma = 0.3;
    } else if (!strcmp(param->name, "phi_spin1")) {
      sigma = 0.6;
    } else if (!strcmp(param->name, "a_spin2")) {
      sigma = 0.1;
    } else if (!strcmp(param->name, "theta_spin2")) {
      sigma = 0.3;
    } else if (!strcmp(param->name, "phi_spin2")) {
      sigma = 0.6;
    } else {
      fprintf(stderr, "Could not find parameter %s!", param->name);
      exit(1);
    }
    *(REAL8 *)param->value += gsl_ran_ugaussian(GSLrandom)*sigma;
  } else {
    if (!strcmp(param->name,"massratio") || !strcmp(param->name,"asym_massratio") || !strcmp(param->name,"time") || !strcmp(param->name,"a_spin2") || !strcmp(param->name,"a_spin1")){
      *(REAL8 *)param->value += gsl_ran_ugaussian(GSLrandom)*big_sigma*sigma*0.001;
    } else if (!strcmp(param->name,"polarisation") || !strcmp(param->name,"phase") || !strcmp(param->name,"inclination")){
      *(REAL8 *)param->value += gsl_ran_ugaussian(GSLrandom)*big_sigma*sigma*0.1;
    } else {
      *(REAL8 *)param->value += gsl_ran_ugaussian(GSLrandom)*big_sigma*sigma*0.01;
    }
  }
  LALInferenceCyclicReflectiveBound(proposedParams, runState->priorArgs);
  
  /* Symmetric Proposal. */
  LALInferenceSetLogProposalRatio(runState, 0.0);

  INT4 as = 1;
  LALInferenceSetVariable(runState->proposalArgs, "adaptableStep", &as);
  
  LALInferenceSetVariable(runState->proposalArgs, "proposedVariableNumber", &varNr);
  
  LALInferenceSetVariable(runState->proposalArgs, "proposedArrayNumber", &i);
  
}

void LALInferenceOrbitalPhaseJump(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = orbitalPhaseJumpName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  REAL8 phi;

  LALInferenceCopyVariables(runState->currentParams, proposedParams);
  
  phi = *((REAL8 *) LALInferenceGetVariable(proposedParams, "phase"));

  phi = fmod(phi+M_PI, 2.0*M_PI);

  LALInferenceSetVariable(proposedParams, "phase", &phi);

  LALInferenceSetLogProposalRatio(runState, 0.0);

  /* Probably not needed, but play it safe. */
  LALInferenceCyclicReflectiveBound(proposedParams, runState->priorArgs);
}

/* The idea for this jump proposal is to explore the cos(I)-d
   degeneracy that we see in our MCMC's.  If we had exactly one
   detector in the network, this would be a perfect degeneracy for
   non-spinning signals, since the inclination and distance both enter
   only in the amplitude of the signal.  With multiple detectors, the
   degeneracy is broken because each detector responds differently to
   the + and x polarizations (i.e. has different f+, fx), and
   therefore differently to changes in the inclination of the system.  

   This jump proposal selects one of the detectors at random, and then
   jumps to a random location on the cos(I)-d curve that keeps that
   detector's received GW amplitude constant.  We choose to keep the
   amplitude constant in only one of the detectors instead of, for
   example, keeping the SNR-weighted amplitude constant, or the
   uniformly-weighted amplitude, or the sensitivity-weighted
   amplitude, or ... because it makes for the simplest jump proposal.
   */
void LALInferenceInclinationDistance(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = inclinationDistanceName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  UINT4 nIFO = 0;
  LALInferenceIFOData *data = runState->data;
  while (data != NULL) {
    nIFO++;
    data = data->next;
  }

  /* Grab one of the detectors. */
  UINT4 iIFO = gsl_rng_uniform_int(runState->GSLrandom, nIFO);
  data = runState->data;
  while (iIFO > 0) {
    iIFO--;
    data = data->next;
  }

  REAL8 ra, dec, psi, t, gmst;
  LIGOTimeGPS tGPS;
  ra = *(REAL8 *)LALInferenceGetVariable(proposedParams, "rightascension");
  dec = *(REAL8 *)LALInferenceGetVariable(proposedParams, "declination");
  psi = *(REAL8 *)LALInferenceGetVariable(proposedParams, "polarisation");
  t = *(REAL8 *)LALInferenceGetVariable(proposedParams, "time");
  
  XLALGPSSetREAL8(&tGPS, t);
  gmst = XLALGreenwichMeanSiderealTime(&tGPS);

  REAL8 fPlus, fCross;
  XLALComputeDetAMResponse(&fPlus, &fCross, data->detector->response, ra, dec, psi, gmst);

  /* Choose new inclination uniformly in cos(i). */
  REAL8 inc = *(REAL8 *)LALInferenceGetVariable(proposedParams, "inclination");
  REAL8 cosI = cos(inc);
  REAL8 cosINew = 2.0*gsl_rng_uniform(runState->GSLrandom) - 1.0;

  REAL8 d = *(REAL8 *)LALInferenceGetVariable(proposedParams, "distance");

  /* This is the constant that describes the curve. */
  REAL8 C = (fPlus*(1 + cosI*cosI) + 2.0*fCross*cosI)/d;

  REAL8 dNew = (fPlus*(1 + cosINew*cosINew) + 2.0*fCross*cosI) / C;

  REAL8 incNew = acos(cosINew);

  /* This is the determinant of the Jacobian d(cos(i),C)/d(i, d),
     which is the probability density in (i,d) space of a proposal
     that is uniform in cos(i), C.*/
  REAL8 dcosiCdid = C*C*sqrt(1-cosINew*cosINew) / (fPlus + 2.0*fCross*cosINew + fPlus*cosINew*cosINew);

  LALInferenceSetVariable(proposedParams, "inclination", &incNew);
  LALInferenceSetVariable(proposedParams, "distance", &dNew);

  LALInferenceSetLogProposalRatio(runState, log(dcosiCdid));
}

void LALInferenceCovarianceEigenvectorJump(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = covarianceEigenvectorJumpName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceVariables *proposalArgs = runState->proposalArgs;
  gsl_matrix *eigenvectors = *((gsl_matrix **)LALInferenceGetVariable(proposalArgs, "covarianceEigenvectors"));
  REAL8Vector *eigenvalues = *((REAL8Vector **)LALInferenceGetVariable(proposalArgs, "covarianceEigenvalues"));
  REAL8 temp = *((REAL8 *)LALInferenceGetVariable(proposalArgs, "temperature"));
  UINT4 N = eigenvalues->length;
  gsl_rng *rng = runState->GSLrandom;
  UINT4 i = gsl_rng_uniform_int(rng, N);
  REAL8 jumpSize = sqrt(temp*eigenvalues->data[i])*gsl_ran_ugaussian(rng);
  UINT4 j;
  LALInferenceVariableItem *proposeIterator;

  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  j = 0;
  proposeIterator = proposedParams->head;
  if (proposeIterator == NULL) {
    fprintf(stderr, "Bad proposed params in %s, line %d\n",
            __FILE__, __LINE__);
    exit(1);
  }
  do {
    if (proposeIterator->vary != LALINFERENCE_PARAM_FIXED && proposeIterator->vary != LALINFERENCE_PARAM_OUTPUT) {
      REAL8 tmp = *((REAL8 *)proposeIterator->value);
      REAL8 inc = jumpSize*gsl_matrix_get(eigenvectors, j, i);

      tmp += inc;

      memcpy(proposeIterator->value, &tmp, sizeof(REAL8));

      j++;
    }
  } while ((proposeIterator = proposeIterator->next) != NULL && j < N);

  LALInferenceSetLogProposalRatio(runState, 0.0);
}

void LALInferenceSkyLocWanderJump(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = skyLocWanderJumpName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  gsl_rng *rng = runState->GSLrandom;
  LALInferenceVariables *proposalArgs = runState->proposalArgs;
  REAL8 temp = *((REAL8 *)LALInferenceGetVariable(proposalArgs, "temperature"));
  REAL8 one_deg = 1.0 / (2.0*M_PI);
  REAL8 sigma = sqrt(temp)*one_deg;
  REAL8 XU = gsl_ran_ugaussian(rng);
  REAL8 YU = gsl_ran_ugaussian(rng);
  REAL8 jumpX = sigma*XU;
  REAL8 jumpY = sigma*YU;
  REAL8 RA, DEC;
  REAL8 newRA, newDEC;

  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  RA = *((REAL8 *)LALInferenceGetVariable(proposedParams, "rightascension"));
  DEC = *((REAL8 *)LALInferenceGetVariable(proposedParams, "declination"));

  newRA = RA + jumpX;
  newDEC = DEC + jumpY;

  LALInferenceSetVariable(proposedParams, "rightascension", &newRA);
  LALInferenceSetVariable(proposedParams, "declination", &newDEC);

  LALInferenceSetLogProposalRatio(runState, 0.0);
}

void LALInferenceDifferentialEvolutionFull(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = differentialEvolutionFullName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceDifferentialEvolutionNames(runState, proposedParams, NULL);
}

void LALInferenceDifferentialEvolutionNames(LALInferenceRunState *runState, 
                                            LALInferenceVariables *proposedParams,
                                            const char **names) {
  if (names == NULL) {
    size_t i;
    size_t N = LALInferenceGetVariableDimension(runState->currentParams) + 1; /* More names than we need. */
    names = alloca(N*sizeof(char *)); /* Hope we have alloca---saves
                                         having to deallocate after
                                         proposal. */

    LALInferenceVariableItem *item = runState->currentParams->head;
    i = 0;
    while (item != NULL) {
      if (item->vary != LALINFERENCE_PARAM_FIXED && item->vary != LALINFERENCE_PARAM_OUTPUT) {
        names[i] = item->name;
        i++;
      }

      item = item->next;
    }
    names[i]=NULL; /* Terminate */
  }

  size_t Ndim = 0;
  const char *name = names[0];
  while (name != NULL) {
    Ndim++;
    name = names[Ndim];
  }

  LALInferenceVariables **dePts = runState->differentialPoints;
  size_t nPts = runState->differentialPointsLength;

  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  if (dePts == NULL || nPts <= 1) {
    return; /* Quit now, since we don't have any points to use. */
  }

  size_t i,j;

  i = gsl_rng_uniform_int(runState->GSLrandom, nPts);
  do {
    j = gsl_rng_uniform_int(runState->GSLrandom, nPts);
  } while (j == i);

  LALInferenceVariables *ptI = dePts[i];
  LALInferenceVariables *ptJ = dePts[j];
  REAL8 scale;

  /* Some small fraction of the time, we do a "mode hopping" jump,
     where we jump exactly along the difference vector. */
  if (gsl_rng_uniform(runState->GSLrandom) < modeHoppingFrac) {
    scale = 1.0;
  } else {  
    scale = 2.38 * gsl_ran_ugaussian(runState->GSLrandom) / sqrt(2.0*Ndim);
  }

  for (i = 0; names[i] != NULL; i++) {
    if (!LALInferenceCheckVariable(proposedParams, names[i]) || !LALInferenceCheckVariable(ptJ, names[i]) || !LALInferenceCheckVariable(ptI, names[i])) {
      /* Ignore variable if it's not in each of the params. */
    } else {
      REAL8 x = *((REAL8 *)LALInferenceGetVariable(proposedParams, names[i]));
      x += scale * (*((REAL8 *) LALInferenceGetVariable(ptJ, names[i])));
      x -= scale * (*((REAL8 *) LALInferenceGetVariable(ptI, names[i])));
      
      LALInferenceSetVariable(proposedParams, names[i], &x);
    }
  }
  
  LALInferenceSetLogProposalRatio(runState, 0.0); /* Symmetric proposal. */
}
  
void LALInferenceDifferentialEvolutionMasses(LALInferenceRunState *runState, LALInferenceVariables *pp) {
  const char *propName = differentialEvolutionMassesName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  const char *names[] = {"chirpmass", "asym_massratio", NULL};
  if (LALInferenceCheckVariable(pp, "massratio")) {
    names[1] = "massratio";
  }
  LALInferenceDifferentialEvolutionNames(runState, pp, names);
}

void LALInferenceDifferentialEvolutionAmp(LALInferenceRunState *runState, LALInferenceVariables *pp) {
  const char *propName = differentialEvolutionAmpName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  const char *names[] = {"rightascension", "declination", "polarisation", "inclination", "distance", "time", NULL};
  LALInferenceDifferentialEvolutionNames(runState, pp, names);
}

void LALInferenceDifferentialEvolutionSpins(LALInferenceRunState *runState, LALInferenceVariables *pp) {
  const char *propName = differentialEvolutionSpinsName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  const char *names[] = {"a_spin1", "a_spin2", "phi_spin1", "phi_spin2", "theta_spin1", "theta_spin2", NULL};
  LALInferenceDifferentialEvolutionNames(runState, pp, names);
}

void LALInferenceDifferentialEvolutionSky(LALInferenceRunState *runState, LALInferenceVariables *pp) {
  const char *propName = differentialEvolutionSkyName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  const char *names[] = {"rightascension", "declination", "time", NULL};
  LALInferenceDifferentialEvolutionNames(runState, pp, names);
}

static REAL8 
draw_distance(LALInferenceRunState *runState) {
  REAL8 dmin, dmax;

  LALInferenceGetMinMaxPrior(runState->priorArgs, "distance", &dmin, &dmax);

  REAL8 x = gsl_rng_uniform(runState->GSLrandom);

  return pow(x*(dmax*dmax*dmax - dmin*dmin*dmin) + dmin*dmin*dmin, 1.0/3.0);
}

static REAL8 
draw_colatitude(LALInferenceRunState *runState, const char *name) {
  REAL8 min, max;

  LALInferenceGetMinMaxPrior(runState->priorArgs, name, &min, &max);

  REAL8 x = gsl_rng_uniform(runState->GSLrandom);

  return acos(cos(min) - x*(cos(min) - cos(max)));
}

static REAL8 
draw_dec(LALInferenceRunState *runState) {
  REAL8 min, max;
  
  LALInferenceGetMinMaxPrior(runState->priorArgs, "declination", &min, &max);

  REAL8 x = gsl_rng_uniform(runState->GSLrandom);
  
  return asin(x*(sin(max) - sin(min)) + sin(min));
}

static REAL8 
draw_flat(LALInferenceRunState *runState, const char *name) {
  REAL8 min, max;

  LALInferenceGetMinMaxPrior(runState->priorArgs, name, &min, &max);

  REAL8 x = gsl_rng_uniform(runState->GSLrandom);

  return min + x*(max - min);
}

static REAL8 
draw_chirp(LALInferenceRunState *runState) {
  REAL8 min, max;

  LALInferenceGetMinMaxPrior(runState->priorArgs, "chirpmass", &min, &max);

  REAL8 x = gsl_rng_uniform(runState->GSLrandom);

  return pow(pow(min, -5.0/6.0) - x*(pow(min, -5.0/6.0) - pow(max, -5.0/6.0)), -6.0/5.0);
}

static REAL8
approxLogPrior(LALInferenceVariables *params) {
  REAL8 logP = 0.0;

  REAL8 Mc = *(REAL8 *)LALInferenceGetVariable(params, "chirpmass");
  logP += -11.0/6.0*log(Mc);

  /* Flat in eta. */

  REAL8 iota = *(REAL8 *)LALInferenceGetVariable(params, "inclination");
  logP += log(sin(iota));

  /* Flat in time, ra, psi, phi. */

  REAL8 dist = *(REAL8 *)LALInferenceGetVariable(params, "distance");
  logP += 2.0*log(dist);

  REAL8 dec = *(REAL8 *)LALInferenceGetVariable(params, "declination");
  logP += log(cos(dec));

  if (LALInferenceCheckVariable(params, "theta_spin1")) {
    REAL8 theta1 = *(REAL8 *)LALInferenceGetVariable(params, "theta_spin1");
    logP += log(sin(theta1));
  }

  if (LALInferenceCheckVariable(params, "theta_spin2")) {
    REAL8 theta2 = *(REAL8 *)LALInferenceGetVariable(params, "theta_spin2");
    logP += log(sin(theta2));
  }

  return logP;
}

void 
LALInferenceDrawApproxPrior(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = drawApproxPriorName;
  UINT4 goodProp = 0;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  while (goodProp == 0) {
    REAL8 Mc = draw_chirp(runState);
    LALInferenceSetVariable(proposedParams, "chirpmass", &Mc);

    if (LALInferenceCheckVariableNonFixed(runState->currentParams, "asym_massratio")) {
      REAL8 q = draw_flat(runState, "asym_massratio");
      LALInferenceSetVariable(proposedParams, "asym_massratio", &q);
    }
    else if (LALInferenceCheckVariableNonFixed(runState->currentParams, "massratio")) {
      REAL8 eta = draw_flat(runState, "massratio");
      LALInferenceSetVariable(proposedParams, "massratio", &eta);
    }

    REAL8 theTime = draw_flat(runState, "time");
    LALInferenceSetVariable(proposedParams, "time", &theTime);

    REAL8 phase = draw_flat(runState, "phase");
    LALInferenceSetVariable(proposedParams, "phase", &phase);

    REAL8 inc = draw_colatitude(runState, "inclination");
    LALInferenceSetVariable(proposedParams, "inclination", &inc);

    REAL8 pol = draw_flat(runState, "polarisation");
    LALInferenceSetVariable(proposedParams, "polarisation", &pol);

    REAL8 dist = draw_distance(runState);
    LALInferenceSetVariable(proposedParams, "distance", &dist);

    REAL8 ra = draw_flat(runState, "rightascension");
    LALInferenceSetVariable(proposedParams, "rightascension", &ra);

    REAL8 dec = draw_dec(runState);
    LALInferenceSetVariable(proposedParams, "declination", &dec);

    if (LALInferenceCheckVariableNonFixed(proposedParams, "a_spin1")) {
      REAL8 a1 = draw_flat(runState, "a_spin1");
      LALInferenceSetVariable(proposedParams, "a_spin1", &a1);
    }

    if (LALInferenceCheckVariableNonFixed(proposedParams, "a_spin2")) {
      REAL8 a2 = draw_flat(runState, "a_spin2");
      LALInferenceSetVariable(proposedParams, "a_spin2", &a2);
    }

    if (LALInferenceCheckVariableNonFixed(proposedParams, "phi_spin1")) {
      REAL8 phi1 = draw_flat(runState, "phi_spin1");
      LALInferenceSetVariable(proposedParams, "phi_spin1", &phi1);
    }

    if (LALInferenceCheckVariableNonFixed(proposedParams, "phi_spin2")) {
      REAL8 phi2 = draw_flat(runState, "phi_spin2");
      LALInferenceSetVariable(proposedParams, "phi_spin2", &phi2);
    }

    if (LALInferenceCheckVariableNonFixed(proposedParams, "theta_spin1")) {
      REAL8 theta1 = draw_colatitude(runState, "theta_spin1");
      LALInferenceSetVariable(proposedParams, "theta_spin1", &theta1);
    }

    if (LALInferenceCheckVariableNonFixed(proposedParams, "theta_spin2")) {
      REAL8 theta2 = draw_colatitude(runState, "theta_spin2");
      LALInferenceSetVariable(proposedParams, "theta_spin2", &theta2);
    }
    if (runState->prior(runState, proposedParams) > -DBL_MAX)
      goodProp = 1;
  }

  LALInferenceSetLogProposalRatio(runState, approxLogPrior(runState->currentParams) - approxLogPrior(proposedParams));
}

static void
cross_product(REAL8 x[3], const REAL8 y[3], const REAL8 z[3]) {
  x[0] = y[1]*z[2]-y[2]*z[1];
  x[1] = y[2]*z[0]-y[0]*z[2];
  x[2] = y[0]*z[1]-y[1]*z[0];
}

static REAL8
norm(const REAL8 x[3]) {
  return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

static void 
unit_vector(REAL8 v[3], const REAL8 w[3]) {
  REAL8 n = norm(w);

  if (n == 0.0) { 
    XLALError("unit_vector", __FILE__, __LINE__, XLAL_FAILURE);
    exit(1);
  } else {
    v[0] = w[0] / n;
    v[1] = w[1] / n;
    v[2] = w[2] / n;
  }
}

static REAL8 
dot(const REAL8 v[3], const REAL8 w[3]) {
  return v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
}

static void
project_along(REAL8 vproj[3], const REAL8 v[3], const REAL8 w[3]) {
  REAL8 what[3];
  REAL8 vdotw;

  unit_vector(what, w);
  vdotw = dot(v, w);

  vproj[0] = what[0]*vdotw;
  vproj[1] = what[1]*vdotw;
  vproj[2] = what[2]*vdotw;
}

static void
vsub(REAL8 diff[3], const REAL8 w[3], const REAL8 v[3]) {
  diff[0] = w[0] - v[0];
  diff[1] = w[1] - v[1];
  diff[2] = w[2] - v[2];
}

static void
vadd(REAL8 sum[3], const REAL8 w[3], const REAL8 v[3]) {
  sum[0] = w[0] + v[0];
  sum[1] = w[1] + v[1];
  sum[2] = w[2] + v[2];
}

static void
reflect_plane(REAL8 pref[3], const REAL8 p[3], 
              const REAL8 x[3], const REAL8 y[3], const REAL8 z[3]) {
  REAL8 n[3], nhat[3], xy[3], xz[3], pn[3], pnperp[3];

  vsub(xy, y, x);
  vsub(xz, z, x);

  cross_product(n, xy, xz);
  unit_vector(nhat, n);

  project_along(pn, p, nhat);
  vsub(pnperp, p, pn);

  vsub(pref, pnperp, pn);
}

static void 
sph_to_cart(REAL8 cart[3], const REAL8 lat, const REAL8 longi) {
  cart[0] = cos(longi)*cos(lat);
  cart[1] = sin(longi)*cos(lat);
  cart[2] = sin(lat);
}

static void
cart_to_sph(const REAL8 cart[3], REAL8 *lat, REAL8 *longi) {
  *longi = atan2(cart[1], cart[0]);
  *lat = asin(cart[2] / sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2]));
}

static void
reflected_position_and_time(LALInferenceRunState *runState, const REAL8 ra, const REAL8 dec, const REAL8 oldTime,
                            REAL8 *newRA, REAL8 *newDec, REAL8 *newTime) {
  LALStatus status;
  memset(&status,0,sizeof(status));
  SkyPosition currentEqu, currentGeo, newEqu, newGeo;
  currentEqu.latitude = dec;
  currentEqu.longitude = ra;
  currentEqu.system = COORDINATESYSTEM_EQUATORIAL;
  currentGeo.system = COORDINATESYSTEM_GEOGRAPHIC;
  LALEquatorialToGeographic(&status, &currentGeo, &currentEqu, &(runState->data->epoch));

  /* This function should only be called when we know that we have
     three detectors, or the following will crash. */
  REAL8 x[3], y[3], z[3];
  LALInferenceIFOData *xD = runState->data;
  memcpy(x, xD->detector->location, 3*sizeof(REAL8));

  LALInferenceIFOData *yD = xD->next;
  while (same_detector_location(yD, xD)) {
    yD = yD->next;
  }
  memcpy(y, yD->detector->location, 3*sizeof(REAL8));

  LALInferenceIFOData *zD = yD->next;
  while (same_detector_location(zD, yD) || same_detector_location(zD, xD)) {
    zD = zD->next;
  }
  memcpy(z, zD->detector->location, 3*sizeof(REAL8));

  REAL8 currentLoc[3];
  sph_to_cart(currentLoc, currentGeo.latitude, currentGeo.longitude);

  REAL8 newLoc[3];
  reflect_plane(newLoc, currentLoc, x, y, z);

  REAL8 newGeoLat, newGeoLongi;
  cart_to_sph(newLoc, &newGeoLat, &newGeoLongi);

  newGeo.latitude = newGeoLat;
  newGeo.longitude = newGeoLongi;
  newGeo.system = COORDINATESYSTEM_GEOGRAPHIC;
  newEqu.system = COORDINATESYSTEM_EQUATORIAL;
  LALGeographicToEquatorial(&status, &newEqu, &newGeo, &(runState->data->epoch));

  REAL8 oldDt, newDt;
  oldDt = XLALTimeDelayFromEarthCenter(runState->data->detector->location, currentEqu.longitude,
                                       currentEqu.latitude, &(runState->data->epoch));
  newDt = XLALTimeDelayFromEarthCenter(runState->data->detector->location, newEqu.longitude,
                                       newEqu.latitude, &(runState->data->epoch));

  *newRA = newEqu.longitude;
  *newDec = newEqu.latitude;
  *newTime = oldTime + oldDt - newDt;
}

void LALInferenceSkyReflectDetPlane(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = skyReflectDetPlaneName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  /* Find the number of distinct-position detectors. */
  /* Exit with same parameters (with a warning the first time) if
     there are not three detectors. */
  static UINT4 warningDelivered = 0;
  if (numDetectorsUniquePositions(runState) != 3) {
    if (warningDelivered) {
      /* Do nothing. */
    } else {
      fprintf(stderr, "WARNING: trying to reflect through the decector plane with %d\n", numDetectorsUniquePositions(runState));
      fprintf(stderr, "WARNING: geometrically independent locations,\n");
      fprintf(stderr, "WARNING: but this proposal should only be used with exactly 3 independent detectors.\n");
      fprintf(stderr, "WARNING: %s, line %d\n", __FILE__, __LINE__);
      warningDelivered = 1;
    }

    return; 
  }

  REAL8 ra = *(REAL8 *)LALInferenceGetVariable(proposedParams, "rightascension");
  REAL8 dec = *(REAL8 *)LALInferenceGetVariable(proposedParams, "declination");
  REAL8 baryTime = *(REAL8 *)LALInferenceGetVariable(proposedParams, "time");

  REAL8 newRA, newDec, newTime;
  reflected_position_and_time(runState, ra, dec, baryTime, &newRA, &newDec, &newTime);

  /* Unit normal deviates, used to "fuzz" the state. */
  REAL8 nRA, nDec, nTime;
  const REAL8 epsTime = 6e-6; /* 1e-1 / (16 kHz) */
  const REAL8 epsAngle = 3e-4; /* epsTime*c/R_Earth */
  
  nRA = gsl_ran_ugaussian(runState->GSLrandom);
  nDec = gsl_ran_ugaussian(runState->GSLrandom);
  nTime = gsl_ran_ugaussian(runState->GSLrandom);

  newRA += epsAngle*nRA;
  newDec += epsAngle*nDec;
  newTime += epsTime*nTime;

  /* And the doubly-reflected position (near the original, but not
     exactly due to the fuzzing). */
  REAL8 refRA, refDec, refTime;
  reflected_position_and_time(runState, newRA, newDec, newTime, &refRA, &refDec, &refTime);

  /* The Gaussian increments required to shift us back to the original
     position from the doubly-reflected position. */
  REAL8 nRefRA, nRefDec, nRefTime;
  nRefRA = (ra - refRA)/epsAngle;
  nRefDec = (dec - refDec)/epsAngle;
  nRefTime = (baryTime - refTime)/epsTime;

  REAL8 pForward, pReverse;
  pForward = gsl_ran_ugaussian_pdf(nRA)*gsl_ran_ugaussian_pdf(nDec)*gsl_ran_ugaussian_pdf(nTime);
  pReverse = gsl_ran_ugaussian_pdf(nRefRA)*gsl_ran_ugaussian_pdf(nRefDec)*gsl_ran_ugaussian_pdf(nRefTime);

  LALInferenceSetVariable(proposedParams, "rightascension", &newRA);
  LALInferenceSetVariable(proposedParams, "declination", &newDec);
  LALInferenceSetVariable(proposedParams, "time", &newTime);
  LALInferenceSetLogProposalRatio(runState, log(pReverse/pForward));
}

static void
rotateVectorAboutAxis(REAL8 vrot[3],
                      const REAL8 v[3],
                      const REAL8 axis[3],
                      const REAL8 theta) {
  REAL8 vperp[3], vpar[3], vperprot[3];
  REAL8 xhat[3], yhat[3], zhat[3];
  REAL8 vp;
  UINT4 i;

  project_along(vpar, v, axis);
  vsub(vperp, v, vpar);

  vp = norm(vperp);

  unit_vector(zhat, axis);
  unit_vector(xhat, vperp);
  cross_product(yhat, zhat, xhat);

  for (i = 0; i < 3; i++) {
    vperprot[i] = vp*(cos(theta)*xhat[i] + sin(theta)*yhat[i]);
  }

  vadd(vrot, vpar, vperprot);
}

static void
vectorToColatLong(const REAL8 v[3],
                  REAL8 *colat, REAL8 *longi) { 
  *longi = atan2(v[1], v[0]);
  if (*longi < 0.0) {
    *longi += 2.0*M_PI;
  }

  *colat = acos(v[2] / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]));
}

void 
LALInferenceRotateSpins(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = rotateSpinsName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  REAL8 theta1 = 2.0*M_PI*gsl_rng_uniform(runState->GSLrandom);
  REAL8 theta2 = 2.0*M_PI*gsl_rng_uniform(runState->GSLrandom);

  REAL8 logPr = 0.0;

  if (LALInferenceCheckVariableNonFixed(proposedParams, "theta_spin1")) {
    REAL8 theta, phi, iota;
    REAL8 s1[3], L[3], newS[3];
    
    theta = *(REAL8 *)LALInferenceGetVariable(proposedParams, "theta_spin1");
    phi = *(REAL8 *)LALInferenceGetVariable(proposedParams, "phi_spin1");

    iota = *(REAL8 *)LALInferenceGetVariable(proposedParams, "inclination");

    s1[0] = cos(phi)*sin(theta);
    s1[1] = sin(phi)*sin(theta);
    s1[2] = cos(theta);

    L[0] = sin(iota);
    L[1] = 0.0;
    L[2] = cos(iota);

    rotateVectorAboutAxis(newS, s1, L, theta1);

    REAL8 newPhi, newTheta;

    vectorToColatLong(newS, &newTheta, &newPhi);

    /* Since the proposal is inherently uniform on the surface of the
       sphere, we only need to account for the volume factors between
       cos(theta) and theta. */
    logPr += log(sin(theta)/sin(newTheta));

    LALInferenceSetVariable(proposedParams, "phi_spin1", &newPhi);
    LALInferenceSetVariable(proposedParams, "theta_spin1", &newTheta);
  }

  if (LALInferenceCheckVariableNonFixed(proposedParams, "theta_spin2")) {
    REAL8 theta, phi, iota;
    REAL8 s2[3], L[3], newS[3];
    
    theta = *(REAL8 *)LALInferenceGetVariable(proposedParams, "theta_spin2");
    phi = *(REAL8 *)LALInferenceGetVariable(proposedParams, "phi_spin2");

    iota = *(REAL8 *)LALInferenceGetVariable(proposedParams, "inclination");

    s2[0] = cos(phi)*sin(theta);
    s2[1] = sin(phi)*sin(theta);
    s2[2] = cos(theta);

    L[0] = sin(iota);
    L[1] = 0.0;
    L[2] = cos(iota);

    rotateVectorAboutAxis(newS, s2, L, theta2);

    REAL8 newPhi, newTheta;

    vectorToColatLong(newS, &newTheta, &newPhi);

    /* Since the proposal is inherently uniform on the surface of the
       sphere, we only need to account for the volume factors between
       cos(theta) and theta. */
    logPr += log(sin(theta)/sin(newTheta));

    LALInferenceSetVariable(proposedParams, "phi_spin2", &newPhi);
    LALInferenceSetVariable(proposedParams, "theta_spin2", &newTheta);
  }

  LALInferenceSetLogProposalRatio(runState, logPr);
}

void
LALInferencePolarizationPhaseJump(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = polarizationPhaseJumpName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  REAL8 psi = *(REAL8 *)LALInferenceGetVariable(proposedParams, "polarisation");
  REAL8 phi = *(REAL8 *)LALInferenceGetVariable(proposedParams, "phase");

  phi += M_PI;
  psi += M_PI/2;

  phi = fmod(phi, 2.0*M_PI);
  psi = fmod(psi, M_PI);

  LALInferenceSetVariable(proposedParams, "polarisation", &psi);
  LALInferenceSetVariable(proposedParams, "phase", &phi);

  LALInferenceSetLogProposalRatio(runState, 0.0);
}

typedef enum {
  USES_DISTANCE_VARIABLE,
  USES_LOG_DISTANCE_VARIABLE
} DistanceParam;

void LALInferenceDistanceQuasiGibbsProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = distanceQuasiGibbsProposalName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  DistanceParam distParam;

  if (LALInferenceCheckVariable(proposedParams, "distance")) {
    distParam = USES_DISTANCE_VARIABLE;
  } else if (LALInferenceCheckVariable(proposedParams, "logdistance")) {
    distParam = USES_LOG_DISTANCE_VARIABLE;
  } else {
    XLAL_ERROR_VOID(XLAL_FAILURE, "could not find 'distance' or 'logdistance' in current params");
  }

  REAL8 d0;
  if (distParam == USES_DISTANCE_VARIABLE) {
    d0 = *(REAL8 *)LALInferenceGetVariable(proposedParams, "distance");
  } else {
    d0 = exp(*(REAL8 *)LALInferenceGetVariable(proposedParams, "logdistance"));
  }

  REAL8 u0 = 1.0 / d0;
  REAL8 L0 = runState->currentLikelihood;

  /* We know that the likelihood surface looks like L(u) = A + B*u +
     C*u^2, where u = 1/d is the inverse distance.  We can find these
     coefficients by fitting the value of the likelihood at three
     different points: u0, u0/2, and 2*u0. */
  REAL8 u12 = u0/2.0;
  REAL8 d2 = 1.0/u12;
  if (distParam == USES_DISTANCE_VARIABLE) {
    LALInferenceSetVariable(proposedParams, "distance", &d2);
  } else {
    REAL8 logD2 = log(d2);
    LALInferenceSetVariable(proposedParams, "logdistance", &logD2);
  }
  REAL8 L12 = runState->likelihood(proposedParams, runState->data, runState->template);

  REAL8 u2 = u0*2.0;
  REAL8 d12 = 1.0/u2;
  if (distParam == USES_DISTANCE_VARIABLE) {
    LALInferenceSetVariable(proposedParams, "distance", &d12);
  } else {
    REAL8 logD12 = log(d12);
    LALInferenceSetVariable(proposedParams, "logdistance", &logD12);
  }
  REAL8 L2 = runState->likelihood(proposedParams, runState->data, runState->template);
  
  /* Coefficients of quadratic L(u) = A + B*u + C*u^2 */
  REAL8 B = -(L2 + 4.0*L12 - 5.0*L0)/u0;
  REAL8 C = (2.0*L2 + 4.0*L12 - 6.0*L0)/(3.0*u0*u0);

  /* Convert quadratic log(L) in u to Gaussian parameters. */
  REAL8 mu = -B / (2.0*C);
  REAL8 sigma2 = 1.0 / (2.0*C);

  static INT8 weirdProposalCount = 0;
  static INT8 thresholdProposalCount = 1;

  if (C<=0.0) {
    /* Flat or linear likelihood, or negative curvature in the
       gaussian---choose uniformly in prior range. */
    weirdProposalCount++;
    if (weirdProposalCount >= thresholdProposalCount) {
      thresholdProposalCount *= 2;
      XLAL_PRINT_WARNING("found infinite or negative sigma^2 (%g), using fallback proposal (for the %dth time overall)",
                         sigma2, weirdProposalCount);
    }
    if (distParam == USES_DISTANCE_VARIABLE) {
      REAL8 dMax, dMin;
      LALInferenceGetMinMaxPrior(runState->priorArgs, "distance", &dMin, &dMax);
      REAL8 dNew = dMin + (dMax-dMin)*gsl_rng_uniform(runState->GSLrandom);
      
      LALInferenceSetVariable(proposedParams, "distance", &dNew);
      LALInferenceSetLogProposalRatio(runState, 0.0);
      return;
    } else {
      REAL8 logDMin, logDMax;
      LALInferenceGetMinMaxPrior(runState->priorArgs, "logdistance", &logDMin, &logDMax);
      REAL8 logDNew = logDMin + (logDMax - logDMin)*gsl_rng_uniform(runState->GSLrandom);

      LALInferenceSetVariable(proposedParams, "logdistance", &logDNew);
      LALInferenceSetLogProposalRatio(runState, 0.0);
      return;
    }
  }

  REAL8 sigma = sqrt(sigma2);

  /* Draw new u from Gaussian, convert to d. */
  REAL8 uNew = mu + sigma*gsl_ran_ugaussian(runState->GSLrandom);
  REAL8 dNew = 1.0/uNew;
  
  if (distParam == USES_DISTANCE_VARIABLE) {
    LALInferenceSetVariable(proposedParams, "distance", &dNew);
  } else {
    REAL8 logDNew = log(dNew);
    LALInferenceSetVariable(proposedParams, "logdistance", &logDNew);
  }

  REAL8 LNew = runState->likelihood(proposedParams, runState->data, runState->template);

  /* Store our new sample and set jump probability. */
  if (distParam == USES_DISTANCE_VARIABLE) {
    /* Since we jumped using the likelihood gaussian in u = 1/d, p(d)
       = exp(L(u))/d^2. */
    LALInferenceSetVariable(proposedParams, "distance", &dNew);
    LALInferenceSetLogProposalRatio(runState, L0 - 2.0*log(d0) - LNew + 2.0*log(dNew));
  } else {
    /* Jump probability density is different if we jump in logs.  If
       we jumped in log(d) = -log(u), then we have p(log(d)) = u
       exp(L(u)) = exp(L(u))/d */
    REAL8 logDNew = log(dNew);
    LALInferenceSetVariable(proposedParams, "logdistance", &logDNew);
    LALInferenceSetLogProposalRatio(runState, L0 - log(d0) - LNew + log(dNew));
  }

  return;
}

/* We know that the likelihood with all variables but phase fixed can
   be written as 

   log(L) = <d|d> + 2*Re(<d|h>)cos(dPhi) +/- 2*Im(<d|h>)sin(dPhi) + <h|h>

   This proposal computes the coefficients of the likelihood and draws
   phase from the analytic distribution that results. */
void LALInferenceOrbitalPhaseQuasiGibbsProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  const char *propName = orbitalPhaseQuasiGibbsProposalName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);
  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  REAL8 L0, L1, L2;
  REAL8 dPhi0, dPhi1, dPhi2;
  REAL8 phi0, phi1, phi2;

  phi0 = *(REAL8 *)LALInferenceGetVariable(proposedParams, "phase");
  dPhi0 = 0.0;
  L0 = runState->currentLikelihood;

  dPhi1 = 2.0*M_PI/3.0;
  phi1 = fmod(phi0 + dPhi1, 2.0*M_PI);
  LALInferenceSetVariable(proposedParams, "phase", &phi1);
  L1 = runState->likelihood(proposedParams, runState->data, runState->template);

  dPhi2 = 4.0*M_PI/3.0;
  phi2 = fmod(phi0 + dPhi2, 2.0*M_PI);
  LALInferenceSetVariable(proposedParams, "phase", &phi2);
  L2 = runState->likelihood(proposedParams, runState->data, runState->template);

  /* log(L) = A + C*cos(dPhi) + S*sin(dPhi) */
  REAL8 c0 = cos(dPhi0), c1 = cos(dPhi1), c2 = cos(dPhi2);
  REAL8 s0 = sin(dPhi0), s1 = sin(dPhi1), s2 = sin(dPhi2);
  REAL8 A = (L0 + L1 + L2) / 3.0;
  REAL8 C = (L0 + L1*cos(dPhi1) + L2*cos(dPhi2))/(c0*c0 + c1*c1 + c2*c2);
  REAL8 S = (L1*sin(dPhi1) + L2*sin(dPhi2))/(s0*s0 + s1*s1 + s2*s2);

  /* One extremum of L: */
  REAL8 dphiMax = atan(S/C);
  REAL8 LMax = A + C*cos(dphiMax) + S*sin(dphiMax);

  /* Is the other extremum a maxmim? */
  if (A - C*cos(dphiMax) - S*sin(dphiMax) > LMax) {
    LMax = A - C*cos(dphiMax) - S*sin(dphiMax);
    dphiMax = fmod(dphiMax + M_PI, 2.0*M_PI);
  }

  /* Now von Neumann rejection sample in the dPhi, L plane until we
     get a dPhi that matches.  The only tricky part is that we choose
     the vertical coordinate in log-space---that is, we choose log(y)
     uniformly in *y* between 0 and exp(LMax). */
  REAL8 dPhi;
  REAL8 logY;
  do { 
    dPhi = 2.0*M_PI*gsl_rng_uniform(runState->GSLrandom);
    logY = LMax + log(gsl_rng_uniform(runState->GSLrandom));
  } while (logY > A + C*cos(dPhi) + S*sin(dPhi));
  
  REAL8 LDPhi = A + C*cos(dPhi) + S*sin(dPhi);

  REAL8 phiNew = fmod(phi0 + dPhi, 2.0*M_PI);
  LALInferenceSetVariable(proposedParams, "phase", &phiNew);
  LALInferenceSetLogProposalRatio(runState, L0 - LDPhi);
}

static int inBounds(REAL8 *pt, REAL8 *low, REAL8 *high, size_t N) {
  size_t i;
  for (i = 0; i < N; i++) {
    if (pt[i] < low[i] || pt[i] > high[i]) {
      return 0;
    }
  }

  return 1;
}

void LALInferenceKDNeighborhoodProposal(LALInferenceRunState *runState, LALInferenceVariables *proposedParams) {
  size_t NCell;
  LALInferenceVariables *proposalArgs = runState->proposalArgs;

  if (LALInferenceCheckVariable(runState->proposalArgs, "KDNCell")) {
    NCell = *(INT4 *)LALInferenceGetVariable(runState->proposalArgs, "KDNCell");
  } else {
    /* NCell default value. */
    NCell = 64;
  }

  const char *propName = KDNeighborhoodProposalName;
  LALInferenceSetVariable(runState->proposalArgs, LALInferenceCurrentProposalName, &propName);

  LALInferenceCopyVariables(runState->currentParams, proposedParams);

  if (!LALInferenceCheckVariable(proposalArgs, "kDTree") || !LALInferenceCheckVariable(proposalArgs, "kDTreeVariableTemplate")) {
    /* For whatever reason, the appropriate data are not set up in the
       proposalArgs, so just propose the current point again and
       bail. */
    LALInferenceSetLogProposalRatio(runState, 0.0);
    return;
  }
  
  LALInferenceKDTree *tree = *(LALInferenceKDTree **)LALInferenceGetVariable(proposalArgs, "kDTree");
  LALInferenceVariables *template = *(LALInferenceVariables **)LALInferenceGetVariable(proposalArgs, "kDTreeVariableTemplate");
  REAL8 *pt = XLALMalloc(tree->ndim*sizeof(REAL8));

  /* If tree has zero points, bail. */
  if (tree->npts == 0) {
    LALInferenceSetLogProposalRatio(runState, 0.0);
    return;
  }

  /* A randomly-chosen point from those in the tree. */
  REAL8 *thePt = tree->pts[gsl_rng_uniform_int(runState->GSLrandom, tree->npts)];
  LALInferenceKDCell *aCell = LALInferenceKDFindCell(tree, thePt, NCell);

  /* Proposed params chosen randomly from within the box bounding
     points in aCell. */
  size_t i;
  for (i = 0; i < tree->ndim; i++) {
    REAL8 delta = aCell->pointsUpperRight[i] - aCell->pointsLowerLeft[i];
    pt[i] = aCell->pointsLowerLeft[i] + delta*gsl_rng_uniform(runState->GSLrandom);
  }
  LALInferenceKDREAL8ToVariables(proposedParams, pt, template);

  /* Forward probability is N_Cell / N / Cell_Volume. */
  REAL8 logForwardProb = log(aCell->npts) - log(tree->npts) - LALInferenceKDLogPointsVolume(tree, aCell);

  /* To compute the backward jump probability, we need to know which
     box contains the current point. */
  LALInferenceKDVariablesToREAL8(runState->currentParams, pt, template);
  LALInferenceKDCell *currentCell = LALInferenceKDFindCell(tree, pt, NCell);

  /* Backward jump probability, based on the number of points and
     volume of the cell containing the current point. */
  REAL8 logBackwardProb;

  if (inBounds(pt, currentCell->pointsLowerLeft, currentCell->pointsUpperRight, tree->ndim)) {
    /* If the current point is within its cell's point-bounding-box, then this is jump probability. */
    logBackwardProb = log(currentCell->npts) - log(tree->npts) - LALInferenceKDLogPointsVolume(tree, currentCell);
  } else {
    /* If current point outside its cell's points-bounding-box, then there is no probability of backward jump. */
    logBackwardProb = log(0.0);
  }

  LALInferenceSetLogProposalRatio(runState, logBackwardProb - logForwardProb);

  /* Cleanup the allocated storage for currentPt. */
  XLALFree(pt);
}

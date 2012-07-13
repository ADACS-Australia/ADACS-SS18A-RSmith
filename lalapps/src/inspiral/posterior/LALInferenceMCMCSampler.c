/*
 *  LALInferenceMCMC.c:  Bayesian Followup, MCMC algorithm.
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
#include <lalapps.h>
#include <mpi.h>
#include <lal/LALInference.h>
#include "LALInferenceMCMCSampler.h"
#include <lal/LALInferencePrior.h>
#include <lal/LALInferenceLikelihood.h>
#include <lal/LALInferenceTemplate.h>
#include <lal/LALInferenceProposal.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOLwXMLInspiralRead.h>

#include <LALAppsVCSInfo.h>
#include <lal/LALStdlib.h>

#define PROGRAM_NAME "LALInferenceMCMCSampler.c"
#define CVS_ID_STRING "$Id$"
#define CVS_REVISION "$Revision$"
#define CVS_SOURCE "$Source$"
#define CVS_DATE "$Date$"
#define CVS_NAME_STRING "$Name$"

void LALInferencePTswap(LALInferenceRunState *runState, double *TcurrentLikelihood, REAL8 *parametersVec,
                        REAL8 *tempLadder, int lowerRank, int upperRank, int i, FILE *tempfile);
void LALInferenceAdaptation(LALInferenceRunState *runState, INT4 cycle);
void LALInferenceAdaptationRestart(LALInferenceRunState *runState, INT4 cycle);
void LALInferenceAdaptationEnvelope(LALInferenceRunState *runState, INT4 cycle);
FILE* LALInferencePrintPTMCMCHeader(LALInferenceRunState *runState);
void LALInferencePrintPTMCMCHeaderFile(LALInferenceRunState *runState, FILE *file);
void LALInferencePrintPTMCMCInjectionSample(LALInferenceRunState *runState);
void LALInferenceDataDump(LALInferenceRunState *runState);

static void
accumulateDifferentialEvolutionSample(LALInferenceRunState *runState) {
  if (runState->differentialPointsSize == runState->differentialPointsLength) {
    size_t newSize = runState->differentialPointsSize*2;
    runState->differentialPoints = XLALRealloc(runState->differentialPoints, newSize*sizeof(LALInferenceVariables *));
    runState->differentialPointsSize = newSize;
  }

  runState->differentialPoints[runState->differentialPointsLength] = XLALCalloc(1, sizeof(LALInferenceVariables));
  LALInferenceCopyVariables(runState->currentParams, runState->differentialPoints[runState->differentialPointsLength]);
  runState->differentialPointsLength += 1;
}

static void
accumulateKDTreeSample(LALInferenceRunState *runState) {
  LALInferenceVariables *proposalParams = runState->proposalArgs;

  if (!LALInferenceCheckVariable(proposalParams, "kDTree") || !LALInferenceCheckVariable(proposalParams, "kDTreeVariableTemplate")) {
    /* Not setup correctly. */
    return;
  }

  LALInferenceKDTree *tree = *(LALInferenceKDTree **)LALInferenceGetVariable(proposalParams, "kDTree");
  LALInferenceVariables *template = *(LALInferenceVariables **)LALInferenceGetVariable(proposalParams, "kDTreeVariableTemplate");
  size_t ndim = LALInferenceGetVariableDimensionNonFixed(template);
  REAL8 *pt = XLALMalloc(ndim*sizeof(REAL8));

  LALInferenceKDVariablesToREAL8(runState->currentParams, pt, template);

  LALInferenceKDAddPoint(tree, pt);

  /* Don't free pt, since it is now stored in tree. */
}

static void DEbuffer2array(LALInferenceRunState *runState, INT4 startCycle, INT4 endCycle, REAL8** DEarray) {
  LALInferenceVariableItem *ptr;
  INT4 i=0,p=0;

  INT4 Nskip = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Nskip");
  INT4 totalPoints = runState->differentialPointsLength;
  INT4 start = (INT4)ceil((REAL8)startCycle/(REAL8)Nskip);
  INT4 end = (INT4)floor((REAL8)endCycle/(REAL8)Nskip);
  /* Include last point */
  if (end > totalPoints-1)
    end = totalPoints-1;

  for (i = start; i <= end; i++) {
    ptr=runState->differentialPoints[i]->head;
    p=0;
    while(ptr!=NULL) {
      if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
        DEarray[i-start][p]=*(REAL8 *)ptr->value;
        p++;
      }
      ptr=ptr->next;
    }
  }
}

static void
array2DEbuffer(LALInferenceRunState *runState, INT4 startCycle, INT4 endCycle, REAL8** DEarray) {
  LALInferenceVariableItem *ptr;
  INT4 i=0,p=0;

  INT4 Nskip = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Nskip");
  INT4 totalPoints = runState->differentialPointsLength;
  INT4 start = (INT4)ceil((REAL8)startCycle/(REAL8)Nskip);
  INT4 end = (INT4)floor((REAL8)endCycle/(REAL8)Nskip);
  /* Include last point */
  if (end > totalPoints-1)
    end = totalPoints-1;

  for (i=start; i <= end; i++) {
    ptr=runState->differentialPoints[i]->head;
    p=0;
    while(ptr!=NULL) {
      if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
        *((REAL8 *)ptr->value) = (REAL8)DEarray[i-start][p];
        p++;
      }
      ptr=ptr->next;
    }
  }
}

static void
BcastDifferentialEvolutionPoints(LALInferenceRunState *runState, INT4 sourceTemp) {
  INT4 MPIrank;
  INT4 i=0;
  REAL8** packedDEsamples;
  REAL8*  temp;

  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);
  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);
  INT4 Nskip = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Nskip");
  INT4 nPoints = runState->differentialPointsLength;
  INT4 startCycle=0, endCycle=nPoints*Nskip;

  /* Prepare 2D array for DE points */
  packedDEsamples = (REAL8**) XLALMalloc(nPoints * sizeof(REAL8*));
  temp = (REAL8*) XLALMalloc(nPoints * nPar * sizeof(REAL8));
  for (i=0; i < nPoints; i++) {
    packedDEsamples[i] = temp + (i*nPar);
  }

  /* Pack it up */
  if (MPIrank == sourceTemp)
    DEbuffer2array(runState, startCycle, endCycle, packedDEsamples);

  /* Send it out */
  MPI_Bcast(packedDEsamples[0], nPoints*nPar, MPI_DOUBLE, sourceTemp, MPI_COMM_WORLD);

  /* Unpack it */
  if (MPIrank != sourceTemp) {
    array2DEbuffer(runState, startCycle, endCycle, packedDEsamples);
  }

  /* Clean up */
  XLALFree(temp);
  MPI_Barrier(MPI_COMM_WORLD);
}

static void
computeMaxAutoCorrLen(LALInferenceRunState *runState, INT4 startCycle, INT4 endCycle, INT4* maxACL) {
  INT4 Niter = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Niter");
  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);
  INT4 Nskip = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Nskip");
  INT4 totalPoints = runState->differentialPointsLength;
  INT4 start = (INT4)ceil((REAL8)startCycle/(REAL8)Nskip);
  INT4 end = (INT4)floor((REAL8)endCycle/(REAL8)Nskip);
  /* Include last point */
  if (end > totalPoints-1)
    end = totalPoints-1;
  INT4 nPoints = end - start + 1;
  REAL8** DEarray;
  REAL8*  temp;
  REAL8 mean, ACL, ACF, max=0;
  INT4 par=0, lag=0, i=0;
  int MPIrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);

  if (nPoints > 1) {
    /* Prepare 2D array for DE points */
    DEarray = (REAL8**) XLALMalloc(nPoints * sizeof(REAL8*));
    temp = (REAL8*) XLALMalloc(nPoints * nPar * sizeof(REAL8));
    for (i=0; i < nPoints; i++) {
      DEarray[i] = temp + (i*nPar);
    }

    DEbuffer2array(runState, startCycle, endCycle, DEarray);

    for (par=0; par<nPar; par++) {
      mean = gsl_stats_mean(&DEarray[0][par], nPar, nPoints);
      for (i=0; i<nPoints; i++)
        DEarray[i][par] -= mean;

      lag=1;
      ACL=1;
      ACF=1;
      while (ACF > 0.0005) {
        ACF = gsl_stats_correlation(&DEarray[0][par], nPar, &DEarray[lag][par], nPar, nPoints-lag);
        ACL += 2.0*ACF;
        lag++;
        /* If ACF[nPoints/2] > 0.0005 then assume ACL calculation will be inaccurate */
        if (lag > nPoints/2) {
          ACL=(REAL8)Niter/(REAL8)Nskip;
          break;
        }
      }
      ACL *= Nskip;
      if (ACL>max)
        max=ACL;
    }
    XLALFree(temp);
  } else {
    max = Niter;
  }

  *maxACL = (INT4)max;
}

static void
updateMaxAutoCorrLen(LALInferenceRunState *runState, INT4 currentCycle) {
  INT4 Niter = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Niter");
  REAL8 aclThreshold = *(REAL8*) LALInferenceGetVariable(runState->algorithmParams, "aclThreshold");
  INT4 proposedACL=0;
  INT4 adaptStart = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptStart");
  INT4 adaptLength = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptLength");
  INT4 iEffStart = adaptStart+adaptLength;
  INT4 acl=Niter;
  INT4 goodACL=0;

  if (iEffStart<currentCycle)
    computeMaxAutoCorrLen(runState, iEffStart, currentCycle, &proposedACL);

  if (proposedACL < aclThreshold*(currentCycle-iEffStart) && proposedACL != 0)
    acl = proposedACL;
  else if (LALInferenceCheckVariable(runState->algorithmParams, "goodACL"))
    LALInferenceSetVariable(runState->algorithmParams, "goodACL", &goodACL);

  LALInferenceSetVariable(runState->algorithmParams, "acl", &acl);
}

void PTMCMCAlgorithm(struct tagLALInferenceRunState *runState)
{
  int i,t,p,lowerRank,upperRank; //indexes for for() loops
  int nChain;
  int MPIrank, MPIsize;
  LALStatus status;
  memset(&status,0,sizeof(status));
  INT4 acceptanceCount = 0;
  INT4 swapAttempt=0;
  REAL8 nullLikelihood;
  REAL8 trigSNR = 0.0;
  REAL8 *tempLadder = NULL;			//the temperature ladder
  REAL8 *annealDecay = NULL;
  INT4 *acceptanceCountLadder = NULL;	//array of acceptance counts to compute the acceptance ratios.
  double *TcurrentLikelihood = NULL; //the current likelihood for each chain
  INT4 parameter=0;
  INT4 *intVec = NULL;
  INT4 annealStartIter = 0;
  INT4 iEffStart = 0;
  UINT4 hotChain = 0;                 // Affects proposal setup
  REAL8 *parametersVec = NULL;
  REAL8 tempDelta = 0.0;
  REAL8Vector * parameters = NULL;

  INT4 annealingOn = 0;
  INT4 adapting = 0;
  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);
  INT4 Niter = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Niter");
  INT4 Neff = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Neff");
  INT4 Nskip = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Nskip");
  UINT4 randomseed = *(UINT4*) LALInferenceGetVariable(runState->algorithmParams,"random_seed");
  INT4 acl=Niter, PTacl=Niter, oldACL=0, goodACL=0;
  INT4 quarterAclChecked=0;
  INT4 halfAclChecked=0;
  INT4 iEff=0;

  ProcessParamsTable *ppt;

  MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);

  nChain = MPIsize;		//number of parallel chain
  tempLadder = malloc(nChain * sizeof(REAL8));                  // Array of temperatures for parallel tempering.
  acceptanceCountLadder = (int*) malloc(sizeof(int)*nChain);		// Array of acceptance counts to compute the acceptance ratios.
  annealDecay = malloc(nChain * sizeof(REAL8));           			// Used by annealing scheme
  intVec = malloc(nChain * sizeof(INT4));


  if(MPIrank == 0){
    parametersVec = (REAL8 *)malloc(MPIsize*nPar*sizeof(REAL8));
    for (p=0;p<(nChain*nPar);++p){
      parametersVec[p] = 0.0;
    }
  }

  parameters = XLALCreateREAL8Vector(nPar);

  LALInferenceVariableItem *ptr=runState->currentParams->head;
  p=0;
  while(ptr!=NULL) {
    if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
      parameters->data[p]=*(REAL8 *)ptr->value;
      p++;
    }
    ptr=ptr->next;
  }

  /* If not specified otherwise, set effective sample size to total number of iterations */
  if (!Neff) {
    Neff = Niter;
    LALInferenceSetVariable(runState->algorithmParams, "Neff", &Neff);
  }

  /* Determine network SNR if injection was done */
  REAL8 networkSNRsqrd = 0.0;
  LALInferenceIFOData *IFO = runState->data;
  while (IFO != NULL) {
    networkSNRsqrd  += IFO->SNR * IFO->SNR;
    IFO = IFO->next;
  }

  /* Adaptation settings */
  LALInferenceSetupAdaptiveProposals(runState);
  REAL8Vector *PacceptCount = *((REAL8Vector **)LALInferenceGetVariable(runState->proposalArgs, "PacceptCount"));
  REAL8Vector *PproposeCount = *((REAL8Vector **)LALInferenceGetVariable(runState->proposalArgs, "PproposeCount"));
  REAL8Vector *sigmas = *((REAL8Vector **)LALInferenceGetVariable(runState->proposalArgs, LALInferenceSigmaJumpName));
  INT4  adaptationOn = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adaptationOn")); // Run adapts
  INT4  adaptTau     = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adaptTau"));     // Sets decay of adaption function
  INT4  adaptLength       = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adaptLength"));// Number of iterations to adapt before turning off
  INT4  adaptResetBuffer  = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adaptResetBuffer"));                // Number of iterations before adapting after a restart
  REAL8 s_gamma           = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "s_gamma"));                // Sets the size of changes to jump size during adaptation
  INT4  adaptStart        = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adaptStart"));                  // Keeps track of last iteration adaptation was restarted
  INT4  runPhase          = 0;                  // Phase of run. (0=PT-only run, 1=temporary PT, 2=annealing, 3=single-chain sampling)

  LALInferenceAddVariable(runState->algorithmParams, "acl", &acl,  LALINFERENCE_INT4_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(runState->algorithmParams, "goodACL", &goodACL,  LALINFERENCE_INT4_t, LALINFERENCE_PARAM_LINEAR);

  /* Temperature ladder settings */
  REAL8 tempMin = *(REAL8*) LALInferenceGetVariable(runState->algorithmParams, "tempMin");   // Min temperature in ladder
  REAL8 tempMax = *(REAL8*) LALInferenceGetVariable(runState->algorithmParams, "tempMax");   // Max temperature in ladder
  REAL8 targetHotLike       = 15;               // Targeted max 'experienced' log(likelihood) of hottest chain
  INT4  hotThreshold        = nChain/2-1;       // If MPIrank > hotThreshold, use proposals with higher acceptance rates for hot chains
  REAL8 aclThreshold        = 0.8*0.25;         // Make sure ACL is shorter than this fraction of the length of data used to compute it

  LALInferenceAddVariable(runState->algorithmParams, "aclThreshold", &aclThreshold,  LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);

  /* Set maximum temperature (command line value take precidence) */
  if (LALInferenceGetProcParamVal(runState->commandLine,"--tempMax")) {
    if(MPIrank==0)
      fprintf(stdout,"Using tempMax specified by commandline: %f.\n", tempMax);
  } else if (LALInferenceGetProcParamVal(runState->commandLine,"--trigSNR")) {        //--trigSNR given, choose tempMax to get targetHotLike
    trigSNR = *(REAL8*) LALInferenceGetVariable(runState->algorithmParams, "trigSNR");
    networkSNRsqrd = trigSNR * trigSNR;
    tempMax = networkSNRsqrd/(2*targetHotLike);
    if(MPIrank==0)
      fprintf(stdout,"Trigger SNR of %f specified, setting tempMax to %f.\n", trigSNR, tempMax);
  } else if (networkSNRsqrd > 0.0) {                                                  //injection, choose tempMax to get targetHotLike
    tempMax = networkSNRsqrd/(2*targetHotLike);
    if(MPIrank==0)
      fprintf(stdout,"Injecting SNR of %f, setting tempMax to %f.\n", sqrt(networkSNRsqrd), tempMax);
  } else {                                                                            //If all else fails, use the default
    tempMax = *(REAL8*) LALInferenceGetVariable(runState->algorithmParams, "tempMax");
    if(MPIrank==0)
      fprintf(stdout,"No --trigSNR or --tempMax specified, and not injecting a signal. Setting tempMax to default of %f.\n", tempMax);
  }
  LALInferenceSetVariable(runState->algorithmParams, "tempMax", &tempMax);

  if (tempMin > tempMax) {
    fprintf(stdout,"WARNING: tempMin > tempMax.  Forcing tempMin=1.0.\n");
    tempMin = 1.0;
    LALInferenceSetVariable(runState->algorithmParams, "tempMin", &tempMin);
  }

  /* Parallel tempering settings */
  INT4 tempSwaps        = (nChain-1)*nChain/2;                // Number of proposed swaps between temperatures in one swap iteration
  if (LALInferenceGetProcParamVal(runState->commandLine,"--tempSwaps"))
    tempSwaps = atoi(LALInferenceGetProcParamVal(runState->commandLine,"--tempSwaps")->value);

  INT4 Tskip            = 100;                                // Number of iterations between proposed temperature swaps 
  if (LALInferenceGetProcParamVal(runState->commandLine,"--tempSkip"))
    Tskip = atoi(LALInferenceGetProcParamVal(runState->commandLine,"--tempSkip")->value);

  INT4  annealStart     = 500;                                // # of autocorrelation lengths after adaptation before annealing
  INT4  annealLength    = 100;                                // # of autocorrelation lenghts to cool temperatures to ~1.0

  ppt=LALInferenceGetProcParamVal(runState->commandLine, "--anneal");
  if (ppt) {
    annealingOn = 1;                                          // Flag to indicate annealing is being used during the run
    runPhase=1;

    if (LALInferenceGetProcParamVal(runState->commandLine,"--annealStart"))
      annealStart = atoi(LALInferenceGetProcParamVal(runState->commandLine,"--annealStart")->value);

    if (LALInferenceGetProcParamVal(runState->commandLine,"--annealLength"))
      annealLength = atoi(LALInferenceGetProcParamVal(runState->commandLine,"--annealLength")->value);
  }

  for (t=0; t<nChain; ++t) {
    tempLadder[t] = 0.0;
    acceptanceCountLadder[t] = 0;
  }

  if (MPIrank == 0) {
    TcurrentLikelihood = (double*) malloc(sizeof(double)*nChain);
  }


//  if (runState->likelihood==&LALInferenceTimeDomainLogLikelihood) {
//    fprintf(stderr, "Computing null likelihood in time domain.\n");
//    nullLikelihood = LALInferenceTimeDomainNullLogLikelihood(runState->data);
//  } else 
  if (runState->likelihood==&LALInferenceUndecomposedFreqDomainLogLikelihood ||
      runState->likelihood==&LALInferenceFreqDomainLogLikelihood) {
    nullLikelihood = LALInferenceNullLogLikelihood(runState->data);
  } else if (runState->likelihood==&LALInferenceFreqDomainStudentTLogLikelihood) {
    REAL8 d = *(REAL8 *)LALInferenceGetVariable(runState->currentParams, "distance");
    REAL8 bigD = 1.0 / 0.0;

    LALInferenceSetVariable(runState->currentParams, "distance", &bigD);
    nullLikelihood = runState->likelihood(runState->currentParams, runState->data, runState->template);
    LALInferenceSetVariable(runState->currentParams, "distance", &d);
  } else if (runState->likelihood==&LALInferenceZeroLogLikelihood) {
    nullLikelihood = 0.0;
  } else if (runState->likelihood==&LALInferenceCorrelatedAnalyticLogLikelihood) {
    nullLikelihood = 0.0;
  } else if (runState->likelihood==&LALInferenceBimodalCorrelatedAnalyticLogLikelihood) {
    nullLikelihood = 0.0;
  } else {
    fprintf(stderr, "Unrecognized log(L) function (in %s, line %d)\n",
        __FILE__, __LINE__);
    exit(1);
  }

  // initialize starting likelihood value:
  runState->currentLikelihood = runState->likelihood(runState->currentParams, runState->data, runState->template);
  LALInferenceIFOData *headData = runState->data;
  while (headData != NULL) {
    headData->acceptedloglikelihood = headData->loglikelihood;
    headData = headData->next;
  }
  runState->currentPrior = runState->prior(runState, runState->currentParams);

  LALInferenceAddVariable(runState->algorithmParams, "nChain", &nChain,  LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(runState->algorithmParams, "nPar", &nPar,  LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(runState->proposalArgs, "parameter",&parameter, LALINFERENCE_INT4_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(runState->proposalArgs, "nullLikelihood", &nullLikelihood, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(runState->proposalArgs, "acceptanceCount", &acceptanceCount,  LALINFERENCE_INT4_t, LALINFERENCE_PARAM_LINEAR);
  REAL8 logLAtAdaptStart = runState->currentLikelihood;
  LALInferenceSetVariable(runState->proposalArgs, "logLAtAdaptStart", &(logLAtAdaptStart));

  /* Construct temperature ladder */
  if(nChain > 1){
    if(LALInferenceGetProcParamVal(runState->commandLine, "--inverseLadder")) {     //Spacing uniform in 1/T
      tempDelta = (1.0/tempMin - 1.0/tempMax)/(REAL8)(nChain-1);
      for (t=0; t<nChain; ++t) {
        tempLadder[t]=1.0/(REAL8)(1.0/tempMin-t*tempDelta);
      }
    } else {                                                                        //Geometric spacing
      tempDelta=pow(tempMax-tempMin+1,1.0/(REAL8)(nChain-1));
      for (t=0;t<nChain; ++t) {
        tempLadder[t]=tempMin + pow(tempDelta,t) - 1.0;
      }
    }
  } else {                                                                          //single chain
    if(LALInferenceGetProcParamVal(runState->commandLine,"--tempMax")){             //assume --tempMax specified intentionally
      tempLadder[0]=tempMax;
    }else{
      tempLadder[0]=1.0;
      tempMax=1.0;
    }
  }

  if (MPIrank > hotThreshold) {
    hotChain = 1;
  }

  LALInferenceAddVariable(runState->proposalArgs, "temperature", &(tempLadder[MPIrank]),  LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(runState->proposalArgs, "hotChain", &hotChain, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_OUTPUT);

  if (MPIrank == 0){
    printf("\nTemperature ladder:\n");
    for (t=0; t<nChain; ++t) {
      printf(" tempLadder[%d]=%f\n",t,tempLadder[t]);
    }
  }


  FILE * chainoutput = NULL;

  FILE *statfile = NULL;
  FILE *propstatfile = NULL;
  FILE *tempfile = NULL;
  char statfilename[256];
  char propstatfilename[256];
  char tempfilename[256];
  if(MPIrank == 0){
    if (LALInferenceGetProcParamVal(runState->commandLine, "--tempVerbose")) {
      sprintf(tempfilename,"PTMCMC.tempswaps.%u",randomseed);
      tempfile = fopen(tempfilename, "w");
      fprintf(tempfile, "cycle\tlog(chain_swap)\ttemp_low\ttemp_high\n");  // Print header for temp stat file
    }
  }

  if (adaptationOn && LALInferenceGetProcParamVal(runState->commandLine, "--adaptVerbose")) {
    sprintf(statfilename,"PTMCMC.statistics.%u.%2.2d",randomseed,MPIrank);
    statfile = fopen(statfilename, "w");

    /* Print header for adaptation stats */
    fprintf(statfile,"cycle\ts_gamma");
    ptr=runState->currentParams->head;
    while(ptr!=NULL) {
      if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
        fprintf(statfile, "\tsigma_%s", LALInferenceTranslateInternalToExternalParamName(ptr->name));
      }
      ptr=ptr->next;
    }
    ptr=runState->currentParams->head;
    while(ptr!=NULL) {
      if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
        fprintf(statfile, "\tPaccept_%s", LALInferenceTranslateInternalToExternalParamName(ptr->name));
      }
      ptr=ptr->next;
    }
    fprintf(statfile,"\n");
  }

  if (LALInferenceGetProcParamVal(runState->commandLine, "--propVerbose")) {
    sprintf(propstatfilename,"PTMCMC.propstats.%u.%2.2d",randomseed,MPIrank);
    propstatfile = fopen(propstatfilename, "w");
  }

  chainoutput = LALInferencePrintPTMCMCHeader(runState);
  if (MPIrank == 0) {
    LALInferencePrintPTMCMCInjectionSample(runState);
  }

  /* Print run details */
  if (MPIrank == 0) {
    printf("\nParallel Behavior:\n");
    if (adaptationOn)
      printf(" Adapting with decay power %i for %i iterations after max log(L) increases by nParams/2 (%1.2f).\n", adaptTau, adaptLength, (double)nPar/2.0);
    else
      printf(" Adaptation off.\n");
    if (annealingOn)
      printf(" Annealing linearly for %i effective samples.\n", annealLength);
    else
      printf(" Annealing off.\n");
    if (Neff != Niter)
      printf(" Collecting %i effective samples.\n", Neff);
  }


  if (MPIrank == 0) {
    printf("\nPTMCMCAlgorithm(); starting parameter values:\n");
    LALInferencePrintVariables(runState->currentParams);
    printf(" MCMC iteration: 0\t");
    printf("%f\t", runState->currentLikelihood - nullLikelihood);
    printf("\n");

    /* Print to file the contents of ->freqModelhPlus->data->length. */
    ppt = LALInferenceGetProcParamVal(runState->commandLine, "--data-dump");
    if (ppt) {
      LALInferenceDataDump(runState);
    }
  }

  // iterate:
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  for (i=1; i<=Niter; i++) {

    LALInferenceSetVariable(runState->proposalArgs, "acceptanceCount", &(acceptanceCount));

    if (adaptationOn)
      LALInferenceAdaptation(runState, i);

    if (runPhase < 2) {
      //ACL calculation during parallel tempering
      if (i % (100*Nskip) == 0) {
        adapting = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adapting"));

        MPI_Gather(&adapting,1,MPI_INT,intVec,1,MPI_INT,0,MPI_COMM_WORLD);
        if (MPIrank==0) {
          adapting=0;
          for (p=0; p<nChain; p++) {
            if (intVec[p]>0){
              adapting=1;
              break;
            }
          }
        }
        MPI_Bcast(&adapting, 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* Check if cold chain ACL has been calculated */
        if (!adapting) {
          acl = *((INT4*) LALInferenceGetVariable(runState->algorithmParams, "acl"));

          goodACL = *((INT4*) LALInferenceGetVariable(runState->algorithmParams, "goodACL"));
          if (!goodACL) {
            oldACL = acl;
            updateMaxAutoCorrLen(runState, i);
            acl = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "acl");
            if (acl != Niter && acl<=oldACL) {
              goodACL=1;
              LALInferenceSetVariable(runState->algorithmParams, "goodACL", &goodACL);
            }
          }

          MPI_Gather(&goodACL,1,MPI_INT,intVec,1,MPI_INT,0,MPI_COMM_WORLD);
          if (MPIrank==0) {
            goodACL=1;
            for (p=0; p<nChain; p++) {
              if (intVec[p]==0){
                goodACL=0;
                break;
              }
            }
          }
          MPI_Bcast(&goodACL, 1, MPI_INT, 0, MPI_COMM_WORLD);

          if (goodACL) {
            adaptStart = *((INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptStart"));
            iEffStart = adaptStart+adaptLength;
            iEff = (i - iEffStart)/acl;
            MPI_Bcast(&iEff, 1, MPI_INT, 0, MPI_COMM_WORLD);

            /* Check ACL at quarter and half way through to limit effect of over-estimation of ACL early in the run */
            if (!quarterAclChecked) {
              if ((runPhase==0 && iEff >= Neff/4) || (runPhase==1 && iEff >= annealStart/4)) {
                updateMaxAutoCorrLen(runState, i);
                acl = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "acl");
                iEff = (i - iEffStart)/acl;
                MPI_Bcast(&iEff, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (iEff >= Neff/4) quarterAclChecked=1;
              }
            } else if (!halfAclChecked) {
              if ((runPhase==0 && iEff >= Neff/2) || (runPhase==1 && iEff >= annealStart/2)) {
                updateMaxAutoCorrLen(runState, i);
                acl = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "acl");
                iEff = (i - iEffStart)/acl;
                MPI_Bcast(&iEff, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (iEff >= Neff/2) quarterAclChecked=1;
              }
            }

            if ( (runPhase==0 && iEff >= Neff) || (runPhase==1 && iEff >= annealStart) ) {
              /* Double check ACL before changing phase */
              updateMaxAutoCorrLen(runState, i);
              acl = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "acl");
              iEff = (i - iEffStart)/acl;
              MPI_Bcast(&iEff, 1, MPI_INT, 0, MPI_COMM_WORLD);

              if (runPhase==0 && iEff >= Neff) {
                if (MPIrank==0)
                  fprintf(stdout,"Chain %i has %i effective samples. Stopping...\n", MPIrank, iEff);
                break;                                 // Sampling is done!
              } else if (runPhase==1 && iEff >= annealStart) {
                /* Broadcast the cold chain ACL from parallel tempering */
                PTacl = acl;
                MPI_Bcast(&PTacl, 1, MPI_INT, 0, MPI_COMM_WORLD);
                runPhase += 1;
                annealStartIter=i;
                runState->proposal = &LALInferencePostPTProposal;
                if(MPIrank==0)
                  printf("Starting to anneal at iteration %i.\n",i);

                /* Share DE buffer from cold chain */
                if (!LALInferenceGetProcParamVal(runState->commandLine, "--noDifferentialEvolution"))
                  BcastDifferentialEvolutionPoints(runState, 0);

                /* Force chains to re-adapt */
                if (adaptationOn)
                  LALInferenceAdaptationRestart(runState, i);

                /* Calculate speed of annealing based on ACL */
                for (t=0; t<nChain; ++t) {
                  annealDecay[t] = (tempLadder[t]-1.0)/(REAL8)(annealLength*PTacl);
                }

                /* Reset effective sample size and ACL */
                iEff=0;
                acl = PTacl;
                LALInferenceSetVariable(runState->algorithmParams, "acl", &acl);
              } //else if (runPhase==1 && iEff >= annealStart)
            } //if ( (runPhase==0 && iEff >= Neff) || (runPhase==1 && iEff >= annealStart) )
          } //if (goodACL)
        } //if (!adapting)
      } //if (i % (100*Nskip) == 0)
    } //if (runPhase < 2)


    if (runPhase==2) {
    // Annealing phase
      if (i-annealStartIter < PTacl*annealLength) {
        for (t=0;t<nChain; ++t) {
          tempLadder[t] = tempLadder[t] - annealDecay[t];
          LALInferenceSetVariable(runState->proposalArgs, "temperature", &(tempLadder[MPIrank]));
        }
      } else {
        runPhase += 1;
        if (MPIrank==0)
          printf(" Single-chain sampling starting at iteration %i.\n", i);
      }
    } //if (runState==2)


    if (runPhase==3) {
    //Post-annealing single-chain sampling
      adapting = *((INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adapting"));
      adaptStart = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptStart");
      iEffStart = adaptStart+adaptLength;
      if (!adapting) {
        iEff = (i-iEffStart)/acl;
        if (iEff >= Neff/nChain) {
          /* Double check ACL before ending */
          updateMaxAutoCorrLen(runState, i);
          acl = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "acl");
          iEff = (i-iEffStart)/acl;
          if (iEff >= Neff/nChain) {
            fprintf(stdout,"Chain %i has %i effective samples. Stopping...\n", MPIrank, iEff);
            break;                                 // Sampling is done for this chain!
          }
        }
      }
    } //if (runPhase==3)

    runState->evolve(runState); //evolve the chain with the parameters TcurrentParams[t] at temperature tempLadder[t]
    acceptanceCount = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "acceptanceCount");

    if (i==1){
      ppt = LALInferenceGetProcParamVal(runState->commandLine, "--propVerbose");
      if (ppt) {
        // Make sure numbers are initialized!!!
        LALInferenceProposalStatistics *propStat;
        LALInferenceVariableItem *this;
        this = runState->proposalStats->head;
        while(this){
          propStat = (LALInferenceProposalStatistics *)this->value;
          propStat->accepted = 0;
          propStat->proposed = 0;
          this = this->next;
        }
        fprintf(propstatfile, "cycle\t");
        LALInferencePrintProposalStatsHeader(propstatfile, runState->proposalStats);
        fflush(propstatfile);
      }
    }

    if ((i % Nskip) == 0) {
      if (!LALInferenceGetProcParamVal(runState->commandLine, "--noDifferentialEvolution") && !LALInferenceGetProcParamVal(runState->commandLine, "--nodifferentialevolution")) {
        accumulateDifferentialEvolutionSample(runState);
      }

      if (LALInferenceGetProcParamVal(runState->commandLine, "--kDTree") || LALInferenceGetProcParamVal(runState->commandLine, "--kdtree")) {
        accumulateKDTreeSample(runState);
      }

      fseek(chainoutput, 0L, SEEK_END);
      fprintf(chainoutput, "%d\t%f\t%f\t", i,(runState->currentLikelihood - nullLikelihood)+runState->currentPrior,runState->currentPrior);
      LALInferencePrintSampleNonFixed(chainoutput,runState->currentParams);
      fprintf(chainoutput,"%f\t",runState->currentLikelihood - nullLikelihood);

      LALInferenceIFOData *headIFO = runState->data;
      while (headIFO != NULL) {
        fprintf(chainoutput, "%f\t", headIFO->acceptedloglikelihood - headIFO->nullloglikelihood);
        headIFO = headIFO->next;
      }

      fprintf(chainoutput,"\n");
      fflush(chainoutput);

      if (adaptationOn == 1) {
        if (LALInferenceGetProcParamVal(runState->commandLine, "--adaptVerbose")) {
          fseek(statfile, 0L, SEEK_END);
          fprintf(statfile,"%d\t",i);

          if (LALInferenceGetProcParamVal(runState->commandLine, "--adaptVerbose")){
            s_gamma = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "s_gamma");
            fprintf(statfile,"%f\t",s_gamma);
            for (p=0; p<nPar; ++p) {
              fprintf(statfile,"%g\t",sigmas->data[p]);
            }
            for (p=0; p<nPar; ++p) {
              fprintf(statfile,"%f\t",PacceptCount->data[p]/( PproposeCount->data[p]==0 ? 1.0 : PproposeCount->data[p] ));
            }
          }
          fprintf(statfile,"\n");
          fflush(statfile);
        }
      }

      if (LALInferenceGetProcParamVal(runState->commandLine, "--propVerbose")){
        fprintf(propstatfile, "%d\t", i);
        LALInferencePrintProposalStats(propstatfile,runState->proposalStats);
        fflush(propstatfile);
      }
    }

    if (runPhase < 3) {
      if ((i % Tskip) == 0) {
        ptr=runState->currentParams->head;
        p=0;
        while(ptr!=NULL) {
          if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
            parameters->data[p]=*(REAL8 *)ptr->value;

            p++;
          }
          ptr=ptr->next;
        }

        MPI_Gather(&(runState->currentLikelihood), 1, MPI_DOUBLE, TcurrentLikelihood, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&acceptanceCount, 1, MPI_INT, acceptanceCountLadder, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(parameters->data,nPar,MPI_DOUBLE,parametersVec,nPar,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (MPIrank == 0) { //swap parameters and likelihood between chains
          if(LALInferenceGetProcParamVal(runState->commandLine, "--oldPT")) {
            for(lowerRank=0;lowerRank<nChain-1;lowerRank++) { //swap parameters and likelihood between chains
              for(upperRank=lowerRank+1;upperRank<nChain;upperRank++) {
                LALInferencePTswap(runState, TcurrentLikelihood, parametersVec, tempLadder, lowerRank, upperRank, i, tempfile);
              } //for(upperRank=lowerRank+1;upperRank<nChain;upperRank++)
            } //for(lowerRank=0;lowerRank<nChain-1;lowerRank++)
          } else {
            for(swapAttempt=0; swapAttempt<tempSwaps; ++swapAttempt) {
              lowerRank = gsl_rng_uniform_int(runState->GSLrandom, nChain-1);
              upperRank = lowerRank+1;
              LALInferencePTswap(runState, TcurrentLikelihood, parametersVec, tempLadder, lowerRank, upperRank, i, tempfile);
            } //for(swapAttempt=0; swapAttempt<50; ++swapAttempt)
          } //else
        } //if (MPIrank == 0)

        MPI_Scatter(parametersVec,nPar,MPI_DOUBLE,parameters->data,nPar,MPI_DOUBLE,0, MPI_COMM_WORLD);
        MPI_Scatter(TcurrentLikelihood, 1, MPI_DOUBLE, &(runState->currentLikelihood), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(acceptanceCountLadder, 1, MPI_INT, &acceptanceCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

        ptr=runState->currentParams->head;
        p=0;
        while(ptr!=NULL) {
          if (ptr->vary != LALINFERENCE_PARAM_FIXED) {
            memcpy(ptr->value,&(parameters->data[p]),LALInferenceTypeSize[ptr->type]);
            p++;
          }
          ptr=ptr->next;
        }

        MPI_Barrier(MPI_COMM_WORLD);
      }// if ((i % Tskip) == 0)
    }// if (runPhase < 3)
  }// for (i=1; i<=Niter; i++)

  MPI_Barrier(MPI_COMM_WORLD);

  fclose(chainoutput);

  if(MPIrank == 0){
    if (LALInferenceGetProcParamVal(runState->commandLine, "--adaptVerbose")) {
      fclose(statfile);
    }
    if (LALInferenceGetProcParamVal(runState->commandLine, "--tempVerbose")) {
      fclose(tempfile);
    }
    if (LALInferenceGetProcParamVal(runState->commandLine, "--propVerbose")) {
      fclose(propstatfile);
    }
  }

  free(tempLadder);
  free(acceptanceCountLadder);
  free(annealDecay);
  free(parametersVec);

  if (MPIrank == 0) {
    free(TcurrentLikelihood);
  }
}


void PTMCMCOneStep(LALInferenceRunState *runState)
  // Metropolis-Hastings sampler.
{
  int MPIrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);
  REAL8 logPriorCurrent, logPriorProposed;
  REAL8 logLikelihoodCurrent, logLikelihoodProposed;
  LALInferenceVariables proposedParams;
  REAL8 logProposalRatio = 0.0;  // = log(P(backward)/P(forward))
  REAL8 logAcceptanceProbability;
  REAL8 temperature;
  REAL8 targetAcceptance = 0.234;
  REAL8 acceptanceRate = 0.0;
  INT4 acceptanceCount;
  INT4 accepted = 0;
  const char *currentProposalName;
  LALInferenceProposalStatistics *propStat;

  // current values:
  logPriorCurrent      = runState->currentPrior;
  logLikelihoodCurrent = runState->currentLikelihood;

  temperature = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "temperature");
  acceptanceCount = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "acceptanceCount");

  // generate proposal:
  proposedParams.head = NULL;
  proposedParams.dimension = 0;
  runState->proposal(runState, &proposedParams);
  if (LALInferenceCheckVariable(runState->proposalArgs, "logProposalRatio"))
    logProposalRatio = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "logProposalRatio");

  // compute prior & likelihood:
  logPriorProposed = runState->prior(runState, &proposedParams);
  if (logPriorProposed > -DBL_MAX)
    logLikelihoodProposed = runState->likelihood(&proposedParams, runState->data, runState->template);
  else
    logLikelihoodProposed = -DBL_MAX;

  //REAL8 nullLikelihood = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "nullLikelihood");
  //printf("%10.10f\t%10.10f\t%10.10f\n", logPriorProposed-logPriorCurrent, logLikelihoodProposed-nullLikelihood, logProposalRatio);
  //LALInferencePrintVariables(&proposedParams);
  
  // determine acceptance probability:
  logAcceptanceProbability = (1.0/temperature)*(logLikelihoodProposed - logLikelihoodCurrent)
    + (logPriorProposed - logPriorCurrent)
    + logProposalRatio;

  // accept/reject:
  if ((logAcceptanceProbability > 0)
      || (log(gsl_rng_uniform(runState->GSLrandom)) < logAcceptanceProbability)) {   //accept
    LALInferenceCopyVariables(&proposedParams, runState->currentParams);
    runState->currentLikelihood = logLikelihoodProposed;
    LALInferenceIFOData *headData = runState->data;
    while (headData != NULL) {
      headData->acceptedloglikelihood = headData->loglikelihood;
      headData = headData->next;
    }
    runState->currentPrior = logPriorProposed;
    acceptanceCount++;
    accepted = 1;
    LALInferenceSetVariable(runState->proposalArgs, "acceptanceCount", &acceptanceCount);

  }

  LALInferenceUpdateAdaptiveJumps(runState, accepted, targetAcceptance);
  LALInferenceDestroyVariables(&proposedParams);
}


//-----------------------------------------
// temperature swap routine:
//-----------------------------------------
void LALInferencePTswap(LALInferenceRunState *runState,
                        double *TcurrentLikelihood,
                        REAL8 *parametersVec,
                        REAL8 *tempLadder,
                        int lowerRank,
                        int upperRank,
                        int i,
                        FILE *tempfile)
{
  REAL8 logChainSwap;
  REAL8 dummyR8;
  INT4 p;
  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);

  logChainSwap = (1.0/tempLadder[lowerRank]-1.0/tempLadder[upperRank]) * (TcurrentLikelihood[upperRank]-TcurrentLikelihood[lowerRank]);

  if ((logChainSwap > 0) || (log(gsl_rng_uniform(runState->GSLrandom)) < logChainSwap )) { //Then swap...
    // Check if --tempVerbose was specified
    if (tempfile != NULL) {
      fprintf(tempfile,"%d\t%f\t%f\t%f\n",i,logChainSwap,tempLadder[lowerRank],tempLadder[upperRank]);
      fflush(tempfile);
    }
    for (p=0; p<(nPar); ++p){
      dummyR8=parametersVec[p+nPar*upperRank];
      parametersVec[p+nPar*upperRank]=parametersVec[p+nPar*lowerRank];
      parametersVec[p+nPar*lowerRank]=dummyR8;
    }
  dummyR8 = TcurrentLikelihood[upperRank];
  TcurrentLikelihood[upperRank] = TcurrentLikelihood[lowerRank];
  TcurrentLikelihood[lowerRank] = dummyR8;
  }
}


//-----------------------------------------
// Adaptation:
//-----------------------------------------
void LALInferenceAdaptation(LALInferenceRunState *runState, INT4 cycle)
{
  INT4 MPIrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);

  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);
  INT4 adapting = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adapting");
  INT4 adaptStart = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptStart");
  INT4 adaptLength = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptLength");
  REAL8 logLAtAdaptStart = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "logLAtAdaptStart");

  /* if maximum logL has increased by more than nParam/2, restart it */
  if (runState->currentLikelihood > logLAtAdaptStart+(REAL8)nPar/2) {
    if (!adapting)
      fprintf(stdout,"Turning on adaptation for chain %u at iteration %u.\n",MPIrank,cycle);
    LALInferenceAdaptationRestart(runState, cycle);
  } else if (adapting) {
    /* Turn off adaption after adaptLength steps without restarting */
    if ((cycle-adaptStart) > adaptLength) {
      adapting = 0;  //turn off adaptation
      LALInferenceSetVariable(runState->proposalArgs, "adapting", &adapting);
      LALInferenceRemoveVariable(runState->proposalArgs,"s_gamma");
      fprintf(stdout,"Ending adaptation for chain %u at iteration %u.\n",MPIrank,cycle);

    /* Else set adaptation envelope */
    } else {
      LALInferenceAdaptationEnvelope(runState, cycle);
    }
  }
}


//-----------------------------------------
// Restart adaptation:
//-----------------------------------------
void LALInferenceAdaptationRestart(LALInferenceRunState *runState, INT4 cycle)
{
  INT4 Niter = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Niter");
  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);
  REAL8Vector *PacceptCount = NULL;
  REAL8Vector *PproposeCount = NULL;
  INT4 adapting=1;
  INT4 p=0;
  INT4 goodACL=0;

  for (p=0; p<nPar; ++p) {
    PacceptCount = *((REAL8Vector **)LALInferenceGetVariable(runState->proposalArgs, "PacceptCount"));
    PproposeCount = *((REAL8Vector **)LALInferenceGetVariable(runState->proposalArgs, "PproposeCount"));
    PacceptCount->data[p] =0;
    PproposeCount->data[p]=0;
  }

  LALInferenceSetVariable(runState->proposalArgs, "adapting", &adapting);
  LALInferenceSetVariable(runState->proposalArgs, "adaptStart", &cycle);
  LALInferenceSetVariable(runState->proposalArgs, "logLAtAdaptStart", &(runState->currentLikelihood));
  LALInferenceSetVariable(runState->algorithmParams, "acl", &Niter);
  LALInferenceSetVariable(runState->algorithmParams, "goodACL", &goodACL);
  LALInferenceAdaptationEnvelope(runState, cycle);
}


//-----------------------------------------
// Adaptation envelope function:
//-----------------------------------------
void LALInferenceAdaptationEnvelope(LALInferenceRunState *runState, INT4 cycle)
{
  INT4 adaptStart = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptStart");
  INT4 adaptTau = *(INT4 *)LALInferenceGetVariable(runState->proposalArgs, "adaptTau");
  INT4 adaptLength = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptLength");
  INT4 adaptResetBuffer = *(INT4*) LALInferenceGetVariable(runState->proposalArgs, "adaptResetBuffer");
  REAL8 s_gamma = 0.0;

  if (cycle-adaptStart <= adaptResetBuffer) {
    s_gamma=(((REAL8)cycle-(REAL8)adaptStart)/(REAL8)adaptResetBuffer)*(((REAL8)cycle-(REAL8)adaptStart)/(REAL8)(adaptResetBuffer));
  } else if (cycle-adaptStart < adaptLength) {
    s_gamma=10.0*exp(-(1.0/adaptTau)*log((REAL8)(cycle-adaptStart)))-1;
  } else {
    s_gamma=0.0;
  }

  if (LALInferenceCheckVariable(runState->proposalArgs, "s_gamma"))
    LALInferenceSetVariable(runState->proposalArgs, "s_gamma", &s_gamma);
  else
    LALInferenceAddVariable(runState->proposalArgs, "s_gamma", &s_gamma, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_LINEAR);
}


//-----------------------------------------
// file output routines:
//-----------------------------------------
FILE *LALInferencePrintPTMCMCHeader(LALInferenceRunState *runState) {
  ProcessParamsTable *ppt;
  char *outFileName = NULL;
  FILE *chainoutput = NULL;
  UINT4 randomseed = *(UINT4*) LALInferenceGetVariable(runState->algorithmParams,"random_seed");
  int MPIrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);

  ppt = LALInferenceGetProcParamVal(runState->commandLine, "--outfile");
  if (ppt) {
    outFileName = (char*)XLALCalloc(strlen(ppt->value)+255,sizeof(char*));
    sprintf(outFileName,"%s.%2.2d",ppt->value,MPIrank);
  } else {
    outFileName = (char*)XLALCalloc(255,sizeof(char*));
    sprintf(outFileName,"PTMCMC.output.%u.%2.2d",randomseed,MPIrank);
  }

  chainoutput = fopen(outFileName,"w");
  if(chainoutput == NULL){
    XLALErrorHandler = XLALExitErrorHandler;
    XLALPrintError("Output file error. Please check that the specified path exists. (in %s, line %d)\n",__FILE__, __LINE__);
    XLAL_ERROR_NULL(XLAL_EIO);
  }
  
  LALInferencePrintPTMCMCHeaderFile(runState, chainoutput);

  fclose(chainoutput);

  chainoutput = fopen(outFileName, "a");
  if (chainoutput == NULL) {
    XLALErrorHandler = XLALExitErrorHandler;
    XLALPrintError("Output file error. Please check that the specified path exists. (in %s, line %d)\n",__FILE__, __LINE__);
    XLAL_ERROR_NULL(XLAL_EIO);
  }

  XLALFree(outFileName);

  return chainoutput;
}

void LALInferencePrintPTMCMCHeaderFile(LALInferenceRunState *runState, FILE *chainoutput)
{
  int MPIrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);
  UINT4 randomseed = *(UINT4*) LALInferenceGetVariable(runState->algorithmParams,"random_seed");
  INT4 nPar = LALInferenceGetVariableDimensionNonFixed(runState->currentParams);
  INT4 Niter = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "Niter");
  REAL8 nullLikelihood = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "nullLikelihood");
  INT4 nChain = *(INT4*) LALInferenceGetVariable(runState->algorithmParams, "nChain");
  REAL8 temperature = *(REAL8*) LALInferenceGetVariable(runState->proposalArgs, "temperature");
  REAL8 SampleRate=4096.0; //default value of the sample rate from LALInferenceReadData()
  UINT4 nIFO=0;
  LALInferenceIFOData *ifodata1=runState->data;

  while(ifodata1){
    nIFO++;
    ifodata1=ifodata1->next;
  }

  int waveform = 0;
  if(LALInferenceCheckVariable(runState->currentParams,"LAL_APPROXIMANT")) waveform= *(INT4 *)LALInferenceGetVariable(runState->currentParams,"LAL_APPROXIMANT");
  double pnorder = 0.0;
  if(LALInferenceCheckVariable(runState->currentParams,"LAL_PNORDER")) pnorder = ((double)(*(INT4 *)LALInferenceGetVariable(runState->currentParams,"LAL_PNORDER")))/2.0;

  char *str;
  str = LALInferencePrintCommandLine(runState->commandLine);

  REAL8 networkSNR=0.0;
  ifodata1=runState->data;
  while(ifodata1){
    networkSNR+=ifodata1->SNR*ifodata1->SNR;
    ifodata1=ifodata1->next;
  }
  networkSNR=sqrt(networkSNR);

  if(LALInferenceGetProcParamVal(runState->commandLine,"--srate")) SampleRate=atof(LALInferenceGetProcParamVal(runState->commandLine,"--srate")->value);


    fprintf(chainoutput, "  LALInference version:%s,%s,%s,%s,%s\n", LALAPPS_VCS_ID,LALAPPS_VCS_DATE,LALAPPS_VCS_BRANCH,LALAPPS_VCS_AUTHOR,LALAPPS_VCS_STATUS);
    fprintf(chainoutput,"  %s\n",str);
    fprintf(chainoutput, "%10s  %10s  %6s  %20s  %6s %8s   %6s  %10s  %12s  %9s  %9s  %8s\n",
        "nIter","Nburn","seed","null likelihood","Ndet","nCorr","nTemps","Tchain","Network SNR","Waveform","pN order","Npar");
    fprintf(chainoutput, "%10d  %10d  %u  %20.10lf  %6d %8d   %6d%12.1f%14.6f  %9i  %9.1f  %8i\n",
        Niter,0,randomseed,nullLikelihood,nIFO,0,nChain,temperature,networkSNR,waveform,(double)pnorder,nPar);
    fprintf(chainoutput, "\n%16s  %16s  %10s  %10s  %10s  %10s  %20s  %15s  %12s  %12s  %12s\n",
        "Detector","SNR","f_low","f_high","before tc","after tc","Sample start (GPS)","Sample length","Sample rate","Sample size","FT size");
    ifodata1=runState->data;
    while(ifodata1){
      fprintf(chainoutput, "%16s  %16.8lf  %10.2lf  %10.2lf  %10.2lf  %10.2lf  %20.8lf  %15.7lf  %.1f  %12d  %12d\n",
          ifodata1->detector->frDetector.name,ifodata1->SNR,ifodata1->fLow,ifodata1->fHigh,atof(LALInferenceGetProcParamVal(runState->commandLine,"--seglen")->value)-2.0,2.00,
          XLALGPSGetREAL8(&(ifodata1->epoch)),atof(LALInferenceGetProcParamVal(runState->commandLine,"--seglen")->value),SampleRate,
          (int)(atof(LALInferenceGetProcParamVal(runState->commandLine,"--seglen")->value)*SampleRate),
          (int)(atof(LALInferenceGetProcParamVal(runState->commandLine,"--seglen")->value)*SampleRate));
      ifodata1=ifodata1->next;
    }
    fprintf(chainoutput, "\n\n%31s\n","");
    fprintf(chainoutput, "cycle\tlogpost\tlogprior\t");
    LALInferenceFprintParameterNonFixedHeaders(chainoutput, runState->currentParams);
    fprintf(chainoutput, "logl\t");
    LALInferenceIFOData *headIFO = runState->data;
    while (headIFO != NULL) {
      fprintf(chainoutput, "logl");
      fprintf(chainoutput, "%s",headIFO->name);
      fprintf(chainoutput, "\t");
      headIFO = headIFO->next;
    }
    fprintf(chainoutput,"\n");
    fprintf(chainoutput, "%d\t%f\t%f\t", 0,(runState->currentLikelihood - nullLikelihood)+runState->currentPrior, runState->currentPrior);
    LALInferencePrintSampleNonFixed(chainoutput,runState->currentParams);
    fprintf(chainoutput,"%f\t",runState->currentLikelihood - nullLikelihood);
    headIFO = runState->data;
    while (headIFO != NULL) {
      fprintf(chainoutput, "%f\t", headIFO->acceptedloglikelihood - headIFO->nullloglikelihood);
      headIFO = headIFO->next;
    }

    fprintf(chainoutput,"\n");
}

static void setIFOAcceptedLikelihoods(LALInferenceRunState *runState) {
  LALInferenceIFOData *data = runState->data;
  LALInferenceIFOData *ifo = NULL;

  for (ifo = data; ifo != NULL; ifo = ifo->next) {
    ifo->acceptedloglikelihood = ifo->loglikelihood;
  }
}

void LALInferencePrintPTMCMCInjectionSample(LALInferenceRunState *runState) {
  ProcessParamsTable *ppt;

  ppt = LALInferenceGetProcParamVal(runState->commandLine, "--inj");
  if (ppt) {
    ProcessParamsTable *ppt2 = LALInferenceGetProcParamVal(runState->commandLine, "--outfile");
    UINT4 randomseed = *(UINT4*) LALInferenceGetVariable(runState->algorithmParams,"random_seed");
    FILE *out = NULL;
    char *fname = NULL;
    LALInferenceVariables *saveParams = NULL;

    saveParams = (LALInferenceVariables *)XLALCalloc(sizeof(LALInferenceVariables), 1);

    if (ppt2) {
      fname = (char *) XLALCalloc((strlen(ppt2->value)+255)*sizeof(char), 1);
      sprintf(fname, "%s.injection", ppt2->value);
    } else {
      fname = (char *) XLALCalloc(255*sizeof(char), 1);
      sprintf(fname, "PTMCMC.output.%u.injection", randomseed);
    }
    out = fopen(fname, "w");

    LALInferenceCopyVariables(runState->currentParams, saveParams);

    SimInspiralTable *injTable = NULL;
    SimInspiralTable *theEventTable = NULL;

    SimInspiralTableFromLIGOLw(&injTable,ppt->value,0,0);
    
    ppt2 = LALInferenceGetProcParamVal(runState->commandLine, "--event");
    if (ppt2) {
      UINT4 event = atoi(ppt2->value);
      UINT4 i;
      theEventTable = injTable;
      for (i = 0; i < event; i++) {
        theEventTable = theEventTable->next;
      }
      theEventTable->next = NULL;
    } else {
      theEventTable=injTable;
      theEventTable->next = NULL;
    }

    REAL8 m1 = theEventTable->mass1;
    REAL8 m2 = theEventTable->mass2;
    REAL8 q = m2/m1;
    REAL8 eta = m1*m2/(m1+m2)/(m1+m2);

    if (q > 1.0) q = 1.0/q;

    REAL8 sx = theEventTable->spin1x;
    REAL8 sy = theEventTable->spin1y;
    REAL8 sz = theEventTable->spin1z;

    REAL8 a_spin1 = sqrt(sx*sx + sy*sy + sz*sz);
    
    REAL8 theta_spin1, phi_spin1;
    if (a_spin1 == 0.0) {
      theta_spin1 = 0.0;
      phi_spin1 = 0.0;
    } else {
      theta_spin1 = acos(sz / a_spin1);
      phi_spin1 = atan2(sy, sx);
      if (phi_spin1 < 0.0) phi_spin1 += 2.0*M_PI;
    }

    sx = theEventTable->spin2x;
    sy = theEventTable->spin2y;
    sz = theEventTable->spin2z;
    
    REAL8 a_spin2 = sqrt(sx*sx + sy*sy + sz*sz), theta_spin2, phi_spin2;
    if (a_spin2 == 0.0) {
      theta_spin2 = 0.0;
      phi_spin2 = 0.0;
    } else {
      theta_spin2 = acos(sz / a_spin2);
      phi_spin2 = atan2(sy, sx);
      if (phi_spin2 < 0.0) phi_spin2 += 2.0*M_PI;
    }

    REAL8 psi = theEventTable->polarization;
    if (psi>=M_PI) psi -= M_PI;

    REAL8 injGPSTime = XLALGPSGetREAL8(&(theEventTable->geocent_end_time));

    REAL8 chirpmass = theEventTable->mchirp;

    REAL8 dist = theEventTable->distance;
    REAL8 inclination = theEventTable->inclination;
    REAL8 phase = theEventTable->coa_phase;
    REAL8 dec = theEventTable->latitude;
    REAL8 ra = theEventTable->longitude;

    LALInferenceSetVariable(runState->currentParams, "chirpmass", &chirpmass);
    if (LALInferenceCheckVariable(runState->currentParams, "asym_massratio")) {
      LALInferenceSetVariable(runState->currentParams, "asym_massratio", &q);
    } else if (LALInferenceCheckVariable(runState->currentParams, "massratio")) {
      LALInferenceSetVariable(runState->currentParams, "massratio", &eta);
    } else {
      /* Restore state, cleanup, and throw error */
      LALInferenceCopyVariables(saveParams, runState->currentParams);
      XLALFree(fname);
      LALInferenceDestroyVariables(saveParams);
      XLALFree(saveParams);
      XLAL_ERROR_VOID(XLAL_EINVAL, "unknown mass ratio parameter name (allowed are 'massratio' or 'asym_massratio')");
    }
    LALInferenceSetVariable(runState->currentParams, "time", &injGPSTime);
    LALInferenceSetVariable(runState->currentParams, "distance", &dist);
    LALInferenceSetVariable(runState->currentParams, "inclination", &inclination);
    LALInferenceSetVariable(runState->currentParams, "polarisation", &(psi));
    LALInferenceSetVariable(runState->currentParams, "phase", &phase);
    LALInferenceSetVariable(runState->currentParams, "declination", &dec);
    LALInferenceSetVariable(runState->currentParams, "rightascension", &ra);
    if (LALInferenceCheckVariable(runState->currentParams, "a_spin1")) {
      LALInferenceSetVariable(runState->currentParams, "a_spin1", &a_spin1);
    }
    if (LALInferenceCheckVariable(runState->currentParams, "theta_spin1")) {
      LALInferenceSetVariable(runState->currentParams, "theta_spin1", &theta_spin1);
    }
    if (LALInferenceCheckVariable(runState->currentParams, "phi_spin1")) {
      LALInferenceSetVariable(runState->currentParams, "phi_spin1", &phi_spin1);
    }
    if (LALInferenceCheckVariable(runState->currentParams, "a_spin2")) {
      LALInferenceSetVariable(runState->currentParams, "a_spin2", &a_spin2);
    }
    if (LALInferenceCheckVariable(runState->currentParams, "theta_spin2")) {
      LALInferenceSetVariable(runState->currentParams, "theta_spin2", &theta_spin2);
    }
    if (LALInferenceCheckVariable(runState->currentParams, "phi_spin2")) {
      LALInferenceSetVariable(runState->currentParams, "phi_spin2", &phi_spin2);
    }

    runState->currentLikelihood = runState->likelihood(runState->currentParams, runState->data, runState->template);
    runState->currentPrior = runState->prior(runState, runState->currentParams);
    setIFOAcceptedLikelihoods(runState);
    LALInferencePrintPTMCMCHeaderFile(runState, out);
    fclose(out);
    
    LALInferenceCopyVariables(saveParams, runState->currentParams);
    runState->currentLikelihood = runState->likelihood(runState->currentParams, runState->data, runState->template);
    runState->currentPrior = runState->prior(runState, runState->currentParams);
    setIFOAcceptedLikelihoods(runState);    

    XLALFree(fname);
    LALInferenceDestroyVariables(saveParams);
    XLALFree(saveParams);
  }
}

void LALInferenceDataDump(LALInferenceRunState *runState){

  const UINT4 nameLength=256;
  char filename[nameLength];
  FILE *out;
  LALInferenceIFOData *headData = runState->data;
  UINT4 ui;

  while (headData != NULL) {

    snprintf(filename, nameLength, "%s-freqTemplatehPlus.dat", headData->name);
    out = fopen(filename, "w");
    for (ui = 0; ui < headData->freqModelhPlus->data->length; ui++) {
      REAL8 f = headData->freqModelhPlus->deltaF * ui;
      COMPLEX16 d = headData->freqModelhPlus->data->data[ui];

      fprintf(out, "%g %g %g\n", f, creal(d), cimag(d));
    }
    fclose(out);

    snprintf(filename, nameLength, "%s-freqTemplatehCross.dat", headData->name);
    out = fopen(filename, "w");
    for (ui = 0; ui < headData->freqModelhCross->data->length; ui++) {
      REAL8 f = headData->freqModelhCross->deltaF * ui;
      COMPLEX16 d = headData->freqModelhCross->data->data[ui];

      fprintf(out, "%g %g %g\n", f, creal(d), cimag(d));
    }
    fclose(out);

    snprintf(filename, nameLength, "%s-freqTemplateStrain.dat", headData->name);
    out = fopen(filename, "w");
    for (ui = 0; ui < headData->freqModelhCross->data->length; ui++) {
      REAL8 f = headData->freqModelhCross->deltaF * ui;
      COMPLEX16 d;
      d = headData->fPlus * headData->freqModelhPlus->data->data[ui] +
             headData->fCross * headData->freqModelhCross->data->data[ui];

      fprintf(out, "%g %g %g\n", f, creal(d), cimag(d) );
    }
    fclose(out);

    snprintf(filename, nameLength, "%s-timeTemplatehPlus.dat", headData->name);
    out = fopen(filename, "w");
    for (ui = 0; ui < headData->timeModelhPlus->data->length; ui++) {
      REAL8 tt = XLALGPSGetREAL8(&(headData->timeModelhPlus->epoch)) +
        ui * headData->timeModelhPlus->deltaT;
      REAL8 d = headData->timeModelhPlus->data->data[ui];

      fprintf(out, "%.6f %g\n", tt, d);
    }
    fclose(out);

    snprintf(filename, nameLength, "%s-timeTemplatehCross.dat", headData->name);
    out = fopen(filename, "w");
    for (ui = 0; ui < headData->timeModelhCross->data->length; ui++) {
      REAL8 tt = XLALGPSGetREAL8(&(headData->timeModelhCross->epoch)) +
        ui * headData->timeModelhCross->deltaT;
      REAL8 d = headData->timeModelhCross->data->data[ui];

      fprintf(out, "%.6f %g\n", tt, d);
    }
    fclose(out);

    snprintf(filename, nameLength, "%s-timeTemplateStrain.dat", headData->name);
    out = fopen(filename, "w");
    for (ui = 0; ui < headData->timeModelhCross->data->length; ui++) {
      REAL8 tt = XLALGPSGetREAL8(&(headData->timeModelhCross->epoch)) +
        headData->timeshift + ui*headData->timeModelhCross->deltaT;
      REAL8 d = headData->fPlus*headData->timeModelhPlus->data->data[ui] +
        headData->fCross*headData->timeModelhCross->data->data[ui];

      fprintf(out, "%.6f %g\n", tt, d);
    }
    fclose(out);

    headData = headData->next;
  }

}

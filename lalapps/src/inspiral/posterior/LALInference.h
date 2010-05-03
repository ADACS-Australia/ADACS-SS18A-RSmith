/*
 *
 *  LALInference:             Bayesian Followup        
 *  include/LALInference.h:   main header file
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
 * \file LALInference.h
 * \brief Main header file
 */

#ifndef LALInference_h
#define LALInference_h

# include <math.h>
# include <stdio.h>
# include <stdlib.h>

#define VARNAME_MAX 128

# include <lal/LALStdlib.h>
# include <lal/LALConstants.h>
# include <lal/SimulateCoherentGW.h>
# include <lal/GeneratePPNInspiral.h>
# include <lal/LIGOMetadataTables.h>
# include <lal/LALDatatypes.h>
# include <lal/FindChirp.h>
# include <lal/Window.h>

#include <lal/LALDetectors.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/time.h>



#define pi 3.141592653589793
//TODO: use M_PI from math library?


//...other includes

struct tagLALInferenceRunState;
struct tagLALIFOData;

/*Data storage type definitions*/

typedef enum tagVariableType {
  INT4_t, 
  INT8_t, 
  REAL4_t, 
  REAL8_t, 
  COMPLEX8_t, 
  COMPLEX16_t, 
  gslMatrix_t
} VariableType;

typedef enum {
  timeDomain, 
  frequencyDomain
} LALDomain;

extern size_t typeSize[];

//VariableItem should NEVER be accessed directly, only through special
//access functions as defined below.
//Implementation may change from linked list to hashtable for faster access
typedef struct
tagVariableItem
{
  char                    name[VARNAME_MAX];
  void                    *value;
  VariableType            type;
  struct tagVariableItem  *next;
} LALVariableItem;

typedef struct
tagLALVariables
{
  LALVariableItem * head;
  INT4 dimension;
} LALVariables;

void *getVariable(LALVariables * vars, const char * name);
INT4 getVariableDimension(LALVariables *vars);
VariableType getVariableType(LALVariables *vars, int index);
char *getVariableName(LALVariables *vars, int index);
void setVariable(LALVariables * vars, const char * name, void * value);
void addVariable(LALVariables * vars, const char * name, void * value, 
	VariableType type);
void removeVariable(LALVariables *vars,const char *name);
int  checkVariable(LALVariables *vars,const char *name);
void destroyVariables(LALVariables *vars);
void copyVariables(LALVariables *origin, LALVariables *target);
void printVariables(LALVariables *var);
int compareVariables(LALVariables *var1, LALVariables *var2);

//Wrapper for template computation 
//(relies on LAL libraries for implementation) <- could be a #DEFINE ?
//typedef void (LALTemplateFunction) (LALVariables *currentParams, struct tagLALIFOData *data); //Parameter Set is modelParams of LALIFOData
typedef void (LALTemplateFunction) (struct tagLALIFOData *data);


//Jump proposal distribution
//Computes proposedParams based on currentParams and additional variables
//stored as proposalArgs, which could include correlation matrix, etc.,
//as well as forward and reverse proposal probability.
//A jump proposal distribution function could call other jump proposal
//distribution functions with various probabilities to allow for multiple
//jump proposal distributions
typedef void (LALProposalFunction) (struct tagLALInferenceRunState *runState,
	LALVariables *proposedParams);

typedef REAL8 (LALPriorFunction) (struct tagLALInferenceRunState *runState,
	LALVariables *params);

//Likelihood calculator 
//Should take care to perform expensive evaluation of h+ and hx 
//only once if possible, unless necessary because different IFOs 
//have different data lengths or sampling rates 
typedef REAL8 (LALLikelihoodFunction) (LALVariables *currentParams,
        struct tagLALIFOData * data, LALTemplateFunction *template);

//Compute next state along chain; replaces currentParams
typedef void (LALEvolveOneStepFunction) (struct tagLALInferenceRunState *runState);

//Main driver function for a run; will distinguish MCMC from NestedSampling
typedef void (LALAlgorithm) (struct tagLALInferenceRunState *runState);

//Structure containing inference run state 
typedef struct 
tagLALInferenceRunState
{
  ProcessParamsTable        *commandLine;
  LALAlgorithm              *algorithm;
  LALEvolveOneStepFunction  *evolve;
  LALPriorFunction          *prior;
  LALLikelihoodFunction     *likelihood;
  LALProposalFunction       *proposal;
  LALTemplateFunction       *template;
  struct tagLALIFOData      *data;
  LALVariables              *currentParams, 
                            *priorArgs, 
                            *proposalArgs;
  REAL8						currentLikelihood;
  gsl_rng                   *GSLrandom;
} LALInferenceRunState;


LALInferenceRunState *initialize(ProcessParamsTable *commandLine);

struct tagLALIFOData * readData (ProcessParamsTable * commandLine);

void injectSignal(struct tagLALIFOData *IFOdata, ProcessParamsTable *commandLine);

typedef struct
tagLALIFOData
{
  REAL8TimeSeries           *timeData, 
                            *timeModelhPlus, *timeModelhCross;
  COMPLEX16FrequencySeries  *freqData, 
                            *freqModelhPlus, *freqModelhCross;
  LALVariables              *modelParams;
  LALDomain                 modelDomain;
  REAL8FrequencySeries      *oneSidedNoisePowerSpectrum;
  REAL8Window               *window;
  REAL8FFTPlan              *timeToFreqFFTPlan, *freqToTimeFFTPlan;
  REAL8                     fLow, fHigh;	//integration limits;
  LALDetector               *detector;
  struct tagLALIFOData      *next;
} LALIFOData;

/* Returns the element of the process params table with "name" */
ProcessParamsTable *getProcParamVal(ProcessParamsTable *procparams,const char *name);

LALIFOData *ReadData(ProcessParamsTable *commandLine);

void parseCharacterOptionString(char *input, char **strings[], int *n);

ProcessParamsTable *parseCommandLine(int argc, char *argv[]);

REAL8 UndecomposedFreqDomainLogLikelihood(LALVariables *currentParams, LALIFOData *data, 
                              LALTemplateFunction *template);

REAL8 FreqDomainLogLikelihood(LALVariables *currentParams, LALIFOData * data, 
                              LALTemplateFunction *template);
void ComputeFreqDomainResponse(LALVariables *currentParams, LALIFOData * dataPtr, 
                              LALTemplateFunction *template, COMPLEX16Vector *freqWaveform);							  
REAL8 ComputeFrequencyDomainOverlap(LALIFOData * data, 
	COMPLEX16Vector * freqData1, COMPLEX16Vector * freqData2);
void COMPLEX16VectorSubtract(COMPLEX16Vector * out, const COMPLEX16Vector * in1, const COMPLEX16Vector * in2);
								  
REAL8 FreqDomainNullLogLikelihood(LALIFOData * data);

void dumptemplateFreqDomain(LALVariables *currentParams, LALIFOData * data, 
                            LALTemplateFunction *template, char *filename);
void dumptemplateTimeDomain(LALVariables *currentParams, LALIFOData * data, 
                            LALTemplateFunction *template, char *filename);

REAL8 NullLogLikelihood(LALIFOData *data);							  

void executeFT(LALIFOData *IFOdata);
void executeInvFT(LALIFOData *IFOdata);

void die(char *message);
void LALTemplateGeneratePPN(LALIFOData *IFOdata);
void templateStatPhase(LALIFOData *IFOdata);
void templateNullFreqdomain(LALIFOData *IFOdata);
void templateNullTimedomain(LALIFOData *IFOdata);
void templateLAL(LALIFOData *IFOdata);
void template3525TD(LALIFOData *IFOdata);
void templateSineGaussian(LALIFOData *IFOdata);
void templateDampedSinusoid(LALIFOData *IFOdata);
void templateSinc(LALIFOData *IFOdata);
void templateLALSTPN(LALIFOData *IFOdata);


void PTMCMCAlgorithm(struct tagLALInferenceRunState *runState);
void PTMCMCOneStep(LALInferenceRunState *runState);
REAL8 PTUniformLALPrior(LALInferenceRunState *runState, LALVariables *params);
void PTMCMCLALProposal(LALInferenceRunState *runState, LALVariables *proposedParams);

#endif


/*
 *  LALInferenceCBCInit.c:  Bayesian Followup initialisation routines.
 *
 *  Copyright (C) 2012 Vivien Raymond, John Veitch, Salvatore Vitale
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
#include <lal/Date.h>
#include <lal/GenerateInspiral.h>
#include <lal/LALInference.h>
#include <lal/FrequencySeries.h>
#include <lal/Units.h>
#include <lal/StringInput.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <lal/TimeSeries.h>
#include <lal/LALInferencePrior.h>
#include <lal/LALInferenceTemplate.h>
#include <lal/LALInferenceProposal.h>
#include <lal/LALInferenceLikelihood.h>
#include <lal/LALInferenceReadData.h>
#include <lal/LALInferenceInit.h>


static void print_flags_orders_warning(SimInspiralTable *injt, ProcessParamsTable *commline);
static void LALInferenceInitSpinVariables(LALInferenceRunState *state, LALInferenceModel *model);

/* Setup the template generation */
/* Defaults to using LALSimulation */
LALInferenceTemplateFunction LALInferenceInitCBCTemplate(LALInferenceRunState *runState)
{
  char help[]="(--template [LAL,PhenSpin,LALGenerateInspiral,LALSim]\tSpecify template (default LAL)\n";
  ProcessParamsTable *ppt=NULL;
  ProcessParamsTable *commandLine=runState->commandLine;
  /* Print command line arguments if help requested */
  //Help is taken care of in LALInferenceInitCBCVariables
  //ppt=LALInferenceGetProcParamVal(commandLine,"--help");
  //if(ppt)
  //{
  //	fprintf(stdout,"%s",help);
  //	return;
  //}
  /* This is the LAL template generator for inspiral signals */
  LALInferenceTemplateFunction templt = &LALInferenceTemplateXLALSimInspiralChooseWaveform;
  ppt=LALInferenceGetProcParamVal(commandLine,"--template");
  if(ppt) {
    if(!strcmp("LALSim",ppt->value))
      templt=&LALInferenceTemplateXLALSimInspiralChooseWaveform;
    else {
      XLALPrintError("Error: unknown template %s\n",ppt->value);
      XLALPrintError(help);
      XLAL_ERROR_NULL(XLAL_EINVAL);
    }
  }
  else if(LALInferenceGetProcParamVal(commandLine,"--LALSimulation")){
    fprintf(stderr,"Warning: --LALSimulation is deprecated, the LALSimulation package is now the default. To use LALInspiral specify:\n\
                    --template LALGenerateInspiral (for time-domain templates)\n\
                    --template LAL (for frequency-domain templates)\n");
  }
  else if(LALInferenceGetProcParamVal(commandLine,"--roq")){
  runState->templt=&LALInferenceTemplateROQ;
  }
  else {
    fprintf(stdout,"Template function called is \"LALInferenceTemplateXLALSimInspiralChooseWaveform\"\n");
  }
  return templt;
}

/* Setup the glitch model */
void LALInferenceInitGlitchVariables(LALInferenceRunState *runState, LALInferenceVariables *currentParams)
{
  ProcessParamsTable    *commandLine   = runState->commandLine;
  LALInferenceIFOData   *dataPtr       = runState->data;
  LALInferenceVariables *priorArgs     = runState->priorArgs;
  LALInferenceVariables *proposalArgs  = runState->proposalArgs;

  UINT4 i,nifo;
  UINT4 n = (UINT4)dataPtr->timeData->data->length;
  UINT4 gflag  = 1;
  REAL8 gmin   = 0.0;
  REAL8 gmax   = 15.0;

  //over-ride default gmax from command line
  if(LALInferenceGetProcParamVal(commandLine, "--glitchNmax"))
    gmax = (REAL8)atoi(LALInferenceGetProcParamVal(commandLine, "--glitchNmax")->value);

  //count interferometers in network before allocating memory
  //compute imin,imax for each IFO -- may be different
  nifo=0;
  dataPtr = runState->data;
  while (dataPtr != NULL)
  {
    dataPtr = dataPtr->next;
    nifo++;
  }
  dataPtr = runState->data;

  UINT4Vector *gsize  = XLALCreateUINT4Vector(nifo);
  //Meyer?? REAL8Vector *gprior = XLALCreateREAL8Vector((int)gmax+1);

  //Morlet??
  gsl_matrix *mAmp = gsl_matrix_alloc(nifo,(int)(gmax));
  gsl_matrix *mf0  = gsl_matrix_alloc(nifo,(int)(gmax));
  gsl_matrix *mQ   = gsl_matrix_alloc(nifo,(int)(gmax));
  gsl_matrix *mt0  = gsl_matrix_alloc(nifo,(int)(gmax));
  gsl_matrix *mphi = gsl_matrix_alloc(nifo,(int)(gmax));

  double Amin,Amax;
  double Qmin,Qmax;
  double f_min,f_max;
  double tmin,tmax;
  double pmin,pmax;

  REAL8 TwoDeltaToverN = 2.0 * dataPtr->timeData->deltaT / ((double) dataPtr->timeData->data->length);
  Amin = 1.0   / sqrt(TwoDeltaToverN);
  Amax = 100.0 / sqrt(TwoDeltaToverN);

  Qmin = 3.0;
  Qmax = 30.0;
  tmin = 0.0;
  tmax = dataPtr->timeData->data->length*dataPtr->timeData->deltaT;
  f_min = dataPtr->fLow;
  f_max = dataPtr->fHigh;
  pmin = 0.0;
  pmax = LAL_TWOPI;

  gsl_matrix  *gFD       = gsl_matrix_alloc(nifo,(int)n); //store the Fourier-domain glitch signal
  gsl_matrix  *gpower    = gsl_matrix_alloc(nifo,(int)n); //store the (normalized) wavelet power in each pixel
  REAL8Vector *maxpower  = XLALCreateREAL8Vector(nifo);   //store the maximum power in any pixel for each ifo (for rejection sampling proposed wavelets)

  for(i=0; i<nifo; i++) gsize->data[i]=0;

  //Morlet wavelet parameters
  LALInferenceAddVariable(currentParams, "morlet_FD",  &gFD,  LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(currentParams, "morlet_Amp", &mAmp, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(currentParams, "morlet_f0" , &mf0,  LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(currentParams, "morlet_Q"  , &mQ,   LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(currentParams, "morlet_t0" , &mt0,  LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(currentParams, "morlet_phi", &mphi, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);

  LALInferenceAddVariable(currentParams, "glitch_size",   &gsize, LALINFERENCE_UINT4Vector_t, LALINFERENCE_PARAM_LINEAR);
  LALInferenceAddVariable(currentParams, "glitchFitFlag", &gflag, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);

  LALInferenceAddMinMaxPrior(priorArgs, "morlet_Amp_prior", &Amin, &Amax, LALINFERENCE_REAL8_t);
  LALInferenceAddMinMaxPrior(priorArgs, "morlet_f0_prior" , &f_min, &f_max, LALINFERENCE_REAL8_t);
  LALInferenceAddMinMaxPrior(priorArgs, "morlet_Q_prior"  , &Qmin, &Qmax, LALINFERENCE_REAL8_t);
  LALInferenceAddMinMaxPrior(priorArgs, "morlet_t0_prior" , &tmin, &tmax, LALINFERENCE_REAL8_t);
  LALInferenceAddMinMaxPrior(priorArgs, "morlet_phi_prior", &pmin, &pmax, LALINFERENCE_REAL8_t);

  LALInferenceAddMinMaxPrior(priorArgs, "glitch_dim", &gmin, &gmax, LALINFERENCE_REAL8_t);

  //Meyer-wavelet based proposal distribution
  LALInferenceAddVariable(proposalArgs, "glitch_max_power", &maxpower, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(proposalArgs, "glitch_power", &gpower, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_FIXED);

}

void LALInferenceRegisterUniformVariableREAL8(LALInferenceRunState *state, LALInferenceVariables *var, const char name[VARNAME_MAX], REAL8 startval, REAL8 min, REAL8 max, LALInferenceParamVaryType varytype)
{
  char minopt[VARNAME_MAX+7];
  char maxopt[VARNAME_MAX+7];
  char valopt[VARNAME_MAX+3];
  char fixopt[VARNAME_MAX+7];
  ProcessParamsTable *ppt=NULL;
  
  sprintf(minopt,"--%s-min",name);
  sprintf(maxopt,"--%s-max",name);
  sprintf(valopt,"--%s",name);
  sprintf(fixopt,"--fix-%s",name);
  
  if((ppt=LALInferenceGetProcParamVal(state->commandLine,minopt))) min=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(state->commandLine,maxopt))) max=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(state->commandLine,fixopt))) varytype=LALINFERENCE_PARAM_FIXED;
  if((ppt=LALInferenceGetProcParamVal(state->commandLine,valopt))) startval=atof(ppt->value);
  else if(varytype!=LALINFERENCE_PARAM_FIXED) startval=min+(max-min)*gsl_rng_uniform(state->GSLrandom);
  
  /* Error checking */
  if(min>max) {
    fprintf(stderr,"ERROR: Prior for %s has min(%lf) > max(%lf)\n",name,min,max);
    exit(1);
  }
  if(startval<min || startval>max){
    fprintf(stderr,"ERROR: Initial value %lf for %s lies outwith prior (%lf,%lf)\n",startval,name,min,max);
    exit(1);
  }
  
  LALInferenceAddVariable(var,name,&startval,LALINFERENCE_REAL8_t,varytype);
  LALInferenceAddMinMaxPrior(state->priorArgs, name, &min, &max, LALINFERENCE_REAL8_t);
  
}


/* Setup the variables to control template generation for the CBC model */
/* Includes specification of prior ranges. Returns address of new LALInferenceVariables */

LALInferenceModel *LALInferenceInitCBCModel(LALInferenceRunState *state)
{

  char help[]="\
               \n\
               ------------------------------------------------------------------------------------------------------------------\n\
               --- Injection Arguments ------------------------------------------------------------------------------------------\n\
               ------------------------------------------------------------------------------------------------------------------\n\
               (--inj injections.xml)          Injection XML file to use.\n\
               (--event N)                     Event number from Injection XML file to use.\n\
               \n\
               ------------------------------------------------------------------------------------------------------------------\n\
               --- Template Arguments -------------------------------------------------------------------------------------------\n\
               ------------------------------------------------------------------------------------------------------------------\n\
               (--symMassRatio)                Jump in symmetric mass ratio eta, instead of q=m2/m1.\n\
               (--use-logdistance)             Jump in log(distance) instead of distance.\n\
               (--system-frame                 Jump in spin parameters defined in the system coordinates, relative to total angular momentum\n\
               (--approx)                      Specify a template approximant and phase order to use.\n\
                                               (default TaylorF2threePointFivePN). Available approximants:\n\
                                               default modeldomain=\"time\": GeneratePPN, TaylorT1, TaylorT2, TaylorT3, TaylorT4, \n\
                                                                           EOB, EOBNR, EOBNRv2, EOBNRv2HM, SEOBNRv1, SpinTaylor, \n\
                                                                           SpinQuadTaylor, SpinTaylorFrameless, SpinTaylorT4, \n\
                                                                           PhenSpinTaylorRD, NumRel.\n\
                                               default modeldomain=\"frequency\": TaylorF1, TaylorF2, TaylorF2RedSpin, \n\
                                                                                TaylorF2RedSpinTidal, IMRPhenomA, IMRPhenomB, IMRPhenomP.\n\
               (--amporder PNorder)            Specify a PN order in amplitude to use (defaults: LALSimulation: max available; LALInspiral: newtownian).\n\
               (--fref fRef)                   Specify a reference frequency at which parameters are defined (default 0).\n\
               (--tidal)                       Enables tidal corrections, only with LALSimulation.\n\
               (--tidalT)                      Enables reparmeterized tidal corrections, only with LALSimulation.\n\
               (--spinOrder PNorder)           Specify twice the PN order (e.g. 5 <==> 2.5PN) of spin effects to use, only for LALSimulation (default: -1 <==> Use all spin effects).\n\
               (--tidalOrder PNorder)          Specify twice the PN order (e.g. 10 <==> 5PN) of tidal effects to use, only for LALSimulation (default: -1 <==> Use all tidal effects).\n\
               (--modeldomain)                 domain the waveform template will be computed in (\"time\" or \"frequency\").\n\
               (--spinAligned or --aligned-spin)  template will assume spins aligned with the orbital angular momentum.\n\
                                               *Enables* spins for TaylorF2, TaylorF2RedSpin, TaylorF2RedSpinTidal, IMRPhenomB.\n\
               (--singleSpin)                  template will assume only the spin of the most massive binary component exists.\n\
               (--noSpin, --disable-spin)      template will assume no spins.\n\
               \n\
               ------------------------------------------------------------------------------------------------------------------\n\
               --- Starting Parameters ------------------------------------------------------------------------------------------\n\
               ------------------------------------------------------------------------------------------------------------------\n\
               (--trigtime time)               Trigger time to use.\n\
               (--time time)                   Waveform time (overrides random about trigtime).\n\
               (--mchirp VALUE)                Trigger chirpmass to use.\n\
               (--eta VALUE)                   Trigger eta (symmetric mass ratio) to use.\n\
               (--q VALUE)                     Trigger q (asymmetric mass ratio) to use.\n\
               (--phase VALUE)                 Trigger phase to use.\n\
               (--inclination VALUE)           Trigger inclination to use.\n\
               (--distance VALUE)              Trigger distance to use.\n\
               (--rightascension VALUE)        Trigger RA.\n\
               (--declination VALUE)           Trigger declination.\n\
               (--polarisation VALUE)          Trigger polarisation angle.\n\
               Spin Parameters:\n\
               (--a1 VALUE)                    Trigger a1.\n\
               (--theta1 VALUE)                Trigger theta1.\n\
               (--phi1 VALUE)                  Trigger phi1.\n\
               (--a2 VALUE)                    Trigger a2.\n\
               (--theta2 VALUE)                Trigger theta2.\n\
               (--phi2 VALUE)                  Trigger phi2.\n\
               Equation of State parameters:\n\
               (--lambda1 VALUE)               Trigger lambda1.\n\
               (--lambda2 VALUE)               Trigger lambda2.\n\
               (--lambdaT VALUE)               Trigger lambdaT.\n\
               (--dLambdaT VALUE)              Trigger dLambdaT.\n\
               \n\
               ------------------------------------------------------------------------------------------------------------------\n\
               --- Prior Arguments ----------------------------------------------------------------------------------------------\n\
               ------------------------------------------------------------------------------------------------------------------\n\
               You can generally use --paramname-min MIN --paramname-max MAX to set the prior range for the parameter paramname  \n\
                                                      where paramname is the name used internally. Examples and defaults below:\n\n\
               (--mchirp-min MIN, --mchirp-max MAX)    Min and max chirp mass\n\
               (--eta-min MIN, --eta-max MAX)          Min and max eta.\n\
               (--q-min MIN, --q-max MAX)              Min and max mass asymmetric mass ratio q.\n\
               (--comp-min min)                        Minimum component mass (1.0).\n\
               (--comp-max max)                        Maximum component mass (30.0).\n\
               (--mtotalmin min)                       Minimum total mass (2.0).\n\
               (--mtotalmax max)                       Maximum total mass (35.0).\n\
               (--a-min max)                           Minimum component spin (-1.0 for spin-aligned, 0.0 for precessing).\n\
               (--a-max max)                           Maximum component spin (1.0).\n\
               (--inclination-min MIN                  Minimum inclination angle (0.0).\n\
               (--inclination-max MAX)                 Maximum inclination angle (pi).\n\
               (--distance-min MIN)                    Minimum distance in Mpc (1).\n\
               (--distance-max MAX)                    Maximum distance in Mpc (100).\n\
               (--dt time)                             Width of time prior, centred around trigger (0.1s).\n\
               (--malmquistPrior)                      Rejection sample based on SNR of template \n\
               Equation of state parameters:\n\
               (--lambda1-min)                         Minimum lambda1 (0).\n\
               (--lambda1-max)                         Maximum lambda1 (3000).\n\
               (--lambda2-min)                         Minimum lambda2 (0).\n\
               (--lambda2-max)                         Maximum lambda2 (3000).\n\
               (--lambdaT-min)                         Minimum lambdaT (0).\n\
               (--lambdaT-max)                         Maximum lambdaT (3000).\n\
               (--dLambdaT-min)                        Minimum dLambdaT (-500).\n\
               (--dLambdaT-max)                        Maximum dLambdaT (500).\n\
               \n\
               ------------------------------------------------------------------------------------------------------------------\n\
               --- Fix Parameters   ---------------------------------------------------------------------------------------------\n\
               ------------------------------------------------------------------------------------------------------------------\n\
               (--fix-chirpmass)                        Do not allow chirpmass to vary.\n\
               (--fix-massratio, --fix-asym_massratio)  Do not allow mass ratio to vary.\n\
               (--fix-phase)                            Do not allow phase to vary.\n\
               (--fix-inclination)                      Do not allow inclination to vary.\n\
               (--fix-distance)                         Do not allow distance to vary.\n\
               (--fix-rightascension)                   Do not allow RA to vary.\n\
               (--fix-declination)                      Do not allow declination to vary.\n\
               (--fix-polarisation)                     Do not allow polarisation to vary.\n\
               (--fix-a_spin1, --fix-a_spin2)           Do not allow spin magnitude to vary (for mass1 and 2 respectively).\n\
               (--fix-theta_spin1, --fix-theta_spin2)   Do not allow spin colatitude to vary.\n\
               (--fix-phi_spin1, --fix-phi_spin2)       Do not allow spin 1 longitude to vary.\n\
               (--fix-time)                             Do not allow coalescence time to vary.\n\
               (--fix-lambda1, --fix-lambda2)           Do not allow lambda EOS parameters for component masses to vary.\n\
               (--fix-lambdaT, --fix-dLambdaT)          Do not allow reparameterized EOS parameters to vary (needs --tidalT).\n\
               (--varyFlow, --flowMin, --flowMax)       Allow the lower frequency bound of integration to vary in given range.\n\
               (--pinparams)                            List of parameters to set to injected values [mchirp,asym_massratio,etc].\n";


  /* Print command line arguments if state was not allocated */
  if(state==NULL)
    {
      fprintf(stdout,"%s",help);
      return(NULL);
    }

  /* Print command line arguments if help requested */
  if(LALInferenceGetProcParamVal(state->commandLine,"--help"))
    {
      fprintf(stdout,"%s",help);
      return(NULL);
    }

  LALStatus status;
  memset(&status,0,sizeof(status));
  int errnum;
  SimInspiralTable *injTable=NULL;
  LALInferenceVariables *priorArgs=state->priorArgs;
  LALInferenceVariables *proposalArgs=state->proposalArgs;
  ProcessParamsTable *commandLine=state->commandLine;
  ProcessParamsTable *ppt=NULL;
  ProcessParamsTable *ppt_order=NULL;
  LALPNOrder PhaseOrder=-1;
  LALPNOrder AmpOrder=-1;
  Approximant approx=NumApproximants;
  REAL8 fRef = 100.0;
  LALInferenceApplyTaper bookends = LALINFERENCE_TAPER_NONE;
  LALInferenceFrame frame = LALINFERENCE_FRAME_SYSTEM;
  UINT4 analytic=0;
  LALInferenceIFOData *dataPtr;
  UINT4 event=0;
  UINT4 i=0;
  REAL8 m1=0;
  REAL8 m2=0;
  REAL8 Dmin=1.0;
  REAL8 Dmax=100.0;
  REAL8 mcMin=1.0;
  REAL8 mcMax=15.3;
  REAL8 mMin=1.0,mMax=30.0;
  REAL8 MTotMax=35.0;
  REAL8 MTotMin=2.0;
  REAL8 etaMin=0.0312;
  REAL8 etaMax=0.25;
  REAL8 qMin=mMin/mMax;
  REAL8 qMax=1.0;
  REAL8 psiMin=0.0,psiMax=LAL_PI;
  REAL8 decMin=-LAL_PI/2.0,decMax=LAL_PI/2.0;
  REAL8 raMin=0.0,raMax=LAL_TWOPI;
  REAL8 phiMin=0.0,phiMax=LAL_TWOPI;


  REAL8 dt=0.1;            /* Width of time prior */
  REAL8 lambda1Min=0.0;
  REAL8 lambda1Max=3000.0;
  REAL8 lambda2Min=0.0;
  REAL8 lambda2Max=3000.0;
  REAL8 lambdaTMin=0.0;
  REAL8 lambdaTMax=3000.0;
  REAL8 dLambdaTMin=-500.0;
  REAL8 dLambdaTMax=500.0;
  gsl_rng *GSLrandom=state->GSLrandom;
  REAL8 endtime=0.0, timeParam=0.0;
  REAL8 timeMin=endtime-dt,timeMax=endtime+dt;

  LALInferenceModel *model = XLALMalloc(sizeof(LALInferenceModel));
  model->params = XLALCalloc(1, sizeof(LALInferenceVariables));
  memset(model->params, 0, sizeof(LALInferenceVariables));

  /* Over-ride prior bounds if analytic test */
  if (LALInferenceGetProcParamVal(commandLine, "--correlatedGaussianLikelihood"))
  {
    return(LALInferenceInitModelReviewEvidence(state));
  }
  else if (LALInferenceGetProcParamVal(commandLine, "--bimodalGaussianLikelihood"))
  {
    return(LALInferenceInitModelReviewEvidence_bimod(state));
  }
  else if (LALInferenceGetProcParamVal(commandLine, "--rosenbrockLikelihood"))
  {
    return(LALInferenceInitModelReviewEvidence_banana(state));
  }

  if(LALInferenceGetProcParamVal(commandLine,"--malmquistPrior"))
  {
    UINT4 malmquistflag=1;
    LALInferenceAddVariable(model->params, "malmquistPrior",&malmquistflag,LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  }

  if(LALInferenceGetProcParamVal(commandLine,"--skyLocPrior")){
    MTotMax=20.0;
    mMin=1.0;
    mMax=15.0;
    qMin=mMin/mMax;
    Dmin=10.0;
    Dmax=40.0;
    REAL8 densityVNR=1000.0;
    LALInferenceAddVariable(state->priorArgs,"densityVNR", &densityVNR , LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
  }

  /* Read injection XML file for parameters if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--inj");
  if(ppt){
    SimInspiralTableFromLIGOLw(&injTable,ppt->value,0,0);
    if(!injTable){
      fprintf(stderr,"Unable to open injection file %s\n",ppt->value);
      exit(1);
    }
    ppt=LALInferenceGetProcParamVal(commandLine,"--event");
    if(ppt){
      event= atoi(ppt->value);
      fprintf(stderr,"Reading event %d from file\n",event);
      i=0;
      while(i<event) {i++; injTable=injTable->next;} /* select event */
    }
  }

  /* See if there are any parameters pinned to injection values */
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--pinparams"))){
    char *pinned_params=ppt->value;
    LALInferenceVariables tempParams;
    memset(&tempParams,0,sizeof(tempParams));
    char **strings=NULL;
    UINT4 N;
    LALInferenceParseCharacterOptionString(pinned_params,&strings,&N);
    LALInferenceInjectionToVariables(injTable,&tempParams);
    LALInferenceVariableItem *node=NULL;
    while(N>0){
      N--;
      char *name=strings[N];
      fprintf(stdout,"Pinning parameter %s\n",node->name);
      node=LALInferenceGetItem(&tempParams,name);
      if(node) LALInferenceAddVariable(model->params,node->name,node->value,node->type,node->vary);
      else {fprintf(stderr,"Error: Cannot pin parameter %s. No such parameter found in injection!\n",node->name);}
    }
  }
  
  
  /* Over-ride approximant if user specifies */
  ppt=LALInferenceGetProcParamVal(commandLine,"--approximant");
  if(ppt){
    approx = XLALGetApproximantFromString(ppt->value);
    ppt_order=LALInferenceGetProcParamVal(commandLine,"--order");
    if(ppt_order) PhaseOrder = XLALGetOrderFromString(ppt_order->value);
  }
  ppt=LALInferenceGetProcParamVal(commandLine,"--approx");
  if(ppt){
    approx = XLALGetApproximantFromString(ppt->value);
    XLAL_TRY(PhaseOrder = XLALGetOrderFromString(ppt->value),errnum);
    if( (int) PhaseOrder == XLAL_FAILURE || errnum) {
      PhaseOrder=-1;
    }
  }
  
  ppt=LALInferenceGetProcParamVal(commandLine,"--amporder");
  if(ppt) AmpOrder=atoi(ppt->value);
  
  if(approx==NumApproximants && injTable){ /* Read aproximant from injection file */
    approx=XLALGetApproximantFromString(injTable->waveform);
  }
  if(approx==NumApproximants){
      
       approx=TaylorF2; /* Defaults to TF2 */
       XLALPrintWarning("You did not provide an approximant for the templates. Using default %s, which might now be what you want!\n",XLALGetStringFromApproximant(approx));
  }
    
  /* Set the model domain appropriately */
  if (XLALSimInspiralImplementedFDApproximants(approx)) {
    model->domain = LAL_SIM_DOMAIN_FREQUENCY;
  } else if (XLALSimInspiralImplementedTDApproximants(approx)) {
    model->domain = LAL_SIM_DOMAIN_TIME;
  } else {
    fprintf(stderr,"ERROR. Unknown approximant number %i. Unable to choose time or frequency domain model.",approx);
    exit(1);
  }

  ppt=LALInferenceGetProcParamVal(commandLine, "--fref");
  if (ppt) fRef = atof(ppt->value);

  ppt=LALInferenceGetProcParamVal(commandLine,"--modeldomain");
  if(ppt){
    if ( ! strcmp( "time", ppt->value ) )
    {
      model->domain = LAL_SIM_DOMAIN_TIME;
    }
    else if ( ! strcmp( "frequency", ppt->value ) )
    {
      model->domain = LAL_SIM_DOMAIN_FREQUENCY;
    }
    else
    {
      fprintf( stderr, "invalid argument to --modeldomain:\n"
              "unknown domain specified: "
              "domain must be one of: time, frequency\n");
      exit( 1 );
    }
  }
  
  /* This flag was added to account for the broken Big Dog
     injection, which had the opposite sign in H and L compared
     to Virgo. */
  if (LALInferenceGetProcParamVal(commandLine, "--crazyInjectionHLSign")) {
    INT4 flag = 1;
    LALInferenceAddVariable(model->params, "crazyInjectionHLSign", &flag, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  } else {
    INT4 flag = 0;
    LALInferenceAddVariable(model->params, "crazyInjectionHLSign", &flag, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  }

  /* Over-ride taper if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--taper");
  if(ppt){
    if(strstr(ppt->value,"STARTEND")) bookends=LALINFERENCE_TAPER_STARTEND;
    if(strstr(ppt->value,"STARTONLY")) bookends=LALINFERENCE_TAPER_START;
    if(strstr(ppt->value,"ENDONLY")) bookends=LALINFERENCE_TAPER_END;
    if(strstr(ppt->value,"RING")) bookends=LALINFERENCE_RING;
    if(strstr(ppt->value,"SMOOTH")) bookends=LALINFERENCE_SMOOTH;
  }

  
  /************ Prior Related Settings START ******************************/
  
  /* Over-ride time prior if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--dt");
  if(ppt){
    dt=atof(ppt->value);
    timeMin=endtime-dt;
    timeMax=endtime+dt;
  }
  
  /* Over-ride component masses */
  ppt=LALInferenceGetProcParamVal(commandLine,"--comp-min");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--compmin");
  if(ppt)	mMin=atof(ppt->value);
  LALInferenceAddVariable(priorArgs,"component_min",&mMin,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
  
  ppt=LALInferenceGetProcParamVal(commandLine,"--comp-max");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--compmax");
  if(ppt)	mMax=atof(ppt->value);
  LALInferenceAddVariable(priorArgs,"component_max",&mMax,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
  
  // Update qMin according to the new masses.
  qMin=mMin/mMax;
  
  /* Over-ride Mass priors if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--mc-min");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--Mmin");
  if(ppt)
  {
    mcMin=atof(ppt->value);
    if (mcMin < 0)
    {
      fprintf(stderr,"ERROR: Minimum value of mchirp must be > 0");
      exit(1);
    }
  }
  else mcMin=pow(mMin*mMin,0.6)/pow(2.0*mMin,0.2);
  
  ppt=LALInferenceGetProcParamVal(commandLine,"--mc-max");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--Mmax");
  if(ppt)
  {
    mcMax=atof(ppt->value);
    if (mcMax <= 0)
    {
      fprintf(stderr,"ERROR: Maximum value of mchirp must be > 0");
      exit(1);
    }
  }
  else mcMax=pow(mMax*mMax,0.6)/pow(2.0*mMax,0.2);
  
  ppt=LALInferenceGetProcParamVal(commandLine,"--eta-min");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--etamin");
  if(ppt)
  {
    etaMin=atof(ppt->value);
    if (etaMin < 0.0)
    {
      fprintf(stderr,"ERROR: Minimum value of eta must be > 0");
      exit(1);
    }
  }
  
  ppt=LALInferenceGetProcParamVal(commandLine,"--eta-max");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--etamax");
  if(ppt)
  {
    etaMax=atof(ppt->value);
    if (etaMax > 0.25 || etaMax <= 0.0)
    {
      fprintf(stderr,"ERROR: Maximum value of eta must be between 0 and 0.25\n");
      exit(1);
    }
  }
   
  ppt=LALInferenceGetProcParamVal(commandLine,"--MTotMax");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--mtotalmax");
  if(ppt)	MTotMax=atof(ppt->value);
  LALInferenceAddVariable(priorArgs,"MTotMax",&MTotMax,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);

  ppt=LALInferenceGetProcParamVal(commandLine,"--MTotMin");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--mtotalmin");
  if(ppt)	MTotMin=atof(ppt->value);
  LALInferenceAddVariable(priorArgs,"MTotMin",&MTotMin,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
   
  ppt=LALInferenceGetProcParamVal(commandLine,"--q-min");
  if(ppt)
  {
    qMin=atof(ppt->value);
    if (qMin <= 0.0 || qMin < mMin/mMax || qMin < mMin/(MTotMax-mMin) || qMin > 1.0)
    {
      fprintf(stderr,"ERROR: invalid qMin ( max{0,mMin/mMax,mMin/(MTotMax-mMin) < q < 1.0} )");
      exit(1);
    }
  }

  ppt=LALInferenceGetProcParamVal(commandLine,"--q-max");
  if(ppt)
  {
    qMax=atof(ppt->value);
    if (qMax > 1.0 || qMax <= 0.0 || qMax < mMin/mMax || qMax < mMin/(MTotMax-mMin))
    {
      fprintf(stderr,"ERROR: invalid qMax ( max{0,mMin/mMax,mMin/(MTotMax-mMin) < q < 1.0} )");
      exit(1);
    }
  }

  /* IMRPhenomP only supports q > 1/20. Set prior consequently*/
  if (approx==IMRPhenomP && qMin<1./20.){
    qMin=1.0/20.;
    XLALPrintWarning("WARNING: IMRPhenomP only supports mass ratio up to 20. Changing the prior accordingly\n");
  }
  
  /* Prior related arguments END */

  /* Initial values set after setting up prior */
  /************ Initial Value Related Argument START *************/

  REAL8 start_mc			=mcMin+gsl_rng_uniform(GSLrandom)*(mcMax-mcMin);
  REAL8 start_eta			=etaMin+gsl_rng_uniform(GSLrandom)*(etaMax-etaMin);
  REAL8 start_q           =qMin+gsl_rng_uniform(GSLrandom)*(qMax-qMin);
  REAL8 start_phase		=0.0+gsl_rng_uniform(GSLrandom)*(LAL_TWOPI-0.0);
  REAL8 start_dist		=Dmin+gsl_rng_uniform(GSLrandom)*(Dmax-Dmin);
  REAL8 start_ra			=0.0+gsl_rng_uniform(GSLrandom)*(LAL_TWOPI-0.0);
  REAL8 start_dec			=-LAL_PI/2.0+gsl_rng_uniform(GSLrandom)*(LAL_PI_2-(-LAL_PI_2));
  REAL8 start_psi			=0.0+gsl_rng_uniform(GSLrandom)*(LAL_PI-0.0);

  
  /* Read time parameter from injection file */
  if(injTable)
  {
    endtime=XLALGPSGetREAL8(&(injTable->geocent_end_time));
    fprintf(stdout,"Using end time from injection file: %lf\n", endtime);
  }
  /* Over-ride end time if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--trigtime");
  if(ppt && !analytic){
    endtime=atof(ppt->value);
    printf("Read end time %f\n",endtime);
  }
  /* Adjust prior accordingly */
  if (!analytic) {
      timeMin=endtime-dt; timeMax=endtime+dt;
  }
  
  /* Over-ride chirp mass if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--mc");
  if(ppt){
    start_mc=atof(ppt->value);
  }

  /* Over-ride eta if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--eta");
  if(ppt){
    start_eta=atof(ppt->value);
    LALInferenceMcEta2Masses(start_mc, start_eta, &m1, &m2);
    start_q=m2/m1;
  }

  /* Over-ride q if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--q");
  if(ppt){
    start_q=atof(ppt->value);
    LALInferenceQ2Eta(start_q, &start_eta);
  }

  /* Set up start time. */
  ppt=LALInferenceGetProcParamVal(commandLine, "--time");
  if (ppt) {
    /* User has specified start time. */
    timeParam = atof(ppt->value);
  } else {
    timeParam = timeMin + (timeMax-timeMin)*gsl_rng_uniform(GSLrandom);
  }

  /* Non-standard names for backward compatibility */
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--phi"))) start_phase=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--dist"))) start_dist=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--Dmin"))) Dmin=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--Dmax"))) Dmax=atof(ppt->value);

  
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--psi")))  start_psi=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--dec")))  start_dec=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--ra")))  start_ra=atof(ppt->value);

  
  /* Initial Value Related END */
  
  
  LALInferenceAddVariable(model->params, "LAL_APPROXIMANT", &approx,        LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(model->params, "LAL_PNORDER",     &PhaseOrder,        LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(model->params, "LAL_AMPORDER",     &AmpOrder,        LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);

  LALInferenceAddVariable(model->params, "fRef", &fRef, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);

  REAL8 fLow = state->data->fLow;
  ppt=LALInferenceGetProcParamVal(commandLine,"--varyFlow");
  if(ppt){
    REAL8 fLow_min = fLow;
    REAL8 fLow_max = 200.0;
    ppt=LALInferenceGetProcParamVal(commandLine,"--flowMin");
    if(ppt){
      fLow_min=strtod(ppt->value,(char **)NULL);
    }
    ppt=LALInferenceGetProcParamVal(commandLine,"--flowMax");
    if(ppt){
      fLow_max=strtod(ppt->value,(char **)NULL);
    }
    if(LALInferenceCheckVariable(model->params,"fRef"))
      fRef = *(REAL8*)LALInferenceGetVariable(model->params, "fRef");
      if (fRef > 0.0 && fLow_max > fRef) {
        fprintf(stdout,"WARNING: fLow can't go higher than the reference frequency.  Setting fLow_max to %f\n",fRef);
        fLow_max = fRef;
      }
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "fLow", fLow, fLow_min, fLow_max, LALINFERENCE_PARAM_LINEAR);
  } else {
    LALInferenceAddVariable(model->params, "fLow", &fLow,  LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
  }

  ppt=LALInferenceGetProcParamVal(commandLine,"--taper");
  if(ppt){
    LALInferenceAddVariable(model->params, "LALINFERENCE_TAPER",     &bookends,        LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  }
  ppt=LALInferenceGetProcParamVal(commandLine,"--newswitch");
  int newswitch=0;
  if(ppt){
    newswitch=1;
    LALInferenceAddVariable(model->params, "newswitch", &newswitch, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  }

  ppt=LALInferenceGetProcParamVal(commandLine, "--radiation-frame");
  if(ppt){
    frame = LALINFERENCE_FRAME_RADIATION;
  }
  LALInferenceAddVariable(model->params, "LALINFERENCE_FRAME", &frame, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);

  /* Set up the variable parameters */

  /********************* TBL: Adding noise-fitting parameters  *********************/
  UINT4 nscale_block; //number of noise parameters per IFO (1 per frequency block)
  UINT4 nscale_bin;   //number of Fourier bins in each noise block
  REAL8 nscale_dflog; //logarithmic spacing for noise parameters
  REAL8 nscale_min;   //minimum value for psd scale parameter
  REAL8 nscale_max;   //maximum value for psd scale parameters
  UINT4 nscale_dim;   //total dimension of noise model (params X detectors)
  UINT4 nscale_flag;  //flag to tell likelihood function if psd fitting is in use

  REAL8Vector *nscale_prior = NULL; //std. dev. of prior distribution
  REAL8Vector *nscale_sigma = NULL; //std. dev. of prior distribution

  //assume no noise fitting
  nscale_flag=0;

  //set Nblock to default unless specified at command line
  ppt = LALInferenceGetProcParamVal(commandLine, "--psdNblock");
  if(ppt) nscale_block = atoi(ppt->value);
  else nscale_block = 8;

  //First, figure out sizes of dataset to set up noise blocks
  UINT4 nifo; //number of data channels
  UINT4 imin; //minimum Fourier bin for integration in IFO
  UINT4 imax; //maximum Fourier bin for integration in IFO
  UINT4 f_min = 1; //minimum Fourier bin for integration over network
  UINT4 f_max = 1; //maximum Fourier bin for integration over network
  REAL8 df = 1.0; //frequency resolution

  /* Set model sampling rates to be consistent with data */
  model->deltaT = state->data->timeData->deltaT;
  model->deltaF = state->data->freqData->deltaF;

  //compute imin,imax for each IFO -- may be different
  nifo=0;
  dataPtr = state->data;
  while (dataPtr != NULL)
  {
    df      = 1.0 / (((double)dataPtr->timeData->data->length) * model->deltaT);
    imin    = (UINT4)ceil( dataPtr->fLow  / df);
    imax    = (UINT4)floor(dataPtr->fHigh / df);

    if(nifo==0)
    {
      f_min=imin;
      f_max=imax;
    }
    else
    {
      if(imin<f_min)
      {
        fprintf(stderr,"Warning: Different IFO's have different minimum frequencies -- bad for noise fitting\n");
        f_min=imin;
      }
      if(imax>f_max)
      {
        fprintf(stderr,"Warning: Different IFO's have different minimum frequencies -- bad for noise fitting\n");
        f_max=imax;
      }
    }

    dataPtr = dataPtr->next;
    nifo++;
  }

  UINT4 j = 0;

  ppt = LALInferenceGetProcParamVal(commandLine, "--psdFit");
  if(ppt)//MARK: Here is where noise PSD parameters are being added to the model
  {
 
    printf("Setting up PSD fitting for %i ifos...\n",nifo);

    dataPtr = state->data;
    UINT4 nscale_block_temp   = 10000;
    gsl_matrix *bands_min_temp      = gsl_matrix_alloc(nifo,nscale_block_temp);
    gsl_matrix *bands_max_temp      = gsl_matrix_alloc(nifo,nscale_block_temp);

    i=0;
    while (dataPtr != NULL)
    {

      for (j = 0; j < nscale_block_temp; j++)
      {
        gsl_matrix_set(bands_min_temp,i,j,-1.0);
        gsl_matrix_set(bands_max_temp,i,j,-1.0);
      }

      printf("ifo=%i  %s\n",i,dataPtr->name);fflush(stdout);

      char ifoPSDFitBands[500];
      snprintf (ifoPSDFitBands,500, "--%s-psdFit",dataPtr->name);

      ppt = LALInferenceGetProcParamVal(commandLine,ifoPSDFitBands);
      if(ppt || LALInferenceGetProcParamVal(commandLine, "--xcorrbands"))
      /* Load in values from file if requested */
      {
        char line [ 128 ];
        char bands_tempfile[500];

        if (LALInferenceGetProcParamVal(commandLine, "--xcorrbands")) {
           snprintf (bands_tempfile,500, "%s-XCorrBands.dat",dataPtr->name);
        }
        else {
           char *bands_tempfile_temp = ppt->value;
           strcpy( bands_tempfile, bands_tempfile_temp );
        }
        printf("Reading bands_temp from %s\n",bands_tempfile);

        UINT4 band_min = 0, band_max = 0;
 
        nscale_block = 0;
        char * pch;
        j = 0;

        FILE *file = fopen ( bands_tempfile, "r" );
        if ( file != NULL )
        {
          while ( fgets ( line, sizeof line, file ) != NULL )
          {
              pch = strtok (line," ");
              int count = 0;
              while (pch != NULL)
              {
                  if (count==0) {band_min = atof(pch);}
                  if (count==1) {band_max = atof(pch);}
                  pch = strtok (NULL, " ");
                  count++;
              }

              gsl_matrix_set(bands_min_temp,i,j,band_min/df);
              gsl_matrix_set(bands_max_temp,i,j,band_max/df);
 
              nscale_block++;
              j++;

          }
        fclose ( file );
        }

        else
        {
          perror ( bands_tempfile ); /* why didn't the file open? */
        }
  

      }
      else // Otherwise use defaults
      {

        nscale_bin   = (f_max+1-f_min)/nscale_block;
        nscale_dflog = log( (double)(f_max+1)/(double)f_min )/(double)nscale_block;

        int freq_min, freq_max;

        for (j = 0; j < nscale_block; j++)
        {

            freq_min = (int) exp(log((double)f_min ) + nscale_dflog*j);
            freq_max = (int) exp(log((double)f_min ) + nscale_dflog*(j+1));

            gsl_matrix_set(bands_min_temp,i,j,freq_min);
            gsl_matrix_set(bands_max_temp,i,j,freq_max);
        }

      }  

      dataPtr = dataPtr->next;
      i++;

    }


    gsl_matrix *bands_min      = gsl_matrix_alloc(nifo,nscale_block);
    gsl_matrix *bands_max      = gsl_matrix_alloc(nifo,nscale_block);

    for (i = 0; i < nifo; i++)
    {
      for (j = 0; j < nscale_block; j++)
      {
        gsl_matrix_set(bands_min,i,j,gsl_matrix_get(bands_min_temp,i,j));
        gsl_matrix_set(bands_max,i,j,gsl_matrix_get(bands_max_temp,i,j));

      }
    }

    printf("Running PSD fitting with bands (Hz)...\n");
    dataPtr = state->data;
    i=0;
    while (dataPtr != NULL)
    {
      printf("%s:",dataPtr->name);
      for (j = 0; j < nscale_block; j++)
      {
        printf(" %f-%f ",gsl_matrix_get(bands_min,i,j)*df,gsl_matrix_get(bands_max,i,j)*df);
      }
      printf("\n");
      dataPtr = dataPtr->next;
      i++;
    }

    nscale_bin   = (f_max+1-f_min)/nscale_block;
    nscale_dflog = log( (double)(f_max+1)/(double)f_min )/(double)nscale_block;

    nscale_min   = 1.0e-1;
    nscale_max   = 1.0e+1;
    nscale_dim   = nscale_block*nifo;
    nscale_flag  = 1;

    // Set noise parameter arrays.
    nscale_prior = XLALCreateREAL8Vector(nscale_block);
    nscale_sigma = XLALCreateREAL8Vector(nscale_block);
    for(i=0; i<nscale_block; i++)
    {
      nscale_prior->data[i] = 1.0/sqrt( gsl_matrix_get(bands_max,0,i)-gsl_matrix_get(bands_min,0,i) );
      nscale_sigma->data[i] = nscale_prior->data[i]/sqrt((double)(nifo*nscale_block));
    }

    gsl_matrix *nscale = gsl_matrix_alloc(nifo,nscale_block);
    gsl_matrix_set_all(nscale, 1.0);

    LALInferenceAddVariable(model->params, "psdscale", &nscale, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_LINEAR);
    LALInferenceAddVariable(model->params, "logdeltaf", &nscale_dflog, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);

    LALInferenceAddVariable(model->params, "psdBandsMin", &bands_min, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(model->params, "psdBandsMax", &bands_max, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_FIXED);

    //Set up noise priors
    LALInferenceAddVariable(priorArgs,      "psddim",   &nscale_dim,  LALINFERENCE_INT4_t,  LALINFERENCE_PARAM_FIXED);
    LALInferenceAddMinMaxPrior(priorArgs,   "psdrange", &nscale_min,  &nscale_max,   LALINFERENCE_REAL8_t);
    LALInferenceAddVariable(priorArgs,      "psdsigma", &nscale_prior, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_FIXED);

    //Store meta data for noise model in proposal
    LALInferenceAddVariable(proposalArgs, "psdblock", &nscale_block, LALINFERENCE_INT4_t,  LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(proposalArgs, "psdbin",   &nscale_bin,   LALINFERENCE_INT4_t,  LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(proposalArgs, "psdsigma", &nscale_sigma, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_FIXED);


  }//End of noise model initialization
  LALInferenceAddVariable(model->params, "psdScaleFlag", &nscale_flag, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);

  UINT4 psdGaussianPrior=1;
  ppt = LALInferenceGetProcParamVal(commandLine, "--psdFlatPrior");
  if(ppt)psdGaussianPrior=0;
  LALInferenceAddVariable(priorArgs, "psdGaussianPrior", &psdGaussianPrior,  LALINFERENCE_INT4_t,  LALINFERENCE_PARAM_FIXED);

  /********************* TBL: Adding line-removal parameters  *********************/
  UINT4 lines_flag  = 0;   //flag tells likelihood if line-removal is turned on

  #define max(x, y) (((x) > (y)) ? (x) : (y))
  ppt = LALInferenceGetProcParamVal(commandLine, "--removeLines");

  if(ppt)//MARK: Here is where noise line removal parameters are being added to the model
  {
    lines_flag = 1;
    dataPtr = state->data;
    UINT4 lines_num_temp   = 10000;
    UINT4 lines_num_ifo;
    UINT4 lines_num = 0;
    gsl_matrix *lines_temp      = gsl_matrix_alloc(nifo,lines_num_temp);
    gsl_matrix *linewidth_temp  = gsl_matrix_alloc(nifo,lines_num_temp);

    i=0;
    while (dataPtr != NULL)
    {

      for (j = 0; j < lines_num_temp; j++)
      {
        gsl_matrix_set(lines_temp,i,j,-1.0);
        gsl_matrix_set(linewidth_temp,i,j,1.0);
      }

      printf("ifo=%i  %s\n",i,dataPtr->name);fflush(stdout);

      char ifoRemoveLines[500];
      snprintf (ifoRemoveLines,500, "--%s-removeLines",dataPtr->name);

      ppt = LALInferenceGetProcParamVal(commandLine,ifoRemoveLines);
      if(ppt || LALInferenceGetProcParamVal(commandLine, "--chisquaredlines") || LALInferenceGetProcParamVal(commandLine, "--KSlines") || LALInferenceGetProcParamVal(commandLine, "--powerlawlines"))
      /* Load in values from file if requested */
      {
        char line [ 128 ];
        char lines_tempfile[500];

        if (LALInferenceGetProcParamVal(commandLine, "--chisquaredlines")) {
             snprintf (lines_tempfile,500, "%s-ChiSquaredLines.dat",dataPtr->name);
        }
        else if (LALInferenceGetProcParamVal(commandLine, "--KSlines")) {
             snprintf (lines_tempfile,500, "%s-KSLines.dat",dataPtr->name);
        }
        else if (LALInferenceGetProcParamVal(commandLine, "--powerlawlines")) {
             snprintf (lines_tempfile,500, "%s-PowerLawLines.dat",dataPtr->name);
        }
        else {
             char *lines_tempfile_temp = ppt->value;
             strcpy( lines_tempfile, lines_tempfile_temp );
        }
        printf("Reading lines_temp from %s\n",lines_tempfile);

        char * pch;
        j = 0;
        double freqline = 0, freqlinewidth = 0;
        lines_num_ifo = 0;
        FILE *file = fopen ( lines_tempfile, "r" );
        if ( file != NULL )
        {
          while ( fgets ( line, sizeof line, file ) != NULL )
          {

            pch = strtok (line," ");
            int count = 0;
            while (pch != NULL)
            {
                if (count==0) {freqline = atof(pch);}
                if (count==1) {freqlinewidth = atof(pch);}
                pch = strtok (NULL, " ");
                count++;
            }

            gsl_matrix_set(lines_temp,i,j,freqline/df);
            gsl_matrix_set(linewidth_temp,i,j,freqlinewidth/df);
            j++;
            lines_num_ifo++;
          }
          fclose ( file );
        }

      }


      else // Otherwise use defaults
      {
        lines_num_ifo = 10;
        /* top lines_temp_num lines_temp for each interferometer, and widths */
        if(!strcmp(dataPtr->name,"H1"))
        {
          gsl_matrix_set(lines_temp,i,0,35.0/df);   gsl_matrix_set(linewidth_temp,i,0,3.0/df);
          gsl_matrix_set(lines_temp,i,1,45.0/df);   gsl_matrix_set(linewidth_temp,i,1,1.5/df);
          gsl_matrix_set(lines_temp,i,2,51.0/df);   gsl_matrix_set(linewidth_temp,i,2,2.5/df);
          gsl_matrix_set(lines_temp,i,3,60.0/df);   gsl_matrix_set(linewidth_temp,i,3,3.0/df);
          gsl_matrix_set(lines_temp,i,4,72.0/df);   gsl_matrix_set(linewidth_temp,i,4,3.0/df);
          gsl_matrix_set(lines_temp,i,5,87.0/df);   gsl_matrix_set(linewidth_temp,i,5,0.5/df);
          gsl_matrix_set(lines_temp,i,6,108.0/df);  gsl_matrix_set(linewidth_temp,i,6,0.5/df);
          gsl_matrix_set(lines_temp,i,7,117.0/df);  gsl_matrix_set(linewidth_temp,i,7,0.5/df);
          gsl_matrix_set(lines_temp,i,8,122.0/df);  gsl_matrix_set(linewidth_temp,i,8,5.0/df);
          gsl_matrix_set(lines_temp,i,9,180.0/df);  gsl_matrix_set(linewidth_temp,i,9,2.0/df);
        }

        if(!strcmp(dataPtr->name,"L1"))
        {
          gsl_matrix_set(lines_temp,i,0,35.0/df);   gsl_matrix_set(linewidth_temp,i,0,3.0/df);
          gsl_matrix_set(lines_temp,i,1,60.0/df);   gsl_matrix_set(linewidth_temp,i,1,4.0/df);
          gsl_matrix_set(lines_temp,i,2,69.0/df);   gsl_matrix_set(linewidth_temp,i,2,2.5/df);
          gsl_matrix_set(lines_temp,i,3,106.4/df);  gsl_matrix_set(linewidth_temp,i,3,0.8/df);
          gsl_matrix_set(lines_temp,i,4,113.0/df);  gsl_matrix_set(linewidth_temp,i,4,1.5/df);
          gsl_matrix_set(lines_temp,i,5,120.0/df);  gsl_matrix_set(linewidth_temp,i,5,2.5/df);
          gsl_matrix_set(lines_temp,i,6,128.0/df);  gsl_matrix_set(linewidth_temp,i,6,3.5/df);
          gsl_matrix_set(lines_temp,i,7,143.0/df);  gsl_matrix_set(linewidth_temp,i,7,1.0/df);
          gsl_matrix_set(lines_temp,i,8,180.0/df);  gsl_matrix_set(linewidth_temp,i,8,2.5/df);
          gsl_matrix_set(lines_temp,i,9,191.5/df);  gsl_matrix_set(linewidth_temp,i,9,4.0/df);
        }

        if(!strcmp(dataPtr->name,"V1"))
        {
          gsl_matrix_set(lines_temp,i,0,35.0/df);   gsl_matrix_set(linewidth_temp,i,0,3.0/df);
          gsl_matrix_set(lines_temp,i,1,60.0/df);   gsl_matrix_set(linewidth_temp,i,1,4.0/df);
          gsl_matrix_set(lines_temp,i,2,69.0/df);   gsl_matrix_set(linewidth_temp,i,2,2.5/df);
          gsl_matrix_set(lines_temp,i,3,106.4/df);  gsl_matrix_set(linewidth_temp,i,3,0.8/df);
          gsl_matrix_set(lines_temp,i,4,113.0/df);  gsl_matrix_set(linewidth_temp,i,4,1.5/df);
          gsl_matrix_set(lines_temp,i,5,120.0/df);  gsl_matrix_set(linewidth_temp,i,5,2.5/df);
          gsl_matrix_set(lines_temp,i,6,128.0/df);  gsl_matrix_set(linewidth_temp,i,6,3.5/df);
          gsl_matrix_set(lines_temp,i,7,143.0/df);  gsl_matrix_set(linewidth_temp,i,7,1.0/df);
          gsl_matrix_set(lines_temp,i,8,180.0/df);  gsl_matrix_set(linewidth_temp,i,8,2.5/df);
          gsl_matrix_set(lines_temp,i,9,191.5/df);  gsl_matrix_set(linewidth_temp,i,9,4.0/df);
        }
      }
      dataPtr = dataPtr->next;
      i++;

      lines_num = max(lines_num,lines_num_ifo);

    }

    gsl_matrix *lines      = gsl_matrix_alloc(nifo,lines_num);
    gsl_matrix *linewidth  = gsl_matrix_alloc(nifo,lines_num);

    for (i = 0; i < nifo; i++)
    {
      for (j = 0; j < lines_num; j++)
      {
        gsl_matrix_set(lines,i,j,gsl_matrix_get(lines_temp,i,j));
        gsl_matrix_set(linewidth,i,j,gsl_matrix_get(linewidth_temp,i,j));
      }
    }

    /* Add line matrices to variable lists */
    LALInferenceAddVariable(model->params, "line_center", &lines,     LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(model->params, "line_width",  &linewidth, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_FIXED);


  }//End of line-removal initialization
   
  LALInferenceAddVariable(model->params, "removeLinesFlag", &lines_flag, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  if(LALInferenceGetProcParamVal(commandLine, "--glitchFit")) LALInferenceInitGlitchVariables(state, model->params);

  UINT4 signal_flag=1;
  ppt = LALInferenceGetProcParamVal(commandLine, "--noiseonly");
  if(ppt)signal_flag=0;
  LALInferenceAddVariable(model->params, "signalModelFlag", &signal_flag,  LALINFERENCE_INT4_t,  LALINFERENCE_PARAM_FIXED);

  //Only add waveform parameters to model if needed
  if(signal_flag)
  {
    ppt=LALInferenceGetProcParamVal(commandLine,"--fixMc");
    if(ppt){
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "chirpmass", start_mc, mcMin, mcMax, LALINFERENCE_PARAM_FIXED);
      if(lalDebugLevel>0) fprintf(stdout,"chirpmass fixed and set to %f\n",start_mc);
    }else{
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "chirpmass", start_mc, mcMin, mcMax, LALINFERENCE_PARAM_LINEAR);
    }

    /* Check if running with symmetric (eta) or asymmetric (q) mass ratio.*/
    ppt=LALInferenceGetProcParamVal(commandLine,"--fixQ");
    if(ppt){
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "asym_massratio", start_q, qMin, qMax, LALINFERENCE_PARAM_FIXED);
      if(lalDebugLevel>0) fprintf(stdout,"q fixed and set to %f\n",start_q);
    }else{
      ppt=LALInferenceGetProcParamVal(commandLine,"--fixEta");
      if(ppt){
	LALInferenceRegisterUniformVariableREAL8(state, model->params, "massratio", start_eta, etaMin, etaMax, LALINFERENCE_PARAM_FIXED);
        if(lalDebugLevel>0) fprintf(stdout,"eta fixed and set to %f\n",start_eta);
      }else{
        ppt=LALInferenceGetProcParamVal(commandLine,"--symMassRatio");
        if(ppt){
	  LALInferenceRegisterUniformVariableREAL8(state, model->params, "massratio", start_eta, etaMin, etaMax, LALINFERENCE_PARAM_LINEAR);
        }else{
	  LALInferenceRegisterUniformVariableREAL8(state, model->params, "asym_massratio", start_q, qMin, qMax, LALINFERENCE_PARAM_LINEAR);
        }
      }
    }
    
    ppt=LALInferenceGetProcParamVal(commandLine,"--fixTime");
    if(ppt){
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "time", timeParam, timeMin, timeMax, LALINFERENCE_PARAM_FIXED);
      if(lalDebugLevel>0) fprintf(stdout,"time fixed and set to %f\n",timeParam);
    }else{
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "time", timeParam, timeMin, timeMax, LALINFERENCE_PARAM_LINEAR);
    }
    
    /* If we are marginalising over the time, remove that variable from the model (having set the prior above) */
    /* Also set the prior in model->params, since Likelihood can't access the state! (ugly hack) */
    if(LALInferenceGetProcParamVal(commandLine,"--margtime") || LALInferenceGetProcParamVal(commandLine, "--margtimephi")){
        LALInferenceVariableItem *p=LALInferenceGetItem(state->priorArgs,"time_min");
        LALInferenceAddVariable(model->params,"time_min",p->value,p->type,p->vary);
        p=LALInferenceGetItem(state->priorArgs,"time_max");
        LALInferenceAddVariable(model->params,"time_max",p->value,p->type,p->vary);
        LALInferenceRemoveVariable(model->params,"time");
    }

    if(!LALInferenceGetProcParamVal(commandLine,"--margphi") && !LALInferenceGetProcParamVal(commandLine, "--margtimephi")){
      ppt=LALInferenceGetProcParamVal(commandLine,"--fixPhi");
      if(ppt){
	LALInferenceRegisterUniformVariableREAL8(state, model->params, "phase", start_phase, phiMin, phiMax, LALINFERENCE_PARAM_FIXED);
	if(lalDebugLevel>0) fprintf(stdout,"phase fixed and set to %f\n",start_phase);
      }else{
	LALInferenceRegisterUniformVariableREAL8(state, model->params, "phase", start_phase, phiMin, phiMax, LALINFERENCE_PARAM_CIRCULAR);
      }
    }
  
  if(LALInferenceGetProcParamVal(commandLine,"--use-logdistance")){
    /* Check for distance priors on command line */
    if((ppt=LALInferenceGetProcParamVal(commandLine,"--distance-max"))) Dmax=atof(ppt->value);
    if((ppt=LALInferenceGetProcParamVal(commandLine,"--distance-min"))) Dmin=atof(ppt->value);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "logdistance", log(start_dist), log(Dmin), log(Dmax), LALInferenceGetProcParamVal(commandLine,"--fixDist")?LALINFERENCE_PARAM_FIXED:LALINFERENCE_PARAM_LINEAR);
  } else {
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "distance", start_dist, Dmin, Dmax, LALInferenceGetProcParamVal(commandLine,"--fixDist")?LALINFERENCE_PARAM_FIXED:LALINFERENCE_PARAM_LINEAR);
  }
  
  LALInferenceRegisterUniformVariableREAL8(state, model->params, "rightascension", start_ra, raMin, raMax, LALInferenceGetProcParamVal(commandLine,"--fixRa")?LALINFERENCE_PARAM_FIXED:LALINFERENCE_PARAM_CIRCULAR);

  LALInferenceRegisterUniformVariableREAL8(state, model->params, "declination", start_dec, decMin, decMax, LALInferenceGetProcParamVal(commandLine,"--fixDec")?LALINFERENCE_PARAM_FIXED:LALINFERENCE_PARAM_LINEAR);
  
  LALInferenceRegisterUniformVariableREAL8(state, model->params, "polarisation", start_psi, psiMin, psiMax, LALInferenceGetProcParamVal(commandLine,"--fixPsi")?LALINFERENCE_PARAM_FIXED:LALINFERENCE_PARAM_LINEAR);

  /* PPE parameters */

  ppt=LALInferenceGetProcParamVal(commandLine, "--TaylorF2ppE");
  if(approx==TaylorF2 && ppt){
    
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppealpha",0.0, -1000.0 , 1000.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppebeta", 0.0, -1000.0 , 1000.0 , LALINFERENCE_PARAM_LINEAR);
    

    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppeuppera", 0.0, -3.0, 3.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppeupperb", 0.0, -3.0, 3.0 , LALINFERENCE_PARAM_LINEAR);
    

    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppelowera", 0.0, -3.0, 2.0/3.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppelowerb", 0.0, -4.5, 1.0, LALINFERENCE_PARAM_LINEAR);
    
  }

  if(LALInferenceGetProcParamVal(commandLine,"--tidalT")&&LALInferenceGetProcParamVal(commandLine,"--tidal")){
    XLALPrintError("Error: cannot use both --tidalT and --tidal.\n");
    XLAL_ERROR_NULL(XLAL_EINVAL);
  } else if(LALInferenceGetProcParamVal(commandLine,"--tidalT")){
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "lambdaT", 0.0, lambdaTMin, lambdaTMax, LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "dLambdaT", 0.0, dLambdaTMin, dLambdaTMax, LALINFERENCE_PARAM_LINEAR);
    
  } else if(LALInferenceGetProcParamVal(commandLine,"--tidal")){
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "lambda1", 0.0, lambda1Min, lambda1Max, LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "lambda2", 0.0, lambda2Min, lambda2Max, LALINFERENCE_PARAM_LINEAR);
    
  }

  LALSimInspiralSpinOrder spinO = LAL_SIM_INSPIRAL_SPIN_ORDER_ALL;
  ppt=LALInferenceGetProcParamVal(commandLine, "--spinOrder");
  if(ppt) {
    spinO = atoi(ppt->value);
    LALInferenceAddVariable(model->params, "spinO", &spinO,
        LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  }
  LALSimInspiralTidalOrder tideO = LAL_SIM_INSPIRAL_TIDAL_ORDER_ALL;
  ppt=LALInferenceGetProcParamVal(commandLine, "--tidalOrder");
  if(ppt) {
    tideO = atoi(ppt->value);
    LALInferenceAddVariable(model->params, "tideO", &tideO,
        LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  }

  model->waveFlags = XLALSimInspiralCreateWaveformFlags();
  XLALSimInspiralSetSpinOrder(model->waveFlags,  spinO);
  XLALSimInspiralSetTidalOrder(model->waveFlags, tideO);

  fprintf(stdout,"\n\n---\t\t ---\n");
  LALInferenceInitSpinVariables(state, model);

  if (injTable)
     print_flags_orders_warning(injTable,commandLine); 

     /* Print info about orders and waveflags used for templates */

     fprintf(stdout,"Templates will run using Approximant %i (%s), phase order %i, amp order %i, spin order %i tidal order %i, in the %s domain.\n",approx,XLALGetStringFromApproximant(approx),PhaseOrder,AmpOrder,(int) spinO, (int) tideO, model->domain==LAL_SIM_DOMAIN_TIME?"time":"frequency");
     fprintf(stdout,"---\t\t ---\n\n");
  }//end of signal only flag
  else
  {
    /* Print info about orders and waveflags used for templates */
    fprintf(stdout,"\n\n------\n");
    fprintf(stdout,"Noise only run\n");
    fprintf(stdout,"------\n\n");
  }

  /* Initialize waveform buffers */
  model->timehPlus  = XLALCreateREAL8TimeSeries("timehPlus",
                                                &(state->data->timeData->epoch),
                                                0.0,
                                                model->deltaT,
                                                &lalDimensionlessUnit,
                                                state->data->timeData->data->length);
  model->timehCross = XLALCreateREAL8TimeSeries("timehCross",
                                                &(state->data->timeData->epoch),
                                                0.0,
                                                model->deltaT,
                                                &lalDimensionlessUnit,
                                                state->data->timeData->data->length);
  model->freqhPlus = XLALCreateCOMPLEX16FrequencySeries("freqhPlus",
                                                &(state->data->freqData->epoch),
                                                0.0,
                                                model->deltaF,
                                                &lalDimensionlessUnit,
                                                state->data->freqData->data->length);
  model->freqhCross = XLALCreateCOMPLEX16FrequencySeries("freqhCross",
                                                &(state->data->freqData->epoch),
                                                0.0,
                                                model->deltaF,
                                                &lalDimensionlessUnit,
                                                state->data->freqData->data->length);

  /* Create arrays for holding single-IFO likelihoods, etc. */
  model->ifo_loglikelihoods = XLALCalloc(nifo, sizeof(REAL8));
  model->ifo_SNRs = XLALCalloc(nifo, sizeof(REAL8));

  /* Choose proper template */
  model->templt = LALInferenceInitCBCTemplate(state);

  /* Use same window and FFT plans on model as data */
  model->window = state->data->window;
  model->timeToFreqFFTPlan = state->data->timeToFreqFFTPlan;
  model->freqToTimeFFTPlan = state->data->freqToTimeFFTPlan;

  /* Initialize waveform cache */
  model->waveformCache = XLALCreateSimInspiralWaveformCache();

  return(model);
}



/* Setup the variable for the evidence calculation test for review */
/* 5-sigma ranges for analytic likeliood function */
/* https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/LALInferenceReviewAnalyticGaussianLikelihood */
LALInferenceModel *LALInferenceInitModelReviewEvidence(LALInferenceRunState *state)
{
    ProcessParamsTable *commandLine=state->commandLine;
    ProcessParamsTable *ppt=NULL;
    char **strings=NULL;
    char *pinned_params=NULL;
    UINT4 N=0,i,j;
    if((ppt=LALInferenceGetProcParamVal(commandLine,"--pinparams"))){
            pinned_params=ppt->value;
            LALInferenceVariables tempParams;
            memset(&tempParams,0,sizeof(tempParams));
            LALInferenceParseCharacterOptionString(pinned_params,&strings,&N);
    }

    LALInferenceModel *model = XLALCalloc(1, sizeof(LALInferenceModel));
    model->params = XLALCalloc(1, sizeof(LALInferenceVariables));

	i=0;

	struct varSettings {const char *name; REAL8 val, min, max;};
	
	struct varSettings setup[]=
	{
		{.name="time", .val=0.0, .min=-0.1073625, .max=0.1073625},
		{.name="m1", .val=16., .min=14.927715, .max=17.072285},
		{.name="m2", .val=7., .min=5.829675, .max=8.170325},
		{.name="distance", .val=50., .min=37.986000000000004, .max=62.013999999999996},
		{.name="inclination", .val=LAL_PI/2., .min=1.4054428267948966, .max=1.7361498267948965},
		{.name="phase", .val=LAL_PI, .min=2.8701521535897934, .max=3.413033153589793},
		{.name="polarisation", .val=LAL_PI/2., .min=1.3885563267948966, .max=1.7530363267948965},
		{.name="rightascension", .val=LAL_PI, .min=2.813050153589793, .max=3.4701351535897933},
		{.name="declination", .val=0., .min=-0.300699, .max=0.300699},
		{.name="a_spin1", .val=0.5, .min=0.3784565, .max=0.6215435},
		{.name="a_spin2", .val=0.5, .min=0.421869, .max=0.578131},
		{.name="theta_spin1", .val=LAL_PI/2., .min=1.3993998267948966, .max=1.7421928267948965},
		{.name="theta_spin2", .val=LAL_PI/2., .min=1.4086158267948965, .max=1.7329768267948966},
		{.name="phi_spin1", .val=LAL_PI, .min=2.781852653589793, .max=3.501332653589793},
		{.name="phi_spin2", .val=LAL_PI, .min=2.777215653589793, .max=3.5059696535897933},
		{.name="END", .val=0., .min=0., .max=0.}
	};

	while(strcmp("END",setup[i].name))
	{
        LALInferenceParamVaryType type=LALINFERENCE_PARAM_CIRCULAR;
        /* Check if it is to be fixed */
        for(j=0;j<N;j++) if(!strcmp(setup[i].name,strings[j])) {type=LALINFERENCE_PARAM_FIXED; printf("Fixing parameter %s\n",setup[i].name); break;}
		LALInferenceRegisterUniformVariableREAL8(state, model->params, setup[i].name, setup[i].val, setup[i].min, setup[i].max, type);
		i++;
	}

	return(model);
}


LALInferenceModel *LALInferenceInitModelReviewEvidence_bimod(LALInferenceRunState *state)
{
  ProcessParamsTable *commandLine=state->commandLine;
  ProcessParamsTable *ppt=NULL;
  char **strings=NULL;
  char *pinned_params=NULL;
  UINT4 N=0,i,j;
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--pinparams"))){
    pinned_params=ppt->value;
    LALInferenceVariables tempParams;
    memset(&tempParams,0,sizeof(tempParams));
    LALInferenceParseCharacterOptionString(pinned_params,&strings,&N);
  }

  LALInferenceModel *model = XLALCalloc(1, sizeof(LALInferenceModel));
  model->params = XLALCalloc(1, sizeof(LALInferenceVariables));

  i=0;
  
  struct varSettings {const char *name; REAL8 val, min, max;};
  
  struct varSettings setup[]=
  {
    {.name="time", .val=0.05589, .min=-0.1373625, .max=0.2491425},
    {.name="m1", .val=16.857828, .min=14.927715, .max=18.787941},
    {.name="m2", .val=7.93626, .min=5.829675, .max=10.042845},
    {.name="distance", .val=34.6112, .min=12.986, .max=56.2364},
    {.name="inclination", .val=0.9176809634, .min=0.6200446634, .max=1.2153172634},
    {.name="phase", .val=1.7879487268, .min=1.2993558268, .max=2.2765416268},
    {.name="polarisation", .val=0.9311901634, .min=0.6031581634, .max=1.2592221634},
    {.name="rightascension", .val=1.8336303268, .min=1.2422538268, .max=2.4250068268},
    {.name="declination", .val=-0.5448389634, .min=-1.0860971634, .max=-0.0035807634},
    {.name="a_spin1", .val=0.2972348, .min=0.0784565, .max=0.5160131},
    {.name="a_spin2", .val=0.2625048, .min=0.121869, .max=0.4031406},
    {.name="theta_spin1", .val=0.9225153634, .min=0.6140016634, .max=1.2310290634},
    {.name="theta_spin2", .val=0.9151425634, .min=0.6232176634, .max=1.2070674634},
    {.name="phi_spin1", .val=1.8585883268, .min=1.2110563268, .max=2.5061203268},
    {.name="phi_spin2", .val=1.8622979268, .min=1.2064193268, .max=2.5181765268},
    {.name="END", .val=0., .min=0., .max=0.}
  };
  
  while(strcmp("END",setup[i].name))
  {
    LALInferenceParamVaryType type=LALINFERENCE_PARAM_CIRCULAR;
    /* Check if it is to be fixed */
    for(j=0;j<N;j++) if(!strcmp(setup[i].name,strings[j])) {type=LALINFERENCE_PARAM_FIXED; printf("Fixing parameter %s\n",setup[i].name); break;}
    LALInferenceRegisterUniformVariableREAL8(state, model->params, setup[i].name, setup[i].val, setup[i].min, setup[i].max, type);
    i++;
  }
  return(model);
}

LALInferenceModel *LALInferenceInitModelReviewEvidence_banana(LALInferenceRunState *state)
{
  ProcessParamsTable *commandLine=state->commandLine;
  ProcessParamsTable *ppt=NULL;
  char **strings=NULL;
  char *pinned_params=NULL;
  UINT4 N=0,i,j;
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--pinparams"))){
    pinned_params=ppt->value;
    LALInferenceVariables tempParams;
    memset(&tempParams,0,sizeof(tempParams));
    LALInferenceParseCharacterOptionString(pinned_params,&strings,&N);
  }

  LALInferenceModel *model = XLALCalloc(1, sizeof(LALInferenceModel));
  model->params = XLALCalloc(1, sizeof(LALInferenceVariables));

  i=0;
  
  struct varSettings {const char *name; REAL8 val, min, max;};
  
  struct varSettings setup[]=
  {
    {.name="time", .val=0.0, .min=-2., .max=2.},
    {.name="m1", .val=16., .min=14., .max=18.},
    {.name="m2", .val=7., .min=5., .max=9.},
    {.name="distance", .val=50., .min=45., .max=55.},
    {.name="inclination", .val=LAL_PI/2., .min=-0.429203673, .max=3.570796327},
    {.name="phase", .val=LAL_PI, .min=1.141592654, .max=5.141592654},
    {.name="polarisation", .val=LAL_PI/2., .min=-0.429203673, .max=3.570796327},
    {.name="rightascension", .val=LAL_PI, .min=1.141592654, .max=5.141592654},
    {.name="declination", .val=0., .min=-2., .max=2.},
    {.name="a_spin1", .val=0.5, .min=-1.5, .max=2.5},
    {.name="a_spin2", .val=0.5, .min=-1.5, .max=2.5},
    {.name="theta_spin1", .val=LAL_PI/2., .min=-0.429203673, .max=3.570796327},
    {.name="theta_spin2", .val=LAL_PI/2., .min=-0.429203673, .max=3.570796327},
    {.name="phi_spin1", .val=LAL_PI, .min=1.141592654, .max=5.141592654},
    {.name="phi_spin2", .val=LAL_PI, .min=1.141592654, .max=5.141592654},
    {.name="END", .val=0., .min=0., .max=0.}
  };
  
  while(strcmp("END",setup[i].name))
  {
    LALInferenceParamVaryType type=LALINFERENCE_PARAM_CIRCULAR;
    /* Check if it is to be fixed */
    for(j=0;j<N;j++) if(!strcmp(setup[i].name,strings[j])) {type=LALINFERENCE_PARAM_FIXED; printf("Fixing parameter %s\n",setup[i].name); break;}
    LALInferenceRegisterUniformVariableREAL8(state, model->params, setup[i].name, setup[i].val, setup[i].min, setup[i].max, type);
    i++;
  }
  return(model);
}

static void print_flags_orders_warning(SimInspiralTable *injt, ProcessParamsTable *commline){

    /* If lalDebugLevel > 0, print information about:
     * 
     * - Eventual injection/template mismatch on phase and amplitude orders, as well as on waveFlags
     * - Those fiels being set only for injection or template
     * 
     **/
    XLALPrintWarning("\n");
    LALPNOrder PhaseOrder=-1;
    LALPNOrder AmpOrder=-1;
    LALSimInspiralSpinOrder default_spinO = LAL_SIM_INSPIRAL_SPIN_ORDER_ALL;
    LALSimInspiralTidalOrder default_tideO = LAL_SIM_INSPIRAL_TIDAL_ORDER_ALL;
    Approximant approx=NumApproximants;
    ProcessParamsTable *ppt=NULL;
    ProcessParamsTable *ppt_order=NULL;
    int errnum;
    ppt=LALInferenceGetProcParamVal(commline,"--approximant");
    if(ppt){
        approx=XLALGetApproximantFromString(ppt->value);
        ppt=LALInferenceGetProcParamVal(commline,"--order");
        if(ppt) PhaseOrder = XLALGetOrderFromString(ppt->value);
    }
    ppt=LALInferenceGetProcParamVal(commline,"--approx");
    if(ppt){
       approx=XLALGetApproximantFromString(ppt->value);
       XLAL_TRY(PhaseOrder = XLALGetOrderFromString(ppt->value),errnum);
       if( (int) PhaseOrder == XLAL_FAILURE || errnum) {
          XLALPrintWarning("WARNING: No phase order given.  Using maximum available order for the template.\n");
          PhaseOrder=-1;
        }
     }
     /* check approximant is given */
    if (approx==NumApproximants){
        approx=XLALGetApproximantFromString(injt->waveform);
        XLALPrintWarning("WARNING: You did not provide an approximant for the templates. Using value in injtable (%s), which might not what you want!\n",XLALGetStringFromApproximant(approx));
     }
    
    /* check inj/rec amporder */
    ppt=LALInferenceGetProcParamVal(commline,"--amporder");
    if(ppt) AmpOrder=atoi(ppt->value);
    ppt=LALInferenceGetProcParamVal(commline,"--ampOrder");
    if(ppt) AmpOrder = XLALGetOrderFromString(ppt->value);
    if(AmpOrder!=(LALPNOrder)injt->amp_order)
       XLALPrintWarning("WARNING: Injection specified amplitude order %i. Template will use  %i\n",
               injt->amp_order,AmpOrder);

    /* check inj/rec phase order */
    if(PhaseOrder!=(LALPNOrder)XLALGetOrderFromString(injt->waveform))
        XLALPrintWarning("WARNING: Injection specified phase order %i. Template will use %i\n",\
             XLALGetOrderFromString(injt->waveform),PhaseOrder);

    /* check inj/rec spinflag */
    ppt=LALInferenceGetProcParamVal(commline, "--spinOrder");
    ppt_order=LALInferenceGetProcParamVal(commline, "--inj-spinOrder");
    if (ppt && ppt_order){
       if (!(atoi(ppt->value)== atoi(ppt_order->value)))
            XLALPrintWarning("WARNING: Set different spin orders for injection (%i ) and template (%i) \n",atoi(ppt_order->value),atoi(ppt->value));       
    }
    else if (ppt || ppt_order){
        if (ppt)
            XLALPrintWarning("WARNING: You set the spin order only for the template (%i). Injection will use default value (%i). You can change that with --inj-spinOrder. \n",atoi(ppt->value),default_spinO);   
        else 
            XLALPrintWarning("WARNING: You set the spin order only for the injection (%i). Template will use default value (%i). You can change that with --spinOrder. \n",atoi(ppt_order->value),default_spinO);     }
    else
        XLALPrintWarning("WARNING: You did not set the spin order. Injection and template will use default values (%i). You change that using --inj-spinOrder (set injection value) and --spinOrder (set template value).\n",default_spinO);    
    /* check inj/rec tidal flag */
    ppt=LALInferenceGetProcParamVal(commline, "--tidalOrder");
    ppt_order=LALInferenceGetProcParamVal(commline, "--inj-tidalOrder");
    if (ppt && ppt_order){
        if (!(atoi(ppt->value)==atoi( ppt_order->value)))
            XLALPrintWarning("WARNING: Set different tidal orders for injection (%i ) and template (%i) \n",atoi(ppt_order->value),atoi(ppt->value));   
    }
    else if (ppt || ppt_order){
        if (ppt)
            XLALPrintWarning("WARNING: You set the tidal order only for the template (%d). Injection will use default value (%i). You can change that with --inj-tidalOrder. \n",atoi(ppt->value),default_tideO);        
        else 
            XLALPrintWarning("WARNING: You set the tidal order only for the injection (%i). Template will use default value (%i). You can  change that with --tidalOrder\n",atoi(ppt_order->value),default_tideO);
        }
    else
       XLALPrintWarning("WARNING: You did not set the tidal order. Injection and template will use default values (%i). You change that using --inj-tidalOrder (set injection value) and --tidalOrder (set template value).\n",default_tideO); 
    return;    
}

void LALInferenceCheckOptionsConsistency(ProcessParamsTable *commandLine)

{ /*
  Go through options and check for possible errors and inconsistencies (e.g. seglen < 0 )
  
  */
  
  ProcessParamsTable *ppt=NULL,*ppt2=NULL;
  REAL8 tmp=0.0;
  INT4 itmp=0;
  
  ppt=LALInferenceGetProcParamVal(commandLine,"--help");
  if (ppt)
    return;
    
  // Check PSDlength > 0
  ppt=LALInferenceGetProcParamVal(commandLine,"--psdlength");
  if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--PSDlength");
  tmp=atof(ppt->value);
  if (tmp<0.0){
    fprintf(stderr,"ERROR: PSD length must be positive. Exiting...\n");
    exit(1);
  }
  // Check seglen > 0
  REAL8 seglen=0.;
  ppt=LALInferenceGetProcParamVal(commandLine,"--seglen");
  if (!ppt){
    XLALPrintError("Must provide segment length with --seglen. Exiting...");
    exit(1);
  }
  else seglen=atof(ppt->value);
  
  tmp=atof(ppt->value);
  if (tmp<0.0){
    fprintf(stderr,"ERROR: seglen must be positive. Exiting...\n");
    exit(1);
  }
  REAL8 timeSkipStart=0.;
  REAL8 timeSkipEnd=0.;
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--time-pad-start")))
     timeSkipStart=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--time-pad-end")))
     timeSkipEnd=atof(ppt->value);
  if(timeSkipStart+timeSkipEnd > seglen)
  {
    fprintf(stderr,"ERROR: --time-pad-start + --time-pad-end is greater than --seglen!");
    exit(1);
  }
     
  /* Flags consistency */
  ppt=LALInferenceGetProcParamVal(commandLine,"--disable-spin");
  ppt2=LALInferenceGetProcParamVal(commandLine,"--noSpin");
  if (ppt || ppt2){
    ppt2=LALInferenceGetProcParamVal(commandLine,"--spinO");
    if (ppt2){
      itmp=atoi(ppt2->value);
      if (itmp>0 || itmp==-1)
        XLALPrintWarning("--spinO > 0 or -1 will be ignored due to --disable-spin. If you want to include spin terms in the template, remove --disable-spin\n");
        exit(1);
      }
    if (!ppt2){
      XLALPrintWarning("--spinO defaulted to -1. This will be ignored due to --disable-spin. If you want to include spin terms in the template, remove --disable-spin\n");
      }
    ppt=LALInferenceGetProcParamVal(commandLine, "--spinAligned");
    ppt2=LALInferenceGetProcParamVal(commandLine,"--aligned-spin");
    if (ppt|| ppt2){
      fprintf(stderr,"--aligned-spin and --disable-spin are incompatible options. Exiting\n");
      exit(1);
    }
    ppt=LALInferenceGetProcParamVal(commandLine,"--singleSpin");
    if(ppt){
      fprintf(stderr,"--singleSpin and --disable-spin are incompatible options. Exiting\n");
      exit(1);
    }
  }
  
  /* lalinference_nest only checks */
  // Check live points
  ppt=LALInferenceGetProcParamVal(commandLine,"--nlive");
  if (ppt){
    itmp=atoi(ppt->value);
    if (itmp<0){
      fprintf(stderr,"ERROR: nlive must be positive. Exiting...\n");
      exit(1);
    }
    if (itmp<100){
      XLALPrintWarning("WARNING: Using %d live points. This is very little and may lead to unreliable results. Consider increasing.\n",itmp);
    }
    if (itmp>5000){
      XLALPrintWarning("WARNING: Using %d live points. This is a very large number and may lead to very long runs. Consider decreasing.\n",itmp);
    }
  }
  // Check nmcmc points
  ppt=LALInferenceGetProcParamVal(commandLine,"--nmcmc");
  if (ppt){
    itmp=atoi(ppt->value);
    if (itmp<0){
      fprintf(stderr,"ERROR: nmcmc must be positive (or omitted). Exiting...\n");
      exit(1);
    }
    if (itmp<100){
      XLALPrintWarning("WARNING: Using %d nmcmc. This is very little and may lead to unreliable results. Consider increasing.\n",itmp);
    }
  }
  
  return;
}

void LALInferenceInitSpinVariables(LALInferenceRunState *state, LALInferenceModel *model){


  LALStatus status;
  memset(&status,0,sizeof(status));

  ProcessParamsTable *commandLine=state->commandLine;
  ProcessParamsTable *ppt=NULL;

  LALInferenceFrame frame = LALINFERENCE_FRAME_SYSTEM;
  ppt=LALInferenceGetProcParamVal(commandLine, "--radiation-frame");
  if(ppt){
    frame = LALINFERENCE_FRAME_RADIATION;
  }
  LALInferenceAddVariable(model->params, "LALINFERENCE_FRAME", &frame, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);

  Approximant approx= *(Approximant*) LALInferenceGetVariable(model->params, "LAL_APPROXIMANT");

  REAL8 a1min=0.0,a1max=1.0;
  REAL8 a2min=0.0,a2max=1.0;
  REAL8 tilt1min=0.0,tilt1max=LAL_PI;
  REAL8 tilt2min=0.0,tilt2max=LAL_PI;
  REAL8 phi12min=0.0,phi12max=LAL_TWOPI;
  REAL8 theta1min=0.0,theta1max=LAL_PI;
  REAL8 theta2min=0.0,theta2max=LAL_PI;
  REAL8 phi1min=0.0,phi1max=LAL_TWOPI;
  REAL8 phi2min=0.0,phi2max=LAL_TWOPI;
  REAL8 thetaJNmin=0.0,thetaJNmax=LAL_PI;
  REAL8 iotaMin=0.0,iotaMax=LAL_PI;
  REAL8 phiJLmin=0.0,phiJLmax=LAL_TWOPI;

  /* Default to precessing spins */
  UINT4 spinAligned=0;
  UINT4 singleSpin=0;
  UINT4 noSpin=0;

  /* Let's first check that the user asked, then we check the approximant can make it happen */
  /* Check for spin disabled */
  ppt=LALInferenceGetProcParamVal(commandLine, "--noSpin");
  if (!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--disable-spin");
  if (ppt) noSpin=1;

  /* Check for aligned spin */
  ppt=LALInferenceGetProcParamVal(commandLine, "--spinAligned");
  if (!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--aligned-spin");
  if(ppt)
    spinAligned=1;

  /* Check for single spin */
  ppt=LALInferenceGetProcParamVal(commandLine,"--singleSpin");
  if(ppt){
    singleSpin=1;
  }

  SpinSupport spin_support=XLALSimInspiralGetSpinSupportFromApproximant(approx);
  LALSimInspiralFrameAxis frame_axis=LAL_SIM_INSPIRAL_FRAME_AXIS_VIEW;

  /* Now check what the approx can do and eventually change user's choices to comply.
   * Also change the reference frame -- For the moment use default as the corresponding patch to LALSimulation has not been finished yet */
  if (spin_support==LAL_SIM_INSPIRAL_SPINLESS)
    noSpin=1;
  else if (spin_support==LAL_SIM_INSPIRAL_SINGLESPIN)
    singleSpin=1;
  else if (spin_support==LAL_SIM_INSPIRAL_ALIGNEDSPIN){
    spinAligned=1;
    /* Restore this line when LALSim has routine to convert frames:
     * frame_axis= LAL_SIM_INSPIRAL_FRAME_AXIS_ORBITAL_L;
     */
  }

  /* Add the frame to waveFlagas */
  XLALSimInspiralSetFrameAxis(model->waveFlags,frame_axis);

  if (spinAligned){
  /* If spin aligned the magnitude is in the range [-1,1] */
    a1min=-1.0;
    a2min=-1.0;
  }

  /* IMRPhenomP only supports spins up to 0.9 and q > 1/20. Set prior consequently*/
  if (approx==IMRPhenomP && (a1max>0.9 || a2max>0.9)){
    a1max=0.9;
    a2max=0.9;
    if (spinAligned){
      a1min=-0.9;
      a2min=-0.9;
      }
    XLALPrintWarning("WARNING: IMRPhenomP only supports spin magnitude up to 0.9. Changing the a1max=a2max=0.9.\n");
  }

  /* Start with parameters that are free (or pinned if the user wants so). The if...else below may force some of them to be fixed or ignore some of them, depending on the spin configuration*/
  LALInferenceParamVaryType tilt1Vary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType tilt2Vary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType phi12Vary = LALINFERENCE_PARAM_CIRCULAR;
  LALInferenceParamVaryType spin1Vary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType spin2Vary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType phiJLVary = LALINFERENCE_PARAM_CIRCULAR;
  LALInferenceParamVaryType theta1Vary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType theta2Vary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType phi1Vary = LALINFERENCE_PARAM_CIRCULAR;
  LALInferenceParamVaryType phi2Vary = LALINFERENCE_PARAM_CIRCULAR;
  LALInferenceParamVaryType thetaJNVary = LALINFERENCE_PARAM_LINEAR;
  LALInferenceParamVaryType iotaVary = LALINFERENCE_PARAM_LINEAR;

  if(frame==LALINFERENCE_FRAME_RADIATION)
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "inclination", 0.0, iotaMin, iotaMax, iotaVary);
  else
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "theta_JN", 0.0, thetaJNmin, thetaJNmax,thetaJNVary);

  /* We may use system or radiation frame, which give different params name. We take the choice here below */
  if (frame==LALINFERENCE_FRAME_SYSTEM){

    /* Add parameters depending on the values of noSpin, singleSpin and spinAligned
     * noSpin -> add nothing
     * spinAligned -> add a_spin1 and a_spin2 (if the approximant is spin aligned only use the names spin1,spin1 instead)
     * singleSpin -> add a_spin1, tilt_spin1, phi_JL
     * singleSpin+spinAligned -> add a_spin1
     * otherwise -> add everything */
     /* Note: To get spin aligned is sufficient to not add the spin angles because LALInferenceTemplate will default to aligned spin if it doesn't find spin angle params in model->params. */
    if (!noSpin){
      LALInferenceRegisterUniformVariableREAL8(state, model->params, spin_support==LAL_SIM_INSPIRAL_ALIGNEDSPIN?"spin1":"a_spin1", 0.0, a1min, a1max,spin1Vary);
      if (!singleSpin)
        LALInferenceRegisterUniformVariableREAL8(state, model->params, spin_support==LAL_SIM_INSPIRAL_ALIGNEDSPIN?"spin2":"a_spin2", 0.0, a2min, a2max,spin2Vary);
      if (!spinAligned){
        LALInferenceRegisterUniformVariableREAL8(state, model->params, "phi_JL", 0.0, phiJLmin,  phiJLmax, phiJLVary);
        LALInferenceRegisterUniformVariableREAL8(state, model->params, "tilt_spin1", 0.0, tilt1min,tilt1max,tilt1Vary);
        if (!singleSpin){
          LALInferenceRegisterUniformVariableREAL8(state, model->params, "tilt_spin2", 0.0, tilt2min,tilt2max,tilt2Vary);
          LALInferenceRegisterUniformVariableREAL8(state, model->params, "phi12", 0.0, phi12min,phi12max,phi12Vary);
        }
      }
    }
  }
  else if(frame==LALINFERENCE_FRAME_RADIATION) {

    /* Now add parameters depending on the values of noSpin, singleSpin and spinAligned
     * noSpin -> add nothing
     * spinAligned -> add a_spin1 and a_spin2 (if the approximant is spin aligned only use the names spin1,spin1 instead)
     * singleSpin -> add a_spin1, theta_spin1, phi_spin1
     * singleSpin+spinAligned -> add a_spin1
     * otherwise -> add everything */
    /* Note: To get spin aligned is sufficient to not add the spin angles because LALInferenceTemplate will default to aligned spin if it doesn't find spin angle params in model->params. */
    if (!noSpin){
      LALInferenceRegisterUniformVariableREAL8(state, model->params,spin_support==LAL_SIM_INSPIRAL_ALIGNEDSPIN?"spin1":"a_spin1", 0.0, a1min, a1max,spin1Vary);
      if (!singleSpin)
        LALInferenceRegisterUniformVariableREAL8(state, model->params,spin_support==LAL_SIM_INSPIRAL_ALIGNEDSPIN?"spin2":"a_spin2", 0.0, a2min, a2max,spin2Vary);
      if (!spinAligned){
        LALInferenceRegisterUniformVariableREAL8(state,model->params,"theta_spin1", 0.0, theta1min, theta1max, theta1Vary);
        LALInferenceRegisterUniformVariableREAL8(state,model->params,"phi_spin1", 0.0, phi1min, phi1max,phi1Vary);
        if (!singleSpin){
          LALInferenceRegisterUniformVariableREAL8(state, model->params, "theta_spin2", 0.0, theta2min, theta2max,theta2Vary);
          LALInferenceRegisterUniformVariableREAL8(state, model->params, "phi_spin2", 0.0, phi2min,phi2max,  phi2Vary);
        }
      }
    }
  }
  else {
    XLALPrintError("Error: unknown frame %i\n",frame);
    exit(1);
  }

  /* Print to stdout what will be used */
  if (noSpin)
    fprintf(stdout,"Templates will run without spins\n");
  else{
    if (spinAligned && singleSpin)
      fprintf(stdout,"Templates will run with spin 1 aligned to L \n");
    if (spinAligned && !singleSpin)
      fprintf(stdout,"Templates will run with spins aligned to L \n");
    if (!spinAligned && singleSpin)
      fprintf(stdout,"Templates will run with precessing spin 1 \n");
    if (!spinAligned && !singleSpin)
      fprintf(stdout,"Templates will run with precessing spins \n");
  }
}

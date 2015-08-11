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
static void LALInferenceInitMassVariables(LALInferenceRunState *state);
static void LALInferenceCheckApproximantNeeds(LALInferenceRunState *state,Approximant approx);

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
      XLALPrintError("%s", help);
      XLAL_ERROR_NULL(XLAL_EINVAL);
    }
  }
  else if(LALInferenceGetProcParamVal(commandLine,"--LALSimulation")){
    fprintf(stderr,"Warning: --LALSimulation is deprecated, the LALSimulation package is now the default. To use LALInspiral specify:\n\
                    --template LALGenerateInspiral (for time-domain templates)\n\
                    --template LAL (for frequency-domain templates)\n");
  }
  else if(LALInferenceGetProcParamVal(commandLine,"--roq")){
  templt=&LALInferenceTemplateROQ;
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
  REAL8 gmax   = 20.0;

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
  double Anorm;

  REAL8 TwoDeltaToverN = 2.0 * dataPtr->timeData->deltaT / ((double) dataPtr->timeData->data->length);

  Anorm = sqrt(TwoDeltaToverN);
  Amin = 10.0/Anorm;
  Amax = 10000.0/Anorm;

  Qmin = 3.0;
  Qmax = 30.0;
  tmin = 0.0;
  tmax = dataPtr->timeData->data->length*dataPtr->timeData->deltaT;
  f_min = dataPtr->fLow;
  f_max = dataPtr->fHigh;
  pmin = 0.0;
  pmax = LAL_TWOPI;

  gsl_matrix_set_all(mAmp, Amin);
  gsl_matrix_set_all(mf0,  f_min);
  gsl_matrix_set_all(mQ,   Qmin);
  gsl_matrix_set_all(mt0,  tmin);
  gsl_matrix_set_all(mphi, pmin);

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

  LALInferenceAddMinMaxPrior(priorArgs, "glitch_size", &gmin, &gmax, LALINFERENCE_REAL8_t);
  LALInferenceAddMinMaxPrior(priorArgs, "glitch_dim", &gmin, &gmax, LALINFERENCE_REAL8_t);

  LALInferenceAddVariable(priorArgs, "glitch_norm", &Anorm, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);

  //Meyer-wavelet based proposal distribution
  LALInferenceAddVariable(proposalArgs, "glitch_max_power", &maxpower, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(proposalArgs, "glitch_power", &gpower, LALINFERENCE_gslMatrix_t, LALINFERENCE_PARAM_FIXED);

}

static void LALInferenceInitCalibrationVariables(LALInferenceRunState *runState, LALInferenceVariables *currentParams) {
  ProcessParamsTable *ppt = NULL;
  LALInferenceIFOData *ifo = NULL;
  LALInferenceIFOData *dataPtr=NULL;
  UINT4 calOn = 1;
  if ((ppt = LALInferenceGetProcParamVal(runState->commandLine, "--enable-spline-calibration"))){
    /* Use spline to marginalize*/
    UINT4 ncal = 5; /* Number of calibration nodes, log-distributed
		between fmin and fmax. */
    REAL8 ampUncertaintyPrior = 0.1; /* 10% amplitude */
    REAL8 phaseUncertaintyPrior = 5*M_PI/180.0; /* 5 degrees phase */
    if ((ppt = LALInferenceGetProcParamVal(runState->commandLine, "--spcal-nodes"))) {
      ncal = atoi(ppt->value);
    }

    if ((ppt = LALInferenceGetProcParamVal(runState->commandLine, "--spcal-amp-uncertainty"))) {
      ampUncertaintyPrior = atof(ppt->value);
    }

    if ((ppt = LALInferenceGetProcParamVal(runState->commandLine, "--spcal-phase-uncertainty"))) {
      phaseUncertaintyPrior = M_PI/180.0*atof(ppt->value); /* CL arg in degrees, variable in radians */
    }

    LALInferenceAddVariable(runState->priorArgs, "spcal_amp_uncertainty", &ampUncertaintyPrior,
          LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(runState->priorArgs, "spcal_phase_uncertainty", &phaseUncertaintyPrior,
          LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(currentParams, "spcal_active", &calOn, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
    LALInferenceAddVariable(currentParams, "spcal_npts", &ncal, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);

    ifo = runState->data;
    do {
      size_t i;

      char freqVarName[VARNAME_MAX];
      char ampVarName[VARNAME_MAX];
      char phaseVarName[VARNAME_MAX];

      REAL8Vector *logfreqs = NULL;
      REAL8Vector *amps = NULL;
      REAL8Vector *phase = NULL;

      REAL8 fMin = ifo->fLow;
      REAL8 fMax = ifo->fHigh;
      REAL8 logFMin = log(fMin);
      REAL8 logFMax = log(fMax);
      REAL8 dLogF = (logFMax - logFMin)/(ncal-1);


      snprintf(freqVarName, VARNAME_MAX, "%s_spcal_logfreq", ifo->name);
      snprintf(ampVarName, VARNAME_MAX, "%s_spcal_amp", ifo->name);
      snprintf(phaseVarName, VARNAME_MAX, "%s_spcal_phase", ifo->name);

      logfreqs = XLALCreateREAL8Vector(ncal);
      amps = XLALCreateREAL8Vector(ncal);
      phase = XLALCreateREAL8Vector(ncal);

      for (i = 0; i < ncal; i++) {
        logfreqs->data[i] = logFMin + i*dLogF;
        amps->data[i] = 0.0;
        phase->data[i] = 0.0;
      }

      LALInferenceAddVariable(currentParams, freqVarName, &logfreqs, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_FIXED);
      LALInferenceAddVariable(currentParams, ampVarName, &amps, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_LINEAR);
      LALInferenceAddVariable(currentParams, phaseVarName, &phase, LALINFERENCE_REAL8Vector_t, LALINFERENCE_PARAM_LINEAR);

      ifo = ifo->next;

    } while (ifo);
  }
  else if(LALInferenceGetProcParamVal(runState->commandLine, "--MarginalizeConstantCalAmp") ||LALInferenceGetProcParamVal(runState->commandLine, "--MarginalizeConstantCalPha")){
    /* Use constant (in frequency) approximation for the errors */
    if (LALInferenceGetProcParamVal(runState->commandLine, "--MarginalizeConstantCalAmp")){
      /*For the moment the prior ranges are the same for the three IFOs */
      REAL8 camp_max_A=0.25; /* plus minus 25% amplitude errors*/
      REAL8 camp_min_A=-0.25;
      REAL8 zero=0.0;
      dataPtr = runState->data;
      while (dataPtr != NULL){
        char CA_A[10]="";
        sprintf(CA_A,"%s_%s","calamp",dataPtr->name);
        LALInferenceRegisterUniformVariableREAL8(runState, currentParams, CA_A, zero, camp_min_A, camp_max_A, LALINFERENCE_PARAM_LINEAR);
        dataPtr = dataPtr->next;
      }

      LALInferenceAddVariable(currentParams, "constantcal_active", &calOn, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
      /*If user specifies a width for the error prior, a gaussian prior will be used, otherwise a flat prior will be used*/
      REAL8 ampUncertaintyPrior=-1.0;
      ppt = LALInferenceGetProcParamVal(runState->commandLine, "--constcal_ampsigma");
      if (ppt) {
        ampUncertaintyPrior = atof(ppt->value);
      }
      LALInferenceAddVariable(runState->priorArgs, "constcal_amp_uncertainty", &ampUncertaintyPrior, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
    }
    if (LALInferenceGetProcParamVal(runState->commandLine, "--MarginalizeConstantCalPha")){
      /* Add linear calibration phase errors to the measurement. For the moment the prior ranges are the same for the three IFOs */
      REAL8 cpha_max_A=0.349;  /* plus/minus 20 degs*/
      REAL8 cpha_min_A=-0.349;
      REAL8 zero=0.0;
      dataPtr = runState->data;
      while (dataPtr != NULL)
      {
        char CP_A[10]="";
        sprintf(CP_A,"%s_%s","calpha",dataPtr->name);
        LALInferenceRegisterUniformVariableREAL8(runState, currentParams, CP_A, zero, cpha_min_A, cpha_max_A, LALINFERENCE_PARAM_LINEAR);
        dataPtr = dataPtr->next;
      }
      LALInferenceAddVariable(currentParams, "constantcal_active", &calOn, LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);

     /*If user specifies a width for the error prior, a gaussian prior will be used, otherwise a flat prior will be used*/
      REAL8 phaseUncertaintyPrior=-1.0;
      ppt = LALInferenceGetProcParamVal(runState->commandLine, "--constcal_phasigma");
      if (ppt) {
        phaseUncertaintyPrior = M_PI/180.0*atof(ppt->value); /* CL arg in degrees, variable in radians */
      }
      LALInferenceAddVariable(runState->priorArgs, "constcal_phase_uncertainty", &phaseUncertaintyPrior, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);

    }
  }
  else{
    /* No calibration marginalization asked. Just exit */
    return;
  }
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
  /* Mass parameters checks*/
  if (!strcmp(name,"eta"))
    if (max>0.25){
      fprintf(stderr,"ERROR: maximum of eta cannot be larger than 0.25. Check --eta-max\n");
      exit(1);
    }
  if (!strcmp(name,"q")){
    REAL8 qMin=min;
    REAL8 qMax=max;

    if (qMin <= 0.0 || qMin > 1.0)
    {
        fprintf(stderr,"ERROR: qMin must be between 0 and 1, got value qMin=%f\n",qMin);
		exit(1);
    }
    if (qMax > 1.0 || qMax <0.0 || qMax < qMin)
    {
      fprintf(stderr,"ERROR: qMax must be between 0 and 1, and qMax > qMin. Got value qMax=%f, qMin=%f\n",qMax,qMin);
	  exit(1);
    }
  }
  /*End of mass parameters check */

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
(--use-eta)                    Jump in symmetric mass ratio eta, instead of q=m1/m2 (m1>m2).\n\
(--use-logdistance)             Jump in log(distance) instead of distance.\n\
(--approx)                      Specify a template approximant and phase order to use.\n\
                               (default TaylorF2threePointFivePN). Available approximants:\n\
                               default modeldomain=\"time\": GeneratePPN, TaylorT1, TaylorT2, TaylorT3, TaylorT4,\n\
                                                           EOB, EOBNR, EOBNRv2, EOBNRv2HM, SEOBNRv1, SpinTaylor,\n\
                                                           SpinQuadTaylor, SpinTaylorFrameless, SpinTaylorT4,\n\
                                                           PhenSpinTaylorRD, NumRel.\n\
                               default modeldomain=\"frequency\": TaylorF1, TaylorF2, TaylorF2RedSpin, \n\
                                                                TaylorF2RedSpinTidal, IMRPhenomA, IMRPhenomB, IMRPhenomP.\n\
(--amporder PNorder)            Specify a PN order in amplitude to use (defaults: LALSimulation: max available; LALInspiral: newtownian).\n\
(--fref f_ref)                   Specify a reference frequency at which parameters are defined (default 100).\n\
(--use-tidal)                   Enables tidal corrections, only with LALSimulation.\n\
(--use-tidalT)                  Enables reparmeterized tidal corrections, only with LALSimulation.\n\
(--spinOrder PNorder)           Specify twice the PN order (e.g. 5 <==> 2.5PN) of spin effects to use, only for LALSimulation (default: -1 <==> Use all spin effects).\n\
(--tidalOrder PNorder)          Specify twice the PN order (e.g. 10 <==> 5PN) of tidal effects to use, only for LALSimulation (default: -1 <==> Use all tidal effects).\n\
(--modeldomain)                 domain the waveform template will be computed in (\"time\" or \"frequency\"). If not given will use LALSim to decide\n\
(--spinAligned or --aligned-spin)  template will assume spins aligned with the orbital angular momentum.\n\
(--singleSpin)                  template will assume only the spin of the most massive binary component exists.\n\
(--noSpin, --disable-spin)      template will assume no spins (giving this will void spinOrder!=0) \n\
(--detector-frame)              model will use detector-centred coordinates instead of RA,dec\n\
\n\
------------------------------------------------------------------------------------------------------------------\n\
--- Starting Parameters ------------------------------------------------------------------------------------------\n\
------------------------------------------------------------------------------------------------------------------\n\
You can generally have MCMC chains to start from a given parameter value by using --parname VALUE. Names currently known to the code are:\n\
 time                         Waveform time (overrides random about trigtime).\n\
 chirpmas                     Chirpmass\n\
 eta                          Symmetric massratio (needs --use-eta)\n\
 q                            Asymmetric massratio (a.k.a. q=m2/m1 with m1>m2)\n\
 phase                        Coalescence phase.\n\
 costheta_jn                  Cosine of angle between J and line of sight [rads]\n\
 distance                     Distance [Mpc]\n\
 logdistance                  Log Distance (requires --use-logdistance)\n\
 rightascension               Rightascensions\n\
 declination                  Declination.\n\
 polarisation                 Polarisation angle.\n\
* Spin Parameters:\n\
 a_spin1                      Spin1 magnitude\n\
 a_spin2                      Spin2 magnitude\n\
 tilt_spin1                   Angle between spin1 and orbital angular momentum\n\
 tilt_spin2                   Angle between spin2 and orbital angular momentum \n\
 phi_12                       Difference between spins' azimuthal angles \n\
 phi_jl                       Difference between total and orbital angular momentum azimuthal angles\n\
* Equation of State parameters (requires --use-tidal or --use-tidalT):\n\
 lambda1                      lambda1.\n\
 lambda2                      lambda2.\n\
 lambdaT                      lambdaT.\n\
 dLambdaT                     dLambdaT.\n\
------------------------------------------------------------------------------------------------------------------\n\
--- Prior Arguments ----------------------------------------------------------------------------------------------\n\
------------------------------------------------------------------------------------------------------------------\n\
You can generally use --paramname-min MIN --paramname-max MAX to set the prior range for the parameter paramname\n\
The names known to the code are listed below.\n\
Component masses, total mass and time have dedicated options listed here:\n\n\
(--trigtime time)                       Center of the prior for the time variable.\n\
(--comp-min min)                        Minimum component mass (1.0).\n\
(--comp-max max)                        Maximum component mass (30.0).\n\
(--mtotal-min min)                      Minimum total mass (2.0).\n\
(--mtotal-max max)                      Maximum total mass (35.0).\n\
(--dt time)                             Width of time prior, centred around trigger (0.2s).\n\
(--malmquistPrior)                      Rejection sample based on SNR of template \n\
\n\
(--varyFlow, --flowMin, --flowMax)       Allow the lower frequency bound of integration to vary in given range.\n\
(--pinparams)                            List of parameters to set to injected values [mchirp,asym_massratio,etc].\n\
------------------------------------------------------------------------------------------------------------------\n\
--- Fix Parameters ----------------------------------------------------------------------------------------------\n\
------------------------------------------------------------------------------------------------------------------\n\
You can generally fix a parameter to be fixed to a given values by using both --paramname VALUE and --fix-paramname\n\
where the known names have been listed above\n\
\n";

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
  REAL8 f_ref = 100.0;
  LALInferenceIFOData *dataPtr;
  UINT4 event=0;
  UINT4 i=0;
  /* Default priors */
  REAL8 Dmin=1.0;
  REAL8 Dmax=2000.0;
  REAL8 mcMin=1.0;
  REAL8 mcMax=15.3;
  REAL8 etaMin=0.0312;
  REAL8 etaMax=0.25;
  REAL8 qMin=1./30.; // The ratio between min and max component mass (see InitMassVariables)
  REAL8 qMax=1.0;
  REAL8 psiMin=0.0,psiMax=LAL_PI;
  REAL8 decMin=-LAL_PI/2.0,decMax=LAL_PI/2.0;
  REAL8 raMin=0.0,raMax=LAL_TWOPI;
  REAL8 phiMin=0.0,phiMax=LAL_TWOPI;
  REAL8 costhetaJNmin=-1.0 , costhetaJNmax=1.0;
  REAL8 dt=0.1;  /* Half the width of time prior */
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
  REAL8 zero=0.0; /* just a number that will be overwritten anyway*/

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

  LALInferenceModel *model = XLALMalloc(sizeof(LALInferenceModel));
  model->params = XLALCalloc(1, sizeof(LALInferenceVariables));
  memset(model->params, 0, sizeof(LALInferenceVariables));

  UINT4 signal_flag=1;
  ppt = LALInferenceGetProcParamVal(commandLine, "--noiseonly");
  if(ppt)signal_flag=0;
  LALInferenceAddVariable(model->params, "signalModelFlag", &signal_flag,  LALINFERENCE_INT4_t,  LALINFERENCE_PARAM_FIXED);

  if(LALInferenceGetProcParamVal(commandLine,"--malmquistPrior"))
  {
    UINT4 malmquistflag=1;
    LALInferenceAddVariable(model->params, "malmquistPrior",&malmquistflag,LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
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
  if (ppt) f_ref = atof(ppt->value);

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

  /* This sets the component masses and total mass priors, if given in command line.
   * The prior for other parameters are now read in in RegisterUniformVariable, if given by the user. */
  LALInferenceInitMassVariables(state);
  /* now we need to update the chirp mass and q limits accordingly */
  REAL8 comp_min = *(REAL8 *)LALInferenceGetVariable(state->priorArgs,"component_min");
  REAL8 comp_max = *(REAL8 *)LALInferenceGetVariable(state->priorArgs,"component_max");
  REAL8 mtot_min = *(REAL8 *)LALInferenceGetVariable(state->priorArgs,"MTotMin");
  REAL8 mtot_max = *(REAL8 *)LALInferenceGetVariable(state->priorArgs,"MTotMax");
  qMin = comp_min/comp_max;
  mcMin =mtot_min*pow(qMin/pow(1.+qMin,2.),3./5.);
  mcMax =mtot_max*pow(0.25,3./5.);

  /************ Initial Value Related Argument START *************/
  /* Read time parameter from injection file */
  if(injTable)
  {
    endtime=XLALGPSGetREAL8(&(injTable->geocent_end_time));
    fprintf(stdout,"Using end time from injection file: %lf\n", endtime);
  }
  /* Over-ride end time if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--trigtime");
  if(ppt){
    endtime=atof(ppt->value);
    printf("Read end time %f\n",endtime);
  }
  /* Over-ride time prior window if specified */
  ppt=LALInferenceGetProcParamVal(commandLine,"--dt");
  if(ppt)
    dt=atof(ppt->value);
  timeMin=endtime-dt; timeMax=endtime+dt;
  timeParam = timeMin + (timeMax-timeMin)*gsl_rng_uniform(GSLrandom);

  /* Initial Value Related END */
  LALInferenceAddVariable(model->params, "LAL_APPROXIMANT", &approx,        LALINFERENCE_UINT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(model->params, "LAL_PNORDER",     &PhaseOrder,        LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(model->params, "LAL_AMPORDER",     &AmpOrder,        LALINFERENCE_INT4_t, LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(model->params, "f_ref", &f_ref, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);

  /* flow handling */
  REAL8 fLow = state->data->fLow;
  ppt=LALInferenceGetProcParamVal(commandLine,"--vary-flow");
  if(ppt){
    REAL8 fLow_min = fLow;
    REAL8 fLow_max = 200.0;
    if(LALInferenceCheckVariable(model->params,"f_ref"))
      f_ref = *(REAL8*)LALInferenceGetVariable(model->params, "f_ref");
      if (f_ref > 0.0 && fLow_max > f_ref) {
        fprintf(stdout,"WARNING: flow can't go higher than the reference frequency.  Setting flow-max to %f\n",f_ref);
        fLow_max = f_ref;
      }
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "flow", fLow, fLow_min, fLow_max, LALINFERENCE_PARAM_LINEAR);
  } else {
    LALInferenceAddVariable(model->params, "flow", &fLow,  LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_FIXED);
  }

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

  /* Get number of interferometers */
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
  imin = f_min;
  imax = f_max;

  UINT4 j = 0;

  ppt = LALInferenceGetProcParamVal(commandLine, "--psdFit");
  if(ppt)//MARK: Here is where noise PSD parameters are being added to the model
  {

    printf("Setting up PSD fitting for %i ifos...\n",nifo);

    dataPtr = state->data;

    gsl_matrix *bands_min = gsl_matrix_alloc(nifo,nscale_block);
    gsl_matrix *bands_max = gsl_matrix_alloc(nifo,nscale_block);

    i=0;
    while (dataPtr != NULL)
    {
      printf("ifo=%i  %s\n",i,dataPtr->name);fflush(stdout);

        nscale_bin   = (imax+1-imin)/nscale_block;
        nscale_dflog = log( (double)(imax+1)/(double)imin )/(double)nscale_block;

        int freq_min, freq_max;

        for (j = 0; j < nscale_block; j++)
        {

            freq_min = (int) exp(log((double)imin ) + nscale_dflog*j);
            freq_max = (int) exp(log((double)imin ) + nscale_dflog*(j+1));

            gsl_matrix_set(bands_min,i,j,freq_min);
            gsl_matrix_set(bands_max,i,j,freq_max);
        }


      dataPtr = dataPtr->next;
      i++;

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
    LALInferenceAddMinMaxPrior(priorArgs,   "psdscale", &nscale_min,  &nscale_max,   LALINFERENCE_REAL8_t);
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

  if(LALInferenceGetProcParamVal(commandLine, "--glitchFit")) LALInferenceInitGlitchVariables(state, model->params);

  /* Handle, if present, requests for calibration parameters. */
  LALInferenceInitCalibrationVariables(state, model->params);

  //Only add waveform parameters to model if needed
  if(signal_flag)
  {
    /* The idea here is the following:
     * We call RegisterUniformVariable with startval=0 and meanigful min and max values.
     * That function will then take care of setting startval to a random value between min and max, or read a value from command line (with --parname VALUE).
     * The user can fix the param to a given value with --fix-parname --parname VALUE
     * */
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "chirpmass", zero, mcMin, mcMax, LALINFERENCE_PARAM_LINEAR);
    /* Check if running with symmetric (eta) or asymmetric (q) mass ratio.*/
    ppt=LALInferenceGetProcParamVal(commandLine,"--use-eta");
    if(ppt)
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "eta", zero, etaMin, etaMax, LALINFERENCE_PARAM_LINEAR);
    else
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "q", zero, qMin, qMax, LALINFERENCE_PARAM_LINEAR);


    if(!LALInferenceGetProcParamVal(commandLine,"--margphi") && !LALInferenceGetProcParamVal(commandLine, "--margtimephi")){
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "phase", zero, phiMin, phiMax, LALINFERENCE_PARAM_CIRCULAR);
    }

  /* Check for distance prior for use if the user samples in logdistance */
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--distance-max"))) Dmax=atof(ppt->value);
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--distance-min"))) Dmin=atof(ppt->value);

  if(LALInferenceGetProcParamVal(commandLine,"--use-logdistance")){
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "logdistance", zero, log(Dmin), log(Dmax),LALINFERENCE_PARAM_LINEAR);
  } else {
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "distance", zero, Dmin, Dmax, LALINFERENCE_PARAM_LINEAR);
  }
  LALInferenceRegisterUniformVariableREAL8(state, model->params, "polarisation", zero, psiMin, psiMax, LALINFERENCE_PARAM_LINEAR);
  LALInferenceRegisterUniformVariableREAL8(state, model->params, "costheta_jn", zero, costhetaJNmin, costhetaJNmax,LALINFERENCE_PARAM_LINEAR);

  /* Option to use the detector-aligned frame */ 
  if(LALInferenceGetProcParamVal(commandLine,"--detector-frame"))
  {
      printf("Using detector-based sky frame\n");
      LALInferenceRegisterUniformVariableREAL8(state,model->params,"t0",timeParam,timeMin,timeMax,LALINFERENCE_PARAM_LINEAR);
      LALInferenceRegisterUniformVariableREAL8(state,model->params,"cosalpha",0,-1,1,LALINFERENCE_PARAM_LINEAR);
      LALInferenceRegisterUniformVariableREAL8(state,model->params,"azimuth",0.0,0.0,LAL_TWOPI,LALINFERENCE_PARAM_CIRCULAR);
      /* add the time parameter then remove it so that the prior is set up properly */
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "time", timeParam, timeMin, timeMax,LALINFERENCE_PARAM_LINEAR);
      LALInferenceRemoveVariable(model->params,"time");
      INT4 one=1;
      LALInferenceAddVariable(model->params,"SKY_FRAME",&one,LALINFERENCE_INT4_t,LALINFERENCE_PARAM_FIXED);
  }
  else
  {
		  LALInferenceRegisterUniformVariableREAL8(state, model->params, "rightascension", zero, raMin, raMax, LALINFERENCE_PARAM_CIRCULAR);
		  LALInferenceRegisterUniformVariableREAL8(state, model->params, "declination", zero, decMin, decMax, LALINFERENCE_PARAM_LINEAR);
		  LALInferenceRegisterUniformVariableREAL8(state, model->params, "time", timeParam, timeMin, timeMax,LALINFERENCE_PARAM_LINEAR);
  }
  /* If we are marginalising over the time, remove that variable from the model (having set the prior above) */
  /* Also set the prior in model->params, since Likelihood can't access the state! (ugly hack) */
  if(LALInferenceGetProcParamVal(commandLine,"--margtime") || LALInferenceGetProcParamVal(commandLine, "--margtimephi")){
	  LALInferenceVariableItem *p=LALInferenceGetItem(state->priorArgs,"time_min");
	  LALInferenceAddVariable(model->params,"time_min",p->value,p->type,p->vary);
	  p=LALInferenceGetItem(state->priorArgs,"time_max");
	  LALInferenceAddVariable(model->params,"time_max",p->value,p->type,p->vary);
	  if (LALInferenceCheckVariable(model->params,"time")) LALInferenceRemoveVariable(model->params,"time");
	  if (LALInferenceGetProcParamVal(commandLine, "--margtimephi")) {
		  UINT4 margphi = 1;
		  LALInferenceAddVariable(model->params, "margtimephi", &margphi, LALINFERENCE_UINT4_t,LALINFERENCE_PARAM_FIXED);
	  }
  }
      
  /* PPE parameters */

  ppt=LALInferenceGetProcParamVal(commandLine, "--TaylorF2ppE");
  if(approx==TaylorF2 && ppt){

    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppealpha",zero, -1000.0 , 1000.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppebeta", zero, -1000.0 , 1000.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppeuppera", zero, -3.0, 3.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppeupperb", zero, -3.0, 3.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppelowera", zero, -3.0, 2.0/3.0 , LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "ppelowerb", zero, -4.5, 1.0, LALINFERENCE_PARAM_LINEAR);

  }

  if(LALInferenceGetProcParamVal(commandLine,"--tidalT")&&LALInferenceGetProcParamVal(commandLine,"--tidal")){
    XLALPrintError("Error: cannot use both --tidalT and --tidal.\n");
    XLAL_ERROR_NULL(XLAL_EINVAL);
  } else if(LALInferenceGetProcParamVal(commandLine,"--tidalT")){
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "lambdaT", zero, lambdaTMin, lambdaTMax, LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "dLambdaT", zero, dLambdaTMin, dLambdaTMax, LALINFERENCE_PARAM_LINEAR);

  } else if(LALInferenceGetProcParamVal(commandLine,"--tidal")){
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "lambda1", zero, lambda1Min, lambda1Max, LALINFERENCE_PARAM_LINEAR);
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "lambda2", zero, lambda2Min, lambda2Max, LALINFERENCE_PARAM_LINEAR);

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

  LALSimInspiralFrameAxis frameAxis = LAL_SIM_INSPIRAL_FRAME_AXIS_DEFAULT;
  if((ppt=LALInferenceGetProcParamVal(commandLine,"--inj-frame-axis"))) {
    frameAxis = XLALSimInspiralGetFrameAxisFromString(ppt->value);
  }

  model->waveFlags = XLALSimInspiralCreateWaveformFlags();
  XLALSimInspiralSetSpinOrder(model->waveFlags,  spinO);
  XLALSimInspiralSetTidalOrder(model->waveFlags, tideO);
  XLALSimInspiralSetFrameAxis(model->waveFlags,frameAxis);


  fprintf(stdout,"\n\n---\t\t ---\n");
  LALInferenceInitSpinVariables(state, model);
  LALInferenceCheckApproximantNeeds(state,approx);

  if (injTable)
     print_flags_orders_warning(injTable,commandLine);

     /* Print info about orders and waveflags used for templates */

     fprintf(stdout,"Templates will run using Approximant %i (%s), phase order %i, amp order %i, spin order %i tidal order %i, frame axis %i in the %s domain.\n",approx,XLALGetStringFromApproximant(approx),PhaseOrder,AmpOrder,(int) spinO, (int) tideO, (int) frameAxis, model->domain==LAL_SIM_DOMAIN_TIME?"time":"frequency");
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
		{.name="costheta_jn", .val=LAL_PI/2., .min=1.4054428267948966, .max=1.7361498267948965},
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
    {.name="costheta_jn", .val=0.9176809634, .min=0.6200446634, .max=1.2153172634},
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
    {.name="costheta_jn", .val=LAL_PI/2., .min=-0.429203673, .max=3.570796327},
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
  if (!ppt)
      ppt=LALInferenceGetProcParamVal(commandLine,"--PSDlength");
  if (!ppt) {
      printf("ERROR: PSD length not specified. Exiting...\n");
      exit(1);
  }
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

  /* Ensure that the user is not trying to marginalise the likelihood
     in an inconsistent way */
  if (LALInferenceGetProcParamVal(commandLine, "--margtime") && LALInferenceGetProcParamVal(commandLine, "--margphi")) {
    fprintf(stderr, "ERROR: trying to separately marginalise in time and phase.  Use '--margtimephi' instead");
    exit(1);
  }

  if (LALInferenceGetProcParamVal(commandLine, "--margtimephi") && LALInferenceGetProcParamVal(commandLine, "--margtime")) {
    fprintf(stderr, "ERROR: cannot marginalise in time and phi and separately time.  Pick either '--margtimephi' OR '--margtime'");
    exit(1);
  }

  if (LALInferenceGetProcParamVal(commandLine, "--margtimephi") && LALInferenceGetProcParamVal(commandLine, "--margphi")) {
    fprintf(stderr, "ERROR: cannot marginalise in time and phi and separately in phi.  Pick either '--margtimephi' OR '--margtime'");
    exit(1);
  }

  /* Check for small sample rates when margtime-ing. */
  if (LALInferenceGetProcParamVal(commandLine, "--margtime") || LALInferenceGetProcParamVal(commandLine, "--margtimephi")) {
    ppt = LALInferenceGetProcParamVal(commandLine, "--srate");
    if (ppt) {
      int srate = atoi(ppt->value);

      if (srate < 4096) {
	XLALPrintWarning("WARNING: you have chosen to marginalise in time with a sample rate of %d, but this typically gives incorrect results for CBCs; use at least 4096 Hz to be safe", srate);
      }
    }
  }

  return;
}

void LALInferenceInitSpinVariables(LALInferenceRunState *state, LALInferenceModel *model){


  LALStatus status;
  memset(&status,0,sizeof(status));

  ProcessParamsTable *commandLine=state->commandLine;
  ProcessParamsTable *ppt=NULL;

  Approximant approx= *(Approximant*) LALInferenceGetVariable(model->params, "LAL_APPROXIMANT");

  REAL8 a1min=0.0,a1max=1.0;
  REAL8 a2min=0.0,a2max=1.0;
  REAL8 tilt1min=0.0,tilt1max=LAL_PI;
  REAL8 tilt2min=0.0,tilt2max=LAL_PI;
  REAL8 phi12min=0.0,phi12max=LAL_TWOPI;
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

  /* Now check what the approx can do and eventually change user's choices to comply.
   * Also change the reference frame -- For the moment use default as the corresponding patch to LALSimulation has not been finished yet */
  if (spin_support==LAL_SIM_INSPIRAL_SPINLESS)
    noSpin=1;
  else if (spin_support==LAL_SIM_INSPIRAL_SINGLESPIN)
    singleSpin=1;
  else if (spin_support==LAL_SIM_INSPIRAL_ALIGNEDSPIN){
    spinAligned=1;
  }

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

  /* Add parameters depending on the values of noSpin, singleSpin and spinAligned
   * noSpin -> add nothing
   * spinAligned -> add a_spin1 and a_spin2 (if the approximant is spin aligned only use the names spin1,spin1 instead)
   * singleSpin -> add a_spin1, tilt_spin1, phi_JL
   * singleSpin+spinAligned -> add a_spin1
   * otherwise -> add everything */
   /* Note: To get spin aligned is sufficient to not add the spin angles because LALInferenceTemplate will default to aligned spin if it doesn't find spin angle params in model->params. */
  if (!noSpin){
    LALInferenceRegisterUniformVariableREAL8(state, model->params, "a_spin1", 0.0, a1min, a1max,spin1Vary);
    if (!singleSpin)
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "a_spin2", 0.0, a2min, a2max,spin2Vary);
    if (!spinAligned){
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "phi_jl", 0.0, phiJLmin,  phiJLmax, phiJLVary);
      LALInferenceRegisterUniformVariableREAL8(state, model->params, "tilt_spin1", 0.0, tilt1min,tilt1max,tilt1Vary);
      if (!singleSpin){
        LALInferenceRegisterUniformVariableREAL8(state, model->params, "tilt_spin2", 0.0, tilt2min,tilt2max,tilt2Vary);
        LALInferenceRegisterUniformVariableREAL8(state, model->params, "phi12", 0.0, phi12min,phi12max,phi12Vary);
      }
    }
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

void LALInferenceInitMassVariables(LALInferenceRunState *state){

  LALStatus status;
  memset(&status,0,sizeof(status));

  ProcessParamsTable *commandLine=state->commandLine;
  ProcessParamsTable *ppt=NULL;
  LALInferenceVariables *priorArgs=state->priorArgs;

  REAL8 mMin=1.0,mMax=30.0;
  REAL8 MTotMax=35.0;
  REAL8 MTotMin=2.0;

  /* Over-ride component masses */
  ppt=LALInferenceGetProcParamVal(commandLine,"--comp-min");
  //if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--compmin");
  if(ppt){
          mMin=atof(ppt->value);
          MTotMin=2.0*mMin;
  }

  ppt=LALInferenceGetProcParamVal(commandLine,"--comp-max");
  //if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--compmax");
  if(ppt){
          mMax=atof(ppt->value);
          MTotMax=2.0*mMax;
  }

  ppt=LALInferenceGetProcParamVal(commandLine,"--mtotal-max");
  if(ppt)	MTotMax=atof(ppt->value);

  ppt=LALInferenceGetProcParamVal(commandLine,"--mtotal-min");
  if(ppt)	MTotMin=atof(ppt->value);

  LALInferenceAddVariable(priorArgs,"component_min",&mMin,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(priorArgs,"component_max",&mMax,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(priorArgs,"MTotMax",&MTotMax,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
  LALInferenceAddVariable(priorArgs,"MTotMin",&MTotMin,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);

  return;

}

void LALInferenceCheckApproximantNeeds(LALInferenceRunState *state,Approximant approx){

  REAL8 min,max;
  UINT4 q=0;

  if (LALInferenceCheckVariable(state->priorArgs,"q_min")){
    LALInferenceGetMinMaxPrior(state->priorArgs, "q", &min, &max);
    q=1;
  }
  else if (LALInferenceCheckVariable(state->priorArgs,"eta_min"))
    LALInferenceGetMinMaxPrior(state->priorArgs, "eta", &min, &max);

  /* IMRPhenomP only supports q > 1/10. Set prior consequently  */
  if (q==1 && approx==IMRPhenomP && min<1./10.){
    min=1.0/10.;
    LALInferenceRemoveVariable(state->priorArgs,"q_min");
    LALInferenceAddVariable(state->priorArgs,"q_min",&min,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
    fprintf(stdout,"WARNING: IMRPhenomP only supports mass ratios up to 10 ( suggested max: 4). Changing the min prior for q to 1/10\n");
  }
  if (q==0 && approx==IMRPhenomP && min<0.08264462810){
     min=0.08264462810;  //(that is eta for a 1-10 system)
     LALInferenceRemoveVariable(state->priorArgs,"eta_min");
     LALInferenceAddVariable(state->priorArgs,"eta_min",&min,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
     fprintf(stdout,"WARNING: IMRPhenomP only supports mass ratios up to 10 ( suggested max: 4). Changing the min prior for eta to 0.083\n");
  }

  (void) max;
  return;
}

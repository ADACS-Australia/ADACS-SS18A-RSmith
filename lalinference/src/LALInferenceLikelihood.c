/* 
 *  LALInferenceLikelihood.c:  Bayesian Followup likelihood functions
 *
 *  Copyright (C) 2009 Ilya Mandel, Vivien Raymond, Christian Roever,
 *  Marc van der Sluys and John Veitch, Will M. Farr
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

#define LAL_USE_OLD_COMPLEX_STRUCTS
#include <complex.h>
#include <lal/LALInferenceLikelihood.h>
#include <lal/LALInferencePrior.h>
#include <lal/LALInference.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lal/TimeSeries.h>
#include <lal/Units.h>
#include <lal/Sequence.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_dawson.h>
#include <gsl/gsl_sf_erf.h>

#include <lal/LALInferenceTemplate.h>

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

void LALInferenceInitLikelihood(LALInferenceRunState *runState)
{
    char help[]="\
                 (--zeroLogLike)                  Use flat, null likelihood.\n\
                 (--studentTLikelihood)           Use the Student-T Likelihood that marginalizes over noise.\n\
                 (--correlatedGaussianLikelihood) Use analytic, correlated Gaussian for Likelihood.\n\
                 (--bimodalGaussianLikelihood)    Use analytic, bimodal correlated Gaussian for Likelihood.\n\
                 (--rosenbrockLikelihood)         Use analytic, Rosenbrock banana for Likelihood.\n\
                 (--noiseonly)                    Using noise-only likelihood.\n\
                 (--margphi)                      Using marginalised phase likelihood.\n\
                 (--margtime)                     Using marginalised time likelihood.\n\
                 (--margtimephi)                  Using marginalised in time and phase likelihood\n";

    ProcessParamsTable *commandLine=runState->commandLine;
    LALInferenceIFOData *ifo=runState->data;

    /* Print command line arguments if help requested */
    if(LALInferenceGetProcParamVal(runState->commandLine,"--help"))
    {
        fprintf(stdout,"%s",help);
        while(ifo) {
            fprintf(stdout,"(--dof-%s DoF)\tDegrees of freedom for %s\n",ifo->name,ifo->name);
            ifo=ifo->next;
        }
        return;
    }

   if (LALInferenceGetProcParamVal(commandLine, "--zeroLogLike")) {
    /* Use zero log(L) */
    runState->likelihood=&LALInferenceZeroLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--correlatedGaussianLikelihood")) {
    runState->likelihood=&LALInferenceCorrelatedAnalyticLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--bimodalGaussianLikelihood")) {
    runState->likelihood=&LALInferenceBimodalCorrelatedAnalyticLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--rosenbrockLikelihood")) {
    runState->likelihood=&LALInferenceRosenbrockLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--studentTLikelihood")) {
    fprintf(stderr, "Using Student's T Likelihood.\n");
    runState->likelihood=&LALInferenceFreqDomainStudentTLogLikelihood;

    /* Set the noise model evidence to the student t model value */
    LALInferenceTemplateNullFreqdomain(runState->data);
    REAL8 noiseZ=LALInferenceFreqDomainStudentTLogLikelihood(runState->currentParams,runState->data,&LALInferenceTemplateNullFreqdomain);
    LALInferenceAddVariable(runState->algorithmParams,"logZnoise",&noiseZ,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_FIXED);
    fprintf(stdout,"Student-t Noise evidence %lf\n",noiseZ);

   } else if (LALInferenceGetProcParamVal(commandLine, "--noiseonly")) {
    fprintf(stderr, "Using noise-only likelihood.\n");
    runState->likelihood=&LALInferenceNoiseOnlyLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--margphi")) {
    fprintf(stderr, "Using marginalised phase likelihood.\n");
    runState->likelihood=&LALInferenceMarginalisedPhaseLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--margtime")) {
    fprintf(stderr, "Using marginalised time likelihood.\n");
    runState->likelihood=&LALInferenceMarginalisedTimeLogLikelihood;
   } else if (LALInferenceGetProcParamVal(commandLine, "--margtimephi")) {
     UINT4 margphi = 1;
     fprintf(stderr, "Using marginalised in time and phase likelihood.\n");
     runState->likelihood=&LALInferenceMarginalisedTimeLogLikelihood;
     LALInferenceAddVariable(runState->currentParams, "margtimephi", &margphi, LALINFERENCE_UINT4_t,LALINFERENCE_PARAM_FIXED);
   } else {
    runState->likelihood=&LALInferenceUndecomposedFreqDomainLogLikelihood;
   }

    return;
}

/* Scaling used for the analytic likelihood parameters */
  static const REAL8 scaling[15] = {
    1.0,
    1.0,
    20.0/M_PI,
    10.0/M_PI,
    20.0/M_PI,
    10.0/M_PI,
    10.0/M_PI,
    0.1,
    10.0,
    10.0,
    10.0,
    20.0/M_PI,
    20.0/M_PI,
    10.0/M_PI,
    10.0/M_PI};

/* Covariance matrix for use in analytic likelihoods */
  static const REAL8 CM[15][15] = {{0.045991865933182365, -0.005489748382557155, -0.01025067223674548, 0.0020087713726603213, -0.0032648855847982987, -0.0034218261781145264, -0.0037173401838545774, -0.007694897715679858, 0.005260905282822458, 0.0013607957548231718, 0.001970785895702776, 0.006708452591621081, -0.005107684668720825, 0.004402554308030673, -0.00334987648531921},
                              {-0.005489748382557152, 0.05478640427684032, -0.004786202916836846, -0.007930397407501268, -0.0005945107515129139, 0.004858466255616657, -0.011667819871670204, 0.003169780190169035, 0.006761345004654851, -0.0037599761532668475, 0.005571796842520162, -0.0071098291510566895, -0.004477773540640284, -0.011250694688474672, 0.007465228985669282},
                              {-0.01025067223674548, -0.004786202916836844, 0.044324704403674524, -0.0010572820723801645, -0.009885693540838514, -0.0048321205972943464, -0.004576186966267275, 0.0025107211483955676, -0.010126911913571181, 0.01153595152487264, 0.005773054728678472, 0.005286558230422045, -0.0055438798694137734, 0.0044772210361854765, -0.00620416958073918},
                              {0.0020087713726603213, -0.007930397407501268, -0.0010572820723801636, 0.029861342087731065, -0.007803477134405363, -0.0011466944120756021, 0.009925736654597632, -0.0007664415942207051, -0.0057593957402320385, -0.00027248233573270216, 0.003885350572544307, 0.00022362281488693097, 0.006609741178005571, -0.003292722856107429, -0.005873218251875897},
                              {-0.0032648855847982987, -0.0005945107515129156, -0.009885693540838514, -0.007803477134405362, 0.0538403407841302, -0.007446654755103316, -0.0025216534232170153, 0.004499568241334517, 0.009591034277125177, 0.00008612746932654223, 0.003386296829505543, -0.002600737873367083, 0.000621148057330571, -0.006603857049454523, -0.009241221870695395},
                              {-0.0034218261781145264, 0.004858466255616657, -0.004832120597294347, -0.0011466944120756015, -0.007446654755103318, 0.043746559133865104, 0.008962713024625965, -0.011099652042761613, -0.0006620240117921668, -0.0012591530037708058, -0.006899982952117269, 0.0019732354732442878, -0.002445676747004324, -0.006454778807421816, 0.0033303577606412765},
                              {-0.00371734018385458, -0.011667819871670206, -0.004576186966267273, 0.009925736654597632, -0.0025216534232170153, 0.008962713024625965, 0.03664582756831382, -0.009470328827284009, -0.006213741694945105, 0.007118775954484294, -0.0006741237990418526, -0.006003374957986355, 0.005718636997353189, -0.0005191095254772077, -0.008466566781233205},
                              {-0.007694897715679857, 0.0031697801901690347, 0.002510721148395566, -0.0007664415942207059, 0.004499568241334515, -0.011099652042761617, -0.009470328827284016, 0.057734267068088, 0.005521731225009532, -0.017008048805405164, 0.006749693090695894, -0.006348460110898, -0.007879244727681924, -0.005321753837620446, 0.011126783289057604},
                              {0.005260905282822458, 0.0067613450046548505, -0.010126911913571181, -0.00575939574023204, 0.009591034277125177, -0.0006620240117921668, -0.006213741694945106, 0.005521731225009532, 0.04610670018969681, -0.010427010812879566, -0.0009861561285861987, -0.008896020395949732, -0.0037627528719902485, 0.00033704453138913093, -0.003173552163182467},
                              {0.0013607957548231744, -0.0037599761532668475, 0.01153595152487264, -0.0002724823357326985, 0.0000861274693265406, -0.0012591530037708062, 0.007118775954484294, -0.01700804880540517, -0.010427010812879568, 0.05909125052583998, 0.002192545816395299, -0.002057672237277737, -0.004801518314458135, -0.014065326026662672, -0.005619012077114913},
                              {0.0019707858957027763, 0.005571796842520162, 0.005773054728678472, 0.003885350572544309, 0.003386296829505542, -0.006899982952117272, -0.0006741237990418522, 0.006749693090695893, -0.0009861561285862005, 0.0021925458163952988, 0.024417715762416557, -0.003037163447600162, -0.011173674374382736, -0.0008193127407211239, -0.007137012700864866},
                              {0.006708452591621083, -0.0071098291510566895, 0.005286558230422046, 0.00022362281488693216, -0.0026007378733670806, 0.0019732354732442886, -0.006003374957986352, -0.006348460110897999, -0.008896020395949732, -0.002057672237277737, -0.003037163447600163, 0.04762367868805726, 0.0008818947598625008, -0.0007262691465810616, -0.006482422704208912},
                              {-0.005107684668720825, -0.0044777735406402895, -0.005543879869413772, 0.006609741178005571, 0.0006211480573305693, -0.002445676747004324, 0.0057186369973531905, -0.00787924472768192, -0.003762752871990247, -0.004801518314458137, -0.011173674374382736, 0.0008818947598624995, 0.042639958466440225, 0.0010194948614718209, 0.0033872675386130637},
                              {0.004402554308030674, -0.011250694688474675, 0.004477221036185477, -0.003292722856107429, -0.006603857049454523, -0.006454778807421815, -0.0005191095254772072, -0.005321753837620446, 0.0003370445313891318, -0.014065326026662679, -0.0008193127407211239, -0.0007262691465810616, 0.0010194948614718226, 0.05244900188599414, -0.000256550861960499},
                              {-0.00334987648531921, 0.007465228985669282, -0.006204169580739178, -0.005873218251875899, -0.009241221870695395, 0.003330357760641278, -0.008466566781233205, 0.011126783289057604, -0.0031735521631824654, -0.005619012077114915, -0.007137012700864866, -0.006482422704208912, 0.0033872675386130632, -0.000256550861960499, 0.05380987317762257}};

const char *non_intrinsic_params[] = {"rightascension", "declination", "polarisation", "time",
                                "deltaLogL", "logL", "deltaloglH1", "deltaloglL1", "deltaloglV1",
                                "logw", "logPrior", "distance", "logdistance", NULL};

LALInferenceVariables LALInferenceGetInstrinsicParams(LALInferenceVariables *currentParams)
/***************************************************************/
/* Return a variables structure containing only intrinsic      */
/* parameters.                                                 */
/***************************************************************/
{
    // TODO: add pointer to template function here.
    // (otherwise same parameters but different template will lead to no re-computation!!)
    LALInferenceVariables intrinsicParams;
    const char **non_intrinsic_param = non_intrinsic_params;

    intrinsicParams.head      = NULL;
    intrinsicParams.dimension = 0;
    LALInferenceCopyVariables(currentParams, &intrinsicParams);

    while (*non_intrinsic_param) {
        if (LALInferenceCheckVariable(&intrinsicParams, *non_intrinsic_param))
            LALInferenceRemoveVariable(&intrinsicParams, *non_intrinsic_param);
        non_intrinsic_param++;
    }

    return intrinsicParams;
}

INT4 LALInferenceLineSwitch(INT4 lineFlag, INT4 Nlines, INT4 *lines_array, INT4 *widths_array, INT4 i)
{
    INT4 lineimin = 0;
    INT4 lineimax = 0;
    INT4 lineSwitch=1;
    INT4 j;

    if(lineFlag) {
        for(j=0;j<Nlines;j++) {
            //find range of fourier fourier bins which are excluded from integration
            lineimin = lines_array[j] - widths_array[j];
            lineimax = lines_array[j] + widths_array[j];

            //if the current bin is inside the exluded region, set the switch to 0
            if(i>lineimin && i<lineimax) lineSwitch=0;
        }
    }
    return lineSwitch;
}

/* ============ Likelihood computations: ========== */

/**
 * For testing purposes (for instance sampling the prior), likelihood that returns 0.0 = log(1) every
 * time.  Activated with the --zeroLogLike command flag.
 */
REAL8 LALInferenceZeroLogLikelihood(LALInferenceVariables UNUSED *currentParams, LALInferenceIFOData UNUSED *data, LALInferenceTemplateFunction UNUSED template) {
  return 0.0;
}

REAL8 LALInferenceNoiseOnlyLogLikelihood(LALInferenceVariables *currentParams, LALInferenceIFOData *data, LALInferenceTemplateFunction UNUSED template)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood           */
/* for noise-only models of the data                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "psdscale"  (gslMatrix)                                 */
/***************************************************************/
{
  double diffRe, diffIm, diffSquared;
  double dataReal, dataImag;
  REAL8 loglikeli;
  int i, j, lower, upper, ifo;
  LALInferenceIFOData *dataPtr;
  double chisquared;
  double deltaT, TwoDeltaToverN, deltaF;

  //noise model meta parameters
  gsl_matrix *lines   = NULL;//pointer to matrix holding line centroids
  gsl_matrix *widths  = NULL;//pointer to matrix holding line widths
  gsl_matrix *nparams = NULL;//pointer to matrix holding noise parameters

  gsl_matrix *psdBandsMin  = NULL;//pointer to matrix holding min frequencies for psd model
  gsl_matrix *psdBandsMax = NULL;//pointer to matrix holding max frequencies for psd model

  int Nblock = 1;            //number of frequency blocks per IFO
  int Nlines = 1;            //number of lines to be removed
  int psdFlag;               //flag for including psd fitting
  int lineFlag;              //flag for excluding lines from integration

  //line removal parameters
  lineFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "removeLinesFlag"));
  if(lineFlag)
  {
    //Add line matrices to variable lists
    lines  = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_center");
    widths = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_width");
    Nlines = (int)lines->size2;
  }
  int lines_array[Nlines];
  int widths_array[Nlines];

  //check if psd parameters are included in the model
  psdFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "psdScaleFlag"));
  if(psdFlag)
  {
    //if so, store current noise parameters in easily accessible matrix
    nparams = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdscale"));
    Nblock = (int)nparams->size2;

    psdBandsMin = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMin"));
    psdBandsMax = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMax"));

  }
  double alpha[Nblock];
  double lnalpha[Nblock];

  double psdBandsMin_array[Nblock];
  double psdBandsMax_array[Nblock];

  chisquared = 0.0;
  /* loop over data (different interferometers): */
  dataPtr = data;
  ifo=0;

  while (dataPtr != NULL) {
    /* The parameters the Likelihood function can handle by itself   */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
    /* t_c corresponds to the "time" parameter in                    */
    /* IFOdata->modelParams (set, e.g., from the trigger value).     */

    /* Reset log-likelihood */
    dataPtr->loglikelihood = 0.0;

    /* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);

    lower = (UINT4)ceil(dataPtr->fLow / deltaF);
    upper = (UINT4)floor(dataPtr->fHigh / deltaF);
    TwoDeltaToverN = 2.0 * deltaT / ((double) dataPtr->timeData->data->length);

    //Set up noise PSD meta parameters
    for(i=0; i<Nblock; i++)
    {
      if(psdFlag)
      {
        alpha[i]   = gsl_matrix_get(nparams,ifo,i);
        lnalpha[i] = log(alpha[i]);

        psdBandsMin_array[i] = gsl_matrix_get(psdBandsMin,ifo,i);
        psdBandsMax_array[i] = gsl_matrix_get(psdBandsMax,ifo,i);

      }
      else
      {
        alpha[i]=1.0;
        lnalpha[i]=0.0;
      }
    }

    //Set up psd line arrays
    for(j=0;j<Nlines;j++)
    {
      if(lineFlag)
      {

        //find range of fourier fourier bins which are excluded from integration
        lines_array[j]  = (int)gsl_matrix_get(lines,ifo,j);
        widths_array[j] = (int)gsl_matrix_get(widths,ifo,j);
      }
      else
      {
        lines_array[j]=0;
        widths_array[j]=0;
      }
    }

    for (i=lower; i<=upper; ++i)
    {

      dataReal     = creal(dataPtr->freqData->data->data[i]) / deltaT;
      dataImag     = cimag(dataPtr->freqData->data->data[i]) / deltaT;
      
      /* compute squared difference & 'chi-squared': */
      diffRe       = dataReal;         // Difference in real parts...
      diffIm       = dataImag;         // ...and imaginary parts, and...
      diffSquared  = diffRe*diffRe + diffIm*diffIm ;  // ...squared difference of the 2 complex figures.
      
      REAL8 temp = ((TwoDeltaToverN * diffSquared) / dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);

      /* Add noise PSD parameters to the model */
      if(psdFlag)
      {
        for(j=0; j<Nblock; j++)
        {
            if (i >= psdBandsMin_array[j] && i <= psdBandsMax_array[j])
            {
                temp  /= alpha[j];
                temp  += lnalpha[j];
            }
        }
      }

      /*only sum over bins which are outside of excluded regions */
      if(LALInferenceLineSwitch(lineFlag, Nlines, lines_array, widths_array, i))
      {
        chisquared  += temp;
        dataPtr->loglikelihood -= temp;
      }
      
     }
    ifo++; //increment IFO counter for noise parameters
    dataPtr = dataPtr->next;
  }

  loglikeli = -1.0 * chisquared; // note (again): the log-likelihood is unnormalised!
  return(loglikeli);
}

REAL8 LALInferenceUndecomposedFreqDomainLogLikelihood(LALInferenceVariables *currentParams, LALInferenceIFOData * data, 
                              LALInferenceTemplateFunction templt)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood.          */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/
{
  double Fplus, Fcross;
  double FplusScaled, FcrossScaled;
  double diffRe, diffIm, diffSquared;
  double dataReal, dataImag;
  REAL8 loglikeli;
  REAL8 plainTemplateReal, plainTemplateImag;
  REAL8 templateReal, templateImag;
  int i, j, lower, upper, ifo;
  LALInferenceIFOData *dataPtr;
  double ra, dec, psi, distMpc, gmst;
  double GPSdouble;
  LIGOTimeGPS GPSlal;
  double chisquared;
  double timedelay;  /* time delay b/w iterferometer & geocenter w.r.t. sky location */
  double timeshift;  /* time shift (not necessarily same as above)                   */
  double deltaT, TwoDeltaToverN, deltaF, twopit, re, im, dre, dim, newRe, newIm;
  double timeTmp;
  int different;
	double mc;
	UINT4 logDistFlag=0;
  LALStatus status;
  memset(&status,0,sizeof(status));
  LALInferenceVariables intrinsicParams;

  if(data==NULL) {XLAL_ERROR_REAL8(XLAL_EINVAL,"ERROR: Encountered NULL data pointer in likelihood\n");}

  //noise model meta parameters
  gsl_matrix *lines   = NULL;//pointer to matrix holding line centroids
  gsl_matrix *widths  = NULL;//pointer to matrix holding line widths
  gsl_matrix *nparams = NULL;//pointer to matrix holding noise parameters

  gsl_matrix *psdBandsMin  = NULL;//pointer to matrix holding min frequencies for psd model
  gsl_matrix *psdBandsMax = NULL;//pointer to matrix holding max frequencies for psd model

  int Nblock = 1;            //number of frequency blocks per IFO
  int Nlines = 1;            //number of lines to be removed
  int psdFlag = 0;           //flag for including psd fitting
  int lineFlag = 0;          //flag for excluding lines from integration

  //line removal parameters
  if(LALInferenceCheckVariable(currentParams, "removeLinesFlag"))
    lineFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "removeLinesFlag"));
  if(lineFlag)
  {
    //Add line matrices to variable lists
    lines  = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_center");
    widths = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_width");
    Nlines = (int)lines->size2;
  }
  int lines_array[Nlines];
  int widths_array[Nlines];

  //check if psd parameters are included in the model
  if(LALInferenceCheckVariable(currentParams, "psdScaleFlag"))
    psdFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "psdScaleFlag"));
  if(psdFlag)
  {
    //if so, store current noise parameters in easily accessible matrix
    nparams = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdscale"));
    Nblock = (int)nparams->size2;

    psdBandsMin = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMin"));
    psdBandsMax = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMax"));

  }
  double alpha[Nblock];
  double lnalpha[Nblock];

  double psdBandsMin_array[Nblock];
  double psdBandsMax_array[Nblock];

  logDistFlag=LALInferenceCheckVariable(currentParams, "logdistance");
  if(LALInferenceCheckVariable(currentParams,"logmc")){
    mc=exp(*(REAL8 *)LALInferenceGetVariable(currentParams,"logmc"));
    LALInferenceAddVariable(currentParams,"chirpmass",&mc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }

  /* determine source's sky location & orientation parameters: */
  ra        = *(REAL8*) LALInferenceGetVariable(currentParams, "rightascension"); /* radian      */
  dec       = *(REAL8*) LALInferenceGetVariable(currentParams, "declination");    /* radian      */
  psi       = *(REAL8*) LALInferenceGetVariable(currentParams, "polarisation");   /* radian      */
  GPSdouble = *(REAL8*) LALInferenceGetVariable(currentParams, "time");           /* GPS seconds */
  if(logDistFlag)
    distMpc = exp(*(REAL8*)LALInferenceGetVariable(currentParams,"logdistance"));
  else
    distMpc = *(REAL8*) LALInferenceGetVariable(currentParams, "distance");       /* Mpc         */

  /* figure out GMST: */
  XLALGPSSetREAL8(&GPSlal, GPSdouble);
  gmst=XLALGreenwichMeanSiderealTime(&GPSlal);

  intrinsicParams = LALInferenceGetInstrinsicParams(currentParams);

  chisquared = 0.0;
  /* loop over data (different interferometers): */
  dataPtr = data;
  ifo=0;

  while (dataPtr != NULL) {
    /* The parameters the Likelihood function can handle by itself   */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
	/* t_c corresponds to the "time" parameter in                    */
	/* IFOdata->modelParams (set, e.g., from the trigger value).     */
    
    /* Reset log-likelihood */
    dataPtr->loglikelihood = 0.0;

    /* Compare parameter values with parameter values corresponding  */
    /* to currently stored template; ignore "time" variable:         */
    if (LALInferenceCheckVariable(dataPtr->modelParams, "time")) {
      timeTmp = *(REAL8 *) LALInferenceGetVariable(dataPtr->modelParams, "time");
      LALInferenceRemoveVariable(dataPtr->modelParams, "time");
    }
    else timeTmp = GPSdouble;

    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */
    different = LALInferenceCompareVariables(dataPtr->modelParams, &intrinsicParams);

    if (different) { /* template needs to be re-computed: */
      LALInferenceCopyVariables(&intrinsicParams, dataPtr->modelParams);
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
      templt(dataPtr);
      if(XLALGetBaseErrno()==XLAL_FAILURE) /* Template generation failed in a known way, set -Inf likelihood */
          return(-DBL_MAX);

      if (dataPtr->modelDomain == LAL_SIM_DOMAIN_TIME) {
        /* TD --> FD. */
        LALInferenceExecuteFT(dataPtr);
      }
    }
    else { /* no re-computation necessary. Return back "time" value, do nothing else: */
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
    }

    /* Template is now in dataPtr->timeFreqModelhPlus and hCross */

    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross, (const REAL4(*)[3])dataPtr->detector->response, ra, dec, psi, gmst);

    /* signal arrival time (relative to geocenter); */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location, ra, dec, &GPSlal);
    /* (negative timedelay means signal arrives earlier at Ifo than at geocenter, etc.) */
    /* amount by which to time-shift template (not necessarily same as above "timedelay"): */
    timeshift =  (GPSdouble - (*(REAL8*) LALInferenceGetVariable(dataPtr->modelParams, "time"))) + timedelay;

    twopit    = LAL_TWOPI * timeshift;

    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;

    if (LALInferenceCheckVariable(currentParams, "crazyInjectionHLSign") &&
        *((INT4 *)LALInferenceGetVariable(currentParams, "crazyInjectionHLSign"))) {
      if (strstr(dataPtr->name, "H") || strstr(dataPtr->name, "L")) {
        FplusScaled *= -1.0;
        FcrossScaled *= -1.0;
      }
    }

    dataPtr->fPlus = FplusScaled;
    dataPtr->fCross = FcrossScaled;
    dataPtr->timeshift = timeshift;

    /* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    lower = (UINT4)ceil(dataPtr->fLow / deltaF);
    upper = (UINT4)floor(dataPtr->fHigh / deltaF);
    TwoDeltaToverN = 2.0 * deltaT / ((double) dataPtr->timeData->data->length);

    /* Employ a trick here for avoiding cos(...) and sin(...) in time
       shifting.  We need to multiply each template frequency bin by
       exp(-J*twopit*deltaF*i) = exp(-J*twopit*deltaF*(i-1)) +
       exp(-J*twopit*deltaF*(i-1))*(exp(-J*twopit*deltaF) - 1) .  This
       recurrance relation has the advantage that the error growth is
       O(sqrt(N)) for N repetitions. */
    
    /* Values for the first iteration: */
    re = cos(twopit*deltaF*lower);
    im = -sin(twopit*deltaF*lower);

    /* Incremental values, using cos(theta) - 1 = -2*sin(theta/2)^2 */
    dim = -sin(twopit*deltaF);
    dre = -2.0*sin(0.5*twopit*deltaF)*sin(0.5*twopit*deltaF);

    //Set up noise PSD meta parameters
    for(i=0; i<Nblock; i++)
    {
      if(psdFlag)
      {
        alpha[i]   = gsl_matrix_get(nparams,ifo,i);
        lnalpha[i] = log(alpha[i]);

        psdBandsMin_array[i] = gsl_matrix_get(psdBandsMin,ifo,i);
        psdBandsMax_array[i] = gsl_matrix_get(psdBandsMax,ifo,i);
      }
      else
      {
        alpha[i]=1.0;
        lnalpha[i]=0.0;
      }
    }

    //Set up psd line arrays
    for(j=0;j<Nlines;j++)
    {
      if(lineFlag)
      {
        //find range of fourier fourier bins which are excluded from integration
        lines_array[j]  = (int)gsl_matrix_get(lines,ifo,j);
        widths_array[j] = (int)gsl_matrix_get(widths,ifo,j);
      }
      else
      {
        lines_array[j]=0;
        widths_array[j]=0;
      }
    }

    for (i=lower; i<=upper; ++i){
      /* derive template (involving location/orientation parameters) from given plus/cross waveforms: */
      plainTemplateReal = FplusScaled * creal(dataPtr->freqModelhPlus->data->data[i])  
                          +  FcrossScaled * creal(dataPtr->freqModelhCross->data->data[i]);
      plainTemplateImag = FplusScaled * cimag(dataPtr->freqModelhPlus->data->data[i])  
                          +  FcrossScaled * cimag(dataPtr->freqModelhCross->data->data[i]);

      /* do time-shifting...             */
      /* (also un-do 1/deltaT scaling): */
      templateReal = (plainTemplateReal*re - plainTemplateImag*im) / deltaT;
      templateImag = (plainTemplateReal*im + plainTemplateImag*re) / deltaT;
      dataReal     = creal(dataPtr->freqData->data->data[i]) / deltaT;
      dataImag     = cimag(dataPtr->freqData->data->data[i]) / deltaT;
      /* compute squared difference & 'chi-squared': */
      diffRe       = dataReal - templateReal;         // Difference in real parts...
      diffIm       = dataImag - templateImag;         // ...and imaginary parts, and...
      diffSquared  = diffRe*diffRe + diffIm*diffIm ;  // ...squared difference of the 2 complex figures.
      REAL8 temp = ((TwoDeltaToverN * diffSquared) / dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);

      /* Add noise PSD parameters to the model */
      if(psdFlag)
      {
        for(j=0; j<Nblock; j++)
        {
            if (i >= psdBandsMin_array[j] && i <= psdBandsMax_array[j])
            {
                temp  /= alpha[j];
                temp  += lnalpha[j];
            }
        }
      }

      /*only sum over bins which are outside of excluded regions */
      if(LALInferenceLineSwitch(lineFlag, Nlines, lines_array, widths_array, i))
      {
        chisquared  += temp;
        dataPtr->loglikelihood -= temp;
      }
 
      /* Now update re and im for the next iteration. */
      newRe = re + re*dre - im*dim;
      newIm = im + re*dim + im*dre;

      re = newRe;
      im = newIm;
    }
    ifo++; //increment IFO counter for noise parameters
    dataPtr = dataPtr->next;
  }
  loglikeli = -1.0 * chisquared; // note (again): the log-likelihood is unnormalised!
  LALInferenceClearVariables(&intrinsicParams);
  return(loglikeli);
}

/***************************************************************/
/* Student-t (log-) likelihood function                        */
/* as described in Roever/Meyer/Christensen (2011):            */
/*   "Modelling coloured residual noise                        */
/*   in gravitational-wave signal processing."                 */
/*   Classical and Quantum Gravity, 28(1):015010.              */
/*   http://dx.doi.org/10.1088/0264-9381/28/1/015010           */
/*   http://arxiv.org/abs/0804.3853                            */
/* Returns the non-normalised logarithmic likelihood.          */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, > 0)                     */
/*   - "time"            (REAL8, GPS sec.)                     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* This function is essentially the same as the                */
/* "UndecomposedFreqDomainLogLikelihood()" function.           */
/* The additional parameter to be supplied is the (REAL8)      */
/* degrees-of-freedom parameter (nu) for each Ifo.             */
/* The additional "df" argument gives the corresponding        */
/* d.f. parameter for each element of the "*data" list.        */
/* The names of "df" must match the "->name" slot of           */
/* the elements of "data".                                     */
/*                                                             */
/* (TODO: allow for d.f. parameter to vary with frequency,     */
/*        i.e., to be a set of vectors corresponding to        */
/*        frequencies)                                         */
/***************************************************************/

REAL8 LALInferenceFreqDomainStudentTLogLikelihood(LALInferenceVariables *currentParams, LALInferenceIFOData *data, 
                                      LALInferenceTemplateFunction templt)
{
  double Fplus, Fcross;
  double FplusScaled, FcrossScaled;
  double diffRe, diffIm, diffSquared;
  double dataReal, dataImag;
  REAL8 loglikeli;
  REAL8 plainTemplateReal, plainTemplateImag;
  REAL8 templateReal, templateImag;
  int i, lower, upper;
  LALInferenceIFOData *dataPtr;
  double ra, dec, psi, distMpc, gmst,mc;
  double GPSdouble;
  LIGOTimeGPS GPSlal;
  double chisquared;
  double timedelay;  /* time delay b/w iterferometer & geocenter w.r.t. sky location */
  double timeshift;  /* time shift (not necessarily same as above)                   */
  double deltaT, FourDeltaToverN, deltaF, twopit, re, im, singleFreqBinTerm, dre, dim, newRe, newIm;
  double degreesOfFreedom, nu;
  double timeTmp;
  int different;
  LALStatus status;
  memset(&status,0,sizeof(status));
  LALInferenceVariables intrinsicParams;
  
  /* Fill in derived parameters if necessary */
  if(LALInferenceCheckVariable(currentParams,"logdistance")){
    distMpc=exp(*(REAL8 *) LALInferenceGetVariable(currentParams,"logdistance"));
    LALInferenceAddVariable(currentParams,"distance",&distMpc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }

  if(LALInferenceCheckVariable(currentParams,"logmc")){
    mc=exp(*(REAL8 *)LALInferenceGetVariable(currentParams,"logmc"));
    LALInferenceAddVariable(currentParams,"chirpmass",&mc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }
  /* determine source's sky location & orientation parameters: */
  ra        = *(REAL8*) LALInferenceGetVariable(currentParams, "rightascension"); /* radian      */
  dec       = *(REAL8*) LALInferenceGetVariable(currentParams, "declination");    /* radian      */
  psi       = *(REAL8*) LALInferenceGetVariable(currentParams, "polarisation");   /* radian      */
  GPSdouble = *(REAL8*) LALInferenceGetVariable(currentParams, "time");           /* GPS seconds */
  distMpc   = *(REAL8*) LALInferenceGetVariable(currentParams, "distance");       /* Mpc         */

  /* figure out GMST: */
  XLALGPSSetREAL8(&GPSlal, GPSdouble);
  gmst=XLALGreenwichMeanSiderealTime(&GPSlal);

  intrinsicParams = LALInferenceGetInstrinsicParams(currentParams);
  /*  TODO: add pointer to template function here.                                         */
  /*  (otherwise same parameters but different template will lead to no re-computation!!)  */

  chisquared = 0.0;
  /* loop over data (different interferometers): */
  dataPtr = data;

  while (dataPtr != NULL) {
    /* The parameters the Likelihood function can handle by itself    */
    /* (and which shouldn't affect the template function) are         */
    /* sky location (ra, dec), polarisation and signal arrival time.  */
    /* Note that the template function shifts the waveform to so that */
    /* t_c corresponds to the "time" parameter in                     */
    /* IFOdata->modelParams (set, e.g., from the trigger value).      */
    
    /* Reset log-likelihood */
    dataPtr->loglikelihood = 0.0;

    /* Compare parameter values with parameter values corresponding */
    /* to currently stored template; ignore "time" variable:        */
    if (LALInferenceCheckVariable(dataPtr->modelParams, "time")) {
      timeTmp = *(REAL8 *) LALInferenceGetVariable(dataPtr->modelParams, "time");
      LALInferenceRemoveVariable(dataPtr->modelParams, "time");
    }
    else timeTmp = GPSdouble;
    different = LALInferenceCompareVariables(dataPtr->modelParams, &intrinsicParams);
    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */

    if (different) { /* template needs to be re-computed: */
      LALInferenceCopyVariables(&intrinsicParams, dataPtr->modelParams);
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
      templt(dataPtr);

      if (dataPtr->modelDomain == LAL_SIM_DOMAIN_TIME) {
	/* TD --> FD. */
	LALInferenceExecuteFT(dataPtr);
      }
    }
    else { /* no re-computation necessary. Return back "time" value, do nothing else: */
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
    }

    /* Template is now in dataPtr->freqModelhPlus hCross. */

    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross,
                             (const REAL4(*)[3])dataPtr->detector->response,
			     ra, dec, psi, gmst);
    /* signal arrival time (relative to geocenter); */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location,
                                             ra, dec, &GPSlal);
    /* (negative timedelay means signal arrives earlier at Ifo than at geocenter, etc.)    */
    /* amount by which to time-shift template (not necessarily same as above "timedelay"): */
    timeshift =  (GPSdouble - (*(REAL8*) LALInferenceGetVariable(dataPtr->modelParams, "time"))) + timedelay;

    twopit    = LAL_TWOPI * timeshift;

    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;

    if (LALInferenceCheckVariable(currentParams, "crazyInjectionHLSign") &&
        *((INT4 *)LALInferenceGetVariable(currentParams, "crazyInjectionHLSign"))) {
      if (strstr(dataPtr->name, "H") || strstr(dataPtr->name, "L")) {
        FplusScaled *= -1.0;
        FcrossScaled *= -1.0;
      }
    }

    dataPtr->fPlus = FplusScaled;
    dataPtr->fCross = FcrossScaled;
    dataPtr->timeshift = timeshift;

    /* extract the element from the "df" vector that carries the current Ifo's name: */
    CHAR df_variable_name[64];
    sprintf(df_variable_name,"df_%s",dataPtr->name);
    if(LALInferenceCheckVariable(currentParams,df_variable_name)){
      degreesOfFreedom = *(REAL8*) LALInferenceGetVariable(currentParams,df_variable_name);
    }
    else {
      degreesOfFreedom = dataPtr->STDOF;
    }
    if (!(degreesOfFreedom>0)) {
      XLALPrintError(" ERROR in StudentTLogLikelihood(): degrees-of-freedom parameter must be positive.\n");
      XLAL_ERROR_REAL8(XLAL_EDOM);
    }

    /* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    lower = (UINT4)ceil(dataPtr->fLow / deltaF);
    upper = (UINT4)floor(dataPtr->fHigh / deltaF);
    FourDeltaToverN = 4.0 * deltaT / ((double) dataPtr->timeData->data->length);

    /* Employ a trick here for avoiding cos(...) and sin(...) in time
       shifting.  We need to multiply each template frequency bin by
       exp(-J*twopit*deltaF*i) = exp(-J*twopit*deltaF*(i-1)) +
       exp(-J*twopit*deltaF*(i-1))*(exp(-J*twopit*deltaF) - 1) .  This
       recurrance relation has the advantage that the error growth is
       O(sqrt(N)) for N repetitions. */
    
    /* Values for the first iteration: */
    re = cos(twopit*deltaF*lower);
    im = -sin(twopit*deltaF*lower);

    /* Incremental values, using cos(theta) - 1 = -2*sin(theta/2)^2 */
    dim = -sin(twopit*deltaF);
    dre = -2.0*sin(0.5*twopit*deltaF)*sin(0.5*twopit*deltaF);

    for (i=lower; i<=upper; ++i){
      /* degrees-of-freedom parameter (nu_j) for this particular frequency bin: */
      nu = degreesOfFreedom;
      /* (for now constant across frequencies)                                  */
      /* derive template (involving location/orientation parameters) from given plus/cross waveforms: */
      plainTemplateReal = FplusScaled * creal(dataPtr->freqModelhPlus->data->data[i])  
                          +  FcrossScaled * creal(dataPtr->freqModelhCross->data->data[i]);
      plainTemplateImag = FplusScaled * cimag(dataPtr->freqModelhPlus->data->data[i])  
                          +  FcrossScaled * cimag(dataPtr->freqModelhCross->data->data[i]);

      /* do time-shifting...            */
      /* (also un-do 1/deltaT scaling): */
      templateReal = (plainTemplateReal*re - plainTemplateImag*im) / deltaT;
      templateImag = (plainTemplateReal*im + plainTemplateImag*re) / deltaT;
      dataReal     = creal(dataPtr->freqData->data->data[i]) / deltaT;
      dataImag     = cimag(dataPtr->freqData->data->data[i]) / deltaT;
      /* compute squared difference & 'chi-squared': */
      diffRe       = dataReal - templateReal;         /* Difference in real parts...                     */
      diffIm       = dataImag - templateImag;         /* ...and imaginary parts, and...                  */
      diffSquared  = diffRe*diffRe + diffIm*diffIm ;  /* ...squared difference of the 2 complex figures. */
      singleFreqBinTerm = ((nu+2.0)/2.0) * log(1.0 + (FourDeltaToverN * diffSquared) / (nu * dataPtr->oneSidedNoisePowerSpectrum->data->data[i]));
      chisquared  += singleFreqBinTerm;   /* (This is a sum-of-squares, or chi^2, term in the Gaussian case, not so much in the Student-t case...)  */
      dataPtr->loglikelihood -= singleFreqBinTerm;

      /* Now update re and im for the next iteration. */
      newRe = re + re*dre - im*dim;
      newIm = im + re*dim + im*dre;

      re = newRe;
      im = newIm;
    }

    dataPtr = dataPtr->next;
  }
  loglikeli = -1.0 * chisquared; /* note (again): the log-likelihood is unnormalised! */
  LALInferenceClearVariables(&intrinsicParams);  
  return(loglikeli);
}

REAL8 LALInferenceFreqDomainLogLikelihood(LALInferenceVariables *currentParams, LALInferenceIFOData * data, 
                              LALInferenceTemplateFunction templt)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood.          */
/* Slightly slower but cleaner than							   */
/* UndecomposedFreqDomainLogLikelihood().          `		   */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/
{
  REAL8 loglikeli, totalChiSquared=0.0;
  LALInferenceIFOData *ifoPtr=data;
  COMPLEX16Vector *freqModelResponse=NULL;

  /* loop over data (different interferometers): */
  while (ifoPtr != NULL) {
    ifoPtr->loglikelihood = 0.0;

	if(freqModelResponse==NULL)
		freqModelResponse= XLALCreateCOMPLEX16Vector(ifoPtr->freqData->data->length);
	else
		freqModelResponse= XLALResizeCOMPLEX16Vector(freqModelResponse, ifoPtr->freqData->data->length);
	/*compute the response*/
	LALInferenceComputeFreqDomainResponse(currentParams, ifoPtr, templt, freqModelResponse);
        REAL8 temp = LALInferenceComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, ifoPtr->freqData->data)
          -2.0*LALInferenceComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, freqModelResponse)
          +LALInferenceComputeFrequencyDomainOverlap(ifoPtr, freqModelResponse, freqModelResponse);
	totalChiSquared+=temp;
        ifoPtr->loglikelihood -= 0.5*temp;

    ifoPtr = ifoPtr->next;
  }
  loglikeli = -0.5 * totalChiSquared; // note (again): the log-likelihood is unnormalised!
  XLALDestroyCOMPLEX16Vector(freqModelResponse);
  return(loglikeli);
}

REAL8 LALInferenceChiSquareTest(LALInferenceVariables *currentParams, LALInferenceIFOData * data, LALInferenceTemplateFunction templt)
/***************************************************************/
/* Chi-Square function.                                        */
/* Returns the chi square of a template:                       */
/* chisq= p * sum_i (dx_i)^2, with dx_i  =  <s,h>_i  - <s,h>/p */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/
{
  REAL8 ChiSquared=0.0, dxp, xp, x, norm, binPower, nextBin;
  REAL8 lowerF, upperF, deltaT, deltaF;
  REAL8 *segnorm;
  INT4  i, chisqPt, imax,  kmax, numBins=0;
  INT4  *chisqBin;
  LALInferenceIFOData *ifoPtr=data;
  COMPLEX16Vector *freqModelResponse=NULL;

  /* Allocate memory for local pointers */
  segnorm=XLALMalloc(sizeof(REAL8) * ifoPtr->freqData->data->length);
  chisqBin=XLALMalloc(sizeof(INT4) * (numBins + 1));

  /* loop over data (different interferometers): */
  while (ifoPtr != NULL) {
    if(freqModelResponse==NULL)
      freqModelResponse= XLALCreateCOMPLEX16Vector(ifoPtr->freqData->data->length);
    else
      freqModelResponse= XLALResizeCOMPLEX16Vector(freqModelResponse, ifoPtr->freqData->data->length);
    /*compute the response*/
    LALInferenceComputeFreqDomainResponse(currentParams, ifoPtr, templt, freqModelResponse);

    deltaT = ifoPtr->timeData->deltaT;
    deltaF = 1.0 / (((REAL8)ifoPtr->timeData->data->length) * deltaT);
    
    /* Store values of fLow and fHigh to use later */
    lowerF = ifoPtr->fLow;
    upperF = ifoPtr->fHigh;
   
    /* Generate bin boundaries */
    numBins = *(INT4*) LALInferenceGetVariable(currentParams, "numbins");
    kmax = floor(ifoPtr->fHigh / deltaF);
    imax = kmax > (INT4) ifoPtr->freqData->data->length-1 ? (INT4) ifoPtr->freqData->data->length-1 : kmax;
    
    memset(segnorm,0,sizeof(REAL8) * ifoPtr->freqData->data->length);
    norm = 0.0;
    
    for (i=1; i < imax; ++i){  	  	  
      norm += ((4.0 * deltaF * (creal(freqModelResponse->data[i])*creal(freqModelResponse->data[i])
              +cimag(freqModelResponse->data[i])*cimag(freqModelResponse->data[i]))) 
              / ifoPtr->oneSidedNoisePowerSpectrum->data->data[i]);
      segnorm[i] = norm;
    }


    memset(chisqBin,0,sizeof(INT4) * (numBins +1));

    binPower = norm / (REAL8) numBins;
    nextBin   = binPower;
    chisqPt   = 0;
    chisqBin[chisqPt++] = 0;

    for ( i = 1; i < imax; ++i )
    {
      if ( segnorm[i] >= nextBin )
      {
        chisqBin[chisqPt++] = i;
        nextBin += binPower;
        if ( chisqPt == numBins ) break;
      }
    }
    chisqBin[16]=imax;
    /* check that we have sucessfully allocated all the bins */
    if ( i == (INT4) ifoPtr->freqData->data->length && chisqPt != numBins )
    {
      /* if we have reaced the end of the template power vec and not
       * */
      /* allocated all the bin boundaries then there is a problem
       * */
      fprintf(stderr,"Error constructing frequency bins\n"); 
    }

    /* the last bin boundary is at can be at Nyquist since   */
    /* qtilde is zero above the ISCO of the current template */

    /* end */
    
    x = LALInferenceComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, freqModelResponse)/(sqrt(norm));
    
    ChiSquared=0.0;
    
    for (i=0; i < numBins; ++i){
      
      ifoPtr->fLow = chisqBin[i] * deltaF;
      ifoPtr->fHigh = chisqBin[i+1] * deltaF;
      
      xp = LALInferenceComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, freqModelResponse)/(sqrt(norm));
      dxp = ((REAL8) numBins) * xp - x;
      ChiSquared += (dxp * dxp);
      
    }
    ChiSquared = ChiSquared / (REAL8) numBins;
    ifoPtr->fLow = lowerF;
    ifoPtr->fHigh = upperF;
    
    printf("Chi-Square for %s\t=\t%f\n",ifoPtr->detector->frDetector.name,ChiSquared);
    
    ifoPtr = ifoPtr->next;
  }
  XLALFree(chisqBin);
  XLALFree(segnorm);
  return(ChiSquared);
}

void LALInferenceComputeFreqDomainResponse(LALInferenceVariables *currentParams, LALInferenceIFOData * dataPtr, 
                              LALInferenceTemplateFunction templt, COMPLEX16Vector *freqWaveform)
/***************************************************************/
/* Frequency-domain single-IFO response computation.           */
/* Computes response for a given template.                     */
/* Will re-compute template only if necessary                  */
/* (i.e., if previous, as stored in data->freqModelhCross,     */
/* was based on different parameters or template function).    */
/* Carries out timeshifting for a given detector               */
/* and projection onto this detector.                          */
/* Result stored in freqResponse, assumed to be correctly      */
/* initialized												   */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "distance"        (REAL8, Mpc, >0)                      */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/							  
{

	double ra, dec, psi, distMpc, gmst;
	
	double GPSdouble;
	double timeTmp;
	LIGOTimeGPS GPSlal;
	double timedelay;  /* time delay b/w iterferometer & geocenter w.r.t. sky location */
	double timeshift;  /* time shift (not necessarily same as above)                   */
	double deltaT, deltaF, twopit, re, im, dre, dim, newRe, newIm;

	int different;
	LALInferenceVariables intrinsicParams;
	LALStatus status;
	memset(&status,0,sizeof(status));
	
	double Fplus, Fcross;
	double FplusScaled, FcrossScaled;
	REAL8 plainTemplateReal, plainTemplateImag;
	UINT4 i;
	REAL8 mc;
	/* Fill in derived parameters if necessary */
	if(LALInferenceCheckVariable(currentParams,"logdistance")){
		distMpc=exp(*(REAL8 *) LALInferenceGetVariable(currentParams,"logdistance"));
		LALInferenceAddVariable(currentParams,"distance",&distMpc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
	}

	if(LALInferenceCheckVariable(currentParams,"logmc")){
		mc=exp(*(REAL8 *)LALInferenceGetVariable(currentParams,"logmc"));
		LALInferenceAddVariable(currentParams,"chirpmass",&mc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
	}
		
	
	/* determine source's sky location & orientation parameters: */
	ra        = *(REAL8*) LALInferenceGetVariable(currentParams, "rightascension"); /* radian      */
	dec       = *(REAL8*) LALInferenceGetVariable(currentParams, "declination");    /* radian      */
	psi       = *(REAL8*) LALInferenceGetVariable(currentParams, "polarisation");   /* radian      */
	GPSdouble = *(REAL8*) LALInferenceGetVariable(currentParams, "time");           /* GPS seconds */
	distMpc   = *(REAL8*) LALInferenceGetVariable(currentParams, "distance");       /* Mpc         */

		
	/* figure out GMST: */
	XLALGPSSetREAL8(&GPSlal, GPSdouble);
	gmst=XLALGreenwichMeanSiderealTime(&GPSlal);

    intrinsicParams = LALInferenceGetInstrinsicParams(currentParams);

	// TODO: add pointer to template function here.
	// (otherwise same parameters but different template will lead to no re-computation!!)
      
	/* The parameters the response function can handle by itself     */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
	/* t_c corresponds to the "time" parameter in                    */
	/* IFOdata->modelParams (set, e.g., from the trigger value).     */
    
    /* Compare parameter values with parameter values corresponding  */
    /* to currently stored template; ignore "time" variable:         */
    if (LALInferenceCheckVariable(dataPtr->modelParams, "time")) {
      timeTmp = *(REAL8 *) LALInferenceGetVariable(dataPtr->modelParams, "time");
      LALInferenceRemoveVariable(dataPtr->modelParams, "time");
    }
    else timeTmp = GPSdouble;
    different = LALInferenceCompareVariables(dataPtr->modelParams, &intrinsicParams);
    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */

    if (different) { /* template needs to be re-computed: */
      LALInferenceCopyVariables(&intrinsicParams, dataPtr->modelParams);
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
      templt(dataPtr);

      if (dataPtr->modelDomain == LAL_SIM_DOMAIN_TIME) {
	/* TD --> FD. */
	LALInferenceExecuteFT(dataPtr);
      }
    }
    else { /* no re-computation necessary. Return back "time" value, do nothing else: */
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
    }
    /* Template is now in dataPtr->freqModelhPlus and
       dataPtr->freqModelhCross */

    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross, (const REAL4(*)[3])dataPtr->detector->response,
			     ra, dec, psi, gmst);
		 
    /* signal arrival time (relative to geocenter); */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location,
                                             ra, dec, &GPSlal);
    /* (negative timedelay means signal arrives earlier at Ifo than at geocenter, etc.) */

    /* amount by which to time-shift template (not necessarily same as above "timedelay"): */
    timeshift =  (GPSdouble - (*(REAL8*) LALInferenceGetVariable(dataPtr->modelParams, "time"))) + timedelay;

    twopit    = LAL_TWOPI * timeshift;

    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;
    
    if (LALInferenceCheckVariable(currentParams, "crazyInjectionHLSign") &&
        *((INT4 *)LALInferenceGetVariable(currentParams, "crazyInjectionHLSign"))) {
      if (strstr(dataPtr->name, "H") || strstr(dataPtr->name, "L")) {
        FplusScaled *= -1.0;
        FcrossScaled *= -1.0;
      }
    }

	if(freqWaveform->length!=dataPtr->freqModelhPlus->data->length){
		printf("fW%d data%d\n", freqWaveform->length, dataPtr->freqModelhPlus->data->length);
		printf("Error!  Frequency data vector must be same length as original data!\n");
		exit(1);
	}
	
	deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);

#ifdef DEBUG
FILE* file=fopen("TempSignal.dat", "w");	
#endif
/* Employ a trick here for avoiding cos(...) and sin(...) in time
   shifting.  We need to multiply each template frequency bin by
   exp(-J*twopit*deltaF*i) = exp(-J*twopit*deltaF*(i-1)) +
   exp(-J*twopit*deltaF*(i-1))*(exp(-J*twopit*deltaF) - 1) .  This
   recurrance relation has the advantage that the error growth is
   O(sqrt(N)) for N repetitions. */
    
/* Values for the first iteration: */
 re = 1.0;
 im = 0.0;

 /* Incremental values, using cos(theta) - 1 = -2*sin(theta/2)^2 */
 dim = -sin(twopit*deltaF);
 dre = -2.0*sin(0.5*twopit*deltaF)*sin(0.5*twopit*deltaF);

	for(i=0; i<freqWaveform->length; i++){
		/* derive template (involving location/orientation parameters) from given plus/cross waveforms: */
		plainTemplateReal = FplusScaled * creal(dataPtr->freqModelhPlus->data->data[i])  
                          +  FcrossScaled * creal(dataPtr->freqModelhCross->data->data[i]);
		plainTemplateImag = FplusScaled * cimag(dataPtr->freqModelhPlus->data->data[i])  
                          +  FcrossScaled * cimag(dataPtr->freqModelhCross->data->data[i]);

		/* do time-shifting...             */
		freqWaveform->data[i].real_FIXME= (plainTemplateReal*re - plainTemplateImag*im);
		freqWaveform->data[i].imag_FIXME= (plainTemplateReal*im + plainTemplateImag*re);		
#ifdef DEBUG
		fprintf(file, "%lg %lg \t %lg\n", f, freqWaveform->data[i].re, freqWaveform->data[i].im);
#endif
		/* Now update re and im for the next iteration. */
		newRe = re + re*dre - im*dim;
		newIm = im + re*dim + im*dre;

		re = newRe;
		im = newIm;
	}
#ifdef DEBUG
fclose(file);
#endif
	LALInferenceClearVariables(&intrinsicParams);
}

REAL8 LALInferenceComputeFrequencyDomainOverlap(LALInferenceIFOData * dataPtr,
                                    COMPLEX16Vector * freqData1, 
                                    COMPLEX16Vector * freqData2)
{
  if (dataPtr==NULL || freqData1 ==NULL || freqData2==NULL){
  	XLAL_ERROR_REAL8(XLAL_EFAULT); 
  	}
  	
  int lower, upper, i;
  double deltaT, deltaF;
  
  double overlap=0.0;
  
  /* determine frequency range & loop over frequency bins: */
  deltaT = dataPtr->timeData->deltaT;
  deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
  lower = ceil(dataPtr->fLow / deltaF);
  upper = floor(dataPtr->fHigh / deltaF);
	
  for (i=lower; i<=upper; ++i){  	  	  
    overlap  += ((4.0*deltaF*(creal(freqData1->data[i])*creal(freqData2->data[i])+cimag(freqData1->data[i])*cimag(freqData2->data[i]))) 
                 / dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
  }

  return overlap;
}

REAL8 LALInferenceNullLogLikelihood(LALInferenceIFOData *data)
/*Identical to FreqDomainNullLogLikelihood                        */
{
	REAL8 loglikeli, totalChiSquared=0.0;
	LALInferenceIFOData *ifoPtr=data;
	
	/* loop over data (different interferometers): */
	while (ifoPtr != NULL) {
          ifoPtr->nullloglikelihood = 0.0;
          REAL8 temp = LALInferenceComputeFrequencyDomainOverlap(ifoPtr, ifoPtr->freqData->data, ifoPtr->freqData->data);
          totalChiSquared+=temp;
          ifoPtr->nullloglikelihood -= 0.5*temp;
		ifoPtr = ifoPtr->next;
	}
	loglikeli = -0.5 * totalChiSquared; // note (again): the log-likelihood is unnormalised!
	return(loglikeli);
}

static void extractDimensionlessVariableVector(LALInferenceVariables *currentParams, REAL8 *x, INT4 mode) {
  REAL8 m1, m2, d, iota, phi, psi, ra, dec, t, a1, a2, theta1, theta2, phi1, phi2;
  
  REAL8 mean[15];
  REAL8 Mc;
  
  memset(x, 0, 15*sizeof(REAL8));
  memset(mean, 0, 15*sizeof(REAL8));

  if (mode==0) {
    mean[0] = 16.0;
    mean[1] = 7.0;
    mean[2] = M_PI/2.0;
    mean[3] = M_PI;
    mean[4] = M_PI/2.0;
    mean[5] = M_PI;
    mean[6] = 0.0;
    mean[7] = 50.0;
    mean[8] = 0.0;
    mean[9] =0.5;
    mean[10]=0.5;
    mean[11]=M_PI/2.0;
    mean[12]=M_PI/2.0;
    mean[13]=M_PI;
    mean[14]=M_PI;
  } else if (mode==1) {
    mean[0] = 16.0;
    mean[1] = 7.0;
    mean[2] = 1.0*M_PI/4.0;
    mean[3] = 1.0*M_PI/2.0;
    mean[4] = 1.0*M_PI/4.0;
    mean[5] = 1.0*M_PI/2.0;
    mean[6] = -M_PI/4.0;
    mean[7] = 25.0;
    mean[8] = -0.03;
    mean[9] =0.2;
    mean[10]=0.2;
    mean[11]=1.0*M_PI/4.0;
    mean[12]=1.0*M_PI/4.0;
    mean[13]=1.0*M_PI/2.0;
    mean[14]=1.0*M_PI/2.0;
  } else if (mode==2) {
    /* set means of second mode to be 8 sigma from first mode */
    mean[0] = 16.0 + 8./scaling[0]*sqrt(CM[0][0]);
    mean[1] = 7.0 + 8./scaling[1]*sqrt(CM[1][1]);
    mean[2] = 1.0*M_PI/4.0 + 8./scaling[2]*sqrt(CM[2][2]);
    mean[3] = 1.0*M_PI/2.0 + 8./scaling[3]*sqrt(CM[3][3]);
    mean[4] = 1.0*M_PI/4.0 + 8./scaling[4]*sqrt(CM[4][4]);
    mean[5] = 1.0*M_PI/2.0 + 8./scaling[5]*sqrt(CM[5][5]);
    mean[6] = -M_PI/4.0 + 8./scaling[6]*sqrt(CM[6][6]);
    mean[7] = 25.0 + 8./scaling[7]*sqrt(CM[7][7]);
    mean[8] = -0.03 + 8./scaling[8]*sqrt(CM[8][8]);
    mean[9] =0.2 + 8./scaling[9]*sqrt(CM[9][9]);
    mean[10]=0.2 + 8./scaling[10]*sqrt(CM[10][10]);
    mean[11]=1.0*M_PI/4.0 + 8./scaling[11]*sqrt(CM[11][11]);
    mean[12]=1.0*M_PI/4.0 + 8./scaling[12]*sqrt(CM[12][12]);
    mean[13]=1.0*M_PI/2.0 + 8./scaling[13]*sqrt(CM[13][13]);
    mean[14]=1.0*M_PI/2.0 + 8./scaling[14]*sqrt(CM[14][14]);
  } else {
    printf("Error!  Unrecognized mode in analytic likelihood!\n");
    exit(1);
  }


  if (LALInferenceCheckVariable(currentParams,"m1")&&LALInferenceCheckVariable(currentParams,"m2"))
  {
    m1=*(REAL8 *)LALInferenceGetVariable(currentParams,"m1");
    m2=*(REAL8 *)LALInferenceGetVariable(currentParams,"m2");
  }
  else
  {
  	if (LALInferenceCheckVariable(currentParams, "chirpmass")) {
    	Mc = *(REAL8 *)LALInferenceGetVariable(currentParams, "chirpmass");
  	} else if (LALInferenceCheckVariable(currentParams, "logmc")) {
    	Mc = exp(*(REAL8 *)LALInferenceGetVariable(currentParams, "logmc"));
  	} else {
    	fprintf(stderr, "Could not find chirpmass or logmc in LALInferenceCorrelatedAnalyticLogLikelihood (in %s, line %d)\n", 
        	    __FILE__, __LINE__);
    	exit(1);
  	}

  	if (LALInferenceCheckVariable(currentParams, "massratio")) {
    	REAL8 eta = *(REAL8 *)LALInferenceGetVariable(currentParams, "massratio");
    	LALInferenceMcEta2Masses(Mc, eta, &m1, &m2);
  	} else if (LALInferenceCheckVariable(currentParams, "asym_massratio")) {
    	REAL8 q = *(REAL8 *)LALInferenceGetVariable(currentParams, "asym_massratio");
    	LALInferenceMcQ2Masses(Mc, q, &m1, &m2);
  	} else {
    	fprintf(stderr, "Could not find eta or q in LALInferenceCorrelatedAnalyticLogLikelihood (in %s, line %d)\n",
        	    __FILE__, __LINE__);
    	exit(1);
  	}
  }

  if (LALInferenceCheckVariable(currentParams, "distance")) {
    d = *(REAL8 *)LALInferenceGetVariable(currentParams, "distance");
  } else if (LALInferenceCheckVariable(currentParams, "logdistance")) {
    d = exp(*(REAL8 *)LALInferenceGetVariable(currentParams, "logdistance"));
  } else {
    fprintf(stderr, "Could not find distance or log(d) in LALInferenceCorrelatedAnalyticLogLikelihood (in %s, line %d)\n",
            __FILE__, __LINE__);
    exit(1);
  }

  iota = *(REAL8 *)LALInferenceGetVariable(currentParams, "inclination");
  psi = *(REAL8 *)LALInferenceGetVariable(currentParams, "polarisation");
  phi = *(REAL8 *)LALInferenceGetVariable(currentParams, "phase");
  ra = *(REAL8 *)LALInferenceGetVariable(currentParams, "rightascension");
  dec = *(REAL8 *)LALInferenceGetVariable(currentParams, "declination");
  t = *(REAL8 *)LALInferenceGetVariable(currentParams, "time");
  
  if (LALInferenceCheckVariable(currentParams, "a_spin1")) {
    a1 = *(REAL8 *)LALInferenceGetVariable(currentParams, "a_spin1");
  } else {
    a1 = 0.0;
  }

  if (LALInferenceCheckVariable(currentParams, "a_spin2")) {
    a2 = *(REAL8 *)LALInferenceGetVariable(currentParams, "a_spin2");
  } else {
    a2 = 0.0;
  }

  if (LALInferenceCheckVariable(currentParams, "phi_spin1")) {
    phi1 = *(REAL8 *)LALInferenceGetVariable(currentParams, "phi_spin1");
  } else {
    phi1 = 0.0;
  }
  
  if (LALInferenceCheckVariable(currentParams, "phi_spin2")) {
    phi2 = *(REAL8 *)LALInferenceGetVariable(currentParams, "phi_spin2");
  } else {
    phi2 = 0.0;
  }
  
  if (LALInferenceCheckVariable(currentParams, "theta_spin1")) {
    theta1 = *(REAL8 *)LALInferenceGetVariable(currentParams, "theta_spin1");
  } else {
    theta1 = 0.0;
  }

  if (LALInferenceCheckVariable(currentParams, "theta_spin2")) {
    theta2 = *(REAL8 *)LALInferenceGetVariable(currentParams, "theta_spin2");
  } else {
    theta2 = 0.0;
  }

  x[0] = scaling[0] * (m1    - mean[0]);
  x[1] = scaling[1] * (m2    - mean[1]);
  x[2] = scaling[2] * (iota  - mean[2]);
  x[3] = scaling[3] * (phi   - mean[3]);
  x[4] = scaling[4] * (psi   - mean[4]);
  x[5] = scaling[5] * (ra    - mean[5]);
  x[6] = scaling[6] * (dec   - mean[6]);
  x[7] = scaling[7] * (d     - mean[7]);
  x[8] = scaling[8] * (t     - mean[8]);
  x[9] = scaling[9] * (a1     - mean[9]);
  x[10]= scaling[10] * (a2     - mean[10]);
  x[11]= scaling[11] * (theta1 - mean[11]);
  x[12]= scaling[12] * (theta2 - mean[12]);
  x[13]= scaling[13] * (phi1   - mean[13]);
  x[14]= scaling[14] * (phi2   - mean[14]);
}

REAL8 LALInferenceCorrelatedAnalyticLogLikelihood(LALInferenceVariables *currentParams, 
                                                  LALInferenceIFOData UNUSED *data, 
                                                  LALInferenceTemplateFunction UNUSED template) {
  const INT4 DIM = 15;
  static gsl_matrix *LUCM = NULL;
  static gsl_permutation *LUCMPerm = NULL;
  INT4 mode = 0;
  
  REAL8 x[DIM];
  REAL8 xOrig[DIM];

  gsl_vector_view xView = gsl_vector_view_array(x, DIM);

  extractDimensionlessVariableVector(currentParams, x, mode);

  memcpy(xOrig, x, DIM*sizeof(REAL8));

  if (LUCM==NULL) {
    gsl_matrix_const_view CMView = gsl_matrix_const_view_array(&(CM[0][0]), DIM, DIM);
    int signum;

    LUCM = gsl_matrix_alloc(DIM, DIM);
    LUCMPerm = gsl_permutation_alloc(DIM);

    gsl_matrix_memcpy(LUCM, &(CMView.matrix));

    gsl_linalg_LU_decomp(LUCM, LUCMPerm, &signum);
  }

  gsl_linalg_LU_svx(LUCM, LUCMPerm, &(xView.vector));

  INT4 i;
  REAL8 sum = 0.0;
  for (i = 0; i < DIM; i++) {
    sum += xOrig[i]*x[i];
  }
  return -sum/2.0;
}

REAL8 LALInferenceBimodalCorrelatedAnalyticLogLikelihood(LALInferenceVariables *currentParams,
                                                  LALInferenceIFOData UNUSED *data,
                                                  LALInferenceTemplateFunction UNUSED template) {
  const INT4 DIM = 15;
  const INT4 MODES = 2;
  INT4 i, mode;
  REAL8 sum = 0.0;
  REAL8 a, b;
  static gsl_matrix *LUCM = NULL;
  static gsl_permutation *LUCMPerm = NULL;
  gsl_vector_view xView;

  REAL8 x[DIM];
  REAL8 xOrig[DIM];
  REAL8 exps[MODES];

  if (LUCM==NULL) {
    gsl_matrix_const_view CMView = gsl_matrix_const_view_array(&(CM[0][0]), DIM, DIM);
    int signum;

    LUCM = gsl_matrix_alloc(DIM, DIM);
    LUCMPerm = gsl_permutation_alloc(DIM);

    gsl_matrix_memcpy(LUCM, &(CMView.matrix));

    gsl_linalg_LU_decomp(LUCM, LUCMPerm, &signum);
  }

  for(mode = 1; mode < 3; mode++) {
    xView = gsl_vector_view_array(x, DIM);

    extractDimensionlessVariableVector(currentParams, x, mode);

    memcpy(xOrig, x, DIM*sizeof(REAL8));

    gsl_linalg_LU_svx(LUCM, LUCMPerm, &(xView.vector));

    sum = 0.0;
    for (i = 0; i < DIM; i++) {
      sum += xOrig[i]*x[i];
    }
    exps[mode-1] = -sum/2.0;
  }

  /* Assumes only two modes used from here on out */
  if (exps[0] > exps[1]) {
    a = exps[0];
    b = exps[1];
  } else {
    a = exps[1];
    b = exps[0];
  }

  /* attempt to keep returned values finite */
  return a + log1p(exp(b-a));
}

REAL8 LALInferenceRosenbrockLogLikelihood(LALInferenceVariables *currentParams,
                                          LALInferenceIFOData UNUSED *data,
                                          LALInferenceTemplateFunction UNUSED template) {
  const INT4 DIM = 15;
  REAL8 x[DIM];

  REAL8 sum = 0;
  INT4 mode = 0;
  INT4 i;

  extractDimensionlessVariableVector(currentParams, x, mode);

  for (i = 0; i < DIM-1; i++) {
    REAL8 oneMX = 1.0 - x[i];
    REAL8 dx = x[i+1] - x[i]*x[i];

    sum += oneMX*oneMX + 100.0*dx*dx;
  }

  return -sum;
}

REAL8 LALInferenceMarginalisedPhaseLogLikelihood(LALInferenceVariables *currentParams, LALInferenceIFOData * data,LALInferenceTemplateFunction templt)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood.          */
/* Analytically marginalised over phase and distance           */
/* See LIGO-T1300326 for details                               */
/* At a distance of 1 Mpc for phi_0=0                          */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/*   - "time"            (REAL8, GPS sec.)                     */
/***************************************************************/
{
  double Fplus, Fcross;
  double FplusScaled, FcrossScaled;
  double dataReal, dataImag;
  REAL8 loglikeli=0.0;
  REAL8 plainTemplateReal, plainTemplateImag;
  REAL8 templateReal, templateImag;
  int i, lower, upper;
  LALInferenceIFOData *dataPtr;
  double ra, dec, psi, distMpc, gmst;
  double GPSdouble;
  LIGOTimeGPS GPSlal;
  double timedelay;  /* time delay b/w iterferometer & geocenter w.r.t. sky location */
  double timeshift;  /* time shift (not necessarily same as above)                   */
  double deltaT, TwoDeltaToverN, deltaF, twopit, re, im, dre, dim, newRe, newIm;
  double timeTmp;
  double mc;
  int different;
  int logDistFlag=0;
  //noise model meta parameters
  gsl_matrix *lines   = NULL;//pointer to matrix holding line centroids
  gsl_matrix *widths  = NULL;//pointer to matrix holding line widths
  gsl_matrix *nparams = NULL;//pointer to matrix holding noise parameters
  
  gsl_matrix *psdBandsMin  = NULL;//pointer to matrix holding min frequencies for psd model
  gsl_matrix *psdBandsMax = NULL;//pointer to matrix holding max frequencies for psd model
  
  int Nblock = 1;            //number of frequency blocks per IFO
  int Nlines = 1;            //number of lines to be removed
  int psdFlag = 0;           //flag for including psd fitting
  int lineFlag = 0;          //flag for excluding lines from integration
  
  //line removal parameters
  if(LALInferenceCheckVariable(currentParams, "removeLinesFlag"))
    lineFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "removeLinesFlag"));
  if(lineFlag)
  {
    //Add line matrices to variable lists
    lines  = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_center");
    widths = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_width");
    Nlines = (int)lines->size2;
  }
  int lines_array[Nlines];
  int widths_array[Nlines];
  
  //check if psd parameters are included in the model
  if(LALInferenceCheckVariable(currentParams, "psdScaleFlag"))
    psdFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "psdScaleFlag"));
  if(psdFlag)
  {
    //if so, store current noise parameters in easily accessible matrix
    nparams = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdscale"));
    Nblock = (int)nparams->size2;
    
    psdBandsMin = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMin"));
    psdBandsMax = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMax"));
    
  }
  double alpha[Nblock];
  double lnalpha[Nblock];
  
  double psdBandsMin_array[Nblock];
  double psdBandsMax_array[Nblock];
  
  LALStatus status;
  memset(&status,0,sizeof(status));
  LALInferenceVariables intrinsicParams;
  logDistFlag=LALInferenceCheckVariable(currentParams, "logdistance");
  if(LALInferenceCheckVariable(currentParams,"logmc")){
    mc=exp(*(REAL8 *)LALInferenceGetVariable(currentParams,"logmc"));
    LALInferenceAddVariable(currentParams,"chirpmass",&mc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }
  
  if(LALInferenceCheckVariable(currentParams,"LAL_AMPORDER") && LALInferenceCheckVariable(currentParams,"LAL_APPROXIMANT"))
  {
    INT4 apprx=*(INT4 *)LALInferenceGetVariable(currentParams,"LAL_APPROXIMANT");
    /* This is only valid at Newtonian amp order for most templates, and at all orders for
     * the F-domain templates listed here. This will need to be updated if the F-domain phase
     * handling changes. */
    if(!( (*(INT4 *)LALInferenceGetVariable(currentParams,"LAL_AMPORDER") == LAL_PNORDER_NEWTONIAN) \
      || (apprx == TaylorF2 || apprx==TaylorF2RedSpin || apprx==TaylorF2RedSpinTidal || 
          apprx==SpinTaylorF2 || apprx==IMRPhenomA || apprx==IMRPhenomB || apprx==IMRPhenomC) ))
    {
      XLALPrintError("Error: Cannot use non-Newtonian amplitude order and analytically marginalise over phase!\n");
      exit(1);
    }
  }
  
  /* determine source's sky location & orientation parameters: */
  ra        = *(REAL8*) LALInferenceGetVariable(currentParams, "rightascension"); /* radian      */
  dec       = *(REAL8*) LALInferenceGetVariable(currentParams, "declination");    /* radian      */
  psi       = *(REAL8*) LALInferenceGetVariable(currentParams, "polarisation");   /* radian      */
  GPSdouble = *(REAL8*) LALInferenceGetVariable(currentParams, "time");           /* GPS seconds */
  
  if(logDistFlag)
    distMpc = exp(*(REAL8*)LALInferenceGetVariable(currentParams,"logdistance"));
  else
    distMpc   = *(REAL8*) LALInferenceGetVariable(currentParams, "distance");       /* Mpc         */
  
  REAL8 phi0=0.0;
  
  /* figure out GMST: */
  XLALGPSSetREAL8(&GPSlal, GPSdouble);
  gmst=XLALGreenwichMeanSiderealTime(&GPSlal);
  
  /* Create parameter set to pass to the template function */
  intrinsicParams = LALInferenceGetInstrinsicParams(currentParams);
  LALInferenceAddVariable(&intrinsicParams, "phase",&phi0,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  
  // TODO: add pointer to template function here.
  // (otherwise same parameters but different template will lead to no re-computation!!)
  
  /* loop over data (different interferometers): */
  dataPtr = data;
  UINT4 ifo=0;
  
  REAL8 S=0.,D=0.,R=0., Rre=0., Rim=0.;
  /* Need to compute S=h*h/S_h, D=d*d/S_h, R=h*d/S_h^2 */
  while (dataPtr != NULL) {
    /* The parameters the Likelihood function can handle by itself   */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
    /* t_c corresponds to the "time" parameter in                    */
    /* IFOdata->modelParams (set, e.g., from the trigger value).     */
    
    /* Reset log-likelihood */
    dataPtr->loglikelihood = 0.0;
    
    /* Compare parameter values with parameter values corresponding  */
    /* to currently stored template; ignore "time" variable:         */
    if (LALInferenceCheckVariable(dataPtr->modelParams, "time")) {
      timeTmp = *(REAL8 *) LALInferenceGetVariable(dataPtr->modelParams, "time");
      LALInferenceRemoveVariable(dataPtr->modelParams, "time");
    }
    else timeTmp = GPSdouble;
    different = LALInferenceCompareVariables(dataPtr->modelParams, &intrinsicParams);
    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */
    
    if (different) { /* template needs to be re-computed: */
      LALInferenceCopyVariables(&intrinsicParams, dataPtr->modelParams);
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
      templt(dataPtr);
      if(XLALGetBaseErrno()==XLAL_FAILURE) /* Template generation failed in a known way, set -Inf likelihood */
        return(-DBL_MAX);
      
      if (dataPtr->modelDomain == LAL_SIM_DOMAIN_TIME) {
        LALInferenceExecuteFT(dataPtr);
        /* note that the dataPtr->modelParams "time" element may have changed here!! */
        /* (during "template()" computation)  */
      }
    }
    else { /* no re-computation necessary. Return back "time" value, do nothing else: */
      LALInferenceAddVariable(dataPtr->modelParams, "time", &timeTmp, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
    }
    
    /*-- Template is now in dataPtr->freqModelhPlus and dataPtr->freqModelhCross. --*/
    /*-- (Either freshly computed or inherited.)                            --*/
    
    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross,
                             (const REAL4(*)[3])dataPtr->detector->response,
                             ra, dec, psi, gmst);
    /* signal arrival time (relative to geocenter); */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location,
                                             ra, dec, &GPSlal);
    /* (negative timedelay means signal arrives earlier at Ifo than at geocenter, etc.) */
    /* amount by which to time-shift template (not necessarily same as above "timedelay"): */
    timeshift =  (GPSdouble - (*(REAL8*) LALInferenceGetVariable(dataPtr->modelParams, "time"))) + timedelay;
    twopit    = LAL_TWOPI * timeshift;
    
    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;
    
    /* Check for wrong calibration sign */
    if (LALInferenceCheckVariable(currentParams, "crazyInjectionHLSign") &&
        *((INT4 *)LALInferenceGetVariable(currentParams, "crazyInjectionHLSign"))) {
      if (strstr(dataPtr->name, "H") || strstr(dataPtr->name, "L")) {
        FplusScaled *= -1.0;
        FcrossScaled *= -1.0;
      }
    }
    
    dataPtr->fPlus = FplusScaled;
    dataPtr->fCross = FcrossScaled;
    dataPtr->timeshift = timeshift;
    

    //Set up noise PSD meta parameters
    for(i=0; i<Nblock; i++)
    {
      if(psdFlag)
      {
	alpha[i]   = gsl_matrix_get(nparams,ifo,i);
	lnalpha[i] = log(alpha[i]);
	
	psdBandsMin_array[i] = gsl_matrix_get(psdBandsMin,ifo,i);
	psdBandsMax_array[i] = gsl_matrix_get(psdBandsMax,ifo,i);
	
      }
      else
      {
	alpha[i]=1.0;
	lnalpha[i]=0.0;
      }
    }
    
    //Set up psd line arrays
    for(INT4 j=0;j<Nlines;j++)
    {
      if(lineFlag)
      {
	
	//find range of fourier fourier bins which are excluded from integration
	lines_array[j]  = (int)gsl_matrix_get(lines,ifo,j);
	widths_array[j] = (int)gsl_matrix_get(widths,ifo,j);
      }
      else
      {
	lines_array[j]=0;
	widths_array[j]=0;
      }
    }
    /* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    lower = (UINT4)ceil(dataPtr->fLow / deltaF);
    upper = (UINT4)floor(dataPtr->fHigh / deltaF);
    TwoDeltaToverN = 2.0 * deltaT / ((double) dataPtr->timeData->data->length);
    
    /* Employ a trick here for avoiding cos(...) and sin(...) in time
     *       shifting.  We need to multiply each template frequency bin by
     *       exp(-J*twopit*deltaF*i) = exp(-J*twopit*deltaF*(i-1)) +
     *       exp(-J*twopit*deltaF*(i-1))*(exp(-J*twopit*deltaF) - 1) .  This
     *       recurrance relation has the advantage that the error growth is
     *       O(sqrt(N)) for N repetitions. */
    
    /* Values for the first iteration: */
    re = cos(twopit*deltaF*lower);
    im = -sin(twopit*deltaF*lower);
    
    /* Incremental values, using cos(theta) - 1 = -2*sin(theta/2)^2 */
    dim = -sin(twopit*deltaF);
    dre = -2.0*sin(0.5*twopit*deltaF)*sin(0.5*twopit*deltaF);
    /* Loop over freq domain */
    for (i=lower; i<=upper; ++i){

    /*only sum over bins which are outside of excluded regions */
    if(LALInferenceLineSwitch(lineFlag, Nlines, lines_array, widths_array, i))
    {
	
	/* derive template (involving location/orientation parameters) from given plus/cross waveforms: */
	plainTemplateReal = FplusScaled * creal(dataPtr->freqModelhPlus->data->data[i])
	+  FcrossScaled * creal(dataPtr->freqModelhCross->data->data[i]);
	plainTemplateImag = FplusScaled * cimag(dataPtr->freqModelhPlus->data->data[i])
	+  FcrossScaled * cimag(dataPtr->freqModelhCross->data->data[i]);
	/* do time-shifting...             */
	/* (also un-do 1/deltaT scaling): */
	templateReal = (plainTemplateReal*re - plainTemplateImag*im) / deltaT;
	templateImag = (plainTemplateReal*im + plainTemplateImag*re) / deltaT;
	dataReal     = creal(dataPtr->freqData->data->data[i]) / deltaT;
	dataImag     = cimag(dataPtr->freqData->data->data[i]) / deltaT;
	
	REAL8 S_h = dataPtr->oneSidedNoisePowerSpectrum->data->data[i];
	S+=TwoDeltaToverN*((templateReal*templateReal)+(templateImag*templateImag)) / S_h;
	D+=TwoDeltaToverN*(dataReal*dataReal + dataImag*dataImag)/S_h;
	REAL8 dhstarRe=dataReal*templateReal+dataImag*templateImag; /* (-i^2=1) */
	REAL8 dhstarIm=dataImag*templateReal-dataReal*templateImag;
	Rre+=TwoDeltaToverN*dhstarRe/S_h;
	Rim+=TwoDeltaToverN*dhstarIm/S_h;
	/* Add noise PSD parameters to the model */
	if(psdFlag)
	{
	  for(INT4 j=0; j<Nblock; j++)
	  {
	    if (i >= psdBandsMin_array[j] && i <= psdBandsMax_array[j])
	    {
	      S /= alpha[j];
	      S += lnalpha[j];
	      D /=alpha[j];
	      D+= lnalpha[j];
	      Rre /= alpha[j];
	      Rre += lnalpha[j];
	      Rim /= alpha[j];
	      Rim += lnalpha[j];
	    }
	  }
	}
      }

      /* Now update re and im for the next iteration. */
      newRe = re + re*dre - im*dim;
      newIm = im + re*dim + im*dre;
      
      re = newRe;
      im = newIm;
    }
    dataPtr = dataPtr->next;
  }
  R=2.0*sqrt(Rre*Rre+Rim*Rim);
  gsl_sf_result result;
  REAL8 I0x=0.0;
  if(GSL_SUCCESS==gsl_sf_bessel_I0_scaled_e(R, &result))
  {
    I0x=result.val;
  }
  else printf("ERROR: Cannot calculate I0(%lf)\n",R);
  /* This is marginalised over phase only for now */
  REAL8 thislogL=-(S+D) + log(I0x) + R ;
  loglikeli=thislogL;
  LALInferenceClearVariables(&intrinsicParams);
  return(loglikeli);
}

static double logaddexp(double x, double y) {
  if (x == -INFINITY && y == -INFINITY) {
    /* 0 + 0 == 0 */
    return -INFINITY;
  } else if (x > y) {
    return x + log1p(exp(y-x));
  } else {
    return y + log1p(exp(x-y));
  }
}

static double log_quadratic_integral_log(double h, double lx0, double lx1, double lx2) {
  double a = (lx0 - 2.0*lx1 + lx2)/(2.0*h*h);
  double b = -(3.0*lx0 - 4.0*lx1 + lx2)/(2.0*h);
  double c = lx0;

  if (lx0 == lx1 && lx1 == lx2) return log(2.0) + log(h) + lx0;

  if (a > 0.0) {
    if (b + 2.0*a*h > 0.0) {
      double log_norm = c - 0.5*log(a) + 2.0*h*(b + 2.0*a*h);
      double other_term = gsl_sf_dawson((b+4.0*a*h)/(2.0*sqrt(a))) - exp(-2.0*h*(b+2.0*a*h))*gsl_sf_dawson(b/(2.0*sqrt(a)));
      return log_norm + log(other_term);
    } else {
      double log_norm = c - 0.5*log(a);
      double other_term = exp(2.0*h*(b + 2.0*a*h))*gsl_sf_dawson((b + 4.0*a*h)/(2.0*sqrt(a))) - gsl_sf_dawson(b/(2.0*sqrt(a)));
      return log_norm + log(other_term);
    }
  } else if (a < 0.0) {
    double A = -a;
    return c + b*b/(4.0*A) + log(sqrt(M_PI/A)/2.0) + log(gsl_sf_erf(b/(2.0*sqrt(A))) + gsl_sf_erf((4.0*A*h-b)/(2.0*sqrt(A))));
  } else {
    if (b > 0) {
      return c - log(b) + 2.0*b*h + log(1.0 - exp(-2.0*b*h));
    } else {
      double B = -b;
      return c - log(B) + log1p(-exp(-2.0*B*h));
    }
  }
}

static double integrate_interpolated_log(double h, double *log_ys, size_t n) {
  size_t i;
  double log_integral = -INFINITY;
  
  for (i = 0; i < n-2; i++) {
    double l0, l1, l2;

    l0 = log_ys[i];
    l1 = log_ys[i+1];
    l2 = log_ys[i+2];
    
    log_integral = logaddexp(log_integral, log_quadratic_integral_log(h, l0, l1, l2));
  }

  log_integral = logaddexp(log_integral, log_quadratic_integral_log(h, log_ys[n-2], log_ys[n-1], log_ys[0]));

  return log_integral - log(2.0);
}

REAL8 LALInferenceMarginalisedTimeLogLikelihood(LALInferenceVariables *currentParams, LALInferenceIFOData * data, 
                              LALInferenceTemplateFunction templt)
/***************************************************************/
/* (log-) likelihood function.                                 */
/* Returns the non-normalised logarithmic likelihood.          */
/* Analytically marginalised over time                         */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* Required (`currentParams') parameters are:                  */
/*   - "rightascension"  (REAL8, radian, 0 <= RA <= 2pi)       */
/*   - "declination"     (REAL8, radian, -pi/2 <= dec <=pi/2)  */
/*   - "polarisation"    (REAL8, radian, 0 <= psi <= ?)        */
/***************************************************************/
{
  double Fplus, Fcross;
  double FplusScaled, FcrossScaled;
  double dataReal, dataImag;
  REAL8 templateReal, templateImag, plainTemplateReal, plainTemplateImag;
  UINT4 i, j, lower, upper, ifo;
  LALInferenceIFOData *dataPtr;
  double ra, dec, psi, distMpc, gmst;
  LIGOTimeGPS GPSlal;
  double chisquared;
  double loglike;
  double deltaT=0.0, TwoDeltaToverN, deltaF;
  double timedelay, timeshift, twopitimeshift;
  int different;
	double mc;
	UINT4 logDistFlag=0;
  LALStatus status;
  memset(&status,0,sizeof(status));
  LALInferenceVariables intrinsicParams;
  int margphi;

  if(data==NULL) {XLAL_ERROR_REAL8(XLAL_EINVAL,"ERROR: Encountered NULL data pointer in likelihood\n");}

  //noise model meta parameters
  gsl_matrix *lines   = NULL;//pointer to matrix holding line centroids
  gsl_matrix *widths  = NULL;//pointer to matrix holding line widths
  gsl_matrix *nparams = NULL;//pointer to matrix holding noise parameters

  gsl_matrix *psdBandsMin  = NULL;//pointer to matrix holding min frequencies for psd model
  gsl_matrix *psdBandsMax = NULL;//pointer to matrix holding max frequencies for psd model

  UINT4 Nblock = 1;            //number of frequency blocks per IFO
  UINT4 Nlines = 1;            //number of lines to be removed
  UINT4 psdFlag = 0;           //flag for including psd fitting
  UINT4 lineFlag = 0;          //flag for excluding lines from integration

  //line removal parameters
  if(LALInferenceCheckVariable(currentParams, "removeLinesFlag"))
    lineFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "removeLinesFlag"));
  if(lineFlag)
  {
    //Add line matrices to variable lists
    lines  = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_center");
    widths = *(gsl_matrix **)LALInferenceGetVariable(currentParams, "line_width");
    Nlines = (int)lines->size2;
  }
  int lines_array[Nlines];
  int widths_array[Nlines];

  //check if psd parameters are included in the model
  if(LALInferenceCheckVariable(currentParams, "psdScaleFlag"))
    psdFlag = *((INT4 *)LALInferenceGetVariable(currentParams, "psdScaleFlag"));
  if(psdFlag)
  {
    //if so, store current noise parameters in easily accessible matrix
    nparams = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdscale"));
    Nblock = (int)nparams->size2;

    psdBandsMin = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMin"));
    psdBandsMax = *((gsl_matrix **)LALInferenceGetVariable(currentParams, "psdBandsMax"));

  }
  double alpha[Nblock];
  double lnalpha[Nblock];

  double psdBandsMin_array[Nblock];
  double psdBandsMax_array[Nblock];

  if (LALInferenceCheckVariable(currentParams, "margtimephi")) {
    margphi = *(INT4 *)LALInferenceGetVariable(currentParams, "margtimephi");
  } else {
    margphi = 0;
  }

  logDistFlag=LALInferenceCheckVariable(currentParams, "logdistance");
  if(LALInferenceCheckVariable(currentParams,"logmc")){
    mc=exp(*(REAL8 *)LALInferenceGetVariable(currentParams,"logmc"));
    LALInferenceAddVariable(currentParams,"chirpmass",&mc,LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_OUTPUT);
  }

  /* determine source's sky location & orientation parameters: */
  ra        = *(REAL8*) LALInferenceGetVariable(currentParams, "rightascension"); /* radian      */
  dec       = *(REAL8*) LALInferenceGetVariable(currentParams, "declination");    /* radian      */
  psi       = *(REAL8*) LALInferenceGetVariable(currentParams, "polarisation");   /* radian      */
  if(logDistFlag)
    distMpc = exp(*(REAL8*)LALInferenceGetVariable(currentParams,"logdistance"));
  else
    distMpc = *(REAL8*) LALInferenceGetVariable(currentParams, "distance");       /* Mpc         */

  /* loop over data (different interferometers): */
  dataPtr = data;

  /* Setup times to integrate over */
  UINT4 freq_length = dataPtr->freqData->data->length;
  UINT4 time_length = 2*(freq_length-1);
  REAL8 approx_tc = XLALGPSGetREAL8(&(dataPtr->freqData->epoch)) + time_length*deltaT - 2.0;
  COMPLEX16Vector * dh_S_tilde = XLALCreateCOMPLEX16Vector(freq_length);
  COMPLEX16Vector * dh_S_tilde_im = NULL;
  REAL8Vector * dh_S = XLALCreateREAL8Vector(time_length);
  REAL8Vector * dh_S_im = NULL;

  if (margphi) {
    dh_S_tilde_im = XLALCreateCOMPLEX16Vector(freq_length);
    dh_S_im = XLALCreateREAL8Vector(time_length);
    if (dh_S_tilde_im == NULL || dh_S_im == NULL) {
      XLAL_ERROR_REAL8(XLAL_ENOMEM, "Out of memory in LALInferenceMarginalisedTimeLogLikelihood.");
    }
    for (i = 0; i < freq_length; i++) {
      dh_S_tilde_im->data[i].real_FIXME = 0.0;
      dh_S_tilde_im->data[i].imag_FIXME = 0.0;
    }
  }

  if (dh_S_tilde ==NULL || dh_S == NULL)
    XLAL_ERROR_REAL8(XLAL_ENOMEM, "Out of memory in LALInferenceMarginalisedTimeLogLikelihood.");

  for (i = 0; i < freq_length; i++) {
    dh_S_tilde->data[i].real_FIXME = 0.0;
    dh_S_tilde->data[i].imag_FIXME = 0.0;
  }

  /* Calculate gmst at upper bound of prior for antenna pattern calculation */
  XLALGPSSetREAL8(&GPSlal, approx_tc);
  gmst=XLALGreenwichMeanSiderealTime(&GPSlal);

  intrinsicParams = LALInferenceGetInstrinsicParams(currentParams);

  loglike = 0.0;

  ifo=0;

  while (dataPtr != NULL) {
    REAL8 reshift, imshift, dreshift, dimshift, newReshift, newImshift;

    /* The parameters the Likelihood function can handle by itself   */
    /* (and which shouldn't affect the template function) are        */
    /* sky location (ra, dec), polarisation and signal arrival time. */
    /* Note that the template function shifts the waveform to so that*/
	/* t_c corresponds to the "time" parameter in                    */
	/* IFOdata->modelParams (set, e.g., from the trigger value).     */
    
    /* Reset log-likelihood.  Marginalization over time ruins the
       relationship that log(L) = sum_i log(L_i), so just set detector
       log(L) to 0.0 */
    dataPtr->loglikelihood = 0.0;
    chisquared = 0.0;

    /* "different" now may also mean that "dataPtr->modelParams" */
    /* wasn't allocated yet (as in the very 1st iteration).      */
    different = LALInferenceCompareVariables(dataPtr->modelParams, &intrinsicParams);

    if (different) { /* template needs to be re-computed: */
      LALInferenceCopyVariables(&intrinsicParams, dataPtr->modelParams);
      LALInferenceAddVariable(dataPtr->modelParams, "time", &approx_tc, LALINFERENCE_REAL8_t,LALINFERENCE_PARAM_LINEAR);
      if (margphi) {
	double pi2 = M_PI / 2.0;
	LALInferenceAddVariable(dataPtr->modelParams, "phase", &pi2, LALINFERENCE_REAL8_t, LALINFERENCE_PARAM_LINEAR);
      }
      templt(dataPtr);
      if(XLALGetBaseErrno()==XLAL_FAILURE) /* Template generation failed in a known way, set -Inf likelihood */
          return(-DBL_MAX);

      if (dataPtr->modelDomain == LAL_SIM_DOMAIN_TIME) {
        /* TD --> FD. */
        LALInferenceExecuteFT(dataPtr);
      }
    }

    /* Template is now in dataPtr->timeFreqModelhPlus and hCross */

    /* Time between arrival at geocenter and arrival at detector */
    timedelay = XLALTimeDelayFromEarthCenter(dataPtr->detector->location,
                                             ra, dec, &GPSlal);
    timeshift = approx_tc - *(REAL8 *)LALInferenceGetVariable(dataPtr->modelParams, "time") + timedelay;
    twopitimeshift = 2.0*M_PI*timeshift;

    /* determine beam pattern response (F_plus and F_cross) for given Ifo: */
    XLALComputeDetAMResponse(&Fplus, &Fcross, (const REAL4(*)[3])dataPtr->detector->response, ra, dec, psi, gmst);

    /* include distance (overall amplitude) effect in Fplus/Fcross: */
    FplusScaled  = Fplus  / distMpc;
    FcrossScaled = Fcross / distMpc;

    if (LALInferenceCheckVariable(currentParams, "crazyInjectionHLSign") &&
        *((INT4 *)LALInferenceGetVariable(currentParams, "crazyInjectionHLSign"))) {
      if (strstr(dataPtr->name, "H") || strstr(dataPtr->name, "L")) {
        FplusScaled *= -1.0;
        FcrossScaled *= -1.0;
      }
    }

    dataPtr->fPlus = FplusScaled;
    dataPtr->fCross = FcrossScaled;

    /* determine frequency range & loop over frequency bins: */
    deltaT = dataPtr->timeData->deltaT;
    deltaF = 1.0 / (((double)dataPtr->timeData->data->length) * deltaT);
    lower = (UINT4)ceil(dataPtr->fLow / deltaF);
    upper = (UINT4)floor(dataPtr->fHigh / deltaF);
    TwoDeltaToverN = 2.0 * deltaT / ((double) dataPtr->timeData->data->length);
    
    /* Employ a trick here for avoiding cos(...) and sin(...) in time
       shifting.  We need to multiply each template frequency bin by
       exp(-J*twopit*deltaF*i) = exp(-J*twopit*deltaF*(i-1)) +
       exp(-J*twopit*deltaF*(i-1))*(exp(-J*twopit*deltaF) - 1) .  This
       recurrance relation has the advantage that the error growth is
       O(sqrt(N)) for N repetitions. */
    
    /* Values for the first iteration: */
    reshift = cos(twopitimeshift*deltaF*lower);
    imshift = -sin(twopitimeshift*deltaF*lower);

    /* Incremental values, using cos(theta) - 1 = -2*sin(theta/2)^2 */
    dimshift = -sin(twopitimeshift*deltaF);
    dreshift = -2.0*sin(0.5*twopitimeshift*deltaF)*sin(0.5*twopitimeshift*deltaF);

    //Set up noise PSD meta parameters
    for(i=0; i<Nblock; i++)
    {
      if(psdFlag)
      {
        alpha[i]   = gsl_matrix_get(nparams,ifo,i);
        lnalpha[i] = log(alpha[i]);

        psdBandsMin_array[i] = gsl_matrix_get(psdBandsMin,ifo,i);
        psdBandsMax_array[i] = gsl_matrix_get(psdBandsMax,ifo,i);
      }
      else
      {
        alpha[i]=1.0;
        lnalpha[i]=0.0;
      }
    }

    //Set up psd line arrays
    for(j=0;j<Nlines;j++)
    {
      if(lineFlag)
      {
        //find range of fourier fourier bins which are excluded from integration
        lines_array[j]  = (int)gsl_matrix_get(lines,ifo,j);
        widths_array[j] = (int)gsl_matrix_get(widths,ifo,j);
      }
      else
      {
        lines_array[j]=0;
        widths_array[j]=0;
      }
    }

    for (i=lower; i<=upper; ++i){
      if(LALInferenceLineSwitch(lineFlag, Nlines, lines_array, widths_array, i)) {
          REAL8 alph=0.0, lnalph=0.0;

          if (psdFlag) {
              for (j=0; j<Nblock; j++) {
                if (i >= psdBandsMin_array[j] && i <= psdBandsMax_array[j]) {
                    alph = alpha[j];
                    lnalph = lnalpha[j];
                }
              }
          } else {
              alph = 1.;
              lnalph = 0.;
          }

          plainTemplateReal = FplusScaled * creal(dataPtr->freqModelhPlus->data->data[i])  
	    +  FcrossScaled * creal(dataPtr->freqModelhCross->data->data[i]);
          plainTemplateImag = FplusScaled * cimag(dataPtr->freqModelhPlus->data->data[i])  
                              +  FcrossScaled * cimag(dataPtr->freqModelhCross->data->data[i]);

          plainTemplateReal /= deltaT;
          plainTemplateImag /= deltaT;

	  templateReal = reshift*plainTemplateReal - imshift*plainTemplateImag;
	  templateImag = imshift*plainTemplateReal + reshift*plainTemplateImag;

          dataReal     = creal(dataPtr->freqData->data->data[i]) / deltaT;
          dataImag     = cimag(dataPtr->freqData->data->data[i]) / deltaT;

          REAL8 dh_S_real, dh_S_imag;

	  /* Terms in conj(d)*h */
          dh_S_real = dataReal * templateReal + dataImag * templateImag;
          dh_S_imag = dataReal * templateImag - dataImag * templateReal;

	  if (margphi) {
	    dh_S_tilde->data[i].real_FIXME += dh_S_real * TwoDeltaToverN / (alph * dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
	    dh_S_tilde_im->data[i].imag_FIXME += dh_S_imag * TwoDeltaToverN / (alph * dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
	  } else {
	    dh_S_tilde->data[i].real_FIXME += dh_S_real * TwoDeltaToverN / (alph * dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
	    dh_S_tilde->data[i].imag_FIXME += dh_S_imag * TwoDeltaToverN / (alph * dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
	  }

          chisquared += 2.0 * TwoDeltaToverN * (templateReal*templateReal + templateImag*templateImag 
                                                + dataReal*dataReal + dataImag*dataImag)
                        / (alph * dataPtr->oneSidedNoisePowerSpectrum->data->data[i]);
          chisquared += lnalph;

      }

      newReshift = reshift + reshift*dreshift - imshift*dimshift;
      newImshift = imshift + reshift*dimshift + imshift*dreshift;

      reshift = newReshift;
      imshift = newImshift;
    }

    loglike -= 0.5 * chisquared;

    ifo++; //increment IFO counter for noise parameters
    dataPtr = dataPtr->next;
  }

  /* LALSuite only performs complex->real reverse-FFTs. We reverse the
   *  array to effectively perform a complex->real forward-FFT.  This
   *  is the reason for the "-i*deltaT" above.  */
  dh_S_tilde->data[0].imag_FIXME = 0.;
  XLALREAL8ReverseFFT(dh_S, dh_S_tilde, data->freqToTimeFFTPlan);

  if (margphi) {
    dh_S_tilde_im->data[0].imag_FIXME = 0.0;
    XLALREAL8ReverseFFT(dh_S_im, dh_S_tilde_im, data->freqToTimeFFTPlan);
  }

  if (margphi) {
    /* We've got the real and imaginary parts of the FFT in the two
       arrays.  Now combine them into one Bessel function. */
    for (i = 0; i < time_length; i++) {
      double x = sqrt(dh_S->data[i]*dh_S->data[i] + dh_S_im->data[i]*dh_S_im->data[i]);
      dh_S->data[i] = log(gsl_sf_bessel_I0_scaled(x)) + fabs(x);
    }
  }     

  loglike += integrate_interpolated_log(deltaT, dh_S->data, time_length) - log(time_length*deltaT);

  XLALDestroyCOMPLEX16Vector(dh_S_tilde);
  XLALDestroyREAL8Vector(dh_S);
  if (margphi) {
    XLALDestroyCOMPLEX16Vector(dh_S_tilde_im);
    XLALDestroyREAL8Vector(dh_S_im);
  }

  LALInferenceClearVariables(&intrinsicParams);
  return(loglike);
}

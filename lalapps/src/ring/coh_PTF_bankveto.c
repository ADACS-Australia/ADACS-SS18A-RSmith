#include "coh_PTF.h"

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

UINT4 coh_PTF_read_sub_bank(
struct coh_PTF_params   *params,
InspiralTemplate        **PTFBankTemplates)
{
  UINT4 i,numTemplates;
  InspiralTemplate *bankTemplate;

  numTemplates = InspiralTmpltBankFromLIGOLw( PTFBankTemplates,
      params->bankVetoBankName,-1, -1 );

  bankTemplate = *PTFBankTemplates;

  for (i=0; (i < numTemplates); bankTemplate = bankTemplate->next,i++)
  {
    bankTemplate->fLower = params->lowTemplateFrequency;
  }
  return numTemplates;
}

void coh_PTF_initialise_sub_bank(
struct coh_PTF_params   *params,
InspiralTemplate        *PTFBankTemplates,
FindChirpTemplate       *bankFcTmplts,
UINT4                    subBankSize,
UINT4                    numPoints)
{
  /* I think this function is now unused and can be removed */
  UINT4 i;
  srand(params->randomSeed);

  REAL4 maxmass1 = 30.;
  REAL4 minmass1 = 1.;
  REAL4 minmass2 = 1.;
  REAL4 maxchi = 0.;
  REAL4 minchi = 0.;
  REAL4 maxkappa = 1.;
  REAL4 minkappa = 1.;
 
  for ( i=0 ; i < subBankSize ; i++ )
  {
    bankFcTmplts[i].PTFQtilde =
      XLALCreateCOMPLEX8VectorSequence( 1, numPoints / 2 + 1 );
    PTFBankTemplates[i].approximant = FindChirpPTF;
    PTFBankTemplates[i].order = LAL_PNORDER_TWO;
    PTFBankTemplates[i].mass1 = pow(rand()/(float)RAND_MAX,2)*(maxmass1-minmass1)+minmass1;
    PTFBankTemplates[i].mass2 = rand()/(float)RAND_MAX*(PTFBankTemplates[i].mass1 - minmass2) + minmass2;
    PTFBankTemplates[i].chi = rand()/(float)RAND_MAX*(maxchi-minchi)+minchi;
    PTFBankTemplates[i].kappa = rand()/(float)RAND_MAX*(maxkappa-minkappa)+minkappa;
    PTFBankTemplates[i].fLower = 38.;
  }
}

/* FIXME length of bankeigen* is hardcoded. Limits to 50 bank templates */
REAL4 coh_PTF_calculate_bank_veto(
UINT4           numPoints,
UINT4           position,
UINT4           subBankSize,
REAL4           a[LAL_NUM_IFO],
REAL4           b[LAL_NUM_IFO],
struct coh_PTF_params      *params,
struct bankCohTemplateOverlaps *cohBankOverlaps,
struct bankComplexTemplateOverlaps *bankOverlaps,
struct bankDataOverlaps *dataOverlaps,
struct bankTemplateOverlaps *bankNormOverlaps,
COMPLEX8VectorSequence  *PTFqVec[LAL_NUM_IFO+1],
REAL8Array      *PTFM[LAL_NUM_IFO+1],
INT4            timeOffsetPoints[LAL_NUM_IFO],
gsl_matrix *Bankeigenvecs[50],
gsl_vector *Bankeigenvals[50],
UINT4       detectorNum,
UINT4       vecLength,
UINT4       vecLengthTwo
)
{
  /* WARNING: THIS FUNCTION IS NON-SPIN ONLY. DO NOT ATTEMPT TO RUN WITH
     PTF TEMPLATES IN SPIN MODE! */
  UINT4 ui,uj,uk,ifoNumber,halfNumPoints;
  INT4 calTimeOffsetPoints[LAL_NUM_IFO];
  gsl_matrix *rotReOverlaps;
  gsl_matrix *rotImOverlaps;
  UINT4 snglDetMode = 1;
  if (detectorNum == LAL_NUM_IFO)
  {
    snglDetMode = 0;
  }

  REAL4 *SNRu1,*SNRu2;
  REAL4 BankVetoTemp[2*vecLengthTwo];
  REAL4 BankVeto=0;
  REAL4 normFac,overlapNorm,bankOverRe,bankOverIm;
  REAL4 *TjwithS1,*TjwithS2;
  SNRu1 = LALCalloc(vecLengthTwo,sizeof(REAL4));
  SNRu2 = LALCalloc(vecLengthTwo,sizeof(REAL4));
  TjwithS1 = LALCalloc(vecLengthTwo,sizeof(REAL4));
  TjwithS2 = LALCalloc(vecLengthTwo,sizeof(REAL4));

  /* Ensure that the time sliding doesn't push into data points that don't exist */
  for ( ifoNumber = 0; ifoNumber < LAL_NUM_IFO ; ifoNumber++ )
  {
    if ( params->haveTrig[ifoNumber] )
    {
      if ( position+timeOffsetPoints[ifoNumber] >= 3*numPoints/4 +5000)
      {
        calTimeOffsetPoints[ifoNumber] = 3*numPoints/4 +4999 - position;
        fprintf(stderr,"Overflow occured in time shifting in bank veto\n");
      }
      else if  ( position+timeOffsetPoints[ifoNumber] < numPoints/4 - 5000)
      {
        calTimeOffsetPoints[ifoNumber] = numPoints/4 - 4999 - position;
        fprintf(stderr,"Overflow occured in time shifting in bank veto\n");
      }
      else
        calTimeOffsetPoints[ifoNumber]=timeOffsetPoints[ifoNumber];
    }
  }

  /* Begin by calculating the components of the SNR */
  if (snglDetMode)
  {
    SNRu1[0] = PTFqVec[detectorNum]->data[position+timeOffsetPoints[detectorNum]].re;
    SNRu1[0] = SNRu1[0] / pow(PTFM[detectorNum]->data[0],0.5);
    SNRu2[0] = PTFqVec[detectorNum]->data[position+timeOffsetPoints[detectorNum]].im;
    SNRu2[0] = SNRu2[0] / pow(PTFM[detectorNum]->data[0],0.5);
  }
  else
  {
    coh_PTF_calculate_rotated_vectors(params,PTFqVec,SNRu1,SNRu2,a,b,
        timeOffsetPoints,Bankeigenvecs[subBankSize],Bankeigenvals[subBankSize],
        numPoints,position,vecLength,vecLengthTwo);
  }
  
  /* The normalization factors are already calculated, they are the eigenvalues*/
  halfNumPoints = 3*numPoints/4 - numPoints/4 + 10000;

  for ( ui = 0 ; ui < subBankSize ; ui++ )
  {
    if (snglDetMode)
    {
      TjwithS1[0] = dataOverlaps[ui].PTFqVec[detectorNum]->data[position \
          -numPoints/4+5000 + calTimeOffsetPoints[detectorNum]].re;
      TjwithS1[0] = TjwithS1[0] / \
          pow(bankNormOverlaps[ui].PTFM[detectorNum]->data[0],0.5);
      TjwithS2[0] = dataOverlaps[ui].PTFqVec[detectorNum]->data[position \
          -numPoints/4+5000 + calTimeOffsetPoints[detectorNum]].im;
      TjwithS2[0] = TjwithS2[0] / \
          pow(bankNormOverlaps[ui].PTFM[detectorNum]->data[0],0.5);
      overlapNorm = PTFM[detectorNum]->data[0];
      overlapNorm *= bankNormOverlaps[ui].PTFM[detectorNum]->data[0];
      overlapNorm = pow(overlapNorm,0.5);
      bankOverRe = bankOverlaps[ui].PTFM[detectorNum]->data[0].re;
      bankOverIm = bankOverlaps[ui].PTFM[detectorNum]->data[0].im;
      bankOverRe = bankOverRe/overlapNorm;
      bankOverIm = bankOverIm/overlapNorm;
      for (uj = 0; uj < 2 * vecLengthTwo; uj++)
      {
        normFac = 0;
        if (uj == 0)
          BankVetoTemp[uj] = TjwithS1[0];
        else
          BankVetoTemp[uj] = TjwithS2[0];
        for (uk = 0; uk < 2*vecLengthTwo; uk++)
        {
          if (uj == 0 && uk == 0)
          {
            BankVetoTemp[uj] -= bankOverRe*SNRu1[0];
            normFac += pow(bankOverRe,2);
          }
          if (uj == 0 && uk == vecLengthTwo )
          {
            BankVetoTemp[uj] -= bankOverIm*SNRu2[0];
            normFac += pow(bankOverIm,2);
          }
          if (uj == vecLengthTwo && uk == 0 )
          {
            BankVetoTemp[uj] += bankOverIm*SNRu1[0];
            normFac += pow(bankOverIm,2);
          }
          if (uj == vecLengthTwo && uk == vecLengthTwo )
          {
            BankVetoTemp[uj]-=bankOverRe*SNRu2[0];
            normFac += pow(bankOverRe,2);
          }
        }
        BankVeto+=pow(BankVetoTemp[uj],2)/(1-normFac);

      }
  
    }
    else
    {
      rotReOverlaps = cohBankOverlaps[ui].rotReOverlaps;
      rotImOverlaps = cohBankOverlaps[ui].rotImOverlaps;
      /* Calculate the components of subBank template with data */

      coh_PTF_calculate_rotated_vectors(params,dataOverlaps[ui].PTFqVec,
          TjwithS1,TjwithS2,a,b,calTimeOffsetPoints,Bankeigenvecs[ui],
          Bankeigenvals[ui],halfNumPoints,position-numPoints/4+5000,
          vecLength,vecLengthTwo);
      for (uj = 0; uj < 2*vecLengthTwo; uj++)
      {
        normFac = 0;
        if (uj < vecLengthTwo)
          BankVetoTemp[uj] = TjwithS1[uj];
        else
          BankVetoTemp[uj] = TjwithS2[uj-vecLengthTwo];
        for (uk = 0; uk < 2*vecLengthTwo; uk++)
        {
          if (uj < vecLengthTwo && uk < vecLengthTwo)
          {
            BankVetoTemp[uj] -= gsl_matrix_get(rotReOverlaps,uk,uj)*SNRu1[uk];
            normFac += pow(gsl_matrix_get(rotReOverlaps,uk,uj),2);
          }
          if (uj < vecLengthTwo && uk >= vecLengthTwo )
          {
            BankVetoTemp[uj] -= SNRu2[uk-vecLengthTwo] * gsl_matrix_get(
                rotImOverlaps, uk-vecLengthTwo,uj);
            normFac += pow(gsl_matrix_get(rotImOverlaps,uk-vecLengthTwo,uj),2);
          }
          if (uj >= vecLengthTwo && uk < vecLengthTwo )
          {
            BankVetoTemp[uj] += SNRu1[uk] * gsl_matrix_get(
                rotImOverlaps,uk,uj-vecLengthTwo);
            normFac += pow(gsl_matrix_get(rotImOverlaps,uk,uj-vecLengthTwo),2);
          }
	  if (uj >= vecLengthTwo && uk >= vecLengthTwo )
          {
            BankVetoTemp[uj]-= SNRu2[uk-vecLengthTwo] * gsl_matrix_get(
                rotReOverlaps,uk-vecLengthTwo,uj-vecLengthTwo);
            normFac += pow(gsl_matrix_get(rotReOverlaps,
                           uk-vecLengthTwo,uj-vecLengthTwo),2);
          }
        }
        BankVeto+=pow(BankVetoTemp[uj],2)/(1-normFac);
     
      }
    }
  }

  LALFree(SNRu1);
  LALFree(SNRu2);
  LALFree(TjwithS1);
  LALFree(TjwithS2);

  return BankVeto;
}
 
// FIXME: Consider merging function with bank_veto calculation??
REAL4 coh_PTF_calculate_auto_veto(
UINT4           numPoints,
UINT4           position,
REAL4           a[LAL_NUM_IFO],
REAL4           b[LAL_NUM_IFO],
struct coh_PTF_params      *params,
struct bankCohTemplateOverlaps *cohAutoOverlaps,
struct bankComplexTemplateOverlaps *autoTempOverlaps,
COMPLEX8VectorSequence  *PTFqVec[LAL_NUM_IFO+1],
REAL8Array      *PTFM[LAL_NUM_IFO+1],
INT4            timeOffsetPoints[LAL_NUM_IFO],
gsl_matrix *Autoeigenvecs,
gsl_vector *Autoeigenvals,
UINT4       detectorNum,
UINT4       vecLength,
UINT4       vecLengthTwo
)
{
  /* WARNING: THIS FUNCTION IS NON-SPIN ONLY. DO NOT ATTEMPT TO RUN WITH
     PTF TEMPLATES IN SPIN MODE! */
  UINT4 ui,uj,uk;
  gsl_matrix *rotReOverlaps;
  gsl_matrix *rotImOverlaps;
  UINT4 timeStepPoints = 0;
  timeStepPoints = params->autoVetoTimeStep*params->sampleRate;

  UINT4 snglDetMode = 1;
  if (detectorNum == LAL_NUM_IFO)
  {
    snglDetMode = 0;
  }

  REAL4 *SNRu1,*SNRu2;
  REAL4 AutoVetoTemp[2*vecLengthTwo];
  REAL4 AutoVeto=0;
  REAL4 normFac,autoOverRe,autoOverIm;
  REAL4 *TjwithS1,*TjwithS2;
  SNRu1 = LALCalloc(vecLengthTwo,sizeof(REAL4));
  SNRu2 = LALCalloc(vecLengthTwo,sizeof(REAL4));
  TjwithS1 = LALCalloc(vecLengthTwo,sizeof(REAL4));
  TjwithS2 = LALCalloc(vecLengthTwo,sizeof(REAL4));

  /* Begin by calculating the components of the SNR */
  if (snglDetMode)
  {
    SNRu1[0] = PTFqVec[detectorNum]->data[position+timeOffsetPoints[detectorNum]].re;
    SNRu1[0] = SNRu1[0] / pow(PTFM[detectorNum]->data[0],0.5);
    SNRu2[0] = PTFqVec[detectorNum]->data[position+timeOffsetPoints[detectorNum]].im;
    SNRu2[0] = SNRu2[0] / pow(PTFM[detectorNum]->data[0],0.5);
  }
  else
  {
    coh_PTF_calculate_rotated_vectors(params,PTFqVec,SNRu1,SNRu2,a,b,
        timeOffsetPoints,Autoeigenvecs,Autoeigenvals,
        numPoints,position,vecLength,vecLengthTwo);
  }

  for ( ui = 0 ; ui < params->numAutoPoints ; ui++ )
  {
    if (snglDetMode)
    {
      TjwithS1[0] = PTFqVec[detectorNum]->data[position \
          - ((ui+1) * timeStepPoints)+timeOffsetPoints[detectorNum]].re;
      TjwithS1[0] = TjwithS1[0] / pow(PTFM[detectorNum]->data[0],0.5);
      TjwithS2[0] = PTFqVec[detectorNum]->data[position \
          - ((ui+1) * timeStepPoints)+timeOffsetPoints[detectorNum]].im;
      TjwithS2[0] = TjwithS2[0] / pow(PTFM[detectorNum]->data[0],0.5);
      autoOverRe = autoTempOverlaps[ui].PTFM[detectorNum]->data[0].re;
      autoOverIm = autoTempOverlaps[ui].PTFM[detectorNum]->data[0].im;
      autoOverRe = autoOverRe / PTFM[detectorNum]->data[0];
      autoOverIm = autoOverIm / PTFM[detectorNum]->data[0];
      for (uj = 0; uj < 2*vecLengthTwo; uj++)
      {
        normFac = 0;
        if (uj == 0)
          AutoVetoTemp[uj] = TjwithS1[0];
        else
          AutoVetoTemp[uj] = TjwithS2[0];
        for (uk = 0; uk < 2*vecLengthTwo; uk++)
        {
          if (uj == 0 && uk == 0)
          {
            AutoVetoTemp[uj] -= autoOverRe*SNRu1[0];
            normFac += pow(autoOverRe,2);
          }
          if (uj == 0 && uk == 1 )
          {
            AutoVetoTemp[uj] += autoOverIm*SNRu2[0];
            normFac += pow(autoOverIm,2);
          }
          if (uj == 1 && uk == 0 )
          {
            AutoVetoTemp[uj] -= autoOverIm*SNRu1[0];
            normFac += pow(autoOverIm,2);
          }
          if (uj == 1 && uk == 1 )
          {
            AutoVetoTemp[uj] -= autoOverRe*SNRu2[0];
            normFac += pow(autoOverRe,2);
          }
        }
        AutoVeto+=pow(AutoVetoTemp[uj],2)/(1-normFac);

      }

    }
    else
    {
      rotReOverlaps = cohAutoOverlaps[ui].rotReOverlaps;
      rotImOverlaps = cohAutoOverlaps[ui].rotImOverlaps;
      /* Calculate the components of subBank template with data */

      coh_PTF_calculate_rotated_vectors(params,PTFqVec,TjwithS1,
          TjwithS2,a,b,timeOffsetPoints,Autoeigenvecs,
          Autoeigenvals,numPoints,position-((ui+1) * timeStepPoints),vecLength,
          vecLengthTwo);
      for (uj = 0; uj < 2*vecLengthTwo; uj++)
      {
        normFac = 0;
        if (uj < vecLengthTwo)
          AutoVetoTemp[uj] = TjwithS1[uj];
        else
          AutoVetoTemp[uj] = TjwithS2[uj-vecLengthTwo];
        for (uk = 0; uk < 2*vecLengthTwo; uk++)
        {
          if (uj < vecLengthTwo && uk < vecLengthTwo)
          {
            AutoVetoTemp[uj] -= gsl_matrix_get(rotReOverlaps,uk,uj)*SNRu1[uk];
            normFac += pow(gsl_matrix_get(rotReOverlaps,uk,uj),2);
          }
          if (uj < vecLengthTwo && uk >= vecLengthTwo )
          {
            AutoVetoTemp[uj] += SNRu2[uk-vecLengthTwo] * gsl_matrix_get(
                rotImOverlaps,uk-vecLengthTwo,uj);
            normFac += pow(gsl_matrix_get(rotImOverlaps,uk-vecLengthTwo,uj),2);
          }
          if (uj >= vecLengthTwo && uk < vecLengthTwo )
          {
            AutoVetoTemp[uj] -= SNRu1[uk] * gsl_matrix_get(
                rotImOverlaps,uk,uj-vecLengthTwo);
            normFac += pow(gsl_matrix_get(rotImOverlaps,uk,uj-vecLengthTwo),2);
          }
          if (uj >= vecLengthTwo && uk >= vecLengthTwo )
          {
            AutoVetoTemp[uj]-= SNRu2[uk-vecLengthTwo] * gsl_matrix_get(
                rotReOverlaps,uk-vecLengthTwo,uj-vecLengthTwo);
            normFac += pow(gsl_matrix_get(rotReOverlaps,
                           uk-vecLengthTwo,uj-vecLengthTwo),2);
          }
        }
        AutoVeto+=pow(AutoVetoTemp[uj],2)/(1-normFac);
      }
    }
  }

  LALFree(SNRu1);
  LALFree(SNRu2);
  LALFree(TjwithS1);
  LALFree(TjwithS2);

  return AutoVeto;
}

void coh_PTF_free_bank_veto_memory(
  struct bankTemplateOverlaps *bankNormOverlaps,
  InspiralTemplate        *PTFBankTemplates,
  FindChirpTemplate       *bankFcTmplts,
  UINT4 subBankSize,
  struct bankComplexTemplateOverlaps *bankOverlaps,
  struct bankDataOverlaps *dataOverlaps)
{
  UINT4 ui,ifoNumber;
 
  if ( bankNormOverlaps )
  {
    for ( ui = 0 ; ui < subBankSize ; ui++ )
    {
      for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
      {
        if ( bankNormOverlaps[ui].PTFM[ifoNumber] )
        {
          XLALDestroyREAL8Array(bankNormOverlaps[ui].PTFM[ifoNumber]);
        }
      }
    }
    LALFree(bankNormOverlaps);
  }

  if ( PTFBankTemplates )
    LALFree( PTFBankTemplates);

  if ( bankFcTmplts )
  {
    for ( ui = 0 ; ui < subBankSize ; ui++ )
    {
      if ( bankFcTmplts[ui].PTFQtilde )
        XLALDestroyCOMPLEX8VectorSequence( bankFcTmplts[ui].PTFQtilde );
    }
    LALFree( bankFcTmplts );
  } 

  if (dataOverlaps)
    LALFree(dataOverlaps);
  if (bankOverlaps)
    LALFree(bankOverlaps);
}


void coh_PTF_calculate_coherent_bank_overlaps(
  struct coh_PTF_params   *params,
  struct bankComplexTemplateOverlaps bankOverlaps,
  struct bankCohTemplateOverlaps cohBankOverlaps,
  REAL4           a[LAL_NUM_IFO],
  REAL4           b[LAL_NUM_IFO],
  gsl_matrix *eigenvecs,
  gsl_vector *eigenvals,
  gsl_matrix *Bankeigenvecs,
  gsl_vector *Bankeigenvals,
  UINT4 vecLength,
  UINT4 vecLengthTwo
)
{
  /* THIS FUNCTION IS NON-SPIN ONLY! DO NOT TRY TO RUN WITH PTF IN SPIN MODE */

  UINT4 uk,uj,ul;
  gsl_matrix *rotReOverlaps = cohBankOverlaps.rotReOverlaps;
  gsl_matrix *rotImOverlaps = cohBankOverlaps.rotImOverlaps;  

  gsl_matrix *reOverlaps = gsl_matrix_alloc(vecLengthTwo,vecLengthTwo);
  gsl_matrix *imOverlaps = gsl_matrix_alloc(vecLengthTwo,vecLengthTwo);
  gsl_matrix *tempM = gsl_matrix_alloc(vecLengthTwo,vecLengthTwo);
  REAL4 reOverlapsA[vecLengthTwo*vecLengthTwo];
  REAL4 imOverlapsA[vecLengthTwo*vecLengthTwo];
  REAL4 fone,ftwo;

  /* First step would be to create a matrix of overlaps in non rotated frame*/
  for (uj = 0; uj < vecLengthTwo; uj++)
  {
    for (uk = 0 ; uk < vecLengthTwo; uk++)
    {
      reOverlapsA[uj*vecLengthTwo + uk] = 0;
      imOverlapsA[uj*vecLengthTwo + uk] = 0;
    }
  }

  for (ul = 0; ul < LAL_NUM_IFO; ul++)
  {
    if ( params->haveTrig[ul] )
    {
      for (uj = 0; uj < vecLengthTwo; uj++)
      {
        if (uj < vecLength)
          fone = a[ul];
        else
          fone = b[ul];
        for (uk = 0 ; uk < vecLengthTwo; uk++)
        {
          if (uk < vecLength)
            ftwo = a[ul];
          else
            ftwo = b[ul];
          reOverlapsA[uj*vecLengthTwo + uk] +=
              fone * ftwo *  bankOverlaps.PTFM[ul]->data[0].re;
          imOverlapsA[uj*vecLengthTwo + uk] +=
              fone * ftwo *  bankOverlaps.PTFM[ul]->data[0].im;
        }
      }
    }
  }
  for (uj = 0; uj < vecLengthTwo; uj++)
  {
    for (uk = 0; uk < vecLengthTwo; uk++)
    {
      gsl_matrix_set(reOverlaps,uj,uk,reOverlapsA[uj*vecLengthTwo+uk]);
      gsl_matrix_set(imOverlaps,uj,uk,imOverlapsA[uj*vecLengthTwo+uk]);
    }
  }
 
  /* And rotate by both set of eigenvectors */

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1,reOverlaps,eigenvecs,0.,tempM);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1,Bankeigenvecs,tempM,0.,rotReOverlaps);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1,imOverlaps,eigenvecs,0.,tempM);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1,Bankeigenvecs,tempM,0.,rotImOverlaps);

  for (uj=0; uj<vecLengthTwo; uj++)
  {
    for (uk=0; uk<vecLengthTwo; uk++ )
    {
      gsl_matrix_set(rotReOverlaps,uj,uk,gsl_matrix_get(rotReOverlaps,uj,uk)/(sqrt(gsl_vector_get(eigenvals,uj))*sqrt(gsl_vector_get(Bankeigenvals,uk))));
      gsl_matrix_set(rotImOverlaps,uj,uk,gsl_matrix_get(rotImOverlaps,uj,uk)/(sqrt(gsl_vector_get(eigenvals,uj))*sqrt(gsl_vector_get(Bankeigenvals,uk))));
    }
  }

  gsl_matrix_free(reOverlaps);
  gsl_matrix_free(imOverlaps);
  gsl_matrix_free(tempM);

}

void coh_PTF_calculate_standard_chisq_freq_ranges(
    struct coh_PTF_params   *params,
    FindChirpTemplate       *fcTmplt,
    REAL4FrequencySeries    *invspec[LAL_NUM_IFO+1],
    REAL8Array              *PTFM[LAL_NUM_IFO+1],
    REAL4 a[LAL_NUM_IFO],
    REAL4 b[LAL_NUM_IFO],
    REAL4 *frequencyRangesPlus,
    REAL4 *frequencyRangesCross,
    gsl_matrix *eigenvecs,
    UINT4 detectorNum,
    UINT4 singlePolFlag
)
{
  /* THIS FUNCTION IS NON-SPIN ONLY! DO NOT TRY TO RUN WITH PTF IN SPIN MODE */
  UINT4 i,k,kmin,kmax,len,freqBinPlus,freqBinCross,numFreqBins;
  REAL4 v1,v2,v3,u1,u2,overlapCont,SNRtempPlus,SNRtempCross,SNRmaxPlus,SNRmaxCross;
  REAL8         f_min, deltaF, fFinal;
  COMPLEX8     *PTFQtilde   = NULL;
  REAL4 a2[LAL_NUM_IFO];
  REAL4 b2[LAL_NUM_IFO];

  PTFQtilde = fcTmplt->PTFQtilde->data;  
  len = 0;
  deltaF = 0;
  for ( k = 0; k < LAL_NUM_IFO; k++)
  {
    if ( params->haveTrig[k] )
    {
      len       = invspec[k]->data->length;
      deltaF    = invspec[k]->deltaF;
      break;
    }
  }
  /* This is explicit as I want f_min of template lower than f_min of filter*/
  /* Note that these frequencies are not just hardcoded here, if you change*/
  /* these values you will need to change them in other places as well */
  f_min     = params->lowFilterFrequency;
  kmin      = f_min / deltaF > 1 ?  f_min / deltaF : 1;
  fFinal    = params->highFilterFrequency;
  kmax      = fFinal / deltaF < (len - 1) ? fFinal / deltaF : (len - 1);

  numFreqBins = params->numChiSquareBins;

  v1 = 0;
  v2 = 0;
  v3 = 0;
  if (detectorNum == LAL_NUM_IFO)
  {
    /* If only one polarization the value of a2 and b2 is irrelevant! */
    if ( singlePolFlag )
    {
      for( k = 0; k < LAL_NUM_IFO; k++)
      {
        if ( params->haveTrig[k] )
        {
          a2[k] = 1;
          b2[k] = 1;
        }
      }
    }
    else
    {
      for( k = 0; k < LAL_NUM_IFO; k++)
      {
        if ( params->haveTrig[k] )
        {
          a2[k] = a[k]*gsl_matrix_get(eigenvecs,0,0) + b[k]*gsl_matrix_get(eigenvecs,1,0);
          b2[k] = a[k]*gsl_matrix_get(eigenvecs,0,1) + b[k]*gsl_matrix_get(eigenvecs,1,1);
        }
      }
    }
    for( k = 0; k < LAL_NUM_IFO; k++)
    {
      if ( params->haveTrig[k] )
      {
        v1 += a2[k]*a2[k]*PTFM[k]->data[0];
        v2 += b2[k]*b2[k]*PTFM[k]->data[0];
        v3 += a2[k]*b2[k]*PTFM[k]->data[0];
      }
    }
  }
  else
  {
    v1 = PTFM[detectorNum]->data[0];
    v2 = PTFM[detectorNum]->data[0];
  }

  u1 = v1;
  u2 = v2;
  SNRmaxPlus = u1;
  if (SNRmaxPlus < 0) SNRmaxPlus = -SNRmaxPlus;
  SNRmaxCross = u2;
  if (SNRmaxCross < 0) SNRmaxCross = -SNRmaxCross;

  v1 = 0;
  v2 = 0;

  freqBinPlus = 1;
  freqBinCross = 1;
  SNRtempPlus = 0;
  SNRtempCross = 0;
  for ( i = kmin; i < kmax ; ++i )
  {
    if (detectorNum == LAL_NUM_IFO)
    {
      for( k = 0; k < LAL_NUM_IFO; k++)
      {
        if ( params->haveTrig[k] )
        {
          overlapCont = (PTFQtilde[i].re * PTFQtilde[i].re +
                 PTFQtilde[i].im * PTFQtilde[i].im )* invspec[k]->data->data[i] ;
          v1 += a2[k] * a2[k] * overlapCont * 4 * deltaF;
          v2 += b2[k] * b2[k] * overlapCont * 4 * deltaF;
        }
      }
    }
    else
    {
      overlapCont = ( PTFQtilde[i].re * PTFQtilde[i].re +
                      PTFQtilde[i].im * PTFQtilde[i].im ) * 
                      invspec[detectorNum]->data->data[i] ;
      v1 += overlapCont * 4 * deltaF;
      v2 = v1;
    }
    /* Calculate SNR */
    u1 = v1;
    u2 = v2;
    SNRtempPlus = u1;
    if (SNRtempPlus < 0) SNRtempPlus = -SNRtempPlus;
    SNRtempCross = u2;
    if (SNRtempCross < 0) SNRtempCross = -SNRtempCross;
    /* Compare to max SNR */
    if (SNRtempPlus > SNRmaxPlus * ((REAL4)freqBinPlus/(REAL4)numFreqBins))
    {
      if (freqBinPlus < numFreqBins)
      {
        /* Record the frequency */
        frequencyRangesPlus[freqBinPlus-1] = i*deltaF;
        freqBinPlus+=1;
      }
    }
    if (SNRtempCross > SNRmaxCross * ((REAL4)freqBinCross/(REAL4)numFreqBins))
    {
      if (freqBinCross < numFreqBins)
      {
        /* Record the frequency */
        frequencyRangesCross[freqBinCross-1] = i*deltaF;
        freqBinCross+=1;
      }
    }
  }
}

void coh_PTF_calculate_standard_chisq_power_bins(
    struct coh_PTF_params   *params,
    FindChirpTemplate       *fcTmplt,
    REAL4FrequencySeries    *invspec[LAL_NUM_IFO+1],
    REAL8Array              *PTFM[LAL_NUM_IFO+1],
    REAL4 a[LAL_NUM_IFO],
    REAL4 b[LAL_NUM_IFO],
    REAL4 *frequencyRangesPlus,
    REAL4 *frequencyRangesCross,
    REAL4 *powerBinsPlus,
    REAL4 *powerBinsCross,
    gsl_matrix *eigenvecs,
    UINT4 detectorNum,
    UINT4 singlePolFlag
)
{
  /* THIS FUNCTION IS NON-SPIN ONLY! DO NOT TRY TO RUN WITH PTF IN SPIN MODE */
  UINT4 i,k,kmin,kmax,len,freqBinPlus,freqBinCross,numFreqBins;
  REAL4 v1,v2,v3,overlapCont,SNRtempPlus,SNRtempCross,SNRmaxPlus,SNRmaxCross;
  REAL4 SNRplusLast,SNRcrossLast;
  REAL8         f_min, deltaF, fFinal;
  COMPLEX8     *PTFQtilde   = NULL;
  REAL4 a2[LAL_NUM_IFO];
  REAL4 b2[LAL_NUM_IFO];

  PTFQtilde = fcTmplt->PTFQtilde->data;
  len = 0;
  deltaF = 0;
  for ( k = 0; k < LAL_NUM_IFO; k++)
  {
    if ( params->haveTrig[k] )
    {
      len       = invspec[k]->data->length;
      deltaF    = invspec[k]->deltaF;
      break;
    }
  }
  /* This is explicit as I want f_min of template lower than f_min of filter*/
  /* Note that these frequencies are not just hardcoded here, if you change*/
  /* these values you will need to change them in other places as well */
  f_min     = params->lowFilterFrequency;
  kmin      = f_min / deltaF > 1 ?  f_min / deltaF : 1;
  fFinal    = params->highFilterFrequency;
  kmax      = fFinal / deltaF < (len - 1) ? fFinal / deltaF : (len - 1);

  numFreqBins = params->numChiSquareBins;

  // NOTE: v3 is calculated for verification. It should = 0.
  v1 = 0;
  v2 = 0;
  v3 = 0;
  if (detectorNum == LAL_NUM_IFO)
  {
    /* If only one polarization the value of a2 and b2 is irrelevant! */
    if ( singlePolFlag )
    {
      for( k = 0; k < LAL_NUM_IFO; k++)
      {
        if ( params->haveTrig[k] )
        { 
          a2[k] = 1;
          b2[k] = 1;
        }
      }
    }
    else
    {
      for( k = 0; k < LAL_NUM_IFO; k++)
      {         if ( params->haveTrig[k] )
        {       
          a2[k] = a[k]*gsl_matrix_get(eigenvecs,0,0) + b[k]*gsl_matrix_get(eigenvecs,1,0);
          b2[k] = a[k]*gsl_matrix_get(eigenvecs,0,1) + b[k]*gsl_matrix_get(eigenvecs,1,1);
        }
      } 
    }
    for( k = 0; k < LAL_NUM_IFO; k++)
    {
      if ( params->haveTrig[k] )
      {
        v1 += a2[k]*a2[k]*PTFM[k]->data[0];
        v2 += b2[k]*b2[k]*PTFM[k]->data[0];
        v3 += a2[k]*b2[k]*PTFM[k]->data[0];
      }
    }
  }
  else
  {
    v1 = PTFM[detectorNum]->data[0];
    v2 = PTFM[detectorNum]->data[0];
  }

  SNRmaxPlus = v1;
  SNRmaxCross = v2;
  if (SNRmaxPlus < 0) SNRmaxPlus = -SNRmaxPlus;
  if (SNRmaxCross < 0) SNRmaxCross = -SNRmaxCross;

  v1 = 0;
  v2 = 0;

  freqBinPlus = 0;
  freqBinCross = 0;
  SNRtempPlus = 0;
  SNRtempCross = 0;
  SNRplusLast = 0.;
  SNRcrossLast = 0.;

  for ( i = kmin; i < kmax ; ++i )
  {
    if (detectorNum == LAL_NUM_IFO)
    { 
      for( k = 0; k < LAL_NUM_IFO; k++)
      {
        if ( params->haveTrig[k] )
        {
          overlapCont = (PTFQtilde[i].re * PTFQtilde[i].re +
                 PTFQtilde[i].im * PTFQtilde[i].im )* invspec[k]->data->data[i] ;
          v1 += a2[k] * a2[k] * overlapCont * 4 * deltaF;
          v2 += b2[k] * b2[k] * overlapCont * 4 * deltaF;
        }
      }
    }
    else
    {
      overlapCont = (PTFQtilde[i].re * PTFQtilde[i].re +
             PTFQtilde[i].im * PTFQtilde[i].im )* invspec[detectorNum]->data->data[i] ;
      v1 += overlapCont * 4 * deltaF;
      v2 = v1;
    }
    SNRtempPlus = v1;
    if (SNRtempPlus < 0) SNRtempPlus = -SNRtempPlus;
    SNRtempCross = v2;
    if (SNRtempCross < 0) SNRtempCross = -SNRtempCross;

    if (i * deltaF > frequencyRangesPlus[freqBinPlus] && freqBinPlus < (numFreqBins-1))
    {
      powerBinsPlus[freqBinPlus] = SNRtempPlus/SNRmaxPlus - SNRplusLast;
      SNRplusLast = SNRtempPlus/SNRmaxPlus;
      freqBinPlus++;
    }
    if (i * deltaF > frequencyRangesCross[freqBinCross] && freqBinCross < (numFreqBins-1))
    {
      powerBinsCross[freqBinCross] = SNRtempCross/SNRmaxCross - SNRcrossLast;
      SNRcrossLast = SNRtempCross/SNRmaxCross;
      freqBinCross++;
    }

  }
  if (freqBinPlus == (numFreqBins-1))  
  {
    powerBinsPlus[freqBinPlus] = SNRtempPlus/SNRmaxPlus - SNRplusLast;
  }
  if (freqBinCross == (numFreqBins-1))
  {
    powerBinsCross[freqBinCross] = SNRtempCross/SNRmaxCross - SNRcrossLast;
  }   

  /* Ensure that the power Bins add to 1. This should already be true but
  numerical counting errors can have a small effect here.*/
  SNRplusLast = 0;
  SNRcrossLast = 0;
  for ( i = 0 ; i < (numFreqBins); i++)
  {
    SNRplusLast += powerBinsPlus[i];
    SNRcrossLast += powerBinsCross[i];
  }
  for ( i = 0 ; i < (numFreqBins); i++)
  {
    powerBinsPlus[i] = powerBinsPlus[i]/SNRplusLast;
    powerBinsCross[i] = powerBinsCross[i]/SNRcrossLast;
  }
}

REAL4 coh_PTF_calculate_chi_square(
struct coh_PTF_params   *params,
UINT4           numPoints,
UINT4           position,
struct bankDataOverlaps *chisqOverlaps,    
COMPLEX8VectorSequence  *PTFqVec[LAL_NUM_IFO+1],
REAL8Array      *PTFM[LAL_NUM_IFO+1],
REAL4           a[LAL_NUM_IFO],
REAL4           b[LAL_NUM_IFO],
INT4            timeOffsetPoints[LAL_NUM_IFO],
gsl_matrix *eigenvecs,
gsl_vector *eigenvals,   
REAL4 *powerBinsPlus,
REAL4 *powerBinsCross,
UINT4 detectorNum,
UINT4 vecLength,
UINT4 vecLengthTwo
)
{
  /* THIS FUNCTION IS NON-SPIN ONLY! DO NOT TRY TO RUN WITH PTF IN SPIN MODE */
  UINT4 i,halfNumPoints;
  REAL4 *v1Plus,*v2Plus,*v1full,*v2full;
  REAL4 *v1Cross,*v2Cross;
  REAL4 chiSq,SNRtemp,SNRexp;
  UINT4 numChiSquareBins = params->numChiSquareBins;

  v1Plus = LALCalloc(vecLengthTwo,sizeof(REAL4));
  v2Plus = LALCalloc(vecLengthTwo,sizeof(REAL4));
  v1Cross = LALCalloc(vecLengthTwo,sizeof(REAL4));
  v2Cross = LALCalloc(vecLengthTwo,sizeof(REAL4));

  v1full = LALCalloc(vecLengthTwo,sizeof(REAL4));
  v2full = LALCalloc(vecLengthTwo,sizeof(REAL4));

  halfNumPoints = 3*numPoints/4 - numPoints/4 + 10000;

  chiSq = 0;
  SNRexp = 0;

  if (detectorNum == LAL_NUM_IFO)
  {
    coh_PTF_calculate_rotated_vectors(params,PTFqVec,v1full,v2full,a,b,
          timeOffsetPoints,eigenvecs,eigenvals,numPoints,
          position,vecLength,vecLengthTwo);
  }
  else
  {
    v1full[0] = PTFqVec[detectorNum]->data[position+timeOffsetPoints[detectorNum]].re;
    v1full[0] = v1full[0]/pow(PTFM[detectorNum]->data[0],0.5);
    v2full[0] = PTFqVec[detectorNum]->data[position+timeOffsetPoints[detectorNum]].im;
    v2full[0] = v2full[0]/pow(PTFM[detectorNum]->data[0],0.5);
  }

  for (i = 0; i < vecLengthTwo; i++)
  {
    SNRexp += v1full[i]*v1full[i];
    SNRexp += v2full[i]*v2full[i];
  } 
  SNRexp = pow(SNRexp,0.5);

  for (i = 0; i < numChiSquareBins; i++ )
  {
    if (detectorNum == LAL_NUM_IFO)
    {
      /* calculate SNR in this frequency bin */
      coh_PTF_calculate_rotated_vectors(params,chisqOverlaps[i].PTFqVec,v1Plus,
          v2Plus,a,b,timeOffsetPoints,eigenvecs,eigenvals,halfNumPoints,
          position-numPoints/4+5000,vecLength,vecLengthTwo);

      coh_PTF_calculate_rotated_vectors(params,
          chisqOverlaps[i+numChiSquareBins].PTFqVec,v1Cross,
          v2Cross,a,b,timeOffsetPoints,eigenvecs,eigenvals,halfNumPoints,
          position-numPoints/4+5000,vecLength,vecLengthTwo);

      SNRtemp= pow((v1Plus[0] - v1full[0]*powerBinsPlus[i]),2)/powerBinsPlus[i];
      SNRtemp+= pow((v2Plus[0] - v2full[0]*powerBinsPlus[i]),2)/powerBinsPlus[i];
      if (vecLengthTwo == 2)
      {
        SNRtemp+= pow((v1Cross[1]-v1full[1]*powerBinsCross[i]),2)/powerBinsCross[i];
        SNRtemp+= pow((v2Cross[1]-v2full[1]*powerBinsCross[i]),2)/powerBinsCross[i];
      }
      chiSq += SNRtemp;
    }
    else
    {
      v1Plus[0] = chisqOverlaps[i].PTFqVec[detectorNum]->data[position-numPoints/4+5000+timeOffsetPoints[detectorNum]].re;
      v1Plus[0] = v1Plus[0]/pow(PTFM[detectorNum]->data[0],0.5);
      v2Plus[0] = chisqOverlaps[i].PTFqVec[detectorNum]->data[position-numPoints/4+5000+timeOffsetPoints[detectorNum]].im;
      v2Plus[0] = v2Plus[0]/pow(PTFM[detectorNum]->data[0],0.5);
      SNRtemp = pow((v1Plus[0] - v1full[0]*powerBinsPlus[i]),2)/powerBinsPlus[i];
      SNRtemp += pow((v2Plus[0] - v2full[0]*powerBinsCross[i]),2)/powerBinsCross[i];
      chiSq += SNRtemp;
    }
  }
  LALFree(v1Plus);
  LALFree(v2Plus);
  LALFree(v1Cross);
  LALFree(v2Cross);
  LALFree(v1full);
  LALFree(v2full);
  return chiSq;
}

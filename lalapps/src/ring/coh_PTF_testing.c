#include "coh_PTF.h"

RCSID( "$Id$" );
#define PROGRAM_NAME "lalapps_coh_PTF_inspiral"
#define CVS_REVISION "$Revision$"
#define CVS_SOURCE   "$Source$"
#define CVS_DATE     "$Date$"

/* This function should be migrated to option.c */
/* warning: returns a pointer to a static variable... not reenterant */
/* only call this routine once to initialize params! */
/* also do not attempt to free this pointer! */
static struct coh_PTF_params *coh_PTF_get_params( int argc, char **argv )
{
  static struct coh_PTF_params params;
  static char programName[] = PROGRAM_NAME;
  static char cvsRevision[] = CVS_REVISION;
  static char cvsSource[]   = CVS_SOURCE;
  static char cvsDate[]     = CVS_DATE;
  coh_PTF_parse_options( &params, argc, argv );
  coh_PTF_params_sanity_check( &params ); /* this also sets various params */
  coh_PTF_params_inspiral_sanity_check( &params );
  params.programName = programName;
  params.cvsRevision = cvsRevision;
  params.cvsSource   = cvsSource;
  params.cvsDate     = cvsDate;
  return &params;
}

int main( int argc, char **argv )
{
// Declarations of parameters
  INT4 i,j,k;
  UINT4 ui,uj;
  struct coh_PTF_params      *params    = NULL;
  ProcessParamsTable      *procpar   = NULL;
  REAL4FFTPlan            *fwdplan   = NULL;
  REAL4FFTPlan            *revplan   = NULL;
  COMPLEX8FFTPlan         *invPlan   = NULL;
  REAL4TimeSeries         *channel[LAL_NUM_IFO+1];
  REAL4FrequencySeries    *invspec[LAL_NUM_IFO+1];
  RingDataSegments        *segments[LAL_NUM_IFO+1];
  INT4                    numTmplts = 0;
  INT4                    numSpinTmplts = 0;
  INT4                    numNoSpinTmplts = 0;
  INT4  startTemplate     = -1;           /* index of first template      */
  INT4  stopTemplate      = -1;           /* index of last template       */
  INT4 numSegments        = 0;
  InspiralTemplate        *PTFSpinTemplate = NULL;
  InspiralTemplate        *PTFNoSpinTemplate = NULL;
  InspiralTemplate        *PTFtemplate = NULL;
  InspiralTemplate        *PTFbankhead = NULL;
  FindChirpTemplate       *fcTmplt     = NULL;
  InspiralTemplate        *PTFBankTemplates = NULL;
  InspiralTemplate        *PTFBankvetoHead = NULL;
  FindChirpTemplate       *bankFcTmplts = NULL;
  FindChirpTmpltParams    *fcTmpltParams      = NULL;
  FindChirpInitParams     *fcInitParams = NULL;
  UINT4                   numPoints,ifoNumber,spinTemplate;
  REAL8Array              *PTFM[LAL_NUM_IFO+1];
  REAL8Array              *PTFN[LAL_NUM_IFO+1];
  COMPLEX8VectorSequence  *PTFqVec[LAL_NUM_IFO+1];
  time_t                  startTime;
  LALDetector             *detectors[LAL_NUM_IFO+1];
  REAL8                   *timeOffsets;
  REAL8                   *Fplus;
  REAL8                   *Fcross;
  REAL8                   detLoc[3];
  REAL4TimeSeries         *cohSNR = NULL;
  REAL4TimeSeries         *pValues[10];
  REAL4TimeSeries         *snrComps[LAL_NUM_IFO];
  REAL4TimeSeries         *gammaBeta[2];
  REAL4TimeSeries         *nullSNR = NULL;
  REAL4TimeSeries         *traceSNR = NULL;
  REAL4TimeSeries         *bankVeto = NULL;
  REAL4TimeSeries         *autoVeto = NULL;
  REAL4TimeSeries         *chiSquare = NULL;
  LIGOTimeGPS             segStartTime;
  MultiInspiralTable      *eventList = NULL;
  MultiInspiralTable      *thisEvent = NULL;
  UINT8                   eventId = 0;
  UINT4                   numDetectors = 0;
  UINT4                   singleDetector = 0;
  UINT4                   UNUSED spinBank = 0;
  char                    spinFileName[256];
  char                    noSpinFileName[256];
  
  startTime = time(NULL);

  /* set error handlers to abort on error */
  set_abrt_on_error();

  /* options are parsed and debug level is set here... */
  

  /* no lal mallocs before this! */
  params = coh_PTF_get_params( argc, argv );

  /* create process params */
  procpar = create_process_params( argc, argv, PROGRAM_NAME );

  verbose("Read input params %ld \n", time(NULL)-startTime);

  /* create forward and reverse fft plans */
  fwdplan = coh_PTF_get_fft_fwdplan( params );
  revplan = coh_PTF_get_fft_revplan( params );

  verbose("Made fft plans %ld \n", time(NULL)-startTime);

  /* Determine if we are analyzing single or multiple ifo data */

  for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
  {
    if ( params->haveTrig[ifoNumber] )
    {
      numDetectors++;
    }
  }
  /* NULL out pointers where necessary */
  for ( i = 0 ; i < 10 ; i++ )
  {
    pValues[i] = NULL;
  }   
  for ( i = 0 ; i < LAL_NUM_IFO ; i++ )
  {
    snrComps[i] = NULL;
  }
  gammaBeta[0] = NULL;
  gammaBeta[1] = NULL;

  /* Initialise some of the input file names */
  if ( params->spinBank )
  {
    spinBank = 1;
    strncpy(spinFileName,params->spinBank,sizeof(spinFileName)-1);
  }
  if ( params->noSpinBank )
    strncpy(noSpinFileName,params->noSpinBank,sizeof(noSpinFileName)-1);

  if (numDetectors == 0 )
  {
    fprintf(stderr,"You have not specified any detectors to analyse");
    return 1;
  }
  else if (numDetectors == 1 )
  {
    fprintf(stdout,"You have only specified one detector, why are you using the coherent code? \n");
    singleDetector = 1;
  }

  /* In this loop we read the data, generate segments and the PSD */
  for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
  {
    /* Initialize some of the structures */
    channel[ifoNumber] = NULL;
    invspec[ifoNumber] = NULL;
    segments[ifoNumber] = NULL;
    PTFM[ifoNumber] = NULL;
    PTFN[ifoNumber] = NULL;
    PTFqVec[ifoNumber] = NULL;
    if ( params->haveTrig[ifoNumber] )
    {
      /* Read in data from the various ifos */
      params->doubleData = 1;
      if ( params->simData )
          params->doubleData = 0;
      else if ( ifoNumber == LAL_IFO_V1 )
          params->doubleData = 0;
      channel[ifoNumber] = coh_PTF_get_data(params,params->channel[ifoNumber],\
                               params->dataCache[ifoNumber],ifoNumber );
      coh_PTF_rescale_data (channel[ifoNumber],1E20);

      /* compute the spectrum */
      invspec[ifoNumber] = coh_PTF_get_invspec( channel[ifoNumber], fwdplan,\
                               revplan, params );

      /* create the segments */
      segments[ifoNumber] = coh_PTF_get_segments( channel[ifoNumber],\
           invspec[ifoNumber], fwdplan, params );
      
      numSegments = segments[ifoNumber]->numSgmnt;

      verbose("Created segments for one ifo %ld \n", time(NULL)-startTime);
    }
  }

  /* Determine time delays and response functions */
  /* This is computed for all detectors, even if not being analyzed */ 
  timeOffsets = LALCalloc(1, LAL_NUM_IFO*sizeof( REAL8 ));
  Fplus = LALCalloc(1, LAL_NUM_IFO*sizeof( REAL8 ));
  Fcross = LALCalloc(1, LAL_NUM_IFO*sizeof( REAL8 ));
  for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
  {
    detectors[ifoNumber] = LALCalloc( 1, sizeof( *detectors[ifoNumber] ));
    XLALReturnDetector(detectors[ifoNumber] ,ifoNumber);
    for ( i = 0; i < 3; i++ )
    {
      detLoc[i] = (double) detectors[ifoNumber]->location[i];
    }
    for ( j = 0; j < numSegments; ++j )
    {
      /* Despite being called segStartTime we use the time at the middle 
      * of a segment */
      segStartTime = params->trigTime;
      
      /*XLALGPSAdd(&segStartTime,(j+1)*params->segmentDuration/2.0);*/
//      XLALGPSAdd(&segStartTime,8.5*params->segmentDuration/2.0);
      /*XLALGPSMultiply(&segStartTime,0.);
      XLALGPSAdd(&segStartTime,874610713.072549154);*/
      timeOffsets[ifoNumber] = 
          XLALTimeDelayFromEarthCenter(detLoc,params->rightAscension,
          params->declination,&segStartTime);
      XLALComputeDetAMResponse(&Fplus[ifoNumber],
         &Fcross[ifoNumber],
         detectors[ifoNumber]->response,params->rightAscension,
         params->declination,0.,XLALGreenwichMeanSiderealTime(&segStartTime));
    }
    LALFree(detectors[ifoNumber]);
  }
  

  numPoints = floor( params->segmentDuration * params->sampleRate + 0.5 );

  /* Initialize some of the structures */
  ifoNumber = LAL_NUM_IFO;
  channel[ifoNumber] = NULL;
  invspec[ifoNumber] = NULL;
  segments[ifoNumber] = NULL;
  PTFM[ifoNumber] = NULL;
  PTFN[ifoNumber] = NULL;
  PTFqVec[ifoNumber] = NULL;


  /* Construct the null stream, its segments and its PSD */
  if ( params->doNullStream )
  {
    /* Read in data from the various ifos */
    if (coh_PTF_get_null_stream(params,channel,Fplus,Fcross,timeOffsets ))
    {
      fprintf(stderr,"Null stream construction failure\n");
      return 1;
    }

    /* compute the spectrum */
    invspec[ifoNumber] = coh_PTF_get_invspec( channel[ifoNumber], fwdplan,\
                             revplan, params );
    /* If white spectrum need to scale this. FIX ME!!! */
    if (params->whiteSpectrum)
    {
      for( ui=0 ; ui < invspec[ifoNumber]->data->length; ui++)
      {
        invspec[ifoNumber]->data->data[ui] *= pow(1./0.3403324,2);
      }
    }

    /* create the segments */
    segments[ifoNumber] = coh_PTF_get_segments( channel[ifoNumber],\
         invspec[ifoNumber], fwdplan, params );

    numSegments = segments[ifoNumber]->numSgmnt;

    verbose("Created segments for null stream at %ld \n", time(NULL)-startTime);
    PTFM[ifoNumber] = XLALCreateREAL8ArrayL( 2, 5, 5 );
    PTFN[ifoNumber] = XLALCreateREAL8ArrayL( 2, 5, 5 );
    PTFqVec[ifoNumber] = XLALCreateCOMPLEX8VectorSequence ( 5, numPoints );
  }

  /* At this point we can discard the calibrated data, only the segments
     and spectrum is needed now */

  for( ifoNumber = 0; ifoNumber < (LAL_NUM_IFO+1); ifoNumber++)
  {
    if ( channel[ifoNumber] )
    {
      XLALDestroyREAL4Vector( channel[ifoNumber]->data );
      LALFree( channel[ifoNumber] );
      channel[ifoNumber] = NULL;
    }
  }

  /* Create the relevant structures that will be needed */
  fcInitParams = LALCalloc( 1, sizeof( *fcInitParams ));
  fcTmplt = LALCalloc( 1, sizeof( *fcTmplt ) );
  fcTmpltParams = LALCalloc ( 1, sizeof( *fcTmpltParams ) );
  fcTmpltParams->approximant = FindChirpPTF;
  /* Note that although non-spinning only uses Q1, the PTF
  generator still generates Q1-5, thus size of these vectors */
  fcTmplt->PTFQtilde =
      XLALCreateCOMPLEX8VectorSequence( 5, numPoints / 2 + 1 );
  fcTmpltParams->PTFQ = XLALCreateVectorSequence( 5, numPoints );
  fcTmpltParams->PTFphi = XLALCreateVector( numPoints );
  fcTmpltParams->PTFomega_2_3 = XLALCreateVector( numPoints );
  fcTmpltParams->PTFe1 = XLALCreateVectorSequence( 3, numPoints );
  fcTmpltParams->PTFe2 = XLALCreateVectorSequence( 3, numPoints );
  fcTmpltParams->fwdPlan =
        XLALCreateForwardREAL4FFTPlan( numPoints, 0 );
  fcTmpltParams->deltaT = 1.0/params->sampleRate;
  for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
  {
    if ( params->haveTrig[ifoNumber] )
    {
      PTFM[ifoNumber] = XLALCreateREAL8ArrayL( 2, 5, 5 );
      PTFN[ifoNumber] = XLALCreateREAL8ArrayL( 2, 5, 5 );
      PTFqVec[ifoNumber] = XLALCreateCOMPLEX8VectorSequence ( 5, numPoints );
    }
  }
  /* Create an inverse FFT plan */
  invPlan = XLALCreateReverseCOMPLEX8FFTPlan( numPoints, 0 );

  /* Read in the tmpltbank xml files */
  if ( params->spinBank )
  {
    numSpinTmplts = InspiralTmpltBankFromLIGOLw( &PTFSpinTemplate,
      spinFileName,startTemplate, stopTemplate );
    if (numSpinTmplts != 0 )
    {
      PTFtemplate = PTFSpinTemplate;
      numTmplts = numSpinTmplts;
    }
    else
      params->spinBank = NULL;
      spinBank = 0;
  }
  if ( params->noSpinBank )
  {
    numNoSpinTmplts = InspiralTmpltBankFromLIGOLw( &PTFNoSpinTemplate,
      noSpinFileName,startTemplate, stopTemplate );
    if ( numNoSpinTmplts != 0 )
    {
      PTFtemplate = PTFNoSpinTemplate;
      numTmplts = numNoSpinTmplts;
    }
    else
      params->noSpinBank = NULL;
  }
  /* If both banks present combine them and mark where to swap over */
  if ( params->spinBank && params->noSpinBank )
  {
    for (i = 0; (i < numNoSpinTmplts); PTFtemplate = PTFtemplate->next, i++)
    {
      if (i == (numNoSpinTmplts - 1))
      {
        PTFtemplate->next = PTFSpinTemplate;
        numTmplts = numSpinTmplts + numNoSpinTmplts;
      }
    }
    PTFtemplate = PTFNoSpinTemplate;
  }

  /* Create the templates needed for the bank veto, if necessary */
  UINT4 subBankSize = 0;
  struct bankTemplateOverlaps *bankNormOverlaps = NULL;
  struct bankComplexTemplateOverlaps *bankOverlaps = NULL;
  struct bankDataOverlaps *dataOverlaps = NULL;
  
  if ( params->doBankVeto )
  {
    /* Reads in and initializes the bank veto sub bank */
    subBankSize = coh_PTF_read_sub_bank(params,&PTFBankTemplates);
    bankNormOverlaps = LALCalloc( subBankSize,sizeof( *bankNormOverlaps));
    bankOverlaps = LALCalloc( subBankSize,sizeof( *bankOverlaps));
    dataOverlaps = LALCalloc(subBankSize,sizeof( *dataOverlaps));
    bankFcTmplts = LALCalloc( subBankSize, sizeof( *bankFcTmplts ));
    /* Create necessary structure to hold Q(f) */
    for (ui =0 ; ui < subBankSize; ui++)
    {
      bankFcTmplts[ui].PTFQtilde = 
          XLALCreateCOMPLEX8VectorSequence( 1, numPoints / 2 + 1 );
    }
    PTFBankvetoHead = PTFBankTemplates;
    
    for ( ui=0 ; ui < subBankSize ; ui++ )
    {
      coh_PTF_template(fcTmplt,PTFBankTemplates,
          fcTmpltParams);
      PTFBankTemplates = PTFBankTemplates->next;
      /* Only store Q1. Structures used in fcTmpltParams will be overwritten */
      for ( uj = 0 ; uj < (numPoints/2 +1) ; uj++ )
      {
        bankFcTmplts[ui].PTFQtilde->data[uj] = fcTmplt->PTFQtilde->data[uj];
      }
    }
    /* Calculate the overlap between templates for bank veto */
    for ( ui = 0 ; ui < subBankSize; ui++ )
    {
      for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
      {
        if ( params->haveTrig[ifoNumber] )
        {
          bankNormOverlaps[ui].PTFM[ifoNumber]=
              XLALCreateREAL8ArrayL( 2, 1, 1 );
          memset( bankNormOverlaps[ui].PTFM[ifoNumber]->data,
              0, 1 * sizeof(REAL8) );
          /* This function calculates the overlaps between templates */
          /* This returns a REAL4 as the overlap between identical templates*/
          /* must be real. */
          coh_PTF_template_overlaps(params,&(bankFcTmplts[ui]),
              &(bankFcTmplts[ui]),invspec[ifoNumber],0,
              bankNormOverlaps[ui].PTFM[ifoNumber]);
        }
      }
    }
 
    verbose("Generated bank veto filters at %ld \n", time(NULL)-startTime);
        
  }

  /* Create the structures needed for the auto veto, if necessary */
  UINT4 timeStepPoints = 0;
  struct bankComplexTemplateOverlaps *autoTempOverlaps = NULL;

  if ( params->doAutoVeto )
  {
    /* Initializations */
    autoTempOverlaps = LALCalloc( params->numAutoPoints,
        sizeof( *autoTempOverlaps));
    timeStepPoints = params->autoVetoTimeStep*params->sampleRate;
    for (uj = 0; uj < params->numAutoPoints; uj++ )
    {
      for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
      {
        if ( params->haveTrig[ifoNumber] )
        {
          /* If it will be used initialize and zero out the overlap structure*/
          autoTempOverlaps[uj].PTFM[ifoNumber] =
              XLALCreateCOMPLEX8ArrayL( 2, 1, 1 );
          memset( autoTempOverlaps[uj].PTFM[ifoNumber]->data,
              0, 1 * sizeof(COMPLEX8) );
        }
        else
          autoTempOverlaps[uj].PTFM[ifoNumber] = NULL;
      }
    }
  }
    

  PTFbankhead = PTFtemplate;
  for ( j = 0; j < numSegments; ++j ) /* Loop over segments */
  {
    if ( params->doBankVeto )
    {
      /* Calculate overlap between bank templates and data for bank veto */
      for ( ui = 0 ; ui < subBankSize ; ui++ ) /* Loop over bank veto temps*/
      {
        for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
        {
          if ( params->haveTrig[ifoNumber] )
          {
            dataOverlaps[ui].PTFqVec[ifoNumber] =
                XLALCreateCOMPLEX8VectorSequence ( 1, 
                3*numPoints/4 - numPoints/4 + 10000);
            bankOverlaps[ui].PTFM[ifoNumber]=XLALCreateCOMPLEX8ArrayL(2,1,1);
            /* This function calculates the overlap */
            coh_PTF_bank_filters(params,&(bankFcTmplts[ui]),0,
                &segments[ifoNumber]->sgmnt[j],invPlan,PTFqVec[ifoNumber],
                dataOverlaps[ui].PTFqVec[ifoNumber],0,0);
          }
        }
      }
      verbose("Generated bank veto filters for segment %d at %ld \n",j, time(NULL)-startTime);
    }
    PTFtemplate = PTFbankhead;

    /* Loop over templates in the bank */
    for (i = 0; (i < numTmplts); PTFtemplate = PTFtemplate->next, i++)
    {
      /* Determine if this template is non-spinning */
      if (i >= numNoSpinTmplts)
        spinTemplate = 1;
      else
        spinTemplate = 0;

      /* Currently we can only run PTF. It would be nice to be able to run
         other stuff */    

      PTFtemplate->approximant = FindChirpPTF;
      PTFtemplate->order = LAL_PNORDER_TWO;
   
     /* This value is used for template generation */

      PTFtemplate->fLower = params->lowTemplateFrequency;

      /* Generate the template */
      /* PTF generator called here. For non spin Q1-5 is generated and stored
         only Q1 will be used. This would need some alteration if we planned to
         use other templates. */
       
      coh_PTF_template(fcTmplt,PTFtemplate,fcTmpltParams);

      if (spinTemplate)
        verbose("Generated spin template %d at %ld \n",i,time(NULL)-startTime);
      else
        verbose("Generated no spin template %d at %ld \n",i,time(NULL)-startTime);
      for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
      {
        if ( params->haveTrig[ifoNumber] )
        {
          segStartTime = segments[ifoNumber]->sgmnt[j].epoch;
          break;
        }
      }
      /* We only analyse middle half so add duration/4 to epoch */
      XLALGPSAdd(&segStartTime,params->segmentDuration/4.0);

      /* Generate the various time series as needed*/
      /* Need to zero these out */
      /* We only want to store data from middle half of segment */
      cohSNR = XLALCreateREAL4TimeSeries("cohSNR",
          &segStartTime,PTFtemplate->fLower,
          (1.0/params->sampleRate),&lalDimensionlessUnit,
          3*numPoints/4 - numPoints/4);
      if (params->doTraceSNR)
      {
        for (ifoNumber = 0;ifoNumber < LAL_NUM_IFO; ifoNumber++)
        {
          if ( params->haveTrig[ifoNumber] )
          {
            snrComps[ifoNumber] = XLALCreateREAL4TimeSeries("snrComps",
                &cohSNR->epoch,cohSNR->f0,cohSNR->deltaT,
                &lalDimensionlessUnit,cohSNR->data->length);
          }
        }
      }
      if (params->doNullStream)
        nullSNR = XLALCreateREAL4TimeSeries("nullSNR",
            &segStartTime,PTFtemplate->fLower,
            (1.0/params->sampleRate),&lalDimensionlessUnit,
            3*numPoints/4 - numPoints/4);
      if (params->doTraceSNR)
        traceSNR = XLALCreateREAL4TimeSeries("traceSNR",
            &segStartTime,PTFtemplate->fLower,
            (1.0/params->sampleRate),&lalDimensionlessUnit,
            3*numPoints/4 - numPoints/4);
      if ( params->doBankVeto )
      {
        bankVeto = XLALCreateREAL4TimeSeries("bankVeto",
            &segStartTime,PTFtemplate->fLower,
            (1.0/params->sampleRate),&lalDimensionlessUnit,
            3*numPoints/4 - numPoints/4);
      }
      if ( params->doAutoVeto )
      {
        autoVeto = XLALCreateREAL4TimeSeries("autoVeto",
            &segStartTime,PTFtemplate->fLower,
            (1.0/params->sampleRate),&lalDimensionlessUnit,
            3*numPoints/4 - numPoints/4);
      }
      if ( params->doChiSquare )
      {
        chiSquare = XLALCreateREAL4TimeSeries("chiSquare",
            &segStartTime,PTFtemplate->fLower,
            (1.0/params->sampleRate),&lalDimensionlessUnit,
            3*numPoints/4 - numPoints/4);
      }

      /* Loop over ifos */
      for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
      {
        if ( params->haveTrig[ifoNumber] )
        {
          /* Zero the storage vectors for the PTF filters */
          memset( PTFM[ifoNumber]->data, 0, 25 * sizeof(REAL8) );
          memset( PTFqVec[ifoNumber]->data, 0, 
                  5 * numPoints * sizeof(COMPLEX8) );

          /* Here (h|s) and (h|h) are calculated */
          coh_PTF_normalize(params,fcTmplt,invspec[ifoNumber],PTFM[ifoNumber],
              NULL,PTFqVec[ifoNumber],&segments[ifoNumber]->sgmnt[j],invPlan,
              spinTemplate);

          // In this subroutine the overlap between template h and the 
          // bank veto template is calculated
          if ( params->doBankVeto )
          {
          for ( ui = 0 ; ui < subBankSize ; ui++ )
          {
            memset(bankOverlaps[ui].PTFM[ifoNumber]->data,0,1*sizeof(COMPLEX8));
            coh_PTF_complex_template_overlaps(params,&(bankFcTmplts[ui]),
                fcTmplt,invspec[ifoNumber],0,bankOverlaps[ui].PTFM[ifoNumber]);
          }
          }
          // And if necessary the overlap between h(deltaT) and h
          if ( params->doAutoVeto )
          {
            coh_PTF_auto_veto_overlaps(params,fcTmplt,autoTempOverlaps,
                invspec[ifoNumber],invPlan,0,params->numAutoPoints,
                timeStepPoints,ifoNumber);
          }

          verbose("Made filters for ifo %d,segment %d, template %d at %ld \n", 
              ifoNumber,j,i,time(NULL)-startTime);
        }
      }
      /* If necessary calculate the null stream filters */
      if ( params->doNullStream)
      {
        memset( PTFM[LAL_NUM_IFO]->data, 0, 25 * sizeof(REAL8) );
        memset( PTFqVec[LAL_NUM_IFO]->data, 0,
                5 * numPoints * sizeof(COMPLEX8) );
        coh_PTF_normalize(params,fcTmplt,invspec[LAL_NUM_IFO],PTFM[LAL_NUM_IFO],
              NULL,PTFqVec[LAL_NUM_IFO],&segments[LAL_NUM_IFO]->sgmnt[j],
              invPlan,spinTemplate);
        verbose("Made filters for NULL stream,segmen %d, template %d at %ld\n",
              j,i,time(NULL)-startTime);
      }
      
      /* FIXME: Sky location looping should go here.
         Clustering may be required if we are doing this! */

      // This function calculates the cohSNR time series and all of the
      // signal based vetoes as appropriate
      coh_PTF_statistic(cohSNR,PTFM,PTFqVec,params,
          spinTemplate,singleDetector,timeOffsets,Fplus,Fcross,j,pValues,
          gammaBeta,snrComps,nullSNR,traceSNR,bankVeto,autoVeto,chiSquare,
          subBankSize,bankOverlaps,bankNormOverlaps,dataOverlaps,
          autoTempOverlaps,fcTmplt,invspec,segments,invPlan);
     
      verbose("Made coherent statistic for segment %d, template %d at %ld \n",
          j,i,time(NULL)-startTime);      

      // This function adds any loud events to the list of triggers 
      eventId = coh_PTF_add_triggers(params,&eventList,&thisEvent,cohSNR,*PTFtemplate,eventId,spinTemplate,singleDetector,pValues,gammaBeta,snrComps,nullSNR,traceSNR,bankVeto,autoVeto,chiSquare,PTFM);
      verbose("Generated triggers for segment %d, template %d at %ld \n",
          j,i,time(NULL)-startTime);
//      Clustering could happen here. The clustering routine needs refining
//      coh_PTF_cluster_triggers(params,&eventList,&thisEvent);
//      verbose("Clustered triggers for segment %d, template %d at %ld \n",
 //         j,i,time(NULL)-startTime);
      // Then we get a bunch of memory freeing statements
      for ( k = 0 ; k < 10 ; k++ )
      {
        if (pValues[k])
        {
            XLALDestroyREAL4TimeSeries(pValues[k]);
            pValues[k] = NULL;
        }
      }
      for ( k = 0; k < LAL_NUM_IFO; k++ )
      {
        if (snrComps[k])
        {
          XLALDestroyREAL4TimeSeries(snrComps[k]);
          snrComps[k] = NULL;
        }
      }
      if (gammaBeta[0]) XLALDestroyREAL4TimeSeries(gammaBeta[0]);
      if (gammaBeta[1]) XLALDestroyREAL4TimeSeries(gammaBeta[1]);
      if (nullSNR) XLALDestroyREAL4TimeSeries(nullSNR);
      if (traceSNR) XLALDestroyREAL4TimeSeries(traceSNR);
      if (bankVeto) XLALDestroyREAL4TimeSeries(bankVeto);
      if (autoVeto) XLALDestroyREAL4TimeSeries(autoVeto);
      if (chiSquare) XLALDestroyREAL4TimeSeries(chiSquare);
      XLALDestroyREAL4TimeSeries(cohSNR);
    }
    if ( params->doBankVeto )
    {
      for ( ui = 0 ; ui < subBankSize ; ui++ )
      {
        for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
        {
          if ( dataOverlaps[ui].PTFqVec[ifoNumber] )
            XLALDestroyCOMPLEX8VectorSequence( dataOverlaps[ui].PTFqVec[ifoNumber]);
          if ( bankOverlaps[ui].PTFM[ifoNumber] )
            XLALDestroyCOMPLEX8Array( bankOverlaps[ui].PTFM[ifoNumber]);
        }
      }
    }
  } // Main loop is ended here
  coh_PTF_output_events_xml( params->outputFile, eventList, procpar, params );

  verbose("Generated output xml file, cleaning up and exiting at %ld \n",
      time(NULL)-startTime);

  // This function cleans up memory usage
  coh_PTF_cleanup(procpar,fwdplan,revplan,invPlan,channel,
      invspec,segments,eventList,PTFbankhead,fcTmplt,fcTmpltParams,
      fcInitParams,PTFM,PTFN,PTFqVec,Fplus,Fcross,timeOffsets);
  
  while ( PTFBankvetoHead )
  {
    InspiralTemplate *thisTmplt;
    thisTmplt = PTFBankvetoHead;
    PTFBankvetoHead = PTFBankvetoHead->next;
    if ( thisTmplt->event_id )
    {
      LALFree( thisTmplt->event_id );
    }
    LALFree( thisTmplt );
  }

  coh_PTF_free_bank_veto_memory(bankNormOverlaps,PTFBankTemplates,bankFcTmplts,subBankSize,bankOverlaps,dataOverlaps);

  if ( autoTempOverlaps )
  {
    for (uj = 0; uj < params->numAutoPoints; uj++ )
    {
      for( ifoNumber = 0; ifoNumber < LAL_NUM_IFO; ifoNumber++)
      {
        if ( params->haveTrig[ifoNumber] )
        {
          if ( autoTempOverlaps[uj].PTFM[ifoNumber] )
          {
            XLALDestroyCOMPLEX8Array( autoTempOverlaps[uj].PTFM[ifoNumber] );
          }
        }
      }
    }
    LALFree( autoTempOverlaps );
  }


  verbose("Generated output xml file, cleaning up and exiting at %ld \n",
      time(NULL)-startTime);
  LALCheckMemoryLeaks();
  return 0;
}

void coh_PTF_statistic(
    REAL4TimeSeries         *cohSNR,
    REAL8Array              *PTFM[LAL_NUM_IFO+1],
    COMPLEX8VectorSequence  *PTFqVec[LAL_NUM_IFO+1],
    struct coh_PTF_params   *params,
    UINT4                   spinTemplate,
    UINT4                   singleDetector,
    REAL8                   *timeOffsets,
    REAL8                   *Fplus,
    REAL8                   *Fcross,
    INT4                    segmentNumber,
    REAL4TimeSeries         *pValues[10],
    REAL4TimeSeries         *gammaBeta[2],
    REAL4TimeSeries         *snrComps[LAL_NUM_IFO],
    REAL4TimeSeries         *nullSNR,
    REAL4TimeSeries         *traceSNR,
    REAL4TimeSeries         *bankVeto,
    REAL4TimeSeries         *autoVeto,
    REAL4TimeSeries         *chiSquare,
    UINT4                   subBankSize,
    struct bankComplexTemplateOverlaps *bankOverlaps,
    struct bankTemplateOverlaps *bankNormOverlaps,
    struct bankDataOverlaps *dataOverlaps,
    struct bankComplexTemplateOverlaps *autoTempOverlaps,
    FindChirpTemplate       *fcTmplt,
    REAL4FrequencySeries    *invspec[LAL_NUM_IFO+1],
    RingDataSegments        *segments[LAL_NUM_IFO+1],
    COMPLEX8FFTPlan         *invPlan
)

{
// This function generates the SNR for every point in time and, where
// appropriate calculates the desired signal based vetoes.

  UINT4 i, j, k, m, vecLength, vecLengthTwo, UNUSED vecLengthSquare, UNUSED vecLengthTwoSquare;
  INT4 l;
  INT4 timeOffsetPoints[LAL_NUM_IFO];
  REAL4 deltaT = cohSNR->deltaT;
  UINT4 numPoints = floor( params->segmentDuration * params->sampleRate + 0.5 );

  // Code works slightly differently if spin/non spin and single/coherent
  if (spinTemplate)
    vecLength = 5;
  else
    vecLength = 1;
  if (singleDetector)
    vecLengthTwo = vecLength;
  else
    vecLengthTwo = 2* vecLength;
  vecLengthSquare = vecLength*vecLength;
  vecLengthTwoSquare = vecLengthTwo * vecLengthTwo;

  // These arrays are used to store the maximized quantities
  // For non spin these are the 4 F-stat parameters (only 2 for one detector)
  // For spin these are the P values
  for ( i = 0 ; i < vecLengthTwo ; i++ )
  {
    pValues[i] = XLALCreateREAL4TimeSeries("Pvalue",
          &cohSNR->epoch,cohSNR->f0,cohSNR->deltaT,
          &lalDimensionlessUnit,cohSNR->data->length);
  }
  if (! spinTemplate)
  {
    for ( i = vecLengthTwo ; i < 2*vecLengthTwo ; i++ )
    {
      pValues[i] = XLALCreateREAL4TimeSeries("Pvalue",
          &cohSNR->epoch,cohSNR->f0,cohSNR->deltaT,
          &lalDimensionlessUnit,cohSNR->data->length);
    }
  }
  if (spinTemplate)
  {
    /* These store a amplitude and phase information for PTF search */
    gammaBeta[0] = XLALCreateREAL4TimeSeries("Gamma",
            &cohSNR->epoch,cohSNR->f0,cohSNR->deltaT,
            &lalDimensionlessUnit,cohSNR->data->length);
    gammaBeta[1] = XLALCreateREAL4TimeSeries("Beta",
            &cohSNR->epoch,cohSNR->f0,cohSNR->deltaT,
            &lalDimensionlessUnit,cohSNR->data->length);
  }

  /* FIXME: All the time series should be outputtable */
  REAL4 u1[vecLengthTwo],u2[vecLengthTwo],v1[vecLengthTwo],v2[vecLengthTwo];
  REAL4 *v1p,*v2p;
  REAL4 u1N[vecLength],u2N[vecLength],v1N[vecLength],v2N[vecLength];
  REAL4 v1_dot_u1, v1_dot_u2, v2_dot_u1, v2_dot_u2,max_eigen;
  REAL4 recSNR,traceSNRsq;
  REAL4 dAlpha,dBeta,dCee;
  REAL4 pValsTemp[vecLengthTwo];
  REAL4 betaGammaTemp[2];
  REAL4 a[LAL_NUM_IFO], b[LAL_NUM_IFO];

  gsl_matrix *BNull = gsl_matrix_alloc(vecLength,vecLength);
  gsl_matrix *B2Null = gsl_matrix_alloc(vecLength,vecLength);
  /* FIXME: the 50s below seem to hardcode a limit on the number of templates
     this should not be hardcoded. Note that this value is hardcoded in some
     function declarations as well as here! */
  gsl_matrix *Bankeigenvecs[50];
  gsl_vector *Bankeigenvals[50];
  gsl_matrix *Autoeigenvecs = NULL;
  gsl_vector *Autoeigenvals = NULL;
  for (i = 0; i < 50; i++)
  {
    Bankeigenvecs[i] = NULL;
    Bankeigenvals[i] = NULL;  
  }
  gsl_permutation *p = gsl_permutation_alloc(vecLengthTwo);
  gsl_permutation *pNull = gsl_permutation_alloc(vecLength);
  gsl_eigen_symmv_workspace *matTempNull = gsl_eigen_symmv_alloc (vecLength);
  gsl_matrix *eigenvecs = gsl_matrix_alloc(vecLengthTwo,vecLengthTwo);
  gsl_vector *eigenvals = gsl_vector_alloc(vecLengthTwo);
  gsl_matrix *eigenvecsNull = gsl_matrix_alloc(vecLength,vecLength);
  gsl_vector *eigenvalsNull = gsl_vector_alloc(vecLength);

  // a = Fplus , b = Fcross
  /* FIXME: Replace all instances with a and b with Fplus and Fcross */
  for (i = 0; i < LAL_NUM_IFO; i++)
  {
    a[i] = Fplus[i];
    b[i] = Fcross[i];
  }

  // This function takes the (Q_i|Q_j) matrices, combines it across the ifos
  // and returns the eigenvalues and eigenvectors of this new matrix.
  // We later rotate and rescale the (Q_i|s) values such that in the new basis
  // this matrix will be the identity matrix.
  // For non-spin this describes the rotation into the dominant polarization
  coh_PTF_calculate_bmatrix(params,eigenvecs,eigenvals,a,b,PTFM,vecLength,vecLengthTwo,5);

  // If required also calculate these eigenvalues/vectors for the null stream 
  if ( params->doNullStream )
  {
    for (i = 0; i < vecLength; i++ )
    {
      for (j = 0; j < vecLength; j++ )
      {
        gsl_matrix_set(BNull,i,j,PTFM[LAL_NUM_IFO]->data[i*5+j]);
        gsl_matrix_set(B2Null,i,j,PTFM[LAL_NUM_IFO]->data[i*5+j]);
      }
    }
    gsl_eigen_symmv(B2Null,eigenvalsNull,eigenvecsNull,matTempNull); 
  }

  /* This loop takes the time offset in seconds and converts to time offset
  * in data points */
  for (i = 0; i < LAL_NUM_IFO; i++ )
  {
    timeOffsetPoints[i]=(int)(timeOffsets[i]/deltaT);
  }

  v1p = LALCalloc(vecLengthTwo , sizeof(REAL4));
  v2p = LALCalloc(vecLengthTwo , sizeof(REAL4));


  for ( i = numPoints/4; i < 3*numPoints/4; ++i ) /* Main loop over time */
  {
    // This function combines the various (Q_i | s) and rotates them into
    // the basis as discussed above.
    coh_PTF_calculate_rotated_vectors(params,PTFqVec,v1p,v2p,a,b,
      timeOffsetPoints,eigenvecs,eigenvals,numPoints,i,vecLength,vecLengthTwo);

    /* Compute the dot products */
    v1_dot_u1 = v1_dot_u2 = v2_dot_u1 = v2_dot_u2 = max_eigen = 0.0;
    for (j = 0; j < vecLengthTwo; j++)
    {
      v1_dot_u1 += v1p[j] * v1p[j];
      v1_dot_u2 += v1p[j] * v2p[j];
      v2_dot_u2 += v2p[j] * v2p[j];
    }
    // And SNR is calculated
    // For non spin: v1p[0] * v1p[0] = ( \bf{F}_+\bf{h}_0 | \bf{s})^2
    //               v1p[1] * v1p[1] = ( \bf{F}_x\bf{h}_0 | \bf{s})^2
    //               v2p[0] * v2p[0] = ( \bf{F}_+\bf{h}_{\pi/2} | \bf{s})^2
    //               v2p[1] * v2p[1] = ( \bf{F}_x\bf{h}_{\pi/2} | \bf{s})^2
    //
    // For spin this follows Diego's notation
    if (spinTemplate == 0)
    {
      max_eigen = ( v1_dot_u1 + v2_dot_u2 );
    }
    else
    {
      max_eigen = 0.5 * ( v1_dot_u1 + v2_dot_u2 + sqrt( (v1_dot_u1 - v2_dot_u2)
          * (v1_dot_u1 - v2_dot_u2) + 4 * v1_dot_u2 * v1_dot_u2 ));
    }
    cohSNR->data->data[i-numPoints/4] = sqrt(max_eigen);
  }

  UINT4 check;
  UINT4 chisqCheck = 0;
  REAL4 bestNR,snglSNRsq;
  INT4 numPointCheck = floor(params->timeWindow/cohSNR->deltaT + 0.5);
  struct bankCohTemplateOverlaps *bankCohOverlaps = NULL;
  struct bankCohTemplateOverlaps *autoCohOverlaps = NULL;
  struct bankDataOverlaps *chisqOverlaps = NULL;
  COMPLEX8VectorSequence *tempqVec = NULL;
  REAL4 *frequencyRangesPlus = NULL;
  REAL4 *frequencyRangesCross = NULL;
  REAL4 *powerBinsPlus = NULL;
  REAL4 *powerBinsCross = NULL;
  REAL4 fLow,fHigh;

  // Now we calculate all the extrinsic parameters and signal based vetoes
  // Only calculated if this will be a trigger

  for ( i = numPoints/4; i < 3*numPoints/4; ++i ) /* loop over time */
  {
    if (cohSNR->data->data[i-numPoints/4] > params->threshold)
    {
      check = 1;
      for (l = (INT4)(i-numPoints/4)-numPointCheck; l < (INT4)(i-numPoints/4)+numPointCheck; l++)
      {
        if (l < 0)
          l = 0;
        if (l > (INT4)(cohSNR->data->length-1))
          break;
        if (cohSNR->data->data[l] > cohSNR->data->data[i-numPoints/4])
        {
          check = 0;
          break;
        }
      }
      check = 1;
      if (check)
      {
        // The following block extracts the values of extrinsic parameters.
        // This follows the method set out in Diego's thesis to do this.
        /* FIXME: This is probably broke for coherent case now! */
        coh_PTF_calculate_rotated_vectors(params,PTFqVec,v1p,v2p,a,b,timeOffsetPoints,
        eigenvecs,eigenvals,numPoints,i,vecLength,vecLengthTwo);
        v1_dot_u1 = v1_dot_u2 = v2_dot_u1 = v2_dot_u2 = 0;
        for (j = 0; j < vecLengthTwo; j++)
        {
          u1[j] = v1p[j] / (pow(gsl_vector_get(eigenvals,j),0.5));
          u2[j] = v2p[j] / (pow(gsl_vector_get(eigenvals,j),0.5));
          v1[j] = u1[j] * gsl_vector_get(eigenvals,j);
          v2[j] = u2[j] * gsl_vector_get(eigenvals,j);
          v1_dot_u1 += v1[j]*u1[j];
          v1_dot_u2 += v1[j]*u2[j];
          v2_dot_u2 += v2[j]*u2[j];
        }
        if (! spinTemplate)
        {
          /* This is a lot easier when there is no spin 
             Note that we output values in the dominant polarization frame
             for the case of non-spin */
          for ( j = 0 ; j < vecLengthTwo ; j++ )
          {
            pValues[j]->data->data[i - numPoints/4] = sqrt(v1[j]*u1[j]);
            pValues[j+vecLengthTwo]->data->data[i - numPoints/4] =sqrt(v2[j]*u2[j]);
          }
        }
            
        /* For PTF it is a bit more tricksy. Here we use the methods given
           in Diego's thesis.
           Values are outputted in the original basis */
        if ( spinTemplate )
        {
          dCee = (max_eigen - v1_dot_u1) / v1_dot_u2;
          dAlpha = 1./(v1_dot_u1 + dCee * 2 * v1_dot_u2 + dCee*dCee*v2_dot_u2);
          dAlpha = pow(dAlpha,0.5);
          dBeta = dCee*dAlpha;
          // The p Values are calculated in the rotated frame
          for ( j = 0 ; j < vecLengthTwo ; j++ )
          {
            pValsTemp[j] = dAlpha*u1[j] + dBeta*u2[j];  
            pValues[j]->data->data[i - numPoints/4] = 0.;
          } 
          // This loop can be used to verify that the SNR obtained is as before
          recSNR = 0;
          for ( j = 0 ; j < vecLengthTwo ; j++ )
          {
            for ( k = 0 ; k < vecLengthTwo ; k++ )
            {
              recSNR += pValsTemp[j]*pValsTemp[k] * (v1[j]*v1[k]+v2[j]*v2[k]);
            }
          }
          // Then we calculate the two phase/amplitude terms beta and gamma
          // These are explained in Diego's thesis
          betaGammaTemp[0] = 0;
          betaGammaTemp[1] = 0;
          for ( j = 0 ; j < vecLengthTwo ; j++ )
          {
            betaGammaTemp[0] += pValsTemp[j]*v1[j];
            betaGammaTemp[1] += pValsTemp[j]*v2[j];
          }
          gammaBeta[0]->data->data[i - numPoints/4] = betaGammaTemp[0];
          gammaBeta[1]->data->data[i - numPoints/4] = betaGammaTemp[1];
  
          // The p Values need to be rotated back into the original frame.
          // Currently we are recording values in rotated frame
          for ( j = 0 ; j < vecLengthTwo ; j++ )
          {
            for ( k = 0 ; k < vecLengthTwo ; k++ )
            {
              pValues[j]->data->data[i-numPoints/4]+=gsl_matrix_get(eigenvecs,j,k)*pValsTemp[k];
            }
          }
 
          // And we check that this still gives the expected SNR in the
          // unrotated basis.
          for ( j = 0; j < vecLengthTwo ; j++ ) /* Construct the vi vectors */
          {
            v1[j] = 0.;
            v2[j] = 0.;
            for( k = 0; k < LAL_NUM_IFO; k++)
            {
              if ( params->haveTrig[k] )
              {
                if (j < vecLength)
                {
                  v1[j] += a[k] * PTFqVec[k]->data[j*numPoints+i+timeOffsetPoints[k]].re;
                  v2[j] += a[k] * PTFqVec[k]->data[j*numPoints+i+timeOffsetPoints[k]].im;
                }
                else
                {
                  v1[j] += b[k] * PTFqVec[k]->data[(j-vecLength)*numPoints+i+timeOffsetPoints[k]].re;
                  v2[j] += b[k] * PTFqVec[k]->data[(j-vecLength)*numPoints+i+timeOffsetPoints[k]].im;
                }
              }
            }
          }
          recSNR = 0;
          for ( j = 0 ; j < vecLengthTwo ; j++ )
          {
            for ( k = 0 ; k < vecLengthTwo ; k++ )
            {
              recSNR += pValues[j]->data->data[i-numPoints/4]*pValues[k]->data->data[i-numPoints/4] * (v1[j]*v1[k]+v2[j]*v2[k]);
            }          
          }
        }

        // First sbv to be calculated is the null SNR.
        if (params->doNullStream)
        {
          // As with SNR do the rotation and calculate SNR.
          for ( j = 0; j < vecLength; j++ ) /* Construct the vi vectors */
          {
            v1N[j] = PTFqVec[LAL_NUM_IFO]->data[j*numPoints+i].re;
            v2N[j] = PTFqVec[LAL_NUM_IFO]->data[j*numPoints+i].im;
          }
     
          for ( j = 0 ; j < vecLength ; j++ )
          {
            u1N[j] = 0.;
            u2N[j] = 0.;
            for ( k = 0 ; k < vecLength ; k++ )
            {
              u1N[j] += gsl_matrix_get(eigenvecsNull,k,j)*v1N[k];
              u2N[j] += gsl_matrix_get(eigenvecsNull,k,j)*v2N[k];
            }
            u1N[j] = u1N[j] / (pow(gsl_vector_get(eigenvalsNull,j),0.5));
            u2N[j] = u2N[j] / (pow(gsl_vector_get(eigenvalsNull,j),0.5));
          }
          /* Compute the dot products */
          v1_dot_u1 = v1_dot_u2 = v2_dot_u1 = v2_dot_u2 = max_eigen = 0.0;
          for (j = 0; j < vecLength; j++)
          {
            v1_dot_u1 += u1N[j] * u1N[j];
            v1_dot_u2 += u1N[j] * u2N[j];
            v2_dot_u2 += u2N[j] * u2N[j];
          }
          if (spinTemplate == 0)
          {
            max_eigen = 0.5 * ( v1_dot_u1 + v2_dot_u2 );
          }
          else
          {
            max_eigen = 0.5*(v1_dot_u1+v2_dot_u2+sqrt((v1_dot_u1-v2_dot_u2)
                * (v1_dot_u1 - v2_dot_u2) + 4 * v1_dot_u2 * v1_dot_u2 ));
          }
          nullSNR->data->data[i-numPoints/4] = sqrt(max_eigen);
        }

        // Next up is Trace SNR and the SNR components
        /* FIXME: Sngl detector SNRs are only correct for coherent non spin*/
        /* FIXME: This loop should never be run in single detector mode */
        if (params->doTraceSNR)
        {
          // traceSNR is calculated as normal SNR but cross terms are not added
          traceSNRsq = 0;
          for( k = 0; k < LAL_NUM_IFO; k++)
          {
            if ( params->haveTrig[k] )
            {
              for ( j = 0; j < vecLengthTwo ; j++ )
              {
                if (j < vecLength)
                {
                  v1[j] = a[k] * PTFqVec[k]->data[j*numPoints+i+timeOffsetPoints[k]].re;
                  v2[j] = a[k] * PTFqVec[k]->data[j*numPoints+i+timeOffsetPoints[k]].im;
                }
                else
                {
                  v1[j] = b[k] * PTFqVec[k]->data[(j-vecLength)*numPoints+i+timeOffsetPoints[k]].re;
                  v2[j] = b[k] * PTFqVec[k]->data[(j-vecLength)*numPoints+i+timeOffsetPoints[k]].im;
                }
              }
              for ( j = 0 ; j < vecLengthTwo ; j++ )
              {
                u1[j] = 0.;
                u2[j] = 0.;
                for ( m = 0 ; m < vecLengthTwo ; m++ )
                {
                  u1[j] += gsl_matrix_get(eigenvecs,m,j)*v1[m];
                  u2[j] += gsl_matrix_get(eigenvecs,m,j)*v2[m];
                }
                u1[j] = u1[j] / (pow(gsl_vector_get(eigenvals,j),0.5));
                u2[j] = u2[j] / (pow(gsl_vector_get(eigenvals,j),0.5));
              }
              /* Compute the dot products */
              v1_dot_u1 = v1_dot_u2 = v2_dot_u1 = v2_dot_u2 = max_eigen = 0.0;
              for (j = 0; j < vecLengthTwo; j++)
              {
                v1_dot_u1 += u1[j] * u1[j];
                v1_dot_u2 += u1[j] * u2[j];
                v2_dot_u2 += u2[j] * u2[j];
              }
              if (spinTemplate == 0)
              {
                max_eigen = ( v1_dot_u1 + v2_dot_u2 );
              }
              else
              {
                max_eigen = 0.5 * ( v1_dot_u1 + v2_dot_u2 + sqrt( (v1_dot_u1 - v2_dot_u2)
                * (v1_dot_u1 - v2_dot_u2) + 4 * v1_dot_u2 * v1_dot_u2 ));
              }
              // This needs to be converted for spinning case!
              snglSNRsq = v1[0]*v1[0] + v2[0]*v2[0];
              snglSNRsq = snglSNRsq/(a[k]*a[k]*PTFM[k]->data[0]);
              traceSNRsq += max_eigen;
              snrComps[k]->data->data[i-numPoints/4]=sqrt(snglSNRsq);
            }
          }
          traceSNR->data->data[i-numPoints/4] = sqrt(traceSNRsq);
        }

        // Next is the bank veto
        if ( params->doBankVeto )
        {
          if (! singleDetector)
          {
            for ( j = 0 ; j < subBankSize+1 ; j++ )
            {
              if (! Bankeigenvecs[j] )
              {
                // FIXME: Lots of hardcoded vector lengths under here
                Bankeigenvecs[j] = gsl_matrix_alloc(2,2);
                Bankeigenvals[j] = gsl_vector_alloc(2);
                // Here we calculate the eigenvectors for each bank template
                if (j == subBankSize)
                {
                  coh_PTF_calculate_bmatrix(params,Bankeigenvecs[j],
                      Bankeigenvals[j],a,b,PTFM,1,2,5);
                }
                else
                {
                  coh_PTF_calculate_bmatrix(params,Bankeigenvecs[j],
                      Bankeigenvals[j],a,b,bankNormOverlaps[j].PTFM,1,2,1);
                }
              }
            }

            if (! bankCohOverlaps)
            {
              bankCohOverlaps = LALCalloc(subBankSize,sizeof(*bankCohOverlaps));
              for ( j = 0 ; j < subBankSize; j++ )
              {
                bankCohOverlaps[j].rotReOverlaps = gsl_matrix_alloc(2,2);
                bankCohOverlaps[j].rotImOverlaps = gsl_matrix_alloc(2,2);
                // We calculate the coherent overlaps in this function
                coh_PTF_calculate_coherent_bank_overlaps(params,bankOverlaps[j],
                    bankCohOverlaps[j],a,b,Bankeigenvecs[subBankSize],
                    Bankeigenvals[subBankSize],Bankeigenvecs[j],
                    Bankeigenvals[j]);
              }
            }
            // In this function all the filters are combined to produce the
            // value of the bank veto.
            bankVeto->data->data[i-numPoints/4] = coh_PTF_calculate_bank_veto(numPoints,i,subBankSize,a,b,params,bankCohOverlaps,dataOverlaps,PTFqVec,timeOffsetPoints,Bankeigenvecs,Bankeigenvals);
          }
          // The single detector function is a little messy at the moment
          // FIXME: Chisq single detector stuff doesn't really work at all
          // currently we can produce a bank veto value only but the function
          // is terribly written! This needs merging into the main functions
          if (singleDetector)
            bankVeto->data->data[i-numPoints/4] = coh_PTF_calculate_bank_veto_max_phase(numPoints,i,subBankSize,PTFM,params,bankOverlaps,bankNormOverlaps,dataOverlaps,PTFqVec,timeOffsetPoints);
        }   

        // Now we do the auto veto
        if ( params->doAutoVeto )
        {
          if (! singleDetector)
          {
            if (! Autoeigenvecs )
            {
              Autoeigenvecs = gsl_matrix_alloc(2,2);
              Autoeigenvals = gsl_vector_alloc(2);
              // Again the eigenvectors/values are calculated
              /* FIXME: I think these vectors are the same as the ones used
                 for the SNR! */
              coh_PTF_calculate_bmatrix(params,Autoeigenvecs,Autoeigenvals,
                  a,b,PTFM,1,2,5);
            }

            if (! autoCohOverlaps)
            {
              autoCohOverlaps = LALCalloc(params->numAutoPoints,sizeof(*autoCohOverlaps));
              for ( j = 0 ; j < params->numAutoPoints; j++ )
              {
                autoCohOverlaps[j].rotReOverlaps = gsl_matrix_alloc(2,2);
                autoCohOverlaps[j].rotImOverlaps = gsl_matrix_alloc(2,2);
                // The coherent rotated overlaps are calculated
                coh_PTF_calculate_coherent_bank_overlaps(
                    params,autoTempOverlaps[j],
                    autoCohOverlaps[j],a,b,Autoeigenvecs,
                    Autoeigenvals,Autoeigenvecs,Autoeigenvals);
              }
            }
          }
          // Auto veto is calculated
          autoVeto->data->data[i-numPoints/4] = coh_PTF_calculate_auto_veto(numPoints,i,a,b,params,autoCohOverlaps,PTFqVec,timeOffsetPoints,Autoeigenvecs,Autoeigenvals);
        }
      }
    }
  }
  /* To save memory we cut the loop here, clean the memory before calculating
  chi square */
  LALFree(v1p);
  LALFree(v2p);

  if (params->doBankVeto)
  {
    for ( j = 0 ; j < subBankSize+1 ; j++ )
    {
      if (Bankeigenvecs[j])
        gsl_matrix_free(Bankeigenvecs[j]);
      if (Bankeigenvals[j])
        gsl_vector_free(Bankeigenvals[j]);
    }
    if (bankCohOverlaps)
    {
      for ( j = 0 ; j < subBankSize ; j++ )
      {
        gsl_matrix_free(bankCohOverlaps[j].rotReOverlaps);
        gsl_matrix_free(bankCohOverlaps[j].rotImOverlaps);
      }
      LALFree(bankCohOverlaps);
    }
  }

  if (params->doAutoVeto)
  {
    if ( Autoeigenvecs )
      gsl_matrix_free( Autoeigenvecs );
      Autoeigenvecs = NULL;
    if ( Autoeigenvals )
      gsl_vector_free( Autoeigenvals );
      Autoeigenvals = NULL;
    if (autoCohOverlaps)
    {
      for ( j = 0 ; j < params->numAutoPoints ; j++ )
      {
        gsl_matrix_free(autoCohOverlaps[j].rotReOverlaps);
        gsl_matrix_free(autoCohOverlaps[j].rotImOverlaps);
      }
      LALFree(autoCohOverlaps);
    }
  }

  /* And do the loop again to calculate chi square */

  for ( i = numPoints/4; i < 3*numPoints/4; ++i ) /* loop over time */
  {
    if (cohSNR->data->data[i-numPoints/4] > params->threshold)
    {
      check = 1;
      for (l = (INT4)(i-numPoints/4)-numPointCheck; l < (INT4)(i-numPoints/4)+numPointCheck; l++)
      {
        if (l < 0)
          l = 0;
        if (l > (INT4)(cohSNR->data->length-1))
          break;
        if (cohSNR->data->data[l] > cohSNR->data->data[i-numPoints/4])
        {
          check = 0;
          break;
        }
      }
      if (check)
      {
        /* Test whether to do chi^2 */
        if ( params->chiSquareCalcThreshold )
        {
          chisqCheck = 1;
          
          bestNR = cohSNR->data->data[i-numPoints/4];

          /* IS the null stream too large? */
          if (params->doNullStream)
          {
            if (nullSNR->data->data[i-numPoints/4] > params->nullStatThreshold \
                && bestNR < params->nullStatGradOn)
            {
              chisqCheck = 0;
            }
            else if (bestNR > params->nullStatGradOn)
            {
              if (nullSNR->data->data[i-numPoints/4] > (params->nullStatThreshold + (bestNR - params->nullStatGradOn)*params->nullStatGradient))
              {
                chisqCheck = 0;
              }
            }
          }
  
          /* Is bank new SNR too large? */
          if (params->doBankVeto)
          {
            if (bankVeto->data->data[i-numPoints/4] > 40)
              bestNR = bestNR/pow(( 1 + pow(bankVeto->data->data[i-numPoints/4]/((REAL4)subBankSize*4.),params->bankVetoq/params->bankVeton))/2.,1./params->bankVetoq);
            if (bestNR < params->chiSquareCalcThreshold)
              chisqCheck = 0;
          }

          bestNR = cohSNR->data->data[i-numPoints/4];

          /* Is auto new SNR too large */
          if (params->doAutoVeto)
          {
            if (autoVeto->data->data[i-numPoints/4] > 40)
              bestNR = bestNR/pow(( 1 + pow(autoVeto->data->data[i-numPoints/4]/((REAL4)params->numAutoPoints*4.),params->autoVetoq/params->autoVeton))/2.,1./params->autoVetoq);
            if (bestNR < params->chiSquareCalcThreshold)
              chisqCheck = 0;
          } 
        }
        else
          chisqCheck = 1;

        /* If no problems then calculate chi squared */
        if (params->doChiSquare && chisqCheck )
        {
          if (! Autoeigenvecs )
          {
            /* FIXME: Again hardcoded vector lengths */
            Autoeigenvecs = gsl_matrix_alloc(2,2);
            Autoeigenvals = gsl_vector_alloc(2);
            // Again the eigenvectors/values are calculated
            /* FIXME: Again these are the same as the SNR vecs.*/
            coh_PTF_calculate_bmatrix(params,Autoeigenvecs,Autoeigenvals,
                a,b,PTFM,1,2,5);
          }
          if (! frequencyRangesPlus)
          {
            frequencyRangesPlus = (REAL4 *)
              LALCalloc( params->numChiSquareBins-1, sizeof(REAL4) );
            frequencyRangesCross = (REAL4 *)
              LALCalloc( params->numChiSquareBins-1, sizeof(REAL4) );
            powerBinsPlus = (REAL4 *) 
              LALCalloc( params->numChiSquareBins, sizeof(REAL4) );
            powerBinsCross = (REAL4 *)
              LALCalloc( params->numChiSquareBins, sizeof(REAL4) );
            coh_PTF_calculate_standard_chisq_freq_ranges(params,fcTmplt,invspec,PTFM,a,b,frequencyRangesPlus,frequencyRangesCross,Autoeigenvecs);
            coh_PTF_calculate_standard_chisq_power_bins(params,fcTmplt,invspec,PTFM,a,b,frequencyRangesPlus,powerBinsPlus,powerBinsCross,Autoeigenvecs);
          }
          if (! tempqVec)
            tempqVec = XLALCreateCOMPLEX8VectorSequence ( 1, numPoints );
          if (! chisqOverlaps)
          {
            chisqOverlaps = LALCalloc(params->numChiSquareBins,sizeof( *chisqOverlaps));
            for( j = 0; j < params->numChiSquareBins; j++)
            {
              if (params->numChiSquareBins == 1)
              {
                fLow = 0;
                fHigh = 0;
              }
              else if (j == 0)
              {
                fLow = 0;
                fHigh = frequencyRangesPlus[0];
              }
              else if (j == params->numChiSquareBins-1)
              {
                fLow = frequencyRangesPlus[params->numChiSquareBins-2];
                fHigh = 0;
              }
              else
              {
                fLow = frequencyRangesPlus[j-1];
                fHigh = frequencyRangesPlus[j];
              }                 
              for( k = 0; k < LAL_NUM_IFO; k++)
              {
                if ( params->haveTrig[k] )
                {
                  chisqOverlaps[j].PTFqVec[k] =
                      XLALCreateCOMPLEX8VectorSequence ( 1,
                      3*numPoints/4 - numPoints/4 + 10000);
                  coh_PTF_bank_filters(params,fcTmplt,0,
                  &segments[k]->sgmnt[segmentNumber],invPlan,tempqVec,
                  chisqOverlaps[j].PTFqVec[k],fLow,fHigh);
                }
                else
                  chisqOverlaps[j].PTFqVec[k] = NULL;
              }
            }
          }
          /* Calculate chi square here */
          chiSquare->data->data[i-numPoints/4] = coh_PTF_calculate_chi_square(params,numPoints,i,chisqOverlaps,PTFqVec,a,b,timeOffsetPoints,Autoeigenvecs,Autoeigenvals,powerBinsPlus,powerBinsCross);
        
        }
        else if (params->doChiSquare)
          chiSquare->data->data[i-numPoints/4] = 0;
      }
    }
  }

  if (params->doChiSquare)
  {
    if ( Autoeigenvecs )
      gsl_matrix_free( Autoeigenvecs );
    if ( Autoeigenvals )
      gsl_vector_free( Autoeigenvals );
    if (frequencyRangesPlus)
      LALFree(frequencyRangesPlus);
    if (frequencyRangesCross)
      LALFree(frequencyRangesCross);
    if (tempqVec)
      XLALDestroyCOMPLEX8VectorSequence( tempqVec );
    if (chisqOverlaps)
    {
      for( j = 0; j < params->numChiSquareBins; j++)
      {
        for( k = 0; k < LAL_NUM_IFO; k++)
        {
          if (chisqOverlaps[j].PTFqVec[k])
          {
            XLALDestroyCOMPLEX8VectorSequence(chisqOverlaps[j].PTFqVec[k]);
          }
        }
      }
      LALFree(chisqOverlaps);      
    }

  }

  gsl_matrix_free(BNull);
  gsl_matrix_free(B2Null);
  gsl_permutation_free(p);
  gsl_permutation_free(pNull);
  gsl_eigen_symmv_free(matTempNull);
  gsl_matrix_free(eigenvecs);
  gsl_vector_free(eigenvals);
  gsl_matrix_free(eigenvecsNull);
  gsl_vector_free(eigenvalsNull);


}

UINT8 coh_PTF_add_triggers(
    struct coh_PTF_params   *params,
    MultiInspiralTable      **eventList,
    MultiInspiralTable      **thisEvent,
    REAL4TimeSeries         *cohSNR,
    InspiralTemplate        PTFTemplate,
    UINT8                   eventId,
    UINT4                   spinTrigger,
    UINT4                   singleDetector,
    REAL4TimeSeries         *pValues[10],
    REAL4TimeSeries         *gammaBeta[2],
    REAL4TimeSeries         *snrComps[LAL_NUM_IFO],
    REAL4TimeSeries         *nullSNR,
    REAL4TimeSeries         *traceSNR,
    REAL4TimeSeries         *bankVeto,
    REAL4TimeSeries         *autoVeto,
    REAL4TimeSeries         *chiSquare,
    REAL8Array              *PTFM[LAL_NUM_IFO+1]
)
{
  // This function adds a trigger to the event list

  UINT4 i;
  INT4 j;
  UINT4 check;
  INT4 numPointCheck = floor(params->timeWindow/cohSNR->deltaT + 0.5);
  LIGOTimeGPS trigTime;
  MultiInspiralTable *lastEvent = NULL;
  MultiInspiralTable *currEvent = *thisEvent;

  for (i = 0 ; i < cohSNR->data->length ; i++)
  {
    if (cohSNR->data->data[i] > params->threshold)
    {
      check = 1;
      for (j = ((INT4)i)-numPointCheck; j < ((INT4)i)+numPointCheck; j++)
      {
        if (j < 0)
          j = 0;
        if (j > (INT4)(cohSNR->data->length-1))
          break;
        if (cohSNR->data->data[j] > cohSNR->data->data[i])
        {
          check = 0;
          break;
        }
      }
      if (check) /* Add trigger to event list */
      {
        if ( !*eventList ) 
        {
          *eventList = (MultiInspiralTable *) 
              LALCalloc( 1, sizeof(MultiInspiralTable) );
          currEvent = *eventList;
        }
        else
        {
          lastEvent = currEvent;
          currEvent = (MultiInspiralTable *) 
              LALCalloc( 1, sizeof(MultiInspiralTable) );
          lastEvent->next = currEvent;
        }
        currEvent->event_id = (EventIDColumn *) 
            LALCalloc(1, sizeof(EventIDColumn) );
        currEvent->event_id->id=eventId;
        eventId++;
        trigTime = cohSNR->epoch;
        XLALGPSAdd(&trigTime,i*cohSNR->deltaT);
        currEvent->snr = cohSNR->data->data[i];
        currEvent->mass1 = PTFTemplate.mass1;
        currEvent->mass2 = PTFTemplate.mass2;
        currEvent->chi = PTFTemplate.chi;
        currEvent->kappa = PTFTemplate.kappa;
        currEvent->mchirp = PTFTemplate.totalMass*pow(PTFTemplate.eta,3.0/5.0);
        currEvent->eta = PTFTemplate.eta;
        currEvent->end_time = trigTime;
        if (params->doNullStream)
          currEvent->null_statistic = nullSNR->data->data[i];
        if (params->doTraceSNR)
          currEvent->trace_snr = traceSNR->data->data[i];
        if (params->doBankVeto)
        {
          currEvent->bank_chisq = bankVeto->data->data[i];
          /* FIXME: For now only coherent non-spin bank chisq
          is implemented. When other versions are implemented
          this should be changed. Same for auto and chisq */
          currEvent->bank_chisq_dof = 4. * params->BVsubBankSize;
        }
        if (params->doAutoVeto)
        {
          currEvent->cont_chisq = autoVeto->data->data[i];
          currEvent->cont_chisq_dof = 4. * params->numAutoPoints;
        }
        if (params->doChiSquare)
        {
          currEvent->chisq = chiSquare->data->data[i];
          currEvent->chisq_dof = 4. * (params->numChiSquareBins - 1);
        }
        if (pValues[0])
          currEvent->amp_term_1 = pValues[0]->data->data[i];
        if (pValues[1]) 
          currEvent->amp_term_2 = pValues[1]->data->data[i];
        if (pValues[2]) 
          currEvent->amp_term_3 = pValues[2]->data->data[i];
        if (pValues[3]) 
          currEvent->amp_term_4 = pValues[3]->data->data[i];
        if (pValues[4]) 
          currEvent->amp_term_5 = pValues[4]->data->data[i];
        if (pValues[5]) 
          currEvent->amp_term_6 = pValues[5]->data->data[i];
        if (pValues[6]) 
          currEvent->amp_term_7 = pValues[6]->data->data[i];
        if (pValues[7]) 
          currEvent->amp_term_8 = pValues[7]->data->data[i];
        if (pValues[8]) 
          currEvent->amp_term_9 = pValues[8]->data->data[i];
        if (pValues[9]) 
          currEvent->amp_term_10 = pValues[9]->data->data[i];
        /* Note that these two terms are only used for debugging
        at the moment. When they are used properly they will be
        moved into sane columns! For spin they give Amp*cos(Phi_0) and
        Amp*sin(Phi_0). For non spinning the second is 0 and the
        first is some arbitrary amplitude. */
        if (gammaBeta[0])
        {
          currEvent->g1quad.re = gammaBeta[0]->data->data[i];
          currEvent->g1quad.im = gammaBeta[1]->data->data[i];
        }
        if (snrComps[LAL_IFO_G1])
        {
          currEvent->snr_g = snrComps[LAL_IFO_G1]->data->data[i];
          currEvent->sigmasq_g = PTFM[LAL_IFO_G1]->data[0];
        }
        if (snrComps[LAL_IFO_H1])
        {
          currEvent->snr_h1 = snrComps[LAL_IFO_H1]->data->data[i];
          currEvent->sigmasq_h1 = PTFM[LAL_IFO_H1]->data[0];
        }
        if (snrComps[LAL_IFO_H2])
        {
          currEvent->snr_h2 = snrComps[LAL_IFO_H2]->data->data[i];
          currEvent->sigmasq_h2 = PTFM[LAL_IFO_H2]->data[0];
        }
        if (snrComps[LAL_IFO_L1])
        {
          currEvent->snr_l = snrComps[LAL_IFO_L1]->data->data[i];
          currEvent->sigmasq_l = PTFM[LAL_IFO_L1]->data[0];
        }
        if (snrComps[LAL_IFO_T1])
        {
          currEvent->snr_t = snrComps[LAL_IFO_T1]->data->data[i];
          currEvent->sigmasq_t = PTFM[LAL_IFO_T1]->data[0];
        }
        if (snrComps[LAL_IFO_V1])
        {
          currEvent->snr_v = snrComps[LAL_IFO_V1]->data->data[i];
          currEvent->sigmasq_v = PTFM[LAL_IFO_V1]->data[0];
        }
        if (spinTrigger == 1)
        {
          if (singleDetector == 1)
            currEvent->snr_dof = 6;
          else
            currEvent->snr_dof = 12;
        }
        else
        {
          if (singleDetector == 1)
            currEvent->snr_dof = 2;
          else
            currEvent->snr_dof = 4;
        }
      }
    }
  }
  *thisEvent = currEvent;
  return eventId;
}

void coh_PTF_cluster_triggers(
  MultiInspiralTable      **eventList,
  MultiInspiralTable      **thisEvent
)
{
  /* This clustering function is currently unused. Currently clustering is
     done in post-processing, though this may need to be changed. */
  MultiInspiralTable *currEvent = *eventList;
  MultiInspiralTable *currEvent2 = NULL;
  MultiInspiralTable *newEvent = NULL;
  MultiInspiralTable *newEventHead = NULL;
  LIGOTimeGPS time1,time2;
  UINT4 rejectTrigger;
  UINT4 triggerNum = 0;
  UINT4 lenTriggers = 0;
  UINT4 numRemovedTriggers = 0;

  while (currEvent)
  {
    lenTriggers+=1;
    currEvent = currEvent->next;
  }

  currEvent = *eventList;
  UINT4 rejectTriggers[lenTriggers];

  while (currEvent)
  {
    rejectTrigger = 0;
    time1.gpsSeconds=currEvent->end_time.gpsSeconds;
    time1.gpsNanoSeconds = currEvent->end_time.gpsNanoSeconds;
    currEvent2 = *eventList;
    while (currEvent2)
    {
      time2.gpsSeconds=currEvent2->end_time.gpsSeconds;
      time2.gpsNanoSeconds=currEvent2->end_time.gpsNanoSeconds;
      if (fabs(XLALGPSDiff(&time1,&time2)) < 0.1)
      {
        if (currEvent->snr < currEvent2->snr && (currEvent->event_id->id != currEvent2->event_id->id))
        {
          rejectTrigger = 1;
          numRemovedTriggers +=1;
          break;
        }
        else
        {
          currEvent2 = currEvent2->next;
        }
      }
      else
      {
        currEvent2 = currEvent2->next;
      }
    }
    rejectTriggers[triggerNum] = rejectTrigger;
    triggerNum += 1;
    currEvent = currEvent->next;
  }

  currEvent = *eventList;
  triggerNum = 0;

  while (currEvent)
  {
    if (! rejectTriggers[triggerNum])
    {
      if (! newEventHead)
      {
        newEventHead = currEvent;
        newEvent = currEvent;
      }
      else
      {
        newEvent->next = currEvent;
        newEvent = currEvent;
      }
      currEvent = currEvent->next;
    }
    else
    {
      if ( currEvent->event_id )
      {
        LALFree( currEvent->event_id );
      }
      currEvent2 = currEvent->next;
      LALFree( currEvent );  
      currEvent = currEvent2;
    }
    triggerNum+=1;
  }
  if (newEvent)
  {
    newEvent->next = NULL;
    *eventList = newEventHead;
    *thisEvent = newEvent;
  }
} 
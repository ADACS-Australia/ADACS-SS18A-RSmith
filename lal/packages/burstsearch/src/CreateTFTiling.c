/******** <lalVerbatim file="CreateTFTilingCV"> ********
Author: Eanna Flanagan
$Id$
********* </lalVerbatim> **********/


#include <lal/LALRCSID.h>


NRCSID (CREATETFTILINGC, "$Id$");


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/SeqFactories.h>
#include <lal/RealFFT.h>
#include <lal/Thresholds.h>
#include <lal/ExcessPower.h>
#include <lal/Random.h>


#define TRUE 1
#define FALSE 0

extern INT4 lalDebugLevel;

static INT4 pow1(INT4 a, INT4 b)
{
  /* returns a^b */
  INT4 t=1;
  INT4 i;
  for(i=0;i<b;i++) t*=a;
  return(t);
}

/******** <lalVerbatim file="CreateTFTilingCP"> ********/
void
LALCreateTFTiling (
		 LALStatus              *status,
		 TFTiling               **tfTiling,
		 CreateTFTilingIn       *input
		 )
/******** </lalVerbatim> *********/
{
  INT4                          numPlanes;
  INT4                          nf;
  INT4                          i;
  INT4                          tileCount=0;

  REAL8                         fhigh;
  REAL8                         flow;

  TFTile                        **currentTile;

  INITSTATUS (status, "LALCreateTFTiling", CREATETFTILINGC);
  ATTATCHSTATUSPTR (status);

  /* Check input structure: report if NULL */
  ASSERT (input, status, EXCESSPOWERH_ENULLP, EXCESSPOWERH_MSGENULLP);
      

  /* 
   * Check return structure: tfTiling should point to a valid pointer
   * which should not yet point to anything.
   *
   */
  ASSERT (tfTiling != NULL, status, EXCESSPOWERH_ENULLP, EXCESSPOWERH_MSGENULLP);
  ASSERT (*tfTiling == NULL, status, EXCESSPOWERH_ENONNULL, 
           EXCESSPOWERH_MSGENONNULL);


  /* 
   *
   *  Make sure that input parameters are reasonable, compatible etc.
   *
   */


  ASSERT (input->overlapFactor > 1, status, 
          EXCESSPOWERH_EPOSARG, EXCESSPOWERH_MSGEPOSARG);
  ASSERT (input->length > 0, status,
          EXCESSPOWERH_EPOSARG, EXCESSPOWERH_MSGEPOSARG);
  ASSERT (input->deltaF > 0.0, status,
          EXCESSPOWERH_EPOSARG, EXCESSPOWERH_MSGEPOSARG);
  ASSERT (input->flow >= 0.0, status,
          EXCESSPOWERH_EPOSARG, EXCESSPOWERH_MSGEPOSARG);
  ASSERT (input->minFreqBins > 0, status, 
          EXCESSPOWERH_EPOSARG, EXCESSPOWERH_MSGEPOSARG);
  ASSERT (input->minTimeBins > 0, status, 
          EXCESSPOWERH_EPOSARG, EXCESSPOWERH_MSGEPOSARG);


  /*
   *
   *  compute some parameters 
   *
   */


  /* lowest and highest frequency to be used */
  flow = input->flow;
  fhigh = input->flow + input->deltaF*(REAL8)(input->length);

  /* number of frequency bins to be used in computation */
  nf = input->length;

  /* 
   * minimum sizes of TF tiles to be searched over in time
   * and freq directions must not exceed length of data used
   *
   */
  ASSERT(input->minFreqBins < nf, status, EXCESSPOWERH_EINCOMP, 
          EXCESSPOWERH_MSGEINCOMP);
  ASSERT(input->minTimeBins < nf, status, EXCESSPOWERH_EINCOMP, 
          EXCESSPOWERH_MSGEINCOMP);

  /* number of time frequency planes to be constructed */
  numPlanes = 1+(INT4)(0.5+log( (REAL8)(nf)) / log(2.0));
  /*printf("nf: %d,  numPlanes: %d\n", nf, numPlanes);*/

  /* check that length of data to be used is a power of 2 */
  ASSERT( nf == pow1(2, numPlanes-1), status, EXCESSPOWERH_EPOW2, 
          EXCESSPOWERH_MSGEPOW2);

  /*  Assign memory for *tfTiling   */
  *tfTiling = (TFTiling *) LALMalloc(sizeof(TFTiling));
  
  /*  Make sure that the allocation was succesful */
  if ( !(*tfTiling) ){
    ABORT (status, EXCESSPOWERH_EMALLOC, EXCESSPOWERH_MSGEMALLOC);
  }

  /* set some parameters */
  (*tfTiling)->numPlanes = numPlanes;
  (*tfTiling)->planesComputed = FALSE;
  (*tfTiling)->excessPowerComputed = FALSE;
  (*tfTiling)->tilesSorted = FALSE;
  (*tfTiling)->tfp = NULL;
  (*tfTiling)->dftParams = NULL;

  /* set things up for recursive generation of linked list below */
  currentTile = &((*tfTiling)->firstTile);
  *currentTile = NULL;

  /* allocate memory for vector of pointers to TF planes */
  (*tfTiling)->tfp = (COMPLEX8TimeFrequencyPlane **) 
       LALMalloc (numPlanes*sizeof(COMPLEX8TimeFrequencyPlane *));

  /*  Make sure that the allocation was succesful */
  if ( !((*tfTiling)->tfp) ){
    LALFree( *tfTiling );
    ABORT (status, EXCESSPOWERH_ENULLP, EXCESSPOWERH_MSGENULLP);
  }

  /* allocate memory for vector of pointers to DFTParams */
  (*tfTiling)->dftParams = (ComplexDFTParams **) 
       LALMalloc (numPlanes*sizeof(ComplexDFTParams *));

  /*  Make sure that the allocation was succesful */
  if ( !((*tfTiling)->dftParams) ){
    LALFree( (*tfTiling)->tfp );
    LALFree( *tfTiling );
    ABORT (status, EXCESSPOWERH_ENULLP, EXCESSPOWERH_MSGENULLP);
  }




  /* 
   *  
   *  create the set of time frequency planes and DFTParams
   *
   */


  for(i=0;i<numPlanes;i++)
    {
      TFPlaneParams                 params;
      COMPLEX8TimeFrequencyPlane    **thisPlane;
      ComplexDFTParams              **thisDftParams;

      /* setup parameter structure for creating TF plane */
      params.timeBins = pow1(2,i);
      params.freqBins = nf / params.timeBins;

      params.deltaT = 1.0 / ( (REAL8)(params.timeBins) * input->deltaF);
      params.flow = flow;

      /* Create TF plane structure */
      thisPlane = (*tfTiling)->tfp + i;
      *thisPlane=NULL;
      LALCreateTFPlane( status->statusPtr, thisPlane, &params);
      CHECKSTATUSPTR (status);

      /* create the DFTParams structure */
      {
	LALWindowParams winParams;
	winParams.type=Rectangular;
	winParams.length=params.timeBins;

	thisDftParams  = (*tfTiling)->dftParams + i;
	*thisDftParams = NULL;
	LALCreateComplexDFTParams( status->statusPtr, thisDftParams, 
                             &winParams, -1);
	CHECKSTATUSPTR (status);
	/* Its an inverse transform instead of a forward transform */
      }

    }





  /* 
   *  
   *  compute the linked list of Time Frequency Tiles
   *
   */


  /* loop over time-frequency Planes */
  for(i=0; i<numPlanes; i++)
    {
      /* coordinates of a given TF tile */
      INT4                          fstart;
      INT4                          deltaf;
      INT4                          tstart;
      INT4                          deltat;
      INT4                          incrementT;
      INT4                          incrementF;
  
      INT4                          timeBins = pow1(2,i);
      INT4                          freqBins = nf/timeBins; 

      

      deltat=input->minTimeBins;
      while (deltat <= timeBins)      
	{
	  incrementT = 1+deltat/input->overlapFactor;
	  tstart=0;
	  while (tstart <= timeBins - deltat)
	    {
	      deltaf=input->minFreqBins;
	      while (deltaf <= freqBins)      
		{
		  incrementF = 1+deltaf/input->overlapFactor;
		  fstart=0;
		  while (fstart <= freqBins - deltaf)
		    {
		      /* 
		       * 
		       *  add new Tile to linked list 
		       *
		       */
	
		      /*  Assign memory for tile */
		      *currentTile = (TFTile *) LALMalloc(sizeof(TFTile));

		      /*  Make sure that the allocation was succesful */
		      ASSERT (*currentTile, status, EXCESSPOWERH_EMALLOC, 
			      EXCESSPOWERH_MSGEMALLOC); 

		      /* assign the various fields */
		      (*currentTile)->fstart=fstart;
		      (*currentTile)->fend=fstart+deltaf-1;
		      (*currentTile)->tstart=tstart;
		      (*currentTile)->tend=tstart+deltat-1;
		      (*currentTile)->whichPlane=i;
		      (*currentTile)->nextTile=NULL;
		      (*currentTile)->excessPower=0.0;
		      (*currentTile)->alpha=0.0;
		      (*currentTile)->weight=1.0;
		      (*currentTile)->firstCutFlag=FALSE;


		      /* keep track of how many tiles */
		      tileCount++;

		      /* 
		       *  update currentTile to point at location of 
		       *  pointer to next tile
		       *
		       */
		      currentTile = &((*currentTile)->nextTile);

  		      fstart += incrementF;

		    }
		  
		  deltaf += incrementF;
		};  /* while (deltaf .. */
	      tstart += incrementT;
	    }
	  deltat += incrementT;
	};  /* while (deltat .. */
    } /* for(i=.. */


  (*tfTiling)->numTiles=tileCount;

  /* Normal exit */
  DETATCHSTATUSPTR (status);
  RETURN (status);
}




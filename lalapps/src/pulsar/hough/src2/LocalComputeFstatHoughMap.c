/*  
 *  Copyright (C) 2005-2008 Badri Krishnan, Alicia Sintes, Bernd Machenschalk
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
 * 
 */

/*
   This is a local copy of ComputeFstatHoughMap() of HierarchicalSearch.c
   See that file for details

   $Id$
*/

#include"HierarchicalSearch.h"
#include "LocalOptimizationFlags.h"

RCSID( "$Id$");

/* specials for Apples assembler */ 
#ifdef __APPLE__ 
#define AD_FLOAT ".single " 
#define AD_ASCII ".ascii " 
#define AD_ALIGN16 ".align 4" 
#define AD_ALIGN32 ".align 5" 
#define AD_ALIGN64 ".align 6" 
#else /* x86 gas */ 
#define AD_FLOAT ".float " 
#define AD_ASCII ".string " 
#define AD_ALIGN16 ".align 16" 
#define AD_ALIGN32 ".align 32" 
#define AD_ALIGN64 ".align 64" 
#endif 

#ifdef EAH_HOUGH_BATCHSIZELD
#define EAH_HOUGH_BATCHSIZE (1 << EAH_HOUGH_BATCHSIZELD)
#endif

/* prefetch compiler directives */
#if EAH_HOUGH_PREFETCH == EAH_HOUGH_PREFETCH_DIRECT
#if defined(__INTEL_COMPILER) ||  defined(_MSC_VER)
// not tested yet with icc or MS Visual C 
#include "xmmintrin.h"
#define PREFETCH(a) _mm_prefetch((char *)(void *)(a),_MM_HINT_T0)
#elif defined(__GNUC__)
#define PREFETCH(a) __builtin_prefetch(a)
#else
#define PREFETCH(a) a
#endif
#else 
#define PREFETCH(a) a
#endif

#define HSMAX(x,y) ( (x) > (y) ? (x) : (y) )
#define HSMIN(x,y) ( (x) < (y) ? (x) : (y) )
#define INIT_MEM(x) memset(&(x), 0, sizeof((x)))

/* comparison function for the toplist */
static int smallerHough(const void *a,const void *b) {
  SemiCohCandidate a1, b1;
  a1 = *((const SemiCohCandidate *)a);
  b1 = *((const SemiCohCandidate *)b);
  
  if( a1.significance < b1.significance )
    return(1);
  else if( a1.significance > b1.significance)
    return(-1);
  else
    return(0);
}

/* we point nearly all LocalHOUGH functions back to LAL */

#define LocalHOUGHComputeSizePar     LALHOUGHComputeSizePar
#define LocalHOUGHFillPatchGrid      LALHOUGHFillPatchGrid
#define LocalHOUGHParamPLUT          LALHOUGHParamPLUT
#define LocalHOUGHConstructPLUT      LALHOUGHConstructPLUT
#define LocalHOUGHConstructSpacePHMD LALHOUGHConstructSpacePHMD
#define LocalHOUGHWeighSpacePHMD     LALHOUGHWeighSpacePHMD
#define LocalHOUGHInitializeHT       LALHOUGHInitializeHT
#define LocalHOUGHupdateSpacePHMDup  LALHOUGHupdateSpacePHMDup
#define LocalHOUGHWeighSpacePHMD     LALHOUGHWeighSpacePHMD

/* possibly optimized local copies of LALHOUGH functions */

void
LocalHOUGHConstructHMT_W  (LALStatus                  *status, 
			   HOUGHMapTotal              *ht     , /**< The output hough map */
			   UINT8FrequencyIndexVector  *freqInd, /**< time-frequency trajectory */ 
			   PHMDVectorSequence         *phmdVS); /**< set of partial hough map derivatives */

void
LocalHOUGHAddPHMD2HD_W    (LALStatus      *status, /**< the status pointer */
			   HOUGHMapDeriv  *hd,     /**< the Hough map derivative */
			   HOUGHphmd      *phmd);  /**< info from a partial map */ 

inline void
LocalHOUGHAddPHMD2HD_Wlr  (LALStatus*    status,
			   HoughDT*      map,
			   HOUGHBorder** pBorderP,
			   UINT2         length,
			   HoughDT       weight,
			   UINT2         xSide, 
			   UINT2         ySide);

void
LocalComputeFstatHoughMap (LALStatus            *status,
			   SemiCohCandidateList *out,    /* output candidates */
			   HOUGHPeakGramVector  *pgV,    /* peakgram vector */
			   SemiCoherentParams   *params)
{
  /* hough structures */
  HOUGHMapTotal ht;
  HOUGHptfLUTVector   lutV; /* the Look Up Table vector*/
  PHMDVectorSequence  phmdVS;  /* the partial Hough map derivatives */
  UINT8FrequencyIndexVector freqInd; /* for trajectory in time-freq plane */
  HOUGHResolutionPar parRes;   /* patch grid information */
  HOUGHPatchGrid  patch;   /* Patch description */ 
  HOUGHParamPLUT  parLut;  /* parameters needed to build lut  */
  HOUGHDemodPar   parDem;  /* demodulation parameters */
  HOUGHSizePar    parSize; 

  UINT2  xSide, ySide, maxNBins, maxNBorders;
  INT8  fBinIni, fBinFin, fBin;
  INT4  iHmap, nfdot;
  UINT4 k, nStacks ;
  REAL8 deltaF, dfdot, alpha, delta;
  REAL8 patchSizeX, patchSizeY;
  REAL8VectorSequence *vel, *pos;
  REAL8 fdot, refTime;
  LIGOTimeGPS refTimeGPS;
  LIGOTimeGPSVector   *tsMid;
  REAL8Vector *timeDiffV=NULL;

  toplist_t *houghToplist;

  INITSTATUS( status, "LocalComputeFstatHoughMap", rcsid );
  ATTATCHSTATUSPTR (status);

  /* check input is not null */
  if ( out == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  
  if ( out->length == 0 ) {
    ABORT ( status, HIERARCHICALSEARCH_EVAL, HIERARCHICALSEARCH_MSGEVAL );
  }  
  if ( out->list == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_EVAL, HIERARCHICALSEARCH_MSGEVAL );
  }  
  if ( pgV == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  
  if ( pgV->length == 0 ) {
    ABORT ( status, HIERARCHICALSEARCH_EVAL, HIERARCHICALSEARCH_MSGEVAL );
  }  
  if ( pgV->pg == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  
  if ( params == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  



  /* copy some parameters from peakgram vector */
  deltaF = pgV->pg->deltaF;
  nStacks = pgV->length;
  fBinIni = pgV->pg[0].fBinIni;
  fBinFin = pgV->pg[0].fBinFin;

  /* copy some params to local variables */
  nfdot = params->nfdot;
  dfdot = params->dfdot;
  alpha = params->alpha;
  delta = params->delta;
  vel = params->vel;
  pos = params->pos;
  fdot = params->fdot;
  tsMid = params->tsMid;
  refTimeGPS = params->refTime;  
  TRY ( LALGPStoFloat( status->statusPtr, &refTime, &refTimeGPS), status);

  /* set patch size */
  /* this is supposed to be the "educated guess" 
     delta theta = 1.0 / (Tcoh * f0 * Vepi )
     where Tcoh is coherent time baseline, 
     f0 is frequency and Vepi is rotational velocity 
     of detector */
  patchSizeX = params->patchSizeX;
  patchSizeY = params->patchSizeY;

  /* calculate time differences from start of observation time for each stack*/
  TRY( LALDCreateVector( status->statusPtr, &timeDiffV, nStacks), status);
  
  for (k=0; k<nStacks; k++) {
    REAL8 tMidStack;

    TRY ( LALGPStoFloat ( status->statusPtr, &tMidStack, tsMid->data + k), status);
    timeDiffV->data[k] = tMidStack - refTime;
  }



  /*--------------- first memory allocation --------------*/
  /* look up table vector */
  lutV.length = nStacks;
  lutV.lut = NULL;
  lutV.lut = (HOUGHptfLUT *)LALCalloc(1,nStacks*sizeof(HOUGHptfLUT));
  if ( lutV.lut == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  


  /* partial hough map derivative vector */
  phmdVS.length  = nStacks;

  {
    REAL8 maxTimeDiff, startTimeDiff, endTimeDiff;

    startTimeDiff = fabs(timeDiffV->data[0]);
    endTimeDiff = fabs(timeDiffV->data[timeDiffV->length - 1]);
    maxTimeDiff = HSMAX( startTimeDiff, endTimeDiff);

    /* set number of freq. bins for which LUTs will be calculated */
    /* this sets the range of residual spindowns values */
    /* phmdVS.nfSize  = 2*nfdotBy2 + 1; */
    phmdVS.nfSize  = 2 * floor((nfdot-1) * (REAL4)(dfdot * maxTimeDiff / deltaF) + 0.5f) + 1; 
  }

  phmdVS.deltaF  = deltaF;
  phmdVS.phmd = NULL;
  phmdVS.phmd=(HOUGHphmd *)LALCalloc( 1,phmdVS.length * phmdVS.nfSize *sizeof(HOUGHphmd));
  if ( phmdVS.phmd == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }    

  /* residual spindown trajectory */
  freqInd.deltaF = deltaF;
  freqInd.length = nStacks;
  freqInd.data = NULL;
  freqInd.data =  ( UINT8 *)LALCalloc(1,nStacks*sizeof(UINT8));
  if ( freqInd.data == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  
   
  /* resolution in space of residual spindowns */
  ht.dFdot.length = 1;
  ht.dFdot.data = NULL;
  ht.dFdot.data = (REAL8 *)LALCalloc( 1, ht.dFdot.length * sizeof(REAL8));
  if ( ht.dFdot.data == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  

  /* the residual spindowns */
  ht.spinRes.length = 1;
  ht.spinRes.data = NULL;
  ht.spinRes.data = (REAL8 *)LALCalloc( 1, ht.spinRes.length*sizeof(REAL8));
  if ( ht.spinRes.data == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  

  /* the residual spindowns */
  ht.spinDem.length = 1;
  ht.spinDem.data = NULL;
  ht.spinDem.data = (REAL8 *)LALCalloc( 1, ht.spinRes.length*sizeof(REAL8));
  if ( ht.spinDem.data == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  

  /* the demodulation params */
  parDem.deltaF = deltaF;
  parDem.skyPatch.alpha = alpha;
  parDem.skyPatch.delta = delta;
  parDem.spin.length = 1;
  parDem.spin.data = NULL;
  parDem.spin.data = (REAL8 *)LALCalloc(1, sizeof(REAL8));
  if ( parDem.spin.data == NULL ) {
    ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
  }  
  parDem.spin.data[0] = fdot;

  /* the skygrid resolution params */
  parRes.deltaF = deltaF;
  parRes.patchSkySizeX  = patchSizeX;
  parRes.patchSkySizeY  = patchSizeY;
  parRes.pixelFactor = params->pixelFactor;
  parRes.pixErr = PIXERR;
  parRes.linErr = LINERR;
  parRes.vTotC = VTOT;
 
  /* adjust fBinIni and fBinFin to take maxNBins into account */
  /* and make sure that we have fstat values for sufficient number of bins */
  parRes.f0Bin =  fBinIni;      

  fBinIni += params->extraBinsFstat;
  fBinFin -= params->extraBinsFstat;
  /* this is not very clean -- the Fstat calculation has to know how many extra bins are needed */

  LogPrintf(LOG_DETAIL, "Freq. range analyzed by Hough = [%fHz - %fHz] (%d bins)\n", 
	    fBinIni*deltaF, fBinFin*deltaF, fBinFin - fBinIni + 1);
  ASSERT ( fBinIni < fBinFin, status, HIERARCHICALSEARCH_EVAL, HIERARCHICALSEARCH_MSGEVAL );

  /* initialise number of candidates -- this means that any previous candidates 
     stored in the list will be lost for all practical purposes*/
  out->nCandidates = 0; 
  
  /* create toplist of candidates */
  if (params->useToplist) {
    create_toplist(&houghToplist, out->length, sizeof(SemiCohCandidate), smallerHough);
  }
  else { 
    /* if no toplist then use number of hough maps */
    INT4 numHmaps = (fBinFin - fBinIni + 1)*phmdVS.nfSize;
    if (out->length != numHmaps) {
      out->length = numHmaps;
      out->list = (SemiCohCandidate *)LALRealloc( out->list, out->length * sizeof(SemiCohCandidate));
      if ( out->list == NULL ) {
	ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
      }  
    }
  }

  /*------------------ start main Hough calculation ---------------------*/

  /* initialization */  
  fBin= fBinIni; /* initial search bin */
  iHmap = 0; /* hough map index */

  while( fBin <= fBinFin ){
    INT8 fBinSearch, fBinSearchMax;
    UINT4 i,j; 
    	
    parRes.f0Bin =  fBin;      
    TRY( LocalHOUGHComputeSizePar( status->statusPtr, &parSize, &parRes ),  status );
    xSide = parSize.xSide;
    ySide = parSize.ySide;

    maxNBins = parSize.maxNBins;
    maxNBorders = parSize.maxNBorders;
	
    /*------------------ create patch grid at fBin ----------------------*/
    patch.xSide = xSide;
    patch.ySide = ySide;
    patch.xCoor = NULL;
    patch.yCoor = NULL;
    patch.xCoor = (REAL8 *)LALCalloc(1,xSide*sizeof(REAL8));
    if ( patch.xCoor == NULL ) {
      ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
    }  

    patch.yCoor = (REAL8 *)LALCalloc(1,ySide*sizeof(REAL8));
    if ( patch.yCoor == NULL ) {
      ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
    }  
    TRY( LocalHOUGHFillPatchGrid( status->statusPtr, &patch, &parSize ), status );
    
    /*------------- other memory allocation and settings----------------- */
    for(j=0; j<lutV.length; ++j){
      lutV.lut[j].maxNBins = maxNBins;
      lutV.lut[j].maxNBorders = maxNBorders;
      lutV.lut[j].border = (HOUGHBorder *)LALCalloc(1,maxNBorders*sizeof(HOUGHBorder));
      if ( lutV.lut[j].border == NULL ) {
	ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
      }  

      lutV.lut[j].bin =	(HOUGHBin2Border *)LALCalloc(1,maxNBins*sizeof(HOUGHBin2Border));
      if ( lutV.lut[j].bin == NULL ) {
	ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
      }  

      for (i=0; i<maxNBorders; ++i){
	lutV.lut[j].border[i].ySide = ySide;
	lutV.lut[j].border[i].xPixel = (COORType *)LALCalloc(1,ySide*sizeof(COORType));
	if ( lutV.lut[j].border[i].xPixel == NULL ) {
	  ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
	}  
      }
    }

    for(j = 0; j < phmdVS.length * phmdVS.nfSize; ++j){
      phmdVS.phmd[j].maxNBorders = maxNBorders;
      phmdVS.phmd[j].leftBorderP = (HOUGHBorder **)LALCalloc(1,maxNBorders*sizeof(HOUGHBorder *));
      if ( phmdVS.phmd[j].leftBorderP == NULL ) {
	ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
      }  

      phmdVS.phmd[j].rightBorderP = (HOUGHBorder **)LALCalloc(1,maxNBorders*sizeof(HOUGHBorder *));
      if ( phmdVS.phmd[j].rightBorderP == NULL ) {
	ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
      }  

      phmdVS.phmd[j].ySide = ySide;
      phmdVS.phmd[j].firstColumn = NULL;
      phmdVS.phmd[j].firstColumn = (UCHAR *)LALCalloc(1,ySide*sizeof(UCHAR));
      if ( phmdVS.phmd[j].firstColumn == NULL ) {
	ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
      }  
    }
    
    /*------------------- create all the LUTs at fBin ---------------------*/  
    for (j=0; j < (UINT4)nStacks; j++){  /* create all the LUTs */
      parDem.veloC.x = vel->data[3*j];
      parDem.veloC.y = vel->data[3*j + 1];
      parDem.veloC.z = vel->data[3*j + 2];      
      parDem.positC.x = pos->data[3*j];
      parDem.positC.y = pos->data[3*j + 1];
      parDem.positC.z = pos->data[3*j + 2];
      parDem.timeDiff = timeDiffV->data[j];

      /* calculate parameters needed for buiding the LUT */
      TRY( LocalHOUGHParamPLUT( status->statusPtr, &parLut, &parSize, &parDem), status);

      /* build the LUT */
      TRY( LocalHOUGHConstructPLUT( status->statusPtr, &(lutV.lut[j]), &patch, &parLut ), status);

    }
    

    
    /*--------- build the set of  PHMD centered around fBin -------------*/     
    phmdVS.fBinMin = fBin - phmdVS.nfSize/2;
    TRY( LocalHOUGHConstructSpacePHMD(status->statusPtr, &phmdVS, pgV, &lutV), status );
    TRY( LocalHOUGHWeighSpacePHMD(status->statusPtr, &phmdVS, params->weightsV), status);
    
    /*-------------- initializing the Total Hough map space ------------*/   
    ht.xSide = xSide;
    ht.ySide = ySide;
    ht.skyPatch.alpha = alpha;
    ht.skyPatch.delta = delta;
    ht.mObsCoh = nStacks;
    ht.deltaF = deltaF;
    ht.spinDem.data[0] = fdot;
    ht.patchSizeX = patchSizeX;
    ht.patchSizeY = patchSizeY;
    ht.dFdot.data[0] = dfdot;
    ht.map   = NULL;
    ht.map   = (HoughTT *)LALCalloc(1,xSide*ySide*sizeof(HoughTT));
    if ( ht.map == NULL ) {
      ABORT ( status, HIERARCHICALSEARCH_ENULL, HIERARCHICALSEARCH_MSGENULL );
    }  

    TRY( LocalHOUGHInitializeHT( status->statusPtr, &ht, &patch), status); /*not needed */
    
    /*  Search frequency interval possible using the same LUTs */
    fBinSearch = fBin;
    fBinSearchMax = fBin + parSize.nFreqValid - 1;
     
    /* Study all possible frequencies with one set of LUT */    
    while ( (fBinSearch <= fBinFin) && (fBinSearch < fBinSearchMax) )  { 

      /* finally we can construct the hough maps and select candidates */
      {
	INT4   n, nfdotBy2;

	nfdotBy2 = nfdot/2;
	ht.f0Bin = fBinSearch;

	/*loop over all values of residual spindown */
	/* check limits of loop */
	for( n = -nfdotBy2; n <= nfdotBy2 ; n++ ){ 

	  ht.spinRes.data[0] =  n*dfdot; 
	  
	  for (j=0; j < (UINT4)nStacks; j++) {
	    freqInd.data[j] = fBinSearch + floor( (REAL4)(timeDiffV->data[j]*n*dfdot/deltaF) + 0.5f);
	  }

	  TRY( LocalHOUGHConstructHMT_W(status->statusPtr, &ht, &freqInd, &phmdVS),status );

	  /* get candidates */
	  if ( params->useToplist ) {
	    TRY(GetHoughCandidates_toplist( status->statusPtr, houghToplist, &ht, &patch, &parDem), status);
	  }
	  else {
	    TRY(GetHoughCandidates_threshold( status->statusPtr, out, &ht, &patch, &parDem, params->threshold), status);
	  }
	  
	  /* increment hough map index */ 	  
	  ++iHmap;
	  
	} /* end loop over spindown trajectories */

      } /* end of block for calculating total hough maps */
      

      /*------ shift the search freq. & PHMD structure 1 freq.bin -------*/
      ++fBinSearch;
      TRY( LocalHOUGHupdateSpacePHMDup(status->statusPtr, &phmdVS, pgV, &lutV), status );
      TRY( LocalHOUGHWeighSpacePHMD(status->statusPtr, &phmdVS, params->weightsV), status);      

    }   /* closing while loop over fBinSearch */

#ifdef OUTPUT_TIMING
    /* printf ("xside x yside = %d x %d = %d\n", parSize.xSide, parSize.ySide, parSize.xSide * parSize.ySide ); */
    nSkyRefine = parSize.xSide * parSize.ySide;
#endif
    
    fBin = fBinSearch;
    
    /*--------------  Free partial memory -----------------*/
    LALFree(patch.xCoor);
    LALFree(patch.yCoor);
    LALFree(ht.map);

    for (j=0; j<lutV.length ; ++j){
      for (i=0; i<maxNBorders; ++i){
	LALFree( lutV.lut[j].border[i].xPixel);
      }
      LALFree( lutV.lut[j].border);
      LALFree( lutV.lut[j].bin);
    }
    for(j=0; j<phmdVS.length * phmdVS.nfSize; ++j){
      LALFree( phmdVS.phmd[j].leftBorderP);
      LALFree( phmdVS.phmd[j].rightBorderP);
      LALFree( phmdVS.phmd[j].firstColumn);
    }
    
  } /* closing first while */

  
  /* free remaining memory */
  LALFree(ht.spinRes.data);
  LALFree(ht.spinDem.data);
  LALFree(ht.dFdot.data);
  LALFree(lutV.lut);
  LALFree(phmdVS.phmd);
  LALFree(freqInd.data);
  LALFree(parDem.spin.data);

  TRY( LALDDestroyVector( status->statusPtr, &timeDiffV), status);

  /* copy toplist candidates to output structure if necessary */
  if ( params->useToplist ) {
    for ( k=0; k<houghToplist->elems; k++) {
      out->list[k] = *((SemiCohCandidate *)(toplist_elem(houghToplist, k)));
    }
    out->nCandidates = houghToplist->elems;
    free_toplist(&houghToplist);
  }

  DETATCHSTATUSPTR (status);
  RETURN(status);

}



/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/** Adds a hough map derivative into a total hough map derivative taking into
    account the weight of the partial hough map */
/* *******************************  <lalVerbatim file="HoughMapD"> */
void LocalHOUGHAddPHMD2HD__W (LALStatus      *status, /**< the status pointer */
			      HOUGHMapDeriv  *hd,  /**< the Hough map derivative */
			      HOUGHphmd      *phmd) /**< info from a partial map */ 
{ /*   *********************************************  </lalVerbatim> */

  INT4     k,j;
  INT4     yLower, yUpper;
  UINT4    lengthLeft,lengthRight, xSide,ySide,xSideP1,xSideP1_2,xSideP1_3;
  COORType     *xPixel;
  HOUGHBorder  *borderP;
  HoughDT    weight;
  register HoughDT    tempM0,tempM1,tempM2,tempM3;
  INT4       sidx,sidx0,sidx1,sidx2,sidx3,sidxBase, sidxBase_n; /* pre-calcuted array index for sanity check */
  INT4	     c_c, c_n ,offs;

  HoughDT  *map_pointer;
  HoughDT  *pf_addr[8]; 



   /* --------------------------------------------- */
  INITSTATUS (status, "LocalHOUGHAddPHMD2HD__W", rcsid);
  ATTATCHSTATUSPTR (status); 

  /*   Make sure the arguments are not NULL: */ 
  ASSERT (hd,   status, HOUGHMAPH_ENULL, HOUGHMAPH_MSGENULL);
  ASSERT (phmd, status, HOUGHMAPH_ENULL, HOUGHMAPH_MSGENULL);

  PREFETCH(phmd->leftBorderP);
  PREFETCH(phmd->rightBorderP);


  /* -------------------------------------------   */
  /* Make sure the map contains some pixels */
  ASSERT (hd->xSide, status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
  ASSERT (hd->ySide, status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);

  weight = phmd->weight;

  xSide = hd->xSide;
  ySide = hd->ySide;
  xSideP1=xSide+1;
  xSideP1_2=xSideP1+xSideP1;
  xSideP1_3=xSideP1_2+xSideP1;
  map_pointer = &( hd->map[0]);

  lengthLeft = phmd->lengthLeft;
  lengthRight= phmd->lengthRight;
  

  if(lengthLeft > 0) {
      borderP = phmd->leftBorderP[0];
#if EAH_HOUGH_PREFETCH > EAH_HOUGH_PREFETCH_NONE
      PREFETCH(&(borderP->xPixel[borderP->yLower]));
#endif
  }	

  if(lengthRight > 0) {
      borderP = phmd->rightBorderP[0];
#if EAH_HOUGH_PREFETCH > EAH_HOUGH_PREFETCH_NONE
      PREFETCH(&(borderP->xPixel[borderP->yLower]));
#endif
  }	
  
  /* first column correction */
  for ( k=0; k< ySide; ++k ){
    map_pointer[k*(xSide+1) + 0] += phmd->firstColumn[k] * weight;
  }


  /* left borders =>  increase according to weight*/
  for (k=0; k< lengthLeft; ++k){

    /*  Make sure the arguments are not NULL: (Commented for performance) */ 
    /*  ASSERT (phmd->leftBorderP[k], status, HOUGHMAPH_ENULL,
	HOUGHMAPH_MSGENULL); */

    borderP = phmd->leftBorderP[k];
    xPixel =  &( (*borderP).xPixel[0] );

    yLower = (*borderP).yLower;
    yUpper = (*borderP).yUpper;

#if EAH_HOUGH_PREFETCH > EAH_HOUGH_PREFETCH_NONE
    if(k < lengthLeft-1) {
	INT4 ylkp1 = phmd->leftBorderP[k+1]->yLower;
	PREFETCH(&(phmd->leftBorderP[k+1]->xPixel[ylkp1]));
    } 	

    if(k < lengthLeft-2) {
	PREFETCH(phmd->leftBorderP[k+2]);
    } 	
#endif

   
    if (yLower < 0) {
      fprintf(stderr,"WARNING: Fixing yLower (%d -> 0) [HoughMap.c %d]\n",
	      yLower, __LINE__);
      yLower = 0;
    }
    if (yUpper >= ySide) {
      fprintf(stderr,"WARNING: Fixing yUpper (%d -> %d) [HoughMap.c %d]\n",
	      yUpper, ySide-1, __LINE__);
      yUpper = ySide - 1;
    }


#if EAH_HOUGH_ASS == EAH_HOUGH_ASS_X87


#ifdef __GNUC__

/* don't clobber ebx , used for PIC on Mac OS */

__asm __volatile (
	"push %%ebx				\n\t"
	"mov %[xPixel], %%eax  			\n\t"
	"mov %[yLower], %%ebx  			\n\t"
	"lea (%%eax,%%ebx,0x2), %%esi  		\n\t"
	"mov %[xSideP1], %%edx   		\n\t"

	"mov %[yUpper] , %%edi  		\n\t"
	"lea -0x2(%%eax,%%edi,0x2),%%eax  	\n\t"
	
	"mov %[map] , %%edi  			\n\t"
	"mov %%ebx,%%ecx  			\n\t"
	"imul %%edx, %%ecx  			\n\t"	
	"lea (%%edi, %%ecx, 0x8), %%edi  	\n\t"
	"fldl %[w]  				\n\t"

	"cmp  %%eax,%%esi  			\n\t"
	"jmp  2f 				\n\t"

	AD_ALIGN32                             "\n"
	"1:  					\n\t"
	"movzwl (%%esi),%%ebx			\n\t"
	"movzwl 2(%%esi),%%ecx			\n\t"
		
	"lea (%%edi, %%ebx, 0x8) , %%ebx  	\n\t"
	"fldl (%%ebx)  				\n\t"	
	"lea (%%edi,%%edx,0x8) , %%edi  	\n\t"
	"lea (%%edi,%%ecx,0x8) , %%ecx   	\n\t"
	"fldl (%%ecx)  				\n\t"

	"fxch %%st(1)   			\n\t"
	"fadd %%st(2),%%st  			\n\t"
	"fstpl (%%ebx)  			\n\t"
	"fadd %%st(1),%%st	  		\n\t"	
	"fstpl (%%ecx)  			\n\t"
	"lea (%%edi,%%edx,0x8), %%edi   	\n\t"	

	"lea 4(%%esi) , %%esi   		\n\t"
	"cmp  %%eax,%%esi       		\n"

	"2:	  				\n\t"
	"jbe 1b	  				\n\t"
	"add $0x2,%%eax				\n\t"
	"cmp %%eax,%%esi			\n\t"
	"jne 3f  				\n\t"

	"movzwl (%%esi) , %%ebx  		\n\t"
	"lea (%%edi, %%ebx, 0x8) , %%ebx  	\n\t"
	"fldl (%%ebx)  				\n\t"
	"fadd %%st(1),%%st  			\n\t"
	"fstpl (%%ebx)  			\n"
	
	"3:  					\n\t"
	"fstp %%st  				\n\t"
	"pop %%ebx				\n\t"

	: 
	:
	[xPixel]  "m" (xPixel) ,
	[yLower]  "m" (yLower) ,
	[yUpper]  "m" (yUpper),
	[xSideP1] "m" (xSideP1) ,
	[map]     "m" (map_pointer) ,
	[w]       "m" (weight)
	:
	"eax", "ecx", "edx", "esi", "edi", "cc",
	"st","st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)" 
	);

#else 

     _asm{
	push     ebx 
	mov      eax, xPixel
	mov      ebx, yLower
	lea      esi, DWORD PTR [eax+ebx*2]
	mov      edx, xSideP1
	mov      edi, yUpper
	lea      eax, DWORD PTR [eax+edi*2-2]
	mov      edi, map_pointer
	mov      ecx, ebx
	imul     ecx, edx
	lea      edi, DWORD PTR [edi+ecx*8]
	fld      QWORD PTR weight
	cmp      esi, eax
	jmp      l1_a
	
		ALIGN 16 
	
	l2_a:

	movzx    ebx, WORD PTR [esi]
	movzx    ecx, WORD PTR [esi+0x2]
	lea      ebx, DWORD PTR [edi+ebx*8]
	fld      QWORD PTR [ebx]
	lea      edi, DWORD PTR [edi+edx*8]
	lea      ecx, DWORD PTR [edi+ecx*8]
	fld      QWORD PTR [ecx]
	fxch     st(1)
	fadd     st(0), st(2)
	fstp     QWORD PTR [ebx]
	fadd     st(0), st(1)
	fstp     QWORD PTR [ecx]
	lea      edi, DWORD PTR [edi+edx*8]
	lea      esi, DWORD PTR [esi+0x4]
	cmp      esi, eax
	
	l1_a:
	
	jbe      l2_a
	
	add      eax, 0x2
	cmp      esi, eax
	jnz      l_end_a
	movzx    ebx, WORD PTR [esi]
	lea      ebx, DWORD PTR [edi+ebx*8]
	fld      QWORD PTR [ebx]
	fadd     st(0), st(1)
	fstp     QWORD PTR [ebx]

	l_end_a:    

	fstp     st(0)
	pop      ebx 
	};
#endif

#elif defined(EAH_HOUGH_BATCHSIZELD)

    sidxBase=yLower*xSideP1;
    sidxBase_n = sidxBase+(xSideP1 << EAH_HOUGH_BATCHSIZELD);
    
    /* fill first cache entries */
    c_c =0;
    c_n =EAH_HOUGH_BATCHSIZE;

    offs = yUpper - yLower+1;
    if (offs > EAH_HOUGH_BATCHSIZE) {
	offs = EAH_HOUGH_BATCHSIZE; 
    }	
	
    	
    for(j=yLower; j < yLower+offs; j++) {
        PREFETCH(pf_addr[c_c++] = map_pointer + xPixel[j] + j*xSideP1);		
#ifndef LAL_NDEBUG       
      	sidx0=xPixel[j]+ j*xSideP1;
        if ((sidx0 < 0) || (sidx0 >= ySide*(xSide+1)) || xPixel[j] < 0 || xPixel[j] >= xSideP1 ) {
  	  fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
	  	  __FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j],xSide );
	  ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
        }
#endif	
    }		
		
    c_c=0;
    for(j=yLower; j<=yUpper-(2*EAH_HOUGH_BATCHSIZE-1);j+=EAH_HOUGH_BATCHSIZE){

      	
      sidx0 = xPixel[j+EAH_HOUGH_BATCHSIZE]+sidxBase_n;; 
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      sidx1 = xPixel[j+(EAH_HOUGH_BATCHSIZE+1)]+sidxBase_n+xSideP1;
#endif
#if (EAH_HOUGH_BATCHSIZE == 4)
      sidx2 = xPixel[j+(EAH_HOUGH_BATCHSIZE+2)]+sidxBase_n+xSideP1_2;
      sidx3 = xPixel[j+(EAH_HOUGH_BATCHSIZE+3)]+sidxBase_n+xSideP1_3;;
#endif
	
      PREFETCH(xPixel +(j+(EAH_HOUGH_BATCHSIZE+EAH_HOUGH_BATCHSIZE)));

      PREFETCH(pf_addr[c_n] = map_pointer+sidx0);
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      PREFETCH(pf_addr[c_n+1] = map_pointer+sidx1);
#endif
#if (EAH_HOUGH_BATCHSIZE == 4)
      PREFETCH(pf_addr[c_n+2] = map_pointer+sidx2);
      PREFETCH(pf_addr[c_n+3] = map_pointer+sidx3);
#endif

#ifndef LAL_NDEBUG 
      if ((sidx0 < 0) || (sidx0 >= ySide*(xSide+1))|| xPixel[j+EAH_HOUGH_BATCHSIZE] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE] >= xSideP1) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE,xPixel[j+EAH_HOUGH_BATCHSIZE],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      if ((sidx1 < 0) || (sidx1 >= ySide*(xSide+1))|| xPixel[j+EAH_HOUGH_BATCHSIZE+1] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE+1] >= xSideP1) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE+1,xPixel[j+EAH_HOUGH_BATCHSIZE+1],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#endif
#if (EAH_HOUGH_BATCHSIZE == 4)
      if ((sidx2 < 0) || (sidx2 >= ySide*(xSide+1))|| xPixel[j+EAH_HOUGH_BATCHSIZE+2] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE+2] >= xSideP1) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx2,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE+2,xPixel[j+EAH_HOUGH_BATCHSIZE+2],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
      if ((sidx3 < 0) || (sidx3 >= ySide*(xSide+1))|| xPixel[j+EAH_HOUGH_BATCHSIZE+3] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE+3] >= xSideP1) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx3,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE+3,xPixel[j+EAH_HOUGH_BATCHSIZE+3],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#endif
#endif 

      tempM0 = *(pf_addr[c_c]) +weight;
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      tempM1 = *(pf_addr[c_c+1]) +weight;
#endif
#if (EAH_HOUGH_BATCHSIZE == 4)
      tempM2 = *(pf_addr[c_c+2]) +weight;
      tempM3 = *(pf_addr[c_c+3]) +weight;
#endif
      sidxBase = sidxBase_n;
      sidxBase_n+=xSideP1 << EAH_HOUGH_BATCHSIZE_LOG2;
      
      (*(pf_addr[c_c]))=tempM0;
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      (*(pf_addr[c_c+1]))=tempM1;
#endif
#if (EAH_HOUGH_BATCHSIZE == 4)
      (*(pf_addr[c_c+2]))=tempM2;
      (*(pf_addr[c_c+3]))=tempM3;
#endif 

      c_c ^= EAH_HOUGH_BATCHSIZE;
      c_n ^= EAH_HOUGH_BATCHSIZE;
    }

    sidxBase=j*xSideP1;
    for(; j<=yUpper;++j){
      sidx = sidxBase + xPixel[j];
#ifndef LAL_NDEBUG
      if ((sidx < 0) || (sidx >= ySide*(xSide+1)) || xPixel[j] < 0 || xPixel[j] >= xSideP1) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j],xSide );
 
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#endif
      map_pointer[sidx] += weight;
      sidxBase+=xSideP1;
    }


#else
    for(j=yLower; j<=yUpper;++j){
      sidx = j *(xSide+1) + xPixel[j];
      if ((sidx < 0) || (sidx >= ySide*(xSide+1))) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j] );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
      map_pointer[sidx] += weight;
    }

#endif

  }

  /* right borders => decrease according to weight*/
  for (k=0; k< lengthRight; ++k){
  
    /*  Make sure the arguments are not NULL: (Commented for performance) */ 
    /*  ASSERT (phmd->rightBorderP[k], status, HOUGHMAPH_ENULL,
	HOUGHMAPH_MSGENULL); */

    borderP = phmd->rightBorderP[k];
  	
    yLower = (*borderP).yLower;
    yUpper = (*borderP).yUpper;
    xPixel =  &( (*borderP).xPixel[0] );

#if EAH_HOUGH_PREFETCH > EAH_HOUGH_PREFETCH_NONE
    if(k < lengthRight-1) {
	INT4 ylkp1 = phmd->rightBorderP[k+1]->yLower;
	PREFETCH(&(phmd->rightBorderP[k+1]->xPixel[ylkp1]));
    } 	

    if(k < lengthRight-2) {
	PREFETCH(phmd->rightBorderP[k+2]);
    } 	
#endif
   
    if (yLower < 0) {
      fprintf(stderr,"WARNING: Fixing yLower (%d -> 0) [HoughMap.c %d]\n",
	      yLower, __LINE__);
      yLower = 0;
    }
    if (yUpper >= ySide) {
      fprintf(stderr,"WARNING: Fixing yUpper (%d -> %d) [HoughMap.c %d]\n",
	      yUpper, ySide-1, __LINE__);
      yUpper = ySide - 1;
    }


#if EAH_HOUGH_ASS == EAH_HOUGH_ASS_X87

#ifdef __GNUC__

__asm __volatile (
	"push %%ebx				\n\t"
	"mov %[xPixel], %%eax  			\n\t"
	"mov %[yLower], %%ebx  			\n\t"
	"mov %[xSideP1], %%edx   		\n\t"
	"lea (%%eax,%%ebx,0x2), %%esi  		\n\t"

	"mov %[yUpper] , %%edi  		\n\t"
	"lea -0x2(%%eax,%%edi,0x2),%%eax  	\n\t"
	
	"mov %[map] , %%edi  			\n\t"
	"mov %%ebx,%%ecx  			\n\t"
	"imul %%edx, %%ecx  			\n\t"	
	"lea (%%edi, %%ecx, 0x8), %%edi  	\n\t"
	"fldl %[w]  				\n\t"

	"cmp  %%eax,%%esi  			\n\t"
	"jmp  2f 				\n\t"

	AD_ALIGN32                             "\n"
	"1:  					\n\t"
	"movzwl (%%esi),%%ebx			\n\t"
	"movzwl 2(%%esi),%%ecx			\n\t"

	"lea (%%edi, %%ebx, 0x8) , %%ebx  	\n\t"
	"fldl (%%ebx)  				\n\t"	
	"lea (%%edi,%%edx,0x8) , %%edi  	\n\t"
	"lea (%%edi,%%ecx,0x8) , %%ecx   	\n\t"
	"fldl (%%ecx)  				\n\t"

	"fxch %%st(1)   			\n\t"
	"fsub %%st(2),%%st  			\n\t"
	"fstpl (%%ebx)  			\n\t"
	"fsub %%st(1),%%st	  		\n\t"	
	"fstpl (%%ecx)  			\n\t"
	"lea (%%edi,%%edx,0x8), %%edi   	\n\t"	
	"lea 4(%%esi) , %%esi   		\n\t"
	"cmp  %%eax,%%esi       		\n"

	"2:	  				\n\t"
	"jbe 1b	  				\n\t"
	"add $0x2,%%eax				\n\t"
	"cmp %%eax,%%esi			\n\t"
	"jne 3f  				\n\t"

	"movzwl (%%esi) , %%ebx  		\n\t"
	"lea (%%edi, %%ebx, 0x8) , %%ebx  	\n\t"
	"fldl (%%ebx)  				\n\t"
	"fsub %%st(1),%%st  			\n\t"
	"fstpl (%%ebx)  			\n"
	
	"3:  					\n\t"	
	"fstp %%st  				\n\t"
	"pop %%ebx				\n\t"
	
	: 
	:
	[xPixel]  "m" (xPixel) ,
	[yLower]  "m" (yLower) ,
	[yUpper]  "m" (yUpper),
	[xSideP1] "m" (xSideP1) ,
	[map]     "m" (map_pointer) ,
	[w]       "m" (weight)
	:
	"eax", "ecx", "edx", "esi", "edi", "cc",
	"st","st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)" 
	);
#else


    _asm{
	push     ebx 
	mov      eax, xPixel
	mov      ebx, yLower
	lea      esi, DWORD PTR [eax+ebx*2]
	mov      edx, xSideP1
	mov      edi, yUpper
	lea      eax, DWORD PTR [eax+edi*2-2]
	mov      edi, map_pointer
	mov      ecx, ebx
	imul     ecx, edx
	lea      edi, DWORD PTR [edi+ecx*8]
	fld      QWORD PTR weight
	cmp      esi, eax
	jmp      l1_b
	
	ALIGN 16
	
	l2_b:

	movzx    ebx, WORD PTR [esi]
	movzx    ecx, WORD PTR [esi+0x2]
	lea      ebx, DWORD PTR [edi+ebx*8]
	fld      QWORD PTR [ebx]
	lea      edi, DWORD PTR [edi+edx*8]
	lea      ecx, DWORD PTR [edi+ecx*8]
	fld      QWORD PTR [ecx]
	fxch     st(1)
	fsub     st(0), st(2)
	fstp     QWORD PTR [ebx]
	fsub     st(0), st(1)
	fstp     QWORD PTR [ecx]
	lea      edi, DWORD PTR [edi+edx*8]
	lea      esi, DWORD PTR [esi+0x4]
	cmp      esi, eax

	l1_b:

	jbe      l2_b
	
	add      eax, 0x2
	cmp      esi, eax
	jnz      l_end_b
	movzx    ebx, WORD PTR [esi]
	lea      ebx, DWORD PTR [edi+ebx*8]
	fld      QWORD PTR [ebx]
	fsub     st(0), st(1)
	fstp     QWORD PTR [ebx]

	l_end_b:    

	fstp     st(0)
	pop      ebx 
    }


#endif

#elif defined(EAH_HOUGH_BATCHSIZELD)

    sidxBase=yLower*xSideP1;
    sidxBase_n = sidxBase+(xSideP1 << EAH_HOUGH_BATCHSIZELD);	
    /* fill first cache entries */	

    
    c_c =0;
    c_n =EAH_HOUGH_BATCHSIZE;

    offs = yUpper - yLower+1;
    if (offs > EAH_HOUGH_BATCHSIZE) {
	offs = EAH_HOUGH_BATCHSIZE; 
    }	
	
    	
    for(j=yLower; j < yLower+offs; j++) {
        PREFETCH(pf_addr[c_c++] = map_pointer + xPixel[j] + j*xSideP1);			
#ifndef LAL_NDEBUG       
      	sidx0=xPixel[j]+ j*xSideP1;
        if ((sidx0 < 0) || (sidx0 >= ySide*(xSide+1)) || xPixel[j] < 0 || xPixel[j] >= xSideP1 ) {
  	  fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
	  	  __FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j],xSide );
	  ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
        }
#endif
    }		
		
    c_c=0;
    for(j=yLower; j<=yUpper-(EAH_HOUGH_BATCHSIZE*2-1);j+=EAH_HOUGH_BATCHSIZE){

      	
      sidx0 = xPixel[j+EAH_HOUGH_BATCHSIZE]+sidxBase_n;; 
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      sidx1 = xPixel[j+(EAH_HOUGH_BATCHSIZE+1)]+sidxBase_n+xSideP1;
#endif
#if (EAH_HOUGH_BATCHSIZE == 4) 
      sidx2 = xPixel[j+(EAH_HOUGH_BATCHSIZE+2)]+sidxBase_n+xSideP1_2;
      sidx3 = xPixel[j+(EAH_HOUGH_BATCHSIZE+3)]+sidxBase_n+xSideP1_3;;
#endif
	
      PREFETCH(xPixel +(j+(EAH_HOUGH_BATCHSIZE+EAH_HOUGH_BATCHSIZE)));

      PREFETCH(pf_addr[c_n] = map_pointer+sidx0);
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      PREFETCH(pf_addr[c_n+1] = map_pointer+sidx1);
#endif
#if (EAH_HOUGH_BATCHSIZE == 4) 
      PREFETCH(pf_addr[c_n+2] = map_pointer+sidx2);
      PREFETCH(pf_addr[c_n+3] = map_pointer+sidx3);
#endif

#ifndef LAL_NDEBUG       
      if ((sidx0 < 0) || (sidx0 >= ySide*(xSide+1)) || xPixel[j+EAH_HOUGH_BATCHSIZE] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE] >= xSideP1 ) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE,xPixel[j+EAH_HOUGH_BATCHSIZE],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      if ((sidx1 < 0) || (sidx1 >= ySide*(xSide+1)) || xPixel[j+EAH_HOUGH_BATCHSIZE+1] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE+1] >= xSideP1 ) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE+1,xPixel[j+EAH_HOUGH_BATCHSIZE+1],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#endif
#if (EAH_HOUGH_BATCHSIZE == 4) 
      if ((sidx2 < 0) || (sidx2 >= ySide*(xSide+1)) || xPixel[j+EAH_HOUGH_BATCHSIZE+2] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE+2] >= xSideP1 ) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx2,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE+2,xPixel[j+EAH_HOUGH_BATCHSIZE+2],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
      if ((sidx3 < 0) || (sidx3 >= ySide*(xSide+1)) || xPixel[j+EAH_HOUGH_BATCHSIZE+3] < 0 || xPixel[j+EAH_HOUGH_BATCHSIZE+3] >= xSideP1) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx3,ySide*(xSide+1),j+EAH_HOUGH_BATCHSIZE+3,xPixel[j+EAH_HOUGH_BATCHSIZE+3],xSide );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#endif
#endif 

      tempM0 = *(pf_addr[c_c]) -weight;
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      tempM1 = *(pf_addr[c_c+1]) -weight;
#endif
#if (EAH_HOUGH_BATCHSIZE == 4) 
      tempM2 = *(pf_addr[c_c+2]) -weight;
      tempM3 = *(pf_addr[c_c+3]) -weight;
#endif

      sidxBase = sidxBase_n;
      sidxBase_n+=xSideP1 << EAH_HOUGH_BATCHSIZE_LOG2;
      
      (*(pf_addr[c_c]))=tempM0;
#if (EAH_HOUGH_BATCHSIZE == 4) || (EAH_HOUGH_BATCHSIZE == 2)
      (*(pf_addr[c_c+1]))=tempM1;
#endif
#if (EAH_HOUGH_BATCHSIZE == 4) 
      (*(pf_addr[c_c+2]))=tempM2;
      (*(pf_addr[c_c+3]))=tempM3;
#endif
      c_c ^= EAH_HOUGH_BATCHSIZE;
      c_n ^= EAH_HOUGH_BATCHSIZE;
    }

    sidxBase=j*xSideP1;
    for(; j<=yUpper;++j){
      sidx = sidxBase + xPixel[j];
#ifndef LAL_NDEBUG
      if ((sidx < 0) || (sidx >= ySide*(xSide+1)) || xPixel[j] < 0 || xPixel[j] >= xSideP1 ) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d xSide:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j],xSide );
  	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
#endif
      map_pointer[sidx] -= weight;
      sidxBase += xSideP1;
    }

#else

    for(j=yLower; j<=yUpper;++j){
      sidx = j*(xSide+1) + xPixel[j];
      if ((sidx < 0) || (sidx >= ySide*(xSide+1))) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j] );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
      map_pointer[sidx] -= weight;
    }
#endif

  }


  /* -------------------------------------------   */
  
  DETATCHSTATUSPTR (status);
  
  /* normal exit */
  RETURN (status);
}


/** Calculates the total hough map for a given trajectory in the 
    time-frequency plane and a set of partial hough map derivatives allowing 
    each PHMD to have a different weight factor to account for varying
    sensitivity at different sky-locations. */ 
/* *******************************  <lalVerbatim file="DriveHoughD"> */
void LocalHOUGHConstructHMT_W (LALStatus                  *status, 
			       HOUGHMapTotal              *ht, /**< The output hough map */
			       UINT8FrequencyIndexVector  *freqInd, /**< time-frequency trajectory */ 
			       PHMDVectorSequence         *phmdVS) /**< set of partial hough map derivatives */
{ /*   *********************************************  </lalVerbatim> */


  UINT4    k,j;
  UINT4    breakLine;
  UINT4    nfSize;    /* number of different frequencies */
  UINT4    length;    /* number of elements for each frequency */
  UINT8    fBinMin;   /* present minimum frequency bin */ 
  INT8     fBin;      /* present frequency bin */
  UINT2    xSide,ySide;
 
  HOUGHMapDeriv hd; /* the Hough map derivative */

  /* --------------------------------------------- */
  INITSTATUS (status, "LALHOUGHConstructHMT_W", rcsid);
  ATTATCHSTATUSPTR (status); 

  /*   Make sure the arguments are not NULL: */ 
  ASSERT (phmdVS,  status, LALHOUGHH_ENULL, LALHOUGHH_MSGENULL);
  ASSERT (ht,      status, LALHOUGHH_ENULL, LALHOUGHH_MSGENULL);
  ASSERT (freqInd, status, LALHOUGHH_ENULL, LALHOUGHH_MSGENULL);
  /* -------------------------------------------   */

  ASSERT (phmdVS->phmd,  status, LALHOUGHH_ENULL, LALHOUGHH_MSGENULL);
  ASSERT (freqInd->data, status, LALHOUGHH_ENULL, LALHOUGHH_MSGENULL);
  /* -------------------------------------------   */

  /* Make sure there is no size mismatch */
  ASSERT (freqInd->length == phmdVS->length, status, 
	  LALHOUGHH_ESZMM, LALHOUGHH_MSGESZMM);
  ASSERT (freqInd->deltaF == phmdVS->deltaF, status, 
	  LALHOUGHH_ESZMM, LALHOUGHH_MSGESZMM);
  /* -------------------------------------------   */

  /* Make sure there are elements  */
  ASSERT (phmdVS->length, status, LALHOUGHH_ESIZE, LALHOUGHH_MSGESIZE);
  ASSERT (phmdVS->nfSize, status, LALHOUGHH_ESIZE, LALHOUGHH_MSGESIZE);
  /* -------------------------------------------   */
  
   /* Make sure the ht map contains some pixels */
  ASSERT (ht->xSide, status, LALHOUGHH_ESIZE, LALHOUGHH_MSGESIZE);
  ASSERT (ht->ySide, status, LALHOUGHH_ESIZE, LALHOUGHH_MSGESIZE);

  length = phmdVS->length;
  nfSize = phmdVS->nfSize; 
  
  fBinMin = phmdVS->fBinMin; /* initial frequency value  od the cilinder*/
  
  breakLine = phmdVS->breakLine;

  /* number of physical pixels */
  xSide = ht->xSide;
  ySide = ht->ySide;
  
  /* Make sure initial breakLine is in [0,nfSize)  */
  ASSERT ( breakLine < nfSize, status, LALHOUGHH_EVAL, LALHOUGHH_MSGEVAL);
  
  /* -------------------------------------------   */
  
  /* Initializing  hd map and memory allocation */
  hd.xSide = xSide;
  hd.ySide = ySide;
  hd.map = (HoughDT *)LALMalloc(ySide*(xSide+1)*sizeof(HoughDT));
  if (hd. map == NULL) {
    ABORT( status, LALHOUGHH_EMEM, LALHOUGHH_MSGEMEM); 
  }

  /* -------------------------------------------   */
 
  TRY( LALHOUGHInitializeHD(status->statusPtr, &hd), status);
  for ( k=0; k<length; ++k ){ 
    /* read the frequency index and make sure is in the proper interval*/
    fBin =freqInd->data[k] -fBinMin;

    ASSERT ( fBin < nfSize, status, LALHOUGHH_EVAL, LALHOUGHH_MSGEVAL);
    ASSERT ( fBin >= 0,     status, LALHOUGHH_EVAL, LALHOUGHH_MSGEVAL);
 
    /* find index */
    j = (fBin + breakLine) % nfSize;

    /* Add the corresponding PHMD to HD */
    TRY( LocalHOUGHAddPHMD2HD_W(status->statusPtr,
				&hd, &(phmdVS->phmd[j*length+k]) ), status);
  }

  TRY( LALHOUGHIntegrHD2HT(status->statusPtr, ht, &hd), status);
  
  /* Free memory and exit */
  LALFree(hd.map);

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/** Adds a hough map derivative into a total hough map derivative taking into
    account the weight of the partial hough map */
/* *******************************  <lalVerbatim file="HoughMapD"> */
void LocalHOUGHAddPHMD2HD_W (LALStatus      *status, /**< the status pointer */
			     HOUGHMapDeriv  *hd,  /**< the Hough map derivative */
			     HOUGHphmd      *phmd) /**< info from a partial map */ 
{ /*   *********************************************  </lalVerbatim> */

  INT2     k;
  UINT2    xSide,ySide;
  HoughDT  weight;

  INITSTATUS (status, "LALHOUGHAddPHMD2HD_W", rcsid);
  ATTATCHSTATUSPTR (status); 

  /*   Make sure the arguments are not NULL: */ 
  ASSERT (hd,   status, HOUGHMAPH_ENULL, HOUGHMAPH_MSGENULL);
  ASSERT (phmd, status, HOUGHMAPH_ENULL, HOUGHMAPH_MSGENULL);

  /* Make sure the map contains some pixels */
  ASSERT (hd->xSide, status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
  ASSERT (hd->ySide, status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);

  /* aliases */
  weight = phmd->weight;
  xSide = hd->xSide;
  ySide = hd->ySide;
  
  /* first column correction */
  for ( k=0; k< ySide; ++k ){
    hd->map[k*(xSide+1) + 0] += phmd->firstColumn[k] * weight;
  }

  /* left borders =>  increase according to weight */
  TRY ( LocalHOUGHAddPHMD2HD_Wlr (status,
				  hd->map,
				  phmd->leftBorderP,
				  phmd->lengthLeft,
				  weight,
				  xSide,
				  ySide), status );
  
  /* right borders => decrease according to weight */
  TRY ( LocalHOUGHAddPHMD2HD_Wlr (status,
				  hd->map,
				  phmd->rightBorderP,
				  phmd->lengthRight,
				  - weight,
				  xSide,
				  ySide), status );
  
  /* cleanup */
  DETATCHSTATUSPTR (status);
  
  /* normal exit */
  RETURN (status);
}

inline void
LocalHOUGHAddPHMD2HD_Wlr (LALStatus*    status,
			  HoughDT*      map,
			  HOUGHBorder** pBorderP,
			  UINT2         length,
			  HoughDT       weight,
			  UINT2         xSide, 
			  UINT2         ySide)
{
  INT2        k,j;
  INT2        yLower, yUpper;
  COORType    *xPixel;
  HOUGHBorder *borderP;
  INT4        sidx;

  for (k=0; k< length; ++k) {

    /* local aliases */
    borderP = pBorderP[k];
    yLower = (*borderP).yLower;
    yUpper = (*borderP).yUpper;
    xPixel =  &((*borderP).xPixel[0]);
   
    /* check boundary conditions */
    if (yLower < 0) {
      fprintf(stderr,"WARNING: Fixing yLower (%d -> 0) [HoughMap.c %d]\n",
	      yLower, __LINE__);
      yLower = 0;
    }
    if (yUpper >= ySide) {
      fprintf(stderr,"WARNING: Fixing yUpper (%d -> %d) [HoughMap.c %d]\n",
	      yUpper, ySide-1, __LINE__);
      yUpper = ySide - 1;
    }

    /* increase / decrease according to weight */
    for(j = yLower; j <= yUpper; j++){
      sidx = j*(xSide+1) + xPixel[j];
      if ((sidx < 0) || (sidx >= ySide*(xSide+1))) {
	fprintf(stderr,"\nERROR: %s %d: map index out of bounds: %d [0..%d] j:%d xp[j]:%d\n",
		__FILE__,__LINE__,sidx,ySide*(xSide+1),j,xPixel[j] );
	ABORT(status, HOUGHMAPH_ESIZE, HOUGHMAPH_MSGESIZE);
      }
      map[sidx] += weight;
    }
  }
}

/*-----------------------------------------------------------------------
 *
 * File Name: SFTbin.c
 *
 * Authors: Sintes, A.M.,  Krishnan, B. & inspired from Siemens, X.
 *
 * Revision: $Id$
 *
 * History:   Created by Sintes May 21, 2003
 *            Modified by Krishnan on Feb 22, 2004
 *
 *-----------------------------------------------------------------------
 */

/************************************ <lalVerbatim file="SFTbinCV">
Author: Sintes, A.M., Krishnan, B.
$Id$
************************************* </lalVerbatim> */

/* <lalLaTeX>  *******************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{MOdule \texttt{SFTbin.c}}
\label{ss:SFTbin.c}
Routines for reading SFT binary files

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Prototypes}
\vspace{0.1in}
\input{SFTbinD}
\index{\verb&ReadSFTbinHeader1()&}
\index{\verb&ReadCOMPLEX8SFTbinData1()&}
\index{\verb&ReadCOMPLEX16SFTbinData1()&}
\index{\verb&COMPLEX8SFT2Periodogram1()&}
\index{\verb&COMPLEX16SFT2Periodogram1()&}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Description}

the output of the periodogram should be properly normalized !!!


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Uses}
\begin{verbatim}
LALHO()
\end{verbatim}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Notes}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\vfill{\footnotesize\input{SFTbinCV}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

*********************************************** </lalLaTeX> */


#include "./SFTbin.h"

NRCSID (SFTBINC, "$Id$");

/*
 * The functions that make up the guts of this module
 */


/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */

/* *******************************  <lalVerbatim file="SFTbinD"> */
void ReadSFTbinHeader1 (LALStatus  *status,
                     SFTHeader1   *header,
                     CHAR          *fname)
{ /*   *********************************************  </lalVerbatim> */
  
  FILE     *fp = NULL;
  SFTHeader1  header1;
  size_t    errorcode;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "ReadSFTbinHeader1", SFTBINC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (header, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
  ASSERT (fname,  status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
  
  /* opening the SFT binary file */
  fp = fopen( fname, "r");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);
  
  /* read the header */
  errorcode = fread( (void *)&header1, sizeof(SFTHeader1), 1, fp);
  ASSERT (errorcode ==1, status, SFTBINH_EHEADER,  SFTBINH_MSGEHEADER);
  ASSERT (header1.endian ==1, status, SFTBINH_EENDIAN, SFTBINH_MSGEENDIAN);

  header->endian     = header1.endian;
  header->gpsSeconds = header1.gpsSeconds;
  header->gpsNanoSeconds = header1.gpsNanoSeconds;
  header->timeBase       = header1.timeBase;
  header->fminBinIndex   = header1.fminBinIndex ;
  header->length     = header1.length ;
  
  fclose(fp);
  
  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void ReadCOMPLEX8SFTbinData1(LALStatus  *status,
                   COMPLEX8SFTData1    *sft,
                   CHAR                *fname)
{ /*   *********************************************  </lalVerbatim> */

  FILE        *fp = NULL;
  SFTHeader1  header1;
  size_t     errorcode;
  INT4       offset;
  INT4       length;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "ReadCOMPLEX8SFTbinData1", SFTBINC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (fname, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  
  /* opening the SFT binary file */
  fp = fopen( fname, "r");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);
  
  /* read the header */
  errorcode = fread( (void *)&header1, sizeof(SFTHeader1), 1, fp);
  ASSERT (errorcode == 1,      status, SFTBINH_EHEADER, SFTBINH_MSGEHEADER);
  ASSERT (header1.endian == 1, status, SFTBINH_EENDIAN, SFTBINH_MSGEENDIAN);
  
  /* check that the data we want is in the file and it is correct */
  ASSERT (header1.timeBase > 0.0, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  sft->timeBase             = header1.timeBase;
  sft->epoch.gpsSeconds     = header1.gpsSeconds;
  sft->epoch.gpsNanoSeconds = header1.gpsNanoSeconds;
  
  offset = sft->fminBinIndex - header1.fminBinIndex;
  ASSERT (offset >= 0, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);  

  length=sft->length;
  
  if (length > 0){
    COMPLEX8  *dataIn1 = NULL;
    COMPLEX8  *dataIn;
    COMPLEX8  *dataOut;
    INT4      n;
    REAL8     factor;
    
    ASSERT (sft->data, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
    ASSERT (header1.length >= offset+length, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
    dataIn1 = (COMPLEX8 *)LALMalloc(length*sizeof(COMPLEX8) );
    /* skip offset data points and read the required amount of data */
    ASSERT (0 == fseek(fp, offset * sizeof(COMPLEX8), SEEK_CUR), status, SFTBINH_EVAL, SFTBINH_MSGEVAL); 
    errorcode = fread( (void *)dataIn1, length*sizeof(COMPLEX8), 1, fp);

    dataIn = dataIn1;
    dataOut = sft->data;
    n= length;
    
    /* is this correct  ? */
    /* or if the data is normalized properly, there will be no need to consider. */
    factor = ((double) length)/ ((double) header1.length);
    
    /* let's normalize */
    while (n-- >0){
      dataOut->re = dataIn->re*factor;
      dataOut->im = dataIn->im*factor;
      ++dataOut;
      ++dataIn;
    }    
    LALFree(dataIn1);    
  }
  
  fclose(fp);
  
  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void ReadCOMPLEX16SFTbinData1(LALStatus  *status,
                   COMPLEX16SFTData1    *sft,
                   CHAR                *fname)
{ /*   *********************************************  </lalVerbatim> */

  FILE        *fp = NULL;
  SFTHeader1  header1;
  size_t     errorcode;
  INT4       offset;
  INT4       length;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "ReadCOMPLEX16SFTbinData1", SFTBINC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (fname, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  
  /* opening the SFT binary file */
  fp = fopen( fname, "r");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);
  
  /* read the header */
  errorcode = fread( (void *)&header1, sizeof(SFTHeader1), 1, fp);
  ASSERT (errorcode == 1,      status, SFTBINH_EHEADER, SFTBINH_MSGEHEADER);
  ASSERT (header1.endian == 1, status, SFTBINH_EENDIAN, SFTBINH_MSGEENDIAN);
  
  /* check that the data we want is in the file and it is correct */
  ASSERT (header1.timeBase > 0.0, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  sft->timeBase             = header1.timeBase;
  sft->epoch.gpsSeconds     = header1.gpsSeconds;
  sft->epoch.gpsNanoSeconds = header1.gpsNanoSeconds;
  
  offset = sft->fminBinIndex - header1.fminBinIndex;
  ASSERT (offset >= 0, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  
  length=sft->length;
  
  if (length > 0){
    COMPLEX16  *dataIn1 = NULL;
    COMPLEX16  *dataIn;
    COMPLEX16  *dataOut;
    INT4      n;
    REAL8     factor;
    
    ASSERT (sft->data, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
    ASSERT (header1.length >= offset+length, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
    dataIn1 = (COMPLEX16 *)LALMalloc(length*sizeof(COMPLEX16) );
    /* skip offset data points and read the required amount of data */
    ASSERT (0 == fseek(fp, offset * sizeof(COMPLEX16), SEEK_CUR), status, SFTBINH_EVAL, SFTBINH_MSGEVAL); 
    errorcode = fread( (void *)dataIn1, length*sizeof(COMPLEX16), 1, fp);
    
    dataOut = sft->data;
    dataIn = dataIn1;
    n= length;
    
    /* is this correct  ? */
    /* or if the data is normalized properly, there will be no need to consider. */
    factor = ((double) length)/ ((double) header1.length);
    
    /* let's normalize */
    while (n-- >0){
      dataOut->re = dataIn->re*factor;
      dataOut->im = dataIn->im*factor;
      ++dataOut;
      ++dataIn;
    }    
    LALFree(dataIn1);    
  }
  
  fclose(fp);
 
  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void COMPLEX8SFT2Periodogram1 (LALStatus  *status,
                   REAL8Periodogram1     *peri,
		   COMPLEX8SFTData1      *sft)
{ /*   *********************************************  </lalVerbatim> */

  INT4       length;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "COMPLEX8SFT2Periodogram1", SFTBINC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (peri,  status, SFTBINH_ENULL, SFTBINH_MSGENULL);

  peri->epoch.gpsSeconds = sft->epoch.gpsSeconds;
  peri->epoch.gpsNanoSeconds = sft->epoch.gpsNanoSeconds;
  peri->timeBase = sft->timeBase;
  peri->fminBinIndex = sft->fminBinIndex;
  
  length = sft->length;
  ASSERT (length==peri->length,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);

  if (length > 0){
    REAL8      *out;
    COMPLEX8   *in1;
    REAL8      re,im, factor;
    INT4      n;

    ASSERT (sft->data, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
    ASSERT (peri->data, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
    out = peri->data;
    in1 = sft->data;
    n= length;

    /* if data was properly normalized */
    factor = 1./peri->timeBase;
    
    /* if data is not normalized... this factor needs to be clarified  */
    /* note inconsistency with the previous function and quantity *factor*  there */
    
    /* if bin zero is included should be treated properly because of factor 2 */
    
    while (n-- >0){
      re = in1->re;
      im = in1->im;
      *out = (re*re + im*im) *factor; /* factor 2 still missing if one-sided*/
      ++out;
      ++in1;
    }
  }
  
  

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}
 

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void COMPLEX16SFT2Periodogram1 (LALStatus  *status,
                   REAL8Periodogram1     *peri,
		   COMPLEX16SFTData1      *sft)
{ /*   *********************************************  </lalVerbatim> */

  INT4       length;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "COMPLEX8SFT2Periodogram1", SFTBINC);
  ATTATCHSTATUSPTR (status); 
  
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (peri,  status, SFTBINH_ENULL, SFTBINH_MSGENULL);

  peri->epoch.gpsSeconds = sft->epoch.gpsSeconds;
  peri->epoch.gpsNanoSeconds = sft->epoch.gpsNanoSeconds;
  peri->timeBase = sft->timeBase;
  peri->fminBinIndex = sft->fminBinIndex;
  
  length = sft->length;
  ASSERT (length==peri->length,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);

  if (length > 0){
    REAL8      *out;
    COMPLEX16   *in1;
    REAL8      re,im, factor;
    INT4      n;

    ASSERT (sft->data, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
    ASSERT (peri->data, status, SFTBINH_ENULL,  SFTBINH_MSGENULL);
    out = peri->data;
    in1 = sft->data;
    n= length;

    /* if data was properly normalized */
    factor = 1./peri->timeBase;
    
    /* if data is not normalized... this factor needs to be clarified  */
    /* note inconsistency with the previous function and quantity *factor*  there */
    
    /* if bin zero is included should be treated properly because of factor 2 */
    
    while (n-- >0){
      re = in1->re;
      im = in1->im;
      *out = (re*re + im*im) *factor; /* factor 2  still missing if one_sided */
      ++out;
      ++in1;
    }
  }
  
 
  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}
 

/*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/
/* *******************************  <lalVerbatim file="SFTbinD"> */
void FindNumberLines (LALStatus          *status,
		      LineNoiseInfo      *lineInfo,            
		      CHAR               *fname)
{/*   *********************************************  </lalVerbatim> */
  /* this function counts the number of lines present in ths file fname and
     checks that the format of the lines is correct */

  FILE *fp = NULL;
  REAL8 temp1,temp2,temp3;
  INT4  lineCount, r;
  CHAR  dump[128]; 

  INITSTATUS (status, "FindNumberLines", SFTBINC);
  ATTATCHSTATUSPTR (status);  

  /* make sure arguments are not null */
  ASSERT (lineInfo, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (fname, status, SFTBINH_ENULL, SFTBINH_MSGENULL);

  /* open line noise file for reading */
  fp = fopen( fname, "r");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);

  lineCount = 0;
  do {
    r=fscanf(fp,"%lf%lf%lf%s\n", &temp1, &temp2, &temp3, dump);
    /* make sure the line has the right number of entries or is EOF */
    ASSERT( (r==4)||(r==EOF), status, SFTBINH_EHEADER, SFTBINH_MSGEVAL);
    if (r==4) lineCount++;
  } while ( r != EOF);

  lineInfo->nLines = lineCount;

  fclose(fp);

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void  ReadLineInfo (LALStatus     *status,
		    LineNoiseInfo *lineInfo,
		    CHAR          *fname)
{/*   *********************************************  </lalVerbatim> */
  /* this reads the information about the lines: central frequency, left wing and 
     right wing */
  FILE    *fp = NULL;
  INT4    r, count, nLines;
  REAL8   *lineFreq=NULL;
  REAL8   *leftWing=NULL;
  REAL8   *rightWing=NULL;
  CHAR    dump[128];

  INITSTATUS (status, "ReadLineInfo", SFTBINC);
  ATTATCHSTATUSPTR (status);  

  /* make sure arguments are not null */
  ASSERT (lineInfo, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->nLines > 0, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (lineInfo->lineFreq, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->leftWing, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->rightWing, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (fname, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
 
  /* open line noise file for reading */
  fp = fopen( fname, "r");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);

  nLines = lineInfo->nLines;
  lineFreq = lineInfo->lineFreq;
  leftWing = lineInfo->leftWing;
  rightWing = lineInfo->rightWing;
  /* read line information from file */
  for (count = 0; count < nLines; count++){
    r=fscanf(fp,"%lf%lf%lf%s\n", lineFreq+count, leftWing+count, rightWing+count, dump);
    ASSERT(r==4, status, SFTBINH_EHEADER, SFTBINH_MSGEVAL);
  }

  fclose(fp);

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);

}


/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void CleanCOMPLEX8SFT (LALStatus          *status,
		       COMPLEX8SFTData1   *sft,
		       INT4               width,            
		       LineNoiseInfo      *lineInfo)
{/*   *********************************************  </lalVerbatim> */
  /* function to clean the SFT based on the line information read earlier */

  INT4     nLines, count, leftCount, rightCount, lineBin, minBin, maxBin;
  INT4     leftWingBins, rightWingBins, length;
  REAL8    tBase;
  REAL8    *lineFreq=NULL;
  REAL8    *leftWing=NULL;
  REAL8    *rightWing=NULL;
  COMPLEX8 *inData;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "CleanCOMPLEX8SFT", SFTBINC);
  ATTATCHSTATUSPTR (status);   
 
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  inData = sft->data;
  ASSERT (inData, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (lineInfo, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (lineInfo->nLines, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (lineInfo->lineFreq, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->leftWing, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->rightWing, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (width >= 0, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  length = sft->length;
  ASSERT (length > 0, status, SFTBINH_EHEADER, SFTBINH_MSGEVAL);

  /* copy pointers from input */
  nLines = lineInfo->nLines;
  lineFreq = lineInfo->lineFreq;
  leftWing = lineInfo->leftWing;
  rightWing = lineInfo->rightWing;

  tBase = sft->timeBase;
  minBin = sft->fminBinIndex;
  maxBin = minBin + sft->length - 1;

  /* loop over the lines */
  for (count = 0; count < nLines; count++){

    /* find frequency bins for line frequency and wings */
    lineBin = floor(tBase * lineFreq[count] + 0.5);
    leftWingBins = floor(tBase * leftWing[count]);
    rightWingBins = floor(tBase * rightWing[count]);
  
    /* check that frequency is within band of sft and line is not too wide*/
    if ((lineBin >= minBin) && (lineBin <= maxBin) && (leftWingBins <= width) && (rightWingBins <= width)){
      /* set sft value at central frequency to zero */
      inData = sft->data + lineBin - minBin;
      inData->re = 0.0;
      inData->im = 0.0;

      /* now go left and set the left wing to zero */
      /* make sure that we are always within the sft band */
      /* and set bins to zero only if Wing width is smaller than "width" */ 
      if ((leftWingBins <= width)){
	for (leftCount = 0; leftCount < leftWingBins; leftCount++){
	  if ( (lineBin - minBin - leftCount > 0)){
	    inData = sft->data + lineBin - minBin - leftCount - 1;
	    inData->re = 0.0;
	    inData->im = 0.0;
	  }
	}
      }

      /* now go right making sure again to stay within the sft band */
      if ((rightWingBins <= width )){
	for (rightCount = 0; rightCount < rightWingBins; rightCount++){
	  if ( (maxBin - lineBin - rightCount > 0)){
	    inData = sft->data + lineBin - minBin + rightCount + 1;
	    inData->re = 0.0;
	    inData->im = 0.0;
	  }
	}
      }
    }
  }

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}




/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void CleanCOMPLEX16SFT (LALStatus          *status,
		       COMPLEX16SFTData1   *sft,
		       INT4               width,            
		       LineNoiseInfo      *lineInfo)
{/*   *********************************************  </lalVerbatim> */
  /* function to clean the SFT based on the line information read earlier */

  INT4      nLines, count, leftCount, rightCount, lineBin, minBin, maxBin;
  INT4      leftWingBins, rightWingBins, length;
  REAL8     tBase;
  REAL8     *lineFreq=NULL;
  REAL8     *leftWing=NULL;
  REAL8     *rightWing=NULL;
  COMPLEX16 *inData;
  
  /* --------------------------------------------- */
  INITSTATUS (status, "CleanCOMPLEX16SFT", SFTBINC);
  ATTATCHSTATUSPTR (status);   
 
  /*   Make sure the arguments are not NULL: */ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  inData = sft->data;
  ASSERT (inData, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (lineInfo, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (lineInfo->nLines, status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (lineInfo->lineFreq, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->leftWing, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (lineInfo->rightWing, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 
  ASSERT (width >= 0, status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  length = sft->length;
  ASSERT (length > 0, status, SFTBINH_EHEADER, SFTBINH_MSGEVAL);

  /* copy pointers from input */
  nLines = lineInfo->nLines;
  lineFreq = lineInfo->lineFreq;
  leftWing = lineInfo->leftWing;
  rightWing = lineInfo->rightWing;

  tBase = sft->timeBase;
  minBin = sft->fminBinIndex;
  maxBin = minBin + sft->length - 1;

  /* loop over the lines */
  for (count = 0; count < nLines; count++){

    /* find frequency bins for line frequency and wings */
    lineBin = floor(tBase * lineFreq[count] + 0.5);
    leftWingBins = floor(tBase * leftWing[count]);
    rightWingBins = floor(tBase * rightWing[count]);
  
    /* check that frequency is within band of sft and line is not too wide*/
    if ((lineBin >= minBin) && (lineBin <= maxBin) && (leftWingBins <= width) && (rightWingBins <= width)){
      /* set sft value at central frequency to zero */
      inData = sft->data + lineBin - minBin;
      inData->re = 0.0;
      inData->im = 0.0;

      /* now go left and set the left wing to zero */
      /* make sure that we are always within the sft band */
      /* and set bins to zero only if Wing width is smaller than "width" */ 
      if ((leftWingBins <= width)){
	for (leftCount = 0; leftCount < leftWingBins; leftCount++){
	  if ( (lineBin - minBin - leftCount > 0)){
	    inData = sft->data + lineBin - minBin - leftCount - 1;
	    inData->re = 0.0;
	    inData->im = 0.0;
	  }
	}
      }

      /* now go right making sure again to stay within the sft band */
      if ((rightWingBins <= width )){
	for (rightCount = 0; rightCount < rightWingBins; rightCount++){
	  if ( (maxBin - lineBin - rightCount > 0)){
	    inData = sft->data + lineBin - minBin + rightCount + 1;
	    inData->re = 0.0;
	    inData->im = 0.0;
	  }
	}
      }
    }
  }

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}


/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void WriteCOMPLEX8SFT (LALStatus          *status,
		       COMPLEX8SFTData1   *sft,
		       CHAR               *outfname)
{/*   *********************************************  </lalVerbatim> */
  /* function to write an entire SFT to a file which has been read previously*/
  
  FILE  *fp = NULL;
  COMPLEX8  *inData;
  SFTHeader1  header;
  INT4  i, w;
  REAL4  rePt, imPt;

  /* --------------------------------------------- */
  INITSTATUS (status, "WriteCOMPLEX8SFT", SFTBINC);
  ATTATCHSTATUSPTR (status);   
 
  /*   Make sure the arguments are not NULL and perform basic checks*/ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (sft->length > 0,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (sft->fminBinIndex > 0,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (sft->timeBase > 0,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (sft->data,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (outfname, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 

  /* fill in the header information */
  header.endian = 1.0;
  header.gpsSeconds = sft->epoch.gpsSeconds;
  header.gpsNanoSeconds = sft->epoch.gpsNanoSeconds;
  header.timeBase = sft->timeBase;
  header.fminBinIndex = sft->fminBinIndex;
  header.length = sft->length; 


  /* open the file for writing */
  fp = fopen(outfname, "w");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);

  /* write the header*/
  w = fwrite((void* )&header, sizeof(header), 1, fp);
  ASSERT (w == 1, status, SFTBINH_EWRITE, SFTBINH_MSGEWRITE);

  inData = sft->data;
  for ( i = 0; i < header.length; i++){

    rePt = inData[i].re;
    imPt = inData[i].im;

    /* write the real and imaginary parts */
    w = fwrite((void* )&rePt, sizeof(REAL4), 1, fp);
    ASSERT (w == 1, status, SFTBINH_EWRITE, SFTBINH_MSGEWRITE);
    w = fwrite((void* )&imPt, sizeof(REAL4), 1, fp);
    ASSERT (w == 1, status, SFTBINH_EWRITE, SFTBINH_MSGEWRITE);
  }

  fclose(fp);

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}



/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* *******************************  <lalVerbatim file="SFTbinD"> */
void WriteCOMPLEX16SFT (LALStatus          *status,
		       COMPLEX16SFTData1   *sft,
		       CHAR               *outfname)
{/*   *********************************************  </lalVerbatim> */
  /* function to write entirely an SFT which has been previously read */
  
  FILE  *fp = NULL;
  COMPLEX16  *inData;
  SFTHeader1 header;
  INT4  i, w;
  REAL4  rePt, imPt;

  /* --------------------------------------------- */
  INITSTATUS (status, "WriteCOMPLEX16SFT", SFTBINC);
  ATTATCHSTATUSPTR (status);   
 
  /*   Make sure the arguments are not NULL and perform basic checks*/ 
  ASSERT (sft,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (sft->length > 0,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (sft->fminBinIndex > 0,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (sft->timeBase > 0,  status, SFTBINH_EVAL, SFTBINH_MSGEVAL);
  ASSERT (sft->data,   status, SFTBINH_ENULL, SFTBINH_MSGENULL);
  ASSERT (outfname, status, SFTBINH_ENULL, SFTBINH_MSGENULL); 

  /* fill in the header information */
  header.endian = 1.0;
  header.gpsSeconds = sft->epoch.gpsSeconds;
  header.gpsNanoSeconds = sft->epoch.gpsNanoSeconds;
  header.timeBase = sft->timeBase;
  header.fminBinIndex = sft->fminBinIndex;
  header.length = sft->length; 


  /* open the file for writing */
  fp = fopen(outfname, "w");
  ASSERT (fp, status, SFTBINH_EFILE,  SFTBINH_MSGEFILE);

  /* write the header*/
  w = fwrite((void* )&header, sizeof(header), 1, fp);
  ASSERT (w == 1, status, SFTBINH_EWRITE, SFTBINH_MSGEWRITE);

  inData = sft->data;
  for ( i = 0; i < header.length; i++){

    rePt = inData[i].re;
    imPt = inData[i].im;

    /* write the real and imaginary parts */
    w = fwrite((void* )&rePt, sizeof(REAL8), 1, fp);
    ASSERT (w == 1, status, SFTBINH_EWRITE, SFTBINH_MSGEWRITE);
    w = fwrite((void* )&imPt, sizeof(REAL8), 1, fp);
    ASSERT (w == 1, status, SFTBINH_EWRITE, SFTBINH_MSGEWRITE);
  }

  fclose(fp);

  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}







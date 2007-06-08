/*
*  Copyright (C) 2007 Badri Krishnan, Iraj Gholami
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

/*-----------------------------------------------------------------------
 *
 * File Name: ComputePSD.c
 * Authors:  Krishnan, B.  ; Sintes, A. M.
 *
 * Revision: $Id$
 *
 * History:   Created by Krishnan 
 *            Modified by Sintes 
 *
 *-----------------------------------------------------------------------
 */
/************************************ <lalVerbatim file="ComputePSDCV">
Author: Krishnan, B. , Sintes, A.M. 
$Id$
************************************* </lalVerbatim> */

/* <lalLaTeX>  *******************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*********************************************** </lalLaTeX> */


#include <glob.h> 
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <lal/SFTfileIO.h>
#include <lal/Random.h>
#include <lal/PulsarDataTypes.h>
#include <lal/UserInput.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/LALInitBarycenter.h>
#include <lal/SFTClean.h>

#include <lalapps.h>

RCSID( "$Id$");

/* Error codes and messages */

/************** <lalErrTable file="ComputePSDCErrorTable"> */
#define COMPUTEPSDC_ENORM 0
#define COMPUTEPSDC_ESUB  1
#define COMPUTEPSDC_EARG  2
#define COMPUTEPSDC_EBAD  3
#define COMPUTEPSDC_EFILE 4
#define COMPUTEPSDC_ENULL 5

#define COMPUTEPSDC_MSGENORM "Normal exit"
#define COMPUTEPSDC_MSGESUB  "Subroutine failed"
#define COMPUTEPSDC_MSGEARG  "Error parsing arguments"
#define COMPUTEPSDC_MSGEBAD  "Bad argument values"
#define COMPUTEPSDC_MSGEFILE "Could not create output file"
#define COMPUTEPSDC_MSGENULL "Null Pointer"

/******************************************** </lalErrTable> */


/* Default parameters. */

extern INT4 lalDebugLevel;


#define MAXFILENAMELENGTH 256
/* defaults chosen for L1 */
/*#define INPUTSFTDIR "/nfs/morbo/geo600/hannover/sft/S2-LIGO/S2_L1_Funky-v3Calv5DQ30MinSFTs"*/
#define INPUTSFTDIR "/home/badkri/L1sfts/"
/*#define OUTPUTSFTDIR "/nfs/morbo/geo600/hannover/sft/S2-LIGO-clean/S2_L1_Funky-v3Calv5DQ30MinSFTs-clean"*/
#define OUTPUTPSDFILE "./psd"
#define STARTFREQ 200.0
#define BANDFREQ 20.0


#define TRUE (1==1)
#define FALSE (1==0)


void ReadTimeStampsFile (LALStatus *status, LIGOTimeGPSVector  *ts, CHAR *filename);

int main(int argc, char *argv[]){ 

  static LALStatus       status;  /* LALStatus pointer */ 
  
  INT4 j, k, nBins, iIFO, iSFT, numsft; 
  INT4 numifo;
  REAL8 deltaF;
  FILE *fpOut=NULL;

  REAL8 **periodo=NULL;
  REAL8 *periodoOut=NULL;

  SFTCatalog *catalog = NULL;
  static SFTConstraints constraints;
  MultiSFTVector *inputSFTs = NULL;
 

  MultiPSDVector *multPSD = NULL;  

  LIGOTimeGPS startTimeGPS, endTimeGPS;
  LIGOTimeGPSVector inputTimeStampsVector;
  
  /* log file and strings */
  FILE   *fpLog=NULL;
  CHAR   *fnameLog=NULL; 
  CHAR   *logstr=NULL; 

  /* user input variables */
  BOOLEAN uvar_help;
  CHAR *uvar_inputData;    /* directory for unclean sfts */
  CHAR *uvar_outputPSDFILE;   /* directory for cleaned sfts */
  REAL8 uvar_fStart, uvar_fBand;
  BOOLEAN uvar_log; /* logging done if true */
  REAL8   uvar_startTime, uvar_endTime;
  CHAR   *uvar_timeStampsFile=NULL;
  LALStringVector *uvar_linefiles=NULL;
  INT4     uvar_blocksRngMed, uvar_maxBinsClean;


  /* LALDebugLevel must be called before anything else */
  lalDebugLevel = 0;
  LAL_CALL( LALGetDebugLevel( &status, argc, argv, 'd'), &status);

  /* set defaults */
  uvar_help = FALSE;
  uvar_log = FALSE;  

  uvar_maxBinsClean = 100;
  uvar_blocksRngMed = 101;

  uvar_startTime = 0.0;
  uvar_endTime = 0.0;

  uvar_inputData = (CHAR *)LALMalloc(256 * sizeof(CHAR));
  strcpy(uvar_inputData, INPUTSFTDIR);

  uvar_outputPSDFILE = (CHAR *)LALMalloc(256 * sizeof(CHAR));
  strcpy(uvar_outputPSDFILE, OUTPUTPSDFILE);

  uvar_fStart = STARTFREQ;
  uvar_fBand = BANDFREQ;  


  /* register user input variables */
  LAL_CALL(LALRegisterBOOLUserVar(  &status, "help", 'h', UVAR_HELP,    "Print this message", &uvar_help), &status);  
  LAL_CALL(LALRegisterSTRINGUserVar(&status, "inputData", 'i', UVAR_OPTIONAL, "Input SFT pattern", &uvar_inputData), &status);
  LAL_CALL(LALRegisterSTRINGUserVar(&status, "outputFILE", 'o', UVAR_OPTIONAL, "Output PSD file", &uvar_outputPSDFILE), &status);
  LAL_CALL(LALRegisterREALUserVar(  &status, "fStart", 'f', UVAR_OPTIONAL, "Frequency to start from", &uvar_fStart), &status);
  LAL_CALL(LALRegisterREALUserVar(  &status, "fBand", 'b', UVAR_OPTIONAL, "Frequency Band", &uvar_fBand), &status);
  LAL_CALL(LALRegisterREALUserVar(  &status, "startTime", 's', UVAR_OPTIONAL, "GPS start time", &uvar_startTime), &status);
  LAL_CALL(LALRegisterREALUserVar(  &status, "endTime",  'e',  UVAR_OPTIONAL, "GPS end time", &uvar_endTime), &status);
  LAL_CALL(LALRegisterSTRINGUserVar(&status, "timeStampsFile", 't', UVAR_OPTIONAL, "Time-stamps file", &uvar_timeStampsFile), &status);

  LAL_CALL(LALRegisterINTUserVar(   &status, "blocksRngMed",  'w',  UVAR_OPTIONAL, "Running Median window size", &uvar_blocksRngMed), &status);
  LAL_CALL(LALRegisterINTUserVar(   &status, "maxBinsClean", 'm', UVAR_OPTIONAL, "Maximum Cleaning Bins", &uvar_maxBinsClean), &status);

  LAL_CALL( LALRegisterLISTUserVar(   &status, "linefiles", 0,  UVAR_OPTIONAL, "Comma separated List of linefiles (filenames must contain IFO name)", &uvar_linefiles), &status);
  LAL_CALL( LALRegisterBOOLUserVar(   &status, "log", 0,  UVAR_OPTIONAL, "Write log file", &uvar_log), &status);  


  /* read all command line variables */
  LAL_CALL( LALUserVarReadAllInput(&status, argc, argv), &status);

  /* exit if help was required */
  if (uvar_help)
    exit(0); 

  /********logging the user input variables*********************/

  if (uvar_log) {
    /* open log file for writing */
    fnameLog = (CHAR *)LALMalloc( 512*sizeof(CHAR));
    strcpy(fnameLog,uvar_outputPSDFILE);
    strcat(fnameLog, "_log");
    if ((fpLog = fopen(fnameLog, "w")) == NULL) {
      fprintf(stderr, "Unable to open file %s for writing\n", fnameLog);
      LALFree(fnameLog);
      exit(1);
    }
    
    /* get the log string */
    LAL_CALL( LALUserVarGetLog(&status, &logstr, UVAR_LOGFMT_CFGFILE), &status);  
    
    fprintf( fpLog, "## LOG FILE FOR SFT PSD COMPUTATION\n\n");
    fprintf( fpLog, "# User Input:\n");
    fprintf( fpLog, "#-------------------------------------------\n");
    fprintf( fpLog, logstr);
    LALFree(logstr);
    
    /* append an ident-string defining the exact CVS-version of the code used */
    {
      CHAR command[1024] = "";
      fprintf (fpLog, "\n\n# CVS-versions of executable:\n");
      fprintf (fpLog, "# -----------------------------------------\n");
      fclose (fpLog);
      
      sprintf (command, "ident %s | sort -u >> %s", argv[0], fnameLog);
      system (command);	/* we don't check this. If it fails, we assume that */
    			/* one of the system-commands was not available, and */
    			/* therefore the CVS-versions will not be logged */
      
      LALFree(fnameLog); 
    }
  } /* done with logging */



  /* set detector constraint */
  constraints.detector = NULL;
  
  if ( LALUserVarWasSet( &uvar_startTime ) ) {
    LAL_CALL ( LALFloatToGPS( &status, &startTimeGPS, &uvar_startTime), &status);
    constraints.startTime = &startTimeGPS;
  }
  
  if ( LALUserVarWasSet( &uvar_endTime ) ) {
    LAL_CALL ( LALFloatToGPS( &status, &endTimeGPS, &uvar_endTime), &status);
    constraints.endTime = &endTimeGPS;
  }
  
  if ( LALUserVarWasSet( &uvar_timeStampsFile ) ) {
    LAL_CALL ( ReadTimeStampsFile ( &status, &inputTimeStampsVector, uvar_timeStampsFile), &status);
    constraints.timestamps = &inputTimeStampsVector;
  }
  
  /* get sft catalog */
  LAL_CALL( LALSFTdataFind( &status, &catalog, uvar_inputData, &constraints), &status);
  if ( (catalog == NULL) || (catalog->length == 0) ) {
    fprintf (stderr,"Unable to match any SFTs with pattern '%s'\n", uvar_inputData );
    exit(1);
  }
  
  /* now we can free the inputTimeStampsVector */
  if ( LALUserVarWasSet( &uvar_timeStampsFile ) ) {
    LALFree( inputTimeStampsVector.data );
  }

  /* read the sfts */  
  LAL_CALL( LALLoadMultiSFTs ( &status, &inputSFTs, catalog, uvar_fStart, uvar_fStart + uvar_fBand), &status);

  LAL_CALL( LALDestroySFTCatalog( &status, &catalog ), &status);  	



  /* clean sfts if required */
  if ( LALUserVarWasSet( &uvar_linefiles ) )
    {
      RandomParams *randPar=NULL;
      FILE *fpRand=NULL;
      INT4 seed, ranCount;  
      
      if ( (fpRand = fopen("/dev/urandom", "r")) == NULL ) {
	fprintf(stderr,"Error in opening /dev/urandom" ); 
	exit(1);
      } 
      
      if ( (ranCount = fread(&seed, sizeof(seed), 1, fpRand)) != 1 ) {
	fprintf(stderr,"Error in getting random seed" );
	exit(1);
      }
      
      LAL_CALL ( LALCreateRandomParams (&status, &randPar, seed), &status );
      
      LAL_CALL( LALRemoveKnownLinesInMultiSFTVector ( &status, inputSFTs, uvar_maxBinsClean, uvar_blocksRngMed, uvar_linefiles, randPar), &status);
      
      LAL_CALL ( LALDestroyRandomParams (&status, &randPar), &status);
      fclose(fpRand);
    } /* end cleaning */
  

  /* normalize sfts */
  LAL_CALL( LALNormalizeMultiSFTVect (&status, &multPSD, inputSFTs, uvar_blocksRngMed), &status);


  /* allocate memory for periodogram for all IFO */
  deltaF = multPSD->data[0]->data->deltaF;
  numifo = multPSD->length;
  nBins = multPSD->data[0]->data->data->length;  
  periodoOut = (REAL8 *)LALCalloc(1, nBins*sizeof(REAL8));
  periodo = (REAL8 **)LALCalloc(1, numifo*sizeof(REAL8 *));
  for (k = 0; k < numifo; k++) {
    periodo[k] = (REAL8 *)LALCalloc(1,nBins*sizeof(REAL8));
  }


  for ( iIFO = 0; iIFO < numifo; iIFO++ ) {
    
    numsft = multPSD->data[iIFO]->length;
    
    for ( iSFT = 0; iSFT < numsft; iSFT++) {

      for ( k = 0; k < nBins; k++) {       

	periodo[iIFO][k] += multPSD->data[iIFO]->data[iSFT].data->data[k]/numsft;

      } /* loop over frequency bins */

    } /* loop over SFTs */
    
  } /* loop over IFOs */
  

  for ( k = 0; k < nBins; k++) {       
    REAL8 temp = 0;
    
    for (iIFO = 0; iIFO < numifo; iIFO++) {
      
      temp += 1.0/periodo[iIFO][k];
	
    } /* loop over IFOs */
    
    periodoOut[k] = 1.0/temp;
    
  } /* look over frequency bins */
  

  
  /* write periodo to the output file */
  if (  (fpOut = fopen(uvar_outputPSDFILE, "w")) == NULL)
    {
      fprintf(stderr, "Unable to open output file %s for writing...exiting \n", uvar_outputPSDFILE);
      exit(1);
    }
  for (k = 0; k < nBins; k++)
    fprintf(fpOut, "%f   %e\n", uvar_fStart + k*deltaF, periodoOut[k]);

  fclose(fpOut);


  /* we are now done with the psd */
  LAL_CALL ( LALDestroyMultiPSDVector  ( &status, &multPSD), &status);
  
  /* we are done with the sfts and ucharpeakgram now */
  LAL_CALL (LALDestroyMultiSFTVector(&status, &inputSFTs), &status );
    
  LAL_CALL (LALDestroyUserVars(&status), &status);

  /* free periodo */
  for (k = 0; k < numifo; k++) {
    LALFree(periodo[k]);
  }
  LALFree(periodo);
  LALFree(periodoOut);

  LALCheckMemoryLeaks(); 

  /* INFO( COMPUTEPSDC_MSGENORM ); */
  return COMPUTEPSDC_ENORM;
}




/* read timestamps file */
void ReadTimeStampsFile (LALStatus          *status,
			 LIGOTimeGPSVector  *ts,
			 CHAR               *filename)
{

  FILE  *fp = NULL;
  INT4  numTimeStamps, r;
  UINT4 j;
  REAL8 temp1, temp2;

  INITSTATUS (status, "ReadTimeStampsFile", rcsid);
  ATTATCHSTATUSPTR (status);

  ASSERT(ts, status, COMPUTEPSDC_ENULL,COMPUTEPSDC_MSGENULL); 
  ASSERT(ts->data == NULL, status, COMPUTEPSDC_ENULL,COMPUTEPSDC_MSGENULL); 
  ASSERT(ts->length == 0, status, COMPUTEPSDC_ENULL,COMPUTEPSDC_MSGENULL); 
  ASSERT(filename, status, COMPUTEPSDC_ENULL,COMPUTEPSDC_MSGENULL); 

  if ( (fp = fopen(filename, "r")) == NULL) {
    ABORT( status, COMPUTEPSDC_EFILE, COMPUTEPSDC_MSGEFILE);
  }

  /* count number of timestamps */
  numTimeStamps = 0;     

  do {
    r = fscanf(fp,"%lf%lf\n", &temp1, &temp2);
    /* make sure the line has the right number of entries or is EOF */
    if (r==2) numTimeStamps++;
  } while ( r != EOF);
  rewind(fp);

  ts->length = numTimeStamps;
  ts->data = LALCalloc (1, numTimeStamps * sizeof(LIGOTimeGPS));;
  if ( ts->data == NULL ) {
    fclose(fp);
    ABORT( status, COMPUTEPSDC_ENULL, COMPUTEPSDC_MSGENULL);
  }
  
  for (j = 0; j < ts->length; j++)
    {
      r = fscanf(fp,"%lf%lf\n", &temp1, &temp2);
      ts->data[j].gpsSeconds = (INT4)temp1;
      ts->data[j].gpsNanoSeconds = (INT4)temp2;
    }
  
  fclose(fp);
  	 
  DETATCHSTATUSPTR (status);
  /* normal exit */
  RETURN (status);
}    









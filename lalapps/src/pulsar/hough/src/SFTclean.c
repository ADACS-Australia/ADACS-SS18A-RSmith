/*-----------------------------------------------------------------------
 *
 * File Name: SFTclean.c
 * Authors:  Krishnan, B. 
 *
 * Revision: $Id$
 *
 * History:   Created by Krishnan July 12, 2004
 *            Modified...
 *
 *-----------------------------------------------------------------------
 */
/************************************ <lalVerbatim file="SFTbinTestCV">
Author: Krishnan, B.  
$Id$
************************************* </lalVerbatim> */

/* <lalLaTeX>  *******************************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*********************************************** </lalLaTeX> */


#include <lal/SFTClean.h>
#include <glob.h> 
#include <lalapps.h>

RCSID ( "$Id$");

/* Error codes and messages */

/************** <lalErrTable file="SFTCLEANCErrorTable"> */
#define SFTCLEANC_ENORM 0
#define SFTCLEANC_ESUB  1
#define SFTCLEANC_EARG  2
#define SFTCLEANC_EBAD  3
#define SFTCLEANC_EFILE 4

#define SFTCLEANC_MSGENORM "Normal exit"
#define SFTCLEANC_MSGESUB  "Subroutine failed"
#define SFTCLEANC_MSGEARG  "Error parsing arguments"
#define SFTCLEANC_MSGEBAD  "Bad argument values"
#define SFTCLEANC_MSGEFILE "Could not create output file"
/******************************************** </lalErrTable> */


/* Default parameters. */

extern int lalDebugLevel;


#define MAXFILENAMELENGTH 256
/* defaults chosen for L1 */
#define INPUTSFTDIR "./S3H1data/"
#define OUTPUTSFTDIR "./test/"
#define STARTFREQ 950.0
#define ENDFREQ 970.0
#define WINDOWSIZE 100
#define MAXBINS 20



/* A global pointer for debugging. */
#ifndef NDEBUG
char *lalWatch;
#endif

#define TRUE (1==1)
#define FALSE (1==0)



int main(int argc, char *argv[]){ 

  static LALStatus       status;  /* LALStatus pointer */ 

  static MultiSFTVector  *inputSFTs = NULL;
  static SFTCatalog *catalog = NULL;
  static SFTConstraints constraints;


  /* 09/09/05 gam; randPar now a parameter for LALCleanCOMPLEX8SFT */
  FILE *fp=NULL;   
  INT4 seed, ranCount;
  RandomParams *randPar=NULL; 
  UINT4 k;  

  /* user input variables */
  BOOLEAN uvar_help;
  LALStringVector *uvar_linefiles=NULL; /* files with harmonics info */
  CHAR *uvar_sftDir;    /* directory for unclean sfts */
  CHAR *uvar_outDir;   /* directory for cleaned sfts */
  REAL8 uvar_fMin, uvar_fMax;
  INT4  uvar_window, uvar_maxBins;

  /* set defaults */

  lalDebugLevel = 0;
  /* LALDebugLevel must be called before anything else */
  LAL_CALL( LALGetDebugLevel( &status, argc, argv, 'd'), &status);

  uvar_help = FALSE;
  uvar_sftDir = (CHAR *)LALMalloc(256 * sizeof(CHAR));
  strcpy(uvar_sftDir, INPUTSFTDIR);

  uvar_outDir = (CHAR *)LALMalloc(256 * sizeof(CHAR));
  strcpy(uvar_outDir, OUTPUTSFTDIR);

  uvar_fMin = STARTFREQ;
  uvar_fMax = ENDFREQ;  
  uvar_window = WINDOWSIZE;
  uvar_maxBins = MAXBINS;

  /* register user input variables */
  LAL_CALL( LALRegisterBOOLUserVar(   &status, "help",     'h',UVAR_HELP,     "Print this message",     &uvar_help),   &status);  
  LAL_CALL( LALRegisterSTRINGUserVar( &status, "sftDir",   'i',UVAR_OPTIONAL, "Input SFT file pattern", &uvar_sftDir), &status);
  LAL_CALL( LALRegisterSTRINGUserVar( &status, "outDir",   'o',UVAR_OPTIONAL, "Output SFT Directory",   &uvar_outDir), &status);
  LAL_CALL( LALRegisterREALUserVar(   &status, "fMin",      0, UVAR_OPTIONAL, "start Frequency",        &uvar_fMin),   &status);
  LAL_CALL( LALRegisterREALUserVar(   &status, "fMax",      0, UVAR_OPTIONAL, "Max Frequency Band",     &uvar_fMax),   &status);
  LAL_CALL( LALRegisterINTUserVar(    &status, "window",   'w',UVAR_OPTIONAL, "Window size",            &uvar_window), &status);
  LAL_CALL( LALRegisterINTUserVar(    &status, "maxBins",  'm',UVAR_OPTIONAL, "Max. bins to clean",     &uvar_maxBins),&status);
  LAL_CALL( LALRegisterLISTUserVar(   &status, "linefiles", 0, UVAR_OPTIONAL, "List of linefiles (filenames must contain IFO name)", &uvar_linefiles),  &status);



  /* read all command line variables */
  LAL_CALL( LALUserVarReadAllInput(&status, argc, argv), &status);

  /* exit if help was required */
  if (uvar_help)
    exit(0); 
  
  /* set detector constraint */
  constraints.detector = NULL;

  /* get sft catalog */
  LAL_CALL( LALSFTdataFind( &status, &catalog, uvar_sftDir, &constraints), &status);
  if ( (catalog == NULL) || (catalog->length == 0) ) {
    fprintf (stderr,"Unable to match any SFTs with pattern '%s'\n", uvar_sftDir );
    exit(1);
  }
  
  /* read the sfts */
  LAL_CALL( LALLoadMultiSFTs ( &status, &inputSFTs, catalog, uvar_fMin, uvar_fMax), &status);
  
  /* generate random number seed */
  fp=fopen("/dev/urandom", "r");
  if (!fp) 
    { 
      fprintf(stderr,"Error in opening /dev/urandom \n"); 
      exit(1); 
    } 
  ranCount = fread(&seed, sizeof(seed), 1, fp);
  if (!(ranCount==1)) 
    { 
      fprintf(stderr,"Error in reading random seed \n"); 
      exit(1); 
    } 
  LAL_CALL ( LALCreateRandomParams (&status, &randPar, seed), &status );


  /* clean  lines */
  if ( LALUserVarWasSet( &uvar_linefiles ) ) {
    LAL_CALL( LALRemoveKnownLinesInMultiSFTVector ( &status, inputSFTs, uvar_maxBins, 
						  uvar_window, uvar_linefiles, randPar), &status);
  }

  /* write output */
  for (k = 0; k < inputSFTs->length; k++) {
    LAL_CALL( LALWriteSFTVector2Dir ( &status, inputSFTs->data[k], uvar_outDir, "cleaned", "cleaned"), &status);
  }

  /* Free memory */
  fclose(fp);
  LAL_CALL( LALDestroySFTCatalog( &status, &catalog ), &status);
  LAL_CALL( LALDestroyMultiSFTVector(&status, &inputSFTs), &status );
  LAL_CALL( LALDestroyUserVars(&status), &status);
  LAL_CALL( LALDestroyRandomParams (&status, &randPar), &status);

  LALCheckMemoryLeaks(); 

  return 0;
}









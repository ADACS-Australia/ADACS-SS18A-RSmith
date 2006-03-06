/********************************************************************************************/
/*      FreqFiducialTime - shifting frequency parameters of Einstein at Home result files   */
/*              to a fixed fiducial time for post-processing coincidence analysis           */
/*                                                                                          */
/*                              Holger Pletsch, UWM - March 2006                            */ 
/********************************************************************************************/

/* This code simply shifts all frequency parameters of a combined result file generated
   by combiner_v2.py, to a fixed fiducial GPS time for later coincidence analysis. 
   Note: The code makes use of the current Einstein at Home setup file.
   (i.e. 'CFS_S4run_setup.h')  
*/

/* ----------------------------------------------------------------------------- */
/* defines */
#ifndef FALSE
#define FALSE (1==0)
#endif
#ifndef TRUE
#define TRUE  (1==1)
#endif

#define DONE_MARKER "%DONE\n"
/* maximum depth of a linked structure. */
#define LINKEDSTR_MAX_DEPTH 1024 


/* ----------------------------------------------------------------------------- */
/* file includes */
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "getopt.h"
#include <math.h>


#include <unistd.h>

#include <lal/LALDatatypes.h>
#include <lal/LALMalloc.h>
#include <lal/LALConstants.h>
#include <lal/LALStatusMacros.h>
#include <lal/ConfigFile.h>
#include <lal/UserInput.h>

#include <lalapps.h>

#include "CFS_S4run_setup.h"  /* current Einstein at Home setup */



/* this is defined in C99 and *should* be in math.h.  Long term
   protect this with a HAVE_FINITE */
#ifdef _MSC_VER
#include <float.h>
#define finite _finite
#else
int finite(double);
#endif




/* ----------------------------------------------------------------------------- */
/* some error codes and messages */
#define FIDUCIALC_ENULL            1
#define FIDUCIALC_ENONULL          2
#define FIDUCIALC_ESYS             3
#define FIDUCIALC_EINVALIDFSTATS   4
#define FIDUCIALC_EMEM             5
#define FIDUCIALC_ENORMAL          6


#define FIDUCIALC_MSGENULL         "Arguments contained an unexpected null pointer"
#define FIDUCIALC_MSGENONULL       "Input pointer was not NULL"
#define FIDUCIALC_MSGESYS          "System call failed (probably file IO"
#define FIDUCIALC_MSGEINVALIDFSTATS "Invalid Fstats file"
#define FIDUCIALC_MSGEMEM          "Sorry, ran out of memory... bye."


#define FIDUCIAL_EXIT_OK 0
#define FIDUCIAL_EXIT_ERR     31
#define FIDUCIAL_EXIT_READCND  32
#define FIDUCIAL_EXIT_FCTEST   33
#define FIDUCIAL_EXIT_OUTFAIL  34


/* ----------------------------------------------------------------------------- */
/* structures */

typedef struct FiducialTimeConfigVarsTag 
{
  CHAR *OutputFile;  /*  Name of output file */
  CHAR *InputFile;   /*  Name of input file (combined result file produced by combiner_v2.py */
} FiducialTimeConfigVars;

typedef struct CandidateListTag
{
  UINT4 iCand;       /*  Candidate id: unique with in this program.  */
  REAL8 f;           /*  Frequency of the candidate */
  REAL8 Alpha;       /*  right ascension of the candidate */
  REAL8 Delta;       /*  declination  of the candidate */
  REAL8 F1dot;       /*  spindown (d/dt f) of the candidate */
  REAL8 TwoF;        /*  Maximum value of F for the cluster */
  INT4 FileID;       /*  File ID to specify from which file the candidate under consideration originally came. */
  CHAR resultfname[256];  /*  Name of the particular result file where values originally came from. */
} CandidateList;     




/* ----------------------------------------------------------------------------- */
/* Function declarelations */
void ReadCommandLineArgs( LALStatus *, INT4 argc, CHAR *argv[], FiducialTimeConfigVars *CLA ); 
void ReadCombinedFile( LALStatus *lalStatus, CandidateList **CList, FiducialTimeConfigVars *CLA, INT4 *candlen );

void ComputeFiducialTimeFrequency( LALStatus *, CandidateList *CList, INT4 candlen );

void PrintResultFile( LALStatus *, const FiducialTimeConfigVars *CLA, CandidateList *CList, INT4 candlen );

void FreeMemory( LALStatus *, FiducialTimeConfigVars *CLA, CandidateList *CList, const UINT4 datalen );
void FreeConfigVars( LALStatus *, FiducialTimeConfigVars *CLA );




/* ----------------------------------------------------------------------------- */
/* Global Variables */
/*! @param global_status LALStatus Used to initialize LALStatus lalStatus. */
LALStatus global_status;
/*! @param lalDebugLevel INT4 Control debugging behaviours. Defined in lalapps.h */
extern INT4 lalDebugLevel;
/*! @param vrbflg        INT4 Control debugging messages. Defined in lalapps.h */
extern INT4 vrbflg;

const REAL8 FIXED_FIDUCIAL_TIME = 793555944; /* here e.g. GPS startTime of first data stretch in S4 is chosen */


RCSID ("$Id$");



/* ------------------------------------------------------------------------------------------*/
/* Code starts here.                                                                         */
/* ------------------------------------------------------------------------------------------*/
/* ########################################################################################## */
/*!
  Main function  
*/

int main(INT4 argc,CHAR *argv[]) 
{
  LALStatus *lalStatus = &global_status;
  INT4 CLength=0;
  CandidateList *AllmyC = NULL;
  FiducialTimeConfigVars FTCV;


  lalDebugLevel = 0 ;  
  vrbflg = 1;   /* verbose error-messages */

  /* Get the debuglevel from command line arg, then set laldebuglevel. */
  LAL_CALL (LALGetDebugLevel(lalStatus, argc, argv, 'v'), lalStatus);

  /* Reads command line arguments */
  LAL_CALL( ReadCommandLineArgs( lalStatus, argc, argv, &FTCV ), lalStatus); 

  /* Reads in combined candidare file, set CLength */
  LAL_CALL( ReadCombinedFile(lalStatus, &AllmyC, &FTCV, &CLength), lalStatus);

  /* -----------------------------------------------------------------------------------------*/      
  /* Compute shifting of frequency parameters */
  LAL_CALL( ComputeFiducialTimeFrequency( lalStatus, AllmyC, CLength),lalStatus );

  /* -----------------------------------------------------------------------------------------*/      
  /* Output result file */
  LAL_CALL( PrintResultFile( lalStatus, &FTCV, AllmyC, CLength),lalStatus );

  /* -----------------------------------------------------------------------------------------*/      
  /* Clean-up */
  LAL_CALL( FreeMemory(lalStatus, &FTCV, AllmyC, CLength), lalStatus );

  LALCheckMemoryLeaks(); 

  return(FIDUCIAL_EXIT_OK);
 
} /* main() */


/* ########################################################################################## */

void PrintResultFile(LALStatus *lalStatus, const FiducialTimeConfigVars *CLA, CandidateList *CList, INT4 candlen)
{
  INT4 iindex;
  

  FILE *fp = NULL;
  INT4 *count;
  INT4 nmax = 0;
  
  INITSTATUS( lalStatus, "PrintResultFile", rcsid );
  ATTATCHSTATUSPTR (lalStatus);

  ASSERT( CLA != NULL, lalStatus, FIDUCIALC_ENULL, FIDUCIALC_MSGENULL);
  ASSERT( CList != NULL, lalStatus, FIDUCIALC_ENULL, FIDUCIALC_MSGENULL);
  
  if( (count = (INT4 *) LALCalloc( (size_t) (nmax + 1), sizeof(INT4))) == NULL ) {
    LALPrintError("Could not allocate Memory! \n");
    ABORT (lalStatus, FIDUCIALC_EMEM, FIDUCIALC_MSGEMEM);
  }
  


  /* ------------------------------------------------------------- */
  /* Print out to the user-specified output file.*/
  if( (fp = fopen(CLA->OutputFile,"w")) == NULL ) 
    {
      LALPrintError("\n Cannot open file %s\n",CLA->OutputFile); 
      ABORT (lalStatus, FIDUCIALC_EMEM, FIDUCIALC_MSGEMEM);
    }

  /* output lines */
  /*INITSTATUS( lalStatus, "print_output", rcsid ); */
 
  iindex=0;
  while( iindex < candlen )
  {
    fprintf(fp,"%" LAL_INT4_FORMAT " %.13g %.7g %.7g %.5g %.6g\n",
	    CList[iindex].FileID, 
	    CList[iindex].f, 
	    CList[iindex].Alpha, 
	    CList[iindex].Delta, 
	    CList[iindex].F1dot, 
	    CList[iindex].TwoF );
    iindex++;
  }
  fprintf(fp, "%s", DONE_MARKER);
  
  BEGINFAIL(lalStatus) {fclose(fp);} ENDFAIL(lalStatus);

  LALFree( count );

  DETATCHSTATUSPTR (lalStatus);
  RETURN (lalStatus);
} /* PrintResult() */



/* ########################################################################################## */

void FreeMemory( LALStatus *lalStatus, 
	    FiducialTimeConfigVars *CLA, 
	    CandidateList *CList, 
	    const UINT4 CLength)
{
  INITSTATUS( lalStatus, "FreeMemory", rcsid );
  ATTATCHSTATUSPTR (lalStatus);

  FreeConfigVars( lalStatus->statusPtr, CLA );

  if( CList != NULL ) LALFree(CList);

  DETATCHSTATUSPTR (lalStatus);
  RETURN (lalStatus);
} /* FreeMemory */


/* ########################################################################################## */

void FreeConfigVars(LALStatus *lalStatus, FiducialTimeConfigVars *CLA )
{
  INITSTATUS( lalStatus, "FreeConfigVars", rcsid );

  if( CLA->OutputFile != NULL ) LALFree(CLA->OutputFile);
  if( CLA->InputFile != NULL ) LALFree(CLA->InputFile);

  RETURN (lalStatus);
} /* FreeCOnfigVars */



/* ########################################################################################## */

void ReadCombinedFile( LALStatus *lalStatus, 
		      CandidateList **CList, 
		      FiducialTimeConfigVars *CLA, 
		      INT4 *candlen )
{
  UINT4 i;
  UINT4 numlines;
  REAL8 epsilon=1e-5;
  CHAR line1[256];
  FILE *fp;
  INT4 nread;
  UINT4 checksum=0;
  UINT4 bytecount=0;
  const CHAR *fname;
        
  fname = CLA->InputFile;

  INITSTATUS( lalStatus, "ReadCombinedFile", rcsid );
  ATTATCHSTATUSPTR (lalStatus);
  ASSERT( fname != NULL, lalStatus, FIDUCIALC_ENULL, FIDUCIALC_MSGENULL);
  ASSERT( *CList == NULL, lalStatus, FIDUCIALC_ENONULL, FIDUCIALC_MSGENONULL);

  /* ------ Open and count candidates file ------ */
  i=0;
  fp=fopen(fname,"rb");
  if (fp==NULL) 
    {
      LALPrintError("File %s doesn't exist!\n",fname);
      ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
     }
  while(fgets(line1,sizeof(line1),fp)) {
    UINT4 k;
    size_t len=strlen(line1);

    /* check that each line ends with a newline char (no overflow of
       line1 or null chars read) */
    if (!len || line1[len-1] != '\n') {
      LALPrintError(
              "Line %d of file %s is too long or has no NEWLINE.  First 255 chars are:\n%s\n",
              i+1, fname, line1);
      fclose(fp);
      ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
     }

    /* increment line counter */
    i++;

    /* maintain a running checksum and byte count */
    bytecount+=len;
    for (k=0; k<len; k++)
      checksum+=(INT4)line1[k];
  }
  numlines=i;
  /* -- close candidate file -- */
  fclose(fp);     

  if ( numlines == 0) 
    {
      LALPrintError ("ERROR: File '%s' has no lines so is not properly terminated by: %s", fname, DONE_MARKER);
      ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
    }

  /* output a record of the running checksun amd byte count */
  LALPrintError( "%% %s: bytecount %" LAL_UINT4_FORMAT " checksum %" LAL_UINT4_FORMAT "\n", fname, bytecount, checksum);

  /* check validity of this Fstats-file */
  if ( strcmp(line1, DONE_MARKER ) ) 
    {
      LALPrintError ("ERROR: File '%s' is not properly terminated by: %sbut has %s instead", fname, DONE_MARKER, line1);
      ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
    }
  else
    numlines --;        /* avoid stepping on DONE-marker */

  *candlen=numlines;

#if 0 /* Do we need to check this? */
  if (*candlen <= 0  )
    {
      LALPrintError("candidate length = %ud!\n",*candlen);
      exit(FIDUCIAL_EXIT_ERR);;
    }/* check that we have candidates. */
#endif

  
  /* reserve memory for fstats-file contents */
  if (numlines > 0) 
    { 
      *CList = (CandidateList *)LALMalloc (numlines*sizeof(CandidateList));
      if ( !CList ) 
        { 
          LALPrintError ("Could not allocate memory for candidate file %s\n\n", fname);
	  ABORT (lalStatus, FIDUCIALC_EMEM, FIDUCIALC_MSGEMEM);
        }
    }

  /* ------ Open and count candidates file ------ */
  i=0;
  fp=fopen(fname,"rb");
  if (fp==NULL) 
    {
      LALPrintError("fopen(%s) failed!\n", fname);
      LALFree ((*CList));
      ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
    }
  while(i < numlines && fgets(line1,sizeof(line1),fp))
    {
      CHAR newline='\0';
      CandidateList *cl=&(*CList)[i];

      if (strlen(line1)==0 || line1[strlen(line1)-1] != '\n') {
        LALPrintError(
                "Line %d of file %s is too long or has no NEWLINE.  First 255 chars are:\n%s\n",
                i+1, fname, line1);
        LALFree ((*CList));
        fclose(fp);
	ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
      }
      
      nread = sscanf (line1,
		      "%s %" LAL_INT4_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT " %" LAL_REAL8_FORMAT "%c", &(cl->resultfname), &(cl->FileID), &(cl->f), &(cl->Alpha), &(cl->Delta), &(cl->F1dot), &(cl->TwoF), &newline );

      /* check that values that are read in are sensible */
      if (
          cl->FileID < 0                     ||
          cl->f < 0.0                        ||
          cl->TwoF < 0.0                     ||
          cl->Alpha <         0.0 - epsilon  ||
          cl->Alpha >   LAL_TWOPI + epsilon  ||
          cl->Delta < -0.5*LAL_PI - epsilon  ||
          cl->Delta >  0.5*LAL_PI + epsilon  ||
	  !finite(cl->FileID)                ||                                                                 
          !finite(cl->f)                     ||
          !finite(cl->Alpha)                 ||
          !finite(cl->Delta)                 ||
	  !finite(cl->F1dot)                 ||
          !finite(cl->TwoF)
          ) {
          LALPrintError(
                  "Line %d of file %s has invalid values.\n"
                  "First 255 chars are:\n"
                  "%s\n"
                  "1st and 4th field should be positive.\n" 
                  "2nd field should lie between 0 and %1.15f.\n" 
                  "3rd field should lie between %1.15f and %1.15f.\n"
                  "All fields should be finite\n",
                  i+1, fname, line1, (double)LAL_TWOPI, (double)-LAL_PI/2.0, (double)LAL_PI/2.0);
          LALFree ((*CList));
          fclose(fp);
	  ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
      }
           
           

      /* check that the FIRST character following the Fstat value is a
         newline.  Note deliberate LACK OF WHITE SPACE char before %c
         above */
      if (newline != '\n') {
        LALPrintError(
                "Line %d of file %s had extra chars after F value and before newline.\n"
                "First 255 chars are:\n"
                "%s\n",
                i+1, fname, line1);
        LALFree ((*CList));
        fclose(fp);
	ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
      }

      /* check that we read 8 quantities with exactly the right format */
      if ( nread != 8 )
        {
          LALPrintError ("Found %d not %d values on line %d in file '%s'\n"
                         "Line in question is\n%s",
                         nread, 8, i+1, fname, line1);               
          LALFree ((*CList));
          fclose(fp);
	  ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
        }



      i++;
    } /*  end of main while loop */
  /* check that we read ALL lines! */
  if (i != numlines) {
    LALPrintError(
            "Read of file %s terminated after %d line but numlines=%d\n",
            fname, i, numlines);
    LALFree((*CList));
    fclose(fp);
    ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
  }

  /* read final line with %DONE\n marker */
  if (!fgets(line1, sizeof(line1), fp)) {
    LALPrintError(
            "Failed to find marker line of file %s\n",
            fname);
    LALFree((*CList));
    fclose(fp);
    ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
  }

  /* check for %DONE\n marker */
  if (strcmp(line1, DONE_MARKER)) {
    LALPrintError(
            "Failed to parse marker: 'final' line of file %s contained %s not %s",
            fname, line1, DONE_MARKER);
    LALFree ((*CList));
    fclose(fp);
    ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
  }

  /* check that we are now at the end-of-file */
  if (fgetc(fp) != EOF) {
    LALPrintError(
            "File %s did not terminate after %s",
            fname, DONE_MARKER);
    LALFree ((*CList));
    fclose(fp);
    ABORT (lalStatus, FIDUCIALC_EINVALIDFSTATS, FIDUCIALC_MSGEINVALIDFSTATS);
  }

  /* -- close candidate file -- */
  fclose(fp);     

  DETATCHSTATUSPTR (lalStatus);
  RETURN (lalStatus);

} /* ReadCombinedFile() */




/* ########################################################################################## */

void ReadCommandLineArgs( LALStatus *lalStatus, 
		     INT4 argc, 
		     CHAR *argv[], 
		     FiducialTimeConfigVars *CLA ) 
{

  CHAR* uvar_InputData;
  CHAR* uvar_OutputData;
  BOOLEAN uvar_help;

  INITSTATUS( lalStatus, "ReadCommandLineArgs", rcsid );
  ATTATCHSTATUSPTR (lalStatus);

  ASSERT( CLA != NULL, lalStatus, FIDUCIALC_ENULL, FIDUCIALC_MSGENULL);

  uvar_help = 0;
  uvar_InputData = NULL;
  uvar_OutputData = NULL;
 
  /* register all our user-variables */
  LALregBOOLUserVar(lalStatus,       help,           'h', UVAR_HELP,     "Print this message"); 

  LALregSTRINGUserVar(lalStatus,     OutputData,     'o', UVAR_REQUIRED, "Ouput file name");
  LALregSTRINGUserVar(lalStatus,     InputData,      'i', UVAR_OPTIONAL, "Input file name");

  TRY (LALUserVarReadAllInput(lalStatus->statusPtr,argc,argv),lalStatus); 


  if (uvar_help) {	/* if help was requested, we're done here */
    LALPrintError("%s\n",rcsid);
    fflush(stderr);
    LALDestroyUserVars(lalStatus->statusPtr);
    exit(FIDUCIAL_EXIT_OK);
  }


  CLA->OutputFile = NULL;
  CLA->InputFile = NULL;
  
  CLA->OutputFile = (CHAR *) LALMalloc(strlen(uvar_OutputData)+1);
  if(CLA->OutputFile == NULL)
    {
      TRY( FreeConfigVars( lalStatus->statusPtr, CLA ), lalStatus);
      exit(FIDUCIAL_EXIT_ERR);
    }      
  strcpy(CLA->OutputFile,uvar_OutputData);

  CLA->InputFile = (CHAR *) LALMalloc(strlen(uvar_InputData)+1);
  if(CLA->InputFile == NULL)
    {
      TRY( FreeConfigVars( lalStatus->statusPtr, CLA ), lalStatus);
      exit(FIDUCIAL_EXIT_ERR);
    }
  strcpy(CLA->InputFile,uvar_InputData);
    
  LALDestroyUserVars(lalStatus->statusPtr);
  BEGINFAIL(lalStatus) {
    LALFree(CLA->InputFile);
    LALFree(CLA->OutputFile);
  } ENDFAIL(lalStatus);

  DETATCHSTATUSPTR (lalStatus);
  RETURN (lalStatus);
} /* void ReadCommandLineArgs()  */



/* ########################################################################################## */
  

void ComputeFiducialTimeFrequency( LALStatus *lalStatus, 
				   CandidateList *CList, 
				   INT4 candlen)
{
  REAL8 f_CFS;
  REAL8 F1dot_CFS;
  REAL8 f_fiducial;
  REAL8 deltaT;
  INT4 iindex;
  WU_search_params_t wparams;

  INITSTATUS( lalStatus, "ComputeFiducialTimeFrequency", rcsid );
  ATTATCHSTATUSPTR (lalStatus);
  printf("Frequency values are shifted to fixed fiducial GPS time: %f\n", FIXED_FIDUCIAL_TIME);

  f_CFS=0;
  f_fiducial=0;
  F1dot_CFS=0;
  iindex=0;
  deltaT=0;
  
  while( iindex < candlen )
    {
      f_CFS = CList[iindex].f;
      F1dot_CFS = CList[iindex].F1dot;
      
      findSearchParams4Result( CList[iindex].resultfname, &wparams );
	
      deltaT = wparams.startTime - FIXED_FIDUCIAL_TIME;  /* fixed fiducial time = e.g. GPS time of first SFT in S4 */

      f_fiducial = f_CFS + (F1dot_CFS * deltaT);
     
      CList[iindex].f = f_fiducial;
 
      f_CFS = 0;
      f_fiducial = 0;
      F1dot_CFS = 0;
      deltaT=0;

      iindex++;
  }
 
  DETATCHSTATUSPTR (lalStatus);
  RETURN (lalStatus);

} /* ComputeFiducialTimeFrequencies () */

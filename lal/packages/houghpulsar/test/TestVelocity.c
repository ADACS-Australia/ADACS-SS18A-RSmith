/*-----------------------------------------------------------------------
 *
 * File Name: TestVelocity.c
 *
 * Authors: Krishnan, B., Sintes, A.M., 
 *
 * Revision: $Id$
 *
 * History:   Created by Badri Krishnan May 24, 2003
 *        
 *
 *-----------------------------------------------------------------------
 */

/*
 * 1.  An author and Id block
 */

/************************************ <lalVerbatim file="TestVelocityCV">
Author: Krishnan, B., Sintes, A.M. 
$Id$
************************************* </lalVerbatim> */

/*
 * 2. Commented block with the documetation of this module
 */
/* ************************************************<lalLaTeX>
\subsection{Program \ \texttt{TestVelocity.c}}
\label{s:TestVelocity.c}
Tests the calculation of the averaged velocity of a given detector.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Usage}
\begin{verbatim}
TestVelocity  [-d debuglevel] [-a accuracy] 
\end{verbatim}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Description}
This program computes the averaged velocity  of the GEO600 detector 
between the times 730000044 and 730003644 with a default accuracy of 0.01.
The two ephemeris files (e.g., for data taken in 2003, \verb@sun03.dat@ and
\verb@earth03.dat@) are assumed to be in the  directory
\verb@lal/packages/pulsar/test/@.

The \verb@-d@ option sets the debug level to the specified value
\verb@debuglevel@.  The \verb@-a@ flag tells the program which accuracy to use.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Exit codes}
\vspace{0.1in}
\input{TESTVELOCITYCErrorTable}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Uses}
\begin{verbatim}
LALAvgDetectorVel()
LALPrintError()
LALMalloc()
LALFree()
LALCheckMemoryLeaks()
\end{verbatim}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection*{Notes}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\vfill{\footnotesize\input{TestVelocityCV}}

********************************************</lalLaTeX> */

/* #include "./Velocity.h" */
#include <lal/Velocity.h> 

NRCSID (TESTVELOCITYC, "$Id$");


/* Error codes and messages */

/************** <lalErrTable file="TESTVELOCITYCErrorTable"> */
#define TESTVELOCITYC_ENORM 0
#define TESTVELOCITYC_ESUB  1
#define TESTVELOCITYC_EARG  2
#define TESTVELOCITYC_EBAD  3
#define TESTVELOCITYC_EFILE 4

#define TESTVELOCITYC_MSGENORM "Normal exit"
#define TESTVELOCITYC_MSGESUB  "Subroutine failed"
#define TESTVELOCITYC_MSGEARG  "Error parsing arguments"
#define TESTVELOCITYC_MSGEBAD  "Bad argument values"
#define TESTVELOCITYC_MSGEFILE "Could not create output file"
/******************************************** </lalErrTable> */


/* Default parameters. */

INT4 lalDebugLevel=7;

/* #define T0SEC 714153733 */
#define T0SEC 730000044 
#define T0NSEC 0
#define TBASE 3600.0
#define ACCURACY 0.01

/* Locations of the earth and sun ephemeris data */

#define EARTHDATA "earth03.dat" 
#define SUNDATA "sun03.dat" 

/* Usage format string.  */
#define USAGE "Usage: %s [-d debuglevel] [-a accuracy]\n"

/*********************************************************************/
/* Macros for printing errors & testing subroutines (from Creighton) */
/*********************************************************************/

#define ERROR( code, msg, statement )                                \
do {                                                                 \
  if ( lalDebugLevel & LALERROR )                                    \
    LALPrintError( "Error[0] %d: program %s, file %s, line %d, %s\n" \
                   "        %s %s\n", (code), *argv, __FILE__,       \
              __LINE__, TESTVELOCITYC, statement ? statement :  \
                   "", (msg) );                                      \
} while (0)

#define INFO( statement )                                            \
do {                                                                 \
  if ( lalDebugLevel & LALINFO )                                     \
    LALPrintError( "Info[0]: program %s, file %s, line %d, %s\n"     \
                   "        %s\n", *argv, __FILE__, __LINE__,        \
              TESTVELOCITYC, (statement) );                     \
} while (0)

#define SUB( func, statusptr )                                       \
do {                                                                 \
  if ( (func), (statusptr)->statusCode ) {                           \
    ERROR( TESTVELOCITYC_ESUB, TESTVELOCITYC_MSGESUB,      \
           "Function call \"" #func "\" failed:" );                  \
    return TESTVELOCITYC_ESUB;                                  \
  }                                                                  \
} while (0)
/******************************************************************/

/* A global pointer for debugging. */
#ifndef NDEBUG
char *lalWatch;
#endif


/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */
/* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv------------------------------------ */
int main(int argc, char *argv[]){ 

  static LALStatus       status; 
  static VelocityPar     velPar;
  static REAL8           vel[3];
  static  EphemerisData  *edat = NULL;
  INT4   arg;                         /* Argument counter */
  REAL8  vTol=0.01 ;
  /* INT4   c, errflg=0;*/
  /*  optarg = NULL; */
  /* ------------------------------------------------------- */

 
  /* default values */
  velPar.detector = lalCachedDetectors[LALDetectorIndexGEO600DIFF];
  velPar.startTime.gpsSeconds = T0SEC;
  velPar.startTime.gpsNanoSeconds = T0NSEC;
  velPar.tBase = TBASE;
  velPar.vTol = ACCURACY;  

 /********************************************************/
  /* Parse argument list.  i stores the current position. */
  /********************************************************/
  arg = 1;
  while ( arg < argc ) {
    /* Parse debuglevel option. */
    if ( !strcmp( argv[arg], "-d" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
        lalDebugLevel = atoi( argv[arg++] );
      } else {
        ERROR( TESTVELOCITYC_EARG, TESTVELOCITYC_MSGEARG, 0 );
        LALPrintError( USAGE, *argv );
        return TESTVELOCITYC_EARG;
      }
    }
    /* Parse accuracy option. */
    else if ( !strcmp( argv[arg], "-a" ) ) {
      if ( argc > arg + 1 ) {
        arg++;
	vTol= atof( argv[arg++]);
        velPar.vTol= vTol;
      } else {
        ERROR( TESTVELOCITYC_EARG, TESTVELOCITYC_MSGEARG, 0 );
        LALPrintError( USAGE, *argv );
        return TESTVELOCITYC_EARG;
      }
    }
    /* Unrecognized option. */
    else {
      ERROR( TESTVELOCITYC_EARG, TESTVELOCITYC_MSGEARG, 0 );
      LALPrintError( USAGE, *argv );
      return TESTVELOCITYC_EARG;
    }
  } /* End of argument parsing loop. */
  /******************************************************************/

  /* read ephemeris data */
  /* ephemeris info */
  edat = (EphemerisData *)LALMalloc(sizeof(EphemerisData));
  (*edat).ephiles.earthEphemeris = EARTHDATA;
  (*edat).ephiles.sunEphemeris = SUNDATA;
  /* is this the right number of leap seconds? */
  (*edat).leap = 13;

  /* read in ephemeris data */
  SUB( LALInitBarycenter( &status, edat), &status);

  /* fill in ephemeris data in velPar */
  velPar.edat = edat;

  /*  for (i=0; i<1400; i++) */
  SUB( LALAvgDetectorVel ( &status, vel, &velPar), &status );


  printf("AvgDetectorVel= %g, %g, %g \n", vel[0],vel[1],vel[2]);
 

  LALFree(edat->ephemE);
  LALFree(edat->ephemS);
  LALFree(edat);
  LALCheckMemoryLeaks(); 

  INFO(TESTVELOCITYC_MSGENORM);
  return TESTVELOCITYC_ENORM;
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< */









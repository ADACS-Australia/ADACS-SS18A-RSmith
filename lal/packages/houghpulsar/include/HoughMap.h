/*-----------------------------------------------------------------------
 *
 * File Name: HoughMap.h
 *
 * Authors: Sintes, A.M.,  
 *
 * Revision: $Id$
 *
 * History:   Created by Sintes June 22, 2001
 *            Modified    "    August 6, 2001
 *
 *
 *-----------------------------------------------------------------------
 */


/************************************ <lalVerbatim file="HoughMapHV">
Author: Sintes, A.M., 
$Id$
************************************* </lalVerbatim> */

/* <lalLaTeX>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Header \texttt{HoughMap.h}}
\label{s:HoughMap.h}

Provides subroutines for 
initialization and construction of Hough-map derivatives and total Hough-maps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection*{Synopsis}

\begin{verbatim}
#include <lal/HoughMap.h>
\end{verbatim}

 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection*{Error conditions}
\vspace{0.1in}
\input{HoughMapHErrorTable}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection*{Structures and type definitions}


\begin{verbatim}
typedef  CHAR  HoughDT
\end{verbatim}
Hough Map derivative pixel type.


\begin{verbatim}
typedef  UCHAR HoughTT
\end{verbatim}
Total Hough Map pixel type.\\

\noindent Depending of the number of maps to accumulate
 change both types \texttt{HoughDT} and \texttt{HoughTT} to \texttt{INT2} or
 \texttt{UINT2} respectively.

\begin{verbatim}
struct HOUGHMapDeriv
\end{verbatim}
\index{\texttt{HOUGHMapDeriv}}

\noindent This structure stores the Hough map derivative.  The field is:

\begin{description} 
\item[\texttt{UINT2     xSide }] Number of physical pixels in the x direction. 
\item[\texttt{UINT2     ySide }] Number of physical pixels in the y direction.
\item[\texttt{HoughDT   *map}]  The pixel count derivatives.  
The number of elements to allocate is \verb@ySide*(xSide+1)@.
\end{description}

\begin{verbatim}
struct HOUGHMapTotal
\end{verbatim}
\index{\texttt{HOUGHMapTotal}}

\noindent This structure stores the Hough map.  The fields are:

\begin{description}
 
\item[] 

{\it General information in case we want to save results:}
 
\item[\texttt{INT8      f0Bin}]  Frequency bin for which it has been 
constructed  
\item[\texttt{REAL8     deltaF}]  Frequency resolution
\item[\texttt{UINT4     mObsCoh}]   Ratio between the  observation time and
coherent timescale 
\item[\texttt{UINT4     nPG }] Number of peakgrams used \texttt{<= mObsCoh}.
There could be gaps during the observation time.
\item[\texttt{REAL8UnitPolarCoor skyPatch }]    Coordinates of the versor $\hat
N_{center}$ (alpha, delta)  pointing to the center of the sky patch.
\item[\texttt{REAL8Vector spinDem }] Spin parameters used in the demodulation stage.
\item[\texttt{REAL8Vector spinRes }] Refined spin parameters used in the Hough
transform.  

 {\sf There should be some time info, etc}
 
 {\em Here starts what is really needed }:
 
\item[\texttt{UINT2     xSide }] Number of physical pixels in the x direction. 
\item[\texttt{UINT2     ySide }] Number of physical pixels in the y direction.
\item[\texttt{HoughTT   *map}]  The pixel counts.
The number of elements to allocate is \verb@ySide*xSide@.
\end{description}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\vfill{\footnotesize\input{HoughMapHV}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage\input{HoughMapC}

%%%%%%%%%%Test program. %%
\newpage\input{TestHoughMapC}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

</lalLaTeX> */



/*
 * 4.  Protection against double inclusion (include-loop protection)
 *     Note the naming convention!
 */

#ifndef _HOUGHMAP_H
#define _HOUGHMAP_H

/*
 * 5. Includes. This header may include others; if so, they go immediately 
 *    after include-loop protection. Includes should appear in the following 
 *    order: 
 *    a. Standard library includes
 *    b. LDAS includes
 *    c. LAL includes
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>

# include <lal/LUT.h>
# include <lal/PHMD.h>

/*
 *  #include "LALRCSID.h"
 *  not needed, it is included in "LALConstants.h"
 */



/*
 *   Protection against C++ name mangling
 */

#ifdef  __cplusplus
extern "C" {
#endif
  
/*
 * 6. Assignment of Id string using NRCSID()  
 */
  
NRCSID (HOUGHMAPH, "$Id$");
  
/*
 * 7. Error codes and messages. This must be auto-extracted for 
 *    inclusion in the documentation.
 */
  
/* <lalErrTable file="HoughMapHErrorTable"> */
  
#define HOUGHMAPH_ENULL 1
#define HOUGHMAPH_ESIZE 2
#define HOUGHMAPH_ESZMM 4
#define HOUGHMAPH_EINT  6
#define HOUGHMAPH_ESAME 8
#define HOUGHMAPH_EFREQ 10
#define HOUGHMAPH_EVAL 12

#define HOUGHMAPH_MSGENULL "Null pointer"
#define HOUGHMAPH_MSGESIZE "Invalid input size"
#define HOUGHMAPH_MSGESZMM "Size mismatch"
#define HOUGHMAPH_MSGEINT  "Invalid interval"
#define HOUGHMAPH_MSGESAME "Input/Output data vectors are the same" 
#define HOUGHMAPH_MSGEFREQ "Invalid frequency"
#define HOUGHMAPH_MSGEVAL  "Invalid value"
  
/* </lalErrTable>  */

  
/* ******************************************************
 * 8. Macros. But, note that macros are deprecated. 
 *    They could be moved to the modules where are needed 
 */
  

/* *******************************************************
 * 9. Constant Declarations. (discouraged) 
 */
 


/* **************************************************************
 * 10. Structure, enum, union, etc., typdefs.
 */
  
/*  Hough Map derivative pixel type */
typedef CHAR  HoughDT;

/* Total Hough Map pixel type */
typedef UCHAR HoughTT;
  /* Depending of the number of maps to accumulate, */
  /* if needed change both types  to INT2 or UINT2  */


typedef struct tagHOUGHMapDeriv{
  UINT2     xSide;  /* number of physical pixels in the x direction */
  UINT2     ySide;  /* number of physical pixels in the y direction */
  HoughDT   *map ;  /* the pixel count derivatives. 
  			 The number of elements to allocate is ySide*(xSide+1)* */
} HOUGHMapDeriv;


typedef struct tagHOUGHMapTotal{
  /*  >>>>>>>>>> general info in case we want to save results <<<<<<<< */
  INT8      f0Bin;    /* frequency bin for which it has been constructed */
  REAL8     deltaF;   /* frequency resolution */
  UINT4     mObsCoh; /* ratio of observation time and coherent timescale */
  UINT4     nPG;      /* <= mObsCoh number of peakgrams used */
                      /* there could be gaps during the observation time */
  REAL8UnitPolarCoor skyPatch;       /* N_center {alpha, delta } */
  REAL8Vector spinDem;       /* spin parameters used in the demodulation */ 
  REAL8Vector spinRes;          /* refined spin parameters used in Hough */ 
           /* There should be some time info, etc... */
  /* >>>>>>>>>> Here starts what I really need <<<<<<<<  */
  UINT2     xSide;       /* number of physical pixels in the x direction */
  UINT2     ySide;       /* number of physical pixels in the y direction */
  HoughTT   *map;      /* the pixel counts.  
  			The number of elements to allocate is ySide*xSide */
} HOUGHMapTotal;

  

/*
 * 11. Extern Global variables. (discouraged) 
 */
  

/*
 * 12. Functions Declarations (i.e., prototypes).
 */

void LALHOUGHInitializeHD (LALStatus      *status,
			  HOUGHMapDeriv   *hd /* the Hough map derivative */
			  );  

void LALHOUGHAddPHMD2HD (LALStatus      *status,
			 HOUGHMapDeriv  *hd,  /* the Hough map derivative */
			 HOUGHphmd      *phmd  /* info from a partial map */ 
			 );  

void LALHOUGHIntegrHD2HT (LALStatus       *status,
			  HOUGHMapTotal   *ht,     /* the total Hough map */
			  HOUGHMapDeriv   *hd /* the Hough map derivative */
			  );  
void LALHOUGHInitializeHT (LALStatus      *status,
			  HOUGHMapTotal   *ht,     /* the total Hough map */
			  HOUGHPatchGrid  *patch      /* patch information */
			  );

#ifdef  __cplusplus
}                /* Close C++ protection */
#endif

#endif     /* Close double-include protection _HOUGHMAP_H */

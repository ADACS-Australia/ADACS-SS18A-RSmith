/*-----------------------------------------------------------------------
 *
 * File Name: TFClusters.h
 *
 * Author: Julien Sylvestre
 *
 * Revision: $Id$ 
 *
 *-----------------------------------------------------------------------
 *
 * NAME
 * TFClusters.h
 *
 * SYNOPSIS
 * #include "TFClusters.h"
 *
 * DESCRIPTION
 * Function to analyze a spectrogram by clustering analysis in order to
 * detect transients. The output is a list of events, significant according
 * to thresholds on the clusters size in the spectrogram, and the total power
 * in a cluster.
 *
 * DIAGNOSTICS
 *
 *-----------------------------------------------------------------------
 */

/************************************ <lalVerbatim file="TFClustersH">
Author: Sylvestre, J.
$Id$
************************************* </lalVerbatim> */


#ifndef _TFCLUSTERS_H
#define _TFCLUSTERS_H

#include <lal/LALDatatypes.h>
#include <lal/LALStatusMacros.h>
#include <lal/Window.h>
#include <lal/TFTransform.h>
#include <lal/LALMalloc.h>
#include <lal/LALError.h>
#include <lal/Thresholds.h>
#include "lal/LALRCSID.h"

#include <stdlib.h>
#include <math.h>

#ifdef  __cplusplus   /* C++ protection. */
extern "C" {
#endif


NRCSID (TFCLUSTERSH, "$Id$");


  /******************************** <lalErrTable file="TFClustersHErrTab"> */
#define TFCLUSTERSH_ENULLP       1
#define TFCLUSTERSH_ENNULLP      2
#define TFCLUSTERSH_ESTRICTPOS    4
#define TFCLUSTERSH_EPOS    8
#define TFCLUSTERSH_EINCOMP 16
#define TFCLUSTERSH_EMALLOC 32
#define TFCLUSTERSH_ENZERO 64
#define TFCLUSTERSH_E01 128
#define TFCLUSTERSH_EIARG 256
#define TFCLUSTERSH_EMAXITE 512
#define TFCLUSTERSH_MSGENULLP "Null pointer"
#define TFCLUSTERSH_MSGENNULLP "Non-null pointer"
#define TFCLUSTERSH_MSGESTRICTPOS "Argument must be strictly positive"
#define TFCLUSTERSH_MSGEPOS "Argument must be positive"
#define TFCLUSTERSH_MSGEINCOMP "Time Series parameteres incompatible with requested time-frequency parameters"
#define TFCLUSTERSH_MSGEMALLOC "Memory allocation error"
#define TFCLUSTERSH_MSGENZERO "Non-zero parameter"
#define TFCLUSTERSH_MSGE01 "Argument must be in [0,1]"
#define TFCLUSTERSH_MSGEIARG "Invalid Argument"
#define TFCLUSTERSH_MSGEMAXITE "Maximum number of iteration exceeded"
/*************************************************** </lalErrTable> */



/*************************************<lalLaTeX file="TFClustersStructs">
\subsubsection*{struct \texttt{Spectrogram}}
\noindent A container for the power in the spectrogram.
\begin{description}
\item[\texttt{TFPlaneParams *params}] Parameters of the spectrogram.
\item[\texttt{REAL8* power}] A pointer to the power vector: power at time index \texttt{i} and frequency index \texttt{j} is given by \texttt{power[i*params->freqBins + j]}.
\end{description}
******************************************************* </lalLaTeX> */

typedef struct tagSpectrogram {
  REAL8 *power;
  TFPlaneParams *params;} Spectrogram;

/*************************************<lalLaTeX file="TFClustersStructs">
\subsubsection*{struct \texttt{CList}}
\noindent A container for the clusters that are detected.
\begin{description}
\item[\texttt{UINT4 nclusters}] The number of clusters.
\item[\texttt{UINT4* sizes}] Vector of cluster sizes.
\item[\texttt{UINT4** t}] Time coordinates: time index of \texttt{j}$^{\rm th}$ pixel in \texttt{i}$^{\rm th}$ cluster is \texttt{t[i][j]}.
\item[\texttt{UINT4** f}] Frequency coordinates: frequency index of \texttt{j}$^{\rm th}$ pixel in \texttt{i}$^{\rm th}$ cluster is \texttt{f[i][j]}.
\item[\texttt{REAL8** P}] Instantaneous power: power of \texttt{j}$^{\rm th}$ pixel in \texttt{i}$^{\rm th}$ cluster is \texttt{P[i][j]}.
\item[\texttt{TFPlaneParams *params}] Parameters of the spectrogram.
\end{description}
******************************************************* </lalLaTeX> */

typedef struct tagCList
{
  UINT4 nclusters; /* number of clusters */

  UINT4 *sizes,
    **t,
    **f;

  REAL8 **P;

  TFPlaneParams *params;
} CList;


/*************************************<lalLaTeX file="TFClustersStructs">
\subsubsection*{struct \texttt{CListDir}}
\noindent A container for the various thresholds
\begin{description}
\item[\texttt{UINT4 freqBins}] Number of frequency bins in spectrogram.
\item[\texttt{REAL8* rho}] Vector of size \texttt{freqBins} containing the threshold on the power for the first cut, as a function of the frequency index.
\item[\texttt{REAL8 minf}] Minimum frequency in Hertz to be considered in the analysis.
\item[\texttt{REAL8 maxf}] Maximum frequency in Hertz to be considered in the analysis. All clusters with at least one pixel outside of [minf, maxf] are rejected.
\item[\texttt{UINT4 sigma}] Threshold $\sigma$ on cluster size.
\item[\texttt{UINT4* s1,s2}] Size pairs for distance threshold.
\item[\texttt{UINT4* d}] Vector of distance thresholds $\delta_{s_1,s_2}$. For a certain value of \texttt{i} in $[0,\sigma(\sigma-1)/2-1]$, $\delta_{s_1,s_2} = $\texttt{d[i]} for $s_1 = $\texttt{s1[i]} and $s_2 = $\texttt{s2[i]}.
\item[\texttt{UINT4 mdist}] Maximum values of \texttt[d]. 
\item[\texttt{REAL8 alpha}] For white Gaussian noise at input, fraction of clusters that pass the last cut on the total cluster power.
\end{description}
******************************************************* </lalLaTeX> */

typedef struct tagCListDir
{
  
  REAL8 minf, maxf;

  UINT4 sigma;

  UINT4 mdist;

  UINT4 *s1, *s2, *d;

  UINT4 freqBins;
  REAL8 *rho;

  REAL8 alpha;

} CListDir;


/****************Main functions********************/

void 
LALComputeSpectrogram (
		       LALStatus *status, 
		       Spectrogram *out, 
		       TFPlaneParams *tspec, 
		       REAL4TimeSeries *tseries
		       );

void 
LALComputeXSpectrogram (
			LALStatus *status, 
			Spectrogram *out, 
			TFPlaneParams *tspec, 
			REAL4TimeVectorSeries *tseries
			);

void 
LALGetClusters (
		LALStatus *status, 
		CList *clist, 
		Spectrogram *tpower, 
		CListDir *dir
		);


void 
LALClustersPowerThreshold (
			   LALStatus *status, 
			   CList *out,
			   CList *in, 
			   CListDir *dir
			   );



/********************Utilities**********************/

void
LALMergeClusterLists (
		      LALStatus *status, 
		      CList *out,
		      CList *A, 
		      CList *B
		      );

void
LALCopyCList (
	      LALStatus *status, 
	      CList *dest,
	      CList *src
	      );

void 
LALPlainSpectrogram(
		    LALStatus *status,
		    TFPlaneParams *tspec,
		    REAL4TimeSeries *tseries,
		    REAL8 T
		    );

void 
LALPlainSpectrogramWin(
		    LALStatus *status,
		    TFPlaneParams *tspec,
		    REAL4TimeSeries *tseries,
		    REAL8 T
		    );

void 
LALInitCList (
	      LALStatus *status,
	      CList *clist,
	      TFPlaneParams *tspec
	      );


void 
LALFillCListDir (
		 LALStatus *status,
		 CListDir *cldir,
		 REAL8 rho
		 );


void 
LALFreeCList(
	     LALStatus *status, 
	     CList *tlist
	     );


void 
LALFreeSpecgram(
		LALStatus *status, 
		Spectrogram *power
		);


void
LALFreeCListDir (
		 LALStatus *status,
		 CListDir *cdir
		 );



#ifdef  __cplusplus
}
#endif  /* C++ protection. */

#endif

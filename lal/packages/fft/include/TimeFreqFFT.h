/**** <lalVerbatim file="TimeFreqFFTHV">
 * $Id$
 **** </lalVerbatim> */

/**** <lalLaTeX>
 *
 * \section{Header \texttt{TimeFreqFFT.h}}
 * \label{s:TimeFreqFFT.h}
 * 
 * Performs real-to-complex, complex-to-real FFTs and average power
 * spectrum estimation.
 * 
 * \subsection*{Synopsis}
 * \begin{verbatim}
 * #include <lal/TimeFreqFFT.h>
 * \end{verbatim}
 * 
 * \noindent Perform time-to-frequency and frequency-to-time fast Fourier
 * transforms. Also provides a function to compute mean and median power
 * spectra with user specified windowning.
 * 
 **** </lalLaTeX> */

#ifndef _TIMEFREQFFT_H
#define _TIMEFREQFFT_H

#include <lal/LALDatatypes.h>
#include <lal/ComplexFFT.h>
#include <lal/RealFFT.h>
#include <lal/Window.h>

#ifdef  __cplusplus
extern "C" {
#pragma }
#endif

NRCSID( TIMEFREQFFTH, "$Id$" );

/**** <lalLaTeX>
 * \subsection*{Error conditions}
 **** </lalLaTeX> */
/**** <lalErrTable> */

#define TIMEFREQFFTH_ENULL 1
#define TIMEFREQFFTH_ESIZE 2
#define TIMEFREQFFTH_ERATE 4
#define TIMEFREQFFTH_ESIGN 4
#define TIMEFREQFFTH_EALLOC 16
#define TIMEFREQFFTH_EPOSARG 32
#define TIMEFREQFFTH_EMALLOC 64
#define TIMEFREQFFTH_EINCOMP 128
#define TIMEFREQFFTH_ENNUL 256
#define TIMEFREQFFTH_EZSEG 512
#define TIMEFREQFFTH_EZOVR 1024
#define TIMEFREQFFTH_EMISM 2048
#define TIMEFREQFFTH_EUAVG 4096

#define TIMEFREQFFTH_MSGENULL "Null pointer"
#define TIMEFREQFFTH_MSGESIZE "Invalid size"
#define TIMEFREQFFTH_MSGERATE "Invalid rate"
#define TIMEFREQFFTH_MSGESIGN "Incorrect plan sign"
#define TIMEFREQFFTH_MSGEALLOC "Pointer has already been allocated, should be null"
#define TIMEFREQFFTH_MSGEPOSARG "Argument must be positive"
#define TIMEFREQFFTH_MSGEMALLOC "Malloc failure"
#define TIMEFREQFFTH_MSGEINCOMP "Incompatible arguments"
#define TIMEFREQFFTH_MSGENNUL "Non-null pointer"
#define TIMEFREQFFTH_MSGEZSEG "Segment length is zero"
#define TIMEFREQFFTH_MSGEZOVR "Overlap length is zero"
#define TIMEFREQFFTH_MSGEMISM "Mismatch beteen segment, overlap and data length"
#define TIMEFREQFFTH_MSGEUAVG "Unknown average power spectum method"

/**** </lalErrTable> */

/* XXX this should be removed XXX */
typedef struct tagRealDFTParams
{
  WindowType               windowType;
  REAL4Vector              *window;
  REAL4                    sumofsquares;
  RealFFTPlan              *plan;
}
RealDFTParams;

/* <lalLaTeX>
\subsection*{Types}

\subsubsection*{Enum type \texttt{AvgSpecMethod}}
\idx[Type]{AvgSpecMethod}

This type determines the method the type of average that will be used to
compute the power sperum estimate by the \verb|LALREAL4AverageSpectrum()|
function. The function computes a series of (possibly overlapping) power
spectra and computes the average using one of the following methods:

\begin{description}
\item[\texttt{useUnity}] A constant PSD of value unity will be returned
independent of the input data given. This is used for testing purposes.

\item[\texttt{useMean}] The arithmetic mean of the individual power spectra 
computed will be used to compute the output power spectrum.

\item[\texttt{useMedian}] The median value of the individual power spectra 
computed will be used to compute the output power spectrum.
\end{description}

</lalLaTeX> */

typedef enum
{
  useUnity,
  useMean,
  useMedian
} 
AvgSpecMethod;

/* <lalLaTeX>
\subsubsection*{Structure \texttt{AvgerageSpectrumParams}}
\idx[Type]{AverageSpectrumParams}

This structure controls the behaviour of the \verb|LALREAL4AverageSpectrum()|
function.

\begin{description}
\item[\texttt{REAL4Window *window}] The windowing function to use when
computing the individual power spectra from the input time series. The
input time series is broken into smaller time series to compute power spectra
for the estimate. The legth of these time series is determined by the
\texttt{length} parameter of the window vector.

\item[\texttt{UINT4 overlap}] The overlap between sucessive time series used
to compute the power spectra.

\item[\texttt{AvgSpecMethod method}] The method of computing the average
as describe above.

\item[\texttt{RealFFTPlan }] The FFT plan to be used in the computation
of the power spectrum.

\end{description}

</lalLaTeX> */

typedef struct
tagAverageSpectrumParams
{
  REAL4Window          *window;
  UINT4                 overlap;
  AvgSpecMethod         method;
  RealFFTPlan          *plan;
}
AverageSpectrumParams;

/**** <lalLaTeX>
 * \newpage\input{TimeFreqFFTC}
 * \newpage\input{TimeFreqFFTTestC}
 **** </lalLaTeX> */

/* XXX this should be removed XXX */
void
LALCreateRealDFTParams ( 
        LALStatus                         *status, 
        RealDFTParams                  **dftParams, 
        LALWindowParams                   *params,
        INT2                           sign
        );

/* XXX this should be removed XXX */
void
LALDestroyRealDFTParams (
        LALStatus                        *status, 
        RealDFTParams                 **dftParams
        );

void
LALTimeFreqRealFFT(
    LALStatus               *status,
    COMPLEX8FrequencySeries *freq,
    REAL4TimeSeries         *time,
    RealFFTPlan             *plan
    );

void
LALFreqTimeRealFFT(
    LALStatus               *status,
    REAL4TimeSeries         *time,
    COMPLEX8FrequencySeries *freq,
    RealFFTPlan             *plan
    );

void
LALRealAverageSpectrum(
    LALStatus            *status,
    REAL4FrequencySeries *fSeries,
    REAL4TimeSeries      *tSeries,
    RealDFTParams        *params,
    AvgSpecMethod              method
    );

void
LALREAL4AverageSpectrum (
    LALStatus                   *status,
    REAL4FrequencySeries        *fSeries,
    REAL4TimeSeries             *tSeries,
    AverageSpectrumParams       *params
    );

void
LALTimeFreqComplexFFT(
    LALStatus               *status,
    COMPLEX8FrequencySeries *freq,
    COMPLEX8TimeSeries      *time,
    ComplexFFTPlan          *plan
    );

void
LALFreqTimeComplexFFT(
    LALStatus               *status,
    COMPLEX8TimeSeries      *time,
    COMPLEX8FrequencySeries *freq,
    ComplexFFTPlan          *plan
    );

#ifdef  __cplusplus
#pragma {
}
#endif

#endif /* _TIMEFREQFFT_H */

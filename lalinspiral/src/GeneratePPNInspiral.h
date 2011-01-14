/*
*  Copyright (C) 2007 Philip Charlton, Duncan Brown, Jolien Creighton, David McKechan, Stephen Fairhurst, Teviet Creighton, Thomas Cokelaer, John Whelan
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

/************************** <lalVerbatim file="GeneratePPNInspiralHV">
Author: Creighton, T. D.
$Id$
**************************************************** </lalVerbatim> */

/********************************************************** <lalLaTeX>

\section{Header \texttt{GeneratePPNInspiral.h}}
\label{s:GeneratePPNInspiral.h}

Provides routines to generate restricted parametrized
post${}^{5/2}$-Newtonian inspiral waveforms.

\subsection*{Synopsis}
\begin{verbatim}
#include <lal/GeneratePPNInspiral.h>
\end{verbatim}

This header covers routines to generate a ``restricted'' parametrized
post${}^{5/2}$-Newtonian binary inspiral waveform in the time domain.
That is, the calculation of the wave phase is accurate to
post${}^{5/2}$-Newtonian order (including corrections up to order
$v^5/c^5$, where $v$ is the orbital speed), but the wave amplitudes
are accurate only to leading (post${}^0$-Newtonian) order.
Furthermore, at each order the post${}^{n/2}$-Newtonian correction can
be turned on, off, or set to an unphysical value, by adjusting a
parameter $p_n$.

The post-Newtonian expansion implicitly assumes an \emph{adiabatic}
inspiral, where one can represent the waveform by an ``instantaneous''
amplitude and frequency that vary over timescales longer than one wave
period.  The \emph{orbital} frequency of the system to
post${}^{5/2}$-Newtonian order is given in Eqs.~6.4.1 and~6.9.1
of~\cite{GRASP_1.9.8:2000}; here we work entirely in terms of the
\emph{gravitational-wave} frequency, which is twice the orbital
frequency:
\begin{eqnarray}
f(t) & = & \frac{M_\odot}{8\pi T_\odot m_\mathrm{tot}}\left\{
	p_0\Theta^{-3/8}+
	p_1\Theta^{-1/2}+
	p_2\left(\frac{743}{2688}+\frac{11}{32}\eta\right)\Theta^{-5/8}-
	p_3\frac{3\pi}{10}\Theta^{-3/4} \right. \nonumber \\
\label{eq:ppn-freq}
& & \left.+ p_4\left(\frac{1855099}{14450688}+\frac{56975}{258048}\eta+
		\frac{371}{2048}\eta^2\right)\Theta^{-7/8}-
	p_5\left(\frac{7729}{21504}+\frac{3}{256}\eta\right)\pi\Theta^{-1}
	\right\} \; ,
\end{eqnarray}
where $M_\odot$ is the mass of the Sun,
$T_\odot=GM_\odot/c^3=4.925491\times10^{-6}$s is the ``geometrized''
solar mass in time units, $m_\mathrm{tot}=m_1+m_2$ is the total mass
of the binary, $\eta=m_1m_2/m_\mathrm{tot}^2$ is the (symmetric) mass
ratio parameter, and $\Theta$ is a dimensionless time parameter:
\begin{equation}
\label{eq:ppn-theta}
\Theta(t) = \frac{\eta M_\odot}{5T_\odot m_\mathrm{tot}}(t_c-t) \; .
\end{equation}
Here $t_c$ is the time of coalescence of the two masses in the
point-mass approximation.  The post-Newtonian parameters $p_k$ are
defined such that in a normal (physical) post${}^{n/2}$-Newtonian
expansion, one sets $p_1=0$ and $p_{k>n}=0$, and $p_k=1$ for all other
$k$.  However, changing this convention can be used to model in an
approximate way things such as spin, eccentricity, or non-GR theories
of gravity.  We also note that while most terms are normalized to
their normal post-Newtonian values, the normalization on the $p_1$
term is completely arbitrary, since it is zero in a normal
post-Newtonian expansion.

The wave phase as a function of time can be computed analytically from
Eq.~\ref{eq:ppn-freq} as $\phi_\mathrm{orb}=2\pi\int f\,dt$:
\begin{eqnarray}
\phi(t) & = & \phi_c - \frac{2}{\eta}\left\{
	p_0\Theta^{5/8}+
	p_1\frac{5}{4}\Theta^{1/2}+
	p_2\left(\frac{3715}{8064}+\frac{55}{96}\eta\right)\Theta^{3/8}-
	p_3\frac{3\pi}{4}\Theta^{1/4} \right. \nonumber \\
\label{eq:ppn-phi}
& & \left.+ p_4\left(\frac{9275495}{14450688}+\frac{284875}{258048}\eta+
		\frac{1855}{2048}\eta^2\right)\Theta^{1/8}-
	p_5\left(\frac{38645}{172032}+\frac{15}{2048}\eta\right)\pi
		\log\left(\frac{\Theta}{\Theta_0}\right)\right\} \; .
\end{eqnarray}
Here $\Theta_0$ is an arbitrary constant; changing it is equivalent to
changing $\phi_c$.  We note that the post${}^{5/2}$-Newtonian term
introduces a late-time divergence in phase which renders meaningless
the interpretation of $\phi_c$ as ``phase at coalescence''; in our
convention we define $\phi_c$ to correspond to the case $\Theta_0=1$.

We refer the interested reader to Sec.~6.6 of~\cite{GRASP_1.9.8:2000}
for a discussion of how propagation effects shift the phase of the
waveform relative to the orbital phase.  To summarize, though: A
changing propagation delay does introduce a time-dependent phase shift
in the waveform, but the dependence on $t$ is weak except at very late
times; although it looks like a post${}^{3/2}$-Newtonian phase
correction, it can in fact be represented as a post${}^{3}$-Newtonian
phase correction combined with a post${}^{3/2}$-Newtonian amplitude
correction.  Since we are concerned with \emph{restricted}
post${}^{5/2}$-Newtonian waveforms, which model the amplitude only to
leading (post${}^0$-Newtonian) order, we can ignore these propagation
effects.

To leading order, then, the amplitude of the + and $\times$
polarizations of the wave are given by Eqs.~6.6.1--6.6.4
of~\cite{GRASP_1.9.8:2000} as:
\begin{eqnarray}
\label{eq:ppn-aplus}
A_+(t) & = & -\frac{2T_\odot c}{D}(1+\cos^2 i)
	\left(\frac{\eta m_\mathrm{tot}}{M_\odot}\right)
	\left[\frac{\pi T_\odot	m_\mathrm{tot}f(t)}{M_\odot}
	\right]^{2/3} \; , \\
\label{eq:ppn-across}
A_\times(t) & = & -\frac{2T_\odot c}{D}(2\cos i)
	\left(\frac{\eta m_\mathrm{tot}}{M_\odot}\right)
	\left[\frac{\pi T_\odot	m_\mathrm{tot}f(t)}{M_\odot}
	\right]^{2/3} \; ,
\end{eqnarray}
where $D$ is the distance to the source and $i$ is the inclination of
the axis of the source to the line of sight.  The normal polarization
convention in~\cite{Will_C:1996} is used, where the reference
$x$-coordinate axis for the + and $\times$ polarization tensors is the
ascending node of the rotational plane as it crosses the plane
transverse to the propagation direction.  This convention implies that
the + and $\times$ waveforms are elliptically polarized as follows:
\begin{eqnarray}
\label{eq:ppn-hplus}
h_+(t) & = & A_+(t)\cos\phi(t) \; , \\
\label{eq:ppn-hcross}
h_\times(t) & = & A_\times(t)\sin\phi(t) \; .
\end{eqnarray}

******************************************************* </lalLaTeX> */

#ifndef _GENERATEPPNINSPIRAL_H
#define _GENERATEPPNINSPIRAL_H

#include <lal/LALStdlib.h>
#include <lal/SimulateCoherentGW.h>
#include <lal/SkyCoordinates.h>
#include <lal/Random.h>

#ifdef  __cplusplus
extern "C" {
#pragma }
#endif

NRCSID( GENERATEPPNINSPIRALH, "$Id$" );

/********************************************************** <lalLaTeX>
\subsection*{Error conditions}
****************************************** </lalLaTeX><lalErrTable> */
#define GENERATEPPNINSPIRALH_ENUL  1
#define GENERATEPPNINSPIRALH_EOUT  2
#define GENERATEPPNINSPIRALH_ETBAD 3
#define GENERATEPPNINSPIRALH_EFBAD 4
#define GENERATEPPNINSPIRALH_EPBAD 5
#define GENERATEPPNINSPIRALH_EMBAD 6
#define GENERATEPPNINSPIRALH_EDBAD 7
#define GENERATEPPNINSPIRALH_EMEM  8

#define GENERATEPPNINSPIRALH_MSGENUL  "Unexpected null pointer in arguments"
#define GENERATEPPNINSPIRALH_MSGEOUT  "output field a, f, phi, or shift already exists"
#define GENERATEPPNINSPIRALH_MSGETBAD "Bad sampling interval"
#define GENERATEPPNINSPIRALH_MSGEFBAD "Bad starting frequency; could not get valid start time"
#define GENERATEPPNINSPIRALH_MSGEPBAD "Bad post-Newtonian parameters"
#define GENERATEPPNINSPIRALH_MSGEMBAD "Bad masses"
#define GENERATEPPNINSPIRALH_MSGEDBAD "Bad distance"
#define GENERATEPPNINSPIRALH_MSGEMEM  "Out of memory"
/******************************************** </lalErrTable><lalLaTeX>
\subsection*{Termination conditions}

In addition to the error conditions above, there are a number of ways
that the signal generation routine can terminate gracefully while
still returning a valid waveform.  In many cases one \emph{wants} to
continue generating a waveform ``until things fall apart''; the
following codes, returned in the \verb@PPNParamStruc@ below, allow the
waveform generator to report exactly \emph{how} things fell apart.

For the sake of LAL namespace conventions, these termination codes are
\verb@#define@d and autodocumented exactly like error codes.
****************************************** </lalLaTeX><lalErrTable> */
#define GENERATEPPNINSPIRALH_EFSTOP     0
#define GENERATEPPNINSPIRALH_ELENGTH    1
#define GENERATEPPNINSPIRALH_EFNOTMON   2
#define GENERATEPPNINSPIRALH_EPNFAIL    3
#define GENERATEPPNINSPIRALH_ERTOOSMALL 4

#define GENERATEPPNINSPIRALH_MSGEFSTOP     "Reached requested termination frequency"
#define GENERATEPPNINSPIRALH_MSGELENGTH    "Reached maximum length, or end of provided time series vector"
#define GENERATEPPNINSPIRALH_MSGEFNOTMON   "Frequency no longer increasing monotonically"
#define GENERATEPPNINSPIRALH_MSGEPNFAIL    "Evolution dominated by higher-order PN terms"
#define GENERATEPPNINSPIRALH_MSGERTOOSMALL "Orbital radius too small for PN approximation"
/******************************************** </lalErrTable><lalLaTeX>

\subsection*{Types}

\subsubsection*{Structure \texttt{PPNParamStruc}}
\idx[Type]{PPNParamStruc}

This structure stores the parameters for constructing a restricted
post-Newtonian waveform.  It is divided into three parts: parameters
passed along to the output structure but not used by waveform
generator, parameters used as input to the waveform generator, and
parameters set by the generator to evaluate its success.

\bigskip\noindent\textit{Passed fields:}
\begin{description}
\item[\texttt{SkyPosition position}] The location of the source on the
sky, normally in equatorial coordinates.

\item[\texttt{REAL4 psi}] The polarization angle of the source, in
radians.

\item[\texttt{LIGOTimeGPS epoch}] The start time of the output series.
\end{description}

\medskip\noindent\textit{Input fields:}
\begin{description}
\item[\texttt{REAL4 mTot}] The total mass $m_\mathrm{tot}=m_1+m_2$ of
the binary system, in solar masses.

\item[\texttt{REAL4 eta}] The mass ratio
$\eta=m_1m_2/m_\mathrm{tot}^2$ of the binary system.  Physically this
parameter must lie in the range $\eta\in(0,1/4]$; values outside of
this range may be permitted in order to represent ``nonphysical''
post-Newtonian expansions.

\item[\texttt{REAL4 d}] The distance to the system, in metres.

\item[\texttt{REAL4 inc}] The inclination of the system to the line of
sight, in radians.

\item[\texttt{REAL4 phi}] The phase at coalescence $\phi_c$ (or
arbitrary reference phase for a post${}^{5/2}$-Newtonian
approximation), in radians.

\item[\texttt{REAL8 deltaT}] The requested sampling interval of the
waveform, in s.

\item[\texttt{REAL4 fStartIn}] The requested starting frequency of the
waveform, in Hz.

\item[\texttt{REAL4 fStopIn}] The requested termination frequency of
the waveform, in Hz.  If set to 0, the waveform will be generated
until a termination condition (above) is met.  If set to a negative
number, the generator will use its absolute value as the terminating
frequency, but will ignore post-Newtonian breakdown; it will terminate
only at the requested frequency $-\mathtt{fStopIn}$, a local maximum
frequency, or the central singularity.

\item[\texttt{UINT4 lengthIn}] The maximum number of samples in the
generated waveform.  If zero, the waveforms can be arbitrarily long.

\item[\texttt{REAL4Vector *ppn}] The parameters $p_n$ selecting the
type of post-Newtonian expansion.  If \verb@ppn@=\verb@NULL@, a
``normal'' (physical) expansion is assumed.
\end{description}

\medskip\noindent\textit{Output fields:}
\begin{description}
\item[\texttt{REAL8 tc}] The time $t_c-t$ from the start of the
waveform to coalescence (in the point-mass approximation), in s.

\item[\texttt{REAL4 dfdt}] The maximum value of $\Delta f\Delta t$
encountered over any timestep $\Delta t$ used in generating the
waveform.

\item[\texttt{REAL4 fStart}] The actual starting frequency of the
waveform, in Hz (normally close but not identical to \verb@fStartIn@).

\item[\texttt{REAL4 fStop}] The frequency at the termination of the
waveform, in Hz.

\item[\texttt{INT4 length}] The length of the generated waveform.

\item[\texttt{INT4 termCode}] The termination condition (above) that
stopped computation of the waveform.

\item[\texttt{const CHAR *termDescription}] The termination code
description (above).
\end{description}

******************************************************* </lalLaTeX> */

/* <lalVerbatim file="LALInputAxisH">  */
typedef enum {
  View,
  OrbitalL,
  TotalJ
 } InputAxis;
/* </lalVerbatim>  */

/* <lalLaTeX>
\idx[Type]{InputAxis}
</lalLaTeX>  */


typedef struct tagPPNParamStruc {
  /* Passed parameters. */
  SkyPosition position; /* location of source on sky */
  REAL4 psi;            /* polarization angle (radians) */
  LIGOTimeGPS epoch;    /* start time of output time series */

  /* Input parameters. */
  REAL4 mTot;       /* total system mass (Msun) */
  REAL4 eta;        /* mass ratio */
  REAL4 d;          /* distance (metres) */
  REAL4 inc;        /* inclination angle (radians) */
  REAL4 phi;        /* coalescence phase (radians) */
  REAL8 deltaT;     /* requested sampling interval (s) */
  REAL4 fStartIn;   /* requested start frequency (Hz) */
  REAL4 fStopIn;    /* requested stop frequency (Hz) */
  UINT4 lengthIn;   /* maximum length of waveform */
  REAL4Vector *ppn; /* post-Newtonian selection parameters */
  INT4 ampOrder;    /* PN amplitude selection 0-5 */

  /* Output parameters. */
  REAL8 tc;         /* time to coalescence from start of waveform */
  REAL4 dfdt;       /* maximum value of df*dt over any timestep */
  REAL4 fStart;     /* actual start frequency (Hz) */
  REAL4 fStop;      /* actual stop frequency (Hz) */
  UINT4 length;     /* length of signal generated */
  INT4 termCode;    /* termination code */
  InputAxis axisChoice; /* z axis of the reference frame*/
  const CHAR *termDescription; /* description of termination code */
} PPNParamStruc;

/********************************************************** <lalLaTeX>

\subsubsection*{Structure \texttt{GalacticInspiralParamStruc}}
\idx[Type]{GalacticInspiralParamStruc}

This structure stores the position and mass parameters of a galactic
inspiral event.  The fields are:

\begin{description}
\item[\texttt{REAL4 rho}] The distance of the binary system from the
Galactic axis, in kpc.

\item[\texttt{REAL4 z}] The distance of the system from the Galactic
plane, in kpc.

\item[\texttt{REAL4 lGal}] The Galactocentric Galactic longitude of
the system (i.e.\ the Galactic longitude of the direction \emph{from
the Galactic centre} through the system), in radians.
See~\verb@SkyCoordinates.h@ for the definition of this quantity.

\item[\texttt{REAL4 m1, m2}] The masses of the binary components, in
solar masses.

\item[\texttt{LIGOTimeGPS geocentEndTime}] The geocentric end time of
the inspiral event.
\end{description}

******************************************************* </lalLaTeX> */

typedef struct tagGalacticInspiralParamStruc {
  REAL4 rho;    /* Galactocentric axial radius (kpc) */
  REAL4 z;      /* Galactocentric axial height (kpc) */
  REAL4 lGal;   /* Galactocentric longitude (radians) */
  REAL4 m1, m2; /* system masses (solar masses) */
  LIGOTimeGPS geocentEndTime; /* geocentric end time */
} GalacticInspiralParamStruc;


/* <lalLaTeX>
\vfill{\footnotesize\input{GeneratePPNInspiralHV}}
</lalLaTeX> */

typedef struct tagAmpSwitchStruc {
	UINT4 q0, q1, q2, q3, q4, q5;
} AmpSwitchStruc;



/* Function prototypes. */

/* <lalLaTeX>
\newpage\input{GeneratePPNInspiralC}
</lalLaTeX> */
void
LALGeneratePPNInspiral( LALStatus     *,
			CoherentGW    *output,
			PPNParamStruc *params );

/* <lalLaTeX>
\newpage\input{GeneratePPNAmpCorInspiralC}
</lalLaTeX> */
void
LALGeneratePPNAmpCorInspiral( LALStatus     *,
			CoherentGW    *output,
			PPNParamStruc *params );



/* <lalLaTeX>
\newpage\input{GetInspiralParamsC}
</lalLaTeX> */
void
LALGetInspiralParams( LALStatus                  *,
		      PPNParamStruc              *output,
		      GalacticInspiralParamStruc *input,
		      RandomParams               *params );

/* <lalLaTeX>
\newpage\input{GenerateInspiralSmoothC}
</lalLaTeX> */
void
LALGenerateInspiralSmooth( LALStatus            *,
		      	   CoherentGW		**output,
			   PPNParamStruc	*params,
			   REAL4		*qfactor);

/* <lalLaTeX>
\newpage\input{GeneratePPNInspiralTestC}
</lalLaTeX> */

#ifdef  __cplusplus
#pragma {
}
#endif

#endif /* _GENERATEPPNINSPIRAL_H */

/*  <lalVerbatim file="LALInspiralComputeParamsCV">
Author: Churches, D. K.
$Id$
</lalVerbatim>  */

/*  <lalLaTeX>

\subsection{Module \texttt{LALInspiralComputeParams.c}}

This function takes as input $\tau_{0}$, $\tau_{3}$ and $f_a$ (the lower frequency of the detectors
sensitivity). It then calculates $m$ (the total mass of the binary), $\eta$ (the
symmetric mass ratio) and the individual mass of the compact objects. 


\subsubsection*{Prototypes}
\vspace{0.1in}
\input{LALInspiralComputeParamsCP}
\index{\texttt{LALInspiralComputeParams()}}

\subsubsection*{Description}

We start with the definition of the chirp times $\tau_{0}$ and $\tau_{3}$,
\begin{equation}
\tau_{0} = \frac{5}{256 (\pi f_{a} )^{8/3} m^{5/3} \eta}
\end{equation}

and

\begin{equation}
\tau_{3} = \frac{1}{8 (\pi^{2} f_{a}^{3} )^{1/3} m^{2/3} \eta}
\end{equation}
 These equations may be inverted to yield
\begin{equation}
m = \frac{5}{32 \pi^{2} f_{a}} \frac{\tau_{3}}{\tau_{0}}
\end{equation}

and

\begin{equation}
\eta = \left( \frac{2 \pi^{2}}{25 f_{a}^{3}} \frac{\tau_{0}^{2}}{\tau_{3}^{5}}
\right)^{5}\end{equation}

The individual masses may be calculated as follows.

We have

\begin{equation}
m = m_{1} + m_{2}
\label{mass}
\end{equation}

and

\begin{equation}
\eta = \frac{m_{1} m_{2}}{(m_{1} + m_{2})^{2}}
\label{eta}
\end{equation}


From Eq.(\ref{mass}) we may eliminate either $m_{1}$ or $m_{2}$,
 
\begin{equation}
m_{1} = m - m_{2}
\end{equation}
 
This may be substituted into Eq.(\ref{eta}) to give
 
\begin{equation}
\eta = \frac{(m - m_{2}) m_{2}}{\left[ (m - m{2}) + m_{2} \right]^{2}}
\end{equation}
 
which may be re--arranged to give
 
\begin{equation}
m_{2}^{2} - m m_{2} + \eta m^{2} = 0
\end{equation}
 
i.e.\
 
\begin{equation}
m_{2} = \frac{ m \pm \sqrt{m^{2}(1 - 4 \eta) }}{2}
\end{equation}
 
Therefore, since we know that $\eta \leq 1/4$, real roots are guaranteed.
If we had eliminated $m_{2}$ rather than $m_{1}$ then we would have arrived at an identical
expression for
$m_{1}$, and so of one object has mass
 
\begin{equation}
m_{1} = \frac{m + \sqrt{m^{2}(1-4 \eta)}}{2}
\end{equation}
 
then the other object must have mass
 
\begin{equation}
m_{2} = \frac{m - \sqrt{m^{2}(1-4 \eta)}}{2}
\end{equation}


\subsubsection*{Algorithm}


\subsubsection*{Uses}
LALInspiralParameterCalc
\subsubsection*{Notes}

\vfill{\footnotesize\input{LALInspiralComputeParamsCV}}

</lalLaTeX>  */


#include <lal/LALInspiralBank.h>

NRCSID (LALINSPIRALCOMPUTEPARAMSC, "$Id$");

/*  <lalVerbatim file="LALInspiralComputeParamsCP"> */
void 
LALInspiralComputeParams(
   LALStatus               *status,
   InspiralTemplate     *pars, 
   InspiralBankParams   bankParams,
   InspiralCoarseBankIn coarseIn
)
{ /*  </lalVerbatim>  */


  INITSTATUS (status, "LALInspiralComputeParams", LALINSPIRALCOMPUTEPARAMSC);
  ATTATCHSTATUSPTR(status);

  pars->fLower = coarseIn.fLower;

  switch (coarseIn.space) {
     case Tau0Tau2:
        pars->t0 = bankParams.x0;
        pars->t2 = bankParams.x1;
        pars->massChoice = t02;
        break;
     case Tau0Tau3:
        pars->t0 = bankParams.x0;
        pars->t3 = bankParams.x1;
        pars->massChoice = t03;
        break;
     default:
        fprintf(stderr, "LALInspiralComputeParams: No choice for parameter space");
        exit(0);
  }
  LALInspiralParameterCalc(status->statusPtr, pars);
  CHECKSTATUSPTR(status);

  DETATCHSTATUSPTR(status);
  RETURN (status);
}

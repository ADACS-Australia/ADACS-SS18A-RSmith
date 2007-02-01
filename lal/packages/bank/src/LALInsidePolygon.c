/*  <lalVerbatim file="LALInsidePolygonCV">
Author: Cokelaer. T
$Id$
</lalVerbatim>  */

/*  <lalLaTeX>

\subsection{Module \texttt{LALInsidePolygon.c}}

Module to check whether a point with coordinates (x0,y0) is inside 
a polygon defined by the vectors (vx, vy), which size (n) must be 
provided. The functions returns 1 if the point is inside or 0 otherwise. 

\subsubsection*{Prototypes}
\vspace{0.1in}
\input{LALInsidePolygonCP}
\idx{LALInsidePolygon()}
\begin{itemize}
   \item \texttt{vx, vy} Input, two arrays of floats defining the polygon. 
   \item \texttt{n} Input, the size of the vectors.
   \item \texttt{x0, y0} Input, the coordinate of the point.
   \item \texttt{valid} Output, 0 if outside and 1 if inside.
\end{itemize}

\subsubsection*{Description/Algorithm}
None
\subsubsection*{Uses}
None.
\subsubsection*{Notes}
Tested in matlab codes and some BCV tests within lal/lalapps.

\vfill{\footnotesize\input{LALInsidePolygonCV}}

</lalLaTeX>  */



#include <lal/LALInspiralBank.h>

NRCSID (LALINSIDEPOLYGONC, "Id: $");

/*  <lalVerbatim file="LALInsidePolygonCP"> */
void LALInsidePolygon(  LALStatus          *status,
                        REAL4              *inputx,
                        REAL4              *inputy,
                        INT4               n,
                        REAL4              x0,
                        REAL4              y0,
                        INT4               *valid)

                             
{  /*  </lalVerbatim>  */


   INITSTATUS (status, "LALInsidePolygon", LALINSIDEPOLYGONC);
   ATTATCHSTATUSPTR(status);
   ASSERT (n>=3,  status, LALINSPIRALBANKH_ENULL, LALINSPIRALBANKH_MSGENULL);
   
   {
     int i, j, c = 0;
     for (i = 0, j = n-1; i < n; j = i++) {
       if ((((inputy[i] <= y0) && (y0 < inputy[j])) ||
	    ((inputy[j] <= y0) && (y0 < inputy[i]))) &&
	   (x0 < (inputx[j] - inputx[i]) * (y0 - inputy[i]) / (inputy[j] - inputy[i]) + inputx[i]))
	 c = !c;
     }
     *valid = c;
   }
   
   DETATCHSTATUSPTR(status);
   RETURN(status);

}

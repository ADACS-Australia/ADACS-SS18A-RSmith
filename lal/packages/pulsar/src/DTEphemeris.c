/******************************* <lalVerbatim file="DTEphemerisCV">
Author: Jones, D. I.,   Owen, B. J.
$Id$
**************************************************** </lalVerbatim> */

/********************************************************** <lalLaTeX>

\subsection{Module \texttt{DTEphemeris.c}}
\label{ss:DTEphemeris.c}

Computes the barycentric arrival time of an incoming wavefront using
accurate ephemeris-based data files of the Sun and Earth's motions.

\subsubsection*{Prototypes}
\vspace{0.1in}
\input{DTEphemerisCP}
\idx{LALDTEphemeris()}

\subsubsection*{Description}


These routines compute the barycentric time transformation and its
derivatives.  That is, if a signal originating from a right ascension
$\alpha$ and declination $\delta$ on the sky and arrives at the
detector at a time $t$, then it will pass the centre of the solar
system at a time $t_b(t,\alpha,\delta)$.

The input/output features of this function are identical to those of
\texttt{DTBaryPtolemaic()}, whose documentation should be consulted
for the details.  The only differnce lies in the computation itself:
\texttt{DTBaryPtolemaic()} uses the Ptolemaic approximation to model
the Earth/Sun system, while \texttt{DTEphemeris()} uses accurate
ephemeris data read in from files in the calling function, and passed
into \texttt{DTEphemeris()} using the \texttt{EphemerisData}
structure, which is a member of the \texttt{PulsarTimesParamStruc}.

\subsubsection*{Algorithm}

\subsubsection*{Uses}
\begin{verbatim}
lalDebugLevel                LALBarycenterEarth()
LALBarycenter()
\end{verbatim}

\subsubsection*{Notes}

\vfill{\footnotesize\input{DTEphemerisCV}}

******************************************************* </lalLaTeX> */

#include <math.h>
#include <stdio.h>
#include <lal/AVFactories.h>
#include <lal/LALBarycenter.h>
#include <lal/PtoleMetric.h>
#include <lal/StackMetric.h>


NRCSID(DTEPHEMERISC,"$Id$");

/* <lalVerbatim file="DTEphemerisCP"> */
void
LALDTEphemeris( LALStatus             *status,
	        REAL8Vector           *drv,
	        REAL8Vector           *var,
	        PulsarTimesParamStruc *tev )
{ /* </lalVerbatim> */
  LIGOTimeGPS tGPS;           /* Input structure to BartcenterEarth()  */
  EphemerisData *eph;         /* Input structure to BarycenterEarth()  */
  EarthState earth;           /* Output structure of BarycenterEarth() */
                              /* and input structure to Barycenter()   */
  BarycenterInput baryin;     /* Input structure for Barycenter()      */
  EmissionTime emit;          /* Output structure of Barycenter()      */
  REAL8 d_alpha;              /* alpha step size for finite differencing */
  REAL8 d_delta;              /* delta step size for finite differencing */
  REAL8 upper, lower;         /* Quantities for finite differnecing */

  INITSTATUS(status,"DTEphemeris",DTEPHEMERISC);
  ATTATCHSTATUSPTR( status );


  /* This function may be called a lot.  Do error checking only in
     debug mode. */ 
#ifndef NDEBUG
  if(lalDebugLevel){
    /* Make sure parameter structures and their fields exist. */
    ASSERT(drv,status,PULSARTIMESH_ENUL,PULSARTIMESH_MSGENUL);
    ASSERT(drv->data,status,PULSARTIMESH_ENUL,PULSARTIMESH_MSGENUL);
    ASSERT(var,status,PULSARTIMESH_ENUL,PULSARTIMESH_MSGENUL);
    ASSERT(var->data,status,PULSARTIMESH_ENUL,PULSARTIMESH_MSGENUL);
    ASSERT(tev,status,PULSARTIMESH_ENUL,PULSARTIMESH_MSGENUL);
    /* Make sure array sizes are consistent. */
    ASSERT(drv->length==var->length+1,status,
	   PULSARTIMESH_EBAD,PULSARTIMESH_MSGEBAD);
    ASSERT(var->length>2,status,
	   PULSARTIMESH_EBAD,PULSARTIMESH_MSGEBAD);
  }
#endif


  /* First compute the location, velocity, etc... of the Earth:  */

  /* Set the GPS time: */
  tGPS.gpsSeconds = floor( var->data[0] ) + tev->epoch.gpsSeconds;
  tGPS.gpsNanoSeconds = 1e9*fmod( var->data[0], 1.0 ) 
    + tev->epoch.gpsNanoSeconds;
 
  /* Set the ephemeris data: */
  eph = &(tev->ephemeris);

  TRY( LALBarycenterEarth( status->statusPtr, &earth, &tGPS, eph ), status );
  /* Now "earth" contains position of center of Earth. */



  /* Now do the barycentering.  Set the input parameters: */

  /* Get time delay for detector vertex. */
  baryin.tgps.gpsSeconds = tGPS.gpsSeconds;
  baryin.tgps.gpsNanoSeconds = tGPS.gpsNanoSeconds;

  /* Set the detector site...*/
  baryin.site = tev->site;

  /* ...remembering to divide the coordinates by the speed of light: */
  baryin.site.location[0] = lalCachedDetectors[LALDetectorIndexGEO600DIFF].location[0]/LAL_C_SI;
  baryin.site.location[1] = lalCachedDetectors[LALDetectorIndexGEO600DIFF].location[1]/LAL_C_SI;
  baryin.site.location[2] = lalCachedDetectors[LALDetectorIndexGEO600DIFF].location[2]/LAL_C_SI;
  baryin.alpha = var->data[1];
  baryin.delta = var->data[2];

  /* Set 1/distance to zero: */
  baryin.dInv = 0.e0;


  TRY( LALBarycenter( status->statusPtr, &emit, &baryin, &earth ), status );
  /* Now "emit" contains detector position, velocity, time, tdot. */


  /* Now assemble output: */

  /* This is the barycentered GPS time: */
  drv->data[0] = emit.te.gpsSeconds + 1e-9*emit.te.gpsNanoSeconds; 
  /* Subtract off t_gps: */
  drv->data[0] -= tev->epoch.gpsSeconds + 1.0e-9*tev->epoch.gpsNanoSeconds;

  /* This is the derivative of d(barycentered time)/d(detector time): */ 
  drv->data[1] = emit.tDot; /* dtb/dt */

  /* Need to finite difference to get d(tb)/d(alpha), d(tb)/d(delta) */
  /* Set finite difference step sizes: */
  d_alpha = 0.001;
  d_delta = 0.001;    

  /* Get dtb/da by finite differencing.   */
  /* Default upper and lower alpha values: */
  upper = var->data[1] + d_alpha;
  lower = var->data[1] - d_alpha;
  /* Overwrite if alpha is too close to zero or 2 PI: */
  if(var->data[1] < d_alpha)
    lower = var->data[1]; 
  if(var->data[1] > (LAL_TWOPI-d_alpha))
    upper = var->data[1]; 
  /* Evaluate emit at upper value: */
  baryin.alpha = upper;
  TRY( LALBarycenter( status->statusPtr, &emit, &baryin, &earth ), status );
  drv->data[2] = emit.te.gpsSeconds + 1e-9*emit.te.gpsNanoSeconds;
  /* Evaluate emit at lower value: */
  baryin.alpha = lower;
  TRY( LALBarycenter( status->statusPtr, &emit, &baryin, &earth ), status );
  drv->data[2] -= emit.te.gpsSeconds + 1e-9*emit.te.gpsNanoSeconds;
  /* Divide by alpha interval: */
  drv->data[2] /= (upper-lower);
  baryin.alpha = var->data[1];

  /* Get dtb/dd by finite differencing.   */
  /* Default upper and lower alpha values: */
  upper = var->data[2] + d_delta;
  lower = var->data[2] - d_delta;
  /* Overwrite if delta is too close to PI/2 or -PI/2: */
  if(var->data[2] < (-LAL_PI_2+d_alpha))
    lower = var->data[2]; 
  if(var->data[2] > (LAL_PI_2-d_alpha))
    upper = var->data[2]; 
  /* Evaluate emit at upper value: */
  baryin.delta = upper;
  TRY( LALBarycenter( status->statusPtr, &emit, &baryin, &earth ), status );
  drv->data[3] = emit.te.gpsSeconds + 1e-9*emit.te.gpsNanoSeconds;
  /* Evaluate emit at lower value: */
  baryin.delta = lower;
  TRY( LALBarycenter( status->statusPtr, &emit, &baryin, &earth ), status );
  drv->data[3] -= emit.te.gpsSeconds + 1e-9*emit.te.gpsNanoSeconds;
  /* Divide by delta interval: */
  drv->data[3] /= (upper-lower);
  baryin.alpha = var->data[2];

  /* Go home */
  DETATCHSTATUSPTR( status );
  RETURN( status );
}

/************************************ <lalVerbatim file="GeneralMetricTestCV">
Author: Jones, D. I.,     Owen, B. J.
$Id$
********************************************************** </lalVerbatim> */

/**************************************************************** <lalLaTeX>

\subsection{Program \texttt{GeneralMetricTest.c}}
\label{ss:GeneralMetricTest}

Tests the various LAL metric functions, by outputting the metric at a
point in parameter space, and also producing an array of ellipses of
constant mismatch.

\subsubsection*{Usage}
\begin{verbatim}
GeneralMetricTest
\end{verbatim}

\subsubsection*{Description}

This program computes metric components using a metric function of the
user's specification.  The ordering of the components is $(f_0,
\alpha, \delta)$ for the unprojected metric, and $(\alpha, \delta)$
for the metric with $f_0$ projected out.

With no options, this program displays metric components for a single point
in parameter space for the default parameter values.

The \texttt{-a} option determines which LAL metric code is used.  The
options are: 

\hspace{1cm} 1 = PtoleMetric (default), 

\hspace{1cm} 2 = (CoherentMetric \& DTBaryPtolemaic), 

\hspace{1cm} 3 = (CoherentMetric \& DTEphemeris).


The \texttt{-b} option sets the beginning GPS time of integration to
the option argument. (Default is $731265908$ seconds, chosen to lie
within the S2 run).

The \texttt{-c} option determines the point on the sky where the metric
is evaluated.  This option is hard-coded to use equatorial coordinates
and the argument should be given in hh:mm:ss:dd:mm:ss format.
(Default is the center of the globular cluster 47 Tuc).

The \texttt{-d} option sets the detector to the option argument. The
options are: 
 
\hspace{1cm} 1 = LIGO Hanford 

\hspace{1cm} 2 = LIGO Livingston

\hspace{1cm} 3 = VIRGO 

\hspace{1cm} 4 = GEO600 (default)

\hspace{1cm} 5 = TAMA300 

The \texttt{-e} option sets the LAL debug level to 1.  (The default is 0).

The \texttt{-l} option determines the limits in right ascension and
declination of the rectangular region over which the mismatch contours
are computed.  The argument should be given in degrees as
RA(min):RA(max):dec(min):dec(max).  (The default is the octant of the
sky defined by $0 < {\rm RA} 90$ and $0< {\rm dec} 85$; this avoids the
coordinate singularity at the poles.)


The \texttt{-m} option sets the mismatch (default is $0.02$).

The \texttt{-p} option is provided for users who wish to view the
power mismatch contours provided by the \texttt{-x} option (see below)
but don't have xmgrace installed.  All necessary data is simply
written to a file ``nongrace.data''; it's probably best to look at the
code to see the exact format.  The user should then write a small
script to convert the data into the format appropriate to their
favorite graphics package.

The \texttt{-t} option sets the duration of integration in seconds. (The 
default is $39600$ seconds $= 11$ hours, which is chosen because it is of 
the right size for S2 analyses).

The \texttt{-x} option produces a graph of the 2\% power mismatch
contours on the sky. Dimensions of the ellipses have been exaggerated
by a factor of \texttt{MAGNIFY} (specified within the code) for
legibility. The purpose of the graph is to get a feel for how ellipses
are varying across parameter space. Note that this option makes a
system call to the \texttt{xmgrace} program, and will not work if that
program is not installed on your system.


\subsubsection*{Exit Codes}
************************************************ </lalLaTeX><lalErrTable> */
#define GENERALMETRICTESTC_EMEM 1
#define GENERALMETRICTESTC_ESUB 2
#define GENERALMETRICTESTC_ESYS 3
#define GENERALMETRICTESTC_EMET 4
 
#define GENERALMETRICTESTC_MSGEMEM "memory (de)allocation error"
#define GENERALMETRICTESTC_MSGESUB "subroutine failed"
#define GENERALMETRICTESTC_MSGESYS "system call failed"
#define GENERALMETRICTESTC_MSGEMET "determinant of projected metric negative"

/************************************************** </lalErrTable><lalLaTeX>
\subsubsection*{Algorithm}

\subsubsection*{Uses}

\begin{verbatim}
lalDebugLevel
LALCheckMemoryLeaks()
LALCreateVector()
LALDestroyVector()
LALDCreateVector()
LALDDestroyVector()
LALProjectMetric()
LALPtoleMetric()
xmgrace
\end{verbatim}

\subsubsection*{Notes}

For most regions of parameter space the three metric codes seem to
agree well.  However, for short (less than one day) runs, they are all
capable of returning (unphysical) negative determinant metrics for
points very close to the equator.



\vfill{\footnotesize\input{GeneralMetricTestCV}}

************************************************************* </lalLaTeX> */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <lal/AVFactories.h>
#include <lal/LALMalloc.h>
#include <lal/LALStdlib.h>
#include <lal/PtoleMetric.h>
#include <lal/StackMetric.h>
#include <lal/LALBarycenter.h>
#include <lal/LALInitBarycenter.h>

extern char *optarg;

NRCSID( GENERALMETRICTESTC, "$Id" );

#define DEFAULT_DURATION 39600 /* seconds */
#define NUM_SPINDOWN 0         /* No. of spindown parameters: Not yet in use */
#define SPOKES 30
#define MAGNIFY 1.0            /* Magnification factor of ellipses */

int lalDebugLevel = 0;

int main( int argc, char *argv[] ) {
  static LALStatus status;          /* Status structure */
  PtoleMetricIn    in;              /* PtoleMetric() input structure */
  REAL8            mismatch;        /* mismatch threshold of mesh */
  REAL8Vector     *metric;          /* Parameter-space metric */
  int              j, k;            /* Parameter-space indices */
  int              opt;             /* Command-line option. */
  BOOLEAN          grace;           /* Whether or not we use xmgrace */
  BOOLEAN          nongrace;        /* Whether or not to output data to file*/
  int              ra, dec, i;      /* Loop variables for xmgrace option */
  FILE            *pvc = NULL;      /* Temporary file for xmgrace option */
  FILE            *fnongrace = NULL;/* File contaning ellipse coordinates */
  int              metric_code;     /* Which metric code to use: */
                                    /* 1 = Ptolemetric */
                                    /* 2 = CoherentMetric + DTBarycenter */
                                    /* 3 = CoherentMetric + DTEphemeris  */
  REAL8Vector     *tevlambda;       /* (f, a, d) input for CoherentMetric */
  MetricParamStruc tevparam;        /* Input structure for CoherentMetric */
  PulsarTimesParamStruc tevpulse;   /* Input structure for CoherentMetric */
                                    /* (this is a member of tevparam) */
  EphemerisData   *eph;             /* To store ephemeris data */
  int             detector;         /* Which detector to use: */
                                    /* 1 = Hanford,  2 = Livingston,  */
                                    /* 3 = Virgo,  4 = GEO,  5 = TAMA */
  REAL8           ra_point;         /* RA at which metric is evaluated */
  REAL8           dec_point;        /* dec at which metric is evaluated */
  float           a,b,c,d,e,f;      /* To input point in standard format */
  int             ra_min, ra_max;   /* Min and max RA for ellipse plot */
  int             dec_min, dec_max; /* Min and max dec for ellipse plot */
  float           c_ellipse;        /* Centers of ellipses */
  float           r_ellipse;        /* Radii of ellipses */
  REAL8           determinant;      /* Determinant of projected metric */
  REAL4           f0;               /* carrier frequency */

  /* Defaults that can be overwritten: */
  metric_code = 1;
  in.epoch.gpsSeconds = tevpulse.epoch.gpsSeconds = 731265908;
  in.epoch.gpsNanoSeconds = tevpulse.epoch.gpsNanoSeconds = 0.0;
  mismatch = 0.02;
  nongrace = 0;
  in.duration = tevparam.deltaT = DEFAULT_DURATION;
  grace = 0;
  detector = 4;
  ra_point  = (24.1/60)*LAL_PI_180;     /* 47 Tuc */
  dec_point = -(72+5./60)*LAL_PI_180;
  ra_min = 0;
  ra_max = 90;
  dec_min = 0;
  dec_max = 85;
  f0 = 1000;


  /* Parse options. */
  while ((opt = getopt( argc, argv, "a:b:c:d:ef:l:m:pt:x" )) != -1) {
    switch (opt) {
    case 'a':
      metric_code = atoi( optarg );
      break;
    case 'b':
      in.epoch.gpsSeconds = tevpulse.epoch.gpsSeconds = atoi( optarg );
      break;
    case 'c':
      if( sscanf( optarg, "%f:%f:%f:%f:%f:%f", &a, &b, &c, &d, &e, &f ) != 6)
	{
	  fprintf( stderr, "coordinates should be hh:mm:ss:dd:mm:ss\n" );
	}
      ra_point = (15*a+b/60+c/3600)*LAL_PI_180;
      dec_point = (d+e/60+f/3600)*LAL_PI_180;
      break;
    case 'd':
      detector = atoi( optarg );
      break;
    case 'e':
      lalDebugLevel = 1;
      break;
    case 'f':
      f0 = atof( optarg );
      break;
    case 'l':
      if( sscanf( optarg, "%d:%d:%d:%d", 
		  &ra_min, &ra_max, &dec_min, &dec_max) != 4)
	{
	  fprintf( stderr, "coordinates should be ra_min, ra_max, dec_min, dec_max, all in degrees" );
	}
      break;
    case 'm':
      mismatch = atof( optarg );
      break;
    case 'p':
      nongrace = 1;
      break;
    case 't':
      in.duration = tevparam.deltaT = atof( optarg );
      break;
    case 'x':
      grace = 1;
      break;
    }
  }

  /* Allocate storage for output metric. */
  metric = NULL;
  LALDCreateVector( &status, &metric, 6 );
  if( status.statusCode )
    {
      printf( "%s line %d: %s\n", __FILE__, __LINE__,
              GENERALMETRICTESTC_MSGEMEM );
      return GENERALMETRICTESTC_EMEM;
    }
  tevlambda = NULL;
  LALDCreateVector( &status, &tevlambda, 3 );
  if( status.statusCode )
    {
      printf( "%s line %d: %s\n", __FILE__, __LINE__,
              GENERALMETRICTESTC_MSGEMEM );
      return GENERALMETRICTESTC_EMEM;
    }


  /* Other constants */

  /* Communal constants */
  in.position.longitude = tevlambda->data[1] = ra_point;
  in.position.latitude = tevlambda->data[2] = dec_point;
  in.maxFreq = tevlambda->data[0] = f0;

  /* Detector choice */
  if(detector==1)
    tevpulse.site = lalCachedDetectors[LALDetectorIndexLHODIFF];
  if(detector==2)
    tevpulse.site = lalCachedDetectors[LALDetectorIndexLLODIFF];
  if(detector==3)
    tevpulse.site = lalCachedDetectors[LALDetectorIndexVIRGODIFF];
  if(detector==4)
    tevpulse.site = lalCachedDetectors[LALDetectorIndexGEO600DIFF];
  if(detector==5)
    tevpulse.site = lalCachedDetectors[LALDetectorIndexTAMA300DIFF];

  in.site = tevpulse.site.frDetector;
  tevpulse.latitude = in.site.vertexLatitudeRadians;
  tevpulse.longitude = in.site.vertexLongitudeRadians;


  /* Ptolemetric constants */
  in.position.system = COORDINATESYSTEM_EQUATORIAL;
  in.spindown = NULL;

  /* CoherentMetric constants */
  tevparam.constants = &tevpulse;
  tevparam.n = 1;
  tevparam.errors = 0;
  tevparam.start = 0; /* start time relative to epoch */
  tevpulse.t0 = 0.0;  /* Irrelavant */
 
  /* To fill in the fields tevpulse.tMidnight & tevpulse.tAutumn */
  LALGetEarthTimes( &status, &tevpulse );
  if( status.statusCode )
    {
      printf( "%s line %d: %s\n", __FILE__, __LINE__,
              GENERALMETRICTESTC_MSGESUB );
      return GENERALMETRICTESTC_ESUB;
    }

   /* Read in ephemeris data from files: */
   eph = (EphemerisData *)LALMalloc(sizeof(EphemerisData));
   eph->ephiles.earthEphemeris = "earth00-04.dat";
   eph->ephiles.sunEphemeris = "sun00-04.dat";
   eph->leap = 13; /* right number for the years 2000-2004 */


   LALInitBarycenter( &status, eph );
   if( status.statusCode )
    {
      printf( "%s line %d: %s\n", __FILE__, __LINE__,
              GENERALMETRICTESTC_MSGESUB );
      return GENERALMETRICTESTC_ESUB;
    }

   tevpulse.ephemeris = *eph;

   /* Choose CoherentMetric timing function */
   if(metric_code==2)
     tevparam.dtCanon = LALDTBaryPtolemaic;
   if(metric_code==3)
     tevparam.dtCanon = LALDTEphemeris;


   /* Print results for a single  ponit */
    
   printf("\n3-dim metric (f, alpha, delta) at the requested point\n");
   if(metric_code==1)
     {
       LALPtoleMetric( &status, metric, &in );
       if( status.statusCode )
	 {
	   printf( "%s line %d: %s\n", __FILE__, __LINE__,
		   GENERALMETRICTESTC_MSGESUB );
	   return GENERALMETRICTESTC_ESUB;
	 }
     }
   if(metric_code==2  || metric_code==3)
     {
       LALCoherentMetric( &status, metric, tevlambda, &tevparam );
       if( status.statusCode )
	 {
	   printf( "%s line %d: %s\n", __FILE__, __LINE__,
		   GENERALMETRICTESTC_MSGESUB );
	   return GENERALMETRICTESTC_ESUB;
	 }
     }
   for (j=0; j<=2+NUM_SPINDOWN; j++) {
     for (k=0; k<=j; k++)
       printf( "  %+.3e", metric->data[k+j*(j+1)/2] );
     printf("\n");
   }
   LALProjectMetric( &status, metric, 0 );
   if( status.statusCode )
     {
       printf( "%s line %d: %s\n", __FILE__, __LINE__,
	       GENERALMETRICTESTC_MSGESUB );
       return GENERALMETRICTESTC_ESUB;
     }
   
   determinant = metric->data[5]*metric->data[2]-pow(metric->data[4],2.0);
   if(determinant < 0.0)
     {
       printf( "%s line %d: %s\n", __FILE__, __LINE__,
	       GENERALMETRICTESTC_MSGEMET );
       return GENERALMETRICTESTC_EMET;
     }

   printf("\n2-dim metric (alpha, delta) at the requested point\n");
   for (j=1; j<=2+NUM_SPINDOWN; j++) {
     for (k=1; k<=j; k++)
       printf( "  %+.3e", metric->data[k+j*(j+1)/2] );
     printf( "\n" );
      }
    



    


  /* Here is the code that uses xmgrace with the -x option, */
  /* and outputs data to a file with the -t option. */
  if (grace || nongrace) {

    /* Take care of preliminaries. */
    if(grace)
      {
	pvc = popen( "xmgrace -pipe", "w" );
	if( !pvc )
	  {
	    printf( "%s line %d: %s\n", __FILE__, __LINE__,
		    GENERALMETRICTESTC_MSGESYS );
	    return GENERALMETRICTESTC_ESYS;
	  }
	fprintf( pvc, "@xaxis label \"Right ascension (degrees)\"\n" );
	fprintf( pvc, "@yaxis label \"Declination (degrees)\"\n" );
      }
    if(nongrace)
      {
	fnongrace = fopen( "nongrace.data", "w" );
	if( !fnongrace )
	  {
	    printf( "%s line %d: %s\n", __FILE__, __LINE__,
		    GENERALMETRICTESTC_MSGESYS );
	    return GENERALMETRICTESTC_ESYS;
	  }
      }

    /* Step around the sky: a grid in ra and dec. */
    j = 0;
    for (dec=dec_max; dec>=dec_min; dec-=10) {
      for (ra=ra_min; ra<=ra_max; ra+=15) {
        REAL8 gaa, gad, gdd, angle, smaj, smin;
 
        /* Get the metric at this ra, dec. */
        in.position.longitude = tevlambda->data[1] = ra*LAL_PI_180;
        in.position.latitude  = tevlambda->data[2] = dec*LAL_PI_180;
	
	/* Evaluate metric: */
	if(metric_code==1)
	  {
	    LALPtoleMetric( &status, metric, &in );
	    if( status.statusCode )
	      {
		printf( "%s line %d: %s\n", __FILE__, __LINE__,
			GENERALMETRICTESTC_MSGESUB );
		return GENERALMETRICTESTC_ESUB;
	      }
	  }
	if(metric_code==2  || metric_code==3)
	  {
	    LALCoherentMetric( &status, metric, tevlambda, &tevparam );
	    if( status.statusCode )
	      {
		printf( "%s line %d: %s\n", __FILE__, __LINE__,
			GENERALMETRICTESTC_MSGESUB );
		return GENERALMETRICTESTC_ESUB;
	      }
	  }

	/*  Project metric: */ 
	LALProjectMetric( &status, metric, 0 );
	if( status.statusCode )
	  {
          printf( "%s line %d: %s\n", __FILE__, __LINE__,
                  GENERALMETRICTESTC_MSGESUB );
          return GENERALMETRICTESTC_ESUB;
	  }
	determinant = metric->data[5]*metric->data[2]-pow(metric->data[4],2.0);
	if(determinant < 0.0)
	  {
	    printf( "%s line %d: %s\n", __FILE__, __LINE__,
		    GENERALMETRICTESTC_MSGEMET );
	    return GENERALMETRICTESTC_EMET;
	  }



        /* Rename \gamma_{\alpha\alpha}. */
        gaa = metric->data[2];
        /* Rename \gamma_{\alpha\delta}. */
        gad = metric->data[4];
        /* Rename \gamma_{\delta\delta}. */
        gdd = metric->data[5];
        /* Semiminor axis from larger eigenvalue of metric. */
        smin = gaa+gdd + sqrt( pow(gaa-gdd,2) + pow(2*gad,2) );
        smin = sqrt(2*mismatch/smin);
        /* Semiminor axis from smaller eigenvalue of metric. */
        smaj = gaa+gdd - sqrt( pow(gaa-gdd,2) + pow(2*gad,2) );
	/*printf("ra = %d, dec = %d, temp = %g\n", ra, dec, smaj);*/
        smaj = sqrt(2*mismatch/smaj);
        /* Angle of semimajor axis with "horizontal" (equator). */
        angle = atan2( gad, mismatch/smaj/smaj-gdd );
        if (angle <= -LAL_PI_2) angle += LAL_PI;
        if (angle > LAL_PI_2) angle -= LAL_PI;
 
        if(grace)
	  {
	    /* Print set header. */
	    fprintf( pvc, "@s%d color (0,0,0)\n", j );
	    fprintf( pvc, "@target G0.S%d\n@type xy\n", j++ );
	    /* Print center of patch. */
	    fprintf( pvc, "%16.8g %16.8g\n", (float)ra, (float)dec );
	  }
	if(nongrace)
	  /* Print center of patch. */
	  fprintf( fnongrace, "%16.8g %16.8g\n", (float)ra, (float)dec );   
	/* Loop around patch ellipse. */
        for (i=0; i<=SPOKES; i++) {
          c_ellipse = LAL_TWOPI*i/SPOKES;
          r_ellipse = MAGNIFY*LAL_180_PI*smaj*smin / 
	    sqrt( pow(smaj*sin(c_ellipse),2) + pow(smin*cos(c_ellipse),2) );
	  if(grace)
	    fprintf( pvc, "%e %e\n", ra+r_ellipse*cos(angle-c_ellipse), 
		     dec+r_ellipse*sin(angle-c_ellipse) );
	  if(nongrace)
	    fprintf( fnongrace, "%e %e\n", ra+r_ellipse*cos(angle-c_ellipse), 
		     dec+r_ellipse*sin(angle-c_ellipse) );

        } /* for (a...) */
 
      } /* for (ra...) */
    } /* for (dec...) */
    if(grace)
      fclose( pvc );
    if(nongrace)
      fclose( fnongrace );
  } /* if (grace || nongrace) */

  printf("\nCleaning up and leaving...\n");

  LALFree( eph->ephemE );
  if( status.statusCode )
  {
    printf( "%s line %d: %s\n", __FILE__, __LINE__,
            GENERALMETRICTESTC_MSGEMEM );
    return GENERALMETRICTESTC_EMEM;
  }
  LALFree( eph->ephemS );
  if( status.statusCode )
  {
    printf( "%s line %d: %s\n", __FILE__, __LINE__,
            GENERALMETRICTESTC_MSGEMEM );
    return GENERALMETRICTESTC_EMEM;
  }
 LALFree( eph );
 if( status.statusCode )
  {
    printf( "%s line %d: %s\n", __FILE__, __LINE__,
            GENERALMETRICTESTC_MSGEMEM );
    return GENERALMETRICTESTC_EMEM;
  }

  LALDDestroyVector( &status, &metric );
  if( status.statusCode )
  {
    printf( "%s line %d: %s\n", __FILE__, __LINE__,
            GENERALMETRICTESTC_MSGEMEM );
    return GENERALMETRICTESTC_EMEM;
  }
  LALDDestroyVector( &status, &tevlambda );
  if( status.statusCode )
  {
    printf( "%s line %d: %s\n", __FILE__, __LINE__,
            GENERALMETRICTESTC_MSGEMEM );
    return GENERALMETRICTESTC_EMEM;
  }

  if( status.statusCode )
  {
    printf( "%s line %d: %s\n", __FILE__, __LINE__,
            GENERALMETRICTESTC_MSGEMEM );
    return GENERALMETRICTESTC_EMEM;
  }
  LALCheckMemoryLeaks();
  return 0;
} /* main() */

#if 0  /* autodoc block */

<lalVerbatim file="RandomCV">
$Id$
</lalVerbatim>

<lalLaTeX>
\subsection{Module \texttt{Random.c}}
\label{ss:Random.c}

Functions for generating random numbers.

\subsubsection*{Prototypes}
\vspace{0.1in}
\input{RandomCP}
\idx{LALCreateRandomParams()}
\idx{LALDestroyRandomParams()}
\idx{LALUniformDeviate()}
\idx{LALNormalDeviates()}

\subsubsection*{Description}

The routines \verb+LALCreateRandomParams()+ and \verb+LALDestroyRandomParams()+
create and destroy a parameter structure for the generation of random
variables.  The creation routine requires a random number seed \verb+seed+.
If the seed is zero then a seed is generated using the current time.

The routine \verb+LALUniformDeviate()+ returns a single random deviate
distributed uniformly between zero and unity.  

The routine \verb+LALNormalDeviates()+ fills a vector with normal (Gaussian)
     deviates with zero mean and unit variance, whereas the function\verb+XLALNormalDeviate+ just returns one normal distributed random number.

\subsubsection*{Operating Instructions}

\begin{verbatim}
static LALStatus     status;
static RandomParams *params;
static REAL4Vector  *vector;
UINT4 i;
INT4 seed = 0;

LALCreateVector( &status, &vector, 9999 );
LALCreateRandomParams( &status, &params, seed );

/* fill vector with uniform deviates */
for ( i = 0; i < vector->length; ++i )
{
  LALUniformDeviate( &status, vector->data + i, params );
}

/* fill vector with normal deviates */
LALNormalDeviates( &status, vector, params );

LALDestroyRandomParams( &status, &params );
LALDestroyVector( &status, &vector );
\end{verbatim}

\subsubsection*{Algorithm}

This is an implementation of the random number generators \verb+ran1+ and
\verb+gasdev+ described in Numerical Recipes~\cite{ptvf:1992}.

\subsubsection*{Uses}

\subsubsection*{Notes}
\vfill{\footnotesize\input{RandomCV}}

</lalLaTeX>

#endif /* autodoc block */


#include <time.h>
#include <math.h>
#include <lal/LALStdlib.h>
#include <lal/Random.h>
#include <lal/Sequence.h>
#include <lal/XLALError.h>

NRCSID (RANDOMC, "$Id$");

static const INT4 a = 16807;
static const INT4 m = 2147483647;
static const INT4 q = 127773;
static const INT4 r = 2836;

static const REAL4 eps = 1.2e-7;

/*
 *
 * XLAL Routines.
 *
 */

INT4 XLALBasicRandom( INT4 i )
{
  INT4 k;
  k = i/q;
  i = a*(i - k*q) - r*k;
  if (i < 0)
    i += m;
  return i;
}

RandomParams * XLALCreateRandomParams( INT4 seed )
{
  static const char *func = "XLALCreateRandomParams";
  RandomParams *params;
  UINT4 n;

  params = XLALMalloc( sizeof( *params) );
  if ( ! params )
    XLAL_ERROR_NULL( func, XLAL_ENOMEM );

  while ( seed == 0 ) /* use system clock to get seed */
    seed = time( NULL );

  if ( seed < 0 )
    seed = -seed;

  params->i = seed;
  for ( n = 0; n < 8; ++n ) /* shuffle 8 times */
    params->i = XLALBasicRandom( params->i );

  /* populate vector of random numbers */
  for ( n = 0; n < sizeof( params->v )/sizeof( *params->v ); ++n )
    params->v[n] = params->i = XLALBasicRandom( params->i );

  /* first random number is the 0th element of v */
  params->y = params->v[0];

  return params;
}


void XLALDestroyRandomParams( RandomParams *params )
{
  XLALFree( params );
}


REAL4 XLALUniformDeviate( RandomParams *params )
{
  static const char *func = "XLALUniformDeviate";
  REAL4 ans;
  INT4 ndiv;
  INT4 n;

  if ( ! params )
    XLAL_ERROR_REAL4( func, XLAL_EFAULT );

  /* randomly choose which element of the vector of random numbers to use */
  ndiv = 1 + (m - 1)/(sizeof(params->v)/sizeof(*params->v));
  n = params->y/ndiv;
  params->y = params->v[n];

  /* repopulate this element */
  params->v[n] = params->i = XLALBasicRandom( params->i );

  ans = params->y/(REAL4)m;
  if ( ans > 1 - eps ) /* make sure it is not exactly 1 */
    ans = 1 - eps;

  return ans;
}


int XLALNormalDeviates( REAL4Vector *deviates, RandomParams *params )
{
  static const char *func = "XLALNormalDeviates";
  REAL4 *data;
  INT4   half;

  if ( ! deviates || ! deviates->data || ! params )
    XLAL_ERROR( func, XLAL_EFAULT );
  if ( ! deviates->length )
    XLAL_ERROR( func, XLAL_EBADLEN );

  data = deviates->data;
  half = deviates->length/2;

  while (half-- > 0)
  {
    REAL4 u;
    REAL4 v;
    REAL4 x;
    REAL4 y;
    REAL4 rsq;
    REAL4 fac;

    do {
      u = XLALUniformDeviate( params );
      v = XLALUniformDeviate( params );
      x   = 2*u - 1;
      y   = 2*v - 1;
      rsq = x*x + y*y;
    }
    while (rsq >= 1 || rsq == 0);

    fac     = sqrt(-2*log(rsq)/rsq);
    *data++ = fac*x;
    *data++ = fac*y;
  }

  /* do it again if there is an odd amount of data */
  if (deviates->length % 2)
  {
    REAL4 u;
    REAL4 v;
    REAL4 x;
    REAL4 y;
    REAL4 rsq;
    REAL4 fac;

    do {
      u = XLALUniformDeviate( params );
      v = XLALUniformDeviate( params );
      x   = 2*u - 1;
      y   = 2*v - 1;
      rsq = x*x + y*y;
    }
    while (rsq >= 1 || rsq == 0);

    fac   = sqrt(-2*log(rsq)/rsq);
    *data = fac*x;
    /* throw away y */
  }

  return 0;
}

REAL4 XLALNormalDeviate( RandomParams *params )
{
  static const char *func = "XLALNormalDeviate";
  REAL4Sequence *deviates;
  REAL4 deviate;

  if ( ! params )
    XLAL_ERROR_REAL4( func, XLAL_EFAULT );

  /* create a vector */
  deviates = XLALCreateREAL4Sequence(1);
  if(!deviates)
    XLAL_ERROR_REAL4( func, XLAL_EFUNC );

  /* call the actual function */
  XLALNormalDeviates( deviates, params );
  deviate = deviates->data[0];

  /* destroy the vector */
  XLALDestroyREAL4Sequence(deviates);

  return deviate;
}

/*
 *
 * LAL Routines.
 *
 */


/* <lalVerbatim file="RandomCP"> */
void
LALCreateRandomParams (
    LALStatus     *status,
    RandomParams **params,
    INT4           seed
    )
{ /* </lalVerbatim> */
  INITSTATUS (status, "LALCreateRandomParams", RANDOMC);

  ASSERT (params, status, RANDOMH_ENULL, RANDOMH_MSGENULL);
  ASSERT (!*params, status, RANDOMH_ENNUL, RANDOMH_MSGENNUL);

  *params = XLALCreateRandomParams( seed );
  if ( ! params )
  {
    XLALClearErrno();
    ABORT( status, RANDOMH_ENULL, RANDOMH_MSGENULL );
  }

  RETURN (status);
}


/* <lalVerbatim file="RandomCP"> */
void
LALDestroyRandomParams (
    LALStatus     *status,
    RandomParams **params
    )
{ /* </lalVerbatim> */
  INITSTATUS (status, "LALDestroyRandomParams", RANDOMC);

  ASSERT (params, status, RANDOMH_ENULL, RANDOMH_MSGENULL);
  ASSERT (*params, status, RANDOMH_ENULL, RANDOMH_MSGENULL);

  XLALDestroyRandomParams( *params );
  *params = NULL;

  RETURN (status);
}


/* <lalVerbatim file="RandomCP"> */
void
LALUniformDeviate (
    LALStatus    *status,
    REAL4        *deviate,
    RandomParams *params
    )
{ /* </lalVerbatim> */
  INITSTATUS (status, "LALUniformDeviate", RANDOMC);

  ASSERT (deviate, status, RANDOMH_ENULL, RANDOMH_MSGENULL);
  ASSERT (params, status, RANDOMH_ENULL, RANDOMH_MSGENULL);

  *deviate = XLALUniformDeviate( params );
  if ( XLAL_IS_REAL4_FAIL_NAN( *deviate ) )
  {
    XLALClearErrno();
    ABORT( status, RANDOMH_ENULL, RANDOMH_MSGENULL );
  }

  RETURN (status);
}


/* <lalVerbatim file="RandomCP"> */
void
LALNormalDeviates (
    LALStatus    *status,
    REAL4Vector  *deviates,
    RandomParams *params
    )
{ /* </lalVerbatim> */
  INITSTATUS (status, "LALNormalDeviates", RANDOMC);

  ASSERT (params, status, RANDOMH_ENULL, RANDOMH_MSGENULL);
  ASSERT (deviates, status, RANDOMH_ENULL, RANDOMH_MSGENULL);
  ASSERT (deviates->data, status, RANDOMH_ENULL, RANDOMH_MSGENULL);
  ASSERT (deviates->length > 0, status, RANDOMH_ESIZE, RANDOMH_MSGESIZE);

  if ( XLALNormalDeviates( deviates, params ) < 0 )
  {
    int errnum = xlalErrno;
    XLALClearErrno();
    switch ( errnum )
    {
      case XLAL_EFAULT:
        ABORT( status, RANDOMH_ENULL, RANDOMH_MSGENULL );
      case XLAL_EBADLEN:
        ABORT( status, RANDOMH_ESIZE, RANDOMH_MSGESIZE );
      default:
        ABORTXLAL( status );
    }
  }

  RETURN (status);
}

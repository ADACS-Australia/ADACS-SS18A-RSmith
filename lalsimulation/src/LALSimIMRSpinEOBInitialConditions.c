#include <lal/LALSimInspiral.h>
#include <lal/LALSimIMR.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_deriv.h>

#include "LALSimIMRSpinEOB.h"
#include "LALSimIMRSpinEOBHamiltonian.c"
#include "LALSimIMRSpinEOBHcapNumericalDerivative.c"
#include "LALSimIMREOBFactorizedWaveform.c"

#ifndef _LALSIMIMRSPINEOBINITIALCONDITIONS_C
#define _LALSIMIMRSPINEOBINITIALCONDITIONS_C

typedef
struct tagSEOBRootParams
{
  REAL8         values[12];
  SpinEOBParams *params;
  REAL8         omega;
}
SEOBRootParams;

/**
 * Calculates the dot product of two vectors
 */
static inline
REAL8
CalculateDotProduct( const REAL8 a[], const REAL8 b[] )
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


/**
 * calculates the ith component of the cross product of two vectors,
 * where i is in the range 0-2.
 */
static inline
REAL8
CalculateCrossProduct( const INT4 i, const REAL8 a[], const REAL8 b[] )
{
  return a[(i+1)%3]*b[(i+2)%3] - a[(i+2)%3]*b[(i+1)%3];
}


/**
 * Normalizes the given vector
 */
static inline
int
NormalizeVector( REAL8 a[] )
{
  REAL8 norm = sqrt( a[0]*a[0] + a[1]*a[1] + a[2]*a[2] );

  a[0] /= norm;
  a[1] /= norm;
  a[2] /= norm;

  return XLAL_SUCCESS;
}

/**
 * Calculate the rotation matrix and its inverse
 */
static int
CalculateRotationMatrix(
                gsl_matrix *rotMatrix,
                gsl_matrix *rotInverse,
                REAL8       r[],
                REAL8       v[],
                REAL8       L[]
                )
{

  /* a, b, g are the angles alpha, beta and gamma */
  /* Use a, b and g to avoid shadowing gamma and getting a warning */
  REAL8 a, b, g;
  REAL8 cosa, sina, cosb, sinb, cosg, sing;

  /* Calculate the Euclidean Euler angles */

  /* We need to avoid a singularity if L is along z axis */
  if ( L[2] > 0.9999 )
  {
    a = b = g = 0.0;
  }
  else
  {
    a = atan2( L[0], -L[1] );
    b = atan2( sqrt( L[0]*L[0] + L[1]*L[1] ), L[2] );
    g = atan2( r[2], v[2] );
  }

  if ( ( cosa = cos( a ) ) < 1.0e-16 ) cosa = 0.0;
  if ( ( sina = sin( a ) ) < 1.0e-16 ) sina = 0.0;
  if ( ( cosb = cos( b ) ) < 1.0e-16 ) cosb = 0.0;
  if ( ( sinb = sin( b ) ) < 1.0e-16 ) sinb = 0.0;
  if ( ( cosg = cos( g ) ) < 1.0e-16 ) cosg = 0.0;
  if ( ( sing = sin( g ) ) < 1.0e-16 ) sing = 0.0;

  /* populate the matrix */
  gsl_matrix_set( rotMatrix, 0, 0, cosg*cosa - cosb*sina*sing );
  gsl_matrix_set( rotMatrix, 0, 1, cosg*sina + cosb*cosa*sing );
  gsl_matrix_set( rotMatrix, 0, 2, sing*sinb );
  gsl_matrix_set( rotMatrix, 1, 0, -sing*cosa - cosb*sina*cosg );
  gsl_matrix_set( rotMatrix, 1, 1, -sing*sina + cosb*cosa*cosg );
  gsl_matrix_set( rotMatrix, 1, 2, cosg*sinb );
  gsl_matrix_set( rotMatrix, 2, 0, sinb*sina );
  gsl_matrix_set( rotMatrix, 2, 1, -sinb*cosa );
  gsl_matrix_set( rotMatrix, 2, 2, cosb );

  /* Now populate the transpose (which should be the inverse) */
  gsl_matrix_transpose_memcpy( rotInverse, rotMatrix );

  /* Test that the code does what it should do */
  gsl_matrix *ab = gsl_matrix_alloc( 3, 3 );
  gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1., rotMatrix, rotInverse, 0., ab );

  printf( "The generated rotation matrix:?\n" );
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      printf( "%.16e\t", gsl_matrix_get( rotMatrix, i, j ) );
    }
    printf( "\n" );
  }

  printf( "Is the transpose of the rotation matrix the inverse?\n" );
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      printf( "%.16e\t", gsl_matrix_get( ab, i, j ) );
    }
    printf( "\n" );
  }
  gsl_matrix_free( ab );

  return XLAL_SUCCESS;
}


/**
 * Applies a rotation matrix to a given vector
 */
static int
ApplyRotationMatrix(
             gsl_matrix *rotMatrix,
             REAL8      a[]
                   )
{

  gsl_vector *tmpVec1 = gsl_vector_alloc( 3 );
  gsl_vector *tmpVec2 = gsl_vector_calloc( 3 );  

  gsl_vector_set( tmpVec1, 0, a[0] );
  gsl_vector_set( tmpVec1, 1, a[1] );
  gsl_vector_set( tmpVec1, 2, a[2] );

  gsl_blas_dgemv( CblasNoTrans, 1.0,  rotMatrix, tmpVec1, 0.0, tmpVec2 );

  a[0] = gsl_vector_get( tmpVec2, 0 );
  a[1] = gsl_vector_get( tmpVec2, 1 );
  a[2] = gsl_vector_get( tmpVec2, 2 );

  gsl_vector_free( tmpVec1 );
  gsl_vector_free( tmpVec2 );

  return XLAL_SUCCESS;
}

/**
 * Performs a co-ordinate transformation from spherical to Cartesian co-ordinates.
 * In the code from Tyson, this was only applicable to the theta=pi/2, phi=0 case
 * TODO: Make sure I do not need anything more general
 */
static int SphericalToCartesian(
                      REAL8 qCart[],
                      REAL8 pCart[],
                      const REAL8 qSph[],
                      const REAL8 pSph[]
                      )
{

  REAL8 r;
  REAL8 pr, pTheta, pPhi;

  REAL8 x, y, z;
  REAL8 pX, pY, pZ;

  r      = qSph[0];
  pr     = pSph[0];
  pTheta = pSph[1];
  pPhi   = pSph[2];

  x = r;
  y = 0.0;
  z = 0.0;

  pX = pr;
  pY = pPhi / r;
  pZ = - pTheta / r;

  /* Copy these values to the output vectors */
  qCart[0] = x;
  qCart[1] = y;
  qCart[2] = z;
  pCart[0] = pX;
  pCart[1] = pY;
  pCart[2] = pZ;

  return XLAL_SUCCESS;
}


/**
 * Perform a co-ordinate transformation from Cartesian to spherical co-ordinates.
 * As taken from Tyson's code, this only applies for theta=pi/2, phi=0.
 * TODO: Check this is general enough.
 */
static int CartesianToSpherical(
                      REAL8 qSph[],
                      REAL8 pSph[],
                      const REAL8 qCart[],
                      const REAL8 pCart[]
                      )
{

  REAL8 r;
  REAL8 pr, pTheta, pPhi;
  
  REAL8 x; //, y, z;
  REAL8 pX, pY, pZ;

  x  = qCart[0];
  //y  = qCart[1];
  //z  = qCart[2];
  pX = pCart[0];
  pY = pCart[1];
  pZ = pCart[2];

  r  = x;
  pr = pX;
  pTheta = - r * pZ;
  pPhi   =   r * pY;

  /* Fill the output vectors */
  qSph[0] = r;
  qSph[1] = LAL_PI_2;
  qSph[2] = 0.0;
  pSph[0] = pr;
  pSph[1] = pTheta;
  pSph[2] = pPhi;

  return XLAL_SUCCESS;
}


static int
XLALFindSphericalOrbit( const gsl_vector *x,
                        void *params,
                        gsl_vector *f
                      )
{

  SEOBRootParams *rootParams = (SEOBRootParams *) params;

  REAL8 py, pz, r, ptheta, pphi;

  /* Numerical derivative of Hamiltonian wrt given value */
  REAL8 dHdx, dHdpy, dHdpz;
  REAL8 dHdr, dHdptheta, dHdpphi;

  /* Populate the appropriate values */
  rootParams->values[0] = r  = gsl_vector_get( x, 0 );
  rootParams->values[4] = py = gsl_vector_get( x, 1 );
  rootParams->values[5] = pz = gsl_vector_get( x, 2 );

  printf( "Values r = %.16e, py = %.16e, pz = %.16e\n", r, py, pz );

  ptheta = - r * pz;
  pphi   = r * py;

  /* dHdR */
  dHdx = XLALSpinHcapNumDerivWRTParam( 0, rootParams->values, rootParams->params );
  if ( XLAL_IS_REAL8_FAIL_NAN( dHdx ) )
  {
    XLAL_ERROR( XLAL_EFUNC );
  }
  printf( "dHdx = %.16e\n", dHdx );

  /* dHdPphi (I think we can use dHdPy in this coord system) */
  /* TODO: Check this is okay */
  dHdpy = XLALSpinHcapNumDerivWRTParam( 4, rootParams->values, rootParams->params );
  if ( XLAL_IS_REAL8_FAIL_NAN( dHdpy ) )
  {
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* dHdPtheta (I think we can use dHdPz in this coord system) */
  /* TODO: Check this is okay */
  dHdpz = XLALSpinHcapNumDerivWRTParam( 5, rootParams->values, rootParams->params );
  if ( XLAL_IS_REAL8_FAIL_NAN( dHdpz ) )
  {
    XLAL_ERROR( XLAL_EFUNC );
  }

  /* Now convert to spherical polars */
  dHdr      = dHdx - dHdpy * pphi / (r*r) + dHdpz * ptheta / (r*r);
  dHdptheta = - dHdpz / r;
  dHdpphi   = dHdpy / r;

  /* populate the function vector */
  gsl_vector_set( f, 0, dHdr );
  gsl_vector_set( f, 1, dHdptheta );
  gsl_vector_set( f, 2, dHdpphi - rootParams->omega );

  printf( "Current funcvals = %.16e %.16e %.16e\n", gsl_vector_get( f, 0 ), gsl_vector_get( f, 1 ),
     /*gsl_vector_get( f, 2 )*/dHdpphi );

  return XLAL_SUCCESS;
}


/* Wrapper for calculating the second derivative of the Hamiltonian in spherical co-ordinates */
/* It only works for the specific co-ord system we use here */
static double GSLSpinHamiltonianDerivWrapper( double x, void *params )
{

  HcapSphDeriv2Params *dParams = (HcapSphDeriv2Params *) params;

  REAL8 sphValues[12];
  REAL8 cartValues[12];

  REAL8 dHdr, dHdx, dHdpy, dHdpz;
  REAL8 r, ptheta, pphi;

  memcpy( sphValues, dParams->sphValues, sizeof( sphValues ) );
  sphValues[dParams->varyParam1] = x;

  SphericalToCartesian( cartValues, cartValues+3, sphValues, sphValues+3 );
  memcpy( cartValues+6, sphValues+6, 6*sizeof(REAL8) );

  r      = sphValues[0];
  ptheta = sphValues[4];
  pphi   = sphValues[5];

  /* Return the appropriate derivative according to varyParam2 */
  switch ( dParams->varyParam2 )
  {
    case 0:
      /* dHdr */
      dHdx  = XLALSpinHcapNumDerivWRTParam( 0, cartValues, dParams->params );
      dHdpy = XLALSpinHcapNumDerivWRTParam( 4, cartValues, dParams->params );
      dHdpz = XLALSpinHcapNumDerivWRTParam( 5, cartValues, dParams->params );

      dHdr      = dHdx - dHdpy * pphi / (r*r) + dHdpz * ptheta / (r*r);
      printf( "dHdr = %.16e\n", dHdr );
      return dHdr;

      break;
    case 4:
      /* dHdptheta */
      dHdpz = XLALSpinHcapNumDerivWRTParam( 5, cartValues, dParams->params );
      return - dHdpz / r;
      break;
    case 5:
      /* dHdpphi */
      dHdpy = XLALSpinHcapNumDerivWRTParam( 4, cartValues, dParams->params );
      return dHdpy / r;
      break;
    default:
      XLALPrintError( "This option is not supported in the second derivative function!\n" );
      XLAL_ERROR_REAL8( XLAL_EFUNC );
      break;
  }
}


/* Function to calculate the second derivative of the Hamiltonian. */
/* The derivatives are taken with respect to indices idx1, idx2    */
static REAL8 XLALCalculateSphHamiltonianDeriv2(
                                       const int idx1,
                                       const int idx2,
                                       const REAL8 values[],
                                       SpinEOBParams *params
                                       )
{

  static const REAL8 STEP_SIZE = 1.0e-5;

  REAL8 result;
  REAL8 UNUSED absErr;

  HcapSphDeriv2Params dParams;

  gsl_function F;
  INT4 UNUSED gslStatus;

  dParams.sphValues  = values;
  dParams.varyParam1 = idx1;
  dParams.varyParam2 = idx2;
  dParams.params     = params;

  /*printf( " In second deriv function: values\n" );
  for ( int i = 0; i < 12; i++ )
  {
    printf( "%.16e ", values[i] );
  }
  printf( "\n" );
*/
  F.function = GSLSpinHamiltonianDerivWrapper;
  F.params   = &dParams;

  /* GSL seemed to give weird answers - try my own code */
  /*result = GSLSpinHamiltonianDerivWrapper( values[idx1] + STEP_SIZE, &dParams )
         - GSLSpinHamiltonianDerivWrapper( values[idx1] - STEP_SIZE, &dParams );
  printf( "%.16e - %.16e / 2h\n", GSLSpinHamiltonianDerivWrapper( values[idx1] + STEP_SIZE, &dParams ), GSLSpinHamiltonianDerivWrapper( values[idx1] - STEP_SIZE, &dParams ) );

  result = result / ( 2.*STEP_SIZE );
*/
 
  XLAL_CALLGSL( gslStatus = gsl_deriv_central( &F, values[idx1], 
                      STEP_SIZE, &result, &absErr ) );

  if ( gslStatus != GSL_SUCCESS )
  {
    XLALPrintError( "XLAL Error %s - Failure in GSL function\n", __func__ );
    XLAL_ERROR_REAL8( XLAL_EFUNC );
  }

  printf( "Second deriv abs err = %.16e\n", absErr );

  printf( "RESULT = %.16e\n", result );
  return result;
}




/**
 * Main function for calculating the spinning EOB initial conditions
 */

static int XLALSimIMRSpinEOBInitialConditions(
                      REAL8Vector   *initConds,
                      const REAL8    mass1,
                      const REAL8    mass2,
                      const REAL8    fMin,
                      const REAL8    inc,
                      const REAL8    spin1[],
                      const REAL8    spin2[],
                      SpinEOBParams *params
                      )
{

#ifndef LAL_NDEBUG
  if ( !initConds )
  {
    XLAL_ERROR( XLAL_EINVAL );
  }
#endif


  static const int UNUSED lMax = 8;

  int i;

  /* Variable to keep track of whether the user requested the tortoise */
  int tmpTortoise;

  REAL8 mTotal;
  REAL8 eta;
  REAL8 omega, v0;   /* Initial velocity and angular frequency */

  REAL8 ham;      /* Hamiltonian */

  REAL8 LnHat[3]; /* Initial orientation of angular momentum */
  REAL8 rHat[3];  /* Initial orientation of radial vector */
  REAL8 vHat[3];  /* Initial orientation of velocity vector */
  REAL8 Lhat[3];  /* Direction of relativistic ang mom */
  REAL8 qHat[3];
  REAL8 pHat[3];

  /* q and p vectors in Cartesian and spherical coords */
  REAL8 qCart[3], pCart[3];
  REAL8 qSph[3], pSph[3];

  /* We will need to manipulate the spin vectors */
  /* We will use temporary vectors to do this */
  REAL8 tmpS1[3];
  REAL8 tmpS2[3];

  REAL8Vector qCartVec, pCartVec;
  REAL8Vector s1Vec, s2Vec;
  REAL8Vector sKerr, sStar;
  REAL8       sKerrData[3], sStarData[3];
  REAL8       a = 0., chiS, chiA;
  REAL8       chi1, chi2;

  /* We will need a full values vector for calculating derivs of Hamiltonian */
  REAL8 sphValues[12];
  REAL8 cartValues[12];

  /* Matrices for rotating to the new basis set. */
  /* It is more convenient to calculate the ICs in a simpler basis */
  gsl_matrix *rotMatrix  = NULL;
  gsl_matrix *invMatrix  = NULL;
  gsl_matrix *rotMatrix2 = NULL;
  gsl_matrix *invMatrix2 = NULL;

  /* Root finding stuff for finding the spherical orbit */
  SEOBRootParams rootParams;
  const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
  gsl_multiroot_fsolver *rootSolver = NULL;

  gsl_multiroot_function rootFunction;
  gsl_vector *initValues  = NULL;
  gsl_vector *finalValues = NULL;
  int gslStatus;
  const int maxIter = 100;


  memset( &rootParams, 0, sizeof( rootParams ) );

  memcpy( tmpS1, spin1, sizeof(tmpS1) );
  memcpy( tmpS2, spin2, sizeof(tmpS2) );

  mTotal = mass1 + mass2;
  eta    = mass1 * mass2 / (mTotal * mTotal);

  /* We compute the ICs for the non-tortoise p, and convert at the end */
  tmpTortoise      = params->tortoise;
  params->tortoise = 0;

  /* Set the initial orbital ang mom direction. Taken from STPN code */
  LnHat[0] = sin(inc);
  LnHat[1] = 0.;
  LnHat[2] = cos(inc);

  /* Set the radial direction - need to take care to avoid singularity if L is along z axis */
  if ( LnHat[2] > 0.9999 )
  {
    rHat[0] = 1.;
    rHat[1] = rHat[2] = 0.;
  }
  else
  {
    REAL8 theta0 = atan( - LnHat[2] / LnHat[0] );
    rHat[0] = sin( theta0 );
    rHat[1] = 0;
    rHat[2] = cos( theta0 );
  }

  /* Now we can complete the triad */
  vHat[0] = CalculateCrossProduct( 0, LnHat, rHat );
  vHat[1] = CalculateCrossProduct( 1, LnHat, rHat );
  vHat[2] = CalculateCrossProduct( 2, LnHat, rHat );

  NormalizeVector( vHat );

  /* XXX Test code XXX */
  for ( i = 0; i < 3; i++ )
  {
    printf ( " LnHat[%d] = %.16e, rHat[%d] = %.16e, vHat[%d] = %.16e\n", i, LnHat[i], i, rHat[i], i, vHat[i] );
  }

  printf("\n\n" );
  for ( i = 0; i < 3; i++ )
  {
    printf ( " s1[%d] = %.16e, s2[%d] = %.16e\n", i, tmpS1[i], i, tmpS2[i] );
  }

  /* Allocate and compute the rotation matrices */
  XLAL_CALLGSL( rotMatrix = gsl_matrix_alloc( 3, 3 ) );
  XLAL_CALLGSL( invMatrix = gsl_matrix_alloc( 3, 3 ) );
  if ( !rotMatrix || !invMatrix )
  {
    if ( rotMatrix ) gsl_matrix_free( rotMatrix );
    if ( invMatrix ) gsl_matrix_free( invMatrix );
    XLAL_ERROR( XLAL_ENOMEM );
  }

  if ( CalculateRotationMatrix( rotMatrix, invMatrix, rHat, vHat, LnHat ) == XLAL_FAILURE )
  {
    gsl_matrix_free( rotMatrix );
    gsl_matrix_free( invMatrix );
    XLAL_ERROR( XLAL_ENOMEM );
  }

  /* Rotate the orbital vectors and spins */
  ApplyRotationMatrix( rotMatrix, rHat );
  ApplyRotationMatrix( rotMatrix, vHat );
  ApplyRotationMatrix( rotMatrix, LnHat );
  ApplyRotationMatrix( rotMatrix, tmpS1 );
  ApplyRotationMatrix( rotMatrix, tmpS2 );

  /* XXX Test code XXX */
  printf( "\nAfter applying rotation matrix:\n\n" );
  for ( i = 0; i < 3; i++ )
  {
    printf ( " LnHat[%d] = %.16e, rHat[%d] = %.16e, vHat[%d] = %.16e\n", i, LnHat[i], i, rHat[i], i, vHat[i] );
  }

  printf("\n\n" );
  for ( i = 0; i < 3; i++ )
  {
    printf ( " s1[%d] = %.16e, s2[%d] = %.16e\n", i, tmpS1[i], i, tmpS2[i] );
  }

  /* Calculate the initial velocity from the given initial frequency */
  omega = LAL_PI * mTotal * LAL_MTSUN_SI * fMin;
  v0    = cbrt( omega );

  /* Given this, we can start to calculate the initial conditions */
  /* for spherical coords in the new basis */
  rootParams.omega  = omega;
  rootParams.params = params;
  
  /* To start with, we will just assign Newtonian-ish ICs to the system */
  rootParams.values[0] = 1./(v0*v0);  /* Initial r */
  rootParams.values[4] = v0;    /* Initial p */
  memcpy( rootParams.values+6, tmpS1, sizeof( tmpS1 ) );
  memcpy( rootParams.values+9, tmpS2, sizeof( tmpS2 ) );

  printf( "ICs guess: r = %.16e, p = %.16e\n", rootParams.values[0], rootParams.values[4] );

  /* Initialise the gsl stuff */
  XLAL_CALLGSL( rootSolver = gsl_multiroot_fsolver_alloc( T, 3 ) );
  if ( !rootSolver )
  {
    gsl_matrix_free( rotMatrix );
    gsl_matrix_free( invMatrix );
    XLAL_ERROR( XLAL_ENOMEM );
  }

  XLAL_CALLGSL( initValues = gsl_vector_calloc( 3 ) );
  if ( !initValues )
  {
    gsl_multiroot_fsolver_free( rootSolver );
    gsl_matrix_free( rotMatrix );
    gsl_matrix_free( invMatrix );
    XLAL_ERROR( XLAL_ENOMEM );
  }

  gsl_vector_set( initValues, 0, rootParams.values[0] );
  gsl_vector_set( initValues, 1, rootParams.values[4] );

  rootFunction.f      = XLALFindSphericalOrbit;
  rootFunction.n      = 3;
  rootFunction.params = &rootParams;

  gsl_multiroot_fsolver_set( rootSolver, &rootFunction, initValues );

  /* We are now ready to iterate to find the solution */
  i = 0;

  do
  {
    XLAL_CALLGSL( gslStatus = gsl_multiroot_fsolver_iterate( rootSolver ) ); 
    if ( gslStatus != GSL_SUCCESS )
    {
      XLALPrintError( "Error in GSL iteration function!\n" );
      gsl_multiroot_fsolver_free( rootSolver );
      gsl_vector_free( initValues );
      gsl_matrix_free( rotMatrix );
      gsl_matrix_free( invMatrix );
      XLAL_ERROR( XLAL_EFUNC );
    }

    XLAL_CALLGSL( gslStatus = gsl_multiroot_test_residual( rootSolver->f, 1.0e-10 ) );
    i++;
  }
  while ( gslStatus == GSL_CONTINUE && i <= maxIter );

  if ( i > maxIter && gslStatus != GSL_SUCCESS )
  {
    gsl_multiroot_fsolver_free( rootSolver );
    gsl_vector_free( initValues );
    gsl_matrix_free( rotMatrix );
    gsl_matrix_free( invMatrix );
    XLAL_ERROR( XLAL_EMAXITER );
  }

  finalValues = gsl_multiroot_fsolver_root( rootSolver );

  printf( "Spherical orbit conditions here given by the following:\n" );
  printf( " x = %.16e, py = %.16e, pz = %.16e\n", gsl_vector_get( finalValues, 0 ), 
      gsl_vector_get( finalValues, 1 ), gsl_vector_get( finalValues, 2 ) );

  memset( qCart, 0, sizeof(qCart) );
  memset( pCart, 0, sizeof(pCart) );

  qCart[0] = gsl_vector_get( finalValues, 0 );
  pCart[1] = gsl_vector_get( finalValues, 1 );
  pCart[2] = gsl_vector_get( finalValues, 2 );

  /* Free the GSL root finder, since we're done with it */
  gsl_multiroot_fsolver_free( rootSolver );
  gsl_vector_free( initValues );

  /* Now we can calculate the relativistic L and rotate to a new basis */
  memcpy( qHat, qCart, sizeof(qCart) );
  memcpy( pHat, pCart, sizeof(pCart) );

  NormalizeVector( qHat );
  NormalizeVector( pHat );

  Lhat[0] = CalculateCrossProduct( 0, qHat, pHat );
  Lhat[1] = CalculateCrossProduct( 1, qHat, pHat );
  Lhat[2] = CalculateCrossProduct( 2, qHat, pHat );  

  NormalizeVector( Lhat );

  XLAL_CALLGSL( rotMatrix2 = gsl_matrix_alloc( 3, 3 ) );
  XLAL_CALLGSL( invMatrix2 = gsl_matrix_alloc( 3, 3 ) );

  if ( CalculateRotationMatrix( rotMatrix2, invMatrix2, qHat, pHat, Lhat ) == XLAL_FAILURE )
  {
    gsl_matrix_free( rotMatrix );
    gsl_matrix_free( invMatrix );
    XLAL_ERROR( XLAL_ENOMEM );
  }

  ApplyRotationMatrix( rotMatrix2, rHat );
  ApplyRotationMatrix( rotMatrix2, vHat );
  ApplyRotationMatrix( rotMatrix2, LnHat );
  ApplyRotationMatrix( rotMatrix2, tmpS1 );
  ApplyRotationMatrix( rotMatrix2, tmpS2 );
  ApplyRotationMatrix( rotMatrix2, qCart );
  ApplyRotationMatrix( rotMatrix2, pCart );

  /* Now we can calculate the flux. Change to spherical co-ords */
  CartesianToSpherical( qSph, pSph, qCart, pCart );
  memcpy( sphValues, qSph, sizeof( qSph ) );
  memcpy( sphValues+3, pSph, sizeof( pSph ) );
  memcpy( sphValues+6, tmpS1, sizeof(tmpS1) );
  memcpy( sphValues+9, tmpS2, sizeof(tmpS2) );

  memcpy( cartValues, qCart, sizeof(qCart) );
  memcpy( cartValues+3, pCart, sizeof(pCart) );
  memcpy( cartValues+6, tmpS1, sizeof(tmpS1) );
  memcpy( cartValues+9, tmpS2, sizeof(tmpS2) );

  REAL8 dHdpphi, d2Hdr2, d2Hdrdpphi;
  REAL8 rDot, dHdpr, flux, dEdr;

  d2Hdr2     = XLALCalculateSphHamiltonianDeriv2( 0, 0, sphValues, params );
  d2Hdrdpphi = XLALCalculateSphHamiltonianDeriv2( 0, 5, sphValues, params );

  printf( "d2Hdr2 = %.16e, d2Hdrdpphi = %.16e\n", d2Hdr2, d2Hdrdpphi );

  dHdpphi  = XLALSpinHcapNumDerivWRTParam( 4, cartValues, params ) / sphValues[0];
  
  dEdr  = - dHdpphi * d2Hdr2 / d2Hdrdpphi;

  printf( "d2Hdr2 = %.16e d2Hdrdpphi = %.16e dHdpphi = %.16e\n", d2Hdr2, d2Hdrdpphi, dHdpphi );

  if ( d2Hdr2 != 0.0 )
  {
    /* We will need to calculate the Hamiltonian to get the flux */
    s1Vec.length = s2Vec.length = sKerr.length = sStar.length = 3;
    s1Vec.data = tmpS1;
    s2Vec.data = tmpS2;
    sKerr.data = sKerrData;
    sStar.data = sStarData;

    qCartVec.length = pCartVec.length = 3;
    qCartVec.data   = qCart;
    pCartVec.data   = pCart;

    chi1 = tmpS1[0]*LnHat[0] + tmpS1[1]*LnHat[1] + tmpS1[2]*LnHat[2];
    chi2 = tmpS2[0]*LnHat[0] + tmpS2[1]*LnHat[1] + tmpS2[2]*LnHat[2];

    printf( "magS1 = %.16e, magS2 = %.16e\n", chi1, chi2 );

    chiS = 0.5 * ( chi1 / (mass1*mass1) + chi2 / (mass2*mass2) );
    chiA = 0.5 * ( chi1 / (mass1*mass1) - chi2 / (mass2*mass2) );

    XLALSimIMRSpinEOBCalculateSigmaKerr( &sKerr, mass1, mass2, &s1Vec, &s2Vec );
    XLALSimIMRSpinEOBCalculateSigmaStar( &sStar, mass1, mass2, &s1Vec, &s2Vec );

    /* The a in the flux has been set to zero, but not in the Hamiltonian */
    a = sqrt(sKerr.data[0]*sKerr.data[0] + sKerr.data[1]*sKerr.data[1] + sKerr.data[2]*sKerr.data[2]);
    XLALSimIMREOBCalcSpinFacWaveformCoefficients( params->eobParams->hCoeffs, eta, /*a*/0.0, chiS, chiA );
    XLALSimIMRCalculateSpinEOBHCoeffs( params->seobCoeffs, eta, a );

    ham = XLALSimIMRSpinEOBHamiltonian( eta, &qCartVec, &pCartVec, &sKerr, &sStar, params->tortoise, params->seobCoeffs );

    printf( "hamiltonian at this point is %.16e\n", ham );

    /* And now, finally, the flux */
    REAL8Vector polarDynamics;
    REAL8       polarData[4];

    polarDynamics.length = 4;
    polarDynamics.data = polarData;

    polarData[0] = qSph[0];
    polarData[1] = 0.;
    polarData[2] = pSph[0];
    polarData[3] = pSph[2];

    flux  = XLALInspiralSpinFactorizedFlux( &polarDynamics, omega, params, ham, lMax );
    flux  = flux / eta;

    rDot  = - flux / dEdr;

    /* We now need dHdpr - we take it that it is safely linear up to a pr of 1.0e-3 */
    cartValues[3] = 1.0e-3;
    dHdpr         = XLALSpinHcapNumDerivWRTParam( 3, cartValues, params );


    printf( "Ingredients going into prDot:\n" );
    printf( "flux = %.16e, dEdr = %.16e, dHdpr = %.16e\n", flux, dEdr, dHdpr );

    /* We can now calculate what pr should be taking into account the flux */
    pSph[0] = rDot / (dHdpr / cartValues[3] );
  }
  else
  {
    /* Since d2Hdr2 has evaluated to zero, we cannot do the above. Just set pr to zero */
    printf( "d2Hdr2 is zero!\n" );
    pSph[0] = 0;
  }

  /* Now we are done - convert back to cartesian coordinates ) */
  SphericalToCartesian( qCart, pCart, qSph, pSph );

  /* Undo rotations to get back to the original basis */
  /* Second rotation */
  ApplyRotationMatrix( invMatrix2, rHat );
  ApplyRotationMatrix( invMatrix2, vHat );
  ApplyRotationMatrix( invMatrix2, LnHat );
  ApplyRotationMatrix( invMatrix2, tmpS1 );
  ApplyRotationMatrix( invMatrix2, tmpS2 );
  ApplyRotationMatrix( invMatrix2, qCart );
  ApplyRotationMatrix( invMatrix2, pCart );

  /* First rotation */
  ApplyRotationMatrix( invMatrix, rHat );
  ApplyRotationMatrix( invMatrix, vHat );
  ApplyRotationMatrix( invMatrix, LnHat );
  ApplyRotationMatrix( invMatrix, tmpS1 );
  ApplyRotationMatrix( invMatrix, tmpS2 );
  ApplyRotationMatrix( invMatrix, qCart );
  ApplyRotationMatrix( invMatrix, pCart );

  /* If required, apply the tortoise transform */
  if ( tmpTortoise )
  {
    REAL8 r = sqrt(qCart[0]*qCart[0] + qCart[1]*qCart[1] + qCart[2]*qCart[2] );
    REAL8 deltaR = XLALSimIMRSpinEOBHamiltonianDeltaR( params->seobCoeffs, r, eta, a );
    REAL8 deltaT = XLALSimIMRSpinEOBHamiltonianDeltaT( params->seobCoeffs, r, eta, a );
    REAL8 csi    = sqrt( deltaT * deltaR )/(r*r + a*a);

    REAL8 pr = (qCart[0]*pCart[0] + qCart[1]*pCart[1] + qCart[2]*pCart[2])/r;

    params->tortoise = tmpTortoise;

    printf( "Applying the tortoise to p (csi = %.26e)\n", csi );

    for ( i = 0; i < 3; i++ )
    {
      pCart[i] = pCart[i] + qCart[i] * pr * (csi - 1.) / r;
    }
  }

  /* Now copy the initial conditions back to the return vector */
  memcpy( initConds->data, qCart, sizeof(qCart) );
  memcpy( initConds->data+3, pCart, sizeof(pCart) );
  memcpy( initConds->data+6, tmpS1, sizeof(tmpS1) );
  memcpy( initConds->data+9, tmpS2, sizeof(tmpS2) );

  printf( "THE FINAL INITIAL CONDITIONS:\n");
  printf( " %.16e %.16e %.16e\n%.16e %.16e %.16e\n%.16e %.16e %.16e\n%.16e %.16e %.16e\n", initConds->data[0], initConds->data[1], initConds->data[2],
          initConds->data[3], initConds->data[4], initConds->data[5], initConds->data[6], initConds->data[7], initConds->data[8],
          initConds->data[9], initConds->data[10], initConds->data[11] );

  return XLAL_SUCCESS;
}

#endif /*_LALSIMIMRSPINEOBINITIALCONDITIONS_C*/

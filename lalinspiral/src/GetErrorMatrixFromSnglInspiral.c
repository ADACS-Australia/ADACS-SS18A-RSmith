/*
*  Copyright (C) 2007 Craig Robinson
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

/*----------------------------------------------------------------------------
 *
 * File Name: GetErrorMatrixFromSnglInspiral.c
 *
 * Author: Craig Robinson
 *
 *---------------------------------------------------------------------------*/

#include <lal/LALStdlib.h>
#include <lal/LALError.h>
#include <lal/LALGSL.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/CoincInspiralEllipsoid.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

/**
 * \author Craig Robinson
 * \file
 * \ingroup CoincInspiralEllipsoid_h
 *
 * \brief Blah.
 *
 * ### Description ###
 *
 * <tt>XLALGetErrorMatrixFromSnglInspiral()</tt> takes in a
 * \c SnglInspiralTable, and a value for the e-thinca parameter. It returns
 * a \c gsl_matrix containing the the metric scaled appropriately for the
 * given e-thinca parameter.
 *
 * <tt>XLALGetPositionFromSnglInspiral()</tt> takes in a
 * \c SnglInspiralTable, and returns the position vector associated with
 * the trigger in \f$(t_C, \tau_0, \tau_3)\f$ space.
 *
 * <tt>XLALSetTimeInPositionVector()</tt> sets the time co-ordinate in the given
 * position vector to \c time. It returns zero upon successful completion.
 *
 */


/* Function for getting the error matrix from the metric in
 * (tc, tau0, tau3) space.
 */

gsl_matrix * XLALGetErrorMatrixFromSnglInspiral(SnglInspiralTable *event,
                                                REAL8              eMatch
                                               )

{
  gsl_matrix *shape = NULL;

  int xlalStatus;

#ifndef LAL_NDEBUG
  if (!event)
  {
    XLAL_ERROR_NULL( XLAL_EFAULT );
  }
#endif

  /* Allocate memory for the various matrices */
  XLAL_CALLGSL( shape  = gsl_matrix_alloc( 3, 3 ) );

  if ( !shape )
    XLAL_ERROR_NULL( XLAL_ENOMEM );

  /* Fill in the elements of the shape matrix */
  xlalStatus = XLALSetErrorMatrixFromSnglInspiral( shape, event, eMatch );
  if (xlalStatus != XLAL_SUCCESS )
  {
    gsl_matrix_free( shape );
    XLAL_ERROR_NULL( XLAL_EFUNC );
  }

  return shape;
}




int XLALSetErrorMatrixFromSnglInspiral(gsl_matrix        *shape,
                                       SnglInspiralTable *event,
                                       REAL8              eMatch
                                       )

{
  gsl_matrix *fisher = NULL;
  gsl_permutation *p = NULL;

  REAL8 freqRatio0 = 0.0;
  REAL8 freqRatio3 = 0.0;
  REAL8 fLow = 0.0;
  REAL8 mtotal = 0.0;

  int signum;
  int gslStatus;

#ifndef LAL_NDEBUG
  if ( !event )
    XLAL_ERROR( XLAL_EFAULT );

  if ( !shape )
    XLAL_ERROR( XLAL_EFAULT );

  if ( shape->size1 != 3 || shape->size1 != shape->size2 )
    XLAL_ERROR( XLAL_EBADLEN );
#endif

  if ( !event->Gamma[0] )
  {
    XLALPrintError( "Metric components are not set.\n" );
    XLAL_ERROR( XLAL_EINVAL );
  }

  if ( eMatch < 0 )
    XLAL_ERROR( XLAL_EINVAL );

  XLAL_CALLGSL( fisher = gsl_matrix_alloc( 3, 3 ) );
  XLAL_CALLGSL( p      = gsl_permutation_alloc( 3 ) );

  if ( !fisher || !p )
  {
    if ( fisher ) gsl_matrix_free( fisher );
    if ( p ) gsl_permutation_free( p );
    XLAL_ERROR( XLAL_ENOMEM );
  }

  mtotal = (event->mtotal)*LAL_MTSUN_SI;
  fLow =  5.0 / (256.0 * event->eta * pow(mtotal, 5.0/3.0) * event->tau0 );
  fLow = pow(fLow, 3.0/8.0) / LAL_PI;
  freqRatio0 = pow(fLow, 8.0/3.0);
  freqRatio3 = pow(fLow, 5.0/3.0);

  /* Fill in the elements of the fisher matrix */
  gsl_matrix_set( fisher, 0, 0, event->Gamma[0] );
  gsl_matrix_set( fisher, 0, 1, event->Gamma[1]/freqRatio0 );
  gsl_matrix_set( fisher, 1, 0, event->Gamma[1]/freqRatio0 );
  gsl_matrix_set( fisher, 0, 2, event->Gamma[2]/freqRatio3 );
  gsl_matrix_set( fisher, 2, 0, event->Gamma[2]/freqRatio3 );
  gsl_matrix_set( fisher, 1, 1, event->Gamma[3]/(freqRatio0*freqRatio0) );
  gsl_matrix_set( fisher, 1, 2, event->Gamma[4]/(freqRatio0*freqRatio3) );
  gsl_matrix_set( fisher, 2, 1, event->Gamma[4]/(freqRatio0*freqRatio3) );
  gsl_matrix_set( fisher, 2, 2, event->Gamma[5]/(freqRatio3*freqRatio3) );

  XLAL_CALLGSL( gslStatus = gsl_matrix_scale( fisher, 1.0 / eMatch ) );
  if ( gslStatus != GSL_SUCCESS )
  {
     gsl_matrix_free( fisher );
     gsl_permutation_free( p );
     XLAL_ERROR( XLAL_EFUNC );
  }

  /* Now invert to get the matrix we need */
  XLAL_CALLGSL( gslStatus = gsl_linalg_LU_decomp( fisher, p, &signum ) );
  if ( gslStatus == GSL_SUCCESS )
  {
    XLAL_CALLGSL( gslStatus = gsl_linalg_LU_invert( fisher, p, shape ) );
  }

  gsl_matrix_free( fisher );
  gsl_permutation_free( p );

  if ( gslStatus != GSL_SUCCESS )
    XLAL_ERROR( XLAL_EFUNC );

  return XLAL_SUCCESS;
}


/* Returns the position vector in (tc, tau0, tau3) space */

gsl_vector * XLALGetPositionFromSnglInspiral( SnglInspiralTable *table )

{
  gsl_vector *position = NULL;
  REAL8 endTime;

  REAL8 freqRatio0 = 0.0;
  REAL8 freqRatio3 = 0.0;
  REAL8 fLow = 0.0;
  REAL8 mtotal = 0.0;

#ifndef LAL_NDEBUG
  if ( !table )
    XLAL_ERROR_NULL( XLAL_EFAULT );
#endif

  XLAL_CALLGSL( position = gsl_vector_alloc( 3 ) );
  if ( !position )
    XLAL_ERROR_NULL( XLAL_ENOMEM );

  endTime = (REAL8) table->end.gpsSeconds +
        (REAL8) table->end.gpsNanoSeconds * 1.0e-9;

  mtotal = (table->mtotal)*LAL_MTSUN_SI;
  fLow =  5.0 / (256.0 * table->eta * pow(mtotal, 5.0/3.0) * table->tau0 );
  fLow = pow(fLow, 3.0/8.0) / LAL_PI;
  freqRatio0 = pow(fLow, 8.0/3.0);
  freqRatio3 = pow(fLow, 5.0/3.0);

  gsl_vector_set( position, 0, endTime );
  gsl_vector_set( position, 1, freqRatio0*table->tau0 );
  gsl_vector_set( position, 2, freqRatio3*table->tau3 );

  return position;
}


/* Sets the time in the position vector to the given value */

int XLALSetTimeInPositionVector( gsl_vector *position,
                                 REAL8 timeShift)

{
#ifndef LAL_NDEBUG
    if ( !position )
      XLAL_ERROR( XLAL_EFAULT );
#endif

    gsl_vector_set( position, 0, timeShift );

    return XLAL_SUCCESS;
}

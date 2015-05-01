//
// Copyright (C) 2011--2015 Reinhard Prix, Karl Wette
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with with program; see the file COPYING. If not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA 02111-1307 USA
//

#ifndef _METRICUTILS_H
#define _METRICUTILS_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <lal/LALStdlib.h>
#include <lal/UniversalDopplerMetric.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// \defgroup MetricUtils_h Header MetricUtils.h
/// \ingroup lalpulsar_metric
/// \author Reinhard Prix, Karl Wette
/// \brief Various useful utility functions for working with CW parameter-space metrics.
///
/// @{
///

REAL8 XLALCompareMetrics( const gsl_matrix *g1_ij, const gsl_matrix *g2_ij );

double XLALMetricDeterminant( const gsl_matrix *g_ij );
gsl_vector *XLALMetricEllipseBoundingBox( const gsl_matrix *g_ij, const double max_mismatch );

int XLALProjectMetric( gsl_matrix **p_gpr_ij, const gsl_matrix *g_ij, const UINT4 c );
int XLALCholeskyLDLTDecompMetric( gsl_matrix **p_cholesky, const gsl_matrix *g_ij );

int XLALTransformMetric( gsl_matrix **p_gpr_ij, const gsl_matrix *transform, const gsl_matrix *g_ij );
int XLALInverseTransformMetric( gsl_matrix **p_gpr_ij, const gsl_matrix *transform, const gsl_matrix *g_ij );

int XLALDiagNormalizeMetric( gsl_matrix **p_gpr_ij, gsl_matrix **p_transform, const gsl_matrix *g_ij );
int XLALNaturalizeMetric( gsl_matrix **p_gpr_ij, gsl_matrix **p_transform, const gsl_matrix *g_ij, const DopplerMetricParams *metricParams );
int XLALChangeMetricReferenceTime( gsl_matrix **p_gpr_ij, gsl_matrix **p_transform, const gsl_matrix *g_ij, const DopplerCoordinateSystem *coordSys, const double Dtau );

/// @}

#ifdef __cplusplus
}
#endif

#endif // _METRICUTILS_H

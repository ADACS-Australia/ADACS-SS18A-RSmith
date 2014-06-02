//
// Copyright (C) 2014 Karl Wette
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

#ifndef _SUPERSKYMETRICS_H
#define _SUPERSKYMETRICS_H

#include <gsl/gsl_matrix.h>
#include <lal/LALStdlib.h>
#include <lal/UniversalDopplerMetric.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// \file
/// \author Karl Wette
/// \brief Functions for using the super-sky and reduced super-sky metrics of \cite Wette.Prix.2013a
///

///
/// Compute the expanded super-sky metric, which separates spin and orbital sky components.
///
int XLALExpandedSuperSkyMetric(
  gsl_matrix **essky_metric,			///< [out] Pointer to allocated expanded super-sky metric
  const size_t spindowns,			///< [in] Number of frequency spindown coordinates
  const LIGOTimeGPS* ref_time,			///< [in] Reference time for the metric
  const LALSegList* segments,			///< [in] List of segments to average metric over
  const double fiducial_freq,			///< [in] Fiducial frequency for sky-position coordinates
  const MultiLALDetector* detectors,		///< [in] List of detector to average metric over
  const MultiNoiseFloor* detector_weights,	///< [in] Weights used to combine single-detector metrics (default: unit weights)
  const DetectorMotionType detector_motion,	///< [in] Which detector motion to use
  const EphemerisData* ephemerides		///< [in] Earth/Sun ephemerides
  );

///
/// Compute the (untransformed) super-sky metric in equatorial coordinates from the expanded super-sky metric.
///
int XLALSuperSkyMetric(
  gsl_matrix **ssky_metric,			///< [out] Pointer to allocated super-sky metric
  const gsl_matrix* essky_metric		///< [in] Input expanded super-sky metric
  );

///
/// Compute the reduced super-sky metric and coordinate transform data from the expanded super-sky metric.
///
int XLALReducedSuperSkyMetric(
  gsl_matrix **rssky_metric,			///< [out] Pointer to allocated reduced super-sky metric
  gsl_matrix **rssky_transf,			///< [out] Pointer to allocated coordinate transform data
  const gsl_matrix* essky_metric		///< [in] Input expanded super-sky metric
  );

#ifdef __cplusplus
}
#endif

#endif

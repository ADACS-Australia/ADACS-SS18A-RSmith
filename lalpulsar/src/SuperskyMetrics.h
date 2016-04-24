//
// Copyright (C) 2014, 2015 Karl Wette
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
#include <lal/LatticeTiling.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// \defgroup SuperskyMetrics_h Header SuperskyMetrics.h
/// \ingroup lalpulsar_metric
/// \author Karl Wette
/// \brief Compute the supersky metrics and coordinate transforms of \cite WettePrix2013a .
///
/// @{
///

///
/// Computed supersky metrics, returned by XLALComputeSuperskyMetrics().
///
#ifdef SWIG // SWIG interface directives
SWIGLAL( ARRAY_MULTIPLE_LENGTHS( tagSuperskyMetrics, num_segments ) );
#endif // SWIG
typedef struct tagSuperskyMetrics {
  size_t num_segments;                          ///< Number of segments

#ifdef SWIG // SWIG interface directives
  SWIGLAL( ARRAY_1D( SuperskyMetrics, gsl_matrix *, coh_rssky_metric, size_t, num_segments ) );
#endif // SWIG
  gsl_matrix **coh_rssky_metric;                ///< Coherent reduced supersky metric (2-dimensional sky) for each segment
#ifdef SWIG // SWIG interface directives
  SWIGLAL( ARRAY_1D( SuperskyMetrics, gsl_matrix *, coh_rssky_transf, size_t, num_segments ) );
#endif // SWIG
  gsl_matrix **coh_rssky_transf;                ///< Coherent reduced supersky metric coordinate transform data for each segment

  gsl_matrix *semi_rssky_metric;                ///< Semicoherent reduced supersky metric (2-dimensional sky)
  gsl_matrix *semi_rssky_transf;                ///< Semicoherent reduced supersky metric coordinate transform data

} SuperskyMetrics;

///
/// Compute the supersky metrics, which are returned in a #SuperskyMetrics struct.
///
SuperskyMetrics *XLALComputeSuperskyMetrics(
  const size_t spindowns,                       ///< [in] Number of frequency+spindown coordinates
  const LIGOTimeGPS *ref_time,                  ///< [in] Reference time for the metrics
  const LALSegList *segments,                   ///< [in] List of segments to compute metrics over
  const double fiducial_freq,                   ///< [in] Fiducial frequency for sky-position coordinates
  const MultiLALDetector *detectors,            ///< [in] List of detectors to average metrics over
  const MultiNoiseFloor *detector_weights,      ///< [in] Weights used to combine single-detector metrics (default: unit weights)
  const DetectorMotionType detector_motion,     ///< [in] Which detector motion to use
  const EphemerisData *ephemerides              ///< [in] Earth/Sun ephemerides
  );

///
/// Destroy a #SuperskyMetrics struct.
///
void XLALDestroySuperskyMetrics(
  SuperskyMetrics *metrics                      ///< [in] Supersky metrics struct
  );

///
/// Return dimensions of the supersky metrics.
///
int XLALSuperskyMetricsDimensions(
  const SuperskyMetrics *metrics,               ///< [in] Supersky metrics struct
  size_t *spindowns                             ///< [out] Number of spindown dimensions
  );

#ifdef SWIG // SWIG interface directives
SWIGLAL( COPYINOUT_ARRAYS( gsl_matrix, rssky_metric, rssky_transf ) );
#endif // SWIG

///
/// Scale a given supersky metric and its coordinate transform data to a new fiducial frequency.
///
int XLALScaleSuperskyMetricFiducialFreq(
  gsl_matrix *rssky_metric,                     ///< [in] Reduced supersky metric
  gsl_matrix *rssky_transf,                     ///< [in] Reduced supersky metric coordinate transform data
  const double new_fiducial_freq                ///< [in] New fiducial frequency
  );

#ifdef SWIG // SWIG interface directives
SWIGLAL_CLEAR( COPYINOUT_ARRAYS( gsl_matrix, rssky_metric, rssky_transf ) );
#endif // SWIG

///
/// Scale all supersky metrics and their coordinate transform data to a new fiducial frequency.
///
int XLALScaleSuperskyMetricsFiducialFreq(
  SuperskyMetrics *metrics,                     ///< [in] Supersky metrics struct
  const double new_fiducial_freq                ///< [in] New fiducial frequency
  );

///
/// Project and rescale the reduced supersky metrics in the frequency dimension, such that all
/// reduced supersky metrics have the same frequency spacing for the given maximum mismatches.
///
int XLALEqualizeReducedSuperskyMetricsFreqSpacing(
  SuperskyMetrics *metrics,                     ///< [in] Supersky metrics struct
  const double coh_max_mismatch,                ///< [in] Maximum coherent mismatch
  const double semi_max_mismatch                ///< [in] Maximum semicoherent mismatch
  );

///
/// Convert a point from physical to supersky coordinates.
///
int XLALConvertPhysicalToSuperskyPoint(
  gsl_vector *out_rssky,                        ///< [out] Output point in supersky coordinates
  const PulsarDopplerParams *in_phys,           ///< [in] Input point in physical coordinates
  const gsl_matrix *rssky_transf                ///< [in] Reduced supersky coordinate transform data
  );

///
/// Convert a point from supersky to physical coordinates.
///
int XLALConvertSuperskyToPhysicalPoint(
  PulsarDopplerParams *out_phys,                ///< [out] Output point in physical coordinates
  const gsl_vector *in_rssky,                   ///< [in] Input point in supersky coordinates
  const gsl_matrix *rssky_transf                ///< [in] Reduced supersky coordinate transform data
  );

///
/// Convert a point between supersky coordinates. The vectors \c out_rssky and \c in_rssky may be the same.
///
int XLALConvertSuperskyToSuperskyPoint(
  gsl_vector *out_rssky,                        ///< [out] Output point in supersky coordinates
  const gsl_matrix *out_rssky_transf,           ///< [in] Output reduced supersky coordinate transform data
  const gsl_vector *in_rssky,                   ///< [in] Input point in supersky coordinates
  const gsl_matrix *in_rssky_transf             ///< [in] Input reduced supersky coordinate transform data
  );

///
/// Convert a set of points from physical to supersky coordinates.
///
#ifdef SWIG // SWIG interface directives
SWIGLAL( INOUT_STRUCTS( gsl_matrix **, out_rssky ) );
#endif
int XLALConvertPhysicalToSuperskyPoints(
  gsl_matrix **out_rssky,                       ///< [out] Columns are output point in supersky coordinates
  const gsl_matrix *in_phys,                    ///< [in] Columns are input point in physical coordinates
  const gsl_matrix *rssky_transf                ///< [in] Reduced supersky coordinate transform data
  );

///
/// Convert a set of points from supersky to physical coordinates.
///
#ifdef SWIG // SWIG interface directives
SWIGLAL( INOUT_STRUCTS( gsl_matrix **, out_phys ) );
#endif
int XLALConvertSuperskyToPhysicalPoints(
  gsl_matrix **out_phys,                        ///< [out] Columns are output point in physical coordinates
  const gsl_matrix *in_rssky,                   ///< [in] Columns are input point in supersky coordinates
  const gsl_matrix *rssky_transf                ///< [in] Reduced supersky coordinate transform data
  );

#ifdef SWIG // SWIG interface directives
SWIGLAL( COPYINOUT_ARRAYS( gsl_matrix, rssky_metric, rssky_transf ) );
#endif // SWIG

///
/// Set parameter-space bounds on the physical sky position \f$(\alpha, \delta)\f$ for a lattice
/// tiling using the reduced supersky metric. The metric and coordinate transform data must be supplied,
/// since they will be transformed such that the physical sky region maps to a region in the reduced
/// supersky coordinates \f$(n_a,n_b)\f$ which may be covered by the lattice tiling.
///
int XLALSetSuperskyLatticeTilingPhysicalSkyBounds(
  LatticeTiling *tiling,                        ///< [in] Lattice tiling
  gsl_matrix *rssky_metric,                     ///< [in] Reduced supersky metric
  gsl_matrix *rssky_transf,                     ///< [in] Reduced supersky coordinate transform data
  const double alpha1,                          ///< [in] First bound on sky position right ascension
  const double alpha2,                          ///< [in] Second bound on sky position right ascension
  const double delta1,                          ///< [in] First bound on sky position declination
  const double delta2                           ///< [in] Second bound on sky position declination
  );

///
/// Set parameter-space bounds on an equal-area fraction of the physical sky \f$(\alpha, \delta)\f$
/// for a lattice tiling using the reduced supersky metric. The metric and coordinate transform data
/// must be supplied, since they will be transformed such that the physical sky patch maps to a region
/// in the reduced supersky coordinates \f$(n_a,n_b)\f$ which may be covered by the lattice tiling.
///
int XLALSetSuperskyLatticeTilingPhysicalSkyPatch(
  LatticeTiling *tiling,                        ///< [in] Lattice tiling
  gsl_matrix *rssky_metric,                     ///< [in] Reduced supersky metric
  gsl_matrix *rssky_transf,                     ///< [in] Reduced supersky coordinate transform data
  const UINT4 patch_count,                      ///< [in] Number of equal-area patches to divide sky into
  const UINT4 patch_index                       ///< [in] Index of the patch for which to set bounds
  );

#ifdef SWIG // SWIG interface directives
SWIGLAL_CLEAR( COPYINOUT_ARRAYS( gsl_matrix, rssky_metric, rssky_transf ) );
#endif // SWIG

///
/// Set parameter-space bounds on the physical frequency/spindowns \f$f^{(s)}\f$ for a lattice
/// tiling using the reduced supersky metric.
///
int XLALSetSuperskyLatticeTilingPhysicalSpinBound(
  LatticeTiling *tiling,                        ///< [in] Lattice tiling
  const gsl_matrix *rssky_transf,               ///< [in] Reduced supersky coordinate transform data
  const size_t s,                               ///< [in] Spindown order; 0=frequency, 1=first spindown, etc.
  const double bound1,                          ///< [in] First bound on frequency/spindown
  const double bound2                           ///< [in] Second bound on frequency/spindown
  );

///
/// Set parameter-space bounds on the reduced supersky frequency/spindown coordinates \f$\nu^{(s)}\f$
/// for a lattice tiling using the reduced supersky metric. These coordinates are related to the
/// physical frequency/spindowns by \f$\nu^{(s)} = f^{(s)} + \vec\Delta^s \cdot \vec n\f$.
///
int XLALSetSuperskyLatticeTilingCoordinateSpinBound(
  LatticeTiling *tiling,                        ///< [in] Lattice tiling.
  const gsl_matrix *rssky_transf,               ///< [in] Reduced supersky coordinate transform data
  const size_t s,                               ///< [in] Spindown order; 0=frequency, 1=first spindown, etc.
  const double bound1,                          ///< [in] First bound on frequency/spindown
  const double bound2                           ///< [in] Second bound on frequency/spindown
  );

///
/// Fill a PulsarSpinRange with the physical frequency/spindown ranges covered by a reduced supersky
/// lattice tiling.
///
int XLALSuperskyLatticePulsarSpinRange(
  PulsarSpinRange *spin_range,                  ///< [in,out] Physical frequency/spindown range
  LatticeTiling *tiling,                        ///< [in] Lattice tiling
  const gsl_matrix *rssky_transf                ///< [in] Reduced supersky coordinate transform data
  );

/// @}

#ifdef __cplusplus
}
#endif

#endif // _SUPERSKYMETRICS_H

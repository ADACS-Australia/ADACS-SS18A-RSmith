//
// Copyright (C) 2007, 2008, 2012, 2014, 2015 Karl Wette
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

#ifndef _LATTICETILING_H
#define _LATTICETILING_H

#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <lal/LALStdlib.h>
#include <lal/Random.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// \defgroup LatticeTiling_h Header LatticeTiling.h
/// \ingroup lalpulsar_templbank
/// \author Karl Wette
/// \brief Lattice-based template generation for constant-metric parameter spaces, described in
/// \cite Wette2009a and \cite Wette2014a .
///
/// @{
///

///
/// Describes a lattice tiling parameter-space bounds and metric.
///
typedef struct tagLatticeTiling LatticeTiling;

///
/// Iterates over all points in a lattice tiling.
///
typedef struct tagLatticeTilingIterator LatticeTilingIterator;

///
/// Locates the nearest point in a lattice tiling.
///
typedef struct tagLatticeTilingLocator LatticeTilingLocator;

///
/// Statistics related to the number/value of lattice tiling points in a dimension.
///
typedef struct tagLatticeTilingStats {
  UINT8 total_points;			///< Total number of points up to this dimension
  UINT4 min_points;			///< Minimum number of points in this dimension
  double avg_points;			///< Average number of points in this dimension
  UINT4 max_points;			///< Maximum number of points in this dimension
  double min_value;			///< Minimum value of points in this dimension
  double max_value;			///< Maximum value of points in this dimension
} LatticeTilingStats;

///
/// Function which returns a bound on a dimension of the lattice tiling.
///
typedef double(*LatticeTilingBound)(
  const void *data,			///< [in] Arbitrary data describing parameter space bound
  const size_t dim,			///< [in] Dimension on which bound applies
  const gsl_vector *point		///< [in] Point at which to find bound
  );

///
/// Create a new lattice tiling.
///
LatticeTiling *XLALCreateLatticeTiling(
  const size_t ndim			///< [in] Number of parameter-space dimensions
  );

///
/// Destroy a lattice tiling.
///
void XLALDestroyLatticeTiling(
  LatticeTiling *tiling			///< [in] Lattice tiling
  );

///
/// Set a parameter-space bound on a dimension of the lattice tiling.  The bound is described by a
/// function \c func, and two data of length \c data_len, \c data_lower and \c data_upper,
/// describing the lower and upper parameter space bounds respectively. If \c data_lower and \c
/// data_upper are identical, this parameter-space dimension will be treated as a single point, and
/// will not be tiled.
///
int XLALSetLatticeTilingBound(
  LatticeTiling *tiling,		///< [in] Lattice tiling
  const size_t dim,			///< [in] Dimension on which bound applies
  const LatticeTilingBound func,	///< [in] Parameter space bound function
  const size_t data_len,		///< [in] Length of arbitrary data describing parameter space bounds
  void *data_lower,			///< [in] Arbitrary data describing lower parameter space bound
  void *data_upper			///< [in] Arbitrary data describing upper parameter space bound
  );

///
/// Set a constant lattice tiling parameter-space bound, given by the minimum and maximum of the two
/// supplied bounds, on a dimension of the lattice tiling.
///
int XLALSetLatticeTilingConstantBound(
  LatticeTiling *tiling,		///< [in] Lattice tiling
  const size_t dim,			///< [in] Dimension on which bound applies
  const double bound1,			///< [in] First bound on dimension
  const double bound2			///< [in] Second bound on dimension
  );

///
/// Set the tiling lattice, parameter-space metric, and maximum prescribed mismatch.  The lattice
/// tiling \c tiling is now fully initialised, and can be used to create tiling iterators [via
/// XLALCreateLatticeTilingIterator()] and locators [via XLALCreateLatticeTilingLocator()].
///
/// Valid lattice names are:
/// - Cubic (\f$Z_n\f$) lattice: \c Zn or \c Cubic
/// - An-star (\f$A_n^*\f$) lattice: \c Ans or \c An-star
///
int XLALSetTilingLatticeAndMetric(
  LatticeTiling *tiling,		///< [in] Lattice tiling
  const char *lattice_name,		///< [in] Name of lattice to generate tiling with
  const gsl_matrix *metric,		///< [in] Parameter-space metric
  const double max_mismatch		///< [in] Maximum prescribed mismatch
  );

///
/// Return the total number of dimensions of the lattice tiling.
///
size_t XLALTotalLatticeTilingDimensions(
  const LatticeTiling *tiling		///< [in] Lattice tiling
  );

///
/// Return the number of tiled (i.e. not a single point) dimensions of the lattice tiling.
///
size_t XLALTiledLatticeTilingDimensions(
  const LatticeTiling *tiling		///< [in] Lattice tiling
  );

///
/// Return the step size of the lattice tiling in a given dimension, or 0 for non-tiled dimensions.
///
REAL8 XLALLatticeTilingStepSizes(
  const LatticeTiling *tiling,		///< [in] Lattice tiling
  const size_t dim			///< [in] Dimension of which to return step size
  );

///
/// Return the bounding box extent of the lattice tiling in a given dimension, or 0 for non-tiled
/// dimensions.
///
REAL8 XLALLatticeTilingBoundingBox(
  const LatticeTiling *tiling,		///< [in] Lattice tiling
  const size_t dim			///< [in] Dimension of which to return bounding box extent
  );

///
/// Return statistics related to the number/value of lattice tiling points in a dimension.
///
const LatticeTilingStats *XLALLatticeTilingStatistics(
  LatticeTiling *tiling,		///< [in] Lattice tiling
  const size_t dim			///< [in] Dimension in which to return statistics
  );

///
/// Generate random points within the parameter space of the lattice tiling.  Points can be scaled
/// to fill the parameter space exactly (<tt>scale == 0</tt>), fill a subset of the parameter space
/// (<tt>-1 < scale < 0</tt>), or fill outside the parameter space (<tt>scale > 0</tt>).
///
int XLALRandomLatticeTilingPoints(
  const LatticeTiling *tiling,		///< [in] Lattice tiling
  const double scale,			///< [in] Scale of random points
  RandomParams *rng,			///< [in] Random number generator
  gsl_matrix *random_points		///< [out] Matrix whose columns are the random points
  );

///
/// Allocate and return vectors containing the bounds on neighbouring dimesions of the lattice
/// tiling parameter space.
///
int XLALLatticeTilingDimensionBounds(
  const LatticeTiling *tiling,		///< [in] Lattice tiling
  const bool padding,			///< [in] Whether padding is added to parameter space bounds
  const gsl_vector *point,		///< [in] Point at which to return bounds
  const size_t y_dim,			///< [in] Dimension 'y' of which to return bounds
  const double x_scale,			///< [in] Scale of steps in 'x', in units of lattice step size
  gsl_vector **y_lower,			///< [in] Lower bounds of dimension 'y' as function of 'x'
  gsl_vector **y_upper,			///< [in] Upper bounds of dimension 'y' as function of 'x'
  gsl_vector **x			///< [in] Values 'x' in dimension 'y-1'
  );

///
/// Create a new lattice tiling iterator.
///
#ifdef SWIG // SWIG interface directives
SWIGLAL(OWNED_BY_1ST_ARG(int, XLALCreateLatticeTilingIterator));
#endif
LatticeTilingIterator *XLALCreateLatticeTilingIterator(
  const LatticeTiling *tiling,		///< [in] Lattice tiling
  const size_t itr_ndim			///< [in] Number of parameter-space dimensions to iterate over
  );

///
/// Destroy a lattice tiling iterator.
///
void XLALDestroyLatticeTilingIterator(
  LatticeTilingIterator *itr		///< [in] Lattice tiling iterator
  );

///
/// Set whether the lattice tiling iterator should alternate its iteration direction (i.e. lower to
/// upper bound, then upper to lower bound, and so on) after every crossing of each dimension.
///
int XLALSetLatticeTilingAlternatingIterator(
  LatticeTilingIterator *itr,		///< [in] Lattice tiling iterator
  const bool alternating		///< [in] If true, set alternating iterator
  );

///
/// Reset an iterator to the beginning of a lattice tiling.
///
int XLALResetLatticeTilingIterator(
  LatticeTilingIterator *itr		///< [in] Lattice tiling iterator
  );

///
/// Advance lattice tiling iterator, and optionally return the next point in \c point. Returns >0
/// if there are points remaining, 0 if there are no more points, and XLAL_FAILURE on error.
///
int XLALNextLatticeTilingPoint(
  LatticeTilingIterator *itr,		///< [in] Lattice tiling iterator
  gsl_vector *point			///< [out] Next point in lattice tiling
  );

///
/// Advance lattice tiling iterator, and optionally return the next set of points in \c points.
/// Returns the number of points stored in \c points if there are points remaining, 0 if there
/// are no more points, and XLAL_FAILURE on error.
///
#ifdef SWIG // SWIG interface directives
SWIGLAL(RETURN_VALUE(int, XLALNextLatticeTilingPoints));
SWIGLAL(INOUT_STRUCTS(gsl_matrix **, points));
#endif
int XLALNextLatticeTilingPoints(
  LatticeTilingIterator *itr,		///< [in] Lattice tiling iterator
  gsl_matrix **points			///< [out] Columns are next set of points in lattice tiling
  );

///
/// Return the total number of points covered by the lattice tiling iterator.
///
UINT8 XLALTotalLatticeTilingPoints(
  LatticeTilingIterator *itr		///< [in] Lattice tiling iterator
  );

///
/// Return the index of the current point in the lattice tiling iterator.
///
UINT8 XLALCurrentLatticeTilingIndex(
  const LatticeTilingIterator *itr	///< [in] Lattice tiling iterator
  );

///
/// Return the number of points to the left/right of the current point in a given dimension.
///
int XLALCurrentLatticeTilingBlock(
  const LatticeTilingIterator *itr,	///< [in] Lattice tiling iterator
  const size_t dim,			///< [in] Dimension in which to return number of points
  UINT4 *left,				///< [out] Number of points to left of current point
  UINT4 *right				///< [out] Number of points to right of current point
  );

///
/// Create a new lattice tiling locator. If there are tiled dimensions, an index trie is internally built.
///
#ifdef SWIG // SWIG interface directives
SWIGLAL(OWNED_BY_1ST_ARG(int, XLALCreateLatticeTilingLocator));
#endif
LatticeTilingLocator *XLALCreateLatticeTilingLocator(
  const LatticeTiling *tiling		///< [in] Lattice tiling
  );

///
/// Destroy a lattice tiling locator.
///
void XLALDestroyLatticeTilingLocator(
  LatticeTilingLocator *loc		///< [in] Lattice tiling locator
  );

///
/// Locate the nearest point in a lattice tiling to a given point. Return optionally the nearest
/// point in \c nearest_point, and sequential indexes, unique up to each dimension, to the nearest
/// point in \c nearest_index.
///
int XLALNearestLatticeTilingPoint(
  const LatticeTilingLocator *loc,	///< [in] Lattice tiling locator
  const gsl_vector *point,		///< [in] Point for which to find nearest point
  gsl_vector *nearest_point,		///< [out] Nearest point
  UINT8Vector *nearest_index		///< [out] Unique sequential indexes of the nearest point
  );

///
/// Locate the nearest points in a lattice tiling to a given set of points. Return the nearest
/// points in \c nearest_points, and optionally sequential indexes, unique up to each dimension,
/// to the nearest points in \c nearest_seqs_idxs. Outputs are dynamically resized as required.
///
#ifdef SWIG // SWIG interface directives
SWIGLAL(INOUT_STRUCTS(gsl_matrix **, nearest_points));
SWIGLAL(INOUT_STRUCTS(UINT8VectorSequence **, nearest_indexes));
#endif
int XLALNearestLatticeTilingPoints(
  const LatticeTilingLocator *loc,		///< [in] Lattice tiling locator
  const gsl_matrix *points,			///< [in] Columns are set of points for which to find nearest points
  gsl_matrix **nearest_points,			///< [out] Columns are the corresponding nearest points
  UINT8VectorSequence **nearest_indexes		///< [out] Vectors are unique sequential indexes of the nearest points
  );

///
/// Locate the nearest point in a lattice tiling to a given point. Return the nearest point in
/// \c nearest_point, the unique sequential index in dimension <tt>dim-1</tt> to the nearest point
/// in \c nearest_index, and the number of points to the left/right of the nearest point in dimension
/// <tt>dim</tt> in \c nearest_left and \c nearest_right respectively.
///
int XLALNearestLatticeTilingBlock(
  const LatticeTilingLocator *loc,	///< [in] Lattice tiling locator
  const gsl_vector *point,		///< [in] Point for which to find nearest point
  const size_t dim,			///< [in] Dimension for which to return indexes
  gsl_vector *nearest_point,		///< [out] Nearest point
  UINT8 *nearest_index,			///< [out] Unique sequential index of the nearest point in <tt>dim-1</tt>
  UINT4 *nearest_left,			///< [out] Number of points to left of nearest point in <tt>dim</tt>
  UINT4 *nearest_right			///< [out] Number of points to right of nearest point in <tt>dim</tt>
  );

///
/// Print the internal index trie of a lattice tiling locator to the given file pointer.
///
int XLALPrintLatticeTilingIndexTrie(
  const LatticeTilingLocator *loc,	///< [in] Lattice tiling locator
  FILE *file				///< [in] File pointer to print trie to
  );

/// @}

#ifdef __cplusplus
}
#endif

#endif // _LATTICETILING_H

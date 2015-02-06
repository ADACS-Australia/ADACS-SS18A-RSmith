/*
 *  LALInferenceCreateROQ.c: Reduced order quadrature basis and interpolant generation
 *
 *  Copyright (C) 2014 Matthew Pitkin, Rory Smith
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

#include <lal/LALInferenceGenerateROQ.h>

/* internal function definitions */

/* define function to project model vector onto the training set of models */
void project_onto_basis(gsl_vector *weight,
                        gsl_matrix *RB,
                        gsl_matrix *TS,
                        gsl_matrix *projections,
                        INT4 idx,
                        gsl_matrix *projection_coefficients);

void complex_project_onto_basis(gsl_vector *weight,
                                gsl_matrix_complex *RB,
                                gsl_matrix_complex *TS,
                                gsl_matrix_complex *projections,
                                INT4 idx,
                                gsl_matrix_complex *projection_coefficients);

/* the dot product of two real vectors scaled by a weighting factor */
REAL8 weighted_dot_product(gsl_vector *weight, gsl_vector *a, gsl_vector *b);

/* the dot product of two complex vectors scaled by a weighting factor */
gsl_complex complex_weighted_dot_product(gsl_vector *weight, gsl_vector_complex *a, gsl_vector_complex *b);

void normalise(gsl_vector *weight, gsl_vector *a);
void complex_normalise(gsl_vector *weight, gsl_vector_complex *a);

void normalise_training_set(gsl_vector *weight, gsl_matrix *TS);
void complex_normalise_training_set(gsl_vector *weight, gsl_matrix_complex *TS);

/* get the B_matrix */
gsl_matrix *B_matrix(gsl_matrix *V, gsl_matrix *RB);
gsl_matrix_complex *complex_B_matrix(gsl_matrix_complex *V, gsl_matrix_complex *RB);

/* find the index of the absolute maximum value for a complex vector */
int complex_vector_maxabs_index( gsl_vector_complex *c );

/** \brief Function to project the training set onto a given basis vector
 *
 * This is an internal function to be used by \c LALInferenceCreateREAL8OrthonormalBasis
 *
 * @param[in] weight The normalisation weight(s) for the training set waveforms (e.g. time or frequency step(s) size)
 * @param[in] RB The reduced basis set
 * @param[in] TS The training set of waveforms
 * @param[in] projections The set of projections (this is updated in this function)
 * @param[in] idx The index of the reduced basis vector that the training set will project onto
 * @param[out] projection_coefficients The projection coefficients for each basis vector and each training waveform
 */
void project_onto_basis(gsl_vector *weight,
                        gsl_matrix *RB,
                        gsl_matrix *TS,
                        gsl_matrix *projections,
                        INT4 idx,
                        gsl_matrix *projection_coefficients){
  size_t row = 0;
  gsl_vector_view basis;

  XLAL_CALLGSL( basis = gsl_matrix_row(RB, idx) );

  #pragma omp parallel for
  for ( row=0; row < TS->size1; row++ ){
    double prod;
    gsl_vector_view proj, model;
    gsl_vector *basisscale;

    XLAL_CALLGSL( proj = gsl_matrix_row(projections, row) );
    XLAL_CALLGSL( basisscale = gsl_vector_calloc(TS->size2) );

    XLAL_CALLGSL( model = gsl_matrix_row(TS, row) ); /* get model from training set */

    prod = weighted_dot_product(weight, &basis.vector, &model.vector);

    XLAL_CALLGSL( gsl_matrix_set(projection_coefficients, idx, row, prod) );
    XLAL_CALLGSL( gsl_vector_memcpy(basisscale, &basis.vector) );
    XLAL_CALLGSL( gsl_vector_scale(basisscale, prod) );
    XLAL_CALLGSL( gsl_vector_add(&proj.vector, basisscale) );
    XLAL_CALLGSL( gsl_vector_free(basisscale) );
  }
}


/** \brief Function to project the complex training set onto a given basis vector
 *
 * This is an internal function to be used by \c LALInferenceCreateCOMPLEX16OrthonormalBasis
 *
 * @param[in] weight The normalisation weight(s) for the training set waveforms (e.g. time or frequency step(s) size)
 * @param[in] RB The reduced basis set
 * @param[in] TS The training set of waveforms
 * @param[in] projections The set of projections (this is updated in this function)
 * @param[in] idx The index of the reduced basis vector that the training set will project onto
 * @param[out] projection_coefficients The projection coefficients for each basis vector and each training waveform
 */
void complex_project_onto_basis(gsl_vector *weight,
                                gsl_matrix_complex *RB,
                                gsl_matrix_complex *TS,
                                gsl_matrix_complex *projections,
                                INT4 idx,
                                gsl_matrix_complex *projection_coefficients){
  size_t row = 0;
  gsl_vector_complex_view basis;

  XLAL_CALLGSL( basis = gsl_matrix_complex_row(RB, idx) );

  #pragma omp parallel for
  for ( row=0; row < TS->size1; row++ ){
    gsl_complex cprod;
    gsl_vector_complex_view proj, model;
    gsl_vector_complex *basisscale;

    XLAL_CALLGSL( proj = gsl_matrix_complex_row(projections, row) );
    XLAL_CALLGSL( basisscale = gsl_vector_complex_calloc(TS->size2) );

    XLAL_CALLGSL( model = gsl_matrix_complex_row(TS, row) ); /* get model from training set */

    cprod = complex_weighted_dot_product(weight, &basis.vector, &model.vector);

    XLAL_CALLGSL( gsl_matrix_complex_set(projection_coefficients, idx, row, cprod) );
    XLAL_CALLGSL( gsl_vector_complex_memcpy(basisscale, &basis.vector) );
    XLAL_CALLGSL( gsl_vector_complex_scale(basisscale, cprod) );
    XLAL_CALLGSL( gsl_vector_complex_add(&proj.vector, basisscale) );
    XLAL_CALLGSL( gsl_vector_complex_free(basisscale) );
  }
}


/** \brief The dot product of two real vectors scaled by a given weight factor
 *
 * @param[in] weight A (set of) scaling factor(s) for the dot product
 * @param[in] a The first vector
 * @param[in] b The second vector
 *
 * @return The real dot product of the two vectors
 */
REAL8 weighted_dot_product(gsl_vector *weight, gsl_vector *a, gsl_vector *b){
  REAL8 dp;
  gsl_vector *weighted;

  XLAL_CHECK_REAL8( a->size == b->size, XLAL_EFUNC, "Size of input vectors are not the same.");

  XLAL_CALLGSL( weighted = gsl_vector_calloc(a->size) );
  XLAL_CALLGSL( gsl_vector_memcpy(weighted, a) );

  /* multiply vector by weight */
  if ( weight->size == 1 ){ /* just a single weight to scale with */
    XLAL_CALLGSL( gsl_vector_scale(weighted, gsl_vector_get(weight, 0)) );
  }
  else if ( weight->size == a->size ){
    XLAL_CALLGSL( gsl_vector_mul(weighted, weight) );
  }
  else{
    XLAL_ERROR_REAL8( XLAL_EFUNC, "Vector of weights must either contain a single value, or be the same length as the other input vectors." );
  }

  /* get dot product */
  XLAL_CALLGSL( gsl_blas_ddot(weighted, b, &dp) );

  XLAL_CALLGSL( gsl_vector_free(weighted) );

  return dp;
}


/** \brief The dot product of two complex vectors scaled by a given weight factor
 *
 * The dot product is produced using the complex conjugate of the first vector.
 *
 * @param[in] weight A real scaling factor for the dot product
 * @param[in] a The first complex vector
 * @param[in] b The second complex vector
 *
 * @return The absolute value of the complex dot product of the two vectors
 */
gsl_complex complex_weighted_dot_product(gsl_vector *weight, gsl_vector_complex *a, gsl_vector_complex *b){
  gsl_complex dp;
  gsl_vector_complex *weighted;

  if ( a->size != b->size ){ XLAL_PRINT_ERROR( "Size of input vectors are not the same." ); }

  XLAL_CALLGSL( weighted = gsl_vector_complex_calloc(a->size) );
  XLAL_CALLGSL( gsl_vector_complex_memcpy(weighted, a) );

  /* multiply vector by weight */
  if ( weight->size == 1 ){ /* just a single weight to scale with */
    XLAL_CALLGSL( gsl_blas_zdscal(gsl_vector_get(weight, 0), weighted) );
  }
  else if ( weight->size == a->size ){
    gsl_vector_view rview, iview;

    XLAL_CALLGSL( rview = gsl_vector_complex_real(weighted) );
    XLAL_CALLGSL( iview = gsl_vector_complex_imag(weighted) );

    XLAL_CALLGSL( gsl_vector_mul(&rview.vector, weight) );
    XLAL_CALLGSL( gsl_vector_mul(&iview.vector, weight) );
  }
  else{
    XLAL_PRINT_ERROR( "Vector of weights must either contain a single value, or be the same length as the other input vectors." );
  }

  /* get dot product (taking the complex conjugate of the first vector) */
  XLAL_CALLGSL( gsl_blas_zdotc(weighted, b, &dp) );
  XLAL_CALLGSL( gsl_vector_complex_free(weighted) );

  return dp;
}


/** \brief Normalise a real vector with a given weighting
 *
 * @param[in] weight The weighting(s) in the normalisation (e.g. time of frequency step(s) between points)
 * @param[in] a The vector to be normalise (this will be change by the function to return the
 * normalised vector.
 */
void normalise(gsl_vector *weight, gsl_vector *a){
  double norm;

  if ( weight->size == 1 ){
    XLAL_CALLGSL( norm = gsl_blas_dnrm2(a) ); /* use GSL normalisation calculation function */
    XLAL_CALLGSL( gsl_vector_scale(a, 1./(norm*sqrt(gsl_vector_get(weight, 0)))) );
  }
  else if ( weight->size == a->size ){
    norm = 1./sqrt(weighted_dot_product(weight, a, a));
    XLAL_CALLGSL( gsl_vector_scale(a, norm) );
  }
  else{
    XLAL_ERROR_VOID( XLAL_EFUNC, "Vector of weights must either contain a single value, or be the same length as the other input vectors." );
  }
}


/** \brief Normalise a complex vector with a given (real) weighting
 *
 * @param[in] weight The weighting(s) in the normalisation (e.g. time of frequency step(s) between points)
 * @param[in] a The vector to be normalise (this will be change by the function to return the
 * normalised vector.
 */
void complex_normalise(gsl_vector *weight, gsl_vector_complex *a){
  double norm;

  if ( weight->size == 1 ){
    XLAL_CALLGSL( norm = gsl_blas_dznrm2(a) ); /* use GSL normalisation calculation function */
    XLAL_CALLGSL( gsl_blas_zdscal(1./(norm*sqrt(gsl_vector_get(weight, 0))), a) );
  }
  else if ( weight->size == a->size ){
    norm = 1./sqrt(gsl_complex_abs(complex_weighted_dot_product(weight, a, a)));
    XLAL_CALLGSL( gsl_blas_zdscal(norm, a) );
  }
  else{
    XLAL_ERROR_VOID( XLAL_EFUNC, "Vector of weights must either contain a single value, or be the same length as the other input vectors." );
  }
}

/** \brief Normalise the set of training waveforms
 *
 * This function will normalise a set of training waveforms. This will be used within the
 * \a LALInferenceCreateREAL8OrthonormalBasis function.
 *
 * @param[in] weight The e.g. time/frequency step in the waveforms used to normalise the waveforms
 * @param[in] TS The training set to be normalised.
 */
void normalise_training_set(gsl_vector *weight, gsl_matrix *TS){
  gsl_vector_view rowview;
  size_t i = 0;
  for ( i=0; i<TS->size1; i++ ){
    XLAL_CALLGSL( rowview = gsl_matrix_row(TS, i) );
    normalise(weight, &rowview.vector);
  }
}


/** \brief Normalise the set of complex training waveforms
 *
 * This function will normalise a set of complex training waveforms. This will be used within the
 * \a LALInferenceCreateCOMPLEX16OrthonormalBasis function.
 *
 * @param[in] weight The e.g. time/frequency step in the waveforms used to normalise the waveforms
 * @param[in] TS The training set to be normalised.
 */
void complex_normalise_training_set(gsl_vector *weight, gsl_matrix_complex *TS){
  gsl_vector_complex_view rowview;
  size_t i = 0;
  for ( i=0; i<TS->size1; i++ ){
    XLAL_CALLGSL( rowview = gsl_matrix_complex_row(TS, i) );
    complex_normalise(weight, &rowview.vector);
  }
}


/** \brief Get the interpolant of a reduced basis set
 *
 * This is used internally by \c LALInferenceCreateREALROQInterpolant when
 * iteratively calculating the interpolation matrix.
 *
 * @param[in] V The matrix containing the basis vector points at the current interpolation nodes
 * @param[in] RB The set of basis vectors
 *
 * @return The interpolant matrix
 */
gsl_matrix *B_matrix(gsl_matrix *V, gsl_matrix *RB){
  /* get inverse of V */
  size_t n = V->size1;
  gsl_matrix *invV, *LU, *B;
  gsl_permutation *p;
  gsl_matrix_view subRB;

  XLAL_CALLGSL( invV = gsl_matrix_alloc(n, n) );
  int signum;

  /* use LU decomposition to get inverse */
  XLAL_CALLGSL( LU = gsl_matrix_alloc(n, n) );
  XLAL_CALLGSL( gsl_matrix_memcpy(LU, V) );

  XLAL_CALLGSL( p = gsl_permutation_alloc(n) );
  XLAL_CALLGSL( gsl_linalg_LU_decomp(LU, p, &signum) );
  XLAL_CALLGSL( gsl_linalg_LU_invert(LU, p, invV) );
  XLAL_CALLGSL( gsl_permutation_free(p) );
  XLAL_CALLGSL( gsl_matrix_free(LU) );

  /* get B matrix */
  XLAL_CALLGSL( B = gsl_matrix_alloc(n, RB->size2) );
  XLAL_CALLGSL( subRB = gsl_matrix_submatrix(RB, 0, 0, n, RB->size2) );
  XLAL_CALLGSL( gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, invV, &subRB.matrix, 0., B) );

  XLAL_CALLGSL( gsl_matrix_free(invV) );

  return B;
}


/** \brief Get the interpolant of a complex reduced basis set
 *
 * This is used internally by \c LALInferenceCreateCOMPLEXROQInterpolant when
 * iteratively calculating the interpolation matrix.
 *
 * @param[in] V The matrix containing the basis vector points at the current interpolation nodes
 * @param[in] RB The set of basis vectors
 *
 * @return The interpolant matrix
 */
gsl_matrix_complex *complex_B_matrix(gsl_matrix_complex *V, gsl_matrix_complex *RB){
  /* get inverse of V */
  size_t n = V->size1;
  gsl_matrix_complex *invV, *LU, *B;
  gsl_permutation *p;
  gsl_matrix_complex_view subRB;
  gsl_complex scale1, scale0;

  GSL_SET_COMPLEX(&scale1, 1., 0.);
  GSL_SET_COMPLEX(&scale0, 0., 0.);

  XLAL_CALLGSL( invV = gsl_matrix_complex_alloc(n, n) );
  int signum;

  /* use LU decomposition to get inverse */
  XLAL_CALLGSL( LU = gsl_matrix_complex_alloc(n, n) );
  XLAL_CALLGSL( gsl_matrix_complex_memcpy(LU, V) );

  XLAL_CALLGSL( p = gsl_permutation_alloc(n) );
  XLAL_CALLGSL( gsl_linalg_complex_LU_decomp(LU, p, &signum) );
  XLAL_CALLGSL( gsl_linalg_complex_LU_invert(LU, p, invV) );
  XLAL_CALLGSL( gsl_permutation_free(p) );
  XLAL_CALLGSL( gsl_matrix_complex_free(LU) );

  /* get B matrix */
  XLAL_CALLGSL( B = gsl_matrix_complex_alloc(n, RB->size2) );
  XLAL_CALLGSL( subRB = gsl_matrix_complex_submatrix(RB, 0, 0, n, RB->size2) );
  XLAL_CALLGSL( gsl_blas_zgemm(CblasTrans, CblasNoTrans, scale1, invV, &subRB.matrix, scale0, B) );

  XLAL_CALLGSL( gsl_matrix_complex_free(invV) );

  return B;
}

/** \brief Get the index of the maximum absolute value for a complex vector
 *
 * @param[in] c A complex vector
 *
 * @return The index of the maximum absolute value of that vector
 */
int complex_vector_maxabs_index( gsl_vector_complex *c ){
  double maxv = -INFINITY, absval = 0.;
  int idx = 0;
  size_t i = 0;

  for ( i=0; i<c->size; i++ ){
    XLAL_CALLGSL( absval = gsl_complex_abs(gsl_vector_complex_get(c, i)) );

    if ( absval > maxv ){
      maxv = absval;
      idx = (int)i;
    }
  }

  return idx;
}

/* main functions */

/**
 * \brief Create a orthonormal basis set from a training set of real waveforms
 *
 * Given a \c gsl_matrix containing a training set of real waveforms (where the waveforms
 * are created at time or frequency steps seperated by \a delta) an orthonormal basis
 * will be generated using the greedy binning Algorithm 1 of \cite FGHKT2014 . The stopping
 * criteria for the algorithm is controlled by the \a tolerance value, which defined the
 * maximum residual between the current basis set (at a given iteration) and the training
 * set (and example tolerance is \f$10^{-12}\f$. In this function the training set will be
 * normalised, so the input \a TS will be modified.
 *
 * @param[in] delta A vector containging the time/frequency step(s) in the training set used
 * to normalise the models. This can be a vector containing only one value.
 * @param[in] tolerance The tolerance used as a stopping criteria for the basis generation.
 * @param[in] TS A matrix containing the training set, where the number of waveforms in the
 * training set is given by the rows and the waveform points by the columns.
 * @param[out] nbases The number of orthonormal bases produced
 *
 * @return A REAL8 array containing the basis set (best viewed later using a \c gsl_matrix_view).
 */
REAL8 *LALInferenceGenerateREAL8OrthonormalBasis(gsl_vector *delta,
                                                 REAL8 tolerance,
                                                 gsl_matrix *TS,
                                                 size_t *nbases){
  REAL8 *RB = NULL;
  gsl_matrix_view RBview;

  gsl_matrix *projections; /* projections of the basis onto the training set */
  gsl_matrix *residual;
  gsl_matrix *projection_coeffs;
  gsl_vector *projection_errors;
  gsl_vector *firstrow;

  REAL8 sigma = 1.;
  size_t dlength = TS->size2, nts = TS->size1;
  size_t mindex = 0, k=0;
  INT4 idx = 0;

  /* normalise the training set */
  normalise_training_set(delta, TS);

  /* allocate reduced basis (initially just one model vector in length) */
  RB = XLALCalloc(dlength, sizeof(REAL8));

  XLAL_CALLGSL( RBview = gsl_matrix_view_array(RB, 1, dlength) );
  XLAL_CALLGSL( firstrow = gsl_vector_calloc(dlength) );
  XLAL_CALLGSL( gsl_matrix_get_row(firstrow, TS, 0) );
  XLAL_CALLGSL( gsl_matrix_set_row(&RBview.matrix, 0, firstrow) );
  XLAL_CALLGSL( gsl_vector_free(firstrow) );

  XLAL_CALLGSL( projection_errors = gsl_vector_calloc(dlength) );
  XLAL_CALLGSL( residual = gsl_matrix_calloc(nts, dlength) );
  XLAL_CALLGSL( projections = gsl_matrix_calloc(nts, dlength) );
  XLAL_CALLGSL( projection_coeffs = gsl_matrix_calloc(nts, nts) );

  /* create reduced basis set using greedy binning Algorithm 1 of http://arxiv.org/abs/1308.3565 */
  while ( 1 ){
    gsl_vector *next_basis;
    gsl_vector_view proj_basis;

    XLAL_CHECK_NULL( idx < (INT4)nts, XLAL_EFUNC,
                     "Not enough training models (%zu) to produce orthonormal basis given the tolerance of %le\n", nts, tolerance );

    project_onto_basis(delta, &RBview.matrix, TS, projections, idx, projection_coeffs);

    XLAL_CALLGSL( gsl_matrix_memcpy(residual, TS) ); /* copy training set into residual */

    /* get residuals by subtracting projections from training set */
    XLAL_CALLGSL( gsl_matrix_sub(residual, projections) );

    /* get projection errors */
    #pragma omp parallel for
    for( k=0; k < nts; k++ ){
      REAL8 err;
      gsl_vector_view resrow;

      XLAL_CALLGSL( resrow = gsl_matrix_row(residual, k) );

      err = weighted_dot_product(delta, &resrow.vector, &resrow.vector);

      XLAL_CALLGSL( gsl_vector_set(projection_errors, k, err) );
    }

    sigma = fabs(gsl_vector_max(projection_errors));

    //if ( sigma > 1e-5 ){ fprintf(stderr, "%.12lf\t%d\n", sigma, idx); }
    //else { fprintf(stderr, "%.12le\t%d\n", sigma, idx); }

    /* break point */
    if ( sigma < tolerance ) {
      idx++;
      break;
    }

    /* get index of training set with the largest projection errors */
    XLAL_CALLGSL( mindex = gsl_vector_max_index( projection_errors ) );

    XLAL_CALLGSL( next_basis = gsl_vector_calloc(dlength) );
    XLAL_CALLGSL( proj_basis = gsl_matrix_row(projections, mindex) );
    XLAL_CALLGSL( gsl_matrix_get_row(next_basis, TS, mindex) );
    XLAL_CALLGSL( gsl_vector_sub(next_basis, &proj_basis.vector) );

    /* normalise vector */
    normalise(delta, next_basis);

    idx++;

    /* expand reduced basis */
    RB = XLALRealloc(RB, sizeof(REAL8)*dlength*(idx+1));

    /* add on next basis */
    XLAL_CALLGSL( RBview = gsl_matrix_view_array(RB, idx+1, dlength) );
    XLAL_CALLGSL( gsl_matrix_set_row(&RBview.matrix, idx, next_basis) );

    XLAL_CALLGSL( gsl_vector_free(next_basis) );
  }

  *nbases = (size_t)idx;

  /* free memory */
  XLAL_CALLGSL( gsl_matrix_free(projections) );
  XLAL_CALLGSL( gsl_matrix_free(projection_coeffs) );
  XLAL_CALLGSL( gsl_vector_free(projection_errors) );
  XLAL_CALLGSL( gsl_matrix_free(residual) );

  return RB;
}


/**
 * \brief Create a orthonormal basis set from a training set of complex waveforms
 *
 * Given a \c gsl_matrix containing a training set of complex waveforms (where the waveforms
 * are created at time or frequency steps seperated by \a delta) an orthonormal basis
 * will be generated using the greedy binning Algorithm 1 of \cite FGHKT2014 . The stopping
 * criteria for the algorithm is controlled by the \a tolerance value, which defined the
 * maximum residual between the current basis set (at a given iteration) and the training
 * set (and example tolerance is \f$10^{-12}\f$. In this function the training set will be
 * normalised, so the input \a TS will be modified.
 *
 * Note that in this function we have to cast the \c COMPLEX16 array as a double to use
 * \c gsl_matrix_view_array, which assume that the data is passed as a double array with
 * memory laid out so that adjacent double memory blocks hold the corresponding real and
 * imaginary parts.
 *
 * @param[in] delta The time/frequency step(s) in the training set used to normalise the models.
 * This can be a vector containing just one value.
 * @param[in] tolerance The tolerance used as a stopping criteria for the basis generation.
 * @param[in] TS A matrix containing the complex training set, where the number of waveforms in the
 * training set is given by the rows and the waveform points by the columns.
 * @param[out] nbases The number of orthonormal bases produced
 *
 * @return A COMPLEX16 array containing the basis set (best viewed later using a \c gsl_matrix_complex_view).
 */
COMPLEX16 *LALInferenceGenerateCOMPLEX16OrthonormalBasis(gsl_vector *delta,
                                                         REAL8 tolerance,
                                                         gsl_matrix_complex *TS,
                                                         size_t *nbases){
  COMPLEX16 *RB = NULL;
  gsl_matrix_complex_view RBview;

  gsl_matrix_complex *projections; /* projections of the basis onto the training set */
  gsl_matrix_complex *residual;
  gsl_matrix_complex *projection_coeffs;
  gsl_vector *projection_errors;
  gsl_vector_complex *firstrow;

  REAL8 sigma = 1.;
  size_t dlength = TS->size2, nts = TS->size1;
  size_t mindex = 0, k=0;
  INT4 idx = 0;

  /* normalise the training set */
  complex_normalise_training_set(delta, TS);

  /* allocate reduced basis (initially just one model vector in length) */
  RB = XLALCalloc(dlength, sizeof(COMPLEX16));

  XLAL_CALLGSL( RBview = gsl_matrix_complex_view_array((double*)RB, 1, dlength) );
  XLAL_CALLGSL( firstrow = gsl_vector_complex_calloc(dlength) );
  XLAL_CALLGSL( gsl_matrix_complex_get_row(firstrow, TS, 0) );
  XLAL_CALLGSL( gsl_matrix_complex_set_row(&RBview.matrix, 0, firstrow) );
  XLAL_CALLGSL( gsl_vector_complex_free(firstrow) );

  XLAL_CALLGSL( projection_errors = gsl_vector_calloc(dlength) );
  XLAL_CALLGSL( residual = gsl_matrix_complex_calloc(nts, dlength) );
  XLAL_CALLGSL( projections = gsl_matrix_complex_calloc(nts, dlength) );
  XLAL_CALLGSL( projection_coeffs = gsl_matrix_complex_calloc(nts, nts) );

  /* create reduced basis set using greedy binning Algorithm 1 of http://arxiv.org/abs/1308.3565 */
  while ( 1 ){
    gsl_vector_complex *next_basis;
    gsl_vector_complex_view proj_basis;

    XLAL_CHECK_NULL( idx < (INT4)nts, XLAL_EFUNC,
                     "Not enough training models (%zu) to produce orthonormal basis given the tolerance of %le\n", nts, tolerance );

    complex_project_onto_basis(delta, &RBview.matrix, TS, projections, idx, projection_coeffs);

    XLAL_CALLGSL( gsl_matrix_complex_memcpy(residual, TS) ); /* copy training set into residual */

    /* get residuals by subtracting projections from training set */
    XLAL_CALLGSL( gsl_matrix_complex_sub(residual, projections) );

    /* get projection errors */
    #pragma omp parallel for
    for( k=0; k < nts; k++ ){
      gsl_complex err;
      gsl_vector_complex_view resrow;

      XLAL_CALLGSL( resrow = gsl_matrix_complex_row(residual, k) );

      err = complex_weighted_dot_product(delta, &resrow.vector, &resrow.vector);

      XLAL_CALLGSL( gsl_vector_set(projection_errors, k, GSL_REAL(err)) );
    }

    sigma = fabs(gsl_vector_max(projection_errors));

    //if ( sigma > 1e-5 ){ fprintf(stderr, "%.12lf\t%d\n", sigma, idx); }
    //else { fprintf(stderr, "%.12le\t%d\n", sigma, idx); }

    /* break point */
    if ( sigma < tolerance ) {
      idx++;
      break;
    }

    /* get index of training set with the largest projection errors */
    XLAL_CALLGSL( mindex = gsl_vector_max_index( projection_errors ) );

    XLAL_CALLGSL( next_basis = gsl_vector_complex_calloc(dlength) );
    XLAL_CALLGSL( proj_basis = gsl_matrix_complex_row(projections, mindex) );
    XLAL_CALLGSL( gsl_matrix_complex_get_row(next_basis, TS, mindex) );
    XLAL_CALLGSL( gsl_vector_complex_sub(next_basis, &proj_basis.vector) );

    /* normalise vector */
    complex_normalise(delta, next_basis);

    idx++;

    /* expand reduced basis */
    RB = XLALRealloc(RB, sizeof(COMPLEX16)*dlength*(idx+1));

    /* add on next basis */
    XLAL_CALLGSL( RBview = gsl_matrix_complex_view_array((double*)RB, idx+1, dlength) );
    XLAL_CALLGSL( gsl_matrix_complex_set_row(&RBview.matrix, idx, next_basis) );

    XLAL_CALLGSL( gsl_vector_complex_free(next_basis) );
  }

  *nbases = (size_t)idx;

  /* free memory */
  XLAL_CALLGSL( gsl_matrix_complex_free(projections) );
  XLAL_CALLGSL( gsl_matrix_complex_free(projection_coeffs) );
  XLAL_CALLGSL( gsl_vector_free(projection_errors) );
  XLAL_CALLGSL( gsl_matrix_complex_free(residual) );

  return RB;
}


/**
 * \brief Test the real reduced basis against another set of waveforms
 *
 * This function projects a set of waveforms onto the reduced basis and
 * checks that the residuals are within a given tolerance
 *
 * @param[in] delta The time/frequency step(s) in the training set used to normalise the models.
 * This can be a vector containing just one value.
 * @param[in] tolerance The allowed residual tolerence for the test waveforms
 * @param[in] RB The reduced basis set
 * @param[in] testmodels The set of waveform models to project onto the basis
 *
 * @return Returns \c XLAL_SUCCESS if all test waveforms meet the tolerence
 */
INT4 LALInferenceTestREAL8OrthonormalBasis(gsl_vector *delta,
                                           REAL8 tolerance,
                                           gsl_matrix *RB,
                                           gsl_matrix *testmodels){
  size_t dlength = testmodels->size2, nts = testmodels->size1;
  size_t j = 0, k = 0;

  /* normalise the test set */
  normalise_training_set(delta, testmodels);

  /* get projection errors for each test model */
  for ( k = 0; k < nts; k++ ){
    REAL8 projerr = 0.;
    gsl_vector *proj;
    gsl_vector_view testrow;
    XLAL_CALLGSL( proj = gsl_vector_calloc(dlength) ); /* allocated to zero */
    XLAL_CALLGSL( testrow = gsl_matrix_row(testmodels, k) );

    /* loop over reduced basis getting the projection coefficients */
    for ( j = 0; j < RB->size1; j++ ){
      REAL8 projcoeff;
      gsl_vector *RBrow;
      XLAL_CALLGSL( RBrow = gsl_vector_alloc( RB->size2 ) );
      XLAL_CALLGSL( gsl_matrix_get_row( RBrow, RB, j ) );

      /* get dot product of reduced basis vector with test model */
      projcoeff = weighted_dot_product(delta, RBrow, &testrow.vector);

      XLAL_CALLGSL( gsl_vector_scale( RBrow, projcoeff ) );
      XLAL_CALLGSL( gsl_vector_add( proj, RBrow ) );
      XLAL_CALLGSL( gsl_vector_free( RBrow ) );
    }

    /* get residual */
    XLAL_CALLGSL( gsl_vector_sub( proj, &testrow.vector ) );
    projerr = weighted_dot_product(delta, proj, proj);

    /* check projection error against tolerance */
    if ( fabs(projerr) > tolerance ) { return XLAL_FAILURE; }

    XLAL_CALLGSL( gsl_vector_free( proj ) );
  }

  return XLAL_SUCCESS;
}


/**
 * \brief Test the complex reduced basis against another set of waveforms
 *
 * This function projects a set of waveforms onto the reduced basis and
 * checks that the residuals are within a given tolerance
 *
 * @param[in] delta The time/frequency step(s) in the training set used to normalise the models.
 * This can be a vector containing just one value.
 * @param[in] tolerance The allowed residual tolerence for the test waveforms
 * @param[in] RB The reduced basis set
 * @param[in] testmodels The set of waveform models to project onto the basis
 *
 * @return Returns \c XLAL_SUCCESS if all test waveforms meet the tolerence
 */
INT4 LALInferenceTestCOMPLEX16OrthonormalBasis(gsl_vector *delta,
                                           REAL8 tolerance,
                                           gsl_matrix_complex *RB,
                                           gsl_matrix_complex *testmodels){
  size_t dlength = testmodels->size2, nts = testmodels->size1;
  size_t j = 0, k = 0;

  /* normalise the test set */
  complex_normalise_training_set(delta, testmodels);

  /* get projection errors for each test model */
  for ( k = 0; k < nts; k++ ){
    gsl_complex projerr;
    gsl_vector_complex *proj;
    gsl_vector_complex_view testrow;
    XLAL_CALLGSL( proj = gsl_vector_complex_calloc(dlength) ); /* allocated to zero */
    XLAL_CALLGSL( testrow = gsl_matrix_complex_row(testmodels, k) );

    /* loop over reduced basis getting the projection coefficients */
    for ( j = 0; j < RB->size1; j++ ){
      gsl_complex projcoeff;
      gsl_vector_complex *RBrow;
      XLAL_CALLGSL( RBrow = gsl_vector_complex_alloc( RB->size2 ) );
      XLAL_CALLGSL( gsl_matrix_complex_get_row( RBrow, RB, j ) );

      /* get dot product of reduced basis vector with test model */
      projcoeff = complex_weighted_dot_product(delta, RBrow, &testrow.vector);

      XLAL_CALLGSL( gsl_vector_complex_scale( RBrow, projcoeff ) );
      XLAL_CALLGSL( gsl_vector_complex_add( proj, RBrow ) );
      XLAL_CALLGSL( gsl_vector_complex_free( RBrow ) );
    }

    /* get residual */
    XLAL_CALLGSL( gsl_vector_complex_sub( proj, &testrow.vector ) );
    projerr = complex_weighted_dot_product(delta, proj, proj);

    /* check projection error against tolerance */
    if ( gsl_complex_abs(projerr) > tolerance ) { return XLAL_FAILURE; }

    XLAL_CALLGSL( gsl_vector_complex_free( proj ) );
  }

  return XLAL_SUCCESS;
}


/**
 * \brief Create a real empirical interpolant from a set of orthonormal basis functions
 *
 * Given a real \c gsl_matrix containing a set of orthonormal basis functions generate an
 * empirical intopolant, and set of interpolation points, using Algorithm 2 of
 * \cite FGHKT2014 .
 *
 * @param[in] RB The set of basis functions
 *
 * @return A \a LALInferenceREALROQInterpolant structure containing the interpolant and its nodes
 */
LALInferenceREALROQInterpolant *LALInferenceGenerateREALROQInterpolant(gsl_matrix *RB){
  size_t RBsize = RB->size1; /* reduced basis size (no. of reduced bases) */
  size_t dlength = RB->size2; /* length of each base */
  size_t i=1, j=0, k=0;
  REAL8 *V = XLALMalloc(sizeof(REAL8));
  gsl_matrix_view Vview;

  LALInferenceREALROQInterpolant *interp = XLALMalloc(sizeof(LALInferenceREALROQInterpolant));
  int idmax = 0, newidx = 0;

  /* get index of maximum absolute value of first basis */
  gsl_vector_view firstbasis = gsl_matrix_row(RB, 0);
  XLAL_CALLGSL( idmax = (int)gsl_blas_idamax(&firstbasis.vector) ); /* function gets index of maximum absolute value */

  interp->nodes = XLALMalloc(RBsize*sizeof(UINT4));
  interp->nodes[0] = idmax;

  for ( i=1; i<RBsize; i++ ){
    gsl_vector *interpolant, *subbasis;
    gsl_vector_view subview;

    Vview = gsl_matrix_view_array(V, i, i);

    for ( j=0; j<i; j++ ){
      for ( k=0; k<i; k++ ){
        XLAL_CALLGSL( gsl_matrix_set(&Vview.matrix, k, j, gsl_matrix_get(RB, j, interp->nodes[k])) );
      }
    }

    /* get B matrix */
    gsl_matrix *B = B_matrix(&Vview.matrix, RB);

    /* make empirical interpolant of basis */
    XLAL_CALLGSL( interpolant = gsl_vector_calloc(dlength) );
    XLAL_CALLGSL( subbasis = gsl_vector_calloc(i) );
    XLAL_CALLGSL( subview = gsl_matrix_row(RB, i) );

    for ( k=0; k<i; k++ ){
      XLAL_CALLGSL( gsl_vector_set(subbasis, k, gsl_vector_get(&subview.vector, interp->nodes[k])) );
    }

    XLAL_CALLGSL( gsl_blas_dgemv(CblasTrans, 1.0, B, subbasis, 0., interpolant) );

    /* get residuals of interpolant */
    XLAL_CALLGSL( gsl_vector_sub(interpolant, &subview.vector) );

    XLAL_CALLGSL( newidx = (int)gsl_blas_idamax(interpolant) );

    interp->nodes[i] = newidx;

    XLAL_CALLGSL( gsl_vector_free(subbasis) );
    XLAL_CALLGSL( gsl_matrix_free(B) );
    XLAL_CALLGSL( gsl_vector_free(interpolant) );

    /* reallocate memory for V */
    V = XLALRealloc(V, (i+1)*(i+1)*sizeof(REAL8));
  }

  /* get final B vector with all the indices */
  Vview = gsl_matrix_view_array(V, RBsize, RBsize);
  for( j=0; j<RBsize; j++ ){
    for( k=0; k<RBsize; k++ ){
      XLAL_CALLGSL( gsl_matrix_set(&Vview.matrix, k, j, gsl_matrix_get(RB, j, interp->nodes[k])) );
    }
  }

  /* allocate memory for intpolant array */
  interp->B = B_matrix(&Vview.matrix, RB);

  XLALFree(V);

  return interp;
}


/**
 * \brief Create a complex empirical interpolant from a set of orthonormal basis functions
 *
 * Given a complex \c gsl_matrix_complex containing a set of orthonormal basis functions generate an
 * empirical intopolant, and set of interpolation points, using Algorithm 2 of
 * \cite FGHKT2014 .
 *
 * @param[in] RB The set of basis functions
 *
 * @return A \a LALInferenceCOMPLEXROQInterpolant structure containing the interpolant and its nodes
 */
LALInferenceCOMPLEXROQInterpolant *LALInferenceGenerateCOMPLEXROQInterpolant(gsl_matrix_complex *RB){
  size_t RBsize = RB->size1; /* reduced basis size (no. of reduced bases) */
  size_t dlength = RB->size2; /* length of each base */
  size_t i=1, j=0, k=0;
  REAL8 *V = XLALMalloc(sizeof(COMPLEX16));
  gsl_matrix_complex_view Vview;

  gsl_complex scale1, scale0;
  GSL_SET_COMPLEX(&scale1, 1., 0.);
  GSL_SET_COMPLEX(&scale0, 0., 0.);

  LALInferenceCOMPLEXROQInterpolant *interp = XLALMalloc(sizeof(LALInferenceCOMPLEXROQInterpolant));
  int idmax = 0, newidx = 0;

  /* get index of maximum absolute value of first basis */
  gsl_vector_complex_view firstbasis = gsl_matrix_complex_row(RB, 0);
  idmax = complex_vector_maxabs_index(&firstbasis.vector);

  interp->nodes = XLALMalloc(RBsize*sizeof(UINT4));
  interp->nodes[0] = idmax;

  for ( i=1; i<RBsize; i++ ){
    gsl_vector_complex *interpolant, *subbasis;
    gsl_vector_complex_view subview;

    Vview = gsl_matrix_complex_view_array(V, i, i);

    for ( j=0; j<i; j++ ){
      for ( k=0; k<i; k++ ){
        XLAL_CALLGSL( gsl_matrix_complex_set(&Vview.matrix, k, j, gsl_matrix_complex_get(RB, j, interp->nodes[k])) );
      }
    }

    /* get B matrix */
    gsl_matrix_complex *B = complex_B_matrix(&Vview.matrix, RB);

    /* make empirical interpolant of basis */
    XLAL_CALLGSL( interpolant = gsl_vector_complex_calloc(dlength) );
    XLAL_CALLGSL( subbasis = gsl_vector_complex_calloc(i) );
    XLAL_CALLGSL( subview = gsl_matrix_complex_row(RB, i) );

    for ( k=0; k<i; k++ ){
      XLAL_CALLGSL( gsl_vector_complex_set(subbasis, k, gsl_vector_complex_get(&subview.vector, interp->nodes[k])) );
    }

    XLAL_CALLGSL( gsl_blas_zgemv(CblasTrans, scale1, B, subbasis, scale0, interpolant) );

    /* get residuals of interpolant */
    XLAL_CALLGSL( gsl_vector_complex_sub(interpolant, &subview.vector) );

    newidx = complex_vector_maxabs_index(interpolant);

    interp->nodes[i] = newidx;

    XLAL_CALLGSL( gsl_vector_complex_free(subbasis) );
    XLAL_CALLGSL( gsl_matrix_complex_free(B) );
    XLAL_CALLGSL( gsl_vector_complex_free(interpolant) );

    /* reallocate memory for V */
    V = XLALRealloc(V, (i+1)*(i+1)*sizeof(COMPLEX16));
  }

  /* get final B vector with all the indices */
  Vview = gsl_matrix_complex_view_array((double*)V, RBsize, RBsize);
  for( j=0; j<RBsize; j++ ){
    for( k=0; k<RBsize; k++ ){
      XLAL_CALLGSL( gsl_matrix_complex_set(&Vview.matrix, k, j, gsl_matrix_complex_get(RB, j, interp->nodes[k])) );
    }
  }

  /* allocate memory for intpolant array */
  interp->B = complex_B_matrix(&Vview.matrix, RB);

  XLALFree(V);

  return interp;
}


/** \brief Create the weights for the ROQ interpolant for the real data and model dot product
 *
 * @param[in] B The interpolant matrix
 * @param[in] data The real data vector
 * @param[in] vars A vector of data noise variance values (or a single value) to weight the "weights"
 *
 * @return The vector of weights
 */
gsl_vector *LALInferenceGenerateREAL8DataModelWeights(gsl_matrix *B, gsl_vector *data, gsl_vector *vars){
  gsl_vector *weights, *datacopy;

  XLAL_CHECK_NULL( vars->size == 1 || vars->size == B->size2, XLAL_EFUNC, "Vector of variance values is the wrong size" );

  XLAL_CALLGSL( datacopy = gsl_vector_alloc(B->size2) );
  XLAL_CALLGSL( gsl_vector_memcpy(datacopy, data) );

  if ( vars->size == 1 ){ XLAL_CALLGSL( gsl_vector_scale( datacopy, 1./gsl_vector_get(vars, 0) ) ); }
  else{ XLAL_CALLGSL( gsl_vector_div( datacopy, vars ) ); }

  /* create weights */
  XLAL_CALLGSL( weights = gsl_vector_alloc(B->size1) );
  XLAL_CALLGSL( gsl_blas_dgemv(CblasNoTrans, 1.0, B, datacopy, 0., weights) );

  XLAL_CALLGSL( gsl_vector_free( datacopy ) );

  return weights;
}


/** \brief Create the weights for the ROQ interpolant for the complex data and model dot product
 *
 * @param[in] B The interpolant matrix
 * @param[in] data The complex data vector
 * @param[in] vars A vector of data noise variance values (or a single value) to weight the "weights"
 *
 * @return The vector of weights
 */
gsl_vector_complex *LALInferenceGenerateCOMPLEX16DataModelWeights(gsl_matrix_complex *B, gsl_vector_complex *data, gsl_vector *vars){
  gsl_vector_complex *weights, *conjdata;
  gsl_complex scale1, scale0, cconj;
  size_t i = 0;

  XLAL_CHECK_NULL( vars->size == 1 || vars->size == B->size2, XLAL_EFUNC, "Vector of variance values is the wrong size" );

  XLAL_CALLGSL( conjdata = gsl_vector_complex_alloc(B->size2) );

  /* get conjugate of data and scale it */
  for ( i=0; i<conjdata->size; i++ ){
    XLAL_CALLGSL( cconj = gsl_complex_conjugate(gsl_vector_complex_get(data, i)) );
    if ( vars->size == 1 ) { XLAL_CALLGSL( cconj = gsl_complex_div_real( cconj, gsl_vector_get(vars, 0) ) ); }
    else { XLAL_CALLGSL( cconj = gsl_complex_div_real( cconj, gsl_vector_get(vars, i) ) ); }
    XLAL_CALLGSL( gsl_vector_complex_set(conjdata, i, cconj) );
  }

  /* create weights */
  XLAL_CALLGSL( weights = gsl_vector_complex_alloc(B->size1) );
  GSL_SET_COMPLEX(&scale1, 1., 0.);
  GSL_SET_COMPLEX(&scale0, 0., 0.);
  XLAL_CALLGSL( gsl_blas_zgemv(CblasNoTrans, scale1, B, conjdata, scale0, weights) );
  XLAL_CALLGSL( gsl_vector_complex_free(conjdata) );

  return weights;
}


/** \brief Create the weights for the ROQ interpolant for the real model-model dot product
 *
 * @param[in] B The real interpolant matrix
 * @param[in] vars A vector of data noise variance values (or a single value) to weight the "weights"
 *
 * @return The matrix of weights
 */
gsl_matrix *LALInferenceGenerateREALModelModelWeights(gsl_matrix *B, gsl_vector *vars){
  gsl_matrix *weights;

  XLAL_CALLGSL( weights = gsl_matrix_alloc(B->size1, B->size1) );
  size_t i=0, j=0, varsize = vars->size;
  double ressum = 0.;

  XLAL_CHECK_NULL( varsize == 1 || varsize == B->size2, XLAL_EFUNC, "Vector of variance values is the wrong size" );

  for ( i=0; i<B->size1; i++ ){
    for ( j=0; j<B->size1; j++ ){
      gsl_vector *Bi;
      gsl_vector_view Bj;

      XLAL_CALLGSL( Bi = gsl_vector_alloc( B->size2 ) );
      XLAL_CALLGSL( gsl_matrix_get_row( Bi, B, i ) );

      /* scale by the interpolant matrix noise variance (as this is the variance we just need to scale one of the vectors) */
      if ( varsize == 1 ){ XLAL_CALLGSL( gsl_vector_scale( Bi, 1./gsl_vector_get(vars, 0) ) ); }
      else{ XLAL_CALLGSL( gsl_vector_div(Bi, vars) ); }

      XLAL_CALLGSL( Bj = gsl_matrix_row(B, j) );
      XLAL_CALLGSL( gsl_blas_ddot(Bi, &Bj.vector, &ressum) );
      XLAL_CALLGSL( gsl_matrix_set(weights, i, j, ressum) );

      XLAL_CALLGSL( gsl_vector_free( Bi ) );
    }
  }

  return weights;
}


/** \brief Create the weights for the ROQ interpolant for the complex model-model dot product
 *
 * @param[in] B The real interpolant matrix
 * @param[in] vars A vector of data noise variance values (or a single value) to weight the "weights"
 *
 * @return The matrix of weights
 */
gsl_matrix_complex *LALInferenceGenerateCOMPLEXModelModelWeights(gsl_matrix_complex *B, gsl_vector *vars){
  gsl_matrix_complex *weights;

  XLAL_CALLGSL( weights = gsl_matrix_complex_alloc(B->size1, B->size1) );
  size_t i=0, j=0, k=0, varsize = vars->size;
  gsl_complex ressum, scale;

  XLAL_CHECK_NULL( varsize == 1 || varsize == B->size2, XLAL_EFUNC, "Vector of variance values is the wrong size" );

  for ( i=0; i<B->size1; i++ ){
    for ( j=0; j<B->size1; j++ ){
      gsl_vector_complex *Bi;
      gsl_vector_complex_view Bj;

      XLAL_CALLGSL( Bi = gsl_vector_complex_alloc( B->size2 ) );
      XLAL_CALLGSL( gsl_matrix_complex_get_row( Bi, B, i ) );

      /* scale by the interpolant matrix noise variance (as this is the variance we just need to scale one of the vectors) */
      if ( varsize == 1 ){
        GSL_SET_COMPLEX(&scale, 1./gsl_vector_get(vars, 0), 0.);
        XLAL_CALLGSL( gsl_vector_complex_scale( Bi, scale ) );
      }
      else{
        /* scale the vector element by element */
        for ( k=0; k<Bi->size; k++ ){
          XLAL_CALLGSL( scale = gsl_complex_div_real( gsl_vector_complex_get(Bi, k), gsl_vector_get(vars, k) ) );
          XLAL_CALLGSL( gsl_vector_complex_set(Bi, k, scale) );
        }
      }

      XLAL_CALLGSL( Bj = gsl_matrix_complex_row(B, j) );
      XLAL_CALLGSL( gsl_blas_zdotc(Bi, &Bj.vector, &ressum) );
      XLAL_CALLGSL( gsl_matrix_complex_set(weights, i, j, ressum) );

      XLAL_CALLGSL( gsl_vector_complex_free( Bi ) );
    }
  }

  return weights;
}


/** \brief Calculate the dot product of the real data and model using the ROQ iterpolant
 *
 * This function calculates the dot product of the real data and model using the ROQ
 * interpolant. This required the interpolant weights computed with \c LALInferenceCreateREAL8DataModelWeights
 * and the waveform model defined at the interolation node.
 *
 * @param[in] weights The ROQ interpolation weights
 * @param[in] model The waveform model defined at the interpolation points
 *
 * @return The dot product of the data and model
 */
 REAL8 LALInferenceROQREAL8DataDotModel(gsl_vector *weights, gsl_vector *model){
  REAL8 d_dot_m = 0.;
  XLAL_CALLGSL( gsl_blas_ddot(weights, model, &d_dot_m) );

  return d_dot_m;
}


/** \brief Calculate the dot product of the complex data and model using the ROQ iterpolant
 *
 * This function calculates the dot product of the real data and model using the ROQ
 * interpolant. This required the interpolant weights computed with \c LALInferenceCreateCOMPLEX16DataModelWeights
 * and the waveform model defined at the interolation node.
 *
 * @param[in] weights The ROQ interpolation weights
 * @param[in] model The waveform model defined at the interpolation points
 *
 * @return The dot product of the data and model
 */
 COMPLEX16 LALInferenceROQCOMPLEX16DataDotModel(gsl_vector_complex *weights, gsl_vector_complex *model){
  gsl_complex d_dot_m;
  XLAL_CALLGSL( gsl_blas_zdotu(weights, model, &d_dot_m) );

  return GSL_REAL(d_dot_m) + I*GSL_IMAG(d_dot_m);
}


/** \brief Calculate the dot product of the real model with itself using the ROQ iterpolant
 *
 * This function calculates the dot product of the real model with itself using the ROQ
 * interpolant. This required the interpolant weights computed with \c LALInferenceCreateREALModelModelWeights
 * and the model defined at the interpolation nodes.
 *
 * @param[in] weights The ROQ interpolation weights
 * @param[in] model The waveform model defined at the interpolation points
 *
 * @return The dot product of the model with itself
 */
REAL8 LALInferenceROQREAL8ModelDotModel(gsl_matrix *weights, gsl_vector *model){
  gsl_vector *ws;

  XLAL_CALLGSL( ws = gsl_vector_alloc(weights->size1) );
  REAL8 m_dot_m = 0.;

  XLAL_CALLGSL( gsl_blas_dgemv(CblasTrans, 1.0, weights, model, 0., ws) );
  XLAL_CALLGSL( gsl_blas_ddot(ws, model, &m_dot_m) );

  return m_dot_m;
}


/** \brief Calculate the dot product of the complex model with itself using the ROQ iterpolant
 *
 * This function calculates the dot product of the complex model with itself using the ROQ
 * interpolant. This required the interpolant weights computed with \c LALInferenceCreateCOMPLEXModelModelWeights
 * and the model defined at the interpolation nodes.
 *
 * @param[in] weights The ROQ interpolation weights
 * @param[in] model The waveform model defined at the interpolation points
 *
 * @return The dot product of the model with itself
 */
COMPLEX16 LALInferenceROQCOMPLEX16ModelDotModel(gsl_matrix_complex *weights, gsl_vector_complex *model){
  gsl_vector_complex *ws;

  XLAL_CALLGSL( ws = gsl_vector_complex_alloc(weights->size1) );
  gsl_complex m_dot_m, scale1, scale0;
  GSL_SET_COMPLEX(&scale1, 1., 0.);
  GSL_SET_COMPLEX(&scale0, 0., 0.);

  XLAL_CALLGSL( gsl_blas_zgemv(CblasNoTrans, scale1, weights, model, scale0, ws) );
  XLAL_CALLGSL( gsl_blas_zdotc(model, ws, &m_dot_m) );

  return GSL_REAL(m_dot_m) + I*GSL_IMAG(m_dot_m);
}


/** \brief Free memory for a \c LALInferenceREALROQInterpolant
 *
 * @param[in] a A pointer to a  \c LALInferenceREALROQInterpolant
 */
void LALInferenceRemoveREALROQInterpolant( LALInferenceREALROQInterpolant *a ){
  if ( a == NULL ) { return; }

  if ( a->B != NULL ){ XLAL_CALLGSL( gsl_matrix_free( a->B ) ); }
  if ( a->nodes != NULL ){ XLALFree( a->nodes ); }
  XLALFree( a );
  a = NULL;
}


/** \brief Free memory for a \c LALInferenceCOMPLEXROQInterpolant
 *
 * @param[in] a A pointer to a  \c LALInferenceCOMPLEXROQInterpolant
 */
void LALInferenceRemoveCOMPLEXROQInterpolant( LALInferenceCOMPLEXROQInterpolant *a ){
  if ( a == NULL ) { return; }

  if ( a->B != NULL ){ XLAL_CALLGSL( gsl_matrix_complex_free( a->B ) ); }
  if ( a->nodes != NULL ){ XLALFree( a->nodes ); }
  XLALFree( a );
  a = NULL;
}

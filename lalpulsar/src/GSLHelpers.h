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

#ifndef _GSLHELPERS_H
#define _GSLHELPERS_H
/// \cond DONT_DOXYGEN

#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <lal/XLALError.h>
#include <lal/XLALGSL.h>

#define ALLOC_GSL_VAL(val, name, call) \
  name = (call); \
  XLAL_CHECK_VAL(val, (name) != NULL, XLAL_ENOMEM, #call " failed")

#define ALLOC_GSL_1D_VAL(val, type, name, n) \
  name = gsl_##type##_calloc(n); \
  XLAL_CHECK_VAL(val, (name) != NULL, XLAL_ENOMEM, "gsl_"#type"_calloc(%zu) failed", ((size_t) n))

#define ALLOC_GSL_2D_VAL(val, type, name, m, n) \
  name = gsl_##type##_calloc(m, n); \
  XLAL_CHECK_VAL(val, (name) != NULL, XLAL_ENOMEM, "gsl_"#type"_calloc(%zu,%zu) failed", ((size_t) m), ((size_t) n))

#define CLONE_GSL_1D_VAL(val, type, dest, src) \
  if ((src) != NULL) { \
    ALLOC_GSL_1D_VAL(val, type, dest, (src)->size); \
    gsl_##type##_memcpy(dest, src); \
  } else { \
    dest = NULL; \
  }

#define CLONE_GSL_2D_VAL(val, type, dest, src) \
  if ((src) != NULL) { \
    ALLOC_GSL_2D_VAL(val, type, dest, (src)->size1, (src)->size2); \
    gsl_##type##_memcpy(dest, src); \
  } else { \
    dest = NULL; \
  }

#define PRINT_GSL_1D(type, name, fmt) \
  do { \
    fprintf(stderr, "%s:%i ", strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__, __LINE__); \
    fprintf(stderr, "%s = [", #name); \
    for (size_t GH_i = 0; (name) != NULL && GH_i < (name)->size; ++GH_i) { \
      fprintf(stderr, " "fmt, gsl_##type##_get(name, GH_i)); \
    } \
    fprintf(stderr, " ]\n"); \
  } while (0)

#define PRINT_GSL_2D(type, name, fmt) \
  do { \
    fprintf(stderr, "%s:%i ", strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__, __LINE__); \
    fprintf(stderr, "%s = [\n", #name); \
    for (size_t GH_i = 0; (name) != NULL && GH_i < (name)->size1; ++GH_i) { \
      fprintf(stderr, "  "); \
      for (size_t GH_j = 0; GH_j < (name)->size2; ++GH_j) { \
        fprintf(stderr, " "fmt, gsl_##type##_get(name, GH_i, GH_j)); \
      } \
      fprintf(stderr, ";\n"); \
    } \
    fprintf(stderr, "]\n"); \
  } while (0)

#define FREE_GSL(type, ...) \
  do { \
    gsl_##type *GH_ptrs[] = {__VA_ARGS__}; \
    for (size_t GH_i = 0; GH_i < XLAL_NUM_ELEM(GH_ptrs); ++GH_i) { \
      gsl_##type##_free(GH_ptrs[GH_i]); \
    } \
  } while (0)

#define CALL_GSL_VAL(val, ...)		CALL_GSL_VAL_(val, __VA_ARGS__, NULL, NULL)
#define CALL_GSL_VAL_(val, call, fmt, ...) \
  do { \
    int GH_retn = 0; \
    XLAL_CALLGSL(GH_retn = (call)); \
    if (GH_retn != 0) { \
      if (fmt != NULL) { \
        XLAL_ERROR_VAL(val, XLAL_EFAILED, fmt, __VA_ARGS__, NULL); \
      } else { \
        XLAL_ERROR_VAL(val, XLAL_EFAILED, #call " failed: %s", gsl_strerror(GH_retn)); \
      } \
    } \
  } while (0)

#define GALLOC(name, call)		ALLOC_GSL_VAL(XLAL_FAILURE, name, call)
#define GALLOC_NULL(type, name, n)	ALLOC_GSL_VAL(NULL, name, call)
#define GALLOC_MAIN(type, name, n)	ALLOC_GSL_VAL(EXIT_FAILURE, name, call)
#define GALLOC_REAL8(type, name, n)	ALLOC_GSL_VAL(XLAL_REAL8_FAIL_NAN, name, call)
#define GALLOC_REAL4(type, name, n)	ALLOC_GSL_VAL(XLAL_REAL4_FAIL_NAN, name, call)

#define GCALL(...)			CALL_GSL_VAL(XLAL_FAILURE, __VA_ARGS__)
#define GCALL_NULL(...)			CALL_GSL_VAL(NULL, __VA_ARGS__)
#define GCALL_MAIN(...)			CALL_GSL_VAL(EXIT_FAILURE, __VA_ARGS__)
#define GCALL_REAL8(...)		CALL_GSL_VAL(XLAL_REAL8_FAIL_NAN, __VA_ARGS__)
#define GCALL_REAL4(...)		CALL_GSL_VAL(XLAL_REAL4_FAIL_NAN, __VA_ARGS__)

#define GAPERM(name, n)			ALLOC_GSL_1D_VAL(XLAL_FAILURE, permutation, name, n)
#define GAPERM_NULL(name, n)		ALLOC_GSL_1D_VAL(NULL, permutation, name, n)
#define GAPERM_MAIN(name, n)		ALLOC_GSL_1D_VAL(EXIT_FAILURE, permutation, name, n)
#define GAPERM_REAL8(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, permutation, name, n)
#define GAPERM_REAL4(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, permutation, name, n)
#define GCPERM(dest, src)		CLONE_GSL_1D_VAL(XLAL_FAILURE, permutation, dest, src)
#define GCPERM_NULL(dest, src)		CLONE_GSL_1D_VAL(NULL, permutation, dest, src)
#define GCPERM_MAIN(dest, src)		CLONE_GSL_1D_VAL(EXIT_FAILURE, permutation, dest, src)
#define GCPERM_REAL8(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, permutation, dest, src)
#define GCPERM_REAL4(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, permutation, dest, src)
#define GPPERM(name, fmt)		PRINT_GSL_1D(permutation, name, fmt)
#define GFPERM(...)			FREE_GSL(permutation, __VA_ARGS__)

#define GAVEC(name, n)			ALLOC_GSL_1D_VAL(XLAL_FAILURE, vector, name, n)
#define GAVEC_NULL(name, n)		ALLOC_GSL_1D_VAL(NULL, vector, name, n)
#define GAVEC_MAIN(name, n)		ALLOC_GSL_1D_VAL(EXIT_FAILURE, vector, name, n)
#define GAVEC_REAL8(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector, name, n)
#define GAVEC_REAL4(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector, name, n)
#define GCVEC(dest, src)		CLONE_GSL_1D_VAL(XLAL_FAILURE, vector, dest, src)
#define GCVEC_NULL(dest, src)		CLONE_GSL_1D_VAL(NULL, vector, dest, src)
#define GCVEC_MAIN(dest, src)		CLONE_GSL_1D_VAL(EXIT_FAILURE, vector, dest, src)
#define GCVEC_REAL8(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector, dest, src)
#define GCVEC_REAL4(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector, dest, src)
#define GPVEC(name, fmt)		PRINT_GSL_1D(vector, name, fmt)
#define GFVEC(...)			FREE_GSL(vector, __VA_ARGS__)

#define GAVECI(name, n)			ALLOC_GSL_1D_VAL(XLAL_FAILURE, vector_int, name, n)
#define GAVECI_NULL(name, n)		ALLOC_GSL_1D_VAL(NULL, vector_int, name, n)
#define GAVECI_MAIN(name, n)		ALLOC_GSL_1D_VAL(EXIT_FAILURE, vector_int, name, n)
#define GAVECI_REAL8(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_int, name, n)
#define GAVECI_REAL4(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_int, name, n)
#define GCVECI(dest, src)		CLONE_GSL_1D_VAL(XLAL_FAILURE, vector_int, dest, src)
#define GCVECI_NULL(dest, src)		CLONE_GSL_1D_VAL(NULL, vector_int, dest, src)
#define GCVECI_MAIN(dest, src)		CLONE_GSL_1D_VAL(EXIT_FAILURE, vector_int, dest, src)
#define GCVECI_REAL8(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_int, dest, src)
#define GCVECI_REAL4(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_int, dest, src)
#define GPVECI(name, fmt)		PRINT_GSL_1D(vector_int, name, fmt)
#define GFVECI(...)			FREE_GSL(vector_int, __VA_ARGS__)

#define GAVECU(name, n)			ALLOC_GSL_1D_VAL(XLAL_FAILURE, vector_uint, name, n)
#define GAVECU_NULL(name, n)		ALLOC_GSL_1D_VAL(NULL, vector_uint, name, n)
#define GAVECU_MAIN(name, n)		ALLOC_GSL_1D_VAL(EXIT_FAILURE, vector_uint, name, n)
#define GAVECU_REAL8(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_uint, name, n)
#define GAVECU_REAL4(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_uint, name, n)
#define GCVECU(dest, src)		CLONE_GSL_1D_VAL(XLAL_FAILURE, vector_uint, dest, src)
#define GCVECU_NULL(dest, src)		CLONE_GSL_1D_VAL(NULL, vector_uint, dest, src)
#define GCVECU_MAIN(dest, src)		CLONE_GSL_1D_VAL(EXIT_FAILURE, vector_uint, dest, src)
#define GCVECU_REAL8(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_uint, dest, src)
#define GCVECU_REAL4(dest, src)		CLONE_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_uint, dest, src)
#define GPVECU(name, fmt)		PRINT_GSL_1D(vector_uint, name, fmt)
#define GFVECU(...)			FREE_GSL(vector_uint, __VA_ARGS__)

#define GAVECLI(name, n)		ALLOC_GSL_1D_VAL(XLAL_FAILURE, vector_long, name, n)
#define GAVECLI_NULL(name, n)		ALLOC_GSL_1D_VAL(NULL, vector_long, name, n)
#define GAVECLI_MAIN(name, n)		ALLOC_GSL_1D_VAL(EXIT_FAILURE, vector_long, name, n)
#define GAVECLI_REAL8(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_long, name, n)
#define GAVECLI_REAL4(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_long, name, n)
#define GCVECLI(dest, src)		CLONE_GSL_1D_VAL(XLAL_FAILURE, vector_long, dest, src)
#define GCVECLI_NULL(dest, src)		CLONE_GSL_1D_VAL(NULL, vector_long, dest, src)
#define GCVECLI_MAIN(dest, src)		CLONE_GSL_1D_VAL(EXIT_FAILURE, vector_long, dest, src)
#define GCVECLI_REAL8(dest, src)	CLONE_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_long, dest, src)
#define GCVECLI_REAL4(dest, src)	CLONE_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_long, dest, src)
#define GPVECLI(name, fmt)		PRINT_GSL_1D(vector_long, name, fmt)
#define GFVECLI(...)			FREE_GSL(vector_long, __VA_ARGS__)

#define GAVECLU(name, n)		ALLOC_GSL_1D_VAL(XLAL_FAILURE, vector_ulong, name, n)
#define GAVECLU_NULL(name, n)		ALLOC_GSL_1D_VAL(NULL, vector_ulong, name, n)
#define GAVECLU_MAIN(name, n)		ALLOC_GSL_1D_VAL(EXIT_FAILURE, vector_ulong, name, n)
#define GAVECLU_REAL8(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_ulong, name, n)
#define GAVECLU_REAL4(name, n)		ALLOC_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_ulong, name, n)
#define GCVECLU(dest, src)		CLONE_GSL_1D_VAL(XLAL_FAILURE, vector_ulong, dest, src)
#define GCVECLU_NULL(dest, src)		CLONE_GSL_1D_VAL(NULL, vector_ulong, dest, src)
#define GCVECLU_MAIN(dest, src)		CLONE_GSL_1D_VAL(EXIT_FAILURE, vector_ulong, dest, src)
#define GCVECLU_REAL8(dest, src)	CLONE_GSL_1D_VAL(XLAL_REAL8_FAIL_NAN, vector_ulong, dest, src)
#define GCVECLU_REAL4(dest, src)	CLONE_GSL_1D_VAL(XLAL_REAL4_FAIL_NAN, vector_ulong, dest, src)
#define GPVECLU(name, fmt)		PRINT_GSL_1D(vector_ulong, name, fmt)
#define GFVECLU(...)			FREE_GSL(vector_ulong, __VA_ARGS__)

#define GAMAT(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_FAILURE, matrix, name, m, n)
#define GAMAT_NULL(name, m, n)		ALLOC_GSL_2D_VAL(NULL, matrix, name, m, n)
#define GAMAT_MAIN(name, m, n)		ALLOC_GSL_2D_VAL(EXIT_FAILURE, matrix, name, m, n)
#define GAMAT_REAL8(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix, name, m, n)
#define GAMAT_REAL4(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix, name, m, n)
#define GCMAT(dest, src)		CLONE_GSL_2D_VAL(XLAL_FAILURE, matrix, dest, src)
#define GCMAT_NULL(dest, src)		CLONE_GSL_2D_VAL(NULL, matrix, dest, src)
#define GCMAT_MAIN(dest, src)		CLONE_GSL_2D_VAL(EXIT_FAILURE, matrix, dest, src)
#define GCMAT_REAL8(dest, src)		CLONE_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix, dest, src)
#define GCMAT_REAL4(dest, src)		CLONE_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix, dest, src)
#define GPMAT(name, fmt)		PRINT_GSL_2D(matrix, name, fmt)
#define GFMAT(...)			FREE_GSL(matrix, __VA_ARGS__)

#define GAMATI(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_FAILURE, matrix_int, name, m, n)
#define GAMATI_NULL(name, m, n)		ALLOC_GSL_2D_VAL(NULL, matrix_int, name, m, n)
#define GAMATI_MAIN(name, m, n)		ALLOC_GSL_2D_VAL(EXIT_FAILURE, matrix_int, name, m, n)
#define GAMATI_REAL8(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_int, name, m, n)
#define GAMATI_REAL4(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_int, name, m, n)
#define GCMATI(dest, src)		CLONE_GSL_2D_VAL(XLAL_FAILURE, matrix_int, dest, src)
#define GCMATI_NULL(dest, src)		CLONE_GSL_2D_VAL(NULL, matrix_int, dest, src)
#define GCMATI_MAIN(dest, src)		CLONE_GSL_2D_VAL(EXIT_FAILURE, matrix_int, dest, src)
#define GCMATI_REAL8(dest, src)		CLONE_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_int, dest, src)
#define GCMATI_REAL4(dest, src)		CLONE_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_int, dest, src)
#define GPMATI(name, fmt)		PRINT_GSL_2D(matrix_int, name, fmt)
#define GFMATI(...)			FREE_GSL(matrix_int, __VA_ARGS__)

#define GAMATU(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_FAILURE, matrix_uint, name, m, n)
#define GAMATU_NULL(name, m, n)		ALLOC_GSL_2D_VAL(NULL, matrix_uint, name, m, n)
#define GAMATU_MAIN(name, m, n)		ALLOC_GSL_2D_VAL(EXIT_FAILURE, matrix_uint, name, m, n)
#define GAMATU_REAL8(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_uint, name, m, n)
#define GAMATU_REAL4(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_uint, name, m, n)
#define GCMATU(dest, src)		CLONE_GSL_2D_VAL(XLAL_FAILURE, matrix_uint, dest, src)
#define GCMATU_NULL(dest, src)		CLONE_GSL_2D_VAL(NULL, matrix_uint, dest, src)
#define GCMATU_MAIN(dest, src)		CLONE_GSL_2D_VAL(EXIT_FAILURE, matrix_uint, dest, src)
#define GCMATU_REAL8(dest, src)		CLONE_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_uint, dest, src)
#define GCMATU_REAL4(dest, src)		CLONE_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_uint, dest, src)
#define GPMATU(name, fmt)		PRINT_GSL_2D(matrix_uint, name, fmt)
#define GFMATU(...)			FREE_GSL(matrix_uint, __VA_ARGS__)

#define GAMATLI(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_FAILURE, matrix_long, name, m, n)
#define GAMATLI_NULL(name, m, n)	ALLOC_GSL_2D_VAL(NULL, matrix_long, name, m, n)
#define GAMATLI_MAIN(name, m, n)	ALLOC_GSL_2D_VAL(EXIT_FAILURE, matrix_long, name, m, n)
#define GAMATLI_REAL8(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_long, name, m, n)
#define GAMATLI_REAL4(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_long, name, m, n)
#define GCMATLI(dest, src)		CLONE_GSL_2D_VAL(XLAL_FAILURE, matrix_long, dest, src)
#define GCMATLI_NULL(dest, src)		CLONE_GSL_2D_VAL(NULL, matrix_long, dest, src)
#define GCMATLI_MAIN(dest, src)		CLONE_GSL_2D_VAL(EXIT_FAILURE, matrix_long, dest, src)
#define GCMATLI_REAL8(dest, src)	CLONE_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_long, dest, src)
#define GCMATLI_REAL4(dest, src)	CLONE_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_long, dest, src)
#define GPMATLI(name, fmt)		PRINT_GSL_2D(matrix_long, name, fmt)
#define GFMATLI(...)			FREE_GSL(matrix_long, __VA_ARGS__)

#define GAMATLU(name, m, n)		ALLOC_GSL_2D_VAL(XLAL_FAILURE, matrix_ulong, name, m, n)
#define GAMATLU_NULL(name, m, n)	ALLOC_GSL_2D_VAL(NULL, matrix_ulong, name, m, n)
#define GAMATLU_MAIN(name, m, n)	ALLOC_GSL_2D_VAL(EXIT_FAILURE, matrix_ulong, name, m, n)
#define GAMATLU_REAL8(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_ulong, name, m, n)
#define GAMATLU_REAL4(name, m, n)	ALLOC_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_ulong, name, m, n)
#define GCMATLU(dest, src)		CLONE_GSL_2D_VAL(XLAL_FAILURE, matrix_ulong, dest, src)
#define GCMATLU_NULL(dest, src)		CLONE_GSL_2D_VAL(NULL, matrix_ulong, dest, src)
#define GCMATLU_MAIN(dest, src)		CLONE_GSL_2D_VAL(EXIT_FAILURE, matrix_ulong, dest, src)
#define GCMATLU_REAL8(dest, src)	CLONE_GSL_2D_VAL(XLAL_REAL8_FAIL_NAN, matrix_ulong, dest, src)
#define GCMATLU_REAL4(dest, src)	CLONE_GSL_2D_VAL(XLAL_REAL4_FAIL_NAN, matrix_ulong, dest, src)
#define GPMATLU(name, fmt)		PRINT_GSL_2D(matrix_ulong, name, fmt)
#define GFMATLU(...)			FREE_GSL(matrix_ulong, __VA_ARGS__)

/// \endcond
#endif // _GSLHELPERS_H

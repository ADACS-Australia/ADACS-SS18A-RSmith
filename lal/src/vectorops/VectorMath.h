/*
 * Copyright (C) 2015 Reinhard Prix, Karl Wette
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with with program; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA  02111-1307  USA
 *
 */

#ifndef _VECTORMATH_H
#define _VECTORMATH_H

#include <lal/LALStdlib.h>

#ifdef  __cplusplus
extern "C" {
#endif

/**
 * \defgroup VectorMath_h Header VectorMath.h
 * \ingroup lal_vectorops
 * \author Reinhard Prix, Karl Wette
 *
 * \brief Functions for performing fast math on vectors of numbers, using SIMD (SSE, AVX, ...) instructions if available.
 *
 * ### Synopsis ###
 *
 * \code
 * #include <lal/VectorMath.h>
 * \endcode
 *
 * ### Alignment ###
 *
 * Neither input nor output vectors are \b required to have any particular memory alignment. Nevertheless, performance
 * \e may be improved if vectors are 16-byte aligned for SSE, and 32-byte aligned for AVX.
 */
/** @{ */

/** \name Failsafe Aligned Memory Handling */
/** @{ */

/** A special #UINT4Vector with n-byte aligned memory \c data array */
typedef struct tagUINT4VectorAligned {
  UINT4 length;		/**< number of 'usable' array entries (fully aligned) */
  UINT4 *data;		/**< start of aligned memory block */
  UINT4 *data0;		/**< actual physical start of memory block, possibly not aligned */
} UINT4VectorAligned;

/** Create a new #UINT4VectorAligned struct with length \c length and alignment \c align */
UINT4VectorAligned *XLALCreateUINT4VectorAligned ( const UINT4 length, const UINT4 align );

/** Resize an existing #UINT4VectorAligned struct to length \c length and alignment \c align */
UINT4VectorAligned *XLALResizeUINT4VectorAligned ( UINT4VectorAligned *in, const UINT4 length, const UINT4 align );

/** Free a #UINT4VectorAligned struct */
void XLALDestroyUINT4VectorAligned ( UINT4VectorAligned *in );

/** A special #REAL4Vector with n-byte aligned memory \c data array */
typedef struct tagREAL4VectorAligned {
  UINT4 length;		/**< number of 'usable' array entries (fully aligned) */
  REAL4 *data;		/**< start of aligned memory block */
  REAL4 *data0;		/**< actual physical start of memory block, possibly not aligned */
} REAL4VectorAligned;

/** Create a new #REAL4VectorAligned struct with length \c length and alignment \c align */
REAL4VectorAligned *XLALCreateREAL4VectorAligned ( const UINT4 length, const UINT4 align );

/** Resize an existing #REAL4VectorAligned struct to length \c length and alignment \c align */
REAL4VectorAligned *XLALResizeREAL4VectorAligned ( REAL4VectorAligned *in, const UINT4 length, const UINT4 align );

/** Free a #REAL4VectorAligned struct */
void XLALDestroyREAL4VectorAligned ( REAL4VectorAligned *in );

/** A special #REAL8Vector with n-byte aligned memory \c data array */
typedef struct tagREAL8VectorAligned {
  UINT4 length;		/**< number of 'usable' array entries (fully aligned) */
  REAL8 *data;		/**< start of aligned memory block */
  REAL8 *data0;		/**< actual physical start of memory block, possibly not aligned */
} REAL8VectorAligned;

/** Create a new #REAL8VectorAligned struct with length \c length and alignment \c align */
REAL8VectorAligned *XLALCreateREAL8VectorAligned ( const UINT4 length, const UINT4 align );

/** Resize an existing #REAL8VectorAligned struct to length \c length and alignment \c align */
REAL8VectorAligned *XLALResizeREAL8VectorAligned ( REAL8VectorAligned *in, const UINT4 length, const UINT4 align );

/** Free a #REAL8VectorAligned struct */
void XLALDestroyREAL8VectorAligned ( REAL8VectorAligned *in );

/** A special #COMPLEX8Vector with n-byte aligned memory \c data array */
typedef struct tagCOMPLEX8VectorAligned {
  UINT4 length;		/**< number of 'usable' array entries (fully aligned) */
  COMPLEX8 *data;	/**< start of aligned memory block */
  COMPLEX8 *data0;	/**< actual physical start of memory block, possibly not aligned */
} COMPLEX8VectorAligned;

/** Create a new #COMPLEX8VectorAligned struct with length \c length and alignment \c align */
COMPLEX8VectorAligned *XLALCreateCOMPLEX8VectorAligned ( const UINT4 length, const UINT4 align );

/** Resize an existing #COMPLEX8VectorAligned struct to length \c length and alignment \c align */
COMPLEX8VectorAligned *XLALResizeCOMPLEX8VectorAligned ( COMPLEX8VectorAligned *in, const UINT4 length, const UINT4 align );

/** Free a #COMPLEX8VectorAligned struct */
void XLALDestroyCOMPLEX8VectorAligned ( COMPLEX8VectorAligned *in );

/** A special #COMPLEX16Vector with n-byte aligned memory \c data array */
typedef struct tagCOMPLEX16VectorAligned {
  UINT4 length;		/**< number of 'usable' array entries (fully aligned) */
  COMPLEX16 *data;	/**< start of aligned memory block */
  COMPLEX16 *data0;	/**< actual physical start of memory block, possibly not aligned */
} COMPLEX16VectorAligned;

/** Create a new #COMPLEX16VectorAligned struct with length \c length and alignment \c align */
COMPLEX16VectorAligned *XLALCreateCOMPLEX16VectorAligned ( const UINT4 length, const UINT4 align );

/** Resize an existing #COMPLEX16VectorAligned struct to length \c length and alignment \c align */
COMPLEX16VectorAligned *XLALResizeCOMPLEX16VectorAligned ( COMPLEX16VectorAligned *in, const UINT4 length, const UINT4 align );

/** Free a #COMPLEX16VectorAligned struct */
void XLALDestroyCOMPLEX16VectorAligned ( COMPLEX16VectorAligned *in );

/** @} */

/** \name Vector Math Functions */
/** @{ */

/** Compute \f$\text{out} = \sin(\text{in})\f$ over REAL4 vectors \c out, \c in with \c len elements */
int XLALVectorSinREAL4 ( REAL4 *out, const REAL4 *in, const UINT4 len );

/** Compute \f$\text{out} = \cos(\text{in})\f$ over REAL4 vectors \c out, \c in with \c len elements */
int XLALVectorCosREAL4 ( REAL4 *out, const REAL4 *in, const UINT4 len );

/** Compute \f$\text{out} = \exp(\text{in})\f$ over REAL4 vectors \c out, \c in with \c len elements */
int XLALVectorExpREAL4 ( REAL4 *out, const REAL4 *in, const UINT4 len );

/** Compute \f$\text{out} = \log(\text{in})\f$ over REAL4 vectors \c out, \c in with \c len elements */
int XLALVectorLogREAL4 ( REAL4 *out, const REAL4 *in, const UINT4 len );

/** Compute \f$\text{out1} = \sin(\text{in}), \text{out2} = \cos(\text{in})\f$ over REAL4 vectors \c out1, \c out2, \c in with \c len elements */
int XLALVectorSinCosREAL4 ( REAL4 *out1, REAL4 *out2, const REAL4 *in, const UINT4 len );

/** Compute \f$\text{out1} = \sin(2\pi \text{in}), \text{out2} = \cos(2\pi \text{in})\f$ over REAL4 vectors \c out1, \c out2, \c in with \c len elements */
int XLALVectorSinCos2PiREAL4 ( REAL4 *out1, REAL4 *out2, const REAL4 *in, const UINT4 len );

/** @} */

/** \name Vector by Vector Operations */
/** @{ */

/** Compute \f$\text{out} = \text{in1} + \text{in2}\f$ over REAL4 vectors \c in1 and \c in2 with \c len elements */
int XLALVectorAddREAL4 ( REAL4 *out, const REAL4 *in1, const REAL4 *in2, const UINT4 len);

/** Compute \f$\text{out} = \text{in1} \times \text{in2}\f$ over REAL4 vectors \c in1 and \c in2 with \c len elements */
int XLALVectorMultiplyREAL4 ( REAL4 *out, const REAL4 *in1, const REAL4 *in2, const UINT4 len);

/** Compute \f$\text{out} = max ( \text{in1}, \text{in2} )\f$ over REAL4 vectors \c in1 and \c in2 with \c len elements */
int XLALVectorMaxREAL4 ( REAL4 *out, const REAL4 *in1, const REAL4 *in2, const UINT4 len);

/** @} */

/** \name Vector by Scalar Operations */
/** @{ */

/** Compute \f$\text{out} = \text{scalar} + \text{in}\f$ over REAL4 vector \c in with \c len elements */
int XLALVectorShiftREAL4 ( REAL4 *out, REAL4 scalar, const REAL4 *in, const UINT4 len);

/** Compute \f$\text{out} = \text{scalar} \times \text{in}\f$ over REAL4 vector \c in with \c len elements */
int XLALVectorScaleREAL4 ( REAL4 *out, REAL4 scalar, const REAL4 *in, const UINT4 len);

/** Compute \f$\text{out} = \text{scalar} \times \text{in}\f$ over REAL8 vector \c in with \c len elements */
int XLALVectorScaleREAL8 ( REAL8 *out, REAL8 scalar, const REAL8 *in, const UINT4 len);

/** @} */

/** \name Vector Element Finding Operations */
/** @{ */

/** Count and return indexes (in \c count and \c out respectively) of vector elements where \f$\text{in1} \le \text{in2}\f$ */
int XLALVectorFindVectorLessEqualREAL4( UINT4* count, UINT4 *out, const REAL4 *in1, const REAL4 *in2, const UINT4 len );

/** Count and return indexes (in \c count and \c out respectively) of vector elements where \f$\text{scalar} \le \text{in2}\f$ */
int XLALVectorFindScalarLessEqualREAL4( UINT4* count, UINT4 *out, REAL4 scalar, const REAL4 *in, const UINT4 len );

/** @} */

/** @} */

#ifdef  __cplusplus
}
#endif

#endif /* _VECTORMATH_H */

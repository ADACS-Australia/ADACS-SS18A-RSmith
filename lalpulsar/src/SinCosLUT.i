//
// Copyright (C) 2009, 2010, 2011, 2012, 2013 Bernd Machenschalk, Reinhard Prix
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with with program; see the file COPYING. If not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA  02111-1307  USA
//

/*
  This file defines:
  - a macro SINCOS_TRIM_X(y,x) which trims the value x to interval [0..2)
  - global REAL4 arrays sincosLUTbase[] and sincosLUTdiff[] as lookup tables
  - a function void local_sin_cos_2PI_LUT_init(void) that inits these
  - a function local_sin_cos_2PI_LUT_trimmed(*sin,*cos,x) that uses the
    lookup tables to evaluate sin and cos values of 2*Pi*x if x is
    already trimmed to the interval [0..2)
  - macros SINCOS_STEP1..6 for the individual steps of
    local_sin_cos_2PI_LUT_trimmed() (to be mixed into the hotloop code)
  - a type ux_t for a variable ux to be used in these macros
  - macros SINCOS_LUT_RES, SINCOS_ADDS, SINCOS_MASK1, SINCOS_MASK2, SINCOS_SHIFT

  The following macros determine the code that is actually built:
  _MSC_VER       : are we using the Microsoft compiler that doesn't know C99?
  _ARCH_PPC      : are we compiling for PowerPC?
  LAL_NDEBUG     : are we compiling in LAL debug mode?
  __BIG_ENDIAN__ : has the architecture big-endian byt order?
*/

#include <lal/LALConstants.h>
#include <lal/XLALError.h>

/*
  Trimming macro
*/

/* the way of trimming x to the interval [0..2) for the sin_cos_LUT functions
   give significant differences in speed, so we provide various ways here. */
#ifdef SINCOS_REAL4_ARG

#ifdef _MSC_VER /* no C99 rint() */
#define SINCOS_TRIM_X(y,x) \
  { \
    __asm FLD     QWORD PTR x   \
    __asm FRNDINT               \
    __asm FSUBR   QWORD PTR x   \
    __asm FLD1                  \
    __asm FADDP   ST(1),ST	\
    __asm FSTP    QWORD PTR y   \
    }
#elif _ARCH_PPC
/* floor() is actually faster here, as we don't have to set the rounding mode */
#define SINCOS_TRIM_X(y,x) y = x - floor(x);
#else
#define SINCOS_TRIM_X(y,x) y = x - rint(x) + 1.0;
#endif

#else /* SINCOS_REAL4_ARG */

#ifdef _MSC_VER /* no C99 rint() */
#define SINCOS_TRIM_X(y,x) \
  { \
    __asm FLD     DWORD PTR x   \
    __asm FRNDINT               \
    __asm FSUBR   DWORD PTR x   \
    __asm FLD1                  \
    __asm FADDP   ST(1),ST	\
    __asm FSTP    DWORD PTR y   \
    }
#elif _ARCH_PPC
/* floor() is actually faster here, as we don't have to set the rounding mode */
#define SINCOS_TRIM_X(y,x) y = x - floorf(x);
#else
#define SINCOS_TRIM_X(y,x) y = x - rintf(x) + 1.0;
#endif

#endif /* SINCOS_REAL4_ARG */



/*
  Lookup tables (LUT) data
 */

/* Constants */
#define SINCOS_ADDS    402653184.0
#define SINCOS_MASK1   0xFFFFFF
#define SINCOS_MASK2   0x003FFF
#define SINCOS_SHIFT   14
#define SINCOS_LUT_RES 1024 /* should be multiple of 4 */

/* global VARIABLES to be used in (global) macros */
static REAL4 sincosLUTbase[SINCOS_LUT_RES+SINCOS_LUT_RES/4];
static REAL4 sincosLUTdiff[SINCOS_LUT_RES+SINCOS_LUT_RES/4];
/* shift cos tables 90 deg. to sin table */
static const REAL4* cosLUTbase = sincosLUTbase + (SINCOS_LUT_RES/4);
static const REAL4* cosLUTdiff = sincosLUTdiff + (SINCOS_LUT_RES/4);




/*
  LUT initialization
*/

static void local_sin_cos_2PI_LUT_init (void)
{
  static const REAL8 step = LAL_TWOPI / (REAL8)SINCOS_LUT_RES;
  static const REAL8 divide  = 1.0 / ( 1 << SINCOS_SHIFT );
  REAL8 start, end, true_mid, linear_mid;
  int i;

  start = 0.0; /* sin(0 * step) */
  for( i = 0; i < SINCOS_LUT_RES + SINCOS_LUT_RES/4; i++ ) {
    true_mid = sin( ( i + 0.5 ) * step );
    end = sin( ( i + 1 ) * step );
    linear_mid = ( start + end ) * 0.5;
    sincosLUTbase[i] = start + ( ( true_mid - linear_mid ) * 0.5 );
    sincosLUTdiff[i] = ( end - start ) * divide;
    start = end;
  }
}




/*
  LUT evaluation
*/

/* Variables */

static INT4 sincosI, sincosN;

/* A REAL8 variable that allows to read its higher bits as an INT4 */
static union {
  REAL8 asreal;
  struct {
#ifdef __BIG_ENDIAN__
    INT4 dummy;
    INT4 intval;
#else
    INT4 intval;
    INT4 dummy;
#endif
  } as2int;
} sincosUX;



/* Macros to interleave linear sin/cos calculation
   in other code (e.g. AltiVec) */

/* x must already been trimmed to interval [0..2) */
/* - syntactic sugar -|   |- here is the actual code - */
#define SINCOS_PROLOG
#define SINCOS_STEP1(x)   sincosUX.asreal = x + SINCOS_ADDS;
#define SINCOS_STEP2      sincosI = sincosUX.as2int.intval & SINCOS_MASK1;
#define SINCOS_STEP3      sincosN = sincosUX.as2int.intval & SINCOS_MASK2;
#define SINCOS_STEP4      sincosI = sincosI >> SINCOS_SHIFT;
#define SINCOS_STEP5(s)   *s = sincosLUTbase[sincosI] + sincosN * sincosLUTdiff[sincosI];
#define SINCOS_STEP6(c)   *c = cosLUTbase[sincosI]    + sincosN * cosLUTdiff[sincosI];
#define SINCOS_EPILOG(s,c,x)



/* Macros to interleave linear sin/cos calculation
   (in x87 opcodes) with SSE hotloop.*/

/* these actually define only string constants, so they don't harm non-gcc code */
/* handle different register names in AMD64 */
#if defined(__x86_64__)
#define RAX "rax"
#define RDX "rdx"
#define RDI "rdi"
#else
#define RAX "eax"
#define RDX "edx"
#define RDI "edi"
#endif
#define PAX "%%"RAX
#define PDX "%%"RDX
#define PDI "%%"RDI

#ifdef SINCOS_REAL4_ARG
#define SINCOS_FLD "fld"
#else
#define SINCOS_FLD "fldl"
#endif

/* Version 1 : with trimming of input argument to [0,2) */
#define SINCOS_TRIM_P0A(alpha) \
  SINCOS_FLD " %[" #alpha "] \n\t" /* st: alpha */                      \
  "fistpll %[tmp]          \n\t" /* tmp=(INT8)(round((alpha)) */	\
  "fld1                    \n\t" /* st: 1.0 */				\
  "fildll  %[tmp]          \n\t" /* st: 1.0;(round((alpha))*/

#define SINCOS_TRIM_P0B(alpha)						\
  "fsubrp  %%st,%%st(1)    \n\t" /* st: 1.0 -round(alpha) */		\
  "faddl   %[" #alpha "]   \n\t" /* st: alpha -round(alpha)+1.0*/	\
  "faddl   %[sincos_adds]  \n\t" /* ..continue lin. sin/cos as lebow */ \
  "fstpl   %[tmp]          \n\t"

/* Version 2 : assumes input argument is already trimmed */
#define SINCOS_P0(alpha)						\
  SINCOS_FLD " %[" #alpha "] \n\t" /*st:alpha */			\
  "faddl   %[sincos_adds]  \n\t" /*st:alpha+A */                        \
  "fstpl   %[tmp]          \n\t"

#define SINCOS_P1							\
  "mov     %[tmp],"PAX"    \n\t" /* alpha +A ->eax (ix)*/		\
  "mov     "PAX","PDX"     \n\t" /* n  = ix & SINCOS_MASK2 */		\
  "and     $0x3fff,"PAX"   \n\t"
#define SINCOS_P2							\
  "mov     "PAX",%[tmp]    \n\t"					\
  "mov     %[scd], "PAX"   \n\t"					\
  "and     $0xffffff,"PDX" \n\t" /* i  = ix & SINCOS_MASK1;*/
#define SINCOS_P3							\
  "fildl   %[tmp]          \n\t"					\
  "sar     $0xe,"PDX"      \n\t" /*  i  = i >> SINCOS_SHIFT;*/
#define SINCOS_P4							\
  "fld     %%st            \n\t" /* st: n; n; */			\
  "fmuls   ("PAX","PDX",4) \n\t"					\
  "mov     %[scb], "PDI"   \n\t"
#define SINCOS_P4A               /* P4 w/o doubling, used when P6 is not */ \
  "fmuls   ("PAX","PDX",4) \n\t"					\
  "mov     %[scb], "PDI"   \n\t"
#define SINCOS_P5(sin)							\
  "fadds   ("PDI","PDX",4) \n\t" /* st: sincosLUTbase[i]+n*sincosLUTdiff[i]; n*/ \
  "add     $0x100,"PDX"    \n\t" /* edx+=SINCOS_LUT_RES/4*/		\
  "fstps   %[" #sin "]     \n\t" /* (*sin)=sincosLUTbase[i]+n*sincosLUTdiff[i]*/
#define SINCOS_P6(cos) \
  "fmuls   ("PAX","PDX",4) \n\t"					\
  "fadds   ("PDI","PDX",4) \n\t"					\
  "fstps   %[" #cos "]     \n\t" /* (*cos)=cosbase[i]+n*cosdiff[i];*/
/* list of clobbered registers */
#define SINCOS_REGISTERS RAX,RDX,RDI,"st","st(1)","st(2)","cc"



/* LUT evaluation function */

static int local_sin_cos_2PI_LUT_trimmed (REAL4 *s, REAL4 *c, REAL8 x) {

  /* check range of input only in DEBUG mode */
#ifndef LAL_NDEBUG
  if(x > SINCOS_ADDS) {
    XLALPrintError("%s: x too large: %22f > %f\n", __func__, x, SINCOS_ADDS);
    return XLAL_FAILURE;
  } else if(x < -SINCOS_ADDS) {
    XLALPrintError("%s: x too small: %22f < %f\n", __func__, x, -SINCOS_ADDS);
    return XLAL_FAILURE;
  }
#endif

  /* use the macros defined above */
  SINCOS_PROLOG
  SINCOS_STEP1(x)
  SINCOS_STEP2
  SINCOS_STEP3
  SINCOS_STEP4
  SINCOS_STEP5(s)
  SINCOS_STEP6(c)
  SINCOS_EPILOG(s,c,x)

  return XLAL_SUCCESS;
} /* local_sin_cos_2PI_LUT_trimmed */

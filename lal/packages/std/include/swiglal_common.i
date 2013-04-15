//
//  Copyright (C) 2011, 2012 Karl Wette
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with with program; see the file COPYING. If not, write to the
//  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
//  MA  02111-1307  USA
//

// Common SWIG interface code.
// Author: Karl Wette

////////// General SWIG directives //////////

// Ensure that all LAL library wrapping modules share type information.
%begin %{
#define SWIG_TYPE_TABLE swiglal
%}

// Include SWIG headers.
%include <exception.i>
%include <typemaps.i>

// Suppress some SWIG warnings.
#pragma SWIG nowarn=SWIGWARN_PARSE_KEYWORD
#pragma SWIG nowarn=SWIGWARN_LANG_VARARGS_KEYWORD
#pragma SWIG nowarn=SWIGWARN_LANG_OVERLOAD_KEYWORD

// Turn on auto-documentation of functions.
%feature("autodoc", 1);

// Enable keyword arguments for functions.
%feature("kwargs", 1);

////////// Public macros //////////

// Public macros (i.e. those used in headers) are contained inside
// the SWIGLAL() macro, so that they can be easily removed from the
// proprocessing interface. The SWIGLAL_CLEAR() macro is used to
// clear any effects of a public macro. Calls to SWIGLAL_CLEAR()
// are generated from the preprocessing interface.
#define SWIGLAL(...) %swiglal_public_##__VA_ARGS__
#define SWIGLAL_CLEAR(...) %swiglal_public_clear_##__VA_ARGS__

////////// Utility macros and typemaps //////////

// Define a SWIG preprocessor symbol NAME1_NAME2
%define %swiglal_define2(NAME1, NAME2)
%define NAME1##_##NAME2 %enddef
%enddef

// The macro %swiglal_map() maps a one-argument MACRO(X) onto a list
// of arguments (which may be empty). Based on SWIG's %formacro().
%define %_swiglal_map(MACRO, X, ...)
#if #X != ""
MACRO(X);
%_swiglal_map(MACRO, __VA_ARGS__);
#endif
%enddef
%define %swiglal_map(MACRO, ...)
%_swiglal_map(MACRO, __VA_ARGS__, );
%enddef

// The macro %swiglal_map_a() maps a two-argument MACRO(A, X) onto a list
// of arguments (which may be empty), with a common first argument A.
%define %_swiglal_map_a(MACRO, A, X, ...)
#if #X != ""
MACRO(A, X);
%_swiglal_map_a(MACRO, A, __VA_ARGS__);
#endif
%enddef
%define %swiglal_map_a(MACRO, A, ...)
%_swiglal_map_a(MACRO, A, __VA_ARGS__, );
%enddef

// The macro %swiglal_map_ab() maps a three-argument MACRO(A, B, X) onto a list
// of arguments (which may be empty), with a common first arguments A and B.
%define %_swiglal_map_ab(MACRO, A, B, X, ...)
#if #X != ""
MACRO(A, B, X);
%_swiglal_map_ab(MACRO, A, B, __VA_ARGS__);
#endif
%enddef
%define %swiglal_map_ab(MACRO, A, B, ...)
%_swiglal_map_ab(MACRO, A, B, __VA_ARGS__, );
%enddef

// Apply and clear SWIG typemaps.
%define %swiglal_apply(TYPEMAP, TYPE, NAME)
%apply TYPEMAP { TYPE NAME };
%enddef
%define %swiglal_clear(TYPE, NAME)
%clear TYPE NAME;
%enddef

// Apply a SWIG feature.
%define %swiglal_feature(FEATURE, VALUE, NAME)
%feature(FEATURE, VALUE) NAME;
%enddef

// Macros for allocating/copying new instances and arrays
// Analogous to SWIG macros but using XLAL memory functions
#define %swiglal_new_instance(TYPE...) \
  %reinterpret_cast(XLALMalloc(sizeof(TYPE)), TYPE*)
#define %swiglal_new_copy(VAL, TYPE...) \
  %reinterpret_cast(memcpy(%swiglal_new_instance(TYPE), &VAL, sizeof(TYPE)), TYPE*)
#define %swiglal_new_array(SIZE, TYPE...) \
  %reinterpret_cast(XLALMalloc((SIZE)*sizeof(TYPE)), TYPE*)
#define %swiglal_new_copy_array(PTR, SIZE, TYPE...) \
  %reinterpret_cast(memcpy(%swiglal_new_array(SIZE, TYPE), PTR, sizeof(TYPE)*(SIZE)), TYPE*)

////////// General interface code //////////

// Include C99/C++ headers.
%header %{
#ifdef __cplusplus
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <complex>
#else
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>
#endif
%}

// Include LAL headers.
%header %{
#include <lal/LALDatatypes.h>
#include <lal/LALMalloc.h>
#include <lal/XLALError.h>
#include <lal/Date.h>
%}

// Print LAL debugging errors by default.
%init %{
  lalDebugLevel |= LALERROR;
%}

// Version of SWIG used to generate wrapping code.
%inline %{const int swig_version = SWIG_VERSION;%}

// Whether wrapping code was generated in debug mode.
#ifdef NDEBUG
%inline %{const bool swig_debug = false;%}
#else
%inline %{const bool swig_debug = true;%}
#endif

// Constructors for GSL complex numbers, if required.
#ifdef HAVE_LIBGSL
%header %{
  #include <gsl/gsl_complex_math.h>   // provides gsl_complex_rect()
  SWIGINTERNINLINE gsl_complex_float gsl_complex_float_rect(float x, float y) {
    gsl_complex_float z;
    GSL_SET_COMPLEX(&z, x, y);
    return z;
  }
%}
#endif

// Convert XLAL/LAL errors into native scripting-language exceptions:
//  - XLAL: Before performing any action, clear the XLAL error number.
//    Check it after the action is performed; if it is non-zero, raise
//    a native scripting-language exception with the appropriate XLAL
//    error message.
//  - LAL: For any function which has a LALStatus* argument, create a
//    blank LALStatus struct and pass its pointer to the LAL function.
//    After the function is called, check the LALStatus statusCode;
//    if it is non-zero, raise a generic XLAL error (to set the XLAL
//    error number), then a native scripting-language exception. The
//    swiglal_check_LALStatus (C) preprocessor symbol determines if
//    the LAL error handling code is invoked; by default this symbol
//    should be undefined, i.e. functions are XLAL functions by default.
%header %{
static const LALStatus swiglal_empty_LALStatus = {0, NULL, NULL, NULL, NULL, 0, NULL, 0};
#define swiglal_XLAL_error() XLALError(__func__, __FILE__, __LINE__, XLAL_EFAILED)
#undef swiglal_check_LALStatus
%}
%typemap(in, noblock=1, numinputs=0) LALStatus* {
  LALStatus lalstatus = swiglal_empty_LALStatus;
  $1 = &lalstatus;
%#define swiglal_check_LALStatus
}
%exception %{
  XLALClearErrno();
  $action
#ifdef swiglal_check_LALStatus
  if (lalstatus.statusCode) {
    swiglal_XLAL_error();
    SWIG_exception(SWIG_RuntimeError, lalstatus.statusDescription);
  }
#else
  if (xlalErrno) {
    SWIG_exception(SWIG_RuntimeError, XLALErrorString(xlalErrno));
  }
#endif
#undef swiglal_check_LALStatus
%}

////////// General fragments //////////

// Empty fragment, for fragment-generating macros.
%fragment("swiglal_empty_frag", "header") {}

// Wrappers around SWIG's pointer to/from SWIG-wrapped scripting language object functions.
// swiglal_from_SWIGTYPE() simply returns a SWIG-wrapped object containing the input pointer.
// swiglal_from_SWIGTYPE() extracts a pointer from a SWIG-wrapped object, then struct-copies
// the pointer to the supplied output pointer.
%fragment("swiglal_from_SWIGTYPE", "header") {
  SWIGINTERNINLINE SWIG_Object swiglal_from_SWIGTYPE(SWIG_Object self, void *ptr, swig_type_info *tinfo, int tflags) {
    return SWIG_NewPointerObj(ptr, tinfo, tflags);
  }
}
%fragment("swiglal_as_SWIGTYPE", "header") {
  SWIGINTERN int swiglal_as_SWIGTYPE(SWIG_Object self, SWIG_Object obj, void *ptr, size_t len, swig_type_info *tinfo, int tflags) {
    void *vptr = NULL;
    int ecode = SWIG_ConvertPtr(obj, &vptr, tinfo, tflags);
    if (!SWIG_IsOK(ecode)) {
      return ecode;
    }
    memcpy(ptr, vptr, len);
    return ecode;
  }
}

////////// Generate interface //////////

// The SWIG interface file is generated by 'generate_swiglal_iface.py', which
// extracts lists of interface symbols (functions, structs, typedefs to structs)
// and headers from the preprocessing interface, and generates calls to the
// following macros:

// Process an interface function NAME: rename it to RENAME, and set it to
// always return SWIG-owned wrapping objects (unless the function is being
// ignored). If TYPE is given, ignore the return value of the function.
%typemap(out, noblock=1) SWIGTYPE SWIGLAL_RETURN_VOID {
  %set_output(VOID_Object);
}
%typemap(newfree, noblock=1) SWIGTYPE SWIGLAL_RETURN_VOID "";
%define %swiglal_process_function(NAME, RENAME, TYPE)
%rename(#RENAME) NAME;
#if #RENAME != "$ignore"
%feature("new", "1") NAME;
#if #TYPE != ""
%apply SWIGTYPE SWIGLAL_RETURN_VOID { TYPE NAME };
#endif
#endif
%enddef

// Process a typedef to an interface struct TAGNAME: rename it to RENAME.
%define %swiglal_process_tdstruct(TAGNAME, RENAME)
%rename(#RENAME) TAGNAME;
%enddef

// Do not generate any default (copy) contructors or destructors.
%nodefaultctor;
%nocopyctor;
%nodefaultdtor;

// Call the destructor DTORFUNC, but first call swiglal_release_parent()
// to check whether PTR own its own memory (and release any parents).
%define %swiglal_call_dtor(DTORFUNC, PTR)
if (swiglal_release_parent(PTR)) {
  DTORFUNC(PTR);
}
%enddef

// Generate constructors and destructors for a struct NAME.
%define %swiglal_generate_struct_cdtor(NAME, TAGNAME, OPAQUE, DTORFUNC)

  // If this is an opaque struct:
  #if OPAQUE

    // Create an empty struct to represent the opaque struct,
    // so that SWIG has something to attach the destructor to.
    struct TAGNAME {
    };

  #else

    // If there is no XLAL destructor function, this is taken to mean
    // this struct can be validly constructed using XLALCalloc(), i.e.
    // it contains no pointers which must also be initialised.
    #if #DTORFUNC == ""
      %extend TAGNAME {
        TAGNAME() {
          NAME* self = %reinterpret_cast(XLALCalloc(1, sizeof(NAME)), NAME*);
          return self;
        }
        TAGNAME(NAME* src) {
          NAME* self = %reinterpret_cast(XLALCalloc(1, sizeof(NAME)), NAME*);
          memcpy(self, src, sizeof(NAME));
          return self;
        }
      }
    #endif

  #endif

  // If there is no XLAL destructor function, this is taken to mean
  // this struct can be validly destroyed using XLALFree(), i.e.
  // it contains no pointers which must first be destroyed. Otherwise,
  // use the XLAL destructor function.
  #if #DTORFUNC == ""
    %extend TAGNAME {
      ~TAGNAME() {
        %swiglal_call_dtor(XLALFree, $self);
      }
    }
  #else
    %extend TAGNAME {
      ~TAGNAME() {
        %swiglal_call_dtor(%arg(DTORFUNC), $self);
      }
    }
  #endif

%enddef // %swiglal_generate_struct_cdtor()

////////// Fragments and typemaps for arrays //////////

// Map fixed-array types to special variables of their elements,
// e.g. $typemap(swiglal_fixarr_ltype, const int[][]) returns "int".
%typemap(swiglal_fixarr_ltype) SWIGTYPE "$ltype";
%typemap(swiglal_fixarr_ltype) SWIGTYPE[ANY] "$typemap(swiglal_fixarr_ltype, $*type)";
%typemap(swiglal_fixarr_ltype) SWIGTYPE[ANY][ANY] "$typemap(swiglal_fixarr_ltype, $*type)";
%typemap(swiglal_fixarr_pdesc) SWIGTYPE "$&descriptor";
%typemap(swiglal_fixarr_pdesc) SWIGTYPE[ANY] "$typemap(swiglal_fixarr_pdesc, $*type)";
%typemap(swiglal_fixarr_pdesc) SWIGTYPE[ANY][ANY] "$typemap(swiglal_fixarr_pdesc, $*type)";

// The conversion of C arrays to/from scripting-language arrays are performed
// by the following functions:
//  - %swiglal_array_copyin...() copies a scripting-language array into a C array.
//  - %swiglal_array_copyout...() copies a C array into a scripting-language array.
//  - %swiglal_array_viewout...() wraps a C array inside a scripting-language array,
//    if this is supported by the target scripting language.
// These functions accept a subset of the following arguments:
//  - SWIG_Object obj: input scripting-language array.
//  - SWIG_Object parent: SWIG-wrapped object containing parent struct.
//  - void* ptr: pointer to C array.
//  - const size_t esize: size of one C array element, in bytes.
//  - const size_t ndims: number of C array dimensions.
//  - const size_t dims[]: length of each C array dimension.
//  - const size_t strides[]: strides of C array dimension, in number of elements.
//  - swig_type_info *tinfo: SWIG type info of the C array element datatype.
//  - const int tflags: SWIG type conversion flags.
// Return values are:
//  - %swiglal_array_copyin...(): int: SWIG error code.
//  - %swiglal_array_copyout...() SWIG_Object: output scripting-language array.
//  - %swiglal_array_viewout...() SWIG_Object: output scripting-language array view.

// Names of array conversion functions for array type ACFTYPE.
#define %swiglal_array_copyin_func(ACFTYPE) swiglal_array_copyin_##ACFTYPE
#define %swiglal_array_copyout_func(ACFTYPE) swiglal_array_copyout_##ACFTYPE
#define %swiglal_array_viewout_func(ACFTYPE) swiglal_array_viewout_##ACFTYPE

// Names of fragments containing the conversion functions for ACFTYPE.
#define %swiglal_array_copyin_frag(ACFTYPE) "swiglal_array_copyin_" %str(ACFTYPE)
#define %swiglal_array_copyout_frag(ACFTYPE) "swiglal_array_copyout_" %str(ACFTYPE)
#define %swiglal_array_viewout_frag(ACFTYPE) "swiglal_array_viewout_" %str(ACFTYPE)

// The %swiglal_array_type() macro maps TYPEs of C arrays to an ACFTYPE of the
// appropriate array conversion functions, using a special typemap. The typemap also
// ensures that fragments containing the required conversion functions are included.
%define %_swiglal_array_type(ACFTYPE, TYPE)
%fragment("swiglal_array_frags_" %str(ACFTYPE), "header",
          fragment=%swiglal_array_copyin_frag(ACFTYPE),
          fragment=%swiglal_array_copyout_frag(ACFTYPE),
          fragment=%swiglal_array_viewout_frag(ACFTYPE)) {};
%typemap(swiglal_array_typeid, fragment="swiglal_array_frags_" %str(ACFTYPE)) TYPE* %str(ACFTYPE);
%typemap(swiglal_array_typeid, fragment="swiglal_array_frags_" %str(ACFTYPE)) const TYPE* %str(ACFTYPE);
%typemap(swiglal_array_typeid, fragment="swiglal_array_frags_" %str(ACFTYPE)) TYPE[ANY] %str(ACFTYPE);
%typemap(swiglal_array_typeid, fragment="swiglal_array_frags_" %str(ACFTYPE)) TYPE[ANY][ANY] %str(ACFTYPE);
%typemap(swiglal_array_typeid, fragment="swiglal_array_frags_" %str(ACFTYPE)) const TYPE[ANY] %str(ACFTYPE);
%typemap(swiglal_array_typeid, fragment="swiglal_array_frags_" %str(ACFTYPE)) const TYPE[ANY][ANY] %str(ACFTYPE);
%enddef
%define %swiglal_array_type(ACFTYPE, ...)
%swiglal_map_a(%_swiglal_array_type, ACFTYPE, __VA_ARGS__);
%enddef

// Map C array TYPEs to array conversion function ACFTYPEs.
%swiglal_array_type(SWIGTYPE, SWIGTYPE);
%swiglal_array_type(LALchar, char*);
%swiglal_array_type(int8_t, char, signed char, int8_t);
%swiglal_array_type(uint8_t, unsigned char, uint8_t);
%swiglal_array_type(int16_t, short, int16_t);
%swiglal_array_type(uint16_t, unsigned short, uint16_t);
%swiglal_array_type(int32_t, int, int32_t, enum SWIGTYPE);
%swiglal_array_type(uint32_t, unsigned int, uint32_t);
%swiglal_array_type(int64_t, long long, int64_t);
%swiglal_array_type(uint64_t, unsigned long long, uint64_t);
%swiglal_array_type(float, float);
%swiglal_array_type(double, double);
%swiglal_array_type(gsl_complex_float, gsl_complex_float);
%swiglal_array_type(gsl_complex, gsl_complex);
%swiglal_array_type(COMPLEX8, COMPLEX8);
%swiglal_array_type(COMPLEX16, COMPLEX16);

// On modern systems, 'long' could be either 32- or 64-bit...
%swiglal_array_type(int32or64_t, long);
%swiglal_array_type(uint32or64_t, unsigned long);
%define %swiglal_array_int32or64_frags(FRAG, FUNC, INT)
%fragment(FRAG(INT##32or64_t), "header") {
%#if LONG_MAX > INT_MAX
%#define FUNC(INT##32or64_t) FUNC(INT##64_t)
%#else
%#define FUNC(INT##32or64_t) FUNC(INT##32_t)
%#endif
}
%enddef
%swiglal_array_int32or64_frags(%swiglal_array_copyin_frag, %swiglal_array_copyin_func, int);
%swiglal_array_int32or64_frags(%swiglal_array_copyout_frag, %swiglal_array_copyout_func, int);
%swiglal_array_int32or64_frags(%swiglal_array_viewout_frag, %swiglal_array_viewout_func, int);
%swiglal_array_int32or64_frags(%swiglal_array_copyin_frag, %swiglal_array_copyin_func, uint);
%swiglal_array_int32or64_frags(%swiglal_array_copyout_frag, %swiglal_array_copyout_func, uint);
%swiglal_array_int32or64_frags(%swiglal_array_viewout_frag, %swiglal_array_viewout_func, uint);

// Call the appropriate conversion function for C TYPE arrays.
#define %swiglal_array_copyin(TYPE) %swiglal_array_copyin_func($typemap(swiglal_array_typeid, TYPE))
#define %swiglal_array_copyout(TYPE) %swiglal_array_copyout_func($typemap(swiglal_array_typeid, TYPE))
#define %swiglal_array_viewout(TYPE) %swiglal_array_viewout_func($typemap(swiglal_array_typeid, TYPE))

//
// Typemaps which convert to/from fixed-size arrays.
//

// Input typemaps for functions and structs:
%typemap(in) SWIGTYPE[ANY], SWIGTYPE INOUT[ANY] {
  const size_t dims[] = {$1_dim0};
  const size_t strides[] = {1};
  $typemap(swiglal_fixarr_ltype, $1_type) temp[$1_dim0];
  $1 = &temp[0];
  // swiglal_array_typeid input type: $1_type
  int ecode = %swiglal_array_copyin($1_type)(swiglal_no_self(), $input, %as_voidptr($1),
                                             sizeof($1[0]), 1, dims, strides,
                                             $typemap(swiglal_fixarr_pdesc, $1_type),
                                             $disown | %convertptr_flags);
  if (!SWIG_IsOK(ecode)) {
    %argument_fail(ecode, "$type", $symname, $argnum);
  }
}
%typemap(in) SWIGTYPE[ANY][ANY], SWIGTYPE INOUT[ANY][ANY] {
  const size_t dims[] = {$1_dim0, $1_dim1};
  const size_t strides[] = {$1_dim1, 1};
  $typemap(swiglal_fixarr_ltype, $1_type) temp[$1_dim0][$1_dim1];
  $1 = &temp[0];
  // swiglal_array_typeid input type: $1_type
  int ecode = %swiglal_array_copyin($1_type)(swiglal_no_self(), $input, %as_voidptr($1),
                                             sizeof($1[0][0]), 2, dims, strides,
                                             $typemap(swiglal_fixarr_pdesc, $1_type),
                                             $disown | %convertptr_flags);
  if (!SWIG_IsOK(ecode)) {
    %argument_fail(ecode, "$type", $symname, $argnum);
  }
}

// Input typemaps for global variables:
%typemap(varin) SWIGTYPE[ANY] {
  const size_t dims[] = {$1_dim0};
  const size_t strides[] = {1};
  // swiglal_array_typeid input type: $1_type
  int ecode = %swiglal_array_copyin($1_type)(swiglal_no_self(), $input, %as_voidptr($1),
                                             sizeof($1[0]), 1, dims, strides,
                                             $typemap(swiglal_fixarr_pdesc, $1_type),
                                             %convertptr_flags);
  if (!SWIG_IsOK(ecode)) {
    %variable_fail(ecode, "$type", $symname);
  }
}
%typemap(varin) SWIGTYPE[ANY][ANY] {
  const size_t dims[] = {$1_dim0, $1_dim1};
  const size_t strides[] = {$1_dim1, 1};
  // swiglal_array_typeid input type: $1_type
  int ecode = %swiglal_array_copyin($1_type)(swiglal_no_self(), $input, %as_voidptr($1),
                                             sizeof($1[0][0]), 2, dims, strides,
                                             $typemap(swiglal_fixarr_pdesc, $1_type),
                                             %convertptr_flags);
  if (!SWIG_IsOK(ecode)) {
    %variable_fail(ecode, "$type", $symname);
  }
}

// Output typemaps for functions and structs:
%typemap(out) SWIGTYPE[ANY] {
  const size_t dims[] = {$1_dim0};
  const size_t strides[] = {1};
  // swiglal_array_typeid input type: $1_type
%#if $owner & SWIG_POINTER_OWN
  %set_output(%swiglal_array_copyout($1_type)(swiglal_no_self(), %as_voidptr($1),
                                              sizeof($1[0]), 1, dims, strides,
                                              $typemap(swiglal_fixarr_pdesc, $1_type),
                                              $owner | %newpointer_flags));
%#else
  %set_output(%swiglal_array_viewout($1_type)(swiglal_self(), %as_voidptr($1),
                                              sizeof($1[0]), 1, dims, strides,
                                              $typemap(swiglal_fixarr_pdesc, $1_type),
                                              $owner | %newpointer_flags));
%#endif
}
%typemap(out) SWIGTYPE[ANY][ANY] {
  const size_t dims[] = {$1_dim0, $1_dim1};
  const size_t strides[] = {$1_dim1, 1};
  // swiglal_array_typeid input type: $1_type
%#if $owner & SWIG_POINTER_OWN
  %set_output(%swiglal_array_copyout($1_type)(swiglal_no_self(), %as_voidptr($1),
                                              sizeof($1[0][0]), 2, dims, strides,
                                              $typemap(swiglal_fixarr_pdesc, $1_type),
                                              $owner | %newpointer_flags));
%#else
  %set_output(%swiglal_array_viewout($1_type)(swiglal_self(), %as_voidptr($1),
                                              sizeof($1[0][0]), 2, dims, strides,
                                              $typemap(swiglal_fixarr_pdesc, $1_type),
                                              $owner | %newpointer_flags));
%#endif
}

// Output typemaps for global variables:
%typemap(varout) SWIGTYPE[ANY] {
  const size_t dims[] = {$1_dim0};
  const size_t strides[] = {1};
  // swiglal_array_typeid input type: $1_type
  %set_output(%swiglal_array_viewout($1_type)(swiglal_no_self(), %as_voidptr($1),
                                              sizeof($1[0]), 1, dims, strides,
                                              $typemap(swiglal_fixarr_pdesc, $1_type),
                                              %newpointer_flags));
}
%typemap(varout) SWIGTYPE[ANY][ANY] {
  const size_t dims[] = {$1_dim0, $1_dim1};
  const size_t strides[] = {$1_dim1, 1};
  // swiglal_array_typeid input type: $1_type
  %set_output(%swiglal_array_viewout($1_type)(swiglal_no_self(), %as_voidptr($1),
                                              sizeof($1[0][0]), 2, dims, strides,
                                              $typemap(swiglal_fixarr_pdesc, $1_type),
                                              %newpointer_flags));
}

// Argument-output typemaps for functions:
%typemap(in, numinputs=0) SWIGTYPE OUTPUT[ANY] {
  $typemap(swiglal_fixarr_ltype, $1_type) temp[$1_dim0];
  $1 = &temp[0];
}
%typemap(argout) SWIGTYPE OUTPUT[ANY], SWIGTYPE INOUT[ANY] {
  const size_t dims[] = {$1_dim0};
  const size_t strides[] = {1};
  // swiglal_array_typeid input type: $1_type
  %append_output(%swiglal_array_copyout($1_type)(swiglal_no_self(), %as_voidptr($1),
                                                 sizeof($1[0]), 1, dims, strides,
                                                 $typemap(swiglal_fixarr_pdesc, $1_type),
                                                 SWIG_POINTER_OWN | %newpointer_flags));
}
%typemap(in, numinputs=0) SWIGTYPE OUTPUT[ANY][ANY] {
  $typemap(swiglal_fixarr_ltype, $1_type) temp[$1_dim0][$1_dim1];
  $1 = &temp[0];
}
%typemap(argout) SWIGTYPE OUTPUT[ANY][ANY], SWIGTYPE INOUT[ANY][ANY] {
  const size_t dims[] = {$1_dim0, $1_dim1};
  const size_t strides[] = {$1_dim1, 1};
  // swiglal_array_typeid input type: $1_type
  %append_output(%swiglal_array_copyout($1_type)(swiglal_no_self(), %as_voidptr($1),
                                                 sizeof($1[0][0]), 2, dims, strides,
                                                 $typemap(swiglal_fixarr_pdesc, $1_type),
                                                 SWIG_POINTER_OWN | %newpointer_flags));
}

// Public macros to make fixed nD arrays:
// * output-only arguments: SWIGLAL(OUTPUT_nDARRAY(TYPE, ...))
%define %swiglal_public_OUTPUT_1DARRAY(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE OUTPUT[ANY], TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_OUTPUT_1DARRAY(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_OUTPUT_2DARRAY(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE OUTPUT[ANY][ANY], TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_OUTPUT_2DARRAY(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef
// * input-output arguments: SWIGLAL(INOUT_nDARRAY(TYPE, ...))
%define %swiglal_public_INOUT_1DARRAY(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE INOUT[ANY], TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_INOUT_1DARRAY(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_INOUT_2DARRAY(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE INOUT[ANY][ANY], TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_INOUT_2DARRAY(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef

// Get the correct descriptor for a dynamic array element.
%typemap(swiglal_dynarr_pdesc) SWIGTYPE  "$&descriptor";

// The %swiglal_array_dynamic_<n>D() macros create typemaps which convert
// <n>-D dynamically-allocated arrays in structs. The macros must be
// added inside the definition of the struct, before the struct members
// comprising the array are defined. The DATA and N{I,J} members give
// the array data and dimensions, TYPE and SIZET give their respective
// types. The S{I,J} give the strides of the array, in number of elements.
// If the strides are members of the struct, 'arg1->' should be used to
// access the struct itself.
// 1-D arrays:
%define %swiglal_array_dynamic_1D(TYPE, SIZET, DATA, NI, SI)

  // Create immutable members for the array's dimensions.
  %feature("action") NI {result = %static_cast(arg1->NI, SIZET);}
  %extend {
    const SIZET NI;
  }
  %feature("action", "") NI;

  // Typemaps which convert to/from the dynamically-allocated array.
  %typemap(in, noblock=1) TYPE* DATA {
    if (arg1) {
      const size_t dims[] = {arg1->NI};
      const size_t strides[] = {SI};
      $1 = %reinterpret_cast(arg1->DATA, TYPE*);
      // swiglal_array_typeid input type: $1_type
      int ecode = %swiglal_array_copyin($1_type)(swiglal_self(), $input, %as_voidptr($1),
                                                 sizeof(TYPE), 1, dims, strides,
                                                 $typemap(swiglal_dynarr_pdesc, TYPE),
                                                 $disown | %convertptr_flags);
      if (!SWIG_IsOK(ecode)) {
        %argument_fail(ecode, "$type", $symname, $argnum);
      }
    }
  }
  %typemap(out, noblock=1) TYPE* DATA {
    if (arg1) {
      const size_t dims[] = {arg1->NI};
      const size_t strides[] = {SI};
      $1 = %reinterpret_cast(arg1->DATA, TYPE*);
      // swiglal_array_typeid input type: $1_type
      %set_output(%swiglal_array_viewout($1_type)(swiglal_self(), %as_voidptr($1),
                                                  sizeof(TYPE), 1, dims, strides,
                                                  $typemap(swiglal_dynarr_pdesc, TYPE),
                                                  $owner | %newpointer_flags));
    }
  }

  // Clear unneeded typemaps and features.
  %typemap(memberin, noblock=1) TYPE* DATA "";
  %typemap(argout, noblock=1) TYPE* DATA "";
  %typemap(freearg, noblock=1) TYPE* DATA "";
  %feature("action") DATA "";
  %feature("except") DATA "";

  // Create the array's data member.
  %extend {
    TYPE *DATA;
  }

  // Restore modified typemaps and features.
  %feature("action", "") DATA;
  %feature("except", "") DATA;
  %clear TYPE* DATA;

%enddef // %swiglal_array_dynamic_1D()
// 2-D arrays:
%define %swiglal_array_dynamic_2D(TYPE, SIZET, DATA, NI, NJ, SI, SJ)

  // Create immutable members for the array's dimensions.
  %feature("action") NI {result = %static_cast(arg1->NI, SIZET);}
  %feature("action") NJ {result = %static_cast(arg1->NJ, SIZET);}
  %extend {
    const SIZET NI;
    const SIZET NJ;
  }
  %feature("action", "") NI;
  %feature("action", "") NJ;

  // Typemaps which convert to/from the dynamically-allocated array.
  %typemap(in, noblock=1) TYPE* DATA {
    if (arg1) {
      const size_t dims[] = {arg1->NI, arg1->NJ};
      const size_t strides[] = {SI, SJ};
      $1 = %reinterpret_cast(arg1->DATA, TYPE*);
      // swiglal_array_typeid input type: $1_type
      int ecode = %swiglal_array_copyin($1_type)(swiglal_self(), $input, %as_voidptr($1),
                                                 sizeof(TYPE), 2, dims, strides,
                                                 $typemap(swiglal_dynarr_pdesc, TYPE),
                                                 $disown | %convertptr_flags);
      if (!SWIG_IsOK(ecode)) {
        %argument_fail(ecode, "$type", $symname, $argnum);
      }
    }
  }
  %typemap(out, noblock=1) TYPE* DATA {
    if (arg1) {
      const size_t dims[] = {arg1->NI, arg1->NJ};
      const size_t strides[] = {SI, SJ};
      $1 = %reinterpret_cast(arg1->DATA, TYPE*);
      // swiglal_array_typeid input type: $1_type
      %set_output(%swiglal_array_viewout($1_type)(swiglal_self(), %as_voidptr($1),
                                                  sizeof(TYPE), 2, dims, strides,
                                                  $typemap(swiglal_dynarr_pdesc, TYPE),
                                                  $owner | %newpointer_flags));
    }
  }

  // Clear unneeded typemaps and features.
  %typemap(memberin, noblock=1) TYPE* DATA "";
  %typemap(argout, noblock=1) TYPE* DATA "";
  %typemap(freearg, noblock=1) TYPE* DATA "";
  %feature("action") DATA "";
  %feature("except") DATA "";

  // Create the array's data member.
  %extend {
    TYPE *DATA;
  }

  // Restore modified typemaps and features.
  %feature("action", "") DATA;
  %feature("except", "") DATA;
  %clear TYPE* DATA;

%enddef // %swiglal_array_dynamic_2D()

// These macros should be called from within the definitions of
// LAL structs containing dynamically-allocated arrays.
// 1-D arrays:
%define %swiglal_public_1D_ARRAY(TYPE, DATA, SIZET, NI)
%swiglal_array_dynamic_1D(TYPE, SIZET, DATA, NI, 1);
%ignore DATA;
%ignore NI;
%enddef
#define %swiglal_public_clear_1D_ARRAY(TYPE, DATA, SIZET, NI)
// 2-D arrays:
%define %swiglal_public_2D_ARRAY(TYPE, DATA, SIZET, NI, NJ)
%swiglal_array_dynamic_2D(TYPE, SIZET, DATA, NI, NJ, arg1->NJ, 1);
%ignore DATA;
%ignore NI;
%ignore NJ;
%enddef
#define %swiglal_public_clear_2D_ARRAY(TYPE, DATA, SIZET, NI, NJ)

////////// Include scripting-language-specific interface headers //////////

#ifdef SWIGOCTAVE
%include <lal/swiglal_octave.i>
#endif
#ifdef SWIGPYTHON
%include <lal/swiglal_python.i>
#endif

////////// General typemaps and macros //////////

// The SWIGLAL(RETURN_VALUE(TYPE,...)) public macro can be used to ensure
// that the return value of a function is not ignored, if the return value
// has previously been ignored in the generated wrappings.
%define %swiglal_public_RETURN_VALUE(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef
#define %swiglal_public_clear_RETURN_VALUE(TYPE, ...)

// The SWIGLAL(DISABLE_EXCEPTIONS(...)) public macro is useful for
// functions which manipulate XLAL error codes, which thus require
// XLAL exception handling to be disabled.
%define %swiglal_public_DISABLE_EXCEPTIONS(...)
%swiglal_map_ab(%swiglal_feature, "except", "$action", __VA_ARGS__);
%enddef
#define %swiglal_public_clear_DISABLE_EXCEPTIONS(...)

// The SWIGLAL(NO_NEW_OBJECT(...)) macro can be used to turn off
// SWIG object ownership for certain functions.
%define %swiglal_public_NO_NEW_OBJECT(...)
%swiglal_map_ab(%swiglal_feature, "new", "0", __VA_ARGS__);
%enddef
#define %swiglal_public_clear_NO_NEW_OBJECT(...)

// The SWIGLAL(FUNCTION_POINTER(...)) macro can be used to create
// a function pointer constant, for functions which need to be used
// as callback functions.
%define %swiglal_public_FUNCTION_POINTER(...)
%swiglal_map_ab(%swiglal_feature, "callback", "%sPtr", __VA_ARGS__);
%enddef
#define %swiglal_public_clear_FUNCTION_POINTER(...)

// Typemap for functions which return 'int'. If these functions also return
// other output arguments (via 'argout' typemaps), the 'int' return value is
// ignored. This is because 'int' is very commonly used to return an XLAL
// error code, which will be converted into a native scripting-language
// exception, and so the error code itself is not needed directly. To avoid
// having to unpack the error code when collecting the other output arguments,
// therefore, it is ignored in the wrappings. Functions which fit this criteria
// but do return a useful 'int' can use SWIGLAL(RETURN_VALUE(int, ...)) to
// disable this behaviour.
// The 'newfree' typemap is used since its code will appear after all the
// 'argout' typemaps, and will only apply to functions (since only functions
// have %feature("new") set, and thus generate a 'newfree' typemap). The macro
// %swiglal_maybe_drop_first_retval() is defined in the scripting-language-
// specific interface headers; it will drop the first return value (which is
// the 'int') from the output argument list if the argument list contains
// at least 2 items (the 'int' and some other output argument).
%typemap(newfree, noblock=1, fragment="swiglal_maybe_drop_first_retval") int {
  %swiglal_maybe_drop_first_retval();
}

// Typemaps for empty arguments. These typemaps are useful when no input from the
// scripting language is required, and an empty struct needs to be supplied to
// the C function. The SWIGLAL(EMPTY_ARGUMENT(TYPE, ...)) macro applies the typemap which
// supplies a static struct, while the SWIGLAL(NEW_EMPTY_ARGUMENT(TYPE, ...)) macro
// applies the typemap which supplies a dynamically-allocated struct.
%define %swiglal_public_EMPTY_ARGUMENT(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE* SWIGLAL_EMPTY_ARGUMENT, TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_EMPTY_ARGUMENT(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef
%typemap(in, noblock=1, numinputs=0) SWIGTYPE* SWIGLAL_EMPTY_ARGUMENT ($*ltype emptyarg) {
  memset(&emptyarg, 0, sizeof($*type));
  $1 = &emptyarg;
}
%typemap(freearg) SWIGTYPE* SWIGLAL_EMPTY_ARGUMENT "";
%define %swiglal_public_NEW_EMPTY_ARGUMENT(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE* SWIGLAL_NEW_EMPTY_ARGUMENT, TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_NEW_EMPTY_ARGUMENT(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef
%typemap(in, noblock=1, numinputs=0) SWIGTYPE* SWIGLAL_NEW_EMPTY_ARGUMENT {
  $1 = %swiglal_new_instance($*type);
}
%typemap(freearg) SWIGTYPE* SWIGLAL_NEW_EMPTY_ARGUMENT "";

// SWIG conversion functions for C99 integer types.
// These are mapped to the corresponding basic C types,
// conversion functions for which are supplied by SWIG.
%define %swiglal_numeric_typedef(CHECKCODE, BASETYPE, TYPE)
%numeric_type_from(TYPE, BASETYPE);
%numeric_type_asval(TYPE, BASETYPE, "swiglal_empty_frag", false);
%typemaps_primitive(%checkcode(CHECKCODE), TYPE);
%enddef
%swiglal_numeric_typedef(INT8, signed char, int8_t);
%swiglal_numeric_typedef(UINT8, unsigned char, uint8_t);
%swiglal_numeric_typedef(INT16, short, int16_t);
%swiglal_numeric_typedef(UINT16, unsigned short, uint16_t);
%swiglal_numeric_typedef(INT32, int, int32_t);
%swiglal_numeric_typedef(UINT32, unsigned int, uint32_t);
%swiglal_numeric_typedef(INT64, long long, int64_t);
%swiglal_numeric_typedef(UINT64, unsigned long long, uint64_t);

// Fragments and typemaps for the LAL BOOLEAN type. The fragments re-use
// existing scriping-language conversion functions for the C/C++ boolean type.
// Appropriate typemaps are then generated by %typemaps_asvalfromn().
%fragment(SWIG_From_frag(BOOLEAN), "header", fragment=SWIG_From_frag(bool)) {
  SWIGINTERNINLINE SWIG_Object SWIG_From_dec(BOOLEAN)(BOOLEAN value) {
    return SWIG_From(bool)(value ? true : false);
  }
}
%fragment(SWIG_AsVal_frag(BOOLEAN), "header", fragment=SWIG_AsVal_frag(bool)) {
  SWIGINTERN int SWIG_AsVal_dec(BOOLEAN)(SWIG_Object obj, BOOLEAN *val) {
    bool v;
    int ecode = SWIG_AsVal(bool)(obj, val ? &v : 0);
    if (!SWIG_IsOK(ecode)) {
      return SWIG_TypeError;
    }
    if (val) {
      *val = v ? 1 : 0;
    }
    return ecode;
  }
}
%typemaps_primitive(%checkcode(BOOL), BOOLEAN);

// Fragments and typemaps for LAL strings, which should be (de)allocated
// using LAL memory functions. The fragments re-use existing scriping-language
// conversion functions for ordinary char* strings. Appropriate typemaps are
// then generated by %typemaps_string_alloc(), with custom memory allocators.
%fragment("SWIG_FromLALcharPtrAndSize", "header", fragment="SWIG_FromCharPtrAndSize") {
  SWIGINTERNINLINE SWIG_Object SWIG_FromLALcharPtrAndSize(const char *str, size_t size) {
    return SWIG_FromCharPtrAndSize(str, size);
  }
}
%fragment("SWIG_AsLALcharPtrAndSize", "header", fragment="SWIG_AsCharPtrAndSize") {
  SWIGINTERN int SWIG_AsLALcharPtrAndSize(SWIG_Object obj, char **pstr, size_t *psize, int *alloc) {
    char *slstr = 0;
    size_t slsize = 0;
    int slalloc = 0;
    // Get pointer to scripting-language string 'slstr' and size 'slsize'.
    // The 'slalloc' argument indicates whether a new string was allocated.
    int ecode = SWIG_AsCharPtrAndSize(obj, &slstr, &slsize, &slalloc);
    if (!SWIG_IsOK(ecode)) {
      return SWIG_TypeError;
    }
    // Return the string, if needed.
    if (pstr) {
      // Free the LAL string if it is already allocated.
      if (*pstr) {
        XLALFree(*pstr);
      }
      if (alloc) {
        // Copy the scripting-language string into a LAL-managed memory string.
        *pstr = %swiglal_new_copy_array(slstr, slsize, char);
        *alloc = SWIG_NEWOBJ;
      }
      else {
        return SWIG_TypeError;
      }
    }
    // Return the size (length+1) of the string, if needed.
    if (psize) {
      *psize = slsize;
    }
    // Free the scripting-language string, if it was allocated.
    if (slalloc == SWIG_NEWOBJ) {
      %delete_array(slstr);
    }
    return ecode;
  }
}
%fragment("SWIG_AsNewLALcharPtr", "header", fragment="SWIG_AsLALcharPtrAndSize") {
  SWIGINTERN int SWIG_AsNewLALcharPtr(SWIG_Object obj, char **pstr) {
    int alloc = 0;
    return SWIG_AsLALcharPtrAndSize(obj, pstr, 0, &alloc);
  }
}
%typemaps_string_alloc(%checkcode(STRING), %checkcode(char), char, LALchar,
                       SWIG_AsLALcharPtrAndSize, SWIG_FromLALcharPtrAndSize,
                       strlen, %swiglal_new_copy_array, XLALFree,
                       "<limits.h>", CHAR_MIN, CHAR_MAX);

// Do not try to free const char* return arguments.
%typemap(newfree,noblock=1) const char* "";

// Typemap for output SWIGTYPEs. This typemaps will match either the SWIG-wrapped
// return argument from functions (which will have the SWIG_POINTER_OWN bit set
// in $owner) or return a member of a struct through a 'get' functions (in which
// case SWIG_POINTER_OWN will not be set). If it is the latter case, the function
// swiglal_store_parent() is called to store a reference to the struct containing
// the member being accessed, in order to prevent it from being destroyed as long
// as the SWIG-wrapped member object is in scope. The return object is then always
// created with SWIG_POINTER_OWN, so that its destructor will always be called.
%define %swiglal_store_parent(PTR, OWNER)
%#if !(OWNER & SWIG_POINTER_OWN)
  if (%as_voidptr(PTR) != NULL) {
    swiglal_store_parent(%as_voidptr(PTR), swiglal_self());
  }
%#endif
%enddef
%typemap(out,noblock=1) SWIGTYPE *, SWIGTYPE &, SWIGTYPE[] {
  %swiglal_store_parent($1, $owner);
  %set_output(SWIG_NewPointerObj(%as_voidptr($1), $descriptor, ($owner | %newpointer_flags) | SWIG_POINTER_OWN));
}
%typemap(out,noblock=1) const SWIGTYPE *, const SWIGTYPE &, const SWIGTYPE[] {
  %set_output(SWIG_NewPointerObj(%as_voidptr($1), $descriptor, ($owner | %newpointer_flags) & ~SWIG_POINTER_OWN));
}
%typemap(out, noblock=1) SWIGTYPE *const& {
  %swiglal_store_parent(*$1, $owner);
  %set_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*descriptor, ($owner | %newpointer_flags) | SWIG_POINTER_OWN));
}
%typemap(out, noblock=1) const SWIGTYPE *const& {
  %set_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*descriptor, ($owner | %newpointer_flags) & ~SWIG_POINTER_OWN));
}
%typemap(out, noblock=1) SWIGTYPE (void* copy = NULL) {
  copy = %swiglal_new_copy($1, $ltype);
  %swiglal_store_parent(copy, SWIG_POINTER_OWN);
  %set_output(SWIG_NewPointerObj(copy, $&descriptor, (%newpointer_flags) | SWIG_POINTER_OWN));
}
%typemap(varout, noblock=1) SWIGTYPE *, SWIGTYPE [] {
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr($1), $descriptor, (%newpointer_flags) & ~SWIG_POINTER_OWN));
}
%typemap(varout, noblock=1) SWIGTYPE & {
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr(&$1), $descriptor, (%newpointer_flags) & ~SWIG_POINTER_OWN));
}
%typemap(varout, noblock=1) SWIGTYPE {
  %set_varoutput(SWIG_NewPointerObj(%as_voidptr(&$1), $&descriptor, (%newpointer_flags) & ~SWIG_POINTER_OWN));
}

// Typemaps for pointers to primitive scalars. These are treated as output-only
// arguments by default, by globally applying the SWIG OUTPUT typemaps. The INOUT
// typemaps can be supplied as needed using the SWIGLAL(INOUT_SCALARS(TYPE, ...)) macro.
%apply int* OUTPUT { enum SWIGTYPE* };
%apply short* OUTPUT { short* };
%apply unsigned short* OUTPUT { unsigned short* };
%apply int* OUTPUT { int* };
%apply unsigned int* OUTPUT { unsigned int* };
%apply long* OUTPUT { long* };
%apply unsigned long* OUTPUT { unsigned long* };
%apply long long* OUTPUT { long long* };
%apply unsigned long long* OUTPUT { unsigned long long* };
%apply float* OUTPUT { float* };
%apply double* OUTPUT { double* };
%apply int8_t* OUTPUT { int8_t* };
%apply uint8_t* OUTPUT { uint8_t* };
%apply int16_t* OUTPUT { int16_t* };
%apply uint16_t* OUTPUT { uint16_t* };
%apply int32_t* OUTPUT { int32_t* };
%apply uint32_t* OUTPUT { uint32_t* };
%apply int64_t* OUTPUT { int64_t* };
%apply uint64_t* OUTPUT { uint64_t* };
%apply gsl_complex_float* OUTPUT { gsl_complex_float* };
%apply gsl_complex* OUTPUT { gsl_complex* };
%apply COMPLEX8* OUTPUT { COMPLEX8* };
%apply COMPLEX16* OUTPUT { COMPLEX16* };
%define %swiglal_public_INOUT_SCALARS(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, TYPE INOUT, TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_INOUT_SCALARS(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef

// Typemaps for double pointers. By default, treat arguments of type TYPE**
// as output-only arguments, which do not require a scripting-language input
// argument, and return their results in the output argument list. Also supply
// an INOUT typemap for input-output arguments, which allows a scripting-language
// input argument to be supplied. The INOUT typemaps can be applied as needed
// using the SWIGLAL(INOUT_STRUCTS(TYPE, ...)) macro.
%typemap(in, noblock=1, numinputs=0) SWIGTYPE ** (void *argp = NULL, int owner = 0) {
  $1 = %reinterpret_cast(&argp, $ltype);
  owner = (argp == NULL) ? SWIG_POINTER_OWN : 0;
}
%typemap(in, noblock=1) SWIGTYPE ** INOUT (void  *argp = NULL, int owner = 0, int ecode = 0) {
  ecode = SWIG_ConvertPtr($input, &argp, $*descriptor, ($disown | %convertptr_flags) | SWIG_POINTER_DISOWN);
  if (!SWIG_IsOK(ecode)) {
    %argument_fail(ecode, "$type", $symname, $argnum);
  }
  $1 = %reinterpret_cast(&argp, $ltype);
  owner = (argp == NULL) ? SWIG_POINTER_OWN : 0;
}
%typemap(argout, noblock=1) SWIGTYPE ** {
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*descriptor, owner$argnum | %newpointer_flags));
}
%typemap(freearg) SWIGTYPE ** "";
%define %swiglal_public_INOUT_STRUCTS(TYPE, ...)
%swiglal_map_ab(%swiglal_apply, SWIGTYPE ** INOUT, TYPE, __VA_ARGS__);
%enddef
%define %swiglal_public_clear_INOUT_STRUCTS(TYPE, ...)
%swiglal_map_a(%swiglal_clear, TYPE, __VA_ARGS__);
%enddef

// Make the wrapping of printf-style LAL functions a little safer, as suggested in
// the SWIG 2.0 documentation (section 13.5). These functions should now be safely
// able to print any string, so long as the format string is named "format" or "fmt".
%typemap(in, fragment="SWIG_AsCharPtrAndSize") (const char *SWIGLAL_PRINTF_FORMAT, ...)
(char fmt[] = "%s", char *str = 0, int alloc = 0)
{
  $1 = fmt;
  int ecode = SWIG_AsCharPtrAndSize($input, &str, NULL, &alloc);
  if (!SWIG_IsOK(ecode)) {
    %argument_fail(ecode, "const char*, ...", $symname, $argnum);
  }
  $2 = (void *) str;
}
%typemap(freearg, match="in") (const char *format, ...) {
  if (SWIG_IsNewObj(alloc$argnum)) {
    %delete_array(str$argnum);
  }
}
%apply (const char *SWIGLAL_PRINTF_FORMAT, ...) {
  (const char *format, ...), (const char *fmt, ...)
};

// This macro supports functions which require a variable-length
// list of arguments of type TYPE, i.e. a list of strings. It
// generates SWIG "compact" default arguments, i.e. only one
// wrapping function where all missing arguments are assigned ENDVALUE,
// generates 11 additional optional arguments of type TYPE, and
// creates a contract ensuring that the last argument is always
// ENDVALUE, so that the argument list is terminated by ENDVALUE.
%define %swiglal_public_VARIABLE_ARGUMENT_LIST(FUNCTION, TYPE, ENDVALUE)
%feature("kwargs", 0) FUNCTION;
%feature("compactdefaultargs") FUNCTION;
%varargs(11, TYPE arg = ENDVALUE) FUNCTION;
%contract FUNCTION {
require:
  arg11 == ENDVALUE;
}
%enddef
#define %swiglal_public_clear_VARIABLE_ARGUMENT_LIST(FUNCTION, TYPE, ENDVALUE)

// This macro can be used to support structs which are not
// declared in LALSuite. It treats the struct as opaque,
// and attaches a destructor function to it.
%define %swiglal_public_EXTERNAL_STRUCT(NAME, DTORFUNC)
typedef struct {} NAME;
%ignore DTORFUNC;
%extend NAME {
  ~NAME() {
    DTORFUNC($self);
  }
}
%enddef
#define %swiglal_public_clear_EXTERNAL_STRUCT(NAME, DTORFUNC)

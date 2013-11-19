# SWIG configuration
# Author: Karl Wette, 2011, 2012
#
# serial 46

# enable SWIG wrapping modules
AC_DEFUN([LALSUITE_ENABLE_SWIG],[

  # option to enable/disable all languages
  AC_ARG_ENABLE(
    [swig],
    AC_HELP_STRING(
      [--enable-swig],
      [generate SWIG wrapping modules for all languages]
    ),[
      AS_CASE(["${enableval}"],
        [yes],[swig_build_all=true],
        [no],[swig_build_all=false],
        [*],[AC_MSG_ERROR([invalid value "${enableval}" for --enable-swig])]
      )
    ],[
      swig_build_all=
    ]
  )

  # options to enable/disable languages
  swig_build_any=false
  swig_min_version=0.0
  LALSUITE_ENABLE_SWIG_LANGUAGE([Octave],[false],[2.0.7],[LALSUITE_REQUIRE_CXX])
  LALSUITE_ENABLE_SWIG_LANGUAGE([Python],[false],[2.0.7])

])

# options to enable/disable languages
# args: $1=language, $2=default enabled?, $3=SWIG version required, [$4=action if enabled]
AC_DEFUN([LALSUITE_ENABLE_SWIG_LANGUAGE],[
  m4_pushdef([lowercase],translit([$1],[A-Z],[a-z]))

  # command line option to enable/disable $1
  AC_ARG_ENABLE(
    [swig-]lowercase,
    AC_HELP_STRING(
      [--enable-swig-]lowercase,
      [generate SWIG wrapping module for $1]
    ),[
      AS_CASE(["${enableval}"],
        [yes],[swig_build_]lowercase[=true],
        [no],[swig_build_]lowercase[=false],
        [*],[AC_MSG_ERROR([invalid value "${enableval}" for --enable-swig-]]lowercase[)]
      )
    ],[
      swig_build_]lowercase[=${swig_build_all:-$2}
    ]
  )

  # if $1 is enabled, set minimum SWIG version and perform other actions
  AS_IF([test "${swig_build_]lowercase[}" = true],[:
    swig_build_any=true
    AS_VERSION_COMPARE([${swig_min_version}],[$3],[swig_min_version=$3])
    $4
  ])

  m4_popdef([lowercase])
])

# check the version of ${SWIG}, and store it in ${swig_version}
# return swig_version=0.0 if ${SWIG} is not an executable
AC_DEFUN([_LALSUITE_CHECK_SWIG_VERSION],[
  swig_version=0.0
  swig_version_output=[`${SWIG} -version 2>/dev/null`]
  AS_IF([test $? -eq 0],[
    swig_version_regex=['s|^ *SWIG [Vv]ersion \([0-9.][0-9.]*\)|\1|p;d']
    swig_version=[`echo "${swig_version_output}" | ${SED} "${swig_version_regex}"`]
    AS_IF([test "x${swig_version}" = x],[
      AC_MSG_ERROR([could not determine version of ${SWIG}])
    ])
  ])
])

# configure SWIG wrapping modules
# args: $1=symbol prefixes
AC_DEFUN([LALSUITE_USE_SWIG],[

  # save and clear global compiler/linker variables
  swig_save_CPPFLAGS=${CPPFLAGS}
  swig_save_CFLAGS=${CFLAGS}
  swig_save_CXXFLAGS=${CXXFLAGS}
  swig_save_LDFLAGS=${LDFLAGS}
  swig_save_LIBS=${LIBS}
  CPPFLAGS=
  CFLAGS=
  CXXFLAGS=
  LDFLAGS=
  LIBS=

  # check for required programs
  AC_REQUIRE([AC_PROG_LN_S])
  AC_REQUIRE([AC_PROG_MKDIR_P])
  AC_REQUIRE([AC_PROG_SED])

  # if we are wrapping the LAL library (instead of one of the LAL* libraries)
  AS_IF([test "x${PACKAGE}" = xlal],[
    lalswig=true
  ],[
    lalswig=false
  ])

  # if any language was configured
  AM_CONDITIONAL(SWIG_BUILD,[test "${swig_build_any}" = true])
  AM_COND_IF(SWIG_BUILD,[

    # check for SWIG binary with version >= ${swig_min_version}; use
    ## value of ${SWIG} if set, otherwise check common SWIG binary names
    AS_IF([test "x${SWIG}" != x],[
      AC_MSG_CHECKING([${SWIG} version])
      _LALSUITE_CHECK_SWIG_VERSION
      AS_VERSION_COMPARE([${swig_version}],[${swig_min_version}],[
        AC_MSG_ERROR([require ${SWIG} version >= ${swig_min_version}])
      ])
      AC_MSG_RESULT([${swig_version}])
    ],[
      AC_MSG_CHECKING([for SWIG with version >= ${swig_min_version}])
      for SWIG in swig swig2.0; do
        _LALSUITE_CHECK_SWIG_VERSION
        AS_VERSION_COMPARE([${swig_version}],[${swig_min_version}],[],[break],[break])
        SWIG=
      done
      AS_IF([test "x${SWIG}" = x],[
        AC_MSG_ERROR([could not find SWIG with version >= ${swig_min_version}])
      ])
      AC_MSG_RESULT([${SWIG} (version ${swig_version})])
    ])

    # get full path of SWIG binary
    AC_PATH_PROG(SWIG,["${SWIG}"])

    # symbol prefixes for this LAL library
    AC_SUBST(SWIG_SYMBOL_PREFIXES,["$1"])

    # flags for preprocessing/generating SWIG wrapping module sources
    AC_SUBST(SWIG_SWIGFLAGS,["-Wextra -Werror -I\$(abs_top_builddir)/include"])

    # add -MP option if SWIG is greater than version 2.0.9
    AS_VERSION_COMPARE([${swig_version}],[2.0.9],[],[],[
      SWIG_SWIGFLAGS="${SWIG_SWIGFLAGS} -MP"
    ])

    # send language-specific SWIG output files to libtool directory
    AC_SUBST(SWIG_OUTDIR,["\$(abs_builddir)/${objdir}"])
    SWIG_SWIGFLAGS="${SWIG_SWIGFLAGS} -outdir \$(SWIG_OUTDIR)"

    # flags for generating/compiling SWIG wrapping module sources
    AC_SUBST(SWIG_CPPFLAGS,["-I\$(abs_top_builddir)/include -I\$(abs_top_builddir)/src ${swig_save_CPPFLAGS} -I/usr/include"])

    # are we (not) in debugging mode?
    AS_IF([test "x${enable_debug}" = xno],[
      SWIG_SWIGFLAGS="${SWIG_SWIGFLAGS} -DNDEBUG"
      SWIG_CPPFLAGS="${SWIG_CPPFLAGS} -DNDEBUG"
    ])

    # flags for compiling SWIG wrapping module sources
    AC_SUBST(SWIG_CFLAGS,[])
    for arg in ${swig_save_CFLAGS}; do
      AS_CASE([${arg}],
        [-g*|-O*],[SWIG_CFLAGS="${SWIG_CFLAGS} ${arg}"]
      )
    done
    AC_SUBST(SWIG_CXXFLAGS,[])
    for arg in ${swig_save_CXXFLAGS}; do
      AS_CASE([${arg}],
        [-g*|-O*],[SWIG_CXXFLAGS="${SWIG_CXXFLAGS} ${arg}"]
      )
    done

    # define C99 constant and limit macros for C++ sources
    SWIG_CXXFLAGS="${SWIG_CXXFLAGS} -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS"

    # make SWIG use C++ casts in typemaps in C++ mode
    SWIG_CXXFLAGS="${SWIG_CXXFLAGS} -DSWIG_CPLUSPLUS_CAST"

    # disable optimisation in debug mode, for faster compilation
    AS_IF([test "x${enable_debug}" != xno],[
      SWIG_CFLAGS="${SWIG_CFLAGS} -O0"
      SWIG_CXXFLAGS="${SWIG_CXXFLAGS} -O0"
    ])

    # check for additional compiler flags
    extra_flags="-Wno-uninitialized -Wno-unused-variable -fno-strict-aliasing"
    for flag in ${extra_flags}; do
      AC_MSG_CHECKING([if ${flag} is supported])
      CFLAGS=${flag}
      AC_LANG_PUSH([C])
      AC_COMPILE_IFELSE([
        AC_LANG_PROGRAM([AC_INCLUDES_DEFAULT],[])
      ],[
        AC_MSG_RESULT([yes])
        SWIG_CFLAGS="${SWIG_CFLAGS} ${flag}"
        SWIG_CXXFLAGS="${SWIG_CXXFLAGS} ${flag}"
      ],[
        AC_MSG_RESULT([no])
      ])
      CFLAGS=
      AC_LANG_POP([C])
    done

    # flags for linking SWIG wrapping modules
    AC_SUBST(SWIG_LDFLAGS,[])

    # libraries SWIG wrapping module should be linked against
    AC_SUBST(SWIG_LIBS,[])
    AS_IF([test ${lalswig} = true],[
      SWIG_LIBS="\$(abs_top_builddir)/lib/lalsupport/src/liblalsupport.la \$(abs_top_builddir)/lib/lal/liblal.la"
    ],[
      SWIG_LIBS="\$(abs_top_builddir)/src/lib${PACKAGE}.la"
    ])

    # dynamic linker search path for pre-installed LAL libraries
    AS_IF([test "x${LALSUITE_BUILD}" = xtrue],[
      AC_SUBST(SWIG_LD_LIBRARY_PATH,[])
      for arg in ${swig_save_LIBS} ${SWIG_LIBS}; do
        SWIG_LD_LIBRARY_PATH=["${SWIG_LD_LIBRARY_PATH} "`echo ${arg} | ${SED} -n 's|/liblal[^.]*\.la|/'"${objdir}"'|p'`]
      done
      SWIG_LD_LIBRARY_PATH=[`echo ${SWIG_LD_LIBRARY_PATH} | ${SED} 's|(top_builddir)|(abs_top_builddir)|g;s|  *|:|g'`]
      AS_IF([test "${build_vendor}" = apple],[
        SWIG_LD_LIBPATH_NAME=DYLD_LIBRARY_PATH
      ],[
        SWIG_LD_LIBPATH_NAME=LD_LIBRARY_PATH
      ])
      AC_SUBST(SWIG_LD_LIBPATH_NAME)
    ])

    # list of other LAL SWIG modules that this module depends on
    AC_MSG_CHECKING([for SWIG module dependencies])
    AC_SUBST(SWIG_MODULE_DEPENDS,[""])
    for arg in ${LALSUITE_CHECKED_LIBS}; do
      AS_IF([test "x`echo ${arg} | ${SED} -n '/^lalsupport$/d;/^lal/p'`" != x],[
        SWIG_MODULE_DEPENDS="${SWIG_MODULE_DEPENDS} ${arg}"
      ])
    done
    AS_IF([test "x${SWIG_MODULE_DEPENDS}" = x],[
      AC_MSG_RESULT([none])
    ],[
      AC_MSG_RESULT([${SWIG_MODULE_DEPENDS}])
    ])

    # scripting-language path to search for pre-installed SWIG modules
    AC_SUBST(SWIG_PREINST_PATH,["\$(SWIG_OUTDIR)"])
    AS_IF([test "x${LALSUITE_BUILD}" = xtrue],[
      for dir in ${LALSUITE_SUBDIRS}; do
        SWIG_PREINST_PATH="${SWIG_PREINST_PATH}:\$(abs_top_builddir)/../${dir}/\$(subdir)/${objdir}"
      done
    ])

  ])

  # configure SWIG languages
  LALSUITE_USE_SWIG_OCTAVE
  LALSUITE_USE_SWIG_PYTHON

  # restore global compiler/linker variables
  CPPFLAGS=${swig_save_CPPFLAGS}
  CFLAGS=${swig_save_CFLAGS}
  CXXFLAGS=${swig_save_CXXFLAGS}
  LDFLAGS=${swig_save_LDFLAGS}
  LIBS=${swig_save_LIBS}

])

# configure SWIG language wrapping module
# args: $1=language, $2=actions if enabled
AC_DEFUN([LALSUITE_USE_SWIG_LANGUAGE],[
  m4_pushdef([uppercase],translit([$1],[a-z],[A-Z]))
  m4_pushdef([lowercase],translit([$1],[A-Z],[a-z]))

  # check whether to configure $1
  AM_CONDITIONAL(SWIG_BUILD_[]uppercase,[test ${swig_build_]lowercase[} = true])
  AM_COND_IF(SWIG_BUILD_[]uppercase,[

    # at least one language was configured
    swig_build=true

    # set message string to indicate language will be built
    SWIG_]uppercase[_ENABLE_VAL=ENABLED

    # configure $1
    $2
    # $1 has been configured

  ],[
    SWIG_]uppercase[_ENABLE_VAL=DISABLED
  ])

  m4_popdef([uppercase])
  m4_popdef([lowercase])
])

# configure SWIG Octave wrapping module
AC_DEFUN([LALSUITE_USE_SWIG_OCTAVE],[
  LALSUITE_USE_SWIG_LANGUAGE([Octave],[

    # check for Octave binary
    AC_PATH_PROG(OCTAVE,[octave],[],[])
    AS_IF([test "x${OCTAVE}" = x],[
      AC_MSG_ERROR([could not find "octave" in path])
    ])
    octave_eval="env - ${OCTAVE} -qfH --eval"
    octave_prefix=[`${octave_eval} "disp(octave_config_info('prefix'))" 2>/dev/null | ${SED} 's|/*$||'`]

    # check for Octave mkoctfile binary
    AC_MSG_CHECKING([for mkoctfile])
    AS_IF([test "x`${octave_eval} 'mkoctfile -p CXX' 2>/dev/null`" != x],[
      AC_MSG_RESULT([yes])
    ],[
      AC_MSG_ERROR([mkoctfile is not installed])
    ])

    # check Octave version
    octave_min_version=3.2.0
    AC_MSG_CHECKING([${OCTAVE} version])
    octave_version=[`${octave_eval} "disp(version)" 2>/dev/null`]
    AS_IF([test "x${octave_version}" = x],[
      AC_MSG_ERROR([could not determine ${OCTAVE} version])
    ])
    AC_MSG_RESULT([${octave_version}])
    AS_VERSION_COMPARE([${octave_min_version}],[${octave_version}],[],[],[
      AC_MSG_ERROR([require ${OCTAVE} version >= ${octave_min_version}])
    ])

    # check that wrappings are being compiled with the same C++ compiler used to compile Octave itself
    AC_MSG_CHECKING([C++ compiler used for building ${OCTAVE}])
    octave_CXX=`${octave_eval} "mkoctfile -p CXX" 2>/dev/null`
    AC_MSG_RESULT([${octave_CXX}])
    octave_CXX_version=`${octave_CXX} --version 2>/dev/null | ${SED} -n '1p'`
    lalsuite_CXX_version=`${CXX} --version 2>/dev/null | ${SED} -n '1p'`
    AS_IF([test "x${lalsuite_CXX_version}" != "x${octave_CXX_version}"],[
      AC_MSG_ERROR([configured C++ compiler "${CXX}" differs from ${OCTAVE} C++ compiler "${octave_CXX}"])
    ])

    # determine Octave module flags
    AC_MSG_CHECKING([for ${OCTAVE} module CPPFLAGS])
    AC_SUBST(OCTAVE_CPPFLAGS,[""])
    for arg in CPPFLAGS INCFLAGS; do
      OCTAVE_CPPFLAGS="${OCTAVE_CPPFLAGS} "`${octave_eval} "mkoctfile -p ${arg}" 2>/dev/null`
    done
    AC_MSG_RESULT([${OCTAVE_CPPFLAGS}])
    AC_MSG_CHECKING([for ${OCTAVE} module CXXFLAGS])
    AC_SUBST(OCTAVE_CXXFLAGS,[""])
    for arg in ALL_CXXFLAGS; do
      OCTAVE_CXXFLAGS="${OCTAVE_CXXFLAGS} "`${octave_eval} "mkoctfile -p ${arg}" 2>/dev/null`
    done
    AC_MSG_RESULT([${OCTAVE_CXXFLAGS}])

    # check for Octave headers
    CPPFLAGS=${OCTAVE_CPPFLAGS}
    AC_LANG_PUSH([C++])
    AC_CHECK_HEADERS([octave/oct.h],[],[
      AC_MSG_ERROR([could not find the header "octave/oct.h"])
    ],[
      AC_INCLUDES_DEFAULT
    ])
    CPPFLAGS=
    AC_LANG_POP([C++])

    # determine where to install Octave module: take versioned site .oct file
    # directory given by octave-config, and strip off prefix; thus, if LALSuite
    # is installed in the same directory as Octave, .oct module files will be
    # found by Octave without having to add to OCTAVE_PATH
    AC_MSG_CHECKING([for ${OCTAVE} module installation directory])
    octexecdir=[`${octave_eval} "disp(octave_config_info('localveroctfiledir'))" 2>/dev/null | ${SED} 's|/*$||'`]
    octexecdir=[`echo ${octexecdir} | ${SED} "s|^${octave_prefix}/||"`]
    AS_IF([test "x`echo ${octexecdir} | ${SED} -n '\|^/|p'`" != x],[
      AC_MSG_ERROR([could not build relative path from "${octexecdir}"])
    ])
    octexecdir='${prefix}'/"${octexecdir}"
    AC_MSG_RESULT([${octexecdir}])
    AC_SUBST(octexecdir)

    # save Octave path to search for dependent SWIG modules
    AC_ARG_VAR(OCTAVE_PATH,[Octave path to search for dependent SWIG modules])

  ])
])

# configure SWIG Python wrapping module
AC_DEFUN([LALSUITE_USE_SWIG_PYTHON],[
  LALSUITE_USE_SWIG_LANGUAGE([Python],[

    # check for Python
    python_min_version=2.5
    AM_PATH_PYTHON([${python_min_version}])

    # check for distutils
    AC_MSG_CHECKING([for distutils])
    cat <<EOD | ${PYTHON} - 2>/dev/null
import distutils
EOD
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not import distutils])
    ])
    AC_MSG_RESULT([yes])

    # check for NumPy
    numpy_min_version=1.3
    AC_MSG_CHECKING([for NumPy])
    numpy_version=[`cat <<EOD | ${PYTHON} - 2>/dev/null
import numpy
print(numpy.__version__)
EOD`]
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not import NumPy])
    ])
    AC_MSG_RESULT([yes])

    # check NumPy version
    AC_MSG_CHECKING([NumPy version])
    AS_VERSION_COMPARE([${numpy_min_version}],[${numpy_version}],[],[],[
      AC_MSG_ERROR([require NumPy version >= ${numpy_min_version}])
    ])
    AC_MSG_RESULT([${numpy_version}])

    # determine Python module CPPFLAGS
    AC_MSG_CHECKING([for ${PYTHON} module CPPFLAGS])
    PYTHON_CPPFLAGS=[`cat <<EOD | ${PYTHON} - 2>/dev/null
import sys
import distutils.sysconfig as cfg
import numpy.lib.utils as npyutil
sys.stdout.write( '-I' + cfg.get_python_inc())
sys.stdout.write(' -I' + cfg.get_python_inc(plat_specific=1))
sys.stdout.write(' -I' + npyutil.get_include())
EOD`]
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not determine ${PYTHON} module CPPFLAGS])
    ])
    AC_SUBST(PYTHON_CPPFLAGS)
    AC_MSG_RESULT([${PYTHON_CPPFLAGS}])

    # determine Python module CFLAGS
    AC_MSG_CHECKING([for ${PYTHON} module CFLAGS])
    PYTHON_CFLAGS=[`cat <<EOD | ${PYTHON} - 2>/dev/null
import sys
import distutils.sysconfig as cfg
cflags = cfg.get_config_var('CFLAGS').split()
cflags = [f for f in cflags if f != '-DNDEBUG']
sys.stdout.write(" ".join(cflags))
EOD`]
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not determine ${PYTHON} module CFLAGS])
    ])
    AC_SUBST(PYTHON_CFLAGS)
    AC_MSG_RESULT([${PYTHON_CFLAGS}])

    # check for Python headers
    CPPFLAGS=${PYTHON_CPPFLAGS}
    AC_LANG_PUSH([C])
    AC_CHECK_HEADERS([Python.h],[],[
      AC_MSG_ERROR([could not find the header "Python.h"])
    ],[
      AC_INCLUDES_DEFAULT
    ])
    AC_CHECK_HEADERS([numpy/arrayobject.h],[],[
      AC_MSG_ERROR([could not find the header "numpy/arrayobject.h"])
    ],[
      AC_INCLUDES_DEFAULT
      #include <Python.h>
    ])
    CPPFLAGS=
    AC_LANG_POP([C])

    # save Python path to search for dependent SWIG modules
    AC_ARG_VAR(PYTHONPATH,[Python path to search for dependent SWIG modules])

  ])
])

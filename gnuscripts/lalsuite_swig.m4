# -*- mode: autoconf; -*-
# lalsuite_swig.m4 - SWIG configuration
# Author: Karl Wette, 2011--2017
#
# serial 95

AC_DEFUN([_LALSUITE_CHECK_SWIG_VERSION],[
  # $0: check the version of $1, and store it in ${swig_version}
  swig_version=
  swig_version_output=[`$1 -version 2>/dev/null`]
  AS_IF([test $? -eq 0],[
    swig_version_regex=['s|^ *SWIG [Vv]ersion \([0-9.][0-9.]*\)|\1|p;d']
    swig_version=[`echo "${swig_version_output}" | ${SED} -e "${swig_version_regex}"`]
  ])
  AS_IF([test "x${swig_version}" = x],[
    AC_MSG_ERROR([could not determine version of $1])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_ENABLE_SWIG],[
  # $0: enable SWIG bindings
  AC_ARG_ENABLE(
    [swig],
    AC_HELP_STRING(
      [--enable-swig],
      [generate SWIG bindings for all languages]
    ),[
      AS_CASE(["${enableval}"],
        [yes],[swig_build_all=true],
        [no],[swig_build_all=false],
        [AC_MSG_ERROR([invalid value "${enableval}" for --enable-swig])]
      )
    ],[
      swig_build_all=
    ]
  )
  AC_ARG_ENABLE(
    [swig_iface],
    AC_HELP_STRING(
      [--enable-swig-iface],
      [generate SWIG interface only]
    ),[
      AS_CASE(["${enableval}"],
        [yes],[swig_build_iface=true],
        [no],[swig_build_iface=false],
        [AC_MSG_ERROR([invalid value "${enableval}" for --enable-swig-iface])]
      )
    ],[
      swig_build_iface=false
    ]
  )
  LALSUITE_ENABLE_SWIG_LANGUAGE([Octave],[false],[
    # C++ is required to build Octave wrappings
    LALSUITE_REQUIRE_CXX
  ])
  LALSUITE_ENABLE_SWIG_LANGUAGE([Python],[false],[
    # Python is required to configure Python wrappings
    LALSUITE_REQUIRE_PYTHON([2.6])
  ])
  AS_IF([test "${swig_build_iface}" = true],[
    # Python is required to run generate_swig_iface.py
    LALSUITE_REQUIRE_PYTHON([2.6])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_ENABLE_SWIG_LANGUAGE],[
  # $0: enable SWIG binding languages
  m4_pushdef([uppercase],m4_translit([$1],[a-z],[A-Z]))
  m4_pushdef([lowercase],m4_translit([$1],[A-Z],[a-z]))
  AC_ARG_ENABLE(
    [swig-]lowercase,
    AC_HELP_STRING(
      [--enable-swig-]lowercase,
      [generate SWIG bindings for $1]
    ),[
      AS_CASE(["${enableval}"],
        [yes],[swig_build_]lowercase[=true],
        [no],[swig_build_]lowercase[=false],
        [AC_MSG_ERROR([invalid value "${enableval}" for --enable-swig-]]lowercase[)]
      )
    ],[
      swig_build_]lowercase[=${swig_build_all:-$2}
    ]
  )
  AS_IF([test "${swig_build_]lowercase[}" = true],[
    swig_build_iface=true
    SWIG_BUILD_]uppercase[_ENABLE_VAL=ENABLED
    $3
  ],[
    SWIG_BUILD_]uppercase[_ENABLE_VAL=DISABLED
  ])
  AM_CONDITIONAL([SWIG_BUILD_]uppercase,[test "${swig_build_]lowercase[}" = true])
  m4_popdef([uppercase])
  m4_popdef([lowercase])
  # end $0
])

AC_DEFUN([LALSUITE_USE_SWIG],[
  # $0: configure enabled SWIG bindings
  AS_IF([test "${swig_build_iface}" = true],[

    # configure SWIG binding languages
    swig_min_version=2.0.12
    swig_min_version_info=""
    LALSUITE_USE_SWIG_OCTAVE
    LALSUITE_USE_SWIG_PYTHON

    # check for SWIG binary with version ${swig_min_version} or later;
    # use ${SWIG} if set, otherwise check common SWIG binary names
    AC_SUBST([SWIG])
    AS_IF([test "x${SWIG}" != x],[
      AC_MSG_CHECKING([if ${SWIG} version is at least ${swig_min_version}])
      _LALSUITE_CHECK_SWIG_VERSION([${SWIG}])
      LALSUITE_VERSION_COMPARE([${swig_version}],[<],[${swig_min_version}],[
        AC_MSG_RESULT([no (${swig_version})])
        AC_MSG_ERROR([SWIG version ${swig_min_version} or later is required ${swig_min_version_info}])
      ])
      AC_MSG_RESULT([yes (${swig_version})])
    ],[
      AC_PATH_PROGS_FEATURE_CHECK([SWIG],[swig swig2.0 swig3.0],[
        AC_MSG_CHECKING([if ${ac_path_SWIG} version is at least ${swig_min_version}])
        _LALSUITE_CHECK_SWIG_VERSION([${ac_path_SWIG}])
        LALSUITE_VERSION_COMPARE([${swig_version}],[>=],[${swig_min_version}],[
          ac_path_SWIG_found=true
          AC_MSG_RESULT([yes (${swig_version})])
          ac_cv_path_SWIG="${ac_path_SWIG}"
        ],[
          ac_path_SWIG_found=false
          AC_MSG_RESULT([no (${swig_version})])
        ])
      ],[
        AC_MSG_ERROR([SWIG version ${swig_min_version} or later is required ${swig_min_version_info}])
      ])
      SWIG="${ac_cv_path_SWIG}"
    ])

    # check if SWIG works with ccache
    ccache_swig_env="CCACHE_CPP2=1"
    AC_MSG_CHECKING([if ${SWIG} works with ${ccache_swig_env}])
    echo '%module conftest;' > conftest-swig.i
    env_ccache_swig_cmd="env ${ccache_swig_env} ${SWIG} -includeall -ignoremissing -xml -xmllite -MP -MD -MF conftest-swig.deps -o conftest-swig.xml conftest-swig.i"
    _AS_ECHO_LOG([${env_ccache_swig_cmd}])
    ${env_ccache_swig_cmd} >&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD
    result=$?
    _AS_ECHO_LOG([\$? = ${result}])
    AS_IF([test ${result} -eq 0],[
      AC_MSG_RESULT([yes])
    ],[
      AC_MSG_RESULT([no])
      ccache_swig_env="CCACHE_DISABLE=1"
    ])
    SWIG="env ${ccache_swig_env} ${SWIG}"
    rm -f conftest-swig.i conftest-swig.deps conftest-swig.xml

    # extract -I and -D flags from LALSuite library preprocessor flags
    AC_SUBST([SWIG_CPPFLAGS],[])
    for flag in ${CPPFLAGS}; do
      AS_CASE([${flag}],
        [-I*|-D*],[SWIG_CPPFLAGS="${SWIG_CPPFLAGS} ${flag}"]
      )
    done

    # extract -L flags from LALSuite library linker flags
    AC_SUBST([SWIG_LDFLAGS],[])
    for flag in ${LDFLAGS}; do
      AS_CASE([${flag}],
        [-L*],[SWIG_LDFLAGS="${SWIG_LDFLAGS} ${flag}"]
      )
    done

    # determine various libtool parameters
    AC_SUBST([SWIG_LTLIBDIR],["${objdir}"])
    AC_SUBST([SWIG_SOEXT],[`module=yes; eval shrext=\"${shrext_cmds}\"; echo ${shrext}`])

    # substitute list of LALSuite bindings that these bindings depend on
    AC_MSG_CHECKING([for LALSuite binding dependencies])
    AC_SUBST([SWIG_DEPENDENCIES],[])
    for arg in ${lalsuite_libs}; do
      AS_CASE([${arg}],
        [lalsupport],[:],
        [SWIG_DEPENDENCIES="${SWIG_DEPENDENCIES} ${arg}"]
      )
    done
    AS_IF([test "x${SWIG_DEPENDENCIES}" = x],[
      AC_MSG_RESULT([none])
    ],[
      AC_MSG_RESULT([${SWIG_DEPENDENCIES}])
    ])

  ])
  AM_CONDITIONAL([SWIG_BUILD],[test "${swig_build_iface}" = true])

  # run SWIG binding language post-actions
  _LALSUITE_USE_SWIG_POST_ACTIONS

  # end $0
])

m4_define([_LALSUITE_USE_SWIG_POST_ACTIONS],[])

AC_DEFUN([LALSUITE_USE_SWIG_LANGUAGE],[
  # $0: configure SWIG binding languages
  m4_pushdef([uppercase],m4_translit([$1],[a-z],[A-Z]))
  m4_pushdef([lowercase],m4_translit([$1],[A-Z],[a-z]))
  AS_IF([test "${swig_build_]lowercase[}" = true],[
    AC_LANG_PUSH([$2])
    swig_build=true
    $3
    AC_LANG_POP([$2])
  ])
  m4_append([_LALSUITE_USE_SWIG_POST_ACTIONS],[
    AS_IF([test "${swig_build_]]lowercase[[}" = true],[
      $4
    ])
  ])
  m4_popdef([uppercase])
  m4_popdef([lowercase])
  # end $0
])

AC_DEFUN([LALSUITE_USE_SWIG_OCTAVE],[
  # $0: configure SWIG Octave bindings
  LALSUITE_USE_SWIG_LANGUAGE([Octave],[C++],[

    # check for Octave
    AC_PATH_PROGS(OCTAVE,[octave-cli octave],[],[])
    AS_IF([test "x${OCTAVE}" = x],[
      AC_MSG_ERROR([could not find octave in PATH])
    ])

    # check for Octave utilities octave-config and mkoctfile
    octave_dir=`AS_DIRNAME(["${OCTAVE}"])`
    AC_MSG_CHECKING([for octave-config])
    octave_cfg="${octave_dir}/octave-config"
    AS_IF([test -x "${octave_cfg}"],[
      AC_MSG_RESULT([${octave_cfg}])
    ],[
      AC_MSG_RESULT([not found])
      AC_MSG_ERROR([could not find octave-config in ${octave_dir}])
    ])
    octave_cfg="env - ${octave_cfg}"
    AC_MSG_CHECKING([for mkoctfile])
    mkoctfile="${octave_dir}/mkoctfile"
    AS_IF([test -x "${mkoctfile}"],[
      AC_MSG_RESULT([${mkoctfile}])
    ],[
      AC_MSG_RESULT([not found])
      AC_MSG_ERROR([could not find mkoctfile in ${octave_dir}])
    ])
    mkoctfile="env - ${mkoctfile}"

    # check Octave version
    octave_min_version=3.2.0
    AC_MSG_CHECKING([${OCTAVE} version])
    octave_version=[`${octave_cfg} -p VERSION 2>/dev/null`]
    AS_IF([test "x${octave_version}" = x],[
      AC_MSG_ERROR([could not determine ${OCTAVE} version])
    ])
    AC_MSG_RESULT([${octave_version}])
    LALSUITE_VERSION_COMPARE([${octave_version}],[<],[${octave_min_version}],[
      AC_MSG_ERROR([Octave version ${octave_min_version} or later is required])
    ])
    LALSUITE_VERSION_COMPARE([${octave_version}],[>=],[4.0.0],[
      LALSUITE_VERSION_COMPARE([${swig_min_version}],[<],[3.0.7],[
        swig_min_version=3.0.7
        swig_min_version_info="for Octave version ${octave_version}"
      ])
    ])
    LALSUITE_VERSION_COMPARE([${octave_version}],[>=],[4.2.0],[
      LALSUITE_VERSION_COMPARE([${swig_min_version}],[<],[3.0.12],[
        swig_min_version=3.0.12
        swig_min_version_info="for Octave version ${octave_version}"
      ])
    ])

    # determine where to install Octave bindings: take versioned site .oct file
    # directory given by octave-config, and strip off prefix; thus, if LALSuite
    # is installed in the same directory as Octave, .oct files will be found by
    # Octave without having to add to OCTAVE_PATH
    AC_MSG_CHECKING([${OCTAVE} .oct installation directory])
    octave_prefix=[`${octave_cfg} -p PREFIX 2>/dev/null | ${SED} -e 's|/*$||'`]
    octexecdir=[`${octave_cfg} -p LOCALVEROCTFILEDIR 2>/dev/null | ${SED} -e 's|/*$||'`]
    octexecdir=[`echo ${octexecdir} | ${SED} -e "s|^${octave_prefix}/||"`]
    AS_IF([test "x`echo ${octexecdir} | ${SED} -n -e '\|^/|p'`" != x],[
      AC_MSG_ERROR([could not build relative path from "${octexecdir}"])
    ])
    octexecdir='${prefix}'/"${octexecdir}"
    AC_MSG_RESULT([${octexecdir}])
    AC_SUBST([octexecdir])

    # determine C++ compiler used to compile Octave itself
    AC_MSG_CHECKING([C++ compiler used for building ${OCTAVE}])
    octave_CXX=`${mkoctfile} -p CXX 2>/dev/null`
    AC_MSG_RESULT([${octave_CXX}])

    # check that configured C++ compiler is compatible with C++ compiler used to
    # compile Octave itself, i.e. that both compilers link against compatible C++
    # libraries (e.g. libstdc++ vs libc++).
    AC_MSG_CHECKING([configured C++ compiler "${CXX}" is compatible with ${OCTAVE} C++ compiler "${octave_CXX}"])
    AS_IF([test "x${build_vendor}" = xapple && otool --version >/dev/null 2>&1],[
      print_shared_libs="otool -L"
    ],[ldd --version >/dev/null 2>&1],[
      print_shared_libs="ldd"
    ],[
      AC_MSG_ERROR([could not determine tool to print shared library dependencies])
    ])
    swig_save_CXX=${CXX}
    LALSUITE_PUSH_UVARS
    LALSUITE_CLEAR_UVARS
    m4_foreach([cxxloop],[CXX,octave_CXX],[
      CXX=${cxxloop}
      AC_LINK_IFELSE([
        AC_LANG_SOURCE([[
#include <string>
int main() { std::string s = "a"; return 0; }
        ]])
      ],[
        print_shared_libs_regex=["\|conftest${EXEEXT}|d;"'s|(0x[^)]*)||g;s|^ *||g;s| *$||g']
        ${print_shared_libs} conftest${EXEEXT} | ${SED} -e "${print_shared_libs_regex}" | sort > conftest_lalsuite_swig_[]cxxloop[]_shared_libs
        echo "${as_me}:${as_lineno-$LINENO}:${CXX} shared libraries:" >&AS_MESSAGE_LOG_FD
        ${SED} -e ["s/^/${as_me}:${as_lineno-$LINENO}:/"] conftest_lalsuite_swig_[]cxxloop[]_shared_libs >&AS_MESSAGE_LOG_FD
      ],[
        AC_MSG_ERROR([could not link using ${CXX}])
      ])
    ])
    LALSUITE_POP_UVARS
    CXX=${swig_save_CXX}
    AS_IF([diff conftest_lalsuite_swig_CXX_shared_libs conftest_lalsuite_swig_octave_CXX_shared_libs >/dev/null 2>&1],[
      AC_MSG_RESULT([yes])
    ],[
      AC_MSG_RESULT([no])
      AC_MSG_ERROR([configured C++ compiler "${CXX}" is incompatible with ${OCTAVE} C++ compiler "${octave_CXX}"])
    ])

    # determine Octave preprocessor flags
    AC_SUBST([SWIG_OCTAVE_CPPFLAGS],[])
    AC_SUBST([SWIG_OCTAVE_CPPFLAGS_IOCTAVE],[])
    for arg in CPPFLAGS INCFLAGS; do
      for flag in `${mkoctfile} -p ${arg} 2>/dev/null`; do
        AS_CASE([${flag}],
          [-I*/octave],[SWIG_OCTAVE_CPPFLAGS_IOCTAVE="${flag}"],
          [SWIG_OCTAVE_CPPFLAGS="${SWIG_OCTAVE_CPPFLAGS} ${flag}"]
        )
      done
    done

    # determine Octave compiler flags
    AC_SUBST([SWIG_OCTAVE_CXXFLAGS],[])
    swig_octave_cxxflags=
    for arg in CXX CXXPICFLAG ALL_CXXFLAGS; do
      for flag in `${mkoctfile} -p ${arg} 2>/dev/null`; do
        AS_CASE([${flag}],
          [-*],[swig_octave_cxxflags="${swig_octave_cxxflags} ${flag}"]
        )
      done
    done
    LALSUITE_CHECK_COMPILE_FLAGS([
      ${swig_octave_cxxflags}
      -Wno-uninitialized
      -Wno-unused-variable
      -Wno-unused-but-set-variable
      -Wno-format-extra-args
      -Wno-tautological-compare
      -Wno-deprecated-declarations
      -fno-strict-aliasing
      -O0
      -Wp[,]-U_FORTIFY_SOURCE
      ],[SWIG_OCTAVE_CXXFLAGS="${SWIG_OCTAVE_CXXFLAGS} ${flag}"]
    )

    # determine Octave linker flags
    AC_SUBST([SWIG_OCTAVE_LDFLAGS],[])
    swig_octave_ldflags=
    for arg in LFLAGS LIBOCTINTERP LIBOCTAVE LIBCRUFT OCT_LINK_OPTS OCT_LINK_DEPS; do
      for flag in `${mkoctfile} -p ${arg} 2>/dev/null`; do
        AS_CASE([${flag}],
          [-L/usr/lib|-L/usr/lib64],[:],
          [-Xlinker],[swig_octave_ldflags="${swig_octave_ldflags}-Wl,"],
          [swig_octave_ldflags="${swig_octave_ldflags}${flag} "]
        )
      done
    done
    LALSUITE_CHECK_LINK_FLAGS([
      ${swig_octave_ldflags}
      ],[SWIG_OCTAVE_LDFLAGS="${SWIG_OCTAVE_LDFLAGS} ${flag}"]
    )

    # check for Octave headers
    LALSUITE_PUSH_UVARS
    CPPFLAGS="${SWIG_OCTAVE_CPPFLAGS_IOCTAVE} ${SWIG_OCTAVE_CPPFLAGS}"
    CXXFLAGS="${SWIG_OCTAVE_CXXFLAGS}"
    AC_CHECK_HEADER([octave/oct.h],[],[
      AC_MSG_ERROR([could not find the header "octave/oct.h"])
    ],[
      AC_INCLUDES_DEFAULT
    ])
    LALSUITE_POP_UVARS

  ],[

    # determine SWIG Octave flags
    AC_SUBST([SWIG_OCTAVE_FLAGS],["-globals . -DSWIG_CPLUSPLUS_CAST"])

  ])
  # end $0
])

AC_DEFUN([LALSUITE_USE_SWIG_PYTHON],[
  # $0: configure SWIG Python bindings
  LALSUITE_USE_SWIG_LANGUAGE([Python],[C],[

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
    LALSUITE_VERSION_COMPARE([${numpy_version}],[<],[${numpy_min_version}],[
      AC_MSG_ERROR([NumPy version ${numpy_min_version} or later is required])
    ])
    AC_MSG_RESULT([${numpy_version}])

    # determine Python preprocessor flags
    AC_SUBST([SWIG_PYTHON_CPPFLAGS],[])
    python_out=[`cat <<EOD | ${PYTHON} - 2>/dev/null
import sys
import distutils.sysconfig as cfg
import numpy.lib.utils as npyutil
sys.stdout.write( '-I' + cfg.get_python_inc())
sys.stdout.write(' -I' + cfg.get_python_inc(plat_specific=1))
sys.stdout.write(' -I' + npyutil.get_include())
EOD`]
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not determine Python preprocessor flags])
    ])
    for flag in ${python_out}; do
      SWIG_PYTHON_CPPFLAGS="${SWIG_PYTHON_CPPFLAGS} ${flag}"
    done

    # determine Python compiler flags
    AC_SUBST([SWIG_PYTHON_CFLAGS],[])
    python_out=[`cat <<EOD | ${PYTHON} - 2>/dev/null
import sys
import distutils.sysconfig as cfg
cflags = cfg.get_config_var('CFLAGS').split()
cflags = [f for f in cflags if f != '-DNDEBUG']
sys.stdout.write(" ".join(cflags))
EOD`]
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not determine Python compiler flags])
    ])
    for flag in ${python_out}; do
      swig_python_cflags="${swig_python_cflags} ${flag}"
    done
    LALSUITE_CHECK_COMPILE_FLAGS([
      ${swig_python_cflags}
      -Wno-uninitialized
      -Wno-unused-variable
      -Wno-unused-but-set-variable
      -Wno-format-extra-args
      -Wno-tautological-compare
      -fno-strict-aliasing
      ],[SWIG_PYTHON_CFLAGS="${SWIG_PYTHON_CFLAGS} ${flag}"]
    )

    # determine Python linker flags
    AC_SUBST([SWIG_PYTHON_LDFLAGS],[])
    python_out=[`cat <<EOD | ${PYTHON} - 2>/dev/null
import sys, os
import distutils.sysconfig as cfg
sys.stdout.write(cfg.get_config_var('LDFLAGS'))
sys.stdout.write(' -L' + cfg.get_python_lib())
sys.stdout.write(' -L' + cfg.get_python_lib(plat_specific=1))
sys.stdout.write(' -L' + cfg.get_python_lib(plat_specific=1,standard_lib=1))
sys.stdout.write(' -L' + cfg.get_config_var('LIBDIR'))
EOD`]
    AS_IF([test $? -ne 0],[
      AC_MSG_ERROR([could not determine Python linker flags])
    ])
    swig_python_ldflags=
    for flag in ${python_out}; do
      AS_CASE([${flag}],
        [-L/usr/lib|-L/usr/lib64],[:],
        [-Xlinker],[swig_python_ldflags="${swig_python_ldflags}-Wl,"],
        [swig_python_ldflags="${swig_python_ldflags}${flag} "]
      )
    done
    LALSUITE_CHECK_LINK_FLAGS([
      ${swig_python_ldflags}
      ],[SWIG_PYTHON_LDFLAGS="${SWIG_PYTHON_LDFLAGS} ${flag}"]
    )

    # check for Python and NumPy headers
    LALSUITE_PUSH_UVARS
    CPPFLAGS="${SWIG_PYTHON_CPPFLAGS}"
    AC_CHECK_HEADER([Python.h],[],[
      AC_MSG_ERROR([could not find the header "Python.h"])
    ],[
      AC_INCLUDES_DEFAULT
    ])
    AC_CHECK_HEADER([numpy/arrayobject.h],[],[
      AC_MSG_ERROR([could not find the header "numpy/arrayobject.h"])
    ],[
      AC_INCLUDES_DEFAULT
      #include <Python.h>
    ])
    LALSUITE_POP_UVARS

    # remove deprecated code in NumPy API >= 1.7
    SWIG_PYTHON_CPPFLAGS="${SWIG_PYTHON_CPPFLAGS} -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"

    # check for declarations which may need compatibility code for NumPy API < 1.7
    LALSUITE_PUSH_UVARS
    CPPFLAGS="${SWIG_PYTHON_CPPFLAGS}"
    AC_CHECK_DECL([NPY_ARRAY_WRITEABLE],[
      SWIG_PYTHON_CPPFLAGS="${SWIG_PYTHON_CPPFLAGS} -DSWIGLAL_HAVE_NPY_ARRAY_WRITEABLE"
    ],[],[
      AC_INCLUDES_DEFAULT
      #include <Python.h>
      #include <numpy/arrayobject.h>
    ])
    AC_CHECK_DECL([PyArray_SetBaseObject],[
      SWIG_PYTHON_CPPFLAGS="${SWIG_PYTHON_CPPFLAGS} -DSWIGLAL_HAVE_PyArray_SetBaseObject"
    ],[],[
      AC_INCLUDES_DEFAULT
      #include <Python.h>
      #include <numpy/arrayobject.h>
    ])
    LALSUITE_POP_UVARS

  ],[

    # determine SWIG Python flags
    AC_SUBST([SWIG_PYTHON_FLAGS],["-O -builtin -globals globalvar"])
    AC_MSG_CHECKING([if SWIG supports relative Python imports])
    LALSUITE_VERSION_COMPARE([${swig_version}],[<],[3.0.0],[
      AC_MSG_RESULT([no])
    ],[
      AC_MSG_RESULT([yes])
      SWIG_PYTHON_FLAGS="-py3 -relativeimport ${SWIG_PYTHON_FLAGS}"
    ])

  ])
  # end $0
])

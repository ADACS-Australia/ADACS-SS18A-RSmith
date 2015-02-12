# -*- mode: autoconf; -*-
# lalsuite_build.m4 - top level build macros
#
# serial 105

# not present in older versions of pkg.m4
m4_pattern_allow([^PKG_CONFIG(_(PATH|LIBDIR|SYSROOT_DIR|ALLOW_SYSTEM_(CFLAGS|LIBS)))?$])
m4_pattern_allow([^PKG_CONFIG_(DISABLE_UNINSTALLED|TOP_BUILD_DIR|DEBUG_SPEW)$])

# forbid LALSUITE_... from appearing in output (./configure)
#m4_pattern_forbid([^_?LALSUITE_[A-Z_]+$])

# list of user variables; see section 4.8.1 of the Autoconf manual
m4_define([uvar_list],[CPPFLAGS CFLAGS CXXFLAGS FCFLAGS FFLAGS LDFLAGS])
# prefix used to save/restore user variables in
m4_define([uvar_orig_prefix],[lalsuite_uvar_])
m4_define([uvar_prefix],uvar_orig_prefix)

AC_DEFUN([LALSUITE_ARG_VAR],[
  AC_ARG_VAR(LAL_DATA_PATH,[Location of LAL data files])
  AC_ARG_VAR(LAL_OCTAVE_PATH,[Location of LAL octave files])
  AC_ARG_VAR(LAL_PYTHON_PATH,[Location of LAL python files])
  AC_ARG_VAR(LALSUITE_BUILD,[Set if part of lalsuite build])
])

m4_append([AC_INIT],[
  # just after AC_INIT:
  # save user-supplied values of user variables
  m4_foreach_w([uvar],uvar_list,[
    uvar_prefix[]uvar="${uvar}"
  ])
  m4_pushdef([uvar_prefix],uvar_prefix[]p_)
])

m4_rename([AC_OUTPUT],[lalsuite_AC_OUTPUT])
AC_DEFUN([AC_OUTPUT],[
  # just before AC_OUTPUT:
  # check for unbalanced LALSUITE_{PUSH,POP}_UVAR pairs
  m4_popdef([uvar_prefix])
  m4_if(uvar_prefix,uvar_orig_prefix,[],[
    m4_fatal([unbalanced LALSUITE_{PUSH,POP}_UVAR pairs])
  ])
  # prepend compiler configuration e.g. CFLAGS to AM_CFLAGS,
  # then restore original user-supplied values of user variables
  m4_foreach_w([uvar],uvar_list,[
    AC_SUBST(AM_[]uvar,"${AM_[]uvar} ${sys_[]uvar}")
    uvar="${uvar_prefix[]uvar}"
  ])
  # call original AC_OUTPUT
  lalsuite_AC_OUTPUT
])

AC_DEFUN([LALSUITE_PUSH_UVARS],[
  # $0: save current values of user variables and LIBS
  m4_foreach_w([uvar],uvar_list[ LIBS],[
    uvar_prefix[]uvar="${uvar}"
    uvar_prefix[]uvar[]_lineno=$LINENO; _AS_ECHO_LOG([pushed uvar=${uvar}])
  ])
  m4_pushdef([uvar_prefix],uvar_prefix[]p_)
  # end $0
])

AC_DEFUN([LALSUITE_CLEAR_UVARS],[
  # $0: clear current values of user variables and LIBS
  m4_foreach_w([uvar],uvar_list[ LIBS],[
    AS_UNSET(uvar)
  ])
  # end $0
])

AC_DEFUN([LALSUITE_POP_UVARS],[
  # $0: restore previous values of user variables and LIBS
  m4_popdef([uvar_prefix])
  m4_foreach_w([uvar],uvar_list[ LIBS],[
    uvar="${uvar_prefix[]uvar}"
   _AS_ECHO_LOG([popped uvar from line ${uvar_prefix[]uvar[]_lineno}])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_ADD_FLAGS],[
  # $0: prepend flags to AM_CPPFLAGS/AM_$1FLAGS/AM_LDFLAGS/LIBS,
  # and update values of CPPFLAGS/$1FLAGS/LDFLAGS for Autoconf tests
  # - arg 1: prefix of the compiler flag variable, e.g. C for CFLAGS, CXX for CXXFLAGS
  # - arg 2: compiler flags
  # - arg 3: linker flags
  m4_ifval([$1],[m4_ifval([$2],[
    pre_AM_CPPFLAGS=
    pre_sys_CPPFLAGS=
    pre_AM_$1FLAGS=
    for flag in $2; do
      # AM_CPPFLAGS gets unique -I, -D and -U flags
      # sys_CPPFLAGS gets unique system -I flags
      # AM_$1FLAGS gets everything else
      AS_CASE([${flag}],
        [-I/opt/*|-I/usr/*],[
          AS_CASE([" ${sys_CPPFLAGS} "],
            [*" ${flag} "*],[:],
            [pre_sys_CPPFLAGS="${pre_sys_CPPFLAGS} ${flag}"]
          )
        ],
        [-I*],[
          AS_CASE([" ${AM_CPPFLAGS} "],
            [*" ${flag} "*],[:],
            [pre_AM_CPPFLAGS="${pre_AM_CPPFLAGS} ${flag}"]
          )
        ],
        [-D*|-U*],[pre_AM_CPPFLAGS="${pre_AM_CPPFLAGS} ${flag}"],
        [pre_AM_$1FLAGS="${pre_AM_$1FLAGS} ${flag}"]
      )
    done
    AS_IF([test "x${pre_AM_CPPFLAGS}" != x],[
      AM_CPPFLAGS="${pre_AM_CPPFLAGS} ${AM_CPPFLAGS}"
      _AS_ECHO_LOG([prepended ${pre_AM_CPPFLAGS} to AM_CPPFLAGS])
    ])
    AS_IF([test "x${pre_sys_CPPFLAGS}" != x],[
      sys_CPPFLAGS="${pre_sys_CPPFLAGS} ${sys_CPPFLAGS}"
      _AS_ECHO_LOG([prepended ${pre_sys_CPPFLAGS} to system AM_CPPFLAGS])
    ])
    AS_IF([test "x${pre_AM_$1FLAGS}" != x],[
      AM_$1FLAGS="${pre_AM_$1FLAGS} ${AM_$1FLAGS}"
      _AS_ECHO_LOG([prepended ${pre_AM_$1FLAGS} to AM_$1FLAGS])
    ])
    CPPFLAGS="${AM_CPPFLAGS} ${sys_CPPFLAGS} ${uvar_orig_prefix[]CPPFLAGS}"
    $1FLAGS="${AM_$1FLAGS} ${uvar_orig_prefix[]$1FLAGS}"
  ])])
  m4_ifval([$3],[
    pre_AM_LDFLAGS=
    pre_sys_LDFLAGS=
    pre_LIBS=
    for flag in $3; do
      # LIBS gets -l flags and .la files
      # sys_LDFLAGS gets unique system -L flags
      # AM_LDFLAGS gets unique -L flags and everything else
      AS_CASE([${flag}],
        [-L/opt/*|-L/usr/*],[
          AS_CASE([" ${sys_LDFLAGS} "],
            [*" ${flag} "*],[:],
            [pre_sys_LDFLAGS="${pre_sys_LDFLAGS} ${flag}"]
          )
        ],
        [-L*],[
          AS_CASE([" ${AM_LDFLAGS} "],
            [*" ${flag} "*],[:],
            [pre_AM_LDFLAGS="${pre_AM_LDFLAGS} ${flag}"]
          )
        ],
        [-l*|*.la],[pre_LIBS="${pre_LIBS} ${flag}"],
        [pre_AM_LDFLAGS="${pre_AM_LDFLAGS} ${flag}"]
      )
    done
    AS_IF([test "x${pre_AM_LDFLAGS}" != x],[
      AM_LDFLAGS="${pre_AM_LDFLAGS} ${AM_LDFLAGS}"
      _AS_ECHO_LOG([prepended ${pre_AM_LDFLAGS} to AM_LDFLAGS])
    ])
    AS_IF([test "x${pre_sys_LDFLAGS}" != x],[
      sys_LDFLAGS="${pre_sys_LDFLAGS} ${sys_LDFLAGS}"
      _AS_ECHO_LOG([prepended ${pre_sys_LDFLAGS} to system AM_LDFLAGS])
    ])
    AS_IF([test "x${pre_LIBS}" != x],[
      LIBS="${pre_LIBS} ${LIBS}"
      _AS_ECHO_LOG([prepended ${pre_LIBS} to LIBS])
    ])
    LDFLAGS="${AM_LDFLAGS} ${sys_LDFLAGS} ${uvar_orig_prefix[]LDFLAGS}"
  ])
  # end $0
])

AC_DEFUN([LALSUITE_ADD_PATH],[
  # $0: prepend path to $1, removing duplicates, first value taking precedence
  # - arg 1: name of path variable
  # - arg 2: path to prepend
  # - arg 3: whether to exclude /opt/... and /usr/... directories (default: yes)
  tokens=$2
  tokens=`echo ${tokens} ${$1} | sed 's/:/ /g'`
  $1=
  for token in ${tokens}; do
    AS_CASE([m4_default([$3],[yes]):${token}],
      [yes:/opt/*|yes:/usr/*],[:],
      AS_CASE([":${$1}:"],
        [*:${token}:*],[:],
        AS_IF([test "x${$1}" = x],[
          $1="${token}"
        ],[
          $1="${$1}:${token}"
        ])
      )
    )
  done
  _AS_ECHO_LOG([$1=${$1}])
  # end $0
])

AC_DEFUN([LALSUITE_VERSION_COMPARE],[
  # $0: compare versions using the given operator
  m4_case([$2],[<],,[<=],,[=],,[>=],,[>],,[m4_fatal([$0: invalid operator $2])])
  AS_VERSION_COMPARE([$1],[$3],[lalsuite_op='<'],[lalsuite_op='='],[lalsuite_op='>'])
  AS_CASE(['$2'],[*${lalsuite_op}*],[m4_default([$4],[:])],[m4_default([$5],[:])])
  # end $0
])

AC_DEFUN([LALSUITE_CHECK_GIT_REPO],[
  # $0: check for git
  AC_PATH_PROGS([GIT],[git],[false])
  # check whether building from a git repository
  have_git_repo=no
  AS_IF([test "x${GIT}" != xfalse],[
    AC_MSG_CHECKING([whether building from a git repository])
    # git log will print:
    # * the last log message, if the cwd is in a git repository
    # * nothing, if the cwd is not part of the git repo (e.g. ignored)
    # * an error msg to stderr if the cwd is not in a git repository
    git_log=`( cd "${srcdir}" && ${GIT} log --pretty=oneline -n 1 -- . ) 2>/dev/null`
    AS_IF([test "x${git_log}" != x],[have_git_repo=yes])
    AC_MSG_RESULT([${have_git_repo}])
  ])
  # conditional for git and building from a git repository
  AM_CONDITIONAL([HAVE_GIT_REPO],[test "x${have_git_repo}" = xyes])
  # end $0
])

AC_DEFUN([LALSUITE_VERSION_CONFIGURE_INFO],[
  # $0: define version/configure info
  m4_pushdef([uppercase],m4_translit(AC_PACKAGE_NAME, [a-z], [A-Z]))
  m4_pushdef([lowercase],m4_translit(AC_PACKAGE_NAME, [A-Z], [a-z]))
  m4_pushdef([withoutlal],m4_bpatsubst(AC_PACKAGE_NAME, [^LAL], []))
  version_major=`echo "$VERSION" | cut -d. -f1`
  version_minor=`echo "$VERSION" | cut -d. -f2`
  version_micro=`echo "$VERSION" | cut -d. -f3`
  version_devel=`echo "$VERSION" | cut -d. -f4-`
  test -z "$version_micro" && version_micro=0
  test -z "$version_devel" && version_devel=0
  configure_date=`date +"%Y-%m-%dT%H:%M:%S%z"`
  AC_DEFINE_UNQUOTED(uppercase[_VERSION],["$VERSION"],AC_PACKAGE_NAME[ Version])
  AC_DEFINE_UNQUOTED(uppercase[_VERSION_MAJOR],[$version_major],AC_PACKAGE_NAME[ Version Major Number])
  AC_DEFINE_UNQUOTED(uppercase[_VERSION_MINOR],[$version_minor],AC_PACKAGE_NAME[ Version Minor Number])
  AC_DEFINE_UNQUOTED(uppercase[_VERSION_MICRO],[$version_micro],AC_PACKAGE_NAME[ Version Micro Number])
  AC_DEFINE_UNQUOTED(uppercase[_VERSION_DEVEL],[$version_devel],AC_PACKAGE_NAME[ Version Devel Number])
  AC_SUBST([ac_configure_args])
  AC_SUBST([configure_date])
  AC_SUBST([PACKAGE_NAME_UCASE],uppercase)
  AC_SUBST([PACKAGE_NAME_LCASE],lowercase)
  AC_SUBST([PACKAGE_NAME_NOLAL],withoutlal)
  m4_popdef([uppercase])
  m4_popdef([lowercase])
  m4_popdef([withoutlal])
  # end $0
])

AC_DEFUN([LALSUITE_REQUIRE_CXX],[
  # $0: require a C++ compiler
  lalsuite_require_cxx=true
  # end $0
])

AC_DEFUN([LALSUITE_REQUIRE_F77],[
  # $0: require an F77 compiler
  lalsuite_require_f77=true
  # end $0
])

# because we want to conditionally decide whether to check for
# C++/Fortran compilers only at ./configure run time, we must erase the
# following macros; in Autoconf 2.64 and later, they AC_REQUIRE the
# C++/Fortran AC_PROG_... macros, which forces the C++/Fortran compilers
# to always be checked for, which prevents us from instead conditionally
# deciding that at ./configure run time
m4_foreach([lang],[[C++],[Fortran 77],[Fortran]],[
  m4_defun([AC_LANG_COMPILER(]lang[)],[])
  m4_defun([AC_LANG_PREPROC(]lang[)],[])
])

AC_DEFUN([_LALSUITE_PRE_PROG_COMPILERS],[
  # $0: just before LALSUITE_PROG_COMPILERS:
  # save current values of user variables, then unset them
  LALSUITE_PUSH_UVARS
  LALSUITE_CLEAR_UVARS
  # end $0
])

AC_DEFUN([_LALSUITE_POST_PROG_COMPILERS],[
  # $0: just after LALSUITE_PROG_COMPILERS:
  # save values of user variables set during compiler configuration,
  # restore previous values of user variables, then add compiler values
  # of user variables to then using LALSUITE_ADD_FLAGS
  m4_foreach_w([uvar],uvar_list,[
    lalsuite_compiler_[]uvar="${uvar}"
  ])
  LALSUITE_POP_UVARS
  LALSUITE_ADD_FLAGS([C],[${lalsuite_compiler_CPPFLAGS} ${lalsuite_compiler_CFLAGS}],[${lalsuite_compiler_LDFLAGS}])
  AS_IF([test "${lalsuite_require_cxx}" = true],[
    LALSUITE_ADD_FLAGS([CXX],[${lalsuite_compiler_CXXFLAGS}],[])
  ])
  AS_IF([test "${lalsuite_require_f77}" = true],[
    LALSUITE_ADD_FLAGS([FC],[${lalsuite_compiler_FCFLAGS}],[])
    LALSUITE_ADD_FLAGS([F],[${lalsuite_compiler_FFLAGS}],[])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_PROG_COMPILERS],[
  # $0: check for C/C++/Fortran compilers
  AC_REQUIRE([_LALSUITE_PRE_PROG_COMPILERS])

  # check for C99 compiler
  AC_REQUIRE([AC_PROG_CC])
  AC_REQUIRE([AC_PROG_CC_C99])
  AC_REQUIRE([AC_PROG_CPP])

  # check for clang
  AS_IF([test "x$GCC" = xyes],
    [AS_IF([test "`$CC -v 2>&1 | grep -c 'clang'`" != "0"],[CLANG_CC=1])],
    [CLANG_CC=])
  AC_SUBST(CLANG_CC)

  # check for C++ compiler, if needed
  AS_IF([test "${lalsuite_require_cxx}" = true],[
    AC_PROG_CXX
    AC_PROG_CXXCPP

    # check for clang++
    AS_IF([test "x$GXX" = xyes],
      [AS_IF([test "`$CXX -v 2>&1 | grep -c 'clang'`" != "0"],[CLANG_CXX=1])],
      [CLANG_CXX=])
    AC_SUBST(CLANG_CXX)

    # define C99 constant and limit macros for C++ sources
    CXXFLAGS="${CXXFLAGS} -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS"

  ],[
    CXX=
    CXXCPP=
    AM_CONDITIONAL([am__fastdepCXX],[false])
  ])

  # check for F77 compiler, if needed
  AS_IF([test "${lalsuite_require_f77}" = true],[
    AC_PROG_F77
  ],[
    F77=
  ])

  _LALSUITE_POST_PROG_COMPILERS
  # end $0
])

AC_DEFUN([LALSUITE_REQUIRE_PYTHON],[
  # $0: require Python version $1 or later
  AS_IF([test "x${lalsuite_require_pyvers}" = x],[
    lalsuite_require_pyvers="$1"
  ],[
    LALSUITE_VERSION_COMPARE([$1],[>],[${lalsuite_require_pyvers}],[
      lalsuite_require_pyvers="$1"
    ])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_CHECK_PYTHON],[
  # $0: check for Python
  lalsuite_pyvers="$1"
  AS_IF([test "x${lalsuite_require_pyvers}" != x],[
    LALSUITE_VERSION_COMPARE([${lalsuite_require_pyvers}],[>],[${lalsuite_pyvers}],[
      lalsuite_pyvers="${lalsuite_require_pyvers}"
    ])
  ])
  AS_IF([test "x${PYTHON}" != xfalse],[
    AM_PATH_PYTHON([${lalsuite_pyvers}],,[
      AS_IF([test "x${lalsuite_require_pyvers}" = x],[
        PYTHON=false
      ],[
        AC_MSG_ERROR([Python version ${lalsuite_pyvers} or later is required])
      ])
    ])
  ])
  AM_CONDITIONAL([HAVE_PYTHON],[test "x${PYTHON}" != xfalse])
  AM_COND_IF([HAVE_PYTHON],[
    AC_SUBST([python_prefix], [`${PYTHON} -c 'import sys; print(sys.prefix)' 2>/dev/null`])
    AC_SUBST([python_exec_prefix], [`${PYTHON} -c 'import sys; print(sys.exec_prefix)' 2>/dev/null`])
    PYTHON_ENABLE_VAL=ENABLED
  ],[
    PYTHON_ENABLE_VAL=DISABLED
  ])
  # end $0
])

AC_DEFUN([LALSUITE_USE_LIBTOOL],[
  # $0: Generate a libtool script for use in configure tests
  AC_REQUIRE([LT_INIT])
  LT_OUTPUT
  m4_append([AC_LANG(C)],[
    ac_link="./libtool --mode=link --tag=CC $ac_link"
  ])
  AC_PROVIDE_IFELSE([AC_PROG_CXX],[
    m4_append([AC_LANG(C++)],[
      ac_link="./libtool --mode=link --tag=CXX $ac_link"
    ])
  ])
  AC_LANG(_AC_LANG)
  # end $0
])

AC_DEFUN([LALSUITE_MULTILIB_LIBTOOL_HACK],[
  # $0: libtool incorrectly determine library path on SL6
  case "${host}" in
    x86_64-*-linux-gnu*)
      case `cat /etc/redhat-release 2> /dev/null` in
        "Scientific Linux"*|"CentOS"*)
          AC_MSG_NOTICE([hacking round broken libtool multilib support on RedHat systems])
          lt_cv_sys_lib_dlsearch_path_spec="/lib64 /usr/lib64"
          ;;
      esac
      ;;
  esac
  # end $0
])

AC_DEFUN([LALSUITE_DISTCHECK_CONFIGURE_FLAGS],[
  # $0: store configure flags for 'make distcheck'
  DISTCHECK_CONFIGURE_FLAGS=
  for arg in ${ac_configure_args}; do
    case ${arg} in
      (\'--enable-*\'|\'--disable-*\')
        # save any --enable/--disable arguments
        DISTCHECK_CONFIGURE_FLAGS="${DISTCHECK_CONFIGURE_FLAGS} ${arg}";;
      (\'--with-*\'|\'--without-*\')
        # save any --with/--without arguments
        DISTCHECK_CONFIGURE_FLAGS="${DISTCHECK_CONFIGURE_FLAGS} ${arg}";;
      (\'--*\')
        # skip all other ./configure arguments
        : ;;
      (\'DISTCHECK_CONFIGURE_FLAGS=*\')
        # append value of DISTCHECK_CONFIGURE_FLAGS
        DISTCHECK_CONFIGURE_FLAGS="${DISTCHECK_CONFIGURE_FLAGS} "`expr "X${arg}" : "X'DISTCHECK_CONFIGURE_FLAGS=\(.*\)'"`;;
      (\'*=*\')
        # save any environment variables given to ./configure
        DISTCHECK_CONFIGURE_FLAGS="${DISTCHECK_CONFIGURE_FLAGS} ${arg}";;
    esac
  done
  AC_SUBST(DISTCHECK_CONFIGURE_FLAGS)
  # end $0
])

AC_DEFUN([LALSUITE_ENABLE_MODULE],[
  # $0: enable/disable module $1
  m4_pushdef([lowercase],m4_translit([[$1]], [A-Z], [a-z]))
  m4_pushdef([uppercase],m4_translit([[$1]], [a-z], [A-Z]))
  AM_CONDITIONAL(uppercase,[test "x${lowercase}" = xtrue])
  AS_IF([test "${lowercase}" = "true"],[
    uppercase[]_ENABLE_VAL=ENABLED
  ],[
    uppercase[]_ENABLE_VAL=DISABLED
  ])
  _AS_ECHO_LOG([module $1 is ${]uppercase[_ENABLE_VAL}])
  m4_popdef([lowercase])
  m4_popdef([uppercase])
  # end $0
])

AC_DEFUN([LALSUITE_CHECK_LIB],[
  # $0: check for LAL library
  # - arg 1: name of LAL library
  # - arg 2: minimum version required
  # - arg 3: library function to check for
  # - arg 4: library header to check for
  m4_pushdef([lowercase],m4_translit([[$1]], [A-Z], [a-z]))
  m4_pushdef([uppercase],m4_translit([[$1]], [a-z], [A-Z]))

  # substitute required library version in pkg-config files
  AC_SUBST(uppercase[]_VERSION,[$2])

  # set up pkg-config environment
  AS_UNSET([PKG_CONFIG_ALLOW_SYSTEM_CFLAGS])
  AS_UNSET([PKG_CONFIG_ALLOW_SYSTEM_LIBS])

  # prepend to CFLAGS, CPPFLAGS, LDFLAGS, LIBS, LAL_DATA_PATH, LAL_OCTAVE_PATH, LAL_PYTHON_PATH, LAL_HTMLDIR
  PKG_CHECK_MODULES(uppercase, [lowercase >= $2], [lowercase="true"], [lowercase="false"])
  PKG_CHECK_VAR(uppercase[]_DATA_PATH, [lowercase >= $2], uppercase[]_DATA_PATH,,)
  PKG_CHECK_VAR(uppercase[]_OCTAVE_PATH, [lowercase >= $2], uppercase[]_OCTAVE_PATH,,)
  PKG_CHECK_VAR(uppercase[]_PYTHON_PATH, [lowercase >= $2], uppercase[]_PYTHON_PATH,,)
  PKG_CHECK_VAR(uppercase[]_HTMLDIR, [lowercase >= $2], htmldir,,)
  if test "$lowercase" = "true"; then
    LALSUITE_ADD_FLAGS([C],$[]uppercase[]_CFLAGS,$[]uppercase[]_LIBS)
    LALSUITE_ADD_PATH(LAL_DATA_PATH,"$[]uppercase[]_DATA_PATH")
    LALSUITE_ADD_PATH(LAL_OCTAVE_PATH,"$[]uppercase[]_OCTAVE_PATH")
    LALSUITE_ADD_PATH(LAL_PYTHON_PATH,"$[]uppercase[]_PYTHON_PATH")
  fi

  # add system include flags to LAL_SYSTEM_INCLUDES
  if test -n "$PKG_CONFIG" -a "$LALSUITE_BUILD" != "true"; then
    # use pkg-config to get system paths
    PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1
    export PKG_CONFIG_ALLOW_SYSTEM_CFLAGS
    for flag in `$PKG_CONFIG --cflags-only-I "lowercase >= $2"`; do
      AS_CASE([" $CPPFLAGS $LAL_SYSTEM_INCLUDES "],
        [*" ${flag} "*],[:],
        [LAL_SYSTEM_INCLUDES="$LAL_SYSTEM_INCLUDES $flag"]
      )
    done
    AS_UNSET([PKG_CONFIG_ALLOW_SYSTEM_CFLAGS])
  else
    # use standard include paths
    save_IFS="$IFS"
    IFS=:
    for flag in "$C_INCLUDE_PATH:$CPLUS_INCLUDE_PATH:/usr/include" ; do
      test -n "$flag" && flag="-I$flag"
      AS_CASE([" $CPPFLAGS $LAL_SYSTEM_INCLUDES "],
        [*" ${flag} "*],[:],
        [LAL_SYSTEM_INCLUDES="$LAL_SYSTEM_INCLUDES $flag"]
      )
    done
    IFS="$save_IFS"
  fi
  AC_SUBST([LAL_SYSTEM_INCLUDES])

  if test "$LALSUITE_BUILD" = "true"; then
    if test "$lowercase" = "false"; then
      # should never get here: bug in build system
      AC_MSG_ERROR([could not find the $1 library])
    fi
  else
    AC_CHECK_LIB(lowercase,[$3],,[AC_MSG_ERROR([could not find the $1 library])])
    AC_CHECK_HEADERS([$4],,[AC_MSG_ERROR([could not find the $4 header])])
    if test "$1" != "LALSupport"; then
      LALSUITE_HEADER_LIBRARY_MISMATCH_CHECK([$1])
    fi
  fi
  AC_DEFINE([HAVE_LIB]uppercase,[1],[Define to 1 if you have the $1 library])
  # add $1 to list of LALSuite libraries
  lalsuite_libs="${lalsuite_libs} lowercase"
  lowercase="true"
  LALSUITE_ENABLE_MODULE($1)
  m4_popdef([lowercase])
  m4_popdef([uppercase])
  # end $0
])

AC_DEFUN([LALSUITE_CHECK_OPT_LIB],[
  # $0: check for optional LAL library
  # - arg 1: name of LAL library
  # - arg 2: minimum version required
  # - arg 3: library function to check for
  # - arg 4: library header to check for
  m4_pushdef([lowercase],m4_translit([[$1]], [A-Z], [a-z]))
  m4_pushdef([uppercase],m4_translit([[$1]], [a-z], [A-Z]))

  # optional check for $1
  AS_IF([test "${lowercase}" = "true"],[
    LALSUITE_CHECK_LIB($1,$2,$3,$4)
  ])

  # enable/disable $1
  LALSUITE_ENABLE_MODULE($1)

  m4_popdef([lowercase])
  m4_popdef([uppercase])
  # end $0
])

AC_DEFUN([LALSUITE_HEADER_LIBRARY_MISMATCH_CHECK],[
  # $0: check for version mismatch between library $1 and its headers
  AC_MSG_CHECKING([whether $1 headers match the library])
  lib_structure=`echo $1 | sed 's/LAL/lal/'`VCSInfo
  header_structure=`echo $1 | sed 's/LAL/lal/'`VCSInfoHeader
  AC_RUN_IFELSE([
    AC_LANG_SOURCE([[
#include <string.h>
#include <stdlib.h>
#include <lal/$1VCSInfoHeader.h>
int main(void) { exit(XLALVCSInfoCompare(&$lib_structure, &$header_structure) ? 1 : 0); }
    ]])
  ],[
    AC_MSG_RESULT(yes)
  ],[
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([Your $1 headers do not match your library. Check config.log for details.])
  ],[
    AC_MSG_WARN([cross compiling: not checking])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_CHECK_LIBRARY_FOR_SUPPORT],[
  # $0: check if library $1 supports $2
  m4_pushdef([uppercase],m4_translit([[$1]], [a-z], [A-Z]))
  LALSUITE_PUSH_UVARS
  # because library may be uninstalled, remove everything but -I flags
  save_CPPFLAGS="${CPPFLAGS}"
  LALSUITE_CLEAR_UVARS
  CPPFLAGS="${save_CPPFLAGS}"
  AC_MSG_CHECKING([whether $1 has been compiled with $2 support])
  AC_COMPILE_IFELSE([
    AC_LANG_SOURCE([[
#include <lal/$1Config.h>
#ifndef ]uppercase[_$2_ENABLED
#error ]uppercase[_$2_ENABLED is not defined
#endif
    ]])
  ],[
    AC_MSG_RESULT([yes])
    m4_default([$3],[:])
  ],[
    AC_MSG_RESULT([no])
    m4_default([$4],[:])
  ])
  LALSUITE_POP_UVARS
  m4_popdef([uppercase])
  # end $0
])

AC_DEFUN([LALSUITE_ENABLE_NIGHTLY],
[AC_ARG_ENABLE(
  [nightly],
  AC_HELP_STRING([--enable-nightly],[nightly build [default=no]]),
  [ case "${enableval}" in
      yes) NIGHTLY_VERSION=`date -u +"%Y%m%d"`
           VERSION="${VERSION}.${NIGHTLY_VERSION}" ;;
      no) NIGHTLY_VERSION="";;
      *) NIGHTLY_VERSION="${enableval}"
         VERSION="${VERSION}.${NIGHTLY_VERSION}" ;;
      esac ],
  [ NIGHTLY_VERSION="" ] )
  AC_SUBST(NIGHTLY_VERSION)
])

AC_DEFUN([LALSUITE_ENABLE_ALL_LAL],
[AC_ARG_ENABLE(
  [all_lal],
  AC_HELP_STRING([--enable-all-lal],[enable/disable compilation of all LAL libraries]),
  [ case "${enableval}" in
      yes) all_lal=true;;
      no) all_lal=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-all-lal) ;;
    esac
  ], [ all_lal= ] )
])

AC_DEFUN([LALSUITE_ENABLE_LALFRAME],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalframe],
  AC_HELP_STRING([--enable-lalframe],[compile code that requires lalframe library [default=yes]]),
  [ case "${enableval}" in
      yes) lalframe=true;;
      no) lalframe=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalframe) ;;
    esac
  ], [ lalframe=${all_lal:-true} ] )
if test "$frame" = "false"; then
  lalframe=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALMETAIO],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalmetaio],
  AC_HELP_STRING([--enable-lalmetaio],[compile code that requires lalmetaio library [default=yes]]),
  [ case "${enableval}" in
      yes) lalmetaio=true;;
      no) lalmetaio=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalmetaio) ;;
    esac
  ], [ lalmetaio=${all_lal:-true} ] )
if test "$metaio" = "false"; then
  lalmetaio=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALXML],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalxml],
  AC_HELP_STRING([--enable-lalxml],[compile code that requires lalxml library [default=yes]]),
  [ case "${enableval}" in
      yes) lalxml=true;;
      no) lalxml=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalxml) ;;
    esac
  ], [ lalxml=${all_lal:-true} ] )
])

AC_DEFUN([LALSUITE_ENABLE_LALSIMULATION],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalsimulation],
  AC_HELP_STRING([--enable-lalsimulation],[compile code that requires lalsimulation library [default=yes]]),
  [ case "${enableval}" in
      yes) lalsimulation=true;;
      no) lalsimulation=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalsimulation) ;;
    esac
  ], [ lalsimulation=${all_lal:-true} ] )
])

AC_DEFUN([LALSUITE_ENABLE_LALBURST],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalburst],
  AC_HELP_STRING([--enable-lalburst],[compile code that requires lalburst library [default=yes]]),
  [ case "${enableval}" in
      yes) lalburst=true;;
      no) lalburst=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalburst) ;;
    esac
  ], [ lalburst=${all_lal:-true} ] )
if test "$lalmetaio" = "false"; then
  lalburst=false
fi
if test "$lalsimulation" = "false"; then
  lalburst=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALDETCHAR],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [laldetchar],
  AC_HELP_STRING([--enable-laldetchar],[compile code that requires laldetchar library [default=yes]]),
  [ case "${enableval}" in
      yes) laldetchar=true;;
      no) laldetchar=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-laldetchar) ;;
    esac
  ], [ laldetchar=${all_lal:-true} ] )
if test "$lalmetaio" = "false"; then
  laldetchar=false
fi
if test "$lalburst" = "false"; then
  laldetchar=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALINSPIRAL],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalinspiral],
  AC_HELP_STRING([--enable-lalinspiral],[compile code that requires lalinspiral library [default=yes]]),
  [ case "${enableval}" in
      yes) lalinspiral=true;;
      no) lalinspiral=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalinspiral) ;;
    esac
  ], [ lalinspiral=${all_lal:-true} ] )
if test "$lalframe" = "false"; then
  lalinspiral=false
fi
if test "$lalmetaio" = "false"; then
  lalinspiral=false
fi
if test "$lalsimulation" = "false"; then
  lalinspiral=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALPULSAR],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalpulsar],
  AC_HELP_STRING([--enable-lalpulsar],[compile code that requires lalpulsar library [default=yes]]),
  [ case "${enableval}" in
      yes) lalpulsar=true;;
      no) lalpulsar=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalpulsar) ;;
    esac
  ], [ lalpulsar=${all_lal:-true} ] )
])

AC_DEFUN([LALSUITE_ENABLE_LALSTOCHASTIC],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalstochastic],
  AC_HELP_STRING([--enable-lalstochastic],[compile code that requires lalstochastic library [default=yes]]),
  [ case "${enableval}" in
      yes) lalstochastic=true;;
      no) lalstochastic=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalstochastic) ;;
    esac
  ], [ lalstochastic=${all_lal:-true} ] )
if test "$lalmetaio" = "false"; then
  lalstochastic=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALINFERENCE],
[AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
AC_ARG_ENABLE(
  [lalinference],
  AC_HELP_STRING([--enable-lalinference],[compile code that requires lalinference library [default=yes]]),
  [ case "${enableval}" in
      yes) lalinference=true;;
      no) lalinference=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalinference) ;;
    esac
  ], [ lalinference=${all_lal:-true} ] )
if test "$lalmetaio" = "false"; then
  lalinference=false
fi
if test "$lalframe" = "false"; then
  lalinference=false
fi
if test "$lalinspiral" = "false"; then
  lalinference=false
fi
if test "$lalpulsar" = "false"; then
  lalinference=false
fi
])

AC_DEFUN([LALSUITE_ENABLE_LALAPPS],[
  AC_REQUIRE([LALSUITE_ENABLE_ALL_LAL])
  AC_ARG_ENABLE(
    [lalapps],
    AC_HELP_STRING([--enable-lalapps],[compile lalapps [default=yes]]),
    [
      case "${enableval}" in
        yes) lalapps=true ;;
        no) lalapps=false ;;
        *) AC_MSG_ERROR(bad value ${enableval} for --enable-lalapps) ;;
      esac
    ],[
      lalapps=${all_lal:-true}
    ]
  )
])

AC_DEFUN([LALSUITE_ENABLE_BOINC],[
  # $0: enable/disable BOINC
  AC_ARG_ENABLE(
    [boinc],
    AC_HELP_STRING([--enable-boinc],[enable BOINC support [default=no]]),
    AS_CASE([${enableval}],
      [yes],[boinc=true],
      [no],[boinc=false],
      AC_MSG_ERROR([bad value '${enableval}' for --enable-boinc])
    ),
    [boinc=false]
  )
  LALSUITE_ENABLE_MODULE([BOINC])
  AS_IF([test "${boinc}" = true],[LALSUITE_REQUIRE_CXX])
  # end $0
])

AC_DEFUN([LALSUITE_CHECK_BOINC],[
  # $0: check for BOINC support
  AS_IF([test "${boinc}" = "true"],[
    LALSUITE_CHECK_LIBRARY_FOR_SUPPORT([LAL],[BOINC],[:],[
      AC_MSG_ERROR([BOINC was enabled but LAL was not compiler with BOINC support])
    ])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_WITH_CUDA],[
AC_ARG_WITH(
  [cuda],
  AC_HELP_STRING([--with-cuda=PATH],[specify location of CUDA [/opt/cuda]]),[
    AS_CASE([${with_cuda}],
      [no],[cuda=false],
      [yes],[cuda=true; cuda_path=/opt/cuda],
      [cuda=true; cuda_path=${with_cuda}]
    )
  ],[
    cuda=false
  ])
  AS_IF([test "${cuda}" = true],[
    LALSUITE_REQUIRE_CXX
    AC_MSG_NOTICE([Using ${with_cuda} as CUDA path])
    AS_CASE([$build_os],
      [linux*],[
        AS_IF([test "x$build_cpu" = "xx86_64"],[
          cuda_libdir=lib64
        ],[
          cuda_libdir=lib
        ])
      ],
      [cuda_libdir=lib]
    )
    CUDA_LIBS="-L${cuda_path}/${cuda_libdir} -Wl,-rpath -Wl,${cuda_path}/${cuda_libdir} -lcufft -lcudart"
    CUDA_CFLAGS="-I${with_cuda}/include"
    LALSUITE_ADD_FLAGS([C],${CUDA_CFLAGS},${CUDA_LIBS})
    AC_SUBST(CUDA_LIBS)
    AC_SUBST(CUDA_CFLAGS)
    AC_PATH_PROGS(NVCC,[nvcc],[],[${cuda_path}/bin:${PATH}])
    AS_IF([test "x${NVCC}" = x],[
      AC_MSG_ERROR([could not find 'nvcc' in path])
    ])
  ])
  LALSUITE_ENABLE_MODULE([CUDA])
])


AC_DEFUN([LALSUITE_CHECK_GSL_VERSION],[
  # $0: check for GSL version
  lal_min_gsl_version=m4_normalize([$1])
  AC_MSG_CHECKING(for GSL version >= $lal_min_gsl_version)
  AC_TRY_RUN([
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_version.h>
int main(void)
{
  int required_major, required_minor;
  int major, minor;
  char required_version[] = "$lal_min_gsl_version";
  char version[] = GSL_VERSION;
  if ( strcmp(GSL_VERSION, gsl_version) ) {
    printf("error\n*** mismatch between header and library versions of GSL\n" );
    printf("\n*** header  has version %s\n", GSL_VERSION);
    printf("\n*** library has version %s\n", gsl_version);
    exit(1);
  }
  sscanf(required_version, "%d.%d", &required_major, &required_minor);
  sscanf(version, "%d.%d", &major, &minor);
  if ( major < required_major || (major == required_major && minor < required_minor) ) {
    printf("no\n*** found version %s of GSL but minimum version is %d.%d\n", GSL_VERSION, required_major, required_minor );
    exit(1);
  }
  return 0;
}
  ],[
    AC_MSG_RESULT([yes])
  ],[
    AC_MSG_ERROR([could not find required version of GSL])
  ],[
    AC_MSG_WARN([cross compiling; assumed OK...])
  ])
  # end $0
])

AC_DEFUN([LALSUITE_ENABLE_FAST_GSL],[
  # $0: enable/disable fast/inline GSL code
  AC_ARG_ENABLE(
    [fast_gsl],
    AC_HELP_STRING([--enable-fast-gsl],[enable fast/inline GSL code [default=no]]),
    AS_CASE(["${enableval}"],
      [yes],[
        AC_DEFINE([HAVE_INLINE],[1],[Define to 1 to use inline code])
        AC_DEFINE([GSL_C99_INLINE],[1],[Define to 1 to use GSL C99 inline code])
        AC_DEFINE([GSL_RANGE_CHECK_OFF],[1],[Define to 1 to turn GSL range checking off])
      ],
      [no],[:],
      AC_MSG_ERROR([bad value ${enableval} for --enable-fast-gsl])
    )
  )
  # end $0
])

AC_DEFUN([LALSUITE_ENABLE_OSX_VERSION_CHECK],
[AC_ARG_ENABLE(
  [osx_version_check],
  AC_HELP_STRING([--enable-osx-version-check],[disable OS X version check [default=yes]]),
  AS_CASE(["${enableval}"],
    [yes],[osx_version_check=true],
    [no],[osx_version_check=false],
    AC_MSG_ERROR([bad value ${enableval} for --enable-osx-version-check])
  ),[osx_version_check=true])
])

AC_DEFUN([LALSUITE_OSX_VERSION_CHECK],[
LALSUITE_ENABLE_OSX_VERSION_CHECK
AS_IF([test "x${osx_version_check}" = "xtrue"],[
  AS_IF([test "x$build_vendor" = "xapple"],[
    AC_CHECK_PROGS([SW_VERS],[sw_vers])
    AS_IF([test "x$SW_VERS" != "x"],[
      AC_MSG_CHECKING([Mac OS X version])
      MACOSX_VERSION=`$SW_VERS -productVersion`
      AC_MSG_RESULT([$MACOSX_VERSION])])
    AS_CASE(["$MACOSX_VERSION"],
      [10.0*|10.1|10.1.*|10.2*|10.3*],AC_MSG_ERROR([This version of Mac OS X is not supported]),
      [10.4*|10.5*|10.6*|10.7*|10.8*|10.9*|10.10*],,
      AC_MSG_WARN([Unknown Mac OS X version]))
])])])

AC_DEFUN([LALSUITE_WITH_NVCC_CFLAGS],
[AC_ARG_WITH(
  [nvcc_cflags],
  AC_HELP_STRING([--with-nvcc-cflags=NVCC_CFLAGS],[NVCC compiler flags]),
  AS_IF([test -n "${with_nvcc_cflags}"],[NVCC_CFLAGS="$NVCC_CFLAGS ${with_nvcc_cflags}"]),)
])

AC_DEFUN([LALSUITE_CHECK_CUDA],
[AC_MSG_CHECKING([whether LAL has been compiled with CUDA support])
AC_TRY_RUN([
#include <lal/LALConfig.h>
#ifdef LAL_CUDA_ENABLED
int main( void ) { return 0; }
#else
int main( void ) { return 1; }
#endif
],
AC_MSG_RESULT([yes])
[cuda=true],
AC_MSG_RESULT([no])
[cuda=false],
AC_MSG_RESULT([unknown])
[cuda=false])
])

AC_DEFUN([LALSUITE_ENABLE_DOXYGEN],[
  # $0: enable Doxygen documentation
  AC_ARG_ENABLE(
    [doxygen],
    AC_HELP_STRING(
      [--enable-doxygen],
      [generate Doxygen documentation]
    ),[
      AS_CASE(["${enableval}"],
        [yes],[doxygen=true],
        [no],[doxygen=false],
        [AC_MSG_ERROR([invalid value "${enableval}" for --enable-doxygen])]
      )
    ],[
      doxygen=false
    ]
  )
  # end $0
])

AC_DEFUN([LALSUITE_USE_DOXYGEN],[
  # $0: configure Doxygen documentation
  AC_REQUIRE([LALSUITE_CHECK_GIT_REPO])

  # add configuration option for MathJax installation
  AC_ARG_WITH(
    [mathjax],
    AC_HELP_STRING(
      [--with-mathjax=PATH],
      [use MathJax installation at PATH [default: use CDN]]
    ),[
      DOXYGEN_MATHJAXDIR="${withval}"
    ],[
      DOXYGEN_MATHJAXDIR=
    ]
  )

  # enable Doxygen module
  LALSUITE_ENABLE_MODULE([DOXYGEN])

  AS_IF([test "x${doxygen}" = xtrue],[

    # configure Doxygen filter script
    AC_CONFIG_FILES([doxygen/filter],[chmod +x doxygen/filter])

    # Python is required to run some scripts
    LALSUITE_REQUIRE_PYTHON([2.6])

    # check for Doxygen
    AC_PATH_PROG([DOXYGEN],[doxygen],[],[])
    AS_IF([test "x${DOXYGEN}" = x],[
      AC_MSG_ERROR([could not find Doxygen in PATH])
    ])
    doxygen_min_version=1.8.1.2   # minimum required Doxygen version
    AC_MSG_CHECKING([if ${DOXYGEN} version is at least ${doxygen_min_version}])
    doxygen_version=`${DOXYGEN} --version 2>/dev/null`
    LALSUITE_VERSION_COMPARE([${doxygen_version}],[<],[${doxygen_min_version}],[
      AC_MSG_RESULT([no (${doxygen_version})])
      AC_MSG_ERROR([Doxygen version ${doxygen_min_version} or later is required])
    ])
    AC_MSG_RESULT([yes (${doxygen_version})])

    # ignore some Doxygen warnings due to Doxygen bugs
    AC_SUBST([DOXYGEN_WARNING_REGEX],[])
    LALSUITE_VERSION_COMPARE([${doxygen_version}],[<],[1.8.9.1],[
      # https://bugzilla.gnome.org/show_bug.cgi?id=742151
      DOXYGEN_WARNING_REGEX=["${DOXYGEN_WARNING_REGEX} -e '/^citelist/d'"]
    ])
    LALSUITE_VERSION_COMPARE([1.8.8],[<=],[${doxygen_version}],[
      LALSUITE_VERSION_COMPARE([${doxygen_version}],[<=],[1.8.9.1],[
        # https://bugzilla.gnome.org/show_bug.cgi?id=743604
        DOXYGEN_WARNING_REGEX=["${DOXYGEN_WARNING_REGEX} -e '/warning: Duplicate anchor/d'"]
      ])
    ])
    LALSUITE_VERSION_COMPARE([1.8.8],[<=],[${doxygen_version}],[
      LALSUITE_VERSION_COMPARE([${doxygen_version}],[<=],[1.8.9.1],[
        # https://bugzilla.gnome.org/show_bug.cgi?id=743605
        DOXYGEN_WARNING_REGEX=["${DOXYGEN_WARNING_REGEX} -e '/warning: explicit link request/d'"]
      ])
    ])

    # build some substituted variables from list of configured LAL libraries
    AC_SUBST([DOXYGEN_ENABLED_SECTIONS])
    AC_SUBST([DOXYGEN_TAGFILES],[])
    AC_SUBST([DOXYGEN_INSTALL_DIRMAP],[])
    for arg in ${lalsuite_libs}; do
      AS_CASE([${arg}],
        [lalsupport],[:],[
          ucarg=[`echo ${arg} | tr '[a-z]' '[A-Z]'`]
          DOXYGEN_ENABLED_SECTIONS="${DOXYGEN_ENABLED_SECTIONS} ${ucarg}"
          arg_htmldir_name="${ucarg}_HTMLDIR"
          AS_VAR_COPY([arg_htmldir],[${arg_htmldir_name}])
          AS_IF([test "x${LALSUITE_BUILD}" = xtrue],[
            tagpath="\$(abs_top_builddir)/../${arg}/doxygen/out"
          ],[
            tagpath="${arg_htmldir}"
          ])
          DOXYGEN_TAGFILES="${DOXYGEN_TAGFILES} ${tagpath}/${arg}.tag=${tagpath}"
          DOXYGEN_INSTALL_DIRMAP="${DOXYGEN_INSTALL_DIRMAP} ${tagpath}:${arg_htmldir}"
        ]
      )
    done

    # configure MathJax
    AC_SUBST([DOXYGEN_MATHJAXDIR])
    AS_IF([test "x${DOXYGEN_MATHJAXDIR}" != x],[
      AS_IF([test ! -f "${DOXYGEN_MATHJAXDIR}/MathJax.js"],[
        AC_MSG_ERROR([no MathJax installation found in ${DOXYGEN_MATHJAXDIR}])
      ])
    ],[
      for dir in /usr/share/javascript/mathjax; do
        AC_MSG_CHECKING([for MathJax installation in ${dir}])
        AS_IF([test -f "${dir}/MathJax.js"],[
          AC_MSG_RESULT([yes])
          DOXYGEN_MATHJAXDIR="${dir}"
          break
        ],[
          AC_MSG_RESULT([no])
        ])
      done
      AS_IF([test "x${DOXYGEN_MATHJAXDIR}" = x],[
        DOXYGEN_MATHJAXDIR='https://cdn.mathjax.org/mathjax/latest'
        AC_MSG_NOTICE([using MathJax CDN at ${DOXYGEN_MATHJAXDIR}])
      ])
    ])

  ])
  # end $0
])

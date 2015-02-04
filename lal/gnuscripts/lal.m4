# lal.m4 - lal specific macros
#
# serial 19

AC_DEFUN([LAL_ENABLE_DEBUG],
[AC_ARG_ENABLE(
  [debug],
  AC_HELP_STRING([--enable-debug],[set default lalDebugLevel to 1, which turns on output of error messages [default=yes]]),
  [AS_CASE(["${enableval}"],
    [yes],,
    [no],AC_DEFINE(LAL_NDEBUG, 1, Set default lalDebugLevel to 0),
    AC_MSG_ERROR(bad value for ${enableval} for --enable-debug))
  ], )
])

AC_DEFUN([LAL_ENABLE_FFTW3_MEMALIGN],
[AC_ARG_ENABLE(
  [fftw3_memalign],
  AC_HELP_STRING([--enable-fftw3-memalign],[use aligned memory optimizations with fftw3 [default=no]]),
  AS_CASE(["${enableval}"],
    [yes],[fftw3_memalign=true],
    [no],[fftw3_memalign=false],
    AC_MSG_ERROR([bad value for ${enableval} for --enable-fftw3-memalign])
  ),[fftw3_memalign=false])
])

AC_DEFUN([LAL_ENABLE_INTELFFT],
[AC_ARG_ENABLE(
  [intelfft],
  AC_HELP_STRING([--enable-intelfft],[use Intel FFT libraries insted of FFTW [default=no]]),
  AS_CASE(["${enableval}"],
    [yes],[intelfft=true],
    [no],[intelfft=false],
    [condor],[intelfft=true; qthread=tru; AC_DEFINE([LALQTHREAD],[1],[Use fake qthread library for MKL Condor compatibility])],
    AC_MSG_ERROR([bad value for ${enableval} for --enable-intelfft])
  ),[intelfft=false])
])

AC_DEFUN([LAL_ENABLE_MACROS],
[AC_ARG_ENABLE(
  [macros],
  AC_HELP_STRING([--enable-macros],[use LAL macros [default=yes]]),
  AS_CASE(["${enableval}"],
    [yes],,
    [no],AC_DEFINE([NOLALMACROS],[1],[Use functions rather than macros]),
    AC_MSG_ERROR([bad value for ${enableval} for --enable-macros])
  ),)
])

AC_DEFUN([LAL_ENABLE_PTHREAD_LOCK],
[AC_ARG_ENABLE(
  [pthread_lock],
  AC_HELP_STRING([--enable-pthread-lock],[use pthread mutex lock for threadsafety [default=yes]]),
  AS_CASE(["${enableval}"],
    [yes],[lal_pthread_lock=true],
    [no],[lal_pthread_lock=false],
    AC_MSG_ERROR([bad value for ${enableval} for --enable-pthread-lock])
  ),[lal_pthread_lock=true])
  AS_IF([test "x$lal_pthread_lock" = "xtrue"],
    AC_DEFINE([LAL_PTHREAD_LOCK],[1],[Use pthread mutex lock for threadsafety])
  )
])

AC_DEFUN([LAL_INTEL_MKL_QTHREAD_WARNING],
[echo "**************************************************************"
 echo "* LAL will be linked against the fake POSIX thread library!  *"
 echo "*                                                            *"
 echo "* This build of LAL will not be thread safe and cannot be    *"
 echo "* linked against the system pthread library.                 *"
 echo "*                                                            *"
 echo "* The environment variables                                  *"
 echo "*                                                            *"
 echo "*    MKL_SERIAL=YES                                          *"
 echo "*    KMP_LIBRARY=serial                                      *"
 echo "*                                                            *"
 echo "* must be set before running executables linked against this *"
 echo "* build of LAL.                                              *"
 echo "*                                                            *"
 echo "* Please see the documention of the FFT package for details. *"
 echo "**************************************************************"
])

AC_DEFUN([LAL_INTEL_FFT_LIBS_MSG_ERROR],
[echo "**************************************************************"
 echo "* The --enable-static and --enable-shared options are        *"
 echo "* mutually exclusive with Intel FFT libraries.               *"
 echo "*                                                            *"
 echo "* Please reconfigure with:                                   *"
 echo "*                                                            *"
 echo "*   --enable-static --disable-shared                         *"
 echo "*                                                            *"
 echo "* for static libraries or                                    *"
 echo "*                                                            *"
 echo "*   --disable-static --enable-shared                         *"
 echo "*                                                            *"
 echo "* for shared libraries.                                      *"
 echo "*                                                            *"
 echo "* Please see the instructions in the file INSTALL.           *"
 echo "**************************************************************"
AC_MSG_ERROR([Intel FFT must use either static or shared libraries])
])

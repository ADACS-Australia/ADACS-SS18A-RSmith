dnl $Id$
dnl process this file with m4 to get installation instructions
INSTALLATION INSTRUCTIONS
changequote(@@,@@)
This file describes how to perform a basic install of LAL.  This is not
quite a minimal install -- support for the frame library is included -- but it
is probably the best basic installation.  The instructions are written for use
with a Bourne-like shell (e.g., bash) rather than a C-shell (e.g., tcsh).
Really the only difference is that you need to change statements like

        MYENV=value
        export MYENV

(Bourne shell syntax) to

        setenv MYENV value

The instructions in this file are extracted from the shell scripts:

        doc/lal-preinstall-x.sh
        doc/lal-install-x.sh

Also extracted are nicely-formatted versions of these shell scripts:

        doc/lal-preinstall.sh
        doc/lal-install.sh

You can edit these scripts (as appropriate) and run them to install LAL.
Or just follow the instructions below.


PRE-INSTALLATION

include(@@lal-preinstall.txt@@)

INSTALLING LAL

include(@@lal-install.txt@@)

MORE DETAILS

Other useful make targets are:

        make dvi                # make documentation
        make check              # run the basic test suite
        make uninstall          # uninstall the library and header files
        make clean              # clean up compiled code (as before "make")
        make distclean          # clean up distribution (as before "configure")
        make cvs-clean          # clean up to cvs files (as before "00boot")

see the file INSTALL for additional details on configuring LAL.


SYSTEM-SPECIFIC INSTALLATION INSTRUCTIONS

SGI running IRIX 6.5 with gcc:

  * Configure with the option --with-cc="gcc -n32".

  * If you put shared objects (e.g., of the frame library) in non-standard
    places and are hoping to use LD_LIBRARY_PATH to locate them, you may need
    to set the environment variable LD_LIBRARYN32_PATH too.

  * If you have command-lines that are too long, you'll need to change
    the length of the lines allowed.  To do this use systune -i
    (or perhaps systune -r):

        $ systune -r
        systune-> ncargs 204800
        systune-> quit
    
    This increases the command line length maximum until reboot.
    Change it permanently with systune -b.

Alpha running Linux with ccc --- making shared libraries:

  * Problem: libtool doesn't know that ccc makes position-independent code
    by default (I hope ccc makes PIC by default...).

  * Solution: trick the configure script into thinking that you are using
    OSF/1 by using the --host option:

      ./configure --host=alpha-dec-osf3 --enable-shared

  * Note: use the same trick to make shared libraries for fftw!


HP-UX 10.20 may need

   --with-extra-cppflags="-D_HPUX_SOURCE"


Mac OS X (10.2.x, possibly 10.1.x, but NOT 10.3.x) with bundled cc/gcc:

  * Configure with:  --with-extra-cflags="-D_ANSI_SOURCE -no-cpp-precomp"

  * Note: I (Jolien) don't need these with 10.2 ... perhaps it depends on the
    version of the developer tools.  Also, do NOT use these flags with 10.3.


TROUBLESHOOTING

* If you need to re-run configure after it has failed while checking for a
  working FFTW, FrameL, or MPI, make sure to remove the file config.cache.

* The configure script assumes that ranlib is necessary unless it cannot find
  the program in your path.  If ranlib is on your path and you don't need
  ranlib, set the environment RANLIB to echo.

* "make dvi" must be run after "make" since make dvi requires the program
  laldoc must be compiled for "make dvi" to work.

* If you want to use a different latex program than the one chosen by the
  configure script, set it in the environment variable LATEX.  Also, if you
  want to have different tex flags (e.g., you want to disable the batchmode
  that the configure script uses) set the TEXFLAGS environment variable
  to the flags you want (or a space if you don't want any flags used).

* If you want to make a shared library version (default) of LAL with frame
  library and/or MPI interface, you need to make a shared library version of
  fftw, the frame library, and mpi too.  To make a static LAL library only,
  use the --disable-shared option when configuring LAL.


#!/bin/sh -x

## Perform the pre-install for LAL.  The following commands will download,
## build, and install the software required to build LAL.  The software
## includes the following:
##verse
##	autoconf-2.59
##	automake-1.8.5
##	fftw-3.0.1
##	gsl-1.5
##	libframe-6.13 (optional but recommended)
##	libmetaio-5.4 (optional but recommended)
##/verse
## If this software is already on your system, you can use the existing
## software.  If some of the software is missing, you can use the appropriate
## part of these instructions to install that component.  Pre-compiled versions
## of the software are also available for installation.  See the RPMs
## section for instructions on obtaining these.
##
## The software is installed in the directory "LSCSOFT_PREFIX".
## If this variable is not set, it will be installed in "$HOME/opt/lscsoft"
## by default. To install in some other location, set "LSCSOFT_PREFIX"
## to that location.
## 
## The commands listed below are appropriate for a Bourne-shell (e.g., bash);
## they will need to be modified appropriately for C-shells (e.g., tcsh).

## Edit "LSCSOFT_PREFIX" to change where to install the software:
#verbatim
LSCSOFT_PREFIX=${LSCSOFT_PREFIX:-"$HOME/opt/lscsoft"}
LSCSOFT_BINDIR=$LSCSOFT_PREFIX/bin
LSCSOFT_ETCDIR=$LSCSOFT_PREFIX/etc
LSCSOFT_INCDIR=$LSCSOFT_PREFIX/include
LSCSOFT_LIBDIR=$LSCSOFT_PREFIX/lib
LSCSOFT_SRCDIR=$LSCSOFT_PREFIX/src
LSCSOFT_TMPDIR=$LSCSOFT_PREFIX/tmp
#/verbatim

## This is where to get sources:
#verbatim
LALSRCURL=http://www.lsc-group.phys.uwm.edu/lal/sources
#/verbatim

### uncomment to use lynx instead of curl
###curl() {
###lynx -dump $1
###}

### uncomment to use wget instead of curl
###curl() {
###wget -O- $1
###}

###
### the rest of this script should not need to be edited
###

### simple failure
#ignore
fail() {
  echo "!!! Failure" 1>&2
  exit 1
}
#/ignore

# update "PATH" so that the correct programs will be run
#verbatim
PATH=$LSCSOFT_BINDIR:$PATH
export PATH
#/verbatim

# setup directories
#verbatim
mkdir -p $LSCSOFT_PREFIX || fail
mkdir -p $LSCSOFT_BINDIR || fail
mkdir -p $LSCSOFT_ETCDIR || fail
mkdir -p $LSCSOFT_INCDIR || fail
mkdir -p $LSCSOFT_LIBDIR || fail
mkdir -p $LSCSOFT_SRCDIR || fail
mkdir -p $LSCSOFT_TMPDIR || fail
#/verbatim

# get required autoconf, automake, fftw3, frame, gsl, and metaio
# you can use "lynx -dump" or "wget -O-" instead of "curl"
#verbatim
curl $LALSRCURL/autoconf-2.59.tar.gz > $LSCSOFT_TMPDIR/autoconf-2.59.tar.gz || fail
curl $LALSRCURL/automake-1.8.5.tar.gz > $LSCSOFT_TMPDIR/automake-1.8.5.tar.gz || fail
curl $LALSRCURL/fftw-3.0.1.tar.gz > $LSCSOFT_TMPDIR/fftw-3.0.1.tar.gz || fail
curl $LALSRCURL/gsl-1.5.tar.gz > $LSCSOFT_TMPDIR/gsl-1.5.tar.gz || fail
curl $LALSRCURL/libframe-6.13.tar.gz > $LSCSOFT_TMPDIR/libframe-6.13.tar.gz || fail
curl $LALSRCURL/libmetaio-5.4.tar.gz > $LSCSOFT_TMPDIR/libmetaio-5.4.tar.gz || fail
#/verbatim

# unpack these archives in "LSCSOFT_SRCDIR"
#verbatim
cd $LSCSOFT_SRCDIR || fail
tar -zxvf $LSCSOFT_TMPDIR/autoconf-2.59.tar.gz || fail
tar -zxvf $LSCSOFT_TMPDIR/automake-1.8.5.tar.gz || fail
tar -zxvf $LSCSOFT_TMPDIR/fftw-3.0.1.tar.gz || fail
tar -zxvf $LSCSOFT_TMPDIR/gsl-1.5.tar.gz || fail
tar -zxvf $LSCSOFT_TMPDIR/libframe-6.13.tar.gz || fail
tar -zxvf $LSCSOFT_TMPDIR/libmetaio-5.4.tar.gz || fail
#/verbatim

# build and install autoconf
#verbatim
cd $LSCSOFT_SRCDIR/autoconf-2.59 || fail
./configure --prefix=$LSCSOFT_PREFIX || fail
make || fail
make install || fail
#/verbatim

# build and install automake
#verbatim
cd $LSCSOFT_SRCDIR/automake-1.8.5 || fail
./configure --prefix=$LSCSOFT_PREFIX || fail
make || fail
make install || fail
#/verbatim

# build and install fftw3
#verbatim
cd $LSCSOFT_SRCDIR/fftw-3.0.1 || fail
./configure --prefix=$LSCSOFT_PREFIX --enable-shared --enable-float || fail
make  # note: ignore fail... the build fails on MacOSX, but not seriously
make install # note: ignore fail
make distclean || fail
./configure --prefix=$LSCSOFT_PREFIX --enable-shared || fail
make # note: ignore fail
make install # note: ignore fail
#/verbatim

# build and install gsl
#verbatim
cd $LSCSOFT_SRCDIR/gsl-1.5 || fail
./configure --prefix=$LSCSOFT_PREFIX || fail
make || fail
make install || fail
#/verbatim

# build and install libframe
#verbatim
cd $LSCSOFT_SRCDIR/libframe-6.13 || fail
./configure --prefix=$LSCSOFT_PREFIX || fail
make || fail
make install || fail
#/verbatim

# build and install libmetaio
#verbatim
cd $LSCSOFT_SRCDIR/libmetaio-5.4 || fail
./configure --prefix=$LSCSOFT_PREFIX || fail
make || fail
make install || fail
#/verbatim

### write environment configuration file
#ignore
rm -f $LSCSOFT_ETCDIR/lscsoft-user-env.sh || fail
cat > $LSCSOFT_ETCDIR/lscsoft-user-env.sh <<\EOF
# Source this file to set up your environment to use lscsoft software.
# This requires that LSCSOFT_LOCATION be set.
# LSCSOFT_PREFIX will be set by this script to save the current location
# so that the old LSCSOFT_PREFIX information can be removed from your
# environment if LSCSOFT_LOCATION is changed and this file is resourced.
# If LSCSOFT_LOCATION is set but empty then the previous location is
# removed from the environment.

if [ "${LSCSOFT_LOCATION-X}" = "X" ]; then
  echo "ERROR: environment variable LSCSOFT_LOCATION not defined" 1>&2
  return 1
fi

if [ -n "${LSCSOFT_PREFIX}" ]; then
  PATH=`echo "${PATH}" | sed -e "s%:${LSCSOFT_PREFIX}[^:]*%%g" -e "s%^${LSCSOFT_PREFIX}[^:]*:\{0,1\}%%"`
  LD_LIBRARY_PATH=`echo "${LD_LIBRARY_PATH}" | sed -e "s%:${LSCSOFT_PREFIX}[^:]*%%g" -e "s%^${LSCSOFT_PREFIX}[^:]*:\{0,1\}%%"`
  MANPATH=`echo "${MANPATH}" | sed -e "s%:${LSCSOFT_PREFIX}[^:]*%%g" -e "s%^${LSCSOFT_PREFIX}[^:]*:\{0,1\}%%"`
  export PATH MANPATH LD_LIBRARY_PATH
fi

LSCSOFT_PREFIX=${LSCSOFT_LOCATION}
export LSCSOFT_PREFIX

if [ -n "${LSCSOFT_PREFIX}" ]; then
  PATH=`echo "${PATH}" | sed -e "s%:${LSCSOFT_PREFIX}[^:]*%%g" -e "s%^${LSCSOFT_PREFIX}[^:]*:\{0,1\}%%"`
  LD_LIBRARY_PATH=`echo "${LD_LIBRARY_PATH}" | sed -e "s%:${LSCSOFT_PREFIX}[^:]*%%g" -e "s%^${LSCSOFT_PREFIX}[^:]*:\{0,1\}%%"`
  MANPATH=`echo "${MANPATH}" | sed -e "s%:${LSCSOFT_PREFIX}[^:]*%%g" -e "s%^${LSCSOFT_PREFIX}[^:]*:\{0,1\}%%"`
  PATH=${LSCSOFT_LOCATION}/bin:${PATH}
  LD_LIBRARY_PATH=${LSCSOFT_LOCATION}/lib:${LD_LIBRARY_PATH}
  MANPATH=${LSCSOFT_LOCATION}/man:${MANPATH}
  export PATH MANPATH LD_LIBRARY_PATH
fi
EOF
#/ignore

## Finally, you need to set the environment variable "LSCSOFT_LOCATION"
## to be where you have installed this software.
### (don't actually do it... for illustration purposes only)
#ignore
if 0 ; then
#/ignore
#verbatim
  LSCSOFT_LOCATION=$LSCSOFT_PREFIX
  export LSCSOFT_LOCATION
#/verbatim
#ignore
fi
#/ignore

### Print a message alerting user to set "LSCSOFT_LOCATION"
#ignore
cat <<EOF
=======================================================================

To setup your environment to use the software that has been installed
please add the following to your .profile:

        LSCSOFT_LOCATION=$LSCSOFT_PREFIX
        export LSCSOFT_LOCATION
        . \${LSCSOFT_LOCATION}/etc/lscsoft-user-env.sh

=======================================================================
EOF
#/ignore

### all done
#ignore
exit 0
#/ignore

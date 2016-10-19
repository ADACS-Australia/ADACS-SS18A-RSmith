#!/bin/bash

# simple failure
fail() {
    echo
    echo "*******************************************"
    echo "Automated installation has failed" 1>&2
    echo "*******************************************"
    echo

    if test -n "$LOGFILE" -a -r "$LOGFILE" ; then
	date '+[%Y-%m-%d %H:%M:%S]' >> "$LOGFILE"
	echo "Transcript of failure is in ${LOGFILE}"
	echo "Final fifteen lines are:"
	tail -15 "$LOGFILE"
    fi

    exit 1
} ## fail()

log_and_show() {
    echo `date '+[%Y-%m-%d %H:%M:%S]'` "$@" >> "$LOGFILE"
    echo "$@" >&2
}

log_and_do() {
    echo `date '+[%Y-%m-%d %H:%M:%S]'` "$@" >> "$LOGFILE"
    "$@" >> "$LOGFILE" 2>&1 || fail
}

log_and_dont_fail() {
    echo `date '+[%Y-%m-%d %H:%M:%S]'` "$@" >> "$LOGFILE"
    "$@" >> "$LOGFILE" 2>&1
}

download() {
    if [ ".$2" = "." ]; then
      u='http://www.aei.mpg.de/~bema'
      f="$1"
    else
      u="$1"
      f="$2"
    fi
    trial=1
    while ! {
        log_and_do rm -f "$f" &&
        echo "`date '+[%Y-%m-%d %H:%M:%S]'` wget --no-check-certificate --passive-ftp --no-continue $u/$f" >> "$LOGFILE" &&
        wget --no-check-certificate --passive-ftp --tries=1 "$u/$f" >> "$LOGFILE" 2>&1 ||
        echo "wget exited with error $? - falling back to curl" >> "$LOGFILE" &&
        log_and_do rm -f "$f" &&
        echo "`date '+[%Y-%m-%d %H:%M:%S]'` curl -o $f $u/$f" >> "$LOGFILE" &&
        curl -o "$f" "$u/$f" >> "$LOGFILE" 2>&1
    } ; do
        trial=`expr $trial + 1`
        test $trial -gt $retries && log_and_show "failed" && fail
        sleep 30
    done
}

eah_build2_loc="`echo $PWD/$0 | sed 's%/[^/]*$%%'`"

test ".$appname" = "." && appname=einstein_S6Bucket
test ".$appversion" = "." && appversion=0.00
boinc_repo="https://gitlab.aei.uni-hannover.de/einsteinathome/boinc.git"
boinc_rev=current_gw_apps
#previous:-r22844 -r22825 -r22804 -r22794 -r22784 -r22561 -r22503 -r22363 -r21777 -r'{2008-12-01}'
retries=1

gsl=gsl-1.16
fftw=fftw-3.3.4
zlib=zlib-1.2.8
binutils=binutils-2.19

zlib_shared="--static"

for i; do
    case "$i" in
	--win32)
	    build_win32=true
	    cross_prefix=i586-mingw32msvc ;;
	--cross-prefix=*)
	    cross_prefix=`echo "$i" | sed 's/--cross-prefix=//'`
	    if echo "$cross_prefix"|grep w64-mingw32 >/dev/null; then
		LDFLAGS="-static-libstdc++ $LDFLAGS"
	    fi ;;
	--static)
	    shared_copt="--disable-shared" ;;
	--rebuild)
	    rebuild_zlib=true
	    rebuild_binutils=true
	    rebuild_boinc=true
	    rebuild_lal=true
	    rebuild=true ;;
	--rebuild-lal)
	    rebuild_lal=true ;;
	--rebuild-boinc)
	    rebuild_boinc=true ;;
	--rebuild-zlib)
	    rebuild_zlib=true ;;
	--rebuild-binutils)
	    rebuild_binutils=true ;;
	--release)
	    rebuild_zlib=true
	    rebuild_binutils=true
	    rebuild_boinc=true
	    rebuild_lal=true
	    rebuild=true
	    release=true
	    CFLAGS="-O3 $CFLAGS"
	    LDFLAGS="-static-libgcc $LDFLAGS"
	    shared_copt="--disable-shared"  ;;
	--appname=*)
	    appname=`echo "$i" | sed 's/--appname=//'` ;;
	--appversion=*)
	    appversion=`echo "$i" | sed 's/--appversion=//'` ;;
	--norebuild) # dangerous, for testing only!
	    rebuild_zlib=""
	    rebuild_binutils=""
	    rebuild_boinc=""
	    rebuild_lal=""
	    rebuild="" ;;
        --noupdate)
            noupdate=true ;;
	--gc-opt)
	    CPPFLAGS="-DGC_SSE2_OPT $CPPFLAGS" ;;
        --avx)
            fftw_copts_single="--enable-avx $fftw_copts_single"
	    planclass=__AVX ;;
	--64)
	    fftw_copts_single="--enable-sse --enable-sse2 $fftw_copts_single"
	    CPPFLAGS="-m64 $CPPFLAGS"
	    CXXFLAGS="-m64 $CXXFLAGS"
	    CFLAGS="-m64 $CFLAGS"
	    LDFLAGS="-m64 $LDFLAGS" ;;
	--32)
	    CPPFLAGS="-m32 $CPPFLAGS"
	    CXXFLAGS="-m32 $CXXFLAGS"
	    CFLAGS="-m32 $CFLAGS"
	    LDFLAGS="-m32 $LDFLAGS" ;;
	--sse)
	    CPPFLAGS="-DENABLE_SSE_EXCEPTIONS $CPPFLAGS"
	    CFLAGS="-msse -march=pentium3 $CFLAGS"
	    fftw_copts_single="--enable-sse $fftw_copts_single"
	    planclass=__SSE
	    acc="_sse";;
	--sse2)
	    CPPFLAGS="-DENABLE_SSE_EXCEPTIONS -DGC_SSE2_OPT $CPPFLAGS"
	    CFLAGS="-msse -msse2 -mfpmath=sse $CFLAGS"
	    if echo $CFLAGS|grep -e -m64 >/dev/null; then
		CFLAGS="$CFLAGS -march=core2"
	    else
		CFLAGS="$CFLAGS -march=pentium-m"
	    fi
            fftw_copts_single="--enable-sse $fftw_copts_single"
            fftw_copts_double="--enable-sse2 $fftw_copts_double"
	    planclass=__SSE2
	    acc="_sse2";;
	--altivec)
	    CPPFLAGS="-arch ppc -maltivec -faltivec $CPPFLAGS"
	    CFLAGS="-arch ppc -fast -mcpu=G4 -maltivec -faltivec $CFLAGS"
	    CXXFLAGS="-arch ppc -mcpu=G4 $CXXFLAGS"
	    LDFLAGS="-arch ppc $LDFLAGS"
	    fftw_copts_single="--enable-altivec $fftw_copts_single"
	    planclass=__ALTIVEC
	    acc="_altivec"
	    cross_copt=--host=powerpc-apple-darwin ;;
	--cuda)
	    cuda=true
	    acc="_cuda" ;;
	--tiger)
	    export MACOSX_DEPLOYMENT_TARGET=10.4
	    export SDKROOT="/Developer/SDKs/MacOSX10.4u.sdk"
	    pflags="-D_NONSTD_SOURCE -isystem $SDKROOT"
	    CPPFLAGS="$pflags $CPPFLAGS -DMAC_OS_X_VERSION_MAX_ALLOWED=1040 -DMAC_OS_X_VERSION_MIN_REQUIRED=1040"
	    CFLAGS="$pflags $CFLAGS"
	    CXXFLAGS="$pflags $CXXFLAGS"
	    LDFLAGS="$pflags -Wl,-syslibroot,$SDKROOT $LDFLAGS"
	    export RELEASE_LDADD="/usr/lib/libstdc++-static.a" ;;
	--with-ssl=*)
	    WITH_SSL="$i"
	    ssldir=`echo "$i" | sed 's/--with-ssl=//'`
	    CPPFLAGS="$CPPFLAGS -I$ssldir/include"
	    LIBS="$LIBS -L$ssldir/lib" ;;
	--check)
	    check=true ;;
	--check-only)
	    check=true
	    check_only=true ;;
	--check-app=*)
	    check=true
	    check_only=true
	    check_app=`echo $PWD/$i | sed 's/--check-app=//;s%.*//%/%'`;;
	--boinc-tag=*)
	    boinc_rev="`echo $i | sed 's/^.*=//'`";;
	--boinc-commit=*)
	    boinc_rev="`echo $i | sed 's/^.*=//'`";;
	--retries=*)
	    retries="`echo $i | sed 's/^.*=//'`";;
	--zlib-shared*)
	    echo "$i" | fgrep = >/dev/null &&
	        zlib="zlib-`echo $i | sed 's/^.*=//'`"
	    build_zlib=true
	    zlib_shared="--shared";;
	--help)
	    echo "$0 builds Einstein@home Applications of LALApps HierarchSearchGCT code"
	    echo "  --win32           cros-compile a Win32 App (requires MinGW, target i586-mingw32msvc-gcc)"
	    echo "  --cross-prefix=<p> use a compiler toolchain with a prefix other than i586-mingw32msvc"
	    echo "  --32              build 32Bit (add -m32 to  CPPFLAGS, CXXFLAGS, CFLAGS and LDFLAGS)"
	    echo "  --64              build 64Bit (add -m64 to  CPPFLAGS, CXXFLAGS, CFLAGS and LDFLAGS)"
	    echo "                    - needs to precede --sse2 if that is also used"
	    echo "  --tiger           build to run on Mac OS 10.4"
	    echo "  --cuda            build an App that uses CUDA"
	    echo "  --sse             build an App that uses SSE"
	    echo "  --sse2            build an App that uses SSE2 (implies --sse and --gc-opt)"
            echo "  --avx             build an App that uses AVX (currently in FFTW only)"
	    echo "  --altivec         build an App that uses AltiVec"
	    echo "  --gc-opt          build an App that uses SSE2 GC optimization"
	    echo "  --boinc-tag=<tag>|--boinc-commit=<sha1> specify a BOINC commit to use (defaults to 'current_gw_apps')"
	    echo "  --with-ssl=<path> gets paased to BOINC configure"
	    echo "  --static          try to link statically (configure with --disable-shared)"
	    echo "  --rebuild         build FFTW, gsl, BOINC and LAL from source even if they are found by pkg-config"
	    echo "  --rebuild-zlib    rebuild zlib"
	    echo "  --zlib-shared[=<v>] EXPERIMENTAL: build a local, shard version of zlib"
	    echo "  --rebuild-binutils rebuild binutils"
	    echo "  --rebuild-lal     rebuild lal & lalpulsar"
	    echo "  --rebuild-boinc   rebuild BOINC"
	    echo "  --release         use some dark magic to make the App most compatible and add remote debugging."
	    echo "                    Implies --static and --rebuild and even more dirty hacks"
	    echo "  --appname=<name>  set an application name (only used in --release builds, defaults to einstein_S5GC1HF)"
	    echo "  --appversion=N.NN set an application version (only used in --release builds, defaults to 0.00)"
	    echo "  --norebuild       disables --rebuild on --release. DANGEROUS! Use only for testing the build script"
	    echo "  --noupdate        use previously retrieved (possibly locally modified) sources, doesn't need internet"
	    echo "  --retries=<n>     try downloads (curl, git) this many times (defaults to 1)"
	    echo "  --check           test the newly built HierarchSearchGC App"
	    echo "  --check-only      only test the already built HierarchSearchGC App"
	    echo "  --check-app=<app> only test the app specified, not necessarily the one just built"
	    echo "  --help            show this message and exit"
	    exit ;;
	*) echo "unknown option '$i', try $0 --help"; exit ;;
    esac
done

EAH="$PWD/EinsteinAtHome"
LOGFILE="$EAH/build.log"
SOURCE="$EAH/source"
BUILD="$EAH/build$acc"
INSTALL="$EAH/install$acc"

# Jenkins build info
if [ ".$BUILD_INFO" = "." ]; then
  test ".$BUILD_TAG" = "." || BUILD_INFO="Build $BUILD_TAG"
  test ".$NODE_NAME" = "." || BUILD_INFO="$BUILD_INFO on $NODE_NAME"
  test ".$BUILD_ID" = "."  || BUILD_INFO="$BUILD_INFO at $BUILD_ID"
fi
if [ -n "$BUILD_INFO" ]; then
  LOGFILE="$EAH/${appname}_`echo build_$BUILD_INFO | sed 's%/%_%g;s/ on /./;s/  */_/g'`.log"
  CPPFLAGS="$CPPFLAGS -DHAVE_BUILD_INFO_H"
fi

# make sure the E@H directory exists (for logging)
mkdir -p "$EAH" || fail

echo " " >> "$LOGFILE"
log_and_show "==========================================="
log_and_show "Build start `date`"

missing_wine_warning=false
if [ ."$build_win32" = ."true" ] ; then
    if echo "$LDFLAGS" | grep -w -e -m64 >/dev/null; then
	platform=windows_x86_64
	BUILD="${BUILD}_win64"
	INSTALL="${INSTALL}_win64"
    else
	platform=windows_intelx86
	BUILD="${BUILD}_win32"
	INSTALL="${INSTALL}_win32"
    fi
    export CC=${cross_prefix}-gcc
    export CXX=${cross_prefix}-g++
    export AR=${cross_prefix}-ar
    export RANLIB=${cross_prefix}-ranlib
    export LIBTOOL=${cross_prefix}-libtool
    CPPFLAGS="-DMINGW_WIN32 -DWIN32 -D_WIN32 -D_WIN32_WINDOWS=0x0410 $CPPFLAGS"
    # -include $INSTALL/include/win32_hacks.h
    cross_copt=--host=i586-pc-mingw32
    shared_copt="--disable-shared"
    fftw_copts_single="$fftw_copts_single --with-our-malloc16"
    fftw_copts_double="$fftw_copts_double --with-our-malloc16"
    build_zlib=true
    ext=".exe"
    wine=`which wine`
    if [ ".$wine" = "." -a ".$check" = ".true" ]; then
        missing_wine_warning=true
        check=false
    fi
    if [ ".$cuda" = ".true" ] ; then
	test ".$WINEPREFIX" = "." &&
	WINEPREFIX="$HOME/.wine"
	CPPFLAGS="-I$WINEPREFIX/drive_c/CUDA/include $CPPFLAGS"
	export CUDART="$WINEPREFIX/drive_c/CUDA/lib/cudart.lib"
	export NVCC="$PWD/nvcc-wine-wrapper.sh"
    elif [ ".$release" = ".true" ] ; then
	CPPFLAGS="-DHAVE_EXCHNDL -I$INSTALL/include/bfd $CPPFLAGS"
	CFLAGS="-gstabs3 $CFLAGS"
	CXXFLAGS="-gstabs3 $CXXFLAGS"
	export RELEASE_DEPS="exchndl.o"
	export RELEASE_LDADD="$RELEASE_LDADD exchndl.o -lbfd -liberty -lintl"
	build_binutils=true
    fi
else
    case `uname -s` in
	Darwin)
            if echo "$LDFLAGS" | grep -w -e -m64 >/dev/null; then
		platform=x86_64-apple-darwin
		BUILD="${BUILD}_64"
		INSTALL="${INSTALL}_64"
	    else
		platform=i686-apple-darwin
	    fi
	    LDFLAGS="-framework Carbon -framework AppKit -framework IOKit -framework CoreFoundation $LDFLAGS" ;;
	Linux)
	    LDFLAGS="-lpthread $LDFLAGS"
	    if echo "$LDFLAGS" | grep -w -e -m64 >/dev/null; then
	        platform=x86_64-pc-linux-gnu
		BUILD="${BUILD}_64"
		INSTALL="${INSTALL}_64"
	    else
	        platform=i686-pc-linux-gnu
	    fi
	    if [ ".$WITH_SSL" = "." -a -d /usr/local/ssl ]; then
	        ssldir=/usr/local/ssl
		CPPFLAGS="$CPPFLAGS -I$ssldir/include"
		LIBS="$LIBS -L$ssldir/lib"
		WITH_SSL="--with-ssl=$ssldir"
	    fi
	    if [ ".$release" = ".true" ]; then
		CPPFLAGS="-DDLOPEN_LIBGCC -DEXT_STACKTRACE -I$INSTALL/include/bfd $CPPFLAGS"
		export RELEASE_DEPS="erp_execinfo_plus.o libstdc++.a"
		export RELEASE_LDADD="$RELEASE_LDADD erp_execinfo_plus.o -lbfd -liberty -ldl"
		build_zlib=true
		build_binutils=true
		enable_linux_compatibility_workarounds=true
	    fi ;;
    esac
fi

if [ ".$cuda" = ".true" -a ."$build_win32" = ."true" ]; then
    export CFLAGS="-g0 $CFLAGS"
else
    export CFLAGS="-g $CFLAGS"
fi
LDFLAGS="-L$INSTALL/lib $LDFLAGS"
if echo "$LDFLAGS" | grep -e -m64 >/dev/null; then
    LDFLAGS="-L$INSTALL/lib64 $LDFLAGS"
fi

# export environment variables
export CPPFLAGS="-DPULSAR_MAX_DETECTORS=2 -DUSEXLALLOADSFTS -DBOINC_APIV6 -D__NO_CTYPE -DUSE_BOINC -DEAH_BOINC -I$INSTALL/include $CPPFLAGS"
export LDFLAGS
export LD_LIBRARY_PATH="$INSTALL/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$INSTALL/lib:$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL/lib/pkgconfig:$PKG_CONFIG_PATH"
export BOINC_PREFIX="$INSTALL"

# log environment variables
echo "$0" "$@" >> "$LOGFILE"
echo CC="\"$CC\"" >> "$LOGFILE"
echo CXX="\"$CXX\"" >> "$LOGFILE"
echo AR="\"$AR\"" >> "$LOGFILE"
echo RANLIB="\"$RANLIB\"" >> "$LOGFILE"
echo CFLAGS="\"$CFLAGS\"" >> "$LOGFILE"
echo CPPFLAGS="\"$CPPFLAGS\"" >> "$LOGFILE"
echo LDFLAGS="\"$LDFLAGS\"" >> "$LOGFILE"
echo LD_LIBRARY_PATH="\"$LD_LIBRARY_PATH\"" >> "$LOGFILE"
echo DYLD_LIBRARY_PATH="\"$DYLD_LIBRARY_PATH\"" >> "$LOGFILE"
echo PKG_CONFIG_PATH="\"$PKG_CONFIG_PATH\"" >> "$LOGFILE"
echo BOINC_PREFIX="\"$BOINC_PREFIX\"" >> "$LOGFILE"
echo RELEASE_DEPS="\"$RELEASE_DEPS\"" >> "$LOGFILE"
echo RELEASE_LDADD="\"$RELEASE_LDADD\"" >> "$LOGFILE"
echo BUILD_INFO="\"$BUILD_INFO\"" >> "$LOGFILE"

# warning message delaeyed until log file was known
if [ ."$missing_wine_warning" = ."true" ] ; then
    log_and_show "WARNING: 'wine' not found, disabling check as it won't work"
fi
# augment wine's PATH such that
#   /usr/lib/gcc/i686-w64-mingw32/4.8/libstdc++-6.dll
# and
#   /usr/i686-w64-mingw32/lib/libwinpthread-1.dll
# are found
# test -r ~/.wine/system.reg && !fgrep i686-w64-mingw32 ~/.wine/system.reg &&
#   sed -i~ 's/^\("PATH"=.*\)"$/\1;Z:\\\\usr\\\\i686-w64-mingw32\\\\lib;Z:\\\\usr\\\\lib\\\\gcc\\\\i686-w64-mingw32\\\\4.8/' ~/.wine/system.reg
# see https://fedoraproject.org/wiki/MinGW/Configure_wine

if ! [ .$check_only = .true ]; then

log_and_do rm -rf "$BUILD"
test -n "$rebuild" &&
  log_and_do rm -rf "$INSTALL"

log_and_do mkdir -p "$SOURCE" "$BUILD/$fftw" "$BUILD/$gsl" "$BUILD/lal" "$BUILD/lalpulsar" "$BUILD/lalapps" "$BUILD/boinc" "$INSTALL/include"

if [ ."$build_win32" = ."true" ] ; then
    echo '#define bzero(a,b) memset(a,0,b)' > "$INSTALL/include/win32_hacks.h"
    echo '#define index(s,c) strchr(s,c)'  >> "$INSTALL/include/win32_hacks.h"
fi

if [ -n "$BUILD_INFO" ]; then
  echo '#define BUILD_INFO "'"$BUILD_INFO"'"' > "$INSTALL/include/build_info.h"
fi

log_and_do cd "$SOURCE"

if test -z "$rebuild" && pkg-config --exists gsl; then
    log_and_show "using existing gsl source"
elif test -z "$noupdate"; then
    log_and_show "retrieving $gsl"
    download ftp://ftp.fu-berlin.de/unix/gnu/gsl $gsl.tar.gz
    log_and_do tar xzf "$gsl.tar.gz"
fi

if test -z "$rebuild" && pkg-config --exists fftw3 fftw3f; then
    log_and_show "using existing fftw source"
elif test -z "$noupdate"; then
    log_and_show "retrieving $fftw"
    download http://www.fftw.org $fftw.tar.gz
    log_and_do tar xzf "$fftw.tar.gz"
fi

if test ."$build_zlib" = ."true"; then
    if test -z "$rebuild" -a -d "$zlib"; then
        log_and_show "using existing zlib source"
    elif test -z "$noupdate"; then
        log_and_show "retrieving $zlib"
        download $zlib.tar.gz
        log_and_do cd "$BUILD"
        log_and_do tar xzf "$SOURCE/$zlib.tar.gz"
        log_and_do cd "$SOURCE"
    fi
fi

if test -n "$build_binutils" -a -n "$rebuild_binutils" -a -z "$noupdate"; then
    log_and_show "retrieving $binutils"
# alternative locations:
#    download https://atlas3.atlas.aei.uni-hannover.de/~bema/tarballs $binutils.tar.gz
#    download http://mirrors.usc.edu/pub/gnu/binutils $binutils.tar.gz
    download $binutils.tar.gz
    log_and_do rm -rf "$binutils"
    log_and_do tar xzf "$binutils.tar.gz"
fi

if test -n "$noupdate" -o -z "$rebuild_boinc" -a -d "$SOURCE/boinc"; then
    log_and_show "using existing boinc source"
else
    log_and_show "retrieving boinc"
    if test -d "$SOURCE/boinc" -a -d "$SOURCE/boinc/.git" ; then
        log_and_do cd "$SOURCE/boinc"
        # if "$boinc_rev" is a tag that already exists locally,
        # delete it locally first in order to get updated from remote. Praise git !!
        if git tag | fgrep -x "$boinc_rev" >/dev/null ; then
            log_and_dont_fail git tag -d "$boinc_rev"
        fi
        log_and_do git fetch
    else
        log_and_do cd "$SOURCE"
        log_and_do rm -rf boinc
        trial=1
        while ! log_and_dont_fail git clone "$boinc_repo" ; do
            trial=`expr $trial + 1`
            test $trial -gt $retries && log_and_show "failed" && fail
            sleep 30
            log_and_do rm -rf boinc
        done
        log_and_do cd boinc
    fi
    log_and_do git fetch --tags
    log_and_do git checkout -f "$boinc_rev"
    log_and_do cd "$SOURCE"
fi

if test \! -d lalsuite/.git ; then
    log_and_do rm -rf lalsuite
    log_and_do ln -s "$eah_build2_loc/../../../.." lalsuite
fi

if test ."$build_zlib" = ."true"; then
    if test -z "$rebuild_zlib" && pkg-config --exists zlib; then
        log_and_show "using existing zlib"
    else
        log_and_show "compiling zlib"
        log_and_do cd "$BUILD/$zlib"
        if [ "$zlib_shared" = "--shared" -a "$zlib" = "zlib-1.2.3" ] && echo "$CFLAGS" | grep -w -e -m64 >/dev/null; then
            CC="gcc -m64" log_and_do "./configure" $zlib_shared --prefix="$INSTALL"
        else
            log_and_do "./configure" $zlib_shared --prefix="$INSTALL"
        fi
        log_and_do make
        log_and_do make install
        log_and_do mkdir -p "$INSTALL/lib/pkgconfig"
        echo 'sed "s%^prefix=.*%prefix=$INSTALL%;s/^Version: .*/Version: $zlib/;s/^Version: zlib-/Version: /" "$eah_build2_loc/zlib.pc.in" > "$INSTALL/lib/pkgconfig/zlib.pc"' >> "$LOGFILE"
        sed "s%^prefix=.*%prefix=$INSTALL%;s/^Version: .*/Version: $zlib/;s/^Version: zlib-/Version: /" "$eah_build2_loc/zlib.pc.in" > "$INSTALL/lib/pkgconfig/zlib.pc" || fail
    fi
fi

if test -z "$rebuild" && pkg-config --exists fftw3 fftw3f; then
    log_and_show "using existing fftw"
else
    log_and_show "compiling fftw"
    log_and_do cd "$BUILD/$fftw"
    log_and_do "$SOURCE/$fftw/configure" $fftw_copts_double "$shared_copt" "$cross_copt" --prefix="$INSTALL"
    log_and_dont_fail make uninstall
    log_and_do make
    log_and_do make install
    log_and_do "$SOURCE/$fftw/configure" $fftw_copts_single --enable-single "$shared_copt" "$cross_copt" --prefix="$INSTALL"
    log_and_dont_fail make uninstall
    log_and_do make
    log_and_do make install
fi

if test -z "$rebuild" && pkg-config --exists gsl; then
    log_and_show "using existing gsl"
else
    log_and_show "compiling gsl"
    log_and_do cd "$BUILD/$gsl"
    log_and_do "$SOURCE/$gsl/configure" "$shared_copt" "$cross_copt" --prefix="$INSTALL"
    log_and_dont_fail make uninstall
    log_and_do make
    log_and_do make install
fi

if test -n "$build_binutils"; then
  if test -z "$rebuild_binutils"; then
    log_and_show "using existing binutils"
  else
    log_and_show "compiling binutils"
    log_and_do mkdir -p "$BUILD/$binutils"
    log_and_do cd "$BUILD/$binutils"
    log_and_do "$SOURCE/$binutils/configure" --disable-werror "$shared_copt" "$cross_copt" --prefix="$INSTALL"
    log_and_dont_fail make uninstall
    if [ ".$enable_linux_compatibility_workarounds" = ".true" ]; then
        log_and_dont_fail make -k
        log_and_dont_fail make -k install
    else
        log_and_do make
        log_and_do make install
    fi
    # some post-build installation due to targets missing in the library
    log_and_do cd "$SOURCE/$binutils"
    log_and_do mkdir -p "$INSTALL/include/bfd"
    log_and_do cp -r include/* bfd/*.h "$BUILD/$binutils/binutils/config.h" "$INSTALL/include/bfd"
    log_and_do rm -f "$INSTALL/include/bfd/getopt.h"
    if [ ."$build_win32" = ."true" ] ; then
	log_and_do cd "$BUILD/$binutils"
	log_and_do cp "intl/libintl.a" "$INSTALL/lib"
        # patch a few headers
	( cd "$INSTALL/include/bfd" &&
	    patch -N -p0 <<EOF
diff -ur include.org/coff/internal.h include/coff/internal.h
--- coff/internal.h	2008-10-06 17:29:08.000000000 +0200
+++ coff/internal.h	2008-10-06 17:31:26.000000000 +0200
@@ -98,11 +98,6 @@
 #define F_DLL           (0x2000)
 
 /* Extra structure which is used in the optional header.  */
-typedef struct _IMAGE_DATA_DIRECTORY 
-{
-  bfd_vma VirtualAddress;
-  long    Size;
-}  IMAGE_DATA_DIRECTORY;
 #define PE_EXPORT_TABLE			0
 #define PE_IMPORT_TABLE			1
 #define PE_RESOURCE_TABLE		2
Only in include/coff: internal.h~
diff -ur include.org/libcoff.h include/libcoff.h
--- libcoff.h	2008-10-06 17:29:08.000000000 +0200
+++ libcoff.h	2008-10-06 17:33:28.000000000 +0200
@@ -256,8 +256,10 @@
   /* Symbol type.  */
   unsigned short type;
 
+#ifndef __cplusplus
   /* Symbol class.  */
   unsigned char class;
+#endif
 
   /* Number of auxiliary entries.  */
   char numaux;
@@ -396,8 +398,10 @@
   /* Next type with the same name.  */
   struct coff_debug_merge_type *next;
 
+#ifndef __cplusplus
   /* Class of type.  */
   int class;
+#endif
 
   /* Symbol index where this type is defined.  */
   long indx;
Only in include: libcoff.h~
EOF
) >> "$LOGFILE" 2>&1 || fail
    fi
  fi
fi

if test -z "$rebuild_boinc" && test -r "$INSTALL/lib/libboinc_api.a" ; then
    log_and_show "using installed boinc"
else
    log_and_show "compiling boinc"
    if [ ."$build_win32" = ."true" ] ; then
	log_and_do cd "$BUILD/boinc"
        makefile="$SOURCE/boinc/lib/Makefile.mingw"
	export BOINC_SRC="$SOURCE/boinc" BOINC_PREFIX="$INSTALL"
	log_and_dont_fail make -f "$makefile" uninstall
	log_and_do make -f "$makefile" clean
	if log_and_dont_fail make -f "$makefile" all-la; then
	    log_and_do make -f "$makefile" install-la
	else
            log_and_do make -f "$makefile"
	    log_and_do make -f "$makefile" install
        fi
        sed -i~ '/#include "boinc_win.h"/d' "$INSTALL/include/boinc/filesys.h"
    else
	log_and_do cd "$SOURCE/boinc"
	log_and_do ./_autosetup -f
	log_and_do cd "$BUILD/boinc"
	log_and_do "$SOURCE/boinc/configure" --disable-server --disable-manager --disable-client "$WITH_SSL" "$shared_copt" "$cross_copt" --prefix="$INSTALL" # --target=powerpc-apple-darwin7.9.0
	log_and_dont_fail make uninstall
	log_and_do make
	log_and_do make install
    fi
fi

lalsuite_copts="--disable-gcc-flags --without-hdf5 --disable-frame --disable-metaio --disable-lalsimulation --disable-lalxml --enable-boinc --disable-silent-rules --disable-pthread-lock $shared_copt $cross_copt --prefix=$INSTALL"
if [ ."$build_win32" = ."true" ] ; then
    export BOINC_EXTRA_LIBS="-lpsapi"
fi
if test -z "$rebuild_lal" && pkg-config --exists lal; then
    log_and_show "using existing lal"
else
    log_and_show "compiling LAL"
    log_and_do cd "$SOURCE/lalsuite/lal"
    log_and_do ./00boot
    log_and_do cd "$BUILD/lal"
    log_and_do "$SOURCE/lalsuite/lal/configure" $lalsuite_copts
    log_and_dont_fail make uninstall
    log_and_do make
    log_and_do make install
    log_and_do sed -i~ 's/.*typedef .* UINT8 *;.*/#define UINT8 uint64_t/;s/.*typedef .* INT8 *;.*/#define INT8 int64_t/' "$INSTALL/include/lal/LALAtomicDatatypes.h"
fi

if test -z "$rebuild_lal" && pkg-config --exists lalpulsar; then
    log_and_show "using existing lalpulsar"
else
    log_and_show "compiling LALPulsar"
    log_and_do cd "$SOURCE/lalsuite/lalpulsar"
    log_and_do ./00boot
    log_and_do cd "$BUILD/lalpulsar"
    log_and_do "$SOURCE/lalsuite/lalpulsar/configure" $lalsuite_copts
    log_and_dont_fail make uninstall
    log_and_do make
    log_and_do make install
fi

# work around a bug in current LALApps build
#log_and_do cp -R "$SOURCE/lalsuite/lalmetaio/src/LIGOMetadataTables.h" "$INSTALL/include/lal"

log_and_show "configuring LALApps"
log_and_do cd "$SOURCE/lalsuite/lalapps"
log_and_do ./00boot
if [ ."$build_win32" = ."true" ] ; then
    sed -i~ 's/test  *"${boinc}"  *=  *"true"/test "true" = "true"/' configure
fi
log_and_do cd "$BUILD/lalapps"
log_and_do "$SOURCE/lalsuite/lalapps/configure" $lalsuite_copts

log_and_show "building Apps"

log_and_do cd "$BUILD/lalapps/src/lalapps"
log_and_do make LALAppsVCSInfo.h LALAppsVCSInfoHeader.h liblalapps.la

log_and_do cd "$BUILD/lalapps/src/pulsar/GCT"
log_and_dont_fail make gitID
log_and_do make "eah_HierarchSearchGCT$ext"
log_and_do cp "eah_HierarchSearchGCT$ext" "$EAH/eah_HierarchSearchGCT$acc$ext"
test ".$release" = ".true" &&
    log_and_do cp "$EAH/eah_HierarchSearchGCT$acc$ext" "$EAH/${appname}_${appversion}_$platform$planclass$ext" &&
    log_and_do gzip -f "$EAH/${appname}_${appversion}_$platform$planclass$ext"

log_and_do cd "$BUILD/lalapps/src/pulsar/Injections"
log_and_do make eah_Makefakedata_v5$ext
log_and_do cp eah_Makefakedata_v5$ext "$EAH"
log_and_do cd "$BUILD/lalapps/src/pulsar/Fstatistic"
log_and_do make eah_PredictFstat$ext eah_ComputeFstatistic_v2$ext
log_and_do cp eah_PredictFstat$ext eah_ComputeFstatistic_v2$ext "$EAH"

log_and_show "==========================================="
log_and_show "Einstein@home App was built, find it at"
log_and_show "$EAH/${appname}_${appversion}_$platform$planclass${ext}.gz"
log_and_show "==========================================="

fi # check-only

if [ .$check = .true ]; then
    if [ -z "$check_app" ]; then
        check_app="../eah_HierarchSearchGCT$acc$ext"
    fi
    log_and_show "Running test"
    log_and_do cd "$EAH"
    log_and_do rm -rf Injections Fstatistic test
    log_and_do mkdir Injections
    log_and_do ln -s Injections Fstatistic
    log_and_do cd Injections
    log_and_do cp ../eah_Makefakedata_v5$ext lalapps_Makefakedata_v5
    log_and_do cp ../eah_PredictFstat$ext lalapps_PredictFstat
    log_and_do cp ../eah_ComputeFstatistic_v2$ext lalapps_ComputeFstatistic_v2
    LAL_DATA_PATH="$INSTALL/share/lalpulsar" DEBUG=1 NOCLEANUP=1 PATH="$PWD:$PATH" \
	log_and_do ../source/lalsuite/lalapps/src/pulsar/GCT/testGCT.sh $wine "$check_app"
    log_and_show "==========================================="
    log_and_show "Test passed"
    log_and_show "==========================================="
fi

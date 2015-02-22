#!/bin/bash

## set LAL debug level
echo "Setting LAL_DEBUG_LEVEL=${LAL_DEBUG_LEVEL:-msglvl1,memdbg}"
export LAL_DEBUG_LEVEL

## allow 'make test' to work from builddir != srcdir
if [ -z "${srcdir}" ]; then
    srcdir=`dirname $0`
fi

## make sure we work in 'C' locale here to avoid awk sillyness
LC_ALL_old=$LC_ALL
export LC_ALL=C

builddir="./";
injectdir="../Injections/"

## ----- user-controlled level of debug-output detail
if [ -n "$DEBUG" ]; then
    debug=${DEBUG}
else
    debug=0	## default=quiet
fi

## ----- allow user-control of hotloop variant to use
if [ -n "$FSTAT_METHOD" ]; then
    FstatMethod="--FstatMethod=${FSTAT_METHOD}"
fi

##---------- names of codes and input/output files
mfd_code="${injectdir}lalapps_Makefakedata_v4"
saf_code="${builddir}lalapps_SemiAnalyticF"
cmp_code="${builddir}lalapps_compareFstats"
## allow user to specify a different CFSv2 version to test by passing as cmdline-argument
if test $# -eq 0 ; then
    cfsv2_code="${builddir}lalapps_ComputeFstatistic_v2"
else
    cfsv2_code="$@"
fi

Dterms=8
# ---------- fixed parameter of our test-signal
Tsft=1800;
startTime=711595934
duration=144000		## 40 hours

mfd_FreqBand=2.0;

Alpha=2.0
Delta=-0.5

h0=1
cosi=-0.3
psi=0.6
phi0=1.5

Freq=100.12345
f1dot=-1e-10;

## mfd-specific bands
mfd_fmin=$(echo $Freq $mfd_FreqBand | awk '{printf "%g", $1 - $2 / 2.0}');

## cfs search bands
NFreq=500;
cfs_FreqBand=$(echo $duration | awk '{printf "%.16g", 1.0 / $1 }');	## fix band to 1/T so we're close to signal peak always
cfs_Freq=$(echo $Freq $cfs_FreqBand | awk '{printf "%.16g", $1 - $2 / 2.0}');
cfs_dFreq=$(echo $cfs_FreqBand $NFreq | awk '{printf "%.16g", $1 / $2 }');
cfs_nCands=$NFreq	## toplist length: keep all cands

cfs_f1dotBand=0;
cfs_f1dot=$(echo $f1dot $cfs_f1dotBand | awk '{printf "%.16g", $1 - $2 / 2.0}');
##Nf1dot=10
cfs_df1dot=1 ##$(echo $cfs_f1dotBand $Nf1dot | awk '{printf "%g", $1 / $2}');

noiseSqrtSh=5

## ------------------------------------------------------------

if [ "$noiseSqrtSh" != 0 ]; then
    sqrtSh=$noiseSqrtSh
    haveNoise=true;
else
    sqrtSh=1;	## for SemiAnalyticF signal-only case
    haveNoise=false;
fi

IFO=LHO

## ----- define output directory and files
testDir=testCFSv2.d
rm -rf $testDir
mkdir -p $testDir
SFTdir=${testDir}

outfile_ref=${srcdir}/testCFSv2.dat.ref.gz
outfile_Fstat=${testDir}/testCFSv2.dat
outfile_Loudest=${testDir}/Fstat_loudest.dat

##--------------------------------------------------
## test starts here
##--------------------------------------------------

echo
echo "----------------------------------------------------------------------"
echo " STEP 1: Generate Fake Signal"
echo "----------------------------------------------------------------------"
echo
# this part of the command-line is compatible with SemiAnalyticF:
saf_CL=" --Alpha=$Alpha --Delta=$Delta --IFO=$IFO --Tsft=$Tsft --startTime=$startTime --duration=$duration --h0=$h0 --cosi=$cosi --psi=$psi --phi0=$phi0"
# concatenate this with the mfd-specific switches:
mfd_CL="${saf_CL} --fmin=$mfd_fmin --Band=$mfd_FreqBand --Freq=$Freq --outSFTbname=$SFTdir/$IFO-sfts.sft --f1dot=$f1dot --outSingleSFT  --refTime=$startTime"
if [ "$haveNoise" = true ]; then
    mfd_CL="$mfd_CL --noiseSqrtSh=$sqrtSh";
fi

cmdline="$mfd_code $mfd_CL --randSeed=1"
echo $cmdline;
if ! eval "$cmdline 2> /dev/null"; then
    echo "Error.. something failed when running '$mfd_code' ..."
    exit 1
fi

echo
echo -n "Running '$saf_code' ... "
cmdline="$saf_code $saf_CL --sqrtSh=$sqrtSh"
echo $cmdline
if ! resF=`eval "$cmdline  2> /dev/null"`; then
    echo "Error ... something failed running '$saf_code' ..."
    exit 1;
fi
res2F=$(echo $resF | awk '{printf "%g", 2.0 * $1}')
echo "The SemiAnalyticF calculations predicts: 2F = $res2F"

echo
echo "----------------------------------------------------------------------"
echo "STEP 2: run CFS_v2 with perfect match"
echo "----------------------------------------------------------------------"
echo

cfs_CL="--IFO=$IFO --Alpha=$Alpha --Delta=$Delta --Freq=$cfs_Freq --dFreq=$cfs_dFreq --f1dot=$cfs_f1dot --f1dotBand=$cfs_f1dotBand --df1dot=$cfs_df1dot --DataFiles='$SFTdir/*.sft' --NumCandidatesToKeep=${cfs_nCands} --Dterms=${Dterms} --outputLoudest=${outfile_Loudest} ${FstatMethod} --refTime=$startTime"
if [ "$haveNoise" != "true" ]; then
    cfs_CL="$cfs_CL --SignalOnly"
fi

cmdline="$cfsv2_code $cfs_CL --outputFstat=${outfile_Fstat} --TwoFthreshold=0 --FreqBand=$cfs_FreqBand"
echo $cmdline;
if ! eval "$cmdline 2> /dev/null"; then
    echo "Error.. something failed when running '$cfs_code' ..."
    exit 1;
fi


echo
echo "----------------------------------------------------------------------"
echo " STEP 3: Compare to reference results: "
echo "----------------------------------------------------------------------"

## work around toplist-sorting bugs in CFSv2: manually sort before comparing
sort ${outfile_Fstat} > __tmp_sorted && mv __tmp_sorted ${outfile_Fstat}

echo
cmdline="$cmp_code -1 ./${outfile_Fstat} -2 ${outfile_ref}"
echo -n $cmdline
if ! eval $cmdline; then
    echo "==> OUCH... files differ. Something might be wrong..."
    exit 2
else
    echo "	==> OK."
fi
echo


echo
echo "----------------------------------------------------------------------"
echo " STEP 4: Sanity-check parameter estimation: "
echo "----------------------------------------------------------------------"
echo

if grep -q 'nan;' ${outfile_Loudest}; then
    echo "ERROR: ${outfile_Loudest} contains NaNs!"
    exit 2
fi

esth0=$(grep '^h0' ${outfile_Loudest} | awk -F '[ ;]*' '{print $3}')
estdh0=$(grep '^dh0' ${outfile_Loudest} | awk -F '[ ;]*' '{print $3}')

estPhi0=$(grep '^phi0' ${outfile_Loudest} | awk -F '[ ;]*' '{print $3}')
estdPhi0=$(grep '^dphi0' ${outfile_Loudest} | awk -F '[ ;]*' '{print $3}')

echo "Estimated h0 = $esth0 +/- $estdh0: Injected h0 = $h0"
h0inrange=$(echo $h0 $esth0 $estdh0 | awk '{printf "%i\n", (($1 - $2)^2 < $3^2)}')
if test x$h0inrange != x1; then
    echo "ERROR: estimated h0 was not within error of injected h0!"
    echo
    exit 2
else
    echo "OK: Estimated h0 is within error of injected h0"
    echo
fi

echo "Estimated phi0 = $estPhi0 +/- $estdPhi0: Injected phi0 = $phi0"
phi0inrange=$(echo $phi0 $estPhi0 $estdPhi0 | awk '{printf "%i\n", (($1 - $2)^2 < $3^2)}')
if test x$phi0inrange != x1; then
    echo "ERROR: estimated phi0 was not within error of injected phi0!"
    echo
    exit 2
else
    echo "OK: Estimated phi0 is within error of injected phi0"
    echo
fi


## -------------------------------------------
## clean up files
## -------------------------------------------
if [ -z "$NOCLEANUP" ]; then
    rm -rf $testDir
fi

## restore original locale, just in case someone source'd this file
export LC_ALL=$LC_ALL_old

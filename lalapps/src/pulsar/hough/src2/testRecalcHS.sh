#!/bin/sh

## test-script to test lalapps_RecalcHSCandidates by running it against lalapps_HierarchicalSearch (HS),
## namely by feeding it the HS output candidate-file as input and verifying that it correctly reproduces these
## candidates

## take user-arguments:
extra_args="$@"

## allow 'make test' to work from builddir != srcdir
if [ -n "${srcdir}" ]; then
    builddir="./";
    injectdir="../../Injections/"
else
    srcdir=.
fi

# test if LAL_DATA_PATH has been set ... needed to locate ephemeris-files
if [ -z "$LAL_DATA_PATH" ]; then
    if [ -n "$LALPULSAR_PREFIX" ]; then
	export LAL_DATA_PATH=".:${LALPULSAR_PREFIX}/share/lalpulsar";
    else
	echo
	echo "Need environment-variable LALPULSAR_PREFIX, or LAL_DATA_PATH to be set"
	echo "to your ephemeris-directory (e.g. /usr/local/share/lalpulsar)"
	echo "This might indicate an incomplete LAL+LALPULSAR installation"
	echo
	exit 1
    fi
fi

##---------- names of codes and input/output files
code_MFD="${injectdir}lalapps_Makefakedata_v4"
code_HS="${builddir}/lalapps_HierarchicalSearch"
code_RC="${builddir}/lalapps_RecalcHSCandidates"

## ----- parameters
SFTsH1="./allSFTs_H1.sft"
SFTsL1="./allSFTs_L1.sft"

HS_OUT="./HS_out.dat"
RC_OUT1="./RC_out1.dat"
RC_OUT2="./RC_out2.dat"

Freq0=100.5
dFreq0=1.3e-5
tStack=3600
nStacks=10
Tspan=36000	## should be >= nStacks * tStack

echo "----- STEP 1: produce some fake data:"
cmdline="$code_MFD -I H1 --outSingleSFT --outSFTbname=$SFTsH1 -y 05-09 -G 820108814 --duration=$Tspan --noiseSqrtSh=3e-23 --fmin=100 --Band=1 --randSeed=1 -v1"

echo $cmdline;
if ! eval $cmdline; then
    echo "Error.. something failed when running '$cmdline' ..."
    exit 1
fi

cmdline="$code_MFD -I L1 --outSingleSFT --outSFTbname=$SFTsL1 -y 05-09 -G 820108814 --duration=$Tspan --noiseSqrtSh=3e-23 --fmin=100 --Band=1 --randSeed=2 -v1"

echo $cmdline;
if ! eval $cmdline; then
    echo "Error.. something failed when running '$cmdline' ..."
    exit 1
fi


echo "----- STEP 2: '$HS_OUT' -- run original HierarchicalSearch code on this data, producing some candidates:"
cmdline="$code_HS --method=0 --DataFiles=\"$SFTsH1;$SFTsL1\" --skyRegion='(1,1)' --Freq=$Freq0 --FreqBand=1e-5 --dFreq=$dFreq0 --tStack=$tStack --nStacksMax=$nStacks --printCand1 --nf1dotRes=1 --semiCohToplist -o $HS_OUT -d1"

echo $cmdline;
if ! eval $cmdline; then
    echo "Error.. something failed when running '$cmdline' ..."
    exit 1
fi

echo "## ----- STEP 3a: '$RC_OUT1' -- run 'RecalcHSCandidates' on this output candidate file"
cmdline="$code_RC --DataFiles1=\"$SFTsH1;$SFTsL1\" --tStack=$tStack --nStacksMax=$nStacks --followupList=$HS_OUT --WU_Freq=$Freq0 --WU_dFreq=$dFreq0 --fnameout=$RC_OUT1 --outputFX=true -d1"

echo $cmdline;
if ! eval $cmdline; then
    echo "Error.. something failed when running '$cmdline' ..."
    exit 1
fi

echo "## ----- STEP 3b: '$RC_OUT2' -- re-run RecalcHSCandidates on its own output to test stability"
## NOTE: this time we don't use any HS-frequency bug corrections (--WU_Freq), as RC outputs frequencies correctly!
cmdline="$code_RC --DataFiles1=\"$SFTsH1;$SFTsL1\" --tStack=$tStack --nStacksMax=$nStacks --followupList=$RC_OUT1 --WU_dFreq=$dFreq0 --fnameout=$RC_OUT2 --outputFX=true -d1"

echo $cmdline;
if ! eval $cmdline; then
    echo "Error.. something failed when running '$cmdline' ..."
    exit 1
fi

## currently not really testing these results against HS, just output them for user to look at:
echo "$HS_OUT: --------------------"
cat $HS_OUT
echo "$RC_OUT1: --------------------"
cat $RC_OUT1


if ! diff -q RC_out1.dat RC_out2.dat ; then
    echo "$RC_OUT1 and RC_OUT2 differ ... something is wrong!"
    exit 1;
else
    echo "$RC_OUT1 and RC_OUT2 are identical ... OK!"
fi

if [ -z "$NOCLEANUP" ]; then
    rm $HS_OUT $RC_OUT1 $RC_OUT2 $SFTsH1 $SFTsL1
fi

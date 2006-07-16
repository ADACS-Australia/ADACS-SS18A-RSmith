#!/usr/bin/env tclsh

set PARAMS {
	IFO "H1"
	DETECTOR "H"
	RUN_NAME "fake_random.12"
	ROOT_DIR "/archive/home/volodya/runs/$RUN_NAME/"
	CONF_DIR "$ROOT_DIR/in/"
	OUTPUT_DIR "$ROOT_DIR/output/"
	STORAGE_NODE_LIST {5 6 7 8 9 10 11 12 13 14 15 16 17 18}
	SFT_OUTPUT_DIR_TEMPLATE "/data/node\$nodenum/volodya/$RUN_NAME/"
	ERR_DIR "$ROOT_DIR/err/"
	STATS_DIR "$ROOT_DIR/stats"
	STATS_SUFFIX ".$IFO"
	LOG_FILE "/usr1/volodya/$RUN_NAME.log"
	FREQ_START  150.0
	FREQ_STOP   250.0
	FREQ_STEP    0.25
	INJECTIONS_PER_BAND  20
	DEC_RANGE  { -1.57 1.57 }
	RA_RANGE { 0 6.283 }
	PSI_RANGE { 0 3.1415 }
	PHI_RANGE { 0 3.1415 }
	IOTA_RANGE {0 3.1415 }
	SPINDOWN_LOG10_RANGE { -1.5 0}
	SPINDOWN_MAX 2e-9
	POWER_LOG10_RANGE { -1.5 0}
	POWER_MAX 4e-23
	INJECTION_DENSITY {cos(\$dec)}
	INJECTION_DENSITY_MAX 1.0
	SFT_INPUT "/archive/home/volodya/SFT-3/S4.${IFO}.ht.geo/sft.${IFO}."
	ANALYSIS_PROGRAM "/archive/home/volodya/PowerFlux/powerflux"
	}

set PARAMS_FORMAT { var value }	

set POWERFLUX_CONF_FILE {
label $i
input ${INJ_SFT_DIR}/sft.${IFO}.
input-format GEO
detector L${DETECTOR}O
earth-ephemeris /archive/home/volodya/detresponse/earth05.dat
sun-ephemeris /archive/home/volodya/detresponse/sun05.dat
output $OUTPUT_DIR/$i
first-bin $first_bin
nbins 501
do-cutoff 1
three-bins 0
filter-lines 1
write-dat NONE
write-png NONE
focus-ra $ra
focus-dec $dec
focus-radius 0.1
subtract-background 1
ks-test 1
	}
	
	

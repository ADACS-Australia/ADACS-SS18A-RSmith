#!/usr/bin/env tclsh

set PARAMS { 
	IFO "H1"
	DETECTOR "H"
	RUN_NAME "S3.$IFO.run.short.7"
	ROOT_DIR "/scratch4/volodya/$RUN_NAME/"
	CONF_DIR "$ROOT_DIR/in/"
	OUTPUT_DIR "$ROOT_DIR/output/"
	ERR_DIR "$ROOT_DIR/err/"
	STATS_DIR "$ROOT_DIR/stats"
	STATS_SUFFIX ".$IFO"
	DAG_LOG "/people/volodya/$RUN_NAME.log"
	FREQ_START	50
	FREQ_STEP	0.25
	FREQ_END	250
	FIRST_SPINDOWN  -1.00e-8
	LAST_SPINDOWN  1.00e-8
	SPINDOWN_STEP   5e-10
	MAX_SPINDOWN_COUNT {10*400*400/(\$band*\$band)}
	SKYMARKS {/home/volodya/PowerFlux/sky_marks.txt}
	NBANDS	10
	ANALYSIS_PROGRAM "/home/volodya/PowerFlux/powerflux"
	}

set PARAMS_FORMAT { var value }	

set POWERFLUX_CONF_FILE {
label run $RUN_NAME
input /scratch4/volodya/SFT-3/S3.${IFO}.ht.geo/sft.${IFO}.
input-format GEO
detector L${DETECTOR}O
averaging-mode one
sky-marks $skymarks
spindown-start $spindown_start
spindown-step $SPINDOWN_STEP
spindown-count $spindown_count
ephemeris-path /home/volodya/detresponse
first-bin $firstbin
nbins 501
do-cutoff 1
three-bins 0
filter-lines 1
write-dat NONE
write-png NONE
nbands	$NBANDS
subtract-background 1
ks-test 1
compute-betas 0
output $OUTPUT_DIR/$i
        }


/*
*  Copyright (C) 2007 Vladimir Dergachev
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with with program; see the file COPYING. If not, write to the
*  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*  MA  02111-1307  USA
*/

/**
 * \file
 * \ingroup pulsarApps
 * \author Vladimir Dergachev
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

#define __USE_ISOC99 1
#include <math.h>

#include <lal/LALStdlib.h>
#include <lal/AVFactories.h>
#include <lal/RealFFT.h>
#include <lal/Window.h>
#include <lal/BandPassTimeSeries.h>

#include "make_sft_global.h"
#include "op_method.h"

#include <FrameL.h>

extern long total_samples;
extern long samples_per_second;
extern float *data;
extern INT64 gps_start, gps_end;
extern char * channel;
extern int precision;


int output_mode=OUTPUT_MODE_TEXT;
long freq_start=-1, freq_stop=-1;

#define BUF_SIZE 200000
char buffer[BUF_SIZE];

/* options to bypass parts of processing */
char bypass_highpass_filter=0, bypass_lowpass_filter=0, bypass_first_window=0;

/* Constants for Butterworth filter */
REAL4 highpass_filter_f=100.0, highpass_filter_a=0.5, lowpass_filter_f=-1, lowpass_filter_a=-1.0;
int highpass_filter_order=5, lowpass_filter_order=-1;

/* filename to output results */
char *output_power=NULL, *output_calibrated_data=NULL, *output_sft=NULL;

/* should we trace total power during computation ? */
char trace_power=0, trace_calibration_power=0;

/* filenames to write debug data. NULL if no debugging has been requested */
char *file_debug1=NULL, *file_debug2=NULL;

MAKE_SFT_WINDOW window1={
	type: Hann, 
	tail_size: 512,
	debug: NULL
	};

MAKE_SFT_WINDOW window2={
	type: Hann, 
	tail_size: 512,
	debug: NULL
	};

REAL4Vector *td_data=NULL, *power=NULL;
COMPLEX8Vector *phi[3]={NULL,NULL,NULL};
COMPLEX8Vector *pgram=NULL;

void print_settings(FILE *f)
{
fprintf(f,"#    starting bin:	    %ld\n", freq_start);
fprintf(f,"#    stopping bin:       %ld\n", freq_stop);
fprintf(f, "#    highpass filter N: %d\n", highpass_filter_order);
fprintf(f, "#    highpass filter F: %.*g\n", precision, highpass_filter_f);
fprintf(f, "#    highpass filter A: %.*g\n", precision, highpass_filter_a);
fprintf(f, "#    lowpass filter N: %d\n", lowpass_filter_order);
fprintf(f, "#    lowpass filter F: %.*g\n", precision, lowpass_filter_f);
fprintf(f, "#    lowpass filter A: %.*g\n", precision, lowpass_filter_a);
print_window_settings(f, "first", &window1);
print_fake_data_settings(f);
}

void set_debug_output(char *p)
{
switch(p[5]){
	case '1':
		if(file_debug1!=NULL)free(file_debug1);
		file_debug1=dup_quoted_name(p);
		break;
	case '2':
		if(file_debug2!=NULL)free(file_debug2);
		file_debug2=dup_quoted_name(p);
		break;
	default:
		fprintf(stderr,"** Could not parse statement \"%s\"\n", p);
		exit(-1);
	}
}

int main(int argc, char *argv[])
{
long i;
double window_sum;
LALStatus status={level:0, statusPtr: NULL};
PassBandParamStruc filterpar;
REAL4TimeSeries chan; /* must zero the f0 field */

fprintf(stderr,"make_sft_op version %s\n", MAKE_SFT_VERSION);
fprintf(stderr,"Using frame library %s\n", FrLibVersionF());
fprintf(stderr,"Using LAL version %s\n", LAL_VERSION);

lalDebugLevel=0;

yylex();

if(freq_start<0)freq_start=0;
if(freq_stop<0)freq_stop=total_samples;

/* post_init various subsystems */

post_init_response_files();
post_init_alpha_beta();

/* print settings */
print_settings(stderr);

/* start processing data */
print_data_stats();
print_data(file_debug1, "debug 1");

verify_loaded_data();

generate_fake_data();

linear_interpolate_gaps();
print_data(file_debug2, "debug 2");

/* this is a hack, but it significantly reduces memory footprint */
td_data=do_alloc(1,sizeof(*td_data));
td_data->length=total_samples;
td_data->data=data;

/* setup structure to hold input for Butterworth filter */
chan.data=NULL;
strncpy(chan.name,channel,LALNameLength);
chan.f0=0;
chan.name[LALNameLength-1]=0; /* make sure it is null-terminated */
chan.deltaT=1.0/samples_per_second;
chan.epoch.gpsSeconds=gps_start; /* no need */
chan.epoch.gpsNanoSeconds=0;

if(trace_power){
	fprintf(stderr, "Input data total power=%.*g\n",
		precision, sum_r4_squares(data, total_samples)/samples_per_second);
	}

if(!bypass_highpass_filter && (highpass_filter_f>0) && (highpass_filter_a>0)){
	fprintf(stderr,"Applying high pass filter, f=%g a=%g order=%d\n",
		highpass_filter_f, highpass_filter_a, highpass_filter_order);
	/* Setup Butterworth filter */
	filterpar.name="Butterworth High Pass";
	filterpar.nMax=highpass_filter_order;
	filterpar.f2=highpass_filter_f;
	filterpar.a2=highpass_filter_a;
	/* values that are 'not given' = out of range */
	filterpar.f1=-1.0;
	filterpar.a1=-1.0;
	
	/* REAL4Sequence is the same as REAL4Vector - according to lal/Datatypes.h */
	chan.data=(REAL4Sequence *)td_data;
	LALDButterworthREAL4TimeSeries(&status, &chan, &filterpar);
	TESTSTATUS(&status);

	if(trace_power){
		fprintf(stderr, "After highpass filter total power=%.*g\n",
			precision, sum_r4_squares(data, total_samples)/samples_per_second);
		}
	}
	
if(!bypass_lowpass_filter && (lowpass_filter_f>0) && (lowpass_filter_a > 0)){
	fprintf(stderr,"Applying low pass filter, f=%g a=%g order=%d\n",
		lowpass_filter_f, lowpass_filter_a, lowpass_filter_order);
	/* Setup Butterworth filter */
	filterpar.name="Butterworth Low Pass";
	filterpar.nMax=lowpass_filter_order;
	filterpar.f1=lowpass_filter_f;
	filterpar.a1=lowpass_filter_a;
	/* values that are 'not given' = out of range */
	/* why 2*nsamples ? LAL used to have a bug in it where
	   the pairs were sorted before deciding whether the filter 
	   is lowpass or highpass. Therefore, we need to specify a 
	   large, out of range, frequency f so that we get a low-pass filter.
	   The maximum frequency is Nyquist, hence 2*nsamples is definitely out of range */
	filterpar.f2=2*total_samples;
	filterpar.a2=-1.0;
		/* REAL4Sequence is the same as REAL4Vector - according to lal/Datatypes.h */
	chan.data=(REAL4Sequence *)td_data;
	LALDButterworthREAL4TimeSeries(&status, &chan, &filterpar);
	TESTSTATUS(&status);

	if(trace_power){
		fprintf(stderr, "After lowpass filter total power=%.*g\n",
			precision, sum_r4_squares(data, total_samples)/samples_per_second);
		}
	}

if(!bypass_first_window){
	fprintf(stderr,"Applying Hann window to input data\n");
	for(i=0;i<total_samples;i++)data[i]*=0.5*(1.0-cos((2.0*M_PI*i)/total_samples));
	window_sum=0.5;
	} else {
	window_sum=1.0;
	}

if(trace_power){
	fprintf(stderr, "After windowing total power=%.*g\n",
		precision, sum_r4_squares(data, total_samples)/samples_per_second);
	}

for(i=0;i<3;i++){
	fprintf(stderr,"Allocating phi[%ld]\n", i);
	LALCCreateVector(&status, &(phi[i]), freq_stop-freq_start);
	TESTSTATUS(&status);
	}

compute_test_fft(td_data, phi, 3, freq_start, freq_stop);

/* now free td_data to conserve space */
free(td_data->data);
data=NULL;
free(td_data);
td_data=NULL;

LALCCreateVector(&status, &pgram, freq_stop-freq_start);
TESTSTATUS(&status);

compute_calibrated_periodogram3(phi, freq_start, freq_stop, pgram, window_sum);

print_COMPLEX8Vector(pgram, output_sft, "CALIBRATED FREQUENCY DOMAIN DATA", output_mode, 0, freq_stop-freq_start);

if(output_power!=NULL){
	/* we need more space */
	for(i=0;i<3;i++){
		LALCDestroyVector(&status, &(phi[i]));
		TESTSTATUS(&status);
		}
	LALSCreateVector(&status, &power, freq_stop-freq_start);
	TESTSTATUS(&status);
	for(i=0;i<freq_stop-freq_start;i++)
		power->data[i]=(pgram->data[i].re*pgram->data[i].re+pgram->data[i].im*pgram->data[i].im);
	print_REAL4Vector(power, output_power, "CALIBRATED POWER", output_mode, 0, freq_stop-freq_start);
	}

/* we do not destroy large vectors when we are done unless we need
  to allocate a lot of space again. This reduces run time 
  of the program */
return 0;
}

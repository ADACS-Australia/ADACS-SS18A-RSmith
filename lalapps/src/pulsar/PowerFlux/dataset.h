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

#ifndef __DATASET_H__
#define __DATASET_H__

#include "global.h"
#include "polarization.h"
#include "lines.h"
#include "intervals.h"

typedef struct {
	/* name of the data set - for logging purposes */
	char *name; 
	char *lock_file;
	int validated; /* has this data set been validated ? see corresponding functions */

	/* these describe which time periods we are interested in */
	INTERVAL_SET *segment_list;
	INTERVAL_SET *veto_segment_list;

	/* detector this dataset was observed on */
	char *detector;

	INT64 gps_start;
	INT64 gps_stop;

	INT64 *gps;

	/* real and imaginary parts - they are separate to facilitate use of vectorized operations */
	float *re; 
	float *im;

	/* precomputed power */
	float *power; 

	int size;
	int free;	/* this used to be called nsegments */

	int offset;	/* this points to our place in global coefficient tables */
	int first_bin;  /* first bin of each segment */
	int nbins;	/* number of bins in each segment */
	double coherence_time; /* how long the SFTs are */

	/* statistics and data derived from this data set */
	double *mean;
	double *weighted_mean;
	double *sigma;
	double *bin_weight;

	double *new_mean;
	double *new_weighted_mean;
	double *new_sigma;
	double *new_bin_weight;

	LINES_REPORT *lines_report;

	float *detector_velocity;
	double average_detector_velocity[3];
	double band_axis[3];
	double band_axis_norm;
	double large_S;

	int input_flags;
	#define FLAG_BAND_AXIS		1
	#define FLAG_BAND_AXIS_NORM	2
	#define FLAG_LARGE_S		4
	double user_band_axis[3];
	double user_band_axis_norm;
	double user_large_S;

	double weight; /* additional, user-specified weight factor */

	float *TMedians;
	float *FMedians;
	float *expTMedians;
	float TMedian;
	float expTMedian;

	/* helper arrays, for plotting */
	float *hours;
	float *frequencies;
	double *hours_d;
	double *freq_d;

	SKY_GRID_TYPE *AM_coeffs_plus;
	SKY_GRID_TYPE *AM_coeffs_cross;
	long AM_coeffs_size;

	POLARIZATION *polarizations;
	} DATASET;

void output_dataset_info(DATASET *d);
void characterize_dataset(DATASET *d);
void compute_noise_curves(DATASET *dataset);

void load_dataset_from_file(char *file);
long total_segments(void);
float datasets_normalizing_weight(void);
INT64 min_gps(void);
INT64 max_gps(void);
void post_init_datasets(void);
void datasets_average_detector_speed(double *average_det_velocity);
float effective_weight_ratio(float target_ra, float target_dec, float source_ra, float source_dec, float bin_tolerance, float spindown_tolerance);
float stationary_effective_weight_ratio(float target_ra, float target_dec, float bin_tolerance);
void dump_datasets(char *filename);
void test_datasets(void);

#define PHASE_ACCUMULATION_TIMEOUT 50000

#endif 

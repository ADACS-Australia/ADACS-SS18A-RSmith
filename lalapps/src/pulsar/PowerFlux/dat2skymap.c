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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "rastermagic.h"
#include "grid.h"

float resolution;
SKY_GRID *grid=NULL;
RGBPic *pic=NULL;
float *data=NULL;
double spindown=0;

double band_axis_norm=1;

/* bunch of variables that we don't need, but rastermagic and grid want */
FILE *LOG=NULL;
double band_axis[3]={1.0, 0.0, 0.0};
char *output_dir="";
char args_info[1024*65]; /* dummy */

void *stationary_effective_weight_ratio=NULL;
void *effective_weight_ratio=NULL;

void *do_alloc(long a, long b)
{
void *r;
int i=0;
r=calloc(a,b);
while(r==NULL){
        fprintf(stderr,"Could not allocate %ld chunks of %ld bytes each (%ld bytes total)\n",a,b,a*b);
        if(i>10)exit(-1);
        sleep(10);
        r=calloc(a,b);
        i++;
        }
return r;
}

int clear_name_png(char *name)
{
return 1;
}

int main(int argc, char *argv[])
{
FILE *fin;
int i;
float a;

LOG=stderr;

if(argc<7){
	fprintf(stderr, "\nUsage: %s mode resolution lower_cut upper_cut file.dat skymap.png\n", argv[0]);
	fprintf(stderr, "\tmodes: plain, log10\n");
	fprintf(stderr, "\tcutoff: number or \"auto\"\n\n");
	return -1;
	}
resolution=atof(argv[2]);

fin=fopen(argv[5], "r");
if(fin==NULL){
	perror("Cannot read file:");
	return -1;
	}

grid=make_sin_theta_grid(resolution);

data=do_alloc(grid->npoints, sizeof(*data));
fread(data, sizeof(*data), grid->npoints, fin);
fprintf(stderr, "%g %g %g %g\n", data[0], data[1], data[2], data[3]);

if(!strcasecmp(argv[1], "log10")){
	for(i=0;i<grid->npoints;i++)data[i]=log10(data[i]);
	}

if(strcasecmp(argv[3], "auto")){
	a=atof(argv[3]);
	for(i=0;i<grid->npoints;i++)
		if(data[i]<=a)data[i]=a;
	}

if(strcasecmp(argv[4], "auto")){
	a=atof(argv[4]);
	for(i=0;i<grid->npoints;i++)
		if(data[i]>=a)data[i]=a;
	}

pic=make_RGBPic(grid->max_n_ra+140, grid->max_n_dec);
plot_grid_f(pic, grid, data, 1);

RGBPic_dump_png(argv[6], pic);
return 0;
}

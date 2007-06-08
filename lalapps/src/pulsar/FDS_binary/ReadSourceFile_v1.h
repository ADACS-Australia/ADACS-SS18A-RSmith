/*
*  Copyright (C) 2007 Chris Messenger
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

#ifndef _READSOURCE_H
#define _READSOURCE_H

#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <getopt.h>
#include <lal/LALDatatypes.h>
#include <lal/Date.h>
#include <lal/LALStdio.h>
#include <lal/FileIO.h>
#include <lal/LALStdlib.h>

typedef struct skypositiontag {
  REAL8 ra;
  REAL8 dec;
  REAL8 ra_min;
  REAL8 ra_max;
  REAL8 dec_min;
  REAL8 dec_max;
  REAL8 ra_err;
  REAL8 dec_err;
} skyposition;

typedef struct frequencytag {
  INT4 nband;
  REAL8 *f_min;
  REAL8 *f_max;
  REAL8 *f_err;
} frequency;

typedef struct orbitaltag {
  REAL8 period;
  REAL8 period_err;
  REAL8 period_min;
  REAL8 period_max;
  REAL8 sma;
  REAL8 sma_min;
  REAL8 sma_max;
  REAL8 sma_err;
  LIGOTimeGPS tperi;
  LIGOTimeGPS tperi_min;
  LIGOTimeGPS tperi_max;
  REAL8 tperi_err;
  REAL8 argp;
  REAL8 argp_min;
  REAL8 argp_max;
  REAL8 argp_err;
  REAL8 ecc;
  REAL8 ecc_min;
  REAL8 ecc_max;
  REAL8 ecc_err;
} orbital;

typedef struct binarysourcetag {
  char name[256];
  skyposition skypos;
  frequency freq;
  orbital orbit;
} binarysource;

int ReadSource(char *, char *, LIGOTimeGPS *, binarysource *);

#endif

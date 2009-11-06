/*
*  Copyright (C) 2007 David Chin, Jolien Creighton
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
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <XLALLeapSeconds.h>

INT4 lalDebugLevel = 0;

NRCSID(LALTESTLEAPSECSC, "$Id$");

static int do_test(int gpssec, int tai_utc_before, int tai_utc_after)
{
	int result = 0;
	int lal_tai_utc;

	if(lalDebugLevel > 2)
		printf("TestLeapSecs: BEFORE LEAP SECOND ADDED\n");
	lal_tai_utc = XLALLeapSeconds(gpssec - 1);
	if(lalDebugLevel > 0)
		printf("\tGPS = %9d;    TAI-UTC = %d\n", gpssec - 1, lal_tai_utc);

	if(XLALGetBaseErrno() && lalDebugLevel > 0) {
		XLAL_PERROR("do_test()");
		result = -1;
	}
	if(lal_tai_utc != tai_utc_before) {
		if(lalDebugLevel > 0)
			fprintf(stderr, "TestLeapSecs: XLALLeapSeconds() returned wrong value: expected %d, got %d\n", tai_utc_before, lal_tai_utc);
		result = -1;
	}

	if(lalDebugLevel > 2)
		printf("TestLeapSecs: AFTER LEAP SECOND ADDED\n");
	lal_tai_utc = XLALLeapSeconds(gpssec);
	if(lalDebugLevel > 0)
		printf("\tGPS = %9d;    TAI-UTC = %d\n\n", gpssec, lal_tai_utc);

	if(XLALGetBaseErrno() && lalDebugLevel > 0) {
		XLAL_PERROR("do_test()");
		result = -1;
	}
	if(lal_tai_utc != tai_utc_after) {
		if(lalDebugLevel > 0)
			fprintf(stderr, "TestLeapSecs: XLALLeapSeconds() returned wrong value: expected %d, got %d\n", tai_utc_after, lal_tai_utc);
		result = -1;
	}
	return result;
}


int main(int argc, char *argv[])
{
	int i;

	if(argc > 1)
		lalDebugLevel = atoi(argv[1]);

	for(i = 1; i < numleaps; i++)
		do_test(leaps[i].gpssec, leaps[i-1].taiutc, leaps[i].taiutc);

	return 0;
}

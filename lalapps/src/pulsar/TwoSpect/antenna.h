/*
*  Copyright (C) 2010 Evan Goetz
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

#ifndef __ANTENNA_H__
#define __ANTENNA_H__

#include <lal/LALStdlib.h>
#include <lal/AVFactories.h>
#include <lal/DetResponse.h>
#include <lal/LALInitBarycenter.h>
#include <lal/Velocity.h>

EphemerisData * new_Ephemeris(CHAR *earth_ephemeris, CHAR *sun_ephemeris);
void free_Ephemeris(EphemerisData *ephemdata);
void initEphemeris(EphemerisData *ephemdata);

void CompBinShifts(INT4Vector *out, REAL8 freq, REAL8Vector *velocities, REAL8 Tcoh, REAL4 dopplerMultiplier);
void CompAntennaPatternWeights(REAL8Vector *out, REAL4 ra, REAL4 dec, REAL8 t0, REAL8 Tcoh, REAL8 SFToverlap, REAL8 Tobs, LALDetector det);
void CompAntennaVelocity(REAL8Vector *out, REAL4 ra, REAL4 dec, REAL8 t0, REAL8 Tcoh, REAL8 SFToverlap, REAL8 Tobs, LALDetector det, EphemerisData *edat);

REAL4 CompDetectorDeltaVmax(REAL8 t0, REAL8 Tcoh, REAL8 SFToverlap, REAL8 Tobs, LALDetector det, EphemerisData *edat);

#endif


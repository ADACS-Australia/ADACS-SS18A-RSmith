/*
*  Copyright (C) 2007 Curt Cutler, Jolien Creighton, Reinhard Prix, Teviet Creighton
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
 * \author Curt Cutler
 * \date 2001
 * \file
 * \ingroup moduleBarycenter
 * \brief This header defines the API for LALBarycenter.c.
 *
 * <tt>#include <lal/LALBarycenter.h></tt>
 *
 */

#ifndef _LALBARYCENTER_H    /* Protect against double-inclusion */
#define _LALBARYCENTER_H

#include <stdio.h>
#include <math.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/DetectorSite.h>

#ifdef  __cplusplus
extern "C" {
#endif

NRCSID (LALBARYCENTERH,"$Id$");

/** \name Error codes */
/*@{*/
#define LALBARYCENTERH_ENULL  2
#define LALBARYCENTERH_EOUTOFRANGEE  4
#define LALBARYCENTERH_EOUTOFRANGES  8
#define LALBARYCENTERH_EBADSOURCEPOS 16

#define LALBARYCENTERH_MSGENULL  "Null input to Barycenter routine."
#define LALBARYCENTERH_MSGEOUTOFRANGEE  "tgps not in range of earth.dat file"
#define LALBARYCENTERH_MSGEOUTOFRANGES  "tgps not in range of sun.dat file"
#define LALBARYCENTERH_MSGEBADSOURCEPOS "source position not in standard range"
/*@}*/

/** This structure contains two pointers to data files containing arrays
 * of center-of-mass positions for the Earth and Sun, respectively.
 * The tables are derived from the JPL ephemeris.
 *
 * Files tabulate positions for one calendar year
 * (actually, a little more than one year, to deal
 * with overlaps).  The first line of each table summarizes
 * what is in it. Subsequent lines give the time (GPS) and the
 * Earth's position \f$(x,y,z)\f$,
 * velocity \f$(v_x, v_y, v_z)\f$, and acceleration \f$(a_x, a_y, a_z)\f$
 * at that instant.  All in units of seconds; e.g. positions have
 * units of seconds, and accelerations have units 1/sec.
 */
typedef struct
{
  CHAR *earthEphemeris;         /**< File containing Earth's position.  */
  CHAR *sunEphemeris;           /**< File containing Sun's position. */
}
EphemerisFilenames;

/** Structure holding a REAL8 time, and a position, velocity and
 * acceleration vector. */
typedef struct
{
  REAL8 gps;            /**< REAL8 timestamp */
  REAL8 pos[3];         /**< position-vector */
  REAL8 vel[3];         /**< velocity-vector */
  REAL8 acc[3];         /**< acceleration-vector */
}
PosVelAcc;

/** This structure contains all information about the
 * center-of-mass positions of the Earth and Sun, listed at regular
 * time intervals.
 */
typedef struct
{
  EphemerisFilenames ephiles; /**< Names of the two files containing positions of
                               * Earth and Sun, respectively at evenly spaced times. */
  INT4  nentriesE;      /**< The number of entries in Earth ephemeris table. */
  INT4  nentriesS;      /**< The number of entries in Sun ephemeris table. */
  REAL8 dtEtable;       /**< The spacing in sec between consecutive intants in Earth ephemeris table.*/
  REAL8 dtStable;       /**< The spacing in sec between consecutive intants in Sun ephemeris table.*/
  PosVelAcc *ephemE;    /**< Array containing pos,vel,acc of earth, as extracted from earth
                         * ephem file. Units are sec, 1, 1/sec respectively */
  PosVelAcc *ephemS;    /**< Array with pos, vel and acc for the sun (see ephemE) */
}
EphemerisData;

/** Basic output structure of LALBarycenterEarth.c.
 */
typedef struct
{
  REAL8  einstein;      /**<  the einstein delay equiv TDB - TDT */
  REAL8 deinstein;      /**< d(einstein)/d(tgps) */

  REAL8 posNow[3];      /**< Cartesian coords of Earth's center at tgps,
                         * extrapolated from JPL DE405 ephemeris; units= sec */
  REAL8 velNow[3];      /**< dimensionless velocity of Earth's center at tgps,
                         * extrapolated from JPL DE405 ephemeris */

  REAL8 gmstRad;        /**< Greenwich Mean Sidereal Time (GMST) in radians, at tgps */
  REAL8 gastRad;        /**< Greenwich Apparent Sidereal Time, in radians, at tgps;
                         * Is basically the angle thru which Earth has spun at
                         * given time; gast is like gmst, but has
                         * additional correction for short-term nutation */

  REAL8 tzeA;           /**< variable describing effect of lunisolar precession, at tgps */
  REAL8 zA;             /**< variable describing effect of lunisolar precession, at tgps */
  REAL8 thetaA;         /**< variable describing effect of lunisolar precession, at tgps */
  REAL8 delpsi;         /**< variable describing effect of Earth nutation, at tgps*/
  REAL8 deleps;         /**< variable describing effect of Earth nutation, at tgps*/

  REAL8 se[3];          /**< vector that points from Sun to Earth at instant tgps,
                         * in DE405 coords; units = sec */
  REAL8 dse[3];         /**< d(se[3])/d(tgps); Dimensionless */
  REAL8 rse;            /**< length of vector se[3]; units = sec */
  REAL8 drse;           /**< d(rse)/d(tgps); dimensionless */
}
EarthState;

/** Basic input structure to LALBarycenter.c.
 */
typedef struct
{
  LIGOTimeGPS  tgps;    /**< input GPS arrival time. I use tgps (lower case)
                         * to remind that here the LAL structure is a
                         * field in the larger structure BarycenterInput.
                         * I use tGPS as an input structure (by itself) to
                         * LALBarycenterEarth */

  LALDetector site;     /**< detector site info.  <b>NOTE:</b>
                         * the <tt>site.location</tt> field must be modified
                         * to give the detector location in units of
                         * <em>seconds</em> (i.e. divide the values normally
                         * stored in <tt>site.location</tt> by <tt>LAL_C_SI</tt> */

  REAL8 alpha;          /**<  Source right ascension in ICRS J2000 coords (radians). */
  REAL8 delta;          /**< Source declination in ICRS J2000 coords (radians) */
  REAL8 dInv;           /**< 1/(distance to source), in 1/sec.
                         * This is needed to correct Roemer delay for very
                         * nearby sources; correction is about 10 microsec for
                         * source at 100 pc */
}
BarycenterInput;

/**  Basic output structure produced by LALBarycenter.c.
 */
typedef struct
{
  REAL8 deltaT;         /**< \f$t_e\f$(TDB) - \f$t_a\f$(GPS)
                         * + (light-travel-time from source to SSB) */

  LIGOTimeGPS te;       /**< pulse emission time (TDB); also sometimes called
                         * ``arrival time (TDB) of same wavefront at SSB'' */
  REAL8 tDot;           /**< d(emission time in TDB)/d(arrival time in GPS)  */

  REAL8 rDetector[3];   /**< Cartesian coords (0=x,1=y,2=z) of detector position
                         * at $t_a$ (GPS), in ICRS J2000 coords. Units = sec. */

  REAL8 vDetector[3];   /* Cartesian coords (0=x,1=y,2=z) of detector velocity
                         * at \f$t_a\f$ (GPS), in ICRS J2000 coords. Dimensionless. */
}
EmissionTime;


/*Curt: probably best to take 1.0 OUT of tDot--ie., output tDot-1.
But most users would immediately add back the one anyway.
*/

/*Curt: rem te is ``time pulse would arrive at a GPS clock
way out in empty space, if you renormalized  and zero-ed the latter
to give, on average, the same arrival time as the GPS clock on Earth'' */


/* Function prototypes. */

void LALBarycenterEarth(LALStatus *, EarthState *, const LIGOTimeGPS *, const EphemerisData *);

void LALBarycenter(LALStatus *, EmissionTime *, const BarycenterInput *, const EarthState *);

#ifdef  __cplusplus
}
#endif      /* Close C++ protection */

#endif      /* Close double-include protection */

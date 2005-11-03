/*
 * Copyright (C) 2004, 2005 Reinhard Prix
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
 * \author Reinhard Prix
 * \date 2005
 * \file 
 * \ingroup pulsarCommon
 * \brief Some common useful data-types for pulsar-searches.
 *
 * $Id$
 *
 */

#ifndef _PULSARDATATYPES_H  /* Double-include protection. */
#define _PULSARDATATYPES_H

#include <lal/LALDatatypes.h>
#include <lal/DetectorSite.h>
#include <lal/Date.h>
#include <lal/SkyCoordinates.h>

#include "SFTutils.h"

/* C++ protection. */
#ifdef  __cplusplus
extern "C" {
#endif

NRCSID( PULSARDATATYPESH, "$Id$");

/** Type defining the parameters of a pulsar-source of Gravitational waves */
typedef struct {
  LIGOTimeGPS tRef;	/**< reference time of pulsar parameters (in SSB!) */
  SkyPosition position;	/**< source location (in radians) */
  REAL4 psi;            /**< polarization angle (radians) at tRef */
  REAL4 aPlus; 		/**< plus-polarization amplitude at tRef */
  REAL4 aCross;  	/**< cross-polarization amplitude at tRef */
  REAL8 phi0;           /**< initial phase (radians) at tRef */
  REAL8 f0;             /**< WAVE-frequency(!) at tRef (in Hz) */
  REAL8Vector *spindown;/**< wave-frequency spindowns at tRef (NOT f0-normalized!) */
} PulsarSourceParams;

/** Type defining the orbital parameters of a binary pulsar */
typedef struct {
  LIGOTimeGPS orbitEpoch; /**< time of periapsis passage (in SSB) */
  REAL8 omega;            /**< argument of periapsis (radians) */
  REAL8 rPeriNorm;        /**< projected, normalized periapsis (s) */
  REAL8 oneMinusEcc;      /**< 1 - orbital eccentricity */
  REAL8 angularSpeed;     /**< angular speed at periapsis (Hz) */
} BinaryOrbitParams;
  

#ifdef  __cplusplus
}
#endif  
/* C++ protection. */

#endif  /* Double-include protection. */

/*
 * Copyright (C) 2007 Bernd Machenschalk, David Churches, Duncan Brown,
 * Jolien Creighton, Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with with program; see the file COPYING. If not, write to the Free
 * Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 */


#include <lal/LALAtomicDatatypes.h>
#include <lal/LALNoiseModels.h>

/* macro to "use" unused function parameters */
#define UNUSED(expr) do { (void)(expr); } while (0)

NRCSID (LALLIGOIPSDC,"$Id$");


/**
 * Calculate the Initial LIGO SRD noise spectral density at given a
 * frequency.  The input is frequency in Hz, and the return value is the
 * noise spectral density, $S_{h}(f)$, for that frequency.
 *
 * The noise PSD is based on data provided by K. Blackburn (see \ref dis2001) and
 * is approximated by the following:
 *
 * \f[
 * S_h(f) = 9 \times 10^{-46} \left[ \left( 4.49 \frac{f}{f_0}
 * \right)^{-56} + 0.16 \left( \frac{f}{f_0} \right)^{-4.52} + 0.52 + 0.32
 * \left( \frac{f}{f_0} \right)^2 \right]
 * \f]
 *
 * Multiply the return value of this function by 2 \Delta f to put it in
 * the same units as used by the LAL average spectrum functions like
 * XLALWhitenCOMPLEX16FrequencySeries().
 */


REAL8 XLALLIGOIPsd(REAL8 f)
{
	double f_over_f0 = f / 150;

	return 9e-46 * (pow(4.49 * f_over_f0, -56) + 0.16 * pow(f_over_f0, -4.52) + 0.52 + 0.32 * pow(f_over_f0, 2));
}


/**
 * Legacy LAL wrapper of XLALLIGOIPsd().  Note that the return value is
 * scaled up by $s_0 = 10^{46}/9.$ In otherwords, the expected noise PSD is
 * \f$9 \times 10^{-46}\f$ times the return value.
 */


void LALLIGOIPsd(LALStatus *status, REAL8 *psd, REAL8 f)
{
  /* status is unused in this function */
  UNUSED(status);

	/* Deprecation warning temporarily commented out to make life
	 * easier for the nightly build --- please update calling code */
	/*XLALPrintDeprecationWarning("LALLIGOIPsd", "XLALLIGOIPsd");*/
	*psd = XLALLIGOIPsd(f) / 9e-46;
}

/*
*  Copyright (C) 2007 Jolien Creighton, Teviet Creighton
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

#ifndef _BANDPASSTIMESERIES_H
#define _BANDPASSTIMESERIES_H

#include <lal/LALStdlib.h>
#include <lal/IIRFilter.h>
#include <lal/ZPGFilter.h>

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

/**
 * \defgroup BandPassTimeSeries_h Header BandPassTimeSeries.h
 * \ingroup lal_tdfilter
 * \author Creighton, T. D.
 *
 * \brief Provides routines to low- or high-pass filter a time series.
 *
 * ### Synopsis ###
 *
 * \code
 * #include <lal/BandPassTimeSeries.h>
 * \endcode
 *
 * The \ref BandPassTimeSeries_h provides routines an actual implementation
 * of these utilities to the specific task of high- or low-pass filtering
 * of a data stream.  These routines also serve as an example for the more
 * general task of designing time-domain filters with any desired frequency
 * response.
 *
 * This header covers routines that apply a time-domain low- or
 * high-pass filter to a data series of type <tt>\<datatype\>TimeSeries</tt>.
 * Further documentation is given in the individual routines' modules.
 *
 */
/*@{*/

/** \name Error Codes */
/*@{*/
#define BANDPASSTIMESERIESH_ENUL 1	/**< Unexpected null pointer in arguments */
#define BANDPASSTIMESERIESH_EBAD 2	/**< Bad filter parameters */
/*@}*/

/** \cond DONT_DOXYGEN */
#define BANDPASSTIMESERIESH_MSGENUL "Unexpected null pointer in arguments"
#define BANDPASSTIMESERIESH_MSGEBAD "Bad filter parameters"
/** \endcond */

/**
 * This structure stores data used for constructing a low- or high-pass
 * filter: either the order and characteristic frequency of the filter,
 * or the frequencies and desired attenuations at the ends of some
 * transition band.  In the latter case, a nonzero filter order parameter
 * \c n indicates a maximum allowed order
 */
typedef struct tagPassBandParamStruc{
  CHAR *name;	/**< A user-assigned name */
  INT4 nMax;	/**< The maximum desired filter order (actual order may be less if specified attenuations do not require a high order) */
  REAL8 f1;	/**< The reference frequencies of the transition band */
  REAL8 f2;	/**< The reference frequencies of the transition band */
  REAL8 a1;	/**< The minimal desired attenuation factors at the reference frequencies */
  REAL8 a2;	/**< The minimal desired attenuation factors at the reference frequencies */
} PassBandParamStruc;

/*@}*/

/* Function prototypes. */
int XLALButterworthREAL4TimeSeries( REAL4TimeSeries *series, PassBandParamStruc *params );
int XLALButterworthREAL8TimeSeries( REAL8TimeSeries *series, PassBandParamStruc *params );
int XLALButterworthCOMPLEX8TimeSeries( COMPLEX8TimeSeries *series, PassBandParamStruc *params );
int XLALButterworthCOMPLEX16TimeSeries( COMPLEX16TimeSeries *series, PassBandParamStruc *params );
int XLALLowPassREAL4TimeSeries( REAL4TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALLowPassREAL8TimeSeries( REAL8TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALLowPassCOMPLEX8TimeSeries( COMPLEX8TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALLowPassCOMPLEX16TimeSeries( COMPLEX16TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALHighPassREAL4TimeSeries( REAL4TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALHighPassREAL8TimeSeries( REAL8TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALHighPassCOMPLEX8TimeSeries( COMPLEX8TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );
int XLALHighPassCOMPLEX16TimeSeries( COMPLEX16TimeSeries *series,
    REAL8 frequency, REAL8 amplitude, INT4 filtorder );



void
LALButterworthREAL4TimeSeries( LALStatus          *status,
			       REAL4TimeSeries    *series,
			       PassBandParamStruc *params );

void
LALButterworthREAL8TimeSeries( LALStatus          *status,
			       REAL8TimeSeries    *series,
			       PassBandParamStruc *params );

void
LALDButterworthREAL4TimeSeries( LALStatus          *status,
				REAL4TimeSeries    *series,
				PassBandParamStruc *params );

/* Chebyshev filters should also be added, but I'm too busy to write
   the routines now. */

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _BANDPASSTIMESERIES_H */

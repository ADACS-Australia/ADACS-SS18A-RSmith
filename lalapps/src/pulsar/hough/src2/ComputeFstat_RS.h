/*
 * Copyright (C) 2009 Chris Messenger, Pinkesh Patel
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
 * \author Chris Messenger
 * \date 2005
 * \ingroup pulsarCoherent
 * \file
 * \brief Header-file defining the API for the F-statistic functions.
 *
 * This code is (partly) a descendant of an earlier implementation found in
 * LALDemod.[ch] by Jolien Creighton, Maria Alessandra Papa, Reinhard Prix, Steve Berukoff, Xavier Siemens, Bruce Allen
 * ComputSky.[ch] by Jolien Creighton, Reinhard Prix, Steve Berukoff
 * LALComputeAM.[ch] by Jolien Creighton, Maria Alessandra Papa, Reinhard Prix, Steve Berukoff, Xavier Siemens
 *
 * $Id$
 *
 */

#ifndef _COMPUTEFSTATRS_H  /* Double-include protection. */
#define _COMPUTEFSTATRS_H

/* C++ protection. */
#ifdef  __cplusplus
extern "C" {
#endif

#include <lal/LALRCSID.h>
NRCSID( COMPUTEFSTATRSH, "$Id$" );

/*---------- exported INCLUDES ----------*/
#include <lal/LALComputeAM.h>
#include <lal/ComputeFstat.h>
#include <lal/PulsarDataTypes.h>
#include <lal/DetectorStates.h>

/*---------- exported DEFINES ----------*/

/*----- Error-codes -----*/
#define COMPUTEFSTATRSC_ENULL 		1
#define COMPUTEFSTATRSC_ENONULL 	2
#define COMPUTEFSTATRSC_EINPUT   	3
#define COMPUTEFSTATRSC_EMEM   		4
#define COMPUTEFSTATRSC_EXLAL		5
#define COMPUTEFSTATRSC_EIEEE		6

#define COMPUTEFSTATRSC_MSGENULL 	"Arguments contained an unexpected null pointer"
#define COMPUTEFSTATRSC_MSGENONULL 	"Output pointer is non-NULL"
#define COMPUTEFSTATRSC_MSGEINPUT   	"Invalid input"
#define COMPUTEFSTATRSC_MSGEMEM   	"Out of memory. Bad."
#define COMPUTEFSTATRSC_MSGEXLAL	"XLAL function call failed"
#define COMPUTEFSTATRSC_MSGEIEEE	"Floating point failure"

/*---------- exported types ----------*/

/*---------- exported prototypes [API] ----------*/

void ComputeFStatFreqBand_RS ( LALStatus *status,
			       REAL4FrequencySeries *fstatVector, 	
			       const PulsarDopplerParams *doppler,	
			       MultiSFTVector *multiSFTs, 
			       const MultiNoiseWeights *multiWeights,
			       const MultiDetectorStateSeries *multiDetStates,
			       const ComputeFParams *params
			       );

void ResampleMultiSFTs ( LALStatus *status,
			 MultiCOMPLEX8TimeSeries **multitimeseries,
			 REAL8 deltaF,
			 const MultiAMCoeffs *multiAMcoef,
			 const MultiSSBtimes *multiSSB,
			 const MultiSFTVector *multiSFTs
			 );

MultiCOMPLEX8TimeSeries *XLALMultiSFTVectorToCOMPLEX8TimeSeries_CHRIS ( MultiSFTVector *multisfts,  /**< input SFT vector, gets modified! */
									LIGOTimeGPS *start,    /**< input start time */
									LIGOTimeGPS *end       /**< input end time */
									); 

COMPLEX8TimeSeries *XLALSFTVectorToCOMPLEX8TimeSeries_CHRIS ( SFTVector *sfts,      /**< input SFT vector, gets modified! */
							      LIGOTimeGPS *start,    /**< input start time */
							      LIGOTimeGPS *end       /**< input end time */
							      );
  
int XLALEarliestMultiSFTsample ( LIGOTimeGPS *out,
				 MultiSFTVector *multisfts
				 );     
  
int XLALLatestMultiSFTsample ( LIGOTimeGPS *out,
			       MultiSFTVector *multisfts
			       );  
   
int XLALAntennaWeightCOMPLEX8TimeSeries ( COMPLEX8TimeSeries *Faoft,                         /**< [out] the timeseries weighted by a(t) */
					  COMPLEX8TimeSeries *Fboft,                         /**< [out] the timeseries weighted by b(t) */
					  const COMPLEX8TimeSeries *timeseries,         /**< [in] the input timeseries */
					  const AMCoeffs *AMcoef,                       /**< [in] the AM coefficients */
					  const SFTVector *sfts                         /**< [in] the SFT data */
					  );

int XLALAntennaWeightMultiCOMPLEX8TimeSeries(MultiCOMPLEX8TimeSeries *Faoft,                        /**< [out] the timeseries weighted by a(t) */
					     MultiCOMPLEX8TimeSeries *Fboft,                         /**< [out] the timeseries weighted by b(t) */
					     const MultiCOMPLEX8TimeSeries *multiTimeseries,         /**< [in] the input multi-detector timeseries */
					     const MultiAMCoeffs *multiAMcoef,                        /**< [in] the multi-detector AM coefficients */
					     const MultiSFTVector *multisfts                        /**< [in] the multi-detector SFT data */
					     );

int XLALBarycentricResampleMultiCOMPLEX8TimeSeries ( MultiCOMPLEX8TimeSeries *Faoft_RS,                         /**< [out] the timeseries weighted by a(t) */
						     MultiCOMPLEX8TimeSeries *Fboft_RS,                         /**< [out] the timeseries weighted by b(t) */
						     const MultiCOMPLEX8TimeSeries *Faoft,                   /**< [in] the input multi-detector timeseries */
						     const MultiCOMPLEX8TimeSeries *Fboft,                       /**< [in] the multi-detector AM coefficients */
						     const MultiSSBtimes *multiSSB                         /**< [in] the multi-detector SFT data */
						     );

#ifdef  __cplusplus
}
#endif
/* C++ protection. */

#endif  /* Double-include protection. */

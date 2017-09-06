/*
 * Copyright (C) 2014, 2016 Karl Wette
 * Copyright (C) 2010 Chris Messenger
 * Copyright (C) 2005 Reinhard Prix
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

/*---------- INCLUDES ----------*/
#include <stdarg.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_double.h>

#include <lal/AVFactories.h>
#include <lal/SeqFactories.h>
#include <lal/FrequencySeries.h>
#include <lal/NormalizeSFTRngMed.h>
#include <lal/LISAspecifics.h>
#include <lal/Date.h>
#include <lal/Units.h>
#include <lal/LALString.h>
#include <lal/ConfigFile.h>

#include <lal/SFTutils.h>

/*---------- DEFINES ----------*/
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))

/*----- SWITCHES -----*/

/*---------- internal types ----------*/

/*---------- Global variables ----------*/
static REAL8 fudge_up   = 1 + 10 * LAL_REAL8_EPS;	// about ~1 + 2e-15
static REAL8 fudge_down = 1 - 10 * LAL_REAL8_EPS;	// about ~1 - 2e-15

// XLALReadSegmentsFromFile(): applications which still must support
// the deprecated 4-column format should set this variable to non-zero
int XLALReadSegmentsFromFile_support_4column_format = 0;

/*---------- internal prototypes ----------*/
REAL8 TSFTfromDFreq ( REAL8 dFreq );
int compareSFTdesc(const void *ptr1, const void *ptr2);     // defined in SFTfileIO.c

/*==================== FUNCTION DEFINITIONS ====================*/

/**
 * XLAL function to create one SFT-struct.
 * Note: Allows for numBins == 0, in which case only the header is
 * allocated, with a NULL data pointer.
 */
SFTtype *
XLALCreateSFT ( UINT4 numBins )
{
  SFTtype *sft;

  if ( (sft = XLALCalloc (1, sizeof(*sft) )) == NULL )
    XLAL_ERROR_NULL ( XLAL_ENOMEM, "XLALCalloc (1, %zu) failed.\n", sizeof(*sft) );

  if ( numBins )
    {
      if ( (sft->data = XLALCreateCOMPLEX8Vector ( numBins )) == NULL )
        {
          XLALFree ( sft );
          XLAL_ERROR_NULL ( XLAL_EFUNC, "XLALCreateCOMPLEX8Vector ( %d ) failed. xlalErrno = %d\n", numBins, xlalErrno );
        }
    }
  else
    sft->data = NULL;	/* no data, just header */

  return sft;

} /* XLALCreateSFT() */


/** Destructor for one SFT */
void
XLALDestroySFT ( SFTtype *sft )
{
  if ( !sft )
    return;

  if ( sft->data )
    XLALDestroyCOMPLEX8Vector ( sft->data );

  XLALFree ( sft );

  return;

} /* XLALDestroySFT() */


/**
 * XLAL function to create an SFTVector of \c numSFT SFTs with \c SFTlen frequency-bins
 */
SFTVector *
XLALCreateSFTVector ( UINT4 numSFTs, 	/**< number of SFTs */
                      UINT4 numBins	/**< number of frequency-bins per SFT */
                      )
{
  UINT4 iSFT;
  SFTVector *vect;

  if ( (vect = XLALCalloc ( 1, sizeof(*vect) )) == NULL ) {
    XLAL_ERROR_NULL( XLAL_ENOMEM );
  }

  vect->length = numSFTs;
  if ( (vect->data = XLALCalloc (1, numSFTs * sizeof ( *vect->data ) )) == NULL ) {
    XLALFree (vect);
    XLAL_ERROR_NULL( XLAL_ENOMEM );
  }

  for ( iSFT = 0; iSFT < numSFTs; iSFT ++)
    {
      COMPLEX8Vector *data = NULL;

      /* allow SFTs with 0 bins: only header */
      if ( numBins )
	{
	  if ( (data = XLALCreateCOMPLEX8Vector ( numBins )) == NULL )
	    {
	      UINT4 j;
	      for ( j = 0; j < iSFT; j++ )
		XLALDestroyCOMPLEX8Vector ( vect->data[j].data );
	      XLALFree (vect->data);
	      XLALFree (vect);
	      XLAL_ERROR_NULL( XLAL_ENOMEM );
	    }
	}

      vect->data[iSFT].data = data;

    } /* for iSFT < numSFTs */

  return vect;

} /* XLALCreateSFTVector() */


/** Append the given SFTtype to the SFT-vector (no SFT-specific checks are done!) */
int XLALAppendSFT2Vector (SFTVector *vect,		/**< destinatino SFTVector to append to */
                          const SFTtype *sft            /**< the SFT to append */
                          )
{
  UINT4 oldlen = vect->length;

  if ( (vect->data = LALRealloc ( vect->data, (oldlen + 1)*sizeof( *vect->data ) )) == NULL ) {
     XLAL_ERROR(XLAL_ENOMEM);
  }
  memset ( &(vect->data[oldlen]), 0, sizeof( vect->data[0] ) );
  vect->length ++;

  XLALCopySFT(&vect->data[oldlen], sft );

  return XLAL_SUCCESS;

} /* XLALAppendSFT2Vector() */


/**
 * XLAL interface to destroy an SFTVector
 */
void
XLALDestroySFTVector ( SFTVector *vect )
{
  if ( !vect )
    return;

  for ( UINT4 i=0; i < vect->length; i++ )
    {
      SFTtype *sft = &( vect->data[i] );
      if ( sft->data )
	{
	  if ( sft->data->data )
	    XLALFree ( sft->data->data );
	  XLALFree ( sft->data );
	}
    } // for i < numSFTs

  XLALFree ( vect->data );
  XLALFree ( vect );

  return;

} /* XLALDestroySFTVector() */


/**
 * Destroy a PSD-vector
 */
void
XLALDestroyPSDVector ( PSDVector *vect )	/**< the PSD-vector to free */
{
  if ( vect == NULL )	/* nothing to be done */
    return;

  for ( UINT4 i=0; i < vect->length; i++ )
    {
      REAL8FrequencySeries *psd = &( vect->data[i] );
      if ( psd->data )
	{
	  if ( psd->data->data )
	    XLALFree ( psd->data->data );
	  XLALFree ( psd->data );
	}
    } // for i < numPSDs

  XLALFree ( vect->data );
  XLALFree ( vect );

  return;

} /* XLALDestroyPSDVector() */


/**
 * Create an empty multi-IFO SFT vector for given number of IFOs and number of SFTs per IFO
 */
MultiSFTVector *XLALCreateMultiSFTVector (
  UINT4 length,          /**< number of sft data points */
  UINT4Vector *numsft    /**< number of sfts in each sftvect */
  )
{

  XLAL_CHECK_NULL( length > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL( numsft != NULL, XLAL_EFAULT );
  XLAL_CHECK_NULL( numsft->length > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL( numsft->data != NULL, XLAL_EFAULT );

  MultiSFTVector *multSFTVec = NULL;

  XLAL_CHECK_NULL( ( multSFTVec = XLALCalloc( 1, sizeof(*multSFTVec) ) ) != NULL, XLAL_ENOMEM );

  const UINT4 numifo = numsft->length;
  multSFTVec->length = numifo;

  XLAL_CHECK_NULL( ( multSFTVec->data = XLALCalloc( numifo, sizeof(*multSFTVec->data) ) ) != NULL, XLAL_ENOMEM );

  for ( UINT4 k = 0; k < numifo; k++) {
    XLAL_CHECK_NULL( ( multSFTVec->data[k] = XLALCreateSFTVector( numsft->data[k], length ) ) != NULL, XLAL_ENOMEM );
  } /* loop over ifos */

  return multSFTVec;

} /* XLALCreateMultiSFTVector() */


/**
 * Destroy a multi SFT-vector
 */
void
XLALDestroyMultiSFTVector ( MultiSFTVector *multvect )	/**< the SFT-vector to free */
{
  if ( multvect == NULL )	/* nothing to be done */
    return;

  for ( UINT4 i = 0; i < multvect->length; i++ )
    XLALDestroySFTVector ( multvect->data[i] );

  XLALFree( multvect->data );
  XLALFree( multvect );

  return;

} /* XLALDestroyMultiSFTVector() */


/**
 * Destroy a multi PSD-vector
 */
void
XLALDestroyMultiPSDVector ( MultiPSDVector *multvect )	/**< the SFT-vector to free */
{
  if ( multvect == NULL )
    return;

  for ( UINT4 i = 0; i < multvect->length; i++ )
    XLALDestroyPSDVector ( multvect->data[i] );

  XLALFree( multvect->data );
  XLALFree( multvect );

  return;

} /* XLALDestroyMultiPSDVector() */


/** Allocate a LIGOTimeGPSVector */
LIGOTimeGPSVector *
XLALCreateTimestampVector ( UINT4 length )
{
  int len;
  LIGOTimeGPSVector *out = XLALCalloc (1, len = sizeof(LIGOTimeGPSVector));
  if (out == NULL)
    XLAL_ERROR_NULL ( XLAL_ENOMEM, "Failed to allocate XLALCalloc(1,%d)\n", len );

  out->length = length;
  out->data = XLALCalloc (1, len = length * sizeof(LIGOTimeGPS));
  if (out->data == NULL) {
    XLALFree (out);
    XLAL_ERROR_NULL ( XLAL_ENOMEM, "Failed to allocate XLALCalloc(1,%d)\n", len );
  }

  return out;

} /* XLALCreateTimestampVector() */


/** Resize a LIGOTimeGPSVector */
LIGOTimeGPSVector *
XLALResizeTimestampVector ( LIGOTimeGPSVector *vector, UINT4 length )
{
  if ( ! vector ) { return XLALCreateTimestampVector( length ); }
  if ( ! length ){
    XLALDestroyTimestampVector( vector );
    return NULL;
  }

  vector->data = XLALRealloc( vector->data, length * sizeof( LIGOTimeGPS ) );

  if ( ! vector->data ) {
    vector->length = 0;
    XLAL_ERROR_NULL( XLAL_ENOMEM );
  }
  vector->length = length;
  return vector;
}


/** De-allocate a LIGOTimeGPSVector */
void
XLALDestroyTimestampVector ( LIGOTimeGPSVector *vect)
{
  if ( !vect )
    return;

  XLALFree ( vect->data );
  XLALFree ( vect );

  return;

} /* XLALDestroyTimestampVector() */


/**
 * Given a start-time, Tspan, Tsft and Toverlap, returns a list of timestamps
 * covering this time-stretch (allowing for overlapping SFT timestamps).
 *
 * NOTE: boundary-handling: the returned list of timestamps are guaranteed to *cover* the
 * interval [tStart, tStart+duration), assuming a each timestamp covers a length of 'Tsft'
 * This implies that the actual timestamps-coverage can extend up to 'Tsft' beyond 'tStart+duration'.
 */
LIGOTimeGPSVector *
XLALMakeTimestamps ( LIGOTimeGPS tStart,	/**< GPS start-time */
                     REAL8 Tspan, 		/**< total duration to cover, in seconds */
                     REAL8 Tsft,		/**< Tsft: SFT length of each timestamp, in seconds */
                     REAL8 Toverlap		/**< time to overlap successive SFTs by, in seconds */
                     )
{
  XLAL_CHECK_NULL ( Tspan > 0, XLAL_EDOM );
  XLAL_CHECK_NULL ( Tsft  > 0, XLAL_EDOM );
  XLAL_CHECK_NULL ( Toverlap  >= 0, XLAL_EDOM );
  XLAL_CHECK_NULL ( Toverlap < Tsft, XLAL_EDOM );	// we must actually advance

  REAL8 Tstep = Tsft - Toverlap;	// guaranteed > 0
  UINT4 numSFTsMax = ceil ( Tspan * fudge_down / Tstep );			/* >= 1 !*/
  // now we might be covering the end-time several times, if using overlapping SFTs, so
  // let's trim this back down so that end-time is covered exactly once
  UINT4 numSFTs = numSFTsMax;
  while ( (numSFTs >= 2) && ( (numSFTs - 2) * Tstep + Tsft >= Tspan) ) {
    numSFTs --;
  }

  LIGOTimeGPSVector *ret;
  XLAL_CHECK_NULL ( (ret = XLALCreateTimestampVector ( numSFTs )) != NULL, XLAL_EFUNC );

  ret->deltaT = Tsft;

  LIGOTimeGPS tt = tStart;	/* initialize to start-time */
  for ( UINT4 i = 0; i < numSFTs; i++ )
    {
      ret->data[i] = tt;
      /* get next time-stamp */
      /* NOTE: we add the interval tStep successively (rounded correctly to ns each time!)
       * instead of using iSFT*Tsft, in order to avoid possible ns-rounding problems
       * with REAL8 intervals, which becomes critial from about 100days on...
       */
      XLAL_CHECK_NULL ( XLALGPSAdd ( &tt, Tstep ) != NULL, XLAL_EFUNC );

    } /* for i < numSFTs */

  return ret;

} /* XLALMakeTimestamps() */


/**
 * Same as XLALMakeTimestamps() just for several detectors,
 * additionally specify the number of detectors.
 */
MultiLIGOTimeGPSVector *
XLALMakeMultiTimestamps ( LIGOTimeGPS tStart,	/**< GPS start-time */
                          REAL8 Tspan, 		/**< total duration to cover, in seconds */
                          REAL8 Tsft,		/**< Tsft: SFT length of each timestamp, in seconds */
                          REAL8 Toverlap,	/**< time to overlap successive SFTs by, in seconds */
                          UINT4 numDet		/**< number of timestamps-vectors to generate */
                          )
{
  XLAL_CHECK_NULL ( numDet >= 1, XLAL_EINVAL );

  MultiLIGOTimeGPSVector *ret;
  XLAL_CHECK_NULL ( ( ret = XLALCalloc ( 1, sizeof(*ret))) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL ( ( ret->data = XLALCalloc ( numDet, sizeof(ret->data[0]) )) != NULL, XLAL_ENOMEM );
  ret->length = numDet;

  for ( UINT4 X=0; X < numDet; X ++ )
    {
      XLAL_CHECK_NULL ( (ret->data[X] = XLALMakeTimestamps ( tStart, Tspan, Tsft, Toverlap ) ) != NULL, XLAL_EFUNC );
    } // for X < numDet

  return ret;

} /* XLALMakeMultiTimestamps() */


/**
 * Extract timstamps-vector from the given SFTVector
 */
LIGOTimeGPSVector *
XLALExtractTimestampsFromSFTs ( const SFTVector *sfts )		/**< [in] input SFT-vector  */
{
  /* check input consistency */
  if ( !sfts ) {
    XLALPrintError ("%s: invalid NULL input 'sfts'\n", __func__ );
    XLAL_ERROR_NULL ( XLAL_EINVAL );
  }

  UINT4 numSFTs = sfts->length;
  /* create output vector */
  LIGOTimeGPSVector *ret = NULL;
  if ( ( ret = XLALCreateTimestampVector ( numSFTs )) == NULL ) {
    XLALPrintError ("%s: XLALCreateTimestampVector(%d) failed.\n", __func__, numSFTs );
    XLAL_ERROR_NULL ( XLAL_EFUNC );
  }
  ret->deltaT = TSFTfromDFreq ( sfts->data[0].deltaF );

  UINT4 i;
  for ( i=0; i < numSFTs; i ++ )
    ret->data[i] = sfts->data[i].epoch;

  /* done: return Ts-vector */
  return ret;

} /* XLALExtractTimestampsFromSFTs() */

/**
 * Given a multi-SFT vector, return a MultiLIGOTimeGPSVector holding the
 * SFT timestamps
 */
MultiLIGOTimeGPSVector *
XLALExtractMultiTimestampsFromSFTs ( const MultiSFTVector *multiSFTs )
{
  /* check input consistency */
  if ( !multiSFTs || multiSFTs->length == 0 ) {
    XLALPrintError ("%s: illegal NULL or empty input 'multiSFTs'.\n", __func__ );
    XLAL_ERROR_NULL ( XLAL_EINVAL );
  }
  UINT4 numIFOs = multiSFTs->length;

  /* create output vector */
  MultiLIGOTimeGPSVector *ret = NULL;
  if ( (ret = XLALCalloc ( 1, sizeof(*ret) )) == NULL ) {
    XLALPrintError ("%s: failed to XLALCalloc ( 1, %zu ).\n", __func__, sizeof(*ret));
    XLAL_ERROR_NULL ( XLAL_ENOMEM );
  }

  if ( (ret->data = XLALCalloc ( numIFOs, sizeof(*ret->data) )) == NULL ) {
    XLALPrintError ("%s: failed to XLALCalloc ( %d, %zu ).\n", __func__, numIFOs, sizeof(ret->data[0]) );
    XLALFree (ret);
    XLAL_ERROR_NULL ( XLAL_ENOMEM );
  }
  ret->length = numIFOs;

  /* now extract timestamps vector from each SFT-vector */
  UINT4 X;
  for ( X=0; X < numIFOs; X ++ )
    {
      if ( (ret->data[X] = XLALExtractTimestampsFromSFTs ( multiSFTs->data[X] )) == NULL ) {
        XLALPrintError ("%s: XLALExtractTimestampsFromSFTs() failed for X=%d\n", __func__, X );
        XLALDestroyMultiTimestamps ( ret );
        XLAL_ERROR_NULL ( XLAL_EFUNC );
      }

    } /* for X < numIFOs */

  return ret;

} /* XLALExtractMultiTimestampsFromSFTs() */


/**
 * Extract timestamps-vector of *unique* timestamps from the given SFTCatalog
 *
 * NOTE: when dealing with catalogs of frequency-slided SFTs, each timestamp will appear in the
 * catalog multiple times, depending on how many frequency slices have been read in.
 * In such cases this function will return the list of *unique* timestamps.
 *
 * NOTE 2: This function will also enfore the multiplicity of each timestamp to be the
 * same through the whole catalog, corresponding to the case of 'frequency-sliced' SFTs,
 * while non-constant multiplicities would indicate a potential problem somewhere.
 */
LIGOTimeGPSVector *
XLALTimestampsFromSFTCatalog ( const SFTCatalog *catalog )		/**< [in] input SFT-catalog  */
{
  // check input consistency
  XLAL_CHECK_NULL ( catalog != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( catalog->length > 0, XLAL_EINVAL );

  UINT4 numEntries = catalog->length;

  // create output vector, assuming maximal length, realloc at the end
  LIGOTimeGPSVector *ret;
  XLAL_CHECK_NULL ( ( ret = XLALCreateTimestampVector ( numEntries )) != NULL, XLAL_EFUNC );

  REAL8 Tsft0 = 1.0 / catalog->data[0].header.deltaF;
  if ( fabs ( (Tsft0 - round(Tsft0)) ) / Tsft0 < 10 * LAL_REAL8_EPS ) {	// 10-eps 'snap' to closest integer
    ret->deltaT = round(Tsft0);
  } else {
    ret->deltaT = Tsft0;
  }

  // For dealing with SFTCatalogs corresponding to frequency-sliced input SFTs:
  // Given the guaranteed GPS-ordering of XLALSFTDataFind(), we can rely on duplicate
  // timestamps to all be found next to each other, and therefore can easily skip them
  ret->data[0] = catalog->data[0].header.epoch;
  UINT4 numUnique = 1;
  UINT4 stride = 0;
  for ( UINT4 i = 1; i < numEntries; i ++ )
    {
      UINT4 thisStride = 1;
      const LIGOTimeGPS *ti   = &(catalog->data[i].header.epoch);
      const LIGOTimeGPS *tim1 = &(catalog->data[i-1].header.epoch);
      if ( XLALGPSCmp( ti, tim1 ) == 0 ) {
        thisStride ++;
        continue;	// skip duplicates
      }
      ret->data[numUnique] = catalog->data[i].header.epoch;
      numUnique ++;

      // keep track of stride, ensure that it's the same for every unique timestamp
      if ( stride == 0 ) {
        stride = thisStride;
      }
      else {
        XLAL_CHECK_NULL ( stride == thisStride, XLAL_EINVAL, "Suspicious SFT Catalog with non-constant timestamps multiplicities '%u != %u'\n", stride, thisStride );
      }
    } // for i < numEntries


  // now truncate output vector to actual length of unique timestamps
  ret->length = numUnique;
  XLAL_CHECK_NULL ( (ret->data = XLALRealloc ( ret->data, numUnique * sizeof( (*ret->data) ))) != NULL, XLAL_ENOMEM );

  // done: return Ts-vector
  return ret;

} /* XLALTimestampsFromSFTCatalog() */


/**
 * Extract timestamps-vector from a segment file, with functionality based on MakeSFTDAG
 * The filename should point to a file containing \<GPSstart GPSend\> of segments or \<GPSstart GPSend segLength numSFTs\> where segLength is in hours.
 * adjustSegExtraTime is used in MakeSFTDAG to maximize the number of SFTs in each segement by placing the SFTs in the middle of the segment.
 * synchronize is used to force the start times of the SFTs to be integer multiples of Toverlap from the start time of the first SFT.
 * adjustSegExtraTime and synchronize cannot be used concurrently (synchronize will be preferred if both values are non-zero).
 */
LIGOTimeGPSVector *
XLALTimestampsFromSegmentFile ( const char *filename,    //!< filename: Input filename
                                REAL8 Tsft,              //!< Tsft: SFT length of each timestamp, in seconds
                                REAL8 Toverlap,          //!< Toverlap: time to overlap successive SFTs by, in seconds
                                BOOLEAN adjustSegExtraTime, //!< adjustSegExtraTime: remove the unused time from beginning and end of the segments (see MakeSFTDAG)
                                BOOLEAN synchronize         //!< synchronize: synchronize SFT start times according to the start time of the first SFT. Start time of first SFT is shifted to next higher integer value of Tsft
                                )
{
  XLAL_CHECK_NULL ( filename != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( !(adjustSegExtraTime && synchronize), XLAL_EINVAL, "Must specify only one of adjustSegExtraTime or synchronize" );

  LALSegList *list = NULL;
  XLAL_CHECK_NULL ( ( list = XLALReadSegmentsFromFile ( filename )) != NULL, XLAL_EFUNC );

  //Need to know the number of SFTs before creating the timestamps vector, so we have to do the same loop twice
  INT4 numSFTs = 0;
  REAL8 firstSFTstartTime = 0.0, overlapFraction = Toverlap/Tsft;

  for ( UINT4 i = 0; i < list->length; i++ )
    {
      INT4 numThisSeg = 0;
      REAL8 analysisStartTime, analysisEndTime;
      if ( adjustSegExtraTime && !synchronize )
        {
          REAL8 segStartTime = XLALGPSGetREAL8 ( &(list->segs[i].start) );
          REAL8 segEndTime = XLALGPSGetREAL8 ( &(list->segs[i].end) );
          REAL8 segExtraTime = fmod ( (segEndTime - segStartTime), Tsft );
          if (overlapFraction!=0.0)
            {
              if ((segEndTime - segStartTime) > Tsft) {
                segExtraTime = fmod((segEndTime - segStartTime - Tsft), ((1.0 - overlapFraction)*Tsft));
              }
            }
          else
            {
              segExtraTime = fmod((segEndTime - segStartTime), Tsft);
            }
          REAL8 segExtraStart =  segExtraTime / 2;
          REAL8 segExtraEnd = segExtraTime - segExtraStart;
          analysisStartTime = segStartTime + segExtraStart;
          if (analysisStartTime > segEndTime) {
            analysisStartTime = segEndTime;
          }
          analysisEndTime = segEndTime - segExtraEnd;
          if (analysisEndTime < segStartTime) {
            analysisEndTime = segStartTime;
          }
        } // if adjustSegExtraTime && !synchronize
      else if (synchronize)
        {
          REAL8 segStartTime = XLALGPSGetREAL8 ( &(list->segs[i].start) );
          REAL8 segEndTime = XLALGPSGetREAL8 ( &(list->segs[i].end) );
          if ( firstSFTstartTime==0.0 ) {
             firstSFTstartTime = ceil(segStartTime/Tsft)*Tsft;
          }
          analysisStartTime = round ( ceil ( (segStartTime - firstSFTstartTime)/((1.0 - overlapFraction)*Tsft))*(1.0 - overlapFraction)*Tsft) + firstSFTstartTime;
          if (analysisStartTime > segEndTime) {
            analysisStartTime = segEndTime;
          }
          analysisEndTime = round ( floor ( (segEndTime - analysisStartTime - Tsft)/((1.0 - overlapFraction)*Tsft))*(1.0 - overlapFraction)*Tsft) + Tsft + analysisStartTime;
          if (analysisEndTime < segStartTime) {
            analysisEndTime = segStartTime;
          }
        }
      else
        {
          analysisStartTime = XLALGPSGetREAL8 ( &(list->segs[i].start) );
          analysisEndTime = XLALGPSGetREAL8 ( &(list->segs[i].end) );
        }

      REAL8 endTime = analysisStartTime;
      while (endTime < analysisEndTime)
        {
          if (numThisSeg==0) {
            endTime += Tsft;
          }
          else {
            endTime += (1.0 - overlapFraction)*Tsft;
          }
          if (endTime <= analysisEndTime) {
            numThisSeg++;
          }
        } // while endTime < analysisEndTime
      numSFTs += numThisSeg;
    } // for i < length

  LIGOTimeGPSVector *ret;
  XLAL_CHECK_NULL ( ( ret = XLALCreateTimestampVector ( numSFTs )) != NULL, XLAL_EFUNC );

  ret->deltaT = Tsft;

  //Second time doing the same thing, but now we can set the times of the SFTs in the timestamps vector
  firstSFTstartTime = 0.0;
  UINT4 j = 0;
  for ( UINT4 i = 0; i < list->length; i++ )
    {
      INT4 numThisSeg = 0;
      REAL8 analysisStartTime, analysisEndTime;
      if ( adjustSegExtraTime && !synchronize )
        {
          REAL8 segStartTime = XLALGPSGetREAL8 ( &(list->segs[i].start) );
          REAL8 segEndTime = XLALGPSGetREAL8 ( &(list->segs[i].end) );
          REAL8 segExtraTime = fmod ( (segEndTime - segStartTime), Tsft );
          if (overlapFraction!=0.0)
            {
              if ((segEndTime - segStartTime) > Tsft) {
                segExtraTime = fmod((segEndTime - segStartTime - Tsft), ((1.0 - overlapFraction)*Tsft));
              }
            }
          else
            {
              segExtraTime = fmod((segEndTime - segStartTime), Tsft);
            }
          REAL8 segExtraStart =  segExtraTime / 2;
          REAL8 segExtraEnd = segExtraTime - segExtraStart;
          analysisStartTime = segStartTime + segExtraStart;
          if (analysisStartTime > segEndTime) {
            analysisStartTime = segEndTime;
          }
          analysisEndTime = segEndTime - segExtraEnd;
          if (analysisEndTime < segStartTime) {
            analysisEndTime = segStartTime;
          }
        } // if adjustSegExtraTime && !synchronize
      else if (synchronize)
        {
          REAL8 segStartTime = XLALGPSGetREAL8 ( &(list->segs[i].start) );
          REAL8 segEndTime = XLALGPSGetREAL8 ( &(list->segs[i].end) );
          if ( firstSFTstartTime==0.0 ) {
             firstSFTstartTime = ceil(segStartTime/Tsft)*Tsft;
          }
          analysisStartTime = round ( ceil ( (segStartTime - firstSFTstartTime)/((1.0 - overlapFraction)*Tsft))*(1.0 - overlapFraction)*Tsft) + firstSFTstartTime;
          if (analysisStartTime > segEndTime) {
            analysisStartTime = segEndTime;
          }
          analysisEndTime = round ( floor ( (segEndTime - analysisStartTime - Tsft)/((1.0 - overlapFraction)*Tsft))*(1.0 - overlapFraction)*Tsft) + Tsft + analysisStartTime;
          if (analysisEndTime < segStartTime) {
            analysisEndTime = segStartTime;
          }
        }
      else
        {
          analysisStartTime = XLALGPSGetREAL8(&(list->segs[i].start));
          analysisEndTime = XLALGPSGetREAL8(&(list->segs[i].end));
        }

      REAL8 endTime = analysisStartTime;
      while ( endTime < analysisEndTime )
        {
          if ( numThisSeg==0 ) {
            endTime += Tsft;
          }
          else {
            endTime += (1.0 - overlapFraction)*Tsft;
          }
          if ( endTime <= analysisEndTime ) {
            numThisSeg++;
            LIGOTimeGPS sftStart;
            XLALGPSSetREAL8( &sftStart, endTime-Tsft);
            ret->data[j] = sftStart;
            j++;
          }
        } // while ( endTime < analysisEndTime )
    } // for i < length

  XLALSegListFree(list);

  /* done: return Ts-vector */
  return ret;

} // XLALTimestampsFromSegmentFile()


/**
 * Given a multi-SFTCatalogView, return a MultiLIGOTimeGPSVector holding the
 * SFT timestamps
 */
MultiLIGOTimeGPSVector *
XLALTimestampsFromMultiSFTCatalogView ( const MultiSFTCatalogView *multiView )
{
  /* check input consistency */
  XLAL_CHECK_NULL ( multiView != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( multiView->length > 0, XLAL_EINVAL );

  UINT4 numIFOs = multiView->length;

  /* create output vector */
  MultiLIGOTimeGPSVector *ret;
  XLAL_CHECK_NULL ( (ret = XLALCalloc ( 1, sizeof(*ret) )) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL ( (ret->data = XLALCalloc ( numIFOs, sizeof(*(ret->data)) )) != NULL, XLAL_ENOMEM );
  ret->length = numIFOs;

  /* now extract timestamps vector from each IFO's SFT-Catalog */
  for ( UINT4 X=0; X < numIFOs; X ++ )
    {
      XLAL_CHECK_NULL ( (ret->data[X] = XLALTimestampsFromSFTCatalog ( &(multiView->data[X]) )) != NULL, XLAL_EFUNC );
    } /* for X < numIFOs */

  return ret;

} /* XLALTimestampsFromMultiSFTCatalogView() */


/**
 * Destroy a MultiLIGOTimeGPSVector timestamps vector
 */
void
XLALDestroyMultiTimestamps ( MultiLIGOTimeGPSVector *multiTS )
{
  UINT4 numIFOs, X;

  if ( !multiTS )
    return;

  numIFOs = multiTS->length;
  for ( X=0; X < numIFOs; X ++ ) {
    XLALDestroyTimestampVector ( multiTS->data[X] );
  }

  XLALFree ( multiTS->data );
  XLALFree ( multiTS );

  return;

} /* XLALDestroyMultiTimestamps() */

///
/// Parses valid CW detector names and prefixes: 'name' input can be either a valid detector name or prefix
/// \return allocated prefix string (2 characters+0) for valid detectors, NULL otherwise
///
/// If passed a non-NULL pointers 'lalCachedIndex', will set to index >= 0 into the
/// lalCachedDetectors[] array if found there, or -1 if it's one of the "CW special" detectors
///
/// \note this should be the *only* function defining valid CW detector names and prefixes
///
/// \note if first two characters of 'name' match a valid detector prefix, that is accepted, which
/// allows passing longer strings beginning with a detector prefix ('H1:blabla') without extra hassle
///
/// \note the returned string is allocated here and needs to be free'ed by caller!
///
char *
XLALGetCWDetectorPrefix ( INT4 *lalCachedIndex, const char *name )
{
  XLAL_CHECK_NULL ( name != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( strlen ( name ) >= 2, XLAL_EINVAL );	// need at least a full prefix 'letter+number'

  // ----- first check if 'name' corresponds to one of our 'CW special' detectors (LISA and X-ray satellites)
  const char *specialDetectors[] =
    {
      "Z1",	  /* LISA effective IFO 1 */
      "Z2",	  /* LISA effective IFO 2 */
      "Z3",	  /* LISA effective IFO 3 */
      "Z4",	  /* LISA effective IFO 2 minus 3 */
      "Z5",	  /* LISA effective IFO 3 minus 1 */
      "Z6",	  /* LISA effective IFO 1 minus 2 */
      "Z7",	  /* LISA pseudo TDI A */
      "Z8",	  /* LISA pseudo TDI E */
      "Z9",	  /* LISA pseudo TDI T */

      "X1",       /* RXTE PCA */
      "X2",       /* RXTE ASM */
      NULL
    };
  for ( UINT4 i = 0; specialDetectors[i] != NULL; i ++ )
    {
      if ( ( specialDetectors[i][0] == name[0] ) && ( specialDetectors[i][1] == name[1] )  )
        {
          if ( lalCachedIndex != NULL ) {
            (*lalCachedIndex) = -1;
          }
          return XLALStringDuplicate ( specialDetectors[i] );
        }
    } // for i < len(specialDetectors)

  // ----- if not found, go through list of 'official' cached lalsuite detectors
  UINT4 numLALDetectors = sizeof(lalCachedDetectors) / sizeof(lalCachedDetectors[0]);
  for ( UINT4 i = 0; i < numLALDetectors; i ++)
    {
      const char *prefix_i = lalCachedDetectors[i].frDetector.prefix;
      const char *name_i   = lalCachedDetectors[i].frDetector.name;
      if ( ((prefix_i[0] == name[0]) && (prefix_i[1] == name[1]))
           || strncmp ( name, name_i, strlen ( name_i ) ) == 0
           )
        {
          if ( lalCachedIndex != NULL ) {
            (*lalCachedIndex) = i;
          }
          return XLALStringDuplicate ( prefix_i );
        } // found prefix match in lalCachedDetectors[]

    } // for i < numLALDetectors

  XLAL_ERROR_NULL ( XLAL_EINVAL, "Unknown detector name '%s'\n", name );

} // XLALGetCWDetectorPrefix()

/**
 * Extract/construct the unique 2-character "channel prefix" from the given "detector-name".
 * This is only a convenience wrapper around XLALGetCWDetectorPrefix() for backwards compatibility.
 *
 * NOTE: the returned string is allocated here!
 *
 */
CHAR *
XLALGetChannelPrefix ( const CHAR *name )
{
  return XLALGetCWDetectorPrefix ( NULL, name );
} // XLALGetChannelPrefix()


///
/// Find the site geometry-information 'LALDetector' for given a detector name (or prefix).
///
/// \note The LALDetector struct is allocated here and needs to be free'ed by caller!
///
LALDetector *
XLALGetSiteInfo ( const CHAR *name )
{
  XLAL_CHECK_NULL ( name != NULL, XLAL_EINVAL );

  const INT4 numLALDetectors = sizeof(lalCachedDetectors) / sizeof(lalCachedDetectors[0]);

  // first turn the free-form 'detector-name' into a well-defined channel-prefix, and find lalCachedDetector[] index
  INT4 lalCachedIndex = -1;
  CHAR *prefix;
  XLAL_CHECK_NULL ( (prefix = XLALGetCWDetectorPrefix ( &lalCachedIndex, name )) != NULL, XLAL_EFUNC );

  LALDetector *site;
  XLAL_CHECK_NULL ( ( site = XLALCalloc ( 1, sizeof( *site) )) != NULL, XLAL_ENOMEM );

  switch ( prefix[0] )
    {
    case 'X':	    // X-ray satellite data
      XLAL_ERROR_NULL ( XLAL_EINVAL, "Sorry, detector site not implemented for special 'X'-ray detector 'name=%s, prefix=%s'\n", name, prefix );
      break;

    case 'Z':       // create dummy-sites for LISA
      XLAL_CHECK_NULL ( XLALcreateLISA ( site, prefix[1] ) != XLAL_SUCCESS, XLAL_EFUNC, "Failed to created LISA detector 'name=%s, prefix=%s'\n", name, prefix );
      break;

    default:
      XLAL_CHECK_NULL ( (lalCachedIndex >= 0) && (lalCachedIndex < numLALDetectors), XLAL_EFAILED, "Internal inconsistency found (for 'name=%s, prefix=%s')\n", name, prefix );
      (*site) = lalCachedDetectors[lalCachedIndex];
      break;
    } /* switch channel[0] */

  XLALFree ( prefix );

  return site;

} // XLALGetSiteInfo()


/**
 * Computes weight factors arising from MultiSFTs with different noise
 * floors
 */
MultiNoiseWeights *
XLALComputeMultiNoiseWeights ( const MultiPSDVector  *rngmed,
			       UINT4                 blocksRngMed,
			       UINT4                 excludePercentile)
{
  XLAL_CHECK_NULL ( rngmed != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( rngmed->data != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL ( rngmed->length != 0, XLAL_EINVAL );

  UINT4 numIFOs = rngmed->length;
  REAL8 Tsft = TSFTfromDFreq ( rngmed->data[0]->data[0].deltaF );

  /* create multi noise weights for output */
  MultiNoiseWeights *multiWeights = NULL;
  XLAL_CHECK_NULL ( (multiWeights = XLALCalloc(1, sizeof(*multiWeights))) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL ( (multiWeights->data = XLALCalloc ( numIFOs, sizeof(*multiWeights->data))) != NULL, XLAL_ENOMEM );
  multiWeights->length = numIFOs;

  UINT4 numSFTsTot = 0;
  REAL8 sumWeights = 0;

  for ( UINT4 X = 0; X < numIFOs; X++)
    {
      UINT4 numSFTs = rngmed->data[X]->length;
      numSFTsTot += numSFTs;

      /* create k^th weights vector */
      if( ( multiWeights->data[X] = XLALCreateREAL8Vector ( numSFTs ) ) == NULL )
        {
          /* free weights vectors created previously in loop */
          XLALDestroyMultiNoiseWeights ( multiWeights );
          XLAL_ERROR_NULL ( XLAL_EFUNC, "Failed to allocate noiseweights for IFO X = %d\n", X );
        } /* if XLALCreateREAL8Vector() failed */

      /* loop over rngmeds and calculate weights -- one for each sft */
      for ( UINT4 alpha = 0; alpha < numSFTs; alpha++)
	{
	  UINT4 halfBlock = blocksRngMed/2;

	  REAL8FrequencySeries *thisrm = &(rngmed->data[X]->data[alpha]);

	  UINT4 lengthsft = thisrm->data->length;

	  XLAL_CHECK_NULL ( lengthsft >= blocksRngMed, XLAL_EINVAL );

	  UINT4 length = lengthsft - blocksRngMed + 1;
	  UINT4 halfLength = length/2;

	  /* calculate index in power medians vector from which to calculate mean */
	  UINT4 excludeIndex =  excludePercentile * halfLength ; /* integer arithmetic */
	  excludeIndex /= 100; /* integer arithmetic */

	  REAL8 Tsft_avgS2 = 0.0;	// 'S2' refers to double-sided PSD
	  for ( UINT4 k = halfBlock + excludeIndex; k < lengthsft - halfBlock - excludeIndex; k++)
	    {
	      Tsft_avgS2 += thisrm->data->data[k];
	    }
	  Tsft_avgS2 /= lengthsft - 2*halfBlock - 2*excludeIndex;

          REAL8 wXa = 1.0/Tsft_avgS2;	// unnormalized weight
	  multiWeights->data[X]->data[alpha] = wXa;

	  sumWeights += wXa;	// sum the weights to normalize this at the end
	} /* end loop over sfts for each ifo */

    } /* end loop over ifos */

  /* overall noise-normalization factor Sinv = 1/Nsft sum_Xa Sinv_Xa,
   * see Eq.(60) in CFSv2 notes:
   * https://dcc.ligo.org/cgi-bin/private/DocDB/ShowDocument?docid=1665&version=3
   */
  REAL8 TsftS2_inv = sumWeights / numSFTsTot;	// this is double-sided PSD 'S2'

  /* make weights of order unity by normalizing with TsftS2_inv, see Eq.(58) in CFSv2 notes (v3) */
  for ( UINT4 X = 0; X < numIFOs; X ++) {
    UINT4 numSFTs = multiWeights->data[X]->length;
    for ( UINT4 alpha = 0; alpha < numSFTs; alpha ++)
      {
	multiWeights->data[X]->data[alpha] /= TsftS2_inv;
      }
  }

  multiWeights->Sinv_Tsft = 0.5 * Tsft*Tsft * TsftS2_inv;		/* 'Sinv * Tsft' refers to single-sided PSD!! Eq.(60) in CFSv2 notes (v3)*/

  return multiWeights;

} /* XLALComputeMultiNoiseWeights() */

/** Destroy a MultiNoiseWeights object */
void
XLALDestroyMultiNoiseWeights ( MultiNoiseWeights *weights )
{
  if ( weights == NULL)
    return;

  for ( UINT4 k = 0; k < weights->length; k++ )
    XLALDestroyREAL8Vector ( weights->data[k] );

  XLALFree ( weights->data );
  XLALFree ( weights );

  return;

} /* XLALDestroyMultiNoiseWeights() */


/**
 * Interpolate frequency-series to newLen frequency-bins.
 * This is using DFT-interpolation (derived from zero-padding).
 */
COMPLEX8Vector *
XLALrefineCOMPLEX8Vector ( const COMPLEX8Vector *in,
                           UINT4 refineby,
                           UINT4 Dterms
                           )
{
  UINT4 newLen, oldLen, l;
  COMPLEX8Vector *ret = NULL;

  if ( !in )
    return NULL;

  oldLen = in->length;
  newLen = oldLen * refineby;

  /* the following are used to speed things up in the innermost loop */
  if ( (ret = XLALCreateCOMPLEX8Vector ( newLen )) == NULL )
    return NULL;

  for (l=0; l < newLen; l++)
    {

      REAL8 kappa_l_k;
      REAL8 remain, kstarREAL;
      UINT4 kstar, kmin, kmax, k;
      REAL8 sink, coskm1;
      REAL8 Yk_re, Yk_im, Xd_re, Xd_im;

      kstarREAL = 1.0 * l  / refineby;
      kstar = lround( kstarREAL );	/* round to closest bin */
      kstar = MIN ( kstar, oldLen - 1 );	/* stay within the old SFT index-bounds */
      remain = kstarREAL - kstar;

      /* boundaries for innermost loop */
      kmin = MAX( 0, (INT4)kstar - (INT4)Dterms );
      kmax = MIN( oldLen, kstar + Dterms );

      Yk_re = Yk_im = 0;
      if ( fabs(remain) > 1e-5 )	/* denominater doens't vanish */
	{
	  /* Optimization: sin(2pi*kappa(l,k)) = sin(2pi*kappa(l,0) and idem for cos */
	  sink = sin ( LAL_TWOPI * remain );
	  coskm1 = cos ( LAL_TWOPI * remain ) - 1.0;

	  /* ---------- innermost loop: k over 2*Dterms around kstar ---------- */
	  for (k = kmin; k < kmax; k++)
	    {
	      REAL8 Plk_re, Plk_im;

	      Xd_re = crealf(in->data[k]);
	      Xd_im = cimagf(in->data[k]);

	      kappa_l_k = kstarREAL - k;

	      Plk_re = sink / kappa_l_k;
	      Plk_im = coskm1 / kappa_l_k;

	      Yk_re += Plk_re * Xd_re - Plk_im * Xd_im;
	      Yk_im += Plk_re * Xd_im + Plk_im * Xd_re;

	    } /* hotloop over Dterms */
	}
      else	/* kappa -> 0: Plk = 2pi delta(k, l) */
	{
	  Yk_re = LAL_TWOPI * crealf(in->data[kstar]);
	  Yk_im = LAL_TWOPI * cimagf(in->data[kstar]);
	}

      const REAL8 OOTWOPI = (1.0 / LAL_TWOPI );
      ret->data[l] = crectf( OOTWOPI* Yk_re, OOTWOPI * Yk_im );

    }  /* for l < newlen */

  return ret;

} /* XLALrefineCOMPLEX8Vector() */


/**
 * Function to read a segment list from given filename, returns a *sorted* LALSegList
 *
 * The segment list file format is repeated lines (excluding comment lines beginning with
 * <tt>\%</tt> or <tt>#</tt>) of one of the following forms:
 * - <tt>startGPS endGPS</tt>
 * - <tt>startGPS endGPS NumSFTs</tt> (NumSFTs must be a positive integer)
 * - <tt>startGPS endGPS duration NumSFTs</tt> (\b DEPRECATED, duration is ignored)
 *
 * \note We (ab)use the integer \p id field in LALSeg to carry the total number of SFTs
 * contained in that segment if <tt>NumSFTs</tt> was provided in the segment file.
 * This can be used as a consistency check when loading SFTs for these segments.
 */
LALSegList *
XLALReadSegmentsFromFile ( const char *fname	/**< name of file containing segment list */
                           )
{
  LALSegList *segList = NULL;

  /* check input consistency */
  XLAL_CHECK_NULL( fname != NULL, XLAL_EFAULT );

  /* read and parse segment-list file contents*/
  LALParsedDataFile *flines = NULL;
  XLAL_CHECK_NULL( XLALParseDataFile ( &flines, fname ) == XLAL_SUCCESS, XLAL_EFUNC );
  const UINT4 numSegments = flines->lines->nTokens;
  XLAL_CHECK_NULL( numSegments > 0, XLAL_EINVAL, "%s: segment file '%s' does not contain any segments", __func__, fname );

  /* allocate and initialized segment list */
  XLAL_CHECK_NULL( ( segList = XLALCalloc ( 1, sizeof(*segList) ) ) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL( XLALSegListInit ( segList ) == XLAL_SUCCESS, XLAL_EFUNC );

  /* determine number of columns */
  int ncol = 0;
  {
    REAL8 col[4];
    ncol = sscanf( flines->lines->tokens[0], "%lf %lf %lf %lf", &col[0], &col[1], &col[2], &col[3] );
    switch (ncol) {
    case 2:
    case 3:
      break;
    case 4:
      if ( XLALReadSegmentsFromFile_support_4column_format ) {
        XLALPrintError( "\n%s: WARNING: segment file '%s' is in DEPRECATED 4-column format (startGPS endGPS duration NumSFTs, duration is ignored)\n", __func__, fname );
      } else {
        XLAL_ERROR_NULL( XLAL_EIO, "%s: segment file '%s' is in DEPRECATED 4-column format (startGPS endGPS duration NumSFTs)\n", __func__, fname );
      }
      break;
    default:
      XLAL_ERROR_NULL( XLAL_EIO, "%s: segment file '%s' contains an unknown %i-column format", __func__, fname, ncol );
    }
  }

  /* parse segment list */
  for ( UINT4 iSeg = 0; iSeg < numSegments; iSeg ++ )
    {

      /* parse line of segment file, depending on determined number of columns */
      REAL8 start = 0, end = 0, duration = 0;
      INT4 NumSFTs = 0;
      int ret;
      switch (ncol) {
      case 2:
        ret = sscanf( flines->lines->tokens[iSeg], "%lf %lf", &start, &end );
        XLAL_CHECK_NULL( ret == 2, XLAL_EIO, "%s: number of columns in segment file '%s' is inconsistent (line 1: %i, line %u: %i)", __func__, fname, ncol, iSeg+1, ret );
        break;
      case 3:
        ret = sscanf( flines->lines->tokens[iSeg], "%lf %lf %i", &start, &end, &NumSFTs );
        XLAL_CHECK_NULL( ret == 3, XLAL_EIO, "%s: number of columns in segment file '%s' is inconsistent (line 1: %i, line %u: %i)", __func__, fname, ncol, iSeg+1, ret );
        XLAL_CHECK_NULL( NumSFTs > 0, XLAL_EIO, "%s: number of SFTs (3rd column) in segment file '%s' must be a positive integer if given (line %u: %i)", __func__, fname, iSeg+1, NumSFTs );
        break;
      case 4:
        ret = sscanf( flines->lines->tokens[iSeg], "%lf %lf %lf %i", &start, &end, &duration, &NumSFTs );
        XLAL_CHECK_NULL( ret == 4, XLAL_EIO, "%s: number of columns in segment file '%s' is inconsistent (line 1 = %i, line %u = %i)", __func__, fname, ncol, iSeg+1, ret );
        break;
      default:
        XLAL_ERROR_NULL( XLAL_EFAILED, "Unexpected error!" );
      }

      /* set GPS start and end times */
      LIGOTimeGPS startGPS, endGPS;
      XLALGPSSetREAL8( &startGPS, start );
      XLALGPSSetREAL8( &endGPS, end );

      /* create segment and append to list
         - we set number of SFTs as 'id' field, as we have no other use for it */
      LALSeg thisSeg;
      XLAL_CHECK_NULL( XLALSegSet ( &thisSeg, &startGPS, &endGPS, NumSFTs ) == XLAL_SUCCESS, XLAL_EFUNC );
      XLAL_CHECK_NULL( XLALSegListAppend ( segList, &thisSeg ) == XLAL_SUCCESS, XLAL_EFUNC );

    } /* for iSeg < numSegments */

  /* sort final segment list in increasing GPS start-times */
  XLAL_CHECK_NULL( XLALSegListSort( segList ) == XLAL_SUCCESS, XLAL_EFUNC );

  /* free parsed segment file contents */
  XLALDestroyParsedDataFile ( flines );

  return segList;

} /* XLALReadSegmentsFromFile() */

/**
 * Return a vector of SFTs containing only the bins in [fMin, fMin+Band].
 * Note: the output SFT is guaranteed to "cover" the input boundaries 'fMin'
 * and 'fMin+Band', ie if necessary the output SFT contains one additional
 * bin on either end of the interval.
 *
 * This uses the conventions in XLALFindCoveringSFTBins() to determine
 * the 'effective' frequency-band to extract.
 *
 */
SFTVector *
XLALExtractBandFromSFTVector ( const SFTVector *inSFTs,	///< [in] input SFTs
                               REAL8 fMin,		///< [in] lower end of frequency interval to return
                               REAL8 Band		///< [in] band width of frequency interval to return
                               )
{
  XLAL_CHECK_NULL ( inSFTs != NULL, XLAL_EINVAL, "Invalid NULL input SFT vector 'inSFTs'\n");
  XLAL_CHECK_NULL ( inSFTs->length > 0, XLAL_EINVAL, "Invalid zero-length input SFT vector 'inSFTs'\n");
  XLAL_CHECK_NULL ( fMin >= 0, XLAL_EDOM, "Invalid negative frequency fMin = %g\n", fMin );
  XLAL_CHECK_NULL ( Band > 0, XLAL_EDOM, "Invalid non-positive Band = %g\n", Band );

  UINT4 numSFTs = inSFTs->length;

  SFTVector *ret;
  XLAL_CHECK_NULL ( (ret = XLALCreateSFTVector ( numSFTs, 0 )) != NULL, XLAL_EFUNC );

  for ( UINT4 i = 0; i < numSFTs; i ++ )
    {
      SFTtype *dest = &(ret->data[i]);
      SFTtype *src =  &(inSFTs->data[i]);

      XLAL_CHECK_NULL ( XLALExtractBandFromSFT ( &dest, src, fMin, Band ) == XLAL_SUCCESS, XLAL_EFUNC );

    } /* for i < numSFTs */

  /* return final SFT-vector */
  return ret;

} /* XLALExtractBandFromSFTVector() */


/**
 * Return an SFTs containing only the bins in [fMin, fMin+Band].
 * Note: the output SFT is guaranteed to "cover" the input boundaries 'fMin'
 * and 'fMin+Band', ie if necessary the output SFT contains one additional
 * bin on either end of the interval.
 *
 * This uses the conventions in XLALFindCoveringSFTBins() to determine
 * the 'effective' frequency-band to extract.
 *
 */
int
XLALExtractBandFromSFT ( SFTtype **outSFT,	///< [out] output SFT (alloc'ed or re-alloced as required)
                         const SFTtype *inSFT,	///< [in] input SFT
                         REAL8 fMin,		///< [in] lower end of frequency interval to return
                         REAL8 Band		///< [in] band width of frequency interval to return
                         )
{
  XLAL_CHECK ( outSFT != NULL, XLAL_EINVAL );
  XLAL_CHECK ( inSFT != NULL, XLAL_EINVAL );
  XLAL_CHECK ( inSFT->data != NULL, XLAL_EINVAL );
  XLAL_CHECK ( fMin >= 0, XLAL_EDOM, "Invalid negative frequency fMin = %g\n", fMin );
  XLAL_CHECK ( Band > 0, XLAL_EDOM, "Invalid non-positive Band = %g\n", Band );

  REAL8 df      = inSFT->deltaF;
  REAL8 Tsft    = TSFTfromDFreq ( df );

  REAL8 fMinSFT    = inSFT->f0;
  UINT4 numBinsSFT = inSFT->data->length;
  UINT4 firstBinSFT= round ( fMinSFT / df );	// round to closest bin
  UINT4 lastBinSFT = firstBinSFT + ( numBinsSFT - 1 );

  // find 'covering' SFT-band to extract
  UINT4 firstBinExt, numBinsExt;
  XLAL_CHECK ( XLALFindCoveringSFTBins ( &firstBinExt, &numBinsExt, fMin, Band, Tsft ) == XLAL_SUCCESS, XLAL_EFUNC );
  UINT4 lastBinExt = firstBinExt + ( numBinsExt - 1 );

  XLAL_CHECK ( firstBinExt >= firstBinSFT && (lastBinExt <= lastBinSFT), XLAL_EINVAL,
               "Requested frequency-bins [%f,%f]Hz = [%d, %d] not contained within SFT's [%f, %f]Hz = [%d,%d].\n",
               fMin, fMin + Band, firstBinExt, lastBinExt, fMinSFT, fMinSFT + (numBinsSFT-1) * df, firstBinSFT, lastBinSFT );

  INT4 firstBinOffset = firstBinExt - firstBinSFT;

  if ( (*outSFT) == NULL ) {
    XLAL_CHECK ( ((*outSFT) = XLALCalloc(1, sizeof(*(*outSFT)))) != NULL, XLAL_ENOMEM );
  }
  if ( (*outSFT)->data == NULL ) {
    XLAL_CHECK ( ((*outSFT)->data = XLALCreateCOMPLEX8Vector ( numBinsExt )) != NULL, XLAL_EFUNC );
  }
  if ( (*outSFT)->data->length != numBinsExt ) {
    XLAL_CHECK ( ((*outSFT)->data->data = XLALRealloc ( (*outSFT)->data->data, numBinsExt * sizeof((*outSFT)->data->data[0]))) != NULL, XLAL_ENOMEM );
    (*outSFT)->data->length = numBinsExt;
  }

  COMPLEX8Vector *ptr = (*outSFT)->data;	// keep copy to data-pointer
  (*(*outSFT)) = (*inSFT);			// copy complete header
  (*outSFT)->data = ptr;	  		// restore data-pointer
  (*outSFT)->f0 = firstBinExt * df ;	  	// set correct new fMin

  /* copy the relevant part of the data */
  memcpy ( (*outSFT)->data->data, inSFT->data->data + firstBinOffset, numBinsExt * sizeof( (*outSFT)->data->data[0] ) );

  return XLAL_SUCCESS;

} // XLALExtractBandFromSFT()

/**
 * Return a MultiSFT vector containing only the bins in [fMin, fMin+Band].
 * Note: the output MultiSFT is guaranteed to "cover" the input boundaries 'fMin'
 * and 'fMin+Band', ie if necessary the output SFT contains one additional
 * bin on either end of the interval.
 *
 * This uses the conventions in XLALFindCoveringSFTBins() to determine
 * the 'effective' frequency-band to extract.
 *
 */
MultiSFTVector *
XLALExtractBandFromMultiSFTVector ( const MultiSFTVector *inSFTs,      ///< [in] input MultiSFTs
                                    REAL8 fMin,                                ///< [in] lower end of frequency interval to return
                                    REAL8 Band                         ///< [in] band width of frequency interval to return
                                    )
{
  XLAL_CHECK_NULL ( inSFTs != NULL, XLAL_EINVAL, "Invalid NULL input MultiSFT vector 'inSFTs'\n");
  XLAL_CHECK_NULL ( inSFTs->length > 0, XLAL_EINVAL, "Invalid zero-length input MultiSFT vector 'inSFTs'\n");
  XLAL_CHECK_NULL ( fMin >= 0, XLAL_EDOM, "Invalid negative frequency fMin = %g\n", fMin );
  XLAL_CHECK_NULL ( Band > 0, XLAL_EDOM, "Invalid non-positive Band = %g\n", Band );

  MultiSFTVector *ret = NULL;
  XLAL_CHECK_NULL ( (ret = XLALCalloc(1, sizeof(*ret))) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL ( (ret->data = XLALCalloc(inSFTs->length, sizeof(ret->data[0]))) != NULL, XLAL_ENOMEM );
  ret->length = inSFTs->length;

  for (UINT4 X = 0; X < inSFTs->length; X++) {
     XLAL_CHECK_NULL( (ret->data[X] = XLALExtractBandFromSFTVector(inSFTs->data[X], fMin, Band)) != NULL, XLAL_EFUNC );
  }

  return ret;
} //XLALExtractBandFromMultiSFTVector()

/**
 * Resize the frequency-band of a given multi-SFT vector to [f0, f0+Band].
 *
 * NOTE: If the frequency band is extended in any direction, the corresponding bins
 * will be set to zero
 *
 * NOTE2: This uses the conventions in XLALFindCoveringSFTBins() to determine
 * the 'effective' frequency-band to resize to, in order to coincide with SFT frequency bins.
 *
 */
int
XLALMultiSFTVectorResizeBand ( MultiSFTVector *multiSFTs,	///< [in/out] multi-SFT vector to resize
                               REAL8 f0,			///< [in] new start frequency
                               REAL8 Band			///< [in] new frequency Band
                               )
{
  XLAL_CHECK ( multiSFTs != NULL, XLAL_EINVAL );
  XLAL_CHECK ( f0 >= 0, XLAL_EINVAL );
  XLAL_CHECK ( Band >= 0, XLAL_EINVAL );

  for ( UINT4 X = 0; X < multiSFTs->length; X ++ ) {
    XLAL_CHECK ( XLALSFTVectorResizeBand ( multiSFTs->data[X], f0, Band ) == XLAL_SUCCESS, XLAL_EFUNC );
  }

  return XLAL_SUCCESS;

} // XLALMultiSFTVectorResizeBand()

/**
 * Resize the frequency-band of a given SFT vector to [f0, f0+Band].
 *
 * NOTE: If the frequency band is extended in any direction, the corresponding bins
 * will be set to zero
 *
 * NOTE2: This uses the conventions in XLALFindCoveringSFTBins() to determine
 * the 'effective' frequency-band to resize to, in order to coincide with SFT frequency bins.
 *
 */
int
XLALSFTVectorResizeBand ( SFTVector *SFTs,	///< [in/out] SFT vector to resize
                          REAL8 f0,		///< [in] new start frequency
                          REAL8 Band		///< [in] new frequency Band
                          )
{
  XLAL_CHECK ( SFTs != NULL, XLAL_EINVAL );
  XLAL_CHECK ( f0 >= 0, XLAL_EINVAL );
  XLAL_CHECK ( Band >= 0, XLAL_EINVAL );

  for ( UINT4 alpha = 0; alpha < SFTs->length; alpha ++ ) {
    XLAL_CHECK ( XLALSFTResizeBand ( &(SFTs->data[alpha]), f0, Band ) == XLAL_SUCCESS, XLAL_EFUNC );
  }

  return XLAL_SUCCESS;

} // XLALSFTVectorResizeBand()

/**
 * Resize the frequency-band of a given SFT to [f0, f0+Band].
 *
 * NOTE: If the frequency band is extended in any direction, the corresponding bins
 * will be set to zero
 *
 * NOTE2: This uses the conventions in XLALFindCoveringSFTBins() to determine
 * the 'effective' frequency-band to resize to, in order to coincide with SFT frequency bins.
 *
 */
int
XLALSFTResizeBand ( SFTtype *SFT,	///< [in/out] SFT to resize
                    REAL8 f0,		///< [in] new start frequency
                    REAL8 Band		///< [in] new frequency Band
                    )
{
  XLAL_CHECK ( SFT != NULL, XLAL_EINVAL );
  XLAL_CHECK ( f0 >= 0, XLAL_EINVAL );
  XLAL_CHECK ( Band >= 0, XLAL_EINVAL );


  REAL8 Tsft = TSFTfromDFreq ( SFT->deltaF );
  REAL8 f0In = SFT->f0;

  UINT4 firstBinIn = (UINT4) lround ( f0In / SFT->deltaF );

  UINT4 firstBinOut;
  UINT4 numBinsOut;
  XLAL_CHECK ( XLALFindCoveringSFTBins ( &firstBinOut, &numBinsOut, f0, Band, Tsft ) == XLAL_SUCCESS, XLAL_EFUNC );

  int firstRelative = firstBinOut - firstBinIn;

  XLAL_CHECK ( (SFT = XLALResizeCOMPLEX8FrequencySeries ( SFT, firstRelative, numBinsOut )) != NULL, XLAL_EFUNC );

  return XLAL_SUCCESS;

} // XLALSFTResizeBand()



/**
 * Adds SFT-data from MultiSFTvector 'b' to elements of MultiSFTVector 'a'
 *
 * NOTE: the inputs 'a' and 'b' must have consistent number of IFO, number of SFTs,
 * IFO-names, start-frequency, frequency-spacing, timestamps, units and number of bins.
 *
 * The 'name' field of input/output SFTs in 'a' is not modified!
 */
int
XLALMultiSFTVectorAdd ( MultiSFTVector *a,	/**< [in/out] MultiSFTVector to be added to */
                        const MultiSFTVector *b	/**< [in] MultiSFTVector data to be added */
                        )
{
  XLAL_CHECK ( a != NULL, XLAL_EINVAL );
  XLAL_CHECK ( b != NULL, XLAL_EINVAL );

  XLAL_CHECK ( a->length == b->length, XLAL_EINVAL );
  UINT4 numIFOs = a->length;

  for ( UINT4 X = 0; X < numIFOs; X ++ )
    {
      SFTVector *vect1 = a->data[X];
      SFTVector *vect2 = b->data[X];

      XLAL_CHECK ( strncmp ( vect1->data[0].name, vect2->data[0].name, 2 ) == 0, XLAL_EINVAL, "SFT detectors differ '%c%c' != '%c%c'\n", vect1->data[0].name[0], vect1->data[0].name[1], vect2->data[0].name[0], vect2->data[0].name[1] );

      XLAL_CHECK ( XLALSFTVectorAdd ( vect1, vect2 ) == XLAL_SUCCESS, XLAL_EFUNC, "XLALSFTVectorAdd() failed for SFTVector %d out of %d\n", X+1, numIFOs );

    } // for X < numIFOs

  return XLAL_SUCCESS;

} /* XLALMultiSFTVectorAdd() */


/**
 * Adds SFT-data from SFTvector 'b' to elements of SFTVector 'a'
 *
 * NOTE: the inputs 'a' and 'b' must have consistent number of SFTs,
 * start-frequency, frequency-spacing, timestamps, units and number of bins.
 *
 * The 'name' field of input/output SFTs in 'a' is not modified!
 */
int
XLALSFTVectorAdd ( SFTVector *a,	/**< [in/out] SFTVector to be added to */
                   const SFTVector *b	/**< [in] SFTVector data to be added */
                   )
{
  XLAL_CHECK ( a != NULL, XLAL_EINVAL );
  XLAL_CHECK ( b != NULL, XLAL_EINVAL );

  XLAL_CHECK ( a->length == b->length, XLAL_EINVAL );
  UINT4 numSFTs = a->length;

  for ( UINT4 k = 0; k < numSFTs; k ++ )
    {
      SFTtype *sft1 = &(a->data[k]);
      SFTtype *sft2 = &(b->data[k]);

      XLAL_CHECK ( XLALSFTAdd ( sft1, sft2 ) == XLAL_SUCCESS, XLAL_EFUNC, "XLALSFTAdd() failed for SFTs k = %d out of %d SFTs\n", k, numSFTs );

    } // for k < numSFTs

  return XLAL_SUCCESS;
} /* XLALSFTVectorAdd() */


/**
 * Adds SFT-data from SFT 'b' to SFT 'a'
 *
 * NOTE: the inputs 'a' and 'b' must have consistent
 * start-frequency, frequency-spacing, timestamps, units and number of bins.
 *
 * The 'name' field of input/output SFTs in 'a' is not modified!
 */
int
XLALSFTAdd ( SFTtype *a,		/**< [in/out] SFT to be added to */
             const SFTtype *b	/**< [in] SFT data to be added */
             )
{
  XLAL_CHECK ( a != NULL, XLAL_EINVAL );
  XLAL_CHECK ( b != NULL, XLAL_EINVAL );
  XLAL_CHECK ( a->data != NULL, XLAL_EINVAL );
  XLAL_CHECK ( b->data != NULL, XLAL_EINVAL );

  XLAL_CHECK ( XLALGPSDiff ( &(a->epoch), &(b->epoch) ) == 0, XLAL_EINVAL, "SFT epochs differ %"LAL_INT8_FORMAT" != %"LAL_INT8_FORMAT" ns\n", XLALGPSToINT8NS ( &(a->epoch) ), XLALGPSToINT8NS ( &(b->epoch) ) );

  REAL8 tol = 10 * LAL_REAL8_EPS;	// generously allow up to 10*eps tolerance
  XLAL_CHECK ( gsl_fcmp ( a->f0, b->f0, tol ) == 0, XLAL_ETOL, "SFT frequencies relative deviation exceeds %g: %.16g != %.16g\n", tol, a->f0, b->f0 );
  XLAL_CHECK ( gsl_fcmp ( a->deltaF, b->deltaF, tol ) == 0, XLAL_ETOL, "SFT frequency-steps relative deviation exceeds %g: %.16g != %.16g\n", tol, a->deltaF, b->deltaF );
  XLAL_CHECK ( XLALUnitCompare ( &(a->sampleUnits), &(b->sampleUnits) ) == 0, XLAL_EINVAL, "SFT sample units differ\n" );
  XLAL_CHECK ( a->data->length == b->data->length, XLAL_EINVAL, "SFT lengths differ: %d != %d\n", a->data->length, b->data->length );

  UINT4 numBins = a->data->length;
  for ( UINT4 k = 0; k < numBins; k ++ )
    {
      a->data->data[k] += b->data->data[k];
    }

  return XLAL_SUCCESS;

} /* XLALSFTAdd() */

/**
 * Return the 'effective' frequency-band [fMinEff, fMaxEff] = [firstBin, lastBin] * 1/Tsft,
 * with numBins = lastBin - firstBin + 1
 * which is the smallest band of SFT-bins that fully covers a given band [fMin, fMin+Band]
 *
 * ==> calculate "effective" fMinEff by rounding down from fMin to closest (firstBin/Tsft)
 * and rounds up in the same way to fMaxEff = (lastBin/Tsft).
 *
 * The 'fudge region' allowing for numerical noise is eps= 10*LAL_REAL8_EPS ~2e-15
 * relative deviation: ie if the SFT contains a bin at 'fi', then we consider for example
 * "fMin == fi" if  fabs(fi - fMin)/fi < eps.
 */
int
XLALFindCoveringSFTBins ( UINT4 *firstBin,	///< [out] effective lower frequency-bin fMinEff = firstBin/Tsft
                          UINT4 *numBins,	///< [out] effective Band of SFT-bins, such that BandEff = (numBins-1)/Tsft
                          REAL8 fMinIn,		///< [in] input lower frequency
                          REAL8 BandIn,		///< [in] input frequency band
                          REAL8 Tsft		///< [in] SFT duration 'Tsft'
                          )
{
  XLAL_CHECK ( firstBin != NULL, XLAL_EINVAL );
  XLAL_CHECK ( numBins  != NULL, XLAL_EINVAL );
  XLAL_CHECK ( fMinIn >= 0, XLAL_EDOM );
  XLAL_CHECK ( BandIn >= 0, XLAL_EDOM );
  XLAL_CHECK ( Tsft > 0, XLAL_EDOM );

  volatile REAL8 dFreq = 1.0 / Tsft;
  volatile REAL8 tmp;
  // NOTE: don't "simplify" this: we try to make sure
  // the result of this will be guaranteed to be IEEE-compliant,
  // and identical to other locations, such as in SFT-IO

  // ----- lower effective frequency
  tmp = fMinIn / dFreq;
  UINT4 imin = (UINT4) floor( tmp * fudge_up );	// round *down*, allowing for eps 'fudge'

  // ----- upper effective frequency
  REAL8 fMaxIn = fMinIn + BandIn;
  tmp = fMaxIn / dFreq;
  UINT4 imax = (UINT4) ceil ( tmp * fudge_down );  // round *up*, allowing for eps fudge

  // ----- effective band
  UINT4 num_bins = (UINT4) (imax - imin + 1);

  // ----- return these
  (*firstBin) = imin;
  (*numBins)  = num_bins;

  return XLAL_SUCCESS;

} // XLALFindCoveringSFTBins()

/** Finds the earliest timestamp in a multi-SFT data structure
 *
*/
int
XLALEarliestMultiSFTsample ( LIGOTimeGPS *out,              /**< [out] earliest GPS time */
                             const MultiSFTVector *multisfts      /**< [in] multi SFT vector */
                             )
{
  UINT4 i,j;

  /* check sanity of input */
  if ( !multisfts || (multisfts->length == 0) )
    {
      XLALPrintError ("%s: empty multiSFT input!\n", __func__ );
      XLAL_ERROR (XLAL_EINVAL);
    }
  for (i=0;i<multisfts->length;i++)
    {
      if ( !multisfts->data[i] || (multisfts->data[i]->length == 0) )
        {
          XLALPrintError ("%s: empty multiSFT->data[%d] input!\n", __func__,i );
          XLAL_ERROR (XLAL_EINVAL);
        }
    }

  /* initialise the earliest sample value */
  out->gpsSeconds = multisfts->data[0]->data[0].epoch.gpsSeconds;
  out->gpsNanoSeconds = multisfts->data[0]->data[0].epoch.gpsNanoSeconds;

  /* loop over detectors */
  for (i=0;i<multisfts->length;i++) {

    /* loop over all SFTs to determine the earliest SFT epoch */
    for (j=0;j<multisfts->data[i]->length;j++) {

      /* compare current SFT epoch with current earliest */
      if ( (XLALGPSCmp(out,&multisfts->data[i]->data[j].epoch) == 1 ) ) {
        out->gpsSeconds = multisfts->data[i]->data[j].epoch.gpsSeconds;
        out->gpsNanoSeconds = multisfts->data[i]->data[j].epoch.gpsNanoSeconds;
      }

    }

  }

  /* success */
  return XLAL_SUCCESS;

} /* XLALEarliestMultiSFTsample() */

/** Find the time of the end of the latest SFT in a multi-SFT data structure
 *
*/
int
XLALLatestMultiSFTsample ( LIGOTimeGPS *out,              /**< [out] latest GPS time */
                           const MultiSFTVector *multisfts      /**< [in] multi SFT vector */
                           )
{
  UINT4 i,j;
  SFTtype *firstSFT;

  /* check sanity of input */
  if ( !multisfts || (multisfts->length == 0) )
    {
      XLALPrintError ("%s: empty multiSFT input!\n", __func__ );
      XLAL_ERROR (XLAL_EINVAL);
    }
  for (i=0;i<multisfts->length;i++)
    {
      if ( !multisfts->data[i] || (multisfts->data[i]->length == 0) )
        {
          XLALPrintError ("%s: empty multiSFT->data[%d] input!\n", __func__,i );
          XLAL_ERROR (XLAL_EINVAL);
        }
    }

  /* define some useful quantities */
  firstSFT = (multisfts->data[0]->data);        /* a pointer to the first SFT of the first detector */
  REAL8 Tsft = TSFTfromDFreq ( firstSFT->deltaF );    /* the length of the SFTs in seconds assuming 1/T freq resolution */

  /* initialise the latest sample value */
  out->gpsSeconds = firstSFT->epoch.gpsSeconds;
  out->gpsNanoSeconds = firstSFT->epoch.gpsNanoSeconds;

  /* loop over detectors */
  for (i=0;i<multisfts->length;i++) {

    /* loop over all SFTs to determine the earliest SFT midpoint of the input data in the SSB frame */
    for (j=0;j<multisfts->data[i]->length;j++) {

      /* compare current SFT epoch with current earliest */
      if ( (XLALGPSCmp(out,&multisfts->data[i]->data[j].epoch) == -1 ) ) {
        out->gpsSeconds = multisfts->data[i]->data[j].epoch.gpsSeconds;
        out->gpsNanoSeconds = multisfts->data[i]->data[j].epoch.gpsNanoSeconds;
      }
    }

  }

  /* add length of SFT to the result so that we output the end of the SFT */
  if ( XLALGPSAdd(out,Tsft) == NULL )
    {
      XLALPrintError ("%s: NULL pointer returned from XLALGPSAdd()!\n", __func__ );
      XLAL_ERROR (XLAL_EFAULT);
    }

  /* success */
  return XLAL_SUCCESS;

} /* XLALLatestMultiSFTsample() */

/**
 * Create a 'fake' SFT catalog which contains only detector and timestamp information.
 */
SFTCatalog *
XLALAddToFakeSFTCatalog ( SFTCatalog *catalog,                          /**< [in] SFT catalog; if NULL, a new catalog is created */
                          const CHAR *detector,                         /**< [in] Name of detector to set fake catalog entries to */
                          const LIGOTimeGPSVector *timestamps           /**< [in] Timestamps of each fake catalog entry */
                          )
{

  // Check input
  XLAL_CHECK_NULL( detector != NULL, XLAL_EFAULT );
  XLAL_CHECK_NULL( timestamps != NULL, XLAL_EFAULT );
  XLAL_CHECK_NULL( timestamps->length > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL( timestamps->data != NULL, XLAL_EFAULT );

  // Get channel prefix
  CHAR *channel = XLALGetChannelPrefix( detector );
  XLAL_CHECK_NULL( channel != NULL, XLAL_EFUNC );

  // If catalog is NULL, create a new fake catalog
  if (catalog == NULL) {
    catalog = XLALCalloc(1, sizeof(*catalog));
    XLAL_CHECK_NULL( catalog != NULL, XLAL_ENOMEM );
  }

  // Extend catalog to add new timestamps
  const UINT4 new_length = catalog->length + timestamps->length;
  catalog->data = XLALRealloc(catalog->data, new_length * sizeof(catalog->data[0]));
  XLAL_CHECK_NULL( catalog->data != NULL, XLAL_ENOMEM );

  // Fill out new SFT descriptors with channel name and timestamps info
  for (UINT4 i = 0; i < timestamps->length; ++i) {
    SFTDescriptor *desc = &catalog->data[catalog->length + i];
    memset(desc, 0, sizeof(*desc));
    strncpy(desc->header.name, channel, 2);
    desc->header.epoch = timestamps->data[i];
    desc->header.deltaF = 1.0 / timestamps->deltaT;
    desc->header.sampleUnits = lalDimensionlessUnit;
    desc->version = 2;
  }

  // Set new catalog length
  catalog->length = new_length;

  // Sort catalog
  qsort( (void*)catalog->data, catalog->length, sizeof( catalog->data[0] ), compareSFTdesc );

  // Cleanup
  XLALFree(channel);

  return catalog;

} /* XLALAddToFakeSFTCatalog() */

/**
 * Multi-detector and multi-timestamp wrapper of XLALAddToFakeSFTCatalog().
 */
SFTCatalog *
XLALMultiAddToFakeSFTCatalog ( SFTCatalog *catalog,                          /**< [in] SFT catalog; if NULL, a new catalog is created */
                               const LALStringVector *detectors,             /**< [in] Detector names to set fake catalog entries to */
                               const MultiLIGOTimeGPSVector *timestamps      /**< [in] Timestamps for each detector of each fake catalog entry */
                               )
{
  // Check input
  XLAL_CHECK_NULL( detectors != NULL, XLAL_EFAULT );
  XLAL_CHECK_NULL( detectors->length > 0, XLAL_EINVAL );
  XLAL_CHECK_NULL( detectors->data != NULL, XLAL_EFAULT );
  XLAL_CHECK_NULL( timestamps != NULL, XLAL_EFAULT );
  XLAL_CHECK_NULL( timestamps->length == detectors->length, XLAL_EINVAL );
  XLAL_CHECK_NULL( timestamps->data != NULL, XLAL_EFAULT );

  // Loop over detectors, calling XLALAddToFakeSFTCatalog()
  for (UINT4 X = 0; X < detectors->length; ++X) {
    catalog = XLALAddToFakeSFTCatalog( catalog, detectors->data[X], timestamps->data[X] );
    XLAL_CHECK_NULL( catalog != NULL, XLAL_EFUNC );
  }

  return catalog;

} /* XLALMultiAddToFakeSFTCatalog() */

///
/// Set a SFT catalog 'slice' to a timeslice of a larger SFT catalog 'catalog', with entries
/// restricted to the interval ['minStartGPS','maxStartGPS') according to XLALCWGPSinRange().
/// The catalog 'slice' just points to existing data in 'catalog', and therefore should not
/// be deallocated.
///
int XLALSFTCatalogTimeslice(
  SFTCatalog *slice,			///< [out] Timeslice of SFT catalog
  const SFTCatalog *catalog,		///< [in] SFT catalog
  const LIGOTimeGPS *minStartGPS,	///< [in] Minimum starting GPS time
  const LIGOTimeGPS *maxStartGPS	///< [in] Maximum starting GPS time
  )
{

  // Check input
  XLAL_CHECK( slice != NULL, XLAL_EFAULT );
  XLAL_CHECK( catalog != NULL, XLAL_EFAULT );
  XLAL_CHECK( minStartGPS != NULL && maxStartGPS != NULL, XLAL_EFAULT );
  XLAL_CHECK( catalog->length > 0, XLAL_EINVAL );
  XLAL_CHECK( XLALGPSCmp( minStartGPS, maxStartGPS ) < 1 , XLAL_EINVAL , "minStartGPS (%"LAL_GPS_FORMAT") is greater than maxStartGPS (%"LAL_GPS_FORMAT")\n",
              LAL_GPS_PRINT(*minStartGPS), LAL_GPS_PRINT(*maxStartGPS) );

  // get a temporary timestamps vector with SFT epochs so we can call XLALFindTimesliceBounds()
  LIGOTimeGPSVector timestamps;
  timestamps.length = catalog->length;
  XLAL_CHECK ( (timestamps.data = XLALCalloc ( timestamps.length, sizeof(timestamps.data[0]) )) != NULL, XLAL_ENOMEM );
  for ( UINT4 i = 0; i < timestamps.length; i ++ ) {
    timestamps.data[i] = catalog->data[i].header.epoch;
  }

  UINT4 iStart, iEnd;
  XLAL_CHECK ( XLALFindTimesliceBounds ( &iStart, &iEnd, &timestamps, minStartGPS, maxStartGPS ) == XLAL_SUCCESS, XLAL_EFUNC );
  XLALFree ( timestamps.data );

  // Initialise timeslice of SFT catalog
  XLAL_INIT_MEM(*slice);

  // If not empty: set timeslice of SFT catalog
  if ( iStart < iEnd )
    {
      slice->length = iEnd - iStart + 1;
      slice->data = &catalog->data[iStart];
    }

  return XLAL_SUCCESS;

} // XLALSFTCatalogTimeslice()

// Find index values of first and last timestamp within given timeslice range XLALCWGPSinRange(minStartGPS, maxStartGPS)
int
XLALFindTimesliceBounds ( UINT4 *iStart,
                          UINT4 *iEnd,
                          const LIGOTimeGPSVector *timestamps,
                          const LIGOTimeGPS *minStartGPS,
                          const LIGOTimeGPS *maxStartGPS
                          )
{
  XLAL_CHECK ( (iStart != NULL) && (iEnd != NULL) && (timestamps != NULL) && (minStartGPS != NULL) && (maxStartGPS != NULL), XLAL_EINVAL );
  XLAL_CHECK( XLALGPSCmp( minStartGPS, maxStartGPS ) < 1 , XLAL_EINVAL , "minStartGPS (%"LAL_GPS_FORMAT") is greater than maxStartGPS (%"LAL_GPS_FORMAT")\n",
              LAL_GPS_PRINT(*minStartGPS), LAL_GPS_PRINT(*maxStartGPS) );

  UINT4 N = timestamps->length;
  (*iStart) = 0;
  (*iEnd)   = N - 1;

  // check if there's any timestamps falling into the requested timeslice at all
  if( ( ( XLALCWGPSinRange( timestamps->data[0], minStartGPS, maxStartGPS ) == 1 ) || ( XLALCWGPSinRange( timestamps->data[N-1], minStartGPS, maxStartGPS ) == -1 ) ) )
    {// if not: set an emtpy index interval in this case
      (*iStart) = 1;
      (*iEnd)   = 0;
      XLALPrintInfo ("Returning empty timeslice: Timestamps span [%"LAL_GPS_FORMAT", %"LAL_GPS_FORMAT "]"
                     " has no overlap with requested timeslice range [%"LAL_GPS_FORMAT", %"LAL_GPS_FORMAT").\n",
                     LAL_GPS_PRINT(timestamps->data[0]), LAL_GPS_PRINT(timestamps->data[N-1]),
                     LAL_GPS_PRINT(*minStartGPS), LAL_GPS_PRINT(*maxStartGPS)
                     );
      return XLAL_SUCCESS;
    }

  while ( (*iStart) <= (*iEnd) && XLALCWGPSinRange ( timestamps->data[ (*iStart) ], minStartGPS, maxStartGPS ) < 0 ) {
    ++ (*iStart);
  }
  while ( (*iStart) <= (*iEnd) && XLALCWGPSinRange ( timestamps->data[ (*iEnd) ], minStartGPS, maxStartGPS ) > 0 ) {
    -- (*iEnd);
  }
  // note: *iStart >=0, *iEnd >= 0 is now guaranteed due to previous range overlap-check

  // check if there is any timestamps found witin the interval, ie if iStart <= iEnd
  if ( (*iStart) > (*iEnd) )
    {
      XLALPrintInfo ( "Returning empty timeslice: no sfttimes fall within given GPS range [%"LAL_GPS_FORMAT", %"LAL_GPS_FORMAT"). "
                      "Closest timestamps are: %"LAL_GPS_FORMAT" and %"LAL_GPS_FORMAT"\n",
                      LAL_GPS_PRINT(*minStartGPS), LAL_GPS_PRINT(*maxStartGPS),
                      LAL_GPS_PRINT(timestamps->data[ (*iEnd) ]), LAL_GPS_PRINT(timestamps->data[ (*iStart) ])
                      );
      (*iStart) = 1;
      (*iEnd)   = 0;
    }

  return XLAL_SUCCESS;

} // XLALFindTimesliceBounds()


/**
 * Copy an entire SFT-type into another.
 * We require the destination-SFT to have a NULL data-entry, as the
 * corresponding data-vector will be allocated here and copied into
 *
 * Note: the source-SFT is allowed to have a NULL data-entry,
 * in which case only the header is copied.
 */
int
XLALCopySFT ( SFTtype *dest, 		/**< [out] copied SFT (needs to be allocated already) */
              const SFTtype *src	/**< input-SFT to be copied */
              )
{
  // check input sanity
  XLAL_CHECK ( dest != NULL, XLAL_EINVAL );
  XLAL_CHECK ( dest->data == NULL, XLAL_EINVAL );
  XLAL_CHECK ( src != NULL, XLAL_EINVAL );

  /* copy complete head (including data-pointer, but this will be separately alloc'ed and copied in the next step) */
  (*dest) = (*src);	// struct copy

  /* copy data (if there's any )*/
  if ( src->data )
    {
      UINT4 numBins = src->data->length;
      XLAL_CHECK ( (dest->data = XLALCreateCOMPLEX8Vector ( numBins )) != NULL, XLAL_EFUNC );
      memcpy ( dest->data->data, src->data->data, numBins * sizeof (dest->data->data[0]));
    }

  return XLAL_SUCCESS;

} // XLALCopySFT()

/**
 * Extract an SFTVector from another SFTVector but only those timestamps matching
 *
 * Timestamps must be a subset of those sfts in the SFTVector or an error occurs
 */
SFTVector *
XLALExtractSFTVectorWithTimestamps ( const SFTVector *sfts,                 /**< input SFTs */
                                     const LIGOTimeGPSVector *timestamps    /**< timestamps */
                                     )
{
  // check input sanity
  XLAL_CHECK_NULL( sfts != NULL, XLAL_EINVAL);
  XLAL_CHECK_NULL( timestamps != NULL, XLAL_EINVAL );
  XLAL_CHECK_NULL( sfts->length >= timestamps->length, XLAL_EINVAL );

  SFTVector *ret = NULL;
  XLAL_CHECK_NULL( (ret = XLALCreateSFTVector(timestamps->length, 0)) != NULL, XLAL_EFUNC );

  UINT4 indexOfInputSFTVector = 0;
  UINT4 numberOfSFTsLoadedIntoOutputVector = 0;
  for (UINT4 ii=0; ii<timestamps->length; ii++)
    {
      XLAL_CHECK_NULL( indexOfInputSFTVector < sfts->length, XLAL_FAILURE, "At least one timestamp is not in the range specified by the SFT vector" );

      for (UINT4 jj=indexOfInputSFTVector; jj<sfts->length; jj++)
        {
        if ( XLALGPSCmp(&(sfts->data[jj].epoch), &(timestamps->data[ii])) == 0 )
          {
            indexOfInputSFTVector = jj+1;
            XLAL_CHECK_NULL( XLALCopySFT(&(ret->data[ii]), &(sfts->data[jj])) == XLAL_SUCCESS, XLAL_EFUNC );
            numberOfSFTsLoadedIntoOutputVector++;
            break;
          } // if SFT epoch matches timestamp epoch
        } // for jj < sfts->length
    } // for ii < timestamps->length

  XLAL_CHECK_NULL( numberOfSFTsLoadedIntoOutputVector == ret->length, XLAL_FAILURE, "Not all timestamps were found in the input SFT vector" );

  return ret;

} // XLALExtractSFTVectorWithTimestamps

/**
 * Extract a MultiSFTVector from another MultiSFTVector but only those timestamps matching
 *
 * Timestamps in each LIGOTimeGPSVector must be a subset of those sfts in each SFTVector or an error occurs
 */
MultiSFTVector *
XLALExtractMultiSFTVectorWithMultiTimestamps ( const MultiSFTVector *multiSFTs,                 /**< input SFTs */
                                               const MultiLIGOTimeGPSVector *multiTimestamps    /**< timestamps */
                                               )
{
  // check input sanity
  XLAL_CHECK_NULL( multiSFTs != NULL, XLAL_EINVAL);
  XLAL_CHECK_NULL( multiTimestamps != NULL, XLAL_EINVAL );

  MultiSFTVector *ret = NULL;
  XLAL_CHECK_NULL( (ret = XLALCalloc(1, sizeof(*ret))) != NULL, XLAL_ENOMEM );
  XLAL_CHECK_NULL( (ret->data = XLALCalloc(multiSFTs->length, sizeof(*ret->data))) != NULL, XLAL_ENOMEM );
  ret->length = multiSFTs->length;

  for (UINT4 X=0; X<multiSFTs->length; X++)
    {
       XLAL_CHECK_NULL( (ret->data[X] = XLALExtractSFTVectorWithTimestamps(multiSFTs->data[X], multiTimestamps->data[X])) != NULL, XLAL_EFUNC );
    }

  return ret;

} // XLALExtractMultiSFTVectorWithMultiTimestamps

/**
 * Create a complete copy of an SFT vector
 */
SFTVector *
XLALDuplicateSFTVector ( const SFTVector *sftsIn )
{
  XLAL_CHECK_NULL ( (sftsIn != NULL) && ( sftsIn->length > 0), XLAL_EINVAL );

  UINT4 numSFTs = sftsIn->length;
  UINT4 numBins = sftsIn->data[0].data->length;

  SFTVector *sftsOut;
  XLAL_CHECK_NULL ( (sftsOut = XLALCreateSFTVector ( numSFTs, numBins )) != NULL, XLAL_EFUNC );

  for ( UINT4 alpha=0; alpha < numSFTs; alpha++ )
    {
      SFTtype *thisSFTIn = &sftsIn->data[alpha];
      SFTtype *thisSFTOut = &sftsOut->data[alpha];

      COMPLEX8Vector *tmp = thisSFTOut->data;
      memcpy ( thisSFTOut, thisSFTIn, sizeof(*thisSFTOut) );
      thisSFTOut->data = tmp;
      thisSFTOut->data->length = numBins;
      memcpy ( thisSFTOut->data->data, thisSFTIn->data->data, numBins * sizeof(thisSFTOut->data->data[0]) );

    } // for alpha < numSFTs

  return sftsOut;

} // XLALDuplicateSFTVector()

/**
 * Reorder the MultiSFTVector with specified list of IFOs
 */
int XLALReorderMultiSFTVector( MultiSFTVector *multiSFTs, const LALStringVector *IFOs)
{
  XLAL_CHECK( multiSFTs!=NULL && IFOs!=NULL && multiSFTs->length==IFOs->length && multiSFTs->length<=PULSAR_MAX_DETECTORS, XLAL_EINVAL );

  // Initialize array of reordered SFTVector pointers
  SFTVector *reordered[PULSAR_MAX_DETECTORS];
  XLAL_INIT_MEM(reordered);

  // Loop through IFO list and reorder if necessary
  for (UINT4 i=0; i < IFOs->length; i ++ )
    {
      UINT4 j=0;
      while ( (j < IFOs->length) && (strncmp ( IFOs->data[i], multiSFTs->data[j]->data[0].name, 2 ) != 0) ) {
        j++;
      }
      XLAL_CHECK ( j < IFOs->length, XLAL_EINVAL, "IFO %c%c not found", IFOs->data[i][0], IFOs->data[i][1] );
      reordered[i] = multiSFTs->data[j]; // copy the SFTVector pointer
    }

  // Replace the old pointers with the new values
  for ( UINT4 i=0; i < multiSFTs->length; i ++ )
    {
      multiSFTs->data[i] = reordered[i];
    }

  return XLAL_SUCCESS;

} // XLALReorderMultiSFTVector()

///
/// Function to 'safely' invert Tsft=1/dFreq to avoid roundoff error for close-but-not-quite integers after inversion
/// policy: if (1/dFreq) is within 10 * eps of an integer, round, otherwise leave as a fractional number
/// comment: unfortunately Tsft is allowed by spec to be a double, but really should be limited to integer seconds,
/// however with this function we should be able to safely work with integer-valued Tsft without leaving spec (yet)
REAL8
TSFTfromDFreq ( REAL8 dFreq )
{
  REAL8 Tsft0 = 1.0 / dFreq;
  REAL8 Tsft;
  if ( fabs ( (Tsft0 - round(Tsft0)) ) / Tsft0 < 10 * LAL_REAL8_EPS ) {
    Tsft = round(Tsft0);
  } else {
    Tsft = Tsft0;
  }

  return Tsft;

} // TSFTfromDFreq()

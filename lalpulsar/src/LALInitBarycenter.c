/*
 * Copyright (C) 2010 Reinhard Prix (XLALified)
*  Copyright (C) 2007 Curt Cutler, Jolien Creighton, Reinhard Prix, Teviet Creighton, Bernd Machenschalk
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

#include <lal/FileIO.h>
#include <lal/LALBarycenter.h>
#include <lal/LALInitBarycenter.h>
#include <lal/ConfigFile.h>
#include <lal/LALString.h>
#include <lal/Date.h>

/** \cond DONT_DOXYGEN */

/* ----- defines and macros ---------- */
#define SQ(x) ((x) * (x))
#define NORM3D(x) ( SQ( (x)[0]) + SQ( (x)[1] ) + SQ ( (x)[2] ) )
#define LENGTH3D(x) ( sqrt( NORM3D ( (x) ) ) )

/** \endcond */

/* ----- local type definitions ---------- */

/**
 * Generic ephemeris-vector type, holding one timeseries of pos, vel, acceleration.
 * This is used for the generic ephemeris-reader XLAL-function, at the end the resulting
 * ephemeris-data  will be stored in the 'old type \a EphemerisData for backwards compatibility.
 */
typedef struct
{
  UINT4 length;      	/**< number of ephemeris-data entries */
  REAL8 dt;      	/**< spacing in seconds between consecutive instants in ephemeris table.*/
  PosVelAcc *data;    	/**< array containing pos,vel,acc as extracted from ephem file. Units are sec, 1, 1/sec respectively */
}
EphemerisVector;

/* ----- internal prototypes ---------- */
EphemerisVector *XLALCreateEphemerisVector ( UINT4 length );
void XLALDestroyEphemerisVector ( EphemerisVector *ephemV );

EphemerisVector * XLALReadEphemerisFile ( const CHAR *fname);
int XLALCheckEphemerisRanges ( const EphemerisVector *ephemEarth, REAL8 avg[3], REAL8 range[3] );

/* ----- function definitions ---------- */

/* ========== exported API ========== */

/**
 * An XLAL interface for reading a time correction file containing a table
 * of values for converting between Terrestrial Time TT (or TDT) to either
 * TDB (i.e. the file contains time corrections related to the Einstein delay)
 * or Teph (a time system from Irwin and Fukushima, 1999, closely related to
 * Coordinate Barycentric Time, TCB) depending on the file.
 *
 * The file contains a header with the GPS start time, GPS end time, time
 * interval between subsequent entries (seconds), and the number of entries.
 *
 * The rest of the file contains a list of the time delays (in seconds).
 *
 * The tables for the conversion to TDB and Teph are derived from the ephemeris
 * file TDB.1950.2050 and TIMEEPH_short.te405 within TEMPO2
 * http://www.atnf.csiro.au/research/pulsar/tempo2/. They are created from the
 * Chebychev polynomials in these files using the conversion in the lalapps code
 * lalapps_create_time_correction_ephemeris
 *
 * \ingroup LALBarycenter_h
 */
TimeCorrectionData *
XLALInitTimeCorrections ( const CHAR *timeCorrectionFile /**< File containing Earth's position.  */
                          )
{
  REAL8 *tvec = NULL; /* create time vector */
  LALParsedDataFile *flines = NULL;
  UINT4 numLines = 0, j = 0;
  REAL8 endtime = 0.;

  /* check user input consistency */
  if ( !timeCorrectionFile )
    XLAL_ERROR_NULL( XLAL_EINVAL, "Invalid NULL input for 'timeCorrectionFile'\n" );

  char *fname_path;
  XLAL_CHECK_NULL ( (fname_path = XLALPulsarFileResolvePath ( timeCorrectionFile )) != NULL, XLAL_EINVAL );

  /* read in file with XLALParseDataFile to ignore comment header lines */
  if ( XLALParseDataFile ( &flines, fname_path ) != XLAL_SUCCESS ) {
    XLALFree ( fname_path );
    XLAL_ERROR_NULL ( XLAL_EFUNC );
  }
  XLALFree ( fname_path );

  /* prepare output ephemeris struct for returning */
  TimeCorrectionData *tdat;
  if ( ( tdat = XLALCalloc ( 1, sizeof(*tdat) ) ) == NULL )
    XLAL_ERROR_NULL ( XLAL_ENOMEM, "XLALCalloc ( 1, %zu ) failed.\n", sizeof(*tdat) );

  numLines = flines->lines->nTokens;

  /* read in info from first line (an uncommented header) */
  if ( 4 != sscanf(flines->lines->tokens[0], "%lf %lf %lf %u", &tdat->timeCorrStart, &endtime, &tdat->dtTtable, &tdat->nentriesT) )
    {
      XLALDestroyParsedDataFile ( flines );
      XLALDestroyTimeCorrectionData( tdat );
      XLAL_ERROR_NULL ( XLAL_EDOM, "Couldn't parse first line of %s\n", timeCorrectionFile );
    }

  if( numLines - 1 != tdat->nentriesT )
    {
      XLALDestroyParsedDataFile ( flines );
      XLALDestroyTimeCorrectionData( tdat );
      XLAL_ERROR_NULL ( XLAL_EDOM, "Header says file has '%d' data-lines, but found '%d'.\n", tdat->nentriesT, numLines -1 );
    }

  /* allocate memory for table entries */
  if ( (tvec = XLALCalloc( tdat->nentriesT, sizeof(REAL8) )) == NULL )
    {
      XLALDestroyParsedDataFile ( flines );
      XLALDestroyTimeCorrectionData( tdat );
      XLAL_ERROR_NULL ( XLAL_ENOMEM, " XLALCalloc(%u, sizeof(REAL8))\n", tdat->nentriesT );
    }

  /* read in table data */
  int ret;
  for (j=1; j < numLines; j++)
    {
      if ( (ret = sscanf( flines->lines->tokens[j], "%lf", &tvec[j-1] )) != 1 )
        {
          XLALFree( tvec );
          XLALDestroyParsedDataFile ( flines );
          XLALDestroyTimeCorrectionData( tdat );
          XLAL_ERROR_NULL ( XLAL_EDOM, "Couldn't parse line %d of %s: read %d instead of 1\n", j+2, timeCorrectionFile, ret);
        }
    } // for j < numLines

  XLALDestroyParsedDataFile ( flines );

  /* set output time delay vector */
  tdat->timeCorrs = tvec;

  return tdat;

} /* XLALInitTimeCorrections() */

/**
 * Destructor for TimeCorrectionData struct, NULL robust.
 * \ingroup LALBarycenter_h
 */
void
XLALDestroyTimeCorrectionData ( TimeCorrectionData *tcd )
{
  if ( !tcd )
    return;

  if ( tcd->timeCorrs )
    XLALFree ( tcd->timeCorrs );

  XLALFree ( tcd );

  return;

} /* XLALDestroyTimeCorrectionData() */

/**
 * XLAL interface to reading ephemeris files 'earth' and 'sun', and return
 * ephemeris-data in old backwards-compatible type \a EphemerisData
 *
 * These ephemeris data files contain arrays
 * of center-of-mass positions for the Earth and Sun, respectively.
 *
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
 *
 * \ingroup LALBarycenter_h
 */
EphemerisData *
XLALInitBarycenter ( const CHAR *earthEphemerisFile,         /**< File containing Earth's position.  */
                     const CHAR *sunEphemerisFile            /**< File containing Sun's position. */
                     )
{
  EphemerisType sun_etype, earth_etype, etype;

  /* check user input consistency */
  if ( !earthEphemerisFile || !sunEphemerisFile )
    XLAL_ERROR_NULL (XLAL_EINVAL, "Invalid NULL input earthEphemerisFile=%p, sunEphemerisFile=%p\n", earthEphemerisFile, sunEphemerisFile );

  /* determine EARTH ephemeris type from file name*/
  if ( strstr( earthEphemerisFile, "DE200" ) )
    earth_etype = EPHEM_DE200;
  else if ( strstr( earthEphemerisFile, "DE405" ) )
    earth_etype = EPHEM_DE405;
  else if ( strstr( earthEphemerisFile, "DE414" ) )
    earth_etype = EPHEM_DE414;
  else if ( strstr( earthEphemerisFile, "DE421" ) )
    earth_etype = EPHEM_DE421;
  else
    earth_etype = EPHEM_DE405;

  /* determine SUN ephemeris type from file name */
  if ( strstr( sunEphemerisFile, "DE200" ) )
    sun_etype = EPHEM_DE200;
  else if ( strstr( sunEphemerisFile, "DE405" ) )
    sun_etype = EPHEM_DE405;
  else if ( strstr( sunEphemerisFile, "DE414" ) )
    sun_etype = EPHEM_DE414;
  else if ( strstr( sunEphemerisFile, "DE421" ) )
    sun_etype = EPHEM_DE421;
  else
    sun_etype = EPHEM_DE405;

  // check consistency
  if ( earth_etype != sun_etype )
    XLAL_ERROR_NULL (XLAL_EINVAL, "Earth '%s' and Sun '%s' ephemeris-files have inconsistent coordinate-types %d != %d\n",
                     earthEphemerisFile, sunEphemerisFile, earth_etype, sun_etype );
  else
    etype = earth_etype;

  EphemerisVector *ephemV;
  /* ----- read EARTH ephemeris file ---------- */
  if ( ( ephemV = XLALReadEphemerisFile ( earthEphemerisFile )) == NULL )
    XLAL_ERROR_NULL (XLAL_EFUNC, "XLALReadEphemerisFile('%s') failed\n", earthEphemerisFile );

  /* typical position, velocity and acceleration and allowed ranged */
  REAL8 avgE[3] = {499.0,  1e-4, 2e-11 };
  REAL8 rangeE[3] = {25.0, 1e-5, 3e-12 };

  if ( XLALCheckEphemerisRanges ( ephemV, avgE, rangeE ) != XLAL_SUCCESS )
    {
      XLALDestroyEphemerisVector ( ephemV );
      XLAL_ERROR_NULL ( XLAL_EFUNC, "Earth-ephemeris range error in XLALCheckEphemerisRanges()!\n" );
    }

  /* prepare output ephemeris struct for returning */
  EphemerisData *edat;
  if ( ( edat = XLALCalloc ( 1, sizeof(*edat) ) ) == NULL )
    XLAL_ERROR_NULL ( XLAL_ENOMEM, "XLALCalloc ( 1, %zu ) failed.\n", sizeof(*edat) );

  /* store in ephemeris-struct */
  edat->nentriesE = ephemV->length;
  edat->dtEtable  = ephemV->dt;
  edat->ephemE    = ephemV->data;
  edat->etype     = etype;
  XLALFree ( ephemV );	/* don't use 'destroy', as we linked the data into edat! */
  ephemV = NULL;

  /* ----- read SUN ephemeris file ---------- */
  if ( ( ephemV = XLALReadEphemerisFile ( sunEphemerisFile )) == NULL )
    {
      XLALDestroyEphemerisData ( edat );
      XLAL_ERROR_NULL ( XLAL_EFUNC, "XLALReadEphemerisFile('%s') failed\n", sunEphemerisFile );
    }

  /* typical position, velocity and acceleration and allowed ranged */
  REAL8 avgS[3]   = { 5.5, 5.5e-8, 50.5e-16 };
  REAL8 rangeS[3] = { 4.5, 4.5e-8, 49.5e-16 };

  if ( XLALCheckEphemerisRanges ( ephemV, avgS, rangeS ) != XLAL_SUCCESS )
    {
      XLALDestroyEphemerisVector ( ephemV );
      XLALDestroyEphemerisData ( edat );
      XLAL_ERROR_NULL ( XLAL_EFUNC, "Sun-ephemeris range error in XLALCheckEphemerisRanges()!\n" );
    }

  /* store in ephemeris-struct */
  edat->nentriesS = ephemV->length;
  edat->dtStable  = ephemV->dt;
  edat->ephemS    = ephemV->data;
  XLALFree ( ephemV );	/* don't use 'destroy', as we linked the data into edat! */
  ephemV = NULL;

  // store *copy* of ephemeris-file names in output structure
  edat->filenameE = XLALStringDuplicate( earthEphemerisFile );
  edat->filenameS = XLALStringDuplicate( sunEphemerisFile );

  /* return resulting ephemeris-data */
  return edat;

} /* XLALInitBarycenter() */


/**
 * Destructor for EphemerisData struct, NULL robust.
 * \ingroup LALBarycenter_h
 */
void
XLALDestroyEphemerisData ( EphemerisData *edat )
{
  if ( !edat )
    return;

  if ( edat->filenameE )
    XLALFree ( edat->filenameE );

  if ( edat->filenameS )
    XLALFree ( edat->filenameS );

  if ( edat->ephemE )
    XLALFree ( edat->ephemE );

  if ( edat->ephemS )
    XLALFree ( edat->ephemS );

  XLALFree ( edat );

  return;

} /* XLALDestroyEphemerisData() */


/**
 * Restrict the EphemerisData 'edat' to the smallest number of entries
 * required to cover the GPS time range ['startGPS', 'endGPS']
 *
 * \ingroup LALBarycenter_h
 */
int XLALRestrictEphemerisData ( EphemerisData *edat, const LIGOTimeGPS *startGPS, const LIGOTimeGPS *endGPS ) {

  // Check input
  XLAL_CHECK(edat != NULL, XLAL_EFAULT);
  XLAL_CHECK(startGPS != NULL, XLAL_EFAULT);
  XLAL_CHECK(endGPS != NULL, XLAL_EFAULT);
  XLAL_CHECK(XLALGPSCmp(startGPS, endGPS) < 0, XLAL_EINVAL);

  // Convert 'startGPS' and 'endGPS' to REAL8s
  const REAL8 start = XLALGPSGetREAL8(startGPS), end = XLALGPSGetREAL8(endGPS);

  // Increase 'ephemE' and decrease 'nentriesE' to fit the range ['start', 'end']
  PosVelAcc *const old_ephemE = edat->ephemE;
  do {
    if (edat->nentriesE > 1 && edat->ephemE[0].gps < start && edat->ephemE[1].gps <= start) {
      ++edat->ephemE;
      --edat->nentriesE;
    } else if (edat->nentriesE > 1 && edat->ephemE[edat->nentriesE-1].gps > end && edat->ephemE[edat->nentriesE-2].gps >= end) {
      --edat->nentriesE;
    } else {
      break;
    }
  } while(1);

  // Reallocate 'ephemE' to new table size, and free old table
  PosVelAcc *const new_ephemE = XLALMalloc(edat->nentriesE * sizeof(*new_ephemE));
  XLAL_CHECK(new_ephemE != NULL, XLAL_ENOMEM);
  memcpy(new_ephemE, edat->ephemE, edat->nentriesE * sizeof(*new_ephemE));
  edat->ephemE = new_ephemE;
  XLALFree(old_ephemE);

  // Increase 'ephemS' and decrease 'nentriesS' to fit the range ['start', 'end']
  PosVelAcc *const old_ephemS = edat->ephemS;
  do {
    if (edat->nentriesS > 1 && edat->ephemS[0].gps < start && edat->ephemS[1].gps <= start) {
      ++edat->ephemS;
      --edat->nentriesS;
    } else if (edat->nentriesS > 1 && edat->ephemS[edat->nentriesS-1].gps > end && edat->ephemS[edat->nentriesS-2].gps >= end) {
      --edat->nentriesS;
    } else {
      break;
    }
  } while(1);

  // Reallocate 'ephemS' to new table size, and free old table
  PosVelAcc *const new_ephemS = XLALMalloc(edat->nentriesS * sizeof(*new_ephemS));
  XLAL_CHECK(new_ephemS != NULL, XLAL_ENOMEM);
  memcpy(new_ephemS, edat->ephemS, edat->nentriesS * sizeof(*new_ephemS));
  edat->ephemS = new_ephemS;
  XLALFree(old_ephemS);

  return XLAL_SUCCESS;

} /* XLALRestrictEphemerisData() */


/* ========== internal function definitions ========== */

/** simple creator function for EphemerisVector type */
EphemerisVector *
XLALCreateEphemerisVector ( UINT4 length )
{
  EphemerisVector * ret;
  if ( ( ret = XLALCalloc ( 1, sizeof (*ret) )) == NULL )
    XLAL_ERROR_NULL ( XLAL_ENOMEM, "Failed to XLALCalloc(1, %zu)\n", sizeof (*ret) );

  if ( ( ret->data = XLALCalloc ( length, sizeof(*ret->data) ) ) == NULL )
    {
      XLALFree ( ret );
      XLAL_ERROR_NULL ( XLAL_ENOMEM, "Failed to XLALCalloc (%d, %zu)\n", length, sizeof(*ret->data) );
    }

  ret->length = length;

  return ret;

} /* XLALCreateEphemerisVector() */

/**
 * Destructor for EphemerisVector, NULL robust.
 */
void
XLALDestroyEphemerisVector ( EphemerisVector *ephemV )
{
  if ( !ephemV )
    return;

  if ( ephemV->data )
    XLALFree ( ephemV->data );

  XLALFree ( ephemV );

  return;

} /* XLALDestroyEphemerisVector() */


/** Simple wrapper to XLALFileResolvePathLong(), using hardcoded fallbackdir=PKG_DATA_DIR
 */
char *
XLALPulsarFileResolvePath ( const char *fname	  //!< [in] filename or file-path to resolve
                            )
{
  return XLALFileResolvePathLong ( fname, PKG_DATA_DIR );
} // XLALPulsarFileResolvePath()

/**
 * XLAL function to read ephemeris-data from one file, returning a EphemerisVector.
 * This is a helper-function to XLALInitBarycenter().
 *
 * NOTE: This function tries to read ephemeris from "<fname>" first, if that fails it also tries
 * to read "<fname>.gz" instead. This allows us to handle gzip-compressed ephemeris-files without having
 * to worry about the detailed filename extension used in the case of compression.
 *
 * NOTE2: files are searches first locally, then in LAL_DATA_PATH, and finally in PKG_DATA_DIR
 * using XLALPulsarFileResolvePath()
 */
EphemerisVector *
XLALReadEphemerisFile ( const CHAR *fname )
{
  /* check input consistency */
  XLAL_CHECK_NULL ( fname != NULL, XLAL_EINVAL );

  char *fname_path = NULL;

  // first check if "<fname>" can be resolved ...
  if ( (fname_path = XLALPulsarFileResolvePath ( fname )) == NULL )
    {
      // if not, check if we can find "<fname>.gz" instead ...
      char *fname_gz;
      XLAL_CHECK_NULL ( (fname_gz = XLALMalloc ( strlen(fname) + strlen(".gz") + 1 )) != NULL, XLAL_ENOMEM );
      sprintf ( fname_gz, "%s.gz", fname );
      if ( (fname_path = XLALPulsarFileResolvePath ( fname_gz )) == NULL )
        {
          XLALFree ( fname_gz );
          XLAL_ERROR_NULL ( XLAL_EINVAL, "Failed to find ephemeris-file '%s[.gz]'\n", fname );
        } // if 'fname_gz' could not be resolved
      XLALFree ( fname_gz );
    } // if 'fname' couldn't be resolved

  // if we're here, it means we found it

  // read in whole file (compressed or not) with XLALParseDataFile(), which ignores comment header lines
  LALParsedDataFile *flines = NULL;
  XLAL_CHECK_NULL ( XLALParseDataFile ( &flines, fname_path ) == XLAL_SUCCESS, XLAL_EFUNC );
  XLALFree ( fname_path );

  UINT4 numLines = flines->lines->nTokens;

  INT4 gpsYr; /* gpsYr + leap is the time on the GPS clock
               * at first instant of new year, UTC; equivalently
               * leap is # of leap secs added between Jan.6, 1980 and
               * Jan. 2 of given year
               */
  REAL8 dt;		/* ephemeris-file time-step in seconds */
  UINT4 nEntries;	/* number of ephemeris-file entries */

  /* read first line */
  if ( 3 != sscanf(flines->lines->tokens[0],"%d %le %u\n", &gpsYr, &dt, &nEntries))
    {
      XLALDestroyParsedDataFile( flines );
      XLAL_ERROR_NULL ( XLAL_EDOM, "Couldn't parse first line of %s\n", fname );
    }

  /* check that number of lines is correct */
  if( nEntries != (numLines - 1)/4 )
    {
      XLALDestroyParsedDataFile( flines );
      XLAL_ERROR_NULL ( XLAL_EDOM, "Inconsistent number of data-lines (%d) in file '%s' compared to header information (%d)\n", (numLines - 1)/4, fname, nEntries);
    }

  /* prepare output ephemeris vector */
  EphemerisVector *ephemV;
  if ( (ephemV = XLALCreateEphemerisVector ( nEntries )) == NULL )
    XLAL_ERROR_NULL ( XLAL_EFUNC, "Failed to XLALCreateEphemerisVector(%d)\n", nEntries );

  ephemV->dt = dt;

  /* first column in ephemeris-file is gps time--one long integer
   * giving the number of secs that have ticked since start of GPS epoch
   * +  on 1980 Jan. 6 00:00:00 UTC
   */

  /* the ephemeris files are created with each entry spanning 4 lines with the
   * format:
   *  gps\tposX\tposY\n
   *  posZ\tvelX\tvelY\n
   *  velZ\taccX\taccY\n
   *  accZ\n
   ***************************************************************************/

  /* read the remaining lines */
  for ( UINT4 j = 0; j < nEntries; j++ )
    {
      UINT4 i_line;
      int ret;

      i_line = 1 + 4*j;
      ret = sscanf( flines->lines->tokens[ i_line ], "%le %le %le\n", &ephemV->data[j].gps, &ephemV->data[j].pos[0], &ephemV->data[j].pos[1] );
      XLAL_CHECK_NULL ( ret == 3, XLAL_EDOM, "Couldn't parse line %d of %s: read %d items instead of 3\n", i_line, fname, ret );

      i_line ++;
      ret = sscanf( flines->lines->tokens[ i_line ], "%le %le %le\n", &ephemV->data[j].pos[2], &ephemV->data[j].vel[0], &ephemV->data[j].vel[1] );
      XLAL_CHECK_NULL ( ret == 3, XLAL_EDOM, "Couldn't parse line %d of %s: read %d items instead of 3\n", i_line, fname, ret );

      i_line ++;
      ret = sscanf( flines->lines->tokens[ i_line ], "%le %le %le\n", &ephemV->data[j].vel[2], &ephemV->data[j].acc[0], &ephemV->data[j].acc[1] );
      XLAL_CHECK_NULL ( ret == 3, XLAL_EDOM, "Couldn't parse line %d of %s: read %d items instead of 3\n", i_line, fname, ret );

      i_line ++;
      ret = sscanf( flines->lines->tokens[ i_line ], "%le\n", &ephemV->data[j].acc[2] );
      XLAL_CHECK_NULL ( ret == 1, XLAL_EDOM, "Couldn't parse line %d of %s: read %d items instead of 1\n", i_line, fname, ret );

      /* check timestamps */
      if(j == 0)
        {
          if (gpsYr - ephemV->data[j].gps > 3600 * 24 * 365 )
            {
              XLALDestroyEphemerisVector ( ephemV );
              XLALDestroyParsedDataFile( flines );
              XLAL_ERROR_NULL ( XLAL_EDOM, "Wrong timestamp in line %d of %s: %d/%le\n", j+2, fname, gpsYr, ephemV->data[j].gps );
            }
        }
      else
        {
          if (ephemV->data[j].gps != ephemV->data[j-1].gps + ephemV->dt )
            {
              XLALDestroyEphemerisVector ( ephemV );
              XLALDestroyParsedDataFile( flines );
              XLAL_ERROR_NULL ( XLAL_EDOM, "Wrong timestamp in line %d of %s: %le/%le\n", j+2, fname, ephemV->data[j].gps, ephemV->data[j-1].gps + ephemV->dt );
            }
        }

    } /* for j < nEntries */

  XLALDestroyParsedDataFile( flines );

  /* return result */
  return ephemV;

} /* XLALReadEphemerisFile() */


/**
 * Function to check rough consistency of ephemeris-data with being an actual
 * 'Earth' ephemeris: ie check position, velocity and acceleration are within
 * reasonable ranges {avg +- range}. where 'avg' and 'range' are 3-D arrays
 * with [0]=position, [1]=velocity and [2]=acceleration
 */
int
XLALCheckEphemerisRanges ( const EphemerisVector *ephemV, REAL8 avg[3], REAL8 range[3] )
{
  /* check input consistency */
  if ( !ephemV )
    XLAL_ERROR ( XLAL_EINVAL, "Invalid NULL input for 'ephemV' \n" );

  UINT4 numEntries = ephemV->length;
  REAL8 dt = ephemV->dt;

  /* check position, velocity and acceleration */
  UINT4 j;
  REAL8 tjm1 = 0;
  for ( j=0; j < numEntries; j ++ )
    {
      REAL8 length;
      length = LENGTH3D ( ephemV->data[j].pos );
      if ( fabs( avg[0] - length) >  range[0] )
        XLAL_ERROR ( XLAL_EDOM, "Position out of range in entry %d: vr=(%le, %le, %le), sqrt{|vr|} = %le [%g +- %g]\n",
                     j, ephemV->data[j].pos[0], ephemV->data[j].pos[1], ephemV->data[j].pos[2], length, avg[0], range[0] );

      length = LENGTH3D ( ephemV->data[j].vel );
      if ( fabs(avg[1] - length) > range[1] ) /* 10% */
        XLAL_ERROR ( XLAL_EDOM, "Velocity out of range in entry %d: vv=(%le, %le, %le), sqrt{|vv|} = %le, [%g +- %g]\n",
                     j, ephemV->data[j].vel[0], ephemV->data[j].vel[1], ephemV->data[j].vel[2], length, avg[1], range[1] );

      length = LENGTH3D ( ephemV->data[j].acc );
      if ( fabs(avg[2] - length) > range[2] ) /* 15% */
        XLAL_ERROR ( XLAL_EDOM, "Acceleration out of range in entry %d: va=(%le, %le, %le), sqrt{|va|} = %le, [%g +- %g]\n",
                     j, ephemV->data[j].acc[0], ephemV->data[j].acc[1], ephemV->data[j].acc[2], length, avg[2], range[2] );

      /* check timestep */
      if ( j > 0 ) {
        if ( ephemV->data[j].gps - tjm1 != dt )
          XLAL_ERROR ( XLAL_EDOM, "Invalid timestep in entry %d: t_i - t_{i-1} = %g != %g\n", j, ephemV->data[j].gps - tjm1, dt );
      }
      tjm1 = ephemV->data[j].gps;	/* keep track of previous timestamp */

    } /* for j < nEntries */


  /* all seems ok */
  return XLAL_SUCCESS;

} /* XLALCheckEphemerisRanges() */

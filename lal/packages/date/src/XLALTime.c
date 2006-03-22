#include <math.h>
#include <lal/LALStdlib.h>
#include <lal/Date.h>

#define XLAL_BILLION_INT8 LAL_INT8_C( 1000000000 )
#define XLAL_BILLION_REAL8 1e9

/* mostly internal functions */

/** Converts GPS time to nano seconds stored as an INT8. */
INT8 XLALGPSToINT8NS( const LIGOTimeGPS *epoch )
{
  INT8 ns;
  ns  = XLAL_BILLION_INT8 * (INT8)epoch->gpsSeconds;
  ns += (INT8)epoch->gpsNanoSeconds;
  return ns;
}

/** Converts nano seconds stored as an INT8 to GPS time. */
LIGOTimeGPS * XLALINT8NSToGPS( LIGOTimeGPS *epoch, INT8 ns )
{
  epoch->gpsSeconds     = (INT4)( ns / XLAL_BILLION_INT8 );
  epoch->gpsNanoSeconds = (INT4)( ns % XLAL_BILLION_INT8 );
  return epoch;
}

/** Sets GPS time given GPS integer seconds and residual nanoseconds. */
LIGOTimeGPS * XLALGPSSet( LIGOTimeGPS *epoch, INT4 gpssec, INT4 gpsnan )
{
  INT8 ns;
  ns = XLAL_BILLION_INT8 * (INT8)gpssec + (INT8)gpsnan;
  XLALINT8NSToGPS( epoch, ns );
  return epoch;
}

/** Sets GPS time given GPS seconds as a REAL8. */
LIGOTimeGPS * XLALGPSSetREAL8( LIGOTimeGPS *epoch, REAL8 t )
{
  INT8 ns;
  ns = (INT8)floor( XLAL_BILLION_REAL8 * t + 0.5 );
  XLALINT8NSToGPS( epoch, ns );
  return epoch;
}

/** Returns the GPS time as a REAL8. */
REAL8 XLALGPSGetREAL8( const LIGOTimeGPS *epoch )
{
  return XLALGPSToINT8NS( epoch ) / XLAL_BILLION_REAL8;
}

/* general purpose functions */

/** Adds dt to a GPS time. */
LIGOTimeGPS * XLALGPSAdd( LIGOTimeGPS *epoch, REAL8 dt )
{
  INT8 ns;
  ns  = XLALGPSToINT8NS( epoch );
  ns += (INT8)floor( XLAL_BILLION_REAL8 * dt + 0.5 );
  XLALINT8NSToGPS( epoch, ns );
  return epoch;
}

/** Difference between two GPS times. */
REAL8 XLALGPSDiff( const LIGOTimeGPS *t1, const LIGOTimeGPS *t0 )
{
  REAL8 dt;
  INT8  ns;
  ns = XLALGPSToINT8NS( t1 ) - XLALGPSToINT8NS( t0 );
  dt = (REAL8)ns / XLAL_BILLION_REAL8; 
  return dt;
}

/** Compares two GPS times.
 * Returns:
 *  - -1 if t0 < t1
 *  - 0 if t0 == t1
 *  - 1 if t0 > t1.
 */
int XLALGPSCmp( const LIGOTimeGPS *t0, const LIGOTimeGPS *t1 )
{
  INT8 ns0;
  INT8 ns1;
  ns0 = XLALGPSToINT8NS( t0 );
  ns1 = XLALGPSToINT8NS( t1 );
  if ( ns0 < ns1 )
    return -1;
  else if ( ns0 > ns1 )
    return 1;
  else
    return 0;
}

/** Multiply a GPS time by a number. */
LIGOTimeGPS *XLALGPSMultiply( LIGOTimeGPS *gps, REAL8 x )
{
  LIGOTimeGPS hi, mi;
  INT8 lo = gps->gpsNanoSeconds * x + 0.5;

  XLALGPSSetREAL8(&mi, (gps->gpsSeconds % 512) * x);
  gps->gpsSeconds -= gps->gpsSeconds % 512;
  XLALGPSSetREAL8(&hi, gps->gpsSeconds * x);
  hi.gpsNanoSeconds += 256;
  hi.gpsNanoSeconds -= hi.gpsNanoSeconds % 512;

  XLALGPSSet(gps, hi.gpsSeconds + mi.gpsSeconds, hi.gpsNanoSeconds + mi.gpsNanoSeconds + lo);

  return gps;
}

/** Divide a GPS time by a number. */
LIGOTimeGPS *XLALGPSDivide( LIGOTimeGPS *gps, REAL8 x )
{
  LIGOTimeGPS quotient;
  int i;

  XLALGPSSet(&quotient, 0, 0);
  for(i = 0; i < 3; i++) {	/* is 3 too many? */
    LIGOTimeGPS tmp = quotient;
    XLALGPSAdd(&quotient, XLALGPSDiff(gps, XLALGPSMultiply(&tmp, x)) / x);
  }
  *gps = quotient;

  return gps;
}

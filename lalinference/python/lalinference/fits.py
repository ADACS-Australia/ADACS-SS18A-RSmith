#!/usr/bin/env python
#
# Copyright (C) 2013-2015  Leo Singer
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
"""
Reading and writing HEALPix FITS files. An example FITS header looks like this:

$ funhead -a test.fits.gz
SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                    8 / array data type
NAXIS   =                    0 / number of array dimensions
EXTEND  =                    T
END
      Extension: xtension

XTENSION= 'BINTABLE'           / binary table extension
BITPIX  =                    8 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                 4096 / length of dimension 1
NAXIS2  =                  192 / length of dimension 2
PCOUNT  =                    0 / number of group parameters
GCOUNT  =                    1 / number of groups
TFIELDS =                    1 / number of table fields
TTYPE1  = 'PROB    '
TFORM1  = '1024E   '
TUNIT1  = 'pix-1   '
PIXTYPE = 'HEALPIX '           / HEALPIX pixelisation
ORDERING= 'RING    '           / Pixel ordering scheme, either RING or NESTED
COORDSYS= 'C       '           / Ecliptic, Galactic or Celestial (equatorial)
EXTNAME = 'xtension'           / name of this binary table extension
NSIDE   =                  128 / Resolution parameter of HEALPIX
FIRSTPIX=                    0 / First pixel # (0 based)
LASTPIX =               196607 / Last pixel # (0 based)
INDXSCHM= 'IMPLICIT'           / Indexing: IMPLICIT or EXPLICIT
OBJECT  = 'FOOBAR 12345'       / Unique identifier for this event
REFERENC= 'http://www.youtube.com/watch?v=0ccKPSVQcFk' / URL of this event
DATE-OBS= '2013-04-08T21:37:32.25' / UTC date of the observation
MJD-OBS =      56391.151064815 / modified Julian date of the observation
DATE    = '2013-04-08T21:50:32' / UTC date of file creation
CREATOR = 'fits.py '           / Program that created this file
RUNTIME =                 21.5 / Runtime in seconds of the CREATOR program
END
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"
__all__ = ("read_sky_map", "write_sky_map")


import gzip
import math
import os
import shutil
import tempfile
import healpy as hp
from glue.ligolw import lsctables
from healpy.fitsfunc import getformat, pixelfunc, standard_column_names, pf, np
import time
import lal


#
# Based on https://github.com/healpy/healpy/blob/1.6.1/healpy/fitsfunc.py.
# Reproduced with permission from Andrea Zonca.
#
# Modifications:
#  * Added extra_metadata= argument to inject additional values into header.
#  * Added optional unit= argument to set units of table data.
#  * Support writing to gzip-compressed FITS files.
#
# FIXME: Instead of pyfits, use astropy.io.fits; it supports gzip compression.
#
def write_map(filename,m,nest=False,dtype=np.float32,fits_IDL=True,coord=None,column_names=None,unit=None,extra_metadata=()):
    """Writes an healpix map into an healpix file.

    Parameters
    ----------
    filename : str
      the fits file name
    m : array or sequence of 3 arrays
      the map to write. Possibly a sequence of 3 maps of same size.
      They will be considered as I, Q, U maps.
      Supports masked maps, see the `ma` function.
    nest : bool, optional
      If False, ordering scheme is NESTED, otherwise, it is RING. Default: RING.
    fits_IDL : bool, optional
      If True, reshapes columns in rows of 1024, otherwise all the data will
      go in one column. Default: True
    coord : str
      The coordinate system, typically 'E' for Ecliptic, 'G' for Galactic or 'C' for
      Celestial (equatorial)
    column_names : str or list
      Column name or list of column names, if None we use:
      I_STOKES for 1 component,
      I/Q/U_STOKES for 3 components,
      II, IQ, IU, QQ, QU, UU for 6 components,
      COLUMN_0, COLUMN_1... otherwise
    """
    if not hasattr(m, '__len__'):
        raise TypeError('The map must be a sequence')
    # check the dtype and convert it
    fitsformat = getformat(dtype)

    m = pixelfunc.ma_to_array(m)
    if pixelfunc.maptype(m) == 0: # a single map is converted to a list
        m = [m]

    if column_names is None:
        column_names = standard_column_names.get(len(m), ["COLUMN_%d" % n for n in range(len(m))])
    else:
        assert len(column_names) == len(m), "Length column_names != number of maps"

    # maps must have same length
    assert len(set(map(len, m))) == 1, "Maps must have same length"
    nside = pixelfunc.npix2nside(len(m[0]))

    if nside < 0:
        raise ValueError('Invalid healpix map : wrong number of pixel')

    cols=[]
    for cn, mm in zip(column_names, m):
        if len(mm) > 1024 and fits_IDL:
            # I need an ndarray, for reshape:
            mm2 = np.asarray(mm)
            cols.append(pf.Column(name=cn,
                                   format='1024%s' % fitsformat,
                                   array=mm2.reshape(mm2.size/1024,1024),
                                   unit=unit))
        else:
            cols.append(pf.Column(name=cn,
                                   format='%s' % fitsformat,
                                   array=mm,
                                   unit=unit))

    tbhdu = pf.new_table(cols)
    # add needed keywords
    tbhdu.header.update('PIXTYPE','HEALPIX','HEALPIX pixelisation')
    if nest: ordering = 'NESTED'
    else:    ordering = 'RING'
    tbhdu.header.update('ORDERING',ordering,
                        'Pixel ordering scheme, either RING or NESTED')
    if coord:
        tbhdu.header.update('COORDSYS',coord,
                            'Ecliptic, Galactic or Celestial (equatorial)')
    tbhdu.header.update('EXTNAME','xtension',
                        'name of this binary table extension')
    tbhdu.header.update('NSIDE',nside,'Resolution parameter of HEALPIX')
    tbhdu.header.update('FIRSTPIX', 0, 'First pixel # (0 based)')
    tbhdu.header.update('LASTPIX',pixelfunc.nside2npix(nside)-1,
                        'Last pixel # (0 based)')
    tbhdu.header.update('INDXSCHM','IMPLICIT',
                        'Indexing: IMPLICIT or EXPLICIT')

    for metadata in extra_metadata:
        tbhdu.header.update(*metadata)

    # FIXME: use with-clause, but GzipFile doesn't support it in Python 2.6.
    # We can't even use GzipFile because the ancient version of PyFITS that is
    # in SL6 is too broken, so we have to write the file and then compress it.
    basename, ext = os.path.splitext(filename)
    if ext == '.gz':
        with tempfile.NamedTemporaryFile(suffix='.fits') as tmpfile:
            tbhdu.writeto(tmpfile.name, clobber=True)
            gzfile = gzip.GzipFile(filename, 'wb')
            try:
                try:
                    shutil.copyfileobj(tmpfile, gzfile)
                finally:
                    gzfile.close()
            except:
                os.unlink(gzfile.name)
                raise
    else:
        tbhdu.writeto(filename, clobber=True)


def gps_to_iso8601(gps_time):
    """
    Convert a floating-point GPS time in seconds to an ISO 8601 date string.

    Parameters
    ----------

    gps : float
        Time in seconds since GPS epoch

    Returns
    -------

    iso8601 : str
        ISO 8601 date string (with fractional seconds)

    Example
    -------

    >>> gps_to_iso8601(1000000000.01)
    '2011-09-14T01:46:25.010000'
    >>> gps_to_iso8601(1000000000)
    '2011-09-14T01:46:25.000000'
    >>> gps_to_iso8601(1000000000.999999)
    '2011-09-14T01:46:25.999999'
    >>> gps_to_iso8601(1000000000.9999999)
    '2011-09-14T01:46:26.000000'
    >>> gps_to_iso8601(1000000814.999999)
    '2011-09-14T01:59:59.999999'
    >>> gps_to_iso8601(1000000814.9999999)
    '2011-09-14T02:00:00.000000'
    """
    gps_seconds_fraction, gps_seconds = math.modf(gps_time)
    gps_seconds_fraction = '{0:6f}'.format(gps_seconds_fraction)
    if gps_seconds_fraction[0] == '1':
        gps_seconds += 1
    else:
        assert gps_seconds_fraction[0] == '0'
    assert gps_seconds_fraction[1] == '.'
    gps_seconds_fraction = gps_seconds_fraction[1:]
    year, month, day, hour, minute, second, _, _, _ = lal.GPSToUTC(int(gps_seconds))
    return '{0:04d}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:02d}{6:s}'.format(
        year, month, day, hour, minute, second, gps_seconds_fraction)


def iso8601_to_gps(iso8601):
    """
    Convert an ISO 8601 date string to a floating-point GPS time in seconds.

    Parameters
    ----------

    iso8601 : str
        ISO 8601 date string (with fractional seconds)

    Returns
    -------

    gps : float
        Time in seconds since GPS epoch

    Example
    -------

    >>> gps_to_iso8601(1129501781.2)
    '2015-10-21T22:29:24.200000'
    >>> iso8601_to_gps('2015-10-21T22:29:24.2')
    1129501781.2
    """
    iso8601, _, second_fraction = iso8601.partition('.')
    second_fraction = float('0.' + second_fraction)
    tm = time.strptime(iso8601, "%Y-%m-%dT%H:%M:%S")
    gps_seconds = lal.UTCToGPS(tm)
    return gps_seconds + second_fraction


def gps_to_mjd(gps_time):
    """
    Convert a floating-point GPS time in seconds to a modified Julian day.

    Parameters
    ----------

    gps_time : float
        Time in seconds since GPS epoch

    Returns
    -------

    mjd : float
        Modified Julian day

    Example
    -------

    >>> '%.9f' % round(gps_to_mjd(1129501781.2), 9)
    '57316.937085648'
    """
    gps_seconds_fraction, gps_seconds = math.modf(gps_time)
    mjd = lal.ConvertCivilTimeToMJD(lal.GPSToUTC(int(gps_seconds)))
    return mjd + gps_seconds_fraction / 86400.


def write_sky_map(filename, prob, nest=False, objid=None, url=None, instruments=None,
    gps_time=None, gps_creation_time=None, creator=None, origin=None,
    runtime=None):
    """Write a gravitational-wave sky map to a file, populating the header
    with optional metadata."""

    #
    # Populate optional header fieds.
    #

    extra_metadata = []

    if objid is not None:
        extra_metadata.append(('OBJECT', objid,
            'Unique identifier for this event'))

    if url is not None:
        extra_metadata.append(('REFERENC', url,
            'URL of this event'))

    if instruments is not None:
        if not isinstance(instruments, basestring):
            instruments = str(lsctables.ifos_from_instrument_set(instruments))
        extra_metadata.append(('INSTRUME', instruments,
            'Instruments that triggered this event'))

    if gps_time is not None:
        extra_metadata.append(('DATE-OBS', gps_to_iso8601(gps_time),
            'UTC date of the observation'))
        extra_metadata.append(('MJD-OBS', gps_to_mjd(gps_time),
            'modified Julian date of the observation'))

    if gps_creation_time is None:
        gps_creation_time = lal.GPSTimeNow()
    extra_metadata.append(('DATE', gps_to_iso8601(gps_creation_time),
        'UTC date of file creation'))

    if creator is not None:
        extra_metadata.append(('CREATOR', creator,
            'Program that created this file'))

    if origin is not None:
        extra_metadata.append(('ORIGIN', origin,
            'Organization responsible for this FITS file'))

    if runtime is not None:
        extra_metadata.append(('RUNTIME', runtime,
            'Runtime in seconds of the CREATOR program'))

    write_map(filename, prob, nest=nest, fits_IDL=True, coord='C',
        column_names=('PROB',), unit='pix-1', extra_metadata=extra_metadata)


def read_sky_map(filename, nest=False):
    """
    Read a LIGO/Virgo-type sky map and return a tuple of the HEALPix array
    and a dictionary of metadata from the header.

    Parameters
    ----------

    filename: string
        Path to the optionally gzip-compressed FITS file.

    nest: bool, optional
        If omitted or False, then detect the pixel ordering in the FITS file
        and rearrange if necessary to RING indexing before returning.

        If True, then detect the pixel ordering and rearrange if necessary to
        NESTED indexing before returning.

        If None, then preserve the ordering from the FITS file.

        Regardless of the value of this option, the ordering used in the FITS
        file is indicated as the value of the 'nest' key in the metadata
        dictionary.
    """
    prob, header = hp.read_map(filename, h=True, verbose=False, nest=nest)
    header = dict(header)

    metadata = {}

    ordering = header['ORDERING']
    if ordering == 'RING':
        metadata['nest'] = False
    elif ordering == 'NESTED':
        metadata['nest'] = True
    else:
        raise ValueError(
            'ORDERING card in header has unknown value: {0}'.format(ordering))

    try:
        value = header['OBJECT']
    except KeyError:
        pass
    else:
        metadata['objid'] = value

    try:
        value = header['REFERENC']
    except KeyError:
        pass
    else:
        metadata['url'] = value

    try:
        value = header['INSTRUME']
    except KeyError:
        pass
    else:
        value = set(str(ifo) for ifo in lsctables.instrument_set_from_ifos(value))
        metadata['instruments'] = value

    try:
        value = header['DATE-OBS']
    except KeyError:
        pass
    else:
        metadata['gps_time'] = iso8601_to_gps(value)

    try:
        value = header['DATE']
    except KeyError:
        pass
    else:
        metadata['gps_creation_time'] = iso8601_to_gps(value)

    try:
        value = header['CREATOR']
    except KeyError:
        pass
    else:
        metadata['creator'] = value

    try:
        value = header['ORIGIN']
    except KeyError:
        pass
    else:
        metadata['origin'] = value

    try:
        value = header['RUNTIME']
    except KeyError:
        pass
    else:
        metadata['runtime'] = value

    return prob, metadata


if __name__ == '__main__':
    import healpy as hp
    import numpy as np
    nside = 128
    npix = hp.nside2npix(nside)
    prob = np.random.random(npix)
    prob /= sum(prob)

    write_sky_map('test.fits.gz', prob,
        objid='FOOBAR 12345',
        gps_time=1049492268.25,
        creator=os.path.basename(__file__),
        url='http://www.youtube.com/watch?v=0ccKPSVQcFk',
        origin='LIGO Scientific Collaboration',
        runtime=21.5)

    print read_sky_map('test.fits.gz')

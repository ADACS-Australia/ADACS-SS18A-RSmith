#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2017  Leo Singer
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

import math
import healpy as hp
import numpy as np
from astropy.io import fits
from astropy import units as u
from glue.ligolw import lsctables
import itertools
import time
import lal
import six
from astropy.table import Table
from ..bayestar import moc

# FIXME: Remove this after all Astropy monkeypatches are obsolete.
import astropy
import distutils.version
astropy_version = distutils.version.LooseVersion(astropy.__version__)

__all__ = ("read_sky_map", "write_sky_map")


def gps_to_iso8601(gps_time):
    """Convert a floating-point GPS time in seconds to an ISO 8601 date string.

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
    year, month, day, hour, minute, second, _, _, _ = lal.GPSToUTC(
        int(gps_seconds))
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


def identity(x):
    return x


def instruments_to_fits(value):
    if not isinstance(value, six.string_types):
        value = str(lsctables.instrumentsproperty.set(value))
    return value


def instruments_from_fits(value):
    return {str(ifo) for ifo in lsctables.instrumentsproperty.get(value)}


DEFAULT_NUNIQ_NAMES = ('PROBDENSITY', 'DISTMU', 'DISTSIGMA', 'DISTNORM')
DEFAULT_NUNIQ_UNITS = (u.steradian**-1, u.Mpc, u.Mpc, u.Mpc**-2)
DEFAULT_NESTED_NAMES = ('PROB', 'DISTMU', 'DISTSIGMA', 'DISTNORM')
DEFAULT_NESTED_UNITS = (u.pix**-1, u.Mpc, u.Mpc, u.Mpc**-2)
FITS_META_MAPPING = (
    ('objid', 'OBJECT', 'Unique identifier for this event',
     identity, identity),
    ('url', 'REFERENC', 'URL of this event', identity, identity),
    ('instruments', 'INSTRUME', 'Instruments that triggered this event',
     instruments_to_fits, instruments_from_fits),
    ('gps_time', 'DATE-OBS', 'UTC date of the observation',
     gps_to_iso8601, iso8601_to_gps),
    ('gps_time', 'MJD-OBS', 'modified Julian date of the observation',
     gps_to_mjd, None),
    ('gps_creation_time', 'DATE', 'UTC date of file creation',
     gps_to_iso8601, iso8601_to_gps),
    ('creator', 'CREATOR', 'Program that created this file',
     identity, identity),
    ('origin', 'ORIGIN', 'Organization responsible for this FITS file',
     identity, identity),
    ('runtime', 'RUNTIME', 'Runtime in seconds of the CREATOR program',
     identity, identity),
    ('distmean', 'DISTMEAN', 'Posterior mean distance (Mpc)',
     identity, identity),
    ('diststd', 'DISTSTD', 'Posterior standard deviation of distance (Mpc)',
     identity, identity),
    ('log_bci', 'LOGBCI', 'Log Bayes factor: coherent vs. incoherent',
     identity, identity),
    ('log_bsn', 'LOGBSN', 'Log Bayes factor: signal vs. noise',
     identity, identity),
    ('vcs_info', 'VCSVERS', 'Software version',
     lambda _: _.name + ' ' + _.version, None),
    ('vcs_info', 'VCSSTAT', 'Software version control status',
     lambda _: _.vcsStatus, None),
    ('vcs_info', 'VCSREV', 'Software revision (Git)',
     lambda _: _.vcsId, None),
    ('vcs_info', 'DATE-BLD', 'Software build date',
     lambda _: _.buildDate, None))


def write_sky_map(filename, m, **kwargs):
    """Write a gravitational-wave sky map to a file, populating the header
    with optional metadata.

    Parameters
    ----------

    filename: string
        Path to the optionally gzip-compressed FITS file.

    m : astropy.table.Table or numpy.array
        If a Numpy record array or astorpy.table.Table instance, and has a
        column named 'UNIQ', then interpret the input as NUNIQ-style
        multi-order map [1]_. Otherwise, interpret as a NESTED or RING ordered
        map.

    **kwargs
        Additional metadata to add to FITS header. If m is an
        astropy.table.Table instance, then the header is initialized from both
        m.meta and **kwargs.

    References
    ----------
    .. [1] Górski, K.M., Wandelt, B.D., Hivon, E., Hansen, F.K., & Banday, A.J.
        2017. The HEALPix Primer. The Unique Identifier scheme.
        http://healpix.sourceforge.net/html/intronode4.htm#SECTION00042000000000000000

    Examples
    --------

    Test header contents:

    >>> order = 9
    >>> nside = 2 ** order
    >>> npix = hp.nside2npix(nside)
    >>> prob = np.ones(npix, dtype=np.float) / npix

    >>> import tempfile
    >>> from lalinference import InferenceVCSInfo as vcs_info
    >>> with tempfile.NamedTemporaryFile(suffix='.fits') as f:
    ...     write_sky_map(f.name, prob, nest=True, vcs_info=vcs_info)
    ...     for card in fits.getheader(f.name, 1).cards:
    ...         print(str(card).rstrip())
    XTENSION= 'BINTABLE'           / binary table extension
    BITPIX  =                    8 / array data type
    NAXIS   =                    2 / number of array dimensions
    NAXIS1  =                    8 / length of dimension 1
    NAXIS2  =              3145728 / length of dimension 2
    PCOUNT  =                    0 / number of group parameters
    GCOUNT  =                    1 / number of groups
    TFIELDS =                    1 / number of table fields
    TTYPE1  = 'PROB    '
    TFORM1  = 'D       '
    TUNIT1  = 'pix-1   '
    PIXTYPE = 'HEALPIX '           / HEALPIX pixelisation
    ORDERING= 'NESTED  '           / Pixel ordering scheme: RING, NESTED, or NUNIQ
    COORDSYS= 'C       '           / Ecliptic, Galactic or Celestial (equatorial)
    NSIDE   =                  512 / Resolution parameter of HEALPIX
    INDXSCHM= 'IMPLICIT'           / Indexing: IMPLICIT or EXPLICIT
    VCSVERS = 'LALInference ...' / Software version
    VCSSTAT = '...: ...' / Software version control status
    VCSREV  = '...' / Software revision (Git)
    DATE-BLD= '...' / Software build date

    >>> uniq = moc.nest2uniq(np.uint8(order), np.arange(npix, dtype=np.uint64))
    >>> probdensity = prob / hp.nside2pixarea(nside)
    >>> moc_data = np.rec.fromarrays(
    ...     [uniq, probdensity], names=['UNIQ', 'PROBDENSITY'])
    >>> with tempfile.NamedTemporaryFile(suffix='.fits') as f:
    ...     write_sky_map(f.name, moc_data, vcs_info=vcs_info)
    ...     for card in fits.getheader(f.name, 1).cards:
    ...         print(str(card).rstrip())
    XTENSION= 'BINTABLE'           / binary table extension
    BITPIX  =                    8 / array data type
    NAXIS   =                    2 / number of array dimensions
    NAXIS1  =                   16 / length of dimension 1
    NAXIS2  =              3145728 / length of dimension 2
    PCOUNT  =                    0 / number of group parameters
    GCOUNT  =                    1 / number of groups
    TFIELDS =                    2 / number of table fields
    TTYPE1  = 'UNIQ    '
    TFORM1  = 'K       '
    TZERO1  =  9223372036854775808
    TTYPE2  = 'PROBDENSITY'
    TFORM2  = 'D       '
    TUNIT2  = 'sr-1    '
    PIXTYPE = 'HEALPIX '           / HEALPIX pixelisation
    ORDERING= 'NUNIQ   '           / Pixel ordering scheme: RING, NESTED, or NUNIQ
    COORDSYS= 'C       '           / Ecliptic, Galactic or Celestial (equatorial)
    MOCORDER=                    9 / MOC resolution (best order)
    VCSVERS = 'LALInference ...' / Software version
    VCSSTAT = '...: ...' / Software version control status
    VCSREV  = '...' / Software revision (Git)
    DATE-BLD= '...' / Software build date
    """

    if isinstance(m, Table) or (isinstance(m, np.ndarray) and m.dtype.names):
        m = Table(m)
    else:
        if np.ndim(m) == 1:
            m = [m]
        m = Table(m, names=DEFAULT_NESTED_NAMES[:len(m)])
    m.meta.update(kwargs)

    if 'UNIQ' in m.colnames:
        default_names = DEFAULT_NUNIQ_NAMES
        default_units = DEFAULT_NUNIQ_UNITS
        extra_header = [
            ('PIXTYPE', 'HEALPIX',
             'HEALPIX pixelisation'),
            ('ORDERING', 'NUNIQ',
             'Pixel ordering scheme: RING, NESTED, or NUNIQ'),
            ('COORDSYS', 'C',
             'Ecliptic, Galactic or Celestial (equatorial)'),
            ('MOCORDER', moc.uniq2order(m['UNIQ'].max()),
             'MOC resolution (best order)')]
    else:
        default_names = DEFAULT_NESTED_NAMES
        default_units = DEFAULT_NESTED_UNITS
        ordering = 'NESTED' if m.meta.pop('nest', False) else 'RING'
        extra_header = [
            ('PIXTYPE', 'HEALPIX',
             'HEALPIX pixelisation'),
            ('ORDERING', ordering,
             'Pixel ordering scheme: RING, NESTED, or NUNIQ'),
            ('COORDSYS', 'C',
             'Ecliptic, Galactic or Celestial (equatorial)'),
            ('NSIDE', hp.npix2nside(len(m)),
             'Resolution parameter of HEALPIX'),
            ('INDXSCHM', 'IMPLICIT',
             'Indexing: IMPLICIT or EXPLICIT')]

    for key, rows in itertools.groupby(FITS_META_MAPPING, lambda row: row[0]):
        try:
            value = m.meta.pop(key)
        except KeyError:
            pass
        else:
            for row in rows:
                _, fits_key, fits_comment, to_fits, _ = row
                if to_fits is not None:
                    extra_header.append(
                        (fits_key, to_fits(value), fits_comment))

    for default_name, default_unit in zip(default_names, default_units):
        try:
            col = m[default_name]
        except KeyError:
            pass
        else:
            if not col.unit:
                col.unit = default_unit

    if astropy_version >= '1.3.1':
        hdu = fits.table_to_hdu(m)
        hdu.header.extend(extra_header)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdulist.writeto(filename, clobber=True)
    else:
        # FIXME: This code path works around a number of issues with older
        # versions of Astropy. Remove it once we drop support for
        # astropy < 1.3.1.
        #
        # astropy.io.fits.table_to_hdu was added in astropy 1.2.
        # We must currently support astropy >= 1.1.1 on the LIGO Data Grid's
        # Scientific Linux 7 computing clusters.
        #
        # With some old versions of astropy that we still have to
        # support, the astropy.table.Table.write method did not support the
        # clobber argument. So we have to manually delete the file first so
        # that astropy.io.fits does not complain that the file exists.
        #
        # Also this works around https://github.com/astropy/astropy/pull/5720,
        # which was fixed in astropy 1.3.1.
        from ..bayestar.command import rm_f
        rm_f(filename)
        m.write(filename, format='fits')

        hdulist = fits.open(filename)
        _, hdu = hdulist
        hdu.header.extend(extra_header)
        hdulist.writeto(filename, clobber=True)


def read_sky_map(filename, nest=False, distances=False, moc=False, **kwargs):
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

    distances: bool, optional
        If true, then read also read the additional HEALPix layers representing
        the conditional mean and standard deviation of distance as a function
        of sky location.

    moc: bool, optional
        If true, then preserve multi-order structure if present.

    Example
    -------

    Test that we can read a legacy IDL-compatible file
    (https://bugs.ligo.org/redmine/issues/5168):

    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile(suffix='.fits') as f:
    ...     nside = 512
    ...     npix = hp.nside2npix(nside)
    ...     ipix_nest = np.arange(npix)
    ...     hp.write_map(f.name, ipix_nest, nest=True, column_names=['PROB'])
    ...     m, meta = read_sky_map(f.name)
    ...     np.testing.assert_array_equal(m, hp.ring2nest(nside, ipix_nest))
    """
    m = Table.read(filename, format='fits', **kwargs)

    # Remove some keys that we do not need
    for key in (
            'PIXTYPE', 'EXTNAME', 'NSIDE', 'FIRSTPIX', 'LASTPIX', 'INDXSCHM'):
        m.meta.pop(key, None)

    if m.meta.pop('COORDSYS', 'C') != 'C':
        raise ValueError('LALInference only reads and writes sky maps in '
                         'equatorial coordinates.')

    try:
        value = m.meta.pop('ORDERING')
    except KeyError:
        pass
    else:
        if value == 'RING':
            m.meta['nest'] = False
        elif value == 'NESTED':
            m.meta['nest'] = True
        elif value == 'NUNIQ':
            pass
        else:
            raise ValueError(
                'ORDERING card in header has unknown value: {0}'.format(value))

    for fits_key, rows in itertools.groupby(
            FITS_META_MAPPING, lambda row: row[1]):
        try:
            value = m.meta.pop(fits_key)
        except KeyError:
            pass
        else:
            for row in rows:
                key, _, _, _, from_fits = row
                if from_fits is not None:
                    m.meta[key] = from_fits(value)

    if 'UNIQ' not in m.colnames:
        m = Table([col.ravel() for col in m.columns.values()], meta=m.meta)

    if 'UNIQ' in m.colnames and not moc:
        from ..bayestar.sky_map import rasterize
        m = rasterize(m)
        m.meta['nest'] = True
    elif 'UNIQ' not in m.colnames and moc:
        from ..bayestar.sky_map import derasterize
        if not m.meta['nest']:
            npix = len(m)
            nside = hp.npix2nside(npix)
            m = m[hp.nest2ring(nside, np.arange(npix))]
        m = derasterize(m)
        m.meta.pop('nest', None)

    if 'UNIQ' not in m.colnames:
        npix = len(m)
        nside = hp.npix2nside(npix)

        if nest is None:
            pass
        elif m.meta['nest'] and not nest:
            m = m[hp.ring2nest(nside, np.arange(npix))]
        elif not m.meta['nest'] and nest:
            m = m[hp.nest2ring(nside, np.arange(npix))]

    if moc:
        return m
    elif distances:
        return tuple(
            np.asarray(m[name]) for name in DEFAULT_NESTED_NAMES), m.meta
    else:
        return np.asarray(m[DEFAULT_NESTED_NAMES[0]]), m.meta


if __name__ == '__main__':
    import os
    nside = 128
    npix = hp.nside2npix(nside)
    prob = np.random.random(npix)
    prob /= sum(prob)

    write_sky_map(
        'test.fits.gz', prob,
        objid='FOOBAR 12345',
        gps_time=1049492268.25,
        creator=os.path.basename(__file__),
        url='http://www.youtube.com/watch?v=0ccKPSVQcFk',
        origin='LIGO Scientific Collaboration',
        runtime=21.5)

    print(read_sky_map('test.fits.gz'))

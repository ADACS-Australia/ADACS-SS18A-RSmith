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
r"""
Listen for new events from LVAlert and perform sky localization.

`bayestar_localize_lvalert` supports two modes of operation. You can
explicitly specify the GraceDb ID on the command line, as in:

    $ bayestar_localize_lvalert T90713

Or, `bayetar_localize_lvalert` can GraceDB IDs from stdin (e.g., from the
terminal, or redirected from a fifo):

    $ mkfifo /var/run/bayestar
    $ tail -F /var/run/bayestar | bayestar_localize_lvalert &
    $ echo T90713 > /var/run/bayestar
"""


#
# Command line interface
#

from lalinference.bayestar import command
import logging

logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('BAYESTAR')

methods = '''
    toa_phoa_snr
    toa_phoa_snr_mcmc
    '''.split()
default_method = 'toa_phoa_snr'
command.skymap_parser.add_argument(
    '--method', choices=methods, default=default_method,
    help='Sky localization method [default: %(default)s]')
parser = command.ArgumentParser(
    parents=[
        command.waveform_parser, command.prior_parser, command.skymap_parser])
parser.add_argument(
    '-N', '--dry-run', default=False, action='store_true',
    help='Dry run; do not update GraceDB entry [default: %(default)s]')
parser.add_argument(
    '--no-tag', default=False, action='store_true',
    help='Do not set lvem tag for GraceDB entry [default: %(default)s]')
parser.add_argument(
    '-o', '--output', metavar='FILE.fits[.gz]', default='bayestar.fits.gz',
    help='Name for uploaded file [default: %(default)s]')
parser.add_argument(
    'graceid', metavar='G123456', nargs='*',
    help='Run on these GraceDB IDs. If no GraceDB IDs are listed on the '
    'command line, then read newline-separated GraceDB IDs from stdin.')
opts = parser.parse_args()


#
# Late imports
#

import os
import sys
from lalinference.bayestar.sky_map import localize, rasterize
from lalinference.io import fits
from lalinference.io import events
import ligo.gracedb.logging
import ligo.gracedb.rest


# If no GraceDB IDs were specified on the command line, then read them
# from stdin line-by-line.
graceids = opts.graceid if opts.graceid else command.iterlines(sys.stdin)

# Fire up a GraceDb client
# FIXME: Mimic the behavior of the GraceDb command line client, where the
# environment variable GRACEDB_SERVICE_URL overrides the default service URL.
# It would be nice to get this behavior into the gracedb package itself.
gracedb = ligo.gracedb.rest.GraceDb(
    os.environ.get(
        'GRACEDB_SERVICE_URL', ligo.gracedb.rest.DEFAULT_SERVICE_URL))

if opts.chain_dump:
    chain_dump = opts.output.replace('.fits.gz', '').replace('.fits', '') + \
        '.chain.npy'
else:
    chain_dump = None

tags = ("sky_loc",)
if not opts.no_tag:
    tags += ("lvem",)

for graceid, event in events.gracedb.open(gracedb, graceids).items():

    # Send log messages to GraceDb too
    if not opts.dry_run:
        handler = ligo.gracedb.logging.GraceDbLogHandler(gracedb, graceid)
        handler.setLevel(logging.INFO)
        logging.root.addHandler(handler)

    # A little bit of Cylon humor
    log.info('by your command...')

    try:
        # perform sky localization
        log.info("starting sky localization")
        sky_map = rasterize(localize(
            event, opts.waveform, opts.f_low, opts.min_distance,
            opts.max_distance, opts.prior_distance_power, opts.cosmology,
            method=opts.method, nside=opts.nside,
            chain_dump=chain_dump, enable_snr_series=opts.enable_snr_series,
            f_high_truncate=opts.f_high_truncate))
        sky_map.meta['objid'] = str(graceid)
        sky_map.meta['url'] = 'https://gracedb.ligo.org/events/{0}'.format(
            graceid)
        log.info("sky localization complete")

        # upload FITS file
        with command.TemporaryDirectory() as fitsdir:
            fitspath = os.path.join(fitsdir, opts.output)
            fits.write_sky_map(fitspath, sky_map, nest=True)
            log.debug('wrote FITS file: %s', opts.output)
            if opts.dry_run:
                command.rename(fitspath, os.path.join('.', opts.output))
            else:
                gracedb.writeLog(
                    graceid, "BAYESTAR rapid sky localization ready",
                    filename=fitspath, tagname=tags)
            log.debug('uploaded FITS file')
    except:
        # Produce log message for any otherwise uncaught exception
        log.exception("sky localization failed")
        if opts.dry_run:
            # Then re-raise the exception if we are in dry-run mode
            raise

    if not opts.dry_run:
        # Remove old log handler
        logging.root.removeHandler(handler)
        del handler

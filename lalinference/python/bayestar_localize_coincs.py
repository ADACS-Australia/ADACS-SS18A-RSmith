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
Produce GW sky maps for all coincidences in a LIGO-LW XML file.

The filename of the (optionally gzip-compressed) LIGO-LW XML input is an
optional argument; if omitted, input is read from stdin.

The distance prior is controlled by the --prior-distance-power argument.
If you set --prior-distance-power=k, then the distance prior is
proportional to r^k. The default is 2, uniform in volume.

If the --min-distance argument is omitted, it defaults to zero. If the
--max-distance argument is omitted, it defaults to the SNR=4 horizon
distance of the most sensitive detector.

A FITS file is created for each sky map, having a filename of the form

  "X.toa_phoa_snr.fits"
  "X.toa_snr_mcmc.fits"
  "X.toa_phoa_snr_mcmc.fits"

where X is the LIGO-LW row id of the coinc and "toa" or "toa_phoa_snr"
identifies whether the sky map accounts for times of arrival (TOA),
PHases on arrival (PHOA), and amplitudes on arrival (SNR).
"""


# Command line interface.
import argparse
from lalinference.bayestar import command

methods = '''
    toa_phoa_snr
    toa_phoa_snr_mcmc
    '''.split()
default_method = 'toa_phoa_snr'
command.skymap_parser.add_argument(
    '--method', choices=methods, default=[default_method], nargs='*',
    help='Sky localization methods [default: %(default)s]')
parser = command.ArgumentParser(
    parents=[
        command.waveform_parser, command.prior_parser, command.skymap_parser])
parser.add_argument(
    '--keep-going', '-k', default=False, action='store_true',
    help='Keep processing events if a sky map fails to converge [default: no]')
parser.add_argument(
    'input', metavar='INPUT.xml[.gz]', default='-', nargs='+',
    type=argparse.FileType('rb'),
    help='Input LIGO-LW XML file [default: stdin] or PyCBC HDF5 files. '
         'For PyCBC, you must supply the coincidence file '
         '(e.g. "H1L1-HDFINJFIND.hdf" or "H1L1-STATMAP.hdf"), '
         'the template bank file (e.g. H1L1-BANK2HDF.hdf), '
         'the single-detector merged PSD files '
         '(e.g. "H1-MERGE_PSDS.hdf" and "L1-MERGE_PSDS.hdf"), '
         'and the single-detector merged trigger files '
         '(e.g. "H1-HDF_TRIGGER_MERGE.hdf" and "L1-HDF_TRIGGER_MERGE.hdf"), '
         'in any order.')
parser.add_argument(
    '--pycbc-sample', default='foreground',
    help='sample population [PyCBC only; default: %(default)s]')
parser.add_argument(
    '--coinc-event-id', type=int, nargs='*',
    help='run on only these specified events')
parser.add_argument(
    '--output', '-o', default='.',
    help='output directory [default: current directory]')
parser.add_argument(
    '--condor-submit', action='store_true',
    help='submit to Condor instead of running locally')
opts = parser.parse_args()

#
# Logging
#

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('BAYESTAR')

# BAYESTAR imports.
from lalinference.io import fits, events
from lalinference.bayestar.sky_map import localize

# Other imports.
import os
from collections import OrderedDict
import sys

# Read coinc file.
log.info('%s:reading input files', ','.join(file.name for file in opts.input))
event_source = events.open(*opts.input, sample=opts.pycbc_sample)

command.mkpath(opts.output)

if opts.condor_submit:
    if opts.coinc_event_id:
        raise ValueError('must not set --coinc-event-id with --condor-submit')
    cmd = ['condor_submit', 'accounting_group=ligo.dev.o3.cbc.pe.bayestar',
           'on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)',
           'on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)',
           'on_exit_hold_reason = (ExitBySignal == True ? strcat("The job exited with signal ", ExitSignal) : strcat("The job exited with signal ", ExitCode))',
           'request_memory = 1000 MB',
           'universe=vanilla', 'getenv=true', 'executable=' + sys.executable,
           'JobBatchName=BAYESTAR', 'environment="OMP_NUM_THREADS=1"',
           'error=' + os.path.join(opts.output, '$(CoincEventId).err'),
           'log=' + os.path.join(opts.output, '$(CoincEventId).log'),
           'arguments="-B ' + ' '.join(arg for arg in sys.argv
               if arg != '--condor-submit') + ' --coinc-event-id $(CoincEventId)"',
           '-append', 'queue CoincEventId in ' + ' '.join(
               str(coinc_event_id) for coinc_event_id in event_source),
           '/dev/null']
    os.execvp('condor_submit', cmd)

# Loop over all coinc_event <-> sim_inspiral coincs.
if opts.coinc_event_id:
    event_source = OrderedDict(
        (key, event_source[key]) for key in opts.coinc_event_id)

count_sky_maps_failed = 0

for int_coinc_event_id, event in event_source.items():
    coinc_event_id = 'coinc_event:coinc_event_id:{}'.format(int_coinc_event_id)

    # Loop over sky localization methods
    for method in opts.method:
        log.info("%s:method '%s':computing sky map", coinc_event_id, method)
        if opts.chain_dump:
            chain_dump = '%s.chain.npy' % int_coinc_event_id
        else:
            chain_dump = None
        try:
            sky_map = localize(
                event, opts.waveform, opts.f_low, opts.min_distance,
                opts.max_distance, opts.prior_distance_power, opts.cosmology,
                method=method, nside=opts.nside, chain_dump=chain_dump,
                enable_snr_series=opts.enable_snr_series,
                f_high_truncate=opts.f_high_truncate)
            sky_map.meta['objid'] = coinc_event_id
        except (ArithmeticError, ValueError):
            log.exception(
                "%s:method '%s':sky localization failed",
                coinc_event_id, method)
            count_sky_maps_failed += 1
            if not opts.keep_going:
                raise
        else:
            log.info(
                "%s:method '%s':saving sky map",
                coinc_event_id, method)
            filename = '%d.%s.fits' % (int_coinc_event_id, method)
            fits.write_sky_map(
                os.path.join(opts.output, filename), sky_map, nest=True)

if count_sky_maps_failed > 0:
    raise RuntimeError("{0} sky map{1} did not converge".format(
        count_sky_maps_failed, 's' if count_sky_maps_failed > 1 else ''))

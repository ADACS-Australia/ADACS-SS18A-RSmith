#
# Copyright (C) 2015  Leo Singer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Create a contours for the credible levels of an all-sky probability map.
The input is a HEALPix probability map.
The output is a GeoJSON FeatureCollection (http://geojson.org/).
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


# Command line interface

import argparse
from lalinference.bayestar import command
parser = command.ArgumentParser()
parser.add_argument(
    '-o', '--output', metavar='FILE.geojson',
    default='-', type=argparse.FileType('w'),
    help='name of output file [default: stdout]')
parser.add_argument(
    '--contour', metavar='PERCENT', type=float, nargs='+', required=True,
    help='plot contour enclosing this percentage of probability mass')
parser.add_argument(
    '-i', '--interpolate',
    choices='nearest nested bilinear'.split(), default='nearest',
    help='resampling interpolation method [default: %(default)s]')
parser.add_argument(
    '-s', '--simplify', default=False, action='store_true',
    help='simplify contour paths [default: %(default)s]')
parser.add_argument('-n', '--nside', metavar='NSIDE', type=int,
    help='optionally resample to the specified resolution '
    ' before generating contours [default: no downsampling]')
parser.add_argument(
    'input', metavar='INPUT.fits[.gz]', type=argparse.FileType('rb'),
    default='-', nargs='?', help='Input FITS file [default: stdin]')
opts = parser.parse_args()

# Late imports

from lalinference import fits
from lalinference.bayestar import postprocess
import healpy as hp
import numpy as np
import json


# Read input file
prob, _ = fits.read_sky_map(opts.input.name, nest=True)

# Resample if requested
if opts.nside is not None and opts.interpolate in ('nearest', 'nested'):
    prob = hp.ud_grade(prob, opts.nside, order_in='NESTED', power=-2)
elif opts.nside is not None and opts.interpolate == 'bilinear':
    prob = postprocess.smooth_ud_grade(prob, opts.nside, nest=True)
if opts.interpolate == 'nested':
    prob = postprocess.interpolate_nested(prob, nest=True)

# Find credible levels
i = np.flipud(np.argsort(prob))
cumsum = np.cumsum(prob[i])
cls = np.empty_like(prob)
cls[i] = cumsum * 100

# Generate contours
paths = list(postprocess.contour(
    cls, opts.contour, nest=True, degrees=True, simplify=opts.simplify))

json.dump({
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {
                'credible_level': contour
            },
            'geometry': {
                'type': 'MultiLineString',
                'coordinates': path
            }
        }
        for contour, path in zip(opts.contour, paths)
    ]
}, opts.output)

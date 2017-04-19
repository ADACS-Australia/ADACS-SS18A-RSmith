#
# Copyright (C) 2011-2017  Leo Singer
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
Plot an all-sky map on a Mollweide projection.
By default, plot in celestial coordinates (RA, Dec).

To plot in geographic coordinates (longitude, latitude) with major
coastlines overlaid, provide the --geo flag.

Public-domain cartographic data is courtesy of Natural Earth
(http://www.naturalearthdata.com) and processed with MapShaper
(http://www.mapshaper.org).
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


# Command line interface

import argparse
from lalinference.bayestar import command
parser = command.ArgumentParser(parents=[command.figure_parser])
parser.add_argument(
    '--annotate', default=False, action='store_true',
    help='annotate plot with information about the event')
parser.add_argument(
    '--contour', metavar='PERCENT', type=float, nargs='+',
    help='plot contour enclosing this percentage of'
    ' probability mass [may be specified multiple times, default: none]')
parser.add_argument(
    '--colorbar', default=False, action='store_true',
    help='Show colorbar [default: %(default)s]')
parser.add_argument(
    '--radec', nargs=2, metavar='deg', type=float, action='append',
    default=[], help='right ascension (deg) and declination (deg) to mark'
    ' [may be specified multiple times, default: none]')
parser.add_argument(
    '--inj-database', metavar='FILE.sqlite', type=command.SQLiteType('r'),
    help='read injection positions from database [default: none]')
parser.add_argument(
    '--geo', action='store_true', default=False,
    help='Plot in geographic coordinates, (lat, lon) instead of (RA, Dec)'
    ' [default: %(default)s]')
parser.add_argument(
    'input', metavar='INPUT.fits[.gz]', type=argparse.FileType('rb'),
    default='-', nargs='?', help='Input FITS file [default: stdin]')
opts = parser.parse_args()

# Late imports

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import healpy as hp
import lal
from lalinference.io import fits
from lalinference import plot
from lalinference.bayestar import postprocess

fig = plt.figure(frameon=False)
ax = plt.axes(projection='mollweide' if opts.geo else 'astro hours mollweide')
ax.grid()

skymap, metadata = fits.read_sky_map(opts.input.name, nest=None)
nside = hp.npix2nside(len(skymap))

if opts.geo:
    dlon = -lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(metadata['gps_time'])) % (2*np.pi)
else:
    dlon = 0

# Convert sky map from probability to probability per square degree.
deg2perpix = hp.nside2pixarea(nside, degrees=True)
probperdeg2 = skymap / deg2perpix

# Plot sky map.
vmax = probperdeg2.max()
plot.healpix_heatmap(
    probperdeg2, dlon=dlon, nest=metadata['nest'], vmin=0., vmax=vmax)

# Add colorbar.
if opts.colorbar:
    cb = plot.colorbar()
    cb.set_label(r'prob. per deg$^2$')

# Add contours.
if opts.contour:
    cls = 100 * postprocess.find_greedy_credible_levels(skymap)
    cs = plot.healpix_contour(
        cls, dlon=dlon, nest=metadata['nest'],
        colors='k', linewidths=0.5, levels=opts.contour)
    fmt = r'%g\%%' if rcParams['text.usetex'] else '%g%%'
    plt.clabel(cs, fmt=fmt, fontsize=6, inline=True)

# Add continents.
if opts.geo:
    geojson_filename = os.path.join(os.path.dirname(plot.__file__),
        'ne_simplified_coastline.json')
    with open(geojson_filename, 'r') as geojson_file:
        geojson = json.load(geojson_file)
    for shape in geojson['geometries']:
        verts = np.deg2rad(shape['coordinates'])
        plt.plot(verts[:, 0], verts[:, 1], color='0.5', linewidth=0.5)

radecs = np.deg2rad(opts.radec).tolist()
if opts.inj_database:
    query = '''SELECT DISTINCT longitude, latitude FROM sim_inspiral AS si
               INNER JOIN coinc_event_map AS cm1
               ON (si.simulation_id = cm1.event_id)
               INNER JOIN coinc_event_map AS cm2
               ON (cm1.coinc_event_id = cm2.coinc_event_id)
               WHERE cm2.event_id = ?'''
    (ra, dec), = opts.inj_database.execute(
        query, (metadata['objid'],)).fetchall()
    radecs.append([ra, dec])

# Add markers (e.g., for injections or external triggers).
for ra, dec in radecs:
    # Convert the right ascension to either a reference angle from -pi to pi
    # or a wrapped angle from 0 to 2 pi.
    if opts.geo:
        ra = plot.reference_angle(ra + dlon)
    else:
        ra = plot.wrapped_angle(ra + dlon)
    ax.plot(ra, dec, '*', markerfacecolor='white', markeredgecolor='black', markersize=10)

# Add a white outline to all text to make it stand out from the background.
plot.outline_text(ax)

if opts.annotate:
    text = 'event ID: {}'.format(metadata['objid'])
    text += '\nFITS file: {}'.format(opts.input.name)
    if opts.contour:
        pp = np.round(opts.contour).astype(int)
        ii = np.round(np.searchsorted(np.sort(cls), opts.contour) *
                      deg2perpix).astype(int)
        for i, p in zip(ii, pp):
            text += '\n{:d}% area: {:d} deg$^2$'.format(p, i, grouping=True)
    ax.text(1, 1, text, transform=ax.transAxes, horizontalalignment='right')

# Show or save output.
opts.output()

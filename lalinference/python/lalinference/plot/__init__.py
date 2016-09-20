#
# Copyright (C) 2012-2015  Leo Singer
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
Plotting tools for drawing skymaps
"""
from __future__ import division
__author__ = "Leo Singer <leo.singer@ligo.org>"
__all__ = ("AstroDegreesMollweideAxes", "AstroHoursMollweideAxes",
           "AstroMollweideAxes", "reference_angle", "make_rect_poly", "heatmap")


import warnings
import functools

# FIXME: Remove this after all Matplotlib monkeypatches are obsolete.
import matplotlib
import distutils.version
mpl_version = distutils.version.LooseVersion(matplotlib.__version__)

from matplotlib.axes import Axes
from matplotlib import text
from matplotlib import ticker
from matplotlib import patheffects
from matplotlib.ticker import Formatter, FixedLocator
from matplotlib.projections import projection_registry
from matplotlib.transforms import Transform, Affine2D
from matplotlib.projections.geo import LambertAxes, MollweideAxes
import _geoslib as geos
from matplotlib import pyplot as plt
import scipy.stats
import numpy as np
import healpy as hp


# FIXME: Remove this after all Matplotlib monkeypatches are obsolete.
if mpl_version >= '1.3.0':
    FixedMollweideAxes = MollweideAxes
elif mpl_version < '1.2.0':
    class FixedMollweideAxes(MollweideAxes):
        """Patched version of matplotlib's Mollweide projection that implements a
        correct inverse transform."""

        name = 'fixed mollweide'

        class FixedMollweideTransform(MollweideAxes.MollweideTransform):

            def inverted(self):
                return FixedMollweideAxes.InvertedFixedMollweideTransform(self._resolution)
            inverted.__doc__ = Transform.inverted.__doc__

        class InvertedFixedMollweideTransform(MollweideAxes.InvertedMollweideTransform):

            def inverted(self):
                return FixedMollweideAxes.FixedMollweideTransform(self._resolution)
            inverted.__doc__ = Transform.inverted.__doc__

            def transform(self, xy):
                x = xy[:, 0:1]
                y = xy[:, 1:2]

                sqrt2 = np.sqrt(2)
                sintheta = y / sqrt2
                with np.errstate(invalid='ignore'):
                    costheta = np.sqrt(1. - 0.5 * y * y)
                longitude = 0.25 * sqrt2 * np.pi * x / costheta
                latitude = np.arcsin(2 / np.pi * (np.arcsin(sintheta) + sintheta * costheta))
                return np.concatenate((longitude, latitude), 1)
            transform.__doc__ = Transform.transform.__doc__

            transform_non_affine = transform

        def _get_core_transform(self, resolution):
            return self.FixedMollweideTransform(resolution)
else:
    class FixedMollweideAxes(MollweideAxes):
        """Patched version of matplotlib's Mollweide projection that implements a
        correct inverse transform."""

        name = 'fixed mollweide'

        class FixedMollweideTransform(MollweideAxes.MollweideTransform):

            def inverted(self):
                return FixedMollweideAxes.InvertedFixedMollweideTransform(self._resolution)
            inverted.__doc__ = Transform.inverted.__doc__

        class InvertedFixedMollweideTransform(MollweideAxes.InvertedMollweideTransform):

            def inverted(self):
                return FixedMollweideAxes.FixedMollweideTransform(self._resolution)
            inverted.__doc__ = Transform.inverted.__doc__

            def transform_non_affine(self, xy):
                x = xy[:, 0:1]
                y = xy[:, 1:2]

                sqrt2 = np.sqrt(2)
                sintheta = y / sqrt2
                with np.errstate(invalid='ignore'):
                    costheta = np.sqrt(1. - 0.5 * y * y)
                longitude = 0.25 * sqrt2 * np.pi * x / costheta
                latitude = np.arcsin(2 / np.pi * (np.arcsin(sintheta) + sintheta * costheta))
                return np.concatenate((longitude, latitude), 1)
            transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        def _get_core_transform(self, resolution):
            return self.FixedMollweideTransform(resolution)


class AstroDegreesMollweideAxes(FixedMollweideAxes):
    """Mollweide axes with phi axis flipped and in degrees from 360 to 0
    instead of in degrees from -180 to 180."""

    name = 'astro degrees mollweide'

    def cla(self):
        super(AstroDegreesMollweideAxes, self).cla()
        self.set_xlim(0, 2*np.pi)

    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, 0., 2*np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    def _get_core_transform(self, resolution):
        return Affine2D().translate(-np.pi, 0.) + super(AstroDegreesMollweideAxes, self)._get_core_transform(resolution)

    def set_longitude_grid(self, degrees):
        # Copied from matplotlib.geo.GeoAxes.set_longitude_grid and modified
        super(AstroDegreesMollweideAxes, self).set_longitude_grid(degrees)
        number = (360.0 / degrees) + 1
        self.xaxis.set_major_locator(
            FixedLocator(
                np.linspace(0, 2*np.pi, number, True)[1:-1]))

    def _set_lim_and_transforms(self):
        # Copied from matplotlib.geo.GeoAxes._set_lim_and_transforms and modified
        super(AstroDegreesMollweideAxes, self)._set_lim_and_transforms()

        # This is the transform for latitude ticks.
        yaxis_stretch = Affine2D().scale(np.pi * 2.0, 1.0)
        yaxis_space = Affine2D().scale(-1.0, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space + \
             self.transAffine + \
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8.0, 0.0)

    def _get_affine_transform(self):
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform_point((0, 0))
        _, yscale = transform.transform_point((0, np.pi / 2.0))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)


projection_registry.register(AstroDegreesMollweideAxes)


class AstroHoursMollweideAxes(AstroDegreesMollweideAxes):
    """Mollweide axes with phi axis flipped and in hours from 24 to 0 instead of
    in degrees from -180 to 180."""

    name = 'astro hours mollweide'

    class RaFormatter(Formatter):
        # Copied from matplotlib.geo.GeoAxes.ThetaFormatter and modified
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            hours = (x / np.pi) * 12.
            hours = round(15 * hours / self._round_to) * self._round_to / 15
            return r"%0.0f$^\mathrm{h}$" % hours

    def set_longitude_grid(self, degrees):
        super(AstroHoursMollweideAxes, self).set_longitude_grid(degrees)
        self.xaxis.set_major_formatter(self.RaFormatter(degrees))


projection_registry.register(AstroHoursMollweideAxes)


# For backward compatibility
class AstroMollweideAxes(AstroHoursMollweideAxes):

    name = 'astro mollweide'

    def __init__(self, *args, **kwargs):
        warnings.warn("The AstroMollweideAxes ('astro mollweide') class has "
                      "been deprecated. Please use AstroHoursMollweideAxes "
                      "('astro hours mollweide') instead.", stacklevel=2)
        super(AstroMollweideAxes, self).__init__(*args, **kwargs)


projection_registry.register(AstroMollweideAxes)


class AstroLambertAxes(LambertAxes):
    name = 'astro lambert'

    def cla(self):
        super(AstroLambertAxes, self).cla()
        self.set_xlim(0, 2*np.pi)

    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, 0., 2*np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    def _get_core_transform(self, resolution):
        return Affine2D().translate(-np.pi, 0.).scale(-1, 1) + super(AstroLambertAxes, self)._get_core_transform(resolution)

    class RaFormatter(Formatter):
        # Copied from matplotlib.geo.GeoAxes.ThetaFormatter and modified
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            hours = (x / np.pi) * 12.
            hours = round(15 * hours / self._round_to) * self._round_to / 15
            return r"%0.0f$^\mathrm{h}$" % hours

    def set_longitude_grid(self, degrees):
        # Copied from matplotlib.geo.GeoAxes.set_longitude_grid and modified
        number = (360.0 / degrees) + 1
        self.xaxis.set_major_locator(
            FixedLocator(
                np.linspace(0, 2*np.pi, number, True)[1:-1]))
        self.xaxis.set_major_formatter(self.RaFormatter(degrees))


projection_registry.register(AstroLambertAxes)


def wrapped_angle(a):
    """Convert an angle to a reference angle between 0 and 2*pi."""
    return np.mod(a, 2 * np.pi)


def wrapped_angle_deg(a):
    """Convert an angle to a reference angle between 0 and 2*pi."""
    return np.mod(a, 360)


def reference_angle(a):
    """Convert an angle to a reference angle between -pi and pi."""
    a = np.mod(a, 2 * np.pi)
    return np.where(a <= np.pi, a, a - 2 * np.pi)


def reference_angle_deg(a):
    """Convert an angle to a reference angle between -180 and 180 degrees."""
    a = np.mod(a, 360)
    return np.where(a <= 180, a, a - 360)


def subdivide_vertices(vertices, subdivisions):
    """Subdivide a list of vertices by inserting subdivisions additional vertices
    between each original pair of vertices using linear interpolation."""
    subvertices = np.empty((subdivisions * len(vertices), vertices.shape[1]))
    frac = np.atleast_2d(np.arange(subdivisions + 1, dtype=float) / subdivisions).T.repeat(vertices.shape[1], 1)
    for i in range(len(vertices)):
        subvertices[i*subdivisions:(i+1)*subdivisions] = frac[:0:-1, :] * np.expand_dims(vertices[i-1, :], 0).repeat(subdivisions, 0)  + frac[:-1, :] * np.expand_dims(vertices[i, :], 0).repeat(subdivisions, 0)
    return subvertices


# FIXME: Remove this after all Matplotlib monkeypatches are obsolete.
def cut_dateline(vertices):
    """Cut a polygon across the dateline, possibly splitting it into multiple
    polygons.  Vertices consist of (longitude, latitude) pairs where longitude
    is always given in terms of a reference angle (between -pi and pi).

    This routine is not meant to cover all possible cases; it will only work for
    convex polygons that extend over less than a hemisphere."""

    out_vertices = []

    # Ensure that the list of vertices does not contain a repeated endpoint.
    if (vertices[0, :] == vertices[-1, :]).all():
        vertices = vertices[:-1, :]

    def count_dateline_crossings(phis):
        n = 0
        for i in range(len(phis)):
            if crosses_dateline(phis[i - 1], phis[i]):
                n += 1
        return n

    def crosses_dateline(phi0, phi1):
        """Test if the segment consisting of v0 and v1 croses the meridian."""
        phi0, phi1 = sorted((phi0, phi1))
        return phi1 - phi0 > np.pi

    dateline_crossings = count_dateline_crossings(vertices[:, 0])
    if dateline_crossings % 2:
        # FIXME: Use this simple heuristic to decide which pole to enclose.
        sign_lat = np.sign(np.sum(vertices[:, 1]))

        # Determine index of the (unique) line segment that crosses the dateline.
        for i in range(len(vertices)):
            v0 = vertices[i - 1, :]
            v1 = vertices[i, :]
            if crosses_dateline(v0[0], v1[0]):
                delta_lat = abs(reference_angle(v1[0] - v0[0]))
                lat = (np.pi - abs(v0[0])) / delta_lat * v0[1] + (np.pi - abs(v1[0])) / delta_lat * v1[1]
                out_vertices += [np.vstack((vertices[:i, :], [
                    [np.sign(v0[0]) * np.pi, lat],
                    [np.sign(v0[0]) * np.pi, sign_lat * np.pi / 2],
                    [-np.sign(v0[0]) * np.pi, sign_lat * np.pi / 2],
                    [-np.sign(v0[0]) * np.pi, lat],
                ], vertices[i:, :]))]
                break
    elif dateline_crossings:
        frame_poly = geos.Polygon(np.asarray([[-np.pi, np.pi/2], [-np.pi, -np.pi/2], [np.pi, -np.pi/2], [np.pi, np.pi/2]]))
        poly = geos.Polygon(np.column_stack((vertices[:, 0] % (2 * np.pi), vertices[:, 1])))
        if poly.intersects(frame_poly):
            out_vertices += [p.get_coords() for p in poly.intersection(frame_poly)]
        poly = geos.Polygon(np.column_stack((vertices[:, 0] % (-2 * np.pi), vertices[:, 1])))
        if poly.intersects(frame_poly):
            out_vertices += [p.get_coords() for p in poly.intersection(frame_poly)]
    else:
        out_vertices += [vertices]

    return out_vertices


def cut_prime_meridian(vertices):
    """Cut a polygon across the prime meridian, possibly splitting it into multiple
    polygons.  Vertices consist of (longitude, latitude) pairs where longitude
    is always given in terms of a wrapped angle (between 0 and 2*pi).

    This routine is not meant to cover all possible cases; it will only work for
    convex polygons that extend over less than a hemisphere."""

    out_vertices = []

    # Ensure that the list of vertices does not contain a repeated endpoint.
    if (vertices[0, :] == vertices[-1, :]).all():
        vertices = vertices[:-1, :]

    # Ensure that the longitudes are wrapped from 0 to 2*pi.
    vertices = np.column_stack((wrapped_angle(vertices[:, 0]), vertices[:, 1]))

    def count_meridian_crossings(phis):
        n = 0
        for i in range(len(phis)):
            if crosses_meridian(phis[i - 1], phis[i]):
                n += 1
        return n

    def crosses_meridian(phi0, phi1):
        """Test if the segment consisting of v0 and v1 croses the meridian."""
        # If the two angles are in [0, 2pi), then the shortest arc connecting
        # them crosses the meridian if the difference of the angles is greater
        # than pi.
        phi0, phi1 = sorted((phi0, phi1))
        return phi1 - phi0 > np.pi

    # Count the number of times that the polygon crosses the meridian.
    meridian_crossings = count_meridian_crossings(vertices[:, 0])

    if meridian_crossings % 2:
        # FIXME: Use this simple heuristic to decide which pole to enclose.
        sign_lat = np.sign(np.sum(vertices[:, 1]))

        # If there are an odd number of meridian crossings, then the polygon
        # encloses the pole. Any meridian-crossing edge has to be extended
        # into a curve following the nearest polar edge of the map.
        for i in range(len(vertices)):
            v0 = vertices[i - 1, :]
            v1 = vertices[i, :]
            # Loop through the edges until we find one that crosses the meridian.
            if crosses_meridian(v0[0], v1[0]):
                # If this segment crosses the meridian, then fill it to
                # the edge of the map by inserting new line segments.

                # Find the latitude at which the meridian crossing occurs by
                # linear interpolation.
                delta_lon = abs(reference_angle(v1[0] - v0[0]))
                lat = abs(reference_angle(v0[0])) / delta_lon * v0[1] + abs(reference_angle(v1[0])) / delta_lon * v1[1]

                # Find the closer of the left or the right map boundary for
                # each vertex in the line segment.
                lon_0 = 0. if v0[0] < np.pi else 2*np.pi
                lon_1 = 0. if v1[0] < np.pi else 2*np.pi

                # Set the output vertices to the polar cap plus the original
                # vertices.
                out_vertices += [np.vstack((vertices[:i, :], [
                    [lon_0, lat],
                    [lon_0, sign_lat * np.pi / 2],
                    [lon_1, sign_lat * np.pi / 2],
                    [lon_1, lat],
                ], vertices[i:, :]))]

                # Since the polygon is assumed to be convex, the only possible
                # odd number of meridian crossings is 1, so we are now done.
                break
    elif meridian_crossings:
        # Since the polygon is assumed to be convex, if there is an even number
        # of meridian crossings, we know that the polygon does not enclose
        # either pole. Then we can use ordinary Euclidean polygon intersection
        # algorithms.

        # Construct polygon representing map boundaries in longitude and latitude.
        frame_poly = geos.Polygon(np.asarray([[0., np.pi/2], [0., -np.pi/2], [2*np.pi, -np.pi/2], [2*np.pi, np.pi/2]]))

        # Intersect with polygon re-wrapped to lie in [pi, 3*pi).
        poly = geos.Polygon(np.column_stack((reference_angle(vertices[:, 0]) + 2 * np.pi, vertices[:, 1])))
        if poly.intersects(frame_poly):
            out_vertices += [p.get_coords() for p in poly.intersection(frame_poly)]

        # Intersect with polygon re-wrapped to lie in [-pi, pi).
        poly = geos.Polygon(np.column_stack((reference_angle(vertices[:, 0]), vertices[:, 1])))
        if poly.intersects(frame_poly):
            out_vertices += [p.get_coords() for p in poly.intersection(frame_poly)]
    else:
        # Otherwise, there were zero meridian crossings, so we can use the
        # original vertices as is.
        out_vertices += [vertices]

    # Done!
    return out_vertices


def make_rect_poly(width, height, theta, phi, subdivisions=10):
    """Create a Polygon patch representing a rectangle with half-angles width
    and height rotated from the north pole to (theta, phi)."""

    # Convert width and height to radians, then to Cartesian coordinates.
    w = np.sin(np.deg2rad(width))
    h = np.sin(np.deg2rad(height))

    # Generate vertices of rectangle.
    v = np.asarray([[-w, -h], [w, -h], [w, h], [-w, h]])

    # Subdivide.
    v = subdivide_vertices(v, subdivisions)

    # Project onto sphere by calculating z-coord from normalization condition.
    v = np.hstack((v, np.sqrt(1. - np.expand_dims(np.square(v).sum(1), 1))))

    # Transform vertices.
    v = np.dot(v, hp.rotator.euler_matrix_new(phi, theta, 0, Y=True))

    # Convert to spherical polar coordinates.
    thetas, phis = hp.vec2ang(v)

    # FIXME: Remove this after all Matplotlib monkeypatches are obsolete.
    if mpl_version < '1.2.0':
        # Return list of vertices as longitude, latitude pairs.
        return np.column_stack((reference_angle(phis), 0.5 * np.pi - thetas))
    else:
        # Return list of vertices as longitude, latitude pairs.
        return np.column_stack((wrapped_angle(phis), 0.5 * np.pi - thetas))


def heatmap(func, *args, **kwargs):
    "Plot a function on the sphere using the current geographic projection."""

    # Get current axis.
    ax = plt.gca()

    # Set up a regular grid tiling the bounding box of the axes.
    x = np.arange(ax.bbox.x0, ax.bbox.x1 + 0.5, 0.5)
    y = np.arange(ax.bbox.y0, ax.bbox.y1 + 0.5, 0.5)
    xx, yy = np.meshgrid(x, y)

    # Get axis data limits.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Retrieve the inverse transform of the current axes (which converts display
    # coodinates to data coordinates).
    itrans = ax.transData.inverted()

    # Get the longitude and latitude of every point in the bounding box.
    lons, lats = itrans.transform(np.column_stack((xx.ravel(), yy.ravel()))).T

    # Create a mask that selects only the pixels that fall inside the map boundary.
    mask = np.isfinite(lons) & np.isfinite(lats) & (lons >= xmin) & (lons <= xmax)
    zz = np.ma.array(np.empty(lons.shape), mask=~mask)

    # Evaluate the function everywhere that the mask is set.
    zz[mask] = func(lons[mask], lats[mask])

    # Plot bitmap using imshow.
    aximg = plt.imshow(zz.reshape(xx.shape), aspect=ax.get_aspect(),
        origin='upper', extent=(xmin, xmax, ymax, ymin),
        *args, **kwargs)

    # Hide masked-out values by displaying them in transparent white.
    aximg.cmap.set_bad('w', alpha=0.)

    # Done.
    return aximg


def contour(func, *args, **kwargs):
    "Plot a function on the sphere using the current geographic projection."""

    # Get current axis.
    ax = plt.gca()

    # Set up a regular grid tiling in right ascension and declination
    x = np.linspace(*ax.get_xlim(), num=500)
    y = np.linspace(*ax.get_ylim(), num=500)
    xx, yy = np.meshgrid(x, y)

    # Evaluate the function everywhere.
    zz = func(xx, yy)

    # Add contour plot
    ax = plt.contour(xx, yy, zz, *args, **kwargs)

    # Done.
    return ax


def contourf(func, *args, **kwargs):
    "Plot a function on the sphere using the current geographic projection."""

    # Get current axis.
    ax = plt.gca()

    # Set up a regular grid tiling in right ascension and declination
    x = np.linspace(*ax.get_xlim(), num=500)
    y = np.linspace(*ax.get_ylim(), num=500)
    xx, yy = np.meshgrid(x, y)

    # Evaluate the function everywhere.
    zz = func(xx, yy)

    # Add contour plot
    ax = plt.contourf(xx, yy, zz, *args, **kwargs)

    # Done.
    return ax


def _healpix_lookup(map, lon, lat, nest=False, dlon=0):
    """Look up the value of a HEALPix map in the pixel containing the point
    with the specified longitude and latitude."""
    nside = hp.npix2nside(len(map))
    return map[hp.ang2pix(nside, 0.5 * np.pi - lat, lon - dlon, nest=nest)]


def healpix_heatmap(map, *args, **kwargs):
    """Produce a heatmap from a HEALPix map."""
    mpl_kwargs = dict(kwargs)
    dlon = mpl_kwargs.pop('dlon', 0)
    nest = mpl_kwargs.pop('nest', False)
    return heatmap(
        functools.partial(_healpix_lookup, map, nest=nest, dlon=dlon),
        *args, **mpl_kwargs)


def healpix_contour(map, *args, **kwargs):
    """Produce a contour plot from a HEALPix map."""
    mpl_kwargs = dict(kwargs)
    dlon = mpl_kwargs.pop('dlon', 0)
    nest = mpl_kwargs.pop('nest', False)
    return contour(
        functools.partial(_healpix_lookup, map, nest=nest, dlon=dlon),
        *args, **mpl_kwargs)


def healpix_contourf(map, *args, **kwargs):
    """Produce a contour plot from a HEALPix map."""
    mpl_kwargs = dict(kwargs)
    dlon = mpl_kwargs.pop('dlon', 0)
    nest = mpl_kwargs.pop('nest', False)
    return contourf(
        functools.partial(_healpix_lookup, map, nest=nest, dlon=dlon),
        *args, **mpl_kwargs)


def colorbar():
    usetex = matplotlib.rcParams['text.usetex']
    locator = ticker.AutoLocator()
    formatter = ticker.ScalarFormatter(useMathText=not usetex)
    formatter.set_scientific(True)
    formatter.set_powerlimits((1e-1, 100))

    # Plot colorbar
    cb = plt.colorbar(
        orientation='horizontal', shrink=0.4,
        ticks=locator, format=formatter)

    if cb.orientation == 'vertical':
        axis = cb.ax.yaxis
    else:
        axis = cb.ax.xaxis

    # Move order of magnitude text into last label.
    ticklabels = [label.get_text() for label in axis.get_ticklabels()]
    # Avoid putting two '$' next to each other if we are in tex mode.
    if usetex:
        fmt = '{{{0}}}{{{1}}}'
    else:
        fmt = u'{0}{1}'
    ticklabels[-1] = fmt.format(ticklabels[-1], formatter.get_offset())
    axis.set_ticklabels(ticklabels)
    last_ticklabel = axis.get_ticklabels()[-1]
    last_ticklabel.set_horizontalalignment('left')

    # Draw edges in colorbar bands to correct thin white bands that
    # appear in buggy PDF viewers. See:
    # https://github.com/matplotlib/matplotlib/pull/1301
    cb.solids.set_edgecolor("face")

    # Done.
    return cb


def outline_text(ax):
    """If we are using a new enough version of matplotlib, then
    add a white outline to all text to make it stand out from the background."""
    effects = [patheffects.withStroke(linewidth=2, foreground='w')]
    for artist in ax.findobj(text.Text):
        artist.set_path_effects(effects)


class PPPlot(Axes):
    """Construct a probability--probability (P--P) plot.

    Example usage::

        import lalinference.plot
        from matplotlib import pyplot as plt
        import numpy as np

        n = 100
        p_values_1 = np.random.uniform(size=n) # One experiment
        p_values_2 = np.random.uniform(size=n) # Another experiment
        p_values_3 = np.random.uniform(size=n) # Yet another experiment

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111, projection='pp_plot')
        ax.add_confidence_band(n, alpha=0.95) # Add 95% confidence band
        ax.add_diagonal() # Add diagonal line
        ax.add_lightning(n, 20) # Add some random realizations of n samples
        ax.add_series(p_values_1, p_values_2, p_values_3) # Add our data
        fig.savefig('example.png')

    Or, you can also create an instance of ``PPPlot`` by calling its
    constructor directly::

        from lalinference.plot import PPPlot
        from matplotlib import pyplot as plt
        import numpy as np

        rect = [0.1, 0.1, 0.8, 0.8] # Where to place axes in figure
        fig = plt.figure(figsize=(3, 3))
        ax = PPPlot(fig, rect)
        fig.add_axes(ax)
        # ...
        fig.savefig('example.png')
    """

    name = 'pp_plot'

    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super(PPPlot, self).__init__(*args, **kwargs)

        # Square axes, limits from 0 to 1
        self.set_aspect(1.0)
        self.set_xlim(0.0, 1.0)
        self.set_ylim(0.0, 1.0)

    @staticmethod
    def _make_series(p_values):
        for ps in p_values:
            if np.ndim(ps) == 1:
                ps = np.sort(np.atleast_1d(ps))
                n = len(ps)
                xs = np.concatenate(([0.], ps, [1.]))
                ys = np.concatenate(([0.], np.arange(1, n + 1) / n, [1.]))
            elif np.ndim(ps) == 2:
                xs = np.concatenate(([0.], ps[0], [1.]))
                ys = np.concatenate(([0.], ps[1], [1.]))
            else:
                raise ValueError('All series must be 1- or 2-dimensional')
            yield xs
            yield ys

    def add_series(self, *p_values, **kwargs):
        """Add a series of P-values to the plot.

        Parameters
        ----------

        p_values: list or `np.ndarray`
            One or more lists of P-values.

            If an entry in the list is one-dimensional, then it is interpreted
            as an unordered list of P-values. The ranked values will be plotted
            on the horizontal axis, and the cumulative fraction will be plotted
            on the vertical axis.

            If an entry in the list is two-dimensional, then the first subarray
            is plotted on the horizontal axis and the second subarray is plotted
            on the vertical axis.

        drawstyle: ``steps`` or ``lines`` or ``default``
            Plotting style. If ``steps``, then plot steps to represent a
            piecewise constant function. If ``lines``, then connect points with
            straight lines. If ``default`` then use steps if there are more
            than 2 pixels per data point, or else lines.

        Other parameters
        ----------------

        kwargs: optional extra arguments to `~matplotlib.axes.Axes.plot`
        """

        # Construct sequence of x, y pairs to pass to plot()
        args = list(self._make_series(p_values))
        min_n = min(len(ps) for ps in p_values)

        # Make copy of kwargs to pass to plot()
        kwargs = dict(kwargs)
        ds = kwargs.pop('drawstyle', 'default')
        if (ds == 'default' and 2 * min_n > self.bbox.width) or ds == 'lines':
            kwargs['drawstyle'] = 'default'
        else:
            kwargs['drawstyle'] = 'steps-post'

        return self.plot(*args, **kwargs)

    def add_worst(self, *p_values):
        """
        Mark the point at which the deviation is largest.

        Parameters
        ----------

        p_values: list or `np.ndarray`
            Same as in `add_series`.
        """
        series = list(self._make_series(p_values))
        for xs, ys in zip(series[0::2], series[1::2]):
            i = np.argmax(np.abs(ys - xs))
            x = xs[i]
            y = ys[i]
            if y == x:
                continue
            self.plot([x, x, 0], [0, y, y], '--', color='black', linewidth=0.5)
            if y < x:
                self.plot([x, y], [y, y], '-', color='black', linewidth=1)
                self.text(
                    x, y, ' {0:.02f} '.format(np.around(x - y, 2)),
                    ha='left', va='top')
            else:
                self.plot([x, x], [x, y], '-', color='black', linewidth=1)
                self.text(
                    x, y, ' {0:.02f} '.format(np.around(y - x, 2)),
                    ha='right', va='bottom')

    def add_diagonal(self, *args, **kwargs):
        """Add a diagonal line to the plot, running from (0, 0) to (1, 1).

        Other parameters
        ----------------

        kwargs: optional extra arguments to `~matplotlib.axes.Axes.plot`
        """

        # Make copy of kwargs to pass to plot()
        kwargs = dict(kwargs)
        kwargs.setdefault('color', 'black')
        kwargs.setdefault('linestyle', 'dashed')
        kwargs.setdefault('linewidth', 0.5)

        # Plot diagonal line
        return self.plot([0, 1], [0, 1], *args, **kwargs)

    def add_lightning(self, nsamples, ntrials, **kwargs):
        """Add P-values drawn from a random uniform distribution, as a visual
        representation of the acceptable scatter about the diagonal.

        Parameters
        ----------

        nsamples: int
            Number of P-values in each trial

        ntrials: int
            Number of line series to draw.

        Other parameters
        ----------------

        kwargs: optional extra arguments to `~matplotlib.axes.Axes.plot`
        """

        # Draw random samples
        args = np.random.uniform(size=(ntrials, nsamples))

        # Make copy of kwargs to pass to plot()
        kwargs = dict(kwargs)
        kwargs.setdefault('color', 'black')
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('linewidth', 0.25)

        # Plot series
        return self.add_series(*args, **kwargs)

    def add_confidence_band(self, nsamples, alpha=0.95, annotate=True, **kwargs):
        """Add a target confidence band.

        Parameters
        ----------

        nsamples: int
            Number of P-values

        alpha: float, default: 0.95
            Confidence level

        annotate: bool, optional, default: True
            If True, then label the confidence band.

        Other parameters
        ----------------

        kwargs: optional extra arguments to `~matplotlib.axes.Axes.fill_betweenx`
        """
        n = nsamples
        k = np.arange(0, n + 1)
        p = k / n
        ci_lo, ci_hi = scipy.stats.beta.interval(alpha, k + 1, n - k + 1)

        # Make copy of kwargs to pass to fill_betweenx()
        kwargs = dict(kwargs)
        kwargs.setdefault('color', 'lightgray')
        kwargs.setdefault('edgecolor', 'gray')
        kwargs.setdefault('linewidth', 0.5)
        fontsize = kwargs.pop('fontsize', 'x-small')

        if annotate:
            percent_sign = r'\%' if matplotlib.rcParams['text.usetex'] else '%'
            label = 'target {0:g}{1:s}\nconfidence band'.format(
                100 * alpha, percent_sign)

            self.annotate(
                label,
                xy=(1, 1),
                xytext=(0, 0),
                xycoords='axes fraction',
                textcoords='offset points',
                annotation_clip=False,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=fontsize,
                arrowprops=dict(
                    arrowstyle="->",
                    shrinkA=0, shrinkB=2, linewidth=0.5,
                    connectionstyle="angle,angleA=0,angleB=45,rad=0")
                )

        return self.fill_betweenx(p, ci_lo, ci_hi, **kwargs)

    @classmethod
    def _as_mpl_axes(cls):
        """Support placement in figure using the `projection` keyword argument.
        See http://matplotlib.org/devel/add_new_projection.html"""
        return cls, {}

projection_registry.register(PPPlot)

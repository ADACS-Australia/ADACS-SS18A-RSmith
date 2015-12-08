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
Convenience function to produce a sky map from LIGO-LW rows.
"""
from __future__ import division
__author__ = "Leo Singer <leo.singer@ligo.org>"


import itertools
import logging
import time
import numpy as np
import healpy as hp
from . import filter
from . import postprocess
from . import timing
from . import sky_map
import lal, lalsimulation

log = logging.getLogger('BAYESTAR')


def toa_phoa_snr_log_prior(
        (ra, sin_dec, distance, u, twopsi, t),
        min_distance, max_distance, prior_distance_power, max_abs_t):
    return (
        prior_distance_power * np.log(distance)
        if 0 <= ra < 2*np.pi
        and -1 <= sin_dec <= 1
        and min_distance <= distance <= max_distance
        and -1 <= u <= 1
        and 0 <= twopsi < 2*np.pi
        and -max_abs_t <= t <= max_abs_t
        else -np.inf)


def emcee_sky_map(
        logl, loglargs, logp, logpargs, xmin, xmax,
        nside=-1, kde=False, chain_dump=None, max_horizon=1.0):
    # Set up sampler
    import emcee
    ntemps = 20
    nwalkers = 100
    nburnin = 1000
    nthin = 10
    niter = 10000 + nburnin
    ndim = len(xmin)
    sampler = emcee.PTSampler(
        ntemps=ntemps, nwalkers=nwalkers, dim=ndim, logl=logl, logp=logp,
        loglargs=loglargs, logpargs=logpargs)

    # Draw initial state from multivariate uniform distribution
    p0 = np.random.uniform(xmin, xmax, (ntemps, nwalkers, ndim))

    # Collect samples. The .copy() is important because PTSampler.sample()
    # reuses p on every iteration.
    chain = np.vstack([
        p[0, :, :].copy() for p, _, _
        in itertools.islice(
            sampler.sample(p0, iterations=niter, storechain=False),
            nburnin, niter, nthin
        )])

    # Extract polar coordinates. For all likelihoodds, the first two parameters
    # are ra, sin(dec).
    theta = np.arccos(chain[:, 1])
    phi = chain[:, 0]

    # Do adaptive histogram binning if the user has not selected the KDE,
    # or if the user has selected the KDE but we need to guess the resolution.
    if nside == -1 or not kde:
        samples_per_bin = int(np.ceil(0.005 * len(chain)))
        prob = postprocess.adaptive_healpix_histogram(
            theta, phi, samples_per_bin, nside=nside, nest=True)

        # Determine what nside what actually used.
        nside = hp.npix2nside(len(prob))
        # Go one level finer, because the KDE will find more detail.
        nside *= 2

    if kde:
        from sky_area.sky_area_clustering import ClusteredKDEPosterior
        dec = 0.5 * np.pi - theta
        pts = np.column_stack((ra, dec))
        # Pass a random subset of 1000 points to the KDE, to save time.
        pts = np.random.permutation(pts)[:1000, :]
        prob = ClusteredKDEPosterior(pts).as_healpix(nside)

    # Optionally save posterior sample chain to file.
    # Read back in with np.load().
    if chain_dump:
        # Undo numerical conditioning of distances; convert back to Mpc
        chain[:, 2] *= max_horizon
        names = 'ra sin_dec distance cos_inclination twopsi time'.split()[:ndim]
        np.save(chain_dump, np.rec.fromrecords(chain, names=names))

    # Done!
    return prob


def ligolw_sky_map(
        sngl_inspirals, waveform, f_low,
        min_distance=None, max_distance=None, prior_distance_power=None,
        method="toa_phoa_snr", psds=None, nside=-1, chain_dump=None,
        phase_convention='antifindchirp'):
    """Convenience function to produce a sky map from LIGO-LW rows. Note that
    min_distance and max_distance should be in Mpc.

    Returns a 'NESTED' ordering HEALPix image as a Numpy array.
    """

    # Ensure that sngl_inspiral is either a single template or a list of
    # identical templates
    for key in 'mass1 mass2 spin1x spin1y spin1z spin2x spin2y spin2z'.split():
        if hasattr(sngl_inspirals[0], key):
            value = getattr(sngl_inspirals[0], key)
            if any(value != getattr(_, key) for _ in sngl_inspirals):
                raise ValueError(
                    '{0} field is not the same for all detectors'.format(key))

    ifos = [sngl_inspiral.ifo for sngl_inspiral in sngl_inspirals]

    # Extract TOAs in GPS nanoseconds from table.
    toas_ns = [long(sngl_inspiral.get_end().ns())
        for sngl_inspiral in sngl_inspirals]

    # Retrieve phases on arrival from table.
    phoas = np.asarray([sngl_inspiral.coa_phase
        for sngl_inspiral in sngl_inspirals])

    # If using 'findchirp' phase convention rather than gstlal/mbta,
    # then flip signs of phases.
    if phase_convention.lower() == 'antifindchirp':
        log.warn('Using anti-FINDCHIRP phase convention; inverting phases. '
                 'This is currently the default and it is appropriate for '
                 'gstlal and MBTA but not pycbc as of observing run 1 ("O1"). '
                 'The default setting is likely to change in the future.')
        phoas = -phoas

    # Extract SNRs from table.
    snrs = np.asarray([sngl_inspiral.snr
        for sngl_inspiral in sngl_inspirals])

    # Fudge factor for excess estimation error in gstlal_inspiral.
    snrs *= 0.83

    # Look up physical parameters for detector.
    detectors = [lalsimulation.DetectorPrefixToLALDetector(str(ifo))
        for ifo in ifos]
    responses = np.asarray([det.response for det in detectors])
    locations = np.asarray([det.location for det in detectors])

    # Power spectra for each detector.
    if psds is None:
        psds = [timing.get_noise_psd_func(ifo) for ifo in ifos]

    H = filter.sngl_inspiral_psd(sngl_inspirals[0], waveform, f_min=f_low)
    HS = [filter.signal_psd_series(H, S) for S in psds]

    # Signal models for each detector.
    signal_models = [timing.SignalModel(_) for _ in HS]

    # Get SNR=1 horizon distances for each detector.
    horizons = np.asarray([signal_model.get_horizon_distance()
        for signal_model in signal_models])

    weights = [1/np.square(signal_model.get_crb_toa_uncert(snr))
        for signal_model, snr in zip(signal_models, snrs)]

    # Center detector array.
    locations -= np.average(locations, weights=weights, axis=0)

    # Center times of arrival and compute GMST at mean arrival time.
    # Pre-center in integer nanoseconds to preserve precision of
    # initial datatype.
    epoch = sum(toas_ns) // len(toas_ns)
    toas = 1e-9 * (np.asarray(toas_ns) - epoch)
    mean_toa = np.average(toas, weights=weights, axis=0)
    toas -= mean_toa
    epoch += long(round(1e9 * mean_toa))
    epoch = lal.LIGOTimeGPS(0, epoch)
    gmst = lal.GreenwichMeanSiderealTime(epoch)

    # Maximum barycentered arrival time error:
    # |distance from array barycenter to furthest detector| / c + 5 ms
    max_abs_t = np.max(
        np.sqrt(np.sum(np.square(locations / lal.C_SI), axis=1))) + 0.005

    acors, sample_rates = zip(
        *[filter.autocorrelation(_, max_abs_t) for _ in HS])
    sample_rate = sample_rates[0]
    nsamples = len(acors[0])
    assert all(sample_rate == _ for _ in sample_rates)
    assert all(nsamples == len(_) for _ in acors)
    acors = np.asarray(acors)

    # If minimum distance is not specified, then default to 0 Mpc.
    if min_distance is None:
        min_distance = 0

    # If maximum distance is not specified, then default to the SNR=4
    # horizon distance of the most sensitive detector.
    if max_distance is None:
        max_distance = max(horizons) / 4

    # If prior_distance_power is not specified, then default to 2
    # (p(r) ~ r^2, uniform in volume).
    if prior_distance_power is None:
        prior_distance_power = 2

    # Raise an exception if 0 Mpc is the minimum effective distance and the prior
    # is of the form r**k for k<0
    if min_distance == 0 and prior_distance_power < 0:
        raise ValueError(("Prior is a power law r^k with k={}, "
            + "undefined at min_distance=0").format(prior_distance_power))

    # Rescale distances to horizon distance of most sensitive detector.
    max_horizon = np.max(horizons)
    horizons /= max_horizon
    min_distance /= max_horizon
    max_distance /= max_horizon

    # Use KDE for density estimation?
    if method.endswith('_kde'):
        method = method[:-4]
        kde = True
    else:
        kde = False

    # Time and run sky localization.
    start_time = time.time()
    if method == "toa_phoa_snr":
        prob = sky_map.toa_phoa_snr(
            min_distance, max_distance, prior_distance_power, gmst, sample_rate,
            acors, responses, locations, horizons, toas, phoas, snrs, nside)
    elif method == "toa_snr_mcmc":
        prob = emcee_sky_map(
            logl=sky_map.log_likelihood_toa_snr,
            loglargs=(gmst, sample_rate, acors, responses, locations, horizons,
                toas, snrs),
            logp=toa_phoa_snr_log_prior,
            logpargs=(min_distance, max_distance, prior_distance_power,
                max_abs_t),
            xmin=[0, -1, min_distance, -1, 0, -max_abs_t],
            xmax=[2*np.pi, 1, max_distance, 1, 2*np.pi, max_abs_t],
            nside=nside, kde=kde, chain_dump=chain_dump,
            max_horizon=max_horizon)
    elif method == "toa_phoa_snr_mcmc":
        prob = emcee_sky_map(
            logl=sky_map.log_likelihood_toa_phoa_snr,
            loglargs=(gmst, sample_rate, acors, responses, locations, horizons,
                toas, phoas, snrs),
            logp=toa_phoa_snr_log_prior,
            logpargs=(min_distance, max_distance, prior_distance_power,
                max_abs_t),
            xmin=[0, -1, min_distance, -1, 0, -max_abs_t],
            xmax=[2*np.pi, 1, max_distance, 1, 2*np.pi, max_abs_t],
            nside=nside, kde=kde, chain_dump=chain_dump,
            max_horizon=max_horizon)
    else:
        raise ValueError("Unrecognized method: %s" % method)
    end_time = time.time()

    # Find elapsed run time.
    elapsed_time = end_time - start_time

    # Done!
    return prob, epoch, elapsed_time


def gracedb_sky_map(
        coinc_file, psd_file, waveform, f_low, min_distance=None,
        max_distance=None, prior_distance_power=None, nside=-1,
        phase_convention='antifindchirp', f_high_truncate=1.0):
    # LIGO-LW XML imports.
    from . import ligolw
    from glue.ligolw import table as ligolw_table
    from glue.ligolw import utils as ligolw_utils
    from glue.ligolw import lsctables
    import lal.series

    # Read input file.
    xmldoc, _ = ligolw_utils.load_fileobj(
        coinc_file, contenthandler=ligolw.LSCTablesContentHandler)

    # Locate the tables that we need.
    coinc_inspiral_table = ligolw_table.get_table(xmldoc,
        lsctables.CoincInspiralTable.tableName)
    coinc_map_table = ligolw_table.get_table(xmldoc,
        lsctables.CoincMapTable.tableName)
    sngl_inspiral_table = ligolw_table.get_table(xmldoc,
        lsctables.SnglInspiralTable.tableName)

    # Locate the sngl_inspiral rows that we need.
    coinc_inspiral = coinc_inspiral_table[0]
    coinc_event_id = coinc_inspiral.coinc_event_id
    event_ids = [coinc_map.event_id for coinc_map in coinc_map_table
        if coinc_map.coinc_event_id == coinc_event_id]
    sngl_inspirals = [(sngl_inspiral for sngl_inspiral in sngl_inspiral_table
        if sngl_inspiral.event_id == event_id).next() for event_id in event_ids]
    instruments = set(sngl_inspiral.ifo for sngl_inspiral in sngl_inspirals)

    # Read PSDs.
    xmldoc, _ = ligolw_utils.load_fileobj(
        psd_file, contenthandler=lal.series.PSDContentHandler)
    psds = lal.series.read_psd_xmldoc(xmldoc)

    # Rearrange PSDs into the same order as the sngl_inspirals.
    psds = [psds[sngl_inspiral.ifo] for sngl_inspiral in sngl_inspirals]

    # Interpolate PSDs.
    psds = [timing.InterpolatedPSD(filter.abscissa(psd), psd.data.data,
            f_high_truncate=f_high_truncate)
        for psd in psds]

    # TOA+SNR sky localization
    prob, epoch, elapsed_time = ligolw_sky_map(sngl_inspirals, waveform, f_low,
        min_distance, max_distance, prior_distance_power,
        nside=nside, psds=psds, phase_convention=phase_convention)

    return prob, epoch, elapsed_time, instruments

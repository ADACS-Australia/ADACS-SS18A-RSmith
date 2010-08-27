# Copyright (C) 2009  Kipp Cannon
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
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import math
import sys


from glue import iterutils
from glue import segmentsUtils
from pylal import ligolw_burca_tailor
from pylal import git_version
from pylal import inject
from pylal import rate


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


#
# =============================================================================
#
#                             Likelihood Machinery
#
# =============================================================================
#


#
# Coinc params function
#


def coinc_params_func(events, offsetdict):
	#
	# check for coincs that have been vetoed entirely
	#

	if len(events) < 2:
		return None

	#
	# one-instrument parameters
	#

	params = {}

	for event in events:
		prefix = "%s_" % event.ifo

		params["%ssnr2_chi2" % prefix] = (event.snr**2.0, event.chisq / event.chisq_dof)

	#
	# two-instrument parameters
	#

	for event1, event2 in iterutils.choices(sorted(events, key = lambda event: event.ifo), 2):
		if event1.ifo == event2.ifo:
			# shouldn't happen, but might as well check for it
			continue

		prefix = "%s_%s_" % (event1.ifo, event2.ifo)

		dt = float((event1.get_peak() + offsetdict[event1.ifo]) - (event2.get_peak() + offsetdict[event2.ifo]))
		params["%sdt" % prefix] = (dt,)

		dA = math.log10(abs(event1.amplitude / event2.amplitude))
		params["%sdA" % prefix] = (dA,)

		df = float((event1.central_freq + 0.5*event1.bandwidth - event2.central_freq - 0.5*event2.bandwidth)/(event1.central_freq + 0.5*event1.bandwidth + event2.central_freq + 0.5*event2.bandwidth))
		params["%sdf" % prefix] = (df,)

	#
	# done
	#

	return params


#
# Parameter distributions
#


def dt_binning(instrument1, instrument2):
	dt = 0.005 + inject.light_travel_time(instrument1, instrument2)	# seconds
	return rate.NDBins((rate.ATanBins(-dt, +dt, 3001),))


class DistributionsStats(ligolw_burca_tailor.Stats):
	"""
	A subclass of the Stats class used to populate a
	CoincParamsDistribution instance with the data from the outputs of
	ligolw_burca and ligolw_binjfind.
	"""

	binnings = {
		"H1_snr2_chi2": rate.NDBins((rate.ATanLogarithmicBins(10, 1e7, 1201), rate.ATanLogarithmicBins(.1, 1e4, 1201))),
		"H2_snr2_chi2": rate.NDBins((rate.ATanLogarithmicBins(10, 1e7, 1201), rate.ATanLogarithmicBins(.1, 1e4, 1201))),
		"L1_snr2_chi2": rate.NDBins((rate.ATanLogarithmicBins(10, 1e7, 1201), rate.ATanLogarithmicBins(.1, 1e4, 1201))),
		"V1_snr2_chi2": rate.NDBins((rate.ATanLogarithmicBins(10, 1e7, 1201), rate.ATanLogarithmicBins(.1, 1e4, 1201))),
		"H1_H2_dt": dt_binning("H1", "H2"),
		"H1_L1_dt": dt_binning("H1", "L1"),
		"H1_V1_dt": dt_binning("H1", "V1"),
		"H2_L1_dt": dt_binning("H2", "L1"),
		"H2_V1_dt": dt_binning("H2", "V1"),
		"L1_V1_dt": dt_binning("L1", "V1"),
		"H1_H2_dA": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H1_L1_dA": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H1_V1_dA": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H2_L1_dA": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H2_V1_dA": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"L1_V1_dA": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H1_H2_df": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H1_L1_df": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H1_V1_df": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H2_L1_df": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"H2_V1_df": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),)),
		"L1_V1_df": rate.NDBins((rate.ATanBins(-0.5, +0.5, 6001),))
	}

	filters = {
		"H1_snr2_chi2": rate.gaussian_window2d(11, 11, sigma = 20),
		"H2_snr2_chi2": rate.gaussian_window2d(11, 11, sigma = 20),
		"L1_snr2_chi2": rate.gaussian_window2d(11, 11, sigma = 20),
		"V1_snr2_chi2": rate.gaussian_window2d(11, 11, sigma = 20),
		"H1_H2_dt": rate.gaussian_window(11, sigma = 20),
		"H1_L1_dt": rate.gaussian_window(11, sigma = 20),
		"H1_V1_dt": rate.gaussian_window(11, sigma = 20),
		"H2_L1_dt": rate.gaussian_window(11, sigma = 20),
		"H2_V1_dt": rate.gaussian_window(11, sigma = 20),
		"L1_V1_dt": rate.gaussian_window(11, sigma = 20),
		"H1_H2_dA": rate.gaussian_window(11, sigma = 20),
		"H1_L1_dA": rate.gaussian_window(11, sigma = 20),
		"H1_V1_dA": rate.gaussian_window(11, sigma = 20),
		"H2_L1_dA": rate.gaussian_window(11, sigma = 20),
		"H2_V1_dA": rate.gaussian_window(11, sigma = 20),
		"L1_V1_dA": rate.gaussian_window(11, sigma = 20),
		"H1_H2_df": rate.gaussian_window(11, sigma = 20),
		"H1_L1_df": rate.gaussian_window(11, sigma = 20),
		"H1_V1_df": rate.gaussian_window(11, sigma = 20),
		"H2_L1_df": rate.gaussian_window(11, sigma = 20),
		"H2_V1_df": rate.gaussian_window(11, sigma = 20),
		"L1_V1_df": rate.gaussian_window(11, sigma = 20)
	}

	def __init__(self):
		ligolw_burca_tailor.Stats.__init__(self)
		self.distributions = ligolw_burca_tailor.CoincParamsDistributions(**self.binnings)

	def _add_zero_lag(self, param_func, events, offsetdict, vetosegs, *args):
		self.distributions.add_zero_lag(param_func, [event for event in events if event.ifo not in vetosegs or event.get_peak() not in vetosegs[event.ifo]], offsetdict, *args)

	def _add_background(self, param_func, events, offsetdict, vetosegs, *args):
		self.distributions.add_background(param_func, [event for event in events if event.ifo not in vetosegs or event.get_peak() not in vetosegs[event.ifo]], offsetdict, *args)

	def _add_injections(self, param_func, sim, events, offsetdict, vetosegs, *args):
		self.distributions.add_injection(param_func, [event for event in events if event.ifo not in vetosegs or event.get_peak() not in vetosegs[event.ifo]], offsetdict, *args)

	def finish(self):
		self.distributions.finish(filters = self.filters)


#
# Livetime
#


def time_slides_livetime(seglists, time_slides, min_instruments, verbose = False, clip = None):
	"""
	seglists is a segmentlistdict of times when each of a set of
	instruments were on, time_slides is a sequence of
	instrument-->offset dictionaries, each vector of offsets in the
	sequence is applied to the segmentlists and the total time during
	which at least min_instruments were on is summed and returned.  If
	clip is not None, after each offset vector is applied to seglists
	the result is intersected with clip before computing the livetime.
	If verbose is True then progress reports are printed to stderr.
	"""
	livetime = 0.0
	seglists = seglists.copy()	# don't modify original
	N = len(time_slides)
	if verbose:
		print >>sys.stderr, "computing the live time for %d time slides:" % N
	for n, time_slide in enumerate(time_slides):
		if verbose:
			print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / N),
		seglists.offsets.update(time_slide)
		if clip is None:
			livetime += float(abs(segmentsUtils.vote(seglists.values(), min_instruments)))
		else:
			livetime += float(abs(segmentsUtils.vote((seglists & clip).values(), min_instruments)))
	if verbose:
		print >>sys.stderr, "\t100.0%"
	return livetime


def time_slides_livetime_for_instrument_combo(seglists, time_slides, instruments, verbose = False, clip = None):
	"""
	like time_slides_livetime() except computes the time for which
	exactly the instruments given by the sequence instruments were on
	(and nothing else).
	"""
	livetime = 0.0
	# segments for instruments that must be on
	onseglists = seglists.copy(keys = instruments)
	# segments for instruments that must be off
	offseglists = seglists.copy(keys = set(seglists) - set(instruments))
	N = len(time_slides)
	if verbose:
		print >>sys.stderr, "computing the live time for %s in %d time slides:" % (", ".join(instruments), N)
	for n, time_slide in enumerate(time_slides):
		if verbose:
			print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / N),
		onseglists.offsets.update(time_slide)
		offseglists.offsets.update(time_slide)
		if clip is None:
			livetime += float(abs(onseglists.intersection(onseglists.keys()) - offseglists.union(offseglists.keys())))
		else:
			livetime += float(abs((onseglists & clip).intersection(onseglists.keys()) - offseglists.union(offseglists.keys())))
	if verbose:
		print >>sys.stderr, "\t100.0%"
	return livetime


#
# I/O
#


def get_coincparamsdistributions(xmldoc):
	coincparamsdistributions, process_id = ligolw_burca_tailor.coinc_params_distributions_from_xml(xmldoc, u"string_cusp_likelihood")
	return coincparamsdistributions


#
# =============================================================================
#
#                              Database Utilities
#
# =============================================================================
#


def create_recovered_likelihood_table(connection, bb_coinc_def_id):
	"""
	Create a temporary table containing two columns:  the simulation_id
	of an injection, and the highest likelihood ratio at which that
	injection was recovered by a coincidence of type bb_coinc_def_id.
	"""
	cursor = connection.cursor()
	cursor.execute("""
CREATE TEMPORARY TABLE recovered_likelihood (simulation_id TEXT PRIMARY KEY, likelihood REAL)
	""")
	cursor.execute("""
INSERT OR REPLACE INTO
	recovered_likelihood
SELECT
	sim_burst.simulation_id AS simulation_id,
	MAX(coinc_event.likelihood) AS likelihood
FROM
	sim_burst
	JOIN coinc_event_map AS a ON (
		a.table_name == "sim_burst"
		AND a.event_id == sim_burst.simulation_id
	)
	JOIN coinc_event_map AS b ON (
		b.coinc_event_id == a.coinc_event_id
	)
	JOIN coinc_event ON (
		b.table_name == "coinc_event"
		AND b.event_id == coinc_event.coinc_event_id
	)
WHERE
	coinc_event.coinc_def_id == ?
GROUP BY
	sim_burst.simulation_id
	""", (bb_coinc_def_id,))
	cursor.close()

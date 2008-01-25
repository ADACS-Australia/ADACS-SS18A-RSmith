# $Id$
#
# Copyright (C) 2007  Kipp C. Cannon
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
import numpy
from scipy.stats import stats


from glue import iterutils
from glue.ligolw import ligolw
from glue.ligolw import ilwd
from glue.ligolw import param
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils
from pylal import date
from pylal import inject
from pylal import llwapp
from pylal import rate


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__version__ = "$Revision$"[11:-2]
__date__ = "$Date$"[7:-2]


#
# =============================================================================
#
#                                 Speed Hacks
#
# =============================================================================
#


lsctables.LIGOTimeGPS = date.LIGOTimeGPS


#
# =============================================================================
#
#             Generating Coincidence Parameters from Burst Events
#
# =============================================================================
#


#
# All sky version.
#


def coinc_params(events, offsetdict):
	params = {}
	events.sort(lambda a, b: cmp(a.ifo, b.ifo))
	if events:
		# the "time" is the ms_snr squared weighted average of the
		# peak times neglecting light-travel times.  because
		# LIGOTimeGPS objects have overflow problems in this sort
		# of a calculation, the first event's peak time is used as
		# an epoch and the calculations are done w.r.t. that time.
		t = events[0].get_peak()
		t += sum(float(event.get_peak() - t) * event.ms_snr**2.0 for event in events) / sum(event.ms_snr**2.0 for event in events)
		#params["gmst"] = date.XLALGreenwichMeanSiderealTime(t)
	for event1, event2 in iterutils.choices(events, 2):
		if event1.ifo == event2.ifo:
			# a coincidence is parameterized only by
			# inter-instrument deltas
			continue

		prefix = "%s_%s_" % (event1.ifo, event2.ifo)

		# in each of the following, if the list of events contains
		# more than one event from a given instrument, the smallest
		# deltas are recorded

		dt = float(event1.get_peak() + offsetdict[event1.ifo] - event2.get_peak() - offsetdict[event2.ifo])
		name = "%sdt" % prefix
		if name not in params or abs(params[name]) > abs(dt):
			params[name] = dt

		df = (event1.peak_frequency - event2.peak_frequency) / ((event1.peak_frequency + event2.peak_frequency) / 2)
		name = "%sdf" % prefix
		if name not in params or abs(params[name]) > abs(df):
			params[name] = df

		dh = (event1.ms_hrss - event2.ms_hrss) / ((event1.ms_hrss + event2.ms_hrss) / 2)
		name = "%sdh" % prefix
		if name not in params or abs(params[name]) > abs(dh):
			params[name] = dh

		dband = (event1.ms_bandwidth - event2.ms_bandwidth) / ((event1.ms_bandwidth + event2.ms_bandwidth) / 2)
		name = "%sdband" % prefix
		if name not in params or abs(params[name]) > abs(dband):
			params[name] = dband

		ddur = (event1.ms_duration - event2.ms_duration) / ((event1.ms_duration + event2.ms_duration) / 2)
		name = "%sddur" % prefix
		if name not in params or abs(params[name]) > abs(ddur):
			params[name] = ddur
	# convert values to 1-D tuples
	return dict([(name, (value,)) for name, value in params.items()])


#
# Galactic core version.
#


def delay_and_amplitude_correct(event, ra, dec):
	# retrieve station metadata

	detector = inject.cached_detector[inject.prefix_to_name[event.ifo]]

	# delay-correct the event to the geocentre

	peak = event.get_peak()
	delay = date.XLALTimeDelayFromEarthCenter(detector.location, ra, dec, peak)
	event.set_peak(peak - delay)
	event.set_start(event.get_start() - delay)
	event.set_ms_start(event.get_ms_start() - delay)

	# amplitude-correct the event using the polarization-averaged
	# antenna response

	fp, fc = inject.XLALComputeDetAMResponse(detector.response, ra, dec, 0, XLALGreenwichMeanSiderealTime(peak))
	mean_response = math.sqrt(fp**2 + fc**2)
	event.amplitude /= mean_response
	event.ms_hrss /= mean_response

	# done

	return event


def galactic_core_coinc_params(events, offsetdict):
	return coinc_params([delay_and_amplitude_correct(event, MW_CENTER_J2000_RA_RAD, MW_CENTER_J2000_DEC_RAD) for event in events], offsetdict)


#
# =============================================================================
#
#                                 Book-keeping
#
# =============================================================================
#


#
# A class for measuring parameter distributions
#


class CoincParamsDistributions(object):
	def __init__(self, **kwargs):
		self.background_rates = {}
		self.injection_rates = {}
		for param, binning in kwargs.iteritems():
			self.background_rates[param] = rate.BinnedArray(binning)
			self.injection_rates[param] = rate.BinnedArray(binning)

	def __iadd__(self, other):
		if type(other) != type(self):
			raise TypeError, other
		for param, rate in other.background_rates.iteritems():
			if param in self.background_rates:
				self.background_rates[param] += rate
			else:
				self.background_rates[param] = rate
		for param, rate in other.injection_rates.iteritems():
			if param in self.injection_rates:
				self.injection_rates[param] += rate
			else:
				self.injection_rates[param] = rate
		return self

	def add_background(self, param_func, events, timeslide):
		for param, value in param_func(events, timeslide).iteritems():
			rate = self.background_rates[param]
			try:
				rate[value] += 1.0
			except IndexError:
				# param value out of range
				pass

	def add_injection(self, param_func, events, timeslide):
		for param, value in param_func(events, timeslide).iteritems():
			rate = self.injection_rates[param]
			try:
				rate[value] += 1.0
			except IndexError:
				# param value out of range
				pass

	def finish(self, filters = {}):
		default_filter = rate.gaussian_window(21)
		# normalizing each array so that its sum is 1 has the
		# effect of making the integral of P(x) dx equal 1 after
		# the array is transformed to an array of densities (which
		# is done by dividing each bin by dx).
		for name, binnedarray in self.background_rates.items():
			binnedarray.array /= numpy.sum(binnedarray.array)
			rate.to_moving_mean_density(binnedarray, filters.get(name, default_filter))
		for name, binnedarray in self.injection_rates.items():
			binnedarray.array /= numpy.sum(binnedarray.array)
			rate.to_moving_mean_density(binnedarray, filters.get(name, default_filter))
		return self


#
# =============================================================================
#
#                                  Interface
#
# =============================================================================
#


#
# Base class used to hook the database contents into a statistics analyzer.
#


class Stats(object):
	def _add_background(self, param_func, events, timeslide):
		"""
		A subclass should provide an override of this method to do
		whatever it needs to do with a tuple of coincidence events
		identified as "background".
		"""
		raise NotImplementedError


	def _add_injections(self, param_func, sim, events, timeslide):
		"""
		A subclass should provide an override of this method to do
		whatever it needs to do with a tuple of coincidence events
		identified as "injection".
		"""
		raise NotImplementedError


	def add_background(self, param_func, database):
		# iterate over non-zero-lag burst<-->burst coincs
		for (coinc_event_id,) in database.connection.cursor().execute("""
SELECT
	coinc_event_id
FROM
	coinc_event
WHERE
	coinc_def_id == ?
	AND EXISTS (
		SELECT
			*
		FROM
			time_slide
		WHERE
			time_slide.time_slide_id == coinc_event.time_slide_id
			AND time_slide.offset != 0
	)
		""", (database.bb_definer_id,)):
			# retrieve the list of the sngl_bursts in this
			# coinc, and their time slide dictionary
			events = []
			offsetdict = {}
			for values in database.connection.cursor().execute("""
SELECT
	sngl_burst.*,
	time_slide.offset
FROM
	sngl_burst
	JOIN coinc_event_map ON (
		coinc_event_map.table_name == 'sngl_burst'
		AND coinc_event_map.event_id == sngl_burst.event_id
	)
	JOIN coinc_event ON (
		coinc_event.coinc_event_id == coinc_event_map.coinc_event_id
	)
	JOIN time_slide ON (
		coinc_event.time_slide_id == time_slide.time_slide_id
		AND sngl_burst.ifo == time_slide.instrument
	)
WHERE
	coinc_event.coinc_event_id == ?
ORDER BY
	sngl_burst.ifo
			""", (coinc_event_id,)):
				# reconstruct the event
				event = database.sngl_burst_table._row_from_cols(values)

				# add to list
				events.append(event)

				# store the time slide offset
				offsetdict[event.ifo] = values[-1]

			self._add_background(param_func, events, offsetdict)


	def add_injections(self, param_func, database):
		# iterate over burst<-->burst coincs matching injections
		# "exactly"
		for values in database.connection.cursor().execute("""
SELECT
	sim_burst.*,
	burst_coinc_event_map.event_id
FROM
	sim_burst
	JOIN coinc_event_map AS sim_coinc_event_map ON (
		sim_coinc_event_map.table_name == 'sim_burst'
		AND sim_coinc_event_map.event_id == sim_burst.simulation_id
	)
	JOIN coinc_event AS sim_coinc_event ON (
		sim_coinc_event.coinc_event_id == sim_coinc_event_map.coinc_event_id
	)
	JOIN coinc_event_map AS burst_coinc_event_map ON (
		burst_coinc_event_map.coinc_event_id == sim_coinc_event_map.coinc_event_id
		AND burst_coinc_event_map.table_name == 'coinc_event'
	)
WHERE
	sim_coinc_event.coinc_def_id == ?
		""", (database.sce_definer_id,)):
			# retrieve the injection and the coinc_event_id
			sim = database.sim_burst_table._row_from_cols(values)
			coinc_event_id = values[-1]

			# retrieve the list of the sngl_bursts in this
			# coinc, and their time slide dictionary
			events = []
			offsetdict = {}
			for values in database.connection.cursor().execute("""
SELECT
	sngl_burst.*,
	time_slide.offset
FROM
	sngl_burst
	JOIN coinc_event_map ON (
		coinc_event_map.table_name == 'sngl_burst'
		AND coinc_event_map.event_id == sngl_burst.event_id
	)
	JOIN coinc_event ON (
		coinc_event.coinc_event_id == coinc_event_map.coinc_event_id
	)
	JOIN time_slide ON (
		coinc_event.time_slide_id == time_slide.time_slide_id
		AND time_slide.instrument == sngl_burst.ifo
	)
WHERE
	coinc_event.coinc_event_id == ?
ORDER BY
	sngl_burst.ifo
			""", (coinc_event_id,)):
				# reconstruct the burst event
				event = database.sngl_burst_table._row_from_cols(values)

				# add to list
				events.append(event)

				# store the time slide offset
				offsetdict[event.ifo] = values[-1]

			# pass the events to whatever wants them
			self._add_injections(param_func, sim, events, offsetdict)

	def finish(self):
		pass


#
# Covariance matrix
#


def covariance_normalize(c):
	"""
	Normalize a covariance matrix so that the variances (diagonal
	elements) are 1.
	"""
	std_dev = numpy.sqrt(numpy.diagonal(c))
	return c / numpy.outer(std_dev, std_dev)


class Covariance(Stats):
	def __init__(self):
		Stats.__init__(self)
		self.bak_observations = []
		self.inj_observations = []

	def _add_background(self, param_func, events, offsetdict):
		items = param_func(events, offsetdict).items()
		items.sort()
		self.bak_observations.append(tuple([value for name, value in items]))

	def _add_injections(self, param_func, sim, events, offsetdict):
		items = param_func(events, offsetdict).items()
		items.sort()
		self.inj_observations.append(tuple([value for name, value in items]))

	def finish(self):
		self.bak_cov = covariance_normalize(stats.cov(self.bak_observations))
		self.inj_cov = covariance_normalize(stats.cov(self.inj_observations))


#
# Parameter distributions
#


def dt_binning(instrument1, instrument2):
	dt = 0.01 + inject.light_travel_time(instrument1, instrument2)
	return rate.NDBins((rate.ATanBins(-dt, +dt, 24001),))


class DistributionsStats(Stats):
	"""
	A subclass of the Stats class used to populate a
	CoincParamsDistribution instance with the data from the outputs of
	ligolw_burca and ligolw_binjfind.
	"""

	binnings = {
		"H1_H2_dband": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_L1_dband": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H2_L1_dband": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_H2_ddur": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_L1_ddur": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H2_L1_ddur": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_H2_df": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_L1_df": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H2_L1_df": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_H2_dh": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_L1_dh": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H2_L1_dh": rate.NDBins((rate.LinearBins(-2.0, +2.0, 24001),)),
		"H1_H2_dt": dt_binning("H1", "H2"),
		"H1_L1_dt": dt_binning("H1", "L1"),
		"H2_L1_dt": dt_binning("H2", "L1")#,
		#"gmst": rate.NDBins((rate.LinearBins(0.0, 2 * math.pi, 24001),))
	}

	filters = {
		"H1_H2_dband": rate.gaussian_window(21),
		"H1_L1_dband": rate.gaussian_window(21),
		"H2_L1_dband": rate.gaussian_window(21),
		"H1_H2_ddur": rate.gaussian_window(21),
		"H1_L1_ddur": rate.gaussian_window(21),
		"H2_L1_ddur": rate.gaussian_window(21),
		"H1_H2_df": rate.gaussian_window(21),
		"H1_L1_df": rate.gaussian_window(21),
		"H2_L1_df": rate.gaussian_window(21),
		"H1_H2_dh": rate.gaussian_window(21),
		"H1_L1_dh": rate.gaussian_window(21),
		"H2_L1_dh": rate.gaussian_window(21),
		"H1_H2_dt": rate.gaussian_window(21),
		"H1_L1_dt": rate.gaussian_window(21),
		"H2_L1_dt": rate.gaussian_window(21)#,
		#"gmst": rate.gaussian_window(21)
	}

	def __init__(self):
		Stats.__init__(self)
		self.distributions = CoincParamsDistributions(**self.binnings)

	def _add_background(self, param_func, events, offsetdict):
		self.distributions.add_background(param_func, events, offsetdict)

	def _add_injections(self, param_func, sim, events, offsetdict):
		self.distributions.add_injection(param_func, events, offsetdict)

	def finish(self):
		self.distributions.finish(filters = self.filters)


#
# =============================================================================
#
#                                     I/O
#
# =============================================================================
#


def coinc_params_distributions_to_xml(process, coinc_params_distributions, name):
	xml = ligolw.LIGO_LW({u"Name": u"%s:pylal_ligolw_burca_tailor_coincparamsdistributions" % name})
	xml.appendChild(param.new_param(u"process_id", u"ilwd:char", process.process_id))
	for name, binnedarray in coinc_params_distributions.background_rates.iteritems():
		xml.appendChild(rate.binned_array_to_xml(binnedarray, "background:%s" % name))
	for name, binnedarray in coinc_params_distributions.injection_rates.iteritems():
		xml.appendChild(rate.binned_array_to_xml(binnedarray, "injection:%s" % name))
	return xml


def coinc_params_distributions_from_xml(xml, name):
	xml, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:pylal_ligolw_burca_tailor_coincparamsdistributions" % name]
	process_id = param.get_pyvalue(xml, u"process_id")
	names = [elem.getAttribute("Name").split(":")[1] for elem in xml.childNodes if elem.getAttribute("Name")[:11] == "background:"]
	c = CoincParamsDistributions()
	for name in names:
		c.background_rates[name] = rate.binned_array_from_xml(xml, "background:%s" % name)
		c.injection_rates[name] = rate.binned_array_from_xml(xml, "injection:%s" % name)
	return c, process_id


def coinc_params_distributions_from_filename(filename, name, verbose = False):
	xmldoc = utils.load_filename(filename, verbose = verbose, gz = (filename or "stdin").endswith(".gz"))
	result, process_id = coinc_params_distributions_from_xml(xmldoc, name)
	seglists = table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName).get_out_segmentlistdict([process_id])
	xmldoc.unlink()
	return result, seglists


#
# =============================================================================
#
#                             Process Information
#
# =============================================================================
#


process_program_name = "ligolw_burca_tailor"


def append_process(xmldoc, **kwargs):
	return llwapp.append_process(xmldoc, program = process_program_name, version = __version__, cvs_repository = "lscsoft", cvs_entry_time = __date__, comment = kwargs["comment"])


#
# =============================================================================
#
#                           Likelihood Control File
#
# =============================================================================
#


#
# Construct LIGO Light Weight likelihood distributions document.
#


def gen_likelihood_control(coinc_params_distributions, seglists):
	xmldoc = ligolw.Document()
	node = xmldoc.appendChild(ligolw.LIGO_LW())

	node.appendChild(lsctables.New(lsctables.ProcessTable))
	node.appendChild(lsctables.New(lsctables.ProcessParamsTable))
	node.appendChild(lsctables.New(lsctables.SearchSummaryTable))
	process = append_process(xmldoc, comment = u"")
	llwapp.append_search_summary(xmldoc, process, ifos = "+".join(seglists.keys()), inseg = seglists.extent_all(), outseg = seglists.extent_all())

	node.appendChild(coinc_params_distributions_to_xml(process, coinc_params_distributions, u"ligolw_burca_tailor"))

	llwapp.set_process_end_time(process)

	return xmldoc


#
# =============================================================================
#
#                           param_dist_definer:table
#
# =============================================================================
#


ParamDistDefinerID = ilwd.get_ilwdchar_class(u"param_dist_definer", u"param_dist_def_id")


class ParamDistDefinerTable(table.Table):
	tableName = "param_dist_definer:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"param_dist_def_id": "ilwd:char",
		"search": "lstring",
		"distribution_name": "lstring",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"array_name": "lstring"
	}
	constraints = "PRIMARY KEY (param_dist_def_id)"
	next_id = ParamDistDefinerID(0)


class ParamDistDefiner(object):
	__slots__ = ParamDistDefinerTable.validcolumns.keys()


ParamDistDefinerTable.RowType = ParamDistDefiner

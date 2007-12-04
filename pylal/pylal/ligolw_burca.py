# $Id$
#
# Copyright (C) 2006  Kipp C. Cannon
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


import bisect
import math
import sys


from glue import segments
from glue.ligolw import table
from glue.ligolw import lsctables
from pylal import llwapp
from pylal import snglcoinc
from pylal.date import LIGOTimeGPS
from pylal.xlal import tools


lsctables.CoincMapTable.RowType = lsctables.CoincMap = tools.CoincMap


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


lsctables.LIGOTimeGPS = LIGOTimeGPS


def sngl_burst___cmp__(self, other):
	# compare self's peak time to the LIGOTimeGPS instance other
	return cmp(self.peak_time, other.seconds) or cmp(self.peak_time_ns, other.nanoseconds)


lsctables.SnglBurst.__cmp__ = sngl_burst___cmp__


#
# =============================================================================
#
#                           Add Process Information
#
# =============================================================================
#


process_program_name = "ligolw_burca"


def append_process(xmldoc, **kwargs):
	process = llwapp.append_process(xmldoc, program = process_program_name, version = __version__, cvs_repository = "lscsoft", cvs_entry_time = __date__, comment = kwargs["comment"])

	params = [
		("--program", "lstring", kwargs["program"]),
		("--coincidence-algorithm", "lstring", kwargs["coincidence_algorithm"])
	]
	if "stringcusp_params" in kwargs:
		params += [("--stringcusp-params", "lstring", kwargs["stringcusp_params"])]
	if "force" in kwargs:
		params += [("--force", "lstring", "")]
	if kwargs["coincidence_algorithm"] in ("stringcusp",):
		for key, value in kwargs["thresholds"].iteritems():
			if key[0] < key[1]:
				params += [("--thresholds", "lstring", "%s,%s=%s" % (key[0], key[1], ",".join(map(str, value))))]

	llwapp.append_process_params(xmldoc, process, params)

	return process


def dbget_thresholds(connection):
	"""
	Extract the --thresholds arguments that had been given to the
	ligolw_burca job recorded in the process table of database to which
	connection points.
	"""
	thresholds = snglcoinc.parse_thresholds(map(str, llwapp.dbget_process_params(connection, process_program_name, "--thresholds")))
	for pair, (dt, df, dhrss) in thresholds.items():
		thresholds[pair] = (float(dt), float(df), float(dhrss))
	return thresholds


#
# =============================================================================
#
#                          CoincTables Customizations
#
# =============================================================================
#


#
# For use with excess power.
#


ExcessPowerCoincDef = lsctables.CoincDef(search = u"excesspower", search_coinc_type = 0, description = u"sngl_burst<-->sngl_burst coincidences")


def make_multi_burst(process_id, coinc_event_id, events):
	multiburst = lsctables.MultiBurst()
	multiburst.process_id = process_id
	multiburst.coinc_event_id = coinc_event_id

	# snr = root sum of ms_snr squares
	multiburst.snr = math.sqrt(sum(event.ms_snr**2.0 for event in events))

	# duration = ms_snr squared weighted average of durations
	multiburst.duration = sum(event.ms_snr**2.0 * event.duration for event in events) / multiburst.snr**2.0

	# central_freq = ms_snr squared weighted average of peak frequencies
	multiburst.central_freq = sum(event.ms_snr**2.0 * event.peak_frequency for event in events) / multiburst.snr**2.0

	# bandwidth = ms_snr squared weighted average of bandwidths
	multiburst.bandwidth = sum(event.ms_snr**2.0 * event.bandwidth for event in events) / multiburst.snr**2.0

	# confidence = minimum of confidences
	multiburst.confidence = min(event.confidence for event in events)

	# "amplitude" = h_rss of event with highest confidence
	multiburst.amplitude = max((event.ms_confidence, event.ms_hrss) for event in events)[1]

	# done
	return multiburst


class ExcessPowerCoincTables(snglcoinc.CoincTables):
	def __init__(self, xmldoc, coinc_definer_row_type):
		snglcoinc.CoincTables.__init__(self, xmldoc, coinc_definer_row_type)

		# find the multi_burst table or create one if not found
		try:
			self.multibursttable = table.get_table(xmldoc, lsctables.MultiBurstTable.tableName)
		except ValueError:
			self.multibursttable = lsctables.New(lsctables.MultiBurstTable, ("process_id", "duration", "central_freq", "bandwidth", "snr", "confidence", "amplitude", "coinc_event_id"))
			xmldoc.childNodes[0].appendChild(self.multibursttable)

	def append_coinc(self, process_id, time_slide_id, events):
		coinc = snglcoinc.CoincTables.append_coinc(self, process_id, time_slide_id, events)
		self.multibursttable.append(make_multi_burst(process_id, coinc.coinc_event_id, events))
		return coinc


#
# For use with string cusp
#


StringCuspCoincDef = lsctables.CoincDef(search = u"StringCusp", search_coinc_type = 0, description = u"sngl_burst<-->sngl_burst coincidences")


#
# =============================================================================
#
#                            Event List Management
#
# =============================================================================
#


#
# For use with excess power coincidence test
#


def ExcessPowerMaxSegmentGap(xmldoc, thresholds):
	"""
	Determine the maximum allowed segment gap for use with the excess
	power coincidence test.
	"""
	# force triggers from all processes in the input file to be
	# considered:  the pipeline script solves the problem of assembling
	# coincident trigger files for processing by burca.
	return float("inf")


class ExcessPowerEventList(snglcoinc.EventList):
	"""
	A customization of the EventList class for use with the excess
	power search.
	"""
	def make_index(self):
		"""
		Sort events by peak time so that a bisection search can
		retrieve them.  Note that the bisection search relies on
		the __cmp__() method of the SnglBurst row class having
		previously been set to compare the event's peak time to a
		LIGOTimeGPS.
		"""
		# sort by peak time
		self.sort(lambda a, b: cmp(a.peak_time, b.peak_time) or cmp(a.peak_time_ns, b.peak_time_ns))

		# for the events in this list, record the largest
		# difference between an event's peak time and either its
		# start or stop times
		if self:
			self.max_edge_peak_delta = max([max(float(event.get_peak() - event.get_start()), float(event.get_start() + event.duration - event.get_peak())) for event in self])
		else:
			# max() doesn't like empty lists
			self.max_edge_peak_delta = 0

	def _add_offset(self, delta):
		"""
		Add an amount to the peak time of each event.
		"""
		for event in self:
			event.set_peak(event.get_peak() + delta)
			event.set_start(event.get_start() + delta)

	def get_coincs(self, event_a, light_travel_time, comparefunc):
		# location of event_a's peak time
		min_peak = max_peak = dt = event_a.get_peak()

		# event_a's start time
		s = event_a.get_start()

		# largest difference between event_a's peak time and either
		# its start or stop times
		dt = max(float(dt - s), float(s + event_a.duration - dt))

		# add our own max_edge_peak_delta and the light travel time
		# between the two instruments (when done, if event_a's peak
		# time differs by more than this much from the peak time of
		# an event in this list then it is *impossible* for them to
		# be coincident)
		dt += self.max_edge_peak_delta + light_travel_time

		# add to and subtract from event_a's peak time to get the
		# earliest and latest peak times of events in this list
		# that could possibly be coincident with event_a
		min_peak -= dt
		max_peak += dt

		# extract the subset of events from this list that pass
		# coincidence with event_a (use bisection searches for the
		# minimum and maximum allowed peak times to quickly
		# identify a subset of the full list)
		return [event_b for event_b in self[bisect.bisect_left(self, min_peak) : bisect.bisect_right(self, max_peak)] if not comparefunc(event_a, event_b, light_travel_time)]


#
# For use with string coincidence test
#


def StringMaxSegmentGap(xmldoc, thresholds):
	"""
	Determine the maximum allowed segment gap for use with the string
	coincidence test.
	"""
	# force triggers from all processes in the input file to be
	# considered:  the pipeline script solves the problem of assembling
	# coincident trigger files for processing by burca.
	return float("inf")


class StringEventList(snglcoinc.EventList):
	"""
	A customization of the EventList class for use with the string
	search.
	"""
	def make_index(self):
		"""
		Sort events by peak time so that a bisection search can
		retrieve them.  Note that the bisection search relies on
		the __cmp__() method of the SnglBurst row class having
		previously been set to compare the event's peak time to a
		LIGOTimeGPS.
		"""
		self.sort(lambda a, b: cmp(a.peak_time, b.peak_time) or cmp(a.peak_time_ns, b.peak_time_ns))

	def _add_offset(self, delta):
		"""
		Add an amount to the peak time of each event.
		"""
		for event in self:
			event.set_peak(event.get_peak() + delta)

	def get_coincs(self, event_a, threshold, comparefunc):
		min_peak = max_peak = event_a.get_peak()
		min_peak -= threshold[0]
		max_peak += threshold[0]
		return [event_b for event_b in self[bisect.bisect_left(self, min_peak) : bisect.bisect_right(self, max_peak)] if not comparefunc(event_a, event_b, threshold)]


#
# =============================================================================
#
#                              Coincidence Tests
#
# =============================================================================
#


def ExcessPowerCoincCompare(a, b, light_travel_time):
	"""
	The events are coincident if their time-frequency tiles intersect
	after considering the light travel time between instruments.

	Returns False (a & b are coincident) if the two events match within
	the tresholds.  Retruns non-zero otherwise.
	"""
	return a.get_band().disjoint(b.get_band()) or a.get_period().protract(light_travel_time).disjoint(b.get_period())


def StringCoincCompare(a, b, thresholds):
	"""
	Returns False (a & b are coincident) if their peak times agree
	within dt, and in the case of H1+H2 pairs if their amplitudes agree
	according to some kinda test.
	"""
	# unpack thresholds
	dt, kappa, epsilon = thresholds

	# test for time coincidence
	coincident = abs(float(a.get_peak() - b.get_peak())) <= dt

	# for H1+H2, also test for amplitude coincidence
	if a.ifo in ("H1", "H2") and b.ifo in ("H1", "H2"):
		adelta = abs(a.amplitude) * (kappa / a.snr + epsilon)
		bdelta = abs(b.amplitude) * (kappa / b.snr + epsilon)
		coincident = coincident and a.amplitude - adelta <= b.amplitude <= a.amplitude + adelta and b.amplitude - bdelta <= a.amplitude <= b.amplitude + bdelta

	# return result
	return not coincident


#
# =============================================================================
#
#                                 Library API
#
# =============================================================================
#


def ligolw_burca(
	xmldoc,
	program,
	process_id,
	EventListType,
	CoincTables,
	coinc_definer_row,
	event_comparefunc,
	thresholds,
	ntuple_comparefunc = lambda events: False,
	get_max_segment_gap = lambda xmldoc, thresholds: float("inf"),
	verbose = False
):
	#
	# prepare the coincidence table interface
	#

	if verbose:
		print >>sys.stderr, "indexing ..."
	coinc_tables = CoincTables(xmldoc, coinc_definer_row)

	#
	# build the event list accessors, populated with events from those
	# processes that can participate in a coincidence
	#

	eventlists = snglcoinc.make_eventlists(xmldoc, EventListType, lsctables.SnglBurstTable.tableName, get_max_segment_gap(xmldoc, thresholds), program)
	avail_instruments = set(eventlists.keys())

	#
	# iterate over time slides
	#

	time_slide_ids = coinc_tables.time_slide_ids()
	for n, time_slide_id in enumerate(time_slide_ids):
		#
		# retrieve the current time slide
		#

		offsetdict = coinc_tables.get_time_slide(time_slide_id)
		offset_instruments = set(offsetdict.keys())
		if verbose:
			print >>sys.stderr, "time slide %d/%d: %s" % (n + 1, len(time_slide_ids), ", ".join(["%s = %+.16g s" % (i, o) for i, o in offsetdict.items()]))

		#
		# can we do it?
		#

		if len(offset_instruments) < 2:
			if verbose:
				print >>sys.stderr, "\tsingle-instrument time slide: skipped"
			continue
		if not offset_instruments.issubset(avail_instruments):
			if verbose:
				print >>sys.stderr, "\twarning: skipping due to insufficient data"
			continue

		#
		# apply offsets to events
		#

		if verbose:
			print >>sys.stderr, "\tapplying time offsets ..."
		eventlists.set_offsetdict(offsetdict)

		#
		# search for and record coincidences
		#

		if verbose:
			print >>sys.stderr, "\tsearching ..."
		for ntuple in snglcoinc.CoincidentNTuples(eventlists, event_comparefunc, offset_instruments, thresholds, verbose = verbose):
			if not ntuple_comparefunc(ntuple):
				coinc_tables.append_coinc(process_id, time_slide_id, ntuple)

	#
	# remove time offsets from events
	#

	eventlists.remove_offsetdict()

	#
	# done
	#

	return xmldoc

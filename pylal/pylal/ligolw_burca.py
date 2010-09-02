# Copyright (C) 2006  Kipp Cannon
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
import itertools
import math
import sys


from glue.ligolw import lsctables
from glue.ligolw.utils import process as ligolw_process
from pylal import git_version
from pylal import llwapp
from pylal import snglcoinc
from pylal.xlal import tools
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


#
# =============================================================================
#
#                                 Speed Hacks
#
# =============================================================================
#


lsctables.CoincMapTable.RowType = lsctables.CoincMap = tools.CoincMap
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
	process = llwapp.append_process(xmldoc, program = process_program_name, version = __version__, cvs_repository = u"lscsoft", cvs_entry_time = __date__, comment = kwargs["comment"])

	params = [
		(u"--program", u"lstring", kwargs["program"]),
		(u"--coincidence-algorithm", u"lstring", kwargs["coincidence_algorithm"])
	]
	if "stringcusp_params" in kwargs:
		params += [(u"--stringcusp-params", u"lstring", kwargs["stringcusp_params"])]
	if "force" in kwargs and kwargs["force"]:
		params += [(u"--force", None, None)]
	if kwargs["coincidence_algorithm"] in ("stringcusp",):
		for (a, b), value in kwargs["thresholds"].items():
			if a < b:
				params += [(u"--thresholds", u"lstring", u"%s,%s=%s" % (a, b, ",".join(map(str, value))))]
	if "coincidence_segments" in kwargs and kwargs["coincidence_segments"] is not None:
		params += [(u"--coincidence-segments", u"lstring", kwargs["coincidence_segments"])]

	ligolw_process.append_process_params(xmldoc, process, params)

	return process


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


ExcessPowerBBCoincDef = lsctables.CoincDef(search = u"excesspower", search_coinc_type = 0, description = u"sngl_burst<-->sngl_burst coincidences")


def make_multi_burst(process_id, coinc_event_id, events, offset_vector):
	multiburst = lsctables.MultiBurst()
	multiburst.process_id = process_id
	multiburst.coinc_event_id = coinc_event_id

	# snr = root sum of ms_snr squares

	multiburst.snr = math.sqrt(sum(event.ms_snr**2.0 for event in events))

	# peak time = ms_snr squared weighted average of peak times (no
	# attempt to account for inter-site delays).  LIGOTimeGPS objects
	# don't like being multiplied by things, so the first event's peak
	# time is used as a reference epoch.

	t = events[0].get_peak() + offset_vector[events[0].ifo]
	multiburst.set_peak(t + sum(event.ms_snr**2.0 * float(event.get_peak() + offset_vector[event.ifo] - t) for event in events) / multiburst.snr**2.0)

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

	# populate the false alarm rate with none.

	multiburst.false_alarm_rate = None

	# done

	return multiburst


class ExcessPowerCoincTables(snglcoinc.CoincTables):
	def __init__(self, xmldoc):
		snglcoinc.CoincTables.__init__(self, xmldoc)

		# find the multi_burst table or create one if not found
		try:
			self.multibursttable = lsctables.table.get_table(xmldoc, lsctables.MultiBurstTable.tableName)
		except ValueError:
			self.multibursttable = lsctables.New(lsctables.MultiBurstTable, ("process_id", "duration", "central_freq", "bandwidth", "snr", "confidence", "amplitude", "coinc_event_id"))
			xmldoc.childNodes[0].appendChild(self.multibursttable)

	def append_coinc(self, process_id, time_slide_id, coinc_def_id, events):
		coinc = snglcoinc.CoincTables.append_coinc(self, process_id, time_slide_id, coinc_def_id, events)
		self.multibursttable.append(make_multi_burst(process_id, coinc.coinc_event_id, events, self.time_slide_index[time_slide_id]))
		return coinc


#
# For use with string cusp
#


StringCuspBBCoincDef = lsctables.CoincDef(search = u"StringCusp", search_coinc_type = 0, description = u"sngl_burst<-->sngl_burst coincidences")


class StringCuspCoincTables(snglcoinc.CoincTables):
	def append_coinc(self, process_id, time_slide_id, coinc_def_id, events):
		coinc = snglcoinc.CoincTables.append_coinc(self, process_id, time_slide_id, coinc_def_id, events)
		coinc.set_instruments(event.ifo for event in events)
		return coinc


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

	def get_coincs(self, event_a, offset_a, light_travel_time, ignored, comparefunc):
		# event_a's peak time
		peak = event_a.get_peak()

		# difference between event_a's peak and start times
		dt = float(peak - event_a.get_start())

		# largest difference between event_a's peak time and either
		# its start or stop times
		dt = max(dt, event_a.duration - dt)

		# add our own max_edge_peak_delta and the light travel time
		# between the two instruments (when done, if event_a's peak
		# time differs by more than this much from the peak time of
		# an event in this list then it is *impossible* for them to
		# be coincident)
		dt += self.max_edge_peak_delta + light_travel_time

		# apply time shift
		peak += offset_a - self.offset

		# extract the subset of events from this list that pass
		# coincidence with event_a (use bisection searches for the
		# minimum and maximum allowed peak times to quickly
		# identify a subset of the full list)
		return [event_b for event_b in self[bisect.bisect_left(self, peak - dt) : bisect.bisect_right(self, peak + dt)] if not comparefunc(event_a, offset_a, event_b, self.offset, light_travel_time, ignored)]


#
# For use with string coincidence test
#


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

	def get_coincs(self, event_a, offset_a, light_travel_time, threshold, comparefunc):
		min_peak = max_peak = event_a.get_peak() + offset_a - self.offset
		min_peak -= threshold[0] + light_travel_time
		max_peak += threshold[0] + light_travel_time
		return [event_b for event_b in self[bisect.bisect_left(self, min_peak) : bisect.bisect_right(self, max_peak)] if not comparefunc(event_a, offset_a, event_b, self.offset, light_travel_time, threshold)]


#
# =============================================================================
#
#                              Coincidence Tests
#
# =============================================================================
#


def ExcessPowerCoincCompare(a, offseta, b, offsetb, light_travel_time, thresholds):
	if abs(a.central_freq - b.central_freq) > (a.bandwidth + b.bandwidth) / 2:
		return True

	astart = a.get_start() + offseta
	bstart = b.get_start() + offsetb
	if astart > bstart + b.duration + light_travel_time:
		# a starts after the end of b
		return True

	if bstart > astart + a.duration + light_travel_time:
		# b starts after the end of a
		return True

	# time-frequency times intersect
	return False


def StringCoincCompare(a, offseta, b, offsetb, light_travel_time, thresholds):
	"""
	Returns False (a & b are coincident) if the events' peak times
	differ from each other by no more than dt plus the light travel
	time from one instrument to the next.
	"""
	# unpack thresholds (it's just the \Delta t window)
	dt, = thresholds

	# test for time coincidence
	coincident = abs(float(a.get_peak() + offseta - b.get_peak() - offsetb)) <= (dt + light_travel_time)

	# return result
	return not coincident


def StringNTupleCoincCompare(events, offset_vector):
	instruments = set(event.ifo for event in events)

	# disallow H1,H2 only coincs
	coincident = instruments != set(["H1", "H2"])

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
	process_id,
	EventListType,
	CoincTables,
	coinc_definer_row,
	event_comparefunc,
	thresholds,
	ntuple_comparefunc = lambda events, offset_vector: False,
	verbose = False
):
	#
	# prepare the coincidence table interface.
	#

	if verbose:
		print >>sys.stderr, "indexing ..."
	coinc_tables = CoincTables(xmldoc)
	coinc_def_id = llwapp.get_coinc_def_id(xmldoc, coinc_definer_row.search, coinc_definer_row.search_coinc_type, create_new = True, description = coinc_definer_row.description)
	sngl_index = dict((row.event_id, row) for row in lsctables.table.get_table(xmldoc, lsctables.SnglBurstTable.tableName))

	#
	# build the event list accessors, populated with events from those
	# processes that can participate in a coincidence
	#

	eventlists = snglcoinc.make_eventlists(xmldoc, EventListType, lsctables.SnglBurstTable.tableName)

	#
	# construct offset vector assembly graph
	#

	time_slide_graph = snglcoinc.TimeSlideGraph(coinc_tables.time_slide_index, verbose = verbose)

	#
	# loop over the items in time_slide_graph.head, producing all of
	# those n-tuple coincidences
	#

	if verbose:
		print >>sys.stderr, "constructing coincs for target offset vectors ..."
	for n, node in enumerate(time_slide_graph.head):
		if verbose:
			print >>sys.stderr, "%d/%d: %s" % (n + 1, len(time_slide_graph.head), ", ".join(("%s = %+.16g s" % x) for x in sorted(node.offset_vector.items())))
		for coinc in itertools.chain(node.get_coincs(eventlists, event_comparefunc, thresholds, verbose), node.unused_coincs):
			ntuple = tuple(sngl_index[id] for id in coinc)
			if not ntuple_comparefunc(ntuple, node.offset_vector):
				coinc_tables.append_coinc(process_id, node.time_slide_id, coinc_def_id, ntuple)

	#
	# remove time offsets from events
	#

	eventlists.remove_offsetdict()

	#
	# done
	#

	return xmldoc

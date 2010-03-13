# Copyright (C) 2006  Kipp Cannon, Drew G. Keppel
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


"""
Generic coincidence engine for use with time-based event lists in LIGO
Light Weight XML documents.
"""


import bisect
import sys


from glue import iterutils
from glue.ligolw import table
from glue.ligolw import lsctables
from pylal import git_version
from pylal import inject
from pylal import llwapp
from pylal import ligolw_tisi
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS # FIXME:  not needed once graph construction arithmetic is fixed


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


#
# =============================================================================
#
#                               Time Slide Graph
#
# =============================================================================
#


def offset_vector_str(offset_vector, compact = False):
	if compact:
		return ",".join(("%s=%g" % x) for x in sorted(offset_vector.items()))
	return ", ".join(("%s = %+.16g s" % x) for x in sorted(offset_vector.items()))


class TimeSlideGraphNode(object):
	def __init__(self, offset_vector, time_slide_id = None):
		self.time_slide_id = time_slide_id
		self.offset_vector = offset_vector
		self.deltas = frozenset(ligolw_tisi.offset_vector_to_deltas(offset_vector).items())
		self.components = None
		self.coincs = None
		self.unused_coincs = set()

	def get_coincs(self, eventlists, event_comparefunc, thresholds, verbose = False):
		#
		# has this node already been visited?  if so, return the
		# answer we already know
		#

		if self.coincs is not None:
			if verbose:
				print >>sys.stderr, "\treusing %s" % offset_vector_str(self.offset_vector)
			return self.coincs

		#
		# is this a leaf node?  construct the coincs explicitly
		#

		if self.components is None:
			if verbose:
				print >>sys.stderr, "\tconstructing %s ..." % offset_vector_str(self.offset_vector)
			#
			# can we do it?
			#

			avail_instruments = set(eventlists)
			offset_instruments = set(self.offset_vector)
			if not offset_instruments.issubset(avail_instruments):
				if verbose:
					print >>sys.stderr, "\twarning: do not have data for instrument(s) %s ... assuming 0 coincs" % ", ".join(offset_instruments - avail_instruments)
				self.coincs = tuple()
				return self.coincs

			#
			# apply offsets to events
			#

			if verbose:
				print >>sys.stderr, "\tapplying offsets ..."
			eventlists.set_offsetdict(self.offset_vector)

			#
			# search for and record coincidences
			#

			if verbose:
				print >>sys.stderr, "\tsearching ..."
			# FIXME:  assumes the instrument column is named
			# "ifo".  works for inspirals, bursts, and
			# ring-downs.
			self.coincs = sorted(tuple(event.event_id for event in sorted(double, lambda a, b: cmp(a.ifo, b.ifo))) for double in get_doubles(eventlists, event_comparefunc, offset_instruments, thresholds, verbose = verbose))
			self.coincs = tuple(self.coincs)
			return self.coincs

		#
		# is this a head node, or some other node that magically
		# has only one component?  copy coincs from component
		#

		if len(self.components) == 1:
			if verbose:
				print >>sys.stderr, "\tgetting coincs from %s ..." % offset_vector_str(self.components[0].offset_vector)
			self.coincs = self.components[0].get_coincs(eventlists, event_comparefunc, thresholds, verbose = verbose)
			self.unused_coincs = self.components[0].unused_coincs

			#
			# done.  unlink the graph as we go to release
			# memory
			#

			self.components = None
			return self.coincs

		#
		# len(self.components) == 2 is impossible
		#

		assert len(self.components) > 2

		#
		# this is a regular node in the graph.  use coincidence
		# synthesis algorithm to populate its coincs
		#

		self.coincs = []
		self.unused_coincs = reduce(lambda a, b: a | b, (set(component.get_coincs(eventlists, event_comparefunc, thresholds, verbose = verbose)) for component in self.components))
		self.unused_coincs |= reduce(lambda a, b: a & b, (component.unused_coincs for component in self.components))

		if verbose:
			print >>sys.stderr, "\tassembling %s ..." % offset_vector_str(self.offset_vector)
		allcoincs0 = self.components[0].get_coincs(eventlists, event_comparefunc, thresholds, verbose = False)
		allcoincs1 = self.components[1].get_coincs(eventlists, event_comparefunc, thresholds, verbose = False)
		allcoincs2 = self.components[-1].get_coincs(eventlists, event_comparefunc, thresholds, verbose = False)
		length = len(allcoincs0)
		for n, coinc0 in enumerate(allcoincs0):
			if verbose and not (n % 200):
				print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / length),
			coincs1 = allcoincs1[bisect.bisect_left(allcoincs1, coinc0[:-1]):bisect.bisect_left(allcoincs1, coinc0[:-2] + (coinc0[-2] + 1,))]
			coincs2 = allcoincs2[bisect.bisect_left(allcoincs2, coinc0[1:]):bisect.bisect_left(allcoincs2, coinc0[1:-1] + (coinc0[-1] + 1,))]
			for coinc1 in coincs1:
				i = bisect.bisect_left(coincs2, coinc0[1:] + coinc1[-1:])
				if i < len(coincs2) and coincs2[i] == coinc0[1:] + coinc1[-1:]:
					new_coinc = coinc0[:1] + coincs2[i]
					self.unused_coincs -= set(iterutils.choices(new_coinc, len(new_coinc) - 1))
					self.coincs.append(new_coinc)
		if verbose:
			print >>sys.stderr, "\t100.0%"
		self.coincs.sort()
		self.coincs = tuple(self.coincs)

		#
		# done.  unlink the graph as we go to release memory
		#

		self.components = None
		return self.coincs


class TimeSlideGraph(object):
	def __init__(self, offset_vector_dict, verbose = False):
		if verbose:
			print >>sys.stderr, "constructing coincidence assembly graph for %d target offset vectors ..." % len(offset_vector_dict)

		#
		# populate the graph head nodes.  these represent the
		# target offset vectors requested by the calling code.
		#

		self.head = tuple(TimeSlideGraphNode(offset_vector, id) for id, offset_vector in sorted(offset_vector_dict.items()))

		#
		# populate the graph generations.  generations[n] is a
		# tuple of the nodes in the graph representing all unique
		# n-instrument offset vectors to be constructed as part of
		# the analysis (including normalized forms of any
		# n-instrument target offset vectors).
		#

		self.generations = {}
		n = max(len(offset_vector) for offset_vector in offset_vector_dict.values())
		self.generations[n] = tuple(TimeSlideGraphNode(offset_vector) for offset_vector in ligolw_tisi.time_slide_component_vectors((node.offset_vector for node in self.head if len(node.offset_vector) == n), n))
		for n in range(n, 2, -1):	# [n, n-1, ..., 3]
			offset_vectors = [node.offset_vector for node in self.head if len(node.offset_vector) == n]
			if n in self.generations:
				offset_vectors += [node.offset_vector for node in self.generations[n]]
			self.generations[n - 1] = tuple(TimeSlideGraphNode(offset_vector) for offset_vector in ligolw_tisi.time_slide_component_vectors(offset_vectors, n - 1))

		#
		# link each n-instrument node to the n-1 instrument nodes
		# from which it will be constructed.  NOTE:  the components
		# are sorted according to the alphabetically-sorted tuples
		# of instrument names involved in each component;  this is
		# a critical part of the coincidence synthesis algorithm
		#

		for node in self.head:
			node.components = tuple(sorted((component for component in self.generations[len(node.offset_vector)] if node.deltas == component.deltas), lambda a, b: cmp(sorted(a.offset_vector), sorted(b.offset_vector))))
		for n, nodes in self.generations.items():
			if n <= 2:
				continue
			for node in nodes:
				component_deltas = set(frozenset(ligolw_tisi.offset_vector_to_deltas(offset_vector).items()) for offset_vector in ligolw_tisi.time_slide_component_vectors([node.offset_vector], n - 1))
				node.components = tuple(sorted((component for component in self.generations[n - 1] if component.deltas in component_deltas), lambda a, b: cmp(sorted(a.offset_vector), sorted(b.offset_vector))))

		#
		# done
		#

		if verbose:
			print >>sys.stderr, "graph contains:"
			for n in sorted(self.generations):
				print >>sys.stderr,"\t%d %d-insrument offset vectors (%s)" % (len(self.generations[n]), n, ((n == 2) and "to be constructed directly" or "to be constructed indirectly"))
			print >>sys.stderr, "\t%d offset vectors total" % sum(len(self.generations[n]) for n in self.generations)


	def write(self, fileobj):
		"""
		Write a DOT graph representation of the time slide graph to
		fileobj.
		"""
		print >>fileobj, "digraph \"Time Slides\" {"
		for nodes in self.generations.values():
			for node in nodes:
				node_name = offset_vector_str(node.offset_vector, compact = True)
				print >>fileobj, "\t\"%s\" [shape=box];" % node_name
				if node.components is not None:
					for component in node.components:
						print >>fileobj, "\t\"%s\" -> \"%s\";" % (offset_vector_str(component.offset_vector, compact = True), node_name)
		for node in self.head:
			node_name = offset_vector_str(node.offset_vector, compact = True)
			print >>fileobj, "\t\"%s\" [shape=ellipse];" % node_name
			for component in node.components:
				print >>fileobj, "\t\"%s\" -> \"%s\";" % (offset_vector_str(component.offset_vector, compact = True), node_name)
		print >>fileobj, "}"


#
# =============================================================================
#
#                              Document Interface
#
# =============================================================================
#


class CoincTables(object):
	"""
	A convenience interface to the XML document's coincidence tables,
	allowing for easy addition of coincidence events.
	"""
	def __init__(self, xmldoc):
		# find the coinc table or create one if not found
		try:
			self.coinctable = table.get_table(xmldoc, lsctables.CoincTable.tableName)
		except ValueError:
			self.coinctable = lsctables.New(lsctables.CoincTable)
			xmldoc.childNodes[0].appendChild(self.coinctable)
		self.coinctable.sync_next_id()

		# find the coinc_map table or create one if not found
		try:
			self.coincmaptable = table.get_table(xmldoc, lsctables.CoincMapTable.tableName)
		except ValueError:
			self.coincmaptable = lsctables.New(lsctables.CoincMapTable)
			xmldoc.childNodes[0].appendChild(self.coincmaptable)

		# find the time_slide table
		self.time_slide_table = table.get_table(xmldoc, lsctables.TimeSlideTable.tableName)
		self.time_slide_index = self.time_slide_table.as_dict()

		# cast all offsets to LIGOTimeGPS for reversable arithmetic
		# FIXME:  I believe the arithmetic in the time slide graph
		# construction can be cleaned up so that this isn't
		# required.  when that is fixed, remove this
		self.time_slide_index = dict((time_slide_id, dict((instrument, LIGOTimeGPS(offset)) for instrument, offset in offset_vector.items())) for time_slide_id, offset_vector in self.time_slide_index.items())

	def append_coinc(self, process_id, time_slide_id, coinc_def_id, events):
		"""
		Takes a process ID, a time slide ID, and a list of events,
		and adds the events as a new coincidence to the coinc_event
		and coinc_map tables.
		"""
		# so we can iterate over it more than once incase we've
		# been given a generator expression.
		events = tuple(events)

		coinc = self.coinctable.RowType()
		coinc.process_id = process_id
		coinc.coinc_def_id = coinc_def_id
		coinc.coinc_event_id = self.coinctable.get_next_id()
		coinc.time_slide_id = time_slide_id
		coinc.set_instruments(None)
		coinc.nevents = len(events)
		coinc.likelihood = None
		self.coinctable.append(coinc)
		for event in events:
			coincmap = self.coincmaptable.RowType()
			coincmap.coinc_event_id = coinc.coinc_event_id
			coincmap.table_name = event.event_id.table_name
			coincmap.event_id = event.event_id
			self.coincmaptable.append(coincmap)
		return coinc


#
# =============================================================================
#
#                                Process Filter
#
# =============================================================================
#


def coincident_process_ids(xmldoc, offset_vectors, max_segment_gap, program):
	"""
	Take an XML document tree and determine the set of process IDs
	that will participate in coincidences identified by the time slide
	table therein.  It is OK for xmldoc to contain time slides
	involving instruments not represented in the list of processes:
	these time slides are ignored.  max_segment_gap is the largest gap
	(in seconds) that can exist between two segments and it still be
	possible for the two to provide events coincident with one another.
	"""
	# get the list of all process IDs for the given program
	proc_ids = table.get_table(xmldoc, lsctables.ProcessTable.tableName).get_ids_by_program(program)
	if not proc_ids:
		# hmm... that program's output is not in this file.
		raise KeyError, program

	# extract a segmentlistdict
	search_summ_table = table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName)
	seglistdict = search_summ_table.get_out_segmentlistdict(proc_ids).coalesce()

	# fast path:  if the largest gap anywhere in the lists is smaller
	# than max_segment_gap then all process_ids participate.  NOTE:
	# this also handles the case of max_segment_gap being passed in as
	# float("inf") (which would otherwise break the LIGOTimeGPS
	# arithmetic).
	#
	# this is checking the gaps between segments in the *same
	# instrument*.  this assumption is that if max_segment_gap can
	# close all the gaps within each instrument's segment list then,
	# finally, all processes will be found to be required for the
	# coincidence analysis.  this is not really true, but it's safe.
	if len([segs for segs in seglistdict.values() if len(segs) == 1]):
		return proc_ids
	elif max(b[0] - a[1] for segs in seglistdict.values() for a, b in zip(segs[:-1], segs[1:])) <= max_segment_gap:
		return proc_ids

	# protract by half the largest coincidence window so as to not miss
	# edge effects
	seglistdict.protract(max_segment_gap / 2)

	# determine what time slides are possible given the instruments in
	# the search summary table
	avail_instruments = set(seglistdict)
	offset_vectors = [offset_vector for offset_vector in offset_vectors if set(offset_vector).issubset(avail_instruments)]

	# determine the coincident segments for each instrument
	seglistdict = llwapp.get_coincident_segmentlistdict(seglistdict, offset_vectors)

	# find the IDs of the processes that contributed to the coincident
	# segments
	coinc_proc_ids = set()
	for row in search_summ_table:
		if row.process_id in proc_ids and row.process_id not in coinc_proc_ids and seglistdict.intersection(row.get_ifos()).intersects_segment(row.get_out()):
			coinc_proc_ids.add(row.process_id)
	return coinc_proc_ids


#
# =============================================================================
#
#                             Event List Interface
#
# =============================================================================
#


class EventList(list):
	"""
	A parent class for managing a list of events:  applying time
	offsets, and retrieving subsets of the list selected by time
	interval.  To be useful, this class must be subclassed with
	overrides provided for certain methods.  The only methods that
	*must* be overridden in a subclass are the _add_offset() and
	get_coincs() methods.  The make_index() method can be overridden if
	needed.  None of the other methods inherited from the list parent
	class need to be overridden, indeed they probably should not be
	unless you know what you're doing.
	"""
	def __init__(self, instrument):
		self.offset = lsctables.LIGOTimeGPS(0)
		self.instrument = instrument

	def make_index(self):
		"""
		Provided to allow for search-specific look-up tables or
		other indexes to be constructed for use in increasing the
		speed of the get_coincs() method.  This will be called
		after all events have been added to the list, and again if
		the list is ever modified, and before get_coincs() is ever
		called.
		"""
		pass

	def _add_offset(self, delta):
		"""
		Add an amount to the time of each event.  This method is
		used internally by the set_offset() method, and is provided
		to simplify the construction of search-specific subclasses.
		Typically, the _add_offset() method will be overridden with
		a search-specific implementation, while the set_offset()
		method is not modified (which does some additional book
		keeping, such as updating the offset attribute).
		"""
		raise NotImplementedError

	def set_offset(self, offset):
		"""
		Set an offset on the times of all events in the list.
		"""
		# cast offset to LIGOTimeGPS to avoid repeated conversion
		# when applying the offset to each event.  also prevents
		# round-off errors since addition and subtraction of
		# LIGOTimeGPS objects are exact, so by recording the offset
		# as a LIGOTimeGPS it should be possible to return the
		# events to exactly their original times before exiting.
		offset = lsctables.LIGOTimeGPS(offset)

		# check for no-op
		if offset != self.offset:
			# apply the delta
			self._add_offset(offset - self.offset)

			# record the new offset
			self.offset = offset

	def get_coincs(self, event_a, light_travel_time, threshold, comparefunc):
		"""
		Return a list of the events from this list that are
		coincident with event_a.

		The threshold argument will be the thresholds appropriate
		for "instrument_a, instrument_b", in that order, where
		instrument_a is the instrument for event_a, and
		instrument_b is the instrument for the events in this
		EventList.  Because it is frequently needed by
		implementations of this method, the distance in light
		seconds between the two instruments is provided as the
		light_travel_time parameter.
		"""
		raise NotImplementedError


class EventListDict(dict):
	"""
	A dictionary of EventList objects, indexed by instrument,
	initialized from an XML trigger table and a list of process IDs
	whose events should be included.
	"""
	def __new__(self, *args, **kwargs):
		# wrapper to shield dict.__new__() from our arguments.
		return dict.__new__(self)

	def __init__(self, EventListType, event_table, process_ids = None):
		"""
		Initialize a newly-created instance.  EventListType is a
		subclass of EventList (the subclass itself, not an instance
		of the subclass).  event_table is a list of events (e.g.,
		an instance of a glue.ligolw.table.Table subclass).  If the
		optional process_ids arguments is not None, then it is
		assumed to be a list or set or other thing implementing the
		"in" operator which is used to define the set of
		process_ids whose events should be considered in the
		coincidence analysis, otherwise all events are considered.
		"""
		for event in event_table:
			if (process_ids is None) or (event.process_id in process_ids):
				if event.ifo not in self:
					self[event.ifo] = EventListType(event.ifo)
				self[event.ifo].append(event)
		for l in self.values():
			l.make_index()

	def set_offsetdict(self, offsetdict):
		"""
		Set the event list offsets to those in the dictionary of
		instrument/offset pairs.  Instruments not in offsetdict are
		not modified.  KeyError is raised if the dictionary of
		instrument/offset pairs contains a key (instrument) that
		this dictionary does not.
		"""
		for instrument, offset in offsetdict.items():
			self[instrument].set_offset(offset)

	def remove_offsetdict(self):
		"""
		Remove the offsets from all event lists (reset them to 0).
		"""
		for l in self.values():
			l.set_offset(0)


def make_eventlists(xmldoc, EventListType, event_table_name, process_ids = None):
	"""
	Convenience wrapper for constructing a dictionary of event lists
	from an XML document tree, the name of a table from which to get
	the events, a maximum allowed time window, and the name of the
	program that generated the events.
	"""
	return EventListDict(EventListType, table.get_table(xmldoc, event_table_name), process_ids = process_ids)


#
# =============================================================================
#
#                          N-way Coincidence Iterator
#
# =============================================================================
#


def get_doubles(eventlists, comparefunc, instruments, thresholds, verbose = False):
	"""
	Given an instance of an EventListDict, an event comparison
	function, an iterable (e.g., a list) of instruments, and a
	dictionary mapping instrument pair to threshold data, generate a
	sequence of tuples of mutually coincident events.

	The signature of the comparison function should be

	>>> comparefunc(event1, event2, light_travel_time, threshold_data)

	where event1 and event2 are two objects drawn from the event lists
	(from different instruments), light_travel_time is the distance
	between the instruments in light seconds, and threshold_data is the
	value contained in the thresholds dictionary for the pair of
	instruments from which event1 and event2 have been drawn.  The
	return value should be 0 (False) if the events are coincident, and
	non-zero otherwise (the behaviour of the comparison function is
	like a subtraction operator, returning 0 when the two events are
	"the same").

	The thresholds dictionary should look like

	>>> {("H1", "L1"): 10.0, ("L1", "H1"): -10.0}

	i.e., the keys are tuples of instrument pairs and the values
	specify the "threshold" for that instrument pair.  The threshold
	itself is arbitrary.  Here simple floats are used as an example,
	but any Python object can be provided and will be passed to the
	comparefunc().  Note that it is assumed that order matters in the
	comparison function and so the thresholds dictionary must provide a
	threshold for the instruments in both orders.

	Each tuple returned by this generator will contain exactly one
	event from each of the instruments in the instrument list.

	NOTE:  the instruments iterator must contain exactly two
	instruments.
	"""
	# retrieve the event lists for the requested instrument combination

	instruments = tuple(instruments)
	assert len(instruments) == 2
	for instrument in instruments:
		assert eventlists[instrument].instrument == instrument
	eventlists = [eventlists[instrument] for instrument in instruments]

	# determine the shorter and longer of the two event lists;  record
	# the list of the shortest

	if len(eventlists[0]) <= len(eventlists[1]):
		eventlista, eventlistb = eventlists
	else:
		eventlistb, eventlista = eventlists
	length = len(eventlista)

	# extract the thresholds and pre-compute the light travel time

	try:
		thresholds = thresholds[(eventlista.instrument, eventlistb.instrument)]
	except KeyError, e:
		raise KeyError, "no coincidence thresholds provided for instrument pair %s, %s" % str(e)
	light_travel_time = inject.light_travel_time(eventlista.instrument, eventlistb.instrument)

	# for each event in the shortest list

	for n, eventa in enumerate(eventlista):
		if verbose and not (n % 2000):
			print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / length),

		# iterate over events from the other list that are
		# coincident with the event.

		for eventb in eventlistb.get_coincs(eventa, light_travel_time, thresholds, comparefunc):
			yield (eventa, eventb)
	if verbose:
		print >>sys.stderr, "\t100.0%"

	# done

# Copyright (C) 2006--2010  Kipp Cannon, Drew G. Keppel, Jolien Creighton
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
import itertools
import math
import numpy
import random
try:
	from scipy.constants import c as speed_of_light
except ImportError:
	from pylal.lalconstants import LAL_C_SI as speed_of_light
	speed_of_light = float(speed_of_light)
import scipy.optimize
import sys


from glue import iterutils
from glue.ligolw import table
from glue.ligolw import lsctables
from pylal import git_version
from pylal import inject
from pylal import ligolw_tisi


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


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
		# the offset that should be added to the times of events in
		# this list when comparing to the times of other events.
		# used to implement time-shifted coincidence tests
		self.offset = lsctables.LIGOTimeGPS(0)

		# the name of the instrument from which the events in this
		# list have been taken
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

	def set_offset(self, offset):
		"""
		Set an offset on the times of all events in the list.
		"""
		# cast offset to LIGOTimeGPS to avoid repeated conversion
		# when applying the offset to each event.
		self.offset = lsctables.LIGOTimeGPS(offset)

	def get_coincs(self, event_a, offset_a, light_travel_time, threshold, comparefunc):
		"""
		Return a list of the events from this list that are
		coincident with event_a.

		offset_a is the time shift to be added to the time of
		event_a before comparing to the times of events in this
		list.  The offset attribute of this object will contain the
		time shift to be added to the times of the events in this
		list before comparing to event_a.  That is, the times of
		arrival of the events in this list should have (self.offset
		- offset_a) added to them before comparing to the time of
		arrival of event_a.  Or, equivalently, the time of arrival
		of event_a should have (offset_a - self.offset) added to it
		before comparing to the times of arrival of the events in
		this list.  This behaviour is to support the construction
		of time shifted coincidences.

		Because it is frequently needed by implementations of this
		method, the distance in light seconds between the two
		instruments is provided as the light_travel_time parameter.

		The threshold argument will be the thresholds appropriate
		for "instrument_a, instrument_b", in that order, where
		instrument_a is the instrument for event_a, and
		instrument_b is the instrument for the events in this
		EventList.

		comparefunc is the function to use to compare events in
		this list to event_a.
		"""
		raise NotImplementedError


class EventListDict(dict):
	"""
	A dictionary of EventList objects, indexed by instrument,
	initialized from an XML trigger table and a list of process IDs
	whose events should be included.
	"""
	def __new__(cls, *args, **kwargs):
		# wrapper to shield dict.__new__() from our arguments.
		return dict.__new__(cls)

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
				# FIXME:  only works when the instrument
				# name is in the "ifo" column.  true for
				# inspirals, bursts and ringdowns
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
#                         Double Coincidence Iterator
#
# =============================================================================
#


def get_doubles(eventlists, comparefunc, instruments, thresholds, verbose = False):
	"""
	Given an instance of an EventListDict, an event comparison
	function, an iterable (e.g., a list) of instruments, and a
	dictionary mapping instrument pair to threshold data for use by the
	event comparison function, generate a sequence of tuples of
	mutually coincident events.

	The signature of the comparison function should be

	>>> comparefunc(event1, offset1, event2, offset2, light_travel_time, threshold_data)

	where event1 and event2 are two objects drawn from the event lists
	(of different instruments), offset1 and offset2 are the time shifts
	that should be added to the arrival times of event1 and event2
	respectively, light_travel_time is the distance in light seconds
	between the instruments from which event1 and event2 have been
	drawn, and threshold_data is the value contained in the thresholds
	dictionary for that pair of instruments.  The return value should
	be 0 (False) if the events are coincident, and non-zero otherwise
	(the behaviour of the comparison function is like a subtraction
	operator, returning 0 when the two events are "the same").

	The thresholds dictionary should look like

	>>> {("H1", "L1"): 10.0, ("L1", "H1"): -10.0}

	i.e., the keys are tuples of instrument pairs and the values
	specify the "threshold data" for that instrument pair.  The
	threshold data itself is an arbitrary Python object.  Floats are
	shown in the example above, but any Python object can be provided
	and will be passed to the comparefunc().  Note that it is assumed
	that order matters in the comparison function and so the thresholds
	dictionary must provide a threshold for the instruments in both
	orders.

	Each tuple returned by this generator will contain exactly two
	events, one from each of the two instruments in the instruments
	sequence.

	NOTE:  the instruments sequence must contain exactly two
	instruments.

	NOTE:  the order of the events in each tuple returned by this
	function is arbitrary, in particular it does not necessarily match
	the order of the instruments sequence.
	"""
	# retrieve the event lists for the requested instrument combination

	instruments = tuple(instruments)
	assert len(instruments) == 2
	for instrument in instruments:
		assert eventlists[instrument].instrument == instrument
	eventlista, eventlistb = [eventlists[instrument] for instrument in instruments]

	# insure eventlist a is the shorter of the two event lists;  record
	# the length of the shortest

	if len(eventlista) > len(eventlistb):
		eventlista, eventlistb = eventlistb, eventlista
	length = len(eventlista)

	# extract the thresholds and pre-compute the light travel time

	try:
		threshold_data = thresholds[(eventlista.instrument, eventlistb.instrument)]
	except KeyError, e:
		raise KeyError, "no coincidence thresholds provided for instrument pair %s, %s" % e.args[0]
	light_travel_time = inject.light_travel_time(eventlista.instrument, eventlistb.instrument)

	# for each event in the shortest list

	for n, eventa in enumerate(eventlista):
		if verbose and not (n % 2000):
			print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / length),

		# iterate over events from the other list that are
		# coincident with the event, and return the pairs

		for eventb in eventlistb.get_coincs(eventa, eventlista.offset, light_travel_time, threshold_data, comparefunc):
			yield (eventa, eventb)
	if verbose:
		print >>sys.stderr, "\t100.0%"

	# done


#
# =============================================================================
#
#                               Time Slide Graph
#
# =============================================================================
#


class TimeSlideGraphNode(object):
	def __init__(self, offset_vector, time_slide_id = None):
		self.time_slide_id = time_slide_id
		self.offset_vector = offset_vector
		self.deltas = frozenset(offset_vector.deltas.items())
		self.components = None
		self.coincs = None
		self.unused_coincs = set()

	def name(self):
		return self.offset_vector.__str__(compact = True)

	def get_coincs(self, eventlists, event_comparefunc, thresholds, verbose = False):
		#
		# has this node already been visited?  if so, return the
		# answer we already know
		#

		if self.coincs is not None:
			if verbose:
				print >>sys.stderr, "\treusing %s" % str(self.offset_vector)
			return self.coincs

		#
		# is this a leaf node?  construct the coincs explicitly
		#

		if self.components is None:
			if verbose:
				print >>sys.stderr, "\tconstructing %s ..." % str(self.offset_vector)
			#
			# can we do it?
			#

			assert len(self.offset_vector) == 2
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
			# search for and record coincidences.  coincs is a
			# sorted tuple of event ID pairs, where each pair
			# of IDs is, itself, ordered alphabetically by
			# instrument name
			#

			if verbose:
				print >>sys.stderr, "\tsearching ..."
			# FIXME:  assumes the instrument column is named
			# "ifo".  works for inspirals, bursts, and
			# ring-downs.  note that the event order in each
			# tuple returned by get_doubles() is arbitrary so
			# we need to sort each tuple by instrument name
			# explicitly
			self.coincs = tuple(sorted((a.event_id, b.event_id) if a.ifo <= b.ifo else (b.event_id, a.event_id) for (a, b) in get_doubles(eventlists, event_comparefunc, offset_instruments, thresholds, verbose = verbose)))
			return self.coincs

		#
		# is this a head node, or some other node that magically
		# has only one component?  copy coincs from component
		#

		if len(self.components) == 1:
			if verbose:
				print >>sys.stderr, "\tgetting coincs from %s ..." % str(self.components[0].offset_vector)
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

		# all coincs with n-1 instruments from the component time
		# slides are potentially unused.  they all go in, we'll
		# remove things from this set as we use them
		# NOTE:  this function call is the recursion into the
		# components to ensure they are initialized, it must be
		# executed before any of what follows
		for component in self.components:
			self.unused_coincs |= set(component.get_coincs(eventlists, event_comparefunc, thresholds, verbose = verbose))
		# of the (< n-1)-instrument coincs that were not used in
		# forming the (n-1)-instrument coincs, any that remained
		# unused after forming two compontents cannot have been
		# used by any other components, they definitely won't be
		# used to construct our n-instrument coincs, and so they go
		# into our unused pile
		for componenta, componentb in iterutils.choices(self.components, 2):
			self.unused_coincs |= componenta.unused_coincs & componentb.unused_coincs

		if verbose:
			print >>sys.stderr, "\tassembling %s ..." % str(self.offset_vector)
		# magic:  we can form all n-instrument coincs by knowing
		# just three sets of the (n-1)-instrument coincs no matter
		# what n is (n > 2).  note that we pass verbose=False
		# because we've already called the .get_coincs() methods
		# above, these are no-ops to retrieve the answers again
		allcoincs0 = self.components[0].get_coincs(eventlists, event_comparefunc, thresholds, verbose = False)
		allcoincs1 = self.components[1].get_coincs(eventlists, event_comparefunc, thresholds, verbose = False)
		allcoincs2 = self.components[-1].get_coincs(eventlists, event_comparefunc, thresholds, verbose = False)
		# for each coinc in list 0
		length = len(allcoincs0)
		for n, coinc0 in enumerate(allcoincs0):
			if verbose and not (n % 200):
				print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / length),
			# find all the coincs in list 1 whose first (n-2)
			# event IDs are the same as the first (n-2) event
			# IDs in coinc0.  note that they are guaranteed to
			# be arranged together in the list of coincs and
			# can be identified with two bisection searches
			# note:  cannot use bisect_right() because we're
			# only comparing against the first (n-2) of (n-1)
			# things in each tuple, we need to use bisect_left
			# after incrementing the last of the (n-2) things
			# by one to obtain the correct range of indexes
			coincs1 = allcoincs1[bisect.bisect_left(allcoincs1, coinc0[:-1]):bisect.bisect_left(allcoincs1, coinc0[:-2] + (coinc0[-2] + 1,))]
			# find all the coincs in list 2 whose first (n-2)
			# event IDs are the same as the last (n-2) event
			# IDs in coinc0.  note that they are guaranteed to
			# be arranged together in the list and can be
			# identified with two bisection searches
			coincs2 = allcoincs2[bisect.bisect_left(allcoincs2, coinc0[1:]):bisect.bisect_left(allcoincs2, coinc0[1:-1] + (coinc0[-1] + 1,))]
			# for each coinc extracted from list 1 above search
			# for a coinc extracted from list 2 above whose
			# first (n-2) event IDs are the last (n-2) event
			# IDs in coinc 0 and whose last event ID is the
			# last event ID in coinc 1.  when found, the first
			# ID from coinc 0 prepended to the (n-1) coinc IDs
			# from coinc 2 forms an n-instrument coinc.  how
			# this works is as follows:  coinc 0 and coinc 1,
			# both (n-1)-instrument coincs, together identify a
			# unique potential n-instrument coinc.  coinc 2's
			# role is to confirm the coincidence by showing
			# that the event from the instrument in coinc 1
			# that isn't found in coinc 0 is coincident with
			# all the other events that are in coinc 1.  if the
			# coincidence holds then that combination of event
			# IDs must be found in the coincs2 list, because we
			# assume the coincs2 list is complete  the
			# bisection search above to extract the coincs2
			# list could be skipped, but by starting with a
			# shorter list the bisection searches inside the
			# following loop are faster.
			for coinc1 in coincs1:
				i = bisect.bisect_left(coincs2, coinc0[1:] + coinc1[-1:])
				if i < len(coincs2) and coincs2[i] == coinc0[1:] + coinc1[-1:]:
					new_coinc = coinc0[:1] + coincs2[i]
					# break the new coinc into
					# (n-1)-instrument components and
					# remove them from the unused list
					# because we just used them, then
					# record the coinc and move on
					self.unused_coincs -= set(iterutils.choices(new_coinc, len(new_coinc) - 1))
					self.coincs.append(new_coinc)
		if verbose:
			print >>sys.stderr, "\t100.0%"
		# sort the coincs we just constructed by the component
		# event IDs and convert to a tuple for speed
		self.coincs.sort()
		self.coincs = tuple(self.coincs)

		#
		# done.  we won't be back here again so unlink the graph as
		# we go to release memory
		#

		self.components = None
		return self.coincs


class TimeSlideGraph(object):
	def __init__(self, offset_vector_dict, verbose = False):
		#
		# validate input
		#

		if min(len(offset_vector) for offset_vector in offset_vector_dict.values()) < 2:
			raise ValueError("offset vectors must have at least two instruments")

		#
		# populate the graph head nodes.  these represent the
		# target offset vectors requested by the calling code.
		#

		if verbose:
			print >>sys.stderr, "constructing coincidence assembly graph for %d target offset vectors ..." % len(offset_vector_dict)
		self.head = tuple(TimeSlideGraphNode(offset_vector, time_slide_id) for time_slide_id, offset_vector in sorted(offset_vector_dict.items()))

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
			#
			# collect all offset vectors of length n that we
			# need to be able to construct
			#

			offset_vectors = [node.offset_vector for node in self.head if len(node.offset_vector) == n] + [node.offset_vector for node in self.generations[n]]

			#
			# determine the smallest set of offset vectors of
			# length n-1 required to construct the length-n
			# offset vectors, build a graph node for each of
			# the vectors of length n-1, and record the nodes
			# as the n-1'st generation
			#

			self.generations[n - 1] = tuple(TimeSlideGraphNode(offset_vector) for offset_vector in ligolw_tisi.time_slide_component_vectors(offset_vectors, n - 1))

		#
		# link each n-instrument node to the n-1 instrument nodes
		# from which it will be constructed.  NOTE:  the components
		# are sorted according to the alphabetically-sorted tuples
		# of instrument names involved in each component;  this is
		# a critical part of the coincidence synthesis algorithm
		#

		for node in self.head:
			#
			# the offset vector of a head node should be found
			# directly in its generation, and it must be unique
			# or there's a bug above.  despite this, we still
			# go to the trouble of sorting to make it easy to
			# keep this code in sync with the code for other
			# graph nodes below, but the assert makes sure the
			# result contains just one entry
			#

			node.components = tuple(sorted((component for component in self.generations[len(node.offset_vector)] if node.deltas == component.deltas), key = lambda x: sorted(x.offset_vector)))
			assert len(node.components) == 1

		for n, nodes in self.generations.items():
			assert n >= 2	# failure indicates bug in code that constructed generations
			if n == 2:
				# leaf nodes have no components
				continue
			for node in nodes:
				component_deltas = set(frozenset(offset_vector.deltas.items()) for offset_vector in ligolw_tisi.time_slide_component_vectors([node.offset_vector], n - 1))
				node.components = tuple(sorted((component for component in self.generations[n - 1] if component.deltas in component_deltas), key = lambda x: sorted(x.offset_vector)))

		#
		# done
		#

		if verbose:
			print >>sys.stderr, "graph contains:"
			for n in sorted(self.generations):
				print >>sys.stderr,"\t%d %d-insrument offset vectors (%s)" % (len(self.generations[n]), n, ((n == 2) and "to be constructed directly" or "to be constructed indirectly"))
			print >>sys.stderr, "\t%d offset vectors total" % sum(len(self.generations[n]) for n in self.generations)


	def get_coincs(self, eventlists, event_comparefunc, thresholds, include_small_coincs = True, verbose = False):
		if verbose:
			print >>sys.stderr, "constructing coincs for target offset vectors ..."
		for n, node in enumerate(self.head):
			if verbose:
				print >>sys.stderr, "%d/%d: %s" % (n + 1, len(self.head), str(node.offset_vector))
			if include_small_coincs:
				# note that unused_coincs must be retrieved
				# after the call to .get_coincs() because
				# the former is computed as a side effect
				# of the latter
				iterator = itertools.chain(node.get_coincs(eventlists, event_comparefunc, thresholds, verbose), node.unused_coincs)
			else:
				iterator = node.get_coincs(eventlists, event_comparefunc, thresholds, verbose)
			for coinc in iterator:
				yield node, coinc


	def write(self, fileobj):
		"""
		Write a DOT graph representation of the time slide graph to
		fileobj.
		"""
		print >>fileobj, "digraph \"Time Slides\" {"
		for node in itertools.chain(*self.generations.values()):
			print >>fileobj, "\t\"%s\" [shape=box];" % node.name()
			if node.components is not None:
				for component in node.components:
					print >>fileobj, "\t\"%s\" -> \"%s\";" % (component.name(), node.name())
		for node in self.head:
			print >>fileobj, "\t\"%s\" [shape=ellipse];" % node.name()
			for component in node.components:
				print >>fileobj, "\t\"%s\" -> \"%s\";" % (component.name(), node.name())
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
		self.time_slide_index = dict((time_slide_id, type(offset_vector)((instrument, lsctables.LIGOTimeGPS(offset)) for instrument, offset in offset_vector.items())) for time_slide_id, offset_vector in self.time_slide_index.items())

	def append_coinc(self, process_id, time_slide_id, coinc_def_id, events):
		"""
		Takes a process ID, a time slide ID, and a list of events,
		and adds the events as a new coincidence to the coinc_event
		and coinc_map tables.

		Subclasses that wish to override this method should first
		chain to this method to construct and initialize the
		coinc_event and coinc_event_map rows.  When subclassing
		this method, if the time shifts that were applied to the
		events in constructing the coincidence are required to
		compute additional metadata, they can be retrieved from
		self.time_slide_index using the time_slide_id.
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
#                       Time-slideless Coinc Synthesizer
#
# =============================================================================
#


def slideless_coinc_generator_mu_tau(eventlists, segmentlists, delta_t):
	"""
	Compute the mean event rates in Hz and the maximum allowed \Delta t
	windows between pairs of instruments.

	eventlists is a dictionary mapping instrument name to a list of
	"events" (arbitrary objects).  segmentlists is a
	glue.segments.segmentlistdict object describing the observation
	segments for each of instruments.  delta_t is a time window in
	seconds, the light travel time between instrument pairs is added to
	this internally.

	The return value is a tuple of dictionaries.  The first, "mu", maps
	instrument name to mean event rate in Hz.  The second, "tau", maps
	pairs of instrument names (stored as frozensets) to the maximum
	allowed time difference between events from those instruments in
	order for them to be coincident.
	"""
	mu = dict((instrument, len(eventlist) / float(abs(segmentlists[instrument]))) for (instrument, eventlist) in eventlists.items())
	tau = dict((frozenset([a, b]), delta_t + inject.light_travel_time(a, b)) for (a, b) in iterutils.choices(tuple(eventlists), 2))
	return mu, tau


def slideless_coinc_generator_plausible_toas(instruments, tau):
	"""
	Construct and return a generator that yields dictionaries of random
	event time-of-arrivals for the instruments in instruments such that
	the time-of-arrivals are mutually coincident given the maximum
	allowed inter-instrument \Delta ts given by tau.

	Example:

	>>> tau = {frozenset(['V1', 'H1']): 0.028287979933844225, frozenset(['H1', 'L1']): 0.011012846152223924, frozenset(['V1', 'L1']): 0.027448341016726496}
	>>> instruments = set(("H1", "L1", "V1"))
	>>> toas = slideless_coinc_generator_plausible_toas(instruments, tau)
	>>> toas.next()
	>>> toas.next()
	"""
	# this algorithm is documented in slideless_coinc_generator_rates()
	instruments = tuple(instruments)
	anchor, instruments = instruments[0], instruments[1:]
	windows = tuple((-tau[frozenset((anchor, instrument))], +tau[frozenset((anchor, instrument))]) for instrument in instruments)
	ijseq = tuple((i, j, tau[frozenset((instruments[i], instruments[j]))]) for (i, j) in iterutils.choices(range(len(instruments)), 2))
	while True:
		dt = tuple(random.uniform(*window) for window in windows)
		if all(abs(dt[i] - dt[j]) <= maxdt for i, j, maxdt in ijseq):
			yield dict([(anchor, 0.0)] + zip(instruments, dt))


def slideless_coinc_generator_rates(mu, tau, verbose = False, abundance_rel_accuracy = 1e-4):
	"""
	From the mean event rates for N instruments and the maximum allowed
	coincidence windows between pairs of those instruments, compute and
	return the mean event rates for coincidences for all combinations of
	instruments from 2 to N inclusively.

	The mu and tau input parameters are the return values of
	slideless_coinc_generator_mu_tau().  If verbose is True then
	diagnostic information is printed to stderr.

	abundance_rel_accuracy sets the fractional error tolerated in the
	Monte Carlo integrator used to estimate the relative abundances of
	the different kinds of coincs.
	"""
	all_instruments = tuple(mu)
	mu_coinc = {}
	for n in range(len(all_instruments), 1, -1):
		for instruments in iterutils.choices(all_instruments, n):
			# choose the instrument whose TOA forms the "epoch"
			# of the coinc.  to improve the convergence rate
			# this should be the instrument with the smallest
			# coincidence windows
			key = frozenset(instruments)
			anchor = min((tau[frozenset(ab)], ab[0]) for ab in iterutils.choices(instruments, 2))[1]
			instruments = tuple(key - set([anchor]))
			# compute \mu_{1} * \mu_{2} ... \mu_{N} * 2 *
			# \tau_{12} * 2 * \tau_{13} ... 2 * \tau_{1N}.
			# this is the rate at which events from instrument
			# 1 are coincident with events from all of
			# instruments 2...N.  later, we will multiply this
			# by the probability that events from instruments
			# 2...N known to be coincident with an event from
			# instrument 1 are themselves mutually coincident
			rate = mu[anchor]
			for instrument in instruments:
				# the factor of 2 is because to be
				# coincident the time difference can be
				# anywhere in [-tau, +tau], so the size of
				# the coincidence window is 2 tau
				rate *= mu[instrument] * 2 * tau[frozenset((anchor, instrument))]
			if verbose:
				print >>sys.stderr, "%s uncorrected mean event rate = %g Hz" % (",".join(sorted(key)), rate)

			# if there are more than two instruments, correct
			# for the probability of full N-way coincidence by
			# computing the volume of the allowed parameter
			# space by stone throwing.  FIXME:  it might be
			# practical to solve this with some sort of
			# computational geometry library and convex hull
			# volume calculator.
			if len(instruments) > 1:
				# for each instrument 2...N, the interval
				# within which an event is coincident with
				# instrument 1
				windows = tuple((-tau[frozenset((anchor, instrument))], +tau[frozenset((anchor, instrument))]) for instrument in instruments)
				# pre-assemble a sequence of instrument
				# index pairs and the maximum allowed
				# \Delta t between them to avoid doing the
				# work associated with assembling the
				# sequence inside a loop
				ijseq = tuple((i, j, tau[frozenset((instruments[i], instruments[j]))]) for (i, j) in iterutils.choices(range(len(instruments)), 2))
				# compute the numerator and denominator of
				# the fraction of events coincident with
				# the anchor instrument that are also
				# mutually coincident.  this is done by
				# picking a vector of allowed \Delta ts and
				# testing them against the coincidence
				# windows.  the loop's exit criterion is
				# arrived at as follows.  the binomial
				# distribution's variance is d p (1 - p)
				# where d is the number of trials and p is
				# the probability of a successful outcome,
				# which we replace here with p=n/d.  we
				# quit when \sqrt{d p (1 - p)} / n <= rel
				# accuracy.  by connecting the bailout
				# condition to the results of the loop we
				# bias the final answer (e.g., we get lucky
				# and increment n on the first trial).  to
				# minimize the effect of this we require at
				# least 1 / rel accuracy iterations before
				# considering the binomial criterion.  note
				# that if the true probability is 0 or 1,
				# so that n=0 or n=d identically then the
				# loop will never terminate;  from the
				# nature of the problem we know 0<p<1 so
				# the loop will, eventually, terminate
				n, d = 0, 0
				while abundance_rel_accuracy * d < 1.0 or n < d / (1.0 + abundance_rel_accuracy**2 * d):
					dt = tuple(random.uniform(*window) for window in windows)
					if all(abs(dt[i] - dt[j]) <= maxdt for i, j, maxdt in ijseq):
						n += 1
					d += 1

				rate *= float(n) / float(d)
				if verbose:
					print >>sys.stderr, "	multi-instrument correction factor = %g" % (float(n)/float(d))
					print >>sys.stderr, "	%s mean event rate = %g Hz" % (",".join(sorted(key)), rate)

			# subtract from the rate the rate at which this
			# combination of instruments is found in
			# higher-order coincs
			all_other_instruments = tuple(set(all_instruments) - key)
			for m in range(1, len(all_other_instruments) + 1):
				for otherinstruments in iterutils.choices(all_other_instruments, m):
					rate -= mu_coinc[key | set(otherinstruments)]

			# done
			assert rate >= 0
			mu_coinc[key] = rate
			if verbose:
				print >>sys.stderr, "%s mean event rate = %g Hz" % (",".join(sorted(key)), rate)

	# done
	return mu_coinc


def slideless_coinc_generator(eventlists, mu_coinc, tau, timefunc, allow_zero_lag = False, verbose = False):
	"""
	Generator function to return time shifted coincident event tuples
	without the use of explicit time shift vectors.

	eventlists is a dictionary mapping instrument name to a list of
	"events" (arbitrary objects).  mu_coinc is a dictionary mapping
	frozensets of instrument names to mean event rates in Hz;  this
	diciontary can be generated using
	slideless_coinc_generator_rates().  tau is a dictionary mappin
	pairs of instrument names (stored as frozensets) to the maximum
	allowed time difference between events from those instruments in
	order for them to be coincident;  this dictionary can be generated
	with slideless_coinc_generator_mu_tau().  timefunc is a function
	for computing the "time" of an event, its signature should be

		t = timefunc(event)

	If allow_zero_lag is False (the default), then only event tuples
	with no genuine zero-lag coincidences are returned, that is only
	tuples in which no event pairs would be considered to be coincident
	without time shifts applied.


	Example:

	>>> eventlists = {"H1": [0, 1, 2, 3], "L1": [10, 11, 12, 13], "V1": [20, 21, 22, 23]}
	>>> segmentlists = {"H1": 30, "L1": 40, "V1": 50}
	>>> mu, tau = slideless_coinc_generator_mu_tau(eventlists, segmentlists, 0.001)
	>>> mu
	{'V1': 0.080000000000000002, 'H1': 0.13333333333333333, 'L1': 0.10000000000000001}
	>>> tau
	{frozenset(['V1', 'H1']): 0.028287979933844225, frozenset(['H1', 'L1']): 0.011012846152223924, frozenset(['V1', 'L1']): 0.027448341016726496}
	>>> mu_coinc = slideless_coinc_generator_rates(mu, tau)
	>>> mu_coinc
	{frozenset(['V1', 'H1']): 0.00060229803796414379, frozenset(['V1', 'H1', 'L1']): 1.1788672911996494e-06, frozenset(['H1', 'L1']): 0.00029249703010143834, frozenset(['V1', 'L1']): 0.00043799458897642431}
	>>> coincs = slideless_coinc_generator(eventlists, mu_coinc, tau, (lambda x: 0), allow_zero_lag = True)
	>>> coincs.next()	# returns a tuple of events
	"""
	#
	# from the rates compute the relative abundances
	#

	assert all(len(key) > 1 for key in mu_coinc.keys())	# check for single-instrument rates
	P = dict((key, value / sum(sorted(mu_coinc.values()))) for key, value in mu_coinc.items())
	if verbose:
		for key, value in P.items():
			print >>sys.stderr, "%s relative abundance = %g" % (",".join(sorted(key)), value)

	#
	# convert to a sorted tuple of (probability mass, instrument combo)
	# pairs.  while at it, convert the instrument sets to tuples to
	# avoid doing this in a loop later, and remove instrument combos
	# whose probability mass is 0.  if no combos remain then we can't
	# form coincidences
	#

	P = tuple(sorted([mass, tuple(instruments)] for instruments, mass in P.items() if mass != 0))
	if not P:
		return

	#
	# replace the probability masses with cummulative probabilities
	#

	for i in range(1, len(P)):
		P[i][0] += P[i - 1][0]

	#
	# normalize (should be already, just be certain)
	#

	for i in range(len(P)):
		P[i][0] /= P[-1][0]
	assert P[-1][0] == 1.0
	if verbose:
		for lo, (hi, instruments) in zip([0] + [p[0] for p in P], P):
			print "[%g, %g) --> %s" % (lo, hi, "+".join(instruments))

	#
	# generate random coincidences
	#

	while True:
		# select an instrument combination
		instruments = P[bisect.bisect_left(P, [random.uniform(0.0, 1.0)])][1]

		# randomly selected events from those instruments
		events = tuple(random.choice(eventlists[instrument]) for instrument in instruments)

		# test for a genuine zero-lag coincidence among them
		keep = True
		if not allow_zero_lag and any(abs(ta - tb) < tau[frozenset((instrumenta, instrumentb))] for (instrumenta, ta), (instrumentb, tb) in iterutils.choices(zip(instruments, [timefunc(event) for event in events]), 2)):
			keep = False

		# return acceptable event tuples
		if keep:
			yield events


#
# =============================================================================
#
#                                Triangulation
#
# =============================================================================
#


#
#


class TOATriangulator(object):
	"""
	Time-of-arrival triangulator.  See section 6.6.4 of
	"Gravitational-Wave Physics and Astronomy" by Creighton and
	Anderson.

	An instance of this class is a function-like object that accepts a
	tuple of event arival times and returns a tuple providing
	information derived by solving for the maximum-likelihood source
	location assuming Gaussian-distributed timing errors.
	"""
	def __init__(self, rs, sigmas, v = speed_of_light):
		"""
		Create and initialize a triangulator object.

		rs is a sequence of location 3-vectors, sigmas is a
		sequence of the timing uncertainties for those locations.
		Both sequences must be in the same order --- the first
		sigma in the sequence is interpreted as belonging to the
		first location 3-vector --- and, of course, they must be
		the same length.

		v is the speed at which the wave carrying the signals
		travels.  The rs 3-vectors carry units of distance, the
		sigmas carry units of time, v carries units of
		distance/time.  What units are used for the three is
		arbitrary, but they must be mutually consistent.  The
		default value for v in c, the speed of light, in
		metres/second, therefore the location 3-vectors should be
		given in metres and the sigmas should be given in seconds
		unless a value for v is provided with different units.

		Example:

		>>> from numpy import array
		>>> triangulator = TOATriangulator([
			array([-2161414.92636, -3834695.17889, 4600350.22664]),
			array([  -74276.0447238, -5496283.71971  ,  3224257.01744  ]),
			array([ 4546374.099   ,   842989.697626,  4378576.96241 ])
		], [
			0.005,
			0.005,
			0.005
		])

		This creates a TOATriangulator instance configured for the
		LIGO Hanford, LIGO Livingston and Virgo antennas with 5 ms
		time-of-arrival uncertainties at each location.

		Note:  rs and sigmas may be iterated over multiple times.
		"""
		assert len(rs) == len(sigmas)
		assert len(rs) >= 2

		self.rs = numpy.vstack(rs)
		self.sigmas = numpy.array(sigmas)
		self.v = v

		# sigma^-2 -weighted mean of locations
		rbar = sum(self.rs / self.sigmas[:,numpy.newaxis]**2) / sum(1 / self.sigmas**2)

		# the ith row is r - \bar{r} for the ith location
		self.R = self.rs - rbar

		# ith row is \sigma_i^-2 (r_i - \bar{r}) / c
		M = self.R / (self.v * self.sigmas[:,numpy.newaxis]**2)

		if len(rs) >= 3:
			self.U, self.S, self.VT = numpy.linalg.svd(M)

			# if the smallest singular value is less than 10^-8 * the
			# largest singular value, assume the network is degenerate
			self.singular = abs(self.S.min() / self.S.max()) < 1e-8
		else:
			# len(rs) == 2
			self.max_dt = numpy.dot(self.rs[1] - self.rs[0], self.rs[1] - self.rs[0])**.5 / self.v

	def __call__(self, ts):
		"""
		Triangulate the direction to the source of a signal based
		on a tuple of times when the signal was observed.  ts is a
		sequence of signal arrival times.  One arrival time must be
		provided for each of the observation locations provided
		when the instance was created, and the units of the arrival
		times must be the same as the units used for the sequence
		of sigmas.

		The return value is a tuple of information derived by
		solving for the maximum-likelihood source location assuming
		Gaussian-distributed timing errors.  The return value is

			(n, toa, chi2 / DOF, dt)

		where n is a unit 3-vector pointing from the co-ordinate
		origin towards the source of the signal, toa is the
		time-of-arrival of the signal at the co-ordinate origin,
		chi2 / DOF is the \chi^{2} per degree-of-freedom from to
		the arrival time residuals, and dt is the root-sum-square
		of the arrival time residuals.

		Example:

		>>> n, toa, chi2_per_dof, dt = triangulator([
			794546669.429688,
			794546669.41333,
			794546669.431885
		])
		>>> n
		array([ 0.28747132, -0.37035214,  0.88328904])
		>>> toa
		794546669.40874898
		>>> chi2_per_dof
		2.7407579727907194
		>>> dt
		0.01433725384999875
		"""
		assert len(ts) == len(self.sigmas)

		# change of t co-ordinate to avoid LIGOTimeGPS overflow
		t0 = min(ts)
		ts = numpy.array([float(t - t0) for t in ts])

		# sigma^-2 -weighted mean of arrival times
		tbar = sum(ts / self.sigmas**2) / sum(1 / self.sigmas**2)
		# the i-th element is ts - tbar for the i-th location
		tau = ts - tbar

		if len(self.rs) >= 3:
			tau_prime = numpy.dot(self.U.T, tau)[:3]

			if self.singular:
				l = 0.0
				np = tau_prime / self.S
				try:
					np[2] = math.sqrt(1.0 - np[0]**2 - np[1]**2)
				except ValueError:
					np[2] = 0.0
					np /= math.sqrt(numpy.dot(np, np))
			else:
				def n_prime(l, Stauprime = self.S * tau_prime, S2 = self.S * self.S):
					return Stauprime / (S2 + l)
				def secular_equation(l):
					np = n_prime(l)
					return numpy.dot(np, np) - 1

				# values of l that make the denominator of n'(l) 0
				lsing = -self.S * self.S
				# least negative of them is used as lower bound for
				# bisection search root finder (elements of S are
				# ordered from greatest to least, so the last
				# element of lsing is the least negative)
				l_lo = lsing[-1]

				# find a suitable upper bound for the root finder
				# FIXME:  in Jolien's original code l_hi was
				# hard-coded to 1 but we can't figure out why the
				# root must be <= 1, so I put this loop to be safe
				# but at some point it would be good to figure out
				# if 1.0 can be used because it would allow this
				# loop to be skipped
				l_hi = 1.0
				while secular_equation(l_lo) / secular_equation(l_hi) > 0:
					l_lo, l_hi = l_hi, l_hi * 2

				# solve for l
				l = scipy.optimize.brentq(secular_equation, l_lo, l_hi)

				# compute n'
				np = n_prime(l)

			# compute n from n'
			n = numpy.dot(self.VT.T, np)

			# safety check the nomalization of the result
			assert abs(numpy.dot(n, n) - 1.0) < 1e-8

			# arrival time at origin
			toa = sum((ts - numpy.dot(self.rs, n) / self.v) / self.sigmas**2) / sum(1 / self.sigmas**2)

			# chi^{2}
			chi2 = sum(((numpy.dot(self.R, n) / self.v - tau) / self.sigmas)**2)

			# root-sum-square timing residual
			dt = ts - toa - numpy.dot(self.rs, n) / self.v
			dt = math.sqrt(numpy.dot(dt, dt))
		else:
			# len(rs) == 2
			# FIXME:  fill in n and toa (is chi2 right?)
			n = numpy.zeros((3,), dtype = "double")
			toa = 0.0
			dt = max(abs(ts[1] - ts[0]) - self.max_dt, 0)
			chi2 = dt**2 / sum(self.sigmas**2)

		# done
		return n, t0 + toa, chi2 / len(self.sigmas), dt

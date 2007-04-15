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

"""
A collection of utilities to assist in writing applications that manipulate
data in LIGO Light-Weight XML format.
"""

import bisect
import os
import pickle
import socket
import time

from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import param
from glue.ligolw import lsctables
from glue.ligolw import utils
from pylal.date import XLALUTCToGPS

__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__version__ = "$Revision$"[11:-2]
__date__ = "$Date$"[7:-2]


#
# =============================================================================
#
#                                    Tables
#
# =============================================================================
#


def segmentlistdict_fromsearchsummary(xmldoc, program = None):
	"""
	Convenience wrapper for a common case usage of the segmentlistdict
	class:  searches the process table in xmldoc for occurances of a
	program named program, then scans the search summary table for
	matching process IDs and constructs a segmentlistdict object from
	the out segments in those rows.
	"""
	stbl = table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName)
	ptbl = table.get_table(xmldoc, lsctables.ProcessTable.tableName)
	return stbl.get_out_segmentlistdict(program and ptbl.get_ids_by_program(program))


def get_time_slide_id(xmldoc, time_slide, create_new = None):
	"""
	Return the time_slide_id corresponding to the time slide described
	by time_slide, a dictionary of instrument/offset pairs.  If the
	document does not contain exactly 1 time_slide table, or the table
	does not describe a matching time slide, then KeyError is raised.
	If, however, the optional create_new argument is set to an
	lsctables.Process object (or any other object with a process_id
	attribute), then a time slide table will be created if needed and
	or the missing time slide description added to the table, and
	indicated as having been created by the given process.
	"""
	try:
		tisitable = table.get_table(xmldoc, lsctables.TimeSlideTable.tableName)
	except ValueError:
		# table not found
		if create_new is None:
			raise KeyError, time_slide
		tisitable = lsctables.New(lsctables.TimeSlideTable)
		xmldoc.childNodes[0].appendChild(tisitable)
	for id in tisitable.dict.iterkeys():
		if tisitable.get_offset_dict(id) == time_slide:
			return id
	# time slide not found in table
	if create_new is None:
		raise KeyError, time_slide
	id = tisitable.sync_ids().next()
	for instrument, offset in time_slide.iteritems():
		row = lsctables.TimeSlide()
		row.process_id = create_new.process_id
		row.time_slide_id = id
		row.instrument = instrument
		row.offset = offset
		tisitable.append(row)
	return id


def get_zero_lag_time_slides(xmldoc, instrument_combinations = None):
	"""
	Return a dictionary of the time slides that have all zero offsets.
	The dictionary maps time slide IDs to dictionaries of instrument
	--> offset mappings.  The optional instrument_combinations argument
	can be used to provide a list of lists of instrument combinations
	to consider.  For example, [["H1", "H2"], ["H1", "H2", "L1"]]
	requests time slides describing all-zero offsets for either the
	H1+H2 or H1+H2+L1 instrument combinations.  Order doesn't matter
	within an individual instrument combination.  Passing
	instrument_combinations = None (the default) requests time slides
	for all instrument combinations.
	"""
	# convert instrument combinations into sets for easy comparison
	if instrument_combinations is not None:
		instrument_combinations = map(set, instrument_combinations)

	# extract zero-lag ID --> offset dictionary mapping
	zero_lag_offset_dicts = {}
	for id, offset_dict in table.get_table(xmldoc, lsctables.TimeSlideTable.tableName).get_offsets().iteritems():
		for offset in offset_dict.itervalues():
			if offset != 0:
				# not zero-lag
				break
		else:
			# loop exited normally (all offsets == 0)
			zero_lag_offset_dicts[id] = offset_dict

	# remove unwanted instrument combinations
	if instrument_combinations is not None:
		for id, offset_dict in zero_lag_offset_dicts.items():
			if set(offset_dict.keys()) not in instrument_combinations:
				del zero_lag_offset_dicts[id]

	# done
	return zero_lag_offset_dicts


def get_coinc_def_id(xmldoc, table_names, create_new = True):
	"""
	Return the coinc_def_id corresponding to coincidences consisting
	exclusively of events from the given table names.  If no matching
	coinc_def_id is found, then a new one is created and the ID
	returned.  If the document does not contain a coinc_definer table,
	then one is added, a new coind_def_id created, and the ID returned.
	If, however, create_new is False, and for any reason the ID isn't
	found then KeyError is raised.
	"""
	try:
		coincdeftable = table.get_table(xmldoc, lsctables.CoincDefTable.tableName)
	except ValueError:
		# table not found
		if not create_new:
			raise KeyError, table_names
		coincdeftable = lsctables.New(lsctables.CoincDefTable)
		xmldoc.childNodes[0].appendChild(coincdeftable)
	table_names = list(table_names)
	table_names.sort()
	for id in coincdeftable.dict.iterkeys():
		if coincdeftable.get_contributors(id) == table_names:
			return id
	# contributor list not found in table
	if not create_new:
		raise KeyError, table_names
	id = coincdeftable.sync_ids().next()
	for name in table_names:
		row = lsctables.CoincDef()
		row.coinc_def_id = id
		row.table_name = name
		coincdeftable.append(row)
	return id


#
# =============================================================================
#
#                                    Params
#
# =============================================================================
#


def pickle_to_param(obj, name):
	"""
	Return the top-level element of a document sub-tree containing the
	pickled serialization of a Python object.
	"""
	return param.from_pyvalue(u"pickle:%s" % name, unicode(pickle.dumps(obj)))


def pickle_from_param(elem, name):
	"""
	Retrieve a pickled Python object from the document tree rooted at
	elem.
	"""
	return pickle.loads(str(param.get_pyvalue(elem, u"pickle:%s" % name)))


#
# =============================================================================
#
#                               Process Metadata
#
# =============================================================================
#


def append_process(doc, program = "", version = "", cvs_repository = "", cvs_entry_time = "", comment = "", is_online = False, jobid = 0, domain = "", ifos = ""):
	"""
	Add an entry to the process table in doc.  program, version,
	cvs_repository, comment, domain, and ifos should all be strings.
	cvs_entry_time should be a string in the format "YYYY/MM/DD
	HH:MM:SS".  is_online should be a boolean, jobid an integer.
	"""
	proctable = table.get_table(doc, lsctables.ProcessTable.tableName)
	process = lsctables.Process()
	process.program = program
	process.version = version
	process.cvs_repository = cvs_repository
	process.cvs_entry_time = XLALUTCToGPS(time.strptime(cvs_entry_time, "%Y/%m/%d %H:%M:%S")).seconds
	process.comment = comment
	process.is_online = int(is_online)
	process.node = socket.gethostbyaddr(socket.gethostname())[0]
	process.username = os.environ["LOGNAME"]
	process.unix_procid = os.getpid()
	process.start_time = XLALUTCToGPS(time.gmtime()).seconds
	process.end_time = 0
	process.jobid = jobid
	process.domain = domain
	process.ifos = ifos
	process.process_id = proctable.sync_ids().next()
	proctable.append(process)
	return process


def set_process_end_time(process):
	"""
	Set the end time in a row in a process table to the current time.
	"""
	process.end_time = XLALUTCToGPS(time.gmtime()).seconds
	return process


def append_process_params(doc, process, params):
	"""
	doc is an XML document tree, process is the row in the process
	table for which these are the parameters, and params is a list of
	(name, type, value) tuples one for each parameter.
	"""
	paramtable = table.get_table(doc, lsctables.ProcessParamsTable.tableName)
	for name, type, value in params:
		row = lsctables.ProcessParams()
		row.program = process.program
		row.process_id = process.process_id
		row.param = str(name)
		row.type = str(type)
		row.value = str(value)
		paramtable.append(row)
	return process


def get_process_params(xmldoc, program, param):
	process_ids = table.get_table(xmldoc, lsctables.ProcessTable.tableName).get_ids_by_program(program)
	if len(process_ids) != 1:
		raise ValueError, "process table must contain exactly one program named %s" % program
	return [row.value for row in table.get_table(xmldoc, lsctables.ProcessParamsTable.tableName) if (row.process_id in process_ids) and (row.param == param)]


def dbget_process_params(connection, program, param):
	process_ids = set()
	values = []
	for process_id, value in connection.cursor().execute("""
SELECT process_id, value FROM
	process_params
WHERE
	program == ?
	AND param == ?
	""", (program, param)):
		process_ids.add(process_id)
		values.append(value)
	if len(process_ids) != 1:
		raise ValueError, "process table must contain exactly one program named %s with params %s" % (program, param)
	return values


def doc_includes_process(doc, program):
	"""
	Return True if the process table in doc includes entries for a
	program named program.
	"""
	return program in table.get_table(doc, lsctables.ProcessTable.tableName).getColumnByName("program")


#
# =============================================================================
#
#                               Search Metadata
#
# =============================================================================
#


def append_search_summary(doc, process, shared_object = "standalone", lalwrapper_cvs_tag = "", lal_cvs_tag = "", comment = None, ifos = None, inseg = None, outseg = None, nevents = 0, nnodes = 1):
	"""
	Append search summary information associated with the given process
	to the search summary table in doc.
	"""
	summary = lsctables.SearchSummary()
	summary.process_id = process.process_id
	summary.shared_object = shared_object
	summary.lalwrapper_cvs_tag = lalwrapper_cvs_tag
	summary.lal_cvs_tag = lal_cvs_tag
	summary.comment = comment or process.comment
	summary.ifos = ifos or process.ifos
	summary.set_in(inseg)
	summary.set_out(outseg)
	summary.nevents = nevents
	summary.nnodes = nnodes
	table.get_table(doc, lsctables.SearchSummaryTable.tableName).append(summary)
	return summary


#
# =============================================================================
#
#                                    Other
#
# =============================================================================
#


def smallest_enclosing_seg(a, b):
	"""
	Return the smallest segment that contains both a and b.
	"""
	return segments.segment(min(a[0], b[0]), max(a[1], b[1]))


def bisect_contains(array, val):
	"""
	Uses a bisection search to determine if val is in array.  Returns
	True or False.  NOTE:  this is going to be removed, use Python sets
	instead.
	"""
	try:
		return array[bisect.bisect_left(array, val)] == val
	except IndexError:
		return False


def get_coincident_segmentlistdict(seglistdict, offsetdictlist):
	"""
	This function answers the question "Given a set of segment lists,
	and a set of time slides to apply to those segment lists, what
	segments do I need to keep in the original lists so that I have all
	the segments that will participate in a coincidence analysis done
	over those time slides?

	This function constructs and returns a segmentlistdict object that,
	for each key in seglistdict, contains the segments from the
	corresponding list in seglistdict which are coincident under at
	least one of the time slides described by offsetdictlist.

	offsetdictlist is a list of dictionaries of instrument/offset
	pairs, with each dictionary describing a time slide and the
	instruments that participate in it.  Each element in the list is
	free to contain only subsets of the keys in seglistdict.  In those
	cases, the coincidence is computed only between the segment lists
	corresponding to the given keys.

	For example, let us say that "input" is a segmentlistdict object
	containing segment lists for three instruments, "H1", "H2" and
	"L1".  And let us say that "slides" is a list of dictionaries, and
	is equal to [{"H1":0, "H2":0, "L1":0}, {"H1":0, "H2":10}].  Then if

	output = get_coincident_segmentlistdict(input, slides)

	output will contain, for each of the three instruments, the
	segments (or parts thereof) from the original lists that are
	required in order to perform a triple-coincident analysis at zero
	lag betwen the three instruments, *and* a double-coincident
	analysis between H1 and H2 with H2 offset by 10 seconds.

	During the computations, the input segmentlistdict object will have
	offsets applied to it in place, but they will be restored to their
	original values upon exit.  The segmentlistdict object returned by
	this function has its offsets set to those of the input
	segmentlistdict.
	"""
	origoffsets = dict(seglistdict.offsets)
	coincseglists = segments.segmentlistdict()
	for offsetdict in offsetdictlist:
		seglistdict.offsets.update(offsetdict)
		intersection = seglistdict.extract_common(offsetdict.iterkeys())
		intersection.offsets.clear()
		coincseglists |= intersection
	seglistdict.offsets.update(origoffsets)
	coincseglists.offsets.update(origoffsets)
	return coincseglists

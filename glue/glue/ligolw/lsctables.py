# $Id$
#
# Copyright (C) 2006  Kipp C. Cannon
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
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
LSC Table definitions.  These must be kept synchronized with the official
definitions in the LDAS CVS repository at
http://www.ldas-sw.ligo.caltech.edu/cgi-bin/cvsweb.cgi/ldas/dbms/db2/sql.
Maintainership of the table definitions is left as an excercise to
interested users.
"""

__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"[7:-2]
__version__ = "$Revision$"[11:-2]


from xml import sax
try:
	set
except NameError:
	# Python < 2.4
	from sets import Set as set
try:
	any
	all
except NameError:
	# Python < 2.5
	from glue.iterutils import any, all


from glue import segments
from glue.lal import LIGOTimeGPS
import ligolw
import table
import types as ligolwtypes
import ilwd


#
# =============================================================================
#
#                            Convenience Functions
#
# =============================================================================
#


def New(Type, columns = None):
	"""
	Convenience function for constructing pre-defined LSC tables.  The
	optional columns argument is a list of the names of the columns the
	table should be constructed with.  If columns = None, then the
	table is constructed with all valid columns included (pass columns
	= [] to create a table with no columns).

	Example:

	>>> tbl = New(ProcessTable)
	>>> tbl.write()
	"""
	new = Type(sax.xmlreader.AttributesImpl({u"Name": Type.tableName}))
	colnamefmt = u":".join(Type.tableName.split(":")[:-1]) + u":%s"
	if columns is not None:
		for key in columns:
			if key not in new.validcolumns:
				raise ligolw.ElementError, "invalid Column '%s' for Table '%s'" % (key, new.tableName)
			new.appendChild(table.Column(sax.xmlreader.AttributesImpl({u"Name": colnamefmt % key, u"Type": new.validcolumns[key]})))
	else:
		for key, value in new.validcolumns.items():
			new.appendChild(table.Column(sax.xmlreader.AttributesImpl({u"Name": colnamefmt % key, u"Type": value})))
	new._end_of_columns()
	new.appendChild(table.TableStream(sax.xmlreader.AttributesImpl({u"Name": Type.tableName})))
	return new


def IsTableElement(Type, elem):
	"""
	Convenience function to check that an element is a Table of type
	Type.
	"""
	if elem.tagName != ligolw.Table.tagName:
		return False
	return table.CompareTableNames(elem.getAttribute(u"Name"), Type.tableName) == 0


def IsTableProperties(Type, tagname, attrs):
	"""
	Convenience function to check that the given tag name and
	attributes match those of a Table of type Type.
	"""
	if tagname != ligolw.Table.tagName:
		return False
	return table.CompareTableNames(attrs[u"Name"], Type.tableName) == 0


def getTablesByType(elem, Type):
	"""
	Return a list of tables of type Type under elem.
	"""
	return table.getTablesByName(elem, Type.tableName)


def HasNonLSCTables(elem):
	"""
	Return True if the document tree below elem contains non-LSC
	tables, otherwise return False.
	"""
	for t in elem.getElementsByTagName(ligolw.Table.tagName):
		if table.StripTableName(t.getAttribute(u"Name")) not in TableByName:
			return True
	return False


def instrument_set_from_ifos(ifos):
	"""
	Convenience function for parsing the values stored in the "ifos"
	columns found in many tables.  This function is mostly for internal
	use by the .get_ifos() methods of the corresponding row classes.
	The mapping from input to output is as follows (rules are applied
	in order):

	input is None --> output is None

	input contains "," --> output is set of strings split on "," with
	leading and trailing whitespace stripped from each piece

	input contains "+" --> output is set of strings split on "," with
	leading and trailing whitespace stripped from each piece

	after stripping input of leading and trailing whitespace,

	input has an even length greater than two --> output is set of
	two-character pieces

	input is a non-empty string --> output is a set containing input as
	single value

	otherwise --> output is an empty set.
	"""
	if ifos is None:
		return None
	if "," in ifos:
		return set(map(unicode.strip, ifos.split(",")))
	if "+" in ifos:
		return set(map(unicode.strip, ifos.split("+")))
	ifos = ifos.strip()
	if len(ifos) > 2 and not len(ifos) % 2:
		# if ifos is a string with an even number of characters
		# greater than two, split it into two-character pieces.
		# FIXME:  remove this when the inspiral codes don't write
		# ifos strings like this anymore
		return set(ifos[n:n+2] for n in range(0, len(ifos), 2))
	if ifos:
		return set([ifos])
	return set()


def ifos_from_instrument_set(instruments):
	"""
	Convenience function to convert an iterable of instrument names
	into a value suitable for storage in the "ifos" column found in
	many tables.  This function is mostly for internal use by the
	.set_ifos() methods of the corresponding row classes.  The input
	can be None or an interable of zero or more instrument names, none
	of which may contain "," or "+" characters.  The output is a single
	string containing the instrument names concatenated using "," as a
	delimiter.  Whitespace is allowed in instrument names but may not
	be preserved.
	"""
	if instruments is None:
		return None
	instruments = sorted(instrument.strip() for instrument in instruments)
	if any(map(lambda instrument: "," in instrument or "+" in instrument, instruments)):
		raise ValueError, instruments
	return ",".join(instruments)


#
# =============================================================================
#
#                                process:table
#
# =============================================================================
#


ProcessID = ilwd.get_ilwdchar_class(u"process", u"process_id")


class ProcessTable(table.Table):
	tableName = "process:table"
	validcolumns = {
		"program": "lstring",
		"version": "lstring",
		"cvs_repository": "lstring",
		"cvs_entry_time": "int_4s",
		"comment": "lstring",
		"is_online": "int_4s",
		"node": "lstring",
		"username": "lstring",
		"unix_procid": "int_4s",
		"start_time": "int_4s",
		"end_time": "int_4s",
		"jobid": "int_4s",
		"domain": "lstring",
		"ifos": "lstring",
		"process_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (process_id)"
	next_id = ProcessID(0)

	def get_ids_by_program(self, program):
		"""
		Return a set containing the process IDs from rows whose
		program string equals the given program.
		"""
		return set([row.process_id for row in self if row.program == program])


class Process(object):
	__slots__ = ProcessTable.validcolumns.keys()

	def get_ifos(self):
		"""
		Return a set of the instruments for this row.
		"""
		return instrument_set_from_ifos(self.ifos)

	def set_ifos(self, instruments):
		"""
		Serialize a sequence of instruments into the ifos
		attribute.  The instrument names must not contain the ","
		character.
		"""
		self.ifos = ifos_from_instrument_set(instruments)


ProcessTable.RowType = Process


#
# =============================================================================
#
#                                lfn:table
#
# =============================================================================
#


LfnID = ilwd.get_ilwdchar_class(u"lfn", u"lfn_id")


class LfnTable(table.Table):
	tableName = "lfn:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"lfn_id": "ilwd:char",
		"name": "lstring",
		"comment": "lstring",
		"start_time": "int_4s",
		"end_time": "int_4s"
	}
	constraints = "PRIMARY KEY (lfn_id)"
	next_id = LfnID(0)


class Lfn(object):
	__slots__ = LfnTable.validcolumns.keys()


LfnTable.RowType = Lfn


#
# =============================================================================
#
#                             process_params:table
#
# =============================================================================
#


class ProcessParamsTable(table.Table):
	tableName = "process_params:table"
	validcolumns = {
		"program": "lstring",
		"process_id": "ilwd:char",
		"param": "lstring",
		"type": "lstring",
		"value": "lstring"
	}
	# FIXME: these constraints break ID remapping in the DB backend
	#constraints = "PRIMARY KEY (process_id, param)"

	def append(self, row):
		if row.type is not None and row.type not in ligolwtypes.Types:
			raise ligolw.ElementError, "unrecognized type '%s'" % row.type
		table.Table.append(self, row)


class ProcessParams(object):
	__slots__ = ProcessParamsTable.validcolumns.keys()

	def get_pyvalue(self):
		if self.value is None:
			return None
		return ligolwtypes.ToPyType[self.type or "lstring"](self.value)


ProcessParamsTable.RowType = ProcessParams


#
# =============================================================================
#
#                             search_summary:table
#
# =============================================================================
#


class SearchSummaryTable(table.Table):
	tableName = "search_summary:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"shared_object": "lstring",
		"lalwrapper_cvs_tag": "lstring",
		"lal_cvs_tag": "lstring",
		"comment": "lstring",
		"ifos": "lstring",
		"in_start_time": "int_4s",
		"in_start_time_ns": "int_4s",
		"in_end_time": "int_4s",
		"in_end_time_ns": "int_4s",
		"out_start_time": "int_4s",
		"out_start_time_ns": "int_4s",
		"out_end_time": "int_4s",
		"out_end_time_ns": "int_4s",
		"nevents": "int_4s",
		"nnodes": "int_4s"
	}

	def get_inlist(self):
		"""
		Return a segmentlist object describing the times spanned by
		the input segments of all rows in the table.

		Note:  the result is not coalesced, the segmentlist
		contains the segments as they appear in the table.
		"""
		return segments.segmentlist([row.get_in() for row in self])

	def get_outlist(self):
		"""
		Return a segmentlist object describing the times spanned by
		the output segments of all rows in the table.

		Note:  the result is not coalesced, the segmentlist
		contains the segments as they appear in the table.
		"""
		return segments.segmentlist([row.get_out() for row in self])

	def get_in_segmentlistdict(self, process_ids = None):
		"""
		Return a segmentlistdict mapping instrument to in segment
		list.  If process_ids is a sequence of process IDs, then
		only rows with matching IDs are included otherwise all rows
		are included.

		Note:  the result is not coalesced, each segmentlist
		contains the segments listed for that instrument as they
		appeared in the table.
		"""
		seglists = segments.segmentlistdict()
		for row in self:
			ifos = row.get_ifos()
			if ifos is None:
				ifos = (None,)
			if process_ids is None or row.process_id in process_ids:
				seglists.extend(dict((ifo, segments.segmentlist([row.get_in()])) for ifo in ifos))
		return seglists

	def get_out_segmentlistdict(self, process_ids = None):
		"""
		Return a segmentlistdict mapping instrument to out segment
		list.  If process_ids is a sequence of process IDs, then
		only rows with matching IDs are included otherwise all rows
		are included.

		Note:  the result is not coalesced, each segmentlist
		contains the segments listed for that instrument as they
		appeared in the table.
		"""
		seglists = segments.segmentlistdict()
		for row in self:
			ifos = row.get_ifos()
			if ifos is None:
				ifos = (None,)
			if process_ids is None or row.process_id in process_ids:
				seglists.extend(dict((ifo, segments.segmentlist([row.get_out()])) for ifo in ifos))
		return seglists


class SearchSummary(object):
	__slots__ = SearchSummaryTable.validcolumns.keys()

	def get_ifos(self):
		"""
		Return a set of the instruments for this row.
		"""
		return instrument_set_from_ifos(self.ifos)

	def set_ifos(self, instruments):
		"""
		Serialize a sequence of instruments into the ifos
		attribute.  The instrument names must not contain the ","
		character.
		"""
		self.ifos = ifos_from_instrument_set(instruments)

	def get_in(self):
		"""
		Return the input segment.
		"""
		return segments.segment(LIGOTimeGPS(self.in_start_time, self.in_start_time_ns), LIGOTimeGPS(self.in_end_time, self.in_end_time_ns))

	def set_in(self, seg):
		"""
		Set the input segment.
		"""
		self.in_start_time, self.in_start_time_ns = seg[0].seconds, seg[0].nanoseconds
		self.in_end_time, self.in_end_time_ns = seg[1].seconds, seg[1].nanoseconds

	def get_out(self):
		"""
		Get the output segment.
		"""
		return segments.segment(LIGOTimeGPS(self.out_start_time, self.out_start_time_ns), LIGOTimeGPS(self.out_end_time, self.out_end_time_ns))

	def set_out(self, seg):
		"""
		Set the output segment.
		"""
		self.out_start_time, self.out_start_time_ns = seg[0].seconds, seg[0].nanoseconds
		self.out_end_time, self.out_end_time_ns = seg[1].seconds, seg[1].nanoseconds


SearchSummaryTable.RowType = SearchSummary


#
# =============================================================================
#
#                            search_summvars:table
#
# =============================================================================
#


SearchSummVarsID = ilwd.get_ilwdchar_class(u"search_summvars", u"search_summvar_id")


class SearchSummVarsTable(table.Table):
	tableName = "search_summvars:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"search_summvar_id": "ilwd:char",
		"name": "lstring",
		"string": "lstring",
		"value": "real_8"
	}
	constraints = "PRIMARY KEY (search_summvar_id)"
	next_id = SearchSummVarsID(0)


class SearchSummVars(object):
	__slots__ = SearchSummVarsTable.validcolumns.keys()


SearchSummVarsTable.RowType = SearchSummVars


#
# =============================================================================
#
#                               sngl_burst:table
#
# =============================================================================
#


SnglBurstID = ilwd.get_ilwdchar_class(u"sngl_burst", u"event_id")


class SnglBurstTable(table.Table):
	tableName = "sngl_burst:table"
	validcolumns = {
		"creator_db": "int_4s",
		"process_id": "ilwd:char",
		"filter_id": "ilwd:char",
		"ifo": "lstring",
		"search": "lstring",
		"channel": "lstring",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"stop_time": "int_4s",
		"stop_time_ns": "int_4s",
		"duration": "real_4",
		"flow": "real_4",
		"fhigh": "real_4",
		"central_freq": "real_4",
		"bandwidth": "real_4",
		"amplitude": "real_4",
		"snr": "real_4",
		"confidence": "real_4",
		"tfvolume": "real_4",
		"hrss": "real_4",
		"time_lag": "real_4",
		"peak_time": "int_4s",
		"peak_time_ns": "int_4s",
		"peak_frequency": "real_4",
		"peak_strain": "real_4",
		"peak_time_error": "real_4",
		"peak_frequency_error": "real_4",
		"peak_strain_error": "real_4",
		"ms_start_time": "int_4s",
		"ms_start_time_ns": "int_4s",
		"ms_stop_time": "int_4s",
		"ms_stop_time_ns": "int_4s",
		"ms_duration": "real_4",
		"ms_flow": "real_4",
		"ms_fhigh": "real_4",
		"ms_bandwidth": "real_4",
		"ms_hrss": "real_4",
		"ms_snr": "real_4",
		"ms_confidence": "real_4",
		"param_one_name": "lstring",
		"param_one_value": "real_8",
		"param_two_name": "lstring",
		"param_two_value": "real_8",
		"param_three_name": "lstring",
		"param_three_value": "real_8",
		"event_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (event_id)"
	next_id = SnglBurstID(0)
	interncolumns = ("process_id", "ifo", "search", "channel")


class SnglBurst(object):
	__slots__ = SnglBurstTable.validcolumns.keys()

	#
	# Tile properties
	#

	def get_start(self):
		return LIGOTimeGPS(self.start_time, self.start_time_ns)

	def set_start(self, gps):
		self.start_time, self.start_time_ns = gps.seconds, gps.nanoseconds

	def get_stop(self):
		return LIGOTimeGPS(self.stop_time, self.stop_time_ns)

	def set_stop(self, gps):
		self.stop_time, self.stop_time_ns = gps.seconds, gps.nanoseconds

	def get_peak(self):
		return LIGOTimeGPS(self.peak_time, self.peak_time_ns)

	def set_peak(self, gps):
		self.peak_time, self.peak_time_ns = gps.seconds, gps.nanoseconds

	def get_period(self):
		start = LIGOTimeGPS(self.start_time, self.start_time_ns)
		return segments.segment(start, start + self.duration)

	def set_period(self, period):
		self.start_time, self.start_time_ns = period[0].seconds, period[0].nanoseconds
		self.duration = float(abs(period))

	def get_band(self):
		low = self.central_freq - self.bandwidth / 2
		return segments.segment(low, low + self.bandwidth)

	def set_band(self, band):
		self.central_freq = (band[0] + band[1])/2.0
		self.bandwidth = abs(band)

	#
	# "Most significant pixel" properties
	#

	def get_ms_start(self):
		return LIGOTimeGPS(self.ms_start_time, self.ms_start_time_ns)

	def set_ms_start(self, gps):
		self.ms_start_time, self.ms_start_time_ns = gps.seconds, gps.nanoseconds

	def get_ms_stop(self):
		return LIGOTimeGPS(self.ms_stop_time, self.ms_stop_time_ns)

	def set_ms_stop(self, gps):
		self.ms_stop_time, self.ms_stop_time_ns = gps.seconds, gps.nanoseconds

	def get_ms_period(self):
		start = LIGOTimeGPS(self.ms_start_time, self.ms_start_time_ns)
		return segments.segment(start, start + self.ms_duration)

	def set_ms_period(self, period):
		self.ms_start_time, self.ms_start_time_ns = period[0].seconds, period[0].nanoseconds
		self.ms_duration = float(abs(period))

	def get_ms_band(self):
		return segments.segment(self.ms_flow, self.ms_flow + self.ms_bandwidth)

	def set_ms_band(self, band):
		self.ms_flow = band[0]
		self.ms_bandwidth = abs(band)


SnglBurstTable.RowType = SnglBurst


#
# =============================================================================
#
#                              multi_burst:table
#
# =============================================================================
#


#
# FIXME:  I think extra columns have been added here that aren't in other
# places where this table is defined.
#


class MultiBurstTable(table.Table):
	tableName = "multi_burst:table"
	validcolumns = {
		"creator_db": "int_4s",
		"process_id": "ilwd:char",
		"filter_id": "ilwd:char",
		"ifos": "lstring",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"duration": "real_4",
		"peak_time": "int_4s",
		"peak_time_ns": "int_4s",
		"central_freq": "real_4",
		"bandwidth": "real_4",
		"amplitude": "real_4",
		"snr": "real_4",
		"confidence": "real_4",
		"ligo_axis_ra": "real_4",
		"ligo_axis_dec": "real_4",
		"ligo_angle": "real_4",
		"ligo_angle_sig": "real_4",
		"coinc_event_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (coinc_event_id)"


class MultiBurst(object):
	__slots__ = MultiBurstTable.validcolumns.keys()

	def get_ifos(self):
		"""
		Return a set of the instruments for this row.
		"""
		return instrument_set_from_ifos(self.ifos)

	def set_ifos(self, instruments):
		"""
		Serialize a sequence of instruments into the ifos
		attribute.  The instrument names must not contain the ","
		character.
		"""
		self.ifos = ifos_from_instrument_set(instruments)

	def get_peak(self):
		return LIGOTimeGPS(self.peak_time, self.peak_time_ns)

	def set_peak(self, gps):
		self.peak_time, self.peak_time_ns = gps.seconds, gps.nanoseconds


MultiBurstTable.RowType = MultiBurst


#
# =============================================================================
#
#                             sngl_inspiral:table
#
# =============================================================================
#


SnglInspiralID = ilwd.get_ilwdchar_class(u"sngl_inspiral", u"event_id")


class SnglInspiralID_old(object):
	"""
	Custom row ID thing for sngl_inspiral tables with int_8s event IDs.
	"""
	# FIXME: remove this class when the event_id column no longer
	# encodes time slide information.
	column_name = "event_id"

	def __init__(self, n = 0):
		self.n = n

	def new(self, row):
		self.n += 1
		a = self.n // 100000
		b = self.n % 100000
		return SnglInspiralID(a * 1000000000 + row.get_id_parts()[1] * 100000 + b)


class SnglInspiralTable(table.Table):
	tableName = "sngl_inspiral:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"ifo": "lstring",
		"search": "lstring",
		"channel": "lstring",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"end_time_gmst": "real_8",
		"impulse_time": "int_4s",
		"impulse_time_ns": "int_4s",
		"template_duration": "real_8",
		"event_duration": "real_8",
		"amplitude": "real_4",
		"eff_distance": "real_4",
		"coa_phase": "real_4",
		"mass1": "real_4",
		"mass2": "real_4",
		"mchirp": "real_4",
		"mtotal": "real_4",
		"eta": "real_4",
		"kappa": "real_4",
		"chi": "real_4",
		"tau0": "real_4",
		"tau2": "real_4",
		"tau3": "real_4",
		"tau4": "real_4",
		"tau5": "real_4",
		"ttotal": "real_4",
		"psi0": "real_4",
		"psi3": "real_4",
		"alpha": "real_4",
		"alpha1": "real_4",
		"alpha2": "real_4",
		"alpha3": "real_4",
		"alpha4": "real_4",
		"alpha5": "real_4",
		"alpha6": "real_4",
		"beta": "real_4",
		"f_final": "real_4",
		"snr": "real_4",
		"chisq": "real_4",
		"chisq_dof": "int_4s",
		"bank_chisq": "real_4",
		"bank_chisq_dof": "int_4s",
		"cont_chisq": "real_4",
		"cont_chisq_dof": "int_4s",
		"sigmasq": "real_8",
		"rsqveto_duration": "real_4",
		"Gamma0": "real_4",
		"Gamma1": "real_4",
		"Gamma2": "real_4",
		"Gamma3": "real_4",
		"Gamma4": "real_4",
		"Gamma5": "real_4",
		"Gamma6": "real_4",
		"Gamma7": "real_4",
		"Gamma8": "real_4",
		"Gamma9": "real_4",
		"event_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (event_id)"
	# FIXME:  uncomment the next line when the event_id column no
	# longer encodes time slide information
	# FIXME:  lal uses an ID of 0 to indicate "no valid ID has been
	# set", so we start at 1 for safety, but eventually that should be
	# fixed in LAL and then this can be put back to 0 for cleanliness.
	#next_id = SnglInspiralID(1)
	interncolumns = ("process_id", "ifo", "search", "channel")

	def updateKeyMapping(self, mapping):
		# FIXME: remove this method when the event_id column no
		# longer encodes time slide information
		if self.next_id is not None:
			for row in self:
				if row.event_id not in mapping:
					mapping[row.event_id] = self.next_id.new(row)
				row.event_id = mapping[row.event_id]
		return mapping

	def get_column(self,column):
		if column == 'reduced_bank_chisq':
			return self.get_reduced_bank_chisq()
		if column == 'reduced_cont_chisq':
			return self.get_reduced_cont_chisq()
		if column == 'effective_snr':
			return self.get_effective_snr()
		if column == 'snr_over_chi':
			return self.get_snr_over_chi()
		elif column == 'chirp_distance':
			return self.get_chirp_dist()
		else:
			return self.getColumnByName(column).asarray()

	def get_end(self):
		return [row.get_end() for row in self]

	def get_reduced_bank_chisq(self):
		return self.get_column('bank_chisq') / self.get_column('bank_chisq_dof')

	def get_reduced_cont_chisq(self):
		return self.get_column('cont_chisq') / self.get_column('cont_chisq_dof')

	def get_effective_snr(self, fac=250.0):    
		snr = self.get_column('snr')
		chisq = self.get_column('chisq')
		chisq_dof = self.get_column('chisq_dof')
		return snr/ (1 + snr**2/fac)**(0.25) / (chisq/(2*chisq_dof - 2) )**(0.25)
    
	def get_chirp_distance(self,ref_mass = 1.40):
		mchirp = self.get_column('mchirp')
		eff_dist = self.get_column('eff_distance')
		return eff_dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)

	def get_snr_over_chi(self):
		return self.get_column('snr')/self.get_column('chisq')**(1./2)

	def ifocut(self,ifo):
		ifoTrigs = table.new_from_template(self)
		for row in self:
			if row.ifo == ifo:
				ifoTrigs.append(row)
		return ifoTrigs

	def veto(self,seglist):
		vetoed = table.new_from_template(self)
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end()
			if time in seglist:
				vetoed.append(row)
			else:
				keep.append(row)
		return keep
	
	def vetoed(self, seglist):
		"""
		Return the inverse of what veto returns, i.e., return the triggers
		that lie within a given seglist.
		"""
		vetoed = table.new_from_template(self)
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end()
			if time in seglist:
				vetoed.append(row)
			else:
				keep.append(row)
		return vetoed
	
	def veto_seglistdict(self, seglistdict):
		vetoed = table.new_from_template(self)
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end()
			if time in seglistdict[row.ifo]:
				vetoed.append(row)
			else:
				keep.append(row)
		return keep
	
	def vetoed_seglistdict(self, seglistdict):
		vetoed = table.new_from_template(self)
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end()
			if time in seglistdict[row.ifo]:
				vetoed.append(row)
			else:
				keep.append(row)
		return vetoed
	
	def getslide(self,slide_num):
		"""
		Return the triggers with a specific slide number.
		@param slide_num: the slide number to recover (contained in the event_id)
		"""
		slideTrigs = table.new_from_template(self)
		slideTrigs.extend([row for row in self if row.get_slide_number() == slide_num])
		return slideTrigs


class SnglInspiral(object):
	__slots__ = SnglInspiralTable.validcolumns.keys()

	def get_end(self):
		return LIGOTimeGPS(self.end_time, self.end_time_ns)

	def set_end(self, gps):
		self.end_time, self.end_time_ns = gps.seconds, gps.nanoseconds

	def get_effective_snr(self,fac=250.0):
		return self.snr/ (1 + self.snr**2/fac)**(0.25)/(self.chisq/(2*self.chisq_dof - 2) )**(0.25) 

	def get_far(self):
		return self.alpha

	def get_ifar(self):
		if self.alpha < 0.000000001:
			self.alpha = 0.000000001
		return 1./self.alpha

	def get_id_parts(self):
		"""
		Return the three pieces of the int_8s-style sngl_inspiral
		event_id.
		"""
		int_event_id = int(self.event_id)
		a = int_event_id // 1000000000
		slidenum = (int_event_id % 1000000000) // 100000
		b = int_event_id % 100000
		return int(a), int(slidenum), int(b)

	def get_slide_number(self):
		"""
		Return the slide-number for this trigger
		"""
		a, slide_number, b = self.get_id_parts()
		if slide_number > 5000:
			slide_number = 5000 - slide_number
		return slide_number

	# FIXME: how are two inspiral events defined to be the same?
	def __eq__(self, other):
		return not (
			cmp(self.ifo, other.ifo) or
			cmp(self.end_time, other.end_time) or
			cmp(self.end_time_ns, other.end_time_ns) or
			cmp(self.mass1, other.mass1) or
			cmp(self.mass2, other.mass2) or
			cmp(self.search, other.search)
		)
		
	def __hash__(self):
		# FIXME:  the things in this tuple must be the same things
		# used above for comparison (if two objects have different
		# hashes they must compare as not equal or stuff breaks),
		# so make sure to keep this updated if the choice of how to
		# compare to triggers changes above.
		return hash((self.ifo, self.end_time, self.end_time_ns, self.mass1, self.mass2, self.search))


SnglInspiralTable.RowType = SnglInspiral


#
# =============================================================================
#
#                             coinc_inspiral:table
#
# =============================================================================
#


class CoincInspiralTable(table.Table):
	tableName = "coinc_inspiral:table"
	validcolumns = {
		"coinc_event_id": "ilwd:char",
                "ifos": "lstring",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"mchirp": "real_8",
		"snr": "real_8",
		"false_alarm_rate": "real_8"
	}
	# FIXME:  like some other tables here, this table should have the
	# constraint that the coinc_event_id column is a primary key.  this
	# breaks ID reassignment in ligolw_sqlite, so until that is fixed
	# the constraint is being replaced with an index.
	#constraints = "PRIMARY KEY (coinc_event_id)"
	how_to_index = {
		"ci_cei_index": ("coinc_event_id",)
	}
	interncolumns = ("coinc_event_id",)


class CoincInspiral(object):
	__slots__ = CoincInspiralTable.validcolumns.keys()

	def get_end(self):
		return LIGOTimeGPS(self.end_time, self.end_time_ns)

	def set_end(self, gps):
		self.end_time, self.end_time_ns = gps.seconds, gps.nanoseconds

	def set_ifos(self, ifos):
		self.ifos = ifos_from_instrument_set(ifos)

	def get_ifos(self):
		return instrument_set_from_ifos(self.ifos)

CoincInspiralTable.RowType = CoincInspiral


#
# =============================================================================
#
#                             sngl_ringdown:table
#
# =============================================================================
#


SnglRingDownID = ilwd.get_ilwdchar_class(u"sngl_ringdown", u"event_id")


class SnglRingDownTable(table.Table):
	tableName = "sngl_ringdown:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"ifo": "lstring",
		"channel": "lstring",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"start_time_gmst": "real_8",
		"frequency": "real_4",
		"quality": "real_4",
		"phase": "real_4",
		"mass": "real_4",
		"spin": "real_4",
		"epsilon": "real_4",
		"num_clust_trigs": "int_4s",
		"ds2_H1H2": "real_4",
		"ds2_H1L1": "real_4",
		"ds2_H2L1": "real_4",
		"amplitude": "real_4",
		"snr": "real_4",
		"eff_dist": "real_4",
		"sigma_sq": "real_8",
		"event_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (event_id)"
	# FIXME:  ringdown pipeline needs to not encode data in event_id
	#next_id = SnglRingDownID(0)
	interncolumns = ("process_id", "ifo", "search", "channel")


class SnglRingDown(object):
	__slots__ = SnglRingDownTable.validcolumns.keys()

	def get_start(self):
		return LIGOTimeGPS(self.start_time, self.start_time_ns)

	def set_start(self, gps):
		self.start_time, self.start_time_ns = gps.seconds, gps.nanoseconds


SnglRingDownTable.RowType = SnglRingDown


#
# =============================================================================
#
#                             multi_inspiral:table
#
# =============================================================================
#


MultiInspiralID = ilwd.get_ilwdchar_class(u"multi_inspiral", u"event_id")


class MultiInspiralTable(table.Table):
	tableName = "multi_inspiral:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"ifos": "lstring",
		"search": "lstring",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"end_time_gmst": "real_8",
		"impulse_time": "int_4s",
		"impulse_time_ns": "int_4s",
		"amplitude": "real_4",
		"ifo1_eff_distance": "real_4",
		"ifo2_eff_distance": "real_4",
		"eff_distance": "real_4",
		"coa_phase": "real_4",
		"mass1": "real_4",
		"mass2": "real_4",
		"mchirp": "real_4",
		"eta": "real_4",
		"tau0": "real_4",
		"tau2": "real_4",
		"tau3": "real_4",
		"tau4": "real_4",
		"tau5": "real_4",
		"ttotal": "real_4",
		"ifo1_snr": "real_4",
		"ifo2_snr": "real_4",
		"snr": "real_4",
		"chisq": "real_4",
		"chisq_dof": "int_4s",
		"bank_chisq": "real_4",
		"bank_chisq_dof": "int_4s",
		"cont_chisq": "real_4",
		"cont_chisq_dof": "int_4s",
		"sigmasq": "real_4",
		"ligo_axis_ra": "real_4",
		"ligo_axis_dec": "real_4",
		"ligo_angle": "real_4",
		"ligo_angle_sig": "real_4",
		"inclination": "real_4",
		"polarization": "real_4",
		"event_id": "ilwd:char",
		"null_statistic": "real_4",
		"h1quad_re": "real_4",
		"h1quad_im": "real_4",
		"h2quad_re": "real_4",
		"h2quad_im": "real_4",
		"l1quad_re": "real_4",
		"l1quad_im": "real_4",
		"v1quad_re": "real_4",
		"v1quad_im": "real_4",
		"g1quad_re": "real_4",
		"g1quad_im": "real_4",
		"t1quad_re": "real_4",
		"t1quad_im": "real_4"
	}
	constraints = "PRIMARY KEY (event_id)"
	next_id = MultiInspiralID(0)
	interncolumns = ("process_id", "ifos", "search")

	def get_column(self,column):
		return self.getColumnByName(column).asarray()

	def getstat(self):
		return self.get_column('snr')

	def veto(self,seglist):
		vetoed = table.new_from_template(self)
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end()
			if time in seglist:
				vetoed.append(row)
			else:
				keep.append(row)
		return keep

	def vetoed(self, seglist):
		"""
		Return the inverse of what veto returns, i.e., return the triggers
		that lie within a given seglist.
		"""
		vetoed = table.new_from_template(self)
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end()
			if time in seglist:
				vetoed.append(row)
			else:
				keep.append(row)
		return vetoed

	def getslide(self,slide_num):
		"""
		Return the triggers with a specific slide number.
		@param slide_num: the slide number to recover (contained in the event_id)
		"""
		slideTrigs = table.new_from_template(self)
		slideTrigs.extend([row for row in self if row.get_slide_number() == slide_num])
		return slideTrigs

class MultiInspiral(object):
	__slots__ = MultiInspiralTable.validcolumns.keys()

	def get_ifos(self):
		"""
		Return a set of the instruments for this row.
		"""
		return instrument_set_from_ifos(self.ifos)

	def set_ifos(self, instruments):
		"""
		Serialize a sequence of instruments into the ifos
		attribute.  The instrument names must not contain the ","
		character.
		"""
		self.ifos = ifos_from_instrument_set(instruments)

	def get_id_parts(self):
		"""
		Return the three pieces of the int_8s-style event_id.
		"""
		int_event_id = int(self.event_id)
		a = int_event_id // 1000000000
		slidenum = (int_event_id % 1000000000) // 100000
		b = int_event_id % 100000
		return int(a), int(slidenum), int(b)

	def get_slide_number(self):
		"""
		Return the slide-number for this trigger
		"""
		a, slide_number, b = self.get_id_parts()
		if slide_number > 5000:
			slide_number = 5000 - slide_number
		return slide_number

MultiInspiralTable.RowType = MultiInspiral


#
# =============================================================================
#
#                              sim_inspiral:table
#
# =============================================================================
#


SimInspiralID = ilwd.get_ilwdchar_class(u"sim_inspiral", u"simulation_id")


class SimInspiralTable(table.Table):
	tableName = "sim_inspiral:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"waveform": "lstring",
		"geocent_end_time": "int_4s",
		"geocent_end_time_ns": "int_4s",
		"h_end_time": "int_4s",
		"h_end_time_ns": "int_4s",
		"l_end_time": "int_4s",
		"l_end_time_ns": "int_4s",
		"g_end_time": "int_4s",
		"g_end_time_ns": "int_4s",
		"t_end_time": "int_4s",
		"t_end_time_ns": "int_4s",
		"v_end_time": "int_4s",
		"v_end_time_ns": "int_4s",
		"end_time_gmst": "real_8",
		"source": "lstring",
		"mass1": "real_4",
		"mass2": "real_4",
		"mchirp": "real_4",
		"eta": "real_4",
		"distance": "real_4",
		"longitude": "real_4",
		"latitude": "real_4",
		"inclination": "real_4",
		"coa_phase": "real_4",
		"polarization": "real_4",
		"psi0": "real_4",
		"psi3": "real_4",
		"alpha": "real_4",
		"alpha1": "real_4",
		"alpha2": "real_4",
		"alpha3": "real_4",
		"alpha4": "real_4",
		"alpha5": "real_4",
		"alpha6": "real_4",
		"beta": "real_4",
		"spin1x": "real_4",
		"spin1y": "real_4",
		"spin1z": "real_4",
		"spin2x": "real_4",
		"spin2y": "real_4",
		"spin2z": "real_4",
		"theta0": "real_4",
		"phi0": "real_4",
		"f_lower": "real_4",
		"f_final": "real_4",
		"eff_dist_h": "real_4",
		"eff_dist_l": "real_4",
		"eff_dist_g": "real_4",
		"eff_dist_t": "real_4",
		"eff_dist_v": "real_4",
		"numrel_mode_min": "int_4s",
		"numrel_mode_max": "int_4s",
		"numrel_data": "lstring",
		"amp_order": "int_4s",
		"taper": "lstring",
		"bandpass": "int_4s",
		"simulation_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (simulation_id)"
	next_id = SimInspiralID(0)
	interncolumns = ("process_id", "waveform", "source")

	def get_column(self,column):
		if 'chirp_dist' in column:
			site = column[-1]
			return self.get_chirp_dist(site)
		elif column == 'spin1':
			return self.get_spin_mag(1)
		elif column == 'spin2':
			return self.get_spin_mag(2)
		elif column == 'total_mass' or column == 'mtotal':
			m1=self.getColumnByName('mass1').asarray()
			m2=self.getColumnByName('mass2').asarray()
			return m1+m2 
		else:
			return self.getColumnByName(column).asarray()

	def get_chirp_dist(self,site,ref_mass = 1.40):
		mchirp = self.get_column('mchirp')
		eff_dist = self.get_column('eff_dist_' + site)
		return eff_dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)

	def get_spin_mag(self,objectnumber):
		sx = self.get_column('spin' + str(objectnumber) + 'x')
		sy = self.get_column('spin' + str(objectnumber) + 'y')
		sz = self.get_column('spin' + str(objectnumber) + 'z')
		return (sx**2 + sy**2 + sz**2)**(0.5)

	def veto(self,seglist,site=None):
		keep = table.new_from_template(self)
		for row in self:
			time = row.get_end(site)
			if time not in seglist:
				keep.append(row)
		return keep


class SimInspiral(object):
	__slots__ = SimInspiralTable.validcolumns.keys()

	def get_end(self,site = None):
		if not site:
			return LIGOTimeGPS(self.geocent_end_time, self.geocent_end_time_ns)
		else:
			return LIGOTimeGPS(getattr(self,site.lower() + '_end_time'), getattr(self,site.lower() + '_end_time_ns'))


SimInspiralTable.RowType = SimInspiral


#
# =============================================================================
#
#                               sim_burst:table
#
# =============================================================================
#


SimBurstID = ilwd.get_ilwdchar_class(u"sim_burst", u"simulation_id")


class SimBurstTable(table.Table):
	tableName = "sim_burst:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"waveform": "lstring",
		"ra": "real_8",
		"dec": "real_8",
		"psi": "real_8",
		"time_geocent_gps": "int_4s",
		"time_geocent_gps_ns": "int_4s",
		"time_geocent_gmst": "real_8",
		"duration": "real_8",
		"frequency": "real_8",
		"bandwidth": "real_8",
		"q": "real_8",
		"pol_ellipse_angle": "real_8",
		"pol_ellipse_e": "real_8",
		"amplitude": "real_8",
		"hrss": "real_8",
		"egw_over_rsquared": "real_8",
		"waveform_number": "int_8u",
		"simulation_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (simulation_id)"
	next_id = SimBurstID(0)
	interncolumns = ("process_id", "waveform")


class SimBurst(object):
	__slots__ = SimBurstTable.validcolumns.keys()

	def get_time_geocent(self):
		return LIGOTimeGPS(self.time_geocent_gps, self.time_geocent_gps_ns)

	def set_time_geocent(self, gps):
		self.time_geocent_gps, self.time_geocent_gps_ns = gps.seconds, gps.nanoseconds


SimBurstTable.RowType = SimBurst


#
# =============================================================================
#
#                              sim_ringdown:table
#
# =============================================================================
#


SimRingDownID = ilwd.get_ilwdchar_class(u"sim_ringdown", u"simulation_id")


class SimRingDownTable(table.Table):
	tableName = "sim_ringdown:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"waveform": "lstring",
		"coordinates": "lstring",
		"geocent_start_time": "int_4s",
		"geocent_start_time_ns": "int_4s",
		"h_start_time": "int_4s",
		"h_start_time_ns": "int_4s",
		"l_start_time": "int_4s",
		"l_start_time_ns": "int_4s",
		"start_time_gmst": "real_8",
		"longitude": "real_4",
		"latitude": "real_4",
		"distance": "real_4",
		"inclination": "real_4",
		"polarization": "real_4",
		"frequency": "real_4",
		"quality": "real_4",
		"phase": "real_4",
		"mass": "real_4",
		"spin": "real_4",
		"epsilon": "real_4",
		"amplitude": "real_4",
		"eff_dist_h": "real_4",
		"eff_dist_l": "real_4",
		"hrss": "real_4",
		"hrss_h": "real_4",
		"hrss_l": "real_4",
		"simulation_id": "ilwd:char"
	}
	constraints = "PRIMARY KEY (simulation_id)"
	next_id = SimRingDownID(0)
	interncolumns = ("process_id", "waveform", "coordinates")


class SimRingDown(object):
	__slots__ = SimRingDownTable.validcolumns.keys()

	def get_start(self, site = None):
		if not site:
			return LIGOTimeGPS(self.geocent_start_time, self.geocent_start_time_ns)
		else:
			site = site[0].lower()
			return LIGOTimeGPS(getattr(self, site + '_start_time'), getattr(self, site + '_start_time_ns'))


SimRingDownTable.RowType = SimRingDown


#
# =============================================================================
#
#                               summ_value:table
#
# =============================================================================
#


SummValueID = ilwd.get_ilwdchar_class(u"summ_value", u"summ_value_id")


class SummValueTable(table.Table):
	tableName = "summ_value:table"
	validcolumns = {
		"summ_value_id": "ilwd:char",
		"program": "lstring",
		"process_id": "ilwd:char",
		"frameset_group": "lstring",
		"segment_def_id": "ilwd:char",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"ifo": "lstring",
		"name": "lstring",
		"value": "real_4",
		"error": "real_4",
		"intvalue": "int_4s",
		"comment": "lstring"
	}
	constraints = "PRIMARY KEY (summ_value_id)"
	next_id = SummValueID(0)


class SummValue(object):
	__slots__ = SummValueTable.validcolumns.keys()


SummValueTable.RowType = SummValue


#
# =============================================================================
#
#                            sim_inst_params:table
#
# =============================================================================
#


SimInstParamsID = ilwd.get_ilwdchar_class(u"sim_inst_params", u"simulation_id")


class SimInstParamsTable(table.Table):
	tableName = "sim_inst_params:table"
	validcolumns = {
		"simulation_id": "ilwd:char",
		"name": "lstring",
		"comment": "lstring",
		"value": "real_8"
	}
	next_id = SimInstParamsID(0)


class SimInstParams(object):
	__slots__ = SimInstParamsTable.validcolumns.keys()


SimInstParamsTable.RowType = SimInstParams


#
# =============================================================================
#
#                               stochastic:table
#
# =============================================================================
#


class StochasticTable(table.Table):
	tableName = "stochastic:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"ifo_one": "lstring",
		"ifo_two": "lstring",
		"channel_one": "lstring",
		"channel_two": "lstring",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"duration": "int_4s",
		"duration_ns": "int_4s",
		"f_min": "real_8",
		"f_max": "real_8",
		"cc_stat": "real_8",
		"cc_sigma": "real_8"
	}


class Stochastic(object):
	__slots__ = StochasticTable.validcolumns.keys()


StochasticTable.RowType = Stochastic


#
# =============================================================================
#
#                               stochsumm:table
#
# =============================================================================
#


class StochSummTable(table.Table):
	tableName = "stochsumm:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"ifo_one": "lstring",
		"ifo_two": "lstring",
		"channel_one": "lstring",
		"channel_two": "lstring",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"f_min": "real_8",
		"f_max": "real_8",
		"y_opt": "real_8",
		"error": "real_8"
	}


class StochSumm(object):
	__slots__ = StochSummTable.validcolumns.keys()


StochSummTable.RowType = StochSumm


#
# =============================================================================
#
#                            external_trigger:table
#
# =============================================================================
#


# FIXME: this table is completely different from the official definition.
# Someone *HAS* to sort this out.


class ExtTriggersTable(table.Table):
	tableName = "external_trigger:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"det_alts": "lstring",
		"det_band": "lstring",
		"det_fluence": "lstring",
		"det_fluence_int": "lstring",
		"det_name": "lstring",
		"det_peak": "lstring",
		"det_peak_int": "lstring",
		"det_snr": "lstring",
		"email_time": "int_4s",
		"event_dec": "real_4",
		"event_dec_err": "real_4",
		"event_epoch": "lstring",
		"event_err_type": "lstring",
		"event_ra": "real_4",
		"event_ra_err": "real_4",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"event_type": "lstring",
		"event_z": "real_4",
		"event_z_err": "real_4",
		"notice_comments": "lstring",
		"notice_id": "lstring",
		"notice_sequence": "lstring",
		"notice_time": "int_4s",
		"notice_type": "lstring",
		"notice_url": "lstring",
		"obs_fov_dec": "real_4",
		"obs_fov_dec_width": "real_4",
		"obs_fov_ra": "real_4",
		"obs_fov_ra_width": "real_4",
		"obs_loc_ele": "real_4",
		"obs_loc_lat": "real_4",
		"obs_loc_long": "real_4",
		"ligo_fave_lho": "real_4",
		"ligo_fave_llo": "real_4",
		"ligo_delay": "real_4",
		"event_number_gcn": "int_4s",
		"event_number_grb": "lstring",
		"event_status": "int_4s"
	}


class ExtTriggers(object):
	__slots__ = ExtTriggersTable.validcolumns.keys()


ExtTriggersTable.RowType = ExtTriggers


#
# =============================================================================
#
#                                 filter:table
#
# =============================================================================
#


FilterID = ilwd.get_ilwdchar_class(u"filter", u"filter_id")


class FilterTable(table.Table):
	tableName = "filter:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"program": "lstring",
		"start_time": "int_4s",
		"filter_name": "lstring",
		"filter_id": "ilwd:char",
		"param_set": "int_4s",
		"comment": "lstring"
	}
	constraints = "PRIMARY KEY (filter_id)"
	next_id = FilterID(0)


class Filter(object):
	__slots__ = FilterTable.validcolumns.keys()


FilterTable.RowType = Filter


#
# =============================================================================
#
#                                segment:table
#
# =============================================================================
#


SegmentID = ilwd.get_ilwdchar_class(u"segment", u"segment_id")


class SegmentTable(table.Table):
	tableName = "segment:table"
	validcolumns = {
		"creator_db": "int_4s",
		"process_id": "ilwd:char",
		"segment_id": "ilwd:char",
		"start_time": "int_4s",
		"start_time_ns": "int_4s",
		"end_time": "int_4s",
		"end_time_ns": "int_4s",
		"segment_def_id": "ilwd:char",
		"segment_def_cdb": "int_4s",
		"active": "int_4s",
		"segnum": "int_4s",
		"insertion_time": "int_4s"
	}
	constraints = "PRIMARY KEY (segment_id)"
	next_id = SegmentID(0)
	interncolumns = ("process_id",)


class Segment(object):
	__slots__ = SegmentTable.validcolumns.keys()

	def __cmp__(self, other):
		return cmp(self.get(), other.get()) or cmp(self.get_active(), other.get_active())

	def get(self):
		"""
		Return the segment described by this row.
		"""
		return segments.segment(LIGOTimeGPS(self.start_time, self.start_time_ns), LIGOTimeGPS(self.end_time, self.end_time_ns))

	def set(self, segment):
		"""
		Set the segment described by this row.
		"""
		self.start_time, self.start_time_ns = segment[0].seconds, segment[0].nanoseconds
		self.end_time, self.end_time_ns = segment[1].seconds, segment[1].nanoseconds

	def get_active(self):
		"""
		Return True if the segment is active, False if the segment
		is inactive and None if neither is the case.
		"""
		if self.active == 1:
			return True
		if self.active == 0:
			return False
		if self.active is None:
			return None
		raise ValueError, self.active

	def set_active(self, active):
		"""
		Sets the segment to active if active is True, to inactive
		if active if False, and undefined if active is None.
		"""
		if active:
			# Anything that is "True" --> 1
			self.active = 1
		elif active is None:
			# None --> None
			self.active = None
		else:
			# Antyhing that is "False" except None --> 0
			self.active = 0
		return self


SegmentTable.RowType = Segment


#
# =============================================================================
#
#                            segment_def_map:table
#
# =============================================================================
#


SegmentDefMapID = ilwd.get_ilwdchar_class(u"segment_def_map", u"seg_def_map_id")


class SegmentDefMapTable(table.Table):
	tableName = "segment_def_map:table"
	validcolumns = {
		"creator_db": "int_4s",
		"process_id": "ilwd:char",
		"seg_def_map_id": "ilwd:char",
		"segment_cdb": "int_4s",
		"segment_id": "ilwd:char",
		"segment_def_cdb": "int_4s",
		"segment_def_id": "ilwd:char",
		"state_vec_map": "int_4s",
		"insertion_time": "int_4s"
	}
	constraints = "PRIMARY KEY (seg_def_map_id)"
	next_id = SegmentDefMapID(0)
	interncolumns = ("process_id", "segment_def_id")


class SegmentDefMap(object):
	__slots__ = SegmentDefMapTable.validcolumns.keys()


SegmentDefMapTable.RowType = SegmentDefMap


#
# =============================================================================
#
#                            segment_definer:table
#
# =============================================================================
#


SegmentDefID = ilwd.get_ilwdchar_class(u"segment_definer", u"segment_def_id")


class SegmentDefTable(table.Table):
	tableName = "segment_definer:table"
	validcolumns = {
		"creator_db": "int_4s",
		"process_id": "ilwd:char",
		"segment_def_id": "ilwd:char",
		"run": "lstring",
		"ifos": "lstring",
		"name": "lstring",
		"version": "int_4s",
		"comment": "lstring",
		"state_vec_major": "int_4s",
		"state_vec_minor": "int_4s",
		"insertion_time": "int_4s"
	}
	constraints = "PRIMARY KEY (segment_def_id)"
	next_id = SegmentDefID(0)
	interncolumns = ("process_id",)


class SegmentDef(object):
	__slots__ = SegmentDefTable.validcolumns.keys()

	def get_ifos(self):
		"""
		Return a set of the instruments for this row.
		"""
		return instrument_set_from_ifos(self.ifos)

	def set_ifos(self, instruments):
		"""
		Serialize a sequence of instruments into the ifos
		attribute.  The instrument names must not contain the ","
		character.
		"""
		self.ifos = ifos_from_instrument_set(instruments)


SegmentDefTable.RowType = SegmentDef


#
# =============================================================================
#
#                               time_slide:table
#
# =============================================================================
#


TimeSlideID = ilwd.get_ilwdchar_class(u"time_slide", u"time_slide_id")


class TimeSlideTable(table.Table):
	tableName = "time_slide:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"time_slide_id": "ilwd:char",
		"instrument": "lstring",
		"offset": "real_8"
	}
	constraints = "PRIMARY KEY (time_slide_id, instrument)"
	next_id = TimeSlideID(0)
	interncolumns = ("process_id", "time_slide_id", "instrument")

	def get_offset_dict(self, id):
		"""
		Return a dictionary of instrument/offset pairs as described
		by the rows having the given ID.
		"""
		d = {}
		for row in self:
			if row.time_slide_id == id:
				if row.instrument in d:
					raise KeyError, "%s: duplicate instrument %s" % (id, row.instrument)
				d[row.instrument] = row.offset
		if not d:
			raise KeyError, id
		return d

	def as_dict(self):
		"""
		Return a ditionary mapping time slide IDs to offset
		dictionaries.
		"""
		d = {}
		for row in self:
			if row.time_slide_id not in d:
				d[row.time_slide_id] = {}
			if row.instrument in d[row.time_slide_id]:
				raise KeyError, "%s: duplicate instrument %s" % (row.time_slide_id, row.instrument)
			d[row.time_slide_id][row.instrument] = row.offset
		return d

	def get_time_slide_id(self, offsetdict, create_new = None):
		"""
		Return the time_slide_id corresponding to the time slide
		described by offsetdict, a dictionary of instrument/offset
		pairs.  If no matching time_slide_id is found, then
		KeyError is raised.  If, however, the optional create_new
		argument is set to an lsctables.Process object (or any
		other object with a process_id attribute), then new rows
		are added to the table to describe the desired time slide,
		and the ID of the new rows is returned.
		"""
		# look for the ID
		for id, slide in self.as_dict().iteritems():
			if offsetdict == slide:
				# found it
				return id

		# time slide not found in table
		if create_new is None:
			raise KeyError, offsetdict
		id = self.get_next_id()
		for instrument, offset in offsetdict.iteritems():
			row = self.RowType()
			row.process_id = create_new.process_id
			row.time_slide_id = id
			row.instrument = instrument
			row.offset = offset
			self.append(row)

		# return new ID
		return id

	def is_null(self, id):
		"""
		Test that a time slide ID identifies an all-zero time
		slide.
		"""
		for offset in self.get_offset_dict(id).itervalues():
			if offset:
				return False
		return True


class TimeSlide(object):
	__slots__ = TimeSlideTable.validcolumns.keys()


TimeSlideTable.RowType = TimeSlide


#
# =============================================================================
#
#                             coinc_definer:table
#
# =============================================================================
#


CoincDefID = ilwd.get_ilwdchar_class(u"coinc_definer", u"coinc_def_id")


class CoincDefTable(table.Table):
	tableName = "coinc_definer:table"
	validcolumns = {
		"coinc_def_id": "ilwd:char",
		"search": "lstring",
		"search_coinc_type": "int_4u",
		"description": "lstring"
	}
	constraints = "PRIMARY KEY (coinc_def_id)"
	next_id = CoincDefID(0)
	how_to_index = {
		"cd_ssct_index": ("search", "search_coinc_type")
	}

	def get_coinc_def_id(self, search, search_coinc_type, create_new = True, description = u""):
		"""
		Return the coinc_def_id for the row in the table whose
		search string and search_coinc_type integer have the values
		given.  If a matching row is not found, the default
		behaviour is to create a new row and return the ID assigned
		to the new row.  If, instead, create_new is False then
		KeyError is raised when a matching row is not found.  The
		optional description parameter can be used to set the
		description string assigned to the new row if one is
		created, otherwise the new row is left with an empty
		description.
		"""
		# look for the ID
		for row in self:
			if (row.search, row.search_coinc_type) == (search, search_coinc_type):
				# found it
				return row.coinc_def_id

		# coinc type not found in table
		if not create_new:
			raise KeyError, (search, search_coinc_type)
		row = self.RowType()
		row.coinc_def_id = self.get_next_id()
		row.search = search
		row.search_coinc_type = search_coinc_type
		row.description = description
		self.append(row)

		# return new ID
		return row.coinc_def_id


class CoincDef(object):
	__slots__ = CoincDefTable.validcolumns.keys()

	def __init__(self, **kwargs):
		for name, value in kwargs.items():
			setattr(self, name, value)


CoincDefTable.RowType = CoincDef


#
# =============================================================================
#
#                              coinc_event:table
#
# =============================================================================
#


CoincID = ilwd.get_ilwdchar_class(u"coinc_event", u"coinc_event_id")


class CoincTable(table.Table):
	tableName = "coinc_event:table"
	validcolumns = {
		"process_id": "ilwd:char",
		"coinc_def_id": "ilwd:char",
		"coinc_event_id": "ilwd:char",
		"time_slide_id": "ilwd:char",
		"instruments": "lstring",
		"nevents": "int_4u",
		"likelihood": "real_8"
	}
	constraints = "PRIMARY KEY (coinc_event_id)"
	next_id = CoincID(0)
	interncolumns = ("process_id", "coinc_def_id", "time_slide_id", "instruments")
	how_to_index = {
		"ce_cdi_index": ("coinc_def_id",),
		"ce_tsi_index": ("time_slide_id",)
	}


class Coinc(object):
	__slots__ = CoincTable.validcolumns.keys()

	def get_instruments(self):
		"""
		Return a set of the instruments for this row.
		"""
		return instrument_set_from_ifos(self.instruments)

	def set_instruments(self, instruments):
		"""
		Serialize a sequence of instruments into the ifos
		attribute.  The instrument names must not contain the ","
		character.
		"""
		self.instruments = ifos_from_instrument_set(instruments)


CoincTable.RowType = Coinc


#
# =============================================================================
#
#                            coinc_event_map:table
#
# =============================================================================
#


class CoincMapTable(table.Table):
	tableName = "coinc_event_map:table"
	validcolumns = {
		"coinc_event_id": "ilwd:char",
		"table_name": "char_v",
		"event_id": "ilwd:char"
	}
	interncolumns = ("table_name",)
	how_to_index = {
		"cem_tn_ei_index": ("table_name", "event_id"),
		"cem_cei_index": ("coinc_event_id",)
	}


class CoincMap(object):
	__slots__ = CoincMapTable.validcolumns.keys()


CoincMapTable.RowType = CoincMap


#
# =============================================================================
#
#                                dq_list Table
#
# =============================================================================
#


DQSpecListID = ilwd.get_ilwdchar_class(u"dq_list", u"dq_list_id")
DQSpecListRowID = ilwd.get_ilwdchar_class(u"dq_list", u"dq_list_row_id")


class DQSpecListTable(table.Table):
	tableName = "dq_list:table"
	validcolumns = {
		"dq_list_id": "ilwd:char",
		"dq_list_row_id": "ilwd:char",
		"instrument": "lstring",
		"flag": "lstring",
		"low_window": "real_8",
		"high_window": "real_8"
	}
	constraints = "PRIMARY KEY (dq_list_id, dq_list_row_id)"
	next_id = DQSpecListID(0)


class DQSpec(object):
	__slots__ = DQSpecListTable.validcolumns.keys()

	def apply_to_segmentlist(self, seglist):
		"""
		Apply our low and high windows to the segments in a
		segmentlist.
		"""
		for i, seg in enumerate(seglist):
			seglist[i] = seg.__class__(seg[0] - self.low_window, seg[1] + self.high_window)


DQSpecListTable.RowType = DQSpec


#
# =============================================================================
#
#                               ligolw_mon:table
#
# =============================================================================
#


LIGOLWMonID = ilwd.get_ilwdchar_class(u"ligolw_mon", u"event_id")


class LIGOLWMonTable(table.Table):
	tableName = "ligolw_mon:table"
	validcolumns = {
		"creator_db": "int_4s",
		"process_id": "ilwd:char",
		"time": "int_4s",
		"time_ns": "int_4s",
		"amplitude": "real_8",
		"confidence": "real_8",
		"frequency": "real_8",
		"event_id": "ilwd:char",
		"insertion_time": "int_4s"
	}
	constraints = "PRIMARY KEY (event_id)"
	next_id = LIGOLWMonID(0)


class LIGOLWMon(object):
	__slots__ = LIGOLWMonTable.validcolumns.keys()

	def get_time(self):
		return LIGOTimeGPS(self.time, self.time_ns)

	def set_time(self, gps):
		self.time, self.time_ns = gps.seconds, gps.nanoseconds


LIGOLWMonTable.RowType = LIGOLWMon


#
# =============================================================================
#
#                                Table Metadata
#
# =============================================================================
#


#
# Table name ---> table type mapping.
#


TableByName = {
	table.StripTableName(ProcessTable.tableName): ProcessTable,
	table.StripTableName(LfnTable.tableName): LfnTable,
	table.StripTableName(ProcessParamsTable.tableName): ProcessParamsTable,
	table.StripTableName(SearchSummaryTable.tableName): SearchSummaryTable,
	table.StripTableName(SearchSummVarsTable.tableName): SearchSummVarsTable,
	table.StripTableName(SnglBurstTable.tableName): SnglBurstTable,
	table.StripTableName(MultiBurstTable.tableName): MultiBurstTable,
	table.StripTableName(SnglInspiralTable.tableName): SnglInspiralTable,
	table.StripTableName(CoincInspiralTable.tableName): CoincInspiralTable,
	table.StripTableName(SnglRingDownTable.tableName): SnglRingDownTable,
	table.StripTableName(MultiInspiralTable.tableName): MultiInspiralTable,
	table.StripTableName(SimInspiralTable.tableName): SimInspiralTable,
	table.StripTableName(SimBurstTable.tableName): SimBurstTable,
	table.StripTableName(SimRingDownTable.tableName): SimRingDownTable,
	table.StripTableName(SummValueTable.tableName): SummValueTable,
	table.StripTableName(SimInstParamsTable.tableName): SimInstParamsTable,
	table.StripTableName(StochasticTable.tableName): StochasticTable,
	table.StripTableName(StochSummTable.tableName): StochSummTable,
	table.StripTableName(ExtTriggersTable.tableName): ExtTriggersTable,
	table.StripTableName(FilterTable.tableName): FilterTable,
	table.StripTableName(SegmentTable.tableName): SegmentTable,
	table.StripTableName(SegmentDefMapTable.tableName): SegmentDefMapTable,
	table.StripTableName(SegmentDefTable.tableName): SegmentDefTable,
	table.StripTableName(TimeSlideTable.tableName): TimeSlideTable,
	table.StripTableName(CoincDefTable.tableName): CoincDefTable,
	table.StripTableName(CoincTable.tableName): CoincTable,
	table.StripTableName(CoincMapTable.tableName): CoincMapTable,
	table.StripTableName(DQSpecListTable.tableName): DQSpecListTable,
	table.StripTableName(LIGOLWMonTable.tableName): LIGOLWMonTable
}


#
# =============================================================================
#
#                               Content Handler
#
# =============================================================================
#


#
# Override portions of the ligolw.LIGOLWContentHandler class
#


__parent_startTable = ligolw.LIGOLWContentHandler.startTable


def startTable(self, attrs):
	name = table.StripTableName(attrs[u"Name"])
	if name in TableByName:
		return TableByName[name](attrs)
	return __parent_startTable(self, attrs)


ligolw.LIGOLWContentHandler.startTable = startTable

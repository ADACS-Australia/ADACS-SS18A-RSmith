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


"""
This module provides an implementation of the Table element that uses a
database engine for storage.  On top of that it then re-implements a number
of the tables from the lsctables module to provide versions of their
methods that work against the SQL database.

*** CAUTION *** the API exported by this module is NOT STABLE.  The code
works well, and it hugely simplifies my life and it can yours too, but if
you use it you will have to be willing to track changes as I make them.
I'm still figuring out how this should work.
"""


import os
import re
import shutil
import sys
import tempfile
from xml.sax.xmlreader import AttributesImpl
# Python 2.3 compatibility
try:
	set
except NameError:
	from sets import Set as set

import ligolw
import table
import lsctables
import types
from glue import segments


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"[7:-2]
__version__ = "$Revision$"[11:-2]


#
# =============================================================================
#
#                                  Connection
#
# =============================================================================
#


def DBTable_set_connection(connection):
	"""
	Set the Python DB-API 2.0 compatible connection the DBTable class
	will use.
	"""
	DBTable.connection = connection


def DBTable_get_connection():
	"""
	Return the current connection object.
	"""
	return DBTable.connection


def DBTable_commit():
	"""
	Run commit on the DBTable class' connection.
	"""
	DBTable.connection.commit()


def get_connection_filename(filename, tmp_path = None, replace_file = False, verbose = False):
	"""
	Experimental utility code for moving database files to a
	(presumably local) working location for improved performance and
	reduced fileserver load.  The API is not stable, don't use unless
	you're prepared to track changes.
	"""
	def truncate(filename, verbose = False):
		if verbose:
			print >>sys.stderr, "%s exists, truncating ..." % filename
		try:
			fd = os.open(filename, os.O_WRONLY | os.O_TRUNC)
		except:
			raise e, "cannot truncate %s" % filename
		os.close(fd)
		if verbose:
			print >>sys.stderr, "done."

	database_exists = os.access(filename, os.F_OK)

	if tmp_path:
		target = tempfile.mkstemp(suffix = ".sqlite", dir = tmp_path)[1]
		if database_exists:
			if replace_file:
				# truncate database so that if this job
				# fails the user won't think the database
				# file is valid
				truncate(filename, verbose = verbose)
			else:
				# need to copy existing database to work
				# space for new inserts
				if verbose:
					print >>sys.stderr, "copying %s to %s ..." % (filename, target)
				shutil.copy(filename, target)
	else:
		target = filename
		if database_exists and replace_file:
			truncate(target, verbose = verbose)

	return target


def put_connection_filename(filename, working_filename, verbose = False):
	if working_filename != filename:
		if verbose:
			print >>sys.stderr, "moving %s to %s ..." % (working_filename, filename)
		shutil.move(working_filename, filename)


#
# =============================================================================
#
#                                  ID Mapping
#
# =============================================================================
#


def DBTable_idmap_create():
	"""
	Create the _idmap_ table.  This table has columns "old" and "new"
	containing text strings mapping old IDs to new IDs.  The old column
	is a primary key (is indexed and must contain unique entries).  The
	table is created as a temporary table, so it will be automatically
	dropped when the database connection is closed.
	"""
	DBTable.connection.cursor().execute("CREATE TEMPORARY TABLE _idmap_ (old TEXT PRIMARY KEY, new TEXT)")


def DBTable_idmap_reset():
	"""
	Erase the contents of the _idmap_ table.
	"""
	DBTable.connection.cursor().execute("DELETE FROM _idmap_")


def DBTable_idmap_get_new(old, ids):
	"""
	From the old ID string, obtain a replacement ID string by either
	grabbing it from the _idmap_ table if one has already been assigned
	to the old ID, or by creating a new one with the given ILWD
	generator.  In the latter case, the newly-generated ID is recorded
	in the _idmap_ table.  For internal use only.
	"""
	cursor = DBTable.connection.cursor()
	new = cursor.execute("SELECT new FROM _idmap_ WHERE old == ?", (old,)).fetchone()
	if new is not None:
		return new[0]
	new = ids.next()
	cursor.execute("INSERT INTO _idmap_ VALUES (?, ?)", (old, new))
	return new


#
# =============================================================================
#
#                             Database Information
#
# =============================================================================
#


#
# SQL parsing
#


_sql_create_table_pattern = re.compile(r"CREATE\s+TABLE\s+(?P<name>\w+)\s*\((?P<coldefs>.*)\)", re.IGNORECASE)
_sql_coldef_pattern = re.compile(r"\s*(?P<name>\w+)\s+(?P<type>\w+)[^,]*")


#
# Database info extraction utils
#


def DBTable_table_names():
	"""
	Return a list of the table names in the database.
	"""
	return [name for (name,) in DBTable.connection.cursor().execute("SELECT name FROM sqlite_master WHERE type == 'table'")]


def DBTable_column_info(table_name):
	"""
	Return an in order list of (name, type) tuples describing the
	columns for the given table.
	"""
	statement, = DBTable.connection.cursor().execute("SELECT sql FROM sqlite_master WHERE type == 'table' AND name == ?", (table_name,)).fetchone()
	coldefs = re.match(_sql_create_table_pattern, statement).groupdict()["coldefs"]
	return [(coldef.groupdict()["name"], coldef.groupdict()["type"]) for coldef in re.finditer(_sql_coldef_pattern, coldefs) if coldef.groupdict()["name"] not in ("PRIMARY", "UNIQUE", "CHECK")]


def DBTable_get_xml():
	"""
	Construct an XML document tree wrapping around the contents of the
	currently connected database.  On success the return value is a
	ligolw.LIGO_LW element containing the tables as children.
	"""
	ligo_lw = ligolw.LIGO_LW()
	for table_name in DBTable_table_names():
		# build the table document tree.  copied from
		# lsctables.New()
		try:
			cls = TableByName[table_name]
		except KeyError:
			cls = DBTable
		table_elem = cls(AttributesImpl({u"Name": table_name + ":table"}))
		colnamefmt = table_name + ":%s"
		for column_name, column_type in DBTable_column_info(table_elem.dbtablename):
			if table_elem.validcolumns is not None:
				# use the pre-defined column type
				column_type = table_elem.validcolumns[column_name]
			else:
				# guess the column type
				column_type = types.FromSQLiteType[column_type]
			column_name = colnamefmt % column_name
			table_elem.appendChild(table.Column(AttributesImpl({u"Name": column_name, u"Type": column_type})))

		table_elem._end_of_columns()
		table_elem.appendChild(table.TableStream(AttributesImpl({u"Name": table_name + ":table"})))
		ligo_lw.appendChild(table_elem)
	return ligo_lw


#
# =============================================================================
#
#                            DBTable Element Class
#
# =============================================================================
#


class DBTable(table.Table):
	"""
	A special version of the Table class using an SQL database for
	storage.  Many of the features of the Table class are not available
	here, but instead the user can use SQL to query the table's
	contents.  Before use, the connection attribute must be set to a
	Python DB-API 2.0 "connection" object.  The constraints attribute
	can be set to a text string that will be added to the table's
	CREATE statement where constraints go, for example you might wish
	to set this to "PRIMARY KEY (event_id)" for a table with an
	event_id column.

	Note:  because the table is stored in an SQL database, the use of
	this class imposes the restriction that table names be unique
	within a document.

	Also note that at the present time there is really only proper
	support for the pre-defined tables in the lsctables module.  It is
	possible to load unrecognized tables into a database from LIGO
	Light Weight XML files, but without developer intervention there is
	no way to indicate the constraints that should be imposed on the
	columns, for example which columns should be used as primary keys
	and so on.  This can result in poor query performance.  It is also
	possible to write a database' contents to a LIGO Light Weight XML
	file even when the database contains unrecognized tables, but
	without developer intervention the column types will be guessed.
	"""
	#
	# Global Python DB-API 2.0 connection object shared by all code.
	#

	connection = None

	def __init__(self, *attrs):
		"""
		Initialize
		"""
		table.Table.__init__(self, *attrs)
		self.dbtablename = table.StripTableName(self.getAttribute(u"Name"))
		try:
			# try to find info in lsctables module
			cls = lsctables.TableByName[self.dbtablename]
		except KeyError:
			# unknown table
			pass
		else:
			# copy metadata from lsctables
			self.tableName = cls.tableName
			self.validcolumns = cls.validcolumns
			self.constraints = cls.constraints
			self.ids = cls.ids
			self.RowType = cls.RowType
			self.how_to_index = cls.how_to_index
		if self.connection is None:
			raise ligolw.ElementError, "connection attribute not set"
		self.cursor = self.connection.cursor()

	def _end_of_columns(self):
		table.Table._end_of_columns(self)
		# dbcolumnnames and types have the "not loaded" columns
		# removed
		if self.loadcolumns is not None:
			self.dbcolumnnames = [name for name in self.columnnames if name in self.loadcolumns]
			self.dbcolumntypes = [name for i, name in enumerate(self.columntypes) if self.columnnames[i] in self.loadcolumns]
		else:
			self.dbcolumnnames = self.columnnames
			self.dbcolumntypes = self.columntypes

		# create the table
		statement = "CREATE TABLE IF NOT EXISTS " + self.dbtablename + " (" + ", ".join(map(lambda n, t: "%s %s" % (n, types.ToSQLiteType[t]), self.dbcolumnnames, self.dbcolumntypes))
		if self.constraints is not None:
			statement += ", " + self.constraints
		statement += ")"
		self.cursor.execute(statement)

		# record the highest internal row ID
		self.last_maxrowid = self.maxrowid() or 0

		# construct the SQL to be used to insert new rows
		self.append_statement = "INSERT INTO " + self.dbtablename + " VALUES (" + ",".join("?" * len(self.dbcolumnnames)) + ")"

	def _end_of_rows(self):
		# FIXME:  is this needed?
		table.Table._end_of_rows(self)
		self.connection.commit()

	def sync_ids(self):
		if self.ids is not None:
			last = self.cursor.execute("SELECT MAX(CAST(SUBSTR(%s, %d, 10) AS INTEGER)) FROM %s" % (self.ids.column_name, self.ids.index_offset + 1, self.dbtablename)).fetchone()[0]
			if last is None:
				self.ids.set_next(0)
			else:
				self.ids.set_next(last + 1)
		return self.ids

	def maxrowid(self):
		return self.cursor.execute("SELECT MAX(ROWID) FROM %s" % self.dbtablename).fetchone()[0]

	def __len__(self):
		return self.cursor.execute("SELECT COUNT(*) FROM " + self.dbtablename).fetchone()[0]

	def __iter__(self):
		for values in self.connection.cursor().execute("SELECT * FROM " + self.dbtablename):
			yield self._row_from_cols(values)

	def _append(self, row):
		# FIXME: in Python 2.5 use attrgetter() for attribute
		# tuplization.
		self.cursor.execute(self.append_statement, map(lambda n: getattr(row, n), self.dbcolumnnames))

	def _remapping_append(self, row):
		"""
		Replacement for the standard append() method.  This version
		performs on the fly row ID reassignment, and so also
		performs the function of the updateKeyMapping() method.
		SQL does not permit the PRIMARY KEY of a row to be
		modified, so it needs to be done prior to insertion.  This
		method is intended for internal use only.
		"""
		if self.ids is not None:
			# assign (and record) a new ID before inserting the
			# row to avoid collisions with existing rows
			setattr(row, self.ids.column_name, DBTable_idmap_get_new(getattr(row, self.ids.column_name), self.ids))
		# FIXME: in Python 2.5 use attrgetter() for attribute
		# tuplization.
		self.cursor.execute(self.append_statement, map(lambda n: getattr(row, n), self.dbcolumnnames))

	append = _append

	def _row_from_cols(self, values):
		"""
		Given an iterable of values in the order of columns in the
		database, construct and return a row object.  This is a
		convenience function for turning the results of database
		queries into Python objects.
		"""
		row = self.RowType()
		for c, v in zip(self.dbcolumnnames, values):
			setattr(row, c, v)
		return row

	def unlink(self):
		table.Table.unlink(self)
		self.cursor = None

	def applyKeyMapping(self):
		"""
		Used as the second half of the key reassignment algorithm.
		Loops over each row in the table, replacing references to
		old row keys with the new values from the _idmap_ table.
		"""
		assignments = []
		for colname in [colname for coltype, colname in zip(self.dbcolumntypes, self.dbcolumnnames) if coltype in types.IDTypes and (self.ids is None or colname != self.ids.column_name)]:
			assignments.append("%s = (SELECT new FROM _idmap_ WHERE old == %s)" % (colname, colname))
		if not assignments:
			# table has no columns to update
			return
		# SQLite documentation says ROWID is monotonically
		# increasing starting at 1 for the first row unless it ever
		# wraps around, then it is randomly assigned.  ROWID is a
		# 64 bit integer, so the only way it will wrap is if
		# somebody sets it to a very high number manually.  This
		# library does not do that, so I don't bother checking.
		statement = "UPDATE " + self.dbtablename + " SET " + ", ".join(assignments) + " WHERE ROWID > %d" % self.last_maxrowid
		self.cursor.execute(statement)
		self.last_maxrowid = self.maxrowid() or 0


#
# =============================================================================
#
#                                  LSC Tables
#
# =============================================================================
#


class ProcessTable(DBTable):
	tableName = lsctables.ProcessTable.tableName
	validcolumns = lsctables.ProcessTable.validcolumns
	constraints = lsctables.ProcessTable.constraints
	ids = lsctables.ProcessTable.ids
	RowType = lsctables.ProcessTable.RowType
	how_to_index = lsctables.ProcessTable.how_to_index

	def get_ids_by_program(self, program):
		"""
		Return a set of the process IDs from rows whose program
		string equals the given program.
		"""
		return set([id for (id,) in self.cursor.execute("SELECT process_id FROM process WHERE program == ?", (program,))])


class ProcessParamsTable(DBTable):
	tableName = lsctables.ProcessParamsTable.tableName
	validcolumns = lsctables.ProcessParamsTable.validcolumns
	constraints = lsctables.ProcessParamsTable.constraints
	ids = lsctables.ProcessParamsTable.ids
	RowType = lsctables.ProcessParamsTable.RowType
	how_to_index = lsctables.ProcessParamsTable.how_to_index

	def append(self, row):
		if row.type not in types.Types:
			raise ligolw.ElementError, "unrecognized type '%s'" % row.type
		DBTable.append(self, row)


class SearchSummaryTable(DBTable):
	tableName = lsctables.SearchSummaryTable.tableName
	validcolumns = lsctables.SearchSummaryTable.validcolumns
	constraints = lsctables.SearchSummaryTable.constraints
	ids = lsctables.SearchSummaryTable.ids
	RowType = lsctables.SearchSummaryTable.RowType
	how_to_index = lsctables.SearchSummaryTable.how_to_index

	def get_out_segmentlistdict(self, process_ids = None):
		"""
		Return a segmentlistdict mapping instrument to out segment
		list.  If process_ids is a list of process IDs, then only
		rows with matching IDs are included otherwise all rows are
		included.
		"""
		seglists = segments.segmentlistdict()
		for row in self:
			if process_ids is None or row.process_id in process_ids:
				if "," in row.ifos:
					ifos = map(str.strip, row.ifos.split(","))
				elif "+" in row.ifos:
					ifos = map(str.strip, row.ifos.split("+"))
				else:
					ifos = [row.ifos]
				seglists |= segments.segmentlistdict([(ifo, segments.segmentlist([row.get_out()])) for ifo in ifos])
		return seglists


class SnglBurstTable(DBTable):
	tableName = lsctables.SnglBurstTable.tableName
	validcolumns = lsctables.SnglBurstTable.validcolumns
	constraints = lsctables.SnglBurstTable.constraints
	ids = lsctables.SnglBurstTable.ids
	RowType = lsctables.SnglBurstTable.RowType
	how_to_index = lsctables.SnglBurstTable.how_to_index

class SimBurstTable(DBTable):
	tableName = lsctables.SimBurstTable.tableName
	validcolumns = lsctables.SimBurstTable.validcolumns
	constraints = lsctables.SimBurstTable.constraints
	ids = lsctables.SimBurstTable.ids
	RowType = lsctables.SimBurstTable.RowType
	how_to_index = lsctables.SimBurstTable.how_to_index

class SnglInspiralTable(DBTable):
	tableName = lsctables.SnglInspiralTable.tableName
	validcolumns = lsctables.SnglInspiralTable.validcolumns
	constraints = lsctables.SnglInspiralTable.constraints
	ids = lsctables.SnglInspiralTable.ids
	RowType = lsctables.SnglInspiralTable.RowType
	how_to_index = lsctables.SnglInspiralTable.how_to_index

class SimInspiralTable(DBTable):
	tableName = lsctables.SimInspiralTable.tableName
	validcolumns = lsctables.SimInspiralTable.validcolumns
	constraints = lsctables.SimInspiralTable.constraints
	ids = lsctables.SimInspiralTable.ids
	RowType = lsctables.SimInspiralTable.RowType
	how_to_index = lsctables.SimInspiralTable.how_to_index

class TimeSlideTable(DBTable):
	tableName = lsctables.TimeSlideTable.tableName
	validcolumns = lsctables.TimeSlideTable.validcolumns
	constraints = lsctables.TimeSlideTable.constraints
	ids = lsctables.TimeSlideTable.ids
	RowType = lsctables.TimeSlideTable.RowType
	how_to_index = lsctables.TimeSlideTable.how_to_index

	def __len__(self):
		raise NotImplementedError

	def __getitem__(*args):
		raise NotImplementedError

	def get_offset_dict(self, id):
		offsets = dict(self.cursor.execute("SELECT instrument, offset FROM time_slide WHERE time_slide_id == ?", (id,)))
		if not offsets:
			raise KeyError, id
		return offsets

	def as_dict(self):
		"""
		Return a ditionary mapping time slide IDs to offset
		dictionaries.
		"""
		d = {}
		for id, instrument, offset in self.cursor.execute("SELECT time_slide_id, instrument, offset FROM time_slide"):
			if id not in d:
				d[id] = {}
			d[id][instrument] = offset
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
		id = self.sync_ids().next()
		for instrument, offset in offsetdict.iteritems():
			row = self.RowType()
			row.process_id = create_new.process_id
			row.time_slide_id = id
			row.instrument = instrument
			row.offset = offset
			self.append(row)

		# return new ID
		return id

	def iterkeys(self):
		raise NotImplementedError

	def is_null(self, id):
		return not self.cursor.execute("SELECT EXISTS (SELECT * FROM time_slide WHERE time_slide_id == ? AND offset != 0.0)", (id,)).fetchone()[0]


class CoincDefTable(DBTable):
	tableName = lsctables.CoincDefTable.tableName
	validcolumns = lsctables.CoincDefTable.validcolumns
	constraints = lsctables.CoincDefTable.constraints
	ids = lsctables.CoincDefTable.ids
	RowType = lsctables.CoincDefTable.RowType
	how_to_index = lsctables.CoincDefTable.how_to_index

	def as_dict(self):
		"""
		Return a dictionary mapping coinc_def_id to sorted lists of
		contributing table names.
		"""
		d = {}
		for id, table_name in self.cursor.execute("SELECT coinc_def_id, table_name FROM coinc_definer"):
			if id not in d:
				d[id] = []
			d[id].append(table_name)
		for l in d.itervalues():
			l.sort()
		return d

	def get_coinc_def_id(self, table_names, create_new = True):
		"""
		Return the coinc_def_id corresponding to coincidences
		consisting exclusively of events from the given table
		names.  If no matching coinc_def_id is found, then a new
		one is created and the ID returned, unless create_new is
		False in which case the KeyError is raised.
		"""
		# sort the contributor table names
		table_names = list(table_names)
		table_names.sort()

		# look for the ID
		for id, names in self.as_dict().iteritems():
			if names == table_names:
				# found it
				return id

		# contributor list not found in table
		if not create_new:
			raise KeyError, table_names
		id = self.sync_ids().next()
		for name in table_names:
			row = self.RowType()
			row.coinc_def_id = id
			row.table_name = name
			row.description = u""
			self.append(row)

		# return new ID
		return id

	def get_description(self, coinc_def_id):
		"""
		Get the description string for the given coinc_def_id.
		"""
		return self.cursor.execute("SELECT description FROM coinc_definer WHERE coinc_def_id == ?", (coinc_def_id,)).fetchone()[0]

	def set_description(self, coinc_def_id, description):
		"""
		Set the description string for the given coinc_def_id.
		"""
		self.cursor.execute("UPDATE coinc_definer SET description = ? WHERE coinc_def_id == ?", (description, coinc_def_id))


class CoincTable(DBTable):
	tableName = lsctables.CoincTable.tableName
	validcolumns = lsctables.CoincTable.validcolumns
	constraints = lsctables.CoincTable.constraints
	ids = lsctables.CoincTable.ids
	RowType = lsctables.CoincTable.RowType
	how_to_index = lsctables.CoincTable.how_to_index


class CoincMapTable(DBTable):
	tableName = lsctables.CoincMapTable.tableName
	validcolumns = lsctables.CoincMapTable.validcolumns
	constraints = lsctables.CoincMapTable.constraints
	ids = lsctables.CoincMapTable.ids
	RowType = lsctables.CoincMapTable.RowType
	how_to_index = lsctables.CoincMapTable.how_to_index


class MultiBurstTable(DBTable):
	tableName = lsctables.MultiBurstTable.tableName
	validcolumns = lsctables.MultiBurstTable.validcolumns
	constraints = lsctables.MultiBurstTable.constraints
	ids = lsctables.MultiBurstTable.ids
	RowType = lsctables.MultiBurstTable.RowType
	how_to_index = lsctables.MultiBurstTable.how_to_index


class SegmentTable(DBTable):
	tableName = lsctables.SegmentTable.tableName
	validcolumns = lsctables.SegmentTable.validcolumns
	constraints = lsctables.SegmentTable.constraints
	ids = lsctables.SegmentTable.ids
	RowType = lsctables.SegmentTable.RowType
	how_to_index = lsctables.SegmentTable.how_to_index


class SegmentDefMapTable(DBTable):
	tableName = lsctables.SegmentDefMapTable.tableName
	validcolumns = lsctables.SegmentDefMapTable.validcolumns
	constraints = lsctables.SegmentDefMapTable.constraints
	ids = lsctables.SegmentDefMapTable.ids
	RowType = lsctables.SegmentDefMapTable.RowType
	how_to_index = lsctables.SegmentDefMapTable.how_to_index


class SegmentDefTable(DBTable):
	tableName = lsctables.SegmentDefTable.tableName
	validcolumns = lsctables.SegmentDefTable.validcolumns
	constraints = lsctables.SegmentDefTable.constraints
	ids = lsctables.SegmentDefTable.ids
	RowType = lsctables.SegmentDefTable.RowType
	how_to_index = lsctables.SegmentDefTable.how_to_index


#
# =============================================================================
#
#                                Table Metadata
#
# =============================================================================
#


def build_indexes(verbose = False):
	"""
	Using the how_to_index annotations in the table class definitions,
	construct a set of indexes for the database at the current
	connection.
	"""
	cursor = DBTable_get_connection().cursor()
	for table_name in DBTable_table_names():
		try:
			# FIXME:  figure out how to do this extensibly
			how_to_index = TableByName[table_name].how_to_index
		except KeyError:
			continue
		if how_to_index:
			if verbose:
				print >>sys.stderr, "indexing %s table ..." % table_name
			for index_name, cols in how_to_index.iteritems():
				cursor.execute("CREATE INDEX IF NOT EXISTS %s ON %s (%s)" % (index_name, table_name, ",".join(cols)))


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
	table.StripTableName(ProcessParamsTable.tableName): ProcessParamsTable,
	table.StripTableName(SearchSummaryTable.tableName): SearchSummaryTable,
	table.StripTableName(SnglBurstTable.tableName): SnglBurstTable,
	table.StripTableName(SimBurstTable.tableName): SimBurstTable,
	table.StripTableName(SnglInspiralTable.tableName): SnglInspiralTable,
	table.StripTableName(SimInspiralTable.tableName): SimInspiralTable,
	table.StripTableName(TimeSlideTable.tableName): TimeSlideTable,
	table.StripTableName(CoincDefTable.tableName): CoincDefTable,
	table.StripTableName(CoincTable.tableName): CoincTable,
	table.StripTableName(CoincMapTable.tableName): CoincMapTable,
	table.StripTableName(MultiBurstTable.tableName): MultiBurstTable,
	table.StripTableName(SegmentTable.tableName): SegmentTable,
	table.StripTableName(SegmentDefMapTable.tableName): SegmentDefMapTable,
	table.StripTableName(SegmentDefTable.tableName): SegmentDefTable
}


#
# The database-backed table implementation requires there to be no more
# than one table of each name in the document.  Some documents require
# multiple tables with the same name, and those tables cannot be stored in
# the database.  Use this list to set which tables are not to be stored in
# the database.
#


NonDBTableNames = []


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
	if name in map(table.StripTableName, NonDBTableNames):
		return __parent_startTable(self, attrs)
	if name in TableByName:
		return TableByName[name](attrs)
	return DBTable(attrs)


ligolw.LIGOLWContentHandler.startTable = startTable

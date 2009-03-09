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
A collection of utilities to assist applications in manipulating the
process and process_params tables in LIGO Light-Weight XML documents.
"""


import os
import socket
import StringIO
import time


from glue import gpstime
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import types as ligolwtypes


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>, Larne Pekowsky <lppekows@physics.syr.edu>"
__version__ = "$Revision$"[11:-2]
__date__ = "$Date$"[7:-2]


#
# =============================================================================
#
#                               Process Metadata
#
# =============================================================================
#


def append_process(xmldoc, program = None, version = None, cvs_repository = None, cvs_entry_time = None, comment = None, is_online = False, jobid = 0, domain = None, ifos = None):
	"""
	Add an entry to the process table in xmldoc.  program, version,
	cvs_repository, comment, domain, and ifos should all be strings or
	unicodes.  cvs_entry_time should be a string or unicode in the
	format "YYYY/MM/DD HH:MM:SS".  is_online should be a boolean, jobid
	an integer.
	"""
	try:
		proctable = table.get_table(xmldoc, lsctables.ProcessTable.tableName)
	except ValueError:
		proctable = lsctables.New(lsctables.ProcessTable)
		xmldoc.childNodes[0].appendChild(proctable)

	proctable.sync_next_id()

	process = proctable.RowType()
	process.program = program
	process.version = version
	process.cvs_repository = cvs_repository
	if cvs_entry_time is not None:
		process.cvs_entry_time = gpstime.GpsSecondsFromPyUTC(time.mktime(time.strptime(cvs_entry_time, "%Y/%m/%d %H:%M:%S")))
	else:
		process.cvs_entry_time = None
	process.comment = comment
	process.is_online = int(is_online)
	process.node = socket.gethostname()
	process.username = os.environ["LOGNAME"]
	process.unix_procid = os.getpid()
	process.start_time = gpstime.GpsSecondsFromPyUTC(time.time())
	process.end_time = None
	process.jobid = jobid
	process.domain = domain
	process.set_ifos(ifos)
	process.process_id = proctable.get_next_id()
	proctable.append(process)
	return process


def set_process_end_time(process):
	"""
	Set the end time in a row in a process table to the current time.
	"""
	process.end_time = gpstime.GpsSecondsFromPyUTC(time.time())
	return process


def append_process_params(xmldoc, process, params):
	"""
	xmldoc is an XML document tree, process is the row in the process
	table for which these are the parameters, and params is a list of
	(name, type, value) tuples one for each parameter.
	"""
	try:
		paramtable = table.get_table(xmldoc, lsctables.ProcessParamsTable.tableName)
	except ValueError:
		paramtable = lsctables.New(lsctables.ProcessParamsTable)
		xmldoc.childNodes[0].appendChild(paramtable)

	for name, type, value in params:
		row = paramtable.RowType()
		row.program = process.program
		row.process_id = process.process_id
		row.param = unicode(name)
		if type is not None:
			row.type = unicode(type)
			if row.type not in ligolwtypes.Types:
				raise ValueError, "invalid type '%s' for parameter '%s'" % (row.type, row.param)
		else:
			row.type = None
		if value is not None:
			row.value = unicode(value)
		else:
			row.value = None
		paramtable.append(row)
	return process


def get_process_params(xmldoc, program, param):
	"""
	Return a list of the values stored in the process_params table for
	params named param for the program named program.  Raises
	ValueError if not exactly one program by that name is listed in the
	document.  The values are returned as Python native types, not as
	the strings appearing in the XML document.
	"""
	process_ids = table.get_table(xmldoc, lsctables.ProcessTable.tableName).get_ids_by_program(program)
	if len(process_ids) != 1:
		raise ValueError, "process table must contain exactly one program named '%s'" % program
	return [row.get_pyvalue() for row in table.get_table(xmldoc, lsctables.ProcessParamsTable.tableName) if (row.process_id in process_ids) and (row.param == param)]


def doc_includes_process(xmldoc, program):
	"""
	Return True if the process table in xmldoc includes entries for a
	program named program.
	"""
	return program in table.get_table(xmldoc, lsctables.ProcessTable.tableName).getColumnByName("program")


def register_to_xmldoc(xmldoc, program, options, version = None, cvs_date = None):
	"""
	Register the current process and params to an xml document
	"""
	process = append_process(xmldoc, program = program, version = version, cvs_entry_time = cvs_date)	
	params  = map(lambda key:(key, 'lstring', options.__dict__[key]), filter(lambda x: options.__dict__[x], options.__dict__))
	append_process_params(xmldoc, process, params)

	return process.process_id


# The tables in the segment database declare most fields "NOT NULL", so provide stub values
def register_to_ldbd(client, program, options, version = '0', cvs_repository = '-', cvs_entry_time = 0, comment = '-', is_online = False, jobid = 0, domain = None, ifos = '-'):
	"""
	Register the current process and params to a database via a LDBDClient
	"""
	xmldoc = ligolw.Document()
	xmldoc.appendChild(ligolw.LIGO_LW())

	process = append_process(xmldoc, program, version, cvs_repository, cvs_entry_time, comment, is_online, jobid, domain, ifos)
	params  = map(lambda key:(key, 'lstring', options.__dict__[key]), filter(lambda x: options.__dict__[x], options.__dict__))
	append_process_params(xmldoc, process, params)

	fake_file = StringIO.StringIO()
	xmldoc.write(fake_file)
	client.insert(fake_file.getvalue())

	return process.process_id

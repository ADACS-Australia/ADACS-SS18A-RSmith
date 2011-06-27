# Copyright (C) 2006  Kipp Cannon
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
DOM-like library for handling LIGO Light Weight XML files.  For more
information on the Python DOM specification and SAX document content
handlers, please refer to the Python standard library reference and the
documentation it links to.

The most important thing to understand about the glue.ligolw package is
that the import of most modules has side effects.  Due to a bad design
decision for which I (Kipp) take full responsibility, some of the package's
configuration information is stored in module-level symbols ("global
variables" in the language of C).  In particular, the SAX document content
handler used to parse documents is stored as a module-level symbol.  In
order to "enable themselves", many modules override portions of the default
content handler when they are imported.  It is therefore important to
import modules in the correct order and to import only the modules you wish
to use.  I have been working to correct this design flaw, and a solution is
in place but due to the need to support legacy code it's not possible to
disable this undesirable behaviour at this time.

Here is a brief tutorial for a common use case:  load a LIGO Light-Weight
XML document containing tabular data complying with the LSC table
definitions, access rows in the tables including the use of ID-based cross
references, modify the contents of a table, and finally write the document
back to disk.  Please see the documentation for the modules, classes,
functions, and methods shown below for more information.

Example:

>>> # import modules
>>> from glue.ligolw import table
>>> from glue.ligolw import lsctables
>>> from glue.ligolw import utils
>>> 
>>> # load a document.  gzip'ed files are auto-detected
>>> filename = "demo.xml.gz"
>>> xmldoc = utils.load_filename(filename, verbose = True)
>>> 
>>> # retrieve the process and sngl_inspiral tables.  these are list-like
>>> # objects of rows.  the row objects' attributes are the column names
>>> process_table = table.get_table(xmldoc, lsctables.ProcessTable.tableName)
>>> sngl_inspiral_table = table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
>>> 
>>> # fix the mtotal column in the sngl_inspiral table
>>> for row in sngl_inspiral_table:
...	row.mtotal = row.mass1 + row.mass2
...
>>> # construct a look-up table mapping process_id to row in process table
>>> index = dict((row.process_id, row) for row in process_table)
>>> 
>>> # for each trigger in the sngl_inspiral table, print the name of the user
>>> # who ran the job that produced it, the computer on which the job ran, and
>>> # the GPS end time of the trigger
>>> for row in sngl_inspiral_table:
...	process = index[row.process_id]
...	print "%s@%s: %s s" % (process.username, process.node, str(row.get_end()))
...
>>> # write document.  must explicitly state whether or not the file is to be
>>> # compressed
>>> utils.write_filename(xmldoc, filename, gz = filename.endswith(".gz"), verbose = True)
"""


from glue import git_version


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


__all__ = [
	"ligolw",
	"types",
	"ilwd",
	"table",
	"array",
	"param",
	"lsctables",
	"utils"
]

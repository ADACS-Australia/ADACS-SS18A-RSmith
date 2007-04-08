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
Add (merge) LIGO LW XML files containing LSC tables.
"""


import os
import sys
from urlparse import urlparse


from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"[7:-2]
__version__ = "$Revision$"[11:-2]


#
# =============================================================================
#
#                                    Input
#
# =============================================================================
#


def url2path(url):
	"""
	If url identifies a file on the local host, return the path to the
	file otherwise raise ValueError.
	"""
	scheme, host, path, nul, nul, nul = urlparse(url)
	if scheme.lower() in ("", "file") and host.lower() in ("", "localhost"):
		return path
	raise ValueError, url


def remove_input(urls, preserves, verbose = False):
	"""
	Attempt to delete all files identified by the URLs in urls except
	any that are the same as the files in the preserves list.
	"""
	for path in map(url2path, urls):
		if True in map(os.path.samefile, [path] * len(preserves), preserves):
			continue
		if verbose:
			print >>sys.stderr, "removing %s ..." % path
		try:
			os.remove(path)
		except:
			pass


#
# =============================================================================
#
#                                Document Merge
#
# =============================================================================
#


def reassign_ids(doc, verbose = False):
	"""
	Assign new IDs to all rows in all LSC tables in doc so that there
	are no collisions when the LIGO_LW elements are merged.
	"""
	# Can't simply run reassign_ids() on doc because we need to
	# construct a fresh old --> new mapping within each LIGO_LW block.
	for n, elem in enumerate(doc.childNodes):
		if verbose:
			print >>sys.stderr, "reassigning row IDs: %.1f%%\r" % (100.0 * (n + 1) / len(doc.childNodes)),
		if elem.tagName == ligolw.LIGO_LW.tagName:
			table.reassign_ids(elem)
	if verbose:
		print >>sys.stderr, "reassigning row IDs: 100.0%"
	return doc


def merge_ligolws(elem):
	"""
	Merge all LIGO_LW elements that are immediate children of elem by
	appending their children to the first.
	"""
	ligolws = [child for child in elem.childNodes if child.tagName == ligolw.LIGO_LW.tagName]
	if ligolws:
		dest = ligolws.pop(0)
		for src in ligolws:
			# copy children;  LIGO_LW elements have no attributes
			map(dest.appendChild, src.childNodes)
			# unlink from parent
			if src.parentNode is not None:
				src.parentNode.removeChild(src)
	return elem


def compare_table_cols(a, b):
	"""
	Return False if the two tables a and b have the same columns
	(ignoring order) according to LIGO LW name conventions, return True
	otherwise.
	"""
	acols = [(table.StripColumnName(col.getAttribute("Name")), col.getAttribute("Type")) for col in a.getElementsByTagName(ligolw.Column.tagName)]
	acols.sort()
	bcols = [(table.StripColumnName(col.getAttribute("Name")), col.getAttribute("Type")) for col in b.getElementsByTagName(ligolw.Column.tagName)]
	bcols.sort()
	return cmp(acols, bcols)


def merge_compatible_tables(elem):
	"""
	Below the given element, find all Tables whose structure is
	described in lsctables, and merge compatible ones of like type.
	That is, merge all SnglBurstTables that have the same columns into
	a single table, etc..
	"""
	for name in lsctables.TableByName.keys():
		tables = table.getTablesByName(elem, name)
		if tables:
			dest = tables.pop(0)
			for src in tables:
				if table.CompareTableNames(src.getAttribute("Name"), dest.getAttribute("Name")):
					# src and dest have different names
					continue
				# src and dest have the same names
				if compare_table_cols(dest, src):
					# but they have different columns
					raise ValueError, "document contains %s tables with incompatible columns" % dest.getAttribute("Name")
				# and the have the same columns
				# copy src rows to dest
				map(dest.append, src)
				# unlink src from parent
				if src.parentNode is not None:
					src.parentNode.removeChild(src)
	return elem


#
# =============================================================================
#
#                                 Library API
#
# =============================================================================
#


def ligolw_add(doc, urls, non_lsc_tables_ok = False, verbose = False):
	"""
	An implementation of the LIGO LW add algorithm.  urls is a list of
	URLs (or filenames) to load, doc is the XML document tree to which
	they should be added.
	"""
	# Input
	for n, url in enumerate(urls):
		if verbose:
			print >>sys.stderr, "%d/%d:" % (n + 1, len(urls)),
		utils.load_url(url, verbose = verbose, gz = url[-3:] == ".gz", xmldoc = doc)

	# ID reassignment
	if not non_lsc_tables_ok and lsctables.HasNonLSCTables(doc):
		raise ValueError, "non-LSC tables found.  Use --non-lsc-tables-ok to force"
	reassign_ids(doc, verbose = verbose)

	# Document merge
	if verbose:
		print >>sys.stderr, "merging elements ..."
	merge_ligolws(doc)
	merge_compatible_tables(doc)

	return doc

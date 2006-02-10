__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"
__version__ = "$Revision$"

try:
	import numarray
except:
	pass
import re
import sys
from xml import sax

import ligolw


#
# =============================================================================
#
#                               Type Information
#
# =============================================================================
#

StringTypes = ["char_s", "ilwd:char", "ilwd:char_u", "lstring", "string"]
IntTypes = ["int_2s", "int_2u", "int_4s", "int_4u", "int_8s", "int_8u", "int"]
FloatTypes = ["real_4", "real_8", "float", "double"]

Types = StringTypes + IntTypes + FloatTypes

ToNumArrayType = {
	"int_2s": "Int16",
	"int_2u": "UInt16",
	"int_4s": "Int32",
	"int_4u": "UInt32",
	"int_8s": "Int64",
	"int_8u": "UInt64",
	"int": "Int32",
	"real_4": "Float32",
	"real_8": "Float64",
	"float": "Float64",
	"double": "Float64"
}


#
# =============================================================================
#
#                           Column Name Manipulation
#
# =============================================================================
#

# Regular expression to extract the significant part of a column name
# according to the LIGO LW naming conventions.

# FIXME: the pattern should be
#
# r"(?:\A[a-z0-9_]+:|\A)(?P<FullName>(?:[a-z0-9_]+:|\A)(?P<Name>[a-z0-9_]+))\Z"
#
# but people are putting upper case letters in names!!!!!  Someone is going
# to get the beats.

ColumnPattern = re.compile(r"(?:\A\w+:|\A)(?P<FullName>(?:\w+:|\A)(?P<Name>\w+))\Z")


def StripColumnName(name):
	"""
	Return the significant portion of a column name according to LIGO
	LW naming conventions.
	"""
	try:
		return ColumnPattern.search(name).group("Name")
	except AttributeError:
		return name


def CompareColumnNames(name1, name2):
	"""
	Convenience function to compare two column names according to LIGO
	LW naming conventions.
	"""
	return cmp(StripColumnName(name1), StripColumnName(name2))


#
# =============================================================================
#
#                           Table Name Manipulation
#
# =============================================================================
#

# Regular expression used to extract the signifcant portion of a table or
# stream name, according to LIGO LW naming conventions.

TablePattern = re.compile(r"(?:\A[a-z0-9_]+:|\A)(?P<Name>[a-z0-9_]+):table\Z")


def StripTableName(name):
	"""
	Return the significant portion of a table name according to LIGO LW
	naming conventions.
	"""
	try:
		return TablePattern.search(name).group("Name")
	except AttributeError:
		return name


def CompareTableNames(name1, name2):
	"""
	Convenience function to compare two table names according to LIGO
	LW naming conventions.
	"""
	return cmp(StripTableName(name1), StripTableName(name2))


#
# =============================================================================
#
#                               Element Classes
#
# =============================================================================
#

class _ColumnIter(object):
	"""
	An iterator class for looping through the values in a column.
	"""
	def __init__(self, column):
		self.rowiter = iter(column.parentNode.rows)
		self.attr = column.asattribute

	def next(self):
		return getattr(self.rowiter.next(), self.attr)

class Column(ligolw.Column):
	"""
	High-level column element that provides list-like access to the
	values in a column.
	"""
	def __init__(self, attrs):
		ligolw.Column.__init__(self, attrs)
		self.asattribute = StripColumnName(self.getAttribute("Name"))

	def __len__(self):
		"""
		Return the number of values in this column.
		"""
		return len(self.parentNode.rows)

	def __getitem__(self, i):
		"""
		Retrieve the value in this column in row i.
		"""
		return getattr(self.parentNode.rows[i], self.asattribute)

	def __setitem__(self, i, value):
		"""
		Set the value in this column in row i.
		"""
		setattr(self.parentNode.rows[i], self.asattribute, value)

	def __iter__(self):
		"""
		Return an iterator object for iterating over values in this
		column.
		"""
		return _ColumnIter(self)

	def count(self, value):
		"""
		Return the number of rows with this column equal to value.
		"""
		n = 0
		for r in self.parentNode.rows:
			if getattr(r, self.asattribute) == value:
				n += 1
		return n

	def index(self, val):
		"""
		Return the smallest index of the row(s) with this column equal
		to value.
		"""
		for i in range(len(self.parentNode.rows)):
			if getattr(self.parentNode.rows[i], self.asattribute) == value:
				return i
		raise ValueError, "%s not found" % repr(val)

	def asarray(self):
		"""
		Construct a numarray array from this column.
		"""
		if self.getAttribute("Type") in StringTypes:
			raise TypeError, "Column does not have numeric type"
		# hack to work around bug in numarray:  numarray tests that
		# an object can be turned into an array, that is it is
		# "list like", by trying to retrieve element 0.  This fails
		# if the list like object has 0 length, causing numarray to
		# barf.  If the object is, in fact, a real Python list then
		# numarray is made happy.
		if not len(self):
			return numarray.array([], type = ToNumArrayType[self.getAttribute("Type")], shape = (len(self),))
		return numarray.array(self, type = ToNumArrayType[self.getAttribute("Type")], shape = (len(self),))

	# FIXME: This function is for the metaio library:  metaio cares
	# what order the attributes of XML tags come in.  This function
	# will be removed when the people responsible for the metaio
	# library fix it.
	def start_tag(self):
		"""
		See the source code for an explanation.
		"""
		return "<%s Name=\"%s\" Type=\"%s\"/>" % (self.tagName, self.getAttribute("Name"), self.getAttribute("Type"))


class TableStream(ligolw.Stream):
	"""
	High-level Stream element for use inside Tables.  This element
	knows how to parse the delimited character stream into rows in the
	parent element, and knows how to turn the parent's rows back into a
	character stream.
	"""
	def __init__(self, attrs):
		ligolw.Stream.__init__(self, attrs)
		self.tokenizer = re.compile(r"""\s*(?:"([^"]*)")|(?:([^""" + self.getAttribute("Delimiter") + r"""\s]+))\s*""" + self.getAttribute("Delimiter"))
		self.tokens = []

	def appendData(self, content):
		# append new data to buffer
		ligolw.Stream.appendData(self, content)

		# move tokens from buffer to token list
		match = None
		for match in self.tokenizer.finditer(self.pcdata):
			self.tokens.append(match.group(match.lastindex))
		if match != None:
			self.pcdata = self.pcdata[match.end():]

		# construct row objects from tokens, and append to parent
		while len(self.tokens) >= len(self.parentNode.columninfo):
			row = self.parentNode.RowType()
			for i, (colname, pytype) in enumerate(self.parentNode.columninfo):
				try:
					setattr(row, colname, pytype(self.tokens[i]))
				except ValueError, e:
					raise ligolw.ElementError, "Stream parsing error near tokens %s: %s" % (str(self.tokens), str(e))
				except AttributeError, e:
					pass
			self.tokens = self.tokens[i+1:]
			self.parentNode.append(row)

	def _rowstr(self, row, columns):
		# FIXME: after calling getattr(), should probably check that
		# the result has the expected type.
		strs = []
		for column in columns:
			if column.getAttribute("Type") in StringTypes:
				strs.append("\"" + getattr(row, StripColumnName(column.getAttribute("Name"))) + "\"")
			else:
				strs.append(str(getattr(row, StripColumnName(column.getAttribute("Name")))))
		return self.getAttribute("Delimiter").join(strs)

	def write(self, file = sys.stdout, indent = ""):
		columns = self.parentNode.getElementsByTagName(ligolw.Column.tagName)

		# loop over parent's rows.  This is complicated because we
		# need to not put a comma at the end of the last row.
		print >>file, indent + self.start_tag()
		rowiter = iter(self.parentNode)
		try:
			file.write(indent + ligolw.Indent + self._rowstr(rowiter.next(), columns))
			while True:
				file.write(self.getAttribute("Delimiter") + "\n" + indent + ligolw.Indent + self._rowstr(rowiter.next(), columns))
		except StopIteration:
			file.write("\n")
		print >>file, indent + self.end_tag()

	# FIXME: This function is for the metaio library:  metaio cares
	# what order the attributes of XML tags come in.  This function
	# will be removed when the people responsible for the metaio
	# library fix it.
	def start_tag(self):
		"""
		See the source code for an explanation.
		"""
		return "<%s Name=\"%s\" Type=\"%s\" Delimiter=\"%s\">" % (self.tagName, self.getAttribute("Name"), self.getAttribute("Type"), self.getAttribute("Delimiter"))


class TableRow(object):
	"""
	Helpful parent class for row objects.
	"""
	pass


class Table(ligolw.Table):
	"""
	High-level Table element that knows about its columns and rows.
	"""
	validcolumns = None
	RowType = TableRow

	def __init__(self, *attrs):
		"""
		Initialize
		"""
		ligolw.Table.__init__(self, *attrs)
		self.columninfo = []
		self.rows = []


	#
	# Sequence methods
	#

	def __len__(self):
		"""
		Return the number of rows in this table.
		"""
		return len(self.rows)

	def __getitem__(self, key):
		"""
		Retrieve row(s).
		"""
		return self.rows[key]

	def __setitem__(self, key, value):
		"""
		Set row(s).
		"""
		self.rows[key] = value

	def __delitem__(self, key):
		"""
		Remove row(s).
		"""
		del self.rows[key]

	def __iter__(self):
		"""
		Return an iterator object for iterating over rows in this
		table.
		"""
		return iter(self.rows)

	def append(self, row):
		"""
		Append row to the list of rows for this table.
		"""
		self.rows.append(row)

	def extend(self, rows):
		"""
		Add a list of rows to the end of the table.
		"""
		self.rows.extend(rows)

	def pop(self, key):
		return self.rows.pop(key)

	def filterRows(self, func):
		"""
		Delete all rows for which func(row) evaluates to False.
		"""
		i = 0
		while i < len(self.rows):
			if not func(self.rows[i]):
				del self.rows[i]
			else:
				i += 1
		return self


	#
	# Column access
	#

	def getColumnByName(self, name):
		try:
			return self.getElements(lambda e: (e.tagName == ligolw.Column.tagName) and (CompareColumnNames(e.getAttribute("Name"), name) == 0))[0]
		except IndexError:
			raise KeyError, "no Column matching name \"%s\"" % name


	#
	# Element methods
	#

	def appendChild(self, child):
		if child.tagName == ligolw.Column.tagName:
			colname = StripColumnName(child.getAttribute("Name"))
			llwtype = child.getAttribute("Type")
			if self.validcolumns != None:
				if colname not in self.validcolumns.keys():
					raise ligolw.ElementError, "invalid Column name \"%s\" for Table" % child.getAttribute("Name")
				if self.validcolumns[colname] != llwtype:
					raise ligolw.ElementError, "invalid type \"%s\" for Column \"%s\"" % (llwtype, child.getAttribute("Name"))
			if colname in [c[0] for c in self.columninfo]:
				raise ligolw.ElementError, "duplicate Column \"%s\"" % child.getAttribute("Name")
			if llwtype in StringTypes:
				self.columninfo.append((colname, str))
			elif llwtype in IntTypes:
				self.columninfo.append((colname, int))
			elif llwtype in FloatTypes:
				self.columninfo.append((colname, float))
			else:
				raise ligolw.ElementError, "unrecognized Type attribute \"%s\" for Column element" % llwtype
		elif child.tagName == ligolw.Stream.tagName:
			if child.getAttribute("Name") != self.getAttribute("Name"):
				raise ligolw.ElementError, "Stream name \"%s\" does not match Table name \"%s\"" % (child.getAttribute("Name"), self.getAttribute("Name"))
		ligolw.Table.appendChild(self, child)

	def removeChild(self, child):
		"""
		Remove a child from this element.  The child element is
		returned, and it's parentNode element is reset.
		"""
		ligolw.Table.removeChild(self, child)
		if child.tagName == ligolw.Column.tagName:
			for n in [n for n, item in enumerate(self.columninfo) if item[0] == StripColumnName(child.getAttribute("Name"))]:
				del self.columinfo[n]
		return child


#
# =============================================================================
#
#                               Content Handler
#
# =============================================================================
#

class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	"""
	ContentHandler that redirects Column, Stream and Table elements to
	those defined in this module.
	"""
	def startColumn(self, attrs):
		return Column(attrs)

	def startStream(self, attrs):
		if self.current.tagName == ligolw.Table.tagName:
			return TableStream(attrs)
		return ligolw.LIGOLWContentHandler.startStream(self, attrs)

	def endStream(self):
		# stream tokenizer uses comma to identify end of each
		# token, so add a final comma to induce the last token to
		# get parsed.  FIXME: this is a hack, and hacks are the
		# source of bugs.
		if self.current.parentNode.tagName == ligolw.Table.tagName:
			self.current.appendData(self.current.getAttribute("Delimiter"))
		else:
			ligolw.LIGOLWContentHandler.endStream(self)

	def startTable(self, attrs):
		return Table(attrs)

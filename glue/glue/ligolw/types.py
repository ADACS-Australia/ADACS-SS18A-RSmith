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
Definitions of type strings found in LIGO Light Weight XML files.

Notes.  To guarantee that a double-precision floating-point number can be
reconstructed exactly from its representation as a decimal number, one must
use 17 decimal digits;  for single-precision, the number is 9.  Python uses
only double-precision numbers, but LIGO Light Weight XML allows for
single-precision values, so I provide distinct format specifiers for those
cases here.  In both cases, I have elected to use 1 fewer digits than are
required to uniquely reconstruct the number:  the XML written by this
library is lossy.  I made this choice to reduce the file size, for example

>>> "%.17g" % 0.1
'0.10000000000000001'

while

>>> "%.16g" % 0.1
'0.1'

In this worst case, storing full precision increases the size of the XML by
more than an order of magnitude.  If you wish to make a different choice
for your files, for example if you wish your XML files to be lossless,
simply include the lines

	glue.ligolw.types.FormatFunc.update({
		"real_4": u"%.9g".__mod__,
		"real_8": u"%.17g".__mod__,
		"float": u"%.9g".__mod__,
		"double": u"%.17g".__mod__
	})

anywhere in your code, but before you write the document to a file.

References:

	- http://docs.sun.com/source/806-3568/ncg_goldberg.html
"""


import base64


import ilwd


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"[7:-2]
__version__ = "$Revision$"[11:-2]


#
# =============================================================================
#
#                               Type Information
#
# =============================================================================
#


IDTypes = [u"ilwd:char"]
BlobTypes = [u"blob"]
StringTypes = IDTypes + [u"char_s", u"char_v", u"lstring", u"string"]
IntTypes = [u"int_2s", u"int_2u", u"int_4s", u"int_4u", u"int_8s", u"int_8u", u"int"]
FloatTypes = [u"real_4", u"real_8", u"float", u"double"]
TimeTypes = [u"GPS", u"Unix", u"ISO-8601"]
Types = BlobTypes + StringTypes + IntTypes + FloatTypes + TimeTypes


def ligolw_string_format_func(s):
	return u"\"%s\"" % unicode(s).replace(u"\\", u"\\\\").replace(u"\"", u"\\\"")


FormatFunc = {
	u"char_s": ligolw_string_format_func,
	u"char_v": ligolw_string_format_func,
	u"ilwd:char": ligolw_string_format_func,
	u"blob": lambda b: "\"%s\"" % base64.standard_b64encode(b),
	u"lstring": ligolw_string_format_func,
	u"string": ligolw_string_format_func,
	u"int_2s": u"%d".__mod__,
	u"int_2u": u"%u".__mod__,
	u"int_4s": u"%d".__mod__,
	u"int_4u": u"%u".__mod__,
	u"int_8s": u"%d".__mod__,
	u"int_8u": u"%u".__mod__,
	u"int": u"%d".__mod__,
	u"real_4": u"%.8g".__mod__,
	u"real_8": u"%.16g".__mod__,
	u"float": u"%.8g".__mod__,
	u"double": u"%.16g".__mod__
}


ToPyType = {
	u"char_s": unicode,
	u"char_v": unicode,
	u"ilwd:char": ilwd.get_ilwdchar,
	u"blob": lambda s: buffer(base64.b64decode(s)),
	u"lstring": unicode,
	u"string": unicode,
	u"int_2s": int,
	u"int_2u": int,
	u"int_4s": int,
	u"int_4u": int,
	u"int_8s": int,
	u"int_8u": int,
	u"int": int,
	u"real_4": float,
	u"real_8": float,
	u"float": float,
	u"double": float
}


FromPyType = {
	ilwd.ilwdchar: u"ilwd:char",
	buffer: u"blob",
	str: u"lstring",
	unicode: u"lstring",
	int: u"int_4s",
	long: u"int_8s",
	float: u"real_8"
}


ToNumPyType = {
	u"int_2s": "int16",
	u"int_2u": "uint16",
	u"int_4s": "int32",
	u"int_4u": "uint32",
	u"int_8s": "int64",
	u"int_8u": "uint64",
	u"int": "int32",
	u"real_4": "float32",
	u"real_8": "float64",
	u"float": "float64",
	u"double": "float64"
}


FromNumPyType = {
	"int16": u"int_2s",
	"uint16": u"int_2u",
	"int32": u"int_4s",
	"uint32": u"int_4u",
	"int64": u"int_8s",
	"uint64": u"int_8u",
	"float32": u"real_4",
	"float64": u"real_8"
}


ToSQLiteType = {
	u"char_s": "TEXT",
	u"char_v": "TEXT",
	u"ilwd:char": "TEXT",
	u"blob": "BLOB",
	u"lstring": "TEXT",
	u"string": "TEXT",
	u"int_2s": "INTEGER",
	u"int_2u": "INTEGER",
	u"int_4s": "INTEGER",
	u"int_4u": "INTEGER",
	u"int_8s": "INTEGER",
	u"int_8u": "INTEGER",
	u"int": "INTEGER",
	u"real_4": "REAL",
	u"real_8": "REAL",
	u"float": "REAL",
	u"double": "REAL"
}


FromSQLiteType = {
	"BLOB": u"blob",
	"TEXT": u"lstring",
	"STRING": u"lstring",
	"INTEGER": u"int_4s",
	"REAL": u"real_8"
}

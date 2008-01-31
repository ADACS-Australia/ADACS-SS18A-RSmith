# $Id$
#
# Copyright (C) 2007  Kipp C. Cannon, Nickolas Fotopoulos
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
A collection of iteration utilities.
"""


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__version__ = "$Revision$"[11:-2]
__date__ = "$Date$"[7:-2]


#
# =============================================================================
#
#                               Iteration Tools
#
# =============================================================================
#


def MultiIter(*sequences):
	"""
	A generator for iterating over the elements of multiple sequences
	simultaneously.  With N sequences given as input, the generator
	yields all possible distinct N-tuples that contain one element from
	each of the input sequences.

	Example:

	>>> x = MultiIter([0, 1, 2], [10, 11])
	>>> list(x)
	[(0, 10), (1, 10), (2, 10), (0, 11), (1, 11), (2, 11)]

	The elements in each output tuple are in the order of the input
	sequences, and the left-most input sequence is iterated over first.

	The input sequences are each iterated over only once, so it is safe
	to pass generators as arguments.  Also, this generator is
	significantly faster if the longest input sequence is given as the
	first argument.  For example, this code

	>>> lengths = range(1, 12)
	>>> for x in MultiIter(*map(range, lengths)):
	...	pass
	...

	runs approximately 5 times faster if the lengths list is reversed.
	"""
	if sequences:
		# FIXME:  experiment with a generator expression in Python
		# >= 2.5
		# FIXME:  this loop is about 5% faster if done the other
		# way around, if the last list is iterated over in the
		# inner loop.  but there is code, like snglcoinc.py in
		# pylal, that has been optimized for the current order and
		# would need to be reoptimized if this function were to be
		# reversed.
		head = tuple([(x,) for x in sequences[0]])
		for t in MultiIter(*sequences[1:]):
			for h in head:
				yield h + t
	else:
		yield ()


def choices(vals, n):
	"""
	A generator for iterating over all choices of n elements from the
	input sequence vals.  In each result returned, the original order
	of the values is preserved.

	Example:

	>>> x = choices(["a", "b", "c"], 2)
	>>> list(x)
	[('a', 'b'), ('a', 'c'), ('b', 'c')]

	The order of combinations in the output sequence is always the
	same, so if choices() is called twice with two different sequences
	of the same length the first combination in each of the two output
	sequences will contain elements from the same positions in the two
	different input sequences, and so on for each subsequent pair of
	output combinations.

	Example:

	>>> x = choices(["a", "b", "c"], 2)
	>>> y = choices(["1", "2", "3"], 2)
	>>> zip(x, y)
	[(('a', 'b'), ('1', '2')), (('a', 'c'), ('1', '3')), (('b', 'c'),
	('2', '3'))]
	"""
	if n == len(vals):
		yield tuple(vals)
	elif n > 1:
		n -= 1
		for i, v in enumerate(vals[:-n]):
			v = (v,)
			for c in choices(vals[i+1:], n):
				yield v + c
	elif n == 1:
		for v in vals:
			yield (v,)
	else:
		# n < 1
		raise ValueError, n


def uniq(iterable):
	"""
	Yield the unique items of an iterable, preserving order.
	http://mail.python.org/pipermail/tutor/2002-March/012930.html

	Example:

	>>> x = uniq([0, 0, 2, 6, 2, 0, 5])
	>>> list(x)
	[0, 2, 6, 5]
	"""
	temp_dict = {}
	for e in iterable:
		if e not in temp_dict:
			yield temp_dict.setdefault(e, e)


#
# =============================================================================
#
#    any() and all() are built-ins in Python 2.5, but I don't want to wait.
#
# =============================================================================
#


try:
	any = any
	all = all
except NameError:
	# These short-circuit, returning as soon as the return value can be
	# determined.  These are a factor of a few slower than Python 2.5's
	# implementation.
	def any(S):
		"""
		any(iterable) -> bool

		Return True if bool(x) is True for any x in the iterable.
		"""
		for x in S:
			if x: return True
		return False
	def all(S):
		"""
		all(iterable) -> bool

		Return True if bool(x) is True for all values x in the iterable.
		"""
		for x in S:
			if not x: return False
		return True


#
# =============================================================================
#
#           Thing for keeping only the highest values in a sequence
#
# =============================================================================
#


class Highest(list):
	"""
	A class for use when you need to collect the largest in a very long
	sequence of things, too long a sequence to hold in memory all at
	once and sort.  This class behaves like a list, in fact it is a
	Python list, but one that stores only some fraction of all items
	that have been added to it.  The list is always ordered, so the
	insert() and __setitem__() methods are not supported, only append()
	and extend().

	Example:

	>>> import random
	>>> l = Highest(max = 3)
	>>> for i in range(10000):
	...	l.append(random.random())
	...
	>>> l
	[0.99997649136673972, 0.99997199878829768, 0.99991682393505932]
	>>> len(l)
	10000
	>>> list.__len__(l)
	3
	"""
	def __init__(self, sequence = tuple(), max = None):
		list.__init__(self, sequence)
		self.n = list.__len__(self)
		self.max = int(max)
		list.sort(self, reverse = True)
		del self[self.max:]

	def __len__(self):
		return self.n

	def append(self, value):
		hi = list.__len__(self)
		lo = 0
		while lo < hi:
			mid = (lo + hi) // 2
			if value > self[mid]:
				hi = mid
			else:
				lo = mid + 1
		list.insert(self, lo, value)
		self.n += 1
		del self[self.max:]

	def extend(self, sequence):
		before = list.__len__(self)
		list.extend(self, sequence)
		self.n += list.__len__(self) - before
		list.sort(self, reverse = True)
		del self[self.max:]

	#
	# Stubs to prevent bugs
	#

	def __setitem__(*args, **kwargs):
		raise NotImplementedError

	def reverse(*args, **kwargs):
		raise NotImplementedError

	def remove(*args, **kwargs):
		raise NotImplementedError

	def pop(*args, **kwargs):
		raise NotImplementedError

	def insert(*args, **kwargs):
		raise NotImplementedError

	def index(*args, **kwargs):
		raise NotImplementedError

	def count(*args, **kwargs):
		raise NotImplementedError

	def sort(*args, **kwargs):
		raise NotImplementedError


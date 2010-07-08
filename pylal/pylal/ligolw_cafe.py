# Copyright (C) 2006  Kipp Cannon
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
LIGO Light-Weight XML coincidence analysis front end.
"""


from math import log10
import sys

from glue import segments
from glue.lal import CacheEntry
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils
from pylal import git_version
from pylal import ligolw_tisi
from pylal import llwapp
from pylal import packing


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__date__ = "git id %s" % git_version.id
__version__ = git_version.date


#
# =============================================================================
#
#                                    Input
#
# =============================================================================
#


def load_cache(filename, verbose = False):
	"""
	Parse a LAL cache file named filename into a list of
	glue.lal.CacheEntry objects.  If filename is None then input is
	taken from stdin.
	"""
	if verbose:
		print >>sys.stderr, "reading %s ..." % (filename or "stdin")
	if filename is not None:
		f = file(filename)
	else:
		f = sys.stdin
	return [CacheEntry(line, coltype = lsctables.LIGOTimeGPS) for line in f]


def cache_to_seglistdict(cache):
	"""
	Construct a coalesced segmentlistdict object from a list of
	glue.lal.CacheEntry objects.
	"""
	s = segments.segmentlistdict()
	for c in cache:
		s |= c.to_segmentlistdict()
	return s


#
# =============================================================================
#
#                             Performance Helpers
#
# =============================================================================
#


def segmentlistdict_normalize(seglistdict, origin):
	"""
	Convert the times in a segmentlist dictionary to floats relative to
	origin.  The purpose is to allow segment lists stored as
	LIGOTimeGPS times to be manipulated more quickly without loss of
	precision.  The modification is done in place.
	"""
	for seglist in seglistdict.itervalues():
		for i, seg in enumerate(seglist):
			seglist[i] = segments.segment(float(seg[0] - origin), float(seg[1] - origin))


def segmentlistdict_unnormalize(seglistdict, origin):
	"""
	The opposite of segmentlistdict_normalize(), restores the times in
	a segmentlist dictionary to absolute times.  The modification is
	done in place.
	"""
	for seglist in seglistdict.itervalues():
		for i, seg in enumerate(seglist):
			seglist[i] = segments.segment(origin + seg[0], origin + seg[1])


#
# =============================================================================
#
#                             Output Cache Packing
#
# =============================================================================
#


class LALCacheBin(packing.Bin):
	"""
	Subclass of the packing.Bin class representing a LAL file cache.
	The files contained in the bin are available in the .objects
	attribute, which is a list of glue.lal.CacheEntry objects.  The
	.size attribute holds a glue.segments.segmentlistdict object giving
	the times spanned by the files in the bin.  The .extent attribute
	holds the result of running .extent_all() on the .size attribute.
	"""
	def __init__(self):
		packing.Bin.__init__(self)
		self.size = segments.segmentlistdict()
		self.extent = None

	def add(self, *args):
		packing.Bin.add(self, *args)
		self.extent = self.size.extent_all()
		return self

	def __iadd__(self, *args):
		packing.Bin.__iadd__(self, *args)
		self.extent = self.size.extent_all()
		return self

	def __cmp__(self, other):
		return cmp(self.extent, other.extent)

	def __str__(self):
		return "\n".join(map(str, self.objects))


class CafePacker(packing.Packer):
	"""
	Packing algorithm implementing the ligolw_cafe file list packing
	algorithm.
	"""
	def set_offset_vectors(self, offset_vectors):
		"""
		Set the list of offset vectors to be considered when
		deciding the bins in which each file belongs.  Must be
		called before packing any files.  The input is a list of
		dictionaries, each mapping instruments to offsets.
		"""
		self.offset_vectors = offset_vectors
		min_offset = min(min(offset_vector.values()) for offset_vector in offset_vectors)
		max_offset = max(max(offset_vector.values()) for offset_vector in offset_vectors)
		# largest gap that can conceivably be closed by the time
		# slides
		self.max_gap = max_offset - min_offset
		assert self.max_gap >= 0

	def pack(self, cache_entry):
		"""
		Find all bins in which this glue.lal.CacheEntry instance
		belongs, merge them, and add this cache entry to the
		result.  Create a new bin for this cache entry if it does
		not belong in any of the existing bins.

		The cache entry "belongs" in a bin if after each of the
		preset offset vectors (see the .set_offset_vectors()
		method) is applied to both the contents of a bin and the
		cache entry, any of the segment lists of the bin and cache
		entry are found to intersect.  When checking for
		intersection, only the segment lists whose instrument names
		are listed in the offset vector are compared.
		"""
		#
		# add the cache entry to a new bin by itself
		#

		new = LALCacheBin()
		new.add(cache_entry, cache_entry.to_segmentlistdict())

		#
		# assemble a list of bins in which the cache entry belongs.
		# iterate over existing bins backwards so that we record
		# the indeces of matching bins in descending order.  bail
		# out when we find a bin that precedes the new one
		#

		matching_bins = []
		for n in xrange(len(self.bins) - 1, -1, -1):
			bin = self.bins[n]
			if bin.extent[1] < new.extent[0] - self.max_gap:
				break
			for offset_vector in self.offset_vectors:
				new.size.offsets.update(offset_vector)
				bin.size.offsets.update(offset_vector)
				if bin.size.is_coincident(new.size, keys = offset_vector.keys()):
					matching_bins.append(n)
					break
			bin.size.offsets.clear()
		new.size.offsets.clear()

		#
		# add new cache entry to bins
		#

		if not matching_bins:
			#
			# no existing bins match, add a new one
			#

			self.bins.append(new)
		else:
			#
			# put cache entry into first bin that was found to
			# match.  if cache entry belongs in more than one
			# bin, merge them.  note that the matching bin
			# indexes are given in descending order so poping
			# the bins as we go does not affect the indexes of
			# the remaining, matching, bins.
			#

			dest = self.bins[matching_bins.pop(-1)]
			dest += new
			for n in matching_bins:
				dest += self.bins.pop(n)

		#
		# time-order the bins so the bail-out above works next time
		# this method is called
		#

		self.bins.sort()


def split_bins(cafepacker, extentlimit):
	"""
	Split bins of stored in CafePacker until each bin has an extent no
	longer than extentlimit.
	"""

	#
	# loop overall the bins in cafepacker.bins. we pop items out of
	# cafepacker.bins and append new ones to the end so need a while loop
	# checking the extent of each bin in cafepacker.bins until all bins are
	# done being split
	#

	idx = 0
	while idx < len(cafepacker.bins):
		if abs(cafepacker.bins[idx].extent) <= extentlimit:
			#
			# bin doesn't need splitting so move to next
			#

			idx += 1
			continue

		#
		# split this bin so pop it out of the list
		#

		bigbin = cafepacker.bins.pop(idx)

		#
		# calculate the central time of the union of all the input
		# files in the bin
		#

		splittime = lsctables.LIGOTimeGPS(bigbin.extent[0] + (bigbin.extent[1] - bigbin.extent[0])/2)

		#
		# split the segmentlistdict at this time
		#

		splitseglistdict = segments.segmentlistdict()
		for key in bigbin.size.keys():
			splitseglistdict[key] = segments.segmentlist([segments.segment(-segments.infinity(),splittime)])

		#
		# create bins for the first and second halves
		#

		bin1 = LALCacheBin()
		bin1.size = bigbin.size & splitseglistdict
		bin1.extent = bigbin.extent & splitseglistdict.values()[0][0]

		bin2 = LALCacheBin()
		bin2.size = bigbin.size & ~splitseglistdict
		bin2.extent = bigbin.extent & (~splitseglistdict.values()[0])[0]

		#
		# remove unused keys from the smaller bins' segmentlistdicts
		#

		newsize = segments.segmentlistdict()
		for key in bin1.size.keys():
			if len(bin1.size[key]):
				newsize[key] = bin1.size[key]
		bin1.size = newsize

		newsize = segments.segmentlistdict()
		for key in bin2.size.keys():
			if len(bin2.size[key]):
				newsize[key] = bin2.size[key]
		bin2.size = newsize

		#
		# find which of the objects in bigbin.objects intersect the two
		# smaller bins
		#

		for cache in bigbin.objects:
			thisseglistdict = cache.to_segmentlistdict()
			coinc1 = 0
			coinc2 = 0
			for offset_vector in self.offset_vectors:
				#
				# loop over offset vectors updating the smaller
				# bins and the object we are checking
				#

				bin1.size.offsets.update(offset_vector)
				bin2.size.offsets.update(offset_vector)
				thisseglistdict.offsets.update(offset_vector)
				if not coinc1 and bin1.size.is_coincident(thisseglistdict, keys = offset_vector.keys()):
					#
					# object is coicident with bin1
					#

					coinc1 = 1
					bin1.objects.append(cache)
				if not coinc2 and bin2.size.is_coincident(thisseglistdict, keys = offset_vector.keys()):
					#
					# object is coincident with bin2
					#

					coinc2 = 1
					bin2.objects.append(cache)

				#
				# end loop if known to be coincident with both
				# bins
				#

				if coinc1 and coinc2: break

			#
			# clear offsets applied to object
			#

			thisseglistdict.offsets.clear()

		#
		# clear offsets applied to bins
		#

		bin1.size.offsets.clear()
		bin2.size.offsets.clear()

		#
		# append smaller bins to list of bins
		#

		cafepacker.bins.append(bin1)
		cafepacker.bins.append(bin2)

		#
		# do not increment idx as we popped the large bin out of
		# cafepacker.bins
		#

	#
	# sort the bins in cafepacker
	#

	cafepacker.bins.sort()

	return cafepacker

#
# =============================================================================
#
#                                    Output
#
# =============================================================================
#


def write_caches(base, bins, instruments, verbose = False):
	filenames = []
	if len(bins):
		pattern = "%%s%%0%dd.cache" % int(log10(len(bins)) + 1)
	for n, bin in enumerate(bins):
		filename = pattern % (base, n)
		filenames.append(filename)
		if verbose:
			print >>sys.stderr, "writing %s ..." % filename
		f = file(filename, "w")
		for cacheentry in bin.objects:
			if instruments & set(cacheentry.to_segmentlistdict().keys()):
				print >>f, str(cacheentry)
	return filenames


def write_single_instrument_caches(base, bins, instruments, verbose = False):
	for instrument in instruments:
		write_caches("%s%s_" % (base, instrument), bins, [instrument], verbose)


#
# =============================================================================
#
#                                 Library API
#
# =============================================================================
#


def ligolw_cafe(cache, offset_vectors, verbose = False, extentlimit = None):
	"""
	Transform a LAL cache into a list of caches each of whose contents
	can be subjected to a coincidence analysis independently of the
	contents of the other caches, assuming the coincidence analyses
	will involve the application of the given offset vectors.

	cache is a sequence (e.g., list, tuple, etc.) of
	glue.lal.CacheEntry objects.  offset_vectors is a sequence of
	instrument/offset dictionaries describing the offset vectors to
	consider.  Set verbose to True for verbosity.

	The output is a two-element tuple.  The first element is a
	glue.segments.segmentlistdict object describing the times for which
	coincident data is available (derived from the segment metadata of
	the input cache).  The second element is a list of LALCacheBin
	objects, providing the file groups.
	"""
	#
	# Construct a segment list dictionary from the cache
	#

	if verbose:
		print >>sys.stderr, "computing segment list ..."
	seglists = cache_to_seglistdict(cache)

	#
	# For each instrument compute the times for which it will (could)
	# contribute to a coincidence analysis.
	#

	epoch = min([min(seg[0] for seg in seglist) for seglist in seglists.values() if seglist] or [None])
	segmentlistdict_normalize(seglists, epoch)
	seglists = llwapp.get_coincident_segmentlistdict(seglists, [offset_vector for offset_vector in ligolw_tisi.time_slide_component_vectors(offset_vectors, 2) if set(offset_vector.keys()).issubset(set(seglists.keys()))])
	segmentlistdict_unnormalize(seglists, epoch)

	#
	# Remove files that will not participate in a coincidence.  Take
	# care not to modify the calling code's data.  Note that because we
	# have established that this segment list describes exactly the
	# times spanned by the input files that are coincident under at
	# least one time slide, a file participates in a multi-instrument
	# coincidence if and only if it intersects these times.
	#

	if verbose:
		print >>sys.stderr, "filtering input cache ..."
	cache = [c for c in cache if seglists.intersects_all(c.to_segmentlistdict())]

	#
	# Optimization: adding files to bins in time order keeps the number
	# of bins from growing larger than needed.
	#

	if verbose:
		print >>sys.stderr, "sorting input cache ..."
	cache.sort(lambda a, b: cmp(a.segment, b.segment))

	#
	# Pack cache entries into output caches.  Having reduced the file
	# list to just those that participate in coincidences, it only
	# remains to determine which other files each must be grouped with.
	#

	outputcaches = []
	packer = CafePacker(outputcaches)
	packer.set_offset_vectors(offset_vectors)
	if verbose:
		print >>sys.stderr, "packing files ..."
	for n, cacheentry in enumerate(cache):
		if verbose and not n % 13:
			print >>sys.stderr, "\t%.1f%%\t(%d files, %d caches)\r" % (100.0 * n / len(cache), n + 1, len(outputcaches)),
		packer.pack(cacheentry)
	if verbose:
		print >>sys.stderr, "\t100.0%%\t(%d files, %d caches)" % (n + 1, len(outputcaches))

	#
	# Split caches with extent more than extentlimit
	#

	if extentlimit is not None:
		if verbose:
			print >>sys.stderr, "splitting caches with extent more than %d ..." % extentlimit
		packer = split_bins(packer, extentlimit)
		if verbose:
			print >>sys.stderr, "\t\t(%d files, %d caches)" % (n + 1, len(outputcaches))

	#
	# Sort output caches
	#

	if verbose:
		print >>sys.stderr, "sorting output caches ..."
	for cache in outputcaches:
		cache.objects.sort()

	#
	# Done.
	#

	return seglists, outputcaches

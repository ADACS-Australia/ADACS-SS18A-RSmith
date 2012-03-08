import sys
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue import segments
from glue import segmentsUtils
from glue.ligolw import table
from pylal import db_thinca_rings
from pylal import rate
import numpy
import copy
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import process
from pylal import llwapp

try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3


def allowed_analysis_table_names():
	return (dbtables.lsctables.MultiBurstTable.tableName, dbtables.lsctables.CoincInspiralTable.tableName, dbtables.lsctables.CoincRingdownTable.tableName)


def make_sim_inspiral_row_from_columns_in_db(connection):
	"""
	get the unique mapping of a sim inspiral row from columns in this
	database
	"""
	return lsctables.table.get_table(dbtables.get_xml(connection), lsctables.SimInspiralTable.tableName).row_from_cols


def time_within_segments(geocent_end_time, geocent_end_time_ns, zero_lag_segments = None):
	"""
	Return True if injection was made in the given segmentlist, if no
	segments just return True
	"""
	if zero_lag_segments is None:
		return True
	else:
		return lsctables.LIGOTimeGPS(geocent_end_time, geocent_end_time_ns) in zero_lag_segments


def get_min_far_inspiral_injections(connection, segments = None, table_name = "coinc_inspiral"):
	"""
	This function returns the found injections from a database and the
	minimum far associated with them as tuple of the form (far, sim). It also tells
	you all of the injections that should have been injected.  Subtracting the two
	outputs	should tell you the missed injections
	"""

	if table_name == dbtables.lsctables.CoincInspiralTable.tableName:
		found_query = 'SELECT sim_inspiral.*, coinc_inspiral.combined_far FROM sim_inspiral JOIN coinc_event_map AS mapA ON mapA.event_id == sim_inspiral.simulation_id JOIN coinc_event_map AS mapB ON mapB.coinc_event_id == mapA.coinc_event_id JOIN coinc_inspiral ON coinc_inspiral.coinc_event_id == mapB.event_id JOIN coinc_event on coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id WHERE mapA.table_name = "sim_inspiral" AND mapB.table_name = "coinc_event" AND injection_in_segments(sim_inspiral.geocent_end_time, sim_inspiral.geocent_end_time_ns)'

	elif table_name == dbtables.lsctables.CoincRingdownTable.tableName:
		found_query = 'SELECT sim_inspiral.*, coinc_ringdown.false_alarm_rate FROM sim_inspiral JOIN coinc_event_map AS mapA ON mapA.event_id == sim_inspiral.simulation_id JOIN coinc_event_map AS mapB ON mapB.coinc_event_id == mapA.coinc_event_id JOIN coinc_ringdown ON coinc_ringdown.coinc_event_id == mapB.event_id JOIN coinc_event on coinc_event.coinc_event_id == coinc_ringdown.coinc_event_id WHERE mapA.table_name = "sim_inspiral" AND mapB.table_name = "coinc_event" AND injection_in_segments(sim_inspiral.geocent_end_time, sim_inspiral.geocent_end_time_ns)'
	
	elif table_name == dbtables.lsctables.MultiBurstTable.tableName:
		found_query = 'SELECT sim_inspiral.*, multi_burst.false_alarm_rate FROM sim_inspiral JOIN coinc_event_map AS mapA ON mapA.event_id == sim_inspiral.simulation_id JOIN coinc_event_map AS mapB ON mapB.coinc_event_id == mapA.coinc_event_id JOIN multi_burst ON multi_burst.coinc_event_id == mapB.event_id JOIN coinc_event on coinc_event.coinc_event_id == multi_burst.coinc_event_id WHERE mapA.table_name = "sim_inspiral" AND mapB.table_name = "coinc_event" AND injection_in_segments(sim_inspiral.geocent_end_time, sim_inspiral.geocent_end_time_ns)'
	
	else:
		raise ValueError("table must be in " + " ".join(allowed_analysis_table_names()))
	
	def injection_was_made(end_time, end_time_ns, segments = segments):
		return time_within_segments(end_time, end_time_ns, segments)

	# restrict the found injections to only be within certain segments
	connection.create_function("injection_in_segments", 2, injection_was_made)

	# get the mapping of a record returned by the database to a sim
	# inspiral row. Note that this is DB dependent potentially, so always
	# do this!
	make_sim_inspiral = make_sim_inspiral_row_from_columns_in_db(connection)

	found_injections = {}

	for values in connection.cursor().execute(found_query):
		# all but the last column is used to build a sim inspiral object
		sim = make_sim_inspiral(values[:-1])
		far = values[-1]
		# update with the minimum far seen until now
		this_inj = found_injections.setdefault(sim.simulation_id, (far, sim))
		if far < this_inj[0]:
			found_injections[sim.simulation_id] = (far, sim)

	total_query = 'SELECT * FROM sim_inspiral WHERE injection_in_segments(geocent_end_time, geocent_end_time_ns)'

	total_injections = []
	for values in connection.cursor().execute(total_query):
		sim = make_sim_inspiral(values)
		total_injections.append(sim)

	return found_injections.values(), total_injections


def found_injections_below_far(found, far_thresh = float("inf")):
	"""
	This function takes an iterable of tuples of the form (far, sim)
	and gives you back all of sims that are below a given far threshold
	"""
	#FIXME this could be made faster
	output = []
	for far, sim in found:
		if far < far_thresh:
			output.append(sim)
	return output


def get_instruments_from_coinc_event_table(connection):
	"""
	This function returns a list of the instruments analyzed according to the coinc_event_table
	"""
	instruments = []
	for ifos in connection.cursor().execute('SELECT DISTINCT(instruments) FROM coinc_event WHERE instruments!=""'):
		# ignore null columns
		if ifos[0]:
			instruments.append(frozenset(lsctables.instrument_set_from_ifos(ifos[0])))
	return instruments


def get_segments(connection, xmldoc, table_name, live_time_program, veto_segments_name = None):
	segs = segments.segmentlistdict()

	if table_name == dbtables.lsctables.CoincInspiralTable.tableName:
		segs = db_thinca_rings.get_thinca_zero_lag_segments(connection, program_name = live_time_program).coalesce()
		if veto_segments_name is not None:
			veto_segs = db_thinca_rings.get_veto_segments(connection, veto_segments_name)
			segs -= veto_segs
		return segs
	elif table_name == dbtables.lsctables.CoincRingdownTable.tableName:
		segs = llwapp.segmentlistdict_fromsearchsummary(xmldoc, live_time_program).coalesce()
		if veto_segments_name is not None:
			veto_segs = ligolw_segments.segmenttable_get_by_name(xmldoc, veto_segments_name).coalesce()
			segs -= veto_segs
		return segs
	elif table_name == dbtables.lsctables.MultiBurstTable.tableName:
		if live_time_program == "omega_to_coinc": segs = llwapp.segmentlistdict_fromsearchsummary(xmldoc, live_time_program).coalesce()
		elif live_time_program == "waveburst": segs = db_thinca_rings.get_thinca_zero_lag_segments(connection, program_name = live_time_program).coalesce()
		else:
			raise ValueError("for burst tables livetime program must be one of omega_to_coinc, waveburst")
		if veto_segments_name is not None:
			# FIXME handle burst vetoes!!!
			pass
		return segs
	else:
		raise ValueError("table must be in " + " ".join(allowed_analysis_table_names()))


def get_loudest_event_far_thresholds(connection, table_name, segments = None):
	"""
	return the false alarm rate of the most rare zero-lag coinc by instruments
	"""
	query = 'CREATE TEMPORARY TABLE distinct_instruments AS SELECT DISTINCT(instruments) as instruments FROM coinc_event;'
	connection.cursor().execute(query)

	def event_in_requested_segments(end_time, end_time_ns, segments = segments):
		return time_within_segments(end_time, end_time_ns, segments)

	connection.create_function("event_in_requested_segments", 2, event_in_requested_segments)

	if table_name == dbtables.lsctables.CoincInspiralTable.tableName:
		query = 'SELECT distinct_instruments.instruments, (SELECT MIN(coinc_inspiral.combined_far) AS combined_far FROM coinc_inspiral JOIN coinc_event ON (coinc_inspiral.coinc_event_id == coinc_event.coinc_event_id) WHERE coinc_event.instruments == distinct_instruments.instruments AND NOT EXISTS(SELECT * FROM time_slide WHERE time_slide.time_slide_id == coinc_event.time_slide_id AND time_slide.offset != 0) AND event_in_requested_segments(coinc_inspiral.end_time, coinc_inspiral.end_time_ns) ) FROM distinct_instruments;'

	elif table_name == dbtables.lsctables.MultiBurstTable.tableName:
		query = 'SELECT distinct_instruments.instruments, (SELECT MIN(multi_burst.false_alarm_rate) AS combined_far FROM multi_burst JOIN coinc_event ON (multi_burst.coinc_event_id == coinc_event.coinc_event_id) WHERE coinc_event.instruments == distinct_instruments.instruments AND NOT EXISTS(SELECT * FROM time_slide WHERE time_slide.time_slide_id == coinc_event.time_slide_id AND time_slide.offset != 0) AND event_in_requested_segments(multi_burst.peak_time, multi_burst.peak_time_ns) ) FROM distinct_instruments;'

	elif table_name == dbtables.lsctables.CoincRingdownTable.tableName:
		query = 'SELECT distinct_instruments.instruments, (SELECT MIN(coinc_ringdown.false_alarm_rate) AS combined_far FROM coinc_ringdown JOIN coinc_event ON (coinc_ringdown.coinc_event_id == coinc_event.coinc_event_id) WHERE coinc_event.instruments == distinct_instruments.instruments AND NOT EXISTS(SELECT * FROM time_slide WHERE time_slide.time_slide_id == coinc_event.time_slide_id AND time_slide.offset != 0) AND event_in_requested_segments(coinc_ringdown.start_time, coinc_ringdown.start_time_ns) ) FROM distinct_instruments;'

	else:
		raise ValueError("table must be in " + " ".join(allowed_analysis_table_names()))

	output = []
	for inst, far in connection.cursor().execute(query):
		inst = frozenset(lsctables.instrument_set_from_ifos(inst))
		output.append((inst, far))

	query = 'DROP TABLE distinct_instruments'
	connection.cursor().execute(query)
	return output


def compute_search_volume_in_bins(found, total, ndbins, sim_to_bins_function):
	"""
	This program creates the search volume in the provided ndbins.  The
	first dimension of ndbins must be the distance over which to integrate.  You
	also must provide a function that maps a sim inspiral row to the correct tuple
	to index the ndbins.
	"""

	input = rate.BinnedRatios(ndbins)

	# we have one less dimension on the output
	output = rate.BinnedArray(rate.NDBins(ndbins[1:]))

	# increment the numerator with the missed injections
	[input.incnumerator(sim_to_bins_function(sim)) for sim in found]

	# increment the denominator with the total injections
	[input.incdenominator(sim_to_bins_function(sim)) for sim in total]

	# compute the dx in the distance bins REMEMBER it is the first dimension by requirement :)
	dx = input.bins()[0].upper() - input.bins()[0].lower()
	r = input.bins()[0].centres()
	# regularize by setting denoms to 1 to avoid nans
	input.regularize()
	# pull out the efficiency array, it is the ratio
	efficiency_array = input.ratio()
	output.array = (efficiency_array.T * 4. * numpy.pi * r**2 * dx).sum(-1)

	return output


def guess_distance_mass1_mass2_bins_from_sims(sims, mass1bins = 11, mass2bins = 11, distbins = 200):
	"""
	Given a list of the injections, guess at the mass1, mass2 and distance
	bins. Floor and ceil will be used to round down to the nearest integers.
	"""

	minmass1 = numpy.floor(min([sim.mass1 for sim in sims]))
	maxmass1 = numpy.ceil(max([sim.mass1 for sim in sims]))
	minmass2 = numpy.floor(min([sim.mass2 for sim in sims]))
	maxmass2 = numpy.ceil(max([sim.mass2 for sim in sims]))
	mindist = numpy.floor(min([sim.distance for sim in sims]))
	maxdist = numpy.ceil(max([sim.distance for sim in sims]))

	return rate.NDBins((rate.LogarithmicBins(mindist, maxdist, distbins), rate.LinearBins(minmass1, maxmass1, mass1bins), rate.LinearBins(minmass2, maxmass2, mass2bins)))


def guess_distance_spin1z_spin2z_bins_from_sims(sims, spin1bins = 11, spin2bins = 11, distbins = 200):
	"""
	Given a list of the injections, guess at the spin1, spin2 and distance
	bins. Floor and ceil will be used to round down to the nearest integers.
	"""

	minspin1 = numpy.floor(min([sim.spin1z for sim in sims]))
	maxspin1 = numpy.ceil(max([sim.spin1z for sim in sims]))
	minspin2 = numpy.floor(min([sim.spin2z for sim in sims]))
	maxspin2 = numpy.ceil(max([sim.spin2z for sim in sims]))
	mindist = numpy.floor(min([sim.distance for sim in sims]))
	maxdist = numpy.ceil(max([sim.distance for sim in sims]))

	return rate.NDBins((rate.LogarithmicBins(mindist, maxdist, distbins), rate.LinearBins(minspin1, maxspin1, spin1bins), rate.LinearBins(minspin2, maxspin2, spin2bins)))


def sim_to_distance_mass1_mass2_bins_function(sim):
	"""
	create a function to map a sim to a distance, mass1, mass2 NDBins based object
	"""

	return (sim.distance, sim.mass1, sim.mass2)


def sim_to_distance_spin1z_spin2z_bins_function(sim):
	"""
	create a function to map a sim to a distance, spin1z, spin2z NDBins based object
	"""

	return (sim.distance, sim.spin1z, sim.spin2z)


def symmetrize_sims(sims, col1, col2):
	"""
	duplicate the sims to symmetrize by two columns that should be symmetric.  For example mass1 and mass2
	"""
	out = []
	for sim in sims:
		out.append(sim)
		newsim = copy.deepcopy(sim)
		setattr(newsim, col1, getattr(sim, col2))
		setattr(newsim, col2, getattr(sim, col1))
		out.append(newsim)
	return out

class DataBaseSummary(object):
	"""
	This class stores summary information gathered across the databases
	"""

	def __init__(self, filelist, live_time_program = None, veto_segments_name = "vetoes", tmp_path = None, verbose = False):

		self.segments = segments.segmentlistdict()
		self.instruments = []
		self.table_name = None
		self.found_injections_by_instrument_set = {}
		self.total_injections_by_instrument_set = {}
		self.far_thresholds_by_instrument_set = {}

		for f in filelist:
			if verbose:
				print >> sys.stderr, "Gathering stats from: %s...." % (f,)
			working_filename = dbtables.get_connection_filename(f, tmp_path = tmp_path, verbose = verbose)
			connection = sqlite3.connect(working_filename)
			dbtables.DBTable_set_connection(connection)
			xmldoc = dbtables.get_xml(connection)

			sim = False

			# look for a sim inspiral table.  This is IMR work we have to have one of these :)
			try:
				sim_inspiral_table = table.get_table(xmldoc, dbtables.lsctables.SimInspiralTable.tableName)
				sim = True
			except ValueError:
				pass

			# look for the relevant table for analyses
			for table_name in allowed_analysis_table_names():
				try:
					setattr(self, table_name, table.get_table(xmldoc, table_name))
					if self.table_name is None or self.table_name == table_name:
						self.table_name = table_name
					else:
						raise ValueError("detected more than one table type out of " + " ".join(allowed_analysis_table_names()))
				except ValueError:
					setattr(self, table_name, None)

			# the non simulation databases are where we get information about segments
			if not sim:
				self.instruments += get_instruments_from_coinc_event_table(connection)
				# save a reference to the segments for this file, needed to figure out the missed and found injections
				self.this_segments = get_segments(connection, xmldoc, self.table_name, live_time_program, veto_segments_name)
				# FIXME we don't really have any reason to use playground segments, but I put this here as a reminder
				# self.this_playground_segments = segmentsUtils.S2playground(self.this_segments.extent_all())
				self.segments += self.this_segments

				# get the far thresholds for the loudest events in these databases
				for instruments_set, far in get_loudest_event_far_thresholds(connection, self.table_name):
					self.far_thresholds_by_instrument_set.setdefault(instruments_set, []).append(far)
			# get the injections
			else:
				# We need to know the segments in this file to determine which injections are found
				self.this_injection_segments = get_segments(connection, xmldoc, self.table_name, live_time_program, veto_segments_name)
				self.this_injection_instruments = []
				distinct_instruments = connection.cursor().execute('SELECT DISTINCT(instruments) FROM coinc_event WHERE instruments!=""').fetchall()
				for instruments, in distinct_instruments:
					instruments_set = frozenset(lsctables.instrument_set_from_ifos(instruments))
					self.this_injection_instruments.append(instruments_set)
					segments_to_consider_for_these_injections = self.this_injection_segments.intersection(instruments_set) - self.this_injection_segments.union(set(self.this_injection_segments.keys()) - instruments_set)
					# FIXME check to see if a maxextent option was used.  Currently only effect ligolw_rinca, but will effect ligolw_thinca someday
					if self.table_name == dbtables.lsctables.CoincRingdownTable.tableName:
						coinc_end_time_seg_param = process.get_process_params(xmldoc, "ligolw_rinca", "--coinc-end-time-segment")
						if len(coinc_end_time_seg_param) == 1:
							segments_to_consider_for_these_injections &= segmentsUtils.from_range_strings(coinc_end_time_seg_param, boundtype = float)
						else:
							# FIXME what would that mean if it is greater than one???
							raise ValueError("len(coinc_end_time_seg_param) > 1")

					found, total = get_min_far_inspiral_injections(connection, segments = segments_to_consider_for_these_injections, table_name = self.table_name)
					self.found_injections_by_instrument_set.setdefault(instruments_set, []).extend(found)
					self.total_injections_by_instrument_set.setdefault(instruments_set, []).extend(total)

			# All done
			dbtables.discard_connection_filename(f, working_filename, verbose = verbose)
			dbtables.DBTable_set_connection(None)

			# FIXME
			# Things left to do
			# 1) summarize the far threshold over the entire dataset

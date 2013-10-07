# Copyright (C) 2007--2013  Kipp Cannon
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


try:
	from fpconst import NaN, PosInf
except ImportError:
	# fpconst is not part of the standard library and might not be
	# available
	NaN = float("nan")
	PosInf = float("+inf")
import math
import sys
import traceback
import warnings


from pylal import git_version
from pylal import rate


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


#
# =============================================================================
#
#                                  Likelihood
#
# =============================================================================
#


# starting from Bayes' theorem:
#
# P(coinc is a g.w. | its parameters)
#     P(those parameters | a coinc known to be a g.w.) * P(coinc is g.w.)
#   = -------------------------------------------------------------------
#                                P(parameters)
#
#     P(those parameters | a coinc known to be a g.w.) * P(coinc is g.w.)
#   = -------------------------------------------------------------------
#     P(noise params) * P(coinc is not g.w.) + P(inj params) * P(coinc is g.w.)
#
#                       P(inj params) * P(coinc is g.w.)
#   = -------------------------------------------------------------------
#     P(noise params) * [1 - P(coinc is g.w.)] + P(inj params) * P(coinc is g.w.)
#
#                        P(inj params) * P(coinc is g.w.)
#   = ----------------------------------------------------------------------
#     P(noise params) + [P(inj params) - P(noise params)] * P(coinc is g.w.)
#
# this last form above is used below to compute the LHS
#
#          [P(inj params) / P(noise params)] * P(coinc is g.w.)
#   = --------------------------------------------------------------
#     1 + [[P(inj params) / P(noise params)] - 1] * P(coinc is g.w.)
#
#          Lambda * P(coinc is g.w.)                       P(inj params)
#   = -----------------------------------  where Lambda = ---------------
#     1 + (Lambda - 1) * P(coinc is g.w.)                 P(noise params)
#
# Differentiating w.r.t. Lambda shows the derivative is always positive, so
# thresholding on Lambda is equivalent to thresholding on P(coinc is a g.w.
# | its parameters).  The limits:  Lambda=0 --> P(coinc is a g.w. | its
# parameters)=0, Lambda=+inf --> P(coinc is a g.w. | its parameters)=1.  We
# interpret Lambda=0/0 to mean P(coinc is a g.w. | its parameters)=0 since
# although it can't be noise it's definitely not a g.w..  We do not protect
# against NaNs in the Lambda = +inf/+inf case.


class LikelihoodRatio(object):
	"""
	Class for computing signal hypothesis / noise hypothesis likelihood
	ratios from the measurements in a
	snglcoinc.CoincParamsDistributions instance.
	"""
	def __init__(self, coinc_param_distributions):
		# check input
		if set(coinc_param_distributions.background_rates) != set(coinc_param_distributions.injection_rates):
			raise ValueError("distribution density name mismatch:  found background data with names %s and injection data with names %s" % (", ".join(sorted(coinc_param_distributions.background_rates)), ", ".join(sorted(coinc_param_distributions.injection_rates))))
		for name, binnedarray in coinc_param_distributions.background_rates.items():
			if len(binnedarray.array.shape) != len(coinc_param_distributions.injection_rates[name].array.shape):
				raise ValueError("background data with name %s has shape %s but injection data has shape %s" % (name, str(binnedarray.array.shape), str(coinc_param_distributions.injection_rates[name].array.shape)))

		# construct interpolators from the distribution data
		self.background_rates = dict((name, rate.InterpBinnedArray(binnedarray)) for name, binnedarray in coinc_param_distributions.background_rates.items())
		self.injection_rates = dict((name, rate.InterpBinnedArray(binnedarray)) for name, binnedarray in coinc_param_distributions.injection_rates.items())

	def P(self, params):
		if params is None:
			return None, None
		P_bak = 1.0
		P_inj = 1.0
		for name, value in params.items():
			P_bak *= self.background_rates[name](*value)
			P_inj *= self.injection_rates[name](*value)
		return P_bak, P_inj

	def __call__(self, params):
		"""
		Compute the likelihood ratio for the hypothesis that the
		list of events are the result of a gravitational wave.  The
		likelihood ratio is the ratio P(inj params) / P(noise
		params).  The probability that the events are the result of
		a gravitiational wave is a monotonically increasing
		function of the likelihood ratio, so ranking events from
		"most like a gravitational wave" to "least like a
		gravitational wave" can be performed by calculating the
		likelihood ratios, which has the advantage of not requiring
		a prior probability to be provided.
		"""
		P_bak, P_inj = self.P(params)
		if P_bak is None and P_inj is None:
			return None
		if P_bak == 0.0 and P_inj == 0.0:
			# "correct" answer is 0, not NaN, because if a
			# tuple of events has been found in a region of
			# parameter space where the probability of an
			# injection occuring is 0 then there is no way this
			# is an injection.  there is also, aparently, no
			# way it's a noise event, but that's irrelevant
			# because we are supposed to be computing something
			# that is a monotonically increasing function of
			# the probability that an event tuple is a
			# gravitational wave, which is 0 in this part of
			# the parameter space.
			return 0.0
		if math.isinf(P_bak) and math.isinf(P_inj):
			warnings.warn("inf/inf encountered")
			return NaN
		try:
			return  P_inj / P_bak
		except ZeroDivisionError:
			# P_bak == 0.0, P_inj != 0.0.  this is a
			# "guaranteed detection", not a failure
			return PosInf


#
# =============================================================================
#
#                              Library Interface
#
# =============================================================================
#


#
# Core routine
#


def assign_likelihood_ratios(connection, coinc_def_id, offset_vectors, vetoseglists, events_func, veto_func, likelihood_ratio_func, likelihood_params_func, verbose = False, params_func_extra_args = ()):
	"""
	Assigns likelihood ratio values to coincidences.
	"""
	#
	# Convert offset vector keys to strings so that we can use the
	# dictionary inside an SQL query (they might be
	# glue.ligolw.ilwd_char objects)
	#

	offset_vectors = dict((unicode(time_slide_id), offset_vector) for time_slide_id, offset_vector in offset_vectors.items())

	#
	# Create a cursor object for events_func() to reuse
	#

	cursor = connection.cursor()

	#
	# Construct the in-SQL likelihood ratio function.  Rely on Python's
	# closure mechanism to retain all local variables at the time of
	# this function's creation for use inside the function.
	#

	def likelihood_ratio(coinc_event_id, time_slide_id):
		try:
			return likelihood_ratio_func(likelihood_params_func([event for event in events_func(cursor, coinc_event_id) if veto_func(event, vetoseglists)], offset_vectors[time_slide_id], *params_func_extra_args))
		except:
			traceback.print_exc()
			raise

	connection.create_function("likelihood_ratio", 2, likelihood_ratio)

	#
	# Iterate over all coincs, assigning likelihood ratios.
	#

	if verbose:
		print >>sys.stderr, "computing likelihood ratios ..."

	connection.cursor().execute("""
UPDATE
	coinc_event
SET
	likelihood = likelihood_ratio(coinc_event_id, time_slide_id)
WHERE
	coinc_def_id == ?
	""", (unicode(coinc_def_id),))

	#
	# Done
	#

	connection.commit()
	cursor.close()


#
# Burst-specific interface
#


def sngl_burst_events_func(cursor, coinc_event_id, row_from_cols):
	return map(row_from_cols, cursor.execute("""
SELECT
	sngl_burst.*
FROM
	sngl_burst
	JOIN coinc_event_map ON (
		coinc_event_map.table_name == 'sngl_burst'
		AND coinc_event_map.event_id == sngl_burst.event_id
	)
WHERE
	coinc_event_map.coinc_event_id == ?
	""", (coinc_event_id,)))


def sngl_burst_veto_func(event, vetoseglists):
	# return True if event should be *retained*
	return event.ifo not in vetoseglists or event.get_peak() not in vetoseglists[event.ifo]


def ligolw_burca2(database, likelihood_ratio, params_func, verbose = False, params_func_extra_args = ()):
	"""
	Assigns likelihood ratio values to excess power coincidences.
	database is pylal.SnglBurstUtils.CoincDatabase instance, and
	likelihood_ratio is a LikelihoodRatio class instance.
	"""
	#
	# Run core function
	#

	assign_likelihood_ratios(
		connection = database.connection,
		coinc_def_id = database.bb_definer_id,
		offset_vectors = database.time_slide_table.as_dict(),
		vetoseglists = database.vetoseglists,
		events_func = lambda cursor, coinc_event_id: sngl_burst_events_func(cursor, conic_event_id, database.sngl_burst_table.row_from_cols),
		veto_func = sngl_burst_veto_func,
		likelihood_ratio_func = likelihood_ratio,
		likelihood_params_func = params_func,
		verbose = verbose,
		params_func_extra_args = params_func_extra_args
	)

	#
	# Done
	#

	return database

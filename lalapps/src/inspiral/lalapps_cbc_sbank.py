# Copyright (C) 2011  Nickolas Fotopoulos, Stephen Privitera
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

from __future__ import division

from time import strftime
from collections import deque
import numpy as np
import sys

from scipy.interpolate import UnivariateSpline
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from pylal.inspiral_metric import compute_metric
from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries
from pylal.xlal.datatypes.snglinspiraltable import SnglInspiralTable

from optparse import OptionParser

#from sbank import git_version FIXME
from lalinspiral.sbank.bank import Bank
from lalinspiral.sbank.tau0tau3 import proposals
from lalinspiral.sbank.psds import noise_models, read_psd, get_PSD
from lalinspiral.sbank.waveforms import waveforms

import lal

class ContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(ContentHandler)

usage = """

lalapps_cbc_sbank: This program generates a template bank for compact
binary searches covering a given region of mass and spin parameter
space. The program supports a number of waveform approximants and is
extensible to other waveform approximants as they become available.

Example command lines:

** Generate a template bank of positively aligned-spin
inspiral-merger-ringdown binary black hole waveforms for use in an
aLIGO search.

lalapps_cbc_sbank --approximant IMRPhenomC --aligned-spin \\
        --mass1-min 15.0 --mass1-max 25.0 \\
        --spin1-min 0.0 --spin1-max 0.5 \\
        --match-min 0.97 --flow 20.0 --noise-model aLIGOZeroDetHighPower \\
        --instrument H1 --gps-start-time 961545543  --gps-end-time 962150343 \\
        --user-tag BBH-IMRPhenomC-aLIGOZeroDetHighPower --verbose


** Generate a template bank of mildly spinning neutron stars and
highly spinning black holes using inspiral-only waveforms for use in
an aLIGO search. Approximate the match calculation using the
semi-analytic expression for the overlap metric.

lalapps_cbc_sbank --approximant TaylorF2RedSpin --aligned-spin --use-metric \\
        --mass1-min 1.0 --mass1-max 2.0 \\
        --mass2-min 5.0 --mass1-max 10.0 \\
        --spin1-min 0.0 --spin1-max 0.05 \\
        --spin2-min 0.0 --spin2-max 0.5 \\
        --match-min 0.97 --flow 20.0 --noise-model aLIGOZeroDetHighPower \\
        --instrument H1 --gps-start-time 961545543  --gps-end-time 962150343 \\
        --user-tag NSBH-TaylorF2RedSpin-aLIGOZeroDetHighPower --verbose


** Generate a template bank of mildly spinning binary neutron star
inspiral-only waveforms for use in an aLIGO search. Approximate the
match calculation using the semi-analytic expression for the overlap
metric.

lalapps_cbc_sbank --approximant TaylorF2RedSpin --aligned-spin --use-metric \\
        --mass1-min 1.0 --mass1-max 2.0 \\
        --spin1-min 0.0 --spin1-max 0.05 \\
        --match-min 0.97 --flow 20.0 --noise-model aLIGOZeroDetHighPower \\
        --instrument H1 --gps-start-time 961545543  --gps-end-time 962150343 \\
        --user-tag BNS-TaylorF2RedSpin-aLIGOZeroDetHighPower --verbose


For large parameter spaces with many templates, it is recommended that
you split the space into smaller sub-regions and ligolw_add the
resulting banks. One can also seed the template placement process with
a pre-generated bank, produced for instance by lalapps_tmpltbank, and
SBank will fill in whichever gaps remain.
"""


def parse_command_line():

    parser = OptionParser(usage = usage)

    #
    # waveform options
    #
    parser.add_option("--approximant", choices=waveforms.keys(), metavar='|'.join(waveforms.keys()), default=None, help="Required. Specify the approximant to use for waveform generation.")
    parser.add_option("--use-metric", action="store_true", default=False, help="Use analytic approximation to the numerical match calculation (if available).")

    #
    # mass parameter options
    #
    parser.add_option("--mass1-min",help="Required. Set minimum mass of the first component.", type="float", metavar="MASS")
    parser.add_option("--mass1-max",help="Required. Set maximum mass of the first component.", type="float", metavar="MASS")
    parser.add_option("--mass2-min",help="Set minimum mass of the second component. If not specified, the mass limits provided on the first component will be assumed for the second component.", type="float", metavar="MASS")
    parser.add_option("--mass2-max",help="Set maximum mass of the second component. If not specified, the mass limits provided on the first component will be assumed for the second component.", type="float", metavar="MASS")
    parser.add_option("--mtotal-min", help="Set minimum total mass of the system.", type="float", metavar="MASS")
    parser.add_option("--mtotal-max", help="Set maximum total mass of the system.",  type="float", metavar="MASS")
    parser.add_option("--mratio-min", dest="qmin", help="Set minimum allowed mass ratio of the system (convention is that q=m1/m2).", metavar="RATIO", type="float", default=1.0)
    parser.add_option("--mratio-max", dest="qmax", help="Set maximum allowed mass ratio of the system (convention is that q=m1/m2).", metavar="RATIO", type="float")

    #
    # spin parameter options
    #
    parser.add_option("--spin1-min", help="Set minimum allowed value for the spin of the first component. If spins are aligned, this parameter is interpreted as the projection of the spin vector along the orbital angualr momentum and can be positive or negative. If the spins are not aligned, this parameter is interpreted as the magnitude of the spin vector and must be positive.", type="float", default = -1.0, metavar="SPIN")
    parser.add_option("--spin1-max", help="Set maximum allowed value for the spin of the first component.", type="float", default = 1.0, metavar="SPIN")
    parser.add_option("--spin2-min", help="Set minimum allowed value for the spin of the second component. If not specified, the spin2 limits will equal the spin1 limits.", type="float", default = None, metavar="SPIN")
    parser.add_option("--spin2-max", help="Set maximum allowed value for the spin of the second component.", type="float", default = None, metavar="SPIN")
    parser.add_option("--aligned-spin", action="store_true", default=False, help="Only generate templates whose spins are parallel to the orbital angular momentum.")

    #
    # initial condition options
    #
    parser.add_option("--seed", help="Set the seed for the random number generator used by SBank for waveform parameter (masss, spins, ...) generation.", metavar="INT", default=1729, type="int")
    parser.add_option("--bank-seed",help="Initialize the bank with specified template bank. For instance, one might generate a template bank by geomtretic/lattice placement methods and use SBank to \"complete\" the bank.", metavar="FILE")

    #
    # noise model options
    #
    parser.add_option("--noise-model", choices=noise_models.keys(), metavar='|'.join(noise_models.keys()), default="aLIGOZeroDetHighPower", help="Choose a noise model for the PSD from a set of available analytical model.")
    parser.add_option("--reference-psd", help="Read PSD from an xml file instead of using analytical noise model.", metavar="FILE")

    #
    # match calculation options
    #
    parser.add_option("--flow", type="float", help="Required. Set the low-frequency cutoff to use for the match caluclation.")
    parser.add_option("--match-min",help="Set minimum match of the bank. Note that since this is a stochastic process, the requested minimal match may not be strictly guaranteed but should be fulfilled on a statistical basis. Default: 0.95.", type="float", default=0.95)
    parser.add_option("--convergence-threshold", metavar="N", help="Set the criterion for convergence of the stochastic bank. The code terminates when there are N rejected proposals for each accepted proposal, averaged over the last ten acceptances. Default 1000.", type="int", default=1000)

    #
    # output options
    #
    parser.add_option("--instrument", metavar="IFO", help="Specify the instrument for which to generate a template bank. This option is used for naming of the output file but also for reading in PSDs or template bank seeds from file.")
    parser.add_option("--gps-start-time", type="int", default=0, help="GPS time of start. Used only for naming of output file.", metavar="INT")
    parser.add_option("--gps-end-time", type="int", default=999999999, help="GPS time of end. Used only for naming of output file", metavar="INT")
    parser.add_option("--user-tag", default=None, help="Apply descriptive tag to output filename.")
    parser.add_option("--verbose", default=False,action="store_true", help="Be verbose and write diagnostic information out to file.")

    #
    # deprecated options
    #
    parser.add_option("--mchirp-boundaries-file", metavar="FILE", help="Deprecated. File containing chirp mass bin boundaries")
    parser.add_option("--mchirp-boundaries-index", metavar="INDEX", type="int", help="Deprecated. Integer index into --mchirp-boundaries-file line number such that boundaries[INDEX] is taken as --mchirp-min and boundaries[INDEX + 1] is taken as --mchirp-max")
    parser.add_option("--mchirp-min", help="Deprecated. Set minimum chirp-mass of the system (in solar masses)", type="float")
    parser.add_option("--mchirp-max", help="Deprecated. Set maximum chirp-mass of the system (in solar masses)", type="float")

    opts, args = parser.parse_args()

    #
    # check for required arguments
    #
    for opt in ("flow", "match_min", "mass1_min", "mass1_max", "instrument"):
        if getattr(opts, opt) is None:
            parser.error("--%s is required" % opt.replace("_", "-"))

    #
    # check for argument consistency
    #
    if opts.qmin < 1:
        parser.error("Mass ratio is assumed to be >= 1.")

    if not opts.spin2_min:
        opts.spin2_min = opts.spin1_min

    if not opts.spin2_max:
        opts.spin2_max = opts.spin1_max

    if not -1 <= opts.spin1_min <= opts.spin1_max <=1:
        raise ValueError("unphysical spin bounds: [%.2f, %.2f]" % (opts.spin1_min, opts.spin1_max))

    if not -1 <= opts.spin2_min <= opts.spin2_max <=1:
        raise ValueError("unphysical spin bounds: [%.2f, %.2f]" % (opts.spin2_min, opts.spin2_max))

    if opts.approximant in ["TaylorF2RedSpin", "IMRPhenomB","SEOBNRv1"] and not opts.aligned_spin:
        parser.error("--aligned-spin is required for the %s approximant" % opts.approximant)

    if (opts.mchirp_boundaries_file is not None) ^ (opts.mchirp_boundaries_index is not None):
        parser.error("must supply both --mchirp-boundaries-file and --mchirp-boundaries-index or neither")

    if opts.mchirp_boundaries_file and (opts.mchirp_min or opts.mchirp_max):
        parser.error("--mchirp-boundaries-file supercedes --mchirp-min and --mchirp-max")

    if opts.mchirp_boundaries_file:
        boundaries = [float(line) for line in open(opts.mchirp_boundaries_file)]
        if opts.mchirp_boundaries_index > len(boundaries):
            raise ValueError("mchirp boundaries file not long enough for requested index")

        if opts.mchirp_boundaries_index > 0:
            opts.mchirp_min = float(boundaries[opts.mchirp_boundaries_index - 1])
        if opts.mchirp_boundaries_index + 1 < len(boundaries):
            opts.mchirp_max = float(boundaries[opts.mchirp_boundaries_index])

    return opts, args

opts, args = parse_command_line()
np.random.mtrand.seed(opts.seed)

# prepare a new XML document
xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())

# Prepare process table with information about the current program
opts_dict = dict((k, v) for k, v in opts.__dict__.iteritems() if v is not False and v is not None)
#process = ligolw_process.register_to_xmldoc(xmldoc, "lalapps_cbc_sbank", FIXME
#    opts_dict, version=git_version.tag or git_version.id, FIXME
#    cvs_repository="sbank", cvs_entry_time=git_version.date) FIXME
process = ligolw_process.register_to_xmldoc(xmldoc, "lalapps_cbc_sbank",
    opts_dict, version="no version",
    cvs_repository="sbank", cvs_entry_time=strftime('%Y/%m/%d %H:%M:%S'))

# Choose waveform approximant
waveform = waveforms[opts.approximant]

# Choose noise model
if opts.reference_psd is not None:
    psd = read_psd(opts.reference_psd)[opts.instrument]
    f_orig = psd.f0 + np.arange(len(psd.data)) * psd.deltaF
    interpolator = UnivariateSpline(f_orig, np.log(psd.data), s=0)
    noise_model = lambda g: np.exp(interpolator(g))
else:
    noise_model = noise_models[opts.noise_model]

# Seed the bank
if opts.bank_seed is None:
    # seed the process with an empty bank
    # the first proposal will always be accepted
    bank = Bank(waveform, noise_model, opts.flow, opts.use_metric)
else:
    # seed bank with input bank. we do not prune the bank
    # for overcoverage, but take it as is
    if opts.verbose:
        print>>sys.stdout,"Seeding the template bank..."
    tmpdoc = utils.load_filename(opts.bank_seed, contenthandler=ContentHandler)
    sngl_inspiral = table.get_table(tmpdoc, lsctables.SnglInspiralTable.tableName)
    bank = Bank.from_sngls(sngl_inspiral, waveform, noise_model, opts.flow, opts.use_metric)

    # update mchirp bounds
    # FIXME store boundaries in metadata of bank seed file
    A0 = 5. / (256 * (lal.PI * opts.flow)**(8./3)) # eqn B3
    if opts.mchirp_min is None:
        opts.mchirp_min = min([b._mchirp for b in bank])
    if opts.mchirp_max is None:
        opts.mchirp_max = max([b._mchirp for b in bank])

    tmpdoc.unlink()
    del sngl_inspiral, tmpdoc
    if opts.verbose:
        print>>sys.stdout,"\tinitial bank size: %d"%len(bank)
        print>>sys.stdout,"Filling regions of parameter space not covered by bank seed..."

# Choose proposal function
# TODO: Implement more of these
# populate params dictionary to be passed to the generators
params = {'mass1': (opts.mass1_min, opts.mass1_max),
          'mass2': (opts.mass2_min, opts.mass2_max),
          'mtotal': (opts.mtotal_min, opts.mtotal_max),
          'mratio': (opts.qmin, opts.qmax),
          'mchirp': (opts.mchirp_min, opts.mchirp_max),
	  'spin1': (opts.spin1_min, opts.spin1_max),
	  'spin2': (opts.spin2_min, opts.spin2_max)
	  }

# get the correct generator
proposal = proposals[opts.approximant](opts.flow, **params)

# For robust convergence, ensure that an average of kmax/len(ks) of
# the last len(ks) proposals have been rejected by SBank.
ks = deque(10*[0], maxlen=10)

k = 0 # k is nprop per iteration
nprop = 1  # count total number of proposed templates
status_format = "bank size: %d\tproposed: %d\t" + "\t".join("%s: %s" % name_format for name_format in zip(waveform.param_names, waveform.param_formats))
while (k + float(sum(ks)))/len(ks) < opts.convergence_threshold:
    tmplt = waveform(*proposal.next(), bank=bank)
    k += 1
    nprop += 1
    if not bank.covers(tmplt, opts.match_min):
        bank.insort(tmplt)
        if opts.verbose:
            print >>sys.stdout, status_format % ((len(bank), k) + tmplt.params)
        ks.append(k)
        k = 0

print "total number of proposed templates: %d" % nprop
print "total number of match calculations: %d" % bank._nmatch
print "final bank size: %d" % len(bank)

bank.clear()  # clear caches

# Set up PSD for metric computation; calling into pylal, so need pylal types
psd = REAL8FrequencySeries(name="psd", f0=0., deltaF=1., data=get_PSD(1., opts.flow, 1570., noise_model))

# insert our rows
# Replace row with C datatype; nice side effect: initializes elements to 0 or ""
lsctables.SnglInspiralTable.RowType = SnglInspiralTable
tbl = lsctables.New(lsctables.SnglInspiralTable)
xmldoc.childNodes[-1].appendChild(tbl)
for template in bank:
    row = template.to_sngl()
    row.ifo = opts.instrument
    row.process_id = process.process_id
    row.Gamma0, row.Gamma1, row.Gamma2, row.Gamma3, row.Gamma4, row.Gamma5,\
        row.Gamma6, row.Gamma7, row.Gamma8, row.Gamma9 = \
        compute_metric(opts.flow, 1570., 4, row.tau0, row.tau3, psd)
    tbl.append(row)

# write out the document
ligolw_process.set_process_end_time(process)

# FIXME output naming conventions to match IHOPE/tmpltbank break sbank_pipe
if opts.user_tag:
    fout = "%s-SBANK_%s-%d-%d.xml.gz" % (opts.instrument, opts.user_tag, opts.gps_start_time, opts.gps_end_time-opts.gps_start_time)
else:
    fout = "%s-SBANK-%d-%d.xml.gz" % (opts.instrument, opts.gps_start_time, opts.gps_end_time-opts.gps_start_time)

utils.write_filename(xmldoc, fout,  gz=fout.endswith("gz"))

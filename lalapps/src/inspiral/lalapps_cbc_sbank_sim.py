# Copyright (C) 2011,2012  Nickolas Fotopoulos, Melissa Frei, Stephen Privitera
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

from operator import attrgetter
from optparse import OptionParser

from time import strftime
import numpy as np
from h5py import File as H5File

from scipy.interpolate import UnivariateSpline
from glue.ligolw import table
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import ligolw_add

from lalinspiral.sbank.bank import Bank
#from sbank import git_version FIXME
from lalinspiral.sbank.waveforms import waveforms
from lalinspiral.sbank.psds import noise_models, read_psd

class ContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(ContentHandler)

def ligolw_table_to_array(tab):
    # get mapping of LIGO_LW types to numpy numeric types; add string types
    from glue.ligolw.types import ToNumPyType
    type_map = ToNumPyType.copy()
    type_map[u"ilwd:char"] = "S40"
    type_map[u"lstring"] = "S32"

    # construct numpy dtype and row accessor
    name_type_pairs = tab.validcolumns.items()
    name_type_pairs.sort()
    dtype = np.dtype({
        "names": [n for n, _ in name_type_pairs],
        "formats": [type_map[t] for _, t in name_type_pairs]})
    extract_in_order = attrgetter(*dtype.names)

    # populate array
    arr = np.empty(len(tab), dtype=dtype)
    for i, row in enumerate(tab):
        arr[i] = extract_in_order(row)

    return arr

def ligolw_copy_process(xmldoc_src, xmldoc_dest):
    """
    We want to copy over process and process_params tables to eventually
    merge them.
    """
    proc = table.get_table(xmldoc_src, lsctables.ProcessTable.tableName)
    pp = table.get_table(xmldoc_src, lsctables.ProcessParamsTable.tableName)

    xmldoc_dest.appendChild(ligolw.LIGO_LW())
    xmldoc_dest.childNodes[-1].appendChild(proc)
    xmldoc_dest.childNodes[-1].appendChild(pp)

## get arguments
def parse_command_line():
    parser = OptionParser()
    parser.add_option("--user-tag",default="banksim")
    parser.add_option("--injection-file",default = None)
    parser.add_option("--injection-approx",default = None)
    parser.add_option("--mratio-max",default = float('inf'), type="float", help="Cull injections based on mass ratio")
    parser.add_option("--mtotal-min",default = 0, type="float", help="Cull injections based on total mass")
    parser.add_option("--mtotal-max",default = float('inf'), type="float", help="Cull injections based on total mass")
    parser.add_option("--template-bank",default = [], action="append", metavar="FILE[:APPROX]")
    parser.add_option("--template-approx",default = None, help="Approximant to be used for templates when not specified with file.")
    parser.add_option("--noise-model", choices=noise_models.keys(), default="aLIGOZeroDetHighPower", help="Choose a noise model for the PSD from a set of available analytical model")
    parser.add_option("--reference-psd", help="Read PSD from xml instead of using analytical noise model.")
    parser.add_option("--instrument", metavar="IFO", help="Specify the instrument for which to generate a template bank.")
    parser.add_option("--flow",default = 40.0, type="float")
    parser.add_option("--verbose",default = False, action="store_true")
    parser.add_option("--use-gpu-match", default=False, action="store_true",
        help="Perform match calculations using the first available GPU")
    parser.add_option("--cache-waveforms", default = False, action="store_true", help="A given waveform in the template bank will be used many times throughout the bank simulation process. You can save a considerable amount of CPU by caching the waveform from the first time it is generated; however, do so only if you are sure that storing the waveforms in memory will not overload the system memory.")
    parser.add_option("--neighborhood-size", metavar="N", default = 1.0, type="float", help="Specify the window size in seconds to define \"nearby\" templates used to compute the match against each injection. The neighborhood is chosen symmetric about the injection; \"nearby\" is defined using the option --neighborhood-type. The default value of 1.0 is *not a guarantee of performance*. Choosing the neighborhood too small may lead to the reporting of lower-than-actual fitting factors.")
    parser.add_option("--neighborhood-param", default="tau0", choices=["tau0", "dur"], help="Choose how the neighborhood is sorted for match calculations.")

    opts, args = parser.parse_args()

    if opts.reference_psd and not opts.instrument:
        raise ValueError, "--instrument is a required option when specifying reference PSD"

    return opts, args

opts, args = parse_command_line()

# Choose noise model
if opts.reference_psd is not None:
    psd = read_psd(opts.reference_psd)[opts.instrument]
    f_orig = psd.f0 + np.arange(len(psd.data)) * psd.deltaF
    interpolator = UnivariateSpline(f_orig, np.log(psd.data), s=0)
    noise_model = lambda g: np.exp(interpolator(g))
else:
    noise_model = noise_models[opts.noise_model]

usertag = opts.user_tag  # use to label output
inj_file = opts.injection_file # pre-sorted by mchirp
inj_approx = waveforms[opts.injection_approx]
flow = opts.flow
verbose = opts.verbose
if opts.use_gpu_match:
    from lalinspiral.sbank.overlap_cuda import compute_match, create_workspace_cache
    import sbank.waveforms
    import sbank.bank
    sbank.waveforms.compute_match = compute_match
    sbank.bank.create_workspace_cache = create_workspace_cache

# initialize output doc; real hdf5 output file and virtual xml file for metadata
fake_xmldoc = ligolw.Document()
fake_xmldoc.appendChild(ligolw.LIGO_LW())
opts_dict = dict((k, v) for k, v in opts.__dict__.iteritems() if v is not False and v is not None)
#process = ligolw_process.register_to_xmldoc(fake_xmldoc, "lalapps_cbc_sbank_sim", FIXME
#    opts_dict, version=git_version.tag or git_version.id, FIXME
#    cvs_repository="sbank", cvs_entry_time=git_version.date) FIXME
process = ligolw_process.register_to_xmldoc(fake_xmldoc, "lalapps_cbc_sbank_sim",
    opts_dict, version="no version",
    cvs_repository="sbank", cvs_entry_time=strftime('%Y/%m/%d %H:%M:%S'))
h5file = H5File("%s.h5" % usertag, "w")

# load templates
#
# initialize the bank
#
bank = Bank(noise_model, flow, use_metric=False, cache_waveforms=opts.cache_waveforms, nhood_size=opts.neighborhood_size, nhood_param=opts.neighborhood_param)
for file_approx in opts.template_bank:

    # if no approximant specified, take from opts.template_approx
    if len(file_approx.split(":")) == 1:
        seed_file = file_approx
        approx = opts.template_approx
    else:
        # if this fails, you have an input error
        seed_file, approx = file_approx.split(":")

    # add templates to bank
    tmpdoc = utils.load_filename(seed_file, contenthandler=ContentHandler)
    sngl_inspiral = table.get_table(tmpdoc, lsctables.SnglInspiralTable.tableName)
    seed_waveform = waveforms[approx]
    bank.add_from_sngls(sngl_inspiral, seed_waveform)

    if opts.verbose:
        print "Added %d %s templates from %s to bank." % (len(sngl_inspiral), approx, seed_file)

    tmpdoc.unlink()
    del sngl_inspiral, tmpdoc

if opts.verbose:
    print "Initialized the template bank with %d templates." % len(bank)

# write bank to h5 file, but note that from_sngls() has resorted the
# bank by neighborhood_param
sngls = lsctables.SnglInspiralTable()
for s in bank._templates:
    sngls.append(s.to_sngl())
h5file.create_dataset("/sngl_inspiral", data=ligolw_table_to_array(sngls), compression='gzip', compression_opts=1)
h5file.flush()
del sngls[:]

# pick injection parameters
xmldoc2 = utils.load_filename(inj_file, contenthandler=ContentHandler)
ligolw_copy_process(xmldoc2, fake_xmldoc)
sims = table.get_table(xmldoc2, lsctables.SimInspiralTable.tableName)

#
# sometime inspinj doesn't give us everything we want, so we give the
# user a little extra control here over the injections acutally used
#
newtbl = lsctables.SimInspiralTable()
for s in sims:
    if 1./opts.mratio_max <= s.mass1/s.mass2 <= opts.mratio_max and opts.mtotal_min <= s.mass1 + s.mass2 <= opts.mtotal_max:
        s.polarization = np.random.uniform(0, 2*np.pi)
        newtbl.append(s)
sims = newtbl

h5file.create_dataset("/sim_inspiral", data=ligolw_table_to_array(sims), compression='gzip', compression_opts=1)
h5file.flush()
inj_wfs = Bank.from_sims(sims, inj_approx, noise_model, flow)
del xmldoc2, sims[:]
if verbose:
    print "Loaded %d injections" % len(inj_wfs)
    inj_format = "".join("%s: %s   " % name_format for name_format in zip(inj_approx.param_names, inj_approx.param_formats))

# main worker loop
match_map = np.empty(len(inj_wfs), dtype=[("inj_index", np.uint32), ("inj_sigmasq", np.float32), ("match", np.float32), ("best_match_tmplt_index", np.uint32)])
for inj_ind, inj_wf in enumerate(inj_wfs):

    if verbose:
        print "injection %d/%d" % (inj_ind+1, len(inj_wfs))
        print inj_format % inj_wf.params

    # NB: sigmasq set during argmax_match
    match_tup = bank.argmax_match(inj_wf)

    if verbose:
        print "\tbest matching template:  ",
        print bank._templates[match_tup[1]].params
        print "\tbest match:  %f\n" % match_tup[0]

    match_map[inj_ind] = (inj_ind, inj_wf.sigmasq) + match_tup
    inj_wf.clear()  # prune inj waveform


if verbose:
    print "total number of match calculations:", bank._nmatch

# merge process and process_params tables, then complete ourselves
table.reset_next_ids((lsctables.ProcessTable, lsctables.ProcessParamsTable))
ligolw_add.reassign_ids(fake_xmldoc)
ligolw_add.merge_ligolws(fake_xmldoc)
ligolw_add.merge_compatible_tables(fake_xmldoc)
ligolw_process.set_process_end_time(process)

# output
h5file.create_dataset("/match_map", data=match_map, compression='gzip', compression_opts=1)
proc = table.get_table(fake_xmldoc, lsctables.ProcessTable.tableName)
for p in proc:
    p.cvs_entry_time = 0
    p.end_time = 0
h5file.create_dataset("/process", data=ligolw_table_to_array(proc))
pp = table.get_table(fake_xmldoc, lsctables.ProcessParamsTable.tableName)
h5file.create_dataset("/process_params", data=ligolw_table_to_array(pp))
h5file.close()

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

from optparse import OptionParser
import sys

import numpy as np
from h5py import File as H5File

## get arguments
def parse_command_line():
    parser = OptionParser(description="Merge banksim .h5 files together, maximizing match for each injection over bank fragments as necessary. We assume that all banksims were given the identical HL-INJECTIONS file and disjoint bank fragments.")
    parser.add_option("-o", "--output", help="Write output to hdf5 output")
    parser.add_option("--independent",action='store_true', dest='independent', default=False, help="Independent injection sets, no need to perform maximisation step")

    opts, args = parser.parse_args()

    return opts, args

opts, args = parse_command_line()

if not args: sys.exit("Nothing to do")

# initialize output file
outfile = H5File(opts.output, "w")

if opts.independent:
	out_sims = []
	out_sngls = []
	out_match = []
	out_process = []
	out_processParams = []
	prevNinj = 0
	out_map = []
	print args[0];
	with H5File(args[0], "r") as infile:
		out_sngls.extend(infile["/sngl_inspiral"]);
		out_sngls_dtype = infile["/sngl_inspiral"].dtype
	for f in args[:]:
		with H5File(f, "r") as infile:
			out_process.extend(infile["/process"]);
			out_processParams.extend(infile["/process_params"]);
			out_sims.extend(infile["/sim_inspiral"]);
			match_map_tmp = np.zeros(shape=infile["/match_map"].shape, dtype=infile["/match_map"].dtype)
			for inj_id, inj_sigmasq, match, best_tmplt_id in infile["/match_map"]:
				match_map_tmp[inj_id] = (inj_id+prevNinj, inj_sigmasq, match, best_tmplt_id)
			out_match.extend(match_map_tmp);
			prevNinj += len(infile["/match_map"])

			out_process_dtype = infile["/process"].dtype
			out_processParams_dtype = infile["/process_params"].dtype
			out_sims_dtype = infile["/sim_inspiral"].dtype
			out_match_dtype = infile["/match_map"].dtype

	outfile.create_dataset("/process", data=np.array(out_process, dtype=out_process_dtype), compression='gzip', compression_opts=1)
	outfile.create_dataset("/process_params", data=np.array(out_processParams, dtype=out_processParams_dtype), compression='gzip', compression_opts=1)
	outfile.create_dataset("/sim_inspiral", data=np.array(out_sims, dtype=out_sims_dtype), compression='gzip', compression_opts=1)
	outfile.create_dataset("/sngl_inspiral", data=np.array(out_sngls, dtype=out_sngls_dtype), compression='gzip', compression_opts=1)
	outfile.create_dataset("/match_map", data=np.array(out_match, dtype=out_match_dtype), compression='gzip', compression_opts=1)
else:
	# based on the first file...
	with H5File(args[0], "r") as infile:
	    # populate sims completely
	    out_sims = outfile.create_dataset("/sim_inspiral", data=infile["sim_inspiral"], compression='gzip', compression_opts=1)
	    outfile.flush()
	
	    # copy process and process params table
	    # FIXME: only takes metadata from first file!
	    outfile.create_dataset("/process", data=infile["/process"])
	    outfile.create_dataset("/process_params", data=infile["/process_params"])
	
	    # but we'll have to build up sngls and the match map as we go
	    out_map = np.zeros(shape=infile["/match_map"].shape,
	                       dtype=infile["/match_map"].dtype)
	    out_sngls_dtype = infile["/sngl_inspiral"].dtype
	out_sngls = []
	
	# build sngls and match map
	for f in args[:]:
	    with H5File(f, "r") as infile:
	        # sanity check that we're probably using the same injection set
	        assert len(infile["/sim_inspiral"]) == len(out_sims)
	
	        # sanity check that the sim has one match_map entry per inj
	        assert len(infile["/match_map"]) == len(out_sims)
	
	        # tabulate best match per injection
	        # NB: inj_ids have the same meaning across files, but best_tmplt_ids
	        # require remapping to the output order.
	        for inj_id2, (inj_id, inj_sigmasq, match, best_tmplt_id) in enumerate(infile["/match_map"]):
	            assert inj_id == inj_id2  # we assume that the ids are in order
	            if match > out_map[inj_id]["match"]:
	                out_map[inj_id] = (inj_id, inj_sigmasq, match, best_tmplt_id + len(out_sngls))
	
	        # and copy in the templates represented here
	        out_sngls.extend(infile["/sngl_inspiral"])
	
	# write to file
	outfile.create_dataset("/sngl_inspiral", data=np.array(out_sngls, dtype=out_sngls_dtype), compression='gzip', compression_opts=1)
	outfile.create_dataset("/match_map", data=out_map, compression='gzip', compression_opts=1)

outfile.close()

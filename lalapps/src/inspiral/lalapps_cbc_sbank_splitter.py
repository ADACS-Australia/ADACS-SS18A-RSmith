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

from time import strftime
from operator import attrgetter
from optparse import OptionParser
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw import ligolw
from glue.ligolw.utils import process as ligolw_process
from pylal.datatypes import LIGOTimeGPS
import numpy

class ContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(ContentHandler)

def parse_command_line():
    parser = OptionParser()
    parser.add_option("-o", "--output-path", metavar = "path", default = ".", help = "Set the path to the directory where output files will be written.  Default is \".\".")
    parser.add_option("-t", "--user-tag", metavar = "tag", default = ".", help = "Make me feel special with a custom tag.")
    parser.add_option("-n", "--nbanks", metavar = "count", type = "int", help = "Set the number of subbanks to split the input bank. All output banks will have the same number of templates (within 1, with the excess templates are spread evenly across the banks).")
    parser.add_option("-s", "--sort-by", default="mchirp", metavar = "{mchirp|ffinal|chirptime|chi}", help = "Select the template sort order.")
    parser.add_option("-i", "--instrument", metavar = "ifo", type="string", help = "override the instrument")
    parser.add_option("-v", "--verbose", action = "store_true", help = "Be verbose.")
    options, filenames = parser.parse_args()

    required_options = ("nbanks", "sort_by")
    missing_options = [option for option in required_options if getattr(options, option) is None]
    if missing_options:
        raise ValueError, "missing required option(s) %s" % ", ".join("--%s" % option.replace("_", "-") for option in missing_options)

    if options.sort_by not in ("mchirp", "chi"):
        raise ValueError, "unrecognized --sort-by \"%s\"" % options.sort_by

    if not filenames:
        raise ValueError, "must provide list of filenames"

    return options, filenames

options, filenames = parse_command_line()

opts_dict = dict((k, v) for k, v in options.__dict__.iteritems() if v is not False and v is not None)

for fname in filenames:

    xmldoc=utils.load_filename(fname, gz=fname.endswith(".gz"), verbose = options.verbose, contenthandler=ContentHandler)
    sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
    process_params_table = lsctables.table.get_table(xmldoc, lsctables.ProcessParamsTable.tableName)

    # Prepare process table with information about the current program
    process = ligolw_process.register_to_xmldoc(xmldoc,
    "lalapps_cbc_sbank_splitter", opts_dict,
    version="no version", cvs_repository="sbank",
    cvs_entry_time=strftime('%Y/%m/%d %H:%M:%S'))

    # sort template rows
    sngl_inspiral_table.sort(key=attrgetter(options.sort_by))
    sngl_inspiral_table_split = lsctables.table.new_from_template(sngl_inspiral_table)
    sngl_inspiral_table.parentNode.replaceChild(sngl_inspiral_table_split, sngl_inspiral_table)

    if len(sngl_inspiral_table) < options.nbanks:
        raise ValueError, "Not enough templates to create the requested number of subbanks."

    # override/read instrument column
    if options.instrument:
        for row in sngl_inspiral_table:
            row.ifo = options.instrument
    else:
	for row in process_params_table:
	    if row.param=='--ifos':
	        options.instrument = row.value

    # split into disjoint sub-banks
    if min([row.f_final for row in sngl_inspiral_table]) > 0:
        # check that this column is actually populated...
        weights = [row.tau0*row.f_final for row in sngl_inspiral_table]
    else:
        weights = [1 for row in sngl_inspiral_table]
    weights_cum = numpy.array(weights).cumsum()

    first_row = 0
    for bank in range(options.nbanks):

        last_row = numpy.searchsorted(weights_cum, (bank+1)*weights_cum[-1]/options.nbanks)
        sngl_inspiral_table_split[:] = sngl_inspiral_table[first_row:last_row]
        first_row = last_row

	ligolw_process.set_process_end_time(process)
	utils.write_filename(xmldoc, "%s-SBANK_SPLIT_%04d-%s.xml"%(options.instrument,bank+1,options.user_tag), gz = False, verbose = options.verbose)

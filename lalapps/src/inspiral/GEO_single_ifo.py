#!/usr/bin/env python2.2
"""
inspiral_pipe_s3_geo.in - S2 inspiral analysis pipeline script

$Id$

This script generates the necessary condor DAG files to analyze the
GEO S3 data with lalapps.

The example geo.ini file can be also found in this directory.
This python script will create only tmplate bank and inspiral jobs!
The cache file is hardcoded "GEOS3_hoft_all.cache", this cache file 
contains path to frame files on explorer cluster at Cardiff.
"""

__author__ = 'Stas Babak <babak@astro.cf.ac.uk>'
__date__ = '$Date$'
__version__ = '$Revision$'[11:-2]

##############################################################################
# import standard modules and append the lalapps prefix to the python path
import sys, os, copy
import getopt, re, string
import tempfile
import ConfigParser
#sys.path.append('/geopptools/stow_pkgs/lalapps/lib/python')
sys.path.append('/home/stas/S3GL/lib/python')

# import the modules we need to build the pipeline
import pipeline, inspiral

##############################################################################
# some functions to make life easier later
class AnalyzedIFOData:
  """
  Contains the information for the data that needs to be filtered.
  """
  def __init__(self,chunk,node):
    self.__analysis_chunk = chunk
    self.__dag_node = node

  def set_chunk(self,chunk):
    self.__analysis_chunk = chunk

  def get_chunk(self):
    return self.__analysis_chunk

  def set_dag_node(self,node):
    self.__dag_node = node

  def get_dag_node(self):
    return self.__dag_node

def chunk_in_segment(chunk,seg):
  if ( 
    chunk.start() >= seg.start()
    and chunk.start() <= seg.end()
    ) or (
    chunk.end() >= seg.start() 
    and chunk.end() <= seg.end()
    ) or (
    seg.start()  >= chunk.start() 
    and seg.end() <= chunk.end() ):
    return 1
  else:
    return 0

def chunks_overlap(chunk1,chunk2):
  if ( 
    chunk1.start() >= chunk2.start() 
    and chunk1.start() <= chunk2.end()
    ) or (
    chunk1.end() >= chunk2.start() 
    and chunk1.end() <= chunk2.end()
    ) or (
    chunk1.start() >= chunk2.start() 
    and chunk1.end() <= chunk2.end() ):
    return 1
  else:
    return 0

def analyze_ifo(ifo_data,ifo_name,tmplt_job,insp_job,snr,chisq,pad,dag,usertag=None):
  """
  Analyze single ifo
  ifo_data  =  the master science segs for the IFO
  ifo_name  =  the name of IFO
  tmplt_job =  the template bank job we should use
  insp_job  =  the condor job for filtering the data
  snr       =  SNR threshold for this IFO
  chisq     =  chi^2 threshold for this IFO
  pad       =  data start/end padding
  dag       =  the DAG to attach the nodes to
  """
  chunks_analyzed = []
  # loop over the master science segments
  for seg in ifo_data:
    # loop over the master analysis chunks in the science segment
    for chunk in seg:
     # done_this_chunk = 0

      # make a template bank job for the master chaunk
      bank = inspiral.TmpltBankNode(tmplt_job)
      bank.set_start(chunk.start())
      bank.set_end(chunk.end())
      bank.set_ifo(ifo_name)
      bank.set_cache("GEOS3_hoft_all.cache")
      dag.add_node(bank)
      # make an inspiral job for the master chunk
      insp = inspiral.InspiralNode(insp_job)
      insp.set_start(chunk.start())
      insp.set_end(chunk.end())
      insp.add_var_opt('snr-threshold',snr)
      insp.add_var_opt('chisq-threshold',chisq)
      insp.add_var_opt('trig-start-time',chunk.trig_start())
      insp.add_var_opt('trig-end-time',chunk.trig_end())
      insp.set_ifo(ifo_name)
      insp.set_cache("GEOS3_hoft_all.cache")
      insp.set_bank(bank.get_output())
      insp.add_parent(bank)
      dag.add_node(insp)
      # store this chunk in the list of filtered L1 data
      chunks_analyzed.append(AnalyzedIFOData(chunk,insp))

  return chunks_analyzed

##  !!! in the return tuple above, prev_df is not defined here since we do not 
##      use LIGODataFind !!!  


##############################################################################
# help message
def usage():
  msg = """\
Usage: lalapps_inspiral_pipe [options]

  -h, --help               display this message
  -v, --version            print version information and exit
  -u, --user-tag TAG       tag the job with TAG (overrides value in ini file)

  -p, --playground         analyze only playground segments
  -l, --log-path PATH      directory to write condor log file 
  -j, --injections FILE    add simulated inspirals from sim_inspiral in FILE
  -f, --config-file FILE   use configuration file FILE
  

  -P, --priority PRIO      run jobs with condor priority PRIO

"""
  print >> sys.stderr, msg

##############################################################################
# pasrse the command line options to figure out what we should do
shortop = "hvu:pl:j:f:P:"
longop = [
  "help",
  "version",
  "user-tag=",
  "playground",
  "log-path=",
  "injections=",
  "config-file=",
  "priority="
  ]

try:
  opts, args = getopt.getopt(sys.argv[1:], shortop, longop)
except getopt.GetoptError:
  usage()
  sys.exit(1)

usertag = None
inj_file = None
condor_prio = None
log_path = None
config_file = None
playground_only = 0

for o, a in opts:
  if o in ("-h", "--help"):
    usage()
    sys.exit(0)
  elif o in ("-v", "--version"):
    print "Single IFO GEO Inspiral pipeline DAG generation script"
    print "Stas Babak <babak@astro.cf.ac.uk>"
    print "CVS Version:",\
      "$Id$"
    print "CVS Tag: $Name$"
    sys.exit(0)
  elif o in ("-u", "--user-tag"):
    usertag = a
  elif o in ("-p", "--playground"):
    playground_only = 1
  elif o in ("-j", "--injections"):
    inj_file = a
  elif o in ("-P", "--priority"):
    condor_prio = a
  elif o in ("-l", "--log-path"):
    log_path = a
  elif o in ("-f", "--config-file"):
    config_file = a
  else:
    print >> sys.stderr, "Unknown option:", o
    usage()
    sys.exit(1)

if not log_path:
  print >> sys.stderr, "No log file path specified."
  print >> sys.stderr, "Use --log-path PATH to specify a location."
  sys.exit(1)

if not config_file:
  print >> sys.stderr, "No configuration file specified."
  print >> sys.stderr, "Use --config-file FILE to specify location."
  sys.exit(1)


##############################################################################
# try and make a directory to store the cache files and job logs
try: os.mkdir('logs')
except: pass

##############################################################################
# create the config parser object and read in the ini file
cp = ConfigParser.ConfigParser()
cp.read(config_file)

##############################################################################
# if a usertag has been specified, override the config file
if usertag:
  cp.set('pipeline','user-tag',usertag)
else:
  try:
    usertag = string.strip(cp.get('pipeline','user-tag'))
  except:
    usertag = None

##############################################################################
# create a log file that the Condor jobs will write to
basename = re.sub(r'\.ini',r'',config_file)
tempfile.tempdir = log_path
if usertag:
  tempfile.template = basename + '.' + usertag + '.dag.log.'
else:
  tempfile.template = basename + '.dag.log.'
logfile = tempfile.mktemp()
fh = open( logfile, "w" )
fh.close()

##############################################################################
# create the DAG writing the log to the specified directory
dag = pipeline.CondorDAG(logfile)
if usertag:
  dag.set_dag_file(basename + '.' + usertag + '.dag')
else:
  dag.set_dag_file(basename + '.dag')

##############################################################################
# create the Condor jobs that will be used in the DAG
#df_job = inspiral.DataFindJob(cp)
tmplt_job = inspiral.TmpltBankJob(cp)
insp_job = inspiral.InspiralJob(cp)


# set better submit file names than the default
if usertag:
  subsuffix = '.' + usertag + '.sub'
else:
  subsuffix = '.sub'
tmplt_job.set_sub_file( basename + '.tmpltbank' + subsuffix )
insp_job.set_sub_file( basename + '.inspiral' + subsuffix )

if usertag:
  tmplt_job.add_opt('user-tag',usertag)
  insp_job.add_opt('user-tag',usertag)

# add the injections
if inj_file:
  insp_job.add_opt('injection-file',inj_file)
  trig_insp_job.add_opt('injection-file',inj_file)

# set the condor job priority
if condor_prio:
  tmplt_job.add_condor_cmd('priority',condor_prio)
  insp_job.add_condor_cmd('priority',condor_prio)


##############################################################################
# get the thresholds, pad and chunk lengths from the values in the ini file
g1_snr = cp.get('pipeline','g1-snr-threshold')

g1_chisq = cp.get('pipeline','g1-chisq-threshold')

pad = int(cp.get('data', 'pad-data'))
n = int(cp.get('data', 'segment-length'))
s = int(cp.get('data', 'number-of-segments'))
r = int(cp.get('data', 'sample-rate'))
o = int(cp.get('inspiral', 'segment-overlap'))
length = ( n * s - ( s - 1 ) * o ) / r
overlap = o / r

##############################################################################
# Step 1: read science segs that are greater or equal to the length of data
# in a chunk (defined by the variable "length" above)
print "reading in single ifo science segments...",
sys.stdout.flush()

g1_data = pipeline.ScienceData()

g1_data.read(cp.get('input','g1-segments'),length)

print "done"


##############################################################################
# Step 2: Create analysis chunks from the single IFO science segments.  The
# instance g1_data will contain all data that we can analyze 

print "making master chunks...",
sys.stdout.flush()

g1_data.make_chunks(length,overlap,playground_only)

g1_data.make_chunks_from_unused(length,overlap/2,playground_only,0)

print "done"

##############################################################################
# Step 3: find all data that is analyzable.
print "determining analyzable data",
sys.stdout.flush()

g1_data_out = copy.deepcopy(g1_data)
for seg in g1_data_out:
    seg.set_start(seg.start()+overlap/2)
    seg.set_end(seg.end()-overlap/2)

not_g1_data_out = copy.deepcopy(g1_data_out)
not_g1_data_out.invert()

print "done"

"""
##############################################################################
# Step 4: find all single ifo GEO data that we can analyze
print "computing L1 single ifo data...",
sys.stdout.flush()

g1_single_data = copy.deepcopy(l1_data_out)
l1_single_data.intersection(not_h1_data_out)
l1_single_data.intersection(not_h2_data_out)
if tama_overlap:
  l1_single_data.intersection(t_data)
l1_single_data.coalesce()
print "done"
"""

##############################################################################
# Step 5: Determine which of the GEO master chunks needs to be filtered
g1_chunks_analyzed = []

print "setting up jobs to filter GEO data...",
sys.stdout.flush()

g1_chunks_analyzed = analyze_ifo(g1_data,'G1',tmplt_job,insp_job,g1_snr,g1_chisq,pad,dag,usertag=None)

print "done"


##############################################################################
# Write out the DAG, help message and log file
dag.write_sub_files()
dag.write_dag()

print "\nCreated a DAG file which can be submitted by executing"
print "\n   condor_submit_dag", dag.get_dag_file()
print "\nfrom a condor submit machine (e.g. explorer.astro.cf.ac.uk)\n"


# write out a log file for this script
if usertag:
  log_fh = open(basename + '.pipeline.' + usertag + '.log', 'w')
else:
  log_fh = open(basename + '.pipeline.log', 'w')
  
log_fh.write( "$Id$" + "\n" )
log_fh.write( "$Name$" + "\n\n" )
log_fh.write( "Invoked with arguments:\n" )
for o, a in opts:
  log_fh.write( o + ' ' + a + '\n' )

log_fh.write( "Config file has CVS strings:\n" )
log_fh.write( cp.get('pipeline','version') + "\n" )
log_fh.write( cp.get('pipeline','cvs-tag') + "\n\n" )


print >> log_fh, "\n===========================================\n"
print >> log_fh, "Science Segments and master chunks:\n"
print >> log_fh, g1_data
for seg in g1_data:
  print >> log_fh, " ", seg
  for chunk in seg:
    print >> log_fh, "   ", chunk


print >> log_fh, "\n===========================================\n"
log_fh.write( 
  "Filtering " + str(len(g1_chunks_analyzed)) + " G1 master chunks\n" )
total_time = 0
for g1_done in g1_chunks_analyzed:
  print >> log_fh, g1_done.get_chunk()
  total_time += len(g1_done.get_chunk())
print >> log_fh, "\n total time", total_time, "seconds"

if playground_only:
    g1_data.play()
    f = open('geo_playground.txt', 'w')
    for seg in g1_data:
      f.write('%4d %10d %10d %6d\n' % (seg.id(), seg.start(), seg.end(), seg.dur()))
    f.close()
#sys.exit(0)






















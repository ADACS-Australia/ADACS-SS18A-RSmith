"""
inspiral_sip.in - standalone Single-stage Inspiral Pipeline driver script

This script generates the condor DAG necessary to analyze LIGO, GEO, Virgo
data through the inspiral pipeline.  The DAG does datafind, tmpltbank,
inspiral, and thinca steps of the pipeline.  It analyzes the
single, double, triple and quadro ifo times accordingly.  It can also be
run with injections.
"""

__author__ = 'Stephen Fairhurst <sfairhur@gravity.phys.uwm.edu>, Drew Keppel <drew.keppel@ligo.org>'
__date__ = '$Date$'
__version__ = '$Revision$'

##############################################################################
# import standard modules
import sys, os, copy, math, shutil
import socket, time
import re, string
from optparse import *
import tempfile
import urlparse

##############################################################################
# import the modules we need to build the pipeline
from glue import pipeline
from glue import segments as glue_segments
from glue import lal
from lalburst import timeslides as ligolw_tisi
from pylal import ligolw_cafe
from lalapps import inspiral
from lalapps import power
from lalapps import inspiralutils


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


##############################################################################
# function to generate the coincident segments from four input segment lists
def generate_segments(ifo1_data, ifo2_data, ifo3_data, ifo4_data):
  """
    compute the segments arising as the overlap of the four sets of single
    ifo segment lists.
    ifo1_data = data segments for ifo1
    ifo2_data = data segments for ifo2
    ifo3_data = data segments for ifo3
    ifo4_data = data segments for ifo4
  """ 

  segment_list = pipeline.ScienceData()
  segment_list = copy.deepcopy(ifo1_data)
  segment_list.intersect_4(ifo2_data, ifo3_data, ifo4_data)
   
  return segment_list

    
##############################################################################
# function to set up datafind, template bank and inspiral jobs for an ifo  
def analyze_ifo(ifo_name,ifo_data,ifo_to_do,tmplt_job,insp_job,df_job,\
  prev_df,dag, usertag=None, inspinjNode = None, insp_ckpt_job = None):
  """
  Analyze the data from a single IFO.  Since the way we treat all this data is
  the same, this function is the same for all interferometers. Returns the last
  LSCdataFind job that was executed and the chunks analyzed.
  
  ifo_name = the name of the IFO
  ifo_data = the master science segs 
  ifo_to_do = the science segments we need to analyze
  tmplt_job = if not FixedBank: template bank job we should use
  insp_job = the condor job that we should use to analyze data
  df_job = the condor job to find the data
  prev_df = the previous LSCdataFind job that was executed
  dag = the DAG to attach the nodes to
  usertag = the usertag to add to the job names
  inspinjNode = the inspinj node to be added as a parent to inspirals
  insp_ckpt_job = a checkpoint restore job for the inspiral code
  """

  # add the non veto inspiral options
  if cp.has_section('no-veto-inspiral'): 
    insp_job.add_ini_opts(cp,'no-veto-inspiral')
  
  # add the ifo specific options
  if cp.has_section(ifo_name.lower() + '-inspiral'): 
    insp_job.add_ini_opts(cp,ifo_name.lower() + '-inspiral')

  if cp.has_section(ifo_name.lower() + '-tmpltbank'):
    tmplt_job.add_ini_opts(cp,ifo_name.lower() + '-tmpltbank')

  # we may use a fixed bank specified in ini file
  try:
    FixedBank = cp.get('input','fixed-bank')
    print "For %s we use bank %s"%(ifo_name, FixedBank)
  except:
    FixedBank = None

  # get datatype info from config file
  data_opts, type, channel = inspiralutils.get_data_options(cp,ifo_name)
  
  if cp.has_section('tmpltbank-1'):
    tmplt_job.add_ini_opts(cp, 'tmpltbank-1')
  if cp.has_section(data_opts):
    tmplt_job.add_ini_opts(cp,data_opts)
    insp_job.add_ini_opts(cp,data_opts)

  tmplt_job.set_channel(channel)
  insp_job.set_channel(channel)

  # see if we are using calibrated data
  if cp.has_section(data_opts) and cp.has_option(data_opts,'calibrated-data'):
    calibrated = True
    print "we use calibrated data for", ifo_name
  else: calibrated = False

  # prepare the injection filename
  if ifo_data:
    injStart = ifo_data[0].start()
    injDuration = ifo_data[-1].end()-injStart
    injectionFileTemplate = "HL-INJECTION_%%s-%d-%d.xml" % \
      (injStart, injDuration)

  chunks_analyzed = []
  # loop over the master science segments
  for seg in ifo_data:

    # loop over the master analysis chunks in the science segment
    for chunk in seg:
      done_this_chunk = False

      # now loop over all the data that we need to filter
      for seg_to_do in ifo_to_do:

        # if the current chunk is in one of the segments we need to filter
        if not done_this_chunk and inspiral.overlap_test(chunk,seg_to_do):

          # make sure we only filter the master chunk once
          done_this_chunk = True

          # make sure we have done one and only one datafind for the segment
          if not opts.read_cache:
            if not seg.get_df_node():
              df = pipeline.LSCDataFindNode(df_job)
              if not opts.disable_dag_categories:
                df.set_category('datafind')
              if not opts.disable_dag_priorities:
                df.set_priority(100)
              df.set_observatory(ifo_name[0])
              # add a padding time to the start of the datafind call (but don't change datafind output name)
              if ifo_name == 'G1':
                dfsect = 'geo-data'
              elif ifo_name == 'V1':
                dfsect = 'virgo-data'
              else:
                dfsect = 'ligo-data'
              if cp.has_option(dfsect,ifo_name.lower() + '-datafind-start-padding'):
                padding=cp.get(dfsect,ifo_name.lower()+'-datafind-start-padding')
              else:
                padding=0.
              df.set_start(seg.start(),padding)
              df.set_end(seg.end())
              seg.set_df_node(df)
              if type: df.set_type(type)
              if prev_df and opts.disable_dag_categories:
                df.add_parent(prev_df)
              if opts.datafind: dag.add_node(df)
              prev_df = df
          else:
            prev_df = None  

          # make a template bank job for the master chunk
          bank = inspiral.TmpltBankNode(tmplt_job)
          if not opts.disable_dag_categories:
            bank.set_category('tmpltbank')
          if not opts.disable_dag_priorities:
            bank.set_priority(1)
          bank.set_start(chunk.start())
          bank.set_end(chunk.end())
          bank.set_ifo(ifo_name)
          bank.set_vds_group(ifo_name[0] + str(chunk.start()))
          if not opts.read_cache: bank.set_cache(df.get_output())
          else: bank.set_cache(cp.get('datafind',ifo_name+"-cache"))
          if not calibrated: bank.calibration()
          if opts.datafind: bank.add_parent(df)
          if (opts.template_bank and not FixedBank): dag.add_node(bank)
                  
          # make an inspiral job for the master chunk
          insp = inspiral.InspiralNode(insp_job)
          if not opts.disable_dag_categories:
            insp.set_category('inspiral1')
          if not opts.disable_dag_priorities:
            insp.set_priority(2)
          if usertag:
            insp.set_user_tag(usertag.split('_CAT')[0])
          insp.set_start(chunk.start())
          insp.set_end(chunk.end())
          insp.set_trig_start(chunk.trig_start())
          insp.set_trig_end(chunk.trig_end())
          insp.set_ifo(ifo_name)
          insp.set_ifo_tag("FIRST")
          insp.set_vds_group(ifo_name[0] + str(chunk.start()))
          if not opts.read_cache: insp.set_cache(df.get_output())
          else:  insp.set_cache(cp.get('datafind',ifo_name+"-cache"))
          if not calibrated: insp.calibration()
          if FixedBank:
            insp.set_bank(FixedBank)
          else:
            insp.set_bank(bank.get_output())
          
          if opts.datafind: insp.add_parent(df)
          if inspinjNode and opts.inspinj: insp.add_parent(inspinjNode) 
          if (opts.template_bank and not FixedBank): insp.add_parent(bank)
          if opts.inspiral: dag.add_node(insp)

          if opts.data_checkpoint:
            # make an inspiral checkpoint restore job
            insp_job.set_universe("vanilla")
            insp.set_data_checkpoint()
            insp.set_post_script(cp.get('condor','checkpoint-post-script'))
            insp.add_post_script_arg(os.path.join(os.getcwd(),insp.get_checkpoint_image()))
            insp_ckpt = inspiral.InspiralCkptNode(insp_ckpt_job)
            insp_ckpt.set_output(insp.get_output())
            insp_ckpt.set_injections(insp.get_injections())
            insp_ckpt.set_checkpoint_image(insp.get_checkpoint_image())

            if cp.has_option('pipeline','condor-c-site'):
              # additional requirements to launch jon on remote pool
              insp_ckpt_job.set_universe("grid")
              insp_ckpt.set_grid_start("pegasuslite")
              insp_ckpt.add_pegasus_profile("condor","grid_resource","condor %s" % cp.get('pipeline','condor-c-site'))
              insp_ckpt.add_pegasus_profile("condor","+remote_jobuniverse","5")
              insp_ckpt.add_pegasus_profile("condor","+remote_requirements","True")
              insp_ckpt.add_pegasus_profile("condor","+remote_ShouldTransferFiles","True")
              insp_ckpt.add_pegasus_profile("condor","+remote_WhenToTransferOutput","ON_EXIT")
              insp_ckpt.add_pegasus_profile("condor","+remote_TransferInputFiles",'"' + insp.get_checkpoint_image() + '"')
              insp_ckpt.add_pegasus_profile("condor","+remote_PeriodicRelease",'( JobStatus == 5 && HoldReasonCode == 13 && NumSystemHolds < 3 )')
            else:
              insp_ckpt_job.set_universe("vanilla")

            insp_ckpt.add_parent(insp)
            if opts.inspiral: dag.add_node(insp_ckpt)

            # ensure output is added to list of output files
            output = insp_ckpt.get_output()

            # store this chunk in the list of filtered data
            chunks_analyzed.append(AnalyzedIFOData(chunk,insp_ckpt))

          else:
            # XXX: ensure output is added to list of output files
            output = insp.get_output()

            # store this chunk in the list of filtered data
            chunks_analyzed.append(AnalyzedIFOData(chunk,insp))         

  return tuple([prev_df,chunks_analyzed])


##############################################################################
# function to do thinca on coincident IFO data  
def thinca_coinc(ifo_list, single_data_analyzed, cafe_caches, cafe_base,
    lladd_job, tisi_file_name, lladd_veto_file, 
    coinc_job, dag, do_coinc, do_insp, usertag=None, inspinjNode=None):
  """
  Run thinca on the coincident times from each of the sets of IFOs. 
  Since the way we treat all this data is the same, this function is the same 
  for all. 

  ifo_list = a list of the ifos we are to analyze
  single_data_analyzed = dictionary of single ifo data analyzed
  cafe_caches = the caches from ligolw_cafe.ligolw_cafe()
  cafe_base = the base name for the cafe caches
  lladd_job = the condor job to do ligolw_add
  tisi_file_name = the name of the tisi file to add
  lladd_veto_file = the name of the veto file to add or None
  coinc_job = the condor job to do thinca
  dag = the DAG to attach the nodes to
  do_coinc = whether we should add the thinca jobs to the dag
  do_insp  = whether previous inspiral jobs are in the dag
  usertag = the usertag to add to the output file name
  inspinjNode = the inspinj node to be added as a parent to ligolw_add jobs
  """

  # create caches using ligolw_cafe
  cache_names = ligolw_cafe.write_caches(cafe_base, cafe_caches, set(ifo_list))
  coinc_analyzed = []

  # loop over caches
  for idx in range(len(cafe_caches)):
    if len(cafe_caches[idx].objects):
      cache = cafe_caches[idx]
      cachename = cache_names[idx]
      thincabase = cafe_base.split('.')[0].replace('CAFE_','')
      ifos = set(cache_entry.observatory for cache_entry in cache.objects)

      # extract segment information
      seg = power.cache_span(cache.objects)
      seg = pipeline.AnalysisChunk(seg[0],seg[1])

      # create node for ligolw_add to create xml file
      lladd = pipeline.LigolwAddNode(lladd_job)
      
      # add the tisi and veto files
      lladd.add_file_arg(tisi_file_name)
      if lladd_veto_file:
        lladd.add_file_arg(lladd_veto_file)

      # add the input xml files from the cafe cache
      cachefp = open(cachename,'r')
      cacheobject = lal.Cache().fromfile(cachefp)
      cachefp.close()
      cachepfns = cacheobject.pfnlist()
      for pfn in cachepfns:
        lladd.add_file_arg(pfn)

      # create node for ligolw_thinca to analyze xml file
      thinca = inspiral.ThincaNode(coinc_job)
      thinca.set_start(seg.start(), pass_to_command_line=False)
      thinca.set_end(seg.end(), pass_to_command_line=False)
      thinca.set_zip_output(True)
      if usertag: thinca.set_user_tag(thincabase, pass_to_command_line=False)

      # check if caches are adjacent
      coinc_end_time_segment = ''
      if idx and (cache.extent[0] == cafe_caches[idx-1].extent[1]):
        coinc_end_time_segment += str(cache.extent[0])
      coinc_end_time_segment += ':'
      if idx + 1 - len(cafe_caches) and (cache.extent[1] == cafe_caches[idx+1].extent[0]):
        coinc_end_time_segment += str(cache.extent[1])
      thinca.add_var_opt('coinc-end-time-segment',coinc_end_time_segment)

      # scroll through ifos, adding the appropriate ones
      for ifo in ifo_list:
        if ifo in ifos:
          thinca.set_ifo(ifo, pass_to_command_line=False)
                
      # add all inspiral jobs in this cache to input
      if do_insp:
        for cache_entry in cache.objects:
          lladd.add_parent(single_data_analyzed[cache_entry])

      # add inspinj job as parent of each ligolw_add job
      if inspinjNode and opts.inspinj: lladd.add_parent(inspinjNode)

      # set output of ligolw_add jobs to follow thinca's convention
      lladd_outfile = re.sub('THINCA','LLWADD',thinca.get_output())
      lladd.set_output(lladd_outfile)
      thinca.set_input(lladd.get_output(), pass_to_command_line=False)
      thinca.add_file_arg(lladd.get_output())

      # check for condor settings
      if not opts.disable_dag_categories:
        lladd.set_category('ligolw_add')
        thinca.set_category('thinca')
      if not opts.disable_dag_priorities:
        lladd.set_priority(3)
        thinca.set_priority(3)

      # add ligolw_add and ligolw_thinca nodes to dag
      if do_coinc:
        dag.add_node(lladd)
        thinca.add_parent(lladd)
        dag.add_node(thinca)

      # add ligolw_thinca coincident segment
      coinc_analyzed.append(AnalyzedIFOData(seg,thinca))

  return coinc_analyzed



##############################################################################
# function to sweep up data using sire
def sire_segments(segments, sire_job, dag, do_sire, do_input, ifo, 
  sire_start, sire_end, inj_file=None, ifotag=None, usertag=None, 
  inspinjNode=None):
  """
  Do a sire to sweep up all the triggers of a specific kind
  
  segments  = list of segments to use
  sire_job  = sire job to use for the analysis
  dag       = name of the dag
  do_sire   = whether the sired jobs are to be run by the dag
  do_input  = whether the files to be sired were produced by this dag
  ifo       = which ifo triggers to keep
  sire_start= start time for sire job
  sire_end  = end time for sire job
  inj_file  = name of injection file
  ifotag    = ifotag used in naming the file
  usertag   = the usertag to add to the output file name
  inspinjNode = add an inspinj node as a parent (default false)
  """
  # only write the input file if we're doing the job
  sire = inspiral.SireNode(sire_job)
  if not opts.disable_dag_categories:
    sire.set_category('sire')
  if not opts.disable_dag_priorities:
    sire.set_priority(100)
  if cp.has_option('pipeline', 'collapse-sire'):
    sire.set_dax_collapse(cp.get('pipeline','collapse-sire'))

  # set the options:
  sire.set_ifo(ifo)
  if inj_file: 
    sire.set_inj_file(inj_file)
  if ifotag: sire.set_ifo_tag(ifotag)
  if usertag: sire.set_user_tag(usertag)

  # set the segment
  sire.set_start(sire_start)
  sire.set_end(sire_end)

  if inspinjNode and opts.inspinj: sire.add_parent(inspinjNode)

  for seg in segments:
    # set the glob file for sire
    if do_input: sire.add_parent(seg.get_dag_node())

    if seg.get_dag_node().get_output():
      output = seg.get_dag_node().get_output()
    else: 
      output = seg.get_dag_node().get_output_a()

    sire.add_file_arg(output)
    
  if do_sire: dag.add_node(sire) 

  return sire


##############################################################################
# function to sweep up data using sire
def sire_segments_individually(segments, sire_job, dag, do_sire, do_input, ifo,
  inj_file=None, ifotag=None, usertag=None, inspinjNode=None):
  """
  Do a sire to sweep up all the triggers of a specific kind
  
  segments  = list of thinca segments to use
  sire_job  = sire job to use for the analysis
  dag       = name of the dag
  do_sire   = whether the sired jobs are to be run by the dag
  do_input  = whether the files to be sired were produced by this dag
  ifo       = which ifo triggers to keep
  inj_file  = name of injection file
  ifotag    = ifotag used in naming the file
  usertag   = the usertag to add to the output file name
  """
  sire_analyzed = []
  for seg in segments:
    sire_node = sire_segments([seg], sire_job, dag, do_sire, do_input, ifo, \
        seg.get_chunk().start(), seg.get_chunk().end(), inj_file, ifotag, 
        usertag, inspinjNode=inspinjNode)

    sire_analyzed.append(AnalyzedIFOData(seg.get_chunk(),sire_node))

  return sire_analyzed


##############################################################################
#
#  MAIN PROGRAM
#
##############################################################################
usage = """usage: %prog [options] 
"""

parser = OptionParser( usage )

parser.add_option("-v", "--version",action="store_true",default=False,\
    help="print version information and exit")
    
parser.add_option("-u", "--user-tag",action="store",type="string",\
    default=None,metavar=" USERTAG",\
    help="tag the jobs with USERTAG (overrides value in ini file)")

parser.add_option("-g", "--g1-data",action="store_true",default=False,\
    help="analyze g1 data")
parser.add_option("-a", "--h1-data",action="store_true",default=False,\
    help="analyze h1 data")
parser.add_option("-b", "--h2-data",action="store_true",default=False,\
    help="analyze h2 data")
parser.add_option("-l", "--l1-data",action="store_true",default=False,\
    help="analyze l1 data")
parser.add_option("-n", "--v1-data",action="store_true",default=False,\
    help="analyze v1 data")

parser.add_option("-S", "--one-ifo",action="store_true",default=False,\
    help="analyze single ifo data (not usable for GEO)")
parser.add_option("-D", "--two-ifo",action="store_true",default=False,\
    help="analyze two interferometer data")
parser.add_option("-T", "--three-ifo",action="store_true",default=False,\
    help="analyze three interferometer data")
parser.add_option("-Q", "--four-ifo",action="store_true",default=False,\
    help="analyze four intereferometer data")
  
parser.add_option("-A", "--analyze-all",action="store_true",default=False,\
    help="analyze all ifos and all data (over-rides above)" \
    "(option not available since there are 5 instruments and the code " \
    "only supports quadruple coincidence)")


parser.add_option("-d", "--datafind",action="store_true",default=False,\
    help="run LSCdataFind to create frame cache files")
parser.add_option("-I", "--inspinj",action="store_true",default=False,\
    help="run lalapps_inspinj to generate injection files")
parser.add_option("-t", "--template-bank",action="store_true",default=False,\
    help="run lalapps_tmpltbank to generate template banks")
parser.add_option("-i", "--inspiral" ,action="store_true",default=False,\
    help="run lalapps_inspiral to generate triggers")
parser.add_option("-s", "--sire-inspiral",action="store_true",default=False,\
    help="do sires to sweep up triggers")
parser.add_option("-c", "--coincidence",action="store_true",default=False,\
    help="run lalapps_thinca to test for coincidence")
parser.add_option("-Y", "--summary-inspiral-triggers",action="store_true", \
    default=False, \
    help="Produce summary triggers from run.  This will produce "
    "sire (first inspiral) files containing triggers from the whole run. "
    "Use with care, as files can become very large.")

parser.add_option("-R", "--read-cache",action="store_true",default=False,\
    help="read cache file from ini-file (if LSCDataFind is broken)")

parser.add_option("-P", "--priority",action="store",type="int",\
    metavar=" PRIO",help="run jobs with condor priority PRIO")

parser.add_option("", "--disable-dag-categories",action="store_true",
    default=False,help="disable the internal dag category maxjobs")
parser.add_option("", "--disable-dag-priorities",action="store_true",
    default=False,help="disable the depth first priorities")

parser.add_option("", "--noop-inspinj", action="store_true", default=False,
    help="create a DAG with fake (no-op) inspinj jobs")
 
parser.add_option("-f", "--config-file",action="store",type="string",\
    metavar=" FILE",help="use configuration file FILE")

parser.add_option("-p", "--log-path",action="store",type="string",\
    metavar=" PATH",help="directory to write condor log file")

parser.add_option("-o", "--output-segs",action="store_true",default=False,\
    help="output the segment lists of analyzed data")

parser.add_option("-x","--dax", action="store_true", default=False,\
    help="create a dax instead of a dag")

parser.add_option("-w", "--write-script", action="store_true", default=False,
      help="write the workflow to a locally executable script")

parser.add_option("--data-checkpoint", action="store_true", default=False,\
    help="checkpoint the inspiral code")

command_line = sys.argv[1:]
(opts,args) = parser.parse_args()


#################################
# if --version flagged
if opts.version:
  print "$Id$"
  sys.exit(0)

#################################
# Sanity check of input arguments
if not opts.config_file:
  print >> sys.stderr, "No configuration file specified."
  print >> sys.stderr, "Use --config-file FILE to specify location."
  sys.exit(1)

if not opts.log_path:
  print >> sys.stderr, "No log file path specified."
  print >> sys.stderr, "Use --log-path PATH to specify a location."
  sys.exit(1)

if not opts.g1_data and not opts.h1_data and not opts.h2_data and \
    not opts.l1_data and not opts.v1_data and not opts.analyze_all:
  print >> sys.stderr, "No ifos specified.  Please specify at least one of"
  print >> sys.stderr, "--g1-data, --h1-data, --h2-data, --l1-data, --v1-data"
  print >> sys.stderr, "or use --analyze-all to analyze all ifos all data"
  sys.exit(1)
elif opts.analyze_all:
  print >> sys.stderr, "The --analyze-all flag is currently not available."
  print >> sys.stderr, "The code supports quadruple coincidence, so you can"
  print >> sys.stderr, "choose at most four instruments to analyze."
  sys.exit(1)

if opts.g1_data and opts.h1_data and opts.h2_data and opts.l1_data \
    and opts.v1_data:
  print >> sys.stderr, "Too many IFOs specified. " \
      "Please choose up to four IFOs, but not five."
  sys.exit(1)

if not opts.one_ifo and not opts.two_ifo and not opts.three_ifo and \
    not opts.four_ifo and not opts.analyze_all:
  print >> sys.stderr, "No number of ifos given. Please specify at least one of"
  print >> sys.stderr, "--one-ifo, --two-ifo, --three-ifo, --four-ifo"
  print >> sys.stderr, "or use --analyze-all to analyze all ifos all data"
  sys.exit(1)
elif opts.analyze_all:
  print >> sys.stderr, "The --analyze-all flag can not be used to specify the"
  print >> sys.stderr, "number of ifos to analyze. The code supports quadruple"
  print >> sys.stderr, "coincidence, so you can choose at most four instruments"
  print >> sys.stderr, "to analyze."
  sys.exit(1)

if not (opts.datafind or opts.template_bank or opts.inspiral \
    or opts.sire_inspiral or opts.coincidence):
  print >> sys.stderr, """  No steps of the pipeline specified.
  Please specify at least one of
  --datafind, --template-bank, --inspiral, --sire-inspiral, --coincidence"""
  sys.exit(1)
   
ifo_list = ['H1','H2','L1','V1','G1']

#################################################################
# If using G1 data, rearrange ifo_list since it only uses
# the first four ifos named in ifo_list for quadruple coincidence

if opts.g1_data:
  if not opts.h1_data and ifo_list[4]=='G1':
    ifo_list[0]='G1'
    ifo_list[4]='H1'
  if not opts.h2_data and ifo_list[4]=='G1':
    ifo_list[1]='G1'
    ifo_list[4]='H2'
  if not opts.l1_data and ifo_list[4]=='G1':
    ifo_list[2]='G1'
    ifo_list[4]='L1'
  if not opts.v1_data and ifo_list[4]=='G1':
    ifo_list[3]='G1'
    ifo_list[4]='V1'

  ifo_list = ifo_list[:4]

ifotag = None
#################################
# store the values
do = {}
do['G1'] = opts.g1_data
do['H1'] = opts.h1_data
do['H2'] = opts.h2_data
do['L1'] = opts.l1_data
do['V1'] = opts.v1_data

ifo_analyze = []
for ifo in ifo_list:
  if do[ifo]: 
    ifo_analyze.append(ifo)
ifo_analyze.sort()

#################################
# analyze everything if --analyze-all set
if opts.analyze_all:
  for ifo in ifo_list:
    do[ifo] = True
  opts.one_ifo   = True
  opts.two_ifo   = True
  opts.three_ifo = True
  opts.four_ifo  = True

##############################################################################
# determine all possible coincident sets of ifos and those to analyze:
analyze = []
ifo_coincs = []

# one ifo
for ifo1 in ifo_list:
  if opts.one_ifo and do[ifo1]:
      analyze.append(ifo1)

# two ifo
for ifo1 in ifo_list:
  for ifo2 in ifo_list:
    if ifo1 < ifo2:
      ifo_coincs.append(ifo1 + ifo2)
      if opts.two_ifo and do[ifo1] and do[ifo2]:
        analyze.append(ifo1 + ifo2)

# three ifo
for ifo1 in ifo_list:
  for ifo2 in ifo_list:
    for ifo3 in ifo_list:
      if ifo1 < ifo2 and ifo2 < ifo3:
        ifo_coincs.append(ifo1 + ifo2 + ifo3)
        if opts.three_ifo and do[ifo1] and do[ifo2] and do[ifo3]:
          analyze.append(ifo1 + ifo2 + ifo3)

# four ifo
for ifo1 in ifo_list:
  for ifo2 in ifo_list:
    for ifo3 in ifo_list:
      for ifo4 in ifo_list:
        if ifo1 < ifo2 and ifo2 < ifo3 and ifo3 < ifo4:
          ifo_coincs.append(ifo1 + ifo2 + ifo3 + ifo4)
          if opts.four_ifo and do[ifo1] and do[ifo2] and do[ifo3] and do[ifo4]:
            analyze.append(ifo1 + ifo2 + ifo3 + ifo4)

ifo_combinations = copy.deepcopy(ifo_list)
ifo_combinations.extend(ifo_coincs)


##############################################################################
# try to make a directory to store the cache files and job logs
try: os.mkdir('cache')
except: pass
try: os.mkdir('logs')
except: pass

##############################################################################
# create the config parser object and read in the ini file
cp = pipeline.DeepCopyableConfigParser()
cp.read(opts.config_file)

##############################################################################
# if a usertag has been specified, override the config file
if opts.user_tag:
  usertag = opts.user_tag
  cp.set('pipeline','user-tag',usertag)
else:
  try:
    usertag = string.strip(cp.get('pipeline','user-tag'))
  except:
    usertag = None
  
##############################################################################
# create a log file that the Condor jobs will write to
basename = re.sub(r'\.ini',r'',opts.config_file)
tempfile.tempdir = opts.log_path
if usertag:
  tempfile.template = basename + '.' + usertag + '.dag.log.'
else:
  tempfile.template = basename + '.dag.log.'
logfile = tempfile.mktemp()
fh = open( logfile, "w" )
fh.close()

##############################################################################
# create the DAG writing the log to the specified directory
dag = pipeline.CondorDAG(logfile, opts.dax)
if usertag:
  dag.set_dag_file(basename + '.' + usertag )
  dag.set_dax_file(basename + '.' + usertag )
else:
  dag.set_dag_file(basename )
  dag.set_dax_file(basename )

# set better submit file names than the default
if usertag:
  subsuffix = '.' + usertag + '.sub'
else:
  subsuffix = '.sub'

##############################################################################
# create the Condor jobs that will be used in the DAG

# datafind:
frame_types = []
try:
  lsync_file = cp.get('pipeline','lsync-cache-file')
  try: frame_types.append(cp.get('input','ligo-type'))
  except: pass
  try: frame_types.append(cp.get('input','virgo-type'))
  except: pass
  try: frame_types.append(cp.get('input','geo-type'))
  except: pass
  frame_types = [t for t in frame_types if t]
except:
  lsync_file = None
df_job = pipeline.LSCDataFindJob(
  'cache','logs',cp,opts.dax,lsync_file,'|'.join(frame_types))
df_job.set_sub_file( basename + '.datafind'+ subsuffix )

# tmpltbank:
tmplt_jobs = {}

for ifo in ifo_list:
  tmplt_jobs[ifo] = inspiral.TmpltBankJob(cp,opts.dax)
  tmplt_jobs[ifo].set_sub_file( basename + '.tmpltbank_' + ifo + subsuffix )

# inspinj:
inspinj_job = inspiral.InspInjJob(cp) 
inspinj_job.set_sub_file( basename + '.inspinj' + subsuffix )

if opts.noop_inspinj:
  inspinj_job.add_condor_cmd("noop_job", "true")

# inspiral:
insp_jobs = {}

for ifo in ifo_list:
  insp_jobs[ifo] = inspiral.InspiralJob(cp,opts.dax)
  insp_jobs[ifo].set_sub_file( basename + '.inspiral_' + ifo + subsuffix )

# create inspiral checkpoint job
insp_ckpt_job = inspiral.InspiralCkptJob(cp,opts.dax)
if cp.has_option('pipeline','remote-site'):
  insp_ckpt_job.set_executable_installed(False)

# ligolw_add:
lladd_job = pipeline.LigolwAddJob('logs', cp, opts.dax)
lladd_job.set_sub_file( basename + '.ligolw_add' + subsuffix )

# thinca:
thinca_job = inspiral.ThincaJob(cp,opts.dax)
thinca_job.set_universe('vanilla')
thinca_job.set_sub_file(basename + '.thinca' + subsuffix )
thinca_job.add_condor_cmd("getenv", "True")

# sire:
sire_job = inspiral.SireJob(cp)
sire_job.set_sub_file( basename + '.sire' + subsuffix )
sire_summary_job = inspiral.SireJob(cp)
sire_summary_job.set_sub_file( basename + '.sire_summary' + subsuffix )

all_jobs = [inspinj_job, sire_job, sire_summary_job]
all_jobs.extend(tmplt_jobs.values())
all_jobs.extend(insp_jobs.values())

##############################################################################
# set the usertag in the jobs
if usertag:
  for job in all_jobs:
    if not job.get_opts().has_key('user-tag'):
      job.add_opt('user-tag',usertag)
all_jobs.append(lladd_job)
all_jobs.append(thinca_job)
all_jobs.append(insp_ckpt_job)
all_jobs.append(df_job)

##############################################################################
# set the condor job priority
if opts.priority:
  for job in all_jobs:
    job.add_condor_cmd('priority',str(opts.priority))


##############################################################################
# read in the GPS start and end times from the ini file
# Only used to set the cache files.
# XXX Should we use this to cut segments??
try:
  gps_start_time = cp.getint('input','gps-start-time')
except:
  gps_start_time = None

try:
  gps_end_time = cp.getint('input','gps-end-time')
except:
  gps_end_time = None

#############################################################################
# read in playground data mask from ini file 
# set the playground_only option and add to inca and sire jobs
try:
  play_data_mask = string.strip(cp.get('pipeline','playground-data-mask'))
except:
  play_data_mask = None

play_jobs = [sire_job, sire_summary_job]

if play_data_mask == 'playground_only':
  playground_only = 2
  
  for job in play_jobs:
    job.add_opt('data-type','playground_only')

elif play_data_mask == 'exclude_playground':
  playground_only = 0
  
  for job in play_jobs:
    job.add_opt('data-type','exclude_play')


elif play_data_mask == 'all_data':
  playground_only = 0

  for job in play_jobs:
    job.add_opt('data-type','all_data')

else:
  print "Invalid playground data mask " + play_data_mask + " specified"
  sys.exit(1)

 
 
##############################################################################
# get the pad and chunk lengths from the values in the ini file
pad = int(cp.get('data', 'pad-data'))
n = int(cp.get('data', 'segment-length'))
s = int(cp.get('data', 'number-of-segments'))
r = int(cp.get('data', 'sample-rate'))
o = int(cp.get('inspiral', 'segment-overlap'))
length = ( n * s - ( s - 1 ) * o ) / r
overlap = o / r


##############################################################################
#  The meat of the DAG generation comes below
#
#
#  The various data sets we compute are:
# 
#  data[ifo] : the science segments and master chunks
#
#  data_out[ifo] : the analyzable data 
#
#  not_data_out[ifo] : non analyzable data
#
#  analyzed_data[ifos] : the 1,2,3,4 ifo coincident data 
#
#  data_to_do[ifo] : the data to analyze for each ifo
#       (depends upon which of single,double,triple, quadruple data we analyze) 
#
#  And the lists of jobs are:
#
#  chunks_analyzed[ifo] : list of chunks analyzed for each ifo
#
#  coinc_nodes : the double, triple, quadruple coincident thinca nodes
#
#
##############################################################################



##############################################################################
#   Step 1: read science segs that are greater or equal to a chunk 
#   from the input file

print "reading in single ifo science segments and creating master chunks...",
sys.stdout.flush()

segments = {}
data = {}

for ifo in ifo_list:
  try:
    segments[ifo] = cp.get('input', ifo +'-segments')
  except:
    segments[ifo] = None
  
  data[ifo] = pipeline.ScienceData() 
  if segments[ifo]:
    data[ifo].read(segments[ifo],length + 2 * pad) 
    data[ifo].make_chunks(length,overlap,playground_only,0,overlap/2,pad)
    data[ifo].make_chunks_from_unused(length,overlap/2,playground_only,
        0,0,overlap/2,pad)

print "done"

# work out the earliest and latest times that are being analyzed
if not gps_start_time:
  gps_start_time = 10000000000
  for ifo in ifo_list:
    if data[ifo] and (data[ifo][0].start() < gps_start_time):
      gps_start_time = data[ifo][0].start()
  print "GPS start time not specified, obtained from segment lists as " + \
    str(gps_start_time)


if not gps_end_time:
  gps_end_time = 0
  for ifo in ifo_list:
    if data[ifo] and (data[ifo][-1].end() > gps_end_time):
      gps_end_time = data[ifo][0].end()
  print "GPS end time not specified, obtained from segment lists as " + \
    str(gps_end_time)

##############################################################################
#   Step 2: determine analyzable times

data_out = {}
not_data_out = {}
for ifo in ifo_list:
  data_out[ifo] = copy.deepcopy(data[ifo])

  # remove start and end of science segments which aren't analyzed for triggers
  for seg in data_out[ifo]:
    seg.set_start(seg.start() + overlap/2 + pad)
    seg.set_end(seg.end() - overlap/2 - pad)

  if playground_only:
    data_out[ifo].play()

  not_data_out[ifo] = copy.deepcopy(data_out[ifo])
  not_data_out[ifo].coalesce()
  not_data_out[ifo].invert()

# determine the data we can analyze for various detector combinations
analyzed_data = {}

# determine the coincident data, if it is to be analyzed
for ifos in ifo_combinations:
  analyzed_data[ifos] = pipeline.ScienceData()
  if ifos in analyze:
    selected_data = []
    for ifo in ifo_list:
      if ifo in ifos:
        selected_data.append(data_out[ifo])
      else:
        selected_data.append(not_data_out[ifo])
    analyzed_data[ifos] = generate_segments(selected_data[0], 
        selected_data[1], selected_data[2], selected_data[3])

##############################################################################
# Step 3: Compute the Science Segments to analyze

data_to_do = {}

for ifo in ifo_list:
  data_to_do[ifo] = copy.deepcopy(analyzed_data[ifo])
  for ifos in analyze:
    if ifo in ifos:
      data_to_do[ifo].union(analyzed_data[ifos])
  data_to_do[ifo].coalesce() 

##############################################################################
# Step 3b: Set up the injection job

# if doing injections then set up the injection analysis
try: seed = cp.get("input","injection-seed")
except: seed = None

if cp.has_option("input","hardware-injection"):
  inj_file_loc = cp.get("input","hardware-inj-file")
  inspinj = inspiral.InspInjNode(inspinj_job)
  inspinj.set_start(gps_start_time)
  inspinj.set_end(gps_end_time)
  inspinj.set_seed(0)
  inj_file = inspinj.get_output()
  print inj_file
  shutil.copy( inj_file_loc, inj_file)
  inspinj = None
elif seed:
  inspinj = inspiral.InspInjNode(inspinj_job)
  inspinj.set_start(gps_start_time)
  inspinj.set_end(gps_end_time)
  inspinj.set_seed(seed)
  if opts.inspinj: dag.add_node(inspinj)
  inj_file = inspinj.get_output()
else: 
  inj_file = None
  inspinj = None

# add the injection details to inspiral/sire jobs  
if inj_file:
  inj_jobs = [sire_job, sire_summary_job] 
  inj_jobs.extend(insp_jobs.values())

  for job in inj_jobs:  
     job.add_file_opt('injection-file',inj_file)

if inj_file:
  # add the injection coincidence to the sire jobs
  try: 
    sire_job.add_ini_opts(cp,'sire-inj')
    sire_summary_job.add_ini_opts(cp,'sire-inj')
  except: pass

##############################################################################
# Step 4: Determine which of the master chunks needs to be filtered
chunks_analyzed = {}

prev_df = None

for ifo in ifo_list:
  print "setting up jobs to filter " + ifo + " data..."
  sys.stdout.flush()

  (prev_df,chunks_analyzed[ifo]) = analyze_ifo(ifo,data[ifo],data_to_do[ifo],  
      tmplt_jobs[ifo],insp_jobs[ifo],df_job,prev_df,dag,usertag,inspinj)

  # Step 4S: Run sire on the single ifo triggers
  sire_nodes = sire_segments_individually(chunks_analyzed[ifo], sire_job, 
      dag, opts.sire_inspiral, opts.inspiral, ifo, inj_file = inj_file, 
      ifotag="FIRST", usertag = usertag, inspinjNode = inspinj)

  if len(sire_nodes):
    sire_segments(sire_nodes, sire_summary_job, dag, 
        opts.summary_inspiral_triggers, opts.sire_inspiral, ifo, 
        gps_start_time, gps_end_time, inj_file = inj_file, 
        ifotag="SUMMARY_FIRST", usertag = usertag, inspinjNode=inspinj)

  print "done" 


##############################################################################
# Step 6: Run thinca on each of the disjoint sets of coincident data

if opts.coincidence:
  print "setting up thinca jobs..."
  sys.stdout.flush()

  # create a cache of the inspiral jobs
  single_data_analyzed = {}
  inspiral_cache = lal.Cache()
  for ifo in ifo_list:
    for insp in chunks_analyzed[ifo]:
      output_file = insp.get_dag_node().get_output()
      output_cache_entry = lal.Cache.from_urls([output_file])[0]
      inspiral_cache.append(output_cache_entry)
      single_data_analyzed[output_cache_entry] = insp.get_dag_node()

  # get the ligolw_thinca command line arguments
  thinca_job.add_ini_opts(cp, 'thinca')

  # add the vetoes to the ligolw_add and ligolw_thinca jobs
  if cp.has_section("vetoes"):
    lladd_veto_file = cp.get("vetoes","vetoes-file")
    thinca_job.add_opt("vetoes-name",cp.get("vetoes","vetoes-name"))
  else:
    lladd_veto_file = None

  # load the number of distint time-slide files
  if inj_file:
    tisi_file_nums = [0]
  else:
    tisi_file_nums = range(int(cp.get("ligolw_cafe","num-slides-files")))

  coinc_nodes = []
  # loop over time-slide files and create coinc nodes
  for tisi_file_num in tisi_file_nums:

    # we need different jobs for different time-slide files
    tisi_file_name = cp.get("ligolw_cafe","slides-file-%i"%tisi_file_num)

    # create cafe caches
    cafe_extent_limit = float(cp.get("ligolw_cafe","extentlimit"))
    print "\tsetting up cafe caches for tisi file %s"%tisi_file_name
    cafe_caches = ligolw_cafe.ligolw_cafe(inspiral_cache,
        ligolw_tisi.load_time_slides(tisi_file_name,
            gz = tisi_file_name.endswith(".gz")).values(),
        extentlimit = cafe_extent_limit)[1]

    # create the base name for the cafe jobs
    cafe_base = "%%s_%%0%dd%%s" % int(math.log10(len(tisi_file_nums))+1)
    cafeusertag = ""
    if usertag:
      cafeusertag += "_" + usertag
    cafe_base = cafe_base % ('CAFE',tisi_file_num,cafeusertag)

    # create the coinc nodes
    new_coinc_nodes = thinca_coinc(ifo_list, single_data_analyzed, cafe_caches,
        cafe_base, lladd_job, tisi_file_name, lladd_veto_file,
        thinca_job, dag, opts.coincidence,
        opts.inspiral, usertag=usertag, inspinjNode=inspinj)

    coinc_nodes.extend(new_coinc_nodes)

  print "done"


##############################################################################
# Step 7: Write out the LAL cache files for the various output data

if gps_start_time is not None and gps_end_time is not None:
  print "generating cache files for output data products...",
  cache_fname = ''
  for ifo in ifo_analyze:
    cache_fname += ifo
  cache_fname += '-INSPIRAL_HIPE'
  if usertag: cache_fname += '_' + usertag
  cache_fname += '-' + str(gps_start_time) + '-' + \
    str(gps_end_time - gps_start_time) + '.cache'
  output_data_cache = lal.Cache()

  for node in dag.get_nodes():
    if opts.dax and isinstance(node,pipeline.LSCDataFindNode):
      # ignore datafind nodes, as their output is a cache file
      continue

    # add the data generated by the job to the output data cache
    output_file = node.get_output()
    
    output_data_cache.append(lal.Cache.from_urls([output_file])[0])
    if (isinstance(node,inspiral.CoireNode) or \
        isinstance(node,inspiral.SireNode)) and \
        node.get_missed():
      output_data_cache.append(lal.Cache.from_urls([node.get_missed()])[0])

  output_data_cache.tofile(open(cache_fname, "w"))
  print "done"
else:
  print "gps start and stop times not specified: cache files not generated"     

##############################################################################
# Step 8: Setup the Maximum number of jobs for different categories
if not opts.disable_dag_categories:
  for cp_opt in cp.options('condor-max-jobs'):
    dag.add_maxjobs_category(cp_opt,cp.getint('condor-max-jobs',cp_opt))

# Add number of retries to jobs as specified by ihope.ini
if cp.has_option("pipeline", "retry-jobs"):
  num_retries = cp.getint("pipeline", "retry-jobs")
  for node in dag.get_nodes():
    node.set_retry(num_retries)

##############################################################################
# Step 9: Write out the DAG, help message and log file
dag.write_sub_files()
dag.write_dag()

if opts.write_script:
  dag.write_script()

##############################################################################  
# write a message telling the user that the DAG has been written
if opts.dax:
  
  print "\nCreated an abstract DAX file", dag.get_dag_file()
  print "which can be transformed into a concrete DAG with gencdag."
  print "\nSee the documentation on http://www.lsc-group.phys.uwm.edu/lscdatagrid/griphynligo/pegasus_lsc.html"



else:
  print "\nCreated a DAG file which can be submitted by executing"
  print "\n   condor_submit_dag", dag.get_dag_file()
  print """\nfrom a condor submit machine (e.g. hydra.phys.uwm.edu)\n
  If you are running LSCdataFind jobs, do not forget to initialize your grid 
  proxy certificate on the condor submit machine by running the commands
  
    unset X509_USER_PROXY
    grid-proxy-init -hours 72

  Enter your pass phrase when prompted. The proxy will be valid for 72 hours. 
  If you expect the LSCdataFind jobs to take longer to complete, increase the
  time specified in the -hours option to grid-proxy-init. You can check that 
  the grid proxy has been sucessfully created by executing the command:
  
    grid-cert-info -all -file /tmp/x509up_u`id -u`
  
  This will also give the expiry time of the proxy. You should also make sure
  that the environment variable LSC_DATAFIND_SERVER is set the hostname and
  optional port of server to query. For example on the UWM medusa cluster this
  you should use
  
    export LSC_DATAFIND_SERVER=dataserver.phys.uwm.edu
  
  Contact the administrator of your cluster to find the hostname and port of the
  LSCdataFind server.
  """

##############################################################################
# write out a log file for this script
if usertag:
  log_fh = open(basename + '.pipeline.' + usertag + '.log', 'w')
else:
  log_fh = open(basename + '.pipeline.log', 'w')
  
# FIXME: the following code uses obsolete CVS ID tags.
# It should be modified to use git version information.
log_fh.write( "$Id$" + "\n" )
log_fh.write( "$Name$" + "\n\n" )
log_fh.write( "Invoked with arguments:" )
for arg in command_line:
  if arg[0] == '-':
    log_fh.write( "\n" )
  log_fh.write( arg + ' ')

log_fh.write( "\n" )
log_fh.write( "Config file has CVS strings:\n" )
log_fh.write( cp.get('pipeline','version') + "\n" )
log_fh.write( cp.get('pipeline','cvs-tag') + "\n\n" )

print >> log_fh, "\n===========================================\n"
print >> log_fh, "Science Segments and master chunks:\n"

for ifo in ifo_list:
  print >> log_fh, "\n===========================================\n"
  print >> log_fh, ifo + "Data\n"
  for seg in data[ifo]:
    print >> log_fh, " ", seg
    for chunk in seg:
      print >> log_fh, "   ", chunk


for ifo in ifo_list:
  print >> log_fh, "\n===========================================\n"
  log_fh.write( 
    "Filtering " + str(len(chunks_analyzed[ifo])) + " " + ifo + \
    " master chunks\n" )
  total_time = 0
  for ifo_done in chunks_analyzed[ifo]:
    print >> log_fh, ifo_done.get_chunk()
    total_time += len(ifo_done.get_chunk())
  print >> log_fh, "\n total time", total_time, "seconds"

for ifo in ifo_list:
  print >> log_fh, "\n===========================================\n"
  log_fh.write( "Writing " + str(len(analyzed_data[ifo])) + " " + ifo + \
    " single IFO science segments\n" )
  total_time = 0
  for seg in analyzed_data[ifo]:
    print >> log_fh, seg
    total_time += seg.dur()
  print >> log_fh, "\n total time", total_time, "seconds"

  if opts.output_segs and len(analyzed_data[ifo]):
    if playground_only:
      f = open(ifo + '_play_segs_analyzed.txt', 'w')
    else:  
      f = open(ifo + '_segs_analyzed.txt', 'w')
    for seg in analyzed_data[ifo]:
      f.write('%4d %10d %10d %6d\n' % (seg.id(), seg.start(), seg.end(), 
        seg.dur()))
    f.close()


for ifos in ifo_coincs:  
  print >> log_fh, "\n===========================================\n"
  log_fh.write( "Writing " + str(len(analyzed_data[ifos])) + " " + ifos + \
    " coincident segments\n" )
  total_time = 0
  for seg in analyzed_data[ifos]:
    print >> log_fh, seg
    total_time += seg.dur()
  print >> log_fh, "\n total time", total_time, "seconds"

  if opts.output_segs and len(analyzed_data[ifos]):
    if playground_only:
      f = open(ifos + '_play_segs_analyzed.txt', 'w')
    else:  
      f = open(ifos + '_segs_analyzed.txt', 'w')
    for seg in analyzed_data[ifos]:
      f.write('%4d %10d %10d %6d\n' % (seg.id(), seg.start(), seg.end(), 
        seg.dur()))
    f.close()

sys.exit(0)


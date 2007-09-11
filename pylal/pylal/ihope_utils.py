#!/usr/bin/env @PYTHONPROG@
"""
this code contains some functions used by ihope (and followup_pipe)

$Id$
"""
__author__ = 'Stephen Fairhurst <sfairhur@gravity.phys.uwm.edu>'
__date__ = '$Date$'
__version__ = '$Revision$'[11:-2]

##############################################################################
# import standard modules and append the lalapps prefix to the python path
import os, sys, copy
import ConfigParser
import optparse
import tempfile
import urllib
sys.path.append('@PYTHONLIBDIR@')

##############################################################################
# import the modules we need to build the pipeline
from glue import segments
from glue import segmentsUtils
from glue import pipeline

##############################################################################
# Functions used in setting up the dag:
def make_external_call(command, show_stdout=False, show_command=False):
  """
  Run a program on the shell and print informative messages on failure.
  """
  if show_command: print command

  stdin, out, err = os.popen3(command)
  pid, status = os.wait()

  if status != 0:
      print >>sys.stderr, "External call failed."
      print >>sys.stderr, "  status: %d" % status
      print >>sys.stderr, "  stdout: %s" % out.read()
      print >>sys.stderr, "  stderr: %s" % err.read()
      print >>sys.stderr, "  command: %s" % command
      sys.exit(status)
  if show_stdout:
      print out.read()
  stdin.close()
  out.close()
  err.close()

##############################################################################
def mkdir( newdir ):
  """
  Create a directory, don't complain if it exists
  
  newdir = name of directory to be created
  """
  if os.path.isdir(newdir): pass
  elif os.path.isfile(newdir):
    raise OSError("a file with the same name as the desired " \
                  "dir, '%s', already exists." % newdir)
  else: os.mkdir(newdir)
 
##############################################################################
def link_executables(directory, config):
  """
  link executables to given directory
  """
  for (job, executable) in config.items("condor"):
    if job != "universe":
      if executable[0] != "/": 
        executable = "../../" + executable
      config.set("condor", job, executable)

##############################################################################
# Function to set up the segments for the analysis
def science_segments(ifo, config, opts):
  """
  generate the segments for the specified ifo
  """
  segFindFile = ifo + "-SCIENCE_SEGMENTS-" + str(opts.gps_start_time) + "-" + \
      str(opts.gps_end_time - opts.gps_start_time) + ".txt"

  # if not generating segments, all we need is the name of the segment file
  if not opts.generate_segments: return segFindFile

  executable = config.get("condor", "segfind")
  if executable[0] != "/": executable = "../" + executable

  # run segFind to determine science segments
  segFindCall = executable + " --interferometer=" + ifo + \
      " --type=\"" + config.get("segments", "analyze") + "\""\
      " --gps-start-time=" + str(opts.gps_start_time) + \
      " --gps-end-time=" + str(opts.gps_end_time) + " > " + segFindFile
  make_external_call(segFindCall)
  return segFindFile

##############################################################################
# Function to set up the segments for the analysis
def veto_segments(ifo, config, segmentList, dqSegFile, categories, opts):
  """
  generate veto segments for the given ifo
  
  ifo         = name of the ifo
  segmentList = list of science mode segments
  dqSegfile   = the file containing dq flags
  categories  = list of veto categories 
  """
  executable = config.get("condor", "query_dq")
  if executable[0] != "/": executable = "../" + executable

  vetoFiles = {}

  for category in categories:    
    dqFile = config.get("segments", ifo.lower() + "-cat-" + str(category) + \
        "-veto-file")
    if dqFile[0] != "/": dqFile = "../" + dqFile

    vetoFile = ifo + "-CATEGORY_" + str(category) + "_VETO_SEGS-" + \
        str(opts.gps_start_time) + "-" + \
        str(opts.gps_end_time - opts.gps_start_time) + ".txt"

    dqCall = executable + " --ifo " + ifo + " --dq-segfile " + dqSegFile + \
        " --segfile " + segmentList + " --flagfile " + dqFile + \
        " --outfile " + vetoFile

    # generate the segments
    make_external_call(dqCall)

    # if there are previous vetoes, generate combined
    try: previousSegs = \
        segmentsUtils.fromsegwizard(open(vetoFiles[category-1]))
    except: previousSegs = None

    if previousSegs:
      combinedFile = ifo + "-COMBINED_CAT_" + str(category) + "_VETO_SEGS-" + \
          str(opts.gps_start_time) + "-" + \
          str(opts.gps_end_time - opts.gps_start_time) + ".txt"

      vetoSegs = segmentsUtils.fromsegwizard(open(vetoFile)).coalesce()
      vetoSegs |= previousSegs
      segmentsUtils.tosegwizard(file(combinedFile,"w"), vetoSegs)
      vetoFiles[category] = combinedFile

    else: vetoFiles[category] = vetoFile

  return vetoFiles

##############################################################################
# Function to set up the segments for the analysis
def datafind_segments(ifo, config, opts):
  ligoIfos = ["H1","H2","L1"]

  if ifo in ligoIfos: type = config.get("input","ligo-type")
  elif ifo == "G1": type =   config.get("input","geo-type")
 
  executable = config.get("condor", "datafind")
  if executable[0] != "/": executable = "../" + executable

  ifo_type = ifo + "_" + type
  dataFindFile = ifo_type + "-" + str(opts.gps_start_time) + "-" + \
      str(opts.gps_end_time - opts.gps_start_time) + ".txt"

  print "Running LSCdataFind to determine available data from " + type + \
      " frames for " + ifo
  dataFindCall = executable + " --observatory=" + ifo[0] + \
      " --type=" + ifo_type + \
      " --gps-start-time=" + str(opts.gps_start_time) + \
      " --gps-end-time=" + str(opts.gps_end_time) + " --show-times > " + \
      dataFindFile
  make_external_call(dataFindCall)
  dfSegs = segmentsUtils.fromsegwizard(file(dataFindFile)).coalesce()

  return dfSegs

##############################################################################
# Function to set up lalapps_inspiral_hipe
def hipe_setup(hipeDir, config, opts, ifos, injFile=None, dfOnly = False, playOnly = False, vetoCat = None, vetoFiles = None):
  """
  run lalapps_inspiral_hipe and add job to dag
  hipeDir   = directory in which to run inspiral hipe
  config    = config file 
  injFile   = injection file to use when running
  dfOnly    = only run the datafind step of the pipeline
  vetoCat   = run this category of veto
  vetoFiles = dictionary of veto files
  """

  # make the directory for running hipe
  mkdir(hipeDir)

  # create the hipe config parser, keep only relevant info
  hipecp = copy.deepcopy(config)
  if dfOnly:
    hipeSections = ['condor', 'pipeline', 'input', 'datafind','data',\
        'ligo-data','inspiral']
  elif vetoCat:
    hipeSections = ['condor', 'pipeline', 'input', 'data', 'ligo-data', \
        'inspiral', 'thinca', 'thinca-2', 'datafind', \
        'thinca-slide', 'coire', 'coire-inj']
  else:
    hipeSections = ['condor', 'pipeline', 'input', 'calibration', 'datafind',\
        'ligo-data', 'geo-data', 'data', 'tmpltbank', 'tmpltbank-1', \
        'tmpltbank-2', 'no-veto-inspiral', 'veto-inspiral', 'inspiral', \
        'h1-inspiral', 'h2-inspiral', 'l1-inspiral', 'g1-inspiral', \
        'thinca', 'thinca-1', 'thinca-2', 'thinca-slide', 'trigtotmplt', \
        'sire', 'sire-inj', 'coire', 'coire-inj']

  for seg in hipecp.sections():
    if not seg in hipeSections: hipecp.remove_section(seg)

  hipecp.remove_option("condor","hipe")
  hipecp.remove_option("condor","follow")

  hipecp.set("input", "gps-start-time", opts.gps_start_time)
  hipecp.set("input", "gps-end-time", opts.gps_end_time)

  # set the data type
  if playOnly:
    hipecp.set("pipeline", "playground-data-mask", "playground_only")
  else:
    hipecp.set("pipeline", "playground-data-mask", "all_data")

  # deal with vetoes
  if vetoCat:
    for section in ["thinca", "coire"]: 
      hipecp.set(section, "user-tag","CAT_" + str(vetoCat) + "_VETO")
    for ifo in ifos:
      hipecp.set("thinca", ifo.lower() + "-veto-file", "../" + vetoFiles[ifo][vetoCat])
      
  if injFile:
    # add the injection options to the ini file
    if injFile[0] != "/": injFile = "../../" + injFile
    hipecp.set("input", "injection-file", injFile )
    hipecp.set("input", "num-slides", "")
  else: 
    # add the time slide to the ini file
    hipecp.set("input","num-slides", config.get("input","num-slides") )
    hipecp.set("input", "injection-file", "" )

    # sanity check of numSlides
    maxLength = None
    if playOnly: maxLength = 600
    elif hipecp.has_option("input", "max-thinca-segment"):
      maxLength = hipecp.getint("input", "max-thinca-segment")

    if maxLength:
      maxSlide = max([config.getint("thinca-slide", ifo.lower() + "-slide") \
          for ifo in ifos])
      numSlides = (maxLength/2/maxSlide - 1)
      if numSlides < hipecp.getint("input", "num-slides"):
        print "Setting number of slides to " + str(numSlides) + \
            " to avoid double wrapping"
        hipecp.set("input","num-slides", str(numSlides))

  # link the executables 
  link_executables(hipeDir, hipecp)

  # return to the directory, write ini file and run hipe
  os.chdir(hipeDir)
  iniFile = "inspiral_hipe_" 
  if vetoCat: iniFile += "cat" + str(vetoCat) + "_veto_"
  iniFile += hipeDir + ".ini"

  hipecp.write(file(iniFile,"w"))
  
  print "Running hipe in directory " + hipeDir 
  if injFile: print "Injection file: " + hipecp.get("input", "injection-file") 
  else: print "No injections, " + str(hipecp.get("input","num-slides")) + \
      " time slides"
  if vetoCat: print "Running the category " + str(vetoCat) + " vetoes"
  print

  # work out the hipe call:
  hipeCommand = config.get("condor","hipe")
  hipeCommand += " --log-path " + opts.log_path
  hipeCommand += " --config-file " + iniFile
  if playOnly: hipeCommand += " --priority 10"
  for item in config.items("ifo-details"):
      hipeCommand += " --" + item[0] + " " + item[1]

  for item in config.items("hipe-arguments"):
    if (dfOnly and item[0] == "datafind") or  \
        (vetoCat and item[0] in ["second-coinc", "coire-second-coinc"]) or \
        (not dfOnly and not vetoCat and item[0] != "datafind"):
      hipeCommand += " --" + item[0] + " " + item[1]

  # run lalapps_inspiral_hipe
  make_external_call(hipeCommand) 

  # link datafind
  if not dfOnly and not vetoCat:
    try:
      os.rmdir("cache")
      os.symlink("../datafind/cache", "cache")
    except: pass

  # make hipe job/node
  hipeJob = pipeline.CondorDAGManJob(hipeDir + "/" + iniFile.rstrip("ini") + \
      "dag")
  if vetoCat: hipeJob.add_opt("maxjobs", "5")
  hipeNode = pipeline.CondorDAGNode(hipeJob)

  # add postscript to deal with rescue dag
  fix_rescue(hipeNode)

  # return to the original directory
  os.chdir("..")

  return hipeNode

##############################################################################
# Function to set up lalapps_followup_pipe
def followup_setup(followupDir, config, opts, hipeDir):
  """
  run lalapps_followup_pipe and add job to dag
  followupDir = directory to output the followup
  config    = config file 
  """

  # make the directory for followup pipe
  mkdir(followupDir)

  # create the followup config parser, keep only relevant info
  followupcp = copy.deepcopy(config)
  followupSections = ['condor', 'hipe-cache', 'triggers', 'datafind', \
      'q-datafind', 'qscan', 'q-hoft-datafind', 'qscan-hoft', \
      'plots', 'output', 'seg']

  for seg in followupcp.sections():
    if not seg in followupSections: followupcp.remove_section(seg)

  followupcp.remove_option("condor","hipe")
  followupcp.remove_option("condor","follow")

  # XXX this should be replaced by getting the information from the hipe cache
  # set the cache paths
  followupcp.add_section("hipe-cache")
  followupcp.set("hipe-cache", "hipe-cache-path", "hipe_cache")
  followupcp.set("hipe-cache", "science-run", "S5")

  for path in ["tmpltbank-path", "trigbank-path", "first-inspiral-path", \
      "second-inspiral-path", "first-coinc-path", "second-coinc-path"]: 
    followupcp.set("hipe-cache", path, "../" + hipeDir)

  # set the xml-glob
  followupcp.set("triggers", "xml-glob", "../" + hipeDir + "/*COIRE*H*xml")
  # to here XXX

  # correct paths to qscan config files
  for section in ["qscan", "qscan-hoft"]:
    for (opt, arg) in followupcp.items(section):
      if "config-file" in opt and arg[0] != "/": 
        arg = "../../" + arg
        followupcp.set(section, opt, arg)

  # link the executables 
  link_executables(followupDir, followupcp)

  # return to the directory, write ini file and run hipe
  os.chdir(followupDir)
  iniFile = "followup_pipe_" + followupDir + ".ini"
  followupcp.write(file(iniFile,"w"))
  
  # link datafind output from original hipe
  try: os.symlink("../datafind/cache", "hipe_cache")
  except: pass
  print "Running followup pipe in directory " + followupDir 

  # work out the followup_pipe call:
  followupCommand = config.get("condor","follow")
  followupCommand += " --log-path " + opts.log_path
  followupCommand += " --config-file " + iniFile

  for item in config.items("followup-arguments"):
    followupCommand += " --" + item[0] + " " + item[1]

  # set up a fake followup dag -- the real one can't be generated until the
  # analysis is done
  followupDag = iniFile.rstrip("ini") + "dag"
  f = open(followupDag,"w")
  f.write("\n")
  f.close()

  # add job to dag
  followupJob = pipeline.CondorDAGManJob(followupDir + "/" + followupDag)
  followupNode = pipeline.CondorDAGNode(followupJob)

  # write the pre-script to run lalapps_followup_pipe at the appropriate time
  f = open(followupDag + ".pre","w")
  f.write("#! /bin/bash\n")
  f.write("cd followup\n")
  f.write(followupCommand)
  f.write("cd ..\n")
  f.close()
  os.chmod(followupDag + ".pre", 0744)
  followupNode.set_pre_script(followupDir + "/" + followupDag + ".pre")

  # add postscript to deal with rescue dag
  fix_rescue(followupNode)

  # return to the original directory
  os.chdir("..")

  return followupNode

##############################################################################
# Function to fix the rescue of inner dags
def fix_rescue(dagNode):
  """
  add a postscript to deal with the rescue dag correctly
 
  dagNode = the node for the subdag
  """
  dagNode.set_post_script( "rescue.sh")
  dagNode.add_post_script_arg( "$RETURN" ) 
  dagNode.add_post_script_arg( dagNode.job().get_sub_file().rstrip(".condor.sub") )

def write_rescue():
  # Write the rescue post-script
  # XXX FIXME: This is a hack, required until condor is fixed XXX
  f = open("rescue.sh", "w")
  f.write("""#! /bin/bash
  if [ ! -n "${2}" ]
  then
    echo "Usage: `basename $0` DAGreturn DAGfile"
    exit
  fi
  
  if (( ${1}>0 ))
  then
    DIR=${2%/*}
    DAG=${2#*/}
    mv ${2} ${2}.orig
    `sed "s/DIR ${DIR}//g" ${DAG}.rescue > ${2}`
    exit ${1}
  fi""")
  f.close()
  os.chmod("rescue.sh", 0744)

##############################################################################
# Function to determine the segments to analyze (science segments, data quality, missing segments)

def findSegmentsToAnalyze(config,opts,ifo,dq_url_pattern,segFile,dqVetoes=None):

  segFile[ifo] = ifo + "-SELECTED_SEGS.txt"
  missedFile = ifo + "-MISSED_SEGS.txt"
  dqSegFile = ifo + "-DQ_SEGMENTS.txt"

  if config.has_section("input"):
    config.set("input", ifo.lower() + "-segments", "../" + segFile[ifo])

  if opts.generate_segments:
    print "Generating science segments for " + ifo + " ...",
    sys.stdout.flush()
  sciSegFile = science_segments(ifo, config, opts)
  sciSegs = segmentsUtils.fromsegwizard(file(sciSegFile)).coalesce()
  if opts.generate_segments: print " done."

  # download the dq segments to generate the veto files
  if opts.generate_segments:
    print "Downloading the latest daily dump of segment database to " \
        + dqSegFile + " ...",
    dqSegFile, info = urllib.urlretrieve(dq_url_pattern % ifo, dqSegFile)
    print "done"

    print "Generating cat 1 veto segments for " + ifo + " ...",
    sys.stdout.flush()
    vetoFiles = veto_segments(ifo, config, sciSegFile, dqSegFile, [1], opts)
    print "done"

    # remove cat 1 veto times
    vetoSegs = segmentsUtils.fromsegwizard(open(vetoFiles[1])).coalesce()
    sciSegs = sciSegs.__and__(vetoSegs.__invert__())

    if opts.use_available_data:
      dfSegs = datafind_segments(ifo, config, opts)
      analyzedSegs = sciSegs.__and__(dfSegs)
      missedSegs = sciSegs.__and__(dfSegs.__invert__())
      segmentsUtils.tosegwizard(file(missedFile,"w"), missedSegs)
      print "Writing " + ifo + " segments which cannot be analyzed to file " + \
          missedFile
      print "Not analyzing %d s, representing %.2f percent of time" %  \
         (missedSegs.__abs__(),
         100. * missedSegs.__abs__() / analyzedSegs.__abs__() )

    else: analyzedSegs = sciSegs

    segmentsUtils.tosegwizard(file(segFile[ifo],"w"), analyzedSegs)
    print "Writing " + ifo + " segments of total time " + \
        str(analyzedSegs.__abs__()) + "s to file: " + segFile[ifo]
    print "done"

  if opts.run_data_quality:
    print "Generating veto segments for " + ifo + "..."
    sys.stdout.flush()
    if dqVetoes == None:
      print >> sys.stderr, "the dqVetoes list needs to be specified as input"
      sys.exit(1)
    dqVetoes[ifo] = veto_segments(ifo, config, segFile[ifo], dqSegFile, [2,3,4], opts)
    print "done"


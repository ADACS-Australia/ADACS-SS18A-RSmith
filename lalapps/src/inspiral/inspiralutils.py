"""
Classes needed for ihope.
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
def hipe_cache(ifos, usertag, gps_start_time, gps_end_time):
  """
  return the name of the hipe cache 

  ifos    = list of ifos
  usertag = usertag
  gps_start_time = start time of analysis
  gps_end_time   = end time of analysis
  """
  hipeCache = "".join(ifos) + "-INSPIRAL_HIPE" 
  if usertag:
    hipeCache += "_" + usertag 
  hipeCache += "-" + str(gps_start_time) + "-" + \
      str(gps_end_time - gps_start_time)  + ".cache"

  return hipeCache

##############################################################################
# Function to set up the segments for the analysis
def science_segments(ifo, config, generate_segments = True):
  """
  generate the segments for the specified ifo
  @param ifo:    the name of the ifo
  @param config: the configParser object with analysis details
  @param generate_segments: whether or not to actually make segments
  """
  start = config.getint("input","gps-start-time")
  end = config.getint("input","gps-end-time")

  segFindFile = ifo + "-SCIENCE_SEGMENTS-" + str(start) + "-" + \
      str(end - start) + ".txt"

  # if not generating segments, all we need is the name of the segment file
  if not generate_segments: return segFindFile

  executable = config.get("condor", "segfind")

  whichtoanalyze = ifo.lower() + "-analyze"
  print "For " + ifo + ", analyze " +config.get("segments",whichtoanalyze)

  # run segFind to determine science segments
  segFindCall = executable 
  for opt,arg in config.items("segfind"):
    segFindCall += " --" + opt + " " + arg
  segFindCall +=  " --interferometer=" + ifo + \
      " --type=\"" + config.get("segments", whichtoanalyze) + "\""\
      " --gps-start-time=" + str(start) + \
      " --gps-end-time=" + str(end) + " > " + segFindFile
  make_external_call(segFindCall)

  return segFindFile

##############################################################################
# Function to set up the segments for the analysis
def veto_segments(ifo, config, segmentList, dqSegFile, categories, 
  generateVetoes): 
  """
  generate veto segments for the given ifo

  @param ifo         : name of the ifo
  @param segmentList : list of science mode segments
  @param dqSegfile   : the file containing dq flags
  @param categories  : list of veto categories
  @param generateVetoes: generate the veto files
  """
  executable = config.get("condor", "query_dq")
  start = config.getint("input","gps-start-time")
  end = config.getint("input","gps-end-time")
  vetoFiles = {}

  for category in categories:
    dqFile = config.get("segments", ifo.lower() + "-cat-" + str(category) + \
        "-veto-file")

    vetoFile = ifo + "-CATEGORY_" + str(category) + "_VETO_SEGS-" + \
        str(start) + "-" + \
        str(end - start) + ".txt"

    dqCall = executable + " --ifo " + ifo + " --dq-segfile " + dqSegFile + \
        " --segfile " + segmentList + " --flagfile " + dqFile + \
        " --outfile " + vetoFile

    # generate the segments
    if generateVetoes: make_external_call(dqCall)

    # if there are previous vetoes, generate combined
    if (category-1) in categories: 

      combinedFile = ifo + "-COMBINED_CAT_" + str(category) + "_VETO_SEGS-" + \
          str(start) + "-" + \
          str(end - start) + ".txt"
      vetoFiles[category] = combinedFile

      if generateVetoes: 
        previousSegs = segmentsUtils.fromsegwizard(open(vetoFiles[category-1]))
        vetoSegs = segmentsUtils.fromsegwizard(open(vetoFile)).coalesce()
        vetoSegs |= previousSegs
        segmentsUtils.tosegwizard(file(combinedFile,"w"), vetoSegs)

    else: vetoFiles[category] = vetoFile

  return vetoFiles

##############################################################################
# Function to set up the segments for the analysis
def datafind_segments(ifo, config):
  """
  generate datafind segments for the given ifo

  @param ifo         : name of the ifo
  @param config: the configParser object with analysis details
  """
  ligoIfos = ["H1","H2","L1"]

  if ifo in ligoIfos: type = config.get("input","ligo-type")
  elif ifo == "G1": type = config.get("input","geo-type")
  elif ifo == "V1": type = config.get("input","virgo-type")

  if ifo == "V1": ifo_type = type
  else: ifo_type = ifo + "_" + type

  executable = config.get("condor", "datafind")
  start = config.getint("input","gps-start-time")
  end = config.getint("input","gps-end-time")


  dataFindFile = ifo_type + "-" + str(start) + "-" + \
      str(end - start) + ".txt"

  print "Running LSCdataFind to determine available data from " + type + \
      " frames for " + ifo
  dataFindCall = executable 
  for opt,arg in config.items("datafind"):
    dataFindCall += " --" + opt + " " + arg
  dataFindCall += " --observatory=" + ifo[0] + \
      " --type=" + ifo_type + \
      " --gps-start-time=" + str(start) + \
      " --gps-end-time=" + str(end) + " --show-times > " + \
      dataFindFile
  make_external_call(dataFindCall)
  dfSegs = segmentsUtils.fromsegwizard(file(dataFindFile)).coalesce()

  return dfSegs

##############################################################################
# Function to determine the segments to analyze 
#(science segments, data quality, missing segments)
def findSegmentsToAnalyze(config,ifo,generate_segments=True,\
    use_available_data=False,data_quality_vetoes=False):
  """
  generate segments for the given ifo

  @param ifo         : name of the ifo
  @param config      : the configParser object with analysis details
  @param generate_segments: whether the segment files should be generated
  @param use_available_data: restrict segments to data which actually available
  @param data_quality_vetoes: generate the cat2,3,4 DQ veto segments
  """

  start = config.getint("input","gps-start-time")
  end = config.getint("input","gps-end-time")
    
  # file names
  segFile = ifo + "-SELECTED_SEGS-" + str(start) + "-" + \
      str(end - start) + ".txt"
  missedFile = ifo + "-MISSED_SEGS-" + str(start) + "-" + \
      str(end - start) + ".txt"
  dqSegFile = ifo + "-DQ_SEGMENTS-" + str(start) + "-" + \
      str(end - start) + ".txt"

  if generate_segments:
    print "Generating science segments for " + ifo + " ...",
    sys.stdout.flush()
  sciSegFile = science_segments(ifo, config, generate_segments)
  if generate_segments: 
    sciSegs = segmentsUtils.fromsegwizard(file(sciSegFile)).coalesce()
    print " done."

  # download the dq segments to generate the veto files
  if generate_segments:
    # XXX FIXME:hard coded path to get the dq-segments from:
    dq_url_pattern = \
        "http://ldas-cit.ligo.caltech.edu/segments/S5/%s/dq_segments.txt"

    print "Downloading the latest daily dump of segment database to " \
        + dqSegFile + " ...",
    sys.stdout.flush()
    dqSegFile, info = urllib.urlretrieve(dq_url_pattern % ifo, dqSegFile)
    print "done"

    print "Generating cat 1 veto segments for " + ifo + " ...",
    sys.stdout.flush()
    vetoFiles = veto_segments(ifo, config, sciSegFile, dqSegFile, [1], \
        generate_segments )
    print "done"

    # remove cat 1 veto times
    vetoSegs = segmentsUtils.fromsegwizard(open(vetoFiles[1])).coalesce()
    sciSegs = sciSegs.__and__(vetoSegs.__invert__())

    if use_available_data:
      dfSegs = datafind_segments(ifo, config)
      analyzedSegs = sciSegs.__and__(dfSegs)
      missedSegs = sciSegs.__and__(dfSegs.__invert__())
      segmentsUtils.tosegwizard(file(missedFile,"w"), missedSegs)
      print "Writing " + ifo + " segments which cannot be analyzed to file " \
          + missedFile
      print "Not analyzing %d s, representing %.2f percent of time" %  \
         (missedSegs.__abs__(),
         100. * missedSegs.__abs__() / analyzedSegs.__abs__() )

    else: analyzedSegs = sciSegs

    segmentsUtils.tosegwizard(file(segFile,"w"), analyzedSegs)
    print "Writing " + ifo + " segments of total time " + \
        str(analyzedSegs.__abs__()) + "s to file: " + segFile
    print "done"

  if data_quality_vetoes: 
    print "Generating veto segments for " + ifo + "..."
    sys.stdout.flush()
  dqVetoes = veto_segments(ifo, config, segFile, dqSegFile, [2,3,4], 
      data_quality_vetoes )
  if data_quality_vetoes: print "done"

  return tuple([segFile, dqVetoes])


##############################################################################
# Function to set up lalapps_inspiral_hipe
def hipe_setup(hipeDir, config, ifos, logPath, injSeed=None, dfOnly = False, \
    playOnly = False, vetoCat = None, vetoFiles = None):
  """
  run lalapps_inspiral_hipe and add job to dag
  hipeDir   = directory in which to run inspiral hipe
  config    = config file
  logPath   = location where log files will be written
  injSeed   = injection file to use when running
  dfOnly    = only run the datafind step of the pipeline
  vetoCat   = run this category of veto
  vetoFiles = dictionary of veto files
  """

  # make the directory for running hipe
  mkdir(hipeDir)

  # create the hipe config parser, keep only relevant info
  hipecp = copy.deepcopy(config)
  if dfOnly:
    hipeSections = ["condor", "pipeline", "input", "datafind","data",\
        "ligo-data","inspiral","virgo-data"]
  elif vetoCat:
    hipeSections = ["condor", "pipeline", "input", "data", "ligo-data", \
        "inspiral", "thinca", "thinca-2", "datafind", "virgo-data", \
        "thinca-slide", "coire", "coire-inj"]
  else:
    hipeSections = ["condor", "pipeline", "input", "calibration", "datafind",\
        "ligo-data", "virgo-data", "geo-data", "data", "tmpltbank", \
        "tmpltbank-1", "tmpltbank-2", "h1-tmpltbank", "h2-tmpltbank", \
        "l1-tmpltbank", "v1-tmpltbank", "g1-tmpltbank", "no-veto-inspiral", \
        "veto-inspiral", "inspiral", "h1-inspiral", "h2-inspiral", \
        "l1-inspiral", "g1-inspiral", "v1-inspiral", "thinca", "thinca-1", \
        "thinca-2", "thinca-slide", "trigbank", "sire", "sire-inj", \
        "coire", "coire-inj"]

  for seg in hipecp.sections():
    if not seg in hipeSections: hipecp.remove_section(seg)

  hipecp.remove_option("condor","hipe")
  hipecp.remove_option("condor","plot")
  hipecp.remove_option("condor","follow")

  # add the inspinj section
  hipecp.add_section("inspinj")

  # set the data type
  if playOnly:
    hipecp.set("pipeline", "playground-data-mask", "playground_only")
  else:
    hipecp.set("pipeline", "playground-data-mask", "all_data")

  # set the user-tag to be the same as the directory name
  if hipecp.get("pipeline","user-tag"):
    usertag = hipecp.get("pipeline", "user-tag") + "_" + hipeDir.upper()
  else:
    usertag = hipeDir.upper()

  if vetoCat:
    # set the old usertag in inspiral and inspinj, 
    # so that we pick up the correct xml inputs
    sections = ["inspiral"]
    for section in ["inspiral","inspinj"]:
      hipecp.set(section, "user-tag",usertag)

    # set the correct pipeline usertag
    usertag += "_CAT_" + str(vetoCat) + "_VETO"

    # add the veto files in the thinca section
    for ifo in ifos:
      hipecp.set("thinca", ifo.lower() + "-veto-file", vetoFiles[ifo][vetoCat])
    hipecp.set("thinca", "do-veto", "")

  # set the usertag
  hipecp.set("pipeline", "user-tag",usertag)

  if injSeed:
    # copy over the arguments from the relevant injection section
    for (name,value) in config.items(hipeDir):
      hipecp.set("inspinj",name,value)
    hipecp.remove_section(hipeDir)
    hipecp.set("input","injection-seed",injSeed)

  else:
    # add the time slide to the ini file
    hipecp.set("input","num-slides", config.get("input","num-slides") )

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

  # return to the directory, write ini file and run hipe
  os.chdir(hipeDir)
  iniFile = "inspiral_hipe_"
  iniFile += hipeDir 
  if vetoCat: iniFile += "_cat" + str(vetoCat) + "_veto"
  iniFile += ".ini"

  hipecp.write(file(iniFile,"w"))

  print "Running hipe in directory " + hipeDir
  if dfOnly: print "Running datafind only"
  elif injSeed: print "Injection seed: " + injSeed
  else: print "No injections, " + str(hipecp.get("input","num-slides")) + \
      " time slides"
  if vetoCat: print "Running the category " + str(vetoCat) + " vetoes"
  print

  # work out the hipe call:
  hipeCommand = config.get("condor","hipe")
  hipeCommand += " --log-path " + logPath
  hipeCommand += " --config-file " + iniFile
  if playOnly: hipeCommand += " --priority 10"
  for item in config.items("ifo-details"):
    hipeCommand += " --" + item[0] + " " + item[1]

  def test_and_add_hipe_arg(hipeCommand, hipe_arg):
    if config.has_option("hipe-arguments",hipe_arg):
      hipeCommand += "--" + hipe_arg + " " + \
        config.get("hipe-arguments",hipe_arg)
    return(hipeCommand)

  if dfOnly:
    hipeCommand = test_and_add_hipe_arg(hipeCommand,"datafind")
  elif vetoCat:
    hipeCommand = test_and_add_hipe_arg(hipeCommand,"second-coinc")
    hipeCommand = test_and_add_hipe_arg(hipeCommand,"coire-second-coinc")
  else:
    omit = ["datafind", "disable-dag-categories", "disable-dag-priorities"]
    for (opt, arg) in config.items("hipe-arguments"):
      if opt not in omit:
        hipeCommand += "--" + opt + " " + arg 

  hipeCommand = test_and_add_hipe_arg(hipeCommand,"disable-dag-categories")
  hipeCommand = test_and_add_hipe_arg(hipeCommand,"disable-dag-priorities")

  # run lalapps_inspiral_hipe
  make_external_call(hipeCommand)

  # link datafind
  if not dfOnly and not vetoCat:
    try:
      os.rmdir("cache")
      os.symlink("../datafind/cache", "cache")
    except: pass

  # make hipe job/node
  hipeDag = iniFile.rstrip("ini") + usertag + ".dag"
  hipeJob = pipeline.CondorDAGManJob(hipeDir + "/" + hipeDag)
  if vetoCat: hipeJob.add_opt("maxjobs", "5")
  hipeNode = pipeline.CondorDAGNode(hipeJob)

  hipeNode.add_output_file( hipe_cache(ifos, usertag, \
      hipecp.getint("input", "gps-start-time"), \
      hipecp.getint("input", "gps-end-time")) )
  # add postscript to deal with rescue dag
  fix_rescue(hipeNode)

  # return to the original directory
  os.chdir("..")

  return hipeNode

##############################################################################
# Function to set up lalapps_plot_hipe
def plot_setup(plotDir, config, logPath, stage, injectionSuffix,
    zerolagSuffix, slideSuffix, bankSuffix, cacheFile, tag = None):
  """
  run lalapps_plot_hipe and add job to dag
  plotDir   = directory in which to run inspiral hipe
  config    = config file
  logPath   = location where log files will be written
  stage     = which stage to run (first, second or both)
  injectionSuffix = the string to restrict to for injections
  zerolagSuffix   = the string to restrict to for zero lag
  slideSuffix     = the string to restrict to for time slides
  bankSuffix      = the string to restrict to for bank plots
  cacheFile       = the input cache file for plotting
  tag             = extra tag for naming
  """
  # make the directory for running hipe
  mkdir(plotDir)

  plotcp = copy.deepcopy(config)

  # set details for the common section
  plotcp.add_section("common")
  plotcp.set("common","gps-start-time", plotcp.get("input","gps-start-time") )
  plotcp.set("common","gps-end-time", plotcp.get("input","gps-end-time") )
  plotcp.set("common","output-path", ".")
  plotcp.set("common","enable-output","")

  plotSections = ["common", "pipeline", "condor",\
      "plotinspiral", "plotinspiral-meta", \
      "plotthinca", "plotthinca-meta", \
      "plotnumtemplates", "plotnumtemplates-meta", \
      "plotinjnum", "plotinjnum-meta", \
      "plotethinca", "plotethinca-meta", \
      "plotinspmissed", "plotinspmissed-meta", \
      "plotinspinj", "plotinspinj-meta", \
      "plotsnrchi", "plotsnrchi-meta", \
      "plotinspiralrange", "plotinspiralrange-meta"]

  for seg in plotcp.sections():
    if not seg in plotSections: plotcp.remove_section(seg)

  plotcp.remove_option("condor","hipe")
  plotcp.remove_option("condor","plot")
  plotcp.remove_option("condor","follow")

  # XXX Can't yet run the plotting codes in standard universe
  if plotcp.get("condor","universe") == "standard":
    plotcp.set("condor","universe","vanilla")

  # set the various suffixes in pipeline
  plotcp.set("pipeline","injection-suffix",injectionSuffix)
  plotcp.set("pipeline","inj-suffix",injectionSuffix)
  plotcp.set("pipeline","found-suffix",injectionSuffix)
  plotcp.set("pipeline","missed-suffix",injectionSuffix)
  plotcp.set("pipeline","bank-suffix",bankSuffix)
  plotcp.set("pipeline","trigbank-suffix",bankSuffix)
  plotcp.set("pipeline","zerolag-suffix",zerolagSuffix)
  plotcp.set("pipeline","trig-suffix",zerolagSuffix)
  plotcp.set("pipeline","coinc-suffix",zerolagSuffix)
  plotcp.set("pipeline","slide-suffix",slideSuffix)


  # set the user-tag
  if plotcp.get("pipeline","user-tag"):
    usertag = plotcp.get("pipeline","user-tag")
    plotcp.set("pipeline","input-user-tag",usertag)
    usertag += plotDir.upper() 
  else:
    usertag = plotDir.upper()
    plotcp.set("pipeline","input-user-tag","")

  if tag: usertag += "_" + tag
  plotcp.set("pipeline","user-tag",usertag)
  
  plotcp.set("common","cache-file",cacheFile)

  # return to the directory, write ini file and run hipe
  os.chdir(plotDir)
  iniFile = "plot_hipe_"
  iniFile += plotDir 
  if tag: iniFile += "_" + tag.lower()
  iniFile += ".ini"

  plotcp.write(file(iniFile,"w"))

  print "Running plot hipe in directory " + plotDir
  print "Using zero lag sieve: " + zerolagSuffix 
  print "Using time slide sieve: " + slideSuffix  
  print "Using injection sieve: " + injectionSuffix 
  print "Using bank sieve: " + bankSuffix 

  # work out the hipe call:
  plotCommand = config.get("condor","plot")
  plotCommand += " --log-path " + logPath
  plotCommand += " --config-file " + iniFile
  for item in config.items("ifo-details"):
      plotCommand += " --" + item[0] + " " + item[1]

  for item in config.items("plot-arguments"):
      plotCommand += " --" + item[0] + " " + item[1]

  if stage == "first" or stage == "both":
    plotCommand += " --first-stage"
  if stage == "second" or stage == "both":
    plotCommand += " --second-stage"
 
  # run lalapps_inspiral_hipe
  make_external_call(plotCommand)

  # make hipe job/node
  plotDag = iniFile.rstrip("ini") + usertag + ".dag"
  plotJob = pipeline.CondorDAGManJob(plotDir + "/" + plotDag)
  plotJob.add_opt("maxjobs", "2")
  plotNode = pipeline.CondorDAGNode(plotJob)

  # add postscript to deal with rescue dag
  fix_rescue(plotNode)

  # return to the original directory
  os.chdir("..")

  return plotNode


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
  followupSections = ["condor", "hipe-cache", "triggers", "datafind", \
      "q-datafind", "qscan", "q-hoft-datafind", "qscan-hoft", \
      "plots", "output", "seg"]

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
    DAG=${2##*/}
    mv ${2} ${2}.orig
    `sed "s+DIR ${DIR}++g" ${DAG}.rescue > ${2}`
    exit ${1}
  fi""")
  f.close()
  os.chmod("rescue.sh", 0744)

##############################################################################
# plot_hipe helpers

def determine_sieve_patterns(cp, plot_name, ifotag, usertag=None):
    """
    From the given plot configuration file, determine the --*-pattern values
    for a given plot name.  Returns as a dictionary of command-line option-
    value pairs.
    
    Example ini file:
    [pipeline]
    trig-suffix = PLAYGROUND
    bank-suffix = PLAYGROUND
    missed-suffix = INJ*
    
    [plotinspiral-meta]
    cache-patterns = trig,bank,missed
    trig-program-tag = SIRE
    bank-program-tag = TRIGBANK
    missed-program-tag = SIRE
    
    Example invocation:
    >>> from lalapps.inspiralutils import determine_sieve_patterns
    >>> from ConfigParser import ConfigParser
    >>> cp = ConfigParser()
    >>> cp.read("plot_hipe.ini")
    >>> print determine_sieve_patterns(cp, "plotinspiral", "H1")
    {'bank-pattern': 'TRIGBANK_H1*_PLAYGROUND', 'missed-pattern':
     'SIRE_H1*_INJ*', 'trig-pattern': 'SIRE_H1*_PLAYGROUND'}
    
    """
    meta_name = plot_name + "-meta"
    usertag = cp.get("pipeline", "input-user-tag") or usertag
    suffixes = dict([(opt[:-len("-suffix")],val) for opt,val in cp.items("pipeline") if opt.endswith("-suffix")])

    patterns = {}
    for pattern_name in cp.get(meta_name, "cache-patterns").split(","):
        program_tag = cp.get(meta_name, pattern_name + "-program-tag")
        pattern = program_tag + "_" + ifotag + "*"
        if usertag is not None:
            pattern += "_" + usertag
        suffix = suffixes.get(pattern_name)
        if suffix is not None:
            pattern += "_" + suffix
        patterns[pattern_name + "-pattern"] = pattern
    return patterns


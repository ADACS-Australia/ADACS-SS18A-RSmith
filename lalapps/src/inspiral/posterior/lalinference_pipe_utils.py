#flow DAG Class definitions for LALInference Pipeline
# (C) 2012 John Veitch, Kiersten Ruisard, Kan Wang

import itertools
import glue
from glue import pipeline,segmentsUtils,segments
import os
from lalapps import inspiralutils
import uuid
import ast
import pdb
import string
from math import floor,ceil,log,pow
import sys
import random

# We use the GLUE pipeline utilities to construct classes for each
# type of job. Each class has inputs and outputs, which are used to
# join together types of jobs into a DAG.

class Event():
  """
  Represents a unique event to run on
  """
  new_id=itertools.count().next
  def __init__(self,trig_time=None,SimInspiral=None,SnglInspiral=None,CoincInspiral=None,event_id=None,timeslide_dict=None,GID=None,ifos=None, duration=None,srate=None,trigSNR=None):
    self.trig_time=trig_time
    self.injection=SimInspiral
    self.sngltrigger=SnglInspiral
    if timeslide_dict is None:
      self.timeslides={}
    else:
      self.timeslides=timeslide_dict
    self.GID=GID
    self.coinctrigger=CoincInspiral
    if ifos is None:
      self.ifos = []
    else:
      self.ifos = ifos
    self.duration = duration
    self.srate = srate
    self.trigSNR = trigSNR
    if event_id is not None:
        self.event_id=event_id
    else:
        self.event_id=Event.new_id()
    if self.injection is not None:
        self.trig_time=self.injection.get_end()
        if event_id is None: self.event_id=int(str(self.injection.simulation_id).split(':')[2])
    if self.sngltrigger is not None:
        self.trig_time=self.sngltrigger.get_end()
        self.event_id=int(str(self.sngltrigger.event_id).split(':')[2])
    if self.coinctrigger is not None:
        self.trig_time=self.coinctrigger.end_time + 1.0e-9 * self.coinctrigger.end_time_ns
    if self.GID is not None:
        self.event_id=int(''.join(i for i in self.GID if i.isdigit()))

dummyCacheNames=['LALLIGO','LALVirgo','LALAdLIGO','LALAdVirgo']

def readLValert(lvalertfile,SNRthreshold=0,gid=None):
  """
  Parse LV alert file, continaing coinc, sngl, coinc_event_map.
  and create a list of Events as input for pipeline
  Based on Chris Pankow's script
  """ 
  output=[]
  from glue.ligolw import utils
  from glue.ligolw import lsctables
  from glue.ligolw import ligolw
  from glue.ligolw import param
  from glue.ligolw import array
  from pylal import series as lalseries
  import numpy as np
  xmldoc=utils.load_filename(lvalertfile)
  coinctable = lsctables.getTablesByType(xmldoc, lsctables.CoincInspiralTable)[0]
  coinc_events = [event for event in coinctable]
  sngltable = lsctables.getTablesByType(xmldoc, lsctables.SnglInspiralTable)[0]
  sngl_events = [event for event in sngltable]
  #Issues to identify IFO with good data that did not produce a trigger
  #search_summary = lsctables.getTablesByType(xmldoc, lsctables.SearchSummaryTable)[0]
  #ifos = search_summary[0].ifos.split(",")
  coinc_table = lsctables.getTablesByType(xmldoc, lsctables.CoincTable)[0]
  ifos = coinc_table[0].instruments.split(",")
  trigSNR = coinctable[0].snr
  # Parse PSD
  xmlpsd = utils.load_filename("psd.xml.gz")
  psddict = dict((param.get_pyvalue(elem, u"instrument"), lalseries.parse_REAL8FrequencySeries(elem)) for elem in xmlpsd.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"REAL8FrequencySeries")
  for instrument, psd in psddict.items():
    combine=[]
    for i,p in enumerate(psd.data):
      combine.append([psd.f0+i*psd.deltaF,np.sqrt(p)])
    np.savetxt(instrument+'psd.txt',combine)
  srate = combine[-1][0]
  
  # Logic for template duration and sample rate disabled
  coinc_map = lsctables.getTablesByType(xmldoc, lsctables.CoincMapTable)[0]
  for coinc in coinc_events:
    these_sngls = [e for e in sngl_events if e.event_id in [c.event_id for c in coinc_map if c.coinc_event_id == coinc.coinc_event_id] ]
    dur = min([e.template_duration for e in these_sngls]) + 2 # Add 2s padding
    #srate = pow(2.0, ceil( log(max([e.f_final]), 2) ) ) # Round up to power of 2
    ev=Event(CoincInspiral=coinc, GID=gid, ifos = ifos, duration = dur, srate = srate, trigSNR = trigSNR)
    if(coinc.snr>SNRthreshold): output.append(ev)
  
  print "Found %d coinc events in table." % len(coinc_events)
  return output

def open_pipedown_database(database_filename,tmp_space):
    """
    Open the connection to the pipedown database
    """
    if not os.access(database_filename,os.R_OK):
	raise Exception('Unable to open input file: %s'%(database_filename))
    from glue.ligolw import dbtables
    import sqlite3
    working_filename=dbtables.get_connection_filename(database_filename,tmp_path=tmp_space)
    connection = sqlite3.connect(working_filename)
    if tmp_space:
	dbtables.set_temp_store_directory(connection,tmp_space)
    dbtables.DBTable_set_connection(connection)
    return (connection,working_filename) 


def get_zerolag_pipedown(database_connection, dumpfile=None, gpsstart=None, gpsend=None, max_cfar=-1):
	"""
	Returns a list of Event objects
	from pipedown data base. Can dump some stats to dumpfile if given,
	and filter by gpsstart and gpsend to reduce the nunmber or specify
	max_cfar to select by combined FAR
	"""
	output={}
	if gpsstart is not None: gpsstart=float(gpsstart)
	if gpsend is not None: gpsend=float(gpsend)
	# Get coincs
	get_coincs = "SELECT sngl_inspiral.end_time+sngl_inspiral.end_time_ns*1e-9,sngl_inspiral.ifo,coinc_event.coinc_event_id,sngl_inspiral.snr,sngl_inspiral.chisq,coinc_inspiral.combined_far \
		FROM sngl_inspiral join coinc_event_map on (coinc_event_map.table_name=='sngl_inspiral' and coinc_event_map.event_id ==\
		sngl_inspiral.event_id) join coinc_event on (coinc_event.coinc_event_id==coinc_event_map.coinc_event_id) \
		join coinc_inspiral on (coinc_event.coinc_event_id==coinc_inspiral.coinc_event_id) \
		WHERE coinc_event.time_slide_id=='time_slide:time_slide_id:10049'\
		"
	if gpsstart is not None:
		get_coincs=get_coincs+' and sngl_inspiral.end_time+sngl_inspiral.end_time_ns*1.0e-9 > %f'%(gpsstart)
	if gpsend is not None:
		get_coincs=get_coincs+' and sngl_inspiral.end_time+sngl_inspiral.end_time_ns*1.0e-9 < %f'%(gpsend)
	if max_cfar !=-1:
		get_coincs=get_coincs+' and coinc_inspiral.combined_far < %f'%(max_cfar)
	db_out=database_connection.cursor().execute(get_coincs)
    	extra={}
	for (sngl_time, ifo, coinc_id, snr, chisq, cfar) in db_out:
      		coinc_id=int(coinc_id.split(":")[-1])
	  	if not coinc_id in output.keys():
			output[coinc_id]=Event(trig_time=sngl_time,timeslide_dict={},event_id=int(coinc_id))
			extra[coinc_id]={}
		output[coinc_id].timeslides[ifo]=0
		output[coinc_id].ifos.append(ifo)
		extra[coinc_id][ifo]={'snr':snr,'chisq':chisq,'cfar':cfar}
	if dumpfile is not None:
		fh=open(dumpfile,'w')
		for co in output.keys():
			for ifo in output[co].ifos:
				fh.write('%s %s %s %s %s %s %s\n'%(str(co),ifo,str(output[co].trig_time),str(output[co].timeslides[ifo]),str(extra[co][ifo]['snr']),str(extra[co][ifo]['chisq']),str(extra[co][ifo]['cfar'])))
		fh.close()
	return output.values()
	

def get_timeslides_pipedown(database_connection, dumpfile=None, gpsstart=None, gpsend=None, max_cfar=-1):
	"""
	Returns a list of Event objects
	with times and timeslide offsets
	"""
	output={}
	if gpsstart is not None: gpsstart=float(gpsstart)
	if gpsend is not None: gpsend=float(gpsend)
	db_segments=[]
	sql_seg_query="SELECT search_summary.out_start_time, search_summary.out_end_time from search_summary join process on process.process_id==search_summary.process_id where process.program=='thinca'"
	db_out = database_connection.cursor().execute(sql_seg_query)
	for d in db_out:
		if d not in db_segments:
			db_segments.append(d)
	seglist=segments.segmentlist([segments.segment(d[0],d[1]) for d in db_segments])
	db_out_saved=[]
	# Get coincidences
	get_coincs="SELECT sngl_inspiral.end_time+sngl_inspiral.end_time_ns*1e-9,time_slide.offset,sngl_inspiral.ifo,coinc_event.coinc_event_id,sngl_inspiral.snr,sngl_inspiral.chisq,coinc_inspiral.combined_far \
		    FROM sngl_inspiral join coinc_event_map on (coinc_event_map.table_name == 'sngl_inspiral' and coinc_event_map.event_id \
		    == sngl_inspiral.event_id) join coinc_event on (coinc_event.coinc_event_id==coinc_event_map.coinc_event_id) join time_slide\
		    on (time_slide.time_slide_id == coinc_event.time_slide_id and time_slide.instrument==sngl_inspiral.ifo)\
		    join coinc_inspiral on (coinc_inspiral.coinc_event_id==coinc_event.coinc_event_id) where coinc_event.time_slide_id!='time_slide:time_slide_id:10049'"
	joinstr = ' and '
	if gpsstart is not None:
		get_coincs=get_coincs+ ' where sngl_inspiral.end_time+sngl_inspiral.end_time_ns*1e-9 > %f'%(gpsstart)
	if gpsend is not None:
		get_coincs=get_coincs+ joinstr+' sngl_inspiral.end_time+sngl_inspiral.end_time*1e-9 <%f'%(gpsend)
	if max_cfar!=-1:
		get_coincs=get_coincs+joinstr+' coinc_inspiral.combined_far < %f'%(max_cfar)
	db_out=database_connection.cursor().execute(get_coincs)
        from pylal import SnglInspiralUtils
        extra={}
	for (sngl_time, slide, ifo, coinc_id, snr, chisq, cfar) in db_out:
          coinc_id=int(coinc_id.split(":")[-1])
	  seg=filter(lambda seg:sngl_time in seg,seglist)[0]
	  slid_time = SnglInspiralUtils.slideTimeOnRing(sngl_time,slide,seg)
	  if not coinc_id in output.keys():
	    output[coinc_id]=Event(trig_time=slid_time,timeslide_dict={},event_id=int(coinc_id))
            extra[coinc_id]={}
	  output[coinc_id].timeslides[ifo]=slid_time-sngl_time
	  output[coinc_id].ifos.append(ifo)
          extra[coinc_id][ifo]={'snr':snr,'chisq':chisq,'cfar':cfar}
	if dumpfile is not None:
		fh=open(dumpfile,'w')
		for co in output.keys():
			for ifo in output[co].ifos:
				fh.write('%s %s %s %s %s %s %s\n'%(str(co),ifo,str(output[co].trig_time),str(output[co].timeslides[ifo]),str(extra[co][ifo]['snr']),str(extra[co][ifo]['chisq']),str(extra[co][ifo]['cfar'])))
		fh.close()
	return output.values()

def mkdirs(path):
  """
  Helper function. Make the given directory, creating intermediate
  dirs if necessary, and don't complain about it already existing.
  """
  if os.access(path,os.W_OK) and os.path.isdir(path): return
  else: os.makedirs(path)

def chooseEngineNode(name):
  if name=='lalinferencenest':
    return LALInferenceNestNode
  if name=='lalinferencemcmc':
    return LALInferenceMCMCNode
  return EngineNode

def scan_timefile(timefile):
    import re
    p=re.compile('[\d.]+')
    times=[]
    timefilehandle=open(timefile,'r')
    for time in timefilehandle:
      if not p.match(time):
	continue
      if float(time) in times:
	print 'Skipping duplicate time %s'%(time)
	continue
      print 'Read time %s'%(time)
      times.append(float(time))
    timefilehandle.close()
    return times
  
class LALInferencePipelineDAG(pipeline.CondorDAG):
  def __init__(self,cp,dax=False):
    self.subfiles=[]
    self.config=cp
    self.engine=cp.get('analysis','engine')
    self.EngineNode=chooseEngineNode(self.engine)
    if cp.has_option('paths','basedir'):
      self.basepath=cp.get('paths','basedir')
    else:
      self.basepath=os.getcwd()
      print 'No basepath specified, using current directory: %s'%(self.basepath)
    mkdirs(self.basepath)
    self.posteriorpath=os.path.join(self.basepath,'posterior_samples')
    mkdirs(self.posteriorpath)
    daglogdir=cp.get('paths','daglogdir')
    mkdirs(daglogdir)
    self.daglogfile=os.path.join(daglogdir,'lalinference_pipeline-'+str(uuid.uuid1())+'.log')
    pipeline.CondorDAG.__init__(self,self.daglogfile,dax)
    if cp.has_option('paths','cachedir'):
      self.cachepath=cp.get('paths','cachedir')
    else:
      self.cachepath=os.path.join(self.basepath,'caches')
    mkdirs(self.cachepath)
    if cp.has_option('paths','logdir'):
      self.logpath=cp.get('paths','logdir')
    else:
      self.logpath=os.path.join(self.basepath,'log')
    mkdirs(self.logpath)
    if cp.has_option('analysis','ifos'):
      self.ifos=ast.literal_eval(cp.get('analysis','ifos'))
    else:
      self.ifos=['H1','L1','V1']
    self.segments={}
    if cp.has_option('datafind','veto-categories'):
      self.veto_categories=cp.get('datafind','veto-categories')
    else: self.veto_categories=[]
    for ifo in self.ifos:
      self.segments[ifo]=[]
    self.dq={}
    self.frtypes=ast.literal_eval(cp.get('datafind','types'))
    self.channels=ast.literal_eval(cp.get('data','channels'))
    self.use_available_data=False
    self.webdir=cp.get('paths','webdir')
    if cp.has_option('analysis','dataseed'):
      self.dataseed=cp.getint('analysis','dataseed')
    else:
      self.dataseed=None
    # Set up necessary job files.
    self.datafind_job = pipeline.LSCDataFindJob(self.cachepath,self.logpath,self.config)
    self.datafind_job.add_opt('url-type','file')
    self.datafind_job.set_sub_file(os.path.join(self.basepath,'datafind.sub'))
    self.engine_job = EngineJob(self.config, os.path.join(self.basepath,'lalinference.sub'),self.logpath)
    self.results_page_job = ResultsPageJob(self.config,os.path.join(self.basepath,'resultspage.sub'),self.logpath)
    self.merge_job = MergeNSJob(self.config,os.path.join(self.basepath,'merge_runs.sub'),self.logpath)
    self.coherence_test_job = CoherenceTestJob(self.config,os.path.join(self.basepath,'coherence_test.sub'),self.logpath)
    self.gracedbjob = GraceDBJob(self.config,os.path.join(self.basepath,'gracedb.sub'),self.logpath)
    # Process the input to build list of analyses to do
    self.events=self.setup_from_inputs()
    self.times=[e.trig_time for e in self.events]

    # Sanity checking
    if len(self.events)==0:
      print 'No input events found, please check your config. Will generate an empty DAG'
    
    # Set up the segments
    (mintime,maxtime)=self.get_required_data(self.times)
    if not self.config.has_option('input','gps-start-time'):
      self.config.set('input','gps-start-time',str(int(floor(mintime))))
    if not self.config.has_option('input','gps-end-time'):
      self.config.set('input','gps-end-time',str(int(ceil(maxtime))))
    self.add_science_segments()
    
    # Save the final configuration that is being used
    conffilename=os.path.join(self.basepath,'config.ini')
    with open(conffilename,'wb') as conffile:
      self.config.write(conffile)
    
    # Generate the DAG according to the config given
    if self.engine=='lalinferencenest':
      for event in self.events: self.add_full_analysis_lalinferencenest(event)
    elif self.engine=='lalinferencemcmc':
      for event in self.events: self.add_full_analysis_lalinferencemcmc(event)

    self.dagfilename="lalinference_%s-%s"%(self.config.get('input','gps-start-time'),self.config.get('input','gps-end-time'))
    self.set_dag_file(os.path.join(self.basepath,self.dagfilename))
    
  def get_required_data(self,times):
    """
    Calculate the data that will be needed to process all events
    """
    psdlength=self.config.getint('input','max-psd-length')
    padding=self.config.getint('input','padding')
    seglen = self.config.getint('lalinference','seglen')
    # Assume that the data interval is (end_time - seglen -padding , end_time + psdlength +padding )
    # Also require padding before start time
    return (min(times)-padding-seglen,max(times)+padding+psdlength)

  def setup_from_times(self,times):
    """
    Generate a DAG from a list of times
    """
    for time in self.times:
      self.add_full_analysis_lalinferencenest(Event(trig_time=time))
      
  def select_events(self):
    """
    Read events from the config parser. Understands both ranges and comma separated events, or combinations
    eg. events=[0,1,5:10,21] adds to the analysis the events: 0,1,5,6,7,8,9,10 and 21
    """
    events=[]
    times=[]
    raw_events=self.config.get('input','events').replace('[','').replace(']','').split(',')
    for raw_event in raw_events:
        if ':' in raw_event:
            limits=raw_event.split(':')
            if len(limits) != 2:
                print "Error: in event config option; ':' must separate two numbers."
                exit(0)
            low=int(limits[0])
            high=int(limits[1])
            if low>high:
                events.extend(range(int(high),int(low)+1))
            elif high>low:
                events.extend(range(int(low),int(high)+1))
        else:
            events.append(int(raw_event))
    return events

  def setup_from_inputs(self):
    """
    Scan the list of inputs, i.e.
    gps-time-file, injection-file, sngl-inspiral-file, coinc-inspiral-file, pipedown-database
    in the [input] section of the ini file.
    And process the events found therein
    """
    gpsstart=None
    gpsend=None
    inputnames=['gps-time-file','injection-file','sngl-inspiral-file','coinc-inspiral-file','pipedown-db','lvalert-file']
    if sum([ 1 if self.config.has_option('input',name) else 0 for name in inputnames])!=1:
        print 'Plese specify only one input file'
        sys.exit(1)
    if self.config.has_option('input','events'):
      selected_events=self.config.get('input','events')
      print 'Selected events %s'%(str(selected_events))
      
      if selected_events=='all':
          selected_events=None
      else:
          selected_events=self.select_events()
    else:
        selected_events=None
    if self.config.has_option('input','gps-start-time'):
      gpsstart=self.config.getfloat('input','gps-start-time')
    if self.config.has_option('input','gps-end-time'):
      gpsend=self.config.getfloat('input','gps-end-time')
    # ASCII list of GPS times
    if self.config.has_option('input','gps-time-file'):
      times=scan_timefile(self.config.get('input','gps-time-file'))
      events=[Event(trig_time=time) for time in times]
    # Siminspiral Table
    if self.config.has_option('input','injection-file'):
      from pylal import SimInspiralUtils
      injTable=SimInspiralUtils.ReadSimInspiralFromFiles([self.config.get('input','injection-file')])
      events=[Event(SimInspiral=inj) for inj in injTable]
    # SnglInspiral Table
    if self.config.has_option('input','sngl-inspiral-file'):
      from pylal import SnglInspiralUtils
      trigTable=SnglInspiralUtils.ReadSnglInspiralFromFiles([self.config.get('input','sngl-inspiral-file')])
      events=[Event(SnglInspiral=trig) for trig in trigTable]
    if self.config.has_option('input','coinc-inspiral-file'):
      from pylal import CoincInspiralUtils
      coincTable = CoincInspiralUtils.readCoincInspiralFromFiles([self.config.get('input','coinc-inspiral-file')])
      events = [Event(CoincInspiral=coinc) for coinc in coincTable]
    # LVAlert CoincInspiral Table
    if self.config.has_option('input','gid'): gid=self.config.get('input','gid')
    else: gid=None
    if self.config.has_option('input','lvalert-file'):
      events = readLValert(self.config.get('input','lvalert-file'),gid=gid)
    # pipedown-database
    if self.config.has_option('input','pipedown-db'):
      db_connection = open_pipedown_database(self.config.get('input','pipedown-db'),None)[0]
      # Timeslides
      if self.config.has_option('input','time-slide-dump'):
        timeslidedump=self.config.get('input','time-slide-dump')
      else:
        timeslidedump=None
      if self.config.has_option('input','max-cfar'):
	maxcfar=self.config.getfloat('input','max-cfar')
      else:
	maxcfar=-1
      if self.config.get('input','timeslides').lower()=='true':
	events=get_timeslides_pipedown(db_connection, gpsstart=gpsstart, gpsend=gpsend,dumpfile=timeslidedump,max_cfar=maxcfar)
      else:
	events=get_zerolag_pipedown(db_connection, gpsstart=gpsstart, gpsend=gpsend, dumpfile=timeslidedump,max_cfar=maxcfar)
    if(selected_events is not None):
        used_events=[]
        for i in selected_events:
            e=events[i]
            e.event_id=i
            used_events.append(e)
        events=used_events
    return events

  def add_full_analysis_lalinferencenest(self,event):
    """
    Generate an end-to-end analysis of a given event (Event class)
    For LALinferenceNest code. Uses parallel runs if specified
    """
    evstring=str(event.event_id)
    if event.trig_time is not None:
        evstring=str(event.trig_time)+'-'+str(event.event_id)
    Npar=self.config.getint('analysis','nparallel')
    # Set up the parallel engine nodes
    enginenodes=[]
    for i in range(Npar):
      n=self.add_engine_node(event)
      if n is not None: enginenodes.append(n)
    if len(enginenodes)==0:
      return False
    myifos=enginenodes[0].get_ifos()
    # Merge the results together
    pagedir=os.path.join(self.webdir,evstring,myifos)
    mkdirs(pagedir)
    mergenode=MergeNSNode(self.merge_job,parents=enginenodes)
    mergenode.set_pos_output_file(os.path.join(self.posteriorpath,'posterior_%s_%s.dat'%(myifos,evstring)))
    self.add_node(mergenode)
    respagenode=self.add_results_page_node(outdir=pagedir,parent=mergenode)
    # Call finalize to build final list of available data
    enginenodes[0].finalize()
    respagenode.set_bayes_coherent_noise(mergenode.get_B_file())
    if self.config.has_option('input','injection-file') and event.event_id is not None:
        respagenode.set_injection(self.config.get('input','injection-file'),event.event_id)
    if event.GID is not None:
        self.add_gracedb_log_node(respagenode,event.GID)
    if self.config.getboolean('analysis','coherence-test') and len(enginenodes[0].ifos)>1:
        mkdirs(os.path.join(self.basepath,'coherence_test'))
        par_mergenodes=[]
        for ifo in enginenodes[0].ifos:
            cotest_nodes=[self.add_engine_node(event,ifos=[ifo]) for i in range(Npar)]
            for co in cotest_nodes:
              co.set_psdstart(enginenodes[0].GPSstart)
              co.set_psdlength(enginenodes[0].psdlength)
            pmergenode=MergeNSNode(self.merge_job,parents=cotest_nodes)
            pmergenode.set_pos_output_file(os.path.join(self.posteriorpath,'posterior_%s_%s.dat'%(ifo,evstring)))
            self.add_node(pmergenode)
            par_mergenodes.append(pmergenode)
            presultsdir=os.path.join(pagedir,ifo)
            mkdirs(presultsdir)
            subresnode=self.add_results_page_node(outdir=presultsdir,parent=pmergenode)
            subresnode.set_bayes_coherent_noise(pmergenode.get_B_file())
            if self.config.has_option('input','injection-file') and event.event_id is not None:
                subresnode.set_injection(self.config.get('input','injection-file'),event.event_id)
        coherence_node=CoherenceTestNode(self.coherence_test_job,outfile=os.path.join(self.basepath,'coherence_test','coherence_test_%s_%s.dat'%(myifos,evstring)))
        coherence_node.add_coherent_parent(mergenode)
        map(coherence_node.add_incoherent_parent, par_mergenodes)
        self.add_node(coherence_node)
        respagenode.add_parent(coherence_node)
        respagenode.set_bayes_coherent_incoherent(coherence_node.get_output_files()[0])
    return True
	
  def add_full_analysis_lalinferencemcmc(self,event):
    """
    Generate an end-to-end analysis of a given event
    For LALInferenceMCMC.
    """
    evstring=str(event.event_id)
    if event.trig_time is not None:
        evstring=str(event.trig_time)+'-'+str(event.event_id)
    Npar=self.config.getint('analysis','nparallel')
    enginenodes=[]
    for i in range(Npar):
        enginenodes.append(self.add_engine_node(event))
    myifos=enginenodes[0].get_ifos()
    pagedir=os.path.join(self.webdir,evstring,myifos)
    mkdirs(pagedir)
    respagenode=self.add_results_page_node(outdir=pagedir)
    map(respagenode.add_engine_parent, enginenodes)
    if event.GID is not None:
        self.add_gracedb_log_node(respagenode,event.GID)

  def add_science_segments(self):
    # Query the segment database for science segments and
    # add them to the pool of segments
    if self.config.has_option('input','ignore-science-segments'):
      if self.config.getboolean('input','ignore-science-segments'):
        start=self.config.getfloat('input','gps-start-time')
        end=self.config.getfloat('input','gps-end-time')
        i=0
        for ifo in self.ifos:
          sciseg=pipeline.ScienceSegment((i,start,end,end-start))
          df_node=self.get_datafind_node(ifo,self.frtypes[ifo],int(sciseg.start()),int(sciseg.end()))
          sciseg.set_df_node(df_node)
          self.segments[ifo].append(sciseg)
          i+=1
        return
    # Look up science segments as required
    segmentdir=os.path.join(self.basepath,'segments')
    mkdirs(segmentdir)
    curdir=os.getcwd()
    os.chdir(segmentdir)
    for ifo in self.ifos:
      (segFileName,dqVetoes)=inspiralutils.findSegmentsToAnalyze(self.config, ifo, self.veto_categories, generate_segments=True,\
	use_available_data=self.use_available_data , data_quality_vetoes=False)
      self.dqVetoes=dqVetoes
      segfile=open(segFileName)
      segs=segmentsUtils.fromsegwizard(segfile)
      segs.coalesce()
      segfile.close()
      for seg in segs:
	    sciseg=pipeline.ScienceSegment((segs.index(seg),seg[0],seg[1],seg[1]-seg[0]))
	    df_node=self.get_datafind_node(ifo,self.frtypes[ifo],int(sciseg.start()),int(sciseg.end()))
	    sciseg.set_df_node(df_node)
	    self.segments[ifo].append(sciseg)
    os.chdir(curdir)
  
  def get_datafind_node(self,ifo,frtype,gpsstart,gpsend):
    node=pipeline.LSCDataFindNode(self.datafind_job)
    node.set_observatory(ifo[0])
    node.set_type(frtype)
    node.set_start(gpsstart)
    node.set_end(gpsend)
    #self.add_node(node)
    return node
  
  def add_engine_node(self,event,ifos=None,extra_options=None):
    """
    Add an engine node to the dag. Will find the appropriate cache files automatically.
    Will determine the data to be read and the output file.
    Will use all IFOs known to the DAG, unless otherwise specified as a list of strings
    """
    if ifos is None and len(event.ifos)>0:
      ifos=event.ifos
    if ifos is None:
      ifos=self.ifos
    node=self.EngineNode(self.engine_job)
    end_time=event.trig_time
    node.set_trig_time(end_time)
    node.set_seed(random.randint(1,2**31))
    if event.srate: node.set_srate(event.srate)
    if event.trigSNR: node.set_trigSNR(event.trigSNR)
    if self.dataseed:
      node.set_dataseed(self.dataseed+event.event_id)
    gotdata=0
    for ifo in ifos:
      if event.timeslides.has_key(ifo):
        slide=event.timeslides[ifo]
      else:
        slide=0
      for seg in self.segments[ifo]:
        if end_time >= seg.start() and end_time < seg.end():
          gotdata+=node.add_ifo_data(ifo,seg,self.channels[ifo],timeslide=slide)
    if self.config.has_option('lalinference','fake-cache'):
      node.cachefiles=ast.literal_eval(self.config.get('lalinference','fake-cache'))
      node.channels=ast.literal_eval(self.config.get('data','channels'))
      if len(ifos)==0: node.ifos=node.cachefiles.keys()
      else: node.ifos=ifos
      node.timeslides=dict([ (ifo,0) for ifo in node.ifos])
      gotdata=1
    if self.config.has_option('lalinference','flow'):
      node.flows=ast.literal_eval(self.config.get('lalinference','flow'))
    if self.config.has_option('lalinference','ER2-cache'):
      node.cachefiles=ast.literal_eval(self.config.get('lalinference','ER2-cache'))
      node.channels=ast.literal_eval(self.config.get('data','channels'))
      node.psds=ast.literal_eval(self.config.get('lalinference','psds'))
      for ifo in ifos:
        node.add_input_file(os.path.join(self.basepath,node.cachefiles[ifo]))
        node.cachefiles[ifo]=os.path.join(self.basepath,node.cachefiles[ifo])
        node.add_input_file(os.path.join(self.basepath,node.psds[ifo]))
        node.psds[ifo]=os.path.join(self.basepath,node.psds[ifo])
      if len(ifos)==0: node.ifos=node.cachefiles.keys()
      else: node.ifos=ifos
      node.timeslides=dict([ (ifo,0) for ifo in node.ifos])
      gotdata=1
    else:
      # Add the nodes it depends on
      for seg in node.scisegs.values():
        dfnode=seg.get_df_node()
        if dfnode is not None and dfnode not in self.get_nodes():
    	  self.add_node(dfnode)
    if gotdata:
      self.add_node(node)
    else:
      'Print no data found for time %f'%(end_time)
      return None
    if extra_options is not None:
      for opt in extra_options.keys():
	    node.add_var_arg('--'+opt+' '+extra_options[opt])
    # Add control options
    if self.config.has_option('input','injection-file'):
       node.set_injection(self.config.get('input','injection-file'),event.event_id)
    node.set_seglen(self.config.getint('lalinference','seglen'))
    if self.config.has_option('input','psd-length'):
      node.set_psdlength(self.config.getint('input','psd-length'))
    if self.config.has_option('input','psd-start-time'):
      node.set_psdstart(self.config.getfloat('input','psd-start-time'))
    node.set_max_psdlength(self.config.getint('input','max-psd-length'))
    out_dir=os.path.join(self.basepath,'engine')
    mkdirs(out_dir)
    node.set_output_file(os.path.join(out_dir,node.engine+'-'+str(event.event_id)+'-'+node.get_ifos()+'-'+str(node.get_trig_time())+'-'+str(node.id)))
    return node
    
  def add_results_page_node(self,outdir=None,parent=None,extra_options=None):
    node=ResultsPageNode(self.results_page_job)
    if parent is not None:
      node.add_parent(parent)
      infile=parent.get_pos_file()
      node.add_file_arg(infile)
    node.set_output_path(outdir)
    self.add_node(node)
    return node

  def add_gracedb_log_node(self,respagenode,gid):
    node=GraceDBNode(self.gracedbjob,parent=respagenode,gid=gid)
    node.add_parent(respagenode)
    self.add_node(node)
    return node

class EngineJob(pipeline.CondorDAGJob):
  def __init__(self,cp,submitFile,logdir):
    self.engine=cp.get('analysis','engine')
    if self.engine=='lalinferencemcmc':
      exe=cp.get('condor','mpirun')
      self.binary=cp.get('condor',self.engine)
      #universe="parallel"
      universe="vanilla"
      self.write_sub_file=self.__write_sub_file_mcmc_mpi
    else:
      exe=cp.get('condor',self.engine)
      universe="standard"
    pipeline.CondorDAGJob.__init__(self,universe,exe)
    # Set the options which are always used
    self.set_sub_file(submitFile)
    if self.engine=='lalinferencemcmc':
      #openmpipath=cp.get('condor','openmpi')
      machine_count=cp.get('mpi','machine-count')
      #self.add_condor_cmd('machine_count',machine_count)
      #self.add_condor_cmd('environment','CONDOR_MPI_PATH=%s'%(openmpipath))
      self.add_condor_cmd('Requirements','CAN_RUN_MULTICORE')
      self.add_condor_cmd('+RequiresMultipleCores','True')
      self.add_condor_cmd('request_cpus',machine_count)
      self.add_condor_cmd('getenv','true')
      
    self.add_ini_opts(cp,self.engine)
    self.set_stdout_file(os.path.join(logdir,'lalinference-$(cluster)-$(process)-$(node).out'))
    self.set_stderr_file(os.path.join(logdir,'lalinference-$(cluster)-$(process)-$(node).err'))
  
  def __write_sub_file_mcmc_mpi(self):
    """
    Nasty hack to insert the MPI stuff into the arguments
    when write_sub_file is called
    """
    # First write the standard sub file
    pipeline.CondorDAGJob.write_sub_file(self)
    # Then read it back in to mangle the arguments line
    outstring=""
    MPIextraargs= ' -np 8 '+self.binary+' -- ' #'--verbose --stdout cluster$(CLUSTER).proc$(PROCESS).mpiout --stderr cluster$(CLUSTER).proc$(PROCESS).mpierr '+self.binary+' -- '
    subfilepath=self.get_sub_file()
    subfile=open(subfilepath,'r')
    for line in subfile:
       if line.startswith('arguments = "'):
          print 'Hacking sub file for MPI'
          line=line.replace('arguments = "','arguments = "'+MPIextraargs,1)
       outstring=outstring+line
    subfile.close()
    subfile=open(subfilepath,'w')
    subfile.write(outstring)
    subfile.close()
      
 
class EngineNode(pipeline.CondorDAGNode):
  new_id = itertools.count().next
  def __init__(self,li_job):
    pipeline.CondorDAGNode.__init__(self,li_job)
    self.ifos=[]
    self.scisegs={}
    self.channels={}
    self.psds={}
    self.flows={}
    self.timeslides={}
    self.seglen=None
    self.psdlength=None
    self.maxlength=None
    self.psdstart=None
    self.cachefiles={}
    self.id=EngineNode.new_id()
    self.__finaldata=False

  def set_seglen(self,seglen):
    self.seglen=seglen

  def set_psdlength(self,psdlength):
    self.psdlength=psdlength

  def set_max_psdlength(self,psdlength):
    self.maxlength=psdlength

  def set_psdstart(self,psdstart):
    self.psdstart=psdstart

  def set_seed(self,seed):
    self.add_var_opt('randomseed',str(seed))
  
  def set_srate(self,srate):
    self.add_var_opt('srate',str(srate))

  def set_trigSNR(self,trigSNR):
    self.add_var_opt('trigSNR',str(trigSNR))

  def set_dataseed(self,seed):
    self.add_var_opt('dataseed',str(seed))

  def get_ifos(self):
    return ''.join(map(str,self.ifos))
      
  def set_trig_time(self,time):
    """
    Set the end time of the signal for the centre of the prior in time
    """
    self.__trigtime=float(time)
    self.add_var_opt('trigtime',str(time))
    
  def set_event_number(self,event):
    """
    Set the event number in the injection XML.
    """
    if event is not None:
      self.__event=int(event)
      self.add_var_opt('event',str(event))
  
  def set_injection(self,injfile,event):
    """
    Set a software injection to be performed.
    """
    self.add_file_opt('inj',injfile)
    self.set_event_number(event)

  def get_trig_time(self): return self.__trigtime
  
  def add_ifo_data(self,ifo,sciseg,channelname,timeslide=0):
    self.ifos.append(ifo)
    self.scisegs[ifo]=sciseg
    parent=sciseg.get_df_node()
    if parent is not None:
        self.add_parent(parent)
        self.cachefiles[ifo]=parent.get_output_files()[0]
        self.add_input_file(self.cachefiles[ifo])
        self.timeslides[ifo]=timeslide
        self.channels[ifo]=channelname
        return 1
    else: return 0
  
  def finalize(self):
    if not self.__finaldata:
      self._finalize_ifo_data()
    pipeline.CondorDAGNode.finalize(self)
    
  def _finalize_ifo_data(self):
      """
      Add final list of IFOs and data to analyse to command line arguments.
      """
      ifostring='['
      cachestring='['
      psdstring='['
      flowstring='['
      channelstring='['
      slidestring='['
      first=True
      for ifo in self.ifos:
        if first:
          delim=''
          first=False
        else: delim=','
        ifostring=ifostring+delim+ifo
        cachestring=cachestring+delim+self.cachefiles[ifo]
        if self.psds: psdstring=psdstring+delim+self.psds[ifo]
        if self.flows: flowstring=flowstring+delim+self.flows[ifo]
        channelstring=channelstring+delim+self.channels[ifo]
        slidestring=slidestring+delim+str(self.timeslides[ifo])
      ifostring=ifostring+']'
      cachestring=cachestring+']'
      psdstring=psdstring+']'
      flowstring=flowstring+']'
      channelstring=channelstring+']'
      slidestring=slidestring+']'
      self.add_var_opt('ifo',ifostring)
      self.add_var_opt('channel',channelstring)
      self.add_var_opt('cache',cachestring)
      if self.psds: self.add_var_opt('psd',psdstring)
      if self.flows: self.add_var_opt('flow',flowstring)
      if any(self.timeslides):
	self.add_var_opt('timeslide',slidestring)
      # Start at earliest common time
      # NOTE: We perform this arithmetic for all ifos to ensure that a common data set is
      # Used when we are running the coherence test.
      # Otherwise the noise evidence will differ.
      if self.scisegs!={}:
        starttime=max([int(self.scisegs[ifo].start()) for ifo in self.ifos])
        endtime=min([int(self.scisegs[ifo].end()) for ifo in self.ifos])
      else:
        starttime=self.get_trig_time()-0.5*self.maxlength
        endtime=starttime+self.maxlength
      self.GPSstart=starttime
      self.__GPSend=endtime
      length=endtime-starttime
      
      # Now we need to adjust the start time and length to make sure the maximum data length
      # is not exceeded.
      trig_time=self.get_trig_time()
      maxLength=self.maxlength
      if(length > maxLength):
        while(self.GPSstart+maxLength<trig_time and self.GPSstart+maxLength<self.__GPSend):
          self.GPSstart+=maxLength/2.0
      # Override calculated start time if requested by user in ini file
      if self.psdstart is not None:
        self.GPSstart=self.psdstart
        print 'Over-riding start time to user-specified value %f'%(self.GPSstart)
        #if self.GPSstart<starttime or self.GPSstart>endtime:
        #  print 'ERROR: Over-ridden time lies outside of science segment!'
        #  raise Exception('Bad psdstart specified') 
      self.add_var_opt('psdstart',str(self.GPSstart))
      if self.psdlength is None:
        self.psdlength=self.__GPSend-self.GPSstart
        if(self.psdlength>self.maxlength):
          self.psdlength=self.maxlength
      self.add_var_opt('psdlength',self.psdlength)
      self.add_var_opt('seglen',self.seglen)
      self.__finaldata=True

class LALInferenceNestNode(EngineNode):
  def __init__(self,li_job):
    EngineNode.__init__(self,li_job)
    self.engine='lalinferencenest'
    self.outfilearg='outfile'
    
  def set_output_file(self,filename):
    self.nsfile=filename+'.dat'
    self.add_file_opt(self.outfilearg,self.nsfile,file_is_output_file=True)
    self.paramsfile=self.nsfile+'_params.txt'
    self.Bfilename=self.nsfile+'_B.txt'
    self.headerfile=self.nsfile+'_params.txt'

  def get_B_file(self):
    return self.Bfilename

  def get_ns_file(self):
    return self.nsfile

  def get_header_file(self):
    return self.headerfile

class LALInferenceMCMCNode(EngineNode):
  def __init__(self,li_job):
    EngineNode.__init__(self,li_job)
    self.engine='lalinferencemcmc'
    self.outfilearg='outfile'

  def set_output_file(self,filename):
    self.posfile=filename+'.00'
    self.add_file_opt(self.outfilearg,filename)
    self.add_output_file(self.posfile)

  def get_pos_file(self):
    return self.posfile

class ResultsPageJob(pipeline.CondorDAGJob):
  def __init__(self,cp,submitFile,logdir):
    exe=cp.get('condor','resultspage')
    pipeline.CondorDAGJob.__init__(self,"vanilla",exe)
    self.set_sub_file(submitFile)
    self.set_stdout_file(os.path.join(logdir,'resultspage-$(cluster)-$(process).out'))
    self.set_stderr_file(os.path.join(logdir,'resultspage-$(cluster)-$(process).err'))
    self.add_condor_cmd('getenv','True')
    self.add_condor_cmd('RequestMemory','2000')
    self.add_ini_opts(cp,'resultspage')
    # self.add_opt('Nlive',cp.get('analysis','nlive'))
    
    if cp.has_option('results','skyres'):
        self.add_opt('skyres',cp.get('results','skyres'))

class ResultsPageNode(pipeline.CondorDAGNode):
    def __init__(self,results_page_job,outpath=None):
        pipeline.CondorDAGNode.__init__(self,results_page_job)
        if outpath is not None:
            self.set_output_path(path)
    def set_output_path(self,path):
        self.webpath=path
        self.add_var_opt('outpath',path)
        mkdirs(path)
        self.posfile=os.path.join(path,'posterior_samples.dat')
    def set_injection(self,injfile,eventnumber):
        self.injfile=injfile
        self.add_var_arg('--inj '+injfile)
        self.set_event_number(eventnumber)
    def set_event_number(self,event):
        """
        Set the event number in the injection XML.
        """
        if event is not None:
            self.__event=int(event)
            self.add_var_arg('--eventnum '+str(event))
    def add_engine_parent(self,node):
      """
      Add a parent node which is one of the engine nodes
      And automatically set options accordingly
      """
      self.add_parent(node)
      self.add_file_arg(node.get_pos_file())
	
      if isinstance(node,LALInferenceMCMCNode):
	    self.add_var_opt('lalinfmcmc','')
    def get_pos_file(self): return self.posfile
    def set_bayes_coherent_incoherent(self,bcifile):
        self.add_var_arg('--bci '+bcifile)
    def set_bayes_coherent_noise(self,bsnfile):
        self.add_var_arg('--bsn '+bsnfile)
        
class CoherenceTestJob(pipeline.CondorDAGJob):
    """
    Class defining the coherence test job to be run as part of a pipeline.
    """
    def __init__(self,cp,submitFile,logdir):
      exe=cp.get('condor','coherencetest')
      pipeline.CondorDAGJob.__init__(self,"vanilla",exe)
      self.add_opt('coherent-incoherent','')
      self.add_condor_cmd('getenv','True')
      self.set_stdout_file(os.path.join(logdir,'coherencetest-$(cluster)-$(process).out'))
      self.set_stderr_file(os.path.join(logdir,'coherencetest-$(cluster)-$(process).err'))
      self.set_sub_file(submitFile)

class CoherenceTestNode(pipeline.CondorDAGNode):
    """
    Class defining the node for the coherence test
    """
    def __init__(self,coherencetest_job,outfile=None):
      pipeline.CondorDAGNode.__init__(self,coherencetest_job)
      self.incoherent_parents=[]
      self.coherent_parent=None
      self.finalized=False
      if outfile is not None:
        self.add_file_opt('outfile',outfile,file_is_output_file=True)

    def add_coherent_parent(self,node):
      """
      Add a parent node which is an engine node, and process its outputfiles
      """
      self.coherent_parent=node
      self.add_parent(node)
    def add_incoherent_parent(self,node):
      """
      Add a parent node which provides one of the single-ifo evidence values
      """
      self.incoherent_parents.append(node)
      self.add_parent(node)
    def finalize(self):
      """
      Construct command line
      """
      if self.finalized==True: return
      self.finalized=True
      self.add_file_arg(self.coherent_parent.get_B_file())
      for inco in self.incoherent_parents:
        self.add_file_arg(inco.get_B_file())

class MergeNSJob(pipeline.CondorDAGJob):
    """
    Class defining a job which merges several parallel nested sampling jobs into a single file
    Input arguments:
    cp        - A configparser object containing the setup of the analysis
    submitFile    - Path to store the submit file
    logdir        - A directory to hold the stderr, stdout files of the merge runs
    """
    def __init__(self,cp,submitFile,logdir):
      exe=cp.get('condor','mergescript')
      pipeline.CondorDAGJob.__init__(self,"vanilla",exe)
      self.set_sub_file(submitFile)
      self.set_stdout_file(os.path.join(logdir,'merge-$(cluster)-$(process).out'))
      self.set_stderr_file(os.path.join(logdir,'merge-$(cluster)-$(process).err'))
      self.add_opt('Nlive',cp.get('lalinferencenest','nlive'))
      self.add_condor_cmd('getenv','True')
      if cp.has_option('merge','npos'):
      	self.add_opt('npos',cp.get('merge','npos'))


class MergeNSNode(pipeline.CondorDAGNode):
    """
    Class defining the DAG node for a merge job
    Input arguments:
    merge_job = A MergeJob object
    parents = iterable of parent LALInferenceNest nodes (must have get_ns_file() method)
    """
    def __init__(self,merge_job,parents=None):
        pipeline.CondorDAGNode.__init__(self,merge_job)
        if parents is not None:
          for parent in parents:
            self.add_engine_parent(parent)

    def add_engine_parent(self,parent):
        self.add_parent(parent)
        self.add_file_arg(parent.get_ns_file())
        self.add_file_opt('headers',parent.get_header_file())

    def set_pos_output_file(self,file):
        self.add_file_opt('pos',file,file_is_output_file=True)
        self.posfile=file
    
    def get_pos_file(self): return self.posfile
    def get_B_file(self): return self.posfile+'_B.txt'

class GraceDBJob(pipeline.CondorDAGJob):
    """
    Class for a gracedb job
    """
    def __init__(self,cp,submitFile,logdir):
      exe=cp.get('condor','gracedb')
      #pipeline.CondorDAGJob.__init__(self,"vanilla",exe)
      pipeline.CondorDAGJob.__init__(self,"scheduler",exe)
      self.set_sub_file(submitFile)
      self.set_stdout_file(os.path.join(logdir,'gracedb-$(cluster)-$(process).out'))
      self.set_stderr_file(os.path.join(logdir,'gracedb-$(cluster)-$(process).err'))
      self.add_condor_cmd('getenv','True')
      self.baseurl=cp.get('paths','baseurl')
      self.basepath=cp.get('paths','webdir')

class GraceDBNode(pipeline.CondorDAGNode):
    """
    Run the gracedb executable to report the results
    """
    def __init__(self,gracedb_job,gid=None,parent=None):
        pipeline.CondorDAGNode.__init__(self,gracedb_job)
        self.resultsurl=""
        if gid: self.set_gid(gid)
        if parent: self.set_parent_resultspage(parent,gid)
        self.__finalized=False
        
    def set_page_path(self,path):
        """
        Set the path to the results page, after self.baseurl.
        """
        self.resultsurl=os.path.join(self.job().baseurl,path)

    def set_gid(self,gid):
        """
        Set the GraceDB ID to log to
        """
        self.gid=gid

    def set_parent_resultspage(self,respagenode,gid):
        """
        Setup to log the results from the given parent results page node
        """
        res=respagenode
        #self.set_page_path(res.webpath.replace(self.job().basepath,self.job().baseurl))
        self.resultsurl=res.webpath.replace(self.job().basepath,self.job().baseurl)
        self.set_gid(gid)
    def finalize(self):
        if self.__finalized:
            return
        self.add_var_arg('log')
        self.add_var_arg(str(self.gid))
        #self.add_var_arg('"Parameter estimation finished. <a href=\"'+self.resultsurl+'/posplots.html\">'+self.resultsurl+'/posplots.html</a>"')
        self.add_var_arg('Parameter estimation finished. '+self.resultsurl+'/posplots.html')
        self.__finalized=True

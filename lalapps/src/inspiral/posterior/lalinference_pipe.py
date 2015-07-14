# DAG generation code for running LALInference pipeline
# (C) 2012 John Veitch, Vivien Raymond

from lalapps import lalinference_pipe_utils as pipe_utils
from lalapps import inspiralutils
import ConfigParser
from optparse import OptionParser,OptionValueError
import sys
import ast
import os

usage=""" %prog [options] config.ini
Setup a Condor DAG file to run the LALInference pipeline based on
the config.ini file.

The user can specify either an injection file to analyse, with the --inj option,
a list of SnglInspiralTable or CoincInspiralTable triggers with the --<x>-triggers options,
a GraceDB ID with the --gid option,
or an ASCII list of GPS times with the --gps-time-file option.

If none of the above options are given, the pipeline will analyse the entire
stretch of time between gps-start-time and gps-end-time, specified in the config.ini file.

The user must also specify and ini file which will contain the main analysis config.

"""
parser=OptionParser(usage)
parser.add_option("-r","--run-path",default=None,action="store",type="string",help="Directory to run pipeline in (default: $PWD)",metavar="RUNDIR")
parser.add_option("-p","--daglog-path",default=None,action="store",type="string",help="Path to directory to contain DAG log file. SHOULD BE LOCAL TO SUBMIT NODE",metavar="LOGDIR")
parser.add_option("-g","--gps-time-file",action="store",type="string",default=None,help="Text file containing list of GPS times to analyse",metavar="TIMES.txt")
parser.add_option("-t","--single-triggers",action="store",type="string",default=None,help="SnglInspiralTable trigger list",metavar="SNGL_FILE.xml")
parser.add_option("-C","--coinc-triggers",action="store",type="string",default=None,help="CoinInspiralTable trigger list",metavar="COINC_FILE.xml")
parser.add_option("--gid",action="store",type="string",default=None,help="GraceDB ID")
parser.add_option("-I","--injections",action="store",type="string",default=None,help="List of injections to perform and analyse",metavar="INJFILE.xml")
parser.add_option("-P","--pipedown-db",action="store",type="string",default=None,help="Pipedown database to read and analyse",metavar="pipedown.sqlite")
parser.add_option("--condor-submit",action="store_true",default=False,help="Automatically submit the condor dag")
parser.add_option("--pegasus-submit",action="store_true",default=False,help="Automatically submit the pegasus dax")
parser.add_option("-x", "--dax",action="store_true",default=False, help="Delete the ligo_data_find jobs and populate frame LFNs in the DAX")
parser.add_option("-G", "--grid-site",action="store",type="string",metavar="SITE", help="Specify remote site in conjunction with --dax option. e.g. --grid-site=creamce for Bologna cluster.\
Supported options are: creamce and local",default=None)

(opts,args)=parser.parse_args()

if len(args)!=1:
  parser.print_help()
  print 'Error: must specify one ini file'
  sys.exit(1)

inifile=args[0]

cp=ConfigParser.ConfigParser()
cp.optionxform = str
cp.readfp(open(inifile))

if opts.run_path is not None:
  cp.set('paths','basedir',os.path.abspath(opts.run_path))

if not cp.has_option('paths','basedir'):
  print 'Error: Must specify a directory with --run-path DIR'
  sys.exit(1)

if opts.daglog_path is not None:
  cp.set('paths','daglogdir',os.path.abspath(opts.daglog_path))
elif opts.run_path is not None:
  cp.set('paths','daglogdir',os.path.abspath(opts.run_path))
else:
  cp.set('paths','daglogdir',os.path.abspath(cp.get('paths','basedir')))

local_work_dir=cp.get('paths','daglogdir')

if opts.gps_time_file is not None:
  cp.set('input','gps-time-file',os.path.abspath(opts.gps_time_file))

if opts.single_triggers is not None:
  cp.set('input','sngl-inspiral-file',os.path.abspath(opts.single_triggers))

if opts.injections is not None:
  cp.set('input','injection-file',os.path.abspath(opts.injections))

if opts.coinc_triggers is not None:
  cp.set('input','coinc-inspiral-file',os.path.abspath(opts.coinc_triggers))

#if opts.lvalert is not None:
#  cp.set('input','lvalert-file',os.path.abspath(opts.lvalert))

if opts.gid is not None:
  cp.set('input','gid',opts.gid)

if opts.pipedown_db is not None:
  cp.set('input','pipedown-db',os.path.abspath(opts.pipedown_db))


# Create the DAG from the configparser object
dag=pipe_utils.LALInferencePipelineDAG(cp,dax=opts.dax,site=opts.grid_site)
if((opts.dax) and not cp.has_option('lalinference','fake-cache')):
# Create a text file with the frames listed
  pfnfile = dag.create_frame_pfn_file()
  peg_frame_cache = inspiralutils.create_pegasus_cache_file(pfnfile)
else:
  peg_frame_cache = '/dev/null'

# A directory to store the DAX temporary files
import uuid
execdir=os.path.join(local_work_dir,'lalinference_pegasus_'+str(uuid.uuid1()))
olddir=os.getcwd()
os.chdir(cp.get('paths','basedir'))
if opts.grid_site is not None:
    site='local,'+opts.grid_site
else:
    site=None
# Create the DAX scripts
if opts.dax:
  dag.prepare_dax(tmp_exec_dir=execdir,grid_site=site,peg_frame_cache=peg_frame_cache)
  # Ugly hack to replace pegasus.transfer.links=true in the pegasus.properties files created by pipeline.py
  # Turns off the creation of links for files on the local file system. We use pegasus.transfer.links=false
  # to make sure we have a copy of the data in the runing directory (useful when the data comes from temporary
  # low latency buffer).
  if cp.has_option('analysis','pegasus.transfer.links'):
    if cp.get('analysis','pegasus.transfer.links')=='false':
      lines=[]
      with open('pegasus.properties') as fin:
        for line in fin:
          line = line.replace('pegasus.transfer.links=true', 'pegasus.transfer.links=false')
          lines.append(line)
      with open('pegasus.properties','w') as fout:
        for line in lines:
          fout.write(line)

dag.write_sub_files()
dag.write_dag()
dag.write_script()
os.chdir(olddir)
# End of program
print 'Successfully created DAG file.'
fulldagpath=os.path.join(cp.get('paths','basedir'),dag.get_dag_file())
if not opts.dax:
    print 'Now run condor_submit_dag %s\n'%(fulldagpath)

if opts.condor_submit:
    import subprocess
    from subprocess import Popen

    x = subprocess.Popen(['condor_submit_dag',fulldagpath])
    x.wait()
    if x.returncode==0:
      print 'Submitted DAG file'
    else:
      print 'Unable to submit DAG file'

if opts.pegasus_submit:
    import subprocess
    from subprocess import Popen

    os.chdir(os.path.abspath(cp.get('paths','basedir')))
    x = subprocess.Popen('./pegasus_submit_dax')
    x.wait()
    if x.returncode==0:
      print 'Submitted DAX file'
    else:
      print 'Unable to submit DAX file'

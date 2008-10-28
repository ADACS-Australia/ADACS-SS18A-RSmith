"""
This modules contains objects that make it simple for the user to 
create python scripts that build Condor DAGs to run code on the LSC
Data Grid.

This file is part of the Grid LSC User Environment (GLUE)

GLUE is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Duncan Brown <duncan@gravity.phys.uwm.edu>'
__date__ = '$Date$'
__version__ = '$Revision$'[11:-2]

import os
import sys
import string, re
import exceptions
import time
import random
import md5
import math
import urlparse
import stat

def s2play(t):
  """
  Return 1 if t is in the S2 playground, 0 otherwise
  t = GPS time to test if playground
  """
  if ((t - 729273613) % 6370) < 600:
    return 1
  else:
    return 0


class CondorError(exceptions.Exception):
  """Error thrown by Condor Jobs"""
  def __init__(self, args=None):
    self.args = args
class CondorJobError(CondorError):
  pass
class CondorSubmitError(CondorError):
  pass
class CondorDAGError(CondorError):
  pass
class CondorDAGNodeError(CondorError):
  pass
class SegmentError(exceptions.Exception):
  def __init__(self, args=None):
    self.args = args


class CondorJob:
  """
  Generic condor job class. Provides methods to set the options in the
  condor submit file for a particular executable
  """
  def __init__(self, universe, executable, queue):
    """
    @param universe: the condor universe to run the job in.
    @param executable: the executable to run.
    @param queue: number of jobs to queue.
    """
    self.__universe = universe
    self.__executable = executable
    self.__queue = queue

    # These are set by methods in the class
    self.__options = {}
    self.__short_options = {}
    self.__arguments = []
    self.__condor_cmds = {}
    self.__notification = None
    self.__log_file = None
    self.__err_file = None
    self.__out_file = None
    self.__sub_file_path = None
    self.__output_files = []
    self.__input_files = []
    self.__grid_type = None
    self.__grid_server = None
    self.__grid_scheduler = None

  def get_executable(self):
    """
    Return the name of the executable for this job.
    """
    return self.__executable

  def set_executable(self, executable):
    """
    Set the name of the executable for this job.
    """
    self.__executable = executable

  def get_universe(self):
    """
    Return the condor universe that the job will run in.
    """
    return self.__universe

  def set_universe(self, universe):
    """
    Set the condor universe for the job to run in.
    @param universe: the condor universe to run the job in.
    """
    self.__universe = universe

  def get_grid_type(self):
    """
    Return the grid type of the job.
    """
    return self.__grid_type

  def set_grid_type(self, grid_type):
    """
    Set the type of grid resource for the job.
    @param grid_type: type of grid resource.
    """
    self.__grid_type = grid_type

  def get_grid_server(self):
    """
    Return the grid server on which the job will run.
    """
    return self.__grid_server

  def set_grid_server(self, grid_server):
    """
    Set the grid server on which to run the job.
    @param grid_server: grid server on which to run.
    """
    self.__grid_server = grid_server

  def get_grid_scheduler(self):
    """
    Return the grid scheduler.
    """
    return self.__grid_scheduler

  def set_grid_scheduler(self, grid_scheduler):
    """
    Set the grid scheduler.
    @param grid_scheduler: grid scheduler on which to run.
    """
    self.__grid_scheduler = grid_scheduler

  def add_condor_cmd(self, cmd, value):
    """
    Add a Condor command to the submit file (e.g. a class add or evironment).
    @param cmd: Condor command directive.
    @param value: value for command.
    """
    self.__condor_cmds[cmd] = value

  def add_input_file(self, filename):
    """
    Add filename as a necessary input file for this DAG node.

    @param filename: input filename to add
    """
    if filename not in self.__input_files:
      self.__input_files.append(filename)

  def add_output_file(self, filename):
    """
    Add filename as a output file for this DAG node.

    @param filename: output filename to add
    """
    if filename not in self.__output_files:
      self.__output_files.append(filename)

  def get_input_files(self):
    """
    Return list of input files for this DAG node.
    """
    return self.__input_files

  def get_output_files(self):
    """
    Return list of output files for this DAG node.
    """
    return self.__output_files

  def add_arg(self, arg):
    """
    Add an argument to the executable. Arguments are appended after any
    options and their order is guaranteed.
    @param arg: argument to add.
    """
    self.__arguments.append(arg)

  def add_file_arg(self, filename):
    """
    Add a file argument to the executable. Arguments are appended after any
    options and their order is guaranteed. Also adds the file name to the
    list of required input data for this job.
    @param filename: file to add as argument.
    """
    self.__arguments.append(filename)
    if filename not in self.__input_files:
      self.__input_files.append(filename)

  def get_args(self):
    """
    Return the list of arguments that are to be passed to the executable.
    """
    return self.__arguments

  def add_opt(self, opt, value):
    """
    Add a command line option to the executable. The order that the arguments
    will be appended to the command line is not guaranteed, but they will
    always be added before any command line arguments. The name of the option
    is prefixed with double hyphen and the program is expected to parse it
    with getopt_long().
    @param opt: command line option to add.
    @param value: value to pass to the option (None for no argument).
    """
    self.__options[opt] = value

  def get_opt( self, opt):
    """
    Returns the value associated with the given command line option.
    Returns None if the option does not exist in the options list.
    @param opt: command line option
    """
    if self.__options.has_key(opt):
      return self.__options[opt]
    return None

  def add_file_opt(self, opt, filename):
    """
    Add a command line option to the executable. The order that the arguments
    will be appended to the command line is not guaranteed, but they will
    always be added before any command line arguments. The name of the option
    is prefixed with double hyphen and the program is expected to parse it
    with getopt_long().
    @param opt: command line option to add.
    @param value: value to pass to the option (None for no argument).
    """
    self.__options[opt] = filename
    if filename not in self.__input_file:
      self.__input_files.append(filename)

  def get_opts(self):
    """
    Return the dictionary of opts for the job.
    """
    return self.__options

  def add_short_opt(self, opt, value):
    """
    Add a command line option to the executable. The order that the arguments
    will be appended to the command line is not guaranteed, but they will
    always be added before any command line arguments. The name of the option
    is prefixed with single hyphen and the program is expected to parse it
    with getopt() or getopt_long() (if a single character option), or
    getopt_long_only() (if multiple characters).  Long and (single-character)
    short options may be mixed if the executable permits this.
    @param opt: command line option to add.
    @param value: value to pass to the option (None for no argument).
    """
    self.__short_options[opt] = value

  def get_short_opts(self):
    """
    Return the dictionary of short options for the job.
    """
    return self.__short_options

  def add_ini_opts(self, cp, section):
    """
    Parse command line options from a given section in an ini file and
    pass to the executable.
    @param cp: ConfigParser object pointing to the ini file.
    @param section: section of the ini file to add to the options.
    """
    for opt in cp.options(section):
      arg = string.strip(cp.get(section,opt))
      self.__options[opt] = arg 


  def set_notification(self, value):
    """
    Set the email address to send notification to.
    @param value: email address or never for no notification.
    """
    self.__notification = value

  def set_log_file(self, path):
    """
    Set the Condor log file.
    @param path: path to log file.
    """
    self.__log_file = path
    
  def set_stderr_file(self, path):
    """
    Set the file to which Condor directs the stderr of the job.
    @param path: path to stderr file.
    """
    self.__err_file = path

  def get_stderr_file(self):
    """
    Get the file to which Condor directs the stderr of the job.
    """
    return self.__err_file

  def set_stdout_file(self, path):
    """
    Set the file to which Condor directs the stdout of the job.
    @param path: path to stdout file.
    """
    self.__out_file = path
  
  def get_stdout_file(self):
    """
    Get the file to which Condor directs the stdout of the job.
    """
    return self.__out_file

  def set_sub_file(self, path):
    """
    Set the name of the file to write the Condor submit file to when
    write_sub_file() is called.
    @param path: path to submit file.
    """
    self.__sub_file_path = path

  def get_sub_file(self):
    """
    Get the name of the file which the Condor submit file will be
    written to when write_sub_file() is called.
    """
    return self.__sub_file_path

  def write_sub_file(self):
    """
    Write a submit file for this Condor job.
    """
    if not self.__log_file:
      raise CondorSubmitError, "Log file not specified."
    if not self.__err_file:
      raise CondorSubmitError, "Error file not specified."
    if not self.__out_file:
      raise CondorSubmitError, "Output file not specified."
  
    if not self.__sub_file_path:
      raise CondorSubmitError, 'No path for submit file.'
    try:
      subfile = open(self.__sub_file_path, 'w')
    except:
      raise CondorSubmitError, "Cannot open file " + self.__sub_file_path

    if self.__universe == 'grid':
      if self.__grid_type == None:
        raise CondorSubmitError, 'No grid type specified.'
      elif self.__grid_type == 'gt2':
        if self.__grid_server == None:
          raise CondorSubmitError, 'No server specified for grid resource.'
      elif self.__grid_type == 'gt4':
        if self.__grid_server == None:
          raise CondorSubmitError, 'No server specified for grid resource.'
        if self.__grid_scheduler == None:
          raise CondorSubmitError, 'No scheduler specified for grid resource.'
      else:
        raise CondorSubmitError, 'Unsupported grid resource.'

    subfile.write( 'universe = ' + self.__universe + '\n' )
    subfile.write( 'executable = ' + self.__executable + '\n' )

    if self.__universe == 'grid':
      if self.__grid_type == 'gt2':
        subfile.write('grid_resource = %s %s\n' % (self.__grid_type,
          self.__grid_server))
      if self.__grid_type == 'gt4':
        subfile.write('grid_resource = %s %s %s\n' % (self.__grid_type,
          self.__grid_server, self.__grid_scheduler))

    if self.__options.keys() or self.__short_options.keys() or self.__arguments:
      subfile.write( 'arguments =' )
      for c in self.__options.keys():
        if self.__options[c]:
          subfile.write( ' --' + c + ' ' + self.__options[c] )
        else:
          subfile.write( ' --' + c )
      for c in self.__short_options.keys():
        if self.__short_options[c]:
          subfile.write( ' -' + c + ' ' + self.__short_options[c] )
        else:
          subfile.write( ' -' + c )
      for c in self.__arguments:
        subfile.write( ' ' + c )
      subfile.write( '\n' )

    for cmd in self.__condor_cmds.keys():
      subfile.write( cmd + " = " + self.__condor_cmds[cmd] + '\n' )

    subfile.write( 'log = ' + self.__log_file + '\n' )
    subfile.write( 'error = ' + self.__err_file + '\n' )
    subfile.write( 'output = ' + self.__out_file + '\n' )
    if self.__notification:
      subfile.write( 'notification = ' + self.__notification + '\n' )
    subfile.write( 'queue ' + str(self.__queue) + '\n' )

    subfile.close()



class CondorDAGJob(CondorJob):
  """
  A Condor DAG job never notifies the user on completion and can have variable
  options that are set for a particular node in the DAG. Inherits methods
  from a CondorJob.
  """
  def __init__(self, universe, executable):
    """
    universe = the condor universe to run the job in.
    executable = the executable to run in the DAG.
    """
    CondorJob.__init__(self, universe, executable, 1)
    CondorJob.set_notification(self, 'never')
    self.__var_opts = []
    self.__have_var_args = 0
    self.__bad_macro_chars = re.compile(r'[_-]')

  def add_var_opt(self, opt):
    """
    Add a variable (or macro) option to the condor job. The option is added 
    to the submit file and a different argument to the option can be set for
    each node in the DAG.
    @param opt: name of option to add.
    """
    if opt not in self.__var_opts:
      self.__var_opts.append(opt)
      macro = self.__bad_macro_chars.sub( r'', opt )
      self.add_opt(opt,'$(macro' + macro + ')')

  def add_var_arg(self):
    """
    Add a command to the submit file to allow variable (macro) arguments
    to be passed to the executable.
    """
    if not self.__have_var_args:
      self.add_arg('$(macroarguments)')
      self.__have_var_args = 1


class CondorDAGManJob:
  """
  Condor DAGMan job class. Appropriate for setting up DAGs to run within a
  DAG.
  """
  def __init__(self, dag, dir=None):
    """
    dag = the name of the condor dag file to run
    dir = the diretory in which the dag file is located
    """
    self.__dag = dag
    self.__options = {} 
    self.__notification = None
    self.__dag_directory= dir

  def add_opt(self, opt, value):
    """
    Add a command line option to the executable. The order that the arguments
    will be appended to the command line is not guaranteed, but they will
    always be added before any command line arguments. The name of the option
    is prefixed with single hyphen.
    @param opt: command line option to add.
    @param value: value to pass to the option (None for no argument).
    """
    self.__options[opt] = value

  def get_opts(self):
    """
    Return the dictionary of opts for the job.
    """
    return self.__options

  def set_dag_directory(self, dir):
    """
    Set the directory where the dag will be run
    @param dir: the name of the directory where the dag will be run
    """
    self.__dag_directory = dir

  def get_dag_directory(self):
    """
    Get the directory where the dag will be run
    """
    return self.__dag_directory
    
  def set_notification(self, value):
    """
    Set the email address to send notification to.
    @param value: email address or never for no notification.
    """
    self.__notification = value

  def get_sub_file(self):
    """
    Get the name of the file which the Condor submit file will be
    written to when write_sub_file() is called.
    """
    return self.__dag + ".condor.sub"

  def write_sub_file(self):
    """
    Write a submit file for this Condor job.
    """
    cwd = os.getcwd()
    if self.get_dag_directory() is not None:
      try: os.chdir(self.get_dag_directory())
      except: raise CondorSubmitError, \
          "directory " + self.get_dag_directory() + " doesn't exist"

    command = "condor_submit_dag -f -no_submit "

    if self.__options.keys():
      for c in self.__options.keys():
        if self.__options[c]:
          command +=  ' -' + c + ' ' + self.__options[c] 
        else:
          command += ' -' + c 
      command += ' '

    command += self.__dag

    stdin, out, err = os.popen3(command)
    pid, status = os.wait()

    if status != 0:
      raise CondorSubmitError, command + " failed."

    os.chdir(cwd)


class CondorDAGNode:
  """
  A CondorDAGNode represents a node in the DAG. It corresponds to a particular
  condor job (and so a particular submit file). If the job has variable
  (macro) options, they can be set here so each nodes executes with the
  correct options.
  """
  def __init__(self, job):
    """
    @param job: the CondorJob that this node corresponds to.
    """
    if not isinstance(job, CondorDAGJob) and \
        not isinstance(job,CondorDAGManJob):
      raise CondorDAGNodeError, \
          "A DAG node must correspond to a Condor DAG job or Condor DAGMan job"
    self.__name = None
    self.__job = job
    self.__category = None
    self.__priority = None
    self.__pre_script = None
    self.__pre_script_args = []
    self.__post_script = None
    self.__post_script_args = []
    self.__macros = {}
    self.__opts = {}
    self.__args = []
    self.__retry = 0
    self.__parents = []
    self.__bad_macro_chars = re.compile(r'[_-]')
    self.__output_files = []
    self.__input_files = []
    self.__vds_group = None

    # generate the md5 node name
    t = str( long( time.time() * 1000 ) )
    r = str( long( random.random() * 100000000000000000L ) )
    a = str( self.__class__ )
    self.__name = md5.md5(t + r + a).hexdigest()
    self.__md5name = self.__name

  def __repr__(self):
    return self.__name

  def job(self):
    """
    Return the CondorJob that this node is associated with.
    """
    return self.__job
  
  def set_pre_script(self,script):
    """
    Sets the name of the pre script that is executed before the DAG node is
    run.
    @param script: path to script
    """
    self.__pre_script = script

  def add_pre_script_arg(self,arg):
    """
    Adds an argument to the pre script that is executed before the DAG node is
    run.
    """
    self.__pre_script_args.append(arg)

  def set_post_script(self,script):
    """
    Sets the name of the post script that is executed before the DAG node is
    run.
    @param script: path to script
    """
    self.__post_script = script

  def add_post_script_arg(self,arg):
    """
    Adds an argument to the post script that is executed before the DAG node is
    run.
    """
    self.__post_script_args.append(arg)

  def set_name(self,name):
    """
    Set the name for this node in the DAG.
    """
    self.__name = str(name)

  def get_name(self):
    """
    Get the name for this node in the DAG.
    """
    return self.__name

  def set_category(self,category):
    """
    Set the category for this node in the DAG.
    """
    self.__category = str(category)

  def get_category(self):
    """
    Get the category for this node in the DAG.
    """
    return self.__category

  def set_priority(self,priority):
    """
    Set the priority for this node in the DAG.
    """
    self.__priority = str(priority)

  def get_priority(self):
    """
    Get the priority for this node in the DAG.
    """
    return self.__priority

  def add_input_file(self, filename):
    """
    Add filename as a necessary input file for this DAG node.

    @param filename: input filename to add
    """
    if filename not in self.__input_files:
      self.__input_files.append(filename)

  def add_output_file(self, filename):
    """
    Add filename as a output file for this DAG node.

    @param filename: output filename to add
    """
    if filename not in self.__output_files:
      self.__output_files.append(filename)

  def get_input_files(self):
    """
    Return list of input files for this DAG node and it's job.
    """
    input_files = list(self.__input_files)
    if isinstance(self.job(), CondorDAGJob):
      input_files = input_files + self.job().get_input_files()
    return input_files

  def get_output_files(self):
    """
    Return list of output files for this DAG node and it's job.
    """
    output_files = list(self.__output_files)
    if isinstance(self.job(), CondorDAGJob):
      output_files = output_files + self.job().get_output_files()
    return output_files

  def set_vds_group(self,group):
    """
    Set the name of the VDS group key when generating a DAX
    @param group: name of group for thus nore
    """
    self.__vds_group = str(group)

  def get_vds_group(self):
    """
    Returns the VDS group key for this node
    """
    return self.__vds_group

  def add_macro(self,name,value):
    """
    Add a variable (macro) for this node.  This can be different for
    each node in the DAG, even if they use the same CondorJob.  Within
    the CondorJob, the value of the macro can be referenced as
    '$(name)' -- for instance, to define a unique output or error file
    for each node.
    @param name: macro name.
    @param value: value of the macro for this node in the DAG
    """
    macro = self.__bad_macro_chars.sub( r'', name )
    self.__opts[macro] = value

  def get_opts(self):
    """
    Return the opts for this node. Note that this returns only
    the options for this instance of the node and not those
    associated with the underlying job template.
    """
    return self.__opts

  def add_var_opt(self,opt,value):
    """
    Add a variable (macro) option for this node. If the option
    specified does not exist in the CondorJob, it is added so the submit
    file will be correct when written.
    @param opt: option name.
    @param value: value of the option for this node in the DAG.
    """
    macro = self.__bad_macro_chars.sub( r'', opt )
    self.__opts['macro' + macro] = value
    self.__job.add_var_opt(opt)

  def add_file_opt(self,opt,filename,file_is_output_file=False):
    """
    Add a variable (macro) option for this node. If the option
    specified does not exist in the CondorJob, it is added so the submit
    file will be correct when written. The value of the option is also
    added to the list of input files for the DAX.
    @param opt: option name.
    @param value: value of the option for this node in the DAG.
    @param file_is_output_file: A boolean if the file will be an output file
    instead of an input file.  The default is to have it be an input.
    """
    self.add_var_opt(opt,filename)
    if file_is_output_file: self.add_output_file(filename)
    else: self.add_input_file(filename)

  def add_var_arg(self, arg):
    """
    Add a variable (or macro) argument to the condor job. The argument is
    added to the submit file and a different value of the argument can be set
    for each node in the DAG.
    @param arg: name of option to add.
    """
    self.__args.append(arg)
    self.__job.add_var_arg()

  def add_file_arg(self, filename):
    """
    Add a variable (or macro) file name argument to the condor job. The
    argument is added to the submit file and a different value of the 
    argument can be set for each node in the DAG. The file name is also
    added to the list of input files for the DAX.
    @param filename: name of option to add.
    """
    self.__args.append(filename)
    self.__job.add_var_arg()
    self.add_input_file(filename)

  def get_args(self):
    """
    Return the arguments for this node. Note that this returns
    only the arguments for this instance of the node and not those
    associated with the underlying job template.
    """
    return self.__args

  def set_retry(self, retry):
    """
    Set the number of times that this node in the DAG should retry.
    @param retry: number of times to retry node.
    """
    self.__retry = retry

  def write_job(self,fh):
    """
    Write the DAG entry for this node's job to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    """
    fh.write( 'JOB ' + self.__name + ' ' + self.__job.get_sub_file() )
    if isinstance(self.job(),CondorDAGManJob) and \
        self.job().get_dag_directory() is not None:
      fh.write( ' DIR ' + self.job().get_dag_directory() )
    fh.write( '\n')

    fh.write( 'RETRY ' + self.__name + ' ' + str(self.__retry) + '\n' )

  def write_category(self,fh):
    """
    Write the DAG entry for this node's category to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    """
    fh.write( 'CATEGORY ' + self.__name + ' ' + self.__category +  '\n' )

  def write_priority(self,fh):
    """
    Write the DAG entry for this node's priority to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    """
    fh.write( 'PRIORITY ' + self.__name + ' ' + self.__priority +  '\n' )

  def write_vars(self,fh):
    """
    Write the variable (macro) options and arguments to the DAG file
    descriptor.
    @param fh: descriptor of open DAG file.
    """
    if self.__macros.keys() or self.__opts.keys() or self.__args:
      fh.write( 'VARS ' + self.__name )
    for k in self.__macros.keys():
      fh.write( ' ' + str(k) + '="' + str(self.__macros[k]) + '"' )
    for k in self.__opts.keys():
      fh.write( ' ' + str(k) + '="' + str(self.__opts[k]) + '"' )
    if self.__args:
      fh.write( ' macroarguments="' + ' '.join(self.__args) + '"' )
    fh.write( '\n' )

  def write_parents(self,fh):
    """
    Write the parent/child relations for this job to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    """
    for parent in self.__parents:
      fh.write( 'PARENT ' + str(parent) + ' CHILD ' + str(self) + '\n' )

  def write_pre_script(self,fh):
    """
    Write the pre script for the job, if there is one
    @param fh: descriptor of open DAG file.
    """
    if self.__pre_script:
      fh.write( 'SCRIPT PRE ' + str(self) + ' ' + self.__pre_script + ' ' +
        ' '.join(self.__pre_script_args) + '\n' )

  def write_post_script(self,fh):
    """
    Write the post script for the job, if there is one
    @param fh: descriptor of open DAG file.
    """
    if self.__post_script:
      fh.write( 'SCRIPT POST ' + str(self) + ' ' + self.__post_script + ' ' +
        ' '.join(self.__post_script_args) + '\n' )

  def write_input_files(self, fh):
    """
    Write as a comment into the DAG file the list of input files
    for this DAG node.

    @param fh: descriptor of open DAG file.
    """
    for f in self.__input_files:
       print >>fh, "## Job %s requires input file %s" % (self.__name, f)
 
  def write_output_files(self, fh):
    """
    Write as a comment into the DAG file the list of output files
    for this DAG node.

    @param fh: descriptor of open DAG file.
    """
    for f in self.__output_files:
       print >>fh, "## Job %s generates output file %s" % (self.__name, f)

  def set_log_file(self,log):
    """
    Set the Condor log file to be used by this CondorJob.
    @param log: path of Condor log file.
    """
    self.__job.set_log_file(log)

  def add_parent(self,node):
    """
    Add a parent to this node. This node will not be executed until the
    parent node has run sucessfully.
    @param node: CondorDAGNode to add as a parent.
    """
    if not isinstance(node, CondorDAGNode):
      raise CondorDAGNodeError, "Parent must be a Condor DAG node"
    self.__parents.append( node )

  def get_cmd_line(self):
    """
    Return the full command line that will be used when this node
    is run by DAGman.
    """

    # pattern to find DAGman macros
    pat = re.compile(r'\$\((.+)\)')

    # first parse the options and replace macros with values
    options = self.job().get_opts()
    macros = self.get_opts()

    cmd = ""
    
    for k in options:
      val = options[k]
      m = pat.match(val)
      if m:
        key = m.group(1)
        value = macros[key]

        cmd += "--%s %s " % (k, value)
      else:
        cmd += "--%s %s " % (k, val)

    # second parse the short options and replace macros with values
    options = self.job().get_short_opts()

    for k in options:
      val = options[k]
      m = pat.match(val)
      if m:
        key = m.group(1)
        value = macros[key]

        cmd += "-%s %s " % (k, value)
      else:
        cmd += "-%s %s " % (k, val)

    # lastly parse the arguments and replace macros with values
    args = self.job().get_args()
    macros = self.get_args()

    for a in args:
      m = pat.match(a)
      if m:
        value = ' '.join(macros)

        cmd += "%s " % (value)
      else:
        cmd += "%s " % (a)

    return cmd
    
  def finalize(self):
    """
    The finalize method of a node is called before the node is
    finally added to the DAG and can be overridden to do any last
    minute clean up (such as setting extra command line arguments)
    """
    pass


class CondorDAG:
  """
  A CondorDAG is a Condor Directed Acyclic Graph that describes a collection
  of Condor jobs and the order in which to run them. All Condor jobs in the
  DAG must write their Codor logs to the same file.
  NOTE: The log file must not be on an NFS mounted system as the Condor jobs
  must be able to get an exclusive file lock on the log file.
  """
  def __init__(self,log,dax=False):
    """
    @param log: path to log file which must not be on an NFS mounted file system.
    @param dax: Set to 1 to create an abstract DAG (a DAX)
    """
    self.__log_file_path = log
    self.__dax = dax
    self.__dag_file_path = None
    self.__jobs = []
    self.__nodes = []
    self.__maxjobs_categories = []
    self.__integer_node_names = 0
    self.__node_count = 0
    self.__nodes_finalized = 0
    self.__rls_filelist = []
    self.__data_find_files = []

  def get_nodes(self):
    """
    Return a list containing all the nodes in the DAG
    """
    return self.__nodes
  
  def get_jobs(self):
    """
    Return a list containing all the jobs in the DAG
    """
    return self.__jobs


  def is_dax(self):
    """
    Returns true if this DAG is really a DAX
    """
    return self.__dax

  def set_integer_node_names(self):
    """
    Use integer node names for the DAG
    """
    self.__integer_node_names = 1

  def set_dag_file(self, path, no_append=0):
    """
    Set the name of the file into which the DAG is written.
    @param path: path to DAG file.
    """
    if no_append:
      self.__dag_file_path = path
    else:
      if self.__dax:
        self.__dag_file_path = path + '.dax'
      else:
        self.__dag_file_path = path + '.dag'

  def get_dag_file(self):
    """
    Return the path to the DAG file.
    """
    if not self.__log_file_path:
      raise CondorDAGError, "No path for DAG or DAX file"
    else:
      return self.__dag_file_path

  def add_node(self,node):
    """
    Add a CondorDAGNode to this DAG. The CondorJob that the node uses is 
    also added to the list of Condor jobs in the DAG so that a list of the
    submit files needed by the DAG can be maintained. Each unique CondorJob
    will be added once to prevent duplicate submit files being written.
    @param node: CondorDAGNode to add to the CondorDAG.
    """
    if not isinstance(node, CondorDAGNode):
      raise CondorDAGError, "Nodes must be class CondorDAGNode or subclass"
    if not isinstance(node.job(), CondorDAGManJob):
      node.set_log_file(self.__log_file_path)
    self.__nodes.append(node)
    if self.__integer_node_names:
      node.set_name(str(self.__node_count))
    self.__node_count += 1
    if node.job() not in self.__jobs:
      self.__jobs.append(node.job())

  def add_maxjobs_category(self,categoryName,maxJobsNum):
    """
    Add a category to this DAG called categoryName with a maxjobs of maxJobsNum.
    @param node: Add (categoryName,maxJobsNum) tuple to CondorDAG.__maxjobs_categories.
    """
    self.__maxjobs_categories.append((str(categoryName),str(maxJobsNum)))

  def write_maxjobs(self,fh,category):
    """
    Write the DAG entry for this category's maxjobs to the DAG file descriptor.
    @param fh: descriptor of open DAG file.
    @param category: tuple containing type of jobs to set a maxjobs limit for
        and the maximum number of jobs of that type to run at once.
    """
    fh.write( 'MAXJOBS ' + str(category[0]) + ' ' + str(category[1]) +  '\n' )

  def write_sub_files(self):
    """
    Write all the submit files used by the dag to disk. Each submit file is
    written to the file name set in the CondorJob.
    """
    if not self.__nodes_finalized:
      for node in self.__nodes:
        node.finalize()
    if not self.is_dax():
      for job in self.__jobs:
        job.write_sub_file()

  def write_concrete_dag(self):
    """
    Write all the nodes in the DAG to the DAG file.
    """
    if not self.__dag_file_path:
      raise CondorDAGError, "No path for DAG file"
    try:
      dagfile = open( self.__dag_file_path, 'w' )
    except:
      raise CondorDAGError, "Cannot open file " + self.__dag_file_path
    for node in self.__nodes:
      node.write_job(dagfile)
      node.write_vars(dagfile)
      if node.get_category():
        node.write_category(dagfile)
      if node.get_priority():
        node.write_priority(dagfile)
      node.write_pre_script(dagfile)
      node.write_post_script(dagfile)
      node.write_input_files(dagfile)
      node.write_output_files(dagfile)
    for node in self.__nodes:
      node.write_parents(dagfile)
    for category in self.__maxjobs_categories:
      self.write_maxjobs(dagfile, category)
    dagfile.close()

  def write_abstract_dag(self):
    """
    Write all the nodes in the workflow to the DAX file.
    """
    if not self.__dag_file_path:
      raise CondorDAGError, "No path for DAX file"
    try:
      dagfile = open( self.__dag_file_path, 'w' )
    except:
      raise CondorDAGError, "Cannot open file " + self.__dag_file_path

    # write the preamble
    preamble = """\
<?xml version="1.0" encoding="UTF-8"?>
<adag xmlns="http://www.griphyn.org/chimera/DAX"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
        xsi:schemaLocation="http://www.griphyn.org/chimera/DAX
        http://www.griphyn.org/chimera/dax-1.8.xsd"
"""
    preamble_2 = 'name="' + os.path.split(self.__dag_file_path)[-1]  + '" index="0" count="1" version="1.8">'
    print >>dagfile, preamble,preamble_2

    # find unique input and output files from nodes
    input_file_dict = {}
    output_file_dict = {}
 
    # creating dictionary for input- and output-files
    for node in self.__nodes:
      if isinstance(node, LSCDataFindNode):
        # make a list of the output files here so that I can have a 
        # more sensible rls_cache method that doesn't just ignore .gwf
        self.__data_find_files.extend(node.get_output())
        
      else:
        input_files = node.get_input_files()
        output_files = node.get_output_files()
        for f in input_files:
          input_file_dict[f] = 1
        for f in output_files:
          output_file_dict[f] = 1

    # move union of input and output into inout
    inout_file_dict = {}

    for f in input_file_dict:
      if output_file_dict.has_key(f):
         inout_file_dict[f] = 1

    for f in inout_file_dict:
      del input_file_dict[f]
      del output_file_dict[f]

    # print input, inout, and output to dax
    input_filelist = input_file_dict.keys()
    input_filelist.sort()
    self.__rls_filelist = input_filelist
    for f in input_filelist:
      msg = """\
    <filename file="%s" link="input"/>\
"""
      print >>dagfile, msg % f

    inout_filelist = inout_file_dict.keys()
    inout_filelist.sort()
    for f in inout_filelist:
      msg = """\
    <filename file="%s" link="inout"/>\
"""
      print >>dagfile, msg % f

    output_filelist = output_file_dict.keys()
    output_filelist.sort()
    for f in output_filelist:
      msg = """\
    <filename file="%s" link="output"/>\
"""
      print >>dagfile, msg % f

    # write the jobs themselves to the DAX, making sure
    # to replace logical file references by the appropriate
    # xml, and adding the files used by each job both for
    # input and output

    # we save the ID number to DAG node name mapping so that
    # we can easily write out the child/parent relationship
    # later
    node_name_id_dict = {}

    id = 0
    for node in self.__nodes:
      if isinstance(node, LSCDataFindNode):
        pass
      else:
        executable = node.job()._CondorJob__executable
        node_name = node._CondorDAGNode__name

        id += 1
        id_tag = "ID%06d" % id
        node_name_id_dict[node_name] = id_tag

        cmd_line = node.get_cmd_line()
        
        # loop through all filenames looking for them in the command
        # line so that they can be replaced appropriately by xml tags
        node_file_dict = {}
        for f in node.get_input_files():
          node_file_dict[f] = 1
        for f in node.get_output_files():      
          node_file_dict[f] = 1
        for f in node_file_dict.keys():
          xml = '<filename file="%s" />' % f
          cmd_line = cmd_line.replace(f, xml)

        template = """\
<job id="%s" namespace="ligo" name="%s" version="1.0" level="1" dv-name="%s">
     <argument>%s
     </argument>\
"""
        xml = template % (id_tag, os.path.basename(executable), node_name, cmd_line)

        # write the group if this node has one
        if node.get_vds_group():
          template = """<profile namespace="vds" key="group">%s</profile>"""
          xml = xml + template % (node.get_vds_group())

        print >>dagfile, xml
        
        for f in node.get_input_files():
          if f in inout_filelist:
            print >>dagfile, """\
     <uses file="%s" link="inout" dontRegister="true" dontTransfer="false"/>\
""" % f
          else:
            print >>dagfile, """\
     <uses file="%s" link="input" dontRegister="true" dontTransfer="false"/>\
""" % f

        for f in node.get_output_files():
          print >>dagfile, """\
     <uses file="%s" link="output" dontRegister="true" dontTransfer="false"/>\
""" % f

        print >>dagfile, "</job>"

    # print parent-child relationships to DAX
    for node in self.__nodes:
      if isinstance(node, LSCDataFindNode):
        pass
      elif ( len(node._CondorDAGNode__parents) == 1 ) and isinstance(node._CondorDAGNode__parents[0], LSCDataFindNode):
        pass
      else:
        child_id = node_name_id_dict[str(node)]
        if node._CondorDAGNode__parents:
          print >>dagfile, '<child ref="%s">' % child_id
          for parent in node._CondorDAGNode__parents:
            if isinstance(parent, LSCDataFindNode):
              pass
            else:
              parent_id = node_name_id_dict[str(parent)]
              print >>dagfile, '     <parent ref="%s"/>' % parent_id
          print >>dagfile, '</child>'

    print >>dagfile, "</adag>"

    dagfile.close()

  def write_pegasus_rls_cache(self,gsiftp,pool):
    try:
      outfilename = self.__dag_file_path+'.peg_cache'
      outfile = open(outfilename, "w")
    except:
      raise CondorDAGError, "Cannot open file " + self.__dag_file_path
    
    for filename in set(self.__rls_filelist):
      if filename in self.__data_find_files: continue
      # try to figure out if the path is absolute
      outfile.write(os.path.split(filename)[-1] + ' ' + 'gsiftp://'+gsiftp +os.path.abspath(filename)+' pool="'+pool+'"\n')
  def write_dag(self):
    """
    Write either a dag or a dax.
    """
    if not self.__nodes_finalized:
      for node in self.__nodes:
        node.finalize()
    if self.is_dax():
      self.write_abstract_dag()
    else:
      self.write_concrete_dag()

  def write_script(self):
    """
    Write the workflow to a script (.sh instead of .dag).
    
    Assuming that parents were added to the DAG before their children,
    dependencies should be handled correctly.
    """
    if not self.__dag_file_path:
      raise CondorDAGError, "No path for DAG file"
    try:
      dfp = self.__dag_file_path
      outfilename = ".".join(dfp.split(".")[:-1]) + ".sh"
      outfile = open(outfilename, "w")
    except:
      raise CondorDAGError, "Cannot open file " + self.__dag_file_path

    for node in self.__nodes:
        outfile.write("# Job %s\n" % node.get_name())
        outfile.write("%s %s\n\n" % (node.job().get_executable(),
            node.get_cmd_line()))
    outfile.close()
    
    os.chmod(outfilename, os.stat(outfilename)[0] | stat.S_IEXEC)


class AnalysisJob:
  """
  Describes a generic analysis job that filters LIGO data as configured by
  an ini file.
  """
  def __init__(self,cp,dax=False):
    """
    @param cp: ConfigParser object that contains the configuration for this job.
    """
    self.__cp = cp
    self.__dax = dax
    try:
      self.__channel = string.strip(self.__cp.get('input','channel'))
    except:
      self.__channel = None

  def is_dax(self):
    """
    Returns true if this job should behave as a DAX
    """
    return self.__dax

  def get_config(self,sec,opt):
    """
    Get the configration variable in a particular section of this jobs ini
    file.
    @param sec: ini file section.
    @param opt: option from section sec.
    """
    return string.strip(self.__cp.get(sec,opt))

  def set_channel(self,channel):
    """
    Set the name of the channel that this job is filtering.  This will 
    overwrite the value obtained at initialization.
    """
    self.__channel = channel

  def channel(self):
    """
    Returns the name of the channel that this job is filtering. Note that 
    channel is defined to be IFO independent, so this may be LSC-AS_Q or
    IOO-MC_F. The IFO is set on a per node basis, not a per job basis.
    """
    return self.__channel


class AnalysisNode(CondorDAGNode):
  """
  Contains the methods that allow an object to be built to analyse LIGO
  data in a Condor DAG.
  """
  def __init__(self):
    self.__start = 0
    self.__end = 0
    self.__data_start = 0
    self.__data_end = 0
    self.__trig_start = 0
    self.__trig_end = 0
    self.__ifo = None
    self.__ifo_tag = None
    self.__input = None
    self.__output = None
    self.__calibration = None
    self.__calibration_cache = None
    self.__LHO2k = re.compile(r'H2')
    self.__user_tag = self.job().get_opts().get("user-tag", None)

  def set_start(self,time):
    """
    Set the GPS start time of the analysis node by setting a --gps-start-time
    option to the node when it is executed.
    @param time: GPS start time of job.
    """
    self.add_var_opt('gps-start-time',time)
    self.__start = time
    self.__data_start = time
    #if not self.__calibration and self.__ifo and self.__start > 0:
    #  self.calibration()

  def get_start(self):
    """
    Get the GPS start time of the node.
    """
    return self.__start
    
  def set_end(self,time):
    """
    Set the GPS end time of the analysis node by setting a --gps-end-time
    option to the node when it is executed.
    @param time: GPS end time of job.
    """
    self.add_var_opt('gps-end-time',time)
    self.__end = time
    self.__data_end = time

  def get_end(self):
    """
    Get the GPS end time of the node.
    """
    return self.__end

  def set_data_start(self,time):
    """
    Set the GPS start time of the data needed by this analysis node.
    @param time: GPS start time of job.
    """
    self.__data_start = time

  def get_data_start(self):
    """
    Get the GPS start time of the data needed by this node.
    """
    return self.__data_start
    
  def set_data_end(self,time):
    """
    Set the GPS end time of the data needed by this analysis node.
    @param time: GPS end time of job.
    """
    self.__data_end = time

  def get_data_end(self):
    """
    Get the GPS end time of the data needed by this node.
    """
    return self.__data_end

  def set_trig_start(self,time):
    """
    Set the trig start time of the analysis node by setting a 
    --trig-start-time option to the node when it is executed.
    @param time: trig start time of job.
    """
    self.add_var_opt('trig-start-time',time)
    self.__trig_start = time

  def get_trig_start(self):
    """
    Get the trig start time of the node.
    """
    return self.__trig_start

  def set_trig_end(self,time):
    """
    Set the trig end time of the analysis node by setting a --trig-end-time
    option to the node when it is executed.
    @param time: trig end time of job.
    """
    self.add_var_opt('trig-end-time',time)
    self.__trig_end = time

  def get_trig_end(self):
    """
    Get the trig end time of the node.
    """
    return self.__trig_end

  def set_input(self,filename):
    """
    Add an input to the node by adding a --input option.
    @param filename: option argument to pass as input.
    """
    self.__input = filename
    self.add_var_opt('input', filename)
    self.add_input_file(filename)

  def get_input(self):
    """
    Get the file that will be passed as input.
    """
    return self.__input

  def set_output(self, filename):
    """
    Add an output to the node by adding a --output option.
    @param filename: option argument to pass as output.
    """
    self.__output = filename
    self.add_var_opt('output', filename)
    self.add_output_file(filename)

  def get_output(self):
    """
    Get the file that will be passed as output.
    """
    return self.__output

  def set_ifo(self,ifo):
    """
    Set the ifo name to analyze. If the channel name for the job is defined,
    then the name of the ifo is prepended to the channel name obtained
    from the job configuration file and passed with a --channel-name option.
    @param ifo: two letter ifo code (e.g. L1, H1 or H2).
    """
    self.__ifo = ifo
    if self.job().channel():
      self.add_var_opt('channel-name', ifo + ':' + self.job().channel())

  def get_ifo(self):
    """
    Returns the two letter IFO code for this node.
    """
    return self.__ifo

  def set_ifo_tag(self,ifo_tag):
    """
    Set the ifo tag that is passed to the analysis code.
    @param ifo_tag: a string to identify one or more IFOs
    """
    self.__ifo_tag = ifo_tag
    self.add_var_opt('ifo-tag', ifo_tag)

  def get_ifo_tag(self):
    """
    Returns the IFO tag string
    """
    return self.__ifo_tag

  def set_user_tag(self,usertag):
    """
    Set the user tag that is passed to the analysis code.
    @param user_tag: the user tag to identify the job
    """
    self.__user_tag = usertag
    self.add_var_opt('user-tag', usertag)
 
  def get_user_tag(self):
    """
    Returns the usertag string
    """
    return self.__user_tag

  def set_cache(self,filename):
    """
    Set the LAL frame cache to to use. The frame cache is passed to the job
    with the --frame-cache argument.
    @param filename: calibration file to use.
    """
    if isinstance( filename, str ):
      # the name of a lal cache file created by a datafind node
      self.add_var_opt('frame-cache', filename)
      self.add_input_file(filename)
    else:
      # check we have an LFN list
      from glue import LDRdataFindClient
      if isinstance( file, LDRdataFindClient.lfnlist ):
        self.add_var_opt('glob-frame-data',' ')
        # only add the LFNs that actually overlap with this job
        # FIXME this doesnt handle edge cases quite right
        for lfn in filename:
          a, b, c, d = lfn.split('.')[0].split('-')
          t_start = int(c)
          t_end = int(c) + int(d)
          if (t_start <= (self.__data_end+int(d)+1) and t_end >= (self.__data_start-int(d)-1)):
            self.add_input_file(lfn)
        # set the frame type based on the LFNs returned by datafind
        self.add_var_opt('frame-type',b)
      else:
        raise CondorDAGNodeError, "Unknown LFN cache format"
    
  def calibration_cache_path(self):
    """
    Determine the path to the correct calibration cache file to use.
    """
    if self.__ifo and self.__start > 0:
        cal_path = self.job().get_config('calibration','path')

        # check if this is S2: split calibration epochs
        if ( self.__LHO2k.match(self.__ifo) and 
          (self.__start >= 729273613) and (self.__start <= 734367613) ):
          if self.__start < int(
            self.job().get_config('calibration','H2-cal-epoch-boundary')):
            cal_file = self.job().get_config('calibration','H2-1')
          else:
            cal_file = self.job().get_config('calibration','H2-2')
        else:
            # if not: just add calibration cache
            cal_file = self.job().get_config('calibration',self.__ifo)

        cal = os.path.join(cal_path,cal_file)
        self.__calibration_cache = cal
    else:
       msg = "IFO and start-time must be set first"
       raise CondorDAGNodeError, msg 

  def calibration(self):
    """
    Set the path to the calibration cache file for the given IFO.
    During S2 the Hanford 2km IFO had two calibration epochs, so 
    if the start time is during S2, we use the correct cache file.
    """
    # figure out the name of the calibration cache files
    # as specified in the ini-file
    self.calibration_cache_path()

    if self.job().is_dax():
      # new code for DAX
      self.add_var_opt('glob-calibration-data','')
      cache_filename=self.get_calibration()
      pat = re.compile(r'(file://.*)')
      f = open(cache_filename, 'r')
      lines = f.readlines()

      # loop over entries in the cache-file...
      for line in lines:
        m = pat.search(line)
        if not m:
          raise IOError
        url = m.group(1)
        # ... and add files to input-file list
        path = urlparse.urlparse(url)[2]
        calibration_lfn = os.path.basename(path)
        self.add_input_file(calibration_lfn)
    else:
      # old .calibration for DAG's
      self.add_var_opt('calibration-cache', self.__calibration_cache)
      self.__calibration = self.__calibration_cache
      self.add_input_file(self.__calibration)

  def get_calibration(self):
    """
    Return the calibration cache file to be used by the
    DAG.
    """
    return self.__calibration_cache



class AnalysisChunk:
  """
  An AnalysisChunk is the unit of data that a node works with, usually some
  subset of a ScienceSegment.
  """
  def __init__(self, start, end, trig_start = 0, trig_end = 0):
    """
    @param start: GPS start time of the chunk.
    @param end: GPS end time of the chunk.
    @param trig_start: GPS time at which to start generating triggers
    @param trig_end: GPS time at which to stop generating triggers
    """
    self.__start = start
    self.__end = end
    self.__length = end - start
    self.__trig_start = trig_start
    self.__trig_end = trig_end

  def __repr__(self):
    if self.__trig_start and self.__trig_end:
      return '<AnalysisChunk: start %d, end %d, trig_start %d, trig_end %d>' % (
        self.__start, self.__end, self.__trig_start, self.__trig_end)
    elif self.__trig_start and not self.__trig_end:
      return '<AnalysisChunk: start %d, end %d, trig_start %d>' % (
        self.__start, self.__end, self.__trig_start)
    elif not self.__trig_start and self.__trig_end:
      return '<AnalysisChunk: start %d, end %d, trig_end %d>' % (
        self.__start, self.__end, self.__trig_end)
    else:
      return '<AnalysisChunk: start %d, end %d>' % (self.__start, self.__end)

  def __len__(self):
    """
    Returns the length of data for which this AnalysisChunk will produce
    triggers (in seconds).
    """
    if self.__trig_start and self.__trig_end:
      x = self.__trig_end - self.__trig_start
    elif self.__trig_start and not self.__trig_end:
      x = self.__end - self.__trig_start
    elif not self.__trig_start and self.__trig_end:
      x = self.__trig_end - self.__start
    else:
      x = self.__end - self.__start

    if x < 0:
      raise SegmentError, self + 'has negative length'
    else:
      return x
    
  def start(self):
    """
    Returns the GPS start time of the chunk.
    """
    return self.__start

  def end(self):
    """
    Returns the GPS end time of the chunk.
    """
    return self.__end
    
  def dur(self):
    """
    Returns the length (duration) of the chunk in seconds.
    """
    return self.__length

  def trig_start(self):
    """
    Return the first GPS time at which triggers for this chunk should be
    generated.
    """
    return self.__trig_start

  def trig_end(self):
    """
    Return the last GPS time at which triggers for this chunk should be
    generated.
    """
    return self.__trig_end

  def set_trig_start(self,start):
    """
    Set the first GPS time at which triggers for this chunk should be
    generated.
    """
    self.__trig_start = start

  def set_trig_end(self,end):
    """
    Set the last GPS time at which triggers for this chunk should be
    generated.
    """
    self.__trig_end = end



class ScienceSegment:
  """
  A ScienceSegment is a period of time where the experimenters determine
  that the inteferometer is in a state where the data is suitable for 
  scientific analysis. A science segment can have a list of AnalysisChunks
  asscociated with it that break the segment up into (possibly overlapping)
  smaller time intervals for analysis.
  """
  def __init__(self,segment):
    """
    @param segment: a tuple containing the (segment id, gps start time, gps end
    time, duration) of the segment.
    """
    self.__id = segment[0]
    self.__start = segment[1]
    self.__end = segment[2]
    self.__dur = segment[3]
    self.__chunks = []
    self.__unused = self.dur()
    self.__ifo = None
    self.__df_node = None

  def __getitem__(self,i):
    """
    Allows iteration over and direct access to the AnalysisChunks contained
    in this ScienceSegment.
    """
    if i < 0: raise IndexError, "list index out of range"
    return self.__chunks[i]
    
  def __len__(self):
    """
    Returns the number of AnalysisChunks contained in this ScienceSegment.
    """
    return len(self.__chunks)

  def __repr__(self):
    return '<ScienceSegment: id %d, start %d, end %d, dur %d, unused %d>' % (
    self.id(),self.start(),self.end(),self.dur(),self.__unused)

  def __cmp__(self,other):
    """
    ScienceSegments are compared by the GPS start time of the segment.
    """
    return cmp(self.start(),other.start())

  def make_chunks(self,length=0,overlap=0,play=0,sl=0,excl_play=0,pad_data=0):
    """
    Divides the science segment into chunks of length seconds overlapped by
    overlap seconds. If the play option is set, only chunks that contain S2
    playground data are generated. If the user has a more complicated way
    of generating chunks, this method should be overriden in a sub-class.
    Any data at the end of the ScienceSegment that is too short to contain a 
    chunk is ignored. The length of this unused data is stored and can be
    retrieved with the unused() method.
    @param length: length of chunk in seconds.
    @param overlap: overlap between chunks in seconds.
    @param play: 1 : only generate chunks that overlap with S2 playground data.
                 2 : as play = 1 plus compute trig start and end times to 
                     coincide with the start/end of the playground
    @param sl: slide by sl seconds before determining playground data.
    @param excl_play: exclude the first excl_play second from the start and end
    of the chunk when computing if the chunk overlaps with playground.
    @param pad_data: exclude the first and last pad_data seconds of the segment
    when generating chunks
    """
    time_left = self.dur() - (2 * pad_data)
    start = self.start() + pad_data
    increment = length - overlap
    while time_left >= length:
      end = start + length
      if (not play) or (play and (((end-sl-excl_play-729273613) % 6370) < 
        (600+length-2*excl_play))):
        if (play == 2):
        # calculate the start of the playground preceeding the chunk end
          play_start = 729273613 + 6370 * \
           math.floor((end-sl-excl_play-729273613) / 6370)
          play_end = play_start + 600
          trig_start = 0
          trig_end = 0
          if ( (play_end - 6370) > start ):
            print "Two playground segments in this chunk:",
            print "  Code to handle this case has not been implemented"
            sys.exit(1)
          else:
            if play_start > start:
              trig_start = int(play_start)
            if play_end < end:
              trig_end = int(play_end)
          self.__chunks.append(AnalysisChunk(start,end,trig_start,trig_end))
        else:
          self.__chunks.append(AnalysisChunk(start,end))
      start += increment
      time_left -= increment
    self.__unused = time_left - overlap

  def add_chunk(self,start,end,trig_start=0,trig_end=0):
    """
    Add an AnalysisChunk to the list associated with this ScienceSegment.
    @param start: GPS start time of chunk.
    @param end: GPS end time of chunk.
    @param trig_start: GPS start time for triggers from chunk
    """
    self.__chunks.append(AnalysisChunk(start,end,trig_start,trig_end))

  def unused(self):
    """
    Returns the length of data in the science segment not used to make chunks.
    """
    return self.__unused

  def set_unused(self,unused):
    """
    Set the length of data in the science segment not used to make chunks.
    """
    self.__unused = unused

  def id(self):
    """
    Returns the ID of this ScienceSegment.
    """
    return self.__id
    
  def start(self):
    """
    Returns the GPS start time of this ScienceSegment.
    """
    return self.__start

  def end(self):
    """
    Returns the GPS end time of this ScienceSegment.
    """
    return self.__end

  def set_start(self,t):
    """
    Override the GPS start time (and set the duration) of this ScienceSegment.
    @param t: new GPS start time.
    """
    self.__dur += self.__start - t
    self.__start = t

  def set_end(self,t):
    """
    Override the GPS end time (and set the duration) of this ScienceSegment.
    @param t: new GPS end time.
    """
    self.__dur -= self.__end - t
    self.__end = t

  def dur(self):
    """
    Returns the length (duration) in seconds of this ScienceSegment.
    """
    return self.__dur

  def set_df_node(self,df_node):
    """
    Set the DataFind node associated with this ScienceSegment to df_node.
    @param df_node: the DataFind node for this ScienceSegment.
    """
    self.__df_node = df_node

  def get_df_node(self):
    """
    Returns the DataFind node for this ScienceSegment.
    """
    return self.__df_node

    
class ScienceData:
  """
  An object that can contain all the science data used in an analysis. Can
  contain multiple ScienceSegments and has a method to generate these from
  a text file produces by the LIGOtools segwizard program.
  """
  def __init__(self):
    self.__sci_segs = []
    self.__filename = None

  def __getitem__(self,i):
    """
    Allows direct access to or iteration over the ScienceSegments associated
    with the ScienceData.
    """
    return self.__sci_segs[i]

  def __repr__(self):
    return '<ScienceData: file %s>' % self.__filename

  def __len__(self):
    """
    Returns the number of ScienceSegments associated with the ScienceData.
    """
    return len(self.__sci_segs)

  def read(self,filename,min_length,slide_sec=0,buffer=0):
    """
    Parse the science segments from the segwizard output contained in file.
    @param filename: input text file containing a list of science segments generated by
    segwizard.
    @param min_length: only append science segments that are longer than min_length.
    @param slide_sec: Slide each ScienceSegment by::

      delta > 0:
        [s,e] -> [s+delta,e].
      delta < 0:
        [s,e] -> [s,e-delta].

    @param buffer: shrink the ScienceSegment::

      [s,e] -> [s+buffer,e-buffer]
    """
    self.__filename = filename
    octothorpe = re.compile(r'\A#')
    for line in open(filename):
      if not octothorpe.match(line) and int(line.split()[3]) >= min_length:
        (id,st,en,du) = map(int,line.split())

        # slide the data if doing a background estimation
        if slide_sec > 0:
          st += slide_sec
        elif slide_sec < 0:
          en += slide_sec
        du -= abs(slide_sec)

        # add a buffer
        if buffer > 0:
          st += buffer
          en -= buffer
          du -= 2*abs(buffer)

        x = ScienceSegment(tuple([id,st,en,du]))
        self.__sci_segs.append(x)

  def append_from_tuple(self,seg_tuple):
    x = ScienceSegment(seg_tuple)
    self.__sci_segs.append(x)

  def tama_read(self,filename):
    """
    Parse the science segments from a tama list of locked segments contained in
                file.
    @param filename: input text file containing a list of tama segments.
    """
    self.__filename = filename
    for line in open(filename):
      columns = line.split()
      id = int(columns[0])
      start = int(math.ceil(float(columns[3])))
      end = int(math.floor(float(columns[4])))
      dur = end - start 
    
      x = ScienceSegment(tuple([id, start, end, dur]))
      self.__sci_segs.append(x)


  def make_chunks(self,length,overlap=0,play=0,sl=0,excl_play=0,pad_data=0):
    """
    Divide each ScienceSegment contained in this object into AnalysisChunks.
    @param length: length of chunk in seconds.
    @param overlap: overlap between segments.
    @param play: if true, only generate chunks that overlap with S2 playground 
    data.
    @param sl: slide by sl seconds before determining playground data.
    @param excl_play: exclude the first excl_play second from the start and end
    of the chunk when computing if the chunk overlaps with playground.
    """
    for seg in self.__sci_segs:
      seg.make_chunks(length,overlap,play,sl,excl_play,pad_data)

  def make_chunks_from_unused(self,length,trig_overlap,play=0,min_length=0,
    sl=0,excl_play=0,pad_data=0):
    """
    Create an extra chunk that uses up the unused data in the science segment.
    @param length: length of chunk in seconds.
    @param trig_overlap: length of time start generating triggers before the
    start of the unused data.
    @param play: 
                - 1 : only generate chunks that overlap with S2 playground data.
                - 2 : as 1 plus compute trig start and end times to coincide
                        with the start/end of the playground
    @param min_length: the unused data must be greater than min_length to make a
    chunk.
    @param sl: slide by sl seconds before determining playground data.
    @param excl_play: exclude the first excl_play second from the start and end
    of the chunk when computing if the chunk overlaps with playground.
    @param pad_data: exclude the first and last pad_data seconds of the segment
    when generating chunks

    """
    for seg in self.__sci_segs:
      # if there is unused data longer than the minimum chunk length
      if seg.unused() > min_length:
        end = seg.end() - pad_data
        start = end - length
        if (not play) or (play and (((end-sl-excl_play-729273613)%6370) < 
          (600+length-2*excl_play))):
          trig_start = end - seg.unused() - trig_overlap
          if (play == 2):
            # calculate the start of the playground preceeding the chunk end
            play_start = 729273613 + 6370 * \
              math.floor((end-sl-excl_play-729273613) / 6370)
            play_end = play_start + 600
            trig_end = 0
            if ( (play_end - 6370) > start ):
              print "Two playground segments in this chunk"
              print "  Code to handle this case has not been implemented"
              sys.exit(1)
            else:
              if play_start > trig_start:
                trig_start = int(play_start)
              if (play_end < end):
                trig_end = int(play_end)
              if (trig_end == 0) or (trig_end > trig_start):
                seg.add_chunk(start, end, trig_start, trig_end)
          else:
            seg.add_chunk(start, end, trig_start)
        seg.set_unused(0)

  def make_short_chunks_from_unused(
    self,min_length,overlap=0,play=0,sl=0,excl_play=0):
    """
    Create a chunk that uses up the unused data in the science segment
    @param min_length: the unused data must be greater than min_length to make a
    chunk.
    @param overlap: overlap between chunks in seconds.
    @param play: if true, only generate chunks that overlap with S2 playground data.
    @param sl: slide by sl seconds before determining playground data.
    @param excl_play: exclude the first excl_play second from the start and end
    of the chunk when computing if the chunk overlaps with playground.
    """
    for seg in self.__sci_segs:
      if seg.unused() > min_length:
        start = seg.end() - seg.unused() - overlap
        end = seg.end()
        length = start - end
        if (not play) or (play and (((end-sl-excl_play-729273613)%6370) < 
        (600+length-2*excl_play))):
          seg.add_chunk(start, end, start)
        seg.set_unused(0)

  def make_optimised_chunks(self, min_length, max_length, pad_data=0):
    """
    Splits ScienceSegments up into chunks, of a given maximum length.
    The length of the last two chunks are chosen so that the data
    utilisation is optimised.
    @param min_length: minimum chunk length.
    @param max_length: maximum chunk length.
    @param pad_data: exclude the first and last pad_data seconds of the
    segment when generating chunks
    """
    for seg in self.__sci_segs:
      # pad data if requested
      seg_start = seg.start() + pad_data
      seg_end = seg.end() - pad_data

      if seg.unused() > max_length:
        # get number of max_length chunks
        N = (seg_end - seg_start)/max_length

        # split into chunks of max_length
        for i in range(N-1):
          start = seg_start + (i * max_length)
          stop = start + max_length
          seg.add_chunk(start, stop)

        # optimise data usage for last 2 chunks
        start = seg_start + ((N-1) * max_length)
        middle = (start + seg_end)/2
        seg.add_chunk(start, middle)
        seg.add_chunk(middle, seg_end)
        seg.set_unused(0)
      elif seg.unused() > min_length:
        # utilise as single chunk
        seg.add_chunk(seg_start, seg_end)
      else:
        # no chunk of usable length
        seg.set_unused(0)

  def intersection(self, other):
    """
    Replaces the ScienceSegments contained in this instance of ScienceData
    with the intersection of those in the instance other. Returns the number
    of segments in the intersection.
    @param other: ScienceData to use to generate the intersection
    """

    # we only deal with the case of two lists here
    length1 = len(self)
    length2 = len(other)

    # initialize list of output segments
    ostart = -1
    outlist = []
    iseg2 = -1
    start2 = -1
    stop2 = -1

    for seg1 in self:
      start1 = seg1.start()
      stop1 = seg1.end()
      id = seg1.id()

      # loop over segments from the second list which overlap this segment
      while start2 < stop1:
        if stop2 > start1:
          # these overlap

          # find the overlapping range
          if start1 < start2:
            ostart = start2
          else:
            ostart = start1
          if stop1 > stop2:
            ostop = stop2
          else:
            ostop = stop1

          x = ScienceSegment(tuple([id, ostart, ostop, ostop-ostart]))
          outlist.append(x)

          if stop2 > stop1:
            break

        # step forward
        iseg2 += 1
        if iseg2 < len(other):
          seg2 = other[iseg2]
          start2 = seg2.start()
          stop2 = seg2.end()
        else:
          # pseudo-segment in the far future
          start2 = 2000000000
          stop2 = 2000000000

    # save the intersection and return the length
    self.__sci_segs = outlist
    return len(self)

  

  def union(self, other):
    """
    Replaces the ScienceSegments contained in this instance of ScienceData
    with the union of those in the instance other. Returns the number of
    ScienceSegments in the union.
    @param other: ScienceData to use to generate the intersection
    """

    # we only deal with the case of two lists here
    length1 = len(self)
    length2 = len(other)

    # initialize list of output segments
    ostart = -1
    seglist = []

    i1 = -1
    i2 = -1
    start1 = -1
    start2 = -1
    id = -1
    
    while 1:
      # if necessary, get a segment from list 1
      if start1 == -1:
        i1 += 1
        if i1 < length1:
          start1 = self[i1].start()
          stop1 = self[i1].end()
          id = self[i1].id()
        elif i2 == length2:
          break

      # if necessary, get a segment from list 2
      if start2 == -1:
        i2 += 1
        if i2 < length2:
          start2 = other[i2].start()
          stop2 = other[i2].end()
        elif i1 == length1:
          break

      # pick the earlier segment from the two lists
      if start1 > -1 and ( start2 == -1 or start1 <= start2):
        ustart = start1
        ustop = stop1
        # mark this segment has having been consumed
        start1 = -1
      elif start2 > -1:
        ustart = start2
        ustop = stop2
        # mark this segment has having been consumed
        start2 = -1
      else:
        break

      # if the output segment is blank, initialize it; otherwise, see
      # whether the new segment extends it or is disjoint
      if ostart == -1:
        ostart = ustart
        ostop = ustop
      elif ustart <= ostop:
        if ustop > ostop:
          # this extends the output segment
          ostop = ustop
        else:
          # This lies entirely within the current output segment
          pass
      else:
         # flush the current output segment, and replace it with the
         # new segment
         x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
         seglist.append(x)
         ostart = ustart
         ostop = ustop

    # flush out the final output segment (if any)
    if ostart != -1:
      x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
      seglist.append(x)

    self.__sci_segs = seglist
    return len(self)


  def coalesce(self):
    """
    Coalesces any adjacent ScienceSegments. Returns the number of 
    ScienceSegments in the coalesced list.
    """

    # check for an empty list
    if len(self) == 0:
      return 0

    # sort the list of science segments
    self.__sci_segs.sort()

    # coalesce the list, checking each segment for validity as we go
    outlist = []
    ostop = -1

    for seg in self:
      start = seg.start()
      stop = seg.end()
      id = seg.id()
      if start > ostop:
        # disconnected, so flush out the existing segment (if any)
        if ostop >= 0:
          x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
          outlist.append(x)
        ostart = start
        ostop = stop
      elif stop > ostop:
        # extend the current segment
        ostop = stop

    # flush out the final segment (if any)
    if ostop >= 0:
      x = ScienceSegment(tuple([id,ostart,ostop,ostop-ostart]))
      outlist.append(x)

    self.__sci_segs = outlist
    return len(self)


  def invert(self):
    """
    Inverts the ScienceSegments in the class (i.e. set NOT).  Returns the
    number of ScienceSegments after inversion.
    """

    # check for an empty list
    if len(self) == 0:
      # return a segment representing all time
      self.__sci_segs = ScienceSegment(tuple([0,0,1999999999,1999999999]))

    # go through the list checking for validity as we go
    outlist = []
    ostart = 0
    for seg in self:
      start = seg.start()
      stop = seg.end()
      if start < 0 or stop < start or start < ostart:
        raise SegmentError, "Invalid list"
      if start > 0:
        x = ScienceSegment(tuple([0,ostart,start,start-ostart]))
        outlist.append(x)
      ostart = stop

    if ostart < 1999999999:
      x = ScienceSegment(tuple([0,ostart,1999999999,1999999999-ostart]))
      outlist.append(x)

    self.__sci_segs = outlist
    return len(self)

  
  def play(self):
    """
    Keep only times in ScienceSegments which are in the playground
    """

    length = len(self)

    # initialize list of output segments
    ostart = -1
    outlist = []
    begin_s2 = 729273613
    play_space = 6370
    play_len = 600

    for seg in self:
      start = seg.start()
      stop = seg.end()
      id = seg.id()
     
      # select first playground segment which ends after start of seg
      play_start = begin_s2+play_space*( 1 + 
        int((start - begin_s2 - play_len)/play_space) )

      while play_start < stop:
        if play_start > start:
          ostart = play_start
        else:
          ostart = start
        
        
        play_stop = play_start + play_len

        if play_stop < stop:
          ostop = play_stop
        else:
          ostop = stop

        x = ScienceSegment(tuple([id, ostart, ostop, ostop-ostart]))
        outlist.append(x)

        # step forward
        play_start = play_start + play_space 

    # save the playground segs and return the length
    self.__sci_segs = outlist
    return len(self)


  def intersect_3(self, second, third):
    """
    Intersection routine for three inputs.  Built out of the intersect, 
    coalesce and play routines
    """
    self.intersection(second)
    self.intersection(third)
    self.coalesce()
    return len(self)

  def intersect_4(self, second, third, fourth):
    """
     Intersection routine for four inputs.
    """
    self.intersection(second)
    self.intersection(third)
    self.intersection(fourth)
    self.coalesce()
    return len(self)

  def split(self, dt):
    """
      Split the segments in the list is subsegments at least as long as dt
    """
    outlist=[]
    for seg in self:
      start = seg.start()
      stop = seg.end()
      id = seg.id()

      while start < stop:
        tmpstop = start + dt
        if tmpstop > stop:
          tmpstop = stop
        elif tmpstop + dt > stop:
          tmpstop = int( (start + stop)/2 )
        x = ScienceSegment(tuple([id,start,tmpstop,tmpstop-start]))
        outlist.append(x)
        start = tmpstop

    # save the split list and return length
    self.__sci_segs = outlist
    return len(self)


  
class LSCDataFindJob(CondorDAGJob, AnalysisJob):
  """
  An LSCdataFind job used to locate data. The static options are
  read from the section [datafind] in the ini file. The stdout from
  LSCdataFind contains the paths to the frame files and is directed to a file
  in the cache directory named by site and GPS start and end times. The stderr
  is directed to the logs directory. The job always runs in the scheduler
  universe. The path to the executable is determined from the ini file.
  """
  def __init__(self,cache_dir,log_dir,config_file,dax=0):
    """
    @param cache_dir: the directory to write the output lal cache files to.
    @param log_dir: the directory to write the stderr file to.
    @param config_file: ConfigParser object containing the path to the LSCdataFind
    executable in the [condor] section and a [datafind] section from which
    the LSCdataFind options are read.
    """
    self.__executable = config_file.get('condor','datafind')
    self.__universe = 'local'
    CondorDAGJob.__init__(self,self.__universe,self.__executable)
    AnalysisJob.__init__(self,config_file)
    self.__cache_dir = cache_dir
    self.__dax = dax
    self.__config_file = config_file

    # we have to do this manually for backwards compatibility with type
    for o in self.__config_file.options('datafind'):
      opt = string.strip(o)
      if opt[:4] != "type":
        arg = string.strip(self.__config_file.get('datafind',opt))
        self.add_opt(opt,arg)

    if self.__dax:
      # only get the LFNs not the PFNs
      self.add_opt('names-only','')
    else:
      # we need a lal cache for file PFNs
      self.add_opt('lal-cache','')
      self.add_opt('url-type','file')

    self.add_condor_cmd('getenv','True')

    self.set_stderr_file(os.path.join(log_dir, 'datafind-$(macroobservatory)-$(macrotype)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err'))
    self.set_stdout_file(os.path.join(log_dir, 'datafind-$(macroobservatory)-$(macrotype)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out'))
    self.set_sub_file('datafind.sub')

  def get_cache_dir(self):
    """
    returns the directroy that the cache files are written to.
    """
    return self.__cache_dir

  def is_dax(self):      
    """          
    returns the dax flag         
    """          
    return self.__dax    

  def get_config_file(self):
    """
    return the configuration file object
    """
    return self.__config_file
  

class LSCDataFindNode(CondorDAGNode, AnalysisNode):
  """
  A DataFindNode runs an instance of LSCdataFind in a Condor DAG.
  """
  def __init__(self,job):
    """
    @param job: A CondorDAGJob that can run an instance of LALdataFind.
    """
    CondorDAGNode.__init__(self,job)
    AnalysisNode.__init__(self)
    self.__start = 0
    self.__end = 0
    self.__observatory = None
    self.__output = None
    self.__job = job
    self.__dax = job.is_dax()    
    self.__lfn_list = None
     
    # try and get a type from the ini file and default to type None
    try:
      self.set_type(self.job().get_config_file().get('datafind','type'))
    except:
      self.__type = None
 
  def __set_output(self):
    """
    Private method to set the file to write the cache to. Automaticaly set
    once the ifo, start and end times have been set.
    """
    if self.__start and self.__end and self.__observatory and self.__type:
      self.__output = os.path.join(self.__job.get_cache_dir(), self.__observatory + '-' + self.__type + '-' + str(self.__start) + '-' + str(self.__end) + '.cache')
      self.set_output(self.__output)

  def set_start(self,time):
    """
    Set the start time of the datafind query.
    @param time: GPS start time of query.
    """
    self.add_var_opt('gps-start-time', time)
    self.__start = time
    self.__set_output()

  def get_start(self):
    """
    Return the start time of the datafind query
    """
    return self.__start

  def set_end(self,time):
    """
    Set the end time of the datafind query.
    @param time: GPS end time of query.
    """
    self.add_var_opt('gps-end-time', time)
    self.__end = time
    self.__set_output()

  def get_end(self):
    """
    Return the start time of the datafind query
    """
    return self.__end

  def set_observatory(self,obs):
    """
    Set the IFO to retrieve data for. Since the data from both Hanford 
    interferometers is stored in the same frame file, this takes the first 
    letter of the IFO (e.g. L or H) and passes it to the --observatory option
    of LSCdataFind.
    @param obs: IFO to obtain data for.
    """
    self.add_var_opt('observatory',obs)
    self.__observatory = str(obs)
    self.__set_output()

  def get_observatory(self):
    """
    Return the start time of the datafind query
    """
    return self.__observatory

  def set_type(self,type):
    """
    sets the frame type that we are querying
    """
    self.add_var_opt('type',str(type))
    self.__type = str(type)
    self.__set_output()

  def get_type(self):
    """
    gets the frame type that we are querying
    """
    return self.__type

  def get_output_cache(self):
    return  self.__output

  def get_output(self):
    """
    Return the output file, i.e. the file containing the frame cache data.
    or the files itself as tuple (for DAX)       
    """  
    if self.__dax:
      if not self.__lfn_list:
        # call the datafind client to get the LFNs
        from pyGlobus import security
        from glue import LDRdataFindClient
        from glue import gsiserverutils

        hostPortString = os.environ['LSC_DATAFIND_SERVER']
        print >>sys.stderr, ".",
        if hostPortString.find(':') < 0:
          # no port specified
          myClient = LDRdataFindClient.LSCdataFindClient(hostPortString)
        else:
          # server and port specified
          host, portString = hostPortString.split(':')
          port = int(portString)
          myClient = LDRdataFindClient.LSCdataFindClient(host,port)

        clientMethod = 'findFrameNames'
        clientMethodArgDict = {
          'observatory': self.get_observatory(),
          'end': str(self.get_end()),
          'start': str(self.get_start()),
          'type': self.get_type(),
          'filename': None,
          'urlType': None,
          'match': None,
          'limit': None,
          'offset': None,
          'strict' : None,
          'namesOnly' : True
          }

        print >>sys.stderr, ".",
        time.sleep( 1 )
        result = eval("myClient.%s(%s)" % (clientMethod, clientMethodArgDict))
        
        if not isinstance(result,LDRdataFindClient.lfnlist):
          msg = "datafind server did not return LFN list : " + str(result)
          raise SegmentError, msg
        if len(result) == 0:
          msg = "No LFNs returned for segment %s %s" % ( str(self.get_start()),
            str(self.get_end()) )
          raise SegmentError, msg
        self.__lfn_list = result
      return self.__lfn_list
    else:        
      return self.__output

class LigolwAddJob(CondorDAGJob, AnalysisJob):
  """
  A ligolw_add job can be used to concatenate several ligo lw files
  """
  def __init__(self,log_dir,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','ligolw_add')
    self.__universe = 'vanilla'
    CondorDAGJob.__init__(self,self.__universe,self.__executable)
    AnalysisJob.__init__(self,cp,dax)
    self.add_ini_opts(cp, "ligolw_add")

    self.add_condor_cmd('getenv','True')

    self.set_stdout_file(os.path.join( log_dir, 'ligolw_add-$(cluster)-$(process).out') )
    self.set_stderr_file(os.path.join( log_dir, 'ligolw_add-$(cluster)-$(process).err') )
    self.set_sub_file('ligolw_add.sub')


class LigolwAddNode(CondorDAGNode, AnalysisNode):
  """
  Runs an instance of ligolw_add in a Condor DAG.
  """
  def __init__(self,job):
    """
    @param job: A CondorDAGJob that can run an instance of ligolw_add
    """
    CondorDAGNode.__init__(self,job)
    AnalysisNode.__init__(self)


class LigolwCutJob(CondorDAGJob, AnalysisJob):
  """
  A ligolw_cut job can be used to remove parts of a ligo lw file
  """
  def __init__(self,log_dir,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','ligolw_cut')
    self.__universe = 'vanilla'
    CondorDAGJob.__init__(self,self.__universe,self.__executable)
    AnalysisJob.__init__(self,cp,dax)

    self.add_condor_cmd('getenv','True')

    self.set_stdout_file(os.path.join( log_dir, 'ligolw_cut-$(cluster)-$(process).out') )
    self.set_stderr_file(os.path.join( log_dir, 'ligolw_cut-$(cluster)-$(process).err') )
    self.set_sub_file('ligolw_cut.sub')


class LigolwCutNode(CondorDAGNode, AnalysisNode):
  """
  Runs an instance of ligolw_cut in a Condor DAG.
  """
  def __init__(self,job):
    """
    @param job: A CondorDAGJob that can run an instance of ligolw_cut
    """
    CondorDAGNode.__init__(self,job)
    AnalysisNode.__init__(self)


class LDBDCJob(CondorDAGJob, AnalysisJob):
  """
  A ldbdc job can be used to insert data or fetch data from the database.
  """
  def __init__(self,log_dir,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','ldbdc')
    self.__universe = 'local'
    CondorDAGJob.__init__(self,self.__universe,self.__executable)
    AnalysisJob.__init__(self,cp,dax)

    self.add_condor_cmd('getenv','True')

    self.set_stdout_file(os.path.join( log_dir, 'ldbdc-$(cluster)-$(process).out') )
    self.set_stderr_file(os.path.join( log_dir, 'ldbdc-$(cluster)-$(process).err') )
    self.set_sub_file('ldbdc.sub')


class LDBDCNode(CondorDAGNode, AnalysisNode):
  """
  Runs an instance of ldbdc in a Condor DAG.
  """
  def __init__(self,job):
    """
    @param job: A CondorDAGJob that can run an instance of ligolw_add
    """
    CondorDAGNode.__init__(self,job)
    AnalysisNode.__init__(self)
    self.__server = None
    self.__identity = None
    self.__insert = None
    self.__pfn = None
    self.__query = None

  def set_server(self, server):
    """
    Set the server name.
    """
    self.add_var_opt('server',server)
    self.__server = server

  def get_server(self, server):
    """
    Get the server name.
    """
    return self.__server

  def set_identity(self, identity):
    """
    Set the identity name.
    """
    self.add_var_opt('identity',identity)
    self.__identity = identity

  def get_identity(self, identity):
    """
    Get the identity name.
    """
    return self.__identity

  def set_insert(self, insert):
    """
    Set the insert name.
    """
    self.add_var_opt('insert',insert)
    self.__insert = insert

  def get_insert(self, insert):
    """
    Get the insert name.
    """
    return self.__insert

  def set_pfn(self, pfn):
    """
    Set the pfn name.
    """
    self.add_var_opt('pfn',pfn)
    self.__pfn = pfn

  def get_pfn(self, pfn):
    """
    Get the pfn name.
    """
    return self.__pfn

  def set_query(self, query):
    """
    Set the query name.
    """
    self.add_var_opt('query',query)
    self.__query = query

  def get_query(self, query):
    """
    Get the query name.
    """
    return self.__query


class NoopJob(CondorDAGJob, AnalysisJob):
  """
  A Noop Job does nothing.
  """
  def __init__(self,log_dir,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = 'true'
    self.__universe = 'local'
    CondorDAGJob.__init__(self,self.__universe,self.__executable)
    AnalysisJob.__init__(self,cp,dax)

    self.add_condor_cmd('getenv','True')
    self.add_condor_cmd('noop_job','True')

    self.set_stdout_file(os.path.join( log_dir, 'noop-$(cluster)-$(process).out') )
    self.set_stderr_file(os.path.join( log_dir, 'noop-$(cluster)-$(process).err') )
    self.set_sub_file('noop.sub')


class NoopNode(CondorDAGNode, AnalysisNode):
  """
  Run an noop job in a Condor DAG.
  """
  def __init__(self,job):
    """
    @param job: A CondorDAGJob that does nothing.
    """
    CondorDAGNode.__init__(self,job)
    AnalysisNode.__init__(self)
    self.__server = None
    self.__identity = None
    self.__insert = None
    self.__pfn = None
    self.__query = None

"""
Classes needed for the inspiral analysis pipeline.
This script produced the necessary condor submit and dag files to run
the standalone inspiral code on LIGO data
"""

__author__ = 'Duncan Brown <duncan@gravity.phys.uwm.edu>'
__date__ = '$Date$'
__version__ = '$Revision$'[11:-2]

import string
import exceptions
from glue import pipeline


class InspiralError(exceptions.Exception):
  def __init__(self, args=None):
    self.args = args


class TmpltBankJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_tmpltbank job used by the inspiral pipeline. The static options
  are read from the sections [data] and [tmpltbank] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file.
  """
  def __init__(self,cp,dax=False,tag_base='TMPLTBANK'):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','tmpltbank')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)
    self.tag_base = tag_base

    for sec in ['data','tmpltbank']:
      self.add_ini_opts(cp,sec)
  
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/tmpltbank-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/tmpltbank-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('tmpltbank.sub')


class InspInjJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_inspinj job used by the grb inspiral pipeline. The static options
  are read from the section [inspinj] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The
  job runs in the universe specified in the ini file. The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','inspinj')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)
    self.__listDone=[]
    self.__listNodes=[]

    for sec in ['inspinj']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/inspinj-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/inspinj-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')

  def set_done(self, number, node):
    self.__listDone.append(number)
    self.__listNodes.append(node)

  def check_node(self, number):
    if self.__listDone.count(number):
      index=self.__listDone.index(number)
      return self.__listNodes[index]
    return None    


class BbhInjJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_bbhinj job used by the online inspiral pipeline. The static options
  are read from the section [bbhinj] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The
  job runs in the universe specified in the ini file. The path to the 
  executable is determined from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','bbhinj')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)

    for sec in ['bbhinj']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/bbhinj-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/bbhinj-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')


class RandomBankJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_randombank job used by the inspiral pipeline. The static options
  are read from the section [randombank] in the ini file. The stdout and
  stderr from the job are directed to the logs directory. The job runs in the
  universe specfied in the ini file. The path to the executable is determined
  from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','randombank')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)

    self.add_ini_opts(cp,'randombank')

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/randombank-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/randombank-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('randombank.sub')


class SplitBankJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_splitbank job used by the inspiral pipeline. The static options
  are read from the section [splitbank] in the ini file. The stdout and stderr
  from the job are directed to the logs directory. The job runs in the
  universe specfied in the ini file. The path to the executable is determined
  from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','splitbank')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)

    for sec in ['splitbank']:
      self.add_ini_opts(cp,sec)
  
    self.set_stdout_file('logs/splitbank-$(macrobankfile)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/splitbank-$(macrobankfile)-$(cluster)-$(process).err')
    self.set_sub_file('splitbank.sub')
    

class InspiralJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_inspiral job used by the inspiral pipeline. The static options
  are read from the sections [data] and [inspiral] in the ini file. The
  stdout and stderr from the job are directed to the logs directory. The job
  runs in the universe specfied in the ini file. The path to the executable
  is determined from the ini file.
  """
  def __init__(self,cp,dax=False,tag_base='INSPIRAL'):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','inspiral')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)
    self.tag_base = tag_base

    for sec in ['data','inspiral']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/inspiral-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/inspiral-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('inspiral.sub')
    

class TrigToTmpltJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_trigtotmplt job used by the inspiral pipeline. The static
  options are read from the section [trigtotmplt] in the ini file.  The
  stdout and stderr from the job are directed to the logs directory. The job
  always runs in the scheduler universe. The path to the executable is
  determined from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','trigtotmplt')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)
    
    for sec in ['trigtotmplt']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")
    
    self.set_stdout_file('logs/trigtotmplt-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/trigtotmplt-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('trigtotmplt.sub')


class IncaJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_inca job used by the inspiral pipeline. The static options are
  read from the section [inca] in the ini file.  The stdout and stderr from
  the job are directed to the logs directory.  The path to the executable is 
  determined from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','inca')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)
    
    for sec in ['inca']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/inca-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/inca-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('inca.sub')


class ThincaJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_thinca job used by the inspiral pipeline. The static options are
  read from the section [thinca] in the ini file.  The stdout and stderr from
  the job are directed to the logs directory.  The path to the executable is 
  determined from the ini file.
  """
  def __init__(self,cp,dax=False,tag_base='THINCA'):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','thinca')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,False)
    self.tag_base = tag_base
    
    for sec in ['thinca']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/thinca-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/thinca-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('thinca.sub')


class SireJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_sire job used by the inspiral pipeline. The stdout and stderr from
  the job are directed to the logs directory. The path to the executable is 
  determined from the ini file.
  """
  def __init__(self,cp,dax=False):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','sire')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp,dax)
    
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/sire-$(macroifo)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/sire-$(macroifo)-$(cluster)-$(process).err')
    self.set_sub_file('sire.sub')

class FrJoinJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_frjoin job used by the inspiral pipeline. The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','frjoin')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp)
    
    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/frjoin-$(cluster)-$(process).out')
    self.set_stderr_file('logs/frjoin-$(cluster)-$(process).err')
    self.set_sub_file('frjoin.sub')

class CohBankJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_coherent_inspiral job used by the inspiral pipeline. The static
  options are read from the section [cohbank] in the ini file.  The stdout and
  stderr from the job are directed to the logs directory.  The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','cohbank')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp)
    
    for sec in ['cohbank']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/cohbank-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/cohbank-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('cohbank.sub')

class ChiaJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
  """
  A lalapps_coherent_inspiral job used by the inspiral pipeline. The static
  options are read from the section [chia] in the ini file.  The stdout and
  stderr from the job are directed to the logs directory.  The path to the
  executable is determined from the ini file.
  """
  def __init__(self,cp):
    """
    cp = ConfigParser object from which options are read.
    """
    self.__executable = cp.get('condor','chia')
    self.__universe = cp.get('condor','universe')
    pipeline.CondorDAGJob.__init__(self,self.__universe,self.__executable)
    pipeline.AnalysisJob.__init__(self,cp)
    
    for sec in ['chia']:
      self.add_ini_opts(cp,sec)

    self.add_condor_cmd('environment',"KMP_LIBRARY=serial;MKL_SERIAL=yes")

    self.set_stdout_file('logs/chia-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out')
    self.set_stderr_file('logs/chia-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err')
    self.set_sub_file('chia.sub')   

class InspInjNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A InspInjNode runs an instance of the inspinj generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspinj.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')

  def set_seed(self,seed):
    """
    Set the seed of the injection file by setting a --seed option to the
    node when it is executed. The seed is automatically the number of
    the injection 'round'.
    @param seed: seed of the job
    """
    self.add_var_opt('seed',seed)
    self.__seed = seed

  def set_output(self, outputName):
    """
    Set the output name of the injection file
    @param outputName: name of the injection file created
    """
    self.add_var_opt('output',outputName)
    self.__outputName = outputName

class BbhInjNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A BbhInjNode runs an instance of the bbhinj generation job in a 
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_bbhinj.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')

  def set_start(self,time):
    """
    Set the GPS start time of the analysis node by setting a --gps-start-time
    option to the node when it is executed. We override the default method to
    cope with the data padding.
    @param time:GPS start time of analysis segment
    """
    self.add_var_opt('gps-start-time',time)
    pipeline.AnalysisNode.set_start(self,time)
    pad = int(self.job().get_config('data','pad-data'))
    pipeline.AnalysisNode.set_data_start(self,time - pad)

  def set_end(self,time):
    """
    Set the GPS end time of the analysis node by setting a --gps-end-time
    option to the node when it is executed. We override the default method to
    cope with the data padding.
    @param time: GPS end time of the job.
    """
    self.add_var_opt('gps-end-time',time)
    pipeline.AnalysisNode.set_end(self,time)
    pad = int(self.job().get_config('data','pad-data'))
    pipeline.AnalysisNode.set_data_end(self,time + pad)

  def set_seed(self,seed):
    """
    Set the seed of the injection file by setting a --seed option to the
    node when it is executed.
    @param seed: seed of the job
    """
    self.add_var_opt('seed',seed)
    self.__seed = seed

  def get_output(self):
    """
    Returns the file name of output from the injection generation code. This 
    must be kept synchronized with the name of the output file in bbhinj.c.
    """
    if not self.get_start() or not self.get_end():
      raise InspiralError, "Start time or end time has not been set"
    if self.__usertag:
      bbhinject = 'HL-INJECTIONS_' + self.__usertag + '-'
      bbhinject = bbhinject + str(self.get_start()) + '-'
      bbhinject = bbhinject + str(self.get_end()-self.get_start()) + '.xml'
    elif self.__seed:
      bbhinject = 'HL-INJECTIONS_' + str(self.__seed) + '-'
      bbhinject = bbhinject + str(self.get_start()) + '-'
      bbhinject = bbhinject + str(self.get_end()-self.get_start()) + '.xml'
    else:
      bbhinject = 'HL-INJECTIONS-' + str(self.get_start()) + '-'
      bbhinject = bbhinject + str(self.get_end()-self.get_start()) + '.xml'

    self.add_output_file(bbhinject)

    return bbhinject



class TmpltBankNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A TmpltBankNode runs an instance of the template bank generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_tmpltbank.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')
    try:
      self.__zip_output = job.get_config('tmpltbank','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_start(self,time):
    """
    Set the GPS start time of the analysis node by setting a --gps-start-time
    option to the node when it is executed. We override the default method to
    cope with the data padding.
    @param time: GPS start time of job.
    """
    self.add_var_opt('gps-start-time',time)
    pipeline.AnalysisNode.set_start(self,time)
    pad = int(self.job().get_config('data','pad-data'))
    pipeline.AnalysisNode.set_data_start(self,time - pad)

  def set_end(self,time):
    """
    Set the GPS end time of the analysis node by setting a --gps-end-time
    option to the node when it is executed. We override the default method to
    cope with the data padding.
    @param time: GPS end time of job.
    """
    self.add_var_opt('gps-end-time',time)
    pipeline.AnalysisNode.set_end(self,time)
    pad = int(self.job().get_config('data','pad-data'))
    pipeline.AnalysisNode.set_data_end(self,time + pad)

  def get_output(self):
    """
    Returns the file name of output from the template bank code. This must
    be kept synchronized with the name of the output file in tmpltbank.c.
    """
    tag_base = self.job().tag_base
    if not self.get_start() or not self.get_end() or not self.get_ifo():
      raise InspiralError, "Start time, end time or ifo has not been set"
    if self.__usertag and self.get_ifo_tag():
      bank = self.get_ifo() + '-' + tag_base + '_' + self.get_ifo_tag() + "_" + self.__usertag + '-' 
      bank = bank + str(self.get_start())
    elif self.__usertag:
      bank = self.get_ifo() + '-' + tag_base + '_' + self.__usertag + '-'  
      bank = bank + str(self.get_start())
    elif self.get_ifo_tag():
      bank = self.get_ifo() + '-' + tag_base + '_' + self.get_ifo_tag() + '-'  
      bank = bank + str(self.get_start())
    else:
      bank = self.get_ifo() + '-' + tag_base + '-' + str(self.get_start())
    bank = bank + '-' + str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      bank += '.gz'

    self.add_output_file(bank)

    return bank


class RandomBankNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A RandomBankNode runs an instance of the random bank generation job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_randombank.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')

  def get_output(self):
    """
    Returns the file name of output from the template bank code. This must
    be kept synchronized with the name of the output file in randombank.c.
    """
    if not self.get_start() or not self.get_end():
      raise InspiralError, "Start time or end time has not been set"
    if self.__usertag:
      bank = 'P-TMPLTBANK_' + self.__usertag + '-' 
      bank = bank + str(self.get_start())
    else:
      bank = 'P-TMPLTBANK-' + str(self.get_start())
    bank = bank + '-' + str(self.get_end() - self.get_start()) + '.xml'

    self.add_output_file(bank)

    return bank


class SplitBankNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A SplitBankNode runs an instance of the split template bank job in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_tmpltbank.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')
    self.__bankfile = None
    self.__numbanks = None

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)
    self.__bankfile = bank

  def get_bank(self):
    return self.__bankfile

  def set_num_banks(self,numbanks):
    self.add_var_opt('number-of-banks',numbanks)
    self.__numbanks = int(numbanks)

  def get_num_banks(self):
    return self.__numbanks

  def get_output(self):
    """
    Returns a list of the file names of split banks. This must be kept
    synchronized with the name of the output files in splitbank.c.
    """
    if not self.get_bank() or not self.get_num_banks():
      raise InspiralError, "Bank file or number of banks has not been set"

    banks = []
    x = self.__bankfile.split('-')
    for i in range( 0, int(self.get_num_banks()) ):
      banks.append("%s-%s_%2.2d-%s-%s" % (x[0], x[1], i, x[2], x[3]))

    return banks


class InspiralNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  An InspiralNode runs an instance of the inspiral code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inspiral.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')
    try:
      self.__zip_output = job.get_config('inspiral','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_start(self,time):
    """
    Set the GPS start time of the analysis node by setting a --gps-start-time
    option to the node when it is executed. We override the default method to
    cope with the data padding.
    @param time: GPS start time of job.
    """
    self.add_var_opt('gps-start-time',time)
    pipeline.AnalysisNode.set_start(self,time)
    pad = int(self.job().get_config('data','pad-data'))
    pipeline.AnalysisNode.set_data_start(self,time - pad)

  def set_end(self,time):
    """
    Set the GPS end time of the analysis node by setting a --gps-end-time
    option to the node when it is executed. We override the default method to
    cope with the data padding.
    @param time: GPS end time of job.
    """
    self.add_var_opt('gps-end-time',time)
    pipeline.AnalysisNode.set_end(self,time)
    pad = int(self.job().get_config('data','pad-data'))
    pipeline.AnalysisNode.set_data_end(self,time + pad)

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)

  def set_injections(self,bbhinjct):
    self.add_var_opt('injection-file', bbhinjct)
    self.add_input_file(bbhinjct)

  def set_user_tag(self,usertag):
    self.__usertag = usertag
    self.add_var_opt('user-tag',usertag)

  def get_user_tag(self):
    return self.__usertag

  def get_output(self):
    """
    Returns the file name of output from the inspiral code. This must be kept
    synchronized with the name of the output file in inspiral.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo():
      raise InspiralError, "Start time, end time or ifo has not been set"

    tag_base = self.job().tag_base
    basename = self.get_ifo() + '-' + tag_base

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.__usertag:
      basename += '_' + self.__usertag

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      filename += '.gz'

    self.add_output_file(filename)

    return filename

  def get_froutput(self):
    """
    Returns the file name of output frame from the inspiral code. This
    must be kept synchronized with the name of the output file in inspiral.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo():
      raise InspiralError, "Start time, end time or ifo has not been set"

    tag_base = self.job().tag_base
    basename = self.get_ifo() + '-' + tag_base

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.__usertag:
      basename += '_' + self.__usertag

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.gwf'

    return filename  


class TrigToTmpltNode(pipeline.CondorDAGNode,pipeline.AnalysisNode):
  """
  A TrigToTmpltNode runs an instance of the triggered bank generator in a
  Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of inca in trigtotmplt mode.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__output = None
    self.__input_ifo = None
    self.__output_ifo = None
    self.__usertag = job.get_config('pipeline','user-tag')
    try:
      self.__zip_output = job.get_config('trigtotmplt','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_user_tag(self,usertag):
    self.__usertag = usertag
    self.add_var_opt('user-tag',usertag)

  def get_user_tag(self):
    return self.__usertag

  def make_trigbank(self,chunk,max_slide,source_ifo,dest_ifo,
    usertag=None,ifo_tag=None,zip=False):
    """
    Sets the name of triggered template bank file.
    chunk = the analysis chunk that is being 
    max_slide = the maximum length of a time slide for background estimation
    source_ifo = the name of the ifo that the triggers come from
    dest_ifo = the name of the ifo that the templates will be used for
    usertag = usertag to tag the output filename with
    ifo_tag = string to tag source interferometers, overrides the source_ifo
    for naming files
    """
    if chunk.trig_start():
      self.set_start(chunk.trig_start() - max_slide)
    else:
      self.set_start(chunk.start() - max_slide)
    if chunk.trig_end():
      self.set_end(chunk.trig_end() + max_slide)
    else:
      self.set_end(chunk.end() + max_slide)

    self.add_var_opt('ifo-a',source_ifo)

    outfile = dest_ifo + '-TRIGBANK_'
    if ifo_tag:
      outfile += ifo_tag
    else:
      outfile += source_ifo
    if usertag:
      outfile += '_' + usertag 
    outfile += '-' + str(chunk.start()) + '-' + str(chunk.dur()) + '.xml'
    if self.__zip_output:
      outfile += '.gz'
    self.__output = outfile
    self.add_var_opt('triggered-bank',outfile)

  def set_input_ifo(self,ifo):
    self.add_var_opt('input-ifo', ifo)
    self.__input_ifo = ifo

  def get_input_ifo(self):
    return self.__input_ifo

  def set_output_ifo(self,ifo):
    self.add_var_opt('output-ifo', ifo)
    self.__output_ifo = ifo

  def get_output_ifo(self):
    return self.__output_ifo

  def get_output(self):
    """
    Returns the name of the triggered template bank file.
    """
    self.add_output_file(self.__output)
    return self.__output

  def get_trig_out(self,zip=False):
    """
    Returns the name of the output file from lalapps_trigbank
    """
    if not self.get_start() or not self.get_end() or not self.get_output_ifo():
      raise InspiralError, "Start time, end time or output ifo is not set"
      
    basename = self.get_output_ifo() + '-TRIGBANK'

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.__usertag:
      basename += '_' + self.__usertag 

    trigbank_name = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      trigbank_name += '.gz'

    self.add_output_file(trigbank_name)
    return trigbank_name


class IncaNode(pipeline.CondorDAGNode,pipeline.AnalysisNode):
  """
  An IncaNode runs an instance of the inspiral coincidence code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inca.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__ifo_a = None
    self.__ifo_b = None
    self.__usertag = job.get_config('pipeline','user-tag')
    try:
      self.__zip_output = job.get_config('inca','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_ifo_a(self, ifo):
    """
    Set the interferometer code to use as IFO A.
    ifo = IFO code (e.g. L1, H1 or H2).
    """
    self.add_var_opt('ifo-a', ifo)
    self.__ifo_a = ifo

  def get_ifo_a(self):
    """
    Returns the IFO code of the primary interferometer.
    """
    return self.__ifo_a

  def set_ifo_b(self, ifo):
    """
    Set the interferometer code to use as IFO B.
    ifo = IFO code (e.g. L1, H1 or H2).
    """
    self.add_var_opt('ifo-b', ifo)
    self.__ifo_b = ifo

  def get_ifo_b(self):
    """
    Returns the IFO code of the primary interferometer.
    """
    return self.__ifo_b

  def set_user_tag(self,usertag):
    """
    Set the usertag for a given job
    """
    self.__usertag = usertag
    self.add_var_opt('user-tag',usertag)

  def get_user_tag(self):
    """
    Returns the usertag of the job
    """
    return self.__usertag

  def get_output_a(self):
    """
    Returns the file name of output from inca for ifo a. This must be kept
    synchronized with the name of the output file in inca.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo_a():
      raise InspiralError, "Start time, end time or ifo a has not been set"

    basename = self.get_ifo_a() + '-INCA'

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.__usertag:
      basename += '_' + self.__usertag 

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      filename += '.gz'

    self.add_output_file(filename)
    return filename

  def get_output_b(self):
    """
    Returns the file name of output from inca for ifo b. This must be kept
    synchronized with the name of the output file in inca.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifo_b():
      raise InspiralError, "Start time, end time or ifo a has not been set"

    basename = self.get_ifo_b() + '-INCA'

    if self.get_ifo_tag():
      basename += '_' + self.get_ifo_tag()
    if self.__usertag:
      basename += '_' + self.__usertag 

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      filename += '.gz'

    self.add_output_file(filename)
    return filename


class ThincaNode(pipeline.CondorDAGNode,pipeline.AnalysisNode):
  """
  A ThincaNode runs an instance of the inspiral coincidence code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inca.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__ifo_g1 = None
    self.__ifo_h1 = None
    self.__ifo_h2 = None
    self.__ifo_l1 = None
    self.__ifo_t1 = None
    self.__ifo_v1 = None
    self.__num_slides = None
    self.__usertag = job.get_config('pipeline','user-tag')
    self.__ifotag = None
    try:
      self.__zip_output = job.get_config('thinca','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_ifo(self, ifo):
    """
    Add the interferometer to the list of ifos
    ifo = IFO code (e.g. G1,L1, H1 or H2).
    """
    if ifo == 'G1':
      self.add_var_opt('g1-triggers','')
      self.__ifo_g1 = 'G1'
    elif ifo == 'H1':
      self.add_var_opt('h1-triggers','')
      self.__ifo_h1 = 'H1'
    elif ifo == 'H2':
      self.add_var_opt('h2-triggers','')
      self.__ifo_h2 = 'H2'
    elif ifo == 'L1':
      self.add_var_opt('l1-triggers','')
      self.__ifo_l1 = 'L1'
    elif ifo == 'T1':
      self.add_var_opt('t1-triggers','')
      self.__ifo_t1 = 'T1'
    elif ifo == 'V1':
      self.add_var_opt('v1-triggers','')
      self.__ifo_v1 = 'V1'
  
  def get_ifo_g1(self):
    """
    Returns the IFO code of g1.
    """
    return self.__ifo_g1
    
  def get_ifo_h1(self):
    """
    Returns the IFO code of h1.
    """
    return self.__ifo_h1

  def get_ifo_h2(self):
    """
    Returns the IFO code of h2.
    """
    return self.__ifo_h2

  def get_ifo_l1(self):
    """
    Returns the IFO code of l1.
    """
    return self.__ifo_l1

  def get_ifo_t1(self):
    """
    Returns the IFO code of t1.
    """
    return self.__ifo_t1

  def get_ifo_v1(self):
    """
    Returns the IFO code of v1.
    """
    return self.__ifo_v1

  def get_ifos(self):
    """
    Returns the ordered list of ifos.
    """
    ifos = ''
    if self.get_ifo_g1():
      ifos += self.get_ifo_g1()
    if self.get_ifo_h1():
      ifos += self.get_ifo_h1()
    if self.get_ifo_h2():
      ifos += self.get_ifo_h2()
    if self.get_ifo_l1():
      ifos += self.get_ifo_l1()
    if self.get_ifo_t1():
      ifos += self.get_ifo_t1()
    if self.get_ifo_v1():
      ifos += self.get_ifo_v1()

    return ifos

  def set_num_slides(self, num_slides):
    """
    Set number of time slides to undertake
    """
    self.add_var_opt('num-slides',num_slides)
    self.__num_slides = num_slides

  def get_num_slides(self):
    """
    Returns the num_slides from .ini (>0 => time slides desired)
    """
    return self.__num_slides

  def set_user_tag(self,usertag):
    """
    Set the usertag for a given job
    """
    self.__usertag = usertag
    self.add_var_opt('user-tag',usertag)

  def get_user_tag(self):
    """
    Returns the usertag of the job
    """
    return self.__usertag

  def set_ifo_tag(self,ifotag):
    """
    Set the ifotag for a given job (for second thinca)
    """
    self.__ifotag = ifotag
    self.add_var_opt('ifo-tag',ifotag)

  def get_ifo_tag(self):
    """
    Returns the ifo tag of the job
    """
    return self.__ifotag

  def get_output(self):
    """
    Returns the file name of output from thinca.  This must be kept
    synchronized with the name of the output file in thinca.c.
    """
    if not self.get_start() or not self.get_end() or not self.get_ifos():
      raise InspiralError, "Start time, end time or ifos have not been set"
    
    tag_base = self.job().tag_base
    if self.__num_slides:
      basename = self.get_ifos() + '-' + tag_base + '_SLIDE'
    else:
      basename = self.get_ifos() + '-' + tag_base

    if self.__ifotag:
      basename += '_' + self.__ifotag  

    if self.__usertag:
      basename += '_' + self.__usertag

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      filename += '.gz'

    self.add_output_file(filename)
    return filename


class SireNode(pipeline.CondorDAGNode,pipeline.AnalysisNode):
  """
  A SireNode runs an instance of the single inspiral reader code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_inca.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__ifo = None
    self.__usertag = job.get_config('pipeline','user-tag')

  def set_outputs(self,out_name,zip=False,usertag=None,cluster=None,
    slide_time=None):
    """
    Sets the name of the sire output file.
    out_name = name of sire output file
    usertag = usertag to tag the output filename with
    cluster = cluster time (ms)
    slide_time = slide time (sec)
    """
    outfile = out_name
    
    if usertag:
      outfile += '_' + usertag
    
    if cluster:
      outfile += '_CLUSTER' + str(cluster)
      
    if slide_time:
      if slide_time < 0: outfile += '_SLIDEneg' + str(abs(slide_time))
      else: outfile += '_SLIDE' + str(slide_time)
    
    summ_file = outfile + '.txt' 
    self.add_var_opt('summary',summ_file)
    
    outfile += '.xml'

    self.__output = outfile
    self.add_var_opt('output',outfile)

  def set_inj_outputs(self,out_name,inj_coinc,zip=False,usertag=None,
    cluster=None,slide_time=None):
    """
    Sets the name of the sire output file.
    out_name = name of sire output file
    inj_coinc = injection coincidence window (ms)
    usertag = usertag to tag the output filename with
    cluster = cluster time (ms)
    slide_time = slide time (sec)
    """
    outfile = out_name
    
    if usertag:
      outfile += '_' + usertag
    
    if cluster:
      outfile += '_CLUSTER' + str(cluster)
    
    if slide_time:
      if slide_time < 0: outfile += '_SLIDEneg' + str(abs(slide_time))
      else: outfile += '_SLIDE' + str(slide_time)
    
    missed_file = outfile + '_MISSED' + str(inj_coinc) + '.xml'

    self.add_var_opt('missed-injections',missed_file)
    
    outfile += '_FOUND' + str(inj_coinc)
    
    summ_file = outfile + '.txt' 
    self.add_var_opt('summary',summ_file)
    
    outfile += '.xml'

    if zip:
      outfile += '.gz'

    self.__output = outfile
    self.add_var_opt('output',outfile)
    self.add_var_opt


  def get_output(self):
    """
    Returns the name of the sire output.
    """
    self.add_output_file(self.__output)
    return self.__output


class FrJoinNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A FrJoinNode runs an instance of lalapps_frjoin in a Condor DAG
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_frjoin.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)

  def set_output(self, outputName):
    """
    Set the output name of the frame file
    @param outputName: name of the injection file created
    """
    self.add_var_opt('output',outputName)
    self.__outputName = outputName
    
  def get_output(self):
    """
    Get the output name of the frame file
    """
    return self.__outputName



class CohBankNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
  """
  A CohBankNode runs an instance of the coherent code in a Condor DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_coherent_inspiral.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    self.__usertag = job.get_config('pipeline','user-tag')
    self.__bank = None
    self.__ifos = None
    try:
      self.__zip_output = job.get_config('cohbank','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_user_tag(self,usertag):
    """
    Set the usertag for a given job
    """
    self.__usertag = usertag
    self.add_var_opt('user-tag',usertag)

  def get_user_tag(self):
    """
    Returns the usertag of the job
    """
    return self.__usertag     
    
  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)
    self.__bank = bank

  def get_bank(self):
    return self.__bank

  def set_ifos(self,ifos):
    self.add_var_opt('ifos', ifos)
    self.__ifos = ifos
   
  def get_ifos(self):
    return self.__ifos
    
  def get_output(self,zip=False):
    """
    Returns the file name of output from the coherent bank. 
    """
    
    if not self.get_ifos():
      raise InspiralError, "Ifos have not been set"
    
    basename = self.get_ifos() + '-COHBANK'

    if self.__usertag:
      basename += '_' + self.__usertag

    filename = basename + '-' + str(self.get_start()) + '-' + \
      str(self.get_end() - self.get_start()) + '.xml'

    if self.__zip_output:
      filename += '.gz'

    self.add_output_file(filename)

    return filename    


class ChiaNode(pipeline.CondorDAGNode,pipeline.AnalysisNode):
  """
  A ChiaNode runs an instance of the coherent_inspiral code in a Condor
  DAG.
  """
  def __init__(self,job):
    """
    job = A CondorDAGJob that can run an instance of lalapps_coherent_inspiral.
    """
    pipeline.CondorDAGNode.__init__(self,job)
    pipeline.AnalysisNode.__init__(self)
    try:
      self.__zip_output = job.get_config('chia','write-compress')
      self.__zip_output = True
    except:
      self.__zip_output = False

  def set_bank(self,bank):
    self.add_var_opt('bank-file', bank)
    self.add_input_file(bank)

    
##############################################################################
# some functions to make life easier later

def overlap_test(interval1, interval2, slide_sec=0):
  """
  Test whether the two intervals could possibly overlap with one of them being
  slid by a maximum time of slide_sec.  Perform three tests:
  1)  Does the start of interval 1 lie within interval 2's range (with the 
    start decremented by slide_sec and the end incremented by slide_sec)
  2)  Does the end of interval 1 lie within interval 2's range (with the start 
    decremented by slide_sec and the end incremented by slide_sec)
  3)  Does interval 1 completely cover (the extended) interval 2, 
    ie is interval_1 start before (interval 2 start - slide_sec) AND 
    interval 1 end after (interval 2 end + slide_sec)
  If any of the above conditions are satisfied then return 1, else 0.
  """
  if ( 
    interval1.start() >= interval2.start() - slide_sec
    and interval1.start() <= interval2.end() + slide_sec
    ) or (
    interval1.end() >= interval2.start() - slide_sec 
    and interval1.end() <= interval2.end() + slide_sec
    ) or (
    interval1.start() <= interval2.start() - slide_sec    
    and interval1.end() >= interval2.end() + slide_sec ):
    return 1
  else:
    return 0




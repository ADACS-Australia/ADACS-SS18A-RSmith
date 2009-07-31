#!/usr/bin/python

import sys
import os
import urlparse
from optparse import *
from subprocess import Popen,PIPE


from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils

##############################################################################
#
#          definition of the LVAlertTable
#
##############################################################################

class LVAlertTable(table.Table):
  """
  for reference, file is written as
  file: //host/path_to_file/file
  uid: the unique id assigned by gracedb
  temp_data_loc: current location (just the directory)
                 of the ouput of the pipeline (this is VOLATILE)
  """
  tableName = "LVAlert:table"
  validcolumns = {
    "file": "lstring",
    "uid": "lstring",
    "temp_data_loc": "lstring"
    }
    
class LVAlertRow(object):
  __slots__ = LVAlertTable.validcolumns.keys()
  
LVAlertTable.RowType = LVAlertRow

##############################################################################
#
#          useful utilities
#
##############################################################################

def parse_file_url(file_url):
  """
  simple function to parse the file urls of the form:
  file://host_name/path_to_file/file
  where path_to_file is assumed to be the /private subdir of a gracedb entry
  returns:
  host: host_name in the above example
  full_path: path_to_file/file in the above example
  general_dir: the /general subdir of the gracedb entry (where data not
  produced by the event supplier should be placed)
  """
  parsed = urlparse.urlparse(file_url)
  host = parsed[1]
  path, fname = os.path.split(parsed[2])
  
  return host, path, fname

def get_LVAdata_from_stdin(std_in):
  """
  this function takes an LVAlertTable from sys.stdin and it returns:
  host: the machine the payload file was created on
  full_path: the full path to (and including) the payload file
  general_dir: the directory in gracedb that the output of your code should
               be written to
  uid: the gracedb unique id associated with the event in the LVAlertTable
  """
  doc = utils.load_fileobj(std_in)[0]
  lvatable = table.get_table(doc, LVAlertTable.tableName)
  file = lvatable[0].file
  uid = lvatable[0].uid
  data_loc = lvatable[0].temp_data_loc

  return file, uid, data_loc

def get_LVAdata_from_file(filename):
  """
  this function takes the name of an xml file containing a single LVAlertTable
  and it returns:
  host: the machine the payload file was created on
  full_path: the full path to (and including) the payload file
  general_dir: the directory in gracedb that the output of your code should
               be written to
  uid: the gracedb unique id associated with the event in the LVAlertTable
  """
  doc = utils.load_filename(filename)
  lvatable = table.get_table(doc, LVAlertTable.tableName)
  file = lvatable[0].file
  uid = lvatable[0].uid
  data_loc = lvatable[0].temp_data_loc

  return file, uid, data_loc  

def make_LVAlertTable(file_url, uid, data_loc):
  """
  create xml doc which contains an LVAlert Table
  with submission file file_loc and  data located at data_loc
  """
  xmldoc = ligolw.Document()
  xmldoc.appendChild(ligolw.LIGO_LW())
  lvalerttable = lsctables.New(LVAlertTable)
  row = lvalerttable.RowType()
  row.file = file_url
  row.uid = uid
  row.temp_data_loc = data_loc
  lvalerttable.append(row)
  xmldoc.childNodes[0].appendChild(lvalerttable)

  return xmldoc


#the following is meant as a template for small jobs
#notes:
#   * we only use the vanilla universe which is appropriate for python
#    jobs and things not condor-compiled
#   * it only works for single-process jobs; anything more complicated will
#    require a dag
condor_sub_template = \
                    """
                    universe = vanilla
                    executable = macroexecutible
                    arguments = macroargs
                    log = macrolog
                    error = macroerr
                    output = macroout
                    getenv = True
                    notification = never
                    queue
                    """
def write_condor_sub(executible, args, logdir, uid):
  """
  write a simple condor submission file
  uid: unique id used in naming the files (to avoid conflicts)
  executible: the name of the executible file
  args: a list of arguments to executible
  logdir: directory to keep log files
  returns the name of the file
  """
  subfile = condor_sub_template.replace('macroexecutible', executible)\
            .replace('macroargs', args)\
            .replace('macrolog', os.path.join(logdir,str(uid)+'.log'))\
            .replace('macroerr', os.path.join(logdir,str(uid)+'.err'))\
            .replace('macroout', os.path.join(logdir,str(uid)+'.out'))
  fname = str(uid) + '.sub'
  f = open(fname,'w')
  f.write(subfile)
  f.close()

  return fname

def submit_condor_job(subfile):
  """
  submit the subfile to condor
  returns the process id
  """
  p = Popen(["condor_submit "+subfile], shell=True).pid
  
  return p


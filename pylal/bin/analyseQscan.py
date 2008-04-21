#!/usr/bin/env python
"""
Something

$Id$

This program plots the background distribution of qscan auxialiary channels
"""

__author__ = 'Romain Gouaty <romain@phys.lsu.edu>'
__date__ = '$Date$'
__version__ = '$Revision$'[11:-2]

##############################################################################
# import standard modules and append the lalapps prefix to the python path

#import matplotlib

import matplotlib
matplotlib.use('Agg')

from matplotlib.patches     import Patch
from matplotlib.axes        import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors      import normalize, Colormap
from pylab import *

import sys, getopt, os, copy, math
import socket, time
import re, string
from optparse import *
import tempfile
import ConfigParser
import urlparse
from UserDict import UserDict
sys.path.append('@PYTHONLIBDIR@')

# rc('text', usetex=False)

##############################################################################
# import the modules we need to build the pipeline

from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils
from pylal.webUtils import *
from pylal.fu_utils import *

##############################################################################
# function to check the length of the summary files (for debugging)

def checkSummaryLength(opts,inputPath): # used for debugging only

  outputPath = opts.output_path + '/' + 'qscan_length.txt'
  storeLength = open(outputPath,'w')

  listDir = os.listdir(inputPath)
  listDir.sort()
  counter = 0
  for dir in listDir:
    try:
      summary = open(inputPath + "/" + dir + "/summary.txt","r")
      summary_length = len(summary.readlines())
      counter = counter + 1
      storeLength.write(str(counter) + "\t" + dir + "\t" + str(summary_length) + "\n")
    except:
      print >> sys.stderr, "could not check file length for" + inputPath + "/" + dir + "/summary.txt"
      continue
  storeLength.close()

##############################################################################
# class to read qscan summary files

def getQscanTable(opts,type):

  summary = readSummaryFiles()

  if eval("opts.qscan_cache_" + type):
    qscanList = getPathFromCache(eval("opts.qscan_cache_" + type),type)
  else:
    try:
      inputPath = eval("opts." + type + "_input_path")
      qscanList = parseDirectoryList(inputPath)
    except:
      print >> sys.stderr, "cannot get input path for " + type
      print >> sys.stderr, "specify at least one of the following options:"
      print >> sys.stderr, "--qscan-cache, --" + type + "-input-path"
      sys.exit(1)

  table = summary.parseQscanList(qscanList)

  # perform a sanity check
  if not (len(table['channel_name']) == len(table['qscan_dir'])):
    print >> sys.stderr, "the length of channel_name does not match the length of qscan_dir in the " + type + " table"
    print >> sys.stderr, "check for data corruption in the qscan summary files in the " + type + " table"
    sys.exit(1)

  return table

def parseDirectoryList(inputPath):
  list = []
  listDir = os.listdir(inputPath)
  for dir in listDir:
    inputDir = inputPath + '/' + dir
    #list.append([inputDir])
    list.append([inputDir,None,None,None])
  return list

class readSummaryFiles:

  def __init__(self):
    self.table = {'type':[],'ifo':[],'qscan_time':[],'qscan_dir':[],'channel_name':[],'peak_time':[],'peak_frequency':[],'peak_q':[],'peak_significance':[],'peak_amplitude':[]}
    self.paramList = ['channel_name','peak_time','peak_frequency','peak_q','peak_significance','peak_amplitude']
    self.paramNames = ['channelName','peakTime','peakFrequency','peakQ','peakSignificance','peakAmplitude']
    self.paramMaps = map(None,self.paramList,self.paramNames)

  def extractFromTable(self,inputTable,type=None,ifo=None,path=None):
    tableLength = len(inputTable['channel_name'])
    table = inputTable
    if path:
      table = self.extractParam('qscan_dir',path,table,tableLength)
    if ifo:
      table = self.extractParam('ifo',ifo,table,tableLength)
    if type:
      table = self.extractParam('type',type,table,tableLength)
    return table

  def extractParam(self,param,param_value,inputTable,tableLength):
    intermediateTable = {'type':[],'ifo':[],'qscan_time':[],'qscan_dir':[],'channel_name':[],'peak_time':[],'peak_frequency':[],'peak_q':[],'peak_significance':[],'peak_amplitude':[]}
    ind_j = 0
    n_param = inputTable[param].count(param_value)
    for i in range(0,n_param,1):
      ind_j = ind_j + inputTable[param][ind_j:tableLength].index(param_value)
      for parameter in self.paramMaps:
        intermediateTable[parameter[0]].append(inputTable[parameter[0]][ind_j])
      intermediateTable['qscan_dir'].append(inputTable['qscan_dir'][ind_j])
      intermediateTable['type'].append(inputTable['type'][ind_j])
      intermediateTable['ifo'].append(inputTable['ifo'][ind_j])
      intermediateTable['qscan_time'].append(inputTable['qscan_time'][ind_j])
      ind_j = ind_j + 1
    return intermediateTable

  def parseQscanList(self,list):
    for subList in list:
      intermediate_table = self.getAuxChannels(subList)
      for param in self.paramMaps:
        self.table[param[0]] = self.table[param[0]] + intermediate_table[param[0]] 
      self.table['qscan_dir'] = self.table['qscan_dir'] + intermediate_table['qscan_dir']
      self.table['type'] = self.table['type'] + intermediate_table['type']
      self.table['qscan_time'] = self.table['qscan_time'] + intermediate_table['qscan_time']
      self.table['ifo'] = self.table['ifo'] + intermediate_table['ifo']
    return self.table

  def getAuxChannels(self,inputList):
    intermediateTable = {'type':[],'ifo':[],'qscan_time':[],'qscan_dir':[],'channel_name':[],'peak_time':[],'peak_frequency':[],'peak_q':[],'peak_significance':[],'peak_amplitude':[]}
    try:
      doc = utils.load_filename(inputList[0] + "/summary.xml")
      qscanTable = table.get_table(doc, "qscan:summary:table")
    except:
      print >> sys.stderr, "failed to read" + inputList[0] + "/summary.xml"
      return intermediateTable
    for channel in qscanTable:
      for param in self.paramMaps:
        intermediateTable[param[0]].append(eval('channel.' + param[1]))
      intermediateTable['qscan_dir'].append(inputList[0])
      #if len(inputList) == 4:
      intermediateTable['qscan_time'].append(inputList[1])
      intermediateTable['type'].append(inputList[2])
      intermediateTable['ifo'].append(inputList[3])
    return intermediateTable

##############################################################################
# functions to compute and plot histograms

def buildChannelList(table,channel,param1,param2=None):
  ind_j = 0
  dict = {param1:[]}
  if param2: dict[param2]=[]

  tableLength = len(table['channel_name'])
  n_chan = table['channel_name'].count(channel)

  for i in range(0,n_chan,1):
    ind_j = ind_j + table['channel_name'][ind_j:tableLength].index(channel)
    element1 = table[param1][ind_j]
    dict[param1].append(element1)
    if param2:
      element2 = table[param2][ind_j]
      dict[param2].append(element2)
    ind_j = ind_j + 1

  if not n_chan > 0:
    print >> sys.stderr, 'Channel ' + channel + ' not found'

  return dict


def printChannelList(opts,table,channel,param1,param2): # for debugging only

  chan_dict = buildChannelList(table,channel,param1,param2)
  fileName = channel.split(':')[0] + '_' + channel.split(':')[1] + '_' + param1 + '_' + param2 + '.txt'
  filePath = opts.output_path + '/' + fileName
  store = open(filePath,'w')
  for i,value in enumerate(chan_dict[param1]):
    store.write(str(value) + '\t' + str(chan_dict[param2][i]) + '\n')
  store.close()

def printDeltatList(opts,table,channel,param1,thr1,thr2): # for debugging only

  chan_dict = buildChannelList(table,channel,param1,'qscan_dir')
  fileName1 = channel.split(':')[0] + '_' + channel.split(':')[1] + '_deltaT_in.txt'
  filePath1 = opts.output_path + '/' + fileName1
  store1 = open(filePath1,'w')
  fileName2 = channel.split(':')[0] + '_' + channel.split(':')[1] + '_deltaT_out.txt'
  filePath2 = opts.output_path + '/' + fileName2
  store2 = open(filePath2,'w')
  for i,value in enumerate(chan_dict[param1]):
    chan_directory = chan_dict['qscan_dir'][i]
    chan_central_time = chan_directory.split('/')[-1]
    dt = value - float(chan_central_time)
    if abs(dt) < thr1:
      store1.write(str(dt) + '\t' + chan_central_time + '\n')
    if abs(dt) > thr2:
      store2.write(str(dt) + '\t' + chan_central_time + '\n')
  store1.close()
  store2.close()


def computeDeltaT(table,chan,opts,ifo):
  dt_list = []
  
  if opts.ref_dt:
    if ifo:
      refChannel = ifo + ':' + opts.ref_channel
    else:
      refChannel = opts.ref_channel
    darm_dict = buildChannelList(table,refChannel,'peak_time','qscan_dir')

  chan_dict = buildChannelList(table,chan,'peak_time','qscan_dir')  

  for i,time in enumerate(chan_dict['peak_time']):
    if time > 0.0:
      if not opts.ref_dt:
        chan_directory = chan_dict['qscan_dir'][i]
        chan_central_time = chan_directory.split('/')[-1]
        dt = time - float(chan_central_time)
        dt_list.append(dt)
      else:
        if len(darm_dict['peak_time']) > 0:
          try: 
            indx_ref = darm_dict['qscan_dir'].index(chan_dict['qscan_dir'][i])
            if float(darm_dict['peak_time'][indx_ref]) > 0.0:
              dt = time - float(darm_dict['peak_time'][indx_ref])
              dt_list.append(dt)
          except:
            print >> sys.stderr, 'GW channel could not be found in qscan ' + chan_dict['qscan_dir'][i]
            continue

  return dt_list


def makeHistogram(list,distribution,opts,percentiles=None,candidate=None):

  parameter = distribution.split('-')[0]

  # set up the bin boundaries for the histogram
  min_val = eval('opts.' + parameter + '_min')
  max_val = eval('opts.' + parameter + '_max')
  nbins = eval('opts.' + parameter + '_bins')  

  if percentiles:
    max_percentile = float( int(percentiles[2]) ) + 1.0
    max_val = max(max_val,max_percentile)
  if candidate:
    max_val = max(max_val,candidate)

  step = (float(max_val) - float(min_val))/float(nbins)

  bins = arange(min_val - step/2., max_val + step/2., step)

  if len(list):
    # compute the histogram values
    [dist, bin, info] = hist(list,bins,bottom=None,\
    align='edge', orientation='vertical', width=None)

  return dist,bin

def findPercentile(list,percentile):
  list.sort()
  index = int(percentile * len(list))
  return list[index - 1]

def findPercentileForDt(list,percentile):
  list.sort()
  index = int( (1-percentile) * len(list)) +1
  return list[index - 1]

def absList(list):
  newList = []
  for element in list:
    newList.append(abs(element))
  return newList

def computeRank(list,threshold):
  counter = 0
  for element in list:
    if element < threshold:
      counter = counter + 1
  rank = float(counter) / float(len(list))
  return rank

def computeRankForDt(list,threshold):
  counter = 0
  for element in list:
    if abs(element) > threshold:
      counter = counter + 1
  rank = float(counter) / float(len(list))
  return rank

def getAuxVsDarmList(table,chan,opts,ifo=None):
  AuxList = []
  DarmList = []
  chan_dict = buildChannelList(table,chan,'peak_significance','qscan_dir')
  if ifo:
    refChannel = ifo + ':' + opts.ref_channel
  else:
    refChannel = opts.ref_channel
  darm_dict = buildChannelList(table,refChannel,'peak_significance','qscan_dir')
  for i,z in enumerate(chan_dict['peak_significance']):
    if z > opts.z_threshold:
      try:
        indx_ref = darm_dict['qscan_dir'].index(chan_dict['qscan_dir'][i])
      except:
        print >> sys.stderr, "GW channel could not be found in qscan " + chan_dict['qscan_dir'][i]
        continue
      if darm_dict['peak_significance'][indx_ref] > 0.0:
        AuxList.append(z)
        DarmList.append(darm_dict['peak_significance'][indx_ref])
  return AuxList,DarmList


def selectSignificance(table,chan,opts):

  z_list = []
  chan_dict = buildChannelList(table,chan,'peak_significance')

  for z in chan_dict['peak_significance']:
    if z > opts.z_threshold:
     z_list.append(z)

  return z_list


def makeScatteredPlot(chan,opts,distribution,list11=None,list12=None,list21=None,list22=None,list31=None,list32=None,figNumber=None,output=None):
  p1 = None
  p2 = None
  p3 = None
  parameter = distribution.split('-')[0]
  figure(figNumber)

  ax = subplot(111)
  ax.set_xscale('log')
  ax.set_yscale('log')

  if list11 and list12:
    p1 = plot(list11,list12,linestyle='None', marker='o',\
               markerfacecolor='k', markeredgecolor='k',\
               markersize=4, markeredgewidth=0)

  if list21 and list22:
    p2 = plot(list21,list22,linestyle='None', marker='v',\
               markerfacecolor='r', markeredgecolor='r',\
               markersize=4, markeredgewidth=0)

  if list31 and list32:
    p3 = plot(list31,list32,linestyle='None', marker='s',\
               markerfacecolor='r', markeredgecolor='r',\
               markersize=8, markeredgewidth=0)

  grid()
  xlabel('Z in ' + chan,size='x-large')
  ylabel('Z in ' + 'DARM_ERR',size='x-large')
  title('Scattered plot of significance for channel: ' + chan)

  if p1 and p2 and p3:
    legend((p1,p2,p3),('background','foreground','candidate'),loc = 'upper right')
    lim = max(max(list11),max(list12),max(list21),max(list22),max(list31),max(list32))
  if p1 and p2 and not p3:
    legend((p1,p2),('background','foreground'),loc = 'upper right')
    lim = max(max(list11),max(list12),max(list21),max(list22))
  if p1 and not p2 and p3:
    legend((p1,p3),('background','candidate'),loc = 'upper right')
    lim = max(max(list11),max(list12),max(list31),max(list32))    
  if p1 and not p2 and not p3:
#    legend((p1),('background'),loc='upper right')
    lim = max(max(list11),max(list12))

  ax.set_xlim(1e0, lim*2)
  ax.set_ylim(1e0, lim*2)

  figName = chan.split(':')[0] + '-' + chan.split(':')[1] + '-' + parameter + '-scatter.png'

  if output:
    figFileName = output + '/' + figName
  else:
    figFileName = opts.output_path + '/' + figName
  savefig(figFileName)
  close(figNumber)

  return figName  


def plotHistogram(chan,opts,distribution,histoList,binList,figNumber,percentiles=None,candidate=None,candidateRank=None,output=None):

  parameter = distribution.split('-')[0]
      
  step = binList[1] - binList[0]
  counter = sum(histoList)

  xlimInf = min(binList)
  if candidate and not parameter == 'dt':
    xlimSup = max(max(binList),candidate + 2.0)
  else:
    xlimSup = max(binList)

  figure(figNumber)
  # semilogy(bins + step/2., z_dist+0.0001, 'r^',markerfacecolor="b",markersize=12)
  # plot(bins + step/2., z_dist)
  bar(binList, histoList, width=step, bottom=0)

  if percentiles:
    line1 = axvline(x=percentiles[0], ymin=0, ymax=max(histoList), color='g', label='50th percentile', linewidth=2, linestyle='--')
    line2 = axvline(x=percentiles[1], ymin=0, ymax=max(histoList), color='m', label='95th percentile', linewidth=2, linestyle='--')
    line3 = axvline(x=percentiles[2], ymin=0, ymax=max(histoList), color='r', label='99th percentile', linewidth=2, linestyle='--')
    if parameter == 'dt':
      axvline(x=-percentiles[0], ymin=0, ymax=max(histoList), color='g', label='50th percentile', linewidth=2, linestyle='--')
      axvline(x=-percentiles[1], ymin=0, ymax=max(histoList), color='m', label='95th percentile', linewidth=2, linestyle='--')
      axvline(x=-percentiles[2], ymin=0, ymax=max(histoList), color='r', label='99th percentile', linewidth=2, linestyle='--')

  if candidate:
    line0 = axvline(x=candidate, ymin=0, ymax=max(histoList), color='k', label='candidate value (%s percentile)' % (candidateRank), linewidth=2, linestyle='-')

  if percentiles and candidate:
    legend((line0,line1,line2,line3),('candidate','50%','95%','99%'),loc = 'upper right')

  if percentiles and not candidate:
    legend((line1,line2,line3),('50%','95%','99%'),loc = 'upper right')

  xlim(xlimInf,xlimSup)

  xlabel(parameter + ' value',size='large')
  # ylabel(r'#',size='x-large')
  grid()  
  title("Histogram of the " + parameter + " value for " + chan + ', Statistics = ' + str(counter))

  

  figName = chan.split(':')[0] + '-' + chan.split(':')[1] + '-' + parameter + '-distribution.png'

  if output:
    figFileName = output + '/' + figName
  else:
    figFileName = opts.output_path + '/' + figName
  savefig(figFileName) 
  close(figNumber)

  return figName


##############################################################################
#
#  MAIN PROGRAM
#
##############################################################################

######################## OPTION PARSING  #####################################
usage = """usage: %prog [options]
"""

parser = OptionParser( usage )

parser.add_option("-v", "--version",action="store_true",default=False,\
    help="print version information and exit")

parser.add_option("-i", "--qscan-id",action="store",type="string",default=None,\
    metavar=" ID",help="id of the qscan to be investigated (provided by followup_pipe)")

parser.add_option("-C", "--qscan-cache-foreground",action="store",type="string",default=None, metavar=" FILE",help="qscan cache file for foreground (generated by followup_pipe)")

parser.add_option("-j", "--qscan-cache-background",action="store",type="string",default=None, metavar=" FILE",help="qscan cache file for background (generated by followup_pipe)")

parser.add_option("-k", "--background-input-path",action="store",type="string",\
    metavar=" PATH",help="background input path (do not specify the gps times)")

parser.add_option("-f", "--foreground-input-path",action="store",type="string",\
    metavar=" PATH",help="foreground input path (specify the gps time)")

parser.add_option("-o", "--output-path",action="store",type="string",\
    metavar=" PATH",help="output path")

parser.add_option("-p", "--page",action="store",type="string",\
    metavar=" URL",help="location of the web page")

parser.add_option("-R", "--page-rel-path",action="store",type="string",\
    default=None,metavar=" URL",help="relative path")

parser.add_option("-w", "--output-web-file",action="store",type="string",\
    default=None,metavar=" URL",help="location of the sub web page")

parser.add_option("-B", "--process-background-only",action="store_true",\
    default=False,help="only compute the background distributions (does not followup any candidate)")

parser.add_option("-g", "--generate-qscan-xml",action="store_true",\
    default=False,help="Generate qscan xml file")

parser.add_option("-L", "--channel-list",action="store",type="string",\
    metavar=" FILE",help="list of channels to process if no foreground is investigated")

parser.add_option("-Z", "--plot-z-distribution",action="store_true",\
    default=False,help="plot the z distributions")

parser.add_option("-z", "--z-threshold",action="store",type="float",\
    metavar=" VALUE",help="disregard triggers below this threshold")

parser.add_option("-m", "--z-min",action="store",type="float",\
    metavar=" VALUE",help="minimum z value to be plotted")

parser.add_option("-M", "--z-max",action="store",type="float",\
    metavar=" VALUE",help="maximum z value to be plotted")

parser.add_option("-x", "--z-bins",action="store",type="float",\
    metavar=" VALUE",help="number of bins to use in z histogram")

parser.add_option("-r", "--ref-channel",action="store",type="string",\
    metavar=" VALUE",help="channel to be used as reference (for delta t calculation or scattered plots)")

parser.add_option("-T", "--plot-dt-distribution",action="store_true",\
    default=False,help="plot the delta t distributions")

parser.add_option("-t", "--ref-dt",action="store_true",default=False,\
    help="use ref-channel for dt calculation")

parser.add_option("-n", "--dt-min",action="store",type="float",\
    metavar=" VALUE",help="minimum dt value to be plotted")

parser.add_option("-N", "--dt-max",action="store",type="float",\
    metavar=" VALUE",help="maximum dt value to be plotted")

parser.add_option("-y", "--dt-bins",action="store",type="float",\
    metavar=" VALUE",help="number of bins to use in dt histogram")

parser.add_option("-S", "--plot-z-scattered",action="store_true",\
    default=False,help="plot auxiliary channel versus Darm significance")

parser.add_option("-l", "--check-length",action="store_true",\
    default=False,help="check the length of the summary txt files (for debugging only)")

parser.add_option("-c", "--create-param-list",action="store_true",\
    default=False,help="create .txt files containing the list of parameters for each channel (for debugging only)")


command_line = sys.argv[1:]
(opts,args) = parser.parse_args()

if opts.version:
  print "$ Id: makeQscanBackground.py,v 1.0 2007/08/02 18:46:00 romain Exp $"
  sys.exit(0)

####################### SANITY CHECKS #####################################


################# NOW START THE REAL WORK ##################################

# initialize some variables
figNumber = 0
currentDir = os.path.abspath('.')

# Get the list of qscan channels to be analyzed (should become optional later...). The script needs to be improved to follow this behavior: if the text file is not specified in the configuration file, then all the channels found in the summary files should be analyzed...

if opts.process_background_only:
  channelList = listFromFile(opts.channel_list)
  if len(channelList) == 0:
    print >> sys.stderr, "channel list not found"
    sys.exit(1)

# Check the summary length
if opts.check_length:
  checkSummaryLength(opts,opts.background_input_path)


# Read the qscan summary files and hold the information in memory
if opts.generate_qscan_xml:
  backgroundTable = getQscanTable(opts,"background")
  if not opts.process_background_only:
    foregroundTable = getQscanTable(opts,"foreground")


# analyse candidate qscan (loop over all the channels which have triggered at the time of the candidate)
if not opts.process_background_only:
  
  if opts.qscan_id:
    id = opts.qscan_id
    type = id.split('_')[0]
    try:
      ifo = id.split('_')[1]
    except:
      ifo = None
    try:
      time_string = id.split('_')[2]
    except:
      time_string = None

    if opts.qscan_cache_foreground:
      candidates_path = getPathFromCache(opts.qscan_cache_foreground,type,ifo,time_string)
    else:
      print >> sys.stderr, "Please specify the option --qscan-cache= FILE"
      print >> sys.stderr, "This option is required when using --qscan-id= ID"
      sys.exit(1)
  else:
    candidates_path = parseDirectoryList(opts.foreground_input_path)

  for candidate in candidates_path :

    # get output url and create output directory
    if opts.page_rel_path:
      output_url = opts.page + '/' + opts.page_rel_path
    else:
      output_url = opts.page + '/'
    outputdir = opts.output_path

    if candidate[2]:
      output_url = output_url + candidate[2] + '/'
      outputdir = outputdir + candidate[2] + '/'
      createdir(outputdir)
    if candidate[3]:
      output_url = output_url + candidate[3] + '/'
      outputdir = outputdir + candidate[3] + '/'
      createdir(outputdir)      
    if candidate[1]:
      candidate_time = candidate[1]
    else:
      candidate_time = candidate[0].split('/')[-1]
    output_url = output_url + candidate_time + '/'
    outputdir = outputdir + candidate_time + '/'
    createdir(outputdir)

    # extract the information about the candidate in the foreground table
    candidateSummary = readSummaryFiles()
    candidateTable = candidateSummary.extractFromTable(foregroundTable,candidate[2],candidate[3],candidate[0])

    # extract the information about the foreground for this specific type and ifo
    if candidate[2] or candidate[3]:
      foregroundSummary = readSummaryFiles()
      foregroundSubTable = foregroundSummary.extractFromTable(foregroundTable,candidate[2],candidate[3],None)
    else:
      foregroundSubTable = foregroundTable

    # extract the information about the background for this specific type and ifo
    if candidate[2] or candidate[3]:
      backgroundSummary = readSummaryFiles()
      backgroundSubTable = backgroundSummary.extractFromTable(backgroundTable,candidate[2].replace('foreground','background'),candidate[3],None)
    else:
      backgroundSubTable = backgroundTable

    # compute number of channels with significance above threshold in candidate qscan
    candidate_counter = 0
    for channel in candidateTable['channel_name']:
      z = selectSignificance(candidateTable,channel,opts)
      if len(z) > 0:
        candidate_counter = candidate_counter + 1

    if opts.output_web_file and opts.page_rel_path:
      webpage = WebPage("Comparison Foreground/Background versus Candidate",opts.output_web_file, opts.page + '/' + opts.page_rel_path)
      talkback = talkBack(opts.output_web_file)
    else:
      webpage = WebPage("Comparison Foreground/Background versus Candidate",outputdir + "index.html",output_url)
    n_rows = candidate_counter + 1
    n_cols = 4
    webpage.appendTable(n_rows,3,1)
    webpage.table[0].row[0].cell[0].text("Channel")
    webpage.table[0].row[0].cell[1].text("z significance")
    webpage.table[0].row[0].cell[2].text("dt (peak-time - central time)")

    row_number = 0
    for i,channel in enumerate(candidateTable['channel_name']):
      # check that the qscan parameters are not zero for this channel
      zCandidate = candidateTable['peak_significance'][i]
      timeCandidate = candidateTable['qscan_time'][i]
      tpeakCandidate = candidateTable['peak_time'][i]
      if not zCandidate > opts.z_threshold:
        continue
      # first check that the channel is found in the background
      try:
        backgroundSubTable['channel_name'].index(channel)
      except:
        continue
      if opts.plot_z_distribution:
        zList = selectSignificance(backgroundSubTable,channel,opts)
        z_candidate_rank = computeRank(zList,zCandidate)
        print channel

        # be careful: the function findPercentile is currently dangerous, as it sorts the list... this is the reason for the deepcopy
        listTempo = copy.deepcopy(zList)
        percentile_50th = findPercentile(listTempo,0.50)
        percentile_95th = findPercentile(listTempo,0.95)
        percentile_99th = findPercentile(listTempo,0.99)
        percentiles = [percentile_50th,percentile_95th,percentile_99th]

        try:
          zHisto,zBin = makeHistogram(zList,"z-distribution",opts,percentiles,zCandidate)
        except:
          print >> sys.stderr, 'could not make the z histogram for channel ' + channel
          continue

        figNumber = figNumber + 1
        zFigure = plotHistogram(channel,opts,'z-distribution',zHisto,zBin,figNumber,percentiles,zCandidate,z_candidate_rank,outputdir)

      #if opts.plot_z_scattered and not channel=='H1:LSC-DARM_ERR' :
      if opts.plot_z_scattered:
        aux_list_back,darm_list_back = getAuxVsDarmList(backgroundSubTable,channel,opts,candidate[3])
        aux_list_fore,darm_list_fore = getAuxVsDarmList(foregroundSubTable,channel,opts,candidate[3])
        aux_list_cand,darm_list_cand = getAuxVsDarmList(candidateTable,channel,opts,candidate[3])
        figNumber = figNumber + 1
        scatteredFigure = makeScatteredPlot(channel,opts,'z-distribution',aux_list_back,darm_list_back,aux_list_fore,darm_list_fore,aux_list_cand,darm_list_cand,figNumber,outputdir)

      if opts.plot_dt_distribution:
        dtList = computeDeltaT(backgroundSubTable,channel,opts,candidate[3])

        dtCandidate = tpeakCandidate - float(timeCandidate)
        dt_candidate_rank = computeRankForDt(dtList,abs(dtCandidate))

        absoluteList = absList(dtList)
        dtpercentile_50th = findPercentileForDt(absoluteList,0.50)
        dtpercentile_95th = findPercentileForDt(absoluteList,0.95)
        dtpercentile_99th = findPercentileForDt(absoluteList,0.99)
        dtpercentiles = [dtpercentile_50th,dtpercentile_95th,dtpercentile_99th]

        try:
          dtHisto,dtBin = makeHistogram(dtList,'dt-distribution',opts)
        except:
          print >> sys.stderr, 'could not make the dt histogram for channel ' + channel
          continue
        figNumber = figNumber + 1
        dtFigure = plotHistogram(channel,opts,'dt-distribution',dtHisto,dtBin,figNumber,dtpercentiles,dtCandidate,dt_candidate_rank,outputdir)

      #append the html page
      row_number = row_number + 1
      webpage.table[0].row[row_number].cell[0].text(channel)
      if opts.plot_z_distribution:
        webpage.table[0].row[row_number].cell[1].text('background median = ' + '%.3f' % percentiles[0])
        webpage.table[0].row[row_number].cell[1].text('Z = ' + '%.3f' % zCandidate)
        webpage.table[0].row[row_number].cell[1].text('Rank=' + '%.3f' % z_candidate_rank)
        webpage.table[0].row[row_number].cell[1].link(output_url + '/' + zFigure,"Z distribution,")
      if opts.plot_z_scattered:
        webpage.table[0].row[row_number].cell[1].link(output_url + '/' + scatteredFigure,"Scattered plot")
      if opts.plot_dt_distribution:
        webpage.table[0].row[row_number].cell[2].text('background median = ' + '%.3f' % dtpercentiles[0])
        webpage.table[0].row[row_number].cell[2].text('dt = ' + '%.4f' % dtCandidate)
        webpage.table[0].row[row_number].cell[2].text('Rank=' + '%.3f' % dt_candidate_rank)
        webpage.table[0].row[row_number].cell[2].link(output_url + '/' + dtFigure,"Delta t distribution")

    webpage.cleanWrite('IUL')
    if opts.output_web_file and opts.page_rel_path:
      talkback.write()

# Display the list of parameters for each channel
if opts.create_param_list:
  for channel in channelList:
    #printChannelList(opts,backgroundTable,channel,'peak_time','qscan_time')
    #printChannelList(opts,backgroundTable,channel,'peak_significance','qscan_time')
    printDeltatList(opts,backgroundTable,channel,'peak_time',0.01,0.49)


if opts.process_background_only:

  if opts.output_web_file:
    outputWebFile = opts.output_web_file
  else:
    outputWebFile = opts.output_path + '/' + "index.html" 
  webpage = WebPage("Qscan background study",outputWebFile, opts.page + '/')

  channel_counter = len(channelList)
  n_rows = channel_counter + 1
  webpage.appendTable(n_rows,3,1)
  webpage.table[0].row[0].cell[0].text("Channel")
  webpage.table[0].row[0].cell[1].text("z significance")
  webpage.table[0].row[0].cell[2].text("dt (peak-time - central time)")

  row_number = 0

  #loop over the list of channels
  for channel in channelList:
    print channel
    zList = selectSignificance(backgroundTable,channel,opts)
    if not len(zList) > 0:
      continue 
    # prepare and plot the distribution of significance
    if opts.plot_z_distribution:
      # be careful: the function findPercentile is currently dangerous, as it sorts the list... this is the reason for the deepcopy
      listTempo = copy.deepcopy(zList)
      percentile_50th = findPercentile(listTempo,0.50)
      percentile_95th = findPercentile(listTempo,0.95)
      percentile_99th = findPercentile(listTempo,0.99)
      percentiles = [percentile_50th,percentile_95th,percentile_99th]
      try:
        zHisto,zBin = makeHistogram(zList,'z-distribution',opts,percentiles)
      except:
        print >> sys.stderr, 'could not make the z histogram for channel ' + channel
        continue
      figNumber = figNumber + 1
      zFigure = plotHistogram(channel,opts,'z-distribution',zHisto,zBin,figNumber,percentiles,None,None,None)

    # plot the significance scattered plot
    if opts.plot_z_scattered:
      aux_list_back,darm_list_back = getAuxVsDarmList(backgroundTable,channel,opts)
      figNumber = figNumber + 1
      scatteredFigure = makeScatteredPlot(channel,opts,'z-distribution',aux_list_back,darm_list_back,None,None,None,None,figNumber)

    # prepare and plot the distribution of delta t
    if opts.plot_dt_distribution:
      dtList = computeDeltaT(backgroundTable,channel,opts,None)
      absoluteList = absList(dtList)
      dtpercentile_50th = findPercentileForDt(absoluteList,0.50)
      dtpercentile_95th = findPercentileForDt(absoluteList,0.95)
      dtpercentile_99th = findPercentileForDt(absoluteList,0.99)
      dtpercentiles = [dtpercentile_50th,dtpercentile_95th,dtpercentile_99th]
      try:
        dtHisto,dtBin = makeHistogram(dtList,'dt-distribution',opts)
      except:
        print >> sys.stderr, 'could not make the dt histogram for channel ' + channel 
        continue
      figNumber = figNumber + 1
      dtFigure = plotHistogram(channel,opts,'dt-distribution',dtHisto,dtBin,figNumber,dtpercentiles)

    #append the html page
    row_number = row_number + 1
    webpage.table[0].row[row_number].cell[0].text(channel)
    if opts.plot_z_distribution:
      webpage.table[0].row[row_number].cell[1].text('background median = ' + '%.3f' % percentiles[0])
      webpage.table[0].row[row_number].cell[1].text('background 95% = ' + '%.3f' % percentiles[1])
      webpage.table[0].row[row_number].cell[1].text('background 99% = ' + '%.3f' % percentiles[2])
      webpage.table[0].row[row_number].cell[1].link(opts.page + '/' + zFigure,"Z distribution,")
    if opts.plot_z_scattered:
      webpage.table[0].row[row_number].cell[1].link(opts.page + '/' + scatteredFigure,"Scattered plot")
    if opts.plot_dt_distribution:
      webpage.table[0].row[row_number].cell[2].text('background median = ' + '%.3f' % dtpercentiles[0])
      webpage.table[0].row[row_number].cell[2].text('background 95% = ' + '%.3f' % dtpercentiles[1])
      webpage.table[0].row[row_number].cell[2].text('background 99% = ' + '%.3f' % dtpercentiles[2])
      webpage.table[0].row[row_number].cell[2].link(opts.page + '/' + dtFigure,"Delta t distribution")

  webpage.cleanWrite('IUL')

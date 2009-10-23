#!/usr/bin/python
#
# Copyright (C) 2009 Cristina Valeria Torres
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
"""
This checklist script is responsible for creating a MoinMoin Wiki
page.  This checklist page is meant to replace the original HTML based
page created by the script makeCheckList.py.  The output of this
script is a text file that can be cut and paste into a MoinMoin Wiki
editor, resulting in a Wikified checklist under MoinMoin version
control.
"""
__author__  = "Cristina Valeria Torres <cristina.torres@ligo.org>"
__date__    = '$Date$'
__version__ = '$Revision$'
__prog__    = 'makeCheckListWiki.py'


####################################################################
# Read in the standard python modules and the LSC python modules
# needed to create the MoinMoin file
####################################################################
import copy
import numpy
import optparse
import os
import random
import socket
import sys
import time
import urllib
import fnmatch
import shutil
from pylal import stfu_pipe
from glue import cbcwebpage
from pylal import git_version
sys.path.append('@PYTHONLIBDIR@')




####################################################################
# Custom methods for building wiki checklists
####################################################################

def scanTreeFnMatch(parentPath='.',levels=int(100),filemask='*'):
  """
  This recurses the subdirectories of parentPath 0,1,...,N levels.  It
  returns a list of files who match FILEMASK.
  """
  matchingFiles=list()
  myTreeLocal=os.walk(os.path.normpath(parentPath))
  parentLevel=parentPath.count(os.sep)
  #Iterate down the levels
  for root,dir,files in myTreeLocal:
    if root.count(os.sep) <= parentLevel+levels:
      for myFile in [os.path.join(root,x) for x in files]:
        if fnmatch.fnmatch(myFile,filemask):
          matchingFiles.append(myFile)
  return matchingFiles

def matchFiles(fileList=None,jobString=None,instruments=None,ifos=None,time=None):
  """
  Given a list of file paths is tests this list to select the files
  related to the options given to the method.  A list of matches is
  returned.
  """
  matchList=list()
  for thisFile in fileList:
    tPath,tFile=os.path.split(thisFile.replace(",",""))
    if (tFile.__contains__(jobString) and \
        tFile.__contains__(instruments) and \
        tFile.__contains__(ifos) and \
        tFile.__contains__(time)):
      matchList.append(thisFile)
  return matchList

####################################################################
# Custom wiki class to make writing MoinMoin text simpler
####################################################################
class wiki(object):
  def __init__(self,filename="default.wiki"):
    self.file = open(filename,"w")
    self.content=list()

  def tableOfContents(self,levels="1"):
    """
    Take an integer to determine maxdepth of TOC
    """
    self.content.append("\n [[TableOfContents(%i)]] \n"%(levels))

  def image_link(self,path,webserver):
    thumb = "thumb_" + path
    command = 'convert ' + path + ' -resize 300x300 -antialias ' + thumb
    print command
    popen = subprocess.Popen(command.split())
    popen.communicate()
    status = popen.returncode
    s = '[[ImageLink('+webserver+'/'+thumb+','+webserver+'/'+path+',width=300][,alt=none])]]'
    self.file.write(s)

  def image_table(self,image_list, webserver):
    if not image_list: return
    for j, i in enumerate(image_list):
      if not (j) % 3: self.file.write("\n\n||")
      self.image_link(i, webserver)
      self.file.write("||")
    self.file.write("\n\n")
  
  def image_glob(self, pat):
    image_list = []
    for image in glob.glob(pat):
      if 'thumb' in image: continue
      else: image_list.append(image)
    image_list.sort()
    return image_list

  def insertHR(self):
    self.content.append("\n----------\n")

  def title(self,text=""):
    s = "= %s =\n"%(text.strip())
    self.content.append(s)

  def section(self,text=""):
    s = "== %s ==\n"%(text.strip())
    self.content.append(s)

  def subsection(self,text=""):
    s = "=== %s ===\n"%(text.strip())
    self.content.append(s)

  def subsubsection(self,text=""):
    s = "==== %s ====\n"%(text.strip())
    self.content.append(s)

  def putText(self,text=""):
    s = "\n%s\n"%(text.strip())
    self.content.append(s)

  def makeExternalLink(self,url="",label=""):
    """
    Returns the MoinMoin specific wikification string
    for an external link, like an HTML page.  This string
    can be inserted into the page by a putText call.
    """
    s = " [%s %s] "%(url.strip(),label.strip())
    return s

  class wikiTable(object):
    """
    Internal class to manipulate a wiki table
    Addressable as
    X[row][col]=Information
    """
    def __init__(self,rows=1,cols=1):
      if rows < 1:
        self.rows = 1
      else:
        self.rows=rows
      if cols < 1:
        self.cols = 1
      else:
        self.cols=cols
      self.data=list()
      #Create tuple object with number of rows
      blankRow=self.__rowbuilder__(self.cols)
      for rc in range(0,rows):
        self.data.append(blankRow)

    def __rowbuilder__(self,cols):
      return [list().append(x) for x in range(0,cols)]

  def insertTable(self,obj):
    """
    Pass in a wikiTable object then create the relevant
    wiki markup and place that in to the self.content
    list for writing to the file
    """
    tableContent=str("")
    if type(obj) != type(self.wikiTable()):
      raise Exception,"Expecting nested type instance of WikiTable"
    else:
      for row in range(0,obj.rows):
        for col in range(0,obj.cols):
          tableContent="%s|| %s "%(tableContent,obj.data[row][col].rstrip().lstrip())
        tableContent="%s ||\n"%(tableContent)
    tableContent="%s\n"%(tableContent)
    self.content.append(tableContent)                      

  def write(self):
    """
    Writes the contents of the wiki object to disk.
    It flushes the current wikification to disk, more
    wikification steps could be done.  If writing is
    not possible an error is raised.
    """
    try:
      self.file.writelines(self.content)
    except:
      raise
    self.content=list()

  def finish(self):
    self.file.close()  
####################################################################
# End Custom wiki class
####################################################################

####################################################################
# Custom method that represents a checklist
# the checklist is written to disk at the end of the method run
####################################################################

def prepareChecklist(wikiFilename=None,
                     wikiCoinc=None):
  """
  Method to prepare a checklist where data products are isolated in
  directory.
  """
  #
  # Check to see if wiki file with name already exists
  #
  maxCount=0
  while os.path.exists(wikiFilename) and maxCount < 10:
    sys.stdout.write("File %s already exists.\n"%\
                     os.path.split(wikiFilename)[1])
    wikiFilename=wikiFilename+".wiki"
    maxCount=maxCount+1
  #
  #Create the wikipage object etc
  #
  wikiPage=wiki(wikiFilename)
  #
  # Create top two trigger params tables
  #
  cTable=wikiPage.wikiTable(2,8)
  cTable.data=[
    ["Trigger Type",
     "Rank",
     "FAR",
     "SNR",
     "IFOS(Coinc)",
     "Instruments(Active)",
     "Coincidence Time (s)",
     "Total Mass (mSol)"
     ],
    ["%s"%(wikiCoinc.type),
     "%s"%(wikiCoinc.rank),
     "%s"%(wikiCoinc.far),
     "%s"%(wikiCoinc.snr),
     "%s"%(wikiCoinc.ifos),
     "%s"%(wikiCoinc.instruments),
     "%s"%(wikiCoinc.time),
     "%s"%(wikiCoinc.mass)
     ]
    ]
  pTable=wikiPage.wikiTable(len(wikiCoinc.sngls)+1,6)
  pTable.data[0]=[
    "IFO",
    "GPS Time(s)",
    "SNR",
    "CHISQR",
    "Mass 1",
    "Mass 2"
    ]
  for row in range(1,len(wikiCoinc.sngls)+1):
    pTable.data[row]=[
      "%s"%(wikiCoinc.sngls[row-1].ifo),
      "%s"%(wikiCoinc.sngls[row-1].time),
      "%s"%(wikiCoinc.sngls[row-1].snr),
      "%s"%(wikiCoinc.sngls[row-1].chisqr),
      "%s"%(wikiCoinc.sngls[row-1].mass1),
      "%s"%(wikiCoinc.sngls[row-1].mass2)
      ]
  #Write the tables into the Wiki object
  wikiPage.putText("Coincident Trigger Event Information\n")
  wikiPage.insertTable(cTable)
  wikiPage.putText("Corresponding Single IFO Trigger Information\n")
  wikiPage.insertTable(pTable)

  #Generate a table of contents to appear after candidate params table
  wikiPage.tableOfContents(3)

  #Begin including each checklist item as section with subsections
  wikiPage.section("Follow-up Checklist")
  #Put each checklist item
  wikiPage.subsection("Checklist Summary")
  wikiPage.subsubsection("Does this candidate pass this checklist?")
  wikiPage.subsubsection("Answer")
  wikiPage.subsubsection("Relevant Information and Comments")
  wikiPage.insertHR()
  #
  #First real checklist item
  wikiPage.subsection("#0 False Alarm Probability")
  wikiPage.subsubsection("Question")
  wikiPage.putText("What is the false alarm rate associated with this candidate?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #Additional Checklist Item
  #First real checklist item
  wikiPage.subsection("#1 Data Quality Flags")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Can the data quality flags coincident with this candidate be safely disregarded?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  #Select the DQ information to be included here
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #Additional Checklist Item
  #First real checklist item
  wikiPage.subsection("#1 Data Quality Flags")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Can the data quality flags coincident with this candidate be safely disregarded?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #Additional Checklist Item
  #First real checklist item
  wikiPage.subsection("#2 Veto Investigations")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Does the candidate survive the veto investigations performed at its time?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #Additional Checklist Item
  #First real checklist item
  wikiPage.subsection("#3 IFO Status")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are the interferometers operating normally with a reasonable level of sensitivity around the time of the candidate?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #Additional Checklist Item
  #First real checklist item
  wikiPage.subsection("#4 Candidate Appearance")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Do the Qscan figures show what we would expect for a gravitational-wave event?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#5 Seismic Plots")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Is the seismic activity insignificant around the time of the candidate?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#6 Other environmental causes")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Were the environmental disturbances (other than seismic) insignificant at the time of the candidate?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#7 Auxiliary degree of freedom")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Were the auxiliary channel transients coincident with the candidate insignificant?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#8 Electronic Log Book")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Were the instruments behaving normally according to the comments posted by the sci-mons or the operators in the e-log?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiLinkLHOlog=wikiPage.makeExternalLink(stfu_pipe.getiLogURL(myCoinc.time,"H1"),
                                           "Hanford eLog")
  wikiLinkLLOlog=wikiPage.makeExternalLink(stfu_pipe.getiLogURL(myCoinc.time,"L1"),
                                           "Livingston eLog")
  wikiPage.putText("%s\n\n%s\n"%(wikiLinkLHOlog,wikiLinkLLOlog))
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#9 Glitch Report")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Were the instruments behaving normally according to the weekly glitch report?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  if int(wikiCoinc.time) >= 875232014:
    wikiLinkGlitch=wikiPage.makeExternalLink(
      "https://www.lsc-group.phys.uwm.edu/twiki/bin/view/DetChar/GlitchStudies",
      "Glitch Reports for S6"
      )
  else:
    wikiLinkGlitch=wikiPage.makeExternalLink(
      "http://www.lsc-group.phys.uwm.edu/glitch/investigations/s5index.html#shift",
      "Glitch Reports for S5"
      )
  wikiPage.putText("%s\n"%(wikiLinkGlitch))
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#10 Snr versus time")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Is this trigger significant in a SNR versus time plot of all triggers in its analysis chunk?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#11 Parameters of the candidate")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Does the candidate have a high likelihood of being a gravitational-wave according to its parameters?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#12 Snr and Chisq")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are the SNR and CHISQ time series consistent with our expectations for a gravitational wave?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#12 Snr and Chisq")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are the SNR and CHISQ time series consistent with our expectations for a gravitational wave?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#13 Template bank veto")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Is the bank veto value consistent with our expectations for a gravitational wave?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#14 Coherent studies")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are the triggers found in multiple interferometers coherent with each other?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#15 Segmentation Stability")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Is the candidate stable against changes in segmentation?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#16 Calibration Stability")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Is the candidate stable against changes in calibration that are consistent with systematic uncertainties?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  #Additional Checklist Item
  wikiPage.subsection("#17 Frame File Validation")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Is the data used in the analysis free from corruption at the time of the candidate?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  # A complete separate section in the WIKI page
  #
  wikiPage.section("Parameter Estimation")
  wikiPage.subsection("#1 Parameters of the candidate")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Can we get more accurate information on the parameters of this candidate using MCMC or Bayesian methods?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  wikiPage.subsection("#2 Coherent follow-up")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Make a followup with coherent multi-detector code.")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  wikiPage.subsection("#3")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are the results of the Burst analysis astrophysically consistent with a possible detection?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  wikiPage.subsection("#4")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are the results of a ringdown search astrophisycally consistent with a possible detection?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  #
  wikiPage.subsection("#5 EM Triggers")
  wikiPage.subsubsection("Question")
  wikiPage.putText("Are there any EM triggers in coincidence with the candidate? Is the distance estimated from interferometer time-delays or coherent analysis consistent with electro-magnetic observations? Are the distances as measured in several instruments consistent with position information?")
  wikiPage.subsubsection("Answer")
  wikiPage.putText("Edit Here")
  wikiPage.subsubsection("Relevant Information")
  wikiPage.putText("Plots and pipeline data go here!")
  wikiPage.subsubsection("Investigator Comments")
  wikiPage.putText("Edit Here")
  wikiPage.insertHR()
  #
  # Third miscellaneous section
  #
  wikiPage.section("Follow up documentation Miscellaneous Information")
  fuDocLink=wikiPage.makeExternalLink(
      "https://ldas-jobs.ligo.caltech.edu/~ctorres/followUpLivingDoc_LAST.pdf",
      "Living Follow up document"
      )
  wikiPage.putText(fuDocLink)
  wikiPage.insertHR()
  #
  # Wikie page done write out the content
  #
  wikiPage.write()
  #Close the file
  wikiPage.finish()
#
####################################################################
# End custom method prepareChecklist
####################################################################

####################################################################
# Coinc definition
####################################################################
class coinc(object):
  """
  """
  def __init__(self,coincInfoFile=None):
    """
    """
    if os.path.exists(coincInfoFile):
      inputData=open(coincInfoFile).readlines()
    else:
      Raise,"Error coinc info file not found!"
    #First line header
    #Second line is the Coinc Information
    rawCoinc=dict()
    rawCoincKeys=list()
    rawCoincData=list()
    rawCoincKeys=[x.replace("#","") for x in inputData[0].split()]
    rawCoincData=[x.replace(" ","") for x in inputData[1].split()]
    for i in range(0,len(rawCoincKeys)):
      rawCoinc[rawCoincKeys[i]]=rawCoincData[i]
    #Setup Coinc
    self.type=str(rawCoinc["DIR"])
    self.rank=float(rawCoinc["RANK"])
    self.far=float(rawCoinc["FAR"])
    self.snr=float(rawCoinc["SNR"])
    self.ifos=str(rawCoinc["IFOS"])
    self.instruments=str(rawCoinc["INSTRUMENTS"])
    self.time=float(rawCoinc["TIME"])
    self.mass=float(rawCoinc["MASS"])
    #Remaining header for sngl information
    rawSngl=list()
    rawSnglKeys=list()
    rawSnglData=list()
    rawSnglKeys=[x.replace("#","") for x in inputData[2].split()]
    rawSnglData=[x.split() for x in inputData[3:]]
    #Setup associated sngl data
    self.sngls=list()
    for rData in rawSnglData:
      tmp=dict()
      for i in range(0,len(rData)):
        tmp[rawSnglKeys[i]]=rData[i]
      s=sngl
      s.type=str(tmp["DIR"])
      s.ifo=str(tmp["IFO"])
      s.snr=float(tmp["SNR"])
      s.chisqr=float(tmp["CHISQ"])
      s.mass1=float(tmp["MASS1"])
      s.mass2=float(tmp["MASS2"])
      self.sngls.append(s)
      del tmp

####################################################################
# Sngl definition
####################################################################
class sngl(object):
  """
  """
  type=None
  ifo=None
  time=None
  snr=None
  chisqr=None
  mass1=None
  mass2=None

  
####################################################################
# Cache file parser
####################################################################

####################################################################
# Main part of script
####################################################################
usage = """usage: %prog [options]"""
userURL="~%s/"%(os.getenv("USER"))
userHOME="%s"%(os.getenv("HOME"))
hostnameURL="http://%s/"%(socket.gethostbyaddr(socket.gethostname())[0])
#
defaultWeblink="%s%sWEBPATH"%(hostnameURL,userURL)
parser = optparse.OptionParser(usage,version=git_version.verbose_msg)
parser.add_option("-w","--webserver",action="store",type="string",\
                    default=defaultWeblink,\
                    metavar="myURL/WEBPATH", help="This sets the URL to \
associate to all the links generated in the MoinMoin Wiki file.\
 Example : %s"%defaultWeblink)

parser.add_option("-p","--publication-directory",\
                  action="store",type="string",\
                  metavar="WEBPATH", default=None,\
                  help="Set this option so that it is consistent \
with --webserver option. Example %s/WEBPATH"%userHOME)

parser.add_option("-f","--followup-directory",\
                  action="store",type="string",\
                  default=None,\
                  help="Set this to point the the parent directory of \
a follow up pipeline run.  From this location we \
will build checklists for all the events processed \
in this directory structure.")
                    
(opts,args) = parser.parse_args()
followup_directory=os.path.normpath(opts.followup_directory)
publication_directory=os.path.normpath(opts.publication_directory)
#
sourceFiles=scanTreeFnMatch(followup_directory)
for coincFile in scanTreeFnMatch(filemask="*coincEvent.info"):
  sys.stdout.write("Creating checklist for CoincEvent file:%s\n"%(coincFile))
  myCoinc=coinc(coincFile)
  myFilename="CHECKLIST_%s_%s_%s_%s.wiki"%(myCoinc.type,
                                           myCoinc.ifos,
                                           myCoinc.instruments,
                                           myCoinc.time)
  myDirectory=myFilename.rstrip(".wiki")
  #
  #Create directory for checklist in publication location
  #
  mySourcePath=followup_directory
  myDestPath=publication_directory+"/"+myDirectory+"/"
  if not os.path.exists(myDestPath):
    os.makedirs(myDestPath)
  #
  #Copy the output files associated with this trigger
  #
  sourceFiles=scanTreeFnMatch(mySourcePath,filemask="*%s*"%(myCoinc.time))
  for mySngl in myCoinc.sngls:
    sourceFiles.extend(scanTreeFnMatch(mySourcePath,filemask="*%s*%s*"%(mySngl.ifo,mySngl.time)))
  cPath=os.path.commonprefix(sourceFiles)
  for myFile in sourceFiles:
    myDestFile=myFile.replace(cPath,myDestPath)
    if not os.path.exists(os.path.split(myDestFile)[0]):
      os.makedirs(os.path.split(myDestFile)[0])
    shutil.copy2(myFile,myDestFile)
  #
  #Generate the initial wiki checklist
  #
  prepareChecklist(myDestPath+"/"+myFilename,myCoinc)
  

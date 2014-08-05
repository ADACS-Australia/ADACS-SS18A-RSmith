#
# Copyright (C) 2004, 2005 Cristina V. Torres
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with with program; see the file COPYING. If not, write to the
#  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA
#
import sys
import os
import getopt
import time
from optparse import OptionParser

"""
all the method defs used in this script
"""

def gps2float(nList2Element):
    varIn=nList2Element
    result=float(0)
    result=float(varIn[0])+(float(varIn[1])*1e-9)
    return result
#
def gps2gpsInt(nList2Element):
    varIn=nList2Element
    secPart=str(str(varIn[0]).ljust(18)).replace(' ','0')
    nanoSecPart=str(str(varIn[1]).rjust(9)).replace(' ','0')
    output=int(secPart)+int(nanoSecPart)
    return output
#
def gpsInt2float(inGPSInt):
    ans = 0
    strLen=str(inGPSInt).__len__()
    if strLen < 8:
        print 'Suspicious custom GPS time seen!'
        return float(inGPSInt)
    ans = inGPSInt*1e-9
    return ans
#
def float2gps(floatVal):
    secPart=int(floatVal)
    nanoSecPart=(floatVal%1)*1e9
    output=[]
    output.append(secPart)
    output.append(nanoSecPart)
    return output
#
def float2gpsInt(floatVal):
    secPart=str(int(floatVal))
    nanoSecPart=str(int((floatVal%1)*1e9))
    secPart=str(str(secPart).ljust(18)).replace(' ','0')
    nanoSecPart=str(str(nanoSecPart).rjust(9)).replace(' ','0')
    output=int(secPart)+int(nanoSecPart)
    return output
#
def fileDurationGPS(nStr):
    duration=[]
    nStr=os.path.basename(nStr)
    splitlist=nStr.split(':')
    startStr=str(splitlist[2]).split(',')
    stopStr=str(splitlist[4]).split(',')
    duration.append(int(stopStr[0])-int(startStr[0]))
    duration.append(int(stopStr[1])-int(startStr[1]))
    while duration[1] < 0:
        duration[0]=int(duration[0]-1)
        duration[1]=int(duration[1]+1e9)
        
    return duration
#
def fileStartGPS(nStr):
    duration=[]
    nStr=os.path.basename(nStr)
    splitlist=nStr.split(':')
    startStr=str(splitlist[2]).split(',')
    duration=startStr
    return duration
#

def fileStopGPS(nStr):
    duration=[]
    nStr=os.path.basename(nStr)
    splitlist=nStr.split(':')
    stopStr=str(splitlist[4]).split(',')
    duration=stopStr
    return duration
#


def averageFileDuration(inputFilenamelist):
    inVar=inputFilenamelist
    for newName in range(0,len(inVar)):
        inVar[newName]=os.path.basename(inVar[newName])
    avgfloat=float(0)
    avg=[0,0]
    x=[0,0]
    for mapName in inVar:
        time=fileDurationGPS(mapName)
        x[0]=x[0]+int(time[0])
        x[1]=x[1]+int(time[1])
        avg[0]=x[0]/inVar.__len__()
        avg[1]=x[1]/inVar.__len__()

    return avg
#

def filterFileList(inputFilenamelist,startFloat,stopFloat):
    inList=sortFileList(inputFilenamelist)
    indexList=[]
    if inList.__len__() <= 1:
        print 'Error building new map cache listing! Not enough maps.'
        return ''
    stopgpsInt=float2gpsInt(stopFloat)
    startgpsInt=float2gpsInt(startFloat)
    for index in range(0,inList.__len__()):
        fileStop=gps2gpsInt(fileStopGPS(inList[index]))
        fileStart=gps2gpsInt(fileStartGPS(inList[index]))
        if ((fileStop <= stopgpsInt) and (fileStart >= startgpsInt)):
            indexList.append(index)
    indexList.sort()
    #If no appropriate entries for this time interval can be found
    if indexList.__len__() < 1:
        newList=[]
        return newList
    startBound=indexList[0]
    stopBound=indexList[indexList.__len__()-1]
    #Add error check here if indexList is NULL
    if startBound > stopBound:
        print 'Error with cache filter call!'
    if startBound <=0:
            startBound=0
    elif (
         (gps2gpsInt(fileStartGPS(inList[startBound])) > startgpsInt)
         and
         (gps2gpsInt(fileStartGPS(inList[startBound-1])) < startgpsInt)
         ):
        startBound=startBound-1
    if stopBound >= inList.__len__()-1:
        stopBound = inList.__len__()-1
    elif (
        (gps2gpsInt(fileStopGPS(inList[stopBound])) < stopgpsInt)
        and
        (gps2gpsInt(fileStopGPS(inList[stopBound+1])) > stopgpsInt)
        ):
        stopBound=stopBound+1
    """Map does not cover start end points return nothing"""
    if (
        (gps2gpsInt(fileStartGPS(inList[startBound])) > startgpsInt)
        and
        (gps2gpsInt(fileStopGPS(inList[stopBound])) < stopgpsInt)
        ):
        print 'Error filtering list, our list is inappropriate!'
        print '#'
        print newList
        print '#'
        newList=[]
        return newList
    """x.__getslice__(i,i+1) --> x[i]"""
    """x[i]..[x[i],,,x[i+n] --> x.__getslice__(i,i+n)"""
    newList=inList.__getslice__(startBound,(stopBound+1))
    if (1 == 0):
        print 'Verifying the cache listing for time ',startFloat,' to ',stopFloat
        for entry in newList:
            fileStop=gps2gpsInt(fileStopGPS(entry))
            fileStart=gps2gpsInt(fileStartGPS(entry))
            testA='>>I DO NOT KNOW!<<'
            if ((
                (fileStart < startgpsInt)
                and
                (fileStop > startgpsInt)
                )
                or
                (
                (fileStart < stopgpsInt )
                and
                (fileStop > stopgpsInt )
                )):
                testA='on edge of set requested.'
            if ((
                    (fileStart >= startgpsInt)
                    and
                    (fileStop <= stopgpsInt)
                    )):
                    testA='contained in set requested.'
            print 'File: ',entry.replace('\n',''),' is ',testA
    #Make sure each entry has a newline character
    tempList=newList
    newList=[]
    for entry in tempList:
        if not entry.endswith('\n'):
            entry=entry+'\n'
        newList.append(entry)
    return newList

def cacheLength(startFileName,stopFileName):
    timeStamps=[]
    startTime=gps2gpsInt(fileStartGPS(startFileName))
    stopTime=gps2gpsInt(fileStopGPS(stopFileName))
    duration=stopTime-startTime
    return gpsInt2float(duration)
    
#
def sortFileList(inList):
    timeStamps=[]
    elm=[]
    for entry in inList:
        elm.append(gps2gpsInt(fileStartGPS(entry)))
        elm.append(entry)
        timeStamps.append(elm)
        elm=[]
    arrayList=timeStamps
    arrayList.sort()
    outputList=[]
    for entry in arrayList:
        outputList.append(entry[1])
    return outputList
    
def continousCache(inList):
    errFlag=True
    inList=sortFileList(inList)
    fileSpan=gps2gpsInt(averageFileDuration(inList))
    if inList.__len__() < 1:
        return False
    for index in range(1,inList.__len__()-1):
        setY=gps2gpsInt(fileStopGPS(inList[index-1]))
        setX=gps2gpsInt(fileStartGPS(inList[index]))
        if setY < setX:
            print 'Check this'
            print '1) ',setY,' ',inList[index-1],'2) ',setX,' ',inList[index]
            errFlag = False
    return errFlag
    
"""
End subroutines
"""

"""
Current only uses the input cache filename as a command line arguement
it assumes that we want to build an overlaped map cache for next level
of analysis
"""

#MAIN
"""
Setup command line parsing
"""
parser = OptionParser()

parser.add_option("-f", "--file", dest="filename",
                  default=".",
                  help="A complete list with paths of map objects to line sort and break into sets of cache files with the name NAME_CacheSet_GPSStartSet_GPSStopSet.cache.  This could possibly also be the path to a directory of dat files to process",
                  metavar="NAME")
parser.add_option("-t","--start_time",
                  dest="startTime",
                  default=0,
                  help="This is the time start of interest to construct our new resolution maps.  The default value is 0, we work with any map time.",
                  metavar="sec")
parser.add_option("-l","--map_set_duration",
                  dest="mapSetDuration",
                  default=1e9,
                  help="This is the time in seconds that the new resolution maps will span. This means work only on the maps that have timestamps starting after SEC and ending on or before SEC+SECSTOP.  The default value is 1e9 seconds.",
                  metavar="secstop")
parser.add_option("-n","--new_map_duration",
                  dest="newMapTime",
                  default=1,
                  help="This is the length in seconds that a map should last for.  New maps may exceed this time length in order to be collapsed properly.  Avoid using values < 1 this may cause unusual behavior.",
                  metavar="sec")
parser.add_option("-o","--overlap_maps",
                  dest="mapOverlap",default=0,
                  help="Specify the amount of time the newly created maps should overlap each other by, try either 0 or 50% of their duration.  These are the common choices.",
                  metavar="sec")
parser.add_option("-j","--job_set_size",
                  dest="setSize",default=1,
                  help="Specify the number of map caches to combine for a nodes workload, ie a set of 5 cache each of 10maps to merge in a given job.  The files are named in the following form JobSet_1.cacheSet while individual caches are named like Cacheset_Start_0.0_Stop_1.0.cache",
                  metavar="NUM")
(options,args) = parser.parse_args()
filename=str(options.filename)
startTime=float(options.startTime)
mapSetDuration=float(options.mapSetDuration)
newMapTime=float(options.newMapTime)
mapOverlap=float(options.mapOverlap)
setSize=int(options.setSize)

"""
Read in the main cache list of maps for the run checking that the name
specified isn't just a directory.  If it is a directory make a file listing
"""
if os.path.isfile(filename):
    fp=open(filename,'r')
    mapListing=fp.readlines()
else:
    mapListing=[]
    fileList=os.listdir(filename)
    for entry in fileList:
        if ((str(entry).__contains__(':.dat')) and not ((str(entry).__contains__('cache')) or str(entry).__contains__('DumpMap'))):
            mapListing.append(entry)

print 'Starting to setup map caches at time '+str(startTime)
print 'We will setup caches for all files after this time.'
print 'Maps loaded to possibly create caches for:'+str(len(mapListing))
mapListing=sortFileList(mapListing)
#Clip the map list of all entries before the specified start time
mapIndex=0
newMapListing=[]
for map in mapListing:
    if (gps2float(fileStartGPS(map)) >= float(startTime)):
        newMapListing.append(map)
mapListing=newMapListing
mapListing.sort()
print 'Done removing maps from input file which do not meet arguments specified. Only ',mapListing.__len__(),'maps remain.'

#Start createing the map cache files
#JobCacheNum
jcn=0
#CurrentTime
ct=startTime
#MapListStartTime
mlStart=gps2float(fileStartGPS(mapListing[0]))
if (ct < mlStart):
    startTime=mlStart
    ct=startTime
#MapListendTime
mlt=gps2float(fileStopGPS(mapListing[len(mapListing)-1]))
#Just in case there is more info than requested
if mlt > mapSetDuration+ct:
    mlt=mapSetDuration+ct
jobSetCache=[]
jobSetTSAList=[]
currentCache=[]
myPWD=os.path.dirname(mapListing[0])


print 'You requested new maps at '+str(newMapTime)+' seconds.'
print 'They will span ',str(startTime),' -> ',str(mlt)
while (ct <= mlt):
    currentCache=filterFileList(mapListing,ct,ct+newMapTime)
    #If there is not cache for the interval ct->ct+newMapTime then don't execute the following
    if currentCache.__len__() > 1:
        startPointGPS=fileStartGPS(currentCache[0])
        stopPointGPS=fileStopGPS(currentCache[currentCache.__len__()-1])
        filenameCC1='MAP:Start:'+str(startPointGPS[0])+','+str(startPointGPS[1])+':Stop:'+str(stopPointGPS[0])+','+str(stopPointGPS[1])+':'
        templateName=str(currentCache[0])
        tail=str(templateName.__getslice__(int(templateName.find('TF')),templateName.__len__()-1))
        filenameCC=str(filenameCC1)+str(tail)+'.cache'
        filenameTSAPrep=str(filenameCC1)+str(tail)
    
        setLength=cacheLength(currentCache[0],currentCache[currentCache.__len__()-1])
        setLength=setLength
        time.sleep(0)
        if (continousCache(currentCache) and (setLength >= newMapTime)):
            
            print 'Writing: '+filenameCC+' with length '+str(setLength)+' seconds and '+str(currentCache.__len__())+' map entries.'
            jobSetCache.append(filenameCC+str('\n'))
            jobSetTSAList.append(filenameTSAPrep+str('\n'))
            #If filename resolves to a directory we will explicity prepend the proper path to the files!
            if os.path.isdir(filename):
                tempCache=currentCache
                currentCache=[]
                for entry in tempCache:
                    currentCache.append(filename+entry)
                fp=open(filenameCC,'w')
                fp.writelines(currentCache)
                fp.close()
        else:
            print 'Error incomplete data for: '+filenameCC
            #Write out this file with appended INCOMPLETE
            #If filename resolves to a directory we will explicity prepend the proper path to the files!
            if os.path.isdir(filename):
                tempCache=currentCache
                currentCache=[]
                for entry in tempCache:
                    currentCache.append(filename+entry)
                fp=open(filenameCC+'INCOMPLETE','w')
                fp.writelines(currentCache)
                fp.close()
        overlapTime=newMapTime-mapOverlap
        if (overlapTime == 0):
            print 'Error with overlap time requested'
    ct=ct+overlapTime
print 'Writing out Jobset files.'
index1=0
counter=0
while (index1 <= jobSetCache.__len__()-1):
    counter=counter+1
    filenameJSC='JobSet_'+str(counter)+'.jobCache'
    fp=open(filenameJSC,'w')
    if index1+setSize > jobSetCache.__len__():
        stopIndex=jobSetCache.__len__()
    else:
        stopIndex=index1+setSize
    fp.writelines(jobSetCache.__getslice__(index1,stopIndex))
    fp.close()
    index1=index1+setSize
print 'Done writing Jobset files. Jobset files allow tracksearchAverager'
print ' to batch average the sets of smaller maps listed in *.cache files.'
print 'Writing new tracksearch averager(TSA) map file listings!'
print 'These are the theoretical cache files which will allow'
print ' the tracksearchMAP search to batch process the newly '
print ' averaged maps.'
index1=0
counter=0
while (index1 <= jobSetCache.__len__()-1):
    counter=counter+1
    filenameJSC='JobSet_'+str(counter)+'.cacheTSA'
    fp=open(filenameJSC,'w')
    if index1+setSize > jobSetCache.__len__():
        stopIndex=jobSetCache.__len__()
    else:
        stopIndex=index1+setSize
    subset=jobSetTSAList.__getslice__(index1,stopIndex)
    fp.writelines(subset)
    fp.close()
    index1=index1+setSize
print 'Done writing TSA map sets!'
 

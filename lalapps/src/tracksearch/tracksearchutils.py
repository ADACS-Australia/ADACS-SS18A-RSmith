#!/usr/bin/env python
"""
  * Copyright (C) 2004, 2005 Cristina V. Torres
  *
  *  This program is free software; you can redistribute it and/or modify
  *  it under the terms of the GNU General Public License as published by
  *  the Free Software Foundation; either version 2 of the License, or
  *  (at your option) any later version.
  *
  *  This program is distributed in the hope that it will be useful,
  *  but WITHOUT ANY WARRANTY; without even the implied warranty of
  *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  *  GNU General Public License for more details.
  *
  *  You should have received a copy of the GNU General Public License
  *  along with with program; see the file COPYING. If not, write to the
  *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  *  MA  02111-1307  USA
"""
__author__ = 'Cristina Torres <cristina@phys.utb.edu>'
__date__ = '$Date$'
__version__ = ''

import getopt
import math
import os
import string
import sys
import time
import copy
disableGraphics=False
import numarray
try:
    import pylab
except RuntimeError,ImportError:
    disableGraphics=True

"""
This file provides the class and methods needed to process candidate
files. These claases will provide three features.
Globbing -> Condense list of candidates down into a single larger file.
Clobber -> take two files with dissimiliar or similiar size maps
           processed removing any curves from A that live in list B.
Snapshot -> create a PGM image to use a mental snapshot of where the
            curves found reside in the TF space
"""
class kurve:
    """
    This is a class that will hold the data in primitives of python and
    provide higher level methods to questioning the curve object.
    """
    def __init__(self,curveId,length,power):
        self.curveId=int(curveId)
        self.length=int(length)
        self.power=float(power)
        self.element=list()
        self.sortedByTime=False
        self.sortedByFreq=False
    #End __init__ method

    def __len__(self):
        """
        Determine pixels in curve as an integer and returns that number.
        """
        return self.element.__len__()
    #End __len__ method

    def __timeOrderCurve__(self):
        """
        Use the time stamps to order the input data accordingly.
        LAL modules sometimes find curves in reverse order in TFR.
        We use the gpsInt time structure repsentation of the time in
        long integer form from method __makeInt__() to sort the entries.
        """
        keyList=[] # For [gpsInt.__makeInt__(),index]
        index=0
        for entry in self.element:
            keyList.append([entry[2].__makeInt__(),index])
            index=index.__add__(1)
        keyList.sort()
        sortedCurve=list()
        for entry in keyList:
            sortedCurve.append(self.element[entry[1]])
        #Swap sorted data into element field.
        self.element=sortedCurve
        #Set Boolean key to True
        self.sortedByTime=True
        self.sortedByFreq=False
    #End method __timeOrderCurve__()

    def getSymmetryFactor(self,inputPixel,weight=1):
        """
        The default of weight 1 makes the inner product weight all
        dimensions(pixels) around the point to determine symetry equal.
        If we increase the weighting factor we can make the inner product
        highly sensitivite to asymetry near the point of interest but not
        far from it.
        """
        #Determine the max information we have around our element of
        #interest
        elementIndex=0
        pixelIndex=0
        for entry in self.element:
            if inputPixel==entry:
                pixelIndex=elementIndex
            elementIndex+=1
        #Determine max num pixels we can go out from inputPixel before
        #getting to start of end of Kurve
        workLength=min([(pixelIndex-0),(elementIndex-pixelIndex)])
        Lvect=list()
        Rvect=list()
        if workLength < 2:
            #Can not determine symmetry from these points
            #The point of interest is left or right bounded only
            #It can't be symmetrical
            symmetryFactor=0
            return symmetryFactor
        LTotalVect=self.element.__getslice__(pixelIndex-workLength,pixelIndex-1)
        RTotalVect=self.element.__getslice__(pixelIndex+1,pixelIndex+workLength)
        RTotalVect.reverse()
        for entry in LTotalVect:
            Lvect.append(entry[4])
        for entry in RTotalVect:
            Rvect.append(entry[4])
        #Add in weights!
        for index in range(0,Lvect.__len__(),1):
           Lvect[index]=Lvect[index]/(weight**index)
        for index in range(0,Rvect.__len__(),1):
           Rvect[index]=Rvect[index]/(weight**index)
        LdotR=numarray.innerproduct(Lvect,Rvect)
        NormLR=math.sqrt(
            numarray.innerproduct(Lvect,Lvect)*
            numarray.innerproduct(Rvect,Rvect)
            )
        symmetryFactor=LdotR/NormLR
        return symmetryFactor
    #End def getSymmetryFactor
    
    def appendPixel(self,Row,Col,gpsStamp,Freq,Power):
        """
        Add a new pixel to our curve.  This should not be used manually in
        most cases.  It is invoked via the loadfile method in the candidateList
        class.
        Variable is a list of lists.  Each element is
        [int(row),int(col),printgpsInt(gpsStamp),float(freq),float(power)]
        """
        self.element.append([Row,Col,gpsStamp,Freq,Power])
        self.length=self.__len__()
    #End method appendPixel

    def printCurveID(self):
        """
        Method to return the curveID of the structure IDs are not required
        to be unique to the list.
        """
        return self.curveId
    #End printCurveID method
    
    def printStartGPS(self):
        """
        This method will return the start GPS time as a text string
        """
        if self.element.__len__() > 0:
            result=self.element[0][2].display()
        else:
            print "Object appears to be empty!"
            result=''
        return result
    #End method print StartGPS

    def startGPS(self):
        """
        This method will return the start GPS time as gpsInt
        """
        if self.element.__len__() > 0:
            result=self.element[0][2]
        else:
            print "Object appears to be empty!"
            result=gpsInt(0,0)
        return result
    #End method startGPS

    def printStopGPS(self):
        """
        This method will return the start GPS time as a text string
        """
        curveSize=self.element.__len__()
        if curveSize > 0:
            result=self.element[curveSize-1][2].display()
        else:
            print "Object appears to be empty!"
            result=''
        return result
    #End method print StopGPS

    def stopGPS(self):
        """
        This method will return the stop GPS time as gpsInt
        """
        curveSize=self.element.__len__()
        if curveSize > 0:
            result=self.element[curveSize-1][2]
        else:
            print "Object appears to be empty!"
            result=gpsInt(0,0)
        return result
    #End method stopGPS

    def printStartFreq(self):
        """
        This method will print the start Frequency of the curve
        """
        curveSize=self.element.__len__()
        if curveSize > 0:
            result=self.element[0][3]
        else:
            print "Object appears to be empty!"
            result=float(-1)
        return result
    #End method printStartFreq

    def printStopFreq(self):
        """
        This method will print the stop Frequency of the curve
        """
        curveSize=self.element.__len__()
        if curveSize > 0:
            result=self.element[curveSize-1][3]
        else:
            print "Object appears to be empty!"
            result=float(-1)
        return result
    #End method printStopFreq
        
    def getCandidateBandwidth(self,getBounds=bool(False)):
        """
        This methods figures out the curves maximun bandwidth and returns
        it in units of Hz.  This doesn't necessarily correlate to end
        Freq - start Freq.  If you set getBound to True in call you
        get back the bandwidth, F_min, and F_max
        """
        minFreq=0
        maxFreq=0
        listFreq=[]
        currentFreq=0
        if self.element.__len__() > 0:
            for entry in self.element:
                listFreq.append(entry[3])
            listFreq.sort()
            minFreq=listFreq[0]
            maxFreq=listFreq[listFreq.__len__()-1]
        bandWidth=maxFreq-minFreq
        #Remember bandwidth above is middle bin to middle bin must and .5 and .5 bin edges!
        if getBounds:
            return [abs(bandWidth),min([minFreq,maxFreq]),max([minFreq,maxFreq])]
        else:
            return abs(bandWidth)
    #End method getCandidateBandwidth

    def getCandidateDuration(self):
        """
        This method figures out the curves duration in seconds and returns that
        information instead of a raw pixel length.
        """
        gpsDuration=self.stopGPS().__sub__(self.startGPS())
        #Remember the above duration is from middle of pixel to pixel must add .5 + .5 pixels tail head
        return gpsDuration.getAsFloat()
    #End method getCandidateDuration

    def getCoordTriple(self,n):
        """
        This method will get the GPStime,Freq and Power of a pixel with
        index value n and return it as a three element tuple
        """
        curveSize=self.element.__len__()
        if ((n >= curveSize) and (n < 0)):
            print "Requested Index exceeds available data! :",n,curveSize
            return gpsInt(-1,0),float(-1),float(-1)
        else:
            return self.element[n][2],self.element[n][3],self.element[n][4]
    #End method getCoordTriple

    def getCoordBins(self,n):
        """
        This method will get the interger bin numbers associated with the
        curve.  These values are rather useless once your globbed together
        many coordinateList structures.
        """
        curveSize=element.__len__()
        if ((n > curveSize) and (n >= 0)):
            print "Requested Index exceeds available data! :",n
            return int(-1),int(-1)
        else:
            return element[n][0],element[n][1]
    #End method getCoordBins

    def getKurveHeader(self):
        """
        This method will return the values as n primatives to the calling
        routine: curveId,length,power
        """
        return self.curveId,self.length,self.power
        #End method getKurveHeader

    def getKurveDataBlock(self):
        """
        This method will return an copied iterable N*5 2 dimensional list.
        This method is normally invoked by writefile method in
        candidateList class (See self.appendPixel for variable structure)
        """
        return self.element
        #End method getKurveDataBlock
        
    def getKurveDataBlock_HumanReadable(self):
        """
        This method will call getKurveDataBlock() to get the information
        about the curve self.  The data will be returned as a 2D list.
        Each element is [float(time),float(freq),float(power)].
        """
        tmpVariable=[]
        for entry in self.getKurveDataBlock():
            tmpVariable.append([float(entry[2].getAsFloat()),
                                float(entry[3]),
                                float(entry[4])])
        return tmpVariable
        #End getKurveDataBlock_HumanReadable

    def getSortedByBrightness(self):
        """
        Method that gives you a sorted structure keyed by the
        individual pixel brightness for the curve. The brightest pixel
        will be the one with element [0].
        """
        currentIndex=0
        keyedList=[]
        dataBlock=self.getKurveDataBlock()
        for entry in dataBlock:
            newKey=[]
            newKey.append(entry[4])
            newKey.append(currentIndex)
            currentIndex=currentIndex.__add__(1)
            keyedList.append(newKey)
        #End loop
        keyedList.sort()
        sortedData=[]
        for entry in keyedList:
            sortedData.append(dataBlock[entry[1]])
        if sortedData.__len__() != dataBlock.__len__():
            print "Fatal error sorting a curve by pixel brightness!"
            os.abort()
        return sortedData
        #end getSortedByBrightness method
        
    def getBrightPixelAndStats(self):
        """
        Method gets a single ROW from self.getKurveDataBlock this
        element should be the maximum brightness
        gpsInt,Frequency,Power and in addition it also gives
        mean Power in that curve, std dev of the power for that curve.
        The output structure is then
        X a single lement of self.element of the form described in
        method self.appendPixel.
        [X,mean,var]
        """
        brightestPixel=self.getBrightPixel()
        #Consider migrating the mean var part to new method
        #getKurveMeanVar()
        mean,var=self.__getKurveMeanVar__()
        return [brightestPixel,mean,var]
        #End getBrightPixel method

    def getBrightPixel(self):
        """
        Give the element (pixel) from the brighest one in a Kurve
        """
        dataBlock=self.getSortedByBrightness()
        if dataBlock.__len__() < 1:
            print "Fatal Error unexpected curve length of zero for this curve!"
            os.abort()
        brightestPixel=dataBlock[dataBlock.__len__()-1]
        return brightestPixel
    #End getBrightPixel()
    
    def getCMPixel(self):
        """
        Method which returns the traits of the center of mass point
        for the kurve.  Here center of mass is 1 dim along the
        kurve. It returns a single element of a Kurve which is a list
        with entries
        [Row,Col,gpsStamp,Freq,Power]
        """
        print "Hi girl. The CM is ?"
        if not(self.sortedByTime):
            self.__timeOrderCurve__()
        M=0
        m=0
        r=0
        rCM=1
        tmpCM=0
        index=1
        for element in self.element:
            row,col,gpsStamp,Freq,Power=element
            M=M+Power
            m=Power
            r=0
            index=index+1
            tmpCM=tmpCM + (index*m)
        rCM=int(round(tmpCM/M))
        return self.element[rCM]
    #End def getCMPixel()

    def __getKurveMeanVar__(self):
        """
        This method gets the mean and var of the power in a curve
        returning both in a solution [mean,var]
        """
        pylabNotLoaded=False
        powerArray=[]
        dataBlock=self.getKurveDataBlock()
        for entry in dataBlock:
            powerArray.append(entry[4])
        try:
            mean=pylab.mean(powerArray)
        except:
            pylabNotLoaded=True
        try:
            stddev=pylab.std(powerArray)
            var=stddev*stddev
        except:
            pylabNotLoaded=True
        if pylabNotLoaded:
            mean=float(0)
            sum=float(0)
            sumSqr=float(0)
            var=float(0)
            n=0
            for entry in dataBlock:
                #Add individual power elements
                sum=sum.__add__(float(entry[4]))
                sumSqr=sumSqr + (float(entry[4])*float(entry[4]))
            n=dataBlock.__len__()
            mean=float(sum.__div__(n))
            var=float(sumSqr - float(sum*sum).__div__(n)).__div__(n-1)
        return [mean,var]
#End method __getKurveMeanVar__()
#End class kurve
        
class gpsInt:
    """
    Provides method for converting our string format to GPS of X,Y to
    a long int with implicit decimal for basic numerical manipulations
    """

    def __init__(self,gpsSeconds,gpsNanoSeconds):
        """
        Setup an instance of gpsfloat for use in the code
        """
        try:
            self.gpsSeconds=int(gpsSeconds)
            self.gpsNanoSeconds=int(gpsNanoSeconds)
        except ValueError:
            foundType1=type(gpsSeconds)
            foundType2=type(gpsNanoSeconds)
            print "Error can not cast input arguments to integer types."
            print "Found types :",foundType1,foundType2
            print "Values      :",gpsSeconds,gpsNanoSeconds
            os.abort()
    #End init method

    def __makeInt__(self):
        """
        This method attempts to express XXXXXXXXX.ZZZZZZZZZ as an integer
        object for operations like comparing and sorting rather than
        use the float representation that losses some information
        during the conversion.
        """
        secPartIn=str(self.gpsSeconds)
        nanoPartIn=str(str(self.gpsNanoSeconds).rjust(9)).replace(' ','0')
        try:
            intIn=int(secPartIn+nanoPartIn)
        except ValueError:
            print "Error with gpsInt structure. Inconsistent values in fields!"
            print "Seconds :",self.gpsSeconds," NanoSeconds :",self.gpsNanoSeconds
            if (self.gpsNanoSeconds < 0) and (self.gpsSeconds >= 0):
                print "Assuming - sign belongs on gpsSeconds field."
                self.gpsSeconds= self.gpsSeconds * -1
                self.gpsNanoSeconds = abs(self.gpsNanoSeconds)
                secPartIn=str(self.gpsSeconds)
                nanoPartIn=str(str(self.gpsNanoSeconds).rjust(9)).replace(' ','0')
                intIn=int(secPartIn+nanoPartIn)
            else:
                print "Error is not recoverable! Please check candidate files."
                os.abort()
        return intIn
    #End __makeInt__ method

    def __splitInt__(self,inputInt):
        negFlag=False
        if inputInt < 0:
            inputInt = abs(inputInt)
            negFlag=True
        strInput=str(inputInt).zfill(18)
        gpsSecondInput=int(strInput.__getslice__(0,9))
        gpsNanoInput=int(strInput.__getslice__(9,18))
        if negFlag == True:
            gpsSecondInput=-1*gpsSecondInput
        return gpsSecondInput,gpsNanoInput
    #End __splitInt__ method
    
    def __add__(self,inTime):
        """
        Add the argument gpsInt class to the calling gpsInt object ie
        x.__add__(y) like the operation x+y
        """
        inTimeInt=inTime.__makeInt__()
        selfTimeInt=self.__makeInt__()
        intAnswer=inTimeInt+selfTimeInt
        gpsSecondAnswer,gpsNanoAnswer=self.__splitInt__(intAnswer)
        return gpsInt(gpsSecondAnswer,gpsNanoAnswer)
    #End __add__ method

    def __sub__(self,inTime):
        inTimeInt=inTime.__makeInt__()
        selfTimeInt=self.__makeInt__()
        intAnswer=selfTimeInt-inTimeInt
        gpsSecondAnswer,gpsNanoAnswer=self.__splitInt__(intAnswer)
        return gpsInt(gpsSecondAnswer,gpsNanoAnswer)
    #End __sub__ method

    def __div__(self,divisor):
        secPart=self.gpsSeconds.__floordiv__(divisor)
        secRemain=self.gpsSeconds.__mod__(divisor)
        nanoSecPart=int(self.gpsNanoSeconds.__floordiv__(divisor)).__add__(secRemain)
        return gpsInt(secPart,nanoSecPart)
    #End __div__ method
    
    def display(self):
        secPartIn=str(self.gpsSeconds)
        nanoPartIn=str(str(self.gpsNanoSeconds).rjust(9)).replace(' ','0')
        result=secPartIn+'.'+nanoPartIn
        return result
    #End display method
    
    def __diskPrint__(self):
        secPartIn=str(self.gpsSeconds)
        nanoPartIn=str(str(self.gpsNanoSeconds).rjust(9)).replace(' ','0')
        result=secPartIn+','+nanoPartIn
        return result
    #End __diskPring__ method

    def __abs__(self):
        if (self.gpsSeconds < 0) and (self.gpsNanoSeconds < 0):
            print "Major weirdness with data! See ",self.gpsSeconds,self,gpsNanoSeconds
            print "If you see this something went really wrong!"
            os.abort()
        return gpsInt(abs(self.gpsSeconds),abs(self.gpsNanoSeconds))
    #End of __abs__ method

    def getAsFloat(self):
        """
        This method returns a float representation of the gps number.
        """
        return float(self.display())
    #End of getAsFloat method
#End gpsInt class

class candidateList:
    """
    Provides basic IO for manipulation of candidate lists
    """

    def __init__(self,verboseMode=False):
        self.verboseMode=verboseMode
        self.totalCount=int(0)
        self.filename=list()
        self.numFbins=int(-1)
        self.numTbins=int(-1)
        self.gpsSpan=gpsInt(-1,0)
        self.freqSpan=float(-1)
        self.freqWidth=float(0)
        self.gpsWidth=gpsInt(0,0)
        self.sorted=False
        #Should be list of objects of class kurve
        self.curves=[]
        self.summaryFormat="";
    #End init method

    def __sec2pixel__(self,seconds):
        """
        This method returns the number of rounded pixels which represents
        the requested number of seconds given via:
        numOfPixel=ceil(seconds/pixelDuration)
        """
        return int(math.ceil(seconds/self.gpsWidth.__asFloat__()))

    def __filemaskGlob__(self):
        """
        Method to use the filename is self.filename list to create a new glob
        file name to write to the disk with.
        """
        if self.filename.__len__() == 0:
            print 'Error with header information in instance, using default DefaultName.file'
            return 'DefaultName.file'
        else:
            filename='GLOB::'+os.path.basename(self.filename[0])+\
            'FileCount:'+str(self.filename.__len__())
        return filename

    def __filemaskClob__(self,clobWith):
        """
        Method to us if we want a automagic filename to write the clobber
        result file to.
        """
        clobberVictim=str(str(os.path.basename(self.filename[0])).split('.')[0])
        clobberPerp=str(os.path.basename(clobWith.filename[0])).split('.')
        filename='CLOBBER:'+clobberVictim+clobberPerp
    #end __filemaskClob__ method

    def __len__(self):
        """
        Expected interface that gives a trigger count for the current
        library.  It returns the value of self.totalCount
        """
        if self.curves.__len__() != self.totalCount:
            sys.stderr.write("Inconsistency encountered!\n")
            sys.stderr.write("Adjusting trigger count to number of records in memory.\n")
            self.totalCount=self.curves.__len__()
        return self.totalCount
    #end method __len__()
    
    def __timeOrderLibrary__(self):
        """
        This method orders the trigger using their GPS start time marker.
        This results in a easily searchable list
        """
        #Create sortable key list [gpsStart,gpsStop,index]
        keyList=[]
        index=0
        for entry in self.curves:
            if not entry.sortedByTime:
                entry.__timeOrderCurve__()
            keyList.append([entry.gpsStart.getAsFloat(),entry.gpsStop.getAsFloat(),index])
            index=index+1
        #Sort the key list.
        keyList.sort
        sortedLibrary=list()
        for entry in keyList:
            sortedLibrary.append(self.curves[entry[3]])
        #Swap sorted list for original list
        self.curves=sortedLibrary
        self.sorted=True
    #end __timeOrderLibrary__() method
    
    def loadfile(self,inputFilename):
        """
        Reads in a candidate list from the disk with a given filename
        """
        try:
            input_fp=open(inputFilename,'r')
        except IOError:
            print "File IO error"
            print "Check : ",inputFilename
            print ""
            return
        content=input_fp.readlines()
        input_fp.close()
        if content.__len__() < 1:
            print "Error no lines in file?"
            print "Check :",inputFilename
            self.totalCount=0
            print "Object memory left untouched!"
            return
        self.totalCount=int(list(content[0].split(':'))[1])
        self.filename=[str(inputFilename)]
        #Check for comment lines -> #Comment
        index2pop=[]
        for index in range(0,int(content.__len__())):
            if content[index].startswith('#'):
                index2pop.append(index)
        index2pop.reverse()
        for index in index2pop:
            content.pop(index)
        #Check for empty lines
        index2pop=[]
        for index in range(0,int(content.__len__())):
            if content[index].startswith('\n'):
                index2pop.append(index)
        index2pop.reverse()
        for index in index2pop:
            content.pop(index)
        #Simple file parity check
        if ((int(content.__len__()).__mod__(2) != 0)):
            print 'Error parsing :',self.filename," Number of data lines :",content.__len__()
            os.abort()
#        if (content.__len__() > 1) and (self.totalCount == 0):
#            print "File has ",content.__len__(),"lines with header listing ",self.totalCount," entries."
#            print inputFilename

        if (content.__len__ > 0):
             walkIndex=0
             while walkIndex < content.__len__():
                 entry=[]
                 entryHeader=[]
                 entryElement=[]
                 entryElementTXT=[]
                 entryHeader=str(list(content[walkIndex].split(':'))[1]).split(',')
                 self.curves.append(kurve(entryHeader[0],entryHeader[1],\
                                          entryHeader[2]))
                 walkIndex=walkIndex.__add__(1)
                 entryElementTXT=str(content[walkIndex].replace(';',',')).split(':')
                 walkIndex=walkIndex.__add__(1)
                 entryList=[]
                 #Parse the text line with our delimiters which designate
                 #pixels in the curve appending them to the curve object
                 for coord in entryElementTXT:
                     tmpElement=str(coord).split(',')
                     self.curves[self.curves.__len__()-1].appendPixel(\
                         int(tmpElement[0]),int(tmpElement[1]),\
                         gpsInt(tmpElement[2],tmpElement[3]),\
                         float(tmpElement[4]),float(tmpElement[5]))
                     #Reorganize curve data chronologically for storage.
                     self.curves[self.curves.__len__()-1].__timeOrderCurve__()
             #Determine the bin widths in this structure
             if self.totalCount > 0:
                 try: self.findBinWidths()
                 except ValueError:
                     print "Can not estimate the bin widths from file:",inputFilename
                     print "Assuming file is invalid! Forgeting data."
                     self.curves=[]
                     self.gpsWidth=gpsInt(0,0)
                     self.freqWidth=float(0)
             if self.totalCount != self.curves.__len__():
                 print "Possible problem, Inconsistent file :",inputFilename
        else:
            print "No candidate entries found in:",inputFilename
    #End loadfile method

    def __loadfileQuick__(self,inputFilename):
        """
        Reads in a candidate list from the disk with a given filename
        """
        try:
            if self.verboseMode:
                print "Trying to open file."
            input_fp=open(inputFilename,'r')
        except IOError:
            print "File IO error"
            print "Check : ",inputFilename
            print ""
            return
        if self.verboseMode:
            print "Open successful."
        self.filename=[str(inputFilename)]
        line=str(' ')
        curveCount=0
        try:
            spinner=progressSpinner(self.verboseMode)
            spinner.setTag('Reading')
            while line:
                line=input_fp.readline()
                if not (line.startswith('#') or line.startswith('\n')):
                    if line.startswith('Curve'):
                        [A,B,C]=str(str(line).split(':')[1]).split(',')
                        self.curves.append(kurve(A,B,C))
                        #Advance to next data line expected!
                        line=input_fp.readline()
                        if not line.__contains__(';'):
                            print "Data file seems corrupted:",inputFilename
                        tmpLine=str(line).replace(';',',').split(':')
                        for pixel in tmpLine:
                            [a,b,c,d,e,f]=str(pixel).split(',')
                            self.curves[self.curves.__len__()-1].appendPixel(\
                                int(a),int(b),gpsInt(c,d),float(e),float(f)\
                                )
                        #Reorganize curve data chronologically for storage.
                        self.curves[self.curves.__len__()-1].__timeOrderCurve__()    
                        #Update spinner character
                        curveCount+=1
                        spinner.updateSpinner()
                        del tmpLine
            spinner.closeSpinner()
        except MemoryError:
            print "Consumed maximum allow OS memory!"
            print "Candidates file to large to process properly."
            print "Only manages to load ",curveCount," lines before failing."
            print "Try using text editor or shell to break this file down into smaller sets."
            print "If using Linux/UNIX try:"
            print "grep -v # FILE.candidates > FILE2.candidates; split -n -a 2 -l ",int(curveCount/10)," FILE2.candidates"
            print "File in question is:",inputFilename
            print "Returning empty structure!"
            input_fp.close()
            del self.curves
            del line
            return 
        input_fp.close()
        if self.curves.__len__() < 1:
            print "Error no lines in file?"
            print "Check :",inputFilename
            self.totalCount=0
            print "Object memory left untouched!"
            return
        self.totalCount=int(self.curves.__len__())
        #Determine the bin widths in this structure
        if self.totalCount > 0:
            try: self.findBinWidths()
            except ValueError:
                print "Can not estimate the bin widths from file:",inputFilename
                print "Assuming file is invalid! Forgeting data."
                self.curves=[]
                self.gpsWidth=gpsInt(0,0)
                self.freqWidth=float(0)
    #End __loadfileQuick__ method

    def writefile(self,outputFilename):
        """
        Write the current candidate list structure to disk using given
        filename
        """
        output_fp=open(outputFilename,'w')
        textLine="#Total Curves:"+str(self.totalCount)+"\n"
        output_fp.write(textLine)
        output_fp.write('# Legend: Col,Row;gpsSec,gpsNanoSec,Freq,depth\n')
        output_fp.write('#\n')
        spinner=progressSpinner(self.verboseMode)
        spinner.setTag('Writing')
        for entry in self.curves:
            CurveId,Length,Power=entry.getKurveHeader()
            text="Curve number,length,power:"+str(CurveId)+','+str(Length)+','+str(Power)+'\n'
            output_fp.write(text)
            text=""
            data=[]
            for elements in entry.getKurveDataBlock():
                spinner.updateSpinner()
                data.append(str(elements[0])+','+str(elements[1])+\
                      ';'+str(elements[2].__diskPrint__())+','\
                      +str(elements[3])+','+str(elements[4]))
            output_fp.write(str(':').join(data)+'\n')
        spinner.closeSpinner()
    #End writefile method

    def sortList(self):
        """
        This sorts the input curves into an order with lowest gpsStart time
        first.  The sorting keys only on the curves start time.
        """
        if True:
            #Use the new method __timeOrderLibrary__()
            self.__timeOrderLibrary__()
        else:
            #Created keye list gpsInt.display(),OriginalIndex
            currentIndex=0
            keyedList=[]
            for entry in self.curves:
                tmpPair=[]
                tmpPair.append((entry.startGPS()).__makeInt__())
                tmpPair.append(currentIndex)
                currentIndex=currentIndex.__add__(1)
                keyedList.append(tmpPair)
            #Sort this keyed list lowest->highest
            keyedList.sort()
            newList=[]
            for entry in keyedList:
                newList.append(self.curves[entry[1]])
            if newList.__len__() != self.curves.__len__():
                print "Error with sorting routine!"
                os.abort()
            #Reassign the list to newly sorted version
            self.sorted=True
            self.curves=copy.deepcopy(newList)
    #End sortList method

    def findBinWidths(self):
        """
        Use the information loaded to determine the bin sizes.
        We use up to at most a sample of curveEstimateLimit curves to
        estimate the bin widths.
        Col->F
        Row->T
        """
        TDArrayFreq=[]
        TDArrayGPSFloat=[]
        curveEstimateLimit=5000
        decimateFactor=int(self.curves.__len__()/curveEstimateLimit)
        dataIndex=[]
        counter=0
        if decimateFactor>0:
            while counter < self.curves.__len__():
                dataIndex.append(counter)
                counter=counter+decimateFactor
        else:
            dataIndex=range(self.curves.__len__())
        for currentIndex in dataIndex:
            try:
                curveElement=self.curves[currentIndex]
            except IndexError:
                print "Current Index     :",currentIndex
                print "Length self.curves:",self.curves.__len__()
                print "File being loaded :",self.filename[0]
                if currentIndex<0:
                    print "Invalid index: Skipping bin width calculation"
                    return
                else:
                    print "Continuing!"
            for entry in curveElement.getKurveDataBlock():
                TDArrayFreq.append(entry[3])
                TDArrayGPSFloat.append(entry[2].getAsFloat())
        TDArrayFreq.sort()
        TDArrayGPSFloat.sort()
        uniqT=[]
        uniqF=[]
        for x in TDArrayFreq:
            if x not in uniqF:
                uniqF.append(x)
        for x in TDArrayGPSFloat:
            if x not in uniqT:
                uniqT.append(x)
        del  TDArrayFreq
        del  TDArrayGPSFloat
        uniqT.sort()
        uniqF.sort()
        sumVal=0
        diff_T=[]
        diff_F=[]
        for i in range(1,uniqT.__len__()):
            diff_T.append(uniqT[i]-uniqT[i-1])
        diff_T.sort()
        for i in range(1,uniqF.__len__()):
            diff_F.append(uniqF[i]-uniqF[i-1])
        diff_F.sort()
        if uniqF.__len__() < 2:
            print "Warning unable to uniquely determine f bin width!"
            self.freqWidth=0
            avgF=0
        else:
            avgF=diff_F[0]
        if uniqT.__len__() < 2:
            print "Warning unable to uniquely determine t bin width!"
            self.gpsWidth=gpsInt(0,0)
            avgT=0
        else:
            avgT=diff_T[0]
        if (uniqT.__len__() < 10):
            print "Warning less than ten intervals used to determine bin time width!"
        if (uniqF.__len__() < 10):
            print "Warning less than ten intervals used to determine bin frequency width!"
        self.freqWidth=float(avgF)
        ans=str("%9.9f"%float(avgT)).split('.')
        self.gpsWidth=gpsInt(ans[0],ans[1])
        if self.verboseMode:
            outString='Found: FRes %2.3f TRes %s'%(self.freqWidth,self.gpsWidth.display())
            print outString
    #End def newFindBinWidths
    
    def OLD_findBinWidths(self):
        """
        Use the information loaded to guess the bin size T and F
        we will not know the original map bandwidth but we
        won't really need it Col->F Row->T
        """
        TDArrayFreq=[]
        TDArrayGPS=[]
        for curveElement in self.curves:
            for entry in curveElement.getKurveDataBlock():
                TDArrayFreq.append([entry[0],entry[3]])
                TDArrayGPS.append([entry[1],entry[2]])
#        print "We will use ",TDArrayFreq.__len__()," total pixels to determine time bin width and freq bin width"
        TDArrayFreq.sort()
        TDArrayGPS.sort()
        freqSum=float(0)
        freqSumCount=0
        for i in range(1,TDArrayFreq.__len__()):
            if int(TDArrayFreq[i][0]) - int(TDArrayFreq[i-1][0]) == 1:
                AddMe=float(TDArrayFreq[i][1])-float(TDArrayFreq[i-1][1])
                AddMe=abs(AddMe)
                freqSum=freqSum.__add__(AddMe)
                freqSumCount=freqSumCount+1
        if freqSumCount != 0:
            avgFreqWidth=freqSum.__div__(freqSumCount)
        else:
            avgFreqWidth=0
        gpsSum=gpsInt(0,0)
        gpsSumCount=0
        for i in range(1,TDArrayGPS.__len__()):
            if int(TDArrayGPS[i][0]) - int(TDArrayGPS[i-1][0]) == 1:
                AddMe=TDArrayGPS[i][1].__sub__(TDArrayGPS[i-1][1])
                AddMe=AddMe.__abs__()
                print TDArrayGPS[i][1].display(),"   ",TDArrayGPS[i-1][1].display(),"    ",AddMe.display()
                gpsSum=gpsSum.__add__(AddMe)
                gpsSumCount=gpsSumCount+1

        if gpsSumCount > 0:
            avgGpsWidth=gpsSum.__div__(gpsSumCount)
        else:
            avgGpsWidth=gpsInt(0,0)

        self.freqWidth=float(avgFreqWidth)
        self.gpsWidth=avgGpsWidth.__abs__()
    #End findBinWidths method

    def globListFile(self,file1,file2):
        """
        This takes two files which are assumed to be candidateList
        data.  It then takes file2 and appends it straight to file1.
        It creates file1 if it doesn't exist.  This is a disk
        operation not data what so ever is loaded in this method call.
        """
        file1_fp=file(file1,'a')
        try:
            file2_fp=file(file2,'r')
        except IOError:
            print "File IO error"
            print "Check : ",inputFilename
            print "Glob skipped thie file!"
            print ""
            return
        try:
            file1_fp.write(file2_fp.read())
        except IOError:
            print "Error writing glob file!"
            print "Check disk space!"
            return
        return
    #End globListFile method
    
    def globList(self,inputCandidateList,force):
        """
        Take list arguement and concatinate it with the self candidate list
        only if the frequency and time bin widths match.  The boolean
        variable force if TRUE will push though not checking the
        bin width are matching.  This is viable for groups of files
        that we know have matching bin widths but are incorrectly calculated.
        """
        iCL=inputCandidateList
        globList=self
        globTolerance=1e-4
        if (force==True):
            "Forcing a glob of the data structures!"
            globTolerance=1
        zeroGPS=gpsInt(0,0)
        if self.gpsWidth == zeroGPS:
            self.gpsWidth=iCL.gpsWidth
        if iCL.gpsWidth == zeroGPS:
            iCL.gpsWidth=self.gpsWidth
        if self.freqWidth == 0:
            self.freqWidth=iCL.freqWidth
        if iCL.freqWidth == 0:
            iCL.freqWidth=self.freqWidth
        gpsDiff=self.gpsWidth.__sub__(iCL.gpsWidth)
        if ((float((self.freqWidth-iCL.freqWidth)).__abs__() < globTolerance)\
           and \
           (float(gpsDiff.display()).__abs__() < globTolerance)):
            globList.totalCount=self.totalCount + iCL.totalCount
            globList.filename.extend(iCL.filename)
            globList.curves.extend(iCL.curves)
            if (force == True):
                print "Avoiding:Bin width differences (Hz,Time):",float(self.freqWidth-iCL.freqWidth),float(self.gpsWidth.getAsFloat()-iCL.gpsWidth.getAsFloat())
            return globList
        elif ((self.freqWidth==0) or (int(self.gpsWidth.__makeInt__())==0)):
            globList.freqWidth=self.freqWidth
            globList.gpsWidth=self.gpsWidth
            globList.totalCount=self.totalCount + iCL.totalCount
            globList.filename.extend(iCL.filename)
            globList.curves.extend(iCL.curves)
            return globList
        elif ((iCL.freqWidth==0) or (int(iCL.gpsWidth.__makeInt__())==0)):
            globList.freqWidth=iCL.freqWidth
            globList.gpsWidth=iCL.gpsWidth
            globList.totalCount=self.totalCount + iCL.totalCount
            globList.filename.extend(iCL.filename)
            globList.curves.extend(iCL.curves)
            return globList
        else :
            print "Can not glob lists, due to inconsistent values!"
            print "Original List VS List to Glob"
            print globList.freqWidth,'VS',iCL.freqWidth
            print globList.gpsWidth.display(),'VS',iCL.gpsWidth.display()
            print "Returning original list with no additional entries!"
            print "Orignal list entry count:",self.totalCount," Ignored list entry count:",iCL.totalCount
            return globList
    #End globList method

    def clusterClobberWith(self,inputReferenceList):
        """
        Take instance and compare candidate entries against input reference.
        Delete from this instance any entries which fit into candidate entries
        located in the inputReferenceList given to this method.  Return the
        a reduced candidate reference instance.
        """
        iRL=copy.deepcopy(inputReferenceList)
        #Check to see if list has been sorted if not do it.
        if (self.sorted != True):
            self.sortList()
        if (iRL.sorted != True):
            iRL.sortList()
        selfStopIndex=self.totalCount
        iRLStopIndex=iRL.totalCount
        #Do the two candidate lists overlap in time?
        selfStart=(self.curves[0].startGPS()).__makeInt__()
        selfStop=(self.curves[self.curves.__len__()-1].stopGPS()).__makeInt__()
        iRLStart=(self.curves[0].startGPS()).__makeInt__()
        iRLStop=(self.curves[self.curves.__len__()-1].stopGPS()).__makeInt__()
        if not((selfStop >= iRLStart) or (iRLStop >= selfStart)):
            print "It appears as if these two candidateList instances do not have overlapping data!"
            print "Self's    Boundaries: ",selfStart,selfStop
            print "Clobber's Boundaries: ",iRLStart,iRLStop
            print "Returning copy of the original instance."
            return copy.deepcopy(self)
        #Since lists overlap start processing candidates.
        #Setup bin sizes
        hgw=halfGPSwidth=iRL.gpsWidth.__div__(2)
        hfw=halfFreqWidth=iRL.freqWidth.__div__(2)
        candidateToRemove=[]
        trialCount=0
        for i in range(0,self.totalCount):
            #from kurve i find all j curves that equal or span that time interval
            #build a list of those curves indexs to analyze
            #print "Checking i index:",i
            iRLmatchIndex=[]
            for j in range(0,iRL.totalCount):
                if (iRL.curves[j].startGPS().__makeInt__() <= self.curves[i].startGPS().__makeInt__())\
                   and \
                   (iRL.curves[j].stopGPS().__makeInt__() >= self.curves[i].stopGPS().__makeInt__()):
                    iRLmatchIndex.append(j)
            for j in iRLmatchIndex:
                #print "Curve Indexes self,iRL",self.curves[i].printCurveID(),iRL.curves[j].printCurveID(),"Index :",i,j
                #Build the self TF list
                selfCurve=[]
                for index in range(0,self.curves[i].__len__()):
                    tmpT,tmpF,JNK=self.curves[i].getCoordTriple(index)
                    selfCurve.append([tmpT,tmpF])
                #Build the iRL TF list
                irlCurve=[]
                for index in range(0,iRL.curves[j].__len__()):
                    tmpT,tmpF,JNK=iRL.curves[j].getCoordTriple(index)
                    irlCurve.append([tmpT,tmpF])
                boundPixels=0
                kStop=selfCurve.__len__()
                lStop=irlCurve.__len__()
                k=l=0
                while ((l < lStop) and (k < kStop)):
                    lastK=k
                    lastL=l
                    #TimeStamp Variables
                    A=irlCurve[l][0].__sub__(hgw).__makeInt__()
                    B=selfCurve[k][0].__makeInt__()
                    C=irlCurve[l][0].__add__(hgw).__makeInt__()
                    #FreqStamp Variables
                    X=irlCurve[l][1].__sub__(hfw)
                    Y=selfCurve[k][1]
                    Z=irlCurve[l][1].__add__(hfw)
                    #print "B->",boundPixels,"k,l->",k,l," : ",B," ",A<B,irlCurve[l][0].__makeInt__(),B<C," : ",Y," ",X<Y,irlCurve[l][1],Y<Z
                    if (A < B <= C):
                        k=k+1
                        if (X < Y <= Z):
                            boundPixels=boundPixels+1
                        if (l <lStop-1):
                            if (irlCurve[l][0].__makeInt__() == irlCurve[l+1][0].__makeInt__()):
                                l=l+1
                    elif (B < A):
                        k=k+1
                    else:
                        l=l+1
                fracFound=float(boundPixels).__div__(self.curves[i].length)
                #print "Matching pixels found:",boundPixels," of ",self.curves[i].length," Frac found :",fracFound,"Comparing CurveIDs:",self.curves[i].printCurveID(),iRL.curves[j].printCurveID()
                if (boundPixels == self.curves[i].length):
                    #We remove that candidate from the list
                    candidateToRemove.append(i)
        #Now just copy the candidates which were not found in the clobber
        #listing.
        resultList=copy.deepcopy(self)
        resultList.filename=["ClobberedList"]
        resultList.curves=[]
        for index in range(0,self.curves.__len__()):
            if not candidateToRemove.__contains__(index):
                resultList.curves.append(self.curves[index])
        resultList.totalCount=resultList.curves.__len__()
        #Send back the results
        #print "Keeping :",resultList.totalCount," Removing :",candidateToRemove.__len__(),"Total Candidates Checked :",self.totalCount
        return resultList
    #End clusterClobberWith method

    def coincidenceTimeCheck(self,secondCurveLibrary,tOffset=0.001):
        """
        It compares two trigger libraries self->A and other->B  it
        searches B for matching triggers in A and returns those
        matches.  It must be given a maximum time offset to make the
        decision if two individual triggers are coincident in
        time. The default will be the light travel time between H1 and
        L1 of 10ms. It returns the matches in time coincidence for
        A) self
        B) secondCurveLibrary
        C) the smaller of the two the lowest rate since triggers can break
        into smaller triggers which are related to the
        one larger trigger, (fewer triggers).
        [A,B,C]
        """
        scl=secondCurveLibrary
        #Check to see if lists are time sorted.
        if not self.sorted:
            self.__timeOrderLibrary__()
        if not scl.sorted:
            scl.__timeOrderLibrary__()
        #Break both libraries into smaller lists of libraries for searching
        #self -> list(libA)
        #scl  -> list(libB)
        statsA=self.candidateStats()
        libA_start=statsA[1]
        libA_stop=statsA[0]
        statsB=slc.candidateStats()
        libB_start=statsB[1]
        libB_stop=statsB[0]
        #Crop the non-overlapping sections of both libraries
        cropStart=min(libA_start,libB_start)
        cropStop=max(libA_stop,libB_start)
        libA_CropT=libA.applyArbitraryThresholds("(T>=%f)and(S>=T)"%(cropStart))
        libB_CropT=libB.applyArbitraryThresholds("(T>=%f)and(S>=T)"%(cropStart))
        libA_CropFinal=libA_CropT.applyArbitraryThresholds("(S<=%f)and(T<=S)"%(cropStop))
        libB_CropFinal=libB_CropT.applyArbitraryThresholds("(S<=%f)and(T<=S)"%(cropStop))
        libA=libA_CropFinal
        libB=libB_CropFinal
        #Compare overlapping parts of libA to libB
        #Get list for libA of [index,gpsStart,gpsStop] and libB of [index,gpsStart,gpsStop]
        curvesA=list()
        curvesB=list()
        counter=0
        for entry in libA.curves:
            curvesA.append([counter,entry.startGPS.getAsFloat(),entry.stopGPS.getAsFloat()])
            counter=counter+1
        counter=0
        for entry in libB.curves:
            curvesB.append([counter,entry.startGPS.getAsFloat(),entry.stopGPS.getAsFloat()])
            counter=counter+1
        #Sort this keyed list
        curvesA.sort()
        curvesB.sort()
        resultA=copy.deepcopy(libA)
        resultB=copy.deepcopy(libB)
        del resultA.curves
        del resultB.curves
        for keyA in curvesA:
            keyA[1]=keyA[1]-tOffset
            keyA[2]=keyA[2]+tOffset
            cutIndex_start=0
            cutIndex_stop=0
            k1=0
            k3=curvesB.__len__()
            k2=int(k3/2)+k1
            doThis=True
            while doThis:
                if ((keyA[1] >= curvesB[k2][1]) and (keyA[2] >=curvesB[k2][2])):
                    k2=k2-int((k2-k1)/2)
                elif ((keyA[1] <= curvesB[k2][1]) and (keyA[2] <= curvesB[k2][2])):
                    k2=k2+int((k3-k2)/2)
                else:
                    doThis=False
                    cutIndex_start=k2
            k1=0
            k3=curvesB.__len__()
            k2=int(k3/2)+k1
            doThis=True
            while doThis:
                if ((keyA[2] <= curvesB[k2][1]) and (keyA[1] <=curvesB[k2][1])):
                    k2=k2+int((k2-k1)/2)
                elif ((keyA[2] >= curvesB[k2][2]) and (keyA[1] >= curvesB[k2][1])):
                    k2=k2-int((k3-k2)/2)
                else:
                    doThis=False
                    cutIndex_stop=k2
            #Use the cutIndexs to cut out all the matchin entries from libB
            thisSlice=[]
            thisSlice=libB.curves.__getslice__(cutIndex_start,cutIndex_stop)
            resultB.curves.extend(thisSlice)
            #Record the entry as a match for library A
            if thisSlice.__len__() > 0:
                resultA.curves.append(entry)
            del thisSlice
        #Sort the coincidence lists for the this operation
        #Still need to write __timeOrderLibrary__() method
        resultB.__timeOrderLibrary__()
        resultA.__timeOrderLibrary__()
        #Return as resultC the smallest of the two trigger libraries.
        #Since this is a more honest method to measure trigger rates.
        if resultA.__len__() > resultB.__len__():
            resultC=copy.deepcopy(resultA)
        else:
            resultC=copy.deepcopy(resultB)
        return [resultA,resultB,resultC]
    #END conincidenceTimeCheck method

    def coincidenceGetUnion(self,secondCurveLibrary):
        """
        Looks for matching elements the Union of both libraries.  It
        returns identical elements.  Useful for comparing a time
        coincident processed library with a frequency coincident
        processed library.
        """
        print "HI"
    #END conincidenceGetUnion
    def candidateStats(self):
        """
        This method calculates stats for the candidate list such as number
        of total curves.  The average curve power and length found and
        their standard deviations.
        Output is a list like
        [liststartT,liststopT,curveCount,AvgP,StdevP,AvgL,StdevL]
        """
        maxP=float(0)
        maxL=int(0)
        Lsum=float(0)
        Psum=float(0)
        LsumSqr=float(0)
        PsumSqr=float(0)
        curveCount=[]
        gpsStamps=[]
        for entry in self.curves:
            gpsStamps.append(entry.printStartGPS())
            curveCount.append(entry.printCurveID())
            if int(entry.length)>maxL:
                maxL=int(entry.length)
            if float(entry.power)>maxP:
                maxP=float(entry.power)
            Lsum=Lsum.__add__(float(entry.length))
            LsumSqr=LsumSqr + (float(entry.length)*float(entry.length))
            Psum=Psum.__add__(float(entry.power))
            PsumSqr=PsumSqr + (float(entry.power)*float(entry.power))
        if (curveCount.__len__() < 1):
            print "Stats can not be performed on candidateless candidateList instance!."
            return []
        gpsStamps.sort()
        meanL=meanP=float(0)
        varL=varP=float(0)
        n=curveCount.__len__()
        meanL=Lsum.__div__(n)
        varL=float(LsumSqr - float(Lsum*Lsum).__div__(n)).__div__(n)
        meanP=Psum.__div__(curveCount.__len__())
        varP=float(PsumSqr - float(Psum*Psum).__div__(n)).__div__(n)
        if gpsStamps.__len__() == 0:
            startT=0
            stopT=0
        else:
            startT=gpsStamps[0]
            stopT=gpsStamps[gpsStamps.__len__()-1]
        statList=[startT,stopT,curveCount.__len__(),meanP,varP,meanL,varL,maxP,maxL]
        return statList
    #End candidateStats method

    def candidateStatsFromFile(self,inputFilename):
        """
        This method just scans the candidate list for information like
        integrated power in a curve and curve length in pixels.  It
        then calculates the same stats as method candidateStats() but
        loading the data from the file but creating data structures.
        """
        Lsum=float(0)
        Psum=float(0)
        LsumSqr=float(0)
        PsumSqr=float(0)
        curveCount=int(0)
        curveNum=float(0)
        L=float(0)
        P=float(0)
        #Load the files summary line data
        input_fp=open(inputFilename,'r')
        content=input_fp.readlines()
        input_fp.close()
        #Select out just the summary lines
        for line in content:
            if line.startswith('Curve'):
                try:
                    (curveNum,L,P)=str(line.split(':')[1]).split(',')
                except IndexError:
                    print "Problem with file:",inputFilename
                    print "Returning zeroes for this file stats!"
                    return [0,0,0,0,0,0,0]
                L=float(L)
                P=float(P)
                Lsum=Lsum+L
                Psum=Psum+P
                LsumSqr=LsumSqr+(L*L)
                PsumSqr=PsumSqr+(P*P)
                curveCount=curveCount+1
        meanL=meanP=float(0)
        varL=varP=float(0)
        n=curveCount
        if curveCount < 1:
            return [0,0,0,0,0,0,0]
        meanL=Lsum.__div__(n)
        varL=float(LsumSqr - float(Lsum*Lsum).__div__(n)).__div__(n)
        meanP=Psum.__div__(n)
        varP=float(PsumSqr - float(Psum*Psum).__div__(n)).__div__(n)
        statList=[0,0,n,meanP,varP,meanL,varL]
        return statList
    #end method candidateStatsFromFile
    
    def candidateStatsOnDisk(self,inputFilename):
        """
        This method just scans the candidate list for information like
        integrated power in a curve and curve length in pixels.  It
        then calculates the same stats as method candidateStats() but
        parses the files a line a time constructing stat information
        on the fly with no data loading what so ever! This is slower
        than other stat methods but it does not eat up the memory.
        It also returns two other values the max P and max L found!
        """
        maxL=int(0)
        maxP=int(0)
        Lsum=float(0)
        Psum=float(0)
        LsumSqr=float(0)
        PsumSqr=float(0)
        curveCount=int(0)
        curveNum=float(0)
        L=float(0)
        P=float(0)
        #Create the file pointer and try to open it
        try:
            line=str('start')
            input_fp=open(inputFilename,'r')
        except IOError:
            print "File IO error"
            print "Check : ",inputFilename
            print ""
            line=str('')
            return []
        #Loop through file lines
        while line:
            #Load file info one line at a time!
            line=input_fp.readline()
            if line.startswith('Curve'):
                try:
                    (curveNum,L,P)=str(line.split(':')[1]).split(',')
                except IndexError:
                    print "Problem with file:",inputFilename
                    print "Returning zeroes for this file stats!"
                    return [0,0,0,0,0,0,0,0,0]
                L=float(L)
                P=float(P)
                if maxL<L:
                    maxL=L
                if maxP<P:
                    maxP=P
                Lsum=Lsum+L
                Psum=Psum+P
                LsumSqr=LsumSqr+(L*L)
                PsumSqr=PsumSqr+(P*P)
                curveCount=curveCount+1
        meanL=meanP=float(0)
        varL=varP=float(0)
        n=curveCount
        if curveCount < 1:
            return  []
            #return [0,0,0,0,0,0,0,0,0]
        meanL=Lsum.__div__(n)
        varL=float(LsumSqr - float(Lsum*Lsum).__div__(n)).__div__(n)
        meanP=Psum.__div__(n)
        varP=float(PsumSqr - float(Psum*Psum).__div__(n)).__div__(n)
        statList=[0,0,n,meanP,varP,meanL,varL,maxP,maxL]
        return statList
    #end method candidateStatsOnDisk
    

    def dumpCandidateKurveSummary(self):
        """
        This method creates a variable object that contains the
        summary associated with all the elements in the input
        structure self.curves.  This output can be used to write the
        summary information to a .summary text file.  This method is
        closely related to createSummaryStructure method.  We have as
        output of this method
        """
        summary=[]
        for lineInfo in self.curves:
            curveID,l,p=lineInfo.getKurveHeader()
            #See notes in methods below for explaination
            #Offsets in d,F are likely wrong...Tina-Thu-Oct-04-2007:200710041448 
            d=lineInfo.getCandidateDuration()+self.gpsWidth.getAsFloat()
            F=lineInfo.getCandidateBandwidth()+self.freqWidth
            t=float(lineInfo.printStartGPS())
            s=float(lineInfo.printStopGPS())
            f=float(lineInfo.printStartFreq())
            g=float(lineInfo.printStopFreq())
            tmp=lineInfo.getBrightPixelAndStats()
            v=tmp[0][3] #Freq of Bright Pixel
            h=tmp[0][2].getAsFloat() #GPS time of Bright Pixel
            j=tmp[0][4] #The pixel power value for brightest pixel
            m=tmp[1] #Mean power of pixels in curve
            s=tmp[2] #stddev^2 of pixel power in curve
            #summary.append([t,s,f,g,l,p,d,F])
            summary.append([t,s,f,g,l,p,d,F,v,h,m,s,j])
        return summary
    #End dumpCandidateKurveSummary()

    def createSummaryStructure(self):
        """
        Method to build a summary structure for saving to file or
        printing to screen.
        The summary should contain the following fields
        GPS start, Delta T, F start, F stop,F Band, T bright,F bright, Avg
        brightness,Std dev brightness, Curve Length, Integrate Power
        """
        # Set the output formatting string here for all summary
        # display calls
        self.summaryFormat="%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" 
        summaryData=self.dumpCandidateKurveSummary()
        output=[]
        for entry in summaryData:
            outString=self.summaryFormat%(entry[0],entry[6],entry[2],entry[3],entry[7],entry[9],entry[8],entry[12],entry[10],entry[11],entry[4],entry[5])
#            outString=self.summaryFormat%(entry[0],entry[1],entry[2],entry[3],entry[4],entry[5],entry[6],entry[7],entry[8],entry[9])
            output.append(outString)
        return output
    #End createSummaryStructure method
    
    def writeSummary(self,override=''):
        """
        Method to write summary with formatting specified in
        createSummaryStructure method to a file.
        """
        if override=='':
            sourceFile=self.filename[0]
        else:
            sourceFile=override
        outRoot,outExt=os.path.splitext(sourceFile)
        outFile=outRoot+'.summary'
        fp=open(outFile,'w')
        key="# GPSstart Duration Fstart Fstop Fband Tbright Fbright AvgBright StdDevBright CL IP\n"
        fp.write(key)
        for entry in self.createSummaryStructure():
            fp.write(entry)
        fp.close()
    # End writeSummary method

    def printSummary(self):
        """
        Method to write summary with formatting specified in
        createSummaryStructure method to the screen.
        """
        sourceFile=self.filename[0]
        print "#"
        print "#"+sourceFile
        print "#The fields are:"
        print "#GPS\t    dT\t    Fstart\t    Fstop\t    Fband\t   Tbright\t    Fbright\t    Pbright\t   AvgBright\t    STDBright\t    CL\t    IP\t"
        for entry in self.createSummaryStructure():
            print entry,
    # End printSummary method
    
    def applyArbitraryThresholds(self,expressionString):
        """
        This method takes in a string and uses it literally to construct a
        testing condition to impose on the kurve entries from a candidate file.
        It then returns the list of candidates meeting the express written in the
        string.  Use caution with this function! This is parsed left to right!
        Valid variable labels:
        P     Integrated Power
        L     Curve Length in Pixels
        D     Time Duration in Seconds
        F     Frequency Bandwidth in Hertz
        Example:
        P>10 and L < 5
        D >=2 or P>12
        F<3 and D >= 8
        (D > 10 and L = 3) and F < 4
        """
        #If the variable self.curves is empty just return nothing!
        if self.curves.__len__() < 1:
            print "Warning no information to threshold."
            return self
        #There is no error checking.  We rely on eval to do this for us!
        testExp=expressionString
        #Convert everything to lower case
        testExp=testExp.lower()
        resultsList=[]
        #print "String to threshold with :",testExp
        for lineInfo in self.curves:
            curveID,l,p=lineInfo.getKurveHeader()
            d=lineInfo.getCandidateDuration()
            b=lineInfo.getCandidateBandwidth()
            t=float(lineInfo.printStartGPS())
            s=float(lineInfo.printStopGPS())
            f=float(lineInfo.printStartFreq())
            g=float(lineInfo.printStopFreq())
            #Additional eval stuff?
            tmpBrightnessInfo=lineInfo.getBrightPixelAndStats()
            v=float(tmpBrightnessInfo[0][3]) #Freq of Bright Pixel
            h=float(tmpBrightnessInfo[0][2].getAsFloat()) #float Time of Bright Pixel
            j=float(tmpBrightnessInfo[0][4]) #Power of Bright Pixel
            m=float(tmpBrightnessInfo[1]) #Curve mean pixel brightness
            c=float(tmpBrightnessInfo[2]) #Curve stddev of pixel brightness
            try:
                evalResult=eval(testExp)
            except :
                print "Error with expression string syntax."
                print "Received string:  ",str(testExp).upper()
                os.abort()
            if evalResult:
                resultsList.append(lineInfo)
        #Return the a modified structure with self.curves
        #made only of passing candidates
        outputObject=copy.deepcopy(self)
        outputObject.curves=copy.deepcopy(resultsList)
        outputObject.totalCount=resultsList.__len__()
        return outputObject
    #end applyAbitraryThresholds method
    
    def getPixelList(self):
        """
        Method to get a 3C list of all pixels from this candidate file.  It is
        a list of lists each element is [float(time),float(freq),float(power)].
        The list is loaded from all Kurves contained in self.
        """
        pixelList=[]
        for element in self.curves:
            for point in element.getKurveDataBlock_HumanReadable():
                pixelList.append(point)
        return pixelList
    #End method getPixelList()

    def getGnuplotPixelList(self,startVal):
        """
        Get pixel list as a text list objects for proper GNUPLOT
        formating.  We use startVal to make the time marker relative
        to that time value.
        """
        pixelList=[]
        for element in self.curves:
            for point in element.getKurveDataBlock_HumanReadable():
                pixelList.append([point[0]-startVal,point[1],point[2]])
            pixelList.append([' ',' ',' '])
        return pixelList
    #End method getGnuplotPixelList()
    
    def writePixelList(self,filename,style):
        """
        Write the list of pixels to a 3C file given a FILENAME
        arguement and STYLE argument
        TFP gets 3C data of time,freq,power
        anything else gets just time,freq
        The list is in terms of the first time value and its offset.
        if the style arguement is tf+time then...
        The start time should be listed in the filename! (I hope)
        """
        if type(style) != type(str('test')):
            print "Error on type of pixel list file!"
            os.abort
        output_fp=open(filename,'w')
        format3C="%10.5s %10.5s %10.5s\n"
        format2C="%10.5s %10.5s\n"
        pixelList=self.getPixelList()
        pixelList.sort()
        try:
            minVal=pixelList[0][0]
        except IndexError:
            output_fp.close()
            return
        print "You requested ",style
        if style.lower() == 'tfp':
            for line in self.getGnuplotPixelList(0):
                output_fp.write(format3C%(line[0],line[1],line[2]))
        elif style.lower() == 'tf':
            for line in self.getGnuplotPixelList(0):
                output_fp.write(format2C%(line[0],line[1]))
        elif style.lower() == 'tf+time':
            fTime=str(pixelList[0][0])
            A=0
            B=0
            if fTime.__contains__(','):
                [A,B]=fTime.split(',')
            elif fTime.__contains__('.'):
                [A,B]=fTime.split('.')
            else:
                A=0
                B=0
            gpsStart=gpsInt(A,B)
            minVal=gpsStart.getAsFloat()
            for line in self.getGnuplotPixelList(minVal):
                output_fp.write(format2C%(line[0],line[1]))
        output_fp.close()
    #End method writePixelList()

    def graphdata(self,filename='',gpsReferenceFloat=0.0,timescale='second',useLogColors=True,myColorMap='jet'):
        """
        This method uses matplotlib.py to make plots of curves
        contained in this list!  Currently all plotting functions
        are hard wired to the method!  Method needs to have relative
        offsets specified as an optional argument.  The input args are
        filename,gpsRefTime,timescale,useLogColors,colormap
        each of which is NOT manditory.
        """        
        #Determine version of matplotlib
        matplotlibVersion=int(pylab.matplotlib.__version__.replace('.',''))
        if self.totalCount==0:
            sys.stdout.write("Omitting this plot no triggers to plot.\n");
            return
        #This code creates a scatter plot in X windows
        #If pylab loads Ok.
        brightX=[]
        brightY=[]
        brightP=[]
        minX=gpsReferenceFloat
        line2plot=[]
        brightSpotX=[]
        brightSpotY=[]
        brightSpotZ=[]
        start=True
        # If the GPSreference for plot is given do not rescan data
        # automatically.
        if (gpsReferenceFloat == 0):
            for element in self.curves:
                for point in element.getKurveDataBlock_HumanReadable():
                    if start:
                        minX=float(point[0])
                        start=False
                    if minX >= float(point[0]):
                        minX = float(point[0])
        #Convert the time (X) axis scale to given above argument
        conversionFactor=1;
        timeLabel="(seconds)"
        if timescale.lower().__contains__("minute"):
            conversionFactor=60
            timeLabel="(minutes)"
        if timescale.lower().__contains__("hour"):
            conversionFactor=60*60
            timeLabel="(hours)"
        if timescale.lower().__contains__("day"):
            conversionFactor=60*60*24
            timeLabel="(days)"
        spinner=progressSpinner(self.verboseMode)
        spinner.setTag('Plotting')
        elementIPlist=[]
        for element in self.curves:
            xtmp=[]
            ytmp=[]
            ztmp=[]
            bP=element.getBrightPixelAndStats()
            brightSpotX.append((bP[0][2].getAsFloat()-minX)/conversionFactor)
            brightSpotY.append(bP[0][3])
            if (bP[2] == 0):
                #Unable to determine Z score
                brightSpotZ.append(0)
            else:
                brightSpotZ.append(float(bP[0][4]-bP[1]).__abs__()/bP[2])
            #Get curve stats IP
            for point in element.getKurveDataBlock_HumanReadable():
                xtmp.append((float(point[0])-minX)/conversionFactor)
                ytmp.append(float(point[1]))
                ztmp.append(float(point[2]))
            elementIP=element.getKurveHeader()[2]
            elementIPlist.append(elementIP)
            line2plot.append([xtmp,ytmp,ztmp,elementIP])
            del xtmp
            del ytmp
            del ztmp
            spinner.updateSpinner()
        spinner.closeSpinner()
        maxValue=float(max(elementIPlist))
        minValue=float(min(elementIPlist))
        #Setup colorbar hack
        stepSize=(maxValue-minValue)/256
        linearValueMatrix=pylab.outerproduct(pylab.arange(minValue,maxValue,stepSize),pylab.ones(1))
        #Set jet colormap
        #Determine mapping of IP to colors
        maxValue=max(elementIPlist)
        minValue=min(elementIPlist)
        #fig=pylab.figure()
        pylab.cm.ScalarMappable().set_cmap(myColorMap)
        linearColorScale=pylab.matplotlib.colors.normalize(minValue,maxValue)
        if (minValue > 0) and (maxValue > 0):
            logColorScale=pylab.matplotlib.colors.normalize(math.log(minValue),math.log(maxValue))
        else:
            logColorScale=linearColorScale
        #If we are using version 0.80.0 of below
        version800=int(str('0.80.0').replace('.',''))
        if matplotlibVersion<=version800:
            print "Matlib plot version ",matplotlibVersion
            pylab.imshow(linearValueMatrix,cmap=pylab.cm.get_cmap(myColorMap),origin="upper",extent=[0,0.01,0,0.01])
            pylab.delaxes()
            pylab.hold(True)
            pylab.colorbar(tickfmt='%2.1e',orientation='vertical')
        currentPalette=pylab.get_cmap()
        for entry in line2plot:
            if useLogColors:
                try:
                    myRed,myGreen,myBlue,myAlpha=currentPalette(
                        logColorScale(math.log(entry[3])))
                except:
                    sys.stderr.write("Problem mapping trigger log color.\n")
                    sys.stderr.write("Value causing errors is %e\n"%(entry[3]))
                    myRed,myGreen,myBlue,myAlpha=(0,0,0,1)
            else:
                try:
                    myRed,myGreen,myBlue,myAlpha=currentPalette(
                        linearColorScale(entry[3]))
                except:
                    sys.stderr.write("Problem mapping trigger linear color.\n")
                    sys.stderr.write("Value causing errors is %e\n"%(entry[3]))
                    myRed,myGreen,myBlue,myAlpha=(0,0,0,1)
            pylab.plot(entry[0],entry[1],color=(myRed,myGreen,myBlue))
        #Normalize the brightSpotZ max -> 0..5
        normalizeZscoreTo=100
        if brightSpotZ.__len__() < 1:
            factor=1;
        else:
            factor=normalizeZscoreTo/(max(brightSpotZ))
        tmpZ=[]
        for entry in brightSpotZ:
            tmpZ.append(entry*factor)
        brightSpotZ=tmpZ
        pylab.scatter(brightSpotX,brightSpotY,brightSpotZ,color='black',marker='o')
        pylab.xlabel(str("Time %s"%(timeLabel)))
        pylab.ylabel("Freq (Hz)")
        pylab.figtext(0.01,0.95,"GPS %9.2f"%(minX))
        textLocX=0.80
        if not useLogColors:
            pylab.figtext(textLocX,0.025,str(myColorMap).upper()+":Linear Coloring")
        else:
            pylab.figtext(textLocX,0.025,str(myColorMap).upper()+":Log Coloring")
        if ((filename.upper()=='') or (filename.upper()=='AUTO')):
            [name,extension]=os.path.splitext(self.filename[0])
            figtitle=os.path.basename(name)
        else:
            figtitle=filename
        #If newer version of matplotlib library try colorbar again
        version877=int(str('0.87.7').replace('.',''))
        if ((version800 < matplotlibVersion <= version877) or
            (matplotlibVersion > version877)):
            print "Matlib plot version ",matplotlibVersion
            sys.stderr.write("Error setting colorbar! Sorry, figure will have no colorbar.\n")
            sys.stderr.write("Using matlibplot version :"+pylab.matplotlib.__version__+"\n")
        pylab.title("%s"%(figtitle))
        pylab.grid(True)
        if (filename==''):
            pylab.show()
            pylab.close()
        else:
            if (filename.upper()=='AUTO'):
                [fullpath,extension]=os.path.splitext(self.filename[0])
                filename=os.path.basename(fullpath)+'.png'
            pylab.savefig(filename)
    #End method graphdata

    def createGlitchDatabase(self,verbose=bool(False)):
        """
        Commands to build either a structure which can be graphed
        selectively or to create a database for use with auto-glitch
        classification pipeline. There is a companion summary for this
        method.  If one wants to hand cut the summary you can excise the
        triggers from the output summary from original entire database for
        graphing the triggers.
        Symmetry weights -- Z scores replace this field!
        """
        weight=1
        glitchDatabase=[]
        spinner=progressSpinner(verbose,2)
        spinner.setTag('GlitchDB ')
        for trigger in self.curves:
            spinner.updateSpinner()
            triggerStartString=trigger.startGPS().__diskPrint__()
            triggerStartFloat=trigger.startGPS().getAsFloat()
            triggerStopFloat=trigger.stopGPS().getAsFloat()
            triggerBandwidth,triggerLowF,triggerHighF=trigger.getCandidateBandwidth(bool(True))
            triggerID,triggerLength,triggerIntegratedPower=trigger.getKurveHeader()
            brightPixel=trigger.getBrightPixel()
            cmPixel=trigger.getBrightPixel()
            meanPixelPower,varPixelPower=trigger.__getKurveMeanVar__()
            triggerCentralFreq=triggerLowF+(triggerHighF-triggerLowF)/2
            triggerDuration=triggerStopFloat-triggerStartFloat
            triggerCentralTime=(triggerStopFloat+triggerStartFloat)/2
            #
            relativeTimeBP=brightPixel[2].getAsFloat()-triggerCentralTime
            relativeFreqBP=brightPixel[3]-triggerCentralFreq
            ###symmetryBP=trigger.getSymmetryFactor(brightPixel,weight)
            zScoreBP=(
                (triggerIntegratedPower-meanPixelPower)/
                math.sqrt(varPixelPower)
                )
            relativeTimeCM=cmPixel[2].getAsFloat()-triggerCentralTime
            relativeFreqCM=cmPixel[3]-triggerCentralFreq
            ###symmetryCM=trigger.getSymmetryFactor(brightPixel,weight)
            zScoreCM=(
                (triggerIntegratedPower-meanPixelPower)/
                math.sqrt(varPixelPower)
                )
            #(+) if T_bp > T_cm
            if (triggerDuration > 0):
                spanTnorm=(
                    (brightPixel[2].getAsFloat()-cmPixel[2].getAsFloat())/
                    triggerDuration
                )
            else:
                spanTnorm=0
            #(+) if F_bp>F_cm
            if (triggerBandwidth > 0):
                spanFnorm=(brightPixel[3]-cmPixel[3])/triggerBandwidth
            else:
                spanFnorm=0
            #Create glitch database entry
            glitchDatabaseEntry=[triggerStartString,triggerStartFloat,
                                 triggerLowF,triggerDuration,
                                 triggerBandwidth,int(triggerLength),
                                 triggerIntegratedPower,meanPixelPower,
                                 varPixelPower,relativeTimeBP,
                                 relativeFreqBP,0,
                                 zScoreBP,relativeTimeCM,
                                 relativeTimeCM,0,
                                 zScoreCM,spanTnorm,spanFnorm]
            glitchDatabase.append(glitchDatabaseEntry)
        spinner.closeSpinner()
        return glitchDatabase
        #End createGraphingSummary

    def writeGlitchDatabase(self,glitchDatabase='',override=''):
        """
        Write out the glitch database for the candidateList structure.
        """
        if type(glitchDatabase) == type(''):
            glitchDatabase=self.createGlitchDatabase(self.verboseMode)
        format="%s\t"
        entryFormat="%10.5f\t"
        for index in range(1,glitchDatabase[0].__len__(),1):
            format=format+entryFormat
        format=format.rstrip('\t')+"\n"
        if override=='':
            sourceFile=self.filename[0]
        else:
            sourceFile=override
        outRoot,outExt=os.path.splitext(sourceFile)
        outFile=outRoot+'.glitchDB'
        fp=open(outFile,'w')
        counter=0
        for entry in self.createGlitchDatabase(self.verboseMode):
            counter+=1
            tupleForm=tuple(entry)
            string=format%tupleForm
            fp.write(string)
        fp.close()
    #End writeGlitchDatabase method
#End candidateList class

#Misc Utility classes
class progressSpinner:
    """
    Provides makeshift spinner status text icon for user fun!
    Defaults input verbose arguement to TRUE.  Use the spinner.
    Send a false to __init__ causes all spinner methods do nothing!
    """
    def __init__(self,verbose=False,spinnerNum=int(3)):
        self.verbose=verbose
        self.timesCalled=0
        self.frameCount=0
        self.spinTag=''
        self.timeNow=0
        self.timeLast=0
        self.spinnerString01=".:^"
        self.spinnerString02=".oOo"
        self.spinnerString03="-|/-\|/"
        self.spinnerFrames=[]
        if int(spinnerNum) == 1:
            self.spinnerStringPicked=self.spinnerString01
        if int(spinnerNum) == 2:
            self.spinnerStringPicked=self.spinnerString02
        if int(spinnerNum) == 3:
            self.spinnerStringPicked=self.spinnerString03
        for frame in self.spinnerStringPicked:
            self.spinnerFrames.append(frame)
    #End __init__

    def setTag(self,txtTag):
        """
        Method sets text string as a spinner tag
        """
        self.spinTag=txtTag
        return
    
    def updateSpinner(self):
        """
        Method updates the icon.  If it is first invokation it
        submits a carriage return and then starts the progress
        indicator
        """
        if self.verbose:
            #if self.timesCalled == 0:
            #    sys.stderr.writelines('\n')
            #    sys.stderr.flush()
            sys.stderr.writelines('\r')
            sys.stderr.flush()
            pos=self.timesCalled%self.spinnerFrames.__len__()
            sys.stderr.writelines(self.spinTag+self.spinnerFrames[pos])
            sys.stderr.flush()
            self.timesCalled+=1
    #End updateSpinner method
    def resetSpinner(self):
        if self.verbose:
            self.timesCalled=0;
    #End resetSpinner
    def getSpinCounts(self):
        """
        Gets integer value of calls to updateSpinner since
        either init or resetSpinner was called
        """
        if self.verbose:
            return self.timesCalled
        else:
            return -1
    #End getSpinCounts
    def closeSpinner(self):
        """
        Method that you call when you don't need spinner any longer!
        """
        if self.verbose:
            sys.stderr.writelines('\n\n')
            sys.stderr.flush()
    #End closeSpinner
#Misc methods

def generateFileList(inputTXT):
    """
    This method checks the input string to see if it is
    1) a single file
    2) a list of files
    3) a path
    then we create a list object of files to process
    """
    #print "generating file list for :",str(inputTXT)
    absPathFilename=os.path.abspath(inputTXT)
    objList=[]
    dirnameFilename=str(absPathFilename)
    basenameFilename=''
    extensionFilename=''
    if absPathFilename.__contains__("*"):
        dirnameFilename=os.path.dirname(absPathFilename)
        basenameFilename=os.path.basename(absPathFilename)
        extensionFilename=str(str(basenameFilename).split('.')[1])
    if os.path.isdir(dirnameFilename):
        #Create the listing of files to process
        objList=[]
        fileList=[]
        fileList=os.listdir(dirnameFilename)
        for entry in fileList:
            if (str(entry).lstrip().rstrip().endswith(extensionFilename.lstrip().rstrip())):
                objList.append(dirnameFilename+'/'+entry)
    elif os.path.isfile(dirnameFilename):
            #Read this list in from the file
            #Could possibly be just the name of single candidate file!
            objList=[]
            fp=open(dirnameFilename,'r')
            objList=fp.readlines()
            fp.close()
            if str(objList[0]).__contains__('#'):
                #This is a single candidate file specified
                objList=[dirnameFilename]
    else:
        print "Error with getting candidate information from: ",absPathFilename
        objList=[]
    return objList
#End generateFileList method

def determineDataPadding(cp):
    """
    Method that uses the pipeline config file to determine how much
    extra data that should be sent to a node to accomplish an analysis
    when there is a nonzero bin_buffer flag in the configuration
    file. This method returns a time in integer 'ceiling'ed seconds  to
    amount of data need for the padding.
    """
    thePad=0
    if cp.has_option('tracksearchtime','bin_buffer'):
        #We need to determine the padding needed in seconds
        timeBins=int(cp.get('tracksearchtime','bin_buffer'))
        mapTime=float(cp.get('layerconfig','layer1TimeScale'))
        mapBins=int(cp.get('tracksearchtime','number_of_time_bins'))
        binDuration=float(mapTime/mapBins)
        #Force pad to be at least 1 second long
        thePad=math.ceil((binDuration*(timeBins)))
        return float(thePad)
    else:
        return float(0)
#End determineDataPadding

def autoCreateSegmentList(cp,iniFile,specificGPS):
    """
    This is a utility method which takes a specific GPS time and
    creates an appropriate segment list for building a tracksearch
    pipe.  It uses the information about the pipe properties to create
    the appropriate segment list which accounts for data burning etc.
    """
    outputfile='ERROR'
    #Load needed properties [layerconfig]
    if not cp.has_section('layerconfig'):
        sys.stderr.write("INI file missing [layerconfig] section.\n")
        return outputfile
    if cp.has_option('layerconfig','burndata'):
        burnData=float(cp.get('layerconfig','burndata'))
    else:
        burnData=float(0)
    if cp.has_option('layerconfig','layerTopBlockSize'):
        TBS=float(cp.get('layerconfig','layerTopBlockSize'))
    else:
        #An error can't auto make segment list
        return outputfile
    if cp.has_option('tracksearchtime','bin_buffer'):
        dataPad=determineDataPadding(cp)
    else:
        dataPad=0
    # Determine the start and end GPS that are appropriate for
    # the segment list in segwizard format.
    interval=round((TBS/2)+max([burnData,dataPad]))
    lowerGPS=specificGPS - interval
    upperGPS=specificGPS + interval
    duration=upperGPS - lowerGPS
    #Write output segment file in proper format with commented header
    filecontents=[]
    filecontents.append('# This file was autogenerated for GPS time: '+str(specificGPS))
    filecontents.append('# The INI file associated with this segment list is:')
    filecontents.append('# '+str(iniFile))
    filecontents.append('#')
    filecontents.append('0 '+str(int(lowerGPS))+' '+str(int(upperGPS))+' '+str(int(duration)))
    out_filename=os.path.normpath(
        os.path.dirname(os.path.abspath(iniFile))
        +'/tracksearch_'+str(specificGPS)+'.segment')
    output_fp=open(out_filename,'w')
    for line in filecontents:
        output_fp.write(line+"\n")
    output_fp.close()
    outputfile=out_filename
    return outputfile
#End autoCreateSegmentList

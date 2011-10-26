#!/usr/bin/env python

# Copyright (C) 2011 Duncan Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
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

# =============================================================================
# Preamble
# =============================================================================

from __future__ import division
import re,numpy,math,subprocess,scipy,sys

from glue.ligolw import ligolw,table,lsctables,utils
from glue.ligolw.utils import process as ligolw_process
from glue import segments

from pylal import llwapp
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
from pylal.xlal import constants as XLALConstants
from pylal.dq import dqTriggerUtils

from matplotlib import use
use('Agg')
from pylab import hanning

# Hey, scipy, shut up about your nose already.
import warnings
warnings.filterwarnings("ignore")
from scipy import signal as signal

from matplotlib import mlab
from glue import git_version

__author__  = "Duncan Macleod <duncan.macleod@astro.cf.ac.uk>"
__version__ = "git id %s" % git_version.id
__date__    = git_version.date

"""
This module provides a bank of useful functions for manipulating triggers and trigger files for data quality investigations.
"""

# =============================================================================
# Execute shell command and get output
# =============================================================================

def make_external_call(command):

  """
    Execute shell command and capture standard output and errors. 
    Returns tuple "(stdout,stderr)".
  """

  p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    shell=isinstance(command, str))
  out, err = p.communicate()

  return out,err

# =============================================================================
# Read injection files
# =============================================================================

def frominjectionfile(file, type, ifo=None, start=None, end=None):
  
  """
    Read generic injection file object file containing injections of the given
    type string. Returns an 'Sim' lsctable of the corresponding type.

    Arguments:
   
      file : file object
      type : [ "inspiral" | "burst" | "ringdown" ]

    Keyword arguments:

      ifo : [ "G1" | "H1" | "H2" | "L1" | "V1" ]
  """

  # read type
  type = type.lower()

  # read injection xml
  xml = re.compile('(xml$|xml.gz$)')
  if re.search(xml,file.name):
    xmldoc,digest = utils.load_fileobj(file)
    injtable = table.get_table(xmldoc,'sim_%s:table' % (type))

  # read injection txt
  else:
    cchar = re.compile('[#%<!()_\[\]{}:;\'\"]+')

    #== construct new Sim{Burst,Inspiral,Ringdown}Table
    injtable = lsctables.New(lsctables.__dict__['Sim%sTable' % (type.title())])
    if type=='inspiral':
      columns = ['geocent_end_time.geocent_end_time_ns',\
                 'h_end_time.h_end_time_ns',\
                 'l_end_time.l_end_time_ns',\
                 'v_end_time.v_end_time_ns',\
                 'distance'] 
      for line in file.readlines():
        if re.match(cchar,line):
          continue
        # set up siminspiral object
        inj = lsctables.SimInspiral()
        # split data
        sep = re.compile('[\s,=]+')
        data = sep.split(line)
        # set attributes
        inj.geocent_end_time    = int(data[0].split('.')[0])
        inj.geocent_end_time_ns = int(data[0].split('.')[1])
        inj.h_end_time          = int(data[1].split('.')[0])
        inj.h_end_time_ns       = int(data[1].split('.')[1])
        inj.l_end_time          = int(data[2].split('.')[0])
        inj.l_end_time_ns       = int(data[2].split('.')[1])
        inj.v_end_time          = int(data[3].split('.')[0])
        inj.v_end_time_ns       = int(data[3].split('.')[1])
        inj.distance            = float(data[4])

        injtable.append(inj)

    if type=='burst':
      if file.readlines()[0].startswith('filestart'):
        # if given parsed burst file
        file.seek(0)

        snrcol = { 'G1':23, 'H1':19, 'L1':21, 'V1':25 }

        for line in file.readlines():
          inj = lsctables.SimBurst()
          # split data
          sep = re.compile('[\s,=]+')
          data = sep.split(line)
          # set attributes

          # gps time
          if 'burstgps' in data:
            idx = data.index('burstgps')+1
            geocent = LIGOTimeGPS(data[idx])

            inj.time_geocent_gps    = geocent.seconds
            inj.time_geocent_gps_ns = geocent.nanoseconds
          else:
            continue


          #inj.waveform            = data[4]
          #inj.waveform_number     = int(data[5])

          # frequency
          if 'freq' in data:
            idx = data.index('freq')+1
            inj.frequency = float(data[idx])
          else:
            continue

          # SNR a.k.a. amplitude
          if ifo and 'snr%s' % ifo in data:
            idx = data.index('snr%s' % ifo)+1
            inj.amplitude = float(data[idx])
          elif 'rmsSNR' in data:
            idx = data.index('rmsSNR')+1
            inj.amplitude = float(data[idx])
          else:
            continue

          if 'phi' in data:
            idx = data.index('phi' )+1
            inj.ra = float(data[idx])*24/(2*math.pi)       

          if 'theta' in data:
            idx = data.index('theta' )+1 
            inj.ra = 90-(float(data[idx])*180/math.pi)

          if ifo and 'hrss%s' % ifo in data:
            idx = data.index('hrss%s' % ifo)+1
            inj.hrss = float(data[idx])
          elif 'hrss' in data:
            idx = data.index('hrss')+1
            inj.hrss = float(data[idx])

          # extra columns to be added when I know how
          #inj.q = 0
          #inj.q                   = float(data[11])
          #h_delay = LIGOTimeGPS(data[41])
          #inj.h_peak_time         = inj.time_geocent_gps+h_delay.seconds
          #inj.h_peak_time_ns      = inj.time_geocent_gps_ns+h_delay.nanoseconds
          #l_delay = LIGOTimeGPS(data[43])
          #inj.l_peak_time         = inj.time_geocent_gps+l_delay.seconds
          #inj.l_peak_time_ns      = inj.time_geocent_gps_ns+l_delay.nanoseconds
          #v_delay = LIGOTimeGPS(data[43])
          #inj.v_peak_time         = inj.time_geocent_gps+v_delay.seconds
          #inj.v_peak_time_ns      = inj.time_geocent_gps_ns+v_delay.nanoseconds

          injtable.append(inj)

      else:
        # if given parsed burst file
        file.seek(0)
        for line in file.readlines():
          inj = lsctables.SimBurst()
          # split data
          sep = re.compile('[\s,]+')
          data = sep.split(line)
          # set attributes
          geocent = LIGOTimeGPS(data[0])
          inj.time_geocent_gps    = geocent.seconds
          inj.time_geocent_gps_ns = geocent.nanoseconds

          injtable.append(inj)

  injections = table.new_from_template(injtable)
  if not start:  start = 0
  if not end:    end   = 9999999999
  span = segments.segmentlist([ segments.segment(start, end) ])
  get_time = dqTriggerUtils.def_get_time(injections.tableName)
  injections.extend(inj for inj in injtable if get_time(inj) in span)

  return injections

# =============================================================================
# Calculate band-limited root-mean-square
# =============================================================================

def blrms(data, sampling, average=1, band=None, filter='butter', order=4,\
          remove_mean=False, verbose=False):

  """
    This function will calculate the band-limited root-mean-square of the given
    data, using averages of the given length in the given [f_low,f_high) band.

    Options are included to offset the data, and weight frequencies given a 
    dict object of (frequency:weight) pairs.

    Arguments:

      data : numpy.ndarray
        array of data points
      sampling : int
        number of data points per second

    Keyword arguments:


  """

  nyq = sampling/2

  # verify band variables
  if band==None:
    band=[0,sampling/2]
  fmin = float(band[0])
  fmax = float(band[1])

  if verbose:
    sys.stdout.write("Calculating BLRMS in band %s-%s Hz...\n" % (fmin, fmax))

  #
  # remove mean
  #

  if remove_mean:
    data = data-data.mean()
    if verbose: sys.stdout.write("Data mean removed.\n")

  #
  # construct filter
  # 

  # construct passband
  passband = [fmin*2/sampling,fmax*2/sampling]
  # construct filter
  filter = filter.lower()
  b,a = signal.iirfilter(order, passband, btype='bandpass', output='ba',\
                         ftype=filter, rp=0.5, rs=50)

  #
  # bandpass
  #

  data = signal.lfilter(b,a,data)
  data = data[::-1]
  data = signal.lfilter(b,a,data)
  data = data[::-1]

  if verbose: sys.stdout.write("Bandpass filter applied to data.\n")

  #
  # calculate rms
  #

  # construct output array
  numsamp = average*sampling
  numaverage = numpy.ceil(len(data)/sampling/average)
  output  = numpy.empty(numaverage)

  nanwarned=False

  # loop over averages
  for i in xrange(len(output)):

    # get indices
    idxmin = i*sampling*average
    idxmax = idxmin + numsamp

    # get data chunk
    chunk = data[idxmin:idxmax]

    # get rms
    rms = numpy.sqrt(numpy.power(chunk,2).mean())
    if not nanwarned and numpy.isnan(rms):
      sys.stderr.write("WARNING, NaN found in BLRMS.\n")
      nanwarned=True
    output[i] = rms

  if verbose: sys.stdout.write("RMS calculated for %d averages.\n"\
                               % len(output))

  return output

# =============================================================================
# Function to bandpass a time-series
# =============================================================================

def bandpass(data, f_low, f_high, sampling, order=4):

  """
    This function will bandpass filter data in the given [f_low,f_high) band
    using the given order Butterworth filter.
  """

  # construct passband
  passband = [f_low*2/sampling,f_high*2/sampling]
  # construct filter
  b,a = signal.butter(order,passband,btype='bandpass')
  # filter data forward then backward
  data = signal.lfilter(b,a,data)
  data = data[::-1]
  data = signal.lfilter(b,a,data)
  data = data[::-1]

  return data

# =============================================================================
# Highpass
# =============================================================================

def highpass(x, f_low, sampling, order=8):

  # construct passband
  bpass = 2*f_low/sampling

  # construct filter
  (b, a) = signal.butter(order, bpass, btype='high', analog=0, output='ba')

  # filter data forward then backward
  y = signal.lfilter(b,a,x)
  y = y[::-1]
  y = signal.lfilter(b,a,y)
  y = y[::-1]

  return y

# =============================================================================
# Calculate spectrum
# =============================================================================

def spectrum(data, sampling, NFFT=256, overlap=0.5,\
             window='hanning', detrender=mlab.detrend_linear,\
             sides='onesided', scale='PSD'):

  numpoints  = len(data)
  numoverlap = int(sampling * (1.0 - overlap))

  if isinstance(window,str):
    window=window.lower()

  win = signal.get_window(window, NFFT)

  # calculate PSD with given parameters
  spec,freq = mlab.psd(data, NFFT=NFFT, Fs=sampling, noverlap=numoverlap,\
                       window=win, sides=sides, detrend=detrender)

  # rescale data to meet user's request
  scale = scale.lower()
  if scale == 'asd':
    spec = numpy.sqrt(spec) * numpy.sqrt(2 / (sampling*sum(win**2)))
  elif scale == 'psd':
    spec *= 2/(sampling*sum(win**2))
  elif scale == 'as':
    spec = nump.sqrt(spec) * numpy.sqrt(2) / sum(win)
  elif scale == 'ps':
    spec = spec * 2 / (sum(win)**2)

  return freq, spec.flatten()

# =============================================================================
# Median Mean Spectrum
# =============================================================================

def AverageSpectrumMedianMean(data, fs, NFFT=256, overlap=128,\
                              window='hanning', sides='onesided',\
                              verbose=False):

  """
    Computes power spectral density of a data series using the median-mean
    average method.
  """

  # cast data series to numpy array
  data = numpy.asarray(data)

  # number of segments (must be even)
  if overlap==0:
    numseg = int(len(data)/NFFT)
  else:
    numseg = 1 + int((len(data)-NFFT)/overlap)
  assert (numseg - 1)*overlap + NFFT == len(data),\
         "Data is wrong length to be covered completely, please resize"

  # construct window
  win = scipy.signal.get_window(window, NFFT)

  if verbose: sys.stdout.write("%s window constructed.\nConstructing "
                               "median-mean average spectrum "
                               "with %d segments...\n"\
                               % (window.title(), numseg))

  #
  # construct PSD
  #

  # fft scaling factor for units of Hz^-1
  scaling_factor = 1 / (fs * NFFT)

  # construct frequency
  f = numpy.arange(NFFT//2 + 1) * (fs / NFFT)

  odd  = numpy.arange(0, numseg, 2)
  even = numpy.arange(1, numseg, 2)

  # if odd number of segments, ignore the first one (better suggestions welcome)
  if numseg == 1:
    odd = [0]
    even = []
  elif numseg % 2 == 1:
    odd = odd[:-1]
    numseg -= 1
    sys.stderr.write("WARNING: odd number of FFT segments, skipping last.\n")

  # get bias factor
  biasfac = MedianBias(numseg//2)
  # construct normalisation factor
  normfac = 1/(2*biasfac)

  # set data holder
  S = numpy.empty((numseg, len(f)))

  # loop over segments
  for i in xrange(numseg):

    # get data
    chunk = data[i*overlap:i*overlap+NFFT]
    # apply window
    wdata = WindowDataSeries(chunk, win)
    # FFT
    S[i]  = PowerSpectrum(wdata) * scaling_factor

  if verbose: sys.stdout.write("Generated spectrum for each chunk.\n")

  # compute median-mean average
  if numseg > 1:
    S_odd = scipy.median([S[i] for i in odd])
    S_even = scipy.median([S[i] for i in even])
    S = (S_even  + S_odd) * normfac
  else:
    S = S.flatten()
  if verbose: sys.stdout.write("Calculated median-mean average.\n")

  return f, S

# =============================================================================
# Median bias factor
# =============================================================================

def MedianBias(nn):

  """
    Returns the median bias factor.
  """

  nmax = 1000;
  ans  = 1;
  n    = (nn - 1)//2;
  if nn >= nmax:
   return numpy.log(2)

  for i in xrange(1, n+1):
    ans -= 1.0/(2*i);
    ans += 1.0/(2*i + 1);

  return ans;

# =============================================================================
# Median average spectrum
# =============================================================================

def AverageSpectrumMedian(data, fs, NFFT=256, overlap=128,\
                          window='hanning', sides='onesided',\
                          verbose=False):

  """
    Construct power spectral density for given data set using the median
    average method.  
  """

  # cast data series to numpy array
  data = numpy.asarray(data)

  print data.mean()

  # number of segments (must be even)
  if overlap==0:
    numseg = int(len(data)/NFFT)
  else:
    numseg = 1 + int((len(data)-NFFT)/overlap)
  assert (numseg - 1)*overlap + NFFT == len(data),\
         "Data is wrong length to be covered completely, please resize"

  # construct window
  win = scipy.signal.get_window(window, NFFT)

  if verbose: sys.stdout.write("%s window constructed.\nConstructing "
                               "median average spectrum "
                               "with %d segments...\n"\
                               % (window.title(), numseg))

  #
  # construct PSD
  #

  # fft scaling factor for units of Hz^-1
  scaling_factor = 1 / (fs * NFFT)

  # construct frequency
  f = numpy.arange(NFFT//2 + 1) * (fs / NFFT)

  # get bias factor
  biasfac = MedianBias(numseg)

  # construct normalisation factor
  normfac = 1/(biasfac)

  # set data holder
  S = numpy.empty((numseg, len(f)))

  # loop over segments
  for i in xrange(numseg):

    # get data
    chunk = data[i*overlap:i*overlap+NFFT]
    # apply window
    wdata = WindowDataSeries(chunk, win)
    # FFT
    S[i]  = PowerSpectrum(wdata) * scaling_factor

  if verbose: sys.stdout.write("Generated spectrum for each chunk.\n")

  # compute median-mean average
  if numseg > 1:
    S = scipy.median([S[i] for i in odd])*normfac
  else:
    S = S.flatten()
  if verbose: sys.stdout.write("Calculated median average.\n")

  return f, S 

# =============================================================================
# Apply window
# =============================================================================

def WindowDataSeries(series, window=None):

  """
    Apply window function to data set, defaults to Hanning window.
  """

  # generate default window
  if window == None:
    window = scipy.signal.hanning(len(series))

  # check dimensions
  assert len(series)==len(window), 'Window and data must be same shape'

  # get sum of squares
  sumofsquares = numpy.power(window,2).sum()
  assert sumofsquares > 0, 'Sum of squares of window non-positive.'

  # generate norm
  norm = numpy.sqrt(len(window)/numpy.power(window,2).sum())

  # apply window
  return series * window * norm

# =============================================================================
# Power spectrum
# =============================================================================

def PowerSpectrum(series, sides='onesided'):

  """
    Calculate power spectum of given series
  """

  # cast series to numpy array
  series = numpy.array(series)

  # apply FFT
  tmp = numpy.fft.fft(series, n=len(series))

  # construct spectrum
  if sides=='onesided':
    spec = numpy.empty(len(tmp)//2+1)
  elif sides=='twosided':
    spec = numpy.empty(len(tmp))

  # DC component
  spec[0] = tmp[0]**2

  # others
  s = (len(series)+1)//2
  spec[1:s] = 2 * numpy.power(tmp[1:s].real, 2) + numpy.power(tmp[1:s].real, 2)

  # Nyquist
  if len(series) % 2 == 0:
    spec[len(series)/2] = tmp[len(series)/2]**2

  return spec


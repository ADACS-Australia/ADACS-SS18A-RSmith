#!/usr/bin/env python
import sys, getopt
import matplotlib.cm
from matplotlib.patches     import Patch
from matplotlib.axes        import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors      import normalize, Colormap
from optparse import * 
from pylab    import *
from readMeta import *

class snglInspiral(readSnglInspiralTable,Axes): 

  def __init__(self, triggerfile):
    readSnglInspiralTable.__init__(self,triggerfile)

  def summary(self):
    subplot(221)
    hist(self.snr)
    xlabel(r'SNR', size='x-large')
    ylabel(r'# triggers', size='x-large')
    subplot(222)
    hist(self.mass1)
    title(r'Excess power trigger')

  def plot_snr_v_chisq(self):
    plot(self.snr,self.chisq,'rx')
    title('SNR vs CHISQ')
    xlabel('snr')
    ylabel('chisq')
    gca().grid(True)

  def plot_snr_v_time(self):
    S3start = 751651213
    S4start = 793130413
    secsPerDay = 3600*24
    plot((self.end_time - S3start) / secsPerDay,self.snr,'rx')
    title('SNR vs TIME')
    xlabel('time')
    ylabel('snr')
    gca().grid(True)
 
  def histHistc_snr(self):
    subplot(211)  
    if len(self.snr) > 4:
      nbins = len(self.snr)/4
    else: nbins = 1
    nn,xx,patches = hist(self.snr,nbins,normed=0)
    gca().grid(True)
    nTrigsAbove = 0
    for i in range(0, len(nn)): 
      nTrigsAbove += nn[i]
    mm = []   
    for i in range(0, len(nn)):
      mm.append(nTrigsAbove)
      nTrigsAbove -= nn[i]

    for i in range (0, len(nn)):
      if nn[i] == 0: 
        nn[i] = 1
    lognn = log10( nn )
    logmm = log10( mm )

    subplot(212)
    bar(xx,logmm,(xx[-1]-xx[0])/nbins,color='r')
    bar(xx,lognn,(xx[-1]-xx[0])/nbins,color='g')
    gca().grid(True)
    #gca().set_yscale('log')
    #gca().set_ylim( (0.001,1000))


class doubleCoincInspiral(readSnglInspiralTable,Axes):

  def __init__(self, triggerfile1, triggerfile2):
    self.table1 = readSnglInspiralTable(triggerfile1)
    self.table2 = readSnglInspiralTable(triggerfile2)
    # can't do the following
    # readSnglInspiralTable.__init__(self.table1,triggerfile)

  def plot_m1_v_m2(self):
    plot(self.table1.mass1,self.table1.mass2,'rx')
    plot(self.table2.mass1,self.table2.mass2,'b+')
    gca().grid(True)


class tripleCoincInspiral(readSnglInspiralTable,Axes):

  def __init__(self, triggerfile1, triggerfile2, triggerfile3):
    self.table1 = readSnglInspiralTable(triggerfile1)
    self.table2 = readSnglInspiralTable(triggerfile2)
    self.table3 = readSnglInspiralTable(triggerfile3)
    # can't do the following
    # readSnglInspiralTable.__init__(self.table1,triggerfile)

  def plot_m1_v_m2(self):
    plot(self.table1.mass1,self.table1.mass2,'r+')
    plot(self.table2.mass1,self.table2.mass2,'bx')
    plot(self.table3.mass1,self.table3.mass2,'gx')
    gca().grid(True)


def main():
  # define usage and command line options and arguments - parse
  usage = "usage: %prog ..."
  parser = OptionParser( usage )

  opts_snglInsp = OptionGroup( parser, "Single Inspiral plotting functions",\
	"Example ..." )
  opts_snglInsp.add_option( "-a", "--snglInsp_snrVtime",\
	action="store_true", default=False,\
	help="plot snr vs time from a single inspiral xml" )
  opts_snglInsp.add_option( "-b", "--snglInsp_snrVchisq",\
        action="store_true", default=False,\
        help="plot snr vs chisq from single inspiral xml")
  opts_snglInsp.add_option( "-c", "--snglInsp_histHistc_snr",\
        action="store_true", default=False,\
        help="plot snr histograms from single inspiral xml" )
  opts_snglInsp.add_option( "-d", "--snglInsp_summary",\
        action="store_true", default=False,\
        help="plot summary info from single inspiral xml" )

  parser.add_option_group( opts_snglInsp )

  parser.add_option( "-p", "--show_plot",\
        action="store_true", default=False,\
        help="display plot" )
  # change this so that by default the fig is always saved using 
  # the name convention already implemented. Now instead of --save-fig
  # you have --save-off and --save-as flag to override
  # the standard name. Also add the date+time to the standard name OR
  # check for the existence of the standard name + 001, 002, 003, ...
  # Remove where possible the defns of dest in favour of the long option name
  parser.add_option( "-s", "--save_fig",\
        action="store_true", default=False,\
        help="save figure in .png and .ps format" )
  parser.add_option( "-t", "--temporary-test",\
        action="store_true", default=False,\
        help="only for developers to test this program" )

  ( options, xml_files ) = parser.parse_args()
  
  # check xml_files have been given as required arguments 
  if not xml_files:
    print >> sys.stderr, "No trigger file specified"
    sys.exit(1)

  # read data files and call plotting function desired
  if   options.snglInsp_snrVtime:
    trigs = snglInspiral(xml_files[0])
    trigs.plot_snr_v_time()
  elif options.snglInsp_snrVchisq:
    trigs = snglInspiral(xml_files[0])
    trigs.plot_snr_v_chisq()
  elif options.snglInsp_histHistc_snr:
    trigs = snglInspiral(xml_files[0])
    trigs.histHistc_snr()
  elif options.snglInsp_summary:
    trigs = snglInspiral(xml_files[0])
    trigs.summary()
  else:
    print >> sys.stderr, "No plot option specified"
    sys.exit(1)
  
  # save and show plot if desired
  if options.save_fig:
    png_file = xml_file[:-3] + plot_type + ".png"
    ps_file  = xml_file[:-3] + plot_type + ".ps"
    savefig(png_file)
    savefig(ps_file)
    print "Saved plot to file %s" % (png_file)
    print "Saved plot to file %s" % (ps_file)
  if options.show_plot:
    show()

# execute main if this module is explicitly invoked by the user
if __name__=="__main__":
        main()

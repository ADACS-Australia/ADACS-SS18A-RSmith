import os,sys,numpy,glob,math

from pylal import grbsummary,antenna,llwapp
from glue import segmentsUtils
from glue.ligolw import lsctables
from glue.ligolw.utils import process as ligolw_process

from glue.ligolw import lsctables
from pylal import SimInspiralUtils, InspiralUtils
from pylal.xlal.constants import LAL_PI, LAL_MTSUN_SI
import numpy as np

# define new_snr
def new_snr( snr, chisq, chisq_dof, q=4.0, n=3.0 ):

  if chisq_dof==0:
    chisq_dof = 40

  if chisq <= chisq_dof:
    return snr
  return snr / ((1 + (chisq / chisq_dof)**(q/n)) / 2)**(1/q)

# reverse engineer new_snr for contours
def new_snr_chisq( snr, new_snr, chisq_dof, q=4.0, n=3.0 ):

  chisqnorm = (snr/new_snr)**q
 
  if chisqnorm <= 1:
    return 1E-20
  return chisq_dof * (2*chisqnorm - 1)**(n/q)

def get_bestnr( trig, q=4.0, n=3.0, null_thresh=(4.25,6), fResp = None ):

  """
    Calculate BestNR (coh_PTF detection statistic) through signal based vetoes:

    The signal based vetoes are as follows:
      * Coherent SNR < 6
      * Bank chi-squared reduced (new) SNR < 6
      * Auto veto reduced (new) SNR < 6
      * Single-detector SNR (from two most sensitive IFOs) < 4
      * Null SNR (CoincSNR^2 - CohSNR^2)^(1/2) < nullthresh
    Returns BestNR as float
  """

  # coherent SNR and null SNR cut
  if (trig.snr < 6):
    return 0

  # bank veto cut
  bank_new_snr = new_snr( trig.snr, trig.bank_chisq, trig.bank_chisq_dof, q, n )
  if bank_new_snr < 6:
    return 0

  # auto veto cut
  auto_new_snr = new_snr( trig.snr, trig.cont_chisq, trig.cont_chisq_dof, q, n )
  if auto_new_snr < 6:
    return 0

  # define IFOs for sngl cut
  ifos   = [ trig.ifos[i*2:(i*2)+2] for i in range(int(len(trig.ifos)/2)) ]
  ifoAtt = { 'G1':'g', 'H1':'h1', 'H2':'h2', 'L1':'l', 'T1':'t', 'V1':'v' }

  # single detector SNR cut
  ifoSens = []
  for ifo in ifos:
    ifoSens.append(( ifo,
                     getattr(trig,'sigmasq_%s' %ifoAtt[ifo]) * fResp[ifo] ))
  ifoSens.sort( key=lambda (ifo,sens): sens, reverse=True )
  for i in [0,1]:
    if getattr( trig, 'snr_%s' % ifoAtt[ifoSens[i][0].upper()] ) <4:
      return 0 

  # get chisq reduced (new) SNR
  bestNR = new_snr( trig.snr, trig.chisq, trig.chisq_dof, q, n )

  # get null 

  # get null reduced SNR
  if len(ifos)<3:
    return bestNR

  null_snr = sum([ getattr(trig,'snr_%s' % ifoAtt[ifo])**2\
                     for ifo in ifos ]) - trig.snr**2
  if null_snr < 0:
    print "WARNING: Null SNR is less than 0!"
    print "Sum of single detector SNRs squared", sum([ getattr(trig,'snr_%s' \
                     % ifoAtt[ifo])**2  for ifo in ifos ])
    print "Coherent SNR squared", trig.snr**2
    null_snr = 0
  else:
    null_snr = null_snr**0.5

  if trig.snr > 20:
    null_thresh = numpy.array(null_thresh)
    null_thresh += (trig.snr - 20)*1./5.
  if null_snr > null_thresh[-1]:
    return 0
  elif null_snr > null_thresh[0]:
    bestNR *= 1 / (null_snr - null_thresh[0] + 1)

  return bestNR

def calculate_contours( q=4.0, n=3.0, null_thresh=6.,\
                        null_grad_snr=20 ):

  """
    Generate the plot contours for chisq variable plots
  """
  # initialise chisq contour values and colours
  cont_vals = [5.5,6,6.5,7,8,9,10,11]
  num_vals  = len(cont_vals)
  colors    = ['y-','k-','y-','y-','y-','y-','y-','y-']

  # get SNR values for contours
  snr_low_vals  = numpy.arange(6,30,0.1)
  snr_high_vals = numpy.arange(30,500,1)
  snr_vals      = numpy.asarray(list(snr_low_vals) + list(snr_high_vals))

  # initialise contours
  bank_conts = numpy.zeros([len(cont_vals),len(snr_vals)], dtype=numpy.float64)
  auto_conts = numpy.zeros([len(cont_vals),len(snr_vals)], dtype=numpy.float64)
  chi_conts  = numpy.zeros([len(cont_vals),len(snr_vals)], dtype=numpy.float64)
  null_cont  = []

  # set chisq dof
  # FIXME should be done with variables
  chisq_dof = 60
  bank_chisq_dof = 40
  cont_chisq_dof = 160

  # loop over each and calculate chisq variable needed for SNR contour
  for j,snr in enumerate(snr_vals):
    for i,new_snr in enumerate(cont_vals):

      bank_conts[i][j] = new_snr_chisq(snr, new_snr, bank_chisq_dof, q, n)
      auto_conts[i][j] = new_snr_chisq(snr, new_snr, cont_chisq_dof, q, n)
      chi_conts[i][j]  = new_snr_chisq(snr, new_snr, chisq_dof, q, n)

    if snr > null_grad_snr:
      null_cont.append(null_thresh + (snr-null_grad_snr)*1./5.)
    else:
      null_cont.append(null_thresh)

  return bank_conts,auto_conts,chi_conts,null_cont,snr_vals,colors

def plot_contours( axis, snr_vals, contours, colors ):

  for i in range(len(contours)):
    plot_vals_x = []
    plot_vals_y = []
    for j in range(len(snr_vals)):
      if contours[i][j] > 1E-15:
        plot_vals_x.append(snr_vals[j])
        plot_vals_y.append(contours[i][j])
    axis.plot(plot_vals_x,plot_vals_y,colors[i])

def readSegFiles(segdir):

  times = {}
  for name,fileName in\
      zip( ["buffer",       "off",             "on"],\
           ["bufferSeg.txt","offSourceSeg.txt","onSourceSeg.txt"] ):

    segs = segmentsUtils.fromsegwizard(open(os.path.join(segdir,fileName), 'r'))
    if len(segs)>1:
      raise AttributeError, 'More than one segment, an error has occured.'
    times[name] = segs[0]
  return times
       
def makePaperPlots():

  import pylab

  pylab.rcParams.update({
    "text.usetex": True,
    "text.verticalalignment": "center",
#    "lines.markersize": 12,
#    "lines.markeredgewidth": 2,
#    "lines.linewidth": 2.5,
    "figure.figsize": [8.0, 6.0],
    "font.size": 20,
    "axes.titlesize": 16,
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "font.family": "serif",
    "font.weight": "bold",
    })

def get_ra_dec(grbFile):

  """
    DEPRECATED
  """

  ext_trigs = grbsummary.load_external_trigs(grbFile)
  ra = ext_trigs[0].event_ra
  dec = ext_trigs[0].event_dec
  return ra,dec

def read_sigma_vals( sigmaFile ):

  """
    DEPRECATED
  """

  sigmaVals = {}
  file = open(sigmaFile,'r')
  for line in file:

    line = line.replace('\n','')
    ifo,min,max = line.split(' ')
    sigmaVals[ifo + 'min'] = float(min)
    sigmaVals[ifo + 'max'] = float(max)

  return sigmaVals
    
def get_det_response( ra, dec, trigTime ):

  """
    Return detector response for complete set of IFOs for given sky location
    and time. Inclination and polarization are unused so are arbitrarily set to 0 
  """

  f_plus  = {}
  f_cross = {}
  inclination   = 0
  polarization  = 0
  for ifo in ['G1','H1','H2','L1','T1','V1']:
    f_plus[ifo],f_cross[ifo],_,_ = antenna.response( trigTime, ra, dec,\
                                                     inclination, polarization,\
                                                     'degree', ifo )
  return f_plus,f_cross

def append_process_params( xmldoc, args, version, date ):

  """
    Construct and append process and process_params tables to ligolw.Document
    object xmldoc, using the given sys.argv variable args and other parameters.
  """

  xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.ProcessTable))
  xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.ProcessParamsTable))

  # build and seed process params
  progName = args[0]
  process = llwapp.append_process( xmldoc, program=progName,\
                                   version=version,\
                                   cvs_repository = 'lscsoft',\
                                   cvs_entry_time = date)
  params = []
  for i in range(len(args)):
    p = args[i]
    if not p.startswith('-'):
      continue
    v = ''
    if i < len(sys.argv)-1:
      v = sys.argv[i+1]
    params.append( map( unicode, (p,'string',v) ) )

  ligolw_process.append_process_params(xmldoc,process,params)

  return xmldoc

def identify_bad_injections(log_dir):

  files = glob.glob(os.path.join(log_dir,"*err"))

  badInjs = []

  for file in files:
    if os.stat(file)[6] != 0:
      fp = open(file,"r")
      conts = fp.read()
      if conts.find('terminated') != -1:
        conts=conts.split('\n')
        for line in conts:
          line = line.split(' ')
          line = [entry.replace(',','') for entry in line if entry]
          if 'terminated' in line:
            injDict = {}
            injDict['mass1'] = float(line[6])
            injDict['mass2'] = float(line[8])
            injDict['spin1x'] = float(line[10])
            injDict['spin1y'] = float(line[12])
            injDict['spin1z'] = float(line[14])
            injDict['spin2x'] = float(line[16])
            injDict['spin2y'] = float(line[18])
            injDict['spin2z'] = float(line[20])
            if not injDict in badInjs:
              badInjs.append(injDict)
  return badInjs

def remove_bad_injections(sims,badInjs):

  new_sims = []
  for sim in sims:
    for badInj in badInjs:
      if (abs(sim.mass1-badInj['mass1'])) < 0.001:
        if (abs(sim.mass2-badInj['mass2'])) < 0.001:
          if (abs(sim.spin1x-badInj['spin1x'])) < 0.001:
            if (abs(sim.spin1y-badInj['spin1y'])) < 0.001:
              if (abs(sim.spin1z-badInj['spin1z'])) < 0.001:
                if (abs(sim.spin2x-badInj['spin2x'])) < 0.001:
                  if (abs(sim.spin2y-badInj['spin2y'])) < 0.001:
                    if (abs(sim.spin2z-badInj['spin2z'])) < 0.001:
                      print "Removing injection:",sim.mass1,sim.mass2,sim.spin1x,sim.spin1y,sim.spin1z,sim.spin2x,sim.spin2y,sim.spin2z
                      break
    else:
      new_sims.append(sim)

  return new_sims

def sim_inspiral_get_theta(self):

  # conversion factor for the angular momentum
  angmomfac = self.mass1 * self.mass2 * \
              numpy.power(LAL_PI * LAL_MTSUN_SI * (self.mass1 + self.mass2) * \
                       self.f_lower, -1.0/3.0)
  m1sq = self.mass1 * self.mass1
  m2sq = self.mass2 * self.mass2

  # compute the orbital angular momentum
  L = numpy.zeros(3)
  L[0] = angmomfac * numpy.sin(self.inclination)
  L[1] = 0
  L[2] = angmomfac * numpy.cos(self.inclination)

  # compute the spins
  S = np.zeros(3)
  S[0] =  m1sq * self.spin1x + m2sq * self.spin2x
  S[1] =  m1sq * self.spin1y + m2sq * self.spin2y
  S[2] =  m1sq * self.spin1z + m2sq * self.spin2z

  # and finally the total angular momentum
  J = L + S

  theta = math.atan2(math.sqrt(J[0]*J[0] + J[1]*J[1]),J[2])
  if theta > math.pi/2.:
    theta = math.pi - theta

  if theta < 0 or theta > math.pi/2.:
    raise Error, "Theta is too big or too small"

  return theta



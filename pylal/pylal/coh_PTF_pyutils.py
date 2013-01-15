# Copyright (C) 2012 Ian W. Harry, Duncan M. Macleod
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

"""Utilities for the coherent PTF inspiral analysis.
"""

import os
import sys
import numpy
import glob
import math
import re

from pylal import grbsummary, antenna, llwapp, InspiralUtils, SimInspiralUtils
from pylal.xlal.constants import LAL_PI, LAL_MTSUN_SI

from glue import segmentsUtils
from glue.ligolw import lsctables, table
from glue.ligolw.utils import process as ligolw_process


def new_snr_chisq(snr, new_snr, chisq_dof, q=4.0, n=3.0):
    """Returns the chisq value needed to weight snr into new_snr
    """
    chisqnorm = (snr/new_snr)**q
    if chisqnorm <= 1:
        return 1E-20
    return chisq_dof * (2*chisqnorm - 1)**(n/q)


def get_bestnr( trig, q=4.0, n=3.0, null_thresh=(4.25,6)):
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

    snr = trig.snr

    # coherent SNR and null SNR cut
    if (snr < 6.) or (trig.get_new_snr(q/n*3, 'bank_chisq') < 6.) or\
         (trig.get_new_snr(q/n*3, 'cont_chisq') < 6.):
        return 0

    # define IFOs for sngl cut
    ifos = map(str,trig.get_ifos())

    # single detector SNR cut
    sens = {}
    fPlus,fCross = get_det_response(numpy.degrees(trig.ra),\
                                    numpy.degrees(trig.dec),\
                                    trig.get_end())
    for ifo in ifos:
        if ifo.lower()[0] == 'h' :
            i = ifo.lower()
        else:
            i = ifo[0].lower()
        sens[ifo] = getattr(trig, 'sigmasq_%s' % i.lower()) * \
                        sum(numpy.array([fPlus[ifo], fCross[ifo]])**2)
    ifos.sort(key=lambda ifo: sens[ifo], reverse=True)
    for i in xrange(0,2):
        if ifos[i].lower()[0] == 'h' :
            i = ifos[i].lower()
        else:
            i = ifos[i][0].lower()
        if getattr(trig, 'snr_%s' % i) <4:
            return 0

    # get chisq reduced (new) SNR
    bestNR = trig.get_new_snr(q/n*3, 'chisq')

    # get null reduced SNR
    if len(ifos)<3:
        return bestNR

    null_snr = trig.get_null_snr()

    if snr > 20:
        null_thresh = numpy.array(null_thresh)
        null_thresh += (snr - 20)*1./5.
    if null_snr > null_thresh[-1]:
        return 0
    elif null_snr > null_thresh[0]:
        bestNR *= 1 / (null_snr - null_thresh[0] + 1)

    return bestNR

def calculate_contours(q=4.0, n=3.0, null_thresh=6., null_grad_snr=20):
    """Generate the plot contours for chisq variable plots
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
    bank_conts = numpy.zeros([len(cont_vals),len(snr_vals)],
                             dtype=numpy.float64)
    auto_conts = numpy.zeros([len(cont_vals),len(snr_vals)],
                             dtype=numpy.float64)
    chi_conts = numpy.zeros([len(cont_vals),len(snr_vals)],
                            dtype=numpy.float64)
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
            zip(["buffer", "off", "on"],\
                ["bufferSeg.txt","offSourceSeg.txt","onSourceSeg.txt"]):

        segs = segmentsUtils.fromsegwizard(open(os.path.join(segdir,fileName),
                                                'r'))
        if len(segs)>1:
            raise AttributeError, 'More than one segment, an error has occured.'
        times[name] = segs[0]
    return times


def makePaperPlots():
    import pylab
    pylab.rcParams.update({
        "text.usetex": True,
        "text.verticalalignment": "center",
#        "lines.markersize": 12,
#        "lines.markeredgewidth": 2,
#        "lines.linewidth": 2.5,
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


def get_det_response(ra, dec, trigTime):
    """Return detector response for complete set of IFOs for given sky
    location and time. Inclination and polarization are unused so are
    arbitrarily set to 0.
    """
    f_plus  = {}
    f_cross = {}
    inclination   = 0
    polarization  = 0
    for ifo in ['G1','H1','H2','L1','T1','V1']:
        f_plus[ifo],f_cross[ifo],_,_ = antenna.response(trigTime, ra, dec,\
                                                        inclination,
                                                        polarization, 'degree',
                                                        ifo)
    return f_plus,f_cross


def get_f_resp(self):
    """FIXME
    """
    if re.search('SimInspiral', str(self)):
        ra = numpy.degrees(self.longitude)
        dec = numpy.degrees(self.latitude)
        t = self.get_time_geocent()
    else:
        ra = numpy.degrees(self.ra)
        dec = numpy.degrees(self.dec)
        t = self.get_end()

    fplus, fcross = get_det_response(ra, dec, t)
    return dict((ifo, fplus[ifo]**2 + fcross[ifo]**2) for ifo in fplus.keys())


def append_process_params(xmldoc, args, version, date):
    """Construct and append process and process_params tables to
    ligolw.Document object xmldoc, using the given sys.argv variable
    args and other parameters.
    """
    xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.ProcessTable))
    xmldoc.childNodes[-1].appendChild(
        lsctables.New(lsctables.ProcessParamsTable))

    # build and seed process params
    progName = args[0]
    process = llwapp.append_process(xmldoc, program=progName,
                                    version=version, cvs_repository='lscsoft',
                                    cvs_entry_time=date)
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
            if ((abs(sim.mass1-badInj['mass1']) < 0.001) and
                (abs(sim.mass2-badInj['mass2']) < 0.001) and
                (abs(sim.spin1x-badInj['spin1x']) < 0.001) and
                (abs(sim.spin1y-badInj['spin1y']) < 0.001) and
                (abs(sim.spin1z-badInj['spin1z']) < 0.001) and
                (abs(sim.spin2x-badInj['spin2x']) < 0.001) and
                (abs(sim.spin2y-badInj['spin2y']) < 0.001) and
                (abs(sim.spin2z-badInj['spin2z']) < 0.001)):
                print("Removing injection:", sim.mass1, sim.mass2, sim.spin1x,
                      sim.spin1y, sim.spin1z, sim.spin2x, sim.spin2y,
                      sim.spin2z)
                break
            else:
                new_sims.append(sim)

    return new_sims


def sim_inspiral_get_theta(self):
    # conversion factor for the angular momentum
    angmomfac = self.mass1 * self.mass2 * \
                numpy.power(LAL_PI * LAL_MTSUN_SI * (self.mass1 + self.mass2) *
                            self.f_lower, -1.0/3.0)
    m1sq = self.mass1 * self.mass1
    m2sq = self.mass2 * self.mass2

    # compute the orbital angular momentum
    L = numpy.zeros(3)
    L[0] = angmomfac * numpy.sin(self.inclination)
    L[1] = 0
    L[2] = angmomfac * numpy.cos(self.inclination)

    # compute the spins
    S = numpy.zeros(3)
    S[0] =  m1sq * self.spin1x + m2sq * self.spin2x
    S[1] =  m1sq * self.spin1y + m2sq * self.spin2y
    S[2] =  m1sq * self.spin1z + m2sq * self.spin2z

    # and finally the total angular momentum
    J = L + S

    theta = math.atan2(math.sqrt(J[0]*J[0] + J[1]*J[1]),J[2])
    if theta > math.pi/2.:
        theta = math.pi - theta

    if theta < 0 or theta > math.pi/2.:
        raise Error("Theta is too big or too small")

    return theta


def apply_snr_veto(mi_table, snr=6.0, return_index=False):
    """Veto events in a MultiInspiralTable based on their (coherent) SNR
    value.

    @param mi_table
        a MultiInspiralTable from which to veto events
    @param snr
        the value of coherent SNR on which to threshold
    @param return_index
        boolean to return the index array of non-vetoed elements rather
        than a new table containing the elements themselves

    @returns
        a new MultiInspiralTable with those events not vetoed OR
        the indices of the original mi_table not vetoed if return_index=True
    """
    mi_snr = numpy.asarray(mi_table.get_column("snr"))
    keep = mi_snr >= snr
    if return_index:
        return keep
    else:
        out = table.new_from_template(mi_table)
        out.extend(numpy.asarray(mi_table)[keep])
        return out


def apply_bank_veto(mi_table, snr=6.0, chisq_index=4.0, return_index=False):
    """Veto events in a MultiInspiralTable based on their bank chisq-
    weighted (new) coherent SNR.

    @param mi_table
        a MultiInspiralTable from which to veto events
    @param snr
        the value of coherent new SNR on which to threshold
    @param chisq_index
        the index \f$\iota\f$ used in the newSNR calculation:
        \f[\rho_{\mbox{new}} =
            \frac{\rho}{\left[\frac{1}{2}
                \left(1 + \left(\frac{\chi^2}{n_\mbox{dof}}\right)^{\iota/3}
                \right)\right]^{1/\iota}}
        \f]
    @param return_index
        boolean to return the index array of non-vetoed elements rather
        than a new table containing the elements themselves

    @returns
        a new MultiInspiralTable with those events not vetoed OR
        the indices of the original mi_table not vetoed if return_index=True
    """
    bank_new_snr = numpy.asarray(mi_table.get_new_snr(column="bank_chisq"))
    keep = bank_new_snr >= snr
    if return_index:
        return keep
    else:
        out = table.new_from_template(mi_table)
        out.extend(numpy.asarray(mi_table)[keep])
        return out


def apply_auto_veto(mi_table, snr=6.0, chisq_index=4.0, return_index=False):
    """Veto events in a MultiInspiralTable based on their auto chisq-
    weighted (new) coherent SNR.

    @param mi_table
        a MultiInspiralTable from which to veto events
    @param snr
        the value of coherent new SNR on which to threshold
    @param chisq_index
        the index \f$\iota\f$ used in the newSNR calculation:
        \f[\rho_{\mbox{new}} =
            \frac{\rho}{\left[\frac{1}{2}
                \left(1 + \left(\frac{\chi^2}{n_\mbox{dof}}\right)^{\iota/3}
                \right)\right]^{1/\iota}}
        \f]
    @param return_index
        boolean to return the index array of non-vetoed elements rather
        than a new table containing the elements themselves

    @returns
        a new MultiInspiralTable with those events not vetoed OR
        the indices of the original mi_table not vetoed if return_index=True
    """
    cont_new_snr = numpy.asarray(mi_table.get_new_snr(column="cont_chisq"))
    keep = cont_new_snr >= snr
    if return_index:
        return keep
    else:
        out = table.new_from_template(mi_table)
        out.extend(numpy.asarray(mi_table)[keep])
        return out


def apply_sngl_snr_veto(mi_table, snrs=[4.0, 4.0], return_index=False):
    """Veto events in a MultiInspiralTable based on their single-detector
    snr in the most sensitive detectors.

    @param mi_table
        a MultiInspiralTable from which to veto events
    @param snrs
        an X-element list of single-detector SNRs on which to threshold
        for the X most sensitive detectors (in sensitivity order)
    @param return_index
        boolean to return the index array of non-vetoed elements rather
        than a new table containing the elements themselves

    @returns
        a new MultiInspiralTable with those events not vetoed OR
        the indices of the original mi_table not vetoed if return_index=True
    """
    if len(mi_table) == 0:
        return mi_table
    # parse table
    ifos = lsctables.instrument_set_from_ifos(mi_table[0].ifos)
    mi_time = numpy.asarray(mi_table.get_end()).astype(float)
    mi_ra = numpy.asarray(mi_table.get_column("ra"))
    mi_dec = numpy.asarray(mi_table.get_column("dec"))
    mi_sngl_snr = numpy.asarray([numpy.asarray(mi_table.get_sngl_snr(ifo)) for
                                 ifo in ifos])
    mi_sigmasq = numpy.asarray([numpy.asarray(mi_table.get_sigmasq(ifo)) for
                                ifo in ifos])
    # make sure number of thresholds is relevant
    if len(snrs) > len(ifos):
        raise ValueError("%s single-detector thresholds given, but only %d "
                         "detectors found." % (len(snrs), len(ifos)))
    # find most sensitive detectors for each event
    sens = numpy.zeros((len(ifos), len(mi_table)))
    keep = numpy.ones(len(mi_table)).astype(bool)
    for i,ifo in enumerate(ifos):
        sens[i,:] = map(lambda t: antenna.response(mi_time[t], mi_ra[t],
                                                   mi_dec[t], 0, 0, "radians",
                                                   ifo)[2],
                        range(len(mi_table))) * mi_sigmasq[i,:]
    sens_ifo = sens.argsort(axis=0)[::-1][:sens.shape[0]]
    for i,snr in enumerate(snrs):
        keep &= (mi_sngl_snr[sens_ifo[i,:],
                 numpy.arange(sens.shape[1])] >= snr)
    if return_index:
        return keep
    else:
        out = table.new_from_template(mi_table)
        out.extend(numpy.asarray(mi_table)[keep])
        return out

def apply_null_snr_veto(mi_table, null_snr=6.0, snr=20.0, return_index=False):
    """Veto events in a MultiInspiralTable based on their null SNR.

    @param mi_table
        a MultiInspiralTable from which to veto events
    @param null_snr
        the value of null SNR on which to threshold
    @param snr
        the value of coherent SNR on above which to grade the null SNR
        threshold
    @param return_index
        boolean to return the index array of non-vetoed elements rather
        than a new table containing the elements themselves

    @returns
        a new MultiInspiralTable with those events not vetoed OR
        the indices of the original mi_table not vetoed if return_index=True
    """
    mi_snr = mi_table.get_column("snr")
    mi_null_snr = mi_table.get_null_snr()
    # apply gradient to threshold for high SNR
    null_thresh = numpy.ones(len(mi_table)) * null_snr
    grade = mi_snr >= snr
    null_thresh[grade] += (mi_snr[grade] - snr)/5.0
    # apply veto
    keep = mi_null_snr < null_thresh
    if return_index:
        return keep
    else:
        out = table.new_from_template(mi_table)
        out.extend(numpy.asarray(mi_table)[keep])
        return out

def veto(self, seglist, time_slide_table=None):
    """Return a MultiInspiralTable with those row from self not lying
    inside (i.e. not vetoed by) any elements of seglist.

    If time_slide_table is not None, any time shifts will be undone and each
    detector checked individually
    """
    seglist = type(seglist)([type(seg)(*map(float, seg)) for seg in seglist])
    keep = table.new_from_template(self)
    if time_slide_table:
        slides = time_slide_table.as_dict()
        for id_,vector in slides.iteritems():
            idx = str(id_).split(":")[-1]
            slides["multi_inspiral:time_slide_id:%s" % idx] = vector
            del slides[id_]
        for row in self:
            ifos = row.get_ifos()
            for i,ifo in enumerate(ifos):
                ifo_time = float(row.get_end() -
                                 slides[str(row.time_slide_id)][ifo])
                if ifo_time in seglist:
                    i = -1
                    break
            if i != -1:
                keep.append(row)
    else:
        for row in self:
            time = float(row.get_end())
            if time in seglist:
                continue
            else:
                keep.append(row)
    return keep


def vetoed(self, seglist, time_slide_table=None):
    """Return a MultiInspiralTable with those row from self lying
    inside (i.e. vetoed by) any elements of seglist.

    If time_slide_table is not None, any time shifts will be undone and each
    detector checked individually
    """
    seglist = type(seglist)(map(float, seglist))
    vetoed = table.new_from_template(self)
    if time_slide_table:
        slides = time_slide_table.as_dict()
        for id_,vector in slides.iteritems():
            idx = str(id_).split(":")[-1]
            slides["multi_inspiral:time_slide_id:%s" % idx] = vector
            del slides[id_]
        slides = get_slide_vectors(time_slide_table)
        for row in self:
            ifos = row.get_ifos()
            for i,ifo in enumerate(ifos):
                ifo_time = (float(row.get_end()) -
                            slides[str(row.time_slide_id)][ifo])
                if ifo_time in seglist:
                    vetoed.append(row)
                    break
    else:
        for row in self:
            time = float(row.get_end())
            if time in seglist:
                vetoed.append(row)
    return vetoed

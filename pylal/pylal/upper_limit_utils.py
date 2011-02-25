import numpy
from scipy import random
from scipy import interpolate
import bisect
import sys

from pylal import rate


L10s_per_Mpc3 = 50.0 #FIXME


def compute_posterior(vA, vA2, dvA, mu, prior):
    '''
    This function computes the posterior distribution on the rate parameter
    mu resulting from an experiment which was sensitive to a volume vA. This
    function implements the analytic marginalization over uncertainty in the
    efficiency at the loudest event if the input vA2 is nonzero (see Biswas,
    Creighton, Brady, Fairhurst, eqn 24). Where the sensitive volume is zero,
    the posterior is equal to the prior, which is taken to be a constant.
    '''
    if vA == 0: return mu, prior

    if vA2 == 0:
        # we have perfectly measured our efficiency in this mass bin
	# so the posterior is given by eqn (11) in BCB
	post = prior*(1+mu*vA*dvA)*numpy.exp(-mu*vA)
    else:
        # we have uncertainty in our efficiency in this mass bin and
	# want to marginalize it out using eqn (24) of BCB
	k = vA**2/vA2
	# FIXME it remains to check whether using a Gamma distribution for
	# the volume error model is sensible
	post = prior*( (1.0 + mu*vA/k)**(-k-1) + (mu*vA*dvA)*(1.0 + 1.0/k)/(1.0 + mu*vA/k)**(k+2) )

    # NB: mu here is actually the rate R = mu/T as in eqn 9 of BCB and the
    # 4-volume vA is eps*T. In eqns 14,24 of BCB, only the product
    # mu*eps = R*vA matters, except in the overall normalization, which we
    # explicitly deal with here
    post /= post.sum()

    return mu, post


def compute_upper_limit(mu, post, alpha = 0.9):
    """
    Returns the upper limit mu_high of confidence level alpha for a
    posterior distribution post on the given parameter mu.
    """
    high_idx = bisect.bisect_right( post.cumsum()/post.sum(), alpha )

    if high_idx < len(mu):
        mu_high = mu[high_idx]
    else:
        mu_high = mu[high_idx-1]

    return mu_high


def compute_lower_limit(mu, cumpost, alpha = 0.9):
    """
    Returns the lower limit mu_low of confidence level alpha for a
    cumulative ditribution cumpost on the given parameter mu.
    """
    low_idx = bisect.bisect_left( cumpost, 1 - alpha )
    mu_low = mu[low_idx]

    return mu_low


def confidence_interval( mu, post, alpha = 0.9 ):
    '''
    Returns the minimal-width confidence interval [mu_low,mu_high] of
    confidence level alpha for a distribution post on the parameter mu.
    '''
    cumpost = post.cumsum()/post.sum()

    # choose a step size for the sliding confidence window
    trust_factor = 0.9 #how much do you trust Steve to get this right? -- must be 0 < tf < 1
    whatithinkthestepsizeshouldbe = numpy.min(cumpost[cumpost[1:]-cumpost[:-1]>0])
    alpha_step = trust_factor*whatithinkthestepsizeshouldbe

    # initialize the lower and upper limits
    mu_low = numpy.min(mu)
    mu_high = numpy.max(mu)

    # find the smallest window (by delta-mu) stepping by dalpha
    for ai in numpy.arange( 0, 1-alpha, alpha_step ):
        ml = compute_lower_limit( mu, cumpost, 1 - ai )
        mh = compute_upper_limit( mu, cumpost, alpha + ai)
        if mh - ml < mu_high - mu_low:
            mu_low = ml
            mu_high = mh

    return mu_low, mu_high


def compute_efficiency( f_dist, m_dist, dbins = None):
    '''
    Compute the efficiency as a function of distance for the given sets of found
    and missed injection distances.
    '''
    if dbins is None:
        dbins = numpy.logspace( numpy.min(numpy.log10(f_dist)), numpy.max(numpy.log10(f_dist)), 50)

    efficiency = numpy.zeros( len(dbins)-1 )
    for j, dlow in enumerate(dbins[:-1]):
        dhigh = dbins[j+1]
        found = numpy.sum( f_dist[(dlow <= f_dist)*(f_dist < dhigh)] )
        missed = numpy.sum( m_dist[(dlow <= m_dist)*(m_dist < dhigh)] )
        if found+missed == 0: missed = 1.0 #avoid divide by 0 in empty bins
        efficiency[j] = found /(found + missed)

    return dbins, efficiency


def integrate_efficiency(dbins, efficiency):
    logd = numpy.log(dbins)
    dlogd = logd[1:]-logd[:-1]
    vol = numpy.sum( 4*numpy.pi *dbins[:-1]**3 *efficiency *dlogd )
    return vol


def compute_luminosity_from_volume(found, missed, dbins=None, bootnum=1, relerr=0.0):
    """
    FIXME
    """
    # convert to distances
    found_dist = numpy.array([l.distance for l in found])
    missed_dist = numpy.array([l.distance for l in missed])

    # store the volume for each bootstrap trial
    lum = numpy.zeros(bootnum)

    # bootstrap to account for statistical and amplitude calibration errors
    for trial in range(bootnum):

      if trial > 0:
          # resample with replacement from injection population
          ix = random.randint(-len(missed_dist), len(found_dist), (len(found_dist)+len(missed_dist),))
          f_dist = numpy.array([found_dist[i] for i in ix if i >= 0])
          m_dist = numpy.array([missed_dist[-(i+1)] for i in ix if i < 0])

          # apply log-normal random amplitude (distance) error
          f_dist *= numpy.exp( relerr*random.randn(len(f_dist)) )
          m_dist *= numpy.exp( relerr*random.randn(len(m_dist)) )
      else:
          # use what we got first time through
          f_dist, m_dist = found_dist, missed_dist

      # compute the efficiency and volume
      dbins, efficiency = compute_efficiency( f_dist, m_dist, dbins )
      lum[trial] = L10s_per_Mpc3*integrate_efficiency( dbins, efficiency )

    return lum


def find_host_luminosity(inj, catalog):
    '''
    Find the luminosity of the host galaxy of the given injection.
    '''
    host_galaxy = [gal for gal in catalog if inj.source == gal.name]
    if len(host_galaxy) != 1:
        raise ValueError("Injection does not have a unique host galaxy.")

    return host_galaxy[0].luminosity_mwe


def find_injections_from_host(host, injset):
    '''
    Find the set of injections that came from a given host galaxy.
    '''
    injections = [inj for inj in injset if inj.source == host.name]

    return injections

def compute_luminosity_from_catalog(found, missed, catalog):
    """
    Compute the average luminosity an experiment was sensitive to given the sets
    of found and missed injections and assuming that all luminosity comes from
    the given catalog.
    """
    # compute the efficiency to each galaxy in the catalog
    lum = 0
    for gal in catalog:

        # get the set of injections that came from this galaxy
        gal_found = find_injections_from_host(gal, found)
        gal_missed = find_injections_from_host(gal, missed)

        if len(gal_found) == 0: continue #no sensitivity here

        efficiency = len(gal_found)/(len(gal_found)+len(gal_missed))

        lum += gal.luminosity_mwe*efficiency

    return lum

def filter_injections_by_mass(injs, mlow, mhigh, bin_type):
    '''
    For a given set of injections (sim_inspiral rows), return the subset
    of injections that fall within the given mass range.
    '''
    if bin_type == "Chirp_Mass":
        newinjs = [l for l in injs if (mlow <= l.mchirp < mhigh)]
    elif bin_type == "Total_Mass":
        newinjs = [l for l in injs if (mlow <= l.mass1+l.mass2 < mhigh)]
    elif bin_type == "Component_Mass": #it is assumed that m2 is fixed
        newinjs = [l for l in injs if (mlow <= l.mass1 < mhigh)]

    return newinjs


def compute_luminosity_vs_mass(found, missed, mass_bins, bin_type, bootnum=1, catalog=None, dbins=None):
    """
    Compute the average luminosity an experiment was sensitive to given the sets
    of found and missed injections and assuming luminosity is unformly distributed
    in space.
    """
    # mean and std estimate for luminosity (in L10s)
    lumArray = rate.BinnedArray(mass_bins)
    lum2Array = rate.BinnedArray(mass_bins)

    # found/missed stats
    foundArray = rate.BinnedArray(mass_bins)
    missedArray = rate.BinnedArray(mass_bins)

    #
    # compute the mean luminosity in each mass bin
    #
    for ml,mc,mh in zip(mass_bins.lower()[0],mass_bins.centres()[0],mass_bins.upper()[0]):

        # filter out injections not in this mass bin
        newfound = filter_injections_by_mass( found, ml, mh, bin_type)
        newmissed = filter_injections_by_mass( missed, ml, mh, bin_type)

        foundArray[(mc,)] = len(newfound)
        missedArray[(mc,)] = len(newmissed)

        if len(newfound) == 0: continue #no sensitivity in this mass bin

        # compute the volume using this injection set
        if catalog is None:
            lum = compute_luminosity_from_volume(newfound, newmissed, bootnum=bootnum, dbins=dbins)
        else:
            lum = compute_luminosity_from_catalog(newfound, newmissed, catalog)

        lumArray[(mc,)] = lum.mean()
        lum2Array[(mc,)] = lum.std()

    return lumArray, lum2Array, foundArray, missedArray


def log_volume_derivative_fit(x, vols, xhat):
    '''
    Relies on scipy spline fits for each mass bin to find the (logarithmic)
    derivitave of the search volume vs x at the given xhat.
    '''
    fit = interpolate.splrep(x,numpy.log10(vols),k=3)
    val = interpolate.splev(xhat,fit,der=1)
    if val < 0:
        val = 0 #prevents negative derivitives arising from bad fits
        print >> sys.stderr, "Warning: Derivative fit resulted in Lambda < 0."

    return val



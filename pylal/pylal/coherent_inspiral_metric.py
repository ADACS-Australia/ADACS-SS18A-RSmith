# Copyright (C) 2012  Drew Keppel
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
# ============================================================================
#
#                                   Preamble
#
# ============================================================================
#


"""
This module provides functions for computing different metrics associated with
the signal manifold based on the coherent statistic for inspiral signals. The
main assumption made for this calculation is that the detectors are stationary
for the duration of the signal. The longest signals considered here are
1M_sun-1M_sun binaries starting at 10Hz for aLIGO. These signals would last for
30 minutes, and thus approximately satifisy this assumption.

The derivation of the contents of this module is very close to that for
continuous wave signals. It follows the examples of Prix arXiv:gr-qc/0606088
although uses some of the notation of Harry and Fairhurst arXiv:1012.4939 for
coherent inspiral signals.

Some of this code should be moved to c code in LAL in order to up calculations.
"""

import sys
import numpy
import scipy
from scipy import log,exp,sin,cos,arccos,arctan2,pi
from scipy import linalg
import copy

__author__ = "Drew Keppel <drew.keppel@ligo.org>"

# FIXME: these constants should come from LAL
R_earth = 6.3781e6 # radius of Earth in meters
M_sun = 4.92549095e-6 # mass of sun in seconds
c_speed = 2.99792458e8 # speed of light in meters per second
pc2m = 3.0856775807e16 # par secs in meters
pc2s = pc2m / c_speed # par secs in seconds

def A_plus(cosi, distance):
	"""
	The plus amplitude.
	"""
	Ap = (1.+cosi**2)/(2.*distance)
	return Ap

def A_cross(cosi, distance):
	"""
	The cross amplitude.
	"""
	Ax = cosi/distance
	return Ax

def A_1(cosi, distance, phi0, psi):
	"""
	1st amplitude parameter for a coherent signal.
	"""
	Ap = A_plus(cosi, distance)
	Ax = A_cross(cosi, distance)
	A1 = Ap*cos(phi0)*cos(2.*psi) - Ax*sin(phi0)*sin(2.*psi)
	return A1

def A_2(cosi, distance, phi0, psi):
	"""
	2nd amplitude parameter for a coherent signal.
	"""
	Ap = A_plus(cosi, distance)
	Ax = A_cross(cosi, distance)
	A2 = Ap*cos(phi0)*sin(2.*psi) + Ax*sin(phi0)*cos(2.*psi)
	return A2

def A_3(cosi, distance, phi0, psi):
	"""
	3rd amplitude parameter for a coherent signal.
	"""
	Ap = A_plus(cosi, distance)
	Ax = A_cross(cosi, distance)
	A3 = -Ap*sin(phi0)*cos(2.*psi) - Ax*cos(phi0)*sin(2.*psi)
	return A3

def A_4(cosi, distance, phi0, psi):
	"""
	4th amplitude parameter for a coherent signal.
	"""
	Ap = A_plus(cosi, distance)
	Ax = A_cross(cosi, distance)
	A4 = -Ap*sin(phi0)*sin(2.*psi) + Ax*cos(phi0)*cos(2.*psi)
	return A4

def coherent_signal(cosi, distance, phi0, psi, RA, dec, detector_list, hcos, hsin, f):
	"""
	Generates a coherent signal in the frequency domain given an inclination,
	a distance, a reference phase, a polarization angle, a sky location, a
	list of detectors, the cos and sin pieces of the signal, and a frequency vector.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	s = []
	for detector in detector_list:
		fp = Fp(RA,dec,detector)
		fx = Fx(RA,dec,detector)
		dt = -rn(RA,dec,detector)
		phasor = exp(2j*pi*f*dt)

		h1 = fp*hcos
		h2 = fx*hcos
		h3 = fp*hsin
		h4 = fx*hsin

		h = (A1*h1 + A2*h2 + A3*h3 + A4*h4)*phasor

		s.append(h)
	return s

def maximum_likelihood_matrix(RA, dec, detector_list):
	"""
	Computes the elements associated with the matrix used for computing
	the maximum likelihood SNR^2 for a coherent analysis of data from the
	given detectors at the given sky location.
	"""
	A = scipy.zeros(scipy.shape(RA))
	B = scipy.zeros(scipy.shape(RA))
	C = scipy.zeros(scipy.shape(RA))
	for detector in detector_list:
		fp = Fp(RA,dec,detector)
		fx = Fx(RA,dec,detector)

		A += fp*fp*detector.I_n[-7./3.]
		B += fx*fx*detector.I_n[-7./3.]
		C += fp*fx*detector.I_n[-7./3.]
	D = A*B - C**2

	return A,B,C,D

def average_snr(RA, dec, detector_list):
	"""
	Computes the average SNR^2 for a signal at a given sky location
	given a list of detectors.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	return 2./5. * (A + B)

def expected_coherent_snr(cosi, distance, phi0, psi, RA, dec, detector_list):
	"""
	Computes the expected coherent SNR^2 for a coherent signal given an
	inclination, a distance, a reference phase, a polarization angle, a sky
	location and a list of detectors.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	rhocoh = 0.
	rhocoh += (A1*A1 + A3*A3)*A
	rhocoh += (A2*A2 + A4*A4)*B
	rhocoh += 2.*(A1*A2 + A3*A4)*C

	return rhocoh

def coherent_match(s1, s2, detector_list, f):
	"""
	Computes the match between two signals given the observed waveforms in
	a set of detectors, the list of detectors, and a frequency vector.
	"""
	rho1 = 0.
	rho2 = 0.
	rho12 = 0.
	for s1_x,s2_x,detector in zip(s1,s2,detector_list):
		rho1 += 2*scipy.real(sum(s1_x*scipy.conj(s1_x)/detector.psd))
		rho2 += 2*scipy.real(sum(s2_x*scipy.conj(s2_x)/detector.psd))
		rho12 += 2*scipy.real(sum(s1_x*scipy.conj(s2_x)/detector.psd))

	return rho12/(rho1*rho2)**.5

def coherent_filter(s, RA, dec, detector_list, hcos, hsin, f):
	"""
	Computes the coherent SNR^2 time series for a template given the
	data observed in different detectors, the sky location, the list of detectors,
	the cos and sin parts of the template waveform (N.B., the template waveform should
	not include the detectors PSDs as the same waveform is used for all detectors and
	the inner product is scaled by the appropriate PSD internally), and a frequency
	vector.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	deltaF = f[1]-f[0]

	rhocos = []
	rhosin = []
	fps = []
	fxs = []
	for s_x,detector in zip(s,detector_list):
		dt = -rn(RA,dec,detector)
		rc = scipy.real(scipy.ifft(s_x*scipy.conj(hcos*exp(2j*pi*f*dt))/detector.psd))*deltaF*len(hcos)
		rs = scipy.real(scipy.ifft(s_x*scipy.conj(hsin*exp(2j*pi*f*dt))/detector.psd))*deltaF*len(hsin)
		rhocos.append(rc)
		rhosin.append(rs)
		fps.append(Fp(RA,dec,detector))
		fxs.append(Fx(RA,dec,detector))

	rhocoh = scipy.zeros(len(f), dtype='float')
	for rhocos1,rhosin1,fp1,fx1 in zip(rhocos,rhosin,fps,fxs):
		for rhocos2,rhosin2,fp2,fx2 in zip(rhocos,rhosin,fps,fxs):
			rhocoh += B*fp1*fp2*rhocos1*rhocos2
			rhocoh -= C*fp1*fx2*rhocos1*rhocos2
			rhocoh += A*fx1*fx2*rhocos1*rhocos2
			rhocoh -= C*fx1*fp2*rhocos1*rhocos2
			rhocoh += B*fp1*fp2*rhosin1*rhosin2
			rhocoh -= C*fp1*fx2*rhosin1*rhosin2
			rhocoh += A*fx1*fx2*rhosin1*rhosin2
			rhocoh -= C*fx1*fp2*rhosin1*rhosin2
	rhocoh /= D
	return rhocoh

def moments(f, PSD, n):
	"""
	Compute the integral f**n/PSD.
	"""
	df = f[1]-f[0]
	if f[0] == 0.:
		return sum(df*abs(f[1:])**n/PSD[1:])
	else:
		return sum(df*abs(f)**n/PSD)

def moments_required():
	"""
	These moments are required in the metric calculation.
	"""
	return [-1./3., -4./3., -5./3., -6./3., -7./3., -8./3., -9./3.,
		-10./3., -11./3., -12./3., -13./3., -14./3., -15./3., -17./3.]

def dx_n2(y,dx):
	"""
	Compute the first derivative of y with a second order stencil.
	"""
	out = y*0.
	for idx in range(len(y)):
		if idx == 0:
			out[idx] = -3.*y[idx] + 4.*y[idx + 1] - y[idx + 2]
			out[idx] /= 2.*dx
		elif idx == len(y) - 1:
			out[idx] = 3.*y[idx] - 4.*y[idx - 1] + y[idx - 2]
			out[idx] /= 2.*dx
		else:
			out[idx] = y[idx + 1] - y[idx - 1]
			out[idx] /= 2.*dx

	return out

class Detector:
	"""
	A class to store the necessary information associated with a given
	detector for the coherent inspiral metric calculation.
	"""
	def __init__(self, name, xarm, yarm, vertex, psd=None):
		self.xarm = xarm
		self.yarm = yarm
		self.vertex = vertex
		self.response = 0.5*(scipy.outer(xarm,xarm) - scipy.outer(yarm,yarm))
		self.psd = psd
		self.I_n = {}
		self.name = name

	def add_moment(self, f, n):
		if self.psd is None:
			print >> sys.stderr, "No psd for this detector!"
			sys.exit()
		self.I_n[n] = moments(f, self.psd, n)

	def set_psd(self, psd):
		self.psd = psd

	def set_required_moments(self, f, A=1):
		for n in moments_required():
			if self.psd is None:
				print >> sys.stderr, "No psd for this detector!"
				sys.exit()
			self.add_moment(f, n)
			self.I_n[n] *= A

def eps_plus(RA, dec):
	"""
	The plus response tensor for an interferometric GW detector for a
	given sky location.
	FIXME: Get this from LAL.
	"""
	s1 = sin(RA)
	c1 = cos(RA)
	s2 = sin(dec)
	c2 = cos(dec)

	e00 = s1**2 - (c1*s2)**2
	e01 = -s1*c1*(1+s2**2)
	e02 = c1*s2*c2
	e11 = c1**2 - (s1*s2)**2
	e12 = s1*s2*c2
	e22 = -c2**2

	return scipy.array([
			[e00, e01, e02],
			[e01, e11, e12],
			[e02, e12, e22]
		])

def Fp(RA, dec, detector):
	"""
	Compute the plus-polarization antenna factor.
	FIXME: Get this from LAL.
	"""

	d00 = detector.response[0,0]
	d01 = detector.response[0,1]
	d02 = detector.response[0,2]
	d11 = detector.response[1,1]
	d12 = detector.response[1,2]
	d22 = detector.response[2,2]

	eps = eps_plus(RA, dec)
	e00 = eps[0,0]
	e01 = eps[0,1]
	e02 = eps[0,2]
	e11 = eps[1,1]
	e12 = eps[1,2]
	e22 = eps[2,2]

	return d00*e00 + d11*e11 + d22*e22 + 2*(d01*e01 + d02*e02 + d12*e12)

def deps_plus_dRA(RA, dec):
	"""
	The derivative of the plus response tensor for an interferometric GW detector for a
	given sky location with respect to the Right Ascension.
	FIXME: Move this to LAL.
	"""
	s1 = sin(RA)
	c1 = cos(RA)
	s2 = sin(dec)
	c2 = cos(dec)

	e00 = 2*s1*c1*(1 + s2**2)
	e01 = (s1**2 - c1**2)*(1 + s2**2)
	e02 = -s1*s2*c2
	e11 = -2*s1*c1*(1 + s2**2)
	e12 = c1*s2*c2
	e22 = 0

	return scipy.array([
			[e00, e01, e02],
			[e01, e11, e12],
			[e02, e12, e22]
		])

def dFp_dRA(RA, dec, detector):
	"""
	Compute the partial derivate of the plus-polarization antenna factor
	with respect to RA, the right ascension.
	FIXME: Move this to LAL.
	"""
	d00 = detector.response[0,0]
	d01 = detector.response[0,1]
	d02 = detector.response[0,2]
	d11 = detector.response[1,1]
	d12 = detector.response[1,2]
	d22 = detector.response[2,2]

	eps = deps_plus_dRA(RA, dec)
	e00 = eps[0,0]
	e01 = eps[0,1]
	e02 = eps[0,2]
	e11 = eps[1,1]
	e12 = eps[1,2]
	e22 = eps[2,2]

	return d00*e00 + d11*e11 + d22*e22 + 2*(d01*e01 + d02*e02 + d12*e12)

def deps_plus_ddec(RA, dec):
	"""
	The derivative of the plus response tensor for an interferometric GW detector for a
	given sky location with respect to the declination.
	FIXME: Move this to LAL.
	"""
	s1 = sin(RA)
	c1 = cos(RA)
	s2 = sin(dec)
	c2 = cos(dec)

	e00 = -2*c1**2*s2*c2
	e01 = -2*s1*c1*s2*c2
	e02 = c1*(c2**2 - s2**2)
	e11 = -2*s1**2*s2*c2
	e12 = s1*(c2**2 - s2**2)
	e22 = 2*s2*c2

	return scipy.array([
			[e00, e01, e02],
			[e01, e11, e12],
			[e02, e12, e22]
		])

def dFp_ddec(RA, dec, detector):
	"""
	Compute the partial derivate of the plus-polarization antenna factor
	with respect to dec, the declination.
	FIXME: Move this to LAL.
	"""
	d00 = detector.response[0,0]
	d01 = detector.response[0,1]
	d02 = detector.response[0,2]
	d11 = detector.response[1,1]
	d12 = detector.response[1,2]
	d22 = detector.response[2,2]

	eps = deps_plus_ddec(RA, dec)
	e00 = eps[0,0]
	e01 = eps[0,1]
	e02 = eps[0,2]
	e11 = eps[1,1]
	e12 = eps[1,2]
	e22 = eps[2,2]

	return d00*e00 + d11*e11 + d22*e22 + 2*(d01*e01 + d02*e02 + d12*e12)

def eps_cross(RA, dec):
	"""
	The cross response tensor for an interferometric GW detector for a
	given sky location.
	FIXME: Get this from LAL.
	"""
	s1 = sin(RA)
	c1 = cos(RA)
	s2 = sin(dec)
	c2 = cos(dec)

	e00 = -2*s1*c1*s2
	e01 = s2*(c1**2 - s1**2)
	e02 = s1*c2
	e11 = 2*s1*c1*s2
	e12 = -c1*c2
	e22 = 0

	return scipy.array([
			[e00, e01, e02],
			[e01, e11, e12],
			[e02, e12, e22]
		])

def Fx(RA, dec, detector):
	"""
	Compute the cross-polarization antenna factor.
	FIXME: Get this from LAL.
	"""

	d00 = detector.response[0,0]
	d01 = detector.response[0,1]
	d02 = detector.response[0,2]
	d11 = detector.response[1,1]
	d12 = detector.response[1,2]
	d22 = detector.response[2,2]

	eps = eps_cross(RA, dec)
	e00 = eps[0,0]
	e01 = eps[0,1]
	e02 = eps[0,2]
	e11 = eps[1,1]
	e12 = eps[1,2]
	e22 = eps[2,2]

	return d00*e00 + d11*e11 + d22*e22 + 2*(d01*e01 + d02*e02 + d12*e12)

def deps_cross_dRA(RA, dec):
	"""
	The derivative of the cross response tensor for an interferometric GW detector for a
	given sky location with respect to the Rigth Ascension.
	FIXME: Move this to LAL.
	"""
	s1 = sin(RA)
	c1 = cos(RA)
	s2 = sin(dec)
	c2 = cos(dec)

	e00 = 2*(s1**2 - c1**2)*s2
	e01 = -4*s1*c1*s2
	e02 = c1*c2
	e11 = 2*(c1**2 - s1**2)*s2
	e12 = s1*c2
	e22 = 0

	return scipy.array([
			[e00, e01, e02],
			[e01, e11, e12],
			[e02, e12, e22]
		])

def dFx_dRA(RA, dec, detector):
	"""
	Compute the partial derivate of the cross-polarization antenna factor
	with respect to RA, the right ascension.
	FIXME: Move this to LAL.
	"""
	d00 = detector.response[0,0]
	d01 = detector.response[0,1]
	d02 = detector.response[0,2]
	d11 = detector.response[1,1]
	d12 = detector.response[1,2]
	d22 = detector.response[2,2]

	eps = deps_cross_dRA(RA, dec)
	e00 = eps[0,0]
	e01 = eps[0,1]
	e02 = eps[0,2]
	e11 = eps[1,1]
	e12 = eps[1,2]
	e22 = eps[2,2]

	return d00*e00 + d11*e11 + d22*e22 + 2*(d01*e01 + d02*e02 + d12*e12)

def deps_cross_ddec(RA, dec):
	"""
	The derivative of the cross response tensor for an interferometric GW detector for a
	given sky location with respect to the declination.
	FIXME: Move this to LAL.
	"""
	s1 = sin(RA)
	c1 = cos(RA)
	s2 = sin(dec)
	c2 = cos(dec)

	e00 = -2*s1*c1*c2
	e01 = (c1**2 - s1**2)*c2
	e02 = -s1*s2
	e11 = 2*s1*c1*c2
	e12 = c1*s2
	e22 = 0

	return scipy.array([
			[e00, e01, e02],
			[e01, e11, e12],
			[e02, e12, e22]
		])

def dFx_ddec(RA, dec, detector):
	"""
	Compute the partial derivate of the cross-polarization antenna factor
	with respect to dec, the declination.
	FIXME: Move this to LAL.
	"""
	d00 = detector.response[0,0]
	d01 = detector.response[0,1]
	d02 = detector.response[0,2]
	d11 = detector.response[1,1]
	d12 = detector.response[1,2]
	d22 = detector.response[2,2]

	eps = deps_cross_ddec(RA, dec)
	e00 = eps[0,0]
	e01 = eps[0,1]
	e02 = eps[0,2]
	e11 = eps[1,1]
	e12 = eps[1,2]
	e22 = eps[2,2]

	return d00*e00 + d11*e11 + d22*e22 + 2*(d01*e01 + d02*e02 + d12*e12)

def dlnA_dmchirp_dict(mchirp, eta):
	"""
	Computes the derivative of an inspiral signal's Ln[amplitude] with
	respect to the chirp mass. Returns as a dictionary for use in the
	metric calculation.
	"""
	dlnA_dmchirp = {}
	dlnA_dmchirp[0] = 5./(6.*mchirp)

	return dlnA_dmchirp

def dPhase_dmchirp_dict(mchirp, eta):
	"""
	Computes the derivative of an inspiral signal's phase with respect to
	the chirp mass. Returns as a dictionary for use in the metric
	calculation.
	"""
	dPhase_dmchirp = {}
	dPhase_dmchirp[-5] = 5./(128.*pi**(5./3.)*mchirp**(8./3.))
	dPhase_dmchirp[-3] = 3./(128.*pi*mchirp**2.)*(3715./756.+55./9.*eta)*eta**(-2./5.)
	dPhase_dmchirp[-2] = 2./(128.*pi**(2./3.)*mchirp**(5./3.))*(-16.*pi)*eta**(-3./5.)
	dPhase_dmchirp[-1] = 1./(128.*pi**(1./3.)*mchirp**(4./3.))*(15293365./508032.+27145./504.*eta+3085./72.*eta**2)*eta**(-4./5.)

	return dPhase_dmchirp

def dPhase_deta_dict(mchirp, eta):
	"""
	Computes the derivative of an inspiral signal's phase with respect to
	the symmetric mass ratio. Returns as a dictionary for use in the
	metric calculation.
	"""
	dPhase_deta = {}
	dPhase_deta[-3] = (-3./128.)*(3715./756.*(-2./5.)*eta**(-7./5.) + 55./9.*(3./5.)*eta**(-2./5.))*(pi*mchirp)**-1
	dPhase_deta[-2] = (-3./128.)*(-16.*pi*(-3./5.)*eta**(-8./5.))*(pi*mchirp)**(-2./3.)
	dPhase_deta[-1] = (-3./128.)*(15293365./508032.*(-4./5.)*eta**(-9./5.) + 27145./504.*(1./5.)*eta**(-4./5.) + 3085./72.*(6./5.)*eta**(1./5.))*(pi*mchirp)**(-1./3.)

	return dPhase_deta

def dPhase_dt_dict():
	"""
	Computes the derivative of an inspiral signal's phase with respect to
	the end time. Returns as a dictionary for use in the metric
	calculation.
	"""
	dPhase_dt = {}
	dPhase_dt[3] = 2*pi

	return 	dPhase_dt

# FIXME: should be rotating with time
def rn(RA, dec, detector):
	"""
	The inner product of n(RA,dec) and r_X for detector X.
	"""
	dx,dy,dz = detector.vertex
	tmp = dx*cos(RA)*cos(dec) + dy*sin(RA)*cos(dec) + dz*sin(dec)
	return tmp / c_speed

def drn_dRA(RA, dec, detector):
	"""
	The derivative of the inner product of n(RA,dec) and r_X for
	detector X with respect to the Right Ascension.
	"""
	dx,dy,dz = detector.vertex
	tmp = -dx*sin(RA)*cos(dec) + dy*cos(RA)*cos(dec)
	return tmp / c_speed

def drn_ddec(RA, dec, detector):
	"""
	The derivative of the inner product of n(RA,dec) and r_X for
	detector X with respect to the declination.
	"""
	dx,dy,dz = detector.vertex
	tmp = -dx*cos(RA)*sin(dec) - dy*sin(RA)*sin(dec) + dz*cos(dec)
	return tmp / c_speed

def mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs=False):
	"""
	Computes the first mismatch component associated with the derivatives
	with respect to parameters lambda_i and lambda_j. Take as input the
	list of detectors, the components of the maximum likelihood matrix,
	the plus and cross antenna responses for these detectors, derivatives
	of the plus and cross antenna responses with respect to lambda_i and
	lambda_j, the derivatives of the inspiral waveform phase with respect
	to lambda_i and lambda_j, the derivatives of the inspiral waveform
	Ln[amplitude] with respect to lambda_i and lambda_j, and a flag saying
	whether to include derivatives of the antenna responses in the
	calculation.
	"""
	h11ij = scipy.zeros(scipy.shape(A))
	R11i = scipy.zeros(scipy.shape(A))
	R11j = scipy.zeros(scipy.shape(A))
	R21i = scipy.zeros(scipy.shape(A))
	R21j = scipy.zeros(scipy.shape(A))
	R31i = scipy.zeros(scipy.shape(A))
	R31j = scipy.zeros(scipy.shape(A))
	R41i = scipy.zeros(scipy.shape(A))
	R41j = scipy.zeros(scipy.shape(A))
	for detector in detector_list:
		I_n = detector.I_n
		I_dLnAmp_dLnAmp = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			for key2 in dLnAmpj[detector.name].keys():
				n = -7 + key1 + key2
				I_dLnAmp_dLnAmp += dLnAmpi[detector.name][key1]*dLnAmpj[detector.name][key2]*I_n[n/3.]
		I_dPhase_dPhase = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			for key2 in dPhasej[detector.name].keys():
				n = -7 + key1 + key2
				I_dPhase_dPhase += dPhasei[detector.name][key1]*dPhasej[detector.name][key2]*I_n[n/3.]
		h11ij += fp[detector.name]*fp[detector.name] * (I_dLnAmp_dLnAmp + I_dPhase_dPhase)
		I_dLnAmpi = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			n = -7 + key1
			I_dLnAmpi += dLnAmpi[detector.name][key1]*I_n[n/3.]
		I_dLnAmpj = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpj[detector.name].keys():
			n = -7 + key1
			I_dLnAmpj += dLnAmpj[detector.name][key1]*I_n[n/3.]
		R11i += fp[detector.name]*fp[detector.name] * I_dLnAmpi
		R11j += fp[detector.name]*fp[detector.name] * I_dLnAmpj
		R21i += fx[detector.name]*fp[detector.name] * I_dLnAmpi
		R21j += fx[detector.name]*fp[detector.name] * I_dLnAmpj
		if Fderivs:
			h11ij += dfpi[detector.name]*fp[detector.name] * I_dLnAmpj
			h11ij += fp[detector.name]*dfpj[detector.name] * I_dLnAmpi
		I_dPhasei = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			n = -7 + key1
			I_dPhasei += dPhasei[detector.name][key1]*I_n[n/3.]
		I_dPhasej = scipy.zeros(scipy.shape(A))
		for key1 in dPhasej[detector.name].keys():
			n = -7 + key1
			I_dPhasej += dPhasej[detector.name][key1]*I_n[n/3.]
		R31i += fp[detector.name]*fp[detector.name] * I_dPhasei
		R31j += fp[detector.name]*fp[detector.name] * I_dPhasej
		R41i += fx[detector.name]*fp[detector.name] * I_dPhasei
		R41j += fx[detector.name]*fp[detector.name] * I_dPhasej
		if Fderivs:
			h11ij += dfpi[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R11i += fp[detector.name]*dfpi[detector.name] * I_n[-7./3.]
			R11j += fp[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R21i += fx[detector.name]*dfpi[detector.name] * I_n[-7./3.]
			R21j += fx[detector.name]*dfpj[detector.name] * I_n[-7./3.]

	p = h11ij
	q = B*(R11i*R11j + R31i*R31j) + A*(R21i*R21j + R41i*R41j)\
		- C*(R11i*R21j + R21i*R11j + R31i*R41j + R41i*R31j)
	m1 = p - q/D

	return m1

def mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs=False):
	"""
	Computes the second mismatch component associated with the derivatives
	with respect to parameters lambda_i and lambda_j. Take as input the
	list of detectors, the components of the maximum likelihood matrix,
	the plus and cross antenna responses for these detectors, derivatives
	of the plus and cross antenna responses with respect to lambda_i and
	lambda_j, the derivatives of the inspiral waveform phase with respect
	to lambda_i and lambda_j, the derivatives of the inspiral waveform
	Ln[amplitude] with respect to lambda_i and lambda_j, and a flag saying
	whether to include derivatives of the antenna responses in the
	calculation.
	"""
	h22ij = scipy.zeros(scipy.shape(A))
	R12i = scipy.zeros(scipy.shape(A))
	R12j = scipy.zeros(scipy.shape(A))
	R22i = scipy.zeros(scipy.shape(A))
	R22j = scipy.zeros(scipy.shape(A))
	R32i = scipy.zeros(scipy.shape(A))
	R32j = scipy.zeros(scipy.shape(A))
	R42i = scipy.zeros(scipy.shape(A))
	R42j = scipy.zeros(scipy.shape(A))
	for detector in detector_list:
		I_n = detector.I_n
		I_dLnAmp_dLnAmp = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			for key2 in dLnAmpj[detector.name].keys():
				n = -7 + key1 + key2
				I_dLnAmp_dLnAmp += dLnAmpi[detector.name][key1]*dLnAmpj[detector.name][key2]*I_n[n/3.]
		I_dPhase_dPhase = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			for key2 in dPhasej[detector.name].keys():
				n = -7 + key1 + key2
				I_dPhase_dPhase += dPhasei[detector.name][key1]*dPhasej[detector.name][key2]*I_n[n/3.]
		h22ij += fx[detector.name]*fx[detector.name] * (I_dLnAmp_dLnAmp + I_dPhase_dPhase)
		I_dLnAmpi = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			n = -7 + key1
			I_dLnAmpi += dLnAmpi[detector.name][key1]*I_n[n/3.]
		I_dLnAmpj = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpj[detector.name].keys():
			n = -7 + key1
			I_dLnAmpj += dLnAmpj[detector.name][key1]*I_n[n/3.]
		R12i += fp[detector.name]*fx[detector.name] * I_dLnAmpi
		R12j += fp[detector.name]*fx[detector.name] * I_dLnAmpj
		R22i += fx[detector.name]*fx[detector.name] * I_dLnAmpi
		R22j += fx[detector.name]*fx[detector.name] * I_dLnAmpj
		if Fderivs:
			h22ij += dfxi[detector.name]*fx[detector.name] * I_dLnAmpj
			h22ij += fx[detector.name]*dfxj[detector.name] * I_dLnAmpi
		I_dPhasei = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			n = -7 + key1
			I_dPhasei += dPhasei[detector.name][key1]*I_n[n/3.]
		I_dPhasej = scipy.zeros(scipy.shape(A))
		for key1 in dPhasej[detector.name].keys():
			n = -7 + key1
			I_dPhasej += dPhasej[detector.name][key1]*I_n[n/3.]
		R32i += fp[detector.name]*fx[detector.name] * I_dPhasei
		R32j += fp[detector.name]*fx[detector.name] * I_dPhasej
		R42i += fx[detector.name]*fx[detector.name] * I_dPhasei
		R42j += fx[detector.name]*fx[detector.name] * I_dPhasej
		if Fderivs:
			h22ij += dfxi[detector.name]*dfxj[detector.name] * I_n[-7./3.]
			R12i += fp[detector.name]*dfxi[detector.name] * I_n[-7./3.]
			R12j += fp[detector.name]*dfxj[detector.name] * I_n[-7./3.]
			R22i += fx[detector.name]*dfxi[detector.name] * I_n[-7./3.]
			R22j += fx[detector.name]*dfxj[detector.name] * I_n[-7./3.]

	p = h22ij
	q = B*(R12i*R12j + R32i*R32j) + A*(R22i*R22j + R42i*R42j)\
		- C*(R12i*R22j + R22i*R12j + R32i*R42j + R42i*R32j)
	m2 = p - q/D

	return m2

def mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs=False):
	"""
	Computes the third mismatch component associated with the derivatives
	with respect to parameters lambda_i and lambda_j. Take as input the
	list of detectors, the components of the maximum likelihood matrix,
	the plus and cross antenna responses for these detectors, derivatives
	of the plus and cross antenna responses with respect to lambda_i and
	lambda_j, the derivatives of the inspiral waveform phase with respect
	to lambda_i and lambda_j, the derivatives of the inspiral waveform
	Ln[amplitude] with respect to lambda_i and lambda_j, and a flag saying
	whether to include derivatives of the antenna responses in the
	calculation.
	"""
	h12ij = scipy.zeros(scipy.shape(A))
	h21ij = scipy.zeros(scipy.shape(A))
	R11i = scipy.zeros(scipy.shape(A))
	R11j = scipy.zeros(scipy.shape(A))
	R12i = scipy.zeros(scipy.shape(A))
	R12j = scipy.zeros(scipy.shape(A))
	R21i = scipy.zeros(scipy.shape(A))
	R21j = scipy.zeros(scipy.shape(A))
	R22i = scipy.zeros(scipy.shape(A))
	R22j = scipy.zeros(scipy.shape(A))
	R31i = scipy.zeros(scipy.shape(A))
	R31j = scipy.zeros(scipy.shape(A))
	R32i = scipy.zeros(scipy.shape(A))
	R32j = scipy.zeros(scipy.shape(A))
	R41i = scipy.zeros(scipy.shape(A))
	R41j = scipy.zeros(scipy.shape(A))
	R42i = scipy.zeros(scipy.shape(A))
	R42j = scipy.zeros(scipy.shape(A))
	for detector in detector_list:
		I_n = detector.I_n
		I_dLnAmp_dLnAmp = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			for key2 in dLnAmpj[detector.name].keys():
				n = -7 + key1 + key2
				I_dLnAmp_dLnAmp += dLnAmpi[detector.name][key1]*dLnAmpj[detector.name][key2]*I_n[n/3.]
		I_dPhase_dPhase = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			for key2 in dPhasej[detector.name].keys():
				n = -7 + key1 + key2
				I_dPhase_dPhase += dPhasei[detector.name][key1]*dPhasej[detector.name][key2]*I_n[n/3.]
		h12ij += fp[detector.name]*fx[detector.name] * (I_dLnAmp_dLnAmp + I_dPhase_dPhase)
		h21ij += fx[detector.name]*fp[detector.name] * (I_dLnAmp_dLnAmp + I_dPhase_dPhase)
		I_dLnAmpi = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			n = -7 + key1
			I_dLnAmpi += dLnAmpi[detector.name][key1]*I_n[n/3.]
		I_dLnAmpj = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpj[detector.name].keys():
			n = -7 + key1
			I_dLnAmpj += dLnAmpj[detector.name][key1]*I_n[n/3.]
		R11i += fp[detector.name]*fp[detector.name] * I_dLnAmpi
		R11j += fp[detector.name]*fp[detector.name] * I_dLnAmpj
		R12i += fp[detector.name]*fx[detector.name] * I_dLnAmpi
		R12j += fp[detector.name]*fx[detector.name] * I_dLnAmpj
		R21i += fx[detector.name]*fp[detector.name] * I_dLnAmpi
		R21j += fx[detector.name]*fp[detector.name] * I_dLnAmpj
		R22i += fx[detector.name]*fx[detector.name] * I_dLnAmpi
		R22j += fx[detector.name]*fx[detector.name] * I_dLnAmpj
		if Fderivs:
			h12ij += dfpi[detector.name]*fx[detector.name] * I_dLnAmpj
			h12ij += fp[detector.name]*dfxj[detector.name] * I_dLnAmpi
			h21ij += dfxi[detector.name]*fp[detector.name] * I_dLnAmpj
			h21ij += fx[detector.name]*dfpj[detector.name] * I_dLnAmpi
		I_dPhasei = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			n = -7 + key1
			I_dPhasei += dPhasei[detector.name][key1]*I_n[n/3.]
		I_dPhasej = scipy.zeros(scipy.shape(A))
		for key1 in dPhasej[detector.name].keys():
			n = -7 + key1
			I_dPhasej += dPhasej[detector.name][key1]*I_n[n/3.]
		R31i += fp[detector.name]*fp[detector.name] * I_dPhasei
		R31j += fp[detector.name]*fp[detector.name] * I_dPhasej
		R32i += fp[detector.name]*fx[detector.name] * I_dPhasei
		R32j += fp[detector.name]*fx[detector.name] * I_dPhasej
		R41i += fx[detector.name]*fp[detector.name] * I_dPhasei
		R41j += fx[detector.name]*fp[detector.name] * I_dPhasej
		R42i += fx[detector.name]*fx[detector.name] * I_dPhasei
		R42j += fx[detector.name]*fx[detector.name] * I_dPhasej
		if Fderivs:
			h12ij += dfpi[detector.name]*dfxj[detector.name] * I_n[-7./3.]
			h21ij += dfxi[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R11i += fp[detector.name]*dfpi[detector.name] * I_n[-7./3.]
			R11j += fp[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R12i += fp[detector.name]*dfxi[detector.name] * I_n[-7./3.]
			R12j += fp[detector.name]*dfxj[detector.name] * I_n[-7./3.]
			R21i += fx[detector.name]*dfpi[detector.name] * I_n[-7./3.]
			R21j += fx[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R22i += fx[detector.name]*dfxi[detector.name] * I_n[-7./3.]
			R22j += fx[detector.name]*dfxj[detector.name] * I_n[-7./3.]

	p = h12ij + h21ij
	q  = B*(R11i*R12j + R31i*R32j) + A*(R21i*R22j + R41i*R42j)\
		- C*(R11i*R22j + R21i*R12j + R31i*R42j + R41i*R32j)
	q += B*(R12i*R11j + R32i*R31j) + A*(R22i*R21j + R42i*R41j)\
		- C*(R12i*R21j + R22i*R11j + R32i*R41j + R42i*R31j)
	m3 = p - q/D
	m3 /= 2.

	return m3

def mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs=False):
	"""
	Computes the fourth mismatch component associated with the derivatives
	with respect to parameters lambda_i and lambda_j. Take as input the
	list of detectors, the components of the maximum likelihood matrix,
	the plus and cross antenna responses for these detectors, derivatives
	of the plus and cross antenna responses with respect to lambda_i and
	lambda_j, the derivatives of the inspiral waveform phase with respect
	to lambda_i and lambda_j, the derivatives of the inspiral waveform
	Ln[amplitude] with respect to lambda_i and lambda_j, and a flag saying
	whether to include derivatives of the antenna responses in the
	calculation.
	"""
	h14ij = scipy.zeros(scipy.shape(A))
	h41ij = scipy.zeros(scipy.shape(A))
	R11i = scipy.zeros(scipy.shape(A))
	R11j = scipy.zeros(scipy.shape(A))
	R14i = scipy.zeros(scipy.shape(A))
	R14j = scipy.zeros(scipy.shape(A))
	R21i = scipy.zeros(scipy.shape(A))
	R21j = scipy.zeros(scipy.shape(A))
	R24i = scipy.zeros(scipy.shape(A))
	R24j = scipy.zeros(scipy.shape(A))
	R31i = scipy.zeros(scipy.shape(A))
	R31j = scipy.zeros(scipy.shape(A))
	R34i = scipy.zeros(scipy.shape(A))
	R34j = scipy.zeros(scipy.shape(A))
	R41i = scipy.zeros(scipy.shape(A))
	R41j = scipy.zeros(scipy.shape(A))
	R44i = scipy.zeros(scipy.shape(A))
	R44j = scipy.zeros(scipy.shape(A))
	for detector in detector_list:
		I_n = detector.I_n
		I_dLnAmpi = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpi[detector.name].keys():
			n = -7 + key1
			I_dLnAmpi += dLnAmpi[detector.name][key1]*I_n[n/3.]
		I_dLnAmpj = scipy.zeros(scipy.shape(A))
		for key1 in dLnAmpj[detector.name].keys():
			n = -7 + key1
			I_dLnAmpj += dLnAmpj[detector.name][key1]*I_n[n/3.]
		R11i += fp[detector.name]*fp[detector.name] * I_dLnAmpi
		R11j += fp[detector.name]*fp[detector.name] * I_dLnAmpj
		R21i += fx[detector.name]*fp[detector.name] * I_dLnAmpi
		R21j += fx[detector.name]*fp[detector.name] * I_dLnAmpj
		R34i += fp[detector.name]*fx[detector.name] * I_dLnAmpi
		R34j += fp[detector.name]*fx[detector.name] * I_dLnAmpj
		R44i += fx[detector.name]*fx[detector.name] * I_dLnAmpi
		R44j += fx[detector.name]*fx[detector.name] * I_dLnAmpj
		I_dPhasei = scipy.zeros(scipy.shape(A))
		for key1 in dPhasei[detector.name].keys():
			n = -7 + key1
			I_dPhasei += dPhasei[detector.name][key1]*I_n[n/3.]
		I_dPhasej = scipy.zeros(scipy.shape(A))
		for key1 in dPhasej[detector.name].keys():
			n = -7 + key1
			I_dPhasej += dPhasej[detector.name][key1]*I_n[n/3.]
		R14i -= fp[detector.name]*fx[detector.name] * I_dPhasei
		R14j -= fp[detector.name]*fx[detector.name] * I_dPhasej
		R24i -= fx[detector.name]*fx[detector.name] * I_dPhasei
		R24j -= fx[detector.name]*fx[detector.name] * I_dPhasej
		R31i += fp[detector.name]*fp[detector.name] * I_dPhasei
		R31j += fp[detector.name]*fp[detector.name] * I_dPhasej
		R41i += fx[detector.name]*fp[detector.name] * I_dPhasei
		R41j += fx[detector.name]*fp[detector.name] * I_dPhasej
		if Fderivs:
			h14ij += (fp[detector.name]*dfxi[detector.name] - dfpi[detector.name]*fx[detector.name]) * I_dPhasej
			h41ij += (dfxj[detector.name]*fp[detector.name] - fx[detector.name]*dfpj[detector.name]) * I_dPhasei
			R11i += fp[detector.name]*dfpi[detector.name] * I_n[-7./3.]
			R11j += fp[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R21i += fx[detector.name]*dfpi[detector.name] * I_n[-7./3.]
			R21j += fx[detector.name]*dfpj[detector.name] * I_n[-7./3.]
			R34i += fp[detector.name]*dfxi[detector.name] * I_n[-7./3.]
			R34j += fp[detector.name]*dfxj[detector.name] * I_n[-7./3.]
			R44i += fx[detector.name]*dfxi[detector.name] * I_n[-7./3.]
			R44j += fx[detector.name]*dfxj[detector.name] * I_n[-7./3.]

	p = h14ij + h41ij
	q  = B*(R11i*R14j + R31i*R34j) + A*(R21i*R24j + R41i*R44j)\
		- C*(R11i*R24j + R21i*R14j + R31i*R44j + R41i*R34j)
	q += B*(R14i*R11j + R34i*R31j) + A*(R24i*R21j + R44i*R41j)\
		- C*(R24i*R11j + R14i*R21j + R44i*R31j + R34i*R41j)
	m4 = p - q/D
	m4 /= 2.

	return m4

def g_A1_A1(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st amplitude parameter.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	z = A*scipy.ones(scipy.shape(RA))
	return z

def g_A1_A2(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st and 2nd amplitude
	parameters.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	z = C*scipy.ones(scipy.shape(RA))
	return z

def g_A1_A3(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st and 3rd amplitude
	parameters.
	"""
	z = 0.*scipy.ones(scipy.shape(RA))
	return z

def g_A1_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st and 4th amplitude
	parameters.
	"""
	z = 0.*scipy.ones(scipy.shape(RA))
	return z

def g_A2_A2(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd amplitude parameters.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	z = B*scipy.ones(scipy.shape(RA))
	return z

def g_A2_A3(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd and 3rd amplitude
	parameters.
	"""
	z = 0.*scipy.ones(scipy.shape(RA))
	return z

def g_A2_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd and 4th amplitude
	parameters.
	"""
	z = 0.*scipy.ones(scipy.shape(RA))
	return z

def g_A3_A3(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd amplitude parameter.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	z = A*scipy.ones(scipy.shape(RA))
	return z

def g_A3_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd and 4th amplitude
	parameters.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	z = C*scipy.ones(scipy.shape(RA))
	return z

def g_A4_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 4th amplitude parameter.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	z = B*scipy.ones(scipy.shape(RA))
	return z

def g_A1_RA(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st amplitude parameter
	and the Right Ascension.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			n = -7 + key1
			z_deriv += dPhase_dRA1[key1]*I_n1[n/3.]
		z += (-A3*fp1*fp1 - A4*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z += (A1*fp1*dfp_dRA1 + A2*fp1*dfx_dRA1) * I_n1[-7./3.]
	return z

def g_A2_RA(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd amplitude parameter
	and the Right Ascension.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			n = -7 + key1
			z_deriv += dPhase_dRA1[key1]*I_n1[n/3.]
		z += (-A3*fx1*fp1 - A4*fx1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z += (A1*fx1*dfp_dRA1 + A2*fx1*dfx_dRA1) * I_n1[-7./3.]
	return z

def g_A3_RA(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd amplitude parameter
	and the Right Ascension.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			n = -7 + key1
			z_deriv += dPhase_dRA1[key1]*I_n1[n/3.]
		z += (A1*fp1*fp1 + A2*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z += (A3*fp1*dfp_dRA1 + A4*fp1*dfx_dRA1) * I_n1[-7./3.]
	return z

def g_A4_RA(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 4th amplitude parameter
	and the Right Ascension.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			n = -7 + key1
			z_deriv += dPhase_dRA1[key1]*I_n1[n/3.]
		z += (A1*fx1*fp1 + A2*fx1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z += (A3*fx1*dfp_dRA1 + A4*fx1*dfx_dRA1) * I_n1[-7./3.]
	return z

def g_A1_dec(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st amplitude parameter
	and the declination.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			n = -7 + key1
			z_deriv += dPhase_ddec1[key1]*I_n1[n/3.]
		z += (-A3*fp1*fp1 - A4*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z += (A1*fp1*dfp_ddec1 + A2*fp1*dfx_ddec1) * I_n1[-7./3.]
	return z

def g_A2_dec(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd amplitude parameter
	and the declination.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			n = -7 + key1
			z_deriv += dPhase_ddec1[key1]*I_n1[n/3.]
		z += (-A3*fx1*fp1 - A4*fx1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z += (A1*fx1*dfp_ddec1 + A2*fx1*dfx_ddec1) * I_n1[-7./3.]
	return z

def g_A3_dec(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd amplitude parameter
	and the declination.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			n = -7 + key1
			z_deriv += dPhase_ddec1[key1]*I_n1[n/3.]
		z += (A1*fp1*fp1 + A2*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z += (A3*fp1*dfp_ddec1 + A4*fp1*dfx_ddec1) * I_n1[-7./3.]
	return z

def g_A4_dec(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 4th amplitude parameter
	and the declination.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			n = -7 + key1
			z_deriv += dPhase_ddec1[key1]*I_n1[n/3.]
		z += (A1*fx1*fp1 + A2*fx1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z += (A3*fx1*dfp_ddec1 + A4*fx1*dfx_ddec1) * I_n1[-7./3.]
	return z

def g_A1_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st amplitude parameter
	and the end time.
	"""
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			n = -7 + key1
			z_deriv += dPhase_dt[key1]*I_n1[n/3.]
		z += (-A3*fp1*fp1 - A4*fp1*fx1) * z_deriv
	return z

def g_A2_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd amplitude parameter
	and the end time.
	"""
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			n = -7 + key1
			z_deriv += dPhase_dt[key1]*I_n1[n/3.]
		z += (-A3*fx1*fp1 - A4*fx1*fx1) * z_deriv
	return z

def g_A3_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd amplitude parameter
	and the end time.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			n = -7 + key1
			z_deriv += dPhase_dt[key1]*I_n1[n/3.]
		z += (A1*fp1*fp1 + A2*fp1*fx1) * z_deriv
	return z

def g_A4_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 4th amplitude parameter
	and the end time.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			n = -7 + key1
			z_deriv += dPhase_dt[key1]*I_n1[n/3.]
		z += (A1*fx1*fp1 + A2*fx1*fx1) * z_deriv
	return z

def g_A1_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st amplitude parameter
	and the chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dlnA_dmchirp.keys():
			n = -7 + key1
			z_deriv += dlnA_dmchirp[key1]*I_n1[n/3.]
		z += (A1*fp1*fp1 + A2*fp1*fx1) * z_deriv
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dmchirp.keys():
			n = -7 + key1
			z_deriv += dPhase_dmchirp[key1]*I_n1[n/3.]
		z += (-A3*fp1*fp1 - A4*fp1*fx1) * z_deriv
	return z

def g_A2_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd amplitude parameter
	and the chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dlnA_dmchirp.keys():
			n = -7 + key1
			z_deriv += dlnA_dmchirp[key1]*I_n1[n/3.]
		z += (A1*fx1*fp1 + A2*fx1*fx1) * z_deriv
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dmchirp.keys():
			n = -7 + key1
			z_deriv += dPhase_dmchirp[key1]*I_n1[n/3.]
		z += (-A3*fx1*fp1 - A4*fx1*fx1) * z_deriv
	return z

def g_A3_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd amplitude parameter
	and the chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dlnA_dmchirp.keys():
			n = -7 + key1
			z_deriv += dlnA_dmchirp[key1]*I_n1[n/3.]
		z += (A3*fp1*fp1 + A4*fp1*fx1) * z_deriv
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dmchirp.keys():
			n = -7 + key1
			z_deriv += dPhase_dmchirp[key1]*I_n1[n/3.]
		z += (A1*fp1*fp1 + A2*fp1*fx1) * z_deriv
	return z

def g_A4_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 4th amplitude parameter
	and the chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dlnA_dmchirp.keys():
			n = -7 + key1
			z_deriv += dlnA_dmchirp[key1]*I_n1[n/3.]
		z += (A3*fx1*fp1 + A4*fx1*fx1) * z_deriv
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dmchirp.keys():
			n = -7 + key1
			z_deriv += dPhase_dmchirp[key1]*I_n1[n/3.]
		z += (A1*fx1*fp1 + A2*fx1*fx1) * z_deriv
	return z

def g_A1_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 1st amplitude parameter
	and the symmetric mass ratio.
	"""
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_deta.keys():
			n = -7 + key1
			z_deriv += dPhase_deta[key1]*I_n1[n/3.]
		z += (-A3*fp1*fp1 - A4*fp1*fx1) * z_deriv
	return z

def g_A2_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 2nd amplitude parameter
	and the symmetric mass ratio.
	"""
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_deta.keys():
			n = -7 + key1
			z_deriv += dPhase_deta[key1]*I_n1[n/3.]
		z += (-A3*fx1*fp1 - A4*fx1*fx1) * z_deriv
	return z

def g_A3_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 3rd amplitude parameter
	and the symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_deta.keys():
			n = -7 + key1
			z_deriv += dPhase_deta[key1]*I_n1[n/3.]
		z += (A1*fp1*fp1 + A2*fp1*fx1) * z_deriv
	return z

def g_A4_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the 4th amplitude parameter
	and the symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_deta.keys():
			n = -7 + key1
			z_deriv += dPhase_deta[key1]*I_n1[n/3.]
		z += (A1*fx1*fp1 + A2*fx1*fx1) * z_deriv
	return z

def g_RA_RA(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the Right Ascension.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			for key2 in dPhase_dRA1.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dRA1[key1]*dPhase_dRA1[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_dRA1.keys():
				n = -7 + key1 
				z_deriv += dPhase_dRA1[key1]*I_n1[n/3.]
			z += 2*(A1*A4*(fp1*dfx_dRA1 - fx1*dfp_dRA1) + A2*A3*(fx1*dfp_dRA1 - fp1*dfx_dRA1)) * z_deriv
			z += ((A1*A1 + A3*A3)*dfp_dRA1*dfp_dRA1 + (A2*A2 + A4*A4)*dfx_dRA1*dfx_dRA1 \
				+ (A1*A2 + A3*A4)*(dfp_dRA1*dfx_dRA1 + dfx_dRA1*dfp_dRA1)) \
				*I_n1[-7./3.]
	return z

def m1_RA_RA(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the Right Ascension.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_dRA(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_dRA(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_RA_RA(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the Right Ascension.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_dRA(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_dRA(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_RA_RA(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the Right Ascension.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_dRA(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_dRA(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_RA_RA(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the Right Ascension.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_dRA(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_dRA(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_RA_RA(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the Right Ascension.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_RA(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_RA(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_RA_RA(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_RA_RA(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the Right Ascension.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_RA(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_RA(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_RA_dec(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the Right Ascension and
	declination.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			for key2 in dPhase_ddec1.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dRA1[key1]*dPhase_ddec1[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_dRA1.keys():
				n = -7 + key1 
				z_deriv += dPhase_dRA1[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_ddec1 - fx1*dfp_ddec1) + A2*A3*(fx1*dfp_ddec1 - fp1*dfx_ddec1)) * z_deriv
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_ddec1.keys():
				n = -7 + key1 
				z_deriv += dPhase_ddec1[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_dRA1 - fx1*dfp_dRA1) + A2*A3*(fx1*dfp_dRA1 - fp1*dfx_dRA1)) * z_deriv
			z += ((A1*A1 + A3*A3)*dfp_dRA1*dfp_ddec1 + (A2*A2 + A4*A4)*dfx_dRA1*dfx_ddec1 \
				+ (A1*A2 + A3*A4)*(dfp_dRA1*dfx_ddec1 + dfx_dRA1*dfp_ddec1)) \
				*I_n1[-7./3.]
	return z

def m1_RA_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the Right Ascension and declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_RA_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the Right Ascension and declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_RA_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the Right Ascension and declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_RA_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the Right Ascension and declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_RA_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the Right Ascension and declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_dec(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_dec(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_RA_dec(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_RA_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the Right Ascension and declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_dec(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_dec(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_dec_dec(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the declination.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			for key2 in dPhase_ddec1.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_ddec1[key1]*dPhase_ddec1[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_ddec1.keys():
				n = -7 + key1 
				z_deriv += dPhase_ddec1[key1]*I_n1[n/3.]
			z += 2*(A1*A4*(fp1*dfx_ddec1 - fx1*dfp_ddec1) + A2*A3*(fx1*dfp_ddec1 - fp1*dfx_ddec1)) * z_deriv
			z += ((A1*A1 + A3*A3)*dfp_ddec1*dfp_ddec1 + (A2*A2 + A4*A4)*dfx_ddec1*dfx_ddec1 \
				+ (A1*A2 + A3*A4)*(dfp_ddec1*dfx_ddec1 + dfx_ddec1*dfp_ddec1)) \
				*I_n1[-7./3.]
	return z

def m1_dec_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_dec_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_dec_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_dec_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = dFp_ddec(RA, dec, detector)
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = dFx_ddec(RA, dec, detector)
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = {}
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		drnj = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni
		dPhasej[detector.name][3] = -2*pi*drnj

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_dec_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_dec(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_dec(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_dec_dec(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_dec_dec(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the declination.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_dec(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_dec(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_RA_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the Right Ascension and the
	end time.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			for key2 in dPhase_dt.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dRA1[key1]*dPhase_dt[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_dt.keys():
				n = -7 + key1 
				z_deriv += dPhase_dt[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_dRA1 - fx1*dfp_dRA1) + A2*A3*(fx1*dfp_dRA1 - fp1*dfx_dRA1)) * z_deriv
	return z

def m1_RA_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the Right Ascension and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_RA_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the Right Ascension and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_RA_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the Right Ascension and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_RA_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the Right Ascension and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_RA_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the Right Ascension and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_RA_t(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_RA_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the Right Ascension and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_t(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_dec_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the declination and the
	end time.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			for key2 in dPhase_dt.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_ddec1[key1]*dPhase_dt[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_dt.keys():
				n = -7 + key1 
				z_deriv += dPhase_dt[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_ddec1 - fx1*dfp_ddec1) + A2*A3*(fx1*dfp_ddec1 - fp1*dfx_ddec1)) * z_deriv
	return z

def m1_dec_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the declination and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_dec_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the declination and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_dec_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the declination and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_dec_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the declination and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_dec_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the declination and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_dec_t(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_dec_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the declination and the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_t(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_RA_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the Right Ascension and the
	chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			for key2 in dPhase_dmchirp.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dRA1[key1]*dPhase_dmchirp[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dlnA_dmchirp.keys():
				n = -7 + key1 
				z_deriv += dlnA_dmchirp[key1]*I_n1[n/3.]
			z += ((A1*A1 + A3*A3)*fp1*dfp_dRA1 + (A2*A2 + A4*A4)*fx1*dfx_dRA1 \
				+ (A1*A2 + A3*A4)*(fp1*dfx_dRA1 + fx1*dfp_dRA1)) * z_deriv
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_dmchirp.keys():
				n = -7 + key1 
				z_deriv += dPhase_dmchirp[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_dRA1 - fx1*dfp_dRA1) + A2*A3*(fx1*dfp_dRA1 - fp1*dfx_dRA1)) * z_deriv
	return z

def m1_RA_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the Right Ascension and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_RA_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the Right Ascension and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_RA_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the Right Ascension and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_RA_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the Right Ascension and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_RA_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the Right Ascension and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_RA_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_RA_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the Right Ascension and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_dec_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the declination and the
	chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			for key2 in dPhase_dmchirp.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_ddec1[key1]*dPhase_dmchirp[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dlnA_dmchirp.keys():
				n = -7 + key1 
				z_deriv += dlnA_dmchirp[key1]*I_n1[n/3.]
			z += ((A1*A1 + A3*A3)*fp1*dfp_ddec1 + (A2*A2 + A4*A4)*fx1*dfx_ddec1 \
				+ (A1*A2 + A3*A4)*(fp1*dfx_ddec1 + fx1*dfp_ddec1)) * z_deriv
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_dmchirp.keys():
				n = -7 + key1 
				z_deriv += dPhase_dmchirp[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_ddec1 - fx1*dfp_ddec1) + A2*A3*(fx1*dfp_ddec1 - fp1*dfx_ddec1)) * z_deriv
	return z

def m1_dec_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the declination and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_dec_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the declination and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_dec_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the declination and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_dec_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the declination and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_dec_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the declination and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_dec_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_dec_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the declination and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_RA_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the Right Ascension and the
	symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_dRA1 = drn_dRA(RA, dec, detector1)
		dPhase_dRA1 = {}
		dPhase_dRA1[3] = -2*pi*drn_dRA1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dRA1.keys():
			for key2 in dPhase_deta.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dRA1[key1]*dPhase_deta[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_dRA1 = dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = dFx_dRA(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_deta.keys():
				n = -7 + key1 
				z_deriv += dPhase_deta[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_dRA1 - fx1*dfp_dRA1) + A2*A3*(fx1*dfp_dRA1 - fp1*dfx_dRA1)) * z_deriv
	return z

def m1_RA_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the Right Ascension and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_RA_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the Right Ascension and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_RA_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the Right Ascension and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_RA_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the Right Ascension and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_dRA(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_dRA(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_dRA(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_RA_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the Right Ascension and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_RA_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_RA_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the Right Ascension and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_RA_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_RA_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_dec_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the declination and the
	symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		drn_ddec1 = drn_ddec(RA, dec, detector1)
		dPhase_ddec1 = {}
		dPhase_ddec1[3] = -2*pi*drn_ddec1
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_ddec1.keys():
			for key2 in dPhase_deta.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_ddec1[key1]*dPhase_deta[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
		if Fderivs:
			dfp_ddec1 = dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = dFx_ddec(RA, dec, detector1)
			z_deriv = scipy.zeros(scipy.shape(RA))
			for key1 in dPhase_deta.keys():
				n = -7 + key1 
				z_deriv += dPhase_deta[key1]*I_n1[n/3.]
			z += (A1*A4*(fp1*dfx_ddec1 - fx1*dfp_ddec1) + A2*A3*(fx1*dfp_ddec1 - fp1*dfx_ddec1)) * z_deriv
	return z

def m1_dec_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the declination and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_dec_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the declination and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_dec_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the declination and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_dec_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the declination and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = dFp_ddec(RA, dec, detector)
		dfpj[detector.name] = 0.
		dfxi[detector.name] = dFx_ddec(RA, dec, detector)
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = {}
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}
		drni = drn_ddec(RA, dec, detector)
		dPhasei[detector.name][3] = -2*pi*drni

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_dec_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the declination and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_dec_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_dec_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the declination and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_dec_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_dec_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_t_t(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the end time.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			for key2 in dPhase_dt.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dt[key1]*dPhase_dt[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
	return z

def m1_t_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_t_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_t_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_t_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dt
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_t_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_t_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_t_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_t_t(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_t_t(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the end time.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_t_t(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_t_t(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_t_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the end time and the
	chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			for key2 in dPhase_dmchirp.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dt[key1]*dPhase_dmchirp[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
	return z

def m1_t_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the end time and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_t_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the end time and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_t_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the end time and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_t_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the end time and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_t_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the end time and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_t_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_t_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_t_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_t_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the end time and the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_t_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_t_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_t_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the end time and the
	symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_dt = dPhase_dt_dict()
	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dt.keys():
			for key2 in dPhase_deta.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dt[key1]*dPhase_deta[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
	return z

def m1_t_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the end time and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_t_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the end time and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_t_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the end time and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_t_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the end time and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_dt = dPhase_dt_dict()
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dt
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_t_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the end time and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_t_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_t_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_t_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_t_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the end time and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_t_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_t_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_mchirp_mchirp(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the chirp mass.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dlnA_dmchirp.keys():
			for key2 in dlnA_dmchirp.keys():
				n = -7 + key1 + key2
				z_deriv += dlnA_dmchirp[key1]*dlnA_dmchirp[key2]*I_n1[n/3.]
		for key1 in dPhase_dmchirp.keys():
			for key2 in dPhase_dmchirp.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dmchirp[key1]*dPhase_dmchirp[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
	return z

def m1_mchirp_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_mchirp_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_mchirp_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_mchirp_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_dmchirp
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = dlnA_dmchirp

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_mchirp_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_mchirp_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_mchirp_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_mchirp_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_mchirp_mchirp(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the chirp mass.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_mchirp_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_mchirp_mchirp(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_mchirp_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the chirp mass and the
	symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_dmchirp.keys():
			for key2 in dPhase_deta.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_dmchirp[key1]*dPhase_deta[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
	return z

def m1_mchirp_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the chirp mass and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = {}

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_mchirp_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the chirp mass and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = {}

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_mchirp_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the chirp mass and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = {}

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_mchirp_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the chirp mass and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dlnA_dmchirp = dlnA_dmchirp_dict(mchirp, eta)
	dPhase_dmchirp = dPhase_dmchirp_dict(mchirp, eta)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_dmchirp
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = dlnA_dmchirp
		dLnAmpj[detector.name] = {}

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_mchirp_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the chirp mass and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_mchirp_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_mchirp_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m3 = m3_mchirp_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_mchirp_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the chirp mass and the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_mchirp_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_mchirp_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= A + B

	return gbar

def g_eta_eta(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full metric component associated with the symmetric mass ratio.
	"""
	A1 = A_1(cosi, distance, phi0, psi)
	A2 = A_2(cosi, distance, phi0, psi)
	A3 = A_3(cosi, distance, phi0, psi)
	A4 = A_4(cosi, distance, phi0, psi)

	dPhase_deta = dPhase_deta_dict(mchirp, eta)

	z = scipy.zeros(scipy.shape(RA))
	for detector1 in detector_list:
		I_n1 = detector1.I_n
		fp1 = Fp(RA, dec, detector1)
		fx1 = Fx(RA, dec, detector1)
		z_deriv = scipy.zeros(scipy.shape(RA))
		for key1 in dPhase_deta.keys():
			for key2 in dPhase_deta.keys():
				n = -7 + key1 + key2
				z_deriv += dPhase_deta[key1]*dPhase_deta[key2]*I_n1[n/3.]
		z += ((A1*A1 + A3*A3)*fp1*fp1 + (A2*A2 + A4*A4)*fx1*fx1 \
			+ 2*(A1*A2 + A3*A4)*fp1*fx1) * z_deriv
	return z

def m1_eta_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 1st mismatch component associated
	with the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_deta
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch1(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m2_eta_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 2nd mismatch component associated
	with the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_deta
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch2(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m3_eta_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 3rd mismatch component associated
	with the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_deta
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch3(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def m4_eta_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized metric's 4th mismatch component associated
	with the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)
	dPhase_deta = dPhase_deta_dict(mchirp, eta)
	dPhasei = {}
	dPhasej = {}
	dLnAmpi = {}
	dLnAmpj = {}
	fp = {}
	fx = {} 
	dfpi = {}
	dfpj = {}
	dfxi = {}
	dfxj = {}
	for detector in detector_list:
		fp[detector.name] = Fp(RA, dec, detector)
		fx[detector.name] = Fx(RA, dec, detector)
		dfpi[detector.name] = 0.
		dfpj[detector.name] = 0.
		dfxi[detector.name] = 0.
		dfxj[detector.name] = 0.
		dPhasei[detector.name] = dPhase_deta
		dPhasej[detector.name] = dPhase_deta
		dLnAmpi[detector.name] = {}
		dLnAmpj[detector.name] = {}

	return mismatch4(detector_list, A, B, C, D, fp, fx, dfpi, dfpj, dfxi, dfxj, dPhasei, dPhasej, dLnAmpi, dLnAmpj, Fderivs)

def gbar_eta_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged metric component
	associated with the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_eta_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_eta_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = B*m1 + A*m2 - 2.*C*m3
	gbar /= 2.*D

	return gbar

def ogbar_eta_eta(RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged metric component associated
	with the symmetric mass ratio.
	"""
	A,B,C,D = maximum_likelihood_matrix(RA, dec, detector_list)

	m1 = m1_eta_eta(RA, dec, detector_list, mchirp, eta, Fderivs)
	m2 = m2_eta_eta(RA, dec, detector_list, mchirp, eta, Fderivs)

	gbar = m1 + m2
	gbar /= (A + B)

	return gbar

def full_metric(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The full amplitude-sky-mass-time metric is symmetric and is defined
	with the following signature:
	                  0    1   2   3   4    5   6    7    8
	                  A1   A2  A3  A4  RA  dec  t mchirp eta
	     0   A1    /  A    J   K   L   M    N   O    P    Q  \
	     1   A2   |   -    B   R   S   T    U   V    W    X   |
	     2   A3   |   -    -   C   Y   Z    AA  AB   AC   AD  |
	     3   A4   |   -    -   -   D   AE   AF  AG   AH   AI  |
	 g = 4   RA   |   -    -   -   -   E    AJ  AK   AL   AM  |
	     5  dec   |   -    -   -   -   -    F   AN   AO   AP  |
	     6   t    |   -    -   -   -   -    -   G    AQ   AR  |
	     7 mchirp |   -    -   -   -   -    -   -    H    AS  |
	     8  eta    \  -    -   -   -   -    -   -    -    I  /
	"""
	g = scipy.zeros((9,9))
	# the amplitude-amplitude block
	g[0,0] =		g_A1_A1(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,1] = g[1,0] =	g_A1_A2(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,2] = g[2,0] =	g_A1_A3(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,3] = g[3,0] =	g_A1_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,1] =		g_A2_A2(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,2] = g[2,1] =	g_A2_A3(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,3] = g[3,1] =	g_A2_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,2] =		g_A3_A3(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,3] = g[3,2] =	g_A3_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[3,3] =		g_A4_A4(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)

	# the amplitude-phase block
	g[0,4] = g[4,0] =	g_A1_RA		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,5] = g[5,0] =	g_A1_dec	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,6] = g[6,0] =	g_A1_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,7] = g[7,0] =	g_A1_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[0,8] = g[8,0] =	g_A1_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,4] = g[4,1] =	g_A2_RA		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,5] = g[5,1] =	g_A2_dec	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,6] = g[6,1] =	g_A2_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,7] = g[7,1] =	g_A2_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[1,8] = g[8,1] =	g_A2_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,4] = g[4,2] =	g_A3_RA		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,5] = g[5,2] =	g_A3_dec	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,6] = g[6,2] =	g_A3_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,7] = g[7,2] =	g_A3_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[2,8] = g[8,2] =	g_A3_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[3,4] = g[4,3] =	g_A4_RA		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[3,5] = g[5,3] =	g_A4_dec	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[3,6] = g[6,3] =	g_A4_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[3,7] = g[7,3] =	g_A4_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[3,8] = g[8,3] =	g_A4_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)


	# the phase-phase block
	g[4,4] =		g_RA_RA		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[4,5] = g[5,4] =	g_RA_dec	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[4,6] = g[6,4] =	g_RA_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[4,7] = g[7,4] =	g_RA_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[4,8] = g[8,4] =	g_RA_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[5,5] =		g_dec_dec	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[5,6] = g[6,5] =	g_dec_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[5,7] = g[7,5] =	g_dec_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[5,8] = g[8,5] =	g_dec_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[6,6] =		g_t_t		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[6,7] = g[7,6] =	g_t_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[6,8] = g[8,6] =	g_t_eta		(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[7,7] =		g_mchirp_mchirp	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[7,8] = g[8,7] =	g_mchirp_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)
	g[8,8] =		g_eta_eta	(cosi, distance, phi0, psi, RA, dec, detector_list, mchirp, eta, Fderivs)

	rho2 = expected_coherent_snr(cosi,distance,phi0,psi,RA,dec,detector_list)

	return g/rho2

def average_metric(RA, dec, detectors, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized amplitude-parameter-averaged sky-mass-time
	metric is symmetric and is defined with the following signature:
	                  0    1  2    3    4
	                  RA  dec t mchirp eta
	     0   RA    /  A    F  J    M    P  \
	     1  dec   |   -    B  G    K    N  |
	 g = 2   t    |   -    -  C    H    L  |
	     3 mchirp |   -    -  -    D    I  |
	     4  eta    \  -    -  -    -    E /
	"""
	g = scipy.zeros((5,5))
	g[0,0] =		gbar_RA_RA		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,1] = g[1,0] =	gbar_RA_dec		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,2] = g[2,0] =	gbar_RA_t		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,3] = g[3,0] =	gbar_RA_mchirp		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,4] = g[4,0] =	gbar_RA_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,1] = 		gbar_dec_dec		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,2] = g[2,1] =	gbar_dec_t		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,3] = g[3,1] =	gbar_dec_mchirp		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,4] = g[4,1] =	gbar_dec_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[2,2] =		gbar_t_t		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[2,3] = g[3,2] =	gbar_t_mchirp		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[2,4] = g[4,2] =	gbar_t_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[3,3] =		gbar_mchirp_mchirp	(RA, dec, detectors, mchirp, eta, Fderivs)
	g[3,4] = g[4,3] =	gbar_mchirp_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[4,4] =		gbar_eta_eta		(RA, dec, detectors, mchirp, eta, Fderivs)

	return g

def other_average_metric(RA, dec, detectors, mchirp=2.*.25**.6*M_sun, eta=0.25, Fderivs=False):
	"""
	The amplitude maximized SNR^2-averaged sky-mass-time metric is
	symmetric and is defined with the following signature:
	                  0    1  2    3    4
	                  RA  dec t mchirp eta
	     0   RA    /  A    F  J    M    P  \
	     1  dec   |   -    B  G    K    N  |
	 g = 2   t    |   -    -  C    H    L  |
	     3 mchirp |   -    -  -    D    I  |
	     4  eta    \  -    -  -    -    E /
	"""
	g = scipy.zeros((5,5))
	g[0,0] =		ogbar_RA_RA		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,1] = g[1,0] =	ogbar_RA_dec		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,2] = g[2,0] =	ogbar_RA_t		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,3] = g[3,0] =	ogbar_RA_mchirp		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[0,4] = g[4,0] =	ogbar_RA_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,1] = 		ogbar_dec_dec		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,2] = g[2,1] =	ogbar_dec_t		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,3] = g[3,1] =	ogbar_dec_mchirp	(RA, dec, detectors, mchirp, eta, Fderivs)
	g[1,4] = g[4,1] =	ogbar_dec_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[2,2] =		ogbar_t_t		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[2,3] = g[3,2] =	ogbar_t_mchirp		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[2,4] = g[4,2] =	ogbar_t_eta		(RA, dec, detectors, mchirp, eta, Fderivs)
	g[3,3] =		ogbar_mchirp_mchirp	(RA, dec, detectors, mchirp, eta, Fderivs)
	g[3,4] = g[4,3] =	ogbar_mchirp_eta	(RA, dec, detectors, mchirp, eta, Fderivs)
	g[4,4] =		ogbar_eta_eta		(RA, dec, detectors, mchirp, eta, Fderivs)

	return g

def extract_diagonal(g):
	"""
	This transforms the metric into the following form:
	                  0    1  2    3    4
	                  RA  dec t mchirp eta
	     0   RA    /  1    f  j    m    p  \
	     1  dec   |   -    1  g    k    n  |
	 g = 2   t    |   -    -  1    h    l  |
	     3 mchirp |   -    -  -    1    i  |
	     4  eta    \  -    -  -    -    1 /

	where g'_ij := g_ij / (g_ii * g_jj)**.5 and returns this along with
	the square root of the diagonal. This operation is applied in order to
	scale the matrix such that the matrix norm is more nearly equal to the
	eigenvalues. 
	"""
	rootdiag = []
	for idx in range(len(g)):
		rootdiag.append(g[idx,idx]**.5)
	for idx in range(len(g)):
		for jdx in range(len(g)):
			if rootdiag[idx]*rootdiag[jdx]:
				g[idx,jdx] /= rootdiag[idx]*rootdiag[jdx]
	return g,rootdiag

def restore_diagonal(g, rootdiag):
	"""
	Takes the modified metric and the square root of the diagonal and
	undoes the scaling operation of extract_diagonal(g).
	"""
	for idx in range(len(g)):
		for jdx in range(len(g)):
			g[idx,jdx] *= rootdiag[idx]*rootdiag[jdx]
	return g

def project_out_dimension(g, rootdiag, dim):
	"""
	Projects out the dim'th dimension from g and rootdiag.
	"""
	old_shape = scipy.shape(g)
	new_shape = []
	for d in old_shape:
		new_shape.append(d-1)
	g_projected = scipy.zeros(new_shape)

	for idx in range(len(g)):
		out_idx = idx
		if idx == dim:
			continue
		elif idx > dim:
			out_idx -= 1
		for jdx in range(len(g)):
			out_jdx = jdx
			if jdx == dim:
				continue
			elif jdx > dim:
				out_jdx -= 1
			g_projected[out_idx,out_jdx] = g[idx,jdx] - g[idx,dim]*g[dim,jdx]/g[dim,dim]

	rootdiag_projected = copy.deepcopy(rootdiag)
	rootdiag_projected.pop(dim)

	return g_projected,rootdiag_projected

def metric_distance_on_sphere(test_point, point, metric_at_point):
	"""
	Computes the metric distance between two points using the metric
	assuming the points can be wrapped as though on a sphere.
	"""
	dRA = test_point[0] - point[0]
	ddec = test_point[1] - point[1]
	if dRA > pi:
		dRA = -2*pi + dRA
	if dRA < -pi:
		dRA = 2*pi + dRA
	dist = metric_at_point[0,0]*dRA**2 \
		+  metric_at_point[1,1]*ddec**2 \
		+ 2.*metric_at_point[0,1]*dRA*ddec

	return dist

def metric_distance(test_point, point, metric_at_point):
	"""
	Computes the metric distance between two points using the metric.	
	"""
	dx = test_point[0] - point[0]
	dy = test_point[1] - point[1]
	dist = metric_at_point[0,0]*dx**2 \
		+  metric_at_point[1,1]*dy**2 \
		+ 2.*metric_at_point[0,1]*dx*dy

	return dist

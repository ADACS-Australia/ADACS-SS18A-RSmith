/*
*  Copyright (C) 2007 Chris Messenger, Jolien Creighton, Reinhard Prix
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
*/

/**
\author  Messenger, C.J., Berukoff, S.J., Papa, M.A.
\file
\ingroup pulsarTODO

\brief Computes the phase model coefficients necessary for a successful demodulation
for the case of a continuous wave source in a binary system.

*/

#include <math.h>
#include <lal/LALConstants.h>
#include <lal/ComputeSkyBinary.h>
#include <lal/FindRoot.h>
#include <lal/LALBarycenter.h>

static void Ft(LALStatus *status, REAL8 *tr, REAL8 t, void *tr0);

static REAL8 a;      /* semi major axis */
static REAL8 Period;    /* Period */
static REAL8 ecc;    /* eccentricity */
static REAL8 parg;   /* argument of periapse */
static LIGOTimeGPS Tperi;  /* Time of periapse passage as measured in SSB frame */
static REAL8 p,q;    /* coeficients of phase model */
static REAL8 E;      /* eccentric anomoly */


/** Given a set of input parameters defining a source location in the sky and
the binary system in which the source resides, this routine returns the phase
model coefficients \f$A_{s\alpha}\f$ and \f$B_{s\alpha}\f$ which are needed to
correctly account for the phase variance of a signal over time. The
\c CSBParams parameter structure contains relevant information
for this routine to properly run.  In particular, it contains an array of
timestamps in \c LIGOTimeGPS format, which are the GPS times of the first
data from each SFT.  The \c input is an \c INT4 variable
\c iSkyCoh, which is the index of the sky location under consideration.  For
each sky location and set of orbital parameters, this code needs to be run
once; the necessary phase model
coefficients are calculated, and can then be applied to the relevant spindown
parameter sets one is using in their search.

\heading{Algorithm}
The routine uses a simplistic nested for-loop structure.  The outer loop is
over the number of SFTs in the observation timescale; this accounts for the
temporal variability of the phase model coefficients.  The inner loop is over
the number of spindown parameters in one set.  Inside the inner loop, the
values are calculated using the analytical formulae given in the
\ref ComputeSkyBinary.h documentation.
*/
void LALComputeSkyBinary (LALStatus	*status,
                          REAL8 	*skyConst,
                          INT8 		iSkyCoh,
                          CSBParams 	*params)
{

	INT4	m, n, nP;
	REAL8	dTbary;
	REAL8   dTbarySP;
	REAL8   dTperi;
	REAL8   dTcoord;
	REAL8   Tdotbin;
	REAL8   basedTperi;
	DFindRootIn input;
	REAL8 tr0;
	REAL8 acc;

 INITSTATUS(status);
 ATTATCHSTATUSPTR(status);

/* Check for non-negativity of sky positions in SkyCoh[] */
 ASSERT(iSkyCoh>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);

/* Check to make sure sky positions are loaded */
 ASSERT(params->skyPos!=NULL, status, COMPUTESKYBINARYH_ENULL, COMPUTESKYBINARYH_MSGENULL);
 ASSERT(params->skyPos!=NULL, status, COMPUTESKYBINARYH_ENULL, COMPUTESKYBINARYH_MSGENULL);

/* Check to make sure parameters are loaded and reasonable */
 ASSERT(params->spinDwnOrder>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT(params->mObsSFT>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT(params->tSFT>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);

/* Check to make sure orbital parameters are loaded and reasonable */
 ASSERT(params->SemiMajorAxis>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT(params->OrbitalPeriod>0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT(params->OrbitalEccentricity>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT((params->ArgPeriapse>=0)&&(params->ArgPeriapse<=LAL_TWOPI), status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT(params->TperiapseSSB.gpsSeconds>=0, status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);
 ASSERT((params->TperiapseSSB.gpsNanoSeconds>=0)&&(params->TperiapseSSB.gpsNanoSeconds<1e9), status, COMPUTESKYBINARYH_ENEGA, COMPUTESKYBINARYH_MSGENEGA);


 /* Here we redefine the orbital variables for ease of use */
 a=params->SemiMajorAxis;  /* This is the projected semi-major axis of the orbit normalised by the speed of light */
 Period=params->OrbitalPeriod;  /* This is the period of the orbit in seconds */
 ecc=params->OrbitalEccentricity;  /* This is the eccentricity of the orbit */
 parg=params->ArgPeriapse;  /* This is the argument of periapse defining the angular location of the source at periapsis */
                            /* measured relative to the ascending node */
 Tperi.gpsSeconds=params->TperiapseSSB.gpsSeconds;  /* This is the GPS time as measured in the SSB of the observed */
 Tperi.gpsNanoSeconds=params->TperiapseSSB.gpsNanoSeconds;  /* periapse passage of the source */

 /* Here we check that the GPS timestamps are greater than zero */
 for(n=0;n<params->mObsSFT;n++)
   {
     ASSERT(params->tGPS[n].gpsSeconds>=0, status, COMPUTESKYBINARYH_ENEGA, 	COMPUTESKYBINARYH_MSGENEGA);
   }

 /* Check to make sure pointer to output is not NULL */
 ASSERT(skyConst!=NULL, status, COMPUTESKYBINARYH_ENNUL, COMPUTESKYBINARYH_MSGENNUL);

 /* prepare params input sky position structure */
 params->baryinput->alpha=params->skyPos[iSkyCoh];
 params->baryinput->delta=params->skyPos[iSkyCoh+1];

 /* calculate phase model coefficients p and q which are defined in the ComputeSkyBinary header LAL documentation  */
 p=((LAL_TWOPI/(Period*1.0))*a*sqrt(1-(ecc*ecc))*cos(parg))-ecc;
 q=(LAL_TWOPI/(Period*1.0))*a*sin(parg);

 /* Calculate the required accuracy for the root finding procedure in the main loop */
 acc=LAL_TWOPI*(REAL8)ACC/Period;   /* ACC is defined in ComputeSkyBinary.h and represents the required */
                                    /* timing precision in seconds (roughly)*/

 /* begin loop over SFT's */
 for (n=0; n<params->mObsSFT; n++)
   {
     /* Calculate the detector time at the mid point of current SFT ( T(i)+(tsft/2) ) using LAL functions */
     params->baryinput->tgps = params->tGPS[n];
     XLALGPSAdd(&(params->baryinput->tgps), params->tSFT / 2.0);

     /* Convert this mid point detector time into barycentric time (SSB) */
     LALBarycenterEarth(status->statusPtr, params->earth, &(params->baryinput->tgps), params->edat);
     LALBarycenter(status->statusPtr, params->emit, params->baryinput, params->earth);

     /* Calculate the time difference since the observed periapse passage in barycentric time (SSB). */
     /* This time difference, when converted to REAL8, should lose no precision unless we are dealing */
     /* with periods >~ 1 Year */
     dTbary = XLALGPSDiff(&(params->emit->te),&Tperi);

     /* Calculate the time since the last periapse passage ( < Single Period (SP) ) */
     dTbarySP=Period*((dTbary/(1.0*Period))-(REAL8)floor(dTbary/(1.0*Period)));

     /* Calculate number of full orbits completed since the input observed periapse passage */
     nP=(INT4)floor(dTbary/(1.0*Period));

     /* begin root finding procedure */
     tr0 = dTbarySP;        /* we wish to find the value of the eccentric amomoly E coresponding to the time */
                            /* since the last periapse passage */
     input.function = Ft;   /* This is the name of the function we must solve to find E */
     input.xmin = 0.0;      /* We know that E will be found between 0 and 2PI */
     input.xmax = LAL_TWOPI;
     input.xacc = acc;      /* The accuracy of the root finding procedure */

     /* expand domain until a root is bracketed */
     LALDBracketRoot(status->statusPtr,&input,&tr0);

     /* bisect domain to find eccentric anomoly E corresponding to the current midpoint timestamp */
     LALDBisectionFindRoot(status->statusPtr,&E,&input,&tr0);

     /* Now we calculate the time interval since the input periapse passage as measured at the source */
     dTperi=(Period/LAL_TWOPI)*(E-(ecc*sin(E)))+((REAL8)nP*Period);

     /* The following quantity is the derivative of the time coordinate measured at the source with */
     /* respect to the time coordinate measured in the SSB */
     dTcoord=(1.0-(ecc*cos(E)))/(1.0+(p*cos(E))-(q*sin(E)));

     /* The following quantity is the derivitive of the time coordinate measured in the SSB with */
     /* respect to the time coordinate measured at the chosen detector.  It was calculated via the */
     /* last call to LALBarycenter. */
     Tdotbin = params->emit->tDot*dTcoord;

     /* Loop over all spin down orders plus 0th order (f0) */
     /* In this loop we calculate the SkyConstants defined in the documentation as A_{s,alpha} and B_{s,alpha} */
     for (m=0; m<params->spinDwnOrder+1; m++)
       {
	 /* raise the quantity dTperi to the power m */
	 basedTperi = pow(dTperi, (REAL8)m);
	 /* Calculate A coefficients */
	 skyConst[2*n*(params->spinDwnOrder+1)+2*(INT4)m]=1.0/((REAL8)m+1.0)*basedTperi*dTperi-0.5*params->tSFT*basedTperi*Tdotbin;
	 /* Calculate B coefficients */
	 skyConst[2*n*(params->spinDwnOrder+1)+2*(INT4)m+1]= params->tSFT*basedTperi*Tdotbin;
       }
   }

 /* Normal Exit */
 DETATCHSTATUSPTR(status);
 RETURN(status);
}

/**************************************************************************************************/

static void Ft(LALStatus *status, REAL8 *tr, REAL8 lE, void *tr0)
{
  INITSTATUS(status);
  ASSERT(tr0,status, 1, "Null pointer");

  /* this is the function relating the observed time since periapse in the SSB to the true eccentric anomoly E */

  *tr = *(REAL8 *)tr0*(-1.0) + (Period/LAL_TWOPI)*(lE+(p*sin(lE))+q*(cos(lE)-1.0));
  RETURN(status);
}


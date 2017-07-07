/*
*  Copyright (C) 2007 Alexander Dietz, Drew Keppel, Duncan Brown, Eirini Messaritaki, Jolien Creighton, Patrick Brady, Stephen Fairhurst, Craig Robinson , Thomas Cokelaer
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

/*-----------------------------------------------------------------------
 *
 * File Name: SnglInspiralUtils.c
 *
 * Author: Brady, P. R., Brown, D. A., Fairhurst, S. and Messaritaki, E.
 *
 *-----------------------------------------------------------------------
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataInspiralUtils.h>
#include <lal/Date.h>
#include <lal/SkyCoordinates.h>
#include <lal/GeneratePPNInspiral.h>
#include <lal/DetectorSite.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>

/**
 * \author Brown, D. A., Fairhurst, S. and Messaritaki, E.
 * \file
 *
 * \brief Provides a set of utilities for manipulating \c snglInspiralTables.
 *
 * ### Description ###
 *
 * The function <tt>LALFreeSnglInspiral()</tt> and XLALFreeSnglInspiral()
 * free the memory associated to a single inspiral table.  The single inspiral
 * table may point to a linked list of EventIDColumns.  Thus, it is necessary to
 * free all event ids associated with the single inspiral.
 *
 * The function <tt>LALSortSnglInspiral()</tt> and <tt>XLALSortSnglInspiral()</tt>
 * sorts a list of single inspiral tables.  The function simply calls qsort with
 * the appropriate comparison function, \c comparfunc.  It then ensures that
 * the head of the sorted list is returned.  There then follow several comparison
 * functions for single inspiral tables.  <tt>LALCompareSnglInspiralByMass()</tt>
 * first compares the \c mass1 entry of the two inspiral tables, returning 1
 * if the first mass is larger and -1 if the second is larger.  In the case that
 * the \c mass1 fields are equal, a similar comparsion is performed on
 * \c mass2.  If these also agree, 0 is returned.
 * <tt>LALCompareSnglInspiralByPsi()</tt> compares the \c Psi0 and
 * \c Psi3 fields in two single inspiral tables.  The function is analogous
 * to the mass comparison described above.  \c LALCompareSnglInspiralByTime()
 * compares the end times of two single inspiral tables, returnng 1 if the first
 * time is larger, 0 if equal and -1 if the second time is larger.
 *
 * <tt>LALCompareSnglInspiral()</tt> tests whether two single inspiral tables
 * pass a coincidence test.  The coincidence parameters are given by
 * \c params which is a \c ::SnglInspiralAccuracy structure.  It tests
 * first that the \c ifo fields are different.  If they are, it then tests
 * for time and mass coincidence, where mass coincidence may be any one of
 * \c psi0_and_psi3, \c m1_and_m2, \c mchirp_and_eta.
 * Finally, if the test is on \c m1_and_m2, consistency of effective
 * distances is also checked.  If the two single inspiral tables pass
 * coincidences the <tt>params.match</tt> is set to 1, otherwise it is set to
 * zero.
 *
 * <tt>LALClusterSnglInspiralTable()</tt> clusters single inspiral triggers
 * within a time window \c dtimeNS.  The triggers are compared either by
 * \c snr, \c snr_and_chisq or \c snrsq_over_chisq.  The
 * "loudest" trigger, as determined by the selected algorithm, within each time
 * window is returned.
 *
 * <tt>LALTimeCutSingleInspiral()</tt> and
 * <tt>XLALTimeCutSingleInspiral()</tt>takes in a linked list of single inspiral
 * tables and returns only those which occur after the given \c startTime
 * and before the \c endTime.
 *
 * <tt>LALSNRCutSingleInspiral()</tt> and <tt>XLALSNRCutSingleInspiral()</tt>
 * take in a linked list of single inspiral tables and returns only those
 * triggers which have snr values above a specific snrCut.
 *
 * <tt>XLALRsqCutSingleInspiral()</tt> performs the R-squared veto on a linked
 * list of single inspiral tables.  Triggers whose snr is less than
 * \c rsqSnrMax and whose \c rsqveto_duration is greater than
 * \c rsqVetoThresh or <tt>(optional)</tt> whose snr is greater than
 * \c rsqSnrMax and whose \c rsqveto_duration is greater than
 * \f$\mathtt{rsqAboveSnrCoeff} \times \mathtt{snr}^{\mathtt{rsqAboveSnrPow}}\f$
 *
 * <tt>XLALVetoSingleInspiral()</tt> takes in a linked list of single inspiral
 * tables and a list of segments and returns only those triggers which do not lie
 * in within the \c vetoSegs.
 *
 * <tt>LALBCVCVetoSingleInspiral()</tt> takes in a linked list of single inspiral
 * tables and returns only those triggers which have alphaF/SNR values below a
 * specific threshold and alphaF value between alphaF-hi and alphaF-lo values.  It
 * is relevant for the BCVC or BCVU search only.
 *
 * <tt>LALalphaFCutSingleInspiral()</tt> takes in a linked list of single
 * inspiral tables and returns only those triggers which have alphaF values below
 * a specific alphaFcut. It is relevant for the BCV search only.
 *
 * <tt>LALIfoCutSingleInspiral()</tt> scans through a linked list of single
 * inspiral tables and returns those which are from the requested \c ifo.
 * On input, \c eventHead is a pointer to the head of a linked list of
 * single inspiral tables.  On output, this list contains only single inspirals
 * from the requested \c ifo.  <tt>XLALIfoCutSingleInspiral()</tt> works
 * similarly, although slightly differently.  This function returns the list of
 * events from the specified \c ifo, while on completion,
 * \c eventHead contains the list of events from \e other ifos.
 *
 * <tt>LALIfoCountSingleInspiral()</tt> scans through a linked list of single
 * inspiral tables and counts the number which are from the requested IFO.
 * This count is returned as \c numTrigs.
 *
 * <tt>XLALTimeSlideSingleInspiral()</tt> performs a time slide on the triggers
 * contained in the \c triggerList.  The time slide for each instrument is
 * specified by <tt>slideTimes[LAL_NUM_IFO]</tt>.  If \c startTime and
 * \c endTime are specified, then the time slide is performed on a ring.  If
 * the slide takes any trigger outside of the window
 * <tt>[startTime,endTime]</tt>, then the trigger is wrapped to be in
 * this time window.
 *
 * <tt>LALPlayTestSingleInspiral()</tt> and <tt>XLALPlayTestSingleInspiral()</tt>
 * test whether single inspiral events occured in playground or non-playground
 * times.  It then returns the requested subset of events which occurred in the
 * times specified by \c dataType which must be one of
 * \c playground_only, \c exclude_play or \c all_data.
 *
 * <tt>LALCreateTrigBank()</tt> takes in a list of single inspiral tables and
 * returns a template bank.  The function tests whether a given template produced
 * multiple triggers.  If it did, only one copy of the template is retained.
 * Triggers are tested for coincidence in \c m1_and_m2 or
 * \c psi0_and_psi3.
 *
 * <tt>LALIncaCoincidenceTest()</tt> performs a coincidence test between triggers
 * from two interferometers.  It tests pairs of events for both time and mass
 * coincidence and returns two equal length lists of coincident events.  Note
 * that if an event in one detector is coincident with several events in the
 * other detector, the output lists will contain several copies of this event.
 *
 * <tt>LALTamaCoincidenceTest()</tt> also performs a coincidence test between
 * triggers from two interferometers, but with a slightly different coincidence
 * test.  First, it locates all triggers in the second instrument which are
 * coincident with triggers in the first instrument.  Then, it clusters these
 * triggers using the appropriate \c clusterchioce.  Finally, it tests for
 * mass coincidence between the first trigger and the clustered trigger from the
 * second instrument.
 *
 * <tt>XLALAddSnglInspiralCData</tt> determines if the complex time-series of the
 * matched-filter output, or "CData", for a given trigger has been queued for
 * writing into a frame-file.
 *
 * ### Algorithm ###
 *
 * None.
 *
 * ### Uses ###
 *
 * LALCalloc(), LALFree(), LALINT8NanoSecIsPlayground().
 *
 */

/*
 * A few quickies for convenience.
 */

static INT8 end_time(const SnglInspiralTable *x)
{
	return(XLALGPSToINT8NS(&x->end));
}

static INT4 end_time_sec(const SnglInspiralTable *x)
{
	return(x->end.gpsSeconds);
}

static INT4 end_time_nsec(const SnglInspiralTable *x)
{
	return(x->end.gpsNanoSeconds);
}



void
LALFreeSnglInspiral (
    LALStatus          *status,
    SnglInspiralTable **eventHead
    )

{
  INITSTATUS(status);
  XLALFreeSnglInspiral( eventHead );
  RETURN( status );
}


int
XLALFreeSnglInspiral (
    SnglInspiralTable **eventHead
    )

{
  LALFree( *eventHead );

  return (0);
}


void
LALSortSnglInspiral (
    LALStatus          *status,
    SnglInspiralTable **eventHead,
    int(*comparfunc)    (const void *, const void *)
    )

{
  INITSTATUS(status);

  *eventHead = XLALSortSnglInspiral ( *eventHead, comparfunc );

  RETURN( status );
}


SnglInspiralTable *
XLALSortSnglInspiral (
    SnglInspiralTable *eventHead,
    int(*comparfunc)   (const void *, const void *)
    )

{
  INT4                  i;
  INT4                  numEvents = 0;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable   **eventHandle = NULL;

  /* count the number of events in the linked list */
  for ( thisEvent = eventHead; thisEvent; thisEvent = thisEvent->next )
  {
    ++numEvents;
  }
  if ( ! numEvents )
  {
    XLALPrintInfo(
      "XLALSortSnglInspiral: Empty SnglInspiralTable passed as input\n" );
    return( eventHead );
  }

  /* allocate memory for an array of pts to sort and populate array */
  eventHandle = (SnglInspiralTable **)
    LALCalloc( numEvents, sizeof(SnglInspiralTable *) );
  for ( i = 0, thisEvent = eventHead; i < numEvents;
      ++i, thisEvent = thisEvent->next )
  {
    eventHandle[i] = thisEvent;
  }

  /* qsort the array using the specified function */
  qsort( eventHandle, numEvents, sizeof(eventHandle[0]), comparfunc );

  /* re-link the linked list in the right order */
  thisEvent = eventHead = eventHandle[0];
  for ( i = 1; i < numEvents; ++i )
  {
    thisEvent = thisEvent->next = eventHandle[i];
  }
  thisEvent->next = NULL;

  /* free the internal memory */
  LALFree( eventHandle );

  return( eventHead );
}




int
LALCompareSnglInspiralByMass (
    const void *a,
    const void *b
    )

{
  const SnglInspiralTable *aPtr = *((const SnglInspiralTable * const *)a);
  const SnglInspiralTable *bPtr = *((const SnglInspiralTable * const *)b);

  if ( aPtr->mass1 > bPtr->mass1 )
  {
    return 1;
  }
  else if ( aPtr->mass1 < bPtr->mass1 )
  {
    return -1;
  }
  else if ( aPtr->mass2 > bPtr->mass2 )
  {
    return 1;
  }
  else if ( aPtr->mass2 < bPtr->mass2 )
  {
    return -1;
  }
  else
  {
    return 0;
  }
}



int
LALCompareSnglInspiralByPsi (
    const void *a,
    const void *b
    )

{
  const SnglInspiralTable *aPtr = *((const SnglInspiralTable * const *)a);
  const SnglInspiralTable *bPtr = *((const SnglInspiralTable * const *)b);

  if ( aPtr->psi0 > bPtr->psi0 )
  {
    return 1;
  }
  else if ( aPtr->psi0 < bPtr->psi0 )
  {
    return -1;
  }
  else if ( aPtr->psi3 > bPtr->psi3 )
  {
    return 1;
  }
  else if ( aPtr->psi3 < bPtr->psi3 )
  {
    return -1;
  }
  else
  {
    return 0;
  }
}




int
LALCompareSnglInspiralByTime (
    const void *a,
    const void *b
    )

{
  LALStatus     status;
  const SnglInspiralTable *aPtr = *((const SnglInspiralTable * const *)a);
  const SnglInspiralTable *bPtr = *((const SnglInspiralTable * const *)b);
  INT8 ta, tb;

  memset( &status, 0, sizeof(LALStatus) );
  ta = XLALGPSToINT8NS( &(aPtr->end) );
  tb = XLALGPSToINT8NS( &(bPtr->end) );

  if ( ta > tb )
  {
    return 1;
  }
  else if ( ta < tb )
  {
    return -1;
  }
  else
  {
    return 0;
  }
}



void
LALCompareSnglInspiral (
    LALStatus                *status,
    SnglInspiralTable        *aPtr,
    SnglInspiralTable        *bPtr,
    SnglInspiralAccuracy     *params
    )

{
  INT8 ta, tb;
  REAL4 dm1, dm2;
  REAL4 dmchirp, deta;
  REAL4 dpsi0, dpsi3;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  params->match = 1;

  /* check that triggers come from different IFOs */
  if( strcmp(aPtr->ifo, bPtr->ifo) )
  {
    LALInfo( status, "Triggers from different IFOs");
    params->match = 1;
  }
  else
  {
    LALInfo( status, "Triggers from same IFO");
    params->match = 0;
  }

  ta = XLALGPSToINT8NS( &(aPtr->end) );
  tb = XLALGPSToINT8NS( &(bPtr->end) );

  /* compare on trigger time coincidence */
  if ( labs( ta - tb ) < params->dt && params->match)
  {
    LALInfo( status, "Triggers pass time coincidence test");
    params->match = 1;
  }
  else if ( labs( ta - tb ) < params->dt && params->match)
  {
    LALInfo( status, "Triggers fail time coincidence test" );
    params->match = 0;
  }

  /* perform the mass parameter test */
  if( params->match )
  {
    /* compare psi0 and psi3 parameters */
    if ( params->test == psi0_and_psi3 )
    {
      dpsi0 = fabs( aPtr->psi0 - bPtr->psi0 );
      dpsi3 = fabs( aPtr->psi3 - bPtr->psi3 );

      if ( dpsi0 <= params->dpsi0 && dpsi3 <= params->dpsi3 )
      {
        LALInfo( status, "Triggers are coincident in psi0 and psi3" );
        params->match = 1;
      }
      else
      {
        LALInfo( status, "Triggers are not coincident in psi0 and psi3" );
        params->match = 0;
      }
    }
    else if ( params->test == m1_and_m2 )
    {
      dm1 = fabs( aPtr->mass1 - bPtr->mass1 );
      dm2 = fabs( aPtr->mass2 - bPtr->mass2 );

      /* compare mass1 and mass2 parameters */
      if ( dm1 <= params->dm && dm2 <= params->dm )
      {
        LALInfo( status, "Triggers are coincident in mass1 and mass2" );
        params->match = 1;
      }
      else
      {
        LALInfo( status, "Triggers are not coincident in mass1 and mass2" );
        params->match = 0;
      }
    }
    else if ( params->test == mchirp_and_eta )
    {
      dmchirp = fabs( aPtr->mchirp - bPtr->mchirp );
      deta = fabs( aPtr->eta - bPtr->eta );

      /* compare mchirp and eta parameters */
      if ( dmchirp <= params->dmchirp && deta <= params->deta )
      {
        LALInfo( status, "Triggers are coincident in mchirp and eta" );
        params->match = 1;
      }
      else
      {
        LALInfo( status, "Triggers fail mchirp, eta coincidence test" );
        params->match = 0;
      }
    }
    else
    {
      LALInfo( status, "error: unknown test\n" );
      params->match = 0;
    }
  }

  /* check for distance consistency */
  if ( params->match && params->test == m1_and_m2 )
  {
    if ( fabs( (aPtr->eff_distance - bPtr->eff_distance) / aPtr->eff_distance)
        < params->epsilon / bPtr->snr + params->kappa )
    {
      LALInfo( status, "Triggers are coincident in eff_distance" );
      params->match = 1;
    }
    else
    {
      LALInfo( status, "Triggers fail eff_distance coincidence test" );
      params->match = 0;
    }
  }

  DETATCHSTATUSPTR (status);
  RETURN (status);
}


void
LALCompareInspirals (
    LALStatus                *status,
    SnglInspiralTable        *aPtr,
    SnglInspiralTable        *bPtr,
    InspiralAccuracyList     *params
    )

{

  INITSTATUS(status);

  XLALCompareInspirals( aPtr, bPtr, params );

  RETURN( status );
}




int
XLALCompareInspirals (
    SnglInspiralTable        *aPtr,
    SnglInspiralTable        *bPtr,
    InspiralAccuracyList     *params
    )

{
  INT8    ta,  tb;
  REAL4   dmass1, dmass2;
  REAL4   dmchirp, deta;
  REAL4   dpsi0, dpsi3;
  REAL4   dtau0, dtau3;
  InterferometerNumber ifoaNum,  ifobNum;
  SnglInspiralAccuracy aAcc, bAcc;

  params->match = 1;

  /* check that triggers come from different IFOs */
  if( strcmp(aPtr->ifo, bPtr->ifo) )
  {
    XLALPrintInfo( "Triggers from different IFOs\n");
    params->match = 1;
  }
  else
  {
    XLALPrintInfo( "Triggers from same IFO\n");
    params->match = 0;
    return params->match;
  }

  ifoaNum = (InterferometerNumber) XLALIFONumber( aPtr->ifo );
  ifobNum = (InterferometerNumber) XLALIFONumber( bPtr->ifo );

  ta = XLALGPSToINT8NS( &(aPtr->end) );
  tb = XLALGPSToINT8NS( &(bPtr->end) );

  /* compare on trigger time coincidence */
  aAcc = params->ifoAccuracy[ifoaNum];
  bAcc = params->ifoAccuracy[ifobNum];


  if ( params->exttrig &&
       labs( ta - tb + params->lightTravelTime[ifoaNum][ifobNum]) < (aAcc.dt + bAcc.dt) )
  {
    XLALPrintInfo( "Triggers pass time coincidence test\n" );
    params->match = 1;
  }
  else if (  !params->exttrig &&
      labs( ta - tb ) < (aAcc.dt + bAcc.dt)
      + params->lightTravelTime[ifoaNum][ifobNum])
  {
    XLALPrintInfo( "Triggers pass time coincidence test\n" );
    params->match = 1;
  }
  else
  {
    XLALPrintInfo( "Triggers fail time coincidence test\n" );
    params->match = 0;
    return params->match;
  }

  switch ( params->test )
  {
    case psi0_and_psi3:
      dpsi0 = fabs( aPtr->psi0 - bPtr->psi0 );
      dpsi3 = fabs( aPtr->psi3 - bPtr->psi3 );

      /* compare psi0 and psi3 parameters */
      if ( ( dpsi0 <= (aAcc.dpsi0 + bAcc.dpsi0) )
          && ( dpsi3 <= (aAcc.dpsi3 + bAcc.dpsi3) ))
      {
        XLALPrintInfo( "Triggers are coincident in psi0 and psi3\n" );
        params->match = 1;
      }
      else
      {
        XLALPrintInfo( "Triggers are not coincident in psi0 and psi3\n" );
        params->match = 0;
      }
      break;

    case m1_and_m2:
      dmass1 = fabs( aPtr->mass1 - bPtr->mass1 );
      dmass2 = fabs( aPtr->mass2 - bPtr->mass2 );

      /* compare mass1 and mass2 parameters */
      if ( (dmass1 <= (aAcc.dm + bAcc.dm) )
        && (dmass2 <= (aAcc.dm + bAcc.dm) ))
      {
        XLALPrintInfo( "Triggers are coincident in mass1 and mass2\n" );
        params->match = 1;
      }
      else
      {
        XLALPrintInfo( "Triggers are not coincident in mass1 and mass2\n" );
        params->match = 0;
      }
      break;

    case mchirp_and_eta:
      {
      REAL4 dmchirpTest;
      dmchirp = fabs( aPtr->mchirp - bPtr->mchirp );
      deta = fabs( aPtr->eta - bPtr->eta );

      /* compare mchirp and eta parameters */
      if (aAcc.highMass &&
      ((aPtr->mass1 + aPtr->mass2 > aAcc.highMass) ||
      (bPtr->mass1 + bPtr->mass2 > bAcc.highMass)))
        dmchirpTest = aAcc.dmchirpHi + bAcc.dmchirpHi;
      else
        dmchirpTest = aAcc.dmchirp + bAcc.dmchirp;
      if ( (dmchirp <= dmchirpTest)
            && (deta <= (aAcc.deta + bAcc.deta)) )
      {
        XLALPrintInfo( "Triggers are coincident in mchirp and eta\n" );
        params->match = 1;
      }
      else
      {
        XLALPrintInfo( "Triggers fail mchirp, eta coincidence test\n" );
        params->match = 0;
      }
      }
      break;

    case tau0_and_tau3:
      dtau0 = fabs( aPtr->tau0 - bPtr->tau0 );
      dtau3 = fabs( aPtr->tau3 - bPtr->tau3 );

      /* compare tau0 and tau3 parameters */
      if ( (dtau0 <= (aAcc.dtau0 + bAcc.dtau0) )
        && (dtau3 <= (aAcc.dtau3 + bAcc.dtau3) ))
      {
        XLALPrintInfo( "Triggers are coincident in tau0 and tau3\n" );
        params->match = 1;
      }
      else
      {
        XLALPrintInfo( "Triggers are not coincident in tau0 and tau3\n" );
        params->match = 0;
      }
      break;

    default:
      XLALPrintInfo( "error: unknown test\n" );
      params->match = 0;
      break;
  }

  return params->match;
}



void
LALClusterSnglInspiralTable (
    LALStatus                  *status,
    SnglInspiralTable         **inspiralEvent,
    INT8                        dtimeNS,
    SnglInspiralClusterChoice   clusterchoice
    )

{
  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  ASSERT( inspiralEvent, status,
      LIGOMETADATAUTILSH_ENULL, LIGOMETADATAUTILSH_MSGENULL );

  XLALClusterSnglInspiralTable ( inspiralEvent, dtimeNS, clusterchoice );

  /* normal exit */
  DETATCHSTATUSPTR (status);
  RETURN (status);
}

REAL4
XLALSnglInspiralStat(
    SnglInspiralTable         *snglInspiral,
    SnglInspiralClusterChoice  snglStat
    )
{
  REAL4 statValue = 0;

  if ( snglStat == snr )
  {
    statValue = snglInspiral->snr;
  }
  else if ( snglStat == snrsq_over_chisq )
  {
    statValue = snglInspiral->snr * snglInspiral->snr / snglInspiral->chisq;
  }
  else if ( snglStat == new_snr )
  {
    REAL4 rchisq = 0.;
    rchisq = snglInspiral->chisq / (2. * snglInspiral->chisq_dof - 2.);

    if ( rchisq <= 1. )
      { statValue = snglInspiral->snr; }
    else
    {
      /* newsnr formula with standard choice q=6 from arXiv:1111.7314      */
      /* \hat{\rho} = \rho / [(1 + \chi^2_r^3) / 2]^(1/6) for \chi^2_r > 1 */
      statValue = snglInspiral->snr / pow((1. + rchisq*rchisq*rchisq) / 2., (1. / 6.));
    }
  }
  else
  {
    statValue = 0;
  }
  return( statValue );
}



int
XLALClusterSnglInspiralTable (
    SnglInspiralTable         **inspiralList,
    INT8                        dtimeNS,
    SnglInspiralClusterChoice   clusterchoice
    )

{
  SnglInspiralTable     *thisEvent=NULL;
  SnglInspiralTable     *prevEvent=NULL;
  SnglInspiralTable     *nextEvent=NULL;
  int                    numSnglClust = 0;

  if ( !inspiralList )
  {
    XLAL_ERROR(XLAL_EIO);
  }

  if ( ! *inspiralList )
  {
    XLALPrintInfo(
      "XLALClusterSnglInspiralTable: Empty coincList passed as input\n" );
    return( 0 );
  }



  thisEvent = *inspiralList;
  nextEvent = (*inspiralList)->next;
  *inspiralList = NULL;

  while ( nextEvent )
  {
    INT8 thisTime = XLALGPSToINT8NS( &(thisEvent->end) );
    INT8 nextTime = XLALGPSToINT8NS( &(nextEvent->end) );;

    /* find events within the cluster window */
    if ( (nextTime - thisTime) < dtimeNS )
    {
      REAL4 thisStat = XLALSnglInspiralStat( thisEvent, clusterchoice );
      REAL4 nextStat = XLALSnglInspiralStat( nextEvent, clusterchoice );

      if ( nextStat > thisStat )
      {
        /* displace previous event in cluster */
        if( prevEvent )
        {
          prevEvent->next = nextEvent;
        }
        XLALFreeSnglInspiral( &thisEvent );
        thisEvent = nextEvent;
        nextEvent = thisEvent->next;
      }
      else
      {
        /* otherwise just dump next event from cluster */
        thisEvent->next = nextEvent->next;
        XLALFreeSnglInspiral ( &nextEvent );
        nextEvent = thisEvent->next;
      }
    }
    else
    {
      /* otherwise we keep this unique event trigger */
      if ( ! *inspiralList )
      {
        *inspiralList = thisEvent;
      }
      prevEvent = thisEvent;
      thisEvent = thisEvent->next;
      nextEvent = thisEvent->next;
      ++numSnglClust;
    }
  }

    /* store the last event */
  if ( ! (*inspiralList) )
  {
    *inspiralList = thisEvent;
  }
  ++numSnglClust;

  return(numSnglClust);
}


void
LALTimeCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    LIGOTimeGPS                *startTime,
    LIGOTimeGPS                *endTime
    )

{
  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  *eventHead = XLALTimeCutSingleInspiral( *eventHead, startTime, endTime );

  DETATCHSTATUSPTR (status);
  RETURN (status);

}


SnglInspiralTable *
XLALTimeCutSingleInspiral(
    SnglInspiralTable          *eventHead,
    LIGOTimeGPS                *startTime,
    LIGOTimeGPS                *endTime
    )

{
  SnglInspiralTable    *inspiralEventList = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;
  INT8                  startTimeNS = XLALGPSToINT8NS( startTime );
  INT8                  endTimeNS = XLALGPSToINT8NS( endTime );


  /* Remove all the triggers before and after the requested */
  /* gps start and end times */

  thisEvent = eventHead;

  while ( thisEvent )
  {
    SnglInspiralTable *tmpEvent = thisEvent;
    thisEvent = thisEvent->next;

    if ( end_time(tmpEvent) >= startTimeNS &&
        end_time(tmpEvent) < endTimeNS )
    {
      /* keep this template */
      if ( ! inspiralEventList  )
      {
        inspiralEventList = tmpEvent;
      }
      else
      {
        prevEvent->next = tmpEvent;
      }
      tmpEvent->next = NULL;
      prevEvent = tmpEvent;
    }
    else
    {
      /* discard this template */
      XLALFreeSnglInspiral ( &tmpEvent );
    }
  }
  eventHead = inspiralEventList;

  return (eventHead);
}



void LALSNRCutSingleInspiral (
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    REAL4                       snrCut
    )

{
  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  *eventHead = XLALSNRCutSingleInspiral( *eventHead, snrCut );

  DETATCHSTATUSPTR (status);
  RETURN (status);
}


SnglInspiralTable *
XLALSNRCutSingleInspiral (
    SnglInspiralTable          *eventHead,
    REAL4                       snrCut
    )

{
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;


  thisEvent = eventHead;
  eventHead = NULL;

  while ( thisEvent )
  {
    SnglInspiralTable *tmpEvent = thisEvent;
    thisEvent = thisEvent->next;

    if ( tmpEvent->snr >= snrCut )
    {
      /* keep this template */
      if ( ! eventHead  )
      {
        eventHead = tmpEvent;
      }
      else
      {
        prevEvent->next = tmpEvent;
      }
      tmpEvent->next = NULL;
      prevEvent = tmpEvent;
    }
    else
    {
      /* discard this template */
      XLALFreeSnglInspiral ( &tmpEvent );
    }
  }
  return( eventHead );
}




SnglInspiralTable *
XLALRsqCutSingleInspiral (
    SnglInspiralTable          *eventHead,
    REAL4                       rsqVetoTimeThresh,
    REAL4                       rsqMaxSnr,
    REAL4                       rsqAboveSnrCoeff,
    REAL4                       rsqAboveSnrPow
    )

{
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;
  int                   numTriggers = 0;


  thisEvent = eventHead;
  eventHead = NULL;

  while ( thisEvent )
  {
    SnglInspiralTable *tmpEvent = thisEvent;
    thisEvent = thisEvent->next;

    if ( (tmpEvent->snr <= rsqMaxSnr)
      && (tmpEvent->rsqveto_duration >= rsqVetoTimeThresh) )
    {
      /* discard this event */
      XLALFreeSnglInspiral ( &tmpEvent );
    }
    else if ( ( (tmpEvent->snr > rsqMaxSnr) && (rsqAboveSnrCoeff > 0)
      && (rsqAboveSnrPow > 0) ) && (tmpEvent->rsqveto_duration >=
      rsqAboveSnrCoeff * pow(tmpEvent->snr,rsqAboveSnrPow) ) )
    {
      /* discard this event */
      XLALFreeSnglInspiral ( &tmpEvent );
    }
    else
    {
      /* keep this event */
      if ( ! eventHead  )
      {
        eventHead = tmpEvent;
      }
      else
      {
        prevEvent->next = tmpEvent;
      }
      tmpEvent->next = NULL;
      prevEvent = tmpEvent;
      numTriggers++;
    }
  }
  return( eventHead );
}


SnglInspiralTable *
XLALVetoSingleInspiral (
    SnglInspiralTable *eventHead,
    LALSegList        *vetoSegs,
    const CHAR        *ifo
    )

{
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;

  thisEvent = eventHead;
  eventHead = NULL;

  while ( thisEvent )
  {
    /*-- Check the time of this event against the veto segment list --*/
    if ( XLALSegListSearch( vetoSegs, &(thisEvent->end) )
	&& (strcmp(thisEvent->ifo, ifo)==0) )
    {
      /*-- This event's end_time falls within one of the veto segments --*/
      /* discard the trigger and move to the next one */
      SnglInspiralTable    *tmpEvent = NULL;
      if ( prevEvent ) prevEvent->next = thisEvent->next;
      tmpEvent = thisEvent;
      thisEvent = thisEvent->next;
      XLALFreeSnglInspiral ( &tmpEvent );
    }
    else
    {
      /* This inspiral trigger does not fall within any veto segment */
      /* keep the trigger and increment the count of triggers */
      if ( ! eventHead ) eventHead = thisEvent;
      prevEvent = thisEvent;
      thisEvent = thisEvent->next;
    }
  }
  return( eventHead );
}

void
LALBCVCVetoSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    SnglInspiralBCVCalphafCut   alphafParams
    )

{
  SnglInspiralTable    *inspiralEventList = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;
  REAL4 alphaF;
  INT4 veto;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  thisEvent = *eventHead;

  while ( thisEvent )
  {
    SnglInspiralTable *tmpEvent = thisEvent;

    /* calculate the alphaf-value for this trigger */
    thisEvent = thisEvent->next;
    alphaF = tmpEvent->tau5 * pow( tmpEvent->f_final,(2.0/3.0) );
    veto = 0;

    /* check the alphaf-range for each trigger */
    if (strstr(tmpEvent->ifo, "H1") &&
	( (alphaF < alphafParams.h1_lo) || (alphaF > alphafParams.h1_hi ) ) )
    {
      veto = 1;
    }
    else if (strstr(tmpEvent->ifo, "H2") &&
	( (alphaF < alphafParams.h2_lo) || (alphaF > alphafParams.h2_hi ) ) )
    {
      veto = 1;
    }
    else if (strstr(tmpEvent->ifo, "L1") &&
	( (alphaF < alphafParams.l1_lo) || (alphaF > alphafParams.l1_hi ) ) )
    {
      veto = 1;
    }

    if (  (tmpEvent->psi0 < alphafParams.psi0cut))
    {
      veto =0;
    }

    if ( veto == 0 )
    {
      /* keep this template */
      if ( ! inspiralEventList  )
      {
        inspiralEventList = tmpEvent;
      }
      else
      {
        prevEvent->next = tmpEvent;
      }
      tmpEvent->next = NULL;
      prevEvent = tmpEvent;
    }
    else
    {
      /* discard this template */
      LALFreeSnglInspiral ( status->statusPtr, &tmpEvent );
    }
  }
  *eventHead = inspiralEventList;

  DETATCHSTATUSPTR (status);
  RETURN (status);

}



void
LALalphaFCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    REAL4                       alphaFhi,
    REAL4                       alphaFlo
    )

{
  SnglInspiralTable    *inspiralEventList = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;


  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );


  /* Remove all the triggers that are not in alphaFlo <= alphaF <= alphaFhi */

  thisEvent = *eventHead;

  while ( thisEvent )
  {
    SnglInspiralTable *tmpEvent = thisEvent;
    thisEvent = thisEvent->next;

    if ( ( (tmpEvent->alpha * pow(tmpEvent->f_final,(2.0/3.0))) <= alphaFhi )
      && ( (tmpEvent->alpha * pow(tmpEvent->f_final,(2.0/3.0))) >= alphaFlo ) )
    {
      /* keep this template */
      if ( ! inspiralEventList  )
      {
        inspiralEventList = tmpEvent;
      }
      else
      {
        prevEvent->next = tmpEvent;
      }
      tmpEvent->next = NULL;
      prevEvent = tmpEvent;
    }
    else
    {
      /* discard this template */
      LALFreeSnglInspiral ( status->statusPtr, &tmpEvent );
    }
  }
  *eventHead = inspiralEventList;

  DETATCHSTATUSPTR (status);
  RETURN (status);

}




void
LALIfoCutSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    CHAR                       *ifo
    )

{
  SnglInspiralTable    *ifoHead   = NULL;
  SnglInspiralTable    *thisEvent = NULL;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  ifoHead = XLALIfoCutSingleInspiral( eventHead, ifo );

  /* free events from other ifos */
  while ( *eventHead )
  {
    thisEvent = *eventHead;
    *eventHead = (*eventHead)->next;

    XLALFreeSnglInspiral( &thisEvent );
  }

  *eventHead = ifoHead;
  DETATCHSTATUSPTR (status);
  RETURN (status);
}


SnglInspiralTable *
XLALIfoCutSingleInspiral(
    SnglInspiralTable         **eventHead,
    char                       *ifo
    )

{
  SnglInspiralTable    *prevEvent   = NULL;
  SnglInspiralTable    *thisEvent   = NULL;
  SnglInspiralTable    *ifoHead     = NULL;
  SnglInspiralTable    *thisIfoTrig = NULL;

  /* check that eventHead is non-null */
  if ( ! eventHead )
  {
    XLAL_ERROR_NULL(XLAL_EIO);
  }

  /* Scan through a linked list of sngl_inspiral tables and return a
     pointer to the head of a linked list of tables for a specific IFO */

  thisEvent  = *eventHead;
  *eventHead = NULL;

  while ( thisEvent )
  {
    if ( ! strcmp( thisEvent->ifo, ifo ) )
    {
      /* ifos match so keep this event */
      if (  ifoHead  )
      {
        thisIfoTrig = thisIfoTrig->next = thisEvent;
      }
      else
      {
        ifoHead = thisIfoTrig = thisEvent;
      }

      /* remove from eventHead list */
      if ( prevEvent )
      {
        prevEvent->next = thisEvent->next;
      }

      /* move to next event */
      thisEvent = thisEvent->next;
      /* terminate ifo list */
      thisIfoTrig->next = NULL;
    }
    else
    {
      /* move along the list */
      if ( ! *eventHead )
      {
        *eventHead = thisEvent;
      }

      prevEvent = thisEvent;
      thisEvent = thisEvent->next;
    }
  }

  return( ifoHead );
}



void
LALIfoCountSingleInspiral(
    LALStatus                  *status,
    UINT4                      *numTrigs,
    SnglInspiralTable          *input,
    InterferometerNumber        ifoNumber
    )

{
  SnglInspiralTable    *thisEvent = NULL;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  /* check that output is null and input non-null */
  ASSERT( !(*numTrigs), status,
      LIGOMETADATAUTILSH_ENNUL, LIGOMETADATAUTILSH_MSGENNUL );
  ASSERT( input, status,
      LIGOMETADATAUTILSH_ENULL, LIGOMETADATAUTILSH_MSGENULL );

  /* Scan through a linked list of sngl_inspiral tables and return a
     pointer to the head of a linked list of tables for a specific IFO */
  for( thisEvent = input; thisEvent; thisEvent = thisEvent->next )
  {
    if ( ifoNumber == XLALIFONumber(thisEvent->ifo) )
    {
      /* IFOs match so count this trigger */
      ++(*numTrigs);
    }
  }

  DETATCHSTATUSPTR (status);
  RETURN (status);
}


/**
 * utility function to wrap a time t into the interval [start,
 * start+length).  Used by thinca.
 */


static INT8 thinca_ring_wrap(INT8 t, INT8 ring_start, INT8 ring_length)
{
  t = (t - ring_start) % ring_length;
  if( t < 0 )
    t += ring_length;
  return ring_start + t;
}


/**
 * Time slide segs in a seglist by a time = slideTime.
 */


int
XLALTimeSlideSegList(
       LALSegList        *seglist,
       const LIGOTimeGPS *ringStartTime,
       const LIGOTimeGPS *ringEndTime,
       const LIGOTimeGPS *slideTime
)
{
  INT8 ringStartNS = 0;	/* initialized to silence warning */
  INT8 ringEndNS = 0;	/* initialized to silence warning */
  INT8 slideNS;
  LALSeg tmpSeg;
  LALSegList tmplist;
  unsigned i;

  /* make sure the segment list has been properly initialized */
  if ( seglist->initMagic != SEGMENTSH_INITMAGICVAL ) {
    XLALPrintError("%s(): segment list not initialized\n", __func__);
    XLAL_ERROR(XLAL_EINVAL);
  }

  /* calculate the slide time in nanoseconds */
  slideNS = XLALGPSToINT8NS( slideTime );

  /* convert the ring boundaries to nanoseconds */
  if( ringStartTime && ringEndTime ) {
    ringStartNS = XLALGPSToINT8NS( ringStartTime );
    ringEndNS = XLALGPSToINT8NS( ringEndTime );
  }

  /* initialize segment list */
  XLALSegListInit( &tmplist );

  /* loop over the entries in seglist */
  for( i = 0; i < seglist->length; i++ ) {
    /* convert the segment boundaries to nanoseconds */
    INT8 segStartNS = XLALGPSToINT8NS( &seglist->segs[i].start );
    INT8 segEndNS = XLALGPSToINT8NS( &seglist->segs[i].end );

    /* ignore zero-length segments */
    if( segEndNS == segStartNS )
      continue;

    /* verify segment lies in ring */
    if( ringStartTime && ringEndTime && ( segStartNS < ringStartNS || segEndNS > ringEndNS ) ) {
      XLALPrintError("%s(): detected segment outside of ring\n", __func__);
      XLAL_ERROR(XLAL_EINVAL);
    }

    /* slide the segment */
    segStartNS += slideNS;
    segEndNS += slideNS;

    if( ringStartTime && ringEndTime ) {
      /* wrap segment around ring */
      segStartNS = thinca_ring_wrap(segStartNS, ringStartNS, ringEndNS - ringStartNS);
      segEndNS = thinca_ring_wrap(segEndNS, ringStartNS, ringEndNS - ringStartNS);

      if( segEndNS <= segStartNS ) {
        /* segment was split.  before adding each piece, confirm that it
         * has non-zero length */
        if( segEndNS != ringStartNS ) {
          XLALINT8NSToGPS( &tmpSeg.start, ringStartNS );
          XLALINT8NSToGPS( &tmpSeg.end, segEndNS );
        }
        /* this piece can't have zero length */
        XLALINT8NSToGPS( &tmpSeg.start, segStartNS );
        XLALINT8NSToGPS( &tmpSeg.end, ringEndNS );
      } else {
        /* segment was not split */
        XLALINT8NSToGPS( &tmpSeg.start, segStartNS );
        XLALINT8NSToGPS( &tmpSeg.end, segEndNS );
      }
    } else {
      /* no ring to wrap segment around */
      XLALINT8NSToGPS( &tmpSeg.start, segStartNS );
      XLALINT8NSToGPS( &tmpSeg.end, segEndNS );
    }

    XLALSegListAppend( &tmplist, &tmpSeg );
  }

  /* clear the old list */
  XLALSegListClear( seglist );

  /* copy segments from new list into original */
  for ( i = 0; i < tmplist.length; i++)
    XLALSegListAppend( seglist, &(tmplist.segs[i]));

  /* clear the temporary list */
  XLALSegListClear( &tmplist );

  /* sort and clean up the new list */
  XLALSegListCoalesce( seglist );

  /* done */
  return 0;
}


/* ======================================= */

void
XLALTimeSlideSingleInspiral(
    SnglInspiralTable          *triggerList,
    const LIGOTimeGPS          *startTime,
    const LIGOTimeGPS          *endTime,
    const LIGOTimeGPS           slideTimes[LAL_NUM_IFO]
    )

{
  INT8 ringStartNS = 0;	/* initialized to silence warning */
  INT8 ringLengthNS = 0;	/* initialized to silence warning */

  if ( startTime && endTime )
  {
    ringStartNS = XLALGPSToINT8NS( startTime );
    ringLengthNS = XLALGPSToINT8NS( endTime ) - ringStartNS;
  }

  for( ; triggerList; triggerList = triggerList->next )
  {
    /* calculate the slide time in nanoseconds */
    INT8 slideNS = XLALGPSToINT8NS( &slideTimes[XLALIFONumber(triggerList->ifo)] );
    /* and trigger time in nanoseconds */
    INT8 trigTimeNS = XLALGPSToINT8NS( &triggerList->end );

    /* slide trigger time */
    trigTimeNS += slideNS;

    /* wrap trigger time to be in [startTime, endTime) */
    if ( startTime && endTime )
      trigTimeNS = thinca_ring_wrap(trigTimeNS, ringStartNS, ringLengthNS);

    /* convert back to LIGOTimeGPS */
    XLALINT8NSToGPS( &triggerList->end, trigTimeNS);
  }
}



SnglInspiralTable *
XLALPlayTestSingleInspiral(
    SnglInspiralTable          *eventHead,
    LALPlaygroundDataMask      *dataType
    )

{
  SnglInspiralTable    *inspiralEventList = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;

  INT8 triggerTime = 0;
  INT4 isPlay = 0;
  INT4 numTriggers;

  /* Remove all the triggers which are not of the desired type */

  numTriggers = 0;
  thisEvent = eventHead;

  if ( (*dataType == playground_only) || (*dataType == exclude_play) )
  {
    while ( thisEvent )
    {
      SnglInspiralTable *tmpEvent = thisEvent;
      thisEvent = thisEvent->next;

      triggerTime = XLALGPSToINT8NS( &(tmpEvent->end) );
      isPlay = XLALINT8NanoSecIsPlayground( triggerTime );

      if ( ( (*dataType == playground_only)  && isPlay ) ||
          ( (*dataType == exclude_play) && ! isPlay) )
      {
        /* keep this trigger */
        if ( ! inspiralEventList  )
        {
          inspiralEventList = tmpEvent;
        }
        else
        {
          prevEvent->next = tmpEvent;
        }
        tmpEvent->next = NULL;
        prevEvent = tmpEvent;
        ++numTriggers;
      }
      else
      {
        /* discard this template */
        XLALFreeSnglInspiral ( &tmpEvent );
      }
    }
    eventHead = inspiralEventList;
    if ( *dataType == playground_only )
    {
      XLALPrintInfo( "Kept %d playground triggers \n", numTriggers );
    }
    else if ( *dataType == exclude_play )
    {
      XLALPrintInfo( "Kept %d non-playground triggers \n", numTriggers );
    }
  }
  else if ( *dataType == all_data )
  {
    XLALPrintInfo( "Keeping all triggers since all_data specified\n" );
  }
  else
  {
    XLALPrintInfo( "Unknown data type, returning no triggers\n" );
    eventHead = NULL;
  }

  return(eventHead);
}



void
LALPlayTestSingleInspiral(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    LALPlaygroundDataMask      *dataType
    )

{
  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  *eventHead = XLALPlayTestSingleInspiral(*eventHead, dataType);

  DETATCHSTATUSPTR (status);
  RETURN (status);
}



void
LALCreateTrigBank(
    LALStatus                  *status,
    SnglInspiralTable         **eventHead,
    SnglInspiralParameterTest  *test
    )

{
  SnglInspiralTable    *trigBankList = NULL;
  SnglInspiralTable   **eventHandle = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;

  INT4 numEvents = 0;
  INT4 i = 0;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );


  /* count the number of events */
  for ( thisEvent = *eventHead; thisEvent; thisEvent = thisEvent->next )
  {
    ++numEvents;
  }

  eventHandle = (SnglInspiralTable **)
    LALCalloc( numEvents, sizeof(SnglInspiralTable *) );

  for ( i = 0, thisEvent = *eventHead; i < numEvents;
      ++i, thisEvent = thisEvent->next )
  {
    eventHandle[i] = thisEvent;
  }

  if ( *test == m1_and_m2 )
  {
    LALInfo( status, "sorting events by mass... " );
    qsort( eventHandle, numEvents, sizeof(eventHandle[0]),
        LALCompareSnglInspiralByMass );
    LALInfo( status, "done\n" );
  }
  else if ( *test == psi0_and_psi3 )
  {
    LALInfo( status, "sorting events by psi... " );
    qsort( eventHandle, numEvents, sizeof(eventHandle[0]),
        LALCompareSnglInspiralByPsi );
    LALInfo( status, "done\n" );
  }
  else
  {
    ABORT( status, LIGOMETADATAUTILSH_ETEST, LIGOMETADATAUTILSH_MSGETEST );
  }

  /* create a linked list of sorted templates */
  LALInfo( status, "discarding template with duplicate masses: " );

  trigBankList = prevEvent = eventHandle[0];

  for ( i = 1; i < numEvents; ++i )
  {
    if ( *test == m1_and_m2 )
    {
      if ( (prevEvent->mass1 == eventHandle[i]->mass1)  &&
          (prevEvent->mass2 == eventHandle[i]->mass2) )
      {
        /* discard the event as it is a duplicate */
        LALFreeSnglInspiral( status->statusPtr, &(eventHandle[i]) );
        LALInfo( status, "-" );
      }
      else
      {
        /* add the event to the linked list */
        prevEvent = prevEvent->next = eventHandle[i];
        LALInfo( status, "+" );
      }
    }
    else if ( *test == psi0_and_psi3 )
    {
      if ( (prevEvent->psi0 == eventHandle[i]->psi0)  &&
          (prevEvent->psi3 == eventHandle[i]->psi3) )
      {
        /* discard the event as it is a duplicate */
        LALFreeSnglInspiral( status->statusPtr, &(eventHandle[i]) );
        LALInfo( status, "-" );
      }
      else
      {
        /* add the event to the linked list */
        prevEvent = prevEvent->next = eventHandle[i];
        LALInfo( status, "+" );
      }
    }
    else
    {
      ABORT( status, LIGOMETADATAUTILSH_ETEST, LIGOMETADATAUTILSH_MSGETEST );
    }
  }

  /* if the list is non-emnpty, make sure it is terminated */
  if ( prevEvent ) prevEvent->next = NULL;

  LALFree( eventHandle );

  /* return the head of the linked list in eventHead */

  *eventHead = trigBankList;

  DETATCHSTATUSPTR (status);
  RETURN (status);
}



void
LALIncaCoincidenceTest(
    LALStatus                  *status,
    SnglInspiralTable         **ifoAOutput,
    SnglInspiralTable         **ifoBOutput,
    SnglInspiralTable          *ifoAInput,
    SnglInspiralTable          *ifoBInput,
    SnglInspiralAccuracy       *errorParams
    )

{
  SnglInspiralTable    *currentTrigger[2];
  SnglInspiralTable    *coincidentEvents[2];
  SnglInspiralTable    *outEvent[2];
  SnglInspiralTable    *currentEvent;

  INT8 ta,tb;
  INT4 j;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  memset( currentTrigger, 0, 2 * sizeof(SnglInspiralTable *) );
  memset( coincidentEvents, 0, 2 * sizeof(SnglInspiralTable *) );
  memset( outEvent, 0, 2 * sizeof(SnglInspiralTable *) );


  if ( ! ifoAInput )
  {
    LALInfo( status, "No input triggers from IFO A, exiting");
  }

  if ( ! ifoBInput )
  {
    LALInfo( status, "No input triggers from IFO B, exiting");
  }

  currentTrigger[1] = ifoBInput;

  for( currentTrigger[0]=ifoAInput; currentTrigger[0];
      currentTrigger[0] = currentTrigger[0]->next  )
  {
    ta = XLALGPSToINT8NS( &(currentTrigger[0]->end) );

    /* spin ifo b until the current trigger is within the coinicdence */
    /* window of the current ifo a trigger                            */
    while ( currentTrigger[1] )
    {
      tb = XLALGPSToINT8NS( &(currentTrigger[1]->end) );

      if ( tb > ta - errorParams->dt )
      {
        /* we have reached the time coinicidence window */
        break;
      }
      currentTrigger[1] = currentTrigger[1]->next;
    }

    /* look for coincident events in B within the time window */
    currentEvent = currentTrigger[1];

    while ( currentTrigger[1] )
    {
      tb = XLALGPSToINT8NS( &(currentTrigger[1]->end) );

      if (tb > ta + errorParams->dt )
      {
        /* we are outside the time coincidence so move to next event */
        break;
      }
      else
      {
        /* call the LAL function which compares events parameters */
        LALCompareSnglInspiral( status->statusPtr, currentTrigger[0],
            currentTrigger[1], errorParams );
      }

      if ( errorParams->match )
      {
        /* store this event for output */
        LALInfo( status, "    >>> found coincidence <<<" );

        for ( j = 0; j < 2; ++j )
        {
          if ( ! coincidentEvents[j] )
          {
            coincidentEvents[j] = outEvent[j] = (SnglInspiralTable *)
              LALCalloc( 1, sizeof(SnglInspiralTable) );
          }
          else
          {
            outEvent[j] = outEvent[j]->next = (SnglInspiralTable *)
              LALCalloc( 1, sizeof(SnglInspiralTable) );
          }

          memcpy( outEvent[j], currentTrigger[j], sizeof(SnglInspiralTable) );
          outEvent[j]->next = NULL;
        }
      }

      currentTrigger[1] = currentTrigger[1]->next;

    } /* end loop over current events */

    /* go back to saved current IFO B trigger */
    currentTrigger[1] = currentEvent;

  } /* end loop over ifo A events */

  *ifoAOutput = coincidentEvents[0];
  *ifoBOutput = coincidentEvents[1];

  DETATCHSTATUSPTR (status);
  RETURN (status);
}



void
LALTamaCoincidenceTest(
    LALStatus                  *status,
    SnglInspiralTable         **ifoAOutput,
    SnglInspiralTable         **ifoBOutput,
    SnglInspiralTable          *ifoAInput,
    SnglInspiralTable          *ifoBInput,
    SnglInspiralAccuracy       *errorParams,
    SnglInspiralClusterChoice   clusterchoice
    )

{
  SnglInspiralTable    *currentTrigger[2];
  SnglInspiralTable    *coincidentEvents[2];
  SnglInspiralTable    *outEvent[2];
  SnglInspiralTable    *currentEvent = NULL;
  SnglInspiralTable    *timeCoincHead = NULL;
  SnglInspiralTable    *thisTimeCoinc = NULL;

  INT8 ta,tb;
  INT4 j;

  INITSTATUS(status);
  ATTATCHSTATUSPTR( status );

  memset( currentTrigger, 0, 2 * sizeof(SnglInspiralTable *) );
  memset( coincidentEvents, 0, 2 * sizeof(SnglInspiralTable *) );
  memset( outEvent, 0, 2 * sizeof(SnglInspiralTable *) );

  if ( ! ifoAInput )
  {
    LALInfo( status, "No input triggers from IFO A, exiting");
  }

  if ( ! ifoBInput )
  {
    LALInfo( status, "No input triggers from IFO B, exiting");
  }

  currentTrigger[1] = ifoBInput;

  for( currentTrigger[0]=ifoAInput; currentTrigger[0];
      currentTrigger[0] = currentTrigger[0]->next  )
  {
    ta = XLALGPSToINT8NS( &(currentTrigger[0]->end) );

    LALInfo( status, printf("  using IFO A trigger at %d + %10.10f\n",
          currentTrigger[0]->end.gpsSeconds,
          ((REAL4) currentTrigger[0]->end.gpsNanoSeconds * 1e-9) ));

    /* spin ifo b until the current trigger is within the coinicdence */
    /* window of the current ifo a trigger                            */
    while ( currentTrigger[1] )
    {
      tb = XLALGPSToINT8NS( &(currentTrigger[1]->end) );

      if ( tb > ta - errorParams->dt )
      {
        /* we have reached the time coinicidence window */
        break;
      }
      currentTrigger[1] = currentTrigger[1]->next;
    }


    /* look for coincident events in B within the time window */
    currentEvent = currentTrigger[1];

    while ( currentTrigger[1] )
    {
      tb = XLALGPSToINT8NS( &(currentTrigger[1]->end) );

      if (tb > ta + errorParams->dt )
      {
        /* we are outside the time coincidence so move to next event */
        LALInfo( status, "outside the time coincidence window\n" );
        break;
      }
      else
      {
        /* store all time coincident triggers */
        if ( ! timeCoincHead )
        {
          timeCoincHead = thisTimeCoinc = (SnglInspiralTable *)
            LALCalloc( 1, sizeof(SnglInspiralTable) );
        }
        else
        {
          thisTimeCoinc = thisTimeCoinc->next = (SnglInspiralTable *)
            LALCalloc( 1, sizeof(SnglInspiralTable) );
        }

        memcpy( thisTimeCoinc, currentTrigger[1],
            sizeof(SnglInspiralTable) );

        thisTimeCoinc->next = NULL;
      }
      currentTrigger[1] = currentTrigger[1]->next;


    }  /* end loop over current events */


    /* take the loudest time coincident trigger and compare other params */
    if ( timeCoincHead )
    {
      LALClusterSnglInspiralTable ( status->statusPtr, &timeCoincHead,
          2 * errorParams->dt, clusterchoice);

      currentTrigger[1] = timeCoincHead;


      /* call the LAL function which compares events parameters */
      LALCompareSnglInspiral( status->statusPtr, currentTrigger[0],
          currentTrigger[1], errorParams );

      if ( errorParams->match )
      {
        /* store this event for output */
        LALInfo( status, "    >>> found coincidence <<<\n" );

        for ( j = 0; j < 2; ++j )
        {
          if ( ! coincidentEvents[j] )
          {
            coincidentEvents[j] = outEvent[j] = (SnglInspiralTable *)
              LALCalloc( 1, sizeof(SnglInspiralTable) );
          }
          else
          {
            outEvent[j] = outEvent[j]->next = (SnglInspiralTable *)
              LALCalloc( 1, sizeof(SnglInspiralTable) );
          }

          memcpy( outEvent[j], currentTrigger[j], sizeof(SnglInspiralTable) );
          outEvent[j]->next = NULL;
        }
      }

      /* reset the list of time coincident triggers to null */
      LALFreeSnglInspiral( status->statusPtr, &timeCoincHead );
      timeCoincHead = NULL;
    }
    /* go back to saved current IFO B trigger */
    currentTrigger[1] = currentEvent;

  } /* end loop over ifo A events */

  *ifoAOutput = coincidentEvents[0];
  *ifoBOutput = coincidentEvents[1];

  DETATCHSTATUSPTR (status);
  RETURN (status);
}



int
XLALMaxSnglInspiralOverIntervals(
    SnglInspiralTable         **eventHead,
    INT4                       deltaT
    )

{
  SnglInspiralTable    *inspiralEventList = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *nextEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;

  /* if there are no events, then no-op */
  if ( ! *eventHead )
    return (0);

  inspiralEventList = *eventHead;
  thisEvent = *eventHead;
  nextEvent = thisEvent->next;

  while ( nextEvent )
  {
    if ( end_time_sec(nextEvent) == end_time_sec(thisEvent) &&
        end_time_nsec(nextEvent)/deltaT == end_time_nsec(thisEvent)/deltaT )
    {
      if ( nextEvent->snr > thisEvent->snr )
      {
        /* replace thisEvent with nextEvent */
        XLALFreeSnglInspiral ( &thisEvent );

        /* deal with start of the list */
        if (prevEvent)
          prevEvent->next = nextEvent;
        else
          inspiralEventList = nextEvent;

        /* standard stuff */
        thisEvent = nextEvent;
        nextEvent = thisEvent->next;
      }
      else
      {
        /* get rid of nextEvent */
        thisEvent->next = nextEvent->next;
        XLALFreeSnglInspiral ( &nextEvent );
        nextEvent = thisEvent->next;
      }
    }
    else
    {
      /* step to next set of events */
      prevEvent=thisEvent;
      thisEvent=nextEvent;
      nextEvent = thisEvent->next;
    }
  }

  *eventHead = inspiralEventList;

  return (0);
}


INT4 XLALCountSnglInspiral( SnglInspiralTable *head )

{
  INT4 length;
  SnglInspiralTable *event;

  if ( !head )
  {
    return( 0 );
  }

  /* count the number of events in the list */
  for(length = 0, event = head; event; event = event->next)
    length++;

  return length;
}


SnglInspiralTable *
XLALMassCut(
    SnglInspiralTable         *eventHead,
    const char                *massCut,
    REAL4                      massRangeLow,
    REAL4                      massRangeHigh,
    REAL4                      mass2RangeLow,
    REAL4                      mass2RangeHigh
    )

{
  SnglInspiralTable    *inspiralEventList = NULL;
  SnglInspiralTable    *thisEvent = NULL;
  SnglInspiralTable    *prevEvent = NULL;

  REAL4 massParam;
  REAL4 mass2Param;
  INT4 numTriggers;
  INT4 massBOOL;
  REAL4 eps = 1.e-08; /* Safeguard against roundoff error in eta */

  /* Remove all the triggers which are not of the desired type */

  numTriggers = 0;
  thisEvent = eventHead;

  while ( thisEvent )
  {
    SnglInspiralTable *tmpEvent = thisEvent;
    thisEvent = thisEvent->next;
    massParam = 0;
    mass2Param = 0;

    if ( ! strcmp(massCut,"mchirp") )
    {
      massParam = tmpEvent->mchirp;
    }
    else if ( ! strcmp(massCut,"eta") )
    {
      massParam = tmpEvent->eta;
    }
    else if ( ! strcmp(massCut,"mtotal") )
    {
      massParam = tmpEvent->mass1 + tmpEvent->mass2;
    }
    else if ( ! strcmp(massCut,"mcomp") )
    {
      massParam = tmpEvent->mass1;
      mass2Param = tmpEvent->mass2;
    }

    if ( ! strcmp(massCut,"mcomp") )
    {
      if ( ( massParam >= massRangeLow ) && ( massParam < massRangeHigh ) &&
           ( mass2Param >= mass2RangeLow ) && ( mass2Param < mass2RangeHigh ) )
      {
        massBOOL = 1;
      }
      else
      {
        massBOOL = 0;
      }
    }
    else if ( ! strcmp(massCut,"eta") )
    {
      if ( ( massParam >= massRangeLow - eps ) &&
           ( massParam <= massRangeHigh + eps ) )
      {
        massBOOL = 1;
      }
      else
      {
        massBOOL = 0;
      }
    }
    else
    {
      if ( ( massParam >= massRangeLow ) && ( massParam < massRangeHigh ) )
      {
        massBOOL = 1;
      }
      else
      {
        massBOOL = 0;
      }
    }

    if ( massBOOL )
    {
      /* keep this trigger */
      if ( ! inspiralEventList  )
      {
        inspiralEventList = tmpEvent;
      }
      else
      {
        prevEvent->next = tmpEvent;
      }
      tmpEvent->next = NULL;
      prevEvent = tmpEvent;
      ++numTriggers;
    }
    else
    {
      /* discard this template */
      XLALFreeSnglInspiral ( &tmpEvent );
    }
  }

  eventHead = inspiralEventList;
  return(eventHead);
}


int XLALAddSnglInspiralCData( CDataNode **cdataStrCat, CHAR *id )
{
  int notPresent = 1;
  int addedCData = 0;
  CDataNode *thisCData = NULL;

  thisCData = *cdataStrCat;
  *cdataStrCat = NULL;

  while ( thisCData ) {
    if ( strcmp(thisCData->cdataStrNode, id ) )  {
      notPresent *= 1;
    }
    else {
      notPresent *= 0;
    }

    if ( ! *cdataStrCat ) {
      *cdataStrCat = thisCData;
    }

    thisCData = thisCData->next;
  }

  if ( notPresent ) {
    (*cdataStrCat)->next = (CDataNode *) LALCalloc( 1, sizeof(CDataNode) );
    strcpy( (*cdataStrCat)->next->cdataStrNode , id );
    addedCData = 1;
  }

  return( addedCData );
}

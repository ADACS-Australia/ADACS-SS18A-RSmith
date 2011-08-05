/*
*  Copyright (C) 2007 Drew Keppel, Duncan Brown, Gareth Jones, Peter Shawhan, Thomas Cokelaer, Laszlo Vereb
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
 * \defgroup GenerateInspiral_h GenerateInspiral_h
 * \ingroup inject
 */

/**
\author Cokelaer, T.
\file
\ingroup GenerateInspiral_h

\brief %Header file for the inspiral injection interface code.

The code contained in GenerateInspiral.c is an interface between the
injection package and the inspiral package. More precisely, the
function GenerateInspiral.c is used within the FindChirpSimulation.c
file of the FindChirp package in order to inject waveforms into real
data. The injection is done through the inject package in order to
take into account the interferometer position, binary orientation ...

GenerateInspiral has the capability of injecting both waveform designed
within the inspiral package (TaylorT1, T2, T3, PadeT1, EOB, and spinning
waveform) and the inject package (so-called PPN waveform).
also a test code as well which allows to check the output of
code. It is called InjectionInterfaceTest.c

\heading{Synopsis}
\code
#include <lal/GenerateInspiral.h>
\endcode

*/

#ifndef _GENERATEINSPIRAL_H
#define _GENERATEINSPIRAL_H

#include <lal/LALStdlib.h>
#include <lal/LALInspiral.h>
#include <lal/GeneratePPNInspiral.h>

#include <lal/LIGOMetadataTables.h>
#include <lal/SeqFactories.h>

#include <lal/Units.h>

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

NRCSID( GENERATEINSPIRALH,
    "$Id$" );

/**\name Error Codes */ /*@{*/
#define GENERATEINSPIRALH_ENORM 0
#define GENERATEINSPIRALH_ENULL 1
#define GENERATEINSPIRALH_EDFDT 2
#define GENERATEINSPIRALH_EZERO 3
#define GENERATEINSPIRALH_MSGENORM "Normal exit"
#define GENERATEINSPIRALH_MSGENULL "Null pointer"
#define GENERATEINSPIRALH_MSGEDFDT "Waveform sampling interval is too large"
#define GENERATEINSPIRALH_MSGEZERO "inclination zero for SpinTaylor waveform"
/*@}*/


/* parameter for the EOB at 3PN. In principle, the three */
/* following parameter should be set to zero.            */
#define GENERATEINSPIRAL_ZETA2       0.
#define GENERATEINSPIRAL_OMEGAS      0.
#define GENERATEINSPIRAL_THETA       0.

/* For the spinning case. might be changed later or include */
/* in the injection itself                                  */
#define GENERATEINSPIRAL_SOURCETHETA 1.
#define GENERATEINSPIRAL_SOURCEPHI   2.

/* Default low freqnecy cutoff for injections */
#define GENERATEINSPIRAL_DEFAULT_FLOWER 40


void
LALGenerateInspiral(
    LALStatus        *status,
    CoherentGW       *waveform,
    SimInspiralTable *params,
    PPNParamStruc    *ppnParamsInputOutput
    );

/* Legacy wrappers for parsing approximants and orders */
/* three function to read the order and approximant from a string */
void
LALGetOrderFromString(
    LALStatus   *status,
    CHAR        *message,
    LALPNOrder  *result
    );

void
LALGetApproximantFromString(
    LALStatus   *status,
    CHAR        *message,
    Approximant *result
    );

/**	Convert a string provided by the #CoherentGW structure in order to retrieve
 *	the approximant of the waveform to generate.
 *	@param[out]	inter	: the level of the spin interaction
 *	@param[in]	thisEvent	: string containing the spin interaction
 *	@return error code
 */
int XLALGetSpinInteractionFromString(LALSpinInteraction *inter, CHAR *thisEvent);

int XLALGetAxisChoiceFromString(InputAxis *axisChoice, CHAR *thisEvent);

int XLALGetAdaptiveIntFromString(UINT4 *fixedStep, CHAR *thisEvent);

int XLALGetInspiralOnlyFromString(UINT4 *inspiralOnly, CHAR *thisEvent);

/*  three function to populate the needed structures */
void
LALGenerateInspiralPopulatePPN(
    LALStatus           *status,
    PPNParamStruc       *ppnParams,
    SimInspiralTable    *thisEvent
    );

void
LALGenerateInspiralPopulateInspiral(
    LALStatus           *status,
    InspiralTemplate    *inspiralParams,
    SimInspiralTable    *thisEvent,
    PPNParamStruc       *ppnParams
    );

/* XLAL functions for parsing approximants and orders */
/* and also for populating structures */
int
XLALGetOrderFromString(
    CHAR       * restrict message,
    LALPNOrder * restrict result
    );

int
XLALGetApproximantFromString(
    CHAR        * restrict message,
    Approximant * restrict result
    );

int
XLALGenerateInspiralPopulatePPN(
    PPNParamStruc    * restrict ppnParams,
    SimInspiralTable * restrict thisEvent
    );

int
XLALGenerateInspiralPopulateInspiral(
    InspiralTemplate * restrict inspiralParams,
    SimInspiralTable * restrict thisEvent,
    PPNParamStruc    * restrict ppnParams
    );

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _GENERATEINSPIRAL_H */

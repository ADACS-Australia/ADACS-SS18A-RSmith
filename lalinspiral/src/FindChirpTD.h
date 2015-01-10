/*
*  Copyright (C) 2007 Duncan Brown, Jolien Creighton
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
 * File Name: FindChirpTD.h
 *
 * Author: Brown, D. A., and Creighton, J. D. E.
 *
 *-----------------------------------------------------------------------
 */

#ifndef _FINDCHIRPTDH_H
#define _FINDCHIRPTDH_H

#include <lal/LALDatatypes.h>
#include <lal/RealFFT.h>
#include <lal/LALInspiral.h>
#include <lal/FindChirp.h>
#include <lal/FindChirpChisq.h>

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

/**
 * \defgroup FindChirpTD_h Header FindChirpTD.h
 * \ingroup lalinspiral_findchirp
 * \author Brown, D. A., and Creighton, J. D. E.
 *
 * \brief Provides structures and functions to condition interferometer data
 * and generate binary inspiral chirps using time domain waveforms.
 *
 * ### Synopsis ###
 *
 * \code
 * #include <lal/FindChirpTD.h>
 * \endcode
 *
 */
/*@{*/

/**\name Error Codes */
/*@{*/
#define FINDCHIRPTDH_ENULL 1
#define FINDCHIRPTDH_ENNUL 2
#define FINDCHIRPTDH_EALOC 3
#define FINDCHIRPTDH_ENUMZ 4
#define FINDCHIRPTDH_ESEGZ 5
#define FINDCHIRPTDH_EMISM 6
#define FINDCHIRPTDH_EDELT 7
#define FINDCHIRPTDH_EFLOW 8
#define FINDCHIRPTDH_EDYNR 9
#define FINDCHIRPTDH_EISTN 10
#define FINDCHIRPTDH_EDIVZ 11
#define FINDCHIRPTDH_EMAPX 12
#define FINDCHIRPTDH_ELONG 13
#define FINDCHIRPTDH_EEMTY 14
#define FINDCHIRPTDH_ESMPL 15
/*@}*/

/** \cond DONT_DOXYGEN */
#define FINDCHIRPTDH_MSGENULL "Null pointer"
#define FINDCHIRPTDH_MSGENNUL "Non-null pointer"
#define FINDCHIRPTDH_MSGEALOC "Memory allocation error"
#define FINDCHIRPTDH_MSGENUMZ "Invalid number of segments"
#define FINDCHIRPTDH_MSGESEGZ "Invalid number of points in segments"
#define FINDCHIRPTDH_MSGEMISM "Mismatch between number of points in segments"
#define FINDCHIRPTDH_MSGEDELT "deltaT is zero or negative"
#define FINDCHIRPTDH_MSGEFLOW "Low frequency cutoff is negative"
#define FINDCHIRPTDH_MSGEDYNR "Dynamic range scaling is zero or negative"
#define FINDCHIRPTDH_MSGEISTN "Truncation of inverse power spectrum is negative"
#define FINDCHIRPTDH_MSGEDIVZ "Attempting to divide by zero"
#define FINDCHIRPTDH_MSGEMAPX "Mismatch in waveform approximant"
#define FINDCHIRPTDH_MSGELONG "Time domain template too long"
#define FINDCHIRPTDH_MSGEEMTY "Could not find end of chirp in xfacVec"
#define FINDCHIRPTDH_MSGESMPL "Waveform sampling interval is too large"
/** \endcond */

void
LALFindChirpTDData (
    LALStatus                  *status,
    FindChirpSegmentVector     *fcSegVec,
    DataSegmentVector          *dataSegVec,
    FindChirpDataParams        *params
    );

void
LALFindChirpTDTemplate (
    LALStatus                  *status,
    FindChirpTemplate          *fcTmplt,
    InspiralTemplate           *theTmplt,
    FindChirpTmpltParams       *params
    );

void
LALFindChirpTDNormalize(
    LALStatus                  *status,
    FindChirpTemplate          *fcTmplt,
    FindChirpSegment           *fcSeg,
    FindChirpDataParams        *params
    );


/*@}*/

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _FINDCHIRPTDH_H */

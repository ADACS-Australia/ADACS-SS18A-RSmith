 /*
  * Copyright (C) 2004, 2005 Cristina V. Torres
  *                          E. Chassande-Mottin
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
 * File Name: DestroyTimeFreqParam.c
 *
 * Maintainer: Torres, C (Univ TX at Browsville)
 * Author: Chassande-Mottin, E.
 *
 *
 *-----------------------------------------------------------------------
 *
 * NAME
 * DestroyTimeFreqParam
 *
 * SYNOPSIS
 * void LALDestroyTimeFreqParam ( LALStatus *,  TimeFreqParam **param );
 *
 * DESCRIPTION
 * Returns to system storage allocated by CreateTimeFreqParam
 *
 * DIAGNOSTICS
 * param == NULL, *param == NULL, free failure
 *
 * CALLS
 * LALFree
 *
 * NOTES
 *
 *-----------------------------------------------------------------------
 */

#include <lal/TimeFreq.h>

void LALDestroyTimeFreqParam (LALStatus *status, TimeFreqParam **param)
{
  /*  Initialize status */
  INITSTATUS(status);

  /* Check param: report if NULL */
  ASSERT (param != NULL, status, DESTROYTFP_ENULL, DESTROYTFP_MSGENULL);

  /*  Check *param: report if NULL */
  ASSERT (*param != NULL, status, DESTROYTFP_ENULL, DESTROYTFP_MSGENULL);

  switch ((*param)->type) {
  case Spectrogram :

    LALSDestroyVector(status,&(*param)->windowT);
    (*param)->type = Undefined;

    break;
  case WignerVille :

    (*param)->type = Undefined;

    break;
  case PSWignerVille :

    LALSDestroyVector(status,&(*param)->windowT);
    LALSDestroyVector(status,&(*param)->windowF);
    (*param)->type = Undefined;

    break;
  case RSpectrogram :

    LALSDestroyVector(status,&(*param)->windowT);
    (*param)->type = Undefined;

    break;
  default :
    ABORT(status,DESTROYTFP_ETYPE, DESTROYTFP_MSGETYPE);

  }
  /* Ok, now let's free allocated storage */

  LALFree ( *param );	      /* free param struct itself */
  *param = NULL;	      /* make sure we don't point to freed struct */

  RETURN (status);
}

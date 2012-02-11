	/*
*  Copyright (C) 2007 David Churches, B.S. Sathyaprakash
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
\author Sathyaprakash, B. S.
\file

\brief Module to compute the correlation of two data sets.

Suitable only when REAL4VectorFFT is used (i.e. rfftwi_one of fftw).

\c Notation: The input struct has two vectors: \c signal1
and <tt>signal2.</tt> This module computes the correlation by
shifting \c signal2 with respect to positive time-direction
relative to \c signal1. Thus, if \c signal1
denotes the detector output in which a signal, say <tt>signal2,</tt>
is present at time \f$t_0,\f$ then the correlation peaks at \f$t_0.\f$

\heading{Prototypes}

<tt>LALInspiralWaveCorrelate()</tt>

\heading{Description}
The module expects two inputs <tt>signal1, signal2</tt>
in the Fourier-domain, computes their correlation weighted by
the noise \c psd,  and returns the correlated output in
the time-domain. More precisely, given the Discrete
Fourier transform (in the notation of {\em fftw})
\f$H_k\f$ and \f$Q_k\f$ of vectors \f$h_k\f$ and \f$q_k,\f$
\f$k=0,\ldots n-1,\f$ this module computes the inverse Fourier
transform of the weighted correlation \f$C_k\f$ defined as
\f[C_k = \frac{H_k Q_k + H_{n-k} Q_{n-k} }{S_k}, \ \
C_{n-k} = \frac{H_k Q_{n-k} + H_{n-k} Q_k }{S_k}, \ \
           k=1,\ldots,\frac{n}{2}-1.\f]

\heading{Uses}
\code
LALREAL4VectorFFT()
\endcode

*/

#include <lal/LALNoiseModelsInspiral.h>
#include <lal/RealFFT.h>

void
LALInspiralWaveCorrelate
   (
   LALStatus                *status,
   REAL4Vector              *output,
   InspiralWaveCorrelateIn   corrin
   )
{
  INT4 n, nby2, i, k;
  REAL8 psd, r1, r2, i1, i2, f;
  REAL4Vector buff;


  INITSTATUS(status);
  ATTATCHSTATUSPTR(status);

  ASSERT (output,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);
  ASSERT (output->data,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);
  ASSERT (corrin.signal1.data,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);
  ASSERT (corrin.signal2.data,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);
  ASSERT (corrin.psd.data,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);

  ASSERT (corrin.signal1.length == corrin.signal2.length, status, LALNOISEMODELSH_ESIZE, LALNOISEMODELSH_MSGESIZE);
  ASSERT (output->length == corrin.signal1.length, status, LALNOISEMODELSH_ESIZE, LALNOISEMODELSH_MSGESIZE);
  ASSERT (corrin.psd.length == corrin.signal1.length/2+1, status, LALNOISEMODELSH_ESIZE, LALNOISEMODELSH_MSGESIZE);

  n = corrin.signal1.length;

  buff.length = corrin.signal1.length;
  if (! (buff.data = (REAL4*) LALMalloc(sizeof(REAL4)*buff.length)))
  {
     ABORT(status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
  }

  nby2 = n/2;
  for (i=1; i<nby2; i++)
  {
     k=n-i;
     psd = corrin.psd.data[i];
     f = (REAL8)i * corrin.df;
     if (psd && f < corrin.fCutoff) {
/*
     if (psd) {
     the following line computes output = signal1 . signal2*
*/
        r1 = corrin.signal1.data[i];
        r2 = corrin.signal2.data[i];
        i1 = corrin.signal1.data[k];
        i2 = corrin.signal2.data[k];

        buff.data[i] = (r1*r2 + i1*i2) / (0.5*psd);
        buff.data[k] = (i1*r2 - r1*i2) / (0.5*psd);
        /* reversed chirp analysis needs to simply change a sign in i2*/
        /*r1 = corrin.signal1.data[i];
        r2 = corrin.signal2.data[i];
        i1 = corrin.signal1.data[k];
        i2 = corrin.signal2.data[k];

        buff.data[i] = (r1*r2 - i1*i2) / (0.5*psd);
        buff.data[k] = (i1*r2 + r1*i2) / (0.5*psd);
        */

	/*
	 * printf("%d %e %e\n", i, buff.data[i], buff.data[k]);
	 */

     } else {

        buff.data[i] = 0;
        buff.data[k] = 0;
     }
  }
  psd = corrin.psd.data[0];
  f = 0;
  if (psd && f<corrin.fCutoff)
  {
     r1 = corrin.signal1.data[0];
     r2 = corrin.signal2.data[0];
     buff.data[0] = r1*r2 / (0.5*psd);
     /*
      * printf("%d %e %e\n", i, buff.data[0], buff.data[0]);
      */
  }
  else
  {
     buff.data[0] = 0;
  }

  psd = corrin.psd.data[nby2];
  f = nby2 * corrin.df;
  if (psd && f<corrin.fCutoff)
  {
     r1 = corrin.signal1.data[nby2];
     r2 = corrin.signal2.data[nby2];
     buff.data[nby2] = r1*r2 / (0.5*psd);
     /*
      * printf("%d %e %e\n", i, buff.data[nby2], buff.data[nby2]);
      */
  }
  else
  {
     buff.data[nby2] = 0;
  }

  if (XLALREAL4VectorFFT(output,&buff,corrin.revp) != 0)
    ABORTXLAL(status);
  for (i=0; i<n; i++) output->data[i] /= ((double) n * corrin.samplingRate);


  /*
   * printf("&\n");
   */
  LALFree(buff.data);
  DETATCHSTATUSPTR(status);
  RETURN(status);
}

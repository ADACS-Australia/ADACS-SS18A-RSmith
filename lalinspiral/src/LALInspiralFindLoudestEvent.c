/*
*  Copyright (C) 2007 David Churches, Jolien Creighton, B.S. Sathyaprakash
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

/*  <lalVerbatim file="LALInspiralFindLoudestEventCV">
Author: Sathyaprakash, B. S.
$Id$
</lalVerbatim>  */
/* <lalLaTeX>
\subsection{Module \texttt{LALInspiralFindLoudestEvent.c}}
Module to find events in a given data set with an SNR
larger than a pre-specified threshold. The module uses
two orthogonal inspiral signals of specified parameters
with a weight specified in a psd array. The code returns
the number of events found, and for each event the snr,
the bin number and the phase of the template at that bin.

\subsubsection*{Prototypes}
\vspace{0.1in}
\input{LALInspiralFindLoudestEventCP}
\index{\verb&LALInspiralFindLoudestEvent()&}

\subsubsection*{Description}
\subsubsection*{Algorithm}
\subsubsection*{Uses}
\begin{verbatim}
LALInspiralWave
LALREAL4VectorFFT
LALInspiralWaveNormalise
LALInspiralWaveCorrelate
\end{verbatim}

\subsubsection*{Notes}

\vfill{\footnotesize\input{LALInspiralFindLoudestEventCV}}
</lalLaTeX>  */
#include <lal/LALNoiseModelsInspiral.h>

NRCSID (LALINSPIRALFINDEVENTSC, "$Id$");

/*  <lalVerbatim file="LALInspiralFindLoudestEventCP"> */
void
LALInspiralFindLoudestEvent
   (
   LALStatus            *status,
   INT4                 *nEvents,
   InspiralEventsList   *eventlist,
   InspiralFindEventsIn *findeventsin
   )
{  /*  </lalVerbatim>  */

   /*
    * We shall assume that the number of events to be found in
    * each template is no more than 100000.
   */
   INT4 i;
   INT4 nBegin;
   INT4 nEnd;

   REAL8 eSec;
   REAL8 dt;
   REAL8 df;
   REAL8 f;
   REAL8 norm;
   REAL8 distanceNorm;
   REAL8 flso;
   REAL8 msevenby3;
   REAL8 x;
   REAL8 y;
   REAL8 z;

   REAL8 totalMass;
   REAL8 eta;
   REAL8 dist;
   REAL4 chisq;

   REAL4Vector filter1;
   REAL4Vector filter2;
   REAL4Vector output1;
   REAL4Vector output2;
   REAL4Vector buffer;
   REAL4Vector SNRZeroIntegrand;

   StatsREAL4VectorOut statsout1;
   StatsREAL4VectorOut statsout2;

   InspiralWaveCorrelateIn corrin;
   InspiralSNRIntegrandParams params;
   InspiralChisqDataVec chisqDataVec;
   InspiralChisqParams chisqParams;
   InspiralWaveNormaliseIn normin;

   INITSTATUS (status, "LALInspiralFindLoudestEvent", LALINSPIRALFINDEVENTSC);
   ATTATCHSTATUSPTR(status);

   ASSERT (findeventsin->psd.data,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);
   ASSERT (findeventsin->signal.data,  status, LALNOISEMODELSH_ENULL, LALNOISEMODELSH_MSGENULL);

   output1.length = output2.length = findeventsin->signal.length;
   filter1.length = filter2.length = findeventsin->signal.length;
   SNRZeroIntegrand.length = findeventsin->signal.length;

   ASSERT (findeventsin->nEnd >= 0,  status, LALNOISEMODELSH_ECHOICE, LALNOISEMODELSH_MSGECHOICE);
   ASSERT (findeventsin->nEnd <= (INT4)output1.length,  status, LALNOISEMODELSH_ECHOICE, LALNOISEMODELSH_MSGECHOICE);
   ASSERT (findeventsin->nBegin >= 0,  status, LALNOISEMODELSH_ECHOICE, LALNOISEMODELSH_MSGECHOICE);
   ASSERT (findeventsin->nBegin <= (INT4)output1.length,  status, LALNOISEMODELSH_ECHOICE, LALNOISEMODELSH_MSGECHOICE);


   dt = 1.L/findeventsin->param.tSampling;
   df = 1.L/(output1.length * dt);
   nBegin = findeventsin->nBegin;
   nEnd = findeventsin->signal.length - findeventsin->nEnd;

   buffer.length = nEnd - nBegin;
   if (!(output1.data = (REAL4*) LALMalloc(sizeof(REAL4)*output1.length))) {
      ABORT (status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
   }
   if (!(output2.data = (REAL4*) LALMalloc(sizeof(REAL4)*output2.length))) {
      LALFree(output1.data);
      output1.data = NULL;
      ABORT (status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
   }
   if (!(filter1.data = (REAL4*) LALMalloc(sizeof(REAL4)*filter1.length))) {
      LALFree(output1.data);
      LALFree(output2.data);
      output1.data = NULL;
      output2.data = NULL;
      ABORT (status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
   }
   if (!(filter2.data = (REAL4*) LALMalloc(sizeof(REAL4)*filter2.length))) {
      LALFree(output1.data);
      LALFree(output2.data);
      LALFree(filter1.data);
      output1.data = NULL;
      output2.data = NULL;
      filter1.data = NULL;
      ABORT (status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
   }
   if (!(buffer.data = (REAL4*) LALMalloc(sizeof(REAL4)*buffer.length))) {
      LALFree(output1.data);
      LALFree(output2.data);
      LALFree(filter1.data);
      LALFree(filter2.data);
      output1.data = NULL;
      output2.data = NULL;
      filter1.data = NULL;
      filter2.data = NULL;
      ABORT (status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
   }
   if (!(SNRZeroIntegrand.data = (REAL4*) LALMalloc(sizeof(REAL4)*SNRZeroIntegrand.length))) {
      LALFree(output1.data);
      LALFree(output2.data);
      LALFree(filter1.data);
      LALFree(filter2.data);
      LALFree(buffer.data);
      output1.data = NULL;
      output2.data = NULL;
      filter1.data = NULL;
      filter2.data = NULL;
      buffer.data = NULL;
      ABORT (status, LALNOISEMODELSH_EMEM, LALNOISEMODELSH_MSGEMEM);
   }

   findeventsin->param.nStartPad = 0;
   switch (findeventsin->param.approximant)
   {
	   case TaylorT1:
	   case TaylorT2:
	   case TaylorT3:
	   case PadeT1:
	   case EOB:
	   case SpinTaylorT3:

		   findeventsin->param.startPhase = LAL_PI_2;
		   LALInspiralWave(status->statusPtr, &output2, &findeventsin->param);
		   CHECKSTATUSPTR(status);
		   findeventsin->param.startPhase = 0.;
		   LALInspiralWave(status->statusPtr, &output1, &findeventsin->param);
		   CHECKSTATUSPTR(status);

		   if (findeventsin->displayTemplates)
		   {
			   for (i=0;i<(INT4)output1.length;i++)
				   printf("%e %e %e\n", i*dt, output1.data[i], output2.data[i]);printf("&\n");
		   }
		   if (XLALREAL4VectorFFT(&filter1, &output1, findeventsin->fwdp) != 0)
         ABORTXLAL(status);
		   if (XLALREAL4VectorFFT(&filter2, &output2, findeventsin->fwdp) != 0)
         ABORTXLAL(status);
		   break;

	   case TaylorF1:
	   case TaylorF2:
	   case PadeF1:
	   case BCV:

		   findeventsin->param.startPhase = LAL_PI_2;
		   LALInspiralWave(status->statusPtr, &filter2, &findeventsin->param);
		   CHECKSTATUSPTR(status);
		   findeventsin->param.startPhase = 0.;
		   LALInspiralWave(status->statusPtr, &filter1, &findeventsin->param);
		   CHECKSTATUSPTR(status);

		   if (findeventsin->displayTemplates)
		   {
			   if (XLALREAL4VectorFFT(&output1, &filter1, findeventsin->revp) != 0)
				   ABORTXLAL(status);
			   if (XLALREAL4VectorFFT(&output2, &filter2, findeventsin->revp) != 0)
				   ABORTXLAL(status);
			   for (i=0;i<(INT4)output1.length;i++)
				   printf("%e %e %e\n", i*dt, output1.data[i], output2.data[i]);printf("&\n");

			   if (XLALREAL4VectorFFT(&filter1, &output1, findeventsin->fwdp) != 0)
				   ABORTXLAL(status);
			   if (XLALREAL4VectorFFT(&filter2, &output2, findeventsin->fwdp) != 0)
				   ABORTXLAL(status);
		   }
		   break;
           default:
                   ABORT( status, 9999, "Unknown case in switch." );
   }

   normin.psd = &(findeventsin->psd);
   normin.df = df;
   normin.fCutoff = findeventsin->param.fFinal;
   normin.samplingRate = findeventsin->param.tSampling;
   LALInspiralWaveNormaliseLSO(status->statusPtr, &filter1, &norm, &normin);
   CHECKSTATUSPTR(status);
   LALInspiralWaveNormaliseLSO(status->statusPtr, &filter2, &norm, &normin);
   CHECKSTATUSPTR(status);
   corrin.psd = findeventsin->psd;
   corrin.revp = findeventsin->revp;
   corrin.signal1 = findeventsin->signal;
   corrin.fCutoff = findeventsin->param.fFinal;
   corrin.samplingRate = findeventsin->param.tSampling;

   corrin.signal2 = filter1;
   LALInspiralWaveCorrelate(status->statusPtr, &output1, corrin);
   CHECKSTATUSPTR(status);
   corrin.signal2 = filter2;
   LALInspiralWaveCorrelate(status->statusPtr, &output2, corrin);
   CHECKSTATUSPTR(status);


   for (i=nBegin;i<nEnd;i++)
	   buffer.data[i-nBegin] = output1.data[i];
   LALStatsREAL4Vector(status->statusPtr, &statsout1, &buffer);
   CHECKSTATUSPTR(status);

   for (i=nBegin;i<nEnd;i++)
	   buffer.data[i-nBegin] = output2.data[i];
   LALStatsREAL4Vector(status->statusPtr, &statsout2, &buffer);
   CHECKSTATUSPTR(status);

   if (findeventsin->displayCorrelationStats)
   {
	   fprintf(stderr, "mean=%e std=%e min=%e max=%e\n", statsout1.mean, statsout1.stddev, statsout1.min, statsout1.max);

	   fprintf(stderr, "mean=%e std=%e min=%e max=%e\n", statsout2.mean, statsout2.stddev, statsout2.min, statsout2.max);
   }

   if (findeventsin->displayCorrelation)
   {
      for (i=nBegin;i<nEnd;i++)
         {
            x = pow ( pow( output1.data[i], 2.) + pow( output2.data[i], 2.), 0.5);
            printf("%e %e\n", i*dt, x);
         }
         printf("&\n");
   }
   msevenby3 = -7.L/3.L;
   distanceNorm = 0.;
   totalMass = findeventsin->param.totalMass*LAL_MTSUN_SI;
   eta = findeventsin->param.eta;
   flso = 1.L/(pow(6.L,1.5L)*totalMass*LAL_PI);
   for (i=1; i<(INT4)findeventsin->psd.length; i++)
   {
	   f = i*df;
	   if (f > flso) break;
	   if (findeventsin->psd.data[i]) distanceNorm += pow(f,msevenby3)/findeventsin->psd.data[i];
   }
   distanceNorm = sqrt(distanceNorm);
   distanceNorm *= df * pow(totalMass, 5.L/6.L) * sqrt(5.L*eta/12.L)/pow(LAL_PI,2.L/3.L) ;
   /*
   printf("FindEvents: flso=%e mass=%e eta=%e norm=%e\n", flso, totalMass/LAL_MTSUN_SI, eta, distanceNorm);
   */

   *nEvents = 0;

   x = output1.data[nBegin];
   y = output2.data[nBegin];
   z = sqrt(x*x + y*y);
   if (z>findeventsin->Threshold) (*nEvents)++;

   eventlist->snr = z;
   eventlist->param = findeventsin->param;
   eventlist->phase = atan2(y,x);
   eventlist->bin = nBegin;

   dist = distanceNorm/ z;
   eventlist->effDistance = LAL_C_SI * dist / LAL_PC_SI /1.e6;
   eventlist->amplitude = 4.*eta*(totalMass/dist)*pow(LAL_PI*totalMass*100.,2.L/3.L);
   eventlist->bin = nBegin;
   eSec = (double) nBegin / findeventsin->param.tSampling;
   eventlist->impulseTime = findeventsin->currentGPSTime + (int) eSec;
   eventlist->impulseTimeNS = (int) (1.e9 * (eSec - (int) eSec));
   eSec += findeventsin->param.tC;
   eventlist->endTime = findeventsin->currentGPSTime + (int) eSec;
   eventlist->endTimeNS = (int) (1.e9 * (eSec - (int) eSec));
   eventlist->sigmasq = (statsout1.var + statsout2.var)/2.;

   for (i=nBegin+1; i<nEnd; i++)
   {
       x = output1.data[i];
       y = output2.data[i];
       z = sqrt(x*x + y*y);
       if (z>findeventsin->Threshold) (*nEvents)++;
       if (z>eventlist->snr)
       {
          eventlist->snr = z;
          eventlist->param = findeventsin->param;
          eventlist->phase = atan2(y,x);

          dist = distanceNorm/ z;
          eventlist->effDistance = LAL_C_SI * dist / LAL_PC_SI /1.e6;
          eventlist->amplitude = 4.*eta*(totalMass/dist)*pow(LAL_PI*totalMass*100.,2.L/3.L);
          eventlist->bin = i;

	  eSec = (double) i / findeventsin->param.tSampling;
          eventlist->impulseTime = findeventsin->currentGPSTime + (int) eSec;
          eventlist->impulseTimeNS = (int) (1.e9 * (eSec - (int) eSec));
	  eSec += findeventsin->param.tC;
          eventlist->endTime = findeventsin->currentGPSTime + (int) eSec;
          eventlist->endTimeNS = (int) (1.e9 * (eSec - (int) eSec));
          eventlist->sigmasq = (statsout1.var + statsout2.var)/2.;
       }

   }


   /* Now call the chi-squared code */

   params.lag = eventlist->bin;
   params.phase = eventlist->phase;
   params.deltaT = 1.0/findeventsin->param.tSampling;
   corrin.signal2 = filter1;
   LALInspiralComputeSNRIntegrand(status->statusPtr, &SNRZeroIntegrand, corrin, &params);

   chisqDataVec.SNRIntegrand = &SNRZeroIntegrand;
   chisqDataVec.psd = &(findeventsin->psd);

   chisqParams.totalMass = totalMass;
   chisqParams.fLower = findeventsin->param.fLower;
   chisqParams.deltaT = params.deltaT;
   chisqParams.nBins = 20;
   LALInspiralComputeChisq(status->statusPtr, &chisq, &chisqDataVec, &chisqParams);

   eventlist->chisq = chisq;
   eventlist->chisqDOF = 2*(chisqParams.nBins-1);

   LALFree(filter1.data);
   LALFree(filter2.data);
   LALFree(output1.data);
   LALFree(output2.data);
   LALFree(buffer.data);
   LALFree(SNRZeroIntegrand.data);
   DETATCHSTATUSPTR(status);
   RETURN(status);
}


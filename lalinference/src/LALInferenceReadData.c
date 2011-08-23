/* 
 *  LALInferenceReadData.c:  Bayesian Followup functions
 *
 *  Copyright (C) 2009 Ilya Mandel, Vivien Raymond, Christian Roever, Marc van der Sluys and John Veitch
 *
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

#include <stdio.h>
#include <stdlib.h>

#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>

#include <lal/LALInspiral.h>
#include <lal/FrameCache.h>
#include <lal/FrameStream.h>
#include <lal/TimeFreqFFT.h>
#include <lal/LALDetectors.h>
#include <lal/AVFactories.h>
#include <lal/ResampleTimeSeries.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/Units.h>
#include <lal/Date.h>
#include <lal/StringInput.h>
#include <lal/VectorOps.h>
#include <lal/Random.h>
#include <lal/LALNoiseModels.h>
#include <lal/XLALError.h>
#include <lal/GenerateInspiral.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LIGOLwXMLInspiralRead.h>


#include <lal/LALInference.h>
#include <lal/LALInferenceReadData.h>
#include <lal/LALInferenceLikelihood.h>

static const LALUnit strainPerCount={0,{0,0,0,0,0,1,-1},{0,0,0,0,0,0,0}};

static REAL8TimeSeries *readTseries(CHAR *cachefile, CHAR *channel, LIGOTimeGPS start, REAL8 length);
static void makeWhiteData(LALInferenceIFOData *IFOdata);

static REAL8TimeSeries *readTseries(CHAR *cachefile, CHAR *channel, LIGOTimeGPS start, REAL8 length)
{
	LALStatus status;
	memset(&status,0,sizeof(status));
	FrCache *cache = NULL;
	FrStream *stream = NULL;
	REAL8TimeSeries *out = NULL;
	
	cache  = XLALFrImportCache( cachefile );
        int err;
        err = *XLALGetErrnoPtr();
	if(cache==NULL) {fprintf(stderr,"ERROR: Unable to import cache file \"%s\",\n       XLALError: \"%s\".\n",cachefile, XLALErrorString(err)); exit(-1);}
	stream = XLALFrCacheOpen( cache );
	if(stream==NULL) {fprintf(stderr,"ERROR: Unable to open stream from frame cache file\n"); exit(-1);}
	out = XLALFrInputREAL8TimeSeries( stream, channel, &start, length , 0 );
	if(out==NULL) fprintf(stderr,"ERROR: unable to read channel %s from %s at time %i\nCheck the specified data duration is not too long\n",channel,cachefile,start.gpsSeconds);
	LALDestroyFrCache(&status,&cache);
	LALFrClose(&status,&stream);
	return out;
}
#define USAGE "\
 --IFO [IFO1,IFO2,...]          IFOs can be H1,L1,V1\n\
 --cache [cache1,cache2,...]    LAL cache files (LALLIGO, LALAdLIGO, LALVirgo to simulate these detectors)\n\
 --PSDstart GPStime             GPS start time of PSD estimation data\n\
 --PSDlength length             length of PSD estimation data in seconds\n\
 --seglen length                length of segments for PSD estimation and analysis in seconds\n\
 --trigtime GPStime             GPS time of the trigger to analyse\n\
(--srate rate)                  Downsample data to rate in Hz\n\
(--fLow [freq1,freq2,...])      Specify lower frequency cutoff for overlap integral\n\
(--fHigh [freq1,freq2,...])     Specify higher frequency cutoff for overlap integral\n\
(--channel [chan1,chan2,...])   Specify channel names when reading cache files\n\
(--dataseed number)             Specify random seed to use when generating data\n"

LALInferenceIFOData *LALInferenceReadData(ProcessParamsTable *commandLine)
/* Read in the data and store it in a LALInferenceIFOData structure */
{
	LALStatus status;
	INT4 dataseed=0;
	memset(&status,0,sizeof(status));
	ProcessParamsTable *procparam=NULL;
	LALInferenceIFOData *headIFO=NULL,*IFOdata=NULL;
	REAL8 SampleRate=4096.0,SegmentLength=0;
	if(LALInferenceGetProcParamVal(commandLine,"--srate")) SampleRate=atof(LALInferenceGetProcParamVal(commandLine,"--srate")->value);
        const REAL8 defaultFLow = 40.0;
        const REAL8 defaultFHigh = SampleRate/2.0;
	int nSegs=0;
	size_t seglen=0;
	REAL8TimeSeries *PSDtimeSeries=NULL;
	REAL8 padding=0.4;//Default was 1.0 second. However for The Event the Common Inputs specify a Tukey parameter of 0.1, so 0.4 second of padding for 8 seconds of data.
	UINT4 Ncache=0,Nifo=0,Nchannel=0,NfLow=0,NfHigh=0;
	UINT4 i,j;
	//int FakeFlag=0; - set but not used
	char strainname[]="LSC-STRAIN";
	
	typedef void (NoiseFunc)(LALStatus *statusPtr,REAL8 *psd,REAL8 f);
	NoiseFunc *PSD=NULL;
	REAL8 scalefactor=1;

	RandomParams *datarandparam;

	char *chartmp=NULL;
	char **channels=NULL;
	char **caches=NULL;
	char **IFOnames=NULL;
	char **fLows=NULL,**fHighs=NULL;
	LIGOTimeGPS GPSstart,GPStrig,segStart;
	REAL8 PSDdatalength=0;

	if(!LALInferenceGetProcParamVal(commandLine,"--cache")||!LALInferenceGetProcParamVal(commandLine,"--IFO")||
	   !LALInferenceGetProcParamVal(commandLine,"--PSDstart")||!LALInferenceGetProcParamVal(commandLine,"--trigtime")||
	   !LALInferenceGetProcParamVal(commandLine,"--PSDlength")||!LALInferenceGetProcParamVal(commandLine,"--seglen"))
	{fprintf(stderr,USAGE); return(NULL);}
	
  //TEMPORARY. JUST FOR CHECKING USING SPINSPIRAL PSD
  char **spinspiralPSD=NULL;
  UINT4 NspinspiralPSD = 0;
  if (LALInferenceGetProcParamVal(commandLine, "--spinspiralPSD")) {
    LALInferenceParseCharacterOptionString(LALInferenceGetProcParamVal(commandLine,"--spinspiralPSD")->value,&spinspiralPSD,&NspinspiralPSD);
  }    
  
	if(LALInferenceGetProcParamVal(commandLine,"--channel")){
		LALInferenceParseCharacterOptionString(LALInferenceGetProcParamVal(commandLine,"--channel")->value,&channels,&Nchannel);
	}
	LALInferenceParseCharacterOptionString(LALInferenceGetProcParamVal(commandLine,"--cache")->value,&caches,&Ncache);
	LALInferenceParseCharacterOptionString(LALInferenceGetProcParamVal(commandLine,"--IFO")->value,&IFOnames,&Nifo);
	if(LALInferenceGetProcParamVal(commandLine,"--fLow")){
		LALInferenceParseCharacterOptionString(LALInferenceGetProcParamVal(commandLine,"--fLow")->value,&fLows,&NfLow);
	}
	if(LALInferenceGetProcParamVal(commandLine,"--fHigh")){
		LALInferenceParseCharacterOptionString(LALInferenceGetProcParamVal(commandLine,"--fHigh")->value,&fHighs,&NfHigh);
	}
	if(LALInferenceGetProcParamVal(commandLine,"--dataseed")){
		procparam=LALInferenceGetProcParamVal(commandLine,"--dataseed");
		dataseed=atoi(procparam->value);
	}
								   
	if(Nifo!=Ncache) {fprintf(stderr,"ERROR: Must specify equal number of IFOs and Cache files\n"); exit(1);}
	if(Nchannel!=0 && Nchannel!=Nifo) {fprintf(stderr,"ERROR: Please specify a channel for all caches, or omit to use the defaults\n"); exit(1);}
	
	IFOdata=headIFO=calloc(sizeof(LALInferenceIFOData),Nifo);
	
	procparam=LALInferenceGetProcParamVal(commandLine,"--PSDstart");
	LALStringToGPS(&status,&GPSstart,procparam->value,&chartmp);
	procparam=LALInferenceGetProcParamVal(commandLine,"--trigtime");
	LALStringToGPS(&status,&GPStrig,procparam->value,&chartmp);
	PSDdatalength=atof(LALInferenceGetProcParamVal(commandLine,"--PSDlength")->value);
	SegmentLength=atof(LALInferenceGetProcParamVal(commandLine,"--seglen")->value);
	seglen=(size_t)(SegmentLength*SampleRate);
	nSegs=(int)floor(PSDdatalength/SegmentLength);
	
	for(i=0;i<Nifo;i++) {
          IFOdata[i].fLow=fLows?atof(fLows[i]):defaultFLow; 
          IFOdata[i].fHigh=fHighs?atof(fHighs[i]):defaultFHigh;
          strncpy(IFOdata[i].name, IFOnames[i], DETNAMELEN);
        }

	/* Only allocate this array if there weren't channels read in from the command line */
	if(!Nchannel) channels=calloc(Nifo,sizeof(char *));
	for(i=0;i<Nifo;i++) {
		if(!Nchannel) channels[i]=malloc(VARNAME_MAX);
		IFOdata[i].detector=calloc(1,sizeof(LALDetector));
		
		if(!strcmp(IFOnames[i],"H1")) {			
			memcpy(IFOdata[i].detector,&lalCachedDetectors[LALDetectorIndexLHODIFF],sizeof(LALDetector));
			if(!Nchannel) sprintf((channels[i]),"H1:%s",strainname); continue;}
		if(!strcmp(IFOnames[i],"H2")) {
			memcpy(IFOdata[i].detector,&lalCachedDetectors[LALDetectorIndexLHODIFF],sizeof(LALDetector));
			if(!Nchannel) sprintf((channels[i]),"H2:%s",strainname); continue;}
		if(!strcmp(IFOnames[i],"LLO")||!strcmp(IFOnames[i],"L1")) {
			memcpy(IFOdata[i].detector,&lalCachedDetectors[LALDetectorIndexLLODIFF],sizeof(LALDetector));
			if(!Nchannel) sprintf((channels[i]),"L1:%s",strainname); continue;}
		if(!strcmp(IFOnames[i],"V1")||!strcmp(IFOnames[i],"VIRGO")) {
			memcpy(IFOdata[i].detector,&lalCachedDetectors[LALDetectorIndexVIRGODIFF],sizeof(LALDetector));
			if(!Nchannel) sprintf((channels[i]),"V1:h_16384Hz"); continue;}
		if(!strcmp(IFOnames[i],"GEO")||!strcmp(IFOnames[i],"G1")) {
			memcpy(IFOdata[i].detector,&lalCachedDetectors[LALDetectorIndexGEO600DIFF],sizeof(LALDetector));
			if(!Nchannel) sprintf((channels[i]),"G1:DER_DATA_H"); continue;}
		/*		if(!strcmp(IFOnames[i],"TAMA")||!strcmp(IFOnames[i],"T1")) {inputMCMC.detector[i]=&lalCachedDetectors[LALDetectorIndexTAMA300DIFF]; continue;}*/
		fprintf(stderr,"Unknown interferometer %s. Valid codes: H1 H2 L1 V1 GEO\n",IFOnames[i]); exit(-1);
	}
	
	/* Set up FFT structures and window */
	for (i=0;i<Nifo;i++){
		/* Create FFT plans */
		IFOdata[i].timeToFreqFFTPlan = XLALCreateForwardREAL8FFTPlan((UINT4) seglen, 0 );
		IFOdata[i].freqToTimeFFTPlan = XLALCreateReverseREAL8FFTPlan((UINT4) seglen,0);
		
		/* Setup windows */
		IFOdata[i].window=XLALCreateTukeyREAL8Window(seglen,(REAL8)2.0*padding*SampleRate/(REAL8)seglen);
	}
	
	
	/* Trigger time = 2 seconds before end of segment (was 1 second, but Common Inputs for The Events are -6 +2*/
	memcpy(&segStart,&GPStrig,sizeof(LIGOTimeGPS));
	XLALGPSAdd(&segStart,-SegmentLength+2);
	
	
	/* Read the PSD data */
	for(i=0;i<Nifo;i++) {
		memcpy(&(IFOdata[i].epoch),&segStart,sizeof(LIGOTimeGPS));
		/* Check if fake data is requested */
		if(!(strcmp(caches[i],"LALLIGO") && strcmp(caches[i],"LALVirgo") && strcmp(caches[i],"LALGEO") && strcmp(caches[i],"LALEGO")
			 && strcmp(caches[i],"LALAdLIGO")))
		{
			//FakeFlag=1; - set but not used
			datarandparam=XLALCreateRandomParams(dataseed?dataseed+(int)i:dataseed);
			/* Selection of the noise curve */
			if(!strcmp(caches[i],"LALLIGO")) {PSD = &LALLIGOIPsd; scalefactor=9E-46;}
			if(!strcmp(caches[i],"LALVirgo")) {PSD = &LALVIRGOPsd; scalefactor=1.0;}
			if(!strcmp(caches[i],"LALGEO")) {PSD = &LALGEOPsd; scalefactor=1E-46;}
			if(!strcmp(caches[i],"LALEGO")) {PSD = &LALEGOPsd; scalefactor=1.0;}
			if(!strcmp(caches[i],"LALAdLIGO")) {PSD = &LALAdvLIGOPsd; scalefactor = 10E-49;}
			//if(!strcmp(caches[i],"LAL2kLIGO")) {PSD = &LALAdvLIGOPsd; scalefactor = 36E-46;}
			if(PSD==NULL) {fprintf(stderr,"Error: unknown simulated PSD: %s\n",caches[i]); exit(-1);}
			IFOdata[i].oneSidedNoisePowerSpectrum=(REAL8FrequencySeries *)
						XLALCreateREAL8FrequencySeries("spectrum",&GPSstart,0.0,
										   (REAL8)(SampleRate)/seglen,&lalDimensionlessUnit,seglen/2 +1);
			for(j=0;j<IFOdata[i].oneSidedNoisePowerSpectrum->data->length;j++)
			{
				PSD(&status,&(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]),j*IFOdata[i].oneSidedNoisePowerSpectrum->deltaF);
				IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]*=scalefactor;
			}
			IFOdata[i].freqData = (COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("stilde",&segStart,0.0,IFOdata[i].oneSidedNoisePowerSpectrum->deltaF,&lalDimensionlessUnit,seglen/2 +1);
			/* Create the fake data */
			int j_Lo = (int) IFOdata[i].fLow/IFOdata[i].freqData->deltaF;
			for(j=j_Lo;j<IFOdata[i].freqData->data->length;j++){
				IFOdata[i].freqData->data->data[j].re=XLALNormalDeviate(datarandparam)*(0.5*sqrt(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]/IFOdata[i].freqData->deltaF));
				IFOdata[i].freqData->data->data[j].im=XLALNormalDeviate(datarandparam)*(0.5*sqrt(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]/IFOdata[i].freqData->deltaF));
			}
			IFOdata[i].freqData->data->data[0].re=0; 			IFOdata[i].freqData->data->data[0].im=0;
			const char timename[]="timeData";
			IFOdata[i].timeData=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries(timename,&segStart,0.0,(REAL8)1.0/SampleRate,&lalDimensionlessUnit,(size_t)seglen);
			XLALREAL8FreqTimeFFT(IFOdata[i].timeData,IFOdata[i].freqData,IFOdata[i].freqToTimeFFTPlan);
			if(*XLALGetErrnoPtr()) printf("XLErr: %s\n",XLALErrorString(*XLALGetErrnoPtr()));
			XLALDestroyRandomParams(datarandparam);

		}
		else{
			fprintf(stderr,"Estimating PSD for %s using %i segments of %i samples (%lfs)\n",IFOnames[i],nSegs,(int)seglen,SegmentLength);
			
			PSDtimeSeries=readTseries(caches[i],channels[i],GPSstart,PSDdatalength);
			if(!PSDtimeSeries) {fprintf(stderr,"Error reading PSD data for %s\n",IFOnames[i]); exit(1);}
			XLALResampleREAL8TimeSeries(PSDtimeSeries,1.0/SampleRate);
			PSDtimeSeries=(REAL8TimeSeries *)XLALShrinkREAL8TimeSeries(PSDtimeSeries,(size_t) 0, (size_t) seglen*nSegs);
			IFOdata[i].oneSidedNoisePowerSpectrum=(REAL8FrequencySeries *)XLALCreateREAL8FrequencySeries("spectrum",&PSDtimeSeries->epoch,0.0,(REAL8)(SampleRate)/seglen,&lalDimensionlessUnit,seglen/2 +1);
			if (LALInferenceGetProcParamVal(commandLine, "--PSDwelch")) {
        XLALREAL8AverageSpectrumWelch(IFOdata[i].oneSidedNoisePowerSpectrum ,PSDtimeSeries, seglen, (UINT4)seglen, IFOdata[i].window, IFOdata[i].timeToFreqFFTPlan);
      }
      else {
        XLALREAL8AverageSpectrumMedian(IFOdata[i].oneSidedNoisePowerSpectrum ,PSDtimeSeries, seglen, (UINT4)seglen, IFOdata[i].window, IFOdata[i].timeToFreqFFTPlan);	
			}
        XLALDestroyREAL8TimeSeries(PSDtimeSeries);
			
			/* Read the data segment */
			IFOdata[i].timeData=readTseries(caches[i],channels[i],segStart,SegmentLength);

                        /* FILE *out; */
                        /* char fileName[256]; */
                        /* snprintf(fileName, 256, "readTimeData-%d.dat", i); */
                        /* out = fopen(fileName, "w"); */
                        /* for (j = 0; j < IFOdata[i].timeData->data->length; j++) { */
                        /*   fprintf(out, "%g %g\n", j*IFOdata[i].timeData->deltaT, IFOdata[i].timeData->data->data[j]); */
                        /* } */
                        /* fclose(out); */
                        
			if(!IFOdata[i].timeData) {fprintf(stderr,"Error reading segment data for %s at %i\n",IFOnames[i],segStart.gpsSeconds); exit(1);}
			XLALResampleREAL8TimeSeries(IFOdata[i].timeData,1.0/SampleRate);	 
			if(!IFOdata[i].timeData) {fprintf(stderr,"Error reading segment data for %s\n",IFOnames[i]); exit(1);}
			IFOdata[i].freqData=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("freqData",&(IFOdata[i].timeData->epoch),0.0,1.0/SegmentLength,&lalDimensionlessUnit,seglen/2+1);
			IFOdata[i].windowedTimeData=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("windowed time data",&(IFOdata[i].timeData->epoch),0.0,1.0/SampleRate,&lalDimensionlessUnit,seglen);
			XLALDDVectorMultiply(IFOdata[i].windowedTimeData->data,IFOdata[i].timeData->data,IFOdata[i].window->data);
			XLALREAL8TimeFreqFFT(IFOdata[i].freqData,IFOdata[i].windowedTimeData,IFOdata[i].timeToFreqFFTPlan);
			
			for(j=0;j<IFOdata[i].freqData->data->length;j++){
				IFOdata[i].freqData->data->data[j].re/=sqrt(IFOdata[i].window->sumofsquares / IFOdata[i].window->data->length);
				IFOdata[i].freqData->data->data[j].im/=sqrt(IFOdata[i].window->sumofsquares / IFOdata[i].window->data->length);
                                IFOdata[i].windowedTimeData->data->data[j] /= sqrt(IFOdata[i].window->sumofsquares / IFOdata[i].window->data->length);
			}
			
		}
                /* Now that the PSD is set up, make the TDW. */
                IFOdata[i].timeDomainNoiseWeights = 
                  (REAL8TimeSeries *)XLALCreateREAL8TimeSeries("time domain weights", 
                                                               &(IFOdata[i].oneSidedNoisePowerSpectrum->epoch),
                                                               0.0,
                                                               1.0/SampleRate,
                                                               &lalDimensionlessUnit,
                                                               seglen);
                LALInferencePSDToTDW(IFOdata[i].timeDomainNoiseWeights, IFOdata[i].oneSidedNoisePowerSpectrum, IFOdata[i].freqToTimeFFTPlan,
                         IFOdata[i].fLow, IFOdata[i].fHigh);

                makeWhiteData(&(IFOdata[i]));
    
    if (LALInferenceGetProcParamVal(commandLine, "--spinspiralPSD")) {
      FILE *in;
      //char fileNameIn[256];
      //snprintf(fileNameIn, 256, spinspiralPSD);
      double freq_temp, psd_temp, temp;
      int n=0;
      int k=0;
      int templen=0;
      char buffer[256];
      char * line=buffer;
    
      //in = fopen(fileNameIn, "r");
      in = fopen(spinspiralPSD[i], "r");
      while(fgets(buffer, 256, in)){
        templen++;
      }
    
     // REAL8 *tempPSD = NULL;
     // REAL8 *tempfreq = NULL;
     // tempPSD=calloc(sizeof(REAL8),templen+1);
     // tempfreq=calloc(sizeof(REAL8),templen+1);
    
      rewind(in);
      IFOdata[i].oneSidedNoisePowerSpectrum->data->data[0] = 1.0;
      while(fgets(buffer, 256, in)){
        line=buffer;
      
        sscanf(line, "%lg%n", &freq_temp,&n);
        line+=n;
        sscanf(line, "%lg%n", &psd_temp,&n);
        line+=n;
        sscanf(line, "%lg%n", &temp,&n);
        line+=n;
      
     // tempfreq[k]=freq_temp;
     // tempPSD[k]=psd_temp*psd_temp;
        
        IFOdata[i].oneSidedNoisePowerSpectrum->data->data[k+1]=psd_temp*psd_temp;
        
      k++;
      //fprintf(stdout, "%g %g \n",freq_temp, psd_temp); fflush(stdout);
      }
      fclose(in);
    }
                
                if (LALInferenceGetProcParamVal(commandLine, "--data-dump")) {
                  const UINT4 nameLength=256;
                  char filename[nameLength];
                  FILE *out;

                  snprintf(filename, nameLength, "%s-PSD.dat", IFOdata[i].name);
                  out = fopen(filename, "w");
                  for (j = 0; j < IFOdata[i].oneSidedNoisePowerSpectrum->data->length; j++) {
                    REAL8 f = IFOdata[i].oneSidedNoisePowerSpectrum->deltaF*j;
                    REAL8 psd = IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j];

                    fprintf(out, "%g %g\n", f, psd);
                  }
                  fclose(out);

                  snprintf(filename, nameLength, "%s-timeData.dat", IFOdata[i].name);
                  out = fopen(filename, "w");
                  for (j = 0; j < IFOdata[i].timeData->data->length; j++) {
                    REAL8 t = XLALGPSGetREAL8(&(IFOdata[i].timeData->epoch)) + 
                      j * IFOdata[i].timeData->deltaT;
                    REAL8 d = IFOdata[i].timeData->data->data[j];

                    fprintf(out, "%.6f %g\n", t, d);
                  }
                  fclose(out);

                  snprintf(filename, nameLength, "%s-freqData.dat", IFOdata[i].name);
                  out = fopen(filename, "w");
                  for (j = 0; j < IFOdata[i].freqData->data->length; j++) {
                    REAL8 f = IFOdata[i].freqData->deltaF * j;
                    REAL8 dre = IFOdata[i].freqData->data->data[j].re;
                    REAL8 dim = IFOdata[i].freqData->data->data[j].im;

                    fprintf(out, "%g %g %g\n", f, dre, dim);
                  }
                  fclose(out);
                  
                }

	}
	
	for (i=0;i<Nifo-1;i++) IFOdata[i].next=&(IFOdata[i+1]);
	
	for(i=0;i<Nifo;i++) {
		if(channels) if(channels[i]) free(channels[i]);
		if(caches) if(caches[i]) free(caches[i]);
		if(IFOnames) if(IFOnames[i]) free(IFOnames[i]);
		if(fLows) if(fLows[i]) free(fLows[i]);
		if(fHighs) if(fHighs[i]) free(fHighs[i]);
	}
	if(channels) free(channels);
	if(caches) free(caches);
	if(IFOnames) free(IFOnames);
	if(fLows) free(fLows);
	if(fHighs) free(fHighs);
	
	return headIFO;
}

static void makeWhiteData(LALInferenceIFOData *IFOdata) {
  REAL8 deltaF = IFOdata->freqData->deltaF;
  REAL8 deltaT = IFOdata->timeData->deltaT;

  IFOdata->whiteFreqData = 
    XLALCreateCOMPLEX16FrequencySeries("whitened frequency data", 
                                       &(IFOdata->freqData->epoch),
                                       0.0,
                                       deltaF,
                                       &lalDimensionlessUnit,
                                       IFOdata->freqData->data->length);
  IFOdata->whiteTimeData = 
    XLALCreateREAL8TimeSeries("whitened time data",
                              &(IFOdata->timeData->epoch),
                              0.0,
                              deltaT,
                              &lalDimensionlessUnit,
                              IFOdata->timeData->data->length);


  REAL8 iLow = IFOdata->fLow / deltaF;
  REAL8 iHighDefaultCut = 0.95 * IFOdata->freqData->data->length;
  REAL8 iHighFromFHigh = IFOdata->fHigh / deltaF;
  REAL8 iHigh = (iHighDefaultCut < iHighFromFHigh ? iHighDefaultCut : iHighFromFHigh);
  REAL8 windowSquareSum = 0.0;

  UINT4 i;

  for (i = 0; i < IFOdata->freqData->data->length; i++) {
    IFOdata->whiteFreqData->data->data[i].re = IFOdata->freqData->data->data[i].re / IFOdata->oneSidedNoisePowerSpectrum->data->data[i];
    IFOdata->whiteFreqData->data->data[i].im = IFOdata->freqData->data->data[i].im / IFOdata->oneSidedNoisePowerSpectrum->data->data[i];

    if (i == 0) {
      /* Cut off the average trend in the data. */
      IFOdata->whiteFreqData->data->data[i].re = 0.0;
      IFOdata->whiteFreqData->data->data[i].im = 0.0;
    }
    if (i <= iLow) {
      /* Need to taper to implement the fLow cutoff.  Tukey window
         that starts at zero, and reaches 100% at fLow. */
      REAL8 weight = 0.5*(1.0 + cos(M_PI*(i-iLow)/iLow)); /* Starts at -Pi, runs to zero at iLow. */

      IFOdata->whiteFreqData->data->data[i].re *= weight;
      IFOdata->whiteFreqData->data->data[i].im *= weight;

      windowSquareSum += weight*weight;
    } else if (i >= iHigh) {
      /* Also taper at high freq end, Tukey window that starts at 100%
         at fHigh, then drops to zero at Nyquist.  Except that we
         always taper at least 5% of the data at high freq to avoid a
         sharp edge in freq space there. */
      REAL8 NWind = IFOdata->whiteFreqData->data->length - iHigh;
      REAL8 weight = 0.5*(1.0 + cos(M_PI*(i-iHigh)/NWind)); /* Starts at 0, runs to Pi at i = length */

      IFOdata->whiteFreqData->data->data[i].re *= weight;
      IFOdata->whiteFreqData->data->data[i].im *= weight;

      windowSquareSum += weight*weight;
    } else {
      windowSquareSum += 1.0;
    }
  }

  REAL8 norm = sqrt(IFOdata->whiteFreqData->data->length / windowSquareSum);
  for (i = 0; i < IFOdata->whiteFreqData->data->length; i++) {
    IFOdata->whiteFreqData->data->data[i].re *= norm;
    IFOdata->whiteFreqData->data->data[i].im *= norm;
  }

  XLALREAL8FreqTimeFFT(IFOdata->whiteTimeData, IFOdata->whiteFreqData, IFOdata->freqToTimeFFTPlan);
}

void LALInferenceInjectInspiralSignal(LALInferenceIFOData *IFOdata, ProcessParamsTable *commandLine)
{
	LALStatus status;
	memset(&status,0,sizeof(status));
	SimInspiralTable *injTable=NULL;
  SimInspiralTable *injEvent=NULL;
	UINT4 Ninj=0;
	UINT4 event=0;
	UINT4 i=0,j=0;
	//CoherentGW InjectGW;
	//PPNParamStruc InjParams;
	LIGOTimeGPS injstart;
	REAL8 SNR=0,NetworkSNR=0;
	DetectorResponse det;
	memset(&injstart,0,sizeof(LIGOTimeGPS));
	//memset(&InjParams,0,sizeof(PPNParamStruc));
	COMPLEX16FrequencySeries *injF=NULL;
	LALInferenceIFOData *thisData=IFOdata->next;
	REAL8 minFlow=IFOdata->fLow;
	REAL8 MindeltaT=IFOdata->timeData->deltaT;
	REAL4TimeSeries *injectionBuffer=NULL;
  REAL8 padding=0.4; //default, set in LALInferenceReadData()
  
	while(thisData){
          minFlow   = minFlow>thisData->fLow ? thisData->fLow : minFlow;
          MindeltaT = MindeltaT>thisData->timeData->deltaT ? thisData->timeData->deltaT : MindeltaT;
          thisData  = thisData->next;
	}
	//InjParams.deltaT = MindeltaT;
	//InjParams.fStartIn=(REAL4)minFlow;
	
	if(!LALInferenceGetProcParamVal(commandLine,"--injXML")) {fprintf(stdout,"No injection file specified, not injecting\n"); return;}
	if(LALInferenceGetProcParamVal(commandLine,"--event")){
    event= atoi(LALInferenceGetProcParamVal(commandLine,"--event")->value);
    fprintf(stdout,"Injecting event %d\n",event);
	}
	Ninj=SimInspiralTableFromLIGOLw(&injTable,LALInferenceGetProcParamVal(commandLine,"--injXML")->value,0,0);
	REPORTSTATUS(&status);
	printf("Ninj %d\n", Ninj);
	if(Ninj<event) fprintf(stderr,"Error reading event %d from %s\n",event,LALInferenceGetProcParamVal(commandLine,"--injXML")->value);
	while(i<event) {i++; injTable = injTable->next;} /* Select event */
  if(LALInferenceGetProcParamVal(commandLine,"--event")){ 
    injEvent = injTable;
    injEvent->next = NULL;
  }
	//memset(&InjectGW,0,sizeof(InjectGW));
	Approximant injapprox;
	LALGetApproximantFromString(&status,injTable->waveform,&injapprox);
    printf("Injecting approximant %s\n", injTable->waveform);
	REPORTSTATUS(&status);
	printf("Approximant %x\n", injapprox);
	//LALGenerateInspiral(&status,&InjectGW,injTable,&InjParams);
	//if(status.statusCode!=0) {fprintf(stderr,"Error generating injection!\n"); REPORTSTATUS(&status); }
		
	/* Begin loop over interferometers */
	while(IFOdata){
		memset(&det,0,sizeof(det));
		det.site=IFOdata->detector;
		COMPLEX8FrequencySeries *resp = XLALCreateCOMPLEX8FrequencySeries("response",&IFOdata->timeData->epoch,
																		  0.0,
																		  IFOdata->freqData->deltaF,
																		  &strainPerCount,
																		  IFOdata->freqData->data->length);
		
		for(i=0;i<resp->data->length;i++) {resp->data->data[i].re=(REAL4)1.0; resp->data->data[i].im=0.0;}
		/* Originally created for injecting into DARM-ERR, so transfer function was needed.  
		But since we are injecting into h(t), the transfer function from h(t) to h(t) is 1.*/

		/* We need a long buffer to inject into so that FindChirpInjectSignals() works properly
		 for low mass systems. Use 100 seconds here */
		REAL8 bufferLength = 100.0;
		UINT4 bufferN = (UINT4) (bufferLength/IFOdata->timeData->deltaT);
		LIGOTimeGPS bufferStart;
		memcpy(&bufferStart,&IFOdata->timeData->epoch,sizeof(LIGOTimeGPS));
		XLALGPSAdd(&bufferStart,(REAL8) IFOdata->timeData->data->length * IFOdata->timeData->deltaT);
		XLALGPSAdd(&bufferStart,-bufferLength);
		injectionBuffer=(REAL4TimeSeries *)XLALCreateREAL4TimeSeries(IFOdata->detector->frDetector.prefix,
																	 &bufferStart, 0.0, IFOdata->timeData->deltaT,
																	 &lalADCCountUnit, bufferN);
		/* This marks the sample in which the real segment starts, within the buffer */
		INT4 realStartSample=(INT4)((IFOdata->timeData->epoch.gpsSeconds - injectionBuffer->epoch.gpsSeconds)/IFOdata->timeData->deltaT);
		realStartSample+=(INT4)((IFOdata->timeData->epoch.gpsNanoSeconds - injectionBuffer->epoch.gpsNanoSeconds)*1e-9/IFOdata->timeData->deltaT);

		/*LALSimulateCoherentGW(&status,injWave,&InjectGW,&det);*/
    if(LALInferenceGetProcParamVal(commandLine,"--event")) LALFindChirpInjectSignals(&status,injectionBuffer,injEvent,resp);
		else LALFindChirpInjectSignals(&status,injectionBuffer,injTable,resp);
		if(status.statusCode) REPORTSTATUS(&status);

		XLALDestroyCOMPLEX8FrequencySeries(resp);

		
    /* Checking the lenght of the injection waveform with respect of IFOdata->timeData->data->length */
    CoherentGW            waveform;
    PPNParamStruc         ppnParams;
    memset( &waveform, 0, sizeof(CoherentGW) );
    memset( &ppnParams, 0, sizeof(PPNParamStruc) );
    ppnParams.deltaT   = IFOdata->timeData->deltaT;
    ppnParams.lengthIn = 0;
    ppnParams.ppn      = NULL;
    unsigned lengthTest = 0;
    
    LALGenerateInspiral(&status, &waveform, injEvent, &ppnParams ); //Recompute the waveform just to get access to ppnParams.tc and waveform.h->data->length or waveform.phi->data->length
    
    if(waveform.h){lengthTest = waveform.h->data->length;}
    if(waveform.phi){lengthTest = waveform.phi->data->length;}
    
    if(lengthTest>IFOdata->timeData->data->length-(UINT4)ceil((2.0*padding+2.0)/IFOdata->timeData->deltaT)){
      fprintf(stderr, "WARNING: waveform length = %u is longer than IFOdata->timeData->data->length = %d minus the window width = %d and the 2.0 seconds after tc (total of %d points available).\n", lengthTest, IFOdata->timeData->data->length, (INT4)ceil((2.0*padding)/IFOdata->timeData->deltaT) , IFOdata->timeData->data->length-(INT4)ceil((2.0*padding+2.0)/IFOdata->timeData->deltaT));
      fprintf(stderr, "The waveform injected is %f seconds long. Consider increasing the %f seconds segment length (--seglen) to be greater than %f. (in %s, line %d)\n",ppnParams.tc , IFOdata->timeData->data->length * IFOdata->timeData->deltaT, ppnParams.tc + 2.0*padding + 2.0, __FILE__, __LINE__);
    }
    
		/* Now we cut the injection buffer down to match the time domain wave size */
		injectionBuffer=(REAL4TimeSeries *)XLALCutREAL4TimeSeries(injectionBuffer,realStartSample,IFOdata->timeData->data->length);
		
		if(status.statusCode) REPORTSTATUS(&status);
/*		for(j=0;j<injWave->data->length;j++) printf("%f\n",injWave->data->data[j]);*/
		REAL8TimeSeries *inj8Wave=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("injection8",
																			  &IFOdata->timeData->epoch,
																			  0.0,
																			  IFOdata->timeData->deltaT,
																			  &lalDimensionlessUnit,
																			  IFOdata->timeData->data->length);
		for(i=0;i<injectionBuffer->data->length;i++) inj8Wave->data->data[i]=(REAL8)injectionBuffer->data->data[i];
		XLALDestroyREAL4TimeSeries(injectionBuffer);
		injF=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("injF",
																			&IFOdata->timeData->epoch,
																			0.0,
																			IFOdata->freqData->deltaF,
																			&lalDimensionlessUnit,
																			IFOdata->freqData->data->length);
		/* Window the data */
		REAL4 WinNorm = sqrt(IFOdata->window->sumofsquares/IFOdata->window->data->length);
		for(j=0;j<inj8Wave->data->length;j++) inj8Wave->data->data[j]*=IFOdata->window->data->data[j]/WinNorm;
		XLALREAL8TimeFreqFFT(injF,inj8Wave,IFOdata->timeToFreqFFTPlan);
/*		for(j=0;j<injF->data->length;j++) printf("%lf\n",injF->data->data[j].re);*/
		if(IFOdata->oneSidedNoisePowerSpectrum){
			for(SNR=0.0,j=IFOdata->fLow/injF->deltaF;j<injF->data->length;j++){
				SNR+=pow(injF->data->data[j].re,2.0)/IFOdata->oneSidedNoisePowerSpectrum->data->data[j];
				SNR+=pow(injF->data->data[j].im,2.0)/IFOdata->oneSidedNoisePowerSpectrum->data->data[j];
			}
		}
		NetworkSNR+=SNR;
		
		/* Actually inject the waveform */
		for(j=0;j<inj8Wave->data->length;j++) IFOdata->timeData->data->data[j]+=inj8Wave->data->data[j];

FILE* file=fopen("InjSignal.dat", "w");
//FILE* file2=fopen("Noise.dat", "w");
		for(j=0;j<injF->data->length;j++){
//fprintf(file2, "%lg %lg \t %lg\n", IFOdata->freqData->deltaF*j, IFOdata->freqData->data->data[j].re, IFOdata->freqData->data->data[j].im);

			IFOdata->freqData->data->data[j].re+=injF->data->data[j].re;
			IFOdata->freqData->data->data[j].im+=injF->data->data[j].im;
fprintf(file, "%lg %lg \t %lg\n", IFOdata->freqData->deltaF*j, injF->data->data[j].re, injF->data->data[j].im);
		}
		fprintf(stdout,"Injected SNR in detector %s = %g\n",IFOdata->detector->frDetector.name,sqrt(SNR));
fclose(file);		
//fclose(file2);
		
    char filename[256];
    sprintf(filename,"%s_time.dat",IFOdata->detector->frDetector.name);
    file=fopen(filename, "w");
		for(j=0;j<inj8Wave->data->length;j++){   
      fprintf(file, "%.6f\t%lg\n", XLALGPSGetREAL8(&IFOdata->timeData->epoch) + IFOdata->timeData->deltaT*j, inj8Wave->data->data[j]);
		}
    fclose(file);
		
		XLALDestroyREAL8TimeSeries(inj8Wave);
		XLALDestroyCOMPLEX16FrequencySeries(injF);
		IFOdata=IFOdata->next;
	}
	NetworkSNR=sqrt(NetworkSNR);
	REPORTSTATUS(&status);

	fprintf(stdout,"Network SNR of event %d = %g\n",event,NetworkSNR);
	return;
}


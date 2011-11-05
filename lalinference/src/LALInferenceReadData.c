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
(--srate rate)                  Downsample data to rate in Hz (4096.0,)\n\
(--fLow [freq1,freq2,...])      Specify lower frequency cutoff for overlap integral (40.0)\n\
(--fHigh [freq1,freq2,...])     Specify higher frequency cutoff for overlap integral (2048.0)\n\
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
	UINT4 q=0;	
	typedef void (NoiseFunc)(LALStatus *statusPtr,REAL8 *psd,REAL8 f);
	NoiseFunc *PSD=NULL;
	REAL8 scalefactor=1;
	SimInspiralTable *injTable=NULL;
	RandomParams *datarandparam;
	UINT4 event=0;
	char *chartmp=NULL;
	char **channels=NULL;
	char **caches=NULL;
	char **IFOnames=NULL;
	char **fLows=NULL,**fHighs=NULL;
	LIGOTimeGPS GPSstart,GPStrig,segStart;
	REAL8 PSDdatalength=0;
	REAL8 trigtime=0;
  REAL8 AIGOang=0.0; //orientation angle for the proposed Australian detector.
  if(LALInferenceGetProcParamVal(commandLine,"--AIGOang")) {
    procparam=LALInferenceGetProcParamVal(commandLine,"--AIGOang");
    AIGOang=atof(procparam->value)*LAL_PI/180.0;
  }
	if(!LALInferenceGetProcParamVal(commandLine,"--cache")||!LALInferenceGetProcParamVal(commandLine,"--IFO")||
	   !LALInferenceGetProcParamVal(commandLine,"--PSDstart")||//!LALInferenceGetProcParamVal(commandLine,"--trigtime") ||
	   !LALInferenceGetProcParamVal(commandLine,"--PSDlength")||!LALInferenceGetProcParamVal(commandLine,"--seglen"))
	{fprintf(stderr,USAGE); return(NULL);}
	
  /* ET detectors */
	LALDetector dE1,dE2,dE3;
  /* response of the detectors */
  dE1.type = dE2.type = dE3.type = LALDETECTORTYPE_IFODIFF;
  dE1.location[0] = dE2.location[0] = dE3.location[0] = 4.5464e6;
  dE1.location[1] = dE2.location[1] = dE3.location[1] = 8.4299e5;
  dE1.location[2] = dE2.location[2] = dE3.location[2] = 4.3786e6;
  LALFrDetector dE1Fr;
  sprintf(dE1Fr.name,"ET-1");
  sprintf(dE1Fr.prefix,"E1");
  dE1.frDetector=dE1Fr;
  dE1.response[0][0] = 0.1666;
  dE1.response[1][1] = -0.2484;
  dE1.response[2][2] = 0.0818;
  dE1.response[0][1] = dE1.response[1][0] = -0.2188;
  dE1.response[0][2] = dE1.response[2][0] = -0.1300;
  dE1.response[1][2] = dE1.response[2][1] = 0.2732;
  LALFrDetector dE2Fr;
  sprintf(dE2Fr.name,"ET-2");
  sprintf(dE2Fr.prefix,"E2");
  dE2.frDetector=dE2Fr;
  dE2.response[0][0] = -0.1992;
  dE2.response[1][1] = 0.4234;
  dE2.response[2][2] = 0.0818;
  dE2.response[0][1] = dE2.response[1][0] = -0.0702;
  dE2.response[0][2] = dE2.response[2][0] = 0.2189;
  dE2.response[1][2] = dE2.response[2][1] = -0.0085;
  LALFrDetector dE3Fr;
  sprintf(dE3Fr.name,"ET-3");
  sprintf(dE3Fr.prefix,"E3");
  dE3.frDetector=dE3Fr;
  dE3.response[0][0] = 0.0326;
  dE3.response[1][1] = -0.1750;
  dE3.response[2][2] = 0.1423;
  dE3.response[0][1] = dE3.response[1][0] = 0.2891;
  dE3.response[0][2] = dE3.response[2][0] = -0.0889;
  dE3.response[1][2] = dE3.response[2][1] = -0.2647;  
  
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
	if(!IFOdata) XLAL_ERROR_NULL(XLAL_ENOMEM);
	
	procparam=LALInferenceGetProcParamVal(commandLine,"--injXML");
	if(procparam){
		SimInspiralTableFromLIGOLw(&injTable,procparam->value,0,0);
    if(!injTable){
			XLALPrintError("Unable to open injection file(LALInferenceReadData) %s\n",procparam->value);
			XLAL_ERROR_NULL(XLAL_EFUNC);
    }
	}
	if(LALInferenceGetProcParamVal(commandLine,"--sw_inj")){
		event=atoi(LALInferenceGetProcParamVal(commandLine,"--event")->value);
    while(q<event) {q++; injTable = injTable->next;}
    trigtime=XLALGPSGetREAL8(&(injTable->geocent_end_time));
	}
	
	procparam=LALInferenceGetProcParamVal(commandLine,"--PSDstart");
	LALStringToGPS(&status,&GPSstart,procparam->value,&chartmp);
	if(status.statusCode) REPORTSTATUS(&status);
	
	if(LALInferenceGetProcParamVal(commandLine,"--trigtime")){
		procparam=LALInferenceGetProcParamVal(commandLine,"--trigtime");
		LALStringToGPS(&status,&GPStrig,procparam->value,&chartmp);
	}
	else{
		char tt[]="";
		sprintf(tt,"%f",trigtime);
		LALStringToGPS(&status,&GPStrig,tt,&chartmp);
	}
	if(status.statusCode) REPORTSTATUS(&status);

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
		/*		if(!strcmp(IFOnames[i],"TAMA")||!strcmp(IFOnames[i],"T1")) {memcpy(IFOdata[i].detector,&lalCachedDetectors[LALDetectorIndexTAMA300DIFF]); continue;}*/
    
    if(!strcmp(IFOnames[i],"E1")){
			memcpy(IFOdata[i].detector,&dE1,sizeof(LALDetector));
      if(!Nchannel) sprintf((channels[i]),"E1:STRAIN"); continue;}
		if(!strcmp(IFOnames[i],"E2")){
      memcpy(IFOdata[i].detector,&dE2,sizeof(LALDetector));
      if(!Nchannel) sprintf((channels[i]),"E2:STRAIN"); continue;}
    if(!strcmp(IFOnames[i],"E3")){
      memcpy(IFOdata[i].detector,&dE3,sizeof(LALDetector));
      if(!Nchannel) sprintf((channels[i]),"E3:STRAIN"); continue;}
		if(!strcmp(IFOnames[i],"HM1")){
			/* Note, this is a sqrt(2)*7.5-km 3rd gen detector */
			LALFrDetector ETHomestakeFr;
			sprintf(ETHomestakeFr.name,"ET-HomeStake1");
			sprintf(ETHomestakeFr.prefix,"M1");
			/* Location of Homestake Mine vertex is */
			/* 44d21'23.11" N, 103d45'54.71" W */
			ETHomestakeFr.vertexLatitudeRadians = (44.+ 21./60  + 23.11/3600)*LAL_PI/180.0;
			ETHomestakeFr.vertexLongitudeRadians = - (103. +45./60 + 54.71/3600)*LAL_PI/180.0;
			ETHomestakeFr.vertexElevation=0.0;
			ETHomestakeFr.xArmAltitudeRadians=0.0;
			ETHomestakeFr.xArmAzimuthRadians=LAL_PI/2.0;
			ETHomestakeFr.yArmAltitudeRadians=0.0;
			ETHomestakeFr.yArmAzimuthRadians=0.0;
			ETHomestakeFr.xArmMidpoint = ETHomestakeFr.yArmMidpoint = sqrt(2.0)*7.5/2.0;
			IFOdata[i].detector=calloc(1,sizeof(LALDetector));
			XLALCreateDetector(IFOdata[i].detector,&ETHomestakeFr,LALDETECTORTYPE_IFODIFF);
			printf("Created Homestake Mine ET detector, location: %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
			printf("detector tensor:\n");
			for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
			continue;
		}
    if(!strcmp(IFOnames[i],"HM2")){
      /* Note, this is a sqrt(2)*7.5-km 3rd gen detector */
      LALFrDetector ETHomestakeFr;
      sprintf(ETHomestakeFr.name,"ET-HomeStake2");
      sprintf(ETHomestakeFr.prefix,"M2");
      /* Location of Homestake Mine vertex is */
      /* 44d21'23.11" N, 103d45'54.71" W */
      ETHomestakeFr.vertexLatitudeRadians = (44.+ 21./60  + 23.11/3600)*LAL_PI/180.0;
      ETHomestakeFr.vertexLongitudeRadians = - (103. +45./60 + 54.71/3600)*LAL_PI/180.0;
      ETHomestakeFr.vertexElevation=0.0;
      ETHomestakeFr.xArmAltitudeRadians=0.0;
      ETHomestakeFr.xArmAzimuthRadians=3.0*LAL_PI/4.0;
      ETHomestakeFr.yArmAltitudeRadians=0.0;
      ETHomestakeFr.yArmAzimuthRadians=LAL_PI/4.0;
      ETHomestakeFr.xArmMidpoint = ETHomestakeFr.yArmMidpoint = sqrt(2.0)*7500./2.0;
      IFOdata[i].detector=calloc(1,sizeof(LALDetector));
      XLALCreateDetector(IFOdata[i].detector,&ETHomestakeFr,LALDETECTORTYPE_IFODIFF);
      printf("Created Homestake Mine ET detector, location: %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
      printf("detector tensor:\n");
      for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
      continue;
    }
		if(!strcmp(IFOnames[i],"EM1")){
			LALFrDetector ETmic1;
			sprintf(ETmic1.name,"ET Michelson 1");
			sprintf(ETmic1.prefix,"F1");
			ETmic1.vertexLatitudeRadians = (43. + 37./60. + 53.0921/3600)*LAL_PI/180.0;
			ETmic1.vertexLongitudeRadians = (10. + 30./60. + 16.1878/3600.)*LAL_PI/180.0;
			ETmic1.vertexElevation = 0.0;
			ETmic1.xArmAltitudeRadians = ETmic1.yArmAltitudeRadians = 0.0;
			ETmic1.xArmAzimuthRadians = LAL_PI/2.0;
			ETmic1.yArmAzimuthRadians = 0.0;
			ETmic1.xArmMidpoint = ETmic1.yArmMidpoint = sqrt(2.0)*7500./2.;
			IFOdata[i].detector=calloc(1,sizeof(LALDetector));
			XLALCreateDetector(IFOdata[i].detector,&ETmic1,LALDETECTORTYPE_IFODIFF);
			printf("Created ET L-detector 1 (N/E) arms, location: %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
      printf("detector tensor:\n");
      for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
			continue;
		}
    if(!strcmp(IFOnames[i],"EM2")){
      LALFrDetector ETmic2;
      sprintf(ETmic2.name,"ET Michelson 2");
      sprintf(ETmic2.prefix,"F2");
      ETmic2.vertexLatitudeRadians = (43. + 37./60. + 53.0921/3600)*LAL_PI/180.0;
      ETmic2.vertexLongitudeRadians = (10. + 30./60. + 16.1878/3600.)*LAL_PI/180.0;
      ETmic2.vertexElevation = 0.0;
      ETmic2.xArmAltitudeRadians = ETmic2.yArmAltitudeRadians = 0.0;
      ETmic2.xArmAzimuthRadians = 3.0*LAL_PI/4.0;
      ETmic2.yArmAzimuthRadians = LAL_PI/4.0;
      ETmic2.xArmMidpoint = ETmic2.yArmMidpoint = sqrt(2.0)*7500./2.;
      IFOdata[i].detector=calloc(1,sizeof(LALDetector));
      XLALCreateDetector(IFOdata[i].detector,&ETmic2,LALDETECTORTYPE_IFODIFF);
      printf("Created ET L-detector 2 (NE/SE) arms, location: %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
      printf("detector tensor:\n");
      for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
      continue;
		}
    if(!strcmp(IFOnames[i],"I1")||!strcmp(IFOnames[i],"LIGOIndia")){
      /* Detector in India with 4k arms */
      LALFrDetector LIGOIndiaFr;
      sprintf(LIGOIndiaFr.name,"LIGO India");
      sprintf(LIGOIndiaFr.prefix,"I1");
      /* Location of India site is */
      /* 14d14' N 76d26' E */
      LIGOIndiaFr.vertexLatitudeRadians = (14. + 14./60.)*LAL_PI/180.0;
      LIGOIndiaFr.vertexLongitudeRadians = (76. + 26./60.)*LAL_PI/180.0;
      LIGOIndiaFr.vertexElevation = 0.0;
      LIGOIndiaFr.xArmAltitudeRadians = 0.0;
      LIGOIndiaFr.yArmAltitudeRadians = 0.0;
      LIGOIndiaFr.yArmMidpoint = 2000.;
      LIGOIndiaFr.xArmMidpoint = 2000.;
      LIGOIndiaFr.xArmAzimuthRadians = LAL_PI/2.;
      LIGOIndiaFr.yArmAzimuthRadians = 0.;
      IFOdata[i].detector=malloc(sizeof(LALDetector));
      memset(IFOdata[i].detector,0,sizeof(LALDetector));
      XLALCreateDetector(IFOdata[i].detector,&LIGOIndiaFr,LALDETECTORTYPE_IFODIFF);
      printf("Created LIGO India Detector, location %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
      printf("Detector tensor:\n");
      for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
      continue;
    }
		if(!strcmp(IFOnames[i],"A1")||!strcmp(IFOnames[i],"LIGOSouth")){
      /* Construct a detector at AIGO with 4k arms */
      LALFrDetector LIGOSouthFr;
      sprintf(LIGOSouthFr.name,"LIGO-South");
      sprintf(LIGOSouthFr.prefix,"A1");
      /* Location of the AIGO detector vertex is */
      /* 31d21'27.56" S, 115d42'50.34"E */
      LIGOSouthFr.vertexLatitudeRadians = - (31. + 21./60. + 27.56/3600.)*LAL_PI/180.0;
      LIGOSouthFr.vertexLongitudeRadians = (115. + 42./60. + 50.34/3600.)*LAL_PI/180.0;
      LIGOSouthFr.vertexElevation=0.0;
      LIGOSouthFr.xArmAltitudeRadians=0.0;
      LIGOSouthFr.xArmAzimuthRadians=AIGOang+LAL_PI/2.;
      LIGOSouthFr.yArmAltitudeRadians=0.0;
      LIGOSouthFr.yArmAzimuthRadians=AIGOang;
      LIGOSouthFr.xArmMidpoint=2000.;
      LIGOSouthFr.yArmMidpoint=2000.;
      IFOdata[i].detector=malloc(sizeof(LALDetector));
      memset(IFOdata[i].detector,0,sizeof(LALDetector));
      XLALCreateDetector(IFOdata[i].detector,&LIGOSouthFr,LALDETECTORTYPE_IFODIFF);
      printf("Created LIGO South detector, location: %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
      printf("Detector tensor:\n");
      for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
      continue;
    }
		if(!strcmp(IFOnames[i],"J1")||!strcmp(IFOnames[i],"LCGT")){
			/* Construct the LCGT telescope */
			REAL8 LCGTangle=19.0*(LAL_PI/180.0);
			LALFrDetector LCGTFr;
			sprintf(LCGTFr.name,"LCGT");
			sprintf(LCGTFr.prefix,"J1");
			LCGTFr.vertexLatitudeRadians  = 36.25 * LAL_PI/180.0;
			LCGTFr.vertexLongitudeRadians = (137.18 * LAL_PI/180.0);
			LCGTFr.vertexElevation=0.0;
			LCGTFr.xArmAltitudeRadians=0.0;
			LCGTFr.xArmAzimuthRadians=LCGTangle+LAL_PI/2.;
			LCGTFr.yArmAltitudeRadians=0.0;
			LCGTFr.yArmAzimuthRadians=LCGTangle;
			LCGTFr.xArmMidpoint=1500.;
			LCGTFr.yArmMidpoint=1500.;
			IFOdata[i].detector=malloc(sizeof(LALDetector));
			memset(IFOdata[i].detector,0,sizeof(LALDetector));
			XLALCreateDetector(IFOdata[i].detector,&LCGTFr,LALDETECTORTYPE_IFODIFF);
			printf("Created LCGT telescope, location: %lf, %lf, %lf\n",IFOdata[i].detector->location[0],IFOdata[i].detector->location[1],IFOdata[i].detector->location[2]);
      printf("Detector tensor:\n");
      for(int jdx=0;jdx<3;jdx++){
        for(j=0;j<3;j++) printf("%f ",IFOdata[i].detector->response[jdx][j]);
        printf("\n");
      }
      continue;
		}
		fprintf(stderr,"Unknown interferometer %s. Valid codes: H1 H2 L1 V1 GEO A1 J1 I1 E1 E2 E3 HM1 HM2 EM1 EM2\n",IFOnames[i]); exit(-1);
	}
	
	/* Set up FFT structures and window */
	for (i=0;i<Nifo;i++){
		/* Create FFT plans */
		IFOdata[i].timeToFreqFFTPlan = XLALCreateForwardREAL8FFTPlan((UINT4) seglen, 0 );
		if(!IFOdata[i].timeToFreqFFTPlan) XLAL_ERROR_NULL(XLAL_EFUNC);
		IFOdata[i].freqToTimeFFTPlan = XLALCreateReverseREAL8FFTPlan((UINT4) seglen,0);
		if(!IFOdata[i].freqToTimeFFTPlan) XLAL_ERROR_NULL(XLAL_EFUNC);		
		/* Setup windows */
		IFOdata[i].window=XLALCreateTukeyREAL8Window(seglen,(REAL8)2.0*padding*SampleRate/(REAL8)seglen);
		if(!IFOdata[i].window) XLAL_ERROR_NULL(XLAL_EFUNC);
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
			if(!datarandparam) XLAL_ERROR_NULL(XLAL_EFUNC);
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
			if(!IFOdata[i].oneSidedNoisePowerSpectrum) XLAL_ERROR_NULL(XLAL_EFUNC);
			for(j=0;j<IFOdata[i].oneSidedNoisePowerSpectrum->data->length;j++)
			{
				PSD(&status,&(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]),j*IFOdata[i].oneSidedNoisePowerSpectrum->deltaF);
				IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]*=scalefactor;
			}
			IFOdata[i].freqData = (COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("stilde",&segStart,0.0,IFOdata[i].oneSidedNoisePowerSpectrum->deltaF,&lalDimensionlessUnit,seglen/2 +1);
			if(!IFOdata[i].freqData) XLAL_ERROR_NULL(XLAL_EFUNC);

			/* Create the fake data */
			int j_Lo = (int) IFOdata[i].fLow/IFOdata[i].freqData->deltaF;
			for(j=j_Lo;j<IFOdata[i].freqData->data->length;j++){
				IFOdata[i].freqData->data->data[j].re=XLALNormalDeviate(datarandparam)*(0.5*sqrt(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]/IFOdata[i].freqData->deltaF));
				IFOdata[i].freqData->data->data[j].im=XLALNormalDeviate(datarandparam)*(0.5*sqrt(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]/IFOdata[i].freqData->deltaF));
			}
			IFOdata[i].freqData->data->data[0].re=0; 			IFOdata[i].freqData->data->data[0].im=0;
			const char timename[]="timeData";
			IFOdata[i].timeData=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries(timename,&segStart,0.0,(REAL8)1.0/SampleRate,&lalDimensionlessUnit,(size_t)seglen);
			if(!IFOdata[i].timeData) XLAL_ERROR_NULL(XLAL_EFUNC);
			XLALREAL8FreqTimeFFT(IFOdata[i].timeData,IFOdata[i].freqData,IFOdata[i].freqToTimeFFTPlan);
			if(*XLALGetErrnoPtr()) printf("XLErr: %s\n",XLALErrorString(*XLALGetErrnoPtr()));
			XLALDestroyRandomParams(datarandparam);
		}
		else{ /* Not using fake data, load the data from a cache file */
			fprintf(stderr,"Estimating PSD for %s using %i segments of %i samples (%lfs)\n",IFOnames[i],nSegs,(int)seglen,SegmentLength);
			PSDtimeSeries=readTseries(caches[i],channels[i],GPSstart,PSDdatalength);
			if(!PSDtimeSeries) {XLALPrintError("Error reading PSD data for %s\n",IFOnames[i]); XLAL_ERROR_NULL(XLAL_EFUNC);}
			XLALResampleREAL8TimeSeries(PSDtimeSeries,1.0/SampleRate);
			PSDtimeSeries=(REAL8TimeSeries *)XLALShrinkREAL8TimeSeries(PSDtimeSeries,(size_t) 0, (size_t) seglen*nSegs);
			if(!PSDtimeSeries) XLAL_ERROR_NULL(XLAL_EFUNC);
			IFOdata[i].oneSidedNoisePowerSpectrum=(REAL8FrequencySeries *)XLALCreateREAL8FrequencySeries("spectrum",&PSDtimeSeries->epoch,0.0,(REAL8)(SampleRate)/seglen,&lalDimensionlessUnit,seglen/2 +1);
			if(!IFOdata[i].oneSidedNoisePowerSpectrum) XLAL_ERROR_NULL(XLAL_EFUNC);
			if (LALInferenceGetProcParamVal(commandLine, "--PSDwelch"))
				XLALREAL8AverageSpectrumWelch(IFOdata[i].oneSidedNoisePowerSpectrum ,PSDtimeSeries, seglen, (UINT4)seglen, IFOdata[i].window, IFOdata[i].timeToFreqFFTPlan);
      else
        XLALREAL8AverageSpectrumMedian(IFOdata[i].oneSidedNoisePowerSpectrum ,PSDtimeSeries, seglen, (UINT4)seglen, IFOdata[i].window, IFOdata[i].timeToFreqFFTPlan);	

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
                        
			if(!IFOdata[i].timeData) {
				XLALPrintError("Error reading segment data for %s at %i\n",IFOnames[i],segStart.gpsSeconds);
				XLAL_ERROR_NULL(XLAL_EFUNC);
			}
			XLALResampleREAL8TimeSeries(IFOdata[i].timeData,1.0/SampleRate);	 
			if(!IFOdata[i].timeData) {XLALPrintError("Error reading segment data for %s\n",IFOnames[i]); XLAL_ERROR_NULL(XLAL_EFUNC);}
			IFOdata[i].freqData=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("freqData",&(IFOdata[i].timeData->epoch),0.0,1.0/SegmentLength,&lalDimensionlessUnit,seglen/2+1);
			if(!IFOdata[i].freqData) XLAL_ERROR_NULL(XLAL_EFUNC);
			IFOdata[i].windowedTimeData=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("windowed time data",&(IFOdata[i].timeData->epoch),0.0,1.0/SampleRate,&lalDimensionlessUnit,seglen);
			if(!IFOdata[i].windowedTimeData) XLAL_ERROR_NULL(XLAL_EFUNC);
			XLALDDVectorMultiply(IFOdata[i].windowedTimeData->data,IFOdata[i].timeData->data,IFOdata[i].window->data);
			XLALREAL8TimeFreqFFT(IFOdata[i].freqData,IFOdata[i].windowedTimeData,IFOdata[i].timeToFreqFFTPlan);
			
			for(j=0;j<IFOdata[i].freqData->data->length;j++){
				IFOdata[i].freqData->data->data[j].re/=sqrt(IFOdata[i].window->sumofsquares / IFOdata[i].window->data->length);
				IFOdata[i].freqData->data->data[j].im/=sqrt(IFOdata[i].window->sumofsquares / IFOdata[i].window->data->length);
				IFOdata[i].windowedTimeData->data->data[j] /= sqrt(IFOdata[i].window->sumofsquares / IFOdata[i].window->data->length);
			}
		} /* End of data reading process */

		/* Now that the PSD is set up, make the TDW. */
    IFOdata[i].timeDomainNoiseWeights = 
                  (REAL8TimeSeries *)XLALCreateREAL8TimeSeries("time domain weights", 
                                                               &(IFOdata[i].oneSidedNoisePowerSpectrum->epoch),
                                                               0.0,
                                                               1.0/SampleRate,
                                                               &lalDimensionlessUnit,
                                                               seglen);
		if(!IFOdata[i].timeDomainNoiseWeights) XLAL_ERROR_NULL(XLAL_EFUNC);
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
  
	for (i=0;i<Nifo;i++) IFOdata[i].SNR=0.0; //SNR of the injection ONLY IF INJECTION. Set to 0.0 by default.
  
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
	if(!IFOdata->whiteFreqData) XLAL_ERROR_VOID(XLAL_EFUNC);
  IFOdata->whiteTimeData = 
    XLALCreateREAL8TimeSeries("whitened time data",
                              &(IFOdata->timeData->epoch),
                              0.0,
                              deltaT,
                              &lalDimensionlessUnit,
                              IFOdata->timeData->data->length);
	if(!IFOdata->whiteTimeData) XLAL_ERROR_VOID(XLAL_EFUNC);

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
	FILE *rawWaveform=NULL;
	ProcessParamsTable *ppt=NULL;
	REAL8 bufferLength = 512.0; /* Default length of buffer for injections (seconds) */
	UINT4 bufferN=0;
	LIGOTimeGPS bufferStart;

	
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
	thisData=IFOdata;
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
	injEvent = injTable;
	injEvent->next = NULL;
	
	//memset(&InjectGW,0,sizeof(InjectGW));
	Approximant injapprox;
	LALGetApproximantFromString(&status,injTable->waveform,&injapprox);
	printf("Injecting approximant %i: %s\n", injapprox, injTable->waveform);
	REPORTSTATUS(&status);
	//LALGenerateInspiral(&status,&InjectGW,injTable,&InjParams);
	//if(status.statusCode!=0) {fprintf(stderr,"Error generating injection!\n"); REPORTSTATUS(&status); }
	
	/* Begin loop over interferometers */
	while(thisData){
		memset(&det,0,sizeof(det));
		det.site=thisData->detector;
		COMPLEX8FrequencySeries *resp = XLALCreateCOMPLEX8FrequencySeries("response",&thisData->timeData->epoch,
																		  0.0,
																		  thisData->freqData->deltaF,
																		  &strainPerCount,
																		  thisData->freqData->data->length);
		
		for(i=0;i<resp->data->length;i++) {resp->data->data[i].re=(REAL4)1.0; resp->data->data[i].im=0.0;}
		/* Originally created for injecting into DARM-ERR, so transfer function was needed.  
		But since we are injecting into h(t), the transfer function from h(t) to h(t) is 1.*/

		/* We need a long buffer to inject into so that FindChirpInjectSignals() works properly
		 for low mass systems. Use 100 seconds here */
		bufferN = (UINT4) (bufferLength/thisData->timeData->deltaT);
		memcpy(&bufferStart,&thisData->timeData->epoch,sizeof(LIGOTimeGPS));
		XLALGPSAdd(&bufferStart,(REAL8) thisData->timeData->data->length * thisData->timeData->deltaT);
		XLALGPSAdd(&bufferStart,-bufferLength);
		injectionBuffer=(REAL4TimeSeries *)XLALCreateREAL4TimeSeries(thisData->detector->frDetector.prefix,
																	 &bufferStart, 0.0, thisData->timeData->deltaT,
																	 &lalADCCountUnit, bufferN);
		/* This marks the sample in which the real segment starts, within the buffer */
		INT4 realStartSample=(INT4)((thisData->timeData->epoch.gpsSeconds - injectionBuffer->epoch.gpsSeconds)/thisData->timeData->deltaT);
		realStartSample+=(INT4)((thisData->timeData->epoch.gpsNanoSeconds - injectionBuffer->epoch.gpsNanoSeconds)*1e-9/thisData->timeData->deltaT);

		/*LALSimulateCoherentGW(&status,injWave,&InjectGW,&det);*/
    LALFindChirpInjectSignals(&status,injectionBuffer,injEvent,resp);
		if(status.statusCode) REPORTSTATUS(&status);

		XLALDestroyCOMPLEX8FrequencySeries(resp);

    /* Checking the lenght of the injection waveform with respect of thisData->timeData->data->length */
    CoherentGW            waveform;
    PPNParamStruc         ppnParams;
    memset( &waveform, 0, sizeof(CoherentGW) );
    memset( &ppnParams, 0, sizeof(PPNParamStruc) );
    ppnParams.deltaT   = thisData->timeData->deltaT;
    ppnParams.lengthIn = 0;
    ppnParams.ppn      = NULL;
    unsigned lengthTest = 0;
    
    LALGenerateInspiral(&status, &waveform, injEvent, &ppnParams ); //Recompute the waveform just to get access to ppnParams.tc and waveform.h->data->length or waveform.phi->data->length
		if(status.statusCode) REPORTSTATUS(&status);

    if(waveform.h){lengthTest = waveform.h->data->length;}
    if(waveform.phi){lengthTest = waveform.phi->data->length;}
    
    if(lengthTest>thisData->timeData->data->length-(UINT4)ceil((2.0*padding+2.0)/thisData->timeData->deltaT)){
      fprintf(stderr, "WARNING: waveform length = %u is longer than thisData->timeData->data->length = %d minus the window width = %d and the 2.0 seconds after tc (total of %d points available).\n", lengthTest, thisData->timeData->data->length, (INT4)ceil((2.0*padding)/thisData->timeData->deltaT) , thisData->timeData->data->length-(INT4)ceil((2.0*padding+2.0)/thisData->timeData->deltaT));
      fprintf(stderr, "The waveform injected is %f seconds long. Consider increasing the %f seconds segment length (--seglen) to be greater than %f. (in %s, line %d)\n",ppnParams.tc , thisData->timeData->data->length * thisData->timeData->deltaT, ppnParams.tc + 2.0*padding + 2.0, __FILE__, __LINE__);
    }
    
		/* Now we cut the injection buffer down to match the time domain wave size */
		injectionBuffer=(REAL4TimeSeries *)XLALCutREAL4TimeSeries(injectionBuffer,realStartSample,thisData->timeData->data->length);
		if (!injectionBuffer) XLAL_ERROR_VOID(XLAL_EFUNC);
		
		if(status.statusCode) REPORTSTATUS(&status);
/*		for(j=0;j<injWave->data->length;j++) printf("%f\n",injWave->data->data[j]);*/
		REAL8TimeSeries *inj8Wave=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("injection8",
																			  &thisData->timeData->epoch,
																			  0.0,
																			  thisData->timeData->deltaT,
																			  &lalDimensionlessUnit,
																			  thisData->timeData->data->length);
		if(!inj8Wave) XLAL_ERROR_VOID(XLAL_EFUNC);
		for(i=0;i<injectionBuffer->data->length;i++) inj8Wave->data->data[i]=(REAL8)injectionBuffer->data->data[i];
		XLALDestroyREAL4TimeSeries(injectionBuffer);
		injF=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("injF",
																			&thisData->timeData->epoch,
																			0.0,
																			thisData->freqData->deltaF,
																			&lalDimensionlessUnit,
																			thisData->freqData->data->length);
		if(!injF) XLAL_ERROR_VOID(XLAL_EFUNC);
		/* Window the data */
		REAL4 WinNorm = sqrt(thisData->window->sumofsquares/thisData->window->data->length);
		for(j=0;j<inj8Wave->data->length;j++) inj8Wave->data->data[j]*=thisData->window->data->data[j]/WinNorm;
		XLALREAL8TimeFreqFFT(injF,inj8Wave,thisData->timeToFreqFFTPlan);
/*		for(j=0;j<injF->data->length;j++) printf("%lf\n",injF->data->data[j].re);*/
		if(thisData->oneSidedNoisePowerSpectrum){
			for(SNR=0.0,j=thisData->fLow/injF->deltaF;j<injF->data->length;j++){
				SNR+=2.0*pow(injF->data->data[j].re,2.0)/(4.0*thisData->oneSidedNoisePowerSpectrum->data->data[j]);
				SNR+=2.0*pow(injF->data->data[j].im,2.0)/(4.0*thisData->oneSidedNoisePowerSpectrum->data->data[j]);
			}
		}
    thisData->SNR=sqrt(SNR);
		NetworkSNR+=SNR;
		
		/* Actually inject the waveform */
		for(j=0;j<inj8Wave->data->length;j++) thisData->timeData->data->data[j]+=inj8Wave->data->data[j];

FILE* file=fopen("InjSignal.dat", "w");
//FILE* file2=fopen("Noise.dat", "w");
		for(j=0;j<injF->data->length;j++){
//fprintf(file2, "%lg %lg \t %lg\n", thisData->freqData->deltaF*j, thisData->freqData->data->data[j].re, thisData->freqData->data->data[j].im);

			thisData->freqData->data->data[j].re+=injF->data->data[j].re;
			thisData->freqData->data->data[j].im+=injF->data->data[j].im;
fprintf(file, "%lg %lg \t %lg\n", thisData->freqData->deltaF*j, injF->data->data[j].re, injF->data->data[j].im);
		}
		fprintf(stdout,"Injected SNR in detector %s = %g\n",thisData->detector->frDetector.name,thisData->SNR);
fclose(file);		
//fclose(file2);
		
    char filename[256];
    sprintf(filename,"%s_time.dat",thisData->detector->frDetector.name);
    file=fopen(filename, "w");
		for(j=0;j<inj8Wave->data->length;j++){   
      fprintf(file, "%.6f\t%lg\n", XLALGPSGetREAL8(&thisData->timeData->epoch) + thisData->timeData->deltaT*j, inj8Wave->data->data[j]);
		}
    fclose(file);
		
		XLALDestroyREAL8TimeSeries(inj8Wave);
		XLALDestroyCOMPLEX16FrequencySeries(injF);
		thisData=thisData->next;
	}
	NetworkSNR=sqrt(NetworkSNR);
	fprintf(stdout,"Network SNR of event %d = %g\n",event,NetworkSNR);
	
	/* Output waveform raw h-plus mode */
	if( (ppt=LALInferenceGetProcParamVal(commandLine,"--rawwaveform")) )
	{
		rawWaveform=fopen(ppt->value,"w");
		bufferN = (UINT4) (bufferLength/IFOdata->timeData->deltaT);
		memcpy(&bufferStart,&IFOdata->timeData->epoch,sizeof(LIGOTimeGPS));
		XLALGPSAdd(&bufferStart,(REAL8) IFOdata->timeData->data->length * IFOdata->timeData->deltaT);
		XLALGPSAdd(&bufferStart,-bufferLength);
		
		COMPLEX8FrequencySeries *resp = XLALCreateCOMPLEX8FrequencySeries("response",&IFOdata->timeData->epoch,
																																			0.0,
																																			IFOdata->freqData->deltaF,
																																			&strainPerCount,
																																			IFOdata->freqData->data->length);
		if(!resp) XLAL_ERROR_VOID(XLAL_EFUNC);
		injectionBuffer=(REAL4TimeSeries *)XLALCreateREAL4TimeSeries("None",
																																 &bufferStart, 0.0, IFOdata->timeData->deltaT,
																																 &lalADCCountUnit, bufferN);
		if(!injectionBuffer) XLAL_ERROR_VOID(XLAL_EFUNC);
		/* This marks the sample in which the real segment starts, within the buffer */
		INT4 realStartSample=(INT4)((IFOdata->timeData->epoch.gpsSeconds - injectionBuffer->epoch.gpsSeconds)/IFOdata->timeData->deltaT);
		realStartSample+=(INT4)((IFOdata->timeData->epoch.gpsNanoSeconds - injectionBuffer->epoch.gpsNanoSeconds)*1e-9/IFOdata->timeData->deltaT);
		
		LALFindChirpInjectSignals(&status,injectionBuffer,injEvent,resp);
		if(status.statusCode) REPORTSTATUS(&status);
		XLALDestroyCOMPLEX8FrequencySeries(resp);
		injectionBuffer=(REAL4TimeSeries *)XLALCutREAL4TimeSeries(injectionBuffer,realStartSample,IFOdata->timeData->data->length);
		for(j=0;j<injectionBuffer->data->length;j++) fprintf(rawWaveform,"%.6f\t%g\n", XLALGPSGetREAL8(&IFOdata->timeData->epoch) + IFOdata->timeData->deltaT*j, injectionBuffer->data->data[j]);
		fclose(rawWaveform);
		XLALDestroyREAL4TimeSeries(injectionBuffer);
	}
	
	return;
}


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

#include <lal/SeqFactories.h>
#include <lal/DetectorSite.h>
#include <lal/GenerateInspiral.h>
#include <lal/GeneratePPNInspiral.h>
#include <lal/SimulateCoherentGW.h>
#include <lal/Inject.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOMetadataInspiralUtils.h>
#include <lal/LIGOMetadataRingdownUtils.h>
#include <lal/LALInspiralBank.h>
#include <lal/FindChirp.h>
#include <lal/LALInspiralBank.h>
#include <lal/GenerateInspiral.h>
#include <lal/NRWaveInject.h>
#include <lal/GenerateInspRing.h>
#include <lal/LALErrno.h>
#include <math.h>
#include <lal/LALInspiral.h>
#include <lal/LALSimulation.h>

#include <lal/LALInference.h>
#include <lal/LALInferenceReadData.h>
#include <lal/LALInferenceLikelihood.h>

struct fvec {
	REAL8 f;
	REAL8 x;
};

struct fvec *interpFromFile(char *filename);

struct fvec *interpFromFile(char *filename){
	UINT4 fileLength=0;
	UINT4 i=0;
	UINT4 minLength=100; /* size of initial file buffer, and also size of increment */
	FILE *interpfile=NULL;
	struct fvec *interp=NULL;
	interp=calloc(minLength,sizeof(struct fvec)); /* Initialise array */
	if(!interp) {printf("Unable to allocate memory buffer for reading interpolation file\n");}
	fileLength=minLength;
	REAL8 f=0.0,x=0.0;
	interpfile = fopen(filename,"r");
	if (interpfile==NULL){
		printf("Unable to open file %s\n",filename);
		exit(1);
	}
	while(2==fscanf(interpfile," %lf %lf ", &f, &x )){
		interp[i].f=f; interp[i].x=x*x;
		i++;
		if(i>fileLength-1){ /* Grow the array */
			interp=realloc(interp,(fileLength+minLength)*sizeof(struct fvec));
			fileLength+=minLength;
		}
	}
	interp[i].f=0; interp[i].x=0;
	fileLength=i+1;
	interp=realloc(interp,fileLength*sizeof(struct fvec)); /* Resize array */
	fclose(interpfile);
	printf("Read %i records from %s\n",fileLength-1,filename);
	return interp;
}

REAL8 interpolate(struct fvec *fvec, REAL8 f);
REAL8 interpolate(struct fvec *fvec, REAL8 f){
	int i=0;
	REAL8 a=0.0; /* fractional distance between bins */
	REAL8 delta=0.0;
	if(f<fvec[0].f) return(0.0);
	while(fvec[i].f<f && (fvec[i].x!=0.0 && fvec[i].f!=0.0)){i++;};
	if (fvec[i].f==0.0 && fvec[i].x==0.0) /* Frequency above moximum */
	{
		return (fvec[i-1].x);
	}
	a=(fvec[i].f-f)/(fvec[i].f-fvec[i-1].f);
	delta=fvec[i].x-fvec[i-1].x;
	return (fvec[i-1].x + delta*a);
}

typedef void (NoiseFunc)(LALStatus *statusPtr,REAL8 *psd,REAL8 f);
void MetaNoiseFunc(LALStatus *status, REAL8 *psd, REAL8 f, struct fvec *interp, NoiseFunc *noisefunc);

void MetaNoiseFunc(LALStatus *status, REAL8 *psd, REAL8 f, struct fvec *interp, NoiseFunc *noisefunc){
	if(interp==NULL&&noisefunc==NULL){
		printf("ERROR: Trying to calculate PSD with NULL inputs\n");
		exit(1);
	}
	if(interp!=NULL && noisefunc!=NULL){
		printf("ERROR: You have specified both an interpolation vector and a function to calculate the PSD\n");
		exit(1);
	}
	if(noisefunc!=NULL){
		noisefunc(status,psd,f);
		return;
	}
	if(interp!=NULL){ /* Use linear interpolation of the interp vector */
		*psd=interpolate(interp,f);
		return;
	}
}

void
LALInferenceLALFindChirpInjectSignals (
                                       LALStatus                  *status,
                                       REAL4TimeSeries            *chan,
                                       SimInspiralTable           *events,
                                       COMPLEX8FrequencySeries    *resp,
                                       LALDetector                *detector
                                       );
static int FindTimeSeriesStartAndEnd (
                                      REAL4Vector *signalvec,
                                      UINT4 *start,
                                      UINT4 *end
                                      );

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
 --ifo [IFO1,IFO2,...]          IFOs can be H1,L1,V1\n\
 --cache [cache1,cache2,...]    LAL cache files (LALLIGO, LALAdLIGO, LALVirgo to simulate these detectors)\n\
 --psdstart GPStime             GPS start time of PSD estimation data\n\
 --psdlength length             length of PSD estimation data in seconds\n\
 --seglen length                length of segments for PSD estimation and analysis in seconds\n\
 --trigtime GPStime             GPS time of the trigger to analyse\n\
(--srate rate)                  Downsample data to rate in Hz (4096.0,)\n\
(--flow [freq1,freq2,...])      Specify lower frequency cutoff for overlap integral (40.0)\n\
(--fhigh [freq1,freq2,...])     Specify higher frequency cutoff for overlap integral (2048.0)\n\
(--channel [chan1,chan2,...])   Specify channel names when reading cache files\n\
(--dataseed number)             Specify random seed to use when generating data\n\
(--lalsimulationinjection)      Enables injections via the LALSimulation package\n\
(--inj-lambda1)                 value of lambda1 to be injected, LALSimulation only (0)\n\
(--inj-lambda2)                 value of lambda1 to be injected, LALSimulation only (0)\n\
(--inj-interactionFlags)        value of the interaction flag to be injected, LALSimulation only (LAL_SIM_INSPIRAL_INTERACTION_ALL)\n"


LALInferenceIFOData *LALInferenceReadData(ProcessParamsTable *commandLine)
/* Read in the data and store it in a LALInferenceIFOData structure */
{
	LALStatus status;
	INT4 dataseed=0;
	memset(&status,0,sizeof(status));
	ProcessParamsTable *procparam=NULL,*ppt=NULL;
	LALInferenceIFOData *headIFO=NULL,*IFOdata=NULL;
	REAL8 SampleRate=4096.0,SegmentLength=0;
	if(LALInferenceGetProcParamVal(commandLine,"--srate")) SampleRate=atof(LALInferenceGetProcParamVal(commandLine,"--srate")->value);
        const REAL8 defaultFLow = 40.0;
	int nSegs=0;
	size_t seglen=0;
	REAL8TimeSeries *PSDtimeSeries=NULL;
	REAL8 padding=0.4;//Default was 1.0 second. However for The Event the Common Inputs specify a Tukey parameter of 0.1, so 0.4 second of padding for 8 seconds of data.
	UINT4 Ncache=0,Nifo=0,Nchannel=0,NfLow=0,NfHigh=0;
	UINT4 i,j;
	//int FakeFlag=0; - set but not used
	char strainname[]="LSC-STRAIN";
	UINT4 q=0;	
	//typedef void (NoiseFunc)(LALStatus *statusPtr,REAL8 *psd,REAL8 f);
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
  REAL8 AIGOang=0.0; //orientation angle for the proposed Australian detector.
  procparam=LALInferenceGetProcParamVal(commandLine,"--aigoang");
  if(!procparam) procparam=LALInferenceGetProcParamVal(commandLine,"--AIGOang");
  if(procparam)
      AIGOang=atof(procparam->value)*LAL_PI/180.0;
  
  struct fvec *interp;
  int interpFlag=0;
	if(!LALInferenceGetProcParamVal(commandLine,"--cache")||!(LALInferenceGetProcParamVal(commandLine,"--IFO")||LALInferenceGetProcParamVal(commandLine,"--ifo"))  ||
	   !(LALInferenceGetProcParamVal(commandLine,"--PSDstart")||LALInferenceGetProcParamVal(commandLine,"--psdstart")) ||
	   !(LALInferenceGetProcParamVal(commandLine,"--PSDlength")||LALInferenceGetProcParamVal(commandLine,"--psdlength")) ||!LALInferenceGetProcParamVal(commandLine,"--seglen"))
	{fprintf(stderr,USAGE); return(NULL);}
	
  /* ET detectors */
	LALDetector dE1,dE2,dE3;
  /* response of the detectors */
  dE1.type = dE2.type = dE3.type = LALDETECTORTYPE_IFODIFF;
  dE1.location[0] = dE2.location[0] = dE3.location[0] = 4.5464e6;
  dE1.location[1] = dE2.location[1] = dE3.location[1] = 8.4299e5;
  dE1.location[2] = dE2.location[2] = dE3.location[2] = 4.3786e6;
  sprintf(dE1.frDetector.name,"ET-1");
  sprintf(dE1.frDetector.prefix,"E1");
  dE1.response[0][0] = 0.1666;
  dE1.response[1][1] = -0.2484;
  dE1.response[2][2] = 0.0818;
  dE1.response[0][1] = dE1.response[1][0] = -0.2188;
  dE1.response[0][2] = dE1.response[2][0] = -0.1300;
  dE1.response[1][2] = dE1.response[2][1] = 0.2732;
  sprintf(dE2.frDetector.name,"ET-2");
  sprintf(dE2.frDetector.prefix,"E2");
  dE2.response[0][0] = -0.1992;
  dE2.response[1][1] = 0.4234;
  dE2.response[2][2] = 0.0818;
  dE2.response[0][1] = dE2.response[1][0] = -0.0702;
  dE2.response[0][2] = dE2.response[2][0] = 0.2189;
  dE2.response[1][2] = dE2.response[2][1] = -0.0085;
  sprintf(dE3.frDetector.name,"ET-3");
  sprintf(dE3.frDetector.prefix,"E3");
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
	ppt=LALInferenceGetProcParamVal(commandLine,"--ifo");
	if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--IFO");
	LALInferenceParseCharacterOptionString(ppt->value,&IFOnames,&Nifo);
	
	ppt=LALInferenceGetProcParamVal(commandLine,"--flow");
	if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--fLow");
	if(ppt){
		LALInferenceParseCharacterOptionString(ppt->value,&fLows,&NfLow);
	}
	ppt=LALInferenceGetProcParamVal(commandLine,"--fhigh");
	if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--fHigh");
	if(ppt){
		LALInferenceParseCharacterOptionString(ppt->value,&fHighs,&NfHigh);
	}
	if(LALInferenceGetProcParamVal(commandLine,"--dataseed")){
		procparam=LALInferenceGetProcParamVal(commandLine,"--dataseed");
		dataseed=atoi(procparam->value);
	}
								   
	if(Nifo!=Ncache) {fprintf(stderr,"ERROR: Must specify equal number of IFOs and Cache files\n"); exit(1);}
	if(Nchannel!=0 && Nchannel!=Nifo) {fprintf(stderr,"ERROR: Please specify a channel for all caches, or omit to use the defaults\n"); exit(1);}
	
	IFOdata=headIFO=calloc(sizeof(LALInferenceIFOData),Nifo);
	if(!IFOdata) XLAL_ERROR_NULL(XLAL_ENOMEM);
	
	if(LALInferenceGetProcParamVal(commandLine,"--injXML"))
	{
		XLALPrintError("ERROR: --injXML option is deprecated. Use --inj and update your scripts\n");
        exit(1);
	}
	procparam=LALInferenceGetProcParamVal(commandLine,"--inj");
	if(procparam){
		SimInspiralTableFromLIGOLw(&injTable,procparam->value,0,0);
		if(!injTable){
			XLALPrintError("Unable to open injection file(LALInferenceReadData) %s\n",procparam->value);
			XLAL_ERROR_NULL(XLAL_EFUNC);
		}
        procparam=LALInferenceGetProcParamVal(commandLine,"--event");
        if(procparam) event=atoi(procparam->value);
        while(q<event) {q++; injTable=injTable->next;}
	}
	
	procparam=LALInferenceGetProcParamVal(commandLine,"--psdstart");
	if (!procparam) procparam=LALInferenceGetProcParamVal(commandLine,"--PSDstart");
	LALStringToGPS(&status,&GPSstart,procparam->value,&chartmp);
	if(status.statusCode) REPORTSTATUS(&status);
	
	if(LALInferenceGetProcParamVal(commandLine,"--trigtime")){
		procparam=LALInferenceGetProcParamVal(commandLine,"--trigtime");
		LALStringToGPS(&status,&GPStrig,procparam->value,&chartmp);
	}
	else{
		if(injTable) memcpy(&GPStrig,&(injTable->geocent_end_time),sizeof(GPStrig));
		else {
            XLALPrintError("Error: No trigger time specifed and no injection given \n");
            XLAL_ERROR_NULL(XLAL_EINVAL);
        }
	}
	if(status.statusCode) REPORTSTATUS(&status);
	ppt=LALInferenceGetProcParamVal(commandLine,"--psdlength");
	if(!ppt) ppt=LALInferenceGetProcParamVal(commandLine,"--PSDlength");
	PSDdatalength=atof(ppt->value);
	SegmentLength=atof(LALInferenceGetProcParamVal(commandLine,"--seglen")->value);
	seglen=(size_t)(SegmentLength*SampleRate);
	nSegs=(int)floor(PSDdatalength/SegmentLength);
	
	for(i=0;i<Nifo;i++) {
          IFOdata[i].fLow=fLows?atof(fLows[i]):defaultFLow; 
          IFOdata[i].fHigh=fHighs?atof(fHighs[i]):(SampleRate/2.0-(1.0/SegmentLength));
          strncpy(IFOdata[i].name, IFOnames[i], DETNAMELEN);
          IFOdata[i].STDOF = 4.0 / M_PI * nSegs;
          fprintf(stderr, "Detector %s will run with %g DOF if Student's T likelihood used.\n",
                  IFOdata[i].name, IFOdata[i].STDOF);
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
			sprintf(ETmic1.name,"ET_Michelson_1");
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
      sprintf(ETmic2.name,"ET_Michelson_2");
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
      sprintf(LIGOIndiaFr.name,"LIGO_India");
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
    /* Check to see if an interpolation file is specified */
		interpFlag=0;
		interp=NULL;
		if(strstr(caches[i],"interp:")==caches[i]){
			/* Extract the file name */
			char *interpfilename=&(caches[i][7]);
			printf("Looking for interpolation file %s\n",interpfilename);
			interpFlag=1;
			interp=interpFromFile(interpfilename);
		}    
		/* Check if fake data is requested */
		if(interpFlag || (!(strcmp(caches[i],"LALLIGO") && strcmp(caches[i],"LALVirgo") && strcmp(caches[i],"LALGEO") && strcmp(caches[i],"LALEGO")
			 && strcmp(caches[i],"LALAdLIGO"))))
		{
			//FakeFlag=1; - set but not used
			datarandparam=XLALCreateRandomParams(dataseed?dataseed+(int)i:dataseed);
			if(!datarandparam) XLAL_ERROR_NULL(XLAL_EFUNC);
			/* Selection of the noise curve */
			if(!strcmp(caches[i],"LALLIGO")) {PSD = &LALLIGOIPsd; scalefactor=9E-46;}
			if(!strcmp(caches[i],"LALVirgo")) {PSD = &LALVIRGOPsd; scalefactor=1.0;}
			if(!strcmp(caches[i],"LALGEO")) {PSD = &LALGEOPsd; scalefactor=1E-46;}
			if(!strcmp(caches[i],"LALEGO")) {PSD = &LALEGOPsd; scalefactor=1.0;}
			if(!strcmp(caches[i],"LALAdLIGO")) {PSD = &LALAdvLIGOPsd; scalefactor = 1E-49;}
      if(interpFlag) {PSD=NULL; scalefactor=1.0;}
			//if(!strcmp(caches[i],"LAL2kLIGO")) {PSD = &LALAdvLIGOPsd; scalefactor = 36E-46;}
			if(PSD==NULL && !interpFlag) {fprintf(stderr,"Error: unknown simulated PSD: %s\n",caches[i]); exit(-1);}
      
      
			IFOdata[i].oneSidedNoisePowerSpectrum=(REAL8FrequencySeries *)
						XLALCreateREAL8FrequencySeries("spectrum",&GPSstart,0.0,
																					 (REAL8)(SampleRate)/seglen,&lalDimensionlessUnit,seglen/2 +1);
			if(!IFOdata[i].oneSidedNoisePowerSpectrum) XLAL_ERROR_NULL(XLAL_EFUNC);
			for(j=0;j<IFOdata[i].oneSidedNoisePowerSpectrum->data->length;j++)
			{
				MetaNoiseFunc(&status,&(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]),j*IFOdata[i].oneSidedNoisePowerSpectrum->deltaF,interp,PSD);
        //PSD(&status,&(IFOdata[i].oneSidedNoisePowerSpectrum->data->data[j]),j*IFOdata[i].oneSidedNoisePowerSpectrum->deltaF);
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
	
	if(!LALInferenceGetProcParamVal(commandLine,"--inj")) {fprintf(stdout,"No injection file specified, not injecting\n"); return;}
	if(LALInferenceGetProcParamVal(commandLine,"--event")){
    event= atoi(LALInferenceGetProcParamVal(commandLine,"--event")->value);
    fprintf(stdout,"Injecting event %d\n",event);
	}
	Ninj=SimInspiralTableFromLIGOLw(&injTable,LALInferenceGetProcParamVal(commandLine,"--inj")->value,0,0);
	REPORTSTATUS(&status);
	printf("Ninj %d\n", Ninj);
	if(Ninj<event) fprintf(stderr,"Error reading event %d from %s\n",event,LALInferenceGetProcParamVal(commandLine,"--inj")->value);
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
		REAL8TimeSeries *inj8Wave=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("injection8",
                                                                           &thisData->timeData->epoch,
                                                                           0.0,
                                                                           thisData->timeData->deltaT,
                                                                           //&lalDimensionlessUnit,
                                                                           &lalStrainUnit,
                                                                           thisData->timeData->data->length);
		if(!inj8Wave) XLAL_ERROR_VOID(XLAL_EFUNC);
		/* This marks the sample in which the real segment starts, within the buffer */
		for(i=0;i<injectionBuffer->data->length;i++) injectionBuffer->data->data[i]=0.0;
    for(i=0;i<inj8Wave->data->length;i++) inj8Wave->data->data[i]=0.0;
        INT4 realStartSample=(INT4)((thisData->timeData->epoch.gpsSeconds - injectionBuffer->epoch.gpsSeconds)/thisData->timeData->deltaT);
		realStartSample+=(INT4)((thisData->timeData->epoch.gpsNanoSeconds - injectionBuffer->epoch.gpsNanoSeconds)*1e-9/thisData->timeData->deltaT);

		/*LALSimulateCoherentGW(&status,injWave,&InjectGW,&det);*/
    //LALFindChirpInjectSignals(&status,injectionBuffer,injEvent,resp);
    if(LALInferenceGetProcParamVal(commandLine,"--lalsimulationinjection")){
      
      REAL8TimeSeries *hplus=NULL;  /**< +-polarization waveform */
      REAL8TimeSeries *hcross=NULL; /**< x-polarization waveform */
      REAL8TimeSeries       *signalvecREAL8=NULL;
      LALPNOrder        order;              /* Order of the model             */
      Approximant       approximant;        /* And its approximant value      */
      INT4              amporder=0;         /* Amplitude order of the model   */

      LALGetApproximantFromString(&status, injEvent->waveform, &approximant);
      LALGetOrderFromString(&status, injEvent->waveform, &order);
      amporder = injEvent->amp_order;
      //if(amporder<0) amporder=0;
      /* FIXME - tidal lambda's and interactionFlag are just set to command line values here.
       * They should be added to injEvent and set to appropriate values 
       */
      REAL8 lambda1 = 0.;
      if(LALInferenceGetProcParamVal(commandLine,"--inj-lambda1")) {
        lambda1= atof(LALInferenceGetProcParamVal(commandLine,"--inj-lambda1")->value);
        fprintf(stdout,"Injection lambda1 set to %f\n",lambda1);
      }
      REAL8 lambda2 = 0.;
      if(LALInferenceGetProcParamVal(commandLine,"--inj-lambda2")) {
        lambda2= atof(LALInferenceGetProcParamVal(commandLine,"--inj-lambda2")->value);
        fprintf(stdout,"Injection lambda2 set to %f\n",lambda2);
      }      
      LALSimInspiralInteraction interactionFlags = LAL_SIM_INSPIRAL_INTERACTION_ALL;
      ppt=LALInferenceGetProcParamVal(commandLine,"--inj-interactionFlags");
      if(ppt){
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_NONE")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_NONE;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_15PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_15PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_2PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_2PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_SELF_2PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_SPIN_SPIN_SELF_2PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_QUAD_MONO_2PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_QUAD_MONO_2PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_25PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_SPIN_ORBIT_25PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_TIDAL_5PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_TIDAL_5PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_TIDAL_6PN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_TIDAL_6PN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_ALL_SPIN")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_ALL_SPIN;
        if(strstr(ppt->value,"LAL_SIM_INSPIRAL_INTERACTION_ALL")) interactionFlags=LAL_SIM_INSPIRAL_INTERACTION_ALL;
      }
            
        XLALSimInspiralChooseWaveform(&hplus, &hcross, injEvent->coa_phase, thisData->timeData->deltaT,
                                                injEvent->mass1*LAL_MSUN_SI, injEvent->mass2*LAL_MSUN_SI, injEvent->spin1x,
                                                injEvent->spin1y, injEvent->spin1z, injEvent->spin2x, injEvent->spin2y,
                                                injEvent->spin2z, injEvent->f_lower, injEvent->distance*LAL_PC_SI * 1.0e6,
                                                injEvent->inclination, lambda1, lambda2, interactionFlags, 
                                                amporder, order, approximant);
      
      if(!hplus || !hcross) {
        fprintf(stderr,"Error: XLALSimInspiralChooseWaveform() failed to produce waveform.\n");
        exit(-1);
        //XLALPrintError("XLALSimInspiralChooseWaveform() failed to produce waveform.\n");
        //XLAL_ERROR_VOID(XLAL_EFUNC);
      }
      
      XLALGPSAddGPS(&(hplus->epoch), &(injEvent->geocent_end_time));
      XLALGPSAddGPS(&(hcross->epoch), &(injEvent->geocent_end_time));
      //XLALGPSAdd(&(hplus->epoch), -(REAL8)hplus->data->length*hplus->deltaT);
      //XLALGPSAdd(&(hcross->epoch), -(REAL8)hcross->data->length*hplus->deltaT);
      
      signalvecREAL8=XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, injEvent->longitude, injEvent->latitude, injEvent->polarization, det.site);      
      if (!signalvecREAL8) XLAL_ERROR_VOID(XLAL_EFUNC);      
      
      for(i=0;i<signalvecREAL8->data->length;i++){
        if(isnan(signalvecREAL8->data->data[i])) signalvecREAL8->data->data[i]=0.0;
      }
      
      if(signalvecREAL8->data->length > thisData->timeData->data->length-(UINT4)ceil((2.0*padding+2.0)/thisData->timeData->deltaT)){
        fprintf(stderr, "WARNING: waveform length = %u is longer than thisData->timeData->data->length = %d minus the window width = %d and the 2.0 seconds after tc (total of %d points available).\n", signalvecREAL8->data->length, thisData->timeData->data->length, (INT4)ceil((2.0*padding)/thisData->timeData->deltaT) , thisData->timeData->data->length-(INT4)ceil((2.0*padding+2.0)/thisData->timeData->deltaT));
        fprintf(stderr, "The waveform injected is %f seconds long. Consider increasing the %f seconds segment length (--seglen) to be greater than %f. (in %s, line %d)\n",signalvecREAL8->data->length * thisData->timeData->deltaT , thisData->timeData->data->length * thisData->timeData->deltaT, signalvecREAL8->data->length * thisData->timeData->deltaT + 2.0*padding + 2.0, __FILE__, __LINE__);
      }      
      
      XLALSimAddInjectionREAL8TimeSeries(inj8Wave, signalvecREAL8, NULL);
      
      if ( hplus ) XLALDestroyREAL8TimeSeries(hplus);
      if ( hcross ) XLALDestroyREAL8TimeSeries(hcross);
      
    }else{      
      LALInferenceLALFindChirpInjectSignals (&status,injectionBuffer,injEvent,resp,det.site);
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
      for(i=0;i<injectionBuffer->data->length;i++) inj8Wave->data->data[i]=(REAL8)injectionBuffer->data->data[i];
    }
		XLALDestroyREAL4TimeSeries(injectionBuffer);
		injF=(COMPLEX16FrequencySeries *)XLALCreateCOMPLEX16FrequencySeries("injF",
																			&thisData->timeData->epoch,
																			0.0,
																			thisData->freqData->deltaF,
																			&lalDimensionlessUnit,
																			thisData->freqData->data->length);
		if(!injF) {
            XLALPrintError("Unable to allocate memory for injection buffer\n");
            XLAL_ERROR_VOID(XLAL_EFUNC);
        }
		/* Window the data */
		REAL4 WinNorm = sqrt(thisData->window->sumofsquares/thisData->window->data->length);
		for(j=0;j<inj8Wave->data->length;j++) inj8Wave->data->data[j]*=thisData->window->data->data[j]/WinNorm;
        XLALREAL8TimeFreqFFT(injF,inj8Wave,thisData->timeToFreqFFTPlan);
		/*for(j=0;j<injF->data->length;j++) printf("%lf\n",injF->data->data[j].re);*/
		if(thisData->oneSidedNoisePowerSpectrum){
			for(SNR=0.0,j=thisData->fLow/injF->deltaF;j<injF->data->length;j++){
				SNR+=pow(injF->data->data[j].re,2.0)/thisData->oneSidedNoisePowerSpectrum->data->data[j];
				SNR+=pow(injF->data->data[j].im,2.0)/thisData->oneSidedNoisePowerSpectrum->data->data[j];
			}
            SNR*=4.0*injF->deltaF;
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


NRCSID( FINDCHIRPSIMULATIONC, "$Id$" );
//temporary? replacement function for FindChirpInjectSignals in order to accept any detector.site and not only the ones in lalCachedDetectors.
void
LALInferenceLALFindChirpInjectSignals (
    LALStatus                  *status,
    REAL4TimeSeries            *chan,
    SimInspiralTable           *events,
    COMPLEX8FrequencySeries    *resp,
    LALDetector                *LALInference_detector
    )

{
  UINT4                 k;
  DetectorResponse      detector;
  SimInspiralTable     *thisEvent = NULL;
  PPNParamStruc         ppnParams;
  CoherentGW            waveform;
  INT8                  waveformStartTime;
  REAL4TimeSeries       signalvec;
  COMPLEX8Vector       *unity = NULL;
  CHAR                  warnMsg[512];
  CHAR                  ifo[LIGOMETA_IFO_MAX];
  REAL8                 timeDelay;
  UINT4                  i; 
  REAL8TimeSeries       *hplus=NULL;
  REAL8TimeSeries       *hcross=NULL;
  REAL8TimeSeries       *signalvecREAL8=NULL;
 // REAL4TimeSeries       *signalvecREAL4=NULL;
  
  INITSTATUS( status, "LALFindChirpInjectSignals", FINDCHIRPSIMULATIONC );
  ATTATCHSTATUSPTR( status );

  ASSERT( chan, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( chan->data, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( chan->data->data, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  ASSERT( events, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );

  ASSERT( resp, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( resp->data, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );
  ASSERT( resp->data->data, status,
      FINDCHIRPH_ENULL, FINDCHIRPH_MSGENULL );


  /*
   *
   * set up structures and parameters needed
   *
   */


  /* fixed waveform injection parameters */
  memset( &ppnParams, 0, sizeof(PPNParamStruc) );
  ppnParams.deltaT   = chan->deltaT;
  ppnParams.lengthIn = 0;
  ppnParams.ppn      = NULL;


  /*
   *
   * compute the transfer function from the given response function
   *
   */


  /* allocate memory and copy the parameters describing the freq series */
  memset( &detector, 0, sizeof( DetectorResponse ) );
  detector.transfer = (COMPLEX8FrequencySeries *)
    LALCalloc( 1, sizeof(COMPLEX8FrequencySeries) );
  if ( ! detector.transfer )
  {
    ABORT( status, FINDCHIRPH_EALOC, FINDCHIRPH_MSGEALOC );
  }
  memcpy( &(detector.transfer->epoch), &(resp->epoch),
      sizeof(LIGOTimeGPS) );
  detector.transfer->f0 = resp->f0;
  detector.transfer->deltaF = resp->deltaF;

  detector.site = (LALDetector *) LALMalloc( sizeof(LALDetector) );
  /* set the detector site */
  
  detector.site = LALInference_detector;
  strcpy(ifo, LALInference_detector->frDetector.prefix);
  printf("computing waveform for %s\n",LALInference_detector->frDetector.name);

  /* set up units for the transfer function */
  {
    RAT4 negOne = { -1, 0 };
    LALUnit unit;
    LALUnitPair pair;
    pair.unitOne = &lalADCCountUnit;
    pair.unitTwo = &lalStrainUnit;
    LALUnitRaise( status->statusPtr, &unit, pair.unitTwo, &negOne );
    CHECKSTATUSPTR( status );
    pair.unitTwo = &unit;
    LALUnitMultiply( status->statusPtr, &(detector.transfer->sampleUnits),
        &pair );
    CHECKSTATUSPTR( status );
  }

  /* invert the response function to get the transfer function */
  LALCCreateVector( status->statusPtr, &( detector.transfer->data ),
      resp->data->length );
  CHECKSTATUSPTR( status );

  LALCCreateVector( status->statusPtr, &unity, resp->data->length );
  CHECKSTATUSPTR( status );
  for ( k = 0; k < resp->data->length; ++k )
  {
    unity->data[k].re = 1.0;
    unity->data[k].im = 0.0;
  }

  LALCCVectorDivide( status->statusPtr, detector.transfer->data, unity,
      resp->data );
  CHECKSTATUSPTR( status );

  LALCDestroyVector( status->statusPtr, &unity );
  CHECKSTATUSPTR( status );


  /*
   *
   * loop over the signals and inject them into the time series
   *
   */


  for ( thisEvent = events; thisEvent; thisEvent = thisEvent->next )
  {
    /*
     *
     * generate waveform and inject it into the data
     *
     */


    /* clear the waveform structure */
    memset( &waveform, 0, sizeof(CoherentGW) );
    
    LALGenerateInspiral(status->statusPtr, &waveform, thisEvent, &ppnParams );
    CHECKSTATUSPTR( status );

    LALInfo( status, ppnParams.termDescription );

    if ( strstr( thisEvent->waveform, "KludgeIMR") ||
         strstr( thisEvent->waveform, "KludgeRingOnly") )
     {
       CoherentGW *wfm;
       SimRingdownTable *ringEvent;
       int injectSignalType = LALRINGDOWN_IMR_INJECT;


       ringEvent = (SimRingdownTable *)
         LALCalloc( 1, sizeof(SimRingdownTable) );
       wfm = XLALGenerateInspRing( &waveform, thisEvent, ringEvent,
           injectSignalType);
       LALFree(ringEvent);

       if ( !wfm )
       {
         LALInfo( status, "Unable to generate merger/ringdown, "
             "injecting inspiral only");
         ABORT( status, FINDCHIRPH_EIMRW, FINDCHIRPH_MSGEIMRW );
       }
       waveform = *wfm;
     }


    if ( thisEvent->geocent_end_time.gpsSeconds )
    {
      /* get the gps start time of the signal to inject */
      waveformStartTime = XLALGPSToINT8NS( &(thisEvent->geocent_end_time) );
      waveformStartTime -= (INT8) ( 1000000000.0 * ppnParams.tc );
    }
    else
    {
      LALInfo( status, "Waveform start time is zero: injecting waveform "
          "into center of data segment" );

      /* center the waveform in the data segment */
      waveformStartTime = XLALGPSToINT8NS( &(chan->epoch) );

      waveformStartTime += (INT8) ( 1000000000.0 *
          ((REAL8) (chan->data->length - ppnParams.length) / 2.0) * chan->deltaT
          );
    }

    snprintf( warnMsg, sizeof(warnMsg)/sizeof(*warnMsg),
        "Injected waveform timing:\n"
        "thisEvent->geocent_end_time.gpsSeconds = %d\n"
        "thisEvent->geocent_end_time.gpsNanoSeconds = %d\n"
        "ppnParams.tc = %e\n"
        "waveformStartTime = %" LAL_INT8_FORMAT "\n",
        thisEvent->geocent_end_time.gpsSeconds,
        thisEvent->geocent_end_time.gpsNanoSeconds,
        ppnParams.tc,
        waveformStartTime );
    LALInfo( status, warnMsg );

      /* clear the signal structure */
      memset( &signalvec, 0, sizeof(REAL4TimeSeries) );

      /* set the start time of the signal vector to the appropriate start time of the injection */
      if ( detector.site )
      {
        timeDelay = XLALTimeDelayFromEarthCenter( detector.site->location, thisEvent->longitude,
          thisEvent->latitude, &(thisEvent->geocent_end_time) );
        if ( XLAL_IS_REAL8_FAIL_NAN( timeDelay ) )
        {
          ABORTXLAL( status );
        }
      }
      else
      {
        timeDelay = 0.0;
      }
      /* Give a little more breathing space to aid band-passing */
      XLALGPSSetREAL8( &(signalvec.epoch), (waveformStartTime * 1.0e-9) - 0.25 + timeDelay );
      /* set the parameters for the signal time series */
      signalvec.deltaT = chan->deltaT;
      if ( ( signalvec.f0 = chan->f0 ) != 0 )
      {
        ABORT( status, FINDCHIRPH_EHETR, FINDCHIRPH_MSGEHETR );
      }
      signalvec.sampleUnits = lalADCCountUnit;
      
      if(waveform.h == NULL){
      /* set the start times for injection */
      XLALINT8NSToGPS( &(waveform.a->epoch), waveformStartTime );
      /* put a rug on a polished floor? */
      waveform.f->epoch = waveform.a->epoch;
      waveform.phi->epoch = waveform.a->epoch;
      /* you might as well set a man trap */
      if ( waveform.shift )
      {
        waveform.shift->epoch = waveform.a->epoch;
      }
      /* and to think he'd just come from the hospital */
      }else{
        /* set the start times for injection */
        XLALINT8NSToGPS( &(waveform.h->epoch), waveformStartTime );  
      }
      /* simulate the detectors response to the inspiral */
      LALSCreateVector( status->statusPtr, &(signalvec.data), chan->data->length );
      CHECKSTATUSPTR( status );

      if(waveform.h == NULL){ //LALSimulateCoherentGW only for waveform generators filling CoherentGW.a and CoherentGW.phi
        LALSimulateCoherentGW( status->statusPtr, &signalvec, &waveform, &detector );
      }else{
      hplus=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("hplus",
                                                                &(waveform.h->epoch),
                                                                0.0,
                                                                waveform.h->deltaT,
                                                                &lalDimensionlessUnit,
                                                                waveform.h->data->length);

      hcross=(REAL8TimeSeries *)XLALCreateREAL8TimeSeries("hcross",
                                                                  &(waveform.h->epoch),
                                                                  0.0,
                                                                  waveform.h->deltaT,
                                                                  &lalDimensionlessUnit,
                                                                  waveform.h->data->length);
      for( i = 0; i < waveform.h->data->length; i++)
      {
        hplus->data->data[i] = waveform.h->data->data[2*i];
        hcross->data->data[i] = waveform.h->data->data[(2*i)+1];
      }

      signalvecREAL8=XLALSimDetectorStrainREAL8TimeSeries(hplus, 
                                                          hcross,
                                                          thisEvent->longitude,
                                                          thisEvent->latitude,
                                                          thisEvent->polarization,
                                                          LALInference_detector);
        
      INT8 offset = ( signalvecREAL8->epoch.gpsSeconds - signalvec.epoch.gpsSeconds ) / signalvec.deltaT;
      offset += ( signalvecREAL8->epoch.gpsNanoSeconds - signalvec.epoch.gpsNanoSeconds ) * 1.0e-9 / signalvec.deltaT;

      for (i=0; i<signalvec.data->length; i++){
        if(i<offset || i>=signalvecREAL8->data->length+offset || isnan(signalvecREAL8->data->data[i-offset])) signalvec.data->data[i]=0.0; //The isnan() condition should not be necessary. To be investigated.
				else signalvec.data->data[i]=(REAL4) signalvecREAL8->data->data[i-offset];
			}
      }
      CHECKSTATUSPTR( status );

      /* Taper the signal */
      {

          if ( ! strcmp( "TAPER_START", thisEvent->taper ) )
          {
              XLALInspiralWaveTaper( signalvec.data, INSPIRAL_TAPER_START );
          }
          else if (  ! strcmp( "TAPER_END", thisEvent->taper ) )
          {
              XLALInspiralWaveTaper( signalvec.data, INSPIRAL_TAPER_END );
          }
          else if (  ! strcmp( "TAPER_STARTEND", thisEvent->taper ) )
          {
              XLALInspiralWaveTaper( signalvec.data, INSPIRAL_TAPER_STARTEND );
          }
          else if ( strcmp( "TAPER_NONE", thisEvent->taper ) )
          {
              XLALPrintError( "Invalid injection tapering option specified: %s\n",
                 thisEvent->taper );
              ABORT( status, LAL_BADPARM_ERR, LAL_BADPARM_MSG );
          }
      }
      /* Band pass the signal */
      if ( thisEvent->bandpass )
      {
          UINT4 safeToBandPass = 0;
          UINT4 start=0, end=0;
          REAL4Vector *bandpassVec = NULL;

          safeToBandPass = FindTimeSeriesStartAndEnd (
                  signalvec.data, &start, &end );

          if ( safeToBandPass )
          {
              /* Check if we can grab some padding at the extremeties.
               * This will make the bandpassing better
               */

              if (((INT4)start - (int)(0.25/chan->deltaT)) > 0 )
                    start -= (int)(0.25/chan->deltaT);
              else
                    start = 0;

              if ((end + (int)(0.25/chan->deltaT)) < signalvec.data->length )
                    end += (int)(0.25/chan->deltaT);
              else
                    end = signalvec.data->length - 1;

              bandpassVec = (REAL4Vector *)
                      LALCalloc(1, sizeof(REAL4Vector) );

              bandpassVec->length = (end - start + 1);
              bandpassVec->data = signalvec.data->data + start;

              if ( XLALBandPassInspiralTemplate( bandpassVec,
                          1.1*thisEvent->f_lower,
                          1.05*thisEvent->f_final,
                          1./chan->deltaT) != XLAL_SUCCESS )
              {
                  LALError( status, "Failed to Bandpass signal" );
                  ABORT (status, LALINSPIRALH_EBPERR, LALINSPIRALH_MSGEBPERR);
              };

              LALFree( bandpassVec );
          }
      }
      /* inject the signal into the data channel */
      LALSSInjectTimeSeries( status->statusPtr, chan, &signalvec );

      CHECKSTATUSPTR( status );


    if ( waveform.shift )
    {
      LALSDestroyVector( status->statusPtr, &(waveform.shift->data) );
      CHECKSTATUSPTR( status );
      LALFree( waveform.shift );
    }

    if( waveform.h )
    {
      LALSDestroyVectorSequence( status->statusPtr, &(waveform.h->data) );
      CHECKSTATUSPTR( status );
      LALFree( waveform.h );
    }
    if( waveform.a )
    {
      LALSDestroyVectorSequence( status->statusPtr, &(waveform.a->data) );
      CHECKSTATUSPTR( status );
      LALFree( waveform.a );
      /*
       * destroy the signal only if waveform.h is NULL as otherwise it won't
       * be created
       * */
      if ( waveform.h == NULL )
      {
	LALSDestroyVector( status->statusPtr, &(signalvec.data) );
        CHECKSTATUSPTR( status );
      }
    }
    if( waveform.f )
    {
      LALSDestroyVector( status->statusPtr, &(waveform.f->data) );
      CHECKSTATUSPTR( status );
      LALFree( waveform.f );
    }
    if( waveform.phi )
    {
      LALDDestroyVector( status->statusPtr, &(waveform.phi->data) );
      CHECKSTATUSPTR( status );
      LALFree( waveform.phi );
    }
  }

  
  if(hplus) XLALDestroyREAL8TimeSeries(hplus);
  if(hcross) XLALDestroyREAL8TimeSeries(hcross);
  if(signalvecREAL8) XLALDestroyREAL8TimeSeries(signalvecREAL8);
  
  LALCDestroyVector( status->statusPtr, &( detector.transfer->data ) );
  CHECKSTATUSPTR( status );

//  if ( detector.site ) LALFree( detector.site );
  LALFree( detector.transfer );

  DETATCHSTATUSPTR( status );
  RETURN( status );
}

static int FindTimeSeriesStartAndEnd (
                                      REAL4Vector *signalvec,
                                      UINT4 *start,
                                      UINT4 *end
                                      )
{
  UINT4 i; /* mid, n; indices */
  UINT4 flag, safe = 1;
  UINT4 length;
  
#ifndef LAL_NDEBUG
  if ( !signalvec )
    XLAL_ERROR( XLAL_EFAULT );
  
  if ( !signalvec->data )
    XLAL_ERROR( XLAL_EFAULT );
#endif
  
  length = signalvec->length;
  
  /* Search for start and end of signal */
  flag = 0;
  i = 0;
  while(flag == 0 && i < length )
  {
    if( signalvec->data[i] != 0.)
    {
      *start = i;
      flag = 1;
    }
    i++;
  }
  if ( flag == 0 )
  {
    return flag;
  }
  
  flag = 0;
  i = length - 1;
  while(flag == 0)
  {
    if( signalvec->data[i] != 0.)
    {
      *end = i;
      flag = 1;
    }
    i--;
  }
  
  /* Check we have more than 2 data points */
  if(((*end) - (*start)) <= 1)
  {
    XLALPrintWarning( "Data less than 3 points in this signal!\n" );
    safe = 0;
  }
  
  return safe;
  
}



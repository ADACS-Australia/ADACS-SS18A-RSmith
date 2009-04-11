/**********************************************************************
*        Copyright (C) 2009 John Veitch, Stephen Fairhurst
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
**********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <config.h>
#include <math.h>
#include <getopt.h>
#include <string.h>

#include <lalapps.h>
#include <processtable.h>
#include <lal/LALStdio.h>
#include <lal/LALStdlib.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/Date.h>
#include <lal/FindChirp.h>
#include <lal/Units.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeSeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/InspiralInjectionParams.h>
#include <lal/VectorOps.h>
#include <lal/FrameStream.h>

#define PROGRAM_NAME "coinj"

#define USAGE \
"lalpps_coinj [options]\n \
--help                       display this message \n \
--input <injection.xml>      Specify input SimInspiralTable xml file\n\
--response-type TYPE         TYPE of injection, [ strain | etmx | etmy ]\n\
--frames                     Create h(t) frame files\n\n\
lalapps_coinj: create coherent injection files for LIGO and VIRGO\n"


RCSID("$Id");

extern int vrbflg;
extern int lalDebugLevel;

typedef enum
{
	noResponse,
	unityResponse,
	design,
	actuationX,
	actuationY
} ResponseType;

typedef struct actuationparameters
{
	REAL4	ETMXcal;
	REAL4	ETMYcal;
	REAL4	pendFX;
	REAL4	pendFY;
	REAL4	pendQX;
	REAL4	pendQY;
	REAL4	length;
} ActuationParametersType;

typedef void (NoiseFunc)(LALStatus *status,REAL8 *psd,REAL8 f);

int main(int argc, char *argv[])
{
LALStatus	status=blank_status;
CHAR		inputfile[FILENAME_MAX];
CHAR		injtype[30];
CHAR		det_name[10];
LIGOTimeGPS inj_epoch;
REAL8		deltaT= 1.0/16384.0;
REAL8		injLength=100.0; /* Ten seconds at end */
REAL8		LeadupTime=95.0;
REAL8		dynRange=1.0/3.0e-23;
REAL8		tmp;
UINT4		Nsamples,det_idx,i,inj_num=0;
ActuationParametersType actuationParams[LAL_NUM_IFO];
ActuationParametersType actData;
ResponseType injectionResponse=noResponse;
FILE *		outfile;
CHAR		outfilename[FILENAME_MAX];
const LALUnit strainPerCount={0,{0,0,0,0,0,1,-1},{0,0,0,0,0,0,0}};
const LALUnit countPerStrain={0,{0,0,0,0,0,-1,1},{0,0,0,0,0,0,0}};
NoiseFunc *PSD;
int c;
SimInspiralTable *injTable=NULL;
SimInspiralTable this_injection;
REAL4TimeSeries *TimeSeries;
REAL4TimeSeries *SNRTimeSeries;
REAL4TimeSeries *actuationTimeSeries;
COMPLEX8FrequencySeries *resp;
COMPLEX8FrequencySeries *actuationResp;
COMPLEX8FrequencySeries *transfer;
COMPLEX8Vector *unity;
COMPLEX8FrequencySeries *DesignNoise;
FrOutPar	VirgoOutPars;
CHAR		VirgoParsSource[100];
CHAR		VirgoParsInfo[100];
REAL4		SNR,NetworkSNR;
INT4  makeFrames=0;


/*vrbflg=6;
lalDebugLevel=6; */

struct option long_options[]=
 {
	{"help", no_argument, 0, 'h'},
	{"input",required_argument,0, 'i'},
	{"response-type",required_argument,0,'r'},
	{"frames",no_argument,&makeFrames,'F'},
	{"verbose",no_argument,&vrbflg,1},
	{0,0,0,0}
 };

  /*taken from Calibration CVS file: 
	* calibration/frequencydomain/runs/S5/H1/model/V3/H1DARMparams_849677446.m */
  actuationParams[LAL_IFO_H1].ETMXcal = -0.795e-9;
  actuationParams[LAL_IFO_H1].pendFX  = 0.767;
  actuationParams[LAL_IFO_H1].pendQX  = 10.0;
  actuationParams[LAL_IFO_H1].ETMYcal = -0.827e-9;
  actuationParams[LAL_IFO_H1].pendFY  = 0.761;
  actuationParams[LAL_IFO_H1].pendQY  = 10.0;
  actuationParams[LAL_IFO_H1].length  = 4000.0;

  /*taken from Calibration CVS file: 
   * calibration/frequencydomain/runs/S5/H2/model/V3/H2DARMparams_849678155.m */
  actuationParams[LAL_IFO_H2].ETMXcal = -0.876e-9;
  actuationParams[LAL_IFO_H2].pendFX  = 0.749;
  actuationParams[LAL_IFO_H2].pendQX  = 10.0;
  actuationParams[LAL_IFO_H2].ETMYcal = -0.912e-9;
  actuationParams[LAL_IFO_H2].pendFY  = 0.764;
  actuationParams[LAL_IFO_H2].pendQY  = 10.0;
  actuationParams[LAL_IFO_H2].length  = 2000.0;

  /*taken from Calibration CVS file: 
   * calibration/frequencydomain/runs/S5/L1/model/V3/L1DARMparams_841930071.m */
  actuationParams[LAL_IFO_L1].ETMXcal = -0.447e-9;
  actuationParams[LAL_IFO_L1].pendFX  = 0.766;
  actuationParams[LAL_IFO_L1].pendQX  = 100.0;
  actuationParams[LAL_IFO_L1].ETMYcal = -0.438e-9;
  actuationParams[LAL_IFO_L1].pendFY  = 0.756;
  actuationParams[LAL_IFO_L1].pendQY  = 100.0;
  actuationParams[LAL_IFO_L1].length  = 4000.0;


/*******************************************************************************/

/* Process input arguments */
while(1)
{
	int option_idx=0;
	c=getopt_long_only(argc,argv,"hFi:",long_options,&option_idx);
	if(c==-1) break;
	switch(c)
	{
		case 'h':
			fprintf(stderr,USAGE);
			exit(0);
			break;
		case 'i':
		  strncpy(inputfile,optarg,FILENAME_MAX-1);
			break;
		case 'r':
			if(!strcmp("strain",optarg))	  injectionResponse = unityResponse;
			else if(!strcmp("etmx",optarg))   injectionResponse = actuationX;
			else if(!strcmp("etmy",optarg))   injectionResponse = actuationY;
			else {fprintf(stderr,"Invalid argument to response-type: %s\nResponse type must be strain, etmy or etmx\n",\
						  optarg); exit(1);}
			break;
	}
}

memset(&status,0,sizeof(status));

/* Read in the input XML */
SimInspiralTableFromLIGOLw(&injTable,inputfile,0,0);
Nsamples = (UINT4)injLength/deltaT;

do{
memcpy(&this_injection,injTable,sizeof(SimInspiralTable));
this_injection.next=NULL;
injTable=injTable->next;
NetworkSNR=0.0;
/* Set epoch */
memcpy(&inj_epoch,&(this_injection.geocent_end_time),sizeof(LIGOTimeGPS));
LAL_CALL(LALAddFloatToGPS(&status,&inj_epoch,&(this_injection.geocent_end_time),-LeadupTime),&status);
inj_epoch.gpsNanoSeconds=0;

/* Loop over detectors */
for(det_idx=0;det_idx<LAL_NUM_IFO;det_idx++){
	if(det_idx==LAL_IFO_T1||det_idx==LAL_IFO_G1) continue; /* Don't generate for GEO or TAMA */
	if(det_idx==LAL_IFO_V1 && injectionResponse!=unityResponse){
		fprintf(stdout,"Skipping generation of non-strain injection for VIRGO\n");
		continue;
	}
	
	switch(det_idx)
	{
		case LAL_IFO_H1: sprintf(det_name,"H1"); PSD=&LALLIGOIPsd; break;
		case LAL_IFO_H2: sprintf(det_name,"H2"); PSD=&LALLIGOIPsd; break;
		case LAL_IFO_L1: sprintf(det_name,"L1"); PSD=&LALLIGOIPsd; break;
		case LAL_IFO_V1: sprintf(det_name,"V1"); PSD=&LALVIRGOPsd; break;
		case LAL_IFO_G1: sprintf(det_name,"G1"); PSD=&LALGEOPsd;   break;
		case LAL_IFO_T1: sprintf(det_name,"T1"); PSD=&LALTAMAPsd;  break;
	}

	TimeSeries=XLALCreateREAL4TimeSeries(det_name,&inj_epoch,0.0,deltaT,&lalADCCountUnit,(size_t)Nsamples);
	for(i=0;i<Nsamples;i++) TimeSeries->data->data[i]=0.0;
	resp = XLALCreateCOMPLEX8FrequencySeries("response",&inj_epoch,0.0,1.0/injLength,&strainPerCount,(size_t)Nsamples/2+1);
	for(i=0;i<resp->data->length;i++) {resp->data->data[i].re=(REAL4)1.0/dynRange; resp->data->data[i].im=0.0;}

	/* Create h(t) time series for this detector */
	LAL_CALL( LALFindChirpInjectSignals(&status,TimeSeries,&this_injection,resp) , &status);

	XLALDestroyCOMPLEX8FrequencySeries(resp);

	/* -=-=-=-=-=-=- Prepare actuations -=-=-=-=-=-=- */

	if(injectionResponse==actuationX || injectionResponse==actuationY) actData=actuationParams[det_idx];
	actuationResp = XLALCreateCOMPLEX8FrequencySeries("actuationResponse",&inj_epoch,0.0,1.0/(2.0*injLength),&strainPerCount,(size_t)Nsamples+1);
	/* Create actuation response */
	switch(injectionResponse){
		case unityResponse:
			sprintf(injtype,"STRAIN");
			for(i=0;i<actuationResp->data->length;i++){actuationResp->data->data[i].re=1.0; actuationResp->data->data[i].im=0.0;}
			break;
		case actuationX:
			sprintf(injtype,"ETMX");
			actuationResp=generateActuation(actuationResp,actData.ETMXcal/actData.length,actData.pendFX,actData.pendQX);
			break;
		case actuationY:
			sprintf(injtype,"ETMY");
			actuationResp=generateActuation(actuationResp,actData.ETMYcal/actData.length,actData.pendFY,actData.pendQY);
			break;
		default:
			fprintf(stderr,"Must specify response function: strain, etmy or etmx\n"); exit(1);
			break;
	}


       	if(injectionResponse!=unityResponse) {
	  	actuationTimeSeries=XLALCreateREAL4TimeSeries(det_name,&inj_epoch,0.0,deltaT,&lalADCCountUnit,(size_t)Nsamples);
		unity = XLALCreateCOMPLEX8Vector(actuationResp->data->length);
		transfer = XLALCreateCOMPLEX8FrequencySeries("transfer",&inj_epoch,0.0,1.0/(2.0*injLength),&countPerStrain,(size_t)Nsamples+1);
		for(i=0;i<unity->length;i++) {unity->data[i].re=1.0; unity->data[i].im=0.0;}
		XLALCCVectorDivide(transfer->data,unity,resp->data);
		actuationTimeSeries = XLALRespFilt(TimeSeries,transfer);
		XLALDestroyCOMPLEX8FrequencySeries(transfer);
		XLALDestroyCOMPLEX8Vector(unity);
		for(i=0;i<actuationTimeSeries->data->length;i++) actuationTimeSeries->data->data[i]/=dynRange;
	}
	else actuationTimeSeries=TimeSeries;

	XLALDestroyCOMPLEX8FrequencySeries(actuationResp);

	for(i=0;i<TimeSeries->data->length;i++) {
	  TimeSeries->data->data[i]=TimeSeries->data->data[i]/dynRange +0.0;
	}
	
	sprintf(outfilename,"%s_HWINJ_%i_%s_%i.txt",det_name,inj_num,injtype,inj_epoch.gpsSeconds);
	outfile=fopen(outfilename,"w");
	fprintf(stdout,"Injected signal %i for %s into file %s\n",inj_num,det_name,outfilename);
	for(i=0;i<actuationTimeSeries->data->length;i++) fprintf(outfile,"%10.10e\n",actuationTimeSeries->data->data[i]);
	fclose(outfile);

	if(makeFrames){ /* Also output frames for Virgo */
		sprintf(VirgoParsSource,"%s-INSP%i",det_name,inj_num);
		VirgoOutPars.source=VirgoParsSource;
		sprintf(VirgoParsInfo,"HWINJ-STRAIN",injtype);
		VirgoOutPars.description=VirgoParsInfo;
		VirgoOutPars.type=ProcDataChannel;
		VirgoOutPars.nframes=(UINT4)injLength;
		VirgoOutPars.frame=0;
		VirgoOutPars.run=2;
		fprintf(stdout,"Generating frame file for %s-%s-%i\n",VirgoParsSource,VirgoParsInfo,TimeSeries->epoch.gpsSeconds);
		LALFrWriteREAL4TimeSeries(&status,TimeSeries,&VirgoOutPars);
	}
	
	XLALDestroyREAL4TimeSeries(TimeSeries);
	if(injectionResponse!=unityResponse) XLALDestroyREAL4TimeSeries(actuationTimeSeries);

} /* End loop over detectors */
/*fprintf(stdout,"Finished injecting signal %i, network SNR %f\n",inj_num,sqrt(NetworkSNR));*/
inj_num++;
}while(injTable!=NULL);
return(0);
}

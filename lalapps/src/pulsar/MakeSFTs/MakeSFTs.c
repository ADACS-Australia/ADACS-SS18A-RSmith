/*********************************************************************************/
/*                                                                               */
/* File: MakeSFTs.c                                                              */
/* Purpose: generate SFTs                                                        */
/* Origin: first written by Xavier Siemens, UWM - May 2005,                      */
/*         based on make_sfts.c by Bruce Allen                                   */
/* $Id$                     */
/*********************************************************************************/

/* REVISIONS: */
/* 11/02/05 gam; To save memory, change lalDebugLevel to 0; note when this is set to 3 that 3x the memory is used! */
/* 11/02/05 gam; To save memory, do not save the window function in memory; just recompute this for each SFT. */
/* 11/02/05 gam; To save memory, do not hold dataSingle.data and vtilde in memory at the same time. */
/* 11/03/05 gam; Add TRACKMEMUSE preprocessor flag and function for tracking memory usage; copied from make_sfts.c */
/* 11/19/05 gam; Reorganize code so that function appear in order called */
/* 11/19/05 gam; Add command line option to use single rather than double precision; will use LALDButterworthREAL4TimeSeries in single precision case. */
/* 11/19/05 gam; Rename vtilde fftDataDouble; add in fftDatasingle; rename fplan fftPlanDouble; add in fftPlanSingle */
/* 11/29/05 gam; Add PRINTEXAMPLEDATA to print the first NUMTOPRINT, middle NUMTOPRINT, and last NUMTOPRINT input/ouput data at various stages */
/* 12/27/05 gam; Add option make-gps-dirs, -D <num>, to make directory based on this many GPS digits. */
/* 12/28/05 gam; Add option misc-desc, -X <string> giving misc. part of the SFT description field in the filename */
/* 12/28/05 gam; Make FMIN = 48.0 and DF = 2000.0 global variables, add start-freq -F and band -B options to enter these */
/* 12/28/05 gam; Add in window-type options; 0 = no window, 1 = default = Matlab style Tukey window; 2 = make_sfts.c Tukey window; 3 = Hann window */
/* 12/28/05 gam; Add option --overlap-fraction -P (for use with windows; e.g., use -P 0.5 with -w 3 Hann windows; default is 0.0) */
/* 12/28/05 gam; Add sft-version, -v option to select output SFT version (1 = default is version 1 SFTs; 2 = version 2 SFTs */
/* 12/28/05 gam; Add comment-field, -c option, for comment for version 2 SFTs */
/* 01/05/06 gam; Add in version 2 normalization; add function to print example version 2 SFT data; add memory checking */
/* 01/09/06 gam; Add make-tmp-file, -Z option; write SFT to .*.tmp file, then move to final file name. */

#include <config.h>
#if !defined HAVE_LIBGSL || !defined HAVE_LIBLALFRAME
#include <stdio.h>
int main(void) {fputs("disabled, no gsl or no lal frame library support.\n", stderr);return 1;}
#else

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <glob.h>
#include <errno.h>
#include <getopt.h>
#include <stdarg.h>

#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/LALStdio.h>
#include <lal/FileIO.h>
#include <lal/AVFactories.h>
#include <lal/FrameCache.h>
#include <lal/FrameStream.h>
#include <lal/Window.h>
#include <lal/Calibration.h>
#include <lal/LALConstants.h>
#include <lal/BandPassTimeSeries.h>
#include <lal/LALStdlib.h>
#include <lal/LALConstants.h>
#include <lal/AVFactories.h>
#include <lal/TimeFreqFFT.h>
#include <lal/RealFFT.h>
#include <lal/ComplexFFT.h>
#include <lal/SFTfileIO.h>

extern char *optarg;
extern int optind, opterr, optopt;

/* track memory usage under linux */
#define TRACKMEMUSE 0

/* print the first NUMTOPRINT, middle NUMTOPRINT, and last NUMTOPRINT input/ouput data at various stages */
#define PRINTEXAMPLEDATA 0
#define NUMTOPRINT       2

#define TESTSTATUS( pstat ) \
  if ( (pstat)->statusCode ) { REPORTSTATUS(pstat); return 100; } else ((void)0)

/* This repeatedly tries to re-open a file.  This is useful on a
   cluster because automount may fail or time out.  This allows a
   number of tries with a bit of rest in between.
*/
FILE* tryopen(char *name, char *mode){
  int count=0;
  FILE *fp;
  
  while (!(fp=fopen(name, mode))){
    fprintf(stderr,"Unable to open file %s in mode %s.  Will retry...\n", name, mode);
    if (count++<10)
      sleep(10);
    else
      exit(3);
  }
  return fp;
}

/***************************************************************************/

/* STRUCTURES */
struct CommandLineArgsTag {
  REAL8 HPf;              /* High pass filtering frequency */
  INT4 T;                 /* SFT duration */
  char *stringT;          /* 12/27/05 gam; string with SFT duration */
  INT4 GPSStart;
  INT4 GPSEnd;
  INT4 makeGPSDirs;        /* 12/27/05 gam; add option to make directories based on gps time */
  INT4 sftVersion;         /* 12/28/05 gam; output SFT version */
  char *commentField;      /* 12/28/05 gam; string comment for version 2 SFTs */
  BOOLEAN htdata;          /* flag that indicates we're doing h(t) data */
  BOOLEAN makeTmpFile;     /* 01/09/06 gam */
  char *FrCacheFile;       /* Frame cache file */
  char *ChannelName;
  char *SFTpath;           /* path to SFT file location */
  char *miscDesc;          /* 12/28/05 gam; string giving misc. part of the SFT description field in the filename */
  INT4 windowOption;       /* 12/28/05 gam; window options; 0 = no window, 1 = default = Matlab style Tukey window; 2 = make_sfts.c Tukey window; 3 = Hann window */
  REAL8 overlapFraction;   /* 12/28/05 gam; overlap fraction (for use with windows; e.g., use -P 0.5 with -w 3 Hann windows; default is 1.0). */
  BOOLEAN useSingle;       /* 11/19/05 gam; use single rather than double precision */
} CommandLineArgs;

struct headertag {
  REAL8 endian;
  INT4  gps_sec;
  INT4  gps_nsec;
  REAL8 tbase;
  INT4  firstfreqindex;
  INT4  nsamples;
} header;


/***************************************************************************/

/* GLOBAL VARIABLES */
REAL8 FMIN = 48.0; /* default start frequency */
REAL8 DF = 2000.0; /* default band */

static LALStatus status;
INT4 lalDebugLevel=0;        /* 11/02/05 gam; changed from 3 to 0. */
FrCache *framecache;         /* frame reading variables */
FrStream *framestream=NULL;

REAL8TimeSeries dataDouble;
REAL4TimeSeries dataSingle;
INT4 SegmentDuration;
LIGOTimeGPS gpsepoch;

/* REAL8Vector *window; */ /* 11/02/05 gam */

REAL8FFTPlan *fftPlanDouble;           /* fft plan and data container, double precision case */
COMPLEX16Vector *fftDataDouble = NULL;

REAL4FFTPlan *fftPlanSingle;           /* 11/19/05 gam; fft plan and data container, single precision case */
COMPLEX8Vector *fftDataSingle = NULL;

/***************************************************************************/

/* FUNCTION PROTOTYPES */
/* Reads the command line */
int ReadCommandLine(int argc,char *argv[],struct CommandLineArgsTag *CLA);

/* Allocates space for data time series */
int AllocateData(struct CommandLineArgsTag CLA);

/* Reads data */
int ReadData(struct CommandLineArgsTag CLA);

/* High passes data */
int HighPass(struct CommandLineArgsTag CLA);

/* Windows data */
int WindowData(struct CommandLineArgsTag CLA);
int WindowDataTukey2(struct CommandLineArgsTag CLA);
int WindowDataHann(struct CommandLineArgsTag CLA);

/* create an SFT */
int CreateSFT(struct CommandLineArgsTag CLA);

/* writes out an SFT */
int WriteSFT(struct CommandLineArgsTag CLA);
int WriteVersion2SFT(struct CommandLineArgsTag CLA);

/* Frees the memory */
int FreeMem(struct CommandLineArgsTag CLA);

/* 12/27/05 gam; function to create sftDescField for output directory or filename. */
void getSFTDescField(CHAR *sftDescField, CHAR *numSFTs, CHAR *ifo, CHAR *stringT, CHAR *typeMisc) {
       strcpy(sftDescField,numSFTs);
       strcat(sftDescField, "_");
       strcat(sftDescField,ifo);
       strcat(sftDescField, "_");
       strcat(sftDescField,stringT);
       strcat(sftDescField, "SFT");
       if (typeMisc != NULL) {
          strcat(sftDescField, "_");
          strcat(sftDescField, typeMisc);
       }
}

/* 12/27/05 gam; concat to the sftPath the directory name based on GPS time; make this directory if it does not already exist */
void mkSFTDir(CHAR *sftPath, CHAR *site, CHAR *numSFTs, CHAR *ifo, CHAR *stringT, CHAR *typeMisc,CHAR *gpstime, INT4 numGPSdigits) {
     CHAR sftDescField[256];
     CHAR mkdirCommand[256];
     strcat(sftPath,"/");
     strcat(sftPath,site);
     strcat(sftPath,"-");
     getSFTDescField(sftDescField, numSFTs, ifo, stringT, typeMisc);
     strcat(sftPath,sftDescField);
     strcat(sftPath,"-");
     strncat(sftPath,gpstime,numGPSdigits);
     sprintf(mkdirCommand,"mkdir -p %s",sftPath);
     system(mkdirCommand);
}

/* 12/27/05 gam; make SFT file name according to LIGO T040164-01 specification */
void mkSFTFilename(CHAR *sftFilename, CHAR *site, CHAR *numSFTs, CHAR *ifo, CHAR *stringT, CHAR *typeMisc,CHAR *gpstime) {
     CHAR sftDescField[256];
     strcpy(sftFilename,site);
     strcat(sftFilename,"-");
     getSFTDescField(sftDescField, numSFTs, ifo, stringT, typeMisc);
     strcat(sftFilename,sftDescField);
     strcat(sftFilename,"-");
     strcat(sftFilename,gpstime);
     strcat(sftFilename,"-");
     strcat(sftFilename,stringT);
     strcat(sftFilename,".sft");
}

/* 01/09/06 gam; move filename1 to filename2 */
void mvFilenames(CHAR *filename1, CHAR *filename2) {
     CHAR mvFilenamesCommand[512];
     sprintf(mvFilenamesCommand,"mv %s %s",filename1,filename2);
     system(mvFilenamesCommand);
}

#if TRACKMEMUSE
void printmemuse() {
   pid_t mypid=getpid();
   char commandline[256];
   fflush(NULL);
   sprintf(commandline,"cat /proc/%d/status | /bin/grep Vm | /usr/bin/fmt -140 -u", (int)mypid);
   system(commandline);
   fflush(NULL);
 }
#endif

#if PRINTEXAMPLEDATA
void printExampleDataSingle() {
      INT4 i,dataLength;
      dataLength = dataSingle.data->length;
      fprintf(stdout,"dataSingle.deltaT, 1.0/dataSingle.deltaT = %23.16e, %23.16e\n",dataSingle.deltaT,1.0/dataSingle.deltaT);
      fprintf(stdout,"dataSingle.epoch.gpsSeconds,dataSingle.epoch.gpsNanoSeconds = %i, %i\n",dataSingle.epoch.gpsSeconds,dataSingle.epoch.gpsNanoSeconds);
      for (i=0;i<NUMTOPRINT;i++) {
        fprintf(stdout,"%i %23.16e\n",i,dataSingle.data->data[i]);
      }
      for (i=dataLength/2;i<dataLength/2+NUMTOPRINT;i++) {
        fprintf(stdout,"%i %23.16e\n",i,dataSingle.data->data[i]);
      }
      for (i=dataLength-NUMTOPRINT;i<dataLength;i++) {
        fprintf(stdout,"%i %23.16e\n",i,dataSingle.data->data[i]);
      }
      fflush(stdout);
}    

void printExampleDataDouble() {
      INT4 i,dataLength;
      dataLength = dataDouble.data->length;
      fprintf(stdout,"dataDouble.deltaT, 1.0/dataDouble.deltaT = %23.16e, %23.16e\n",dataDouble.deltaT,1.0/dataDouble.deltaT);
      fprintf(stdout,"dataDouble.epoch.gpsSeconds,dataDouble.epoch.gpsNanoSeconds = %i, %i\n",dataDouble.epoch.gpsSeconds,dataDouble.epoch.gpsNanoSeconds);
      for (i=0;i<NUMTOPRINT;i++) {
        fprintf(stdout,"%i %23.16e\n",i,dataDouble.data->data[i]);
      }
      for (i=dataLength/2;i<dataLength/2+NUMTOPRINT;i++) {
        fprintf(stdout,"%i %23.16e\n",i,dataDouble.data->data[i]);
      }
      for (i=dataLength-NUMTOPRINT;i<dataLength;i++) {
        fprintf(stdout,"%i %23.16e\n",i,dataDouble.data->data[i]);
      }
      fflush(stdout);
}

void printExampleFFTData(struct CommandLineArgsTag CLA)
{
  int firstbin=(INT4)(FMIN*CLA.T+0.5);
  int k;
  INT4 nsamples=(INT4)(DF*CLA.T+0.5);

  if(CLA.useSingle) {
    for (k=0; k<NUMTOPRINT; k++) {
      fprintf(stdout,"%i %23.16e %23.16e\n",k,fftDataSingle->data[k+firstbin].re,fftDataSingle->data[k+firstbin].im);
    }
    for (k=nsamples/2; k<nsamples/2+NUMTOPRINT; k++) {
      fprintf(stdout,"%i %23.16e %23.16e\n",k,fftDataSingle->data[k+firstbin].re,fftDataSingle->data[k+firstbin].im);
    }
    for (k=nsamples-NUMTOPRINT; k<nsamples; k++) {
      fprintf(stdout,"%i %23.16e %23.16e\n",k,fftDataSingle->data[k+firstbin].re,fftDataSingle->data[k+firstbin].im);
    }    
  } else {
    for (k=0; k<NUMTOPRINT; k++) {
      fprintf(stdout,"%i %23.16e %23.16e\n",k,fftDataDouble->data[k+firstbin].re,fftDataDouble->data[k+firstbin].im);
    }
    for (k=nsamples/2; k<nsamples/2+NUMTOPRINT; k++) {
      fprintf(stdout,"%i %23.16e %23.16e\n",k,fftDataDouble->data[k+firstbin].re,fftDataDouble->data[k+firstbin].im);
    }
    for (k=nsamples-NUMTOPRINT; k<nsamples; k++) {
      fprintf(stdout,"%i %23.16e %23.16e\n",k,fftDataDouble->data[k+firstbin].re,fftDataDouble->data[k+firstbin].im);
    }    
  }
  fflush(stdout);
}

void printExampleSFTDataGoingToFile(struct CommandLineArgsTag CLA)
{
  int firstbin=(INT4)(FMIN*CLA.T+0.5);
  int k;
  INT4 nsamples=(INT4)(DF*CLA.T+0.5);
  REAL4 rpw,ipw;

  if(CLA.useSingle) {
    for (k=0; k<NUMTOPRINT; k++) {
      rpw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].re;
      ipw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].im;
      fprintf(stdout,"%i %23.16e %23.16e\n",k,rpw,ipw);
    }
    for (k=nsamples/2; k<nsamples/2+NUMTOPRINT; k++) {
      rpw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].re;
      ipw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].im;
      fprintf(stdout,"%i %23.16e %23.16e\n",k,rpw,ipw);
    }
    for (k=nsamples-NUMTOPRINT; k<nsamples; k++) {
      rpw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].re;
      ipw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].im;
      fprintf(stdout,"%i %23.16e %23.16e\n",k,rpw,ipw);
    }    
  } else {
    for (k=0; k<NUMTOPRINT; k++) {
      rpw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].re;
      ipw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].im;
      fprintf(stdout,"%i %23.16e %23.16e\n",k,rpw,ipw);
    }
    for (k=nsamples/2; k<nsamples/2+NUMTOPRINT; k++) {
      rpw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].re;
      ipw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].im;
      fprintf(stdout,"%i %23.16e %23.16e\n",k,rpw,ipw);
    }
    for (k=nsamples-NUMTOPRINT; k<nsamples; k++) {
      rpw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].re;
      ipw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].im;
      fprintf(stdout,"%i %23.16e %23.16e\n",k,rpw,ipw);
    }    
  }
  fflush(stdout);
}

void printExampleVersion2SFTDataGoingToFile(struct CommandLineArgsTag CLA, SFTtype *oneSFT)
{
  int k;
  INT4 nsamples=(INT4)(DF*CLA.T+0.5);

  for (k=0; k<NUMTOPRINT; k++) {  
    fprintf(stdout,"%i %23.16e %23.16e\n",k,oneSFT->data->data[k].re,oneSFT->data->data[k].im);
  }
  for (k=nsamples/2; k<nsamples/2+NUMTOPRINT; k++) {
    fprintf(stdout,"%i %23.16e %23.16e\n",k,oneSFT->data->data[k].re,oneSFT->data->data[k].im);
  }
  for (k=nsamples-NUMTOPRINT; k<nsamples; k++) {
    fprintf(stdout,"%i %23.16e %23.16e\n",k,oneSFT->data->data[k].re,oneSFT->data->data[k].im);
  }    
}
#endif

/************************************* MAIN PROGRAM *************************************/

int main(int argc,char *argv[])
{
  /* int j; */ /* 12/28/05 gam */

  #if TRACKMEMUSE
    printf("Memory use at startup is:\n"); printmemuse();
  #endif

  if (ReadCommandLine(argc,argv,&CommandLineArgs)) return 1;
  SegmentDuration = CommandLineArgs.GPSEnd - CommandLineArgs.GPSStart ;

  /* create Frame cache, open frame stream and delete frame cache */
  LALFrCacheImport(&status,&framecache,CommandLineArgs.FrCacheFile);
  TESTSTATUS( &status );
  LALFrCacheOpen(&status,&framestream,framecache);
  TESTSTATUS( &status );
  LALDestroyFrCache(&status,&framecache);
  TESTSTATUS( &status );

  #if TRACKMEMUSE
    printf("Memory use after reading command line arguments and reading frame cache:\n"); printmemuse();
  #endif

  if( SegmentDuration < CommandLineArgs.T)
    {
      fprintf(stderr,"Cannot fit an SFT of duration %d between %d and %d\n",
	      CommandLineArgs.T, CommandLineArgs.GPSStart, CommandLineArgs.GPSEnd );
      return 0;;
    }

  gpsepoch.gpsSeconds = CommandLineArgs.GPSStart;
  gpsepoch.gpsNanoSeconds = 0;

  /* Allocates space for data */
  if (AllocateData(CommandLineArgs)) return 2;

  /* for(j=0; j < SegmentDuration/CommandLineArgs.T; j++) */ /* 12/28/05 gam */
  while(gpsepoch.gpsSeconds + CommandLineArgs.T <= CommandLineArgs.GPSEnd)
    {

      /* gpsepoch.gpsSeconds = CommandLineArgs.GPSStart + j*CommandLineArgs.T;
      gpsepoch.gpsNanoSeconds = 0; */ /* 12/28/05 gam; now updated at the bottom of while loop */

      /* Reads T seconds of data */
      if (ReadData(CommandLineArgs)) return 3;

      /* High-pass data with Butterworth filter */
      if(HighPass(CommandLineArgs)) return 4;

      /* Window data; 12/28/05 gam; add options */
      if (CommandLineArgs.windowOption==1) {
         if(WindowData(CommandLineArgs)) return 5; /* CommandLineArgs.windowOption==1 is the default */
      } else if (CommandLineArgs.windowOption==2) {
         if(WindowDataTukey2(CommandLineArgs)) return 5;
      } else if (CommandLineArgs.windowOption==3) {
         if(WindowDataHann(CommandLineArgs)) return 5;
      } else {
        /* Continue with no windowing; parsing of command line args makes sure options are one of the above or 0 for now windowing. */
      }

      /* create an SFT */
      if(CreateSFT(CommandLineArgs)) return 6;

      /* write out sft */
      if (CommandLineArgs.sftVersion==2) {
        if(WriteVersion2SFT(CommandLineArgs)) return 7;
      } else {
        if(WriteSFT(CommandLineArgs)) return 7;  /* default is to output version 1 SFTs */
      }

      gpsepoch.gpsSeconds = gpsepoch.gpsSeconds + (INT4)((1.0 - CommandLineArgs.overlapFraction)*((REAL8)CommandLineArgs.T));
      gpsepoch.gpsNanoSeconds = 0;
    }

  if(FreeMem(CommandLineArgs)) return 8;

  #if TRACKMEMUSE
        printf("Memory use after freeing all allocated memory:\n"); printmemuse();
  #endif

  return 0;
}

/************************************* MAIN PROGRAM ENDS *************************************/

/*** FUNCTIONS ***/

/*******************************************************************************/
int ReadCommandLine(int argc,char *argv[],struct CommandLineArgsTag *CLA)
{
  INT4 errflg=0;
  struct option long_options[] = {
    {"high-pass-freq",       required_argument, NULL,          'f'},
    {"sft-duration",         required_argument, NULL,          't'},
    {"sft-write-path",       required_argument, NULL,          'p'},
    {"frame-cache",          required_argument, NULL,          'C'},
    {"channel-name",         required_argument, NULL,          'N'},
    {"gps-start-time",       required_argument, NULL,          's'},
    {"gps-end-time",         required_argument, NULL,          'e'},
    {"sft-version",          optional_argument, NULL,          'v'},
    {"comment-field",        optional_argument, NULL,          'c'},
    {"start-freq",           optional_argument, NULL,          'F'},
    {"band",                 optional_argument, NULL,          'B'},
    {"make-gps-dirs",        optional_argument, NULL,          'D'},
    {"make-tmp-file",        optional_argument, NULL,          'Z'},
    {"misc-desc",            optional_argument, NULL,          'X'},
    {"window-type",          optional_argument, NULL,          'w'},
    {"overlap-fraction",     optional_argument, NULL,          'P'},
    {"ht-data",              no_argument,       NULL,          'H'},
    {"use-single",           no_argument,       NULL,          'S'},
    {"help",                 no_argument,       NULL,          'h'},
    {0, 0, 0, 0}
  };
  char args[] = "hHZSf:t:C:N:s:e:v:c:F:B:D:X:w:P:p:";

  /* Initialize default values */
  CLA->HPf=-1.0;
  CLA->T=0.0;
  CLA->stringT=NULL;  /* 12/27/05 gam */
  CLA->FrCacheFile=NULL;
  CLA->GPSStart=0;
  CLA->GPSEnd=0;
  CLA->makeGPSDirs=0; /* 12/27/05 gam; add option to make directories based on gps time */
  CLA->sftVersion=1;  /* 12/28/05 gam; output SFT version; default is version 1 SFTs */
  CLA->ChannelName=NULL;
  CLA->SFTpath=NULL;
  CLA->miscDesc=NULL;   /* 12/28/05 gam; misc. part of the SFT description field in the filename (also used if makeGPSDirs > 0) */
  CLA->commentField=NULL; /* 12/28/05 gam; comment for version 2 SFT header. */
  CLA->windowOption=1;  /* 12/28/05 gam; window options; 0 = no window, 1 = default = Matlab style Tukey window; 2 = make_sfts.c Tukey window; 3 = Hann window */
  CLA->overlapFraction=0.0; /* 12/28/05 gam; overlap fraction (for use with windows; e.g., use -P 0.5 with -w 3 Hann windows; default is 0.0). */
  CLA->htdata = 0;
  CLA->makeTmpFile = 0; /* 01/09/06 gam */  
  CLA->useSingle = 0; /* 11/19/05 gam; default is to use double precision, not single. */

  /* Scan through list of command line arguments */
  while ( 1 )
  {
    int option_index = 0; /* getopt_long stores long option here */
    int c;

    c = getopt_long_only( argc, argv, args, long_options, &option_index );
    if ( c == -1 ) /* end of options */
      break;

    switch ( c )
    {
    case 'H':
      /* high pass frequency */
      CLA->htdata=1;
      break;
    case 'Z':
      /* make tmp file */
      CLA->makeTmpFile=1;
      break;
    case 'S':
      /* use single precision */
      CLA->useSingle=1;
      break;
    case 'f':
      /* high pass frequency */
      CLA->HPf=atof(optarg);
      break;
    case 't':
      /* SFT time */
      CLA->stringT = optarg;  /* 12/27/05 gam; keep pointer to string that gives the SFT duration */
      CLA->T=atoi(optarg);
      break;
    case 'C':
      /* name of frame cache file */
      CLA->FrCacheFile=optarg;
      break;
    case 's':
      /* GPS start */
      CLA->GPSStart=atof(optarg);
      break;
    case 'e':
      /* GPS end */
      CLA->GPSEnd=atof(optarg);
      break;
    case 'F':
      /* 12/28/05 gam; start frequency */
      FMIN=(REAL8)atof(optarg);
      break;
    case 'B':
      /* 12/28/05 gam; band */
      DF=(REAL8)atof(optarg);
      break;
    case 'D':
      /* 12/27/05 gam; make directories based on GPS time */
      CLA->makeGPSDirs=atof(optarg);
      break;
    case 'v':
      /* 12/28/05 gam; output SFT version; default is version 1 SFTs */
      CLA->sftVersion=atoi(optarg);
      break;
    case 'c':
      /* 12/28/05 gam; comment for version 2 SFTs */
      CLA->commentField=optarg;
      break;
    case 'X':
      /* 12/28/05 gam; misc. part of the SFT description field in the filename (also used if makeGPSDirs > 0) */
      CLA->miscDesc=optarg;
      break;
    case 'w':
      /* 12/28/05 gam; window options; 0 = no window, 1 = default = Matlab style Tukey window; 2 = make_sfts.c Tukey window; 3 = Hann window */
      CLA->windowOption=atoi(optarg);
      break;
    case 'P':
      /* 12/28/05 gam; overlap fraction (for use with windows; e.g., use -P 0.5 with -w 3 Hann windows; default is 1.0). */
      CLA->overlapFraction=(REAL8)atof(optarg);
      break;
    case 'N':
      CLA->ChannelName=optarg;       
      break;
    case 'p':
      CLA->SFTpath=optarg;       
      break;
    case 'h':
      /* print usage/help message */
      fprintf(stdout,"Arguments are:\n");
      fprintf(stdout,"\thigh-pass-freq (-f)\tFLOAT\t High pass filtering frequency in Hz.\n");
      fprintf(stdout,"\tsft-duration (-t)\tFLOAT\t SFT duration in seconds.\n");
      fprintf(stdout,"\tsft-write-path (-p)\tFLOAT\t Location of output SFTs.\n");
      fprintf(stdout,"\tframe-cache (-C)\tSTRING\t Path to frame cache file (including the filename).\n");
      fprintf(stdout,"\tgps-start-time (-s)\tINT\t GPS start time of segment.\n");
      fprintf(stdout,"\tgps-end-time (-e)\tINT\t GPS end time of segment.\n");
      fprintf(stdout,"\tchannel-name (-N)\tSTRING\t Name of channel to read within a frame.\n");
      fprintf(stdout,"\tsft-version (-v)\tINT\t (optional) SFT version (1 = default = output version 1 SFTs; 2 = output version 2 SFTs.\n");
      fprintf(stdout,"\tcomment-field (-c)\tSTRING\t (optional) comment for version 2 SFT header.\n");
      fprintf(stdout,"\tstart-freq (-F) \tFLOAT\t (optional) start frequency of the SFTs (default is 48 Hz).\n");
      fprintf(stdout,"\tband (-B)       \tFLOAT\t (optional) frequency band of the SFTs (default is 2000 Hz).\n");
      fprintf(stdout,"\tmake-gps-dirs (-D)\tINT\t (optional) make directories for output SFTs based on this many digits of the GPS time.\n");
      fprintf(stdout,"\tmake-tmp-file (-Z)\tINT\t (optional) write SFT to .*.tmp file, then move to final filename.\n");
      fprintf(stdout,"\tmisc-desc (-X)   \tSTRING\t (optional) misc. part of the SFT description field in the filename (also used if make-gps-dirs, -D option, is > 0)\n");
      fprintf(stdout,"\twindow-type (-w)\tINT\t (optional) 0 = no window, 1 = default = Matlab style Tukey window; 2 = make_sfts.c Tukey window; 3 = Hann window\n");
      fprintf(stdout,"\toverlap-fraction (-P)\tFLOAT\t (optional) overlap fraction (for use with windows; e.g., use -P 0.5 with -w 3 Hann windows; default is 0.0).\n");
      fprintf(stdout,"\tht-data (-H)\t\tFLAG\t This is h(t) data (use Real8's; Procdata ).\n");
      fprintf(stdout,"\tuse-single (-S)\t\tFLAG\t Use single precision for window, plan, and fft; double precision filtering is always done.\n");
      fprintf(stdout,"\thelp (-h)\t\tFLAG\t This message\n");    
      exit(0);
      break;
    default:
      /* unrecognized option */
      errflg++;
      fprintf(stderr,"Unrecognized option argument %c\n",c);
      exit(1);
      break;
    }
  }

  if(CLA->HPf < 0 )
    {
      fprintf(stderr,"No high pass filtering frequency specified.\n"); 
      fprintf(stderr,"If you don't want to high pass filter set the frequency to 0.\n");
      fprintf(stderr,"Try %s -h \n",argv[0]);
      return 1;
    }      
  if(CLA->T == 0.0)
    {
      fprintf(stderr,"No SFT duration specified.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }      
  if(CLA->GPSStart == 0)
    {
      fprintf(stderr,"No GPS start time specified.\n");
       fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if(CLA->GPSEnd == 0)
    {
      fprintf(stderr,"No GPS end time specified.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if(FMIN < 0.0)
    {
      fprintf(stderr,"Illegal start-freq option given.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if(DF < 0.0)
    {
      fprintf(stderr,"Illegal band option given.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if(CLA->makeGPSDirs < 0)
    {
      fprintf(stderr,"Illegal make-gps-dirs option given.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if( (CLA->sftVersion < 1) || (CLA->sftVersion > 2) )
    {
      fprintf(stderr,"Illegal sft-version given.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if( (CLA->windowOption < 0) || (CLA->windowOption > 3) )
    {
      fprintf(stderr,"Illegal window-type given.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if( (CLA->overlapFraction < 0.0) || (CLA->overlapFraction >= 1.0) )
    {
      fprintf(stderr,"Illegal overlap-fraction given.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }
  if(CLA->FrCacheFile == NULL)
    {
      fprintf(stderr,"No frame cache file specified.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }      
  if(CLA->ChannelName == NULL)
    {
      fprintf(stderr,"No data channel name specified.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }      
  if(CLA->SFTpath == NULL)
    {
      fprintf(stderr,"No output path specified for SFTs.\n");
      fprintf(stderr,"Try %s -h \n", argv[0]);
      return 1;
    }      

  return errflg;
}
/*******************************************************************************/

/*******************************************************************************/
int AllocateData(struct CommandLineArgsTag CLA)
{
  static FrChanIn chanin;

  chanin.name  = CLA.ChannelName;

  /* These calls just return deltaT for the channel */
  if(CLA.htdata)
    {
      chanin.type  = ProcDataChannel;
      /* Get channel time step size by calling LALFrGetREAL8TimeSeries */
      LALFrSeek(&status,&gpsepoch,framestream);
      TESTSTATUS( &status );
      LALFrGetREAL8TimeSeries(&status,&dataDouble,&chanin,framestream);
      TESTSTATUS( &status );
      dataSingle.deltaT=dataDouble.deltaT;      
    }
  else
    {
      chanin.type  = ADCDataChannel;
      /* Get channel time step size by calling LALFrGetREAL4TimeSeries */
      LALFrSeek(&status,&gpsepoch,framestream);
      TESTSTATUS( &status );
      LALFrGetREAL4TimeSeries(&status,&dataSingle,&chanin,framestream);
      TESTSTATUS( &status );
      dataDouble.deltaT=dataSingle.deltaT;
    }

  /*  11/19/05 gam; will keep dataDouble or dataSingle in memory at all times, depending on whether using single or double precision */
  if(CLA.useSingle) {
      LALCreateVector(&status,&dataSingle.data,(UINT4)(CLA.T/dataSingle.deltaT +0.5));
      TESTSTATUS( &status );

      #if TRACKMEMUSE
         printf("Memory use after creating dataSingle and before calling LALCreateForwardRealFFTPlan.\n"); printmemuse();
      #endif

      LALCreateForwardRealFFTPlan( &status, &fftPlanSingle, dataSingle.data->length, 0 );
      TESTSTATUS( &status );

      #if TRACKMEMUSE
         printf("Memory use after creating dataSingle and after calling LALCreateForwardRealFFTPlan.\n"); printmemuse();
      #endif

  } else {  
      LALDCreateVector(&status,&dataDouble.data,(UINT4)(CLA.T/dataDouble.deltaT +0.5));
      TESTSTATUS( &status );

      #if TRACKMEMUSE
         printf("Memory use after creating dataDouble and before calling LALCreateForwardREAL8FFTPlan.\n"); printmemuse();
      #endif
      
      LALCreateForwardREAL8FFTPlan( &status, &fftPlanDouble, dataDouble.data->length, 0 );
      TESTSTATUS( &status );

      #if TRACKMEMUSE
            printf("Memory use after creating dataDouble and after calling LALCreateForwardREAL8FFTPlan.\n"); printmemuse();
      #endif
  }

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
int ReadData(struct CommandLineArgsTag CLA)
{
  static FrChanIn chanin;
  INT4 k;
  chanin.name  = CLA.ChannelName;
  LALFrSeek(&status,&gpsepoch,framestream);
  TESTSTATUS( &status );

  /* 11/19/05 gam */
  if(CLA.useSingle) {
    
    if(CLA.htdata)
    {

      #if TRACKMEMUSE
        printf("Memory use before creating dataDouble and calling LALFrGetREAL8TimeSeries.\n"); printmemuse();
      #endif

      LALDCreateVector(&status,&dataDouble.data,(UINT4)(CLA.T/dataDouble.deltaT +0.5));
      TESTSTATUS( &status );

      chanin.type  = ProcDataChannel;
      LALFrGetREAL8TimeSeries(&status,&dataDouble,&chanin,framestream);
      TESTSTATUS( &status );

      #if TRACKMEMUSE
        printf("Memory use after creating dataDouble and calling LALFrGetREAL8TimeSeries.\n"); printmemuse();
      #endif

      /*copy the data into the single precision timeseries */
      for (k = 0; k < dataSingle.data->length; k++) {
          dataSingle.data->data[k] = dataDouble.data->data[k];
      }

      LALDDestroyVector(&status,&dataDouble.data);
      TESTSTATUS( &status );

      #if TRACKMEMUSE
        printf("Memory use after destroying dataDouble.\n"); printmemuse();
      #endif

    }
    else
    {

      #if TRACKMEMUSE
        printf("Memory use before calling LALFrGetREAL4TimeSeries.\n"); printmemuse();
      #endif

      chanin.type  = ADCDataChannel;
      LALFrGetREAL4TimeSeries(&status,&dataSingle,&chanin,framestream);
      TESTSTATUS( &status );

      #if TRACKMEMUSE
        printf("Memory use after calling LALFrGetREAL4TimeSeries.\n"); printmemuse();
      #endif

    }

    #if PRINTEXAMPLEDATA
        printf("\nExample dataSingle values after reading data from frames in ReadData:\n"); printExampleDataSingle();
    #endif

  } else {

    if(CLA.htdata)
    {
    
      #if TRACKMEMUSE
        printf("Memory use before calling LALFrGetREAL8TimeSeries.\n"); printmemuse();
      #endif
    
      chanin.type  = ProcDataChannel;
      LALFrGetREAL8TimeSeries(&status,&dataDouble,&chanin,framestream);
      TESTSTATUS( &status );

      #if TRACKMEMUSE
        printf("Memory use after calling LALFrGetREAL8TimeSeries.\n"); printmemuse();
      #endif

    }
    else
    {
      #if TRACKMEMUSE
        printf("Memory use before creating dataSingle and calling LALFrGetREAL4TimeSeries.\n"); printmemuse();
      #endif

      /* 11/02/05 gam; add next two lines */
      LALCreateVector(&status,&dataSingle.data,(UINT4)(CLA.T/dataSingle.deltaT +0.5));
      TESTSTATUS( &status );
      
      chanin.type  = ADCDataChannel;
      LALFrGetREAL4TimeSeries(&status,&dataSingle,&chanin,framestream);
      TESTSTATUS( &status );
 
      #if TRACKMEMUSE
        printf("Memory use after creating dataSingle and calling LALFrGetREAL4TimeSeries.\n"); printmemuse();
      #endif

      /*copy the data into the double precision timeseries */
      for (k = 0; k < (int)dataDouble.data->length; k++) {
          dataDouble.data->data[k] = dataSingle.data->data[k];
      }     

      LALDestroyVector(&status,&dataSingle.data);
      TESTSTATUS( &status );

      #if TRACKMEMUSE
        printf("Memory use after destroying dataSingle.\n"); printmemuse();
      #endif

    }

    #if PRINTEXAMPLEDATA
        printf("\nExample dataDouble values after reading data from frames in ReadData:\n"); printExampleDataDouble();
    #endif

  } /* END if(CLA.useSingle) else */

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
int HighPass(struct CommandLineArgsTag CLA)
{
  PassBandParamStruc filterpar;

  filterpar.name  = "Butterworth High Pass";
  filterpar.nMax  = 10;
  filterpar.f2    = CLA.HPf;
  filterpar.a2    = 0.5;
  filterpar.f1    = -1.0;
  filterpar.a1    = -1.0;

  if (CLA.HPf > 0.0 )
    {
      if(CLA.useSingle) {
        #if TRACKMEMUSE
          printf("Memory use before calling LALDButterworthREAL4TimeSeries:\n"); printmemuse();
        #endif

        /* High pass the time series */
        LALDButterworthREAL4TimeSeries(&status,&dataSingle,&filterpar);
        TESTSTATUS( &status );

        #if TRACKMEMUSE
          printf("Memory use after calling LALDButterworthREAL4TimeSeries:\n"); printmemuse();
        #endif

        #if PRINTEXAMPLEDATA
            printf("\nExample dataSingle values after filtering data in HighPass:\n"); printExampleDataSingle();
        #endif
      } else {
        #if TRACKMEMUSE
          printf("Memory use before calling LALButterworthREAL8TimeSeries:\n"); printmemuse();
        #endif

        /* High pass the time series */
        LALButterworthREAL8TimeSeries(&status,&dataDouble,&filterpar);
        TESTSTATUS( &status );

        #if TRACKMEMUSE
          printf("Memory use after calling LALButterworthREAL8TimeSeries:\n"); printmemuse();
        #endif

        #if PRINTEXAMPLEDATA
            printf("\nExample dataDouble values after filtering data in HighPass:\n"); printExampleDataDouble();
        #endif
      }
    }

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
int WindowData(struct CommandLineArgsTag CLA)
{
  REAL8 r = 0.001;
  INT4 k, N, kl, kh;
  /* This implementation of a Tukey window is describes 
     in the Matlab tukey window documentation */

  /* 11/19/05 gam */
  if(CLA.useSingle) {
    N=dataSingle.data->length;
    kl=r/2*(N-1)+1;
    kh=N-r/2*(N-1)+1;
    for(k = 1; k < kl; k++) 
    {
      dataSingle.data->data[k-1]=dataSingle.data->data[k-1]*0.5*( (REAL4)( 1.0 + cos(LAL_TWOPI/r*(k-1)/(N-1) - LAL_PI) ) );
    }
    for(k = kh; k <= N; k++) 
    {
      dataSingle.data->data[k-1]=dataSingle.data->data[k-1]*0.5*( (REAL4)( 1.0 + cos(LAL_TWOPI/r - LAL_TWOPI/r*(k-1)/(N-1) - LAL_PI) ) );
    }

    #if PRINTEXAMPLEDATA
        printf("\nExample dataSingle values after windowing data in WindowData:\n"); printExampleDataSingle();
    #endif
  } else {
    N=dataDouble.data->length;
    kl=r/2*(N-1)+1;
    kh=N-r/2*(N-1)+1;
    for(k = 1; k < kl; k++) 
    {
      dataDouble.data->data[k-1]=dataDouble.data->data[k-1]*0.5*( 1.0 + cos(LAL_TWOPI/r*(k-1)/(N-1) - LAL_PI) );
    }
    for(k = kh; k <= N; k++) 
    {
      dataDouble.data->data[k-1]=dataDouble.data->data[k-1]*0.5*( 1.0 + cos(LAL_TWOPI/r - LAL_TWOPI/r*(k-1)/(N-1) - LAL_PI) );
    }
    #if PRINTEXAMPLEDATA
        printf("\nExample dataDouble values after windowing data in WindowData:\n"); printExampleDataDouble();
    #endif    
  }

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
/* Same as window function given in lalapps/src/pulsar/make_sfts.c */
int WindowDataTukey2(struct CommandLineArgsTag CLA)
{  
  /* Define the parameters to make the window */
  INT4 WINSTART = 4096;
  INT4 WINEND = 8192;
  INT4 WINLEN = (WINEND-WINSTART);
  INT4 i;
  REAL8 win;
    
  if(CLA.useSingle) {
        /* window data.  Off portion */
        for (i=0; i<WINSTART; i++) {
          dataSingle.data->data[i] = 0.0;
          dataSingle.data->data[dataSingle.data->length - 1 - i] = 0.0;
        }
        /* window data, smooth turn-on portion */
        for (i=WINSTART; i<WINEND; i++) {
          win=((sin((i - WINSTART)*LAL_PI/(WINLEN)-LAL_PI_2)+1.0)/2.0);
          dataSingle.data->data[i] *= win;
          dataSingle.data->data[dataSingle.data->length - 1 - i]  *= win;
        }
        #if PRINTEXAMPLEDATA
           printf("\nExample dataSingle values after windowing data in WindowDataTukey2:\n"); printExampleDataSingle();
        #endif
  } else {
        /* window data.  Off portion */
        for (i=0; i<WINSTART; i++) {
          dataDouble.data->data[i] = 0.0;
          dataDouble.data->data[dataDouble.data->length - 1 - i] = 0.0;
        }
        /* window data, smooth turn-on portion */
        for (i=WINSTART; i<WINEND; i++) {
          win=((sin((i - WINSTART)*LAL_PI/(WINLEN)-LAL_PI_2)+1.0)/2.0);
          dataDouble.data->data[i] *= win;
          dataDouble.data->data[dataDouble.data->length - 1 - i]  *= win;
        }
        #if PRINTEXAMPLEDATA
            printf("\nExample dataDouble values after windowing data in WindowDataTukey2:\n"); printExampleDataDouble();
        #endif  
  }

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
/* Hann window based on Matlab, but with C indexing: w[k] = 0.5*( 1 - cos(2*pi*k/(N-1)) ) k = 0, 1, 2,...N-1 */
int WindowDataHann(struct CommandLineArgsTag CLA)
{
  INT4 k;
  REAL8 win,N,Nm1;
  REAL8 real8TwoPi = 2.0*((REAL8)(LAL_PI));

  if(CLA.useSingle) {
        N = ((REAL8)dataSingle.data->length);
        Nm1 = N - 1;
        for (k=0; k<N; k++) {
          win=0.5*( 1.0 - cos(real8TwoPi*((REAL8)(k))/Nm1) );
          dataSingle.data->data[k] *= win;
        }
        #if PRINTEXAMPLEDATA
           printf("\nExample dataSingle values after windowing data in WindowDataHann:\n"); printExampleDataSingle();
        #endif
  } else {
        N = ((REAL8)dataDouble.data->length);
        Nm1 = N - 1;
        for (k=0; k<N; k++) {
          win=0.5*( 1.0 - cos(real8TwoPi*((REAL8)(k))/Nm1) );
          dataDouble.data->data[k] *= win;
        }  
        #if PRINTEXAMPLEDATA
            printf("\nExample dataDouble values after windowing data in WindowDataHann:\n"); printExampleDataDouble();
        #endif  
  }

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
int CreateSFT(struct CommandLineArgsTag CLA)
{

  /* 11/19/05 gam */
  if(CLA.useSingle) {
      #if TRACKMEMUSE
        printf("Memory use before creating output vector fftDataSingle and calling LALForwardRealFFT:\n"); printmemuse();
      #endif

      /* 11/02/05 gam; allocate container for SFT data */
      LALCCreateVector( &status, &fftDataSingle, dataSingle.data->length / 2 + 1 );
      TESTSTATUS( &status );  

      /* compute sft */
      LALForwardRealFFT(&status, fftDataSingle, dataSingle.data, fftPlanSingle );
      TESTSTATUS( &status );      

      #if TRACKMEMUSE
        printf("Memory use after creating output vector fftDataSingle and calling LALForwardRealFFT:\n"); printmemuse();
      #endif

      #if PRINTEXAMPLEDATA
        printf("\nExample real and imaginary value of fftDataSingle from CreateSFT:\n"); printExampleFFTData(CLA);
      #endif      
  } else {
      #if TRACKMEMUSE
        printf("Memory use before creating output vector fftDataDouble and calling XLALREAL8ForwardFFT:\n"); printmemuse();
      #endif

      /* 11/02/05 gam; allocate container for SFT data */
      LALZCreateVector( &status, &fftDataDouble, dataDouble.data->length / 2 + 1 );
      TESTSTATUS( &status );  

      /* compute sft */
      XLALREAL8ForwardFFT( fftDataDouble, dataDouble.data, fftPlanDouble );

      #if TRACKMEMUSE
        printf("Memory use after creating output vector fftDataDouble and calling XLALREAL8ForwardFFT:\n"); printmemuse();
      #endif

      #if PRINTEXAMPLEDATA
        printf("\nExample real and imaginary value of fftDataDouble from CreateSFT:\n"); printExampleFFTData(CLA);
      #endif      
  }
      return 0;
}
/*******************************************************************************/

/*******************************************************************************/
int WriteSFT(struct CommandLineArgsTag CLA)
{
  char sftname[256];
  char sftnameFinal[256]; /* 01/09/06 gam */
  char numSFTs[2]; /* 12/27/05 gam */
  char site[2];    /* 12/27/05 gam */
  char ifo[3];     /* 12/27/05 gam; allow 3rd charactor for null termination */
  int firstbin=(INT4)(FMIN*CLA.T+0.5), k;
  FILE *fpsft;
  int errorcode1=0,errorcode2=0;
  char gpstime[11]; /* 12/27/05 gam; allow for 10 digit GPS times and null termination */
  REAL4 rpw,ipw;

  /* 12/27/05 gam; set up the number of SFTs, site, and ifo as null terminated strings */
  numSFTs[0] = '1';
  numSFTs[1] = '\0'; /* null terminate */
  strncpy( site, CLA.ChannelName, 1 );
  site[1] = '\0'; /* null terminate */
  strncpy( ifo, CLA.ChannelName, 2 );
  ifo[2] = '\0'; /* null terminate */
  sprintf(gpstime,"%09d",gpsepoch.gpsSeconds);

  strcpy( sftname, CLA.SFTpath );
  /* 12/27/05 gam; add option to make directories based on gps time */
  if (CLA.makeGPSDirs > 0) {
     /* 12/27/05 gam; concat to the sftname the directory name based on GPS time; make this directory if it does not already exist */
     mkSFTDir(sftname, site, numSFTs, ifo, CLA.stringT, CLA.miscDesc, gpstime, CLA.makeGPSDirs);
  }

  /* 01/09/06 gam; sftname will be temporary; will move to sftnameFinal. */
  if(CLA.makeTmpFile) {
    /* set up sftnameFinal with usual SFT name */
    strcpy(sftnameFinal,sftname);
    strcat(sftnameFinal,"/SFT_");
    strncat(sftnameFinal,ifo,2);
    strcat(sftnameFinal,".");
    strcat(sftnameFinal,gpstime);
    /* sftname begins with . and ends in .tmp */
    strcat(sftname,"/.SFT_");
    strncat(sftname,ifo,2);
    strcat(sftname,".");
    strcat(sftname,gpstime);
    strcat(sftname,".tmp");
  } else {     
    strcat(sftname,"/SFT_");
    strncat(sftname,ifo,2);
    strcat(sftname,".");
    strcat(sftname,gpstime);
  }  

  /* open SFT file for writing */
  fpsft=tryopen(sftname,"w");

  /* write header */
  header.endian=1.0;
  header.gps_sec=gpsepoch.gpsSeconds;
  header.gps_nsec=gpsepoch.gpsNanoSeconds;
  header.tbase=CLA.T;
  header.firstfreqindex=firstbin;
  header.nsamples=(INT4)(DF*CLA.T+0.5);
  if (1!=fwrite((void*)&header,sizeof(header),1,fpsft)){
    fprintf(stderr,"Error in writing header into file %s!\n",sftname);
    return 7;
  }

  /* 11/19/05 gam */
  if(CLA.useSingle) {
    /* Write SFT */
    for (k=0; k<header.nsamples; k++)
    {
      rpw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].re;
      ipw=((REAL4)(((REAL8)DF)/(0.5*(REAL8)(1/dataSingle.deltaT))))
	* fftDataSingle->data[k+firstbin].im;
      errorcode1=fwrite((void*)&rpw, sizeof(REAL4),1,fpsft);
      errorcode2=fwrite((void*)&ipw, sizeof(REAL4),1,fpsft);
    }
    #if PRINTEXAMPLEDATA
        printf("\nExample real and imaginary SFT values going to file from fftDataSingle in WriteSFT:\n"); printExampleSFTDataGoingToFile(CLA);
    #endif     
    LALCDestroyVector( &status, &fftDataSingle );
    TESTSTATUS( &status );
    #if TRACKMEMUSE
      printf("Memory use after destroying fftDataSingle in WriteSFT:\n"); printmemuse();
    #endif
  } else {    
    /* Write SFT */
    for (k=0; k<header.nsamples; k++)
    {
      rpw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].re;
      ipw=(((REAL8)DF)/(0.5*(REAL8)(1/dataDouble.deltaT))) 
	* fftDataDouble->data[k+firstbin].im;
      errorcode1=fwrite((void*)&rpw, sizeof(REAL4),1,fpsft);
      errorcode2=fwrite((void*)&ipw, sizeof(REAL4),1,fpsft);
    }
    #if PRINTEXAMPLEDATA
        printf("\nExample real and imaginary SFT values going to file from fftDataDouble in WriteSFT:\n"); printExampleSFTDataGoingToFile(CLA);
    #endif
    LALZDestroyVector( &status, &fftDataDouble );
    TESTSTATUS( &status );
    #if TRACKMEMUSE
      printf("Memory use after destroying fftDataDouble in WriteSFT:\n"); printmemuse();
    #endif
  }

  /* Check that there were no errors while writing SFTS */
  if (errorcode1-1 || errorcode2-1)
    {
      fprintf(stderr,"Error in writing data into SFT file %s!\n",sftname);
      return 8;
    }
  
  fclose(fpsft);

  /* 01/09/06 gam; sftname is temporary; move to sftnameFinal. */
  if(CLA.makeTmpFile) {  
    mvFilenames(sftname,sftnameFinal);
  }

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
/* 12/28/05 gam; write out version 2 SFT */
int WriteVersion2SFT(struct CommandLineArgsTag CLA) 
{
  char sftname[256];
  char sftFilename[256];
  char sftnameFinal[256]; /* 01/09/06 gam */
  char numSFTs[2]; /* 12/27/05 gam */
  char site[2];    /* 12/27/05 gam */
  char ifo[3];     /* 12/27/05 gam; allow 3rd charactor for null termination */
  int firstbin=(INT4)(FMIN*CLA.T+0.5), k;
  char gpstime[11]; /* 12/27/05 gam; allow for 10 digit GPS times and null termination */
  SFTtype *oneSFT = NULL;
  INT4 nBins = (INT4)(DF*CLA.T+0.5);
  REAL4 singleDeltaT = 0.0; /* 01/05/06 gam */
  REAL8 doubleDeltaT = 0.0; /* 01/05/06 gam */

  /* 12/27/05 gam; set up the number of SFTs, site, and ifo as null terminated strings */
  numSFTs[0] = '1';
  numSFTs[1] = '\0'; /* null terminate */
  strncpy( site, CLA.ChannelName, 1 );
  site[1] = '\0'; /* null terminate */
  strncpy( ifo, CLA.ChannelName, 2 );
  ifo[2] = '\0'; /* null terminate */
  sprintf(gpstime,"%09d",gpsepoch.gpsSeconds);

  strcpy( sftname, CLA.SFTpath );
  /* 12/27/05 gam; add option to make directories based on gps time */
  if (CLA.makeGPSDirs > 0) {
     /* 12/27/05 gam; concat to the sftname the directory name based on GPS time; make this directory if it does not already exist */
     mkSFTDir(sftname, site, numSFTs, ifo, CLA.stringT, CLA.miscDesc, gpstime, CLA.makeGPSDirs);
  }

  strcat(sftname,"/");
  mkSFTFilename(sftFilename, site, numSFTs, ifo, CLA.stringT, CLA.miscDesc, gpstime);
  /* 01/09/06 gam; sftname will be temporary; will move to sftnameFinal. */
  if(CLA.makeTmpFile) {
    /* set up sftnameFinal with usual SFT name */
    strcpy(sftnameFinal,sftname);
    strcat(sftnameFinal,sftFilename);
    /* sftname begins with . and ends in .tmp */
    strcat(sftname,".");
    strcat(sftname,sftFilename);
    strcat(sftname,".tmp");
  } else {
    strcat(sftname,sftFilename);
  }  

  /* make container to store the SFT data */
  LALCreateSFTtype (&status, &oneSFT, ((UINT4)nBins));
  TESTSTATUS( &status );
  #if TRACKMEMUSE
      printf("Memory use after creating oneSFT and calling LALCreateSFTtype in WriteVersion2SFT:\n"); printmemuse();
  #endif
  
  /* copy the data to oneSFT */
  strcpy(oneSFT->name,ifo);
  oneSFT->epoch.gpsSeconds=gpsepoch.gpsSeconds;
  oneSFT->epoch.gpsNanoSeconds=gpsepoch.gpsNanoSeconds;
  oneSFT->f0 = FMIN;
  oneSFT->deltaF = 1.0/((REAL8)CLA.T);
  oneSFT->data->length=nBins;
  
  if(CLA.useSingle) {
    singleDeltaT = ((REAL4)dataSingle.deltaT); /* 01/05/06 gam; and normalize SFTs using this below */
    for (k=0; k<nBins; k++)
    {
      oneSFT->data->data[k].re = singleDeltaT*fftDataSingle->data[k+firstbin].re;
      oneSFT->data->data[k].im = singleDeltaT*fftDataSingle->data[k+firstbin].im;
    }
    #if PRINTEXAMPLEDATA
        printf("\nExample real and imaginary SFT values going to file from fftDataSingle in WriteVersion2SFT:\n"); printExampleVersion2SFTDataGoingToFile(CLA,oneSFT);
    #endif
    LALCDestroyVector( &status, &fftDataSingle );
    TESTSTATUS( &status );
    #if TRACKMEMUSE
      printf("Memory use after destroying fftDataSingle in WriteVersion2SFT:\n"); printmemuse();
    #endif
  } else {
    doubleDeltaT = ((REAL8)dataDouble.deltaT); /* 01/05/06 gam; and normalize SFTs using this below */
    for (k=0; k<nBins; k++)
    {
      oneSFT->data->data[k].re = doubleDeltaT*fftDataDouble->data[k+firstbin].re;
      oneSFT->data->data[k].im = doubleDeltaT*fftDataDouble->data[k+firstbin].im;
    }
    #if PRINTEXAMPLEDATA
        printf("\nExample real and imaginary SFT values going to file from fftDataDouble in WriteVersion2SFT:\n"); printExampleVersion2SFTDataGoingToFile(CLA,oneSFT);
    #endif
    LALZDestroyVector( &status, &fftDataDouble );
    TESTSTATUS( &status );
    #if TRACKMEMUSE
      printf("Memory use after destroying fftDataDouble in WriteVersion2SFT:\n"); printmemuse();
    #endif
  }  

  /* write the SFT */
  LALWriteSFT2file(&status, oneSFT, sftname, CLA.commentField);
  TESTSTATUS( &status );

  /* 01/09/06 gam; sftname is temporary; move to sftnameFinal. */
  if(CLA.makeTmpFile) {  
    mvFilenames(sftname,sftnameFinal);
  }

  LALDestroySFTtype (&status,&oneSFT);
  TESTSTATUS( &status );
  #if TRACKMEMUSE
      printf("Memory use after destroying oneSFT and calling LALDestroySFTtype in WriteVersion2SFT:\n"); printmemuse();
  #endif

  return 0;
}
/*******************************************************************************/

/*******************************************************************************/
int FreeMem(struct CommandLineArgsTag CLA)
{

  LALFrClose(&status,&framestream);
  TESTSTATUS( &status );

  /* 11/19/05 gam */
  if(CLA.useSingle) {
    LALDestroyVector(&status,&dataSingle.data);
    TESTSTATUS( &status );
    LALDestroyRealFFTPlan( &status, &fftPlanSingle );
    TESTSTATUS( &status );
  } else {
    LALDDestroyVector(&status,&dataDouble.data);
    TESTSTATUS( &status );
    LALDestroyREAL8FFTPlan( &status, &fftPlanDouble );
    TESTSTATUS( &status );    
  }

  LALCheckMemoryLeaks();
 
  return 0;
}
/*******************************************************************************/

#endif
